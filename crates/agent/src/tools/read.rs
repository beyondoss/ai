//! `read` — read a file, optionally a line range, with line numbers.

use std::io::{BufRead, BufReader};

use agent_core::message::ImageSource;
use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{Value, json};

/// Cap on image file size we'll inline as a data-URI (bytes). Beyond this the base64 payload would
/// dominate the model's context; tell the model to narrow rather than ballooning the request.
const MAX_IMAGE_BYTES: u64 = 5 * 1024 * 1024;

/// Default cap on returned lines when no `limit` is given (keeps large files from flooding context).
const DEFAULT_LIMIT: usize = 2000;
/// Cap on bytes kept per line. Streaming by line already bounds memory to one line, but a file with a
/// single pathological line (a minified bundle, a one-line JSON/CSV blob) is one unbounded line — so
/// cap each line and drain the overflow without storing it, the way `grep` clips long match lines.
const MAX_LINE_BYTES: usize = 4000;
/// Aggregate output budget: the line limit alone (2000 × up to 4000 bytes) could return ~8MB of
/// mostly-long lines into context. Stop once the rendered output crosses this, telling the model how
/// to continue.
const MAX_OUTPUT_BYTES: usize = 256_000;

pub struct Read;

/// Read one line into `buf` (without the trailing newline), keeping at most `cap` bytes; bytes beyond
/// `cap` are consumed from the stream but discarded, so a single huge line can't balloon memory.
/// Returns `(bytes_consumed, truncated)`; `bytes_consumed == 0` means EOF.
fn read_line_capped(
    reader: &mut impl BufRead,
    buf: &mut Vec<u8>,
    cap: usize,
) -> std::io::Result<(usize, bool)> {
    buf.clear();
    let mut consumed = 0usize;
    let mut truncated = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break; // EOF
        }
        match available.iter().position(|&b| b == b'\n') {
            Some(pos) => {
                let want = pos; // bytes before the newline
                let take = want.min(cap.saturating_sub(buf.len()));
                buf.extend_from_slice(&available[..take]);
                truncated |= take < want;
                reader.consume(pos + 1);
                consumed += pos + 1;
                break;
            }
            None => {
                let want = available.len();
                let take = want.min(cap.saturating_sub(buf.len()));
                buf.extend_from_slice(&available[..take]);
                truncated |= take < want;
                reader.consume(want);
                consumed += want;
            }
        }
    }
    Ok((consumed, truncated))
}

#[async_trait]
impl Tool for Read {
    fn name(&self) -> &str {
        "read"
    }
    fn description(&self) -> &str {
        "Read a text file. Optionally start at a 1-based line `offset` and return up to `limit` \
         lines. Output is line-numbered."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to read." },
                "offset": { "type": "integer", "description": "1-based first line to read." },
                "limit": { "type": "integer", "description": "Max lines to return." }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;

        // An image file is returned as an attachment the multimodal model can see, not decoded as
        // UTF-8 text (which would hand back garbage). Detected by extension.
        if let Some(media_type) = image_media_type(path) {
            return read_image(path, media_type);
        }

        let offset = input
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT);

        // Stream the file line-by-line rather than slurping it whole: a windowed read
        // (`offset`/`limit`) into a huge file shouldn't allocate the entire file first — we hold at
        // most one line plus the bounded output window.
        let file = std::fs::File::open(path)
            .map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
        let mut reader = BufReader::new(file);

        let mut out = String::new();
        let mut lineno = 0usize;
        let mut shown = 0usize;
        let mut truncated = false;
        let mut line: Vec<u8> = Vec::new();
        loop {
            let (n, line_clipped) = read_line_capped(&mut reader, &mut line, MAX_LINE_BYTES)
                .map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
            if n == 0 {
                break; // EOF
            }
            lineno += 1;
            if lineno < offset {
                continue; // skipped lines are still drained, so memory stays bounded
            }
            // Stop before showing this line if we've hit the line limit or the byte budget; either way
            // `lineno` is the first *un*shown line, so it's exactly the offset to continue from.
            if shown >= limit || out.len() >= MAX_OUTPUT_BYTES {
                truncated = true;
                break;
            }
            // `read_line_capped` already stripped the trailing `\n`; strip a `\r` too. A capped line
            // may end mid-codepoint, so decode lossily rather than erroring on the split byte.
            let kept = line.strip_suffix(b"\r").unwrap_or(&line);
            let text = String::from_utf8_lossy(kept);
            if line_clipped {
                out.push_str(&format!("{lineno:>6}\t{text}… [line truncated]\n"));
            } else {
                out.push_str(&format!("{lineno:>6}\t{text}\n"));
            }
            shown += 1;
        }

        if lineno == 0 {
            return Ok("(empty file)".into());
        }
        // An offset past the last line returns nothing useful — say so explicitly rather than handing
        // back a blank result the model can't interpret.
        if offset > lineno {
            return Err(ToolError::InvalidInput(format!(
                "offset {offset} is beyond end of file ({lineno} lines total)"
            )));
        }
        if truncated {
            let last = offset + shown - 1;
            out.push_str(&format!(
                "… (showing lines {offset}-{last}; use offset={lineno} to continue)\n"
            ));
        }
        Ok(out.into())
    }
}

/// The image MIME type for a path's extension, or `None` if it isn't a recognized image. Used to
/// route `read` to the attachment path instead of UTF-8 text decoding.
fn image_media_type(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return None,
    })
}

/// Read an image file and return it as a base64 [`ImageSource`] attachment (plus a short text note),
/// so the multimodal model can actually see it. Oversized files are refused rather than ballooning the
/// request with a multi-megabyte base64 payload.
fn read_image(path: &str, media_type: &str) -> Result<ToolOutput, ToolError> {
    let meta =
        std::fs::metadata(path).map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
    if meta.len() > MAX_IMAGE_BYTES {
        return Err(ToolError::InvalidInput(format!(
            "image {path} is {} bytes; larger than the {MAX_IMAGE_BYTES}-byte inline limit",
            meta.len()
        )));
    }
    let bytes =
        std::fs::read(path).map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(ToolOutput::image(
        format!("Read image {path} ({media_type}, {} bytes).", meta.len()),
        ImageSource::base64(media_type, data),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_file(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    #[tokio::test]
    async fn reads_with_line_numbers() {
        let f = tmp_file("alpha\nbeta\ngamma\n");
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("     1\talpha"));
        assert!(out.contains("     3\tgamma"));
    }

    #[tokio::test]
    async fn offset_past_eof_is_an_error() {
        let f = tmp_file("a\nb\nc\n");
        let err = Read
            .run(json!({ "path": f.path().to_str().unwrap(), "offset": 99 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn truncation_reports_next_offset() {
        let f = tmp_file("a\nb\nc\nd\ne\n");
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap(), "limit": 2 }))
            .await
            .unwrap()
            .text;
        //Showed lines 1-2; the model is told to continue at line 3.
        assert!(out.contains("use offset=3 to continue"), "got: {out}");
    }

    #[tokio::test]
    async fn honors_offset_and_limit() {
        let f = tmp_file("a\nb\nc\nd\ne\n");
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap(), "offset": 2, "limit": 2 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("     2\tb"));
        assert!(out.contains("     3\tc"));
        assert!(!out.contains("\td\n"));
    }

    #[tokio::test]
    async fn missing_file_is_execution_error() {
        let err = Read
            .run(json!({ "path": "/no/such/file/xyz" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn caps_a_pathological_single_line() {
        // One line far larger than the per-line cap, followed by a normal line: the giant line is
        // clipped (with a marker) and the file's line structure is still tracked past it.
        let huge = "x".repeat(MAX_LINE_BYTES * 4);
        let f = tmp_file(&format!("{huge}\nnext\n"));
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        //The stored line is bounded, not the full 16k — the cap plus the framing/marker overhead.
        assert!(
            out.len() < MAX_LINE_BYTES * 2,
            "line was not capped: {} bytes",
            out.len()
        );
        assert!(out.contains("[line truncated]"));
        // The next line is read correctly as line 2 (overflow was drained, not mis-parsed).
        assert!(out.contains("     2\tnext"));
    }

    #[tokio::test]
    async fn reads_an_image_as_an_attachment() {
        // A .png is returned as a base64 image attachment, not UTF-8-decoded into garbage text.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        // A minimal PNG header byte sequence is enough — we only base64 the bytes.
        std::fs::write(&path, [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]).unwrap();
        let out = Read
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(
            out.images.len(),
            1,
            "image must be returned as an attachment"
        );
        assert_eq!(out.images[0].media_type, "image/png");
        assert!(!out.images[0].data.is_empty(), "base64 payload present");
        assert!(
            out.text.contains("Read image"),
            "a text note accompanies it"
        );
    }
}
