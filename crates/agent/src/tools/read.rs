//! `read` — read a file, optionally a line range, with line numbers.

use std::io::{BufRead, BufReader};

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};

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

    async fn run(&self, input: Value) -> Result<String, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;

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
        Ok(out)
    }
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
            .unwrap();
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
            .unwrap();
        // Showed lines 1-2; the model is told to continue at line 3.
        assert!(out.contains("use offset=3 to continue"), "got: {out}");
    }

    #[tokio::test]
    async fn honors_offset_and_limit() {
        let f = tmp_file("a\nb\nc\nd\ne\n");
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap(), "offset": 2, "limit": 2 }))
            .await
            .unwrap();
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
            .unwrap();
        // The stored line is bounded, not the full 16k — the cap plus the framing/marker overhead.
        assert!(
            out.len() < MAX_LINE_BYTES * 2,
            "line was not capped: {} bytes",
            out.len()
        );
        assert!(out.contains("[line truncated]"));
        // The next line is read correctly as line 2 (overflow was drained, not mis-parsed).
        assert!(out.contains("     2\tnext"));
    }
}
