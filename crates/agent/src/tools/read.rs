//! `read` — read a file, optionally a line range, with line numbers.

use std::fmt::Write as _;
use std::io::{BufRead, BufReader};

use agent_core::message::ImageSource;
use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use base64::Engine as _;
use image::ImageDecoder as _;
use image::ImageEncoder as _;
use serde_json::{Value, json};

/// Base64-encoded budget for an inlined image, matching pi's `resizeImageInProcess` default — the
/// budget is on the *base64* payload (what actually rides in the request), not the raw file bytes,
/// leaving headroom under Anthropic's real ~5MB request-image limit.
const MAX_IMAGE_BASE64_BYTES: f64 = 4.5 * 1024.0 * 1024.0;
/// Max width/height an oversized image is downscaled to before re-encoding — pi's default.
const MAX_IMAGE_DIMENSION: u32 = 2000;
/// JPEG quality used when an oversized image must be re-encoded to fit the base64 budget — pi's
/// default. Only applies on the resize path; an image that already fits is sent as its original bytes,
/// original format, unmodified.
const JPEG_QUALITY: u8 = 80;

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
        // UTF-8 text (which would hand back garbage). The extension gate decides *whether* to take this
        // path (so an ordinary text-file read never pays for an extra image-format probe); `read_image`
        // then sniffs the real magic bytes to recover from a mislabeled extension.
        if let Some(ext_format) = extension_image_format(path) {
            return read_image(path, ext_format);
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
        let mut has_invalid_utf8 = false;
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
            let text = match std::str::from_utf8(kept) {
                Ok(s) => std::borrow::Cow::Borrowed(s),
                Err(e) => {
                    // A clipped line can legitimately end mid-codepoint — our own cap splitting a
                    // multi-byte sequence, not a sign the file itself is non-UTF-8. Only treat it as
                    // the latter when the invalid bytes start well before the end (more than one
                    // UTF-8 sequence's worth), or the line wasn't clipped at all so there's no
                    // cap-related excuse. `edit` requires strictly valid UTF-8 and will refuse a file
                    // flagged here, so the model needs to know *why* if it later can't edit it.
                    let split_by_our_clip = line_clipped && kept.len() - e.valid_up_to() <= 3;
                    if !split_by_our_clip {
                        has_invalid_utf8 = true;
                    }
                    String::from_utf8_lossy(kept)
                }
            };
            // Write straight into `out` instead of allocating a `format!` temp String per line — a
            // windowed read of a large file did one heap allocation for every line shown. `writeln!`
            // into a `String` can't fail, so the `Result` is discarded.
            let _ = if line_clipped {
                writeln!(out, "{lineno:>6}\t{text}… [line truncated]")
            } else {
                writeln!(out, "{lineno:>6}\t{text}")
            };
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
            out.push_str(&super::output::marker(format_args!(
                "showing lines {offset}-{last}; use offset={lineno} to continue"
            )));
            out.push('\n');
        }
        if has_invalid_utf8 {
            out.push_str(
                "… (this file contains bytes that are not valid UTF-8; the text above used lossy \
                 replacement (\u{FFFD}) for them, so it may not match the file's real bytes, and \
                 `edit` will refuse to touch this file until that's fixed)\n",
            );
        }
        Ok(out.into())
    }
}

/// The image format implied by a path's extension, or `None` if it isn't a recognized one. Used only
/// to decide *whether* `read` should route to the attachment path at all — the actual format sent on
/// the wire comes from sniffing the file's real magic bytes (see [`read_image`]), so a mislabeled
/// extension (a `.jpg` that's actually a PNG) still reports its true format.
fn extension_image_format(path: &str) -> Option<image::ImageFormat> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    Some(match ext.as_str() {
        "png" => image::ImageFormat::Png,
        "jpg" | "jpeg" => image::ImageFormat::Jpeg,
        "gif" => image::ImageFormat::Gif,
        "webp" => image::ImageFormat::WebP,
        "bmp" => image::ImageFormat::Bmp,
        _ => return None,
    })
}

/// The IANA media type for an [`image::ImageFormat`] this tool supports (the five extensions
/// [`extension_image_format`] recognizes).
fn media_type_of(format: image::ImageFormat) -> &'static str {
    match format {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::Gif => "image/gif",
        image::ImageFormat::WebP => "image/webp",
        image::ImageFormat::Bmp => "image/bmp",
        _ => "application/octet-stream",
    }
}

/// The base64-encoded length of `n` raw bytes, without actually encoding — 4 output chars per 3 input
/// bytes, rounded up.
fn base64_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

/// Losslessly re-encode `bytes` (already known to be `format`) as PNG, or `None` if decoding fails —
/// used to normalize a format (BMP) most vision APIs don't accept into one they do.
fn convert_to_png(bytes: &[u8], format: image::ImageFormat) -> Option<Vec<u8>> {
    let img = image::load_from_memory_with_format(bytes, format).ok()?;
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .ok()?;
    Some(out)
}

/// Read an image file and return it as a base64 [`ImageSource`] attachment (plus a short text note), so
/// the multimodal model can actually see it. `ext_format` is only the fallback used when magic-byte
/// sniffing can't identify the real format (a truncated or corrupt header); the sniffed format is
/// preferred whenever it succeeds. An image that already fits [`MAX_IMAGE_BASE64_BYTES`] is sent as its
/// original bytes, unmodified; an oversized one is downscaled/re-encoded by [`resize_image`], and only
/// refused outright if even that can't fit the budget.
fn read_image(path: &str, ext_format: image::ImageFormat) -> Result<ToolOutput, ToolError> {
    let bytes =
        std::fs::read(path).map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
    let format = image::guess_format(&bytes).unwrap_or(ext_format);

    // Most vision APIs (Anthropic: png/jpeg/gif/webp only) reject BMP outright. Convert it losslessly
    // to PNG up front — matching pi's `normalizeSupportedImageMimeType`, which always converts BMP
    // before it can reach the model — rather than sending a media type the provider will 400 on. A
    // failed conversion falls through with the original bytes/format; the caller still gets a
    // best-effort attachment rather than a hard error over a format quirk.
    let (bytes, format) = if format == image::ImageFormat::Bmp {
        match convert_to_png(&bytes, format) {
            Some(png) => (png, image::ImageFormat::Png),
            None => (bytes, format),
        }
    } else {
        (bytes, format)
    };
    let media_type = media_type_of(format);

    if base64_len(bytes.len()) <= MAX_IMAGE_BASE64_BYTES as usize {
        let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
        return Ok(ToolOutput::image(
            format!("Read image {path} ({media_type}, {} bytes).", bytes.len()),
            ImageSource::base64(media_type, data),
        ));
    }

    match resize_image(
        &bytes,
        format,
        MAX_IMAGE_DIMENSION,
        MAX_IMAGE_DIMENSION,
        MAX_IMAGE_BASE64_BYTES as usize,
        JPEG_QUALITY,
    ) {
        Some(resized) => {
            let (orig_w, orig_h) = resized.orig_dimensions;
            let (new_w, new_h) = resized.dimensions;
            let resized_media_type = resized.media_type;
            Ok(ToolOutput::image(
                format!(
                    "Read image {path} ({media_type}, {} bytes); resized from {orig_w}x{orig_h} to \
                     {new_w}x{new_h} ({resized_media_type}) to fit within the inline size budget.",
                    bytes.len()
                ),
                ImageSource::base64(resized_media_type, resized.base64_data),
            ))
        }
        None => Err(ToolError::InvalidInput(format!(
            "image {path} is {} bytes and could not be downscaled to fit the inline size budget",
            bytes.len()
        ))),
    }
}

/// The result of a successful [`resize_image`] call.
struct ResizedImage {
    base64_data: String,
    media_type: &'static str,
    dimensions: (u32, u32),
    orig_dimensions: (u32, u32),
}

/// Downscale/re-encode an oversized image to fit under `max_base64_bytes`. Applies JPEG/WebP EXIF
/// orientation correction before resizing (a camera photo's pixels are often stored un-rotated, with
/// the intended rotation recorded separately in Exif — skipping this would silently show the model a
/// sideways image), then resizes with Lanczos3 (matching pi's `resizeImageInProcess`) to fit within
/// `max_width`x`max_height`. At each size tried, a lossless PNG re-encode is attempted *first* — pi's
/// own `tryEncodings` order — since a downscaled screenshot/diagram/text-heavy image often already fits
/// losslessly, and a lossy JPEG re-encode would otherwise smear small text at block-compression edges
/// for no reason; only when the PNG doesn't fit the budget does it fall back to JPEG (the only format
/// here with a quality knob left to trade against size). If PNG-at-this-size still doesn't fit, both
/// the dimensions and JPEG quality are stepped down and it retries a bounded number of times; returns
/// `None` only if even the smallest re-encode can't fit.
fn resize_image(
    bytes: &[u8],
    format: image::ImageFormat,
    max_width: u32,
    max_height: u32,
    max_base64_bytes: usize,
    jpeg_quality: u8,
) -> Option<ResizedImage> {
    // Read orientation off the decoder itself — `image`'s own generic Exif-orientation support
    // (`ImageDecoder::orientation`), backed by a real parser for every format that can carry Exif
    // (JPEG, WebP, …), rather than a hand-rolled one-format-at-a-time parser here. A camera/phone
    // photo's pixels are often stored un-rotated, with the intended rotation recorded separately in
    // Exif — skipping this would silently show the model a sideways image.
    let mut decoder = image::ImageReader::with_format(std::io::Cursor::new(bytes), format)
        .into_decoder()
        .ok()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder).ok()?;
    img.apply_orientation(orientation);
    let orig_dims = (img.width(), img.height());

    let mut width = max_width.min(orig_dims.0).max(1);
    let mut height = max_height.min(orig_dims.1).max(1);
    let mut quality = jpeg_quality;
    // A handful of shrink-and-retry rounds is enough in practice to find a fit (each round cuts pixel
    // count by ~36% and quality by 10); a pathological image that still can't fit at the size/quality
    // floor genuinely can't be served inline, and `None` tells the caller to say so.
    for _ in 0..6 {
        let scaled = if width < orig_dims.0 || height < orig_dims.1 {
            img.resize(width, height, image::imageops::FilterType::Lanczos3)
        } else {
            img.clone()
        };
        let rgb = scaled.to_rgb8();

        let mut png_buf = Vec::new();
        let png_ok = image::codecs::png::PngEncoder::new(&mut png_buf)
            .write_image(
                rgb.as_raw(),
                scaled.width(),
                scaled.height(),
                image::ExtendedColorType::Rgb8,
            )
            .is_ok();
        if png_ok {
            let data = base64::engine::general_purpose::STANDARD.encode(&png_buf);
            if data.len() <= max_base64_bytes {
                return Some(ResizedImage {
                    base64_data: data,
                    media_type: "image/png",
                    dimensions: (scaled.width(), scaled.height()),
                    orig_dimensions: orig_dims,
                });
            }
        }

        let mut jpeg_buf = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_buf, quality);
        if encoder
            .write_image(
                rgb.as_raw(),
                scaled.width(),
                scaled.height(),
                image::ExtendedColorType::Rgb8,
            )
            .is_err()
        {
            return None;
        }
        let data = base64::engine::general_purpose::STANDARD.encode(&jpeg_buf);
        if data.len() <= max_base64_bytes {
            return Some(ResizedImage {
                base64_data: data,
                media_type: "image/jpeg",
                dimensions: (scaled.width(), scaled.height()),
                orig_dimensions: orig_dims,
            });
        }
        width = ((width as f64) * 0.8) as u32;
        height = ((height as f64) * 0.8) as u32;
        width = width.max(64);
        height = height.max(64);
        quality = quality.saturating_sub(10).max(30);
    }
    None
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
        assert!(
            out.contains("[showing lines 1-2; use offset=3 to continue]"),
            "got: {out}"
        );
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
    async fn invalid_utf8_is_flagged_so_the_model_knows_edit_will_refuse_it() {
        // `edit` requires strictly valid UTF-8 and errors on a file like this one; `read` must not
        // silently lossy-decode it with no signal, or the model has no way to connect a later
        // `edit` failure back to this file's actual problem.
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"good line\n\xff\xfe not valid utf-8\nlast\n")
            .unwrap();
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("not valid UTF-8"),
            "missing the invalid-UTF-8 note: {out}"
        );
        // The other, genuinely valid lines are still shown.
        assert!(out.contains("good line"));
        assert!(out.contains("last"));
    }

    #[tokio::test]
    async fn a_clip_point_splitting_a_codepoint_is_not_flagged_as_invalid_utf8() {
        // A pathologically long but otherwise perfectly valid UTF-8 line can have its cap land
        // mid-codepoint — that's an artifact of our own truncation, not evidence the file itself is
        // non-UTF-8, and must not trigger the same warning as genuinely invalid bytes.
        let huge = "é".repeat(MAX_LINE_BYTES); // 2-byte UTF-8 char, so the byte cap can split one
        let f = tmp_file(&format!("{huge}\n"));
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("[line truncated]"));
        assert!(
            !out.contains("not valid UTF-8"),
            "a clip-induced split codepoint must not be flagged as the file being invalid: {out}"
        );
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

    #[tokio::test]
    async fn oversized_image_is_downscaled_to_fit_the_base64_budget() {
        // An uncompressed BMP well past MAX_IMAGE_DIMENSION on one side and comfortably over the
        // base64 budget in raw size — proves the resize path actually engages and lands under budget.
        // Pixels are pseudo-random noise (a cheap xorshift-style mix, not a real RNG dependency) rather
        // than a smooth gradient: a gradient PNG-compresses so well that this BMP fixture would (after
        // the BMP-to-PNG normalization every image now goes through) land back *under* budget without
        // ever exercising the resize path at all, since PNG's DEFLATE handles smooth data extremely
        // well but not noise.
        let width = MAX_IMAGE_DIMENSION + 800;
        let height = 600;
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            let h = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)).wrapping_add(0x9e3779b9);
            image::Rgb([
                (h & 0xff) as u8,
                ((h >> 8) & 0xff) as u8,
                ((h >> 16) & 0xff) as u8,
            ])
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.bmp");
        image::DynamicImage::ImageRgb8(img)
            .save_with_format(&path, image::ImageFormat::Bmp)
            .unwrap();
        assert!(
            base64_len(std::fs::metadata(&path).unwrap().len() as usize)
                > MAX_IMAGE_BASE64_BYTES as usize,
            "fixture must actually exceed the budget pre-resize"
        );

        let out = Read
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        // PNG is tried first at every size step (see `resize_image`'s doc comment) and, once shrunk
        // enough by the retry loop, even incompressible noise's raw pixel data fits comfortably under
        // budget as lossless PNG — so that's what wins here, not JPEG. (`resize_image_falls_back_to_
        // jpeg_when_png_does_not_fit` isolates and proves the JPEG-fallback path directly, without
        // depending on the outer shrink loop's eventual convergence to a trivially-small image.)
        assert_eq!(
            out.images[0].media_type, "image/png",
            "a small enough downscaled image fits losslessly, so PNG wins over JPEG"
        );
        assert!(
            out.images[0].data.len() <= MAX_IMAGE_BASE64_BYTES as usize,
            "resized payload must fit the budget: {} bytes",
            out.images[0].data.len()
        );
        assert!(
            out.text.contains("resized from"),
            "the resize must be noted in the text: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn small_bmp_is_converted_to_png_not_sent_as_bmp() {
        // Most vision APIs (Anthropic: png/jpeg/gif/webp only) reject BMP outright — a BMP comfortably
        // under the inline size budget (so the resize path never engages) must still come back as PNG,
        // not the original, unsupported `image/bmp`.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.bmp");
        image::DynamicImage::ImageRgb8(img)
            .save_with_format(&path, image::ImageFormat::Bmp)
            .unwrap();
        assert!(
            base64_len(std::fs::metadata(&path).unwrap().len() as usize)
                <= MAX_IMAGE_BASE64_BYTES as usize,
            "fixture must fit the budget as-is, so the resize path never engages"
        );

        let out = Read
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        assert_eq!(
            out.images[0].media_type, "image/png",
            "BMP must be converted to PNG, never sent as-is"
        );
        // The converted bytes must actually decode as a real PNG carrying the same pixels.
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&out.images[0].data)
            .unwrap();
        assert_eq!(
            image::guess_format(&decoded).unwrap(),
            image::ImageFormat::Png
        );
        let round_tripped = image::load_from_memory(&decoded).unwrap().to_rgb8();
        assert_eq!(round_tripped.get_pixel(0, 0), &image::Rgb([10, 20, 30]));
    }

    #[tokio::test]
    async fn mislabeled_extension_is_still_correctly_sniffed() {
        // A real PNG saved under a `.jpg` name: the extension gate still routes it to the image path
        // (so it isn't UTF-8-decoded as garbage text), but the *reported* format comes from sniffing
        // the real magic bytes, not trusting the wrong extension.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 100, 50]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("actually_a_png.jpg");
        std::fs::write(&path, &png_bytes).unwrap();

        let out = Read
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        assert_eq!(
            out.images[0].media_type, "image/png",
            "sniffed format must win over the misleading .jpg extension"
        );
    }

    /// A minimal TIFF-structured IFD0 (one entry: orientation) — the same blob JPEG carries after its
    /// APP1 segment's `Exif\0\0` prefix and WebP carries verbatim as its `EXIF` chunk payload.
    fn tiff_with_orientation(orientation: u16) -> Vec<u8> {
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II"); // little-endian
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 starts right after this header
        tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
        tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // tag: Orientation
        tiff.extend_from_slice(&3u16.to_le_bytes()); // type: SHORT
        tiff.extend_from_slice(&1u32.to_le_bytes()); // count: 1
        tiff.extend_from_slice(&orientation.to_le_bytes());
        tiff.extend_from_slice(&[0, 0]); // pad the 4-byte value field
        tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        tiff
    }

    /// Splice a synthetic Exif APP1 segment (TIFF IFD0, one entry: orientation) right after a JPEG's
    /// SOI marker — real cameras embed Exif the same way, and a conforming decoder skips an APPn
    /// segment it doesn't otherwise care about, so this still decodes as a normal JPEG.
    fn jpeg_with_exif_orientation(width: u32, height: u32, orientation: u16) -> Vec<u8> {
        let img =
            image::RgbImage::from_fn(width, height, |x, _| image::Rgb([(x % 256) as u8, 0, 0]));
        let mut jpeg_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg_bytes),
                image::ImageFormat::Jpeg,
            )
            .unwrap();

        let mut app1 = Vec::new();
        app1.extend_from_slice(b"Exif\0\0");
        app1.extend_from_slice(&tiff_with_orientation(orientation));
        let seg_len = (app1.len() + 2) as u16;

        let mut out = Vec::new();
        out.extend_from_slice(&jpeg_bytes[0..2]); // SOI
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&seg_len.to_be_bytes());
        out.extend_from_slice(&app1);
        out.extend_from_slice(&jpeg_bytes[2..]); // the rest of the real JPEG stream
        out
    }

    /// Wrap a real WebP bitstream in the "extended" (VP8X) container and append a synthetic `EXIF`
    /// chunk (the same TIFF IFD0 blob JPEG carries). A conforming decoder only looks for metadata
    /// chunks *after* seeing `VP8X` declare (via its flags byte) that they're present — the plain
    /// container `image`'s own encoder writes has no room for them at all — so this, not simply
    /// splicing a chunk into a plain file, is what a real Exif-carrying WebP looks like on disk.
    fn webp_with_exif_orientation(width: u32, height: u32, orientation: u16) -> Vec<u8> {
        let img =
            image::RgbImage::from_fn(width, height, |x, _| image::Rgb([(x % 256) as u8, 0, 0]));
        let mut webp_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut webp_bytes),
                image::ImageFormat::WebP,
            )
            .unwrap();
        // Everything after the 12-byte "RIFF"+size+"WEBP" header is the plain VP8/VP8L bitstream
        // chunk (FourCC + size + payload, already padded to an even boundary) — copied verbatim into
        // the extended container built below.
        let bitstream_chunk = &webp_bytes[12..];

        let tiff = tiff_with_orientation(orientation);
        let mut exif_chunk = Vec::new();
        exif_chunk.extend_from_slice(b"EXIF");
        exif_chunk.extend_from_slice(&(tiff.len() as u32).to_le_bytes());
        exif_chunk.extend_from_slice(&tiff);
        if tiff.len() % 2 == 1 {
            exif_chunk.push(0); // RIFF chunks pad to an even byte boundary
        }

        let mut vp8x_payload = Vec::new();
        vp8x_payload.push(0b0000_1000); // flags: bit 3 = Exif metadata present
        vp8x_payload.extend_from_slice(&[0, 0, 0]); // reserved
        vp8x_payload.extend_from_slice(&(width - 1).to_le_bytes()[0..3]); // canvas width - 1
        vp8x_payload.extend_from_slice(&(height - 1).to_le_bytes()[0..3]); // canvas height - 1
        let mut vp8x_chunk = Vec::new();
        vp8x_chunk.extend_from_slice(b"VP8X");
        vp8x_chunk.extend_from_slice(&(vp8x_payload.len() as u32).to_le_bytes());
        vp8x_chunk.extend_from_slice(&vp8x_payload);

        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        let riff_size = (4 + vp8x_chunk.len() + bitstream_chunk.len() + exif_chunk.len()) as u32;
        out.extend_from_slice(&riff_size.to_le_bytes());
        out.extend_from_slice(b"WEBP");
        out.extend_from_slice(&vp8x_chunk);
        out.extend_from_slice(bitstream_chunk);
        out.extend_from_slice(&exif_chunk);
        out
    }

    #[test]
    fn a_plain_webp_with_no_exif_chunk_decodes_unchanged_through_resize_image() {
        // No orientation metadata at all: `resize_image` must decode normally rather than treating a
        // missing Exif chunk as an error (`ImageDecoder::orientation` defaults to `NoTransforms`).
        let img = image::RgbImage::from_pixel(6, 3, image::Rgb([1, 2, 3]));
        let mut plain = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut plain),
                image::ImageFormat::WebP,
            )
            .unwrap();
        let resized = resize_image(
            &plain,
            image::ImageFormat::WebP,
            100,
            100,
            10 * 1024 * 1024,
            80,
        )
        .unwrap();
        assert_eq!(resized.dimensions, (6, 3));
    }

    #[test]
    fn exif_orientation_6_swaps_width_and_height_for_a_webp_through_resize_image() {
        // Same "orientation 6 = rotate 90°" swap the JPEG test proves below, for the WebP path.
        let bytes = webp_with_exif_orientation(8, 4, 6);
        let resized = resize_image(
            &bytes,
            image::ImageFormat::WebP,
            100,
            100,
            10 * 1024 * 1024,
            80,
        )
        .unwrap();
        assert_eq!(resized.dimensions, (4, 8));
    }

    #[test]
    fn exif_orientation_6_swaps_width_and_height_through_resize_image() {
        // Orientation 6 means "rotate 90° CW to display correctly" — an 8-wide x 4-tall raw pixel
        // buffer must come out 4-wide x 8-tall once `resize_image` applies the correction.
        let bytes = jpeg_with_exif_orientation(8, 4, 6);
        let resized = resize_image(
            &bytes,
            image::ImageFormat::Jpeg,
            1000,
            1000,
            10 * 1024 * 1024,
            90,
        )
        .expect("a tiny image at a generous budget must always fit");
        // A tiny image at a 10MB budget fits losslessly, so PNG (tried first — see `resize_image`'s
        // doc comment) wins; this test's real point is the dimension swap below, not the format.
        assert_eq!(resized.media_type, "image/png");
        assert_eq!(
            resized.dimensions,
            (4, 8),
            "dimensions must be swapped by the rotation"
        );
    }

    #[test]
    fn resize_image_prefers_lossless_png_when_it_fits() {
        // A smooth gradient compresses extremely well under PNG — once downscaled, it should come back
        // losslessly rather than needlessly re-encoded as lossy JPEG (pi's own `tryEncodings` order:
        // PNG tried first at every size step, JPEG only as a fallback).
        let img = image::RgbImage::from_fn(400, 400, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        let resized = resize_image(&bytes, image::ImageFormat::Png, 200, 200, 200_000, 80)
            .expect("a downscaled smooth gradient must fit comfortably");
        assert_eq!(resized.media_type, "image/png");
    }

    #[test]
    fn resize_image_falls_back_to_jpeg_when_png_does_not_fit() {
        // Pseudo-random noise at a budget deliberately sized between "fits as JPEG-80" and "too big as
        // lossless PNG" at the *same* dimensions (no shrinking needed) — isolates the per-size
        // PNG-then-JPEG fallback from the outer shrink loop's eventual convergence (a small enough
        // image always fits as PNG *eventually*, which a different test already covers).
        let img = image::RgbImage::from_fn(300, 300, |x, y| {
            let h = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)).wrapping_add(0x9e3779b9);
            image::Rgb([
                (h & 0xff) as u8,
                ((h >> 8) & 0xff) as u8,
                ((h >> 16) & 0xff) as u8,
            ])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        // Sanity: confirm the fixture is actually shaped as intended before trusting the real assertion.
        let mut png_at_size = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png_at_size)
            .write_image(
                &image::load_from_memory(&bytes).unwrap().to_rgb8(),
                300,
                300,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let png_b64_len = base64_len(png_at_size.len());
        let budget = png_b64_len - 1; // just under the lossless size at this exact resolution

        let resized = resize_image(&bytes, image::ImageFormat::Png, 300, 300, budget, 80)
            .expect("JPEG must still fit even where lossless PNG does not");
        assert_eq!(resized.media_type, "image/jpeg");
    }
}
