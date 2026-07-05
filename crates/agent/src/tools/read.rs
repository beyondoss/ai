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

/// Hard ceiling on returned lines: the default when the caller omits `limit` entirely, and a secondary
/// cap even when an explicit, larger `limit` is given — matching pi's `read.ts`, which always runs the
/// selected window through `truncateHead`'s own default `maxLines` (2000) regardless of the caller's
/// `limit`. A single request asking for a much larger explicit limit must not flood context any more
/// than the omitted-limit case already doesn't.
const DEFAULT_LIMIT: usize = 2000;
/// Cap on bytes kept per line. Streaming by line already bounds memory to one line, but a file with a
/// single pathological line (a minified bundle, a one-line JSON/CSV blob) is one unbounded line — so
/// cap each line and drain the overflow without storing it, the way `grep` clips long match lines.
///
/// Deliberately a **mid-line** clip (a `"… [line truncated]"` marker after the kept prefix), not pi's
/// own `truncateHead` (`truncate.ts`), which "never returns partial lines": there, a line that alone
/// exceeds the byte budget is either dropped from the output entirely (hit mid-scan) or, worse, the
/// *whole read* comes back empty with `firstLineExceedsLimit: true` (hit on line one) — a model asking
/// to read a single-giant-line file gets nothing useful at all. Showing a clipped prefix instead means
/// the model always sees *something* of every line — better partial signal than an omission it can't
/// tell apart from "this file has no line-one content."
const MAX_LINE_BYTES: usize = 4000;
/// Aggregate output budget: the line limit alone (2000 × up to 4000 bytes) could return ~8MB of
/// mostly-long lines into context. Stop once the rendered output crosses this, telling the model how
/// to continue (the `offset=N` pagination already exists for exactly this, so a smaller budget here
/// loses nothing — it just means one more `read` call). Shares `grep`/`find`/`ls`'s own budget
/// ([`super::output::DEFAULT_MAX_BYTES`], pi's uniform 50KB) rather than a bespoke, 5x-larger constant.
const MAX_OUTPUT_BYTES: usize = super::output::DEFAULT_MAX_BYTES;

/// pi's `getNonVisionImageNote` (`read.ts`) — appended to an image read's text whenever the active
/// model can't accept image input at all, so the model understands why an attachment (or an
/// omitted-image placeholder) is missing or irrelevant to it. See `Read::run`'s `supports_vision` for
/// how (and how incompletely) that capability reaches this tool.
const NON_VISION_IMAGE_NOTE: &str =
    "[Current model does not support images. The image will be omitted from this request.]";

pub struct Read {
    /// Whether an oversized image is downscaled/re-encoded to fit the inline size budget before being
    /// sent to the model — pi's `ImageSettings.autoResize` (default `true`). `false` (`agent run
    /// --no-image-auto-resize`/`agent settings --image-auto-resize false`) skips [`resize_image`]
    /// entirely and ships the normalized (format-converted, if a BMP) bytes as-is regardless of size,
    /// matching pi's `processImage`, which never calls `resizeImage` in that case either. Set once at
    /// registry-construction time (see `default_registry_with_prefix_and_image_auto_resize` in
    /// `tools/mod.rs`) rather than per-call — mirrors how `bash`'s own tunables (`default_timeout_ms`,
    /// `shell_path`) are configured.
    image_auto_resize: bool,
}

impl Default for Read {
    fn default() -> Self {
        Self {
            image_auto_resize: true,
        }
    }
}

impl Read {
    /// Builder-style: gate the resize/downscale path on `enabled` — see the field's own doc comment.
    pub fn with_image_auto_resize(mut self, enabled: bool) -> Self {
        self.image_auto_resize = enabled;
        self
    }
}

/// If `path` doesn't exist as given, try a small set of Unicode-normalization fallbacks pi's `read`
/// tool also tries (`resolveReadPath`), in order: NFD-decomposed form, straight-to-curly-quote
/// substitution (`'` → U+2019), and the combination of both. A model or user typing an NFC/straight-
/// quote path when the real file — often synced from, or itself referencing, a macOS-authored name —
/// uses NFD/curly-quote form would otherwise fail with a confusing "not found" despite the file
/// genuinely existing. Returns `path` unchanged when it already exists, or when none of the fallbacks
/// resolve to a real file (the caller's own error path still fires, just against the original path so
/// the error message names what the caller actually typed). Deliberately skips pi's macOS-specific
/// AM/PM narrow-no-break-space screenshot variant — that's Finder-clipboard-naming-specific and this
/// agent targets Linux, where it can never fire.
///
/// Each candidate's `exists()` check here is a *selection* probe, not a redundant pre-check
/// immediately shadowed by an open a line later: the winning candidate's path string is threaded
/// through several more steps in `Read::run` (image-format sniffing, the image-vs-text branch, offset/
/// limit line-range handling, error messages) before anything actually opens the file, and — unlike a
/// simple "does this one file exist" gate — there's no single `open()` whose own error could stand in
/// for these checks, since the job is picking *which* of up to four distinct paths to commit to.
/// Collapsing that selection onto `open()`'s error would need every one of `run`'s several separate
/// re-opens (sniff, image decode, the streaming text read) to renegotiate which candidate "won" instead
/// of trusting one already-resolved path string, for no real gain.
///
/// This does leave a TOCTOU window: a symlink swapped between a candidate's `exists()` here and its
/// later `File::open` elsewhere in `run` could change what actually gets read. That's accepted, not
/// overlooked — this crate has no path-sandbox/confinement in any file tool by deliberate design (the
/// agent already runs with the operator's own ambient privileges, and `bash` is an unrestricted shell
/// tool in the same tool set), so there is no privilege boundary here for a symlink swap to cross: an
/// attacker who can win that race can already just name the target path directly.
fn resolve_read_path(path: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    if std::path::Path::new(path).exists() {
        return path.to_string();
    }
    let nfd: String = path.nfd().collect();
    if nfd != path && std::path::Path::new(&nfd).exists() {
        return nfd;
    }
    let curly = path.replace('\'', "\u{2019}");
    if curly != path && std::path::Path::new(&curly).exists() {
        return curly;
    }
    let nfd_curly = nfd.replace('\'', "\u{2019}");
    if nfd_curly != path && std::path::Path::new(&nfd_curly).exists() {
        return nfd_curly;
    }
    path.to_string()
}

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
        "Read a file: text or image (png, jpeg, gif, webp, bmp). Images are sent as attachments the \
         model can see; a model without vision support gets a text note instead. For text files, \
         output is line-numbered and truncated to at most 2000 lines or 50.0KB, whichever is hit \
         first. Use `offset`/`limit` to continue reading a large file from where a truncated call \
         left off."
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
        let path = resolve_read_path(&super::normalize_path(path));

        // Whether the active model can accept image input at all. `agent_core::tool::Tool::run` takes
        // only a bare `Value` — unlike pi's `execute(..., ctx)`, there's no separate model-capability
        // context parameter reaching this tool (adding one is a `Tool`-trait-wide change) — so this is
        // read from an internal, schema-undocumented input field instead: `agent_core::agent`'s dispatch
        // loop injects it before every tool call, keyed off the active model's own
        // `agent_core::models::ModelCaps::supports_vision`. Defaults to `true` (vision capable) if
        // absent (e.g. a direct `Tool::run` call in a test, bypassing dispatch).
        let supports_vision = input
            .get("_model_supports_vision")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        // An image file is returned as an attachment the multimodal model can see, not decoded as
        // UTF-8 text (which would hand back garbage). Routing is decided primarily by sniffing the
        // real magic bytes (matching pi, which always probes regardless of extension) — a real image
        // saved under a missing or wrong extension (an extensionless clipboard/screenshot temp file, a
        // `.dat`) is still detected correctly, not silently UTF-8-decoded into garbage text. The
        // extension only matters as a fallback when sniffing can't identify a format at all (a
        // truncated/corrupt file whose header doesn't parse). `read_image` re-sniffs internally too
        // (for the *reported* format, once it has the full file, not just this prefix).
        //
        // An *animated* PNG is deliberately excluded from image handling entirely, extension fallback
        // included — see `Sniff::AnimatedPng`'s doc comment — so it falls all the way through to the
        // ordinary text path below, the same route a corrupt/non-image file named `.png` takes.
        let sniff = sniff_image_format(&path);
        let sniffed_format = match sniff {
            Sniff::Format(format) => Some(format),
            Sniff::AnimatedPng | Sniff::Unknown => None,
        };
        // An animated PNG must never reach image handling at all — not even via the extension
        // fallback below, which exists only for the genuinely *ambiguous* "sniffing found nothing"
        // case (`Sniff::Unknown`), not for "sniffing positively identified this as unservable".
        let format = match sniff {
            Sniff::AnimatedPng => None,
            _ => sniffed_format.or_else(|| extension_image_format(&path)),
        };
        if let Some(format) = format {
            // Decode/resize/re-encode is CPU-bound and can run for a while on a large image — matching
            // pi's `resizeImageInProcess`, which runs its WASM decode/resize/encode in a worker thread
            // specifically so it doesn't stall the event loop, keep it off this async runtime's worker
            // threads too (same pattern as `grep`/`find`'s own blocking walks).
            let path_for_task = path.clone();
            let auto_resize = self.image_auto_resize;
            let result =
                tokio::task::spawn_blocking(move || read_image(&path_for_task, format, auto_resize))
                    .await
                    .map_err(|e| ToolError::Execution(format!("image read task failed: {e}")))?;
            match result {
                Ok(mut out) => {
                    if !supports_vision {
                        out.images.clear();
                        if !out.text.is_empty() {
                            out.text.push('\n');
                        }
                        out.text.push_str(NON_VISION_IMAGE_NOTE);
                    }
                    return Ok(out);
                }
                // A real magic-byte match (`sniffed_format` is `Some`) that still fails to decode is a
                // genuinely corrupt/truncated image — surface that as a hard error, matching the
                // existing `a_small_file_with_valid_magic_bytes_but_no_real_image_body_is_rejected`
                // behavior.
                Err(err) if sniffed_format.is_some() => return Err(err),
                // Extension-only guess (no real magic-byte match at all) that failed to decode: this
                // "looks like an image by name only" — a Git LFS pointer file saved as `.png`, a
                // corrupted download, or any plain-text file someone named `.jpg`. Matching pi's `read`
                // tool (`tools.test.ts`, "should treat files with image extension but non-image content
                // as text"), fall through to the ordinary text read below instead of erroring.
                Err(_) => {}
            }
        }

        let offset = input
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| (n as usize).min(DEFAULT_LIMIT))
            .unwrap_or(DEFAULT_LIMIT);

        // Stream the file line-by-line rather than slurping it whole: a windowed read
        // (`offset`/`limit`) into a huge file shouldn't allocate the entire file first — we hold at
        // most one line plus the bounded output window.
        let file = std::fs::File::open(&path)
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
        //
        // `lineno` (and every "N lines total" this tool reports) counts real content lines only, via
        // `read_line_capped`'s own line-at-a-time scan — a file ending `"a\nb\nc\n"` reports 3, the
        // same as `"a\nb\nc"` with no trailing newline. This is a deliberate divergence from pi's own
        // `read.ts`, which counts `textContent.split("\n").length`: in JS, `"a\nb\nc\n".split("\n")`
        // yields 4 elements (a trailing empty string for the newline), so pi reports one *more* line
        // than the file actually has whenever it ends with a newline, but the correct count when it
        // doesn't — an inconsistency, not a convention worth matching. Every other line-oriented tool
        // here (`grep`, editors, `wc -l`) already treats "N lines" as "N newline-terminated-or-final
        // content lines," so keeping that same, trailing-newline-independent count is the more
        // correct and more consistent choice, not a parity gap.
        if offset > lineno {
            return Err(ToolError::InvalidInput(format!(
                "offset {offset} is beyond end of file ({lineno} lines total)"
            )));
        }
        if truncated {
            let last = offset + shown - 1;
            // `lineno` right now is the first not-yet-shown line — already consumed from the reader
            // (so it's exactly the right `offset` to continue from), but not counted towards `shown`
            // or written to `out`. Save it before the drain below overwrites `lineno` with the file's
            // true total.
            let continue_offset = lineno;
            // pi's own truncation message (`read.ts`) includes the file's true total line count
            // ("Showing lines X-Y of {totalFileLines}"); without it the model has no idea how much
            // more content is left. We stream and stop at the truncation point by design (see this
            // function's own doc comment on `MAX_LINE_BYTES`/streaming), so getting that total means
            // reading the rest of the file here — but only a cheap, bounded byte scan via
            // `read_line_capped`, the same streaming line counter already used above: no line is
            // stored or decoded, only counted, and reusing it keeps this total consistent with
            // `lineno`'s own "real content lines" convention (see the comment above on `offset >
            // lineno`) rather than a raw `\n` byte count that would double-count that same
            // trailing-newline edge case.
            loop {
                let (n, _clipped) = read_line_capped(&mut reader, &mut line, MAX_LINE_BYTES)
                    .map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
                if n == 0 {
                    break; // EOF
                }
                lineno += 1;
            }
            out.push_str(&super::output::marker(format_args!(
                "showing lines {offset}-{last} of {lineno} total; use offset={continue_offset} to continue"
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

/// How many leading bytes are read once for magic-byte sniffing and, for a PNG specifically, an
/// animated-PNG check. Bumped from a plain-magic-bytes-only 32 to pi's own `IMAGE_TYPE_SNIFF_BYTES`
/// (`mime.ts`) 4100: PNG's signature is 8 bytes, JPEG's SOI+marker is 2-3, GIF's is 6, BMP's is 2, and
/// WebP's `RIFF....WEBP` header is 12 (32 was already comfortably past all of those), but an animated
/// PNG's `acTL` chunk — which, per the APNG spec, always precedes the first `IDAT` — can sit behind
/// other ancillary chunks (`iCCP`, `sRGB`, `pHYs`, …) placed between `IHDR` and `IDAT`; 4100 bytes gives
/// [`is_animated_png`] enough room to actually reach it in practice, matching pi's own budget exactly.
const SNIFF_LEN: usize = 4100;

/// The result of sniffing a file's leading bytes.
enum Sniff {
    /// A recognized, servable image format.
    Format(image::ImageFormat),
    /// An *animated* PNG (an `acTL` chunk precedes the first `IDAT` — see [`is_animated_png`]).
    /// Deliberately its own variant, not folded into `Unknown`: sending a single static frame of an
    /// animated image to a vision model as if it were the whole picture is misleading (matching pi's
    /// `detectSupportedImageMimeType`, which returns `null` for an animated PNG so `read.ts` never
    /// attempts to decode/serve it as an image at all — it goes straight to the ordinary text path,
    /// the raw PNG bytes included, same as any other non-image file would). The `image` crate *can*
    /// decode an APNG's default/first frame as an ordinary static PNG, so without this distinction the
    /// extension-based fallback (see `Read::run`) would silently ship that one frame as if it were the
    /// complete image — this variant exists specifically so the caller can refuse that fallback too,
    /// not just skip the sniffed-format fast path.
    AnimatedPng,
    /// No recognized image signature (or the file couldn't be opened/read at all).
    Unknown,
}

/// Peek the first [`SNIFF_LEN`] bytes of `path` and identify a real image by its magic bytes,
/// regardless of what extension (if any) the file has — matching pi's `detectImageMimeType`, which
/// always probes unconditionally rather than gating on the extension first. Only ever reports one of
/// the five formats this tool actually knows how to encode/re-encode; any other format `guess_format`
/// might recognize (TIFF, ICO, …) is treated the same as "not an image" here, since there's no
/// supported path to serve it as an attachment. A PNG whose chunk stream is animated (see
/// [`is_animated_png`]) reports [`Sniff::AnimatedPng`] instead of [`Sniff::Format`], regardless of how
/// confidently `guess_format` would otherwise call it a plain PNG. A "BM"-prefixed file that doesn't
/// pass [`is_valid_bmp`]'s structural check, or a JPEG-signatured file whose 4th byte is `0xF7` (SOF55 —
/// see the inline match arm), reports [`Sniff::Unknown`] too, same as any other non-image file.
/// [`Sniff::Unknown`] on any read/probe failure (missing file, a directory, too few bytes, no
/// recognizable header) — the caller falls back to the extension gate, or ultimately the ordinary text
/// path.
fn sniff_image_format(path: &str) -> Sniff {
    use std::io::Read as _;
    let mut buf = [0u8; SNIFF_LEN];
    let Ok(mut f) = std::fs::File::open(path) else {
        return Sniff::Unknown;
    };
    let Ok(n) = f.read(&mut buf) else {
        return Sniff::Unknown;
    };
    let buf = &buf[..n];
    match image::guess_format(buf) {
        Ok(image::ImageFormat::Png) if is_animated_png(buf) => Sniff::AnimatedPng,
        // `image::guess_format`'s own BMP signature is just the 2 ASCII bytes "BM" (an empty mask in
        // its `MAGIC_BYTES` table) — any file merely starting with those two bytes (ordinary text, a
        // changelog entry, a variable name) would otherwise sniff as BMP. `is_valid_bmp` requires an
        // actually plausible DIB header first (pi's `isBmp`).
        Ok(image::ImageFormat::Bmp) if !is_valid_bmp(buf) => Sniff::Unknown,
        // SOF55 (JPEG-LS, a rare lossless variant) shares the ordinary JPEG's 3-byte SOI+marker prefix
        // but isn't a format this tool (or most vision APIs) can actually decode/display — pi's
        // `detectSupportedImageMimeType` (`mime.ts:6-9`) excludes it the same way.
        Ok(image::ImageFormat::Jpeg) if buf.get(3) == Some(&0xF7) => Sniff::Unknown,
        Ok(
            format @ (image::ImageFormat::Png
            | image::ImageFormat::Jpeg
            | image::ImageFormat::Gif
            | image::ImageFormat::WebP
            | image::ImageFormat::Bmp),
        ) => Sniff::Format(format),
        _ => Sniff::Unknown,
    }
}

/// Whether `buf` (a PNG's leading bytes, signature included) is an *animated* PNG: an `acTL`
/// (animation control) chunk appears before the first `IDAT` (image data) chunk — the APNG spec's own
/// definition of "animated", and the exact check pi's `isAnimatedPng` (`mime.ts`) makes. A static PNG
/// carrying unrelated ancillary chunks (`tEXt`, `iCCP`, …) between `IHDR` and `IDAT` correctly returns
/// `false`; only `acTL`'s presence — always required to appear before `IDAT` per the spec — signals
/// real animation. Returns `false` (not animated) on any malformed/truncated/oversized-beyond-`buf`
/// chunk stream, the same conservative default pi's version uses: a corrupt file falls through to the
/// ordinary sniff-failed/decode-failed handling instead of being silently misclassified as animated.
fn is_animated_png(buf: &[u8]) -> bool {
    const PNG_SIGNATURE_LEN: usize = 8;
    let mut offset = PNG_SIGNATURE_LEN;
    // Each chunk is a 4-byte big-endian length, a 4-byte ASCII type, `length` bytes of data, and a
    // trailing 4-byte CRC — walk the stream one chunk at a time without copying any of it.
    while offset + 8 <= buf.len() {
        let chunk_len = u32::from_be_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]) as usize;
        let chunk_type = &buf[offset + 4..offset + 8];
        if chunk_type == b"acTL" {
            return true;
        }
        if chunk_type == b"IDAT" {
            return false; // acTL must precede the first IDAT per the APNG spec — none seen means static.
        }
        // `saturating_add`, not plain `+`: `chunk_len` comes straight off untrusted file bytes and this
        // crate runs with `overflow-checks = true` in release — an adversarial/corrupt length must
        // degrade to "not animated" below, never panic.
        let next_offset = offset
            .saturating_add(8)
            .saturating_add(chunk_len)
            .saturating_add(4);
        if next_offset <= offset || next_offset > buf.len() {
            return false; // malformed/truncated chunk, or acTL simply isn't within our sniff window
        }
        offset = next_offset;
    }
    false
}

/// Whether `buf` (a "BM"-prefixed file's leading bytes) is a *structurally plausible* BMP, not just one
/// starting with the right 2-byte ASCII signature — matching pi's `isBmp` (`mime.ts:57-81`). Requires a
/// declared file size of at least 26 bytes (or `0`, some encoders' "unknown" sentinel), a pixel-data
/// offset that lands at or after the end of the file+DIB headers and strictly before the declared end
/// of file, and a DIB header (`BITMAPINFOHEADER`-family, size 12 or 40-124) reporting exactly one color
/// plane and a bits-per-pixel value real BMP encoders actually use (1/4/8/16/24/32). `buf` is only ever
/// the sniff-window prefix, not necessarily the whole file, so out-of-range field reads degrade to `0`
/// (via [`read_u16_le`]/[`read_u32_le`]) rather than panicking — same as pi's own `?? 0` fallback.
fn is_valid_bmp(buf: &[u8]) -> bool {
    if buf.len() < 26 {
        return false;
    }
    let declared_file_size = read_u32_le(buf, 2);
    let pixel_data_offset = read_u32_le(buf, 10);
    let dib_header_size = read_u32_le(buf, 14);
    if declared_file_size != 0 && declared_file_size < 26 {
        return false;
    }
    if pixel_data_offset < 14u32.saturating_add(dib_header_size) {
        return false;
    }
    if declared_file_size != 0 && pixel_data_offset >= declared_file_size {
        return false;
    }
    let (color_planes, bits_per_pixel) = if dib_header_size == 12 {
        (read_u16_le(buf, 22), read_u16_le(buf, 24))
    } else if (40..=124).contains(&dib_header_size) {
        if buf.len() < 30 {
            return false;
        }
        (read_u16_le(buf, 26), read_u16_le(buf, 28))
    } else {
        return false;
    };
    color_planes == 1 && matches!(bits_per_pixel, 1 | 4 | 8 | 16 | 24 | 32)
}

/// Little-endian `u16` at `offset`, or `0` for any byte that falls outside `buf` — mirrors pi's
/// `readUint16LE`'s `?? 0` fallback so a truncated sniff window degrades to "not a valid BMP" rather
/// than panicking.
fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    let lo = *buf.get(offset).unwrap_or(&0) as u16;
    let hi = *buf.get(offset + 1).unwrap_or(&0) as u16;
    lo | (hi << 8)
}

/// Little-endian `u32` at `offset`, with the same out-of-range-degrades-to-`0` behavior as
/// [`read_u16_le`].
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    (0..4u32).fold(0u32, |acc, i| {
        acc | (u32::from(*buf.get(offset + i as usize).unwrap_or(&0)) << (8 * i))
    })
}

/// The image format implied by a path's extension, or `None` if it isn't a recognized one. Used only
/// as a fallback when [`sniff_image_format`] can't identify a format at all (a truncated/corrupt file);
/// the actual format sent on the wire comes from sniffing the real magic bytes (see [`read_image`]).
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

/// pi's `conversionHint` (`image-process.ts`) — appended to the read text whenever [`read_image`] had
/// to normalize a BMP into a supported format before it could be sent to the model, so the transcript
/// states *why* the returned media type/bytes differ from the file's real, on-disk format. The model
/// still receives a correctly-converted image either way — this is purely a transcript/explanation gap
/// otherwise. `to` is the *actual* media type finally sent, which can differ from the immediate
/// post-BMP-conversion PNG when the resize path later re-encodes further to JPEG to fit the byte
/// budget — matching pi's own `conversionHint(normalized.convertedFrom, <final mime type>)` exactly.
fn bmp_conversion_note(converted_from_bmp: bool, to: &str) -> String {
    if converted_from_bmp {
        format!(" [Image converted from image/bmp to {to}.]")
    } else {
        String::new()
    }
}

/// Read an image file and return it as a base64 [`ImageSource`] attachment (plus a short text note), so
/// the multimodal model can actually see it. `ext_format` is only the fallback used when magic-byte
/// sniffing can't identify the real format (a truncated or corrupt header); the sniffed format is
/// preferred whenever it succeeds. An image that already fits [`MAX_IMAGE_BASE64_BYTES`] is sent as its
/// original bytes, unmodified; an oversized one is downscaled/re-encoded by [`resize_image`] — unless
/// `auto_resize` is `false` (pi's `ImageSettings.autoResize`), in which case the resize path is skipped
/// entirely and the normalized bytes ship as-is regardless of size, matching pi's `processImage`, which
/// never calls `resizeImage` in that case either. If the resize path is taken but still can't fit the
/// budget, or a BMP fails to convert to a supported format, this returns `Ok` with a text-only "[Image
/// omitted: …]" result rather than an `Err` — matching pi's `processImage`, which treats both as a
/// normal (if disappointing) outcome for the model to see, not a failed tool call.
fn read_image(
    path: &str,
    ext_format: image::ImageFormat,
    auto_resize: bool,
) -> Result<ToolOutput, ToolError> {
    let bytes =
        std::fs::read(path).map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
    let format = image::guess_format(&bytes).unwrap_or(ext_format);

    // Most vision APIs (Anthropic: png/jpeg/gif/webp only) reject BMP outright. Convert it losslessly
    // to PNG up front — matching pi's `normalizeSupportedImageMimeType`, which always converts BMP
    // before it can reach the model — rather than sending a media type the provider will 400 on. A
    // failed conversion falls through with the original bytes/format; the caller still gets a
    // best-effort attachment rather than a hard error over a format quirk.
    let (bytes, format, converted_from_bmp) = if format == image::ImageFormat::Bmp {
        match convert_to_png(&bytes, format) {
            Some(png) => (png, image::ImageFormat::Png, true),
            // pi: `image-process.ts`'s `processImage`, when `normalizeImage` fails — a text-only,
            // *successful* result (`ok: false` with a message, not a thrown/rejected error), so the
            // model sees an explanation rather than the tool call itself reading as failed. `read.ts`
            // wraps that into a plain `tool_result` text block, no error, no image; mirrored here by
            // returning `Ok` instead of `Err`.
            None => {
                return Ok(ToolOutput::text(
                    "[Image omitted: could not be converted to a supported inline image format.]",
                ));
            }
        }
    } else {
        (bytes, format, false)
    };
    let media_type = media_type_of(format);

    if !auto_resize {
        // pi: `image-process.ts`'s `processImage`, `autoResizeImages === false` branch — `resizeImage`
        // (and the decode-validation that lives inside it, further down) is never called at all; the
        // normalized bytes ship as-is, whatever their size or pixel dimensions.
        let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
        return Ok(ToolOutput::image(
            format!(
                "Read image {path} ({media_type}, {} bytes).{}",
                bytes.len(),
                bmp_conversion_note(converted_from_bmp, media_type)
            ),
            ImageSource::base64(media_type, data),
        ));
    }

    if base64_len(bytes.len()) <= MAX_IMAGE_BASE64_BYTES as usize {
        // Magic-byte sniffing only confirms the file *starts like* an image; a truncated download, a
        // bit-rotted file, or a crafted polyglot could still fail to actually decode. Validate before
        // sending it to the model as one — matching pi's `resizeImageInProcess`, which always decodes
        // first (even on its own already-fits fast path) rather than shipping raw bytes through
        // unchecked. The original bytes are still sent unmodified on success — this only rules out
        // sending something that isn't really a decodable image.
        let decoded = match image::load_from_memory_with_format(&bytes, format) {
            Ok(img) => img,
            Err(e) => {
                return Err(ToolError::InvalidInput(format!(
                    "image {path} does not decode as a valid {media_type} file: {e}"
                )));
            }
        };
        // The fast path only applies when *both* the byte size and the pixel dimensions are within
        // budget — matching pi, which checks `originalWidth <= maxWidth && originalHeight <= maxHeight
        // && inputBase64Size < maxBytes` together. A large-dimension image that happens to compress to
        // a small file (e.g. a huge flat-color PNG) must still be downscaled: sending it unresized would
        // exceed `MAX_IMAGE_DIMENSION` even though it fits the byte budget. An image within the byte
        // budget but over-dimensioned falls through to `resize_image` below instead of returning here.
        if decoded.width() <= MAX_IMAGE_DIMENSION && decoded.height() <= MAX_IMAGE_DIMENSION {
            let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
            return Ok(ToolOutput::image(
                format!(
                    "Read image {path} ({media_type}, {} bytes).{}",
                    bytes.len(),
                    bmp_conversion_note(converted_from_bmp, media_type)
                ),
                ImageSource::base64(media_type, data),
            ));
        }
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
            // Aspect-ratio-preserving resize (`image::resize`, not `resize_exact`) scales both
            // dimensions by the same factor, so either ratio works; width matches pi's own
            // `formatDimensionNote` (`originalWidth / width`) exactly. Precomputing and stating the
            // multiplier — not just the two dimension pairs — saves the model from re-deriving it
            // (and from a division-order mistake) whenever it needs to map a point it identified in
            // the downscaled image back to the original's coordinate space (e.g. a click target).
            let scale = f64::from(orig_w) / f64::from(new_w);
            Ok(ToolOutput::image(
                format!(
                    "Read image {path} ({media_type}, {} bytes); resized from {orig_w}x{orig_h} to \
                     {new_w}x{new_h} ({resized_media_type}) to fit within the inline size budget. \
                     Multiply coordinates by {scale:.2} to map to the original image.{}",
                    bytes.len(),
                    bmp_conversion_note(converted_from_bmp, resized_media_type)
                ),
                ImageSource::base64(resized_media_type, resized.base64_data),
            ))
        }
        // pi: `image-process.ts`'s `processImage`, when `resizeImage` returns `null` — same graceful,
        // *successful* text-only result as the BMP-conversion-failure case above (`ok: false` with a
        // message, not a thrown error): the model needs to know the image couldn't be inlined, but the
        // tool call itself didn't fail.
        None => Ok(ToolOutput::text(
            "[Image omitted: could not be resized below the inline image size limit.]",
        )),
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
/// for no reason; only when the PNG doesn't fit the budget does it fall back to a ladder of JPEG
/// qualities at that same size — `[requested, 85, 70, 55, 40]`, matching pi's own fixed ladder — before
/// shrinking further. If nothing at this size fits, dimensions are cut by 25% (pi's own factor, not
/// 20%) and it retries, with **no round limit** and a floor of 1×1 pixel (both matching pi's
/// `resizeImageInProcess` exactly, instead of a 6-round cap and a 64×64 floor): shrinking always
/// eventually reaches a size small enough for *some* encoding to fit under any sane byte budget, so
/// giving up after a fixed number of rounds — and reporting the image as unservable — was the actual
/// bug; pi's version is "guaranteed to eventually fit" by construction, and this now is too. `None` is
/// only returned once the 1×1 floor itself still doesn't fit — a budget so small no image could ever
/// satisfy it.
///
/// The PNG re-encode preserves an alpha channel when the source image actually carries one (a
/// transparent PNG needing this resize path), matching pi's Photon-based pipeline, which operates on
/// RGBA internally with no flattening step of its own — see the `has_alpha` branch inline below. The
/// JPEG fallback ladder has no such option: JPEG carries no alpha channel at all, so a transparent
/// image that must fall all the way to JPEG unavoidably loses transparency there, same as pi.
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
    // JPEG qualities tried at every size, highest first — pi's own fixed ladder. The caller's
    // requested quality leads it; trying the rest before shrinking further means a lower-quality
    // encode at the *current* size is always preferred over a higher-quality encode at a smaller one.
    let jpeg_quality_ladder = [jpeg_quality, 85, 70, 55, 40];
    loop {
        let scaled = if width < orig_dims.0 || height < orig_dims.1 {
            img.resize(width, height, image::imageops::FilterType::Lanczos3)
        } else {
            img.clone()
        };
        // Whether *this* decoded image actually carries an alpha channel (Rgba8/LumaA8) — most source
        // images (JPEG/BMP/opaque PNG) don't, so this only changes behavior for a genuinely
        // transparent image. PNG can carry alpha losslessly, so the PNG re-encode below preserves it
        // via `to_rgba8()`/`Rgba8` instead of unconditionally flattening through `to_rgb8()`, which
        // silently dropped the channel entirely (not "flatten onto a background" — just discarded, so
        // a half-transparent pixel came back fully opaque) — matching pi's Photon-based pipeline, which
        // operates on RGBA internally with no flattening step before its own PNG re-encode. JPEG has no
        // alpha channel at all, so that ladder below still flattens via `to_rgb8()` regardless — that
        // part is unavoidable and was already correct.
        let has_alpha = scaled.color().has_alpha();

        let mut png_buf = Vec::new();
        let png_ok = if has_alpha {
            let rgba = scaled.to_rgba8();
            image::codecs::png::PngEncoder::new(&mut png_buf)
                .write_image(
                    rgba.as_raw(),
                    scaled.width(),
                    scaled.height(),
                    image::ExtendedColorType::Rgba8,
                )
                .is_ok()
        } else {
            image::codecs::png::PngEncoder::new(&mut png_buf)
                .write_image(
                    scaled.to_rgb8().as_raw(),
                    scaled.width(),
                    scaled.height(),
                    image::ExtendedColorType::Rgb8,
                )
                .is_ok()
        };
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

        // JPEG has no alpha channel at all — flatten via `to_rgb8()` regardless of `has_alpha`. This
        // is the one place a transparent source image genuinely loses its alpha, same as pi (which
        // hits the identical constraint whenever its own PNG re-encode doesn't fit the budget either).
        let rgb = scaled.to_rgb8();
        for &quality in &jpeg_quality_ladder {
            let mut jpeg_buf = Vec::new();
            let encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_buf, quality);
            if encoder
                .write_image(
                    rgb.as_raw(),
                    scaled.width(),
                    scaled.height(),
                    image::ExtendedColorType::Rgb8,
                )
                .is_err()
            {
                continue;
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
        }

        // Nothing fit at this size. Already at the 1×1 floor with nowhere left to shrink → genuinely
        // can't be served inline under this budget.
        if width <= 1 && height <= 1 {
            return None;
        }
        width = (((width as f64) * 0.75) as u32).max(1);
        height = (((height as f64) * 0.75) as u32).max(1);
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
        let out = Read::default()
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
        let err = Read::default()
            .run(json!({ "path": f.path().to_str().unwrap(), "offset": 99 }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn line_count_is_the_same_whether_or_not_the_file_ends_with_a_trailing_newline() {
        // A deliberate divergence from pi's own `read.ts`, which counts
        // `textContent.split("\n").length` — off by one whenever the file ends with a newline (JS's
        // `split` yields a trailing empty-string element for it). Ours must report the same "3 lines
        // total" for `"a\nb\nc\n"` and `"a\nb\nc"` — both really are 3 lines.
        let with_trailing = tmp_file("a\nb\nc\n");
        let without_trailing = tmp_file("a\nb\nc");

        let err_with = Read::default()
            .run(json!({ "path": with_trailing.path().to_str().unwrap(), "offset": 99 }))
            .await
            .unwrap_err();
        let err_without = Read::default()
            .run(json!({ "path": without_trailing.path().to_str().unwrap(), "offset": 99 }))
            .await
            .unwrap_err();
        let msg_with = match err_with {
            ToolError::InvalidInput(m) => m,
            other => panic!("expected InvalidInput, got {other:?}"),
        };
        let msg_without = match err_without {
            ToolError::InvalidInput(m) => m,
            other => panic!("expected InvalidInput, got {other:?}"),
        };
        assert!(msg_with.contains("3 lines total"), "got: {msg_with}");
        assert!(msg_without.contains("3 lines total"), "got: {msg_without}");
    }

    #[tokio::test]
    async fn truncation_message_reports_the_true_total_line_count() {
        // Task 50 (pi-parity fix): pi's own truncation message (`read.ts`) is "Showing lines X-Y of
        // {totalFileLines}" — the total was previously missing entirely here, an information-loss
        // gap the model had no way to work around (was truncation 1 line short of the end, or 10,000
        // lines short?). The total must reflect the *whole* file, not just what was shown or what
        // was left after an `offset` — proven here with an offset partway into a file bigger than
        // what's shown.
        let body = (1..=50).map(|i| i.to_string()).collect::<Vec<_>>().join("\n");
        let f = tmp_file(&format!("{body}\n")); // 50 lines total
        let out = Read::default()
            .run(json!({ "path": f.path().to_str().unwrap(), "offset": 10, "limit": 5 }))
            .await
            .unwrap()
            .text;
        // Shown lines 10-14 (5 lines starting at offset 10); the file has 50 lines total regardless
        // of where the window started or stopped.
        assert!(
            out.contains("[showing lines 10-14 of 50 total; use offset=15 to continue]"),
            "got: {out}"
        );
    }

    #[tokio::test]
    async fn truncation_message_total_is_unaffected_by_a_trailing_newline() {
        // The total in the truncation marker must follow the same trailing-newline-independent
        // "real content lines" convention as every other "N lines total" this tool reports (see
        // `line_count_is_the_same_whether_or_not_the_file_ends_with_a_trailing_newline` above) — not
        // a raw `\n` byte count over the undrained remainder, which would silently be one *higher*
        // than the file's real line count whenever it ends with a newline.
        let with_trailing = tmp_file("a\nb\nc\nd\ne\n");
        let without_trailing = tmp_file("a\nb\nc\nd\ne");

        let out_with = Read::default()
            .run(json!({ "path": with_trailing.path().to_str().unwrap(), "limit": 2 }))
            .await
            .unwrap()
            .text;
        let out_without = Read::default()
            .run(json!({ "path": without_trailing.path().to_str().unwrap(), "limit": 2 }))
            .await
            .unwrap()
            .text;
        assert!(out_with.contains("of 5 total"), "got: {out_with}");
        assert!(out_without.contains("of 5 total"), "got: {out_without}");
    }

    #[tokio::test]
    async fn truncation_reports_next_offset() {
        let f = tmp_file("a\nb\nc\nd\ne\n");
        let out = Read::default()
            .run(json!({ "path": f.path().to_str().unwrap(), "limit": 2 }))
            .await
            .unwrap()
            .text;
        // Showed lines 1-2 of the file's 5 total lines; the model is told to continue at line 3.
        assert!(
            out.contains("[showing lines 1-2 of 5 total; use offset=3 to continue]"),
            "got: {out}"
        );
    }

    #[tokio::test]
    async fn no_limit_argument_defaults_to_a_2000_line_cap() {
        // Every other truncation test passes an explicit `limit`; this proves the actual default
        // (`DEFAULT_LIMIT`) kicks in when the argument is omitted entirely.
        let body = (1..=2500)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let f = tmp_file(&format!("{body}\n"));
        let out = Read::default()
            .run(json!({ "path": f.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("  2000\t2000"), "line 2000 must be shown");
        assert!(!out.contains("\t2001\n"), "line 2001 must not be shown");
        assert!(
            out.contains("use offset=2001 to continue"),
            "got: {}",
            &out[out.len().saturating_sub(200)..]
        );
    }

    #[tokio::test]
    async fn honors_offset_and_limit() {
        let f = tmp_file("a\nb\nc\nd\ne\n");
        let out = Read::default()
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
        let err = Read::default()
            .run(json!({ "path": "/no/such/file/xyz" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn falls_back_to_the_nfd_form_of_a_path_that_does_not_exist_as_nfc() {
        // pi: path-utils.test.ts, `resolveReadPath` NFD fallback. The file on disk is named with a
        // combining accent (NFD: "e" + U+0301), the way macOS stores filenames — the caller passes the
        // precomposed NFC form ("é", a single codepoint), which doesn't byte-match the real filename.
        let dir = tempfile::tempdir().unwrap();
        let nfd_name = "cafe\u{301}.txt"; // NFD: e + combining acute accent
        std::fs::write(dir.path().join(nfd_name), "content").unwrap();

        let nfc_path = dir.path().join("caf\u{e9}.txt"); // NFC: precomposed é
        let out = Read::default()
            .run(json!({ "path": nfc_path.to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("content"), "got: {out}");
    }

    #[tokio::test]
    async fn falls_back_to_a_curly_quote_variant_of_a_path_that_does_not_exist_with_a_straight_quote()
     {
        // pi: path-utils.test.ts, `resolveReadPath` curly-quote fallback — macOS uses U+2019 in some
        // generated filenames; a user/model typing a straight apostrophe should still find the file.
        let dir = tempfile::tempdir().unwrap();
        let curly_name = "don\u{2019}t.txt"; // U+2019 right single quotation mark
        std::fs::write(dir.path().join(curly_name), "content").unwrap();

        let straight_path = dir.path().join("don't.txt"); // U+0027 straight apostrophe
        let out = Read::default()
            .run(json!({ "path": straight_path.to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("content"), "got: {out}");
    }

    #[tokio::test]
    async fn a_genuinely_missing_file_still_errors_after_exhausting_fallbacks() {
        // None of the Unicode-normalization fallbacks should turn a truly nonexistent file into a
        // false positive.
        let err = Read::default()
            .run(json!({ "path": "/no/such/file/xyz-caf\u{e9}-don't.txt" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path`, via its `@`-prefix-strip behavior
        // (needs no `$HOME` mutation — see `expand_tilde`'s own direct unit tests for that half).
        let f = tmp_file("alpha\n");
        let at_prefixed = format!("@{}", f.path().to_str().unwrap());
        let out = Read::default().run(json!({ "path": at_prefixed })).await.unwrap().text;
        assert!(out.contains("alpha"));
    }

    #[tokio::test]
    async fn output_byte_budget_matches_the_shared_50kb_cap_not_a_bespoke_larger_one() {
        // `read`'s aggregate output budget used to be a bespoke 256KB constant, 5x pi's (and this
        // crate's other tools': grep/find/ls all share `output::DEFAULT_MAX_BYTES`, 50KB) uniform
        // budget. Enough short lines to blow well past 50KB but stay far under 256KB proves the smaller
        // cap is actually the one enforced now, not vestigial dead code.
        let line = "x".repeat(100); // + line number prefix + newline, comfortably short per line
        let lines: Vec<String> = (0..1200).map(|_| line.clone()).collect(); // ~120KB of body
        let f = tmp_file(&format!("{}\n", lines.join("\n")));
        let out = Read::default()
            .run(json!({ "path": f.path().to_str().unwrap(), "limit": 100_000 }))
            .await
            .unwrap()
            .text;
        assert!(
            out.len() < 150_000,
            "expected truncation near the 50KB budget, got {} bytes",
            out.len()
        );
        assert!(
            out.contains("use offset="),
            "must report a continuation offset: {}",
            &out[out.len().saturating_sub(200)..]
        );
    }

    #[tokio::test]
    async fn caps_a_pathological_single_line() {
        // One line far larger than the per-line cap, followed by a normal line: the giant line is
        // clipped (with a marker) and the file's line structure is still tracked past it.
        let huge = "x".repeat(MAX_LINE_BYTES * 4);
        let f = tmp_file(&format!("{huge}\nnext\n"));
        let out = Read::default()
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
        let out = Read::default()
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
        let out = Read::default()
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

    #[test]
    fn resize_image_converges_on_a_high_entropy_incompressible_image() {
        // A deliberately high-entropy, incompressible synthetic image (per-pixel pseudo-random
        // noise): PNG can't compress noise meaningfully, and JPEG needs a low quality (and/or a much
        // smaller size) to shrink it much at all. Before this fix, `resize_image` capped out at 6
        // shrink rounds with a 64x64 floor and a single JPEG quality per round — nowhere near enough
        // for noise at a tight byte budget — and gave up with `None`. With no round limit, a 1x1
        // floor, and a ladder of JPEG qualities per round (matching pi's `resizeImageInProcess`),
        // shrinking must always eventually reach a size that fits.
        let (w, h) = (512, 512);
        let mut img = image::RgbImage::new(w, h);
        // Deterministic xorshift32 PRNG — no external RNG dependency needed for a reproducible test.
        let mut state: u32 = 0x1234_5678;
        let mut next_byte = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFF) as u8
        };
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([next_byte(), next_byte(), next_byte()]);
        }
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        // A byte budget the un-shrunk (or lightly-shrunk) noise image can't possibly meet, forcing
        // many rounds of shrinking before anything fits.
        let tiny_budget = 2000;
        let resized = resize_image(&png_bytes, image::ImageFormat::Png, w, h, tiny_budget, 80)
            .expect("resize must converge to a fit instead of giving up");

        assert!(
            resized.base64_data.len() <= tiny_budget,
            "resized payload ({} bytes) exceeds the budget ({tiny_budget})",
            resized.base64_data.len()
        );
    }

    #[tokio::test]
    async fn reads_an_image_as_an_attachment() {
        // A .png is returned as a base64 image attachment, not UTF-8-decoded into garbage text. Must
        // be a genuinely decodable PNG, not just the 8-byte magic-number signature — `read_image`
        // validates the bytes actually decode before sending them as an attachment.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        std::fs::write(&path, &png_bytes).unwrap();
        let out = Read::default()
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
    async fn a_small_file_with_valid_magic_bytes_but_no_real_image_body_is_rejected() {
        // Magic-byte sniffing only looks at the leading signature — a truncated download, bit-rotted
        // file, or crafted polyglot can pass that check without being a real, decodable image. Before
        // this fix, a small-enough-to-skip-resizing file like this would be base64'd and sent to the
        // model completely unvalidated.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.png");
        // The real 8-byte PNG signature, but nothing resembling an actual PNG chunk stream after it.
        std::fs::write(&path, [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]).unwrap();
        let err = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(_)),
            "a non-decodable image must be a clear tool error, not silently forwarded: {err:?}"
        );
    }

    #[tokio::test]
    async fn read_falls_back_to_text_when_extension_implies_image_but_content_is_not_image_bytes() {
        // pi: tools.test.ts, "should treat files with image extension but non-image content as text".
        // No magic-byte match at all here (unlike the sniffed-but-corrupt case above) — the extension
        // is the *only* signal this might be an image, so a failed decode should fall back to a plain
        // text read (e.g. a Git LFS pointer file saved as `.png`) rather than a hard error.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-an-image.png");
        std::fs::write(&path, "definitely not a png").unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(
            out.text.contains("definitely not a png"),
            "got: {}",
            out.text
        );
        assert!(
            out.images.is_empty(),
            "must not be treated as an image attachment"
        );
    }

    /// The 8-byte PNG magic signature every PNG (animated or not) starts with.
    fn png_signature() -> Vec<u8> {
        vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
    }

    /// Build one PNG chunk: 4-byte big-endian length, 4-byte ASCII type, `data` verbatim, and a
    /// (deliberately bogus, never checked by `is_animated_png` or `image::guess_format`) 4-byte CRC.
    fn png_chunk(chunk_type: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(&(data.len() as u32).to_be_bytes());
        c.extend_from_slice(chunk_type);
        c.extend_from_slice(data);
        c.extend_from_slice(&[0, 0, 0, 0]);
        c
    }

    /// Splice a synthetic, minimal `acTL` (animation control) chunk right after a real PNG's `IHDR`
    /// chunk — the same relative position a real APNG encoder places it (`acTL` always precedes the
    /// first `IDAT` per the APNG spec). Its data doesn't need to be semantically meaningful
    /// (`num_frames`/`num_plays`) for detection purposes; only the chunk's presence, type, and correct
    /// length prefix matter to `is_animated_png`'s chunk walk — this is deliberately the *minimal*
    /// fixture that makes a file "an animated PNG" by the same definition `is_animated_png` itself uses.
    fn splice_actl_after_ihdr(png_bytes: &[u8]) -> Vec<u8> {
        let ihdr_len = u32::from_be_bytes(png_bytes[8..12].try_into().unwrap()) as usize;
        let ihdr_end = 8 + 8 + ihdr_len + 4; // signature + (len+type) + data + crc
        let mut out = png_bytes[..ihdr_end].to_vec();
        out.extend(png_chunk(b"acTL", &[0u8; 8])); // num_frames=0, num_plays=0 — content unchecked
        out.extend_from_slice(&png_bytes[ihdr_end..]);
        out
    }

    #[test]
    fn is_animated_png_detects_an_actl_chunk_before_the_first_idat() {
        let mut buf = png_signature();
        buf.extend(png_chunk(b"IHDR", &[0u8; 13]));
        buf.extend(png_chunk(b"acTL", &[0u8; 8]));
        buf.extend(png_chunk(b"IDAT", &[0u8; 4]));
        assert!(is_animated_png(&buf));
    }

    #[test]
    fn is_animated_png_is_false_for_a_static_png_with_ancillary_chunks_before_idat() {
        // A real-world static PNG routinely carries chunks (`tEXt`, `iCCP`, `pHYs`, …) between `IHDR`
        // and `IDAT` — only `acTL`'s actual presence signals animation, not "something sits before
        // IDAT".
        let mut buf = png_signature();
        buf.extend(png_chunk(b"IHDR", &[0u8; 13]));
        buf.extend(png_chunk(b"tEXt", b"comment"));
        buf.extend(png_chunk(b"IDAT", &[0u8; 4]));
        assert!(!is_animated_png(&buf));
    }

    #[test]
    fn is_animated_png_is_false_when_idat_arrives_with_no_actl_at_all() {
        let mut buf = png_signature();
        buf.extend(png_chunk(b"IHDR", &[0u8; 13]));
        buf.extend(png_chunk(b"IDAT", &[0u8; 4]));
        assert!(!is_animated_png(&buf));
    }

    #[test]
    fn is_animated_png_is_false_on_a_truncated_chunk_stream() {
        // A chunk header claiming more data than the buffer actually holds (or a stream cut off
        // mid-chunk) must degrade to "not animated", not panic or loop — the same conservative default
        // pi's `isAnimatedPng` uses on malformed input.
        let mut buf = png_signature();
        buf.extend(png_chunk(b"IHDR", &[0u8; 13]));
        buf.truncate(buf.len() - 2); // cut mid-chunk, past the length+type but short on data/CRC
        assert!(!is_animated_png(&buf));
    }

    #[tokio::test]
    async fn an_animated_png_falls_back_to_text_instead_of_being_sent_as_a_single_static_frame() {
        // pi: `detectSupportedImageMimeType` (`mime.ts`) returns `null` for an animated PNG
        // (`isAnimatedPng`), so `read.ts` never even attempts to decode/serve it as an image at all —
        // sending a vision model one static frame of an animation, unlabeled as such, would misrepresent
        // what the file actually contains. We port the same detect-and-exclude behavior
        // (`Sniff::AnimatedPng`) rather than silently shipping the `image` crate's default/first-frame
        // decode as if it were the complete picture — deliberately *not* treated as "just decode the
        // first frame and go", the way plain APNG-as-first-frame would read.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let apng_bytes = splice_actl_after_ihdr(&png_bytes);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("animated.png");
        std::fs::write(&path, &apng_bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(
            out.images.is_empty(),
            "an animated PNG must not be sent as a single-frame image attachment"
        );
        assert!(
            !out.text.contains("Read image"),
            "must not be reported as an image read at all: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn a_large_dimension_but_small_byte_size_image_is_still_downscaled() {
        // A flat color compresses so well under PNG that the file is tiny even at a huge resolution —
        // the byte-size fast path alone would wrongly ship it unresized, exceeding MAX_IMAGE_DIMENSION.
        // The fast path must also check dimensions, not just base64 byte size (pi: `resizeImageInProcess`
        // checks `originalWidth <= maxWidth && originalHeight <= maxHeight && inputBase64Size < maxBytes`
        // together).
        let width = MAX_IMAGE_DIMENSION + 500;
        let height = MAX_IMAGE_DIMENSION + 500;
        let img = image::RgbImage::from_pixel(width, height, image::Rgb([10, 20, 30]));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge_flat.png");
        image::DynamicImage::ImageRgb8(img)
            .save_with_format(&path, image::ImageFormat::Png)
            .unwrap();
        assert!(
            base64_len(std::fs::metadata(&path).unwrap().len() as usize)
                <= MAX_IMAGE_BASE64_BYTES as usize,
            "fixture must fit the byte budget as-is, so only the dimension check can catch it"
        );

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        let decoded = image::load_from_memory(
            &base64::engine::general_purpose::STANDARD
                .decode(&out.images[0].data)
                .unwrap(),
        )
        .unwrap();
        assert!(
            decoded.width() <= MAX_IMAGE_DIMENSION && decoded.height() <= MAX_IMAGE_DIMENSION,
            "must be downscaled despite fitting the byte budget: {}x{}",
            decoded.width(),
            decoded.height()
        );
        assert!(out.text.contains("resized from"), "got: {}", out.text);
    }

    #[tokio::test]
    async fn resize_note_states_the_exact_coordinate_scale_multiplier() {
        // A clean 2x-oversized square: MAX_IMAGE_DIMENSION * 2 on each side downscales to exactly
        // MAX_IMAGE_DIMENSION, so `orig_w / new_w` is exactly 2.00 — a value precise enough to catch a
        // wrong-direction division (e.g. `new_w / orig_w`, which would print "0.50" instead).
        let side = MAX_IMAGE_DIMENSION * 2;
        let img = image::RgbImage::from_pixel(side, side, image::Rgb([10, 20, 30]));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge_flat_square.png");
        image::DynamicImage::ImageRgb8(img)
            .save_with_format(&path, image::ImageFormat::Png)
            .unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(
            out.text
                .contains("Multiply coordinates by 2.00 to map to the original image"),
            "got: {}",
            out.text
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

        let out = Read::default()
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
        // LOW pi-parity gap (fixed): pi's `formatDimensionNote` precomputes and states the coordinate
        // multiplier the model needs to map a point in the downscaled image back to the original — not
        // just the two dimension pairs, which would leave the model to (maybe incorrectly) re-derive it.
        assert!(
            out.text.contains("Multiply coordinates by")
                && out.text.contains("to map to the original image"),
            "the resize note must state the coordinate-scale multiplier: {}",
            out.text
        );
        // The BMP-to-PNG conversion hint must appear on the resize path too, not just the fast path
        // `small_bmp_is_converted_to_png_not_sent_as_bmp` covers.
        assert!(
            out.text
                .contains("[Image converted from image/bmp to image/png.]"),
            "the BMP-to-PNG conversion must be noted even when the resize path is also taken: {}",
            out.text
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn image_decode_and_resize_does_not_block_the_async_runtime() {
        // On a `current_thread` runtime there is exactly one thread driving async tasks. If
        // `read_image`'s CPU-bound decode/resize/encode ran inline instead of on `spawn_blocking`'s
        // separate pool, it would monopolize that one thread and the concurrent ticker task below
        // could make *zero* progress until the read finished — deterministic, not a timing threshold.
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

        let ticks = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let ticks_task = ticks.clone();
        let ticker = tokio::spawn(async move {
            loop {
                ticks_task.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        });

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        ticker.abort();

        assert_eq!(out.images.len(), 1, "the read must still succeed");
        assert!(
            ticks.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "a concurrent async task must make progress while image decode/resize is running"
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

        let out = Read::default()
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
        // pi-parity gap (fixed): pi's `image-process.ts` appends a "[Image converted from image/bmp
        // to image/png.]" hint to the tool result's text whenever a BMP had to be normalized — the
        // model still receives a correctly-converted image either way, but the transcript previously
        // never explained why the reported media_type/bytes differ from the file's real, on-disk BMP.
        assert!(
            out.text
                .contains("[Image converted from image/bmp to image/png.]"),
            "the BMP-to-PNG conversion must be noted in the text: {}",
            out.text
        );
    }

    /// A noise-filled PNG comfortably past both `MAX_IMAGE_DIMENSION` and `MAX_IMAGE_BASE64_BYTES` —
    /// PNG can't compress incompressible noise, so this genuinely forces the resize path (or, with
    /// auto-resize off, would exceed both budgets) rather than accidentally fitting as-is.
    fn oversized_noise_png() -> Vec<u8> {
        let width = MAX_IMAGE_DIMENSION + 800;
        let height = 600;
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            let h = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)).wrapping_add(0x9e3779b9);
            image::Rgb([(h & 0xff) as u8, ((h >> 8) & 0xff) as u8, ((h >> 16) & 0xff) as u8])
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        bytes
    }

    #[tokio::test]
    async fn default_read_still_downscales_an_oversized_image_auto_resize_on_by_default() {
        // pi-parity feature (Task #4): `image_auto_resize` defaults to `true` (pi's own
        // `ImageSettings.autoResize` default) — existing behavior must be unchanged for a plain
        // `Read::default()`, the construction every registry/CLI path already used before this field
        // existed.
        let png_bytes = oversized_noise_png();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized.png");
        std::fs::write(&path, &png_bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        let decoded = image::load_from_memory(
            &base64::engine::general_purpose::STANDARD
                .decode(&out.images[0].data)
                .unwrap(),
        )
        .unwrap();
        assert!(
            decoded.width() <= MAX_IMAGE_DIMENSION,
            "auto-resize-on must still downscale an oversized image: got width {}",
            decoded.width()
        );
        assert!(out.text.contains("resized from"), "got: {}", out.text);
    }

    #[tokio::test]
    async fn image_auto_resize_off_ships_an_oversized_image_without_downscaling_it() {
        // pi-parity feature (Task #4): pi's `ImageSettings.autoResize` (`processImage`'s
        // `autoResizeImages === false` branch) skips `resizeImage` entirely when disabled — an
        // oversized image must pass through unresized, dimensions and all, rather than being
        // downscaled/re-encoded to fit the inline budget.
        let png_bytes = oversized_noise_png();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized.png");
        std::fs::write(&path, &png_bytes).unwrap();

        let out = Read::default()
            .with_image_auto_resize(false)
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(&out.images[0].data)
            .unwrap();
        assert_eq!(
            decoded_bytes, png_bytes,
            "with auto-resize off, the original (post-normalization) bytes must ship unmodified"
        );
        let decoded = image::load_from_memory(&decoded_bytes).unwrap();
        assert!(
            decoded.width() > MAX_IMAGE_DIMENSION,
            "the image must not be downscaled when auto-resize is off: got width {}",
            decoded.width()
        );
        assert!(
            !out.text.contains("resized from"),
            "no resize note when auto-resize is off: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn image_auto_resize_off_still_converts_a_bmp_and_notes_it() {
        // The BMP-to-PNG normalization (and its hint) is independent of auto-resize — pi's
        // `normalizeImage` always runs before the `autoResizeImages` check.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.bmp");
        image::DynamicImage::ImageRgb8(img)
            .save_with_format(&path, image::ImageFormat::Bmp)
            .unwrap();

        let out = Read::default()
            .with_image_auto_resize(false)
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        assert_eq!(out.images[0].media_type, "image/png");
        assert!(
            out.text
                .contains("[Image converted from image/bmp to image/png.]"),
            "got: {}",
            out.text
        );
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

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        assert_eq!(
            out.images[0].media_type, "image/png",
            "sniffed format must win over the misleading .jpg extension"
        );
    }

    #[tokio::test]
    async fn an_extensionless_image_is_still_detected_by_magic_bytes() {
        // A real image with *no* recognized extension at all (e.g. a clipboard/screenshot temp file,
        // or a plain `.dat`/no-extension name) — before this fix, only the extension gate decided
        // *whether* to route into the image path at all, so this would previously fall through to the
        // text path and get UTF-8-decoded as garbage instead of returned as an attachment.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        // No extension at all.
        let path = dir.path().join("clipboard-image-temp");
        std::fs::write(&path, &png_bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(
            out.images.len(),
            1,
            "an extensionless real image must still be detected and returned as an attachment"
        );
        assert_eq!(out.images[0].media_type, "image/png");
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
    fn resize_image_preserves_alpha_channel_for_a_transparent_png() {
        // Task 37 (pi-parity fix): `resize_image`'s PNG re-encode used to flatten every resized
        // image through `to_rgb8()` unconditionally — not "flatten onto a background", just
        // discarding the alpha channel entirely — so a half-transparent PNG needing this resize
        // path came back fully opaque. pi's Photon-based pipeline operates on RGBA internally with
        // no flattening step before its own PNG re-encode, so this must preserve transparency
        // through a downscale too. A smooth gradient (not noise) keeps this deterministic: it
        // compresses so well under PNG that it's guaranteed to win the PNG-vs-JPEG race regardless
        // of the extra byte cost alpha adds, so the test doesn't depend on incompressible-noise size
        // math at all — only on whether the alpha channel survives.
        let img = image::RgbaImage::from_fn(400, 400, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 100]) // uniform, well below opaque
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        let resized = resize_image(&bytes, image::ImageFormat::Png, 200, 200, 200_000, 80)
            .expect("a downscaled smooth gradient with alpha must fit comfortably");
        assert_eq!(
            resized.media_type, "image/png",
            "a compressible gradient must still win as lossless PNG — the only format here that can carry alpha at all"
        );

        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(&resized.base64_data)
            .unwrap();
        let decoded = image::load_from_memory(&decoded_bytes).unwrap();
        assert!(
            decoded.color().has_alpha(),
            "resized PNG must still carry an alpha channel, not be flattened to RGB"
        );
        let rgba = decoded.to_rgba8();
        assert!(
            rgba.pixels().all(|p| p[3] == 100),
            "alpha values must survive the resize+re-encode intact, not be dropped to fully opaque: \
             sample pixel alpha = {}",
            rgba.get_pixel(0, 0)[3]
        );
    }

    #[tokio::test]
    async fn read_resize_preserves_transparency_for_an_oversized_transparent_png() {
        // Task 37 (pi-parity fix), exercised end-to-end through `Read::run` with the tool's real
        // production budget/dimension constants: a transparent PNG whose dimensions alone exceed
        // `MAX_IMAGE_DIMENSION` forces the resize path (regardless of how small the file itself is),
        // matching this crate's other `MAX_IMAGE_DIMENSION`-forced-resize tests. Before this fix, a
        // half-transparent pixel here would have come back fully opaque, indistinguishable from an
        // originally-opaque image.
        let width = MAX_IMAGE_DIMENSION + 500;
        let height = 600;
        let img = image::RgbaImage::from_fn(width, height, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 90]) // uniform, well below opaque
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transparent_gradient.png");
        image::DynamicImage::ImageRgba8(img)
            .save_with_format(&path, image::ImageFormat::Png)
            .unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        assert!(out.text.contains("resized from"), "got: {}", out.text);
        assert_eq!(
            out.images[0].media_type, "image/png",
            "a compressible gradient must still win as lossless PNG"
        );

        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(&out.images[0].data)
            .unwrap();
        let decoded = image::load_from_memory(&decoded_bytes).unwrap();
        assert!(
            decoded.color().has_alpha(),
            "the resized/re-encoded image must still carry an alpha channel"
        );
        let rgba = decoded.to_rgba8();
        assert!(
            rgba.pixels().any(|p| p[3] != 255),
            "resized image must retain non-opaque pixels, not flatten to fully opaque: sample alpha = {}",
            rgba.get_pixel(0, 0)[3]
        );
    }

    #[test]
    fn resize_image_returns_none_when_the_budget_is_unreachable() {
        // Even at the smallest size/quality floor the retry loop bottoms out at, a 1-byte budget can
        // never be hit — `resize_image` must report that as `None` rather than looping forever or
        // returning something that silently exceeds the caller's budget.
        let img = image::RgbImage::from_pixel(64, 64, image::Rgb([10, 20, 30]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        assert!(resize_image(&bytes, image::ImageFormat::Png, 100, 100, 1, 80).is_none());
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

    #[tokio::test]
    async fn read_returns_a_graceful_text_result_not_an_error_when_resize_image_cannot_fit_it() {
        // pi: `image-process.ts`'s `processImage`, when `resizeImage` returns `null` — a *successful*,
        // text-only result (`ok: false` with a message, not a thrown error), tested explicitly at
        // `image-resize-callers.test.ts:32-52` (asserts `result.content` is text-only, contains "Image
        // omitted", and the call resolves rather than rejecting/erroring). `resize_image_returns_none_
        // when_the_budget_is_unreachable` above only pins the low-level helper directly with an injected
        // 1-byte budget; this goes one level up and exercises the real path a model actually calls
        // (`Read::run` -> `read_image` -> `resize_image`, with the tool's real, un-injectable budget and
        // dimension constants), asserting `Ok`, not `Err`.
        //
        // The production budget (4.5MB base64) can never be genuinely unreachable for a *real* photo —
        // even worst-case incompressible noise shrinks to a few KB at the retry loop's 64x64 floor — so
        // there's no way to construct a real image too big to ever fit. Instead this forces the same
        // `None` outcome `resize_image` returns for *any* internal failure, budget or otherwise: a file
        // comfortably past the byte-size gate (so `read_image`'s small-file fast-path validation, which
        // returns a different, legitimate hard `Err` — see `a_small_file_with_valid_magic_bytes_but_no_
        // real_image_body_is_rejected` above — never runs) carrying a valid PNG signature but a chunk
        // stream that can't actually decode. That reaches the exact `None` branch inside `read_image`
        // this test pins, indistinguishable from the caller's side from a real too-big-to-shrink image.
        let mut bytes = png_signature();
        // Comfortably past MAX_IMAGE_BASE64_BYTES once base64-encoded, so the small-file fast path is
        // skipped and this falls through to `resize_image` unvalidated.
        bytes.extend(vec![0u8; 4 * 1024 * 1024]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized_corrupt.png");
        std::fs::write(&path, &bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .expect("resize_image returning None must not surface as a tool Err");
        assert!(
            out.images.is_empty(),
            "no image attachment when it could not be resized to fit"
        );
        assert_eq!(
            out.text,
            "[Image omitted: could not be resized below the inline image size limit.]"
        );
    }

    /// A complete, structurally valid BMP (`BITMAPFILEHEADER` + 40-byte `BITMAPINFOHEADER`, followed by
    /// zeroed, correctly row-padded pixel data) — passes [`is_valid_bmp`]'s structural checks and its
    /// declared file size genuinely matches the buffer's length.
    fn valid_bmp_header(width: i32, height: i32, bits_per_pixel: u16) -> Vec<u8> {
        let row_bytes = (width.unsigned_abs() * u32::from(bits_per_pixel)).div_ceil(32) * 4;
        let pixel_data_len = (row_bytes * height.unsigned_abs()) as usize;
        let file_size = 54 + pixel_data_len;
        let mut b = vec![0u8; file_size];
        b[0] = b'B';
        b[1] = b'M';
        b[2..6].copy_from_slice(&(file_size as u32).to_le_bytes()); // declared file size
        b[10..14].copy_from_slice(&54u32.to_le_bytes()); // pixel data offset: right after both headers
        b[14..18].copy_from_slice(&40u32.to_le_bytes()); // DIB header size: BITMAPINFOHEADER
        b[18..22].copy_from_slice(&width.to_le_bytes());
        b[22..26].copy_from_slice(&height.to_le_bytes());
        b[26..28].copy_from_slice(&1u16.to_le_bytes()); // color planes
        b[28..30].copy_from_slice(&bits_per_pixel.to_le_bytes());
        b
    }

    /// A BMP whose file+DIB header passes [`is_valid_bmp`]'s structural checks (so it *does* sniff as
    /// BMP) but carries no actual pixel payload — decoding it still fails, exercising `read_image`'s
    /// graceful "could not be converted" outcome on a genuinely BMP-shaped (not just loosely
    /// magic-byte-matched) file.
    fn bmp_header_only_no_pixel_data() -> Vec<u8> {
        let mut b = valid_bmp_header(4, 4, 24);
        b.truncate(54); // drop the pixel data entirely…
        b[2..6].copy_from_slice(&58u32.to_le_bytes()); // …but still declare a few bytes of it present
        b // (needed for `is_valid_bmp` to accept the header at all), so decoding fails on truncated data.
    }

    #[test]
    fn is_valid_bmp_accepts_a_well_formed_header() {
        assert!(is_valid_bmp(&valid_bmp_header(4, 4, 24)));
    }

    #[test]
    fn is_valid_bmp_rejects_a_file_shorter_than_the_minimum_header() {
        assert!(!is_valid_bmp(b"BM"));
    }

    #[test]
    fn is_valid_bmp_rejects_a_pixel_data_offset_inside_the_headers() {
        let mut bmp = valid_bmp_header(4, 4, 24);
        bmp[10..14].copy_from_slice(&0u32.to_le_bytes()); // offset 0 lands before the headers even end
        assert!(!is_valid_bmp(&bmp));
    }

    #[test]
    fn is_valid_bmp_rejects_an_implausible_bits_per_pixel() {
        let mut bmp = valid_bmp_header(4, 4, 24);
        bmp[28..30].copy_from_slice(&7u16.to_le_bytes()); // not in {1,4,8,16,24,32}
        assert!(!is_valid_bmp(&bmp));
    }

    #[test]
    fn is_valid_bmp_rejects_more_than_one_color_plane() {
        let mut bmp = valid_bmp_header(4, 4, 24);
        bmp[26..28].copy_from_slice(&2u16.to_le_bytes());
        assert!(!is_valid_bmp(&bmp));
    }

    #[test]
    fn is_valid_bmp_rejects_plain_text_that_happens_to_start_with_bm() {
        // The exact shape of the original bug: ordinary text reinterpreted as BMP header fields almost
        // never lands on a plausible DIB header size (12, or 40-124) — an all-printable-ASCII buffer's
        // 4-byte little-endian read at the DIB-header-size offset is always some large, implausible
        // value (the high-order bytes are themselves printable-ASCII, i.e. >= 0x20).
        let text = b"BM this is definitely not an image, just some text.";
        assert!(!is_valid_bmp(text));
    }

    #[test]
    fn sniff_image_format_treats_an_invalid_bmp_header_as_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.bmp");
        std::fs::write(&path, b"BM not a real bmp header at all").unwrap();
        assert!(matches!(
            sniff_image_format(path.to_str().unwrap()),
            Sniff::Unknown
        ));
    }

    #[test]
    fn sniff_image_format_still_accepts_a_structurally_valid_bmp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("real.bmp");
        std::fs::write(&path, valid_bmp_header(4, 4, 24)).unwrap();
        assert!(matches!(
            sniff_image_format(path.to_str().unwrap()),
            Sniff::Format(image::ImageFormat::Bmp)
        ));
    }

    #[test]
    fn sniff_image_format_excludes_the_sof55_jpeg_ls_marker() {
        // pi: `detectSupportedImageMimeType` (`mime.ts:6-9`) — a 4th byte of 0xF7 is SOF55 (JPEG-LS), a
        // format this tool can't actually decode/display despite sharing JPEG's SOI+marker prefix.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("maybe.jpg");
        std::fs::write(&path, [0xFF, 0xD8, 0xFF, 0xF7, 0, 0, 0, 0]).unwrap();
        assert!(matches!(
            sniff_image_format(path.to_str().unwrap()),
            Sniff::Unknown
        ));
    }

    #[test]
    fn sniff_image_format_accepts_an_ordinary_jpeg_marker() {
        // Sanity: an ordinary (non-SOF55) 4th byte is unaffected by the new guard.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ordinary.jpg");
        std::fs::write(&path, [0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0]).unwrap(); // 0xE0 = APP0 (JFIF)
        assert!(matches!(
            sniff_image_format(path.to_str().unwrap()),
            Sniff::Format(image::ImageFormat::Jpeg)
        ));
    }

    #[tokio::test]
    async fn a_lossless_jpeg_ls_file_falls_back_to_text_instead_of_a_decode_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-really-jpeg.dat"); // no image extension, so only sniffing decides
        std::fs::write(&path, [0xFF, 0xD8, 0xFF, 0xF7, b'h', b'i']).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.images.is_empty(), "SOF55 must not be sent as an image");
        assert!(!out.text.contains("Read image"), "got: {}", out.text);
    }

    #[tokio::test]
    async fn read_returns_a_graceful_text_result_not_an_error_when_bmp_conversion_fails() {
        // Same graceful-outcome pattern as the resize-failure case above, for the other `read_image`
        // failure mode pi handles identically: BMP-to-PNG normalization failing. pi's `image-process.ts`
        // returns `{ ok: false, message: "[Image omitted: could not be converted to a supported inline
        // image format.]" }` rather than throwing; mirrored here as `Ok` text, not `Err`.
        //
        // This fixture's file+DIB header is *structurally valid* (passes `is_valid_bmp`, so it sniffs as
        // a real BMP, not a fallback-by-extension guess) but carries no actual pixel payload, so the
        // decode genuinely fails — proving the graceful path still works for a real BMP with a corrupt
        // body, distinct from `a_text_file_starting_with_bm_bytes_is_read_as_text_not_a_corrupt_bmp_
        // placeholder` below, which covers a file that was never really BMP at all.
        let bytes = bmp_header_only_no_pixel_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.bmp");
        std::fs::write(&path, &bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .expect("a BMP that fails to convert to PNG must not surface as a tool Err");
        assert!(out.images.is_empty());
        assert_eq!(
            out.text,
            "[Image omitted: could not be converted to a supported inline image format.]"
        );
    }

    #[tokio::test]
    async fn a_text_file_starting_with_bm_bytes_is_read_as_text_not_a_corrupt_bmp_placeholder() {
        // pi-parity gap (fixed): `image::guess_format`'s BMP signature is just the 2 ASCII bytes "BM"
        // (an empty mask in its own `MAGIC_BYTES` table) — before this fix, *any* file starting with
        // those two bytes sniffed as BMP, so an ordinary text file that happened to start with "BM"
        // would get routed into image handling, fail to decode, and come back as the
        // "[Image omitted: …]" placeholder instead of its real text content. `is_valid_bmp` (pi's
        // `isBmp`, `mime.ts:57-81`) now requires a plausible DIB header before treating a "BM"-prefixed
        // file as BMP at all, so a genuinely non-image file never reaches image handling — it's read as
        // ordinary text, same as any other file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        let content =
            "BM this file just happens to start with the letters B and M, nothing more.\n";
        std::fs::write(&path, content).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.images.is_empty(), "must not be treated as an image");
        assert!(
            out.text.contains(content.trim_end()),
            "must show the real text content, not an image placeholder: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn an_explicit_limit_above_2000_is_still_capped_at_the_default_ceiling() {
        // pi's `read.ts` always runs the selected window through `truncateHead`'s own default
        // `maxLines` (2000) *on top of* whatever `limit` the caller passed — an explicit `limit` larger
        // than 2000 must not bypass the ceiling `no_limit_argument_defaults_to_a_2000_line_cap` already
        // proves for the omitted-limit case.
        let body = (1..=3000)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let f = tmp_file(&format!("{body}\n"));
        let out = Read::default()
            .run(json!({ "path": f.path().to_str().unwrap(), "limit": 5000 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("  2000\t2000"), "line 2000 must be shown");
        assert!(
            !out.contains("\t2001\n"),
            "line 2001 must not be shown despite limit=5000"
        );
        assert!(
            out.contains("use offset=2001 to continue"),
            "got: {}",
            &out[out.len().saturating_sub(200)..]
        );
    }

    #[tokio::test]
    async fn a_non_vision_model_gets_a_note_appended_to_an_image_read() {
        // pi: `getNonVisionImageNote` (`read.ts`) — appended whenever the active model lacks image
        // input support. `Tool::run` has no model-capability context of its own, so this is driven via
        // the internal `_model_supports_vision` field `agent_core::agent`'s dispatch loop injects before
        // every tool call, keyed off the active model's own capability-table entry.
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        std::fs::write(&path, &png_bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap(), "_model_supports_vision": false }))
            .await
            .unwrap();
        assert!(
            out.text.contains(
                "[Current model does not support images. The image will be omitted from this request.]"
            ),
            "got: {}",
            out.text
        );
        assert!(
            out.images.is_empty(),
            "the image must actually be stripped when the model can't see it, not just noted in text"
        );
    }

    #[tokio::test]
    async fn a_vision_capable_model_gets_no_note_by_default() {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut png_bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot2.png");
        std::fs::write(&path, &png_bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap() }))
            .await
            .unwrap();
        assert!(
            !out.text.contains("does not support images"),
            "got: {}",
            out.text
        );
    }

    #[tokio::test]
    async fn the_non_vision_note_is_also_appended_when_the_image_is_omitted() {
        // pi appends `getNonVisionImageNote` in both branches (`processed.ok` and not) — the note is
        // about whether the model can *see* an attachment at all, independent of whether this
        // particular image could actually be inlined.
        let bytes = bmp_header_only_no_pixel_data();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt2.bmp");
        std::fs::write(&path, &bytes).unwrap();

        let out = Read::default()
            .run(json!({ "path": path.to_str().unwrap(), "_model_supports_vision": false }))
            .await
            .unwrap();
        assert!(out.text.contains("Image omitted"), "got: {}", out.text);
        assert!(
            out.text.contains("does not support images"),
            "got: {}",
            out.text
        );
    }

    #[test]
    fn description_documents_image_support_and_the_truncation_budget() {
        // Pi-parity fix: the description never mentioned this tool's extensive image/vision handling
        // at all, despite `read_image`/`sniff_image_format` supporting five formats — a model reading
        // the tool description alone had no way to know it could ask `read` to look at a screenshot.
        // Matches pi's `read.ts` description, which states both the supported formats and the
        // text-truncation budget numerically (mirroring `bash`'s own numeric description).
        let desc = Read::default().description().to_string();
        for format in ["png", "jpeg", "gif", "webp", "bmp"] {
            assert!(
                desc.contains(format),
                "description should list {format} as a supported image format, got: {desc}"
            );
        }
        assert!(
            desc.to_lowercase().contains("image"),
            "description should mention image support, got: {desc}"
        );
        assert!(
            desc.contains(&DEFAULT_LIMIT.to_string())
                && desc.contains(&super::super::output::format_size(MAX_OUTPUT_BYTES as u64)),
            "description should state the line/byte truncation budget numerically, got: {desc}"
        );
    }
}
