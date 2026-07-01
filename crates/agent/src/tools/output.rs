//! Bounded, spill-to-disk accumulation of streaming command output — a port of pi's
//! `OutputAccumulator` (`output-accumulator.ts`) plus the `truncateTail`/`formatSize` truncation
//! helpers (`truncate.ts`) and the bash tool's `formatOutput` marker strings (`bash.ts`).
//!
//! A long-running command can emit gigabytes; we must (a) never hold more than a bounded window in
//! memory, (b) still show the model the *end* of the stream (where errors and final results live),
//! and (c) preserve the *complete* output on disk so the model can go read it. This module does all
//! three: it tracks running line/byte totals, keeps a rolling byte tail for display, and spills the
//! full stream to a temp file once it outgrows the in-memory cap.
//!
//! The user-facing strings (the `[Showing …]` markers, `formatSize`'s `KB`/`MB` rendering) are
//! matched to pi exactly so behavior is identical across the two implementations.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default cap on displayed lines. Pairs with the byte cap — whichever bites first wins — so a flood
/// of short lines can't bury the model even while staying under the byte budget.
pub const DEFAULT_MAX_LINES: usize = 2000;
/// Default cap on displayed bytes (50 KiB). Protects the model's context window.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Which limit tripped truncation. Mirrors pi's `truncatedBy: "lines" | "bytes" | null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

/// The truncation metadata a snapshot carries — a port of pi's `TruncationResult`, keeping the
/// fields the bash formatter reads. `content` lives on [`OutputSnapshot`]; `firstLineExceedsLimit`
/// (a head-truncation concern) is omitted because tail truncation never sets it.
#[derive(Debug, Clone)]
pub struct Truncation {
    /// Whether the *complete* output exceeded either limit (computed from the running totals, not
    /// from the possibly-trimmed tail).
    pub truncated: bool,
    /// Which limit tripped, or `None` when not truncated.
    pub truncated_by: Option<TruncatedBy>,
    /// Total lines in the complete output.
    pub total_lines: u64,
    /// Total bytes in the complete output.
    pub total_bytes: u64,
    /// Lines actually shown in `content`.
    pub output_lines: u64,
    /// Bytes actually shown in `content`.
    pub output_bytes: u64,
    /// Whether the single shown line is itself a byte-truncated fragment (the "one huge line"
    /// edge case) rather than a whole line.
    pub last_line_partial: bool,
    /// The line limit that was applied.
    pub max_lines: usize,
    /// The byte limit that was applied.
    pub max_bytes: usize,
}

/// A display-ready view of the accumulated output at a point in time.
///
/// `last_line_bytes` is not part of pi's `OutputSnapshot` (pi reads it off the live accumulator via
/// `getLastLineBytes()`), but [`format_output`] is given only the snapshot, so we carry the current
/// (last) line's full byte size here to render the partial-line marker faithfully.
#[derive(Debug, Clone)]
pub struct OutputSnapshot {
    /// The tail of the output, decoded and truncated to the display limits.
    pub content: String,
    /// Truncation metadata for `content`.
    pub truncation: Truncation,
    /// Path to the temp file holding the *complete* output, when one was opened.
    pub full_output_path: Option<String>,
    /// Full byte size of the current (last) line — used for the partial-line marker.
    pub last_line_bytes: u64,
}

/// Format a bracketed truncation marker — pi's `[<description>]` convention, already used by
/// [`format_output`]'s own `[Showing …]`/`[Full output: …]` markers. `read`/`grep`/`ls`/`find` each
/// wrap their own truncation description in this shared bracket format instead of a hand-rolled
/// `… (…)` string, so a truncation note reads consistently regardless of which tool emitted it.
pub fn marker(description: impl std::fmt::Display) -> String {
    format!("[{description}]")
}

/// Format a byte count the way pi's `formatSize` does: `<1024` → `"{n}B"`; `<1 MiB` →
/// `"{:.1}KB"`; otherwise `"{:.1}MB"`. Divisor is 1024 at each step.
pub fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Ceiling on total rendered output bytes for listing-style tools (`grep`, `find`, `ls`). Their
/// per-item cap (`limit`) bounds *count*, but long paths/lines can still blow past a sane context
/// budget well before `limit` items are reached; this is the backstop on the assembled text itself.
pub const MAX_LISTING_BYTES: usize = 50 * 1024;

/// Truncate an already-assembled listing `out` to [`MAX_LISTING_BYTES`] and append a marker, when it
/// exceeds the cap. Returns whether truncation happened, so callers can skip a redundant count-based
/// marker: the byte cap is checked *before* any count marker is appended, and takes priority when both
/// would otherwise fire, since appending the count marker first and then truncating to the byte cap
/// could slice straight through — and silently drop — the marker just added.
pub fn cap_listing_bytes(out: &mut String, guidance: &str) -> bool {
    if out.len() <= MAX_LISTING_BYTES {
        return false;
    }
    let mut end = MAX_LISTING_BYTES;
    while !out.is_char_boundary(end) {
        end -= 1;
    }
    out.truncate(end);
    out.push_str("\n\n");
    out.push_str(&marker(format_args!(
        "output truncated at {}; {guidance}",
        format_size(MAX_LISTING_BYTES as u64)
    )));
    true
}

/// A unique-enough 16-hex-char (8-byte) token for temp-file names. It only needs to avoid collisions
/// between concurrent commands, not be cryptographic — so we mix wall-clock nanos, the pid, and a
/// per-process atomic counter through splitmix64 rather than pull in a CSPRNG.
fn random_hex16() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = u64::from(std::process::id());
    // splitmix64 finalizer over the mixed seed — cheap, well-distributed avalanche.
    let mut x = nanos ^ pid.rotate_left(32) ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    format!("{x:016x}")
}

/// Incrementally accumulates streaming output with bounded memory.
///
/// Feed raw bytes with [`append`](Self::append); call [`finish`](Self::finish) when the stream ends;
/// take a [`snapshot`](Self::snapshot) for display. Memory is bounded to roughly the rolling tail cap
/// (`2 × max_bytes`) plus the not-yet-spilled raw buffer — once the output outgrows the rolling cap
/// the raw buffer is flushed to a temp file and freed, so the file (not memory) holds the full stream.
pub struct OutputAccumulator {
    // Display limits and the derived rolling-tail cap.
    max_lines: usize,
    max_bytes: usize,
    rolling_cap: usize, // 2 × max_bytes — enough tail to reconstruct any tail-truncation window.
    temp_prefix: String,

    // Running totals over the *complete* stream (never trimmed).
    total_bytes: u64,
    completed_lines: u64,    // count of '\n' seen
    has_open_line: bool,     // is there content after the last '\n'?
    current_line_bytes: u64, // byte size of that open (last) line

    // Rolling byte tail kept for display, capped at `rolling_cap`.
    tail: Vec<u8>,
    tail_starts_at_line_boundary: bool,

    // Complete bytes held in memory until the first spill, then emptied.
    raw_buffer: Vec<u8>,
    spilled: bool,
    spill_failed: bool, // a temp-file error occurred; don't retry, degrade to tail-only

    // The spill target: path is retained for the snapshot even after the writer is flushed.
    temp_path: Option<PathBuf>,
    temp_writer: Option<BufWriter<File>>,

    finished: bool,
}

impl Default for OutputAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputAccumulator {
    /// A new accumulator with the default limits and pi's `"pi-bash"` temp-file prefix.
    pub fn new() -> Self {
        Self::with_prefix("pi-bash")
    }

    /// A new accumulator whose spill files are named `<prefix>-<16 hex>.log`.
    pub fn with_prefix(prefix: &str) -> Self {
        let max_bytes = DEFAULT_MAX_BYTES;
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes,
            // `max(_, 1)` mirrors pi guarding against a zero cap.
            rolling_cap: (max_bytes * 2).max(1),
            temp_prefix: prefix.to_string(),
            total_bytes: 0,
            completed_lines: 0,
            has_open_line: false,
            current_line_bytes: 0,
            tail: Vec::new(),
            tail_starts_at_line_boundary: true,
            raw_buffer: Vec::new(),
            spilled: false,
            spill_failed: false,
            temp_path: None,
            temp_writer: None,
            finished: false,
        }
    }

    /// Total lines over the complete stream: completed lines plus one for a trailing partial line.
    fn total_lines(&self) -> u64 {
        self.completed_lines + u64::from(self.has_open_line)
    }

    /// Accumulate a chunk of raw bytes. Updates the running totals, extends the rolling tail, and
    /// either buffers the bytes in memory or writes them straight to the spill file. Appends after
    /// [`finish`](Self::finish) are ignored (pi throws; we can't panic, so we degrade to a no-op).
    pub fn append(&mut self, data: &[u8]) {
        if self.finished || data.is_empty() {
            return;
        }

        self.total_bytes += data.len() as u64;
        self.update_line_counters(data);
        self.extend_tail(data);

        if self.spilled {
            // Post-spill: stream straight to the file so memory stays bounded.
            if let Some(w) = self.temp_writer.as_mut() {
                if w.write_all(data).is_err() {
                    self.mark_spill_broken();
                }
            }
        } else if !self.spill_failed {
            // Pre-spill: keep the complete bytes in memory so a later spill can flush them whole.
            self.raw_buffer.extend_from_slice(data);
            // Spill once the buffered output outgrows the rolling cap *or* the line count outgrows the
            // display limit — a long, low-byte/high-line-count command (many short lines) would
            // otherwise never trip the byte-only check and so never get a `full_output_path` to point
            // at, even once it's already well past what the tail can show.
            if self.total_bytes as usize > self.rolling_cap
                || self.total_lines() > self.max_lines as u64
            {
                self.ensure_temp_file();
            }
        }
    }

    /// Mark the stream finished and flush any spill writer so a subsequent read sees all bytes.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if let Some(w) = self.temp_writer.as_mut() {
            if w.flush().is_err() {
                self.mark_spill_broken();
            }
        }
    }

    /// Produce a display snapshot: the tail decoded and tail-truncated to the display limits, plus
    /// truncation metadata computed from the *complete-stream* totals. When `persist_if_truncated`
    /// and the stream is truncated, a temp file holding the full output is ensured so
    /// `full_output_path` is populated.
    pub fn snapshot(&mut self, persist_if_truncated: bool) -> OutputSnapshot {
        let text = self.snapshot_text();
        let tail = truncate_tail(&text, self.max_lines, self.max_bytes);

        let total_lines = self.total_lines();
        let truncated =
            total_lines > self.max_lines as u64 || self.total_bytes > self.max_bytes as u64;
        // Prefer the tail truncation's verdict; fall back to whichever limit the totals blew past.
        let truncated_by = if truncated {
            tail.truncated_by
                .or(Some(if self.total_bytes > self.max_bytes as u64 {
                    TruncatedBy::Bytes
                } else {
                    TruncatedBy::Lines
                }))
        } else {
            None
        };

        if persist_if_truncated && truncated {
            self.ensure_temp_file();
            // Flush so a reader (e.g. the model, or a test) sees the bytes we just wrote.
            if let Some(w) = self.temp_writer.as_mut() {
                if w.flush().is_err() {
                    self.mark_spill_broken();
                }
            }
        }

        OutputSnapshot {
            content: tail.content,
            truncation: Truncation {
                truncated,
                truncated_by,
                total_lines,
                total_bytes: self.total_bytes,
                output_lines: tail.output_lines,
                output_bytes: tail.output_bytes,
                last_line_partial: tail.last_line_partial,
                max_lines: self.max_lines,
                max_bytes: self.max_bytes,
            },
            full_output_path: self
                .temp_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            last_line_bytes: self.current_line_bytes,
        }
    }

    /// Update the line/current-line counters for a chunk. Counting `'\n'` bytes over raw UTF-8 is
    /// exact: `0x0A` never appears inside a multi-byte sequence, so it always denotes a newline.
    fn update_line_counters(&mut self, data: &[u8]) {
        let mut newlines = 0u64;
        let mut last_nl: Option<usize> = None;
        for (i, &b) in data.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                last_nl = Some(i);
            }
        }
        match last_nl {
            None => {
                // No newline: this chunk extends the current open line.
                self.current_line_bytes += data.len() as u64;
                self.has_open_line = true;
            }
            Some(pos) => {
                self.completed_lines += newlines;
                // Bytes after the last '\n' form the new open line (possibly empty).
                let tail_len = (data.len() - 1 - pos) as u64;
                self.current_line_bytes = tail_len;
                self.has_open_line = tail_len > 0;
            }
        }
    }

    /// Append to the rolling tail, trimming it back to `rolling_cap` when it overflows so memory
    /// stays bounded regardless of stream length.
    fn extend_tail(&mut self, data: &[u8]) {
        self.tail.extend_from_slice(data);
        if self.tail.len() > self.rolling_cap {
            self.trim_tail();
        }
    }

    /// Drop the front of the tail down to the last `rolling_cap` bytes, advancing the cut point past
    /// any UTF-8 continuation bytes so the retained tail still starts on a char boundary. Records
    /// whether the new front sits at a line boundary (for `snapshot_text`).
    fn trim_tail(&mut self) {
        if self.tail.len() <= self.rolling_cap {
            return;
        }
        let mut start = self.tail.len() - self.rolling_cap;
        // 0b10xxxxxx bytes are UTF-8 continuations — skip forward to the next lead byte.
        while start < self.tail.len() && (self.tail[start] & 0xC0) == 0x80 {
            start += 1;
        }
        if start > 0 {
            self.tail_starts_at_line_boundary = self.tail[start - 1] == b'\n';
        }
        self.tail.drain(0..start);
    }

    /// The tail decoded lossily to UTF-8, with a leading partial line dropped when the tail was
    /// trimmed mid-line — so display never shows a half-line at the top of the window.
    fn snapshot_text(&self) -> String {
        let s = String::from_utf8_lossy(&self.tail).into_owned();
        if self.tail_starts_at_line_boundary {
            return s;
        }
        match s.find('\n') {
            Some(i) => s[i + 1..].to_string(),
            None => s,
        }
    }

    /// Open the spill file if it isn't open yet, writing everything buffered so far into it and then
    /// freeing the in-memory buffer. On any filesystem error we mark `spill_failed`, free the buffer
    /// to stay bounded, and continue tail-only — the full output is lost but nothing panics.
    fn ensure_temp_file(&mut self) {
        if self.spilled || self.spill_failed {
            return;
        }
        let path =
            std::env::temp_dir().join(format!("{}-{}.log", self.temp_prefix, random_hex16()));
        let mut opts = std::fs::OpenOptions::new();
        // `create_new`, not `create`: `random_hex16` isn't cryptographically random, and this file
        // lives in the shared, world-writable system temp dir, so refuse to follow anything already
        // there (planted or coincidental) rather than silently truncating through it. `0600` — a
        // command's full output (env vars, file contents it printed, secrets) shouldn't be readable
        // by another local user on a shared host, set atomically at creation rather than via a
        // `set_permissions` call afterward.
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        match opts.open(&path) {
            Ok(f) => {
                let mut w = BufWriter::new(f);
                if w.write_all(&self.raw_buffer).is_err() {
                    self.spill_failed = true;
                    self.raw_buffer = Vec::new();
                    return;
                }
                self.raw_buffer = Vec::new();
                self.temp_path = Some(path);
                self.temp_writer = Some(w);
                self.spilled = true;
            }
            Err(_) => {
                self.spill_failed = true;
                self.raw_buffer = Vec::new();
            }
        }
    }

    /// Mark the spill file as no longer trustworthy after a write/flush failure partway through
    /// streaming to it (e.g. the disk filled up mid-command): stop advertising it via
    /// `full_output_path` — the bytes on disk are now silently truncated relative to what the
    /// command actually produced — and stop attempting further writes to it.
    fn mark_spill_broken(&mut self) {
        self.spill_failed = true;
        self.temp_path = None;
        self.temp_writer = None;
    }
}

/// Result of tail-truncating a decoded string — the subset of pi's `TruncationResult` that
/// [`OutputAccumulator::snapshot`] reads back out (the rest it overrides with stream totals).
struct TailTruncation {
    content: String,
    output_lines: u64,
    output_bytes: u64,
    last_line_partial: bool,
    truncated_by: Option<TruncatedBy>,
}

/// Keep the last `max_lines` lines / `max_bytes` bytes of `content` — whichever limit bites first —
/// a port of pi's `truncateTail`. Never returns a partial line except the one edge case where a
/// single line alone exceeds `max_bytes`, in which case its trailing `max_bytes` are kept and
/// `last_line_partial` is set.
fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> TailTruncation {
    let total_bytes = content.len(); // Rust `str` length is already the UTF-8 byte length.

    let mut lines: Vec<&str> = content.split('\n').collect();
    // A trailing '\n' yields a spurious empty final element — drop it so "a\nb\n" is two lines.
    if lines.len() > 1 && lines.last() == Some(&"") {
        lines.pop();
    }
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TailTruncation {
            content: content.to_string(),
            output_lines: total_lines as u64,
            output_bytes: total_bytes as u64,
            last_line_partial: false,
            truncated_by: None,
        };
    }

    // Walk backward, prepending whole lines until we'd blow the byte budget or fill the line budget.
    let mut collected: Vec<&str> = Vec::new(); // in reverse order
    let mut output_bytes_count: usize = 0;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;
    let mut partial: Option<String> = None;

    let mut i = lines.len();
    while i > 0 && collected.len() < max_lines {
        i -= 1;
        let line = lines[i];
        // +1 for the '\n' that will rejoin this line to the one already collected below it.
        let line_bytes = line.len() + if collected.is_empty() { 0 } else { 1 };
        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            // Edge case: nothing fits yet and this single line exceeds the byte limit — keep its
            // trailing `max_bytes` as a partial fragment.
            if collected.is_empty() {
                let frag = truncate_str_to_bytes_from_end(line, max_bytes);
                output_bytes_count = frag.len();
                partial = Some(frag.to_string());
                last_line_partial = true;
            }
            break;
        }
        collected.push(line);
        output_bytes_count += line_bytes;
    }

    // Filled the line budget with bytes to spare → it was a line-limit truncation.
    if collected.len() >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let (content_str, output_lines) = if let Some(p) = partial {
        (p, 1u64)
    } else {
        collected.reverse();
        (collected.join("\n"), collected.len() as u64)
    };
    let output_bytes = content_str.len() as u64;

    TailTruncation {
        content: content_str,
        output_lines,
        output_bytes,
        last_line_partial,
        truncated_by: Some(truncated_by),
    }
}

/// Return the longest suffix of `s` whose UTF-8 byte length is `<= max_bytes`, cut on a char
/// boundary. (from_utf8_lossy has already replaced any invalid bytes, so there are no unpaired
/// surrogates to handle as pi's version must.)
fn truncate_str_to_bytes_from_end(s: &str, max_bytes: usize) -> &str {
    if max_bytes == 0 {
        return "";
    }
    if s.len() <= max_bytes {
        return s;
    }
    let mut start = s.len() - max_bytes;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

/// Render a snapshot as the bash tool's display text — a port of `bash.ts`'s `formatOutput`. When not
/// truncated, returns `content` (or `empty_text` when `content` is empty). When truncated, appends a
/// blank line and one of three markers, matching pi's strings byte-for-byte.
pub fn format_output(snapshot: &OutputSnapshot, empty_text: &str) -> String {
    let content = &snapshot.content;
    let mut text = if content.is_empty() {
        empty_text.to_string()
    } else {
        content.clone()
    };

    let t = &snapshot.truncation;
    if t.truncated {
        let end_line = t.total_lines;
        let start_line = t
            .total_lines
            .saturating_sub(t.output_lines)
            .saturating_add(1);
        // pi interpolates the path directly; when there is none we render an empty string rather
        // than the literal "undefined" JS would produce.
        let path = snapshot.full_output_path.as_deref().unwrap_or("");

        let marker = if t.last_line_partial {
            format!(
                "[Showing last {} of line {} (line is {}). Full output: {}]",
                format_size(t.output_bytes),
                end_line,
                format_size(snapshot.last_line_bytes),
                path,
            )
        } else if matches!(t.truncated_by, Some(TruncatedBy::Lines)) {
            format!(
                "[Showing lines {start_line}-{end_line} of {}. Full output: {path}]",
                t.total_lines,
            )
        } else {
            format!(
                "[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {path}]",
                t.total_lines,
                format_size(DEFAULT_MAX_BYTES as u64),
            )
        };

        text.push_str("\n\n");
        text.push_str(&marker);
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;

    // Read a spilled temp file's raw bytes back for full-output assertions.
    fn read_temp(path: &str) -> Vec<u8> {
        let mut f = std::fs::File::open(path).unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        buf
    }

    #[test]
    fn small_output_is_not_truncated() {
        let mut acc = OutputAccumulator::new();
        acc.append(b"a\nb\nc\n");
        acc.finish();
        let snap = acc.snapshot(true);

        assert!(!snap.truncation.truncated);
        assert_eq!(snap.truncation.truncated_by, None);
        assert_eq!(snap.truncation.total_lines, 3);
        assert_eq!(snap.truncation.total_bytes, 6);
        // Not-truncated content is the original text, trailing newline and all.
        assert_eq!(snap.content, "a\nb\nc\n");
        // Nothing to persist → no temp file opened.
        assert!(snap.full_output_path.is_none());
        // format_output returns the content verbatim, no marker.
        assert_eq!(format_output(&snap, "(no output)"), "a\nb\nc\n");
    }

    #[test]
    fn empty_output_uses_empty_text() {
        let mut acc = OutputAccumulator::new();
        acc.finish();
        let snap = acc.snapshot(true);
        assert!(!snap.truncation.truncated);
        assert_eq!(snap.truncation.total_lines, 0);
        assert_eq!(format_output(&snap, "(no output)"), "(no output)");
    }

    #[test]
    fn line_limit_keeps_the_tail_with_correct_markers() {
        // 2500 short lines: over the 2000-line cap but well under the 50 KiB byte cap, so the *line*
        // limit is what bites.
        let mut acc = OutputAccumulator::new();
        let mut full = String::new();
        for n in 1..=2500 {
            let line = format!("line{n}\n");
            full.push_str(&line);
            acc.append(line.as_bytes());
        }
        acc.finish();
        let snap = acc.snapshot(true);

        assert!(snap.truncation.truncated);
        assert_eq!(snap.truncation.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(snap.truncation.total_lines, 2500);
        assert_eq!(snap.truncation.output_lines, 2000);
        // The tail is the last 2000 lines: line501 … line2500.
        assert!(
            snap.content.starts_with("line501\n"),
            "got start: {:?}",
            &snap.content[..16]
        );
        assert!(snap.content.ends_with("line2500"));

        // Marker: [Showing lines 501-2500 of 2500. Full output: <path>]
        let path = snap.full_output_path.clone().expect("spilled on persist");
        let out = format_output(&snap, "(no output)");
        assert!(
            out.ends_with(&format!(
                "[Showing lines 501-2500 of 2500. Full output: {path}]"
            )),
            "got: {out}"
        );

        // The temp file holds the COMPLETE 2500-line output.
        let disk = read_temp(&path);
        assert_eq!(disk, full.as_bytes());
    }

    #[test]
    fn byte_limit_keeps_the_tail_with_correct_markers() {
        // 250 lines of 500 bytes each ≈ 125 KiB: under the 2000-line cap but over the 50 KiB byte
        // cap, so the *byte* limit bites (and it exceeds the rolling cap, forcing an auto-spill).
        let mut acc = OutputAccumulator::new();
        let mut full = Vec::new();
        for _ in 0..250 {
            let mut line = vec![b'a'; 500];
            line.push(b'\n');
            full.extend_from_slice(&line);
            acc.append(&line);
        }
        acc.finish();
        let snap = acc.snapshot(true);

        assert!(snap.truncation.truncated);
        assert_eq!(snap.truncation.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(snap.truncation.total_lines, 250);
        // Shown bytes never exceed the byte cap.
        assert!(snap.truncation.output_bytes <= DEFAULT_MAX_BYTES as u64);

        let out = format_output(&snap, "(no output)");
        assert!(out.contains("(50.0KB limit)"), "got: {out}");
        assert!(out.contains(&format!("of {} ", 250)), "got: {out}");

        // Full output preserved on disk.
        let path = snap.full_output_path.clone().expect("spilled");
        assert_eq!(read_temp(&path), full);
    }

    #[test]
    fn spill_writes_complete_output_and_sets_path() {
        // 300 KiB, comfortably past the ~100 KiB rolling cap → auto-spill during append.
        let mut acc = OutputAccumulator::with_prefix("pi-bash");
        let mut full = Vec::new();
        for n in 0..300u32 {
            let mut line = format!("chunk-{n:04}-").into_bytes();
            line.resize(1023, b'x');
            line.push(b'\n');
            full.extend_from_slice(&line);
            acc.append(&line);
        }
        acc.finish();
        let snap = acc.snapshot(true);

        let path = snap.full_output_path.clone().expect("must spill");
        // The temp file name uses the prefix and lives in the system temp dir.
        assert!(path.contains("pi-bash-"));
        assert!(path.ends_with(".log"));
        // The COMPLETE output round-trips off disk, byte-for-byte.
        assert_eq!(read_temp(&path), full);
        // Display content is bounded, not the whole 300 KiB.
        assert!(snap.content.len() <= DEFAULT_MAX_BYTES);

        // A command's full output can carry secrets (env vars, file contents it printed); this
        // lives in the shared, world-writable system temp dir, so it must not be group/world
        // readable regardless of the process umask.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "spill file must be created private");
        }
    }

    #[test]
    fn spill_write_failure_stops_advertising_the_corrupted_file() {
        // Simulating the actual OS failure (disk full mid-write) isn't practical without an
        // injectable filesystem seam, so this drives the same recovery path `append`/`finish`
        // call on a real write/flush error. The contract under test: once the writer is known
        // broken, `snapshot` must stop pointing callers at a file that's now silently truncated
        // relative to what the command actually produced, and further writes must not panic or
        // resurrect it.
        let mut acc = OutputAccumulator::with_prefix("pi-bash");
        let mut full = Vec::new();
        for n in 0..300u32 {
            let mut line = format!("chunk-{n:04}-").into_bytes();
            line.resize(1023, b'x');
            line.push(b'\n');
            full.extend_from_slice(&line);
            acc.append(&line);
        }
        assert!(
            acc.spilled,
            "test setup: expected auto-spill to have happened"
        );
        assert!(acc.temp_path.is_some());

        acc.mark_spill_broken();

        // Data after the break must not panic, and must not silently re-enable the path.
        acc.append(b"more data after the writer broke\n");
        acc.finish();
        let snap = acc.snapshot(true);

        assert!(
            snap.full_output_path.is_none(),
            "a broken spill file must not be advertised as the complete output"
        );
        assert!(acc.spill_failed);
    }

    #[test]
    fn line_count_alone_triggers_a_spill_well_under_the_byte_cap() {
        // Many short lines: total bytes stay far under the rolling byte cap, but the line count alone
        // must still trigger a spill — a long, low-byte/high-line-count command shouldn't go without a
        // `full_output_path` just because it never got big.
        let mut acc = OutputAccumulator::with_prefix("pi-bash");
        let mut full = Vec::new();
        for n in 0..(DEFAULT_MAX_LINES as u32 + 500) {
            let line = format!("{n}\n").into_bytes();
            full.extend_from_slice(&line);
            acc.append(&line);
        }
        assert!(
            full.len() < DEFAULT_MAX_BYTES,
            "test setup: total bytes ({}) must stay under the byte cap ({DEFAULT_MAX_BYTES})",
            full.len()
        );
        acc.finish();
        let snap = acc.snapshot(true);
        let path = snap
            .full_output_path
            .clone()
            .expect("line count alone must trigger a spill");
        assert_eq!(read_temp(&path), full);
    }

    #[test]
    fn single_huge_line_is_shown_partial() {
        // One 200_000-byte line with no newline: total_lines == 1, over the byte cap → the tail is a
        // byte-truncated fragment of that one line.
        let mut acc = OutputAccumulator::new();
        acc.append(&vec![b'a'; 200_000]);
        acc.finish();
        let snap = acc.snapshot(true);

        assert!(snap.truncation.truncated);
        assert!(snap.truncation.last_line_partial);
        assert_eq!(snap.truncation.total_lines, 1);
        assert_eq!(snap.last_line_bytes, 200_000);
        // Fragment is capped at the byte limit.
        assert!(snap.content.len() <= DEFAULT_MAX_BYTES);

        // Marker: [Showing last 50.0KB of line 1 (line is 195.3KB). Full output: <path>]
        let path = snap.full_output_path.clone().expect("spilled");
        let out = format_output(&snap, "(no output)");
        assert!(
            out.ends_with(&format!(
                "[Showing last 50.0KB of line 1 (line is 195.3KB). Full output: {path}]"
            )),
            "got tail: {}",
            &out[out.len().saturating_sub(120)..]
        );
    }

    #[test]
    fn marker_wraps_the_description_in_brackets() {
        assert_eq!(marker("plain text"), "[plain text]");
        assert_eq!(
            marker(format_args!("{} more entries", 7)),
            "[7 more entries]"
        );
    }

    #[test]
    fn format_size_boundaries() {
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(1023), "1023B");
        assert_eq!(format_size(1024), "1.0KB");
        assert_eq!(format_size(1536), "1.5KB");
        assert_eq!(format_size(51200), "50.0KB");
        assert_eq!(format_size(1024 * 1024), "1.0MB");
        assert_eq!(format_size(1024 * 1024 * 3 / 2), "1.5MB");
    }

    #[test]
    fn format_output_marker_strings_match_pi() {
        // Lines marker.
        let lines_snap = OutputSnapshot {
            content: "tail".to_string(),
            truncation: Truncation {
                truncated: true,
                truncated_by: Some(TruncatedBy::Lines),
                total_lines: 2500,
                total_bytes: 12345,
                output_lines: 2000,
                output_bytes: 9000,
                last_line_partial: false,
                max_lines: DEFAULT_MAX_LINES,
                max_bytes: DEFAULT_MAX_BYTES,
            },
            full_output_path: Some("/tmp/pi-bash-abc.log".to_string()),
            last_line_bytes: 4,
        };
        assert_eq!(
            format_output(&lines_snap, "(no output)"),
            "tail\n\n[Showing lines 501-2500 of 2500. Full output: /tmp/pi-bash-abc.log]"
        );

        // Bytes marker.
        let bytes_snap = OutputSnapshot {
            truncation: Truncation {
                truncated_by: Some(TruncatedBy::Bytes),
                total_lines: 100,
                output_lines: 40,
                ..lines_snap.truncation.clone()
            },
            ..lines_snap.clone()
        };
        assert_eq!(
            format_output(&bytes_snap, "(no output)"),
            "tail\n\n[Showing lines 61-100 of 100 (50.0KB limit). Full output: /tmp/pi-bash-abc.log]"
        );

        // Partial-line marker.
        let partial_snap = OutputSnapshot {
            truncation: Truncation {
                truncated_by: Some(TruncatedBy::Bytes),
                total_lines: 1,
                output_lines: 1,
                output_bytes: 51200,
                last_line_partial: true,
                ..lines_snap.truncation.clone()
            },
            last_line_bytes: 200_000,
            ..lines_snap.clone()
        };
        assert_eq!(
            format_output(&partial_snap, "(no output)"),
            "tail\n\n[Showing last 50.0KB of line 1 (line is 195.3KB). Full output: /tmp/pi-bash-abc.log]"
        );
    }

    #[test]
    fn truncate_tail_no_truncation_returns_original() {
        let r = truncate_tail("a\nb\nc\n", 2000, 50 * 1024);
        assert!(r.truncated_by.is_none());
        assert_eq!(r.content, "a\nb\nc\n");
        assert_eq!(r.output_lines, 3);
        assert!(!r.last_line_partial);
    }
}
