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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default cap on displayed lines. Pairs with the byte cap — whichever bites first wins — so a flood
/// of short lines can't bury the model even while staying under the byte budget.
pub const DEFAULT_MAX_LINES: usize = 2000;
/// Default cap on displayed bytes (50 KiB). Protects the model's context window.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Hard ceiling on how many bytes any one spill file may take (128 MiB).
///
/// [`DEFAULT_MAX_BYTES`] bounds only what the *model is shown*; the full stream still reaches this
/// accumulator, and the spill file used to grow for as long as the command kept talking. Memory stays
/// bounded either way (the rolling tail, plus a pre-spill buffer that's flushed and freed) — the file
/// was the unbounded side, and on this crate's deployment target the file *is* memory: the system temp
/// dir is a tmpfs (see `worktree.rs`), so "spill to disk" is spill to RAM. With `bash`'s 30-minute
/// default timeout, one `yes` or `cat /dev/urandom | base64` could fill it — and the file outlives the
/// call by [`STALE_TEMP_FILE_AGE`], so the damage persists long after the command is gone.
///
/// The number is sized against the question "what is the largest output someone would genuinely want
/// to page back through?": a fully verbose build log, a long test run, a chatty migration — real cases
/// land in the low tens of MiB at the extreme. 128 MiB clears that by roughly an order of magnitude,
/// so no legitimate command loses bytes it had a reader for, while several concurrent firehoses
/// together still stay a small fraction of a multi-GiB tmpfs. Past the budget we simply stop writing:
/// the accumulator keeps running, so the tail, the totals, and the truncation markers all stay
/// correct — the only thing dropped is the middle of a stream nobody was going to read.
pub const MAX_SPILL_BYTES: u64 = 128 * 1024 * 1024;

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
    /// Path to the temp file holding the complete output, when one was opened — or, when
    /// `full_output_capped` is set, only its first `full_output_bytes`.
    pub full_output_path: Option<String>,
    /// Whether the spill file stopped short of the complete stream because it hit
    /// [`MAX_SPILL_BYTES`]. The file still holds a genuine *prefix* of the output (not a corrupt
    /// one — see [`OutputAccumulator::mark_spill_broken`] for that case), so it's worth pointing the
    /// model at; but a marker that names a path while the model assumes it's the whole document is
    /// worse than naming no path at all, so [`format_output`] says so explicitly when this is set.
    pub full_output_capped: bool,
    /// How many bytes actually landed in the spill file. Equals `truncation.total_bytes` unless
    /// `full_output_capped`.
    pub full_output_bytes: u64,
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
///
/// Deliberately defined as [`DEFAULT_MAX_BYTES`], not a second independent `50 * 1024` literal: pi
/// has a single shared `DEFAULT_MAX_BYTES` (`truncate.ts`) for both the streaming (`bash`/`read`) and
/// one-shot (`ls`/`grep`/`find`) truncation paths. The two byte caps here agreed only by coincidence
/// before this alias — changing one without the other would have silently split bash/read's
/// truncation budget from ls/grep/find's, with no compiler or test to catch the drift.
pub const MAX_LISTING_BYTES: usize = DEFAULT_MAX_BYTES;

/// Truncate an already-assembled listing `out` to [`MAX_LISTING_BYTES`] and append a marker, when it
/// exceeds the cap. Returns whether truncation happened, so callers can skip a redundant count-based
/// marker: the byte cap is checked *before* any count marker is appended, and takes priority when both
/// would otherwise fire, since appending the count marker first and then truncating to the byte cap
/// could slice straight through — and silently drop — the marker just added.
///
/// The cut point is backed off twice: first to the nearest UTF-8 char boundary (so we never split a
/// multi-byte codepoint), then further back to the last `\n` (or the very start of the string, if none
/// precedes it) so a whole *line* is always kept — matching pi's `truncateHead` (`truncate.ts`), which
/// is documented and enforced to "never return partial lines." Without the second back-off, the raw
/// byte cut can land mid-filename/mid-match-line: the model then sees what looks like a complete,
/// plausible path or match immediately before the truncation marker, when it's actually an arbitrary
/// fragment that just happened to end where the byte budget ran out.
pub fn cap_listing_bytes(out: &mut String, guidance: &str) -> bool {
    if out.len() <= MAX_LISTING_BYTES {
        return false;
    }
    let mut end = MAX_LISTING_BYTES;
    while !out.is_char_boundary(end) {
        end -= 1;
    }
    end = out[..end].rfind('\n').map(|i| i + 1).unwrap_or(0);
    out.truncate(end);
    out.push_str("\n\n");
    out.push_str(&marker(format_args!(
        "output truncated at {}; {guidance}",
        format_size(MAX_LISTING_BYTES as u64)
    )));
    true
}

/// Format `path` (as yielded by walking `root`) the way pi's `grep.ts`/`find.ts` do: relative to `root`
/// when `root` is a directory, or just the basename when `root` names a single file (the two would
/// otherwise coincide, showing nothing meaningful beyond the name itself). Shared by `grep`/`find` —
/// reporting the full (possibly deeply-nested, possibly absolute) path root-first costs the model extra
/// tokens per line for a prefix it already knows (it's whatever `path` it just asked to search), and
/// diverges from pi's documented "relative to search directory" contract.
pub fn format_path<'a>(
    path: &'a std::path::Path,
    root: &std::path::Path,
) -> std::borrow::Cow<'a, str> {
    if !root.is_dir() {
        return match path.file_name() {
            Some(name) => name.to_string_lossy(),
            None => path.to_string_lossy(),
        };
    }
    path.strip_prefix(root).unwrap_or(path).to_string_lossy()
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

/// How long a spilled full-output temp file may sit before it's swept away as abandoned. Generous,
/// and deliberately much larger than `trust_store.rs`'s `STALE_LOCK_AGE`: that constant reclaims a
/// lock whose owner process almost certainly died (a trust-file write is milliseconds), whereas a
/// spill file's whole purpose is to let something (the model, in a later turn) come back and read the
/// complete output well after the tool call that created it returned — see `full_output_path` on
/// [`OutputSnapshot`]. A day is generous enough that no realistic legitimate reader loses the file out
/// from under it, while still bounding growth on a long-running unattended agent.
const STALE_TEMP_FILE_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Best-effort delete other `<prefix>-*.log` spill files older than [`STALE_TEMP_FILE_AGE`] from the
/// system temp directory. Nothing else in this codebase ever deletes a spill file — it's handed back
/// to the caller specifically so the full output can be read later — so without this sweep, a
/// long-running unattended agent (this crate's own deployment target) that runs enough truncated bash
/// commands accumulates one more `.log` file per call, forever. Run from [`ensure_temp_file`] right
/// before a new one is created: cheap to piggyback on there, and self-limiting since it only runs as
/// often as commands actually spill. Errors (an unreadable temp dir, a file that vanishes mid-scan,
/// one this process can't remove) are swallowed — matching this file's existing "cleanup is
/// best-effort, never fails the caller" convention (see `mark_spill_broken`) — since a missed sweep
/// just means the next one catches it.
fn sweep_stale_temp_files(prefix: &str) {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    let want_prefix = format!("{prefix}-");
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with(&want_prefix) || !name.ends_with(".log") {
            continue;
        }
        let is_stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .and_then(|modified| {
                SystemTime::now()
                    .duration_since(modified)
                    .map_err(std::io::Error::other)
            })
            .is_ok_and(|age| age > STALE_TEMP_FILE_AGE);
        if is_stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
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

    // The spill file's byte budget and what's been spent of it. `spill_capped` means the budget ran
    // out and the writer was closed: the file holds a prefix, and further bytes are counted (totals,
    // tail) but not written anywhere.
    spill_budget: u64,
    spill_bytes: u64,
    spill_capped: bool,

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
            spill_budget: MAX_SPILL_BYTES,
            spill_bytes: 0,
            spill_capped: false,
            finished: false,
        }
    }

    /// Builder-style: shrink the spill file's byte budget. Test-only — production always runs the
    /// real [`MAX_SPILL_BYTES`], and a test that had to actually emit 128 MiB to reach it would be
    /// writing that much to the same tmpfs the budget exists to protect.
    #[cfg(test)]
    fn with_spill_budget(mut self, budget: u64) -> Self {
        self.spill_budget = budget;
        self
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
            // Post-spill: stream straight to the file so memory stays bounded — up to the budget.
            self.write_spill(data);
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
            // Only meaningful while there's still a file to describe: a spill that later *broke* has
            // already dropped its path (`mark_spill_broken`), and a capped-then-broken file must read
            // as "no full output", not as a prefix a reader can trust.
            full_output_capped: self.spill_capped && self.temp_path.is_some(),
            full_output_bytes: self.spill_bytes,
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
    ///
    /// While the command is still running, any trailing bytes that look like an incomplete
    /// multi-byte UTF-8 sequence (a chunk boundary landing mid-character) are held back rather
    /// than lossily decoded to U+FFFD — mirroring pi's stateful `TextDecoder({stream: true})`,
    /// which withholds an incomplete trailing sequence until the rest arrives in a later chunk.
    /// Once [`finish`](Self::finish) has run there's no "more bytes coming" to wait for, so the
    /// whole tail — including any genuinely truncated/invalid trailing bytes — goes through the
    /// same lossy fallback as before.
    fn snapshot_text(&self) -> String {
        let decodable = if self.finished {
            &self.tail[..]
        } else {
            let held_back = incomplete_utf8_suffix_len(&self.tail);
            &self.tail[..self.tail.len() - held_back]
        };
        let s = String::from_utf8_lossy(decodable).into_owned();
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
        // Sweep abandoned spill files from earlier commands before adding a new one — see
        // `sweep_stale_temp_files`'s doc comment for why nothing else ever cleans these up.
        sweep_stale_temp_files(&self.temp_prefix);
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
                self.temp_path = Some(path);
                self.temp_writer = Some(BufWriter::new(f));
                self.spilled = true;
                // The buffered bytes go in through the same budgeted write as every later chunk —
                // one place decides what the file is allowed to hold, so the flush of everything
                // seen before the spill can't slip past the budget by writing itself whole.
                let buffered = std::mem::take(&mut self.raw_buffer);
                self.write_spill(&buffered);
            }
            Err(_) => {
                self.spill_failed = true;
                self.raw_buffer = Vec::new();
            }
        }
    }

    /// Write `data` to the spill file, spending it against the file's byte budget and writing nothing
    /// once that budget is gone — see [`MAX_SPILL_BYTES`] for why an unbounded spill file is a memory
    /// leak on this deployment target, not merely a disk one.
    ///
    /// Reaching the budget is deliberately *not* [`mark_spill_broken`](Self::mark_spill_broken): a
    /// broken file is one whose contents no longer correspond to the command's output at all, so it
    /// must stop being advertised; a budget-capped file is a genuine, complete prefix, still the best
    /// thing the model can go read. What it must never do is masquerade as the whole document, hence
    /// `spill_capped` — the flag [`format_output`] turns into an explicit "this file stops early"
    /// marker. Everything else about the accumulator carries on unchanged past the budget: the running
    /// totals and rolling tail are what make the truncation metadata honest about the bytes we dropped.
    fn write_spill(&mut self, data: &[u8]) {
        let remaining = self.spill_budget.saturating_sub(self.spill_bytes);
        let take = remaining.min(data.len() as u64) as usize;
        let Some(w) = self.temp_writer.as_mut() else {
            return;
        };
        if take > 0 && w.write_all(&data[..take]).is_err() {
            self.mark_spill_broken();
            return;
        }
        self.spill_bytes += take as u64;
        if self.spill_bytes >= self.spill_budget {
            self.close_spill_at_budget();
        }
    }

    /// Retire the spill writer once its byte budget is spent: flush what's buffered so the file on
    /// disk is complete up to the budget, then drop the writer so the fd is released and no later
    /// chunk can reopen the tap. `temp_path` stays — the prefix is still readable, and `spill_capped`
    /// is what tells the caller it stops short.
    fn close_spill_at_budget(&mut self) {
        if let Some(mut w) = self.temp_writer.take() {
            if w.flush().is_err() {
                self.mark_spill_broken();
                return;
            }
        }
        self.spill_capped = true;
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
///
/// No `truncate_head` (pi's opposite-end counterpart, `truncate.ts`) exists here, deliberately: every
/// caller in this crate wants the *end* of a stream (`bash`'s live/final output — errors and results
/// live at the tail of a long-running command — via [`OutputAccumulator::snapshot`], the only caller of
/// this function) or does its own byte-capping a different way entirely (`grep`/`find`/`ls`'s
/// [`cap_listing_bytes`], a straight truncate-the-whole-assembled-string-and-append-a-marker, not a
/// line-aware head/tail split). `read`'s own windowed `offset`/`limit` reads (see that module's doc
/// comment on `MAX_LINE_BYTES`) solve the same "don't return partial lines" problem `truncate_head`
/// exists for, but via streaming line-at-a-time capping rather than truncating an already-assembled
/// string — so there's no genuine head-truncation need anywhere in this codebase to port it for.
fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> TailTruncation {
    let total_bytes = content.len(); // Rust `str` length is already the UTF-8 byte length.

    // A trailing '\n' yields a spurious empty final element from `split`/`rsplit` — drop it so "a\nb\n"
    // is two lines, matching a bare "a\nb" (no accompanying trailing element to skip).
    let ends_with_nl = content.ends_with('\n');
    let mut total_lines = content.split('\n').count();
    if ends_with_nl {
        total_lines -= 1;
    }

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
    // `rsplit` walks from the end lazily — unlike collecting every line into a `Vec` first (the prior
    // approach), this never materializes more than `collected`'s own bounded (`<= max_lines`) entries,
    // even when `content` holds far more lines than either budget allows.
    let mut collected: Vec<&str> = Vec::new(); // in reverse order
    let mut output_bytes_count: usize = 0;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;
    let mut partial: Option<String> = None;

    let mut rev_lines = content.rsplit('\n');
    if ends_with_nl {
        rev_lines.next(); // skip the same spurious trailing empty element `total_lines` already dropped
    }
    for line in rev_lines {
        if collected.len() >= max_lines {
            break;
        }
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

/// Number of trailing bytes in `bytes` that form an incomplete-but-potentially-valid UTF-8
/// sequence — a multi-byte lead byte whose continuation bytes haven't all arrived yet. Returns 0
/// when `bytes` is already valid UTF-8, or when the first error is a genuinely invalid sequence
/// (not simply truncated): either way there's nothing left to wait for, so the caller should just
/// decode the bytes as they are. `std::str::from_utf8`'s own error already distinguishes the two
/// cases exactly (`error_len() == None` means "ran out of input mid-sequence"), so there's no need
/// to hand-roll continuation-byte counting.
fn incomplete_utf8_suffix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => 0,
        Err(e) => match e.error_len() {
            None => bytes.len() - e.valid_up_to(),
            Some(_) => 0,
        },
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

/// The `Full output: <path>` tail of a truncation marker — pi's string verbatim in the ordinary case,
/// where the file really does hold everything the command printed.
///
/// When the spill file hit [`MAX_SPILL_BYTES`] it holds only the first N bytes of a much larger
/// stream, and the marker has to say so: a bare `Full output: <path>` invites the model to read that
/// file and reason about a firehose's opening moments as if they were the whole story — a silently
/// partial document presented as complete, which is a worse failure than admitting the bytes are gone.
/// The totals in the same marker (`of {total_lines}`) already describe the *complete* stream, so this
/// segment names what's actually on disk against it.
fn full_output_segment(snapshot: &OutputSnapshot) -> String {
    // pi interpolates the path directly; when there is none we render an empty string rather than the
    // literal "undefined" JS would produce.
    let path = snapshot.full_output_path.as_deref().unwrap_or("");
    if snapshot.full_output_capped {
        // Sized from the snapshot, not from `MAX_SPILL_BYTES`: the marker must state the bytes that
        // are actually in the file the model is about to read, whatever budget produced them.
        format!(
            "Full output too large to save — only the first {} of {} was written: {path}",
            format_size(snapshot.full_output_bytes),
            format_size(snapshot.truncation.total_bytes),
        )
    } else {
        format!("Full output: {path}")
    }
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
        let full = full_output_segment(snapshot);

        let marker = if t.last_line_partial {
            format!(
                "[Showing last {} of line {} (line is {}). {full}]",
                format_size(t.output_bytes),
                end_line,
                format_size(snapshot.last_line_bytes),
            )
        } else if matches!(t.truncated_by, Some(TruncatedBy::Lines)) {
            format!(
                "[Showing lines {start_line}-{end_line} of {}. {full}]",
                t.total_lines
            )
        } else {
            format!(
                "[Showing lines {start_line}-{end_line} of {} ({} limit). {full}]",
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
    fn ensure_temp_file_sweeps_stale_siblings_but_leaves_fresh_ones() {
        // Regression test for the safety-audit finding: nothing in the codebase ever deleted a
        // spilled full-output temp file, so a long-running unattended agent (this crate's own
        // deployment target) accumulates one `pi-bash-*.log` per truncated command, forever.
        // `ensure_temp_file` now sweeps stale siblings (older than `STALE_TEMP_FILE_AGE`) before
        // creating a new one. A fresh sibling — e.g. one from a command that just finished, whose
        // `full_output_path` a caller may still legitimately be about to read — must survive.
        let prefix = "test-stale-sweep";
        let dir = std::env::temp_dir();
        let stale_path = dir.join(format!("{prefix}-aaaaaaaaaaaaaaaa.log"));
        let fresh_path = dir.join(format!("{prefix}-bbbbbbbbbbbbbbbb.log"));

        std::fs::write(&stale_path, b"old full output").unwrap();
        std::fs::write(&fresh_path, b"recent full output").unwrap();

        // Backdate only the stale file's mtime, well past the sweep's age threshold — mirrors
        // `trust_store.rs`'s `a_stale_lock_file_is_reclaimed_rather_than_blocking_forever` test,
        // which backdates the same way via `File::open(..).set_modified(..)`.
        let long_ago = SystemTime::now() - (STALE_TEMP_FILE_AGE + Duration::from_secs(60));
        File::open(&stale_path)
            .unwrap()
            .set_modified(long_ago)
            .unwrap();

        // Drive a real spill under this prefix — the same `append` → `ensure_temp_file` path a
        // truncated bash command takes — so the sweep is exercised exactly where it runs in
        // production, not via a direct call to a test-only hook.
        let mut acc = OutputAccumulator::with_prefix(prefix);
        for n in 0..300u32 {
            let mut line = format!("chunk-{n:04}-").into_bytes();
            line.resize(1023, b'x');
            line.push(b'\n');
            acc.append(&line);
        }
        acc.finish();
        let snap = acc.snapshot(true);
        let new_path = snap.full_output_path.clone().expect("must spill");

        assert!(
            !stale_path.exists(),
            "a stale sibling temp file must be swept on the next spill"
        );
        assert!(
            fresh_path.exists(),
            "a fresh sibling temp file must survive the sweep"
        );

        let _ = std::fs::remove_file(&fresh_path);
        let _ = std::fs::remove_file(&new_path);
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

    /// Feed `total` bytes of newline-terminated output through `acc` in 8 KiB chunks (the shape a
    /// firehose actually arrives in), with the final line ending in `END` so a tail assertion can
    /// prove the accumulator kept working past whatever the spill file did.
    fn feed_firehose(acc: &mut OutputAccumulator, total: usize) {
        let chunk = vec![b'f'; 8 * 1024];
        let mut written = 0usize;
        while written + chunk.len() <= total {
            acc.append(&chunk);
            written += chunk.len();
        }
        acc.append(b"\nEND\n");
    }

    #[test]
    fn output_under_the_spill_budget_is_written_to_the_file_in_full() {
        // The budget is a backstop, not a haircut: anything short of it must still land on disk
        // byte-for-byte, with no "capped" admission attached to a file that is in fact complete.
        let mut acc = OutputAccumulator::with_prefix("pi-bash").with_spill_budget(1024 * 1024);
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
        assert!(!snap.full_output_capped);
        assert_eq!(snap.full_output_bytes, full.len() as u64);
        assert_eq!(read_temp(&path), full);
        // The ordinary marker, unchanged — an uncapped file says nothing about budgets.
        let out = format_output(&snap, "(no output)");
        assert!(out.contains(&format!("Full output: {path}]")), "got: {out}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_firehose_past_the_spill_budget_stops_growing_the_file() {
        // The leak this guards: `DEFAULT_MAX_BYTES` caps only what the model is *shown*, so before the
        // budget existed the spill file grew for as long as the command talked — and the system temp
        // dir on this crate's deployment target is a tmpfs, so an unbounded spill file is unbounded
        // *RAM* (see `MAX_SPILL_BYTES`). `bash: yes` under the 30-minute default timeout was enough.
        let budget = 256 * 1024;
        let mut acc = OutputAccumulator::with_prefix("pi-bash").with_spill_budget(budget as u64);
        feed_firehose(&mut acc, 8 * 1024 * 1024); // 32x the budget
        acc.finish();
        let snap = acc.snapshot(true);

        let path = snap.full_output_path.clone().expect("must spill");
        let on_disk = std::fs::metadata(&path).unwrap().len();
        assert!(
            on_disk <= budget as u64,
            "spill file must never exceed its budget: {on_disk} > {budget}"
        );
        // And it must actually have written up to the budget, not bailed out at the first chunk.
        assert_eq!(on_disk, budget as u64);
        assert_eq!(snap.full_output_bytes, budget as u64);
        assert!(snap.full_output_capped);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncation_metadata_stays_honest_after_the_spill_budget_is_hit() {
        // Stopping the writes must not stop the accumulator: the tail the model reads, the totals it
        // reasons about, and the marker that admits how much was dropped all have to survive the
        // budget — otherwise a capped firehose would present its opening 256 KiB as the whole story.
        let budget = 256 * 1024usize;
        let total = 8 * 1024 * 1024usize;
        let mut acc = OutputAccumulator::with_prefix("pi-bash").with_spill_budget(budget as u64);
        feed_firehose(&mut acc, total);
        acc.finish();
        let snap = acc.snapshot(true);

        // Totals cover the COMPLETE stream, not the bytes that made it to disk. `feed_firehose` writes
        // whole 8 KiB chunks up to `total` and then "\nEND\n" (5 bytes, 2 newlines → 2 lines: the long
        // 'f' run it terminates, and "END").
        assert_eq!(snap.truncation.total_bytes, (total + 5) as u64);
        assert_eq!(snap.truncation.total_lines, 2);
        assert!(snap.truncation.truncated);
        assert_eq!(snap.truncation.truncated_by, Some(TruncatedBy::Bytes));
        assert!(snap.truncation.output_bytes <= DEFAULT_MAX_BYTES as u64);

        // The tail still tracks bytes that arrived long after the file stopped taking them. (The 'f'
        // run ahead of it is dropped by `snapshot_text`, which never shows a half-line at the top of
        // the window — the rolling tail landed mid-line, as it must for a stream this size.)
        assert_eq!(
            snap.content.trim_end(),
            "END",
            "the rolling tail must keep working past the budget"
        );

        // And the rendered marker says plainly that the saved file stops short — the model must not
        // read a 256 KiB prefix believing it's the complete 8 MiB output.
        let path = snap.full_output_path.clone().expect("must spill");
        let out = format_output(&snap, "(no output)");
        assert!(
            out.contains("only the first 256.0KB of 8.0MB was written"),
            "the marker must quantify what was dropped: {out}"
        );
        assert!(
            out.contains(&path),
            "the prefix is still worth reading: {out}"
        );
        assert!(
            !out.contains("Full output: "),
            "a capped file must not be advertised as the full output: {out}"
        );

        let _ = std::fs::remove_file(&path);
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
    fn cap_listing_bytes_never_cuts_a_line_in_half() {
        // 98-byte lines (97 chars + '\n'): MAX_LISTING_BYTES (51200) is not a multiple of 98
        // (51200 % 98 == 44), so the raw byte cut lands 44 bytes into a line — exactly the mid-line
        // truncation this test guards against.
        assert_ne!(
            MAX_LISTING_BYTES % 98,
            0,
            "test setup must land mid-line at the raw byte cap"
        );
        let line = "x".repeat(97);
        let mut out = String::new();
        for _ in 0..600 {
            out.push_str(&line);
            out.push('\n');
        }
        assert!(
            out.len() > MAX_LISTING_BYTES,
            "test setup: listing must exceed the cap"
        );

        let truncated = cap_listing_bytes(&mut out, "narrow the search");
        assert!(truncated);

        let marker_start = out.find("[output truncated").expect("marker present");
        let kept = out[..marker_start]
            .strip_suffix("\n\n")
            .unwrap_or(&out[..marker_start]);

        assert!(
            kept.is_empty() || kept.ends_with('\n'),
            "kept listing must end on a full line, not mid-line: {:?}",
            &kept[kept.len().saturating_sub(120)..]
        );
        // Every kept line must be a genuine, complete 97-'x' line — not a truncated fragment of one.
        // `kept` ends with '\n' (it's whole lines only), so strip it before splitting or the final
        // split element would be a spurious empty string, not a real line.
        if !kept.is_empty() {
            for l in kept.strip_suffix('\n').unwrap_or(kept).split('\n') {
                assert_eq!(l, line, "a partial/fabricated-looking line leaked through");
            }
        }
    }

    #[test]
    fn max_listing_bytes_stays_in_sync_with_default_max_bytes() {
        // Pi-parity fix (task 51): `MAX_LISTING_BYTES` (ls/grep/find's one-shot output cap) and
        // `DEFAULT_MAX_BYTES` (bash/read's streaming truncation budget) used to be two independently
        // hardcoded `50 * 1024` literals that agreed only by coincidence — changing one without the
        // other would silently split the two budgets apart with nothing to catch it. Pi has a single
        // shared `DEFAULT_MAX_BYTES` (`truncate.ts`) for both. This test doesn't just restate the
        // `const MAX_LISTING_BYTES: usize = DEFAULT_MAX_BYTES` alias (a compiler error would already
        // catch a typo there) — it pins the *value* the two must share, so a future edit that turns
        // the alias back into a second independent literal (reintroducing the drift) fails here even
        // if both literals happen to still parse.
        assert_eq!(MAX_LISTING_BYTES, DEFAULT_MAX_BYTES);
        assert_eq!(MAX_LISTING_BYTES, 50 * 1024);
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
            full_output_capped: false,
            full_output_bytes: 12345,
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

    // Multi-byte UTF-8 boundary coverage for `truncate_tail`/`truncate_str_to_bytes_from_end` — pi:
    // `packages/agent/test/harness/truncate.test.ts`. The single existing test above is ASCII-only, so
    // none of these boundary-splitting cases were previously exercised at all: truncating mid-codepoint
    // must never produce invalid UTF-8 (a `&str` slice on a non-boundary index panics outright, so any
    // of these would fail loudly rather than silently corrupt output — but only if something actually
    // runs them).

    #[test]
    fn truncate_tail_keeps_a_whole_multi_byte_character_and_drops_the_rest_of_the_line() {
        // pi: truncate.test.ts, "truncates tail on UTF-8 boundaries when only a partial last line
        // fits" — `"aé🙂b"` is a *single* line (no newline), 8 bytes total (a=1, é=2, 🙂=4, b=1). At a
        // 5-byte budget the byte cap bites on the one oversized line; the kept suffix must land exactly
        // on the 🙂/b boundary, not split the 4-byte emoji.
        let r = truncate_tail("aé🙂b", 10, 5);
        assert_eq!(r.content, "🙂b");
        assert_eq!(r.truncated_by, Some(TruncatedBy::Bytes));
        assert!(r.last_line_partial);
        assert_eq!(r.output_bytes, 5);
    }

    #[test]
    fn truncate_tail_drops_a_trailing_multi_byte_character_that_cannot_fit_the_byte_budget() {
        // pi: truncate.test.ts, "drops an oversized trailing character when it cannot fit in tail byte
        // limit" — `"abc🙂"` at a 3-byte budget: the last 3 bytes of the 7-byte line would land inside
        // the 4-byte 🙂 (bytes 3..7), which isn't a valid boundary to cut on, so the boundary-skipping
        // walk in `truncate_str_to_bytes_from_end` advances past the *entire* character, correctly
        // yielding an empty fragment rather than a truncated, invalid one.
        let r = truncate_tail("abc🙂", 10, 3);
        assert_eq!(r.content, "");
        assert_eq!(r.truncated_by, Some(TruncatedBy::Bytes));
        assert!(r.last_line_partial);
        assert_eq!(r.output_bytes, 0);
    }

    #[test]
    fn truncate_tail_caps_an_oversized_single_line_with_a_trailing_newline_on_a_char_boundary() {
        // pi: truncate.test.ts, "truncates an oversized single line with a trailing newline" — an
        // ASCII fixture at pi's own scale (300_000 bytes), confirming the single-huge-line path (also
        // exercised with multi-byte content by the two tests above) handles a large fragment correctly
        // too, not just a handful of bytes.
        let input = format!("{}\n", "X".repeat(300_000));
        let r = truncate_tail(&input, 100, 1024);
        assert_eq!(r.content, "X".repeat(1024));
        assert_eq!(r.output_bytes, 1024);
        assert_eq!(r.output_lines, 1);
        assert!(r.last_line_partial);
        assert_eq!(r.truncated_by, Some(TruncatedBy::Bytes));
    }

    #[test]
    fn truncate_tail_multi_byte_content_within_budget_is_not_truncated() {
        // The mirror image of the ASCII-only no-truncation test above: multi-byte content that fits
        // entirely within both limits must come back completely unchanged, not just avoid a crash.
        let content = "café 🙂 中文\n второй line\n";
        let r = truncate_tail(content, 2000, 50 * 1024);
        assert!(r.truncated_by.is_none());
        assert_eq!(r.content, content);
        assert!(!r.last_line_partial);
    }

    #[test]
    fn live_snapshot_holds_back_an_incomplete_trailing_utf8_sequence() {
        // pi: TextDecoder({stream: true}) withholds a trailing partial multi-byte sequence until
        // the rest arrives — a live snapshot taken right after a chunk boundary splits a
        // character mid-sequence must show clean, complete text, not a U+FFFD placeholder.
        let mut acc = OutputAccumulator::new();
        acc.append(b"hi ");
        let emoji = "🙂".as_bytes(); // 4 bytes: F0 9F 99 82
        acc.append(&emoji[..2]); // first half only — an incomplete lead sequence

        let mid_snap = acc.snapshot(false);
        assert!(
            !mid_snap.content.contains('\u{FFFD}'),
            "a live snapshot must not show a replacement character for bytes still in flight: {:?}",
            mid_snap.content
        );
        assert_eq!(mid_snap.content, "hi ");

        // The rest of the character arrives — the next snapshot shows it whole.
        acc.append(&emoji[2..]);
        let healed_snap = acc.snapshot(false);
        assert_eq!(healed_snap.content, "hi 🙂");

        acc.finish();
        let final_snap = acc.snapshot(false);
        assert_eq!(final_snap.content, "hi 🙂");
    }

    #[test]
    fn finished_snapshot_lossily_decodes_a_genuinely_truncated_tail() {
        // Once the stream is finished there's no more data coming, so an incomplete trailing
        // sequence left in the tail is a real truncation, not an in-flight chunk boundary — it
        // must fall back to the existing lossy decode (U+FFFD) rather than being held back
        // forever.
        let mut acc = OutputAccumulator::new();
        acc.append(b"hi ");
        acc.append(&"🙂".as_bytes()[..2]); // never completed
        acc.finish();

        let snap = acc.snapshot(false);
        assert!(snap.content.contains('\u{FFFD}'), "got: {:?}", snap.content);
    }

    #[test]
    fn truncate_str_to_bytes_from_end_never_splits_a_multi_byte_character() {
        // Direct coverage of the helper itself across every cut point through a run of 3-byte
        // characters (`中`) — for every `max_bytes` from 0 to just past the string's total length, the
        // result must (a) never exceed the budget and (b) be a valid `&str` (guaranteed by the type
        // system here: a non-boundary slice would have already panicked before `assert!` even runs).
        let s = "中文测试"; // 4 characters, 3 bytes each = 12 bytes total
        for max_bytes in 0..=(s.len() + 2) {
            let got = truncate_str_to_bytes_from_end(s, max_bytes);
            assert!(
                got.len() <= max_bytes,
                "max_bytes={max_bytes} got {} bytes: {got:?}",
                got.len()
            );
            assert!(
                s.ends_with(got),
                "result must be a genuine suffix of the input: {got:?}"
            );
        }
    }
}
