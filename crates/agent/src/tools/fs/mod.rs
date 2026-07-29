//! `FsBackend` — the seam between a filesystem tool and where its I/O actually lands.
//!
//! The tools in this crate were welded to the host: `read`/`write`/`edit`/`ls`/`grep`/`find` called
//! `std::fs` directly, while `bash` and the `beyond` tools already went through
//! [`CommandRunner`](super::exec::CommandRunner) — the one place the crate admitted that "where does
//! this run" is a decision worth naming. This module is that same admission for the other half.
//!
//! Two implementations, both selected by the caller and held behind an `Arc`:
//!
//! - [`local::LocalFs`] — `std::fs` and ripgrep's in-process walker. The behavior that shipped before
//!   this seam existed, moved rather than rewritten.
//! - [`shell::ShellFs`] — the same operations expressed as *commands*, over any `CommandRunner`. It
//!   works against a box that has nothing of ours installed on it.
//!
//! **Most of a tool's behavior is not in the backend.** The split is deliberately drawn so that only
//! the irreducibly filesystem-shaped part (walk a tree, match lines, read bytes) crosses the seam;
//! everything downstream of that — line clipping, EOL normalization, sorting, limit trimming, and all
//! rendering — stays shared, in this module or in the tool. That is what makes the two impls agree on
//! the fiddly details for free instead of by vigilance: there is only one `clip`, one `trim_eol`, one
//! [`finalize`], and both backends call them.

pub mod local;
pub mod shell;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::ToolError;
use async_trait::async_trait;

/// Long match lines are clipped to keep output readable — a *character* count, matching pi's own
/// `GREP_MAX_LINE_LENGTH` (`truncateLine`'s `line.length`, UTF-16 code units — effectively chars for all
/// BMP text), not a byte count: a CJK/Cyrillic/accented-Latin line costs more bytes per character than
/// ASCII, so byte-clipping the same cap would truncate it to far fewer visible characters for no reason.
pub const MAX_LINE: usize = 500;

/// Hard ceiling on matches collected before a search bails — an OOM guard for pathological patterns
/// (matching nearly every line of a huge tree). Far above any sane `limit`; when it trips, the
/// surviving subset can depend on scheduling, which the output flags.
pub const HARD_CAP: usize = 10_000;

/// Ceiling on the `before`/`after` context window. Context lines aren't bounded by [`HARD_CAP`] (which
/// counts matches), so clamp them to keep a huge `after` on a match-dense file from ballooning memory.
pub const MAX_CONTEXT: usize = 100;

/// Suffix [`clip`] appends to a line it truncates. `grep`'s `run` checks surviving hit text for this
/// suffix to decide whether to append an aggregate "some lines truncated" notice (matching pi's own
/// grep, which surfaces that as an actionable notice alongside the match-limit/byte-cap ones) — a
/// single source of truth instead of threading a separate "was this line clipped" flag through
/// [`Hit`] just for one summary line.
pub const LINE_TRUNCATED_SUFFIX: &str = "… [truncated]";

/// Which filesystem a path belongs to — and therefore whether host syscalls may be used to reason
/// about it.
///
/// This is not decoration. Path resolution has exactly one host-dependent step (`canonicalize`) and
/// one user-dependent step (`~` expansion), and both silently produce a *plausible wrong answer*
/// rather than an error when applied to a path that isn't on this machine. Making the world an
/// explicit parameter means the choice is visible at every call site instead of being an emergent
/// property of whether a syscall happened to fail. See
/// [`write_key`](crate::tools::write_key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathWorld<'a> {
    /// This host. `canonicalize` resolves real symlinks and `~` means this process's `$HOME`.
    Local,
    /// Somewhere else. Resolution stays lexical, and `~` means the *target's* home — `None` when it
    /// isn't known, which leaves a leading `~` untouched rather than guessing.
    Remote { home: Option<&'a str> },
}

/// One reported line: its path, line number, text, and whether it is a match (vs a context line). The
/// path is an `Arc<Path>` so a file with many matches allocates the path **once** and each hit is a
/// refcount bump, not a fresh `PathBuf` per hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    pub path: Arc<Path>,
    pub line: usize,
    pub text: String,
    /// `true` for a matched line, `false` for a `before`/`after` context line.
    pub is_match: bool,
}

/// A prepared search, in terms a *shell* can also express — which is the whole constraint on its
/// shape. The pattern and glob are carried as **strings**, not as a compiled `RegexMatcher` and
/// `GlobMatcher`: those are ripgrep types that only [`local::LocalFs`] could ever hold, and putting
/// them here would have made the trait un-implementable by anything else. Each backend compiles (or
/// forwards) them itself, so a bad pattern surfaces from the backend rather than at parse time.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The regular expression, or a literal string when `literal` is set.
    pub pattern: String,
    /// Search for `pattern` verbatim (regex-escaped) rather than as a regex.
    pub literal: bool,
    /// Fold case.
    pub ignore_case: bool,
    /// Restrict to files matching this glob, or — when the bool is `true` — *exclude* files matching
    /// it (ripgrep-CLI-style, from a leading `!` the caller has already stripped).
    pub glob: Option<(String, bool)>,
    /// The directory or file to search. Already resolved against the tool's root.
    pub root: PathBuf,
    /// Max *matches* to report. Context lines don't count toward it.
    pub limit: usize,
    /// Lines of context before each match (ripgrep's `-B`), clamped to [`MAX_CONTEXT`].
    pub before: usize,
    /// Lines of context after each match (ripgrep's `-A`), clamped to [`MAX_CONTEXT`].
    pub after: usize,
}

impl SearchQuery {
    /// The point at which a search may stop collecting — `limit` in the common case (a low-limit
    /// query against a match-dense tree should stop almost immediately, not scan the whole tree only
    /// to throw away everything past the first `limit` matches), [`HARD_CAP`] as the outer ceiling
    /// when a caller passes an unusually large `limit`.
    pub fn stop_at(&self) -> usize {
        self.limit.min(HARD_CAP)
    }
}

/// What a search produced: hits sorted by `(path, line)`, whether the result was cut short, and — when
/// at least one path couldn't be read (permission denied, a broken symlink, …) — the first such
/// error's message. Matching real ripgrep, which exits non-zero and reports the real error text the
/// moment it can't read *any* path in the tree, rather than silently treating it as "not found here".
#[derive(Debug, Default)]
pub struct SearchOutcome {
    pub hits: Vec<Hit>,
    pub truncated: bool,
    pub first_error: Option<String>,
}

/// Why a backend operation failed. Two variants, mapped 1:1 onto the two [`ToolError`] kinds a
/// filesystem tool can produce, each carrying its message already formatted — so the text the model
/// reads is identical whichever backend produced it.
#[derive(Debug)]
pub enum FsError {
    /// The query itself is malformed (a bad regex, a bad glob). The caller's mistake.
    InvalidQuery(String),
    /// The backend couldn't complete the operation. Not the caller's mistake.
    Backend(String),
}

impl From<FsError> for ToolError {
    fn from(e: FsError) -> Self {
        match e {
            FsError::InvalidQuery(m) => ToolError::InvalidInput(m),
            FsError::Backend(m) => ToolError::Execution(m),
        }
    }
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FsError::InvalidQuery(m) | FsError::Backend(m) => f.write_str(m),
        }
    }
}

/// What kind of thing a path is. The `Other` bucket exists because `read`/`write`/`edit` must refuse
/// FIFOs, sockets and devices *before* opening them: opening a FIFO blocks until a peer appears, and a
/// character device never signals EOF, and neither has a timeout or a kill-on-drop the way a `bash`
/// child does — so one wedges the whole turn, unrecoverably, from inside a blocking syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    File,
    Dir,
    /// A FIFO, socket, or device — never safe for these tools to open.
    Other,
}

/// One path's metadata. Deliberately one struct rather than several accessors: on a remote filesystem
/// each field would otherwise cost its own round trip, and the tools want two or three of them at once.
#[derive(Debug, Clone)]
pub struct Meta {
    pub kind: FileKind,
    pub len: u64,
    /// Last-modified time, used by `edit` as the compare-and-swap baseline for its guarded write.
    pub mtime: Option<std::time::SystemTime>,
    /// Whether *this caller* can actually write the path — a real access check, not an inspection of
    /// the mode bits. A file owned by another uid can have write bits set for its owner and still be
    /// unwritable by us, so a pre-check built on `Permissions::readonly()` passes right before the
    /// real write fails with a generic OS error.
    pub writable: bool,
}

/// One directory entry, as `ls` needs it. The name is a single path component, never a full path.
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: FileKind,
    pub len: u64,
}

/// A prepared `find`. Like [`SearchQuery`], it carries the glob as a **string** so a backend that
/// shells out can pass it through rather than needing a compiled `globset` matcher it has no way to
/// apply.
#[derive(Debug, Clone)]
pub struct GlobQuery {
    pub pattern: String,
    /// Match the pattern against each entry's basename rather than its full path — what `find` does
    /// when the pattern contains no `/`.
    pub basename_only: bool,
    /// Fold case in the glob match.
    pub case_insensitive: bool,
    pub root: PathBuf,
    pub limit: usize,
}

/// What a glob walk produced.
///
/// Each entry carries whether it is a directory, because `find` matches **directories as well as
/// files** — `find "node_modules"` returning nothing would be surprising — and renders a match with a
/// trailing `/` when it is one, the same signal real `fd` gives.
#[derive(Debug, Default)]
pub struct GlobOutcome {
    /// `(path, is_dir)`, sorted, already trimmed to the query's limit by [`finalize_glob`].
    pub paths: Vec<(PathBuf, bool)>,
    pub truncated: bool,
    /// First unreadable path encountered, so an empty result can say "couldn't fully search" rather
    /// than a confidently wrong "no files matching".
    pub first_error: Option<String>,
}

/// Sort, decide truncation, and trim — [`finalize`]'s counterpart for globbing, shared for the same
/// reason: the ordering and the "was this cut short" rule must not depend on which backend walked.
pub fn finalize_glob(
    mut paths: Vec<(PathBuf, bool)>,
    limit: usize,
    hit_hard_cap: bool,
) -> (Vec<(PathBuf, bool)>, bool) {
    paths.sort();
    let truncated = paths.len() > limit || hit_hard_cap;
    paths.truncate(limit);
    (paths, truncated)
}

/// Where a filesystem tool's I/O lands.
///
/// Deliberately narrow: every method here is an operation that *cannot* be composed from the others.
/// `edit`, notably, gets no read or write method of its own — it is [`read_bytes`](FsBackend::read_bytes)
/// plus the existing match logic plus [`write_if_unchanged`](FsBackend::write_if_unchanged), which is
/// why it has no matching behavior to drift.
#[async_trait]
pub trait FsBackend: Send + Sync {
    /// Search file contents. Returns hits already sorted, clipped, and trimmed to `limit` — see
    /// [`finalize`], which both implementations use so this contract holds by construction.
    async fn search(&self, q: &SearchQuery) -> Result<SearchOutcome, FsError>;

    /// Metadata for one path, or `None` if it does not exist.
    ///
    /// Missing is `Ok(None)` rather than an error because every caller treats it as an ordinary
    /// branch — `write` creates the file, `read` wants its own "no such file" wording — and forcing
    /// them to pattern-match an error kind to find that out invites treating a *permissions* failure
    /// as "absent", which is how a pre-check silently stops checking anything.
    async fn stat(&self, path: &Path) -> Result<Option<Meta>, FsError>;

    /// Read at most `max` bytes starting at byte `offset`.
    ///
    /// A short read means end-of-file — which is what lets a caller stream a file of unknown size in
    /// bounded chunks (`read` counts a huge file's total lines this way) without ever holding it whole.
    async fn read_bytes(&self, path: &Path, offset: u64, max: usize) -> Result<Vec<u8>, FsError>;

    /// Replace a file's entire contents atomically — a concurrent reader, or a crash mid-write, must
    /// observe either the old file or the complete new one, never a partial.
    async fn write_bytes(&self, path: &Path, bytes: &[u8]) -> Result<(), FsError>;

    /// Replace a file's contents **only if** its mtime still equals `expected` — `edit`'s guard against
    /// clobbering a change made between reading the file and writing it back. Returns `false` when the
    /// file moved on and nothing was written.
    ///
    /// This is one method rather than a stat-then-write by the caller because the check and the write
    /// have to be as close together as the backend can make them; splitting it across the trait would
    /// put a guaranteed round trip inside the race window it exists to narrow.
    async fn write_if_unchanged(
        &self,
        path: &Path,
        bytes: &[u8],
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, FsError>;

    /// Create a directory and any missing parents. Succeeds if it already exists.
    async fn create_dir_all(&self, path: &Path) -> Result<(), FsError>;

    /// List one directory, stopping after `cap` **kept** entries.
    ///
    /// `include_hidden` is applied by the backend, before the cap, because that is where the existing
    /// local behavior applies it: `ls`'s hard cap counts entries it kept, not entries it saw. Doing
    /// the filtering in the caller instead would silently change how much of a 10,000-entry directory
    /// full of dotfiles actually gets listed. It is also the right place remotely, where filtering at
    /// the source is strictly less data to move.
    async fn list_dir(
        &self,
        path: &Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<DirEntry>, FsError>;

    /// Walk a tree collecting paths matching a glob, honoring `.gitignore`.
    async fn glob(&self, q: &GlobQuery) -> Result<GlobOutcome, FsError>;

    /// Which filesystem this backend's paths live on.
    ///
    /// Defaults to [`PathWorld::Local`] so an implementation that genuinely is the host — including
    /// any test double — gets the right answer without stating it. A backend that is *not* the host
    /// must override this, because the default silently licenses host `canonicalize` and host `~`
    /// expansion on paths that belong to another machine.
    fn world(&self) -> PathWorld<'_> {
        PathWorld::Local
    }
}

/// Strip a trailing `\n` and then a trailing `\r` (a searcher hands us the line terminator), plus —
/// pi-parity fix — any further embedded `\r` the trailing strip doesn't reach. `grep-searcher` only
/// splits on `\n`, so a file using old-Mac-style bare-`\r` line endings (or one with stray corrupted
/// bytes) hands us "lines" that still carry mid-content `\r` characters; pi strips every `\r` in the
/// line, not just a trailing one. Borrows unchanged in the overwhelmingly common case (no embedded
/// `\r` at all) rather than allocating a `String` for every matched line.
///
/// Shared by both backends on purpose: `rg`'s stdout needs exactly the same normalization that
/// `grep-searcher`'s sink does, and two copies of this would drift on the first CRLF fixture.
pub fn trim_eol(s: &str) -> std::borrow::Cow<'_, str> {
    let s = s.strip_suffix('\n').unwrap_or(s);
    let s = s.strip_suffix('\r').unwrap_or(s);
    if s.contains('\r') {
        std::borrow::Cow::Owned(s.replace('\r', ""))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

/// Clip a long line to [`MAX_LINE`] *characters*, not bytes — pi-parity fix: the previous byte-based cap
/// (with a char-boundary backoff to avoid splitting a codepoint) never produced invalid UTF-8, but still
/// truncated a non-ASCII line to far fewer visible characters than an ASCII line under the same nominal
/// cap, unlike pi's own char-counting `truncateLine`.
///
/// Runs on every matched *and* every context line, so the common case (a line well under `MAX_LINE`,
/// which is most source lines) is the one to keep cheap: `char_indices` finds the byte offset of the
/// `MAX_LINE`-th char in one pass, and a line that doesn't reach it is returned via a single `to_string`
/// (a memcpy) rather than rebuilt char-by-char through a `Chars` iterator + `collect`. Only a line that
/// actually needs truncating pays for the slice + suffix, and even then as one allocation, not two.
pub fn clip(line: &str) -> String {
    match line.char_indices().nth(MAX_LINE) {
        None => line.to_string(), // fits within the cap — nothing was truncated
        Some((cut, _)) => {
            let mut clipped = String::with_capacity(cut + LINE_TRUNCATED_SUFFIX.len());
            clipped.push_str(&line[..cut]);
            clipped.push_str(LINE_TRUNCATED_SUFFIX);
            clipped
        }
    }
}

/// Sort, decide truncation, and trim to `limit` — the impl-independent tail of every search.
///
/// This is the largest single reason the two backends agree: sorting by `(path, line)`, counting only
/// matches toward `limit`, cutting at the match that would exceed it, and dropping the now-dangling
/// trailing context are all subtle enough to diverge if written twice, and none of it depends on how
/// the hits were found.
///
/// `stopped_early` is the backend's report that it stopped collecting at [`SearchQuery::stop_at`]
/// rather than exhausting the tree — a result can be truncated either because more matches existed
/// than `limit`, or because the backend bailed before finding them all.
pub fn finalize(mut hits: Vec<Hit>, limit: usize, stopped_early: bool) -> (Vec<Hit>, bool) {
    hits.sort_by(|a, b| a.path.cmp(&b.path).then(a.line.cmp(&b.line)));
    let match_total = hits.iter().filter(|h| h.is_match).count();
    let truncated = match_total > limit || stopped_early;
    // Keep at most `limit` matched lines (context lines don't count toward it). Walk counting matches,
    // cut at the match that would exceed the limit, then drop any now-dangling trailing context (the
    // before-context of the dropped match that trailed into the kept prefix). With no context
    // requested every hit is a match, so this reduces to a plain `truncate(limit)`.
    if match_total > limit {
        let mut seen = 0usize;
        let mut cut = hits.len();
        for (i, h) in hits.iter().enumerate() {
            if h.is_match {
                seen += 1;
                if seen > limit {
                    cut = i;
                    break;
                }
            }
        }
        hits.truncate(cut);
        while hits.last().is_some_and(|h| !h.is_match) {
            hits.pop();
        }
    }
    (hits, truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- trim_eol / clip: moved verbatim from `tools::grep`'s own test module along with the
    // functions they cover. Assertions are unchanged; only their address is.

    #[test]
    fn trim_eol_strips_trailing_newline_and_carriage_return() {
        assert_eq!(trim_eol("hello\r\n"), "hello");
        assert_eq!(trim_eol("hello\n"), "hello");
        assert_eq!(trim_eol("hello"), "hello");
    }

    #[test]
    fn trim_eol_strips_an_embedded_carriage_return_not_just_a_trailing_one() {
        // pi-parity fix (L6): `grep-searcher` only splits on `\n`, so a file using old-Mac-style bare
        // `\r` line endings hands us "lines" that still carry mid-content `\r` characters after the
        // trailing strip — pi strips every `\r` in the line, not just a trailing one.
        assert_eq!(trim_eol("first\rsecond\rthird\n"), "firstsecondthird");
    }

    #[test]
    fn trim_eol_borrows_when_there_is_nothing_to_strip() {
        // Not a behavior a caller can observe directly, but worth pinning: the common case (no
        // embedded `\r`) must not allocate.
        assert!(matches!(
            trim_eol("plain line"),
            std::borrow::Cow::Borrowed(_)
        ));
        assert!(matches!(
            trim_eol("first\rsecond"),
            std::borrow::Cow::Owned(_)
        ));
    }

    #[test]
    fn clip_leaves_a_short_line_untouched() {
        assert_eq!(clip("short line"), "short line");
    }

    #[test]
    fn clip_truncates_an_ascii_line_at_exactly_max_line_characters() {
        let line = "a".repeat(MAX_LINE + 50);
        let clipped = clip(&line);
        assert_eq!(
            clipped,
            format!("{}{LINE_TRUNCATED_SUFFIX}", "a".repeat(MAX_LINE))
        );
    }

    #[test]
    fn clip_counts_characters_not_bytes_for_non_ascii_text() {
        // Pi-parity fix: the previous byte-based cap truncated a non-ASCII line to far fewer visible
        // characters than an equivalent-length ASCII line under the same nominal `MAX_LINE` — pi's own
        // `truncateLine` counts characters (`line.length`, UTF-16 code units), not bytes. A 3-byte-per-
        // character CJK line at exactly `MAX_LINE` characters (so ~3x `MAX_LINE` bytes) must survive
        // whole, not get chopped down to a third of its characters.
        let line = "漢".repeat(MAX_LINE);
        assert!(
            line.len() > MAX_LINE * 2,
            "sanity: this line must be well over MAX_LINE bytes"
        );
        let clipped = clip(&line);
        assert_eq!(
            clipped, line,
            "exactly MAX_LINE characters must not be truncated, even though it's far more than \
             MAX_LINE bytes"
        );

        // One character over the cap must truncate to exactly MAX_LINE characters, not MAX_LINE bytes'
        // worth (which would be roughly a third as many characters for 3-byte-per-char text).
        let over = "漢".repeat(MAX_LINE + 1);
        let clipped_over = clip(&over);
        assert_eq!(
            clipped_over,
            format!("{}{LINE_TRUNCATED_SUFFIX}", "漢".repeat(MAX_LINE))
        );
    }

    #[test]
    fn clip_never_splits_a_multi_byte_character_even_though_it_counts_characters_now() {
        // A mixed-width line where the cut point (character MAX_LINE) falls squarely on a real
        // character boundary already (since we now count whole characters, not bytes) — this can never
        // regress into slicing mid-codepoint the way the old byte-index-with-boundary-backoff version
        // theoretically could if miscounted.
        let line = format!("{}{}", "a".repeat(MAX_LINE - 1), "漢漢漢");
        let clipped = clip(&line);
        assert!(clipped.is_char_boundary(clipped.len() - LINE_TRUNCATED_SUFFIX.len()));
        assert_eq!(
            clipped,
            format!(
                "{}{}{LINE_TRUNCATED_SUFFIX}",
                "a".repeat(MAX_LINE - 1),
                "漢"
            )
        );
    }

    // ---- finalize: new coverage for the shared tail, since it is now the single definition both
    // backends depend on rather than an inlined block inside one search function.

    fn hit(path: &str, line: usize, is_match: bool) -> Hit {
        Hit {
            path: Arc::from(Path::new(path)),
            line,
            text: format!("{path}:{line}"),
            is_match,
        }
    }

    #[test]
    fn finalize_sorts_by_path_then_line() {
        let (hits, _) = finalize(
            vec![
                hit("b.rs", 1, true),
                hit("a.rs", 9, true),
                hit("a.rs", 2, true),
            ],
            10,
            false,
        );
        let order: Vec<_> = hits
            .iter()
            .map(|h| (h.path.to_string_lossy().into_owned(), h.line))
            .collect();
        assert_eq!(
            order,
            vec![("a.rs".into(), 2), ("a.rs".into(), 9), ("b.rs".into(), 1)]
        );
    }

    #[test]
    fn finalize_counts_only_matches_toward_the_limit() {
        // Two matches, each with one context line either side. A limit of 2 must keep everything.
        let hits = vec![
            hit("a.rs", 1, false),
            hit("a.rs", 2, true),
            hit("a.rs", 3, false),
            hit("a.rs", 4, false),
            hit("a.rs", 5, true),
            hit("a.rs", 6, false),
        ];
        let (kept, truncated) = finalize(hits, 2, false);
        assert_eq!(kept.len(), 6);
        assert!(!truncated);
    }

    #[test]
    fn finalize_drops_dangling_context_after_cutting_at_the_limit() {
        // Limit 1: the second match is cut, and the context line that was leading into it must go
        // too — otherwise the output ends on a context line for a match the model never sees.
        let hits = vec![
            hit("a.rs", 2, true),
            hit("a.rs", 3, false),
            hit("a.rs", 4, false),
            hit("a.rs", 5, true),
        ];
        let (kept, truncated) = finalize(hits, 1, false);
        assert!(truncated);
        assert_eq!(kept.len(), 1);
        assert!(kept.last().is_some_and(|h| h.is_match));
    }

    #[test]
    fn finalize_reports_truncation_when_the_backend_stopped_early() {
        // Fewer hits than the limit, but the backend bailed — still truncated, because more matches
        // exist that were never collected.
        let (_, truncated) = finalize(vec![hit("a.rs", 1, true)], 100, true);
        assert!(truncated);
    }
}
