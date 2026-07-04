//! `grep` — regex search across files, gitignore-aware (ripgrep's `ignore` + `regex` crates).
//!
//! The walk runs in parallel (ripgrep's `WalkParallel`): worker threads read and scan files
//! concurrently — the difference between grep and ripgrep on a large tree. The walk quits as soon as
//! `limit` matches are found (not just at the much larger hard cap — see [`search`]'s doc comment), so
//! a low-limit query against a match-dense tree does work proportional to `limit`, not to the tree's
//! size. Whatever was collected before quitting is sorted by `(path, line)` and truncated, so an
//! **untruncated** result (every match found, nothing cut) is always fully deterministic; a
//! **truncated** one returns `limit` matches from wherever the parallel walk happened to reach before
//! quitting, not guaranteed to be the lexicographically-smallest ones — the same trade pi's own `grep`
//! makes by killing its `rg` child the instant it has enough matches. The blocking walk runs on
//! `spawn_blocking` so it never stalls the async runtime.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkMatch};
use ignore::{WalkBuilder, WalkState};
use serde_json::{Value, json};

use super::output::format_path;

/// Default cap on reported matches.
const DEFAULT_LIMIT: usize = 100;
/// Long match lines are clipped to keep output readable — a *character* count, matching pi's own
/// `GREP_MAX_LINE_LENGTH` (`truncateLine`'s `line.length`, UTF-16 code units — effectively chars for all
/// BMP text), not a byte count: a CJK/Cyrillic/accented-Latin line costs more bytes per character than
/// ASCII, so byte-clipping the same cap would truncate it to far fewer visible characters for no reason.
const MAX_LINE: usize = 500;
/// Hard ceiling on matches collected before the walk bails — an OOM guard for pathological patterns
/// (matching nearly every line of a huge tree). Far above any sane `limit`; when it trips, the
/// surviving subset can depend on scheduling, which the output flags.
const HARD_CAP: usize = 10_000;
/// Ceiling on the `before`/`after` context window. Context lines aren't bounded by `HARD_CAP` (which
/// counts matches), so clamp them to keep a huge `after` on a match-dense file from ballooning memory.
const MAX_CONTEXT: usize = 100;
/// Suffix [`clip`] appends to a line it truncates. `Tool::run` checks surviving hit text for this
/// suffix to decide whether to append an aggregate "some lines truncated" notice (matching pi's own
/// grep, which surfaces that as an actionable notice alongside the match-limit/byte-cap ones) — a
/// single source of truth instead of threading a separate "was this line clipped" flag through
/// `Hit`/`Collector` just for one summary line.
const LINE_TRUNCATED_SUFFIX: &str = "… [truncated]";

pub struct Grep;

/// One reported line: its path, line number, text, and whether it is a match (vs a context line). The
/// path is an `Arc<Path>` so a file with many matches allocates the path **once** and each hit is a
/// refcount bump, not a fresh `PathBuf` per hit.
type Hit = (Arc<Path>, usize, String, bool);

/// A prepared grep: ripgrep's regex matcher, an optional file glob (with whether it's negated — a
/// leading `!` in the original pattern, ripgrep-CLI-style — so `search` can invert the match test), the
/// search root, the report cap, and how many lines of context to show around each match (`before`/
/// `after`, like ripgrep's `-B`/`-A`/`-C`).
pub struct GrepJob {
    matcher: RegexMatcher,
    glob: Option<(GlobMatcher, bool)>,
    root: PathBuf,
    limit: usize,
    before: usize,
    after: usize,
}

impl GrepJob {
    /// Compile a job. `literal` regex-escapes the pattern (verbatim search); `ignore_case` folds case.
    /// Returns the regex error message on a bad pattern.
    pub fn new(
        pattern: &str,
        literal: bool,
        ignore_case: bool,
        glob: Option<(GlobMatcher, bool)>,
        root: PathBuf,
        limit: usize,
    ) -> Result<Self, String> {
        let escaped;
        let effective = if literal {
            escaped = regex::escape(pattern);
            escaped.as_str()
        } else {
            pattern
        };
        let matcher = RegexMatcherBuilder::new()
            .case_insensitive(ignore_case)
            .line_terminator(Some(b'\n'))
            .build(effective)
            .map_err(|e| format!("bad regex: {e}"))?;
        Ok(Self {
            matcher,
            glob,
            root,
            limit,
            before: 0,
            after: 0,
        })
    }

    /// Show `before` lines before and `after` lines after each match (clamped to [`MAX_CONTEXT`]).
    pub fn with_context(mut self, before: usize, after: usize) -> Self {
        self.before = before.min(MAX_CONTEXT);
        self.after = after.min(MAX_CONTEXT);
        self
    }
}

/// Run the parallel search. `threads == 0` lets `ignore` choose (≈ CPU count); the bench passes 1 to
/// measure the single-threaded baseline. Returns hits sorted by `(path, line)` — each flagged as a
/// match or a context line — whether the result was truncated (by `limit` or the hard cap), and, when
/// the walk hit at least one unreadable path (permission denied, a broken symlink, …), the first such
/// error's message — matching real ripgrep, which exits non-zero and reports the real error text the
/// moment it can't read *any* path in the tree, rather than silently treating it as "not found here."
/// `limit` and the hard cap count *matches*, not context lines.
pub fn search(job: &GrepJob, threads: usize) -> (Vec<Hit>, bool, Option<String>) {
    let collected: Arc<Mutex<Vec<Hit>>> = Arc::new(Mutex::new(Vec::new()));
    let total = Arc::new(AtomicUsize::new(0));
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    // The walk quits as soon as *either* threshold is crossed — `job.limit` in the common case (a
    // low-limit query against a match-dense tree should stop almost immediately, not walk the whole
    // tree only to throw away everything past the first `limit` matches), `HARD_CAP` as the outer
    // ceiling when a caller passes an unusually large `limit`. Below `HARD_CAP`, "the surviving subset
    // can depend on scheduling" (this module's own doc comment) now applies whenever a query is
    // actually truncated, not just on a pathological hard-cap trip — an accepted trade against
    // `should_compact`-style exhaustive correctness, matching pi's own `rg` process kill on the same
    // threshold.
    let stop_at = job.limit.min(HARD_CAP);

    // `hidden(false)` includes dotfiles (like ripgrep --hidden); .gitignore is respected by default.
    // `require_git(false)` keeps that respect even outside an actual git repository (e.g. a plain
    // checkout with no `.git`, or a tree copied without its VCS metadata) — the ignore crate's default
    // otherwise silently stops honoring `.gitignore` the moment there's no repo to find. Only applied
    // outside a real repo, though: inside one, the default git-aware walk keeps a nested repo's own
    // `.gitignore` from leaking parent rules across its boundary.
    //
    // Pi-parity note: this is a deliberate DIVERGENCE, not a match. Pi's own grep tool shells out to
    // real `rg` with no `--no-require-git`/git-detection logic at all, so it silently stops honoring
    // `.gitignore` outside a git repo (ripgrep's own default). That conditional-on-repo-boundary
    // handling exists only in pi's *find* tool (`fd --no-require-git`, added only when the search root
    // isn't inside a git repo — see this same check in `find.rs`, which correctly cites it). Beyond
    // extends the identical policy to grep too, for consistency between the two search tools and
    // because it's strictly more useful (a plain checkout with no `.git` still gets its `.gitignore`
    // honored) — kept rather than narrowed to match grep.ts's behavior here.
    let mut builder = WalkBuilder::new(&job.root);
    builder.hidden(false);
    if !super::root_is_inside_git_repo(&job.root) {
        builder.require_git(false);
    }
    builder.threads(threads).build_parallel().run(|| {
        let collected = Arc::clone(&collected);
        let total = Arc::clone(&total);
        let first_error = Arc::clone(&first_error);
        let matcher = &job.matcher;
        let glob = &job.glob;
        // One `Searcher` per worker thread (it carries reusable line buffers): ripgrep's engine —
        // byte-oriented, so non-UTF-8 files search correctly (a `BufReader::read_line` loop would
        // error out and silently drop them); binary files quit cleanly on a NUL; large files are
        // mmap'd; line counting is SIMD-accelerated. `before`/`after` context is built in.
        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .before_context(job.before)
            .after_context(job.after)
            .binary_detection(BinaryDetection::quit(b'\x00'))
            .build();
        Box::new(move |entry| {
            if total.load(Ordering::Relaxed) >= stop_at {
                return WalkState::Quit;
            }
            let entry = match entry {
                Ok(entry) => entry,
                // A walk-level error (permission denied on a directory, a broken symlink, …) — the
                // entry itself is skipped, matching the prior behavior, but recorded so an
                // otherwise-empty result can say "couldn't fully search" instead of a confidently
                // wrong "no matches" — matching real ripgrep, which exits non-zero the moment it
                // can't read *any* path in the tree, real or reported here or not.
                Err(e) => {
                    if let Ok(mut guard) = first_error.lock() {
                        guard.get_or_insert_with(|| e.to_string());
                    }
                    return WalkState::Continue;
                }
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                return WalkState::Continue;
            }
            let path = entry.path();
            if let Some((g, negate)) = glob {
                let matched = g.is_match(path);
                let keep = if *negate { !matched } else { matched };
                if !keep {
                    return WalkState::Continue;
                }
            }
            // Collect this file's hits into one local sink, then push once — one lock per matching
            // file. An I/O error (unreadable file) skips it, like the prior behavior, but is likewise
            // recorded for the same reason as a walk-level error above.
            // One `Arc<Path>` allocation for this file; every hit clones the pointer (refcount
            // bump), so a match-dense file no longer pays a `PathBuf` per hit.
            let mut sink = Collector {
                path: Arc::from(path),
                hits: Vec::new(),
                matches: 0,
            };
            if let Err(e) = searcher.search_path(matcher, path, &mut sink) {
                if let Ok(mut guard) = first_error.lock() {
                    guard.get_or_insert_with(|| format!("{}: {e}", path.display()));
                }
                return WalkState::Continue;
            }
            if !sink.hits.is_empty() {
                let matches = sink.matches;
                if let Ok(mut guard) = collected.lock() {
                    guard.extend(sink.hits);
                }
                // Only matches count toward the stop threshold (context is bounded per match).
                if total.fetch_add(matches, Ordering::Relaxed) + matches >= stop_at {
                    return WalkState::Quit;
                }
            }
            WalkState::Continue
        })
    });

    let mut hits = Arc::try_unwrap(collected)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default();
    hits.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let stopped_early = total.load(Ordering::Relaxed) >= stop_at;
    let match_total = hits.iter().filter(|h| h.3).count();
    let truncated = match_total > job.limit || stopped_early;
    // Keep at most `limit` matched lines (context lines don't count toward it). Walk counting matches,
    // cut at the match that would exceed the limit, then drop any now-dangling trailing context (the
    // before-context of the dropped match that trailed into the kept prefix). With no context
    // requested every hit is a match, so this reduces to a plain `truncate(limit)`.
    if match_total > job.limit {
        let mut seen = 0usize;
        let mut cut = hits.len();
        for (i, h) in hits.iter().enumerate() {
            if h.3 {
                seen += 1;
                if seen > job.limit {
                    cut = i;
                    break;
                }
            }
        }
        hits.truncate(cut);
        while hits.last().is_some_and(|h| !h.3) {
            hits.pop();
        }
    }
    let first_error = Arc::try_unwrap(first_error)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .flatten();
    (hits, truncated, first_error)
}

/// A [`Sink`] that gathers one file's matches and context lines. `matched`/`context` are called by the
/// searcher with byte-accurate line numbers; bytes are decoded lossily so non-UTF-8 files still yield
/// readable, matchable text instead of being dropped.
struct Collector {
    path: Arc<Path>,
    hits: Vec<Hit>,
    matches: usize,
}

impl Sink for Collector {
    type Error = std::io::Error;

    fn matched(&mut self, _searcher: &Searcher, m: &SinkMatch<'_>) -> std::io::Result<bool> {
        let lineno = m.line_number().map(|n| n as usize).unwrap_or(0);
        let text = String::from_utf8_lossy(m.bytes());
        self.hits
            .push((self.path.clone(), lineno, clip(&trim_eol(&text)), true));
        self.matches += 1;
        Ok(true)
    }

    fn context(&mut self, _searcher: &Searcher, c: &SinkContext<'_>) -> std::io::Result<bool> {
        let lineno = c.line_number().map(|n| n as usize).unwrap_or(0);
        let text = String::from_utf8_lossy(c.bytes());
        self.hits
            .push((self.path.clone(), lineno, clip(&trim_eol(&text)), false));
        Ok(true)
    }
}

/// Strip a trailing `\n` and then a trailing `\r` (the searcher hands us the line terminator), plus —
/// pi-parity fix — any further embedded `\r` the trailing strip doesn't reach. `grep-searcher` only
/// splits on `\n`, so a file using old-Mac-style bare-`\r` line endings (or one with stray corrupted
/// bytes) hands us "lines" that still carry mid-content `\r` characters; pi strips every `\r` in the
/// line, not just a trailing one. Borrows unchanged in the overwhelming common case (no embedded `\r`
/// at all) rather than allocating a `String` for every matched line.
fn trim_eol(s: &str) -> std::borrow::Cow<'_, str> {
    let s = s.strip_suffix('\n').unwrap_or(s);
    let s = s.strip_suffix('\r').unwrap_or(s);
    if s.contains('\r') {
        std::borrow::Cow::Owned(s.replace('\r', ""))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

/// Clip a long line to `MAX_LINE` *characters*, not bytes — pi-parity fix: the previous byte-based cap
/// (with a char-boundary backoff to avoid splitting a codepoint) never produced invalid UTF-8, but still
/// truncated a non-ASCII line to far fewer visible characters than an ASCII line under the same nominal
/// cap, unlike pi's own char-counting `truncateLine`.
fn clip(line: &str) -> String {
    let mut chars = line.chars();
    let head: String = chars.by_ref().take(MAX_LINE).collect();
    if chars.next().is_none() {
        return head; // the whole line fit within the cap — nothing was truncated
    }
    format!("{head}{LINE_TRUNCATED_SUFFIX}")
}

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }
    fn description(&self) -> &str {
        "Search file contents by regular expression, honoring .gitignore. Optionally restrict to a \
         `path`, a `glob` (e.g. \"*.rs\"), case-insensitive with `ignore_case`. Results are sorted by \
         path; when more than `limit` match, the lexicographically-smallest are reported."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression to search for." },
                "path": { "type": "string", "description": "Directory or file to search (default \".\")." },
                "glob": { "type": "string", "description": "Only search files matching this glob. Prefix with \"!\" to exclude files matching it instead." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive (default false)." },
                "literal": { "type": "boolean", "description": "Treat `pattern` as a literal string, not a regex (default false)." },
                "limit": { "type": "integer", "description": "Max matches to report (default 100)." },
                "context": { "type": "integer", "description": "Lines of context to show before and after each match (like ripgrep -C). Default 0." },
                "before": { "type": "integer", "description": "Lines of context before each match (overrides `context`)." },
                "after": { "type": "integer", "description": "Lines of context after each match (overrides `context`)." }
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let pattern = input
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `pattern`".into()))?;
        let root = input.get("path").and_then(Value::as_str).unwrap_or(".");
        let root = &super::normalize_path(root);
        let ignore_case = input
            .get("ignore_case")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // Floored at 1 — matches pi's own `grep.ts` (`Math.max(1, limit ?? DEFAULT_LIMIT)`). Without
        // this, `limit: 0` makes `stop_at` (in `search`) zero, so the walk's very first "have we hit
        // the threshold?" check is already true before a single file is scanned — a confidently wrong
        // "no matches" even when real matches exist. `find`'s own `limit` deliberately has no such
        // floor (pi's `find.ts` doesn't apply one either — a genuine asymmetry between the two tools
        // upstream, not an oversight to "fix" into false symmetry here).
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT)
            .max(1);

        // Context lines around each match. `context` sets both sides; `before`/`after` override it per
        // side. Absent everywhere → 0 (the prior behavior, just the matching lines).
        let usize_arg = |key: &str| input.get(key).and_then(Value::as_u64).map(|n| n as usize);
        let context = usize_arg("context");
        let before = usize_arg("before").or(context).unwrap_or(0);
        let after = usize_arg("after").or(context).unwrap_or(0);

        // `literal` searches for the pattern verbatim (regex-escaped), so a model can grep for
        // `a.b(c)` or `Vec<T>` without escaping the regex metacharacters itself.
        let literal = input
            .get("literal")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // A leading "!" negates the glob (ripgrep-CLI-style: `rg --glob '!*.test.rs'` excludes rather
        // than restricts). `globset::Glob` has no negation syntax of its own — a leading "!" would
        // otherwise be matched literally, so `!*.test.rs` would confidently match nothing at all and
        // the whole query would report zero results instead of applying the exclusion.
        let glob = match input.get("glob").and_then(Value::as_str) {
            Some(g) => {
                let (negate, pattern) = match g.strip_prefix('!') {
                    Some(rest) => (true, rest),
                    None => (false, g),
                };
                Some((
                    Glob::new(pattern)
                        .map_err(|e| ToolError::InvalidInput(format!("bad glob: {e}")))?
                        .compile_matcher(),
                    negate,
                ))
            }
            None => None,
        };

        let no_match = format!("no matches for {pattern:?}");
        let root = PathBuf::from(root);
        let job = GrepJob::new(pattern, literal, ignore_case, glob, root.clone(), limit)
            .map_err(ToolError::InvalidInput)?
            .with_context(before, after);
        // The walk blocks (its own thread pool, synchronous reads); keep it off the async runtime.
        let (matches, truncated, walk_error) = tokio::task::spawn_blocking(move || search(&job, 0))
            .await
            .map_err(|e| ToolError::Execution(format!("grep task failed: {e}")))?;

        if matches.is_empty() {
            // A genuine zero-match walk and a walk that hit an unreadable path along the way both
            // land here with an empty result — but they aren't the same thing, and reporting them
            // identically ("no matches") tells the model a confident falsehood in the second case.
            // Matches real ripgrep, which exits non-zero and reports the real error the moment it
            // can't read *any* path in the tree.
            if let Some(err) = walk_error {
                return Err(ToolError::Execution(format!(
                    "search was incomplete, so \"no matches\" may not be accurate: {err}"
                )));
            }
            return Ok(no_match.into());
        }
        let mut out = String::new();
        let mut lines_truncated = false;
        for (path, line, text, is_match) in &matches {
            // Write straight into `out` instead of allocating a `format!` temp String per line — same
            // fix as `read`/`ls`'s formatting loops. Match lines use `path:line:` (ripgrep's
            // separator); context lines use `path-line-` so a reader (and the model) can tell a hit
            // from its surrounding context at a glance. `writeln!` into a `String` can't fail, so the
            // `Result` is discarded.
            if text.ends_with(LINE_TRUNCATED_SUFFIX) {
                lines_truncated = true;
            }
            let display_path = format_path(path, &root);
            let _ = if *is_match {
                writeln!(out, "{display_path}:{line}: {text}")
            } else {
                writeln!(out, "{display_path}-{line}- {text}")
            };
        }
        // The byte cap is checked *before* any count/line marker is appended, and takes priority when
        // any would otherwise fire — see `cap_listing_bytes`'s doc comment. The byte-cap marker's own
        // "narrow the pattern, path, or context" guidance already covers both cases, so dropping the
        // more specific markers in favor of it loses nothing actionable.
        let byte_capped = super::output::cap_listing_bytes(
            &mut out,
            "narrow the pattern, path, or context to see more",
        );
        if !byte_capped {
            let mut notices = Vec::new();
            if truncated {
                notices.push(format!(
                    "match limit {limit} reached; narrow the pattern or raise `limit`"
                ));
            }
            if lines_truncated {
                notices.push("some lines truncated; use the read tool to see them in full".into());
            }
            if let Some(err) = &walk_error {
                notices.push(format!(
                    "search was incomplete — some paths couldn't be read, so there may be more \
                     matches than shown: {err}"
                ));
            }
            if !notices.is_empty() {
                let _ = writeln!(out, "{}", super::output::marker(notices.join(". ")));
            }
        }
        Ok(out.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn literal_mode_searches_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "call a.b(c) here\nand axbc too\n").unwrap();
        // As a regex `a.b(c)` matches `axbc`; literal mode must match only the verbatim text.
        let out = Grep
            .run(json!({
                "pattern": "a.b(c)",
                "literal": true,
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("call a.b(c)"));
        assert!(
            !out.contains("axbc"),
            "literal mode must not regex-match: {out}"
        );
    }

    #[tokio::test]
    async fn output_paths_are_relative_to_the_search_root_not_absolute() {
        // A prior bug reported the full path straight from the walk entry (often absolute, since the
        // search root itself usually is) — costing the model extra tokens per line for a prefix it
        // already knows, and diverging from pi's documented "relative to search directory" contract.
        // Exact-match (not `.contains`) so a leftover root prefix can't slip through undetected.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "needle\n").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("b.txt"), "needle\n").unwrap();

        let out = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        let lines: Vec<&str> = out.lines().collect();
        assert!(
            lines.contains(&"a.txt:1: needle"),
            "expected a root-relative bare filename, got: {out}"
        );
        assert!(
            lines.contains(&"sub/b.txt:1: needle"),
            "expected a root-relative nested path, got: {out}"
        );
        assert!(
            !out.contains(dir.path().to_str().unwrap()),
            "the search root's own absolute prefix must not appear in the output: {out}"
        );
    }

    #[tokio::test]
    async fn searching_a_single_file_reports_just_its_basename() {
        // Mirrors pi: when `path` names a single file (not a directory), there's nothing meaningful to
        // show beyond its own name — the "relative to root" and "the file itself" cases coincide.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "needle\n").unwrap();

        let out = Grep
            .run(json!({ "pattern": "needle", "path": file.to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim_end(), "a.txt:1: needle", "got: {out}");
    }

    #[tokio::test]
    async fn finds_matches_with_path_and_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\nhello again\n").unwrap();
        std::fs::write(dir.path().join("b.log"), "nothing here\n").unwrap();

        let out = Grep
            .run(json!({ "pattern": "hello", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:1: hello"));
        assert!(out.contains("a.txt:3: hello again"));
        assert!(!out.contains("b.log"));
    }

    #[tokio::test]
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path`, via its `@`-prefix-strip behavior
        // (needs no `$HOME` mutation — see `expand_tilde`'s own direct unit tests for that half).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
        let at_prefixed = format!("@{}", dir.path().to_str().unwrap());
        let out = Grep
            .run(json!({ "pattern": "hello", "path": at_prefixed }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:1: hello"));
    }

    #[tokio::test]
    async fn glob_filters_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("keep.rs"), "fn target() {}\n").unwrap();
        std::fs::write(dir.path().join("skip.txt"), "fn target() {}\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "target", "path": dir.path().to_str().unwrap(), "glob": "*.rs" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("keep.rs"));
        assert!(!out.contains("skip.txt"));
    }

    #[tokio::test]
    async fn a_leading_bang_in_the_glob_negates_it() {
        // pi-parity fix (task 16): `globset::Glob` has no negation syntax, so a leading "!" was
        // matched literally and `!*.test.rs` confidently matched nothing at all — the entire query
        // returned zero results instead of applying the exclusion, unlike real `rg --glob`.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("plain.rs"), "fn target() {}\n").unwrap();
        std::fs::write(dir.path().join("thing.test.rs"), "fn target() {}\n").unwrap();
        let out = Grep
            .run(json!({
                "pattern": "target",
                "path": dir.path().to_str().unwrap(),
                "glob": "!*.test.rs"
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("plain.rs"),
            "the non-excluded file must still be found: {out}"
        );
        assert!(
            !out.contains("thing.test.rs"),
            "the excluded file must not be reported: {out}"
        );
    }

    #[tokio::test]
    async fn gitignore_is_honored_even_outside_a_git_repository() {
        // `tempfile::tempdir()` lives under the system temp dir, not inside any git repository, so
        // this exercises `require_git(false)`: without it, the `ignore` crate's default behavior is
        // to stop respecting `.gitignore` files entirely once there's no `.git` ancestor to find.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "needle\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("kept.txt"), "got: {out}");
        assert!(
            !out.contains("ignored.txt"),
            ".gitignore should be honored even without a .git directory: {out}"
        );
    }

    #[tokio::test]
    async fn gitignore_is_still_honored_inside_a_real_git_repository() {
        // pi-parity fix (L7): `require_git(false)` is now conditional on the search root *not* being
        // inside a real git repo (a deliberate divergence from pi's own `rg` invocation, which has no
        // such conditional at all — see this module's doc comment on `require_git(false)`) — this
        // guards against the conditional accidentally breaking the far more common case, a search root
        // that *is* inside a real repo, which must still honor `.gitignore` exactly as before.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "needle\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("kept.txt"), "got: {out}");
        assert!(
            !out.contains("ignored.txt"),
            ".gitignore must still be honored inside a real git repository: {out}"
        );
    }

    #[tokio::test]
    async fn no_matches_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "abc\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "zzz", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("no matches"));
    }

    /// Skip a permission-bit test under a runtime that doesn't actually enforce them (root, some
    /// sandboxes) — matches the same guard `resources.rs`'s unreadable-file tests use.
    #[cfg(unix)]
    fn mode_actually_blocks_reads(path: &std::path::Path) -> bool {
        std::fs::read_dir(path).is_err()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_search_root_reports_an_error_not_a_false_no_matches() {
        // Pi-parity audit H69: real ripgrep exits non-zero the moment it can't read a path in the
        // tree, root included — this crate's own walk previously swallowed that into `WalkState::
        // Continue`, so an unreadable root silently reported "no matches" instead of surfacing the
        // real problem.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "needle\n").unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(dir.path());

        let result = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await;
        let _ = std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755));

        if !blocks {
            return;
        }
        let err =
            result.expect_err("an unreadable root must surface as an error, not empty output");
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error, got {err:?}")
        };
        assert!(
            msg.contains("incomplete") && msg.contains("Permission denied"),
            "must surface the real reason, not a bare confident-empty result: {msg}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_subdirectory_with_no_matches_elsewhere_reports_an_error() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readable.txt"), "no match here\n").unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::write(locked.join("secret.txt"), "needle\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(&locked);

        let result = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await;
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));

        if !blocks {
            return;
        }
        let err = result.expect_err(
            "an unreadable subdirectory with no matches found elsewhere must surface as an error, \
             not a false \"no matches\"",
        );
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error, got {err:?}")
        };
        assert!(
            !msg.contains("no matches for"),
            "must not phrase this as a confident empty result: {msg}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_subdirectory_alongside_real_matches_still_returns_them_with_a_notice() {
        // A real match elsewhere in the tree must not be discarded just because some other, unrelated
        // path couldn't be read — a deliberate, documented divergence from pi's own all-or-nothing
        // behavior (discarding every match the moment `rg` exits non-zero): silently losing real,
        // already-found matches over an unrelated permission hiccup would make the tool much less
        // useful in practice. The incompleteness is still surfaced, just as a notice, not a failure.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readable.txt"), "needle here\n").unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::write(locked.join("secret.txt"), "needle too\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(&locked);

        let result = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await;
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));

        if !blocks {
            return;
        }
        let out = result.unwrap().text;
        assert!(
            out.contains("readable.txt"),
            "a real match elsewhere must still be reported: {out}"
        );
        assert!(
            out.contains("incomplete"),
            "the result must note it may be incomplete: {out}"
        );
    }

    #[tokio::test]
    async fn output_is_path_sorted_and_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        // Created out of order; the report must come back path-sorted regardless of walk/thread order.
        for name in ["z.txt", "a.txt", "m.txt"] {
            std::fs::write(dir.path().join(name), "needle\n").unwrap();
        }
        let path = dir.path().to_str().unwrap();
        let run_once = || async {
            Grep.run(json!({ "pattern": "needle", "path": path }))
                .await
                .unwrap()
                .text
        };
        let out = run_once().await;
        let order: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split(':').next())
            .filter_map(|p| p.rsplit('/').next())
            .collect();
        assert_eq!(order, vec!["a.txt", "m.txt", "z.txt"]);
        // Stable across runs.
        assert_eq!(out, run_once().await);
    }

    #[tokio::test]
    async fn context_lines_surround_each_match() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\nNEEDLE\nfour\nfive\n").unwrap();
        let path = dir.path().to_str().unwrap();

        // `context: 1` shows one line on each side. The match keeps the `:` separator; context lines
        // use the `-` separator.
        let out = Grep
            .run(json!({ "pattern": "NEEDLE", "path": path, "context": 1 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:3: NEEDLE"), "match line: {out}");
        assert!(out.contains("a.txt-2- two"), "before-context: {out}");
        assert!(out.contains("a.txt-4- four"), "after-context: {out}");
        assert!(
            !out.contains("one"),
            "context window must not reach line 1: {out}"
        );
        assert!(
            !out.contains("five"),
            "context window must not reach line 5: {out}"
        );

        // `before`/`after` set each side independently (here: only after).
        let out = Grep
            .run(json!({ "pattern": "NEEDLE", "path": path, "before": 0, "after": 2 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:3: NEEDLE"));
        assert!(
            !out.contains("two"),
            "before=0 must show no preceding context: {out}"
        );
        assert!(out.contains("a.txt-4- four") && out.contains("a.txt-5- five"));
    }

    #[tokio::test]
    async fn output_byte_cap_truncates_even_under_the_match_limit() {
        let dir = tempfile::tempdir().unwrap();
        // 100 long matching lines with generous context: well under the 100-match `limit` (so the
        // match-count marker never fires), but the rendered text — line + surrounding context — blows
        // past the 50KB output cap on its own. Each line is also individually clipped by `MAX_LINE`, so
        // this exercises the per-line "[… truncated]" marker and the whole-output byte-cap marker
        // together in one realistic large-context scenario.
        let long_line = format!("needle {}", "x".repeat(600));
        let body = std::iter::repeat_n(long_line, 100)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "limit": 1000,
                "context": 5,
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
        assert!(
            out.contains("… [truncated]"),
            "per-line clip marker missing"
        );
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {}",
            &out[out.len().saturating_sub(200)..]
        );
        assert!(
            !out.contains("match limit"),
            "match-count marker must not co-fire when count never exceeded the limit"
        );
        assert!(
            !out.contains("some lines truncated"),
            "the aggregate line-truncation notice must not survive the byte cap either — same \
             precedence as the match-count marker, and for the same reason (it could be sliced \
             through by the truncation it would be describing)"
        );
    }

    #[tokio::test]
    async fn aggregate_notice_reports_when_lines_were_individually_clipped() {
        let dir = tempfile::tempdir().unwrap();
        // One over-long matching line among otherwise-short ones: neither the match-count marker nor
        // the byte-cap marker fires (both stay well clear of their thresholds), isolating the
        // aggregate "some lines truncated" notice as the only truncation signal in the output.
        let long_line = format!("needle {}", "x".repeat(600));
        let body = format!("{long_line}\nneedle short\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains(LINE_TRUNCATED_SUFFIX),
            "per-line clip missing: {out}"
        );
        assert!(
            out.contains("[some lines truncated; use the read tool to see them in full]"),
            "aggregate notice missing: {out}"
        );
        assert!(
            !out.contains("match limit"),
            "match-count marker must not co-fire: {out}"
        );
    }

    #[tokio::test]
    async fn aggregate_notice_joins_match_limit_and_line_truncation_when_both_fire() {
        let dir = tempfile::tempdir().unwrap();
        // First matching line is over-long (individually clipped); many more short matches follow so
        // the match-count marker also fires. A single file keeps in-file line order deterministic
        // (unlike a multi-file walk), so the over-long first line is guaranteed to survive the
        // `limit` truncation and both notices are guaranteed to co-fire.
        let long_line = format!("needle {}", "x".repeat(600));
        let mut lines = vec![long_line];
        lines.extend(std::iter::repeat_n("needle short".to_string(), 20));
        let body = lines.join("\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "limit": 5,
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains(
                "[match limit 5 reached; narrow the pattern or raise `limit`. some lines \
                 truncated; use the read tool to see them in full]"
            ),
            "joined notice missing or in the wrong order: {out}"
        );
    }

    #[tokio::test]
    async fn output_byte_cap_takes_priority_over_the_match_count_marker() {
        let dir = tempfile::tempdir().unwrap();
        // 200 long matching lines, default `limit` of 100: both the match-count marker (200 > 100) and
        // the byte cap would fire on the same rendered text. Appending the count marker and *then*
        // truncating to the byte cap would risk slicing straight through (and corrupting) that just-
        // appended marker — the byte-cap check runs on the body first and, when it trips, wins outright:
        // only its own marker (whose "narrow the pattern, path, or context" advice already subsumes the
        // count case) appears, never a mangled partial marker.
        let long_line = format!("needle {}", "x".repeat(600));
        let body = std::iter::repeat_n(long_line, 200)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "context": 5,
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {out}"
        );
        assert!(
            !out.contains("match limit"),
            "the byte-cap marker must win cleanly, not leave a mangled match-count marker behind"
        );
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
    }

    #[tokio::test]
    async fn limit_truncates_to_at_most_limit_matches() {
        // The walk now quits as soon as `limit` matches are found (see `search`'s doc comment on why),
        // so which 3 of the 10 files survive can vary with thread scheduling across a *parallel* walk —
        // unlike the untruncated case (`output_is_path_sorted_and_deterministic`), which always finds
        // everything and so is always fully deterministic. This test asserts the contract that's still
        // guaranteed: truncation actually happened, and no more than `limit` matches came back.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i:02}.txt")), "needle\n").unwrap();
        }
        let out = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap(), "limit": 3 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("[match limit 3 reached; narrow the pattern or raise `limit`]"));
        let match_lines = out.lines().filter(|l| l.contains("needle")).count();
        assert_eq!(
            match_lines, 3,
            "at most `limit` matches must be reported: {out}"
        );
    }

    #[tokio::test]
    async fn a_limit_of_zero_is_floored_to_one_not_a_confident_no_matches() {
        // Pi-parity audit M10: matches pi's own `grep.ts` (`Math.max(1, limit ?? DEFAULT_LIMIT)`).
        // Without the floor, `limit: 0` made `stop_at` zero, so `search`'s very first "have we hit the
        // threshold?" check was already true before a single file was scanned — a confidently wrong
        // "no matches" even though a real match exists.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "needle\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap(), "limit": 0 }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("needle"),
            "limit: 0 must still find the real match (floored to 1), not report false emptiness: {out}"
        );
    }

    #[tokio::test]
    async fn an_untruncated_result_is_still_fully_deterministic() {
        // Fewer matches than `limit` — the walk never quits early, so every match is found regardless
        // of thread scheduling, and the sorted output is exactly reproducible run to run (unlike the
        // truncated case, which the early-exit optimization deliberately no longer guarantees).
        let dir = tempfile::tempdir().unwrap();
        for name in ["z.txt", "a.txt", "m.txt"] {
            std::fs::write(dir.path().join(name), "needle\n").unwrap();
        }
        let path = dir.path().to_str().unwrap();
        let run_once = || async {
            Grep.run(json!({ "pattern": "needle", "path": path, "limit": 100 }))
                .await
                .unwrap()
                .text
        };
        let first = run_once().await;
        assert!(first.contains("a.txt") && first.contains("m.txt") && first.contains("z.txt"));
        for _ in 0..5 {
            assert_eq!(
                first,
                run_once().await,
                "an untruncated result must be stable across runs"
            );
        }
    }

    #[tokio::test]
    async fn searches_non_utf8_files() {
        // A file with an invalid UTF-8 byte (0xE9, latin-1 'é') around an ASCII match. The old
        // `BufReader::read_line` loop errored on the bad byte and silently dropped the whole file; the
        // byte-oriented ripgrep engine finds the match and decodes the line lossily. This is the
        // correctness regression vs pi's `rg` that the rewrite fixes.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("latin1.txt"), b"caf\xe9 NEEDLE here\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("NEEDLE"),
            "must match inside a non-UTF-8 file: {out}"
        );
        assert!(out.contains("latin1.txt"));
    }

    #[tokio::test]
    async fn skips_binary_files() {
        // A NUL byte marks a file as binary; the searcher quits on it instead of emitting garbage
        // (ripgrep's default). The adjacent text file still matches.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("blob.bin"), b"\x00\x00 NEEDLE \x00binary").unwrap();
        std::fs::write(dir.path().join("text.txt"), b"NEEDLE in text\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("text.txt"), "the text file must match: {out}");
        assert!(
            !out.contains("blob.bin"),
            "the binary file must be skipped, not reported: {out}"
        );
    }
}
