//! [`LocalFs`] — the host filesystem, via `std::fs` and ripgrep's in-process walker.
//!
//! This is the behavior that shipped before [`super`] existed, **moved rather than rewritten**. The
//! walk runs in parallel (ripgrep's `WalkParallel`): worker threads read and scan files concurrently —
//! the difference between grep and ripgrep on a large tree. The walk quits as soon as `stop_at`
//! matches are found, so a low-limit query against a match-dense tree does work proportional to the
//! limit, not to the tree's size. Whatever was collected before quitting is sorted and truncated by
//! [`super::finalize`], so an **untruncated** result (every match found, nothing cut) is always fully
//! deterministic; a **truncated** one returns `limit` matches from wherever the parallel walk happened
//! to reach before quitting, not guaranteed to be the lexicographically-smallest ones — the same trade
//! pi's own `grep` makes by killing its `rg` child the instant it has enough matches.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkMatch};
use ignore::{WalkBuilder, WalkState};

use super::{
    DirEntry, FileKind, FsBackend, FsError, GlobOutcome, GlobQuery, Hit, MAX_CONTEXT, Meta,
    SearchOutcome, SearchQuery, clip, finalize, trim_eol,
};

/// The host filesystem.
///
/// `threads` is passed straight to `ignore`'s walker: `0` lets it choose (≈ CPU count), which is what
/// production uses. The search benchmark constructs one with `1` to measure the single-threaded
/// baseline — the reason this knob is on the *backend* rather than in [`SearchQuery`], where every
/// implementation would have had to pretend to honor a thread count it has no concept of.
#[derive(Debug, Clone, Default)]
pub struct LocalFs {
    threads: usize,
}

impl LocalFs {
    /// A backend using the walker's own thread-count heuristic.
    pub fn new() -> Self {
        Self { threads: 0 }
    }

    /// Pin the walk to `threads` worker threads. `0` restores the default heuristic.
    pub fn with_threads(threads: usize) -> Self {
        Self { threads }
    }

    /// The blocking body of [`FsBackend::search`], exposed so the search benchmark can drive it
    /// directly without a tokio runtime — and so the async wrapper below is nothing but
    /// `spawn_blocking`, with no logic of its own to diverge from what's measured.
    pub fn search_blocking(&self, q: &SearchQuery) -> Result<SearchOutcome, FsError> {
        // Glob before regex, deliberately: `grep`'s `run` used to compile the glob first and a query
        // that is malformed in *both* ways has always reported the glob error. Swapping the order here
        // would silently change which message the model gets.
        let glob = compile_glob(q.glob.as_ref())?;
        let matcher = compile_matcher(q)?;
        Ok(walk(
            &matcher,
            glob.as_ref(),
            &q.root,
            q.stop_at(),
            q.limit,
            q.before.min(MAX_CONTEXT),
            q.after.min(MAX_CONTEXT),
            self.threads,
        ))
    }
}

#[async_trait]
impl FsBackend for LocalFs {
    async fn search(&self, q: &SearchQuery) -> Result<SearchOutcome, FsError> {
        let this = self.clone();
        let q = q.clone();
        // The walk blocks (its own thread pool, synchronous reads); keep it off the async runtime.
        tokio::task::spawn_blocking(move || this.search_blocking(&q))
            .await
            .map_err(|e| FsError::Backend(format!("grep task failed: {e}")))?
    }

    async fn stat(&self, path: &Path) -> Result<Option<Meta>, FsError> {
        let path = path.to_path_buf();
        blocking("stat", move || Ok(stat_blocking(&path))).await
    }

    async fn read_bytes(&self, path: &Path, offset: u64, max: usize) -> Result<Vec<u8>, FsError> {
        let path = path.to_path_buf();
        blocking("read", move || {
            use std::io::{Read as _, Seek as _, SeekFrom};
            let mut f = std::fs::File::open(&path)
                .map_err(|e| FsError::Backend(format!("read {}: {e}", path.display())))?;
            if offset > 0 {
                f.seek(SeekFrom::Start(offset))
                    .map_err(|e| FsError::Backend(format!("read {}: {e}", path.display())))?;
            }
            let mut buf = Vec::new();
            // `take` bounds the read at the source rather than reading everything and slicing, so a
            // caller asking for a 4 KiB sniffing prefix of a 10 GiB file pays for 4 KiB.
            f.take(max as u64)
                .read_to_end(&mut buf)
                .map_err(|e| FsError::Backend(format!("read {}: {e}", path.display())))?;
            Ok(buf)
        })
        .await
    }

    async fn write_bytes(&self, path: &Path, bytes: &[u8]) -> Result<(), FsError> {
        let path = path.to_path_buf();
        let bytes = bytes.to_vec();
        blocking("write", move || {
            // The existing shared helper, unchanged: sibling temp file + `rename`, `create_new` to
            // refuse a planted symlink, existing mode bits preserved across the swap.
            crate::tools::write_atomic(&path.to_string_lossy(), &bytes)
                .map_err(|e| FsError::Backend(format!("write {}: {e}", path.display())))
        })
        .await
    }

    async fn write_if_unchanged(
        &self,
        path: &Path,
        bytes: &[u8],
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, FsError> {
        let path = path.to_path_buf();
        let bytes = bytes.to_vec();
        blocking("write", move || {
            let current = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
            if current != expected {
                return Ok(false);
            }
            crate::tools::write_atomic(&path.to_string_lossy(), &bytes)
                .map_err(|e| FsError::Backend(format!("write {}: {e}", path.display())))?;
            Ok(true)
        })
        .await
    }

    async fn create_dir_all(&self, path: &Path) -> Result<(), FsError> {
        let path = path.to_path_buf();
        blocking("mkdir", move || {
            std::fs::create_dir_all(&path)
                .map_err(|e| FsError::Backend(format!("create {}: {e}", path.display())))
        })
        .await
    }

    async fn list_dir(
        &self,
        path: &Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<DirEntry>, FsError> {
        let path = path.to_path_buf();
        blocking("ls", move || {
            let rd = std::fs::read_dir(&path)
                .map_err(|e| FsError::Backend(format!("ls {}: {e}", path.display())))?;
            let mut out = Vec::new();
            for entry in rd {
                // Checked at the top of every iteration so a pathologically large directory stops both
                // walking *and* stat-ing well before `out` could grow unbounded, rather than being
                // truncated after the fact.
                if out.len() >= cap {
                    break;
                }
                let Ok(entry) = entry else { continue };
                let name = entry.file_name().to_string_lossy().into_owned();
                if !include_hidden && name.starts_with('.') {
                    continue;
                }
                // Fast path: `DirEntry::file_type()` comes straight from the directory read itself (on
                // Linux, from `d_type` — no extra syscall on filesystems that populate it), so a
                // non-symlink entry, the overwhelming majority, never needs the full `stat`.
                //
                // Slow path: `file_type` is lstat-like and reports a *symlink's own* type, not its
                // target's, so a link to a directory would be mislabeled a plain file. Falling back to
                // `metadata` (which follows) fixes that and subsumes the unstattable cases — a dangling
                // symlink or an entry that vanished between readdir and here both fail and are skipped,
                // rather than being guessed at as a non-directory.
                let (kind, len) = match entry.file_type() {
                    Ok(ft) if !ft.is_symlink() => {
                        (kind_of(&ft), entry.metadata().map(|m| m.len()).unwrap_or(0))
                    }
                    _ => {
                        let Ok(meta) = std::fs::metadata(entry.path()) else {
                            continue;
                        };
                        (kind_of(&meta.file_type()), meta.len())
                    }
                };
                out.push(DirEntry { name, kind, len });
            }
            Ok(out)
        })
        .await
    }

    async fn glob(&self, q: &GlobQuery) -> Result<GlobOutcome, FsError> {
        let q = q.clone();
        blocking("find", move || glob_blocking(&q)).await
    }
}

/// Run a blocking filesystem body on the blocking pool, mapping a join failure to a backend error.
/// Every `LocalFs` method funnels through this so none of them can accidentally run a synchronous
/// syscall inline on an async worker — which in `serve_ws` would pin a per-session current-thread
/// runtime for the whole call.
async fn blocking<T, F>(what: &'static str, f: F) -> Result<T, FsError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, FsError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| FsError::Backend(format!("{what} task failed: {e}")))?
}

fn kind_of(t: &std::fs::FileType) -> FileKind {
    if t.is_file() {
        FileKind::File
    } else if t.is_dir() {
        FileKind::Dir
    } else {
        FileKind::Other
    }
}

fn stat_blocking(path: &Path) -> Option<Meta> {
    // `metadata` follows symlinks, so a link pointing *at* a FIFO is classified as one too — not just
    // a bare FIFO. That is the point: the hazard is what gets opened, not what the name is.
    let meta = std::fs::metadata(path).ok()?;
    let kind = kind_of(&meta.file_type());
    // `is_writable` performs a *real* access check by opening the path for writing — which is exactly
    // what must never happen to a FIFO: `open(2)` for write blocks until a reader appears, with no
    // timeout and no kill-on-drop, wedging the whole turn from inside a blocking syscall. The kind
    // check therefore gates the access check; a non-regular path reports `writable: false` and its
    // caller refuses it on `kind` alone.
    let writable = match kind {
        FileKind::File | FileKind::Dir => crate::tools::is_writable(&path.to_string_lossy()),
        FileKind::Other => false,
    };
    Some(Meta {
        kind,
        len: meta.len(),
        mtime: meta.modified().ok(),
        writable,
    })
}

/// The blocking glob walk, moved from `find`. Shares `LocalFs`'s `.gitignore` policy with
/// [`walk`] — `hidden(false)` to include dotfiles, and `require_git(false)` only outside a real
/// repository so a nested repo's own ignores can't leak across its boundary.
fn glob_blocking(q: &GlobQuery) -> Result<GlobOutcome, FsError> {
    use globset::GlobBuilder;

    // `literal_separator` is deliberately left at its default (`false`), matching what `find` compiled
    // before this seam existed: `**/` prefixing is done by the tool when it builds the pattern, so
    // forcing `/` to be literal here would change which paths a `*.rs` pattern reaches.
    let matcher = GlobBuilder::new(&q.pattern)
        .case_insensitive(q.case_insensitive)
        .build()
        .map_err(|e| FsError::InvalidQuery(format!("bad glob: {e}")))?
        .compile_matcher();

    let mut builder = WalkBuilder::new(&q.root);
    builder.hidden(false);
    if !crate::tools::root_is_inside_git_repo(&q.root) {
        builder.require_git(false);
    }

    let mut paths: Vec<(PathBuf, bool)> = Vec::new();
    let mut first_error: Option<String> = None;
    let mut hit_hard_cap = false;
    for entry in builder.build() {
        if paths.len() >= super::HARD_CAP {
            hit_hard_cap = true;
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            // Recorded rather than merely skipped, so an otherwise-empty result can say "couldn't
            // fully search" instead of a confidently wrong "no files matching".
            Err(e) => {
                first_error.get_or_insert_with(|| e.to_string());
                continue;
            }
        };
        let path = entry.path();
        // Skip the search root itself, but match both files *and* directories — a directory-name
        // pattern like `node_modules` would otherwise return nothing.
        if path == q.root {
            continue;
        }
        // `to_string_lossy()` borrows for the overwhelmingly common valid-UTF-8 path, so nothing is
        // allocated per walked entry — only per match.
        let candidate = if q.basename_only {
            path.file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or(std::borrow::Cow::Borrowed(""))
        } else {
            path.to_string_lossy()
        };
        if matcher.is_match(&*candidate) {
            // The type already cached from the directory read that produced this entry — cheaper than
            // a fresh `stat`, and matches real `fd`'s default of marking a directory match.
            let is_dir = entry.file_type().is_some_and(|ft| ft.is_dir());
            paths.push((path.to_path_buf(), is_dir));
        }
    }
    let (paths, truncated) = super::finalize_glob(paths, q.limit, hit_hard_cap);
    Ok(GlobOutcome {
        paths,
        truncated,
        first_error,
    })
}

/// Compile the file glob, carrying the caller-visible error text unchanged. The bool is the
/// ripgrep-CLI-style negation the caller already stripped a leading `!` to produce.
fn compile_glob(glob: Option<&(String, bool)>) -> Result<Option<(GlobMatcher, bool)>, FsError> {
    match glob {
        None => Ok(None),
        Some((pattern, negate)) => {
            let g = Glob::new(pattern)
                .map_err(|e| FsError::InvalidQuery(format!("bad glob: {e}")))?
                .compile_matcher();
            Ok(Some((g, *negate)))
        }
    }
}

/// Compile the pattern. `literal` regex-escapes it (verbatim search); `ignore_case` folds case.
fn compile_matcher(q: &SearchQuery) -> Result<RegexMatcher, FsError> {
    let escaped;
    let effective = if q.literal {
        escaped = regex::escape(&q.pattern);
        escaped.as_str()
    } else {
        q.pattern.as_str()
    };
    RegexMatcherBuilder::new()
        .case_insensitive(q.ignore_case)
        .line_terminator(Some(b'\n'))
        .build(effective)
        .map_err(|e| FsError::InvalidQuery(format!("bad regex: {e}")))
}

/// Run the parallel search. `stop_at` and the hard cap count *matches*, not context lines.
#[allow(clippy::too_many_arguments)]
fn walk(
    matcher: &RegexMatcher,
    glob: Option<&(GlobMatcher, bool)>,
    root: &Path,
    stop_at: usize,
    limit: usize,
    before: usize,
    after: usize,
    threads: usize,
) -> SearchOutcome {
    let collected: Arc<Mutex<Vec<Hit>>> = Arc::new(Mutex::new(Vec::new()));
    let total = Arc::new(AtomicUsize::new(0));
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

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
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false);
    if !crate::tools::root_is_inside_git_repo(root) {
        builder.require_git(false);
    }
    builder.threads(threads).build_parallel().run(|| {
        let collected = Arc::clone(&collected);
        let total = Arc::clone(&total);
        let first_error = Arc::clone(&first_error);
        // One `Searcher` per worker thread (it carries reusable line buffers): ripgrep's engine —
        // byte-oriented, so non-UTF-8 files search correctly (a `BufReader::read_line` loop would
        // error out and silently drop them); binary files quit cleanly on a NUL; large files are
        // mmap'd; line counting is SIMD-accelerated. `before`/`after` context is built in.
        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .before_context(before)
            .after_context(after)
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
                // wrong "no matches".
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
                total: Arc::clone(&total),
                stop_at,
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

    let hits = Arc::try_unwrap(collected)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default();
    let stopped_early = total.load(Ordering::Relaxed) >= stop_at;
    let first_error = Arc::try_unwrap(first_error)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .flatten();
    let (hits, truncated) = finalize(hits, limit, stopped_early);
    SearchOutcome {
        hits,
        truncated,
        first_error,
    }
}

/// A [`Sink`] that gathers one file's matches and context lines. `matched`/`context` are called by the
/// searcher with byte-accurate line numbers; bytes are decoded lossily so non-UTF-8 files still yield
/// readable, matchable text instead of being dropped.
struct Collector {
    path: Arc<Path>,
    hits: Vec<Hit>,
    matches: usize,
    /// The walk's shared match counter and its stop threshold — so a single match-dense file (an
    /// unignored log dump, a minified bundle) can bail out of *its own* scan the instant the running
    /// total would cross `stop_at`, instead of always finishing the file and only re-checking the cap
    /// between files. Without this, one such file could be fully scanned and every hit collected
    /// before the cap logic ever saw it — an OOM/hang vector with no adversarial input required.
    total: Arc<AtomicUsize>,
    stop_at: usize,
}

impl Collector {
    /// This file's own matches so far, added to every earlier file's already-flushed count — the same
    /// running total the per-entry closure checks between files, just readable mid-file too.
    fn running_total(&self) -> usize {
        self.total.load(Ordering::Relaxed) + self.matches
    }
}

impl Sink for Collector {
    type Error = std::io::Error;

    fn matched(&mut self, _searcher: &Searcher, m: &SinkMatch<'_>) -> std::io::Result<bool> {
        let line = m.line_number().map(|n| n as usize).unwrap_or(0);
        let text = String::from_utf8_lossy(m.bytes());
        self.hits.push(Hit {
            path: self.path.clone(),
            line,
            text: clip(&trim_eol(&text)),
            is_match: true,
        });
        self.matches += 1;
        Ok(self.running_total() < self.stop_at)
    }

    fn context(&mut self, _searcher: &Searcher, c: &SinkContext<'_>) -> std::io::Result<bool> {
        let line = c.line_number().map(|n| n as usize).unwrap_or(0);
        let text = String::from_utf8_lossy(c.bytes());
        self.hits.push(Hit {
            path: self.path.clone(),
            line,
            text: clip(&trim_eol(&text)),
            is_match: false,
        });
        Ok(self.running_total() < self.stop_at)
    }
}

/// Build a [`SearchQuery`] with the defaults every caller wants — used by tests and the benchmark so
/// neither has to spell out seven fields to vary one.
#[doc(hidden)]
pub fn query(pattern: &str, root: PathBuf, limit: usize) -> SearchQuery {
    SearchQuery {
        pattern: pattern.to_string(),
        literal: false,
        ignore_case: false,
        glob: None,
        root,
        limit,
        before: 0,
        after: 0,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn collector_stops_scanning_a_single_file_once_the_cap_is_reached() {
        // Regression, moved here verbatim with `Collector` itself: `matched`/`context` must signal the
        // searcher to stop mid-file the instant the running total would cross `stop_at`, not just rely
        // on the per-entry closure re-checking the cap *between* files — see `Collector::running_total`.
        // Before this fix, a single match-dense file (an unignored log dump, a minified bundle) was
        // always scanned to completion regardless of the cap, an OOM/hang vector needing no
        // adversarial input.
        let matcher = RegexMatcherBuilder::new()
            .line_terminator(Some(b'\n'))
            .build("needle")
            .unwrap();
        let mut searcher = SearcherBuilder::new().line_number(true).build();
        let total = Arc::new(AtomicUsize::new(0));
        let mut sink = Collector {
            path: Arc::from(Path::new("f.txt")),
            hits: Vec::new(),
            matches: 0,
            total: Arc::clone(&total),
            stop_at: 3,
        };
        // 1,000 matching lines — if the searcher scanned the whole thing, `sink.hits.len()` would be
        // 1,000; it must instead stop the moment the 3rd match is recorded.
        let haystack = "needle\n".repeat(1_000);
        searcher
            .search_slice(&matcher, haystack.as_bytes(), &mut sink)
            .unwrap();
        assert_eq!(
            sink.hits.len(),
            3,
            "a match-dense file must stop being scanned once the cap is crossed, not run to \
             completion"
        );
    }
}
