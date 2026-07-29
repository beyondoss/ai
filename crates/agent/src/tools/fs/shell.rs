//! [`ShellFs`] — the same filesystem operations expressed as *commands*, over any
//! [`CommandRunner`](crate::tools::exec::CommandRunner).
//!
//! This backend assumes nothing is installed on the target beyond ordinary Linux utilities. It probes
//! once for what's actually there ([`Capabilities`]) and picks the best available translation, so the
//! same code works against a box we control and one we don't.
//!
//! Because it is generic over the runner rather than over a transport, it runs over
//! [`RealRunner`](crate::tools::exec::RealRunner) on the host — against the very same files
//! [`super::local::LocalFs`] sees. That is what makes the two backends differentially testable in CI
//! with no VM, no container, and no infrastructure, and it is the main reason the seam is shaped this
//! way.
//!
//! ## Never build a shell string
//!
//! [`CommandRunner::run`](crate::tools::exec::CommandRunner::run) takes `program` and `args`
//! separately, so every path and pattern here rides as its own argv entry and is never interpolated
//! into text a shell will parse. A model-supplied path of `'; rm -rf /` is inert — it is one argument
//! containing those characters, not a command. Nothing in this module may construct a command string;
//! if a pipeline ever becomes unavoidable, pass paths as positional parameters to a *fixed* script,
//! never as substituted text.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::{
    DirEntry, FileKind, FsBackend, FsError, GlobOutcome, GlobQuery, Hit, MAX_CONTEXT, Meta,
    PathWorld, SearchOutcome, SearchQuery, clip, finalize, trim_eol,
};
use crate::tools::exec::{CommandRunner, ExecResult};

/// How long a single backend command may run before it is reaped.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
/// Capability probes answer in milliseconds or not at all; they must never hold up an attach.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// What the target actually has, probed once at attach.
///
/// Recorded rather than assumed so the choice is *reportable*: an operator can see that a box fell
/// back to POSIX `grep` and therefore lost `.gitignore` awareness, instead of discovering it from a
/// search that walked `target/` and returned ten thousand irrelevant hits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Capabilities {
    /// `rg` is present. Its flags map nearly 1:1 onto [`SearchQuery`] and it uses the same regex
    /// engine as [`super::local::LocalFs`], so results match. Without it, see
    /// [`Capabilities::search_engine`].
    pub rg: bool,
    /// `grep -Z` — a NUL after the filename, so a path containing `:` or a newline stays parseable.
    /// GNU grep has it; **busybox does not**, and busybox is what Alpine ships, which is what a large
    /// share of real containers are.
    pub grep_null: bool,
    /// `find -printf` — type and size in one pass. GNU find and `bfs` have it; **busybox does not**.
    pub find_printf: bool,
}

/// Which program a search will actually use — the fallback ladder, named so it can be logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchEngine {
    /// `rg`. Same regex crate, same `.gitignore` semantics, same binary-file skipping as `LocalFs`.
    Ripgrep,
    /// GNU-flavored `grep -rnZIE`. **Does not honor `.gitignore`** — it will walk `target/` and
    /// `node_modules/` — and its ERE dialect is not Rust's regex dialect.
    PosixGrep,
    /// Busybox `grep -rn`, the rung Alpine gets. Everything `PosixGrep` gives up, plus: no `-Z`, so a
    /// path containing `:` cannot be told from the `path:line:` separator; no `--include`, so glob
    /// filtering happens host-side; and no `-I`, so binary files are not skipped by the tool.
    BusyboxGrep,
}

impl Capabilities {
    /// Probe the target. Each check is one argv-only command with a short timeout; a probe that errors
    /// is read as "absent" rather than failing the attach, because a missing optional tool is a
    /// degraded search, not a broken box.
    pub async fn probe(runner: &dyn CommandRunner) -> Self {
        async fn ok(runner: &dyn CommandRunner, program: &str, args: &[&str]) -> bool {
            runner
                .run(
                    program,
                    &args.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
                    None,
                    PROBE_TIMEOUT,
                )
                .await
                // Exit 1 means "ran fine, found nothing" for grep/find; only a usage error (2 on GNU
                // and busybox alike) means the flag is unsupported.
                .map(|r| matches!(r.code, Some(0) | Some(1)))
                .unwrap_or(false)
        }
        let rg = runner
            .run("rg", &["--version".to_string()], None, PROBE_TIMEOUT)
            .await
            .map(|r| r.code == Some(0))
            .unwrap_or(false);
        // Probed against `/` because it exists everywhere; the pattern matches nothing, so this is a
        // flag check rather than a search.
        let grep_null = ok(
            runner,
            "grep",
            &["-rnZ", "-e", "__cap_probe__", "--", "/dev/null"],
        )
        .await;
        let find_printf = ok(
            runner,
            "find",
            &["/dev/null", "-maxdepth", "0", "-printf", "%y"],
        )
        .await;
        Self {
            rg,
            grep_null,
            find_printf,
        }
    }

    /// Which engine [`ShellFs::search`] will use.
    pub fn search_engine(&self) -> SearchEngine {
        match (self.rg, self.grep_null) {
            (true, _) => SearchEngine::Ripgrep,
            (false, true) => SearchEngine::PosixGrep,
            (false, false) => SearchEngine::BusyboxGrep,
        }
    }
}

/// A filesystem reached by running commands on it.
pub struct ShellFs {
    runner: Arc<dyn CommandRunner>,
    caps: Capabilities,
    timeout: Duration,
    /// The target's `$HOME`, for expanding a leading `~`. `None` leaves `~` untouched rather than
    /// guessing — a wrong home silently produces a plausible path for the wrong user.
    home: Option<String>,
}

impl ShellFs {
    /// Build a backend over `runner`, probing the target for what it has.
    pub async fn connect(runner: Arc<dyn CommandRunner>) -> Self {
        let caps = Capabilities::probe(runner.as_ref()).await;
        Self::with_capabilities(runner, caps)
    }

    /// Build with capabilities already known — used by tests to exercise a specific rung of the
    /// fallback ladder (notably: pretending `rg` is absent on a box that has it, which is how the
    /// POSIX path gets real coverage instead of only running where nobody installed ripgrep).
    pub fn with_capabilities(runner: Arc<dyn CommandRunner>, caps: Capabilities) -> Self {
        Self {
            runner,
            caps,
            timeout: DEFAULT_TIMEOUT,
            home: None,
        }
    }

    /// Tell this backend the target's `$HOME`, so a model-supplied `~/notes.md` expands against the
    /// right user. Without it a leading `~` is left alone — see [`ShellFs::home`].
    pub fn with_home(mut self, home: impl Into<String>) -> Self {
        self.home = Some(home.into());
        self
    }

    pub fn capabilities(&self) -> Capabilities {
        self.caps
    }
}

#[async_trait]
impl FsBackend for ShellFs {
    /// Never [`PathWorld::Local`], even when the runner happens to be `RealRunner` and the files
    /// really are on this host — as they are throughout the differential test suite.
    ///
    /// That case is *exactly* why this is not conditional on the runner. A `ShellFs` whose world
    /// depended on which runner it held would take the local path in every test and the remote path
    /// only in production, so the resolution logic that ships would be the one never exercised. The
    /// world is a property of the abstraction, not of today's transport.
    fn world(&self) -> PathWorld {
        PathWorld::Remote {
            home: self.home.clone(),
        }
    }

    async fn search(&self, q: &SearchQuery) -> Result<SearchOutcome, FsError> {
        let engine = self.caps.search_engine();
        let args = match engine {
            SearchEngine::Ripgrep => rg_args(q),
            SearchEngine::PosixGrep => posix_grep_args(q, true),
            SearchEngine::BusyboxGrep => posix_grep_args(q, false),
        };
        let program = match engine {
            SearchEngine::Ripgrep => "rg",
            SearchEngine::PosixGrep | SearchEngine::BusyboxGrep => "grep",
        };

        let result: ExecResult = self
            .runner
            .run(program, &args, None, self.timeout)
            .await
            .map_err(|e| FsError::Backend(format!("{program}: {e}")))?;

        // Both `rg` and `grep` use exit 1 for "no matches" — a perfectly ordinary outcome, not a
        // failure. Anything above that is a real error, and the overwhelmingly common cause is a
        // pattern the target's regex dialect rejected, so it maps to the caller's mistake rather than
        // the backend's.
        match result.code {
            Some(0) | Some(1) => {}
            _ if result.timed_out => {
                return Err(FsError::Backend(format!(
                    "{program} timed out after {}s",
                    self.timeout.as_secs()
                )));
            }
            _ => {
                let detail = result.stderr.trim();
                return Err(FsError::InvalidQuery(format!(
                    "bad regex: {}",
                    if detail.is_empty() {
                        format!("{program} rejected the pattern")
                    } else {
                        detail.to_string()
                    }
                )));
            }
        }

        let hits = if engine == SearchEngine::BusyboxGrep {
            // No `-Z`, so records are `path:line:text` with no unambiguous path terminator, and no
            // `--include`, so the glob is applied here instead.
            parse_colon_records(&result.stdout, q)
        } else {
            parse_records(&result.stdout)
        };
        // `ExecResult` keeps only the first and last 128 KiB of each stream and drops the middle. On a
        // large result set that silently removes records from the middle of the output — so the flag
        // must reach the outcome, or this backend would report a confidently incomplete result as
        // complete. Treated as "stopped early", which is exactly what it is.
        let (hits, truncated) = finalize(hits, q.limit, result.truncated);
        Ok(SearchOutcome {
            hits,
            truncated,
            // Unlike the in-process walker, a command gives us no per-path error stream to mine: an
            // unreadable subdirectory shows up only as noise on stderr, in a format that varies by
            // implementation. Reported when the command still succeeded but said something — enough
            // for the "search was incomplete" notice to fire, without pretending to name the path.
            first_error: incomplete_note(&result),
        })
    }

    async fn stat(&self, path: &Path) -> Result<Option<Meta>, FsError> {
        let args = sh_script(STAT_SCRIPT, &[path.to_string_lossy().into_owned()]);
        let result = self.exec("sh", &args).await?;
        // A missing path is a non-zero exit with a message on stderr — an ordinary branch, not a
        // failure, so it maps to `None` exactly as `LocalFs`'s `metadata().ok()?` does.
        if result.code != Some(0) || result.stdout.trim().is_empty() {
            return Ok(None);
        }
        Ok(parse_stat(&result.stdout))
    }

    async fn read_bytes(&self, path: &Path, offset: u64, max: usize) -> Result<Vec<u8>, FsError> {
        let args = sh_script(
            READ_WINDOW_SCRIPT,
            &[
                path.to_string_lossy().into_owned(),
                offset.to_string(),
                max.to_string(),
            ],
        );
        let result = self.exec("sh", &args).await?;
        if result.code != Some(0) {
            return Err(FsError::Backend(format!(
                "read {}: {}",
                path.display(),
                first_line_or(&result.stderr, "command failed")
            )));
        }
        decode_base64(result.stdout.trim())
            .map_err(|e| FsError::Backend(format!("read {}: {e}", path.display())))
    }

    async fn write_bytes(&self, path: &Path, bytes: &[u8]) -> Result<(), FsError> {
        let args = sh_script(
            WRITE_ATOMIC_SCRIPT,
            &[path.to_string_lossy().into_owned(), encode_base64(bytes)],
        );
        let result = self.exec("sh", &args).await?;
        if result.code != Some(0) {
            return Err(FsError::Backend(format!(
                "write {}: {}",
                path.display(),
                first_line_or(&result.stderr, "command failed")
            )));
        }
        Ok(())
    }

    async fn write_if_unchanged(
        &self,
        path: &Path,
        bytes: &[u8],
        expected: Option<std::time::SystemTime>,
    ) -> Result<bool, FsError> {
        // Two round trips, and therefore a genuinely wider race window than `LocalFs`'s single
        // blocking check-then-write. This is honest rather than hidden: the guard exists to catch a
        // *human or another agent* editing the file between this edit's read and its write — a
        // multi-second window — not to be a lock. Narrowing it to one command would mean shipping the
        // mtime comparison into the shell script, where a mismatch could not be reported as cleanly.
        let current = self.stat(path).await?.and_then(|m| m.mtime);
        if !mtimes_match(current, expected) {
            return Ok(false);
        }
        self.write_bytes(path, bytes).await?;
        Ok(true)
    }

    async fn create_dir_all(&self, path: &Path) -> Result<(), FsError> {
        let result = self
            .exec(
                "mkdir",
                &[
                    "-p".to_string(),
                    "--".to_string(),
                    path.to_string_lossy().into_owned(),
                ],
            )
            .await?;
        if result.code != Some(0) {
            return Err(FsError::Backend(format!(
                "create {}: {}",
                path.display(),
                first_line_or(&result.stderr, "command failed")
            )));
        }
        Ok(())
    }

    async fn list_dir(
        &self,
        path: &Path,
        cap: usize,
        include_hidden: bool,
    ) -> Result<Vec<DirEntry>, FsError> {
        let root = path.to_string_lossy().into_owned();
        if self.caps.find_printf {
            // `%Y` is the type *after* following a symlink, so a link to a directory lists as a
            // directory — matching `LocalFs`'s use of `metadata` rather than `symlink_metadata`. A
            // broken link reports `N`/`?` and is skipped, matching the local `metadata().ok()` skip.
            // NUL record separators, because a filename may contain a newline.
            let result = self
                .exec(
                    "find",
                    &[
                        root.clone(),
                        "-mindepth".into(),
                        "1".into(),
                        "-maxdepth".into(),
                        "1".into(),
                        "-printf".into(),
                        "%Y\\t%s\\t%f\\0".into(),
                    ],
                )
                .await?;
            if result.code != Some(0) && result.stdout.is_empty() {
                return Err(FsError::Backend(format!(
                    "ls {}: {}",
                    path.display(),
                    first_line_or(&result.stderr, "command failed")
                )));
            }
            return Ok(parse_dir_entries(&result.stdout, cap, include_hidden));
        }

        // Busybox has no `-printf`, so type comes from a second pass restricted to directories. Two
        // execs instead of one; `ls` renders only the directory distinction (the trailing `/`) and
        // never reads a size, so nothing else is lost.
        let listing = self
            .exec(
                "find",
                &[
                    root.clone(),
                    "-mindepth".into(),
                    "1".into(),
                    "-maxdepth".into(),
                    "1".into(),
                    "-print0".into(),
                ],
            )
            .await?;
        if listing.code != Some(0) && listing.stdout.is_empty() {
            return Err(FsError::Backend(format!(
                "ls {}: {}",
                path.display(),
                first_line_or(&listing.stderr, "command failed")
            )));
        }
        let dirs = self
            .exec(
                "find",
                &[
                    root,
                    "-mindepth".into(),
                    "1".into(),
                    "-maxdepth".into(),
                    "1".into(),
                    "-type".into(),
                    "d".into(),
                    "-print0".into(),
                ],
            )
            .await?;
        let dir_set: std::collections::BTreeSet<&str> =
            dirs.stdout.split('\0').filter(|r| !r.is_empty()).collect();
        let mut out = Vec::new();
        for raw in listing.stdout.split('\0') {
            if raw.is_empty() || out.len() >= cap {
                continue;
            }
            let name = Path::new(raw)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.is_empty() || (!include_hidden && name.starts_with('.')) {
                continue;
            }
            out.push(DirEntry {
                name,
                kind: if dir_set.contains(raw) {
                    FileKind::Dir
                } else {
                    FileKind::File
                },
                len: 0,
            });
        }
        Ok(out)
    }

    async fn glob(&self, q: &GlobQuery) -> Result<GlobOutcome, FsError> {
        // The target enumerates candidate paths; the **glob match runs host-side against our own
        // `globset`**. That is deliberate: it makes glob semantics — `**`, character classes, case
        // folding — structurally incapable of differing from `LocalFs`, because it is literally the
        // same matcher. Only *which paths get enumerated* can differ, which is what the two rungs
        // below are careful about.
        let matcher = compile_glob_matcher(q)?;
        let root = q.root.to_string_lossy().into_owned();
        let mut entries: Vec<(PathBuf, bool)> = Vec::new();
        let stderr_note;

        match self.caps.search_engine() {
            SearchEngine::Ripgrep => {
                // Pass 1: `rg --files` — a `.gitignore`-aware listing of *files*, the same set
                // `LocalFs`'s walk yields. `--null` for the reason the search parser scans for NUL:
                // the default is newline-separated, so a filename legally containing a newline would
                // be split into two bogus paths and the real one lost.
                let mut a = vec![
                    "--files".to_string(),
                    "--null".to_string(),
                    "--hidden".to_string(),
                ];
                if !crate::tools::root_is_inside_git_repo(&q.root) {
                    a.push("--no-require-git".to_string());
                }
                a.push("--".to_string());
                a.push(root.clone());
                let files = self.exec("rg", &a).await?;
                if files.code != Some(0)
                    && files.stdout.is_empty()
                    && !files.stderr.trim().is_empty()
                {
                    return Err(FsError::Backend(format!(
                        "rg: {}",
                        first_line_or(&files.stderr, "command failed")
                    )));
                }
                stderr_note = incomplete_note(&files);
                for raw in files.stdout.split('\0').filter(|r| !r.is_empty()) {
                    entries.push((PathBuf::from(raw), false));
                }

                // Pass 2: what `rg --files` structurally cannot report, both found by randomized
                // differential testing rather than by inspection.
                //
                //   * **Symlinks.** `rg` neither follows nor lists them; `LocalFs`'s walk yields the
                //     entry itself (its `file_type` is lstat-like, so a link — even a broken one — is
                //     a file and matches). Their absence was a *false negative*, the invisible kind.
                //   * **Empty directories.** Non-empty directories are derived below from the
                //     ancestors of listed files, which inherits ignore-awareness for free; an empty
                //     one has no files and so no ancestors to derive it from.
                //
                // Narrow, accepted consequence: this pass is not `.gitignore`-filtered, so an ignored
                // symlink or ignored *empty* directory can appear here where it would not locally.
                // That trades an invisible false negative for a visible false positive.
                let extra = self
                    .exec(
                        "find",
                        &[
                            root.clone(),
                            "(".into(),
                            "-type".into(),
                            "l".into(),
                            "-o".into(),
                            "(".into(),
                            "-type".into(),
                            "d".into(),
                            "-empty".into(),
                            ")".into(),
                            ")".into(),
                            "-printf".into(),
                            "%y\\t%p\\0".into(),
                        ],
                    )
                    .await?;
                entries.extend(parse_typed_paths(&extra.stdout));
            }
            SearchEngine::PosixGrep | SearchEngine::BusyboxGrep => {
                // One pass when `-printf` exists: with no ignore-awareness to preserve there is
                // nothing to derive, so every entry is enumerated directly *with its type* — which is
                // also how a directory keeps its trailing slash in the rendered output. Busybox has
                // no `-printf`, so it takes two `-print0` passes instead (all, then directories).
                if !self.caps.find_printf {
                    let all = self.exec("find", &[root.clone(), "-print0".into()]).await?;
                    if all.code != Some(0) && all.stdout.is_empty() && !all.stderr.trim().is_empty()
                    {
                        return Err(FsError::Backend(format!(
                            "find: {}",
                            first_line_or(&all.stderr, "command failed")
                        )));
                    }
                    let dirs = self
                        .exec(
                            "find",
                            &[root.clone(), "-type".into(), "d".into(), "-print0".into()],
                        )
                        .await?;
                    let dir_set: std::collections::BTreeSet<&str> =
                        dirs.stdout.split('\0').filter(|r| !r.is_empty()).collect();
                    stderr_note = incomplete_note(&all);
                    for raw in all.stdout.split('\0').filter(|r| !r.is_empty()) {
                        entries.push((PathBuf::from(raw), dir_set.contains(raw)));
                    }
                    let matcher2 = matcher;
                    let mut paths: Vec<(PathBuf, bool)> = Vec::new();
                    let mut hit_hard_cap = false;
                    for (p, is_dir) in entries {
                        if paths.len() >= super::HARD_CAP {
                            hit_hard_cap = true;
                            break;
                        }
                        if p == q.root {
                            continue;
                        }
                        let candidate = if q.basename_only {
                            p.file_name()
                                .map(|n| n.to_string_lossy())
                                .unwrap_or(std::borrow::Cow::Borrowed(""))
                        } else {
                            p.to_string_lossy()
                        };
                        if matcher2.is_match(&*candidate) {
                            paths.push((p.clone(), is_dir));
                        }
                    }
                    let (paths, truncated) = super::finalize_glob(paths, q.limit, hit_hard_cap);
                    return Ok(GlobOutcome {
                        paths,
                        truncated,
                        first_error: stderr_note,
                    });
                }
                let all = self
                    .exec(
                        "find",
                        &[root.clone(), "-printf".into(), "%y\\t%p\\0".into()],
                    )
                    .await?;
                if all.code != Some(0) && all.stdout.is_empty() && !all.stderr.trim().is_empty() {
                    return Err(FsError::Backend(format!(
                        "find: {}",
                        first_line_or(&all.stderr, "command failed")
                    )));
                }
                stderr_note = incomplete_note(&all);
                entries.extend(parse_typed_paths(&all.stdout));
            }
        }

        // Derive the non-empty directories from the listed files' ancestors (ripgrep rung only — the
        // POSIX rung already enumerated them). Bounded by the root, and dedup'd against what is
        // already present so an empty directory found above is not added twice.
        if matches!(self.caps.search_engine(), SearchEngine::Ripgrep) {
            let known: std::collections::BTreeSet<PathBuf> =
                entries.iter().map(|(p, _)| p.clone()).collect();
            let mut derived: std::collections::BTreeSet<PathBuf> = Default::default();
            for (p, _) in &entries {
                let mut cur = p.parent();
                while let Some(d) = cur {
                    if d == q.root || d.as_os_str().is_empty() || known.contains(d) {
                        break;
                    }
                    if !derived.insert(d.to_path_buf()) {
                        break; // this ancestor chain is already recorded
                    }
                    cur = d.parent();
                }
            }
            entries.extend(derived.into_iter().map(|d| (d, true)));
        }

        let mut paths: Vec<(PathBuf, bool)> = Vec::new();
        let mut hit_hard_cap = false;
        for (p, is_dir) in entries {
            if paths.len() >= super::HARD_CAP {
                hit_hard_cap = true;
                break;
            }
            if p == q.root {
                continue;
            }
            let candidate = if q.basename_only {
                p.file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or(std::borrow::Cow::Borrowed(""))
            } else {
                p.to_string_lossy()
            };
            if matcher.is_match(&*candidate) {
                paths.push((p.clone(), is_dir));
            }
        }
        let (paths, truncated) = super::finalize_glob(paths, q.limit, hit_hard_cap);
        Ok(GlobOutcome {
            paths,
            truncated,
            first_error: stderr_note,
        })
    }
}

impl ShellFs {
    /// One command, one place: every operation's timeout, error wrapping and argv discipline.
    async fn exec(&self, program: &str, args: &[String]) -> Result<ExecResult, FsError> {
        self.runner
            .run(program, args, None, self.timeout)
            .await
            .map_err(|e| FsError::Backend(format!("{program}: {e}")))
    }
}

/// Compile the glob with exactly the settings [`super::local::LocalFs`] uses, so the two backends
/// share one definition of what a pattern means.
fn compile_glob_matcher(q: &GlobQuery) -> Result<globset::GlobMatcher, FsError> {
    globset::GlobBuilder::new(&q.pattern)
        .case_insensitive(q.case_insensitive)
        .build()
        .map(|g| g.compile_matcher())
        .map_err(|e| FsError::InvalidQuery(format!("bad glob: {e}")))
}

/// Parse `w|-` + `\t<type>\t<size>\t<mtime>` from the one-shot stat command.
fn parse_stat(stdout: &str) -> Option<Meta> {
    let line = stdout.lines().next()?;
    let mut parts = line.split('\t');
    let writable = parts.next()? == "w";
    let kind = match parts.next()? {
        "f" => FileKind::File,
        "d" => FileKind::Dir,
        _ => FileKind::Other,
    };
    let len: u64 = parts.next()?.trim().parse().ok()?;
    // `%T@` is fractional seconds; only the whole-second part is comparable to a `SystemTime` built
    // from the same source on a later call, which is all `write_if_unchanged` needs.
    let secs: u64 = parts
        .next()
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    Some(Meta {
        kind,
        len,
        mtime: Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs)),
        writable,
    })
}

/// Parse busybox `grep -rn` output: `<path><sep><line><sep><text>`, where `<sep>` is `:` for a match
/// and `-` for a context line.
///
/// Without `-Z` there is no unambiguous end-of-path marker, so the split is the **first** separator
/// followed by digits followed by the same separator. That is correct for every path without a `:` or
/// `-` immediately preceding digits, and this is the documented cost of the busybox rung — see
/// [`SearchEngine::BusyboxGrep`]. A path containing a newline cannot be represented at all here,
/// which is why the NUL-capable rungs are preferred whenever available.
///
/// The `glob` is applied here too, because busybox `grep` has no `--include`. It is the same
/// `globset` matcher [`super::local::LocalFs`] uses, so glob *semantics* stay identical; only where
/// the filtering happens changes.
fn parse_colon_records(stdout: &str, q: &SearchQuery) -> Vec<Hit> {
    let glob = q.glob.as_ref().and_then(|(pattern, negate)| {
        globset::GlobBuilder::new(pattern)
            .case_insensitive(q.ignore_case)
            .build()
            .ok()
            .map(|g| (g.compile_matcher(), *negate))
    });
    let mut hits = Vec::new();
    for line in stdout.lines() {
        if line == "--" {
            continue; // context-group separator
        }
        let Some((path, line_no, is_match, text)) = split_colon_record(line) else {
            continue;
        };
        if let Some((matcher, negate)) = &glob {
            let matched = matcher.is_match(Path::new(path));
            if if *negate { matched } else { !matched } {
                continue;
            }
        }
        hits.push(Hit {
            path: Arc::from(Path::new(path)),
            line: line_no,
            text: clip(&trim_eol(text)),
            is_match,
        });
    }
    hits
}

/// Find the first `<sep><digits><sep>` boundary and split around it.
fn split_colon_record(line: &str) -> Option<(&str, usize, bool, &str)> {
    let bytes = line.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if c != b':' && c != b'-' {
            continue;
        }
        let rest = &bytes[i + 1..];
        let digits = rest.iter().take_while(|b| b.is_ascii_digit()).count();
        if digits == 0 || rest.get(digits) != Some(&c) {
            continue;
        }
        let line_no: usize = line[i + 1..i + 1 + digits].parse().ok()?;
        return Some((&line[..i], line_no, c == b':', &line[i + 1 + digits + 1..]));
    }
    None
}

/// Parse NUL-terminated `<type>\t<path>` records from `find -printf '%y\t%p\0'`.
fn parse_typed_paths(stdout: &str) -> Vec<(PathBuf, bool)> {
    let mut out = Vec::new();
    for record in stdout.split('\0') {
        if record.is_empty() {
            continue;
        }
        let Some((kind, path)) = record.split_once('\t') else {
            continue;
        };
        // `%y` is the entry's own type (lstat-like), matching `LocalFs`, where a symlink — including
        // one pointing at a directory — is reported as a non-directory and rendered without a slash.
        out.push((PathBuf::from(path), kind == "d"));
    }
    out
}

/// Parse NUL-terminated `<type>\t<size>\t<name>` records.
///
/// Hidden entries are dropped *before* the cap is applied, matching the local backend — see
/// [`FsBackend::list_dir`](super::FsBackend::list_dir).
fn parse_dir_entries(stdout: &str, cap: usize, include_hidden: bool) -> Vec<DirEntry> {
    let mut out = Vec::new();
    for record in stdout.split('\0') {
        if record.is_empty() || out.len() >= cap {
            break;
        }
        let mut parts = record.splitn(3, '\t');
        let Some(kind) = parts.next() else { continue };
        let Some(size) = parts.next() else { continue };
        let Some(name) = parts.next() else { continue };
        if !include_hidden && name.starts_with('.') {
            continue;
        }
        let kind = match kind {
            "f" => FileKind::File,
            "d" => FileKind::Dir,
            // `N` (broken link) / `?` (unstattable) — skipped, matching the local backend's
            // `metadata().ok()` skip rather than reported as an entry of unknown type.
            "N" | "?" => continue,
            _ => FileKind::Other,
        };
        out.push(DirEntry {
            name: name.to_string(),
            kind,
            len: size.trim().parse().unwrap_or(0),
        });
    }
    out
}

/// Compare two mtimes at whole-second granularity.
///
/// The remote path can only report whole seconds (`%T@`'s fractional part is not reliably
/// round-trippable), so comparing at finer resolution would make every guarded write spuriously
/// report "changed". Coarser comparison can only ever *miss* a change made within the same second as
/// the read — which for `edit`'s guard means falling back to the behavior of not having a guard, not
/// to a wrong answer.
fn mtimes_match(a: Option<std::time::SystemTime>, b: Option<std::time::SystemTime>) -> bool {
    fn secs(t: Option<std::time::SystemTime>) -> Option<u64> {
        t?.duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs())
    }
    secs(a) == secs(b)
}

fn first_line_or<'a>(stderr: &'a str, fallback: &'a str) -> &'a str {
    stderr.trim().lines().next().unwrap_or(fallback)
}

/// Minimal base64, both directions. Written here rather than pulled in as a dependency: the crate
/// needs exactly these two functions, over data it already holds in memory, and the encoding is
/// twenty lines that can be tested exhaustively against a round trip.
fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn decode_base64(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }
    let cleaned: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(cleaned.len() / 4 * 3);
    for chunk in cleaned.chunks(4) {
        let mut n = 0u32;
        for (i, c) in chunk.iter().enumerate() {
            let v = val(*c).ok_or_else(|| format!("invalid base64 byte {c:?}"))?;
            n |= v << (18 - 6 * i);
        }
        // A 4-char group carries 3 bytes, a 3-char tail 2, a 2-char tail 1.
        let keep = match chunk.len() {
            4 => 3,
            3 => 2,
            2 => 1,
            _ => return Err("truncated base64".to_string()),
        };
        for i in 0..keep {
            out.push((n >> (16 - 8 * i)) as u8);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The non-search operations.
//
// Two constraints shape all of these.
//
// 1. `ExecResult::stdout` is a `String`, so any byte a command emits that isn't valid UTF-8 is
//    lossily replaced before this code ever sees it. Reading a PNG through it would silently corrupt
//    the image. Every operation that moves file *content* therefore base64s it on the target — pure
//    ASCII, losslessly survivable — and decodes host-side.
// 2. A byte *window* (`read_bytes(path, offset, max)`) has no argv-only spelling: `head -c` cannot
//    skip, `base64` cannot window. These are the module's only pipelines, and they use the sanctioned
//    form — a **fixed** script with paths passed as positional parameters, never interpolated.

/// Build argv for `sh -c <fixed script> sh <arg>...`.
///
/// The script text is always a compile-time constant and the caller's values arrive as `$1`, `$2`, …
/// so a path of `'; rm -rf /` is a positional parameter's *value*, not syntax the shell parses. This
/// is the only sanctioned way to run a pipeline here; see the module doc.
fn sh_script(script: &'static str, args: &[String]) -> Vec<String> {
    let mut v = vec!["-c".to_string(), script.to_string(), "sh".to_string()];
    v.extend(args.iter().cloned());
    v
}

/// One `stat`, one round trip: writability, kind, size and mtime together.
///
/// `-writable` runs a real access check as the invoking user rather than reporting mode bits, which
/// is what [`Meta::writable`] promises. `%y` is the entry's own type and `%s`/`%T@` its size and
/// mtime. Works on GNU find and on `bfs`, which is what `find` actually is on some hosts.
/// One `stat`, one round trip, portable everywhere.
///
/// Built from shell builtins (`[ -e ]`, `[ -d ]`, `[ -w ]`) plus `stat -c`, rather than
/// `find -printf`, because busybox `find` has no `-printf` — and busybox is what Alpine ships, which
/// is what a large share of real sandboxes are. The `[ -w ]` test is a real access check as the
/// invoking user, which is what `Meta::writable` promises.
const STAT_SCRIPT: &str = r#"[ -e "$1" ] || exit 1
if [ -d "$1" ]; then t=d
elif [ -f "$1" ]; then t=f
else t=o
fi
if [ -w "$1" ]; then w=w; else w=-; fi
printf '%s	%s	' "$w" "$t"
stat -c '%s	%Y' -- "$1" 2>/dev/null || printf '0	0'
printf '
'"#;

/// Read a byte window and base64 it. `dd`'s `skip_bytes`/`count_bytes` are what make an arbitrary
/// offset expressible at all.
///
/// **The readability guard is load-bearing, not defensive.** A shell pipeline exits with the status of
/// its *last* command, and POSIX `sh` has no `pipefail`. Without the guard, `dd` failing on a missing
/// or unreadable file still leaves `base64` succeeding on empty input, so the whole command exits 0
/// with empty output — and a missing file reads back as an *empty file* rather than an error. The
/// directory test is part of the same check: `-r` is true for a directory, but `dd` cannot read one.
/// The guard reports the *specific* condition using the same wording `std::io::Error` renders for it,
/// because the error text is part of what the model reads and must not depend on which backend ran.
/// A randomized differential run caught this: every missing-file read diverged, local saying
/// "No such file or directory (os error 2)" and this saying "cannot read <path>". These are the
/// standard messages for the conditions actually being tested (`-e` is false for a broken symlink
/// too, matching `File::open` following it to ENOENT), not invented errnos.
const READ_WINDOW_SCRIPT: &str = r#"if [ ! -e "$1" ]; then
  echo "No such file or directory (os error 2)" >&2
  exit 1
fi
if [ -d "$1" ]; then
  echo "Is a directory (os error 21)" >&2
  exit 1
fi
if [ ! -r "$1" ]; then
  echo "Permission denied (os error 13)" >&2
  exit 1
fi
exec dd if="$1" iflag=skip_bytes,count_bytes skip="$2" count="$3" status=none | base64 -w0"#;

/// Decode base64 into a sibling temp file and `mv` it over the target — the same
/// write-temp-then-rename shape [`crate::tools::write_atomic`] uses locally, so a reader or a crash
/// sees the old file or the new one and never a partial. The temp is removed on any failure.
const WRITE_ATOMIC_SCRIPT: &str = r#"tmp="$1.tmp.$$"
if printf %s "$2" | base64 -d > "$tmp"; then
  mv -f "$tmp" "$1" || { rm -f "$tmp"; exit 1; }
else
  rm -f "$tmp"; exit 1
fi"#;

/// stderr from a *successful* search — an unreadable path, typically. Empty stderr means nothing to
/// report, which is the common case.
fn incomplete_note(result: &ExecResult) -> Option<String> {
    let detail = result.stderr.trim();
    if detail.is_empty() {
        return None;
    }
    // Keep it to the first line: these tools emit one message per unreadable path, and the tool's
    // notice only has room to name the first anyway — matching `LocalFs`, which records only the
    // first error it saw.
    Some(detail.lines().next().unwrap_or(detail).to_string())
}

/// `rg` invocation. Flags chosen to match [`super::local::LocalFs`]'s walker configuration exactly:
/// `--hidden` mirrors `WalkBuilder::hidden(false)`, and `--no-require-git` mirrors the
/// `require_git(false)` applied outside a real repository so a plain checkout still honors its
/// `.gitignore`.
fn rg_args(q: &SearchQuery) -> Vec<String> {
    let mut args = vec![
        "--null".to_string(),
        "--line-number".to_string(),
        "--no-heading".to_string(),
        "--color=never".to_string(),
        "--hidden".to_string(),
    ];
    if !crate::tools::root_is_inside_git_repo(&q.root) {
        args.push("--no-require-git".to_string());
    }
    if q.ignore_case {
        args.push("--ignore-case".to_string());
    }
    if q.literal {
        args.push("--fixed-strings".to_string());
    }
    if let Some((pattern, negate)) = &q.glob {
        args.push("--glob".to_string());
        args.push(if *negate {
            format!("!{pattern}")
        } else {
            pattern.clone()
        });
    }
    push_context(&mut args, q);
    // `-e` and `--` so a pattern or path beginning with `-` is never read as a flag.
    args.push("-e".to_string());
    args.push(q.pattern.clone());
    args.push("--".to_string());
    args.push(q.root.to_string_lossy().into_owned());
    args
}

/// POSIX `grep` invocation — the no-ripgrep rung.
///
/// `-Z` makes the filename NUL-terminated, which is what lets [`parse_records`] serve both engines
/// unchanged. `-I` skips binary files, matching ripgrep's default (and `LocalFs`'s
/// `BinaryDetection::quit`). `-E` selects extended regex, without which the default BRE dialect would
/// treat `+`, `?`, `|` and `()` as literals — a silent and very confusing difference. It is still not
/// Rust's regex dialect: `\d`, `\b`, and non-greedy quantifiers are GNU extensions at best and absent
/// at worst. **This rung does not honor `.gitignore`.**
fn posix_grep_args(q: &SearchQuery, gnu: bool) -> Vec<String> {
    let mut args = vec!["-r".to_string(), "-n".to_string()];
    if gnu {
        // Busybox has neither: `-Z` (NUL after the filename) nor `-I` (skip binary files), and
        // rejects `--color` outright.
        args.push("-Z".to_string());
        args.push("-I".to_string());
        args.push("--color=never".to_string());
    }
    if q.literal {
        // `-F` and `-E` are mutually exclusive; a literal search needs no dialect at all.
        args.push("-F".to_string());
    } else {
        args.push("-E".to_string());
    }
    if q.ignore_case {
        args.push("-i".to_string());
    }
    // `--include`/`--exclude` are GNU-only; on busybox the glob is applied host-side instead (see
    // `parse_colon_records`), which is the same `globset` matcher `LocalFs` uses either way.
    if gnu && let Some((pattern, negate)) = &q.glob {
        args.push(if *negate {
            format!("--exclude={pattern}")
        } else {
            format!("--include={pattern}")
        });
    }
    push_context(&mut args, q);
    args.push("-e".to_string());
    args.push(q.pattern.clone());
    args.push("--".to_string());
    args.push(q.root.to_string_lossy().into_owned());
    args
}

/// `-B`/`-A` context flags, clamped the same way the query itself is. Shared because both engines
/// spell them identically and a divergence here would be invisible until someone asked for context.
fn push_context(args: &mut Vec<String>, q: &SearchQuery) {
    let before = q.before.min(MAX_CONTEXT);
    let after = q.after.min(MAX_CONTEXT);
    if before > 0 {
        args.push("-B".to_string());
        args.push(before.to_string());
    }
    if after > 0 {
        args.push("-A".to_string());
        args.push(after.to_string());
    }
}

/// Parse `<path>\0<line><sep><text>\n` records, where `<sep>` is `:` for a match and `-` for a context
/// line. Both `rg --null` and `grep -Z` emit exactly this, which is why one parser serves both.
///
/// **Scans for the NUL, not for newlines.** A filename may legally contain a newline, so splitting the
/// stream on `\n` first would shred such a record into pieces and attribute its text to a path that
/// doesn't exist. The NUL is unambiguous: everything from the start of a record to the NUL is the
/// path, and everything from the NUL to the next newline is `line`+`sep`+`text` (which cannot contain
/// a newline, because that is what ended the line).
///
/// Records with no NUL before the next newline are group separators (`--`, emitted between context
/// blocks) and are skipped.
fn parse_records(stdout: &str) -> Vec<Hit> {
    let bytes = stdout.as_bytes();
    let mut hits = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        // The group separator is matched *first*, against the exact bytes `--\n`. It has to be,
        // because it is the only NUL-less record and every other way of recognizing it (looking for
        // the next newline before the next NUL) would misfire on a path that legally contains a
        // newline. Matching `--\n` rather than a bare `--` keeps a file actually *named* `--` — whose
        // record is `--\0…` — on the path branch where it belongs.
        let rest_bytes = &bytes[i..];
        if rest_bytes.starts_with(b"--\n") {
            i += 3;
            continue;
        }
        if rest_bytes == b"--" {
            break; // a trailing separator with no final newline
        }
        let Some(nul) = memchr_from(bytes, i, b'\0') else {
            break; // no NUL anywhere in the remainder — nothing parseable is left
        };
        // Searched from the NUL, not from the record start: everything before the NUL is the path and
        // may contain newlines, so only the terminator *after* the path ends this record.
        let nl = memchr_from(bytes, nul, b'\n').unwrap_or(bytes.len());
        let path = &stdout[i..nul];
        let rest = &stdout[nul + 1..nl];
        if let Some(hit) = parse_body(path, rest) {
            hits.push(hit);
        }
        i = nl + 1;
    }
    hits
}

/// Split `<line><sep><text>` into its parts. Returns `None` for a record whose line number isn't
/// digits — malformed output from an implementation we didn't anticipate, dropped rather than
/// surfaced as a bogus hit at line 0.
fn parse_body(path: &str, rest: &str) -> Option<Hit> {
    let digits_end = rest.find(|c: char| !c.is_ascii_digit())?;
    if digits_end == 0 {
        return None;
    }
    let line: usize = rest[..digits_end].parse().ok()?;
    let sep = rest.as_bytes()[digits_end];
    let is_match = match sep {
        b':' => true,
        b'-' => false,
        _ => return None,
    };
    let text = &rest[digits_end + 1..];
    Some(Hit {
        path: Arc::from(Path::new(path)),
        line,
        // The same normalization `LocalFs`'s sink applies, from the same functions — see `super`.
        text: clip(&trim_eol(text)),
        is_match,
    })
}

/// `bytes[from..]`-relative byte search returning an absolute index. A tiny helper rather than pulling
/// `memchr` in for two call sites on strings that are already bounded by the runner's capture cap.
fn memchr_from(bytes: &[u8], from: usize, needle: u8) -> Option<usize> {
    bytes[from..]
        .iter()
        .position(|b| *b == needle)
        .map(|p| p + from)
}

/// Build a [`SearchQuery`] the way the `grep` tool does, for tests.
#[doc(hidden)]
pub fn query(pattern: &str, root: PathBuf, limit: usize) -> SearchQuery {
    super::local::query(pattern, root, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_match_record() {
        let hits = parse_records("src/a.rs\u{0}12:let x = 1;\n");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path.to_string_lossy(), "src/a.rs");
        assert_eq!(hits[0].line, 12);
        assert_eq!(hits[0].text, "let x = 1;");
        assert!(hits[0].is_match);
    }

    #[test]
    fn distinguishes_context_lines_from_matches() {
        let hits = parse_records("a.rs\u{0}1-before\na.rs\u{0}2:hit\na.rs\u{0}3-after\n");
        let flags: Vec<bool> = hits.iter().map(|h| h.is_match).collect();
        assert_eq!(flags, vec![false, true, false]);
    }

    #[test]
    fn skips_the_group_separator_between_context_blocks() {
        // Both engines emit a bare `--` between non-adjacent context blocks. It has no NUL, so it must
        // be skipped whole — not mistaken for a path that swallows the next record.
        let hits = parse_records("a.rs\u{0}1:one\n--\nb.rs\u{0}9:two\n");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].path.to_string_lossy(), "b.rs");
        assert_eq!(hits[1].line, 9);
    }

    #[test]
    fn a_newline_inside_a_filename_does_not_shred_the_record() {
        // The reason this parser scans for NUL rather than splitting on newlines. A path containing a
        // newline is legal on every Unix filesystem, and line-splitting would attribute "9:two" to a
        // path of "line2" that never existed.
        let hits = parse_records("we\nird.rs\u{0}9:two\n");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path.to_string_lossy(), "we\nird.rs");
        assert_eq!(hits[0].line, 9);
        assert_eq!(hits[0].text, "two");
    }

    #[test]
    fn a_colon_in_the_path_is_not_mistaken_for_the_separator() {
        // The NUL, not the first colon, ends the path — otherwise `odd:name.rs` would parse as a path
        // of "odd" with an unparseable line number.
        let hits = parse_records("odd:name.rs\u{0}3:body\n");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path.to_string_lossy(), "odd:name.rs");
        assert_eq!(hits[0].text, "body");
    }

    #[test]
    fn strips_carriage_returns_the_same_way_the_local_backend_does() {
        let hits = parse_records("a.rs\u{0}1:crlf line\r\n");
        assert_eq!(hits[0].text, "crlf line");
    }

    #[test]
    fn rg_args_put_the_pattern_behind_dash_e_and_the_path_behind_dash_dash() {
        // A pattern or path starting with `-` must never be read as a flag.
        let q = query("-v", PathBuf::from("-weird-dir"), 100);
        let args = rg_args(&q);
        let e = args.iter().position(|a| a == "-e").unwrap();
        assert_eq!(args[e + 1], "-v");
        let dd = args.iter().position(|a| a == "--").unwrap();
        assert_eq!(args[dd + 1], "-weird-dir");
    }

    #[test]
    fn a_negated_glob_becomes_a_bang_prefixed_rg_glob() {
        let mut q = query("x", PathBuf::from("."), 100);
        q.glob = Some(("*.test.rs".into(), true));
        let args = rg_args(&q);
        let g = args.iter().position(|a| a == "--glob").unwrap();
        assert_eq!(args[g + 1], "!*.test.rs");
    }

    #[test]
    fn a_negated_glob_becomes_an_exclude_for_posix_grep() {
        let mut q = query("x", PathBuf::from("."), 100);
        q.glob = Some(("*.test.rs".into(), true));
        assert!(
            posix_grep_args(&q, true)
                .iter()
                .any(|a| a == "--exclude=*.test.rs")
        );
        q.glob = Some(("*.rs".into(), false));
        assert!(
            posix_grep_args(&q, true)
                .iter()
                .any(|a| a == "--include=*.rs")
        );
    }

    #[test]
    fn literal_mode_uses_fixed_strings_and_never_pairs_f_with_e() {
        let mut q = query("a.b(c)", PathBuf::from("."), 100);
        q.literal = true;
        let args = posix_grep_args(&q, true);
        assert!(args.iter().any(|a| a == "-F"));
        assert!(
            !args.iter().any(|a| a == "-E"),
            "-F and -E are mutually exclusive: {args:?}"
        );
    }
}
