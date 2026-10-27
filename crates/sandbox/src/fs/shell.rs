//! [`ShellFs`] — the same filesystem operations expressed as *commands*, over any
//! [`CommandRunner`](crate::exec::CommandRunner).
//!
//! This backend assumes nothing is installed on the target beyond ordinary Linux utilities. It probes
//! once for what's actually there ([`Capabilities`]) and picks the best available translation, so the
//! same code works against a box we control and one we don't.
//!
//! Because it is generic over the runner rather than over a transport, it runs over
//! [`RealRunner`](crate::testing::Host) on the host — against the very same files
//! [`super::local::LocalFs`] sees. That is what makes the two backends differentially testable in CI
//! with no VM, no container, and no infrastructure, and it is the main reason the seam is shaped this
//! way.
//!
//! ## Never build a shell string
//!
//! [`CommandRunner::run`](crate::exec::CommandRunner::run) takes `program` and `args`
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
    PathWorld, SearchOutcome, SearchQuery, finalize, trim_eol,
};
use crate::exec::{CommandRunner, ExecResult};

/// How long a single backend command may run before it is reaped.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
/// Capability probes answer in milliseconds or not at all; they must never hold up an attach.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// What the stdin probe feeds `cat`, and must read back.
const STDIN_PROBE: &str = "stdin-probe";
/// The read pipeline on a known input; see [`Capabilities::no_read_window`].
const READ_WINDOW_PROBE: &str = "printf abc | dd iflag=skip_bytes,count_bytes skip=1 count=1 status=none 2>/dev/null | base64 -w0 2>/dev/null";
/// Raw bytes per argv chunk when stdin is unavailable: 48 KiB, which base64s to exactly 64 KiB — half
/// of Linux's 128 KiB cap on a single argv string (`MAX_ARG_STRLEN`), leaving the whole argv far under
/// the kernel's total `ARG_MAX` as well.
const WRITE_CHUNK: usize = 48 * 1024;
/// Raw bytes per `read_bytes` command: 96 KiB, which base64s to exactly 128 KiB — the head a runner's
/// capture keeps whole ([`crate::tools::exec`]'s `STREAM_HEAD`), so a chunk is never cut.
const READ_CHUNK: usize = 96 * 1024;
const _: () = assert!(READ_CHUNK.div_ceil(3) * 4 <= crate::exec::STREAM_HEAD);

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
    /// The runner demonstrably delivers stdin: `cat` echoed a probe string back. File writes then send
    /// their content on stdin in one call. Without it they go as argv-sized chunks, because an argv
    /// entry is capped at 128 KiB and a whole file does not fit in one.
    ///
    /// **Proven, never assumed.** A v1 exec endpoint ignores the stdin field it does not know and
    /// runs `cat` on empty input; `docker exec` without `-i` does the same. Writing through either as
    /// though stdin arrived would replace the file with an *empty* one.
    pub stdin: bool,
    /// The target's `dd` or `base64` lacks the flags a read leans on (`dd iflag=skip_bytes,count_bytes`,
    /// `base64 -w0`): older busybox, for one. Negative so an unprobed [`Capabilities::default`] reads
    /// as capable.
    ///
    /// **Reads then fail, loudly.** In a pipeline whose last command is `base64`, a `dd` that rejects
    /// its flags still leaves `base64` succeeding on empty input, so an existing file would read back
    /// as an *empty* one, and a model that believes a file is empty overwrites it.
    pub no_read_window: bool,
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
    /// Probe the target. Each check is one argv-only command with a short timeout. A probe that *ran*
    /// and gave the "absent" answer (non-zero exit, a command that cannot be started, a runner with
    /// no stdin) is read as absent rather than failing the attach, because a missing optional tool
    /// is a degraded search, not a broken box. A probe whose *request* failed (unreachable, `401`,
    /// a timeout, a protocol error) says nothing about the box and is returned: caching it as "no
    /// stdin, no `dd`" would turn an outage or a wrong bearer into a misleading failure later.
    pub async fn probe(runner: &dyn CommandRunner) -> std::io::Result<Self> {
        /// `Ok(None)`: the probe could not run on the target at all, which is its answer.
        fn settle(r: std::io::Result<ExecResult>) -> std::io::Result<Option<ExecResult>> {
            match r {
                Ok(r) => Ok(Some(r)),
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::NotFound
                            | std::io::ErrorKind::Unsupported
                            // The command ran and the endpoint answered; the answer was too big.
                            | std::io::ErrorKind::FileTooLarge
                    ) =>
                {
                    Ok(None)
                }
                Err(e) => Err(e),
            }
        }
        async fn ok(
            runner: &dyn CommandRunner,
            program: &str,
            args: &[&str],
        ) -> std::io::Result<bool> {
            Ok(settle(
                runner
                    .run(
                        program,
                        &args.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
                        None,
                        PROBE_TIMEOUT,
                    )
                    .await,
            )?
            // Exit 1 means "ran fine, found nothing" for grep/find; only a usage error (2 on GNU
            // and busybox alike) means the flag is unsupported.
            .is_some_and(|r| matches!(r.code, Some(0) | Some(1))))
        }
        let rg = settle(
            runner
                .run("rg", &["--version".to_string()], None, PROBE_TIMEOUT)
                .await,
        )?
        .is_some_and(|r| r.code == Some(0));
        // Probed against `/` because it exists everywhere; the pattern matches nothing, so this is a
        // flag check rather than a search.
        let grep_null = ok(
            runner,
            "grep",
            &["-rnZ", "-e", "__cap_probe__", "--", "/dev/null"],
        )
        .await?;
        let find_printf = ok(
            runner,
            "find",
            &["/dev/null", "-maxdepth", "0", "-printf", "%y"],
        )
        .await?;
        // An echo, not an `Ok`: only the bytes coming back prove they went in. See `Self::stdin`.
        let stdin = settle(
            runner
                .run_with_stdin("cat", &[], None, PROBE_TIMEOUT, STDIN_PROBE.as_bytes())
                .await,
        )?
        .is_some_and(|r| r.code == Some(0) && r.stdout == STDIN_PROBE);
        // The exact pipeline `READ_WINDOW_SCRIPT` ends in, on three bytes: `Yg==` is "b" and nothing
        // else. Judged on stdout, because the pipeline's exit status is `base64`'s alone.
        let no_read_window = !settle(
            runner
                .run(
                    "sh",
                    &["-c".to_string(), READ_WINDOW_PROBE.to_string()],
                    None,
                    PROBE_TIMEOUT,
                )
                .await,
        )?
        .is_some_and(|r| r.code == Some(0) && r.stdout.trim() == "Yg==");
        Ok(Self {
            rg,
            grep_null,
            find_printf,
            stdin,
            no_read_window,
        })
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
    /// Memoized "is this root inside a git repository", answered **on the target** — see
    /// [`ShellFs::root_in_git_repo`].
    ///
    /// A `Mutex<HashMap>` rather than anything fancier because the map holds one entry per distinct
    /// search root a session ever uses (in practice: one), and the lock is never held across an
    /// `await`. Probing per search would otherwise add a round trip to every `grep` and `find`.
    git_roots: std::sync::Mutex<std::collections::HashMap<PathBuf, bool>>,
}

impl ShellFs {
    /// The file helpers over any exec: [`connect`](Self::connect) by the layer-2 name.
    pub async fn over(exec: Arc<dyn CommandRunner>) -> std::io::Result<Self> {
        Self::connect(exec).await
    }

    /// Build a backend over `runner`, probing the target for what it has.
    ///
    /// Fails with the runner's own error when a probe request fails (see [`Capabilities::probe`]);
    /// nothing is learned from a probe that did not run.
    pub async fn connect(runner: Arc<dyn CommandRunner>) -> std::io::Result<Self> {
        let caps = Capabilities::probe(runner.as_ref()).await?;
        Ok(Self::with_capabilities(runner, caps))
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
            git_roots: std::sync::Mutex::new(std::collections::HashMap::new()),
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

    /// [`FsBackend::write_bytes`] over borrowed bytes, which `write_if_unchanged` also writes
    /// through.
    async fn write_slice(&self, path: &Path, bytes: &[u8]) -> Result<(), FsError> {
        let target = path.to_string_lossy().into_owned();
        // The temp's suffix. Chosen here rather than by the script's `$$` because the chunked path
        // spans several shells, and unpredictable for the reason `LocalFs`'s is: a preplanted name
        // cannot redirect the write. Every script resolves the target the same way (`RESOLVE_TARGET`)
        // and names the temp beside the file it resolves to.
        let suffix = super::temp_suffix();
        let outcome = if self.caps.stdin {
            let args = sh_script(
                WRITE_STDIN_SCRIPT,
                &[target.clone(), suffix.clone(), bytes.len().to_string()],
            );
            self.runner
                .run_with_stdin("sh", &args, None, self.timeout, bytes)
                .await
                .map_err(|e| FsError::Backend(format!("sh: {e}")))
        } else {
            self.write_chunks(&target, &suffix, bytes).await
        };
        match outcome {
            Ok(result) if result.code == Some(0) => Ok(()),
            failed => {
                // The compensating action. The scripts remove the temp on every failure *they* see;
                // this covers the ones they cannot — a transport error or a timeout between chunks.
                let _ = self
                    .exec("sh", &sh_script(REMOVE_TEMP_SCRIPT, &[target, suffix]))
                    .await;
                Err(match failed {
                    Ok(result) => FsError::Backend(format!(
                        "write {}: {}",
                        path.display(),
                        first_line_or(&result.stderr, "command failed")
                    )),
                    Err(e) => e,
                })
            }
        }
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
            // The git-repo question is asked only on the rung that can act on it: `--no-require-git`
            // is a ripgrep flag, and the POSIX rungs honor no `.gitignore` at all.
            SearchEngine::Ripgrep => rg_args(q, self.root_in_git_repo(&q.root).await),
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
        // failure. Exit 2 is shared by three different situations that must not be conflated: a
        // pattern the target's regex dialect rejected (the caller's mistake), a missing root, and
        // per-path read errors (the matches found elsewhere are still good).
        let mut skipped_note = None;
        match result.code {
            Some(0) | Some(1) => {}
            _ if result.timed_out || result.code == Some(124) => {
                return Err(FsError::Backend(format!(
                    "{program} timed out after {}s",
                    self.timeout.as_secs()
                )));
            }
            _ => match classify_search_failure(&result.stderr, &q.root.to_string_lossy()) {
                SearchFailure::BadRegex => {
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
                SearchFailure::MissingRoot(line) => return Err(FsError::Backend(line)),
                SearchFailure::Skipped(note) => skipped_note = Some(note),
            },
        }

        let mut hits = if engine == SearchEngine::BusyboxGrep {
            // No `-Z`, so records are `path:line:text` with no unambiguous path terminator, and no
            // `--include`, so the glob is applied here instead.
            parse_colon_records(&result.stdout, q)
        } else {
            parse_records(&result.stdout)
        };
        // The engine was asked to skip `.git` where it can; this is the guarantee for the rungs that
        // can't (busybox `grep`, and a `.git` *file* GNU `--exclude-dir` does not cover).
        hits.retain(|h| !super::is_under_git_dir(&q.root, &h.path));
        // A long match line is cut to the window around its match, the same way `LocalFs` cuts it.
        // (A line longer than the runner's 128 KiB capture arrives already cut, which is flagged as
        // truncation below.)
        super::clip_hits(&mut hits, &super::MatchFinder::new(q));
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
            first_error: skipped_note.or_else(|| incomplete_note(&result)),
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

    /// One command per [`READ_CHUNK`] bytes, so every command's output fits the runner's capture
    /// whole. A window bigger than that cannot be one command: the capture keeps the head and tail of
    /// a stream and drops the middle, and base64 of a glued head and tail still decodes, into a file
    /// that never existed. A short chunk is end-of-file and ends the loop.
    async fn read_bytes(&self, path: &Path, offset: u64, max: usize) -> Result<Vec<u8>, FsError> {
        if self.caps.no_read_window {
            return Err(FsError::Backend(format!(
                "read {}: this target's dd or base64 does not support iflag=skip_bytes,count_bytes / -w0, so files cannot be read",
                path.display()
            )));
        }
        let target = path.to_string_lossy().into_owned();
        let mut out = Vec::new();
        loop {
            let want = (max - out.len()).min(READ_CHUNK);
            let args = sh_script(
                READ_WINDOW_SCRIPT,
                &[
                    target.clone(),
                    (offset + out.len() as u64).to_string(),
                    want.to_string(),
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
            // The backstop for a runner whose capture is smaller than the chunk: refuse a cut-short
            // read rather than return it as the file.
            if result.truncated {
                return Err(FsError::Backend(format!(
                    "read {}: the command's output was cut short",
                    path.display()
                )));
            }
            let chunk = decode_base64(result.stdout.trim())
                .map_err(|e| FsError::Backend(format!("read {}: {e}", path.display())))?;
            let short = chunk.len() < want;
            if out.is_empty() && (short || want == max) {
                return Ok(chunk); // the common case: the whole window in one command, no copy
            }
            out.extend_from_slice(&chunk);
            if short || out.len() >= max {
                return Ok(out);
            }
        }
    }

    /// Write-temp-then-`mv`, the same shape [`crate::tools::write_atomic`] uses locally, so a reader or
    /// a crash sees the old file or the new one and never a partial. The scripts keep the rest of the
    /// [`FsBackend::write_bytes`] contract too: the target is resolved through a symlink first
    /// (`resolve_target!`), and the temp is given the existing file's mode before content goes in
    /// (`new_temp!`). A bare `mv` would replace a link with a regular file and give the file the temp's
    /// default mode — 0600 became 0644, 0755 lost its execute bits. The content travels on stdin when
    /// the target has proven it delivers stdin ([`Capabilities::stdin`]) and as argv-sized base64
    /// chunks otherwise — never as one argv entry, which Linux caps at 128 KiB.
    ///
    /// Either way the temp's size is checked against `bytes.len()` before the `mv`, so a transport
    /// that silently dropped or cut short the content fails the write instead of committing it.
    async fn write_bytes(&self, path: &Path, bytes: Vec<u8>) -> Result<(), FsError> {
        self.write_slice(path, &bytes).await
    }

    async fn write_if_unchanged(
        &self,
        path: &Path,
        bytes: Vec<u8>,
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
        self.write_slice(path, &bytes).await?;
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
                    "--glob".to_string(),
                    format!("!{}", super::GIT_DIR),
                ];
                // The target's answer, not this host's — see `ShellFs::root_in_git_repo`.
                if !self.root_in_git_repo(&q.root).await {
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
                        if p == q.root || super::is_under_git_dir(&q.root, &p) {
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
            // `.git` is skipped by `rg --glob !.git` on pass 1, but not by `find` (pass 2, and the
            // POSIX rungs) — one host-side rule covers every rung identically.
            if p == q.root || super::is_under_git_dir(&q.root, &p) {
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
    /// Whether `root` — or any of its ancestors — holds a `.git`, asked **on the target** and cached.
    ///
    /// [`crate::tools::root_is_inside_git_repo`] answers the same question with host `stat`s, which is
    /// right for [`super::local::LocalFs`] and meaningless here: the replica's `/work` is not the
    /// sandbox's, so the host answer is a coin flip that decides whether `rg` gets `--no-require-git`
    /// and therefore whether `.gitignore` is honored at all. Getting it wrong is invisible — a search
    /// that quietly walks `node_modules/`, or one that quietly doesn't.
    ///
    /// Cached per root: the answer changes only if someone runs `git init` mid-session, and paying a
    /// round trip on every `grep` to catch that would be the wrong trade. A transport failure answers
    /// `false`, matching what the local version reports for a root it cannot stat.
    async fn root_in_git_repo(&self, root: &Path) -> bool {
        if let Ok(cache) = self.git_roots.lock()
            && let Some(hit) = cache.get(root)
        {
            return *hit;
        }
        let answer = self
            .exec(
                "sh",
                &sh_script(GIT_REPO_SCRIPT, &[root.to_string_lossy().into_owned()]),
            )
            .await
            .is_ok_and(|r| r.code == Some(0));
        if let Ok(mut cache) = self.git_roots.lock() {
            cache.insert(root.to_path_buf(), answer);
        }
        answer
    }

    /// One command, one place: every operation's timeout, error wrapping and argv discipline.
    async fn exec(&self, program: &str, args: &[String]) -> Result<ExecResult, FsError> {
        self.runner
            .run(program, args, None, self.timeout)
            .await
            .map_err(|e| FsError::Backend(format!("{program}: {e}")))
    }

    /// Assemble `bytes` into the temp named by `suffix` one argv-sized chunk per command, committing
    /// to `target` with the last. Returns the first failed command's result, or the committing one's.
    ///
    /// The fallback for a target without stdin: `WRITE_CHUNK` raw bytes per round trip, so a file
    /// costs one command per 48 KiB — the same single command as before for anything smaller.
    async fn write_chunks(
        &self,
        target: &str,
        suffix: &str,
        bytes: &[u8],
    ) -> Result<ExecResult, FsError> {
        let size = bytes.len().to_string();
        let mut chunks = bytes.chunks(WRITE_CHUNK).peekable();
        let mut first = true;
        loop {
            // An empty file is one empty chunk: the temp must still be created and committed.
            let chunk = chunks.next().unwrap_or_default();
            let last = chunks.peek().is_none();
            let args = sh_script(
                WRITE_CHUNK_SCRIPT,
                &[
                    target.to_string(),
                    suffix.to_string(),
                    encode_base64(chunk),
                    if first { "new" } else { "append" }.to_string(),
                    if last { "commit" } else { "" }.to_string(),
                    size.clone(),
                ],
            );
            let result = self.exec("sh", &args).await?;
            if result.code != Some(0) || last {
                return Ok(result);
            }
            first = false;
        }
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
    // `SystemTime + Duration` panics past what the platform can hold; an out-of-range mtime is unknown.
    let mtime = std::time::UNIX_EPOCH.checked_add(std::time::Duration::from_secs(secs));
    Some(Meta {
        kind,
        len,
        mtime,
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
            text: trim_eol(text).into_owned(),
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
//    the image. Every operation that moves file *content* through a string therefore base64s it —
//    pure ASCII, losslessly survivable. (A write whose target delivers stdin sends raw bytes there
//    instead; stdin is bytes, not a string, so the problem never arises.)
// 2. A byte *window* (`read_bytes(path, offset, max)`) has no argv-only spelling: `head -c` cannot
//    skip, `base64` cannot window; and a write must be assembled in a temp and renamed. These are the
//    module's only pipelines, and they use the sanctioned form — a **fixed** script with paths passed
//    as positional parameters, never interpolated.

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
stat -L -c '%s	%Y' -- "$1" 2>/dev/null || printf '0	0'
printf '
'"#;

/// Walk `$1` and its ancestors looking for a `.git`, exiting 0 the moment one is found — the
/// target-side twin of [`crate::tools::root_is_inside_git_repo`], step for step, so the two backends
/// make the same `--no-require-git` decision about the same tree.
///
/// Built from `dirname` and `[ -e ]` rather than `git rev-parse`: git is a *development* tool and a
/// sandbox that has none is ordinary, whereas a missing `.git` genuinely means "not a repository"
/// here. The `[ "$n" = "$d" ]` guard is the loop's only termination proof — `dirname /` is `/`, and
/// `dirname` of a bare relative name is `.`, both fixed points.
const GIT_REPO_SCRIPT: &str = r#"d="$1"
[ -d "$d" ] || d="$(dirname "$d")"
while [ -n "$d" ]; do
  [ -e "$d/.git" ] && exit 0
  n="$(dirname "$d")"
  [ "$n" = "$d" ] && break
  d="$n"
done
exit 1"#;

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

/// The opening of every write script: resolve the target `$1` to the file a write lands in, as `$t`,
/// and name the temp `$tmp` beside it with the host's unpredictable suffix `$2`. The shell spelling of
/// `LocalFs`'s `resolve_symlink_target`, so both backends write through a symlink rather than replacing
/// it: `readlink -f` resolves the whole chain, and for a dangling link, which `readlink -f` may refuse,
/// one `readlink` hop is joined to the link's directory. The temp sits beside the resolved file, so the
/// final `mv` is a rename within one filesystem.
///
/// A path starting with `-` is prefixed with `./` so no command reads it as an option; `--` would do
/// the same, but not every busybox applet takes it.
macro_rules! resolve_target {
    () => {
        r#"t="$1"
case "$t" in -*) t="./$t" ;; esac
if [ -L "$t" ]; then
  r=$(readlink -f "$t" 2>/dev/null) || r=
  if [ -z "$r" ]; then
    r=$(readlink "$t") || exit 1
    case "$r" in
      /*) ;;
      *) case "$t" in */*) r="${t%/*}/$r" ;; esac ;;
    esac
  fi
  t="$r"
  case "$t" in -*) t="./$t" ;; esac
fi
tmp="$t.tmp.$2"
"#
    };
}

/// Refuse a target that is a directory, with the message `std` gives (so both backends read alike).
/// Without it the final `mv -f "$tmp" "$t"` moves the temp *into* the directory and exits 0: a write
/// that reports success and leaves `DIR/DIR.tmp.SUFFIX` behind. `mv -T` would refuse atomically, but
/// it is GNU-only; busybox has no such flag. So the check runs twice, before the temp exists and
/// again immediately before the `mv`; a directory created at the target inside that last instant is
/// the one remaining window, the same width as any check-then-rename on a shell.
macro_rules! not_a_directory {
    () => {
        r#"if [ -d "$t" ]; then
  echo "Is a directory (os error 21)" >&2
  exit 1
fi
"#
    };
}

/// Create the empty temp `$tmp`, with `$t`'s mode when `$t` exists, before any content goes in, so
/// the replacement keeps the file's mode and a 0600 file's bytes are never in a world-readable temp.
/// `chmod --reference` is GNU; busybox, which Alpine ships, has no `--reference`, so its fallback is
/// the octal mode from `stat -L -c %a`. A mode that cannot be copied fails the write: committing would
/// change the file's permissions, which is exactly what this exists to prevent.
///
/// `set -C` is the shell's `create_new`: the temp is refused, not followed, if anything was preplanted
/// at its name.
macro_rules! new_temp {
    () => {
        concat!(
            not_a_directory!(),
            r#"set -C
: > "$tmp" || exit 1
set +C
if [ -e "$t" ]; then
  chmod --reference="$t" "$tmp" 2>/dev/null || chmod "$(stat -L -c %a "$t")" "$tmp" || {
    echo "cannot give the new file the mode of $t" >&2
    rm -f "$tmp"
    exit 1
  }
fi
"#
        )
    };
}

/// Write stdin into a temp, check it holds exactly `$3` bytes, and `mv` it over the target. `$1` is the
/// target and `$2` the temp's suffix (see [`resolve_target!`]). The temp is removed on any failure.
///
/// **The size check is what makes a lost stdin a failed write rather than an empty file.** The probe
/// already refuses this path to a transport that drops stdin, but a probe is one sample: an endpoint
/// behind a load balancer mid-rollout can pass it on one replica and answer the write from another.
/// `$n` is unquoted on purpose: some `wc` implementations pad the count with spaces.
const WRITE_STDIN_SCRIPT: &str = concat!(
    resolve_target!(),
    new_temp!(),
    r#"if cat >> "$tmp"; then
  n=$(wc -c < "$tmp")
  if [ $n -eq "$3" ]; then
    if [ -d "$t" ]; then echo "Is a directory (os error 21)" >&2; rm -f "$tmp"; exit 1; fi
    mv -f "$tmp" "$t" && exit 0
  else
    echo "short write: $n of $3 bytes arrived on stdin" >&2
  fi
fi
rm -f "$tmp"
exit 1"#
);

/// One argv-sized piece of a write, for a target without stdin: decode base64 `$3` onto the temp (`new`
/// in `$4` creates it first). `$1` is the target and `$2` the temp's suffix (see
/// [`resolve_target!`]). When `$5` is set this is the last piece: check the temp holds exactly `$6`
/// bytes and `mv` it into place. The temp is removed on any failure; the target is untouched until the
/// `mv`, so a reader never sees a partial file. POSIX `sh` and busybox alike — this is the path Alpine
/// targets without `-i` take.
const WRITE_CHUNK_SCRIPT: &str = concat!(
    resolve_target!(),
    r#"if [ "$4" = new ]; then
"#,
    new_temp!(),
    r#"fi
printf %s "$3" | base64 -d >> "$tmp" || { rm -f "$tmp"; exit 1; }
[ -z "$5" ] && exit 0
n=$(wc -c < "$tmp")
if [ $n -eq "$6" ]; then
  if [ -d "$t" ]; then echo "Is a directory (os error 21)" >&2; rm -f "$tmp"; exit 1; fi
  mv -f "$tmp" "$t" && exit 0
else
  echo "short write: $n of $6 bytes assembled" >&2
fi
rm -f "$tmp"
exit 1"#
);

/// The host's compensating action for a write that failed where the scripts could not clean up (a
/// transport error, a timeout between chunks): remove the temp, found the way the write named it.
const REMOVE_TEMP_SCRIPT: &str = concat!(resolve_target!(), r#"rm -f "$tmp""#);

/// What a search exit above 1 means, from stderr alone (the three causes share the exit code).
enum SearchFailure {
    /// The engine rejected the pattern.
    BadRegex,
    /// The search root itself does not exist; carries the engine's message.
    MissingRoot(String),
    /// Only per-path read errors: the hits on stdout are still valid. Carries the note.
    Skipped(String),
}

/// Per-path read errors name the path and end in the OS message (`rg` appends ` (os error N)`).
/// Anything else on stderr is the engine complaining about the pattern.
fn classify_search_failure(stderr: &str, root: &str) -> SearchFailure {
    const OS_ERRORS: [&str; 5] = [
        "No such file or directory",
        "Permission denied",
        "Is a directory",
        "Too many levels of symbolic links",
        "Input/output error",
    ];
    let lines: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    let path_error = |l: &str| {
        let l = l.trim_end();
        let l = l
            .strip_suffix(')')
            .and_then(|x| x.rsplit_once(" (os error ").map(|p| p.0))
            .unwrap_or(l);
        OS_ERRORS.iter().any(|e| l.ends_with(e))
    };
    if lines.is_empty() || !lines.iter().all(|l| path_error(l)) {
        return SearchFailure::BadRegex;
    }
    // The root's own error is `PROGRAM: ROOT: [IO error for operation on ROOT: ]No such file or
    // directory`; a path under the root carries more path after ROOT, so matching the exact
    // `: ROOT: ` form tells them apart.
    let root_prefix = format!(": {root}: ");
    if let Some(l) = lines
        .iter()
        .find(|l| l.contains(&root_prefix) && l.contains("No such file or directory"))
    {
        return SearchFailure::MissingRoot(l.trim().to_string());
    }
    let first = lines[0].trim();
    SearchFailure::Skipped(format!(
        "{} path(s) could not be searched; first: {first}",
        lines.len()
    ))
}

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
///
/// `in_git_repo` is the target's answer (see [`ShellFs::root_in_git_repo`]), not this host's — the
/// host's would be about a different tree.
fn rg_args(q: &SearchQuery, in_git_repo: bool) -> Vec<String> {
    let mut args = vec![
        "--null".to_string(),
        // Like `grep -H`: a single-file root otherwise prints no path, and the record parser needs one.
        "--with-filename".to_string(),
        "--line-number".to_string(),
        "--no-heading".to_string(),
        "--color=never".to_string(),
        "--hidden".to_string(),
        // `$` matches before `\r\n` too — the same `crlf(true)` `LocalFs` compiles with.
        "--crlf".to_string(),
    ];
    if !in_git_repo {
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
    // After the caller's glob on purpose: ripgrep gives the *last* matching glob precedence. Never
    // applied to the root itself, so an explicit search of `repo/.git` still searches it.
    args.push("--glob".to_string());
    args.push(format!("!{}", super::GIT_DIR));
    push_context(&mut args, q);
    // No `--` line between context groups. It is the one record with no NUL, so it cannot be told
    // from a path that starts with `--` + newline; without it the stream is only path records.
    args.push("--no-context-separator".to_string());
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
    // `-H`: name the file even when the root *is* a single file — without it `grep` prints bare
    // `line:text` records, which neither parser can attribute, so a file-path search came back empty.
    let mut args = vec!["-r".to_string(), "-n".to_string(), "-H".to_string()];
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
    // GNU `--exclude-dir` also skips a *command-line* directory whose name matches, so it is left off
    // when the root is itself a `.git` — the caller asked for that tree. The host-side
    // `is_under_git_dir` filter covers what this flag cannot (busybox, a worktree's `.git` file).
    if gnu && q.root.file_name().is_none_or(|n| n != super::GIT_DIR) {
        args.push(format!("--exclude-dir={}", super::GIT_DIR));
    }
    push_context(&mut args, q);
    // GNU grep only (busybox rejects it, and prints its `--` where `parse_colon_records` skips it): drop
    // the NUL-less `--` group line, which is indistinguishable from a path starting `--` + newline.
    if gnu {
        args.push("--no-group-separator".to_string());
    }
    args.push("-e".to_string());
    args.push(if q.literal {
        q.pattern.clone()
    } else {
        crlf_tolerant_ere(&q.pattern).into_owned()
    });
    args.push("--".to_string());
    args.push(q.root.to_string_lossy().into_owned());
    args
}

/// Make an ERE's end-of-line anchors tolerate a CRLF line ending — the POSIX rungs' stand-in for
/// `rg --crlf`, which GNU and busybox `grep` have no equivalent of.
///
/// Both split lines on `\n` alone, so on a CRLF file every line ends in `\r` and `foo$` matches
/// nothing. Each `$` that is an anchor at the end of a branch (end of pattern, or before `|` or `)`)
/// becomes `\r?$`, with a literal carriage-return byte, which both dialects accept. A `$` that is
/// escaped or inside a bracket expression is a literal and stays. Post-filtering cannot do this: the
/// lines `grep` failed to match never reach us.
fn crlf_tolerant_ere(pattern: &str) -> std::borrow::Cow<'_, str> {
    if !pattern.contains('$') {
        return std::borrow::Cow::Borrowed(pattern);
    }
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() + 4);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' => {
                out.push(c);
                if let Some(&next) = chars.get(i + 1) {
                    out.push(next);
                    i += 1;
                }
            }
            '[' => {
                // A bracket expression: `]` right after `[` or `[^` is literal, and `[:class:]`,
                // `[=x=]`, `[.x.]` nest. Copied verbatim through its closing `]`.
                let start = i;
                i += 1;
                if chars.get(i) == Some(&'^') {
                    i += 1;
                }
                if chars.get(i) == Some(&']') {
                    i += 1;
                }
                while i < chars.len() && chars[i] != ']' {
                    if chars[i] == '['
                        && let Some(&d @ (':' | '=' | '.')) = chars.get(i + 1)
                    {
                        i += 2;
                        while i + 1 < chars.len() && !(chars[i] == d && chars[i + 1] == ']') {
                            i += 1;
                        }
                        i += 1;
                    }
                    i += 1;
                }
                let end = i.min(chars.len() - 1);
                out.extend(&chars[start..=end]);
            }
            '$' if matches!(chars.get(i + 1), None | Some('|') | Some(')')) => out.push_str("\r?$"),
            _ => out.push(c),
        }
        i += 1;
    }
    std::borrow::Cow::Owned(out)
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
/// There is no group-separator line to skip: the engines are run with it turned off (see `rg_args` and
/// `posix_grep_args`), because `--` + newline is also a legal start of a path. Every record is
/// `PATH NUL ...`, so a path beginning with `--` + newline round-trips.
fn parse_records(stdout: &str) -> Vec<Hit> {
    let bytes = stdout.as_bytes();
    let mut hits = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
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
        // Clipped by `search` once parsed (`clip_hits`), where the query is in hand to find the match.
        text: trim_eol(text).into_owned(),
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
    super::query(pattern, root, limit)
}

/// Entry points for the fuzz targets (`crates/agent/fuzz`); not part of the API. The parsers stay
/// private to this module, as the fuzz crate lives outside the workspace and can reach only `pub`
/// items. Same shape as `execd::__fuzz`: thin wrappers, no logic of their own.
#[doc(hidden)]
pub mod __fuzz {
    use std::path::PathBuf;

    use crate::fs::{DirEntry, Hit, Meta};

    /// `rg --null` / `grep -Z` records.
    pub fn records(stdout: &str) -> Vec<Hit> {
        super::parse_records(stdout)
    }

    /// busybox `grep -rn` records (no glob).
    pub fn colon_records(stdout: &str) -> Vec<Hit> {
        super::parse_colon_records(stdout, &super::query("x", PathBuf::from("."), 1))
    }

    /// The one-shot stat line.
    pub fn stat(stdout: &str) -> Option<Meta> {
        super::parse_stat(stdout)
    }

    /// `find -printf '%y\t%p\0'`.
    pub fn typed_paths(stdout: &str) -> Vec<(PathBuf, bool)> {
        super::parse_typed_paths(stdout)
    }

    /// `<type>\t<size>\t<name>\0` directory listing.
    pub fn dir_entries(stdout: &str, cap: usize, include_hidden: bool) -> Vec<DirEntry> {
        super::parse_dir_entries(stdout, cap, include_hidden)
    }

    pub fn encode_base64(bytes: &[u8]) -> String {
        super::encode_base64(bytes)
    }

    pub fn decode_base64(s: &str) -> Result<Vec<u8>, String> {
        super::decode_base64(s)
    }
}

/// The scripts printed in the appendix of `sdk/conformance/FILES-OVER-EXEC.md` are the ones this file
/// sends, byte for byte. Other clients test their scripts against the same appendix.
#[cfg(test)]
mod appendix_tests {
    const SPEC: &str = include_str!("../../../../sdk/conformance/FILES-OVER-EXEC.md");

    /// The `sh` block under the heading that starts with `### {letter}. `.
    fn appendix(letter: char) -> String {
        let head = format!("### {letter}. ");
        let rest = &SPEC[SPEC
            .find(&head)
            .unwrap_or_else(|| panic!("no appendix {letter}"))..];
        let rest = &rest[rest.find("```sh\n").unwrap() + 6..];
        rest[..rest.find("\n```").unwrap()].to_string()
    }

    /// Scripts known to differ from the appendix until the pending fixes land; each names its fix.
    /// An entry here that now matches fails the test, so it cannot outlive its fix.
    const KNOWN_DIFFERENT: &[(char, &str)] = &[
        // The code is ahead: `stat -L` (#564) and the directory refusal in C/D. The appendix is
        // brought level by #574, and the dangling-link resolve and stdout-`c` probe it adds to
        // C/D/E by B1/B2; then this list is empty.
        ('A', "appendix lacks `stat -L`, #574"),
        ('C', "appendix lacks the directory refusal, #574, B1/B2"),
        ('D', "appendix lacks the directory refusal, #574, B1/B2"),
    ];

    #[test]
    fn the_appendix_scripts_are_the_scripts_sent() {
        let scripts: [(char, &str); 6] = [
            ('A', super::STAT_SCRIPT),
            ('B', super::READ_WINDOW_SCRIPT),
            ('C', super::WRITE_STDIN_SCRIPT),
            ('D', super::WRITE_CHUNK_SCRIPT),
            ('E', super::REMOVE_TEMP_SCRIPT),
            ('F', super::GIT_REPO_SCRIPT),
        ];
        let mut wrong = Vec::new();
        for (letter, code) in scripts {
            let same = appendix(letter) == code.trim_end_matches('\n');
            let known = KNOWN_DIFFERENT.iter().find(|(l, _)| *l == letter);
            match (same, known) {
                (false, None) => wrong.push(format!("{letter} differs from the appendix")),
                (true, Some((_, why))) => wrong.push(format!(
                    "{letter} now matches ({why}); remove it from KNOWN_DIFFERENT"
                )),
                _ => {}
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_mtime_past_systemtime_range_is_unknown_not_a_panic() {
        let m = parse_stat("w\tf\t1\t18446744073709551615\n").expect("a well-formed line parses");
        assert_eq!(m.mtime, None);
        assert_eq!(m.len, 1);
    }

    #[test]
    fn stat_script_follows_a_symlink_for_size_and_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::write(d.join("t"), "hello world\n").unwrap();
        std::os::unix::fs::symlink("t", d.join("a-rather-long-link-name")).unwrap();
        std::os::unix::fs::symlink("gone", d.join("dangling")).unwrap();
        let run = |name: &str| {
            std::process::Command::new("sh")
                .args(["-c", STAT_SCRIPT, "sh"])
                .arg(d.join(name))
                .output()
                .unwrap()
        };
        let out = run("a-rather-long-link-name");
        let line = String::from_utf8(out.stdout).unwrap();
        let f: Vec<&str> = line.trim_end().split('\t').collect();
        assert_eq!((f[1], f[2]), ("f", "12"), "{line:?}");
        assert_eq!(run("dangling").status.code(), Some(1));
    }

    #[test]
    fn search_exit_two_is_told_apart_by_stderr() {
        let root = "/work/proj";
        for bad in [
            "regex parse error:\n    (\n    ^\nerror: unclosed group",
            "grep: Unmatched ( or \\(",
            "grep: bad regex '(': Unmatched ( or \\(",
            "",
        ] {
            assert!(
                matches!(classify_search_failure(bad, root), SearchFailure::BadRegex),
                "{bad:?}"
            );
        }
        for missing in [
            "rg: /work/proj: No such file or directory (os error 2)",
            "rg: /work/proj: IO error for operation on /work/proj: No such file or directory (os error 2)",
            "grep: /work/proj: No such file or directory",
        ] {
            assert!(
                matches!(
                    classify_search_failure(missing, root),
                    SearchFailure::MissingRoot(_)
                ),
                "{missing:?}"
            );
        }
        let SearchFailure::Skipped(note) = classify_search_failure(
            "rg: /work/proj/a: Permission denied (os error 13)\nrg: /work/proj/b: Permission denied (os error 13)\n",
            root,
        ) else {
            panic!("per-file errors must keep the matches");
        };
        assert!(note.starts_with("2 path(s)"), "{note}");
        assert!(matches!(
            classify_search_failure("grep: /work/proj/x: No such file or directory", root),
            SearchFailure::Skipped(_)
        ));
    }

    /// This host's `sh`, with a prelude run before every script. A `PATH` shim, not a shell function,
    /// because the read script `exec`s `dd`, which skips functions.
    struct Prelude(String);

    #[async_trait]
    impl CommandRunner for Prelude {
        async fn run(
            &self,
            program: &str,
            args: &[String],
            cwd: Option<&str>,
            timeout: Duration,
        ) -> std::io::Result<ExecResult> {
            let mut args = args.to_vec();
            if let ("sh", Some(script)) = (program, args.get_mut(1)) {
                *script = format!("{}\n{script}", self.0);
            }
            crate::testing::Host.run(program, &args, cwd, timeout).await
        }

        async fn run_with_stdin(
            &self,
            program: &str,
            args: &[String],
            cwd: Option<&str>,
            timeout: Duration,
            stdin: &[u8],
        ) -> std::io::Result<ExecResult> {
            crate::testing::Host
                .run_with_stdin(program, args, cwd, timeout, stdin)
                .await
        }
    }

    /// A directory whose `dd`/`base64` reject the flags, as a busybox without them does, and the
    /// prelude that puts it first on `PATH`.
    fn shim(dir: &Path, tool: &str) -> Prelude {
        use std::os::unix::fs::PermissionsExt;
        let bin = dir.join("shim");
        std::fs::create_dir_all(&bin).unwrap();
        let f = bin.join(tool);
        std::fs::write(
            &f,
            format!("#!/bin/sh\necho '{tool}: unrecognized option' >&2\nexit 1\n"),
        )
        .unwrap();
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o755)).unwrap();
        Prelude(format!("PATH={}:$PATH", bin.display()))
    }

    #[tokio::test]
    async fn a_target_whose_dd_lacks_the_flags_never_reads_an_existing_file_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("precious.txt");
        std::fs::write(&path, "keep me\n").unwrap();
        let fs = ShellFs::connect(Arc::new(shim(dir.path(), "dd")))
            .await
            .unwrap();
        assert!(fs.capabilities().no_read_window);
        let err = fs.read_bytes(&path, 0, 1024).await.unwrap_err();
        assert!(err.to_string().contains("cannot be read"), "{err}");
        // A target with the flags reads it, and the probe agrees.
        let fs = ShellFs::connect(Arc::new(Prelude(":".into())))
            .await
            .unwrap();
        assert!(!fs.capabilities().no_read_window);
        assert_eq!(fs.read_bytes(&path, 0, 1024).await.unwrap(), b"keep me\n");
    }

    #[tokio::test]
    async fn a_target_whose_base64_lacks_dash_w_is_probed_as_unable_to_read() {
        let dir = tempfile::tempdir().unwrap();
        let caps = Capabilities::probe(&shim(dir.path(), "base64"))
            .await
            .unwrap();
        assert!(caps.no_read_window);
    }

    /// A runner whose every request fails, as a wrong bearer's does.
    struct Refused;

    #[async_trait::async_trait]
    impl CommandRunner for Refused {
        async fn run(
            &self,
            _: &str,
            _: &[String],
            _: Option<&str>,
            _: Duration,
        ) -> std::io::Result<ExecResult> {
            Err(std::io::Error::other(
                "exec endpoint http://x returned 401: unauthorized",
            ))
        }
    }

    /// A runner whose every answer exceeds the response cap.
    struct TooBig;

    #[async_trait::async_trait]
    impl CommandRunner for TooBig {
        async fn run(
            &self,
            _: &str,
            _: &[String],
            _: Option<&str>,
            _: Duration,
        ) -> std::io::Result<ExecResult> {
            Err(std::io::Error::new(
                std::io::ErrorKind::FileTooLarge,
                "the response exceeded the 9-byte cap",
            ))
        }
    }

    #[tokio::test]
    async fn a_probe_that_ran_but_answered_too_much_reads_as_absent() {
        let fs = ShellFs::connect(Arc::new(TooBig)).await.unwrap();
        assert!(!fs.capabilities().rg && !fs.capabilities().stdin);
    }

    #[tokio::test]
    async fn a_probe_whose_request_failed_fails_connect_instead_of_reading_as_absent() {
        let err = ShellFs::connect(Arc::new(Refused)).await.err().unwrap();
        assert!(err.to_string().contains("returned 401"), "got: {err}");
    }

    #[tokio::test]
    async fn a_write_onto_a_directory_fails_and_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().join("d");
        std::fs::create_dir(&d).unwrap();
        let probed = ShellFs::connect(Arc::new(Prelude(":".into())))
            .await
            .unwrap();
        for stdin in [true, false] {
            let mut caps = probed.capabilities();
            caps.stdin = stdin;
            let fs = ShellFs::with_capabilities(Arc::new(Prelude(":".into())), caps);
            let err = fs.write_bytes(&d, b"x".to_vec()).await.unwrap_err();
            assert!(err.to_string().contains("Is a directory"), "{err}");
            let err = fs
                .write_if_unchanged(&d, b"x".to_vec(), fs.stat(&d).await.unwrap().unwrap().mtime)
                .await
                .unwrap_err();
            assert!(err.to_string().contains("Is a directory"), "{err}");
            assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "stdin={stdin}");
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }

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
    fn a_path_starting_with_dashes_and_a_newline_is_not_a_group_separator() {
        // The engines are run without a group separator, so `--` + newline can only be a path.
        let hits = parse_records("--\nb\u{0}2-y\n--\r\nc\u{0}3:z\r\n");
        let got: Vec<_> = hits
            .iter()
            .map(|h| (h.path.to_string_lossy().into_owned(), h.line))
            .collect();
        assert_eq!(got, [("--\nb".into(), 2), ("--\r\nc".into(), 3)]);
    }

    #[test]
    fn engines_are_told_not_to_print_group_separators() {
        let q = query("x", PathBuf::from("."), 10);
        assert!(
            rg_args(&q, true)
                .iter()
                .any(|a| a == "--no-context-separator")
        );
        assert!(
            posix_grep_args(&q, true)
                .iter()
                .any(|a| a == "--no-group-separator")
        );
        assert!(
            !posix_grep_args(&q, false)
                .iter()
                .any(|a| a == "--no-group-separator")
        );
    }

    #[test]
    fn files_named_like_a_separator_are_not_skipped() {
        let hits = parse_records("--\u{0}1:x\n--\r\u{0}2:y\n--\u{0}3:--\n");
        let got: Vec<_> = hits
            .iter()
            .map(|h| (h.path.to_string_lossy().into_owned(), h.line))
            .collect();
        assert_eq!(
            got,
            [("--".into(), 1), ("--\r".into(), 2), ("--".into(), 3)]
        );
        assert_eq!(hits[2].text, "--");
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
        let args = rg_args(&q, false);
        let e = args.iter().position(|a| a == "-e").unwrap();
        assert_eq!(args[e + 1], "-v");
        let dd = args.iter().position(|a| a == "--").unwrap();
        assert_eq!(args[dd + 1], "-weird-dir");
    }

    #[test]
    fn a_negated_glob_becomes_a_bang_prefixed_rg_glob() {
        let mut q = query("x", PathBuf::from("."), 100);
        q.glob = Some(("*.test.rs".into(), true));
        let args = rg_args(&q, false);
        let g = args.iter().position(|a| a == "--glob").unwrap();
        assert_eq!(args[g + 1], "!*.test.rs");
    }

    #[test]
    fn rg_skips_git_metadata_with_a_glob_that_outranks_the_callers() {
        // Ripgrep gives the *last* matching glob precedence, so `!.git` must follow a caller's own.
        let mut q = query("x", PathBuf::from("."), 100);
        q.glob = Some(("*".into(), false));
        let args = rg_args(&q, false);
        let globs: Vec<_> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "--glob")
            .map(|(i, _)| args[i + 1].as_str())
            .collect();
        assert_eq!(globs, ["*", "!.git"]);
    }

    #[test]
    fn gnu_grep_excludes_git_unless_the_root_is_git_itself() {
        let q = query("x", PathBuf::from("/repo"), 100);
        assert!(
            posix_grep_args(&q, true)
                .iter()
                .any(|a| a == "--exclude-dir=.git")
        );
        // GNU `--exclude-dir` would skip the command-line directory itself.
        let q = query("x", PathBuf::from("/repo/.git"), 100);
        assert!(
            !posix_grep_args(&q, true)
                .iter()
                .any(|a| a.starts_with("--exclude-dir"))
        );
        // Busybox has no such flag; the host-side filter covers it.
        let q = query("x", PathBuf::from("/repo"), 100);
        assert!(
            !posix_grep_args(&q, false)
                .iter()
                .any(|a| a.starts_with("--exclude-dir"))
        );
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
    fn end_anchors_tolerate_a_crlf_line_ending_for_posix_grep() {
        assert_eq!(crlf_tolerant_ere("end$"), "end\r?$");
        assert_eq!(crlf_tolerant_ere("(one$|two$)"), "(one\r?$|two\r?$)");
        assert_eq!(crlf_tolerant_ere("a$|b"), "a\r?$|b");
        // Literals stay literal: escaped, or inside a bracket expression (incl. `]` first, classes).
        assert_eq!(crlf_tolerant_ere("cost\\$"), "cost\\$");
        assert_eq!(crlf_tolerant_ere("[$]"), "[$]");
        assert_eq!(crlf_tolerant_ere("[]$]x$"), "[]$]x\r?$");
        assert_eq!(crlf_tolerant_ere("[[:space:]$]$"), "[[:space:]$]\r?$");
        // No anchor at all borrows.
        assert!(matches!(
            crlf_tolerant_ere("plain"),
            std::borrow::Cow::Borrowed(_)
        ));
        // A literal search is not rewritten, and rg gets `--crlf` instead.
        let mut q = query("end$", PathBuf::from("."), 10);
        assert!(posix_grep_args(&q, true).iter().any(|a| a == "end\r?$"));
        q.literal = true;
        assert!(posix_grep_args(&q, true).iter().any(|a| a == "end$"));
        assert!(rg_args(&q, false).iter().any(|a| a == "--crlf"));
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
