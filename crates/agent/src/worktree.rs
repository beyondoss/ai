//! Per-subagent `git worktree` isolation.
//!
//! A child declaring `isolation: worktree` (see [`crate::agents::Isolation`]) runs with its filesystem
//! tools rooted at a private checkout, so two children fanned out in parallel can both edit `src/lib.rs`
//! without racing. This is **conflict avoidance, not containment** — nothing stops a child from naming
//! an absolute path outside its worktree, exactly as under Claude Code's `isolation: "worktree"` and pi's
//! subagent `cwd`. It is what makes parallel *writers* possible at all: `bash` reports no `write_target`
//! (`tools::bash`), so a shared `WriteLockRegistry` provably cannot serialize two children's shell
//! commands, and no amount of locking substitutes for giving them separate trees.
//!
//! # Why a baseline commit
//!
//! `git worktree add --detach <path> HEAD` checks out **HEAD**, not the developer's working tree. A child
//! asked to "fix the bug in what I just wrote" would see none of it. So [`Worktree::create`] seeds the
//! new checkout with the parent's uncommitted state (tracked modifications *and* untracked-but-not-ignored
//! files) and then commits it as a throwaway baseline on the worktree's detached HEAD.
//!
//! That baseline is what makes merge-back correct, not just convenient: [`Worktree::child_delta`] diffs
//! against it, so the patch contains **only what the child changed**. Diffing against the original HEAD
//! would replay the parent's own uncommitted work back onto the parent — applying it twice.
//!
//! The baseline commit is unreachable from any ref and lives only in the shared object database, so it
//! costs no branch, no tag, and is collected by `git gc` once the worktree is gone.
//!
//! # Cleanup
//!
//! `Drop` is best-effort: a `process::exit` does not run destructors (this bit us in PR #13, where
//! detached threads never got to finish), so a killed agent leaks its worktrees. The reliable path is
//! [`sweep`], run at startup, which reaps any worktree whose owning process is gone. Worktrees are keyed
//! by the creating PID precisely so a sweep can tell "orphaned by a crash" from "in use by the other
//! agent this developer is running right now" — Jared routinely runs several concurrently against one
//! repo, and a blind `rm -rf` of the base directory would delete a live session's work.
//!
//! PID liveness is checked via `/proc`, and a *reused* PID makes the sweep skip a directory it could
//! have reaped. That is the safe direction to be wrong in: a leak, never a deletion.
//!
//! # Where `git` runs
//!
//! Every step above is a `git` invocation, and each one has to happen **where the child's files
//! actually are**. On a laptop that is this machine. In service mode it is the session's sandbox, and
//! running them here instead would either fail confusingly or — far worse — succeed against the
//! replica's own checkout and merge one tenant's patch into it. That is why worktree isolation was
//! refused outright whenever the filesystem was remote, which in turn meant a write-capable subagent
//! could never run in `parallel` on a replica: [`crate::tools::subagent`] requires worktree isolation
//! for parallel writers, because `bash` reports no write target and no lock can serialize two
//! children's shell commands.
//!
//! [`Git`] carries that choice. `Git::Local` is `tokio::process` and `std::fs`; `Git::Remote` runs
//! the same sequence through the session's exec endpoint and filesystem backend. Three things differ
//! on the remote side, each for a reason worth knowing:
//!
//! - **Patch output goes through base64.** `ExecResult::stdout` is a `String`, and while `--binary`
//!   renders binary files as ASCII, an ordinary text hunk carries the file's own bytes — which for a
//!   non-UTF-8 source file are not valid UTF-8. A lossy conversion there yields a patch that fails to
//!   apply, or applies corruption.
//! - **The owner key is not a PID.** This process is not in the sandbox's `/proc`; a PID looked up
//!   there names something unrelated, or with a recycled number something live. A sandbox belongs to
//!   exactly one session, so the key only has to differ between incarnations.
//! - **`Drop` cannot clean up.** Every removal step is a round trip and `Drop` cannot await one. A
//!   remote worktree is instead reaped by the next incarnation, which sweeps its base directory
//!   before creating its own — so the cost falls only on sessions that use isolation at all.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::process::Command;

/// The candidate roots under which subagent worktrees may live, most-preferred first.
///
/// Preference is a persistent, on-disk cache directory — **not** `std::env::temp_dir()` as the first
/// choice. On a typical Linux box `/tmp` is `tmpfs` (RAM), and a worktree is a full checkout of the
/// repository (~43 MB for this one); an eight-way parallel fan-out would materialize eight of them, so a
/// fan-out over a large repo would quietly eat hundreds of megabytes of RAM and then fail with `ENOSPC`
/// when the tmpfs filled. Measured, not theorized: the first draft put them in `/tmp` and did exactly
/// that.
///
/// `temp_dir` is kept only as a **last-resort fallback** for when the cache directory can't be created
/// (an unwritable or nonexistent `$HOME`, a locked-down container) — a broken cache dir must degrade
/// subagents to a working-but-RAM-backed location, not disable them outright. [`ensure_base_dir`] picks
/// the first candidate it can actually create under.
///
/// Every candidate is outside the repository, which keeps a worktree from being walked by the parent's
/// own `find`/`grep` and avoids colliding with `.claude/worktrees/` (Claude Code's own agent isolation,
/// which [`sweep`] must never touch).
fn cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        roots.push(PathBuf::from(xdg).join("beyond-agent/worktrees"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".cache/beyond-agent/worktrees"));
    }
    roots.push(std::env::temp_dir().join("beyond-agent/worktrees"));
    roots
}

/// Worktrees are grouped per repository so [`sweep`] only ever reasons about — and only ever deletes —
/// checkouts belonging to the repo it was handed. A flat global directory would let a sweep run from one
/// repo delete an orphan belonging to another, leaving that repo's `git worktree` metadata dangling.
///
/// The basename keeps the path legible; the hash disambiguates two checkouts of the same-named repo. A
/// hand-rolled FNV-1a rather than `DefaultHasher`, whose output is explicitly not stable across releases —
/// a directory name must survive a toolchain upgrade or every worktree becomes an unreapable orphan.
fn repo_id(repo_root: &Path) -> String {
    let canonical = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical.as_os_str().as_encoded_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let name = canonical
        .file_name()
        .map(|n| sanitize(&n.to_string_lossy()))
        .unwrap_or_else(|| "repo".to_string());
    format!("{name}-{hash:016x}")
}

/// This repository's worktree base under a given cache root.
fn base_dir_under(root: &Path, repo_root: &Path) -> PathBuf {
    root.join(repo_id(repo_root))
}

/// Every base directory this repo's worktrees could be under, across all candidate roots — for [`sweep`]
/// to reap orphans regardless of which root created them (e.g. a prior run under a since-fixed `$HOME`).
fn base_dirs(repo_root: &Path) -> Vec<PathBuf> {
    cache_roots()
        .iter()
        .map(|r| base_dir_under(r, repo_root))
        .collect()
}

/// Create (if needed) and return the base directory to put new worktrees in: the first candidate root
/// whose base can actually be created. Falls through to the last candidate (`temp_dir`) so a caller
/// always gets *a* path — if even that fails, the error surfaces at the `git worktree add` that follows,
/// with a clear message, rather than here.
async fn ensure_base_dir(git: &Git, repo_root: &Path) -> PathBuf {
    if git.is_remote() {
        // One fixed root in the sandbox. The host's `$XDG_CACHE_HOME`/`$HOME` describe the replica,
        // not the tenant's world, and a worktree placed inside the workspace would show up in the
        // tenant's own repository. `/tmp` is writable in every sandbox image this runs against.
        let base = Path::new("/tmp/beyond-agent/worktrees").join(repo_id(repo_root));
        let _ = git.create_dir_all(&base).await;
        return base;
    }
    let mut last = None;
    for root in cache_roots() {
        let base = base_dir_under(&root, repo_root);
        if std::fs::create_dir_all(&base).is_ok() {
            return base;
        }
        last = Some(base);
    }
    last.unwrap_or_else(|| {
        std::env::temp_dir()
            .join("beyond-agent/worktrees")
            .join(repo_id(repo_root))
    })
}

/// How long a single `git` invocation gets when it runs in a sandbox. Generous — a worktree add on a
/// cold checkout is real work — but finite, so an exec endpoint that stops answering fails the task
/// rather than hanging the parent's fan-out.
const REMOTE_GIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Where `git` runs for a worktree.
///
/// Worktree isolation is a sequence of `git` invocations against a checkout — `worktree add`, a diff,
/// an apply. Every one of them must happen **wherever the child's files actually are**. On a laptop
/// that is this machine. In service mode it is the session's sandbox, and running them here instead
/// would either fail confusingly or, far worse, succeed against the replica's own checkout and merge
/// one tenant's patch into it.
///
/// So this is not an abstraction for its own sake: it is the difference between `isolation: worktree`
/// working in service mode and being refused there, which is what it was until now.
#[derive(Clone)]
pub enum Git {
    /// This machine: `tokio::process::Command` and `std::fs`.
    Local,
    /// A sandbox, reached through the session's exec endpoint.
    Remote {
        runner: std::sync::Arc<dyn crate::tools::exec::CommandRunner>,
        /// The same sandbox's filesystem. Seeding copies the parent's untracked files into the new
        /// checkout, and those bytes have to move within the sandbox, not through this process.
        backend: std::sync::Arc<dyn crate::tools::fs::FsBackend>,
        /// Identifies the owner of the worktrees this runner creates, in place of a PID.
        ///
        /// A PID is meaningless across the boundary: the process that created the worktree runs on a
        /// replica, not in the sandbox, and `/proc/<pid>` in the sandbox would answer about an
        /// unrelated process — or, with a recycled number, about a live one, which is the direction
        /// that deletes work. A sandbox belongs to exactly one session, so "this session's current
        /// incarnation" is both available and sufficient: anything carrying a different owner is an
        /// orphan by construction.
        owner: String,
    },
}

impl std::fmt::Debug for Git {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => f.write_str("Git::Local"),
            Self::Remote { owner, .. } => write!(f, "Git::Remote({owner})"),
        }
    }
}

impl Git {
    /// The prefix every worktree directory this runner creates is named with, and the key
    /// [`sweep`] reaps by.
    fn owner(&self) -> String {
        match self {
            Self::Local => std::process::id().to_string(),
            Self::Remote { owner, .. } => owner.clone(),
        }
    }

    fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }

    /// Build the runner for a session whose filesystem is in a sandbox.
    pub fn remote(
        runner: std::sync::Arc<dyn crate::tools::exec::CommandRunner>,
        backend: std::sync::Arc<dyn crate::tools::fs::FsBackend>,
    ) -> Self {
        Self::Remote {
            runner,
            backend,
            owner: remote_owner(),
        }
    }

    async fn create_dir_all(&self, path: &Path) -> Result<(), String> {
        match self {
            Self::Local => {
                std::fs::create_dir_all(path).map_err(|e| format!("mkdir {}: {e}", path.display()))
            }
            Self::Remote { backend, .. } => backend
                .create_dir_all(path)
                .await
                .map_err(|e| format!("mkdir {}: {e}", path.display())),
        }
    }

    /// Is `path` a regular file? Used to skip a symlink or a path that vanished between `ls-files`
    /// and the copy — a spawn must not fail over one.
    async fn is_file(&self, path: &Path) -> bool {
        match self {
            Self::Local => path.is_file(),
            Self::Remote { backend, .. } => matches!(
                backend.stat(path).await,
                Ok(Some(m)) if m.kind == crate::tools::fs::FileKind::File
            ),
        }
    }

    /// Like [`run_stdin`](Self::run_stdin), but hands back the outcome instead of turning a non-zero
    /// exit into an error.
    ///
    /// `git apply --3way` exits non-zero for **both** "applied, with conflict markers" and "could not
    /// apply at all", and the two are told apart by its stderr. A helper that collapsed the exit into
    /// `Err` would make a conflicted merge — the case the whole `ApplyOutcome::Conflicted` path
    /// exists for — indistinguishable from a failed one.
    async fn run_stdin_full(
        &self,
        dir: &Path,
        args: &[&str],
        stdin: &[u8],
    ) -> Result<(bool, Vec<u8>, String), String> {
        match self {
            Self::Local => {
                let out = command_output_with_stdin(dir, args, stdin).await?;
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                Ok((out.status.success(), out.stdout, stderr))
            }
            Self::Remote { runner, .. } => {
                let owned: Vec<String> = args.iter().map(|a| (*a).to_string()).collect();
                let res = runner
                    .run_with_stdin(
                        "git",
                        &owned,
                        Some(&dir.display().to_string()),
                        REMOTE_GIT_TIMEOUT,
                        stdin,
                    )
                    .await
                    .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
                Ok((res.code == Some(0), res.stdout.into_bytes(), res.stderr))
            }
        }
    }

    /// Run a shell command in the sandbox. Only for the two things git cannot express: removing a
    /// directory tree, and listing one.
    async fn sh(&self, dir: &Path, script: &str) -> Result<String, String> {
        match self {
            Self::Local => Err("sh is only used on the remote path".to_string()),
            Self::Remote { runner, .. } => {
                let owned = vec!["-c".to_string(), script.to_string()];
                let res = runner
                    .run(
                        "sh",
                        &owned,
                        Some(&dir.display().to_string()),
                        REMOTE_GIT_TIMEOUT,
                    )
                    .await
                    .map_err(|e| format!("sh: {e}"))?;
                Ok(res.stdout)
            }
        }
    }

    /// Remove a worktree, tolerating every "already gone" shape.
    ///
    /// Errors are ignored throughout: the directory may already be gone, or never have been fully
    /// created, and a cleanup that fails the task it was cleaning up after is worse than a leak the
    /// sweep will collect.
    async fn remove_worktree(&self, repo_root: &Path, path: &Path) {
        match self {
            Self::Local => remove_worktree_blocking(repo_root, path),
            Self::Remote { .. } => {
                let p = path.display().to_string();
                // `--force` because the child almost certainly left the checkout dirty, which
                // `git worktree remove` otherwise refuses.
                let _ = self
                    .run_text(repo_root, &["worktree", "remove", "--force", &p])
                    .await;
                // `git worktree remove` deletes the directory on success; if it refused (metadata
                // already pruned, say), the checkout can still be sitting there.
                let _ = self
                    .sh(repo_root, &format!("rm -rf {}", shell_quote(&p)))
                    .await;
                let _ = self.run_text(repo_root, &["worktree", "prune"]).await;
            }
        }
    }

    /// Reap sandbox worktrees left by a previous incarnation, just before creating a new one.
    ///
    /// The local path sweeps once at startup, keyed on PID liveness. A sandbox has neither: this
    /// process is not in its `/proc`, and there is no startup hook inside the sandbox to hang a sweep
    /// on. But it has something better — **a sandbox belongs to exactly one session**, so any
    /// worktree under our own base directory carrying a different owner is, by construction, from an
    /// incarnation that is gone. Sweeping here rather than at session start also means the cost is
    /// paid only by a session that actually uses worktree isolation.
    async fn sweep_siblings(&self, repo_root: &Path, base: &Path) {
        if !self.is_remote() {
            return;
        }
        let mine = self.owner();
        // `-1` one per line; failure (no such directory) is the ordinary first-run case.
        let Ok(listing) = self
            .sh(
                repo_root,
                &format!(
                    "ls -1 {} 2>/dev/null",
                    shell_quote(&base.display().to_string())
                ),
            )
            .await
        else {
            return;
        };
        for leaf in listing.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let Some((owner, _)) = leaf.split_once('-') else {
                // Not a name we wrote — leave it alone.
                continue;
            };
            if owner == mine {
                continue;
            }
            tracing::debug!(
                worktree = %base.join(leaf).display(),
                owner,
                "reaping a sandbox worktree left by a previous incarnation"
            );
            self.remove_worktree(repo_root, &base.join(leaf)).await;
        }
    }

    /// Copy one file. Within one filesystem in both modes — for a sandbox the bytes go out and back
    /// through the backend rather than through this process's own disk.
    async fn copy_file(&self, src: &Path, dst: &Path) -> Result<(), String> {
        match self {
            Self::Local => std::fs::copy(src, dst)
                .map(|_| ())
                .map_err(|e| format!("copy {}: {e}", src.display())),
            Self::Remote { backend, .. } => {
                let bytes = backend
                    .read_bytes(src, 0, usize::MAX)
                    .await
                    .map_err(|e| format!("read {}: {e}", src.display()))?;
                backend
                    .write_bytes(dst, &bytes)
                    .await
                    .map_err(|e| format!("write {}: {e}", dst.display()))
            }
        }
    }

    /// Run `git -C <dir> <args>`, returning stdout as text.
    ///
    /// For everything whose output is a path, a ref, or nothing at all. Patch output goes through
    /// [`run_bytes`](Self::run_bytes) instead — see there for why.
    async fn run_text(&self, dir: &Path, args: &[&str]) -> Result<String, String> {
        match self {
            Self::Local => {
                let out = local_git(dir, args).await?;
                Ok(String::from_utf8_lossy(&out).into_owned())
            }
            Self::Remote { runner, .. } => {
                let owned: Vec<String> = args.iter().map(|a| (*a).to_string()).collect();
                let res = runner
                    .run(
                        "git",
                        &owned,
                        Some(&dir.display().to_string()),
                        REMOTE_GIT_TIMEOUT,
                    )
                    .await
                    .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
                if res.code != Some(0) {
                    return Err(format!(
                        "git {} failed: {}",
                        args.join(" "),
                        res.stderr.trim()
                    ));
                }
                Ok(res.stdout)
            }
        }
    }

    /// Run `git -C <dir> <args>`, returning stdout as **bytes**.
    ///
    /// Only `git diff` needs this, and it needs it badly. `ExecResult::stdout` is a `String`, so a
    /// remote run would put the patch through a lossy UTF-8 conversion — and while `--binary` renders
    /// *binary* files as ASCII base85, an ordinary text hunk carries the file's own bytes, which for a
    /// Latin-1 or otherwise non-UTF-8 source file are not valid UTF-8. Lossy conversion there replaces
    /// them with U+FFFD, and the patch that comes back either fails to apply or applies corruption.
    /// So the remote path pipes the patch through `base64` in the sandbox and decodes it here.
    async fn run_bytes(&self, dir: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
        match self {
            Self::Local => local_git(dir, args).await,
            Self::Remote { runner, .. } => {
                let script = format!(
                    "git {} | base64 | tr -d '\n'",
                    args.iter()
                        .map(|a| shell_quote(a))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                let owned = vec!["-c".to_string(), script];
                let res = runner
                    .run(
                        "sh",
                        &owned,
                        Some(&dir.display().to_string()),
                        REMOTE_GIT_TIMEOUT,
                    )
                    .await
                    .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
                if res.code != Some(0) {
                    return Err(format!(
                        "git {} failed: {}",
                        args.join(" "),
                        res.stderr.trim()
                    ));
                }
                use base64::Engine as _;
                base64::engine::general_purpose::STANDARD
                    .decode(res.stdout.trim())
                    .map_err(|e| format!("git {}: undecodable output: {e}", args.join(" ")))
            }
        }
    }

    /// Run `git -C <dir> <args>` with `stdin` fed to it — `git apply`, which takes its patch there
    /// rather than from a temp file this would then have to clean up.
    async fn run_stdin(&self, dir: &Path, args: &[&str], stdin: &[u8]) -> Result<(), String> {
        match self {
            Self::Local => local_git_with_stdin(dir, args, stdin).await.map(|_| ()),
            Self::Remote { runner, .. } => {
                let owned: Vec<String> = args.iter().map(|a| (*a).to_string()).collect();
                // `run_with_stdin` is the v1.1 exec addition; a shim that predates it answers
                // `Unsupported`, which surfaces here as a task failure naming the endpoint rather
                // than a silently empty patch.
                let res = runner
                    .run_with_stdin(
                        "git",
                        &owned,
                        Some(&dir.display().to_string()),
                        REMOTE_GIT_TIMEOUT,
                        stdin,
                    )
                    .await
                    .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
                if res.code != Some(0) {
                    return Err(format!(
                        "git {} failed: {}",
                        args.join(" "),
                        res.stderr.trim()
                    ));
                }
                Ok(())
            }
        }
    }
}

/// This process's identity as the owner of worktrees it creates in a sandbox.
///
/// Not a PID: the sandbox's `/proc` describes processes in the sandbox, and this process is not one
/// of them — a PID looked up there names something unrelated, or with a recycled number something
/// live, which is the direction that deletes a running child's work. A sandbox belongs to exactly one
/// session, so all this has to do is differ from every previous incarnation that used it: anything
/// carrying another owner is an orphan by construction. Hex, and hyphen-free, because the leaf name
/// is `<owner>-<label>` and the sweep splits it on the first hyphen.
fn remote_owner() -> String {
    static OWNER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    OWNER
        .get_or_init(|| {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            format!("{:x}", (nanos as u64) ^ u64::from(std::process::id()) << 32)
        })
        .clone()
}

/// Wrap `s` for a POSIX shell in single quotes, which quote everything but a single quote itself.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Whether `pid` is a live process. `/proc/<pid>` is Linux's answer; this crate targets Linux (see
/// `tools::exec`'s process-group handling, which is equally POSIX-specific).
fn pid_is_alive(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

/// Run `git` in `dir`, returning stdout on success and a message naming the failing command on failure.
async fn local_git(dir: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

/// Like [`git`], but feeds `stdin` to the command — for `git apply`, which reads a patch from stdin
/// rather than taking a temp file we would then have to clean up.
async fn local_git_with_stdin(dir: &Path, args: &[&str], stdin: &[u8]) -> Result<Vec<u8>, String> {
    git_stdout_with_stdin(dir, args, stdin).await
}

/// The enclosing git repository's top-level directory, or `None` when `cwd` isn't inside one.
pub async fn repo_root(git: &Git, cwd: &Path) -> Option<PathBuf> {
    let path = git
        .run_text(cwd, &["rev-parse", "--show-toplevel"])
        .await
        .ok()?;
    let path = path.trim();
    (!path.is_empty()).then(|| PathBuf::from(path))
}

/// Why `isolation: worktree` cannot be honored here, phrased for the model that asked for it. Returns
/// `Ok(repo_root)` when it can.
///
/// Checked up front rather than letting `git worktree add` fail: "fatal: not a git repository" reaching
/// a model as a tool error is a worse experience than being told the agent definition requires a repo.
pub async fn preflight(git: &Git, cwd: &Path) -> Result<PathBuf, String> {
    let Some(root) = repo_root(git, cwd).await else {
        return Err(format!(
            "`isolation: worktree` requires a git repository, but {} is not inside one",
            cwd.display()
        ));
    };
    // `git worktree add … HEAD` needs a HEAD to check out; a repo with no commits has an unborn one.
    if git
        .run_text(&root, &["rev-parse", "--verify", "HEAD"])
        .await
        .is_err()
    {
        return Err(format!(
            "`isolation: worktree` requires at least one commit, but {} has none yet",
            root.display()
        ));
    }
    Ok(root)
}

/// A private checkout of the repository, seeded with the parent's working state. Removed on [`Drop`]
/// (best-effort) or explicitly via [`Worktree::remove`].
pub struct Worktree {
    path: PathBuf,
    repo_root: PathBuf,
    /// Set once the worktree has been removed, so `Drop` doesn't try again (and so a preserved
    /// conflict worktree is never reaped out from under the developer who needs to look at it).
    detached: bool,
    /// Where this worktree's `git` runs. Carried so every later operation on it — the delta, the
    /// removal — lands on the same filesystem the checkout is actually on.
    git: Git,
}

impl Worktree {
    /// Create a worktree for `label` (a short, filesystem-safe tag — the agent name and task index),
    /// seeded with the parent's uncommitted work and a throwaway baseline commit.
    ///
    /// `repo_root` must come from [`preflight`].
    pub async fn create(git: &Git, repo_root: &Path, label: &str) -> Result<Self, String> {
        // Picks (and creates) the first writable candidate root — so an unwritable `$HOME/.cache`
        // degrades to `temp_dir` rather than failing worktree isolation outright.
        let base = ensure_base_dir(git, repo_root).await;
        // Before adding ours, clear out any left by an incarnation that is gone. A no-op locally,
        // where the startup `sweep` already did it against PID liveness.
        git.sweep_siblings(repo_root, &base).await;

        // The leaf basename becomes git's own name for the worktree, so it must be unique within the
        // repo. The owner prefix does double duty: uniqueness across concurrent agents, and the key
        // `sweep` reaps by — a PID locally, this session's incarnation in a sandbox, where a PID would
        // name a process on the wrong machine.
        let leaf = format!("{}-{}", git.owner(), sanitize(label));
        let path = base.join(leaf);
        // Idempotent per CLAUDE.md: a retry after a crash must not fail on its own leftovers.
        git.remove_worktree(repo_root, &path).await;

        git.run_text(
            repo_root,
            &[
                "worktree",
                "add",
                "--detach",
                "--quiet",
                &path.display().to_string(),
                "HEAD",
            ],
        )
        .await?;

        let wt = Self {
            path,
            repo_root: repo_root.to_path_buf(),
            detached: false,
            git: git.clone(),
        };
        // From here on, any failure must not leak the checkout we just made.
        if let Err(e) = wt.seed_from_parent().await {
            wt.git.remove_worktree(&wt.repo_root, &wt.path).await;
            return Err(e);
        }
        Ok(wt)
    }

    /// Copy the parent's uncommitted state into this worktree and commit it as the baseline
    /// [`child_delta`](Self::child_delta) diffs against. See the module doc for why this exists.
    async fn seed_from_parent(&self) -> Result<(), String> {
        // Tracked modifications, staged and unstaged alike (`diff HEAD`, not `diff`). `--binary` so a
        // changed image or fixture survives the round trip instead of becoming a "binary files differ"
        // stub that `git apply` then refuses.
        let patch = self
            .git
            .run_bytes(&self.repo_root, &["diff", "--binary", "HEAD"])
            .await?;
        if !patch.is_empty() {
            self.git
                .run_stdin(&self.path, &["apply", "--whitespace=nowarn"], &patch)
                .await?;
        }

        // Untracked-but-not-ignored files: `git diff` never sees these, but a file the developer just
        // created is exactly the thing they are most likely to be asking a subagent about. `-z` because
        // a path may legally contain a newline.
        let listed = self
            .git
            .run_bytes(
                &self.repo_root,
                &["ls-files", "--others", "--exclude-standard", "-z"],
            )
            .await?;
        for raw in listed.split(|b| *b == 0).filter(|s| !s.is_empty()) {
            let rel =
                Path::new(std::str::from_utf8(raw).map_err(|e| format!("non-utf8 path: {e}"))?);
            let src = self.repo_root.join(rel);
            let dst = self.path.join(rel);
            // A symlink or a file deleted between `ls-files` and now: skip rather than fail the spawn.
            if !self.git.is_file(&src).await {
                continue;
            }
            if let Some(parent) = dst.parent() {
                self.git.create_dir_all(parent).await?;
            }
            self.git.copy_file(&src, &dst).await?;
        }

        // The baseline. `-A` stages the copied untracked files too, so the child's later delta is
        // measured against the tree the parent actually had, not against HEAD.
        self.git.run_text(&self.path, &["add", "-A"]).await?;
        // A repo whose working tree is clean has nothing to commit, and `git commit` exits non-zero on
        // an empty commit — so only commit when `add` produced something staged. (`diff --cached
        // --quiet` exits non-zero, i.e. `Err` here, precisely when there *are* staged changes.)
        if self
            .git
            .run_text(&self.path, &["diff", "--cached", "--quiet"])
            .await
            .is_err()
        {
            self.git
                .run_text(
                    &self.path,
                    &[
                        "-c",
                        "user.name=beyond-agent",
                        "-c",
                        "user.email=agent@beyond.local",
                        "commit",
                        "--quiet",
                        "--no-verify",
                        "-m",
                        "subagent baseline (throwaway)",
                    ],
                )
                .await?;
        }
        Ok(())
    }

    /// This worktree's checkout directory — what the child's tools are rooted at.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Everything the child changed, as a binary-safe patch applicable to the parent tree. Empty when
    /// the child changed nothing.
    ///
    /// `add -A` first, so files the child *created* appear in the diff (`git diff` alone never shows an
    /// untracked file). Diffed against the baseline commit, so the parent's own uncommitted work — which
    /// is already present in the parent — is excluded rather than replayed onto it.
    pub async fn child_delta(&self) -> Result<Vec<u8>, String> {
        self.git.run_text(&self.path, &["add", "-A"]).await?;
        self.git
            .run_bytes(&self.path, &["diff", "--cached", "--binary", "HEAD"])
            .await
    }

    /// Remove the checkout and its git metadata. Consumes `self` — a removed worktree has no path worth
    /// holding. Idempotent: removing one that's already gone succeeds.
    pub async fn remove(mut self) -> Result<(), String> {
        self.detached = true;
        self.git.remove_worktree(&self.repo_root, &self.path).await;
        Ok(())
    }

    /// Where this worktree's `git` runs — for the merge-back, which touches the *parent* tree and so
    /// has to land on the same filesystem the child's did.
    pub fn git(&self) -> &Git {
        &self.git
    }

    /// Give up ownership *without* deleting the checkout — for a child whose patch conflicted, whose
    /// work the developer now needs to inspect at [`path`](Self::path).
    pub fn preserve(mut self) -> PathBuf {
        self.detached = true;
        self.path.clone()
    }
}

impl Drop for Worktree {
    fn drop(&mut self) {
        if self.detached {
            return;
        }
        // Best-effort, and deliberately blocking: this runs when the parent's agent loop *drops* a
        // cancelled subagent's future, and a detached cleanup thread would not survive `process::exit`
        // (PR #13). `git worktree remove` is a fast local operation; the alternative — leaking until the
        // next `sweep` — is strictly worse for the common Ctrl-C case.
        //
        // A sandbox worktree cannot be removed here at all: every step is a round trip to the exec
        // endpoint and `Drop` cannot await one. That is what `sweep` is for, and why a remote owner is
        // keyed to this incarnation rather than to a PID — the next session to use the sandbox reaps
        // whatever this one left. `Worktree::remove` on the ordinary paths still cleans up promptly.
        if self.git.is_remote() {
            tracing::debug!(
                worktree = %self.path.display(),
                "leaving a sandbox worktree for the next sweep: Drop cannot await its removal"
            );
            return;
        }
        remove_worktree_blocking(&self.repo_root, &self.path);
    }
}

/// Remove a worktree, tolerating every "already gone" shape. Synchronous: [`Drop`] cannot await.
fn remove_worktree_blocking(repo_root: &Path, path: &Path) {
    // `--force` because the child almost certainly left the checkout dirty, which `git worktree remove`
    // otherwise refuses. Errors ignored: the directory may already be gone, or never fully created.
    let _ = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["worktree", "remove", "--force"])
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    // `git worktree remove` deletes the directory itself on success. If it refused (metadata already
    // pruned, say), the checkout can still be sitting there.
    let _ = std::fs::remove_dir_all(path);
    let _ = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["worktree", "prune"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    // Tidy the now-possibly-empty per-repo directory. `remove_dir` (not `remove_dir_all`) fails harmlessly
    // when a sibling worktree is still there, so this can never delete a live one.
    if let Some(parent) = path.parent() {
        let _ = std::fs::remove_dir(parent);
    }
}

/// What happened when a child's patch met the parent tree.
#[derive(Debug, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Every hunk applied. The child's worktree can be removed.
    Clean,
    /// Some files applied with conflict markers left in the parent's working tree. The child's worktree
    /// is preserved so the developer (or the parent model) can compare.
    Conflicted { files: Vec<String> },
}

/// Every path a patch touches, as repo-relative strings. Uses `git apply --numstat`, which parses the
/// patch and reports `<added>\t<deleted>\t<path>` **without modifying anything** — a hand-rolled patch
/// parser here would be a second, subtly-different implementation of a format git already understands.
async fn patch_paths(git: &Git, repo_root: &Path, patch: &[u8]) -> Result<Vec<String>, String> {
    // `-z` is load-bearing, not cosmetic. Without it, git **C-quotes** any path containing non-ASCII
    // or special bytes: `sécrets.env` is reported as `"s\303\251crets.env"` (wrapping quotes + octal
    // escapes). That quoted form then slips past the deny-glob re-check in `apply_patch` — a glob like
    // `**/*.env` can't match a string ending in `.env"` — silently merging a denied file back into the
    // parent repo. `-z` emits each record as `<added>\t<deleted>\t<raw-path>\0`, NUL-terminated and
    // unquoted (renames already normalized to their destination path, same as without `-z`), so the
    // deny check sees the real filename.
    let (ok, out, stderr) = git
        .run_stdin_full(repo_root, &["apply", "--numstat", "-z", "-"], patch)
        .await?;
    if !ok {
        return Err(format!("git apply --numstat failed: {}", stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out)
        .split('\0')
        .filter(|record| !record.is_empty())
        .filter_map(|record| record.rsplit('\t').next())
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect())
}

/// Apply a child's patch to the parent tree, refusing outright if it would write a path the parent's
/// policy denies.
///
/// **This check is the only thing standing between a subagent and a `--deny-path` bypass.**
/// `ToolPolicy::before_tool_call` gates the `write`/`edit` *tools*, but `git apply` is not a tool call
/// and never reaches it. A worktree child could therefore edit `<worktree>/secrets.env` — which an
/// *absolute* deny glob like `/home/me/repo/secrets.env` does not match, because the worktree path is
/// different — and merge-back would land it at the denied location in the real repo. So the patch's
/// target paths are resolved against `repo_root` and re-checked here, against the same compiled globs.
///
/// Refusal is all-or-nothing: a patch touching one denied path is rejected whole, rather than partially
/// applied. A half-applied patch is exactly the un-observable intermediate state CLAUDE.md forbids.
pub async fn apply_patch(
    git: &Git,
    repo_root: &Path,
    patch: &[u8],
    denied_paths: &[globset::GlobMatcher],
) -> Result<ApplyOutcome, String> {
    if patch.is_empty() {
        return Ok(ApplyOutcome::Clean);
    }
    if !denied_paths.is_empty() {
        for rel in patch_paths(git, repo_root, patch).await? {
            let absolute = repo_root.join(&rel);
            let absolute = absolute.display().to_string();
            if let Some(m) = denied_paths
                .iter()
                .find(|m| m.is_match(&absolute) || m.is_match(&rel))
            {
                return Err(format!(
                    "refusing to merge subagent changes: '{rel}' is denied by policy (matches {:?})",
                    m.glob().glob()
                ));
            }
        }
    }

    // `--3way` falls back to a three-way merge when a hunk doesn't apply cleanly, leaving ordinary
    // conflict markers rather than refusing. Both blobs are in the shared object database (the child's
    // worktree writes there), so the merge base is always available.
    let (ok, _stdout, stderr) = git
        .run_stdin_full(
            repo_root,
            &["apply", "--3way", "--whitespace=nowarn", "-"],
            patch,
        )
        .await?;
    if ok {
        return Ok(ApplyOutcome::Clean);
    }
    // `git apply --3way` exits non-zero for *both* "applied with conflicts" and "could not apply at
    // all". It prints a `U <path>` line per conflicted file in the first case and nothing of the sort in
    // the second, which is how the two are told apart.
    let files: Vec<String> = stderr
        .lines()
        .filter_map(|l| l.strip_prefix("U "))
        .map(|p| p.trim().to_string())
        .collect();
    if files.is_empty() {
        return Err(format!(
            "could not apply subagent changes: {}",
            stderr.trim()
        ));
    }
    Ok(ApplyOutcome::Conflicted { files })
}

/// Raw `std::process::Output` from a git command fed `stdin` — [`git_with_stdin`] discards it, but
/// `git apply --3way` needs its exit status and stderr inspected rather than turned into an `Err`.
async fn command_output_with_stdin(
    dir: &Path,
    args: &[&str],
    stdin: &[u8],
) -> Result<std::process::Output, String> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(dir).args(args);
    output_with_stdin(cmd, &format!("git {}", args.join(" ")), stdin).await
}

/// Feed `stdin` to `cmd` while **concurrently** draining its stdout and stderr, and return what it
/// wrote once it exits.
///
/// The concurrency is the whole point of this function, not an optimization. A pipe holds about 64 KiB;
/// past that a writer blocks until the far end reads. Push the entire patch in first and only then start
/// reading, and any child that produces more than a pipeful of output before it has swallowed the whole
/// patch wedges the pair of us forever: it blocks writing stdout, therefore stops reading stdin,
/// therefore our `write_all` never returns. Both callers reach that shape on real input — `git apply
/// --numstat -z` emits a record per file (a patch touching a few thousand files overruns the buffer),
/// and `git apply --3way` emits a `U <path>` line per file on a heavily conflicted patch. The result
/// would be a turn hung for the life of the process, holding a child and three pipe fds.
///
/// The `join!` also means the child is always waited on, including when the write fails.
async fn output_with_stdin(
    mut cmd: Command,
    label: &str,
    stdin: &[u8],
) -> Result<std::process::Output, String> {
    use tokio::io::AsyncWriteExt;

    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{label}: {e}"))?;
    let mut pipe = child
        .stdin
        .take()
        .ok_or_else(|| format!("{label}: stdin unavailable"))?;

    let feed = async move {
        let result = pipe.write_all(stdin).await;
        // Dropping the handle closes the write end: without that EOF the child keeps waiting for more
        // input and `wait_with_output` never returns.
        drop(pipe);
        result
    };
    let (written, out) = tokio::join!(feed, child.wait_with_output());
    let out = out.map_err(|e| format!("{label}: {e}"))?;
    if let Err(e) = written {
        // A child that decides the input is garbage and exits before reading all of it (`git apply` on a
        // malformed patch) closes the pipe under us, and we see EPIPE. That is not the failure worth
        // reporting — the exit status and stderr we did collect say what actually went wrong, and the
        // callers already turn those into a real message. Anything else is a genuine I/O fault.
        if e.kind() != std::io::ErrorKind::BrokenPipe {
            return Err(format!("{label}: writing stdin: {e}"));
        }
    }
    Ok(out)
}

/// [`command_output_with_stdin`], erroring on a non-zero exit and returning stdout.
async fn git_stdout_with_stdin(dir: &Path, args: &[&str], stdin: &[u8]) -> Result<Vec<u8>, String> {
    let out = command_output_with_stdin(dir, args, stdin).await?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

/// Reap subagent worktrees left behind by agent processes that are no longer running. Safe to call at
/// any time, from any number of concurrent agents: a worktree whose owning PID is still alive is never
/// touched, and `git worktree prune` only drops metadata for checkouts that are already gone.
///
/// This — not [`Drop`] — is what actually guarantees cleanup, because a `process::exit` (or a SIGKILL)
/// runs no destructors.
pub fn sweep(repo_root: &Path) {
    // Every candidate base, not just the preferred one: a prior run may have created worktrees under a
    // different root (e.g. `temp_dir` while `$HOME` was briefly unwritable), and those orphans must be
    // reaped too.
    for base in base_dirs(repo_root) {
        let Ok(entries) = std::fs::read_dir(&base) else {
            // A base directory that doesn't exist is the normal case, not an error.
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(owner) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.split_once('-'))
                .and_then(|(pid, _)| pid.parse::<u32>().ok())
            else {
                // Not one of ours (or a name we didn't write) — leave it alone.
                continue;
            };
            if pid_is_alive(owner) {
                continue;
            }
            tracing::debug!(
                worktree = %path.display(),
                owner,
                "reaping subagent worktree orphaned by a dead process"
            );
            remove_worktree_blocking(repo_root, &path);
        }
    }
}

/// Reduce `label` to something safe as a single path component: lowercase alphanumerics and hyphens.
/// An agent name is already validated to that shape (`agents::validate_agent_name`), but a task index or
/// a future caller need not be, and this must never produce a `/` or a `..`.
fn sanitize(label: &str) -> String {
    let cleaned: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "agent".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real git repo with one commit, so `preflight` passes and `worktree add HEAD` has a HEAD.
    async fn repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        Git::Local
            .run_text(p, &["init", "--quiet", "-b", "main"])
            .await
            .unwrap();
        Git::Local
            .run_text(p, &["config", "user.name", "t"])
            .await
            .unwrap();
        Git::Local
            .run_text(p, &["config", "user.email", "t@t"])
            .await
            .unwrap();
        std::fs::write(p.join("tracked.txt"), "original\n").unwrap();
        Git::Local.run_text(p, &["add", "-A"]).await.unwrap();
        Git::Local
            .run_text(p, &["commit", "--quiet", "-m", "init"])
            .await
            .unwrap();
        dir
    }

    #[tokio::test]
    async fn preflight_rejects_a_non_repo_and_a_repo_with_no_commits() {
        let plain = tempfile::tempdir().unwrap();
        let err = preflight(&Git::Local, plain.path()).await.unwrap_err();
        assert!(err.contains("not inside one"), "{err}");

        let empty = tempfile::tempdir().unwrap();
        Git::Local
            .run_text(empty.path(), &["init", "--quiet"])
            .await
            .unwrap();
        let err = preflight(&Git::Local, empty.path()).await.unwrap_err();
        assert!(
            err.contains("no commits") || err.contains("none yet"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn a_fresh_worktree_sees_the_parents_uncommitted_tracked_changes() {
        // The whole point of seeding: `git worktree add HEAD` alone would show "original".
        let repo = repo().await;
        std::fs::write(repo.path().join("tracked.txt"), "work in progress\n").unwrap();

        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "scout-0")
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(wt.path().join("tracked.txt")).unwrap(),
            "work in progress\n"
        );
    }

    #[tokio::test]
    async fn a_fresh_worktree_sees_the_parents_untracked_files_but_not_ignored_ones() {
        let repo = repo().await;
        std::fs::write(repo.path().join(".gitignore"), "secret.txt\n").unwrap();
        std::fs::write(repo.path().join("brand-new.rs"), "fn main() {}\n").unwrap();
        std::fs::write(repo.path().join("secret.txt"), "nope\n").unwrap();

        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "scout-0")
            .await
            .unwrap();
        assert!(
            wt.path().join("brand-new.rs").exists(),
            "untracked file must carry over"
        );
        assert!(
            !wt.path().join("secret.txt").exists(),
            "an ignored file must not be copied into the worktree"
        );
    }

    #[tokio::test]
    async fn child_delta_contains_only_the_childs_changes_not_the_parents_wip() {
        // The correctness property the baseline commit exists for. Without it, the parent's own
        // uncommitted edit would be replayed back onto the parent by merge-back.
        let repo = repo().await;
        std::fs::write(repo.path().join("tracked.txt"), "parent wip\n").unwrap();

        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        std::fs::write(wt.path().join("child.txt"), "child made this\n").unwrap();

        let patch = String::from_utf8(wt.child_delta().await.unwrap()).unwrap();
        assert!(
            patch.contains("child.txt"),
            "child's new file must be in the delta: {patch}"
        );
        assert!(
            !patch.contains("parent wip"),
            "the parent's own uncommitted work must NOT be in the delta: {patch}"
        );
    }

    #[tokio::test]
    async fn child_delta_is_empty_when_the_child_changed_nothing() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "scout-0")
            .await
            .unwrap();
        assert!(wt.child_delta().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn child_delta_captures_a_modification_to_a_tracked_file() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        std::fs::write(wt.path().join("tracked.txt"), "changed by child\n").unwrap();

        let patch = String::from_utf8(wt.child_delta().await.unwrap()).unwrap();
        assert!(patch.contains("tracked.txt"), "{patch}");
        assert!(patch.contains("changed by child"), "{patch}");
    }

    #[tokio::test]
    async fn remove_is_idempotent_and_drop_cleans_up() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();

        let wt = Worktree::create(&Git::Local, &root, "scout-0")
            .await
            .unwrap();
        let path = wt.path().to_path_buf();
        assert!(path.exists());
        wt.remove().await.unwrap();
        assert!(!path.exists());
        // Removing what's already gone must not error — a retry after a crash lands here.
        remove_worktree_blocking(&root, &path);

        // And a dropped (never-explicitly-removed) worktree cleans itself up.
        let path2 = {
            let wt = Worktree::create(&Git::Local, &root, "scout-1")
                .await
                .unwrap();
            wt.path().to_path_buf()
        };
        assert!(!path2.exists(), "Drop must remove the checkout");
    }

    #[tokio::test]
    async fn preserve_keeps_the_checkout_for_inspection() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let path = {
            let wt = Worktree::create(&Git::Local, &root, "worker-0")
                .await
                .unwrap();
            std::fs::write(wt.path().join("conflicted.txt"), "x").unwrap();
            wt.preserve()
        };
        assert!(path.exists(), "a preserved worktree survives Drop");
        assert!(path.join("conflicted.txt").exists());
        remove_worktree_blocking(&root, &path);
    }

    #[tokio::test]
    async fn create_is_idempotent_over_a_leftover_directory_at_the_same_path() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "scout-0")
            .await
            .unwrap();
        let path = wt.path().to_path_buf();
        std::mem::forget(wt); // simulate a crash: no Drop, checkout and metadata left behind

        // Same PID, same label ⇒ same path. A retry must reclaim it rather than fail.
        let wt2 = Worktree::create(&Git::Local, &root, "scout-0")
            .await
            .unwrap();
        assert_eq!(wt2.path(), path);
        assert!(path.join("tracked.txt").exists());
    }

    /// A throwaway repo path plus its (created) worktree base dir, cleaned up on drop. `sweep` is scoped
    /// per-repo, so each test gets its own base and they can't interfere when run in parallel.
    struct SweepFixture {
        repo: tempfile::TempDir,
        base: PathBuf,
    }
    impl SweepFixture {
        async fn new() -> Self {
            let repo = tempfile::tempdir().unwrap();
            let base = ensure_base_dir(&Git::Local, repo.path()).await;
            Self { repo, base }
        }
    }
    impl Drop for SweepFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    /// A `Git::Remote` that reaches a *local* temp repo, through the same `CommandRunner` and
    /// `FsBackend` interfaces a sandbox is reached by.
    ///
    /// This is not a mock: `RealRunner` spawns real `git`, `LocalFs` does real I/O, and every call
    /// goes through the remote arm — `run` rather than `tokio::process`, the base64 patch transport,
    /// `run_with_stdin`, `sh` for the sweep. What it does not exercise is the network between a
    /// replica and a sandbox, which is the exec endpoint's own conformance suite's job. What it does
    /// exercise is every line of this module that only runs when the filesystem is somewhere else.
    fn remote_git() -> Git {
        Git::Remote {
            runner: std::sync::Arc::new(crate::tools::exec::RealRunner),
            backend: std::sync::Arc::new(crate::tools::fs::local::LocalFs::new()),
            owner: remote_owner(),
        }
    }

    #[tokio::test]
    async fn the_remote_path_seeds_a_worktree_and_merges_the_childs_work_back() {
        // The whole reason this PR exists: on a replica this sequence used to be refused outright,
        // because every step of it ran host `git` against a cwd that was not the child's.
        let git = remote_git();
        let repo = repo().await;
        let root = preflight(&git, repo.path()).await.unwrap();

        // The parent has uncommitted work — the child must see it.
        std::fs::write(
            root.join("wip.txt"),
            "parent wip
",
        )
        .unwrap();

        let wt = Worktree::create(&git, &root, "remote-child").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(wt.path().join("wip.txt")).unwrap(),
            "parent wip
",
            "a worktree seeded through the remote path must carry the parent's untracked work"
        );

        std::fs::write(
            wt.path().join("child.txt"),
            "from the child
",
        )
        .unwrap();
        let delta = wt.child_delta().await.unwrap();
        assert!(!delta.is_empty(), "the child changed a file");

        assert_eq!(
            apply_patch(&git, &root, &delta, &[]).await.unwrap(),
            ApplyOutcome::Clean
        );
        assert_eq!(
            std::fs::read_to_string(root.join("child.txt")).unwrap(),
            "from the child
"
        );
        wt.remove().await.unwrap();
    }

    #[tokio::test]
    async fn a_patch_over_the_remote_path_survives_bytes_that_are_not_utf8() {
        // Why `run_bytes` pipes through base64 rather than taking `ExecResult::stdout` as-is. That
        // field is a `String`; a text hunk carries the file's own bytes, and a lossy conversion of a
        // Latin-1 source file replaces them with U+FFFD — producing a patch that either fails to
        // apply or applies corruption. Neither is something a tenant should discover.
        let git = remote_git();
        let repo = repo().await;
        let root = preflight(&git, repo.path()).await.unwrap();

        let wt = Worktree::create(&git, &root, "latin1-child").await.unwrap();
        // 0xE9 is `é` in Latin-1 and invalid on its own in UTF-8.
        std::fs::write(wt.path().join("latin1.txt"), b"caf\xe9 latin1\n").unwrap();
        let delta = wt.child_delta().await.unwrap();

        assert!(
            !delta.windows(3).any(|w| w == [0xEF, 0xBF, 0xBD]),
            "the patch came back with U+FFFD in it — the byte path is lossy"
        );
        assert_eq!(
            apply_patch(&git, &root, &delta, &[]).await.unwrap(),
            ApplyOutcome::Clean
        );
        assert_eq!(
            std::fs::read(root.join("latin1.txt")).unwrap(),
            b"caf\xe9 latin1\n",
            "the merged file must be byte-identical to what the child wrote"
        );
        wt.remove().await.unwrap();
    }

    #[tokio::test]
    async fn the_remote_sweep_reaps_another_incarnation_but_never_our_own() {
        // The sandbox has no PID to check liveness against, so the key is the owner: a sandbox
        // belongs to one session, and anything carrying a different owner is from an incarnation
        // that is gone. Getting this backwards would delete a running child's work.
        let git = remote_git();
        let repo = repo().await;
        let root = preflight(&git, repo.path()).await.unwrap();

        let mine = Worktree::create(&git, &root, "keep-me").await.unwrap();
        let base = mine.path().parent().unwrap().to_path_buf();
        // A worktree that looks like it came from a previous incarnation.
        let stale = base.join("deadbeef-gone");
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(stale.join("marker"), "old").unwrap();

        git.sweep_siblings(&root, &base).await;

        assert!(
            !stale.exists(),
            "a previous incarnation's worktree must be reaped"
        );
        assert!(
            mine.path().exists(),
            "our own live worktree must never be swept"
        );
        mine.remove().await.unwrap();
    }

    #[tokio::test]
    async fn sweep_leaves_a_live_processes_worktree_alone() {
        // Jared runs several agents against one repo at once; a sweep that reaped by name rather than by
        // PID liveness would delete a running session's work.
        let fx = SweepFixture::new().await;
        let mine = fx
            .base
            .join(format!("{}-sweep-live-check", std::process::id()));
        std::fs::create_dir_all(&mine).unwrap();

        sweep(fx.repo.path());
        assert!(mine.exists(), "a live PID's worktree must never be reaped");
    }

    #[tokio::test]
    async fn sweep_reaps_a_dead_processes_worktree() {
        let fx = SweepFixture::new().await;
        // PID 0 is never a live userspace process, so `/proc/0` never exists.
        let orphan = fx.base.join("0-sweep-dead-check");
        std::fs::create_dir_all(&orphan).unwrap();

        sweep(fx.repo.path());
        assert!(!orphan.exists(), "an orphaned worktree must be reaped");
    }

    #[tokio::test]
    async fn sweep_ignores_directories_it_did_not_name() {
        let fx = SweepFixture::new().await;
        let foreign = fx.base.join("not-a-pid-prefix");
        std::fs::create_dir_all(&foreign).unwrap();

        sweep(fx.repo.path());
        assert!(
            foreign.exists(),
            "a directory we didn't create must be left alone"
        );
    }

    #[test]
    fn worktrees_prefer_disk_over_tmpfs() {
        // Regression guard. A worktree is a full repo checkout; `std::env::temp_dir()` is `tmpfs` (RAM)
        // on a stock Linux box, and an 8-way fan-out there exhausted it during development. The *preferred*
        // root (first candidate) must be a persistent cache dir, with `temp_dir` only as the last resort.
        if std::env::var_os("HOME").is_some() || std::env::var_os("XDG_CACHE_HOME").is_some() {
            let preferred = &cache_roots()[0];
            assert!(
                !preferred.starts_with(std::env::temp_dir()),
                "the preferred worktree root must not be under the temp dir: {}",
                preferred.display()
            );
        }
        // …but `temp_dir` must always be present as the final fallback.
        assert!(
            cache_roots()
                .last()
                .unwrap()
                .starts_with(std::env::temp_dir())
        );
    }

    #[test]
    fn base_dir_is_scoped_per_repository() {
        // Two repos must not share a base: `sweep` from one would otherwise reap the other's orphans and
        // leave that repo's `git worktree` metadata dangling.
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        assert_ne!(base_dirs(a.path())[0], base_dirs(b.path())[0]);
        // …and it must be stable across calls, or a restart could never reap its own leftovers.
        assert_eq!(base_dirs(a.path())[0], base_dirs(a.path())[0]);
    }

    fn glob(pattern: &str) -> globset::GlobMatcher {
        globset::Glob::new(pattern).unwrap().compile_matcher()
    }

    #[tokio::test]
    async fn a_clean_child_patch_applies_to_the_parent_tree() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        std::fs::write(wt.path().join("tracked.txt"), "child edit\n").unwrap();
        std::fs::write(wt.path().join("new.txt"), "brand new\n").unwrap();
        let patch = wt.child_delta().await.unwrap();

        assert_eq!(
            apply_patch(&Git::Local, &root, &patch, &[]).await.unwrap(),
            ApplyOutcome::Clean
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join("tracked.txt")).unwrap(),
            "child edit\n"
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join("new.txt")).unwrap(),
            "brand new\n"
        );
    }

    #[tokio::test]
    async fn an_empty_patch_is_clean_and_touches_nothing() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        assert_eq!(
            apply_patch(&Git::Local, &root, b"", &[]).await.unwrap(),
            ApplyOutcome::Clean
        );
    }

    #[tokio::test]
    async fn two_children_editing_different_files_both_apply() {
        // The parallel-writer happy path: disjoint edits merge without conflict.
        let repo = repo().await;
        std::fs::write(repo.path().join("b.txt"), "b original\n").unwrap();
        Git::Local
            .run_text(repo.path(), &["add", "-A"])
            .await
            .unwrap();
        Git::Local
            .run_text(repo.path(), &["commit", "--quiet", "-m", "add b"])
            .await
            .unwrap();
        let root = preflight(&Git::Local, repo.path()).await.unwrap();

        let wt_a = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        let wt_b = Worktree::create(&Git::Local, &root, "worker-1")
            .await
            .unwrap();
        std::fs::write(wt_a.path().join("tracked.txt"), "from a\n").unwrap();
        std::fs::write(wt_b.path().join("b.txt"), "from b\n").unwrap();

        let pa = wt_a.child_delta().await.unwrap();
        let pb = wt_b.child_delta().await.unwrap();
        assert_eq!(
            apply_patch(&Git::Local, &root, &pa, &[]).await.unwrap(),
            ApplyOutcome::Clean
        );
        assert_eq!(
            apply_patch(&Git::Local, &root, &pb, &[]).await.unwrap(),
            ApplyOutcome::Clean
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join("tracked.txt")).unwrap(),
            "from a\n"
        );
        assert_eq!(
            std::fs::read_to_string(repo.path().join("b.txt")).unwrap(),
            "from b\n"
        );
    }

    #[tokio::test]
    async fn two_children_editing_the_same_line_leave_conflict_markers_for_the_parent() {
        // Optimistic concurrency: the first patch wins, the second conflicts. We do not abort — the
        // parent resolves the markers with read/edit, holding both children's task descriptions.
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();

        let wt_a = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        let wt_b = Worktree::create(&Git::Local, &root, "worker-1")
            .await
            .unwrap();
        std::fs::write(wt_a.path().join("tracked.txt"), "written by a\n").unwrap();
        std::fs::write(wt_b.path().join("tracked.txt"), "written by b\n").unwrap();
        let pa = wt_a.child_delta().await.unwrap();
        let pb = wt_b.child_delta().await.unwrap();

        assert_eq!(
            apply_patch(&Git::Local, &root, &pa, &[]).await.unwrap(),
            ApplyOutcome::Clean
        );
        let outcome = apply_patch(&Git::Local, &root, &pb, &[]).await.unwrap();
        let ApplyOutcome::Conflicted { files } = outcome else {
            panic!("second overlapping patch must conflict, got {outcome:?}");
        };
        assert_eq!(files, vec!["tracked.txt".to_string()]);

        let merged = std::fs::read_to_string(repo.path().join("tracked.txt")).unwrap();
        assert!(
            merged.contains("<<<<<<<"),
            "conflict markers must reach the tree: {merged}"
        );
        assert!(merged.contains("written by a"), "{merged}");
        assert!(merged.contains("written by b"), "{merged}");
    }

    #[tokio::test]
    async fn a_patch_that_cannot_apply_at_all_is_a_hard_error_not_a_conflict() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let err = apply_patch(&Git::Local, &root, b"this is not a patch\n", &[])
            .await
            .unwrap_err();
        assert!(
            err.contains("could not apply") || err.contains("failed"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn merge_back_refuses_a_patch_touching_a_denied_path() {
        // THE security regression test. `git apply` is not a tool call, so `ToolPolicy::before_tool_call`
        // never sees it. Without this check a worktree child writes `<worktree>/secrets.env` — which an
        // absolute deny glob does not match, the worktree path being different — and merge-back lands it
        // at the denied location in the real repo.
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        std::fs::write(wt.path().join("secrets.env"), "TOKEN=leaked\n").unwrap();
        let patch = wt.child_delta().await.unwrap();

        let denied = [glob("**/secrets.env")];
        let err = apply_patch(&Git::Local, &root, &patch, &denied)
            .await
            .unwrap_err();
        assert!(err.contains("denied by policy"), "{err}");
        assert!(err.contains("secrets.env"), "{err}");
        assert!(
            !repo.path().join("secrets.env").exists(),
            "the denied file must never reach the parent tree"
        );
    }

    #[tokio::test]
    async fn merge_back_refuses_the_whole_patch_when_only_one_path_is_denied() {
        // All-or-nothing: a partially applied patch is exactly the un-observable intermediate state
        // CLAUDE.md forbids.
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        std::fs::write(wt.path().join("allowed.txt"), "fine\n").unwrap();
        std::fs::write(wt.path().join("secrets.env"), "TOKEN=leaked\n").unwrap();
        let patch = wt.child_delta().await.unwrap();

        let err = apply_patch(&Git::Local, &root, &patch, &[glob("**/*.env")])
            .await
            .unwrap_err();
        assert!(err.contains("denied by policy"), "{err}");
        assert!(
            !repo.path().join("allowed.txt").exists(),
            "the innocent half of a denied patch must not be applied either"
        );
    }

    #[tokio::test]
    async fn merge_back_allows_a_patch_that_matches_no_denied_glob() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        std::fs::write(wt.path().join("allowed.txt"), "fine\n").unwrap();
        let patch = wt.child_delta().await.unwrap();

        let outcome = apply_patch(&Git::Local, &root, &patch, &[glob("**/*.env")])
            .await
            .unwrap();
        assert_eq!(outcome, ApplyOutcome::Clean);
        assert!(repo.path().join("allowed.txt").exists());
    }

    #[tokio::test]
    async fn patch_paths_reports_every_touched_file_without_modifying_the_tree() {
        let repo = repo().await;
        let root = preflight(&Git::Local, repo.path()).await.unwrap();
        let wt = Worktree::create(&Git::Local, &root, "worker-0")
            .await
            .unwrap();
        std::fs::write(wt.path().join("tracked.txt"), "edited\n").unwrap();
        std::fs::write(wt.path().join("added.txt"), "new\n").unwrap();
        let patch = wt.child_delta().await.unwrap();

        let mut paths = patch_paths(&Git::Local, &root, &patch).await.unwrap();
        paths.sort();
        assert_eq!(
            paths,
            vec!["added.txt".to_string(), "tracked.txt".to_string()]
        );
        // `--numstat` must not have applied anything.
        assert!(!repo.path().join("added.txt").exists());
        assert_eq!(
            std::fs::read_to_string(repo.path().join("tracked.txt")).unwrap(),
            "original\n"
        );
    }

    /// The pipe-deadlock regression guard. `cat` echoes its stdin straight back, so a payload far larger
    /// than a pipe buffer (~64 KiB) in *both* directions is exactly the shape that wedges a
    /// write-everything-then-drain implementation: the child blocks writing stdout, stops reading stdin,
    /// and our `write_all` never returns. `git apply --numstat` on a patch touching thousands of files
    /// does the same thing with less convenient setup. The timeout is what makes a regression *fail*
    /// rather than hang CI forever.
    #[tokio::test]
    async fn feeding_a_child_that_floods_its_stdout_does_not_deadlock() {
        let payload = vec![b'x'; 4 << 20];
        let out = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            output_with_stdin(Command::new("cat"), "cat", &payload),
        )
        .await
        .expect("deadlocked writing stdin to a child that was flooding its stdout")
        .expect("cat");
        assert!(out.status.success());
        assert_eq!(out.stdout.len(), payload.len());
    }

    /// A child that exits before reading its whole stdin hands us EPIPE on the write. That is not the
    /// interesting failure — the caller wants the exit status and stderr, which is what says why it quit.
    #[tokio::test]
    async fn a_child_that_exits_without_reading_stdin_yields_its_output_not_a_write_error() {
        let payload = vec![b'x'; 4 << 20];
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "echo nope >&2; exit 3"]);
        let out = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            output_with_stdin(cmd, "sh", &payload),
        )
        .await
        .expect("deadlocked writing stdin to a child that never read it")
        .expect("a broken stdin pipe must not mask the child's own output");
        assert_eq!(out.status.code(), Some(3));
        assert_eq!(String::from_utf8_lossy(&out.stderr).trim(), "nope");
    }

    #[test]
    fn sanitize_never_yields_a_path_separator_or_a_parent_ref() {
        assert_eq!(sanitize("scout-0"), "scout-0");
        assert_eq!(sanitize("../../etc/passwd"), "------etc-passwd");
        assert_eq!(sanitize(""), "agent");
        assert!(!sanitize("a/b").contains('/'));
        // The label reaches `base.join(leaf)`; a traversal here would escape the base directory.
        assert!(!sanitize("../evil").contains(".."));
    }
}
