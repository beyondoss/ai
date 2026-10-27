//! The current git branch of a working directory, as `get_state` and the `ready` frame report it.

/// The current short branch name at `cwd` (Task #25, pi-parity fix), for `get_state`'s `git_branch`
/// field — matters more for this crate than for pi, since the whole point of the RPC protocol is
/// letting a client with no shared filesystem drive the agent remotely; without this, such a client has
/// no way at all to learn which branch the agent's tools are actually operating against.
///
/// `None` — never an error surfaced to the caller — when `cwd` isn't inside a git repo, `HEAD` is
/// detached (matching `git symbolic-ref`'s own behavior: it only ever resolves a real branch ref, not a
/// raw commit), `git` itself isn't installed, or the lookup fails for any other reason: a client polling
/// `get_state` shouldn't have an unrelated git hiccup fail the whole call. Spawned via
/// `tokio::process::Command` (not a blocking `std::process::Command`) so this never stalls the control
/// loop's own task the way a blocking subprocess wait would. That is a strictly stronger property than
/// [`persist_blocking`]'s: an inline `std::process::Command` has no await point at all, so it pins this
/// task *and* the shared runtime thread, whereas awaiting a `spawn_blocking` suspends the caller either
/// way and only ever frees the thread. No filesystem watcher or caching: a plain request/response poll
/// has nothing to invalidate, and a `git` invocation is cheap enough to just redo every call.
pub(crate) async fn git_branch(cwd: &std::path::Path) -> Option<String> {
    let output = tokio::process::Command::new("git")
        .args(["symbolic-ref", "--short", "HEAD"])
        .current_dir(cwd)
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8(output.stdout).ok()?;
    let branch = branch.trim();
    (!branch.is_empty()).then(|| branch.to_string())
}

/// How long the sandbox gets to answer [`git_branch_in`]. `get_state` is a poll a client makes
/// constantly; a sandbox that is slow, paused or gone must make it answer `null`, not hang it.
pub(super) const GIT_BRANCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// [`git_branch`] for a session whose workspace is **not on this host**: the same
/// `symbolic-ref --short HEAD` lookup, run where the files actually are, through the session's own
/// runner.
///
/// Running the host version against a sandbox path is worse than useless — it either fails, or
/// succeeds by walking up to the *replica's* own checkout and reports that branch as the tenant's.
/// Same `None`-on-anything-unexpected contract as [`git_branch`], plus a timeout, since this one
/// crosses a network hop.
///
/// `None` without asking while the sandbox is not attached yet ([`ExecCell::pending`]): a reopened
/// session that has not used its sandbox must not wake it for a poll, so its branch reads `null`
/// until something does.
///
/// [`ExecCell::pending`]: crate::exec_endpoint::ExecCell::pending
pub(crate) async fn git_branch_in(
    exec: &crate::exec_endpoint::ExecCell,
    cwd: &str,
) -> Option<String> {
    if exec.pending() {
        return None;
    }
    let args = [
        "symbolic-ref".to_string(),
        "--short".to_string(),
        "HEAD".to_string(),
    ];
    let result = exec
        .runner()
        .run("git", &args, Some(cwd), GIT_BRANCH_TIMEOUT)
        .await
        .ok()?;
    if result.code != Some(0) {
        return None;
    }
    let stdout = result.stdout_text();
    let branch = stdout.trim();
    (!branch.is_empty()).then(|| branch.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `git` with `args` in `cwd`, synchronously — test setup only (the production `git_branch`
    /// helper uses the async `tokio::process::Command` instead; see its own doc comment for why).
    fn run_git(cwd: &std::path::Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .status()
            .expect("git must be on PATH for this test");
        assert!(status.success(), "git {args:?} failed in {}", cwd.display());
    }

    #[tokio::test]
    async fn git_branch_reports_none_outside_a_git_repository() {
        // Task #25 (pi-parity fix): a lookup failure (no repo here at all) must report `None`, never
        // an error surfaced to the RPC caller.
        let dir = crate::test_support::isolated_tempdir();
        assert_eq!(git_branch(dir.path()).await, None);
    }

    #[tokio::test]
    async fn git_branch_reports_the_current_branch_inside_a_repository() {
        let dir = tempfile::tempdir().unwrap();
        run_git(dir.path(), &["init", "--quiet", "--initial-branch=main"]);
        run_git(
            dir.path(),
            &["commit", "--quiet", "--allow-empty", "-m", "init"],
        );
        run_git(
            dir.path(),
            &["checkout", "--quiet", "-b", "feature/pi-parity"],
        );

        assert_eq!(
            git_branch(dir.path()).await.as_deref(),
            Some("feature/pi-parity")
        );
    }

    #[tokio::test]
    async fn git_branch_reports_none_on_a_detached_head() {
        // `git symbolic-ref --short HEAD` (and this helper) only ever resolves a real branch ref —
        // detached HEAD has none, matching `symbolic-ref`'s own failure in that state rather than
        // falling back to a raw commit SHA.
        let dir = tempfile::tempdir().unwrap();
        run_git(dir.path(), &["init", "--quiet", "--initial-branch=main"]);
        run_git(
            dir.path(),
            &["commit", "--quiet", "--allow-empty", "-m", "init"],
        );
        run_git(dir.path(), &["checkout", "--quiet", "HEAD~0"]);

        assert_eq!(git_branch(dir.path()).await, None);
    }
}
