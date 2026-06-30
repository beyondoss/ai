//! A command-execution seam.
//!
//! `bash` runs real commands through [`RealRunner`]; the Beyond tools (fork/sync/logs, M8) run the
//! `beyond` CLI through the same trait, so their tests inject a [`CommandRunner`] that records the
//! argv instead of shelling out. This is the boundary that keeps shell-driven tools testable.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;

/// The result of running a command.
#[derive(Debug, Clone)]
pub struct ExecResult {
    /// Process exit code, or `None` if killed (e.g. timed out).
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// True if the command was killed for exceeding its timeout.
    pub timed_out: bool,
}

/// Runs an external command. Implemented by [`RealRunner`] (production) and by test doubles that
/// capture the invocation.
#[async_trait]
pub trait CommandRunner: Send + Sync {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult>;
}

/// Spawns the command for real, capturing stdout/stderr and enforcing a wall-clock timeout
/// (`kill_on_drop` reaps the child if it overruns).
pub struct RealRunner;

#[async_trait]
impl CommandRunner for RealRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // Run the command as its own process-group leader so a timeout can kill the *whole* tree —
        // a `sh -c "foo &"` that backgrounds children would otherwise leak them as orphans, since
        // `kill_on_drop` only reaps the direct child.
        #[cfg(unix)]
        cmd.process_group(0);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let child = cmd.spawn()?;
        let pid = child.id();
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(result) => {
                let out = result?;
                Ok(ExecResult {
                    code: out.status.code(),
                    stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                    timed_out: false,
                })
            }
            Err(_) => {
                // Timed out: kill the entire process group (leader pid == group id), reaching any
                // backgrounded grandchildren, then report the timeout.
                #[cfg(unix)]
                if let Some(pid) = pid {
                    kill_process_group(pid).await;
                }
                let _ = pid; // used only on unix
                Ok(ExecResult {
                    code: None,
                    stdout: String::new(),
                    stderr: "command exceeded its timeout".into(),
                    timed_out: true,
                })
            }
        }
    }
}

/// SIGKILL an entire process group via the `kill` binary (`kill -KILL -<pgid>`). Shelling out keeps
/// this free of `unsafe`/`libc`, which the workspace forbids.
#[cfg(unix)]
async fn kill_process_group(pgid: u32) {
    let _ = tokio::process::Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{pgid}"))
        .status()
        .await;
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn timeout_kills_backgrounded_grandchildren() {
        // A backgrounded grandchild would, after 1s, write a marker file. The command times out at
        // 300ms; the process-group kill must reach the grandchild so the marker is never written.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("leaked");
        let script = format!("( sleep 1; echo leaked > {} ) & sleep 30", marker.display());
        let res = RealRunner
            .run(
                "sh",
                &["-c".into(), script],
                None,
                Duration::from_millis(300),
            )
            .await
            .unwrap();
        assert!(res.timed_out);

        // Wait past when the grandchild would have written the marker; it must not exist.
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            !marker.exists(),
            "backgrounded grandchild survived the timeout — process group not killed"
        );
    }
}
