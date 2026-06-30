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
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let child = cmd.spawn()?;
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
            Err(_) => Ok(ExecResult {
                code: None,
                stdout: String::new(),
                stderr: "command exceeded its timeout".into(),
                timed_out: true,
            }),
        }
    }
}
