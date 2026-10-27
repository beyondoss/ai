//! A plain local runner for the crate's unit tests: `sh` and friends on this host, nothing more.
//! (The host process runner proper, with live streaming and group kills, belongs to the agent.)

use std::time::Duration;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt as _;

use crate::exec::{CommandRunner, ExecResult};

pub(crate) struct Host;

impl Host {
    async fn go(
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecResult> {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn()?;
        let mut pipe = child.stdin.take();
        let bytes = stdin.map(<[u8]>::to_vec);
        let feed = async move {
            if let (Some(p), Some(b)) = (pipe.as_mut(), bytes) {
                let _ = p.write_all(&b).await;
            }
            drop(pipe);
        };
        let (out, ()) = tokio::time::timeout(timeout, async {
            tokio::join!(child.wait_with_output(), feed)
        })
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "timed out"))?;
        let out = out?;
        Ok(ExecResult {
            code: out.status.code(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            ..Default::default()
        })
    }
}

#[async_trait]
impl CommandRunner for Host {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        Self::go(program, args, cwd, timeout, None).await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        Self::go(program, args, cwd, timeout, Some(stdin)).await
    }
}
