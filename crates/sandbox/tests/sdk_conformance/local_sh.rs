//! A transport written inside the test: `sh` on this host, through `std::process::Command`.
//!
//! It knows nothing of HTTP, the exec protocol or this crate's clients. It implements [`Exec`] and
//! nothing else; the files and jobs helpers above it are the crate's, unchanged. That the shared
//! scenarios pass over it with the same results as over the HTTP client is the proof that the
//! layers compose.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use beyond_sandbox::procgroup::{GroupId, GroupKillGuard, kill_process_group};
use beyond_sandbox::{Exec, ExecResult};

pub struct LocalSh;

async fn pump<R: tokio::io::AsyncRead + Unpin>(
    mut r: R,
    sink: beyond_sandbox::exec::ChunkSink<'_>,
) {
    use tokio::io::AsyncReadExt as _;
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => sink(&buf[..n]),
        }
    }
}

/// One command in its own process group, under the caller's deadline. The group is killed when the
/// deadline passes and when the call is dropped, so no test leaves a process behind. A command that
/// finishes is left alone: the job baseline's supervisor keeps jobs running past their start.
async fn spawn(
    program: &str,
    args: &[String],
    cwd: Option<&str>,
    stdin: Option<&[u8]>,
    timeout: Duration,
) -> std::io::Result<ExecResult> {
    use tokio::io::AsyncWriteExt as _;
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let mut child = cmd.spawn()?;
    let mut guard = GroupKillGuard {
        group: child.id().map(GroupId::leader),
    };
    let mut pipe = child.stdin.take();
    let bytes = stdin.map(<[u8]>::to_vec);
    let feed = async move {
        if let (Some(p), Some(b)) = (pipe.as_mut(), bytes) {
            let _ = p.write_all(&b).await;
        }
        drop(pipe);
    };
    match tokio::time::timeout(timeout, async {
        tokio::join!(child.wait_with_output(), feed)
    })
    .await
    {
        Ok((out, ())) => {
            guard.disarm();
            let out = out?;
            Ok(ExecResult {
                code: out.status.code(),
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                ..Default::default()
            })
        }
        Err(_) => {
            if let Some(group) = guard.group.take() {
                let _ = tokio::task::spawn_blocking(move || kill_process_group(group)).await;
            }
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "timed out",
            ))
        }
    }
}

#[async_trait]
impl Exec for LocalSh {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        spawn(program, args, cwd, None, timeout).await
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        spawn(program, args, cwd, Some(stdin), timeout).await
    }

    /// Output as it is produced; dropping the call kills the command (`kill_on_drop`).
    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: beyond_sandbox::exec::ChunkSink<'_>,
        on_stderr: beyond_sandbox::exec::ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .kill_on_drop(true);
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn()?;
        // Dropping this call kills the whole group, not just the shell.
        let _guard = GroupKillGuard {
            group: child.id().map(GroupId::leader),
        };
        let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        let (_, _, status) = tokio::join!(
            pump(&mut out, on_stdout),
            pump(&mut err, on_stderr),
            child.wait()
        );
        Ok(ExecResult {
            code: status?.code(),
            ..Default::default()
        })
    }

    async fn attaches(&self) -> bool {
        true
    }
}
