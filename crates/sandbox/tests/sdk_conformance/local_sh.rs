//! A transport written inside the test: `sh` on this host, through `std::process::Command`.
//!
//! It knows nothing of HTTP, the exec protocol or this crate's clients. It implements [`Exec`] and
//! nothing else; the files and jobs helpers above it are the crate's, unchanged. That the shared
//! scenarios pass over it with the same results as over the HTTP client is the proof that the
//! layers compose.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use async_trait::async_trait;
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

/// One blocking `Command`, on a blocking thread. Output is captured whole.
fn spawn(
    program: &str,
    args: &[String],
    cwd: Option<&str>,
    stdin: Option<&[u8]>,
) -> std::io::Result<ExecResult> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let mut child = cmd.spawn()?;
    let feeder = stdin.map(|bytes| {
        let mut pipe = child.stdin.take().unwrap();
        let bytes = bytes.to_vec();
        std::thread::spawn(move || {
            let _ = pipe.write_all(&bytes);
        })
    });
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let err_reader = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = err.read_to_end(&mut v);
        v
    });
    let mut stdout = Vec::new();
    out.read_to_end(&mut stdout)?;
    let status = child.wait()?;
    if let Some(f) = feeder {
        let _ = f.join();
    }
    let stderr = err_reader.join().unwrap();
    Ok(ExecResult {
        code: status.code(),
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        ..Default::default()
    })
}

#[async_trait]
impl Exec for LocalSh {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        _timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        let (p, a, c) = (program.to_string(), args.to_vec(), cwd.map(str::to_string));
        tokio::task::spawn_blocking(move || spawn(&p, &a, c.as_deref(), None))
            .await
            .unwrap()
    }

    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        _timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        let (p, a, c, i) = (
            program.to_string(),
            args.to_vec(),
            cwd.map(str::to_string),
            stdin.to_vec(),
        );
        tokio::task::spawn_blocking(move || spawn(&p, &a, c.as_deref(), Some(&i)))
            .await
            .unwrap()
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
            .kill_on_drop(true);
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn()?;
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
