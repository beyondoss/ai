//! A plain local runner for the crate's unit tests: `sh` and friends on this host, nothing more.
//! (The host process runner proper, with live streaming and group kills, belongs to the agent.)

use std::time::Duration;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt as _;

use crate::exec::{CommandRunner, ExecResult};
use crate::procgroup::{GroupId, GroupKillGuard, kill_process_group};

pub(crate) struct Host;

impl Host {
    /// One command in its own process group, under a deadline. The whole group is killed when the
    /// deadline passes and when the call is dropped, so no test leaves a process behind; a command
    /// that finishes is left alone (the job supervisor's kept jobs outlive their start).
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
                    stdout: out.stdout,
                    stderr: out.stderr,
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

    async fn run_streaming(
        &self,
        _: &str,
        _: &[String],
        _: Option<&str>,
        _: std::time::Duration,
        _: crate::exec::ChunkSink<'_>,
    ) -> std::io::Result<crate::exec::ExecResult> {
        unimplemented!("a test double that only runs to completion")
    }

    async fn run_streaming_tracked(
        &self,
        _: &str,
        _: &[String],
        _: Option<&str>,
        _: &crate::exec::JobHooks<'_>,
    ) -> std::io::Result<crate::exec::ExecResult> {
        unimplemented!("a test double that only runs to completion")
    }

    async fn run_attached(
        &self,
        _: &str,
        _: &[String],
        _: Option<&str>,
        _: crate::exec::ChunkSink<'_>,
        _: crate::exec::ChunkSink<'_>,
    ) -> std::io::Result<crate::exec::ExecResult> {
        unimplemented!("a test double that only runs to completion")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A command that outlives its timeout leaves nothing running, grandchildren included.
    #[tokio::test]
    async fn a_command_past_its_timeout_leaves_no_process_behind() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("pid");
        let script = format!("sleep 300 & echo $! > {}; wait", pidfile.display());
        let err = Host
            .run(
                "sh",
                &["-c".into(), script],
                None,
                Duration::from_millis(500),
            )
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        let pid: i32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        for _ in 0..100 {
            // Gone, or a zombie waiting for init to reap it.
            match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                Err(_) => return,
                Ok(stat) if stat.contains(") Z") => return,
                Ok(_) => {}
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("the grandchild {pid} outlived the timeout");
    }
}
