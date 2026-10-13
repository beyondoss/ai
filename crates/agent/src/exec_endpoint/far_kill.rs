//! Killing a command that a [`TemplateRunner`](super::TemplateRunner) transport started on the far
//! side.
//!
//! **Killing the transport does not kill the command.** Measured with a real `docker exec` against
//! an Alpine container (2026-10-12, Docker on this host): SIGKILL *and* SIGTERM to the local
//! `docker exec` client left `sh -c 'sleep 123'` running inside the container, as its own session
//! leader (`pid = pgid = sid`), indefinitely. `kubectl exec` behaves the same way, and `ssh` without
//! a pty does not signal the remote command on disconnect either. So a timeout or a `bash_job` kill
//! that only reaps the local transport leaves the far command running — a hung build spinning in
//! the tenant's container long after the agent reported it killed.
//!
//! The fix is to learn the command's identity on the far side and kill it there, through the same
//! transport:
//!
//! 1. Every command runs under [`WRAPPER`], which starts it in **a new session** (`setsid`, so it
//!    leads its own process group and everything it spawns is in that group) and has the command's
//!    own process write `\x1eexec-pgid:<pid>\n` to stderr *before* it `exec`s the real program. The
//!    marker is therefore always the first thing on stderr; [`MarkerStrip`] removes it before any
//!    caller sees a byte.
//! 2. To kill, [`far_kill_argv`] runs `kill -KILL -<pgid>` through the template — a second,
//!    ordinary invocation of the same transport. Measured the same way: the whole group (a
//!    backgrounded grandchild included) is gone within the 300 ms the check waited.
//!
//! Without `setsid` on the target (rare: busybox and util-linux both ship it) the marker says
//! `exec-pid:` instead and the kill reaches that one process, not its descendants.
//!
//! Not used for `run_with_stdin`: the wrapper backgrounds the command, and a background job's stdin
//! is `/dev/null` — the file content `ShellFs` writes through stdin would never arrive.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, ReadBuf};

/// The far-side wrapper, run as `sh -c WRAPPER sh <cwd-or-empty> <program> <args...>`. Positional
/// parameters throughout: nothing model-supplied is ever spliced into script text.
///
/// The command is started in the background and `wait`ed on, rather than `exec`ed, because
/// `setsid` *forks* when its caller already leads a process group — and a transport's top-level
/// process often does (`docker exec` makes it a session leader). A background child of a
/// non-interactive shell never leads a group, so `setsid` there calls `setsid(2)` in place and the
/// pid the marker reports is the command's own. The exit status is `wait`'s, so a command killed by
/// signal N exits `128 + N` (the transport can only carry an exit code).
pub(crate) const WRAPPER: &str = r#"d=$1; shift
if [ -n "$d" ]; then cd "$d" || exit 1; fi
if command -v setsid >/dev/null 2>&1; then
  setsid sh -c 'printf "\036exec-pgid:%s\n" "$$" >&2; exec "$@"' sh "$@" &
else
  sh -c 'printf "\036exec-pid:%s\n" "$$" >&2; exec "$@"' sh "$@" &
fi
wait $!"#;

/// The far-side identity of a running command, as its marker reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FarId {
    /// Leads its own process group: kill the group.
    Group(u32),
    /// No `setsid` on the target: only the process itself can be named.
    Pid(u32),
}

impl FarId {
    fn parse(line: &[u8]) -> Option<Self> {
        let line = std::str::from_utf8(line).ok()?;
        if let Some(n) = line.strip_prefix("exec-pgid:") {
            return n.trim().parse().ok().map(FarId::Group);
        }
        line.strip_prefix("exec-pid:")?
            .trim()
            .parse()
            .ok()
            .map(FarId::Pid)
    }
}

/// The argv (before template expansion) that kills `id` on the far side.
///
/// The shell's own `kill`, not `/bin/kill`: slim images (`debian:*-slim`) ship no `procps`, so there
/// is no `kill` binary, while every shell has the builtin. And `kill -KILL -<pgid>` with **no `--`**:
/// busybox `kill`, builtin and applet alike, rejects `--` as "invalid number" and then fails to
/// signal the group (measured on Alpine), while dash, bash and busybox all read `-<pgid>` after the
/// signal as a process group.
pub(crate) fn far_kill_argv(id: FarId) -> (String, Vec<String>) {
    let target = match id {
        FarId::Group(g) => format!("-{g}"),
        FarId::Pid(p) => p.to_string(),
    };
    (
        "sh".to_string(),
        vec![
            "-c".to_string(),
            r#"kill -KILL "$1""#.to_string(),
            "sh".to_string(),
            target,
        ],
    )
}

/// Where a running command's [`FarId`] lands once its marker has been read.
pub(crate) type FarIdCell = Arc<Mutex<Option<FarId>>>;

/// The byte every marker starts with (ASCII record separator): never the first byte of anything a
/// shell prints on its own, so its absence proves there is no marker.
const MARK: u8 = 0x1e;
/// A marker is `\x1eexec-pgid:` plus at most ten digits and a newline; anything longer is not one.
const MAX_MARKER: usize = 32;

/// Wraps the transport's stderr: strips a leading marker line into a [`FarIdCell`] and passes every
/// other byte through untouched. Stderr that does not start with the marker byte (the wrapper
/// failed before it ran — `cd` into a missing directory) passes through whole.
pub(crate) struct MarkerStrip<R> {
    inner: R,
    cell: FarIdCell,
    /// Bytes read while deciding whether there is a marker; `None` once decided.
    sniff: Option<Vec<u8>>,
    /// Bytes owed to the caller after the decision.
    owed: Vec<u8>,
    owed_at: usize,
}

impl<R> MarkerStrip<R> {
    pub(crate) fn new(inner: R, cell: FarIdCell) -> Self {
        Self {
            inner,
            cell,
            sniff: Some(Vec::new()),
            owed: Vec::new(),
            owed_at: 0,
        }
    }

    /// Settle the sniff buffer: record a marker if it is one, owe the caller everything else.
    fn decide(&mut self, buf: Vec<u8>) {
        self.sniff = None;
        if buf.first() == Some(&MARK)
            && let Some(nl) = buf.iter().position(|&b| b == b'\n')
            && let Some(id) = FarId::parse(&buf[1..nl])
        {
            *self.cell.lock().unwrap_or_else(|e| e.into_inner()) = Some(id);
            self.owed = buf[nl + 1..].to_vec();
        } else {
            self.owed = buf;
        }
        self.owed_at = 0;
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for MarkerStrip<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = &mut *self;
        while let Some(sniff) = this.sniff.as_mut() {
            let decided = sniff.first().is_some_and(|&b| b != MARK)
                || sniff.contains(&b'\n')
                || sniff.len() > MAX_MARKER;
            if decided {
                let buf = std::mem::take(sniff);
                this.decide(buf);
                break;
            }
            let mut chunk = [0u8; MAX_MARKER];
            let mut rb = ReadBuf::new(&mut chunk);
            match Pin::new(&mut this.inner).poll_read(cx, &mut rb) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) if rb.filled().is_empty() => {
                    // EOF while sniffing: whatever arrived is ordinary output.
                    let buf = std::mem::take(sniff);
                    this.decide(buf);
                    break;
                }
                Poll::Ready(Ok(())) => sniff.extend_from_slice(rb.filled()),
            }
        }
        if this.owed_at < this.owed.len() {
            let n = (this.owed.len() - this.owed_at).min(out.remaining());
            out.put_slice(&this.owed[this.owed_at..this.owed_at + n]);
            this.owed_at += n;
            if this.owed_at == this.owed.len() {
                this.owed = Vec::new();
                this.owed_at = 0;
            }
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut this.inner).poll_read(cx, out)
    }
}

/// Run `argv` to completion, blocking, killing it if it outlives `limit`. For the far kill, which
/// runs from a `Drop` or the exit path and must never hang either.
pub(crate) fn run_bounded(argv: &[String], limit: Duration) {
    let Some((program, args)) = argv.split_first() else {
        return;
    };
    let Ok(mut child) = std::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        tracing::warn!(program, "could not start the far-side kill");
        return;
    };
    let deadline = std::time::Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                tracing::warn!(program, "the far-side kill did not finish in time");
                return;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use tokio::io::AsyncReadExt;

    async fn strip(input: &[u8]) -> (Vec<u8>, Option<FarId>) {
        let cell = FarIdCell::default();
        let mut r = MarkerStrip::new(input, cell.clone());
        let mut out = Vec::new();
        r.read_to_end(&mut out).await.unwrap();
        let id = *cell.lock().unwrap();
        (out, id)
    }

    #[tokio::test]
    async fn a_leading_marker_is_recorded_and_removed() {
        let (out, id) = strip(b"\x1eexec-pgid:4242\nwarning: x\n").await;
        assert_eq!(out, b"warning: x\n");
        assert_eq!(id, Some(FarId::Group(4242)));
        let (out, id) = strip(b"\x1eexec-pid:7\n").await;
        assert_eq!(out, b"");
        assert_eq!(id, Some(FarId::Pid(7)));
    }

    #[tokio::test]
    async fn stderr_without_a_marker_passes_through_whole() {
        let (out, id) = strip(b"sh: cd: /nope: No such file or directory\n").await;
        assert_eq!(out, b"sh: cd: /nope: No such file or directory\n");
        assert_eq!(id, None);
        let (out, id) = strip(b"").await;
        assert!(out.is_empty() && id.is_none());
        // A record separator that is not followed by a well-formed marker is output, not eaten.
        let (out, id) = strip(b"\x1enot a marker at all, just bytes\n").await;
        assert_eq!(out, b"\x1enot a marker at all, just bytes\n");
        assert_eq!(id, None);
    }

    #[test]
    fn the_far_kill_names_the_group_or_the_pid() {
        assert_eq!(far_kill_argv(FarId::Group(12)).1[3], "-12");
        assert_eq!(far_kill_argv(FarId::Pid(12)).1[3], "12");
    }
}
