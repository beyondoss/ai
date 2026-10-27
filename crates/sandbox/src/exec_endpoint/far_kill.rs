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
//! 3. **Leftovers.** For a tracked `bash` job the wrapper does not exit with the command: while
//!    processes of the command's group run on (`nohup server &`), it writes
//!    `\x1eexec-shell-exit:<status>\n` to stderr and waits for the group to empty, keeping the
//!    transport attached. [`MarkerStrip`] hears the marker (wherever it lands in stderr) and the
//!    runner reports the leftovers to `on_orphans`, so they become a job — stopped by the same far
//!    kill of the group — exactly as a local shell's leftovers are. Before, the wrapper returned at
//!    `wait $!`, and a far-side leftover was tracked by nothing.
//!
//! Without `setsid` on the target (rare: busybox and util-linux both ship it) the marker says
//! `exec-pid:` instead and the kill reaches that one process, not its descendants; leftovers are
//! not tracked.
//!
//! Not used for `run_with_stdin`: the wrapper backgrounds the command, and a background job's stdin
//! is `/dev/null` — the file content `ShellFs` writes through stdin would never arrive.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, ReadBuf};

/// The far-side wrapper, run as `sh -c WRAPPER sh <linger> <cwd-or-empty> <program> <args...>`.
/// Positional parameters throughout: nothing model-supplied is ever spliced into script text.
///
/// The command is started in the background and `wait`ed on, rather than `exec`ed, because
/// `setsid` *forks* when its caller already leads a process group — and a transport's top-level
/// process often does (`docker exec` makes it a session leader). A background child of a
/// non-interactive shell never leads a group, so `setsid` there calls `setsid(2)` in place and the
/// pid the marker reports is the command's own. The exit status is `wait`'s, so a command killed by
/// signal N exits `128 + N` (the transport can only carry an exit code).
///
/// **Leftovers** (`<linger>` = `1`, a tracked `bash` job): when the command exits but processes of
/// its group are still running (`server &`, `nohup … &`), the wrapper writes
/// `\x1eexec-shell-exit:<status>\n` to stderr and stays — keeping the transport attached — until the
/// group has no live member, then exits with the command's status. The marker tells the runner to
/// adopt them as a job (`on_orphans`), exactly as a local shell's leftovers are, and the far kill
/// of the group still reaches them. Liveness is `kill -0 -<pgid>`, then `/proc` to skip zombies: a
/// container whose pid 1 is not an init (`sleep infinity`) never reaps the orphans it inherits, and
/// a group of zombies is not running. Polled every 500 ms with shell builtins only (no fork per
/// process). A process that leaves the group (`setsid`) is not seen, as with a streaming endpoint.
pub(crate) const WRAPPER: &str = r#"l=$1; d=$2; shift 2
if [ -n "$d" ]; then cd "$d" || exit 1; fi
if command -v setsid >/dev/null 2>&1; then
  setsid sh -c 'printf "\036exec-pgid:%s\n" "$$" >&2; exec "$@"' sh "$@" &
  g=$!
else
  sh -c 'printf "\036exec-pid:%s\n" "$$" >&2; exec "$@"' sh "$@" &
  g=
fi
wait $!; s=$?
[ "$l" = 1 ] && [ -n "$g" ] || exit $s
live() {
  kill -0 -"$g" 2>/dev/null || return 1
  [ -d /proc/self ] || return 0
  for f in /proc/[0-9]*/stat; do
    read -r t 2>/dev/null <"$f" || continue
    t=${t##*) }
    set -- $t
    [ "$3" = "$g" ] && [ "$1" != Z ] && return 0
  done
  return 1
}
if live; then
  printf '\036exec-shell-exit:%s\n' "$s" >&2
  while live; do sleep 0.5 2>/dev/null || sleep 1; done
fi
exit $s"#;

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
/// A marker is `\x1eexec-shell-exit:` plus at most ten digits and a newline; anything longer is
/// not one.
const MAX_MARKER: usize = 32;
/// The marker the wrapper writes when the command exits leaving processes of its group running.
const SHELL_EXIT: &[u8] = b"exec-shell-exit:";

/// Told the command's exit status when it exits leaving processes running — see [`WRAPPER`].
pub(crate) type ShellExitSink<'a> = Box<dyn Fn(i32) + Send + Sync + 'a>;

/// Wraps the transport's stderr: strips a leading `exec-pgid`/`exec-pid` marker line into a
/// [`FarIdCell`], strips an `exec-shell-exit` marker line wherever it lands (into `on_shell_exit`),
/// and passes every other byte through untouched. Stderr that does not start with the marker byte
/// (the wrapper failed before it ran — `cd` into a missing directory) passes through whole. Bytes are
/// held back only while they could still be the start of a marker: from a record separator to its
/// newline, at most [`MAX_MARKER`] bytes. Cancel-safe: all state lives here, not in the read.
pub(crate) struct MarkerStrip<'a, R> {
    inner: R,
    cell: FarIdCell,
    on_shell_exit: Option<ShellExitSink<'a>>,
    /// Nothing has been decided yet: the leading identity marker is still possible.
    first: bool,
    /// Bytes read but not yet handed out; `buf[..ready]` is decided output.
    buf: Vec<u8>,
    ready: usize,
    eof: bool,
}

impl<'a, R> MarkerStrip<'a, R> {
    pub(crate) fn new(inner: R, cell: FarIdCell, on_shell_exit: Option<ShellExitSink<'a>>) -> Self {
        Self {
            inner,
            cell,
            on_shell_exit,
            first: true,
            buf: Vec::new(),
            ready: 0,
            eof: false,
        }
    }

    /// Decide as much of the undecided bytes as possible: mark output ready, or consume a marker.
    /// Returns whether anything was decided.
    fn decide(&mut self) -> bool {
        let undecided = &self.buf[self.ready..];
        if undecided.is_empty() {
            return false;
        }
        let Some(at) = undecided.iter().position(|&b| b == MARK) else {
            self.ready = self.buf.len();
            self.first = false;
            return true;
        };
        if at > 0 {
            self.ready += at;
            self.first = false;
            return true;
        }
        // A record separator at the front: a marker, the start of one, or output.
        let rest = &undecided[1..];
        let nl = rest.iter().take(MAX_MARKER).position(|&b| b == b'\n');
        let Some(nl) = nl else {
            let could_be = rest.len() < MAX_MARKER
                && (SHELL_EXIT.starts_with(&rest[..rest.len().min(SHELL_EXIT.len())])
                    || (self.first && b"exec-p".starts_with(&rest[..rest.len().min(6)])));
            if could_be && !self.eof {
                return false;
            }
            self.ready += 1;
            self.first = false;
            return true;
        };
        let line = &rest[..nl];
        let consumed = if self.first
            && let Some(id) = FarId::parse(line)
        {
            *self.cell.lock().unwrap_or_else(|e| e.into_inner()) = Some(id);
            true
        } else if let Some(code) = line
            .strip_prefix(SHELL_EXIT)
            .and_then(|n| std::str::from_utf8(n).ok())
            .and_then(|n| n.trim().parse::<i32>().ok())
        {
            if let Some(sink) = &self.on_shell_exit {
                sink(code);
            }
            true
        } else {
            false
        };
        self.first = false;
        if consumed {
            let start = self.ready;
            self.buf.drain(start..start + 1 + nl + 1);
        } else {
            self.ready += 1;
        }
        true
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for MarkerStrip<'_, R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = &mut *self;
        loop {
            if this.ready > 0 {
                let n = this.ready.min(out.remaining());
                out.put_slice(&this.buf[..n]);
                this.buf.drain(..n);
                this.ready -= n;
                return Poll::Ready(Ok(()));
            }
            if this.decide() {
                continue;
            }
            if this.eof {
                // `decide` settles everything at EOF, so nothing is left: end of stream.
                return Poll::Ready(Ok(()));
            }
            let mut chunk = [0u8; 8 * 1024];
            let mut rb = ReadBuf::new(&mut chunk);
            match Pin::new(&mut this.inner).poll_read(cx, &mut rb) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Ready(Ok(())) if rb.filled().is_empty() => this.eof = true,
                Poll::Ready(Ok(())) => this.buf.extend_from_slice(rb.filled()),
            }
        }
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
        let (out, id, _) = strip_all(input).await;
        (out, id)
    }

    /// What `MarkerStrip` passes on, the identity it recorded, and the shell exits it heard.
    async fn strip_all(input: &[u8]) -> (Vec<u8>, Option<FarId>, Vec<i32>) {
        let cell = FarIdCell::default();
        let exits = Mutex::new(Vec::new());
        let sink: ShellExitSink<'_> = Box::new(|c| exits.lock().unwrap().push(c));
        let mut r = MarkerStrip::new(input, cell.clone(), Some(sink));
        let mut out = Vec::new();
        r.read_to_end(&mut out).await.unwrap();
        drop(r);
        let id = *cell.lock().unwrap();
        (out, id, exits.into_inner().unwrap())
    }

    /// Feeds `input` one byte per read, so every marker straddles reads.
    struct Trickle<'b>(&'b [u8]);

    impl AsyncRead for Trickle<'_> {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            out: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            if let Some((&b, rest)) = self.0.split_first() {
                out.put_slice(&[b]);
                self.0 = rest;
            }
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn a_shell_exit_marker_is_heard_and_removed_wherever_it_lands() {
        let input = b"\x1eexec-pgid:9\nwarn\npartial\x1eexec-shell-exit:3\nlater\n";
        let (out, id, exits) = strip_all(input).await;
        assert_eq!(out, b"warn\npartiallater\n");
        assert_eq!(id, Some(FarId::Group(9)));
        assert_eq!(exits, vec![3]);

        let cell = FarIdCell::default();
        let exits = Mutex::new(Vec::new());
        let sink: ShellExitSink<'_> = Box::new(|c| exits.lock().unwrap().push(c));
        let mut r = MarkerStrip::new(Trickle(input), cell.clone(), Some(sink));
        let mut out = Vec::new();
        r.read_to_end(&mut out).await.unwrap();
        drop(r);
        assert_eq!(out, b"warn\npartiallater\n", "split across one-byte reads");
        assert_eq!(*cell.lock().unwrap(), Some(FarId::Group(9)));
        assert_eq!(exits.into_inner().unwrap(), vec![3]);
    }

    #[tokio::test]
    async fn only_a_leading_identity_marker_counts_and_lookalikes_pass_through() {
        // An identity marker after other output is the command's own bytes, not the wrapper's.
        let (out, id, exits) = strip_all(b"x\x1eexec-pgid:9\n").await;
        assert_eq!(out, b"x\x1eexec-pgid:9\n");
        assert_eq!((id, exits.len()), (None, 0));
        // A malformed shell-exit and a lone separator at EOF are output.
        let (out, _, exits) = strip_all(b"\x1eexec-shell-exit:x\n\x1eexec-sh").await;
        assert_eq!(out, b"\x1eexec-shell-exit:x\n\x1eexec-sh");
        assert!(exits.is_empty());
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
