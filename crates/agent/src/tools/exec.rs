//! A command-execution seam.
//!
//! `bash` runs real commands through [`RealRunner`]; the Beyond tools (fork/sync/logs, M8) run the
//! `beyond` CLI through the same trait, so their tests inject a [`CommandRunner`] that records the
//! argv instead of shelling out. This is the boundary that keeps shell-driven tools testable.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::watch;

/// Per-stream capture caps. We keep the first [`STREAM_HEAD`] and last [`STREAM_TAIL`] bytes of each
/// of stdout/stderr and discard the middle *as it streams*, so a command that emits gigabytes (e.g.
/// `yes`, `cat huge.bin`) holds at most ~`HEAD+TAIL` per stream in memory instead of the whole
/// firehose. Sized well above `bash`'s own 30 KB output cap so any capture-level drop still leaves
/// `bash` plenty to head/tail-truncate (with its own marker) on top.
const STREAM_HEAD: usize = 128 * 1024;
const STREAM_TAIL: usize = 128 * 1024;

/// How long a pipe may sit silent, once the direct child has exited, before we stop waiting on it.
/// Without this, a detached grandchild that inherits the stdout/stderr fd (e.g. `cmd &` backgrounding
/// something before the shell itself exits) keeps the pipe open with no EOF, and the read loop blocks
/// on it until the *outer command timeout* (default 30 minutes — see `bash.rs::DEFAULT_TIMEOUT_MS`)
/// finally fires and reaps the process group. Re-armed on every chunk read after exit, so an actively
/// writing descendant (e.g. a backgrounded `lint-staged` still flushing output) is never cut off early
/// — only a genuinely quiet held-open handle releases us. Matches pi's `waitForChildProcess` fix for
/// earendil-works/pi#5303 (`EXIT_STDIO_GRACE_MS = 100`).
const POST_EXIT_IDLE_GRACE: Duration = Duration::from_millis(100);

/// The result of running a command.
#[derive(Debug, Clone, Default)]
pub struct ExecResult {
    /// Process exit code, or `None` if the command never exited normally — it was killed by a signal
    /// (see [`signal`](Self::signal)) or reaped for exceeding its timeout
    /// ([`timed_out`](Self::timed_out)).
    pub code: Option<i32>,
    /// The signal that terminated the command, when one did (Unix only; always `None` elsewhere, and
    /// `None` whenever `code` is `Some`). Carried separately rather than folded into `code` as the
    /// shell's `128 + N` convention so a consumer can tell a genuine `exit 137` apart from a real
    /// SIGKILL — and so the reported message can name the signal, which is the difference between
    /// "exited with code 137" and "killed by SIGKILL (typically the OOM killer)".
    pub signal: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// True if the command was killed for exceeding its timeout.
    pub timed_out: bool,
    /// True if either stream's middle was dropped because it exceeded the capture cap.
    pub truncated: bool,
}

/// The signal that killed `status`, if one did. `ExitStatus::code()` returns `None` in exactly this
/// case on Unix, which is why a `None` code must never be read as "fine" — see [`ExecResult::signal`].
#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(status)
}

/// Non-Unix has no signal concept — a `None` exit code there really is just "unknown".
#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

/// Human name for the signals a command realistically dies from, so the reported message says
/// something a model (or a human reading the transcript) can act on. Anything outside this short list
/// is reported by number alone rather than pulling in a libc-name lookup for it.
pub fn signal_name(signal: i32) -> Option<&'static str> {
    Some(match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        3 => "SIGQUIT",
        6 => "SIGABRT",
        8 => "SIGFPE",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        13 => "SIGPIPE",
        15 => "SIGTERM",
        24 => "SIGXCPU",
        25 => "SIGXFSZ",
        _ => return None,
    })
}

/// A sink for output chunks as they stream from a running command — used to surface live progress.
/// `Send + Sync` because stdout and stderr are drained on separate concurrent tasks.
pub type ChunkSink<'a> = &'a (dyn Fn(&[u8]) + Send + Sync);

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

    /// Like [`run`](CommandRunner::run), but invokes `on_chunk` with each chunk of stdout/stderr as it
    /// arrives, for live progress. Defaults to the non-streaming `run` (test doubles need not stream).
    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        let _ = on_chunk;
        self.run(program, args, cwd, timeout).await
    }
}

/// Spawns the command for real, capturing stdout/stderr and enforcing a wall-clock timeout
/// (`kill_on_drop` reaps the child if it overruns).
pub struct RealRunner;

impl RealRunner {
    /// Shared spawn/capture body. `on_chunk`, when present, is called with each stdout/stderr chunk as
    /// it streams (live progress); both `run` and `run_streaming` funnel through here.
    async fn exec(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: Option<ChunkSink<'_>>,
    ) -> std::io::Result<ExecResult> {
        // No `PATH` manipulation here — `Command` inherits the parent process's environment
        // (including `PATH`) as-is. Pi-parity note: this is a deliberate, documented DIVERGENCE, not
        // an oversight. Pi's `getShellEnv()` (`packages/coding-agent/src/utils/shell.ts`) prepends a
        // managed-binaries directory (pinned `rg`/`fd` it downloads itself) onto `PATH` for every
        // spawn, so its `grep`/`find` tools work even on a host with neither installed. Beyond's own
        // `grep`/`find` tools are native (ripgrep/globset linked in-process, see `grep.rs`/`find.rs`) —
        // unaffected either way — so this only matters for an arbitrary model-invoked `bash: rg ...` /
        // `fd ...` on a host that happens to lack them. Not worth a from-scratch bundled-binaries
        // mechanism just to cover that narrow case.
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args)
            // Closed, not inherited: a model-run command has no business reading the agent process's
            // real stdin (a shared terminal in `run`, or `serve`'s NDJSON control pipe). Without this,
            // a command that tries to read stdin (bare `cat`, `read`, a prompt left off `-y`) blocks
            // forever waiting for input that will never come, instead of seeing immediate EOF — matches
            // pi's own `spawn(..., { stdio: ["ignore", "pipe", "pipe"] })`.
            .stdin(Stdio::null())
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
        let mut child = cmd.spawn()?;
        // Arms a group-kill for the lifetime of this call: `kill_on_drop` above only reaps the
        // *direct* child, so a future dropped mid-`await` (cancellation, not just a timeout) would
        // otherwise leak a backgrounded grandchild as an orphan. `disarm`ed below once the child has
        // actually exited on its own; still armed on both the timeout branch and an external drop, so
        // either one reaches the whole process group.
        let mut guard = GroupKillGuard { pid: child.id() };
        // Streaming (`bash`'s live path) hands every chunk straight to `on_chunk`'s sink, then ignores
        // the `stdout`/`stderr`/`truncated` fields entirely (see the `Ok` arm's comment and `bash.rs`).
        // So when streaming we skip the ≤256 KiB head/tail `Capture` *and* the final `from_utf8_lossy`
        // allocations below — they'd only be produced to be discarded. The non-streaming `run` path
        // (the `beyond` CLI tools) keeps full capture, since it *does* read these fields.
        let streaming = on_chunk.is_some();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // Drain both pipes *concurrently* with the wait: a child that fills one pipe's OS buffer
        // while we read only the other would deadlock, and an unread pipe stalls the child's exit.
        // `exited` fans the wait's completion out to both drains so each can switch from "block on
        // reads" to "release after a short idle grace" the moment the direct child exits (see
        // `POST_EXIT_IDLE_GRACE`), independent of whether the *other* stream's holder has let go yet.
        let (exited_tx, exited_rx) = watch::channel(false);
        let wait = async {
            let status = child.wait().await;
            let _ = exited_tx.send(true);
            status
        };
        let collect = async {
            let (status, out, err) = tokio::join!(
                wait,
                drain_capped(
                    stdout,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_chunk,
                    exited_rx.clone()
                ),
                drain_capped(
                    stderr,
                    STREAM_HEAD,
                    STREAM_TAIL,
                    on_chunk,
                    exited_rx.clone()
                ),
            );
            (status, out, err)
        };
        match tokio::time::timeout(timeout, collect).await {
            // Lossy on purpose, not by oversight: `bash` (this crate's primary consumer) never reads
            // these two fields for real output — its `run_streaming` sink appends every chunk to its
            // `OutputAccumulator` as raw bytes as they arrive (see `bash.rs`'s `sink` closure), and
            // only falls back to `stdout`/`stderr` here when nothing streamed (a non-streaming test
            // double). The live path this struct actually feeds for `bash` never goes through
            // `from_utf8_lossy`. The Beyond platform tools (`fork`/`sync`/`logs`, `beyond.rs`) *do*
            // consume these fields directly as their whole output — but that's the `beyond` CLI's own
            // stdout/stderr, expected to be human-readable text, not arbitrary binary data the way a
            // `bash`-run command's output can be.
            Ok((status, (stdout, out_trunc), (stderr, err_trunc))) => {
                guard.disarm();
                let status = status?;
                Ok(ExecResult {
                    code: status.code(),
                    signal: exit_signal(&status),
                    // Empty on the streaming path: `drain_capped` returned empty capture buffers there
                    // (nothing to decode), and `bash` never reads these anyway.
                    stdout: if streaming {
                        String::new()
                    } else {
                        String::from_utf8_lossy(&stdout).into_owned()
                    },
                    stderr: if streaming {
                        String::new()
                    } else {
                        String::from_utf8_lossy(&stderr).into_owned()
                    },
                    timed_out: false,
                    truncated: out_trunc || err_trunc,
                })
            }
            Err(_) => {
                // Timed out: unlike a future dropped mid-`await` (`GroupKillGuard`'s own `Drop`, which
                // can't `.await`), we're still in an async context here — so actually wait for the
                // process-group kill to finish via `spawn_blocking` before returning, rather than
                // firing it and hoping it lands before a caller re-checks side effects. Disarms the
                // guard (via `take`) first so its own `Drop` doesn't *also* fire a redundant kill.
                #[cfg(unix)]
                if let Some(pid) = guard.pid.take() {
                    let _ = tokio::task::spawn_blocking(move || kill_process_group(pid)).await;
                }
                drop(guard);
                Ok(ExecResult {
                    code: None,
                    stdout: String::new(),
                    stderr: "command exceeded its timeout".into(),
                    timed_out: true,
                    ..Default::default()
                })
            }
        }
    }
}

/// Owns a spawned process-group leader; kills the whole group on drop unless [`disarm`](Self::disarm)d
/// (the normal, already-reaped exit path). This is what actually closes the cancellation gap
/// `kill_on_drop` leaves open: that flag only reaps the *direct* child on drop, but a future dropped
/// mid-`await` — cancellation racing the whole dispatch, not just this call's own timeout — unwinds
/// through this stack frame the same way, so arming the kill here (rather than only in the timeout
/// branch) covers both.
struct GroupKillGuard {
    pid: Option<u32>,
}

impl GroupKillGuard {
    fn disarm(&mut self) {
        self.pid = None;
    }
}

/// One per in-flight `GroupKillGuard`-triggered cleanup thread, `recv`d by
/// [`wait_for_pending_group_kills`] — see that function's doc comment for why this registry exists
/// at all (in short: `std::process::exit` doesn't wait for threads, so a caller that's about to call
/// it needs an explicit way to wait for a just-dropped guard's cleanup itself).
#[cfg(unix)]
static PENDING_GROUP_KILLS: std::sync::Mutex<Vec<std::sync::mpsc::Receiver<()>>> =
    std::sync::Mutex::new(Vec::new());

impl Drop for GroupKillGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.pid.take() {
            // `Drop` can't `.await`, so this can't be made fully synchronous with the caller — but a
            // real OS thread (not a `tokio::spawn`ed task) is scheduled by the kernel directly, rather
            // than depending on the ambient tokio runtime finding a moment to poll it. That matters
            // specifically when the runtime this drop happens on is a single-threaded `#[tokio::test]`
            // runtime under real contention from many other concurrently-running OS threads (one per
            // test) — a queued task on a busy single-threaded runtime can be delayed well past a
            // `tokio::spawn`'s usual near-instant turnaround; a plain OS thread isn't multiplexed onto
            // that runtime's own poll loop at all.
            //
            // Registered in `PENDING_GROUP_KILLS` (not truly fire-and-forget): a cancellation
            // (SIGTERM/SIGINT/SIGHUP) can reach `std::process::exit` moments after this guard drops,
            // and `process::exit` tears down every thread immediately with no chance for this one to
            // finish — see `wait_for_pending_group_kills`, which such a caller must call first.
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                kill_process_group(pid);
                let _ = tx.send(());
            });
            if let Ok(mut pending) = PENDING_GROUP_KILLS.lock() {
                // Opportunistically reclaim entries whose kill thread has already finished (`Ok`) or
                // whose sender was dropped (`Disconnected`), so this registry — otherwise drained in
                // full only just before `process::exit` — can't grow for the whole lifetime of a
                // long-lived `serve` daemon that cancels one bash after another. A still-running kill
                // thread's receiver (`Err(Empty)`) is kept, so `wait_for_pending_group_kills` can still
                // block on it.
                pending.retain(|rx| {
                    matches!(rx.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty))
                });
                pending.push(rx);
            }
        }
    }
}

/// Block until every `GroupKillGuard` cleanup thread registered so far has actually finished its
/// `kill_process_group` work, or `timeout` elapses overall, whichever comes first.
///
/// Necessary, not just defensive: `GroupKillGuard::drop` can't `.await` (it spawns a detached OS
/// thread instead — see that impl's doc comment), and `std::process::exit` — which every shutdown-
/// signal cancellation path in this crate calls, to set a precise POSIX `128+signal` exit code —
/// terminates every thread immediately with no chance to finish whatever it was doing. Without this,
/// a `run` process that gets SIGTERM'd mid-bash-tool-call could exit before its own cleanup thread
/// ever got to run `kill`, silently orphaning the exact backgrounded grandchild `GroupKillGuard`
/// exists to reap — confirmed live via `run_signal_handling.rs`'s e2e tests, which spawn the real
/// compiled binary and send it a real OS signal. Call this immediately before any
/// `std::process::exit` that follows a turn ending in `agent_core::Error::Cancelled`.
#[cfg(unix)]
pub fn wait_for_pending_group_kills(timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    let Ok(mut pending) = PENDING_GROUP_KILLS.lock() else {
        return;
    };
    let receivers = std::mem::take(&mut *pending);
    drop(pending);
    for rx in receivers {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        // A zero/negative remaining budget still gets one non-blocking `try_recv`-equivalent poll
        // (`recv_timeout(ZERO)`) rather than being skipped outright, so a kill that finished just as
        // the deadline passed is still observed instead of assumed incomplete.
        let _ = rx.recv_timeout(remaining);
    }
}

#[async_trait]
impl CommandRunner for RealRunner {
    async fn run(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, timeout, None).await
    }

    async fn run_streaming(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        on_chunk: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        self.exec(program, args, cwd, timeout, Some(on_chunk)).await
    }
}

/// Read a child pipe, keeping only the first `head_cap` and last `tail_cap` bytes; the middle is
/// discarded as it arrives so memory stays bounded regardless of how much the command emits. Returns
/// the kept bytes (head followed by tail) and whether any middle was dropped.
///
/// Two phases, gated on `exited`: while the direct child is still running, reads block normally (the
/// only other thing worth racing is `exited` itself, so we notice the transition promptly instead of
/// polling). Once the child has exited, a read that doesn't complete within [`POST_EXIT_IDLE_GRACE`]
/// means only a *detached* descendant is holding this pipe open — release rather than block on it
/// indefinitely; a real EOF or actively-arriving output is still captured either way.
async fn drain_capped<R: AsyncRead + Unpin>(
    reader: Option<R>,
    head_cap: usize,
    tail_cap: usize,
    on_chunk: Option<ChunkSink<'_>>,
    mut exited: watch::Receiver<bool>,
) -> (Vec<u8>, bool) {
    let mut cap = Capture::new(head_cap, tail_cap);
    // When streaming, the sink already receives every chunk and the returned capture buffer is
    // discarded by the caller — so don't accumulate it (skips ≤`head_cap`+`tail_cap` bytes of copying
    // plus the head/tail `Vec` growth per streamed command). The non-streaming path (`on_chunk` is
    // `None`) still fills `cap`, since its caller reads the result.
    let capturing = on_chunk.is_none();
    if let Some(mut r) = reader {
        let mut buf = [0u8; 64 * 1024];
        while !*exited.borrow() {
            tokio::select! {
                res = r.read(&mut buf) => {
                    match res {
                        Ok(0) => return cap.finish(),
                        Ok(n) => {
                            // Stream the chunk out for live progress *before* the cap drops any of it.
                            if let Some(sink) = on_chunk {
                                sink(&buf[..n]);
                            }
                            if capturing {
                                cap.push(&buf[..n]);
                            }
                        }
                        // A read error (e.g. the pipe closing under us) ends capture with what we have.
                        Err(_) => return cap.finish(),
                    }
                }
                _ = exited.changed() => {}
            }
        }
        loop {
            match tokio::time::timeout(POST_EXIT_IDLE_GRACE, r.read(&mut buf)).await {
                Ok(Ok(0)) | Err(_) => break, // real EOF, or idle grace elapsed with nothing new
                Ok(Ok(n)) => {
                    if let Some(sink) = on_chunk {
                        sink(&buf[..n]);
                    }
                    if capturing {
                        cap.push(&buf[..n]);
                    }
                }
                Ok(Err(_)) => break,
            }
        }
    }
    cap.finish()
}

/// A bounded head+tail byte accumulator. `head` fills first; once full, further bytes feed a rolling
/// `tail` window holding the most recent `tail_cap` bytes. Work is per-chunk (not per-byte), so the
/// cost is proportional to bytes read, not bytes kept.
struct Capture {
    head: Vec<u8>,
    tail: Vec<u8>,
    total: u64,
    head_cap: usize,
    tail_cap: usize,
}

impl Capture {
    fn new(head_cap: usize, tail_cap: usize) -> Self {
        Self {
            head: Vec::new(),
            tail: Vec::new(),
            total: 0,
            head_cap,
            tail_cap,
        }
    }

    fn push(&mut self, mut buf: &[u8]) {
        self.total += buf.len() as u64;
        if self.head.len() < self.head_cap {
            let take = (self.head_cap - self.head.len()).min(buf.len());
            self.head.extend_from_slice(&buf[..take]);
            buf = &buf[take..];
        }
        if buf.is_empty() {
            return;
        }
        // Maintain `tail` as the last `tail_cap` bytes. A chunk larger than the window replaces it
        // outright; otherwise append and trim the front back down to the cap.
        if buf.len() >= self.tail_cap {
            self.tail.clear();
            self.tail
                .extend_from_slice(&buf[buf.len() - self.tail_cap..]);
        } else {
            self.tail.extend_from_slice(buf);
            if self.tail.len() > self.tail_cap {
                self.tail.drain(..self.tail.len() - self.tail_cap);
            }
        }
    }

    /// `(kept_bytes, dropped_middle)`. The middle was dropped iff more bytes passed through than we
    /// kept — i.e. `head` and `tail` weren't contiguous.
    fn finish(self) -> (Vec<u8>, bool) {
        let dropped = self.total > (self.head.len() + self.tail.len()) as u64;
        let mut out = self.head;
        out.extend_from_slice(&self.tail);
        (out, dropped)
    }
}

/// SIGKILL an entire process group: `kill -KILL -<pgid>` (falling back to `kill -KILL <pgid>`, no
/// leading `-`, if that doesn't succeed — matching pi's own `killProcessTree`,
/// `process.kill(-pid, "SIGKILL")` falling back to `process.kill(pid, "SIGKILL")`), *then* an explicit
/// sweep that individually signals every process still reporting this pgid.
///
/// The sweep isn't defensive padding — it's load-bearing. Confirmed live on a real, resource-contended
/// GitHub Actions runner (a fast dedicated dev box never reproduces this): a backgrounded grandchild,
/// independently confirmed alive and already a member of the target pgid *before* the group-kill was
/// even issued, survived it completely — through its full multi-second sleep — while the group
/// leader died promptly. `kill -KILL -pgid` returning success only means the signal was handed to the
/// kernel for group-wide delivery; it does not appear to guarantee every existing member actually
/// receives it on every environment. Enumerating `ps`'s own pid/pgid columns and individually
/// `kill -KILL <pid>`-ing each match closes that gap without depending on group-signal semantics at
/// all. `pgid` doubles as the original child's own pid: the child is spawned as its own process-group
/// leader (`process_group(0)`), so pid and pgid are the same number. Shelling out (not `libc`/`nix`)
/// keeps this free of `unsafe`, which the workspace forbids.
///
/// Blocking, not async: called from two places that each need a *synchronous* guarantee rather than a
/// detached future — [`GroupKillGuard`]'s `Drop` impl (which can't `.await` at all, so this runs on a
/// dedicated OS thread via `std::thread::spawn`, scheduled by the kernel independent of whatever the
/// ambient tokio runtime is doing) and `exec`'s timeout branch (via `spawn_blocking`, so the caller
/// actually waits for the kill to complete before `run()` returns, rather than firing it and hoping it
/// lands before a caller re-checks side effects).
#[cfg(unix)]
fn kill_process_group(pgid: u32) {
    let group_result = std::process::Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{pgid}"))
        .status();
    // A non-success here (including `kill`'s own exit code, e.g. ESRCH because the group already
    // exited on its own between the timeout firing and this running) isn't worth logging on its own —
    // the group kill covers the overwhelmingly common case, so try the direct fallback next regardless
    // of *why* it didn't succeed; a no-op fallback against an already-gone process is harmless.
    if !matches!(&group_result, Ok(status) if status.success()) {
        match std::process::Command::new("kill")
            .arg("-KILL")
            .arg(pgid.to_string())
            .status()
        {
            // Still only the "couldn't even run `kill`" case is worth a warning — a non-zero exit
            // here most likely just means the process (or its whole group) was already gone.
            Ok(_) => {}
            Err(e) => {
                // If `kill` itself couldn't run (missing binary, restrictive sandboxing), a
                // backgrounded grandchild from the timed-out command may be left running with
                // nothing else to reap it — surface that instead of silently losing the signal.
                tracing::warn!(pgid, error = %e, "failed to run `kill` to reap a timed-out process");
            }
        }
    }
    // Two passes with a short gap: the first catches whatever the group-kill above missed; the
    // second catches anything that was itself mid-fork (and so invisible to `ps`) during the first.
    sweep_kill_remaining_group_members(pgid);
    std::thread::sleep(Duration::from_millis(50));
    sweep_kill_remaining_group_members(pgid);
}

/// Enumerate every process currently reporting `pgid` via `ps -eo pid=,pgid=` and SIGKILL each one
/// individually. See [`kill_process_group`]'s doc comment for why this exists — a bulk
/// `kill -KILL -<pgid>` was empirically observed to miss a live, already-a-member process on a real
/// CI runner. Best-effort: a `ps`/`kill` invocation failing outright is silently ignored (mirrors the
/// group-kill's own "no-op fallback against an already-gone process is harmless" stance) since the
/// caller already tried the cheaper group-kill/direct-kill paths first.
#[cfg(unix)]
fn sweep_kill_remaining_group_members(pgid: u32) {
    let Ok(out) = std::process::Command::new("ps")
        .args(["-eo", "pid=,pgid="])
        .output()
    else {
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let (Some(pid_str), Some(pgid_str)) = (fields.next(), fields.next()) else {
            continue;
        };
        if pgid_str.parse::<u32>() != Ok(pgid) {
            continue;
        }
        // Best-effort per-pid kill: an already-dead pid (ESRCH) is the overwhelmingly common,
        // harmless case (the group-kill above likely already got it) — nothing further to do either
        // way, so the result isn't checked.
        let _ = std::process::Command::new("kill")
            .arg("-KILL")
            .arg(pid_str)
            .status();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawned_commands_see_stdin_already_closed_not_the_real_terminal() {
        // Pi-parity audit H1: a model-run command must never inherit the agent process's real stdin —
        // `serve`'s NDJSON control pipe or an interactive `run`'s terminal. `cat` with no args reads
        // stdin until EOF; if stdin were inherited (and this test's own harness stdin stays open) it
        // would hang until the timeout fired. With stdin closed, `cat` sees immediate EOF and exits
        // clean well within the generous timeout below.
        let res = RealRunner
            .run("cat", &[], None, Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(res.code, Some(0), "got: {res:?}");
        assert!(!res.timed_out, "cat blocked on stdin instead of seeing EOF");
        assert_eq!(res.stdout, "");
    }

    // Both tests below race a `kill -KILL -<pgid>` against a backgrounded grandchild's own delayed
    // write. See `kill_process_group`'s own doc comment for what these margins are guarding against —
    // a real GitHub Actions run proved this isn't merely a delivery-latency question (the grandchild
    // ran its *entire* multi-second sleep to completion, unaffected, while the group leader died
    // promptly) — `sweep_kill_remaining_group_members` is the actual fix; these wider margins
    // (vs. the original 1s/1.2s, sized for a fast uncontended dev box) are cheap insurance on top.
    const GRANDCHILD_DELAY: Duration = Duration::from_secs(3);
    const SAFETY_WAIT: Duration = Duration::from_millis(3500);

    #[tokio::test]
    async fn timeout_kills_backgrounded_grandchildren() {
        // A backgrounded grandchild would, after GRANDCHILD_DELAY, write a marker file. The command
        // times out at 300ms; the process-group kill must reach the grandchild so the marker is never
        // written.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("leaked");
        let script = format!(
            "( sleep {}; echo leaked > {} ) & sleep 30",
            GRANDCHILD_DELAY.as_secs(),
            marker.display()
        );
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
        tokio::time::sleep(SAFETY_WAIT).await;
        assert!(
            !marker.exists(),
            "backgrounded grandchild survived the timeout — process group not killed"
        );
    }

    #[tokio::test]
    async fn cancelling_the_future_kills_backgrounded_grandchildren() {
        // Same fixture as `timeout_kills_backgrounded_grandchildren`, but this time the *caller* drops
        // the `run` future outright (as cancellation does) rather than the command's own timeout
        // firing — `kill_on_drop` alone only reaps the direct child, so without `GroupKillGuard` the
        // backgrounded grandchild would survive this path even though the timeout path already
        // handles it.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("leaked");
        let script = format!(
            "( sleep {}; echo leaked > {} ) & sleep 30",
            GRANDCHILD_DELAY.as_secs(),
            marker.display()
        );

        let args = vec!["-c".to_string(), script];
        tokio::select! {
            _ = RealRunner.run(
                "sh",
                &args,
                None,
                Duration::from_secs(30), // long enough that the timeout branch never fires
            ) => panic!("the run future should have been dropped first"),
            _ = tokio::time::sleep(Duration::from_millis(200)) => {
                // The `run` future is dropped here as `select!`'s other arm is discarded.
            }
        }

        // Wait past when the grandchild would have written the marker; it must not exist.
        tokio::time::sleep(SAFETY_WAIT).await;
        assert!(
            !marker.exists(),
            "backgrounded grandchild survived dropping the future — process group not killed"
        );
    }

    #[tokio::test]
    async fn resolves_promptly_when_a_detached_child_holds_the_pipe_open_but_stays_quiet() {
        // pi issue #5303: a shell exits immediately but a backgrounded sleeper inherits the stdout
        // pipe and holds it open without writing. We must release via the post-exit idle grace rather
        // than block until the pipe's true EOF (which wouldn't arrive for 30s here) — and, before the
        // fix, not until the overall command timeout either.
        let start = std::time::Instant::now();
        let res = RealRunner
            .run(
                "sh",
                &["-c".into(), "printf 'DONE\\n'; ( sleep 30 ) &".into()],
                None,
                Duration::from_secs(25), // far longer than the idle grace should ever need
            )
            .await
            .unwrap();
        assert!(!res.timed_out);
        assert_eq!(res.stdout, "DONE\n");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "took {:?}; must release via the idle grace, not wait on the held-open pipe",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn captures_output_emitted_after_exit_while_a_detached_child_holds_stdout_open() {
        // Companion to the test above: the idle grace must not cut off output that's still actively
        // arriving from a detached descendant — only a genuinely quiet handle releases early. Ticks
        // land every 50ms, each one re-arming the 100ms grace, well past a single grace window.
        let res = RealRunner
            .run(
                "sh",
                &[
                    "-c".into(),
                    "printf 'HEAD\\n'; ( for i in 1 2 3 4 5 6; do sleep 0.05; printf \"TICK$i\\n\"; done ) &"
                        .into(),
                ],
                None,
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert!(!res.timed_out);
        assert!(res.stdout.contains("HEAD"), "got: {:?}", res.stdout);
        assert!(res.stdout.contains("TICK6"), "got: {:?}", res.stdout);
    }

    #[tokio::test]
    async fn run_streaming_emits_chunks_reconstructing_stdout() {
        // The streaming runner must hand each output chunk to `on_chunk` as it arrives, and the
        // concatenated chunks must reconstruct the full stdout (it streams before the cap, not after).
        // The sink *is* the output on this path — the `ExecResult::stdout`/`stderr` fields are left
        // empty on purpose (the head/tail capture is skipped when streaming, since the only consumer,
        // `bash`, reads the streamed chunks, never these fields — see the `exec` `Ok` arm).
        let collected = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let c = collected.clone();
        let sink = move |bytes: &[u8]| c.lock().unwrap().extend_from_slice(bytes);
        let res = RealRunner
            .run_streaming(
                "sh",
                &["-c".into(), "printf 'hello\\nworld\\n'".into()],
                None,
                Duration::from_secs(5),
                &sink,
            )
            .await
            .unwrap();
        assert_eq!(
            res.stdout, "",
            "streaming skips the discarded capture; the sink is the source of truth"
        );
        let streamed = String::from_utf8(collected.lock().unwrap().clone()).unwrap();
        assert_eq!(
            streamed, "hello\nworld\n",
            "streamed chunks must reconstruct stdout"
        );
    }

    #[tokio::test]
    async fn kill_process_group_completes_promptly_when_the_group_kill_finds_nothing_to_kill() {
        // Track L10: exercises the two-attempt sequence end to end. Once the process has already
        // exited and been reaped, the group kill (`kill -KILL -pgid`) finds nothing (ESRCH) and the
        // direct-pid fallback that follows must also complete cleanly — no hang, no panic — rather
        // than the two-step sequence somehow blocking on a process that's already gone.
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg("true")
            .spawn()
            .unwrap();
        let pid = child.id().unwrap();
        child.wait().await.unwrap(); // let it actually exit and get reaped

        tokio::time::timeout(
            Duration::from_secs(5),
            tokio::task::spawn_blocking(move || kill_process_group(pid)),
        )
        .await
        .expect("kill_process_group must not hang when there's nothing left to kill")
        .expect("kill_process_group must not panic");
    }

    #[test]
    fn capture_keeps_head_and_tail_drops_middle() {
        let mut cap = Capture::new(4, 4);
        // Feed 26 bytes through a 4+4 window in several chunks.
        for c in b"abcdefghijklmnopqrstuvwxyz".chunks(7) {
            cap.push(c);
        }
        let (out, dropped) = cap.finish();
        assert!(dropped, "middle should be reported as dropped");
        assert_eq!(&out, b"abcdwxyz", "kept first 4 + last 4 bytes");
    }

    #[test]
    fn capture_no_drop_when_under_cap() {
        let mut cap = Capture::new(8, 8);
        cap.push(b"hello");
        let (out, dropped) = cap.finish();
        assert!(!dropped);
        assert_eq!(&out, b"hello");
    }

    #[tokio::test]
    async fn bounded_capture_survives_a_firehose() {
        // `yes` emits unboundedly fast; with the old `wait_with_output()` this buffered the whole
        // stream and OOMed. Capture must hold ~HEAD+TAIL, return promptly on timeout, and the kept
        // bytes (after lossy decode) must start at the true beginning of the stream.
        let res = RealRunner
            .run(
                "sh",
                &["-c".into(), "yes AAAAAAAA".into()],
                None,
                Duration::from_millis(400),
            )
            .await
            .unwrap();
        // Timed out (yes never exits), so stdout was discarded by the timeout branch — prove instead
        // with a bounded *finite* firehose that exits on its own.
        assert!(res.timed_out);

        let res = RealRunner
            .run(
                "sh",
                // ~5 MiB of output that exits cleanly — far above the 256 KiB capture window.
                &[
                    "-c".into(),
                    "yes ABCDEFGHIJKLMNOPQRST | head -n 250000".into(),
                ],
                None,
                Duration::from_secs(30),
            )
            .await
            .unwrap();
        assert!(!res.timed_out);
        assert!(
            res.truncated,
            "5 MiB through a 256 KiB window must drop the middle"
        );
        assert!(
            res.stdout.len() <= STREAM_HEAD + STREAM_TAIL + 64,
            "captured {} bytes; expected bounded to ~{}",
            res.stdout.len(),
            STREAM_HEAD + STREAM_TAIL
        );
        assert!(
            res.stdout.starts_with("ABCDEFGHIJKLMNOPQRST"),
            "head preserved"
        );
        assert!(
            res.stdout.trim_end().ends_with("ABCDEFGHIJKLMNOPQRST"),
            "tail preserved"
        );
    }
}
