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
    /// Process exit code, or `None` if killed (e.g. timed out).
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// True if the command was killed for exceeding its timeout.
    pub timed_out: bool,
    /// True if either stream's middle was dropped because it exceeded the capture cap.
    pub truncated: bool,
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
                Ok(ExecResult {
                    code: status?.code(),
                    stdout: String::from_utf8_lossy(&stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&stderr).into_owned(),
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
            std::thread::spawn(move || kill_process_group(pid));
        }
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
                            cap.push(&buf[..n]);
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
                    cap.push(&buf[..n]);
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

/// SIGKILL an entire process group via the `kill` binary (`kill -KILL -<pgid>`), falling back to
/// directly killing just the original process (`kill -KILL <pgid>`, no leading `-`) if the group kill
/// doesn't succeed — matching pi's own `killProcessTree` (`process.kill(-pid, "SIGKILL")`, falling back
/// to `process.kill(pid, "SIGKILL")` on failure). `pgid` doubles as the original child's own pid here:
/// the child is spawned as its own process-group leader (`process_group(0)`), so pid and pgid are the
/// same number. Shelling out keeps this free of `unsafe`/`libc`, which the workspace forbids.
///
/// Blocking, not async: called from two places that each need a *synchronous* guarantee rather than a
/// detached future — [`GroupKillGuard`]'s `Drop` impl (which can't `.await` at all, so this runs on a
/// dedicated OS thread via `std::thread::spawn`, scheduled by the kernel independent of whatever the
/// ambient tokio runtime is doing) and `exec`'s timeout branch (via `spawn_blocking`, so the caller
/// actually waits for the kill to complete before `run()` returns, rather than firing it and hoping it
/// lands before a caller re-checks side effects — the fire-and-forget shape here previously raced a
/// backgrounded grandchild's own delayed write under real scheduling contention).
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
    if matches!(&group_result, Ok(status) if status.success()) {
        return;
    }
    match std::process::Command::new("kill")
        .arg("-KILL")
        .arg(pgid.to_string())
        .status()
    {
        // Still only the "couldn't even run `kill`" case is worth a warning — a non-zero exit here
        // most likely just means the process (or its whole group) was already gone.
        Ok(_) => {}
        Err(e) => {
            // If `kill` itself couldn't run (missing binary, restrictive sandboxing), a backgrounded
            // grandchild from the timed-out command may be left running with nothing else to reap
            // it — surface that instead of silently losing the signal.
            tracing::warn!(pgid, error = %e, "failed to run `kill` to reap a timed-out process");
        }
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
    // write. `kill`'s own success just means the signal was handed to the kernel for delivery to every
    // process in the group — not that a *specific* member (the grandchild, which we never directly
    // `wait()` on) has actually been scheduled, signaled, and reaped yet. Confirmed live on a
    // resource-contended CI runner: the group leader died promptly (reparenting the grandchild to PID
    // 1), but the grandchild itself was still alive at the very next instant, apparently stuck mid
    // fork/exec — under I/O contention that can plausibly stretch well past a scant few hundred
    // milliseconds. `GRANDCHILD_DELAY`/`SAFETY_WAIT` below are sized with a wide margin for that, not
    // tuned to a fast, uncontended dev box.
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
        assert_eq!(res.stdout, "hello\nworld\n");
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
