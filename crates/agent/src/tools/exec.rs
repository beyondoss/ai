//! A command-execution seam.
//!
//! `bash` runs real commands through [`RealRunner`]; the Beyond tools (fork/sync/logs, M8) run the
//! `beyond` CLI through the same trait, so their tests inject a [`CommandRunner`] that records the
//! argv instead of shelling out. This is the boundary that keeps shell-driven tools testable.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncReadExt};

/// Per-stream capture caps. We keep the first [`STREAM_HEAD`] and last [`STREAM_TAIL`] bytes of each
/// of stdout/stderr and discard the middle *as it streams*, so a command that emits gigabytes (e.g.
/// `yes`, `cat huge.bin`) holds at most ~`HEAD+TAIL` per stream in memory instead of the whole
/// firehose. Sized well above `bash`'s own 30 KB output cap so any capture-level drop still leaves
/// `bash` plenty to head/tail-truncate (with its own marker) on top.
const STREAM_HEAD: usize = 128 * 1024;
const STREAM_TAIL: usize = 128 * 1024;

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
        let mut child = cmd.spawn()?;
        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // Drain both pipes *concurrently* with the wait: a child that fills one pipe's OS buffer
        // while we read only the other would deadlock, and an unread pipe stalls the child's exit.
        let collect = async {
            let (status, out, err) = tokio::join!(
                child.wait(),
                drain_capped(stdout, STREAM_HEAD, STREAM_TAIL, on_chunk),
                drain_capped(stderr, STREAM_HEAD, STREAM_TAIL, on_chunk),
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
            Ok((status, (stdout, out_trunc), (stderr, err_trunc))) => Ok(ExecResult {
                code: status?.code(),
                stdout: String::from_utf8_lossy(&stdout).into_owned(),
                stderr: String::from_utf8_lossy(&stderr).into_owned(),
                timed_out: false,
                truncated: out_trunc || err_trunc,
            }),
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
                    ..Default::default()
                })
            }
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

/// Read a child pipe to EOF, keeping only the first `head_cap` and last `tail_cap` bytes; the middle
/// is discarded as it arrives so memory stays bounded regardless of how much the command emits.
/// Returns the kept bytes (head followed by tail) and whether any middle was dropped.
async fn drain_capped<R: AsyncRead + Unpin>(
    reader: Option<R>,
    head_cap: usize,
    tail_cap: usize,
    on_chunk: Option<ChunkSink<'_>>,
) -> (Vec<u8>, bool) {
    let mut cap = Capture::new(head_cap, tail_cap);
    if let Some(mut r) = reader {
        let mut buf = [0u8; 64 * 1024];
        loop {
            match r.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    // Stream the chunk out for live progress *before* the cap drops any of it.
                    if let Some(sink) = on_chunk {
                        sink(&buf[..n]);
                    }
                    cap.push(&buf[..n]);
                }
                // A read error (e.g. the pipe closing under us) ends capture with what we have.
                Err(_) => break,
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

/// SIGKILL an entire process group via the `kill` binary (`kill -KILL -<pgid>`). Shelling out keeps
/// this free of `unsafe`/`libc`, which the workspace forbids.
#[cfg(unix)]
async fn kill_process_group(pgid: u32) {
    match tokio::process::Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{pgid}"))
        .status()
        .await
    {
        // `kill`'s own exit code (e.g. ESRCH because the group already exited on its own between
        // the timeout firing and this running) isn't worth logging — only a failure to even run it.
        Ok(_) => {}
        Err(e) => {
            // If `kill` itself couldn't run (missing binary, restrictive sandboxing), a backgrounded
            // grandchild from the timed-out command may be left running with nothing else to reap
            // it — surface that instead of silently losing the signal.
            tracing::warn!(pgid, error = %e, "failed to run `kill` to reap a timed-out process group");
        }
    }
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
