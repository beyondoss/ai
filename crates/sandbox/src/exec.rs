//! A command-execution seam.
//!
//! Layer 1: a command-execution seam. [`CommandRunner`] (re-exported as `Exec`) is everything the
//! helpers above it need; an exec endpoint ([`crate::HttpExec`]), a `docker exec`/`ssh` template
//! ([`crate::TemplateRunner`]), the host's own processes or a test double are just implementations.

use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::watch;

/// Per-stream capture caps. We keep the first [`STREAM_HEAD`] and last [`STREAM_TAIL`] bytes of each
/// of stdout/stderr and discard the middle *as it streams*, so a command that emits gigabytes (e.g.
/// `yes`, `cat huge.bin`) holds at most ~`HEAD+TAIL` per stream in memory instead of the whole
/// firehose. Sized well above `bash`'s own 30 KB output cap so any capture-level drop still leaves
/// `bash` plenty to head/tail-truncate (with its own marker) on top.
pub const STREAM_HEAD: usize = 128 * 1024;
pub const STREAM_TAIL: usize = 128 * 1024;

/// How long a pipe may sit silent, once the direct child has exited, before we stop waiting on it.
/// Without this, a detached grandchild that inherits the stdout/stderr fd (e.g. `cmd &` backgrounding
/// something before the shell itself exits) keeps the pipe open with no EOF, and the read loop blocks
/// on it until the *outer command timeout* finally fires and reaps the process group. Re-armed on every chunk read after exit, so an actively
/// writing descendant (e.g. a backgrounded `lint-staged` still flushing output) is never cut off early
/// — only a genuinely quiet held-open handle releases us. Matches pi's `waitForChildProcess` fix for
/// earendil-works/pi#5303 (`EXIT_STDIO_GRACE_MS = 100`).
const POST_EXIT_IDLE_GRACE: Duration = Duration::from_millis(100);

/// The hard deadline for a command run through [`CommandRunner::run_streaming_tracked`] by a runner
/// that cannot track it (the default implementation; an exec endpoint on protocol v1.1). Such a runner
/// streams nothing and reports no [`Kill`], so nothing else could ever stop a command that hangs, and
/// the call waiting on it would wait forever. Not model-facing: `bash`'s `timeout_ms` only decides
/// when a call returns. 30 minutes is well above the 3 min 44 s a clean release build of this repo
/// takes. See "`bash` timeouts" in `ARCHITECTURE.md`.
pub const UNTRACKED_DEADLINE: Duration = Duration::from_secs(30 * 60);

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

/// What a one-shot call returns when the caller wants the command's output as it was: **bytes**.
///
/// stdout and stderr are the bytes the command wrote, not text. Whether they are *all* of those
/// bytes, unaltered, is [`exact`](Self::exact): `true` when the target delivered them losslessly (an
/// endpoint that honours `output_base64`, a v1.2 stream's base64 frames); `false` only when the
/// target can deliver text and nothing else (a plain v1.1 endpoint answers JSON strings, and an
/// invalid UTF-8 sequence in them was already replaced with U+FFFD before this client saw it). A
/// client that did not get the bytes says so instead of passing text off as them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecOutput {
    /// The exit code. As over v1.1: `128 + N` for a command killed by signal `N`, `124` for one that
    /// hit its timeout (the stderr then ends `command exceeded its timeout`), `127` for one that
    /// could not be started. None of these is an error.
    pub code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// Whether `stdout` and `stderr` are exactly what the command wrote.
    pub exact: bool,
}

impl ExecOutput {
    /// The stdout as text, invalid UTF-8 replaced with U+FFFD. Lossy by construction: use
    /// [`stdout`](Self::stdout) when the bytes matter.
    pub fn stdout_text(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.stdout)
    }

    /// The stderr as text, as [`stdout_text`](Self::stdout_text).
    pub fn stderr_text(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.stderr)
    }

    /// A text result as bytes, which is never exact: whatever made it text may have changed them.
    pub fn from_text(r: ExecResult) -> Self {
        Self {
            code: r
                .code
                .or_else(|| r.signal.map(|s| 128 + s))
                .or(r.timed_out.then_some(124))
                .unwrap_or(-1),
            stdout: r.stdout.into_bytes(),
            stderr: r.stderr.into_bytes(),
            exact: false,
        }
    }
}

/// A sink for output chunks as they stream from a running command — used to surface live progress.
/// `Send + Sync` because stdout and stderr are drained on separate concurrent tasks.
pub type ChunkSink<'a> = &'a (dyn Fn(&[u8]) + Send + Sync);

/// Told how to stop the command once it has started and its output is streaming live — see
/// [`CommandRunner::run_streaming_tracked`].
pub type SpawnSink<'a> = &'a (dyn Fn(Kill) + Send + Sync);

/// How long a stopped job's local processes get to exit on SIGTERM before the SIGKILL sweep.
/// SIGTERM first because SIGKILL cannot be relayed: `sudo cmd` forwards a TERM to its root child
/// (gone 40 ms after the TERM on the dev box, `sudo -n sleep`), but a KILL takes `sudo` down alone
/// and leaves the child — which we may not signal — running. 300 ms is ~7x that, and is only spent
/// in full on a process that ignores or traps TERM; a job whose processes all exit returns as soon
/// as they have.
pub const TERM_GRACE: Duration = Duration::from_millis(300);

/// How often the grace re-checks whether the jobs' processes have exited.
const GRACE_POLL: Duration = Duration::from_millis(10);

/// A set of processes on *this* host that [`kill_all`] can stop: the host process runner's job
/// (its process group plus every process carrying its tag). Implemented by the embedding program,
/// not by this crate, which only drives the three steps so several jobs share one SIGTERM grace.
pub trait LocalKill: Send + Sync + 'static {
    /// SIGTERM every process, without waiting.
    fn terminate(&self);
    /// Whether every process has exited.
    fn gone(&self) -> bool;
    /// The SIGKILL sweep. A description of what still ran after it (processes we may not signal,
    /// e.g. the root child of a `sudo`), or `None`.
    fn sweep(&self) -> Option<String>;
    /// For the implementor to get its own type back from a [`Kill::Local`].
    fn as_any(&self) -> &dyn std::any::Any;
}

/// How to stop a command that a tracked run started, wherever it runs.
///
/// One shape for every place a command can live, so the code that keeps commands running past their
/// call (`bash`'s jobs) kills a local job, a remote stream and a command inside a container the same
/// way, from the same three places: a `bash_job` kill, a dropped job, and the exit path.
#[derive(Clone)]
pub enum Kill {
    /// A job's processes on this host, stopped by the embedding program's [`LocalKill`].
    Local(std::sync::Arc<dyn LocalKill>),
    /// The runner's own way to stop the command: close the exec stream (protocol v1.2), run a kill
    /// inside the container a template transport reaches, or kill a sandbox-owned job
    /// ([`crate::exec_endpoint::jobs`]). Blocking, and returns once the runner has done everything
    /// it can — see [`Kill::kill`] — with a description of what survived, when it can tell.
    Hook(std::sync::Arc<dyn Fn() -> Option<String> + Send + Sync>),
}

impl Kill {
    /// A [`Kill::Local`] for `local`.
    pub fn local(local: impl LocalKill) -> Self {
        Kill::Local(std::sync::Arc::new(local))
    }

    /// The local job's own kill handle, when this is one and it is a `T`.
    pub fn as_local<T: LocalKill>(&self) -> Option<&T> {
        match self {
            Kill::Local(l) => l.as_any().downcast_ref(),
            Kill::Hook(_) => None,
        }
    }

    /// Stop the command — see [`kill_all`], which this is for one. **Blocking**: called from a
    /// dedicated OS thread (a dropped job, the exit path) or `spawn_blocking` (a `bash_job` kill),
    /// never on a runtime worker, so a hook may wait for its runner to confirm. A description of
    /// any process that survived it.
    pub fn kill(&self) -> Option<String> {
        kill_all(std::slice::from_ref(self)).pop().flatten()
    }
}

/// Stop several commands at once, spending one SIGTERM grace for all of them: SIGTERM every local
/// job's processes, run the remote hooks (which overlap the grace), wait out what is left of
/// [`TERM_GRACE`] for the local processes to exit, then SIGKILL-sweep them. Per command, in order:
/// a description of the processes still running after it — ones we may not signal, e.g. the root
/// child of a `sudo` — or `None`. Blocking, as [`Kill::kill`].
pub fn kill_all(kills: &[Kill]) -> Vec<Option<String>> {
    use std::time::Instant;
    let deadline = Instant::now() + TERM_GRACE;
    let locals: Vec<&dyn LocalKill> = kills
        .iter()
        .filter_map(|k| match k {
            Kill::Local(l) => Some(&**l),
            Kill::Hook(_) => None,
        })
        .collect();
    for local in &locals {
        local.terminate();
    }
    // Side by side: each may be a round trip with a grace of its own (a sandbox job's kill), and N
    // of them must cost one, not N.
    let hooked: Vec<Option<String>> = std::thread::scope(|scope| {
        let running: Vec<_> = kills
            .iter()
            .map(|k| match k {
                Kill::Hook(hook) => Some(scope.spawn(|| hook())),
                Kill::Local(_) => None,
            })
            .collect();
        running
            .into_iter()
            .map(|h| h.and_then(|h| h.join().ok().flatten()))
            .collect()
    });
    while locals.iter().any(|l| !l.gone()) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        std::thread::sleep(GRACE_POLL.min(deadline - now));
    }
    kills
        .iter()
        .zip(hooked)
        .map(|(k, hooked)| match k {
            Kill::Local(l) => l.sweep(),
            Kill::Hook(_) => hooked,
        })
        .collect()
}

/// What a caller that keeps a command running past its own return (`bash`'s jobs) hears from
/// [`CommandRunner::run_streaming_tracked`].
pub struct JobHooks<'a> {
    /// Every chunk of stdout/stderr, as for [`CommandRunner::run_streaming`].
    pub on_chunk: ChunkSink<'a>,
    /// How to stop the command, the moment it starts — see
    /// [`CommandRunner::run_streaming_tracked`] for the promise calling it makes.
    pub on_spawn: SpawnSink<'a>,
    /// The shell exited, but processes it started are still running (`cmd &`, a daemon): the
    /// shell's own result, and a short description of what is left. The call goes on — streaming
    /// their output — until they have all exited or been killed. This host's runner scans for them;
    /// a v1.2 endpoint says so with a `shell_exit` frame, and a template transport's wrapper with a
    /// marker (both for the command's process group). A v1.1 endpoint cannot, and never calls it.
    pub on_orphans: &'a (dyn Fn(&ExecResult, String) + Send + Sync),
}

/// Runs an external command. Implemented by the host process runner (production) and by test doubles that
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

    /// Like [`run`](CommandRunner::run) (or [`run_with_stdin`](CommandRunner::run_with_stdin), when
    /// `stdin` is given), but the output comes back as the bytes the command wrote: see
    /// [`ExecOutput`]. This is the call for raw exec; a helper that parses text uses `run`.
    ///
    /// Defaults to `run` and a result that says it is not [`exact`](ExecOutput::exact): a runner whose
    /// `run` returns text cannot know that the text is all the command wrote. A runner that can hand
    /// over bytes overrides this, and the HTTP runner does (`output_base64`, else a v1.2 stream).
    async fn run_exact(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: Option<&[u8]>,
    ) -> std::io::Result<ExecOutput> {
        let r = match stdin {
            Some(stdin) => {
                self.run_with_stdin(program, args, cwd, timeout, stdin)
                    .await?
            }
            None => self.run(program, args, cwd, timeout).await?,
        };
        Ok(ExecOutput::from_text(r))
    }

    /// Like [`run`](CommandRunner::run), but invokes `on_chunk` with each chunk of stdout/stderr as it
    /// arrives, for live progress. Defaults to the non-streaming `run`, which never calls the sink.
    ///
    /// Calling the sink at all is a promise that *every* byte of *both* streams went through it: the
    /// caller (`bash`) takes any chunk as proof the output streamed and never reads
    /// [`ExecResult::stdout`]/[`ExecResult::stderr`]. A runner with nothing to stream incrementally
    /// (a test double, a request/response endpoint) inherits this default rather than replaying part
    /// of its result into the sink.
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

    /// Like [`run_streaming`](CommandRunner::run_streaming), for a caller that keeps the command
    /// running past its own return (`bash`'s jobs): reports to [`JobHooks::on_spawn`], the moment
    /// the command starts, how to stop it ([`Kill`]) — so it can be killed from a `bash_job` kill, a
    /// dropped job, or the exit path where no future is left to drop. A runner that can tell when
    /// the shell exits leaving processes running (this host's, a v1.2 stream, a template transport)
    /// also tells [`JobHooks::on_orphans`], and keeps going until those exit too, so nothing the
    /// command started is left untracked.
    ///
    /// **No deadline.** The command runs until it exits or its [`Kill`] stops it: the caller owns its
    /// lifetime (`bash`'s jobs end with the turn, the run or the session). A remote runner still has
    /// to put *some* deadline on the wire — see `HttpExecRunner`.
    ///
    /// **Calling `on_spawn` is a promise that the output is live**: every byte reaches `on_chunk` as
    /// the command produces it, so silence on the sink means the command is silent. `bash` measures
    /// its idle yield on that promise. Defaults to plain `run_streaming` under
    /// [`UNTRACKED_DEADLINE`], calling neither hook: a runner that cannot see output as it happens (a
    /// v1.1 request/response endpoint) has nothing to measure silence with and nothing to kill, so a
    /// deadline is the only thing that ends a command that hangs.
    async fn run_streaming_tracked(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        hooks: &JobHooks<'_>,
    ) -> std::io::Result<ExecResult> {
        self.run_streaming(program, args, cwd, UNTRACKED_DEADLINE, hooks.on_chunk)
            .await
    }

    /// Whether [`run_streaming_tracked`](CommandRunner::run_streaming_tracked) reports a [`Kill`]
    /// (live output, and a way to stop the command) — what `bash` needs to run a command in the
    /// background. Asked before the command starts, so a runner that cannot is refused without
    /// running anything. Defaults to `false`, matching the default `run_streaming_tracked`.
    async fn tracks_jobs(&self) -> bool {
        false
    }

    /// Run a command with its stdout and stderr streamed to **separate** sinks as they arrive, for
    /// as long as it runs: no deadline, and dropping the future stops it (closes the v1.2 stream,
    /// kills the template transport's far side, kills the local child). The held start of a
    /// sandbox job ([`crate::exec_endpoint::jobs`]) is the one caller: its output is the job's and
    /// its stderr carries the supervisor's markers, so the two must never be merged.
    ///
    /// Defaults to [`std::io::ErrorKind::Unsupported`]; [`Self::attaches`] says whether it works.
    async fn run_attached(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        on_stdout: ChunkSink<'_>,
        on_stderr: ChunkSink<'_>,
    ) -> std::io::Result<ExecResult> {
        let _ = (program, args, cwd, on_stdout, on_stderr);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this command runner cannot stream a command's output as it runs",
        ))
    }

    /// Whether [`Self::run_attached`] works here: the output streams, and dropping the call stops
    /// the command.
    async fn attaches(&self) -> bool {
        false
    }

    /// Make sure the target is reachable before a caller asks something synchronous of it
    /// ([`Self::sandbox_jobs`]): a runner whose target is attached lazily
    /// (the embedding program's lazy attach) connects here, and its error — the probe's —
    /// is the caller's to report. Every other runner is ready already.
    async fn ready(&self) -> std::io::Result<()> {
        Ok(())
    }

    /// The jobs the target this runner reaches can own — [`crate::exec_endpoint::jobs`] — or `None`
    /// when commands run on this host, whose jobs are this process's own.
    fn sandbox_jobs(
        self: std::sync::Arc<Self>,
    ) -> Option<std::sync::Arc<dyn crate::exec_endpoint::jobs::SandboxJobs>> {
        None
    }

    /// Like [`run`](CommandRunner::run), but with `stdin` written to the command's standard input and
    /// then closed.
    ///
    /// For the filesystem backend's internals only — this is how file *content* reaches a remote
    /// target without riding in argv, where Linux caps a single entry at 128 KiB. Model-run commands
    /// never get a stdin: `bash` goes through `run`/`run_streaming`, which keep it closed.
    ///
    /// Defaults to [`std::io::ErrorKind::Unsupported`], so a runner that cannot deliver stdin says so
    /// rather than silently running the command without it. A caller must still not *trust* an `Ok`
    /// here — a remote endpoint that predates the stdin field ignores it and answers normally — which
    /// is why `ShellFs` probes for an echo before it writes this way.
    async fn run_with_stdin(
        &self,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        timeout: Duration,
        stdin: &[u8],
    ) -> std::io::Result<ExecResult> {
        let _ = (program, args, cwd, timeout, stdin);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this command runner cannot feed a command's stdin",
        ))
    }
}

/// Write `bytes` to a child's stdin, then close it (dropping the handle is the EOF).
///
/// Must run *concurrently* with draining the child's output, never before it: a child that writes
/// while it is still reading would fill its stdout pipe and block on it while we block on its stdin.
/// A broken pipe is not an error — it means the child stopped reading, and its exit status is the
/// real story (`ShellFs` also checks the byte count that arrived before it commits a write).
pub async fn feed_stdin(
    pipe: Option<tokio::process::ChildStdin>,
    bytes: Option<&[u8]>,
) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt as _;
    if let (Some(mut pipe), Some(bytes)) = (pipe, bytes) {
        match pipe.write_all(bytes).await {
            Err(e) if e.kind() != std::io::ErrorKind::BrokenPipe => return Err(e),
            _ => {}
        }
    }
    Ok(())
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
pub async fn drain_capped<R: AsyncRead + Unpin>(
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
