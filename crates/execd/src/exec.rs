//! Running a command: one-shot (one request, one JSON answer) and streaming (the connection is the
//! command's lifetime, output streamed as frames).
//!
//! **Bounds, per command.**
//!
//! - one-shot: the captured output, at most [`crate::Options::max_response_bytes`] (16 MiB, the agent's
//!   own cap) across both pipes. Output past it is read and dropped so the command never blocks on
//!   a full pipe, and the exchange is then answered `422` saying the command ran and its output was
//!   too large, never with the prefix as if it were the output (the agent refuses an answer that
//!   big anyway, and a JSON string is never shorter than the bytes it encodes). Freed when the
//!   answer is built. N concurrent cold callers hold up to N times that.
//! - streaming: two pipe readers with one 64 KiB buffer each, a channel of depth 1 between them and the
//!   writer, a channel of depth 1 to hyper, and hyper's own write buffer (400 KiB by default). The
//!   channels are bounded, so a client that stops reading stalls the readers, the pipes fill and the
//!   command blocks on its own write: memory does not grow. That stall ends with the command's
//!   `timeout_ms`, with the client's disconnect, or with the kernel dropping a peer that stopped
//!   acknowledging (see `http`). N concurrent streams hold N times a few hundred KiB.
//! - The process group: killed when the exchange ends any way but a clean finish, by [`GroupSlot`].

use std::os::unix::process::ExitStatusExt as _;
use std::pin::Pin;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use base64::Engine as _;
use bytes::Bytes;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, Sleep};

use crate::frame::{self, READ_BUF};
use crate::http::{self, Line, Request, Resp};
use crate::proc::{self, Proc, locked};
use crate::{Ctx, StreamEnd, StreamStart};

/// How long, after a command's process exits, to keep reading pipes a descendant may be holding
/// open: the client's own grace (streaming "Ordering").
pub(crate) const PIPE_GRACE: Duration = Duration::from_millis(100);

/// How a pipe reader behaves once its command has exited.
///
/// The grace for a pipe a descendant holds open is measured at the pipe, by the reader that reads
/// it: a read that has waited [`PIPE_GRACE`] on an empty pipe. Measured by whoever awaits the
/// reader, the grace also counts every moment the reader was runnable but not running, so a loaded
/// host made it give up on a pipe that still held the command's output. Tokio polls a read before
/// its deadline, so a reader that was starved past the grace still takes what is waiting.
pub(crate) struct AfterExit {
    exited: watch::Receiver<bool>,
    /// What may still be read once the command is gone: all a command can have written and not
    /// had read is its pipe's capacity; a chatty descendant holding the pipe is let go after that.
    left: usize,
}

impl AfterExit {
    pub(crate) fn new(exited: &watch::Sender<bool>, pipe_capacity: usize) -> Self {
        Self {
            exited: exited.subscribe(),
            left: pipe_capacity,
        }
    }

    /// Bytes read into `buf`; `None` when the pipe has closed, or has gone quiet or spent its
    /// allowance since the command exited.
    pub(crate) async fn read<R: tokio::io::AsyncRead + Unpin>(
        &mut self,
        pipe: &mut R,
        buf: &mut [u8],
    ) -> Option<usize> {
        if !*self.exited.borrow() {
            // A closed signal reads as exited too: nobody is left to say the command is running.
            tokio::select! {
                r = pipe.read(buf) => return r.ok().filter(|&n| n > 0),
                _ = self.exited.changed() => {}
            }
        }
        if self.left == 0 {
            return None;
        }
        let n = tokio::time::timeout(PIPE_GRACE, pipe.read(buf))
            .await
            .ok()?
            .ok()
            .filter(|&n| n > 0)?;
        self.left = self.left.saturating_sub(n);
        Some(n)
    }
}

/// The command a request describes, ready to spawn: its `cwd` (default: the root), `HOME`, and
/// piped output. Nothing of execd's own secrets reaches it: every `EXECD_*` variable is removed
/// (the bearer must not be in a command's environment). Under `EXECD_RUN_AS` the switch helper becomes
/// the command, as the target user.
pub(crate) fn command_for(ctx: &Ctx, req: &Request) -> std::process::Command {
    let cwd = req
        .cwd
        .as_deref()
        .map_or_else(|| ctx.opts.root.clone(), Into::into);
    let mut cmd = match &ctx.opts.run_as {
        // The helper changes directory itself, after the switch, so the target's own permissions
        // decide whether it may.
        Some(who) => who.wrap(&req.command, &req.args, &cwd),
        None => {
            let mut cmd = std::process::Command::new(&req.command);
            cmd.args(&req.args).current_dir(&cwd);
            cmd
        }
    };
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    for name in &ctx.scrub {
        cmd.env_remove(name);
    }
    if let Some(home) = &ctx.opts.home {
        cmd.env("HOME", home);
    }
    cmd
}

/// Why `req` could not be started: the exit status and the one line a shell would print, by the
/// shell's convention. 127 is a program that is not there (`execd: pytest -q: command not found`: the
/// caller put a whole command line in `command`, or misspelled it) and a `cwd` that cannot be
/// entered (`execd: cwd DIR: No such file or directory`, `... Not a directory`). 126 is a program that
/// is there and cannot be run (`execd: /: Permission denied`, a script with a bad interpreter,
/// `Argument list too long`). The name is the one the caller sent, because a bare OS error says
/// neither which file nor which directory. The line ends in a newline, like stderr.
pub(crate) fn spawn_failure(ctx: &Ctx, req: &Request, e: &std::io::Error) -> (i32, String) {
    use std::io::ErrorKind::NotFound;
    let cwd = req
        .cwd
        .as_deref()
        .map_or_else(|| ctx.opts.root.clone(), Into::into);
    if !cwd.is_dir()
        && e.raw_os_error().is_some_and(|n| {
            n == nix::errno::Errno::ENOENT as i32 || n == nix::errno::Errno::ENOTDIR as i32
        })
    {
        return (
            127,
            format!("execd: cwd {}: {}\n", cwd.display(), strip_os_code(e)),
        );
    }
    if e.kind() == NotFound {
        return (127, not_found(&req.command));
    }
    (
        126,
        format!("execd: {}: {}\n", req.command, strip_os_code(e)),
    )
}

/// The line for a program that is not there.
pub(crate) fn not_found(command: &str) -> String {
    format!("execd: {command}: command not found\n")
}

/// An I/O error's text without the ` (os error N)` suffix Rust appends.
pub(crate) fn strip_os_code(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.rfind(" (os error ") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

/// Owns a running command's process group: registered for the shutdown sweep, and killed on drop if
/// anything of it is left. A dropped handler (the client went away, the server is shutting down)
/// therefore never leaves its command behind.
pub(crate) struct GroupSlot {
    ctx: Arc<Ctx>,
    pgid: i32,
}

impl GroupSlot {
    pub fn new(ctx: &Arc<Ctx>, pgid: i32) -> Self {
        ctx.groups.insert(pgid);
        Self {
            ctx: ctx.clone(),
            pgid,
        }
    }
}

impl Drop for GroupSlot {
    fn drop(&mut self) {
        self.ctx.groups.remove(self.pgid);
        if proc::group_exists(self.pgid) {
            proc::kill_group(self.pgid);
        }
    }
}

fn exit_code(s: ExitStatus) -> i32 {
    s.code().unwrap_or_else(|| 128 + s.signal().unwrap_or(0))
}

// ---------------------------------------------------------------------------------------------
// one-shot
// ---------------------------------------------------------------------------------------------

/// Take up to `n` bytes from `budget`, once. Two callers (stdout and stderr) can run at once on a
/// multi-thread runtime; a load followed by `fetch_sub` would let both keep the same remainder and,
/// once the counter wraps, keep every later byte. `compare_exchange` publishes one remainder to one
/// caller. `0` means nothing is left.
fn reserve(budget: &AtomicUsize, n: usize) -> usize {
    loop {
        let left = budget.load(Ordering::Relaxed);
        let take = n.min(left);
        if take == 0
            || budget
                .compare_exchange_weak(left, left - take, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            return take;
        }
    }
}

/// Read `pipe` into `into` as it arrives, keeping at most what is left of `budget` across both
/// pipes; the rest is read and dropped.
async fn capture<R: tokio::io::AsyncRead + Unpin>(
    mut pipe: R,
    into: Arc<std::sync::Mutex<Vec<u8>>>,
    budget: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
    mut after: AfterExit,
) {
    let mut buf = vec![0u8; READ_BUF];
    while let Some(n) = after.read(&mut pipe, &mut buf).await {
        let take = reserve(&budget, n);
        if take < n {
            dropped.store(true, Ordering::Release);
        }
        if take > 0 {
            locked(&into).extend_from_slice(&buf[..take]);
        }
    }
}

/// What a one-shot command killed at its deadline is answered with: `timeout(1)`'s status, and a note
/// ending its stderr (the agent's own words for a streaming `timed_out`).
const TIMEOUT_EXIT: i32 = 124;
const TIMEOUT_NOTE: &str = "command exceeded its timeout";

/// An answer: `stdout` and `stderr` are the command's bytes, standard base64, the empty string for
/// none. No lossy conversion, ever.
fn answer(code: i32, stdout: &[u8], stderr: &[u8]) -> Value {
    let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
    json!({ "exit_code": code, "stdout": b64(stdout), "stderr": b64(stderr) })
}

/// one-shot: run to completion, answer with one object.
pub(crate) async fn envelope(ctx: &Arc<Ctx>, req: Request) -> Resp {
    let line = Line::new("exec", &req.command);
    let stdin = match req.stdin.as_deref() {
        None => None,
        Some(b) => match base64::engine::general_purpose::STANDARD.decode(b) {
            Ok(bytes) => Some(bytes),
            Err(_) => {
                line.done(400, "bad stdin");
                return http::error_response(400, "stdin is not base64");
            }
        },
    };
    let mut cmd = command_for(ctx, &req);
    cmd.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mut proc = match proc::spawn(&mut cmd) {
        Ok(p) => p,
        // `Argument list too long` lands here: an ordinary result, as a shell reports it.
        Err(e) => {
            line.done(200, "spawn failed");
            let (code, why) = spawn_failure(ctx, &req, &e);
            return http::json_response(200, &answer(code, b"", why.as_bytes()));
        }
    };
    // Killed on every way out, including this future being dropped when the client leaves (one-shot: a
    // SHOULD once it has answered; a MUST that nothing is left behind an abandoned request).
    let _slot = GroupSlot::new(ctx, proc.pgid);
    let feed = proc.stdin.take().zip(stdin);
    let feeder = tokio::spawn(async move {
        if let Some((mut pipe, bytes)) = feed {
            let _ = pipe.write_all(&bytes).await;
        }
    });
    let budget = Arc::new(AtomicUsize::new(ctx.opts.max_response_bytes));
    let dropped = Arc::new(AtomicBool::new(false));
    let out = Arc::new(std::sync::Mutex::new(Vec::new()));
    let err = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (exited, _) = watch::channel(false);
    let readers = [
        proc.stdout.take().map(|p| {
            let after = AfterExit::new(&exited, proc::pipe_capacity(&p));
            tokio::spawn(capture(
                p,
                out.clone(),
                budget.clone(),
                dropped.clone(),
                after,
            ))
        }),
        proc.stderr.take().map(|p| {
            let after = AfterExit::new(&exited, proc::pipe_capacity(&p));
            tokio::spawn(capture(
                p,
                err.clone(),
                budget.clone(),
                dropped.clone(),
                after,
            ))
        }),
    ];
    // `timeout_ms` is the endpoint's to enforce here too: at the deadline the whole group is killed
    // (the spec, "Timeouts"). Absent, the connection is the only bound.
    let deadline = req.timeout_ms.map(Duration::from_millis);
    let mut timed_out = false;
    let status = match deadline {
        None => proc.wait().await,
        Some(limit) => match tokio::time::timeout(limit, proc.wait()).await {
            Ok(status) => status,
            Err(_) => {
                timed_out = true;
                proc::kill_group(proc.pgid);
                proc.wait().await
            }
        },
    };
    // The command is done. The readers take what its pipes hold, and give a pipe a descendant
    // holds open the grace before letting it go (`AfterExit`).
    exited.send_replace(true);
    for r in readers.into_iter().flatten() {
        let _ = r.await;
    }
    feeder.abort();
    let take = |m: &Arc<std::sync::Mutex<Vec<u8>>>| std::mem::take(&mut *locked(m));
    if dropped.load(Ordering::Acquire) {
        // Output past the cap was thrown away, so there is no honest result to return: a prefix
        // presented as the output would be a confident wrong answer. 422, not 5xx or 413: the
        // agent reads 400 and 422 as "this request, as written, is declined" and does not rebuild
        // the sandbox target, as it does for any other refusal. The message says the command ran.
        let cap = ctx.opts.max_response_bytes;
        line.done(422, "output over the cap");
        return http::error_response(
            422,
            &format!(
                "the command ran, but its output exceeded the {cap}-byte response cap and was \
                 discarded; narrow the output (for example pipe it through `head`) instead of \
                 running it again unchanged"
            ),
        );
    }
    let payload = match status {
        Ok(_) if timed_out => {
            line.done(200, "timed out");
            let mut stderr = take(&err);
            if !stderr.is_empty() && !stderr.ends_with(b"\n") {
                stderr.push(b'\n');
            }
            stderr.extend_from_slice(TIMEOUT_NOTE.as_bytes());
            answer(TIMEOUT_EXIT, &take(&out), &stderr)
        }
        Ok(s) => {
            line.done(200, "exit");
            answer(exit_code(s), &take(&out), &take(&err))
        }
        Err(e) => {
            line.done(200, "wait failed");
            answer(127, &take(&out), e.to_string().as_bytes())
        }
    };
    http::json_response(200, &payload)
}

// ---------------------------------------------------------------------------------------------
// streaming
// ---------------------------------------------------------------------------------------------

/// Why the exchange stopped short of a final frame.
pub(crate) enum Cut {
    /// The client went away.
    Gone,
    /// `timeout_ms` ran out.
    TimedOut,
}

/// The writing half of a stream: frames to the response body, with the connection's death and the
/// command's deadline watched whenever it waits.
pub(crate) struct Sink {
    out: mpsc::Sender<Bytes>,
    /// When a frame last went out; the heartbeat counts from it.
    last: Instant,
    heartbeat: Duration,
    /// The command's deadline. A command with no representable deadline waits on a far future, as
    /// tokio's own `Sleep::far_future` does (30 years).
    timer: Pin<Box<Sleep>>,
}

impl Sink {
    pub fn new(out: mpsc::Sender<Bytes>, heartbeat: Duration, timeout: Option<Duration>) -> Self {
        let now = Instant::now();
        let at = timeout
            .and_then(|t| now.checked_add(t))
            .unwrap_or_else(|| now + Duration::from_secs(86400 * 365 * 30));
        Self {
            out,
            last: now,
            heartbeat,
            timer: Box::pin(tokio::time::sleep_until(at)),
        }
    }

    /// Send a frame, waiting for room: the backpressure point. Gives up if the client leaves or the
    /// deadline passes while it waits.
    pub async fn put(&mut self, b: Bytes) -> Result<(), Cut> {
        tokio::select! {
            biased;
            () = &mut self.timer => Err(Cut::TimedOut),
            () = self.out.closed() => Err(Cut::Gone),
            r = self.out.send(b) => {
                self.last = Instant::now();
                r.map_err(|_| Cut::Gone)
            }
        }
    }

    /// As [`Self::put`], ignoring the deadline: the final frame is owed after one.
    pub async fn put_final(&mut self, b: Bytes) -> Result<(), Cut> {
        tokio::select! {
            biased;
            () = self.out.closed() => Err(Cut::Gone),
            r = self.out.send(b) => r.map_err(|_| Cut::Gone),
        }
    }

    /// Resolves with `Ok` when a heartbeat is due, or `Err` when the stream is cut.
    pub async fn quiet(&mut self) -> Result<(), Cut> {
        let due = self.last + self.heartbeat;
        tokio::select! {
            biased;
            () = &mut self.timer => Err(Cut::TimedOut),
            () = self.out.closed() => Err(Cut::Gone),
            () = tokio::time::sleep_until(due) => Ok(()),
        }
    }
}

/// Read `pipe` into `key` frames of at most [`FRAME_MAX`] bytes as they arrive. A full channel
/// stalls the read, which stalls the command: the backpressure from a slow client.
fn pump<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    mut pipe: R,
    key: &'static str,
    tx: mpsc::Sender<Bytes>,
    mut after: AfterExit,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut buf = vec![0u8; READ_BUF];
        while let Some(n) = after.read(&mut pipe, &mut buf).await {
            if tx.send(frame::output(key, &buf[..n])).await.is_err() {
                return;
            }
        }
    })
}

/// Forward what the pipes have produced until they close, go quiet for [`PIPE_GRACE`], or
/// `budget` more bytes have gone by. The budget is what a chatty descendant holding a pipe open
/// could otherwise keep this going with forever; callers pass the pipes' own capacity, which is
/// all a command can have written before it exited and not been read.
pub(crate) async fn settle(
    rx: &mut mpsc::Receiver<Bytes>,
    sink: &mut Sink,
    mut budget: usize,
) -> Result<(), Cut> {
    loop {
        match tokio::time::timeout(PIPE_GRACE, rx.recv()).await {
            Ok(Some(f)) => {
                budget = budget.saturating_sub(f.len());
                sink.put(f).await?;
                if budget == 0 {
                    return Ok(());
                }
            }
            _ => return Ok(()),
        }
    }
}

fn final_frame(s: ExitStatus) -> Bytes {
    match (s.code(), s.signal()) {
        (Some(code), _) => frame::json(&json!({ "exit_code": code })),
        (None, Some(sig)) => frame::json(&json!({ "signal": sig })),
        (None, None) => frame::json(&json!({ "exit_code": -1 })),
    }
}

/// streaming: frames as the command produces them; the process group dies with the connection.
pub(crate) fn stream(ctx: &Arc<Ctx>, req: Request) -> Resp {
    let (resp, out) = http::stream_response();
    tokio::spawn(run_stream(ctx.clone(), req, out));
    resp
}

async fn run_stream(ctx: Arc<Ctx>, req: Request, out: mpsc::Sender<Bytes>) {
    let line = Line::new("stream", &req.command);
    let mut sink = Sink::new(
        out,
        ctx.opts.heartbeat,
        req.timeout_ms.map(Duration::from_millis),
    );
    let mut cmd = command_for(&ctx, &req);
    cmd.stdin(Stdio::null());
    let mut proc = match proc::spawn(&mut cmd) {
        Ok(p) => p,
        Err(e) => {
            // A command that cannot be spawned is an ordinary result, as a shell reports it.
            let (code, why) = spawn_failure(&ctx, &req, &e);
            let _ = sink
                .put_final(frame::output("stderr", why.as_bytes()))
                .await;
            let _ = sink
                .put_final(frame::json(&json!({ "exit_code": code })))
                .await;
            line.done(200, "spawn failed");
            return;
        }
    };
    let pgid = proc.pgid;
    let slot = GroupSlot::new(&ctx, pgid);
    let seq = ctx.next_seq.fetch_add(1, Ordering::Relaxed);
    if let Some(o) = &ctx.opts.observer {
        o.stream_started(&StreamStart {
            seq,
            pgid: u32::try_from(pgid).unwrap_or(0),
            command: req.command.clone(),
            args: req.args.clone(),
            at: std::time::Instant::now(),
        });
    }

    let (tx, mut rx) = mpsc::channel::<Bytes>(1);
    let (exited, _) = watch::channel(false);
    let mut cap = 0;
    let mut readers = Vec::new();
    if let Some(p) = proc.stdout.take() {
        let n = proc::pipe_capacity(&p);
        cap += n;
        readers.push(pump(p, "stdout", tx.clone(), AfterExit::new(&exited, n)));
    }
    if let Some(p) = proc.stderr.take() {
        let n = proc::pipe_capacity(&p);
        cap += n;
        readers.push(pump(p, "stderr", tx.clone(), AfterExit::new(&exited, n)));
    }
    drop(tx);

    let ran = drive(&mut proc, &mut rx, &mut sink, cap).await;
    let end = match ran {
        Ok(status) => {
            // Everything the pipes still hold, then the final frame. The pumps end when their pipe
            // closes; a descendant that outlived the group and keeps one open is let go after the
            // grace (`AfterExit`), which closes the channel.
            exited.send_replace(true);
            let mut result = Ok(());
            while let Some(f) = rx.recv().await {
                if let Err(e) = sink.put(f).await {
                    result = Err(e);
                    break;
                }
            }
            match result {
                Ok(()) => match sink.put_final(final_frame(status)).await {
                    Ok(()) => StreamEnd::Finished,
                    Err(_) => StreamEnd::Disconnected { killed: true },
                },
                Err(Cut::TimedOut) => timed_out(&mut proc, &mut sink, &mut readers).await,
                Err(Cut::Gone) => StreamEnd::Disconnected { killed: true },
            }
        }
        Err(Cut::TimedOut) => timed_out(&mut proc, &mut sink, &mut readers).await,
        Err(Cut::Gone) => StreamEnd::Disconnected { killed: true },
    };
    for r in &readers {
        r.abort();
    }
    // Whatever the end, nothing of the command outlives its exchange. A clean finish left the group
    // empty already, and a dropped slot kills only a group that still exists.
    if !matches!(end, StreamEnd::Finished) {
        proc::kill_group(pgid);
        let _ = proc.wait().await;
    }
    drop(slot);
    if let Some(o) = &ctx.opts.observer {
        o.stream_ended(seq, end, std::time::Instant::now());
    }
    line.done(
        200,
        match end {
            StreamEnd::Finished => "finished",
            StreamEnd::TimedOut => "timed out",
            StreamEnd::Disconnected { .. } => "client left",
        },
    );
}

/// The deadline passed: kill the group, say so.
async fn timed_out(
    proc: &mut Proc,
    sink: &mut Sink,
    readers: &mut [tokio::task::JoinHandle<()>],
) -> StreamEnd {
    proc::kill_group(proc.pgid);
    let _ = proc.wait().await;
    for r in readers.iter() {
        r.abort();
    }
    let _ = sink
        .put_final(frame::json(&json!({ "timed_out": true })))
        .await;
    StreamEnd::TimedOut
}

/// Run the command to its end: forward its output and heartbeats, and when the shell exits leaving
/// processes behind, say so (`shell_exit`) and wait for the group to empty. The shell's own status
/// is the result; the caller drains the pipes and sends the final frame.
async fn drive(
    proc: &mut Proc,
    rx: &mut mpsc::Receiver<Bytes>,
    sink: &mut Sink,
    cap: usize,
) -> Result<ExitStatus, Cut> {
    let mut readers_done = false;
    let status = loop {
        tokio::select! {
            biased;
            r = sink.quiet() => {
                r?;
                sink.put(frame::heartbeat()).await?;
            }
            f = rx.recv(), if !readers_done => match f {
                Some(f) => sink.put(f).await?,
                None => readers_done = true,
            },
            st = proc.wait() => match st {
                Ok(s) => break s,
                // The child cannot be waited for: nothing more can be said about it.
                Err(_) => return Err(Cut::Gone),
            },
        }
    };
    let pgid = proc.pgid;
    let mut exits = proc::ChildEvents::new();
    if proc::group_gone(pgid, false).await {
        return Ok(status);
    }
    // The shell exited and its group lives on. What it printed goes first (the pipes are read
    // until quiet), then `shell_exit`, and the stream stays open until the group is empty.
    settle(rx, sink, cap).await?;
    sink.put(frame::json(&json!({ "shell_exit": exit_code(status) })))
        .await?;
    loop {
        tokio::select! {
            biased;
            r = sink.quiet() => {
                r?;
                sink.put(frame::heartbeat()).await?;
            }
            f = rx.recv(), if !readers_done => match f {
                Some(f) => sink.put(f).await?,
                None => readers_done = true,
            },
            wake = exits.next() => {
                if proc::group_gone(pgid, wake == proc::Wake::Tick).await {
                    return Ok(status);
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{AfterExit, PIPE_GRACE, pump, reserve};
    use tokio::io::AsyncWriteExt as _;
    use tokio::sync::watch;

    #[test]
    fn reserve_stops_at_what_is_left() {
        let budget = AtomicUsize::new(10);
        assert_eq!(reserve(&budget, 100), 10);
        assert_eq!(reserve(&budget, 1), 0);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    /// Several threads drawing at once must share one counter: the takes add up to the budget and
    /// never past it. A load followed by `fetch_sub` would let two of them keep the same remainder.
    #[test]
    fn two_threads_cannot_reserve_the_same_byte() {
        let budget = Arc::new(AtomicUsize::new(1_000_003));
        let mut threads = Vec::new();
        for size in [1usize, 7, 64, 4096] {
            for _ in 0..4 {
                let budget = Arc::clone(&budget);
                threads.push(std::thread::spawn(move || {
                    let mut got = 0usize;
                    loop {
                        let n = reserve(&budget, size);
                        if n == 0 {
                            break;
                        }
                        got += n;
                    }
                    got
                }));
            }
        }
        let sum: usize = threads.into_iter().map(|t| t.join().unwrap()).sum();
        assert_eq!(sum, 1_000_003);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    /// A reader the scheduler did not run for longer than the grace still takes what its pipe
    /// holds: the grace is a wait on an empty pipe, not time since the command exited.
    #[tokio::test]
    async fn a_starved_reader_still_takes_what_the_pipe_holds() {
        let (exited, _) = watch::channel(false);
        let mut after = AfterExit::new(&exited, 64 * 1024);
        let (mut w, mut r) = tokio::io::duplex(64 * 1024);
        w.write_all(&[7u8; 40_000]).await.unwrap();
        exited.send_replace(true);
        // The runtime is busy elsewhere for longer than the grace.
        std::thread::sleep(PIPE_GRACE * 3);
        let mut buf = [0u8; 16 * 1024];
        let mut got = 0;
        while got < 40_000 {
            got += after.read(&mut r, &mut buf).await.unwrap();
        }
        assert_eq!(got, 40_000);
    }

    /// An empty pipe a descendant holds open is let go after the grace, and a chatty one after
    /// its allowance.
    #[tokio::test]
    async fn an_idle_or_chatty_pipe_is_let_go_after_exit() {
        let (exited, _) = watch::channel(true);
        let (_w, mut r) = tokio::io::duplex(1024);
        let mut buf = [0u8; 16];
        assert_eq!(
            AfterExit::new(&exited, 1024).read(&mut r, &mut buf).await,
            None
        );

        let (mut w, mut r) = tokio::io::duplex(1024);
        w.write_all(&[1u8; 64]).await.unwrap();
        let mut after = AfterExit::new(&exited, 32);
        assert_eq!(after.read(&mut r, &mut buf).await, Some(16));
        assert_eq!(after.read(&mut r, &mut buf).await, Some(16));
        assert_eq!(after.read(&mut r, &mut buf).await, None);
    }

    /// Both workers are busy for longer than the grace while a pump has its command's output
    /// waiting in the pipe. When they come free the exchange must still end with every byte.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pump_that_runs_late_still_delivers_the_whole_pipe() {
        let hog = || tokio::spawn(async { std::thread::sleep(PIPE_GRACE * 4) });
        let (a, b) = (hog(), hog());
        std::thread::sleep(PIPE_GRACE / 2);
        let (mut w, r) = tokio::io::duplex(64 * 1024);
        w.write_all(&[7u8; 40_000]).await.unwrap();
        drop(w);
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let (exited, _) = watch::channel(false);
        let _pump = pump(r, "stdout", tx, AfterExit::new(&exited, 64 * 1024));
        exited.send_replace(true);
        let mut got = 0;
        while let Some(f) = rx.recv().await {
            got += f.len();
        }
        assert!(got >= 40_000, "only {got} wire bytes of 40000 arrived");
        let _ = tokio::join!(a, b);
    }
}
