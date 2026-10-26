//! Running a command: v1.1 (one request, one JSON answer) and v1.2 (the connection is the
//! command's lifetime, output streamed as frames).
//!
//! **Bounds, per command.**
//!
//! - v1.1: the captured output, at most [`crate::Options::max_response_bytes`] (16 MiB, the agent's
//!   own cap) across both pipes. Output past it is read and dropped so the command never blocks on
//!   a full pipe, and the exchange is then answered `422` saying the command ran and its output was
//!   too large, never with the prefix as if it were the output (the agent refuses an answer that
//!   big anyway, and a JSON string is never shorter than the bytes it encodes). Freed when the
//!   answer is built. N concurrent cold callers hold up to N times that.
//! - v1.2: two pipe readers with one 64 KiB buffer each, a channel of depth 1 between them and the
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
use serde_json::json;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::sync::mpsc;
use tokio::time::{Instant, Sleep};

use crate::frame::{self, FRAME_MAX};
use crate::http::{self, Line, Request, Resp};
use crate::proc::{self, Proc};
use crate::{Ctx, StreamEnd, StreamStart};

/// How long, after a command's process exits, to keep reading pipes a descendant may be holding
/// open: the client's own grace (v1.2 "Ordering").
pub(crate) const PIPE_GRACE: Duration = Duration::from_millis(100);

/// The command a request describes, ready to spawn: its `cwd` (default: the root), `HOME`, and
/// piped output.
pub(crate) fn command_for(ctx: &Ctx, req: &Request) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(&req.command);
    cmd.args(&req.args)
        .current_dir(
            req.cwd
                .as_deref()
                .map_or_else(|| ctx.opts.root.clone(), Into::into),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(home) = &ctx.opts.home {
        cmd.env("HOME", home);
    }
    cmd
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
// v1.1
// ---------------------------------------------------------------------------------------------

/// Read `pipe` into `into` as it arrives, keeping at most what is left of `budget` across both
/// pipes; the rest is read and dropped.
async fn capture<R: tokio::io::AsyncRead + Unpin>(
    mut pipe: R,
    into: Arc<std::sync::Mutex<Vec<u8>>>,
    budget: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
) {
    let mut buf = vec![0u8; FRAME_MAX];
    while let Ok(n) = pipe.read(&mut buf).await {
        if n == 0 {
            break;
        }
        let left = budget.load(Ordering::Relaxed);
        let take = n.min(left);
        if take < n {
            dropped.store(true, Ordering::Relaxed);
        }
        if take > 0 {
            budget.fetch_sub(take, Ordering::Relaxed);
            if let Ok(mut v) = into.lock() {
                v.extend_from_slice(&buf[..take]);
            }
        }
    }
}

/// v1.1: run to completion, answer with one object.
pub(crate) async fn envelope(ctx: &Arc<Ctx>, req: Request) -> Resp {
    let line = Line::new("exec", &req.command);
    let stdin = match req.stdin_base64.as_deref() {
        None => None,
        Some(b) => match base64::engine::general_purpose::STANDARD.decode(b) {
            Ok(bytes) => Some(bytes),
            Err(_) => {
                line.done(400, "bad stdin_base64");
                return http::error_response(400, "stdin_base64 is not base64");
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
            return http::json_response(
                200,
                &json!({ "exit_code": 127, "stdout": "", "stderr": e.to_string() }),
            );
        }
    };
    // Killed on every way out, including this future being dropped when the client leaves (v1.1: a
    // SHOULD once it has answered; a MUST that nothing is left behind an abandoned request).
    let _slot = GroupSlot::new(ctx, proc.pgid);
    let feed = proc.child.stdin.take().zip(stdin);
    let feeder = tokio::spawn(async move {
        if let Some((mut pipe, bytes)) = feed {
            let _ = pipe.write_all(&bytes).await;
        }
    });
    let budget = Arc::new(AtomicUsize::new(ctx.opts.max_response_bytes));
    let dropped = Arc::new(AtomicBool::new(false));
    let out = Arc::new(std::sync::Mutex::new(Vec::new()));
    let err = Arc::new(std::sync::Mutex::new(Vec::new()));
    let readers = [
        proc.child
            .stdout
            .take()
            .map(|p| tokio::spawn(capture(p, out.clone(), budget.clone(), dropped.clone()))),
        proc.child
            .stderr
            .take()
            .map(|p| tokio::spawn(capture(p, err.clone(), budget.clone(), dropped.clone()))),
    ];
    let status = proc.child.wait().await;
    // The command is done. Pipes a descendant holds open are given the grace, then let go.
    let mut abort = Vec::new();
    for r in readers.into_iter().flatten() {
        abort.push(r.abort_handle());
        let _ = tokio::time::timeout(PIPE_GRACE, r).await;
    }
    for a in abort {
        a.abort();
    }
    feeder.abort();
    let take = |m: &Arc<std::sync::Mutex<Vec<u8>>>| {
        m.lock()
            .map(|v| String::from_utf8_lossy(&v).into_owned())
            .unwrap_or_default()
    };
    if dropped.load(Ordering::Relaxed) {
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
        Ok(s) => {
            line.done(200, "exit");
            json!({ "exit_code": exit_code(s), "stdout": take(&out), "stderr": take(&err) })
        }
        Err(e) => {
            line.done(200, "wait failed");
            json!({ "exit_code": 127, "stdout": take(&out), "stderr": e.to_string() })
        }
    };
    http::json_response(200, &payload)
}

// ---------------------------------------------------------------------------------------------
// v1.2
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
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut buf = vec![0u8; FRAME_MAX];
        while let Ok(n) = pipe.read(&mut buf).await {
            if n == 0 || tx.send(frame::output(key, &buf[..n])).await.is_err() {
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

/// v1.2: frames as the command produces them; the process group dies with the connection.
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
            let _ = sink
                .put_final(frame::output("stderr", e.to_string().as_bytes()))
                .await;
            let _ = sink
                .put_final(frame::json(&json!({ "exit_code": 127 })))
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
    let mut cap = 0;
    let mut readers = Vec::new();
    if let Some(p) = proc.child.stdout.take() {
        cap += proc::pipe_capacity(&p);
        readers.push(pump(p, "stdout", tx.clone()));
    }
    if let Some(p) = proc.child.stderr.take() {
        cap += proc::pipe_capacity(&p);
        readers.push(pump(p, "stderr", tx.clone()));
    }
    drop(tx);

    let ran = drive(&mut proc, &mut rx, &mut sink, cap).await;
    let end = match ran {
        Ok(status) => {
            // Everything the pipes still hold, then the final frame; a descendant that outlived
            // the group and keeps a pipe open is let go after the grace.
            let drained = settle(&mut rx, &mut sink, cap).await;
            for r in &readers {
                r.abort();
            }
            let mut result = drained;
            if result.is_ok() {
                while let Ok(f) = rx.try_recv() {
                    if let Err(e) = sink.put(f).await {
                        result = Err(e);
                        break;
                    }
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
        let _ = proc.child.wait().await;
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
    let _ = proc.child.wait().await;
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
            st = proc.child.wait() => match st {
                Ok(s) => break s,
                // The child cannot be waited for: nothing more can be said about it.
                Err(_) => return Err(Cut::Gone),
            },
        }
    };
    let pgid = proc.pgid;
    let mut exits = proc::ChildEvents::new();
    if proc::live_members_async(pgid).await.is_empty() {
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
            () = exits.next() => {
                if proc::live_members_async(pgid).await.is_empty() {
                    return Ok(status);
                }
            }
        }
    }
}
