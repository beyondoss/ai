//! jobs: sandbox jobs, natively. A job is a command that outlives the request that started it: it
//! runs in a process group of its own, its output (stdout and stderr, combined) is kept by byte
//! offset, and it ends when its group is empty. Every job belongs to a *scope*, an opaque name for
//! one session's jobs, and an operation only ever sees the jobs of its own scope.
//!
//! **Bounds.**
//!
//! - *Output*: the last [`crate::Options::job_output_retention`] bytes per job (8 MiB, the spec's
//!   floor), as whole chunk files of 64 KiB under execd's jobs directory (see [`crate::spool`]),
//!   so a job holds between that and that plus one chunk on disk, and in memory only its
//!   unfinished chunk. Older bytes are dropped and read back as a gap. A job's tail writes into it
//!   directly; there is no channel to fill, and a slow `attach` never slows the job (it reads a gap
//!   instead). Between the job and the ring is the job's spool file, not a pipe: it outlives execd.
//! - *The table*: one entry per job the agent has started and not removed. The agent keeps at most
//!   `MAX_YIELDED_JOBS` (4) + `MAX_INTENDED_JOBS` (8) running and `MAX_ENDED_JOBS` (8) ended jobs
//!   per session, and removes them (`remove`, `kill` with `remove`, `wipe`) when the session lets
//!   them go, so a session holds at most 20 entries, 160 MiB of output at the floor. execd does not
//!   invent a second limit: an ended job is freed only by `remove` or `wipe`, and a client that
//!   never sends them keeps its jobs until execd exits. N concurrent cold callers (N sessions)
//!   hold N times that, as the sandbox's own memory allows.
//! - *A read*: answers at most the output held, so `max` cannot ask for more than the retention.
//! - *Streams* (`attach`, a held `start`): the same bounded channel as streaming, one task each, freed
//!   when the stream ends or its client leaves.
//! - *The tail*: one task per job reads its two spool files into the ring (`tail.rs`); it ends with
//!   the job (the group is empty and the files are read to their ends) or when the job is removed.

use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use nix::sys::signal::Signal;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::exec::{self, Cut, Sink};
use crate::frame::{self, FRAME_MAX};
use crate::http::{self, Line, Request, Resp};
use crate::job_kernel::{
    self as kernel, Act, Admit, Alive, Ev, Follow, Gate, Keeper, Kind, Pick, Recovery, Resizable,
    Sink as Dest, Size, Term,
};
use crate::ledger::{Entry, Ledger, Saved};
use crate::notify::{self, Notifier, NotifyTarget};
use crate::proc::{self, locked};
use crate::ring::Ring;
use crate::tail::{Journal, Spool, Start, Tail};
use crate::term;
use crate::{Ctx, StreamEnd, StreamStart};

/// `(scope, id)`: a job's address.
pub(crate) type Key = kernel::Key;

/// One job, shared between its supervisor, its pumps and the requests that name it.
pub(crate) type JobCell = Arc<Mutex<Job>>;

/// One job, as the endpoint keeps it. Its life (phase, who owns it, what was seen, removed) is
/// [`kernel::State`], changed only by [`kernel::step`]; the rest is what the shell holds for it.
pub(crate) struct Job {
    /// What every process of the job carries in its environment ([`proc::JOB_MARK_ENV`]).
    mark: String,
    /// Which spawn its process group is, for a signal by number ([`proc::Pg`]); 0 before the spawn.
    sp: u64,
    label: String,
    out: Ring,
    /// Where its output goes while it runs ([`Spool`]), until it is removed.
    spool: Option<Spool>,
    k: kernel::State,
    /// The latest `keep`'s `notify` (notify), if it carried one.
    pub notify: Option<NotifyTarget>,
    /// When it calls back.
    pub notifier: Notifier,
    /// The job's open stdin, when it was started with `stdin_open`.
    stdin: Option<Arc<JobStdin>>,
    /// Streams open on it now (an `attach`, a held start): a terminal someone watches is attached.
    streams: usize,
}

impl Job {
    /// The job's process group as a signal names it: the kernel's number and the spawn it is.
    fn pg(&self) -> proc::Pg {
        proc::Pg {
            id: self.k.pgid(),
            sp: self.sp,
        }
    }
}

/// `notify` reads the kernel's `removed` as a field of the job.
impl std::ops::Deref for Job {
    type Target = kernel::State;

    fn deref(&self) -> &kernel::State {
        &self.k
    }
}

/// Every job, by `(scope, id)`, and a counter bumped on every change a stream may be waiting for.
pub(crate) struct Jobs {
    map: Mutex<kernel::Table<JobCell>>,
    changed: watch::Sender<u64>,
    /// [`crate::Handle::activity_changed`]'s counter.
    activity: Arc<watch::Sender<u64>>,
    /// The table's jobs, kept across a restart of execd ([`crate::ledger`]).
    ledger: Ledger,
}

impl Jobs {
    pub fn new(activity: Arc<watch::Sender<u64>>, ledger: Ledger) -> Self {
        Self {
            map: Mutex::default(),
            changed: watch::Sender::new(0),
            activity,
            ledger,
        }
    }

    /// Write the table's jobs to the ledger: every job that has a group and is not removed. Called
    /// with no job's lock held, at each change of a job's life (spawn, shell exit, end, keep,
    /// removal); a no-op without [`crate::Options::state_dir`].
    pub fn save(&self) {
        if !self.ledger.enabled() {
            return;
        }
        self.ledger.save(|| {
            let cells = locked(&self.map).all();
            cells
                .into_iter()
                .filter_map(|((scope, id), cell)| {
                    let j = locked(&cell);
                    let keeper = j
                        .stdin
                        .as_ref()
                        .and_then(|s| s.term.as_ref())
                        .map(|t| t.keeper_pid);
                    Entry::of(
                        &scope,
                        &id,
                        &j.label,
                        &j.mark,
                        &j.k,
                        j.spool.as_ref().map(Spool::dir),
                        keeper,
                    )
                })
                .collect()
        });
    }

    /// The server is shutting down and kills every group: nothing is left to recover.
    pub fn forget(&self) {
        self.ledger.save(Vec::new);
    }

    fn bump(&self) {
        self.changed.send_modify(|v| *v = v.wrapping_add(1));
        self.bump_activity();
    }

    fn bump_activity(&self) {
        self.activity.send_modify(|v| *v = v.wrapping_add(1));
    }

    /// Every job in the table that has not ended (held or kept, any scope), and the kept ones of them.
    /// And how many jobs are owed a notify callback that is neither settled nor abandoned.
    pub fn alive(&self) -> (usize, Vec<crate::KeptJob>, usize) {
        let cells = locked(&self.map).all();
        let mut alive = 0;
        let mut owed = 0;
        let mut kept = Vec::new();
        for ((scope, id), cell) in cells {
            let j = locked(&cell);
            // An attempt in flight, a retry waiting out its backoff: owed and not yet settled.
            if !j.removed && j.notifier.owes() {
                owed += 1;
            }
            match j.k.alive() {
                Alive::No => {}
                Alive::Yes => alive += 1,
                Alive::Kept => {
                    alive += 1;
                    kept.push(crate::KeptJob {
                        pgid: j.k.pgid(),
                        scope,
                        id,
                    });
                }
            }
        }
        (alive, kept, owed)
    }

    /// The kept terminals that have not ended and whose keeper holds them: what a host that sleeps
    /// idle machines asks of each (a read, not a decision).
    pub fn terminals(&self) -> Vec<crate::Terminal> {
        let cells = locked(&self.map).all();
        let mut out = Vec::new();
        for ((scope, id), cell) in cells {
            let (pgid, mark, (rdev, fs), attached) = {
                let j = locked(&cell);
                if j.removed
                    || !matches!(j.k.term, Term::Open { .. })
                    || !matches!(j.k.alive(), Alive::Kept)
                {
                    continue;
                }
                let ids = j
                    .stdin
                    .as_ref()
                    .and_then(|s| s.term.as_ref())
                    .map_or((0, 0), |t| (t.pts_rdev, t.pts_fs));
                (j.k.pgid(), j.mark.clone(), ids, j.streams > 0)
            };
            // The same members the kill path signals: the group and every marked process that left it.
            out.push(crate::Terminal {
                scope,
                id,
                pgid,
                pts_rdev: rdev,
                pts_fs: fs,
                attached,
                members: proc::job_members(pgid, &mark),
            });
        }
        out
    }

    fn get(&self, key: &Key) -> Option<JobCell> {
        locked(&self.map).get(key)
    }

    /// The jobs of `scope` that `pick` names.
    fn in_scope(&self, scope: &str, pick: Pick<'_>) -> Vec<(Key, JobCell)> {
        locked(&self.map).in_scope(scope, pick)
    }

    /// Take the `checked` jobs out of the table; their pumps and timers stop. A key that now holds
    /// another job than the one checked keeps it (J10).
    fn remove(&self, checked: &[(Key, JobCell)]) {
        let cells = locked(&self.map).take(checked, Arc::ptr_eq);
        for (_, cell) in cells {
            let mut j = locked(&cell);
            if let Act::Release { .. } = kernel::step(&mut j.k, Ev::Taken) {
                j.notifier = Notifier::removed();
                if let Some(s) = &j.stdin {
                    s.close();
                }
                if let Some(sp) = j.spool.take() {
                    sp.remove();
                }
            }
        }
        self.bump();
        self.save();
    }
}

fn epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn state_of(j: &Job) -> (&'static str, Value) {
    let (state, code) = j.k.named();
    (state.as_str(), code.map_or(Value::Null, |c| json!(c)))
}

/// `signal`, when a signal killed the shell: a member of `read` and `list` answers only once the
/// job has a code.
fn put_signal(j: &Job, obj: &mut Value) {
    if let Some(sig) = j.k.signal() {
        obj["signal"] = json!(sig);
    }
}

/// `pty`, on a terminal job's `read` and `list` entries: its size, and whether its keeper still holds
/// it. Absent for a job with no terminal.
fn put_pty(j: &Job, obj: &mut Value) {
    match j.k.term {
        Term::No => {}
        Term::Open { size, .. } => {
            obj["pty"] = json!({ "cols": size.cols, "rows": size.rows, "open": true });
        }
        Term::Gone { size } => {
            obj["pty"] = json!({ "cols": size.cols, "rows": size.rows, "open": false });
        }
    }
}

/// `SIGTERM` the jobs (each `(pgid, mark)`), wait up to `grace` for them to empty, `SIGKILL` what is
/// left, wait up to a second more (jobs `kill`).
async fn stop_jobs(jobs: &[(proc::Pg, String)], grace: Duration) {
    let jobs = jobs.to_vec();
    proc::off_runtime({
        let jobs = jobs.clone();
        move || {
            for (g, m) in &jobs {
                proc::signal_job(*g, m, Signal::SIGTERM);
            }
        }
    })
    .await;
    let any_live = |jobs: &[(proc::Pg, String)]| {
        let jobs = jobs.to_vec();
        async move {
            proc::off_runtime(move || jobs.iter().any(|(g, m)| !proc::job_empty(g.id, m))).await
        }
    };
    let deadline = tokio::time::Instant::now() + grace;
    while any_live(&jobs).await && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    proc::off_runtime({
        let jobs = jobs.clone();
        move || {
            for (g, m) in &jobs {
                if !proc::job_empty(g.id, m) {
                    proc::signal_job(*g, m, Signal::SIGKILL);
                }
            }
        }
    })
    .await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
    while any_live(&jobs).await && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A stop of execd: end every job that has not ended (`SIGTERM`, `SIGKILL` after `grace`), as a
/// `wipe` of every scope does.
pub(crate) async fn stop_all(ctx: &Arc<Ctx>, grace: Duration) -> usize {
    let cells = locked(&ctx.jobs.map).all();
    let targets: Vec<(proc::Pg, String)> = cells
        .into_iter()
        .filter_map(|(_, c)| {
            let mut j = locked(&c);
            let act = kernel::step(&mut j.k, Ev::Stop);
            (act == Act::Stop).then(|| (j.pg(), j.mark.clone()))
        })
        .collect();
    stop_jobs(&targets, grace).await;
    targets.len()
}

/// Fuzz seam: whether `job` is a well-formed job operation (and its `notify`, if it has one, passes
/// the grammar). Never panics.
pub(crate) fn fuzz_job(job: &Value) -> bool {
    match JobReq::deserialize(job) {
        Ok(j) => {
            j.op != "keep"
                || j.notify
                    .as_ref()
                    .is_none_or(|n| notify::parse_notify(n).is_ok())
        }
        Err(_) => false,
    }
}

/// A job operation: the request's `job` object.
#[derive(Deserialize, Default)]
#[serde(default)]
struct JobReq {
    op: String,
    scope: String,
    id: String,
    hold: bool,
    label: String,
    offset: u64,
    max: u64,
    ids: Vec<String>,
    remove: bool,
    grace_ms: Option<u64>,
    num: Option<u64>,
    kind: Option<String>,
    seen: Option<u64>,
    notify: Option<Value>,
    /// `start`: keep the job's stdin open for `write` (default: `/dev/null`).
    stdin_open: bool,
    /// `signal`: the signal's name.
    signal: Option<String>,
    /// `write`: the stream position of the data's first byte and whether to close
    /// the job's stdin after them.
    at: u64,
    eof: bool,
    /// `start`: a terminal of this size (`{cols, rows}`), the job's stdin, stdout and stderr in one.
    pty: Option<Value>,
    /// `resize`: the terminal's new size.
    cols: Option<u64>,
    rows: Option<u64>,
}

/// The signals `signal` sends: names only, no `SIGSTOP` (it would freeze a sandbox's job beyond the
/// reach of anything but another signal) and no real-time ones.
fn parse_signal(name: &str) -> Option<Signal> {
    Some(match name {
        "SIGHUP" => Signal::SIGHUP,
        "SIGINT" => Signal::SIGINT,
        "SIGQUIT" => Signal::SIGQUIT,
        "SIGUSR1" => Signal::SIGUSR1,
        "SIGUSR2" => Signal::SIGUSR2,
        "SIGTERM" => Signal::SIGTERM,
        "SIGKILL" => Signal::SIGKILL,
        _ => return None,
    })
}

/// The most decoded bytes one `write` carries (symmetric with the client's `READ_MAX`).
pub(crate) const WRITE_MAX: usize = 128 * 1024;
/// The most writes waiting on one job's stdin is `kernel::WAITERS_MAX`: each holds at most `WRITE_MAX`
/// bytes plus its base64, so a job that never reads costs a bounded amount.
/// How long a write waits for earlier bytes (a gap) when the request names no `timeout_ms`.
const GAP_WAIT: Duration = Duration::from_secs(30);

/// A job's open stdin. Writes are positional: `written` counts the bytes of the stream applied so
/// far, a write at `at` skips what is below it and waits for what is missing, so a retry, a replay
/// and a reordering all land the same bytes once, in order.
pub(crate) struct JobStdin {
    /// `None` once closed (EOF written, the job ended or removed). The async mutex is FIFO, so
    /// writes that are in position apply in the order they asked.
    pipe: tokio::sync::Mutex<Option<tokio::process::ChildStdin>>,
    written: watch::Sender<u64>,
    /// Cancelled when the job ends or is removed: pending writes stop, the pipe is dropped.
    dead: CancellationToken,
    waiters: AtomicUsize,
    /// A terminal's input is its keeper's: the keeper holds the master and applies the writes.
    /// `written` then mirrors the keeper's count.
    term: Option<Arc<term::Link>>,
}

impl JobStdin {
    fn new(pipe: tokio::process::ChildStdin) -> Arc<Self> {
        Arc::new(Self {
            pipe: tokio::sync::Mutex::new(Some(pipe)),
            written: watch::Sender::new(0),
            dead: CancellationToken::new(),
            waiters: AtomicUsize::new(0),
            term: None,
        })
    }

    /// A terminal's input, which its keeper has applied `written` bytes of.
    fn terminal(link: Arc<term::Link>, written: u64) -> Arc<Self> {
        Arc::new(Self {
            pipe: tokio::sync::Mutex::new(None),
            written: watch::Sender::new(written),
            dead: CancellationToken::new(),
            waiters: AtomicUsize::new(0),
            term: Some(link),
        })
    }

    /// The job is over: no write lands again and the fd is released once any write in flight
    /// has noticed. A terminal's connection to its keeper goes with it (the keeper ends when its
    /// master reads EIO, which it does once no process of the session is left).
    fn close(self: &Arc<Self>) {
        self.dead.cancel();
        let me = self.clone();
        tokio::spawn(async move {
            me.pipe.lock().await.take();
            if let Some(t) = &me.term {
                t.close().await;
            }
        });
    }
}

/// Read a terminal keeper's frames until it is gone: the count of what landed is mirrored into
/// `stdin.written`, the shell's status goes to `exit` once, and the end of the connection is the
/// terminal hanging up (`Ev::Hangup`). Ends with the job (`stdin.dead`).
async fn term_reader(
    ctx: Arc<Ctx>,
    cell: JobCell,
    stdin: Arc<JobStdin>,
    mut rd: tokio::net::unix::OwnedReadHalf,
    mut frames: term::Frames,
    exit: tokio::sync::oneshot::Sender<Option<(i32, Option<i32>)>>,
) {
    use tokio::io::AsyncReadExt as _;
    let mut exit = Some(exit);
    let mut buf = vec![0u8; 4096];
    let mut alive = true;
    'read: loop {
        loop {
            match frames.next() {
                Ok(Some((tag, p))) => match term::parse_said(tag, &p) {
                    Some(term::Said::Written(n)) | Some(term::Said::Ready { written: n, .. }) => {
                        stdin.written.send_modify(|w| {
                            if n > *w {
                                *w = n;
                            }
                        });
                    }
                    Some(term::Said::Exit { code, signal }) => {
                        if let Some(tx) = exit.take() {
                            let _ = tx.send(Some((code, signal)));
                        }
                    }
                    None => {}
                },
                Ok(None) => break,
                Err(_) => break 'read,
            }
        }
        tokio::select! {
            r = rd.read(&mut buf) => match r {
                Ok(0) | Err(_) => break,
                Ok(n) => frames.push(&buf[..n]),
            },
            () = stdin.dead.cancelled() => {
                alive = false;
                break;
            }
        }
    }
    if let Some(tx) = exit.take() {
        let _ = tx.send(None);
    }
    if alive {
        // The keeper is gone: nothing more can be written or resized.
        let hung = {
            let mut j = locked(&cell);
            kernel::step(&mut j.k, Ev::Hangup);
            !j.k.ended()
        };
        stdin.dead.cancel();
        let _ = hung;
        ctx.jobs.bump();
        ctx.jobs.save();
    }
}

struct Waiter<'a>(&'a AtomicUsize);

impl Drop for Waiter<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// `write`: apply `data` at position `at` of the job's stdin. See the spec's "Job stdin".
async fn write_stdin(ctx: &Arc<Ctx>, req: &Request, job: &JobReq, data: Vec<u8>) -> Resp {
    use tokio::io::AsyncWriteExt as _;
    let Some(cell) = ctx.jobs.get(&(job.scope.clone(), job.id.clone())) else {
        return http::error_response(404, "no such job");
    };
    let (term_state, stdin) = {
        let j = locked(&cell);
        (j.k.term, j.stdin.clone())
    };
    // A terminal has no end of input, and nothing goes to one that hung up (T10 to T12).
    match kernel::write_gate(term_state, job.eof) {
        Gate::NoEnd => {
            return http::error_response(
                400,
                "a terminal has no end of input: write 0x04 (Ctrl-D)",
            );
        }
        Gate::HungUp => {
            let written = stdin.as_ref().map_or(0, |s| *s.written.borrow());
            return hung_up(written);
        }
        Gate::Go => {}
    }
    let Some(stdin) = stdin else {
        return http::error_response(409, "the job has no stdin");
    };
    if !kernel::admit_waiter(stdin.waiters.fetch_add(1, Ordering::Relaxed)) {
        stdin.waiters.fetch_sub(1, Ordering::Relaxed);
        return http::error_response(429, "too many writes waiting on this job");
    }
    let _waiting = Waiter(&stdin.waiters);
    let state = |closed: bool| {
        let mut b = json!({ "written": *stdin.written.borrow() });
        if closed {
            b["closed"] = json!(true);
        }
        b
    };
    let len = data.len() as u64;
    let deadline = req
        .timeout_ms
        .map(|ms| tokio::time::Instant::now() + Duration::from_millis(ms));
    // Earlier bytes first: wait (without holding the pipe) until the stream reaches `at`.
    let mut rx = stdin.written.subscribe();
    let gap_by = deadline.unwrap_or_else(|| tokio::time::Instant::now() + GAP_WAIT);
    while kernel::stdin_plan(
        *rx.borrow_and_update(),
        job.at,
        len,
        job.eof,
        true,
        stdin.dead.is_cancelled(),
    ) == kernel::Plan::Wait
    {
        tokio::select! {
            _ = rx.changed() => {}
            () = stdin.dead.cancelled() => break,
            () = tokio::time::sleep_until(gap_by) => {
                let mut b = state(false);
                b["error"] = json!("gap");
                return http::json_response(409, &b);
            }
        }
    }
    let mut pipe = tokio::select! {
        p = stdin.pipe.lock() => p,
        () = stdin.dead.cancelled() => {
            if stdin.term.is_some() {
                return hung_up(*stdin.written.borrow());
            }
            // Over: bytes already applied are idempotent, anything else is refused.
            let plan = kernel::stdin_plan(*stdin.written.borrow(), job.at, len, job.eof, false, true);
            return match plan {
                kernel::Plan::Done { .. } => http::json_response(200, &state(true)),
                _ => closed(&state(true)),
            };
        }
    };
    if let Some(link) = stdin.term.clone() {
        return write_term(&stdin, &link, job, &data, deadline).await;
    }
    // The kernel decides what lands: the shell does the pipe's I/O it names.
    let plan = kernel::stdin_plan(
        *stdin.written.borrow(),
        job.at,
        len,
        job.eof,
        pipe.is_some(),
        stdin.dead.is_cancelled(),
    );
    let (skip, take, close) = match plan {
        // `Wait` cannot be asked here: the loop above left only with the bytes before `at`
        // applied or the job over.
        kernel::Plan::Wait | kernel::Plan::Refuse => return closed(&state(true)),
        kernel::Plan::Done { close } => {
            if close {
                pipe.take();
            }
            return http::json_response(200, &state(pipe.is_none()));
        }
        kernel::Plan::Apply { skip, take, close } => {
            (usize::try_from(skip).unwrap_or(usize::MAX), take, close)
        }
    };
    let body = &data[skip.min(data.len())..];
    let mut landed = 0u64;
    while kernel::remaining(take, landed) > 0 {
        let Some(p) = pipe.as_mut() else { break };
        let step = async {
            tokio::select! {
                r = p.write(&body[landed as usize..]) => Some(r),
                () = stdin.dead.cancelled() => None,
                () = async {
                    match deadline {
                        Some(d) => tokio::time::sleep_until(d).await,
                        None => std::future::pending().await,
                    }
                } => None,
            }
        };
        match step.await {
            Some(Ok(n)) if n > 0 => {
                landed += n as u64;
                stdin
                    .written
                    .send_modify(|w| *w = kernel::advance(*w, n as u64));
            }
            // The reader is gone (EPIPE) or the pipe broke: nothing more can land.
            Some(_) => {
                pipe.take();
                return closed(&state(true));
            }
            // Cancelled or out of time: a short write, exact `written`.
            None => {
                let c = pipe.is_none() || stdin.dead.is_cancelled();
                return if stdin.dead.is_cancelled() {
                    closed(&state(c))
                } else {
                    http::json_response(200, &state(false))
                };
            }
        }
    }
    if kernel::closes(close, kernel::remaining(take, landed)) {
        pipe.take();
    }
    http::json_response(200, &state(pipe.is_none()))
}

/// `409` for a terminal that hung up, with how much of the stream landed.
fn hung_up(written: u64) -> Resp {
    http::json_response(
        409,
        &json!({ "error": "the terminal hung up", "written": written }),
    )
}

/// A write to a terminal: the keeper applies it (and counts it, so a retry across a restart of
/// execd still lands once); this sends it and waits until the keeper says it landed, the terminal
/// hung up, or the request's time ran out (a short write: the exact `written`). The `pipe` mutex,
/// held by the caller, keeps one write in flight.
async fn write_term(
    stdin: &Arc<JobStdin>,
    link: &Arc<term::Link>,
    job: &JobReq,
    data: &[u8],
    deadline: Option<tokio::time::Instant>,
) -> Resp {
    let len = data.len() as u64;
    let written = || *stdin.written.borrow();
    let plan = kernel::stdin_plan(
        written(),
        job.at,
        len,
        false,
        !stdin.dead.is_cancelled(),
        stdin.dead.is_cancelled(),
    );
    match plan {
        kernel::Plan::Done { .. } => {
            return http::json_response(200, &json!({ "written": written() }));
        }
        kernel::Plan::Wait | kernel::Plan::Refuse => return hung_up(written()),
        kernel::Plan::Apply { .. } => {}
    }
    let mut rx = stdin.written.subscribe();
    if !link.send(&term::write_frame(job.at, data)).await {
        return hung_up(written());
    }
    let target = job.at.saturating_add(len);
    loop {
        if *rx.borrow_and_update() >= target {
            return http::json_response(200, &json!({ "written": written() }));
        }
        tokio::select! {
            r = rx.changed() => {
                if r.is_err() {
                    return hung_up(written());
                }
            }
            () = stdin.dead.cancelled() => {
                return if written() >= target {
                    http::json_response(200, &json!({ "written": written() }))
                } else {
                    hung_up(written())
                };
            }
            () = async {
                match deadline {
                    Some(d) => tokio::time::sleep_until(d).await,
                    None => std::future::pending().await,
                }
            } => {
                return http::json_response(200, &json!({ "written": written() }));
            }
        }
    }
}

fn closed(state: &Value) -> Resp {
    let mut b = state.clone();
    b["error"] = json!("stdin is closed");
    http::json_response(409, &b)
}

/// Answer a request carrying a `job` object.
pub(crate) async fn serve(ctx: &Arc<Ctx>, mut req: Request) -> Resp {
    // A write's bytes are decoded once, here, and the request's base64 text is dropped at once: a
    // write that waits (for the pipe, or for earlier bytes) holds its decoded bytes and nothing of
    // the request that carried them.
    let mut data = Vec::new();
    if let Some(Value::Object(o)) = req.job.as_mut()
        && o.get("op").and_then(Value::as_str) == Some("write")
        && let Some(d) = o.remove("data")
    {
        use base64::Engine as _;
        let Value::String(text) = d else {
            return http::error_response(400, "data is not base64");
        };
        if text.len() > WRITE_MAX.div_ceil(3) * 4 {
            return http::error_response(413, "write too large");
        }
        data = match base64::engine::general_purpose::STANDARD.decode(&text) {
            Ok(d) => d,
            Err(_) => return http::error_response(400, "data is not base64"),
        };
        if data.len() > WRITE_MAX {
            return http::error_response(413, "write too large");
        }
    }
    let req = &req;
    let job: JobReq = match req.job.as_ref().map(JobReq::deserialize) {
        Some(Ok(j)) => j,
        _ => return http::error_response(400, "bad job"),
    };
    let line = Line::new("job", &req.command);
    let resp = serve_op(ctx, req, job, data).await;
    line.done(resp.status().as_u16(), "answered");
    resp
}

async fn serve_op(ctx: &Arc<Ctx>, req: &Request, job: JobReq, data: Vec<u8>) -> Resp {
    let jobs = &ctx.jobs;
    let key = |id: &str| (job.scope.clone(), id.to_string());
    match job.op.as_str() {
        "start" => start_job(ctx, req, &job).await,
        "attach" => {
            let k = key(&job.id);
            let Some(cell) = jobs.get(&k) else {
                return http::error_response(404, "no such job");
            };
            let (resp, out) = http::stream_response();
            tokio::spawn(stream_job(
                ctx.clone(),
                k,
                cell,
                job.offset,
                false,
                None,
                Sink::new(out, req.beat, None),
            ));
            resp
        }
        "read" => {
            let body = match jobs.get(&key(&job.id)) {
                None => json!({ "state": "gone", "offset": 0, "end": 0, "data": "" }),
                Some(cell) => {
                    let j = locked(&cell);
                    let (state, code) = state_of(&j);
                    let (from, data, ranges) = j
                        .out
                        .read(job.offset, usize::try_from(job.max).unwrap_or(usize::MAX));
                    let mut body = json!({
                        "state": state,
                        "code": code,
                        "offset": from,
                        "end": j.out.end(),
                        "data": frame::b64(&data),
                    });
                    if !ranges.is_empty() {
                        body["stderr_ranges"] =
                            ranges.iter().map(|&(a, n)| json!([a, n])).collect();
                    }
                    put_signal(&j, &mut body);
                    put_pty(&j, &mut body);
                    body
                }
            };
            http::json_response(200, &body)
        }
        "write" => write_stdin(ctx, req, &job, data).await,
        "signal" => {
            let Some(sig) = job.signal.as_deref().and_then(parse_signal) else {
                return http::error_response(400, "unsupported signal");
            };
            // Never a group whose job has ended: its number may be reused. No grace, no wait.
            let mut seen = std::collections::HashSet::new();
            let mut sent = Vec::new();
            let mut targets = Vec::new();
            for id in &job.ids {
                if !seen.insert(id.as_str()) {
                    continue;
                }
                let Some(cell) = jobs.get(&key(id)) else {
                    continue;
                };
                let mut j = locked(&cell);
                if kernel::step(&mut j.k, Ev::Signal) == Act::Signal {
                    sent.push(id.clone());
                    targets.push((j.pg(), j.mark.clone()));
                }
            }
            proc::off_runtime(move || {
                for (g, m) in &targets {
                    proc::signal_job(*g, m, sig);
                }
            })
            .await;
            http::json_response(200, &json!({ "sent": sent }))
        }
        "resize" => resize_job(ctx, &job).await,
        "keep" | "seen" => keep_or_seen(ctx, &job),
        "kill" | "wipe" => {
            let wipe = job.op == "wipe";
            let grace = Duration::from_millis(job.grace_ms.unwrap_or(300));
            // Never a group whose job has ended: its number may be reused.
            let pick = if wipe { Pick::All } else { Pick::Ids(&job.ids) };
            let targets: Vec<(Key, JobCell, proc::Pg, String)> = jobs
                .in_scope(&job.scope, pick)
                .into_iter()
                .filter_map(|(k, c)| {
                    let (stop, pgid, mark) = {
                        let mut j = locked(&c);
                        let act = kernel::step(&mut j.k, Ev::Stop);
                        (act == Act::Stop, j.pg(), j.mark.clone())
                    };
                    stop.then_some((k, c, pgid, mark))
                })
                .collect();
            let stopping: Vec<(proc::Pg, String)> =
                targets.iter().map(|(_, _, g, m)| (*g, m.clone())).collect();
            stop_jobs(&stopping, grace).await;
            let mut survivors = Vec::new();
            for ((_, id), _, g, m) in &targets {
                let left = proc::job_members_async(g.id, m).await;
                if !left.is_empty() {
                    survivors.push(json!({ "id": id, "pids": left }));
                }
            }
            if wipe || job.remove {
                // Every named job whose group is empty goes, the ended ones included.
                let named = jobs.in_scope(&job.scope, pick);
                let mut gone = Vec::new();
                for (k, c) in named {
                    if removable(&c, false).await {
                        gone.push((k, c));
                    }
                }
                jobs.remove(&gone);
            }
            jobs.bump();
            http::json_response(200, &json!({ "survivors": survivors }))
        }
        "remove" => {
            let named = jobs.in_scope(&job.scope, Pick::Ids(&job.ids));
            let mut gone = Vec::new();
            for (k, c) in named {
                if removable(&c, true).await {
                    gone.push((k, c));
                }
            }
            jobs.remove(&gone);
            http::json_response(200, &json!({}))
        }
        "list" => {
            let list: Vec<Value> = jobs
                .in_scope(&job.scope, Pick::All)
                .into_iter()
                .map(|((_, id), c)| {
                    let j = locked(&c);
                    let (state, code) = state_of(&j);
                    let mut entry = json!({
                        "id": id,
                        "state": state,
                        "code": code,
                        "kept": j.k.kept(),
                        "num": j.k.num(),
                        "kind": j.k.kind(),
                        "seen": j.k.seen,
                        "started": j.k.started,
                        "ended": j.k.ended_at(),
                        "label": j.label,
                    });
                    put_signal(&j, &mut entry);
                    put_pty(&j, &mut entry);
                    entry
                })
                .collect();
            http::json_response(200, &json!({ "jobs": list }))
        }
        other => {
            // An op this server does not implement: the embedder's, if it has a hook.
            let hooked = ctx
                .opts
                .job_op_hook
                .as_ref()
                .zip(req.job.as_ref())
                .and_then(|(hook, whole)| hook(other, whole));
            match hooked {
                Some((status, body)) => http::json_response(status, &body),
                None => http::error_response(400, &format!("unknown job op {other:?}")),
            }
        }
    }
}

/// Whether the kernel lets `cell`'s job leave the table. Asked first without looking at its
/// processes (an ended job needs none, for `remove`), then with what the shell sees of its group.
async fn removable(cell: &JobCell, ended_counts: bool) -> bool {
    let (act, pgid, mark) = {
        let mut j = locked(cell);
        let act = kernel::step(
            &mut j.k,
            Ev::Remove {
                empty: false,
                ended_counts,
            },
        );
        (act, j.k.pgid(), j.mark.clone())
    };
    if act == Act::Take {
        return true;
    }
    // A group exists from the spawn: before it there is nothing to observe.
    let empty = pgid != 0 && proc::job_members_async(pgid, &mark).await.is_empty();
    let mut j = locked(cell);
    kernel::step(
        &mut j.k,
        Ev::Remove {
            empty,
            ended_counts,
        },
    ) == Act::Take
}

/// `resize`: set a terminal's size (absolute, so a retry is idempotent). Nothing is changed unless
/// the terminal is open and its job has not ended (T1); a size asked for while the job is spawning
/// is recorded and applied at its spawn (T1-live).
async fn resize_job(ctx: &Arc<Ctx>, job: &JobReq) -> Resp {
    let Some(size) = job
        .cols
        .zip(job.rows)
        .and_then(|(c, r)| kernel::size_ok(c, r))
    else {
        return http::error_response(400, "pty: cols and rows are integers from 1 to 1024");
    };
    let Some(cell) = ctx.jobs.get(&(job.scope.clone(), job.id.clone())) else {
        return http::error_response(404, "no such job");
    };
    let (answer, act, link) = {
        let mut j = locked(&cell);
        let answer = kernel::resizable(&j.k);
        let act = kernel::step(&mut j.k, Ev::Resize { size });
        (answer, act, j.stdin.as_ref().and_then(|s| s.term.clone()))
    };
    match answer {
        Resizable::NoTerminal => return http::error_response(409, "the job has no terminal"),
        Resizable::Ended => return http::error_response(409, "the job has ended"),
        Resizable::HungUp => return http::error_response(409, "the terminal hung up"),
        Resizable::Yes => {}
    }
    if let (Act::Resize { size }, Some(link)) = (act, link)
        && !link.send(&term::resize_frame(size)).await
    {
        return http::error_response(409, "the terminal hung up");
    }
    ctx.jobs.save();
    http::json_response(200, &json!({ "cols": size.cols, "rows": size.rows }))
}

/// `keep` (the job is the endpoint's now, with its `notify`) and `seen` (how much output the model
/// has been shown).
fn keep_or_seen(ctx: &Arc<Ctx>, job: &JobReq) -> Resp {
    let keep = job.op == "keep";
    // notify: a `notify` that fails the grammar refuses the whole `keep`, before anything is
    // recorded (N4, N5). The reason never quotes the token.
    let notify = match (&job.notify, keep) {
        (Some(n), true) => match notify::parse_notify(n) {
            Ok(t) => Some(t),
            Err(why) => {
                return http::error_response(400, &format!("refused notify: {why}"));
            }
        },
        _ => None,
    };
    let key = (job.scope.clone(), job.id.clone());
    let Some(cell) = ctx.jobs.get(&key) else {
        return http::error_response(404, "no such job");
    };
    let mut j = locked(&cell);
    if keep {
        kernel::step(
            &mut j.k,
            Ev::Keep {
                num: job.num,
                kind: job.kind.clone(),
                seen: job.seen,
            },
        );
        // The latest keep governs (N6): with notify it replaces the job's (and its proof abandons
        // whatever was in flight), without it removes it.
        let event = exec_notify::Event::Keep {
            notify: notify.is_some(),
        };
        j.notify = notify;
        notify::feed(ctx, &cell, &key, &mut j, event);
    } else {
        kernel::step(&mut j.k, Ev::Seen { offset: job.offset });
    }
    drop(j);
    if keep {
        ctx.jobs.bump_activity();
        ctx.jobs.save();
    }
    http::json_response(200, &json!({}))
}

/// The terminal's size a start asked for: `Ok(None)` for none, `Err` for one that is not two integers
/// from 1 to 1024 (T3, T4).
fn pty_size(pty: Option<&Value>) -> Result<Option<Size>, ()> {
    let Some(v) = pty else { return Ok(None) };
    let get = |k: &str| v.get(k).and_then(Value::as_u64);
    match (get("cols"), get("rows")) {
        (Some(c), Some(r)) => kernel::size_ok(c, r).map(Some).ok_or(()),
        _ => Err(()),
    }
}

/// What a start put in motion: a command's own process, or a terminal's keeper and its session.
enum Lead {
    Proc(proc::Proc),
    Term(TermLead),
}

/// A launched terminal: its keeper (execd's child), the connection to it, and what its `Ready` said.
struct TermLead {
    keeper: proc::Proc,
    link: Arc<term::Link>,
    rd: tokio::net::unix::OwnedReadHalf,
    frames: term::Frames,
    pgid: i32,
    written: u64,
}

/// Start a terminal's keeper over `out` (the job's spool file) and wait for its `Ready`: the session
/// leads its own group, and the pty is open at `size`. `Err`: what a start answers.
async fn launch_term(
    ctx: &Arc<Ctx>,
    req: &Request,
    cmd: &mut std::process::Command,
    size: Size,
    out: std::fs::File,
    dir: &Path,
) -> Result<TermLead, (u16, &'static str)> {
    // A cap on open terminals (a test's way to the 503 without filling the machine's ptys).
    if let Some(max) = ctx.opts.terminals_max {
        let open = locked(&ctx.jobs.map)
            .all()
            .into_iter()
            .filter(|(_, c)| matches!(locked(c).k.term, Term::Open { .. }))
            .count();
        // The start itself is in the table already.
        if open > max {
            return Err((503, "no pseudo-terminal is free"));
        }
    }
    // The request's own TERM wins; a terminal otherwise gets the usual one.
    if !req.env_pairs.iter().any(|(k, _)| k == "TERM") {
        cmd.env("TERM", "xterm-256color");
    }
    let chown = ctx.opts.run_as.as_ref().map(|r| (r.uid, r.gid));
    let env = term::full_env(cmd);
    let exe = ctx
        .opts
        .helper_exe
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("/proc/self/exe"));
    let mut k = term::keeper_command(&exe, dir, size, cmd, chown, &env);
    // The keeper's own errors go where execd's do (its log).
    k.stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::inherit());
    let mut keeper = proc::spawn(&mut k).map_err(|e| {
        crate::log_warn!("cannot start a terminal's keeper: {e}");
        (500, "cannot start the terminal")
    })?;
    drop(k);
    ctx.groups.insert(keeper.pg(), Kind::Job);
    let pid = keeper.pgid;
    let give_up = |ctx: &Arc<Ctx>, keeper: &proc::Proc| {
        proc::kill_group(keeper.pg());
        ctx.groups.remove(keeper.pg());
    };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let (stream, said, frames) = loop {
        let d = dir.to_path_buf();
        let mut attempt = Box::pin(proc::off_runtime(move || term::connect_sync(&d, pid).ok()));
        let answered = tokio::select! {
            r = &mut attempt => r,
            st = keeper.wait() => {
                // A keeper whose shell ended at once serves execd before it ends: its answer is
                // already in the socket, so the attempt in flight gets to read it.
                let late = attempt.await;
                if late.is_none() {
                    ctx.groups.remove(keeper.pg());
                    crate::log_warn!("a terminal's keeper ended before it answered: {st:?}");
                    // 124: no pseudo-terminal is free (the table is full, or the kernel is out of memory for one).
                    return Err(match st {
                        Ok(s) if s.code() == Some(124) => (503, "no pseudo-terminal is free"),
                        _ => (500, "cannot start the terminal"),
                    });
                }
                late
            }
        };
        match answered {
            Some(c) => break c,
            None if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            None => {
                crate::log_warn!("a terminal's keeper did not answer");
                give_up(ctx, &keeper);
                return Err((500, "cannot start the terminal"));
            }
        }
    };
    let term::Said::Ready { pgid, written, .. } = said else {
        give_up(ctx, &keeper);
        return Err((500, "cannot start the terminal"));
    };
    let stream = stream
        .set_nonblocking(true)
        .and_then(|()| tokio::net::UnixStream::from_std(stream))
        .map_err(|_| {
            give_up(ctx, &keeper);
            (500, "cannot start the terminal")
        })?;
    let (rd, wr) = stream.into_split();
    Ok(TermLead {
        keeper,
        link: Arc::new(term::Link::new(pid, wr, term::pts_ids(pgid))),
        rd,
        frames,
        pgid,
        written,
    })
}

/// Start a job: its own process group, its output kept under the retention, its end recorded once
/// the group is empty.
async fn start_job(ctx: &Arc<Ctx>, req: &Request, job: &JobReq) -> Resp {
    let pty = match pty_size(job.pty.as_ref()) {
        Ok(p) => p,
        Err(()) => {
            return http::error_response(400, "pty: cols and rows are integers from 1 to 1024");
        }
    };
    let key = (job.scope.clone(), job.id.clone());
    let cell: JobCell = {
        let mut map = locked(&ctx.jobs.map);
        if map.admit(&key) == Admit::Exists {
            return http::error_response(409, "the job exists");
        }
        // Placeholder until the spawn gives the group: the entry is the claim on the id, so two
        // starts of one id cannot both run.
        let cell = Arc::new(Mutex::new(Job {
            mark: proc::new_job_mark(ctx.next_seq.fetch_add(1, Ordering::Relaxed)),
            sp: 0,
            label: job.label.clone(),
            out: Ring::new(ctx.opts.job_output_retention, ctx.spool.new_job()),
            spool: None,
            k: kernel::State::new(epoch(), job.hold, pty),
            notify: None,
            notifier: Notifier::default(),
            stdin: None,
            streams: 0,
        }));
        map.insert(key.clone(), cell.clone());
        cell
    };
    let mut cmd = exec::command_for(ctx, req, Kind::Job);
    cmd.stdin(if job.stdin_open && pty.is_none() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mark = locked(&cell).mark.clone();
    cmd.env(proc::JOB_MARK_ENV, &mark);
    // The job's output goes where the kernel says (S1): its spool files, which the job inherits
    // and which outlive execd, read by a tail that wakes on each write. A terminal's output is one
    // stream: its keeper writes `out` and nothing writes `err` (`kernel::sinks`).
    let mut tail = None;
    let mut keeper_out = None;
    let mut spool_dir = None;
    if kernel::output_sink(Kind::Job) == Dest::Spool {
        let made = Spool::create(spool_base(ctx), &mark).and_then(|(sp, w)| {
            let made = Journal::create(&sp.access(), sp.guard(), [0, 0], Vec::new(), 0)
                .and_then(|j| Tail::open(&ctx.tails, &sp, Start::Zero).map(|t| (t, j)));
            match made {
                Ok((t, j)) => Ok((sp, w, t, j)),
                Err(e) => {
                    sp.remove();
                    Err(e)
                }
            }
        });
        match made {
            Ok((sp, w, t, journal)) => {
                if kernel::sinks(locked(&cell).k.term).err {
                    cmd.stdout(Stdio::from(w.out)).stderr(Stdio::from(w.err));
                } else {
                    keeper_out = Some(w.out);
                }
                spool_dir = Some(sp.dir().to_path_buf());
                {
                    let mut j = locked(&cell);
                    // The ring's chunks live in the spool, which a restart reads again.
                    j.out = Ring::new(ctx.opts.job_output_retention, sp.ring_dir())
                        .holding(sp.guard())
                        .kept();
                    j.spool = Some(sp);
                }
                tail = Some((t, journal));
            }
            Err(e) => {
                ctx.jobs.remove(&[(key, cell.clone())]);
                crate::log_warn!("cannot make a job's spool: {e}");
                let (status, why) = spool_failure(&e);
                return http::error_response(status, why);
            }
        }
    }
    let lead = match (pty, keeper_out, &spool_dir) {
        (Some(size), Some(out), Some(dir)) => {
            match launch_term(ctx, req, &mut cmd, size, out, dir).await {
                Ok(t) => Lead::Term(t),
                Err((status, why)) => {
                    ctx.jobs.remove(&[(key, cell.clone())]);
                    return http::error_response(status, why);
                }
            }
        }
        _ => match proc::spawn(&mut cmd) {
            Ok(p) => Lead::Proc(p),
            Err(e) => {
                ctx.jobs.remove(&[(key, cell.clone())]);
                let (_, why) = exec::spawn_failure(ctx, req, &e);
                return http::error_response(500, why.trim_end());
            }
        },
    };
    // The child has its own copies of the write ends; execd holds none.
    drop(cmd);
    let (pgid, pg) = match &lead {
        Lead::Proc(p) => (p.pgid, p.pg()),
        // The session leads a group of its own, which a keeper started: held as an adopted one.
        Lead::Term(t) => (t.pgid, proc::adopt_group(t.pgid)),
    };
    let mut lead = lead;
    let stdin = match &mut lead {
        Lead::Proc(p) => p.stdin.take().map(JobStdin::new),
        Lead::Term(t) => Some(JobStdin::terminal(t.link.clone(), t.written)),
    };
    let act = {
        let mut j = locked(&cell);
        let act = kernel::step(&mut j.k, Ev::Spawned { pgid });
        j.sp = pg.sp;
        j.stdin = stdin.clone();
        act
    };
    ctx.groups.insert(pg, Kind::Job);
    if let (Act::Resize { size }, Some(link)) = (&act, stdin.as_ref().and_then(|s| s.term.as_ref()))
    {
        // A size asked for while the job was spawning reaches the terminal at its spawn (T1-live).
        link.send(&term::resize_frame(*size)).await;
    }
    ctx.jobs.bump_activity();
    ctx.jobs.save();

    let tail = tail.map(|(t, journal)| Tailing::start(ctx, &cell, t, journal));
    match lead {
        Lead::Proc(proc) => {
            tokio::spawn(supervise(
                ctx.clone(),
                cell.clone(),
                key.clone(),
                proc,
                tail,
            ));
        }
        Lead::Term(t) => {
            let TermLead {
                keeper, rd, frames, ..
            } = t;
            let Some(stdin) = stdin.clone() else {
                unreachable!()
            };
            let (tx, rx) = tokio::sync::oneshot::channel();
            tokio::spawn(term_reader(
                ctx.clone(),
                cell.clone(),
                stdin,
                rd,
                frames,
                tx,
            ));
            tokio::spawn(supervise_term(
                ctx.clone(),
                cell.clone(),
                key.clone(),
                pg,
                (Some(keeper), None),
                rx,
                tail,
            ));
        }
    }

    let opened = job.stdin_open || pty.is_some();
    if job.hold {
        let (resp, out) = http::stream_response();
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
        let ctx2 = ctx.clone();
        let beat = req.beat;
        tokio::spawn(async move {
            let end = stream_job(
                ctx2.clone(),
                key,
                cell,
                0,
                true,
                Some((pgid, opened, pty)),
                Sink::new(out, beat, None),
            )
            .await;
            if let Some(o) = &ctx2.opts.observer {
                o.stream_ended(seq, end, std::time::Instant::now());
            }
        });
        resp
    } else {
        let mut body = json!({ "started": pgid });
        if opened {
            body["stdin"] = json!(true);
        }
        if let Some(size) = pty {
            body["pty"] = json!({ "cols": size.cols, "rows": size.rows });
        }
        http::json_response(200, &body)
    }
}

/// What a start answers when its output spool cannot be made: `507` with a plain reason when the
/// disk is out of space (`ENOSPC`; E2B's envd answers `507` for the same), else `500`.
fn spool_failure(e: &std::io::Error) -> (u16, &'static str) {
    if e.kind() == std::io::ErrorKind::StorageFull {
        (
            507,
            "the sandbox's disk is full: no room for the job's output",
        )
    } else {
        (500, "cannot make the job's output spool")
    }
}

/// Where a job's spool files live (S7, `kernel::spool_root`): under the jobs directory, never the
/// ledger's. The kernel chooses between the directory that outlives this execd (a ledger's spools:
/// a restart finds one by the path the ledger names) and this execd's own.
fn spool_base(ctx: &Ctx) -> &crate::spool::Base {
    match kernel::spool_root(ctx.jobs.ledger.enabled()) {
        kernel::Root::Stable => ctx.spool.stable(),
        kernel::Root::PerServer => ctx.spool.server(),
    }
}

/// A job's tail: its two spool files read into its ring, by a task that wakes when they are
/// written ([`crate::tail`]).
struct Tailing {
    ctx: Arc<Ctx>,
    cell: JobCell,
    tail: tokio::sync::Mutex<Tail>,
    /// The unfinished chunk, on disk beside the ring ([`Journal`]).
    journal: Mutex<Journal>,
    done: tokio_util::sync::CancellationToken,
}

impl Tailing {
    fn start(
        ctx: &Arc<Ctx>,
        cell: &JobCell,
        tail: Tail,
        journal: Journal,
    ) -> (Arc<Self>, tokio::task::JoinHandle<()>) {
        let wake = tail.wake.clone();
        let t = Arc::new(Self {
            ctx: ctx.clone(),
            cell: cell.clone(),
            tail: tokio::sync::Mutex::new(tail),
            journal: Mutex::new(journal),
            done: tokio_util::sync::CancellationToken::new(),
        });
        let task = t.clone();
        let handle = tokio::spawn(async move {
            loop {
                if !task.flush().await {
                    return;
                }
                tokio::select! {
                    () = wake.notified() => {}
                    () = task.done.cancelled() => {
                        // The last read, after the group emptied: nothing is written after it.
                        task.flush().await;
                        return;
                    }
                }
            }
        });
        (t, handle)
    }

    /// Read what the files hold into the ring. `false`: the job was removed, so nothing is kept.
    async fn flush(&self) -> bool {
        let mut tail = self.tail.lock().await;
        tail.drain(|data, stderr| {
            {
                let mut j = locked(&self.cell);
                if kernel::step(&mut j.k, Ev::Push) == Act::Discard {
                    return false;
                }
                j.out.push(data, stderr);
                // Before the spool's bytes are returned (the drain punches after this returns).
                locked(&self.journal).record(stderr, data, j.out.tail_len());
            }
            self.ctx.jobs.bump();
            true
        })
        .await
    }
}

/// The job's life: wait for the shell, report `shell_exit` if its group lives on, wait for the
/// group to empty, record the end. Each is a state change notify calls back about.
async fn supervise(
    ctx: Arc<Ctx>,
    cell: JobCell,
    key: Key,
    mut proc: proc::Proc,
    tail: Option<(Arc<Tailing>, tokio::task::JoinHandle<()>)>,
) {
    let status = proc.wait().await;
    let (code, signal) = status.map_or((-1, None), |s| {
        use std::os::unix::process::ExitStatusExt as _;
        match s.code() {
            Some(c) => (c, None),
            None => {
                let sig = s.signal().unwrap_or(0);
                (128 + sig, Some(sig).filter(|&n| n > 0))
            }
        }
    });
    supervise_after(
        &ctx,
        &cell,
        &key,
        proc.pg(),
        Some((code, signal)),
        tail,
        &[proc.pg()],
    )
    .await;
}

/// [`supervise`] for a terminal: the shell's status comes from its keeper (the frame, else the file
/// the keeper wrote, else lost), and the keeper's own group is let go with the job's.
async fn supervise_term(
    ctx: Arc<Ctx>,
    cell: JobCell,
    key: Key,
    pg: proc::Pg,
    (keeper, adopted_keeper): (Option<proc::Proc>, Option<proc::Pg>),
    exit: tokio::sync::oneshot::Receiver<Option<(i32, Option<i32>)>>,
    tail: Option<(Arc<Tailing>, tokio::task::JoinHandle<()>)>,
) {
    let mut groups = vec![pg];
    groups.extend(adopted_keeper);
    if let Some(mut keeper) = keeper {
        let kpg = keeper.pg();
        groups.push(kpg);
        let ctx2 = ctx.clone();
        // The keeper is execd's child: collect it whenever it ends.
        tokio::spawn(async move {
            let _ = keeper.wait().await;
            ctx2.groups.remove(kpg);
        });
    }
    let status = exit.await.ok().flatten().or_else(|| {
        let dir = locked(&cell)
            .spool
            .as_ref()
            .map(|s| s.dir().to_path_buf())?;
        term::read_status(&dir)
    });
    // No status at all (the keeper died before it wrote one): the shell's exit is unknown, so it is
    // not said; the job ends when its group is empty, with the lost code.
    supervise_after(&ctx, &cell, &key, pg, status, tail, &groups).await;
}

/// What a job's supervisor does once the shell has a status: say `shell_exit` if the group lives
/// on, wait for it to empty and record the end.
async fn supervise_after(
    ctx: &Arc<Ctx>,
    cell: &JobCell,
    key: &Key,
    pg: proc::Pg,
    status: Option<(i32, Option<i32>)>,
    tail: Option<(Arc<Tailing>, tokio::task::JoinHandle<()>)>,
    release: &[proc::Pg],
) {
    let (code, signal) = status.unwrap_or((kernel::LOST_CODE, None));
    let pgid = pg.id;
    let mark = locked(cell).mark.clone();
    let mut exits = proc::ChildEvents::new();
    if !proc::job_gone(pgid, &mark, false).await {
        // What the shell printed goes before `shell_exit`: it is in the files already, so read to
        // their ends. The tail goes on, the group lives.
        if let Some((t, _)) = &tail {
            t.flush().await;
        }
        // The moment a held stream would send `shell_exit`: notify calls back.
        if status.is_some() {
            let mut j = locked(cell);
            if kernel::step(&mut j.k, Ev::ShellExit { code, signal }) == Act::Exited {
                notify::change(ctx, cell, key, &mut j);
            }
        }
        ctx.jobs.bump();
        ctx.jobs.save();
        loop {
            let wake = exits.next().await;
            if proc::job_gone(pgid, &mark, wake == proc::Wake::Tick).await {
                break;
            }
        }
    }
    // The group is empty, so nothing writes the files any more: the tail reads them to their ends.
    if let Some((t, task)) = tail {
        t.done.cancel();
        let _ = task.await;
    }
    // The end is observable from here on: notify calls back.
    {
        let mut j = locked(cell);
        let act = kernel::step(
            &mut j.k,
            Ev::GroupEmpty {
                code,
                signal,
                now: epoch(),
            },
        );
        if let Act::Ended { notify, .. } = act {
            if let Some(s) = &j.stdin {
                s.close();
            }
            if notify {
                notify::change(ctx, cell, key, &mut j);
            }
        }
    }
    for g in release {
        ctx.groups.remove(*g);
    }
    ctx.jobs.bump();
    ctx.jobs.save();
}

/// Start of a server: the jobs a previous execd saved ([`crate::ledger`]) come back as
/// [`kernel::recover`] decides, each in the table under its address. A job whose group lives is
/// adopted (watched until the group empties, then ended with the shell's saved status, else
/// [`kernel::LOST_CODE`]) and a held start no one kept, whose client left with the old process, is
/// killed as any disconnect kills it; the rest come back as they ended or not at all. The output
/// of a recovered job is empty. Idempotent: it writes the ledger back as the table now is.
pub(crate) fn recover(ctx: &Arc<Ctx>) {
    let base = spool_base(ctx);
    let mut adopted: Vec<std::path::PathBuf> = Vec::new();
    for e in ctx.jobs.ledger.load() {
        let alive = !proc::job_members(saved_pgid(&e), &e.mark).is_empty();
        // A terminal's keeper, asked while its group lives: it answers (and is the keeper this
        // execd must trust: its pid and uid are checked) or it is gone, and it may have left the
        // shell's status.
        let mut conn = None;
        let keeper = match (&e.term, e.spool.as_deref().map(Path::new)) {
            (None, _) => Keeper::NotTerminal,
            (Some(t), Some(dir)) => {
                let asked = (alive && t.open && e.phase != Saved::Ended)
                    .then(|| term::connect_sync(dir, t.keeper_pid).ok())
                    .flatten();
                match asked {
                    Some((stream, term::Said::Ready { written, size, .. }, frames)) => {
                        conn = Some((stream, written, frames, t.keeper_pid));
                        Keeper::Answered { size }
                    }
                    _ => Keeper::Silent {
                        status: term::read_status(dir),
                    },
                }
            }
            (Some(_), None) => Keeper::Silent { status: None },
        };
        let (mut state, how) = kernel::recover(e.state(), alive, keeper, epoch());
        if how == Recovery::Drop {
            continue;
        }
        // The shell's status a keeper left, for a group that outlived the shell.
        if let (
            Recovery::Adopt,
            Keeper::Silent {
                status: Some((code, signal)),
            },
        ) = (how, keeper)
        {
            kernel::step(&mut state, Ev::ShellExit { code, signal });
        }
        let key = (e.scope.clone(), e.id.clone());
        let pgid = state.pgid();
        let pg = if how == Recovery::Adopt {
            let pg = proc::adopt_group(pgid);
            // A held start's client was the old process's connection, which is gone.
            if let Act::Kill { marked, .. } =
                kernel::step(&mut state, Ev::Disconnect { hold: true })
            {
                // A terminal's whole session dies with its client (T9), not only its first group.
                if marked {
                    proc::signal_job(pg, &e.mark, Signal::SIGKILL);
                } else {
                    proc::kill_group(pg);
                }
            }
            Some(pg)
        } else {
            None
        };
        // The job's spool, which the old execd made and the ledger names, and which the job still
        // writes: the ring is rebuilt from it and its files are read from where execd stopped.
        let spool = pg.and_then(|_| {
            e.spool
                .as_deref()
                .and_then(|d| Spool::at(base, Path::new(d)))
        });
        let rebuilt = spool
            .as_ref()
            .and_then(|sp| rebuild(ctx, sp).or_else(|| restart_cold(ctx, sp)));
        let (out, tail) = match rebuilt {
            Some((ring, tail, journal)) => (ring, Some((tail, journal))),
            None => (
                Ring::new(ctx.opts.job_output_retention, ctx.spool.new_job()),
                None,
            ),
        };
        if let Some(sp) = &spool {
            adopted.push(sp.dir().to_path_buf());
        }
        let cell: JobCell = Arc::new(Mutex::new(Job {
            mark: e.mark.clone(),
            sp: pg.map_or(0, |p| p.sp),
            label: e.label.clone(),
            out,
            spool,
            k: state,
            notify: None,
            notifier: Notifier::default(),
            stdin: None,
            streams: 0,
        }));
        {
            let mut map = locked(&ctx.jobs.map);
            if map.admit(&key) == Admit::Exists {
                continue;
            }
            map.insert(key.clone(), cell.clone());
        }
        if let Some(pg) = pg {
            ctx.groups.insert(pg, Kind::Job);
            let tail = tail.map(|(t, j)| Tailing::start(ctx, &cell, t, j));
            let open = matches!(locked(&cell).k.term, Term::Open { .. });
            match conn {
                // An adopted terminal: its keeper still holds the master, the count of what landed
                // is the keeper's, and its shell's status arrives as a frame.
                Some((stream, written, frames, kpid)) if open => {
                    let adopted = stream
                        .set_nonblocking(true)
                        .and_then(|()| tokio::net::UnixStream::from_std(stream));
                    match adopted {
                        Ok(stream) => {
                            let (rd, wr) = stream.into_split();
                            let link = Arc::new(term::Link::new(kpid, wr, term::pts_ids(pgid)));
                            let stdin = JobStdin::terminal(link, written);
                            locked(&cell).stdin = Some(stdin.clone());
                            let kpg = proc::adopt_group(kpid);
                            ctx.groups.insert(kpg, Kind::Job);
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            tokio::spawn(term_reader(
                                ctx.clone(),
                                cell.clone(),
                                stdin,
                                rd,
                                frames,
                                tx,
                            ));
                            tokio::spawn(supervise_term(
                                ctx.clone(),
                                cell,
                                key,
                                pg,
                                (None, Some(kpg)),
                                rx,
                                tail,
                            ));
                        }
                        Err(_) => {
                            kernel::step(&mut locked(&cell).k, Ev::Hangup);
                            tokio::spawn(supervise_adopted(ctx.clone(), cell, key, pg, tail));
                        }
                    }
                }
                _ => {
                    tokio::spawn(supervise_adopted(ctx.clone(), cell, key, pg, tail));
                }
            }
        }
    }
    // A spool no job in the table owns (its job ended, or was removed, before the restart).
    if ctx.jobs.ledger.enabled() {
        let keep: Vec<&Path> = adopted.iter().map(std::path::PathBuf::as_path).collect();
        Spool::retain(base, &keep);
    }
    ctx.jobs.bump_activity();
    ctx.jobs.save();
}

/// A recovered job's ring and tail, from the spool the ledger names: the chunk files, then the
/// journal's unfinished chunk, then the files from where execd stopped reading them. `execd_jobs::resume`
/// decides where (X1, X2: after the last byte consumed, the ring ending at the bytes consumed) and
/// `replay_skip` what of the journal the chunks already hold (X3): a client's offset names the same
/// byte before and after, and no byte is served twice. `None` (no journal, or one that does not
/// agree with the chunks): the job's output is read from what the files still hold and the ring
/// starts after a gap.
fn rebuild(ctx: &Arc<Ctx>, sp: &Spool) -> Option<(Ring, Tail, Journal)> {
    let retain = ctx.opts.job_output_retention;
    let (start, pieces) = Journal::load(&sp.access())?;
    let from = start[0].checked_add(start[1])?;
    let total: u64 = pieces.iter().map(|p| p.1.len() as u64).sum();
    let (jo, je) = pieces.iter().fold((0u64, 0u64), |(o, e), (err, d)| {
        if *err {
            (o, e + d.len() as u64)
        } else {
            (o + d.len() as u64, e)
        }
    });
    // The arithmetic below needs the sums to fit (`execd_jobs::OFFSET_MAX`).
    if from.checked_add(total)? > kernel::OFFSET_MAX || from % FRAME_MAX as u64 != 0 {
        return None;
    }
    let mut ring = Ring::recover(retain, sp.ring_dir(), from).holding(sp.guard());
    let chunk_end = ring.end();
    if chunk_end > from + total {
        // Chunks the journal never covered: not one this journal can continue.
        return None;
    }
    let (mut at, mut skipped, mut kept) = (from, [0u64; 2], Vec::new());
    for (err, data) in pieces {
        let len = data.len() as u64;
        let skip = kernel::replay_skip(chunk_end, at, len);
        skipped[usize::from(err)] += skip;
        let skip = usize::try_from(skip).ok()?;
        if skip < data.len() {
            ring.push(&data[skip..], err);
            kept.push((err, data[skip..].to_vec()));
        }
        at += len;
    }
    let r = kernel::resume(start[0], start[1], jo, je);
    // X2: the ring ends at the bytes consumed.
    if ring.end() != r.end {
        return None;
    }
    let journal = Journal::create(
        &sp.access(),
        sp.guard(),
        [start[0] + skipped[0], start[1] + skipped[1]],
        kept,
        ring.tail_len(),
    )
    .ok()?;
    let tail = Tail::open(&ctx.tails, sp, Start::At([r.out, r.err])).ok()?;
    Some((ring, tail, journal))
}

/// [`rebuild`]'s way out: the ring starts empty, and the job's files are read from what they still
/// hold (a gap, and bytes may repeat, but none is lost). Logged: this is the offsets moving.
fn restart_cold(ctx: &Arc<Ctx>, sp: &Spool) -> Option<(Ring, Tail, Journal)> {
    crate::log_warn!(
        "a job's journal is missing or does not agree with its chunks; its output restarts with a gap dir={}",
        sp.dir().display()
    );
    let _ = std::fs::remove_dir_all(sp.ring_dir());
    let tail = Tail::open(&ctx.tails, sp, Start::Data).ok()?;
    let journal = Journal::create(&sp.access(), sp.guard(), tail.offsets(), Vec::new(), 0).ok()?;
    let ring = Ring::new(ctx.opts.job_output_retention, sp.ring_dir())
        .holding(sp.guard())
        .kept();
    Some((ring, tail, journal))
}

fn saved_pgid(e: &Entry) -> i32 {
    e.state().pgid()
}

/// [`supervise`] for a job a previous execd started: not our child, so no status arrives. Wait for
/// its group to empty and end it with what the kernel was given: the shell's saved status, else
/// the lost one.
async fn supervise_adopted(
    ctx: Arc<Ctx>,
    cell: JobCell,
    key: Key,
    pg: proc::Pg,
    tail: Option<(Arc<Tailing>, tokio::task::JoinHandle<()>)>,
) {
    let mark = locked(&cell).mark.clone();
    let mut exits = proc::ChildEvents::new();
    while !proc::job_gone(pg.id, &mark, false).await {
        let wake = exits.next().await;
        if proc::job_gone(pg.id, &mark, wake == proc::Wake::Tick).await {
            break;
        }
    }
    if let Some((t, task)) = tail {
        t.done.cancel();
        let _ = task.await;
    }
    {
        let mut j = locked(&cell);
        let (code, signal) = match j.k.phase {
            kernel::Phase::Exited { code, signal, .. } => (code, signal),
            _ => (kernel::LOST_CODE, None),
        };
        let act = kernel::step(
            &mut j.k,
            Ev::GroupEmpty {
                code,
                signal,
                now: epoch(),
            },
        );
        if let Act::Ended { notify, .. } = act {
            if let Some(s) = &j.stdin {
                s.close();
            }
            if notify {
                notify::change(&ctx, &cell, &key, &mut j);
            }
        }
    }
    ctx.groups.remove(pg);
    ctx.jobs.bump();
    ctx.jobs.save();
}

/// The job a stream attached to, while the table still holds that very job under `key`: a removed
/// job is gone to its streams, and a later job of the same id is not its successor.
fn bound(ctx: &Ctx, key: &Key, attached: &JobCell) -> Option<JobCell> {
    let found = ctx.jobs.get(key);
    let same = found.as_ref().is_some_and(|c| Arc::ptr_eq(c, attached));
    kernel::bound(found, same)
}

/// One open stream on a job, counted for [`Jobs::terminals`] while it lives.
struct Attach {
    ctx: Arc<Ctx>,
    cell: JobCell,
}

impl Attach {
    fn new(ctx: &Arc<Ctx>, cell: &JobCell) -> Self {
        locked(cell).streams += 1;
        ctx.jobs.bump_activity();
        Self {
            ctx: ctx.clone(),
            cell: cell.clone(),
        }
    }
}

impl Drop for Attach {
    fn drop(&mut self) {
        let mut j = locked(&self.cell);
        j.streams = j.streams.saturating_sub(1);
        drop(j);
        self.ctx.jobs.bump_activity();
    }
}

/// Stream a job's output from `offset` to its end. `started`: a held start, which says so first and
/// whose job dies with the connection unless it was kept by then. How the stream ended.
async fn stream_job(
    ctx: Arc<Ctx>,
    key: Key,
    attached: JobCell,
    offset: u64,
    hold: bool,
    started: Option<(i32, bool, Option<Size>)>,
    mut sink: Sink,
) -> StreamEnd {
    // While it is open the job is attached (a host that lets idle machines sleep asks).
    let _open = Attach::new(&ctx, &attached);
    // A job stream has no deadline of its own: it lives as long as the job or the connection.
    if let Ok(end) = follow(&ctx, &key, &attached, offset, started, &mut sink).await {
        return end;
    }
    // The connection is gone before the job's end: a held job nobody kept dies with it. Not when it
    // went because execd is stopping and the kernel leaves this job's group alone (S3): the next
    // execd finds the unkept held start in the ledger and ends it as any disconnect does.
    let stopping = ctx.restarting.load(Ordering::Acquire)
        && !ctx.groups.at_shutdown(Kind::Job, kernel::Stop::Hup).signal;
    if hold
        && !stopping
        && let Some(cell) = bound(&ctx, &key, &attached)
    {
        let job = {
            let mut j = locked(&cell);
            match kernel::step(&mut j.k, Ev::Disconnect { hold }) {
                Act::Kill { marked, .. } => Some((j.pg(), j.mark.clone(), marked)),
                _ => None,
            }
        };
        if let Some((g, m, marked)) = job {
            // The group, and with `marked` every process that carries the mark: a terminal's session
            // (T9). Without it the group alone.
            proc::off_runtime(move || {
                if marked {
                    proc::signal_job(g, &m, Signal::SIGKILL);
                } else {
                    proc::kill_group(g);
                }
            })
            .await;
            return StreamEnd::Disconnected { killed: true };
        }
    }
    StreamEnd::Disconnected { killed: false }
}

/// The frames of a job stream, until its final frame (`Ok`) or the client leaves (`Err`).
async fn follow(
    ctx: &Arc<Ctx>,
    key: &Key,
    attached: &JobCell,
    offset: u64,
    started: Option<(i32, bool, Option<Size>)>,
    sink: &mut Sink,
) -> Result<StreamEnd, Cut> {
    if let Some((pgid, opened, pty)) = started {
        let mut first = json!({ "started": pgid });
        if opened {
            first["stdin"] = json!(true);
        }
        if let Some(size) = pty {
            first["pty"] = json!({ "cols": size.cols, "rows": size.rows });
        }
        sink.put(frame::json(&first)).await?;
    }
    let mut changed = ctx.jobs.changed.subscribe();
    let mut sent = offset;
    let mut shell_told = false;
    loop {
        changed.mark_unchanged();
        enum Next {
            Frame(Bytes),
            End(i32, Option<i32>),
            Wait,
            Missing,
        }
        let next = match bound(ctx, key, attached) {
            None => Next::Missing,
            Some(cell) => {
                let j = locked(&cell);
                let total = j.out.end();
                match kernel::follow(&j.k, sent, j.out.base, total, shell_told) {
                    Follow::Gap(n) => {
                        sent = j.out.base;
                        Next::Frame(frame::json(&json!({ "gap": n })))
                    }
                    Follow::Data => {
                        let (_, mut piece, ranges) = j.out.read(sent, FRAME_MAX);
                        // A frame never spans a stream change: it is the stderr run at its start,
                        // or the stdout before the first stderr run.
                        let mut key = "stdout";
                        if let Some(&(a, n)) = ranges.first() {
                            if a == sent {
                                key = "stderr";
                                piece.truncate(usize::try_from(n).unwrap_or(usize::MAX));
                            } else {
                                piece.truncate(usize::try_from(a - sent).unwrap_or(usize::MAX));
                            }
                        }
                        if piece.is_empty() {
                            // The chunk went away under us: the rest is a gap.
                            let n = total - sent;
                            sent = total;
                            Next::Frame(frame::json(&json!({ "gap": n })))
                        } else {
                            sent += piece.len() as u64;
                            Next::Frame(frame::output(key, &piece))
                        }
                    }
                    Follow::ShellExit(c) => {
                        shell_told = true;
                        Next::Frame(frame::json(&json!({ "shell_exit": c })))
                    }
                    Follow::End(c, signal) => Next::End(c, signal),
                    Follow::Wait => Next::Wait,
                }
            }
        };
        match next {
            Next::Frame(f) => sink.put(f).await?,
            Next::End(c, signal) => {
                // As a stream reports it: `signal` in place of `exit_code` for a death by signal.
                let last = match signal {
                    Some(s) => json!({ "signal": s }),
                    None => json!({ "exit_code": c }),
                };
                sink.put_final(frame::json(&last)).await?;
                return Ok(StreamEnd::Finished);
            }
            Next::Missing => {
                let _ = sink
                    .put_final(frame::json(&json!({ "error": "the job is gone" })))
                    .await;
                return Ok(StreamEnd::Finished);
            }
            Next::Wait => {
                tokio::select! {
                    _ = changed.changed() => {}
                    r = sink.quiet() => {
                        r?;
                        sink.put(frame::heartbeat()).await?;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod spool_failure_tests {
    use super::spool_failure;

    /// A start whose spool cannot be made for want of disk answers `507` with a reason a client
    /// can act on; any other failure stays `500`. Red without the mapping (a bare `500`).
    #[test]
    fn a_full_disk_is_507_and_any_other_failure_is_500() {
        let full = std::io::Error::from_raw_os_error(28) /* ENOSPC */;
        assert_eq!(spool_failure(&full).0, 507);
        assert!(spool_failure(&full).1.contains("disk is full"));
        let other = std::io::Error::from_raw_os_error(13) /* EACCES */;
        assert_eq!(spool_failure(&other).0, 500);
    }
}
