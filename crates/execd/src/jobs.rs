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
    self as kernel, Act, Admit, Alive, Ev, Follow, Kind, Pick, Recovery, Sink as Dest,
};
use crate::ledger::{Entry, Ledger};
use crate::notify::{self, Notifier, NotifyTarget};
use crate::proc::{self, locked};
use crate::ring::Ring;
use crate::tail::{Journal, Spool, Start, Tail};
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
                    Entry::of(
                        &scope,
                        &id,
                        &j.label,
                        &j.mark,
                        &j.k,
                        j.spool.as_ref().map(Spool::dir),
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
            if kernel::step(&mut j.k, Ev::Taken) == Act::Release {
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
}

impl JobStdin {
    fn new(pipe: tokio::process::ChildStdin) -> Arc<Self> {
        Arc::new(Self {
            pipe: tokio::sync::Mutex::new(Some(pipe)),
            written: watch::Sender::new(0),
            dead: CancellationToken::new(),
            waiters: AtomicUsize::new(0),
        })
    }

    /// The job is over: no write lands again and the fd is released once any write in flight
    /// has noticed.
    fn close(self: &Arc<Self>) {
        self.dead.cancel();
        let me = self.clone();
        tokio::spawn(async move {
            me.pipe.lock().await.take();
        });
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
    let Some(stdin) = locked(&cell).stdin.clone() else {
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
            // Over: bytes already applied are idempotent, anything else is refused.
            let plan = kernel::stdin_plan(*stdin.written.borrow(), job.at, len, job.eof, false, true);
            return match plan {
                kernel::Plan::Done { .. } => http::json_response(200, &state(true)),
                _ => closed(&state(true)),
            };
        }
    };
    // The kernel decides what lands: the shell does the pipe's I/O it names.
    let plan = kernel::stdin_plan(
        *stdin.written.borrow(),
        job.at,
        len,
        job.eof,
        pipe.is_some(),
        stdin.dead.is_cancelled(),
    );
    let (skip, close) = match plan {
        // `Wait` cannot be asked here: the loop above left only with the bytes before `at`
        // applied or the job over.
        kernel::Plan::Wait | kernel::Plan::Refuse => return closed(&state(true)),
        kernel::Plan::Done { close } => {
            if close {
                pipe.take();
            }
            return http::json_response(200, &state(pipe.is_none()));
        }
        kernel::Plan::Apply { skip, close, .. } => {
            (usize::try_from(skip).unwrap_or(usize::MAX), close)
        }
    };
    let mut rest = &data[skip.min(data.len())..];
    while !rest.is_empty() {
        let Some(p) = pipe.as_mut() else { break };
        let step = async {
            tokio::select! {
                r = p.write(rest) => Some(r),
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
                rest = &rest[n..];
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
    if close {
        pipe.take();
    }
    http::json_response(200, &state(pipe.is_none()))
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

/// Start a job: its own process group, its output kept under the retention, its end recorded once
/// the group is empty.
async fn start_job(ctx: &Arc<Ctx>, req: &Request, job: &JobReq) -> Resp {
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
            k: kernel::State::new(epoch(), job.hold),
            notify: None,
            notifier: Notifier::default(),
            stdin: None,
        }));
        map.insert(key.clone(), cell.clone());
        cell
    };
    let mut cmd = exec::command_for(ctx, req, Kind::Job);
    cmd.stdin(if job.stdin_open {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mark = locked(&cell).mark.clone();
    cmd.env(proc::JOB_MARK_ENV, &mark);
    // The job's output goes where the kernel says (S1): its spool files, which the job inherits
    // and which outlive execd, read by a tail that wakes on each write.
    let mut tail = None;
    if kernel::output_sink(Kind::Job) == Dest::Spool {
        let made = Spool::create(&spool_base(ctx), &mark).and_then(|(sp, w)| {
            let made = Journal::create(sp.dir(), [0, 0], Vec::new(), 0)
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
                cmd.stdout(Stdio::from(w.out)).stderr(Stdio::from(w.err));
                {
                    let mut j = locked(&cell);
                    // The ring's chunks live in the spool, which a restart reads again.
                    j.out = Ring::new(ctx.opts.job_output_retention, sp.ring_dir()).kept();
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
    let mut proc = match proc::spawn(&mut cmd) {
        Ok(p) => p,
        Err(e) => {
            ctx.jobs.remove(&[(key, cell.clone())]);
            let (_, why) = exec::spawn_failure(ctx, req, &e);
            return http::error_response(500, why.trim_end());
        }
    };
    // The child has its own copies of the write ends; execd holds none.
    drop(cmd);
    let pgid = proc.pgid;
    let stdin = proc.stdin.take().map(JobStdin::new);
    {
        let mut j = locked(&cell);
        kernel::step(&mut j.k, Ev::Spawned { pgid });
        j.sp = proc.sp;
        j.stdin = stdin;
    }
    ctx.groups.insert(proc.pg(), Kind::Job);
    ctx.jobs.bump_activity();
    ctx.jobs.save();

    let tail = tail.map(|(t, journal)| Tailing::start(ctx, &cell, t, journal));
    tokio::spawn(supervise(
        ctx.clone(),
        cell.clone(),
        key.clone(),
        proc,
        tail,
    ));

    let opened = job.stdin_open;
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
                Some((pgid, opened)),
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

/// Where a job's spool files live: under the ledger's directory when there is one (they outlive
/// this execd and a restart finds them by the job's mark), else under the jobs directory.
fn spool_base(ctx: &Ctx) -> std::path::PathBuf {
    match (&ctx.opts.state_dir, ctx.jobs.ledger.enabled()) {
        (Some(d), true) => d.join("spool"),
        _ => ctx.spool.path().to_path_buf(),
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
    let pgid = proc.pgid;
    let mark = locked(&cell).mark.clone();
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
    let mut exits = proc::ChildEvents::new();
    if !proc::job_gone(pgid, &mark, false).await {
        // What the shell printed goes before `shell_exit`: it is in the files already, so read to
        // their ends. The tail goes on, the group lives.
        if let Some((t, _)) = &tail {
            t.flush().await;
        }
        // The moment a held stream would send `shell_exit`: notify calls back.
        {
            let mut j = locked(&cell);
            if kernel::step(&mut j.k, Ev::ShellExit { code, signal }) == Act::Exited {
                notify::change(&ctx, &cell, &key, &mut j);
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
        let mut j = locked(&cell);
        let act = kernel::step(
            &mut j.k,
            Ev::GroupEmpty {
                code,
                signal,
                now: epoch(),
            },
        );
        if let Act::Ended { notify } = act {
            if let Some(s) = &j.stdin {
                s.close();
            }
            if notify {
                notify::change(&ctx, &cell, &key, &mut j);
            }
        }
    }
    ctx.groups.remove(proc.pg());
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
        let (mut state, how) = kernel::recover(e.state(), alive, epoch());
        if how == Recovery::Drop {
            continue;
        }
        let key = (e.scope.clone(), e.id.clone());
        let pgid = state.pgid();
        let pg = if how == Recovery::Adopt {
            let pg = proc::adopt_group(pgid);
            // A held start's client was the old process's connection, which is gone.
            if let Act::Kill { .. } = kernel::step(&mut state, Ev::Disconnect { hold: true }) {
                proc::kill_group(pg);
            }
            Some(pg)
        } else {
            None
        };
        // The job's spool, which the old execd made and the ledger names, and which the job still
        // writes: the ring is rebuilt from it and its files are read from where execd stopped.
        let spool = pg.and_then(|_| e.spool.as_deref().and_then(|d| Spool::at(Path::new(d))));
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
            tokio::spawn(supervise_adopted(ctx.clone(), cell, key, pg, tail));
        }
    }
    // A spool no job in the table owns (its job ended, or was removed, before the restart).
    if ctx.jobs.ledger.enabled() {
        let keep: Vec<&Path> = adopted.iter().map(std::path::PathBuf::as_path).collect();
        Spool::retain(&base, &keep);
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
    let (start, pieces) = Journal::load(sp.dir())?;
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
    let mut ring = Ring::recover(retain, sp.ring_dir(), from);
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
        sp.dir(),
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
    let journal = Journal::create(sp.dir(), tail.offsets(), Vec::new(), 0).ok()?;
    let ring = Ring::new(ctx.opts.job_output_retention, sp.ring_dir()).kept();
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
        if let Act::Ended { notify } = act {
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

/// Stream a job's output from `offset` to its end. `started`: a held start, which says so first and
/// whose job dies with the connection unless it was kept by then. How the stream ended.
async fn stream_job(
    ctx: Arc<Ctx>,
    key: Key,
    attached: JobCell,
    offset: u64,
    hold: bool,
    started: Option<(i32, bool)>,
    mut sink: Sink,
) -> StreamEnd {
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
                Act::Kill { .. } => Some((j.pg(), j.mark.clone())),
                _ => None,
            }
        };
        if let Some((g, m)) = job {
            proc::off_runtime(move || proc::signal_job(g, &m, Signal::SIGKILL)).await;
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
    started: Option<(i32, bool)>,
    sink: &mut Sink,
) -> Result<StreamEnd, Cut> {
    if let Some((pgid, opened)) = started {
        let mut first = json!({ "started": pgid });
        if opened {
            first["stdin"] = json!(true);
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
