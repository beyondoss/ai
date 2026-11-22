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
//!   unfinished chunk. Older bytes are dropped and read back as a gap. A job's two pumps write into it
//!   directly; there is no channel to fill, and a slow `attach` never slows the job (it reads a gap
//!   instead).
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
//! - *Pumps*: two per job, freed when their pipes close, when the job is removed, or when the
//!   supervisor lets go of a pipe a descendant holds open after the grace.

use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use nix::sys::signal::Signal;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::exec::{self, AfterExit, Cut, PIPE_GRACE, Sink};
use crate::frame::{self, FRAME_MAX};
use crate::http::{self, Line, Request, Resp};
use crate::job_kernel::{self as kernel, Act, Admit, Alive, Ev, Follow, Pick};
use crate::notify::{self, Notifier, NotifyTarget};
use crate::proc::{self, locked};
use crate::ring::Ring;
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
    label: String,
    out: Ring,
    k: kernel::State,
    /// The latest `keep`'s `notify` (notify), if it carried one.
    pub notify: Option<NotifyTarget>,
    /// When it calls back.
    pub notifier: Notifier,
    /// The job's open stdin, when it was started with `stdin_open`.
    stdin: Option<Arc<JobStdin>>,
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
}

impl Jobs {
    pub fn new(activity: Arc<watch::Sender<u64>>) -> Self {
        Self {
            map: Mutex::default(),
            changed: watch::Sender::new(0),
            activity,
        }
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
        for cell in cells {
            let mut j = locked(&cell);
            if kernel::step(&mut j.k, Ev::Taken) == Act::Release {
                j.notifier = Notifier::removed();
                if let Some(s) = &j.stdin {
                    s.close();
                }
            }
        }
        self.bump();
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
async fn stop_jobs(jobs: &[(i32, String)], grace: Duration) {
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
    let any_live = |jobs: &[(i32, String)]| {
        let jobs = jobs.to_vec();
        async move { proc::off_runtime(move || jobs.iter().any(|(g, m)| !proc::job_empty(*g, m))).await }
    };
    let deadline = tokio::time::Instant::now() + grace;
    while any_live(&jobs).await && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    proc::off_runtime({
        let jobs = jobs.clone();
        move || {
            for (g, m) in &jobs {
                if !proc::job_empty(*g, m) {
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
/// The most writes waiting on one job's stdin (for the pipe, or for earlier bytes): each holds at
/// most `WRITE_MAX` bytes plus its base64, so a job that never reads costs a bounded amount.
pub(crate) const WAITERS_MAX: usize = 16;
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
    if stdin.waiters.fetch_add(1, Ordering::Relaxed) >= WAITERS_MAX {
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
    let end = job.at.saturating_add(data.len() as u64);
    let deadline = req
        .timeout_ms
        .map(|ms| tokio::time::Instant::now() + Duration::from_millis(ms));
    // Earlier bytes first: wait (without holding the pipe) until the stream reaches `at`.
    let mut rx = stdin.written.subscribe();
    let gap_by = deadline.unwrap_or_else(|| tokio::time::Instant::now() + GAP_WAIT);
    while *rx.borrow_and_update() < job.at {
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
            return if end <= *stdin.written.borrow() {
                http::json_response(200, &state(true))
            } else {
                closed(&state(true))
            };
        }
    };
    // Bytes below `written` are applied: skip them.
    let skip = usize::try_from(stdin.written.borrow().saturating_sub(job.at))
        .unwrap_or(usize::MAX)
        .min(data.len());
    let mut rest = &data[skip..];
    if *stdin.written.borrow() < job.at {
        // The job ended while this waited for the missing bytes.
        return closed(&state(pipe.is_none()));
    }
    if !rest.is_empty() && pipe.is_none() {
        return closed(&state(true));
    }
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
                stdin.written.send_modify(|w| *w += n as u64);
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
    if job.eof {
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
                out,
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
                    targets.push((j.k.pgid(), j.mark.clone()));
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
            let targets: Vec<(Key, JobCell, i32, String)> = jobs
                .in_scope(&job.scope, pick)
                .into_iter()
                .filter_map(|(k, c)| {
                    let (stop, pgid, mark) = {
                        let mut j = locked(&c);
                        let act = kernel::step(&mut j.k, Ev::Stop);
                        (act == Act::Stop, j.k.pgid(), j.mark.clone())
                    };
                    stop.then_some((k, c, pgid, mark))
                })
                .collect();
            let stopping: Vec<(i32, String)> =
                targets.iter().map(|(_, _, g, m)| (*g, m.clone())).collect();
            stop_jobs(&stopping, grace).await;
            let mut survivors = Vec::new();
            for ((_, id), _, g, m) in &targets {
                let left = proc::job_members_async(*g, m).await;
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
            label: job.label.clone(),
            out: Ring::new(ctx.opts.job_output_retention, ctx.spool.new_job()),
            k: kernel::State::new(epoch(), job.hold),
            notify: None,
            notifier: Notifier::default(),
            stdin: None,
        }));
        map.insert(key.clone(), cell.clone());
        cell
    };
    let mut cmd = exec::command_for(ctx, req);
    cmd.stdin(if job.stdin_open {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mark = locked(&cell).mark.clone();
    cmd.env(proc::JOB_MARK_ENV, &mark);
    let mut proc = match proc::spawn(&mut cmd) {
        Ok(p) => p,
        Err(e) => {
            ctx.jobs.remove(&[(key, cell.clone())]);
            let (_, why) = exec::spawn_failure(ctx, req, &e);
            return http::error_response(500, why.trim_end());
        }
    };
    let pgid = proc.pgid;
    let stdin = proc.stdin.take().map(JobStdin::new);
    {
        let mut j = locked(&cell);
        kernel::step(&mut j.k, Ev::Spawned { pgid });
        j.stdin = stdin;
    }
    ctx.groups.insert(pgid);
    ctx.jobs.bump_activity();

    // Both pipes into one ring, in arrival order. `progress` counts what the pumps have read, so
    // the supervisor can tell when they have gone quiet.
    let progress = Arc::new(AtomicUsize::new(0));
    let (exited, _) = tokio::sync::watch::channel(false);
    let mut pipes = Vec::new();
    let mut cap = 0;
    if let Some(p) = proc.stdout.take() {
        let n = proc::pipe_capacity(&p);
        cap += n;
        let after = AfterExit::new(&exited);
        pipes.push(pump(
            ctx.clone(),
            cell.clone(),
            false,
            p,
            progress.clone(),
            after,
        ));
    }
    if let Some(p) = proc.stderr.take() {
        let n = proc::pipe_capacity(&p);
        cap += n;
        let after = AfterExit::new(&exited);
        pipes.push(pump(
            ctx.clone(),
            cell.clone(),
            true,
            p,
            progress.clone(),
            after,
        ));
    }
    tokio::spawn(supervise(
        ctx.clone(),
        cell.clone(),
        key.clone(),
        proc,
        pipes,
        progress,
        cap,
        exited,
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
        tokio::spawn(async move {
            let end = stream_job(ctx2.clone(), key, cell, 0, true, Some((pgid, opened)), out).await;
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

/// Read `pipe` into the job's ring until it closes or the job is removed.
fn pump<R: tokio::io::AsyncRead + std::os::fd::AsFd + Unpin + Send + 'static>(
    ctx: Arc<Ctx>,
    cell: JobCell,
    stderr: bool,
    mut pipe: R,
    progress: Arc<AtomicUsize>,
    mut after: AfterExit,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut buf = vec![0u8; frame::READ_BUF];
        while let Some(n) = after.read(&mut pipe, &mut buf).await {
            {
                let mut j = locked(&cell);
                if kernel::step(&mut j.k, Ev::Push) == Act::Discard {
                    return;
                }
                j.out.push(&buf[..n], stderr);
            }
            progress.fetch_add(n, Ordering::Relaxed);
            ctx.jobs.bump();
        }
    })
}

/// Wait for the pumps to read what the pipes hold: until they close, go quiet for [`PIPE_GRACE`],
/// or `budget` more bytes have gone by (all a command can have written and not been read is the
/// pipes' capacity; a descendant holding a pipe open and chatty is let go).
async fn drain_pumps(
    pipes: &mut [tokio::task::JoinHandle<()>],
    progress: &AtomicUsize,
    budget: usize,
) {
    let start = progress.load(Ordering::Relaxed);
    let mut seen = start;
    loop {
        let all = async {
            for p in pipes.iter_mut() {
                // A pump that finished was awaited by an earlier call (one is made before and one
                // after the group empties); a join handle must not be polled again.
                if !p.is_finished() {
                    let _ = p.await;
                }
            }
        };
        if tokio::time::timeout(PIPE_GRACE, all).await.is_ok() {
            return;
        }
        let now = progress.load(Ordering::Relaxed);
        if now == seen || now.saturating_sub(start) >= budget {
            return;
        }
        seen = now;
    }
}

/// The job's life: wait for the shell, report `shell_exit` if its group lives on, wait for the
/// group to empty, record the end. Each is a state change notify calls back about.
#[allow(clippy::too_many_arguments)] // the job's parts, each used once
async fn supervise(
    ctx: Arc<Ctx>,
    cell: JobCell,
    key: Key,
    mut proc: proc::Proc,
    mut pipes: Vec<tokio::task::JoinHandle<()>>,
    progress: Arc<AtomicUsize>,
    cap: usize,
    exited: tokio::sync::watch::Sender<bool>,
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
        // What the shell printed goes before `shell_exit`; the pumps read on, the group lives.
        drain_pumps(&mut pipes, &progress, cap).await;
        // The moment a held stream would send `shell_exit`: notify calls back.
        {
            let mut j = locked(&cell);
            if kernel::step(&mut j.k, Ev::ShellExit { code, signal }) == Act::Exited {
                notify::change(&ctx, &cell, &key, &mut j);
            }
        }
        ctx.jobs.bump();
        loop {
            let wake = exits.next().await;
            if proc::job_gone(pgid, &mark, wake == proc::Wake::Tick).await {
                break;
            }
        }
    }
    // The group is empty: the pumps take what the pipes still hold and end when a pipe closes or
    // goes quiet (`AfterExit`), so nothing is aborted with output unread.
    exited.send_replace(true);
    for p in &mut pipes {
        // One that finished was awaited by `drain_pumps`; a join handle must not be polled again.
        if !p.is_finished() {
            let _ = p.await;
        }
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
    ctx.groups.remove(pgid);
    ctx.jobs.bump();
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
    out: mpsc::Sender<Bytes>,
) -> StreamEnd {
    // A job stream has no deadline of its own: it lives as long as the job or the connection.
    let mut sink = Sink::new(out, ctx.opts.heartbeat, None);
    if let Ok(end) = follow(&ctx, &key, &attached, offset, started, &mut sink).await {
        return end;
    }
    // The connection is gone before the job's end: a held job nobody kept dies with it.
    if hold && let Some(cell) = bound(&ctx, &key, &attached) {
        let job = {
            let mut j = locked(&cell);
            match kernel::step(&mut j.k, Ev::Disconnect { hold }) {
                Act::Kill { pgid } => Some((pgid, j.mark.clone())),
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
