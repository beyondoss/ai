//! v1.3: sandbox jobs, natively. A job is a command that outlives the request that started it: it
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
//! - *Streams* (`attach`, a held `start`): the same bounded channel as v1.2, one task each, freed
//!   when the stream ends or its client leaves.
//! - *Pumps*: two per job, freed when their pipes close, when the job is removed, or when the
//!   supervisor lets go of a pipe a descendant holds open after the grace.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use nix::sys::signal::Signal;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncReadExt as _;
use tokio::sync::{mpsc, watch};

use crate::exec::{self, Cut, PIPE_GRACE, Sink};
use crate::frame::{self, FRAME_MAX};
use crate::http::{self, Line, Request, Resp};
use crate::notify::{self, Notifier, NotifyTarget};
use crate::proc::{self, locked};
use crate::spool;
use crate::{Ctx, StreamEnd, StreamStart};

/// `(scope, id)`: a job's address.
pub(crate) type Key = (String, String);

/// One job, shared between its supervisor, its pumps and the requests that name it.
pub(crate) type JobCell = Arc<Mutex<Job>>;

/// The last bytes of a job's output, by offset from its first byte, kept as chunk files (see
/// [`crate::spool`]). A chunk is the 64 KiB at a multiple of 64 KiB; whole chunks are evicted from
/// the front, so the retention is a floor. In memory: a path and the unfinished chunk.
pub(crate) struct Ring {
    /// This job's directory, created when the first chunk is written.
    dir: PathBuf,
    made: bool,
    /// Full chunks on disk.
    files: usize,
    /// The bytes after them, fewer than a chunk.
    tail: Vec<u8>,
    /// The offset of the first byte held (a multiple of a chunk).
    base: u64,
    retain: usize,
}

impl Ring {
    fn new(retain: usize, dir: PathBuf) -> Self {
        Self {
            dir,
            made: false,
            files: 0,
            tail: Vec::new(),
            base: 0,
            retain,
        }
    }

    fn held(&self) -> usize {
        self.files * FRAME_MAX + self.tail.len()
    }

    /// The offset of the next byte.
    pub fn end(&self) -> u64 {
        self.base + self.held() as u64
    }

    fn chunk_path(&self, offset: u64) -> PathBuf {
        self.dir.join(format!("{:010x}", offset / FRAME_MAX as u64))
    }

    fn push(&mut self, data: &[u8]) {
        self.tail.extend_from_slice(data);
        while self.tail.len() >= FRAME_MAX {
            let at = self.base + (self.files * FRAME_MAX) as u64;
            let wrote = (self.made || spool::make_job_dir(&self.dir).is_ok())
                && spool::write_chunk(&self.chunk_path(at), &self.tail[..FRAME_MAX]).is_ok();
            self.made |= wrote;
            self.tail.drain(..FRAME_MAX);
            if wrote {
                self.files += 1;
            } else {
                // The disk refused it: what was held and this chunk read back as a gap.
                self.forget_files();
                self.base = at + FRAME_MAX as u64;
            }
        }
        while self.files > 0 && self.held() - FRAME_MAX >= self.retain {
            let _ = std::fs::remove_file(self.chunk_path(self.base));
            self.base += FRAME_MAX as u64;
            self.files -= 1;
        }
    }

    fn forget_files(&mut self) {
        for i in 0..self.files {
            let _ = std::fs::remove_file(self.chunk_path(self.base + (i * FRAME_MAX) as u64));
        }
        self.files = 0;
    }

    /// Up to `max` bytes from `offset` (or from the oldest byte held, when that is later); where
    /// they start.
    fn read(&self, offset: u64, max: usize) -> (u64, Vec<u8>) {
        let from = offset.max(self.base);
        let want = (self.end().saturating_sub(from) as usize).min(max);
        let mut out = vec![0u8; want];
        let mut done = 0;
        let on_disk = self.base + (self.files * FRAME_MAX) as u64;
        while done < want {
            let at = from + done as u64;
            if at >= on_disk {
                let t = (at - on_disk) as usize;
                let n = (want - done).min(self.tail.len() - t);
                out[done..done + n].copy_from_slice(&self.tail[t..t + n]);
                done += n;
            } else {
                let pos = at % FRAME_MAX as u64;
                let n = (want - done).min(FRAME_MAX - pos as usize);
                match spool::read_chunk(&self.chunk_path(at), pos, &mut out[done..done + n]) {
                    Ok(k) if k > 0 => done += k,
                    _ => break,
                }
            }
        }
        out.truncate(done);
        (from, out)
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        if self.made {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// One job, as the endpoint keeps it.
pub(crate) struct Job {
    pgid: i32,
    label: String,
    started: u64,
    ended: u64,
    out: Ring,
    shell_exit: Option<i32>,
    end: Option<i32>,
    kept: bool,
    num: Option<u64>,
    kind: Option<String>,
    seen: u64,
    /// The latest `keep`'s `notify` (v1.4), if it carried one.
    pub notify: Option<NotifyTarget>,
    /// When it calls back.
    pub notifier: Notifier,
    /// Taken out of the table: its pumps and timers stop.
    pub removed: bool,
}

/// Every job, by `(scope, id)`, and a counter bumped on every change a stream may be waiting for.
pub(crate) struct Jobs {
    map: Mutex<HashMap<Key, JobCell>>,
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
    pub fn alive(&self) -> (usize, Vec<crate::KeptJob>) {
        let cells: Vec<(Key, JobCell)> = locked(&self.map)
            .iter()
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect();
        let mut alive = 0;
        let mut kept = Vec::new();
        for ((scope, id), cell) in cells {
            let j = locked(&cell);
            if j.end.is_some() || j.removed || j.pgid == 0 {
                continue;
            }
            alive += 1;
            if j.kept {
                kept.push(crate::KeptJob {
                    pgid: j.pgid,
                    scope,
                    id,
                });
            }
        }
        (alive, kept)
    }

    fn get(&self, key: &Key) -> Option<JobCell> {
        locked(&self.map).get(key).cloned()
    }

    /// The jobs of `scope` matching `pick`.
    fn in_scope(&self, scope: &str, pick: impl Fn(&str) -> bool) -> Vec<(Key, JobCell)> {
        let map = locked(&self.map);
        map.iter()
            .filter(|((sc, id), _)| sc == scope && pick(id))
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect()
    }

    /// Take `keys` out of the table; their pumps and timers stop.
    fn remove(&self, keys: &[Key]) {
        let cells: Vec<JobCell> = {
            let mut map = locked(&self.map);
            keys.iter().filter_map(|k| map.remove(k)).collect()
        };
        for cell in cells {
            let mut j = locked(&cell);
            j.removed = true;
            j.notifier = Notifier::default();
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
    match (j.end, j.shell_exit) {
        (Some(c), _) => ("ended", json!(c)),
        (None, Some(c)) => ("exited", json!(c)),
        (None, None) => ("running", Value::Null),
    }
}

/// `SIGTERM` the groups, wait up to `grace` for them to empty, `SIGKILL` what is left, wait up to a
/// second more (v1.3 `kill`).
async fn stop_groups(pgids: &[i32], grace: Duration) {
    for &g in pgids {
        proc::signal_group(g, Signal::SIGTERM);
    }
    let any_live = |pgids: &[i32]| {
        let pgids = pgids.to_vec();
        async move { proc::off_runtime(move || pgids.iter().any(|&g| !proc::group_empty(g))).await }
    };
    let deadline = tokio::time::Instant::now() + grace;
    while any_live(pgids).await && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for &g in pgids {
        if !proc::group_empty(g) {
            proc::kill_group(g);
        }
    }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
    while any_live(pgids).await && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Fuzz seam: whether `job` is a well-formed job operation (and its `notify`, if it has one, passes
/// the grammar). Never panics.
pub(crate) fn fuzz_job(job: &Value) -> bool {
    match serde_json::from_value::<JobReq>(job.clone()) {
        Ok(j) => {
            j.op != "keep"
                || j.notify
                    .as_ref()
                    .is_none_or(|n| notify::parse_notify(n).is_ok())
        }
        Err(_) => false,
    }
}

/// Fuzz seam: drive a [`Ring`] with the pushes and reads `bytes` describe, and hold it to a plain
/// `Vec` of everything pushed: every read returns the bytes at its offsets, from the oldest kept
/// byte when the offset is older, and the ring keeps at least its retention.
pub(crate) fn fuzz_ring(bytes: &[u8]) {
    let mut it = bytes.iter().copied();
    let retain = 1 + usize::from(it.next().unwrap_or(0)) * 1024;
    // The ring spools to files: give this run a directory of its own, removed when it ends.
    static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "execd-fuzz-ring-{}-{}",
        std::process::id(),
        RUNS.fetch_add(1, Ordering::Relaxed)
    ));
    let _cleanup = scopeguard_remove(dir.clone());
    let _ = std::fs::create_dir_all(&dir);
    let mut ring = Ring::new(retain, dir.join("ring"));
    let mut all: Vec<u8> = Vec::new();
    while let Some(op) = it.next() {
        let a = u16::from_le_bytes([it.next().unwrap_or(0), it.next().unwrap_or(0)]);
        if op % 2 == 0 {
            let seed = op;
            let data: Vec<u8> = (0..usize::from(a))
                .map(|i| (i % 251) as u8 ^ seed)
                .collect();
            ring.push(&data);
            all.extend_from_slice(&data);
            continue;
        }
        let end = ring.end();
        assert_eq!(end as usize, all.len());
        // Offsets: near zero, around the end, and absurd. Sizes: small, huge and unbounded.
        let offset = match op % 8 {
            1 => u64::from(a),
            3 => end.saturating_sub(u64::from(a)),
            5 => end + u64::from(a),
            _ => u64::MAX - u64::from(a),
        };
        let max = match (op / 8) % 4 {
            0 => usize::from(a),
            1 => usize::MAX,
            2 => usize::MAX - usize::from(a),
            _ => 1 << 20,
        };
        let (from, data) = ring.read(offset, max);
        assert_eq!(from, offset.max(ring.base));
        let avail = end.saturating_sub(from).min(max as u64);
        assert_eq!(data.len() as u64, avail);
        if avail > 0 {
            assert_eq!(&data[..], &all[from as usize..from as usize + data.len()]);
        }
        let kept = end - ring.base;
        assert!(kept >= (all.len().min(retain)) as u64);
        assert!(kept < (retain + FRAME_MAX) as u64);
    }
}

/// Removes a directory tree when dropped.
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scopeguard_remove(p: PathBuf) -> RemoveOnDrop {
    RemoveOnDrop(p)
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
}

/// Answer a request carrying a `job` object.
pub(crate) async fn serve(ctx: &Arc<Ctx>, req: &Request) -> Resp {
    let job: JobReq = match req.job.clone().map(serde_json::from_value) {
        Some(Ok(j)) => j,
        _ => return http::error_response(400, "bad job"),
    };
    let line = Line::new("job", &req.command);
    let resp = serve_op(ctx, req, job).await;
    line.done(resp.status().as_u16(), "answered");
    resp
}

async fn serve_op(ctx: &Arc<Ctx>, req: &Request, job: JobReq) -> Resp {
    let jobs = &ctx.jobs;
    let key = |id: &str| (job.scope.clone(), id.to_string());
    match job.op.as_str() {
        "probe" => {
            let mut answer = json!({ "jobs": "v1.3" });
            if ctx.opts.advertises_notify() {
                answer["notify"] = json!(crate::NOTIFY_VERSION);
            }
            http::json_response(200, &answer)
        }
        "start" => start_job(ctx, req, &job).await,
        "attach" => {
            let k = key(&job.id);
            if jobs.get(&k).is_none() {
                return http::error_response(404, "no such job");
            }
            let (resp, out) = http::stream_response();
            tokio::spawn(stream_job(ctx.clone(), k, job.offset, false, None, out));
            resp
        }
        "read" => {
            let body = match jobs.get(&key(&job.id)) {
                None => json!({ "state": "gone", "offset": 0, "end": 0, "data": "" }),
                Some(cell) => {
                    let j = locked(&cell);
                    let (state, code) = state_of(&j);
                    let (from, data) = j
                        .out
                        .read(job.offset, usize::try_from(job.max).unwrap_or(usize::MAX));
                    json!({
                        "state": state,
                        "code": code,
                        "offset": from,
                        "end": j.out.end(),
                        "data": frame::b64(&data),
                    })
                }
            };
            http::json_response(200, &body)
        }
        "keep" | "seen" => keep_or_seen(ctx, &job),
        "kill" | "wipe" => {
            let wipe = job.op == "wipe";
            let grace = Duration::from_millis(job.grace_ms.unwrap_or(300));
            // Never a group whose job has ended: its number may be reused.
            let targets: Vec<(Key, JobCell, i32)> = jobs
                .in_scope(&job.scope, |id| wipe || job.ids.iter().any(|i| i == id))
                .into_iter()
                .filter_map(|(k, c)| {
                    let (running, pgid) = {
                        let j = locked(&c);
                        (j.end.is_none(), j.pgid)
                    };
                    running.then_some((k, c, pgid))
                })
                .collect();
            let pgids: Vec<i32> = targets.iter().map(|(_, _, g)| *g).collect();
            stop_groups(&pgids, grace).await;
            let mut survivors = Vec::new();
            for ((_, id), _, g) in &targets {
                let left = proc::live_members_async(*g).await;
                if !left.is_empty() {
                    survivors.push(json!({ "id": id, "pids": left }));
                }
            }
            if wipe || job.remove {
                // Every named job whose group is empty goes, the ended ones included.
                let named = jobs.in_scope(&job.scope, |id| wipe || job.ids.iter().any(|i| i == id));
                let mut gone = Vec::new();
                for (k, c) in named {
                    let pgid = locked(&c).pgid;
                    // pgid 0: still being spawned, so not yet a job that can be taken out.
                    if pgid != 0 && proc::live_members_async(pgid).await.is_empty() {
                        gone.push(k);
                    }
                }
                jobs.remove(&gone);
            }
            jobs.bump();
            http::json_response(200, &json!({ "survivors": survivors }))
        }
        "remove" => {
            let named = jobs.in_scope(&job.scope, |id| job.ids.iter().any(|i| i == id));
            let mut gone = Vec::new();
            for (k, c) in named {
                let (ended, pgid) = {
                    let j = locked(&c);
                    (j.end.is_some(), j.pgid)
                };
                if ended || (pgid != 0 && proc::live_members_async(pgid).await.is_empty()) {
                    gone.push(k);
                }
            }
            jobs.remove(&gone);
            http::json_response(200, &json!({}))
        }
        "list" => {
            let list: Vec<Value> = jobs
                .in_scope(&job.scope, |_| true)
                .into_iter()
                .map(|((_, id), c)| {
                    let j = locked(&c);
                    let (state, code) = state_of(&j);
                    json!({
                        "id": id,
                        "state": state,
                        "code": code,
                        "kept": j.kept,
                        "num": j.num,
                        "kind": j.kind,
                        "seen": j.seen,
                        "started": j.started,
                        "ended": j.ended,
                        "label": j.label,
                    })
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

/// `keep` (the job is the endpoint's now, with its `notify`) and `seen` (how much output the model
/// has been shown).
fn keep_or_seen(ctx: &Arc<Ctx>, job: &JobReq) -> Resp {
    let keep = job.op == "keep";
    // v1.4: a `notify` that fails the grammar refuses the whole `keep`, before anything is
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
        j.kept = true;
        j.num = job.num;
        j.kind = job.kind.clone();
        j.seen = job.seen.unwrap_or(j.seen);
        // The latest keep governs (N6): with notify it replaces the job's (and its proof abandons
        // whatever was in flight), without it removes it.
        let event = exec_notify::Event::Keep {
            notify: notify.is_some(),
        };
        j.notify = notify;
        notify::feed(ctx, &cell, &key, &mut j, event);
    } else {
        j.seen = job.offset;
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
        if map.contains_key(&key) {
            return http::error_response(409, "the job exists");
        }
        // Placeholder until the spawn gives the group: the entry is the claim on the id, so two
        // starts of one id cannot both run.
        let cell = Arc::new(Mutex::new(Job {
            pgid: 0,
            label: job.label.clone(),
            started: epoch(),
            ended: 0,
            out: Ring::new(ctx.opts.job_output_retention, ctx.spool.new_job()),
            shell_exit: None,
            end: None,
            kept: !job.hold,
            num: None,
            kind: (!job.hold).then(|| "background".to_string()),
            seen: 0,
            notify: None,
            notifier: Notifier::default(),
            removed: false,
        }));
        map.insert(key.clone(), cell.clone());
        cell
    };
    let mut cmd = exec::command_for(ctx, req);
    cmd.stdin(Stdio::null());
    let mut proc = match proc::spawn(&mut cmd) {
        Ok(p) => p,
        Err(e) => {
            ctx.jobs.remove(std::slice::from_ref(&key));
            return http::error_response(500, &e.to_string());
        }
    };
    let pgid = proc.pgid;
    locked(&cell).pgid = pgid;
    ctx.groups.insert(pgid);
    ctx.jobs.bump_activity();

    // Both pipes into one ring, in arrival order. `progress` counts what the pumps have read, so
    // the supervisor can tell when they have gone quiet.
    let progress = Arc::new(AtomicUsize::new(0));
    let mut pipes = Vec::new();
    let mut cap = 0;
    if let Some(p) = proc.stdout.take() {
        cap += proc::pipe_capacity(&p);
        pipes.push(pump(ctx.clone(), cell.clone(), p, progress.clone()));
    }
    if let Some(p) = proc.stderr.take() {
        cap += proc::pipe_capacity(&p);
        pipes.push(pump(ctx.clone(), cell.clone(), p, progress.clone()));
    }
    tokio::spawn(supervise(
        ctx.clone(),
        cell.clone(),
        key.clone(),
        proc,
        pipes,
        progress,
        cap,
    ));

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
            let end = stream_job(ctx2.clone(), key, 0, true, Some(pgid), out).await;
            if let Some(o) = &ctx2.opts.observer {
                o.stream_ended(seq, end, std::time::Instant::now());
            }
        });
        resp
    } else {
        http::json_response(200, &json!({ "started": pgid }))
    }
}

/// Read `pipe` into the job's ring until it closes or the job is removed.
fn pump<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    ctx: Arc<Ctx>,
    cell: JobCell,
    mut pipe: R,
    progress: Arc<AtomicUsize>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut buf = vec![0u8; frame::READ_BUF];
        while let Ok(n) = pipe.read(&mut buf).await {
            if n == 0 {
                break;
            }
            {
                let mut j = locked(&cell);
                if j.removed {
                    return;
                }
                j.out.push(&buf[..n]);
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
/// group to empty, record the end. Each is a state change v1.4 calls back about.
async fn supervise(
    ctx: Arc<Ctx>,
    cell: JobCell,
    key: Key,
    mut proc: proc::Proc,
    mut pipes: Vec<tokio::task::JoinHandle<()>>,
    progress: Arc<AtomicUsize>,
    cap: usize,
) {
    let pgid = proc.pgid;
    let status = proc.wait().await;
    let code = status.map_or(-1, |s| {
        use std::os::unix::process::ExitStatusExt as _;
        s.code().unwrap_or_else(|| 128 + s.signal().unwrap_or(0))
    });
    drain_pumps(&mut pipes, &progress, cap).await;
    let mut exits = proc::ChildEvents::new();
    if !proc::group_gone(pgid, false).await {
        // The moment a held stream would send `shell_exit`: v1.4 calls back.
        {
            let mut j = locked(&cell);
            j.shell_exit = Some(code);
            notify::change(&ctx, &cell, &key, &mut j);
        }
        ctx.jobs.bump();
        loop {
            let wake = exits.next().await;
            if proc::group_gone(pgid, wake == proc::Wake::Tick).await {
                break;
            }
        }
        drain_pumps(&mut pipes, &progress, cap).await;
    }
    for p in &pipes {
        p.abort();
    }
    // The end is observable from here on: v1.4 calls back.
    {
        let mut j = locked(&cell);
        j.end = Some(code);
        j.ended = epoch();
        if !j.removed {
            notify::change(&ctx, &cell, &key, &mut j);
        }
    }
    ctx.groups.remove(pgid);
    ctx.jobs.bump();
}

/// Stream a job's output from `offset` to its end. `started`: a held start, which says so first and
/// whose job dies with the connection unless it was kept by then. How the stream ended.
async fn stream_job(
    ctx: Arc<Ctx>,
    key: Key,
    offset: u64,
    hold: bool,
    started: Option<i32>,
    out: mpsc::Sender<Bytes>,
) -> StreamEnd {
    // A job stream has no deadline of its own: it lives as long as the job or the connection.
    let mut sink = Sink::new(out, ctx.opts.heartbeat, None);
    if let Ok(end) = follow(&ctx, &key, offset, started, &mut sink).await {
        return end;
    }
    // The connection is gone before the job's end: a held job nobody kept dies with it.
    if hold && let Some(cell) = ctx.jobs.get(&key) {
        let pgid = {
            let j = locked(&cell);
            (!j.kept && j.end.is_none()).then_some(j.pgid)
        };
        if let Some(g) = pgid {
            proc::kill_group(g);
            return StreamEnd::Disconnected { killed: true };
        }
    }
    StreamEnd::Disconnected { killed: false }
}

/// The frames of a job stream, until its final frame (`Ok`) or the client leaves (`Err`).
async fn follow(
    ctx: &Arc<Ctx>,
    key: &Key,
    offset: u64,
    started: Option<i32>,
    sink: &mut Sink,
) -> Result<StreamEnd, Cut> {
    if let Some(pgid) = started {
        sink.put(frame::json(&json!({ "started": pgid }))).await?;
    }
    let mut changed = ctx.jobs.changed.subscribe();
    let mut sent = offset;
    let mut shell_told = false;
    loop {
        changed.mark_unchanged();
        enum Next {
            Frame(Bytes),
            End(i32),
            Wait,
            Missing,
        }
        let next = match ctx.jobs.get(key) {
            None => Next::Missing,
            Some(cell) => {
                let j = locked(&cell);
                let total = j.out.end();
                if sent < j.out.base {
                    let n = j.out.base - sent;
                    sent = j.out.base;
                    Next::Frame(frame::json(&json!({ "gap": n })))
                } else if sent < total {
                    let (_, piece) = j.out.read(sent, FRAME_MAX);
                    if piece.is_empty() {
                        // The chunk went away under us: the rest is a gap.
                        let n = total - sent;
                        sent = total;
                        Next::Frame(frame::json(&json!({ "gap": n })))
                    } else {
                        sent += piece.len() as u64;
                        Next::Frame(frame::output("stdout", &piece))
                    }
                } else if let (Some(c), false) = (j.shell_exit, shell_told) {
                    shell_told = true;
                    Next::Frame(frame::json(&json!({ "shell_exit": c })))
                } else if let Some(c) = j.end {
                    Next::End(c)
                } else {
                    Next::Wait
                }
            }
        };
        match next {
            Next::Frame(f) => sink.put(f).await?,
            Next::End(c) => {
                sink.put_final(frame::json(&json!({ "exit_code": c })))
                    .await?;
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn ring(retain: usize) -> (Ring, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let r = Ring::new(retain, dir.path().join("job"));
        (r, dir)
    }

    /// Write `n` bytes whose value at stream offset `i` is `i % 251`.
    fn fill(ring: &mut Ring, n: usize) {
        let mut at = 0usize;
        while at < n {
            let k = (n - at).min(10_000);
            let piece: Vec<u8> = (at..at + k).map(|i| (i % 251) as u8).collect();
            ring.push(&piece);
            at += k;
        }
    }

    #[test]
    fn it_keeps_at_least_the_retention_and_never_more_than_a_chunk_over() {
        let (mut ring, _d) = ring(200_000);
        fill(&mut ring, 1_000_000);
        assert_eq!(ring.end(), 1_000_000);
        let kept = ring.end() - ring.base;
        assert!(
            (200_000..200_000 + FRAME_MAX as u64).contains(&kept),
            "kept {kept}"
        );
        assert_eq!(ring.held() as u64, kept);
        // The oldest bytes are gone: a read from 0 starts where the ring does.
        let (from, data) = ring.read(0, 5);
        assert_eq!(from, ring.base);
        assert_eq!(data.len(), 5);
        // Evicted chunks are deleted: only what is held is on disk.
        let n = std::fs::read_dir(&ring.dir).unwrap().count();
        assert_eq!(n, ring.files);
    }

    #[test]
    fn a_read_spans_chunks_and_matches_what_was_written() {
        let (mut ring, _d) = ring(10_000_000);
        fill(&mut ring, 300_000);
        let (from, data) = ring.read(70_000, 200_000);
        assert_eq!(from, 70_000);
        let expect: Vec<u8> = (70_000..270_000usize).map(|i| (i % 251) as u8).collect();
        assert_eq!(data, expect);
        assert!(ring.read(300_000, 10).1.is_empty());
    }

    /// A `max` the client chose (`u64::MAX` is a valid JSON number) must not overflow the arithmetic
    /// of a read from inside a chunk: it panicked while holding the job's lock.
    #[test]
    fn a_read_of_any_size_from_any_offset_does_not_overflow() {
        let (mut ring, _d) = ring(10_000_000);
        fill(&mut ring, 1000);
        for offset in [0, 1, 500, 999, 1000, 5000, u64::MAX] {
            let (from, data) = ring.read(offset, usize::MAX);
            assert_eq!(from, offset, "{offset}");
            assert_eq!(
                data.len() as u64,
                1000u64.saturating_sub(offset),
                "{offset}"
            );
        }
    }

    #[test]
    fn tiny_writes_do_not_make_tiny_chunks_and_a_small_job_touches_no_disk() {
        let (mut ring, d) = ring(10_000_000);
        for _ in 0..100_000 {
            ring.push(b"x");
        }
        assert_eq!(ring.files, 1);
        assert_eq!(ring.tail.len(), 100_000 - FRAME_MAX);
        let (mut small, d2) = self::ring(10_000_000);
        small.push(b"hello");
        assert!(!d2.path().join("job").exists());
        drop((small, d2));
        // Dropping the ring removes its directory.
        let path = ring.dir.clone();
        assert!(path.exists());
        drop(ring);
        assert!(!path.exists());
        drop(d);
    }

    #[test]
    fn files_are_private_and_never_follow_a_link() {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        let (mut ring, _d) = ring(10_000_000);
        ring.push(&vec![7u8; FRAME_MAX]);
        let dir_mode = std::fs::metadata(&ring.dir).unwrap().mode() & 0o777;
        let file_mode = std::fs::metadata(ring.chunk_path(0))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!((dir_mode, file_mode), (0o700, 0o600));
        // A link planted where the next chunk goes is refused, and the bytes read back as a gap.
        let target = ring.dir.join("elsewhere");
        std::os::unix::fs::symlink(&target, ring.chunk_path(FRAME_MAX as u64)).unwrap();
        ring.push(&vec![8u8; FRAME_MAX]);
        assert!(!target.exists());
        assert_eq!(ring.base, 2 * FRAME_MAX as u64);
    }
}
