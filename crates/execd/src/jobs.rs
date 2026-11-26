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
//!   them go, so a session holds at most 20 entries, 160 MiB of output at the floor. An ended job a
//!   client never frees is bounded by the retention ([`crate::Retention`], [`sweep`]): its output
//!   goes after 15 min or past 256 MiB over ended jobs, its entry past 256 ended entries. N concurrent cold callers (N sessions)
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
    self as kernel, Act, Admit, Alive, Attach, Begin, Ev, Follow, Gate, Keeper, Kind, Launch,
    Persist, Pick, Rec, Recovery, Refusal, Removal, Replied, Resizable, Sink as Dest, Size, Sweep,
    Term,
};
use crate::ledger::{Entry, Idem, Ledger, Saved};
use crate::notify::{self, Notifier, NotifyTarget};
use crate::proc::{self, locked};
use crate::ring::Ring;
use crate::tail::{self, Drained, Journal, Spool, Start, Tail};
use crate::term;
use crate::{Ctx, Retention, StreamEnd, StreamStart};

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
    /// What the job runs, as `list` reports it.
    argv: Argv,
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
    /// The start's idempotency key and spec digest, for the ledger (the table's own record of the key is
    /// [`kernel::Rec`]); `None` for an unkeyed start. Written once, at the start or the restart.
    idem: Option<Idem>,
    /// The retention dropped this job's output and kept its entry (a *tombstone*, [`sweep`]): the
    /// ring holds nothing at the end offset, the spool is gone, and the ledger says so
    /// ([`crate::ledger::Entry::out`]), so a restart answers from the same offset. Never unset.
    tombstoned: bool,
}

/// A job's command and arguments (never its environment), kept with it and in the ledger.
#[derive(Clone, Default)]
pub(crate) struct Argv {
    pub command: String,
    pub args: Vec<String>,
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
    /// Wakes the retention sweeper ([`start_retention`]). Coalescing: one `notify_one` per event a
    /// sweep may act on (a job ended, a stream closed, a callback settled, a restart recovered) and
    /// never one per sweep.
    retention: tokio::sync::Notify,
}

impl Jobs {
    pub fn new(activity: Arc<watch::Sender<u64>>, ledger: Ledger) -> Self {
        Self {
            map: Mutex::default(),
            changed: watch::Sender::new(0),
            activity,
            ledger,
            retention: tokio::sync::Notify::new(),
        }
    }

    /// Write the table's jobs to the ledger: every job that has a group and is not removed. Called
    /// with no job's lock held, at each change of a job's life (spawn, shell exit, end, keep,
    /// removal); a no-op without [`crate::Options::state_dir`].
    pub fn save(&self) {
        if !self.ledger.enabled() {
            return;
        }
        self.ledger.save(|| self.entries());
    }

    /// [`Self::save`] for a keyed start's claim: durable, and how it came out (K10).
    fn claim(&self) -> Persist {
        self.ledger.claim(|| self.entries())
    }

    /// What the ledger holds of the table: each job that has a group (or a keyed intent) and is not removed.
    fn entries(&self) -> Vec<Entry> {
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
                    (&scope, &id),
                    &j.label,
                    &j.argv,
                    &j.mark,
                    &j.k,
                    j.spool.as_ref().map(Spool::dir),
                    j.tombstoned.then(|| j.out.end()),
                    keeper,
                    j.idem.as_ref(),
                )
            })
            .collect()
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
        for sp in self.release(cells) {
            sp.remove();
        }
        self.bump();
        self.save();
    }

    /// The table's half of a removal, for cells already taken out of it: the kernel lets each go
    /// (`Ev::Taken`, the only step that sets `removed`), its callback is abandoned and its stdin
    /// closed, and its spool is handed back. No I/O and no ledger write: the caller deletes the
    /// spools and writes the ledger where that is cheap ([`sweep`], off the runtime thread) — which
    /// is what [`Self::remove`] does inline.
    fn release(&self, cells: Vec<(Key, JobCell)>) -> Vec<Spool> {
        let mut spools = Vec::new();
        for (_, cell) in cells {
            let mut j = locked(&cell);
            spools.extend(let_go(&mut j));
        }
        spools
    }

    /// Ask the retention sweeper to look at the table soon ([`start_retention`]). Coalescing: a
    /// second call before the sweep runs costs one stored permit.
    pub(crate) fn wake_retention(&self) {
        self.retention.notify_one();
    }
}

fn epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// Ended jobs: what is kept of one, and for how long.
//
// A job's end is the moment it stops being worth a stream and starts costing the server for as long
// as no client says `remove`: up to its retention of output (8 MiB, plus ≤1 MiB of stderr labels,
// plus <72 KiB in memory) and one table entry, forever. [`sweep`] bounds both, in two tiers, and
// touches nothing else:
//
// 1. **Tombstone** (on by default, 15 min and 256 MiB): drop the output, keep the entry. The entry
//    keeps its offset, `state: "ended"`, code, signal and end, and the ring goes to `base == end`.
//    A late reader gets what it would get past the retention floor anyway: the status, and a gap
//    (`exec_endpoint.rs`, "dropping old output leaves a gap; offsets never move"). `attach` streams
//    one `{"gap": n}` and then the exit.
// 2. **Reap** (a count ceiling of 256, on; a clock, off): take the entry itself, through the same
//    gate `remove` takes (`removable`, `kernel::step(Ev::Remove { ended_counts: true })`,
//    `Table::take` with `Arc::ptr_eq`). Only then does a client get `404 no such job`, which the
//    client kernel calls terminal, and only past a ceiling or a deadline an operator asked for.
//
// Five things are never reaped, and three of them are why this module exists rather than a timer:
//
// - a job with an open stream (`streams > 0`): its output is the only copy of the last bytes, and
//   a take ends the stream early (J9). Registered under the job's own lock, in `attach` and at a
//   held start, so a sweep cannot find it at 0 while an attach is deciding to stream (see `attach`
//   in `serve_op`);
// - a job whose notify callback is owed (`Notifier::owes`): the take abandons the count and the
//   fleet never learns the end;
// - a job that is not `Phase::Ended`: a `Running`/`Exited` job's group can be *observed* empty and
//   fill again (`removable`'s second step), and its leftovers' bytes are still arriving;
// - a **keyed** job, ever: its key's record leaves with it (K8), so a reaped keyed job would let a
//   retried start run the command twice. It is tombstoned, never reaped (ARCHITECTURE.md: "a keyed
//   job must not be evicted");
// - a job whose remaining deadlines are all in the future.
//
// Every step is idempotent and atomic, in this order: the decision and the state change are one
// critical section under the job's lock; the ledger (the durable marker: the entry names no spool,
// and names the offset) is written before anything is unlinked; a crash between the two leaves a
// spool directory the next start deletes (`Spool::retain`). All of it runs off the runtime thread,
// and a sweep writes the ledger once however many jobs it touched.
// ---------------------------------------------------------------------------------------------

/// What the retention does with one ended job now, and why (for the log, and the tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Keep(Why),
    /// Drop its output, keep its entry.
    Tombstone(Why),
    /// Take its entry out.
    Reap(Why),
}

/// Why a job is kept, tombstoned or reaped. `name` is what the log says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Why {
    /// Not `Phase::Ended`.
    Running,
    /// A stream is open on it.
    Attached,
    /// Its notify callback is owed.
    OwedCallback,
    /// A keyed job a reap would have taken: it keeps its entry (a liveness gap, logged).
    Keyed,
    /// Its output deadline arrived.
    OutputTtl,
    /// Its entry deadline arrived.
    EntryTtl,
    /// The server holds more ended entries than the ceiling.
    Count,
    /// More output is retained over ended jobs than the budget.
    Bytes,
    /// Nothing is due for it.
    WithinDeadline,
    /// The gate refused a take the policy asked for (never happens for an ended job, L1).
    NotRemovable,
}

impl Why {
    fn name(self) -> &'static str {
        match self {
            Why::Running => "running",
            Why::Attached => "attached",
            Why::OwedCallback => "owed_callback",
            Why::Keyed => "keyed",
            Why::OutputTtl => "output_ttl",
            Why::EntryTtl => "entry_ttl",
            Why::Count => "count",
            Why::Bytes => "bytes",
            Why::WithinDeadline => "within_deadline",
            Why::NotRemovable => "not_removable",
        }
    }
}

/// The kernel's `Taken` step for a cell the table has let go, and what the shell does with it: the
/// callback is abandoned, stdin closed, the spool handed back. `removed` is set here, final, and it
/// is what an `attach` that lost the race with a take answers `404` for; idempotent, so a second
/// `Taken` (a retried release) does nothing.
fn let_go(j: &mut Job) -> Option<Spool> {
    if let Act::Release { .. } = kernel::step(&mut j.k, Ev::Taken) {
        j.notifier = Notifier::removed();
        if let Some(s) = &j.stdin {
            s.close();
        }
        return j.spool.take();
    }
    None
}

/// The two ceilings, as the sweep walks the ended jobs oldest first and both move as it acts: what
/// is over when this job is looked at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pressure {
    over_bytes: bool,
    over_count: bool,
}

/// The vetoes, in one place for [`decide`] and [`next_wake`]: the moment the job ended
/// (epoch seconds) when the retention may act on it at all, or why it may not. Each line is the one
/// protection a test in `tests/retention.rs` (or the unit tests below) deletes with a `mut_*` feature
/// and watches go red.
fn candidate(j: &Job) -> Result<u64, Why> {
    // Only an ended job. Not `pgid != 0`, not "its group looks empty": a job the kernel has not
    // ended can still gain a process or a byte.
    let at = match j.k.phase {
        kernel::Phase::Ended { at, .. } => at,
        _ if cfg!(feature = "mut_retention_any_phase") => j.k.started,
        _ => return Err(Why::Running),
    };
    if j.streams > 0 && !cfg!(feature = "mut_retention_ignores_streams") {
        return Err(Why::Attached);
    }
    if j.notifier.owes() && !cfg!(feature = "mut_retention_ignores_owed") {
        return Err(Why::OwedCallback);
    }
    Ok(at)
}

/// The policy: what the retention does with `j` at `now` (epoch seconds), given the ceilings. Pure,
/// so the tests drive it over every [`kernel::Phase`] with no table, no socket and no clock.
fn decide(j: &Job, now: u64, ret: &Retention, p: Pressure) -> Step {
    let at = match candidate(j) {
        Ok(at) => at,
        Err(why) => return Step::Keep(why),
    };
    let age_ms = now.saturating_sub(at).saturating_mul(1000);
    let entry_due = ret.entry_ms > 0 && age_ms >= ret.entry_ms;
    // Output first, whatever the numbers say: an entry deadline never lets output outlive it.
    let output_due = entry_due || (ret.output_ms > 0 && age_ms >= ret.output_ms);
    let holds = j.out.base < j.out.end() || j.spool.is_some();
    let keyed = j.idem.is_some() && !cfg!(feature = "mut_retention_reaps_keyed");
    let reap = if entry_due {
        Some(Why::EntryTtl)
    } else if p.over_count {
        Some(Why::Count)
    } else {
        None
    };
    if let Some(why) = reap.filter(|_| !keyed) {
        return Step::Reap(why);
    }
    if holds {
        if output_due {
            return Step::Tombstone(Why::OutputTtl);
        }
        if p.over_bytes {
            return Step::Tombstone(Why::Bytes);
        }
    }
    match reap {
        // Everything a keyed job may lose is lost, and it keeps its entry.
        Some(_) => Step::Keep(Why::Keyed),
        None => Step::Keep(Why::WithinDeadline),
    }
}

/// What one sweep did, for the log and the tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Swept {
    /// Ended jobs in the table when it began, and the output they held.
    pub ended: usize,
    pub bytes: u64,
    pub tombstoned: usize,
    pub reaped: usize,
    /// Ended jobs a veto or a future deadline kept, by reason.
    pub kept: std::collections::BTreeMap<&'static str, usize>,
}

impl Swept {
    fn keep(&mut self, why: Why) {
        *self.kept.entry(why.name()).or_default() += 1;
    }

    fn acted(&self) -> bool {
        self.tombstoned + self.reaped > 0
    }
}

/// The most jobs one sweep acts on, so a pass never holds the table for long or hands the blocking
/// pool a burst: what is left is due again, and the sweeper looks again at once.
const SWEEP_ACTIONS_MAX: usize = 64;

/// The least the sweeper sleeps between two looks, so a job due but vetoed (or one the gate
/// refused) can never spin.
const SWEEP_MIN_SLEEP_MS: u64 = 50;

/// The most ended rows one `list` answer carries ([`serve_op`]). An ended entry is cheap to keep and
/// the ceiling lets a table hold thousands, while the answer is rebuilt on every reattach: past this
/// many the newest are listed and the rest are only counted (`"more"`). At 4096 ended entries the
/// answer would otherwise be 1.45 MB of output and 37 ms of the runtime thread, measured
/// (`benches/retention_scale.py`).
pub(crate) const LIST_ENDED_MAX: usize = 512;

/// One pass of the retention: drop the output of the ended jobs the policy names, reap the entries
/// it names, then — in **one** pass, off the runtime thread — write the ledger and delete every
/// file no entry names any more.
///
/// Callers: [`start_retention`]. It may run concurrently with `remove`, `kill`, a `start` and a
/// restart: every take goes through `Table::take` with `Arc::ptr_eq` (J10), so a job a client took
/// first is simply not found, and one a client started under the same id afterwards is not the one
/// this sweep gated.
pub(crate) async fn sweep(ctx: &Arc<Ctx>) -> Swept {
    let jobs = &ctx.jobs;
    let ret = &ctx.opts.retention;
    let now = epoch();
    // The ended jobs, oldest first: the order both ceilings act in. A snapshot, with the map guard
    // held while each cell is read (the order `entries` takes them: map, then cell).
    // Every job is looked at (the phase is [`candidate`]'s to judge, and only there); the ceilings
    // count the ended ones.
    let mut ended: Vec<(Key, JobCell, u64)> = {
        let table = locked(&jobs.map);
        let cells = table.all();
        cells
            .into_iter()
            .map(|(key, cell)| {
                let at = locked(&cell).k.ended_at();
                (key, cell, at)
            })
            .collect()
    };
    ended.sort_by_key(|&(_, _, at)| at);
    let (mut count, mut bytes) = (0usize, 0u64);
    for (_, cell, _) in &ended {
        let j = locked(cell);
        if j.k.ended() {
            count += 1;
            bytes += j.out.end() - j.out.base;
        }
    }
    let mut sweep = Swept {
        ended: count,
        bytes,
        ..Swept::default()
    };
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let mut spools: Vec<Spool> = Vec::new();
    {
        let mut table = locked(&jobs.map);
        for (key, cell, _) in &ended {
            if sweep.tombstoned + sweep.reaped >= SWEEP_ACTIONS_MAX {
                break;
            }
            let mut j = locked(cell);
            let step = decide(
                &j,
                now,
                ret,
                Pressure {
                    over_bytes: ret.bytes > 0 && bytes > ret.bytes,
                    over_count: ret.max > 0 && count > ret.max,
                },
            );
            let freed = j.out.end() - j.out.base;
            match step {
                Step::Keep(why) => {
                    if !matches!(why, Why::Running) {
                        sweep.keep(why);
                    }
                    if !matches!(why, Why::Running | Why::WithinDeadline) {
                        crate::log_debug!(
                            "job retained scope={:?} id={:?} reason={}",
                            key.0,
                            key.1,
                            why.name()
                        );
                    }
                }
                Step::Tombstone(why) => {
                    let files = j.out.evict_all();
                    let spool = j.spool.take();
                    j.tombstoned = true;
                    drop(j);
                    paths.extend(files);
                    spools.extend(spool);
                    bytes = bytes.saturating_sub(freed);
                    sweep.tombstoned += 1;
                    crate::log_debug!(
                        "job output dropped scope={:?} id={:?} reason={} freed_bytes={freed}",
                        key.0,
                        key.1,
                        why.name()
                    );
                }
                Step::Reap(why) => {
                    // The same gate `remove` takes, asked and committed in one critical section:
                    // `Ev::Remove` cannot mutate (a refusal leaves nothing behind), the take keeps
                    // only the cell that was checked (J10), and `let_go` sets `removed` before this
                    // lock is dropped — so an attach waiting on it finds a job the table no longer
                    // has and answers `404`, never a stream of a job that is gone.
                    let gate = kernel::step(
                        &mut j.k,
                        Ev::Remove {
                            empty: false,
                            ended_counts: true,
                        },
                    );
                    if gate != Act::Take {
                        sweep.keep(Why::NotRemovable);
                        continue;
                    }
                    let gone = table.take(
                        std::slice::from_ref(&(key.clone(), cell.clone())),
                        Arc::ptr_eq,
                    );
                    if gone.is_empty() {
                        sweep.keep(Why::NotRemovable);
                        continue;
                    }
                    debug_assert_eq!(gone.len(), 1);
                    spools.extend(let_go(&mut j));
                    drop(j);
                    bytes = bytes.saturating_sub(freed);
                    count = count.saturating_sub(1);
                    sweep.reaped += 1;
                    crate::log_debug!(
                        "job entry reaped scope={:?} id={:?} reason={} freed_bytes={freed}",
                        key.0,
                        key.1,
                        why.name()
                    );
                }
            }
        }
    }
    crate::log_debug!(
        "retention sweep ended={} bytes={} tombstoned={} reaped={} kept={:?}",
        sweep.ended,
        sweep.bytes,
        sweep.tombstoned,
        sweep.reaped,
        sweep.kept
    );
    if sweep.tombstoned + sweep.reaped >= SWEEP_ACTIONS_MAX {
        // The pass stopped at its cap with work left: the next one runs as soon as this one is
        // written, not at the next deadline (a ceiling has none).
        jobs.wake_retention();
    }
    if sweep.acted() {
        jobs.bump();
        // One pass off the runtime thread, holding one of `disk_writers` turns (the same gate as a
        // job's output batches, `Ctx::disk_turn`): the ledger first (the durable marker — this
        // entry's output is gone and its offsets are these — then the files it no longer names),
        // and every unlink here. The order is the whole point: a crash between the two leaves a
        // ledger that names no file, which the next start's `retain` finishes, never a file no
        // ledger names (`a_crash_after_the_ledger_write_leaves_no_spool_behind`).
        let turn = ctx.disk_turn().await;
        let ctx = ctx.clone();
        proc::off_runtime(move || {
            ctx.jobs.save();
            if crate::fault::at("sweep_after_save").as_deref() == Some("exit") {
                // A test's crash between the durable marker and the unlinks.
                std::process::exit(86);
            }
            for sp in spools {
                sp.remove();
            }
            for path in paths {
                let _ = std::fs::remove_file(path);
            }
        })
        .await;
        drop(turn);
        crate::log_info!(
            "retention sweep acted ended={} bytes={} tombstoned={} reaped={} kept={:?}",
            sweep.ended,
            sweep.bytes,
            sweep.tombstoned,
            sweep.reaped,
            sweep.kept
        );
    } else if ret.bytes > 0 && sweep.bytes > ret.bytes {
        // Every job holding output is vetoed (attached, owed a callback) or keyed and already
        // tombstoned: said once per sweep, and a sweep runs only on an event, never on a timer for
        // a vetoed job, so this cannot spin.
        crate::log_warn!(
            "retention cannot meet the output budget ended={} bytes={} limit={} kept={:?}",
            sweep.ended,
            sweep.bytes,
            ret.bytes,
            sweep.kept
        );
    }
    sweep
}

/// When the retention must look at the table again, or `None` when no ended job has a deadline
/// ahead and no ceiling is over. Only jobs no veto blocks: a vetoed job is woken by its veto's own
/// release (the stream closing, the callback settling), so nothing is polled.
fn next_wake(jobs: &Jobs, ret: &Retention, now: u64) -> Option<tokio::time::Instant> {
    if !ret.enabled() {
        return None;
    }
    let mut best: Option<u64> = None;
    let table = locked(&jobs.map);
    let cells = table.all();
    drop(table);
    for (_, cell) in cells {
        let j = locked(&cell);
        // A vetoed job is woken by its veto's own release (the stream closing, the callback
        // settling), never by a timer: nothing is polled.
        let Ok(at) = candidate(&j) else {
            continue;
        };
        let age_ms = now.saturating_sub(at).saturating_mul(1000);
        let holds = j.out.base < j.out.end() || j.spool.is_some();
        let mut deadlines = Vec::with_capacity(2);
        if holds && ret.output_ms > 0 {
            deadlines.push(ret.output_ms);
        }
        if (j.idem.is_none() || cfg!(feature = "mut_retention_reaps_keyed")) && ret.entry_ms > 0 {
            deadlines.push(ret.entry_ms);
        }
        for deadline in deadlines {
            let left = deadline.saturating_sub(age_ms);
            best = Some(best.map_or(left, |b: u64| b.min(left)));
        }
    }
    best.map(|ms| tokio::time::Instant::now() + Duration::from_millis(ms.max(SWEEP_MIN_SLEEP_MS)))
}

/// The retention's one task: asleep until a job ends, a vetoed job's veto lifts, or a deadline
/// arrives, and then one [`sweep`]. Started by [`crate::serve`] only when one of the four knobs is
/// on, and cancelled with the server, so a deployment with today's configuration pays no wakeup and
/// no task for it at all.
pub(crate) fn start_retention(ctx: &Arc<Ctx>, stop: CancellationToken) {
    if !ctx.opts.retention.enabled() {
        return;
    }
    let ctx = ctx.clone();
    tokio::spawn(async move {
        loop {
            let wake = next_wake(&ctx.jobs, &ctx.opts.retention, epoch());
            tokio::select! {
                _ = stop.cancelled() => return,
                // A job ended, a stream closed, a callback settled: look now.
                _ = ctx.jobs.retention.notified() => {}
                _ = async {
                    match wake {
                        Some(at) => tokio::time::sleep_until(at).await,
                        None => std::future::pending().await,
                    }
                } => {}
            }
            sweep(&ctx).await;
        }
    });
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
    /// `attach`, `remove`: the incarnation the request means (the `mark` a start answered). `attach`
    /// takes one, `remove` one per id (`marks`); a job whose mark differs is not touched.
    mark: Option<String>,
    marks: std::collections::HashMap<String, String>,
    /// `attach`: stream the job's output (default), or only its end (`false`).
    output: Option<bool>,
    remove: bool,
    grace_ms: Option<u64>,
    num: Option<u64>,
    kind: Option<String>,
    seen: Option<u64>,
    notify: Option<Value>,
    /// `start`: keep the job's stdin open for `write` (default: `/dev/null`).
    stdin_open: bool,
    /// `start`: the idempotency key. A start that carries the key of an earlier start of the same spec
    /// answers that job and runs nothing; the same key with another spec is `409`.
    idempotency_key: Option<String>,
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
            // A test parks the attach between the lookup and the decision (`attach_before_decide`) to
            // prove a job taken in between answers `404`, never a stream of a job that is gone.
            crate::fault::park("attach_before_decide").await;
            let output = job.output.unwrap_or(true);
            // The mark and the stream's start are the kernel's (M1, M4 to M6), read under one lock —
            // and the stream is registered in that same lock (`Watching::open`), so the retention's
            // attached-stream veto never sees this job at `streams == 0` after it has been told to
            // stream. An attach that loses to a take finds no job and answers `404 no such job`, the
            // defined answer.
            let from = {
                let mut j = locked(&cell);
                // The table let this job go between the lookup above and this lock (a `remove`, or
                // a sweep reaping it): the defined answer is `404 no such job`, and a client that
                // retries never gets a stream of a job that is already gone.
                if j.k.removed && !cfg!(feature = "mut_attach_ignores_removed") {
                    return http::error_response(404, "no such job");
                }
                let ok = kernel::mark_ok(&job.mark, &j.mark);
                match kernel::attach_plan(ok, output, job.offset, j.out.end()) {
                    Attach::Refuse => None,
                    Attach::Stream { from } => Some((
                        from,
                        j.mark.clone(),
                        (!cfg!(feature = "mut_attach_registers_late"))
                            .then(|| Watching::open(ctx, &cell, &mut j)),
                    )),
                }
            };
            let Some((from, mark, watching)) = from else {
                return http::error_response(404, "no such job");
            };
            // The window this closes, made testable: a test parks the attach here
            // (`EXECD_TEST_FAULT`, `attach_after_decide`) and proves a sweep cannot take the job
            // while it is parked.
            crate::fault::park("attach_after_decide").await;
            // Today's order, kept only as the mutant that proves the window is real: the count is
            // taken here, after the decision, where a sweep can run between the two.
            let watching = watching.unwrap_or_else(|| Watching::new(ctx, &cell));
            let (resp, out) = http::stream_response();
            tokio::spawn(stream_job(
                ctx.clone(),
                k,
                cell,
                watching,
                false,
                Opening {
                    offset: from,
                    output,
                    first: Some(json!({ "attached": { "mark": mark, "from": from } })),
                },
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
            let (mut gone, mut running, mut mismatch) = (Vec::new(), Vec::new(), Vec::new());
            for (k, c) in named {
                let ok = {
                    let j = locked(&c);
                    kernel::mark_ok(&job.marks.get(&k.1).cloned(), &j.mark)
                };
                let take = removable(&c, true).await;
                match kernel::removal(ok, take) {
                    Removal::Take => gone.push((k, c)),
                    Removal::Running => running.push(k.1),
                    Removal::Mismatch => mismatch.push(k.1),
                }
            }
            jobs.remove(&gone);
            http::json_response(200, &json!({ "running": running, "mismatch": mismatch }))
        }
        "list" => {
            // Every job that has not ended answers in full; the ended ones past [`LIST_ENDED_MAX`]
            // are counted, not listed. The newest rows are the ones kept: the count ceiling reaps the
            // oldest first, and a client that reattaches wants what just happened. A table of 256
            // ended entries — what the old ceiling held — is answered byte for byte as before, since
            // `more` is absent unless rows were left out.
            let cells = jobs.in_scope(&job.scope, Pick::All);
            let mut ended: Vec<(u64, usize)> = Vec::new();
            for (i, (_, cell)) in cells.iter().enumerate() {
                let j = locked(cell);
                if j.k.ended() {
                    ended.push((j.k.ended_at(), i));
                }
            }
            let mut dropped = vec![false; cells.len()];
            if ended.len() > LIST_ENDED_MAX {
                ended.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                for &(_, i) in &ended[LIST_ENDED_MAX..] {
                    dropped[i] = true;
                }
            }
            let more = ended.len().saturating_sub(LIST_ENDED_MAX);
            let list: Vec<Value> = cells
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !dropped[*i])
                .map(|(_, ((_, id), c))| {
                    let j = locked(&c);
                    let (state, code) = state_of(&j);
                    let mut entry = json!({
                        "id": id,
                        "mark": j.mark,
                        "command": j.argv.command,
                        "args": j.argv.args,
                        "outputStart": j.out.base,
                        "outputEnd": j.out.end(),
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
            let mut body = json!({ "jobs": list });
            if more > 0 {
                body["more"] = json!(more);
            }
            http::json_response(200, &body)
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
    let max = ctx.opts.terminals_max.map(|m| m as u64);
    if max.is_some() {
        // The start itself is in the table already.
        let open = locked(&ctx.jobs.map)
            .all()
            .into_iter()
            .filter(|(_, c)| matches!(locked(c).k.term, Term::Open { .. }))
            .count() as u64;
        if kernel::pty_cap(open, max) == kernel::PtyBegin::NoPty {
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
        let mut attempt = Box::pin(proc::off_runtime(move || {
            term::connect_sync(&d, pid)
                .ok()
                .and_then(term::Met::answered)
        }));
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
                    return Err(
                        match kernel::keeper_ended(st.as_ref().ok().and_then(|s| s.code())) {
                            kernel::PtyBegin::NoPty => (503, "no pseudo-terminal is free"),
                            _ => (500, "cannot start the terminal"),
                        },
                    );
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
    if !kernel::id_ok(job.id.as_bytes()) {
        return http::error_response(400, "bad job id: 1 to 64 characters of [0-9a-z_-]");
    }
    let pty = match pty_size(job.pty.as_ref()) {
        Ok(p) => p,
        Err(()) => {
            return http::error_response(400, "pty: cols and rows are integers from 1 to 1024");
        }
    };
    let key = (job.scope.clone(), job.id.clone());
    let idem = match job.idempotency_key.as_deref() {
        None => None,
        Some(k) if kernel::key_valid(k.as_bytes()) => Some(Idem {
            key: k.to_string(),
            digest: spec_digest(req, job),
        }),
        Some(_) => return http::error_response(400, "bad idempotency key"),
    };
    /// What the table's decision leaves to do once its lock is dropped.
    enum Gate {
        Run(JobCell),
        Answer(Resp),
        Replay(Key, Option<JobCell>),
    }
    let gate = {
        let mut map = locked(&ctx.jobs.map);
        // The kernel's decision (K1-K5): a key already used for this command replays its job, a key used
        // for another command is refused, and an id in the table is the claim.
        let begin = map.begin(
            &key.0,
            &key.1,
            idem.as_ref().map(|i| i.key.as_str()),
            idem.as_ref().map_or("", |i| i.digest.as_str()),
        );
        match begin {
            Begin::Refuse(Refusal::IdExists) => {
                Gate::Answer(http::error_response(409, "the job exists"))
            }
            Begin::Refuse(Refusal::KeyReused) => Gate::Answer(http::json_response(
                409,
                &json!({
                    "error": "the key was used for a different command or id",
                    "code": "key_conflict",
                }),
            )),
            Begin::Replay(id) => {
                let replayed = (job.scope.clone(), id);
                let cell = map.get(&replayed);
                Gate::Replay(replayed, cell)
            }
            Begin::Start => {
                // Placeholder until the spawn gives the group: the entry is the claim on the id, so two
                // starts of one id cannot both run.
                let cell = Arc::new(Mutex::new(Job {
                    mark: proc::new_job_mark(ctx.next_seq.fetch_add(1, Ordering::Relaxed)),
                    sp: 0,
                    label: job.label.clone(),
                    argv: Argv {
                        command: req.command.clone(),
                        args: req.args.clone(),
                    },
                    out: Ring::new(ctx.opts.job_output_retention, ctx.spool.new_job()),
                    spool: None,
                    k: kernel::State::new(epoch(), job.hold, pty),
                    notify: None,
                    notifier: Notifier::default(),
                    stdin: None,
                    streams: 0,
                    idem: idem.clone(),
                    tombstoned: false,
                }));
                map.insert(key.clone(), cell.clone());
                // The key's record goes in with the entry, under the same lock: no other start sees the id
                // without it.
                if let Some(i) = &idem {
                    map.record(Rec {
                        scope: key.0.clone(),
                        key: i.key.clone(),
                        id: key.1.clone(),
                        digest: i.digest.clone(),
                    });
                }
                Gate::Run(cell)
            }
        }
    };
    let cell = match gate {
        Gate::Run(cell) => cell,
        Gate::Answer(resp) => return resp,
        Gate::Replay(replayed, cell) => {
            return replay_job(ctx, req, job, replayed, cell).await;
        }
    };
    // A keyed start's claim is durable before anything is spawned (K9, K10): a crash from here to the group's save
    // leaves an intent a restart answers as lost, never a start to run again. A claim that cannot be written is a
    // start that does not run: the kernel says which refusal, and the retry, once the disk takes it, starts.
    match kernel::launch(idem.is_some().then(|| ctx.jobs.claim())) {
        Launch::Spawn => {}
        Launch::DiskFull => {
            ctx.jobs.remove(&[(key, cell)]);
            return http::error_response(
                507,
                "the sandbox's disk is full: no room to record the start",
            );
        }
        Launch::Failed => {
            ctx.jobs.remove(&[(key, cell)]);
            return http::error_response(500, "cannot record the start");
        }
    }
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
    if crate::fault::at("after_spawn").as_deref() == Some("exit") {
        // A test's crash of execd between the spawn and the group's save (K11).
        std::process::exit(86);
    }
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
    let first_frame = started_json(pgid, &mark, opened, pty, job.idempotency_key.as_deref());
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
        let first = first_frame.clone();
        // Registered before the stream is spawned, as `attach` does: the job is spawning, and a held
        // stream on an ended job is exactly what the retention must never take.
        let watching = Watching::new(ctx, &cell);
        tokio::spawn(async move {
            let end = stream_job(
                ctx2.clone(),
                key,
                cell,
                watching,
                true,
                Opening {
                    offset: 0,
                    output: true,
                    first: Some(first),
                },
                Sink::new(out, beat, None),
            )
            .await;
            if let Some(o) = &ctx2.opts.observer {
                o.stream_ended(seq, end, std::time::Instant::now());
            }
        });
        resp
    } else {
        http::json_response(200, &first_frame)
    }
}

/// The digest of what a start runs: command, args, cwd, env, stdin_open and hold, length-prefixed so no two
/// specs share a byte string, hashed with SHA-256. This is all execd keeps of a keyed start's spec: the env can
/// hold secrets, so the spec itself is never stored. The label is not in it (a name, not a command). Every field
/// that decides what runs is here: a field added to `start` that changes the command belongs in this list.
fn spec_digest(req: &Request, job: &JobReq) -> String {
    use sha2::{Digest as _, Sha256};
    let mut h = Sha256::new();
    let mut field = |tag: u8, bytes: &[u8]| {
        h.update([tag]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    };
    field(b'c', req.command.as_bytes());
    for a in &req.args {
        field(b'a', a.as_bytes());
    }
    match &req.cwd {
        Some(c) => field(b'd', c.as_bytes()),
        None => field(b'D', b""),
    }
    // The environment is a set: the order a client wrote it in is not part of the spec.
    let mut env: Vec<&(String, String)> = req.env_pairs.iter().collect();
    env.sort();
    for (k, v) in env {
        field(b'k', k.as_bytes());
        field(b'v', v.as_bytes());
    }
    field(b'i', &[u8::from(job.stdin_open)]);
    field(b'h', &[u8::from(job.hold)]);
    let out = h.finalize();
    out.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The answer to a start the key ledger already has: the original job's, as its start answered (its group, mark and
/// `stdin`; the key, echoed), or, for a held start, its stream from the first byte. Nothing is spawned. A replay that
/// arrives while the original is still spawning waits for its group, so it never answers a job with no process; what
/// it answers at each look is the kernel's (K14, K15). `cell` is `None` when the table no longer holds the job: the
/// same refusal as a job removed before it started.
async fn replay_job(
    ctx: &Arc<Ctx>,
    req: &Request,
    job: &JobReq,
    key: Key,
    cell: Option<JobCell>,
) -> Resp {
    let mut waited = 0;
    let (cell, pgid) = loop {
        let (pgid, ended, removed) = match &cell {
            Some(c) => {
                let j = locked(c);
                (j.k.pgid(), j.k.ended(), j.k.removed)
            }
            None => (0, false, true),
        };
        match (kernel::replay_poll(pgid, ended, removed, waited), &cell) {
            // A job a restart found claimed and never spawned is ended with no group: its answer is that.
            (Replied::Answer, Some(c)) => break (c.clone(), pgid),
            (Replied::Wait, _) => {
                waited += 1;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            // A refusal, or (never: no job is no group, so the kernel refuses it) an answer with no job: fail closed.
            _ => return http::error_response(500, "the job did not start"),
        }
    };
    // The replay answers as the first start did, with the same incarnation's mark and terminal.
    let (mark, pty) = {
        let j = locked(&cell);
        let pty = match j.k.term {
            Term::Open { size, .. } | Term::Gone { size } => Some(size),
            Term::No => None,
        };
        (j.mark.clone(), pty)
    };
    let opened = job.stdin_open || pty.is_some();
    let first = started_json(pgid, &mark, opened, pty, job.idempotency_key.as_deref());
    if job.hold {
        let (resp, out) = http::stream_response();
        let ctx2 = ctx.clone();
        let beat = req.beat;
        let watching = Watching::new(ctx, &cell);
        tokio::spawn(async move {
            stream_job(
                ctx2,
                key,
                cell,
                watching,
                true,
                Opening {
                    offset: 0,
                    output: true,
                    first: Some(first),
                },
                Sink::new(out, beat, None),
            )
            .await;
        });
        resp
    } else {
        http::json_response(200, &first)
    }
}

/// What a start answers: the job's process group, its incarnation's mark, `stdin` when its stdin was opened, the
/// terminal's size when it has one, and the start's key when it had one: the echo that tells a client its key was
/// kept (an execd that ignores keys says nothing). A held start's first frame is the same value.
fn started_json(
    pgid: i32,
    mark: &str,
    opened: bool,
    pty: Option<Size>,
    key: Option<&str>,
) -> Value {
    let mut body = json!({ "started": pgid, "mark": mark });
    if opened {
        body["stdin"] = json!(true);
    }
    if let Some(size) = pty {
        body["pty"] = json!({ "cols": size.cols, "rows": size.rows });
    }
    if let Some(k) = key {
        body["key"] = json!(k);
    }
    body
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
    tail: Mutex<Tail>,
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
            tail: Mutex::new(tail),
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
    ///
    /// With `disk_writers` > 0 the reads and every disk write behind them (chunks, journal,
    /// punching) run off the runtime, a batch of [`tail::ROUNDS`] rounds at a time, each batch
    /// holding one of `disk_writers` permits, so a slow disk stalls this job's tail and nobody else
    /// and the threads are bounded however many jobs write. A batch that finds no permit waits for
    /// one (logged at debug). With `0` it is what ran before: one round at a time on the runtime,
    /// yielding between rounds.
    async fn flush(self: &Arc<Self>) -> bool {
        let threaded = self.ctx.opts.disk_writers > 0;
        loop {
            let drained = if threaded {
                let slot = match self.ctx.disk_slots.try_acquire() {
                    Ok(slot) => slot,
                    Err(_) => {
                        crate::log_debug!(
                            "tail waits for a writer slot mark={}",
                            locked(&self.cell).mark
                        );
                        match self.ctx.disk_slots.acquire().await {
                            Ok(slot) => slot,
                            // Closed only with the server, which never happens.
                            Err(_) => return false,
                        }
                    }
                };
                let this = self.clone();
                let drained = proc::off_runtime(move || this.drain_batch(tail::ROUNDS)).await;
                drop(slot);
                drained
            } else {
                self.drain_batch(1)
            };
            match step_after(drained, threaded) {
                TailStep::Again => {}
                TailStep::AgainYielding => tokio::task::yield_now().await,
                TailStep::Stop(kept) => return kept,
            }
        }
    }

    fn drain_batch(&self, rounds: usize) -> Drained {
        let mut tail = locked(&self.tail);
        let mut gated = false;
        tail.drain_batch(rounds, |data, stderr| {
            if !gated {
                gated = true;
                tail_gate();
            }
            let tail_len = {
                let mut j = locked(&self.cell);
                if kernel::step(&mut j.k, Ev::Push) == Act::Discard {
                    return false;
                }
                j.out.push(data, stderr);
                j.out.tail_len()
            };
            // Before the spool's bytes are returned (the batch punches after this returns).
            locked(&self.journal).record(stderr, data, tail_len);
            self.ctx.jobs.bump();
            true
        })
    }
}

/// The test seam for a slow disk (`EXECD_TEST_FAULT` `tail_gate=<dir>`, debug builds only): a batch
/// with output to write holds here, with a file `<dir>/held-<n>` naming it, for as long as
/// `<dir>/closed` exists. A disk that never answers, released when the test says. The most batches
/// ever held at once is kept in `<dir>/most`.
fn tail_gate() {
    let Some(dir) = crate::fault::at("tail_gate").map(std::path::PathBuf::from) else {
        return;
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    static NOW: AtomicUsize = AtomicUsize::new(0);
    static MOST: Mutex<usize> = Mutex::new(0);
    let held = dir.join(format!("held-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::write(&held, b"");
    let now = NOW.fetch_add(1, Ordering::AcqRel) + 1;
    {
        let mut most = locked(&MOST);
        if now > *most {
            *most = now;
            let _ = std::fs::write(dir.join("most"), now.to_string());
        }
    }
    while dir.join("closed").exists() {
        std::thread::sleep(Duration::from_millis(5));
    }
    NOW.fetch_sub(1, Ordering::AcqRel);
    let _ = std::fs::remove_file(held);
}

/// What a tail does after one batch: read the next one, or stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailStep {
    /// Read the next batch at once.
    Again,
    /// Read the next batch after giving the runtime a turn (the round-at-a-time path).
    AgainYielding,
    /// Stop; `true` keeps the job's output, `false` means the job was removed mid-batch.
    Stop(bool),
}

/// [`TailStep`] for a batch that ended [`Drained`]. A batch that filled its rounds with more to read is
/// never the end: the spool holds bytes that no chunk and no journal hold yet, so stopping there
/// loses them until something else happens to wake the tail (a punch, or the next write). Only a
/// batch that read both files to their ends ([`Drained::Idle`]) or was refused
/// ([`Drained::Stopped`]) stops.
fn step_after(drained: Drained, threaded: bool) -> TailStep {
    match drained {
        Drained::More if threaded => TailStep::Again,
        Drained::More => TailStep::AgainYielding,
        Drained::Idle => TailStep::Stop(true),
        Drained::Stopped => TailStep::Stop(false),
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
    let mut ended = false;
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
            // The end is what makes a job a candidate for the retention (a deadline, or the count
            // ceiling, which has no clock): look when nothing else is owed.
            ended = true;
        }
    }
    for g in release {
        ctx.groups.remove(*g);
    }
    ctx.jobs.bump();
    ctx.jobs.save();
    if ended {
        ctx.jobs.wake_retention();
    }
}

/// How many times a restart's sweep of a lost claim's marked processes looks again after a kill.
const SWEEP_PASSES: usize = 5;

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
        let dir = e.spool.as_deref().map(Path::new);
        let met = match (&e.term, dir) {
            (Some(t), Some(dir)) if alive && t.open && e.phase != Saved::Ended => {
                term::connect_sync(dir, t.keeper_pid).ok()
            }
            _ => None,
        };
        let status = dir.filter(|_| e.term.is_some()).and_then(term::read_status);
        let keeper =
            kernel::keeper_of(e.term.is_some(), met.as_ref().map(term::Met::reply), status);
        if let (Keeper::Answered { .. }, Some(t), Some(m)) = (keeper, &e.term, met)
            && let Some((stream, term::Said::Ready { written, .. }, frames)) = m.ready
        {
            conn = Some((stream, written, frames, t.keeper_pid));
        }
        let (mut state, how, sweep) =
            kernel::recover(e.state(), alive, keeper, epoch(), e.idem.is_some());
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
        if sweep == Sweep::KillMarked {
            // The intent's spawn may have happened and its save not: nobody can find its processes but by the
            // mark, and its answer is lost. Never leave them running unseen (K11). A process still between its
            // fork and its exec carries execd's environment, not the mark, so the sweep looks again until a pass
            // finds none.
            for _ in 0..SWEEP_PASSES {
                let marked = proc::job_members(0, &e.mark);
                if marked.is_empty() {
                    break;
                }
                for pid in marked {
                    if let Ok(raw) = i32::try_from(pid) {
                        let _ = nix::sys::signal::kill(
                            nix::unistd::Pid::from_raw(raw),
                            Signal::SIGKILL,
                        );
                    }
                }
            }
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
        // A job whose output the previous execd dropped keeps its offsets, so a client's reads never
        // go backwards; one whose spool the ledger still names is rebuilt from it; anything else
        // starts empty (S2).
        let tombstoned = e.out.is_some();
        let (out, tail) = match (rebuilt, e.out) {
            (Some((ring, tail, journal)), None) => (ring, Some((tail, journal))),
            (_, Some(end)) => (
                Ring::evicted(ctx.opts.job_output_retention, ctx.spool.new_job(), end),
                None,
            ),
            (None, None) => (
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
            argv: Argv {
                command: e.command.clone(),
                args: e.args.clone(),
            },
            out,
            spool,
            k: state,
            notify: None,
            notifier: Notifier::default(),
            stdin: None,
            streams: 0,
            idem: e.idem.clone(),
            tombstoned,
        }));
        {
            let mut map = locked(&ctx.jobs.map);
            if map.admit(&key) == Admit::Exists {
                continue;
            }
            map.insert(key.clone(), cell.clone());
            // K6: the job came back, so does its key's record.
            if let Some(i) = &e.idem
                && kernel::keeps_record(how)
            {
                map.record(Rec {
                    scope: key.0.clone(),
                    key: i.key.clone(),
                    id: key.1.clone(),
                    digest: i.digest.clone(),
                });
            }
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
    // A spool no job in the table owns (its job ended, or was removed, before the restart, or whose
    // output the retention dropped after the ledger named it and before it was unlinked).
    if ctx.jobs.ledger.enabled() {
        let keep: Vec<&Path> = adopted.iter().map(std::path::PathBuf::as_path).collect();
        Spool::retain(base, &keep);
    }
    ctx.jobs.bump_activity();
    ctx.jobs.save();
    // A recovered ended job is a candidate for the retention at once: its deadline may be long past.
    ctx.jobs.wake_retention();
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
struct Watching {
    ctx: Arc<Ctx>,
    cell: JobCell,
}

impl Watching {
    fn new(ctx: &Arc<Ctx>, cell: &JobCell) -> Self {
        let mut j = locked(cell);
        Self::open(ctx, cell, &mut j)
    }

    /// Register a stream on `cell`, whose lock the caller holds as `j`. The caller must hold it from
    /// the moment it *decides* to stream (`attach_plan`, the held `start`) to this call: the
    /// retention's attached-stream veto is only airtight if no sweep can see `streams == 0` between
    /// the decision to stream and the count. See `attach` in [`serve_op`].
    fn open(ctx: &Arc<Ctx>, cell: &JobCell, j: &mut Job) -> Self {
        j.streams += 1;
        ctx.jobs.bump_activity();
        Self {
            ctx: ctx.clone(),
            cell: cell.clone(),
        }
    }
}

impl Drop for Watching {
    fn drop(&mut self) {
        let mut j = locked(&self.cell);
        j.streams = j.streams.saturating_sub(1);
        // The last stream on an ended job the retention was holding back may have just closed:
        // look again. One permit, and nothing to look at for a job that is still running.
        let ended = j.k.ended();
        drop(j);
        self.ctx.jobs.bump_activity();
        if ended {
            self.ctx.jobs.wake_retention();
        }
    }
}

/// How a job stream opens: where it starts, whether it carries output, and the frame that says
/// which job it is (`started` on a held start, `attached` on an attach).
struct Opening {
    offset: u64,
    output: bool,
    first: Option<Value>,
}

/// Stream a job's output from `open.offset` to its end. `hold`: a held start, whose job dies with
/// the connection unless it was kept by then. How the stream ended.
async fn stream_job(
    ctx: Arc<Ctx>,
    key: Key,
    attached: JobCell,
    // The stream's registration, taken by the caller *before* anything could drop the job's output
    // (`Watching::open`): while this lives the job is attached (a host that lets idle machines sleep
    // asks), and the retention never touches it.
    watching: Watching,
    hold: bool,
    open: Opening,
    mut sink: Sink,
) -> StreamEnd {
    let _open = watching;
    // A job stream has no deadline of its own: it lives as long as the job or the connection.
    if let Ok(end) = follow(&ctx, &key, &attached, open, &mut sink).await {
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
    open: Opening,
    sink: &mut Sink,
) -> Result<StreamEnd, Cut> {
    if let Some(first) = &open.first {
        sink.put(frame::json(first)).await?;
    }
    let output = open.output;
    let mut changed = ctx.jobs.changed.subscribe();
    let mut sent = open.offset;
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
                // A status-only stream is never behind: it sends no bytes and no gap (M7).
                let (at, frame) =
                    kernel::stream_next(output, &j.k, sent, j.out.base, total, shell_told);
                sent = at;
                match frame {
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod retention_policy_tests {
    use super::*;

    fn job(phase: kernel::Phase) -> Job {
        let mut k = kernel::State::new(0, false, None);
        k.phase = phase;
        Job {
            mark: "m".into(),
            sp: 1,
            label: String::new(),
            argv: Argv::default(),
            // A ring that touches no file until something is pushed into it.
            out: Ring::new(1024, std::path::PathBuf::from("/nonexistent/job")),
            spool: None,
            k,
            notify: None,
            notifier: Notifier::default(),
            stdin: None,
            streams: 0,
            idem: None,
            tombstoned: false,
        }
    }

    /// An ended job that still holds output: a few bytes in the unfinished chunk, which touches no
    /// file (a chunk is written only at 64 KiB).
    fn holding(at: u64) -> Job {
        let mut j = job(ended(at));
        j.out.push(b"out", false);
        j
    }

    fn ended(at: u64) -> kernel::Phase {
        kernel::Phase::Ended {
            pgid: 7,
            code: 0,
            signal: None,
            at,
            shell_told: false,
        }
    }

    fn pressure() -> Pressure {
        Pressure {
            over_bytes: true,
            over_count: true,
        }
    }

    fn every_knob(due: bool) -> Retention {
        Retention {
            output_ms: if due { 1 } else { 0 },
            bytes: if due { 1 } else { 0 },
            max: if due { 1 } else { 0 },
            entry_ms: if due { 1 } else { 0 },
        }
    }

    /// **Only an ended job is a candidate.** The policy reads the kernel's phase, and nothing about
    /// the group: a `Running` job whose group was just observed empty (which is enough for
    /// `removable`'s second step, and not enough for this) is kept, and its bytes stay readable when
    /// its leftovers arrive. Red if the policy ages a job from `started`, takes a `Running` or
    /// `Exited` job, or falls back to "its group looks empty".
    #[test]
    fn only_an_ended_job_is_a_candidate() {
        let ret = every_knob(true);
        let now = 10_000;
        for phase in [
            kernel::Phase::Spawning,
            kernel::Phase::Running { pgid: 7 },
            kernel::Phase::Exited {
                pgid: 7,
                code: 0,
                signal: None,
            },
        ] {
            let j = job(phase.clone());
            assert_eq!(
                decide(&j, now, &ret, pressure()),
                Step::Keep(Why::Running),
                "{phase:?} is a candidate"
            );
        }
        // The same job, ended, is a candidate: the phase is the only difference (and the deadline
        // has passed: `ended` is at 0, `now` is 10,000).
        assert_eq!(
            decide(&job(ended(0)), now, &ret, pressure()),
            Step::Reap(Why::EntryTtl)
        );
    }

    /// **The vetoes, one each.** An open stream, an owed callback, a keyed job, and a job nothing is
    /// due for: every one of them is kept even with both ceilings over and every deadline past.
    #[test]
    fn each_veto_keeps_a_job_that_is_due() {
        let ret = every_knob(true);
        // An attached stream: the output is the only copy of the last bytes, and a take ends the
        // stream early (J9).
        let mut attached = job(ended(0));
        attached.streams = 1;
        assert_eq!(
            decide(&attached, 10_000, &ret, pressure()),
            Step::Keep(Why::Attached)
        );
        // An owed callback: a take abandons the count and the fleet never learns the end.
        let mut owing = job(ended(0));
        owing.notifier = Notifier::owing();
        assert_eq!(
            decide(&owing, 10_000, &ret, pressure()),
            Step::Keep(Why::OwedCallback)
        );
        // A keyed job: its key's record leaves with it (K8), so a retried start would run the
        // command a second time. It keeps its entry (`Why::Keyed`), and its output goes like anyone
        // else's.
        let mut keyed = job(ended(0));
        keyed.idem = Some(Idem {
            key: "k".into(),
            digest: "d".into(),
        });
        assert_eq!(
            decide(&keyed, 10_000, &ret, pressure()),
            Step::Keep(Why::Keyed)
        );
        // ...and while it still holds output, the output goes and the entry stays.
        keyed.out.push(b"out", false);
        assert_eq!(
            decide(&keyed, 10_000, &ret, pressure()),
            Step::Tombstone(Why::OutputTtl)
        );
        // Nothing due: no deadline has passed and neither ceiling is over.
        assert_eq!(
            decide(
                &job(ended(9_999)),
                10_000,
                &every_knob(false),
                Pressure::default()
            ),
            Step::Keep(Why::WithinDeadline)
        );
        // Every knob at its off value is exactly today: nothing is ever due, whatever the age.
        assert_eq!(
            decide(
                &holding(0),
                u64::MAX,
                &every_knob(false),
                Pressure::default()
            ),
            Step::Keep(Why::WithinDeadline)
        );
    }

    /// **The tiers.** Tier 1 drops the output and keeps the entry; tier 2 takes the entry; the byte
    /// budget only ever tombstones (a take needs a count ceiling or a clock, so byte pressure alone
    /// never answers a client `404`); and an entry deadline never lets output outlive it.
    #[test]
    fn the_tiers_are_output_first_and_entry_only_on_a_ceiling_or_a_clock() {
        let now = 10_000;
        // `Phase::Ended.at` is epoch seconds: 1,000 s here.
        let output_only = Retention {
            output_ms: 1_000_000,
            ..every_knob(false)
        };
        // Due by time: the output goes, the entry stays.
        assert_eq!(
            decide(&holding(1), now, &output_only, Pressure::default()),
            Step::Tombstone(Why::OutputTtl)
        );
        // Not yet due.
        assert_eq!(
            decide(&holding(9_500), now, &output_only, Pressure::default()),
            Step::Keep(Why::WithinDeadline)
        );
        // Over the byte budget, no clock: the output goes, oldest first (the caller's order), and
        // the entry is never taken for it.
        let bytes_only = Retention {
            bytes: 1,
            ..every_knob(false)
        };
        assert_eq!(
            decide(
                &holding(9_999),
                now,
                &bytes_only,
                Pressure {
                    over_bytes: true,
                    over_count: false,
                }
            ),
            Step::Tombstone(Why::Bytes)
        );
        // Over the count ceiling, no clock: the entry goes (the oldest is the caller's first).
        let max_only = Retention {
            max: 1,
            ..every_knob(false)
        };
        assert_eq!(
            decide(
                &job(ended(9_999)),
                now,
                &max_only,
                Pressure {
                    over_bytes: false,
                    over_count: true,
                }
            ),
            Step::Reap(Why::Count)
        );
        // An entry deadline reaps, and it also takes the output's deadline with it: whatever the
        // numbers say, output never outlives the entry.
        let entry_first = Retention {
            output_ms: 1_000_000,
            entry_ms: 1_000,
            ..every_knob(false)
        };
        assert_eq!(
            decide(&job(ended(1)), now, &entry_first, Pressure::default()),
            Step::Reap(Why::EntryTtl)
        );
    }
}

#[cfg(test)]
mod tail_step_tests {
    use super::{Drained, TailStep, step_after};

    /// A batch that filled its rounds with more to read is never the end, on either path: the bytes
    /// the spool holds past it are in no chunk and no journal, so stopping loses them until
    /// something else happens to wake the tail. Red when the `Drained::More` arms are folded into
    /// `TailStep::Stop(true)` (the dropped-remainder bug), which the integration test cannot see because
    /// punching the spool wakes the tail again.
    #[test]
    fn more_to_read_is_never_the_end() {
        assert_eq!(step_after(Drained::More, true), TailStep::Again);
        assert_eq!(step_after(Drained::More, false), TailStep::AgainYielding);
    }

    /// Only an idle tail (both files read to their ends) keeps the job and stops, and only a
    /// refused batch means the job was removed.
    #[test]
    fn only_idle_keeps_and_only_stopped_loses_the_job() {
        assert_eq!(step_after(Drained::Idle, true), TailStep::Stop(true));
        assert_eq!(step_after(Drained::Idle, false), TailStep::Stop(true));
        assert_eq!(step_after(Drained::Stopped, true), TailStep::Stop(false));
        assert_eq!(step_after(Drained::Stopped, false), TailStep::Stop(false));
    }
}
