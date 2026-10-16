//! `bash`'s job table: commands still running after their call returned, and commands that ended
//! with nobody waiting on them, until the model has been told.
//!
//! A job outlives its call for one of three reasons ([`JobKind`]): its call returned before it
//! finished — it went quiet, kept printing past the call's ceiling, or outlasted the model's
//! `timeout_ms` (**yielded**: unintended, so the oldest is evicted past [`MAX_YIELDED_JOBS`]); the
//! model asked for it with `run_in_background` (**background**); or its command exited but left
//! processes running (**adopted**, `cmd &`). Background and adopted jobs are intended: never evicted,
//! and past [`MAX_INTENDED_JOBS`] a new one is refused.
//!
//! The table is a [`Background`]: a job that ends while parked becomes a finish notice the loop
//! folds into the next turn ([`BashJobs::take_notices`]), a would-stop with jobs still running gets
//! one reminder ([`BashJobs::stop_reminder`]), and a compaction summary lists what is in flight
//! ([`BashJobs::carry`]). An ended job stays in the table, its output unread, until `bash_job` reads
//! it — at most [`MAX_ENDED_JOBS`] of them, the oldest dropped first.
//!
//! What ending a run does to the jobs is the host's ([`EndPolicy`]): `run` stops them when the
//! process exits; `serve` leaves them running and wakes the idle session when one ends
//! ([`BashJobs::subscribe`]).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Instant;

use agent_core::Background;
use agent_core::background::{NOTICE_PREFIX, REMINDER_PREFIX};
use tokio::sync::watch;

use super::exec::{ExecResult, Kill};
use super::output::{OutputAccumulator, OutputSnapshot};

/// Most yielded (unintended) jobs kept running; parking another kills the oldest of them.
pub const MAX_YIELDED_JOBS: usize = 4;
/// Most intended jobs (started in the background, or adopted) running at once. One past it is
/// refused, never made room for: the model meant every one of them.
pub const MAX_INTENDED_JOBS: usize = 8;
/// Most ended jobs whose output is kept for `bash_job` after their notice; the oldest goes first.
pub const MAX_ENDED_JOBS: usize = 8;
/// Lines of new output a finish notice quotes.
pub const NOTICE_TAIL_LINES: usize = 20;
/// Characters of a command quoted in notices, reminders and the carry.
pub const COMMAND_CHARS: usize = 80;
/// How every stop reminder ends: the one reason not to end the turn yet.
pub(crate) const REMINDER_ASK: &str = "If you still need a job's result, wait for it with bash_job \
                                       {\"job\": N}; otherwise just end your turn.";

/// Why a job outlived its call — which also decides how long it may live: a yielded job never
/// outlives the turn ([`BashJobs::end_turn`]); an intended one lives until it exits, is killed, or
/// its owner ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// Its call returned before it finished: it went quiet, kept printing past the call's ceiling,
    /// or outlasted the model's `timeout_ms`. The model did not ask for a job, so the cap may evict
    /// it, and it never outlives the turn.
    Yielded,
    /// Started with `run_in_background`. The model asked for this; never evicted.
    Background,
    /// Its command exited but left processes running (`cmd &`, a daemon). The model asked for this;
    /// never evicted.
    Adopted,
}

impl JobKind {
    /// Whether the model meant this job to run past its call.
    pub fn intended(self) -> bool {
        !matches!(self, JobKind::Yielded)
    }
}

/// What ending a run does to the jobs still running — said in the stop reminder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EndPolicy {
    /// `run` (and a subagent): ending the run stops them.
    #[default]
    StopAtEnd,
    /// `serve`: they keep running, and the session wakes when one ends.
    KeepRunning,
}

/// One command on the detached path: attached (owned by the call waiting on it) or parked in
/// [`BashJobs`]. Dropping it kills every process the command started: a call whose future is
/// dropped, or a table that is dropped, never leaves one behind.
pub(crate) struct Job {
    /// Assigned the first time the job is parked.
    pub id: Option<u64>,
    pub kind: JobKind,
    pub shared: Arc<JobShared>,
    pub handle: tokio::task::JoinHandle<()>,
    pub started: Instant,
    /// A stop reminder already listed it. On the job, not its table entry, so a `bash_job` wait
    /// that takes it out and parks it again does not make it news to the next reminder.
    pub reminded: bool,
    /// The model's command, truncated for notices.
    pub command: String,
}

impl Drop for Job {
    fn drop(&mut self) {
        self.handle.abort();
        // `abort` only drops the runner's future when the runtime next polls the task, which a
        // process on its way out may never do — kill directly too.
        #[cfg(unix)]
        if let Some(group) = self.shared.untrack() {
            super::exec::spawn_tracked_cleanup(move || {
                group.kill();
            });
        }
    }
}

/// How a job's command ended, as the runner task recorded it.
pub(crate) struct JobEnd {
    pub result: Result<ExecResult, String>,
    pub at: Instant,
}

/// What the runner's task and whoever holds the [`Job`] share.
pub(crate) struct JobShared {
    out: Mutex<JobOutput>,
    /// Woken on every chunk, when the shell leaves processes running, and when the job ends — the
    /// attached call re-checks all three.
    pub notify: tokio::sync::Notify,
    /// How to stop the command while it runs — every process it started on this host, or its
    /// remote stream: what a kill, [`Job`]'s `Drop` and [`kill_live_jobs`] use.
    kill: Mutex<Option<Kill>>,
    /// This job's key in [`LIVE_JOBS`].
    key: u64,
    /// The runner reported a [`Kill`], which promises the output is live — see
    /// `Bash::idle_window`.
    pub live: AtomicBool,
    /// Where the command runs (no spill file when remote).
    pub remote: bool,
    state: Mutex<JobState>,
    /// The table to tell when the job ends or is orphaned, so a waiting host can wake.
    table: Weak<BashJobs>,
}

#[derive(Default)]
struct JobState {
    /// The shell exited leaving processes running: its own result, and what is left.
    orphans: Option<(ExecResult, String)>,
    end: Option<JobEnd>,
}

pub(crate) struct JobOutput {
    /// Output since the last report to the model.
    pub acc: OutputAccumulator,
    pub last_output: Instant,
    /// Whether any chunk came through the sink at all — see `Bash::finish_job`'s fallback.
    pub streamed: bool,
}

impl JobShared {
    pub fn new(remote: bool, table: &Arc<BashJobs>) -> Self {
        Self {
            out: Mutex::new(JobOutput {
                acc: accumulator(!remote),
                last_output: Instant::now(),
                streamed: false,
            }),
            notify: tokio::sync::Notify::new(),
            kill: Mutex::new(None),
            key: NEXT_JOB_KEY.fetch_add(1, Ordering::Relaxed),
            live: AtomicBool::new(false),
            remote,
            state: Mutex::new(JobState::default()),
            table: Arc::downgrade(table),
        }
    }

    pub fn out(&self) -> MutexGuard<'_, JobOutput> {
        lock(&self.out)
    }

    pub fn append(&self, bytes: &[u8]) {
        {
            let mut out = lock(&self.out);
            out.acc.append(bytes);
            out.last_output = Instant::now();
            out.streamed = true;
        }
        self.notify.notify_one();
    }

    pub fn track(&self, kill: Kill) {
        *lock(&self.kill) = Some(kill.clone());
        lock(&LIVE_JOBS).push((self.key, kill));
        self.live.store(true, Ordering::Relaxed);
        // The waiting call computed its idle deadline before the output was known to be live; wake
        // it to recompute.
        self.notify.notify_one();
    }

    /// Forget how to kill the command (it exited, or the caller is about to kill it), returning the
    /// kill if it was still recorded.
    pub fn untrack(&self) -> Option<Kill> {
        let kill = lock(&self.kill).take()?;
        lock(&LIVE_JOBS).retain(|(k, _)| *k != self.key);
        Some(kill)
    }

    /// The runner reported that the shell exited leaving processes running.
    pub fn orphaned(&self, shell: &ExecResult, left: String) {
        lock(&self.state).orphans = Some((shell.clone(), left));
        self.notify.notify_one();
        if let Some(t) = self.table.upgrade() {
            t.changed();
        }
    }

    /// The runner's task is done: record how, and wake whoever is waiting.
    pub fn finish(&self, result: Result<ExecResult, String>) {
        lock(&self.state).end = Some(JobEnd {
            result,
            at: Instant::now(),
        });
        self.notify.notify_one();
        if let Some(t) = self.table.upgrade() {
            t.changed();
        }
    }

    pub fn ended(&self) -> bool {
        lock(&self.state).end.is_some()
    }

    pub fn take_end(&self) -> Option<JobEnd> {
        lock(&self.state).end.take()
    }

    /// The shell's result and what it left running, if it exited leaving processes.
    pub fn orphans(&self) -> Option<(ExecResult, String)> {
        lock(&self.state).orphans.clone()
    }

    /// The output since the last report, finished and snapshotted; the next report starts empty.
    pub fn take_segment(&self) -> OutputSnapshot {
        let mut out = lock(&self.out);
        let fresh = accumulator(!self.remote);
        let mut segment = std::mem::replace(&mut out.acc, fresh);
        segment.finish();
        segment.snapshot(true)
    }

    /// The last [`NOTICE_TAIL_LINES`] lines of the output since the last report, without consuming
    /// it (a later `bash_job` returns all of it), and whether that is all of it.
    fn tail(&self) -> (String, bool) {
        let content = lock(&self.out).acc.snapshot(false).content;
        let text = super::bash::clean_output(&content);
        let lines: Vec<&str> = text.trim_end().lines().collect();
        let start = lines.len().saturating_sub(NOTICE_TAIL_LINES);
        (lines[start..].join("\n"), start == 0)
    }
}

/// A fresh output accumulator; `spill: false` when the command runs somewhere this process cannot
/// write a file the model could read.
pub(crate) fn accumulator(spill: bool) -> OutputAccumulator {
    if spill {
        OutputAccumulator::new()
    } else {
        OutputAccumulator::new().without_spill()
    }
}

/// A parked job and what the model has been told about it.
struct Parked {
    job: Job,
    /// Its finish notice has been delivered (or it was reported by an attached wait).
    notified: bool,
    /// A notice already said its shell exited leaving processes running.
    orphan_notified: bool,
}

/// A running job stopped at the end of a turn, a run or a session — see [`BashJobs::end_turn`] and
/// [`BashJobs::stop_all`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StoppedJob {
    pub job: u64,
    pub command: String,
    pub kind: JobKind,
    pub running_secs: u64,
    /// Processes of the job still running after the kill — ones we may not signal (another uid,
    /// e.g. the root child of `sudo`). Absent when everything stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub survived: Option<String>,
}

/// One session's jobs. Shared by every `bash`/`bash_job` built over it (a registry rebuild keeps
/// them), and dropped — killing every job — with the last of them.
pub struct BashJobs {
    next: AtomicU64,
    table: Mutex<BTreeMap<u64, Parked>>,
    /// Bumped whenever a parked job ends or is orphaned: what [`Self::subscribe`] hands out.
    wake: watch::Sender<u64>,
    policy: Mutex<EndPolicy>,
}

impl Default for BashJobs {
    fn default() -> Self {
        Self::new(EndPolicy::default())
    }
}

impl BashJobs {
    pub fn new(policy: EndPolicy) -> Self {
        Self {
            next: AtomicU64::new(0),
            table: Mutex::new(BTreeMap::new()),
            wake: watch::Sender::new(0),
            policy: Mutex::new(policy),
        }
    }

    /// What ending a run does to the jobs, as the stop reminder will say. Set once by the host that
    /// owns the table (`serve`: [`EndPolicy::KeepRunning`]).
    pub fn set_policy(&self, policy: EndPolicy) {
        *lock(&self.policy) = policy;
    }

    fn changed(&self) {
        self.wake.send_modify(|v| *v = v.wrapping_add(1));
    }

    /// A receiver that changes whenever a parked job ends or is orphaned. A host with no run in
    /// flight (`serve`, idle) marks it seen, checks [`Self::has_notice`], then awaits a change.
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.wake.subscribe()
    }

    /// Whether a finish notice is waiting to be delivered.
    pub fn has_notice(&self) -> bool {
        lock(&self.table).values().any(Parked::notice_due)
    }

    pub(crate) fn assign_id(&self, job: &mut Job) -> u64 {
        *job.id
            .get_or_insert_with(|| self.next.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// What a job of `kind` lives until, as the model is told when its call returns.
    pub(crate) fn lifetime(&self, kind: JobKind) -> &'static str {
        match (kind, *lock(&self.policy)) {
            (JobKind::Yielded, _) => "it is stopped when you end your turn",
            (_, EndPolicy::StopAtEnd) => "it is stopped for you when the run ends",
            (_, EndPolicy::KeepRunning) => "it keeps running after your turn ends",
        }
    }

    /// The running intended jobs, listed for the model, when there are already
    /// [`MAX_INTENDED_JOBS`] of them: a new one would be refused.
    pub(crate) fn intended_full(&self) -> Option<String> {
        let running = running_intended(&lock(&self.table));
        (running.len() >= MAX_INTENDED_JOBS).then(|| running.join(", "))
    }

    /// Park a job whose call returned while it runs (any kind: an intended job a `bash_job` wait
    /// returned from is parked again here). Returns the yielded job evicted to stay within
    /// [`MAX_YIELDED_JOBS`], if any (the caller drops it, killing it, outside the lock).
    pub(crate) fn park(&self, job: Job) -> Option<(u64, Job)> {
        let id = job.id?;
        let mut parked = Parked::new(job);
        // Processes its shell left running were already reported — by the adoption, a notice, or
        // the background start — so a re-park must not announce them again.
        parked.orphan_notified = parked.job.shared.orphans().is_some();
        let mut table = lock(&self.table);
        table.insert(id, parked);
        let running_yielded: Vec<u64> = table
            .iter()
            .filter(|(_, p)| p.job.kind == JobKind::Yielded && !p.job.shared.ended())
            .map(|(id, _)| *id)
            .collect();
        if running_yielded.len() > MAX_YIELDED_JOBS {
            let oldest = running_yielded[0];
            return table.remove(&oldest).map(|p| (oldest, p.job));
        }
        None
    }

    /// Park an intended job under a fresh id (or the one it already has), unless
    /// [`MAX_INTENDED_JOBS`] are already running: then the job comes back with the list of running
    /// jobs to report.
    pub(crate) fn park_intended(&self, mut job: Job) -> Result<u64, (Job, String)> {
        let mut table = lock(&self.table);
        let running = running_intended(&table);
        if running.len() >= MAX_INTENDED_JOBS {
            return Err((job, running.join(", ")));
        }
        let id = self.assign_id(&mut job);
        let mut parked = Parked::new(job);
        // Its adoption or start was this call's own result (a background command's shell may also
        // have exited leaving processes, before it settled); the next thing to tell the model is
        // its end.
        parked.orphan_notified = true;
        table.insert(id, parked);
        Ok(id)
    }

    /// Take job `id` out of the table for `bash_job`: it is the caller's now (to wait on, kill, or
    /// report as ended).
    pub(crate) fn take(&self, id: u64) -> Option<Job> {
        lock(&self.table).remove(&id).map(|p| p.job)
    }

    /// Drop every ended job. For a table whose notices nobody takes (`serve`'s host `bash`): this
    /// is the only retention it gets, so ended jobs do not pile up for the life of the session.
    pub fn forget_ended(&self) {
        let ended: Vec<Parked> = {
            let mut table = lock(&self.table);
            let ids: Vec<u64> = table
                .iter()
                .filter(|(_, p)| p.job.shared.ended())
                .map(|(id, _)| *id)
                .collect();
            ids.iter().filter_map(|id| table.remove(id)).collect()
        };
        drop(ended);
    }

    /// Stop every running job, oldest first, and say which. Ended jobs are dropped too: nobody will
    /// read them now. For the end of a `run` and a transcript switch in `serve`.
    pub fn stop_all(&self) -> Vec<StoppedJob> {
        self.stop(|_| true)
    }

    /// The turn (a run) is over: stop the jobs that must not outlive it — the **yielded** ones, a
    /// foreground command the model expected to finish, which in an idle `serve` session would
    /// otherwise be a hung loop spinning forever. Intended jobs live on (until the end of a `run`,
    /// or for as long as a `serve` session). The model was told so by the stop reminder. Ended jobs
    /// stay, their output unread.
    pub fn end_turn(&self) -> Vec<StoppedJob> {
        // A yielded job whose shell has exited leaving processes running is adopted, notice
        // delivered or not.
        self.stop(|p| {
            p.job.kind == JobKind::Yielded
                && !p.job.shared.ended()
                && p.job.shared.orphans().is_none()
        })
    }

    /// Take the jobs `which` selects out of the table, kill those still running (each blocking
    /// until its processes are gone), and say which those were.
    fn stop(&self, which: impl Fn(&Parked) -> bool) -> Vec<StoppedJob> {
        let taken: Vec<(u64, Parked)> = {
            let mut table = lock(&self.table);
            let ids: Vec<u64> = table
                .iter()
                .filter(|(_, p)| which(p))
                .map(|(id, _)| *id)
                .collect();
            ids.into_iter()
                .filter_map(|id| table.remove(&id).map(|p| (id, p)))
                .collect()
        };
        let mut stopped = Vec::new();
        let mut kills = Vec::new();
        for (id, p) in &taken {
            if !p.job.shared.ended() {
                stopped.push(StoppedJob {
                    job: *id,
                    command: p.job.command.clone(),
                    kind: p.job.kind,
                    running_secs: p.job.started.elapsed().as_secs(),
                    survived: None,
                });
            }
            if let Some(kill) = p.job.shared.untrack() {
                kills.push((*id, kill));
            }
        }
        // All at once, so stopping N jobs spends one SIGTERM grace, not N.
        let all: Vec<Kill> = kills.iter().map(|(_, k)| k.clone()).collect();
        for ((id, _), left) in kills.iter().zip(super::exec::kill_all(&all)) {
            if let Some(s) = stopped.iter_mut().find(|s| s.job == *id) {
                s.survived = left;
            }
        }
        drop(taken);
        stopped
    }

    fn notices(&self) -> Option<String> {
        let mut table = lock(&self.table);
        let mut parts = Vec::new();
        for (&id, p) in table.iter_mut() {
            if !p.notice_due() {
                continue;
            }
            let job = &mut p.job;
            let state = lock(&job.shared.state);
            let header = match &state.end {
                Some(end) => {
                    p.notified = true;
                    ended_header(id, job, end, state.orphans.is_some())
                }
                None => {
                    // The shell exited while parked, leaving processes running: the job is the
                    // processes now, which the command meant to leave (`cmd &`). A background job
                    // was intended already, and stays one.
                    p.orphan_notified = true;
                    let (shell, left) = state.orphans.clone().unwrap_or_default();
                    drop(state);
                    if job.kind == JobKind::Yielded {
                        job.kind = JobKind::Adopted;
                    }
                    parts.push(orphan_notice(id, job, &shell, &left));
                    continue;
                }
            };
            drop(state);
            let (tail, whole) = job.shared.tail();
            let body = if tail.is_empty() {
                "(no new output)".to_string()
            } else if whole {
                tail
            } else {
                format!("… (last {NOTICE_TAIL_LINES} lines)\n{tail}")
            };
            parts.push(format!(
                "{header}\n{body}\nbash_job {{\"job\": {id}}} returns the full output."
            ));
        }
        // Retention: ended jobs whose notice is out wait for `bash_job`, at most `MAX_ENDED_JOBS`.
        let ended: Vec<u64> = table
            .iter()
            .filter(|(_, p)| p.notified && p.job.shared.ended())
            .map(|(id, _)| *id)
            .collect();
        if ended.len() > MAX_ENDED_JOBS {
            let drop_ids = &ended[..ended.len() - MAX_ENDED_JOBS];
            for id in drop_ids {
                table.remove(id);
            }
            let ids: Vec<String> = drop_ids.iter().map(|id| format!("job {id}")).collect();
            parts.push(format!(
                "{NOTICE_PREFIX}output discarded: {}] At most {MAX_ENDED_JOBS} ended jobs keep their \
                 output for bash_job; the oldest went first.",
                ids.join(", ")
            ));
        }
        (!parts.is_empty()).then(|| parts.join("\n\n"))
    }

    fn reminder(&self) -> Option<String> {
        let mut table = lock(&self.table);
        let running: Vec<(&u64, &mut Parked)> = table
            .iter_mut()
            .filter(|(_, p)| !p.job.shared.ended())
            .collect();
        if running.iter().all(|(_, p)| p.job.reminded) {
            return None;
        }
        let keep = *lock(&self.policy) == EndPolicy::KeepRunning;
        let mut lines = Vec::new();
        let (mut stopped, mut kept) = (Vec::new(), Vec::new());
        for (id, p) in running {
            p.job.reminded = true;
            lines.push(format!(
                "- job {id}: `{}` ({})",
                p.job.command,
                describe_running(&p.job)
            ));
            // A yielded job never outlives the turn; an intended one follows the host's policy.
            if keep && p.job.kind.intended() {
                kept.push(id.to_string());
            } else {
                stopped.push(id.to_string());
            }
        }
        // What ending does to each, then the one reason not to end yet: a result still needed. No
        // "wait for one" offered unconditionally — read as an instruction, it sent models to wait
        // out a server's silence — and no kill for a job the end stops anyway.
        let mut fates = Vec::new();
        if !stopped.is_empty() {
            fates.push(format!(
                "{} {} {} stopped for you (no need to kill {})",
                plural(stopped.len(), "job", "jobs"),
                stopped.join(", "),
                plural(stopped.len(), "is", "are"),
                plural(stopped.len(), "it", "them"),
            ));
        }
        if !kept.is_empty() {
            fates.push(format!(
                "{} {} {} running and you will be told when {} exits (bash_job {{\"job\": N, \
                 \"kill\": true}} stops one you no longer need)",
                plural(kept.len(), "job", "jobs"),
                kept.join(", "),
                plural(kept.len(), "keeps", "keep"),
                plural(kept.len(), "it", "each"),
            ));
        }
        Some(format!(
            "{REMINDER_PREFIX} You are ending your turn with jobs still running:\n{}\nIf you end \
             your turn now, {}. {REMINDER_ASK}",
            lines.join("\n"),
            fates.join("; ")
        ))
    }

    fn table_text(&self) -> Option<String> {
        let table = lock(&self.table);
        let lines: Vec<String> = table
            .iter()
            .map(|(id, p)| {
                let state = if p.job.shared.ended() {
                    format!("ended, output not yet read: bash_job {{\"job\": {id}}}")
                } else {
                    describe_running(&p.job)
                };
                format!("- job {id}: `{}` ({state})", p.job.command)
            })
            .collect();
        (!lines.is_empty()).then(|| {
            format!(
                "Jobs `bash` is tracking (do not start these again; bash_job waits on or stops \
                 them):\n{}",
                lines.join("\n")
            )
        })
    }
}

impl Parked {
    fn new(job: Job) -> Self {
        Self {
            job,
            notified: false,
            orphan_notified: false,
        }
    }

    fn notice_due(&self) -> bool {
        let state = lock(&self.job.shared.state);
        if state.end.is_some() {
            !self.notified
        } else {
            state.orphans.is_some() && !self.orphan_notified
        }
    }
}

impl Background for BashJobs {
    fn take_notices(&self) -> Option<String> {
        self.notices()
    }

    fn stop_reminder(&self) -> Option<String> {
        self.reminder()
    }

    fn carry(&self) -> Option<String> {
        self.table_text()
    }
}

/// "running 45 s" plus why it is a job.
fn describe_running(job: &Job) -> String {
    let secs = job.started.elapsed().as_secs();
    match job.kind {
        JobKind::Yielded => format!("running {secs} s; its call returned early"),
        JobKind::Background => format!("running {secs} s; started in the background"),
        JobKind::Adopted => format!("running {secs} s; left running by its command"),
    }
}

/// The running intended jobs, as "job N `cmd`".
fn running_intended(table: &BTreeMap<u64, Parked>) -> Vec<String> {
    table
        .iter()
        .filter(|(_, p)| p.job.kind.intended() && !p.job.shared.ended())
        .map(|(id, p)| format!("job {id} `{}`", p.job.command))
        .collect()
}

/// The first line of an ended job's notice.
fn ended_header(id: u64, job: &Job, end: &JobEnd, orphaned: bool) -> String {
    let secs = end.at.duration_since(job.started).as_secs();
    let cmd = &job.command;
    match &end.result {
        Err(e) => format!("{NOTICE_PREFIX}{id} failed after {secs} s: `{cmd}`: {e}]"),
        // `bash` sets no deadline; this is an exec endpoint enforcing the one on the wire.
        Ok(r) if r.timed_out => format!(
            "{NOTICE_PREFIX}{id} timed out after {secs} s (the sandbox's exec endpoint stopped it): \
             `{cmd}`]"
        ),
        Ok(_) if orphaned => format!(
            "{NOTICE_PREFIX}{id} ended after {secs} s: `{cmd}` (the processes its command left \
             running have exited)]"
        ),
        Ok(r) => match (r.code, r.signal) {
            (Some(code), _) => {
                format!("{NOTICE_PREFIX}{id} exited with code {code} after {secs} s: `{cmd}`]")
            }
            (None, Some(signal)) => format!(
                "{NOTICE_PREFIX}{id} was killed by {} after {secs} s: `{cmd}`]",
                signal_text(signal)
            ),
            (None, None) => format!("{NOTICE_PREFIX}{id} exited after {secs} s: `{cmd}`]"),
        },
    }
}

/// A parked job's shell exited, leaving processes running.
fn orphan_notice(id: u64, job: &Job, shell: &ExecResult, left: &str) -> String {
    let secs = job.started.elapsed().as_secs();
    let how = match (shell.code, shell.signal) {
        (Some(code), _) => format!("exited with code {code}"),
        (None, Some(signal)) => format!("was killed by {}", signal_text(signal)),
        (None, None) => "exited".to_string(),
    };
    format!(
        "{NOTICE_PREFIX}{id}'s command {how} after {secs} s but left {left} running: `{}`] That is \
         still job {id}: you will be told when it exits. bash_job {{\"job\": {id}, \"kill\": true}} \
         stops it.",
        job.command
    )
}

fn plural<'a>(n: usize, one: &'a str, many: &'a str) -> &'a str {
    if n == 1 { one } else { many }
}

pub(crate) fn signal_text(signal: i32) -> String {
    match super::exec::signal_name(signal) {
        Some(name) => format!("{name} ({signal})"),
        None => format!("signal {signal}"),
    }
}

/// How to stop every job still running, by job key, for [`kill_live_jobs`].
static LIVE_JOBS: Mutex<Vec<(u64, Kill)>> = Mutex::new(Vec::new());
static NEXT_JOB_KEY: AtomicU64 = AtomicU64::new(0);

/// Kill every job still running. The exit path calls this before `std::process::exit`, which runs no
/// destructors: a parked job would otherwise outlive the agent. The kills are tracked, so the
/// `wait_for_pending_group_kills` that follows waits for them.
pub fn kill_live_jobs() {
    #[cfg(unix)]
    for (_, kill) in std::mem::take(&mut *lock(&LIVE_JOBS)) {
        super::exec::spawn_tracked_cleanup(move || {
            kill.kill();
        });
    }
}

/// Lock a mutex, recovering the data on poison rather than panicking (the crate forbids `unwrap`).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
