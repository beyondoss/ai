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
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;

use agent_core::Background;
use agent_core::background::{NOTICE_PREFIX, REMINDER_PREFIX};
use tokio::sync::watch;

use super::bash_sandbox::SandboxLink;
use super::exec::{ExecResult, Kill};
use super::output::{OutputAccumulator, OutputSnapshot};
use crate::exec_endpoint::jobs::JobScope;

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

/// The session custom-entry kind each local job's processes are journaled under — its process group,
/// the group leader's start time and its environment tag ([`super::bash_proc::TAG_ENV`]) — so a
/// process reopening the session after this one was killed outright (SIGKILL, OOM: no exit path ran
/// to stop them) can find what survived and stop it ([`BashJobs::stop_survivors`]).
pub const PROCS_ENTRY_KIND: &str = "bash_job_procs";

/// Where a [`BashJobs`] journals a job's processes as it spawns them: the host's session store.
pub type ProcRecorder = Arc<dyn Fn(serde_json::Value) + Send + Sync>;

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
        // A job the sandbox owns outlives this process's interest in it: dropping it lets go of
        // its stream (the abort above), and nothing more.
        if self.shared.released() {
            return;
        }
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
    /// Set when the sandbox runs it ([`crate::exec_endpoint::jobs`]): where to reach it.
    sandbox: OnceLock<SandboxLink>,
    /// The sandbox owns it now (it was kept), and this host's policy lets it outlive the session:
    /// dropping the job, or this process exiting, must not kill it.
    released: AtomicBool,
    state: Mutex<JobState>,
    /// The table to tell when the job ends or is orphaned, so a waiting host can wake.
    table: Weak<BashJobs>,
    /// The model's command, truncated — what a journaled record names it by.
    command: OnceLock<String>,
    /// exec v1.4: how the session's notify driver steers the task that holds this job's stream
    /// ([`super::bash_sandbox::Ctl`]). Set by that task while it can be steered.
    ctl: Mutex<Option<tokio::sync::mpsc::UnboundedSender<super::bash_sandbox::Ctl>>>,
    /// When the job ended, if a listing said so before its end was read ([`Self::finish`]).
    end_at: Mutex<Option<Instant>>,
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
    /// How many bytes of output the command has produced in all (from where this job started
    /// reading it): the sandbox's offset of the next byte, for a sandbox job.
    pub offset: u64,
    pub last_output: Instant,
    /// Whether any chunk came through the sink at all — see `Bash::finish_job`'s fallback.
    pub streamed: bool,
}

impl JobShared {
    pub fn new(remote: bool, table: &Arc<BashJobs>) -> Self {
        Self {
            out: Mutex::new(JobOutput {
                acc: accumulator(!remote),
                offset: 0,
                last_output: Instant::now(),
                streamed: false,
            }),
            notify: tokio::sync::Notify::new(),
            kill: Mutex::new(None),
            key: NEXT_JOB_KEY.fetch_add(1, Ordering::Relaxed),
            live: AtomicBool::new(false),
            remote,
            sandbox: OnceLock::new(),
            released: AtomicBool::new(false),
            state: Mutex::new(JobState::default()),
            table: Arc::downgrade(table),
            command: OnceLock::new(),
            ctl: Mutex::new(None),
            end_at: Mutex::new(None),
        }
    }

    /// The model's command, truncated, for the journaled record of its processes.
    pub fn set_command(&self, command: String) {
        let _ = self.command.set(command);
    }

    /// A job the sandbox runs, read from `offset` of its output.
    pub fn in_sandbox(link: SandboxLink, offset: u64, table: &Arc<BashJobs>) -> Self {
        let shared = Self::new(true, table);
        let _ = shared.sandbox.set(link);
        lock(&shared.out).offset = offset;
        shared
    }

    /// The job table it belongs to, while there is one.
    pub(crate) fn table(&self) -> Option<Arc<BashJobs>> {
        self.table.upgrade()
    }

    /// Where the sandbox runs it, if it does.
    pub fn sandbox(&self) -> Option<&SandboxLink> {
        self.sandbox.get()
    }

    /// Run by the sandbox from now on — set once, before the job runs.
    pub fn set_sandbox(&self, link: SandboxLink) {
        let _ = self.sandbox.set(link);
    }

    /// The offset of the next byte of output.
    pub fn offset(&self) -> u64 {
        lock(&self.out).offset
    }

    /// `n` bytes of output the sandbox no longer had when they were read: counted, and marked.
    pub fn append_gap(&self, n: u64) {
        {
            let mut out = lock(&self.out);
            out.offset += n;
            out.acc.append(
                format!(
                    "\n[{} of output not kept by the sandbox]\n",
                    super::output::format_size(n)
                )
                .as_bytes(),
            );
            out.last_output = Instant::now();
            out.streamed = true;
        }
        self.notify.notify_one();
    }

    /// Whether dropping the job leaves it running ([`Self::release`]).
    pub fn released(&self) -> bool {
        self.released.load(Ordering::Relaxed)
    }

    /// The sandbox owns it and this host lets it outlive the session: neither dropping the job nor
    /// this process's exit kills it. Only an explicit kill does, which is why the kill stays
    /// recorded.
    pub fn release(&self) {
        self.released.store(true, Ordering::Relaxed);
        lock(&LIVE_JOBS).retain(|(k, _)| *k != self.key);
    }
    pub fn out(&self) -> MutexGuard<'_, JobOutput> {
        lock(&self.out)
    }

    pub fn append(&self, bytes: &[u8]) {
        {
            let mut out = lock(&self.out);
            out.offset += bytes.len() as u64;
            out.acc.append(bytes);
            out.last_output = Instant::now();
            out.streamed = true;
        }
        self.notify.notify_one();
    }

    pub fn track(&self, kill: Kill) {
        if let (Kill::Procs(group), Some(table)) = (&kill, self.table.upgrade()) {
            table.record_procs(group, None, self.command.get().map(String::as_str));
        }
        *lock(&self.kill) = Some(kill.clone());
        if !self.released() {
            lock(&LIVE_JOBS).push((self.key, kill));
        }
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

    /// The runner's task is done: record how, and wake whoever is waiting. At the time a listing
    /// gave for its end ([`Self::set_end_at`]), if one did; otherwise now.
    pub fn finish(&self, result: Result<ExecResult, String>) {
        let at = lock(&self.end_at).take().unwrap_or_else(Instant::now);
        self.finish_at(result, at);
    }

    /// A listing found the job ended at `at`: its end is recorded then, as a reattach records it.
    pub fn set_end_at(&self, at: Instant) {
        *lock(&self.end_at) = Some(at);
    }

    /// Install the channel the notify driver steers this job's stream-holding task through.
    pub(crate) fn set_ctl(
        &self,
        tx: Option<tokio::sync::mpsc::UnboundedSender<super::bash_sandbox::Ctl>>,
    ) {
        *lock(&self.ctl) = tx;
    }

    /// Steer the job's stream-holding task; `false` when nothing is listening.
    pub(crate) fn ctl(&self, c: super::bash_sandbox::Ctl) -> bool {
        lock(&self.ctl)
            .as_ref()
            .is_some_and(|tx| tx.send(c).is_ok())
    }

    /// As [`Self::finish`], for a job that ended at `at` — earlier, while nobody was following it.
    pub fn finish_at(&self, result: Result<ExecResult, String>, at: Instant) {
        lock(&self.state).end = Some(JobEnd { result, at });
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

/// A job a killed process left running, as [`BashJobs::stop_survivors`] found it on reopen.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SurvivorFate {
    /// Its job number, if it had one (a foreground command still attached when the process died
    /// had none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job: Option<u64>,
    pub command: String,
    /// What is still running after the stop — processes we may not signal. `None`: all stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub survived: Option<String>,
}

/// What [`BashJobs::stop_survivors`] found, until the reopen note reports it.
#[derive(Default)]
struct Survivors {
    found: Vec<SurvivorFate>,
}

/// One session's jobs. Shared by every `bash`/`bash_job` built over it (a registry rebuild keeps
/// them), and dropped — killing every job — with the last of them.
pub struct BashJobs {
    next: AtomicU64,
    /// The transcript the next run is on has been read ([`Background::resume`]): its ids are
    /// counted and what it left running was reported. Cleared by [`Self::stop_all`], which ends a
    /// transcript's jobs, so the next run reads the one it is on.
    resumed: AtomicBool,
    table: Mutex<BTreeMap<u64, Parked>>,
    /// Bumped whenever a parked job ends or is orphaned: what [`Self::subscribe`] hands out.
    wake: watch::Sender<u64>,
    policy: Mutex<EndPolicy>,
    /// Where this session's jobs live in a sandbox. `None`: commands that run remotely keep the
    /// stream-owned lifetime (a table with no session behind it — a subagent's, `serve`'s host
    /// `bash`).
    scope: Mutex<Option<JobScope>>,
    /// Sandbox jobs whose end the model has been told: their directories go once that report is
    /// persisted ([`Self::collect_garbage`]).
    garbage: Mutex<Vec<SandboxLink>>,
    /// Where each local job's processes are journaled ([`PROCS_ENTRY_KIND`]).
    recorder: Mutex<Option<ProcRecorder>>,
    /// What [`Self::stop_survivors`] found of the jobs a killed process left, for the reopen note.
    survivors: Mutex<Survivors>,
    /// A reattach not done yet ([`super::bash_sandbox::reattach_later`]): the sandbox's jobs are
    /// read back at the session's first use of the sandbox, not when it opens.
    reattach: Mutex<Option<Arc<super::bash_sandbox::PendingReattach>>>,
    /// The jobs the reopen note called "not yet checked", reported once the reattach has run.
    unchecked: Mutex<Vec<(u64, String)>>,
    /// Unchecked jobs the sandbox no longer had when it was checked: reported by the next notices.
    checked_gone: Mutex<Vec<(u64, String)>>,
    /// Sandbox jobs a tool call has taken out of the table ([`Self::take`]), while it has them.
    in_call: Mutex<Vec<Weak<JobShared>>>,
    /// exec v1.4: the session's notify driver, where the host registered one (`serve --service`).
    notify: Mutex<Option<Weak<super::job_notify::SessionNotify>>>,
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
            resumed: AtomicBool::new(false),
            table: Mutex::new(BTreeMap::new()),
            wake: watch::Sender::new(0),
            policy: Mutex::new(policy),
            scope: Mutex::new(None),
            garbage: Mutex::new(Vec::new()),
            recorder: Mutex::new(None),
            survivors: Mutex::new(Survivors::default()),
            reattach: Mutex::new(None),
            unchecked: Mutex::new(Vec::new()),
            checked_gone: Mutex::new(Vec::new()),
            in_call: Mutex::new(Vec::new()),
            notify: Mutex::new(None),
        }
    }

    /// The session's notify driver, if the host registered one and it is still there.
    pub(crate) fn notify(&self) -> Option<Arc<super::job_notify::SessionNotify>> {
        lock(&self.notify).as_ref().and_then(Weak::upgrade)
    }

    /// Register (or clear) the session's notify driver.
    pub fn set_notify(&self, notify: Option<&Arc<super::job_notify::SessionNotify>>) {
        *lock(&self.notify) = notify.map(Arc::downgrade);
    }

    /// The job the sandbox knows as `id`: in the table, or in a tool call's hands.
    fn sandbox_job(&self, id: &str) -> Option<Arc<JobShared>> {
        let is_it = |shared: &JobShared| shared.sandbox().is_some_and(|l| l.id.as_str() == id);
        if let Some(found) = lock(&self.table)
            .values()
            .find(|p| is_it(&p.job.shared))
            .map(|p| p.job.shared.clone())
        {
            return Some(found);
        }
        lock(&self.in_call)
            .iter()
            .filter_map(Weak::upgrade)
            .find(|s| is_it(s))
    }

    /// The reattach still to run, if the session's sandbox has not been read back yet.
    pub(crate) fn pending_reattach(&self) -> Option<Arc<super::bash_sandbox::PendingReattach>> {
        lock(&self.reattach).clone()
    }

    /// Defer (`Some`) or settle (`None`) the reattach. Settling it reports the jobs the reopen note
    /// left unchecked: one the reattach found is in the table now (running, or ended with its notice
    /// due); one it did not is gone, and the next notices say so.
    pub(crate) fn set_pending_reattach(
        &self,
        pending: Option<Arc<super::bash_sandbox::PendingReattach>>,
    ) {
        let settled = pending.is_none();
        *lock(&self.reattach) = pending;
        if !settled {
            return;
        }
        let unchecked = std::mem::take(&mut *lock(&self.unchecked));
        let table = lock(&self.table);
        let gone: Vec<(u64, String)> = unchecked
            .into_iter()
            .filter(|(id, _)| !table.contains_key(id))
            .collect();
        drop(table);
        lock(&self.checked_gone).extend(gone);
    }

    /// Whether the sandbox's jobs are still to be read back — a reopened session that has not used
    /// its sandbox yet.
    pub fn reattach_pending(&self) -> bool {
        lock(&self.reattach).is_some()
    }

    /// Whether job `id` has ended (its result is in), or is not in the table at all.
    pub(crate) fn ended_or_gone(&self, id: u64) -> bool {
        lock(&self.table)
            .get(&id)
            .is_none_or(|p| p.job.shared.ended())
    }

    /// Job `id`, if the reattach just found it gone: taken, so it is reported once.
    pub(crate) fn take_checked_gone(&self, id: u64) -> Option<String> {
        let mut gone = lock(&self.checked_gone);
        let at = gone.iter().position(|(j, _)| *j == id)?;
        Some(gone.remove(at).1)
    }

    /// Journal each local job's processes through `recorder` from now on (`None`: stop). Set by the
    /// host that persists the session, with the session it persists to.
    pub fn set_recorder(&self, recorder: Option<ProcRecorder>) {
        *lock(&self.recorder) = recorder;
    }

    fn record_procs(
        &self,
        group: &super::bash_proc::ProcGroup,
        job: Option<u64>,
        command: Option<&str>,
    ) {
        let Some(recorder) = lock(&self.recorder).clone() else {
            return;
        };
        let mut data = serde_json::json!({
            "tag": group.tag,
            "pgid": group.pgid,
            "start": group.leader_start(),
        });
        if let Some(job) = job {
            data["job"] = job.into();
        }
        if let Some(command) = command {
            data["command"] = command.into();
        }
        recorder(data);
    }

    /// Stop the local jobs a killed process left running: `records` are this session's journaled
    /// [`PROCS_ENTRY_KIND`] entries. A SIGKILL or an OOM kill runs no exit path, so a job's
    /// processes outlive the process that started them; the next process to open the session finds
    /// each by its tag (and its group, while the recorded leader still holds the number — see
    /// [`super::bash_proc::ProcGroup`]) and stops it, SIGTERM then SIGKILL, as any other stop.
    /// What it found is kept for the reopen note ([`Background::resume`]), and returned. A record
    /// whose tag no live process carries is gone: its group number alone proves nothing. Never a
    /// tag this process carries itself (it runs inside that job). Sandbox jobs are never here: the
    /// sandbox owns them, and they are reattached instead. **Blocking.**
    pub fn stop_survivors(&self, records: &[serde_json::Value]) -> Vec<SurvivorFate> {
        let own = super::bash_proc::own_tags();
        /// One job's records merged by tag: (pgid, leader start, job number, command).
        type Recorded = (u32, Option<u64>, Option<u64>, Option<String>);
        let mut by_tag: BTreeMap<String, Recorded> = BTreeMap::new();
        for r in records {
            let (Some(tag), Some(pgid)) = (
                r.get("tag").and_then(serde_json::Value::as_str),
                r.get("pgid")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|p| u32::try_from(p).ok()),
            ) else {
                continue;
            };
            if own.iter().any(|t| t == tag) {
                continue;
            }
            let entry = by_tag
                .entry(tag.to_string())
                .or_insert((pgid, None, None, None));
            entry.1 = entry
                .1
                .or(r.get("start").and_then(serde_json::Value::as_u64));
            entry.2 = entry.2.or(r.get("job").and_then(serde_json::Value::as_u64));
            if entry.3.is_none() {
                entry.3 = r
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
            }
        }
        let mut found = Vec::new();
        for (tag, (pgid, start, job, command)) in by_tag {
            let group = super::bash_proc::ProcGroup::recorded(pgid, start, tag);
            if group.tag_alive() {
                found.push((job, command.unwrap_or_default(), Kill::Procs(group)));
            }
        }
        let kills: Vec<Kill> = found.iter().map(|(_, _, k)| k.clone()).collect();
        let left = super::exec::kill_all(&kills);
        let fates: Vec<SurvivorFate> = found
            .into_iter()
            .zip(left)
            .map(|((job, command, _), survived)| SurvivorFate {
                job,
                command,
                survived,
            })
            .collect();
        lock(&self.survivors).found.extend(fates.iter().cloned());
        fates
    }

    /// Run this session's remote jobs in the sandbox, under `scope` — see
    /// [`crate::exec_endpoint::jobs`]. Set by the host that knows the session.
    pub fn set_scope(&self, scope: Option<JobScope>) {
        *lock(&self.scope) = scope;
    }

    pub fn scope(&self) -> Option<JobScope> {
        lock(&self.scope).clone()
    }

    pub(crate) fn policy(&self) -> EndPolicy {
        *lock(&self.policy)
    }

    /// The highest id handed out, raised to `id` if lower: a reattached job keeps its number.
    pub(crate) fn see_id(&self, id: u64) {
        self.next.fetch_max(id, Ordering::Relaxed);
    }

    /// The model has been told how this sandbox job ended: delete its directory once that is
    /// persisted.
    pub(crate) fn discard_later(&self, link: &SandboxLink) {
        lock(&self.garbage).push(link.clone());
    }

    /// Delete the directories of sandbox jobs whose end has been reported. A host calls it where
    /// the reports are known to be persisted (`serve`: the top of its loop; `run`: its end), so a
    /// replica that dies before then leaves them for the next one to report again.
    pub async fn collect_garbage(&self) {
        let garbage = std::mem::take(&mut *lock(&self.garbage));
        super::bash_sandbox::remove_all(garbage).await;
    }

    /// Park a job read back from the sandbox under its own number. `notified` when its end was
    /// already reported; its shell's exit (if any) always was.
    pub(crate) fn insert_reattached(&self, id: u64, job: Job) {
        self.see_id(id);
        let mut parked = Parked::new(job);
        parked.orphan_notified = true;
        lock(&self.table).insert(id, parked);
    }

    /// The session is over without being ended (a reap, a deploy, a shutdown): stop the jobs this
    /// process owns — this host's, and a sandbox's that were never handed over — and let go of the
    /// ones the sandbox owns, which run on for the next replica to reattach. Says which it stopped.
    pub fn release(&self) -> Vec<StoppedJob> {
        let stopped = self.stop(|p| !p.job.shared.released());
        // What is left the sandbox owns: dropping a job only lets go of its stream.
        let kept = std::mem::take(&mut *lock(&self.table));
        drop(kept);
        stopped
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

    /// The highest id handed out (or seeded from a reopened transcript).
    #[cfg(test)]
    pub(crate) fn last_id(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }

    pub(crate) fn assign_id(&self, job: &mut Job) -> u64 {
        if let Some(id) = job.id {
            return id;
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        job.id = Some(id);
        // Its processes were journaled at spawn; now their record names the job too.
        if let Some(Kill::Procs(group)) = lock(&job.shared.kill).as_ref() {
            self.record_procs(group, Some(id), Some(&job.command));
        }
        id
    }

    /// What a job of `kind` lives until, as the model is told when its call returns. `sandbox`:
    /// the sandbox runs it, so it outlives this session's replica too.
    pub(crate) fn lifetime(&self, kind: JobKind, sandbox: bool) -> &'static str {
        match (kind, *lock(&self.policy), sandbox) {
            (JobKind::Yielded, _, _) => "it is stopped when you end your turn",
            (_, EndPolicy::StopAtEnd, _) => "it is stopped for you when the run ends",
            (_, EndPolicy::KeepRunning, true) => {
                "it keeps running in the sandbox until it exits, you stop it, or the session ends"
            }
            (_, EndPolicy::KeepRunning, false) => "it keeps running after your turn ends",
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
        let job = lock(&self.table).remove(&id).map(|p| p.job)?;
        if job.shared.sandbox().is_some() {
            let mut in_call = lock(&self.in_call);
            in_call.retain(|w| w.strong_count() > 0);
            in_call.push(Arc::downgrade(&job.shared));
        }
        Some(job)
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
        self.resumed.store(false, Ordering::Relaxed);
        lock(&self.survivors).found.clear();
        // The transcript's jobs end here, checked or not: nothing of its reopen is owed any more.
        *lock(&self.reattach) = None;
        lock(&self.unchecked).clear();
        lock(&self.checked_gone).clear();
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
        let policy = self.policy();
        // Jobs the reopen note left unchecked that the sandbox, once checked, no longer had.
        let mut parts: Vec<String> = std::mem::take(&mut *lock(&self.checked_gone))
            .into_iter()
            .map(|(id, command)| {
                format!(
                    "{NOTICE_PREFIX}{id} is no longer running: `{command}`] The sandbox, checked \
                     just now, no longer has it, and how it ended was not recorded. bash_job no \
                     longer knows it; start it again if you still need it."
                )
            })
            .collect();
        let mut table = lock(&self.table);
        for (&id, p) in table.iter_mut() {
            if !p.notice_due() {
                continue;
            }
            let job = &mut p.job;
            let state = lock(&job.shared.state);
            let header = match &state.end {
                Some(end) => {
                    p.notified = true;
                    if let Some(link) = job.shared.sandbox() {
                        self.discard_later(link);
                    }
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
                        // Intended now: the sandbox takes it over, as it does an adoption the
                        // call itself saw.
                        super::bash_sandbox::spawn_keep(job.shared.clone(), id, job.kind, policy);
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

    fn resume(&self, messages: &[agent_core::Message]) -> Option<String> {
        if self.resumed.swap(true, Ordering::Relaxed) {
            return None;
        }
        let read = read_transcript(messages);
        // Never back: after a transcript switch this table keeps counting past its own jobs too.
        self.next.fetch_max(read.max_id, Ordering::Relaxed);
        // A job the sandbox kept running (reattached when the session opened) is in the table:
        // still running, or ended with its notice on the way. Only the rest were stopped.
        // A reattach still pending has not read the sandbox back: a job it may still hold is
        // neither running nor gone as far as anyone knows, and the note says exactly that.
        let unread = self.reattach_pending();
        let table = lock(&self.table);
        let mut found = std::mem::take(&mut lock(&self.survivors).found);
        let mut note = Reopened::default();
        for (id, left) in read.running {
            match table.get(&id) {
                Some(p) if p.job.shared.ended() => {}
                Some(p) => note.running.push((id, p.job.command.clone())),
                None => match found.iter().position(|f| f.job == Some(id)) {
                    Some(i) => {
                        let mut fate = found.remove(i);
                        fate.command = left.command;
                        note.push_found(fate);
                    }
                    // Only an intended job can outlive its stream in the sandbox.
                    None if unread && left.intended => note.unchecked.push((id, left.command)),
                    None => note.gone.push((id, left.command)),
                },
            }
        }
        drop(table);
        if unread {
            *lock(&self.unchecked) = note.unchecked.clone();
        }
        // The rest ran when the process died but the transcript does not show them running: a
        // foreground command still attached, or a yielded job a later prompt would have stopped.
        for fate in found {
            note.push_found(fate);
        }
        reopen_note(&note, self.next.load(Ordering::Relaxed) + 1)
    }
}

/// What a reopened session's first run is told about the jobs its last process left: kept running by
/// the sandbox, stopped now (a killed process left them running), still running because they could
/// not be stopped, or no longer running.
#[derive(Default)]
struct Reopened {
    running: Vec<(u64, String)>,
    /// Kept in the sandbox when the session closed, and not read back yet: the session has not
    /// used its sandbox since it reopened.
    unchecked: Vec<(u64, String)>,
    stopped_now: Vec<SurvivorFate>,
    unstoppable: Vec<SurvivorFate>,
    gone: Vec<(u64, String)>,
}

impl Reopened {
    fn push_found(&mut self, fate: SurvivorFate) {
        if fate.survived.is_some() {
            self.unstoppable.push(fate);
        } else {
            self.stopped_now.push(fate);
        }
    }
}

/// How a found job is named in the note: by its number, or as the command it was.
fn found_line(f: &SurvivorFate, state: &str) -> String {
    match f.job {
        Some(id) => format!("- job {id}: `{}` ({state})", f.command),
        None => format!(
            "- `{}` (a command still running when it closed; {state})",
            f.command
        ),
    }
}

/// The note a reopened session's first run gets about the jobs its last process left
/// ([`Background::resume`]). Each entry says what is true now: kept running in the sandbox (still a
/// job `bash_job` knows), stopped just now (the process that ran it was killed before it could stop
/// it), still running because it could not be stopped, or no longer running (stopped when the session
/// closed, or exited). `None` when there is nothing to say. The one place the note is written.
fn reopen_note(r: &Reopened, next: u64) -> Option<String> {
    if r.running.is_empty()
        && r.unchecked.is_empty()
        && r.stopped_now.is_empty()
        && r.unstoppable.is_empty()
        && r.gone.is_empty()
    {
        return None;
    }
    let mut parts = vec![format!(
        "{REMINDER_PREFIX} This session was closed since it last ran."
    )];
    if !r.running.is_empty() {
        let lines: Vec<String> = r
            .running
            .iter()
            .map(|(id, command)| format!("- job {id}: `{command}` (still running)"))
            .collect();
        let (it, each) = if r.running.len() == 1 {
            ("it", "it exits")
        } else {
            ("them", "each exits")
        };
        parts.push(format!(
            "These jobs kept running in the sandbox, and are still job numbers bash_job knows:\n{}\n\
             Do not start them again. You will be told when {each}; bash_job {{\"job\": N, \
             \"timeout_ms\": 0}} shows what one has printed since, and bash_job {{\"job\": N, \
             \"kill\": true}} stops one you no longer need — or leave {it} running.",
            lines.join("\n")
        ));
    }
    if !r.unchecked.is_empty() {
        let lines: Vec<String> = r
            .unchecked
            .iter()
            .map(|(id, command)| format!("- job {id}: `{command}` (not yet checked)"))
            .collect();
        parts.push(format!(
            "{} running in the sandbox when it closed, and the sandbox has not been checked since:\n{}\n\
             It is checked the first time this session uses the sandbox, and bash_job {{\"job\": N, \
             \"timeout_ms\": 0}} checks one now. You will be told if one has ended or is gone. Do \
             not start {} again before you know.",
            plural(r.unchecked.len(), "This job was", "These jobs were"),
            lines.join("\n"),
            plural(r.unchecked.len(), "it", "them"),
        ));
    }
    if !r.stopped_now.is_empty() {
        let lines: Vec<String> = r
            .stopped_now
            .iter()
            .map(|f| found_line(f, "stopped now"))
            .collect();
        parts.push(format!(
            "The process that ran {} was killed before it could stop {}, so {} still running when \
             the session reopened, and {} stopped now:\n{}",
            plural(r.stopped_now.len(), "this", "these"),
            plural(r.stopped_now.len(), "it", "them"),
            plural(r.stopped_now.len(), "it was", "they were"),
            plural(r.stopped_now.len(), "was", "were"),
            lines.join("\n")
        ));
    }
    if !r.unstoppable.is_empty() {
        let lines: Vec<String> = r
            .unstoppable
            .iter()
            .map(|f| {
                let left = f.survived.as_deref().unwrap_or_default();
                found_line(f, &format!("could not be stopped: {left} still running"))
            })
            .collect();
        parts.push(format!(
            "These were left running by a process that was killed, and could not be stopped (they \
             belong to another user):\n{}",
            lines.join("\n")
        ));
    }
    if !r.gone.is_empty() {
        let lines: Vec<String> = r
            .gone
            .iter()
            .map(|(id, command)| format!("- job {id}: `{command}` (not running)"))
            .collect();
        parts.push(format!(
            "These jobs it left running are no longer running (stopped when the session closed, or \
             exited):\n{}",
            lines.join("\n")
        ));
    }
    if !r.stopped_now.is_empty() || !r.unstoppable.is_empty() || !r.gone.is_empty() {
        parts.push(
            "bash_job no longer knows any of those: do not wait on or kill them; start again what \
             you still need."
                .to_string(),
        );
    }
    parts.push(format!("New jobs are numbered from {next}."));
    Some(parts.join(" "))
}

/// A job a transcript shows running when it ends, for [`reopen_note`].
struct LeftRunning {
    command: String,
    /// Background or adopted: it outlives the run. A yielded one is stopped at the next prompt.
    intended: bool,
}

/// What a reopened transcript says about `bash` jobs: the highest id it used, and the jobs still
/// running as it ends.
#[derive(Default)]
struct TranscriptJobs {
    max_id: u64,
    running: BTreeMap<u64, LeftRunning>,
}

/// Read `bash` jobs out of a transcript, in order: a `bash` result that parks a job starts it; a
/// finish notice, a `bash_job` result that is not "still running", or an eviction ends it; and a
/// new prompt ends the yielded ones (their turn stopped them). The highest id comes from everywhere
/// an id can appear — `bash`/`bash_job` calls and results, notices, a summary's prose, assistant
/// text — so an over-count only skips numbers, never repeats one. Other tools' results (a file
/// that happens to say "job 9") are not read.
fn read_transcript(messages: &[agent_core::Message]) -> TranscriptJobs {
    use agent_core::{ContentBlock, Role};
    let mut calls: std::collections::HashMap<&str, (&str, &serde_json::Value)> =
        std::collections::HashMap::new();
    let mut out = TranscriptJobs::default();
    for m in messages {
        let prompt = m.role == Role::User
            && !m
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
            && m.content.iter().any(|b| {
                matches!(b, ContentBlock::Text { text, .. }
                    if !agent_core::background::is_note(text))
            });
        if prompt {
            out.running.retain(|_, j| j.intended);
        }
        for b in &m.content {
            match b {
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => {
                    calls.insert(id, (name, input));
                    if name == super::bash::JOB_TOOL
                        && let Some(n) = input.get("job").and_then(job_number)
                    {
                        out.see(n);
                    }
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    let Some(&(name, input)) = calls.get(tool_use_id.as_str()) else {
                        continue;
                    };
                    if name == "bash" {
                        out.see_text(content);
                        out.bash_result(content, input);
                    } else if name == super::bash::JOB_TOOL {
                        out.see_text(content);
                        if let Some(n) = input.get("job").and_then(job_number) {
                            out.bash_job_result(n, content);
                        }
                    }
                }
                ContentBlock::Text { text, .. } => {
                    out.see_text(text);
                    if m.role == Role::User {
                        out.notices(text);
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// A `bash_job` input's `job`, read as `bash_job` reads it.
fn job_number(v: &serde_json::Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}

/// The id `s` starts with, and what follows it.
fn leading_id(s: &str) -> Option<(u64, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let id = s[..end].parse().ok()?;
    Some((id, &s[end..]))
}

impl TranscriptJobs {
    fn see(&mut self, id: u64) {
        self.max_id = self.max_id.max(id);
    }

    /// Every `job N` and `"job": N` in `text`, in any case.
    fn see_text(&mut self, text: &str) {
        let bytes = text.as_bytes();
        let mut i = 0;
        while let Some(at) = bytes[i..]
            .windows(3)
            .position(|w| w.eq_ignore_ascii_case(b"job"))
        {
            let start = i + at;
            i = start + 3;
            if start > 0 && bytes[start - 1].is_ascii_alphanumeric() {
                continue;
            }
            // `job` is ASCII, so `i` is a char boundary.
            let rest = text[i..]
                .trim_start_matches('"')
                .trim_start_matches(':')
                .trim_start_matches(' ');
            if let Some((id, _)) = leading_id(rest) {
                self.see(id);
            }
        }
    }

    /// A `bash` result: the job it parked, and one a yield evicted.
    fn bash_result(&mut self, content: &str, input: &serde_json::Value) {
        let command = input
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let command = super::bash_proc::truncate_command(command, COMMAND_CHARS);
        for (marker, intended) in [
            ("Running in the background as job ", true),
            ("it is tracked as job ", true),
            ("The command is still running as job ", false),
        ] {
            if let Some((id, _)) = content
                .find(marker)
                .and_then(|at| leading_id(&content[at + marker.len()..]))
            {
                self.running.insert(
                    id,
                    LeftRunning {
                        command: command.clone(),
                        intended,
                    },
                );
            }
        }
        if let Some((id, rest)) = content
            .find(" Job ")
            .and_then(|at| leading_id(&content[at + 5..]))
            && rest.starts_with(" was stopped:")
        {
            self.running.remove(&id);
        }
    }

    /// A `bash_job` result for job `id`: still running (a wait that returned early, or one whose
    /// command left processes that are the job now), or done with — ended, killed, or unknown.
    fn bash_job_result(&mut self, id: u64, content: &str) {
        if content
            .to_ascii_lowercase()
            .contains(&format!("job {id} is still running"))
        {
            return;
        }
        if content.contains(&format!("it is tracked as job {id} ")) {
            if let Some(j) = self.running.get_mut(&id) {
                j.intended = true;
            }
            return;
        }
        self.running.remove(&id);
    }

    /// Finish notices in a user turn's text: `[job N's command … left … running` makes a yielded
    /// job adopted; any other `[job N …]` ended it.
    fn notices(&mut self, text: &str) {
        let mut rest = text;
        while let Some(at) = rest.find(NOTICE_PREFIX) {
            rest = &rest[at + NOTICE_PREFIX.len()..];
            let Some((id, after)) = leading_id(rest) else {
                continue;
            };
            if after.starts_with("'s command") {
                if let Some(j) = self.running.get_mut(&id) {
                    j.intended = true;
                }
            } else {
                self.running.remove(&id);
            }
        }
    }
}

/// "running 45 s" plus why it is a job.
fn describe_running(job: &Job) -> String {
    let secs = super::format_duration(job.started.elapsed());
    match job.kind {
        JobKind::Yielded => format!("running {secs}; its call returned early"),
        JobKind::Background => format!("running {secs}; started in the background"),
        JobKind::Adopted => format!("running {secs}; left running by its command"),
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
    let secs = super::format_duration(end.at.duration_since(job.started));
    let cmd = &job.command;
    match &end.result {
        Err(e) => format!("{NOTICE_PREFIX}{id} failed after {secs}: `{cmd}`: {e}]"),
        // `bash` sets no deadline; this is an exec endpoint enforcing the one on the wire.
        Ok(r) if r.timed_out => format!(
            "{NOTICE_PREFIX}{id} timed out after {secs} (the sandbox's exec endpoint stopped it): \
             `{cmd}`]"
        ),
        Ok(_) if orphaned => format!(
            "{NOTICE_PREFIX}{id} ended after {secs}: `{cmd}` (the processes its command left \
             running have exited)]"
        ),
        Ok(r) => match (r.code, r.signal) {
            (Some(code), _) => {
                format!("{NOTICE_PREFIX}{id} exited with code {code} after {secs}: `{cmd}`]")
            }
            (None, Some(signal)) => format!(
                "{NOTICE_PREFIX}{id} was killed by {} after {secs}: `{cmd}`]",
                signal_text(signal)
            ),
            (None, None) => format!("{NOTICE_PREFIX}{id} exited after {secs}: `{cmd}`]"),
        },
    }
}

/// A parked job's shell exited, leaving processes running.
fn orphan_notice(id: u64, job: &Job, shell: &ExecResult, left: &str) -> String {
    let secs = super::format_duration(job.started.elapsed());
    let how = match (shell.code, shell.signal) {
        (Some(code), _) => format!("exited with code {code}"),
        (None, Some(signal)) => format!("was killed by {}", signal_text(signal)),
        (None, None) => "exited".to_string(),
    };
    format!(
        "{NOTICE_PREFIX}{id}'s command {how} after {secs} but left {left} running: `{}`] That is \
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

/// How exec v1.4's notify driver sees this table ([`super::job_notify`]).
impl super::job_notify::JobTracker for BashJobs {
    fn facts(&self, id: &crate::exec_endpoint::jobs::JobId) -> super::job_notify::kernel::Facts {
        use super::job_notify::kernel::{Facts, TableJob};
        let facts = |job| Facts {
            open_reconcile_done: !self.reattach_pending(),
            job,
        };
        let is_it = |shared: &JobShared| shared.sandbox().is_some_and(|l| &l.id == id);
        let in_table = lock(&self.table)
            .values()
            .find(|p| is_it(&p.job.shared))
            .map(|p| p.job.shared.ended());
        if let Some(ended) = in_table {
            return facts(if ended {
                TableJob::Ended
            } else {
                TableJob::Running
            });
        }
        let in_call = lock(&self.in_call)
            .iter()
            .filter_map(Weak::upgrade)
            .find(|s| is_it(s))
            .map(|s| s.ended());
        match in_call {
            Some(false) => facts(TableJob::Taken),
            Some(true) => facts(TableJob::Ended),
            None if lock(&self.garbage).iter().any(|l| &l.id == id) => facts(TableJob::Ended),
            None => facts(TableJob::Absent),
        }
    }

    fn list(
        &self,
    ) -> futures::future::BoxFuture<
        'static,
        Option<std::collections::BTreeMap<String, super::job_notify::kernel::Listed>>,
    > {
        use super::job_notify::kernel::Listed;
        use crate::exec_endpoint::jobs::JobStatus;
        // The sandbox of the session's jobs, through any job it runs: an unheld job is one.
        let link = lock(&self.table)
            .values()
            .find_map(|p| p.job.shared.sandbox().cloned())
            .or_else(|| {
                lock(&self.in_call)
                    .iter()
                    .filter_map(Weak::upgrade)
                    .find_map(|s| s.sandbox().cloned())
            });
        Box::pin(async move {
            let link = link?;
            let listed = match link.store.list(&link.scope).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::info!(error = %e, "a live reconcile could not list the sandbox's jobs");
                    return None;
                }
            };
            Some(
                listed
                    .into_iter()
                    .filter_map(|j| {
                        let state = match j.status {
                            JobStatus::Running => Listed::Running,
                            JobStatus::ShellExited(code) => Listed::Exited { code },
                            JobStatus::Ended(code) => Listed::Ended { code, at: j.ended },
                            JobStatus::Lost => Listed::Lost {
                                at: j.ended.max(j.started),
                            },
                            JobStatus::Gone => return None,
                        };
                        Some((j.id.as_str().to_string(), state))
                    })
                    .collect(),
            )
        })
    }

    fn perform(&self, action: super::job_notify::kernel::Action) {
        use super::bash_sandbox::Ctl;
        use super::job_notify::kernel::Action;
        let job = match &action {
            Action::CloseStream { job }
            | Action::ApplyShellExit { job, .. }
            | Action::Follow { job, .. }
            | Action::ReadInThenFollow { job, .. } => job.clone(),
            // A `keep` is the keeping caller's to send; nothing else here concerns a job.
            Action::SendKeep { .. } | Action::StartReconcile { .. } | Action::Answer { .. } => {
                return;
            }
        };
        let Some(shared) = self.sandbox_job(&job) else {
            return;
        };
        match action {
            Action::CloseStream { .. } => {
                shared.ctl(Ctl::Close);
            }
            Action::Follow { end, .. } => {
                shared.ctl(Ctl::Follow(end));
            }
            Action::ReadInThenFollow { .. } => {
                shared.ctl(Ctl::ReadIn);
            }
            // As a held stream's `shell_exit`: whether it is news is today's per-job record's.
            Action::ApplyShellExit { code, .. } => shared.orphaned(
                &ExecResult {
                    code: Some(code),
                    ..Default::default()
                },
                super::bash_sandbox::SANDBOX_LEFTOVERS.to_string(),
            ),
            _ => {}
        }
    }
}
