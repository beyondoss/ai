//! Stateright model of exec notify's job-end notify: "Sandbox job notify" in
//! `src/exec_endpoint.rs` (N…) and FLEET.md's "Job-end callbacks" (E…).
//!
//! It checks every interleaving of one session's kept jobs across a ring of two replicas, within
//! bounds: callbacks lost, duplicated, delayed, reordered, replayed and forged; the edge's route
//! and ring walk across a ring that changes; replicas that crash, restart and take the session
//! over (the epoch fence as an abstract owner token); the session opened, closed or reaped, and
//! switched to another transcript; `bash_job` calls; and the sandbox's jobs running, their shells
//! exiting, their ends, and the sandbox dying under them.
//!
//! ## Properties
//!
//! Safety, in every reachable state: (S1) a session task reports each end, and each shell exit, at
//! most once; (S2) a job is unheld only after a `keep` with notify succeeded and a verified
//! callback arrived inside its proof window; (S3) a callback never opens a session; (S4) a forged
//! callback changes no state; (S5) a superseded owner never commits a report; (S6) nothing on a
//! timer alone contacts the sandbox: every `list` answers a callback, a release or a real use;
//! (S7) a job has at most one attempt in flight; (S8) a report lands only in its job's own
//! transcript.
//!
//! Liveness (L1), checked in every settled state (no action left): every end the open session
//! tracks has been reported, by its stream, a callback or the next real use after it. A session that is not open
//! learns of its jobs' ends when it next opens, as today; the model always reopens it, so L1 covers
//! that read-back too.
//!
//! ## The faithfulness contract
//!
//! Every notify decision is made by the REAL kernels, the functions production calls:
//!
//! - the endpoint's notifier, `exec_endpoint::notify_kernel::step` (one notification per job,
//!   what is owed and settled, retry or stop by status, the schedule, abandon on a new `keep`);
//! - the replica's session state, `tools::job_notify::kernel::step` (the release table, the
//!   callback answers, one live reconcile at a time and its spacing, the hand-over's re-check).
//!
//! The model owns only the environment, and the environment contains no copy of either kernel's
//! branching: the only decisions are the two `step` calls in [`Notify::notifier`] and
//! [`Notify::replica`] (and [`Notify::slow_reconcile`] asks the kernel, without keeping its
//! answer, whether a tick would time a callback out). What the environment adds is what lives
//! outside both kernels in production too: the network and the edge, the storage fence, the
//! sandbox's truth, today's stream and follow, today's job table (the [`kernel::Facts`] a callback
//! carries) and today's per-job record of what was reported (`JobShared`).
//!
//! ## Time
//!
//! Both kernels read no clock: time is an input. Each replica has its own clock, which moves only
//! to the replica kernel's `next_wake()` ([`Act::Advance`]) or before a real use; [`Act::Tick`]
//! then delivers a deadline, so a callback can also arrive after its deadline passed and before
//! its tick, as a late timer allows. The notifier's waits end when [`Act::NotifierTimer`] says (it
//! wakes at exactly the time the kernel asked for, on a clock of 0, with a draw of 0). Every other
//! timer (the edge's 25 s, the supervisor's 30 s, a session's idle reap) is a nondeterministic
//! action, so the model covers every order of them.
//!
//! ## Honest boundary
//!
//! - **Today's code is modelled, not called.** The stream and today's follow deliver a job's
//!   `shell_exit` and end; `JobShared`'s once-record keeps a session task from reporting either
//!   twice; a report commits to the transcript through the epoch fence and then removes the job's
//!   record from the sandbox. These are pre-notify behaviour with their own tests. S1 and S5 hold
//!   because of them; their mutations show the properties are not vacuous and that notify's new
//!   sources really lean on them.
//! - **The receiver's checks before the kernel** (N21–N24: classify, verify the token, look up the
//!   session) are modelled as their outcome: a forged token is refused `401` before the session is
//!   looked up, and a replica without the session answers `503`. Their wire-level tests are
//!   `serve_job_ended.rs` and `notify_token.rs`.
//! - **No output is modelled.** Offsets, exit codes and a listing's times are 0. The kernels pass
//!   them through without deciding on them.
//! - **One session.** Sessions share nothing in either kernel (a kernel per session, a notifier
//!   per job), so one session is the sound unit.
//! - **One real use per end, and it must be enough.** The session's current owner makes one real
//!   use after each end, at any moment: at once (a reconcile running, a callback waiting, less
//!   than 30 s after a reconcile ended) or after an idle spell. A use inside N29's 30 s asks for a
//!   reconcile at the 30 s mark, which [`Act::Advance`] reaches.
//! - **A transcript switch waits for the turn**: never during a `bash_job` call.
//!
//! ## Bounds, and why they are enough
//!
//! - **Two replicas** are the least that can disagree about who holds the session: a stale owner
//!   and a new one, a ring target and a walk.
//! - **One job** exercises every release row and every answer. Two add the shared reconcile batch
//!   and a hand-over over several jobs; they run with `NOTIFY_MODEL_FULL=1`.
//! - **A chaos budget**, spent by adversary moves in four classes: the network ([`NET`]: lose,
//!   abandon at 30 s with the request still on its way, replay a real token, forge one, time out
//!   at the edge), ownership ([`OWN`]: crash, close or reap, take over, change the ring), the
//!   sandbox ([`SBX`]: a failed or refused `keep`, the sandbox's death, a switch) and a reconcile
//!   slower than a callback's 20 s ([`SLOW`]). Every protocol step, a restart and a reopen are
//!   free, so every bounded run settles. CI checks any one move of any class, two of the network,
//!   two of the sandbox, and one network move with a `bash_job` call (about 6 M states in all,
//!   seconds). `NOTIFY_MODEL_FULL=1` adds any two moves of any class (about 85 M states), two
//!   ownership moves (25 M) and two jobs (46 M), each minutes. Every mutation's
//!   counterexample needs at most one move, so the default budget reaches every guard.
//! - **A count of attempts runs to [`RETRY_CAP`] (3), not 12.** The kernel resets a count on
//!   every answer that settles it, so a success at attempt 5 leaves the state a success at
//!   attempt 2 leaves; a reopen re-keeps the job, which abandons the count and begins a new one;
//!   and a count that never succeeds is the spec's accepted loss either way. Attempts 4 to 12 add
//!   only later instances of what attempts 2 and 3 already cover.
//!
//! ## Faithful vs mutated
//!
//! The faithful configs feed both kernels honestly. Each `mutation_*` test turns off one guard,
//! by lying to a kernel through its inputs or by switching off the piece of environment the
//! property names, never by editing a kernel, and asserts the checker finds that property's
//! counterexample and no other safety property's (but those the same missing guard must break).

#![allow(clippy::panic)]

use std::collections::BTreeMap;

use beyond_agent::exec_endpoint::notify_kernel as nk;
use beyond_agent::tools::job_notify::kernel;
use stateright::{Checker, Model, Property};

/// Replicas in the ring.
const R: usize = 2;

/// The model lets a count of attempts run to this many; past it, the supervisor's next wait
/// outlasts the run. See "Bounds" for why that loses no behaviour a property can see.
const RETRY_CAP: u32 = 3;

/// The spec's proof window (N37): it closes 30 s after `keep` is answered. The S2 ghost judges a
/// proof by this, not by the kernel's constant.
const SPEC_PROOF_WINDOW_MS: u64 = 30_000;

/// The spec's quiet time before a use starts a reconcile (N29).
const SPEC_USE_QUIET_MS: u64 = 30_000;

// ---------------------------------------------------------------------------------------------
// Environment state
// ---------------------------------------------------------------------------------------------

/// A job's truth in the sandbox: what `list` would report.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Phase {
    Unstarted,
    Running,
    /// The shell exited and a member of its group runs on (`exited`).
    Exited,
    /// Its group is empty (`ended`), or it was killed.
    Ended,
    /// The sandbox died under it.
    Lost,
}

impl Phase {
    fn over(self) -> bool {
        matches!(self, Phase::Ended | Phase::Lost)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SbJob {
    phase: Phase,
    /// The transcript the job was started on.
    transcript: u8,
    /// Its record is gone from the sandbox: removed once its end was reported, or wiped.
    removed: bool,
    /// Wiped by a transcript switch: no report is owed for it anywhere.
    wiped: bool,
}

/// Who sent a callback.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum Origin {
    /// The supervisor's attempt: its answer goes back to the notifier.
    Attempt,
    /// A copy nobody waits for: an attempt abandoned (at 30 s, or by a new `keep`) that the
    /// network still delivers, or a replay of a real token by a program in the sandbox.
    Orphan,
    /// A token that does not verify.
    Forged,
}

/// Where a callback is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum At {
    /// On its way to the edge.
    Edge,
    /// Waiting in replica `r`'s session task (`epoch`) for a live reconcile, as `call`.
    Waiting { r: u8, epoch: u8, call: u8 },
    /// The answer on its way back: `None` is a connection error or no answer in 30 s.
    Back(Option<u16>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
struct Msg {
    job: u8,
    origin: Origin,
    at: At,
}

/// What carries a job's frames in a session task.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Link {
    None,
    /// The start's stream, or today's follow (a broken stream becomes one, and a follow is what a
    /// release row hands an end to): either delivers the job's `shell_exit` and its end.
    Stream,
    /// Being closed after a release; it may still deliver a frame already read.
    Closing,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Recon {
    Idle,
    /// `StartReconcile`: the `list` is on its way.
    Begun(kernel::ReconcileId),
    /// Listed (`Some`) or failed to (`None`); the hand-over has not happened yet.
    Done(kernel::ReconcileId, Option<Vec<(usize, kernel::Listed)>>),
}

/// Per job, the environment's side of one session task.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct TaskJob {
    /// Today's job table tracks it (this task started it, or its open reconcile adopted it).
    tracked: bool,
    link: Link,
    /// A `keep` in flight, with `notify` or not.
    keep: Option<bool>,
    /// Ghost: a `keep` with notify was sent, so its window is open until 30 s after the answer.
    window: bool,
    keep_answered_at: Option<u64>,
    /// Ghost: a `keep` with notify succeeded.
    keep_notify_ok: bool,
    /// Ghost: a verified callback reached this task inside the window.
    proof: bool,
    /// Ghost: the kernel held it unheld at some point.
    was_unheld: bool,
    /// Today's per-job record: ends reported, `shell_exit`s applied (counted, for S1).
    reports: u8,
    exits: u8,
    /// A `bash_job` call has it.
    in_call: bool,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Task {
    /// The owner token: the storage epoch the task opened at.
    epoch: u8,
    transcript: u8,
    opened: bool,
    open_ok: bool,
    now: u64,
    rk: kernel::Kernel,
    jobs: Vec<TaskJob>,
    recon: Recon,
    /// A real use is due: an end happened since the last use.
    use_due: bool,
    /// Ghost for S6: something other than time asked for the next reconcile.
    asked: bool,
    /// When this task's last live reconcile handed over, on its clock.
    last_reconcile: Option<u64>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct St {
    sb_alive: bool,
    job: Vec<SbJob>,
    nk: Vec<nk::Job>,
    net: Vec<Msg>,
    ring: u8,
    up: [bool; R],
    task: [Option<Task>; R],
    epoch: u8,
    transcript: u8,
    chaos: u8,
    calls: u8,
    bad: u16,
    saw: u16,
}

// Safety latches.
const S1_TWICE: u16 = 1 << 0;
const S2_EARLY: u16 = 1 << 1;
const S3_OPENED: u16 = 1 << 2;
const S4_FORGED: u16 = 1 << 3;
const S5_FENCED: u16 = 1 << 4;
const S6_TIMER: u16 = 1 << 5;
const S7_TWO: u16 = 1 << 6;
const S8_TRANSCRIPT: u16 = 1 << 7;

// Witnesses.
const W_UNHELD: u16 = 1 << 0;
const W_UNHELD_END_REPORTED: u16 = 1 << 1;
const W_STALE_TOOK: u16 = 1 << 2;
const W_WALK: u16 = 1 << 3;
const W_FORGED_REFUSED: u16 = 1 << 4;
const W_TAKEOVER_ADOPTS: u16 = 1 << 5;
const W_CALL_ON_UNHELD: u16 = 1 << 6;

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

/// The one guard a mutation turns off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mutation {
    None,
    /// S1: drop today's per-job record of what was reported.
    NoOnceRecord,
    /// S2: tell the replica kernel every `keep` succeeded.
    KeepAlwaysOk,
    /// S2: tell the replica kernel each callback arrived at time 0, so its window never closes.
    WindowNeverCloses,
    /// S3: a replica without the session opens it for a callback.
    CallbackOpens,
    /// S4: the receiver skips the token check.
    NoTokenCheck,
    /// S5: a report commits without the epoch fence.
    NoFence,
    /// S6: a poll timer feeds the replica kernel a use.
    PollTimer,
    /// S7: tell the notifier its attempt ended while it is still in flight.
    EarlyAnswer,
    /// S8: never tell the replica kernel the transcript switched.
    HideSwitch,
    /// L1: drop a real use that arrives while a reconcile runs.
    DropUseDuringReconcile,
    /// L1: drop a real use that arrives less than 30 s after a reconcile ended, none running.
    DropEarlyUse,
}

// Adversary classes: which moves the chaos budget may buy.
/// The network: lose, abandon (a late duplicate), replay a real token, forge one, time out at
/// the edge.
const NET: u8 = 1 << 0;
/// Ownership: crash, close or reap, take over, change the ring.
const OWN: u8 = 1 << 1;
/// The sandbox: a failed or refused `keep`, the sandbox's death, a transcript switch.
const SBX: u8 = 1 << 2;
/// A live reconcile slower than a callback's 20 s.
const SLOW: u8 = 1 << 3;
const ALL: u8 = NET | OWN | SBX | SLOW;

#[derive(Clone, Copy)]
struct Notify {
    jobs: usize,
    chaos: u8,
    /// The adversary classes the budget may buy.
    adversary: u8,
    /// `bash_job` calls the session may make.
    calls: u8,
    /// Let the one real use come less than 30 s after a reconcile ended (N29 then asks for one at
    /// the 30 s mark). On in every faithful config; off only to compare.
    use_in_quiet_window: bool,
    mutation: Mutation,
}

impl Notify {
    fn new(jobs: usize, chaos: u8) -> Self {
        Notify {
            jobs,
            chaos,
            adversary: ALL,
            calls: 1,
            use_in_quiet_window: true,
            mutation: Mutation::None,
        }
    }

    fn mutated(self, mutation: Mutation) -> Self {
        Notify { mutation, ..self }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Keep {
    Ok,
    /// `400`: nothing recorded.
    Refused,
    /// Failed, nothing recorded.
    FailedUnrecorded,
    /// Failed, but the sandbox recorded it (the answer was lost).
    FailedRecorded,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Act {
    StartJob(u8, u8),
    ShellExit(u8),
    JobEnd(u8),
    SandboxDies,
    NotifierTimer(u8),
    Route(u8),
    Answer(u8),
    Lose(u8),
    Abandon(u8),
    Replay(u8),
    Forge(u8),
    EdgeTimeout(u8),
    RingFlip,
    Restart(u8),
    Crash(u8),
    Open(u8),
    Takeover(u8),
    Close(u8),
    OpenReconcile(u8),
    KeepAnswer(u8, u8, Keep),
    StreamExit(u8, u8),
    StreamEnd(u8, u8),
    StreamClosed(u8, u8),
    Advance(u8),
    Tick(u8),
    List(u8),
    Handover(u8),
    /// A real use, at once (`false`) or after an idle spell (`true`).
    Use(u8, bool),
    CallBegin(u8, u8),
    CallEnd(u8, u8),
    Switch(u8),
    Poll(u8),
}

fn job_id(j: usize) -> String {
    format!("{j:016x}")
}

fn job_index(id: &str) -> usize {
    usize::from_str_radix(id, 16).unwrap_or_else(|_| unreachable!("the model names every job"))
}

impl Notify {
    fn current_task(&self, s: &St) -> Option<usize> {
        (0..R).find(|&r| s.up[r] && s.task[r].as_ref().is_some_and(|t| t.epoch == s.epoch))
    }

    /// N36: notify is available in a task on the transcript its session id names.
    fn notify_available(t: &Task) -> bool {
        t.transcript == 0
    }

    // ---- the notifier: the only call into `notify_kernel` ----------------------------------

    fn notifier(&self, s: &mut St, j: usize, event: nk::Event) {
        if !s.sb_alive {
            return; // the supervisors died with the sandbox
        }
        let now = match event {
            nk::Event::Wake => s.nk[j].wake_at.unwrap_or(0),
            _ => 0,
        };
        let (next, actions) = nk::step(s.nk[j], event, now);
        s.nk[j] = next;
        for a in actions {
            match a {
                nk::Action::Begin { .. } => {
                    if s.net
                        .iter()
                        .any(|m| m.job as usize == j && m.origin == Origin::Attempt)
                    {
                        s.bad |= S7_TWO;
                    }
                    s.net.push(Msg {
                        job: j as u8,
                        origin: Origin::Attempt,
                        at: At::Edge,
                    });
                }
                nk::Action::Abandon => {
                    // The connection is dropped; the request may already be on its way.
                    for m in s.net.iter_mut() {
                        if m.job as usize == j && m.origin == Origin::Attempt {
                            m.origin = Origin::Orphan;
                        }
                    }
                    s.net
                        .retain(|m| !(m.origin == Origin::Orphan && matches!(m.at, At::Back(_))));
                }
                nk::Action::WakeAt(_) => {} // `Act::NotifierTimer` is enabled while it waits
            }
        }
    }

    // ---- the replica: the only call into `job_notify::kernel` -------------------------------

    /// Feed task `r` one event and carry out its actions. `timer` marks an event only time caused.
    /// `call_msg` is the network message a `Callback` event stands for.
    fn replica(
        &self,
        s: &mut St,
        r: usize,
        event: kernel::Event,
        timer: bool,
        call_msg: Option<usize>,
    ) {
        let Some(t) = s.task[r].as_mut() else { return };
        // S6's ghost: the events that may ask for a reconcile, read off the kernel's public state.
        let asks = match &event {
            kernel::Event::Callback { job, .. } => {
                matches!(t.rk.job(job), Some(kernel::JobState::Unheld { .. }))
            }
            kernel::Event::StreamClosed { job, .. } => {
                matches!(t.rk.job(job), Some(kernel::JobState::Closing))
            }
            kernel::Event::Use { .. } => !timer && t.rk.has_unheld(),
            _ => false,
        };
        if asks {
            t.asked = true;
        }
        let (next, actions) = kernel::step(&t.rk, event);
        t.rk = next;
        for a in actions {
            self.perform(s, r, a, timer, call_msg);
        }
    }

    fn perform(
        &self,
        s: &mut St,
        r: usize,
        a: kernel::Action,
        timer: bool,
        call_msg: Option<usize>,
    ) {
        let Some(t) = s.task[r].as_mut() else { return };
        match a {
            kernel::Action::Answer { call, status, .. } => {
                let epoch = t.epoch;
                let waiting = s.net.iter().position(|m| {
                    m.at == At::Waiting {
                        r: r as u8,
                        epoch,
                        call: call as u8,
                    }
                });
                if let Some(i) = waiting.or(call_msg) {
                    s.net[i].at = At::Back(Some(status));
                }
            }
            kernel::Action::SendKeep { job, notify } => {
                let j = job_index(&job);
                t.jobs[j].keep = Some(notify);
                if notify {
                    t.jobs[j].window = true;
                    t.jobs[j].keep_answered_at = None;
                }
            }
            kernel::Action::CloseStream { job } => {
                let j = job_index(&job);
                t.jobs[j].link = Link::Closing;
            }
            kernel::Action::StartReconcile { reconcile } => {
                if !t.asked {
                    s.bad |= S6_TIMER;
                }
                t.asked = false;
                t.recon = Recon::Begun(reconcile);
            }
            kernel::Action::ApplyShellExit { job, .. } => {
                let j = job_index(&job);
                self.apply_exit(s, r, j);
            }
            kernel::Action::Follow { job, .. } => {
                if timer {
                    s.bad |= S6_TIMER;
                }
                let j = job_index(&job);
                t.jobs[j].link = Link::Stream;
            }
            kernel::Action::ReadInThenFollow { .. } => {
                // Part of the call in flight: `CallEnd` reads what the sandbox holds.
                s.saw |= W_CALL_ON_UNHELD;
            }
            kernel::Action::CloseFollow { job } => {
                // The call's follow ends with the call; the job is unheld again.
                t.jobs[job_index(&job)].link = Link::None;
            }
            kernel::Action::Held { .. } => {} // a metric: the job stays as it is
        }
    }

    // ---- today's paths, as environment ------------------------------------------------------

    fn apply_exit(&self, s: &mut St, r: usize, j: usize) {
        let Some(t) = s.task[r].as_mut() else { return };
        if t.jobs[j].exits > 0 && self.mutation != Mutation::NoOnceRecord {
            return; // today's per-job record: applied once
        }
        t.jobs[j].exits += 1;
        if t.jobs[j].exits > 1 {
            s.bad |= S1_TWICE;
        }
    }

    /// Today's report of job `j`'s end by task `r`: `JobShared`'s once-record, the commit to the
    /// transcript under the epoch fence, then the record's removal from the sandbox.
    fn report(&self, s: &mut St, r: usize, j: usize) {
        let epoch = s.epoch;
        let Some(t) = s.task[r].as_mut() else { return };
        if t.jobs[j].reports > 0 && self.mutation != Mutation::NoOnceRecord {
            return;
        }
        if t.epoch != epoch {
            if self.mutation == Mutation::NoFence {
                s.bad |= S5_FENCED;
            } else {
                // The fenced owner notices at its commit and lets the session go.
                self.drop_task(s, r);
                return;
            }
        }
        let Some(t) = s.task[r].as_mut() else { return };
        t.jobs[j].reports += 1;
        if t.jobs[j].reports > 1 {
            s.bad |= S1_TWICE;
        }
        if s.job[j].transcript != t.transcript {
            s.bad |= S8_TRANSCRIPT;
        }
        if t.jobs[j].was_unheld {
            s.saw |= W_UNHELD_END_REPORTED;
        }
        if s.sb_alive && !s.job[j].removed {
            s.job[j].removed = true;
            self.notifier(s, j, nk::Event::Removed);
        }
    }

    fn drop_task(&self, s: &mut St, r: usize) {
        s.task[r] = None;
        for m in s.net.iter_mut() {
            if let At::Waiting { r: wr, .. } = m.at
                && wr as usize == r
            {
                m.at = At::Back(Some(503));
            }
        }
    }

    fn new_task(&self, s: &St) -> Task {
        Task {
            epoch: s.epoch,
            transcript: s.transcript,
            opened: false,
            open_ok: false,
            now: 0,
            rk: kernel::Kernel::new(),
            jobs: vec![
                TaskJob {
                    tracked: false,
                    link: Link::None,
                    keep: None,
                    window: false,
                    keep_answered_at: None,
                    keep_notify_ok: false,
                    proof: false,
                    was_unheld: false,
                    reports: 0,
                    exits: 0,
                    in_call: false,
                };
                self.jobs
            ],
            recon: Recon::Idle,
            use_due: false,
            asked: false,
            last_reconcile: None,
        }
    }

    fn mark_use_due(s: &mut St, j: usize) {
        for t in s.task.iter_mut().flatten() {
            if t.jobs[j].tracked {
                t.use_due = true;
            }
        }
    }

    /// What today's job table says of job `j` (the facts a callback carries).
    fn facts(t: &Task, j: usize) -> kernel::Facts {
        let tj = &t.jobs[j];
        let job = if tj.in_call {
            kernel::TableJob::Taken
        } else if tj.reports > 0 {
            kernel::TableJob::Ended
        } else if tj.tracked {
            kernel::TableJob::Running
        } else {
            kernel::TableJob::Absent
        };
        kernel::Facts {
            open_reconcile_done: t.opened && t.open_ok,
            job,
        }
    }

    /// Deliver a verified callback (message `i`) to replica `r`'s session task.
    fn deliver(&self, s: &mut St, i: usize, r: usize) {
        let j = s.net[i].job as usize;
        let epoch = s.epoch;
        let forged = s.net[i].origin == Origin::Forged;
        let used: Vec<u8> = s
            .net
            .iter()
            .filter_map(|m| match m.at {
                At::Waiting { r: wr, call, .. } if wr as usize == r => Some(call),
                _ => None,
            })
            .collect();
        let Some(t) = s.task[r].as_mut() else { return };
        if t.epoch != epoch {
            s.saw |= W_STALE_TOOK;
        }
        // S2's ghost: the spec's window, judged by the environment's clock.
        let now = t.now;
        let tj = &mut t.jobs[j];
        let inside = tj.window
            && tj
                .keep_answered_at
                .is_none_or(|at| now < at + SPEC_PROOF_WINDOW_MS);
        if inside && !forged {
            tj.proof = true;
        }
        // The callback waits as the smallest call number free in this task.
        let call = (0u8..).find(|c| !used.contains(c)).unwrap_or(0);
        s.net[i].at = At::Waiting {
            r: r as u8,
            epoch: t.epoch,
            call,
        };
        let now = if self.mutation == Mutation::WindowNeverCloses {
            0
        } else {
            now
        };
        let event = kernel::Event::Callback {
            call: u64::from(call),
            job: job_id(j),
            facts: Self::facts(t, j),
            now,
        };
        self.replica(s, r, event, false, Some(i));
    }

    /// After every step: S2, and the witnesses that read the kernel's public state.
    fn observe(&self, s: &mut St) {
        for t in s.task.iter_mut().flatten() {
            for j in 0..self.jobs {
                if matches!(
                    t.rk.job(&job_id(j)),
                    Some(kernel::JobState::Unheld { .. } | kernel::JobState::Taken { .. })
                ) {
                    t.jobs[j].was_unheld = true;
                    s.saw |= W_UNHELD;
                    if !(t.jobs[j].keep_notify_ok && t.jobs[j].proof) {
                        s.bad |= S2_EARLY;
                    }
                }
            }
        }
    }

    fn listing(&self, s: &St) -> Vec<(usize, kernel::Listed)> {
        (0..self.jobs)
            .filter_map(|j| {
                let sj = s.job[j];
                if sj.removed {
                    return None;
                }
                let l = match sj.phase {
                    Phase::Unstarted => return None,
                    Phase::Running => kernel::Listed::Running,
                    Phase::Exited => kernel::Listed::Exited { code: 0 },
                    Phase::Ended => kernel::Listed::Ended { code: 0, at: 0 },
                    Phase::Lost => kernel::Listed::Lost { at: 0 },
                };
                Some((j, l))
            })
            .collect()
    }

    /// Whether time reaching `w` answers a waiting callback `503` (its 20 s ran out) while a
    /// reconcile runs: the reconcile was slower than 20 s. Asked of the kernel itself, without
    /// keeping its answer.
    fn slow_reconcile(rk: &kernel::Kernel, w: u64) -> bool {
        rk.reconciling()
            && kernel::step(rk, kernel::Event::Tick { now: w })
                .1
                .iter()
                .any(|a| matches!(a, kernel::Action::Answer { status: 503, .. }))
    }

    /// A callback waiting in a task whose request is gone: the kernel stops waiting for it.
    fn cancel_waiting(&self, s: &mut St, i: usize) {
        if let At::Waiting { r, epoch, call } = s.net[i].at
            && s.task[r as usize]
                .as_ref()
                .is_some_and(|t| t.epoch == epoch)
        {
            let event = kernel::Event::CallClosed {
                call: u64::from(call),
            };
            self.replica(s, r as usize, event, false, None);
        }
    }
}

impl Model for Notify {
    type State = St;
    type Action = Act;

    fn init_states(&self) -> Vec<St> {
        vec![St {
            sb_alive: true,
            job: vec![
                SbJob {
                    phase: Phase::Unstarted,
                    transcript: 0,
                    removed: false,
                    wiped: false,
                };
                self.jobs
            ],
            nk: vec![nk::Job::default(); self.jobs],
            net: Vec::new(),
            ring: 0,
            up: [true; R],
            task: [None, None],
            epoch: 0,
            transcript: 0,
            chaos: self.chaos,
            calls: self.calls,
            bad: 0,
            saw: 0,
        }]
    }

    fn actions(&self, s: &St, acts: &mut Vec<Act>) {
        let chaos = s.chaos > 0;
        let net = chaos && self.adversary & NET != 0;
        let own = chaos && self.adversary & OWN != 0;
        let sbx = chaos && self.adversary & SBX != 0;
        let slow = chaos && self.adversary & SLOW != 0;
        for j in 0..self.jobs {
            let jb = j as u8;
            let sj = s.job[j];
            if s.sb_alive && !sj.removed {
                if sj.phase == Phase::Running {
                    acts.push(Act::ShellExit(jb));
                }
                if matches!(sj.phase, Phase::Running | Phase::Exited) {
                    acts.push(Act::JobEnd(jb));
                }
            }
            if s.sb_alive
                && s.nk[j].wake_at.is_some()
                && s.nk[j].in_flight.is_none()
                && s.nk[j].attempts < RETRY_CAP
            {
                acts.push(Act::NotifierTimer(jb));
            }
            if net && s.sb_alive && sj.phase != Phase::Unstarted && !sj.removed {
                acts.push(Act::Replay(jb));
                acts.push(Act::Forge(jb));
            }
        }
        if sbx && s.sb_alive && s.job.iter().any(|j| j.phase != Phase::Unstarted) {
            acts.push(Act::SandboxDies);
        }
        for (i, m) in s.net.iter().enumerate() {
            if i > 0 && s.net[i - 1] == *m {
                continue; // identical messages: one stands for both
            }
            let ib = i as u8;
            match m.at {
                At::Edge => acts.push(Act::Route(ib)),
                At::Back(_) => acts.push(Act::Answer(ib)),
                At::Waiting { .. } => {}
            }
            if net {
                acts.push(Act::Lose(ib));
                if m.origin == Origin::Attempt && !matches!(m.at, At::Back(_)) {
                    acts.push(Act::Abandon(ib));
                }
                if matches!(m.at, At::Waiting { .. }) {
                    acts.push(Act::EdgeTimeout(ib));
                }
            }
        }
        if own {
            acts.push(Act::RingFlip);
        }
        let held_somewhere = (0..R).any(|r| s.up[r] && s.task[r].is_some());
        for r in 0..R {
            let rb = r as u8;
            if !s.up[r] {
                acts.push(Act::Restart(rb));
                continue;
            }
            if own {
                acts.push(Act::Crash(rb));
            }
            let Some(t) = &s.task[r] else {
                if !held_somewhere {
                    acts.push(Act::Open(rb));
                } else if own {
                    acts.push(Act::Takeover(rb));
                }
                continue;
            };
            if own {
                acts.push(Act::Close(rb));
            }
            if !t.opened {
                acts.push(Act::OpenReconcile(rb));
                continue;
            }
            // A switch is a client command between turns: never during a `bash_job` call.
            if sbx && t.transcript == s.transcript && !t.jobs.iter().any(|j| j.in_call) {
                acts.push(Act::Switch(rb));
            }
            for j in 0..self.jobs {
                let jb = j as u8;
                let tj = &t.jobs[j];
                let sj = s.job[j];
                if s.sb_alive
                    && sj.phase == Phase::Unstarted
                    && t.transcript == s.transcript
                    && t.epoch == s.epoch
                {
                    acts.push(Act::StartJob(rb, jb));
                }
                if let Some(notify) = tj.keep {
                    acts.push(Act::KeepAnswer(rb, jb, Keep::Ok));
                    if sbx {
                        if notify {
                            acts.push(Act::KeepAnswer(rb, jb, Keep::Refused));
                        }
                        acts.push(Act::KeepAnswer(rb, jb, Keep::FailedUnrecorded));
                        acts.push(Act::KeepAnswer(rb, jb, Keep::FailedRecorded));
                    }
                }
                if matches!(tj.link, Link::Stream | Link::Closing) && !tj.in_call {
                    if sj.phase == Phase::Exited && tj.exits == 0 {
                        acts.push(Act::StreamExit(rb, jb));
                    }
                    if sj.phase.over() {
                        acts.push(Act::StreamEnd(rb, jb));
                    }
                }
                if tj.link == Link::Closing {
                    acts.push(Act::StreamClosed(rb, jb));
                }
                if tj.tracked && !tj.in_call && s.calls > 0 && tj.reports == 0 {
                    acts.push(Act::CallBegin(rb, jb));
                }
                if tj.in_call {
                    acts.push(Act::CallEnd(rb, jb));
                }
            }
            match t.rk.next_wake() {
                Some(w) if w > t.now => {
                    // Time reaching a waiting callback's 20 s limit while a reconcile runs means
                    // a reconcile slower than that: an adversary's move. Every other deadline,
                    // that limit included when no reconcile runs, comes for free.
                    if slow || !Self::slow_reconcile(&t.rk, w) {
                        acts.push(Act::Advance(rb));
                    }
                }
                Some(_) => {
                    if kernel::step(&t.rk, kernel::Event::Tick { now: t.now }).0 != t.rk {
                        acts.push(Act::Tick(rb));
                    }
                }
                None => {}
            }
            match t.recon {
                Recon::Begun(_) => acts.push(Act::List(rb)),
                Recon::Done(..) => acts.push(Act::Handover(rb)),
                Recon::Idle => {}
            }
            // The session's current owner makes one real use after each end, and that one use
            // must be enough (L1). It may come at once, whatever runs (a reconcile, a callback's
            // wait, or less than 30 s after a reconcile ended, with `use_in_quiet_window`), or
            // after an idle spell.
            let quiet = t.rk.next_wake().is_none() && !t.rk.reconciling();
            let spaced = t.rk.reconciling()
                || t.last_reconcile
                    .is_none_or(|end| t.now >= end + SPEC_USE_QUIET_MS);
            if t.use_due && t.epoch == s.epoch {
                if spaced || self.use_in_quiet_window {
                    acts.push(Act::Use(rb, false));
                }
                if quiet {
                    acts.push(Act::Use(rb, true));
                }
            }
            if chaos && self.mutation == Mutation::PollTimer && quiet {
                acts.push(Act::Poll(rb));
            }
        }
    }

    fn next_state(&self, s: &St, a: Act) -> Option<St> {
        let mut s = s.clone();
        let spend = |s: &mut St| s.chaos -= 1;
        match a {
            Act::StartJob(r, j) => {
                let (r, j) = (r as usize, j as usize);
                s.job[j].phase = Phase::Running;
                s.job[j].transcript = s.transcript;
                let t = s.task[r].as_mut()?;
                t.jobs[j].tracked = true;
                t.jobs[j].link = Link::Stream;
                let event = kernel::Event::Kept {
                    job: job_id(j),
                    notify_available: Self::notify_available(t),
                    adopted: false,
                };
                self.replica(&mut s, r, event, false, None);
            }
            Act::ShellExit(j) => {
                let j = j as usize;
                s.job[j].phase = Phase::Exited;
                self.notifier(&mut s, j, nk::Event::Change);
            }
            Act::JobEnd(j) => {
                let j = j as usize;
                s.job[j].phase = Phase::Ended;
                self.notifier(&mut s, j, nk::Event::Change);
                Self::mark_use_due(&mut s, j);
            }
            Act::SandboxDies => {
                spend(&mut s);
                s.sb_alive = false;
                for j in 0..self.jobs {
                    if matches!(s.job[j].phase, Phase::Running | Phase::Exited) {
                        s.job[j].phase = Phase::Lost;
                        Self::mark_use_due(&mut s, j);
                    }
                }
                // Its supervisors die with it: nobody waits for an answer any more.
                for m in s.net.iter_mut() {
                    if m.origin == Origin::Attempt {
                        m.origin = Origin::Orphan;
                    }
                }
                s.net
                    .retain(|m| !(m.origin == Origin::Orphan && matches!(m.at, At::Back(_))));
            }
            Act::NotifierTimer(j) => self.notifier(&mut s, j as usize, nk::Event::Wake),
            Act::Route(i) => {
                // E3, in one step: the hashed target first, then the walk in ring order, all
                // inside one request. A replica that is down, or holds no session task for the
                // session (N24), is a `503` that walks on.
                let i = i as usize;
                let forged = s.net[i].origin == Origin::Forged;
                let order = [s.ring as usize, 1 - s.ring as usize];
                let mut answered = false;
                for (hop, r) in order.into_iter().enumerate() {
                    if !s.up[r] {
                        continue;
                    }
                    if forged && self.mutation != Mutation::NoTokenCheck {
                        // N23 (5): refused before the session is looked for.
                        s.net[i].at = At::Back(Some(401));
                        s.saw |= W_FORGED_REFUSED;
                        answered = true;
                        break;
                    }
                    if s.task[r].is_none() {
                        if self.mutation == Mutation::CallbackOpens {
                            s.bad |= S3_OPENED;
                            s.epoch += 1;
                            s.task[r] = Some(self.new_task(&s));
                        } else {
                            continue;
                        }
                    }
                    if hop > 0 {
                        s.saw |= W_WALK;
                    }
                    let before = s.task.clone();
                    self.deliver(&mut s, i, r);
                    if forged && s.task != before {
                        s.bad |= S4_FORGED;
                    }
                    answered = true;
                    break;
                }
                if !answered {
                    s.net[i].at = At::Back(Some(503));
                }
            }
            Act::Answer(i) => {
                let m = s.net.remove(i as usize);
                let At::Back(code) = m.at else { return None };
                if m.origin == Origin::Attempt {
                    let outcome = match code {
                        Some(c) => nk::Outcome::Status(c),
                        None => nk::Outcome::Failed,
                    };
                    let event = nk::Event::AttemptEnded { outcome, draw: 0 };
                    self.notifier(&mut s, m.job as usize, event);
                }
            }
            Act::Lose(i) => {
                spend(&mut s);
                let i = i as usize;
                self.cancel_waiting(&mut s, i);
                if s.net[i].origin == Origin::Attempt {
                    // The supervisor sees a connection error, or nothing in 30 s.
                    s.net[i].at = At::Back(None);
                } else {
                    s.net.remove(i);
                }
            }
            Act::Abandon(i) => {
                // The supervisor gives up at 30 s, but the request goes on, late, while the
                // supervisor retries: a delayed duplicate.
                spend(&mut s);
                let i = i as usize;
                let mut copy = s.net[i];
                copy.origin = Origin::Orphan;
                if self.mutation == Mutation::EarlyAnswer {
                    let event = nk::Event::AttemptEnded {
                        outcome: nk::Outcome::Failed,
                        draw: 0,
                    };
                    self.notifier(&mut s, copy.job as usize, event);
                } else {
                    s.net[i] = copy;
                    s.net.push(Msg {
                        at: At::Back(None),
                        origin: Origin::Attempt,
                        ..copy
                    });
                }
            }
            Act::Replay(j) => {
                spend(&mut s);
                s.net.push(Msg {
                    job: j,
                    origin: Origin::Orphan,
                    at: At::Edge,
                });
            }
            Act::Forge(j) => {
                spend(&mut s);
                s.net.push(Msg {
                    job: j,
                    origin: Origin::Forged,
                    at: At::Edge,
                });
            }
            Act::EdgeTimeout(i) => {
                // E4: 503 at 25 s, and the request is cancelled upstream.
                spend(&mut s);
                let i = i as usize;
                self.cancel_waiting(&mut s, i);
                s.net[i].at = At::Back(Some(503));
            }
            Act::RingFlip => {
                spend(&mut s);
                s.ring = 1 - s.ring;
            }
            Act::Restart(r) => s.up[r as usize] = true,
            Act::Crash(r) => {
                spend(&mut s);
                let r = r as usize;
                self.drop_task(&mut s, r);
                s.up[r] = false;
            }
            Act::Open(r) => {
                s.epoch += 1;
                s.task[r as usize] = Some(self.new_task(&s));
            }
            Act::Takeover(r) => {
                spend(&mut s);
                s.epoch += 1;
                s.task[r as usize] = Some(self.new_task(&s));
            }
            Act::Close(r) => {
                spend(&mut s);
                self.drop_task(&mut s, r as usize);
            }
            Act::OpenReconcile(r) => {
                let r = r as usize;
                let alive = s.sb_alive;
                let t = s.task[r].as_mut()?;
                t.opened = true;
                t.open_ok = alive;
                let transcript = t.transcript;
                if alive {
                    for j in 0..self.jobs {
                        let sj = s.job[j];
                        if sj.phase == Phase::Unstarted || sj.removed || sj.transcript != transcript
                        {
                            continue;
                        }
                        let Some(t) = s.task[r].as_mut() else { break };
                        t.jobs[j].tracked = true;
                        if sj.phase.over() {
                            // Read back what ended, as today's reattach does.
                            self.report(&mut s, r, j);
                        } else {
                            // Adopt a kept job that runs; its follow stands for the stream (N42).
                            t.jobs[j].link = Link::Stream;
                            if t.epoch > 1 {
                                s.saw |= W_TAKEOVER_ADOPTS;
                            }
                            let event = kernel::Event::Kept {
                                job: job_id(j),
                                notify_available: Self::notify_available(t),
                                adopted: true,
                            };
                            self.replica(&mut s, r, event, false, None);
                        }
                    }
                }
            }
            Act::KeepAnswer(r, j, outcome) => {
                let (r, j) = (r as usize, j as usize);
                if outcome != Keep::Ok {
                    spend(&mut s);
                }
                let reachable = s.sb_alive && !s.job[j].removed;
                let outcome = match outcome {
                    Keep::Ok if !reachable => Keep::FailedUnrecorded,
                    o => o,
                };
                let t = s.task[r].as_mut()?;
                let now = t.now;
                let notify = t.jobs[j].keep.take()?;
                if notify {
                    t.jobs[j].keep_answered_at = Some(now);
                    t.jobs[j].keep_notify_ok = outcome == Keep::Ok;
                }
                if matches!(outcome, Keep::Ok | Keep::FailedRecorded) {
                    self.notifier(&mut s, j, nk::Event::Keep { notify });
                }
                if notify {
                    let told = match outcome {
                        Keep::Ok => kernel::KeepOutcome::Succeeded,
                        _ if self.mutation == Mutation::KeepAlwaysOk => {
                            kernel::KeepOutcome::Succeeded
                        }
                        Keep::Refused => kernel::KeepOutcome::Refused,
                        _ => kernel::KeepOutcome::Failed,
                    };
                    let event = kernel::Event::KeepAnswered {
                        job: job_id(j),
                        outcome: told,
                        now,
                    };
                    self.replica(&mut s, r, event, false, None);
                }
            }
            Act::StreamExit(r, j) => self.apply_exit(&mut s, r as usize, j as usize),
            Act::StreamEnd(r, j) => {
                let (r, j) = (r as usize, j as usize);
                s.task[r].as_mut()?.jobs[j].link = Link::None;
                self.report(&mut s, r, j);
                let event = kernel::Event::StreamEnded { job: job_id(j) };
                self.replica(&mut s, r, event, false, None);
            }
            Act::StreamClosed(r, j) => {
                let (r, j) = (r as usize, j as usize);
                let t = s.task[r].as_mut()?;
                t.jobs[j].link = Link::None;
                let event = kernel::Event::StreamClosed {
                    job: job_id(j),
                    offset: 0,
                    shell_exit_seen: t.jobs[j].exits > 0,
                    now: t.now,
                };
                self.replica(&mut s, r, event, false, None);
            }
            Act::Advance(r) => {
                let t = s.task[r as usize].as_mut()?;
                let w = t.rk.next_wake()?;
                let slow = Self::slow_reconcile(&t.rk, w);
                t.now = w;
                if slow {
                    spend(&mut s);
                }
            }
            Act::Tick(r) => {
                let r = r as usize;
                let now = s.task[r].as_ref()?.now;
                self.replica(&mut s, r, kernel::Event::Tick { now }, true, None);
            }
            Act::List(r) => {
                let r = r as usize;
                let listing = s.sb_alive.then(|| self.listing(&s));
                let t = s.task[r].as_mut()?;
                let Recon::Begun(id) = t.recon else {
                    return None;
                };
                t.recon = Recon::Done(id, listing);
            }
            Act::Handover(r) => {
                let r = r as usize;
                let t = s.task[r].as_mut()?;
                let Recon::Done(id, listing) = std::mem::replace(&mut t.recon, Recon::Idle) else {
                    return None;
                };
                let jobs = listing.map(|l| {
                    l.into_iter()
                        .map(|(j, x)| (job_id(j), x))
                        .collect::<BTreeMap<_, _>>()
                });
                t.last_reconcile = Some(t.now);
                let event = kernel::Event::Listed {
                    reconcile: id,
                    jobs,
                    now: t.now,
                };
                self.replica(&mut s, r, event, false, None);
            }
            Act::Use(r, idle) => {
                let r = r as usize;
                let t = s.task[r].as_mut()?;
                if idle && let Some(end) = t.last_reconcile {
                    t.now = t.now.max(end + SPEC_USE_QUIET_MS);
                }
                t.use_due = false;
                let now = t.now;
                // The mutations drop a use that arrives while a reconcile runs, or less than 30 s
                // after one ended, as the kernel did before such a use asked for a reconcile.
                let early = !t.rk.reconciling()
                    && t.last_reconcile
                        .is_some_and(|end| now < end + SPEC_USE_QUIET_MS);
                let dropped = (self.mutation == Mutation::DropUseDuringReconcile
                    && t.rk.reconciling())
                    || (self.mutation == Mutation::DropEarlyUse && early);
                if !dropped {
                    self.replica(&mut s, r, kernel::Event::Use { now }, false, None);
                }
            }
            Act::Poll(r) => {
                // A timer fires on its schedule, so time passes as before a use.
                spend(&mut s);
                let r = r as usize;
                let t = s.task[r].as_mut()?;
                if let Some(end) = t.last_reconcile {
                    t.now = t.now.max(end + SPEC_USE_QUIET_MS);
                }
                let now = t.now;
                self.replica(&mut s, r, kernel::Event::Use { now }, true, None);
            }
            Act::CallBegin(r, j) => {
                let (r, j) = (r as usize, j as usize);
                s.calls -= 1;
                s.task[r].as_mut()?.jobs[j].in_call = true;
                let event = kernel::Event::Taken { job: job_id(j) };
                self.replica(&mut s, r, event, false, None);
            }
            Act::CallEnd(r, j) => {
                // The call reads what the sandbox holds, and returns with the job running, or
                // with its end, which it reports.
                let (r, j) = (r as usize, j as usize);
                s.task[r].as_mut()?.jobs[j].in_call = false;
                let ended = s.job[j].phase.over();
                if ended {
                    self.report(&mut s, r, j);
                }
                let event = kernel::Event::PutBack {
                    job: job_id(j),
                    running: !ended,
                    offset: 0,
                };
                self.replica(&mut s, r, event, false, None);
            }
            Act::Switch(r) => {
                spend(&mut s);
                let r = r as usize;
                let old = s.task[r].as_ref()?.transcript;
                s.transcript += 1;
                // The switch kills and wipes the old transcript's jobs.
                for j in 0..self.jobs {
                    let sj = s.job[j];
                    if sj.transcript == old && sj.phase != Phase::Unstarted && !sj.removed {
                        if !sj.phase.over() {
                            s.job[j].phase = Phase::Ended;
                        }
                        s.job[j].removed = true;
                        s.job[j].wiped = true;
                        self.notifier(&mut s, j, nk::Event::Removed);
                    }
                }
                let tr = s.transcript;
                let t = s.task[r].as_mut()?;
                t.transcript = tr;
                for tj in t.jobs.iter_mut() {
                    tj.tracked = false;
                    tj.link = Link::None;
                    tj.keep = None;
                }
                if self.mutation != Mutation::HideSwitch {
                    self.replica(&mut s, r, kernel::Event::TranscriptSwitched, false, None);
                }
            }
        }
        self.observe(&mut s);
        s.net.sort();
        Some(s)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        let mut props = vec![
            Property::<Self>::always(
                "S1: each end and shell exit is reported at most once per session task",
                |_, s| s.bad & S1_TWICE == 0,
            ),
            Property::<Self>::always(
                "S2: a job is unheld only after keep with notify succeeded and a proof arrived in its window",
                |_, s| s.bad & S2_EARLY == 0,
            ),
            Property::<Self>::always("S3: a callback never opens a session", |_, s| {
                s.bad & S3_OPENED == 0
            }),
            Property::<Self>::always("S4: a forged callback changes no state", |_, s| {
                s.bad & S4_FORGED == 0
            }),
            Property::<Self>::always("S5: a fenced owner never commits a report", |_, s| {
                s.bad & S5_FENCED == 0
            }),
            Property::<Self>::always(
                "S6: nothing on a timer alone contacts the sandbox",
                |_, s| s.bad & S6_TIMER == 0,
            ),
            Property::<Self>::always("S7: at most one attempt per job in flight", |_, s| {
                s.bad & S7_TWO == 0
            }),
            Property::<Self>::always(
                "S8: a report lands only in its job's own transcript",
                |_, s| s.bad & S8_TRANSCRIPT == 0,
            ),
            Property::<Self>::always(
                "L1: once settled, every end the open session tracks is reported",
                |m, s| {
                    let mut acts = Vec::new();
                    m.actions(s, &mut acts);
                    if !acts.is_empty() {
                        return true;
                    }
                    let Some(t) = m.current_task(s).and_then(|r| s.task[r].as_ref()) else {
                        return true;
                    };
                    (0..m.jobs).all(|j| {
                        let sj = s.job[j];
                        !(sj.phase.over() && !sj.wiped && t.jobs[j].tracked)
                            || t.jobs[j].reports > 0
                    })
                },
            ),
            // ---- non-vacuity witnesses: each must be reachable in a config that allows it ----
            Property::<Self>::sometimes("W: a job is unheld", |_, s| s.saw & W_UNHELD != 0),
            Property::<Self>::sometimes("W: an unheld job's end is reported", |_, s| {
                s.saw & W_UNHELD_END_REPORTED != 0
            }),
            Property::<Self>::sometimes("W: the edge walks the ring", |_, s| s.saw & W_WALK != 0),
        ];
        if self.mutation != Mutation::None {
            props.retain(|p| !p.name.starts_with("W:"));
            return props;
        }
        if self.chaos > 0 && self.adversary & NET != 0 {
            props.push(Property::<Self>::sometimes(
                "W: a forged callback is refused",
                |_, s| s.saw & W_FORGED_REFUSED != 0,
            ));
        }
        if self.chaos > 0 && self.adversary & OWN != 0 {
            props.push(Property::<Self>::sometimes(
                "W: a stale owner takes a callback",
                |_, s| s.saw & W_STALE_TOOK != 0,
            ));
            props.push(Property::<Self>::sometimes(
                "W: a new owner adopts and keeps again",
                |_, s| s.saw & W_TAKEOVER_ADOPTS != 0,
            ));
        }
        if self.calls > 0 {
            props.push(Property::<Self>::sometimes(
                "W: a bash_job call takes an unheld job",
                |_, s| s.saw & W_CALL_ON_UNHELD != 0,
            ));
        }
        props
    }
}

fn run(model: Notify, label: &str) -> impl Checker<Notify> + use<> {
    let checker = model.checker().spawn_bfs().join();
    println!(
        "{label}: {} states, {} unique",
        checker.state_count(),
        checker.unique_state_count()
    );
    checker
}

/// The faithful model: every safety property holds over every interleaving, every settled run
/// has reported what it owes, and every witness for the config is reachable.
fn check(model: Notify, label: &str) {
    run(model, label).assert_properties();
}

/// Whether the exhaustive configs run (`NOTIFY_MODEL_FULL=1`); CI runs the rest.
fn full() -> bool {
    std::env::var_os("NOTIFY_MODEL_FULL").is_some()
}

const SAFETY: [&str; 8] = [
    "S1: each end and shell exit is reported at most once per session task",
    "S2: a job is unheld only after keep with notify succeeded and a proof arrived in its window",
    "S3: a callback never opens a session",
    "S4: a forged callback changes no state",
    "S5: a fenced owner never commits a report",
    "S6: nothing on a timer alone contacts the sandbox",
    "S7: at most one attempt per job in flight",
    "S8: a report lands only in its job's own transcript",
];

/// The actions of `path` up to the first state that breaks `property` (the latches stay set, so
/// a discovery's path may run on past it).
fn trace_to_violation(model: &Notify, property: &str, path: stateright::Path<St, Act>) -> String {
    let props = model.properties();
    let Some(p) = props.iter().find(|p| p.name == property) else {
        return String::new();
    };
    let mut trace = Vec::new();
    for (state, action) in path.into_vec() {
        if !(p.condition)(model, &state) {
            break;
        }
        if let Some(a) = action {
            trace.push(format!("{a:?}"));
        }
    }
    trace.join(" ")
}

/// A mutated model must break `property`, and no safety property but it and those in `also`
/// (what the same missing guard necessarily breaks too): it fails for the right reason. Prints
/// the counterexample.
fn mutant(model: Notify, property: &'static str, also: &[&str]) {
    let checker = run(model, &format!("mutation {:?}", model.mutation));
    let Some(path) = checker.discovery(property) else {
        panic!(
            "{:?} must break {property:?}, and the checker found no counterexample",
            model.mutation
        );
    };
    println!(
        "  {property}:\n    {}",
        trace_to_violation(&model, property, path)
    );
    for other in SAFETY
        .iter()
        .filter(|p| **p != property && !also.contains(p))
    {
        assert!(
            checker.discovery(other).is_none(),
            "{:?} should break only {property:?}, but also broke {other:?}",
            model.mutation
        );
    }
}

// =============================================================================================
// FAITHFUL: both kernels fed honestly.
// =============================================================================================

/// One job, any one adversary move of every class: a lost, late, replayed or forged callback,
/// an edge timeout, a crash, a close, a takeover, a ring change, a failed or refused keep, the
/// sandbox's death, a transcript switch, a slow reconcile.
#[test]
fn notify_one_job_any_single_fault() {
    check(
        Notify {
            calls: 0,
            ..Notify::new(1, 1)
        },
        "one job, any single fault",
    );
}

/// One job and a `bash_job` call about it, which takes an unheld job and puts it back.
#[test]
fn notify_one_job_with_a_bash_job_call() {
    check(Notify::new(1, 0), "one job, a bash_job call");
    check(
        Notify {
            adversary: NET,
            ..Notify::new(1, 1)
        },
        "one job, a bash_job call, one network fault",
    );
}

/// Any two network faults: the callback lost, late and duplicated, replayed or forged, timed out.
#[test]
fn notify_one_job_two_network_faults() {
    check(
        Notify {
            adversary: NET,
            calls: 0,
            ..Notify::new(1, 2)
        },
        "one job, two network faults",
    );
}

/// Any two of: a failed or refused keep, the sandbox's death, a transcript switch, a slow
/// reconcile.
#[test]
fn notify_one_job_two_sandbox_faults() {
    check(
        Notify {
            adversary: SBX | SLOW,
            calls: 0,
            ..Notify::new(1, 2)
        },
        "one job, two sandbox faults",
    );
}

/// Exhaustive (`NOTIFY_MODEL_FULL=1`): any two ownership moves.
#[test]
fn notify_full_two_ownership_faults() {
    if full() {
        check(
            Notify {
                adversary: OWN,
                calls: 0,
                ..Notify::new(1, 2)
            },
            "one job, two ownership faults",
        );
    }
}

/// Exhaustive (`NOTIFY_MODEL_FULL=1`): any two moves of any class.
#[test]
fn notify_full_any_two_faults() {
    if full() {
        check(
            Notify {
                calls: 0,
                ..Notify::new(1, 2)
            },
            "one job, any two faults",
        );
    }
}

/// Exhaustive (`NOTIFY_MODEL_FULL=1`): two jobs, no fault.
#[test]
fn notify_full_two_jobs() {
    if full() {
        check(
            Notify {
                calls: 0,
                ..Notify::new(2, 0)
            },
            "two jobs",
        );
    }
}

// =============================================================================================
// MUTATIONS: one guard off each, by lying to a kernel through its inputs or switching off the
// piece of environment the property names; the checker must find that property's
// counterexample, and no other safety property's.
// =============================================================================================

/// Without today's per-job record, a live reconcile's follow and the stream it closed, or a
/// `bash_job` call, report one end twice.
#[test]
fn mutation_no_once_record_reports_twice() {
    mutant(
        Notify::new(1, 0).mutated(Mutation::NoOnceRecord),
        SAFETY[0],
        &[],
    );
}

/// Told a failed `keep` succeeded, the kernel releases a job whose endpoint never stored notify.
#[test]
fn mutation_keep_always_ok_releases_early() {
    mutant(
        Notify {
            adversary: SBX,
            calls: 0,
            ..Notify::new(1, 1)
        }
        .mutated(Mutation::KeepAlwaysOk),
        SAFETY[1],
        &[],
    );
}

/// Told every callback came at time 0, the kernel takes a proof that came after its window.
#[test]
fn mutation_window_never_closes_releases_late() {
    mutant(
        Notify {
            calls: 0,
            ..Notify::new(1, 0)
        }
        .mutated(Mutation::WindowNeverCloses),
        SAFETY[1],
        &[],
    );
}

/// A replica that opens the session for a callback breaks S3.
#[test]
fn mutation_callback_opens_a_session() {
    mutant(
        Notify {
            calls: 0,
            ..Notify::new(1, 0)
        }
        .mutated(Mutation::CallbackOpens),
        SAFETY[2],
        &[],
    );
}

/// Without the token check, a forged callback reaches the session and changes its state (and,
/// reaching a proving job, stands in for its proof).
#[test]
fn mutation_no_token_check_lets_a_forgery_in() {
    mutant(
        Notify {
            adversary: NET,
            calls: 0,
            ..Notify::new(1, 1)
        }
        .mutated(Mutation::NoTokenCheck),
        SAFETY[3],
        // A forgery that reaches a proving job is taken as its proof: an unverified one.
        &[SAFETY[1]],
    );
}

/// Without the epoch fence, a superseded owner commits the end it learned.
#[test]
fn mutation_no_fence_lets_a_stale_owner_commit() {
    mutant(
        Notify {
            adversary: OWN,
            calls: 0,
            ..Notify::new(1, 1)
        }
        .mutated(Mutation::NoFence),
        SAFETY[4],
        &[],
    );
}

/// A poll timer that feeds the kernel a use makes time alone contact the sandbox.
#[test]
fn mutation_poll_timer_contacts_the_sandbox() {
    mutant(
        Notify {
            adversary: 0,
            calls: 0,
            ..Notify::new(1, 1)
        }
        .mutated(Mutation::PollTimer),
        SAFETY[5],
        &[],
    );
}

/// Told its attempt ended while it is still in flight, the notifier begins a second.
#[test]
fn mutation_early_answer_doubles_an_attempt() {
    mutant(
        Notify {
            adversary: NET,
            calls: 0,
            ..Notify::new(1, 1)
        }
        .mutated(Mutation::EarlyAnswer),
        SAFETY[6],
        &[],
    );
}

/// Never told of a transcript switch, the kernel hands a reconcile's findings to the new
/// transcript, which reports the old one's job.
#[test]
fn mutation_hidden_switch_reports_into_the_wrong_transcript() {
    mutant(
        Notify {
            adversary: SBX,
            calls: 0,
            ..Notify::new(1, 1)
        }
        .mutated(Mutation::HideSwitch),
        SAFETY[7],
        &[],
    );
}

/// Dropping a use that arrives while a reconcile runs (the reconcile may have listed before the
/// end) leaves that end unreported after the one real use: strict L1 breaks, and nothing else.
#[test]
fn mutation_drop_use_during_reconcile_misses_an_end() {
    const L1: &str = "L1: once settled, every end the open session tracks is reported";
    let model = Notify {
        adversary: SBX,
        calls: 0,
        ..Notify::new(1, 1)
    }
    .mutated(Mutation::DropUseDuringReconcile);
    mutant(model, L1, &[]);
    // For the right reason: the counterexample's use came while a reconcile ran.
    let checker = run(model, "the same, its trace");
    let path = checker
        .discovery(L1)
        .map(|p| p.into_vec())
        .unwrap_or_default();
    let during = path.iter().any(|(state, action)| {
        matches!(action, Some(Act::Use(r, _))
            if state.task[*r as usize].as_ref().is_some_and(|t| t.rk.reconciling()))
    });
    assert!(
        during,
        "the counterexample's use must come during a reconcile"
    );
}

/// A real use less than 30 s after a reconcile ended (none running) is enough: it asks for a
/// reconcile at the 30 s mark, which finds an end the last one listed too early to see (here the
/// sandbox's death). Strict L1 holds, and the early use is reached.
#[test]
fn a_use_soon_after_a_reconcile_is_enough() {
    const L1: &str = "L1: once settled, every end the open session tracks is reported";
    let model = Notify {
        adversary: SBX,
        calls: 0,
        ..Notify::new(1, 1)
    };
    assert!(model.use_in_quiet_window);
    let checker = run(model, "one job, a use inside N29's 30 s");
    checker.assert_properties();
    assert!(checker.discovery(L1).is_none());
    // Non-vacuous: the mutation that drops exactly that use breaks L1 in the same config.
    let dropped = run(
        model.mutated(Mutation::DropEarlyUse),
        "the same, the early use dropped",
    );
    assert!(dropped.discovery(L1).is_some(), "the early use is reached");
}

/// Dropping a use that arrives less than 30 s after a reconcile ended (that reconcile may have
/// listed before the end) leaves that end unreported after the one real use: strict L1 breaks,
/// and nothing else.
#[test]
fn mutation_drop_early_use_misses_an_end() {
    const L1: &str = "L1: once settled, every end the open session tracks is reported";
    let model = Notify {
        adversary: SBX,
        calls: 0,
        ..Notify::new(1, 1)
    }
    .mutated(Mutation::DropEarlyUse);
    mutant(model, L1, &[]);
    // For the right reason: the counterexample's use came inside the 30 s, none running.
    let checker = run(model, "the same, its trace");
    let path = checker
        .discovery(L1)
        .map(|p| p.into_vec())
        .unwrap_or_default();
    let soon = path.iter().any(|(state, action)| {
        matches!(action, Some(Act::Use(r, false)) if state.task[*r as usize].as_ref().is_some_and(|t| {
            !t.rk.reconciling()
                && t.last_reconcile.is_some_and(|end| t.now < end + SPEC_USE_QUIET_MS)
        }))
    });
    assert!(soon, "the counterexample's use must come inside the 30 s");
}
