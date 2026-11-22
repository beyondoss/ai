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
//! transcript; and, read off every step of the notifier's kernel as it is taken (so a later real
//! use that rescues a lost notice cannot hide a bad step): (S9) a count makes at most 12 attempts;
//! (S10) a wake before its time begins nothing; (S11) the wait after failed attempt n lies in
//! [d(n)/2, d(n)]; (S12) an answer that settles is never retried; (S13) an answer settles only its
//! own attempt; (S14) a `keep` with notify, and a change under one, each owe one callback more.
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
//! its tick, as a late timer allows. The notifier has its own clock, which moves only to the time the
//! kernel asked to be woken at ([`Act::NotifierAdvance`]); a wake can also arrive early
//! ([`Act::NotifierEarly`]). An answer's draw is the longest wait, or in [`Notify::drawn`] the
//! shortest, the midpoint or the longest. Every other
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
//! - **The capped configs run a count to [`RETRY_CAP`] (3), not 12.** The kernel resets a count on
//!   every answer that settles it, so a success at attempt 5 leaves the state a success at
//!   attempt 2 leaves; a reopen re-keeps the job, which abandons the count and begins a new one;
//!   and a count that never succeeds is the spec's accepted loss either way. Attempts 4 to 12 add
//!   only later instances of what attempts 2 and 3 already cover. [`Notify::deep`] does run to 12,
//!   with only the receiver failing, so the limit itself (S9) is checked.
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
use execd_jobs as jk;
use stateright::{Checker, Model, Property};

/// Replicas in the ring.
const R: usize = 2;

/// The capped configs let a count of attempts run to this many; past it, the supervisor's next
/// wait outlasts the run. See "Bounds" for why that loses no behaviour the capped configs' properties
/// can see. The retry-limit config ([`Notify::deep`]) runs to the spec's real limit instead.
const RETRY_CAP: u32 = 3;

/// The spec's attempt limit (N18): a count makes at most this many attempts. The S9 latch judges
/// by this, not by the kernel's constant.
const SPEC_MAX_ATTEMPTS: u32 = 12;

/// Whether the environment feeds a removed job's notifier the way execd ships it: `Jobs::remove`
/// resets it to its default (`jobs.rs`, `Notifier::default()`), never `Event::Removed`.
const SHIPPED_REMOVE_RESETS: bool = true;

/// The wake-up draws a config may hand the kernel: the shortest wait, the midpoint, the longest.
const DRAWS: [u32; 3] = [u32::MAX, 1 << 31, 0];

/// The spec's `d(n)` in ms (N18): `min(2^(n-1), 60)` s, 1 s for `n` 0 or 1.
fn spec_backoff_ms(n: u32) -> u64 {
    1000 * (1u64 << n.saturating_sub(1).min(6)).min(60)
}

/// The spec's table of answers (N17): does the answer ask for a retry? Everything else settles.
fn spec_retries(outcome: nk::Outcome) -> bool {
    match outcome {
        nk::Outcome::Failed => true,
        nk::Outcome::Status(s) => matches!(s, 407 | 408 | 425 | 429 | 500..=599),
    }
}

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

/// One job of the sandbox, as execd holds it: the real kernel's state ([`jk::State`], changed only
/// by [`jk::step`]), and what only the shell and the environment know.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Cell {
    st: jk::State,
    /// The id it was started under (a key is `("s", id)`; a cell's index is `id + jobs * generation`).
    id: u8,
    /// The transcript the job was started on.
    transcript: u8,
    /// The table's entry for its key is this cell (what [`jk::Table`] left in it).
    in_table: bool,
    /// Wiped by a transcript switch: no report is owed for it anywhere.
    wiped: bool,
    /// Its processes were stopped (`kill`, `wipe`): the group is empty, whatever `st` has seen.
    killed: bool,
    /// The sandbox died under it.
    lost: bool,
    /// Ghost: ends of this cell committed to a transcript, by any task (X3).
    commits: u8,
    /// Ghost: a `keep` of the replica's took effect on it (X4).
    kept_ok: bool,
}

impl Cell {
    /// What `list` would report.
    fn phase(&self) -> Phase {
        if self.lost {
            return Phase::Lost;
        }
        match self.st.phase {
            jk::Phase::Spawning | jk::Phase::Running { .. } => Phase::Running,
            jk::Phase::Exited { .. } => Phase::Exited,
            jk::Phase::Ended { .. } => Phase::Ended,
        }
    }

    /// The group is observably empty to the shell.
    fn empty(&self) -> bool {
        self.phase().over() || self.killed
    }
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
    /// The id the callback names (its body is `(scope, id)`).
    job: u8,
    /// The cell whose notifier made the attempt (an attempt's answer goes back to that one).
    cell: u8,
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
    /// The cell it tracks: what its start made, or its open reconcile listed under the id.
    cell: u8,
    /// A `keep`'s table lookup: `None` before it, `Some(None)` a `404`, else the cell it found.
    keep_cell: Option<Option<u8>>,
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
    /// The sandbox's jobs, by cell (see [`Cell`]): `None` is a cell never started.
    cell: Vec<Option<Cell>>,
    /// The notifier of each cell (a job's `Notifier`, which lives in its cell).
    nk: Vec<nk::Job>,
    /// The notifier's clock, in ms. It moves only to a time the notifier asked for.
    nnow: u64,
    net: Vec<Msg>,
    ring: u8,
    up: [bool; R],
    task: [Option<Task>; R],
    epoch: u8,
    transcript: u8,
    chaos: u8,
    calls: u8,
    bad: u32,
    saw: u16,
}

// Safety latches.
const S1_TWICE: u32 = 1 << 0;
const S2_EARLY: u32 = 1 << 1;
const S3_OPENED: u32 = 1 << 2;
const S4_FORGED: u32 = 1 << 3;
const S5_FENCED: u32 = 1 << 4;
const S6_TIMER: u32 = 1 << 5;
const S7_TWO: u32 = 1 << 6;
const S8_TRANSCRIPT: u32 = 1 << 7;
const S9_LIMIT: u32 = 1 << 8;
const S10_EARLY_WAKE: u32 = 1 << 9;
const S11_WAIT: u32 = 1 << 10;
const S12_SETTLED_RETRIED: u32 = 1 << 11;
const S13_FOREIGN_SETTLE: u32 = 1 << 12;
const S14_NOT_OWED: u32 = 1 << 13;
// The seam: the job table and the notifier.
const X1_ATTRIBUTED: u32 = 1 << 14;
const X2_REMOVED_NOTIFIED: u32 = 1 << 15;
const X3_END_TWICE: u32 = 1 << 16;

// Witnesses.
const W_UNHELD: u16 = 1 << 0;
const W_UNHELD_END_REPORTED: u16 = 1 << 1;
const W_STALE_TOOK: u16 = 1 << 2;
const W_WALK: u16 = 1 << 3;
const W_FORGED_REFUSED: u16 = 1 << 4;
const W_TAKEOVER_ADOPTS: u16 = 1 << 5;
const W_CALL_ON_UNHELD: u16 = 1 << 6;
const W_REUSED_WITH_MSG: u16 = 1 << 7;

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
    /// X2: a removed job's notifier is reset to its default, as shipped before the fix, so a
    /// `keep` that reached the job after its removal begins a callback for it.
    RemoveResetsNotifier,
    /// X1: a notify attempt outlives its job's removal and its answer goes to whichever job holds
    /// the key now, not the job whose notifier made the attempt (the notifier's own J9).
    AnswerByKey,
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
/// The receiver: it answers an attempt with an error status of its choosing (a `503`, a `404`).
const RCV: u8 = 1 << 4;
const ALL: u8 = NET | OWN | SBX | SLOW;

#[derive(Clone, Copy)]
struct Notify {
    jobs: usize,
    chaos: u8,
    /// The adversary classes the budget may buy.
    adversary: u8,
    /// `bash_job` calls the session may make.
    calls: u8,
    /// A count of attempts may begin only while it has made fewer than this many ([`RETRY_CAP`]
    /// for the capped configs, past the spec's limit for [`Notify::deep`]).
    retry_cap: u32,
    /// How many of [`DRAWS`] an answer may carry: 1 (the longest wait) or 3.
    draws: usize,
    /// Another client may start an id again once its job was removed (the table admits it: J9's
    /// scenario). Each id then has two cells.
    reuse: bool,
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
            retry_cap: RETRY_CAP,
            draws: 1,
            reuse: false,
            use_in_quiet_window: true,
            mutation: Mutation::None,
        }
    }

    /// The notifier's retry schedule, to the spec's real limit: only the receiver misbehaves
    /// (`failures` times, a `503` or a `404`), one job, no `bash_job` call.
    fn deep(failures: u8) -> Self {
        Notify {
            adversary: RCV,
            calls: 0,
            retry_cap: SPEC_MAX_ATTEMPTS + 1,
            ..Notify::new(1, failures)
        }
    }

    /// The notifier's waits, over the shortest, the midpoint and the longest draw.
    fn drawn(failures: u8) -> Self {
        Notify {
            adversary: RCV,
            calls: 0,
            draws: DRAWS.len(),
            ..Notify::new(1, failures)
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
    /// The notifier's clock reaches the time it asked to be woken at.
    NotifierAdvance(u8),
    /// A wake arrives before its time (a timer fired early, or fired twice).
    NotifierEarly(u8),
    NotifierTimer(u8),
    Route(u8),
    /// An attempt's answer, with the wake-up draw (an index into [`DRAWS`]) the supervisor takes.
    Answer(u8, u8),
    /// The receiver answers an attempt itself with this status ([`RCV`]).
    Reply(u8, u16, u8),
    /// A `keep` request looks its job up (`Jobs::get`), before it takes the job's lock.
    KeepResolve(u8, u8),
    /// Another client starts id `j` again, once its job left the table.
    Reuse(u8),
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

    // ---- the job table: every decision is the real `execd_jobs` kernel's -------------------

    fn key(j: usize) -> jk::Key {
        ("s".to_string(), job_id(j))
    }

    /// Cells in the model: each id has a second one when another client may start it again.
    fn cells(&self) -> usize {
        self.jobs * if self.reuse { 2 } else { 1 }
    }

    fn id_of(&self, c: usize) -> usize {
        c % self.jobs
    }

    /// The table, as the cells' `in_table` left it (the storage is not state: its entries are).
    fn table(&self, s: &St) -> jk::Table<u8> {
        let mut t = jk::Table::default();
        for (c, cell) in s.cell.iter().enumerate() {
            if let Some(cell) = cell
                && cell.in_table
            {
                t.insert(Self::key(cell.id as usize), c as u8);
            }
        }
        t
    }

    /// The cell the table holds under id `j`: a read of the model's own state, for enabling an
    /// action. A decision about it goes through [`Notify::table`].
    fn cur(&self, s: &St, j: usize) -> Option<usize> {
        s.cell
            .iter()
            .position(|c| c.as_ref().is_some_and(|c| c.id as usize == j && c.in_table))
    }

    fn removed(s: &St, c: usize) -> bool {
        s.cell[c].as_ref().is_some_and(|x| x.st.removed)
    }

    /// The job a task's stream goes on with: execd's `bound`, the table's entry under the id kept
    /// only if it is the very job the stream attached to (J9).
    fn followed(&self, s: &St, tj: &TaskJob, j: usize) -> Option<usize> {
        let found = self.table(s).get(&Self::key(j));
        let same = found == Some(tj.cell);
        jk::bound(found, same).map(usize::from)
    }

    /// The requests a notifier's job had on the wire are no longer waited for: its attempt task
    /// is dropped (an abandoned request may still arrive, late).
    fn abandon_net(s: &mut St, c: usize) {
        for m in s.net.iter_mut() {
            if m.cell as usize == c && m.origin == Origin::Attempt {
                m.origin = Origin::Orphan;
            }
        }
        s.net
            .retain(|m| !(m.origin == Origin::Orphan && matches!(m.at, At::Back(_))));
    }

    /// `Jobs::remove` for one cell the table let go: the kernel's `Taken`, then the shell drops
    /// the job's notifier. Shipped, it resets the notifier to its default, and the kernel's
    /// `removed` (the one a late `keep` meets) stays false.
    fn taken(&self, s: &mut St, c: usize) {
        let Some(cell) = s.cell[c].as_mut() else {
            return;
        };
        if jk::step(&mut cell.st, jk::Ev::Taken) != jk::Act::Release {
            return;
        }
        if self.mutation != Mutation::AnswerByKey {
            Self::abandon_net(s, c);
        }
        s.nk[c] = if self.mutation == Mutation::RemoveResetsNotifier || SHIPPED_REMOVE_RESETS {
            nk::Job::default()
        } else {
            nk::step(s.nk[c], nk::Event::Removed, 0).0
        };
    }

    /// execd's `remove` and `wipe` on the sandbox: the kernel says which named jobs may leave
    /// (`Ev::Remove`), the table lets them go (`take`), each is told `Taken`. `expect` is the job
    /// the caller means to remove: another one leaving under its id is X1.
    fn remove(&self, s: &mut St, pick: Option<usize>, ended_counts: bool, expect: Option<usize>) {
        let mut table = self.table(s);
        let ids: Vec<String> = pick.map(job_id).into_iter().collect();
        let named = table.in_scope(
            "s",
            if pick.is_some() {
                jk::Pick::Ids(&ids)
            } else {
                jk::Pick::All
            },
        );
        let mut keys = Vec::new();
        for (k, c) in named {
            let Some(cell) = s.cell[c as usize].as_mut() else {
                continue;
            };
            let ev = jk::Ev::Remove {
                empty: cell.empty(),
                ended_counts,
            };
            if jk::step(&mut cell.st, ev) == jk::Act::Take {
                keys.push(k);
            }
        }
        let gone = table.take(&keys);
        for c in gone {
            let c = c as usize;
            if expect.is_some_and(|e| e != c) {
                s.bad |= X1_ATTRIBUTED;
            }
            self.taken(s, c);
        }
        for (c, cell) in s.cell.iter_mut().enumerate() {
            if let Some(cell) = cell {
                cell.in_table = table.get(&Self::key(cell.id as usize)) == Some(c as u8);
            }
        }
    }

    // ---- the notifier: the only call into `notify_kernel` ----------------------------------

    /// S9 to S14: what the spec (N16 to N18) says of one step, read off the step's inputs and
    /// outputs, never recomputed from the kernel's rules except through the spec's own table
    /// ([`spec_retries`], [`spec_backoff_ms`]). Each attempt is judged when it begins, ends or
    /// waits, so a later real use that rescues a lost notice cannot hide a bad step.
    fn judge_notifier(
        before: &nk::Job,
        event: nk::Event,
        now: u64,
        next: &nk::Job,
        actions: &[nk::Action],
    ) -> u32 {
        let mut bad = 0;
        if before.removed {
            return bad;
        }
        for a in actions {
            // S9: no count makes more than the spec's limit of attempts.
            if let nk::Action::Begin { attempt } = a
                && *attempt > SPEC_MAX_ATTEMPTS
            {
                bad |= S9_LIMIT;
            }
        }
        let owes_one_more = match event {
            nk::Event::Keep { notify } => notify,
            nk::Event::Change => before.notify,
            _ => false,
        };
        // S14: a `keep` with notify, and a change under one, each owe one callback more.
        if owes_one_more && next.owed != before.owed + 1 {
            bad |= S14_NOT_OWED;
        }
        match event {
            nk::Event::Wake => {
                // S10: a wake before the time asked for begins nothing.
                let early = before.wake_at.is_none_or(|at| now < at);
                if early
                    && actions
                        .iter()
                        .any(|a| matches!(a, nk::Action::Begin { .. }))
                {
                    bad |= S10_EARLY_WAKE;
                }
            }
            nk::Event::AttemptEnded { outcome, .. } => {
                let Some(carried) = before.in_flight else {
                    return bad;
                };
                // S13: an answer settles what its attempt carried, nothing owed after.
                if next.settled > before.settled.max(carried) {
                    bad |= S13_FOREIGN_SETTLE;
                }
                // S12: an answer that settles is not sent again (what was owed after it is).
                if !spec_retries(outcome) && carried == before.owed && !actions.is_empty() {
                    bad |= S12_SETTLED_RETRIED;
                }
                // S11: the wait after failed attempt n lies in [d(n)/2, d(n)].
                for a in actions {
                    if let nk::Action::WakeAt(at) = a {
                        let d = spec_backoff_ms(before.attempts);
                        if *at < now + d / 2 || *at > now + d {
                            bad |= S11_WAIT;
                        }
                    }
                }
            }
            _ => {}
        }
        bad
    }

    /// Feed cell `j`'s notifier. execd feeds a timer's wake and an attempt's answer only to a job
    /// that is not removed (`notify.rs`); a `keep` and a change are fed as they come.
    fn notifier(&self, s: &mut St, j: usize, event: nk::Event) {
        if !s.sb_alive {
            return; // the supervisors died with the sandbox
        }
        let removed = Self::removed(s, j);
        if removed && matches!(event, nk::Event::Wake | nk::Event::AttemptEnded { .. }) {
            return;
        }
        let now = s.nnow;
        let before = s.nk[j];
        let (next, actions) = nk::step(before, event, now);
        s.bad |= Self::judge_notifier(&before, event, now, &next, &actions);
        s.nk[j] = next;
        for a in actions {
            match a {
                nk::Action::Begin { .. } => {
                    // X2: a removed job is never notified again.
                    if removed {
                        s.bad |= X2_REMOVED_NOTIFIED;
                    }
                    if s.net
                        .iter()
                        .any(|m| m.cell as usize == j && m.origin == Origin::Attempt)
                    {
                        s.bad |= S7_TWO;
                    }
                    s.net.push(Msg {
                        job: self.id_of(j) as u8,
                        cell: j as u8,
                        origin: Origin::Attempt,
                        at: At::Edge,
                    });
                }
                nk::Action::Abandon => {
                    // The connection is dropped; the request may already be on its way.
                    Self::abandon_net(s, j);
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
    ///
    /// `c` is the job whose end the task read: the cell the stream followed, or listed under the
    /// id. The report is the task's job `j`'s, so a different job's end is X1.
    fn report(&self, s: &mut St, r: usize, j: usize, c: usize) {
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
        if t.jobs[j].cell as usize != c {
            s.bad |= X1_ATTRIBUTED;
        }
        if t.jobs[j].was_unheld {
            s.saw |= W_UNHELD_END_REPORTED;
        }
        let tr = t.transcript;
        let own = t.jobs[j].cell as usize;
        if let Some(cell) = s.cell[c].as_mut() {
            if cell.transcript != tr {
                s.bad |= S8_TRANSCRIPT;
            }
            // X3: an end is committed once, whichever path carried it.
            cell.commits += 1;
            if cell.commits > 1 {
                s.bad |= X3_END_TWICE;
            }
        }
        // Then the record leaves the sandbox: `remove` by id, the job's end counting.
        if s.sb_alive {
            self.remove(s, Some(j), true, Some(own));
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
                    cell: 0,
                    keep_cell: None,
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

    fn mark_use_due(&self, s: &mut St, c: usize) {
        let j = self.id_of(c);
        for t in s.task.iter_mut().flatten() {
            if t.jobs[j].tracked && t.jobs[j].cell as usize == c {
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
        // X2: a removed job owes nothing: nothing in flight, nothing waiting, no callback owed.
        for c in 0..self.cells() {
            if Self::removed(s, c) {
                let k = s.nk[c];
                if k.in_flight.is_some()
                    || k.wake_at.is_some()
                    || (!k.removed && k.notify && k.owed > k.settled)
                {
                    s.bad |= X2_REMOVED_NOTIFIED;
                }
            }
        }
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

    /// `list`: the table's jobs of the scope, as the kernel's `in_scope` names them.
    fn listing(&self, s: &St) -> Vec<(usize, kernel::Listed)> {
        let table = self.table(s);
        table
            .in_scope("s", jk::Pick::All)
            .into_iter()
            .filter_map(|((_, id), c)| {
                let cell = s.cell[c as usize].as_ref()?;
                let l = match cell.phase() {
                    Phase::Running => kernel::Listed::Running,
                    Phase::Exited => kernel::Listed::Exited { code: 0 },
                    Phase::Ended => kernel::Listed::Ended { code: 0, at: 0 },
                    Phase::Lost => kernel::Listed::Lost { at: 0 },
                };
                Some((job_index(&id), l))
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
            cell: vec![None; self.cells()],
            nk: vec![nk::Job::default(); self.cells()],
            nnow: 0,
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
        // The sandbox's own moves, per cell: the shell exits and the group empties (a removed
        // job's processes still end, and the supervisor still observes it), and the notifier's
        // timer.
        for c in 0..self.cells() {
            let cb = c as u8;
            let Some(cell) = &s.cell[c] else { continue };
            if s.sb_alive {
                if cell.phase() == Phase::Running {
                    acts.push(Act::ShellExit(cb));
                }
                if matches!(cell.phase(), Phase::Running | Phase::Exited) {
                    acts.push(Act::JobEnd(cb));
                }
            }
            if s.sb_alive
                && s.nk[c].in_flight.is_none()
                && s.nk[c].attempts < self.retry_cap
                && let Some(at) = s.nk[c].wake_at
            {
                if s.nnow >= at {
                    acts.push(Act::NotifierTimer(cb));
                } else {
                    acts.push(Act::NotifierAdvance(cb));
                    acts.push(Act::NotifierEarly(cb));
                }
            }
        }
        for j in 0..self.jobs {
            let jb = j as u8;
            if net && s.sb_alive && self.cur(s, j).is_some() {
                acts.push(Act::Replay(jb));
                acts.push(Act::Forge(jb));
            }
            // Another client starts the id again, once its job left the table (the table's `admit`
            // is the decision; this is only whether the move is offered).
            if self.reuse
                && s.sb_alive
                && self.cur(s, j).is_none()
                && s.cell[j].is_some()
                && s.cell[j + self.jobs].is_none()
            {
                acts.push(Act::Reuse(jb));
            }
        }
        if sbx && s.sb_alive && s.cell.iter().any(|c| c.is_some()) {
            acts.push(Act::SandboxDies);
        }
        for (i, m) in s.net.iter().enumerate() {
            if i > 0 && s.net[i - 1] == *m {
                continue; // identical messages: one stands for both
            }
            let ib = i as u8;
            match m.at {
                At::Edge => acts.push(Act::Route(ib)),
                At::Back(_) => {
                    let draws = if m.origin == Origin::Attempt {
                        self.draws
                    } else {
                        1
                    };
                    for d in 0..draws {
                        acts.push(Act::Answer(ib, d as u8));
                    }
                }
                At::Waiting { .. } => {}
            }
            if chaos && self.adversary & RCV != 0 && m.origin == Origin::Attempt && m.at == At::Edge
            {
                for code in [404, 503] {
                    for d in 0..self.draws {
                        acts.push(Act::Reply(ib, code, d as u8));
                    }
                }
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
                if s.sb_alive
                    && s.cell[j].is_none()
                    && t.transcript == s.transcript
                    && t.epoch == s.epoch
                {
                    acts.push(Act::StartJob(rb, jb));
                }
                if let Some(notify) = tj.keep {
                    if tj.keep_cell.is_none() && s.sb_alive {
                        acts.push(Act::KeepResolve(rb, jb));
                    }
                    match tj.keep_cell {
                        // Found: the request answers once it has taken the job's lock.
                        Some(Some(_)) => {
                            acts.push(Act::KeepAnswer(rb, jb, Keep::Ok));
                            if sbx {
                                if notify {
                                    acts.push(Act::KeepAnswer(rb, jb, Keep::Refused));
                                }
                                acts.push(Act::KeepAnswer(rb, jb, Keep::FailedUnrecorded));
                                acts.push(Act::KeepAnswer(rb, jb, Keep::FailedRecorded));
                            }
                        }
                        // `404`: an answer the protocol gives, no adversary needed.
                        Some(None) => acts.push(Act::KeepAnswer(rb, jb, Keep::FailedUnrecorded)),
                        // The request never reached a live sandbox: an error, no adversary needed.
                        None if !s.sb_alive => {
                            acts.push(Act::KeepAnswer(rb, jb, Keep::FailedUnrecorded))
                        }
                        None => {
                            if sbx {
                                if notify {
                                    acts.push(Act::KeepAnswer(rb, jb, Keep::Refused));
                                }
                                acts.push(Act::KeepAnswer(rb, jb, Keep::FailedUnrecorded));
                            }
                        }
                    }
                }
                if matches!(tj.link, Link::Stream | Link::Closing)
                    && !tj.in_call
                    && let Some(c) = self.followed(s, tj, j)
                    && let Some(cell) = &s.cell[c]
                {
                    if cell.phase() == Phase::Exited && tj.exits == 0 {
                        acts.push(Act::StreamExit(rb, jb));
                    }
                    if cell.phase().over() {
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
                // A held start: the table admits the key (J1), the job is spawned. The replica
                // mints a fresh id for every job, so this is the id's first cell.
                if self.table(&s).admit(&Self::key(j)) != jk::Admit::Fresh {
                    return None;
                }
                let mut st = jk::State::new(0, true);
                jk::step(&mut st, jk::Ev::Spawned { pgid: 1 });
                s.cell[j] = Some(Cell {
                    st,
                    id: j as u8,
                    transcript: s.transcript,
                    in_table: true,
                    wiped: false,
                    killed: false,
                    lost: false,
                    commits: 0,
                    kept_ok: false,
                });
                let t = s.task[r].as_mut()?;
                t.jobs[j].tracked = true;
                t.jobs[j].cell = j as u8;
                t.jobs[j].link = Link::Stream;
                let event = kernel::Event::Kept {
                    job: job_id(j),
                    notify_available: Self::notify_available(t),
                    adopted: false,
                };
                self.replica(&mut s, r, event, false, None);
            }
            Act::ShellExit(c) => {
                // `supervise`: the shell exited with its group alive; a callback is owed whenever
                // the kernel says the exit is observable (`Act::Exited`), a removed job included.
                let c = c as usize;
                let cell = s.cell[c].as_mut()?;
                let ev = jk::Ev::ShellExit {
                    code: 0,
                    signal: None,
                };
                if jk::step(&mut cell.st, ev) == jk::Act::Exited {
                    self.notifier(&mut s, c, nk::Event::Change);
                }
            }
            Act::JobEnd(c) => {
                // `supervise`: the group is empty. A callback is owed only if the kernel says so
                // (`Ended { notify }`: not for a removed job).
                let c = c as usize;
                let cell = s.cell[c].as_mut()?;
                let ev = jk::Ev::GroupEmpty {
                    code: 0,
                    signal: None,
                    now: 0,
                };
                if let jk::Act::Ended { notify } = jk::step(&mut cell.st, ev) {
                    if notify {
                        self.notifier(&mut s, c, nk::Event::Change);
                    }
                    self.mark_use_due(&mut s, c);
                }
            }
            Act::SandboxDies => {
                spend(&mut s);
                s.sb_alive = false;
                for c in 0..self.cells() {
                    let Some(cell) = s.cell[c].as_mut() else {
                        continue;
                    };
                    if matches!(cell.phase(), Phase::Running | Phase::Exited) {
                        cell.lost = true;
                        self.mark_use_due(&mut s, c);
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
            Act::NotifierAdvance(c) => {
                s.nnow = s.nnow.max(s.nk[c as usize].wake_at?);
            }
            Act::NotifierEarly(c) | Act::NotifierTimer(c) => {
                self.notifier(&mut s, c as usize, nk::Event::Wake)
            }
            Act::KeepResolve(r, j) => {
                // `keep_or_seen`: `ctx.jobs.get(&key)`, the table's lock released; the job's lock
                // comes later, and the job may have left the table in between.
                let (r, j) = (r as usize, j as usize);
                let found = self.table(&s).get(&Self::key(j));
                let t = s.task[r].as_mut()?;
                t.jobs[j].keep_cell = Some(found);
            }
            Act::Reuse(j) => {
                // Another client starts the id again. Only the table's `admit` decides it.
                let j = j as usize;
                if self.table(&s).admit(&Self::key(j)) != jk::Admit::Fresh {
                    return None;
                }
                let mut st = jk::State::new(0, false);
                jk::step(&mut st, jk::Ev::Spawned { pgid: 2 });
                let c = j + self.jobs;
                if s.net.iter().any(|m| m.job as usize == j) {
                    s.saw |= W_REUSED_WITH_MSG;
                }
                s.cell[c] = Some(Cell {
                    st,
                    id: j as u8,
                    transcript: s.transcript,
                    in_table: true,
                    wiped: false,
                    killed: false,
                    lost: false,
                    commits: 0,
                    kept_ok: false,
                });
            }
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
            Act::Reply(i, code, d) => {
                spend(&mut s);
                let m = s.net.remove(i as usize);
                let event = nk::Event::AttemptEnded {
                    outcome: nk::Outcome::Status(code),
                    draw: DRAWS[d as usize],
                };
                self.notifier(&mut s, m.cell as usize, event);
            }
            Act::Answer(i, d) => {
                let m = s.net.remove(i as usize);
                let At::Back(code) = m.at else { return None };
                if m.origin == Origin::Attempt {
                    let outcome = match code {
                        Some(c) => nk::Outcome::Status(c),
                        None => nk::Outcome::Failed,
                    };
                    let draw = DRAWS[d as usize];
                    let event = nk::Event::AttemptEnded { outcome, draw };
                    let mut c = m.cell as usize;
                    if self.mutation == Mutation::AnswerByKey
                        && let Some(now) = self.cur(&s, m.job as usize)
                    {
                        c = now;
                    }
                    // X1: an answer settles the notifier of the job whose attempt it answers.
                    if c != m.cell as usize {
                        s.bad |= X1_ATTRIBUTED;
                    }
                    self.notifier(&mut s, c, event);
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
                    self.notifier(&mut s, copy.cell as usize, event);
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
                let cell = self.cur(&s, j as usize)? as u8;
                s.net.push(Msg {
                    job: j,
                    cell,
                    origin: Origin::Orphan,
                    at: At::Edge,
                });
            }
            Act::Forge(j) => {
                spend(&mut s);
                let cell = self.cur(&s, j as usize)? as u8;
                s.net.push(Msg {
                    job: j,
                    cell,
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
                    // `list` of the scope: every job the table holds, read by id.
                    let listed = self.table(&s);
                    for j in 0..self.jobs {
                        let Some(c) = listed.get(&Self::key(j)).map(usize::from) else {
                            continue;
                        };
                        let Some(sj) = s.cell[c].as_ref() else {
                            continue;
                        };
                        if sj.transcript != transcript {
                            continue;
                        }
                        let over = sj.phase().over();
                        let Some(t) = s.task[r].as_mut() else { break };
                        t.jobs[j].tracked = true;
                        t.jobs[j].cell = c as u8;
                        if over {
                            // Read back what ended, as today's reattach does.
                            self.report(&mut s, r, j, c);
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
                // A `404` and an error from a dead sandbox are the protocol's; the rest is a move.
                let free = s.task[r].as_ref()?.jobs[j].keep_cell == Some(None) || !s.sb_alive;
                if outcome != Keep::Ok && !free {
                    spend(&mut s);
                }
                // The request found its job (`KeepResolve`) and takes its lock: the job may have left
                // the table since, but the request answers `200` all the same (`keep_or_seen`).
                let found = s.task[r].as_ref()?.jobs[j].keep_cell.flatten();
                let applied = s.sb_alive && found.is_some();
                let outcome = match outcome {
                    Keep::Ok | Keep::FailedRecorded if !applied => Keep::FailedUnrecorded,
                    o => o,
                };
                let t = s.task[r].as_mut()?;
                let now = t.now;
                let notify = t.jobs[j].keep.take()?;
                t.jobs[j].keep_cell = None;
                if notify {
                    t.jobs[j].keep_answered_at = Some(now);
                    t.jobs[j].keep_notify_ok = outcome == Keep::Ok;
                }
                if matches!(outcome, Keep::Ok | Keep::FailedRecorded)
                    && let Some(c) = found.map(usize::from)
                {
                    if let Some(cell) = s.cell[c].as_mut() {
                        let ev = jk::Ev::Keep {
                            num: None,
                            kind: None,
                            seen: None,
                        };
                        jk::step(&mut cell.st, ev);
                        cell.kept_ok = true;
                    }
                    self.notifier(&mut s, c, nk::Event::Keep { notify });
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
                let tj = s.task[r].as_ref()?.jobs[j].clone();
                let c = self.followed(&s, &tj, j)?;
                s.task[r].as_mut()?.jobs[j].link = Link::None;
                self.report(&mut s, r, j, c);
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
                // The call reads the job by id.
                let c = self.cur(&s, j);
                let ended = c.is_some_and(|c| s.cell[c].as_ref().is_some_and(|x| x.phase().over()));
                if let Some(c) = c
                    && ended
                {
                    self.report(&mut s, r, j, c);
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
                // The switch kills and wipes the old transcript's jobs: `Stop` (their processes are
                // killed), then every one whose group is empty leaves the table. What the processes
                // do next (the shell's exit, the group's end) the supervisors still observe.
                for c in 0..self.cells() {
                    let Some(cell) = s.cell[c].as_mut() else {
                        continue;
                    };
                    if cell.transcript != old || !cell.in_table {
                        continue;
                    }
                    cell.wiped = true;
                    if s.sb_alive {
                        if jk::step(&mut cell.st, jk::Ev::Stop) == jk::Act::Stop {
                            cell.killed = true;
                        }
                        let id = self.id_of(c);
                        self.remove(&mut s, Some(id), false, Some(c));
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
            Property::<Self>::always(SAFETY[8], |_, s| s.bad & S9_LIMIT == 0),
            Property::<Self>::always(SAFETY[9], |_, s| s.bad & S10_EARLY_WAKE == 0),
            Property::<Self>::always(SAFETY[10], |_, s| s.bad & S11_WAIT == 0),
            Property::<Self>::always(SAFETY[11], |_, s| s.bad & S12_SETTLED_RETRIED == 0),
            Property::<Self>::always(SAFETY[12], |_, s| s.bad & S13_FOREIGN_SETTLE == 0),
            Property::<Self>::always(SAFETY[13], |_, s| s.bad & S14_NOT_OWED == 0),
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
                        let tj = &t.jobs[j];
                        let Some(cell) = &s.cell[tj.cell as usize] else {
                            return true;
                        };
                        !(cell.phase().over() && !cell.wiped && tj.tracked) || tj.reports > 0
                    })
                },
            ),
            Property::<Self>::always(SAFETY[14], |_, s| s.bad & X1_ATTRIBUTED == 0),
            Property::<Self>::always(SAFETY[15], |_, s| s.bad & X2_REMOVED_NOTIFIED == 0),
            Property::<Self>::always(SAFETY[16], |_, s| s.bad & X3_END_TWICE == 0),
            // X4, in every settled state like L1, and over the table's own claim rather than the
            // task's tracking: a job the replica kept has had its end reported.
            Property::<Self>::always(
                "X4: once settled, every kept job's end is reported",
                |m, s| {
                    let mut acts = Vec::new();
                    m.actions(s, &mut acts);
                    if !acts.is_empty() || m.current_task(s).is_none() {
                        return true;
                    }
                    s.cell.iter().flatten().all(|c| {
                        !(c.kept_ok && c.st.kept() && c.phase().over() && !c.wiped) || c.commits > 0
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
        if self.reuse {
            props.push(Property::<Self>::sometimes(
                "W: an id is reused while a message for its old job is on the wire",
                |_, s| s.saw & W_REUSED_WITH_MSG != 0,
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

const SAFETY: [&str; 17] = [
    "S1: each end and shell exit is reported at most once per session task",
    "S2: a job is unheld only after keep with notify succeeded and a proof arrived in its window",
    "S3: a callback never opens a session",
    "S4: a forged callback changes no state",
    "S5: a fenced owner never commits a report",
    "S6: nothing on a timer alone contacts the sandbox",
    "S7: at most one attempt per job in flight",
    "S8: a report lands only in its job's own transcript",
    "S9: a count makes at most 12 attempts",
    "S10: a wake before its time begins nothing",
    "S11: the wait after failed attempt n lies in [d(n)/2, d(n)]",
    "S12: an answer that settles is never retried",
    "S13: an answer settles only its own attempt",
    "S14: a keep with notify, and a change under one, each owe one callback more",
    "X1: a callback, answer, report or removal for a job is never applied to another job that reuses its id",
    "X2: a removed job is never notified again and owes nothing",
    "X3: each job's end is committed at most once, stream, callback and read-back together",
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

/// The retry limit, for real: the receiver fails 12 attempts in a row (a `503`), and the count
/// gives up at the twelfth. Every attempt is judged as it begins and every wake as it comes, with
/// the clock moving only to the time asked for and a wake that may arrive before it.
#[test]
fn notify_one_job_retries_to_the_limit() {
    check(Notify::deep(12), "one job, 12 failed attempts");
}

/// The backoff, over the shortest, the midpoint and the longest draw, with a receiver that
/// answers `503` or `404` (which is never retried).
#[test]
fn notify_one_job_waits_over_every_draw() {
    check(Notify::drawn(3), "one job, three answers, three draws");
}

/// The seam: the real job table with the notifier. One job whose id another client may start
/// again once the job left the table, under any one network or sandbox move (a lost, late or
/// replayed callback while the id is reused; a wipe; the sandbox's death with a kept job).
#[test]
fn notify_one_job_id_reuse() {
    check(
        Notify {
            reuse: true,
            adversary: NET | SBX,
            calls: 0,
            ..Notify::new(1, 1)
        },
        "one job, its id reused, one network or sandbox fault",
    );
}

/// The seam without a fault: a `keep` that looked its job up before the job's report removed it
/// takes the job's lock afterwards, with a `bash_job` call about it too.
#[test]
fn notify_one_job_keep_races_removal() {
    check(Notify::new(1, 0), "one job, a keep racing its removal");
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

/// X2: a `keep` that looked its job up, then waited for its lock while the job's end was reported
/// and the job removed, found a notifier reset to its default (`removed` false): the kernel begins
/// a callback for a job that is gone. Reached with no fault.
#[test]
fn mutation_remove_resets_notifier_notifies_a_removed_job() {
    mutant(
        Notify::new(1, 0).mutated(Mutation::RemoveResetsNotifier),
        SAFETY[15],
        &[],
    );
}

/// X1: an attempt that outlives its job's removal and answers to the key's holder now, a job
/// another client started under the same id.
#[test]
fn mutation_answer_by_key_settles_the_successor() {
    mutant(
        Notify {
            reuse: true,
            adversary: NET,
            calls: 0,
            ..Notify::new(1, 1)
        }
        .mutated(Mutation::AnswerByKey),
        SAFETY[14],
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
