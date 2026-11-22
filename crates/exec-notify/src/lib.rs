//! The exec protocol notify **notification kernel**: when a sandbox calls back about a kept job, as
//! one pure function, [`step`].
//!
//! The normative spec is `crates/sandbox/src/exec_endpoint.rs` ("Sandbox job notify", "When
//! the endpoint calls back", "The answer, and retries"). Everything an endpoint must decide about
//! *when* to call back is here, and nothing else: one notification per job at a time; what is owed,
//! and an answer settling only what was owed when its attempt began; retry or stop by status; the
//! schedule; abandoning what is in flight when a new `keep` arrives; and none of it once the job is
//! removed or its latest `keep` carries no `notify`. Making the request (one `POST`, no redirects, a
//! verified server, 30 s at most) is the caller's, and so is the clock: time arrives as an input,
//! a random draw arrives with the event that needs one, and nothing here does I/O or waits.
//!
//! An endpoint drives it per job: feed every [`Event`] with the time, keep the [`Job`] it returns,
//! and carry out the [`Action`]s: begin an attempt (with the job's notify as recorded *now*),
//! abandon the one in flight, or be woken at a time and feed [`Event::Wake`]. The reference endpoint
//! (`beyond-test-support`'s `exec_mock`) does exactly that. A model checker
//! calls [`step`] as production does, and Verus proves it (the `verus!` block below): the invariant
//! `wf`, what `step` promises (`step_post`), and progress (`lemma_progress`). See the README.

use vstd::prelude::*;

verus! {

/// The most attempts one count makes (N18).
pub const MAX_ATTEMPTS: u32 = 12;

/// How long one attempt may take before it counts as a connection error, in milliseconds (N16).
pub const ATTEMPT_TIMEOUT_MS: u64 = 30_000;

/// The longest wait [`wait_ms`] returns, in milliseconds: `d(n)` is capped at 60 s.
pub const MAX_WAIT_MS: u64 = 60_000;

/// One job's notification state. `Default` is a job never kept with `notify`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Job {
    /// The latest `keep` carried `notify`.
    pub notify: bool,
    /// The job was removed (`remove`, `kill` with `remove`, `wipe`): nothing more, ever.
    pub removed: bool,
    /// How many callbacks the job has been owed: one per accepted `keep` with `notify`, one per
    /// state change while such a keep governs.
    pub owed: u64,
    /// How many of those an answer (or the end of a count) has settled.
    pub settled: u64,
    /// The attempt in flight, if any, as the `owed` it carries: what was owed when it began.
    pub in_flight: Option<u64>,
    /// Attempts made in the current count.
    pub attempts: u32,
    /// When the next attempt of the current count may begin, while it waits.
    pub wake_at: Option<u64>,
}

/// What happened to a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// An accepted `keep`, with `notify` or without it. With it, the job is owed a callback (the
    /// proof) and whatever was in flight is abandoned; without it, nothing more is sent.
    Keep { notify: bool },
    /// A state change while the job's latest keep carries `notify`: the moment a held stream would
    /// send `shell_exit`, or the job's end becoming observable.
    Change,
    /// The job was removed.
    Removed,
    /// The attempt in flight ended so. `draw` is a fresh uniform `u32`; the wait before a retry is
    /// `d(n) · (1 − u/2)` with `u = draw / 2³²`.
    AttemptEnded { outcome: Outcome, draw: u32 },
    /// A time the kernel asked to be woken at ([`Action::WakeAt`]) has come.
    Wake,
}

/// How one attempt ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// An answer with this status.
    Status(u16),
    /// No status: a connection error, a TLS failure, or no complete answer within 30 s.
    Failed,
}

/// What the endpoint does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Begin attempt `attempt` (1 for a count's first) with the job's notify as recorded now.
    Begin { attempt: u32 },
    /// Abandon the attempt in flight, its connection included.
    Abandon,
    /// Feed [`Event::Wake`] at this time (milliseconds, on the caller's clock).
    WakeAt(u64),
}

/// What an attempt's answer means (N17).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    /// The receiver took the hint: what the attempt carried is settled.
    Done,
    /// It never will, or the request itself is wrong: settled, and not sent again.
    Stop,
    /// Not now: try again, per the schedule.
    Retry,
}

// ---------------------------------------------------------------------------------------------
// Specs: what the code promises. A normal build erases everything in this block that is a spec or
// a proof; `mise run verify:kernels` checks it.
// ---------------------------------------------------------------------------------------------

/// The spec's table of answers (N17), as a spec.
pub open spec fn verdict_spec(outcome: Outcome) -> Verdict {
    match outcome {
        Outcome::Failed => Verdict::Retry,
        Outcome::Status(s) => {
            if 200 <= s && s <= 299 {
                Verdict::Done
            } else if s == 401 || s == 403 || s == 404 || s == 410 {
                Verdict::Stop
            } else if s == 407 || s == 408 || s == 425 || s == 429 || (500 <= s && s <= 599) {
                Verdict::Retry
            } else {
                Verdict::Stop
            }
        },
    }
}

/// `d(n)` in seconds: `min(2^(n-1), 60)`, and 1 for `n` 0 or 1 (N18).
pub open spec fn base_s(n: u32) -> int {
    if n <= 1 {
        1
    } else if n == 2 {
        2
    } else if n == 3 {
        4
    } else if n == 4 {
        8
    } else if n == 5 {
        16
    } else if n == 6 {
        32
    } else {
        60
    }
}

/// `d(n)` in milliseconds: the longest wait after failed attempt `n`.
pub open spec fn backoff_ms(n: u32) -> int {
    base_s(n) * 1000
}

/// The job owes a callback: its latest `keep` carries `notify` and something is owed that no
/// answer has settled.
pub open spec fn pending(j: Job) -> bool {
    j.notify && j.owed > j.settled
}

/// The invariant every job `step` is given and returns satisfies, from `Job::default()` on. It
/// is the safety half of the contract:
///
/// * `owed` and `settled` only move together (`settled <= owed`), and a job whose latest `keep`
///   has no `notify` owes nothing;
/// * a count never exceeds [`MAX_ATTEMPTS`] attempts;
/// * the attempt in flight carries something owed and unsettled (`settled < carried <= owed`),
///   exists only under a `notify` keep, and is the count's attempt (`attempts >= 1`), with no
///   wake waiting beside it (one notification per job at a time);
/// * a wait for a retry exists only while the job still owes, between attempts, with the count
///   short of its limit;
/// * **never lost**: a job that owes is either in flight or waiting for a wake. No state of a
///   live job owes a callback and has nothing that will send it;
/// * a removed job is the empty job: nothing owed, in flight or waiting, ever.
pub open spec fn wf(j: Job) -> bool {
    &&& j.settled <= j.owed
    &&& !j.notify ==> j.owed == j.settled
    &&& j.attempts <= MAX_ATTEMPTS
    &&& j.removed ==> (!j.notify && j.owed == 0 && j.in_flight is None && j.attempts == 0 && j.wake_at is None)
    &&& j.in_flight is Some ==> (j.notify && j.settled < j.in_flight->Some_0 && j.in_flight->Some_0 <= j.owed
        && j.attempts >= 1 && j.wake_at is None)
    &&& j.wake_at is Some ==> (j.in_flight is None && pending(j) && 1 <= j.attempts && j.attempts < MAX_ATTEMPTS)
    &&& (!j.removed && pending(j)) ==> (j.in_flight is Some || j.wake_at is Some)
}

/// The only shapes of `step`'s actions: none, one of each kind, or an abandon then a begin (a
/// `keep` replacing the attempt in flight). So an attempt begins at most once per step, last.
pub open spec fn shape(out: Seq<Action>) -> bool {
    out.len() == 0
    || (out.len() == 1 && (out[0] is Abandon || out[0] is Begin || out[0] is WakeAt))
    || (out.len() == 2 && out[0] is Abandon && out[1] is Begin)
}

/// An attempt begins (in a `shape`d `out`, the last action).
pub open spec fn has_begin(out: Seq<Action>) -> bool {
    out.len() > 0 && out[out.len() - 1] is Begin
}

/// The kernel asks to be woken (alone in `out`).
pub open spec fn has_wake(out: Seq<Action>) -> bool {
    out.len() == 1 && out[0] is WakeAt
}

/// The attempt in flight is abandoned (first in `out`).
pub open spec fn has_abandon(out: Seq<Action>) -> bool {
    out.len() > 0 && out[0] is Abandon
}

/// The event is one the job's own timers and attempts cause: the in-flight attempt ending, or
/// the asked-for wake arriving when its time has come. These are the fair steps of the liveness
/// argument; a `Keep`, `Change` or `Removed` is the environment's, and a quiet job gets none.
pub open spec fn fair(job: Job, event: Event, now: u64) -> bool {
    match event {
        Event::AttemptEnded { .. } => job.in_flight is Some,
        Event::Wake => job.in_flight is None && job.wake_at is Some && job.wake_at->Some_0 <= now,
        _ => false,
    }
}

/// The ranking function of the liveness proof: an upper bound, in steps of the fair kind, on
/// how many it takes the job to settle. Zero exactly when nothing is owed.
///
/// Each count has `MAX_ATTEMPTS - attempts` attempts left, and each costs a wake (while waiting)
/// and an end (in flight); an attempt that carries less than is owed costs one more whole count,
/// because its end begins a fresh one.
pub open spec fn rank(j: Job) -> int {
    if !pending(j) {
        0
    } else {
        let left = (MAX_ATTEMPTS - j.attempts) as int;
        match j.in_flight {
            Some(c) => if c < j.owed {
                2 * (MAX_ATTEMPTS as int) + 2 + 2 * left
            } else {
                2 * left + 1
            },
            None => 2 * left,
        }
    }
}

/// What `step` promises, event by event. See the proof functions below for the liveness half
/// that reads it.
pub open spec fn step_post(job: Job, event: Event, now: u64, j2: Job, out: Seq<Action>) -> bool {
    &&& post_removed(job, event, j2, out)
    &&& post_one(job, event, j2, out)
    &&& post_owed(job, event, j2)
    &&& post_answer(job, event, now, j2, out)
    &&& post_wake(job, event, now, j2, out)
}

/// Removal: nothing more, ever.
pub open spec fn post_removed(job: Job, event: Event, j2: Job, out: Seq<Action>) -> bool {
    &&& shape(out)
    // Nothing more, ever, once removed; `Removed` makes a job so, abandoning what was in flight.
    &&& (job.removed ==> (j2 == job && out.len() == 0))
    &&& (!job.removed && event is Removed ==> (j2.removed && !has_begin(out) && !has_wake(out)
        && (has_abandon(out) <==> job.in_flight is Some)))
    &&& (j2.removed ==> (job.removed || event is Removed))
}

/// One notification per job.
pub open spec fn post_one(job: Job, event: Event, j2: Job, out: Seq<Action>) -> bool {
    // One notification per job: an attempt begins only when none is in flight (the one in flight
    // having just ended, or a `keep` abandoning it first), only with something owed, and carries all of it.
    &&& (has_begin(out) ==> (j2.in_flight == Some(j2.owed) && pending(j2) && j2.wake_at is None
        && (job.in_flight is None || event is AttemptEnded || (event is Keep && out[0] is Abandon))
        && out[out.len() - 1] == (Action::Begin { attempt: j2.attempts })))
    // A notice that was delivered (nothing owed) is not sent again unless something new is owed.
    &&& ((!pending(job) && !(event is Keep && event->Keep_notify) && !(event is Change && job.notify))
        ==> !has_begin(out))
}

/// What is owed and settled: nothing is lost.
pub open spec fn post_owed(job: Job, event: Event, j2: Job) -> bool {
    // Everything that happens is owed, exactly once: a `keep` with notify and a change under one
    // each owe one callback more; nothing else changes what is owed.
    &&& (!job.removed && !(event is Removed) ==> j2.owed == job.owed
        + (if (event is Keep && event->Keep_notify) || (event is Change && job.notify) { 1int } else { 0int }))
    // Never lost: what an answer settles is what its attempt carried, and only for an answer
    // that settles (`Done`, `Stop`) or a count that ran out of attempts; a change, a wake or a
    // new `keep` with notify settles nothing; a `keep` without notify settles what was owed,
    // because nothing more is to be sent.
    &&& (!job.removed && (event is Change || event is Wake || (event is Keep && event->Keep_notify))
        ==> j2.settled == job.settled)
    &&& (!job.removed && event is Keep && !event->Keep_notify ==> (j2.settled == j2.owed && !j2.notify
        && j2.in_flight is None && j2.wake_at is None))
}

/// An answer, and the retry schedule.
pub open spec fn post_answer(job: Job, event: Event, now: u64, j2: Job, out: Seq<Action>) -> bool {
    &&& (!job.removed && event is AttemptEnded && job.in_flight is None ==> (j2 == job && out.len() == 0))
    &&& (!job.removed && event is AttemptEnded && job.in_flight is Some ==> {
        let c = job.in_flight->Some_0;
        let v = verdict_spec(event->AttemptEnded_outcome);
        &&& ((v is Done || v is Stop) ==> j2.settled == c)
        &&& (v is Retry ==> (j2.settled == job.settled
            || (j2.settled == c && c == job.owed && job.attempts >= MAX_ATTEMPTS)))
        // What an answer leaves owed starts a fresh count at once.
        &&& ((j2.settled == c && c < job.owed) ==> (j2.in_flight == Some(job.owed) && j2.attempts == 1))
        &&& ((v is Retry && c < job.owed) ==> (j2.in_flight == Some(job.owed) && j2.attempts == 1))
        &&& (c == job.owed && (v is Done || v is Stop || job.attempts >= MAX_ATTEMPTS) ==> j2.settled == j2.owed)
        // The retry schedule: after failed attempt n (the count's), wait d(n)·(1 − u/2) ms, and
        // never past the count's limit.
        &&& (c == job.owed && v is Retry && job.attempts < MAX_ATTEMPTS ==> {
            let at = (now + wait_ms_spec(job.attempts, event->AttemptEnded_draw)) as u64;
            &&& out =~= seq![Action::WakeAt(at)]
            &&& j2.wake_at == Some(at)
            &&& j2.in_flight is None
            &&& j2.attempts == job.attempts
            &&& j2.settled == job.settled
            &&& now + backoff_ms(job.attempts) / 2 <= at
            &&& at <= now + backoff_ms(job.attempts)
        })
        &&& (has_wake(out) ==> (c == job.owed && v is Retry && job.attempts < MAX_ATTEMPTS))
    })
}

/// A wake.
pub open spec fn post_wake(job: Job, event: Event, now: u64, j2: Job, out: Seq<Action>) -> bool {
    // A wake begins the next attempt once its time has come, and not before; it never sends
    // when nothing is owed.
    &&& (!job.removed && event is Wake && job.in_flight is None && job.wake_at is Some
        && job.wake_at->Some_0 <= now ==> (j2.in_flight == Some(job.owed) && j2.attempts == job.attempts + 1))
    &&& (event is Wake && has_begin(out) ==> (job.wake_at is Some && job.wake_at->Some_0 <= now))
    // The wake the kernel asked for is the one it keeps.
    &&& (has_wake(out) ==> (out.len() == 1 && j2.wake_at == Some(out[0]->WakeAt_0)))
}

// The wait after failed attempt `n` in ms: `a - ⌊a·draw / 2³³⌋`, less one more when the remainder
// is over 2³², with `a = d(n)` in ms, rounding `d(n) · (1 − u/2)` to the nearest millisecond (half up).
pub open spec fn wait_ms_spec(n: u32, draw: u32) -> int {
    let a = backoff_ms(n);
    let t = a * (draw as int);
    a - t / 8589934592 - (if t % 8589934592 > 4294967296 { 1int } else { 0int })
}

/// The spec's table of answers (N17).
// Verus does not take `RangeInclusive::contains` in code, so the comparisons are spelled out.
#[allow(clippy::manual_range_contains)]
pub fn verdict(outcome: Outcome) -> (v: Verdict)
    ensures
        v == verdict_spec(outcome),
{
    match outcome {
        Outcome::Failed => Verdict::Retry,
        Outcome::Status(s) => {
            if 200 <= s && s <= 299 {
                Verdict::Done
            } else if s == 401 || s == 403 || s == 404 || s == 410 {
                if cfg!(feature = "mut_stop_is_retried") { Verdict::Retry } else { Verdict::Stop }
            } else if s == 407 || s == 408 || s == 425 || s == 429 || (500 <= s && s <= 599) {
                Verdict::Retry
            } else {
                Verdict::Stop
            }
        },
    }
}

/// The wait after failed attempt `n` (1 … 11), in milliseconds: `d(n) · (1 − u/2)` with
/// `d(n) = min(2^(n−1), 60)` s and `u = draw / 2³²` (N18), rounded to the nearest millisecond
/// (half up). Integer arithmetic only, so any implementation computes exactly the same number:
/// with `a = d(n)` in ms and `t = a · draw`, it is `a − ⌊t / 2³³⌋`, less one more when the remainder
/// is over `2³²`.
///
/// Between half of `d(n)` (draw `u32::MAX`) and all of it (draw 0).
pub fn wait_ms(n: u32, draw: u32) -> (r: u64)
    ensures
        r == wait_ms_spec(n, draw),
        r <= backoff_ms(n),
        2 * r >= backoff_ms(n),
{
    let a: u64 = if n <= 1 {
        1000
    } else if n == 2 {
        2000
    } else if n == 3 {
        4000
    } else if n == 4 {
        8000
    } else if n == 5 {
        16000
    } else if n == 6 {
        32000
    } else {
        60000
    };
    let t: u64 = a * (draw as u64);
    let q: u64 = t / 8589934592;
    let rem: u64 = t % 8589934592;
    let carry: u64 = if rem > 4294967296 { 1 } else { 0 };
    proof {
        assert(a == backoff_ms(n));
        assert(a % 2 == 0 && 0 < a && a <= 60000);
        assert(t <= a * 4294967295) by (nonlinear_arith)
            requires t == a * (draw as u64), draw <= 4294967295u32;
        // q·2³³ <= t < a·2³²: so 2q < a, and q <= a/2 - 1.
        assert(q * 8589934592 <= t) by (nonlinear_arith)
            requires q == t / 8589934592;
        assert(q < a / 2) by (nonlinear_arith)
            requires q * 8589934592 <= t, t <= a * 4294967295, a % 2 == 0, 0 < a;
    }
    a - q - carry
}

/// An attempt begins: the job's next state and the action. Only called with something owed.
fn begin(j: Job) -> (r: (Job, Action))
    requires
        j.notify,
        j.owed > j.settled,
        j.attempts < MAX_ATTEMPTS,
    ensures
        r.0 == (Job { wake_at: None, attempts: (j.attempts + 1) as u32, in_flight: Some(j.owed), ..j }),
        r.1 == (Action::Begin { attempt: (j.attempts + 1) as u32 }),
{
    let mut j = j;
    j.wake_at = None;
    j.attempts += 1;
    j.in_flight = Some(j.owed);
    let a = Action::Begin { attempt: j.attempts };
    (j, a)
}

/// One step: the job's next state, and what to do. Pure: the same inputs give the same outputs.
///
/// `requires`: a job this function (or `Job::default()`) made (`wf`), and a clock and a count
/// that cannot wrap (`now` within 60 s of `u64::MAX`; `owed` below `u64::MAX`): a millisecond
/// clock and a count of events, neither reachable.
pub fn step(job: Job, event: Event, now: u64) -> (r: (Job, Vec<Action>))
    requires
        wf(job),
        job.owed < u64::MAX,
        now <= u64::MAX - MAX_WAIT_MS,
    ensures
        wf(r.0),
        post_removed(job, event, r.0, r.1@),
        post_one(job, event, r.0, r.1@),
        post_owed(job, event, r.0),
        post_answer(job, event, now, r.0, r.1@),
        post_wake(job, event, now, r.0, r.1@),
{
    let mut j = job;
    let mut out: Vec<Action> = Vec::new();
    if j.removed {
        return (j, out);
    }
    match event {
        Event::Keep { notify } => {
            if j.in_flight.is_some() && !cfg!(feature = "mut_keep_keeps_in_flight") {
                j.in_flight = None;
                out.push(Action::Abandon);
            }
            j.wake_at = None;
            j.attempts = 0;
            j.notify = notify;
            if notify {
                j.owed += 1;
                let (b, a) = begin(j);
                j = b;
                out.push(a);
            } else {
                j.settled = j.owed;
            }
        },
        Event::Change => {
            if !j.notify {
                return (j, out);
            }
            if !cfg!(feature = "mut_change_not_owed") || j.in_flight.is_none() {
                j.owed += 1;
            }
            if j.in_flight.is_none() || cfg!(feature = "mut_change_begins_in_flight") {
                // Idle, or waiting out a retry: a new count begins now.
                j.wake_at = None;
                j.attempts = 0;
                let (b, a) = begin(j);
                j = b;
                out.push(a);
            }
        },
        Event::Removed => {
            if j.in_flight.is_some() {
                out.push(Action::Abandon);
            }
            j = Job {
                notify: false,
                removed: true,
                owed: 0,
                settled: 0,
                in_flight: None,
                attempts: 0,
                wake_at: None,
            };
        },
        Event::AttemptEnded { outcome, draw } => {
            let carried = match j.in_flight {
                Some(c) => c,
                None => {
                    return (j, out);
                },
            };
            j.in_flight = None;
            match verdict(outcome) {
                Verdict::Done | Verdict::Stop => {
                    j.settled = if cfg!(feature = "mut_answer_settles_all") {
                        j.owed
                    } else if j.settled > carried {
                        j.settled
                    } else {
                        carried
                    };
                    j.attempts = 0;
                },
                Verdict::Retry => {
                    if j.owed > carried {
                        j.attempts = 0;
                    } else if j.attempts >= MAX_ATTEMPTS && !cfg!(feature = "mut_retry_unbounded") {
                        j.settled = if j.settled > carried { j.settled } else { carried };
                        j.attempts = 0;
                    } else {
                        let at = now + wait_ms(j.attempts, draw);
                        j.wake_at = Some(at);
                        out.push(Action::WakeAt(at));
                        return (j, out);
                    }
                },
            }
            if j.notify && j.owed > j.settled {
                let (b, a) = begin(j);
                j = b;
                out.push(a);
            }
        },
        Event::Wake => {
            if j.in_flight.is_none() && j.notify && j.owed > j.settled
                && match j.wake_at {
                    Some(at) => at <= now || cfg!(feature = "mut_wake_early"),
                    None => false,
                } {
                let (b, a) = begin(j);
                j = b;
                out.push(a);
            }
        },
    }
    (j, out)
}

// ---------------------------------------------------------------------------------------------
// Liveness: a job that owes settles, in at most `rank` fair steps.
// ---------------------------------------------------------------------------------------------

/// **Progress.** Under fairness, every step the job's own timers and attempts cause (`fair`)
/// strictly lowers [`rank`] while the job owes, and rank 0 is exactly "nothing owed": the
/// callback was delivered (`Done`), refused for good (`Stop`), or given up after
/// [`MAX_ATTEMPTS`]. The fairness assumptions, all of them the environment's, none a bound
/// the kernel supplies:
///
/// * **F1, attempts end**: the attempt in flight ends, with the answer or, after
///   [`ATTEMPT_TIMEOUT_MS`], as `Failed`. The caller owes that event.
/// * **F2, time advances**: a wake the kernel asked for is fed at or after its time.
/// * **F3, the job quiesces**: after some point no `Keep` or `Change` arrives. (A job
///   that keeps changing is owed a new callback each time; the rank restarts.)
///
/// Not an `assume`: `step_post`, which `step` proves of the code, is all it uses.
pub proof fn lemma_progress(job: Job, event: Event, now: u64, j2: Job, out: Seq<Action>)
    requires
        wf(job),
        wf(j2),
        job.owed < u64::MAX,
        pending(job),
        fair(job, event, now),
        step_post(job, event, now, j2, out),
    ensures
        rank(j2) < rank(job),
{
    match event {
        Event::AttemptEnded { outcome, draw } => {
            let c = job.in_flight->Some_0;
            let v = verdict_spec(outcome);
            assert(!job.removed);
            if c < job.owed {
                assert(j2.in_flight == Some(job.owed));
            } else {
                assert(c == job.owed);
            }
        },
        Event::Wake => {
            assert(!job.removed);
        },
        _ => {},
    }
}

/// The rank is bounded: a job that owes settles in at most this many fair steps.
pub open spec fn rank_bound() -> int {
    2 * (MAX_ATTEMPTS as int) + 2 + 2 * (MAX_ATTEMPTS as int)
}

/// Every job of the invariant has a bounded rank, and rank 0 is exactly nothing owed.
pub proof fn lemma_rank_bounded(j: Job)
    requires
        wf(j),
    ensures
        0 <= rank(j) <= rank_bound(),
        rank(j) == 0 <==> !pending(j),
{
}

} // verus!

/// The latest clock [`step`] accepts (`requires now <= u64::MAX - MAX_WAIT_MS`), in milliseconds.
pub const MAX_NOW_MS: u64 = u64::MAX - MAX_WAIT_MS;

/// A caller's clock (milliseconds since its own epoch) as [`step`] needs it: saturating at
/// [`MAX_NOW_MS`], never wrapping. Reaching it takes 5.8e8 years; if it ever did, time stops
/// there and no wait wraps.
pub fn clock(ms: u128) -> u64 {
    if ms > MAX_NOW_MS as u128 {
        MAX_NOW_MS
    } else {
        ms as u64
    }
}

/// [`step`] for a caller that cannot prove its preconditions: `now` saturates ([`clock`]'s bound),
/// and an event that would owe one more callback when `owed` is within one of `u64::MAX` is
/// refused (the job is returned as it was, nothing is sent), so `owed < u64::MAX` always holds.
/// Neither is reachable: one callback is owed per `keep` with `notify` or state change.
pub fn step_checked(job: Job, event: Event, now: u64) -> (Job, Vec<Action>) {
    let owes_more = match event {
        Event::Keep { notify } => notify,
        Event::Change => job.notify,
        _ => false,
    };
    if !job.removed && owes_more && job.owed >= u64::MAX - 1 {
        return (job, Vec::new());
    }
    step(job, event, now.min(MAX_NOW_MS))
}

/// `u` in `[0, 1)` for a draw.
pub fn uniform(draw: u32) -> f64 {
    f64::from(draw) / 4_294_967_296.0
}

/// `keep`'s `notify` grammar (N4): `url` absolute, at most 2048 bytes, a non-empty host, no
/// userinfo, no fragment, and `https` (or `http` to `localhost`, `127.0.0.0/8` or `[::1]`); `token`
/// 1 to 512 bytes of `token68`. The rule broken, if any, never quoting the token. Checked on the
/// bytes as given: a URL parser forgives what the grammar does not (surrounding spaces,
/// `https:///x`).
pub fn check_notify(url: &str, token: &str) -> Result<(), &'static str> {
    check_url(url)?;
    if token.is_empty() || token.len() > 512 {
        return Err("token length");
    }
    let body = token.trim_end_matches('=');
    if body.is_empty()
        || !body
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~+/".contains(&b))
    {
        return Err("token68");
    }
    Ok(())
}

fn check_url(url: &str) -> Result<(), &'static str> {
    if url.len() > 2048 {
        return Err("url length");
    }
    if url.bytes().any(|b| b <= b' ' || b >= 0x7f) {
        return Err("absolute");
    }
    let (scheme, rest) = url.split_once("://").ok_or("absolute")?;
    if !scheme
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
        || !scheme
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
    {
        return Err("absolute");
    }
    if url.contains('#') {
        return Err("fragment");
    }
    let authority = &rest[..rest.find(['/', '?']).unwrap_or(rest.len())];
    if authority.contains('@') {
        return Err("userinfo");
    }
    let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
        let end = v6.find(']').ok_or("absolute")?;
        let addr = &v6[..end];
        if addr.parse::<std::net::Ipv6Addr>().is_err() {
            return Err("absolute");
        }
        let after = &v6[end + 1..];
        let port = match after.strip_prefix(':') {
            Some(p) => Some(p),
            None if after.is_empty() => None,
            None => return Err("absolute"),
        };
        (&authority[..end + 2], port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        }
    };
    if host.is_empty() {
        return Err("host");
    }
    if !host.starts_with('[')
        && !host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~%!$&'()*+,;=".contains(&b))
    {
        return Err("absolute");
    }
    if let Some(p) = port
        && !p.is_empty()
        && !(p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok())
    {
        return Err("absolute");
    }
    match scheme.to_ascii_lowercase().as_str() {
        "https" => Ok(()),
        "http" if loopback(host) => Ok(()),
        _ => Err("scheme"),
    }
}

/// `localhost`, an address in `127.0.0.0/8`, or `[::1]`.
fn loopback(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Some(v6) = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        return v6.parse::<std::net::Ipv6Addr>() == Ok(std::net::Ipv6Addr::LOCALHOST);
    }
    host.parse::<std::net::Ipv4Addr>()
        .is_ok_and(|a| a.octets()[0] == 127)
}

/// The callback body (N14): exactly `type`, `scope` and `id`, in that order, as JSON strings.
pub fn body(scope: &str, id: &str) -> String {
    let mut out = String::with_capacity(40 + scope.len() + id.len());
    out.push_str("{\"type\":\"job_ended\",\"scope\":");
    json_string(&mut out, scope);
    out.push_str(",\"id\":");
    json_string(&mut out, id);
    out.push('}');
    out
}

fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests;
