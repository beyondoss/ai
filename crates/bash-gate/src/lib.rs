//! The agent's `bash` decisions, as pure functions proven in Verus.
//!
//! * **B1, the running-command limit** ([`admit_start`]): a new command is admitted below
//!   [`MAX_RUNNING_JOBS`] running ones and refused at it. The decision has no third answer: nothing
//!   that runs is stopped to make room ([`lemma_a_start_stops_nothing`]); the table after a start
//!   holds every job it held before.
//! * **B2, the hand-back** ([`yield_reason`]): when a foreground call returns to the model while its
//!   command keeps running as a held job: after [`IDLE_YIELD_MS`] of silence, at the call's ceiling,
//!   or at the model's own `timeout_ms`. Returning is the whole effect: a hand-back kills nothing, and
//!   a call that has waited past its ceiling always returns (progress).
//!
//! * **B3, a job's end is told once** ([`end_due`]): the session is told that a sandbox job ended
//!   once, across a replica's death. What survives is the transcript: a notice it already holds is
//!   told, whatever the sandbox's directory for the job still says. An ended job whose notice
//!   nobody holds is always due (the retry after a restart tells it).
//!
//!
//! * **B4, a kept job is followed while the turn lives** ([`follow_after_break`], [`backoff_ms`]):
//!   when the followed stream of a kept job breaks, the follow goes on, after a wait that grows from
//!   [`FOLLOW_BACKOFF_BASE_MS`] to at most [`FOLLOW_BACKOFF_CAP_MS`], and stops only on the job's end
//!   (`Lost`, `Gone`), a terminal error class, or the turn's own cancel. No count of breaks and no
//!   timer ends it: a front that is down for an hour is followed for an hour.
//!
//! Durations are in nanoseconds; this crate reads no clock.

use vstd::prelude::*;

verus! {

/// The most commands running at once under one `bash` (every kind counts).
pub const MAX_RUNNING_JOBS: usize = 20;

/// The silence after which a foreground call returns to the model and leaves its command running.
pub const IDLE_YIELD_MS: u64 = 60_000;

/// Nanoseconds in a millisecond.
pub const NS_PER_MS: u64 = 1_000_000;

/// What a start does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// The command starts and takes a place in the table.
    Start,
    /// It is refused, listing what runs; the table is as it was.
    Refuse,
}

/// B1. Whether a command may start, given how many are running. The answer is a start or a refusal,
/// never a stop of something running.
pub fn admit_start(running: usize) -> (r: Admit)
    ensures
        (r is Refuse) <==> running >= MAX_RUNNING_JOBS,
        (r is Start) <==> running < MAX_RUNNING_JOBS,
{
    if cfg!(feature = "mut_gate_ignores_limit") {
        Admit::Start
    } else if running >= MAX_RUNNING_JOBS {
        Admit::Refuse
    } else {
        Admit::Start
    }
}

/// The table of running jobs, by id, after a start: with the new id on a start, as it was on a
/// refusal. There is no arm that removes one.
pub open spec fn table_after(table: Set<int>, id: int, a: Admit) -> Set<int> {
    match a {
        Admit::Start => table.insert(id),
        Admit::Refuse => table,
    }
}

/// **B1.** Whatever the count, a start or a refusal leaves every running job running, at twenty or at
/// any other count.
pub proof fn lemma_a_start_stops_nothing(table: Set<int>, id: int, a: Admit)
    ensures
        forall|j: int| #[trigger] table.contains(j) ==> table_after(table, id, a).contains(j),
{
}

/// How a call holds the command it started (or a wait on it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    /// A background start: returns once its settle time has passed.
    Settle(u64),
    /// A foreground call: its `ceiling`, and the model's `timeout_ms` if it gave one.
    Foreground { ceiling: u64, timeout_ms: Option<u64> },
}

/// Why a call returns to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// A background start's settle time passed.
    Settled,
    /// The model's `timeout_ms` passed.
    Timeout,
    /// The command was silent for the idle window.
    Quiet,
    /// The call kept waiting up to its ceiling.
    Ceiling,
}

/// B2. Why a call that has `waited` since it attached, with the command silent for `quiet`, returns to
/// the model now, if it does; `window` is the idle window (`None`: never hand back for silence). It
/// only ever says a call returns: the command is parked as a held job, never stopped.
pub fn yield_reason(hold: Hold, waited: u64, quiet: u64, window: Option<u64>) -> (r: Option<Why>)
    ensures
        match hold {
            Hold::Settle(s) => r == (if waited >= s { Some(Why::Settled) } else { None::<Why> }),
            Hold::Foreground { ceiling, timeout_ms } => match window {
                None => r is None,
                Some(w) => r == (if timeout_ms is Some && waited / NS_PER_MS >= timeout_ms->Some_0 {
                    Some(Why::Timeout)
                } else if quiet >= w {
                    Some(Why::Quiet)
                } else if waited >= ceiling {
                    Some(Why::Ceiling)
                } else {
                    None::<Why>
                }),
            },
        },
{
    match hold {
        Hold::Settle(s) => if waited >= s { Some(Why::Settled) } else { None },
        Hold::Foreground { ceiling, timeout_ms } => match window {
            None => None,
            Some(w) => {
                let timed_out = match timeout_ms {
                    Some(ms) => waited / NS_PER_MS >= ms,
                    None => false,
                };
                if timed_out {
                    Some(Why::Timeout)
                } else if quiet >= w && !cfg!(feature = "mut_quiet_never_hands_back") {
                    Some(Why::Quiet)
                } else if waited >= ceiling {
                    Some(Why::Ceiling)
                } else {
                    None
                }
            },
        },
    }
}

/// **B2, the default window.** With the 60 s window a command silent for 60 s is handed back (unless
/// the model's `timeout_ms` came first, which also returns): the call returns, whichever applies.
pub proof fn lemma_sixty_seconds_hands_back(hold: Hold, waited: u64, quiet: u64)
    requires
        hold is Foreground,
        quiet as int >= IDLE_YIELD_MS as int * NS_PER_MS as int,
    ensures
        ({
            let w = Some((IDLE_YIELD_MS * NS_PER_MS) as u64);
            let timeout = match hold {
                Hold::Foreground { ceiling, timeout_ms } => timeout_ms is Some && waited / NS_PER_MS >= timeout_ms->Some_0,
                _ => false,
            };
            quiet >= w->Some_0 && (timeout || quiet >= w->Some_0)
        }),
{
}

/// **B2, progress.** A foreground call with a window never waits past its ceiling: once `waited`
/// has reached it, the call returns. (The partner of every refusal to return: it clears.)
pub proof fn lemma_a_call_returns_by_its_ceiling(ceiling: u64, timeout_ms: Option<u64>, waited: u64, quiet: u64, w: u64)
    requires
        waited >= ceiling,
    ensures
        ({
            let r = if timeout_ms is Some && waited / NS_PER_MS >= timeout_ms->Some_0 {
                Some(Why::Timeout)
            } else if quiet >= w {
                Some(Why::Quiet)
            } else if waited >= ceiling {
                Some(Why::Ceiling)
            } else {
                None::<Why>
            };
            r is Some
        }),
{
}

/// B3. Whether a job's end notice is due. `told_live` is this process having delivered it;
/// `told_persisted` is the session's persisted transcript already holding it (a replica that died
/// between persisting the report and deleting the job's directory left the job behind, ended). The
/// iff is both halves: a held notice is never due again, and an ended job nobody holds always is.
pub fn end_due(ended: bool, told_live: bool, told_persisted: bool) -> (r: bool)
    ensures
        r <==> (ended && !told_live && !told_persisted),
{
    if cfg!(feature = "mut_told_ignores_transcript") {
        ended && !told_live
    } else {
        ended && !told_live && !told_persisted
    }
}

/// The first wait before following a broken stream again (B4).
pub const FOLLOW_BACKOFF_BASE_MS: u64 = 500;

/// No wait before a follow is longer than this (B4).
pub const FOLLOW_BACKOFF_CAP_MS: u64 = 5_000;

/// The wait after `breaks` breaks at one offset: 500 ms, doubling, capped.
pub open spec fn backoff_spec(breaks: int) -> int {
    if breaks <= 0 { 500 } else if breaks == 1 { 1000 } else if breaks == 2 { 2000 }
    else if breaks == 3 { 4000 } else { 5000 }
}

/// **B4, backoff.** The wait before the next follow is the spec's, so it is at least
/// [`FOLLOW_BACKOFF_BASE_MS`] (a down front is not spun on) and at most [`FOLLOW_BACKOFF_CAP_MS`]
/// (the wait never grows without bound), for every count of breaks.
pub fn backoff_ms(breaks: u32) -> (ms: u64)
    ensures
        ms as int == backoff_spec(breaks as int),
        FOLLOW_BACKOFF_BASE_MS <= ms,
        ms <= FOLLOW_BACKOFF_CAP_MS,
{
    if cfg!(feature = "mut_backoff_no_cap") {
        if breaks >= 4 { 10_000 } else if breaks == 3 { 4000 } else if breaks == 2 { 2000 }
        else if breaks == 1 { 1000 } else { 500 }
    } else if cfg!(feature = "mut_backoff_spins") {
        0
    } else if breaks == 0 {
        500
    } else if breaks == 1 {
        1000
    } else if breaks == 2 {
        2000
    } else if breaks == 3 {
        4000
    } else {
        5000
    }
}

/// What the agent sees when the followed stream of a kept job has broken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seen {
    /// The sandbox answered and the job is still its own: running, or ended and not yet read to
    /// its end.
    Held,
    /// The sandbox answered `Lost` or `Gone`: the job's end.
    Over,
    /// The sandbox could not be reached this moment (a front with no replica yet, a reset, a
    /// timeout): a retryable answer.
    Unreachable,
    /// An answer no retry changes: not authorised, not offered, a protocol error.
    Refused,
    /// The turn's own cancel.
    Cancelled,
}

/// Why a follow ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopWhy {
    Over,
    Refused,
    Cancelled,
}

/// What to do after a break.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Follow {
    /// Follow again from the offset reached, after `wait_ms`.
    Again { wait_ms: u64 },
    Stop(StopWhy),
}

/// **B4.** What to do after the stream broke for the `breaks`th time at one offset, having `seen`.
///
/// * Safety: the follow stops only when the job is over, the answer is terminal, or the turn is
///   cancelled; each of those stops it, with its own reason.
/// * Liveness: a `Held` or `Unreachable` answer is always followed again (so once the front answers,
///   the next follow reattaches), after a wait of at most [`FOLLOW_BACKOFF_CAP_MS`], whatever
///   `breaks` is.
pub fn follow_after_break(seen: Seen, breaks: u32) -> (r: Follow)
    ensures
        (r is Stop) ==> (seen is Over || seen is Refused || seen is Cancelled),
        (seen is Over) ==> r == Follow::Stop(StopWhy::Over),
        (seen is Refused) ==> r == Follow::Stop(StopWhy::Refused),
        (seen is Cancelled) ==> r == Follow::Stop(StopWhy::Cancelled),
        (seen is Held || seen is Unreachable) ==> r == (Follow::Again { wait_ms: backoff_spec(breaks as int) as u64 }),
{
    match seen {
        Seen::Held | Seen::Unreachable => {
            if cfg!(feature = "mut_follow_gives_up") && breaks > 120 {
                Follow::Stop(StopWhy::Over)
            } else {
                Follow::Again { wait_ms: backoff_ms(breaks) }
            }
        },
        Seen::Over => {
            if cfg!(feature = "mut_follow_past_the_end") {
                Follow::Again { wait_ms: backoff_ms(breaks) }
            } else {
                Follow::Stop(StopWhy::Over)
            }
        },
        Seen::Refused => Follow::Stop(StopWhy::Refused),
        Seen::Cancelled => {
            if cfg!(feature = "mut_follow_past_cancel") {
                Follow::Again { wait_ms: backoff_ms(breaks) }
            } else {
                Follow::Stop(StopWhy::Cancelled)
            }
        },
    }
}

/// **B4, progress.** The ranking function is the front's remaining outage in probes. Under the
/// fairness axiom F (the front eventually answers: some probe `n` sees `Held`) and a front that
/// is merely unreachable before it, no probe up to `n` stops the follow, each waits at most the cap,
/// and probe `n` follows again, so the stream reattaches after at most `n + 1` waits of at most
/// [`FOLLOW_BACKOFF_CAP_MS`] each, whatever `n` is: no count of breaks ends the follow.
pub proof fn lemma_follow_reattaches_once_the_front_answers(trace: Seq<Seen>, n: int)
    requires
        0 <= n < trace.len(),
        trace[n] is Held,
        forall|i: int| 0 <= i < n ==> #[trigger] trace[i] is Unreachable,
    ensures
        forall|i: int| 0 <= i <= n ==> (#[trigger] follow_spec(trace[i], i) is Again
            && follow_wait(follow_spec(trace[i], i)) <= FOLLOW_BACKOFF_CAP_MS as int),
{
    assert forall|i: int| 0 <= i <= n implies (#[trigger] follow_spec(trace[i], i) is Again
        && follow_wait(follow_spec(trace[i], i)) <= FOLLOW_BACKOFF_CAP_MS as int) by {
        assert(trace[i] is Held || trace[i] is Unreachable);
    }
}

/// [`follow_after_break`] as a spec, for the lemmas.
pub open spec fn follow_spec(seen: Seen, breaks: int) -> Follow {
    match seen {
        Seen::Held | Seen::Unreachable => Follow::Again { wait_ms: backoff_spec(breaks) as u64 },
        Seen::Over => Follow::Stop(StopWhy::Over),
        Seen::Refused => Follow::Stop(StopWhy::Refused),
        Seen::Cancelled => Follow::Stop(StopWhy::Cancelled),
    }
}

/// The wait a follow decision carries (0 for a stop).
pub open spec fn follow_wait(f: Follow) -> int {
    match f {
        Follow::Again { wait_ms } => wait_ms as int,
        Follow::Stop(_) => 0,
    }
}

/// **B4, backoff.** The wait never shrinks as breaks pile up and never passes the cap.
pub proof fn lemma_backoff_is_monotone_and_capped(a: int, b: int)
    requires
        a <= b,
    ensures
        backoff_spec(a) <= backoff_spec(b),
        backoff_spec(a) <= FOLLOW_BACKOFF_CAP_MS as int,
{
}

} // verus!

#[cfg(test)]
mod tests {
    use super::*;

    /// B1: below the limit a start is admitted, at it and over it refused. Red under
    /// `mut_gate_ignores_limit`.
    #[test]
    fn the_limit_gates_a_start() {
        assert_eq!(admit_start(0), Admit::Start);
        assert_eq!(admit_start(MAX_RUNNING_JOBS - 1), Admit::Start);
        assert_eq!(admit_start(MAX_RUNNING_JOBS), Admit::Refuse);
        assert_eq!(admit_start(MAX_RUNNING_JOBS + 5), Admit::Refuse);
    }

    /// B3: a notice the transcript holds is not due again; an ended job it does not hold is. Red
    /// under `mut_told_ignores_transcript`.
    #[test]
    fn a_persisted_end_is_not_told_again() {
        assert!(!end_due(true, false, true));
        assert!(end_due(true, false, false));
        assert!(!end_due(true, true, false));
        assert!(!end_due(false, false, false));
    }

    /// B2: 60 s of silence hands a foreground call back; 59 s does not. Red under
    /// `mut_quiet_never_hands_back`.
    #[test]
    fn sixty_quiet_seconds_hand_a_call_back() {
        let w = Some(IDLE_YIELD_MS * NS_PER_MS);
        let hold = Hold::Foreground {
            ceiling: 600_000 * NS_PER_MS,
            timeout_ms: None,
        };
        assert_eq!(
            yield_reason(hold, 59_000 * NS_PER_MS, 59_000 * NS_PER_MS, w),
            None
        );
        assert_eq!(
            yield_reason(hold, 60_000 * NS_PER_MS, 60_000 * NS_PER_MS, w),
            Some(Why::Quiet)
        );
        assert_eq!(
            yield_reason(hold, 600_000 * NS_PER_MS, 1, w),
            Some(Why::Ceiling)
        );
        assert_eq!(yield_reason(hold, 1, 1, None), None);
    }

    /// B4: a stream that keeps breaking is followed again at every count (no cap), the wait grows
    /// to 5 s and stays there; only the job's end, a terminal answer or a cancel stops it. Red
    /// under `mut_follow_gives_up`, `mut_follow_past_the_end`, `mut_follow_past_cancel`,
    /// `mut_backoff_no_cap`, `mut_backoff_spins`.
    #[test]
    fn a_broken_stream_is_followed_until_the_job_is_over() {
        for n in [0u32, 1, 3, 4, 120, 121, 10_000, u32::MAX] {
            for seen in [Seen::Held, Seen::Unreachable] {
                let f = follow_after_break(seen, n);
                assert!(
                    matches!(f, Follow::Again { wait_ms }
                        if (FOLLOW_BACKOFF_BASE_MS..=FOLLOW_BACKOFF_CAP_MS).contains(&wait_ms)),
                    "{n} breaks: {f:?}"
                );
            }
        }
        assert_eq!(backoff_ms(0), 500);
        assert_eq!(backoff_ms(1_000), 5_000);
        assert_eq!(
            follow_after_break(Seen::Over, 0),
            Follow::Stop(StopWhy::Over)
        );
        assert_eq!(
            follow_after_break(Seen::Refused, 0),
            Follow::Stop(StopWhy::Refused)
        );
        assert_eq!(
            follow_after_break(Seen::Cancelled, 0),
            Follow::Stop(StopWhy::Cancelled)
        );
    }
}
