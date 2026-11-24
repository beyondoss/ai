//! execd's wire timing and kill-scope decisions, as pure functions proven in Verus.
//!
//! Three decisions the exec protocol's promises rest on, each written inside `verus!` so the function
//! execd runs is the one Verus checks (`execd`'s `exec.rs` and `proc.rs` call them; no other source
//! restates them):
//!
//! * **W1, the heartbeat** ([`quiet_wait`]): how long a stream (or a one-shot answer) that last sent a
//!   byte `since` ago waits before it sends a heartbeat. Proven: a stream is never silent past one
//!   beat ([`lemma_never_silent_past_a_beat`]), and a silence of any length, 335 s included, has a
//!   heartbeat in every beat-long window ([`lemma_silence_is_carried`]); the beat is 5 s, below the
//!   295 s the gateway allows without a byte ([`lemma_beat_below_idle`]).
//! * **W1b, the interval a client chooses** ([`beat_ms`]): a request may name its own beat, as whole
//!   seconds in the `Keepalive-Ping-Interval` header (E2B's envd takes the same header). Proven: the
//!   beat in force is the client's when it is in 1 to 240 s, else the default, and never outside
//!   1 to 240 s unless the default is ([`lemma_interval_is_in_range_or_default`]); 240 s is below
//!   the gateway's 295 s, the edge's 300 s and the load balancer's 350 s
//!   ([`lemma_interval_below_idle`]), so a stream at any accepted interval is never silent past
//!   one interval ([`lemma_never_silent_past_a_beat`]) and never past an idle limit. There is no
//!   "off": zero and below take the default.
//! * **W2, the one-shot's first answer** ([`oneshot_mode`]): a command that has not answered by the
//!   first beat gets its `200` headers then (the answer is committed), whatever it ends in; one that
//!   has answers as ever, status and all.
//! * **W3, a job's processes** ([`is_member`]): the processes a signal to a job reaches are its group's
//!   members and everything that carries its mark, wherever it went (`setsid`, a double fork). Proven:
//!   every member is reached, by the group signal or by the one sent to each marked process outside
//!   the group ([`lemma_every_member_is_signalled`]).
//!
//! Times are in nanoseconds on the caller's clock; this crate reads none.

use vstd::prelude::*;

verus! {

/// The spec's heartbeat: a stream silent this long gets `{}`; a one-shot a space (W1).
pub const HEARTBEAT_MS: u64 = 5_000;

/// The shortest interval a client may ask for, in seconds (W1b).
pub const MIN_INTERVAL_S: u64 = 1;

/// The longest interval a client may ask for, in seconds (W1b): under the gateway's 295 s, the edge's
/// 300 s and the load balancer's 350 s, with the front lengthening its own stall to stay over it.
pub const MAX_INTERVAL_S: u64 = 240;

/// The longest a response may go without a byte before the gateway cuts it: instd's gateway cuts a
/// response with no headers at 295 s.
pub const GATEWAY_IDLE_MS: u64 = 295_000;

/// The edge's read timeout (300 s) and the load balancer's idle timeout (350 s): the outer hops
/// the gateway's limit nests inside (`api/opkernel` `limits.rs`).
pub const EDGE_IDLE_MS: u64 = 300_000;
pub const LB_IDLE_MS: u64 = 350_000;

/// The front's stall limit for a stream with no byte (api/app/execfront `StallStream`, 45 s): the tightest
/// limit a heartbeat has to cross, and the one Beyond's `configured_limits_nest` takes the heartbeat under.
pub const FRONT_STALL_MS: u64 = 45_000;

/// Nanoseconds in a millisecond.
pub const NS_PER_MS: u64 = 1_000_000;

// ---------------------------------------------------------------------------------------------
// W1: the heartbeat.
// ---------------------------------------------------------------------------------------------

/// The wait, as a spec: nothing once `since` has reached the beat, else the rest of the beat.
pub open spec fn quiet_wait_spec(since: int, beat: int) -> int {
    if since >= beat { 0 } else { beat - since }
}

/// When the next byte goes out for a stream whose last byte went `since` ago: now plus the wait.
pub open spec fn frame_by(since: int, beat: int) -> int {
    since + quiet_wait_spec(since, beat)
}

/// W1. How long a stream whose last byte went out `since` ago waits before its heartbeat, for a
/// `beat` between heartbeats: zero when the beat has passed (the heartbeat is due now), else what is
/// left of it.
pub fn quiet_wait(since: u64, beat: u64) -> (w: u64)
    ensures
        w == quiet_wait_spec(since as int, beat as int),
{
    if cfg!(feature = "mut_beat_never") {
        u64::MAX / 8
    } else if since >= beat {
        0
    } else if cfg!(feature = "mut_beat_late") {
        // Wrong by one whole beat.
        beat - since + beat
    } else {
        beat - since
    }
}

/// **W1, no silence past a beat.** The next byte of a stream goes out no later than one beat after
/// the last (or at once, if the beat has already passed), and exactly one beat after it when the last
/// byte has just gone: whatever the command does or does not print, a hop sees at most one beat of
/// silence from this sender.
pub proof fn lemma_never_silent_past_a_beat(since: int, beat: int)
    requires
        0 <= since,
        0 < beat,
    ensures
        frame_by(since, beat) <= (if since > beat { since } else { beat }),
        frame_by(0, beat) == beat,
        frame_by(since, beat) >= since,
{
}

/// The beat is below the front's stall (45 s) and the gateway's idle limit (295 s, which is below the
/// edge's and the load balancer's), so one beat of silence never reaches any of them.
pub proof fn lemma_beat_below_idle()
    ensures
        HEARTBEAT_MS < FRONT_STALL_MS,
        FRONT_STALL_MS < GATEWAY_IDLE_MS,
        HEARTBEAT_MS < GATEWAY_IDLE_MS,
        HEARTBEAT_MS as int * 59 == GATEWAY_IDLE_MS as int,
{
}

/// Whether `secs` is an interval a client may ask for.
pub open spec fn interval_ok(secs: int) -> bool {
    MIN_INTERVAL_S as int <= secs && secs <= MAX_INTERVAL_S as int
}

/// W1b. The beat, in milliseconds, a request runs at: the client's interval (whole seconds, as the
/// header carried it; `None` when it is absent or not an integer) when it is in 1 to 240 s, else
/// `default_ms`. Zero, a negative number and anything over 240 take the default: there is no way to
/// turn the heartbeat off.
#[allow(clippy::manual_range_contains, clippy::if_same_then_else)]
pub fn beat_ms(requested_s: Option<i64>, default_ms: u64) -> (ms: u64)
    ensures
        requested_s is Some && interval_ok(requested_s->Some_0 as int)
            ==> ms as int == requested_s->Some_0 as int * 1000,
        !(requested_s is Some && interval_ok(requested_s->Some_0 as int)) ==> ms == default_ms,
{
    match requested_s {
        Some(s) => {
            if cfg!(feature = "mut_interval_unbounded") && s >= 1 && s <= 1_000_000 {
                (s as u64) * 1000
            } else if cfg!(feature = "mut_interval_off") && s >= 0 && s <= 240 {
                (s as u64) * 1000
            } else if s >= 1 && s <= 240 {
                (s as u64) * 1000
            } else {
                default_ms
            }
        },
        None => default_ms,
    }
}

/// **W1b, the interval is in range or the default.** Whatever a client sends, the beat in force is
/// either the default or a whole number of seconds from 1 to 240 (never zero: no "off"; never over
/// 240 s).
pub proof fn lemma_interval_is_in_range_or_default(requested_s: Option<i64>, default_ms: u64, ms: u64)
    requires
        0 < default_ms,
        ms == (if requested_s is Some && interval_ok(requested_s->Some_0 as int) {
            requested_s->Some_0 as int * 1000
        } else {
            default_ms as int
        }),
    ensures
        0 < ms,
        ms == default_ms || (ms as int % 1000 == 0 && interval_ok(ms as int / 1000)),
{
    if requested_s is Some && interval_ok(requested_s->Some_0 as int) {
        let s = requested_s->Some_0 as int;
        assert(ms as int == s * 1000);
        assert(ms as int % 1000 == 0 && ms as int / 1000 == s) by (nonlinear_arith)
            requires ms as int == s * 1000;
    }
}

/// **W1b, every interval a client may choose is below every idle limit.** The longest, 240 s, is
/// under the gateway's 295 s, which is under the edge's 300 s and the load balancer's 350 s
/// (`api/opkernel` `limits.rs`), so no hop sees a silence as long as a limit; the default is one of
/// them (5 s); and the shortest is a real beat (1 s), never zero.
pub proof fn lemma_interval_below_idle()
    ensures
        MAX_INTERVAL_S as int * 1000 < GATEWAY_IDLE_MS as int,
        GATEWAY_IDLE_MS < EDGE_IDLE_MS,
        EDGE_IDLE_MS < LB_IDLE_MS,
        MAX_INTERVAL_S as int * 1000 < EDGE_IDLE_MS as int,
        MAX_INTERVAL_S as int * 1000 < LB_IDLE_MS as int,
        interval_ok(HEARTBEAT_MS as int / 1000),
        0 < MIN_INTERVAL_S,
{
}

/// The time of the `k`th heartbeat of a silence: `k` beats in.
pub open spec fn beat_at(k: int, beat: int) -> int {
    k * beat
}

/// A heartbeat falls in the beat-long window after `t`.
pub open spec fn has_beat_in(t: int, beat: int) -> bool {
    exists|k: int| t < #[trigger] beat_at(k, beat) && beat_at(k, beat) <= t + beat
}

/// **A silence of any length is carried.** In a silence of `silent` (a command that prints nothing
/// for 335 s is `silent == 335 * 1000 * NS_PER_MS`), every window of one beat inside it holds a
/// heartbeat: the heartbeats are at multiples of the beat, so no gap between two bytes is longer than
/// one beat, which is below every idle limit.
pub proof fn lemma_silence_is_carried(silent: int, beat: int)
    requires
        0 < beat,
        beat <= silent,
    ensures
        forall|t: int| 0 <= t && t + beat <= silent ==> #[trigger] has_beat_in(t, beat),
{
    assert forall|t: int| 0 <= t && t + beat <= silent implies #[trigger] has_beat_in(t, beat) by {
        let k = t / beat + 1;
        assert(t < k * beat && k * beat <= t + beat) by (nonlinear_arith)
            requires 0 <= t, 0 < beat, k == t / beat + 1;
        assert(beat_at(k, beat) == k * beat);
    }
}

// ---------------------------------------------------------------------------------------------
// W2: the one-shot's first answer.
// ---------------------------------------------------------------------------------------------

/// How a one-shot exchange answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The command ended within the first beat: its own answer, with its own status.
    Direct,
    /// The command was still running at the first beat: `200` headers now, a space every beat, and
    /// the answer (output and exit code, or the error body) as the end of the one document.
    Committed,
}

/// W2. The mode of a one-shot answer, given whether the command ended within the first beat.
pub fn oneshot_mode(ended_within_beat: bool) -> (m: Mode)
    ensures
        ended_within_beat <==> m is Direct,
        !ended_within_beat <==> m is Committed,
{
    if ended_within_beat || cfg!(feature = "mut_oneshot_never_commits") {
        Mode::Direct
    } else {
        Mode::Committed
    }
}

/// The status a one-shot exchange goes out with, given its mode and the status its own answer would
/// have had: a committed answer is `200` whatever its answer's status turns out to be (the error of
/// an output over the cap travels as its body); a direct one is its own.
pub open spec fn wire_status(m: Mode, own: int) -> int {
    match m {
        Mode::Direct => own,
        Mode::Committed => 200,
    }
}

/// **W2.** A committed answer is `200` at its first silence; a direct one keeps its status.
pub proof fn lemma_committed_is_200(own: int)
    ensures
        wire_status(Mode::Committed, own) == 200,
        wire_status(Mode::Direct, own) == own,
{
}

// ---------------------------------------------------------------------------------------------
// W3: a job's processes.
// ---------------------------------------------------------------------------------------------

/// Whether a signal sent to the job's group reaches a process in group `pg`: the job has a real group
/// (above 1: never init's, never the host's) and the process is in it.
pub open spec fn group_reaches_spec(pgid: int, pg: int) -> bool {
    pgid > 1 && pg == pgid
}

/// Whether a process belongs to the job: in its group, or carrying its mark wherever it has gone.
pub open spec fn member_spec(pgid: int, pg: int, marked: bool) -> bool {
    group_reaches_spec(pgid, pg) || marked
}

/// Whether a process is one the group signal does not reach and that carries the mark: the ones a
/// signal is also sent to one by one.
pub open spec fn stray_spec(pgid: int, pg: int, marked: bool) -> bool {
    marked && !group_reaches_spec(pgid, pg)
}

/// Whether a signal to the job's group reaches a process in group `pg`.
pub fn group_reaches(pgid: i32, pg: i32) -> (r: bool)
    ensures
        r == group_reaches_spec(pgid as int, pg as int),
{
    pgid > 1 && pg == pgid
}

/// W3. Whether a live process (in group `pg`, `marked` with the job's mark or not) is one the scan
/// takes for the job: a member, or with `strays_only` just a marked process the group signal does
/// not reach.
pub fn is_member(pgid: i32, pg: i32, marked: bool, strays_only: bool) -> (r: bool)
    ensures
        r == (if strays_only { stray_spec(pgid as int, pg as int, marked) } else { member_spec(pgid as int, pg as int, marked) }),
{
    let group = group_reaches(pgid, pg);
    if strays_only {
        if cfg!(feature = "mut_stray_not_signalled") {
            false
        } else {
            marked && !group
        }
    } else {
        group || marked
    }
}

/// **W3, every process of the job hears the signal.** A member is reached by the group signal or is a
/// stray that is signalled itself: so a process that left the group (`setsid`, a double fork) is
/// signalled as surely as one in it, and a wipe (which signals every job of the scope this way) leaves
/// none of what the jobs started.
pub proof fn lemma_every_member_is_signalled(pgid: int, pg: int, marked: bool)
    ensures
        member_spec(pgid, pg, marked) ==> (group_reaches_spec(pgid, pg) || stray_spec(pgid, pg, marked)),
        stray_spec(pgid, pg, marked) ==> member_spec(pgid, pg, marked),
        stray_spec(pgid, pg, marked) ==> !group_reaches_spec(pgid, pg),
{
}

} // verus!

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const BEAT: u64 = HEARTBEAT_MS * NS_PER_MS;

    /// W1: whatever the time since the last byte, the next goes out within a beat of it (and the one
    /// after a byte just sent, exactly a beat later). Red under `mut_beat_late` and `mut_beat_never`.
    #[test]
    fn a_stream_is_never_silent_past_a_beat() {
        for since in [0, 1, BEAT / 2, BEAT - 1, BEAT, BEAT + 1, 3 * BEAT] {
            let by = since + quiet_wait(since, BEAT);
            assert!(by <= since.max(BEAT), "since {since}: next byte at {by}");
        }
        assert_eq!(quiet_wait(0, BEAT), BEAT);
        assert_eq!(quiet_wait(BEAT, BEAT), 0);
    }

    /// W1: a silent command's whole 335 s has a heartbeat in every beat-long window, so no gap is
    /// longer than a beat. Red under `mut_beat_late` and `mut_beat_never`.
    #[test]
    fn a_335_second_silence_has_no_gap_past_a_beat() {
        let silent = 335_000 * NS_PER_MS;
        let (mut last, mut now, mut worst) = (0u64, 0u64, 0u64);
        while now < silent {
            now += quiet_wait(now - last, BEAT);
            worst = worst.max(now - last);
            last = now;
        }
        assert!(worst <= BEAT, "a gap of {worst} ns");
        assert!(worst < GATEWAY_IDLE_MS * NS_PER_MS);
    }

    /// W1b: 1 to 240 whole seconds is the client's; zero, a negative, an unparsable (`None`) and over
    /// 240 are the default. Red under `mut_interval_unbounded` and `mut_interval_off`.
    #[test]
    fn a_clients_interval_is_taken_from_one_to_240_seconds_and_nothing_else() {
        let d = HEARTBEAT_MS;
        assert_eq!(beat_ms(Some(1), d), 1_000);
        assert_eq!(beat_ms(Some(30), d), 30_000);
        assert_eq!(beat_ms(Some(240), d), 240_000);
        for out in [0, -1, i64::MIN, 241, 295, 1_000, i64::MAX] {
            assert_eq!(beat_ms(Some(out), d), d, "{out} is not an interval");
        }
        assert_eq!(beat_ms(None, d), d, "absent or unparsable");
        assert_eq!(beat_ms(None, 50), 50, "the default is the caller's");
    }

    /// W1b: a stream at the longest interval is never silent past it, and it is under every idle
    /// limit. Red under `mut_interval_unbounded`.
    #[test]
    fn the_longest_interval_is_under_every_idle_limit() {
        let beat = beat_ms(Some(500), HEARTBEAT_MS) * NS_PER_MS;
        assert_eq!(beat, HEARTBEAT_MS * NS_PER_MS);
        let longest = beat_ms(Some(240), HEARTBEAT_MS);
        assert!(longest < GATEWAY_IDLE_MS);
        let beat = longest * NS_PER_MS;
        let (mut last, mut now, mut worst) = (0u64, 0u64, 0u64);
        while now < 335_000 * NS_PER_MS {
            now += quiet_wait(now - last, beat);
            worst = worst.max(now - last);
            last = now;
        }
        assert!(worst <= beat && worst < GATEWAY_IDLE_MS * NS_PER_MS);
    }

    /// W2: a command that has not ended by the first beat is committed. Red under
    /// `mut_oneshot_never_commits`.
    #[test]
    fn a_one_shot_still_running_at_the_first_beat_is_committed() {
        assert_eq!(oneshot_mode(false), Mode::Committed);
        assert_eq!(oneshot_mode(true), Mode::Direct);
    }

    /// W3: a marked process in another group is signalled by the scan for strays; one in the group is
    /// not a stray (the group signal has it). Red under `mut_stray_not_signalled`.
    #[test]
    fn a_marked_process_outside_the_group_is_signalled() {
        assert!(is_member(100, 200, true, true), "a setsid'd child");
        assert!(
            !is_member(100, 100, true, true),
            "in the group: the group signal has it"
        );
        assert!(is_member(100, 100, false, false), "a member by group");
        assert!(is_member(100, 200, true, false), "a member by mark");
        assert!(!is_member(100, 200, false, false), "a stranger");
        assert!(
            is_member(1, 1, true, true),
            "no real group: the mark is all there is"
        );
    }
}
