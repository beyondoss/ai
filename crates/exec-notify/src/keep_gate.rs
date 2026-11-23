//! The keep gate: when a command that is still held by its stream is handed to the sandbox.
//!
//! A held foreground command lives and dies with its stream (`crates/sandbox/src/exec_endpoint.rs`,
//! "Held"), and a replica that is reaped or redeployed closes every stream it holds. The front's
//! drain keeps a closing replica serving for at least [`KEEP_AFTER_MS`] of any stream's life: a
//! stream is closed no earlier than the drain's wait after it began (Beyond's execfront, 90 s), so
//! a command that is `keep`t by then outlives the close, and its client follows it from the
//! offset it has reached (`attach`). Before then nothing is sent: a command that ends inside
//! [`KEEP_AFTER_MS`] costs no extra request.
//!
//! [`keep_due`] is the one decision. It is the same function the agent calls and Verus proves;
//! `mut_keep_early` makes it fire at once and the proof (and the agent's test) go red.

use vstd::prelude::*;

verus! {

/// How long a held foreground command runs before it is kept in the sandbox (30 s). Strictly less
/// than the front's drain wait, so a stream is never cut before its command is kept: Beyond's
/// `opkernel::drain_window` states that rule against this constant.
pub const KEEP_AFTER_MS: u64 = 30_000;

/// Whether the held command, `age_ms` old, is to be kept now. `kept` and `ended` are facts the
/// caller already knows: a kept command is not kept twice, and an ended one is nobody's to keep.
pub open spec fn keep_due_spec(age_ms: u64, kept: bool, ended: bool) -> bool {
    !kept && !ended && age_ms >= KEEP_AFTER_MS
}

pub fn keep_due(age_ms: u64, kept: bool, ended: bool) -> (due: bool)
    ensures
        due == keep_due_spec(age_ms, kept, ended),
        // Never a keep on a command younger than KEEP_AFTER_MS: nothing is added to a short command.
        age_ms < KEEP_AFTER_MS ==> !due,
        // Never twice, never for an ended command.
        kept ==> !due,
        ended ==> !due,
        // Liveness: a live, unkept command that has reached KEEP_AFTER_MS is kept.
        (!kept && !ended && age_ms >= KEEP_AFTER_MS) ==> due,
{
    if cfg!(feature = "mut_keep_early") {
        return !kept && !ended;
    }
    if cfg!(feature = "mut_keep_never") {
        return false;
    }
    !kept && !ended && age_ms >= KEEP_AFTER_MS
}

/// How long to wait from age `age_ms` until the keep is due: the progress measure. It is zero
/// exactly when the keep is due by age, and waiting that long makes it due.
#[allow(clippy::implicit_saturating_sub)]
pub fn keep_wait_ms(age_ms: u64) -> (wait: u64)
    ensures
        age_ms >= KEEP_AFTER_MS ==> wait == 0,
        age_ms < KEEP_AFTER_MS ==> wait == KEEP_AFTER_MS - age_ms,
        age_ms + wait >= KEEP_AFTER_MS || age_ms >= KEEP_AFTER_MS,
{
    if age_ms >= KEEP_AFTER_MS { 0 } else { KEEP_AFTER_MS - age_ms }
}

} // verus!

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_command_is_never_kept() {
        for age in [0, 1, 15_000, KEEP_AFTER_MS - 1] {
            assert!(!keep_due(age, false, false), "age {age}");
        }
    }

    #[test]
    fn a_command_at_the_bound_is_kept_once() {
        assert!(keep_due(KEEP_AFTER_MS, false, false));
        assert!(keep_due(10 * KEEP_AFTER_MS, false, false));
        assert!(!keep_due(KEEP_AFTER_MS, true, false));
        assert!(!keep_due(KEEP_AFTER_MS, false, true));
    }

    #[test]
    fn waiting_the_wait_makes_the_keep_due() {
        for age in [0, 1, 29_999, 30_000, 45_000] {
            assert!(keep_due(age + keep_wait_ms(age), false, false), "age {age}");
        }
    }
}
