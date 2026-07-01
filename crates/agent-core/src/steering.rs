//! Steering — injecting user input into a run in flight.
//!
//! A [`Steering`] handle holds two shared queues, distinguished by *when* the loop injects them:
//!
//! - **steer** ([`push_steer`](Steering::push_steer)) — injected *mid-run*, between tool-executing
//!   turns, folded onto the same tool-results user turn. This is how a client redirects a busy agent
//!   ("actually, also handle X") without waiting for it to stop.
//! - **follow-up** ([`push`](Steering::push)) — injected only at a *would-stop* boundary (the model
//!   ended its turn without asking for tools), as a fresh user turn. This is "keep going / now do the
//!   next thing" once the current work is done.
//!
//! Both injection points place messages where the previous message is the assistant's, so a pushed
//! user turn never lands next to another user turn (which the wire would reject). The two lanes mirror
//! pi's separate `steerQueue` and `followUpQueue`.
//!
//! A third, independent signal lives here too: [`request_stop`](Steering::request_stop) — a graceful,
//! host-initiated "stop after the current turn" request, mirroring pi's `shouldStopAfterTurn`. Unlike
//! cancellation (which drops an in-flight future and can abandon a tool mid-execution), this is checked
//! only at a turn boundary, after that turn's tool calls (if any) have already completed and their
//! results are durably committed — so it never leaves an orphaned `tool_use` behind. It's a flag, not a
//! queue: a second request before the first is observed is indistinguishable from one.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

type Queue = Arc<Mutex<VecDeque<String>>>;

/// A cloneable handle to the shared steering queues. Clones share the same two lanes (and the stop
/// flag).
#[derive(Clone, Default)]
pub struct Steering {
    /// Injected mid-run, between tool turns.
    steer: Queue,
    /// Injected at a would-stop boundary.
    follow_up: Queue,
    /// Set by [`request_stop`](Self::request_stop); consumed by the loop at the next turn boundary.
    stop_requested: Arc<AtomicBool>,
}

impl Steering {
    /// Empty steering queues.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a **follow-up**: injected when the run next reaches a stop boundary.
    pub fn push(&self, message: impl Into<String>) {
        lock(&self.follow_up).push_back(message.into());
    }

    /// Queue a **steer**: injected mid-run, at the next tool-results turn, to redirect a busy agent.
    pub fn push_steer(&self, message: impl Into<String>) {
        lock(&self.steer).push_back(message.into());
    }

    /// Whether either queue currently holds any messages.
    pub fn is_empty(&self) -> bool {
        lock(&self.steer).is_empty() && lock(&self.follow_up).is_empty()
    }

    /// Take the queued mid-run steer messages, leaving that lane empty.
    pub(crate) fn drain_steer(&self) -> Vec<String> {
        lock(&self.steer).drain(..).collect()
    }

    /// Take everything queued for a stop boundary: the follow-up lane, plus any steer messages that
    /// were queued but never reached a mid-run injection point (e.g. a turn with no tool calls), so
    /// nothing is stranded.
    pub(crate) fn drain_at_stop(&self) -> Vec<String> {
        let mut out: Vec<String> = lock(&self.follow_up).drain(..).collect();
        out.extend(lock(&self.steer).drain(..));
        out
    }

    /// Drop everything queued in both lanes without returning it — for a caller (`new_session`/
    /// `switch_session`/`fork`/`switch_branch`) that's about to swap in a different session's
    /// conversation, so a message queued for the *old* session's next turn can't leak into the newly
    /// switched-to one. Also clears any pending stop request, for the same reason: a graceful-stop
    /// request aimed at the old session's run must not cut short a different session's next one.
    pub fn clear(&self) {
        lock(&self.steer).clear();
        lock(&self.follow_up).clear();
        self.stop_requested.store(false, Ordering::Relaxed);
    }

    /// Request that the run stop gracefully at the next turn boundary — after the current turn's tool
    /// calls (if any) finish and their results are committed, but before another model call starts.
    /// Idempotent: a second call before the first is observed has no additional effect.
    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::Relaxed);
    }

    /// Consume and return the pending stop request, if any. Each request is observed at most once.
    pub(crate) fn take_stop_requested(&self) -> bool {
        self.stop_requested.swap(false, Ordering::Relaxed)
    }
}

/// Recover the data on a poisoned lock rather than propagating a panic into the loop.
fn lock(q: &Queue) -> std::sync::MutexGuard<'_, VecDeque<String>> {
    q.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follow_up_and_steer_are_separate_lanes() {
        let s = Steering::new();
        assert!(s.is_empty());
        s.push("follow");
        s.push_steer("steer");
        assert!(!s.is_empty());
        // The steer lane drains on its own; the follow-up stays until the stop boundary.
        assert_eq!(s.drain_steer(), vec!["steer".to_string()]);
        assert_eq!(s.drain_at_stop(), vec!["follow".to_string()]);
        assert!(s.is_empty());
    }

    #[test]
    fn drain_at_stop_sweeps_stranded_steer_messages() {
        // A steer queued on a turn that never ran tools must still be injected at the stop boundary.
        let s = Steering::new();
        s.push_steer("stranded");
        s.push("follow");
        assert_eq!(
            s.drain_at_stop(),
            vec!["follow".to_string(), "stranded".to_string()]
        );
    }

    #[test]
    fn clear_drops_both_lanes_without_returning_them() {
        let s = Steering::new();
        s.push("follow");
        s.push_steer("steer");
        assert!(!s.is_empty());
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.drain_at_stop(), Vec::<String>::new());
    }

    #[test]
    fn clones_share_one_queue() {
        let a = Steering::new();
        let b = a.clone();
        a.push("x");
        assert_eq!(b.drain_at_stop(), vec!["x".to_string()]);
    }

    #[test]
    fn stop_request_is_observed_once() {
        let s = Steering::new();
        assert!(!s.take_stop_requested(), "no request yet");
        s.request_stop();
        assert!(s.take_stop_requested(), "the request must be observed");
        assert!(
            !s.take_stop_requested(),
            "a consumed request must not be observed twice"
        );
    }

    #[test]
    fn a_second_stop_request_before_the_first_is_observed_is_a_no_op() {
        let s = Steering::new();
        s.request_stop();
        s.request_stop();
        assert!(s.take_stop_requested());
        assert!(!s.take_stop_requested());
    }

    #[test]
    fn clear_also_drops_a_pending_stop_request() {
        let s = Steering::new();
        s.request_stop();
        s.clear();
        assert!(
            !s.take_stop_requested(),
            "clear must not leave a stop request that could cut short a different session's run"
        );
    }

    #[test]
    fn stop_request_is_shared_across_clones() {
        let a = Steering::new();
        let b = a.clone();
        a.request_stop();
        assert!(b.take_stop_requested());
    }
}
