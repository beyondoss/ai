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

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

type Queue = Arc<Mutex<VecDeque<String>>>;

/// A cloneable handle to the shared steering queues. Clones share the same two lanes.
#[derive(Clone, Default)]
pub struct Steering {
    /// Injected mid-run, between tool turns.
    steer: Queue,
    /// Injected at a would-stop boundary.
    follow_up: Queue,
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
    fn clones_share_one_queue() {
        let a = Steering::new();
        let b = a.clone();
        a.push("x");
        assert_eq!(b.drain_at_stop(), vec!["x".to_string()]);
    }
}
