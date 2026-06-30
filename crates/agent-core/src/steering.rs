//! Steering — injecting follow-up user input into a run in flight.
//!
//! A [`Steering`] handle is a shared queue of messages. A client can `push` to it while the agent is
//! working; at the next point the loop would otherwise stop (the model ended its turn without asking
//! for tools), the loop drains the queue and continues with those messages as new user turns instead
//! of returning. This is how a headless server lets a client say "also do X" or "keep going" without
//! starting a fresh run.
//!
//! Messages are only injected at a would-stop boundary, where the last message is the assistant's —
//! so a pushed user turn never lands next to another user turn (which the wire would reject).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// A cloneable handle to a shared steering queue. Clones share one queue.
#[derive(Clone, Default)]
pub struct Steering {
    queue: Arc<Mutex<VecDeque<String>>>,
}

impl Steering {
    /// An empty steering queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a message to inject when the run next reaches a stop boundary.
    pub fn push(&self, message: impl Into<String>) {
        self.lock().push_back(message.into());
    }

    /// Whether the queue currently holds any messages.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Take all queued messages, leaving the queue empty.
    pub(crate) fn drain(&self) -> Vec<String> {
        self.lock().drain(..).collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<String>> {
        // Recover the data on a poisoned lock rather than propagating a panic into the loop.
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain_round_trip() {
        let s = Steering::new();
        assert!(s.is_empty());
        s.push("one");
        s.push("two");
        assert!(!s.is_empty());
        assert_eq!(s.drain(), vec!["one".to_string(), "two".to_string()]);
        assert!(s.is_empty());
    }

    #[test]
    fn clones_share_one_queue() {
        let a = Steering::new();
        let b = a.clone();
        a.push("x");
        assert_eq!(b.drain(), vec!["x".to_string()]);
    }
}
