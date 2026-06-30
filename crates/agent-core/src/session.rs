//! Per-run agent state.
//!
//! A [`Session`] is the evolving state of one agent run: the full message history plus token/step
//! counters. It's `serde`-serializable so a headless run (`serve`) can persist it and a client can
//! reattach to a running session later — the foundation for the attach-later remote-control model.

use serde::{Deserialize, Serialize};

use crate::message::Message;

/// The state of one agent run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Session {
    /// Full conversation, oldest first.
    pub messages: Vec<Message>,
    /// Completed loop iterations (model turn + tool dispatch).
    pub steps: u32,
    /// Cumulative input tokens reported by the model.
    pub input_tokens: u64,
    /// Cumulative output tokens reported by the model.
    pub output_tokens: u64,
}

impl Session {
    /// A fresh, empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a message to the history.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Seed (or continue) the conversation with a user text turn.
    pub fn user(&mut self, text: impl Into<String>) -> &mut Self {
        self.push(Message::user(text));
        self
    }

    /// Fold a turn's token usage into the running totals.
    pub fn record_usage(&mut self, input_tokens: u32, output_tokens: u32) {
        self.input_tokens += u64::from(input_tokens);
        self.output_tokens += u64::from(output_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let mut s = Session::new();
        s.user("hello");
        s.steps = 2;
        s.record_usage(10, 5);
        s.record_usage(3, 7);
        let json = serde_json::to_string(&s).unwrap();
        let back: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.steps, 2);
        assert_eq!(back.input_tokens, 13);
        assert_eq!(back.output_tokens, 12);
    }
}
