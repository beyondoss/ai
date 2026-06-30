//! Per-run agent state.
//!
//! A [`Session`] is the evolving state of one agent run: the full message history plus token/step
//! counters. It's `serde`-serializable so a headless run (`serve`) can persist it and a client can
//! reattach to a running session later — the foundation for the attach-later remote-control model.

use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::message::{Message, TokenUsage};

/// The state of one agent run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Session {
    /// Full conversation, oldest first. Held behind an `Arc` so each model request shares the history
    /// by cloning a pointer instead of deep-copying every message every turn (the history grows each
    /// step, so a per-turn deep copy is quadratic over a run). [`Session::push`] mutates via
    /// `Arc::make_mut` — copy-on-write, so it appends in place once the prior turn's request (the only
    /// other holder of the pointer) has been dropped. Serialized as a plain message array, so the
    /// shared representation is invisible on the wire and persisted sessions are byte-identical.
    #[serde(
        serialize_with = "serialize_messages",
        deserialize_with = "deserialize_messages"
    )]
    pub messages: Arc<Vec<Message>>,
    /// Completed loop iterations (model turn + tool dispatch).
    pub steps: u32,
    /// Cumulative uncached input tokens reported by the model.
    pub input_tokens: u64,
    /// Cumulative output tokens reported by the model.
    pub output_tokens: u64,
    /// Cumulative input tokens served from the prompt cache. The signal that the cache breakpoints
    /// the Anthropic dialect stamps are actually hitting.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Cumulative input tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
    /// Cumulative reasoning/thinking tokens billed separately by the provider.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Input tokens of the most recent turn (cached + uncached) — the live context size, which the
    /// cumulative totals above cannot express. The compaction trigger reads this against the model's
    /// context window.
    #[serde(default)]
    pub last_input_tokens: u32,
}

/// Serialize the shared history as a plain `[Message]` array — the `Arc`/`Vec` wrapper is an
/// in-memory sharing detail, not part of the persisted shape.
fn serialize_messages<S>(messages: &Arc<Vec<Message>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    messages.as_slice().serialize(serializer)
}

/// Load a message array back into a freshly-owned `Arc<Vec<Message>>`.
fn deserialize_messages<'de, D>(deserializer: D) -> Result<Arc<Vec<Message>>, D::Error>
where
    D: Deserializer<'de>,
{
    Vec::<Message>::deserialize(deserializer).map(Arc::new)
}

impl Session {
    /// A fresh, empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a message to the history. Copy-on-write via `Arc::make_mut`: in place when the history
    /// is solely owned here (the steady state between turns), cloning only if a request still holds
    /// the prior snapshot.
    pub fn push(&mut self, message: Message) {
        Arc::make_mut(&mut self.messages).push(message);
    }

    /// Seed (or continue) the conversation with a user text turn.
    pub fn user(&mut self, text: impl Into<String>) -> &mut Self {
        self.push(Message::user(text));
        self
    }

    /// Fold a turn's token usage into the running totals and record the live context size.
    pub fn record_usage(&mut self, usage: TokenUsage) {
        self.input_tokens += u64::from(usage.input_tokens);
        self.output_tokens += u64::from(usage.output_tokens);
        self.cache_read_tokens += u64::from(usage.cache_read_tokens);
        self.cache_write_tokens += u64::from(usage.cache_write_tokens);
        self.reasoning_tokens += u64::from(usage.reasoning_tokens);
        // The whole prompt the model just saw = uncached input + everything served from / written to
        // cache. This is the current context size (the cumulative sums above can't express it) and is
        // what the compaction trigger compares against the model's context window.
        self.last_input_tokens =
            usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens;
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
        s.record_usage(TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_tokens: 100,
            cache_write_tokens: 20,
            reasoning_tokens: 4,
        });
        s.record_usage(TokenUsage {
            input_tokens: 3,
            output_tokens: 7,
            cache_read_tokens: 200,
            ..Default::default()
        });
        let json = serde_json::to_string(&s).unwrap();
        let back: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.steps, 2);
        assert_eq!(back.input_tokens, 13);
        assert_eq!(back.output_tokens, 12);
        assert_eq!(back.cache_read_tokens, 300);
        assert_eq!(back.cache_write_tokens, 20);
        assert_eq!(back.reasoning_tokens, 4);
        // Live context size = last turn's input + cache read + cache write (3 + 200 + 0).
        assert_eq!(back.last_input_tokens, 203);
    }
}
