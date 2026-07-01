//! Per-run agent state.
//!
//! A [`Session`] is the evolving state of one agent run: the full message history plus token/step
//! counters. It's `serde`-serializable so a headless run (`serve`) can persist it and a client can
//! reattach to a running session later — the foundation for the attach-later remote-control model.

use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::compaction::CompactionProvenance;
use crate::message::{ContentBlock, Message, TokenUsage};

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
    /// Of `cache_write_tokens`, how many used the 1-hour TTL specifically — see
    /// [`TokenUsage::cache_write_1h_tokens`].
    #[serde(default)]
    pub cache_write_1h_tokens: u64,
    /// Cumulative reasoning/thinking tokens billed separately by the provider.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Input tokens of the most recent turn (cached + uncached) — the live context size, which the
    /// cumulative totals above cannot express. The compaction trigger reads this against the model's
    /// context window.
    #[serde(default)]
    pub last_input_tokens: u32,
    /// `messages.len()` at the moment `last_input_tokens` was last set. Messages appended after this
    /// point (the very turn that usage snapshot came from, plus anything since) aren't reflected in
    /// `last_input_tokens` yet — `compaction::trailing_tokens` estimates them separately so the
    /// compaction trigger isn't comparing the window against a stale, undercounted size.
    #[serde(default)]
    pub last_usage_message_count: usize,
    /// What compaction has recorded about this session so far — folded forward across every round
    /// (see [`CompactionProvenance`]), since `apply_summary` physically replaces the summarized
    /// messages and anything not captured here is lost with them. Default (all-empty, `compactions:
    /// 0`) for a session that's never been compacted, so older persisted sessions round-trip unchanged.
    #[serde(default)]
    pub compaction: CompactionProvenance,
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

    /// Scrub state that doesn't survive a model switch, ahead of resuming the conversation on
    /// `new_model`. Two things are per-producing-model, not portable:
    ///
    /// - Signed `Thinking`/`RedactedThinking` blocks — Anthropic can reject a later turn that replays
    ///   one to a different model than produced it.
    /// - Combined OpenAI-Responses tool-call ids (`"call_id|item_id"`) — the `item_id` half only pairs
    ///   with a `reasoning` item on the *same* model/dialect; replayed to a foreign model it's at best
    ///   dead weight, at worst rejected.
    ///
    /// Applied per-message, gated on `Message::model_id`: a message stamped with `new_model` itself is
    /// untouched (still valid to replay), everything else — including any message with `model_id: None`
    /// (persisted before this field existed, or from a source that never stamped it) — is treated as
    /// foreign and scrubbed. `split_once('|')`-based truncation is a no-op for Anthropic-native tool-use
    /// ids (never contain `|`), so this is safe to run unconditionally regardless of dialect.
    pub fn scrub_cross_model_state(&mut self, new_model: &str) {
        let messages = Arc::make_mut(&mut self.messages);
        for message in messages.iter_mut() {
            if message.model_id.as_deref() == Some(new_model) {
                continue;
            }
            message.content.retain_mut(|block| match block {
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => false,
                ContentBlock::ToolUse { id, .. } => {
                    *id = crate::dialect::openai_responses::call_id_only(id);
                    true
                }
                _ => true,
            });
        }
    }

    /// Fold a turn's token usage into the running totals and record the live context size.
    pub fn record_usage(&mut self, usage: TokenUsage) {
        self.input_tokens += u64::from(usage.input_tokens);
        self.output_tokens += u64::from(usage.output_tokens);
        self.cache_read_tokens += u64::from(usage.cache_read_tokens);
        self.cache_write_tokens += u64::from(usage.cache_write_tokens);
        self.cache_write_1h_tokens += u64::from(usage.cache_write_1h_tokens);
        self.reasoning_tokens += u64::from(usage.reasoning_tokens);
        // The whole prompt the model just saw = uncached input + everything served from / written to
        // cache. This is the current context size (the cumulative sums above can't express it) and is
        // what the compaction trigger compares against the model's context window. `saturating_add`,
        // not `+`: these three fields come straight from parsed model-API usage data with no
        // upper-bound validation (a non-standard proxy could report anomalously large values), and
        // `overflow-checks = true` in the release profile turns a raw `u32` overflow into a panic.
        self.last_input_tokens = usage
            .input_tokens
            .saturating_add(usage.cache_read_tokens)
            .saturating_add(usage.cache_write_tokens);
        self.last_usage_message_count = self.messages.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_cross_model_state_drops_thinking_blocks_from_a_foreign_model() {
        let mut s = Session::new();
        s.push(Message::user("think then answer"));
        s.push(
            Message::assistant(vec![
                ContentBlock::Thinking {
                    text: "let me reason".into(),
                    signature: "sig-abc".into(),
                },
                ContentBlock::RedactedThinking {
                    data: "opaque".into(),
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
            ])
            .with_model_id("claude-opus-4-8"),
        );
        s.scrub_cross_model_state("gpt-5");
        assert_eq!(s.messages[0].content.len(), 1); // user text untouched
        assert_eq!(s.messages[1].content.len(), 1); // only the Text block survives
        assert_eq!(
            s.messages[1].content[0],
            ContentBlock::Text {
                text: "answer".into()
            }
        );
    }

    #[test]
    fn scrub_cross_model_state_leaves_a_message_stamped_with_the_new_model_untouched() {
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::Thinking {
                text: "reasoning".into(),
                signature: "sig".into(),
            }])
            .with_model_id("gpt-5"),
        );
        s.scrub_cross_model_state("gpt-5");
        assert_eq!(s.messages[0].content.len(), 1); // thinking block survives — same model
    }

    #[test]
    fn scrub_cross_model_state_treats_a_message_with_no_model_id_as_foreign() {
        // Persisted before this field existed (or from a source that never stamped it) — always
        // scrubbed, matching the conservative default `strip_thinking_blocks` used to apply to all.
        let mut s = Session::new();
        s.push(Message::assistant(vec![ContentBlock::Thinking {
            text: "reasoning".into(),
            signature: "sig".into(),
        }]));
        s.scrub_cross_model_state("gpt-5");
        assert!(s.messages[0].content.is_empty());
    }

    #[test]
    fn scrub_cross_model_state_truncates_combined_tool_call_ids_from_a_foreign_model() {
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "call_1|fc_1".into(),
                name: "read".into(),
                input: serde_json::json!({}),
            }])
            .with_model_id("gpt-5"),
        );
        s.scrub_cross_model_state("claude-opus-4-8");
        let ContentBlock::ToolUse { id, .. } = &s.messages[0].content[0] else {
            panic!("expected a ToolUse block");
        };
        assert_eq!(id, "call_1");
    }

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
            ..Default::default()
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

    #[test]
    fn record_usage_saturates_instead_of_panicking_on_u32_overflow() {
        // A non-standard proxy/gateway reporting anomalously large usage fields must not crash a
        // long-running `serve` session — `overflow-checks = true` in the release profile turns a raw
        // `u32 + u32 + u32` overflow into a hard panic.
        let mut s = Session::new();
        s.record_usage(TokenUsage {
            input_tokens: u32::MAX - 10,
            cache_read_tokens: 100,
            cache_write_tokens: 100,
            ..Default::default()
        });
        assert_eq!(s.last_input_tokens, u32::MAX);
    }
}
