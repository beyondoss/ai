//! Per-run agent state.
//!
//! A [`Session`] is the evolving state of one agent run: the full message history plus token/step
//! counters. It's `serde`-serializable so a headless run (`serve`) can persist it and a client can
//! reattach to a running session later — the foundation for the attach-later remote-control model.

use std::collections::HashMap;
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
    /// Output tokens the provider reported for the turn `last_usage_message_count` was snapshotted at
    /// — i.e. the exact size of `messages[last_usage_message_count]` (that turn's own assistant
    /// message), once appended. `compaction::trailing_tokens` uses this in place of its usual char/4
    /// estimate for that one message specifically: it's the one message in the trailing window whose
    /// real size is already known exactly, from the provider's own usage report, rather than needing
    /// the heuristic every other trailing message must fall back on. Without this, the just-completed
    /// turn's own output was still *counted* (the heuristic estimates it too, just less precisely) —
    /// this is a precision fix for the proactive trigger's timing, not a data-loss one.
    #[serde(default)]
    pub last_output_tokens: u32,
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

    /// Remove the trailing message if (and only if) it's a synthetic [`Message::error`] record —
    /// called by a whole-run auto-retry layer right before re-invoking the same run on the same
    /// session. `run_events_steered` persists this record on every `Err`-ending run (so a client's own
    /// retry via a fresh `prompt` never stacks a second consecutive `user` turn on a truly abandoned
    /// run), but an *automatic* retry re-runs the exact same attempt from scratch — it must be
    /// genuinely invisible in the session's history, "no trace of the failed attempt in loop-visible
    /// state" (mirroring how `agent_core`'s own mid-stream retry layer, one level down, never persists
    /// anything for a failed attempt in the first place). Leaving the record in place here would let
    /// the retry's own real response land right after it, producing two consecutive assistant turns —
    /// or, on final exhaustion, a spurious extra history entry no client asked for.
    pub fn pop_error_record(&mut self) {
        if self
            .messages
            .last()
            .is_some_and(|m| m.error_message.is_some())
        {
            Arc::make_mut(&mut self.messages).pop();
        }
    }

    /// Scrub state that doesn't survive a model switch, ahead of resuming the conversation on
    /// `new_model`. Two things are per-producing-model, not portable:
    ///
    /// - Signed `Thinking` blocks and encrypted `RedactedThinking` blocks — Anthropic can reject a
    ///   later turn that replays a *signed* one to a different model than produced it, and a redacted
    ///   block is opaque ciphertext meaningless to any other model. A signed `Thinking` block's own
    ///   *text*, though, is ordinary prose — pi downgrades it to a plain `Text` block instead of
    ///   discarding it outright, preserving the visible reasoning trace as context for the new model
    ///   rather than silently erasing every prior turn's chain of thought on a model switch. An
    ///   *empty*-text `Thinking` block (nothing worth preserving) is dropped like `RedactedThinking`,
    ///   not downgraded into a useless empty `Text` block.
    /// - Combined OpenAI-Responses tool-call ids (`"call_id|item_id"`) — the `item_id` half only pairs
    ///   with a `reasoning` item on the *same* model/dialect; replayed to a foreign model it's at best
    ///   dead weight, at worst rejected. Truncating a `ToolUse.id` this way would silently break its
    ///   pairing with the matching `ToolResult.tool_use_id` — possibly in a later message — so every
    ///   rewrite is recorded and replayed onto the paired `ToolResult` in a second pass below.
    ///
    /// Applied per-message, gated on `Message::model_id`: a message stamped with `new_model` itself is
    /// untouched (still valid to replay), everything else — including any message with `model_id: None`
    /// (persisted before this field existed, or from a source that never stamped it) — is treated as
    /// foreign and scrubbed. `split_once('|')`-based truncation is a no-op for Anthropic-native tool-use
    /// ids (never contain `|`), so this is safe to run unconditionally regardless of dialect.
    pub fn scrub_cross_model_state(&mut self, new_model: &str) {
        let messages = Arc::make_mut(&mut self.messages);

        // First pass: downgrade/drop thinking blocks and truncate foreign-model tool-call ids,
        // recording old -> new id for every `ToolUse` rewrite.
        let mut id_remap: HashMap<String, String> = HashMap::new();
        for message in messages.iter_mut() {
            if message.model_id.as_deref() == Some(new_model) {
                continue;
            }
            message.content.retain_mut(|block| match block {
                ContentBlock::Thinking { text, .. } if !text.is_empty() => {
                    let text = std::mem::take(text);
                    *block = ContentBlock::text(text);
                    true
                }
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => false,
                ContentBlock::ToolUse { id, .. } => {
                    let new_id = crate::dialect::openai_responses::call_id_only(id);
                    if new_id != *id {
                        id_remap.insert(std::mem::replace(id, new_id.clone()), new_id);
                    }
                    true
                }
                _ => true,
            });
        }

        // Second pass: replay each rewrite onto its paired `ToolResult` — which may live in a later
        // message, so it's out of scope in the loop above. `ToolResult` blocks live on `User`
        // messages, which never carry a `model_id` (see `Message::model_id`), so there's no
        // same-model shortcut to take here: any match against the remap is by definition a pairing
        // that must follow its `ToolUse` to stay intact, regardless of which message it's on.
        if !id_remap.is_empty() {
            for message in messages.iter_mut() {
                for block in &mut message.content {
                    if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                        if let Some(new_id) = id_remap.get(tool_use_id) {
                            new_id.clone_into(tool_use_id);
                        }
                    }
                }
            }
        }
    }

    /// Whether this session holds any state [`scrub_cross_model_state`](Self::scrub_cross_model_state)
    /// would actually change, were the conversation resumed on `new_model`.
    ///
    /// A read-only pre-check, and the reason the scrub can sit unconditionally on the run path
    /// ([`crate::agent::Agent::run_events_steered`]) without costing anything on the overwhelmingly
    /// common same-model turn: the scrub itself calls `Arc::make_mut` on the message vec, which
    /// **deep-clones the whole transcript** whenever the `Arc` is shared (it is — persistence holds a
    /// handle). Paying that on every turn of a long session to almost always change nothing would be a
    /// real regression, so the mutation is gated behind this scan, which allocates nothing.
    ///
    /// Deliberately narrower than "does any message have a foreign `model_id`". Every `User` message has
    /// `model_id: None` and is therefore foreign by the scrub's own rule, so that question is `true` for
    /// essentially every session and would gate nothing. What actually matters is whether a *foreign*
    /// message carries state that doesn't survive the crossing: a `Thinking`/`RedactedThinking` block
    /// (signature or ciphertext bound to the model that produced it) or an OpenAI-Responses combined
    /// `"call_id|item_id"` tool-call id. Nothing else the scrub touches can differ.
    pub fn needs_cross_model_scrub(&self, new_model: &str) -> bool {
        self.messages
            .iter()
            .filter(|m| m.model_id.as_deref() != Some(new_model))
            .flat_map(|m| m.content.iter())
            .any(|block| match block {
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => true,
                ContentBlock::ToolUse { id, .. } => id.contains('|'),
                _ => false,
            })
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
        let live_input = usage
            .input_tokens
            .saturating_add(usage.cache_read_tokens)
            .saturating_add(usage.cache_write_tokens);
        // A genuinely all-zero report means no real `Usage` event ever arrived this turn (the loop's
        // `Aborted`-turn path calls this unconditionally, before checking `stop_reason`, and a stream
        // cancelled before its `Usage` event carries `TokenUsage::default()`) — not a provider telling
        // us the live context just became empty. Holding the last real snapshot instead of zeroing it
        // out keeps `get_state`'s `context_usage` (and the compaction trigger) reporting the session's
        // actual last-known size through an aborted turn, rather than spuriously reporting "no context
        // at all" the instant a client cancels mid-stream. A message pushed since the last real
        // snapshot (including the very turn this call is for) is still accounted for — just via
        // `compaction::trailing_tokens`'s char/4 estimate instead of this exact figure, the same
        // fallback every message ahead of `last_usage_message_count` already uses.
        if live_input > 0 {
            self.last_input_tokens = live_input;
            self.last_usage_message_count = self.messages.len();
            self.last_output_tokens = usage.output_tokens;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_cross_model_state_downgrades_signed_thinking_to_text_and_drops_redacted() {
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
                ContentBlock::text("answer"),
            ])
            .with_model_id("claude-opus-4-8"),
        );
        s.scrub_cross_model_state("gpt-5");
        assert_eq!(s.messages[0].content.len(), 1); // user text untouched
        // The redacted (opaque ciphertext) block is dropped; the signed thinking block's own text
        // survives, downgraded to a plain Text block — visible reasoning context for the new model,
        // not silently erased.
        assert_eq!(
            s.messages[1].content.len(),
            2,
            "{:?}",
            s.messages[1].content
        );
        assert_eq!(
            s.messages[1].content[0],
            ContentBlock::text("let me reason")
        );
        assert_eq!(s.messages[1].content[1], ContentBlock::text("answer"));
    }

    #[test]
    fn scrub_cross_model_state_drops_an_empty_thinking_block_instead_of_downgrading_it() {
        // Nothing worth preserving — must not become a useless empty Text block, which some
        // providers (Anthropic in particular) reject outright as an invalid content block.
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::Thinking {
                text: String::new(),
                signature: String::new(),
            }])
            .with_model_id("claude-opus-4-8"),
        );
        s.scrub_cross_model_state("gpt-5");
        assert!(
            s.messages[0].content.is_empty(),
            "{:?}",
            s.messages[0].content
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
        // scrubbed (downgraded, for a non-empty signed Thinking block — see the two tests above).
        let mut s = Session::new();
        s.push(Message::assistant(vec![ContentBlock::Thinking {
            text: "reasoning".into(),
            signature: "sig".into(),
        }]));
        s.scrub_cross_model_state("gpt-5");
        assert_eq!(s.messages[0].content, vec![ContentBlock::text("reasoning")]);
    }

    #[test]
    fn scrub_cross_model_state_truncates_combined_tool_call_ids_from_a_foreign_model() {
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::tool_use(
                "call_1|fc_1",
                "read",
                serde_json::json!({}),
            )])
            .with_model_id("gpt-5"),
        );
        s.scrub_cross_model_state("claude-opus-4-8");
        let ContentBlock::ToolUse { id, .. } = &s.messages[0].content[0] else {
            panic!("expected a ToolUse block");
        };
        assert_eq!(id, "call_1");
    }

    #[test]
    fn scrub_cross_model_state_keeps_a_tool_use_and_its_later_tool_result_paired() {
        // Realistic multi-message shape: the ToolUse lives on the assistant turn that requested it,
        // its ToolResult on a later user turn (with unrelated turns in between) — truncating the
        // combined id on one side without the other would leave the pairing broken.
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::tool_use(
                "call_1|fc_1",
                "read",
                serde_json::json!({}),
            )])
            .with_model_id("gpt-5"),
        );
        s.push(
            Message::assistant(vec![ContentBlock::text("still working...")]).with_model_id("gpt-5"),
        );
        s.push(Message::tool_result("call_1|fc_1", "file contents", false));

        s.scrub_cross_model_state("claude-opus-4-8");

        let ContentBlock::ToolUse {
            id: tool_use_id, ..
        } = &s.messages[0].content[0]
        else {
            panic!("expected a ToolUse block");
        };
        let ContentBlock::ToolResult {
            tool_use_id: result_id,
            ..
        } = &s.messages[2].content[0]
        else {
            panic!("expected a ToolResult block");
        };
        assert_eq!(tool_use_id, "call_1");
        assert_eq!(result_id, "call_1");
        assert_eq!(
            tool_use_id, result_id,
            "ToolUse/ToolResult pairing must survive the scrub"
        );
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
        // The second (most recent) call's own reported output size, for `trailing_tokens` to use in
        // place of a rough estimate for that turn's assistant message.
        assert_eq!(back.last_output_tokens, 7);
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

    #[test]
    fn record_usage_holds_the_last_real_snapshot_through_a_genuinely_zero_usage_turn() {
        // pi-parity fix: a stream cancelled before its `Usage` event ever arrives (`Agent`'s `Aborted`
        // turn path) calls `record_usage(TokenUsage::default())` unconditionally — previously this
        // zeroed `last_input_tokens`, making `get_state`'s `context_usage` (and the compaction trigger)
        // report "no context at all" the instant a client cancelled mid-stream, even with a real,
        // sizable prior conversation still live in the session.
        let mut s = Session::new();
        s.record_usage(TokenUsage {
            input_tokens: 5_000,
            cache_read_tokens: 200,
            ..Default::default()
        });
        assert_eq!(s.last_input_tokens, 5_200);
        let count_before = s.last_usage_message_count;

        s.push(Message::assistant(vec![ContentBlock::text("")]));
        s.record_usage(TokenUsage::default());

        assert_eq!(
            s.last_input_tokens, 5_200,
            "a zero-usage report must not overwrite the last real live-context snapshot"
        );
        assert_eq!(
            s.last_usage_message_count, count_before,
            "the snapshot's message-count boundary must also hold, not silently advance past the \
             unaccounted-for message that triggered this call"
        );
    }

    /// The gate must fire on exactly the state the scrub would change — and on nothing else, or every
    /// turn of a long same-model session pays a full transcript deep-clone to change nothing.
    #[test]
    fn needs_cross_model_scrub_fires_only_on_state_that_does_not_survive_a_switch() {
        // A same-model transcript with a signed thinking block: replayable as-is, nothing to do.
        let mut s = Session::new();
        s.push(Message {
            model_id: Some("claude-opus-4-8".to_string()),
            ..Message::assistant(vec![ContentBlock::Thinking {
                text: "hmm".into(),
                signature: "sig".into(),
            }])
        });
        assert!(!s.needs_cross_model_scrub("claude-opus-4-8"));
        // …the same block, now facing a different model: bound to the model that produced it.
        assert!(s.needs_cross_model_scrub("gpt-5"));

        // Plain text from a foreign model is perfectly replayable — this must NOT trigger a scrub, or
        // the gate is just "is there any foreign message", which is true of essentially every session
        // (every `User` message has `model_id: None`) and would gate nothing.
        let mut plain = Session::new();
        plain.push(Message::user("hi"));
        plain.push(Message {
            model_id: Some("gpt-5".to_string()),
            ..Message::assistant(vec![ContentBlock::text("hello")])
        });
        assert!(!plain.needs_cross_model_scrub("claude-opus-4-8"));

        // An OpenAI-Responses combined tool id from a foreign model does need truncating.
        let mut combined = Session::new();
        combined.push(Message {
            model_id: Some("gpt-5".to_string()),
            ..Message::assistant(vec![ContentBlock::ToolUse {
                id: "call_1|item_1".into(),
                name: "read".into(),
                input: serde_json::json!({}),
                thought_signature: None,
            }])
        });
        assert!(combined.needs_cross_model_scrub("claude-opus-4-8"));
        // An Anthropic-native id contains no `|`, so a switch away from it is a no-op.
        let mut native = Session::new();
        native.push(Message {
            model_id: Some("claude-opus-4-8".to_string()),
            ..Message::assistant(vec![ContentBlock::ToolUse {
                id: "toolu_abc".into(),
                name: "read".into(),
                input: serde_json::json!({}),
                thought_signature: None,
            }])
        });
        assert!(!native.needs_cross_model_scrub("gpt-5"));
    }
}
