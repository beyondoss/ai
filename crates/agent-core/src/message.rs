//! The dialect-agnostic conversation model.
//!
//! These types are the single internal representation of a conversation. The wire adapters (added
//! with the transport client) map them to and from the OpenAI (`/v1/chat/completions`) and Anthropic
//! (`/v1/messages`) shapes — the gateway relays bytes verbatim, so the harness owns the dialect.
//!
//! The vocabulary leans Anthropic (content blocks, `tool_use`/`tool_result`) because that's the
//! default dialect for Claude; the OpenAI adapter folds its flatter shape into these blocks.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who authored a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// One piece of message content. A message is a sequence of these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text — a prompt fragment or assistant prose. `id`/`phase` are OpenAI Responses-only
    /// replay metadata (`None` for every other dialect and for text authored locally, e.g. a
    /// compaction summary): `id` is the wire message id, restamped verbatim on replay so the API's
    /// reasoning-item pairing keeps working; `phase` labels the block as `"commentary"` (an
    /// intermediate, non-final message) or `"final_answer"` for models with a channel-split output —
    /// OpenAI's own docs say dropping it on replay measurably degrades those models. See
    /// `dialect::openai_responses::push_assistant_content`.
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<String>,
    },
    /// Extended-thinking output. The wire shape is Anthropic's `{"type":"thinking", thinking, signature}`;
    /// the `signature` is load-bearing — Anthropic rejects a later tool turn whose prior thinking block
    /// is missing or unsigned, so the loop must replay the block verbatim (see the dialect's `build_body`).
    Thinking {
        #[serde(rename = "thinking")]
        text: String,
        signature: String,
    },
    /// Encrypted thinking the provider chose not to expose in clear text. Opaque; replayed verbatim so
    /// the model keeps its reasoning continuity across turns.
    RedactedThinking { data: String },
    /// The model's request to invoke a tool. `input` is the (already-complete) JSON arguments
    /// object; during streaming it's assembled from `StreamEvent::InputJsonDelta` fragments.
    ToolUse {
        id: String,
        name: String,
        input: Value,
        /// Gemini-style opaque reasoning-continuity data tied to this specific call (OpenRouter/Gemini
        /// via the OpenAI Chat Completions fallback dialect: wire field `reasoning_details`, one array
        /// entry per tool call, matched back to it by the entry's own `id`). A single serialized JSON
        /// object — this block owns at most the one entry matched to its `id`, not the whole array —
        /// opaque and replayed verbatim under the assistant message's `reasoning_details` array on a
        /// later tool turn, the same "don't interpret it, just echo it back" contract as
        /// `Thinking.signature`. `None` for every dialect but OpenAI Chat Completions (the only one
        /// that parses this field today — see `dialect::openai::Decoder`'s `reasoning_details`
        /// handling) and for a call whose provider never sent one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    /// The result of running a tool, fed back to the model. Carried on a `User` message (Anthropic
    /// convention). `is_error` lets the model see a failure as a value rather than a dead turn.
    /// `images` carries any visual output (a screenshot, `read` on an image); empty for the common
    /// text-only result, and `skip`ped on the wire/disk so existing sessions round-trip unchanged.
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<ImageSource>,
    },
    /// An image attachment (base64). The wire shape is Anthropic's
    /// `{"type":"image","source":{"type":"base64", media_type, data}}`.
    Image { source: ImageSource },
}

impl ContentBlock {
    /// A plain text block with no replay metadata — the common case for every dialect but OpenAI
    /// Responses (see `Text`'s doc comment).
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text {
            text: text.into(),
            id: None,
            phase: None,
        }
    }

    /// A tool-call block with no reasoning-continuity data — the common case for every dialect but
    /// OpenAI Chat Completions (see `ToolUse`'s doc comment).
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
            thought_signature: None,
        }
    }
}

/// A base64-encoded image source, in Anthropic's nested wire shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageSource {
    /// Always `base64` for now.
    #[serde(rename = "type")]
    pub kind: String,
    /// MIME type, e.g. `image/png`.
    pub media_type: String,
    /// Base64-encoded image bytes.
    pub data: String,
}

impl ImageSource {
    /// A base64 image of the given media type.
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            kind: "base64".into(),
            media_type: media_type.into(),
            data: data.into(),
        }
    }
}

/// A single turn in the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    /// The model id that produced this message, when known. Only stamped on assistant turns (see
    /// `Agent::run_events_steered`) — used to scrub state that doesn't survive a model switch (signed
    /// thinking blocks, combined OpenAI-Responses tool-call ids) from messages produced by a
    /// *different* model than the one about to see them. `None` for user/tool-result turns and for any
    /// message persisted before this field existed — both are treated as foreign (always scrubbed) by
    /// [`Session::scrub_cross_model_state`](crate::session::Session::scrub_cross_model_state).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Set on a synthetic assistant turn created by [`Message::error`] — a whole run that ended in
    /// `Err` (a transport failure that exhausted every retry layer, or a compaction/overflow recovery
    /// that itself failed) with no real model response to persist. `None` on every ordinary message.
    /// Matches pi's `errorMessage` field on an `AssistantMessage` with `stopReason: "error"`
    /// (`agent.ts`'s `handleRunFailure`) — without a persisted record here, the session's last message
    /// stays the user's own un-answered prompt, and a client's retry would append a second consecutive
    /// `user` turn no dialect's wire format allows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Set on an assistant turn whose model stream was cancelled mid-flight (`StopReason::Aborted`) —
    /// distinct from [`error_message`](Self::error_message): this is a user-initiated stop with
    /// whatever partial content had already streamed, not a fault with no response at all. Matches
    /// pi's `AssistantMessage.stopReason === "aborted"` (`agent.ts`'s `handleRunFailure`,
    /// `packages/ai/test/abort.test.ts`) — without persisting the partial turn, a cancelled run leaves
    /// no trace in the transcript and, like an unpersisted error, risks a later retry stacking a second
    /// consecutive `user` turn.
    #[serde(default, skip_serializing_if = "is_false")]
    pub aborted: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl Message {
    /// A user turn carrying a single text block.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
            model_id: None,
            error_message: None,
            aborted: false,
        }
    }

    /// An assistant turn from already-assembled content blocks.
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            model_id: None,
            error_message: None,
            aborted: false,
        }
    }

    /// A synthetic assistant turn closing out a run that ended in `Err` with no real model response —
    /// see the [`error_message`](Self::error_message) field doc for why this must be persisted rather
    /// than leaving the user's prompt as the session's dangling last message. An empty text block
    /// (mirroring pi's own `content: [{type:"text", text:""}]` shape) rather than no content at all,
    /// since an assistant turn with a genuinely empty content array is a shape no wire dialect accepts
    /// on a future replay.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(String::new())],
            model_id: None,
            error_message: Some(message.into()),
            aborted: false,
        }
    }

    /// A user turn carrying text plus image attachments. The text block (if any) comes first, then
    /// one `Image` block per source — the multimodal prompt shape both wire dialects expect.
    pub fn user_with_images(text: impl Into<String>, images: Vec<ImageSource>) -> Self {
        let mut content = Vec::with_capacity(images.len() + 1);
        let text = text.into();
        if !text.is_empty() {
            content.push(ContentBlock::text(text));
        }
        for source in images {
            content.push(ContentBlock::Image { source });
        }
        Self {
            role: Role::User,
            content,
            model_id: None,
            error_message: None,
            aborted: false,
        }
    }

    /// A user turn carrying one tool result (how a tool's output is returned to the model).
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error,
                images: Vec::new(),
            }],
            model_id: None,
            error_message: None,
            aborted: false,
        }
    }

    /// A user turn carrying the results of a batch of tool calls — one `ToolResult` block each, on a
    /// single message. The model returns all of a turn's tool results on one `user` turn, and
    /// Anthropic rejects consecutive same-role messages, so a turn that called N tools must fold its N
    /// results into one message rather than emit N separate ones.
    pub fn tool_results(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: blocks,
            model_id: None,
            error_message: None,
            aborted: false,
        }
    }

    /// Stamp `model_id` in place — used right after `Message::assistant` to record the model that just
    /// produced this turn (see `Agent::run_events_steered`).
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Mark this assistant turn as cancelled mid-stream — see the [`aborted`](Self::aborted) field doc.
    pub fn with_aborted(mut self) -> Self {
        self.aborted = true;
        self
    }

    /// Stamp `error_message` on an assistant turn that otherwise carries *real* content — unlike
    /// [`Message::error`] (a bare closing record for a run with no response at all), this is for a
    /// mid-stream transport failure that struck after genuine content had already streamed: pi's
    /// `anthropic-messages.ts`/`openai-completions.ts` keep whatever text/thinking/tool-call blocks
    /// arrived before the failure, tagging the turn with the error rather than discarding them (the
    /// same treatment its `"aborted"` sibling gets from [`with_aborted`](Self::with_aborted)). Not
    /// replayed on the wire either way — `ensure_non_empty_content`/`transform-messages.ts`'s filtering
    /// only cares about the error/aborted markers, not which one is set.
    pub fn with_error(mut self, message: impl Into<String>) -> Self {
        self.error_message = Some(message.into());
        self
    }

    /// The `ToolUse` blocks in this message, if any (what the loop dispatches each step).
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
    }
}

/// A tool advertised to the model: name, description, and a JSON Schema for its input object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's input arguments.
    pub input_schema: Value,
}

/// Why the model ended a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end of the assistant's turn.
    #[default]
    EndTurn,
    /// The model wants to call one or more tools; the loop should dispatch and continue.
    ToolUse,
    /// Hit the output token ceiling.
    MaxTokens,
    /// Hit a configured stop sequence.
    StopSequence,
    /// The model declined to respond (Anthropic `refusal`). A distinct terminal state so a caller can
    /// tell a refusal apart from a normal end-of-turn rather than reading it as success.
    Refusal,
    /// A client cancelled the run mid-stream — never sent by a provider, only synthesized locally by
    /// `Agent::run_turn_once` when `cancel` trips after some content has already streamed. Distinct
    /// from a fault: matches pi's own `stopReason: "aborted"` (see `Message::aborted`'s doc comment).
    Aborted,
    /// Anything else / dialect-specific.
    Other,
}

/// Per-turn token accounting. Carried on [`StreamEvent::Usage`] and folded into the [`Session`]
/// running totals. The cache fields make the prompt-cache breakpoints we stamp *observable*: without
/// them a cache that silently never hits would re-bill the full prefix and we'd have no signal.
///
/// [`Session`]: crate::session::Session
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Uncached input tokens billed at full price.
    pub input_tokens: u32,
    /// Generated output tokens (includes any thinking tokens the provider folds in).
    pub output_tokens: u32,
    /// Input tokens served from the prompt cache (~10% of input price). Anthropic
    /// `cache_read_input_tokens` / OpenAI `prompt_tokens_details.cached_tokens`.
    #[serde(default)]
    pub cache_read_tokens: u32,
    /// Input tokens written to the prompt cache this turn (~125% of input price), across both TTLs.
    /// Anthropic `cache_creation_input_tokens` — the flat sum `cache_write_1h_tokens` is already
    /// included in.
    #[serde(default)]
    pub cache_write_tokens: u32,
    /// Of `cache_write_tokens`, how many used the 1-hour TTL specifically (Anthropic
    /// `cache_creation.ephemeral_1h_input_tokens`) rather than the default 5-minute one. `0` when the
    /// provider doesn't break this out (OpenAI has no TTL concept) or `cache_long` wasn't requested.
    /// Low-stakes granularity (pricing itself is out of scope for this agent — the gateway meters and
    /// a downstream consumer prices), but distinguishes "the long-TTL breakpoint is actually landing"
    /// from "everything's on the default TTL," which the flat sum alone can't.
    #[serde(default)]
    pub cache_write_1h_tokens: u32,
    /// Reasoning/thinking tokens reported separately by the provider (OpenAI
    /// `completion_tokens_details.reasoning_tokens`); 0 when folded into `output_tokens`.
    #[serde(default)]
    pub reasoning_tokens: u32,
}

/// An incremental event from a streaming model response. The dialect adapters normalize both
/// providers' SSE shapes into this sequence; the loop consumes it to assemble assistant messages.
///
/// Every block-scoped variant (everything between `MessageStart`/`MessageStop` that isn't `Usage`)
/// carries an `index`: which content block it belongs to. This is what lets [`crate::agent`]'s
/// `Accumulator` track more than one block open at once — e.g. two tool calls whose argument deltas
/// genuinely interleave on the wire (OpenAI Responses' `output_index`; Anthropic's own
/// `content_block_start.index`) stream live, in whatever order their deltas actually arrive, instead of
/// one being buffered and replayed as a single burst once the other closes. A dialect whose wire never
/// interleaves (Anthropic in practice, OpenAI Chat Completions' non-tool text) simply always emits
/// `index: 0` for the one block it ever has open at a time — the field costs it nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The assistant turn has begun.
    MessageStart,
    /// A chunk of assistant text for block `index`.
    TextDelta { index: usize, text: String },
    /// A chunk of extended-thinking text for block `index`.
    ThinkingDelta { index: usize, text: String },
    /// The cryptographic signature closing block `index`'s thinking block (arrives after its text).
    /// Captured so the block can be replayed verbatim on the next request.
    SignatureDelta { index: usize, signature: String },
    /// A complete redacted (encrypted) thinking block at `index` — self-contained, no deltas follow.
    RedactedThinking { index: usize, data: String },
    /// A tool-call block opened at `index`; `id` and `name` are known before its arguments stream in.
    ToolUseStart {
        index: usize,
        id: String,
        name: String,
    },
    /// A chunk of block `index`'s in-progress tool-call JSON arguments.
    InputJsonDelta { index: usize, partial_json: String },
    /// The provider's own authoritative, complete tool-call arguments for block `index`, *replacing*
    /// (not appending to) whatever `InputJsonDelta` fragments accumulated so far. Some wire shapes
    /// (currently only OpenAI Responses' `function_call_arguments.done` and `output_item.done`'s
    /// `item.arguments`) supply this as a resync point, so a single dropped or duplicated mid-stream
    /// delta — a relay hiccup with no transport-level error, nothing else would ever catch — can't
    /// silently leave the final call corrupted.
    InputJsonFinal { index: usize, full_json: String },
    /// The provider's own authoritative, complete text for block `index`, replacing whatever
    /// `TextDelta` fragments accumulated so far. Same resync purpose as `InputJsonFinal`, for a
    /// text/refusal item. `id`/`phase` are OpenAI Responses' replay metadata for the finished
    /// message item (see `ContentBlock::Text`'s doc comment) — always `None` from every other
    /// dialect, which never emits this event at all.
    TextFinal {
        index: usize,
        text: String,
        id: Option<String>,
        phase: Option<String>,
    },
    /// Block `index` finished (text or tool-call).
    ContentBlockStop { index: usize },
    /// Token accounting. May arrive at end-of-stream (OpenAI) or alongside other events (Anthropic).
    /// A newtype over [`TokenUsage`] so the wire shape stays `{"type":"usage", …fields}` while the
    /// loop, the session, and future fields all share one struct.
    Usage(TokenUsage),
    /// The assistant turn finished, with the reason it stopped.
    MessageStop { stop_reason: StopReason },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_block_tool_use_round_trips() {
        let block = ContentBlock::tool_use("tu_1", "read", json!({ "path": "README.md" }));
        let wire = serde_json::to_value(&block).unwrap();
        // Internally-tagged on `type`, snake_case — the shape the dialect adapters key on.
        assert_eq!(wire["type"], "tool_use");
        assert_eq!(wire["name"], "read");
        // No reasoning-continuity data on the common path — the field is skipped entirely, not
        // serialized as `null`, so an existing on-disk session round-trips unchanged.
        assert!(wire.get("thought_signature").is_none());
        let back: ContentBlock = serde_json::from_value(wire).unwrap();
        assert_eq!(back, block);
    }

    #[test]
    fn content_block_tool_use_thought_signature_round_trips() {
        // OpenAI Chat Completions' Gemini/OpenRouter `reasoning_details` replay path (see
        // `dialect::openai::Decoder`) — opaque, just needs to survive a wire round-trip verbatim.
        let block = ContentBlock::ToolUse {
            id: "tu_1".into(),
            name: "read".into(),
            input: json!({ "path": "README.md" }),
            thought_signature: Some(
                r#"{"type":"reasoning.encrypted","id":"tu_1","data":"enc"}"#.into(),
            ),
        };
        let wire = serde_json::to_value(&block).unwrap();
        assert_eq!(
            wire["thought_signature"],
            r#"{"type":"reasoning.encrypted","id":"tu_1","data":"enc"}"#
        );
        let back: ContentBlock = serde_json::from_value(wire).unwrap();
        assert_eq!(back, block);
    }

    #[test]
    fn tool_result_defaults_is_error_false_when_absent() {
        let wire = json!({ "type": "tool_result", "tool_use_id": "tu_1", "content": "ok" });
        let block: ContentBlock = serde_json::from_value(wire).unwrap();
        assert_eq!(
            block,
            ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "ok".into(),
                is_error: false,
                images: Vec::new(),
            }
        );
    }

    #[test]
    fn tool_uses_extracts_only_tool_calls() {
        let msg = Message::assistant(vec![
            ContentBlock::text("let me look"),
            ContentBlock::tool_use("a", "read", json!({})),
            ContentBlock::tool_use("b", "bash", json!({})),
        ]);
        let calls: Vec<_> = msg.tool_uses().map(|(id, name, _)| (id, name)).collect();
        assert_eq!(calls, vec![("a", "read"), ("b", "bash")]);
    }

    #[test]
    fn stream_event_tag_is_snake_case() {
        let ev = StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: "{\"p\":".into(),
        };
        assert_eq!(
            serde_json::to_value(&ev).unwrap()["type"],
            "input_json_delta"
        );
    }
}
