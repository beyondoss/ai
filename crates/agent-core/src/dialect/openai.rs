//! OpenAI Chat Completions wire (`/v1/chat/completions`).
//!
//! OpenAI's shape is flatter than the internal model, so both directions need real translation:
//! - **Request:** the system prompt becomes a `system` message; an assistant turn's `ToolUse` blocks
//!   become `tool_calls` (arguments as a JSON *string*); a user turn's `ToolResult` blocks fan out
//!   into separate `role:"tool"` messages.
//! - **Stream:** `tool_calls` arrive as index-keyed deltas (id+name once, then `arguments` fragments)
//!   with no explicit block-stop events, so the decoder synthesizes `ContentBlockStop` and defers
//!   `MessageStop` to end-of-stream (after the trailing `usage` chunk).

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::StreamDecoder;
use crate::error::{Error, Result};
use crate::message::{
    ContentBlock, ImageSource, Message, Role, StopReason, StreamEvent, TokenUsage,
};
use crate::transport::{ModelRequest, ToolChoice};

/// This assistant turn's reasoning text and the wire field name to replay it under, if it carried a
/// `Thinking` block with a non-empty `signature` (the decoder only ever produces one — see
/// `Decoder::reasoning_field`; multiple blocks would be joined, matching the reference agent). `None`
/// when there's no thinking to replay, or its signature is empty (never captured with a field to
/// replay it on, so nowhere safe to put it).
fn assistant_reasoning(content: &[ContentBlock]) -> Option<(&str, String)> {
    let mut field: Option<&str> = None;
    let mut text = String::new();
    for b in content {
        if let ContentBlock::Thinking { text: t, signature } = b {
            if field.is_none() && !signature.is_empty() {
                field = Some(signature.as_str());
            }
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(t);
        }
    }
    field.map(|f| (f, text))
}

/// OpenAI's `prompt_cache_key` cap — a key longer than this is rejected by the API. Shared with the
/// Responses dialect (`super::openai_responses`), which has the identical limit.
pub(super) const PROMPT_CACHE_KEY_MAX_LEN: usize = 64;

/// Clamp a prompt-cache key to OpenAI's length limit, truncating on a char boundary (not a byte one —
/// a session id is expected to be ASCII, but this stays correct if it ever isn't). Shared with the
/// Responses dialect.
pub(super) fn clamp_prompt_cache_key(key: &str) -> &str {
    match key.char_indices().nth(PROMPT_CACHE_KEY_MAX_LEN) {
        Some((byte_idx, _)) => &key[..byte_idx],
        None => key,
    }
}

/// This assistant turn's OpenAI Chat Completions `reasoning_details` replay array — one entry per
/// `ToolUse` block that carries `thought_signature` (Gemini-style opaque reasoning-continuity data;
/// see `ContentBlock::ToolUse`'s doc comment), in content order. `None` when no tool call in this turn
/// carries one, or every entry present fails to parse back as JSON — matches pi's own
/// `.filter(Boolean)` after `JSON.parse` (`openai-completions.ts`): a corrupted signature is silently
/// dropped from the replay rather than sent malformed.
fn assistant_reasoning_details(content: &[ContentBlock]) -> Option<Vec<Value>> {
    let details: Vec<Value> = content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::ToolUse {
                thought_signature: Some(sig),
                ..
            } => serde_json::from_str::<Value>(sig).ok(),
            _ => None,
        })
        .collect();
    (!details.is_empty()).then_some(details)
}

fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Placeholder text substituted for an image sent to a model that doesn't accept one — matching pi's
/// `transform-messages.ts` (`downgradeUnsupportedImages`); the Anthropic dialect uses the identical
/// string for the same purpose.
const USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
/// Same idea, for a tool result's image output specifically.
const TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

/// Build an OpenAI user-message `content` from a turn's text and image blocks (tool results are
/// emitted separately). Returns a plain string when there are no images — OpenAI's common case — and
/// a multimodal parts array (`[{type:"text"}, {type:"image_url"}…]`) when images are present and
/// `supports_vision` is `true`. `None` if the turn carries neither text nor image (e.g. a
/// tool-result-only turn). Without this, image blocks were dropped on the floor: `text_of` keeps only
/// text, so vision input silently vanished. When the model can't accept images, the image is instead
/// replaced with a placeholder note (rather than sent and rejected, or silently dropped).
fn user_content(blocks: &[ContentBlock], supports_vision: bool) -> Option<Value> {
    let has_image = blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    if !has_image {
        let text = text_of(blocks);
        return (!text.is_empty()).then_some(Value::String(text));
    }
    if !supports_vision {
        // pi-parity fix: this used to gate on the turn-wide `has_image` bool alone, so it joined
        // *every* text block via `text_of` and appended a single trailing placeholder regardless of
        // how many images were present or where — two images separated by text collapsed into one
        // placeholder with the text ordering lost, instead of one placeholder per genuinely
        // consecutive run of images, in place. Walking `blocks` in order and only flushing a
        // placeholder when a run of `Image` blocks ends (on the next non-image block or end-of-turn)
        // matches the Anthropic dialect's `downgrade_unsupported_images` state machine.
        let mut segments: Vec<&str> = Vec::new();
        let mut pending_placeholder = false;
        for b in blocks {
            match b {
                ContentBlock::Text { text, .. } if !text.is_empty() => {
                    if pending_placeholder {
                        segments.push(USER_IMAGE_PLACEHOLDER);
                        pending_placeholder = false;
                    }
                    segments.push(text);
                }
                ContentBlock::Image { .. } => pending_placeholder = true,
                _ => {}
            }
        }
        if pending_placeholder {
            segments.push(USER_IMAGE_PLACEHOLDER);
        }
        return (!segments.is_empty()).then(|| Value::String(segments.join("\n")));
    }
    let mut parts: Vec<Value> = Vec::new();
    for b in blocks {
        match b {
            ContentBlock::Text { text, .. } if !text.is_empty() => {
                parts.push(json!({ "type": "text", "text": text }));
            }
            ContentBlock::Image { source } => parts.push(json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", source.media_type, source.data),
                },
            })),
            _ => {}
        }
    }
    (!parts.is_empty()).then_some(Value::Array(parts))
}

/// True if any earlier turn in the conversation already carries a tool call or its result — an
/// assistant `ToolUse` block or a `ToolResult` block (always on a `User` turn in this crate's
/// internal shape). Matches pi's `hasToolHistory` (`openai-completions.ts`), which drives the same
/// `tools: []` fallback below.
fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|m| {
        m.content.iter().any(|b| {
            matches!(
                b,
                ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. }
            )
        })
    })
}

/// Build the streaming request body, translating the internal messages into OpenAI's flat shape.
pub fn build_body(req: &ModelRequest) -> Value {
    let caps = crate::models::capabilities(&req.model);
    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = &req.system {
        messages.push(json!({ "role": "system", "content": system }));
    }

    for m in req.messages.iter() {
        match m.role {
            Role::System => {
                messages.push(json!({ "role": "system", "content": text_of(&m.content) }))
            }
            Role::User => {
                // Text + image blocks form the user message (a multimodal parts array when any image
                // is present); tool results fan out into individual `role:"tool"` messages below.
                if let Some(content) = user_content(&m.content, caps.supports_vision) {
                    messages.push(json!({ "role": "user", "content": content }));
                }
                // Every tool result in this turn is emitted first, contiguously (`tool`, `tool`, …).
                // OpenAI's `tool` role can't carry images, so any visual output is collected here and
                // fanned out to a single trailing `user` message *after* the whole run — not
                // interleaved between individual `tool` messages.
                //
                // pi-parity fix: this used to push a `user` image-carrier message immediately after
                // each `ToolResult` that had one, so two consecutive image-carrying tool results
                // produced `tool, user, tool, user` instead of `tool, tool, user`. A backend that
                // strictly validates a tool-response run is immediately followed by the next real turn
                // (nothing else wedged into the middle of it) can 400 on that. Matches pi's
                // `convertMessages` (`openai-completions.ts`), which batches a consecutive
                // tool-result run's images into one following message the same way.
                let mut pending_images: Vec<(&str, &ImageSource)> = Vec::new();
                for b in &m.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        images,
                        ..
                    } = b
                    {
                        messages.push(json!({ "role": "tool", "tool_call_id": tool_use_id, "content": content }));
                        pending_images.extend(images.iter().map(|src| (tool_use_id.as_str(), src)));
                    }
                }
                if !pending_images.is_empty() {
                    if !caps.supports_vision {
                        // The model can't accept images at all — a plain text placeholder per tool
                        // call, one line each, folded into a single trailing message.
                        let text = pending_images
                            .iter()
                            .map(|(id, _)| {
                                format!(
                                    "Image output from tool call {id}: {TOOL_IMAGE_PLACEHOLDER}"
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        messages.push(json!({ "role": "user", "content": text }));
                    } else {
                        let mut parts: Vec<Value> = Vec::new();
                        let mut last_id: Option<&str> = None;
                        for (id, source) in &pending_images {
                            // A fresh "Image output from tool call {id}:" label each time the
                            // originating call changes, so multiple tool results' images stay
                            // individually attributed within the one combined message.
                            if last_id != Some(*id) {
                                parts.push(json!({
                                    "type": "text",
                                    "text": format!("Image output from tool call {id}:"),
                                }));
                                last_id = Some(*id);
                            }
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{};base64,{}", source.media_type, source.data),
                                },
                            }));
                        }
                        messages.push(json!({ "role": "user", "content": parts }));
                    }
                }
            }
            Role::Assistant => {
                let text = text_of(&m.content);
                let tool_calls: Vec<Value> = m
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, name, input, .. } => Some(json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                            },
                        })),
                        _ => None,
                    })
                    .collect();
                let mut msg = Map::new();
                msg.insert("role".into(), json!("assistant"));
                msg.insert(
                    "content".into(),
                    if text.is_empty() {
                        Value::Null
                    } else {
                        json!(text)
                    },
                );
                if !tool_calls.is_empty() {
                    msg.insert("tool_calls".into(), Value::Array(tool_calls));
                }
                // Replay a prior reasoning turn under the exact field name the decoder captured it
                // from (see `Decoder::reasoning_field`) — different endpoints only accept it back on
                // that one field.
                if let Some((field, text)) = assistant_reasoning(&m.content) {
                    msg.insert(field.to_string(), json!(text));
                }
                // Replay each tool call's Gemini-style opaque reasoning-continuity data (see
                // `ContentBlock::ToolUse`'s doc comment and `Decoder`'s `reasoning_details` handling)
                // under the wire's own array shape — one entry per call that carries one, in content
                // order, matching pi's own replay (`openai-completions.ts`'s `reasoning_details`
                // assembly).
                if let Some(details) = assistant_reasoning_details(&m.content) {
                    msg.insert("reasoning_details".into(), Value::Array(details));
                }
                messages.push(Value::Object(msg));
            }
        }
    }

    let mut map = Map::new();
    map.insert("model".into(), json!(req.model));
    // OpenAI reasoning models (o-series, gpt-5) reject `max_tokens` and require
    // `max_completion_tokens`; non-reasoning chat models take `max_tokens`. The gateway forwards the
    // body verbatim, so the agent must pick the right field per model.
    let max_tokens_field = match caps.max_tokens_field {
        crate::models::MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
        crate::models::MaxTokensField::MaxTokens => "max_tokens",
    };
    // Clamped so the estimated prompt plus this ceiling doesn't already exceed the model's context
    // window — see `super::clamp_max_tokens_to_context`'s doc comment for why this can't be skipped.
    map.insert(
        max_tokens_field.into(),
        json!(super::clamp_max_tokens_to_context(req, &caps)),
    );
    // Reasoning models are driven by `reasoning_effort` (minimal/low/medium/high/xhigh) rather than a
    // thinking-token budget; emit it when the model takes one and the caller set a level. When no
    // level is set but the model can be told explicitly to turn reasoning off, send that instead of
    // omitting the field — matching `dialect::openai_responses`'s `{"effort":"none"}` (this dialect's
    // field is flat, so the equivalent here is `reasoning_effort: "none"`) rather than silently
    // relying on whatever the provider's own undocumented default does when the field is absent.
    if caps.reasoning_effort {
        if let Some(effort) = req.reasoning_effort {
            let effort = crate::models::clamp_reasoning_effort(&caps, effort);
            map.insert("reasoning_effort".into(), json!(effort.as_str()));
        } else if caps.reasoning_disableable {
            map.insert("reasoning_effort".into(), json!("none"));
        }
    }
    map.insert("stream".into(), json!(true));
    // Ask for a trailing usage chunk so token accounting works on the streaming path.
    map.insert("stream_options".into(), json!({ "include_usage": true }));
    // Sent unconditionally when set — matches pi's own unconditional `openai-completions.ts` (a
    // reasoning model that rejects a custom temperature is a caller error, same as pi's).
    if let Some(temperature) = req.temperature {
        map.insert("temperature".into(), json!(temperature));
    }
    // Prompt-cache affinity: OpenAI routes automatic prefix-cache hits by `prompt_cache_key`, so a
    // stable per-conversation key keeps a session pinned to a warm cache node. (OpenAI caches prefixes
    // automatically — there are no explicit breakpoints to set, only this routing hint.) The key has a
    // hard length cap; a caller-supplied key over it would otherwise 400 the request.
    //
    // Gated on `caps.supports_long_cache` (the same per-model capability flag the 24h-retention opt-in
    // just below reuses) rather than sent unconditionally: this dialect is the fallback for every
    // third-party OpenAI-compatible provider the gateway routes to (a native OpenAI id speaks the
    // Responses API instead — see `dialect::Dialect::for_model`), and a strict-schema endpoint with no
    // concept of OpenAI's own caching extensions can 400 the whole request over an unrecognized field.
    // `ModelCaps::unknown()`'s conservative default (`supports_long_cache: false`) means an
    // unrecognized third-party model id omits the field unless a future capability-table entry opts it
    // in explicitly.
    if caps.supports_long_cache {
        if let Some(key) = &req.cache_key {
            map.insert(
                "prompt_cache_key".into(),
                json!(clamp_prompt_cache_key(key)),
            );
        }
    }
    // Opt into the 24h cache-retention tier (vs the default, shorter one) when the caller asked and the
    // model's capability entry allows it — mirrors the Anthropic dialect's `cache_long` gating.
    if req.cache_long && caps.supports_long_cache {
        map.insert("prompt_cache_retention".into(), json!("24h"));
    }
    map.insert("messages".into(), Value::Array(messages));
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({
                "type": "function",
                "function": { "name": t.name, "description": t.description, "parameters": t.input_schema },
            }))
            .collect();
        map.insert("tools".into(), Value::Array(tools));
    } else if has_tool_history(&req.messages) {
        // pi-parity fix: no tools are registered *this* turn (e.g. the caller toggled tools off
        // mid-session, or ran `--no-tools` on a follow-up), but the conversation already contains
        // `tool_calls`/`tool` messages from an earlier one. Anthropic (reached through an
        // OpenAI-compatible proxy such as LiteLLM) 400s a request whose message history has tool
        // calls but which omits `tools` entirely — it wants to see *some* `tools` value, even an
        // empty array, once the conversation has touched them at all. Omitting the field is still
        // correct for the common case (no tool history yet — the `tools:[]`-rejects-immediately
        // DashScope/Qwen case this same omission exists for, see the empty-tools test below);
        // matches pi's `hasToolHistory` fallback (`openai-completions.ts`).
        map.insert("tools".into(), Value::Array(Vec::new()));
    }
    // Constrain tool use only when the caller asked: an unset `tool_choice` emits nothing, leaving
    // OpenAI's default (auto when tools are present), so the common request shape is untouched.
    if let Some(choice) = &req.tool_choice {
        map.insert("tool_choice".into(), tool_choice(choice));
    }
    Value::Object(map)
}

/// Map a [`ToolChoice`] to OpenAI's `tool_choice`. The auto/none/required cases are bare strings; a
/// specific tool is the nested `{type:"function", function:{name}}` object.
fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({ "type": "function", "function": { "name": name } }),
    }
}

fn map_finish_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        // A safety-filtered completion is not a clean stop — collapsing it into `Other` (which the
        // loop treats identically to `EndTurn`) would report a filtered response as if the turn
        // succeeded normally. `Refusal` is this crate's distinct terminal state for "the model didn't
        // actually finish answering, don't read this as success" (see the Anthropic dialect's own
        // `stop_reason: "refusal"`/`"sensitive"` handling). `network_error` is handled separately, in
        // `push`/`finish` below, not here — unlike a genuine refusal it's a transient transport fault,
        // not a terminal answer, so it needs to reach `run_turn`'s retry classifier as an `Err`
        // instead of a successful (if unusable) `Turn`.
        Some("content_filter") => StopReason::Refusal,
        // Never actually surfaced — `finish()` returns an `Err` before this value would be read — but
        // still mapped explicitly (rather than falling into the `Some(other)` catch-all) so a
        // `network_error` finish doesn't spuriously log as an "unrecognized finish_reason".
        Some("network_error") => StopReason::Other,
        Some(other) => {
            // A genuinely unrecognized value (a new finish_reason we don't know about yet) silently
            // collapsing into `Other` — which the loop treats identically to a normal `EndTurn` —
            // would hide a real change in provider behavior. `warn!` so it's at least visible,
            // matching the Anthropic and OpenAI Responses dialects' equivalent handling.
            tracing::warn!(
                finish_reason = other,
                "unrecognized OpenAI finish_reason; treating as Other"
            );
            StopReason::Other
        }
        None => StopReason::Other,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Open {
    None,
    Text,
    Thinking,
}

/// A tool-call block currently accumulating arguments. Tracked independently per call — see
/// `Decoder::open_tools` — so a provider interleaving several parallel calls' `arguments` deltas
/// (rather than completing one before starting the next) can't corrupt one call's buffer with
/// another's fragments.
struct OpenTool {
    /// The `tool_calls[].index` this call opened with, if the provider sent one. Later deltas are
    /// routed back to this call by matching this field first (see `push`'s tool-call arm) — the
    /// stable correlator OpenAI's own docs specify for a parallel-tool-call stream. `None` when the
    /// provider omitted it (only expected for a single, non-parallel call).
    wire_index: Option<i64>,
    /// This call's `id`, captured once from the first chunk that carried a non-empty one and never
    /// overwritten afterward. Some providers (Kimi/Moonshot, observed) resend a *different* `id` on
    /// later fragments of the SAME call; matching pi's `ensureToolCallBlock`
    /// (`packages/ai/src/api/openai-completions.ts`), which only ever sets `block.id` while it's
    /// still empty, this decoder keeps routing by `wire_index` and keeps the first id rather than
    /// treating the changed id as a new call (which used to fragment one real call into two or three
    /// corrupted ones).
    id: String,
    /// The `StreamEvent`/`ContentBlockStop` index this call was assigned, in the order it was first
    /// seen — NOT the wire index (which can be sparse, provider-assigned, or absent). Mirrors pi's
    /// `getContentIndex` (the call's position in `output.content`); see
    /// `Decoder::next_tool_index`.
    stream_index: usize,
}

/// The delta field names different OpenAI-compatible endpoints use for reasoning content:
/// `reasoning_content` (llama.cpp and most providers), `reasoning` (a few others), `reasoning_text`
/// (a third convention). Checked in this order; first-seen field for a stream wins even if a provider
/// later echoes the same text on a second field (observed on some gateways) — using both would double
/// the visible reasoning.
const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

/// Whether a `reasoning_details[]` entry is the one shape this decoder understands — OpenRouter's
/// encrypted-reasoning replay token for Gemini's `thoughtSignature`. Matches pi's
/// `isEncryptedReasoningDetail` (`openai-completions.ts`); any other `type`, or a missing/empty `id`
/// or `data`, is ignored rather than guessed at.
fn is_encrypted_reasoning_detail(detail: &Value) -> bool {
    detail.get("type").and_then(Value::as_str) == Some("reasoning.encrypted")
        && detail
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
        && detail
            .get("data")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
}

/// Decodes OpenAI SSE. Synthesizes the block-stop boundaries OpenAI omits and holds `MessageStop`
/// until `finish()` so it lands after the trailing usage chunk. Text/thinking (`self.open`) are
/// mutually exclusive with each other and always close before the first tool call opens — Chat
/// Completions never resumes lead-in text once tool calls start — so both keep reusing `index: 0`,
/// same as Anthropic reusing an index once `Accumulator` has fully flushed and cleared it.
///
/// Tool calls are different: a provider can have several open *concurrently*, interleaving each
/// call's `arguments` deltas by wire `index` rather than completing one before starting the next
/// (pi-parity fix — this decoder used to track only one open tool call at a time, silently
/// corrupting whichever call wasn't the "current" one when its fragments arrived). Each tool call is
/// tracked independently in `open_tools` and closed together only at `finish_reason`, since OpenAI
/// has no per-call stop event on the wire.
pub struct Decoder {
    started: bool,
    saw_finish: bool,
    open: Open,
    stop_reason: StopReason,
    /// Set when this stream's `finish_reason` was `"network_error"` — a transient transport fault
    /// disguised as a normal completion, not a real answer. Checked in `finish()`, which returns an
    /// `Err` instead of the usual `MessageStop` when set, so the turn reaches `run_turn`'s mid-stream
    /// retry classifier instead of being reported as a successful (refused) turn with nothing to retry.
    network_error: bool,
    usage: TokenUsage,
    /// Which [`REASONING_FIELDS`] entry this stream's reasoning arrives on, once known — remembered
    /// as the `Thinking` block's `signature` so a later replay (`build_body`) sends it back under the
    /// exact field name this provider expects (some accept only one of the three).
    reasoning_field: Option<&'static str>,
    /// Tool calls currently accumulating arguments, in the order first seen. See [`OpenTool`] and the
    /// struct-level doc comment above.
    open_tools: Vec<OpenTool>,
    /// The next `StreamEvent`/`ContentBlockStop` index to hand out to a newly-opened tool call —
    /// declaration order among tool calls only (mirrors pi's `blocks.indexOf`), decoupled from the
    /// wire's own `index` field so a sparse or non-zero-based one still produces compact output
    /// indices.
    next_tool_index: usize,
    /// Gemini-style `reasoning_details` entries (see `ContentBlock::ToolUse`'s doc comment) matched by
    /// id to a tool call that hasn't opened yet — keyed by the entry's own `id`, which is also the
    /// tool call's `id`. A provider (observed: OpenRouter-fronted Gemini) can send the
    /// `reasoning_details` delta *before* the `tool_calls` delta for the call it belongs to, so there's
    /// no open call yet to attach it to when it arrives; drained the moment a call with a matching id
    /// opens (or backfills that id — see `push`'s tool-call arm) rather than only checked as each entry
    /// itself arrives. Mirrors pi's `pendingReasoningDetailsByToolCallId`.
    pending_reasoning_details: HashMap<String, String>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            started: false,
            saw_finish: false,
            open: Open::None,
            stop_reason: StopReason::EndTurn,
            network_error: false,
            usage: TokenUsage::default(),
            reasoning_field: None,
            open_tools: Vec::new(),
            next_tool_index: 0,
            pending_reasoning_details: HashMap::new(),
        }
    }
}

impl Decoder {
    /// Close whichever text/thinking block is open (`self.open`), if any. Tool calls are closed
    /// separately, all together, by `close_tools`.
    fn close_text_or_thinking(&mut self, out: &mut Vec<StreamEvent>) {
        if self.open != Open::None {
            out.push(StreamEvent::ContentBlockStop { index: 0 });
            self.open = Open::None;
        }
    }

    /// Close every still-open tool call at once — the only point OpenAI Chat Completions signals "no
    /// more deltas for any block are coming" (there is no per-call stop event on the wire, unlike
    /// Anthropic's `content_block_stop`).
    fn close_tools(&mut self, out: &mut Vec<StreamEvent>) {
        for tool in self.open_tools.drain(..) {
            out.push(StreamEvent::ContentBlockStop {
                index: tool.stream_index,
            });
        }
    }
}

impl StreamDecoder for Decoder {
    fn push(&mut self, data: &Value) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if !self.started {
            self.started = true;
            out.push(StreamEvent::MessageStart);
        }

        // Trailing usage-only chunk (choices empty) — or, on some providers (Moonshot, observed), a
        // usage object nested in `choices[0].usage` instead of the standard top-level `usage`. The
        // top-level location wins when both are present, matching pi's own fallback order (`chunk.usage`
        // checked first, `choice.usage` only when it's absent — `openai-completions.ts`).
        let choice = data.get("choices").and_then(|c| c.get(0));
        let usage = data
            .get("usage")
            .filter(|u| !u.is_null())
            .or_else(|| choice.and_then(|c| c.get("usage")).filter(|u| !u.is_null()));
        if let Some(usage) = usage {
            let cached = usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            // OpenAI itself never emits `cache_write_tokens` (no such wire concept), but
            // OpenRouter-compatible providers can — separate from `cached_tokens` (cache *reads*).
            let cache_write = usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cache_write_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let prompt = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            // OpenAI's `prompt_tokens` is the *whole* prompt including both cached (read) and
            // freshly cache-written tokens; bill only the genuinely uncached remainder as
            // `input_tokens` so accounting doesn't double-count either kind. Matches pi's
            // `parseChunkUsage` (`input = promptTokens - cacheReadTokens - cacheWriteTokens`).
            self.usage.cache_read_tokens = cached;
            self.usage.cache_write_tokens = cache_write;
            self.usage.input_tokens = prompt.saturating_sub(cached).saturating_sub(cache_write);
            self.usage.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            self.usage.reasoning_tokens = usage
                .get("completion_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            out.push(StreamEvent::Usage(self.usage));
        }

        let Some(choice) = choice else {
            return out;
        };
        let delta = choice.get("delta");

        // Reasoning/thinking content, if this endpoint sends any (see `REASONING_FIELDS`). Checked
        // before plain text so a reasoning-then-answer turn closes the thinking block cleanly when
        // text starts, mirroring the Anthropic decoder's thinking-then-text shape.
        let reasoning_delta = if let Some(field) = self.reasoning_field {
            delta.and_then(|d| d.get(field)).and_then(Value::as_str)
        } else {
            let mut found = None;
            for field in REASONING_FIELDS {
                if let Some(text) = delta
                    .and_then(|d| d.get(field))
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                {
                    self.reasoning_field = Some(field);
                    found = Some(text);
                    break;
                }
            }
            found
        };
        if let Some(text) = reasoning_delta.filter(|t| !t.is_empty()) {
            if self.open == Open::None {
                self.open = Open::Thinking;
                // The field name becomes the signature once, at the start of the block — `finish`
                // doesn't need it again, and the accumulator just appends onto one signature string.
                out.push(StreamEvent::SignatureDelta {
                    index: 0,
                    signature: self.reasoning_field.unwrap_or_default().to_string(),
                });
            }
            out.push(StreamEvent::ThinkingDelta {
                index: 0,
                text: text.to_string(),
            });
        }

        // Plain text content.
        if let Some(text) = delta.and_then(|d| d.get("content")).and_then(Value::as_str) {
            if !text.is_empty() {
                if self.open == Open::Thinking {
                    self.close_text_or_thinking(&mut out);
                }
                if self.open == Open::None {
                    self.open = Open::Text;
                }
                out.push(StreamEvent::TextDelta {
                    index: 0,
                    text: text.to_string(),
                });
            }
        }

        // Tool-call deltas: id+name on first sight of a call, then `arguments` fragments. A provider
        // can have several calls open *concurrently*, interleaving each one's `arguments` deltas
        // rather than completing one before starting the next — routed to the right open call by wire
        // `index` first, falling back to `id` only when `index` is absent, matching pi's own
        // precedence (`ensureToolCallBlock` in `packages/ai/src/api/openai-completions.ts`). Without
        // this, a fragment for a call that wasn't the single "current" one used to either silently
        // corrupt whichever call happened to be open (parallel calls sharing one buffer), or — when a
        // provider like Kimi/Moonshot resent a different `id` mid-stream for the SAME call — fragment
        // one real call into two or three corrupted ones.
        if let Some(calls) = delta
            .and_then(|d| d.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for tc in calls {
                let func = tc.get("function");
                let index = tc.get("index").and_then(Value::as_i64);
                let id = tc
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty());
                let name = func.and_then(|f| f.get("name")).and_then(Value::as_str);
                let args = func
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str);

                let existing = index
                    .and_then(|idx| {
                        self.open_tools
                            .iter()
                            .position(|t| t.wire_index == Some(idx))
                    })
                    .or_else(|| id.and_then(|id| self.open_tools.iter().position(|t| t.id == id)))
                    .or_else(|| {
                        // Neither an index nor an id to key on — the pre-multi-index fallback: a bare
                        // continuation fragment belongs to the one call currently open (some minimal
                        // implementations omit `index` entirely for a single, non-parallel call). Two
                        // or more calls open with nothing to disambiguate is genuine ambiguity, not
                        // something safe to guess at — dropped below instead.
                        (index.is_none() && id.is_none() && self.open_tools.len() == 1).then_some(0)
                    });

                let slot = match existing {
                    Some(pos) => {
                        // Backfill the wire index if this call was first seen without one (e.g.
                        // matched above by `id` alone) — mirrors pi's own backfill
                        // (`block.streamIndex = streamIndex`).
                        if self.open_tools[pos].wire_index.is_none() {
                            self.open_tools[pos].wire_index = index;
                        }
                        Some(pos)
                    }
                    None if index.is_some() || id.is_some() => {
                        // A brand-new tool call. Text/thinking close first — Chat Completions tool
                        // calls never resume after lead-in text, so in practice this only ever fires
                        // once per turn (a no-op every time after).
                        self.close_text_or_thinking(&mut out);
                        let stream_index = self.next_tool_index;
                        self.next_tool_index += 1;
                        self.open_tools.push(OpenTool {
                            wire_index: index,
                            id: id.unwrap_or_default().to_string(),
                            stream_index,
                        });
                        out.push(StreamEvent::ToolUseStart {
                            index: stream_index,
                            id: id.unwrap_or_default().to_string(),
                            name: name.unwrap_or_default().to_string(),
                        });
                        Some(self.open_tools.len() - 1)
                    }
                    None => None,
                };

                let Some(slot) = slot else {
                    tracing::warn!(
                        ?index,
                        "dropping OpenAI tool-call argument fragment: no index/id to match it to an open call"
                    );
                    continue;
                };

                // First id wins — a later mutated id on an already-open index (Kimi/Moonshot) is still
                // routed here by `wire_index`/the id lookup above, but never overwrites the id this
                // call was first opened with.
                if self.open_tools[slot].id.is_empty() {
                    if let Some(id) = id {
                        self.open_tools[slot].id = id.to_string();
                    }
                }

                // A `reasoning_details` entry for this call's id may have arrived before the call
                // itself opened (see the `reasoning_details` handling below) — drain it now that the
                // id is known, rather than only checking at the moment the entry itself arrives.
                if !self.open_tools[slot].id.is_empty() {
                    if let Some(signature) = self
                        .pending_reasoning_details
                        .remove(&self.open_tools[slot].id)
                    {
                        out.push(StreamEvent::SignatureDelta {
                            index: self.open_tools[slot].stream_index,
                            signature,
                        });
                    }
                }

                if let Some(args) = args {
                    if !args.is_empty() {
                        out.push(StreamEvent::InputJsonDelta {
                            index: self.open_tools[slot].stream_index,
                            partial_json: args.to_string(),
                        });
                    }
                }
            }
        }

        // Gemini-style opaque reasoning-continuity data (OpenRouter's `reasoning_details`), tied to a
        // specific tool call by the entry's own `id` — not this decoder's wire `index`, and not
        // guaranteed to arrive after the call it belongs to (see `pending_reasoning_details`'s doc
        // comment). Checked after the tool-call loop above so a call opening in the very same delta is
        // already in `open_tools` by the time this runs. Only the one shape pi understands
        // (`isEncryptedReasoningDetail`) is matched; anything else is ignored rather than guessed at.
        if let Some(details) = delta
            .and_then(|d| d.get("reasoning_details"))
            .and_then(Value::as_array)
        {
            for detail in details {
                if !is_encrypted_reasoning_detail(detail) {
                    continue;
                }
                let id = detail.get("id").and_then(Value::as_str).unwrap_or_default();
                let serialized = serde_json::to_string(detail).unwrap_or_default();
                match self.open_tools.iter().position(|t| t.id == id) {
                    Some(pos) => out.push(StreamEvent::SignatureDelta {
                        index: self.open_tools[pos].stream_index,
                        signature: serialized,
                    }),
                    None => {
                        self.pending_reasoning_details
                            .insert(id.to_string(), serialized);
                    }
                }
            }
        }

        // Finish reason closes every open block and records why we stopped (MessageStop is held).
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.saw_finish = true;
            self.stop_reason = map_finish_reason(Some(reason));
            self.network_error = reason == "network_error";
            self.close_text_or_thinking(&mut out);
            self.close_tools(&mut out);
        }

        out
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        if !self.started {
            return Ok(Vec::new());
        }
        // A started stream that never delivered a `finish_reason` was truncated mid-flight; don't let
        // the partial turn pass as a clean completion.
        if !self.saw_finish {
            return Err(Error::Transport(
                "OpenAI stream ended before finish_reason".into(),
            ));
        }
        // `network_error` reports a transient transport fault via the *body*, not the connection — an
        // in-band failure the same way an SSE `error` event or an HTTP 5xx is, just this provider's own
        // spelling of it. Erroring here (instead of the usual `MessageStop`) routes it through
        // `run_turn`'s mid-stream retry classifier (`is_retryable_mid_stream` matches this message)
        // rather than reporting a "successful" turn the caller has no real answer for.
        if self.network_error {
            return Err(Error::Transport(
                "OpenAI finish_reason: network_error".into(),
            ));
        }
        Ok(vec![StreamEvent::MessageStop {
            stop_reason: self.stop_reason,
        }])
    }

    fn is_terminal(&self) -> bool {
        self.saw_finish
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::decode_sse;
    use crate::message::{Message, ToolDef};

    #[test]
    fn build_body_sends_temperature_when_set() {
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 256).with_temperature(0.4);
        assert_eq!(build_body(&req)["temperature"], 0.4);
    }

    #[test]
    fn build_body_omits_temperature_when_unset() {
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 256);
        assert!(build_body(&req).get("temperature").is_none());
    }

    #[test]
    fn build_body_maps_system_tool_calls_and_results() {
        let req = ModelRequest::new(
            "gpt-4o",
            vec![
                Message::user("weather?"),
                Message::assistant(vec![
                    ContentBlock::text("checking"),
                    ContentBlock::tool_use("call_1", "get_weather", json!({ "city": "SF" })),
                ]),
                Message::tool_result("call_1", "72F", false),
            ],
            256,
        )
        .with_system("be brief")
        .with_tools(vec![ToolDef {
            name: "get_weather".into(),
            description: "weather".into(),
            input_schema: json!({ "type": "object" }),
        }]);
        let body = build_body(&req);

        assert_eq!(
            body["messages"][0],
            json!({ "role": "system", "content": "be brief" })
        );
        assert_eq!(
            body["messages"][1],
            json!({ "role": "user", "content": "weather?" })
        );
        // assistant: content + a tool_call whose arguments are a JSON *string*.
        assert_eq!(body["messages"][2]["role"], "assistant");
        assert_eq!(body["messages"][2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"SF\"}"
        );
        // tool result fanned out into a role:"tool" message.
        assert_eq!(
            body["messages"][3],
            json!({ "role": "tool", "tool_call_id": "call_1", "content": "72F" })
        );
        // tool schema nested under function.parameters.
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn sets_prompt_cache_key_when_present() {
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_cache_key("session-abc");
        let body = build_body(&req);
        assert_eq!(body["prompt_cache_key"], "session-abc");
    }

    #[test]
    fn prompt_cache_key_is_omitted_for_a_model_that_does_not_support_long_cache() {
        // An unrecognized id resolves to `ModelCaps::unknown()` (supports_long_cache: false) — this
        // dialect is the fallback for every third-party OpenAI-compatible provider, and a
        // strict-schema endpoint can 400 the whole request over an unrecognized field.
        let req = ModelRequest::new("some-third-party-model", vec![Message::user("hi")], 64)
            .with_cache_key("session-abc");
        let body = build_body(&req);
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn user_images_become_image_url_parts() {
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "gpt-4o",
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            64,
        );
        let body = build_body(&req);
        let content = &body["messages"][0]["content"];
        // A multimodal parts array, not a bare string — the text first, then the image data-URI.
        assert_eq!(
            content[0],
            json!({ "type": "text", "text": "what is this?" })
        );
        assert_eq!(
            content[1],
            json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AAAA" }
            })
        );
    }

    #[test]
    fn reasoning_effort_emitted_only_for_reasoning_models() {
        use crate::transport::ReasoningEffort;
        // A reasoning model with an effort set emits `reasoning_effort`.
        let body = build_body(
            &ModelRequest::new("o3-mini", vec![Message::user("hi")], 64)
                .with_reasoning_effort(ReasoningEffort::High),
        );
        assert_eq!(body["reasoning_effort"], "high");

        // A non-reasoning model never emits it, even if a level is set.
        let body = build_body(
            &ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
                .with_reasoning_effort(ReasoningEffort::High),
        );
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn reasoning_effort_unset_emits_an_explicit_off_signal_when_disableable() {
        // No level requested this turn, but the model can be told explicitly rather than relying on
        // the provider's own undocumented default for an omitted field.
        let body = build_body(&ModelRequest::new("o3-mini", vec![Message::user("hi")], 64));
        assert_eq!(
            body["reasoning_effort"], "none",
            "a disable-capable reasoning model must get an explicit off signal, not a missing field"
        );
    }

    #[test]
    fn reasoning_effort_unset_omits_the_field_when_not_disableable() {
        // A reasoning model that can't be told to turn reasoning off at all (`reasoning_disableable:
        // false`) must not get a fabricated "none" it doesn't actually support.
        let body = build_body(&ModelRequest::new(
            "gpt-5.3-codex-spark",
            vec![Message::user("hi")],
            64,
        ));
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn tool_result_images_fan_out_to_a_user_message() {
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "gpt-4o",
            vec![Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".into(),
                content: "screenshot attached".into(),
                is_error: false,
                images: vec![ImageSource::base64("image/png", "AAAA")],
            }])],
            256,
        );
        let body = build_body(&req);
        // The tool role carries the text; the image fans out to a following user message (the tool
        // role can't carry images on the OpenAI wire).
        assert_eq!(body["messages"][0]["role"], "tool");
        assert_eq!(body["messages"][0]["content"], "screenshot attached");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"][1]["type"], "image_url");
        assert_eq!(
            body["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn a_tool_result_with_only_an_image_and_no_text_still_fans_the_image_out() {
        // A-L8 pi-parity test gap (fixed, `packages/ai/test/image-tool-result.test.ts`): a tool
        // returning only an image (no text) must still produce a usable `tool` message (empty string
        // content — OpenAI's Chat Completions `tool` role requires a string, even an empty one, not
        // `null`/absent) followed by the image-carrying user message.
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "gpt-4o",
            vec![Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".into(),
                content: String::new(),
                is_error: false,
                images: vec![ImageSource::base64("image/png", "AAAA")],
            }])],
            256,
        );
        let body = build_body(&req);
        assert_eq!(body["messages"][0]["role"], "tool");
        assert_eq!(body["messages"][0]["content"], "");
        assert_eq!(body["messages"][1]["role"], "user");
        let user_content = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(
            user_content
                .iter()
                .filter(|b| b["type"] == "image_url")
                .count(),
            1,
            "expected exactly one image part: {user_content:#?}"
        );
    }

    #[test]
    fn consecutive_tool_result_images_stay_contiguous_with_their_tool_messages() {
        // pi-parity fix: two tool results in the same turn, both carrying images, must keep
        // `tool, tool, user` ordering — one trailing image-carrier message covering both — not
        // `tool, user, tool, user` (an image-carrier message wedged between the two `tool`
        // messages), which risks a 400 from a backend that strictly validates a tool-response run is
        // immediately followed by the next real turn, nothing else in between. Mirrors pi's
        // `packages/ai/test/openai-completions-tool-result-images.test.ts`, "batches tool-result
        // images after consecutive tool results".
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "gpt-4o",
            vec![Message::tool_results(vec![
                ContentBlock::ToolResult {
                    tool_use_id: "tool-1".into(),
                    content: "Read image file [image/png]".into(),
                    is_error: false,
                    images: vec![ImageSource::base64("image/png", "AAAA")],
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tool-2".into(),
                    content: "Read image file [image/png]".into(),
                    is_error: false,
                    images: vec![ImageSource::base64("image/png", "BBBB")],
                },
            ])],
            256,
        );
        let body = build_body(&req);
        let messages = body["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(
            roles,
            vec!["tool", "tool", "user"],
            "the two tool results must stay adjacent, with one trailing image message — not \
             interleaved as tool, user, tool, user"
        );
        assert_eq!(messages[0]["tool_call_id"], "tool-1");
        assert_eq!(messages[1]["tool_call_id"], "tool-2");

        // The combined trailing message carries both tool calls' images, in order.
        let image_parts: Vec<&Value> = messages[2]["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["type"] == "image_url")
            .collect();
        assert_eq!(image_parts.len(), 2);
        assert_eq!(
            image_parts[0]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
        assert_eq!(
            image_parts[1]["image_url"]["url"],
            "data:image/png;base64,BBBB"
        );
    }

    #[test]
    fn consecutive_tool_result_image_placeholders_stay_contiguous_for_a_non_vision_model() {
        // Same contiguity fix as above, for the non-vision placeholder path: both placeholder lines
        // fold into one trailing message rather than one per tool result.
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "o3-mini",
            vec![Message::tool_results(vec![
                ContentBlock::ToolResult {
                    tool_use_id: "tool-1".into(),
                    content: "ok".into(),
                    is_error: false,
                    images: vec![ImageSource::base64("image/png", "AAAA")],
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tool-2".into(),
                    content: "ok".into(),
                    is_error: false,
                    images: vec![ImageSource::base64("image/png", "BBBB")],
                },
            ])],
            256,
        );
        let body = build_body(&req);
        let messages = body["messages"].as_array().unwrap();
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["tool", "tool", "user"]);
        assert_eq!(
            messages[2]["content"],
            "Image output from tool call tool-1: (tool image omitted: model does not support images)\n\
             Image output from tool call tool-2: (tool image omitted: model does not support images)"
        );
    }

    #[test]
    fn images_are_downgraded_to_a_placeholder_for_a_non_vision_model() {
        use crate::message::ImageSource;
        // o3-mini is the one o-series id that isn't vision-capable.
        let req = ModelRequest::new(
            "o3-mini",
            vec![
                Message::user_with_images(
                    "what is this?",
                    vec![ImageSource::base64("image/png", "AAAA")],
                ),
                Message::tool_results(vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "screenshot attached".into(),
                    is_error: false,
                    images: vec![ImageSource::base64("image/png", "BBBB")],
                }]),
            ],
            64,
        );
        let body = build_body(&req);

        // A plain string, not a multimodal parts array — the image became a placeholder note.
        assert_eq!(
            body["messages"][0]["content"],
            "what is this?\n(image omitted: model does not support images)"
        );

        // The tool-result image never fans out to a following user message at all — no `image_url`
        // part anywhere, just the placeholder folded into the tool message's own text content.
        assert_eq!(body["messages"][1]["role"], "tool");
        assert_eq!(body["messages"][1]["content"], "screenshot attached");
        assert_eq!(body["messages"][2]["role"], "user");
        assert_eq!(
            body["messages"][2]["content"],
            "Image output from tool call call_1: (tool image omitted: model does not support images)"
        );

        // A vision-capable model is unaffected.
        let req = ModelRequest::new(
            "gpt-4o",
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            64,
        );
        let body = build_body(&req);
        assert_eq!(body["messages"][0]["content"][1]["type"], "image_url");
    }

    #[test]
    fn non_vision_model_collapses_only_consecutive_image_runs_not_the_whole_turn() {
        // [A-M11] pi-parity fix: the placeholder downgrade used to gate on a single turn-wide
        // `has_image` bool, joining *every* text block via `text_of` and appending one trailing
        // placeholder regardless of how many images were present or where — two images separated by
        // text collapsed into a single placeholder with the text ordering lost, instead of one
        // placeholder per genuinely consecutive run of images, in place. Mirrors the Anthropic
        // dialect's `downgrade_unsupported_images` run-collapsing state machine
        // (`dialect::anthropic`), and pi's `replaceImagesWithPlaceholder`
        // (`transform-messages.ts`).
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "o3-mini", // the one o-series id that isn't vision-capable
            vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::text("first"),
                    ContentBlock::Image {
                        source: ImageSource::base64("image/png", "AAAA"),
                    },
                    ContentBlock::Image {
                        source: ImageSource::base64("image/png", "BBBB"),
                    },
                    ContentBlock::text("middle"),
                    ContentBlock::Image {
                        source: ImageSource::base64("image/png", "CCCC"),
                    },
                    ContentBlock::text("last"),
                ],
                model_id: None,
                error_message: None,
                aborted: false,
            }],
            64,
        );
        let body = build_body(&req);
        assert_eq!(
            body["messages"][0]["content"],
            "first\n\
             (image omitted: model does not support images)\n\
             middle\n\
             (image omitted: model does not support images)\n\
             last",
            "the leading two-image run must collapse into ONE placeholder, and the later lone image \
             into its own separate one — not the whole turn's three images into a single placeholder"
        );
    }

    #[test]
    fn reasoning_models_use_max_completion_tokens() {
        // o-series / gpt-5 reject `max_tokens` on chat-completions; the body must carry
        // `max_completion_tokens` instead. A non-reasoning model keeps `max_tokens`.
        let body = build_body(&ModelRequest::new(
            "o3-mini",
            vec![Message::user("hi")],
            256,
        ));
        assert_eq!(body["max_completion_tokens"], 256);
        assert!(body.get("max_tokens").is_none());

        let body = build_body(&ModelRequest::new("gpt-4o", vec![Message::user("hi")], 256));
        assert_eq!(body["max_tokens"], 256);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn max_tokens_is_clamped_when_the_live_prompt_nears_the_context_window() {
        // HIGH pi-parity gap (fixed): `clamp_max_tokens_to_context` was implemented for the Anthropic
        // dialect only; this Chat-Completions dialect wrote `req.max_tokens` straight onto the wire
        // unclamped. gpt-4o: 128_000 context window. A ~100_000-token prompt (400_000 chars, the
        // chars/4 estimator) leaves 128_000 - 100_000 - 4_096 (the shared margin) = 23_904 tokens of
        // headroom — less than the request's own (artificially large, to force the clamp) max_tokens.
        let big_text = "x".repeat(400_000);
        let req = ModelRequest::new("gpt-4o", vec![Message::user(big_text)], 50_000);
        let body = build_body(&req);
        assert_eq!(
            body["max_tokens"], 23_904,
            "max_tokens must be clamped down to the actual remaining context headroom, not sent as \
             the static 50_000 ceiling regardless of how much of the window the prompt already fills"
        );
    }

    #[test]
    fn max_tokens_is_unchanged_for_a_prompt_nowhere_near_the_context_window() {
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 8_192);
        let body = build_body(&req);
        assert_eq!(body["max_tokens"], 8_192);
    }

    #[test]
    fn tool_choice_emitted_only_when_set() {
        use crate::transport::ToolChoice;
        let tools = vec![ToolDef {
            name: "get_weather".into(),
            description: "weather".into(),
            input_schema: json!({ "type": "object" }),
        }];
        // Unset → no `tool_choice` on the wire.
        let req =
            ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_tools(tools.clone());
        assert!(build_body(&req).get("tool_choice").is_none());

        // The auto/none/required cases are bare strings.
        for (choice, wire) in [
            (ToolChoice::Auto, "auto"),
            (ToolChoice::None, "none"),
            (ToolChoice::Required, "required"),
        ] {
            let body = build_body(
                &ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
                    .with_tools(tools.clone())
                    .with_tool_choice(choice),
            );
            assert_eq!(body["tool_choice"], wire);
        }

        // A specific tool is the nested function object.
        let body = build_body(
            &ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
                .with_tools(tools)
                .with_tool_choice(ToolChoice::Tool("get_weather".into())),
        );
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "function", "function": { "name": "get_weather" } })
        );
    }

    #[test]
    fn text_only_user_stays_a_plain_string() {
        // Regression guard: the common no-image path must keep the flat string shape.
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hello")], 64);
        let body = build_body(&req);
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn tools_field_is_omitted_entirely_when_no_tools_are_registered() {
        // [A-L4] Direct regression pin for the no-tool-history common case (correct by inspection —
        // no prior direct test): some OpenAI-compatible backends (DashScope/Aliyun Qwen via
        // compatible-mode) reject `tools: []` outright ("[] is too short - 'tools'"), so the field
        // must be missing from the wire body, not present-but-empty. Mirrors pi's
        // `openai-completions-empty-tools.test.ts`, "omits tools field when context.tools is
        // undefined".
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64);
        let body = build_body(&req);
        assert!(
            body.as_object().unwrap().get("tools").is_none(),
            "got: {body}"
        );
    }

    #[test]
    fn tools_field_falls_back_to_an_empty_array_when_the_conversation_has_tool_history() {
        // [A-M4] pi-parity fix: a turn with no tools registered *this* time (e.g. tools toggled off
        // mid-session) but whose conversation already contains a `ToolUse`/`ToolResult` pair used to
        // omit `tools` entirely — Anthropic, reached through an OpenAI-compatible proxy (LiteLLM), 400s
        // that shape: it wants to see *some* `tools` value once tool calls have touched the
        // conversation, even an empty array. Mirrors pi's `hasToolHistory` fallback
        // (`openai-completions.ts`) and its test `openai-completions-empty-tools.test.ts`, "still
        // emits tools: [] for Anthropic/LiteLLM proxy when conversation has tool history".
        let req = ModelRequest::new(
            "gpt-4o",
            vec![
                Message::user("use the tool"),
                Message::assistant(vec![ContentBlock::tool_use("t1", "noop", json!({}))]),
                Message::tool_result("t1", "done", false),
            ],
            64,
        );
        let body = build_body(&req);
        assert_eq!(body["tools"], json!([]), "got: {body}");
    }

    #[test]
    fn never_emits_a_strict_tool_schema_flag() {
        // [A-L6] pi emits `strict: false` on each tool's `function` object by default, omitted only
        // under a per-model compat override this codebase has no equivalent concept of (no
        // per-model-compat-override system exists here at all — confirmed out of scope in prior
        // pi-parity audits; the gateway owns per-provider routing/compat instead). Since there's
        // nothing to override *to*, the only correct behavior here is to never emit the field at all
        // — pinned explicitly so a future change doesn't silently start sending a flag no caller here
        // has ever asked for. See pi's `convertTools` (`openai-completions.ts`) and
        // `openai-completions-tool-choice.test.ts:153-196` for the compat-override case this
        // intentionally doesn't have.
        let req =
            ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_tools(vec![ToolDef {
                name: "get_weather".into(),
                description: "weather".into(),
                input_schema: json!({ "type": "object" }),
            }]);
        let body = build_body(&req);
        assert!(
            body["tools"][0]["function"].get("strict").is_none(),
            "got: {body}"
        );
    }

    // A recorded text + tool_call streamed response (trailing usage chunk + [DONE]).
    const FIXTURE: &str = r#"
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"content":"Let me check."},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_42","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"SF\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"choices":[],"usage":{"prompt_tokens":24,"completion_tokens":31,"total_tokens":55}}

data: [DONE]
"#;

    #[test]
    fn decodes_text_then_tool_call_stream() {
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, FIXTURE).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                StreamEvent::TextDelta {
                    index: 0,
                    text: "Let me check.".into()
                },
                // Text block closes when the tool call begins — same shape the Anthropic decoder
                // produces, so the loop assembles both dialects identically.
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: "call_42".into(),
                    name: "get_weather".into()
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "{\"city\":".into()
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "\"SF\"}".into()
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 24,
                    output_tokens: 31,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
    }

    #[test]
    fn a_non_json_trailer_after_finish_reason_is_ignored_not_a_transport_error() {
        // pi-parity fix: `is_terminal()` (used to gate this exact tolerance — see
        // `dialect::push_sse_line`) was only ever overridden on the Anthropic decoder, even though this
        // one already tracks the same `saw_finish` state — a trailing keepalive/stats line from a
        // gateway/proxy after the real `finish_reason` arrived used to hard-error an otherwise-
        // successful turn, unlike the Anthropic path.
        const TRAILING_NOISE: &str = r#"
data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: not-json-at-all

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, TRAILING_NOISE).unwrap();
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn
            })
        );
    }

    #[test]
    fn a_non_json_data_line_before_finish_reason_is_still_a_hard_error() {
        // The flip side of the test above: garbage arriving *before* `finish_reason` is a genuine
        // corrupted/tampered stream, not trailing proxy noise — must still fail loudly.
        const GARBAGE_MID_STREAM: &str = r#"
data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}

data: not-json-at-all
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, GARBAGE_MID_STREAM).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn content_filter_finish_reason_maps_to_refusal_not_a_clean_stop() {
        // A safety-filtered completion must not be indistinguishable from a normal `EndTurn` — the
        // loop treats `Refusal` as a distinct terminal condition (ends the run, doesn't drain queued
        // steer/follow-up messages), where `Other` would silently be read as success.
        const FILTERED: &str = r#"
data: {"choices":[{"index":0,"delta":{"content":"partial an"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, FILTERED).unwrap();
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::Refusal
            })
        );
    }

    #[test]
    fn network_error_finish_reason_is_a_retryable_transport_error_not_a_refusal() {
        // pi-parity fix: `network_error` reports a transient transport fault via the body, not a
        // genuine refusal — unlike `content_filter` above, it must reach `run_turn`'s mid-stream retry
        // classifier as an `Err`, not be reported as a "successful" (if unusable) turn with
        // `StopReason::Refusal` that the loop just gives up on. Both halves matter: the decoder must
        // actually error, and `is_retryable_mid_stream` must actually recognize that error's message.
        const NETWORK_ERROR: &str = r#"
data: {"choices":[{"index":0,"delta":{"content":"partial an"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"network_error"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, NETWORK_ERROR).unwrap_err();
        assert!(
            crate::agent::is_retryable_mid_stream(&err),
            "must be classified as mid-stream-retryable, got: {err:?}"
        );
    }

    #[test]
    fn interleaved_parallel_tool_calls_accumulate_independently_by_index() {
        // pi-parity fix: two parallel tool calls (wire index 0 then index 1) genuinely overlap —
        // index 1 opens *before* index 0's last argument fragment arrives, mirroring a provider that
        // interleaves parallel calls' deltas rather than completing one before starting the next
        // (see pi's `packages/ai/test/openai-completions-tool-choice.test.ts:767-997`, "accumulates
        // mixed content, reasoning, and parallel tool call deltas independently"). Both calls must
        // stay open and accumulate independently — the late index-0 fragment belongs to call_A, not
        // whatever happens to be "currently open". This decoder used to force-close call_A the moment
        // call_B opened, dropping the late fragment as "out of order" and truncating call_A's JSON to
        // `{"x":1` — this test used to pin exactly that corrupted/truncated behavior as expected.
        const INTERLEAVED: &str = r#"
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_A","type":"function","function":{"name":"alpha","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_B","type":"function","function":{"name":"beta","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"y\":2}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, INTERLEAVED).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: "call_A".into(),
                    name: "alpha".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "{\"x\":1".into(),
                },
                // call_B opens at a genuinely distinct index — call_A stays open, not force-closed.
                StreamEvent::ToolUseStart {
                    index: 1,
                    id: "call_B".into(),
                    name: "beta".into(),
                },
                // The late index-0 "}" fragment now correctly reaches call_A's still-open buffer,
                // instead of being dropped as "out of order" or corrupting call_B's.
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "}".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 1,
                    partial_json: "{\"y\":2}".into(),
                },
                // Neither call has a per-call stop event on the wire — both close together at
                // `finish_reason`, in the order they were first declared.
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::ContentBlockStop { index: 1 },
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        );

        // Exact final accumulated arguments per call, not just "no crash": concatenate each index's
        // `InputJsonDelta` fragments and confirm they parse to the right object — call_A's fragments
        // (`{"x":1` + `}`) must join into valid, uncorrupted JSON despite call_B opening in between.
        let args_for = |index: usize| -> String {
            events
                .iter()
                .filter_map(|e| match e {
                    StreamEvent::InputJsonDelta {
                        index: i,
                        partial_json,
                    } if *i == index => Some(partial_json.as_str()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(
            serde_json::from_str::<Value>(&args_for(0)).unwrap(),
            json!({ "x": 1 })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&args_for(1)).unwrap(),
            json!({ "y": 2 })
        );
    }

    #[test]
    fn tool_call_id_mutating_mid_stream_does_not_fragment_the_call() {
        // pi-parity fix: some providers (Kimi/Moonshot, observed) resend a *different* `id` on later
        // fragments of the SAME tool call rather than keeping it stable — coalescing must key on the
        // stable wire `index`, not `id`, or one real call fragments into two or three corrupted ones.
        // Mirrors pi's `packages/ai/test/openai-completions-tool-choice.test.ts:656-765`, "coalesces
        // tool call deltas by stable index when provider mutates ids mid-stream".
        const MUTATING_ID: &str = r#"
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"functions.read:0","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"chatcmpl-tool-a","function":{"arguments":"{\"path\":\"README"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"chatcmpl-tool-b","function":{"arguments":".md\"}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, MUTATING_ID).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                // Only ONE `ToolUseStart` — the id-mutated later chunks route back to the same open
                // call (matched by wire index 0), never opening a second or third block.
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: "functions.read:0".into(),
                    name: "read".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "{\"path\":\"README".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: ".md\"}".into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        );
        // First id wins — the two later id mutations ("chatcmpl-tool-a", "chatcmpl-tool-b") are never
        // surfaced anywhere, matching pi's own `ensureToolCallBlock` (only sets `block.id` once,
        // while it's still empty).
        let id = events.iter().find_map(|e| match e {
            StreamEvent::ToolUseStart { id, .. } => Some(id.as_str()),
            _ => None,
        });
        assert_eq!(id, Some("functions.read:0"));
        let args: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::InputJsonDelta { partial_json, .. } => Some(partial_json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&args).unwrap(),
            json!({ "path": "README.md" })
        );
    }

    #[test]
    fn separates_cached_from_uncached_input() {
        const CACHED: &str = r#"
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":900},"completion_tokens_details":{"reasoning_tokens":12}}}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, CACHED).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.input_tokens, 100); // 1000 prompt - 900 cached
        assert_eq!(usage.cache_read_tokens, 900);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.reasoning_tokens, 12);
    }

    #[test]
    fn separates_cache_write_tokens_from_cache_read_and_uncached_input() {
        // [A-M2] pi-parity fix: `prompt_tokens_details.cache_write_tokens` (OpenRouter-compatible
        // providers; OpenAI itself never emits it) was parsed nowhere — `cache_write_tokens` stayed
        // permanently 0, and worse, `input_tokens` never subtracted it out of `prompt_tokens` either,
        // double-billing cache-written tokens as if they were fresh uncached input. Mirrors pi's
        // `parseChunkUsage` (`input = promptTokens - cacheReadTokens - cacheWriteTokens`) and its test
        // `openai-completions-tool-choice.test.ts`, "preserves prompt_tokens_details cache read/write
        // fields from chunk usage".
        const CACHE_WRITE: &str = r#"
data: {"choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":50,"cache_write_tokens":30},"completion_tokens_details":{"reasoning_tokens":0}}}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, CACHE_WRITE).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        // cached_tokens is cache reads; cache_write_tokens is separate — neither counts as fresh
        // uncached input.
        assert_eq!(usage.input_tokens, 20); // 100 prompt - 50 read - 30 write
        assert_eq!(usage.cache_read_tokens, 50);
        assert_eq!(usage.cache_write_tokens, 30);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn falls_back_to_choice_usage_when_the_top_level_usage_field_is_absent() {
        // [A-M2] pi-parity fix: some providers (Moonshot, observed) report usage nested at
        // `choices[0].usage` instead of the standard top-level `usage` — this decoder only ever read
        // the top-level location, so a Moonshot-style trailing chunk silently reported all-zero usage
        // (input/output/cache tokens all default to 0) instead of erroring or falling back. Mirrors
        // pi's own fallback (`openai-completions.ts`: "some providers... return usage in choice.usage
        // instead of the standard chunk.usage") and its test `openai-completions-tool-choice.test.ts`,
        // "preserves prompt_tokens_details cache read/write fields from choice usage fallback".
        const CHOICE_USAGE: &str = r#"
data: {"choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":50,"cache_write_tokens":30},"completion_tokens_details":{"reasoning_tokens":0}}}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, CHOICE_USAGE).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect(
                "a usage event even though usage arrived at choices[0].usage, not the top level",
            );
        assert_eq!(usage.input_tokens, 20);
        assert_eq!(usage.cache_read_tokens, 50);
        assert_eq!(usage.cache_write_tokens, 30);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn a_literal_null_top_level_usage_chunk_is_ignored() {
        // [A-L5] Direct regression pin (null-safe by inspection — no prior direct test): a trailing
        // chunk some gateways send with a literal JSON `null` for `usage` (rather than omitting the
        // key) must not be treated as "usage present" and blow away the real figures a prior chunk
        // already recorded. Mirrors pi's `openai-completions-tool-choice.test.ts:582-622`.
        const NULL_USAGE: &str = r#"
data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}

data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, NULL_USAGE).unwrap();
        let usages: Vec<TokenUsage> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .collect();
        assert_eq!(
            usages,
            vec![TokenUsage {
                input_tokens: 10,
                output_tokens: 2,
                ..Default::default()
            }],
            "the literal-null usage chunk must not emit a second (zeroed-out) Usage event"
        );
    }

    #[test]
    fn truncated_stream_is_rejected() {
        // Content but no `finish_reason`.
        const TRUNCATED: &str = r#"
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"partial"},"finish_reason":null}]}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, TRUNCATED).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn decodes_reasoning_content_then_text_and_tool_call() {
        // `reasoning_content` (llama.cpp / most OpenAI-compatible endpoints): a thinking block should
        // open, close when text starts, and the field name becomes the block's replay signature.
        const REASONING: &str = r#"
data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me "},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"reasoning_content":"think."},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"content":"Answer."},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, REASONING).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                StreamEvent::SignatureDelta {
                    index: 0,
                    signature: "reasoning_content".into()
                },
                StreamEvent::ThinkingDelta {
                    index: 0,
                    text: "Let me ".into()
                },
                StreamEvent::ThinkingDelta {
                    index: 0,
                    text: "think.".into()
                },
                StreamEvent::ContentBlockStop { index: 0 }, // thinking closes when text starts
                StreamEvent::TextDelta {
                    index: 0,
                    text: "Answer.".into()
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn
                },
            ]
        );
    }

    #[test]
    fn reasoning_field_wins_first_seen_even_if_a_second_field_also_appears() {
        // Some gateways (e.g. chutes.ai per the reference agent) echo the same text on both
        // `reasoning_content` and `reasoning` at once — only the first-seen field should count, or the
        // text would be duplicated.
        const DUAL: &str = r#"
data: {"choices":[{"index":0,"delta":{"reasoning_content":"dup","reasoning":"dup"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"reasoning":"-should-be-ignored"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, DUAL).unwrap();
        let thinking_text: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ThinkingDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(thinking_text, "dup");
    }

    #[test]
    fn assistant_reasoning_block_replays_under_its_captured_field_name() {
        let req = ModelRequest::new(
            "some-oss-reasoning-model",
            vec![
                Message::user("solve it"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: "step one".into(),
                        signature: "reasoning".into(),
                    },
                    ContentBlock::text("42"),
                ]),
            ],
            64,
        );
        let body = build_body(&req);
        assert_eq!(body["messages"][1]["reasoning"], "step one");
        assert_eq!(body["messages"][1]["content"], "42");
        // Never invented a field this decoder didn't actually see.
        assert!(body["messages"][1].get("reasoning_content").is_none());
    }

    #[test]
    fn reasoning_details_arriving_before_the_matching_tool_call_are_still_attached() {
        // [A-M5] pi-parity fix: Gemini-style opaque reasoning-continuity data (OpenRouter's
        // `reasoning_details`, matching pi's `thoughtSignature`) was parsed nowhere in this dialect —
        // dropped on the floor rather than captured and replayed. Mirrors pi's own scenario exactly:
        // OpenRouter-fronted Gemini sends the `reasoning_details` delta *before* the `tool_calls` delta
        // for the call it belongs to, matched after the fact by the entry's own `id`. See
        // `openai-completions-reasoning-details.test.ts`, "preserves reasoning_details that arrive
        // before their matching tool call".
        const REASONING_DETAILS: &str = r#"
data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.encrypted","id":"call_1","data":"encrypted-signature"}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"path\":\"README.md\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, REASONING_DETAILS).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: "call_1".into(),
                    name: "read".into(),
                },
                // The reasoning_details entry, matched to call_1 by id even though it arrived in an
                // earlier chunk with no tool call open yet to attach it to. Re-serialized from the
                // parsed `Value` (alphabetical key order — `serde_json`'s default map — rather than the
                // wire's own field order), which is fine: this string is opaque and only ever parsed
                // back with `serde_json`, never compared byte-for-byte against the original wire text.
                StreamEvent::SignatureDelta {
                    index: 0,
                    signature: r#"{"data":"encrypted-signature","id":"call_1","type":"reasoning.encrypted"}"#
                        .into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "{\"path\":\"README.md\"}".into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        );
    }

    #[test]
    fn a_malformed_or_unrecognized_reasoning_detail_entry_is_ignored() {
        // [A-M5] Only the one shape pi understands (`type: "reasoning.encrypted"` with non-empty `id`
        // and `data`) is captured; anything else — a different `type`, or a missing/empty `id`/`data`
        // — must not produce a `SignatureDelta` at all, matching pi's `isEncryptedReasoningDetail`.
        const UNRECOGNIZED: &str = r#"
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.summary","id":"call_1","data":"plaintext, not encrypted"},{"type":"reasoning.encrypted","id":"","data":"no id"},{"type":"reasoning.encrypted","id":"call_1","data":""}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, UNRECOGNIZED).unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::SignatureDelta { .. })),
            "got: {events:?}"
        );
    }

    #[test]
    fn assistant_tool_use_reasoning_details_replay_under_the_wires_array_shape() {
        // [A-M5] The encode-side half of the same fix: a prior turn's tool call that carried captured
        // reasoning-continuity data must replay it back on the next request under `reasoning_details`
        // — the wire's own array-of-objects shape (one entry per call that has one), not the opaque
        // string this crate stores it as internally. Mirrors pi's own replay
        // (`openai-completions.ts`'s `reasoning_details` assembly) and the second half of
        // `openai-completions-reasoning-details.test.ts`'s scenario (the payload sent on the *next*
        // request after the tool-call turn was received).
        let req = ModelRequest::new(
            "google/gemini-test",
            vec![
                Message::user("solve it"),
                Message::assistant(vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "read".into(),
                    input: json!({ "path": "README.md" }),
                    thought_signature: Some(
                        r#"{"type":"reasoning.encrypted","id":"call_1","data":"encrypted-signature"}"#
                            .into(),
                    ),
                }]),
                Message::tool_result("call_1", "file contents", false),
            ],
            64,
        );
        let body = build_body(&req);
        assert_eq!(
            body["messages"][1]["reasoning_details"],
            json!([{ "type": "reasoning.encrypted", "id": "call_1", "data": "encrypted-signature" }])
        );

        // A call with no captured signature at all must not fabricate the field.
        let req = ModelRequest::new(
            "gpt-4o",
            vec![
                Message::user("solve it"),
                Message::assistant(vec![ContentBlock::tool_use(
                    "call_1",
                    "read",
                    json!({ "path": "README.md" }),
                )]),
            ],
            64,
        );
        let body = build_body(&req);
        assert!(body["messages"][1].get("reasoning_details").is_none());
    }

    #[test]
    fn cache_long_emits_24h_retention_when_supported() {
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_cache_long(true);
        let body = build_body(&req);
        assert_eq!(body["prompt_cache_retention"], "24h");

        // Off by default.
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64);
        assert!(build_body(&req).get("prompt_cache_retention").is_none());
    }

    #[test]
    fn prompt_cache_key_is_clamped_to_64_chars() {
        let long_key = "k".repeat(200);
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_cache_key(long_key.clone());
        let body = build_body(&req);
        let key = body["prompt_cache_key"].as_str().unwrap();
        assert_eq!(key.len(), 64);
        assert_eq!(key, &long_key[..64]);

        // A short key passes through untouched.
        let req =
            ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_cache_key("short-id");
        assert_eq!(build_body(&req)["prompt_cache_key"], "short-id");
    }
}
