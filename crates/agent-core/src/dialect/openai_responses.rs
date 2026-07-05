//! OpenAI Responses wire (`/v1/responses`).
//!
//! Every native OpenAI model id (gpt-4/4.1/4o, the gpt-5 family, o-series — see
//! [`crate::models::ApiKind`]) speaks this API, not Chat Completions: it's what carries reasoning
//! summaries, encrypted-reasoning continuity across turns, and typed refusal/failure events. Every
//! third-party OpenAI-compatible provider stays on [`super::openai`] (Chat Completions), which they
//! actually implement.
//!
//! **Request shape vs Chat Completions:** the conversation is a flat `input` array of typed items
//! (`message` / `function_call` / `function_call_output` / a raw reasoning item), not `messages`;
//! `max_tokens` becomes `max_output_tokens`; tool defs are flat (`{type:"function", name, …}`, no
//! nested `function` wrapper); reasoning is driven by `reasoning.effort` (mirroring the OpenAI Chat
//! Completions dialect's `reasoning_effort`), not a token budget. `store:false` always, since this
//! harness is stateless — every turn resends the full history rather than referencing a server-side
//! `previous_response_id`.
//!
//! **Thinking-block reuse:** no new `ContentBlock` variant is needed. A reasoning item's opaque,
//! replayable representation is the *entire item, JSON-stringified*, carried in
//! [`crate::message::ContentBlock::Thinking`]'s `signature` field (mirroring pi's
//! `JSON.stringify(item)`) — decode parses it back out verbatim on replay. A signature that fails to
//! parse as JSON (a foreign block that reached here despite `set_model`'s thinking scrub) degrades to
//! plain text instead of erroring, mirroring pi's `isSameModel` downgrade.
//!
//! **Stream shape vs Chat Completions:** items open with `response.output_item.added` and close with
//! `response.output_item.done` — genuine block boundaries, unlike Chat Completions' implicit
//! index-keyed deltas. The API can genuinely interleave two items' deltas (concurrent tool calls);
//! since `StreamEvent`'s block-scoped variants all carry an `index` and `Accumulator` (agent-core's
//! stream-to-`ContentBlock` fold) natively tracks as many concurrently-open indices as the wire
//! actually has, this decoder just emits every event with its own true `output_index` immediately, in
//! real arrival order — no buffering, no "focus" item, no replay-on-close. Usage and the terminal stop
//! reason arrive together on `response.completed`/`response.incomplete`, so (unlike the Chat
//! Completions decoder) `MessageStop` is emitted immediately rather than deferred to `finish()`.
//!
//! **Resync at block close:** `response.function_call_arguments.done` and `output_item.done`'s own
//! `item.arguments`/`item.content` are the provider's own authoritative, complete values for the
//! block that's closing — emitted as [`crate::message::StreamEvent::InputJsonFinal`]/`TextFinal`,
//! which *replace* (not append to) whatever the streamed deltas accumulated. This is what keeps a
//! single dropped or duplicated mid-stream delta (a relay hiccup with no transport-level error —
//! nothing else would ever catch it) from silently corrupting the final block.

use serde_json::{Map, Value, json};

use super::StreamDecoder;
use crate::error::{Error, Result};
use crate::message::{ContentBlock, Role, StopReason, StreamEvent, TokenUsage};
use crate::transport::{ModelRequest, ToolChoice};

fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Combine a function call's `call_id` (pairs a `function_call_output` back to its call) and item
/// `id` (pairs the call with a preceding `reasoning` item for OpenAI's own validation) into our
/// internal `ToolUse.id`, so both survive round-tripping through the dialect-agnostic `ContentBlock`.
/// Pack `call_id` and `item_id` into one id. `call_id` is API-generated (OpenAI's own opaque token
/// format) and in practice never contains `|`, but nothing guarantees a non-standard
/// OpenAI-compatible provider honors that — a naive join would let a `|` inside `call_id` corrupt
/// *both* halves on the next `split_tool_id` (it finds the first `|`, so one embedded earlier than
/// intended shifts the split point). Escaping each half first (`\` → `\\`, `|` → `\|`) makes the join
/// round-trip correctly regardless of what either id contains; the common case, where neither needs
/// escaping, costs nothing beyond the two no-op `contains` checks.
///
/// `item_id` gets one more check before escaping: a same-dialect backend's own item id is always
/// short (OpenAI's own are `fc_`-prefixed and well under [`MAX_ITEM_ID_LEN`]), but nothing guarantees
/// that of a non-standard OpenAI-Responses-compatible provider sitting behind the same dialect — e.g.
/// GitHub Copilot's own tool-call ids are shaped `call_id|<450+ char opaque blob>`. Replaying a foreign
/// id like that is exactly what [`escape_tool_id_part`] used to do unconditionally: pass it through
/// mostly as-is, which risks handing a *different* backend an oversized or oddly-charset'd id it
/// rejects outright. Mirroring pi's `buildForeignResponsesItemId`, an oversized `item_id` is replaced
/// with a short, deterministic, one-way digest (`fc_<hash>`) instead: the original is unrecoverable
/// from the digest (same tradeoff as pi's `shortHash`), but the mapping is stable, so a `ToolResult`
/// elsewhere in the session that still references the original combined id keeps pairing correctly.
fn combine_tool_id(call_id: &str, item_id: &str) -> String {
    let item_id = if item_id.len() > MAX_ITEM_ID_LEN {
        std::borrow::Cow::Owned(format!("fc_{}", short_hash(item_id)))
    } else {
        escape_tool_id_part(item_id)
    };
    format!("{}|{}", escape_tool_id_part(call_id), item_id)
}

/// OpenAI's documented cap on a `function_call` item id (mirrors pi's `normalizeIdPart`/
/// `buildForeignResponsesItemId`, both bounded to 64) — also the threshold past which
/// [`combine_tool_id`] treats an `item_id` as foreign/non-standard rather than carrying it through
/// verbatim.
const MAX_ITEM_ID_LEN: usize = 64;

/// Fast, deterministic, one-way digest for bounding an oversized tool-call item id down to a short,
/// `[0-9a-f]`-only token — mirrors pi's `shortHash` (`packages/ai/src/utils/hash.ts`). Not reversible:
/// nothing recovers `s` from the output, only a stable `s -> digest` mapping, which is all replay
/// pairing needs (the same foreign `item_id` always collapses to the same digest). `DefaultHasher` is
/// keyed with fixed (not per-process-random) state, so this is deterministic across calls and process
/// restarts alike — required here since the digest, once computed, is persisted as part of the
/// session's `ToolUse.id`/`ToolResult.tool_use_id` pairing rather than recomputed later. `pub(super)`:
/// shared with `dialect::openai`'s Mistral tool-call-id reshaping, which needs an identical
/// deterministic digest for its own hash-fallback case.
pub(super) fn short_hash(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn escape_tool_id_part(s: &str) -> std::borrow::Cow<'_, str> {
    if s.contains(['\\', '|']) {
        std::borrow::Cow::Owned(s.replace('\\', "\\\\").replace('|', "\\|"))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

fn unescape_tool_id_part(s: &str) -> std::borrow::Cow<'_, str> {
    if !s.contains('\\') {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
                continue;
            }
        }
        out.push(c);
    }
    std::borrow::Cow::Owned(out)
}

/// Find the first *unescaped* `|` in `s` — one not part of a `\|` (or preceded by a `\\` that already
/// consumed the backslash before it) — and split there.
fn split_on_unescaped_separator(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            // `combine_tool_id`'s escaping only ever produces `\\` or `\|`, both two ASCII bytes, so
            // skipping two bytes here always lands back on a real boundary.
            b'\\' if i + 1 < bytes.len() => i += 2,
            b'|' => return Some((&s[..i], &s[i + 1..])),
            _ => i += 1,
        }
    }
    None
}

/// Split a combined id back into `(call_id, item_id)`, unescaping each half. No unescaped `|` (a plain
/// id from another dialect, or a call whose item id we chose not to capture) means there's no item id
/// to replay. An `item_id` that [`combine_tool_id`] replaced with a digest round-trips like any other
/// opaque string here — the digest is plain `[0-9a-f]`, so there's nothing to unescape, and (being
/// one-way) the original foreign id it stood in for is simply not recoverable.
fn split_tool_id(id: &str) -> (std::borrow::Cow<'_, str>, Option<std::borrow::Cow<'_, str>>) {
    match split_on_unescaped_separator(id) {
        Some((call_id, item_id)) => (
            unescape_tool_id_part(call_id),
            (!item_id.is_empty()).then(|| unescape_tool_id_part(item_id)),
        ),
        None => (std::borrow::Cow::Borrowed(id), None),
    }
}

/// Truncate a combined `"call_id|item_id"` tool-call id back down to just `call_id` — used when a
/// message produced by an OpenAI-Responses model is about to be replayed to a *different* model (see
/// [`crate::session::Session::scrub_cross_model_state`]): the `item_id` half only means anything back
/// to the model/reasoning-item pairing that produced it, and a foreign model would either ignore it or
/// reject the combined id outright. A no-op for a plain id with no unescaped `|` (already just a
/// `call_id`, or a dialect that never combines ids in the first place).
pub(crate) fn call_id_only(id: &str) -> String {
    split_tool_id(id).0.into_owned()
}

/// The system/developer-prompt role: `"developer"` for reasoning models (matching pi, which prefers
/// it whenever the model supports it), `"system"` otherwise. `pub(super)`: shared with
/// `dialect::openai::build_body`, which needs the identical gating for the Chat Completions dialect's
/// system-prompt role (pi's `useDeveloperRole = model.reasoning && compat.supportsDeveloperRole`).
///
/// `compat.supportsDeveloperRole` isn't purely reasoning-gated in pi, though — it's `false` for the
/// same `isNonStandard` Chat-Completions-family denylist `supportsStore` uses (`openai-completions.ts`
/// `detectCompat`: DeepSeek, Z.ai/GLM, Moonshot/Kimi, xAI/Grok, Cerebras's native ids, Together, NVIDIA,
/// Cloudflare, Ant-Ling), regardless of whether the model itself is reasoning-capable — these
/// non-standard Chat Completions providers reject or misbehave on `role:"developer"` even when their
/// hosted model has an OpenAI-style reasoning-effort knob (e.g. DeepSeek's/Cerebras's own
/// `gpt-oss-120b`, `gemma-4-31b`, `zai-glm-4.7`, which get `role:"developer"` today though pi always
/// sends them `"system"`). `is_non_standard_store_provider` is the same by-id-shape recognition
/// `supportsStore` already needed for the identical `isNonStandard` boolean — reused rather than
/// duplicated (its own doc comment notes NVIDIA/Cloudflare/Ant-Ling can't be told apart from a generic
/// third-party id by shape alone; same known gap here). Only reached through the Chat Completions
/// dialect in practice — the Responses API (this dialect) only ever routes to OpenAI/Azure/Copilot/
/// Codex, none of which are in the denylist — but the gate lives here since both dialects share this
/// one function.
pub(super) fn instruction_role(model: &str, caps: &crate::models::ModelCaps) -> &'static str {
    if caps.reasoning_effort && !crate::models::is_non_standard_store_provider(model) {
        "developer"
    } else {
        "system"
    }
}

/// Placeholder text substituted for an image sent to a model that doesn't accept one — shared string
/// with the Chat Completions and Anthropic dialects.
const USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
/// Same idea, for a tool result's image output specifically.
const TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

/// pi-parity fix: collapse only a consecutive *run* of non-vision images into one placeholder, not
/// every image anywhere in the turn via a single turn-wide boolean gate — mirrors the Anthropic
/// dialect's per-run state machine (`dialect::anthropic`'s image-collapse pass, `anthropic.rs:282-301`).
/// A single `had_image` flag checked once at the end loses both the position (the placeholder always
/// landed last, even if the image was first) and the grouping (two runs of images separated by text
/// wrongly collapsed into one placeholder instead of two). `pending_placeholder` tracks whether the run
/// currently being scanned still needs its placeholder flushed; it flushes right before any block that
/// isn't part of that run (and once more after the loop, for a run that was still open at the end).
fn push_user_content(input: &mut Vec<Value>, blocks: &[ContentBlock], supports_vision: bool) {
    let mut parts: Vec<Value> = Vec::new();
    let mut pending_placeholder = false;
    for b in blocks {
        if matches!(b, ContentBlock::Image { .. }) && !supports_vision {
            // Still inside a run of non-vision images — extend it rather than flushing yet.
            pending_placeholder = true;
            continue;
        }
        if pending_placeholder {
            parts.push(json!({ "type": "input_text", "text": USER_IMAGE_PLACEHOLDER }));
            pending_placeholder = false;
        }
        match b {
            ContentBlock::Text { text, .. } if !text.is_empty() => {
                parts.push(json!({ "type": "input_text", "text": text }));
            }
            // supports_vision is always true here — the false case was already handled above.
            ContentBlock::Image { source } => {
                parts.push(json!({
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": format!("data:{};base64,{}", source.media_type, source.data),
                }));
            }
            _ => {}
        }
    }
    if pending_placeholder {
        parts.push(json!({ "type": "input_text", "text": USER_IMAGE_PLACEHOLDER }));
    }
    if !parts.is_empty() {
        input.push(json!({ "role": "user", "content": parts }));
    }
}

/// Fan a turn's `ToolResult` blocks out into `function_call_output` items. Images ride directly in
/// `output` as a content-parts list (the Responses API supports this natively — unlike Chat
/// Completions' `tool` role, which can't carry images at all) — or, when the model can't accept
/// images, a text placeholder instead.
fn push_tool_results(input: &mut Vec<Value>, blocks: &[ContentBlock], supports_vision: bool) {
    for b in blocks {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            images,
            ..
        } = b
        {
            let (call_id, _item_id) = split_tool_id(tool_use_id);
            let output = if images.is_empty() {
                json!(content)
            } else if !supports_vision {
                let mut text = content.clone();
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(TOOL_IMAGE_PLACEHOLDER);
                json!(text)
            } else {
                let mut parts: Vec<Value> = Vec::new();
                if !content.is_empty() {
                    parts.push(json!({ "type": "input_text", "text": content }));
                }
                for source in images {
                    parts.push(json!({
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": format!("data:{};base64,{}", source.media_type, source.data),
                    }));
                }
                Value::Array(parts)
            };
            input.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }));
        }
    }
}

/// A message item's replay id when the block never captured a real one from the wire (locally
/// authored text — a compaction/branch summary — or a session persisted before this field existed).
/// Deterministic and stable across rebuilds of the same request so it doesn't churn the prompt-cache
/// prefix turn to turn: `msg_{msg_index}` for a message's first text block, `msg_{msg_index}_{n}` for
/// any further one (a model rarely emits more than one text block per turn, but nothing stops it).
fn fallback_message_id(msg_index: usize, text_block_index: usize) -> String {
    if text_block_index == 0 {
        format!("msg_{msg_index}")
    } else {
        format!("msg_{msg_index}_{text_block_index}")
    }
}

/// Push one assistant `message` item, stamping the `id`/`status`/`phase` OpenAI requires to replay it
/// (see `ContentBlock::Text`'s doc comment) — `id` real when the block carries one from the wire,
/// else the deterministic fallback; `status` always `"completed"` (the only status a *replayed*, i.e.
/// already-finished, block can have); `phase` passed through only when the original block had one —
/// omitting it is fine wire-wise, but OpenAI's own docs say dropping it on replay for gpt-5.3-codex
/// and later degrades those models, so a captured phase is never silently dropped.
fn push_text_message(
    input: &mut Vec<Value>,
    text: &str,
    id: Option<&str>,
    phase: Option<&str>,
    msg_index: usize,
    text_block_index: usize,
) {
    let mut obj = Map::new();
    obj.insert("type".into(), json!("message"));
    obj.insert("role".into(), json!("assistant"));
    obj.insert(
        "content".into(),
        json!([{ "type": "output_text", "text": text }]),
    );
    obj.insert("status".into(), json!("completed"));
    let id = match id {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => fallback_message_id(msg_index, text_block_index),
    };
    obj.insert("id".into(), json!(id));
    if let Some(phase) = phase {
        obj.insert("phase".into(), json!(phase));
    }
    input.push(Value::Object(obj));
}

fn push_assistant_content(input: &mut Vec<Value>, blocks: &[ContentBlock], msg_index: usize) {
    let mut text_block_index = 0;
    for b in blocks {
        match b {
            ContentBlock::Text { text, id, phase } if !text.is_empty() => {
                push_text_message(
                    input,
                    text,
                    id.as_deref(),
                    phase.as_deref(),
                    msg_index,
                    text_block_index,
                );
                text_block_index += 1;
            }
            ContentBlock::Thinking { text, signature } => {
                if !signature.is_empty() {
                    if let Ok(item) = serde_json::from_str::<Value>(signature) {
                        input.push(item);
                        continue;
                    }
                }
                // Non-JSON signature: a cross-model replay after `set_model`'s thinking scrub, or a
                // genuinely foreign block. Can't be replayed as a reasoning item — degrade to plain
                // text so the content isn't silently dropped (mirrors pi's `isSameModel` downgrade).
                if !text.is_empty() {
                    push_text_message(input, text, None, None, msg_index, text_block_index);
                    text_block_index += 1;
                }
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: args,
                // Only the OpenAI Chat Completions dialect ever populates this (Gemini-style
                // `reasoning_details` continuity — see `ContentBlock::ToolUse`'s doc comment); no
                // Responses wire slot exists for it, so it's always `None` here and ignored.
                thought_signature: _,
            } => {
                let (call_id, item_id) = split_tool_id(id);
                let mut obj = Map::new();
                obj.insert("type".into(), json!("function_call"));
                if let Some(item_id) = item_id {
                    obj.insert("id".into(), json!(item_id));
                }
                obj.insert("call_id".into(), json!(call_id));
                obj.insert("name".into(), json!(name));
                obj.insert(
                    "arguments".into(),
                    json!(serde_json::to_string(args).unwrap_or_else(|_| "{}".into())),
                );
                input.push(Value::Object(obj));
            }
            // RedactedThinking has no OpenAI equivalent and no visible text to degrade to (unlike a
            // non-JSON-signature Thinking block above) — nothing safe to replay, so it's dropped.
            _ => {}
        }
    }
}

/// Build the streaming request body.
pub fn build_body(req: &ModelRequest) -> Value {
    let caps = crate::models::capabilities_for_route(&req.model, req.is_codex, req.is_azure);
    let mut input: Vec<Value> = Vec::new();

    // Codex/ChatGPT's own backend wants the system prompt carried in a separate top-level
    // `instructions` field (below) instead of folded into `input[0]` — every other route keeps this
    // vanilla native-OpenAI-Responses shape. See `req.is_codex`'s own doc comment.
    if !req.is_codex {
        if let Some(system) = &req.system {
            input.push(json!({ "role": instruction_role(&req.model, &caps), "content": system }));
        }
    }
    for (msg_index, m) in req.messages.iter().enumerate() {
        match m.role {
            Role::System => {
                input.push(json!({
                    "role": instruction_role(&req.model, &caps),
                    "content": text_of(&m.content),
                }));
            }
            Role::User => {
                push_user_content(&mut input, &m.content, caps.supports_vision);
                push_tool_results(&mut input, &m.content, caps.supports_vision);
            }
            Role::Assistant => push_assistant_content(&mut input, &m.content, msg_index),
        }
    }

    let mut map = Map::new();
    map.insert("model".into(), json!(req.model));
    map.insert("input".into(), Value::Array(input));
    map.insert("stream".into(), json!(true));
    // Stateless harness: every turn resends the full history, so nothing should be retained
    // server-side to reference via `previous_response_id`.
    map.insert("store".into(), json!(false));
    if req.is_codex {
        // Mirrors pi's `openai-codex-responses.ts` `buildRequestBody`: the system prompt rides in a
        // top-level `instructions` field (never folded into `input` for this route — see above),
        // falling back to the same default pi sends when no system prompt was configured at all, so a
        // Codex-routed turn is never sent with no `instructions` field at all. `parallel_tool_calls`
        // and `text.verbosity` are also always sent for this backend specifically, not gated on
        // anything the caller configured — beyond has no equivalent verbosity option yet, hence the
        // hardcoded `"low"` pi itself defaults to.
        map.insert(
            "instructions".into(),
            json!(
                req.system
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("You are a helpful assistant.")
            ),
        );
        map.insert("parallel_tool_calls".into(), json!(true));
        map.insert("text".into(), json!({ "verbosity": "low" }));
        // Codex's backend always wants encrypted reasoning content included, whether or not this turn
        // requested a specific effort — pi's `buildRequestBody` puts `include: ["reasoning.encrypted_
        // content"]` on the base request object unconditionally (`openai-codex-responses.ts:495`), not
        // only inside its `reasoningEffort !== undefined` branch. The `reasoning_effort` block below
        // re-sets the same key/value when an effort *is* requested — a harmless overwrite, not a
        // second, conflicting source of truth.
        map.insert("include".into(), json!(["reasoning.encrypted_content"]));
    }
    // A queueing/latency class, not purely a pricing knob — omitted entirely (leaving OpenAI's own
    // default tier) unless the caller explicitly asked for one.
    if let Some(tier) = req.service_tier {
        map.insert("service_tier".into(), json!(tier.as_str()));
    }
    // Clamped so the estimated prompt plus this ceiling doesn't already exceed the model's context
    // window — see `super::clamp_max_tokens_to_context`'s doc comment for why this can't be skipped.
    map.insert(
        "max_output_tokens".into(),
        json!(super::clamp_max_tokens_to_context(req, &caps)),
    );
    // Sent unconditionally when set — matches pi's own unconditional `openai-responses.ts` (a
    // reasoning model that rejects a custom temperature is a caller error, same as pi's).
    if let Some(temperature) = req.temperature {
        map.insert("temperature".into(), json!(temperature));
    }

    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                })
            })
            .collect();
        map.insert("tools".into(), Value::Array(tools));
    }
    if let Some(choice) = &req.tool_choice {
        map.insert("tool_choice".into(), tool_choice(choice));
    }

    // Reasoning models only: an explicit effort level requests visible summaries and asks for the
    // reasoning item's encrypted content, which is what makes the block replayable next turn (without
    // it, the reasoning item can't be sent back and cross-turn reasoning continuity is lost). When no
    // effort is requested but the model can be told to turn reasoning off explicitly, do so — rather
    // than silently reasoning at the provider's own (non-zero) default effort.
    if caps.reasoning_effort {
        if let Some(effort) = req.reasoning_effort {
            let effort = crate::models::clamp_reasoning_effort(&caps, effort);
            let summary = req
                .reasoning_summary
                .unwrap_or(crate::transport::ReasoningSummary::Auto);
            map.insert(
                "reasoning".into(),
                json!({ "effort": effort.as_str(), "summary": summary.as_str() }),
            );
            map.insert("include".into(), json!(["reasoning.encrypted_content"]));
        } else if caps.reasoning_disableable && !req.is_copilot && !req.is_codex {
            // GitHub Copilot-hosted gpt-5.x ids set `thinkingLevelMap: {"off": null}` in pi's own
            // catalogue (`github-copilot.models.ts`) — no explicit "off" wire shape at all — even
            // though `ModelCaps::reasoning_disableable` (keyed purely by id) reports the *same* id as
            // disable-capable when reached via native OpenAI directly. Mirrors pi's own gate
            // (`model.provider !== "github-copilot" && model.thinkingLevelMap?.off !== null`,
            // `openai-responses.ts`). See `ModelRequest::is_copilot`'s own doc comment: nothing sets it
            // to `true` yet — this is the consuming half of that plumbing point.
            //
            // Codex is excluded too, but for a different reason than Copilot: pi's Codex-specific
            // `streamSimple` (`openai-codex-responses.ts:466-467`) maps a clamped "off" effort to
            // `reasoningEffort: undefined`, and `buildRequestBody` only ever sets `body.reasoning` when
            // `options?.reasoningEffort !== undefined` — there is no Codex code path that ever sends an
            // explicit `reasoning: {effort: "none"}` the way the native Responses dialect does. A
            // no-thinking-requested Codex turn on a reasoning-disableable model must omit `reasoning`
            // entirely, not send this synthetic "off" beyond invented for the native route.
            map.insert("reasoning".into(), json!({ "effort": "none" }));
        }
    }

    // Prompt-cache affinity, same as the Chat Completions dialect: OpenAI caches prefixes
    // automatically, so this is only a routing hint. Still gated on `!req.no_cache`, though —
    // `ModelRequest::no_cache`'s own doc comment promises to skip OpenAI's
    // `prompt_cache_key`/`prompt_cache_retention` too (equivalently to Anthropic's `cache_control`),
    // matching pi's `cacheRetention === "none"` check (`openai-completions.ts`): a genuinely one-off
    // request has no follow-up turn to route back to the same cache node, so the affinity hint is
    // pointless even though sending it wouldn't itself cost a cache-write premium here.
    if !req.no_cache {
        if let Some(key) = &req.cache_key {
            map.insert(
                "prompt_cache_key".into(),
                json!(super::openai::clamp_prompt_cache_key(key)),
            );
        }
    }
    // pi's `azure-openai-responses.ts` never calls `resolveCacheRetention`/`getPromptCacheRetention` at
    // all — only the direct `openai-responses.ts` dialect opts into the 24h retention tier. See
    // `ModelRequest::is_azure`'s own doc comment: nothing sets it to `true` yet — this is the consuming
    // half of that plumbing point.
    if req.cache_long && caps.supports_long_cache && !req.no_cache && !req.is_azure {
        map.insert("prompt_cache_retention".into(), json!("24h"));
    }

    Value::Object(map)
}

/// Map a [`ToolChoice`] to the Responses API's `tool_choice`. Auto/none/required are bare strings,
/// same as Chat Completions; a specific tool is flat (`{type:"function", name}`, no nested `function`
/// wrapper — tool *definitions* are flat here too, see `build_body`).
fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({ "type": "function", "name": name }),
    }
}

fn str_at<'a>(v: Option<&'a Value>, key: &str) -> &'a str {
    v.and_then(|v| v.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn i64_at(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(Value::as_i64).unwrap_or(-1)
}

fn u32_field(v: Option<&Value>, key: &str) -> u32 {
    v.and_then(|v| v.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32
}

/// Build the error message for a `response.failed` event, in the same `code: message` shape
/// `dialect::sse_error` uses for the top-level `error` event type.
fn failure_message(data: &Value) -> String {
    let response = data.get("response");
    if let Some(err) = response
        .and_then(|r| r.get("error"))
        .filter(|e| e.is_object())
    {
        let code = err.get("code").and_then(Value::as_str).unwrap_or("unknown");
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("no message");
        return format!("{code}: {msg}");
    }
    if let Some(reason) = response
        .and_then(|r| r.get("incomplete_details"))
        .and_then(|d| d.get("reason"))
        .and_then(Value::as_str)
    {
        return format!("incomplete: {reason}");
    }
    "unknown error (no error details in response)".to_string()
}

/// Concatenate a `message`-type item's `content` parts' text (`output_text`/`refusal` parts) — the
/// same text `response.output_text.delta`/`response.refusal.delta` events already streamed, used at
/// `output_item.done` as a resync ground truth. `None` when the item has no content array or no
/// text-bearing parts (nothing to resync against).
fn message_item_text(item: Option<&Value>) -> Option<String> {
    let parts = item?.get("content")?.as_array()?;
    let mut text = String::new();
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => {
                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
            }
            Some("refusal") => {
                text.push_str(part.get("refusal").and_then(Value::as_str).unwrap_or(""));
            }
            _ => {}
        }
    }
    (!text.is_empty()).then_some(text)
}

/// Join a `reasoning`-type item's text-bearing parts (`summary` or `content`, both `[{..., "text": ...}]`
/// shaped) with a blank-line separator — mirrors pi's `item.summary?.map((s) => s.text).join("\n\n")`.
fn join_reasoning_parts(parts: &[Value]) -> String {
    parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// A `reasoning`-type item's visible thinking text — its `summary` parts (visible reasoning summaries,
/// the common case) if any are non-empty, else its `content` parts (raw reasoning text, some models'
/// more verbose channel) — used at `output_item.done` as a resync ground truth, same purpose as
/// [`message_item_text`] for a text/refusal item. Mirrors pi's `summaryText || contentText` fallback
/// (`openai-responses-shared.ts`'s `output_item.done` handler). `None` when neither field yields any
/// text — nothing to resync against, so the block's accumulated `ThinkingDelta`s stand as-is.
fn reasoning_item_text(item: Option<&Value>) -> Option<String> {
    let item = item?;
    let summary = item
        .get("summary")
        .and_then(Value::as_array)
        .map(|parts| join_reasoning_parts(parts))
        .unwrap_or_default();
    if !summary.is_empty() {
        return Some(summary);
    }
    let content = item
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| join_reasoning_parts(parts))
        .unwrap_or_default();
    (!content.is_empty()).then_some(content)
}

/// A `message`-type item's `id` and `phase` (see `ContentBlock::Text`'s doc comment) — captured at
/// `output_item.done` alongside its authoritative text so the block can be restamped verbatim on
/// replay. Both are simply absent on models with no channel concept; `phase` is `None` there too.
fn message_item_id_phase(item: Option<&Value>) -> (Option<String>, Option<String>) {
    let id = item
        .and_then(|i| i.get("id"))
        .and_then(Value::as_str)
        .map(String::from);
    let phase = item
        .and_then(|i| i.get("phase"))
        .and_then(Value::as_str)
        .map(String::from);
    (id, phase)
}

/// Decodes Responses SSE. Items are genuine block boundaries (`output_item.added`/`.done`); every
/// event is emitted with its own true `output_index` and forwarded immediately — no buffering, no
/// "focus" concept (see the module doc comment). `open_indices` (insertion-ordered, so tests and
/// `finalize`'s defensive close-everything-left-open path stay deterministic) exists purely so this
/// decoder knows which indices are still open, for two purposes: (1) a stray `done`/delta for an index
/// that was never opened is recognized as stale/duplicate and dropped, and (2) `finalize` can close out
/// anything a malformed/truncated stream left open.
pub struct Decoder {
    started: bool,
    open_indices: Vec<i64>,
    saw_terminal: bool,
    failed: Option<String>,
    saw_tool_call: bool,
    stop_reason: StopReason,
    usage: TokenUsage,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            started: false,
            open_indices: Vec::new(),
            saw_terminal: false,
            failed: None,
            saw_tool_call: false,
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }
    }
}

impl Decoder {
    /// Emit `ev` for `index`, marking it open (a no-op if it already is).
    fn emit(&mut self, out: &mut Vec<StreamEvent>, index: i64, ev: StreamEvent) {
        if !self.open_indices.contains(&index) {
            self.open_indices.push(index);
        }
        out.push(ev);
    }

    /// Close `index`, if it's actually open — a `done` for an index never opened is a stale/duplicate
    /// event, silently ignored rather than emitting a spurious close for a block that doesn't exist.
    fn close(&mut self, out: &mut Vec<StreamEvent>, index: i64) {
        if let Some(pos) = self.open_indices.iter().position(|&i| i == index) {
            self.open_indices.remove(pos);
            out.push(StreamEvent::ContentBlockStop {
                index: index as usize,
            });
        }
    }

    fn finalize(&mut self, data: &Value, out: &mut Vec<StreamEvent>) {
        self.saw_terminal = true;
        // Defensive: a `response.completed`/`response.incomplete` event's own embedded
        // `response.status` can in principle also read `"failed"`/`"cancelled"` (pi's
        // `mapStopReason` guards this exhaustively) rather than the failure only ever arriving via
        // the dedicated `response.failed` event — treated identically either way: no events emitted
        // here, `self.failed` set so `finish()` surfaces it as a hard `Err` instead of the generic
        // warn-and-treat-as-`Other` fallback below, which would otherwise report a genuine failure as
        // if the turn had ended cleanly.
        let status = data
            .get("response")
            .and_then(|r| r.get("status"))
            .and_then(Value::as_str);
        if matches!(status, Some("failed") | Some("cancelled")) {
            self.failed = Some(failure_message(data));
            return;
        }
        // Defensive: normally every item is closed via its own `output_item.done` before the terminal
        // event arrives — but a malformed/truncated stream must not silently drop whatever's still
        // open, at any index.
        for index in std::mem::take(&mut self.open_indices) {
            out.push(StreamEvent::ContentBlockStop {
                index: index as usize,
            });
        }

        let response = data.get("response");
        let usage = response.and_then(|r| r.get("usage"));
        let cached = u32_field(
            usage.and_then(|u| u.get("input_tokens_details")),
            "cached_tokens",
        );
        let input_tokens = u32_field(usage, "input_tokens");
        // Like the Chat Completions dialect: `input_tokens` is the whole prompt including cached
        // tokens, so bill the uncached remainder as `input_tokens` and report the cache hit
        // separately.
        self.usage.cache_read_tokens = cached;
        self.usage.input_tokens = input_tokens.saturating_sub(cached);
        self.usage.output_tokens = u32_field(usage, "output_tokens");
        self.usage.reasoning_tokens = u32_field(
            usage.and_then(|u| u.get("output_tokens_details")),
            "reasoning_tokens",
        );

        let mut stop_reason = match status {
            Some("completed") => StopReason::EndTurn,
            Some("incomplete") => StopReason::MaxTokens,
            // `"failed"`/`"cancelled"` already returned early above.
            Some(other) => {
                tracing::warn!(
                    status = other,
                    "unrecognized OpenAI Responses status; treating as Other"
                );
                StopReason::Other
            }
            None => StopReason::Other,
        };
        // The Responses API has no distinct "stopped to call a tool" status the way Anthropic/Chat
        // Completions do — a turn with tool calls still reports `status:"completed"` — so upgrade it
        // here the same way pi does.
        if stop_reason == StopReason::EndTurn && self.saw_tool_call {
            stop_reason = StopReason::ToolUse;
        }
        self.stop_reason = stop_reason;
        out.push(StreamEvent::Usage(self.usage));
        out.push(StreamEvent::MessageStop { stop_reason });
    }
}

impl StreamDecoder for Decoder {
    fn push(&mut self, data: &Value) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if !self.started {
            self.started = true;
            out.push(StreamEvent::MessageStart);
        }
        let kind = data.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "response.output_item.added" => {
                let index = i64_at(data, "output_index");
                let item = data.get("item");
                if item.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("function_call")
                {
                    self.saw_tool_call = true;
                    let call_id = str_at(item, "call_id");
                    let item_id = str_at(item, "id");
                    let name = str_at(item, "name").to_string();
                    let id = if item_id.is_empty() {
                        call_id.to_string()
                    } else {
                        combine_tool_id(call_id, item_id)
                    };
                    self.emit(
                        &mut out,
                        index,
                        StreamEvent::ToolUseStart {
                            index: index as usize,
                            id,
                            name,
                        },
                    );
                }
                // A message/reasoning item has no explicit "start" `StreamEvent` — it opens implicitly
                // on its first delta, which `emit` handles lazily (marking it open on first mention).
                // Nothing to emit here for those.
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let index = i64_at(data, "output_index");
                self.emit(
                    &mut out,
                    index,
                    StreamEvent::ThinkingDelta {
                        index: index as usize,
                        text: str_at(Some(data), "delta").to_string(),
                    },
                );
            }
            "response.reasoning_summary_part.done" => {
                let index = i64_at(data, "output_index");
                // Only meaningful for an index that's already open (accrued some reasoning text) —
                // otherwise there's nothing to add a paragraph break to.
                if self.open_indices.contains(&index) {
                    self.emit(
                        &mut out,
                        index,
                        StreamEvent::ThinkingDelta {
                            index: index as usize,
                            text: "\n\n".to_string(),
                        },
                    );
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let index = i64_at(data, "output_index");
                self.emit(
                    &mut out,
                    index,
                    StreamEvent::TextDelta {
                        index: index as usize,
                        text: str_at(Some(data), "delta").to_string(),
                    },
                );
            }
            "response.function_call_arguments.delta" => {
                let index = i64_at(data, "output_index");
                self.emit(
                    &mut out,
                    index,
                    StreamEvent::InputJsonDelta {
                        index: index as usize,
                        partial_json: str_at(Some(data), "delta").to_string(),
                    },
                );
            }
            "response.function_call_arguments.done" => {
                // The provider's own authoritative, complete arguments string — resyncs over whatever
                // the streamed deltas produced so far, so a single dropped/duplicated mid-stream delta
                // (a relay hiccup with no transport-level error — nothing else would ever catch it)
                // can't silently leave the final call's arguments corrupted.
                let index = i64_at(data, "output_index");
                self.emit(
                    &mut out,
                    index,
                    StreamEvent::InputJsonFinal {
                        index: index as usize,
                        full_json: str_at(Some(data), "arguments").to_string(),
                    },
                );
            }
            "response.output_item.done" => {
                let index = i64_at(data, "output_index");
                if !self.open_indices.contains(&index) {
                    // A `done` for an index we never saw open — duplicate/stale event, ignored.
                } else {
                    let item = data.get("item");
                    let item_type = item.and_then(|i| i.get("type")).and_then(Value::as_str);
                    match item_type {
                        Some("reasoning") => {
                            // The whole item, JSON-stringified, is the block's replayable signature —
                            // see the module doc comment.
                            self.emit(
                                &mut out,
                                index,
                                StreamEvent::SignatureDelta {
                                    index: index as usize,
                                    signature: item.map(ToString::to_string).unwrap_or_default(),
                                },
                            );
                            // Ground-truth resync, same purpose as the text/tool-call block resyncs
                            // above: a single dropped/duplicated mid-stream
                            // `reasoning_summary_text.delta`/`reasoning_text.delta` chunk (a relay
                            // hiccup with no transport-level error — nothing else would ever catch it)
                            // must not silently leave the persisted/displayed thinking text corrupted.
                            if let Some(text) = reasoning_item_text(item) {
                                self.emit(
                                    &mut out,
                                    index,
                                    StreamEvent::ThinkingFinal {
                                        index: index as usize,
                                        text,
                                    },
                                );
                            }
                        }
                        Some("function_call") => {
                            // Ground-truth resync, same as `function_call_arguments.done` above — belt
                            // and suspenders for a provider that only ever sends the item-level `done`
                            // and skips the dedicated per-field one.
                            if let Some(args) = item
                                .and_then(|i| i.get("arguments"))
                                .and_then(Value::as_str)
                            {
                                self.emit(
                                    &mut out,
                                    index,
                                    StreamEvent::InputJsonFinal {
                                        index: index as usize,
                                        full_json: args.to_string(),
                                    },
                                );
                            }
                        }
                        Some("message") => {
                            if let Some(text) = message_item_text(item) {
                                let (id, phase) = message_item_id_phase(item);
                                self.emit(
                                    &mut out,
                                    index,
                                    StreamEvent::TextFinal {
                                        index: index as usize,
                                        text,
                                        id,
                                        phase,
                                    },
                                );
                            }
                        }
                        Some(other) => {
                            // A genuinely new item type (the Responses API has several built-in
                            // server-side tool items this harness doesn't request today — file/web
                            // search, code interpreter, computer use, MCP calls) silently dropping the
                            // whole item with no signal would hide a provider capability change until a
                            // user notices missing content.
                            tracing::warn!(
                                item_type = other,
                                "unrecognized OpenAI Responses output_item.done item type; dropping"
                            );
                        }
                        None => {}
                    }
                    self.close(&mut out, index);
                }
            }
            // `response.done` is Codex/ChatGPT-OAuth's own backend-specific terminal event
            // (`RouteOverride::Prefixed`-routed requests, see `client.rs`) — functionally identical to
            // `response.completed`/`response.incomplete` (same embedded `response.status`/`usage`
            // shape), just a different event name for that one backend. Mirrors pi's `mapCodexEvents`
            // (`openai-codex-responses.ts`), which normalizes all three into `response.completed`
            // before handing off to the shared processor. Left unrecognized, this event used to fall
            // through the catch-all arm below and do nothing — silently hanging a Codex-routed turn
            // forever waiting for a terminal event that already arrived.
            "response.completed" | "response.incomplete" | "response.done" => {
                self.finalize(data, &mut out)
            }
            "response.failed" => {
                self.saw_terminal = true;
                self.failed = Some(failure_message(data));
            }
            // The always-first event of every response, carrying only the (unused) response id — a
            // real, expected no-op, not worth logging every time it fires.
            "response.created" => {}
            other => {
                // A genuinely new top-level event type (the Responses API has added streaming event
                // types before, and continues to) silently dropping the whole event with no signal
                // would hide a provider capability change until a user notices missing content.
                tracing::warn!(
                    event_type = other,
                    "unrecognized OpenAI Responses event type; dropping"
                );
            }
        }
        out
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        if !self.started {
            return Ok(Vec::new());
        }
        if let Some(msg) = &self.failed {
            return Err(Error::Transport(format!(
                "OpenAI Responses stream failed: {msg}"
            )));
        }
        if !self.saw_terminal {
            return Err(Error::Transport(
                "OpenAI Responses stream ended before a terminal response event".into(),
            ));
        }
        Ok(Vec::new())
    }

    fn is_terminal(&self) -> bool {
        self.saw_terminal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::decode_sse;
    use crate::message::{ImageSource, Message, ToolDef};

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
                    ContentBlock::ToolUse {
                        id: "call_1|fc_1".into(),
                        name: "get_weather".into(),
                        input: json!({ "city": "SF" }),
                        thought_signature: None,
                    },
                ]),
                Message::tool_result("call_1|fc_1", "72F", false),
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

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["max_output_tokens"], 256);
        assert_eq!(body["store"], false);
        // gpt-4o isn't a reasoning model — the system prompt uses "system", not "developer".
        assert_eq!(
            body["input"][0],
            json!({ "role": "system", "content": "be brief" })
        );
        assert_eq!(
            body["input"][1],
            json!({ "role": "user", "content": [{ "type": "input_text", "text": "weather?" }] })
        );
        assert_eq!(body["input"][2]["type"], "message");
        assert_eq!(body["input"][2]["content"][0]["text"], "checking");
        // No captured wire id (a locally-authored/plain-text block) still gets the fields OpenAI
        // requires to replay an assistant message item: `status: "completed"` always, and a
        // deterministic fallback `id` — never omitted just because nothing was captured to replay.
        assert_eq!(body["input"][2]["status"], "completed");
        assert_eq!(body["input"][2]["id"], "msg_1");
        assert!(
            body["input"][2].get("phase").is_none(),
            "no phase was ever captured for this block, so none should be invented on replay"
        );
        // ToolUse round-trips the combined id back into call_id + id.
        assert_eq!(body["input"][3]["type"], "function_call");
        assert_eq!(body["input"][3]["call_id"], "call_1");
        assert_eq!(body["input"][3]["id"], "fc_1");
        assert_eq!(body["input"][3]["arguments"], "{\"city\":\"SF\"}");
        // ToolResult fans out to a function_call_output keyed on the call_id half only.
        assert_eq!(
            body["input"][4],
            json!({ "type": "function_call_output", "call_id": "call_1", "output": "72F" })
        );
        // Tools are flat — no nested `function` wrapper.
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert!(body["tools"][0].get("function").is_none());
    }

    #[test]
    fn fallback_message_id_is_unique_across_multiple_text_blocks_in_one_turn() {
        // pi-parity coverage gap (A-M6): mirrors pi's `packages/ai/test/openai-responses-message-id
        // .test.ts:17-47` ("generates unique fallback message IDs for multiple text blocks in one
        // assistant turn"). `fallback_message_id`'s `msg_{msg_index}` / `msg_{msg_index}_{n}` scheme
        // was implemented but had no test proving 2+ text-producing blocks in a single turn (a model
        // rarely, but not never, emits more than one — see `push_assistant_content`'s doc comment)
        // actually get distinct, correctly-incrementing ids rather than colliding on the same one.
        let req = ModelRequest::new(
            "gpt-4o",
            vec![
                Message::user("hello"),
                Message::assistant(vec![
                    ContentBlock::text("first answer"),
                    ContentBlock::text("second answer"),
                ]),
            ],
            64,
        );
        let body = build_body(&req);

        // messages[1] is the assistant turn, so msg_index == 1: "msg_1" for the first text block,
        // "msg_1_1" for the second — never the same id reused for both.
        assert_eq!(body["input"][1]["type"], "message");
        assert_eq!(body["input"][1]["content"][0]["text"], "first answer");
        assert_eq!(body["input"][1]["id"], "msg_1");
        assert_eq!(body["input"][2]["type"], "message");
        assert_eq!(body["input"][2]["content"][0]["text"], "second answer");
        assert_eq!(body["input"][2]["id"], "msg_1_1");
        assert_ne!(
            body["input"][1]["id"], body["input"][2]["id"],
            "each text block's fallback id must be distinct"
        );
    }

    #[test]
    fn max_output_tokens_is_clamped_when_the_live_prompt_nears_the_context_window() {
        // HIGH pi-parity gap (fixed): `clamp_max_tokens_to_context` was implemented for the Anthropic
        // dialect only; this Responses dialect wrote `req.max_tokens` straight onto the wire unclamped
        // as `max_output_tokens`. "gpt-5": 400_000 context window. A ~350_000-token prompt (1_400_000
        // chars, the chars/4 estimator) leaves 400_000 - 350_000 - 4_096 (the shared margin) = 45_904
        // tokens of headroom — less than the request's own (artificially large) max_tokens.
        let big_text = "x".repeat(1_400_000);
        let req = ModelRequest::new("gpt-5", vec![Message::user(big_text)], 100_000);
        let body = build_body(&req);
        assert_eq!(
            body["max_output_tokens"], 45_904,
            "max_output_tokens must be clamped down to the actual remaining context headroom, not \
             sent as the static 100_000 ceiling regardless of how much of the window the prompt \
             already fills"
        );
    }

    #[test]
    fn max_output_tokens_is_unchanged_for_a_prompt_nowhere_near_the_context_window() {
        let req = ModelRequest::new("gpt-5", vec![Message::user("hi")], 8_192);
        let body = build_body(&req);
        assert_eq!(body["max_output_tokens"], 8_192);
    }

    #[test]
    fn reasoning_model_uses_developer_role_and_emits_reasoning_config() {
        use crate::transport::ReasoningEffort;
        let req = ModelRequest::new("o3-mini", vec![Message::user("hi")], 64)
            .with_system("be terse")
            .with_reasoning_effort(ReasoningEffort::High);
        let body = build_body(&req);
        assert_eq!(body["input"][0]["role"], "developer");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");

        // No effort set on a disable-capable reasoning model (o3-mini defaults to `true`) → an
        // explicit "off" signal, not silent reliance on the provider's own default effort.
        let req = ModelRequest::new("o3-mini", vec![Message::user("hi")], 64);
        assert_eq!(build_body(&req)["reasoning"]["effort"], "none");
        assert!(build_body(&req).get("include").is_none());

        // No effort set on a reasoning model that *isn't* disable-capable (bare "gpt-5" — not in the
        // gpt-5 allowlist) → no `reasoning` field at all, since there's no "off" wire shape to send.
        let req = ModelRequest::new("gpt-5", vec![Message::user("hi")], 64);
        assert!(build_body(&req).get("reasoning").is_none());

        // A non-reasoning model never emits `reasoning`, even with an effort set.
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_reasoning_effort(ReasoningEffort::High);
        assert!(build_body(&req).get("reasoning").is_none());
    }

    #[test]
    fn reasoning_summary_is_configurable_not_hardcoded_to_auto() {
        // MEDIUM pi-parity gap (fixed): `reasoning.summary` was hardcoded to `"auto"` with no
        // `ModelRequest` field to vary it; pi exposes `reasoningSummary` as a first-class caller option
        // defaulted to `"auto"` only when unset.
        use crate::transport::{ReasoningEffort, ReasoningSummary};

        let req = ModelRequest::new("o3-mini", vec![Message::user("hi")], 64)
            .with_reasoning_effort(ReasoningEffort::High)
            .with_reasoning_summary(ReasoningSummary::Detailed);
        assert_eq!(build_body(&req)["reasoning"]["summary"], "detailed");

        let req = ModelRequest::new("o3-mini", vec![Message::user("hi")], 64)
            .with_reasoning_effort(ReasoningEffort::High)
            .with_reasoning_summary(ReasoningSummary::Concise);
        assert_eq!(build_body(&req)["reasoning"]["summary"], "concise");

        // Omitted entirely: still defaults to "auto", matching pi's own default.
        let req = ModelRequest::new("o3-mini", vec![Message::user("hi")], 64)
            .with_reasoning_effort(ReasoningEffort::High);
        assert_eq!(build_body(&req)["reasoning"]["summary"], "auto");
    }

    #[test]
    fn service_tier_is_omitted_unless_explicitly_requested() {
        // MEDIUM pi-parity gap (fixed): pi forwards `OpenAIResponsesOptions.serviceTier` to
        // `params.service_tier` — a queueing/latency class (flex/priority), not purely a pricing knob
        // — but `ModelRequest` had no field to carry it at all.
        use crate::transport::ServiceTier;

        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64);
        assert!(build_body(&req).get("service_tier").is_none());

        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_service_tier(ServiceTier::Flex);
        assert_eq!(build_body(&req)["service_tier"], "flex");

        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_service_tier(ServiceTier::Priority);
        assert_eq!(build_body(&req)["service_tier"], "priority");
    }

    #[test]
    fn reasoning_effort_is_clamped_per_model() {
        use crate::transport::ReasoningEffort;
        // o3-mini has no "xhigh" wire value at all — must clamp down to "high" rather than send a
        // value the Responses API rejects for this model.
        let req = ModelRequest::new("o3-mini", vec![Message::user("hi")], 64)
            .with_reasoning_effort(ReasoningEffort::XHigh);
        assert_eq!(build_body(&req)["reasoning"]["effort"], "high");

        // gpt-5.5-pro rejects both "minimal" and "low" — must clamp up to "medium".
        let req = ModelRequest::new("gpt-5.5-pro", vec![Message::user("hi")], 64)
            .with_reasoning_effort(ReasoningEffort::Minimal);
        assert_eq!(build_body(&req)["reasoning"]["effort"], "medium");

        // gpt-5.2 does support xhigh — passes through unclamped.
        let req = ModelRequest::new("gpt-5.2", vec![Message::user("hi")], 64)
            .with_reasoning_effort(ReasoningEffort::XHigh);
        assert_eq!(build_body(&req)["reasoning"]["effort"], "xhigh");
    }

    #[test]
    fn thinking_block_replays_verbatim_when_signature_is_json() {
        let item = json!({ "type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque" });
        let req = ModelRequest::new(
            "o3-mini",
            vec![
                Message::user("solve it"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: "step one".into(),
                        signature: item.to_string(),
                    },
                    ContentBlock::text("42"),
                ]),
            ],
            64,
        );
        let body = build_body(&req);
        assert_eq!(body["input"][1], item);
        assert_eq!(body["input"][2]["content"][0]["text"], "42");
    }

    #[test]
    fn assistant_text_replay_stamps_the_wires_own_captured_id_and_phase_verbatim() {
        // LOW pi-parity gap (fixed): a text block's `id`/`phase` — captured from a prior
        // `output_item.done` (see `text_item_resyncs_...`/`text_items_phase_is_captured_...` below) —
        // must round-trip onto the *next* request's replayed message item exactly as OpenAI sent it,
        // not be silently dropped. OpenAI's own docs: dropping `phase` on replay measurably degrades
        // gpt-5.3-codex and later.
        let req = ModelRequest::new(
            "gpt-5.3-codex",
            vec![
                Message::user("keep going"),
                Message::assistant(vec![ContentBlock::Text {
                    text: "still working on it".into(),
                    id: Some("msg_real_abc123".into()),
                    phase: Some("commentary".into()),
                }]),
            ],
            64,
        );
        let body = build_body(&req);
        assert_eq!(body["input"][1]["type"], "message");
        assert_eq!(body["input"][1]["id"], "msg_real_abc123");
        assert_eq!(body["input"][1]["status"], "completed");
        assert_eq!(body["input"][1]["phase"], "commentary");
        assert_eq!(
            body["input"][1]["content"][0]["text"],
            "still working on it"
        );
    }

    #[test]
    fn text_items_phase_is_captured_from_output_item_done_alongside_its_id() {
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"ok"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","phase":"final_answer","content":[{"type":"output_text","text":"ok"}]}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            events.contains(&StreamEvent::TextFinal {
                index: 0,
                text: "ok".into(),
                id: Some("msg_1".into()),
                phase: Some("final_answer".into()),
            }),
            "expected the item's phase captured alongside its id and text: {events:#?}"
        );
    }

    #[test]
    fn thinking_block_with_non_json_signature_degrades_to_text() {
        let req = ModelRequest::new(
            "o3-mini",
            vec![
                Message::user("solve it"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: "leftover reasoning".into(),
                        signature: "not-json".into(),
                    },
                    ContentBlock::text("42"),
                ]),
            ],
            64,
        );
        let body = build_body(&req);
        // Degrades to a plain text message rather than a raw reasoning item or being dropped.
        assert_eq!(body["input"][1]["type"], "message");
        assert_eq!(body["input"][1]["content"][0]["text"], "leftover reasoning");
        assert_eq!(body["input"][2]["content"][0]["text"], "42");
    }

    #[test]
    fn user_images_become_input_image_parts() {
        let req = ModelRequest::new(
            "gpt-4o",
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            64,
        );
        let body = build_body(&req);
        let content = &body["input"][0]["content"];
        assert_eq!(
            content[0],
            json!({ "type": "input_text", "text": "what is this?" })
        );
        assert_eq!(
            content[1],
            json!({
                "type": "input_image",
                "detail": "auto",
                "image_url": "data:image/png;base64,AAAA",
            })
        );
    }

    #[test]
    fn tool_result_images_embed_directly_in_output() {
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
        assert_eq!(body["input"][0]["type"], "function_call_output");
        assert_eq!(body["input"][0]["call_id"], "call_1");
        assert_eq!(body["input"][0]["output"][0]["text"], "screenshot attached");
        assert_eq!(body["input"][0]["output"][1]["type"], "input_image");
        // pi-parity strengthening (A-L3): the content-block TYPE alone doesn't prove the image data
        // actually made it onto the wire — assert the `image_url` value itself is the expected
        // data URI, prefixed with the source's real media type and carrying its base64 payload
        // verbatim, not just "some string is present at this key".
        assert_eq!(
            body["input"][0]["output"][1]["image_url"],
            "data:image/png;base64,AAAA"
        );
        assert!(
            body["input"][0]["output"][1]["image_url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,"),
            "image_url must carry the base64 data URI prefix, not just resemble one"
        );
    }

    #[test]
    fn a_tool_result_with_only_an_image_and_no_text_omits_the_empty_text_output_part() {
        // A-L8 pi-parity test gap (fixed, `packages/ai/test/image-tool-result.test.ts`): a tool that
        // returns only an image (no text) must not pad `output` with a spurious empty text part ahead
        // of the real image.
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
        let output = body["input"][0]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "expected only the image part: {output:#?}");
        assert_eq!(output[0]["type"], "input_image");
        assert_eq!(output[0]["image_url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn images_are_downgraded_to_a_placeholder_for_a_non_vision_model() {
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

        // User turn: text part plus one placeholder text part, no `input_image` part at all.
        let content = &body["input"][0]["content"];
        assert_eq!(
            content[0],
            json!({ "type": "input_text", "text": "what is this?" })
        );
        assert_eq!(
            content[1],
            json!({ "type": "input_text", "text": "(image omitted: model does not support images)" })
        );
        assert_eq!(content.as_array().unwrap().len(), 2);

        // Tool result: a plain string output with the placeholder appended, not a parts array.
        assert_eq!(
            body["input"][1]["output"],
            "screenshot attached\n(tool image omitted: model does not support images)"
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
        assert_eq!(body["input"][0]["content"][1]["type"], "input_image");
    }

    #[test]
    fn non_vision_model_collapses_only_consecutive_image_runs_not_the_whole_turn() {
        // pi-parity fix (A-M11): a single turn-wide `had_image` boolean used to collapse *every*
        // image in the turn into one placeholder appended at the very end, regardless of where the
        // images actually sat relative to text. That both loses position and wrongly merges two
        // distinct runs into one. Mirrors `dialect::anthropic`'s per-run state machine
        // (`anthropic.rs:282-301`): text - image - image - text - image - text should produce one
        // placeholder per *run* of images, each positioned right after the text that preceded it, not
        // a single placeholder tacked on at the end.
        let req = ModelRequest::new(
            "o3-mini", // the one o-series id that isn't vision-capable.
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
                usage: None,
                stop_reason: None,
            }],
            64,
        );
        let body = build_body(&req);
        let content = body["input"][0]["content"].as_array().unwrap();

        let placeholder = json!({
            "type": "input_text",
            "text": "(image omitted: model does not support images)"
        });
        assert_eq!(
            content,
            &vec![
                json!({ "type": "input_text", "text": "first" }),
                placeholder.clone(),
                json!({ "type": "input_text", "text": "middle" }),
                placeholder,
                json!({ "type": "input_text", "text": "last" }),
            ],
            "two images then text then one image must produce two placeholders in their own \
             positions — one per consecutive run — not a single placeholder for the whole turn"
        );
        // No `input_image` part should appear at all — every image in this turn is behind the
        // non-vision gate.
        assert!(
            content.iter().all(|p| p["type"] != "input_image"),
            "a non-vision model must never receive a real input_image part"
        );
    }

    #[test]
    fn tool_choice_is_flat_no_nested_function_wrapper() {
        let tools = vec![ToolDef {
            name: "get_weather".into(),
            description: "weather".into(),
            input_schema: json!({ "type": "object" }),
        }];
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_tools(tools)
            .with_tool_choice(ToolChoice::Tool("get_weather".into()));
        assert_eq!(
            build_body(&req)["tool_choice"],
            json!({ "type": "function", "name": "get_weather" })
        );
    }

    #[test]
    fn cache_long_emits_24h_retention_when_supported() {
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_cache_long(true);
        assert_eq!(build_body(&req)["prompt_cache_retention"], "24h");
    }

    #[test]
    fn azure_routed_requests_never_get_prompt_cache_retention() {
        // pi-parity (pass 15): pi's `azure-openai-responses.ts` never calls
        // `resolveCacheRetention`/`getPromptCacheRetention` at all — only the direct
        // `openai-responses.ts` dialect opts into the 24h tier. `ModelRequest::is_azure` is wired up by
        // `GatewayClient::stream` (see its own doc comment); `client.rs`'s
        // `direct_route_with_custom_auth_header_sends_bare_key_and_omits_authorization`-style tests
        // cover the end-to-end path, this test covers the dialect-side gate directly.
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_cache_long(true)
            .with_azure(true);
        assert!(
            build_body(&req).get("prompt_cache_retention").is_none(),
            "got: {:#?}",
            build_body(&req).get("prompt_cache_retention")
        );
        // Non-Azure requests are unaffected — same assertion as `cache_long_emits_24h_retention_when_supported`.
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_cache_long(true);
        assert_eq!(build_body(&req)["prompt_cache_retention"], "24h");
    }

    #[test]
    fn copilot_routed_gpt5_ids_never_get_an_explicit_reasoning_disable() {
        // pi-parity (pass 15): Copilot-hosted gpt-5.x ids set `thinkingLevelMap: {"off": null}` in
        // pi's own catalogue (`github-copilot.models.ts`) — no explicit "off" wire shape — even though
        // the identical id routed directly through OpenAI does have one. `ModelCaps` is keyed purely by
        // id and can't tell the two routes apart; `ModelRequest::is_copilot` is wired up by
        // `GatewayClient::stream` (see its own doc comment) from the same `via_copilot` signal used for
        // dialect selection — this test covers the dialect-side gate directly.
        let req = ModelRequest::new("gpt-5.2", vec![Message::user("hi")], 64).with_copilot(true);
        assert!(
            build_body(&req).get("reasoning").is_none(),
            "got: {:#?}",
            build_body(&req).get("reasoning")
        );
        // Direct OpenAI routing (the default) is unaffected — still gets the explicit "none".
        let req = ModelRequest::new("gpt-5.2", vec![Message::user("hi")], 64);
        assert_eq!(build_body(&req)["reasoning"]["effort"], "none");
    }

    #[test]
    fn codex_routed_requests_never_send_reasoning_when_none_was_requested() {
        // pi-parity (pass 17): pi's Codex-specific `streamSimple` (`openai-codex-responses.ts:466-467`)
        // maps a clamped "off" effort to `reasoningEffort: undefined`, and `buildRequestBody` only ever
        // sets `body.reasoning` inside its `options?.reasoningEffort !== undefined` branch — there is no
        // Codex code path that ever sends the explicit `reasoning: {effort: "none"}` the native
        // Responses dialect sends a reasoning-disableable model. Beyond's shared reasoning block used to
        // be unconditional for `is_codex` too, so a no-thinking-requested Codex turn sent a `reasoning`
        // field pi's own client never sends.
        let req =
            ModelRequest::new("gpt-5.3-codex", vec![Message::user("hi")], 64).with_codex(true);
        let body = build_body(&req);
        assert!(
            body.get("reasoning").is_none(),
            "a Codex request with no reasoning effort requested must omit `reasoning` entirely: {:#?}",
            body.get("reasoning")
        );
        // pi's Codex `buildRequestBody` puts `include: ["reasoning.encrypted_content"]` on the *base*
        // request object unconditionally — present even when `reasoning` itself is omitted.
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));

        // Direct (non-Codex) OpenAI routing of the same model id is unaffected — still gets the
        // explicit "none", and no unconditional `include`.
        let req = ModelRequest::new("gpt-5.3-codex", vec![Message::user("hi")], 64);
        let body = build_body(&req);
        assert_eq!(body["reasoning"]["effort"], "none");
        assert!(body.get("include").is_none());

        // A Codex request that *does* request an effort still gets `reasoning` (and `include`, as before).
        let req = ModelRequest::new("gpt-5.3-codex", vec![Message::user("hi")], 64)
            .with_codex(true)
            .with_reasoning_effort(crate::transport::ReasoningEffort::Low);
        let body = build_body(&req);
        assert_eq!(body["reasoning"]["effort"], "low");
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn codex_routed_requests_send_instructions_field_instead_of_folding_system_into_input() {
        // HIGH pi-parity fix: Codex/ChatGPT's actual backend wants the system prompt in a top-level
        // `instructions` field, with `input` excluding it entirely — unlike every other route, which
        // folds it into `input[0]` (the vanilla native-OpenAI-Responses shape). If the real ChatGPT
        // backend only honors `instructions` for system-level guidance, sending the vanilla shape
        // instead is a functional regression, not cosmetic: every Codex-routed turn's system prompt
        // would silently land in a field/position the backend ignores. Mirrors pi's
        // `openai-codex-responses.ts` `buildRequestBody`, which also always sends
        // `parallel_tool_calls: true` and `text.verbosity` for this backend specifically.
        let req = ModelRequest::new("gpt-5-codex", vec![Message::user("hi")], 64)
            .with_system("be terse")
            .with_codex(true);
        let body = build_body(&req);

        assert_eq!(body["instructions"], "be terse");
        assert_eq!(
            body["input"][0]["role"], "user",
            "a Codex-routed request must not fold the system prompt into input[0]: {:#?}",
            body["input"]
        );
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["text"]["verbosity"], "low");

        // A non-Codex request (the default) is unaffected: system prompt still folds into `input[0]`,
        // and none of Codex's extra fields appear at all.
        let req =
            ModelRequest::new("gpt-5-codex", vec![Message::user("hi")], 64).with_system("be terse");
        let body = build_body(&req);
        assert_eq!(body["input"][0]["content"], "be terse");
        assert!(body.get("instructions").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        assert!(body.get("text").is_none());
    }

    #[test]
    fn codex_routed_requests_default_instructions_when_no_system_prompt_is_set() {
        // pi's own `buildRequestBody` always sends `instructions`, falling back to a default string
        // when no system prompt was configured (`context.systemPrompt || "You are a helpful
        // assistant."`) — a Codex-routed turn is never sent with no `instructions` field at all, unlike
        // every other route, which simply omits the folded-in system message when none is set.
        let req = ModelRequest::new("gpt-5-codex", vec![Message::user("hi")], 64).with_codex(true);
        let body = build_body(&req);
        assert_eq!(body["instructions"], "You are a helpful assistant.");
    }

    #[test]
    fn codex_routed_gpt_5_3_codex_spark_loses_vision_support() {
        // pi-parity (pass 17 follow-up, task 23): `capabilities_for_route` (`models.rs`) is now wired
        // into this dialect's `build_body`. `gpt-5.3-codex-spark`'s Codex catalogue entry
        // (`openai-codex.models.ts`) is genuinely `input: ["text"]` — no vision — even though the
        // identical id natively (`openai.models.ts`) is vision-capable (`capabilities`'s own gpt-5.3
        // -codex-spark branch: `supports_vision: true`). Before this call site threaded `req.is_codex`
        // through, a Codex-routed request for this id got the plain, route-blind `capabilities` lookup's
        // native (vision-capable) answer regardless of route, silently sending a real `input_image` part
        // to a backend whose real catalogue can't accept one.
        let image_message = || Message {
            role: Role::User,
            content: vec![ContentBlock::Image {
                source: ImageSource::base64("image/png", "AAAA"),
            }],
            model_id: None,
            error_message: None,
            aborted: false,
            usage: None,
            stop_reason: None,
        };

        let native_body = build_body(&ModelRequest::new(
            "gpt-5.3-codex-spark",
            vec![image_message()],
            64,
        ));
        assert_eq!(
            native_body["input"][0]["content"][0]["type"], "input_image",
            "native routing keeps this id's real vision support: {:#?}",
            native_body["input"]
        );

        let codex_body = build_body(
            &ModelRequest::new("gpt-5.3-codex-spark", vec![image_message()], 64).with_codex(true),
        );
        assert_eq!(
            codex_body["input"][0]["content"][0],
            json!({ "type": "input_text", "text": USER_IMAGE_PLACEHOLDER }),
            "Codex routing must lose vision support for this id — a real image must degrade to the \
             non-vision placeholder, not ride onto the wire as a real input_image part: {:#?}",
            codex_body["input"]
        );
    }

    #[test]
    fn codex_routed_gpt_5_4_mini_gets_a_lower_effective_output_ceiling_from_its_smaller_context() {
        // pi-parity (pass 17 follow-up, task 23): "gpt-5.4-mini" is 400_000 context natively/Azure
        // (`azure-openai-responses.models.ts` matches native here), but only 272_000 on Codex
        // (`openai-codex.models.ts`) — `capabilities_for_route`'s own doc comment. `max_output_tokens`
        // itself is unchanged across routes for this id; what changes is `clamp_max_tokens_to_context`'s
        // *effective* ceiling, which is derived from `context_window` minus the live prompt size. A
        // 250_000-token prompt (1_000_000 chars, chars/4) leaves native 400_000 - 250_000 - 4_096 =
        // 145_904 tokens of headroom — comfortably above this request's 50_000 max_tokens, so native
        // sends it unclamped — but only 272_000 - 250_000 - 4_096 = 17_904 on Codex, clamping the same
        // request's output down hard. Before this call site threaded `req.is_codex` through to
        // `capabilities_for_route`, a Codex-routed turn this close to its *real*, smaller context window
        // was clamped as if it had native's larger one, risking exactly the 400 this clamp exists to
        // prevent (see `clamp_max_tokens_to_context`'s own doc comment).
        let big_text = "x".repeat(1_000_000);
        let native_req =
            ModelRequest::new("gpt-5.4-mini", vec![Message::user(big_text.as_str())], 50_000);
        assert_eq!(
            build_body(&native_req)["max_output_tokens"], 50_000,
            "native routing's larger real context window leaves enough headroom that this request's \
             own max_tokens ceiling is unaffected"
        );

        let codex_req =
            ModelRequest::new("gpt-5.4-mini", vec![Message::user(big_text.as_str())], 50_000)
                .with_codex(true);
        assert_eq!(
            build_body(&codex_req)["max_output_tokens"], 17_904,
            "Codex routing's genuinely smaller real context window must clamp this request's output \
             down further than native's, not send the same unclamped 50_000 ceiling regardless of route"
        );
    }

    #[test]
    fn azure_routed_gpt_5_4_and_5_5_report_the_larger_real_context_window() {
        // pi-parity (pass 17 follow-up, task 23): "gpt-5.4"/"gpt-5.5" (bare, no mini/nano/pro suffix)
        // are 272_000 context natively and on Codex, but Azure's own catalogue
        // (`azure-openai-responses.models.ts`) genuinely ships a 1.05M context for both —
        // `capabilities_for_route`'s own doc comment. Proven here through `max_output_tokens`'s clamp
        // (the only observable wire effect of `context_window` in this dialect): a prompt sized to
        // exhaust the native/Codex 272_000 window entirely (leaving only the `MIN_CLAMPED_MAX_TOKENS`
        // floor) must still get real, unclamped headroom once Azure's much larger real window is in
        // play, for both ids.
        let big_text = "x".repeat(1_100_000); // 275_000 tokens: over even native/Codex's 272_000.
        for id in ["gpt-5.4", "gpt-5.5"] {
            let native_req = ModelRequest::new(id, vec![Message::user(big_text.as_str())], 50_000);
            assert_eq!(
                build_body(&native_req)["max_output_tokens"], 1_024,
                "{id}: native's 272_000 context is already exhausted by this prompt, so output must \
                 clamp all the way down to the floor"
            );

            let azure_req =
                ModelRequest::new(id, vec![Message::user(big_text.as_str())], 50_000).with_azure(true);
            assert_eq!(
                build_body(&azure_req)["max_output_tokens"], 50_000,
                "{id}: Azure's real 1.05M context leaves this same prompt with plenty of headroom — \
                 must send the request's own unclamped 50_000 ceiling, not the native/Codex-derived \
                 floor"
            );
        }
    }

    #[test]
    fn prompt_cache_key_is_clamped_to_64_chars() {
        let long_key = "k".repeat(200);
        let req =
            ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_cache_key(long_key);
        let body = build_body(&req);
        assert_eq!(body["prompt_cache_key"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn no_cache_suppresses_prompt_cache_key_and_retention() {
        // pi-parity: `ModelRequest::no_cache`'s own doc comment promises to skip
        // `prompt_cache_key`/`prompt_cache_retention` too (equivalently to Anthropic's
        // `cache_control`), matching pi's `cacheRetention === "none"` check — a genuinely one-off
        // request has no follow-up turn to route back to the same cache node.
        let req = ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64)
            .with_cache_key("session-abc")
            .with_cache_long(true)
            .with_no_cache(true);
        let body = build_body(&req);
        assert!(
            body.get("prompt_cache_key").is_none(),
            "got: {:#?}",
            body.get("prompt_cache_key")
        );
        assert!(
            body.get("prompt_cache_retention").is_none(),
            "got: {:#?}",
            body.get("prompt_cache_retention")
        );
    }

    // A recorded text-then-tool-call Responses stream: reasoning summary, then a message, then a
    // function call, terminated by response.completed.
    const FIXTURE: &str = r#"
data: {"type":"response.created","response":{"id":"resp_1"}}

data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}

data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"Let me check."}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Let me check."}]}}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_42","name":"get_weather","arguments":""}}

data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"city\":"}

data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"\"SF\"}"}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_42","name":"get_weather","arguments":"{\"city\":\"SF\"}"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":50,"output_tokens":20,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":8}}}}
"#;

    #[test]
    fn decodes_reasoning_then_tool_call_stream() {
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, FIXTURE).unwrap();

        assert_eq!(events[0], StreamEvent::MessageStart);
        // Reasoning: summary delta, then signature (full item JSON) + stop on `output_item.done`.
        assert_eq!(
            events[1],
            StreamEvent::ThinkingDelta {
                index: 0,
                text: "Let me check.".into()
            }
        );
        assert!(matches!(events[2], StreamEvent::SignatureDelta { .. }));
        if let StreamEvent::SignatureDelta { signature, .. } = &events[2] {
            let parsed: Value = serde_json::from_str(signature).unwrap();
            assert_eq!(parsed["type"], "reasoning");
            assert_eq!(parsed["id"], "rs_1");
        }
        // pi-parity fix: `output_item.done` also resyncs the block's visible thinking text from the
        // item's own authoritative `summary`, same as the text/tool-call resyncs below.
        assert_eq!(
            events[3],
            StreamEvent::ThinkingFinal {
                index: 0,
                text: "Let me check.".into()
            }
        );
        assert_eq!(events[4], StreamEvent::ContentBlockStop { index: 0 });
        // Tool call.
        assert_eq!(
            events[5],
            StreamEvent::ToolUseStart {
                index: 1,
                id: "call_42|fc_1".into(),
                name: "get_weather".into()
            }
        );
        assert_eq!(
            events[6],
            StreamEvent::InputJsonDelta {
                index: 1,
                partial_json: "{\"city\":".into()
            }
        );
        assert_eq!(
            events[7],
            StreamEvent::InputJsonDelta {
                index: 1,
                partial_json: "\"SF\"}".into()
            }
        );
        // `output_item.done` resyncs to the provider's own authoritative arguments before closing —
        // see `message_item_text`'s sibling `Some("function_call")` resync arm.
        assert_eq!(
            events[8],
            StreamEvent::InputJsonFinal {
                index: 1,
                full_json: "{\"city\":\"SF\"}".into()
            }
        );
        assert_eq!(events[9], StreamEvent::ContentBlockStop { index: 1 });
        // Terminal: usage + a ToolUse-upgraded stop reason (status was "completed", but a tool call
        // happened).
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.input_tokens, 50);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.reasoning_tokens, 8);
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse
            })
        );
    }

    #[test]
    fn unrecognized_item_and_event_types_are_dropped_without_breaking_the_rest_of_the_stream() {
        // The Responses API has several built-in server-side tool item types this harness doesn't
        // request today (file/web search, code interpreter, computer use, MCP calls) and has added new
        // top-level event types before; this simulates one arriving mid-stream — it must be dropped,
        // not error out or corrupt decoding of the real content around it.
        const SSE: &str = r#"
data: {"type":"response.created","response":{"id":"resp_1"}}

data: {"type":"response.output_item.added","output_index":0,"item":{"type":"some_future_item_type","id":"x_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"partial"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"some_future_item_type","id":"x_1"}}

data: {"type":"some_future_top_level_event","whatever":"payload"}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":1,"delta":"still works"}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"still works"}]}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            events.iter().any(
                |e| matches!(e, StreamEvent::TextFinal { index: 1, text, .. } if text == "still works")
            ),
            "real content around the unrecognized types must still decode: {events:?}"
        );
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn
            })
        );
    }

    #[test]
    fn separates_cached_from_uncached_input() {
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1000,"output_tokens":20,"input_tokens_details":{"cached_tokens":900}}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.input_tokens, 100); // 1000 - 900 cached
        assert_eq!(usage.cache_read_tokens, 900);
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn
            })
        );
    }

    #[test]
    fn incomplete_status_maps_to_max_tokens() {
        // Usage values mirror pi's own incomplete-path fixture
        // (`openai-responses-terminal-event.test.ts:127-141`'s `createIncompleteEvents`: input_tokens
        // 30, output_tokens 12, cached_tokens 5) so the expected numbers below are directly
        // comparable to pi's `toMatchObject({ input: 25, output: 12, cacheRead: 5, cacheWrite: 0, … })`.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"cut off"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.incomplete","response":{"status":"incomplete","usage":{"input_tokens":30,"output_tokens":12,"input_tokens_details":{"cached_tokens":5}}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::MaxTokens
            })
        );
        // pi-parity strengthening (A-L2): pi's equivalent (`openai-responses-terminal-event
        // .test.ts:206-222`, "finalizes incomplete terminal events as length stops") asserts the full
        // `responseId`/`stopReason`/usage tuple on this exact incomplete path — a truncated turn must
        // still report accurate, billable usage, not just the right stop reason. `responseId` has no
        // equivalent to assert here: this dialect's decoder is deliberately stateless (see the module
        // doc comment — every turn resends full history rather than referencing a server-side
        // `previous_response_id`), so there's no persisted response-id field anywhere in
        // `StreamEvent`/`TokenUsage`, on *any* path, completed or incomplete — N/A by design, not an
        // oversight.
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.input_tokens, 25); // 30 - 5 cached, same "bill the uncached remainder" rule
        // `separates_cached_from_uncached_input` already exercises on the completed path.
        assert_eq!(usage.output_tokens, 12);
        assert_eq!(usage.cache_read_tokens, 5);
        assert_eq!(usage.cache_write_tokens, 0);
    }

    #[test]
    fn response_failed_is_rejected() {
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.failed","response":{"error":{"code":"server_error","message":"boom"}}}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, SSE).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
        // pi-parity strengthening (A-M7): pi's equivalent (`openai-responses-terminal-event
        // .test.ts:224-232`) asserts `.rejects.toThrow("server_error: boom")` — the provider's own
        // error `code` and `message` both showing up in the final error text, not just "some
        // transport error happened." `matches!` alone would still pass if `failure_message` silently
        // dropped the code, dropped the message, or swapped in a generic string — assert on the
        // actual text `failure_message` (this dialect's `code: message` construction) produces.
        let text = err.to_string();
        assert!(
            text.contains("server_error: boom"),
            "error message must carry both the provider's error code and its message text \
             verbatim, got: {text}"
        );
    }

    #[test]
    fn an_embedded_failed_status_on_response_completed_is_rejected_like_response_failed() {
        // pi-parity fix (L2): pi's `mapStopReason` exhaustively guards `response.status` reading
        // `"failed"`/`"cancelled"` even on a `response.completed`/`response.incomplete` event, not
        // only via the dedicated `response.failed` event — our `finalize()` used to only check the
        // event *type*, silently treating this embedded case as a successful (if unrecognized-status)
        // turn instead of a genuine failure.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.completed","response":{"status":"failed","error":{"code":"server_error","message":"boom"}}}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, SSE).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
        let text = err.to_string();
        assert!(text.contains("server_error: boom"), "got: {text}");
    }

    #[test]
    fn an_embedded_cancelled_status_on_response_incomplete_is_rejected() {
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.incomplete","response":{"status":"cancelled"}}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, SSE).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn response_done_is_treated_as_an_equivalent_terminal_event_to_response_completed() {
        // CRITICAL pi-parity fix: Codex/ChatGPT-OAuth-routed requests (`RouteOverride::Prefixed`, see
        // `client.rs`) terminate the stream with a backend-specific `response.done` event instead of
        // `response.completed`/`response.incomplete` — a real backend quirk pi's own `mapCodexEvents`
        // (`openai-codex-responses.ts`) explicitly normalizes into `response.completed` before handing
        // off to its shared processor. Before this fix, `response.done` fell through this decoder's
        // catch-all `_ => {}` arm and did nothing: `saw_terminal` never got set, `finish()` would then
        // hard-error every Codex-routed turn as a stream that "ended before a terminal response event"
        // even though the terminal event had, in fact, already arrived.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"hi"}]}}

data: {"type":"response.done","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            dec.is_terminal(),
            "response.done must set saw_terminal, same as response.completed"
        );
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn
            }),
            "response.done must finalize exactly like response.completed, including the stop reason"
        );
    }

    #[test]
    fn a_non_json_trailer_after_response_completed_is_ignored_not_a_transport_error() {
        // pi-parity fix: `is_terminal()` (used to gate this exact tolerance — see
        // `dialect::push_sse_line`) was only ever overridden on the Anthropic decoder, even though this
        // one already tracks the same `saw_terminal` state — a trailing keepalive/stats line from a
        // gateway/proxy after the real `response.completed` arrived used to hard-error an otherwise-
        // successful turn, unlike the Anthropic path.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}

data: not-json-at-all
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn
            })
        );
    }

    #[test]
    fn a_non_json_data_line_before_response_completed_is_still_a_hard_error() {
        // The flip side of the test above: garbage arriving *before* a terminal response event is a
        // genuine corrupted/tampered stream, not trailing proxy noise — must still fail loudly.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: not-json-at-all
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, SSE).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn truncated_stream_with_no_terminal_event_is_rejected() {
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"partial"}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, SSE).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn interleaved_output_indices_stay_open_concurrently_until_each_closes() {
        // Two items genuinely interleaved (item 1 opens while item 0 is still open) must each produce
        // exactly one ContentBlockStop — item 1's own explicit `output_item.done`, item 0's *implicit*
        // close via `finalize`'s defensive sweep (a malformed/truncated stream must not leave it open
        // forever). Since `Accumulator` now natively tracks as many concurrently-open indices as the
        // wire actually has, index 1 opening does *not* force-close index 0 the way a single-focus
        // design would — never zero closes (a leaked open block) or more than one (a double-close) per
        // item either way.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a"}}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b"}}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b","arguments":"{}"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        let tool_starts = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolUseStart { .. }))
            .count();
        let stops = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockStop { .. }))
            .count();
        assert_eq!(tool_starts, 2);
        assert_eq!(stops, 2);
        // Both items' `ToolUseStart`s arrive back-to-back, live — no buffering delays index 1's own
        // events until index 0 (which never even gets its own explicit close here) does something.
        assert_eq!(
            events[1],
            StreamEvent::ToolUseStart {
                index: 0,
                id: combine_tool_id("call_1", "fc_1"),
                name: "a".into(),
            }
        );
        assert_eq!(
            events[2],
            StreamEvent::ToolUseStart {
                index: 1,
                id: combine_tool_id("call_2", "fc_2"),
                name: "b".into(),
            }
        );
    }

    #[test]
    fn genuinely_interleaved_items_stream_live_not_buffered() {
        // The core of the pi-parity fix this decoder exists to demonstrate: item 1's deltas, which
        // arrive *while item 0 is still open*, are emitted immediately in real arrival order — not
        // buffered and replayed as a single burst once item 0 finally closes. `Accumulator` on the
        // consuming side tracks both indices concurrently and still assembles them in declaration
        // order in the final message, but the *live event stream* itself now shows genuine
        // interleaving, which a client watching for real-time argument-typing progress can observe.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a"}}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b"}}

data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"y\":2}"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a","arguments":"{}"}}

data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"more"}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b","arguments":"{\"y\":2}more"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: combine_tool_id("call_1", "fc_1"),
                    name: "a".into(),
                },
                StreamEvent::ToolUseStart {
                    index: 1,
                    id: combine_tool_id("call_2", "fc_2"),
                    name: "b".into(),
                },
                // Index 1's delta arrives live, right where the wire put it — between index 0's start
                // and its eventual close, not deferred until index 0 finishes.
                StreamEvent::InputJsonDelta {
                    index: 1,
                    partial_json: "{\"y\":2}".into(),
                },
                // Index 0's own `output_item.done` (its arguments never streamed any deltas, so this
                // resync is the *only* source of its final value) closes it — index 1 stays open,
                // untouched.
                StreamEvent::InputJsonFinal {
                    index: 0,
                    full_json: "{}".into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::InputJsonDelta {
                    index: 1,
                    partial_json: "more".into(),
                },
                // Item 1's own `output_item.done` resyncs and closes it — the only close attributable
                // to item 1; index 0's earlier `done` never touched it.
                StreamEvent::InputJsonFinal {
                    index: 1,
                    full_json: "{\"y\":2}more".into(),
                },
                StreamEvent::ContentBlockStop { index: 1 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        );
    }

    /// Reconstructs each tool call's full concatenated JSON-argument buffer from a flat `StreamEvent`
    /// list, mirroring `agent_core::agent::Accumulator::apply` exactly: everything between a
    /// `ToolUseStart` at some `index` and the next `ContentBlockStop` *at that same index* belongs to
    /// that call, `InputJsonDelta` appends, and `InputJsonFinal` *replaces* the buffer outright (the
    /// resync this module's tests exist to prove). Keyed by index — not a single `Option`, since two
    /// calls can now be genuinely open at once — so this test-only helper actually exercises the same
    /// concurrent-tracking behavior the real `Accumulator` does, not the old single-current-block
    /// assumption. Panics on malformed event shapes (an `InputJsonDelta`/`InputJsonFinal` with no open
    /// call at that index, or two `ToolUseStart`s at the same still-open index) — meant to fail loudly
    /// on a genuinely corrupted sequence, not silently produce a wrong map.
    fn reconstruct_tool_args(events: &[StreamEvent]) -> std::collections::HashMap<String, String> {
        let mut out = std::collections::HashMap::new();
        let mut open: std::collections::HashMap<usize, (String, String)> =
            std::collections::HashMap::new();
        for ev in events {
            match ev {
                StreamEvent::ToolUseStart { index, id, .. } => {
                    assert!(
                        !open.contains_key(index),
                        "a ToolUseStart at index {index} while another call at that index is still open"
                    );
                    open.insert(*index, (id.clone(), String::new()));
                }
                StreamEvent::InputJsonDelta {
                    index,
                    partial_json,
                } => {
                    open.get_mut(index)
                        .expect("an InputJsonDelta with no open tool call at this index")
                        .1
                        .push_str(partial_json);
                }
                StreamEvent::InputJsonFinal { index, full_json } => {
                    open.get_mut(index)
                        .expect("an InputJsonFinal with no open tool call at this index")
                        .1
                        .clone_from(full_json);
                }
                StreamEvent::ContentBlockStop { index } => {
                    if let Some((id, args)) = open.remove(index) {
                        out.insert(id, args);
                    }
                }
                _ => {}
            }
        }
        out
    }

    #[test]
    fn preempted_items_arguments_are_never_truncated_or_dropped() {
        // The concrete failure this decoder's per-index tracking exists to prevent: two genuinely
        // interleaved parallel tool calls (item 0 opens, item 1 opens before item 0's own `done`, item
        // 1 closes while item 0 is still open, item 0 keeps streaming more arguments afterward, then
        // item 0 finally closes). A single-focus design would force-close item 0 the moment item 1
        // opened, permanently truncating item 0's arguments at whatever had streamed so far
        // ("{\"a\":" here) — the rest ("1}") would either be silently lost or corrupt item 1's own
        // accumulating buffer. Both calls must end up with their complete, uncorrupted argument
        // strings.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a"}}

data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":"}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b"}}

data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"b\":2"}

data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"}"}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b","arguments":"{\"b\":2}"}}

data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"1}"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a","arguments":"{\"a\":1}"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        let args = reconstruct_tool_args(&events);
        assert_eq!(
            args.get(&combine_tool_id("call_1", "fc_1"))
                .map(String::as_str),
            Some("{\"a\":1}"),
            "item 0's arguments must survive item 1 interleaving whole, not truncated at the \
             preemption point: {args:#?}"
        );
        assert_eq!(
            args.get(&combine_tool_id("call_2", "fc_2"))
                .map(String::as_str),
            Some("{\"b\":2}"),
            "item 1 (buffered, closed before it was ever promoted) must still surface its own \
             complete arguments once promoted, not be dropped: {args:#?}"
        );
        // Exactly two ToolUseStart/ContentBlockStop pairs — no phantom third block from either the
        // promotion or the finalize-time flush double-closing something.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, StreamEvent::ToolUseStart { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, StreamEvent::ContentBlockStop { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn dropped_mid_stream_delta_is_corrected_by_the_final_resync() {
        // The other concrete failure this design fixes: a relay/proxy hiccup silently drops one
        // `function_call_arguments.delta` chunk with no transport-level error at all — nothing else
        // would ever catch this. Before the fix, the decoder never resynced to the provider's own
        // authoritative `function_call_arguments.done`/`output_item.done` value, so the final tool call
        // would silently carry the corrupted (missing-a-chunk) arguments. Simulated here by simply never
        // sending the "missing" delta in the first place — from the decoder's point of view, indis-
        // tinguishable from one that was sent and lost in transit — and asserting the resync event
        // still lands the correct, complete value.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather"}}

data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}

data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"city\":\"SF\"}"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        let args = reconstruct_tool_args(&events);
        assert_eq!(
            args.get(&combine_tool_id("call_1", "fc_1"))
                .map(String::as_str),
            Some("{\"city\":\"SF\"}"),
            "the dedicated function_call_arguments.done resync must recover the complete value even \
             though the streamed deltas alone only ever summed to '{{\"city\":': {args:#?}"
        );
        // Both the dedicated `function_call_arguments.done` and `output_item.done`'s own
        // belt-and-suspenders resync fire here (this test's SSE includes both) — each individually
        // correct and idempotent (same authoritative value both times), so at least one, not
        // necessarily exactly one.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::InputJsonFinal { .. })),
            "expected at least one resync event: {events:#?}"
        );
    }

    #[test]
    fn text_item_resyncs_to_its_authoritative_content_on_output_item_done() {
        // Same resync guarantee as tool-call arguments, for a plain text/refusal item — `output_item
        // .done`'s `item.content` is the ground truth `message_item_text` extracts from.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"Hel"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Hello, world!"}]}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            events.contains(&StreamEvent::TextFinal {
                index: 0,
                text: "Hello, world!".into(),
                id: Some("msg_1".into()),
                phase: None,
            }),
            "expected a TextFinal resync to the item's authoritative content, carrying its real \
             wire id for replay: {events:#?}"
        );
    }

    #[test]
    fn reasoning_item_resyncs_its_visible_thinking_text_on_output_item_done() {
        // LOW-MEDIUM pi-parity fix: unlike the text/tool-call resyncs above, a `reasoning` item's
        // `output_item.done` used to only emit `SignatureDelta` (the raw item JSON, for replay) — never
        // resyncing the *visible* `thinking` text the deltas accumulated. A single dropped/duplicated
        // mid-stream `reasoning_summary_text.delta` chunk (a relay hiccup, no transport error — nothing
        // else would ever catch it) could silently corrupt the persisted/displayed thinking text for
        // the turn. Mirrors pi's `openai-responses-shared.ts` `output_item.done` handler
        // (`summaryText || contentText || slot.block.thinking`). Simulated here by the item's own
        // `summary` disagreeing with what the streamed delta alone would have produced ("Let m" vs the
        // full "Let me check the docs.") — the resync must win.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}

data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"Let m"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Let me check the docs."}]}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            events.contains(&StreamEvent::ThinkingFinal {
                index: 0,
                text: "Let me check the docs.".into(),
            }),
            "expected a ThinkingFinal resync to the item's authoritative summary text, not just the \
             accumulated (truncated) deltas: {events:#?}"
        );
    }

    #[test]
    fn reasoning_item_with_no_summary_or_content_emits_no_thinking_final() {
        // The flip side: when the item carries neither `summary` nor `content` text at all (some
        // models' `output_item.done` for a reasoning item is genuinely empty beyond the signature),
        // there's nothing authoritative to resync against — the accumulated deltas must be left alone,
        // not clobbered with an empty string.
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}

data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"kept as-is"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::ThinkingFinal { .. })),
            "no ThinkingFinal should be emitted when the item has no summary/content text to resync \
             against: {events:#?}"
        );
    }

    #[test]
    fn combine_tool_id_round_trips_through_split() {
        let combined = combine_tool_id("call_1", "fc_1");
        let (call_id, item_id) = split_tool_id(&combined);
        assert_eq!(call_id, "call_1");
        assert_eq!(item_id.as_deref(), Some("fc_1"));
    }

    #[test]
    fn combine_tool_id_round_trips_when_call_id_contains_the_separator() {
        // A `|` (or `\`) inside `call_id` must not corrupt either half on the next `split_tool_id` —
        // a naive join splits on the first `|`, wherever it happens to land.
        let combined = combine_tool_id("call|evil", "fc_1");
        let (call_id, item_id) = split_tool_id(&combined);
        assert_eq!(call_id, "call|evil");
        assert_eq!(item_id.as_deref(), Some("fc_1"));
    }

    #[test]
    fn combine_tool_id_round_trips_when_item_id_contains_the_separator() {
        let combined = combine_tool_id("call_1", "fc|1");
        let (call_id, item_id) = split_tool_id(&combined);
        assert_eq!(call_id, "call_1");
        assert_eq!(item_id.as_deref(), Some("fc|1"));
    }

    #[test]
    fn combine_tool_id_round_trips_when_either_half_contains_a_backslash() {
        let combined = combine_tool_id(r"call\1", r"fc\|2");
        let (call_id, item_id) = split_tool_id(&combined);
        assert_eq!(call_id, r"call\1");
        assert_eq!(item_id.as_deref(), Some(r"fc\|2"));
    }

    #[test]
    fn split_tool_id_treats_a_plain_id_with_no_separator_as_call_id_only() {
        let (call_id, item_id) = split_tool_id("call_1");
        assert_eq!(call_id, "call_1");
        assert_eq!(item_id, None);
    }

    // Regression test mirroring pi's
    // `packages/ai/test/openai-responses-foreign-toolcall-id.test.ts:19-65` — a foreign backend's own
    // combined tool-call id (GitHub Copilot's, in pi's fixture) is far longer than any OpenAI-Responses
    // item id and packs a `/`, `+`, `=` charset from its own opaque encoding. pi's fix hashes the
    // foreign item-id half into a bounded, `fc_`-prefixed, alphanumeric id (`fc_${shortHash(itemId)}`,
    // ≤ 64 chars) rather than replaying it mostly as-is; our `combine_tool_id` does the same at the
    // point it first packs a wire-native id into `ToolUse.id`, using our own digest (not pi's
    // `shortHash` — the two don't need to agree bit-for-bit, only the *shape* of the contract does:
    // bounded, charset-safe, deterministic, one-way).
    const COPILOT_RAW_TOOL_CALL_ID: &str = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==";

    #[test]
    fn combine_tool_id_hashes_a_foreign_oversized_item_id_into_a_bounded_fc_shape() {
        let (call_id, item_id) = COPILOT_RAW_TOOL_CALL_ID.split_once('|').unwrap();
        assert!(
            item_id.len() > MAX_ITEM_ID_LEN,
            "fixture must exercise the oversized path"
        );

        let combined = combine_tool_id(call_id, item_id);
        let (split_call_id, split_item_id) = split_tool_id(&combined);

        // `call_id` is untouched — only `item_id` is foreign/oversized here.
        assert_eq!(split_call_id, call_id);

        let hashed = split_item_id.expect("oversized item id must still produce an item id");
        assert!(
            hashed.len() <= 64,
            "hashed item id must respect OpenAI's own cap: {hashed}"
        );
        assert!(
            hashed.starts_with("fc_"),
            "OpenAI Responses requires the item id to start with \"fc\": {hashed}"
        );
        assert!(
            hashed[3..].chars().all(|c| c.is_ascii_alphanumeric()),
            "hashed item id must be charset-safe: {hashed}"
        );
        // One-way: the digest never contains (and can't be turned back into) the original blob.
        assert_ne!(hashed.as_ref(), item_id);
    }

    #[test]
    fn combine_tool_id_hash_is_deterministic_across_calls() {
        // Not reversible, but stable: the same foreign item id must always collapse to the same
        // digest, or a `ToolResult` elsewhere in the session (which independently references the
        // original combined id) would stop pairing with its `ToolUse` after a second encode pass.
        let (call_id, item_id) = COPILOT_RAW_TOOL_CALL_ID.split_once('|').unwrap();
        let first = combine_tool_id(call_id, item_id);
        let second = combine_tool_id(call_id, item_id);
        assert_eq!(first, second);
    }

    #[test]
    fn combine_tool_id_leaves_an_item_id_at_exactly_the_length_cap_unhashed() {
        let item_id = "x".repeat(MAX_ITEM_ID_LEN);
        let combined = combine_tool_id("call_1", &item_id);
        let (_call_id, split_item_id) = split_tool_id(&combined);
        assert_eq!(split_item_id.as_deref(), Some(item_id.as_str()));
    }

    #[test]
    fn combine_tool_id_hashes_an_item_id_one_over_the_length_cap() {
        let item_id = "x".repeat(MAX_ITEM_ID_LEN + 1);
        let combined = combine_tool_id("call_1", &item_id);
        let (_call_id, split_item_id) = split_tool_id(&combined);
        let hashed = split_item_id.unwrap();
        assert_ne!(hashed.as_ref(), item_id);
        assert!(hashed.starts_with("fc_"));
        assert!(hashed.len() <= MAX_ITEM_ID_LEN);
    }
}
