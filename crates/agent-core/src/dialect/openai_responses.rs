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
//! index-keyed deltas — so this decoder closes the previously-open block on any index switch rather
//! than inferring it from a `finish_reason`. Usage and the terminal stop reason arrive together on
//! `response.completed`/`response.incomplete`, so (unlike the Chat Completions decoder) `MessageStop`
//! is emitted immediately rather than deferred to `finish()`.

use serde_json::{Map, Value, json};

use super::StreamDecoder;
use crate::error::{Error, Result};
use crate::message::{ContentBlock, Role, StopReason, StreamEvent, TokenUsage};
use crate::transport::{ModelRequest, ToolChoice};

fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
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
fn combine_tool_id(call_id: &str, item_id: &str) -> String {
    format!(
        "{}|{}",
        escape_tool_id_part(call_id),
        escape_tool_id_part(item_id)
    )
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
/// to replay.
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
/// it whenever the model supports it), `"system"` otherwise.
fn instruction_role(caps: &crate::models::ModelCaps) -> &'static str {
    if caps.reasoning_effort {
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

fn push_user_content(input: &mut Vec<Value>, blocks: &[ContentBlock], supports_vision: bool) {
    let mut parts: Vec<Value> = Vec::new();
    let mut had_image = false;
    for b in blocks {
        match b {
            ContentBlock::Text { text } if !text.is_empty() => {
                parts.push(json!({ "type": "input_text", "text": text }));
            }
            ContentBlock::Image { source } if supports_vision => {
                parts.push(json!({
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": format!("data:{};base64,{}", source.media_type, source.data),
                }));
            }
            ContentBlock::Image { .. } => had_image = true,
            _ => {}
        }
    }
    if had_image {
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

fn push_assistant_content(input: &mut Vec<Value>, blocks: &[ContentBlock]) {
    for b in blocks {
        match b {
            ContentBlock::Text { text } if !text.is_empty() => {
                input.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text }],
                }));
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
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }],
                    }));
                }
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: args,
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
    let caps = crate::models::capabilities(&req.model);
    let mut input: Vec<Value> = Vec::new();

    if let Some(system) = &req.system {
        input.push(json!({ "role": instruction_role(&caps), "content": system }));
    }
    for m in req.messages.iter() {
        match m.role {
            Role::System => {
                input.push(json!({
                    "role": instruction_role(&caps),
                    "content": text_of(&m.content),
                }));
            }
            Role::User => {
                push_user_content(&mut input, &m.content, caps.supports_vision);
                push_tool_results(&mut input, &m.content, caps.supports_vision);
            }
            Role::Assistant => push_assistant_content(&mut input, &m.content),
        }
    }

    let mut map = Map::new();
    map.insert("model".into(), json!(req.model));
    map.insert("input".into(), Value::Array(input));
    map.insert("stream".into(), json!(true));
    // Stateless harness: every turn resends the full history, so nothing should be retained
    // server-side to reference via `previous_response_id`.
    map.insert("store".into(), json!(false));
    map.insert("max_output_tokens".into(), json!(req.max_tokens));

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
            map.insert(
                "reasoning".into(),
                json!({ "effort": effort.as_str(), "summary": "auto" }),
            );
            map.insert("include".into(), json!(["reasoning.encrypted_content"]));
        } else if caps.reasoning_disableable {
            map.insert("reasoning".into(), json!({ "effort": "none" }));
        }
    }

    // Prompt-cache affinity, same as the Chat Completions dialect: OpenAI caches prefixes
    // automatically, so this is only a routing hint (no `no_cache` gate — there's no cache-write
    // premium to opt out of, unlike Anthropic's `cache_control`).
    if let Some(key) = &req.cache_key {
        map.insert(
            "prompt_cache_key".into(),
            json!(super::openai::clamp_prompt_cache_key(key)),
        );
    }
    if req.cache_long && caps.supports_long_cache {
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

/// Decodes Responses SSE. Items are genuine block boundaries (`output_item.added`/`.done`), so this
/// tracks only which output index is currently open — closing it on any switch — rather than
/// inferring boundaries from deltas the way the Chat Completions decoder does.
pub struct Decoder {
    started: bool,
    open_index: Option<i64>,
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
            open_index: None,
            saw_terminal: false,
            failed: None,
            saw_tool_call: false,
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
        }
    }
}

impl Decoder {
    /// Force-close the previously-open block if this event's index names a different one. The
    /// Responses API always closes an item (`output_item.done`) before opening the next, so this is
    /// defensive — but it's what keeps the decoder correct (rather than corrupting state) if a
    /// provider ever does interleave two items' deltas, since the loop's `Accumulator` can only ever
    /// hold one block open at a time.
    fn close_if_switching(&mut self, out: &mut Vec<StreamEvent>, index: i64) {
        if let Some(open) = self.open_index {
            if open != index {
                out.push(StreamEvent::ContentBlockStop);
                self.open_index = None;
            }
        }
    }

    fn finalize(&mut self, data: &Value, out: &mut Vec<StreamEvent>) {
        self.saw_terminal = true;
        // Defensive: `output_item.done` should always precede the terminal event, but a
        // malformed/truncated stream must not silently drop an unclosed block.
        if self.open_index.is_some() {
            out.push(StreamEvent::ContentBlockStop);
            self.open_index = None;
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

        let status = response
            .and_then(|r| r.get("status"))
            .and_then(Value::as_str);
        let mut stop_reason = match status {
            Some("completed") => StopReason::EndTurn,
            Some("incomplete") => StopReason::MaxTokens,
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
                self.close_if_switching(&mut out, index);
                self.open_index = Some(index);
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
                    out.push(StreamEvent::ToolUseStart { id, name });
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let index = i64_at(data, "output_index");
                self.close_if_switching(&mut out, index);
                self.open_index = Some(index);
                out.push(StreamEvent::ThinkingDelta {
                    text: str_at(Some(data), "delta").to_string(),
                });
            }
            "response.reasoning_summary_part.done" => {
                if self.open_index.is_some() {
                    out.push(StreamEvent::ThinkingDelta {
                        text: "\n\n".to_string(),
                    });
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let index = i64_at(data, "output_index");
                self.close_if_switching(&mut out, index);
                self.open_index = Some(index);
                out.push(StreamEvent::TextDelta {
                    text: str_at(Some(data), "delta").to_string(),
                });
            }
            "response.function_call_arguments.delta" => {
                let index = i64_at(data, "output_index");
                self.close_if_switching(&mut out, index);
                self.open_index = Some(index);
                out.push(StreamEvent::InputJsonDelta {
                    partial_json: str_at(Some(data), "delta").to_string(),
                });
            }
            "response.output_item.done" => {
                let index = i64_at(data, "output_index");
                // Guarded the same way `close_if_switching` is: a `done` for an index that isn't the
                // one currently open (a duplicate, a late/out-of-order event, or one that was already
                // force-closed by an interleaving `close_if_switching` elsewhere) must not close
                // whatever block actually *is* open right now — the `Accumulator` can only ever hold
                // one open, so an unguarded close here would silently truncate it.
                if self.open_index == Some(index) {
                    let item = data.get("item");
                    if item.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("reasoning")
                    {
                        // The whole item, JSON-stringified, is the block's replayable signature — see
                        // the module doc comment.
                        let sig = item.map(ToString::to_string).unwrap_or_default();
                        out.push(StreamEvent::SignatureDelta { signature: sig });
                    }
                    out.push(StreamEvent::ContentBlockStop);
                    self.open_index = None;
                }
            }
            "response.completed" | "response.incomplete" => self.finalize(data, &mut out),
            "response.failed" => {
                self.saw_terminal = true;
                self.failed = Some(failure_message(data));
            }
            _ => {}
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::decode_sse;
    use crate::message::{ImageSource, Message, ToolDef};

    #[test]
    fn build_body_maps_system_tool_calls_and_results() {
        let req = ModelRequest::new(
            "gpt-4o",
            vec![
                Message::user("weather?"),
                Message::assistant(vec![
                    ContentBlock::Text {
                        text: "checking".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "call_1|fc_1".into(),
                        name: "get_weather".into(),
                        input: json!({ "city": "SF" }),
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
                    ContentBlock::Text { text: "42".into() },
                ]),
            ],
            64,
        );
        let body = build_body(&req);
        assert_eq!(body["input"][1], item);
        assert_eq!(body["input"][2]["content"][0]["text"], "42");
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
                    ContentBlock::Text { text: "42".into() },
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
    fn prompt_cache_key_is_clamped_to_64_chars() {
        let long_key = "k".repeat(200);
        let req =
            ModelRequest::new("gpt-4o", vec![Message::user("hi")], 64).with_cache_key(long_key);
        let body = build_body(&req);
        assert_eq!(body["prompt_cache_key"].as_str().unwrap().len(), 64);
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
                text: "Let me check.".into()
            }
        );
        assert!(matches!(events[2], StreamEvent::SignatureDelta { .. }));
        if let StreamEvent::SignatureDelta { signature } = &events[2] {
            let parsed: Value = serde_json::from_str(signature).unwrap();
            assert_eq!(parsed["type"], "reasoning");
            assert_eq!(parsed["id"], "rs_1");
        }
        assert_eq!(events[3], StreamEvent::ContentBlockStop);
        // Tool call.
        assert_eq!(
            events[4],
            StreamEvent::ToolUseStart {
                id: "call_42|fc_1".into(),
                name: "get_weather".into()
            }
        );
        assert_eq!(
            events[5],
            StreamEvent::InputJsonDelta {
                partial_json: "{\"city\":".into()
            }
        );
        assert_eq!(
            events[6],
            StreamEvent::InputJsonDelta {
                partial_json: "\"SF\"}".into()
            }
        );
        assert_eq!(events[7], StreamEvent::ContentBlockStop);
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
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.output_text.delta","output_index":0,"delta":"cut off"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1"}}

data: {"type":"response.incomplete","response":{"status":"incomplete","usage":{"input_tokens":5,"output_tokens":5}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        assert_eq!(
            events.last(),
            Some(&StreamEvent::MessageStop {
                stop_reason: StopReason::MaxTokens
            })
        );
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
    fn interleaved_output_indices_force_close_the_previous_block() {
        // Defensive case: if a provider ever interleaves two items' deltas (rather than the API's
        // documented added→...→done sequencing), the decoder must still produce exactly one
        // ContentBlockStop per item rather than corrupting the Accumulator's single-open-block state
        // (which can only ever hold one block open — see `close_if_switching`'s doc comment).
        const SSE: &str = r#"
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a"}}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b"}}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"b","arguments":"{}"}}

data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, SSE).unwrap();
        // Index 1 opening while index 0 was still open force-closes index 0; index 1's own
        // `output_item.done` closes it explicitly — two ToolUseStart, two ContentBlockStop, no more.
        let tool_starts = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolUseStart { .. }))
            .count();
        let stops = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ContentBlockStop))
            .count();
        assert_eq!(tool_starts, 2);
        assert_eq!(stops, 2);
    }

    #[test]
    fn stale_output_item_done_does_not_close_a_different_open_block() {
        // Item 0 opens, item 1 opens (force-closing item 0 via `close_if_switching`), then item 0's
        // own `output_item.done` arrives *late* — after item 1 is already the one open. Before the
        // fix, `output_item.done`'s `ContentBlockStop` fired unconditionally, so this stale event
        // would prematurely close item 1's still-accumulating block even though its index (0) doesn't
        // match what's actually open (1).
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
                    id: combine_tool_id("call_1", "fc_1"),
                    name: "a".into(),
                },
                // Item 1 opening force-closes item 0 — the only close attributable to item 0.
                StreamEvent::ContentBlockStop,
                StreamEvent::ToolUseStart {
                    id: combine_tool_id("call_2", "fc_2"),
                    name: "b".into(),
                },
                StreamEvent::InputJsonDelta {
                    partial_json: "{\"y\":2}".into(),
                },
                // Item 0's stale `output_item.done` (index 0) is dropped here — item 1 (index 1) is
                // what's actually open, so no ContentBlockStop/SignatureDelta fires for it.
                StreamEvent::InputJsonDelta {
                    partial_json: "more".into(),
                },
                // Item 1's own `output_item.done` closes it — the only close attributable to item 1.
                StreamEvent::ContentBlockStop,
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
}
