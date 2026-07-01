//! OpenAI Chat Completions wire (`/v1/chat/completions`).
//!
//! OpenAI's shape is flatter than the internal model, so both directions need real translation:
//! - **Request:** the system prompt becomes a `system` message; an assistant turn's `ToolUse` blocks
//!   become `tool_calls` (arguments as a JSON *string*); a user turn's `ToolResult` blocks fan out
//!   into separate `role:"tool"` messages.
//! - **Stream:** `tool_calls` arrive as index-keyed deltas (id+name once, then `arguments` fragments)
//!   with no explicit block-stop events, so the decoder synthesizes `ContentBlockStop` and defers
//!   `MessageStop` to end-of-stream (after the trailing `usage` chunk).

use serde_json::{Map, Value, json};

use super::StreamDecoder;
use crate::error::{Error, Result};
use crate::message::{ContentBlock, Role, StopReason, StreamEvent, TokenUsage};
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

fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Build an OpenAI user-message `content` from a turn's text and image blocks (tool results are
/// emitted separately). Returns a plain string when there are no images — OpenAI's common case — and
/// a multimodal parts array (`[{type:"text"}, {type:"image_url"}…]`) when images are present. `None`
/// if the turn carries neither text nor image (e.g. a tool-result-only turn). Without this, image
/// blocks were dropped on the floor: `text_of` keeps only text, so vision input silently vanished.
fn user_content(blocks: &[ContentBlock]) -> Option<Value> {
    let has_image = blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    if !has_image {
        let text = text_of(blocks);
        return (!text.is_empty()).then_some(Value::String(text));
    }
    let mut parts: Vec<Value> = Vec::new();
    for b in blocks {
        match b {
            ContentBlock::Text { text } if !text.is_empty() => {
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

/// Build the streaming request body, translating the internal messages into OpenAI's flat shape.
pub fn build_body(req: &ModelRequest) -> Value {
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
                if let Some(content) = user_content(&m.content) {
                    messages.push(json!({ "role": "user", "content": content }));
                }
                for b in &m.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        images,
                        ..
                    } = b
                    {
                        messages.push(json!({ "role": "tool", "tool_call_id": tool_use_id, "content": content }));
                        // OpenAI's `tool` role can't carry images, so fan any visual output out to a
                        // following `user` message that references the originating call.
                        if !images.is_empty() {
                            let mut parts: Vec<Value> = vec![json!({
                                "type": "text",
                                "text": format!("Image output from tool call {tool_use_id}:"),
                            })];
                            for source in images {
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
            }
            Role::Assistant => {
                let text = text_of(&m.content);
                let tool_calls: Vec<Value> = m
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, name, input } => Some(json!({
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
                messages.push(Value::Object(msg));
            }
        }
    }

    let caps = crate::models::capabilities(&req.model);
    let mut map = Map::new();
    map.insert("model".into(), json!(req.model));
    // OpenAI reasoning models (o-series, gpt-5) reject `max_tokens` and require
    // `max_completion_tokens`; non-reasoning chat models take `max_tokens`. The gateway forwards the
    // body verbatim, so the agent must pick the right field per model.
    let max_tokens_field = match caps.max_tokens_field {
        crate::models::MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
        crate::models::MaxTokensField::MaxTokens => "max_tokens",
    };
    map.insert(max_tokens_field.into(), json!(req.max_tokens));
    // Reasoning models are driven by `reasoning_effort` (minimal/low/medium/high/xhigh) rather than a
    // thinking-token budget; emit it when the model takes one and the caller set a level.
    if caps.reasoning_effort {
        if let Some(effort) = req.reasoning_effort {
            map.insert("reasoning_effort".into(), json!(effort.as_str()));
        }
    }
    map.insert("stream".into(), json!(true));
    // Ask for a trailing usage chunk so token accounting works on the streaming path.
    map.insert("stream_options".into(), json!({ "include_usage": true }));
    // Prompt-cache affinity: OpenAI routes automatic prefix-cache hits by `prompt_cache_key`, so a
    // stable per-conversation key keeps a session pinned to a warm cache node. (OpenAI caches prefixes
    // automatically — there are no explicit breakpoints to set, only this routing hint.) The key has a
    // hard length cap; a caller-supplied key over it would otherwise 400 the request.
    if let Some(key) = &req.cache_key {
        map.insert(
            "prompt_cache_key".into(),
            json!(clamp_prompt_cache_key(key)),
        );
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
    Tool,
}

/// The delta field names different OpenAI-compatible endpoints use for reasoning content:
/// `reasoning_content` (llama.cpp and most providers), `reasoning` (a few others), `reasoning_text`
/// (a third convention). Checked in this order; first-seen field for a stream wins even if a provider
/// later echoes the same text on a second field (observed on some gateways) — using both would double
/// the visible reasoning.
const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

/// Decodes OpenAI SSE. Synthesizes the block-stop boundaries OpenAI omits and holds `MessageStop`
/// until `finish()` so it lands after the trailing usage chunk.
pub struct Decoder {
    started: bool,
    saw_finish: bool,
    open: Open,
    stop_reason: StopReason,
    usage: TokenUsage,
    /// Which [`REASONING_FIELDS`] entry this stream's reasoning arrives on, once known — remembered
    /// as the `Thinking` block's `signature` so a later replay (`build_body`) sends it back under the
    /// exact field name this provider expects (some accept only one of the three).
    reasoning_field: Option<&'static str>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            started: false,
            saw_finish: false,
            open: Open::None,
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            reasoning_field: None,
        }
    }
}

impl Decoder {
    fn close_open(&mut self, out: &mut Vec<StreamEvent>) {
        if self.open != Open::None {
            out.push(StreamEvent::ContentBlockStop);
            self.open = Open::None;
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

        // Trailing usage-only chunk (choices empty).
        if let Some(usage) = data.get("usage").filter(|u| !u.is_null()) {
            let cached = usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let prompt = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            // OpenAI's `prompt_tokens` is the *whole* prompt including cached tokens; bill the
            // uncached remainder as `input_tokens` and report the cache hit separately so accounting
            // doesn't double-count.
            self.usage.cache_read_tokens = cached;
            self.usage.input_tokens = prompt.saturating_sub(cached);
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

        let Some(choice) = data.get("choices").and_then(|c| c.get(0)) else {
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
                    signature: self.reasoning_field.unwrap_or_default().to_string(),
                });
            }
            out.push(StreamEvent::ThinkingDelta {
                text: text.to_string(),
            });
        }

        // Plain text content.
        if let Some(text) = delta.and_then(|d| d.get("content")).and_then(Value::as_str) {
            if !text.is_empty() {
                if self.open == Open::Thinking {
                    self.close_open(&mut out);
                }
                if self.open == Open::None {
                    self.open = Open::Text;
                }
                out.push(StreamEvent::TextDelta {
                    text: text.to_string(),
                });
            }
        }

        // Tool-call deltas: id+name on first sight of an index, then `arguments` fragments.
        if let Some(calls) = delta
            .and_then(|d| d.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for tc in calls {
                let func = tc.get("function");
                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                    self.close_open(&mut out);
                    self.open = Open::Tool;
                    let name = func
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    out.push(StreamEvent::ToolUseStart {
                        id: id.to_string(),
                        name,
                    });
                }
                if let Some(args) = func
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                {
                    if !args.is_empty() {
                        out.push(StreamEvent::InputJsonDelta {
                            partial_json: args.to_string(),
                        });
                    }
                }
            }
        }

        // Finish reason closes the open block and records why we stopped (MessageStop is held).
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.saw_finish = true;
            self.stop_reason = map_finish_reason(Some(reason));
            self.close_open(&mut out);
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
        Ok(vec![StreamEvent::MessageStop {
            stop_reason: self.stop_reason,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::decode_sse;
    use crate::message::{Message, ToolDef};

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
                        id: "call_1".into(),
                        name: "get_weather".into(),
                        input: json!({ "city": "SF" }),
                    },
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
                    text: "Let me check.".into()
                },
                // Text block closes when the tool call begins — same shape the Anthropic decoder
                // produces, so the loop assembles both dialects identically.
                StreamEvent::ContentBlockStop,
                StreamEvent::ToolUseStart {
                    id: "call_42".into(),
                    name: "get_weather".into()
                },
                StreamEvent::InputJsonDelta {
                    partial_json: "{\"city\":".into()
                },
                StreamEvent::InputJsonDelta {
                    partial_json: "\"SF\"}".into()
                },
                StreamEvent::ContentBlockStop,
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
                    signature: "reasoning_content".into()
                },
                StreamEvent::ThinkingDelta {
                    text: "Let me ".into()
                },
                StreamEvent::ThinkingDelta {
                    text: "think.".into()
                },
                StreamEvent::ContentBlockStop, // thinking closes when text starts
                StreamEvent::TextDelta {
                    text: "Answer.".into()
                },
                StreamEvent::ContentBlockStop,
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
                StreamEvent::ThinkingDelta { text } => Some(text.as_str()),
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
                    ContentBlock::Text { text: "42".into() },
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
