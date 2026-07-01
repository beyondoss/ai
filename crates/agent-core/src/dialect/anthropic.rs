//! Anthropic Messages wire (`/v1/messages`).
//!
//! The harness's internal model was chosen to be Anthropic-shaped (content blocks,
//! `tool_use`/`tool_result`), so the request mapping is nearly an identity and the SSE decoder is a
//! direct translation of Anthropic's `content_block_*` / `message_*` events.

use serde_json::{Map, Value, json};

use super::StreamDecoder;
use crate::error::{Error, Result};
use crate::message::{StopReason, StreamEvent, TokenUsage};
use crate::transport::{ModelRequest, ToolChoice};

/// Build the streaming request body. `system` is hoisted to a top-level field (Anthropic keeps it
/// out of `messages`); `messages` and `tools` serialize straight from the internal model.
///
/// Three prompt-cache breakpoints are stamped in. An agent loop re-sends an ever-growing prefix —
/// tools, then system, then the whole prior conversation — on every turn; without caching each turn
/// re-bills that entire prefix at full input price, an O(n²) token cost over a `max_steps`-deep run.
/// Anthropic caches the request prefix up to each `cache_control` mark (reads cost ~10% of input
/// tokens), so we anchor one breakpoint on the fixed tool block, one on the system prompt (a stable
/// anchor that survives Anthropic's ~20-block breakpoint lookback on tool-heavy turns), and roll a
/// third onto the last message to capture the conversation so far. The TTL is 5 min, or 1 hour when
/// `cache_long` is set (see [`cache_control`]).
pub fn build_body(req: &ModelRequest) -> Value {
    let mut map = Map::new();
    map.insert("model".into(), Value::String(req.model.clone()));
    map.insert("max_tokens".into(), Value::from(req.max_tokens));
    map.insert("stream".into(), Value::Bool(true));

    // The 1-hour TTL is only valid on models that support long cache retention; Anthropic 400s
    // otherwise. Gate the request's `cache_long` opt-in on the model's capability so an unsupported
    // model silently falls back to the standard 5-minute TTL instead of erroring the turn.
    let long = req.cache_long && crate::models::capabilities(&req.model).supports_long_cache;
    // `no_cache` skips every breakpoint below: a genuinely one-off request (no follow-up turn to read
    // the cache back) would otherwise eat the ~1.25x cache-write premium for an entry nothing reads.
    let cc = (!req.no_cache).then(|| cache_control(long));

    // Rolling breakpoint: cache the conversation prefix (tools + system + every prior message) up to
    // the final block, so next turn the whole accumulated transcript is a cache read, not a re-bill.
    let mut messages = serde_json::to_value(req.messages.as_ref()).unwrap_or(Value::Null);
    encode_tool_result_images(&mut messages);
    if let Some(cc) = &cc {
        mark_last_block(&mut messages, cc);
    }
    map.insert("messages".into(), messages);

    if let Some(system) = &req.system {
        // System as a single cached text block — a *dedicated* third breakpoint. Anthropic's
        // breakpoint lookback only walks back ~20 content blocks; on a tool-heavy turn (N tool_use +
        // N tool_result blocks) the rolling message breakpoint can fall outside that window, so this
        // stable anchor keeps the (large, fixed) system prompt a cache read. `no_cache` drops the
        // breakpoint but keeps the system block itself (still needed on the wire either way).
        map.insert(
            "system".into(),
            match &cc {
                Some(cc) => json!([{ "type": "text", "text": system, "cache_control": cc }]),
                None => json!([{ "type": "text", "text": system }]),
            },
        );
    }
    if let Some(thinking) = &req.thinking {
        // Extended thinking. Anthropic requires `max_tokens > budget_tokens` and forbids `temperature`
        // alongside it (we never set temperature). Newer models (the capability table's `Adaptive`
        // shape) take an effort-based shape instead of an explicit budget, with `output_config.effort`
        // as a *sibling top-level request field*, not nested under `thinking` — a request-shape detail
        // easy to get wrong. Both shapes explicitly set `display: "summarized"`: Anthropic's own API
        // default for `adaptive` is "omitted" (no visible reasoning text at all), so leaving it unset
        // on an adaptive model silently produces empty thinking output.
        match crate::models::capabilities(&req.model).thinking {
            crate::models::ThinkingShape::Adaptive => {
                map.insert(
                    "thinking".into(),
                    json!({ "type": "adaptive", "display": "summarized" }),
                );
                if let Some(effort) = req.reasoning_effort {
                    map.insert("output_config".into(), json!({ "effort": effort.as_str() }));
                }
            }
            _ => {
                map.insert(
                    "thinking".into(),
                    json!({
                        "type": "enabled",
                        "budget_tokens": thinking.budget_tokens,
                        "display": "summarized",
                    }),
                );
            }
        }
    }
    if !req.tools.is_empty() {
        // Anchor breakpoint: the tool definitions (ten JSON schemas) are identical every turn and sit
        // at the front of the cache order, so this entry stays warm even when the rolling message
        // breakpoint is rewritten each turn. Requires stable tool ordering — see `definitions()`.
        let mut tools = serde_json::to_value(req.tools.as_ref()).unwrap_or(Value::Null);
        if let Some(cc) = &cc {
            mark_last_tool(&mut tools, cc);
        }
        map.insert("tools".into(), tools);
    }
    // Constrain tool use only when the caller asked: an unset `tool_choice` emits nothing, leaving
    // Anthropic's default (auto when tools are present), so the common request shape is untouched.
    if let Some(choice) = &req.tool_choice {
        map.insert("tool_choice".into(), tool_choice(choice));
    }
    Value::Object(map)
}

/// Map a [`ToolChoice`] to Anthropic's `tool_choice` object. Anthropic spells "must call some tool"
/// as `any` and pins a specific tool with `{type:"tool", name}`.
fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({ "type": "auto" }),
        ToolChoice::None => json!({ "type": "none" }),
        ToolChoice::Required => json!({ "type": "any" }),
        ToolChoice::Tool(name) => json!({ "type": "tool", "name": name }),
    }
}

/// The `cache_control` object to stamp on a breakpoint: ephemeral, with the 1-hour TTL when `long`.
fn cache_control(long: bool) -> Value {
    if long {
        json!({ "type": "ephemeral", "ttl": "1h" })
    } else {
        json!({ "type": "ephemeral" })
    }
}

/// Stamp a cache breakpoint onto the last content block of the last message. No-op if the history is
/// empty or the final message carries no content blocks.
fn mark_last_block(messages: &mut Value, cc: &Value) {
    if let Some(block) = messages
        .as_array_mut()
        .and_then(|msgs| msgs.last_mut())
        .and_then(|m| m.get_mut("content"))
        .and_then(Value::as_array_mut)
        .and_then(|content| content.last_mut())
        .and_then(Value::as_object_mut)
    {
        block.insert("cache_control".into(), cc.clone());
    }
}

/// Rewrite `tool_result` blocks carrying images into Anthropic's content-array shape. The derived
/// JSON is `{type:"tool_result", content:"text", images:[…]}`, but Anthropic wants the images *inside*
/// `content`: `{content:[{type:"text",text},{type:"image",source}…]}`. A no-op for the common
/// text-only result (no `images` key), so the existing wire is untouched.
fn encode_tool_result_images(messages: &mut Value) {
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };
    for m in msgs {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content {
            let Some(obj) = block.as_object_mut() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            let images = match obj.remove("images") {
                Some(Value::Array(imgs)) if !imgs.is_empty() => imgs,
                // No images (or the key was already absent): leave the string `content` as-is.
                _ => continue,
            };
            let text = obj
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let mut parts: Vec<Value> = Vec::new();
            if !text.is_empty() {
                parts.push(json!({ "type": "text", "text": text }));
            }
            // A serialized `ImageSource` is exactly Anthropic's `source` object (`type:"base64"`, …).
            for source in images {
                parts.push(json!({ "type": "image", "source": source }));
            }
            obj.insert("content".into(), Value::Array(parts));
        }
    }
}

/// Stamp a cache breakpoint onto the last tool definition.
fn mark_last_tool(tools: &mut Value, cc: &Value) {
    if let Some(tool) = tools
        .as_array_mut()
        .and_then(|t| t.last_mut())
        .and_then(Value::as_object_mut)
    {
        tool.insert("cache_control".into(), cc.clone());
    }
}

fn map_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("refusal") => StopReason::Refusal,
        // `pause_turn` is Anthropic pausing a long-running turn it expects the client to *resubmit* to
        // continue — not a natural end. We have no resubmit step in the loop, so map it to `Other`
        // rather than `EndTurn`: reading it as a clean end-of-turn would silently truncate a turn the
        // model meant to keep going. (A fully distinct `PauseTurn` variant that drives a resubmit would
        // need a `message.rs` enum change plus agent-loop handling — out of scope for this fix.)
        Some("pause_turn") => StopReason::Other,
        // Content flagged by safety filters mid-generation — not yet a named variant in Anthropic's own
        // SDK types, but a real terminal state the reference agent treats as an error, not a clean end.
        // We don't have a distinct explanation to surface for it (unlike `refusal`, which carries one in
        // `stop_details.explanation`), so it shares `Refusal`'s variant rather than earning a new one —
        // both mean "the model was blocked from completing," and the loop already lets a caller tell
        // either apart from a normal end-of-turn instead of reading it as success.
        Some("sensitive") => StopReason::Refusal,
        Some(other) => {
            // A genuinely unrecognized value (Anthropic added a new terminal state we don't know about
            // yet) silently collapsing into `Other` — which the loop treats identically to a normal
            // `EndTurn` — would hide a real change in provider behavior. `warn!` so it's at least
            // visible, without hard-failing the turn (the reference agent throws here; we're more
            // conservative since a false-positive on this match would abort an otherwise-fine turn).
            tracing::warn!(
                stop_reason = other,
                "unrecognized Anthropic stop_reason; treating as Other"
            );
            StopReason::Other
        }
        None => StopReason::Other,
    }
}

/// Decodes Anthropic SSE. Tracks token usage (input + cache reads/writes from `message_start`,
/// output from `message_delta`) and the stop reason, emitting a single `Usage` + `MessageStop` at
/// `message_stop`. `saw_start`/`saw_stop` let `finish` reject a stream truncated before its terminal
/// `message_stop`.
#[derive(Default)]
pub struct Decoder {
    usage: TokenUsage,
    stop_reason: StopReason,
    saw_start: bool,
    saw_stop: bool,
}

impl StreamDecoder for Decoder {
    fn push(&mut self, data: &Value) -> Vec<StreamEvent> {
        let kind = data.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "message_start" => {
                self.saw_start = true;
                let usage = data.get("message").and_then(|m| m.get("usage"));
                self.usage.input_tokens = u32_at(usage, "input_tokens");
                self.usage.output_tokens = u32_at(usage, "output_tokens");
                // Cache accounting is reported only on `message_start` in the real API; capturing it
                // is what makes the prompt-cache breakpoints we stamp in `build_body` observable.
                self.usage.cache_read_tokens = u32_at(usage, "cache_read_input_tokens");
                self.usage.cache_write_tokens = u32_at(usage, "cache_creation_input_tokens");
                // The 1h/5m TTL split lives one level deeper, only when the provider breaks it out.
                self.usage.cache_write_1h_tokens = usage
                    .and_then(|u| u.get("cache_creation"))
                    .map(|cc| u32_at(Some(cc), "ephemeral_1h_input_tokens"))
                    .unwrap_or(0);
                vec![StreamEvent::MessageStart]
            }
            "content_block_start" => {
                let block = data.get("content_block");
                match block.and_then(|b| b.get("type")).and_then(Value::as_str) {
                    Some("tool_use") => {
                        let id = str_at(block, "id").to_string();
                        let name = str_at(block, "name").to_string();
                        vec![StreamEvent::ToolUseStart { id, name }]
                    }
                    // A redacted-thinking block is fully delivered here (no deltas follow): its opaque
                    // `data` must be replayed verbatim so the model keeps reasoning continuity.
                    Some("redacted_thinking") => vec![StreamEvent::RedactedThinking {
                        data: str_at(block, "data").to_string(),
                    }],
                    // Text and (clear) thinking blocks open empty and accrue via deltas — no event.
                    _ => Vec::new(),
                }
            }
            "content_block_delta" => {
                let delta = data.get("delta");
                match delta.and_then(|d| d.get("type")).and_then(Value::as_str) {
                    Some("text_delta") => {
                        vec![StreamEvent::TextDelta {
                            text: str_at(delta, "text").to_string(),
                        }]
                    }
                    Some("thinking_delta") => {
                        vec![StreamEvent::ThinkingDelta {
                            text: str_at(delta, "thinking").to_string(),
                        }]
                    }
                    Some("signature_delta") => {
                        vec![StreamEvent::SignatureDelta {
                            signature: str_at(delta, "signature").to_string(),
                        }]
                    }
                    Some("input_json_delta") => {
                        vec![StreamEvent::InputJsonDelta {
                            partial_json: str_at(delta, "partial_json").to_string(),
                        }]
                    }
                    _ => Vec::new(),
                }
            }
            "content_block_stop" => vec![StreamEvent::ContentBlockStop],
            "message_delta" => {
                let delta = data.get("delta");
                self.stop_reason = map_stop_reason(
                    delta
                        .and_then(|d| d.get("stop_reason"))
                        .and_then(Value::as_str),
                );
                let usage = data.get("usage");
                let out = u32_at(usage, "output_tokens");
                if out > 0 {
                    self.usage.output_tokens = out;
                }
                // Real Anthropic only reports cache fields on `message_start`, never here — but a
                // proxy sitting in front of it could, and a stale `message_start`-only snapshot would
                // then silently under/over-report the rest of the turn. Refresh only when present, so
                // this is a no-op against the real API's actual behavior.
                if let Some(read) = usage.and_then(|u| u.get("cache_read_input_tokens")) {
                    if let Some(read) = read.as_u64() {
                        self.usage.cache_read_tokens = read as u32;
                    }
                }
                if let Some(write) = usage.and_then(|u| u.get("cache_creation_input_tokens")) {
                    if let Some(write) = write.as_u64() {
                        self.usage.cache_write_tokens = write as u32;
                    }
                }
                if let Some(cc) = usage.and_then(|u| u.get("cache_creation")) {
                    self.usage.cache_write_1h_tokens =
                        u32_at(Some(cc), "ephemeral_1h_input_tokens");
                }
                // Reasoning tokens, when broken out separately, are still *included* in
                // `output_tokens`; capture them so a caller can see the thinking share of the spend.
                let thinking = usage
                    .and_then(|u| u.get("output_tokens_details"))
                    .map(|d| u32_at(Some(d), "thinking_tokens"))
                    .unwrap_or(0);
                if thinking > 0 {
                    self.usage.reasoning_tokens = thinking;
                }
                // On a refusal, Anthropic carries a human-readable reason in
                // `delta.stop_details.explanation`. Surface it as a text delta so it lands in the
                // assembled assistant message instead of being dropped — otherwise a refusal arrives as
                // an empty turn with only a `Refusal` stop reason, and the caller can't tell the user
                // *why*. (The block has already closed by `message_delta`, so this trailing text is
                // flushed as its own block; see the loop's `Accumulator`.)
                if self.stop_reason == StopReason::Refusal {
                    let explanation = delta
                        .and_then(|d| d.get("stop_details"))
                        .and_then(|sd| sd.get("explanation"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !explanation.is_empty() {
                        return vec![StreamEvent::TextDelta {
                            text: explanation.to_string(),
                        }];
                    }
                }
                Vec::new()
            }
            "message_stop" => {
                self.saw_stop = true;
                vec![
                    StreamEvent::Usage(self.usage),
                    StreamEvent::MessageStop {
                        stop_reason: self.stop_reason,
                    },
                ]
            }
            _ => Vec::new(),
        }
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        // A stream that opened (`message_start`) but never delivered `message_stop` was truncated
        // mid-flight — a dropped connection or a gateway cut. Reject it rather than let the partial
        // turn pass as a clean completion.
        if self.saw_start && !self.saw_stop {
            return Err(Error::Transport(
                "Anthropic stream ended before message_stop".into(),
            ));
        }
        Ok(Vec::new())
    }
}

fn str_at<'a>(v: Option<&'a Value>, key: &str) -> &'a str {
    v.and_then(|v| v.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn u32_at(v: Option<&Value>, key: &str) -> u32 {
    v.and_then(|v| v.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::decode_sse;
    use crate::message::{ContentBlock, Message, ToolDef};
    use serde_json::json;

    #[test]
    fn build_body_hoists_system_and_keeps_blocks() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256)
            .with_system("be brief")
            .with_tools(vec![ToolDef {
                name: "read".into(),
                description: "read a file".into(),
                input_schema: json!({ "type": "object" }),
            }]);
        let body = build_body(&req);
        assert_eq!(body["model"], "claude-opus-4-8");
        assert_eq!(body["stream"], true);
        // System is a cached text-block array (a dedicated breakpoint), not a bare string.
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "be brief");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["tools"][0]["name"], "read");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn build_body_stamps_cache_breakpoints() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::tool_result("tu_1", "out", false),
            ],
            256,
        )
        .with_tools(vec![
            ToolDef {
                name: "read".into(),
                description: "read a file".into(),
                input_schema: json!({ "type": "object" }),
            },
            ToolDef {
                name: "write".into(),
                description: "write a file".into(),
                input_schema: json!({ "type": "object" }),
            },
        ]);
        let body = build_body(&req);
        // Anchor breakpoint on the last (only the last) tool definition.
        assert!(body["tools"][0].get("cache_control").is_none());
        assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
        // Rolling breakpoint on the last block of the last message, and nowhere earlier.
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn no_cache_skips_every_breakpoint() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("hi"),
                Message::tool_result("tu_1", "out", false),
            ],
            256,
        )
        .with_system("be brief")
        .with_tools(vec![ToolDef {
            name: "read".into(),
            description: "read a file".into(),
            input_schema: json!({ "type": "object" }),
        }])
        .with_no_cache(true);
        let body = build_body(&req);
        assert!(body["tools"][0].get("cache_control").is_none());
        assert!(
            body["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(body["system"][0].get("cache_control").is_none());
        // The system block itself is still present, just uncached.
        assert_eq!(body["system"][0]["text"], "be brief");
    }

    #[test]
    fn image_block_serializes_to_anthropic_source_shape() {
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::user_with_images(
                "what is this?",
                vec![ImageSource::base64("image/png", "AAAA")],
            )],
            256,
        );
        let body = build_body(&req);
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "AAAA");
    }

    #[test]
    fn assistant_tool_use_round_trips_into_body() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("weather?"),
                Message::assistant(vec![ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "get_weather".into(),
                    input: json!({ "city": "SF" }),
                }]),
                Message::tool_result("toolu_1", "72F", false),
            ],
            256,
        );
        let body = build_body(&req);
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(body["messages"][1]["content"][0]["id"], "toolu_1");
        assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "toolu_1");
    }

    // A recorded text + tool_use streamed response.
    const FIXTURE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":24,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_42","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":31}}

event: message_stop
data: {"type":"message_stop"}
"#;

    #[test]
    fn decodes_text_then_tool_use_stream() {
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, FIXTURE).unwrap();
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageStart,
                StreamEvent::TextDelta {
                    text: "Let me check.".into()
                },
                StreamEvent::ContentBlockStop,
                StreamEvent::ToolUseStart {
                    id: "toolu_42".into(),
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
    fn captures_cache_usage_from_message_start() {
        const CACHED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":40,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}
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
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_read_tokens, 900);
        assert_eq!(usage.cache_write_tokens, 40);
    }

    #[test]
    fn captures_the_1h_cache_write_split_when_the_provider_breaks_it_out() {
        const SSE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":40,"cache_creation":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":30},"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
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
        // The flat sum still includes both TTLs; the 1h-specific field breaks out just that share.
        assert_eq!(usage.cache_write_tokens, 40);
        assert_eq!(usage.cache_write_1h_tokens, 30);
    }

    #[test]
    fn message_delta_refreshes_cache_counts_when_a_proxy_reports_them_there() {
        // Real Anthropic only ever reports cache fields on `message_start` — this is a defensive
        // refresh for a proxy that might report updated figures mid-stream, not something the real API
        // does; the initial `message_start` value must still be a sane baseline either way.
        const SSE: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":12,"cache_read_input_tokens":100,"cache_creation_input_tokens":10,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5,"cache_read_input_tokens":150,"cache_creation_input_tokens":10,"cache_creation":{"ephemeral_1h_input_tokens":10}}}

event: message_stop
data: {"type":"message_stop"}
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
        assert_eq!(usage.cache_read_tokens, 150); // refreshed from message_delta
        assert_eq!(usage.cache_write_1h_tokens, 10);
    }

    #[test]
    fn captures_reasoning_tokens_from_message_delta() {
        const SSE: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":10,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50,"output_tokens_details":{"thinking_tokens":32}}}

event: message_stop
data: {"type":"message_stop"}
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
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.reasoning_tokens, 32);
    }

    #[test]
    fn long_retention_sets_1h_ttl_on_breakpoints() {
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 256)
            .with_system("sys")
            .with_tools(vec![ToolDef {
                name: "read".into(),
                description: "d".into(),
                input_schema: json!({ "type": "object" }),
            }])
            .with_cache_long(true);
        let body = build_body(&req);
        assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["ttl"],
            "1h"
        );
    }

    #[test]
    fn tool_result_images_become_content_array() {
        use crate::message::ImageSource;
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "here is the screenshot".into(),
                is_error: false,
                images: vec![ImageSource::base64("image/png", "AAAA")],
            }])],
            256,
        );
        let body = build_body(&req);
        let content = &body["messages"][0]["content"][0]["content"];
        // The string content was rewritten into an array: text block, then image block.
        assert_eq!(
            content[0],
            json!({ "type": "text", "text": "here is the screenshot" })
        );
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "AAAA");
        // The transient `images` field must not leak onto the wire.
        assert!(body["messages"][0]["content"][0].get("images").is_none());
    }

    #[test]
    fn long_retention_gated_off_for_unsupported_model() {
        // Even with `cache_long`, a model whose capabilities don't include long-cache retention must
        // get the default 5-minute TTL (no `ttl` field) — otherwise Anthropic 400s the turn.
        let req = ModelRequest::new("some-unknown-model", vec![Message::user("hi")], 256)
            .with_system("sys")
            .with_cache_long(true);
        let body = build_body(&req);
        assert!(
            body["system"][0]["cache_control"].get("ttl").is_none(),
            "unsupported model must not receive the 1h TTL"
        );
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn build_body_emits_thinking_config() {
        // claude-opus-4-5 predates the adaptive requirement — still the `Budget`/`enabled` shape.
        let req = ModelRequest::new("claude-opus-4-5", vec![Message::user("hi")], 8192)
            .with_thinking(4096);
        let body = build_body(&req);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4096);
        assert_eq!(body["thinking"]["display"], "summarized");
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn build_body_emits_adaptive_thinking_config() {
        // claude-opus-4-8 (our default model) requires the adaptive shape: `output_config.effort` is a
        // sibling top-level field, not nested under `thinking`, and `display` must be set explicitly or
        // Anthropic silently omits visible reasoning text.
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 8192)
            .with_thinking(4096)
            .with_reasoning_effort(crate::transport::ReasoningEffort::High);
        let body = build_body(&req);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
        assert!(body["thinking"].get("budget_tokens").is_none());
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn thinking_block_round_trips_into_body_for_replay() {
        // A prior assistant turn with a signed thinking block must replay verbatim — Anthropic rejects
        // a tool turn whose thinking block is missing or unsigned.
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![
                Message::user("think then answer"),
                Message::assistant(vec![
                    ContentBlock::Thinking {
                        text: "let me reason".into(),
                        signature: "sig-abc".into(),
                    },
                    ContentBlock::Text {
                        text: "answer".into(),
                    },
                ]),
                Message::user("again"),
            ],
            8192,
        );
        let body = build_body(&req);
        let block = &body["messages"][1]["content"][0];
        assert_eq!(block["type"], "thinking");
        assert_eq!(block["thinking"], "let me reason");
        assert_eq!(block["signature"], "sig-abc");
    }

    #[test]
    fn decodes_thinking_then_text_stream() {
        const THINKING: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step one"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, THINKING).unwrap();
        assert!(events.contains(&StreamEvent::ThinkingDelta {
            text: "step one".into()
        }));
        assert!(events.contains(&StreamEvent::SignatureDelta {
            signature: "SIG".into()
        }));
    }

    #[test]
    fn refusal_stop_reason_is_distinct() {
        const REFUSED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, REFUSED).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Refusal
        }));
    }

    #[test]
    fn refusal_explanation_surfaces_as_text() {
        // A refusal carrying `stop_details.explanation` must surface that text (as a text delta) so
        // the caller can see *why* the model declined, rather than getting an empty turn.
        const REFUSED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","explanation":"I can't help with that."}},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, REFUSED).unwrap();
        assert!(events.contains(&StreamEvent::TextDelta {
            text: "I can't help with that.".into()
        }));
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Refusal
        }));
    }

    #[test]
    fn pause_turn_is_not_end_turn() {
        // `pause_turn` must not read as a clean `EndTurn` (which would truncate a turn the model meant
        // to continue); it maps to the non-terminal `Other`.
        const PAUSED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, PAUSED).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Other
        }));
        assert!(!events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn
        }));
    }

    #[test]
    fn sensitive_stop_reason_is_not_end_turn() {
        // Content flagged by safety filters must not read as success either — it shares `Refusal`'s
        // variant (no distinct explanation to surface, unlike an actual `refusal`) rather than
        // silently collapsing into `Other`/`EndTurn`.
        const FLAGGED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"sensitive"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, FLAGGED).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Refusal
        }));
        assert!(!events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::EndTurn
        }));
    }

    #[test]
    fn genuinely_unknown_stop_reason_falls_back_to_other_not_end_turn() {
        // A value Anthropic might add later that we don't recognize yet must not be misread as a clean
        // completion — it's conservatively `Other` (warn!-logged, not hard-failed; see `map_stop_reason`).
        const NOVEL: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"some_future_reason"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = Decoder::default();
        let events = decode_sse(&mut dec, NOVEL).unwrap();
        assert!(events.contains(&StreamEvent::MessageStop {
            stop_reason: StopReason::Other
        }));
    }

    #[test]
    fn tool_choice_emitted_only_when_set() {
        use crate::message::ToolDef;
        use crate::transport::ToolChoice;
        let tools = vec![ToolDef {
            name: "read".into(),
            description: "d".into(),
            input_schema: json!({ "type": "object" }),
        }];
        // Unset → no `tool_choice` on the wire (the default request shape is untouched).
        let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
            .with_tools(tools.clone());
        assert!(build_body(&req).get("tool_choice").is_none());

        // Each variant maps to Anthropic's vocabulary (`any` for required; `{type:"tool",name}`).
        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools.clone())
                .with_tool_choice(ToolChoice::Auto),
        );
        assert_eq!(body["tool_choice"], json!({ "type": "auto" }));

        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools.clone())
                .with_tool_choice(ToolChoice::None),
        );
        assert_eq!(body["tool_choice"], json!({ "type": "none" }));

        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools.clone())
                .with_tool_choice(ToolChoice::Required),
        );
        assert_eq!(body["tool_choice"], json!({ "type": "any" }));

        let body = build_body(
            &ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
                .with_tools(tools)
                .with_tool_choice(ToolChoice::Tool("read".into())),
        );
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "tool", "name": "read" })
        );
    }

    #[test]
    fn truncated_stream_is_rejected() {
        // Opens but never delivers `message_stop`.
        const TRUNCATED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, TRUNCATED).unwrap_err();
        assert!(matches!(err, Error::Transport(_)));
    }

    #[test]
    fn mid_stream_error_event_surfaces() {
        const ERRORED: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,"output_tokens":1}}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"server overloaded"}}
"#;
        let mut dec = Decoder::default();
        let err = decode_sse(&mut dec, ERRORED).unwrap_err();
        match err {
            Error::Transport(msg) => {
                assert!(msg.contains("overloaded"), "got: {msg}");
            }
            other => panic!("expected transport error, got {other:?}"),
        }
    }
}
