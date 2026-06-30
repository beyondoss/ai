//! Anthropic Messages wire (`/v1/messages`).
//!
//! The harness's internal model was chosen to be Anthropic-shaped (content blocks,
//! `tool_use`/`tool_result`), so the request mapping is nearly an identity and the SSE decoder is a
//! direct translation of Anthropic's `content_block_*` / `message_*` events.

use serde_json::{Map, Value, json};

use super::StreamDecoder;
use crate::message::{StopReason, StreamEvent};
use crate::transport::ModelRequest;

/// Build the streaming request body. `system` is hoisted to a top-level field (Anthropic keeps it
/// out of `messages`); `messages` and `tools` serialize straight from the internal model.
///
/// Two prompt-cache breakpoints are stamped in (see [`mark_cache_breakpoint`]/[`mark_last_tool`]).
/// An agent loop re-sends an ever-growing prefix — tools, then system, then the whole prior
/// conversation — on every turn; without caching each turn re-bills that entire prefix at full input
/// price, an O(n²) token cost over a `max_steps`-deep run. Anthropic caches the request prefix up to
/// each `cache_control` mark (reads cost ~10% of input tokens), so we anchor one breakpoint on the
/// fixed tool block and roll a second one onto the last message to capture the conversation so far.
pub fn build_body(req: &ModelRequest) -> Value {
    let mut map = Map::new();
    map.insert("model".into(), Value::String(req.model.clone()));
    map.insert("max_tokens".into(), Value::from(req.max_tokens));
    map.insert("stream".into(), Value::Bool(true));

    // Rolling breakpoint: cache the conversation prefix (tools + system + every prior message) up to
    // the final block, so next turn the whole accumulated transcript is a cache read, not a re-bill.
    let mut messages = serde_json::to_value(req.messages.as_ref()).unwrap_or(Value::Null);
    mark_cache_breakpoint(&mut messages);
    map.insert("messages".into(), messages);

    if let Some(system) = &req.system {
        map.insert("system".into(), Value::String(system.clone()));
    }
    if !req.tools.is_empty() {
        // Anchor breakpoint: the tool definitions (ten JSON schemas) are identical every turn and sit
        // at the front of the cache order, so this entry stays warm even when the rolling message
        // breakpoint is rewritten each turn. Requires stable tool ordering — see `definitions()`.
        let mut tools = serde_json::to_value(req.tools.as_ref()).unwrap_or(Value::Null);
        mark_last_tool(&mut tools);
        map.insert("tools".into(), tools);
    }
    Value::Object(map)
}

/// Stamp an ephemeral cache breakpoint onto the last content block of the last message. No-op if the
/// history is empty or the final message carries no content blocks.
fn mark_cache_breakpoint(messages: &mut Value) {
    if let Some(block) = messages
        .as_array_mut()
        .and_then(|msgs| msgs.last_mut())
        .and_then(|m| m.get_mut("content"))
        .and_then(Value::as_array_mut)
        .and_then(|content| content.last_mut())
        .and_then(Value::as_object_mut)
    {
        block.insert("cache_control".into(), json!({ "type": "ephemeral" }));
    }
}

/// Stamp an ephemeral cache breakpoint onto the last tool definition.
fn mark_last_tool(tools: &mut Value) {
    if let Some(tool) = tools
        .as_array_mut()
        .and_then(|t| t.last_mut())
        .and_then(Value::as_object_mut)
    {
        tool.insert("cache_control".into(), json!({ "type": "ephemeral" }));
    }
}

fn map_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        _ => StopReason::Other,
    }
}

/// Decodes Anthropic SSE. Tracks token counts (input from `message_start`, output from
/// `message_delta`) and the stop reason, emitting a single `Usage` + `MessageStop` at `message_stop`.
#[derive(Default)]
pub struct Decoder {
    input_tokens: u32,
    output_tokens: u32,
    stop_reason: StopReason,
}

impl StreamDecoder for Decoder {
    fn push(&mut self, data: &Value) -> Vec<StreamEvent> {
        let kind = data.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "message_start" => {
                let usage = data.get("message").and_then(|m| m.get("usage"));
                self.input_tokens = u32_at(usage, "input_tokens");
                self.output_tokens = u32_at(usage, "output_tokens");
                vec![StreamEvent::MessageStart]
            }
            "content_block_start" => {
                let block = data.get("content_block");
                if block.and_then(|b| b.get("type")).and_then(Value::as_str) == Some("tool_use") {
                    let id = str_at(block, "id").to_string();
                    let name = str_at(block, "name").to_string();
                    vec![StreamEvent::ToolUseStart { id, name }]
                } else {
                    // A text block opening carries no event in our model — text accrues via deltas.
                    Vec::new()
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
                let out = u32_at(data.get("usage"), "output_tokens");
                if out > 0 {
                    self.output_tokens = out;
                }
                Vec::new()
            }
            "message_stop" => vec![
                StreamEvent::Usage {
                    input_tokens: self.input_tokens,
                    output_tokens: self.output_tokens,
                },
                StreamEvent::MessageStop {
                    stop_reason: self.stop_reason,
                },
            ],
            _ => Vec::new(),
        }
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
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["tools"][0]["name"], "read");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn build_body_stamps_cache_breakpoints() {
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::user("hi"), Message::tool_result("tu_1", "out", false)],
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
        assert!(body["messages"][0]["content"][0].get("cache_control").is_none());
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
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
                StreamEvent::Usage {
                    input_tokens: 24,
                    output_tokens: 31
                },
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
    }
}
