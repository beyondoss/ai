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
use crate::message::{ContentBlock, Role, StopReason, StreamEvent};
use crate::transport::ModelRequest;

fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
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
                let text = text_of(&m.content);
                if !text.is_empty() {
                    messages.push(json!({ "role": "user", "content": text }));
                }
                // Tool results fan out into individual `role:"tool"` messages.
                for b in &m.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = b
                    {
                        messages.push(json!({ "role": "tool", "tool_call_id": tool_use_id, "content": content }));
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
                messages.push(Value::Object(msg));
            }
        }
    }

    let mut map = Map::new();
    map.insert("model".into(), json!(req.model));
    map.insert("max_tokens".into(), json!(req.max_tokens));
    map.insert("stream".into(), json!(true));
    // Ask for a trailing usage chunk so token accounting works on the streaming path.
    map.insert("stream_options".into(), json!({ "include_usage": true }));
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
    Value::Object(map)
}

fn map_finish_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::Other,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Open {
    None,
    Text,
    Tool,
}

/// Decodes OpenAI SSE. Synthesizes the block-stop boundaries OpenAI omits and holds `MessageStop`
/// until `finish()` so it lands after the trailing usage chunk.
pub struct Decoder {
    started: bool,
    open: Open,
    stop_reason: StopReason,
    input_tokens: u32,
    output_tokens: u32,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            started: false,
            open: Open::None,
            stop_reason: StopReason::EndTurn,
            input_tokens: 0,
            output_tokens: 0,
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
            self.input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            self.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            out.push(StreamEvent::Usage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
            });
        }

        let Some(choice) = data.get("choices").and_then(|c| c.get(0)) else {
            return out;
        };
        let delta = choice.get("delta");

        // Plain text content.
        if let Some(text) = delta.and_then(|d| d.get("content")).and_then(Value::as_str) {
            if !text.is_empty() {
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
            self.stop_reason = map_finish_reason(Some(reason));
            self.close_open(&mut out);
        }

        out
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        if self.started {
            vec![StreamEvent::MessageStop {
                stop_reason: self.stop_reason,
            }]
        } else {
            Vec::new()
        }
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
