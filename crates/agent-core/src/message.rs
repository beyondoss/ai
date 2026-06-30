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
    /// Plain text — a prompt fragment or assistant prose.
    Text { text: String },
    /// The model's request to invoke a tool. `input` is the (already-complete) JSON arguments
    /// object; during streaming it's assembled from `StreamEvent::InputJsonDelta` fragments.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// The result of running a tool, fed back to the model. Carried on a `User` message (Anthropic
    /// convention). `is_error` lets the model see a failure as a value rather than a dead turn.
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// A single turn in the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// A user turn carrying a single text block.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// An assistant turn from already-assembled content blocks.
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
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
            }],
        }
    }

    /// The `ToolUse` blocks in this message, if any (what the loop dispatches each step).
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end of the assistant's turn.
    EndTurn,
    /// The model wants to call one or more tools; the loop should dispatch and continue.
    ToolUse,
    /// Hit the output token ceiling.
    MaxTokens,
    /// Hit a configured stop sequence.
    StopSequence,
    /// Anything else / dialect-specific.
    Other,
}

/// An incremental event from a streaming model response. The dialect adapters normalize both
/// providers' SSE shapes into this sequence; the loop consumes it to assemble assistant messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The assistant turn has begun.
    MessageStart,
    /// A chunk of assistant text.
    TextDelta { text: String },
    /// A tool-call block opened; `id` and `name` are known before its arguments stream in.
    ToolUseStart { id: String, name: String },
    /// A chunk of the in-progress tool call's JSON arguments.
    InputJsonDelta { partial_json: String },
    /// The current content block finished (text or tool-call).
    ContentBlockStop,
    /// Token accounting. May arrive at end-of-stream (OpenAI) or alongside other events (Anthropic).
    Usage {
        input_tokens: u32,
        output_tokens: u32,
    },
    /// The assistant turn finished, with the reason it stopped.
    MessageStop { stop_reason: StopReason },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_block_tool_use_round_trips() {
        let block = ContentBlock::ToolUse {
            id: "tu_1".into(),
            name: "read".into(),
            input: json!({ "path": "README.md" }),
        };
        let wire = serde_json::to_value(&block).unwrap();
        // Internally-tagged on `type`, snake_case — the shape the dialect adapters key on.
        assert_eq!(wire["type"], "tool_use");
        assert_eq!(wire["name"], "read");
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
                is_error: false
            }
        );
    }

    #[test]
    fn tool_uses_extracts_only_tool_calls() {
        let msg = Message::assistant(vec![
            ContentBlock::Text {
                text: "let me look".into(),
            },
            ContentBlock::ToolUse {
                id: "a".into(),
                name: "read".into(),
                input: json!({}),
            },
            ContentBlock::ToolUse {
                id: "b".into(),
                name: "bash".into(),
                input: json!({}),
            },
        ]);
        let calls: Vec<_> = msg.tool_uses().map(|(id, name, _)| (id, name)).collect();
        assert_eq!(calls, vec![("a", "read"), ("b", "bash")]);
    }

    #[test]
    fn stream_event_tag_is_snake_case() {
        let ev = StreamEvent::InputJsonDelta {
            partial_json: "{\"p\":".into(),
        };
        assert_eq!(
            serde_json::to_value(&ev).unwrap()["type"],
            "input_json_delta"
        );
    }
}
