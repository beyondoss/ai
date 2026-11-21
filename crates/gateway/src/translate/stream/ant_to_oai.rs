//! Stream: Messages events → Chat Completions chunks.

use super::*;

/// Anthropic Messages events → Chat Completions items.
///
/// Thinking reaches a Chat client twice, as the non-stream body carries it: its text as
/// `reasoning_content` deltas while it streams, and each finished block whole — signed, or
/// redacted — as one entry of the `thinking` list, under its own `index` so an OpenAI SDK's
/// accumulator (which requires one on every list entry) keeps blocks apart. The accumulated
/// message is then exactly what the request side reads back onto Messages; a signature on a
/// string field would concatenate across blocks ("S1S2") and could never be split again.
#[derive(Default)]
pub(super) struct AntToOai {
    meta: ChunkMeta,
    /// Tool calls opened so far: the next one's Chat `index`.
    tools: u32,
    /// The block now open, when it is a client `tool_use`. Anthropic blocks never interleave, so a
    /// JSON delta belongs to it; one for any other block (a server tool's input) must not land on a
    /// client call's arguments.
    open_tool: Option<OpenTool>,
    /// The thinking block now open, gathered until its `content_block_stop`.
    open_thinking: Option<HeldThinking>,
    /// `thinking` list entries sent so far: the next one's `index`.
    thinking_blocks: u32,
    usage: Usage,
    /// The request's cache breakpoints were the gateway's: its writes show as input.
    pub(super) gateway_cache: bool,
}

/// A client `tool_use` block streaming to a Chat client.
struct OpenTool {
    /// Its Chat `index`.
    index: u32,
    /// Whether any argument bytes went out. A block that closes with none still owes the client
    /// JSON arguments: the input `content_block_start` carried (a Messages-compatible host may
    /// send it whole there), else `{}` (a zero-argument call streams one empty delta). `""` is
    /// not JSON, and the non-stream body says `{}`.
    sent: bool,
    /// `content_block_start`'s `input`, when it was not the usual empty object.
    start_input: Option<String>,
}

/// A streaming Anthropic thinking block, held whole so it can be sent back.
#[derive(Default)]
pub(in crate::translate) struct HeldThinking {
    pub(in crate::translate) text: String,
    signature: String,
    /// Past [`MAX_HELD_THINKING`]: dropped, since a signature covers the whole text.
    pub(in crate::translate) overflow: bool,
}

impl HeldThinking {
    pub(in crate::translate) fn push(&mut self, text: &str) {
        if self.overflow {
            return;
        }
        if self.text.len().saturating_add(text.len()) > MAX_HELD_THINKING {
            self.overflow = true;
            self.text = String::new();
            return;
        }
        self.text.push_str(text);
    }
}

impl AntToOai {
    pub(super) fn event(&mut self, event: &str, data: &str, items: &mut Vec<ChatItem>) {
        if event == "ping" {
            return;
        }
        if data == "[DONE]" {
            items.push(ChatItem::Done);
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let typ = if event.is_empty() {
            v.get("type").and_then(Value::as_str).unwrap_or("")
        } else {
            event
        };
        if typ == "error" || looks_like_error(&v) {
            items.push(ChatItem::Error(v));
            return;
        }
        match typ {
            "message_start" => {
                let msg = v.get("message").unwrap_or(&v);
                self.meta.id = id_or_fresh(msg.get("id"), "chatcmpl");
                if let Some(model) = msg.get("model").and_then(Value::as_str) {
                    model.clone_into(&mut self.meta.model);
                }
                self.meta.created = unix_now();
                if let Some(u) = msg.get("usage") {
                    self.usage.merge_anthropic(u);
                }
                items.push(
                    self.meta
                        .chunk(json!({ "role": "assistant", "content": "" }), None),
                );
            }
            "content_block_start" => {
                let block = v.get("content_block").unwrap_or(&Value::Null);
                self.open_tool = None;
                self.open_thinking = None;
                let delta = match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        let index = self.tools;
                        self.tools = self.tools.saturating_add(1);
                        self.open_tool = Some(OpenTool {
                            index,
                            sent: false,
                            start_input: block
                                .get("input")
                                .filter(|i| i.as_object().is_some_and(|o| !o.is_empty()))
                                .map(value_string),
                        });
                        json!({ "tool_calls": [{
                            "index": index,
                            "id": id_or_fresh(block.get("id"), "call"),
                            "type": "function",
                            "function": {
                                "name": block.get("name").and_then(Value::as_str).unwrap_or(""),
                                "arguments": "",
                            },
                        }] })
                    }
                    Some("redacted_thinking") => self.thinking_entry(json!({
                        "type": "redacted_thinking",
                        "data": block.get("data").cloned().unwrap_or(json!("")),
                    })),
                    Some("text") => match non_empty_str(block, "text") {
                        Some(t) => json!({ "content": t }),
                        None => return,
                    },
                    Some("thinking") => {
                        let mut held = HeldThinking::default();
                        let text = non_empty_str(block, "thinking");
                        held.push(text.unwrap_or(""));
                        if let Some(sig) = non_empty_str(block, "signature") {
                            held.signature.push_str(sig);
                        }
                        self.open_thinking = Some(held);
                        match text {
                            Some(t) => reasoning_delta(t),
                            None => return,
                        }
                    }
                    _ => return,
                };
                items.push(self.meta.chunk(delta, None));
            }
            "content_block_delta" => {
                let d = v.get("delta").unwrap_or(&Value::Null);
                let delta = match d.get("type").and_then(Value::as_str) {
                    Some("text_delta") => match non_empty_str(d, "text") {
                        Some(t) => json!({ "content": t }),
                        None => return,
                    },
                    Some("input_json_delta") => {
                        match (self.open_tool.as_mut(), non_empty_str(d, "partial_json")) {
                            (Some(tool), Some(p)) => {
                                tool.sent = true;
                                json!({ "tool_calls": [{
                                    "index": tool.index,
                                    "function": { "arguments": p },
                                }] })
                            }
                            _ => return,
                        }
                    }
                    Some("thinking_delta") => {
                        match non_empty_str(d, "thinking").or_else(|| non_empty_str(d, "text")) {
                            Some(t) => {
                                if let Some(held) = self.open_thinking.as_mut() {
                                    held.push(t);
                                }
                                reasoning_delta(t)
                            }
                            None => return,
                        }
                    }
                    // Held for the block's `content_block_stop`, where the block goes out whole.
                    Some("signature_delta") => {
                        if let (Some(held), Some(sig)) =
                            (self.open_thinking.as_mut(), non_empty_str(d, "signature"))
                        {
                            held.signature.push_str(sig);
                        }
                        return;
                    }
                    _ => return,
                };
                items.push(self.meta.chunk(delta, None));
            }
            "message_delta" => {
                let stop = v
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .or_else(|| v.get("stop_reason").and_then(Value::as_str));
                if let Some(u) = v.get("usage") {
                    self.usage.merge_anthropic(u);
                }
                if stop == Some("refusal")
                    && let Some(why) = v
                        .pointer("/delta/stop_details/explanation")
                        .and_then(Value::as_str)
                {
                    items.push(self.meta.chunk(json!({ "refusal": why }), None));
                }
                items.push(self.meta.chunk(json!({}), Some(map_stop_to_openai(stop))));
                items.push(
                    self.meta
                        .usage(self.usage.writes_as_input(self.gateway_cache).to_chat()),
                );
            }
            "content_block_stop" => {
                if let Some(tool) = self.open_tool.take()
                    && !tool.sent
                {
                    let args = tool.start_input.unwrap_or_else(|| "{}".to_owned());
                    items.push(self.meta.chunk(
                        json!({ "tool_calls": [{
                            "index": tool.index,
                            "function": { "arguments": args },
                        }] }),
                        None,
                    ));
                }
                // A thinking block is whole now. Unsigned (or past the cap), it could never be sent
                // back, so it is not offered for replay; its text already streamed.
                if let Some(held) = self.open_thinking.take()
                    && !held.overflow
                    && !held.signature.is_empty()
                {
                    let delta = self.thinking_entry(json!({
                        "type": "thinking",
                        "thinking": held.text,
                        "signature": held.signature,
                    }));
                    items.push(self.meta.chunk(delta, None));
                }
            }
            "message_stop" => items.push(ChatItem::Done),
            _ => {}
        }
    }

    /// A delta adding one finished block to the Chat message's `thinking` list.
    fn thinking_entry(&mut self, block: Value) -> Value {
        let mut entry = Map::new();
        entry.insert("index".into(), json!(self.thinking_blocks));
        if let Value::Object(fields) = block {
            entry.extend(fields);
        }
        self.thinking_blocks = self.thinking_blocks.saturating_add(1);
        json!({ "thinking": [Value::Object(entry)] })
    }
}
