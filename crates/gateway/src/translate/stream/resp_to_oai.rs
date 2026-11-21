//! Stream: Responses events → Chat Completions chunks.

use super::*;

/// A function (or custom tool) call a Responses upstream announced, and where its Chat deltas go.
struct RespCall {
    index: u32,
    /// Argument bytes already went out, so a closing event's full `arguments` must not repeat them.
    args_sent: bool,
    /// A `custom_tool_call`: its raw `input` rides the Chat call's `custom.input`.
    custom: bool,
}

/// A Responses output item that is a client tool call, and whether it is a custom one.
fn resp_call_kind(item: &Value) -> Option<bool> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => Some(false),
        Some("custom_tool_call") => Some(true),
        _ => None,
    }
}

/// Responses events → Chat Completions items.
#[derive(Default)]
pub(super) struct RespToOai {
    meta: ChunkMeta,
    started: bool,
    calls: Vec<RespCall>,
    /// The first call per `output_index` and per item id: a delta finds its call in O(1), not by
    /// a scan of every call so far.
    by_output: rustc_hash::FxHashMap<u64, usize>,
    by_item: rustc_hash::FxHashMap<Box<str>, usize>,
    /// Bytes of call bookkeeping held (see [`SseBridge::accumulated`]).
    pub(super) held: usize,
}

impl RespToOai {
    pub(super) fn event(&mut self, event: &str, data: &str, items: &mut Vec<ChatItem>) {
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
            "response.created" | "response.in_progress" => {
                let r = v.get("response").unwrap_or(&v);
                if self.meta.id.is_empty() {
                    self.meta.id = id_or_fresh(r.get("id"), "chatcmpl");
                    self.meta.created = created_of(r.get("created_at"));
                }
                if let Some(model) = non_empty_str(r, "model") {
                    model.clone_into(&mut self.meta.model);
                }
                self.start(items);
            }
            "response.output_item.added" => {
                self.start(items);
                let item = v.get("item").unwrap_or(&Value::Null);
                if resp_call_kind(item).is_some() {
                    self.open_call(&v, item, items);
                }
            }
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                if let Some(at) = self.find(&v)
                    && let Some(delta) = non_empty_str(&v, "delta")
                {
                    self.args(at, delta, items);
                }
            }
            "response.function_call_arguments.done"
            | "response.custom_tool_call_input.done"
            | "response.output_item.done" => {
                let item = v.get("item");
                if item.is_some_and(|i| resp_call_kind(i).is_none()) {
                    return;
                }
                let args = item
                    .unwrap_or(&v)
                    .get("arguments")
                    .or_else(|| item.unwrap_or(&v).get("input"))
                    .and_then(Value::as_str);
                match self.find(&v) {
                    Some(at) => {
                        if let Some(args) = args.filter(|a| !a.is_empty())
                            && self.calls.get(at).is_some_and(|c| !c.args_sent)
                        {
                            self.args(at, args, items);
                        }
                    }
                    // A call never announced by `output_item.added`: announce it whole.
                    None => {
                        if let Some(item) = item {
                            self.start(items);
                            self.open_call(&v, item, items);
                        }
                    }
                }
            }
            "response.output_text.delta" => {
                if let Some(t) = text_delta_of(&v) {
                    items.push(self.meta.chunk(json!({ "content": t }), None));
                }
            }
            "response.refusal.delta" => {
                if let Some(t) = text_delta_of(&v) {
                    items.push(self.meta.chunk(json!({ "refusal": t }), None));
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(t) = text_delta_of(&v) {
                    items.push(self.meta.chunk(reasoning_delta(t), None));
                }
            }
            // Separate one summary part from the next, as a single reasoning text.
            "response.reasoning_summary_part.added" => {
                if v.get("summary_index").and_then(Value::as_u64) > Some(0) {
                    items.push(self.meta.chunk(reasoning_delta("\n\n"), None));
                }
            }
            "response.completed" | "response.incomplete" => {
                let r = v.get("response").unwrap_or(&v);
                self.start(items);
                let mut finish = responses_finish(r);
                if finish == "stop" && !self.calls.is_empty() {
                    finish = "tool_calls";
                }
                items.push(self.meta.chunk(json!({}), Some(finish)));
                let usage = Usage::from_responses(r.get("usage").unwrap_or(&Value::Null));
                items.push(self.meta.usage(usage.to_chat()));
                items.push(ChatItem::Done);
            }
            "response.failed" => {
                let err = v
                    .pointer("/response/error")
                    .filter(|e| e.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({ "message": "the upstream response failed" }));
                items.push(ChatItem::Error(json!({ "error": err })));
            }
            _ => {}
        }
    }

    fn start(&mut self, items: &mut Vec<ChatItem>) {
        if !self.started {
            self.started = true;
            items.push(
                self.meta
                    .chunk(json!({ "role": "assistant", "content": "" }), None),
            );
        }
    }

    fn open_call(&mut self, v: &Value, item: &Value, items: &mut Vec<ChatItem>) {
        let index = u32::try_from(self.calls.len()).unwrap_or(u32::MAX);
        let custom = resp_call_kind(item) == Some(true);
        let args = item
            .get(if custom { "input" } else { "arguments" })
            .and_then(Value::as_str)
            .unwrap_or("");
        let output_index = v.get("output_index").and_then(Value::as_u64);
        let item_id = non_empty_str(item, "id");
        let at = self.calls.len();
        if let Some(o) = output_index {
            self.by_output.entry(o).or_insert(at);
        }
        if let Some(id) = item_id
            && !self.by_item.contains_key(id)
        {
            self.by_item.insert(id.into(), at);
        }
        self.held = self.held.saturating_add(
            std::mem::size_of::<RespCall>().saturating_add(item_id.map_or(0, str::len)),
        );
        self.calls.push(RespCall {
            index,
            args_sent: !args.is_empty(),
            custom,
        });
        let id = id_or_fresh(item.get("call_id").or_else(|| item.get("id")), "call");
        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
        let call = if custom {
            json!({
                "index": index, "id": id, "type": "custom",
                "custom": { "name": name, "input": args },
            })
        } else {
            json!({
                "index": index, "id": id, "type": "function",
                "function": { "name": name, "arguments": args },
            })
        };
        items.push(self.meta.chunk(json!({ "tool_calls": [call] }), None));
    }

    fn args(&mut self, at: usize, args: &str, items: &mut Vec<ChatItem>) {
        let Some(call) = self.calls.get_mut(at) else {
            return;
        };
        call.args_sent = true;
        let index = call.index;
        let delta = if call.custom {
            json!({ "index": index, "custom": { "input": args } })
        } else {
            json!({ "index": index, "function": { "arguments": args } })
        };
        items.push(self.meta.chunk(json!({ "tool_calls": [delta] }), None));
    }

    /// The call an event is about, by `output_index`, else by item id.
    fn find(&self, v: &Value) -> Option<usize> {
        let output_index = v.get("output_index").and_then(Value::as_u64);
        let item_id = non_empty_str(v, "item_id")
            .or_else(|| v.get("item").and_then(|i| non_empty_str(i, "id")));
        // The first call matching either key.
        let by_output = output_index.and_then(|o| self.by_output.get(&o).copied());
        let by_item = item_id.and_then(|id| self.by_item.get(id).copied());
        match (by_output, by_item) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
}

fn text_delta_of(v: &Value) -> Option<&str> {
    match v.get("delta") {
        Some(Value::String(s)) => Some(s.as_str()),
        Some(d) => d.get("text").and_then(Value::as_str),
        None => None,
    }
    .filter(|s| !s.is_empty())
}
