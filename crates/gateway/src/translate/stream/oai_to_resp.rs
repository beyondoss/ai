//! Stream: Chat Completions chunks → Responses events.

use super::*;

/// The open message item of a Responses stream.
struct OpenMessage {
    output_index: usize,
    id: String,
    parts: Vec<Value>,
    /// The part being streamed: `true` for a refusal, `false` for output text.
    part: Option<(bool, String)>,
}

/// The open reasoning item of a Responses stream.
struct OpenReasoning {
    output_index: usize,
    id: String,
    /// OpenRouter's `reasoning_details` index, when it numbers blocks.
    key: Option<u64>,
    text: String,
    /// `reasoning_summary_part.added` went out (only once there is text to show).
    part: bool,
    signature: Option<String>,
}

/// Chat Completions items → the Responses streaming event lifecycle: `response.created`,
/// `response.in_progress`, then per output item `output_item.added` → its content events →
/// `output_item.done`, and a terminal `response.completed` (or `response.incomplete`) carrying
/// the whole `output` and usage. Every event has a `sequence_number`; every item event names its
/// `output_index` and `item_id`.
#[derive(Default)]
pub(in crate::translate) struct OaiToResp {
    id: String,
    model: String,
    created_at: u64,
    seq: u64,
    pub(super) phase: Phase,
    /// Items by `output_index`; `Null` until the item is done.
    output: Vec<Value>,
    message: Option<OpenMessage>,
    reasoning: Option<OpenReasoning>,
    pub(super) calls: ToolCalls,
    finish: Option<&'static str>,
    usage: Option<Usage>,
    /// How the request's tools were reshaped, to name each call the way the client offered it.
    pub(super) tools: ToolNames,
    /// Assembling (see [`SseBridge::assembling`]): no event is written, and the end is kept in
    /// `whole` instead.
    pub(super) quiet: bool,
    pub(super) whole: Option<Whole>,
    /// Bytes of text and reasoning gathered (see [`SseBridge::accumulated`]).
    pub(in crate::translate) held: usize,
}

/// How an assembled stream ended.
pub(super) enum Whole {
    /// The finished Responses object, as `response.completed` (or `response.incomplete`) carries it.
    Done(Value),
    /// The upstream's error body, as it came (or the gateway's `stream_truncated`).
    Failed(Value),
}

impl OaiToResp {
    fn emit(&mut self, out: &mut Vec<u8>, typ: &str, mut body: Value) {
        if self.quiet {
            return;
        }
        if let Some(m) = body.as_object_mut() {
            m.insert("type".into(), json!(typ));
            m.insert("sequence_number".into(), json!(self.seq));
        }
        self.seq = self.seq.saturating_add(1);
        out.extend(sse_named(typ, &value_string(&body)));
    }

    pub(super) fn chunk(&mut self, v: &Value, out: &mut Vec<u8>) {
        if self.phase == Phase::Closed {
            return;
        }
        if self.phase == Phase::Unstarted {
            if let Some(id) = non_empty_str(v, "id") {
                id.clone_into(&mut self.id);
            }
            self.created_at = v.get("created").and_then(Value::as_u64).unwrap_or(0);
        }
        if self.model.is_empty()
            && let Some(model) = non_empty_str(v, "model")
        {
            model.clone_into(&mut self.model);
        }
        if self.phase == Phase::Unstarted {
            self.start(out);
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            if let Some(details) = delta
                .get("reasoning_details")
                .and_then(Value::as_array)
                .filter(|d| !d.is_empty())
            {
                for d in details {
                    let key = d.get("index").and_then(Value::as_u64);
                    match d.get("type").and_then(Value::as_str) {
                        Some("reasoning.text") => {
                            if let Some(text) = non_empty_str(d, "text") {
                                self.reasoning_text(key, Some(text), out);
                            }
                            if let Some(sig) = non_empty_str(d, "signature")
                                && anthropic_format(d)
                            {
                                self.reasoning_signature(key, sig, out);
                            }
                        }
                        Some("reasoning.summary") => {
                            if let Some(text) = non_empty_str(d, "summary") {
                                self.reasoning_text(key, Some(text), out);
                            }
                        }
                        _ => {}
                    }
                }
            } else if let Some(r) = non_empty_str(delta, "reasoning_content")
                .or_else(|| non_empty_str(delta, "reasoning"))
            {
                self.reasoning_text(None, Some(r), out);
            }
            // A whole signed block (see `AntToOai`) ends the reasoning item its text streamed into.
            // `redacted_thinking` has no Responses form.
            for b in delta
                .get("thinking")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                let block = anthropic_thinking_block(b);
                if let Some(sig) = non_empty_str(&block, "signature") {
                    let text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                    self.reasoning_block(text, sig, out);
                }
            }
            if let Some(t) = non_empty_str(delta, "content") {
                self.message_delta(false, t, out);
            }
            if let Some(r) = non_empty_str(delta, "refusal") {
                self.message_delta(true, r, out);
            }
            if let Some(calls) = delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .filter(|c| !c.is_empty())
            {
                self.close_reasoning(out);
                self.close_message(out);
                let mut steps = Vec::new();
                for c in calls {
                    self.calls.feed(c, &mut steps);
                }
                self.call_steps(steps, out);
            }
        }
        if let Some(finish) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            .filter(|f| !f.is_empty())
        {
            self.finish = Some(chat_finish(finish));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(Usage::from_chat(u));
        }
        if self.finish.is_some() && self.usage.is_some() {
            self.complete(out);
        }
    }

    fn response(&self, status: &str, incomplete: Value, usage: Value) -> Value {
        responses_object(
            self.id.clone(),
            self.created_at,
            json!(self.model),
            status,
            incomplete,
            self.output
                .iter()
                .filter(|i| !i.is_null())
                .cloned()
                .collect(),
            usage,
        )
    }

    fn start(&mut self, out: &mut Vec<u8>) {
        self.phase.open();
        if self.id.is_empty() {
            self.id = fresh_id("resp");
        }
        if self.created_at == 0 {
            self.created_at = unix_now();
        }
        let resp = self.response("in_progress", Value::Null, Value::Null);
        self.emit(out, "response.created", json!({ "response": resp }));
        self.emit(out, "response.in_progress", json!({ "response": resp }));
    }

    fn add_item(&mut self, out: &mut Vec<u8>, item: Value) -> usize {
        let output_index = self.output.len();
        self.output.push(Value::Null);
        self.emit(
            out,
            "response.output_item.added",
            json!({ "output_index": output_index, "item": item }),
        );
        output_index
    }

    fn done_item(&mut self, out: &mut Vec<u8>, output_index: usize, item: Value) {
        if let Some(slot) = self.output.get_mut(output_index) {
            *slot = item.clone();
        }
        self.emit(
            out,
            "response.output_item.done",
            json!({ "output_index": output_index, "item": item }),
        );
    }

    fn reasoning_text(&mut self, key: Option<u64>, text: Option<&str>, out: &mut Vec<u8>) {
        self.close_message(out);
        self.close_calls(out);
        // A signature ends its block; so does a new OpenRouter block index.
        if let Some(r) = &self.reasoning
            && (r.signature.is_some() || (key.is_some() && r.key.is_some() && key != r.key))
        {
            self.close_reasoning(out);
        }
        if self.reasoning.is_none() {
            let id = fresh_id("rs");
            let output_index =
                self.add_item(out, json!({ "type": "reasoning", "id": id, "summary": [] }));
            self.reasoning = Some(OpenReasoning {
                output_index,
                id,
                key,
                text: String::new(),
                part: false,
                signature: None,
            });
        }
        let Some(text) = text else {
            return;
        };
        let Some((output_index, id, first)) = self.reasoning.as_mut().map(|r| {
            (
                r.output_index,
                r.id.clone(),
                !std::mem::replace(&mut r.part, true),
            )
        }) else {
            return;
        };
        if first {
            self.emit(
                out,
                "response.reasoning_summary_part.added",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": "" },
                }),
            );
        }
        self.emit(
            out,
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": id,
                "output_index": output_index,
                "summary_index": 0,
                "delta": text,
            }),
        );
        if let Some(r) = self.reasoning.as_mut() {
            r.text.push_str(text);
            self.held = self.held.saturating_add(text.len());
        }
    }

    fn reasoning_signature(&mut self, key: Option<u64>, sig: &str, out: &mut Vec<u8>) {
        // Opens a reasoning item when none is (a signed block with no text shown), or a new one
        // when this signature belongs to another block.
        self.reasoning_text(key, None, out);
        if let Some(r) = self.reasoning.as_mut() {
            r.signature = Some(sig.to_owned());
        }
    }

    /// A whole signed thinking block. Its text already streamed into the open reasoning item when
    /// one is waiting for a signature; otherwise (display omitted, nothing streamed) it opens one.
    fn reasoning_block(&mut self, text: &str, sig: &str, out: &mut Vec<u8>) {
        let streamed = self
            .reasoning
            .as_ref()
            .is_some_and(|r| r.signature.is_none());
        if !streamed {
            self.reasoning_text(None, Some(text).filter(|t| !t.is_empty()), out);
        }
        if let Some(r) = self.reasoning.as_mut() {
            r.signature = Some(sig.to_owned());
        }
        self.close_reasoning(out);
    }

    fn close_reasoning(&mut self, out: &mut Vec<u8>) {
        let Some(r) = self.reasoning.take() else {
            return;
        };
        if r.part {
            self.emit(
                out,
                "response.reasoning_summary_text.done",
                json!({
                    "item_id": r.id,
                    "output_index": r.output_index,
                    "summary_index": 0,
                    "text": r.text,
                }),
            );
            self.emit(
                out,
                "response.reasoning_summary_part.done",
                json!({
                    "item_id": r.id,
                    "output_index": r.output_index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": r.text },
                }),
            );
        }
        let mut block = json!({ "type": "thinking", "thinking": r.text });
        if let (Some(sig), Some(m)) = (r.signature, block.as_object_mut()) {
            m.insert("signature".into(), json!(sig));
        }
        if let Some(item) = reasoning_item(&block, r.id) {
            self.done_item(out, r.output_index, item);
        }
    }

    fn message_delta(&mut self, refusal: bool, text: &str, out: &mut Vec<u8>) {
        self.close_reasoning(out);
        self.close_calls(out);
        if self.message.is_none() {
            let id = fresh_id("msg");
            let output_index = self.add_item(
                out,
                json!({
                    "type": "message",
                    "id": id,
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }),
            );
            self.message = Some(OpenMessage {
                output_index,
                id,
                parts: Vec::new(),
                part: None,
            });
        }
        if self
            .message
            .as_ref()
            .is_some_and(|m| m.part.as_ref().is_some_and(|(r, _)| *r != refusal))
        {
            self.close_part(out);
        }
        let Some((output_index, id, content_index, opening)) = self.message.as_mut().map(|m| {
            let opening = m.part.is_none();
            if opening {
                m.part = Some((refusal, String::new()));
            }
            (m.output_index, m.id.clone(), m.parts.len(), opening)
        }) else {
            return;
        };
        if opening {
            let part = if refusal {
                refusal_part("")
            } else {
                output_text_part("")
            };
            self.emit(
                out,
                "response.content_part.added",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            );
        }
        if refusal {
            self.emit(
                out,
                "response.refusal.delta",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": text,
                }),
            );
        } else {
            self.emit(
                out,
                "response.output_text.delta",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": text,
                    "logprobs": [],
                }),
            );
        }
        if let Some((_, buf)) = self.message.as_mut().and_then(|m| m.part.as_mut()) {
            buf.push_str(text);
            self.held = self.held.saturating_add(text.len());
        }
    }

    fn close_part(&mut self, out: &mut Vec<u8>) {
        let Some((output_index, id, content_index, (refusal, text))) =
            self.message.as_mut().and_then(|m| {
                m.part
                    .take()
                    .map(|p| (m.output_index, m.id.clone(), m.parts.len(), p))
            })
        else {
            return;
        };
        let part = if refusal {
            self.emit(
                out,
                "response.refusal.done",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "refusal": text,
                }),
            );
            refusal_part(&text)
        } else {
            self.emit(
                out,
                "response.output_text.done",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                    "logprobs": [],
                }),
            );
            output_text_part(&text)
        };
        self.emit(
            out,
            "response.content_part.done",
            json!({
                "item_id": id,
                "output_index": output_index,
                "content_index": content_index,
                "part": part,
            }),
        );
        if let Some(m) = self.message.as_mut() {
            m.parts.push(part);
        }
    }

    fn close_message(&mut self, out: &mut Vec<u8>) {
        self.close_part(out);
        let Some(m) = self.message.take() else {
            return;
        };
        let (status, _) = responses_status(self.finish);
        let item = json!({
            "type": "message",
            "id": m.id,
            "status": if status == "completed" { "completed" } else { "incomplete" },
            "role": "assistant",
            "content": m.parts,
        });
        self.done_item(out, m.output_index, item);
    }

    fn call_steps(&mut self, steps: Vec<CallStep>, out: &mut Vec<u8>) {
        for step in steps {
            match step {
                CallStep::Open(at) => {
                    let Some(call) = self.calls.calls.get_mut(at) else {
                        continue;
                    };
                    // The name the client offered the tool under (see `ToolNames`).
                    if !self.tools.is_empty() {
                        call.wrapped = self.tools.wrapped(&call.name);
                        call.custom |= call.wrapped;
                        let (name, namespace) = self.tools.restore(&call.name);
                        let (name, namespace) = (name.to_owned(), namespace.map(str::to_owned));
                        call.name = name;
                        call.namespace = namespace;
                    }
                    let (call_id, name, args, custom, wrapped) = (
                        call.id.clone(),
                        call.name.clone(),
                        call.args.clone(),
                        call.custom,
                        call.wrapped,
                    );
                    let namespace = call.namespace.clone();
                    let (item_id, item) = if custom {
                        let id = fresh_id("ctc");
                        let item =
                            custom_call_item(id.clone(), call_id, &name, namespace.as_deref(), "");
                        (id, item)
                    } else {
                        let id = fresh_id("fc");
                        let mut item = json!({
                            "type": "function_call",
                            "id": id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                            "status": "in_progress",
                        });
                        if let (Some(ns), Some(m)) = (namespace, item.as_object_mut()) {
                            m.insert("namespace".into(), json!(ns));
                        }
                        (id, item)
                    };
                    let output_index = self.add_item(out, item);
                    if let Some(call) = self.calls.calls.get_mut(at) {
                        call.slot = output_index;
                        call.item.clone_from(&item_id);
                    }
                    if !args.is_empty() && !wrapped {
                        self.args_delta(out, custom, &item_id, output_index, &args);
                    }
                }
                CallStep::Args(at, args) => {
                    let Some((item_id, output_index, custom)) = self
                        .calls
                        .calls
                        .get(at)
                        .filter(|c| !c.wrapped)
                        .map(|c| (c.item.clone(), c.slot, c.custom))
                    else {
                        continue;
                    };
                    self.args_delta(out, custom, &item_id, output_index, &args);
                }
            }
        }
    }

    fn args_delta(
        &mut self,
        out: &mut Vec<u8>,
        custom: bool,
        item_id: &str,
        output_index: usize,
        args: &str,
    ) {
        let typ = if custom {
            "response.custom_tool_call_input.delta"
        } else {
            "response.function_call_arguments.delta"
        };
        self.emit(
            out,
            typ,
            json!({ "item_id": item_id, "output_index": output_index, "delta": args }),
        );
    }

    fn close_calls(&mut self, out: &mut Vec<u8>) {
        let (status, _) = responses_status(self.finish);
        let status = if status == "completed" {
            "completed"
        } else {
            "incomplete"
        };
        let mut done = Vec::new();
        for call in &mut self.calls.calls {
            if call.opened && !call.closed {
                call.closed = true;
                done.push((
                    call.slot,
                    call.item.clone(),
                    call.id.clone(),
                    call.name.clone(),
                    call.args.clone(),
                    call.custom,
                    call.wrapped,
                    call.namespace.clone(),
                ));
            }
        }
        for (output_index, item_id, call_id, name, args, custom, wrapped, namespace) in done {
            if custom {
                // A wrapped call's input was held back until its arguments object was whole.
                let args = if wrapped {
                    let input = unwrap_custom_input(&args);
                    if !input.is_empty() {
                        self.args_delta(out, true, &item_id, output_index, &input);
                    }
                    input
                } else {
                    args
                };
                self.emit(
                    out,
                    "response.custom_tool_call_input.done",
                    json!({ "item_id": item_id, "output_index": output_index, "input": args }),
                );
                let item = custom_call_item(item_id, call_id, &name, namespace.as_deref(), &args);
                self.done_item(out, output_index, item);
                continue;
            }
            self.emit(
                out,
                "response.function_call_arguments.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "name": name,
                    "arguments": args,
                }),
            );
            let mut item = json!({
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "arguments": args,
                "status": status,
            });
            if let (Some(ns), Some(m)) = (namespace, item.as_object_mut()) {
                m.insert("namespace".into(), json!(ns));
            }
            self.done_item(out, output_index, item);
        }
    }

    fn complete(&mut self, out: &mut Vec<u8>) {
        if self.phase == Phase::Closed {
            return;
        }
        if self.phase == Phase::Unstarted {
            self.start(out);
        }
        let mut steps = Vec::new();
        self.calls.open_rest(&mut steps);
        if !steps.is_empty() {
            self.close_reasoning(out);
            self.close_message(out);
        }
        self.call_steps(steps, out);
        self.close_reasoning(out);
        self.close_message(out);
        self.close_calls(out);
        let (status, incomplete) = responses_status(self.finish);
        let usage = self.usage.unwrap_or_default().to_responses();
        let resp = self.response(status, incomplete, usage);
        let terminal = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        if self.quiet {
            self.whole = Some(Whole::Done(resp));
        } else {
            self.emit(out, terminal, json!({ "response": resp }));
        }
        self.phase = Phase::Closed;
    }

    pub(super) fn finish(&mut self, out: &mut Vec<u8>) {
        self.complete(out);
    }

    /// An upstream error: the Responses `error` event, then `response.failed`. The event also
    /// carries the error under `error`, so an OpenAI SDK raises it as an `APIError` with the
    /// upstream's message instead of ending on a missing `response.completed`.
    pub(super) fn error(&mut self, v: &Value, out: &mut Vec<u8>) {
        if self.phase == Phase::Closed {
            return;
        }
        if self.quiet {
            self.phase = Phase::Closed;
            self.whole = Some(Whole::Failed(v.clone()));
            return;
        }
        if self.phase == Phase::Unstarted {
            self.start(out);
        }
        self.phase = Phase::Closed;
        let envelope = map_error(v, Endpoint::Responses, 200);
        let err = envelope.get("error").cloned().unwrap_or(Value::Null);
        let field = |k: &str| err.get(k).cloned().unwrap_or(Value::Null);
        let (code, message, param) = (field("code"), field("message"), field("param"));
        self.emit(
            out,
            "error",
            json!({ "code": code, "message": message, "param": param, "error": err }),
        );
        let mut resp = self.response("failed", Value::Null, Value::Null);
        if let Some(m) = resp.as_object_mut() {
            let code = responses_error_code(code.as_str().unwrap_or(""));
            m.insert("error".into(), json!({ "code": code, "message": message }));
        }
        self.emit(out, "response.failed", json!({ "response": resp }));
    }
}

/// A failed Response's `error.code` is a closed set in the Responses schema (a typed client rejects
/// any other value, and then reports that instead of the error). The upstream's own code still
/// rides the `error` event, where any string is allowed. Two codes outside the published set pass
/// too: real OpenAI fills `response.failed` with them, and Codex reads them there (it compacts on
/// `context_length_exceeded` and stops retrying on `insufficient_quota`).
fn responses_error_code(code: &str) -> &'static str {
    const CODES: [&str; 23] = [
        "server_error",
        "rate_limit_exceeded",
        OPENAI_OVERFLOW,
        "insufficient_quota",
        "invalid_prompt",
        "data_residency_mismatch",
        "bio_policy",
        "misalignment_policy_violation",
        "vector_store_timeout",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "image_content_policy_violation",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
        "image_file_not_found",
    ];
    match code {
        "rate_limit_error" | "429" => "rate_limit_exceeded",
        _ => CODES
            .iter()
            .find(|c| **c == code)
            .copied()
            .unwrap_or("server_error"),
    }
}
