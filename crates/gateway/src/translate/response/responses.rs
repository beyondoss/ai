//! Response JSON: Responses ↔ Chat Completions.

use crate::translate::*;

/// A Chat Completions response for a Responses client. Items come in the order OpenAI emits
/// them: reasoning, the message, then function calls.
pub(in crate::translate) fn openai_resp_to_responses(v: &Value, tools: &ToolNames) -> Value {
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first());
    let message = choice
        .and_then(|c| c.get("message"))
        .unwrap_or(&Value::Null);
    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str);
    let (status, incomplete) = responses_status(finish);
    let item_status = if status == "completed" {
        "completed"
    } else {
        "incomplete"
    };
    let mut output: Vec<Value> = chat_thinking_blocks(message)
        .iter()
        .filter_map(|b| reasoning_item(b, fresh_id("rs")))
        .collect();
    let mut parts = Vec::new();
    match message.get("content") {
        Some(Value::String(s)) if !s.is_empty() => parts.push(output_text_part(s)),
        Some(Value::Array(content)) => {
            for p in content {
                match p.get("type").and_then(Value::as_str) {
                    Some("refusal") => parts.push(refusal_part(
                        p.get("refusal").and_then(Value::as_str).unwrap_or(""),
                    )),
                    _ => {
                        if let Some(t) = non_empty_str(p, "text") {
                            parts.push(output_text_part(t));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    if let Some(r) = non_empty_str(message, "refusal") {
        parts.push(refusal_part(r));
    }
    if !parts.is_empty() {
        output.push(json!({
            "type": "message",
            "id": fresh_id("msg"),
            "role": "assistant",
            "status": item_status,
            "content": parts,
        }));
    }
    for c in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        // A custom (free-form) tool call is its own item type on Responses, with raw `input`.
        if let Some(custom) = c
            .get("custom")
            .filter(|_| c.get("type").and_then(Value::as_str) != Some("function"))
        {
            let flat = custom.get("name").and_then(Value::as_str).unwrap_or("");
            let (name, namespace) = tools.restore(flat);
            let input = custom.get("input").and_then(Value::as_str).unwrap_or("");
            output.push(custom_call_item(
                fresh_id("ctc"),
                id_or_fresh(c.get("id"), "call"),
                name,
                namespace,
                input,
            ));
            continue;
        }
        let func = c.get("function").unwrap_or(c);
        let flat = func.get("name").and_then(Value::as_str).unwrap_or("");
        let (name, namespace) = tools.restore(flat);
        let arguments = arguments_string(func.get("arguments"));
        if tools.wrapped(flat) {
            output.push(custom_call_item(
                fresh_id("ctc"),
                id_or_fresh(c.get("id"), "call"),
                name,
                namespace,
                &unwrap_custom_input(&arguments),
            ));
            continue;
        }
        let mut item = json!({
            "type": "function_call",
            "id": fresh_id("fc"),
            "call_id": id_or_fresh(c.get("id"), "call"),
            "name": name,
            "arguments": arguments,
            "status": item_status,
        });
        if let (Some(ns), Some(m)) = (namespace, item.as_object_mut()) {
            m.insert("namespace".into(), json!(ns));
        }
        output.push(item);
    }
    responses_object(
        id_or_fresh(v.get("id"), "resp"),
        v.get("created")
            .and_then(Value::as_u64)
            .unwrap_or_else(unix_now),
        v.get("model").cloned().unwrap_or(json!("")),
        status,
        incomplete,
        output,
        Usage::from_chat(v.get("usage").unwrap_or(&Value::Null)).to_responses(),
    )
}

/// A Responses `custom_tool_call` item.
pub(in crate::translate) fn custom_call_item(
    id: String,
    call_id: String,
    name: &str,
    namespace: Option<&str>,
    input: &str,
) -> Value {
    let mut item = json!({
        "type": "custom_tool_call",
        "id": id,
        "call_id": call_id,
        "name": name,
        "input": input,
    });
    if let (Some(ns), Some(m)) = (namespace, item.as_object_mut()) {
        m.insert("namespace".into(), json!(ns));
    }
    item
}

/// A wrapped `custom` tool's raw input, from the `{"input": "…"}` arguments the model called it with
/// (see [`wrapped_custom_tool`]). Arguments of any other shape are passed as they came: the client
/// then sees what the model actually sent.
pub(in crate::translate) fn unwrap_custom_input(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(mut m)) => match m.remove("input") {
            Some(Value::String(s)) => s,
            _ => arguments.to_owned(),
        },
        _ => arguments.to_owned(),
    }
}

/// A Responses object with every field the schema requires. The request's `tools` and
/// `tool_choice` are not known here; the defaults stand in for them.
pub(in crate::translate) fn responses_object(
    id: String,
    created_at: u64,
    model: Value,
    status: &str,
    incomplete: Value,
    output: Vec<Value>,
    usage: Value,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": Value::Null,
        "incomplete_details": incomplete,
        "model": model,
        "output": output,
        "parallel_tool_calls": true,
        // Truthful: a translated response is kept nowhere, so nothing can point back into it.
        "store": false,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    })
}

/// A Responses `reasoning` item for a thinking block: its text as a summary, its Anthropic
/// signature (when there was one) behind [`GATEWAY_SIGNATURE_PREFIX`] as `encrypted_content`.
/// `redacted_thinking` has no text to show and no slot of its own on Responses, and is dropped.
pub(in crate::translate) fn reasoning_item(block: &Value, id: String) -> Option<Value> {
    if block.get("type").and_then(Value::as_str) != Some("thinking") {
        return None;
    }
    let text = block.get("thinking").and_then(Value::as_str).unwrap_or("");
    let summary = if text.is_empty() {
        json!([])
    } else {
        json!([{ "type": "summary_text", "text": text }])
    };
    let mut item = Map::new();
    item.insert("type".into(), json!("reasoning"));
    item.insert("id".into(), json!(id));
    item.insert("summary".into(), summary);
    if let Some(sig) = non_empty_str(block, "signature") {
        item.insert(
            "encrypted_content".into(),
            json!(format!("{GATEWAY_SIGNATURE_PREFIX}{sig}")),
        );
    }
    Some(Value::Object(item))
}

/// Tool arguments as the JSON text Chat Completions and Responses carry them in.
fn arguments_string(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => "{}".to_owned(),
        Some(other) => arguments_text(other),
    }
}

/// A Responses response for a Chat Completions client.
pub(in crate::translate) fn responses_resp_to_openai(v: &Value) -> Value {
    let mut text = String::new();
    let mut refusal = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for item in v
        .get("output")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => tool_calls.push(json!({
                "id": id_or_fresh(item.get("call_id").or_else(|| item.get("id")), "call"),
                "type": "function",
                "function": {
                    "name": item.get("name").and_then(Value::as_str).unwrap_or(""),
                    "arguments": arguments_string(item.get("arguments")),
                },
            })),
            Some("custom_tool_call") => tool_calls.push(json!({
                "id": id_or_fresh(item.get("call_id").or_else(|| item.get("id")), "call"),
                "type": "custom",
                "custom": {
                    "name": item.get("name").and_then(Value::as_str).unwrap_or(""),
                    "input": item.get("input").and_then(Value::as_str).unwrap_or(""),
                },
            })),
            Some("reasoning") => {
                let texts = ["summary", "content"]
                    .into_iter()
                    .filter_map(|k| item.get(k).and_then(Value::as_array))
                    .flatten()
                    .filter_map(|p| non_empty_str(p, "text"));
                for t in texts {
                    if !reasoning.is_empty() {
                        reasoning.push_str("\n\n");
                    }
                    reasoning.push_str(t);
                }
            }
            // Messages; hosted-tool items (web search, file search, …) have no Chat form and only
            // follow tools a Chat client cannot declare.
            Some("message") | None => match item.get("content") {
                Some(Value::String(s)) => text.push_str(s),
                Some(Value::Array(parts)) => {
                    for p in parts {
                        match p.get("type").and_then(Value::as_str) {
                            Some("refusal") => refusal
                                .push_str(p.get("refusal").and_then(Value::as_str).unwrap_or("")),
                            _ => text.push_str(p.get("text").and_then(Value::as_str).unwrap_or("")),
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    let status_finish = responses_finish(v);
    let finish = if status_finish == "stop" && !tool_calls.is_empty() {
        "tool_calls"
    } else {
        status_finish
    };
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    let content = if text.is_empty() && (!tool_calls.is_empty() || !refusal.is_empty()) {
        Value::Null
    } else {
        json!(text)
    };
    msg.insert("content".into(), content);
    if !refusal.is_empty() {
        msg.insert("refusal".into(), json!(refusal));
    }
    if !tool_calls.is_empty() {
        msg.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !reasoning.is_empty() {
        msg.insert("reasoning_content".into(), json!(reasoning));
    }
    json!({
        "id": id_or_fresh(v.get("id"), "chatcmpl"),
        "object": "chat.completion",
        "created": created_of(v.get("created_at")),
        "model": v.get("model").cloned().unwrap_or(json!("")),
        "choices": [{
            "index": 0,
            "message": Value::Object(msg),
            "finish_reason": finish,
            "logprobs": Value::Null,
        }],
        "usage": Usage::from_responses(v.get("usage").unwrap_or(&Value::Null)).to_chat(),
    })
}

/// Responses `created_at` (seconds, sometimes fractional) as Chat's integer `created`.
pub(in crate::translate) fn created_of(v: Option<&Value>) -> u64 {
    v.and_then(|c| c.as_u64().or_else(|| c.as_f64().map(|f| f as u64)))
        .filter(|c| *c > 0)
        .unwrap_or_else(unix_now)
}
