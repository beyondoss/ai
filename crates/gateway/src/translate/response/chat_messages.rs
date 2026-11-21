//! Response JSON: Chat Completions ↔ Messages.

use crate::translate::*;

/// `gateway_cache`: the request's breakpoints were the gateway's, so its writes show as input.
pub(in crate::translate) fn anthropic_resp_to_openai(v: &Value, gateway_cache: bool) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut thinking: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for b in v
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("thinking") => {
                reasoning.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or(""));
                thinking.push(b.clone());
            }
            Some("redacted_thinking") => thinking.push(b.clone()),
            Some("tool_use") => tool_calls.push(json!({
                "id": id_or_fresh(b.get("id"), "call"),
                "type": "function",
                "function": {
                    "name": b.get("name").and_then(Value::as_str).unwrap_or(""),
                    "arguments": b.get("input").map_or_else(|| "{}".to_owned(), arguments_text),
                },
            })),
            // Server-tool blocks (`server_tool_use`, `web_search_tool_result`, …) have no Chat
            // Completions form; they only arise from tools a Chat client cannot declare.
            _ => {}
        }
    }
    let stop = v.get("stop_reason").and_then(Value::as_str);
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    let content = if text.is_empty() && (!tool_calls.is_empty() || stop == Some("refusal")) {
        Value::Null
    } else {
        json!(text)
    };
    msg.insert("content".into(), content);
    if stop == Some("refusal") {
        let why = v
            .pointer("/stop_details/explanation")
            .and_then(Value::as_str);
        msg.insert("refusal".into(), why.map_or(Value::Null, |w| json!(w)));
    }
    if !tool_calls.is_empty() {
        msg.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !reasoning.is_empty() {
        msg.insert("reasoning_content".into(), json!(reasoning));
    }
    if !thinking.is_empty() {
        msg.insert("thinking".into(), Value::Array(thinking));
    }
    json!({
        "id": id_or_fresh(v.get("id"), "chatcmpl"),
        "object": "chat.completion",
        "created": unix_now(),
        "model": v.get("model").cloned().unwrap_or(json!("")),
        "choices": [{
            "index": 0,
            "message": Value::Object(msg),
            "finish_reason": map_stop_to_openai(stop),
            "logprobs": Value::Null,
        }],
        "usage": Usage::from_anthropic(v.get("usage").unwrap_or(&Value::Null))
            .writes_as_input(gateway_cache)
            .to_chat(),
    })
}

pub(in crate::translate) fn openai_resp_to_anthropic(v: &Value) -> Value {
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first());
    let message = choice
        .and_then(|c| c.get("message"))
        .unwrap_or(&Value::Null);
    let mut content: Vec<Value> = chat_thinking_blocks(message)
        .into_iter()
        .filter(is_replayable_thinking)
        .collect();
    let mut refused = false;
    fn push_text(content: &mut Vec<Value>, t: &str) {
        if !t.is_empty() {
            content.push(json!({ "type": "text", "text": t }));
        }
    }
    match message.get("content") {
        Some(Value::String(s)) => push_text(&mut content, s),
        Some(Value::Array(parts)) => {
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("refusal") => {
                        refused = true;
                        push_text(
                            &mut content,
                            p.get("refusal").and_then(Value::as_str).unwrap_or(""),
                        );
                    }
                    _ => push_text(
                        &mut content,
                        p.get("text").and_then(Value::as_str).unwrap_or(""),
                    ),
                }
            }
        }
        _ => {}
    }
    if let Some(r) = non_empty_str(message, "refusal") {
        refused = true;
        push_text(&mut content, r);
    }
    let calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for c in calls {
        let func = c.get("function").unwrap_or(c);
        content.push(json!({
            "type": "tool_use",
            "id": id_or_fresh(c.get("id"), "toolu"),
            "name": func.get("name").and_then(Value::as_str).unwrap_or(""),
            "input": parse_arguments(func.get("arguments")),
        }));
    }
    json!({
        "id": id_or_fresh(v.get("id"), "msg"),
        "type": "message",
        "role": "assistant",
        "model": v.get("model").cloned().unwrap_or(json!("")),
        "content": content,
        "stop_reason": anthropic_stop_of(choice, refused, !calls.is_empty()),
        "stop_sequence": Value::Null,
        "usage": Usage::from_chat(v.get("usage").unwrap_or(&Value::Null)).to_anthropic(),
    })
}
