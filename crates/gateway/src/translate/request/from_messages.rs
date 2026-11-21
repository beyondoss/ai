//! Request: Messages → Chat Completions.

use crate::translate::*;

/// Anthropic `output_config.format` → OpenAI `response_format`.
fn anthropic_format_to_openai(v: &Value) -> Option<Value> {
    let format = v.pointer("/output_config/format")?;
    if format.get("type").and_then(Value::as_str) != Some("json_schema") {
        return None;
    }
    let schema = format.get("schema")?;
    Some(json!({
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": openai_strict_schema(schema),
            "schema": schema,
        },
    }))
}

pub(in crate::translate) fn anthropic_req_to_openai(v: &Value, up: Upstream) -> Value {
    let openai = up.openai;
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    if let Some(t) = v.get("max_tokens") {
        out.insert(openai.max_tokens_key().into(), t.clone());
    }
    copy_if(&mut out, v, "stream");
    // Remote MCP servers have no Chat Completions equivalent and change what the model can do.
    copy_if(&mut out, v, "mcp_servers");
    // Anthropic fast mode is served by direct Anthropic only, and a walk that read the body never
    // sends `speed: "fast"` anywhere else (`route::speed_ask`). One that did not read it may
    // still fail over here: the field goes on for the provider to refuse rather than be served at
    // standard speed under a fast request. `standard` is the default and means nothing here.
    if v.get("speed").and_then(Value::as_str) == Some("fast") {
        out.insert("speed".into(), json!("fast"));
    }
    if let Some(stop) = v.get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        // Chat Completions has no deferred loading: a `defer_loading` tool goes on once a
        // `tool_reference` in the conversation has loaded it, and not before — sending every
        // deferred schema would be the very cost deferring it avoided.
        let loaded = anthropic_tool_references(v);
        let mapped: Vec<Value> = tools
            .iter()
            .filter(|t| {
                t.get("defer_loading").and_then(Value::as_bool) != Some(true)
                    || t.get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|n| loaded.contains(n))
            })
            .filter_map(anthropic_tool_to_openai)
            .collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    // Chat Completions rejects `tool_choice` and `parallel_tool_calls` without `tools` ("only
    // allowed when 'tools' are specified"); Messages accepts them, and they mean nothing there.
    let has_tools = out.contains_key("tools");
    if let Some(choice) = v.get("tool_choice").filter(|_| has_tools) {
        out.insert(
            "tool_choice".into(),
            anthropic_tool_choice_to_openai(choice),
        );
    }
    // Anthropic's effort lives in `output_config.effort`; `thinking` alone implies one. Fit to what
    // the upstream model accepts (see `OpenAiModel`).
    let effort = v
        .pointer("/output_config/effort")
        .and_then(Value::as_str)
        .map(effort_alias)
        .or_else(|| v.get("thinking").and_then(effort_from_thinking))
        .and_then(|e| openai.effort(e));
    if openai.keeps_sampling(effort.as_deref()) {
        copy_if(&mut out, v, "temperature");
        copy_if(&mut out, v, "top_p");
    }
    if let Some(effort) = effort {
        out.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(format) = anthropic_format_to_openai(v) {
        out.insert("response_format".into(), format);
    }
    if has_tools
        && v.pointer("/tool_choice/disable_parallel_tool_use")
            .and_then(Value::as_bool)
            == Some(true)
    {
        out.insert("parallel_tool_calls".into(), json!(false));
    }
    if let Some(user) = v.pointer("/metadata/user_id").filter(|u| u.is_string()) {
        out.insert("user".into(), user.clone());
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = v.get("system")
        && let Some(msg) = anthropic_system_to_openai(sys)
    {
        messages.push(msg);
    }
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            match role {
                "assistant" => messages.push(anthropic_assistant_to_openai(m, up.claude_thinking)),
                // A mid-conversation system message keeps its role and its place. Chat Completions
                // takes `system` anywhere. A directive-only one (`content: []` plus
                // `output_config`) says nothing a Chat model can read, so it has no message.
                "system" => messages.extend(m.get("content").and_then(anthropic_system_to_openai)),
                "user" => messages.extend(anthropic_user_to_openai(m)),
                // A role no dialect has: forwarded whole, role unchanged, for the provider's 400
                // to name.
                _ => messages.push(m.clone()),
            }
        }
    }
    out.insert("messages".into(), Value::Array(messages));
    Value::Object(out)
}

fn anthropic_system_to_openai(sys: &Value) -> Option<Value> {
    match sys {
        Value::String(s) if !s.is_empty() => Some(json!({ "role": "system", "content": s })),
        Value::Array(blocks) => {
            let has_cc = blocks.iter().any(|b| b.get("cache_control").is_some());
            if has_cc {
                let parts: Vec<Value> = blocks
                    .iter()
                    .filter_map(|b| {
                        let t = b
                            .get("text")
                            .and_then(Value::as_str)
                            .or_else(|| b.as_str())?;
                        let mut p = json!({ "type": "text", "text": t });
                        if let Some(obj) = p.as_object_mut() {
                            copy_cache_control(obj, b);
                        }
                        Some(p)
                    })
                    .collect();
                if parts.is_empty() {
                    return None;
                }
                Some(json!({ "role": "system", "content": parts }))
            } else {
                let t = anthropic_system_text(blocks);
                (!t.is_empty()).then(|| json!({ "role": "system", "content": t }))
            }
        }
        _ => None,
    }
}

/// A system prompt's text blocks, one per line.
fn anthropic_system_text(blocks: &[Value]) -> String {
    blocks
        .iter()
        .filter_map(|b| b.get("text").and_then(Value::as_str).or_else(|| b.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// An Anthropic assistant turn → Chat Completions. Its thinking rides `reasoning_content` (the text),
/// our `thinking` array (the blocks, which a Chat client sends back), and, for a Claude model
/// behind a Chat Completions host, `reasoning_details` (see [`reasoning_details`]).
fn anthropic_assistant_to_openai(m: &Value, claude_thinking: bool) -> Value {
    let mut text = String::new();
    let mut thinking_text = String::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    match m.get("content") {
        Some(Value::String(s)) => text = s.clone(),
        Some(Value::Array(blocks)) => {
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("thinking") => {
                        if let Some(t) = b
                            .get("thinking")
                            .and_then(Value::as_str)
                            .or_else(|| b.get("text").and_then(Value::as_str))
                        {
                            thinking_text.push_str(t);
                        }
                        thinking_blocks.push(b.clone());
                    }
                    Some("redacted_thinking") => {
                        thinking_blocks.push(b.clone());
                    }
                    Some("tool_use") => {
                        let id = b.get("id").and_then(Value::as_str).unwrap_or("call_0");
                        let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = b
                            .get("input")
                            .map(arguments_text)
                            .unwrap_or_else(|| "{}".into());
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": args },
                        }));
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    if tool_calls.is_empty() {
        msg.insert("content".into(), json!(text));
    } else {
        msg.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
        msg.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    if !thinking_text.is_empty() {
        msg.insert("reasoning_content".into(), json!(thinking_text));
    }
    attach_thinking(&mut msg, thinking_blocks, claude_thinking);
    Value::Object(msg)
}

fn anthropic_user_to_openai(m: &Value) -> Vec<Value> {
    match m.get("content") {
        Some(Value::String(s)) => vec![json!({ "role": "user", "content": s })],
        Some(Value::Array(blocks)) => {
            let mut out = Vec::new();
            let mut user_parts: Vec<Value> = Vec::new();
            let flush_user = |parts: &mut Vec<Value>, out: &mut Vec<Value>| {
                if parts.is_empty() {
                    return;
                }
                let p = std::mem::take(parts);
                if p.len() == 1 && p[0].get("type").and_then(Value::as_str) == Some("text") {
                    out.push(json!({
                        "role": "user",
                        "content": p[0].get("text").cloned().unwrap_or(json!("")),
                    }));
                } else {
                    out.push(json!({ "role": "user", "content": Value::Array(p) }));
                }
            };
            // Chat Completions wants every `tool` message right after the assistant turn that
            // called it, so the tool results go first and the rest of the turn follows as one
            // user message (Anthropic puts `tool_result` blocks first anyway).
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("tool_result") => {
                        out.push(anthropic_tool_result_to_openai(b, &mut user_parts));
                    }
                    Some("image") => user_parts.extend(anthropic_image_part(b)),
                    Some("text") | None => {
                        if let Some(t) =
                            b.get("text").and_then(Value::as_str).or_else(|| b.as_str())
                        {
                            user_parts.push(text_block(t, b.get("cache_control")));
                        }
                    }
                    Some("document") => user_parts.push(anthropic_document_to_openai(b)),
                    // Anything else (`search_result`, `container_upload`, …) has no Chat
                    // Completions part: forwarded for the provider to reject by name.
                    Some(_) => user_parts.push(b.clone()),
                }
            }
            flush_user(&mut user_parts, &mut out);
            if out.is_empty() {
                out.push(json!({ "role": "user", "content": "" }));
            }
            out
        }
        _ => vec![json!({ "role": "user", "content": "" })],
    }
}

/// Anthropic `tool_result` → a Chat Completions `tool` message holding its text. A tool message
/// holds text only, so images and documents in the result (a screenshot, a rendered PDF) move to
/// `carried`, which the caller sends as the user message right after the tool messages; dropping
/// them would get an answer about output the model never saw. `is_error` has no field: the text
/// says so instead.
fn anthropic_tool_result_to_openai(b: &Value, carried: &mut Vec<Value>) -> Value {
    let id = b
        .get("tool_use_id")
        .and_then(Value::as_str)
        .unwrap_or("call_0");
    let mut text = String::new();
    match b.get("content") {
        Some(Value::String(s)) => text.push_str(s),
        Some(Value::Array(inner)) => {
            for x in inner {
                match x.get("type").and_then(Value::as_str) {
                    Some("text") | None => {
                        if let Some(t) = x.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("image") => carried.extend(anthropic_image_part(x)),
                    Some("document") => carried.push(anthropic_document_to_openai(x)),
                    // A tool search result: the tool it loaded joins `tools` (see
                    // `anthropic_tool_references`), and the model reads which one here.
                    Some("tool_reference") => {
                        if let Some(name) = x.get("tool_name").and_then(Value::as_str) {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str("Loaded tool: ");
                            text.push_str(name);
                        }
                    }
                    Some(_) => carried.push(x.clone()),
                }
            }
        }
        _ => {}
    }
    if b.get("is_error").and_then(Value::as_bool) == Some(true) {
        text = if text.is_empty() {
            "Error".to_owned()
        } else {
            format!("Error: {text}")
        };
    }
    json!({ "role": "tool", "tool_call_id": id, "content": text })
}
