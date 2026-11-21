//! Request: Chat Completions → Responses.

use crate::translate::*;

/// The smallest `max_output_tokens` the Responses API accepts.
const RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;

/// The longest `user` OpenAI's Responses API accepts.
const RESPONSES_MAX_USER: usize = 64;

pub(in crate::translate) fn openai_req_to_responses(v: &Value, openai: OpenAiModel) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    copy_if(&mut out, v, "stream");
    // A Messages client's fast mode, carried by `anthropic_req_to_openai` for the provider to
    // refuse.
    if v.get("speed").and_then(Value::as_str) == Some("fast") {
        out.insert("speed".into(), json!("fast"));
    }
    // The Responses API rejects a limit under 16 ("integer_below_min_value"); Chat Completions
    // and Messages accept 1. Raising a tiny limit to the floor answers the request instead of
    // forwarding a guaranteed 400.
    if let Some(t) = max_tokens_of(v).or_else(|| v.get("max_output_tokens").and_then(Value::as_u64))
    {
        out.insert(
            "max_output_tokens".into(),
            json!(t.max(RESPONSES_MIN_OUTPUT_TOKENS)),
        );
    }
    // Chat Completions stores nothing unless asked; Responses stores every response by default.
    out.insert(
        "store".into(),
        v.get("store")
            .filter(|s| s.is_boolean())
            .cloned()
            .unwrap_or(json!(false)),
    );
    let effort = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str))
        .and_then(|e| openai.effort(e));
    if openai.keeps_sampling(effort.as_deref()) {
        copy_if(&mut out, v, "temperature");
        copy_if(&mut out, v, "top_p");
    }
    if let Some(effort) = effort {
        out.insert(
            "reasoning".into(),
            json!({ "effort": effort, "summary": "auto" }),
        );
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().filter_map(openai_tool_to_responses).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    if let Some(choice) = v.get("tool_choice") {
        out.insert(
            "tool_choice".into(),
            openai_tool_choice_to_responses(choice),
        );
    }
    for key in [
        "parallel_tool_calls",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "service_tier",
        "metadata",
        "top_logprobs",
    ] {
        copy_if(&mut out, v, key);
    }
    // OpenAI's Responses API caps `user` at 64 characters (400 "string too long"), though its Chat
    // Completions and xAI's Responses take longer ones; Claude Code's `metadata.user_id` is ~150.
    // A longer one becomes a hash of itself: the same user still maps to the same id.
    if openai.native
        && let Some(Value::String(user)) = out.get("user")
        && user.chars().count() > RESPONSES_MAX_USER
    {
        let hashed = format!("{:016x}", fnv1a64(user.as_bytes()));
        out.insert("user".into(), json!(hashed));
    }
    // No Responses field, and each changes what the client gets back: forwarded for the provider to
    // reject by name. (`mcp_servers` arrives here from a Messages client.)
    for key in ["stop", "audio", "web_search_options", "mcp_servers"] {
        copy_if(&mut out, v, key);
    }
    if v.get("n").and_then(Value::as_u64).is_some_and(|n| n > 1) {
        copy_if(&mut out, v, "n");
    }
    if v.get("logprobs").and_then(Value::as_bool) == Some(true) {
        copy_if(&mut out, v, "logprobs");
    }
    if v.get("modalities")
        .and_then(Value::as_array)
        .is_some_and(|m| m.iter().any(|x| x != "text"))
    {
        copy_if(&mut out, v, "modalities");
    }
    let mut text = Map::new();
    if let Some(rf) = v.get("response_format") {
        let format = match rf.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                let mut f = Map::new();
                f.insert("type".into(), json!("json_schema"));
                if let Some(js) = rf.get("json_schema") {
                    for key in ["name", "schema", "strict", "description"] {
                        copy_if(&mut f, js, key);
                    }
                }
                Some(Value::Object(f))
            }
            Some("json_object") => Some(json!({ "type": "json_object" })),
            _ => None,
        };
        if let Some(format) = format {
            text.insert("format".into(), format);
        }
    }
    copy_if(&mut text, v, "verbosity");
    if !text.is_empty() {
        out.insert("text".into(), Value::Object(text));
    }

    let mut instructions: Vec<String> = Vec::new();
    let mut input: Vec<Value> = Vec::new();
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        let mut in_conversation = false;
        // Ids of `custom` calls: their results are `custom_tool_call_output`, which OpenAI requires.
        let mut custom_calls: Vec<&Value> = Vec::new();
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            match role {
                "system" | "developer" if !in_conversation => {
                    instructions.extend(openai_system_texts(m).map(str::to_owned));
                }
                // Responses takes system and developer messages in `input`; a mid-conversation one
                // stays where the client put it.
                "system" | "developer" => input.push(json!({
                    "type": "message",
                    "role": role,
                    "content": chat_content_to_responses(m.get("content"), false),
                })),
                "tool" => {
                    in_conversation = true;
                    let custom = m
                        .get("tool_call_id")
                        .is_some_and(|id| custom_calls.contains(&id));
                    input.push(chat_tool_to_function_call_output(m, custom));
                }
                "assistant" => {
                    in_conversation = true;
                    // The turn's text comes before the calls it made, as Responses emits them.
                    let content = chat_content_to_responses(m.get("content"), true);
                    let mut content = content.as_array().cloned().unwrap_or_default();
                    if let Some(r) = m
                        .get("refusal")
                        .and_then(Value::as_str)
                        .filter(|r| !r.is_empty())
                        && content.is_empty()
                    {
                        content.push(json!({ "type": "refusal", "refusal": r }));
                    }
                    if !content.is_empty() {
                        input.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }));
                    }
                    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                        custom_calls.extend(
                            calls
                                .iter()
                                .filter(|c| c.get("type").and_then(Value::as_str) == Some("custom"))
                                .filter_map(|c| c.get("id")),
                        );
                        input.extend(calls.iter().map(chat_tool_call_to_responses_item));
                    }
                }
                "user" | "" => {
                    in_conversation = true;
                    input.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": chat_content_to_responses(m.get("content"), false),
                    }));
                }
                // A role no dialect has: forwarded whole, role unchanged, for the provider's 400
                // to name.
                _ => {
                    in_conversation = true;
                    input.push(m.clone());
                }
            }
        }
    }
    if !instructions.is_empty() {
        // `instructions` is a string; a list of text blocks is not one of its shapes.
        out.insert("instructions".into(), json!(instructions.join("\n\n")));
    }
    out.insert("input".into(), Value::Array(input));
    Value::Object(out)
}

/// A Chat Completions `tool` message → a Responses `function_call_output` (`custom_tool_call_output`
/// when it answers a `custom` call), whose `output` is a string or a list of input parts (text,
/// images, files).
fn chat_tool_to_function_call_output(m: &Value, custom: bool) -> Value {
    let output = match m.get("content") {
        Some(Value::String(s)) => json!(s),
        Some(content @ Value::Array(_)) => chat_content_to_responses(Some(content), false),
        _ => json!(""),
    };
    json!({
        "type": if custom { "custom_tool_call_output" } else { "function_call_output" },
        "call_id": m.get("tool_call_id").cloned().unwrap_or(json!("call_0")),
        "output": output,
    })
}

/// A Chat Completions tool call → a Responses `function_call` (or `custom_tool_call`) item.
fn chat_tool_call_to_responses_item(c: &Value) -> Value {
    let call_id = c.get("id").cloned().unwrap_or(json!("call_0"));
    if c.get("type").and_then(Value::as_str) == Some("custom") {
        let custom = c.get("custom").unwrap_or(c);
        return json!({
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": custom.get("name").cloned().unwrap_or(json!("")),
            "input": custom.get("input").cloned().unwrap_or(json!("")),
        });
    }
    let func = c.get("function").unwrap_or(c);
    json!({
        "type": "function_call",
        "call_id": call_id,
        "name": func.get("name").cloned().unwrap_or(json!("")),
        "arguments": func.get("arguments").cloned().unwrap_or(json!("{}")),
    })
}

/// A Chat Completions system or developer message's text parts (Responses `instructions` is one
/// string: Anthropic's `cache_control` has nowhere to go).
fn openai_system_texts(m: &Value) -> impl Iterator<Item = &str> {
    let (one, parts) = match m.get("content") {
        Some(Value::String(s)) => (Some(s.as_str()), &[][..]),
        Some(Value::Array(parts)) => (None, parts.as_slice()),
        _ => (None, &[][..]),
    };
    one.into_iter().chain(
        parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str())),
    )
}
