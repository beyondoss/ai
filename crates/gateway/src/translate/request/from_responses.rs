//! Request: Responses → Chat Completions.

use crate::translate::*;

/// Drop Responses-only session/control fields and reshape `input` onto Chat Completions
/// `messages`: [`request`] for that one pairing, which the unit tests pin. Same-endpoint Responses
/// must not reshape — those fields pass through as a byte relay.
#[cfg(test)]
pub(in crate::translate) fn responses_to_chat(body: &[u8]) -> Vec<u8> {
    request(Endpoint::Responses, Endpoint::ChatCompletions, body, "")
}

/// `wrap_custom`: the Chat Completions body is an intermediate bound for Messages, which has no
/// free-form tool, so each `custom` tool becomes a function taking one `input` string (see
/// [`ToolNames`]).
pub(in crate::translate) fn responses_req_to_openai(
    v: &Value,
    up: Upstream,
    wrap_custom: bool,
) -> Value {
    let openai = up.openai;
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    copy_if(&mut out, v, "stream");
    if let Some(t) = v
        .get("max_output_tokens")
        .or_else(|| v.get("max_tokens"))
        .or_else(|| v.get("max_completion_tokens"))
    {
        out.insert(openai.max_tokens_key().into(), t.clone());
    }
    let effort = openai.fit_effort_and_sampling(&mut out, v);
    if let Some(effort) = effort {
        out.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let search_chosen = v.get("tool_choice").is_some_and(names_hosted_web_search);
        let mut mapped: Vec<Value> = Vec::with_capacity(tools.len());
        for t in tools {
            // OpenAI's hosted search runs nowhere off its Responses API (see
            // `is_hosted_web_search`).
            if !search_chosen && is_hosted_web_search(t) {
                continue;
            }
            if t.get("type").and_then(Value::as_str) == Some("namespace") {
                let about = non_empty_str(t, "description");
                for member in namespace_members(t) {
                    let mut m = member.tool.clone();
                    if let Some(obj) = m.as_object_mut() {
                        obj.insert("name".into(), json!(member.flat));
                        if let Some(about) = about {
                            // The namespace's description is shown to the model once on
                            // Responses; flat, each member carries it.
                            let desc = match obj
                                .get("description")
                                .and_then(Value::as_str)
                                .filter(|d| !d.is_empty())
                            {
                                Some(d) => format!("{about}\n\n{d}"),
                                None => about.to_owned(),
                            };
                            obj.insert("description".into(), json!(desc));
                        }
                    }
                    mapped.push(responses_tool_to_openai(&m, wrap_custom));
                }
            } else {
                mapped.push(responses_tool_to_openai(t, wrap_custom));
            }
        }
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    // Responses takes `tool_choice` and `parallel_tool_calls` without tools; Chat Completions
    // rejects both ("only allowed when 'tools' are specified").
    if out.contains_key("tools") {
        if let Some(choice) = v.get("tool_choice") {
            out.insert(
                "tool_choice".into(),
                responses_tool_choice_to_openai(choice),
            );
        }
        copy_if(&mut out, v, "parallel_tool_calls");
    }
    for key in ["user", "safety_identifier"] {
        copy_if(&mut out, v, key);
    }
    // OpenAI's Chat Completions takes these too. Elsewhere they are hints (cache routing, latency
    // tier, answer length) and go.
    if openai.native {
        for key in ["prompt_cache_key", "service_tier"] {
            copy_if(&mut out, v, key);
        }
        if let Some(verbosity) = v.pointer("/text/verbosity") {
            out.insert("verbosity".into(), verbosity.clone());
        }
    }
    // A stored prompt template decides what the model is asked, and `top_logprobs` what the
    // client gets back. Neither maps (Chat Completions' `top_logprobs` needs `logprobs`, whose
    // answer a translated client is never shown): forwarded for the provider to reject by name,
    // and onward by `openai_req_to_anthropic` when this body is bound for Messages.
    for key in ["prompt", "top_logprobs"] {
        copy_if(&mut out, v, key);
    }
    // Responses nests the output format under `text.format` and flattens `json_schema`.
    if let Some(format) = v.pointer("/text/format") {
        let rf = match format.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                let mut js = Map::new();
                for key in ["name", "schema", "strict", "description"] {
                    copy_if(&mut js, format, key);
                }
                Some(json!({ "type": "json_schema", "json_schema": js }))
            }
            Some("json_object") => Some(json!({ "type": "json_object" })),
            _ => None,
        };
        if let Some(rf) = rf {
            out.insert("response_format".into(), rf);
        }
    }

    let mut messages: Vec<Value> = Vec::new();
    // `instructions: null` (openai-python's `instructions=None`) is absent, as `copy_if` treats
    // every other null: a system message with null content is a 422 on xAI, a 400 on Together.
    if let Some(instr) = v.get("instructions").filter(|i| !i.is_null()) {
        messages.push(responses_instructions_to_system(instr));
    }
    messages.extend(responses_input_to_messages(v.get("input"), up));
    out.insert("messages".into(), Value::Array(messages));
    Value::Object(out)
}

fn responses_instructions_to_system(instr: &Value) -> Value {
    match instr {
        Value::String(s) => json!({ "role": "system", "content": s }),
        Value::Array(parts) => json!({
            "role": "system",
            "content": parts.iter().filter_map(responses_part_to_openai).collect::<Vec<_>>(),
        }),
        other => json!({ "role": "system", "content": other }),
    }
}

/// Responses `input` → Chat Completions messages. A Claude upstream is sent the thinking the
/// gateway minted as `reasoning` items (see [`gateway_reasoning_block`]). A `developer` message
/// stays one only on OpenAI's own API; other Chat Completions hosts know `system`, not `developer`.
pub(super) fn responses_input_to_messages(input: Option<&Value>, up: Upstream) -> Vec<Value> {
    let claude_thinking = up.claude_thinking;
    let items = match input {
        Some(Value::String(s)) => return vec![json!({ "role": "user", "content": s })],
        Some(Value::Array(items)) => items,
        _ => return Vec::new(),
    };
    let mut out: Vec<Value> = Vec::new();
    // Images and files a tool returned. A Chat Completions tool message holds text, so they follow
    // the run of tool messages as one user message.
    let mut carried: Vec<Value> = Vec::new();
    // Signed thinking from `reasoning` items, waiting for the assistant turn they open.
    let mut thinking: Vec<Value> = Vec::new();
    for item in items {
        let typ = item.get("type").and_then(Value::as_str).unwrap_or("");
        match typ {
            "function_call" | "custom_tool_call" => {
                flush_carried(&mut out, &mut carried);
                let call = responses_call_to_openai(item);
                // Parallel calls arrive as consecutive items; Chat Completions wants them on one
                // assistant message (split across several, OpenAI rejects the history), after
                // whatever text that turn said.
                match out.last_mut().and_then(Value::as_object_mut) {
                    Some(last) if last.get("role").and_then(Value::as_str) == Some("assistant") => {
                        match last.get_mut("tool_calls").and_then(Value::as_array_mut) {
                            Some(calls) => calls.push(call),
                            None => {
                                last.insert("tool_calls".into(), json!([call]));
                            }
                        }
                    }
                    _ => out.push(json!({
                        "role": "assistant",
                        "content": Value::Null,
                        "tool_calls": [call],
                    })),
                }
                if let Some(last) = out.last_mut().and_then(Value::as_object_mut) {
                    attach_thinking(last, std::mem::take(&mut thinking), claude_thinking);
                }
            }
            "function_call_output" | "custom_tool_call_output" => {
                thinking.clear();
                out.push(responses_output_to_tool_message(item, &mut carried));
            }
            // The model's own earlier reasoning. Only a Claude upstream can use any of it, and only
            // what the gateway minted from Claude's signed thinking; OpenAI's encrypted reasoning
            // means nothing off OpenAI's Responses API.
            "reasoning" => {
                if claude_thinking && let Some(block) = gateway_reasoning_block(item) {
                    thinking.push(block);
                }
            }
            _ if typ == "message" || item.get("role").is_some() => {
                flush_carried(&mut out, &mut carried);
                let role = match item.get("role").and_then(Value::as_str).unwrap_or("user") {
                    "developer" if !up.openai.native => "system",
                    role => role,
                };
                let mut msg = Map::new();
                msg.insert("role".into(), json!(role));
                msg.insert(
                    "content".into(),
                    responses_content_to_openai(item.get("content")),
                );
                if role == "assistant" {
                    attach_thinking(&mut msg, std::mem::take(&mut thinking), claude_thinking);
                } else {
                    thinking.clear();
                }
                out.push(Value::Object(msg));
            }
            // Records of a tool the provider ran itself: no Chat Completions equivalent, and
            // nothing the client wrote (the answer that used them follows as a message).
            "web_search_call"
            | "file_search_call"
            | "code_interpreter_call"
            | "image_generation_call"
            | "mcp_call"
            | "mcp_list_tools" => {}
            // OpenAI-held state no other upstream can resolve: a `compaction` is a summary only
            // OpenAI can decrypt, an `item_reference` a pointer into its store. A catalog walk
            // brings only a tool step's `item_reference` here (its preamble or reasoning; the
            // call and output follow in full): one for an earlier turn is session state that
            // `responses_session_field` names, so the row walks its Responses arm or 400s.
            // A `compaction` is dropped, so a compacted session on a translated candidate runs on
            // the history the client holds. The same-wire relay keeps both.
            "compaction" | "item_reference" => {}
            // Anything else (`computer_call_output`, `local_shell_call`, …) is history the client
            // holds and the model would answer without. Forwarded as-is in place, so the provider
            // rejects it by name rather than answering a different conversation.
            _ => {
                flush_carried(&mut out, &mut carried);
                out.push(item.clone());
            }
        }
    }
    flush_carried(&mut out, &mut carried);
    out
}

/// The Anthropic thinking block behind a `reasoning` item this gateway minted for a Responses
/// client (see [`reasoning_item`], [`is_gateway_reasoning`]): its summary is the block's text, and
/// its `encrypted_content` is the block's Anthropic signature. Sent back to a Claude upstream it
/// keeps a thinking + tool loop valid (older Claude models 400 a tool turn without its thinking).
/// Any other reasoning item — OpenAI's own, or one without a signature — is `None`.
fn gateway_reasoning_block(item: &Value) -> Option<Value> {
    if !is_gateway_reasoning(item) {
        return None;
    }
    let content = non_empty_str(item, "encrypted_content")?;
    let signature = content
        .strip_prefix(GATEWAY_SIGNATURE_PREFIX)
        .unwrap_or(content);
    if signature.is_empty() {
        return None;
    }
    let text: String = item
        .get("summary")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    Some(json!({ "type": "thinking", "thinking": text, "signature": signature }))
}

fn flush_carried(out: &mut Vec<Value>, carried: &mut Vec<Value>) {
    if !carried.is_empty() {
        out.push(json!({ "role": "user", "content": std::mem::take(carried) }));
    }
}

/// A Responses `function_call` / `custom_tool_call` item → a Chat Completions tool call.
fn responses_call_to_openai(item: &Value) -> Value {
    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .cloned()
        .unwrap_or(json!("call_0"));
    // A call into a `namespace` tool replays under the flat name the tool was offered as.
    let name = match (
        item.get("name").and_then(Value::as_str),
        non_empty_str(item, "namespace"),
    ) {
        (Some(name), Some(ns)) => json!(namespaced_name(ns, name)),
        _ => item.get("name").cloned().unwrap_or(json!("")),
    };
    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
        return json!({
            "id": id,
            "type": "custom",
            "custom": { "name": name, "input": item.get("input").cloned().unwrap_or(json!("")) },
        });
    }
    json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": match item.get("arguments") {
                Some(Value::String(s)) => Value::String(s.clone()),
                Some(other) => json!(value_string(other)),
                None => json!("{}"),
            },
        }
    })
}

/// A Responses tool output → a Chat Completions `tool` message. An output that is a list of parts
/// keeps its text here; its images and files go to `carried` (see `responses_input_to_messages`).
fn responses_output_to_tool_message(item: &Value, carried: &mut Vec<Value>) -> Value {
    let mut m = Map::new();
    m.insert("role".into(), json!("tool"));
    if let Some(id) = item.get("call_id").or_else(|| item.get("id")) {
        m.insert("tool_call_id".into(), id.clone());
    }
    let content = match item.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text" | "text") | None => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    _ => carried.extend(responses_part_to_openai(p)),
                }
            }
            text
        }
        Some(other) => arguments_text(other),
        None => String::new(),
    };
    m.insert("content".into(), json!(content));
    Value::Object(m)
}
