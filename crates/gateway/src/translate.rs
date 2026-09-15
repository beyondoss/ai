//! Chat Completions ↔ Messages ↔ Responses translation for a managed catalog walk.
//!
//! Triggered when the inbound path names Chat Completions, Messages, or Responses (including the
//! `/auto` suffix) and the catalog row speaks a different one of those three. Same-wire walks stay
//! a byte relay — including `/{provider}/v1/responses`. `/{provider}/…` never translates.
//!
//! Responses ↔ Messages is composed through Chat Completions so thinking / `cache_control` /
//! `reasoning_effort` keep the slice-1 mappings.
//!
//! v1 is allowed to be lossy on extras a stock SDK does not need for a tool loop:
//! - **Dropped:** Responses-only fields (`store`, `previous_response_id`, `include`, `truncation`,
//!   `text` format, …) when leaving Responses; `stream_options` on a Responses or Anthropic body;
//!   image `http(s)` URLs (Anthropic wants base64). Base64 data-URI images are converted both ways.
//! - **Passed both ways:** `thinking` / `redacted_thinking` blocks, `cache_control` on tools and
//!   content, `reasoning_effort` ↔ Anthropic `thinking`. These are what an agent workload sends.
//! - **Required mapping:** system/messages/`input`, `max_tokens`/`max_output_tokens`, temperature,
//!   stop, stream, tools, `tool_choice`, text + tool_use/tool_result, usage. Anthropic requires
//!   `max_tokens`; a missing OpenAI value becomes 4096.
//!
//! Usage/billing parse the **upstream** body. This module only reshapes bytes the client sees.

use crate::route::Endpoint;
use serde_json::{Map, Value, json};

/// Anthropic requires this; OpenAI does not. Used when the Chat Completions body omitted it.
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Per-request translate state, boxed on [`crate::proxy`]'s model-routed path only.
pub struct TranslateState {
    /// Inbound endpoint — what the client sent and what it must receive.
    pub client: Endpoint,
    /// SSE translator, created in `response_filter` once the upstream is known to stream.
    pub sse: Option<SseBridge>,
    /// Non-stream JSON, withheld until end-of-stream so we can map the object.
    pub json_buf: Vec<u8>,
}

impl TranslateState {
    pub fn new(client: Endpoint) -> Self {
        Self {
            client,
            sse: None,
            json_buf: Vec::new(),
        }
    }
}

/// Map a buffered request body from `from` (inbound) to `to` (upstream endpoint).
///
/// Unparseable JSON is returned unchanged so the provider 400s rather than us 502ing after
/// headers have already gone upstream. The candidate `model` id is spliced by the caller
/// **after** this returns.
pub fn request(from: Endpoint, to: Endpoint, body: &[u8]) -> Vec<u8> {
    if from == to {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if !v.is_object() {
        return body.to_vec();
    }
    encode(&map_request(from, to, &v))
}

fn map_request(from: Endpoint, to: Endpoint, v: &Value) -> Value {
    match (from, to) {
        (Endpoint::ChatCompletions, Endpoint::Messages) => openai_req_to_anthropic(v),
        (Endpoint::Messages, Endpoint::ChatCompletions) => anthropic_req_to_openai(v),
        (Endpoint::Responses, Endpoint::ChatCompletions) => responses_req_to_openai(v),
        (Endpoint::ChatCompletions, Endpoint::Responses) => openai_req_to_responses(v),
        (Endpoint::Responses, Endpoint::Messages) => {
            openai_req_to_anthropic(&responses_req_to_openai(v))
        }
        (Endpoint::Messages, Endpoint::Responses) => {
            openai_req_to_responses(&anthropic_req_to_openai(v))
        }
        (a, b) if a == b => v.clone(),
        _ => v.clone(),
    }
}

/// Map a non-stream JSON response from `upstream` into `client`. Error objects are reshaped
/// into the client's error envelope.
pub fn response_json(upstream: Endpoint, client: Endpoint, body: &[u8]) -> Vec<u8> {
    if upstream == client {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if looks_like_error(&v) {
        return encode(&map_error(&v, client));
    }
    encode(&map_response(upstream, client, &v))
}

fn map_response(upstream: Endpoint, client: Endpoint, v: &Value) -> Value {
    match (upstream, client) {
        (Endpoint::Messages, Endpoint::ChatCompletions) => anthropic_resp_to_openai(v),
        (Endpoint::ChatCompletions, Endpoint::Messages) => openai_resp_to_anthropic(v),
        (Endpoint::ChatCompletions, Endpoint::Responses) => openai_resp_to_responses(v),
        (Endpoint::Responses, Endpoint::ChatCompletions) => responses_resp_to_openai(v),
        (Endpoint::Messages, Endpoint::Responses) => {
            openai_resp_to_responses(&anthropic_resp_to_openai(v))
        }
        (Endpoint::Responses, Endpoint::Messages) => {
            openai_resp_to_anthropic(&responses_resp_to_openai(v))
        }
        (a, b) if a == b => v.clone(),
        _ => v.clone(),
    }
}

fn encode(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_else(|_| {
        br#"{"error":{"message":"translate failed","type":"api_error"}}"#.to_vec()
    })
}

fn looks_like_error(v: &Value) -> bool {
    if v.get("error").is_none() {
        return false;
    }
    // Success bodies never carry a top-level `error` next to `choices` / `content` / `output`.
    if v.get("choices").is_some() || v.get("content").is_some() || v.get("output").is_some() {
        return false;
    }
    if v.get("type").and_then(Value::as_str) == Some("message") {
        return false;
    }
    if v.get("object").and_then(Value::as_str) == Some("response") {
        return false;
    }
    true
}

fn map_error(v: &Value, client: Endpoint) -> Value {
    let (typ, msg) = extract_error(v);
    match client {
        Endpoint::Messages => json!({ "type": "error", "error": { "type": typ, "message": msg } }),
        Endpoint::ChatCompletions | Endpoint::Responses => {
            json!({ "error": { "message": msg, "type": typ } })
        }
    }
}

fn extract_error(v: &Value) -> (String, String) {
    let err = v.get("error").unwrap_or(v);
    let typ = err
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| v.get("type").and_then(Value::as_str))
        .unwrap_or("api_error");
    let msg = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("upstream error");
    (typ.to_owned(), msg.to_owned())
}

// --- request: OpenAI → Anthropic --------------------------------------------

fn openai_req_to_anthropic(v: &Value) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    if let Some(t) = max_tokens_of(v) {
        out.insert("max_tokens".into(), json!(t));
    } else {
        out.insert("max_tokens".into(), json!(DEFAULT_MAX_TOKENS));
    }
    copy_if(&mut out, v, "temperature");
    copy_if(&mut out, v, "top_p");
    copy_if(&mut out, v, "stream");
    if let Some(stop) = v.get("stop") {
        match stop {
            Value::String(s) => {
                out.insert("stop_sequences".into(), json!([s]));
            }
            Value::Array(_) => {
                out.insert("stop_sequences".into(), stop.clone());
            }
            _ => {}
        }
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().filter_map(openai_tool_to_anthropic).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    if let Some(choice) = v.get("tool_choice") {
        out.insert(
            "tool_choice".into(),
            openai_tool_choice_to_anthropic(choice),
        );
    }
    openai_reasoning_to_anthropic(v, &mut out);

    let mut system_parts: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        let mut pending_tool_results: Vec<Value> = Vec::new();
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            match role {
                "system" | "developer" => {
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    system_parts.extend(openai_system_blocks(m));
                }
                "tool" => {
                    if let Some(tr) = openai_tool_result(m) {
                        pending_tool_results.push(tr);
                    }
                }
                "assistant" => {
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    push_anth_message(&mut messages, "assistant", openai_assistant_content(m));
                }
                _ => {
                    // user (and anything else treated as user)
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    push_anth_message(&mut messages, "user", openai_user_content(m));
                }
            }
        }
        flush_tool_results(&mut messages, &mut pending_tool_results);
    }
    if !system_parts.is_empty() {
        out.insert("system".into(), anthropic_system_value(system_parts));
    }
    out.insert("messages".into(), Value::Array(messages));
    Value::Object(out)
}

fn flush_tool_results(messages: &mut Vec<Value>, pending: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    let blocks = std::mem::take(pending);
    push_anth_message(messages, "user", Value::Array(blocks));
}

fn push_anth_message(messages: &mut Vec<Value>, role: &str, content: Value) {
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
    {
        merge_content(last, content);
        return;
    }
    messages.push(json!({ "role": role, "content": content }));
}

fn merge_content(msg: &mut Value, add: Value) {
    let existing = msg.get_mut("content");
    let Some(existing) = existing else {
        msg["content"] = add;
        return;
    };
    match (&*existing, &add) {
        (Value::Array(a), Value::Array(b)) => {
            let mut n = a.clone();
            n.extend(b.iter().cloned());
            *existing = Value::Array(n);
        }
        (Value::String(a), Value::String(b)) => {
            *existing = Value::String(format!("{a}{b}"));
        }
        (Value::String(a), Value::Array(b)) => {
            let mut n = vec![json!({ "type": "text", "text": a })];
            n.extend(b.iter().cloned());
            *existing = Value::Array(n);
        }
        (Value::Array(a), Value::String(b)) => {
            let mut n = a.clone();
            n.push(json!({ "type": "text", "text": b }));
            *existing = Value::Array(n);
        }
        _ => *existing = add,
    }
}

fn openai_tool_to_anthropic(t: &Value) -> Option<Value> {
    let func = t.get("function").unwrap_or(t);
    let name = func.get("name").and_then(Value::as_str)?;
    let mut m = Map::new();
    m.insert("name".into(), json!(name));
    if let Some(d) = func.get("description") {
        m.insert("description".into(), d.clone());
    }
    let schema = func
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    m.insert("input_schema".into(), schema);
    copy_cache_control(&mut m, t);
    if !m.contains_key("cache_control") {
        copy_cache_control(&mut m, func);
    }
    Some(Value::Object(m))
}

fn openai_reasoning_to_anthropic(v: &Value, out: &mut Map<String, Value>) {
    // An already-Anthropic `thinking` object wins over `reasoning_effort` so a round-trip that
    // kept the native shape is not re-bucketed.
    if let Some(t) = v.get("thinking").filter(|t| t.is_object()) {
        out.insert("thinking".into(), t.clone());
        return;
    }
    let effort = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str));
    if let Some(effort) = effort {
        out.insert("thinking".into(), thinking_from_effort(effort));
    }
}

fn thinking_from_effort(effort: &str) -> Value {
    match effort {
        "none" | "off" | "disabled" => json!({ "type": "disabled" }),
        _ => json!({
            "type": "enabled",
            "budget_tokens": budget_for_effort(effort),
        }),
    }
}

fn budget_for_effort(effort: &str) -> u64 {
    match effort {
        "minimal" | "low" => 1024,
        "high" => 8192,
        "xhigh" | "max" => 16384,
        _ => 4096,
    }
}

fn effort_from_thinking(thinking: &Value) -> Option<&'static str> {
    match thinking.get("type").and_then(Value::as_str) {
        Some("disabled") => Some("none"),
        Some("adaptive") => Some(
            thinking
                .get("effort")
                .and_then(Value::as_str)
                .map(effort_alias)
                .unwrap_or("high"),
        ),
        Some("enabled") => {
            let budget = thinking
                .get("budget_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(4096);
            Some(effort_from_budget(budget))
        }
        _ => None,
    }
}

fn effort_alias(s: &str) -> &'static str {
    match s {
        "none" | "off" | "disabled" => "none",
        "minimal" | "low" => "low",
        "high" => "high",
        "xhigh" | "max" => "xhigh",
        _ => "medium",
    }
}

fn effort_from_budget(budget: u64) -> &'static str {
    if budget <= 1024 {
        "low"
    } else if budget <= 4096 {
        "medium"
    } else if budget <= 8192 {
        "high"
    } else {
        "xhigh"
    }
}

fn copy_cache_control(out: &mut Map<String, Value>, src: &Value) {
    if let Some(cc) = src.get("cache_control") {
        out.insert("cache_control".into(), cc.clone());
    }
}

fn openai_system_blocks(m: &Value) -> Vec<Value> {
    let cc = m.get("cache_control");
    match m.get("content") {
        Some(Value::String(s)) => vec![text_block(s, cc)],
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                let t = p
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| p.as_str())?;
                let part_cc = p.get("cache_control").or(cc);
                Some(text_block(t, part_cc))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn text_block(text: &str, cache_control: Option<&Value>) -> Value {
    let mut m = json!({ "type": "text", "text": text });
    if let Some(cc) = cache_control
        && let Some(obj) = m.as_object_mut()
    {
        obj.insert("cache_control".into(), cc.clone());
    }
    m
}

fn anthropic_system_value(blocks: Vec<Value>) -> Value {
    if blocks.len() == 1
        && blocks[0].get("cache_control").is_none()
        && let Some(t) = blocks[0].get("text").cloned()
    {
        return t;
    }
    Value::Array(blocks)
}

fn openai_tool_choice_to_anthropic(choice: &Value) -> Value {
    match choice {
        Value::String(s) => match s.as_str() {
            "none" => json!({ "type": "none" }),
            "required" => json!({ "type": "any" }),
            _ => json!({ "type": "auto" }),
        },
        Value::Object(o) => {
            if let Some(name) = o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .or_else(|| o.get("name").and_then(Value::as_str))
            {
                json!({ "type": "tool", "name": name })
            } else {
                json!({ "type": "auto" })
            }
        }
        _ => json!({ "type": "auto" }),
    }
}

fn openai_tool_result(m: &Value) -> Option<Value> {
    let id = m
        .get("tool_call_id")
        .and_then(Value::as_str)
        .unwrap_or("call_0");
    let content = message_text(m).unwrap_or_default();
    Some(json!({
        "type": "tool_result",
        "tool_use_id": id,
        "content": content,
    }))
}

fn openai_assistant_content(m: &Value) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    match m.get("content") {
        Some(Value::String(s)) if !s.is_empty() => {
            blocks.push(json!({ "type": "text", "text": s }));
        }
        Some(Value::Array(parts)) => {
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("thinking") => blocks.push(thinking_block_from_openai(p)),
                    Some("redacted_thinking") => blocks.push(redacted_block_from_openai(p)),
                    Some("text") | None => {
                        if let Some(t) =
                            p.get("text").and_then(Value::as_str).or_else(|| p.as_str())
                            && !t.is_empty()
                        {
                            let mut b = json!({ "type": "text", "text": t });
                            if let Some(obj) = b.as_object_mut() {
                                copy_cache_control(obj, p);
                            }
                            blocks.push(b);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    if let Some(extra) = m.get("thinking").and_then(Value::as_array) {
        for p in extra.iter().rev() {
            let block = match p.get("type").and_then(Value::as_str) {
                Some("redacted_thinking") => redacted_block_from_openai(p),
                _ => thinking_block_from_openai(p),
            };
            blocks.insert(0, block);
        }
    } else if blocks
        .iter()
        .all(|b| b.get("type").and_then(Value::as_str) != Some("thinking"))
        && let Some(reasoning) = m
            .get("reasoning_content")
            .and_then(Value::as_str)
            .or_else(|| m.get("reasoning").and_then(Value::as_str))
        && !reasoning.is_empty()
    {
        blocks.insert(0, json!({ "type": "thinking", "thinking": reasoning }));
    }
    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            let id = c.get("id").and_then(Value::as_str).unwrap_or("call_0");
            let func = c.get("function").unwrap_or(c);
            let name = func.get("name").and_then(Value::as_str).unwrap_or("");
            let input = parse_arguments(func.get("arguments"));
            blocks.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
    }
    if blocks.len() == 1 && blocks[0].get("type").and_then(Value::as_str) == Some("text") {
        return blocks[0]
            .get("text")
            .cloned()
            .unwrap_or(Value::String(String::new()));
    }
    if blocks.is_empty() {
        return Value::String(String::new());
    }
    Value::Array(blocks)
}

fn thinking_block_from_openai(p: &Value) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("thinking"));
    let text = p
        .get("thinking")
        .and_then(Value::as_str)
        .or_else(|| p.get("text").and_then(Value::as_str))
        .unwrap_or("");
    m.insert("thinking".into(), json!(text));
    if let Some(sig) = p.get("signature") {
        m.insert("signature".into(), sig.clone());
    }
    Value::Object(m)
}

fn redacted_block_from_openai(p: &Value) -> Value {
    json!({
        "type": "redacted_thinking",
        "data": p.get("data").cloned().unwrap_or(json!("")),
    })
}

fn openai_user_content(m: &Value) -> Value {
    let msg_cc = m.get("cache_control");
    match m.get("content") {
        Some(Value::String(s)) => {
            if let Some(cc) = msg_cc {
                return Value::Array(vec![text_block(s, Some(cc))]);
            }
            Value::String(s.clone())
        }
        Some(Value::Array(parts)) => {
            let mut blocks: Vec<Value> = Vec::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            blocks.push(text_block(t, p.get("cache_control").or(msg_cc)));
                        }
                    }
                    Some("image_url") => {
                        if let Some(mut b) = openai_image_to_anthropic(p) {
                            if let Some(obj) = b.as_object_mut() {
                                copy_cache_control(obj, p);
                            }
                            blocks.push(b);
                        }
                    }
                    _ => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            blocks.push(text_block(t, p.get("cache_control").or(msg_cc)));
                        }
                    }
                }
            }
            if blocks.len() == 1
                && blocks[0].get("type").and_then(Value::as_str) == Some("text")
                && blocks[0].get("cache_control").is_none()
            {
                return blocks[0]
                    .get("text")
                    .cloned()
                    .unwrap_or(Value::String(String::new()));
            }
            if blocks.is_empty() {
                return Value::String(String::new());
            }
            Value::Array(blocks)
        }
        _ => Value::String(String::new()),
    }
}

fn openai_image_to_anthropic(part: &Value) -> Option<Value> {
    let url = part
        .pointer("/image_url/url")
        .and_then(Value::as_str)
        .or_else(|| part.get("url").and_then(Value::as_str))?;
    let (media_type, data) = parse_data_uri(url)?;
    Some(json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    }))
}

fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(";base64,")?;
    if meta.is_empty() || data.is_empty() {
        return None;
    }
    Some((meta, data))
}

fn parse_arguments(v: Option<&Value>) -> Value {
    match v {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(o) if o.is_object() => o.clone(),
        _ => json!({}),
    }
}

fn max_tokens_of(v: &Value) -> Option<u64> {
    v.get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| v.get("max_completion_tokens").and_then(Value::as_u64))
}

fn message_text(m: &Value) -> Option<String> {
    match m.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let t: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
                .collect();
            Some(t)
        }
        Some(Value::Null) | None => None,
        _ => None,
    }
}

fn copy_if(out: &mut Map<String, Value>, v: &Value, key: &str) {
    if let Some(x) = v.get(key) {
        out.insert(key.to_owned(), x.clone());
    }
}

// --- request: Anthropic → OpenAI --------------------------------------------

fn anthropic_req_to_openai(v: &Value) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    if let Some(t) = v.get("max_tokens") {
        out.insert("max_tokens".into(), t.clone());
    }
    copy_if(&mut out, v, "temperature");
    copy_if(&mut out, v, "top_p");
    copy_if(&mut out, v, "stream");
    if let Some(stop) = v.get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().filter_map(anthropic_tool_to_openai).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    if let Some(choice) = v.get("tool_choice") {
        out.insert(
            "tool_choice".into(),
            anthropic_tool_choice_to_openai(choice),
        );
    }
    if let Some(t) = v.get("thinking")
        && let Some(effort) = effort_from_thinking(t)
    {
        out.insert("reasoning_effort".into(), json!(effort));
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
                "assistant" => messages.push(anthropic_assistant_to_openai(m)),
                _ => messages.extend(anthropic_user_to_openai(m)),
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
                let t = anthropic_system_text(sys)?;
                (!t.is_empty()).then(|| json!({ "role": "system", "content": t }))
            }
        }
        _ => None,
    }
}

fn anthropic_system_text(sys: &Value) -> Option<String> {
    match sys {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => Some(
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str).or_else(|| b.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

fn anthropic_tool_to_openai(t: &Value) -> Option<Value> {
    let name = t.get("name").and_then(Value::as_str)?;
    let mut func = Map::new();
    func.insert("name".into(), json!(name));
    if let Some(d) = t.get("description") {
        func.insert("description".into(), d.clone());
    }
    let params = t
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
    func.insert("parameters".into(), params);
    let mut tool = json!({ "type": "function", "function": Value::Object(func) });
    if let Some(obj) = tool.as_object_mut() {
        copy_cache_control(obj, t);
    }
    Some(tool)
}

fn anthropic_tool_choice_to_openai(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("none") => json!("none"),
        Some("any") => json!("required"),
        Some("tool") => {
            let name = choice.get("name").and_then(Value::as_str).unwrap_or("");
            json!({ "type": "function", "function": { "name": name } })
        }
        _ => json!("auto"),
    }
}

fn anthropic_assistant_to_openai(m: &Value) -> Value {
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
                            .map(|i| serde_json::to_string(i).unwrap_or_else(|_| "{}".into()))
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
    if !thinking_blocks.is_empty() {
        msg.insert("thinking".into(), Value::Array(thinking_blocks));
    }
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
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("tool_result") => {
                        flush_user(&mut user_parts, &mut out);
                        let id = b
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .unwrap_or("call_0");
                        let content = match b.get("content") {
                            Some(Value::String(s)) => s.clone(),
                            Some(Value::Array(inner)) => inner
                                .iter()
                                .filter_map(|x| x.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join(""),
                            _ => String::new(),
                        };
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": id,
                            "content": content,
                        }));
                    }
                    Some("image") => {
                        if let Some(mut part) = anthropic_image_to_openai(b) {
                            if let Some(obj) = part.as_object_mut() {
                                copy_cache_control(obj, b);
                            }
                            user_parts.push(part);
                        }
                    }
                    Some("text") | None => {
                        if let Some(t) =
                            b.get("text").and_then(Value::as_str).or_else(|| b.as_str())
                        {
                            user_parts.push(text_block(t, b.get("cache_control")));
                        }
                    }
                    _ => {}
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

fn anthropic_image_to_openai(b: &Value) -> Option<Value> {
    let src = b.get("source")?;
    if src.get("type").and_then(Value::as_str) != Some("base64") {
        return None;
    }
    let media = src.get("media_type").and_then(Value::as_str)?;
    let data = src.get("data").and_then(Value::as_str)?;
    Some(json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{media};base64,{data}") },
    }))
}

// --- response JSON ----------------------------------------------------------

fn anthropic_resp_to_openai(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("msg_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let assistant = anthropic_assistant_to_openai(&json!({
        "role": "assistant",
        "content": v.get("content").cloned().unwrap_or(json!([])),
    }));
    let finish = map_stop_to_openai(v.get("stop_reason").and_then(Value::as_str));
    let usage = map_usage_to_openai(v.get("usage"));
    json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": assistant,
            "finish_reason": finish,
        }],
        "usage": usage,
    })
}

fn openai_resp_to_anthropic(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("chatcmpl_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first());
    let message = choice.and_then(|c| c.get("message")).unwrap_or(v);
    let content = openai_assistant_content(message);
    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str);
    let usage = map_usage_to_anthropic(v.get("usage"));
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": normalize_anth_content(content),
        "stop_reason": map_stop_to_anthropic(finish),
        "usage": usage,
    })
}

fn normalize_anth_content(content: Value) -> Value {
    match content {
        Value::String(s) => json!([{ "type": "text", "text": s }]),
        other => other,
    }
}

fn map_stop_to_openai(reason: Option<&str>) -> &'static str {
    match reason {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        _ => "stop",
    }
}

fn map_stop_to_anthropic(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        _ => "end_turn",
    }
}

fn map_usage_to_openai(usage: Option<&Value>) -> Value {
    let u = usage.unwrap_or(&Value::Null);
    let input = u
        .get("input_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("prompt_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let output = u
        .get("output_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("completion_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let mut m = json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": input.saturating_add(output),
    });
    if let Some(obj) = m.as_object_mut() {
        let cached = u
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
            })
            .unwrap_or(0);
        if cached > 0 {
            obj.insert(
                "prompt_tokens_details".into(),
                json!({ "cached_tokens": cached }),
            );
        }
        let reasoning = u
            .pointer("/output_tokens_details/thinking_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/completion_tokens_details/reasoning_tokens")
                    .and_then(Value::as_u64)
            });
        if let Some(r) = reasoning {
            obj.insert(
                "completion_tokens_details".into(),
                json!({ "reasoning_tokens": r }),
            );
        }
    }
    m
}

fn map_usage_to_anthropic(usage: Option<&Value>) -> Value {
    let u = usage.unwrap_or(&Value::Null);
    let input = u
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("input_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let output = u
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("output_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let mut m = json!({ "input_tokens": input, "output_tokens": output });
    if let Some(obj) = m.as_object_mut() {
        if let Some(c) = u
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .or_else(|| u.get("cache_read_input_tokens").and_then(Value::as_u64))
            && c > 0
        {
            obj.insert("cache_read_input_tokens".into(), json!(c));
        }
        if let Some(w) = u.get("cache_creation_input_tokens").and_then(Value::as_u64)
            && w > 0
        {
            obj.insert("cache_creation_input_tokens".into(), json!(w));
        }
        if let Some(r) = u
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/output_tokens_details/thinking_tokens")
                    .and_then(Value::as_u64)
            })
        {
            obj.insert(
                "output_tokens_details".into(),
                json!({ "thinking_tokens": r }),
            );
        }
    }
    m
}

// --- request: Responses ↔ Chat Completions ----------------------------------

fn responses_req_to_openai(v: &Value) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    copy_if(&mut out, v, "temperature");
    copy_if(&mut out, v, "top_p");
    copy_if(&mut out, v, "stream");
    if let Some(t) = v
        .get("max_output_tokens")
        .or_else(|| v.get("max_tokens"))
        .or_else(|| v.get("max_completion_tokens"))
    {
        out.insert("max_tokens".into(), t.clone());
    }
    if let Some(effort) = v
        .get("reasoning_effort")
        .cloned()
        .or_else(|| v.pointer("/reasoning/effort").cloned())
    {
        out.insert("reasoning_effort".into(), effort);
    }
    if let Some(tools) = v.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools.iter().filter_map(responses_tool_to_openai).collect();
        if !mapped.is_empty() {
            out.insert("tools".into(), Value::Array(mapped));
        }
    }
    if let Some(choice) = v.get("tool_choice") {
        out.insert("tool_choice".into(), choice.clone());
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(instr) = v.get("instructions") {
        messages.push(responses_instructions_to_system(instr));
    }
    messages.extend(responses_input_to_messages(v.get("input")));
    out.insert("messages".into(), Value::Array(messages));
    Value::Object(out)
}

fn openai_req_to_responses(v: &Value) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    copy_if(&mut out, v, "temperature");
    copy_if(&mut out, v, "top_p");
    copy_if(&mut out, v, "stream");
    if let Some(t) = max_tokens_of(v) {
        out.insert("max_output_tokens".into(), json!(t));
    } else if let Some(t) = v.get("max_output_tokens") {
        out.insert("max_output_tokens".into(), t.clone());
    }
    if let Some(effort) = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str))
    {
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
        out.insert("tool_choice".into(), choice.clone());
    }

    let mut instructions: Vec<Value> = Vec::new();
    let mut input: Vec<Value> = Vec::new();
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            match role {
                "system" | "developer" => instructions.extend(openai_system_to_responses_blocks(m)),
                "tool" => {
                    if let Some(item) = openai_tool_to_function_call_output(m) {
                        input.push(item);
                    }
                }
                "assistant" => {
                    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                        for c in calls {
                            input.push(openai_tool_call_to_function_call(c));
                        }
                    }
                    let content = openai_message_to_responses_content(m, false);
                    if !content_is_empty(&content) {
                        input.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }));
                    }
                }
                _ => {
                    input.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": openai_message_to_responses_content(m, true),
                    }));
                }
            }
        }
    }
    if !instructions.is_empty() {
        out.insert("instructions".into(), anthropic_system_value(instructions));
    }
    out.insert("input".into(), Value::Array(input));
    Value::Object(out)
}

fn responses_tool_to_openai(t: &Value) -> Option<Value> {
    if t.get("function").is_some() {
        return Some(t.clone());
    }
    let typ = t.get("type").and_then(Value::as_str).unwrap_or("function");
    if typ != "function" {
        return None;
    }
    let name = t.get("name").and_then(Value::as_str)?;
    let mut func = Map::new();
    func.insert("name".into(), json!(name));
    if let Some(d) = t.get("description") {
        func.insert("description".into(), d.clone());
    }
    if let Some(p) = t.get("parameters") {
        func.insert("parameters".into(), p.clone());
    }
    if let Some(s) = t.get("strict") {
        func.insert("strict".into(), s.clone());
    }
    let mut out = Map::new();
    out.insert("type".into(), json!("function"));
    out.insert("function".into(), Value::Object(func));
    copy_cache_control(&mut out, t);
    Some(Value::Object(out))
}

fn openai_tool_to_responses(t: &Value) -> Option<Value> {
    let func = t.get("function").unwrap_or(t);
    let name = func.get("name").and_then(Value::as_str)?;
    let mut m = Map::new();
    m.insert("type".into(), json!("function"));
    m.insert("name".into(), json!(name));
    if let Some(d) = t.get("description") {
        m.insert("description".into(), d.clone());
    }
    if let Some(p) = func.get("parameters") {
        m.insert("parameters".into(), p.clone());
    }
    if let Some(s) = func.get("strict") {
        m.insert("strict".into(), s.clone());
    }
    copy_cache_control(&mut m, t);
    if !m.contains_key("cache_control") {
        copy_cache_control(&mut m, func);
    }
    Some(Value::Object(m))
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

fn responses_input_to_messages(input: Option<&Value>) -> Vec<Value> {
    match input {
        Some(Value::String(s)) => vec![json!({ "role": "user", "content": s })],
        Some(Value::Array(items)) => items.iter().flat_map(responses_item_to_messages).collect(),
        _ => Vec::new(),
    }
}

fn responses_item_to_messages(item: &Value) -> Vec<Value> {
    let typ = item.get("type").and_then(Value::as_str).unwrap_or("");
    match typ {
        "function_call" => {
            let call = json!({
                "id": item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(json!("call_0")),
                "type": "function",
                "function": {
                    "name": item.get("name").cloned().unwrap_or(json!("")),
                    "arguments": match item.get("arguments") {
                        Some(Value::String(s)) => Value::String(s.clone()),
                        Some(other) => json!(value_string(other)),
                        None => json!("{}"),
                    },
                }
            });
            vec![json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [call],
            })]
        }
        "function_call_output" => {
            let mut m = Map::new();
            m.insert("role".into(), json!("tool"));
            if let Some(id) = item.get("call_id").or_else(|| item.get("id")) {
                m.insert("tool_call_id".into(), id.clone());
            }
            m.insert(
                "content".into(),
                item.get("output").cloned().unwrap_or(json!("")),
            );
            vec![Value::Object(m)]
        }
        "message" | "" => {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
            vec![json!({
                "role": role,
                "content": responses_content_to_openai(item.get("content")),
            })]
        }
        _ => {
            if item.get("role").is_some() {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                vec![json!({
                    "role": role,
                    "content": responses_content_to_openai(item.get("content")),
                })]
            } else {
                Vec::new()
            }
        }
    }
}

fn responses_content_to_openai(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            let mapped: Vec<Value> = parts.iter().filter_map(responses_part_to_openai).collect();
            if mapped.len() == 1
                && mapped[0].get("type").and_then(Value::as_str) == Some("text")
                && mapped[0].get("cache_control").is_none()
            {
                mapped[0]
                    .get("text")
                    .cloned()
                    .unwrap_or(Value::Array(mapped))
            } else {
                Value::Array(mapped)
            }
        }
        Some(other) => other.clone(),
        None => Value::String(String::new()),
    }
}

fn responses_part_to_openai(part: &Value) -> Option<Value> {
    let typ = part.get("type").and_then(Value::as_str).unwrap_or("text");
    match typ {
        "input_text" | "output_text" | "text" => {
            let mut m = Map::new();
            m.insert("type".into(), json!("text"));
            m.insert(
                "text".into(),
                part.get("text").cloned().unwrap_or(json!("")),
            );
            copy_cache_control(&mut m, part);
            Some(Value::Object(m))
        }
        "input_image" | "image_url" => {
            let url = part
                .pointer("/image_url/url")
                .or_else(|| part.get("image_url"))
                .or_else(|| part.get("url"));
            let url = match url {
                Some(Value::String(s)) => s.as_str(),
                _ => return None,
            };
            if url.starts_with("http://") || url.starts_with("https://") {
                return None;
            }
            let mut m = json!({
                "type": "image_url",
                "image_url": { "url": url },
            });
            if let Some(obj) = m.as_object_mut() {
                copy_cache_control(obj, part);
            }
            Some(m)
        }
        "thinking" | "redacted_thinking" => Some(part.clone()),
        _ => None,
    }
}

fn openai_message_to_responses_content(m: &Value, input: bool) -> Value {
    let text_type = if input { "input_text" } else { "output_text" };
    match m.get("content") {
        Some(Value::String(s)) => json!([{ "type": text_type, "text": s }]),
        Some(Value::Array(parts)) => Value::Array(
            parts
                .iter()
                .filter_map(|p| openai_part_to_responses(p, text_type))
                .collect(),
        ),
        Some(Value::Null) | None => json!([]),
        Some(other) => other.clone(),
    }
}

fn openai_part_to_responses(part: &Value, text_type: &str) -> Option<Value> {
    match part.get("type").and_then(Value::as_str) {
        Some("text") | None if part.get("text").is_some() || part.is_string() => {
            let mut m = Map::new();
            m.insert("type".into(), json!(text_type));
            m.insert(
                "text".into(),
                part.get("text")
                    .cloned()
                    .or_else(|| part.as_str().map(|s| json!(s)))
                    .unwrap_or(json!("")),
            );
            copy_cache_control(&mut m, part);
            Some(Value::Object(m))
        }
        Some("image_url") => {
            let url = part.pointer("/image_url/url").and_then(Value::as_str)?;
            if url.starts_with("http://") || url.starts_with("https://") {
                return None;
            }
            let mut m = json!({
                "type": "input_image",
                "image_url": url,
            });
            if let Some(obj) = m.as_object_mut() {
                copy_cache_control(obj, part);
            }
            Some(m)
        }
        Some("thinking") | Some("redacted_thinking") => Some(part.clone()),
        _ => part
            .as_str()
            .map(|s| json!({ "type": text_type, "text": s })),
    }
}

fn content_is_empty(content: &Value) -> bool {
    match content {
        Value::Array(a) => a.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::Null => true,
        _ => false,
    }
}

fn openai_tool_to_function_call_output(m: &Value) -> Option<Value> {
    Some(json!({
        "type": "function_call_output",
        "call_id": m.get("tool_call_id").cloned().unwrap_or(json!("call_0")),
        "output": match m.get("content") {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(other) => other.clone(),
            None => json!(""),
        },
    }))
}

fn openai_tool_call_to_function_call(c: &Value) -> Value {
    let func = c.get("function").unwrap_or(c);
    json!({
        "type": "function_call",
        "call_id": c.get("id").cloned().unwrap_or(json!("call_0")),
        "name": func.get("name").cloned().unwrap_or(json!("")),
        "arguments": func.get("arguments").cloned().unwrap_or(json!("{}")),
    })
}

fn openai_system_to_responses_blocks(m: &Value) -> Vec<Value> {
    match m.get("content") {
        Some(Value::String(s)) => {
            let mut b = json!({ "type": "text", "text": s });
            if let Some(obj) = b.as_object_mut() {
                copy_cache_control(obj, m);
            }
            vec![b]
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                let text = p
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| p.as_str())?;
                let mut b = json!({ "type": "text", "text": text });
                if let Some(obj) = b.as_object_mut() {
                    copy_cache_control(obj, p);
                    if !obj.contains_key("cache_control") {
                        copy_cache_control(obj, m);
                    }
                }
                Some(b)
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn openai_resp_to_responses(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("resp_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first());
    let message = choice.and_then(|c| c.get("message")).unwrap_or(v);
    let mut output = Vec::new();
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            output.push(openai_tool_call_to_function_call(c));
        }
    }
    let content = openai_message_to_responses_content(message, false);
    if !content_is_empty(&content) {
        output.insert(
            0,
            json!({
                "type": "message",
                "id": "msg_translated",
                "role": "assistant",
                "content": content,
                "status": "completed",
            }),
        );
    }
    json!({
        "id": id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output,
        "usage": map_usage_to_responses(v.get("usage")),
    })
}

fn responses_resp_to_openai(v: &Value) -> Value {
    let id = v.get("id").cloned().unwrap_or(json!("resp_translated"));
    let model = v.get("model").cloned().unwrap_or(json!(""));
    let output = v.get("output").and_then(Value::as_array);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    if let Some(items) = output {
        for item in items {
            match item.get("type").and_then(Value::as_str) {
                Some("function_call") => tool_calls.push(json!({
                    "id": item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or(json!("call_0")),
                    "type": "function",
                    "function": {
                        "name": item.get("name").cloned().unwrap_or(json!("")),
                        "arguments": item.get("arguments").cloned().unwrap_or(json!("{}")),
                    }
                })),
                _ => {
                    if let Some(content) = item.get("content").and_then(Value::as_array) {
                        for p in content {
                            if let Some(t) = p.get("text").and_then(Value::as_str) {
                                text.push_str(t);
                            }
                        }
                    } else if let Some(t) = item.get("content").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
            }
        }
    }
    let mut message = json!({ "role": "assistant", "content": text });
    let finish = if tool_calls.is_empty() {
        "stop"
    } else {
        if let Some(obj) = message.as_object_mut() {
            obj.insert("tool_calls".into(), Value::Array(tool_calls));
            obj.insert("content".into(), Value::Null);
        }
        "tool_calls"
    };
    json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": map_usage_to_openai(v.get("usage")),
    })
}

fn map_usage_to_responses(usage: Option<&Value>) -> Value {
    let u = usage.unwrap_or(&Value::Null);
    let input = u
        .get("input_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("prompt_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let output = u
        .get("output_tokens")
        .and_then(Value::as_u64)
        .or_else(|| u.get("completion_tokens").and_then(Value::as_u64))
        .unwrap_or(0);
    let mut m = json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": input.saturating_add(output),
    });
    if let Some(obj) = m.as_object_mut() {
        let cached = u
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
            })
            .or_else(|| {
                u.pointer("/input_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
            })
            .unwrap_or(0);
        if cached > 0 {
            obj.insert(
                "input_tokens_details".into(),
                json!({ "cached_tokens": cached }),
            );
        }
        if let Some(r) = u
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                u.pointer("/output_tokens_details/reasoning_tokens")
                    .and_then(Value::as_u64)
            })
            .or_else(|| {
                u.pointer("/output_tokens_details/thinking_tokens")
                    .and_then(Value::as_u64)
            })
        {
            obj.insert(
                "output_tokens_details".into(),
                json!({ "reasoning_tokens": r }),
            );
        }
    }
    m
}

// --- SSE --------------------------------------------------------------------

/// Event-by-event SSE translator. Incomplete events stay in `buf`; complete events are mapped
/// immediately. Does not wait for `[DONE]` before forwarding deltas.
pub struct SseBridge {
    client: Endpoint,
    upstream: Endpoint,
    buf: Vec<u8>,
    ant_to_oai: AntToOai,
    oai_to_ant: OaiToAnt,
    oai_to_resp: OaiToResp,
    resp_to_oai: RespToOai,
    /// An `error` event already went to the client. Flush must not invent a success close
    /// (`message_start`/`message_stop` or a trailing `[DONE]`-only envelope that implies a message).
    errored: bool,
}

#[derive(Default)]
struct AntToOai {
    id: String,
    model: String,
    next_tool: u32,
    input_tokens: u64,
    cache_read: u64,
    thinking_tokens: Option<u64>,
    done: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum OpenBlock {
    #[default]
    None,
    Text,
    Tool,
    Thinking,
}

#[derive(Default)]
struct OaiToAnt {
    started: bool,
    id: String,
    model: String,
    next_block: u32,
    open: OpenBlock,
    pending_stop: Option<&'static str>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    finished: bool,
}

#[derive(Default)]
struct OaiToResp {
    id: String,
    model: String,
    created: bool,
    completed: bool,
    usage: Option<Value>,
}

#[derive(Default)]
struct RespToOai {
    done: bool,
}

impl SseBridge {
    pub fn new(client: Endpoint, upstream: Endpoint) -> Self {
        Self {
            client,
            upstream,
            buf: Vec::new(),
            ant_to_oai: AntToOai::default(),
            oai_to_ant: OaiToAnt::default(),
            oai_to_resp: OaiToResp::default(),
            resp_to_oai: RespToOai::default(),
            errored: false,
        }
    }

    /// Feed upstream SSE bytes. Returns client-dialect SSE bytes (possibly empty).
    pub fn feed(&mut self, data: &[u8], end: bool) -> Vec<u8> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();
        while let Some(raw) = take_event(&mut self.buf) {
            out.extend(self.map_event(&raw));
        }
        if end {
            if !self.buf.is_empty() {
                let rest = std::mem::take(&mut self.buf);
                out.extend(self.map_event(&rest));
            }
            out.extend(self.flush());
        }
        out
    }

    fn map_event(&mut self, raw: &[u8]) -> Vec<u8> {
        let (event, data) = parse_sse(raw);
        if data.is_empty() && event.is_empty() {
            return Vec::new();
        }
        match (self.upstream, self.client) {
            (Endpoint::Messages, Endpoint::ChatCompletions) => self.ant_event_to_oai(&event, &data),
            (Endpoint::ChatCompletions, Endpoint::Messages) => self.oai_event_to_ant(&event, &data),
            (Endpoint::ChatCompletions, Endpoint::Responses) => {
                self.oai_event_to_resp(&event, &data)
            }
            (Endpoint::Responses, Endpoint::ChatCompletions) => {
                self.resp_event_to_oai(&event, &data)
            }
            (Endpoint::Messages, Endpoint::Responses) => {
                let chat = self.ant_event_to_oai(&event, &data);
                self.rewrite_chat_sse(&chat, |s, e, d| s.oai_event_to_resp(e, d))
            }
            (Endpoint::Responses, Endpoint::Messages) => {
                let chat = self.resp_event_to_oai(&event, &data);
                self.rewrite_chat_sse(&chat, |s, e, d| s.oai_event_to_ant(e, d))
            }
            (a, b) if a == b => raw.to_vec(),
            _ => Vec::new(),
        }
    }

    fn rewrite_chat_sse(
        &mut self,
        bytes: &[u8],
        mut map: impl FnMut(&mut Self, &str, &str) -> Vec<u8>,
    ) -> Vec<u8> {
        let mut buf = bytes.to_vec();
        let mut out = Vec::new();
        while let Some(raw) = take_event(&mut buf) {
            let (event, data) = parse_sse(&raw);
            if data.is_empty() && event.is_empty() {
                continue;
            }
            out.extend(map(self, &event, &data));
        }
        out
    }

    fn flush(&mut self) -> Vec<u8> {
        if self.errored {
            return match self.client {
                Endpoint::ChatCompletions => {
                    if self.ant_to_oai.done || self.resp_to_oai.done {
                        Vec::new()
                    } else {
                        self.ant_to_oai.done = true;
                        self.resp_to_oai.done = true;
                        sse_data("[DONE]")
                    }
                }
                Endpoint::Messages | Endpoint::Responses => Vec::new(),
            };
        }
        match (self.upstream, self.client) {
            (_, Endpoint::ChatCompletions) => {
                if self.ant_to_oai.done || self.resp_to_oai.done {
                    Vec::new()
                } else {
                    self.ant_to_oai.done = true;
                    self.resp_to_oai.done = true;
                    sse_data("[DONE]")
                }
            }
            (_, Endpoint::Messages) => self.oai_to_ant.finish(),
            (Endpoint::Messages, Endpoint::Responses) => {
                let chat = if self.ant_to_oai.done {
                    Vec::new()
                } else {
                    self.ant_to_oai.done = true;
                    sse_data("[DONE]")
                };
                let mut out = self.rewrite_chat_sse(&chat, |s, e, d| s.oai_event_to_resp(e, d));
                out.extend(self.oai_to_resp.finish());
                out
            }
            (_, Endpoint::Responses) => self.oai_to_resp.finish(),
        }
    }

    fn ant_event_to_oai(&mut self, event: &str, data: &str) -> Vec<u8> {
        if event == "ping" {
            return Vec::new();
        }
        if data == "[DONE]" {
            if self.errored || self.ant_to_oai.done {
                return Vec::new();
            }
            self.ant_to_oai.done = true;
            return sse_data("[DONE]");
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) || event == "error" {
            self.errored = true;
            return sse_data(&value_string(&map_error(&v, Endpoint::ChatCompletions)));
        }
        let typ = if event.is_empty() {
            v.get("type").and_then(Value::as_str).unwrap_or("")
        } else {
            event
        };
        match typ {
            "message_start" => {
                let msg = v.get("message").unwrap_or(&v);
                if let Some(id) = msg.get("id").and_then(Value::as_str) {
                    self.ant_to_oai.id = id.to_owned();
                }
                if let Some(model) = msg.get("model").and_then(Value::as_str) {
                    self.ant_to_oai.model = model.to_owned();
                }
                if let Some(u) = msg.get("usage") {
                    self.ant_to_oai.input_tokens =
                        u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                    self.ant_to_oai.cache_read = u
                        .get("cache_read_input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                }
                self.emit_oai_delta(json!({ "role": "assistant", "content": "" }), None)
            }
            "content_block_start" => {
                let block = v.get("content_block").unwrap_or(&Value::Null);
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        let idx = self.ant_to_oai.next_tool;
                        self.ant_to_oai.next_tool = self.ant_to_oai.next_tool.saturating_add(1);
                        let id = block.get("id").and_then(Value::as_str).unwrap_or("call_0");
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                        self.emit_oai_delta(
                            json!({
                                "tool_calls": [{
                                    "index": idx,
                                    "id": id,
                                    "type": "function",
                                    "function": { "name": name, "arguments": "" },
                                }]
                            }),
                            None,
                        )
                    }
                    Some("redacted_thinking") => {
                        let data = block.get("data").cloned().unwrap_or(json!(""));
                        self.emit_oai_delta(
                            json!({
                                "thinking": [{ "type": "redacted_thinking", "data": data }]
                            }),
                            None,
                        )
                    }
                    // text / thinking: OpenAI has no block-start
                    _ => Vec::new(),
                }
            }
            "content_block_delta" => {
                let delta = v.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
                        if text.is_empty() {
                            return Vec::new();
                        }
                        self.emit_oai_delta(json!({ "content": text }), None)
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let idx = self.ant_to_oai.next_tool.saturating_sub(1);
                        self.emit_oai_delta(
                            json!({
                                "tool_calls": [{
                                    "index": idx,
                                    "function": { "arguments": partial },
                                }]
                            }),
                            None,
                        )
                    }
                    Some("thinking_delta") => {
                        let text = delta
                            .get("thinking")
                            .and_then(Value::as_str)
                            .or_else(|| delta.get("text").and_then(Value::as_str))
                            .unwrap_or("");
                        if text.is_empty() {
                            return Vec::new();
                        }
                        self.emit_oai_delta(
                            json!({ "reasoning_content": text, "reasoning": text }),
                            None,
                        )
                    }
                    Some("signature_delta") => {
                        let sig = delta.get("signature").and_then(Value::as_str).unwrap_or("");
                        if sig.is_empty() {
                            return Vec::new();
                        }
                        self.emit_oai_delta(json!({ "thinking_signature": sig }), None)
                    }
                    _ => Vec::new(),
                }
            }
            "message_delta" => {
                let stop = v
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                    .or_else(|| v.get("stop_reason").and_then(Value::as_str));
                let finish = map_stop_to_openai(stop);
                if let Some(u) = v.get("usage")
                    && let Some(o) = u.get("output_tokens").and_then(Value::as_u64)
                {
                    if let Some(t) = u
                        .pointer("/output_tokens_details/thinking_tokens")
                        .and_then(Value::as_u64)
                    {
                        self.ant_to_oai.thinking_tokens = Some(t);
                    }
                    let mut finish_bytes = self.emit_oai_delta(json!({}), Some(finish));
                    let mut usage = json!({
                        "prompt_tokens": self.ant_to_oai.input_tokens,
                        "completion_tokens": o,
                        "total_tokens": self.ant_to_oai.input_tokens.saturating_add(o),
                    });
                    if let Some(obj) = usage.as_object_mut() {
                        if self.ant_to_oai.cache_read > 0 {
                            obj.insert(
                                "prompt_tokens_details".into(),
                                json!({ "cached_tokens": self.ant_to_oai.cache_read }),
                            );
                        }
                        if let Some(t) = self.ant_to_oai.thinking_tokens {
                            obj.insert(
                                "completion_tokens_details".into(),
                                json!({ "reasoning_tokens": t }),
                            );
                        }
                    }
                    finish_bytes.extend(self.emit_oai_usage(usage));
                    return finish_bytes;
                }
                self.emit_oai_delta(json!({}), Some(finish))
            }
            "message_stop" => {
                if self.errored || self.ant_to_oai.done {
                    Vec::new()
                } else {
                    self.ant_to_oai.done = true;
                    sse_data("[DONE]")
                }
            }
            "content_block_stop" => Vec::new(),
            _ => Vec::new(),
        }
    }

    fn emit_oai_delta(&mut self, delta: Value, finish: Option<&str>) -> Vec<u8> {
        let mut choice = json!({ "index": 0, "delta": delta });
        if let Some(f) = finish
            && let Some(obj) = choice.as_object_mut()
        {
            obj.insert("finish_reason".into(), json!(f));
        }
        let chunk = json!({
            "id": self.ant_to_oai.id,
            "object": "chat.completion.chunk",
            "model": self.ant_to_oai.model,
            "choices": [choice],
        });
        sse_data(&value_string(&chunk))
    }

    fn emit_oai_usage(&self, usage: Value) -> Vec<u8> {
        let chunk = json!({
            "id": self.ant_to_oai.id,
            "object": "chat.completion.chunk",
            "model": self.ant_to_oai.model,
            "choices": [],
            "usage": usage,
        });
        sse_data(&value_string(&chunk))
    }

    fn oai_event_to_ant(&mut self, _event: &str, data: &str) -> Vec<u8> {
        if data == "[DONE]" {
            if self.errored {
                return Vec::new();
            }
            return self.oai_to_ant.finish();
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) {
            self.errored = true;
            let err = map_error(&v, Endpoint::Messages);
            return sse_named("error", &value_string(&err));
        }
        let mut out = Vec::new();
        if let Some(id) = v.get("id").and_then(Value::as_str)
            && !id.is_empty()
        {
            self.oai_to_ant.id = id.to_owned();
        }
        if let Some(model) = v.get("model").and_then(Value::as_str)
            && !model.is_empty()
        {
            self.oai_to_ant.model = model.to_owned();
        }
        if !self.oai_to_ant.started {
            out.extend(self.oai_to_ant.start());
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            if let Some(content) = delta.get("content").and_then(Value::as_str)
                && !content.is_empty()
            {
                out.extend(self.oai_to_ant.text_delta(content));
            }
            let reasoning = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| delta.get("reasoning").and_then(Value::as_str));
            if let Some(r) = reasoning
                && !r.is_empty()
            {
                out.extend(self.oai_to_ant.thinking_delta(r));
            }
            if let Some(sig) = delta.get("thinking_signature").and_then(Value::as_str)
                && !sig.is_empty()
            {
                out.extend(self.oai_to_ant.signature_delta(sig));
            }
            if let Some(blocks) = delta.get("thinking").and_then(Value::as_array) {
                for b in blocks {
                    if b.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
                        out.extend(self.oai_to_ant.redacted_thinking(b));
                    }
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for c in calls {
                    out.extend(self.oai_to_ant.tool_delta(c));
                }
            }
        }
        if let Some(finish) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            && !finish.is_empty()
            && finish != "null"
        {
            self.oai_to_ant.pending_stop = Some(map_stop_to_anthropic(Some(finish)));
        }
        if let Some(u) = v.get("usage") {
            self.oai_to_ant.prompt_tokens = u.get("prompt_tokens").and_then(Value::as_u64);
            self.oai_to_ant.completion_tokens = u.get("completion_tokens").and_then(Value::as_u64);
        }
        if self.oai_to_ant.pending_stop.is_some() && self.oai_to_ant.completion_tokens.is_some() {
            out.extend(self.oai_to_ant.finish());
        }
        out
    }

    fn oai_event_to_resp(&mut self, _event: &str, data: &str) -> Vec<u8> {
        if data == "[DONE]" {
            return self.oai_to_resp.finish();
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) {
            self.errored = true;
            return sse_data(&value_string(&map_error(&v, Endpoint::Responses)));
        }
        if let Some(id) = v.get("id").and_then(Value::as_str)
            && self.oai_to_resp.id.is_empty()
        {
            self.oai_to_resp.id = id.to_owned();
        }
        if let Some(model) = v.get("model").and_then(Value::as_str)
            && self.oai_to_resp.model.is_empty()
        {
            self.oai_to_resp.model = model.to_owned();
        }
        let mut out = Vec::new();
        if !self.oai_to_resp.created {
            self.oai_to_resp.created = true;
            if self.oai_to_resp.id.is_empty() {
                self.oai_to_resp.id = "resp_translated".into();
            }
            let created = json!({
                "type": "response.created",
                "response": {
                    "id": self.oai_to_resp.id,
                    "object": "response",
                    "status": "in_progress",
                    "model": self.oai_to_resp.model,
                }
            });
            out.extend(sse_named("response.created", &value_string(&created)));
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        let delta = choice.and_then(|c| c.get("delta")).unwrap_or(&Value::Null);
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let ev = json!({
                "type": "response.output_text.delta",
                "delta": text,
            });
            out.extend(sse_named("response.output_text.delta", &value_string(&ev)));
        }
        if let Some(reason) = delta
            .get("reasoning_content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            let ev = json!({
                "type": "response.reasoning_text.delta",
                "delta": reason,
            });
            out.extend(sse_named(
                "response.reasoning_text.delta",
                &value_string(&ev),
            ));
        }
        if let Some(u) = v.get("usage") {
            self.oai_to_resp.usage = Some(u.clone());
            out.extend(self.oai_to_resp.finish());
            return out;
        }
        out
    }

    fn resp_event_to_oai(&mut self, event: &str, data: &str) -> Vec<u8> {
        if data == "[DONE]" {
            if self.resp_to_oai.done {
                return Vec::new();
            }
            self.resp_to_oai.done = true;
            return sse_data("[DONE]");
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) || event == "error" {
            self.errored = true;
            return sse_data(&value_string(&map_error(&v, Endpoint::ChatCompletions)));
        }
        let typ = if event.is_empty() {
            v.get("type").and_then(Value::as_str).unwrap_or("")
        } else {
            event
        };
        match typ {
            "response.output_text.delta" => {
                let text = v
                    .get("delta")
                    .and_then(|d| {
                        d.as_str()
                            .map(str::to_owned)
                            .or_else(|| d.get("text").and_then(Value::as_str).map(str::to_owned))
                    })
                    .or_else(|| v.get("text").and_then(Value::as_str).map(str::to_owned))
                    .unwrap_or_default();
                if text.is_empty() {
                    return Vec::new();
                }
                sse_data(&value_string(&json!({
                    "id": "chatcmpl_translated",
                    "object": "chat.completion.chunk",
                    "choices": [{ "index": 0, "delta": { "content": text } }],
                })))
            }
            "response.reasoning_text.delta" | "response.reasoning.delta" => {
                let text = v.get("delta").and_then(Value::as_str).unwrap_or("");
                if text.is_empty() {
                    return Vec::new();
                }
                sse_data(&value_string(&json!({
                    "id": "chatcmpl_translated",
                    "object": "chat.completion.chunk",
                    "choices": [{ "index": 0, "delta": { "reasoning_content": text } }],
                })))
            }
            "response.completed" => {
                let resp = v.get("response").unwrap_or(&v);
                let usage = map_usage_to_openai(resp.get("usage"));
                let mut out = sse_data(&value_string(&json!({
                    "id": "chatcmpl_translated",
                    "object": "chat.completion.chunk",
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                    "usage": usage,
                })));
                if !self.resp_to_oai.done {
                    self.resp_to_oai.done = true;
                    out.extend(sse_data("[DONE]"));
                }
                out
            }
            _ => Vec::new(),
        }
    }
}

impl OaiToResp {
    fn finish(&mut self) -> Vec<u8> {
        if self.completed {
            return Vec::new();
        }
        self.completed = true;
        if self.id.is_empty() {
            self.id = "resp_translated".into();
        }
        let usage = self
            .usage
            .as_ref()
            .map(|u| map_usage_to_responses(Some(u)))
            .unwrap_or_else(|| map_usage_to_responses(None));
        let ev = json!({
            "type": "response.completed",
            "response": {
                "id": self.id,
                "object": "response",
                "status": "completed",
                "model": self.model,
                "usage": usage,
            }
        });
        sse_named("response.completed", &value_string(&ev))
    }
}

impl OaiToAnt {
    fn start(&mut self) -> Vec<u8> {
        self.started = true;
        if self.id.is_empty() {
            self.id = "chatcmpl_translated".into();
        }
        let msg = json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "usage": { "input_tokens": 0, "output_tokens": 0 },
            }
        });
        sse_named("message_start", &value_string(&msg))
    }

    fn close_open(&mut self) -> Vec<u8> {
        if self.open == OpenBlock::None {
            return Vec::new();
        }
        let idx = self.next_block.saturating_sub(1);
        self.open = OpenBlock::None;
        sse_named(
            "content_block_stop",
            &value_string(&json!({ "type": "content_block_stop", "index": idx })),
        )
    }

    fn text_delta(&mut self, text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open == OpenBlock::Tool || self.open == OpenBlock::Thinking {
            out.extend(self.close_open());
        }
        if self.open != OpenBlock::Text {
            let idx = self.next_block;
            self.next_block = self.next_block.saturating_add(1);
            self.open = OpenBlock::Text;
            out.extend(sse_named(
                "content_block_start",
                &value_string(&json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "text", "text": "" },
                })),
            ));
        }
        let idx = self.next_block.saturating_sub(1);
        out.extend(sse_named(
            "content_block_delta",
            &value_string(&json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "text_delta", "text": text },
            })),
        ));
        out
    }

    fn tool_delta(&mut self, call: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open == OpenBlock::Text || self.open == OpenBlock::Thinking {
            out.extend(self.close_open());
        }
        let id = call.get("id").and_then(Value::as_str);
        let name = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .or_else(|| call.get("name").and_then(Value::as_str));
        if self.open != OpenBlock::Tool || id.is_some() {
            if self.open == OpenBlock::Tool {
                out.extend(self.close_open());
            }
            let idx = self.next_block;
            self.next_block = self.next_block.saturating_add(1);
            self.open = OpenBlock::Tool;
            out.extend(sse_named(
                "content_block_start",
                &value_string(&json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": id.unwrap_or("call_0"),
                        "name": name.unwrap_or(""),
                        "input": {},
                    },
                })),
            ));
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(Value::as_str)
            && !args.is_empty()
        {
            let idx = self.next_block.saturating_sub(1);
            out.extend(sse_named(
                "content_block_delta",
                &value_string(&json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "input_json_delta", "partial_json": args },
                })),
            ));
        }
        out
    }

    fn thinking_delta(&mut self, text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open == OpenBlock::Text || self.open == OpenBlock::Tool {
            out.extend(self.close_open());
        }
        if self.open != OpenBlock::Thinking {
            let idx = self.next_block;
            self.next_block = self.next_block.saturating_add(1);
            self.open = OpenBlock::Thinking;
            out.extend(sse_named(
                "content_block_start",
                &value_string(&json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "thinking", "thinking": "" },
                })),
            ));
        }
        let idx = self.next_block.saturating_sub(1);
        out.extend(sse_named(
            "content_block_delta",
            &value_string(&json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "thinking_delta", "thinking": text },
            })),
        ));
        out
    }

    fn signature_delta(&mut self, sig: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open != OpenBlock::Thinking {
            out.extend(self.thinking_delta(""));
        }
        let idx = self.next_block.saturating_sub(1);
        out.extend(sse_named(
            "content_block_delta",
            &value_string(&json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "signature_delta", "signature": sig },
            })),
        ));
        out
    }

    fn redacted_thinking(&mut self, block: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        if self.open != OpenBlock::None {
            out.extend(self.close_open());
        }
        let idx = self.next_block;
        self.next_block = self.next_block.saturating_add(1);
        self.open = OpenBlock::Thinking;
        out.extend(sse_named(
            "content_block_start",
            &value_string(&json!({
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "redacted_thinking",
                    "data": block.get("data").cloned().unwrap_or(json!("")),
                },
            })),
        ));
        out
    }

    fn finish(&mut self) -> Vec<u8> {
        if self.finished {
            return Vec::new();
        }
        // A usage-only / `[DONE]`-only upstream stream never opened a text block. `start()`'s
        // bytes used to be discarded (`let _ =`), so the client got `message_delta`/`message_stop`
        // with no `message_start` — which a stock Anthropic SDK treats as a broken stream.
        let mut out = if self.started {
            Vec::new()
        } else {
            self.start()
        };
        self.finished = true;
        out.extend(self.close_open());
        let stop = self.pending_stop.unwrap_or("end_turn");
        let output = self.completion_tokens.unwrap_or(0);
        let input = self.prompt_tokens.unwrap_or(0);
        out.extend(sse_named(
            "message_delta",
            &value_string(&json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop },
                "usage": { "input_tokens": input, "output_tokens": output },
            })),
        ));
        out.extend(sse_named(
            "message_stop",
            &value_string(&json!({ "type": "message_stop" })),
        ));
        out
    }
}

fn take_event(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    let end = find_event_end(buf)?;
    let event = buf.drain(..end).collect();
    Some(event)
}

fn find_event_end(buf: &[u8]) -> Option<usize> {
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4);
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|i| i + 2);
    match (crlf, lf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        _ => None,
    }
}

fn parse_sse(raw: &[u8]) -> (String, String) {
    let text = String::from_utf8_lossy(raw);
    let mut event = String::new();
    let mut data = String::new();
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(rest) = line.strip_prefix("event:") {
            event = rest.trim().to_owned();
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        }
    }
    (event, data)
}

fn sse_data(data: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() + 8);
    o.extend_from_slice(b"data: ");
    o.extend_from_slice(data.as_bytes());
    o.extend_from_slice(b"\n\n");
    o
}

fn sse_named(event: &str, data: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(event.len() + data.len() + 16);
    o.extend_from_slice(b"event: ");
    o.extend_from_slice(event.as_bytes());
    o.extend_from_slice(b"\ndata: ");
    o.extend_from_slice(data.as_bytes());
    o.extend_from_slice(b"\n\n");
    o
}

fn value_string(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "{}".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oai_req() -> Value {
        json!({
            "model": "claude-opus-4-8",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ],
            "max_tokens": 16,
            "temperature": 0.2,
            "stream": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
            "tool_choice": "auto",
            "stream_options": {"include_usage": true}
        })
    }

    #[test]
    fn openai_request_maps_system_tools_and_drops_stream_options() {
        let body = serde_json::to_vec(&oai_req()).unwrap();
        let out = request(Endpoint::ChatCompletions, Endpoint::Messages, &body);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["system"], "be brief");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["stream"], true);
        assert!(v.get("stream_options").is_none(), "{v}");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["tools"][0]["name"], "get_weather");
        assert_eq!(
            v["tools"][0]["input_schema"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(v["tool_choice"]["type"], "auto");
        assert_eq!(v["model"], "claude-opus-4-8");
    }

    #[test]
    fn openai_request_defaults_max_tokens() {
        let body = br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            body,
        ))
        .unwrap();
        assert_eq!(v["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn tool_loop_round_trips_openai_through_anthropic() {
        let oai = json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "64F"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}}
            }}]
        });
        let anth_bytes = request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        );
        let anth: Value = serde_json::from_slice(&anth_bytes).unwrap();
        assert_eq!(anth["messages"][1]["role"], "assistant");
        assert_eq!(anth["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(anth["messages"][1]["content"][0]["input"]["city"], "SF");
        assert_eq!(anth["messages"][2]["role"], "user");
        assert_eq!(anth["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(anth["messages"][2]["content"][0]["tool_use_id"], "call_1");

        let back_bytes = request(Endpoint::Messages, Endpoint::ChatCompletions, &anth_bytes);
        let back: Value = serde_json::from_slice(&back_bytes).unwrap();
        assert_eq!(back["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(back["messages"][2]["role"], "tool");
        assert_eq!(back["messages"][2]["content"], "64F");
        assert_eq!(back["tools"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn anthropic_stream_true_does_not_inject_stream_options_here() {
        // include_usage is spliced by the proxy onto the *translated OpenAI* body, not here.
        let body = br#"{"model":"gpt-4o-mini","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            body,
        ))
        .unwrap();
        assert_eq!(v["stream"], true);
        assert!(v.get("stream_options").is_none(), "{v}");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["max_tokens"], 8);
    }

    #[test]
    fn response_json_maps_usage_and_text() {
        let anth = json!({
            "id": "msg_mock",
            "type": "message",
            "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 13, "output_tokens": 7}
        });
        let oai: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["object"], "chat.completion");
        assert_eq!(oai["choices"][0]["message"]["content"], "hi");
        assert_eq!(oai["usage"]["prompt_tokens"], 13);
        assert_eq!(oai["usage"]["completion_tokens"], 7);

        let back: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["type"], "message");
        assert_eq!(back["content"][0]["text"], "hi");
        assert_eq!(back["usage"]["input_tokens"], 13);
    }

    #[test]
    fn error_bodies_map_into_the_client_envelope() {
        let oai_err = br#"{"error":{"message":"nope","type":"invalid_request_error"}}"#;
        let anth: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            oai_err,
        ))
        .unwrap();
        assert_eq!(anth["type"], "error");
        assert_eq!(anth["error"]["message"], "nope");

        let back: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn anthropic_sse_becomes_chat_completion_chunk_without_waiting_for_stop() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let start = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
        );
        let delta = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        );
        let out = String::from_utf8(b.feed(start.as_bytes(), false)).unwrap();
        assert!(out.contains("chat.completion.chunk"), "{out}");
        let out = String::from_utf8(b.feed(delta.as_bytes(), false)).unwrap();
        assert!(out.contains("chat.completion.chunk"), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(
            !out.contains("[DONE]"),
            "must not wait for message_stop: {out}"
        );
    }

    #[test]
    fn sse_event_split_across_chunks_is_held() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let first = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"";
        let out = b.feed(first, false);
        assert!(out.is_empty(), "incomplete event must be withheld");
        let rest = b",\"delta\":{\"type\":\"text_delta\",\"text\":\"z\"}}\n\n";
        let out = String::from_utf8(b.feed(rest, false)).unwrap();
        assert!(out.contains("\"z\""), "{out}");
    }

    #[test]
    fn openai_sse_becomes_anthropic_events_and_does_not_wait_for_done() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let chunk = "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        let out = String::from_utf8(b.feed(chunk.as_bytes(), false)).unwrap();
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("text_delta"), "{out}");
        assert!(out.contains("hi"), "{out}");
        assert!(
            !out.contains("message_stop"),
            "must not wait for [DONE]: {out}"
        );
    }

    #[test]
    fn same_wire_is_a_byte_copy() {
        let body = br#"{"model":"x"}"#;
        assert_eq!(
            request(Endpoint::ChatCompletions, Endpoint::ChatCompletions, body),
            body
        );
        assert_eq!(
            response_json(Endpoint::Messages, Endpoint::Messages, body),
            body
        );
    }

    #[test]
    fn openai_sse_flush_without_deltas_still_emits_message_start() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let out = String::from_utf8(b.feed(b"", true)).unwrap();
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("event: message_stop"), "{out}");
    }

    #[test]
    fn anthropic_tool_sse_becomes_openai_tool_calls() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let src = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let mut out = Vec::new();
        for byte in src.as_bytes() {
            out.extend(b.feed(&[*byte], false));
        }
        out.extend(b.feed(b"", true));
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"tool_calls\""), "{text}");
        assert!(text.contains("get_weather"), "{text}");
        assert!(text.contains("toolu_1"), "{text}");
        assert!(text.contains("city"), "{text}");
        assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
        assert!(text.contains("\"prompt_tokens\":13"), "{text}");
        assert!(text.contains("[DONE]"), "{text}");
    }

    #[test]
    fn openai_tool_sse_becomes_anthropic_tool_use() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let src = concat!(
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n",
            "data: [DONE]\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("\"type\":\"tool_use\""), "{out}");
        assert!(out.contains("get_weather"), "{out}");
        assert!(out.contains("call_1"), "{out}");
        assert!(out.contains("input_json_delta"), "{out}");
        assert!(out.contains("\"stop_reason\":\"tool_use\""), "{out}");
        assert!(out.contains("event: message_stop"), "{out}");
        assert!(!out.contains("chat.completion.chunk"), "{out}");
    }

    #[test]
    fn sse_error_events_map_into_the_client_envelope() {
        let mut oai = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let anth_err = concat!(
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try again\"}}\n\n",
        );
        let out = String::from_utf8(oai.feed(anth_err.as_bytes(), true)).unwrap();
        assert!(out.contains("\"message\":\"try again\""), "{out}");
        assert!(out.contains("overloaded_error"), "{out}");
        assert!(!out.contains("event: error"), "{out}");

        let mut anth = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let oai_err = "data: {\"error\":{\"message\":\"try again\",\"type\":\"server_error\"}}\n\n";
        let out = String::from_utf8(anth.feed(oai_err.as_bytes(), true)).unwrap();
        assert!(out.contains("event: error"), "{out}");
        assert!(out.contains("\"type\":\"error\""), "{out}");
        assert!(out.contains("try again"), "{out}");
        assert!(
            !out.contains("event: message_start"),
            "error stream must not invent a success close: {out}"
        );
    }

    #[test]
    fn nonstream_tool_json_round_trips() {
        let anth = json!({
            "id": "msg_mock",
            "type": "message",
            "model": "claude-opus-4-8",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 13, "output_tokens": 7}
        });
        let oai: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            oai["choices"][0]["message"]["tool_calls"][0]["id"],
            "toolu_1"
        );
        assert_eq!(
            oai["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        let args = oai["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert!(args.contains("SF"), "{args}");

        let back: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["stop_reason"], "tool_use");
        assert_eq!(back["content"][0]["type"], "tool_use");
        assert_eq!(back["content"][0]["input"]["city"], "SF");
    }

    #[test]
    fn openai_cache_control_and_reasoning_reach_anthropic_fields() {
        let oai = json!({
            "model": "claude-opus-4-8",
            "reasoning_effort": "high",
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                },
                "cache_control": {"type": "ephemeral"}
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        assert_eq!(v["thinking"]["type"], "enabled");
        assert_eq!(v["thinking"]["budget_tokens"], 8192);
        assert_eq!(
            v["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(v["tools"][0]["cache_control"]["type"], "ephemeral");
        assert!(
            v.get("reasoning_effort").is_none(),
            "Anthropic takes thinking, not reasoning_effort: {v}"
        );
        assert!(
            v.get("max_output_tokens").is_none() && v.get("input").is_none(),
            "Responses-only fields must stay dropped: {v}"
        );
    }

    #[test]
    fn anthropic_thinking_becomes_openai_reasoning_effort() {
        let anth = json!({
            "model": "m",
            "max_tokens": 8,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "hi"}]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(v["reasoning_effort"], "low");
        assert!(v.get("thinking").is_none(), "{v}");
    }

    #[test]
    fn thinking_and_redacted_thinking_round_trip_in_history() {
        let anth = json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "redacted"},
                    {"type": "text", "text": "hi"}
                ]
            }]
        });
        let oai: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["messages"][0]["content"], "hi");
        assert_eq!(oai["messages"][0]["reasoning_content"], "plan");
        assert_eq!(oai["messages"][0]["thinking"][0]["type"], "thinking");
        assert_eq!(
            oai["messages"][0]["thinking"][1]["type"],
            "redacted_thinking"
        );

        let back: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        let content = &back["messages"][0]["content"];
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "plan");
        assert_eq!(content[0]["signature"], "sig");
        assert_eq!(content[1]["type"], "redacted_thinking");
        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["text"], "hi");
    }

    #[test]
    fn http_image_urls_are_still_dropped() {
        let oai = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
                ]
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["content"], "see");
    }

    #[test]
    fn anthropic_thinking_sse_reappears_on_openai_stream() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let src = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"cache_read_input_tokens\":4}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7,\"output_tokens_details\":{\"thinking_tokens\":3}}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("\"reasoning_content\":\"plan\""), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(out.contains("\"reasoning_tokens\":3"), "{out}");
        assert!(out.contains("\"cached_tokens\":4"), "{out}");
    }

    #[test]
    fn openai_reasoning_sse_becomes_anthropic_thinking() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let src = concat!(
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("\"type\":\"thinking\""), "{out}");
        assert!(out.contains("thinking_delta"), "{out}");
        assert!(out.contains("plan"), "{out}");
        assert!(out.contains("text_delta"), "{out}");
        assert!(out.contains("hi"), "{out}");
    }

    #[test]
    fn client_usage_maps_cache_and_reasoning_without_changing_shape_of_tools() {
        let anth = json!({
            "id": "msg_mock",
            "type": "message",
            "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 13,
                "output_tokens": 7,
                "cache_read_input_tokens": 4,
                "output_tokens_details": {"thinking_tokens": 3}
            }
        });
        let oai: Value = serde_json::from_slice(&response_json(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(
            oai["usage"]["completion_tokens_details"]["reasoning_tokens"],
            3
        );
    }

    fn stock_responses(model: &str) -> Value {
        json!({
            "model": model,
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}]
            }],
            "max_output_tokens": 16,
            "stream": true,
            "store": false,
            "reasoning": { "effort": "high" },
        })
    }

    #[test]
    fn stock_responses_body_becomes_chat_completions_and_drops_store() {
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&stock_responses("gpt-4o-mini")).unwrap(),
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["stream"], true);
        assert_eq!(v["reasoning_effort"], "high");
        assert!(v.get("store").is_none(), "{v}");
        assert!(v.get("input").is_none(), "{v}");
        assert!(v.get("max_output_tokens").is_none(), "{v}");
    }

    #[test]
    fn responses_body_with_a_claude_id_becomes_messages() {
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&stock_responses("claude-opus-4-8")).unwrap(),
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["thinking"]["type"], "enabled");
        assert!(v.get("store").is_none(), "{v}");
        assert!(v.get("input").is_none(), "{v}");
    }

    #[test]
    fn openai_sse_becomes_responses_events_without_waiting_for_done() {
        let mut b = SseBridge::new(Endpoint::Responses, Endpoint::ChatCompletions);
        let chunk = "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        let out = String::from_utf8(b.feed(chunk.as_bytes(), false)).unwrap();
        assert!(out.contains("response.output_text.delta"), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(
            !out.contains("response.completed"),
            "must not wait for [DONE]: {out}"
        );
        let rest = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";
        let out = String::from_utf8(b.feed(rest.as_bytes(), true)).unwrap();
        assert!(out.contains("response.completed"), "{out}");
        assert!(out.contains("\"input_tokens\":5"), "{out}");
        assert!(out.contains("\"output_tokens\":9"), "{out}");
    }

    #[test]
    fn anthropic_sse_becomes_responses_via_chat() {
        let mut b = SseBridge::new(Endpoint::Responses, Endpoint::Messages);
        let src = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":13,\"output_tokens\":1}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(out.contains("response.output_text.delta"), "{out}");
        assert!(out.contains("\"hi\""), "{out}");
        assert!(out.contains("response.completed"), "{out}");
        assert!(out.contains("\"input_tokens\":13"), "{out}");
    }

    #[test]
    fn responses_json_round_trips_text_and_usage() {
        let chat = json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18 }
        });
        let resp: Value = serde_json::from_slice(&response_json(
            Endpoint::ChatCompletions,
            Endpoint::Responses,
            &serde_json::to_vec(&chat).unwrap(),
        ))
        .unwrap();
        assert_eq!(resp["object"], "response");
        assert_eq!(resp["output"][0]["content"][0]["text"], "hi");
        assert_eq!(resp["usage"]["input_tokens"], 11);
        assert_eq!(resp["usage"]["output_tokens"], 7);
        assert!(resp.get("choices").is_none(), "{resp}");
    }
}
