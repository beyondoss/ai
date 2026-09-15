//! Chat Completions ↔ Messages translation for a managed catalog walk.
//!
//! Triggered only when the inbound path names one wire (`/v1/chat/completions` or `/v1/messages`,
//! including the `/auto` suffix) and the catalog row's `wire` is the other. Same-wire walks stay a
//! byte relay. `/{provider}/…` never translates.
//!
//! v1 is allowed to be lossy on extras a stock SDK does not need for a tool loop:
//! - **Dropped:** `thinking` / `redacted_thinking` blocks, `cache_control`, `reasoning_effort`,
//!   `stream_options` on the Anthropic body, Responses-only fields (`store`, `previous_response_id`,
//!   `include`, `truncation`) when translating *off* Responses onto Chat Completions / Messages,
//!   image `http(s)` URLs (Anthropic wants base64). Base64 data-URI images are converted both ways.
//!   Same-endpoint Responses is a byte relay: those fields pass through.
//! - **Required mapping:** system/messages, `max_tokens`, temperature, stop, stream, tools,
//!   `tool_choice`, text + tool_use/tool_result, usage. Anthropic requires `max_tokens`; a missing
//!   OpenAI value becomes 4096.
//!
//! Usage/billing parse the **upstream** body. This module only reshapes bytes the client sees.

use crate::route::Dialect;
use serde_json::{Map, Value, json};

/// Anthropic requires this; OpenAI does not. Used when the Chat Completions body omitted it.
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Per-request translate state, boxed on [`crate::proxy`]'s model-routed path only.
pub struct TranslateState {
    /// Inbound dialect — what the client sent and what it must receive.
    pub client: Dialect,
    /// SSE translator, created in `response_filter` once the upstream is known to stream.
    pub sse: Option<SseBridge>,
    /// Non-stream JSON, withheld until end-of-stream so we can map the object.
    pub json_buf: Vec<u8>,
}

impl TranslateState {
    pub fn new(client: Dialect) -> Self {
        Self {
            client,
            sse: None,
            json_buf: Vec::new(),
        }
    }
}

/// Map a buffered request body from `from` (inbound) to `to` (upstream / row wire).
///
/// Unparseable JSON is returned unchanged so the provider 400s rather than us 502ing after
/// headers have already gone upstream. The candidate `model` id is spliced by the caller
/// **after** this returns.
pub fn request(from: Dialect, to: Dialect, body: &[u8]) -> Vec<u8> {
    if from == to {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if !v.is_object() {
        return body.to_vec();
    }
    let mapped = match (from, to) {
        (Dialect::OpenAi, Dialect::Anthropic) => openai_req_to_anthropic(&v),
        (Dialect::Anthropic, Dialect::OpenAi) => anthropic_req_to_openai(&v),
        _ => v,
    };
    encode(&mapped)
}

/// Map a non-stream JSON response from `upstream` into `client`. Error objects are reshaped
/// into the client's error envelope.
pub fn response_json(upstream: Dialect, client: Dialect, body: &[u8]) -> Vec<u8> {
    if upstream == client {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if looks_like_error(&v) {
        return encode(&map_error(&v, client));
    }
    let mapped = match (upstream, client) {
        (Dialect::Anthropic, Dialect::OpenAi) => anthropic_resp_to_openai(&v),
        (Dialect::OpenAi, Dialect::Anthropic) => openai_resp_to_anthropic(&v),
        _ => v,
    };
    encode(&mapped)
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
    // Success bodies never carry a top-level `error` next to `choices` / `content` / `type:message`.
    if v.get("choices").is_some() || v.get("content").is_some() {
        return false;
    }
    if v.get("type").and_then(Value::as_str) == Some("message") {
        return false;
    }
    true
}

fn map_error(v: &Value, client: Dialect) -> Value {
    let (typ, msg) = extract_error(v);
    match client {
        Dialect::OpenAi => json!({ "error": { "message": msg, "type": typ } }),
        Dialect::Anthropic => json!({ "type": "error", "error": { "type": typ, "message": msg } }),
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

    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    if let Some(arr) = v.get("messages").and_then(Value::as_array) {
        let mut pending_tool_results: Vec<Value> = Vec::new();
        for m in arr {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            match role {
                "system" | "developer" => {
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    if let Some(t) = message_text(m) {
                        system_parts.push(t);
                    }
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
        out.insert("system".into(), Value::String(system_parts.join("\n")));
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
    Some(Value::Object(m))
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
    if let Some(t) = message_text(m)
        && !t.is_empty()
    {
        blocks.push(json!({ "type": "text", "text": t }));
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

fn openai_user_content(m: &Value) -> Value {
    match m.get("content") {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            let mut blocks: Vec<Value> = Vec::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            blocks.push(json!({ "type": "text", "text": t }));
                        }
                    }
                    Some("image_url") => {
                        if let Some(b) = openai_image_to_anthropic(p) {
                            blocks.push(b);
                        }
                    }
                    _ => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            blocks.push(json!({ "type": "text", "text": t }));
                        }
                    }
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

/// Root-level Responses session field that cannot be honored off `/v1/responses`.
///
/// `None` means the body is a `store: false` one-shot (no `previous_response_id`). Unparseable
/// JSON is `Some("store")` so a catalog walk fail-closes onto a real Responses arm rather than
/// silently stripping session state.
pub fn responses_session_field(body: &[u8]) -> Option<&'static str> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Some("store");
    };
    if previous_response_id_set(&v) {
        return Some("previous_response_id");
    }
    match v.get("store") {
        Some(Value::Bool(false)) => None,
        _ => Some("store"),
    }
}

fn previous_response_id_set(v: &Value) -> bool {
    match v.get("previous_response_id") {
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    }
}

/// Drop Responses-only session/control fields and reshape `input` onto Chat Completions
/// `messages`. Used when a `store: false` one-shot is allowed to leave the Responses endpoint.
/// Same-endpoint Responses must not call this — those fields pass through as a byte relay.
pub fn responses_to_chat(body: &[u8]) -> Vec<u8> {
    let Ok(Value::Object(mut m)) = serde_json::from_slice(body) else {
        return body.to_vec();
    };
    m.remove("store");
    m.remove("previous_response_id");
    m.remove("include");
    m.remove("truncation");
    if let Some(t) = m.remove("max_output_tokens") {
        m.entry("max_tokens".to_owned()).or_insert(t);
    }
    let instructions = m.remove("instructions");
    if m.get("messages").is_none()
        && let Some(input) = m.remove("input")
    {
        m.insert("messages".into(), input_to_messages(input));
    } else {
        m.remove("input");
    }
    if let Some(instr) = instructions {
        prepend_system_message(&mut m, instr);
    }
    encode(&Value::Object(m))
}

fn input_to_messages(input: Value) -> Value {
    match input {
        Value::String(s) => json!([{ "role": "user", "content": s }]),
        Value::Array(items) => {
            let msgs: Vec<Value> = items
                .into_iter()
                .filter_map(input_item_to_message)
                .collect();
            if msgs.is_empty() {
                json!([{ "role": "user", "content": "" }])
            } else {
                Value::Array(msgs)
            }
        }
        other => json!([{ "role": "user", "content": other }]),
    }
}

fn input_item_to_message(item: Value) -> Option<Value> {
    let Value::Object(mut obj) = item else {
        return Some(json!({ "role": "user", "content": item }));
    };
    if obj.get("role").is_some() {
        rewrite_input_text_parts(&mut obj);
        return Some(Value::Object(obj));
    }
    match obj.get("type").and_then(Value::as_str) {
        Some("message") => {
            rewrite_input_text_parts(&mut obj);
            if obj.get("role").is_none() {
                obj.insert("role".into(), json!("user"));
            }
            Some(Value::Object(obj))
        }
        Some("input_text") => {
            let text = obj.get("text").cloned().unwrap_or(json!(""));
            Some(json!({ "role": "user", "content": text }))
        }
        _ => None,
    }
}

fn rewrite_input_text_parts(obj: &mut Map<String, Value>) {
    let Some(Value::Array(parts)) = obj.get_mut("content") else {
        return;
    };
    for p in parts.iter_mut() {
        if p.get("type").and_then(Value::as_str) != Some("input_text") {
            continue;
        }
        if let Some(t) = p.get("text").cloned() {
            *p = json!({ "type": "text", "text": t });
        }
    }
}

fn prepend_system_message(m: &mut Map<String, Value>, instr: Value) {
    let text = match instr {
        Value::String(s) => s,
        other => other.to_string(),
    };
    if text.is_empty() {
        return;
    }
    let sys = json!({ "role": "system", "content": text });
    match m.get_mut("messages") {
        Some(Value::Array(msgs)) => msgs.insert(0, sys),
        _ => {
            m.insert("messages".into(), json!([sys]));
        }
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

    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = v.get("system")
        && let Some(t) = anthropic_system_text(sys)
        && !t.is_empty()
    {
        messages.push(json!({ "role": "system", "content": t }));
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
    Some(json!({ "type": "function", "function": Value::Object(func) }))
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
                    // thinking / redacted_thinking / cache_control extras: drop
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
                        if let Some(part) = anthropic_image_to_openai(b) {
                            user_parts.push(part);
                        }
                    }
                    Some("text") | None => {
                        if let Some(t) =
                            b.get("text").and_then(Value::as_str).or_else(|| b.as_str())
                        {
                            user_parts.push(json!({ "type": "text", "text": t }));
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
    if let Some(c) = u.get("cache_read_input_tokens").and_then(Value::as_u64)
        && c > 0
        && let Some(obj) = m.as_object_mut()
    {
        obj.insert(
            "prompt_tokens_details".into(),
            json!({ "cached_tokens": c }),
        );
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
    json!({ "input_tokens": input, "output_tokens": output })
}

// --- SSE --------------------------------------------------------------------

/// Event-by-event SSE translator. Incomplete events stay in `buf`; complete events are mapped
/// immediately. Does not wait for `[DONE]` before forwarding deltas.
pub struct SseBridge {
    client: Dialect,
    buf: Vec<u8>,
    ant_to_oai: AntToOai,
    oai_to_ant: OaiToAnt,
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
    done: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum OpenBlock {
    #[default]
    None,
    Text,
    Tool,
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

impl SseBridge {
    pub fn new(client: Dialect) -> Self {
        Self {
            client,
            buf: Vec::new(),
            ant_to_oai: AntToOai::default(),
            oai_to_ant: OaiToAnt::default(),
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
        match self.client {
            Dialect::OpenAi => self.ant_event_to_oai(&event, &data),
            Dialect::Anthropic => self.oai_event_to_ant(&event, &data),
        }
    }

    fn flush(&mut self) -> Vec<u8> {
        if self.errored {
            return match self.client {
                // OpenAI clients conventionally see `[DONE]` after a streamed error chunk.
                Dialect::OpenAi => {
                    if self.ant_to_oai.done {
                        Vec::new()
                    } else {
                        self.ant_to_oai.done = true;
                        sse_data("[DONE]")
                    }
                }
                // Anthropic has no `[DONE]`; a `message_start` after `event: error` is a broken stream.
                Dialect::Anthropic => Vec::new(),
            };
        }
        match self.client {
            Dialect::OpenAi => {
                if self.ant_to_oai.done {
                    return Vec::new();
                }
                self.ant_to_oai.done = true;
                sse_data("[DONE]")
            }
            Dialect::Anthropic => self.oai_to_ant.finish(),
        }
    }

    fn ant_event_to_oai(&mut self, event: &str, data: &str) -> Vec<u8> {
        if event == "ping" {
            return Vec::new();
        }
        if data == "[DONE]" {
            if self.errored {
                return Vec::new();
            }
            return self.flush();
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        if looks_like_error(&v) || event == "error" {
            self.errored = true;
            return sse_data(&value_string(&map_error(&v, Dialect::OpenAi)));
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
                    // text: OpenAI has no block-start; thinking: drop
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
                    let mut finish_bytes = self.emit_oai_delta(json!({}), Some(finish));
                    let mut usage = json!({
                        "prompt_tokens": self.ant_to_oai.input_tokens,
                        "completion_tokens": o,
                        "total_tokens": self.ant_to_oai.input_tokens.saturating_add(o),
                    });
                    if self.ant_to_oai.cache_read > 0
                        && let Some(obj) = usage.as_object_mut()
                    {
                        obj.insert(
                            "prompt_tokens_details".into(),
                            json!({ "cached_tokens": self.ant_to_oai.cache_read }),
                        );
                    }
                    finish_bytes.extend(self.emit_oai_usage(usage));
                    return finish_bytes;
                }
                self.emit_oai_delta(json!({}), Some(finish))
            }
            "message_stop" => self.flush(),
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
            let err = map_error(&v, Dialect::Anthropic);
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
        if self.open == OpenBlock::Tool {
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
        if self.open == OpenBlock::Text {
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
        let out = request(Dialect::OpenAi, Dialect::Anthropic, &body);
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
        let v: Value =
            serde_json::from_slice(&request(Dialect::OpenAi, Dialect::Anthropic, body)).unwrap();
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
            Dialect::OpenAi,
            Dialect::Anthropic,
            &serde_json::to_vec(&oai).unwrap(),
        );
        let anth: Value = serde_json::from_slice(&anth_bytes).unwrap();
        assert_eq!(anth["messages"][1]["role"], "assistant");
        assert_eq!(anth["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(anth["messages"][1]["content"][0]["input"]["city"], "SF");
        assert_eq!(anth["messages"][2]["role"], "user");
        assert_eq!(anth["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(anth["messages"][2]["content"][0]["tool_use_id"], "call_1");

        let back_bytes = request(Dialect::Anthropic, Dialect::OpenAi, &anth_bytes);
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
        let v: Value =
            serde_json::from_slice(&request(Dialect::Anthropic, Dialect::OpenAi, body)).unwrap();
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
            Dialect::Anthropic,
            Dialect::OpenAi,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(oai["object"], "chat.completion");
        assert_eq!(oai["choices"][0]["message"]["content"], "hi");
        assert_eq!(oai["usage"]["prompt_tokens"], 13);
        assert_eq!(oai["usage"]["completion_tokens"], 7);

        let back: Value = serde_json::from_slice(&response_json(
            Dialect::OpenAi,
            Dialect::Anthropic,
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
        let anth: Value =
            serde_json::from_slice(&response_json(Dialect::OpenAi, Dialect::Anthropic, oai_err))
                .unwrap();
        assert_eq!(anth["type"], "error");
        assert_eq!(anth["error"]["message"], "nope");

        let back: Value = serde_json::from_slice(&response_json(
            Dialect::Anthropic,
            Dialect::OpenAi,
            &serde_json::to_vec(&anth).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn anthropic_sse_becomes_chat_completion_chunk_without_waiting_for_stop() {
        let mut b = SseBridge::new(Dialect::OpenAi);
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
        let mut b = SseBridge::new(Dialect::OpenAi);
        let first = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\"";
        let out = b.feed(first, false);
        assert!(out.is_empty(), "incomplete event must be withheld");
        let rest = b",\"delta\":{\"type\":\"text_delta\",\"text\":\"z\"}}\n\n";
        let out = String::from_utf8(b.feed(rest, false)).unwrap();
        assert!(out.contains("\"z\""), "{out}");
    }

    #[test]
    fn openai_sse_becomes_anthropic_events_and_does_not_wait_for_done() {
        let mut b = SseBridge::new(Dialect::Anthropic);
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
        assert_eq!(request(Dialect::OpenAi, Dialect::OpenAi, body), body);
        assert_eq!(
            response_json(Dialect::Anthropic, Dialect::Anthropic, body),
            body
        );
    }

    #[test]
    fn openai_sse_flush_without_deltas_still_emits_message_start() {
        let mut b = SseBridge::new(Dialect::Anthropic);
        let out = String::from_utf8(b.feed(b"", true)).unwrap();
        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("event: message_stop"), "{out}");
    }

    #[test]
    fn anthropic_tool_sse_becomes_openai_tool_calls() {
        let mut b = SseBridge::new(Dialect::OpenAi);
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
        let mut b = SseBridge::new(Dialect::Anthropic);
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
        let mut oai = SseBridge::new(Dialect::OpenAi);
        let anth_err = concat!(
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try again\"}}\n\n",
        );
        let out = String::from_utf8(oai.feed(anth_err.as_bytes(), true)).unwrap();
        assert!(out.contains("\"message\":\"try again\""), "{out}");
        assert!(out.contains("overloaded_error"), "{out}");
        assert!(!out.contains("event: error"), "{out}");

        let mut anth = SseBridge::new(Dialect::Anthropic);
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
            Dialect::Anthropic,
            Dialect::OpenAi,
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
            Dialect::OpenAi,
            Dialect::Anthropic,
            &serde_json::to_vec(&oai).unwrap(),
        ))
        .unwrap();
        assert_eq!(back["stop_reason"], "tool_use");
        assert_eq!(back["content"][0]["type"], "tool_use");
        assert_eq!(back["content"][0]["input"]["city"], "SF");
    }

    #[test]
    fn responses_session_field_names_previous_response_id_first() {
        let body = br#"{"model":"gpt-4o","previous_response_id":"resp_1","store":false}"#;
        assert_eq!(responses_session_field(body), Some("previous_response_id"));
        let omitted = br#"{"model":"gpt-4o","input":"hi"}"#;
        assert_eq!(responses_session_field(omitted), Some("store"));
        let stored = br#"{"model":"gpt-4o","store":true}"#;
        assert_eq!(responses_session_field(stored), Some("store"));
        let one_shot = br#"{"model":"gpt-4o","store":false,"input":"hi"}"#;
        assert_eq!(responses_session_field(one_shot), None);
        let empty_prev = br#"{"model":"gpt-4o","previous_response_id":"","store":false}"#;
        assert_eq!(responses_session_field(empty_prev), None);
        assert_eq!(responses_session_field(b"not-json"), Some("store"));
    }

    #[test]
    fn responses_to_chat_drops_session_fields_and_maps_input() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-4o",
            "input": [{"role":"user","content":[{"type":"input_text","text":"hi"}]}],
            "max_output_tokens": 16,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "truncation": "auto",
            "previous_response_id": "resp_x",
        }))
        .unwrap();
        let v: Value = serde_json::from_slice(&responses_to_chat(&body)).unwrap();
        assert!(v.get("store").is_none(), "{v}");
        assert!(v.get("previous_response_id").is_none(), "{v}");
        assert!(v.get("include").is_none(), "{v}");
        assert!(v.get("truncation").is_none(), "{v}");
        assert!(v.get("input").is_none(), "{v}");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"][0]["type"], "text");
        assert_eq!(v["messages"][0]["content"][0]["text"], "hi");
        assert_eq!(v["model"], "gpt-4o");
    }
}
