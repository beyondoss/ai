//! Chat Completions ↔ Messages ↔ Responses translation for a managed catalog walk.
//!
//! Triggered when the inbound path names Chat Completions, Messages, or Responses (including the
//! `/auto` suffix) and the catalog row speaks a different one of those three. Same-wire walks stay
//! a byte relay — including `/{provider}/v1/responses`. `/{provider}/…` never translates.
//!
//! Responses ↔ Messages is composed through Chat Completions so thinking / `cache_control` /
//! `reasoning_effort` keep the slice-1 mappings.
//!
//! What crosses, and how:
//! - **Dropped:** Responses-only session fields (`store`, `previous_response_id`, `include`,
//!   `truncation`, …) when leaving Responses; `stream_options` on a Responses or Anthropic body.
//!   Same-endpoint Responses is a byte relay: those session fields pass through. Hints with no
//!   equivalent and no effect on the response's shape (`seed`, penalties, `logit_bias`, `top_k`)
//!   are dropped, as is OpenAI JSON mode (`json_object`): Anthropic's format needs a schema, and
//!   OpenAI already requires a JSON-mode prompt to ask for JSON.
//! - **Forwarded for the provider to reject:** input and options that change what the client gets
//!   back and have no equivalent on the target (`input_audio`, a `file_id` or URL document, `n` > 1,
//!   `logprobs`, audio output). Translation runs after the request headers went upstream, so the
//!   gateway cannot 400 here; passing the field through gets the provider's 400 naming it, instead
//!   of an answer about input the model never saw.
//! - **Images, both ways:** base64 data-URI images convert to Anthropic `base64` sources and back;
//!   `http(s)` image URLs become Anthropic `url` sources and back, and pass through between Chat
//!   Completions and Responses. Nothing is fetched: every one of those upstreams downloads the URL
//!   itself. Amazon Bedrock is the exception — it rejects `url` sources — so a URL image that fails
//!   over onto a Bedrock candidate gets Bedrock's 400 rather than a silently image-less answer.
//! - **PDFs, both ways:** a Chat Completions `file` part with inline `file_data` ↔ an Anthropic
//!   base64 `document` ↔ a Responses `input_file`.
//! - **Structured output, both ways:** `response_format` `json_schema` ↔ Anthropic
//!   `output_config.format` ↔ Responses `text.format`.
//! - **Passed both ways:** `thinking` / `redacted_thinking` blocks, `cache_control` on tools and
//!   content, `parallel_tool_calls: false` ↔ `tool_choice.disable_parallel_tool_use`, `user` ↔
//!   `metadata.user_id`.
//! - **Model-aware onto Messages** (see `ClaudeModel`): `reasoning_effort` becomes adaptive
//!   thinking plus `output_config.effort` on Claude 4.6 and later, and a `budget_tokens` below
//!   `max_tokens` before that. `temperature` / `top_p` are dropped where the model rejects them
//!   (4.7 and later, and anything with thinking on).
//! - **Added onto Messages:** default `cache_control` breakpoints when the client set none (see
//!   `auto_cache_breakpoints`). An OpenAI SDK never marks anything, and without a marker Anthropic
//!   caches nothing.
//! - **Required mapping:** system/messages/`input`, `max_tokens`/`max_output_tokens`, stop,
//!   stream, tools, `tool_choice`, text + tool_use/tool_result, usage. Anthropic requires
//!   `max_tokens`; a missing OpenAI value becomes 4096.
//!
//! Usage/billing parse the **upstream** body. This module only reshapes bytes the client sees.

use crate::route::Endpoint;
use serde_json::{Map, Value, json};

/// Anthropic requires this; OpenAI does not. Used when the Chat Completions body omitted it.
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Most bytes translation holds for one response: a whole non-streaming body, or one SSE event not
/// yet terminated. Sized for the largest legitimate case — a Responses `response.completed` event
/// repeats the entire output — with room to spare; past it `proxy` aborts the response rather than
/// let one upstream grow the gateway's memory without bound.
pub const MAX_TRANSLATE_BUFFER: usize = 32 * 1024 * 1024;

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

    /// Append a non-streaming response chunk. `false`, with nothing appended, once the body would
    /// pass [`MAX_TRANSLATE_BUFFER`].
    #[must_use]
    pub fn push_json(&mut self, chunk: &[u8]) -> bool {
        if self.json_buf.len().saturating_add(chunk.len()) > MAX_TRANSLATE_BUFFER {
            return false;
        }
        self.json_buf.extend_from_slice(chunk);
        true
    }
}

/// Map a buffered request body from `from` (inbound) to `to` (upstream endpoint).
///
/// `upstream_model` is the id this attempt's candidate will receive. Onto Messages it decides
/// which reasoning, sampling and tool-choice controls the Claude model accepts (see `ClaudeModel`).
///
/// Unparseable JSON is returned unchanged so the provider 400s rather than us 502ing after
/// headers have already gone upstream. The candidate `model` id is spliced by the caller
/// **after** this returns.
pub fn request(from: Endpoint, to: Endpoint, body: &[u8], upstream_model: &str) -> Vec<u8> {
    if from == to {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if !v.is_object() {
        return body.to_vec();
    }
    encode(&map_request(from, to, &v, ClaudeModel::of(upstream_model)))
}

fn map_request(from: Endpoint, to: Endpoint, v: &Value, claude: ClaudeModel) -> Value {
    match (from, to) {
        (Endpoint::ChatCompletions, Endpoint::Messages) => openai_req_to_anthropic(v, claude),
        (Endpoint::Messages, Endpoint::ChatCompletions) => anthropic_req_to_openai(v),
        (Endpoint::Responses, Endpoint::ChatCompletions) => responses_req_to_openai(v),
        (Endpoint::ChatCompletions, Endpoint::Responses) => openai_req_to_responses(v),
        (Endpoint::Responses, Endpoint::Messages) => {
            openai_req_to_anthropic(&responses_req_to_openai(v), claude)
        }
        (Endpoint::Messages, Endpoint::Responses) => {
            openai_req_to_responses(&anthropic_req_to_openai(v))
        }
        (a, b) if a == b => v.clone(),
        _ => v.clone(),
    }
}

/// Map a non-stream JSON response from `upstream` into `client`. Error objects are reshaped
/// into the client's error envelope. Same as [`response_json_status`] with a 200.
pub fn response_json(upstream: Endpoint, client: Endpoint, body: &[u8]) -> Vec<u8> {
    response_json_status(upstream, client, 200, body)
}

/// Map a non-stream JSON response carrying HTTP `status`. Any non-2xx JSON body is an error
/// whatever its shape — Bedrock's `{"message": …}` and FastAPI's `{"detail": …}` have no `error`
/// key, and used to be reshaped as an empty success. A body that is not JSON passes unchanged.
pub fn response_json_status(
    upstream: Endpoint,
    client: Endpoint,
    status: u16,
    body: &[u8],
) -> Vec<u8> {
    if upstream == client {
        return body.to_vec();
    }
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return body.to_vec();
    };
    if !(200..300).contains(&status) || looks_like_error(&v) {
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
    // A Responses object is an error only once it failed; its `error` is `null` otherwise.
    if v.get("object").and_then(Value::as_str) == Some("response") {
        return v.get("status").and_then(Value::as_str) == Some("failed")
            && v.get("error").is_some_and(Value::is_object);
    }
    if v.get("error").is_none_or(Value::is_null) {
        return false;
    }
    // Success bodies never carry a top-level `error` next to `choices` / `content` / `output`.
    if v.get("choices").is_some() || v.get("content").is_some() || v.get("output").is_some() {
        return false;
    }
    v.get("type").and_then(Value::as_str) != Some("message")
}

/// Everything an upstream error says, whichever dialect said it.
struct ErrorInfo {
    typ: String,
    message: String,
    code: Option<Value>,
    param: Option<Value>,
    /// OpenRouter's `metadata` (`provider_name`, `raw`, moderation `reasons`), kept whole for an
    /// OpenAI-shaped client.
    metadata: Option<Value>,
}

/// Longest upstream error message quoted into the client's envelope. A provider that answers an
/// error with a whole HTML page gets it cut here, not relayed in full.
const MAX_ERROR_MESSAGE: usize = 4096;

fn error_info(v: &Value) -> ErrorInfo {
    // `{"error": {…}}` (OpenAI, OpenRouter, Anthropic's envelope), a Responses `response.failed`
    // object, or a flat body: a Responses `error` event, Bedrock `{"message"}`, `{"detail"}`.
    let err = match v.get("error") {
        Some(e @ Value::Object(_)) => e,
        _ => v,
    };
    let field = |key: &str| {
        err.get(key)
            .or_else(|| v.get(key))
            .filter(|x| !x.is_null())
            .cloned()
    };
    let mut message = plain_error_message(v).unwrap_or_else(|| clip(&value_string(v)));
    let code = field("code");
    let param = field("param");
    let metadata = err.get("metadata").filter(|m| m.is_object()).cloned();
    let mut typ = err
        .get("type")
        .and_then(Value::as_str)
        .filter(|t| *t != "error")
        .map(str::to_owned);
    // OpenRouter wraps the provider's own error: "Provider returned error" alone tells the client
    // nothing, so the provider's message (and type, when ours has none) is quoted after it.
    if let Some(meta) = metadata.as_ref() {
        let provider = meta.get("provider_name").and_then(Value::as_str);
        let inner = match meta.get("raw") {
            Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
                Ok(parsed) if parsed.is_object() => {
                    if typ.is_none() {
                        typ = parsed
                            .get("error")
                            .and_then(|e| e.get("type"))
                            .and_then(Value::as_str)
                            .filter(|t| *t != "error")
                            .map(str::to_owned);
                    }
                    plain_error_message(&parsed).or_else(|| Some(clip(raw)))
                }
                _ => Some(clip(raw)),
            },
            Some(raw @ Value::Object(_)) => plain_error_message(raw),
            _ => None,
        }
        .filter(|s| !s.is_empty());
        let reasons = meta
            .get("reasons")
            .and_then(Value::as_array)
            .map(|r| {
                r.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .filter(|s| !s.is_empty());
        if let Some(detail) = inner.or(reasons) {
            message = match provider {
                Some(p) => format!("{message} ({p}): {detail}"),
                None => format!("{message}: {detail}"),
            };
        }
    }
    ErrorInfo {
        typ: typ.unwrap_or_else(|| "api_error".to_owned()),
        message,
        code,
        param,
        metadata,
    }
}

/// The human message of an error body, wherever its dialect keeps it.
fn plain_error_message(v: &Value) -> Option<String> {
    let err = v.get("error");
    let msg = match err {
        Some(Value::String(s)) => Some(s.as_str()),
        Some(e) => e.get("message").and_then(Value::as_str),
        None => None,
    }
    .or_else(|| v.get("message").and_then(Value::as_str))
    .or_else(|| v.get("detail").and_then(Value::as_str))
    .or_else(|| v.pointer("/response/error/message").and_then(Value::as_str))?;
    Some(clip(msg))
}

fn clip(s: &str) -> String {
    if s.len() <= MAX_ERROR_MESSAGE {
        return s.to_owned();
    }
    let mut end = MAX_ERROR_MESSAGE;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// Anthropic's error types are a closed set its SDKs match on; name the OpenAI ones that have an
/// equivalent there. Anything else passes as the upstream spelled it.
fn anthropic_error_type(typ: &str, code: Option<&Value>) -> String {
    let code = code.and_then(Value::as_str).unwrap_or("");
    match (typ, code) {
        ("server_error", _) => "api_error",
        ("insufficient_quota", _) | (_, "insufficient_quota") => "billing_error",
        ("requests" | "tokens" | "rate_limit_exceeded", _) | (_, "rate_limit_exceeded") => {
            "rate_limit_error"
        }
        (_, "invalid_api_key") => "authentication_error",
        (_, "model_not_found") => "not_found_error",
        _ => typ,
    }
    .to_owned()
}

/// OpenAI's `code` and `param` are strings; OpenRouter sends the HTTP status as a numeric `code`,
/// which a strictly typed client fails to decode — and then reports that instead of the error.
fn openai_error_field(v: Option<Value>) -> Value {
    match v {
        Some(Value::String(s)) => Value::String(s),
        Some(Value::Null) | None => Value::Null,
        Some(other) => Value::String(value_string(&other)),
    }
}

fn map_error(v: &Value, client: Endpoint) -> Value {
    let info = error_info(v);
    match client {
        Endpoint::Messages => {
            let mut err = Map::new();
            err.insert(
                "type".into(),
                json!(anthropic_error_type(&info.typ, info.code.as_ref())),
            );
            err.insert("message".into(), json!(info.message));
            // Not Anthropic fields, but an SDK hands the whole body to the caller: keep them.
            if let Some(code) = info.code {
                err.insert("code".into(), code);
            }
            if let Some(param) = info.param {
                err.insert("param".into(), param);
            }
            json!({ "type": "error", "error": err })
        }
        Endpoint::ChatCompletions | Endpoint::Responses | Endpoint::Embeddings => {
            let mut err = Map::new();
            err.insert("message".into(), json!(info.message));
            err.insert("type".into(), json!(info.typ));
            err.insert("param".into(), openai_error_field(info.param));
            err.insert("code".into(), openai_error_field(info.code));
            if let Some(meta) = info.metadata {
                err.insert("metadata".into(), meta);
            }
            json!({ "error": err })
        }
    }
}

// --- request: OpenAI → Anthropic --------------------------------------------

fn openai_req_to_anthropic(v: &Value, claude: ClaudeModel) -> Value {
    let mut out = Map::new();
    copy_if(&mut out, v, "model");
    let max_tokens = max_tokens_of(v).unwrap_or(DEFAULT_MAX_TOKENS);
    out.insert("max_tokens".into(), json!(max_tokens));
    openai_reasoning_to_anthropic(v, &mut out, claude.reasoning, max_tokens);
    // Current Claude models 400 on non-default sampling, and every model 400s on it alongside
    // thinking. OpenAI clients send `temperature` by habit; it is a hint, so it goes.
    let thinking_on = out
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|t| t != "disabled");
    if claude.reasoning != ClaudeGen::Adaptive && !thinking_on {
        copy_if(&mut out, v, "temperature");
        copy_if(&mut out, v, "top_p");
    }
    copy_if(&mut out, v, "stream");
    if let Some(format) = openai_format_to_anthropic(v.get("response_format")) {
        set_output_config(&mut out, "format", format);
    }
    if let Some(user) = v
        .get("user")
        .or_else(|| v.get("safety_identifier"))
        .filter(|u| u.is_string())
    {
        out.insert("metadata".into(), json!({ "user_id": user }));
    }
    // No Anthropic equivalent, and each changes what the client gets back. Forwarded verbatim so
    // the provider rejects them by name; dropping them would answer a different question.
    for key in ["audio", "web_search_options", "top_logprobs"] {
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
    // Models that reject forced tool use get `auto` plus a closing system instruction naming what
    // must be called (appended after the messages below).
    let mut must_call: Option<String> = None;
    if let Some(choice) = v.get("tool_choice") {
        let mut mapped = openai_tool_choice_to_anthropic(choice);
        if !claude.forced_tool_choice {
            must_call = match mapped.get("type").and_then(Value::as_str) {
                Some("any") => Some("Respond by calling one of the provided tools.".to_owned()),
                Some("tool") => mapped
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|n| format!("Respond by calling the `{n}` tool.")),
                _ => None,
            };
            if must_call.is_some() {
                mapped = json!({ "type": "auto" });
            }
        }
        out.insert("tool_choice".into(), mapped);
    }
    if v.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false)
        && out.contains_key("tools")
    {
        let choice = out
            .entry("tool_choice")
            .or_insert_with(|| json!({ "type": "auto" }));
        if choice.get("type").and_then(Value::as_str) != Some("none")
            && let Some(obj) = choice.as_object_mut()
        {
            obj.insert("disable_parallel_tool_use".into(), json!(true));
        }
    }

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
    auto_cache_breakpoints(&mut out);
    // After the breakpoints, so the instruction sits past the cached prefix. A mid-conversation
    // system message must follow a user turn; when the request ends on an assistant turn (a prefill,
    // which these models also reject), there is nowhere valid to put it.
    if let Some(text) = must_call
        && let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut)
        && messages
            .last()
            .is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("user"))
    {
        messages.push(json!({ "role": "system", "content": text }));
    }
    Value::Object(out)
}

/// Default prompt-cache breakpoints for a translated request that set none.
///
/// OpenAI caches a repeated prefix on its own; Anthropic caches only up to an explicit
/// `cache_control` marker. A stock OpenAI SDK never sends one, so every Chat Completions or
/// Responses call that translated onto Messages paid full input price for its whole prefix on
/// every turn: an agent loop re-bought its system prompt, tools and history each request.
///
/// A client that marked anything keeps full control: one `cache_control` anywhere and nothing is
/// added. Otherwise, at most two markers (Anthropic allows four):
///
/// - **The static prefix**: the last system block, or the last tool when there is no system.
///   Tools render before system, so one marker covers both. An app's system prompt and tools repeat
///   across its requests, so this is written once and read on every later call.
/// - **The conversation so far**: the last cacheable block of the last message, only once the
///   request holds an assistant turn. The next turn appends to this prefix and reads it back
///   (Anthropic looks back up to 20 blocks for the previous marker). A single-turn request skips it:
///   a write costs 1.25× input and a one-shot never reads it.
///
/// Prefixes shorter than the model's minimum (1024 tokens on most Claude models) are simply not
/// cached; the marker costs nothing there.
fn auto_cache_breakpoints(out: &mut Map<String, Value>) {
    if has_cache_control(out) {
        return;
    }
    let ephemeral = || json!({ "type": "ephemeral" });
    let mut marked_prefix = false;
    if let Some(system) = out.get_mut("system") {
        if let Value::String(t) = system
            && !t.is_empty()
        {
            *system = json!([{ "type": "text", "text": std::mem::take(t) }]);
        }
        if let Some(last) = system.as_array_mut().and_then(|a| a.last_mut())
            && let Some(obj) = last.as_object_mut()
        {
            obj.insert("cache_control".into(), ephemeral());
            marked_prefix = true;
        }
    }
    if !marked_prefix
        && let Some(last) = out
            .get_mut("tools")
            .and_then(Value::as_array_mut)
            .and_then(|a| a.last_mut())
            .and_then(Value::as_object_mut)
    {
        last.insert("cache_control".into(), ephemeral());
    }
    let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    if !messages
        .iter()
        .any(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
    {
        return;
    }
    let Some(last) = messages.last_mut() else {
        return;
    };
    if let Some(Value::String(t)) = last.get("content")
        && !t.is_empty()
    {
        let t = t.clone();
        last["content"] = json!([{ "type": "text", "text": t }]);
    }
    // Thinking blocks cannot carry a marker; walk back to the last block that can.
    if let Some(block) = last
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .and_then(|blocks| {
            blocks.iter_mut().rev().find(|b| {
                !matches!(
                    b.get("type").and_then(Value::as_str),
                    Some("thinking" | "redacted_thinking")
                )
            })
        })
        .and_then(Value::as_object_mut)
    {
        block.insert("cache_control".into(), ephemeral());
    }
}

/// Whether the translated request already carries a `cache_control` on a tool, a system block,
/// or a message content block.
fn has_cache_control(out: &Map<String, Value>) -> bool {
    let marked = |v: &Value| v.get("cache_control").is_some();
    let in_array = |v: Option<&Value>| {
        v.and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(marked))
    };
    in_array(out.get("tools"))
        || in_array(out.get("system"))
        || out
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|ms| ms.iter().any(|m| in_array(m.get("content"))))
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

/// Which reasoning and sampling controls a Claude model accepts. Decided from the id the candidate
/// will receive, because the same OpenAI `reasoning_effort` must become different requests: the
/// old `thinking.budget_tokens` shape is a 400 on every current model, and adaptive thinking is a
/// 400 before 4.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClaudeGen {
    /// Claude 3.x through 4.5, and non-Claude models behind a Messages-compatible API:
    /// `thinking: {type: enabled, budget_tokens}` (less than `max_tokens`), sampling allowed
    /// without thinking.
    Budget,
    /// Claude 4.6: adaptive thinking with `output_config.effort` `low`..`max` (no `xhigh`),
    /// sampling allowed without thinking.
    Adaptive46,
    /// Claude 4.7 and later, Fable, Mythos: adaptive thinking and effort (`xhigh` included).
    /// `budget_tokens` and non-default `temperature` / `top_p` are 400s, and several reject
    /// `thinking: disabled`, so "no reasoning" is an omitted `thinking` at effort `low`.
    Adaptive,
}

/// What the Claude model behind a candidate accepts, parsed from the id it will receive (any
/// spelling: `claude-opus-4-8`, OpenRouter's `anthropic/claude-opus-4.8`, Bedrock's
/// `global.anthropic.claude-haiku-4-5-…`). A dated suffix (`claude-sonnet-4-20250514`) is not a
/// minor version. An unrecognized Claude family is assumed to behave like the newest models; a
/// model that is not Claude at all keeps the long-standing shapes Messages-compatible APIs
/// implement (`budget_tokens`, forced `tool_choice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClaudeModel {
    pub(crate) reasoning: ClaudeGen,
    /// Whether forced `tool_choice` (`any` / `tool`) is accepted. Claude Fable 5.1, Mythos 5.1,
    /// Opus 5.5 and Sonnet 5.5 reject it with a 400.
    pub(crate) forced_tool_choice: bool,
}

impl ClaudeModel {
    pub(crate) fn of(model: &str) -> Self {
        const NOT_CLAUDE: ClaudeModel = ClaudeModel {
            reasoning: ClaudeGen::Budget,
            forced_tool_choice: true,
        };
        const NEWEST: ClaudeModel = ClaudeModel {
            reasoning: ClaudeGen::Adaptive,
            forced_tool_choice: false,
        };
        let Some(at) = model.find("claude-") else {
            return NOT_CLAUDE;
        };
        let mut parts = model[at + "claude-".len()..].split(['-', '.']);
        let family = parts.next().unwrap_or("");
        if family.starts_with(|c: char| c.is_ascii_digit()) {
            // `claude-3-haiku`, `claude-3-5-sonnet-…`.
            return NOT_CLAUDE;
        }
        let Some(major) = parts.next().and_then(|p| p.parse::<u32>().ok()) else {
            return NEWEST;
        };
        let minor = parts
            .next()
            .filter(|p| p.len() <= 2)
            .and_then(|p| p.parse::<u32>().ok())
            .unwrap_or(0);
        let version = (major, minor);
        match family {
            "opus" | "sonnet" | "haiku" => ClaudeModel {
                reasoning: match version {
                    (5.., _) | (4, 7..) => ClaudeGen::Adaptive,
                    (4, 6) => ClaudeGen::Adaptive46,
                    _ => ClaudeGen::Budget,
                },
                forced_tool_choice: version < (5, 5),
            },
            "fable" | "mythos" => ClaudeModel {
                reasoning: ClaudeGen::Adaptive,
                forced_tool_choice: version < (5, 1),
            },
            _ => NEWEST,
        }
    }
}

fn openai_reasoning_to_anthropic(
    v: &Value,
    out: &mut Map<String, Value>,
    claude: ClaudeGen,
    max_tokens: u64,
) {
    // An already-Anthropic `thinking` object wins over `reasoning_effort` so a round-trip that
    // kept the native shape is not re-bucketed.
    if let Some(t) = v.get("thinking").filter(|t| t.is_object()) {
        out.insert("thinking".into(), t.clone());
        return;
    }
    let Some(effort) = v
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/reasoning/effort").and_then(Value::as_str))
    else {
        return;
    };
    let off = matches!(effort, "none" | "off" | "disabled");
    match claude {
        ClaudeGen::Budget => {
            if off {
                out.insert("thinking".into(), json!({ "type": "disabled" }));
                return;
            }
            // `budget_tokens` must be at least 1024 and below `max_tokens`, which also has to hold
            // the answer. Leave the answer at least half; too small a request thinks not at all.
            if max_tokens <= MIN_THINKING_BUDGET {
                return;
            }
            let budget = budget_for_effort(effort)
                .min(max_tokens / 2)
                .clamp(MIN_THINKING_BUDGET, max_tokens - 1);
            out.insert(
                "thinking".into(),
                json!({ "type": "enabled", "budget_tokens": budget }),
            );
        }
        ClaudeGen::Adaptive46 | ClaudeGen::Adaptive => {
            let level = match effort_alias(effort) {
                "none" => "low",
                "xhigh" if claude == ClaudeGen::Adaptive46 => "max",
                "xhigh" if effort == "max" => "max",
                other => other,
            };
            if !off {
                out.insert("thinking".into(), json!({ "type": "adaptive" }));
            }
            set_output_config(out, "effort", json!(level));
        }
    }
}

/// Anthropic's floor for `thinking.budget_tokens`.
const MIN_THINKING_BUDGET: u64 = 1024;

/// Set one field of the request's `output_config` (effort, structured-output format).
fn set_output_config(out: &mut Map<String, Value>, key: &str, value: Value) {
    if let Some(Value::Object(m)) = out.get_mut("output_config") {
        m.insert(key.into(), value);
        return;
    }
    let mut m = Map::new();
    m.insert(key.into(), value);
    out.insert("output_config".into(), Value::Object(m));
}

/// OpenAI `response_format` → Anthropic `output_config.format`. Only `json_schema` has an
/// equivalent. `json_object` (JSON mode, no schema) has none: Anthropic's format needs a schema,
/// and OpenAI already requires a JSON-mode prompt to ask for JSON, so it is dropped. `text` is the
/// default.
fn openai_format_to_anthropic(rf: Option<&Value>) -> Option<Value> {
    let rf = rf?;
    if rf.get("type").and_then(Value::as_str) != Some("json_schema") {
        return None;
    }
    let schema = rf.pointer("/json_schema/schema")?;
    Some(json!({ "type": "json_schema", "schema": schema }))
}

/// Anthropic `output_config.format` → OpenAI `response_format`.
fn anthropic_format_to_openai(v: &Value) -> Option<Value> {
    let format = v.pointer("/output_config/format")?;
    if format.get("type").and_then(Value::as_str) != Some("json_schema") {
        return None;
    }
    let schema = format.get("schema")?;
    Some(json!({
        "type": "json_schema",
        "json_schema": { "name": "response", "strict": true, "schema": schema },
    }))
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
                    Some("file") => {
                        let mut b = openai_file_to_anthropic(p);
                        if let Some(obj) = b.as_object_mut() {
                            copy_cache_control(obj, p);
                        }
                        blocks.push(b);
                    }
                    _ => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            blocks.push(text_block(t, p.get("cache_control").or(msg_cc)));
                        } else {
                            // `input_audio` and anything else without text: no Messages
                            // equivalent. Forwarded so the provider rejects it by name; dropping it
                            // would get an answer about input the model never saw.
                            blocks.push(p.clone());
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

/// Chat Completions `file` part → Anthropic `document`. Inline data (`file_data`, a data URI) maps
/// to a base64 source. A `file_id` names an OpenAI Files upload Anthropic cannot read, so the part
/// is forwarded as-is for the provider to reject by name.
fn openai_file_to_anthropic(part: &Value) -> Value {
    let file = part.get("file");
    let Some((media_type, data)) = file
        .and_then(|f| f.get("file_data"))
        .and_then(Value::as_str)
        .and_then(parse_data_uri)
    else {
        return part.clone();
    };
    let mut doc = json!({
        "type": "document",
        "source": { "type": "base64", "media_type": media_type, "data": data },
    });
    if let Some(name) = file
        .and_then(|f| f.get("filename"))
        .filter(|n| n.is_string())
        && let Some(obj) = doc.as_object_mut()
    {
        obj.insert("title".into(), name.clone());
    }
    doc
}

fn openai_image_to_anthropic(part: &Value) -> Option<Value> {
    let url = part
        .pointer("/image_url/url")
        .and_then(Value::as_str)
        .or_else(|| part.get("url").and_then(Value::as_str))?;
    if is_http_url(url) {
        return Some(json!({
            "type": "image",
            "source": { "type": "url", "url": url },
        }));
    }
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

/// An image the upstream downloads itself. Carried as a URL on every wire rather than dropped:
/// a silently image-less request produces a confident answer about a picture the model never saw.
fn is_http_url(url: &str) -> bool {
    url.starts_with("https://") || url.starts_with("http://")
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
    request(Endpoint::Responses, Endpoint::ChatCompletions, body, "")
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
    // Anthropic's effort lives in `output_config.effort`; `thinking` alone implies one.
    let effort = v
        .pointer("/output_config/effort")
        .and_then(Value::as_str)
        .map(effort_alias)
        .or_else(|| v.get("thinking").and_then(effort_from_thinking));
    if let Some(effort) = effort {
        out.insert("reasoning_effort".into(), json!(effort));
    }
    if let Some(format) = anthropic_format_to_openai(v) {
        out.insert("response_format".into(), format);
    }
    if v.pointer("/tool_choice/disable_parallel_tool_use")
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
                    Some("document") => user_parts.push(anthropic_document_to_openai(b)),
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

/// Anthropic `document` → a Chat Completions part. A base64 PDF becomes a `file` part with a data
/// URI and a plain-text document becomes text. A URL or Files-API document has no Chat Completions
/// equivalent and is forwarded as-is, so the provider rejects it by name instead of the model
/// answering without it.
fn anthropic_document_to_openai(b: &Value) -> Value {
    let src = b.get("source");
    let kind = src.and_then(|s| s.get("type")).and_then(Value::as_str);
    let media = src
        .and_then(|s| s.get("media_type"))
        .and_then(Value::as_str);
    let data = src.and_then(|s| s.get("data")).and_then(Value::as_str);
    match (kind, media, data) {
        (Some("base64"), Some(media), Some(data)) => {
            let filename = b
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("document.pdf");
            json!({
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": format!("data:{media};base64,{data}"),
                }
            })
        }
        (Some("text"), _, Some(text)) => json!({ "type": "text", "text": text }),
        _ => b.clone(),
    }
}

fn anthropic_image_to_openai(b: &Value) -> Option<Value> {
    let src = b.get("source")?;
    match src.get("type").and_then(Value::as_str) {
        Some("base64") => {}
        Some("url") => {
            let url = src
                .get("url")
                .and_then(Value::as_str)
                .filter(|u| is_http_url(u))?;
            return Some(json!({
                "type": "image_url",
                "image_url": { "url": url },
            }));
        }
        _ => return None,
    }
    let media = src.get("media_type").and_then(Value::as_str)?;
    let data = src.get("data").and_then(Value::as_str)?;
    Some(json!({
        "type": "image_url",
        "image_url": { "url": format!("data:{media};base64,{data}") },
    }))
}

// --- response JSON ----------------------------------------------------------

/// Unix seconds, for a `created` / `created_at` the upstream did not supply.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A process-unique id (`{prefix}_gw…`) for an object the upstream left unnamed. Stable for the
/// life of one response because the caller keeps it; unique across responses so a client that
/// keys on it (a tool call id, a Responses item id) never sees two alike.
fn fresh_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX));
    format!("{prefix}_gw{t:x}{n:04x}")
}

fn id_or_fresh(v: Option<&Value>, prefix: &str) -> String {
    v.and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map_or_else(|| fresh_id(prefix), str::to_owned)
}

fn non_empty_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// Anthropic stop reason → Chat Completions `finish_reason`.
fn map_stop_to_openai(reason: Option<&str>) -> &'static str {
    match reason {
        Some("tool_use") => "tool_calls",
        // All three leave the answer unfinished. `pause_turn` is a long server-tool turn Anthropic
        // paused; the remedy is the truncation one — send the partial turn back to continue.
        Some("max_tokens" | "model_context_window_exceeded" | "pause_turn") => "length",
        Some("refusal") => "content_filter",
        _ => "stop",
    }
}

/// Chat Completions `finish_reason` → Anthropic stop reason.
fn map_stop_to_anthropic(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        Some("tool_calls" | "function_call") => "tool_use",
        Some("content_filter") => "refusal",
        _ => "end_turn",
    }
}

/// A `finish_reason` that is already an Anthropic stop reason — OpenRouter's
/// `native_finish_reason` on a Claude model — survives exactly (`stop_sequence`, `pause_turn`).
fn anthropic_stop_reason(reason: &str) -> Option<&'static str> {
    Some(match reason {
        "end_turn" => "end_turn",
        "max_tokens" => "max_tokens",
        "stop_sequence" => "stop_sequence",
        "tool_use" => "tool_use",
        "pause_turn" => "pause_turn",
        "refusal" => "refusal",
        "model_context_window_exceeded" => "model_context_window_exceeded",
        _ => return None,
    })
}

/// The Anthropic stop reason for a Chat choice: the native one when the upstream named it, else
/// the mapped `finish_reason`. A refusal the model spoke (`message.refusal`) is a `refusal`, and
/// tool calls under a plain `stop` (some OpenAI-compatible servers) are a `tool_use` — the reason
/// an Anthropic agent loop checks before running them.
fn anthropic_stop_of(choice: Option<&Value>, refused: bool, called: bool) -> &'static str {
    let native = choice
        .and_then(|c| c.get("native_finish_reason"))
        .and_then(Value::as_str)
        .and_then(anthropic_stop_reason);
    let stop = native.unwrap_or_else(|| {
        map_stop_to_anthropic(
            choice
                .and_then(|c| c.get("finish_reason"))
                .and_then(Value::as_str),
        )
    });
    finish_anthropic_stop(stop, refused, called)
}

fn finish_anthropic_stop(stop: &'static str, refused: bool, called: bool) -> &'static str {
    match stop {
        "end_turn" if refused => "refusal",
        "end_turn" if called => "tool_use",
        _ => stop,
    }
}

/// A Chat `finish_reason` kept past its chunk.
fn chat_finish(reason: &str) -> &'static str {
    match reason {
        "length" => "length",
        "tool_calls" | "function_call" => "tool_calls",
        "content_filter" => "content_filter",
        _ => "stop",
    }
}

/// Chat `finish_reason` → Responses `status` and `incomplete_details`.
fn responses_status(finish: Option<&str>) -> (&'static str, Value) {
    match finish {
        Some("length") => ("incomplete", json!({ "reason": "max_output_tokens" })),
        Some("content_filter") => ("incomplete", json!({ "reason": "content_filter" })),
        _ => ("completed", Value::Null),
    }
}

/// Responses `status` → Chat `finish_reason`, before tool calls are considered.
fn responses_finish(resp: &Value) -> &'static str {
    if resp.get("status").and_then(Value::as_str) != Some("incomplete") {
        return "stop";
    }
    match resp
        .pointer("/incomplete_details/reason")
        .and_then(Value::as_str)
    {
        Some("content_filter") => "content_filter",
        _ => "length",
    }
}

fn u64_at(v: &Value, ptr: &str) -> Option<u64> {
    v.pointer(ptr).and_then(Value::as_u64)
}

/// Token counts in one shape, whichever dialect reported them.
///
/// Anthropic's `input_tokens` counts only the prompt tokens that were neither read from nor
/// written to the cache. OpenAI's `prompt_tokens` / Responses `input_tokens` count the whole
/// prompt, with the cached share broken out beneath it. Passing either number through as the
/// other told an OpenAI client a 7,000-token prompt was 10 tokens, and an Anthropic client that a
/// mostly-cached prompt was uncached.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Usage {
    uncached: u64,
    cache_read: u64,
    cache_write: u64,
    output: u64,
    reasoning: Option<u64>,
}

impl Usage {
    fn from_anthropic(u: &Value) -> Self {
        let mut usage = Self::default();
        usage.merge_anthropic(u);
        usage
    }

    /// Take what `u` reports. A `message_delta` repeats cumulative totals and omits those that do
    /// not apply, so an absent field keeps the `message_start` value.
    fn merge_anthropic(&mut self, u: &Value) {
        if let Some(n) = u64_at(u, "/input_tokens") {
            self.uncached = n;
        }
        if let Some(n) = u64_at(u, "/cache_read_input_tokens") {
            self.cache_read = n;
        }
        if let Some(n) = u64_at(u, "/cache_creation_input_tokens") {
            self.cache_write = n;
        }
        if let Some(n) = u64_at(u, "/output_tokens") {
            self.output = n;
        }
        if let Some(n) = u64_at(u, "/output_tokens_details/thinking_tokens") {
            self.reasoning = Some(n);
        }
    }

    fn from_chat(u: &Value) -> Self {
        let cache_read = u64_at(u, "/prompt_tokens_details/cached_tokens").unwrap_or(0);
        // OpenAI and OpenRouter spell cache writes `cache_write_tokens`; LiteLLM keeps Anthropic's.
        let cache_write = u64_at(u, "/prompt_tokens_details/cache_write_tokens")
            .or_else(|| u64_at(u, "/cache_creation_input_tokens"))
            .unwrap_or(0);
        let prompt = u64_at(u, "/prompt_tokens").unwrap_or(0);
        Self {
            uncached: prompt
                .saturating_sub(cache_read)
                .saturating_sub(cache_write),
            cache_read,
            cache_write,
            output: u64_at(u, "/completion_tokens").unwrap_or(0),
            reasoning: u64_at(u, "/completion_tokens_details/reasoning_tokens"),
        }
    }

    fn from_responses(u: &Value) -> Self {
        let cache_read = u64_at(u, "/input_tokens_details/cached_tokens").unwrap_or(0);
        let cache_write = u64_at(u, "/input_tokens_details/cache_write_tokens").unwrap_or(0);
        let input = u64_at(u, "/input_tokens").unwrap_or(0);
        Self {
            uncached: input.saturating_sub(cache_read).saturating_sub(cache_write),
            cache_read,
            cache_write,
            output: u64_at(u, "/output_tokens").unwrap_or(0),
            reasoning: u64_at(u, "/output_tokens_details/reasoning_tokens"),
        }
    }

    fn prompt(&self) -> u64 {
        self.uncached
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }

    fn to_anthropic(self) -> Value {
        let mut m = Map::new();
        m.insert("input_tokens".into(), json!(self.uncached));
        m.insert(
            "cache_creation_input_tokens".into(),
            json!(self.cache_write),
        );
        m.insert("cache_read_input_tokens".into(), json!(self.cache_read));
        m.insert("output_tokens".into(), json!(self.output));
        if let Some(r) = self.reasoning {
            m.insert(
                "output_tokens_details".into(),
                json!({ "thinking_tokens": r }),
            );
        }
        Value::Object(m)
    }

    /// Cache writes ride `prompt_tokens_details.cache_write_tokens`, the field OpenAI and
    /// OpenRouter both use; `prompt_tokens` includes them, as it includes cache reads.
    fn to_chat(self) -> Value {
        let mut m = Map::new();
        m.insert("prompt_tokens".into(), json!(self.prompt()));
        m.insert("completion_tokens".into(), json!(self.output));
        m.insert(
            "total_tokens".into(),
            json!(self.prompt().saturating_add(self.output)),
        );
        m.insert(
            "prompt_tokens_details".into(),
            json!({ "cached_tokens": self.cache_read, "cache_write_tokens": self.cache_write }),
        );
        if let Some(r) = self.reasoning {
            m.insert(
                "completion_tokens_details".into(),
                json!({ "reasoning_tokens": r }),
            );
        }
        Value::Object(m)
    }

    /// Every detail field is required by the Responses schema, so unknown counts are 0.
    fn to_responses(self) -> Value {
        json!({
            "input_tokens": self.prompt(),
            "input_tokens_details": {
                "cached_tokens": self.cache_read,
                "cache_write_tokens": self.cache_write,
            },
            "output_tokens": self.output,
            "output_tokens_details": { "reasoning_tokens": self.reasoning.unwrap_or(0) },
            "total_tokens": self.prompt().saturating_add(self.output),
        })
    }
}

/// Longest thinking text held back waiting for its signature. Past it the block is dropped: a
/// signature covers the whole text, so a truncated block could never be sent back.
const MAX_HELD_THINKING: usize = 8 * 1024 * 1024;

/// Whether an OpenRouter `reasoning_details` entry's opaque payload is Anthropic's. Only those
/// verify at Anthropic; an OpenAI or Gemini blob presented as a signature would 400 there.
fn anthropic_format(detail: &Value) -> bool {
    detail
        .get("format")
        .and_then(Value::as_str)
        .is_none_or(|f| f.starts_with("anthropic"))
}

/// Gathers one thinking block until it is whole. OpenRouter streams a Claude block as
/// `reasoning_details` `reasoning.text` pieces sharing an `index`, then one more carrying only its
/// `signature`; a non-stream message carries both in one entry. Finished blocks come out in
/// Anthropic's shape — `thinking` (with `signature` when there was one) or `redacted_thinking`.
#[derive(Default)]
struct Gather {
    index: Option<u64>,
    text: String,
    active: bool,
    overflow: bool,
}

impl Gather {
    /// One OpenRouter `reasoning_details` entry.
    fn detail(&mut self, d: &Value, done: &mut Vec<Value>) {
        let index = d.get("index").and_then(Value::as_u64);
        match d.get("type").and_then(Value::as_str) {
            Some("reasoning.text") => {
                self.text(
                    index,
                    d.get("text").and_then(Value::as_str).unwrap_or(""),
                    done,
                );
                if let Some(sig) = non_empty_str(d, "signature") {
                    if anthropic_format(d) {
                        self.sign(sig, done);
                    } else {
                        self.close(done);
                    }
                }
            }
            Some("reasoning.summary") => {
                self.text(
                    index,
                    d.get("summary").and_then(Value::as_str).unwrap_or(""),
                    done,
                );
            }
            Some("reasoning.encrypted") => {
                self.close(done);
                if anthropic_format(d)
                    && let Some(data) = non_empty_str(d, "data")
                {
                    done.push(json!({ "type": "redacted_thinking", "data": data }));
                }
            }
            _ => {}
        }
    }

    fn text(&mut self, index: Option<u64>, text: &str, done: &mut Vec<Value>) {
        if self.active && index.is_some() && self.index.is_some() && index != self.index {
            self.close(done);
        }
        if !self.active {
            self.active = true;
            self.index = index;
        }
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

    /// The block is whole and signed.
    fn sign(&mut self, signature: &str, done: &mut Vec<Value>) {
        if !self.overflow {
            done.push(json!({
                "type": "thinking",
                "thinking": std::mem::take(&mut self.text),
                "signature": signature,
            }));
        }
        *self = Self::default();
    }

    /// The block ended without a signature.
    fn close(&mut self, done: &mut Vec<Value>) {
        if self.active && !self.overflow {
            done.push(json!({ "type": "thinking", "thinking": std::mem::take(&mut self.text) }));
        }
        *self = Self::default();
    }

    /// Forget the block being gathered: it ended, and will never be signed.
    fn discard(&mut self) {
        *self = Self::default();
    }
}

/// The thinking blocks of a Chat Completions message, in order: our own `thinking` array (what
/// this module writes for a Messages upstream), else OpenRouter's `reasoning_details`, else the
/// plain `reasoning_content` / `reasoning` text, which is never signed.
fn chat_thinking_blocks(m: &Value) -> Vec<Value> {
    if let Some(blocks) = m.get("thinking").and_then(Value::as_array) {
        return blocks
            .iter()
            .map(|b| match b.get("type").and_then(Value::as_str) {
                Some("redacted_thinking") => json!({
                    "type": "redacted_thinking",
                    "data": b.get("data").cloned().unwrap_or(json!("")),
                }),
                _ => {
                    let mut t = Map::new();
                    t.insert("type".into(), json!("thinking"));
                    let text = b
                        .get("thinking")
                        .or_else(|| b.get("text"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    t.insert("thinking".into(), json!(text));
                    if let Some(sig) = non_empty_str(b, "signature") {
                        t.insert("signature".into(), json!(sig));
                    }
                    Value::Object(t)
                }
            })
            .collect();
    }
    let mut out = Vec::new();
    if let Some(details) = m.get("reasoning_details").and_then(Value::as_array) {
        let mut gather = Gather::default();
        for d in details {
            gather.detail(d, &mut out);
        }
        gather.close(&mut out);
        return out;
    }
    if let Some(text) =
        non_empty_str(m, "reasoning_content").or_else(|| non_empty_str(m, "reasoning"))
    {
        out.push(json!({ "type": "thinking", "thinking": text }));
    }
    out
}

/// A thinking block an Anthropic client may hold: signed, or redacted. An unsigned `thinking`
/// block would be echoed back on the next turn and a turn served by Anthropic would 400 on it
/// ("thinking.signature: Field required") for the rest of the conversation — so it is dropped.
fn is_replayable_thinking(block: &Value) -> bool {
    match block.get("type").and_then(Value::as_str) {
        Some("thinking") => non_empty_str(block, "signature").is_some(),
        Some("redacted_thinking") => true,
        _ => false,
    }
}

fn anthropic_resp_to_openai(v: &Value) -> Value {
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
                    "arguments": b.get("input").map_or_else(|| "{}".to_owned(), value_string),
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
        "usage": Usage::from_anthropic(v.get("usage").unwrap_or(&Value::Null)).to_chat(),
    })
}

fn openai_resp_to_anthropic(v: &Value) -> Value {
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
    for key in ["parallel_tool_calls", "user", "safety_identifier"] {
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
    for key in ["parallel_tool_calls", "user", "safety_identifier"] {
        copy_if(&mut out, v, key);
    }
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
            out.insert("text".into(), json!({ "format": format }));
        }
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
    if let Some(d) = func.get("description") {
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
            let mut m = json!({
                "type": "image_url",
                "image_url": { "url": url },
            });
            if let Some(obj) = m.as_object_mut() {
                copy_cache_control(obj, part);
            }
            Some(m)
        }
        "input_file" => {
            let mut file = Map::new();
            for key in ["file_data", "file_id", "filename"] {
                copy_if(&mut file, part, key);
            }
            Some(json!({ "type": "file", "file": file }))
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
            let mut m = json!({
                "type": "input_image",
                "image_url": url,
            });
            if let Some(obj) = m.as_object_mut() {
                copy_cache_control(obj, part);
            }
            Some(m)
        }
        Some("file") => {
            let mut m = Map::new();
            m.insert("type".into(), json!("input_file"));
            if let Some(file) = part.get("file") {
                for key in ["file_data", "file_id", "filename"] {
                    copy_if(&mut m, file, key);
                }
            }
            Some(Value::Object(m))
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

/// A Chat Completions response for a Responses client. Items come in the order OpenAI emits
/// them: reasoning, the message, then function calls.
fn openai_resp_to_responses(v: &Value) -> Value {
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
        let func = c.get("function").unwrap_or(c);
        output.push(json!({
            "type": "function_call",
            "id": fresh_id("fc"),
            "call_id": id_or_fresh(c.get("id"), "call"),
            "name": func.get("name").and_then(Value::as_str).unwrap_or(""),
            "arguments": arguments_string(func.get("arguments")),
            "status": item_status,
        }));
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

/// A Responses object with every field the schema requires. The request's `tools` and
/// `tool_choice` are not known here; the defaults stand in for them.
fn responses_object(
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
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    })
}

fn output_text_part(text: &str) -> Value {
    json!({ "type": "output_text", "text": text, "annotations": [] })
}

fn refusal_part(refusal: &str) -> Value {
    json!({ "type": "refusal", "refusal": refusal })
}

/// A Responses `reasoning` item for a thinking block: its text as a summary, its Anthropic
/// signature (when there was one) as `encrypted_content`. `redacted_thinking` has no text to show
/// and no slot of its own on Responses, and is dropped.
fn reasoning_item(block: &Value, id: String) -> Option<Value> {
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
        item.insert("encrypted_content".into(), json!(sig));
    }
    Some(Value::Object(item))
}

/// Tool arguments as the JSON text Chat Completions and Responses carry them in.
fn arguments_string(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => "{}".to_owned(),
        Some(other) => value_string(other),
    }
}

/// A Responses response for a Chat Completions client.
fn responses_resp_to_openai(v: &Value) -> Value {
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
fn created_of(v: Option<&Value>) -> u64 {
    v.and_then(|c| c.as_u64().or_else(|| c.as_f64().map(|f| f as u64)))
        .filter(|c| *c > 0)
        .unwrap_or_else(unix_now)
}

// --- SSE --------------------------------------------------------------------

/// Event-by-event SSE translator. Incomplete events stay in `buf`; complete events are mapped
/// immediately. Does not wait for `[DONE]` before forwarding deltas.
///
/// Every pairing meets in Chat Completions chunks: a Messages or Responses upstream is read into
/// [`ChatItem`]s, and a Messages or Responses client is written from them. The middle is values,
/// not bytes, so a composed pairing (Messages → Responses) parses each event once.
pub struct SseBridge {
    client: Endpoint,
    upstream: Endpoint,
    buf: Vec<u8>,
    /// Bytes of `buf` already searched for an event end, so a long unterminated event is scanned
    /// once rather than from the start on every chunk.
    scanned: usize,
    ant_to_oai: AntToOai,
    resp_to_oai: RespToOai,
    oai_to_ant: OaiToAnt,
    oai_to_resp: OaiToResp,
    /// An error already went to the client. Nothing after it is forwarded, and flush must not
    /// invent a success close (`message_stop`, `response.completed`); a Chat client still gets its
    /// `[DONE]`.
    errored: bool,
    /// A Chat client's `[DONE]` went out.
    done: bool,
}

/// One Chat Completions stream item, the dialect every pairing meets in.
enum ChatItem {
    Chunk(Value),
    /// The upstream's error body as it came; each client shapes its own envelope from it.
    Error(Value),
    Done,
}

impl SseBridge {
    pub fn new(client: Endpoint, upstream: Endpoint) -> Self {
        Self {
            client,
            upstream,
            buf: Vec::new(),
            scanned: 0,
            ant_to_oai: AntToOai::default(),
            resp_to_oai: RespToOai::default(),
            oai_to_ant: OaiToAnt::default(),
            oai_to_resp: OaiToResp::default(),
            errored: false,
            done: false,
        }
    }

    /// Bytes held for an event whose terminating blank line has not arrived yet.
    pub fn pending_len(&self) -> usize {
        self.buf.len()
    }

    /// Feed upstream SSE bytes. Returns client-dialect SSE bytes (possibly empty).
    pub fn feed(&mut self, data: &[u8], end: bool) -> Vec<u8> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();
        while let Some(raw) = take_event(&mut self.buf, &mut self.scanned) {
            out.extend(self.map_event(&raw));
        }
        if end {
            if !self.buf.is_empty() {
                self.scanned = 0;
                let rest = std::mem::take(&mut self.buf);
                out.extend(self.map_event(&rest));
            }
            out.extend(self.flush());
        }
        out
    }

    fn map_event(&mut self, raw: &[u8]) -> Vec<u8> {
        if self.upstream == self.client {
            return raw.to_vec();
        }
        let (event, data) = parse_sse(raw);
        if data.is_empty() && event.is_empty() {
            return Vec::new();
        }
        let mut items = Vec::new();
        match self.upstream {
            Endpoint::Messages => self.ant_to_oai.event(&event, &data, &mut items),
            Endpoint::Responses => self.resp_to_oai.event(&event, &data, &mut items),
            Endpoint::ChatCompletions => chat_items(&data, &mut items),
            Endpoint::Embeddings => {}
        }
        let mut out = Vec::new();
        for item in items {
            self.deliver(item, &mut out);
        }
        out
    }

    fn deliver(&mut self, item: ChatItem, out: &mut Vec<u8>) {
        if self.errored {
            return;
        }
        match (self.client, item) {
            (client, ChatItem::Error(v)) => {
                self.errored = true;
                match client {
                    Endpoint::ChatCompletions => out.extend(sse_data(&value_string(&map_error(
                        &v,
                        Endpoint::ChatCompletions,
                    )))),
                    Endpoint::Messages => out.extend(sse_named(
                        "error",
                        &value_string(&map_error(&v, Endpoint::Messages)),
                    )),
                    Endpoint::Responses => self.oai_to_resp.error(&v, out),
                    Endpoint::Embeddings => {}
                }
            }
            (Endpoint::ChatCompletions, ChatItem::Chunk(v)) => {
                out.extend(sse_data(&value_string(&v)));
            }
            (Endpoint::ChatCompletions, ChatItem::Done) => {
                if !self.done {
                    self.done = true;
                    out.extend(sse_data("[DONE]"));
                }
            }
            (Endpoint::Messages, ChatItem::Chunk(v)) => self.oai_to_ant.chunk(&v, out),
            (Endpoint::Messages, ChatItem::Done) => self.oai_to_ant.finish(out),
            (Endpoint::Responses, ChatItem::Chunk(v)) => self.oai_to_resp.chunk(&v, out),
            (Endpoint::Responses, ChatItem::Done) => self.oai_to_resp.finish(out),
            (Endpoint::Embeddings, _) => {}
        }
    }

    fn flush(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.upstream == self.client {
            return out;
        }
        match self.client {
            Endpoint::ChatCompletions => {
                if !self.done {
                    self.done = true;
                    out.extend(sse_data("[DONE]"));
                }
            }
            Endpoint::Messages if !self.errored => self.oai_to_ant.finish(&mut out),
            Endpoint::Responses if !self.errored => self.oai_to_resp.finish(&mut out),
            // Embeddings never stream and never translate.
            Endpoint::Messages | Endpoint::Responses | Endpoint::Embeddings => {}
        }
        out
    }
}

/// A Chat Completions upstream event.
fn chat_items(data: &str, items: &mut Vec<ChatItem>) {
    if data == "[DONE]" {
        items.push(ChatItem::Done);
        return;
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return;
    };
    items.push(if looks_like_error(&v) {
        ChatItem::Error(v)
    } else {
        ChatItem::Chunk(v)
    });
}

/// The fields every Chat Completions chunk repeats.
#[derive(Default)]
struct ChunkMeta {
    id: String,
    model: String,
    created: u64,
}

impl ChunkMeta {
    fn fill(&mut self) {
        if self.id.is_empty() {
            self.id = fresh_id("chatcmpl");
        }
        if self.created == 0 {
            self.created = unix_now();
        }
    }

    fn chunk(&mut self, delta: Value, finish: Option<&str>) -> ChatItem {
        self.fill();
        ChatItem::Chunk(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        }))
    }

    /// The trailing usage-only chunk, as `stream_options.include_usage` shapes it.
    fn usage(&mut self, usage: Value) -> ChatItem {
        self.fill();
        ChatItem::Chunk(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage,
        }))
    }
}

fn reasoning_delta(text: &str) -> Value {
    json!({ "reasoning_content": text, "reasoning": text })
}

/// Anthropic Messages events → Chat Completions items.
#[derive(Default)]
struct AntToOai {
    meta: ChunkMeta,
    /// Tool calls opened so far: the next one's Chat `index`. Anthropic content blocks never
    /// interleave, so a JSON delta always belongs to the latest.
    tools: u32,
    usage: Usage,
}

impl AntToOai {
    fn event(&mut self, event: &str, data: &str, items: &mut Vec<ChatItem>) {
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
                let delta = match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        let index = self.tools;
                        self.tools = self.tools.saturating_add(1);
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
                    Some("redacted_thinking") => json!({ "thinking": [{
                        "type": "redacted_thinking",
                        "data": block.get("data").cloned().unwrap_or(json!("")),
                    }] }),
                    Some("text") => match non_empty_str(block, "text") {
                        Some(t) => json!({ "content": t }),
                        None => return,
                    },
                    Some("thinking") => match non_empty_str(block, "thinking") {
                        Some(t) => reasoning_delta(t),
                        None => return,
                    },
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
                    Some("input_json_delta") => match non_empty_str(d, "partial_json") {
                        Some(p) => json!({ "tool_calls": [{
                            "index": self.tools.saturating_sub(1),
                            "function": { "arguments": p },
                        }] }),
                        None => return,
                    },
                    Some("thinking_delta") => {
                        match non_empty_str(d, "thinking").or_else(|| non_empty_str(d, "text")) {
                            Some(t) => reasoning_delta(t),
                            None => return,
                        }
                    }
                    Some("signature_delta") => match non_empty_str(d, "signature") {
                        Some(sig) => json!({ "thinking_signature": sig }),
                        None => return,
                    },
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
                items.push(self.meta.usage(self.usage.to_chat()));
            }
            "message_stop" => items.push(ChatItem::Done),
            _ => {}
        }
    }
}

/// A function call a Responses upstream announced, and where its Chat deltas go.
struct RespCall {
    output_index: Option<u64>,
    item_id: Option<String>,
    index: u32,
    /// Argument bytes already went out, so a closing event's full `arguments` must not repeat them.
    args_sent: bool,
}

/// Responses events → Chat Completions items.
#[derive(Default)]
struct RespToOai {
    meta: ChunkMeta,
    started: bool,
    calls: Vec<RespCall>,
}

impl RespToOai {
    fn event(&mut self, event: &str, data: &str, items: &mut Vec<ChatItem>) {
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
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    self.open_call(&v, item, items);
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(at) = self.find(&v)
                    && let Some(delta) = non_empty_str(&v, "delta")
                {
                    self.args(at, delta, items);
                }
            }
            "response.function_call_arguments.done" | "response.output_item.done" => {
                let item = v.get("item");
                if item
                    .is_some_and(|i| i.get("type").and_then(Value::as_str) != Some("function_call"))
                {
                    return;
                }
                let args = item.unwrap_or(&v).get("arguments").and_then(Value::as_str);
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
        let args = item.get("arguments").and_then(Value::as_str).unwrap_or("");
        self.calls.push(RespCall {
            output_index: v.get("output_index").and_then(Value::as_u64),
            item_id: non_empty_str(item, "id").map(str::to_owned),
            index,
            args_sent: !args.is_empty(),
        });
        items.push(self.meta.chunk(
            json!({ "tool_calls": [{
                "index": index,
                "id": id_or_fresh(item.get("call_id").or_else(|| item.get("id")), "call"),
                "type": "function",
                "function": {
                    "name": item.get("name").and_then(Value::as_str).unwrap_or(""),
                    "arguments": args,
                },
            }] }),
            None,
        ));
    }

    fn args(&mut self, at: usize, args: &str, items: &mut Vec<ChatItem>) {
        let Some(call) = self.calls.get_mut(at) else {
            return;
        };
        call.args_sent = true;
        let index = call.index;
        items.push(self.meta.chunk(
            json!({ "tool_calls": [{ "index": index, "function": { "arguments": args } }] }),
            None,
        ));
    }

    /// The call an event is about, by `output_index`, else by item id.
    fn find(&self, v: &Value) -> Option<usize> {
        let output_index = v.get("output_index").and_then(Value::as_u64);
        let item_id = non_empty_str(v, "item_id")
            .or_else(|| v.get("item").and_then(|i| non_empty_str(i, "id")));
        self.calls.iter().position(|c| {
            (output_index.is_some() && c.output_index == output_index)
                || (item_id.is_some() && c.item_id.as_deref() == item_id)
        })
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

/// One call in a Chat Completions stream, gathered across its deltas.
struct ChatCall {
    /// The stream's own `index`, when it sends one.
    index: Option<u64>,
    id: String,
    name: String,
    args: String,
    opened: bool,
    closed: bool,
    /// Where the client sees it: an Anthropic block index or a Responses `output_index`.
    slot: usize,
    /// The Responses item id (`fc_…`); unused for an Anthropic client.
    item: String,
}

enum CallStep {
    /// The call is ready to show: emit its start, then all of `args` gathered so far.
    Open(usize),
    /// More arguments for an open call.
    Args(usize, String),
}

/// Chat Completions tool-call deltas, keyed the way the stream means them.
///
/// A delta names its call by `index`. Some upstreams repeat the `id` on every delta, some omit
/// `index`, and some reuse one `index` for parallel calls with different ids; all three used to
/// open a new call per delta or pour one call's arguments into another's. A call opens once its
/// name is known (or at the end), so a client never sees a nameless tool.
#[derive(Default)]
struct ToolCalls {
    calls: Vec<ChatCall>,
}

impl ToolCalls {
    fn feed(&mut self, c: &Value, steps: &mut Vec<CallStep>) {
        let index = c.get("index").and_then(Value::as_u64);
        let id = non_empty_str(c, "id");
        let at = self.locate(index, id);
        let Some(call) = self.calls.get_mut(at) else {
            return;
        };
        if call.id.is_empty()
            && let Some(id) = id
        {
            id.clone_into(&mut call.id);
        }
        let func = c.get("function").unwrap_or(c);
        if call.name.is_empty()
            && let Some(name) = non_empty_str(func, "name")
        {
            name.clone_into(&mut call.name);
        }
        let args = func.get("arguments").and_then(Value::as_str).unwrap_or("");
        call.args.push_str(args);
        if !call.opened && !call.name.is_empty() {
            Self::open(call, at, steps);
        } else if call.opened && !args.is_empty() {
            steps.push(CallStep::Args(at, args.to_owned()));
        }
    }

    /// Open every call still waiting for its name.
    fn open_rest(&mut self, steps: &mut Vec<CallStep>) {
        for (at, call) in self.calls.iter_mut().enumerate() {
            if !call.opened {
                Self::open(call, at, steps);
            }
        }
    }

    fn open(call: &mut ChatCall, at: usize, steps: &mut Vec<CallStep>) {
        call.opened = true;
        if call.id.is_empty() {
            call.id = fresh_id("call");
        }
        steps.push(CallStep::Open(at));
    }

    fn locate(&mut self, index: Option<u64>, id: Option<&str>) -> usize {
        let found = match (index, id) {
            (Some(i), _) => self.calls.iter().rposition(|c| {
                c.index == Some(i) && (id.is_none() || c.id.is_empty() || Some(c.id.as_str()) == id)
            }),
            (None, Some(id)) => self.calls.iter().rposition(|c| c.id == id),
            (None, None) => self.calls.len().checked_sub(1),
        };
        found.unwrap_or_else(|| {
            self.calls.push(ChatCall {
                index,
                id: String::new(),
                name: String::new(),
                args: String::new(),
                opened: false,
                closed: false,
                slot: 0,
                item: String::new(),
            });
            self.calls.len().saturating_sub(1)
        })
    }

    fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

/// Chat Completions items → Anthropic Messages events.
#[derive(Default)]
struct OaiToAnt {
    started: bool,
    id: String,
    model: String,
    next_block: usize,
    /// The open text block's index. Anthropic blocks are sequential: anything else closes it.
    text_block: Option<usize>,
    calls: ToolCalls,
    /// Thinking held back until its signature arrives. See [`is_replayable_thinking`].
    thinking: Gather,
    stop: Option<&'static str>,
    /// The model refused in words (`refusal`), which Anthropic reports as a stop reason.
    refused: bool,
    usage: Option<Usage>,
    finished: bool,
}

fn ant_event(out: &mut Vec<u8>, typ: &str, body: &Value) {
    out.extend(sse_named(typ, &value_string(body)));
}

impl OaiToAnt {
    fn chunk(&mut self, v: &Value, out: &mut Vec<u8>) {
        if self.finished {
            return;
        }
        if !self.started {
            if let Some(id) = non_empty_str(v, "id") {
                id.clone_into(&mut self.id);
            }
            if let Some(model) = non_empty_str(v, "model") {
                model.clone_into(&mut self.model);
            }
            self.start(out);
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            let mut blocks = Vec::new();
            // OpenRouter repeats `reasoning_details` text in `reasoning`: read one, not both.
            if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
                for d in details {
                    self.thinking.detail(d, &mut blocks);
                }
            } else if let Some(r) = non_empty_str(delta, "reasoning_content")
                .or_else(|| non_empty_str(delta, "reasoning"))
            {
                self.thinking.text(None, r, &mut blocks);
            }
            if let Some(sig) = non_empty_str(delta, "thinking_signature") {
                self.thinking.sign(sig, &mut blocks);
            }
            for b in delta
                .get("thinking")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                if b.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
                    self.thinking.close(&mut blocks);
                    blocks.push(json!({
                        "type": "redacted_thinking",
                        "data": b.get("data").cloned().unwrap_or(json!("")),
                    }));
                }
            }
            for b in blocks.iter().filter(|b| is_replayable_thinking(b)) {
                self.thinking_block(b, out);
            }
            if let Some(t) = non_empty_str(delta, "content") {
                self.text(t, out);
            }
            if let Some(r) = non_empty_str(delta, "refusal") {
                self.refused = true;
                self.text(r, out);
            }
            for c in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                self.tool(c, out);
            }
        }
        if let Some(finish) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            .filter(|f| !f.is_empty())
        {
            let native = choice
                .and_then(|c| c.get("native_finish_reason"))
                .and_then(Value::as_str)
                .and_then(anthropic_stop_reason);
            self.stop = Some(native.unwrap_or_else(|| map_stop_to_anthropic(Some(finish))));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(Usage::from_chat(u));
        }
        if self.stop.is_some() && self.usage.is_some() {
            self.finish(out);
        }
    }

    fn start(&mut self, out: &mut Vec<u8>) {
        self.started = true;
        if self.id.is_empty() {
            self.id = fresh_id("msg");
        }
        ant_event(
            out,
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 },
                }
            }),
        );
    }

    fn next_index(&mut self) -> usize {
        let idx = self.next_block;
        self.next_block = self.next_block.saturating_add(1);
        idx
    }

    fn close_text(&mut self, out: &mut Vec<u8>) {
        if let Some(idx) = self.text_block.take() {
            block_stop(out, idx);
        }
    }

    fn close_calls(&mut self, out: &mut Vec<u8>) {
        for call in &mut self.calls.calls {
            if call.opened && !call.closed {
                call.closed = true;
                block_stop(out, call.slot);
            }
        }
    }

    /// A whole thinking block, signed or redacted, emitted at once.
    fn thinking_block(&mut self, b: &Value, out: &mut Vec<u8>) {
        self.close_text(out);
        self.close_calls(out);
        let idx = self.next_index();
        if b.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
            block_start(out, idx, b);
            block_stop(out, idx);
            return;
        }
        block_start(
            out,
            idx,
            &json!({ "type": "thinking", "thinking": "", "signature": "" }),
        );
        if let Some(text) = non_empty_str(b, "thinking") {
            block_delta(
                out,
                idx,
                &json!({ "type": "thinking_delta", "thinking": text }),
            );
        }
        if let Some(sig) = non_empty_str(b, "signature") {
            block_delta(
                out,
                idx,
                &json!({ "type": "signature_delta", "signature": sig }),
            );
        }
        block_stop(out, idx);
    }

    fn text(&mut self, text: &str, out: &mut Vec<u8>) {
        // Text ends any thinking block still waiting for a signature: it never gets one now.
        self.thinking.discard();
        self.close_calls(out);
        let idx = match self.text_block {
            Some(idx) => idx,
            None => {
                let idx = self.next_index();
                self.text_block = Some(idx);
                block_start(out, idx, &json!({ "type": "text", "text": "" }));
                idx
            }
        };
        block_delta(out, idx, &json!({ "type": "text_delta", "text": text }));
    }

    fn tool(&mut self, c: &Value, out: &mut Vec<u8>) {
        self.thinking.discard();
        self.close_text(out);
        let mut steps = Vec::new();
        self.calls.feed(c, &mut steps);
        self.call_steps(steps, out);
    }

    fn call_steps(&mut self, steps: Vec<CallStep>, out: &mut Vec<u8>) {
        for step in steps {
            match step {
                CallStep::Open(at) => {
                    let idx = self.next_index();
                    let Some(call) = self.calls.calls.get_mut(at) else {
                        continue;
                    };
                    call.slot = idx;
                    block_start(
                        out,
                        idx,
                        &json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": {} }),
                    );
                    if !call.args.is_empty() {
                        block_delta(
                            out,
                            idx,
                            &json!({ "type": "input_json_delta", "partial_json": call.args }),
                        );
                    }
                }
                CallStep::Args(at, args) => {
                    if let Some(call) = self.calls.calls.get(at) {
                        block_delta(
                            out,
                            call.slot,
                            &json!({ "type": "input_json_delta", "partial_json": args }),
                        );
                    }
                }
            }
        }
    }

    fn finish(&mut self, out: &mut Vec<u8>) {
        if self.finished {
            return;
        }
        // A usage-only / `[DONE]`-only upstream stream never opened a block; the client still
        // needs `message_start` before `message_delta` / `message_stop`.
        if !self.started {
            self.start(out);
        }
        self.finished = true;
        self.thinking.discard();
        let mut steps = Vec::new();
        self.calls.open_rest(&mut steps);
        if !steps.is_empty() {
            self.close_text(out);
        }
        self.call_steps(steps, out);
        self.close_text(out);
        self.close_calls(out);
        let stop = finish_anthropic_stop(
            self.stop.unwrap_or("end_turn"),
            self.refused,
            !self.calls.is_empty(),
        );
        ant_event(
            out,
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop, "stop_sequence": Value::Null },
                "usage": self.usage.unwrap_or_default().to_anthropic(),
            }),
        );
        ant_event(out, "message_stop", &json!({ "type": "message_stop" }));
    }
}

fn block_start(out: &mut Vec<u8>, index: usize, block: &Value) {
    ant_event(
        out,
        "content_block_start",
        &json!({ "type": "content_block_start", "index": index, "content_block": block }),
    );
}

fn block_delta(out: &mut Vec<u8>, index: usize, delta: &Value) {
    ant_event(
        out,
        "content_block_delta",
        &json!({ "type": "content_block_delta", "index": index, "delta": delta }),
    );
}

fn block_stop(out: &mut Vec<u8>, index: usize) {
    ant_event(
        out,
        "content_block_stop",
        &json!({ "type": "content_block_stop", "index": index }),
    );
}

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
struct OaiToResp {
    id: String,
    model: String,
    created_at: u64,
    seq: u64,
    started: bool,
    /// Items by `output_index`; `Null` until the item is done.
    output: Vec<Value>,
    message: Option<OpenMessage>,
    reasoning: Option<OpenReasoning>,
    calls: ToolCalls,
    finish: Option<&'static str>,
    usage: Option<Usage>,
    completed: bool,
}

impl OaiToResp {
    fn emit(&mut self, out: &mut Vec<u8>, typ: &str, mut body: Value) {
        if let Some(m) = body.as_object_mut() {
            m.insert("type".into(), json!(typ));
            m.insert("sequence_number".into(), json!(self.seq));
        }
        self.seq = self.seq.saturating_add(1);
        out.extend(sse_named(typ, &value_string(&body)));
    }

    fn chunk(&mut self, v: &Value, out: &mut Vec<u8>) {
        if self.completed {
            return;
        }
        if !self.started {
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
        if !self.started {
            self.start(out);
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
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
            if let Some(sig) = non_empty_str(delta, "thinking_signature") {
                self.reasoning_signature(None, sig, out);
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
        self.started = true;
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
                    let Some((call_id, name, args)) = self
                        .calls
                        .calls
                        .get(at)
                        .map(|c| (c.id.clone(), c.name.clone(), c.args.clone()))
                    else {
                        continue;
                    };
                    let item_id = fresh_id("fc");
                    let output_index = self.add_item(
                        out,
                        json!({
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                            "status": "in_progress",
                        }),
                    );
                    if let Some(call) = self.calls.calls.get_mut(at) {
                        call.slot = output_index;
                        call.item.clone_from(&item_id);
                    }
                    if !args.is_empty() {
                        self.args_delta(out, &item_id, output_index, &args);
                    }
                }
                CallStep::Args(at, args) => {
                    let Some((item_id, output_index)) =
                        self.calls.calls.get(at).map(|c| (c.item.clone(), c.slot))
                    else {
                        continue;
                    };
                    self.args_delta(out, &item_id, output_index, &args);
                }
            }
        }
    }

    fn args_delta(&mut self, out: &mut Vec<u8>, item_id: &str, output_index: usize, args: &str) {
        self.emit(
            out,
            "response.function_call_arguments.delta",
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
                ));
            }
        }
        for (output_index, item_id, call_id, name, args) in done {
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
            let item = json!({
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "arguments": args,
                "status": status,
            });
            self.done_item(out, output_index, item);
        }
    }

    fn complete(&mut self, out: &mut Vec<u8>) {
        if self.completed {
            return;
        }
        if !self.started {
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
        self.emit(out, terminal, json!({ "response": resp }));
        self.completed = true;
    }

    fn finish(&mut self, out: &mut Vec<u8>) {
        self.complete(out);
    }

    /// An upstream error: the Responses `error` event, then `response.failed`. The event also
    /// carries the error under `error`, so an OpenAI SDK raises it as an `APIError` with the
    /// upstream's message instead of ending on a missing `response.completed`.
    fn error(&mut self, v: &Value, out: &mut Vec<u8>) {
        if self.completed {
            return;
        }
        if !self.started {
            self.start(out);
        }
        self.completed = true;
        let envelope = map_error(v, Endpoint::Responses);
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
            let code = if code.is_null() {
                json!("server_error")
            } else {
                code
            };
            m.insert("error".into(), json!({ "code": code, "message": message }));
        }
        self.emit(out, "response.failed", json!({ "response": resp }));
    }
}

/// Remove and return the first complete event. `scanned` is how much of `buf` an earlier call
/// already searched without finding an end; the search resumes there, not from byte 0.
fn take_event(buf: &mut Vec<u8>, scanned: &mut usize) -> Option<Vec<u8>> {
    // A terminator is judged at its first `\n`, which needs up to two bytes after it — so the last
    // two positions searched before may have been undecidable and are searched again.
    match find_event_end(buf, scanned.saturating_sub(2)) {
        Some(end) => {
            *scanned = 0;
            Some(buf.drain(..end).collect())
        }
        None => {
            *scanned = buf.len();
            None
        }
    }
}

/// End (exclusive) of the first `\n\n` or `\r\n\r\n`, judging only `\n`s at or after `from`.
///
/// One forward pass over newlines. The earlier two-`windows` form ran both searches to completion,
/// so a stream using only one line ending walked the whole buffer for the other on every call.
fn find_event_end(buf: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    while let Some(off) = memchr::memchr(b'\n', buf.get(at..)?) {
        let nl = at + off;
        match buf.get(nl + 1) {
            Some(b'\n') => return Some(nl + 2),
            Some(b'\r')
                if nl > 0 && buf.get(nl - 1) == Some(&b'\r') && buf.get(nl + 2) == Some(&b'\n') =>
            {
                return Some(nl + 3);
            }
            _ => {}
        }
        at = nl + 1;
    }
    None
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

    /// A current (adaptive-generation) Claude id, what most catalog walks land on.
    const OPUS: &str = "claude-opus-4-8";

    #[test]
    fn json_buffer_refuses_past_the_cap_and_keeps_what_it_has() {
        let mut t = TranslateState::new(Endpoint::ChatCompletions);
        assert!(t.push_json(&vec![b' '; MAX_TRANSLATE_BUFFER - 1]));
        assert!(t.push_json(b"{"), "exactly at the cap is allowed");
        assert!(!t.push_json(b"}"), "one byte past is refused");
        assert_eq!(
            t.json_buf.len(),
            MAX_TRANSLATE_BUFFER,
            "a refused chunk appends nothing"
        );
    }

    #[test]
    fn an_unterminated_event_is_held_and_reported_as_pending() {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        let big = format!("event: content_block_delta\ndata: {}", "x".repeat(10_000));
        for chunk in big.as_bytes().chunks(1000) {
            assert!(b.feed(chunk, false).is_empty());
        }
        assert_eq!(b.pending_len(), big.len());
    }

    /// The resumable search must find exactly what a from-scratch search finds, including a
    /// terminator split across chunks at every possible point, for both line endings.
    #[test]
    fn resumable_event_scan_matches_a_fresh_scan_at_every_split() {
        for stream in [
            &b"data: a\n\ndata: bb\n\n: c\n\n"[..],
            &b"data: a\r\n\r\ndata: bb\r\n\r\n"[..],
            &b"data: a\r\ndata: b\n\ndata: c\r\n\r\n"[..],
        ] {
            let mut fresh = stream.to_vec();
            let mut want = Vec::new();
            let mut s = 0;
            while let Some(e) = take_event(&mut fresh, &mut s) {
                want.push(e);
                s = 0;
            }
            for split in 0..=stream.len() {
                let mut buf = Vec::new();
                let mut scanned = 0;
                let mut got = Vec::new();
                for part in [&stream[..split], &stream[split..]] {
                    buf.extend_from_slice(part);
                    while let Some(e) = take_event(&mut buf, &mut scanned) {
                        got.push(e);
                    }
                }
                assert_eq!(
                    got,
                    want,
                    "split at {split} of {:?}",
                    String::from_utf8_lossy(stream)
                );
            }
        }
    }

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

    fn to_messages(body: &Value) -> Value {
        let out = request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(body).unwrap(),
            OPUS,
        );
        serde_json::from_slice(&out).unwrap()
    }

    fn count_markers(v: &Value) -> usize {
        match v {
            Value::Object(m) => {
                usize::from(m.contains_key("cache_control"))
                    + m.values().map(count_markers).sum::<usize>()
            }
            Value::Array(a) => a.iter().map(count_markers).sum(),
            _ => 0,
        }
    }

    /// A stock OpenAI SDK sends no `cache_control`. Without a marker Anthropic caches nothing, so
    /// the static prefix gets one.
    #[test]
    fn an_unmarked_request_caches_its_system_prompt() {
        let v = to_messages(&oai_req());
        assert_eq!(v["system"][0]["text"], "be brief");
        assert_eq!(v["system"][0]["cache_control"]["type"], "ephemeral");
        assert!(
            v["tools"][0].get("cache_control").is_none(),
            "one marker covers tools + system"
        );
        // Single turn: no conversation marker (a write costs 1.25x and a one-shot never reads it).
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(count_markers(&v), 1, "{v}");
    }

    #[test]
    fn without_a_system_prompt_the_last_tool_is_marked() {
        let mut req = oai_req();
        req["messages"] = json!([{"role": "user", "content": "hi"}]);
        req["tools"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "function", "function": {"name": "second"}}));
        let v = to_messages(&req);
        assert!(v.get("system").is_none());
        assert!(v["tools"][0].get("cache_control").is_none());
        assert_eq!(v["tools"][1]["cache_control"]["type"], "ephemeral");
        assert_eq!(count_markers(&v), 1, "{v}");
    }

    /// Once the request is a conversation, the whole prefix is marked so the next turn reads it.
    #[test]
    fn a_conversation_marks_the_end_of_its_last_message() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "and tomorrow?"}
        ]);
        let v = to_messages(&req);
        let last = v["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(last["role"], "user");
        let blocks = last["content"].as_array().unwrap();
        assert!(blocks[0].get("cache_control").is_none(), "{last}");
        assert_eq!(blocks.last().unwrap()["cache_control"]["type"], "ephemeral");
        assert_eq!(count_markers(&v), 2, "system + conversation: {v}");
    }

    #[test]
    fn a_string_last_message_becomes_a_marked_text_block() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"}
        ]);
        let v = to_messages(&req);
        assert_eq!(
            v["messages"][2]["content"],
            json!([{"type": "text", "text": "again", "cache_control": {"type": "ephemeral"}}])
        );
    }

    /// Anthropic rejects `cache_control` on a thinking block.
    #[test]
    fn the_conversation_marker_skips_trailing_thinking_blocks() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "hello"},
                {"type": "thinking", "thinking": "hm", "signature": "sig"}
            ]}
        ]);
        let v = to_messages(&req);
        let blocks = v["messages"][1]["content"].as_array().unwrap();
        assert!(
            blocks.iter().any(|b| b["type"] == "thinking"),
            "fixture must carry a thinking block: {blocks:?}"
        );
        for b in blocks {
            if matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking")) {
                assert!(b.get("cache_control").is_none(), "{b}");
            }
        }
        assert!(
            blocks.iter().any(|b| b.get("cache_control").is_some()),
            "a non-thinking block carries the marker: {blocks:?}"
        );
    }

    /// A client that manages its own caching keeps full control.
    #[test]
    fn a_client_marker_anywhere_disables_the_defaults() {
        let mut req = oai_req();
        req["messages"] = json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ]},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"}
        ]);
        let v = to_messages(&req);
        assert_eq!(count_markers(&v), 1, "{v}");
        assert_eq!(v["messages"][0]["content"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn responses_onto_messages_gets_the_same_defaults() {
        let body = json!({
            "model": "claude-opus-4-8",
            "instructions": "be brief",
            "input": "hi"
        });
        let out = request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&body).unwrap(),
            OPUS,
        );
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["system"][0]["cache_control"]["type"], "ephemeral", "{v}");
    }

    #[test]
    fn openai_request_maps_system_tools_and_drops_stream_options() {
        let body = serde_json::to_vec(&oai_req()).unwrap();
        let out = request(Endpoint::ChatCompletions, Endpoint::Messages, &body, OPUS);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["system"][0]["text"], "be brief");
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

    /// Chat Completions nests a tool's description under `function`; the Responses shape is flat.
    /// Every field has to come from the nested object, or the model loses what the tool is for.
    #[test]
    fn openai_tools_keep_their_description_on_responses() {
        let body = serde_json::to_vec(&oai_req()).unwrap();
        for from in [Endpoint::ChatCompletions, Endpoint::Messages] {
            let src = if from == Endpoint::Messages {
                request(Endpoint::ChatCompletions, Endpoint::Messages, &body, OPUS)
            } else {
                body.clone()
            };
            let v: Value =
                serde_json::from_slice(&request(from, Endpoint::Responses, &src, OPUS)).unwrap();
            let tool = &v["tools"][0];
            assert_eq!(tool["name"], "get_weather", "{from:?}: {v}");
            assert_eq!(tool["description"], "weather", "{from:?}: {v}");
            assert_eq!(tool["parameters"]["properties"]["city"]["type"], "string");
        }
    }

    #[test]
    fn openai_request_defaults_max_tokens() {
        let body = br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            body,
            OPUS,
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
            OPUS,
        );
        let anth: Value = serde_json::from_slice(&anth_bytes).unwrap();
        assert_eq!(anth["messages"][1]["role"], "assistant");
        assert_eq!(anth["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(anth["messages"][1]["content"][0]["input"]["city"], "SF");
        assert_eq!(anth["messages"][2]["role"], "user");
        assert_eq!(anth["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(anth["messages"][2]["content"][0]["tool_use_id"], "call_1");

        let back_bytes = request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &anth_bytes,
            OPUS,
        );
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
            OPUS,
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
            request(
                Endpoint::ChatCompletions,
                Endpoint::ChatCompletions,
                body,
                OPUS
            ),
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
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["model"], "gpt-4o");
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
            OPUS,
        ))
        .unwrap();
        // Opus 4.8 rejects `budget_tokens`: effort goes to `output_config`.
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["output_config"]["effort"], "high");
        assert!(v["thinking"].get("budget_tokens").is_none(), "{v}");
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
            OPUS,
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
            OPUS,
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
            OPUS,
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
    fn http_image_urls_become_anthropic_url_sources() {
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
            OPUS,
        ))
        .unwrap();
        let content = &v["messages"][0]["content"];
        assert_eq!(content[0]["text"], "see");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "url");
        assert_eq!(content[1]["source"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn anthropic_url_sources_become_openai_image_urls() {
        let msg = json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
                    {"type": "text", "text": "see"}
                ]
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&msg).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &v["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(content[0]["image_url"]["url"], "https://example.com/x.png");
        assert_eq!(content[1]["text"], "see");
    }

    /// A `url` source that is not http(s) (`file:`, `ftp:`, garbage) is not forwarded as an
    /// OpenAI `image_url` the upstream would try, and fail, to fetch.
    #[test]
    fn non_http_url_sources_are_not_forwarded() {
        let msg = json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "file:///etc/passwd"}},
                    {"type": "text", "text": "see"}
                ]
            }]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&msg).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["content"], "see");
    }

    #[test]
    fn http_image_urls_pass_between_responses_and_chat() {
        let resp = json!({
            "model": "m",
            "store": false,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "see"},
                    {"type": "input_image", "image_url": "https://example.com/x.png"}
                ]
            }]
        });
        let chat: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&resp).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &chat["messages"][0]["content"];
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "https://example.com/x.png");

        let back: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Responses,
            &serde_json::to_vec(&chat).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &back["input"][0]["content"];
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "https://example.com/x.png");

        // And on through Messages: Responses → Messages is composed through Chat Completions.
        let msgs: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&resp).unwrap(),
            OPUS,
        ))
        .unwrap();
        let content = &msgs["messages"][0]["content"];
        assert_eq!(content[1]["source"]["type"], "url");
        assert_eq!(content[1]["source"]["url"], "https://example.com/x.png");
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

    /// Unsigned reasoning never reaches a Messages client: echoed back, it would 400 the next
    /// turn Anthropic serves. Signed reasoning is covered in `translate_response_tests.rs`.
    #[test]
    fn openai_reasoning_sse_becomes_anthropic_thinking() {
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let src = concat!(
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"reasoning_content\":\"plan\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let out = String::from_utf8(b.feed(src.as_bytes(), true)).unwrap();
        assert!(!out.contains("\"type\":\"thinking\""), "{out}");
        assert!(!out.contains("plan"), "{out}");
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
            OPUS,
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
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "hi");
        assert_eq!(v["max_tokens"], 16);
        assert_eq!(v["thinking"]["type"], "adaptive");
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

    // ---- Model-aware reasoning and sampling ----

    #[test]
    fn claude_generation_parses_every_candidate_spelling() {
        use ClaudeGen::*;
        // (id, reasoning, forced tool_choice accepted)
        for (id, reasoning, forced) in [
            ("claude-opus-4-8", Adaptive, true),
            ("anthropic/claude-opus-4.8", Adaptive, true),
            ("global.anthropic.claude-opus-4-7-v1:0", Adaptive, true),
            ("claude-opus-5", Adaptive, true),
            ("claude-opus-5-5", Adaptive, false),
            ("anthropic/claude-opus-5.5", Adaptive, false),
            ("claude-sonnet-5", Adaptive, true),
            ("claude-sonnet-5-5", Adaptive, false),
            ("claude-fable-5", Adaptive, true),
            ("claude-fable-5-1", Adaptive, false),
            ("claude-mythos-5-1", Adaptive, false),
            ("claude-opus-4-6", Adaptive46, true),
            ("claude-sonnet-4-6", Adaptive46, true),
            ("claude-haiku-4-5", Budget, true),
            ("anthropic.claude-haiku-4-5-20251001-v1:0", Budget, true),
            ("claude-sonnet-4-20250514", Budget, true),
            ("claude-opus-4-1", Budget, true),
            ("claude-3-haiku", Budget, true),
            ("claude-3-5-sonnet-latest", Budget, true),
            ("claude-nova", Adaptive, false),
            ("kimi-k3", Budget, true),
            ("", Budget, true),
        ] {
            assert_eq!(
                ClaudeModel::of(id),
                ClaudeModel {
                    reasoning,
                    forced_tool_choice: forced
                },
                "{id}"
            );
        }
    }

    fn chat(extra: Value) -> Value {
        let mut req = json!({
            "model": "m",
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": "hi"}]
        });
        if let (Some(r), Some(e)) = (req.as_object_mut(), extra.as_object()) {
            for (k, v) in e {
                r.insert(k.clone(), v.clone());
            }
        }
        req
    }

    fn to_claude(body: &Value, model: &str) -> Value {
        serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &serde_json::to_vec(body).unwrap(),
            model,
        ))
        .unwrap()
    }

    /// Current Claude models 400 on `budget_tokens` and on `temperature` / `top_p`.
    #[test]
    fn current_claude_gets_adaptive_thinking_and_no_sampling() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "xhigh", "temperature": 0.2, "top_p": 0.9})),
            "claude-opus-4-8",
        );
        assert_eq!(v["thinking"], json!({"type": "adaptive"}));
        assert_eq!(v["output_config"]["effort"], "xhigh");
        assert!(v.get("temperature").is_none(), "{v}");
        assert!(v.get("top_p").is_none(), "{v}");
    }

    /// Several current models reject `thinking: disabled`; "no reasoning" is the lowest effort.
    #[test]
    fn reasoning_none_on_current_claude_omits_thinking() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "none"})),
            "claude-opus-5-5",
        );
        assert!(v.get("thinking").is_none(), "{v}");
        assert_eq!(v["output_config"]["effort"], "low");
    }

    #[test]
    fn claude_4_6_has_no_xhigh() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "xhigh"})),
            "claude-sonnet-4-6",
        );
        assert_eq!(v["output_config"]["effort"], "max");
    }

    #[test]
    fn older_claude_gets_a_budget_below_max_tokens() {
        let v = to_claude(
            &chat(json!({"reasoning_effort": "high"})),
            "claude-haiku-4-5",
        );
        assert_eq!(v["thinking"]["type"], "enabled");
        assert_eq!(
            v["thinking"]["budget_tokens"], 8000,
            "half of max_tokens 16000: {v}"
        );
        assert!(
            v.get("output_config").is_none(),
            "Haiku 4.5 rejects effort: {v}"
        );

        let mut small = chat(json!({"reasoning_effort": "high"}));
        small["max_tokens"] = json!(1000);
        let v = to_claude(&small, "claude-haiku-4-5");
        assert!(
            v.get("thinking").is_none(),
            "no room to think under 1024: {v}"
        );
    }

    /// Sampling is fine on older models, except alongside thinking.
    #[test]
    fn older_claude_keeps_sampling_unless_thinking() {
        let v = to_claude(&chat(json!({"temperature": 0.2})), "claude-haiku-4-5");
        assert_eq!(v["temperature"], 0.2);
        let v = to_claude(
            &chat(json!({"temperature": 0.2, "reasoning_effort": "low"})),
            "claude-haiku-4-5",
        );
        assert!(v.get("temperature").is_none(), "{v}");
    }

    // ---- Fields that used to vanish ----

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        })
    }

    #[test]
    fn json_schema_response_format_becomes_output_config_format() {
        let v = to_claude(
            &chat(json!({"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "strict": true, "schema": schema()}
            }})),
            OPUS,
        );
        assert_eq!(
            v["output_config"]["format"],
            json!({"type": "json_schema", "schema": schema()})
        );
        assert!(v.get("response_format").is_none(), "{v}");
    }

    #[test]
    fn format_and_effort_share_one_output_config() {
        let v = to_claude(
            &chat(json!({
                "reasoning_effort": "low",
                "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": schema()}}
            })),
            OPUS,
        );
        assert_eq!(v["output_config"]["effort"], "low");
        assert_eq!(v["output_config"]["format"]["type"], "json_schema");
    }

    #[test]
    fn output_config_format_becomes_response_format() {
        let anth = json!({
            "model": "claude-opus-4-8",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": {"type": "json_schema", "schema": schema()}, "effort": "medium"}
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&anth).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(v["response_format"]["type"], "json_schema");
        assert_eq!(v["response_format"]["json_schema"]["schema"], schema());
        assert_eq!(v["reasoning_effort"], "medium");
    }

    #[test]
    fn parallel_tool_calls_false_disables_parallel_tool_use_both_ways() {
        let v = to_claude(
            &chat(json!({
                "parallel_tool_calls": false,
                "tools": [{"type": "function", "function": {"name": "f"}}]
            })),
            OPUS,
        );
        assert_eq!(
            v["tool_choice"],
            json!({"type": "auto", "disable_parallel_tool_use": true})
        );
        let back: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&v).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(back["parallel_tool_calls"], false);
    }

    #[test]
    fn user_becomes_metadata_user_id_both_ways() {
        let v = to_claude(&chat(json!({"user": "u-123"})), OPUS);
        assert_eq!(v["metadata"], json!({"user_id": "u-123"}));
        let back: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&v).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(back["user"], "u-123");
    }

    #[test]
    fn an_inline_pdf_becomes_a_document_and_back() {
        let v = to_claude(
            &chat(json!({"messages": [{"role": "user", "content": [
                {"type": "file", "file": {"filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}},
                {"type": "text", "text": "summarize"}
            ]}]})),
            OPUS,
        );
        let doc = &v["messages"][0]["content"][0];
        assert_eq!(doc["type"], "document");
        assert_eq!(
            doc["source"],
            json!({"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="})
        );
        assert_eq!(doc["title"], "q3.pdf");
        let back: Value = serde_json::from_slice(&request(
            Endpoint::Messages,
            Endpoint::ChatCompletions,
            &serde_json::to_vec(&v).unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(
            back["messages"][0]["content"][0],
            json!({"type": "file", "file": {"filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}})
        );
    }

    /// No Messages equivalent: forwarded so the provider names the problem. Dropping them would
    /// answer a question about input the model never saw.
    #[test]
    fn unmappable_input_is_forwarded_for_the_provider_to_reject() {
        let audio =
            json!({"type": "input_audio", "input_audio": {"data": "UklG", "format": "wav"}});
        let by_id = json!({"type": "file", "file": {"file_id": "file-abc"}});
        let v = to_claude(
            &chat(
                json!({"n": 2, "logprobs": true, "messages": [{"role": "user", "content": [
                    audio.clone(), by_id.clone(), {"type": "text", "text": "what is this"}
                ]}]}),
            ),
            OPUS,
        );
        assert_eq!(v["n"], 2);
        assert_eq!(v["logprobs"], true);
        assert_eq!(v["messages"][0]["content"][0], audio);
        assert_eq!(v["messages"][0]["content"][1], by_id);
    }

    #[test]
    fn harmless_defaults_are_not_forwarded() {
        let v = to_claude(&chat(json!({"n": 1, "logprobs": false, "seed": 7})), OPUS);
        for key in ["n", "logprobs", "seed"] {
            assert!(v.get(key).is_none(), "{key}: {v}");
        }
    }

    #[test]
    fn responses_text_format_and_input_file_reach_messages() {
        let body = json!({
            "model": "claude-opus-4-8",
            "store": false,
            "text": {"format": {"type": "json_schema", "name": "a", "strict": true, "schema": schema()}},
            "input": [{"role": "user", "content": [
                {"type": "input_file", "filename": "q3.pdf", "file_data": "data:application/pdf;base64,JVBERi0="},
                {"type": "input_text", "text": "summarize"}
            ]}]
        });
        let v: Value = serde_json::from_slice(&request(
            Endpoint::Responses,
            Endpoint::Messages,
            &serde_json::to_vec(&body).unwrap(),
            OPUS,
        ))
        .unwrap();
        assert_eq!(v["output_config"]["format"]["schema"], schema());
        assert_eq!(v["messages"][0]["content"][0]["type"], "document");
    }

    #[test]
    fn chat_response_format_becomes_responses_text_format() {
        let v: Value = serde_json::from_slice(&request(
            Endpoint::ChatCompletions,
            Endpoint::Responses,
            &serde_json::to_vec(&chat(json!({"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a", "strict": true, "schema": schema()}
            }})))
            .unwrap(),
            "gpt-5",
        ))
        .unwrap();
        assert_eq!(
            v["text"]["format"],
            json!({"type": "json_schema", "name": "a", "strict": true, "schema": schema()})
        );
    }

    // ---- Forced tool use on models that reject it ----

    fn tools_req(choice: Value) -> Value {
        chat(json!({
            "tool_choice": choice,
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}},
                {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
            ]
        }))
    }

    #[test]
    fn required_is_forced_where_the_model_allows_it() {
        let v = to_claude(&tools_req(json!("required")), "claude-opus-4-8");
        assert_eq!(v["tool_choice"], json!({"type": "any"}));
        assert!(
            v["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m["role"] != "system"),
            "{v}"
        );
    }

    /// Opus 5.5 / Sonnet 5.5 / Fable 5.1 400 on `any` and `tool`: `auto` plus a closing
    /// instruction, which is Anthropic's documented migration for these models.
    #[test]
    fn required_becomes_auto_plus_an_instruction_where_forcing_is_rejected() {
        for model in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"] {
            let v = to_claude(&tools_req(json!("required")), model);
            assert_eq!(v["tool_choice"], json!({"type": "auto"}), "{model}");
            let last = v["messages"].as_array().unwrap().last().unwrap().clone();
            assert_eq!(last["role"], "system", "{model}: {v}");
            assert!(last["content"].as_str().unwrap().contains("provided tools"));
        }
    }

    #[test]
    fn a_named_tool_choice_names_the_tool_in_the_instruction() {
        let v = to_claude(
            &tools_req(json!({"type": "function", "function": {"name": "get_time"}})),
            "claude-opus-5-5",
        );
        assert_eq!(v["tool_choice"], json!({"type": "auto"}));
        let last = v["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(last["role"], "system");
        assert!(
            last["content"].as_str().unwrap().contains("`get_time`"),
            "{last}"
        );
    }

    #[test]
    fn unforced_choice_keeps_disable_parallel_tool_use() {
        let mut req = tools_req(json!("required"));
        req["parallel_tool_calls"] = json!(false);
        let v = to_claude(&req, "claude-opus-5-5");
        assert_eq!(
            v["tool_choice"],
            json!({"type": "auto", "disable_parallel_tool_use": true})
        );
    }

    /// The instruction goes after the cache marker, so it never invalidates the cached prefix.
    #[test]
    fn the_instruction_follows_the_conversation_cache_marker() {
        let mut req = tools_req(json!("required"));
        req["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "weather?"}
        ]);
        let v = to_claude(&req, "claude-opus-5-5");
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.last().unwrap()["role"], "system");
        assert!(msgs.last().unwrap().get("cache_control").is_none());
        let user = &msgs[msgs.len() - 2];
        assert_eq!(
            user["content"][0]["cache_control"]["type"], "ephemeral",
            "{v}"
        );
    }

    #[test]
    fn auto_and_none_are_untouched_everywhere() {
        for choice in ["auto", "none"] {
            let v = to_claude(&tools_req(json!(choice)), "claude-opus-5-5");
            assert_eq!(v["tool_choice"]["type"], choice);
            assert!(
                v["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|m| m["role"] != "system")
            );
        }
    }
}

#[cfg(test)]
#[path = "translate_response_tests.rs"]
mod response_tests;
