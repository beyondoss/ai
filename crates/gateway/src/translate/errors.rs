//! Error envelopes: an upstream's error body, reshaped into the client's wire.

use super::*;

pub(super) fn looks_like_error(v: &Value) -> bool {
    // A Responses object is an error only once it failed; its `error` is `null` otherwise.
    if v.get("object").and_then(Value::as_str) == Some("response") {
        return v.get("status").and_then(Value::as_str) == Some("failed")
            && v.get("error").is_some_and(Value::is_object);
    }
    // OpenRouter reports a provider that failed mid-generation as a chunk (or body) with `choices`
    // *and* a top-level `error` object, and `finish_reason: "error"`. Read as a success, a Messages
    // or Responses client got a clean `end_turn` / `completed` on a half-written answer.
    if v.pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        == Some("error")
    {
        return true;
    }
    let Some(err) = v.get("error").filter(|e| !e.is_null()) else {
        return false;
    };
    if err.is_object() {
        return true;
    }
    // Success bodies never carry a string `error` next to `choices` / `content` / `output`.
    if v.get("choices").is_some() || v.get("content").is_some() || v.get("output").is_some() {
        return false;
    }
    v.get("type").and_then(Value::as_str) != Some("message")
}

/// Whether an error body is already the client API's own envelope: `{"error": {"message", …}}`
/// on the OpenAI endpoints, `{"type": "error", "error": {"type", "message"}}` on Messages.
pub(super) fn in_client_envelope(v: &Value, client: Endpoint) -> bool {
    let err = v.get("error");
    let has = |k: &str| err.and_then(|e| e.get(k)).is_some_and(Value::is_string);
    match client {
        Endpoint::Messages => {
            v.get("type").and_then(Value::as_str) == Some("error") && has("type") && has("message")
        }
        Endpoint::ChatCompletions | Endpoint::Responses | Endpoint::Embeddings => has("message"),
    }
}

/// The error type an HTTP status means when the body names none (xAI's 400 has no type,
/// and `api_error` told a Messages client its invalid image was a server fault). Anthropic's
/// closed set on Messages; on the OpenAI endpoints the gateway's own envelope's words.
pub(super) fn status_error_type(status: u16, client: Endpoint) -> &'static str {
    match (client, status) {
        (Endpoint::Messages, 401) => "authentication_error",
        (Endpoint::Messages, 402) => "billing_error",
        (Endpoint::Messages, 403) => "permission_error",
        (Endpoint::Messages, 404) => "not_found_error",
        (Endpoint::Messages, 413) => "request_too_large",
        (Endpoint::Messages, 504) => "timeout_error",
        (Endpoint::Messages, 529) => "overloaded_error",
        (_, 429) => "rate_limit_error",
        (_, 400..=499) => "invalid_request_error",
        _ => "api_error",
    }
}

/// Everything an upstream error says, whichever dialect said it.
struct ErrorInfo {
    /// The body's own error type, if it named one.
    typ: Option<String>,
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
    let mut message = plain_error_message(v).unwrap_or_else(|| {
        if v.pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            == Some("error")
        {
            "The upstream ended the response with an error.".to_owned()
        } else {
            clip(&value_string(v))
        }
    });
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
            // JSON that is not an object names no type or message, so it is quoted whole, as text
            // that is not JSON is.
            Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
                Ok(parsed) => {
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
                Err(_) => Some(clip(raw)),
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
        typ,
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

pub(super) fn clip(s: &str) -> String {
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
pub(super) fn anthropic_error_type(typ: &str, code: Option<&Value>) -> String {
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

/// Anthropic's words for a prompt past the context window; Claude Code compacts on them.
const ANTHROPIC_OVERFLOW: &str = "prompt is too long";

/// OpenAI's code for the same; Codex and the Agents SDK compact on it.
pub(super) const OPENAI_OVERFLOW: &str = "context_length_exceeded";

/// `v` in the client's error envelope. `status` is the response's HTTP status (200 for an error
/// carried inside a 2xx stream or body), which types an error whose body names no type.
pub(super) fn map_error(v: &Value, client: Endpoint, status: u16) -> Value {
    let mut info = error_info(v);
    let typ = info
        .typ
        .take()
        .unwrap_or_else(|| status_error_type(status, client).to_owned());
    // A context overflow says so in the words the client's harness acts on.
    let says_overflow = info.message.contains(ANTHROPIC_OVERFLOW);
    let coded_overflow = info.code.as_ref().and_then(Value::as_str) == Some(OPENAI_OVERFLOW);
    match client {
        Endpoint::Messages => {
            if coded_overflow && !says_overflow {
                info.message = format!("{ANTHROPIC_OVERFLOW}: {}", info.message);
            }
            let mut err = Map::new();
            err.insert(
                "type".into(),
                json!(anthropic_error_type(&typ, info.code.as_ref())),
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
            if says_overflow && info.code.is_none() {
                info.code = Some(json!(OPENAI_OVERFLOW));
            }
            let mut err = Map::new();
            err.insert("message".into(), json!(info.message));
            err.insert("type".into(), json!(typ));
            err.insert("param".into(), openai_error_field(info.param));
            err.insert("code".into(), openai_error_field(info.code));
            if let Some(meta) = info.metadata {
                err.insert("metadata".into(), meta);
            }
            json!({ "error": err })
        }
    }
}
