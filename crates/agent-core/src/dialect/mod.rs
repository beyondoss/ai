//! Wire dialects.
//!
//! The gateway relays bytes verbatim, so the harness owns the provider wire. Each [`Dialect`] knows
//! how to (a) build a streaming request body from a [`ModelRequest`] and (b) decode that provider's
//! SSE stream into the dialect-agnostic [`StreamEvent`] sequence the loop consumes.
//!
//! Selection is by model id: Claude speaks Anthropic (`/v1/messages`); every native OpenAI id (see
//! [`crate::models::ApiKind`]) speaks the Responses API (`/v1/responses`); everything else (every
//! third-party OpenAI-compatible provider) speaks Chat Completions (`/v1/chat/completions`), the
//! lingua franca for the gateway's other providers.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::message::StreamEvent;
use crate::models::ApiKind;
use crate::transport::ModelRequest;

pub mod anthropic;
pub mod openai;
pub mod openai_responses;

/// Which provider wire a model speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Anthropic,
    OpenAi,
    OpenAiResponses,
}

impl Dialect {
    /// Pick the dialect for a model id. Claude → Anthropic; a native OpenAI id (per the capability
    /// table's [`ApiKind`]) → the Responses API; everything else → Chat Completions.
    pub fn for_model(model: &str) -> Self {
        if model.starts_with("claude") || model.contains("anthropic") {
            Dialect::Anthropic
        } else if crate::models::capabilities(model).api == ApiKind::Responses {
            Dialect::OpenAiResponses
        } else {
            Dialect::OpenAi
        }
    }

    /// The gateway path this dialect POSTs to. Bare `/v1*` so the gateway's default-provider routing
    /// (OpenAI on `/v1`, Anthropic on `/v1/messages`) applies; a `bai_v1` key picks the real provider.
    pub fn endpoint_path(&self) -> &'static str {
        match self {
            Dialect::Anthropic => "/v1/messages",
            Dialect::OpenAi => "/v1/chat/completions",
            Dialect::OpenAiResponses => "/v1/responses",
        }
    }

    /// Build the JSON body for a streaming completion request.
    pub fn build_body(&self, req: &ModelRequest) -> Value {
        match self {
            Dialect::Anthropic => anthropic::build_body(req),
            Dialect::OpenAi => openai::build_body(req),
            Dialect::OpenAiResponses => openai_responses::build_body(req),
        }
    }

    /// A fresh streaming decoder for this dialect.
    pub fn decoder(&self) -> Box<dyn StreamDecoder> {
        match self {
            Dialect::Anthropic => Box::<anthropic::Decoder>::default(),
            Dialect::OpenAi => Box::<openai::Decoder>::default(),
            Dialect::OpenAiResponses => Box::<openai_responses::Decoder>::default(),
        }
    }
}

/// A stateful decoder turning a provider's SSE `data:` payloads into [`StreamEvent`]s. One per
/// request (it carries cross-event state: token counts, the open content block, the stop reason).
pub trait StreamDecoder: Send {
    /// Feed one parsed SSE `data:` JSON object; return any events it produced.
    fn push(&mut self, data: &Value) -> Vec<StreamEvent>;

    /// Called once at end-of-stream. Flushes any held terminal event (OpenAI defers `MessageStop`
    /// until the stream closes so it can land after the trailing usage chunk), and is the decoder's
    /// chance to reject a stream that ended *before* its terminal marker — a truncated stream
    /// otherwise completes as a clean `EndTurn`/0-usage turn, indistinguishable from success.
    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        Ok(Vec::new())
    }
}

/// Decode a complete SSE body into events. Splits on `data:` lines, skips comments/`event:` lines
/// and the OpenAI `[DONE]` sentinel, and flushes the decoder at the end. The streaming HTTP client
/// reuses the same decoder incrementally; this is the buffered form used by tests.
pub fn decode_sse(decoder: &mut dyn StreamDecoder, raw: &str) -> Result<Vec<StreamEvent>> {
    let mut out = Vec::new();
    for line in raw.lines() {
        out.extend(push_sse_line(decoder, line)?);
    }
    out.extend(decoder.finish()?);
    Ok(out)
}

/// Feed a single SSE line to the decoder, returning any events it produced. Non-`data:` lines
/// (comments, `event:`, blanks) and the OpenAI `[DONE]` sentinel produce nothing. This is the
/// incremental entry point the streaming HTTP client drives line-by-line off the socket; the caller
/// is responsible for calling [`StreamDecoder::finish`] once the stream closes.
pub fn push_sse_line(decoder: &mut dyn StreamDecoder, line: &str) -> Result<Vec<StreamEvent>> {
    let line = line.trim_end_matches('\r');
    let Some(payload) = line.strip_prefix("data:") else {
        return Ok(Vec::new());
    };
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return Ok(Vec::new());
    }
    let v: Value = serde_json::from_str(payload)
        .map_err(|e| Error::Transport(format!("malformed SSE json: {e}")))?;
    // A provider can report a failure *in-band* mid-stream — Anthropic as `{"type":"error",…}`
    // (preceded by an `event: error` line we don't see here), OpenAI as a bare `{"error":{…}}` chunk.
    // Surface it as a transport error; otherwise it falls through every decoder's catch-all arm and
    // the turn ends silently as a successful-looking `EndTurn` with no content and no usage.
    if let Some(msg) = sse_error(&v) {
        return Err(Error::Transport(format!("provider stream error: {msg}")));
    }
    Ok(decoder.push(&v))
}

/// Extract a provider error message from an SSE `data:` payload, if it is one. Handles three shapes:
/// Anthropic's `{"type":"error","error":{"message":…}}`, OpenAI Chat Completions' bare
/// `{"error":{"message":…}}`, and the OpenAI Responses API's flat `{"type":"error","code":…,
/// "message":…}` (no nested `error` object — a genuinely different shape from the other two, since
/// Responses streams a top-level `error` *event* rather than an in-band error field). Returns `None`
/// for ordinary stream events.
fn sse_error(v: &Value) -> Option<String> {
    let is_typed_error = v.get("type").and_then(Value::as_str) == Some("error");
    if is_typed_error {
        // Anthropic's nested shape, if present; otherwise fall through to the Responses flat shape.
        if let Some(err) = v.get("error").filter(|e| e.is_object()) {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown provider error");
            let kind = err.get("type").and_then(Value::as_str);
            return Some(match kind {
                Some(kind) => format!("{kind}: {msg}"),
                None => msg.to_string(),
            });
        }
        let msg = v
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown provider error");
        let code = v.get("code").and_then(Value::as_str);
        return Some(match code {
            Some(code) => format!("{code}: {msg}"),
            None => msg.to_string(),
        });
    }
    // OpenAI Chat Completions' bare shape: an `error` object with no `type:"error"` envelope.
    let err = v.get("error")?;
    if !err.is_object() {
        return None;
    }
    let msg = err
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| err.as_str())
        .unwrap_or("unknown provider error");
    let kind = err.get("type").and_then(Value::as_str);
    Some(match kind {
        Some(kind) => format!("{kind}: {msg}"),
        None => msg.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_selection_by_model() {
        assert_eq!(Dialect::for_model("claude-opus-4-8"), Dialect::Anthropic);
        // Native OpenAI ids speak the Responses API now (see `models::ApiKind`).
        assert_eq!(Dialect::for_model("gpt-4o"), Dialect::OpenAiResponses);
        assert_eq!(Dialect::for_model("o3-mini"), Dialect::OpenAiResponses);
        // Third-party OpenAI-compatible ids stay on Chat Completions.
        assert_eq!(Dialect::for_model("llama-3.1-70b"), Dialect::OpenAi);
        assert_eq!(Dialect::for_model("anthropic/claude"), Dialect::Anthropic);
    }

    #[test]
    fn endpoint_paths_by_dialect() {
        assert_eq!(Dialect::Anthropic.endpoint_path(), "/v1/messages");
        assert_eq!(Dialect::OpenAi.endpoint_path(), "/v1/chat/completions");
        assert_eq!(Dialect::OpenAiResponses.endpoint_path(), "/v1/responses");
    }

    #[test]
    fn sse_error_recognizes_all_three_shapes() {
        use serde_json::json;
        // Anthropic nested shape.
        assert_eq!(
            sse_error(
                &json!({"type":"error","error":{"type":"overloaded_error","message":"busy"}})
            ),
            Some("overloaded_error: busy".to_string())
        );
        // OpenAI Chat Completions bare shape.
        assert_eq!(
            sse_error(&json!({"error":{"type":"server_error","message":"boom"}})),
            Some("server_error: boom".to_string())
        );
        // OpenAI Responses flat shape: no nested `error` object.
        assert_eq!(
            sse_error(&json!({"type":"error","code":"rate_limit_exceeded","message":"slow down"})),
            Some("rate_limit_exceeded: slow down".to_string())
        );
        // Ordinary events are not errors.
        assert_eq!(sse_error(&json!({"type":"message_start"})), None);
    }
}
