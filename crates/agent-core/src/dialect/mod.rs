//! Wire dialects.
//!
//! The gateway relays bytes verbatim, so the harness owns the provider wire. Each [`Dialect`] knows
//! how to (a) build a streaming request body from a [`ModelRequest`] and (b) decode that provider's
//! SSE stream into the dialect-agnostic [`StreamEvent`] sequence the loop consumes.
//!
//! Selection is by model id: Claude speaks Anthropic (`/v1/messages`); everything else speaks the
//! OpenAI wire (`/v1/chat/completions`), which is the lingua franca for the gateway's other providers.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::message::StreamEvent;
use crate::transport::ModelRequest;

pub mod anthropic;
pub mod openai;

/// Which provider wire a model speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Anthropic,
    OpenAi,
}

impl Dialect {
    /// Pick the dialect for a model id. Claude → Anthropic; everything else → OpenAI wire.
    pub fn for_model(model: &str) -> Self {
        if model.starts_with("claude") || model.contains("anthropic") {
            Dialect::Anthropic
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
        }
    }

    /// Build the JSON body for a streaming completion request.
    pub fn build_body(&self, req: &ModelRequest) -> Value {
        match self {
            Dialect::Anthropic => anthropic::build_body(req),
            Dialect::OpenAi => openai::build_body(req),
        }
    }

    /// A fresh streaming decoder for this dialect.
    pub fn decoder(&self) -> Box<dyn StreamDecoder> {
        match self {
            Dialect::Anthropic => Box::<anthropic::Decoder>::default(),
            Dialect::OpenAi => Box::<openai::Decoder>::default(),
        }
    }
}

/// A stateful decoder turning a provider's SSE `data:` payloads into [`StreamEvent`]s. One per
/// request (it carries cross-event state: token counts, the open content block, the stop reason).
pub trait StreamDecoder: Send {
    /// Feed one parsed SSE `data:` JSON object; return any events it produced.
    fn push(&mut self, data: &Value) -> Vec<StreamEvent>;

    /// Called once at end-of-stream. Flushes any held terminal event (OpenAI defers `MessageStop`
    /// until the stream closes so it can land after the trailing usage chunk).
    fn finish(&mut self) -> Vec<StreamEvent> {
        Vec::new()
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
    out.extend(decoder.finish());
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
    Ok(decoder.push(&v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_selection_by_model() {
        assert_eq!(Dialect::for_model("claude-opus-4-8"), Dialect::Anthropic);
        assert_eq!(Dialect::for_model("gpt-4o"), Dialect::OpenAi);
        assert_eq!(Dialect::for_model("llama-3.1-70b"), Dialect::OpenAi);
        assert_eq!(Dialect::for_model("anthropic/claude"), Dialect::Anthropic);
    }
}
