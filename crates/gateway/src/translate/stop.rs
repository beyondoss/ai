//! Stop reasons and finish states between the three wires.

use super::*;

/// Anthropic stop reason → Chat Completions `finish_reason`.
pub(super) fn map_stop_to_openai(reason: Option<&str>) -> &'static str {
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
pub(super) fn map_stop_to_anthropic(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") => "max_tokens",
        Some("tool_calls" | "function_call") => "tool_use",
        Some("content_filter") => "refusal",
        _ => "end_turn",
    }
}

/// A `finish_reason` that is already an Anthropic stop reason — OpenRouter's
/// `native_finish_reason` on a Claude model — survives exactly (`stop_sequence`, `pause_turn`).
pub(super) fn anthropic_stop_reason(reason: &str) -> Option<&'static str> {
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
pub(super) fn anthropic_stop_of(
    choice: Option<&Value>,
    refused: bool,
    called: bool,
) -> &'static str {
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

pub(super) fn finish_anthropic_stop(
    stop: &'static str,
    refused: bool,
    called: bool,
) -> &'static str {
    match stop {
        "end_turn" if refused => "refusal",
        "end_turn" if called => "tool_use",
        _ => stop,
    }
}

/// A Chat `finish_reason` kept past its chunk, as far as [`responses_status`] tells reasons apart
/// (a tool call's `tool_calls` is a completed response, like `stop`).
pub(super) fn chat_finish(reason: &str) -> &'static str {
    match reason {
        "length" => "length",
        "content_filter" => "content_filter",
        _ => "stop",
    }
}

/// Chat `finish_reason` → Responses `status` and `incomplete_details`.
pub(super) fn responses_status(finish: Option<&str>) -> (&'static str, Value) {
    match finish {
        Some("length") => ("incomplete", json!({ "reason": "max_output_tokens" })),
        Some("content_filter") => ("incomplete", json!({ "reason": "content_filter" })),
        _ => ("completed", Value::Null),
    }
}

/// Responses `status` → Chat `finish_reason`, before tool calls are considered.
pub(super) fn responses_finish(resp: &Value) -> &'static str {
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
