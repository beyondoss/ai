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

use crate::agent::repair_json;
use crate::error::{Error, Result};
use crate::message::{ContentBlock, Message, Role, StreamEvent};
use crate::models::{ApiKind, ModelCaps};
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

/// A floor under [`clamp_max_tokens_to_context`]'s result: never clamp down to something so small the
/// turn can't produce a useful response. pi uses no explicit floor of its own (its clamp can only ever
/// reduce `maxTokens`, and a near-zero `available` there is already a sign compaction is badly overdue
/// — this floor is a defensive addition, not a divergence from its behavior in practice).
const MIN_CLAMPED_MAX_TOKENS: u32 = 1_024;
/// Held back below `context_window` on top of the estimated prompt size — matches pi's
/// `clampMaxTokensToContext` margin (`packages/ai/src/api/simple-options.ts`). Token *estimates* (ours:
/// chars/4, see [`crate::compaction::estimate_tokens`]) are approximate; this absorbs that slop so the
/// clamp doesn't itself become the thing that pushes a request over the edge.
const CONTEXT_CLAMP_MARGIN: u32 = 4_096;

/// Clamp `req.max_tokens` down so `estimated prompt tokens + max_tokens` doesn't already exceed the
/// model's context window — mirroring pi's `clampMaxTokensToContext`. Without this, a long-running
/// session sends its *static* `max_tokens` ceiling on every turn regardless of how much of the window
/// the live conversation has already consumed: once the prompt passes `context_window - max_tokens`,
/// every further turn is a guaranteed-to-400 request, recovered only reactively (see
/// `agent::is_context_overflow`) after paying for a wasted round-trip. Never *raises* `max_tokens` (a
/// short prompt still gets the model's real ceiling), and never clamps below what a configured
/// [`crate::transport::ThinkingConfig`] budget requires (Anthropic requires `max_tokens >
/// budget_tokens`) — a request that's already this close to the window is a case for compaction to
/// have caught, not for this clamp to make worse by triggering a *different* 400.
///
/// Shared across all three dialects (Anthropic, OpenAI Chat Completions, OpenAI Responses) — the
/// clamp itself only reads generic [`ModelRequest`]/[`ModelCaps`] fields, nothing dialect-specific;
/// `req.thinking` is simply `None` for a model that doesn't use Anthropic's budget shape, so the floor
/// term is harmlessly `0` there.
pub(crate) fn clamp_max_tokens_to_context(req: &ModelRequest, caps: &ModelCaps) -> u32 {
    let estimated_prompt: u32 = req
        .messages
        .iter()
        .map(crate::compaction::estimate_message_tokens)
        .fold(0u32, |acc, n| acc.saturating_add(n))
        .saturating_add(
            req.system
                .as_deref()
                .map(crate::compaction::estimate_tokens)
                .unwrap_or(0),
        );
    let available = caps
        .context_window
        .saturating_sub(estimated_prompt)
        .saturating_sub(CONTEXT_CLAMP_MARGIN);
    let floor = req
        .thinking
        .map(|t| t.budget_tokens.saturating_add(1))
        .unwrap_or(0)
        .max(MIN_CLAMPED_MAX_TOKENS);
    req.max_tokens.min(available.max(floor))
}

/// Ensure every `tool_use` block reaches the wire with a matching `tool_result` immediately following
/// it, synthesizing an error placeholder for any that don't — matches pi's own fixup
/// (`tool-call-without-result.test.ts`): a completion must not 400 just because *some* message list
/// reaching a request has an orphaned `tool_use` with no matching result. `Agent::run_events_steered`'s
/// `repair_cancelled_dispatch` already covers the one way *this process* can produce that shape
/// (cancellation mid-dispatch, by mutating the persisted session directly); this covers every other way
/// one can reach a request — a hand-edited or externally-loaded session file, a session mutated by a
/// tool outside the normal turn loop, or an as-yet-undiscovered code path — without needing every such
/// path to remember to repair itself individually. Called once, generically, right before any dialect's
/// `build_body` — shared rather than dialect-specific because it's a *message-shape* invariant (every
/// provider requires it), not a wire-format concern.
///
/// Cheap when nothing needs fixing (the overwhelmingly common case): a first read-only pass decides
/// whether any repair is needed at all before ever allocating; only a genuinely orphaned `tool_use`
/// pays for the owned copy.
pub(crate) fn repair_orphaned_tool_use(messages: &[Message]) -> std::borrow::Cow<'_, [Message]> {
    if !messages
        .iter()
        .enumerate()
        .any(|(i, m)| has_orphan(messages, i, m))
    {
        return std::borrow::Cow::Borrowed(messages);
    }
    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 1);
    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        let missing: Vec<ContentBlock> = message
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, .. } if !answered(messages, i, id) => {
                    Some(ContentBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: "no result recorded for this tool call".to_string(),
                        is_error: true,
                        images: Vec::new(),
                    })
                }
                _ => None,
            })
            .collect();
        out.push(message.clone());
        if missing.is_empty() {
            i += 1;
            continue;
        }
        // Every `tool_result` for one assistant turn must land in a *single* immediately-following
        // message (splitting them across two consecutive messages, or leaving the provider-required
        // strict user/assistant alternation, isn't a shape any dialect accepts). If the next message
        // already exists and is a user turn — a real (if partial) `tool_results` message, or ordinary
        // trailing text like pi's own `tool-call-without-result.test.ts` fixture — merge the synthetic
        // results into a copy of it, ahead of whatever it already carried, and consume it here rather
        // than re-visiting it next iteration. Otherwise (end of history, or the next turn is somehow
        // another assistant message) insert a brand-new trailing message instead.
        match messages.get(i + 1).filter(|n| n.role == Role::User) {
            Some(next) => {
                let mut content = missing;
                content.extend(next.content.clone());
                out.push(Message {
                    content,
                    ..next.clone()
                });
                i += 2;
            }
            None => {
                out.push(Message::tool_results(missing));
                i += 1;
            }
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Whether `message` (at index `i` in `messages`) has any `tool_use` block not [`answered`] by the very
/// next message.
fn has_orphan(messages: &[Message], i: usize, message: &Message) -> bool {
    message.content.iter().any(|b| match b {
        ContentBlock::ToolUse { id, .. } => !answered(messages, i, id),
        _ => false,
    })
}

/// Whether `messages[i + 1]` carries a `tool_result` for `tool_use_id` — the shape every provider
/// requires (a `tool_use` must be immediately followed by its result, not merely answered *somewhere*
/// later in history).
fn answered(messages: &[Message], i: usize, tool_use_id: &str) -> bool {
    messages.get(i + 1).is_some_and(|next| {
        next.content.iter().any(
            |b| matches!(b, ContentBlock::ToolResult { tool_use_id: rid, .. } if rid == tool_use_id),
        )
    })
}

/// Ensure every message reaches the wire with at least one non-empty content block, and that an
/// aborted or errored assistant turn never replays its (possibly partial) real content. Two related
/// fixups, both padding rather than dropping the message's *slot* (see below):
///
/// - Filters out whitespace-only `Text` blocks — matching pi's `convertMessages`
///   (`anthropic-messages.ts`) and `openai-completions.ts:1007-1018` ("Some providers require 'either
///   content or tool_calls, but not none'"). Anthropic 400s on an empty `text` content block (see
///   `anthropic::downgrade_unsigned_thinking`'s doc comment); an OpenAI-shaped `{content: null}` with
///   no `tool_calls` hits the same rejection class. Reachable today via `Message::error()`'s empty
///   closing record (a whole-run failure with no partial response), the immediate-abort turn
///   (`Agent::run_events_steered`), and `Session::scrub_cross_model_state` leaving `content: []`
///   behind after dropping a foreign model's empty thinking block
///   (`scrub_cross_model_state_drops_an_empty_thinking_block_instead_of_downgrading_it`).
/// - Drops *all* content from a message marked [`Message::aborted`] or carrying
///   [`Message::error_message`] — matching pi's `transform-messages.ts:186-194`, which skips any
///   assistant message with `stopReason: "error"`/`"aborted"` outright when building the next
///   request ("may have partial content (reasoning without message, incomplete tool calls) …
///   replaying them can cause API errors"). A stream cancelled right after a `Thinking` block's
///   signature landed but before anything else opened persists `Turn::blocks = [Thinking{signature}]`
///   via `with_aborted`/`with_error` — replayed verbatim, that's exactly Anthropic/OpenAI's "reasoning
///   with nothing following it in this turn" rejection, and once one turn hits it every later turn
///   resends the same dangling block and fails identically. `Message::with_error`'s own doc comment
///   already promised this ("Not replayed on the wire either way") — this is where that promise is
///   actually kept for the block *content*, not just the `error_message`/`aborted` marker fields
///   themselves (`anthropic::strip_model_id` already scrubs those).
///
/// Unlike pi, a message left with zero content after either fixup is NOT dropped from the list: this
/// codebase relies on that record staying present to keep role alternation valid for whatever prompt
/// follows it (see `a_prompt_after_a_failed_run_does_not_double_push_a_user_turn` — dropping
/// `assistant(error)` from `[user, assistant(error), user]` would leave two consecutive `user` turns,
/// a shape no dialect accepts either). Instead, an otherwise-empty message is padded with a single
/// placeholder text block. Session storage is untouched — this only shapes the wire copy, mirroring
/// `strip_model_id`/`downgrade_unsigned_thinking`. Called once, generically, right before any
/// dialect's `build_body`, alongside `repair_orphaned_tool_use`.
///
/// Cheap when nothing needs fixing (the overwhelmingly common case): a first read-only pass decides
/// whether any repair is needed at all before ever allocating.
pub(crate) fn ensure_non_empty_content(messages: &[Message]) -> std::borrow::Cow<'_, [Message]> {
    if !messages.iter().any(needs_content_fixup) {
        return std::borrow::Cow::Borrowed(messages);
    }
    std::borrow::Cow::Owned(
        messages
            .iter()
            .map(|m| {
                if !needs_content_fixup(m) {
                    return m.clone();
                }
                let mut content: Vec<ContentBlock> = if is_incomplete_turn(m) {
                    Vec::new()
                } else {
                    m.content
                        .iter()
                        .filter(|b| !is_empty_text(b))
                        .cloned()
                        .collect()
                };
                if content.is_empty() {
                    content.push(ContentBlock::text(EMPTY_MESSAGE_PLACEHOLDER));
                }
                Message {
                    content,
                    ..m.clone()
                }
            })
            .collect(),
    )
}

/// Whether `message` needs [`ensure_non_empty_content`]'s fixup: an aborted/errored turn (whose
/// content must be dropped outright, however much of it there is), a whitespace-only `Text` block, or
/// no content at all.
fn needs_content_fixup(message: &Message) -> bool {
    is_incomplete_turn(message)
        || message.content.is_empty()
        || message.content.iter().any(is_empty_text)
}

/// Whether `message` is a partial/aborted assistant turn whose real content must never reach the
/// wire — see [`ensure_non_empty_content`]'s doc comment.
fn is_incomplete_turn(message: &Message) -> bool {
    message.aborted || message.error_message.is_some()
}

fn is_empty_text(block: &ContentBlock) -> bool {
    matches!(block, ContentBlock::Text { text, .. } if text.trim().is_empty())
}

/// Placeholder substituted for a message that would otherwise reach the wire with zero content — see
/// [`ensure_non_empty_content`].
const EMPTY_MESSAGE_PLACEHOLDER: &str = "(no content)";

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

    /// Whether this decoder has already seen its provider's terminal event (Anthropic:
    /// `message_stop`). `false` by default (most decoders never need this). [`push_sse_line`] uses it
    /// to decide whether a `data:` line that fails to parse as JSON is a real transport error or just
    /// trailing noise — a gateway/proxy appending a keepalive or stats line *after* the real message
    /// has already completed (e.g. `event: proxy.stats` / `data: not-json`) shouldn't fail an otherwise
    /// successful turn.
    fn is_terminal(&self) -> bool {
        false
    }

    /// Whether a `data:` payload that fails its first JSON parse should get a repair pass
    /// (`crate::agent::repair_json`) before [`push_sse_line`] gives up on it. `false` by default.
    /// Anthropic's raw SSE stream can carry a malformed-but-recoverable event body — a backslash
    /// that isn't a valid JSON escape (an unescaped Windows path segment) or a raw control
    /// character inside a string value that should have been escaped (a literal tab/newline) — and
    /// pi tolerates exactly this by running its `repairJson` pass over the event body before
    /// parsing (`parseJsonWithRepair`, `packages/ai/src/utils/json-parse.ts`, wired up in
    /// `iterateAnthropicEvents`, `packages/ai/src/api/anthropic-messages.ts`). Every other pi
    /// dialect hands its outer per-chunk parsing to that provider's own SDK rather than hand-rolling
    /// SSE decode, so this repair is never reached for them even though `repairJson` itself is a
    /// generic utility — `anthropic::Decoder` is the only override, matching that scoping.
    fn repairs_json(&self) -> bool {
        false
    }
}

/// Accumulates consecutive `data:` line payloads belonging to one logical SSE event, per the SSE
/// spec (a blank line, not the `data:` prefix itself, is what terminates an event) — mirrors pi's own
/// `SseDecoderState`/`flushSseEvent` (`anthropic-messages.ts`). Real Anthropic/OpenAI never split a
/// single JSON event across multiple `data:` lines in practice, so before this buffer existed
/// [`push_sse_line`] just parsed each `data:` line the moment it arrived — a spec-conformant
/// intermediary (a proxy that re-wraps long lines) legitimately could, though, so the join has to
/// happen before the first JSON-parse attempt, not after.
///
/// One buffer is threaded through an entire stream by the caller ([`decode_sse`], or the streaming
/// HTTP client) — [`push_sse_line`] itself is stateless and free, the same shape as [`LineFramer`]
/// pairs with the streaming client for byte-level framing.
///
/// [`LineFramer`]: crate::client::LineFramer
#[derive(Default)]
pub struct SseEventBuffer {
    data: Vec<String>,
}

impl SseEventBuffer {
    /// A buffer with no pending data lines.
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffer one `data:` line's payload (already stripped of the `data:` prefix and surrounding
    /// whitespace).
    fn push_data(&mut self, payload: &str) {
        self.data.push(payload.to_string());
    }

    /// Join and clear the buffered payloads with `"\n"`, or `None` if nothing is buffered — matches
    /// pi's `state.data.join("\n")`.
    fn take(&mut self) -> Option<String> {
        if self.data.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.data).join("\n"))
        }
    }
}

/// Decode a complete SSE body into events. Splits on `data:` lines, skips comments/`event:` lines
/// and the OpenAI `[DONE]` sentinel, and flushes the decoder at the end. The streaming HTTP client
/// reuses the same decoder incrementally; this is the buffered form used by tests.
pub fn decode_sse(decoder: &mut dyn StreamDecoder, raw: &str) -> Result<Vec<StreamEvent>> {
    let mut out = Vec::new();
    let mut buf = SseEventBuffer::new();
    for line in raw.lines() {
        out.extend(push_sse_line(decoder, &mut buf, line)?);
    }
    // Flush a final event that never got its trailing blank-line terminator (a stream that ends
    // mid-event, with no closing `\n\n`) — matches pi's own end-of-stream `flushSseEvent` call in
    // `iterateSseMessages`.
    out.extend(push_sse_line(decoder, &mut buf, "")?);
    out.extend(decoder.finish()?);
    Ok(out)
}

/// Feed a single SSE line to the decoder, returning any events it produced. A `data:` line is
/// buffered in `buf`, not parsed immediately — a blank line (the SSE event terminator) joins every
/// payload buffered since the last flush with `"\n"` and parses that as one JSON value, matching pi's
/// `decodeSseLine`/`flushSseEvent`. Non-`data:`, non-blank lines (comments, `event:`) and the OpenAI
/// `[DONE]` sentinel produce nothing and don't touch the buffer. This is the incremental entry point
/// the streaming HTTP client drives line-by-line off the socket; the caller is responsible for
/// flushing any trailing buffered payload (pass an empty `line`) and then calling
/// [`StreamDecoder::finish`] once the stream closes.
pub fn push_sse_line(
    decoder: &mut dyn StreamDecoder,
    buf: &mut SseEventBuffer,
    line: &str,
) -> Result<Vec<StreamEvent>> {
    // Strip both a trailing `\r` (CRLF framing) *and* `\n`: `decode_sse`'s lines come from
    // `str::lines()`, which already strips both terminators, but the streaming client's `LineFramer`
    // hands back each line *including* its trailing `\n` (see its own doc comment) — a blank SSE line
    // (the event terminator) arrives here as the single-character string `"\n"`, not `""`, unless the
    // newline itself is stripped too. Without this, the blank-line flush below would never fire for
    // the live streaming path and every event would join with the next, corrupting the JSON.
    let line = line.trim_end_matches(['\n', '\r']);
    if line.is_empty() {
        return match buf.take() {
            Some(payload) => parse_and_push(decoder, &payload),
            None => Ok(Vec::new()),
        };
    }
    let Some(payload) = line.strip_prefix("data:") else {
        return Ok(Vec::new());
    };
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return Ok(Vec::new());
    }
    buf.push_data(payload);
    Ok(Vec::new())
}

/// Parse one joined SSE `data:` payload (see [`SseEventBuffer`]) and push it into the decoder.
/// Factored out of [`push_sse_line`] so both the blank-line flush and the end-of-stream flush share
/// the same parse/repair/error-surfacing logic.
fn parse_and_push(decoder: &mut dyn StreamDecoder, payload: &str) -> Result<Vec<StreamEvent>> {
    // LOW pi-parity gap, deliberately not fixed: pi's `JSON.parse` accepts a JSON string containing a
    // lone (unpaired) `\uD800`-`\uDFFF` escape — JS strings are UTF-16 code-unit sequences that can
    // hold one — while `serde_json::from_str` rejects it here, since Rust's `String`/`str` guarantee
    // well-formed UTF-8 at the type level and a lone surrogate has no valid UTF-8 encoding at all; a
    // truly faithful "accept it" would require abandoning that guarantee. A lenient equivalent is
    // reachable (pre-scan the raw payload for an unpaired escape and substitute `�` before
    // parsing), but was evaluated and rejected: this is inbound-only, requires either an encoder bug
    // or a corrupted relay to trigger at all (well-formed JSON encoders never emit a lone surrogate),
    // and failing the turn loudly is arguably the *safer* response to a corrupted/tampered stream than
    // silently substituting replacement characters into what might be live tool-call arguments — the
    // fail-closed behavior below is a considered choice, not an oversight.
    let v: Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        // A first-parse failure gets one repair attempt (Anthropic only — see
        // `StreamDecoder::repairs_json`) before falling back to the trailing-noise/hard-error split
        // below, mirroring pi's `parseJsonWithRepair`.
        Err(e) => match repair_and_parse(decoder, payload) {
            Some(v) => v,
            // A non-JSON `data:` line arriving after the decoder's own terminal event is trailing
            // noise (a gateway/proxy's keepalive or stats line tacked on after the real message
            // already completed), not a real transport failure — matching pi, which filters by the
            // SSE `event:` type before ever attempting to parse `data:` as JSON, so an unrecognized
            // trailing event never reaches its JSON parser at all. Before the decoder has seen its
            // terminal event, this is still a hard failure: garbage mid-turn is a genuine
            // corrupted/tampered stream, not noise.
            None if decoder.is_terminal() => {
                tracing::debug!(error = %e, "ignoring non-JSON SSE data after the stream's terminal event");
                return Ok(Vec::new());
            }
            None => return Err(Error::Transport(format!("malformed SSE json: {e}"))),
        },
    };
    // A provider can report a failure *in-band* mid-stream — Anthropic as `{"type":"error",…}`
    // (preceded by an `event: error` line we don't see here), OpenAI as a bare `{"error":{…}}` chunk.
    // Surface it as a transport error; otherwise it falls through every decoder's catch-all arm and
    // the turn ends silently as a successful-looking `EndTurn` with no content and no usage.
    if let Some(msg) = sse_error(&v) {
        return Err(Error::Transport(format!("provider stream error: {msg}")));
    }
    Ok(decoder.push(&v))
}

/// Second-chance parse for a `data:` payload whose first `serde_json::from_str` failed. Mirrors pi's
/// `parseJsonWithRepair` (`packages/ai/src/utils/json-parse.ts`): only attempted when the decoder
/// opts in ([`StreamDecoder::repairs_json`]) and the repair pass actually changed the payload —
/// an unchanged payload would just fail the same way again, so `None` here falls straight through to
/// [`push_sse_line`]'s existing trailing-noise/hard-error handling.
fn repair_and_parse(decoder: &dyn StreamDecoder, payload: &str) -> Option<Value> {
    if !decoder.repairs_json() {
        return None;
    }
    let repaired = repair_json(payload);
    if repaired == payload {
        return None;
    }
    serde_json::from_str(&repaired).ok()
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

    #[test]
    fn multi_line_sse_event_is_joined_before_parsing() {
        // pi-parity: a spec-conformant intermediary can legitimately split one logical SSE event
        // across multiple `data:` lines (the SSE spec's own event boundary is a blank line, not a
        // per-`data:`-line one) — never observed from real Anthropic/OpenAI in practice, but
        // `decodeSseLine`/`flushSseEvent` (pi's `anthropic-messages.ts`) join every `data:` line since
        // the last blank line with `"\n"` before parsing, rather than treating each as standalone
        // JSON. Split `message_start`'s payload across two `data:` lines at a comma (valid JSON
        // whitespace) — a value only reconstructible if the join actually happened before the first
        // parse attempt.
        const SPLIT_EVENT: &str = r#"
event: message_start
data: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":5,
data: "output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut dec = anthropic::Decoder::default();
        let events = decode_sse(&mut dec, SPLIT_EVENT).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(*u),
                _ => None,
            })
            .expect("a usage event, proving the split message_start payload parsed correctly");
        assert_eq!(
            usage.input_tokens, 5,
            "input_tokens only survives a correct join"
        );
        assert_eq!(usage.output_tokens, 7);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "hi")),
            "the rest of the stream must still decode normally: {events:?}"
        );
    }

    #[test]
    fn repair_orphaned_tool_use_leaves_a_well_formed_list_untouched() {
        // pi: tool-call-without-result.test.ts's non-orphaned control case, and the general fast path:
        // a `tool_use` with its matching `tool_result` right after must not be touched — proven via
        // `Cow::Borrowed` (no allocation), not just value equality.
        let messages = vec![
            Message::assistant(vec![ContentBlock::tool_use(
                "t1",
                "bash",
                serde_json::json!({}),
            )]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: "ok".into(),
                is_error: false,
                images: Vec::new(),
            }]),
        ];
        assert!(matches!(
            repair_orphaned_tool_use(&messages),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn repair_orphaned_tool_use_inserts_a_synthetic_result_for_an_unanswered_tool_use() {
        // pi: tool-call-without-result.test.ts — `[assistant([ToolUse{id:"t1"}]), user("nevermind")]`
        // with no matching `tool_result` anywhere must not reach a request unrepaired (every provider
        // 400s on an orphaned `tool_use`). The synthetic result merges into the existing next message
        // rather than becoming a new one — two consecutive user-role messages isn't a shape any dialect
        // accepts, so "nevermind" must survive alongside the synthetic result, not be pushed later.
        let messages = vec![
            Message::assistant(vec![ContentBlock::tool_use(
                "t1",
                "bash",
                serde_json::json!({}),
            )]),
            Message::user("nevermind"),
        ];
        let repaired = repair_orphaned_tool_use(&messages);
        assert!(matches!(repaired, std::borrow::Cow::Owned(_)));
        assert_eq!(repaired.len(), 2, "got: {repaired:?}");
        assert_eq!(repaired[0], messages[0]);
        match &repaired[1].content[..] {
            [
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error,
                    ..
                },
                ContentBlock::Text { text, .. },
            ] => {
                assert_eq!(tool_use_id, "t1");
                assert!(*is_error);
                assert_eq!(text, "nevermind");
            }
            other => panic!("expected [synthetic ToolResult, original Text], got {other:?}"),
        }
    }

    #[test]
    fn repair_orphaned_tool_use_handles_multiple_gaps_independently() {
        let messages = vec![
            Message::assistant(vec![ContentBlock::tool_use(
                "t1",
                "bash",
                serde_json::json!({}),
            )]),
            Message::user("first gap"),
            Message::assistant(vec![ContentBlock::tool_use(
                "t2",
                "bash",
                serde_json::json!({}),
            )]),
            Message::user("second gap"),
        ];
        let repaired = repair_orphaned_tool_use(&messages);
        assert_eq!(repaired.len(), 4, "got: {repaired:?}");
        let ids: Vec<&str> = repaired
            .iter()
            .filter_map(|m| m.content.first())
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec!["t1", "t2"]);
    }

    #[test]
    fn repair_orphaned_tool_use_does_not_touch_an_already_answered_call_among_several() {
        // Only the genuinely orphaned tool_use gets a synthetic result — an already-answered one in
        // the same assistant turn is left alone, not duplicated. Both results land in one merged
        // message (every dialect requires all of one turn's results together, not split across two).
        let messages = vec![
            Message::assistant(vec![
                ContentBlock::tool_use("answered", "bash", serde_json::json!({})),
                ContentBlock::tool_use("orphan", "bash", serde_json::json!({})),
            ]),
            Message::tool_results(vec![ContentBlock::ToolResult {
                tool_use_id: "answered".into(),
                content: "ok".into(),
                is_error: false,
                images: Vec::new(),
            }]),
        ];
        let repaired = repair_orphaned_tool_use(&messages);
        assert_eq!(repaired.len(), 2, "got: {repaired:?}");
        match &repaired[1].content[..] {
            [
                ContentBlock::ToolResult {
                    tool_use_id: first, ..
                },
                ContentBlock::ToolResult {
                    tool_use_id: second,
                    content,
                    is_error,
                    ..
                },
            ] => {
                assert_eq!(first, "orphan", "synthetic result comes first");
                assert_eq!(second, "answered");
                assert_eq!(content, "ok", "the real result must be untouched");
                assert!(!is_error);
            }
            other => panic!("expected exactly two ToolResult blocks merged, got {other:?}"),
        }
    }

    #[test]
    fn ensure_non_empty_content_leaves_a_well_formed_list_untouched() {
        // Proven via `Cow::Borrowed` (no allocation), not just value equality — the common case.
        let messages = vec![
            Message::user("hi"),
            Message::assistant(vec![ContentBlock::text("hello")]),
        ];
        assert!(matches!(
            ensure_non_empty_content(&messages),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn ensure_non_empty_content_pads_message_error_s_empty_closing_record() {
        // The confirmed-reachable case: a whole-run failure with no partial response persists
        // `Message::error(...)` — a single whitespace-only `Text` block, `error_message: Some(_)`,
        // otherwise a completely ordinary assistant turn that must still occupy its slot in the
        // alternation (see the doc comment on `ensure_non_empty_content`).
        let messages = vec![
            Message::user("hi"),
            Message::error("transport error: gateway returned 400"),
        ];
        let fixed = ensure_non_empty_content(&messages);
        assert!(matches!(fixed, std::borrow::Cow::Owned(_)));
        assert_eq!(fixed.len(), 2, "the record must not be dropped: {fixed:?}");
        assert_eq!(fixed[1].role, Role::Assistant);
        assert_eq!(
            fixed[1].content,
            vec![ContentBlock::text(EMPTY_MESSAGE_PLACEHOLDER)]
        );
        // Session-persistence-relevant fields must survive the wire-only fixup untouched.
        assert_eq!(
            fixed[1].error_message.as_deref(),
            Some("transport error: gateway returned 400")
        );
    }

    #[test]
    fn ensure_non_empty_content_pads_a_message_left_with_zero_blocks_after_scrubbing() {
        // Mirrors `session::tests::scrub_cross_model_state_drops_an_empty_thinking_block_instead_of_
        // downgrading_it` — a foreign model's empty thinking block is dropped entirely, leaving
        // `content: []` with no thinking, no text, nothing. That message must still reach the wire
        // with something, not an empty content array (Anthropic 400s on that shape too).
        let messages = vec![Message::user("hi"), Message::assistant(vec![])];
        let fixed = ensure_non_empty_content(&messages);
        assert!(matches!(fixed, std::borrow::Cow::Owned(_)));
        assert_eq!(
            fixed[1].content,
            vec![ContentBlock::text(EMPTY_MESSAGE_PLACEHOLDER)]
        );
    }

    #[test]
    fn ensure_non_empty_content_filters_a_stray_empty_text_block_but_keeps_real_content() {
        // A whitespace-only text block alongside real content (a tool call, an image) is dropped —
        // matching pi's per-block filtering — without disturbing the rest of the message or padding
        // anything (there's already real content left).
        let messages = vec![Message::assistant(vec![
            ContentBlock::text("   \n\t  "),
            ContentBlock::tool_use("t1", "bash", serde_json::json!({})),
        ])];
        let fixed = ensure_non_empty_content(&messages);
        assert!(matches!(fixed, std::borrow::Cow::Owned(_)));
        assert_eq!(fixed[0].content.len(), 1, "got: {:?}", fixed[0].content);
        assert!(matches!(fixed[0].content[0], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn ensure_non_empty_content_drops_a_dangling_thinking_block_from_an_aborted_middle_turn() {
        // pi-parity fix (Task #27): a stream cancelled right after a `Thinking` block's signature
        // landed but before anything else opened persists `Turn::blocks = [Thinking{signature}]` via
        // `with_aborted` — replayed verbatim on every later request, that's Anthropic/OpenAI's own
        // "reasoning with nothing following it" rejection, poisoning every subsequent turn identically.
        // Matches pi's `transform-messages.ts:186-194`, which skips the whole assistant message
        // outright rather than replaying any of its partial content.
        let messages = vec![
            Message::user("start"),
            Message::assistant(vec![ContentBlock::Thinking {
                text: "reasoning that never finished".into(),
                signature: "sig-dangling".into(),
            }])
            .with_model_id("claude-opus-4-8")
            .with_aborted(),
            Message::user("try again"),
        ];
        let fixed = ensure_non_empty_content(&messages);
        assert!(matches!(fixed, std::borrow::Cow::Owned(_)));
        assert_eq!(fixed.len(), 3, "the record's slot must survive: {fixed:?}");
        assert_eq!(
            fixed[1].content,
            vec![ContentBlock::text(EMPTY_MESSAGE_PLACEHOLDER)],
            "the dangling Thinking block must not reach the wire, not even downgraded to text"
        );
        // Persistence-relevant fields (what a human reading the session transcript still sees) survive
        // the wire-only fixup untouched — only the outbound copy's content is affected.
        assert!(fixed[1].aborted);
        assert_eq!(
            messages[1].content,
            vec![ContentBlock::Thinking {
                text: "reasoning that never finished".into(),
                signature: "sig-dangling".into(),
            }],
            "the original session messages must be untouched by the wire-only fixup"
        );
        assert_eq!(
            fixed.iter().map(|m| m.role).collect::<Vec<_>>(),
            vec![Role::User, Role::Assistant, Role::User],
            "role alternation must stay valid"
        );
    }

    #[test]
    fn ensure_non_empty_content_drops_full_real_content_from_a_mid_stream_error() {
        // The `with_error` sibling case: real content (not just a dangling thinking block) that
        // streamed before a mid-turn transport failure must also be fully excluded from the next
        // request, not just have its `error_message` marker field stripped.
        let messages = vec![
            Message::user("hi"),
            Message::assistant(vec![
                ContentBlock::text("here is what I found so far"),
                ContentBlock::tool_use("t1", "bash", serde_json::json!({})),
            ])
            .with_model_id("claude-opus-4-8")
            .with_error("transport error: connection reset"),
        ];
        let fixed = ensure_non_empty_content(&messages);
        assert_eq!(
            fixed[1].content,
            vec![ContentBlock::text(EMPTY_MESSAGE_PLACEHOLDER)],
            "got: {:?}",
            fixed[1].content
        );
    }

    #[test]
    fn ensure_non_empty_content_preserves_alternation_after_a_failed_run_and_a_retry_prompt() {
        // The scenario `agent::tests::a_prompt_after_a_failed_run_does_not_double_push_a_user_turn`
        // guards at the session level: `[user, assistant(error), user]` must stay 3 messages on the
        // wire too — padding, not dropping, is what keeps role alternation valid here.
        let messages = vec![
            Message::user("hi"),
            Message::error("boom"),
            Message::user("after failure"),
        ];
        let fixed = ensure_non_empty_content(&messages);
        assert_eq!(fixed.len(), 3, "got: {fixed:?}");
        assert_eq!(
            fixed.iter().map(|m| m.role).collect::<Vec<_>>(),
            vec![Role::User, Role::Assistant, Role::User],
            "must not collapse into two consecutive user turns"
        );
    }
}
