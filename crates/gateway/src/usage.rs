//! Token-usage extraction — the "passive tap" the gateway emits as billing *facts*.
//!
//! We never compute price here (pricing is a closed downstream consumer); we only extract raw
//! token counts. Two shapes per provider: the non-streaming JSON body, and the terminal event of
//! an SSE stream. For streaming we scan the relayed bytes for the usage event but never block the
//! relay on it (see `proxy`).

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// Reasoning/thinking tokens (a subset already folded into `output_tokens` — this is a breakout,
    /// not additional cost). `None` when the response didn't report the field at all (a non-reasoning
    /// model, or a provider that doesn't surface it), `Some(0)` when it was reported and is zero —
    /// that distinction is unrecoverable once the request completes, so absence must not collapse to
    /// zero.
    pub reasoning_tokens: Option<u64>,
}

// Typed views of just the fields we meter. Deserializing into these (rather than a
// `serde_json::Value` DOM) lets serde skip every field we don't read without allocating a node for
// it — no `Map`/`String`/`Number` tree to build and drop per body or per SSE line. Every field is
// `#[serde(default)]` so a missing or partial `usage` block reads as zeros, matching the prior
// pointer-with-`unwrap_or(0)` behavior.

/// OpenAI `usage` block (chat/completions). `prompt`/`completion` map to in/out; cached input rides
/// in `prompt_tokens_details.cached_tokens`; reasoning in `completion_tokens_details.reasoning_tokens`.
/// No cache-write concept on the OpenAI wire.
///
/// DeepSeek's wire never populates `prompt_tokens_details.cached_tokens` — it reports cache hits via
/// its own flat, top-level `prompt_cache_hit_tokens` field instead (DeepSeek API docs). Both providers
/// are OpenAI-dialect and `looks_anthropic_shaped` doesn't catch this (DeepSeek's body has no
/// Anthropic-style keys either), so without a fallback every DeepSeek cache hit silently bills as a
/// full-price cache miss. `prompt_tokens_details.cached_tokens` wins when present (even `Some(0)`);
/// `prompt_cache_hit_tokens` is only consulted when it's entirely absent — see `From<OpenAiUsage>`.
#[derive(Deserialize, Default)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: OpenAiPromptDetails,
    #[serde(default)]
    completion_tokens_details: OpenAiCompletionDetails,
    /// DeepSeek's flat, top-level cache-hit-token count — the fallback when
    /// `prompt_tokens_details.cached_tokens` is absent. `Option` (not a bare `u64` defaulting to 0) so
    /// "field absent" is distinguishable from "field present and zero", matching how
    /// `OpenAiPromptDetails::cached_tokens` itself distinguishes the two cases.
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u64>,
    /// Anthropic's characteristic field names — **never billed from**, only checked by
    /// [`Self::looks_anthropic_shaped`] to catch a dialect-misconfigured provider (a config-added
    /// Anthropic-wire vendor left at the default OpenAI dialect): a real OpenAI chat/completions
    /// `usage` object never carries these keys, so their presence here means this body isn't actually
    /// OpenAI-shaped.
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct OpenAiPromptDetails {
    /// `Option` so an absent `prompt_tokens_details` (or an absent `cached_tokens` within it — a
    /// DeepSeek response) is distinguishable from an explicit zero, letting
    /// `From<OpenAiUsage>` fall back to `prompt_cache_hit_tokens` only when this is truly missing.
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct OpenAiCompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

impl OpenAiUsage {
    /// See the doc comment on the `input_tokens`/`output_tokens` fields: both present is Anthropic's
    /// unambiguous fingerprint (OpenAI chat/completions never emits these key names in `usage`).
    fn looks_anthropic_shaped(&self) -> bool {
        self.input_tokens.is_some() && self.output_tokens.is_some()
    }
}

impl From<OpenAiUsage> for Usage {
    fn from(u: OpenAiUsage) -> Self {
        Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            // `prompt_tokens_details.cached_tokens` (real OpenAI, and OpenAI-compatible providers that
            // populate it) wins when present; DeepSeek's flat `prompt_cache_hit_tokens` is the fallback
            // for when it's entirely absent. Mirrors pi's
            // `rawUsage.prompt_tokens_details?.cached_tokens ?? rawUsage.prompt_cache_hit_tokens ?? 0`.
            cache_read_tokens: u
                .prompt_tokens_details
                .cached_tokens
                .or(u.prompt_cache_hit_tokens)
                .unwrap_or(0),
            cache_write_tokens: 0,
            reasoning_tokens: u.completion_tokens_details.reasoning_tokens,
        }
    }
}

/// The Responses API's `usage` block — nested under `response.completed.response.usage`, not
/// top-level like chat/completions, and named `input_tokens`/`output_tokens` (Anthropic-style) rather
/// than `prompt_tokens`/`completion_tokens`. This shape is only ever reached through the `response`
/// envelope (see `openai_stream`), which Anthropic's wire never carries — so it needs no dialect-
/// mismatch guard of its own.
#[derive(Deserialize, Default)]
struct OpenAiResponsesUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    input_tokens_details: OpenAiResponsesInputDetails,
    #[serde(default)]
    output_tokens_details: OpenAiResponsesOutputDetails,
}

#[derive(Deserialize, Default)]
struct OpenAiResponsesInputDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Deserialize, Default)]
struct OpenAiResponsesOutputDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

impl From<OpenAiResponsesUsage> for Usage {
    fn from(u: OpenAiResponsesUsage) -> Self {
        Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: u.input_tokens_details.cached_tokens,
            cache_write_tokens: 0,
            reasoning_tokens: u.output_tokens_details.reasoning_tokens,
        }
    }
}

/// Anthropic `usage` block (`/v1/messages` body + streaming events). Thinking/reasoning tokens ride
/// in `output_tokens_details.thinking_tokens` on the final usage update — verified against the live
/// API (some SDKs' own `Usage` type omits the field entirely).
#[derive(Deserialize, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    output_tokens_details: AnthropicOutputDetails,
    /// OpenAI's characteristic field names — **never billed from**, only checked by
    /// [`Self::looks_openai_shaped`] (the symmetric case of `OpenAiUsage::looks_anthropic_shaped`): a
    /// real Anthropic `usage` object never carries these keys.
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct AnthropicOutputDetails {
    #[serde(default)]
    thinking_tokens: Option<u64>,
}

impl AnthropicUsage {
    fn looks_openai_shaped(&self) -> bool {
        self.prompt_tokens.is_some() && self.completion_tokens.is_some()
    }
}

impl From<AnthropicUsage> for Usage {
    fn from(u: AnthropicUsage) -> Self {
        Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: u.cache_read_input_tokens,
            cache_write_tokens: u.cache_creation_input_tokens,
            reasoning_tokens: u.output_tokens_details.thinking_tokens,
        }
    }
}

/// OpenAI non-streaming: top-level `usage`. `None` (absent/`null`, or a dialect mismatch — see
/// `OpenAiUsage::looks_anthropic_shaped`) ⇒ no usage to meter.
pub fn openai_body(body: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Body {
        usage: Option<OpenAiUsage>,
    }
    let usage = serde_json::from_slice::<Body>(body).ok()?.usage?;
    if usage.looks_anthropic_shaped() {
        return None;
    }
    Some(Usage::from(usage))
}

/// Anthropic non-streaming: top-level `usage.{input,output,cache_*}`. `None` on a dialect mismatch —
/// see `AnthropicUsage::looks_openai_shaped`.
pub fn anthropic_body(body: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Body {
        usage: Option<AnthropicUsage>,
    }
    let u = serde_json::from_slice::<Body>(body).ok()?.usage?;
    if u.looks_openai_shaped() {
        return None;
    }
    Some(Usage::from(u))
}

/// Iterate the raw JSON payloads carried on `data:` lines of an SSE byte stream. `[DONE]` and the
/// `data:` framing are stripped; each caller deserializes the payload into its own typed view.
fn sse_data_lines(sse: &[u8]) -> impl Iterator<Item = &[u8]> + '_ {
    sse.split(|&b| b == b'\n').filter_map(|line| {
        let line = line.strip_prefix(b"data:")?;
        // SSE strips *all* leading spaces after the field colon (not exactly one) — OpenAI/Anthropic
        // emit `data: ` (one space), but a config-added OpenAI-wire provider that pads with more
        // would otherwise leave whitespace in the payload and fail the JSON parse → silent zero usage.
        // Trim the trailing end too: SSE permits CRLF line endings (RFC 8895), so splitting on `\n`
        // leaves a trailing `\r` that would otherwise fail the JSON parse → another silent zero.
        let line = line.trim_ascii();
        (line != b"[DONE]").then_some(line)
    })
}

/// OpenAI streaming: chat/completions (requires `stream_options.include_usage`) carries a top-level
/// `usage` object on the penultimate chunk; the Responses API carries it nested under
/// `response.completed.response.usage` instead, with no top-level `usage` key at all — so both shapes
/// are checked per line. Last one with usage wins. A top-level `usage` that looks Anthropic-shaped
/// (dialect mismatch — see `OpenAiUsage::looks_anthropic_shaped`) is skipped, not counted: if every
/// line is mismatched, `found` stays `None`.
pub fn openai_stream(sse: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct ResponsesEnvelope {
        usage: Option<OpenAiResponsesUsage>,
    }
    #[derive(Deserialize)]
    struct Chunk {
        usage: Option<OpenAiUsage>,
        response: Option<ResponsesEnvelope>,
    }
    let mut found = None;
    for line in sse_data_lines(sse) {
        if let Ok(chunk) = serde_json::from_slice::<Chunk>(line) {
            if let Some(u) = chunk.usage {
                if !u.looks_anthropic_shaped() {
                    found = Some(Usage::from(u));
                }
            } else if let Some(u) = chunk.response.and_then(|r| r.usage) {
                found = Some(Usage::from(u));
            }
        }
    }
    found
}

/// Anthropic streaming: input + cache tokens arrive in `message_start.message.usage`; output (and
/// reasoning/thinking tokens) accumulate in `message_delta.usage` (last delta is the cumulative
/// total). A `usage` block that looks OpenAI-shaped (dialect mismatch — see
/// `AnthropicUsage::looks_openai_shaped`) is skipped entirely: if every line is mismatched, `saw_any`
/// stays `false` and the function returns `None`.
pub fn anthropic_stream(sse: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Message {
        usage: Option<AnthropicUsage>,
    }
    #[derive(Deserialize)]
    struct Chunk {
        // `message_start` nests usage under `message`; `message_delta` carries it top-level.
        message: Option<Message>,
        usage: Option<AnthropicUsage>,
    }
    let mut usage = Usage::default();
    let mut saw_any = false;
    for line in sse_data_lines(sse) {
        let Ok(chunk) = serde_json::from_slice::<Chunk>(line) else {
            continue;
        };
        if let Some(u) = chunk.message.and_then(|m| m.usage) {
            if !u.looks_openai_shaped() {
                usage.input_tokens = u.input_tokens;
                usage.cache_read_tokens = u.cache_read_input_tokens;
                usage.cache_write_tokens = u.cache_creation_input_tokens;
                saw_any = true;
            }
        }
        if let Some(u) = chunk.usage {
            if !u.looks_openai_shaped() {
                // message_delta carries the running output token count.
                if u.output_tokens > 0 {
                    usage.output_tokens = u.output_tokens;
                }
                if let Some(rt) = u.output_tokens_details.thinking_tokens {
                    usage.reasoning_tokens = Some(rt);
                }
                saw_any = true;
            }
        }
    }
    saw_any.then_some(usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_nonstreaming() {
        let body = br#"{"usage":{"prompt_tokens":12,"completion_tokens":34,
            "prompt_tokens_details":{"cached_tokens":4}}}"#;
        assert_eq!(
            openai_body(body).unwrap(),
            Usage {
                input_tokens: 12,
                output_tokens: 34,
                cache_read_tokens: 4,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn deepseek_flat_cache_hit_tokens_fallback() {
        // DeepSeek's wire never populates `prompt_tokens_details.cached_tokens` — cache hits ride in
        // a flat, top-level `prompt_cache_hit_tokens` field instead (DeepSeek API docs). Before this
        // fix, `OpenAiUsage` only ever read the nested field, so every DeepSeek cache hit silently
        // billed as a full-price cache miss (zero `cache_read_tokens`) with no parse error tripped —
        // the body has no `prompt_tokens_details` at all, and isn't Anthropic-shaped either, so
        // nothing flagged the mismatch.
        let body = br#"{"usage":{"prompt_tokens":100,"completion_tokens":50,
            "prompt_cache_hit_tokens":64}}"#;
        assert_eq!(
            openai_body(body).unwrap(),
            Usage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_tokens: 64,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }
        );

        // The same shape arriving as the terminal SSE usage chunk (DeepSeek's streaming wire).
        let sse = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\
                    \"prompt_cache_hit_tokens\":64}}\n\n";
        assert_eq!(openai_stream(sse).unwrap().cache_read_tokens, 64);

        // `prompt_tokens_details.cached_tokens`, when present, still wins over the flat DeepSeek
        // field — the nested field is the primary source; the flat one is only a fallback for when
        // it's entirely absent.
        let both = br#"{"usage":{"prompt_tokens":100,"completion_tokens":50,
            "prompt_tokens_details":{"cached_tokens":10},"prompt_cache_hit_tokens":64}}"#;
        assert_eq!(openai_body(both).unwrap().cache_read_tokens, 10);
    }

    #[test]
    fn anthropic_nonstreaming() {
        let body = br#"{"usage":{"input_tokens":100,"output_tokens":50,
            "cache_read_input_tokens":10,"cache_creation_input_tokens":7}}"#;
        assert_eq!(
            anthropic_body(body).unwrap(),
            Usage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_tokens: 10,
                cache_write_tokens: 7,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn openai_streaming_terminal_usage() {
        let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                    data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n\
                    data: [DONE]\n\n";
        assert_eq!(
            openai_stream(sse).unwrap(),
            Usage {
                input_tokens: 5,
                output_tokens: 9,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn openai_responses_streaming_nested_usage() {
        // The Responses API has no top-level `usage` chunk at all — it rides nested under
        // `response.completed.response.usage`, with Anthropic-style field names
        // (`input_tokens`/`output_tokens`, not `prompt_tokens`/`completion_tokens`). Before this fix
        // `openai_stream` only ever checked the top-level field and would silently meter zero tokens
        // for every Responses-routed call.
        let sse = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n\
                    data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\
                    \"usage\":{\"input_tokens\":50,\"output_tokens\":20,\
                    \"input_tokens_details\":{\"cached_tokens\":10}}}}\n\n";
        assert_eq!(
            openai_stream(sse).unwrap(),
            Usage {
                input_tokens: 50,
                output_tokens: 20,
                cache_read_tokens: 10,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn anthropic_streaming_accumulates() {
        let sse = b"event: message_start\n\
                    data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}\n\n";
        assert_eq!(
            anthropic_stream(sse).unwrap(),
            Usage {
                input_tokens: 20,
                output_tokens: 15,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn anthropic_streaming_includes_cache_tokens() {
        // Cache tokens ride in `message_start.message.usage` alongside input_tokens. The earlier
        // accumulation test omits them; this guards the `cache_read`/`cache_creation` pointers so a
        // regression can't silently zero cache billing.
        let sse = b"event: message_start\n\
                    data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0,\"cache_read_input_tokens\":12,\"cache_creation_input_tokens\":8}}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}\n\n";
        assert_eq!(
            anthropic_stream(sse).unwrap(),
            Usage {
                input_tokens: 20,
                output_tokens: 15,
                cache_read_tokens: 12,
                cache_write_tokens: 8,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn tolerates_extra_leading_spaces_after_data_colon() {
        // SSE strips all leading spaces, not just one. A provider padding `data:   {…}` must still
        // parse — the alternative is a silent zero-usage row for that request.
        let sse =
            b"data:   {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":7}}\n\n";
        assert_eq!(
            openai_stream(sse).unwrap(),
            Usage {
                input_tokens: 3,
                output_tokens: 7,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn tolerates_crlf_line_endings() {
        // SSE permits CRLF (RFC 8895). Splitting on `\n` leaves a trailing `\r` on each line; the
        // parser must strip it or the JSON parse silently fails → a phantom zero-token billing row.
        let sse =
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":22}}\r\n\r\n\
              data: [DONE]\r\n\r\n";
        assert_eq!(
            openai_stream(sse).unwrap(),
            Usage {
                input_tokens: 11,
                output_tokens: 22,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: None,
            }
        );
    }

    #[test]
    fn no_usage_returns_none() {
        // Absent `usage` and unparseable bodies must both meter as `None` — never a silent zero-token
        // row that bills nothing while *looking* like a successful meter. A provider dropping `usage`
        // (an error 200, a wire-version change) or returning non-JSON must surface as "no fact", which
        // the proxy logs/alerts on, rather than a phantom 0-token success.

        // --- non-streaming bodies ---
        assert!(
            openai_body(br#"{"choices":[{"message":{"content":"hi"}}]}"#).is_none(),
            "openai body without a `usage` block has nothing to meter"
        );
        assert!(
            openai_body(b"not json at all").is_none(),
            "malformed openai body must not panic or meter zeros"
        );
        assert!(
            anthropic_body(br#"{"content":[{"type":"text","text":"hi"}]}"#).is_none(),
            "anthropic body without a `usage` block has nothing to meter"
        );
        assert!(
            anthropic_body(b"{ broken").is_none(),
            "malformed anthropic body must not panic or meter zeros"
        );

        // --- streaming: well-formed SSE that simply never carries a usage event ---
        assert!(
            openai_stream(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
            )
            .is_none(),
            "an openai stream with content but no usage chunk meters nothing"
        );
        assert!(
            anthropic_stream(
                b"event: content_block_delta\n\
                  data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n"
            )
            .is_none(),
            "an anthropic stream with no usage-bearing event meters nothing"
        );
    }

    #[test]
    fn anthropic_shaped_body_via_openai_parser_returns_none() {
        // Task #30: a config-added provider left at the default OpenAI dialect (e.g. MiniMax,
        // Kimi-Coding — real Anthropic-wire vendors) feeds an Anthropic-shaped `usage` object into
        // `openai_body`/`openai_stream`. Before this fix, `OpenAiUsage`'s `#[serde(default)]` fields
        // all silently defaulted to zero and the parser returned `Some(Usage::default())` — a
        // zero-token billing row indistinguishable from a real (and wrong) zero-usage response. It
        // must now return `None`, tripping `usage_parse_errors_total` instead.
        let anthropic_shaped_body =
            br#"{"usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":10}}"#;
        assert!(
            openai_body(anthropic_shaped_body).is_none(),
            "an Anthropic-shaped usage object must not silently parse as a zeroed OpenAI usage"
        );

        let anthropic_shaped_sse =
            b"data: {\"usage\":{\"input_tokens\":100,\"output_tokens\":50}}\n\n";
        assert!(
            openai_stream(anthropic_shaped_sse).is_none(),
            "an Anthropic-shaped SSE usage chunk must not silently parse as zeroed OpenAI usage"
        );
    }

    #[test]
    fn openai_shaped_body_via_anthropic_parser_returns_none() {
        // The symmetric case: an OpenAI-shaped `usage` object fed to the Anthropic parser (a
        // config-added OpenAI-wire provider misconfigured with `provider_dialects = "anthropic"`)
        // must not silently parse as a zeroed Anthropic usage either.
        let openai_shaped_body = br#"{"usage":{"prompt_tokens":12,"completion_tokens":34}}"#;
        assert!(
            anthropic_body(openai_shaped_body).is_none(),
            "an OpenAI-shaped usage object must not silently parse as a zeroed Anthropic usage"
        );

        let openai_shaped_sse =
            b"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":34}}\n\n";
        assert!(
            anthropic_stream(openai_shaped_sse).is_none(),
            "an OpenAI-shaped SSE usage chunk must not silently parse as zeroed Anthropic usage"
        );
    }

    #[test]
    fn openai_completions_reasoning_tokens_captured() {
        // Task #33: `completion_tokens_details.reasoning_tokens` on the chat/completions wire.
        let body = br#"{"usage":{"prompt_tokens":12,"completion_tokens":34,
            "completion_tokens_details":{"reasoning_tokens":21}}}"#;
        assert_eq!(openai_body(body).unwrap().reasoning_tokens, Some(21));

        // Absent ⇒ None, not Some(0) — distinguishing "not reported" from "reported as zero".
        let no_reasoning = br#"{"usage":{"prompt_tokens":12,"completion_tokens":34}}"#;
        assert_eq!(openai_body(no_reasoning).unwrap().reasoning_tokens, None);
    }

    #[test]
    fn openai_responses_reasoning_tokens_captured() {
        // Task #33: `output_tokens_details.reasoning_tokens` on the Responses wire (nested envelope).
        let sse = b"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\
                    \"usage\":{\"input_tokens\":50,\"output_tokens\":20,\
                    \"output_tokens_details\":{\"reasoning_tokens\":15}}}}\n\n";
        assert_eq!(openai_stream(sse).unwrap().reasoning_tokens, Some(15));
    }

    #[test]
    fn anthropic_reasoning_tokens_captured() {
        // Task #33: Anthropic reports thinking tokens in `output_tokens_details.thinking_tokens` on
        // the final `message_delta` usage update (the SDK's own `Usage` type omits this field —
        // pi reads it via a narrow cast; the gateway parses the wire JSON directly, so no cast needed).
        let body = br#"{"usage":{"input_tokens":100,"output_tokens":50,
            "output_tokens_details":{"thinking_tokens":30}}}"#;
        assert_eq!(anthropic_body(body).unwrap().reasoning_tokens, Some(30));

        let sse = b"event: message_start\n\
                    data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15,\"output_tokens_details\":{\"thinking_tokens\":9}}}\n\n";
        assert_eq!(anthropic_stream(sse).unwrap().reasoning_tokens, Some(9));

        // Absent ⇒ None, not Some(0).
        let no_reasoning = br#"{"usage":{"input_tokens":100,"output_tokens":50}}"#;
        assert_eq!(anthropic_body(no_reasoning).unwrap().reasoning_tokens, None);
    }
}
