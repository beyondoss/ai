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

/// Recover a `usage` block from a body that did **not** parse as a whole JSON document.
///
/// `proxy::logging` is handed a bounded *tail* of the response, so a non-streaming body larger than
/// `USAGE_TAIL_CAP` arrives front-truncated: it begins mid-value, `serde_json` fails at byte 0, and
/// the request meters as zero. That is not an edge case — an OpenAI embeddings response puts its
/// `usage` after the `data` array, and a batch of eight 3072-dimension vectors is ~249 KB, so every
/// batched embeddings call was billing zero tokens.
///
/// The bytes we need are present, just not as a standalone document: both wire formats put `usage`
/// last. So anchor on the **last** `"usage"` in the buffer and deserialize the single value that
/// follows, letting `serde_json` stop at the end of that object and ignore the trailing bytes.
///
/// Only ever called after the whole-body parse has already failed, which is what keeps the heuristic
/// safe: a well-formed body never reaches it, so a `"usage"` appearing inside generated content can
/// only mislead us on a body that was going to meter zero anyway. The `rfind` is what makes even
/// that unlikely — content precedes `usage` in both formats, so the last occurrence is the real one.
fn recover_trailing_usage<T: serde::de::DeserializeOwned>(body: &[u8]) -> Option<T> {
    const KEY: &[u8] = br#""usage""#;
    let at = memchr::memmem::rfind(body, KEY)?;
    let rest = &body[at + KEY.len()..];
    // Step over the `:` separating the key from its value (JSON permits whitespace either side).
    let colon = memchr::memchr(b':', rest)?;
    let mut de = serde_json::Deserializer::from_slice(&rest[colon + 1..]);
    // Deliberately no `de.end()`: the value is followed by the rest of the object, and requiring EOF
    // is precisely what the whole-document parse already failed on.
    T::deserialize(&mut de).ok()
}

/// OpenAI non-streaming: top-level `usage`. `None` (absent/`null`, or a dialect mismatch — see
/// `OpenAiUsage::looks_anthropic_shaped`) ⇒ no usage to meter.
pub fn openai_body(body: &[u8]) -> Option<Usage> {
    #[derive(Deserialize)]
    struct Body {
        usage: Option<OpenAiUsage>,
    }
    let usage = match serde_json::from_slice::<Body>(body) {
        Ok(b) => b.usage?,
        // Front-truncated tail of an oversized body — see `recover_trailing_usage`.
        Err(_) => recover_trailing_usage::<OpenAiUsage>(body)?,
    };
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
    let u = match serde_json::from_slice::<Body>(body) {
        Ok(b) => b.usage?,
        Err(_) => recover_trailing_usage::<AnthropicUsage>(body)?,
    };
    if u.looks_openai_shaped() {
        return None;
    }
    Some(Usage::from(u))
}

/// Strip the SSE `data:` framing from one line, yielding the raw JSON payload, or `None` if the line
/// carries no payload we care about (a non-`data:` field, a blank separator, or the `[DONE]`
/// sentinel).
fn strip_sse_data(line: &[u8]) -> Option<&[u8]> {
    let line = line.strip_prefix(b"data:")?;
    // SSE strips *all* leading spaces after the field colon (not exactly one) — OpenAI/Anthropic
    // emit `data: ` (one space), but a config-added OpenAI-wire provider that pads with more
    // would otherwise leave whitespace in the payload and fail the JSON parse → silent zero usage.
    // Trim the trailing end too: SSE permits CRLF line endings (RFC 8895), so splitting on `\n`
    // leaves a trailing `\r` that would otherwise fail the JSON parse → another silent zero.
    let line = line.trim_ascii();
    (line != b"[DONE]").then_some(line)
}

/// Forward line iterator over an SSE byte stream, split on `\n` with a SIMD-accelerated scan.
///
/// `slice::split(|&b| b == b'\n')` compiles to `position(closure)` — a scalar, byte-at-a-time loop
/// that LLVM does not vectorize. Over a 64 KiB tail that measured **17.6 µs vs 2.65 µs** for
/// `memchr`, i.e. ~15 µs of pure scan waste per streaming request, before a single byte is parsed.
///
/// One deliberate difference from `slice::split`: a trailing `\n` yields no final empty element
/// here. That element could never carry a payload ([`strip_sse_data`] rejects it), so every caller
/// sees the same sequence.
struct SseLines<'a> {
    rest: &'a [u8],
}

fn sse_lines(sse: &[u8]) -> SseLines<'_> {
    SseLines { rest: sse }
}

impl<'a> Iterator for SseLines<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.rest.is_empty() {
            return None;
        }
        match memchr::memchr(b'\n', self.rest) {
            Some(i) => {
                let line = &self.rest[..i];
                self.rest = &self.rest[i + 1..];
                Some(line)
            }
            // Final line with no trailing newline — a tail can end mid-stream.
            None => Some(std::mem::take(&mut self.rest)),
        }
    }
}

/// Iterate the raw JSON payloads carried on `data:` lines, **newest first**. Used by the dialects
/// whose answer is "the last usage block wins", so they can stop at the first hit instead of parsing
/// every line to overwrite the result.
fn sse_data_lines_rev(sse: &[u8]) -> impl Iterator<Item = &[u8]> {
    sse.rsplit(|&b| b == b'\n').filter_map(strip_sse_data)
}

/// Whether a line could possibly carry a usage block.
///
/// Every shape we meter — top-level `usage`, `message.usage`, `response.usage` — spells the key
/// literally, so a line without the substring cannot deserialize one and the JSON parse is pure
/// waste. On an Anthropic stream that is every `content_block_delta`, i.e. almost the whole tail.
///
/// False *negatives* would need a provider to escape the key (`"usage"`), which none does; a
/// false *positive* (the word appearing in generated text) costs only the parse we would have done
/// anyway, so the filter can never change the answer — only how often we pay for it.
fn might_carry_usage(finder: &memchr::memmem::Finder<'_>, payload: &[u8]) -> bool {
    finder.find(payload).is_some()
}

/// OpenAI streaming: chat/completions (requires `stream_options.include_usage`) carries a top-level
/// `usage` object on the penultimate chunk; the Responses API carries it nested under
/// `response.completed.response.usage` instead, with no top-level `usage` key at all — so both shapes
/// are checked per line. Last one with usage wins — which is why this scans the tail **backwards**
/// and returns at the first hit rather than parsing every line to overwrite a result. A top-level
/// `usage` that looks Anthropic-shaped (dialect mismatch — see
/// `OpenAiUsage::looks_anthropic_shaped`) is skipped, not counted: if every line is mismatched the
/// scan runs off the front and returns `None`.
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
    // Scanned in **reverse**, returning at the first accepted usage. "Last accepted in forward
    // order" and "first accepted in reverse order" select the same line by definition, so this is
    // semantics-preserving — including the tricky cases: a trailing Anthropic-shaped `usage` is
    // rejected in both directions and falls through to an earlier line, a trailing `"usage":null`
    // deserializes to `None` in both and falls through, and the front-truncated first line of a
    // 64 KiB tail is reached last here and fails `strip_prefix`/`from_slice` either way.
    //
    // Forward, the loop ran to completion and overwrote `found` on every hit, so on a 64 KiB tail
    // of ~450 `data:` lines it parsed all 450 and discarded 449. Measured 80.1 µs / 261 allocations
    // (serde_json's scratch `Vec`, one malloc+free per line whose ignored fields nest ≥2 deep —
    // which every `choices[0].delta` chunk does) against 0.155 µs / 0 allocations for this.
    let finder = memchr::memmem::Finder::new(b"usage");
    for line in sse_data_lines_rev(sse) {
        if !might_carry_usage(&finder, line) {
            continue;
        }
        if let Ok(chunk) = serde_json::from_slice::<Chunk>(line) {
            if let Some(u) = chunk.usage {
                if !u.looks_anthropic_shaped() {
                    return Some(Usage::from(u));
                }
            } else if let Some(u) = chunk.response.and_then(|r| r.usage) {
                return Some(Usage::from(u));
            }
        }
    }
    None
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
    // Forward, and genuinely a full pass: input/cache tokens ride on `message_start` at the head
    // while the running output count rides on the last `message_delta`, so unlike `openai_stream`
    // there is no single winning line to stop at. What we *can* skip is the JSON parse for every
    // line that cannot carry a usage block at all — on an Anthropic stream that is every
    // `content_block_delta`, which is nearly the entire tail. Measured 62.3 µs → 9.4 µs on a 64 KiB
    // tail (the `memchr` line split accounts for ~15 µs of that; the pre-filter for the rest).
    let finder = memchr::memmem::Finder::new(b"usage");
    let mut usage = Usage::default();
    let mut saw_any = false;
    for line in sse_lines(sse) {
        let Some(line) = strip_sse_data(line) else {
            continue;
        };
        if !might_carry_usage(&finder, line) {
            continue;
        }
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
        let sse =
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\
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

    /// An OpenAI embeddings response: a big `data` array of float vectors, then `model`, then
    /// `usage` — the real wire shape, and the one that overflows the 64 KiB tail.
    fn embeddings_body(vectors: usize, dims: usize) -> Vec<u8> {
        let vec_json = (0..dims)
            .map(|i| format!("{}", 0.0123456 + i as f64 * 1e-7))
            .collect::<Vec<_>>()
            .join(",");
        let items = (0..vectors)
            .map(|i| format!(r#"{{"object":"embedding","index":{i},"embedding":[{vec_json}]}}"#))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"object":"list","data":[{items}],"model":"text-embedding-3-large","usage":{{"prompt_tokens":812,"total_tokens":812}}}}"#
        )
        .into_bytes()
    }

    /// What `proxy::logging` actually passes the parser: the last `USAGE_TAIL_CAP` bytes.
    fn tail_of(body: &[u8]) -> &[u8] {
        const USAGE_TAIL_CAP: usize = 64 * 1024;
        &body[body.len().saturating_sub(USAGE_TAIL_CAP)..]
    }

    #[test]
    fn oversized_non_streaming_body_still_meters_from_the_truncated_tail() {
        // A body larger than the tail cap reaches the parser front-truncated, so `from_slice` fails
        // at byte 0 and the request metered zero — while tripping `usage_parse_errors_total`, so it
        // looked like a provider wire change rather than our own truncation. OpenAI embeddings hit
        // this routinely: `usage` sits after the `data` array, and eight 3072-dim vectors is ~249 KB.
        let small = embeddings_body(1, 3072);
        assert!(small.len() < 64 * 1024, "control case must fit in the tail");
        assert_eq!(openai_body(tail_of(&small)).unwrap().input_tokens, 812);

        for (vectors, dims) in [(8, 3072), (32, 3072)] {
            let body = embeddings_body(vectors, dims);
            assert!(body.len() > 64 * 1024);
            let from_tail = openai_body(tail_of(&body)).unwrap_or_else(|| {
                panic!("{vectors}x{dims} body ({} B) metered nothing", body.len())
            });
            assert_eq!(from_tail.input_tokens, 812);
            // ...and matches what the untruncated body would have reported.
            assert_eq!(from_tail, openai_body(&body).unwrap());
        }
    }

    #[test]
    fn oversized_anthropic_body_recovers_cache_tokens_from_the_tail() {
        // Same shape on the Anthropic wire: a long `content` array, then `usage` last. Cache tokens
        // are the largest line item in a cached agent workload, so silently zeroing them is the
        // expensive half of this bug.
        let text = "x".repeat(120 * 1024);
        let body = format!(
            r#"{{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{{"type":"text","text":"{text}"}}],"usage":{{"input_tokens":5000,"output_tokens":2500,"cache_read_input_tokens":4000,"cache_creation_input_tokens":100}}}}"#
        )
        .into_bytes();
        assert!(body.len() > 64 * 1024);
        let u = anthropic_body(tail_of(&body)).expect("must meter from a truncated tail");
        assert_eq!(
            (
                u.input_tokens,
                u.output_tokens,
                u.cache_read_tokens,
                u.cache_write_tokens
            ),
            (5000, 2500, 4000, 100)
        );
    }

    #[test]
    fn recovery_does_not_fire_on_a_body_that_parses() {
        // The anchored recovery is only reachable once the whole-document parse has failed, which is
        // what keeps it safe. A well-formed body must take the normal path — including the cases
        // that deliberately return `None` (absent usage, dialect mismatch), which recovery must not
        // resurrect into a bogus reading.
        assert!(openai_body(br#"{"choices":[{"message":{"content":"hi"}}]}"#).is_none());
        assert!(
            openai_body(br#"{"usage":{"input_tokens":100,"output_tokens":50}}"#).is_none(),
            "an Anthropic-shaped usage must stay a dialect mismatch, not be recovered"
        );
        assert!(anthropic_body(br#"{"usage":null}"#).is_none());
        // Still genuinely unparseable ⇒ still nothing to meter.
        assert!(openai_body(b"not json at all").is_none());
        assert!(openai_body(b"{ broken").is_none());
    }

    #[test]
    fn openai_reverse_scan_selects_the_same_line_as_a_forward_scan() {
        // `openai_stream` scans backwards and returns at the first hit; the contract it replaced was
        // "keep going, last one with usage wins". These are the cases where the two could diverge.

        // Two usage chunks: the LAST must win, exactly as the forward loop's overwrite did.
        let two = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n\
                    data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":8}}\n\n\
                    data: [DONE]\n\n";
        let u = openai_stream(two).unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (9, 8));

        // A trailing Anthropic-shaped usage is rejected in both directions and must fall through to
        // the earlier, genuinely OpenAI-shaped one — not terminate the scan.
        let mismatched_last =
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4}}\n\n\
              data: {\"usage\":{\"input_tokens\":100,\"output_tokens\":50}}\n\n";
        let u = openai_stream(mismatched_last).unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (3, 4));

        // An explicit `"usage":null` deserializes to `None` and must also fall through.
        let null_last =
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":6}}\n\n\
                          data: {\"choices\":[],\"usage\":null}\n\n";
        let u = openai_stream(null_last).unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (7, 6));

        // A front-truncated first line (what a 64 KiB tail always starts with) is reached last by the
        // reverse scan and must be skipped, not derail it.
        let truncated = b"pletion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                          data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":12}}\n\n";
        let u = openai_stream(truncated).unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (11, 12));
    }

    #[test]
    fn usage_prefilter_cannot_change_the_answer() {
        // The `memmem` pre-filter skips the JSON parse for lines that don't contain "usage". A false
        // positive (the word in generated content) must still parse correctly and not be mistaken
        // for a usage block...
        let word_in_content =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"token usage is billed\"}}]}\n\n\
              data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3}}\n\n";
        let u = openai_stream(word_in_content).unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (2, 3));

        // ...and a stream that genuinely never carries usage still meters nothing rather than
        // picking up a content line that merely mentions the word.
        let only_the_word =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"usage usage usage\"}}]}\n\n\
                              data: [DONE]\n\n";
        assert!(openai_stream(only_the_word).is_none());

        // Same guard on the Anthropic side, which pre-filters every content_block_delta.
        let ant = b"event: content_block_delta\n\
                    data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"usage\"}}\n\n\
                    event: message_start\n\
                    data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n\
                    event: message_delta\n\
                    data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}\n\n";
        let u = anthropic_stream(ant).unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (20, 15));
    }

    #[test]
    fn sse_lines_matches_split_for_every_payload_bearing_line() {
        // `SseLines` drops the empty element a trailing `\n` would produce; that element can never
        // carry a payload, so the payload sequence must be identical to the old `slice::split`.
        for body in [
            &b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"[..],
            &b"data: {\"a\":1}\n\ndata: {\"b\":2}"[..], // no trailing newline (a truncated tail)
            &b"\n\n\n"[..],
            &b""[..],
            &b"data: [DONE]\n"[..],
        ] {
            let via_split: Vec<&[u8]> = body
                .split(|&b| b == b'\n')
                .filter_map(strip_sse_data)
                .collect();
            let via_memchr: Vec<&[u8]> = sse_lines(body).filter_map(strip_sse_data).collect();
            assert_eq!(
                via_split,
                via_memchr,
                "line iteration diverged for {:?}",
                String::from_utf8_lossy(body)
            );
        }
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
