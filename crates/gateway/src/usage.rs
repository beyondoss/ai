//! Token-usage extraction — the "passive tap" the gateway emits as billing *facts*.
//!
//! We never compute price here (pricing is a closed downstream consumer); we only extract raw
//! token counts. Two shapes per provider: the non-streaming JSON body, and the terminal event of
//! an SSE stream. For streaming we scan the relayed bytes for the usage event but never block the
//! relay on it (see `proxy`).

use crate::route::Dialect;
use arrayvec::ArrayString;
use serde::Deserialize;
use tracing_subscriber::filter::{FilterFn, filter_fn};

/// The `tracing` target billing rows are written on.
pub const USAGE_TARGET: &str = "ai.usage";

/// The filter for the billing-row log layer: the [`USAGE_TARGET`] only, and under no `AI_LOG`
/// level, so an operator turning diagnostics down never stops billing. It still caps the most
/// verbose level it wants at INFO (the rows' level): a filter with no hint makes the whole
/// subscriber's max level TRACE, and `LogTracer` then dispatches every pingora `debug!`/`trace!`
/// record only for every layer to drop it.
pub fn usage_log_filter() -> FilterFn<impl Fn(&tracing::Metadata<'_>) -> bool> {
    filter_fn(|meta| meta.target() == USAGE_TARGET)
        .with_max_level_hint(tracing::level_filters::LevelFilter::INFO)
}

/// A provider-echoed service tier (`default`, `flex`, `priority`, `standard`, …). Inline and `Copy`
/// so [`Usage`] stays `Copy`; a value longer than this, or outside `[a-z0-9_-]`, is dropped rather
/// than written to a billing row.
pub type ServiceTier = ArrayString<16>;

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
    /// Cache writes at the 1-hour TTL (Anthropic `cache_creation.ephemeral_1h_input_tokens`): a
    /// subset of `cache_write_tokens`, priced at 2× input where the 5-minute ones are 1.25×.
    pub cache_write_1h_tokens: u64,
    /// Cache writes caused by breakpoints the gateway added (the client sent no `cache_control`).
    /// Already folded into `input_tokens` and absent from `cache_write_tokens`: the gateway chose to
    /// cache, so they bill at the input rate. Kept so the row reconciles against the provider's
    /// usage, which reports them as cache writes. See [`Usage::bill_gateway_cache_writes`].
    pub gateway_cache_write_tokens: u64,
    /// Server-side tool calls the provider ran and prices per call (Anthropic
    /// `server_tool_use.web_search_requests`). OpenAI reports no such count in `usage`.
    pub server_tool_calls: u64,
    /// The service tier the provider says it served at; `None` when it did not say.
    pub service_tier: Option<ServiceTier>,
    /// Which convention `input_tokens` follows: OpenAI's includes cached (and OpenRouter's
    /// cache-written) tokens, Anthropic's excludes both cache reads and writes. Set by the extractor
    /// that read it; `None` when nothing was read (an estimate), where the request's wire answers.
    pub wire: Option<Dialect>,
}

impl Usage {
    /// Bill this request's cache writes as input: the gateway added the breakpoints that caused
    /// them (`translate::request_with_tools`), so they are its optimization, not the client's
    /// request. `wire` is the convention `input_tokens` follows: Anthropic's excludes cache writes,
    /// so they are added; OpenAI's (OpenRouter's) already includes them. Either way the row's
    /// whole prompt is unchanged and a pricer charges the writes at the input rate.
    pub fn bill_gateway_cache_writes(&mut self, wire: Dialect) {
        let writes = self.cache_write_tokens;
        if writes == 0 {
            return;
        }
        if wire == Dialect::Anthropic {
            self.input_tokens = self.input_tokens.saturating_add(writes);
        }
        self.cache_write_tokens = 0;
        self.cache_write_1h_tokens = 0;
        self.gateway_cache_write_tokens = writes;
    }
}

/// Deserialize a `service_tier` string into a [`ServiceTier`], leniently: anything else — `null`,
/// a number, an object, an over-long or oddly spelled string — reads as `None` rather than failing
/// the parse, which would cost the whole usage block for a field nobody bills from directly.
fn de_service_tier<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<ServiceTier>, D::Error> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = Option<ServiceTier>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a service tier")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
            let ok = v
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
            Ok(ok.then(|| ServiceTier::from(v).ok()).flatten())
        }
        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(V)
        }
        fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok(None)
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut a: A,
        ) -> Result<Self::Value, A::Error> {
            while a
                .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                .is_some()
            {}
            Ok(None)
        }
    }
    d.deserialize_any(V)
}

// Typed views of just the fields we meter. Deserializing into these (rather than a
// `serde_json::Value` DOM) lets serde skip every field we don't read without allocating a node for
// it — no `Map`/`String`/`Number` tree to build and drop per body or per SSE line. Every field is
// `#[serde(default)]` so a missing or partial `usage` block reads as zeros, matching the prior
// pointer-with-`unwrap_or(0)` behavior.

/// OpenAI `usage` block (chat/completions). `prompt`/`completion` map to in/out; cached input rides
/// in `prompt_tokens_details.cached_tokens`; reasoning in `completion_tokens_details.reasoning_tokens`.
/// OpenAI itself has no cache-write concept, but OpenRouter serves Claude on this wire and reports
/// Anthropic's cache writes as `prompt_tokens_details.cache_write_tokens` (priced at 1.25× input).
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
    /// Present on both OpenAI usage shapes and never on Anthropic's: what tells a non-stream
    /// Responses body (`/v1/responses`, `/v1/responses/compact`), whose `usage` uses Anthropic's
    /// `input_tokens`/`output_tokens` names, apart from a dialect-misconfigured Anthropic vendor.
    #[serde(default)]
    total_tokens: Option<u64>,
    /// The Responses API's detail blocks, read only when [`Self::is_responses_shaped`].
    #[serde(default)]
    input_tokens_details: OpenAiResponsesInputDetails,
    #[serde(default)]
    output_tokens_details: OpenAiResponsesOutputDetails,
}

#[derive(Deserialize, Default)]
struct OpenAiPromptDetails {
    /// `Option` so an absent `prompt_tokens_details` (or an absent `cached_tokens` within it — a
    /// DeepSeek response) is distinguishable from an explicit zero, letting
    /// `From<OpenAiUsage>` fall back to `prompt_cache_hit_tokens` only when this is truly missing.
    #[serde(default)]
    cached_tokens: Option<u64>,
    /// OpenRouter's Claude cache writes (see [`OpenAiUsage`]). Absent on OpenAI proper.
    #[serde(default)]
    cache_write_tokens: u64,
}

#[derive(Deserialize, Default)]
struct OpenAiCompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

impl OpenAiUsage {
    /// `input_tokens`/`output_tokens` plus `total_tokens`: a non-stream Responses body.
    fn is_responses_shaped(&self) -> bool {
        self.input_tokens.is_some() && self.output_tokens.is_some() && self.total_tokens.is_some()
    }

    /// See the doc comment on the `input_tokens`/`output_tokens` fields: both present without
    /// `total_tokens` is Anthropic's unambiguous fingerprint (chat/completions never emits these key
    /// names in `usage`, and the Responses API always adds `total_tokens`).
    fn looks_anthropic_shaped(&self) -> bool {
        self.input_tokens.is_some() && self.output_tokens.is_some() && self.total_tokens.is_none()
    }
}

impl From<OpenAiUsage> for Usage {
    fn from(u: OpenAiUsage) -> Self {
        if u.is_responses_shaped() {
            return Usage::from(OpenAiResponsesUsage {
                input_tokens: u.input_tokens.unwrap_or(0),
                output_tokens: u.output_tokens.unwrap_or(0),
                input_tokens_details: u.input_tokens_details,
                output_tokens_details: u.output_tokens_details,
            });
        }
        // OpenAI counts reasoning inside `completion_tokens`; xAI reports it beside it
        // (`total_tokens = prompt + completion + reasoning`) and bills it at the output rate. The
        // arithmetic says which convention a body follows, so the output count is right on both
        // without a per-provider switch.
        let reasoning = u.completion_tokens_details.reasoning_tokens.unwrap_or(0);
        let reasoning_outside = reasoning > 0
            && u.total_tokens
                == Some(
                    u.prompt_tokens
                        .saturating_add(u.completion_tokens)
                        .saturating_add(reasoning),
                );
        Usage {
            input_tokens: u.prompt_tokens,
            // Provider numbers are untrusted: saturate rather than panic (overflow-checks are on).
            output_tokens: u.completion_tokens.saturating_add(if reasoning_outside {
                reasoning
            } else {
                0
            }),
            // `prompt_tokens_details.cached_tokens` (real OpenAI, and OpenAI-compatible providers that
            // populate it) wins when present; DeepSeek's flat `prompt_cache_hit_tokens` is the fallback
            // for when it's entirely absent. Mirrors pi's
            // `rawUsage.prompt_tokens_details?.cached_tokens ?? rawUsage.prompt_cache_hit_tokens ?? 0`.
            cache_read_tokens: u
                .prompt_tokens_details
                .cached_tokens
                .or(u.prompt_cache_hit_tokens)
                .unwrap_or(0),
            cache_write_tokens: u.prompt_tokens_details.cache_write_tokens,
            reasoning_tokens: u.completion_tokens_details.reasoning_tokens,
            wire: Some(Dialect::OpenAi),
            ..Usage::default()
        }
    }
}

/// The Responses API's `usage` block — named `input_tokens`/`output_tokens` (Anthropic-style) rather
/// than `prompt_tokens`/`completion_tokens`. Streamed, it is nested under
/// `response.completed.response.usage` (see `openai_stream`), an envelope Anthropic's wire never
/// carries, so it needs no dialect-mismatch guard there. A non-stream body carries it top-level and
/// reaches it through [`OpenAiUsage::is_responses_shaped`].
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
    /// OpenRouter's Claude cache writes, on its Responses mount (see [`OpenAiUsage`]).
    #[serde(default)]
    cache_write_tokens: u64,
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
            cache_write_tokens: u.input_tokens_details.cache_write_tokens,
            reasoning_tokens: u.output_tokens_details.reasoning_tokens,
            wire: Some(Dialect::OpenAi),
            ..Usage::default()
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
    #[serde(default)]
    cache_creation: AnthropicCacheCreation,
    #[serde(default)]
    server_tool_use: AnthropicServerToolUse,
    #[serde(default, deserialize_with = "de_service_tier")]
    service_tier: Option<ServiceTier>,
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

/// The TTL split of `cache_creation_input_tokens`. Only the 1-hour share is priced differently.
#[derive(Deserialize, Default)]
struct AnthropicCacheCreation {
    #[serde(default)]
    ephemeral_1h_input_tokens: u64,
}

/// Server tools Anthropic ran during the turn. Web search is billed per request; web fetch is not.
#[derive(Deserialize, Default)]
struct AnthropicServerToolUse {
    #[serde(default)]
    web_search_requests: u64,
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
            cache_write_1h_tokens: u.cache_creation.ephemeral_1h_input_tokens,
            server_tool_calls: u.server_tool_use.web_search_requests,
            service_tier: u.service_tier,
            wire: Some(Dialect::Anthropic),
            gateway_cache_write_tokens: 0,
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
        // A root sibling of `usage` on Chat Completions and Responses bodies alike.
        #[serde(default, deserialize_with = "de_service_tier")]
        service_tier: Option<ServiceTier>,
    }
    let (usage, service_tier) = match serde_json::from_slice::<Body>(body) {
        Ok(b) => (b.usage?, b.service_tier),
        // Front-truncated tail of an oversized body — see `recover_trailing_usage`.
        Err(_) => (recover_trailing_usage::<OpenAiUsage>(body)?, None),
    };
    if usage.looks_anthropic_shaped() {
        return None;
    }
    Some(Usage {
        service_tier,
        ..Usage::from(usage)
    })
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

/// Iterate SSE lines newest-first, split on `\n` with `memrchr`.
///
/// `slice::rsplit` is `rposition` of a closure, a scalar walk. The forward iterator was already
/// switched to `memchr` for that reason (17.6 µs vs 2.65 µs over a 64 KiB tail). OpenAI's usage
/// parser is the one that walks **backwards**, and the no-usage tail — the case that cannot stop
/// early — was still on the scalar path.
///
/// A trailing `\n` yields an empty element first, matching `rsplit`. [`strip_sse_data`] rejects
/// that element, so the payload sequence is the same.
struct SseLinesRev<'a> {
    rest: &'a [u8],
}

fn sse_lines_rev(sse: &[u8]) -> SseLinesRev<'_> {
    SseLinesRev { rest: sse }
}

impl<'a> Iterator for SseLinesRev<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.rest.is_empty() {
            return None;
        }
        match memchr::memrchr(b'\n', self.rest) {
            Some(i) => {
                let line = &self.rest[i + 1..];
                self.rest = &self.rest[..i];
                Some(line)
            }
            None => Some(std::mem::take(&mut self.rest)),
        }
    }
}

/// Iterate the raw JSON payloads carried on `data:` lines, **newest first**. Used by the dialects
/// whose answer is "the last usage block wins", so they can stop at the first hit instead of parsing
/// every line to overwrite the result.
fn sse_data_lines_rev(sse: &[u8]) -> impl Iterator<Item = &[u8]> {
    sse_lines_rev(sse).filter_map(strip_sse_data)
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
        #[serde(default, deserialize_with = "de_service_tier")]
        service_tier: Option<ServiceTier>,
    }
    #[derive(Deserialize)]
    struct Chunk {
        usage: Option<OpenAiUsage>,
        response: Option<ResponsesEnvelope>,
        // On every Chat Completions chunk, the usage chunk included.
        #[serde(default, deserialize_with = "de_service_tier")]
        service_tier: Option<ServiceTier>,
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
                    return Some(Usage {
                        service_tier: chunk.service_tier,
                        ..Usage::from(u)
                    });
                }
            } else if let Some(r) = chunk.response
                && let Some(u) = r.usage
            {
                return Some(Usage {
                    service_tier: r.service_tier,
                    ..Usage::from(u)
                });
            }
        }
    }
    // No whole line carried usage. The final event can be bigger than the tail: a Responses
    // `response.completed` echoes the request's instructions and tools ahead of `usage`, so a
    // Codex-sized prompt pushes it past 64 KiB and the tail starts mid-way through it. Only the
    // tail's first line can be front-truncated (the tail ends where the stream does), and `usage`
    // is the last thing in that event, so recover it the way a front-truncated body is recovered.
    let first = &sse[..memchr::memchr(b'\n', sse).unwrap_or(sse.len())];
    recover_trailing_usage::<OpenAiUsage>(first)
        .filter(|u| !u.looks_anthropic_shaped())
        .map(Usage::from)
}

/// Anthropic streaming over a single contiguous buffer. See [`anthropic_stream_parts`], which this
/// delegates to — the proxy uses that form because the facts it needs live at *both* ends of the
/// stream and it only retains the two ends.
pub fn anthropic_stream(sse: &[u8]) -> Option<Usage> {
    anthropic_stream_parts(&[sse])
}

/// Anthropic streaming: input + cache tokens arrive in `message_start.message.usage`; output (and
/// reasoning/thinking tokens) accumulate in `message_delta.usage` (last delta is the cumulative
/// total). A `message_delta` that also reports input or cache counts supersedes `message_start`'s.
/// A `usage` block that looks OpenAI-shaped (dialect mismatch — see
/// `AnthropicUsage::looks_openai_shaped`) is skipped entirely: if every line is mismatched,
/// `saw_any` stays `false` and the function returns `None`.
///
/// Takes **parts** because those two facts sit at opposite ends of the stream and the proxy retains
/// only a bounded head and a bounded tail (a whole stream can be megabytes). `message_start` is the
/// first event, so it is in the head; the final `message_delta` is in the tail. Passing a single
/// buffer — which is what a tail-only tap amounted to — silently zeroed `input_tokens` and both
/// cache counters for any stream longer than the tail, without even tripping the parse-error
/// counter, because `saw_any` still went true off the `message_delta`.
///
/// Parts are scanned in order and may safely overlap: every field is *assigned*, never accumulated,
/// so a short response whose head and tail cover the same bytes reads the same as one that doesn't.
pub fn anthropic_stream_parts(parts: &[&[u8]]) -> Option<Usage> {
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
    for part in parts {
        for line in sse_lines(part) {
            let Some(line) = strip_sse_data(line) else {
                continue;
            };
            if !might_carry_usage(&finder, line) {
                continue;
            }
            let Ok(chunk) = serde_json::from_slice::<Chunk>(line) else {
                continue;
            };
            if let Some(u) = chunk.message.and_then(|m| m.usage)
                && !u.looks_openai_shaped()
            {
                usage.input_tokens = u.input_tokens;
                usage.cache_read_tokens = u.cache_read_input_tokens;
                usage.cache_write_tokens = u.cache_creation_input_tokens;
                usage.cache_write_1h_tokens = u.cache_creation.ephemeral_1h_input_tokens;
                usage.service_tier = u.service_tier;
                saw_any = true;
            }
            if let Some(u) = chunk.usage
                && !u.looks_openai_shaped()
            {
                // message_delta carries the running output token count — and, cumulatively, input
                // and cache counts too, which grow past `message_start`'s when a server tool (web
                // search) feeds results back mid-turn. Present wins; absent (zero, by
                // `serde(default)`) keeps what `message_start` said.
                if u.output_tokens > 0 {
                    usage.output_tokens = u.output_tokens;
                }
                if u.input_tokens > 0 {
                    usage.input_tokens = u.input_tokens;
                }
                if u.cache_read_input_tokens > 0 {
                    usage.cache_read_tokens = u.cache_read_input_tokens;
                }
                if u.cache_creation_input_tokens > 0 {
                    usage.cache_write_tokens = u.cache_creation_input_tokens;
                }
                if u.cache_creation.ephemeral_1h_input_tokens > 0 {
                    usage.cache_write_1h_tokens = u.cache_creation.ephemeral_1h_input_tokens;
                }
                // Cumulative, and only ever on the delta: the searches run mid-turn.
                if u.server_tool_use.web_search_requests > 0 {
                    usage.server_tool_calls = u.server_tool_use.web_search_requests;
                }
                if let Some(rt) = u.output_tokens_details.thinking_tokens {
                    usage.reasoning_tokens = Some(rt);
                }
                saw_any = true;
            }
        }
    }
    saw_any.then_some(Usage {
        wire: Some(Dialect::Anthropic),
        ..usage
    })
}

// --- Estimates for a stream cut short -----------------------------------------------------------
//
// A stream's usage block is its *last* event. When the client hangs up first (a cancelled agent
// turn) or the upstream dies mid-stream, the block never arrives — but the provider still bills us
// for everything it generated before it noticed. Emitting zero there was a free-generation hole:
// stream a long answer, disconnect one event before the end, pay nothing.
//
// What follows estimates those rows instead. Both divisors were measured against real providers
// (2026-09-30) and then rounded in the direction that **under**-counts, because an estimate is a
// bill the customer cannot check against anything:
//
// | Measured                               | OpenAI (gpt-4o-mini) | Claude Haiku 4.5 | Claude Sonnet 5 |
// | -------------------------------------- | -------------------- | ---------------- | --------------- |
// | JSON request-body bytes / input token  | 4.63                 | 3.95             | 3.00            |
// | streamed text bytes / output token     | 4.32                 | 3.73             | 2.69            |
// | output tokens / delta event            | 1.00                 | 2.95             | 4.85            |
//
// Hidden reasoning is invisible to both estimates — an OpenAI reasoning model, or Claude with
// thinking display omitted, is billed for thinking the stream never shows. The row is flagged
// `usage_estimated` so a downstream consumer can tell estimated rows from reported ones.

/// JSON-escaped request-body bytes per input token. 5 under-counts every provider measured above.
const INPUT_BYTES_PER_TOKEN: u64 = 5;

/// Streamed text bytes per output token, ×10 so the math stays in integers: 4.5.
const OUTPUT_TEXT_BYTES_PER_TOKEN_X10: u64 = 45;

/// Where binary payloads start in a request body — the bytes that are not text. An inline image is
/// ~1 MB of base64 and ~1–2K tokens, so counting it as text would bill one picture as a novel.
///
/// - `;base64,` — a data URI (`data:image/png;base64,…`): OpenAI `image_url`, Responses
///   `input_image` / `input_file`. The payload runs to the string's closing quote.
/// - `"data":` — the value of any `data` key: Anthropic `base64` image and document sources,
///   OpenAI `input_audio`. The payload is the next string. A `data` key holding something else (a
///   tool schema property, say) is skipped too, which only ever under-counts.
///
/// Both are at most 8 bytes, so a 7-byte carry catches either split across two chunks.
const PAYLOAD_MARKERS: [&[u8]; 2] = [b";base64,", br#""data":"#];
const MARKER_CARRY: usize = 7;

static PAYLOAD_FINDER: std::sync::LazyLock<aho_corasick::AhoCorasick> =
    std::sync::LazyLock::new(|| {
        aho_corasick::AhoCorasick::new(PAYLOAD_MARKERS).unwrap_or_else(|_| unreachable!())
    });

/// Running count of a request body's text bytes, fed chunk by chunk as the body streams past.
///
/// Excludes binary payloads (see [`PAYLOAD_MARKERS`]). A marker split across two chunks is still
/// found: the last few bytes of each chunk are carried into the next. Nothing is buffered — 12
/// bytes of state however large the body, because it sits in `RequestCtx`, which is touched once
/// per response chunk.
#[derive(Default, Clone, Copy)]
pub struct InputTally {
    text_bytes: u32,
    /// Low bits ([`CARRY_LEN`]): how many of `carry`'s bytes are live. High bits: the phase —
    /// [`AWAIT_OPEN`] (after `"data":`, before its string opens) or [`IN_PAYLOAD`].
    state: u8,
    carry: [u8; MARKER_CARRY],
}

const CARRY_LEN: u8 = 0x0f;
const AWAIT_OPEN: u8 = 0x40;
const IN_PAYLOAD: u8 = 0x80;

impl InputTally {
    pub fn feed(&mut self, mut chunk: &[u8]) {
        loop {
            if self.state & (AWAIT_OPEN | IN_PAYLOAD) != 0 {
                // Base64 has no escapes, so the payload ends at the next quote; after `"data":`,
                // the next quote opens the payload.
                let Some(q) = memchr::memchr(b'"', chunk) else {
                    // Whitespace between `"data":` and its string is text; a payload is not.
                    if self.state & AWAIT_OPEN != 0 {
                        self.count(chunk.len());
                    }
                    return;
                };
                if self.state & AWAIT_OPEN != 0 {
                    self.count(q + 1);
                    self.state = IN_PAYLOAD;
                    chunk = &chunk[q + 1..];
                    continue;
                }
                self.state = 0;
                chunk = &chunk[q..];
            }
            let found = self.take_split_marker(chunk).or_else(|| {
                PAYLOAD_FINDER
                    .find(chunk)
                    .map(|m| (m.end(), m.pattern().as_usize()))
            });
            match found {
                Some((end, pattern)) => {
                    self.count(end);
                    chunk = &chunk[end..];
                    self.state = if pattern == 0 { IN_PAYLOAD } else { AWAIT_OPEN };
                }
                None => {
                    self.count(chunk.len());
                    let keep = chunk.len().min(MARKER_CARRY);
                    self.carry[..keep].copy_from_slice(&chunk[chunk.len() - keep..]);
                    self.state = keep as u8;
                    return;
                }
            }
        }
    }

    /// Where a marker that *began* in the previous chunk ends in this one, if it did, and which.
    fn take_split_marker(&mut self, chunk: &[u8]) -> Option<(usize, usize)> {
        // Only reached outside a payload, where `state` is exactly the carry length.
        let carried = usize::from(std::mem::take(&mut self.state) & CARRY_LEN);
        if carried == 0 {
            return None;
        }
        let mut joined = [0u8; 2 * MARKER_CARRY];
        let n = chunk.len().min(MARKER_CARRY);
        joined[..carried].copy_from_slice(&self.carry[..carried]);
        joined[carried..carried + n].copy_from_slice(&chunk[..n]);
        let m = PAYLOAD_FINDER.find(&joined[..carried + n])?;
        // A marker wholly inside `chunk` is the ordinary search's to find.
        (m.start() < carried).then(|| (m.end() - carried, m.pattern().as_usize()))
    }

    fn count(&mut self, n: usize) {
        self.text_bytes = self
            .text_bytes
            .saturating_add(u32::try_from(n).unwrap_or(u32::MAX));
    }

    /// Estimated input tokens. Deliberately low; see the table above.
    pub fn estimate_tokens(&self) -> u64 {
        u64::from(self.text_bytes) / INPUT_BYTES_PER_TOKEN
    }
}

/// Whether an Anthropic stream reached its `message_delta` — the event carrying the final output
/// count. `message_start` alone parses as usage too (input and cache tokens), so a parse succeeding
/// is not the same as the stream finishing.
///
/// The check is structural: an `event: message_delta` line, or a `"type":"message_delta"` member.
/// Generated text cannot forge either — inside a JSON string a newline is `\n` and a quote is
/// `\"` — so a model writing the words `message_delta` does not finish a cut-short stream.
pub fn anthropic_stream_finished(tail: &[u8]) -> bool {
    memchr::memmem::find(tail, b"\nevent: message_delta").is_some()
        || memchr::memmem::find(tail, br#""type":"message_delta""#).is_some()
        || memchr::memmem::find(tail, br#""type": "message_delta""#).is_some()
}

/// Whether a stream carried an error event: Anthropic's `{"type":"error","error":{…}}`, OpenAI's
/// `{"error":{…}}`, OpenRouter's mid-stream `"error":{…}` on a chunk. A `null` error (Responses
/// events carry `"error":null`) is not one, which is why the object brace is part of the needle.
/// Generated text cannot match: inside a JSON string the quotes are escaped.
pub fn stream_carried_error(tail: &[u8]) -> bool {
    memchr::memmem::find(tail, br#""error":{"#).is_some()
}

/// Estimate the output tokens of a stream that ended before its usage block.
///
/// Only the tail is retained, so this measures the tail — delta events and generated text per
/// byte of stream — and scales both up to `total_bytes`, what was relayed over the whole stream.
/// Events in one stream are homogeneous, so the tail is a fair sample; a stream shorter than the
/// tail is measured whole. Scaling by bytes rather than counting events on the relay path keeps
/// the per-chunk cost to one add: counting `data:` with `memmem` on every managed chunk measured
/// +9.5% on a 600 KiB Anthropic stream. The estimate is the larger of "one token per delta
/// event" (exact on OpenAI) and "text bytes / 4.5" (the floor on providers that batch several tokens
/// into one event), each of which under-counts on its own side.
///
/// Runs once, on the cut-short path only, so it parses each line into a `Value` rather than
/// maintaining a typed view per wire.
pub fn estimate_stream_output(tail: &[u8], total_bytes: u64) -> u64 {
    let (mut events, mut deltas, mut text) = (0u64, 0u64, 0u64);
    for line in sse_lines(tail) {
        let Some(payload) = strip_sse_data(line) else {
            continue;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(payload) else {
            continue;
        };
        events += 1;
        let n = delta_text_len(&v);
        if n > 0 {
            deltas += 1;
            text += n;
        }
    }
    if events == 0 || tail.is_empty() {
        return 0;
    }
    // The bytes before the tail are extrapolated at 90%: a stream opens with heavier preamble
    // events (a Responses stream's `response.created` carries the whole response object, ~4× a
    // delta), so the unseen head holds fewer deltas per byte than the tail. Measured +7% over on a
    // Responses stream cut at 50% before this; the discount brings it under.
    let sampled = tail.len() as u64;
    let unseen = total_bytes.saturating_sub(sampled);
    let scale = |n: u64| {
        n.saturating_add(n.saturating_mul(unseen).saturating_mul(9) / sampled.saturating_mul(10))
    };
    let by_events = scale(deltas);
    let by_text = scale(text).saturating_mul(10) / OUTPUT_TEXT_BYTES_PER_TOKEN_X10;
    by_events.max(by_text)
}

/// Estimate the output tokens of a non-stream body cut off before its `usage` block.
///
/// Counts the bytes inside JSON string **values** in the tail — the generated text, plus a few
/// short ids — and not keys or structure, which are envelope; scales them up to `total_bytes`
/// relayed, as [`estimate_stream_output`] does, and divides by the text divisor. A tail that starts
/// mid-string reads its first segment inverted, which moves the estimate by one string's length.
/// Runs once, on a failure path.
pub fn estimate_body_output(tail: &[u8], total_bytes: u64) -> u64 {
    let n = tail.len();
    let (mut text, mut i) = (0u64, 0usize);
    while let Some(open) = tail.get(i..).and_then(|t| memchr::memchr(b'"', t)) {
        let start = i + open + 1;
        // The closing quote, stepping over escapes.
        let mut end = start;
        loop {
            match tail
                .get(end..)
                .and_then(|t| memchr::memchr2(b'"', b'\\', t))
            {
                Some(k) if tail[end + k] == b'\\' => end += k + 2,
                Some(k) => {
                    end += k;
                    break;
                }
                None => {
                    end = n;
                    break;
                }
            }
        }
        let end = end.min(n);
        // A key is followed by `:`; a value is not.
        let is_key = tail
            .get(end + 1..)
            .and_then(|t| t.iter().find(|b| !b.is_ascii_whitespace()))
            == Some(&b':');
        if !is_key {
            text += (end - start) as u64;
        }
        i = end + 1;
    }
    if n == 0 {
        return 0;
    }
    let sampled = n as u64;
    let text =
        text.saturating_add(text.saturating_mul(total_bytes.saturating_sub(sampled)) / sampled);
    text.saturating_mul(10) / OUTPUT_TEXT_BYTES_PER_TOKEN_X10
}

/// Generated text carried by one stream event, across the three wires: Chat Completions
/// (`choices[0].delta` content, reasoning, tool-call arguments), Messages (`delta` text, thinking,
/// tool-input JSON), and Responses (a string `delta`).
fn delta_text_len(v: &serde_json::Value) -> u64 {
    use serde_json::Value;
    let len = |x: Option<&Value>| x.and_then(Value::as_str).map_or(0, |s| s.len() as u64);
    if let Some(d) = v.pointer("/choices/0/delta") {
        let calls = d
            .get("tool_calls")
            .and_then(Value::as_array)
            .map_or(0, |calls| {
                calls
                    .iter()
                    .map(|c| len(c.pointer("/function/arguments")))
                    .sum()
            });
        return len(d.get("content"))
            + len(d.get("reasoning"))
            + len(d.get("reasoning_content"))
            + calls;
    }
    match v.get("delta") {
        Some(Value::String(s)) => s.len() as u64,
        Some(d @ Value::Object(_)) => {
            len(d.get("text")) + len(d.get("thinking")) + len(d.get("partial_json"))
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The billing layer takes no `AI_LOG` filter, but it must still tell `tracing` the most verbose
    /// level it wants. Without a hint the whole subscriber's max level is TRACE, and `LogTracer`
    /// dispatches every pingora `debug!`/`trace!` record only for every layer to drop it.
    /// claim: BIL-4
    /// defect: D93
    #[test]
    fn the_billing_log_layer_caps_the_max_level_at_info() {
        use tracing::Subscriber as _;
        use tracing_subscriber::layer::{Layer as _, SubscriberExt as _};
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::sink)
                .with_filter(usage_log_filter()),
        );
        assert_eq!(
            subscriber.max_level_hint(),
            Some(tracing::level_filters::LevelFilter::INFO)
        );
    }

    // --- cut-short estimates ---

    fn tally(chunks: &[&[u8]]) -> InputTally {
        let mut t = InputTally::default();
        for c in chunks {
            t.feed(c);
        }
        t
    }

    #[test]
    fn input_tally_counts_text_and_skips_base64_payloads() {
        let b64 = "A".repeat(10_000);
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{b64}"}}}},{{"type":"text","text":"hi"}}]}}]}}"#
        );
        let whole = tally(&[body.as_bytes()]);
        assert_eq!(
            u64::from(whole.text_bytes),
            (body.len() - b64.len()) as u64,
            "every byte but the base64 payload is text"
        );
        // Any split point — including inside the marker and inside the payload — agrees.
        for cut in [1, 60, 95, 98, 100, 101, 500, body.len() - 5] {
            let (a, b) = body.as_bytes().split_at(cut);
            assert_eq!(
                tally(&[a, b]).text_bytes,
                whole.text_bytes,
                "split at {cut}"
            );
        }
    }

    #[test]
    fn body_output_estimate_counts_string_values_only() {
        let content = "word ".repeat(2000);
        let full = format!(
            r#"{{"id":"chatcmpl-1","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":"{content}"}}}}],"usage":{{"prompt_tokens":40,"completion_tokens":2000}}}}"#
        );
        let half = &full.as_bytes()[..full.len() / 2];
        let est = estimate_body_output(half, half.len() as u64);
        // ~4.9 KB of "word word …" relayed: about 1100 tokens at 4.5 bytes each. The half that
        // arrived, not the whole 2000 the provider reports, and none of the envelope's keys.
        assert!((900..1200).contains(&est), "{est}");
        // A tail of a larger body scales up.
        let scaled = estimate_body_output(half, 2 * half.len() as u64);
        assert!(scaled >= 2 * est - 1, "{scaled} vs {est}");
        assert_eq!(estimate_body_output(b"", 0), 0);
        assert_eq!(estimate_body_output(br#"{"a":"b\"#, 10), 0);
    }

    #[test]
    fn input_tally_is_twelve_bytes() {
        assert_eq!(std::mem::size_of::<InputTally>(), 12);
    }

    /// Anthropic's image source is `{"type":"base64","data":"…"}` — no data URI — and OpenAI's
    /// `input_audio` is `{"data":"…"}`. Both must be skipped, compact or spaced, at any split.
    #[test]
    fn input_tally_skips_data_key_payloads() {
        let b64 = "B".repeat(10_000);
        for body in [
            format!(
                r#"{{"messages":[{{"role":"user","content":[{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{b64}"}}}},{{"type":"text","text":"hi"}}]}}]}}"#
            ),
            format!(r#"{{"input_audio": {{"data": "{b64}", "format": "wav"}}}}"#),
        ] {
            let whole = tally(&[body.as_bytes()]);
            assert_eq!(
                u64::from(whole.text_bytes),
                (body.len() - b64.len()) as u64,
                "{}",
                &body[..60]
            );
            for cut in 1..body.len().min(200) {
                let (a, b) = body.as_bytes().split_at(cut);
                assert_eq!(
                    tally(&[a, b]).text_bytes,
                    whole.text_bytes,
                    "split at {cut}"
                );
            }
        }
    }

    #[test]
    fn input_tally_without_images_is_the_body_length() {
        let body = br#"{"model":"m","messages":[{"role":"user","content":"hello there"}]}"#;
        let t = tally(&[&body[..10], &body[10..]]);
        assert_eq!(u64::from(t.text_bytes), body.len() as u64);
        assert_eq!(
            t.estimate_tokens(),
            body.len() as u64 / INPUT_BYTES_PER_TOKEN
        );
    }

    fn openai_delta(text: &str) -> String {
        format!(
            "data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{text}\"}}}}]}}\n\n"
        )
    }

    fn anthropic_delta(text: &str) -> String {
        format!(
            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{text}\"}}}}\n\n"
        )
    }

    /// OpenAI sends one token per event, so the delta-event count is the estimate — and a stream
    /// longer than the tail scales up by the events relayed, not just the ones retained.
    #[test]
    fn openai_estimate_is_one_token_per_delta_event() {
        let tail: String = (0..100).map(|_| openai_delta(" tok")).collect();
        let n = tail.len() as u64;
        assert_eq!(estimate_stream_output(tail.as_bytes(), n), 100);
        assert_eq!(
            estimate_stream_output(tail.as_bytes(), 10 * n),
            100 + 900 * 9 / 10,
            "the tail is a sample scaled up to the relayed bytes, the unseen part at 90%"
        );
    }

    /// Providers that batch several tokens into one event are floored by text length instead.
    #[test]
    fn batched_deltas_are_estimated_from_text_length() {
        let chunk = "x".repeat(45);
        let tail: String = (0..10).map(|_| anthropic_delta(&chunk)).collect();
        // 10 events × 45 bytes = 450 bytes of text = 100 tokens at 4.5 bytes/token.
        assert_eq!(
            estimate_stream_output(tail.as_bytes(), tail.len() as u64),
            100
        );
    }

    #[test]
    fn responses_and_tool_call_deltas_count_as_output() {
        let tail = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{\\\"a\\\"\"}}]}}]}\n\n",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"b\"}}\n\n",
        );
        assert_eq!(
            estimate_stream_output(tail.as_bytes(), tail.len() as u64),
            3
        );
    }

    #[test]
    fn empty_or_unparseable_tails_estimate_zero() {
        assert_eq!(estimate_stream_output(b"", 5_000), 0);
        assert_eq!(estimate_stream_output(b"data: [DONE]\n\n", 5_000), 0);
        assert_eq!(estimate_stream_output(b"garbage\n", 5_000), 0);
    }

    #[test]
    fn anthropic_stream_finishes_at_message_delta() {
        let started = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\n";
        assert!(!anthropic_stream_finished(started.as_bytes()));
        let finished = format!(
            "{started}event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":5}}}}\n\n"
        );
        assert!(anthropic_stream_finished(finished.as_bytes()));
        let text = format!(
            "{started}event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"delta\":{{\"type\":\"text_delta\",\"text\":\"event: message_delta \\\"type\\\":\\\"message_delta\\\"\"}}}}\n\n"
        );
        assert!(!anthropic_stream_finished(text.as_bytes()), "{text}");
    }

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
                wire: Some(Dialect::OpenAi),
                ..Usage::default()
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
                wire: Some(Dialect::OpenAi),
                ..Usage::default()
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
                wire: Some(Dialect::Anthropic),
                ..Usage::default()
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

    /// A realistic Anthropic stream: `message_start` (input + cache), `n` text deltas, then the
    /// terminal `message_delta` (output).
    fn anthropic_sse(deltas: usize) -> Vec<u8> {
        let mut s = String::from(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":5000,\"output_tokens\":1,\"cache_read_input_tokens\":4000,\"cache_creation_input_tokens\":100}}}\n\n",
        );
        for i in 0..deltas {
            s.push_str(&format!(
                "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\" token{i}\"}}}}\n\n"
            ));
        }
        s.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2500}}\n\n");
        s.into_bytes()
    }

    #[test]
    fn anthropic_stream_reads_input_tokens_from_the_head_not_the_tail() {
        // `message_start` — which carries input_tokens and BOTH cache counters — is the first event
        // on the wire, but the proxy only retained the last 64 KiB. Past ~500 output tokens (about
        // 62 KiB of `content_block_delta` framing) it was compacted away and those three fields
        // billed zero, while `saw_any` still went true off the `message_delta` so nothing flagged
        // it. Input and cache-read are the *largest* line items in a cached agent workload.
        const HEAD: usize = 8 * 1024;
        const TAIL: usize = 64 * 1024;
        let expected = Usage {
            input_tokens: 5000,
            output_tokens: 2500,
            cache_read_tokens: 4000,
            cache_write_tokens: 100,
            reasoning_tokens: None,
            wire: Some(Dialect::Anthropic),
            ..Usage::default()
        };

        for deltas in [10, 500, 5000] {
            let sse = anthropic_sse(deltas);
            let head = &sse[..HEAD.min(sse.len())];
            let tail = &sse[sse.len().saturating_sub(TAIL)..];

            assert_eq!(
                anthropic_stream_parts(&[head, tail]).unwrap(),
                expected,
                "head+tail must meter fully at {deltas} deltas ({} B)",
                sse.len()
            );
            // The whole buffer, when it is small enough to keep, must agree.
            assert_eq!(anthropic_stream(&sse).unwrap(), expected);
        }

        // ...and the tail *alone* is exactly what used to be wrong, which is what makes the head
        // load-bearing rather than belt-and-braces.
        let big = anthropic_sse(5000);
        let tail_only = anthropic_stream(&big[big.len() - TAIL..]).unwrap();
        assert_eq!(
            (
                tail_only.input_tokens,
                tail_only.cache_read_tokens,
                tail_only.cache_write_tokens
            ),
            (0, 0, 0),
            "a tail-only tap is expected to lose these — that is the bug the head buffer fixes"
        );
        assert_eq!(tail_only.output_tokens, 2500);
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
            let via_rsplit: Vec<&[u8]> = body
                .rsplit(|&b| b == b'\n')
                .filter_map(strip_sse_data)
                .collect();
            let via_memrchr: Vec<&[u8]> = sse_lines_rev(body).filter_map(strip_sse_data).collect();
            assert_eq!(
                via_rsplit,
                via_memrchr,
                "reverse line iteration diverged for {:?}",
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
                wire: Some(Dialect::OpenAi),
                ..Usage::default()
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
                wire: Some(Dialect::OpenAi),
                ..Usage::default()
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
                wire: Some(Dialect::Anthropic),
                ..Usage::default()
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
                wire: Some(Dialect::Anthropic),
                ..Usage::default()
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
                wire: Some(Dialect::OpenAi),
                ..Usage::default()
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
                wire: Some(Dialect::OpenAi),
                ..Usage::default()
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

    /// xAI reports reasoning beside `completion_tokens` (total = prompt + completion + reasoning)
    /// and bills it as output; OpenAI counts it inside. Body shape taken from a live grok-4.3 call.
    ///
    /// claim: BIL-9
    /// defect: D23
    #[test]
    fn reasoning_reported_beside_completion_tokens_is_billed_as_output() {
        let xai = br#"{"usage":{"prompt_tokens":202,"completion_tokens":7,"total_tokens":376,
            "prompt_tokens_details":{"cached_tokens":192},
            "completion_tokens_details":{"reasoning_tokens":167}}}"#;
        let u = openai_body(xai).unwrap();
        assert_eq!(u.output_tokens, 174, "7 visible + 167 reasoning");
        assert_eq!(u.reasoning_tokens, Some(167));
        // OpenAI: reasoning is already inside completion_tokens (total = prompt + completion).
        let openai = br#"{"usage":{"prompt_tokens":20,"completion_tokens":300,"total_tokens":320,
            "completion_tokens_details":{"reasoning_tokens":250}}}"#;
        assert_eq!(openai_body(openai).unwrap().output_tokens, 300);
        // Streaming terminal chunk, same rule.
        let sse = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":202,\"completion_tokens\":7,\"total_tokens\":376,\"completion_tokens_details\":{\"reasoning_tokens\":167}}}\n\ndata: [DONE]\n\n";
        assert_eq!(openai_stream(sse).unwrap().output_tokens, 174);
    }

    #[test]
    fn non_stream_responses_usage_bills() {
        // A non-stream `/v1/responses` (or `/v1/responses/compact`) body: Anthropic's key names, plus
        // the `total_tokens` and detail blocks Anthropic never sends. Shape verified live against
        // `/v1/responses/compact`.
        let body = br#"{"object":"response.compaction","output":[],"usage":{"input_tokens":121,
            "input_tokens_details":{"cache_write_tokens":0,"cached_tokens":64},"output_tokens":40,
            "output_tokens_details":{"reasoning_tokens":12},"total_tokens":161}}"#;
        assert_eq!(
            openai_body(body).unwrap(),
            Usage {
                input_tokens: 121,
                output_tokens: 40,
                cache_read_tokens: 64,
                cache_write_tokens: 0,
                reasoning_tokens: Some(12),
                wire: Some(Dialect::OpenAi),
                ..Usage::default()
            }
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
    fn priced_variants_and_service_tier_are_read_on_every_wire() {
        let body =
            br#"{"usage":{"input_tokens":50,"output_tokens":10,"cache_creation_input_tokens":2000,
            "cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":2000},
            "server_tool_use":{"web_search_requests":3},"service_tier":"priority"}}"#;
        let u = anthropic_body(body).unwrap();
        assert_eq!((u.cache_write_1h_tokens, u.server_tool_calls), (2000, 3));
        assert_eq!(u.service_tier.as_deref(), Some("priority"));
        assert_eq!(u.wire, Some(Dialect::Anthropic));

        // Streamed: the tier and 1h writes on `message_start`, the search count on the delta.
        let sse = b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"cache_creation\":{\"ephemeral_1h_input_tokens\":7},\"service_tier\":\"standard\"}}}\n\n\
data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":4,\"server_tool_use\":{\"web_search_requests\":2}}}\n\n";
        let u = anthropic_stream(sse).unwrap();
        assert_eq!((u.cache_write_1h_tokens, u.server_tool_calls), (7, 2));
        assert_eq!(u.service_tier.as_deref(), Some("standard"));

        let chat = br#"{"service_tier":"flex","usage":{"prompt_tokens":3,"completion_tokens":1}}"#;
        let u = openai_body(chat).unwrap();
        assert_eq!(u.service_tier.as_deref(), Some("flex"));
        assert_eq!(u.wire, Some(Dialect::OpenAi));
        let responses = b"data: {\"type\":\"response.completed\",\"response\":{\"service_tier\":\"default\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4}}}\n\n";
        assert_eq!(
            openai_stream(responses).unwrap().service_tier.as_deref(),
            Some("default")
        );

        // A malformed tier never costs the usage block.
        for tier in [
            r#"7"#,
            r#"{"a":[1]}"#,
            r#""Has Caps""#,
            r#""way-too-long-for-a-tier""#,
        ] {
            let body = format!(
                r#"{{"service_tier":{tier},"usage":{{"prompt_tokens":3,"completion_tokens":1}}}}"#
            );
            let u = openai_body(body.as_bytes()).unwrap();
            assert_eq!((u.input_tokens, u.service_tier), (3, None), "{tier}");
        }
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

/// Verify phase 0, billing: parser-level reproductions. Each asserts the CORRECT behavior.
#[cfg(test)]
mod verify_billing {
    use super::*;

    /// The last 64 KiB of a Responses stream whose `response.completed` is bigger than that — what
    /// `proxy::logging` hands `openai_stream` (see `USAGE_TAIL_CAP`).
    /// claim: BIL-15
    /// defect: D20
    #[test]
    fn openai_stream_reads_usage_from_a_final_event_larger_than_the_tail() {
        let instructions = "You are a coding agent. ".repeat(4 * 1024);
        let sse = format!(
            "event: response.output_text.delta\n\
             data: {{\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}}\n\n\
             event: response.completed\n\
             data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",\"instructions\":\"{instructions}\",\"usage\":{{\"input_tokens\":24000,\"output_tokens\":5,\"total_tokens\":24005}}}}}}\n\n"
        );
        let tail = &sse.as_bytes()[sse.len() - 64 * 1024..];
        let u = openai_stream(tail).expect("usage is in the tail, just not on a whole line");
        assert_eq!((u.input_tokens, u.output_tokens), (24000, 5));
    }

    /// OpenRouter's Chat Completions usage carries Claude cache writes as
    /// `prompt_tokens_details.cache_write_tokens`.
    /// claim: BIL-8
    /// defect: D21
    #[test]
    fn openai_body_reads_openrouter_cache_write_tokens() {
        let body = br#"{"usage":{"prompt_tokens":120,"completion_tokens":7,"total_tokens":127,"prompt_tokens_details":{"cached_tokens":10,"cache_write_tokens":50}}}"#;
        let u = openai_body(body).unwrap();
        assert_eq!(u.cache_read_tokens, 10);
        assert_eq!(u.cache_write_tokens, 50);
    }

    /// Anthropic's `message_delta.usage` is cumulative and, after server tool use, carries larger
    /// input and cache counts than `message_start`. The final counts win.
    /// claim: BIL-8
    /// defect: D22
    #[test]
    fn anthropic_stream_takes_cumulative_input_from_message_delta() {
        let sse = b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"output_tokens\":1}}}\n\n\
data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":2500,\"cache_read_input_tokens\":800,\"cache_creation_input_tokens\":0,\"output_tokens\":300}}\n\n";
        let u = anthropic_stream(sse).unwrap();
        assert_eq!(
            (u.input_tokens, u.cache_read_tokens, u.output_tokens),
            (2500, 800, 300)
        );
    }

    /// Gateway-caused cache writes move to input on either wire, and the whole prompt is kept:
    /// Anthropic's `input_tokens` excludes writes (so they are added), OpenAI's already holds them.
    #[test]
    fn gateway_cache_writes_bill_as_input_on_both_wires() {
        let base = Usage {
            input_tokens: 50,
            cache_read_tokens: 300,
            cache_write_tokens: 2000,
            cache_write_1h_tokens: 0,
            ..Default::default()
        };
        let mut a = base;
        a.bill_gateway_cache_writes(Dialect::Anthropic);
        assert_eq!(
            (a.input_tokens, a.cache_read_tokens, a.cache_write_tokens),
            (2050, 300, 0)
        );
        assert_eq!(a.gateway_cache_write_tokens, 2000);
        let mut o = Usage {
            input_tokens: 2350,
            ..base
        };
        o.bill_gateway_cache_writes(Dialect::OpenAi);
        assert_eq!((o.input_tokens, o.cache_write_tokens), (2350, 0));
        assert_eq!(o.gateway_cache_write_tokens, 2000);
        // No writes, nothing to move.
        let mut none = Usage {
            cache_write_tokens: 0,
            ..base
        };
        none.bill_gateway_cache_writes(Dialect::Anthropic);
        assert_eq!(
            none,
            Usage {
                cache_write_tokens: 0,
                ..base
            }
        );
    }

    /// Provider token counts are untrusted numbers. A garbage or hostile usage block whose sums
    /// overflow a `u64` must still bill (saturated), never panic: release builds keep
    /// overflow-checks on, and a panic in logging loses the billing row.
    ///
    /// claim: BIL-4, REL-17
    /// defect: D87
    #[test]
    fn overflowing_provider_token_counts_saturate_instead_of_panicking() {
        let max = u64::MAX;
        let body = format!(
            r#"{{"usage":{{"prompt_tokens":{max},"completion_tokens":{max},"total_tokens":3,
            "completion_tokens_details":{{"reasoning_tokens":{max}}}}}}}"#
        );
        let u = std::panic::catch_unwind(|| openai_body(body.as_bytes()))
            .expect("must not panic")
            .unwrap();
        assert_eq!(u.input_tokens, max);
        assert_eq!(u.output_tokens, max, "saturated, not wrapped");
        let sse = format!(
            "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":1,\"completion_tokens\":{max},\"total_tokens\":{max},\"completion_tokens_details\":{{\"reasoning_tokens\":5}}}}}}\n\ndata: [DONE]\n\n"
        );
        let u = std::panic::catch_unwind(|| openai_stream(sse.as_bytes()))
            .expect("must not panic")
            .unwrap();
        assert_eq!(u.output_tokens, max);
        // The estimators scale byte counts by multiplication; a huge relayed total saturates.
        let tail = br#"data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}

"#;
        assert!(std::panic::catch_unwind(|| estimate_stream_output(tail, max)).is_ok());
        assert!(
            std::panic::catch_unwind(|| estimate_body_output(br#"{"a":"hello"}"#, max)).is_ok()
        );
    }
}
