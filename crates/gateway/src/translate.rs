//! Chat Completions ↔ Messages ↔ Responses translation for a managed catalog walk.
//!
//! Triggered when the inbound path names Chat Completions, Messages, or Responses (including the
//! `/auto` suffix) and the catalog row speaks a different one of those three. Same-wire walks stay
//! a byte relay — including `/{provider}/v1/responses` — except that a catalog walk's Responses
//! body loses the reasoning items this gateway minted (see [`strip_gateway_reasoning`]).
//! `/{provider}/…` never translates.
//!
//! Responses ↔ Messages is composed through Chat Completions so thinking / `cache_control` /
//! `reasoning_effort` keep the slice-1 mappings.
//!
//! What crosses, and how:
//! - **Dropped:** Responses-only session fields (`store`, `previous_response_id`, `include`,
//!   `truncation`, …) when leaving Responses; `stream_options` on a Responses or Anthropic body.
//!   Same-endpoint Responses is a byte relay: those session fields pass through. Hints with no
//!   equivalent and no effect on the response's shape (`seed`, penalties, `logit_bias`, `top_k`,
//!   a message's `name`, Responses `reasoning` items other than the gateway's own for a Claude
//!   upstream, and off OpenAI's own API `prompt_cache_key`, `service_tier` and `verbosity`) are
//!   dropped, as is OpenAI JSON mode (`json_object`): Anthropic's format needs a schema, and OpenAI
//!   already requires a JSON-mode prompt to ask for JSON. Another vendor's `thinking` blocks and
//!   Anthropic `cache_control` are dropped onto Responses, which reads neither. Bare reasoning text
//!   (`reasoning_content`) is never sent to Anthropic as a block: it was never signed. `tool_choice`
//!   and `parallel_tool_calls` without `tools` are dropped onto Chat Completions, which 400s them.
//!   OpenAI's hosted search leaving Responses, unless `tool_choice` names it (see
//!   `is_hosted_web_search`). Thinking in the assistant messages of a Claude request that does
//!   not think (see `drop_thinking_when_off`).
//! - **Forwarded for the provider to reject:** input and options that change what the client gets
//!   back and have no equivalent on the target (`input_audio`, a `file_id` document or
//!   image, a non-base64 data URI, `n` > 1, `logprobs`, audio output, `stop` onto Responses or a
//!   model that rejects it, hosted / server / `custom` tools, `mcp_servers`, a Responses `prompt`
//!   template and `top_logprobs`, a Responses input item with no Chat Completions shape such as
//!   `computer_call_output`; records of a hosted tool the provider ran are dropped). An explicit
//!   `null` is "not set" and is never forwarded. Translation runs after the request headers went upstream, so the gateway cannot 400
//!   here; passing the field through gets the provider's 400 naming it, instead of an answer about
//!   input the model never saw or without a tool the client offered. `input_audio` is forwarded
//!   in every direction, never dropped.
//! - **Refused before translation:** an image or file URL that is not `http(s)` or inline `data:`
//!   (`file://`, `gs://`, a bare path) names something on the client's machine and is never
//!   forwarded in any shape. `route::refused_input` answers 400 for it before any upstream
//!   (`route::carries_unforwardable_url`, by the rule of [`is_forwardable_url`]), so the `None`
//!   arms below that drop one are unreachable from the proxy.
//! - **Images, both ways:** base64 data-URI images convert to Anthropic `base64` sources and back;
//!   `http(s)` image URLs become Anthropic `url` sources and back, and pass through between Chat
//!   Completions and Responses. Nothing is fetched: every one of those upstreams downloads the URL
//!   itself. Amazon Bedrock is the exception — it rejects `url` sources — so a URL image that fails
//!   over onto a Bedrock candidate gets Bedrock's 400 rather than a silently image-less answer.
//! - **PDFs, both ways:** a Chat Completions `file` part with inline `file_data` ↔ an Anthropic
//!   base64 `document` ↔ a Responses `input_file`.
//! - **Structured output, both ways:** `response_format` `json_schema` ↔ Anthropic
//!   `output_config.format` ↔ Responses `text.format`.
//! - **Passed both ways:** signed `thinking` / `redacted_thinking` blocks (our `thinking` array,
//!   OpenRouter's `reasoning_details`, and the gateway's Responses `reasoning` items; see
//!   `openai_assistant_content` and `reasoning_details`), `cache_control` on tools and content,
//!   `parallel_tool_calls: false` ↔ `tool_choice.disable_parallel_tool_use`, `user` ↔
//!   `metadata.user_id` (an email address, which Anthropic rejects, becomes a stable hash), tool
//!   `strict`, custom (free-form) tool calls, mid-conversation `system` / `developer` messages in
//!   place (see `place_system`), and images or files a tool returned (a Chat Completions `tool`
//!   message holds text, so they follow it as a user message). Legacy `functions` /
//!   `function_call` become the `tools` loop.
//! - **Model-aware onto Messages** (see `ClaudeModel`): `reasoning_effort`, or a client's own
//!   `thinking` object, becomes adaptive thinking plus `output_config.effort` on Claude 4.6 and
//!   later, and a `budget_tokens` below `max_tokens` before that. `temperature` / `top_p` are
//!   dropped where the model rejects them (4.7 and later, and anything with thinking on), clamped
//!   to 0–1, and `top_p` yields to `temperature`. Budget thinking yields to forced tool use.
//!   Conversation-binding models get `thinking.block_binding` (see
//!   `bind_thinking_with_drop_block`). Tool ids are rewritten to Anthropic's pattern.
//! - **Model-aware onto Chat Completions and Responses** (see `OpenAiModel`): OpenAI's own ids get
//!   `max_completion_tokens`, sampling only where the model takes it, and an effort from the set
//!   the family accepts (none on a model that does not reason). Other hosts get the classic
//!   `low`–`high` in place of `xhigh` / `max` / `minimal`.
//! - **Added onto Messages:** default `cache_control` breakpoints when the client set none (see
//!   `auto_cache_breakpoints`). An OpenAI SDK never marks anything, and without a marker Anthropic
//!   caches nothing. The writes those markers cause are the gateway's choice, so the client is shown
//!   them as input, as they are billed (see [`request_with_tools`]).
//! - **Added onto Responses:** `store: false` unless the client asked to store; Chat Completions
//!   stores nothing by default, Responses stores everything.
//! - **Speed:** a Chat Completions or Responses client's `service_tier: "priority"` becomes
//!   Anthropic fast mode (`speed: "fast"`, beta from `proxy`) on an attempt the caller says serves
//!   it ([`request_with_tools`]'s `fast`), and is dropped onto any other Messages candidate. A
//!   Messages client's `speed: "fast"` is forwarded onto Chat Completions and Responses for the
//!   provider to refuse (a walk that read the body never sends it there); Anthropic's own
//!   `service_tier` values are dropped off Messages.
//! - **Required mapping:** system/messages/`input`, `max_tokens`/`max_output_tokens`, stop,
//!   stream, tools, `tool_choice`, text + tool_use/tool_result, usage. Anthropic requires
//!   `max_tokens`; a missing OpenAI value becomes 4096.
//!
//! Usage/billing parse the **upstream** body. This module only reshapes bytes the client sees.

use crate::peek;
use crate::route::Endpoint;
use serde_json::{Map, Value, json};

mod errors;
mod model;
mod parts;
mod request;
mod response;
mod same_wire;
mod stop;
mod stream;
mod thinking;
mod tools;
mod usage_map;
mod wire;

use self::errors::*;
pub(crate) use self::model::*;
pub(crate) use self::parts::*;
pub use self::request::*;
use self::response::*;
pub use self::same_wire::*;
use self::stop::*;
pub use self::stream::*;
use self::thinking::*;
pub use self::tools::*;
use self::usage_map::*;
use self::wire::*;

/// Anthropic requires this; OpenAI does not. Used when the Chat Completions body omitted it.
const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Most bytes translation holds for one response: a whole non-streaming body, or one SSE event not
/// yet terminated. Sized for the largest legitimate case — a Responses `response.completed` event
/// repeats the entire output — with room to spare; past it `proxy` aborts the response rather than
/// let one upstream grow the gateway's memory without bound.
pub const MAX_TRANSLATE_BUFFER: usize = 32 * 1024 * 1024;

/// The largest answer any catalog row allows, in tokens (`max_output_tokens`, published or not).
const MAX_CATALOG_OUTPUT_TOKENS: usize = {
    let rows = providers::catalog::MODEL_ROUTES;
    let (mut i, mut most) = (0, 0u32);
    while i < rows.len() {
        let m = rows[i].card.max_output_tokens;
        if m > most {
            most = m;
        }
        i += 1;
    }
    most as usize
};

/// Most bytes of text one token can carry: the longest entry in OpenAI's `o200k_base` and
/// `cl100k_base` vocabularies is 128 bytes (measured 2026-10-01 from the published files).
const MAX_TOKEN_BYTES: usize = 128;

/// Most output a translated stream may gather: every answer the catalog allows, at the most
/// bytes a token can be (384,000 tokens x 128 bytes, 46.9 MiB). A bridge onto a Responses client
/// keeps every text, reasoning and argument delta for the closing `response.completed`, and onto
/// Messages a queued call's arguments wait whole; past this the upstream is sending more than any
/// model can answer, and `proxy` aborts the response (`ai_rejections_total{reason=
/// "response_too_large"}`), as it does past [`MAX_TRANSLATE_BUFFER`].
pub const MAX_STREAM_OUTPUT: usize = MAX_CATALOG_OUTPUT_TOKENS * MAX_TOKEN_BYTES;

/// The most heap translating `body` onto another wire takes at its peak: the parsed `Value`,
/// the translated `Value` and the encoded output, bounded from the body's length and structure
/// ([`peek::structure`], one pass, no parse) as
/// `5 × len + 2048 × non-empty objects + 128 × structural tokens`.
///
/// Measured with divan's allocation profiler (`benches/unit.rs`, `translate_heap`: peak live bytes
/// of `request`, 4 MiB bodies, 2026-10-01), on the shapes that cost a `Value` the most per byte:
///
/// | body                                                 | heap per byte | estimate |
/// | ---------------------------------------------------- | ------------: | -------: |
/// | one long string                                      |           4.0 |      5.0 |
/// | Chat content parts `{"a":1}` (and Responses items)   |         249.2 |    309.0 |
/// | Chat text parts `{"type":"text","text":"a"}`         |          76.9 |    104.6 |
/// | Responses messages `{"role":"user","content":"hi"}`  |          65.8 |     89.0 |
/// | Chat messages `{"role":"user","content":"hi"}`       |          44.9 |     89.0 |
/// | tool schema of `{"a":1}` objects                     |         168.3 |    309.0 |
/// | tool schema of numbers / empty arrays / `{}`         |    34.0 / 28.7 / 28.7 | 69 / 90 / 90 |
/// | a 300,000-member object                              |          23.9 |     30.0 |
///
/// A string costs ~4 bytes per byte (its text in both `Value`s and the output). Everything else
/// hangs off a node: a non-empty object costs 1.3-1.9 KiB across the two `Value`s (a `BTreeMap`
/// leaf node each, ~0.6 KiB, plus its entries), any other node under 128 bytes. The estimate covers
/// every measured shape; on an ordinary agent history it is ~2x the truth.
pub fn translation_heap(body: &[u8]) -> usize {
    let s = peek::structure(body);
    body.len()
        .saturating_mul(5)
        .saturating_add(s.objects.saturating_mul(2048))
        .saturating_add(s.tokens.saturating_mul(128))
}

/// Per-request translate state, boxed on [`crate::proxy`]'s model-routed path only.
pub struct TranslateState {
    /// Inbound endpoint — what the client sent and what it must receive.
    pub client: Endpoint,
    /// SSE translator, created in `response_filter` once the upstream is known to stream.
    pub sse: Option<SseBridge>,
    /// Non-stream JSON, withheld until end-of-stream so we can map the object.
    pub json_buf: Vec<u8>,
    /// How this attempt's request reshaped a Responses client's tools; set per attempt by
    /// [`request_with_tools`] and read by the response translation.
    pub tools: ToolNames,
    /// This attempt's request carries cache breakpoints the gateway added (see
    /// [`request_with_tools`]). Its cache writes are the gateway's cost, billed and shown as input.
    pub gateway_cache: bool,
    /// Every cache breakpoint this attempt's Chat Completions request carries asks for the 1-hour
    /// TTL (see [`chat_cache_is_1h`]). OpenRouter honours it and bills those writes at the 1-hour
    /// rate, but its usage has no TTL split, so the row attributes the writes from this. Set per
    /// attempt.
    pub cache_1h: bool,
    /// This attempt asked a stream-only candidate for a stream the client did not ask for
    /// ([`force_stream`]); a streamed answer is assembled into one body
    /// ([`SseBridge::assembling`]). Set per attempt.
    pub assemble: bool,
}

impl TranslateState {
    pub fn new(client: Endpoint) -> Self {
        Self {
            client,
            sse: None,
            json_buf: Vec::new(),
            tools: ToolNames::default(),
            gateway_cache: false,
            cache_1h: false,
            assemble: false,
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
    request_with_tools(from, to, body, upstream_model, false).0
}

/// Whether a Chat Completions body marks at least one cache breakpoint and every one asks for the
/// 1-hour TTL. OpenRouter takes `cache_control` (with its `ttl`) on system and message content parts
/// and on tools, and bills 1-hour writes at 2x input (probed: 19,007 tokens on Haiku 4.5 cost
/// $0.038), but reports them only as `prompt_tokens_details.cache_write_tokens`. A mixed request
/// leaves the split unknown, so it reads as 5-minute writes, the cheaper side.
pub fn chat_cache_is_1h(chat: &Value) -> bool {
    let parts = chat
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten();
    let tools = chat
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    let mut marks = parts
        .chain(tools)
        .filter_map(|p| p.get("cache_control"))
        .peekable();
    marks.peek().is_some() && marks.all(|cc| cc.get("ttl").and_then(Value::as_str) == Some("1h"))
}

/// [`request`], plus what the response translation must follow: the [`ToolNames`] the upstream's
/// tool calls map back through (empty unless a Responses body's tools were reshaped), and whether
/// the gateway added default cache breakpoints (see `auto_cache_breakpoints`). When it did, the
/// client never asked for caching: the writes those breakpoints cause are the gateway's
/// optimization, billed and shown to the client as input. The last value is [`chat_cache_is_1h`]
/// of the Chat Completions body sent (`false` onto any other wire).
///
/// `fast` is the caller's decision that this attempt asks for Anthropic fast mode: a Chat
/// Completions or Responses client's `service_tier: "priority"` bound for a candidate that serves
/// it (`providers::catalog::serves_fast_mode`). The Messages body then carries `speed: "fast"`, and
/// `proxy` sends [`FAST_MODE_BETA`] on the same attempt's headers from the same decision.
pub fn request_with_tools(
    from: Endpoint,
    to: Endpoint,
    body: &[u8],
    upstream_model: &str,
    fast: bool,
) -> (Vec<u8>, ToolNames, bool, bool) {
    if from == to {
        return (body.to_vec(), ToolNames::default(), false, false);
    }
    let Some(v) = parse_body(from, body) else {
        return (body.to_vec(), ToolNames::default(), false, false);
    };
    if !v.is_object() {
        return (body.to_vec(), ToolNames::default(), false, false);
    }
    let names = if from == Endpoint::Responses {
        ToolNames::of_responses(&v, to == Endpoint::Messages)
    } else {
        ToolNames::default()
    };
    let (mut out, gateway_cache) = map_request(from, to, &v, Upstream::of(upstream_model));
    if fast
        && to == Endpoint::Messages
        && let Some(obj) = out.as_object_mut()
    {
        obj.insert("speed".into(), json!("fast"));
    }
    let cache_1h = to == Endpoint::ChatCompletions && chat_cache_is_1h(&out);
    (encode(&out), names, gateway_cache, cache_1h)
}

/// What this attempt's upstream model accepts, parsed once from the id it will receive.
#[derive(Debug, Clone, Copy)]
struct Upstream {
    claude: ClaudeModel,
    openai: OpenAiModel,
    /// A Claude model (any spelling, any generation) — the only kind that verifies an Anthropic
    /// thinking signature. On a Chat Completions host (OpenRouter's `anthropic/…`) signed thinking
    /// must also ride `reasoning_details`, the only field OpenRouter replays to Anthropic.
    claude_thinking: bool,
}

impl Upstream {
    fn of(model: &str) -> Self {
        Self {
            claude: ClaudeModel::of(model),
            openai: OpenAiModel::of(model),
            claude_thinking: model.contains("claude"),
        }
    }
}

/// The mapped body, and whether the gateway added cache breakpoints to it.
fn map_request(from: Endpoint, to: Endpoint, v: &Value, up: Upstream) -> (Value, bool) {
    // Legacy `functions` / `function_call` are the same tool loop in an older shape.
    let chat = (from == Endpoint::ChatCompletions).then(|| legacy_functions_to_tools(v));
    let chat = chat.as_ref().and_then(Option::as_ref).unwrap_or(v);
    let out = match (from, to) {
        (Endpoint::ChatCompletions, Endpoint::Messages) => {
            return openai_req_to_anthropic(chat, up.claude);
        }
        (Endpoint::Messages, Endpoint::ChatCompletions) => {
            claude_chat_reasoning_guard(anthropic_req_to_openai(v, up), up)
        }
        (Endpoint::Responses, Endpoint::ChatCompletions) => {
            claude_chat_reasoning_guard(responses_req_to_openai(v, up, false), up)
        }
        (Endpoint::ChatCompletions, Endpoint::Responses) => {
            openai_req_to_responses(chat, up.openai)
        }
        (Endpoint::Responses, Endpoint::Messages) => {
            let mut chat = responses_req_to_openai(v, up, true);
            // Intermediate only (never sent): `reasoning.summary` has no Chat Completions field, but
            // it decides `thinking.display` on Messages.
            if let (Some(r), Some(obj)) = (v.get("reasoning"), chat.as_object_mut()) {
                obj.insert("reasoning".into(), r.clone());
            }
            return openai_req_to_anthropic(&chat, up.claude);
        }
        (Endpoint::Messages, Endpoint::Responses) => {
            openai_req_to_responses(&anthropic_req_to_openai(v, up), up.openai)
        }
        _ => v.clone(),
    };
    (out, false)
}

/// Map a non-stream JSON response from `upstream` into `client`. Error objects are reshaped
/// into the client's error envelope. Same as [`response_json_status`] with a 200.
pub fn response_json(upstream: Endpoint, client: Endpoint, body: &[u8]) -> Vec<u8> {
    response_json_status(upstream, client, 200, body)
}

/// Map a non-stream JSON response carrying HTTP `status`. Any non-2xx JSON body is an error
/// whatever its shape — Bedrock's `{"message": …}` and FastAPI's `{"detail": …}` have no `error`
/// key, and must not be reshaped as an empty success. A body that is not JSON passes unchanged.
pub fn response_json_status(
    upstream: Endpoint,
    client: Endpoint,
    status: u16,
    body: &[u8],
) -> Vec<u8> {
    response_json_tools(upstream, client, status, body, &ToolNames::default(), false)
}

/// [`response_json_status`] for a request whose tools were reshaped (see [`ToolNames`]): each tool
/// call comes back under the name, namespace and kind the client offered. `gateway_cache`: the
/// request's cache breakpoints were the gateway's, so its cache writes show as input (see
/// [`request_with_tools`]).
pub fn response_json_tools(
    upstream: Endpoint,
    client: Endpoint,
    status: u16,
    body: &[u8],
    tools: &ToolNames,
    gateway_cache: bool,
) -> Vec<u8> {
    let ok = (200..300).contains(&status);
    if upstream == client && ok {
        return body.to_vec();
    }
    let Some(v) = parse_body(upstream, body) else {
        return body.to_vec();
    };
    // The same endpoint on another vendor's error shape (xAI's `{"code", "error": "<string>"}`,
    // Bedrock's `{"message"}`): an SDK reads the envelope its own API sends, so anything else is
    // re-encoded in it. One already in it is relayed as it came, byte for byte.
    if upstream == client {
        if in_client_envelope(&v, client) {
            return body.to_vec();
        }
        return encode(&map_error(&v, client, status));
    }
    if !ok || looks_like_error(&v) {
        return encode(&map_error(&v, client, status));
    }
    encode(&map_response(upstream, client, &v, tools, gateway_cache))
}

fn map_response(
    upstream: Endpoint,
    client: Endpoint,
    v: &Value,
    tools: &ToolNames,
    gateway_cache: bool,
) -> Value {
    match (upstream, client) {
        (Endpoint::Messages, Endpoint::ChatCompletions) => {
            anthropic_resp_to_openai(v, gateway_cache)
        }
        (Endpoint::ChatCompletions, Endpoint::Messages) => openai_resp_to_anthropic(v),
        (Endpoint::ChatCompletions, Endpoint::Responses) => openai_resp_to_responses(v, tools),
        (Endpoint::Responses, Endpoint::ChatCompletions) => responses_resp_to_openai(v),
        (Endpoint::Messages, Endpoint::Responses) => {
            openai_resp_to_responses(&anthropic_resp_to_openai(v, gateway_cache), tools)
        }
        (Endpoint::Responses, Endpoint::Messages) => {
            openai_resp_to_anthropic(&responses_resp_to_openai(v))
        }
        _ => v.clone(),
    }
}

fn encode(v: &Value) -> Vec<u8> {
    serde_json::to_vec(&Exact(v)).unwrap_or_else(|_| {
        br#"{"error":{"message":"translate failed","type":"api_error"}}"#.to_vec()
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod request_tests;

#[cfg(test)]
mod response_tests;

#[cfg(test)]
mod mutation_gaps;
