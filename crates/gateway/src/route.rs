//! Provider routing and per-provider wire details — **data-driven**.
//!
//! The provider is the **first path segment** of the request (`/{provider}/…`); the rest of the path
//! is forwarded to the upstream **verbatim** (native passthrough — the gateway holds no per-provider
//! path knowledge). A path with no provider prefix that starts with `/v1` is the drop-in default:
//! BYO dialect-picks openai/anthropic; managed traffic resolves a catalog row from `x-beyond-model`
//! or the body's `model` and walks that row. An unrecognized first segment is a 404
//! (see `proxy::request_filter`).
//!
//! A provider is a *row* in [`known_providers`] (name, upstream authority, wire format, auth scheme) —
//! adding an OpenAI-wire provider (Groq, DeepSeek, Together, …) is one line there, no new code
//! paths. Operators can also add/override providers from config (see `state`/`config`). Same-wire
//! catalog walks and `/{provider}/…` are a byte relay. A managed `/v1` or `/auto` walk whose inbound
//! path is Chat Completions, Messages, or Responses while the row speaks a different one of those
//! three is translated in `translate`. Same-wire Responses (`/{provider}/v1/responses`) stays a
//! byte relay.
//!
//! The table itself lives in the `providers` crate, **shared with the agent**: the same rows that
//! tell this gateway where to proxy `/{name}/…` and which header to swap the pool key into also tell
//! `crates/agent` where to route *directly* when no gateway is deployed, and which env var holds the
//! user's own key. One table, two consumers — a provider's auth scheme cannot be right here and wrong
//! there. What stays here is what only a running gateway has: [`Provider`], the *resolved* row that
//! carries a circuit breaker, metric handles, and the precomputed pool-key header value.

use crate::circuit_breaker::CircuitBreaker;
use crate::metrics::ProviderMetrics;
use crate::secret::Secret;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The shared provider table. `KNOWN_PROVIDERS` is the gateway-routable subset — the BYO-only rows
/// (HuggingFace, NVIDIA, Kimi-Coding, OpenCode) have no `/{name}/…` mount and no pool key, so minting
/// a [`Provider`] for them at boot would only create dead upstreams.
pub use providers::{AuthScheme, ProviderSpec, gateway_providers as known_providers};

/// The provider's wire format. Aliased to the shared crate's [`providers::WireFormat`] — same two
/// variants, and the agent needs the identical fact to pick a request dialect.
pub use providers::WireFormat as Dialect;

/// An HTTP endpoint shape the catalog walk can name. Distinct from [`Dialect`]: OpenAI-wire
/// covers Chat Completions, Responses and Embeddings, and a catalog row's `wire` only
/// distinguishes Messages from the OpenAI family. Translation is per *endpoint*, and only among
/// the three generation endpoints: Embeddings never translates to or from anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    ChatCompletions,
    Messages,
    Responses,
    Embeddings,
}

impl Endpoint {
    /// The endpoint a catalog row's `wire` implies. GPT rows speak Chat Completions; Claude rows
    /// speak Messages. GPT rows also list a parallel `/v1/responses` arm for Responses clients.
    pub fn of_wire(d: Dialect) -> Self {
        match d {
            Dialect::Anthropic => Endpoint::Messages,
            Dialect::OpenAi => Endpoint::ChatCompletions,
        }
    }

    /// The endpoint a **candidate path** (or any upstream path) serves. `/messages` is Messages,
    /// `/responses` is Responses, everything else is Chat Completions — including OpenRouter's
    /// `/api/v1/chat/completions`. Used per attempt so a mixed row never sends the wrong wire.
    pub fn of_upstream_path(path: &str) -> Self {
        if path.ends_with("/embeddings") {
            Endpoint::Embeddings
        } else if path.ends_with("/messages") {
            Endpoint::Messages
        } else if path.contains("/responses") {
            Endpoint::Responses
        } else {
            Endpoint::ChatCompletions
        }
    }

    pub fn wire(self) -> Dialect {
        match self {
            Endpoint::Messages => Dialect::Anthropic,
            Endpoint::ChatCompletions | Endpoint::Responses | Endpoint::Embeddings => {
                Dialect::OpenAi
            }
        }
    }

    /// The endpoint a catalog row serves. An embeddings row is recognized by its primary's path;
    /// every other row is what its `wire` says (Chat Completions or Messages).
    pub fn of_row(row: &ModelRoute) -> Self {
        match row.candidates.first() {
            Some(c) if c.path.ends_with("/embeddings") => Endpoint::Embeddings,
            _ => Endpoint::of_wire(row.wire),
        }
    }

    /// Caller-facing hint for the wire-mismatch 400.
    pub fn post_hint(self) -> &'static str {
        match self {
            Endpoint::Messages => "Anthropic Messages; POST /v1/messages",
            Endpoint::ChatCompletions => "OpenAI Chat Completions; POST /v1/chat/completions",
            Endpoint::Responses => "OpenAI Responses; POST /v1/responses",
            Endpoint::Embeddings => "OpenAI Embeddings; POST /v1/embeddings",
        }
    }
}

/// The default API prefix OpenAI/Anthropic clients use. A request with no provider segment whose
/// path is exactly this or begins with this plus `/` (see [`is_default_prefix`]) is the drop-in
/// default: BYO dialect-picks openai/anthropic; managed traffic is a catalog walk. Anything else
/// with an unknown first segment is a 404.
pub const DEFAULT_PREFIX: &str = "/v1";

/// The model catalog — see [`providers::catalog`]. A `/{AUTO_SEGMENT}/…` request names a *model*
/// rather than a provider, and this resolves it to the ordered providers that can serve it.
pub use providers::{Candidate, MAX_CANDIDATES, ModelRoute, for_model as model_route};

/// The reserved first path segment for **model-routed** requests: `/auto/…` picks its provider from
/// the catalog using `x-beyond-model` if present, else the body's root `model`. Managed `/v1` is
/// the same walk without this segment.
///
/// Reserved, not merely conventional: `state::build_providers` refuses to boot if config tries to
/// register a provider under this name. Provider lookup runs first in `proxy::request_filter`, so a
/// `provider_authorities.auto = …` entry would otherwise shadow the whole feature silently.
pub const AUTO_SEGMENT: &str = "auto";

/// The header carrying the canonical model name on a catalog walk.
///
/// Optional on managed `/v1` and `/auto`: when absent, the body's root `model` is the name. When
/// present it wins, including over a disagreeing body (counted on `ai_model_header_body_mismatch_total`).
///
/// Sits in the `x-beyond-*` namespace the gateway already owns (`x-beyond-request-id`), so it cannot
/// collide with a header a provider defines, and is stripped before the request goes upstream.
pub const MODEL_HEADER: &str = "x-beyond-model";

/// Whether `path` is the bare-default route: exactly [`DEFAULT_PREFIX`], or `DEFAULT_PREFIX`
/// followed by `/`. **Boundary-checked**, not a raw [`str::starts_with`] — a plain prefix check
/// would also match Google Gemini's real path shape (`/v1beta/models/{model}:generateContent`),
/// silently absorbing it into the bare-path default (which routes to OpenAI) instead of rejecting
/// it as an unrecognized provider. `/v1beta`, `/v10`, `/v1-anything` etc. must all be `false`; only
/// `/v1` itself and `/v1/…` are the real default-prefix shape.
pub fn is_default_prefix(path: &str) -> bool {
    match path.strip_prefix(DEFAULT_PREFIX) {
        Some(rest) => rest.is_empty() || rest.starts_with('/'),
        None => false,
    }
}

/// The default provider name for a dialect — used only for the **bare-path** request (no provider
/// segment), where the dialect is derived from the path. A provider-prefixed request names its
/// provider directly.
pub fn dialect_default(d: Dialect) -> &'static str {
    match d {
        Dialect::OpenAi => "openai",
        Dialect::Anthropic => "anthropic",
    }
}

/// The endpoint an inbound catalog-walk path names, if any.
///
/// Exact, not a prefix: `/v1/messages` is Messages, but `/v1/messages/count_tokens`,
/// `/v1/responses/{id}` and `/v1/responses/input_tokens` are endpoints the catalog does not serve.
/// Treating them as their parent used to forward a token count or a retrieve to the candidate's
/// generation path, where it ran (and billed) as a generation. Under `/auto` the `/v1` is optional
/// (`/auto/chat/completions`), and a trailing slash is ignored everywhere.
pub fn implied_endpoint(path: &str) -> Option<Endpoint> {
    catalog_endpoint(path)
        .filter(|e| e.sub.is_none())
        .map(|e| e.endpoint)
}

/// One endpoint the gateway meters: the suffix its path ends with below any provider's mount
/// prefix, the endpoint it is (a sub-resource is its parent's), and the sub-resource, if it is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointPath {
    pub suffix: &'static str,
    pub endpoint: Endpoint,
    pub sub: Option<SubResource>,
}

/// Every endpoint a managed key may reach, and so every one the gateway meters: the one table the
/// managed allowlist (`proxy::is_managed_provider_endpoint`, a security boundary), the forwarded
/// wire, the sub-resources and the catalog's endpoint names all read (D209). Read with
/// [`forward_endpoint`] (a `/{provider}/…` path, by suffix) or [`catalog_endpoint`] (a `/v1` or
/// `/auto` path, exact), each with one normalisation for its kind of path ([`normalize_endpoint_path`]).
pub const ENDPOINT_PATHS: [EndpointPath; 7] = [
    EndpointPath {
        suffix: "/chat/completions",
        endpoint: Endpoint::ChatCompletions,
        sub: None,
    },
    EndpointPath {
        suffix: "/messages",
        endpoint: Endpoint::Messages,
        sub: None,
    },
    EndpointPath {
        suffix: "/responses",
        endpoint: Endpoint::Responses,
        sub: None,
    },
    EndpointPath {
        suffix: "/embeddings",
        endpoint: Endpoint::Embeddings,
        sub: None,
    },
    EndpointPath {
        suffix: "/messages/count_tokens",
        endpoint: Endpoint::Messages,
        sub: Some(SubResource::CountTokens),
    },
    EndpointPath {
        suffix: "/responses/input_tokens",
        endpoint: Endpoint::Responses,
        sub: Some(SubResource::InputTokens),
    },
    EndpointPath {
        suffix: "/responses/compact",
        endpoint: Endpoint::Responses,
        sub: Some(SubResource::Compact),
    },
];

/// The normalisation every forwarded-path lookup applies (no query string): one trailing slash
/// dropped. [`catalog_endpoint`] drops every trailing slash instead.
fn normalize_endpoint_path(path: &str) -> &str {
    path.strip_suffix('/').unwrap_or(path)
}

/// The [`ENDPOINT_PATHS`] row a path (no query string) ends with, under any mount prefix (`/v1`,
/// `/api/v1`, `/openai/v1`, `/inference/v1`, `/anthropic/v1`, `/backend-api/codex`). A path that
/// carries a query does not match: split it off first ([`forward_endpoint`] does).
pub fn endpoint_of_path(path: &str) -> Option<&'static EndpointPath> {
    let path = normalize_endpoint_path(path);
    ENDPOINT_PATHS.iter().find(|e| path.ends_with(e.suffix))
}

/// The [`ENDPOINT_PATHS`] row a forwarded `/{provider}/…` path names (query allowed and ignored).
pub fn forward_endpoint(path_and_query: &str) -> Option<&'static EndpointPath> {
    endpoint_of_path(
        path_and_query
            .split_once('?')
            .map_or(path_and_query, |(p, _)| p),
    )
}

/// The [`ENDPOINT_PATHS`] row a catalog-walk path names, exactly: under `/auto` the `/v1` is
/// optional, and every trailing slash is ignored. Deliberately looser than a forwarded path, which
/// keeps the provider's own spelling (one trailing slash is the same resource, `//` one the
/// provider 404s): a catalog path is the gateway's own name, so `/v1/messages//` is Messages
/// (pinned by `prop_documented_paths_classify_as_the_table_says`).
fn catalog_endpoint(path: &str) -> Option<&'static EndpointPath> {
    let rest = catalog_path_rest(path)?.trim_end_matches('/');
    let rest = rest.strip_prefix("/v1").unwrap_or(rest);
    ENDPOINT_PATHS.iter().find(|e| rest == e.suffix)
}

/// A provider endpoint under one of the generation endpoints that a catalog walk can serve: the
/// same row, the same model re-spelled per candidate, a fixed suffix on the candidate's path. Only
/// candidates of the provider that defines it can serve it, and it never translates: a token count
/// or a compaction is that provider's own API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubResource {
    /// Anthropic `POST /v1/messages/count_tokens` (Claude Code calls it). Free.
    CountTokens,
    /// OpenAI `POST /v1/responses/input_tokens`. Free.
    InputTokens,
    /// OpenAI `POST /v1/responses/compact` (Codex's remote compaction). Runs a model and reports
    /// `usage`, so it bills like a generation.
    Compact,
}

impl SubResource {
    /// The sub-resource a catalog-walk path names, if any. Same rules as [`implied_endpoint`]:
    /// `/v1` optional under `/auto`, trailing slash ignored.
    pub fn of_path(path: &str) -> Option<Self> {
        catalog_endpoint(path).and_then(|e| e.sub)
    }

    /// The sub-resource a forwarded upstream path names (`/{provider}/…` with the provider segment
    /// stripped, query allowed), so a provider-routed token count is as free as a catalog walk's.
    pub fn of_forward_path(path_and_query: &str) -> Option<Self> {
        forward_endpoint(path_and_query).and_then(|e| e.sub)
    }

    /// Appended to a serving candidate's own path (`/v1/messages` → `/v1/messages/count_tokens`).
    pub fn suffix(self) -> &'static str {
        match self {
            Self::CountTokens => "/count_tokens",
            Self::InputTokens => "/input_tokens",
            Self::Compact => "/compact",
        }
    }

    /// Whether the provider charges for it, so the gateway writes a billing row.
    pub fn billed(self) -> bool {
        matches!(self, Self::Compact)
    }

    /// Whether this candidate can serve it: the defining provider, on the parent endpoint. Bedrock
    /// and OpenRouter speak the parent wire but do not document these endpoints.
    pub fn serves(self, c: &Candidate) -> bool {
        match self {
            Self::CountTokens => {
                c.provider == providers::ProviderId::Anthropic && c.path.ends_with("/messages")
            }
            Self::InputTokens | Self::Compact => {
                c.provider == providers::ProviderId::OpenAi && c.path.ends_with("/responses")
            }
        }
    }

    /// The provider that defines it, for the error when a row has no candidate that serves it.
    pub fn provider_name(self) -> &'static str {
        match self {
            Self::CountTokens => "Anthropic",
            Self::InputTokens | Self::Compact => "OpenAI",
        }
    }
}

/// Whether a catalog-walk path names no endpoint at all (bare `/v1` or `/auto`), so the row's
/// primary picks the path.
fn names_no_endpoint(path: &str) -> bool {
    match catalog_path_rest(path) {
        None => true,
        Some(rest) => matches!(rest.trim_end_matches('/'), "" | "/v1"),
    }
}

fn catalog_path_rest(path: &str) -> Option<&str> {
    match path.strip_prefix("/auto") {
        Some("") | Some("/") => None,
        Some(r) => Some(r),
        None => Some(path),
    }
}

/// Whether `path` is a Responses endpoint (including `/auto/v1/responses`).
pub fn is_responses_path(path: &str) -> bool {
    implied_endpoint(path) == Some(Endpoint::Responses)
}

/// Whether a forwarded (`/{provider}/…`, query stripped) path is the Responses generation
/// endpoint itself, under any mount prefix: not a sub-resource or a stored response.
pub fn forward_is_responses(path: &str) -> bool {
    endpoint_of_path(path).is_some_and(|e| e.endpoint == Endpoint::Responses && e.sub.is_none())
}

/// Whether a catalog candidate path is the Responses endpoint.
pub fn candidate_path_is_responses(path: &str) -> bool {
    path.ends_with("/responses")
}

/// What a managed catalog walk should do when the inbound path names a wire that may disagree
/// with the row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireAction {
    /// Same endpoint, or a bare `/v1` / `/auto` path that names none. Byte-relay: the row
    /// picks the path.
    Relay,
    /// Inbound Chat Completions, Messages, or Responses vs a row that speaks a different one
    /// of those three. Translate; `client` is what the caller sent and must receive.
    Translate { client: Endpoint },
    /// Embeddings against a generation row or the reverse, or a named path the catalog does not
    /// serve (`/v1/moderations`, …). 400. Relaying would send the body to the candidate's path,
    /// which is a different endpoint.
    Reject,
}

/// Catalog-walk decision for an inbound path vs the row's endpoint ([`Endpoint::of_row`]).
///
/// Inbound `/v1/responses` is a third client dialect vs the row primary; per-candidate translate
/// then uses the serving path, so a GPT row's walk onto its `/v1/responses` arm is a byte relay.
/// Same-wire Responses on `/{provider}` never reaches here.
pub fn catalog_wire_action(path: &str, row: Endpoint) -> WireAction {
    match implied_endpoint(path) {
        Some(client) if client == row => WireAction::Relay,
        Some(client) if client != Endpoint::Embeddings && row != Endpoint::Embeddings => {
            WireAction::Translate { client }
        }
        Some(_) => WireAction::Reject,
        None if names_no_endpoint(path) => WireAction::Relay,
        None => WireAction::Reject,
    }
}

/// Whether a managed catalog walk on `row` reads the whole body before it chooses candidates, even
/// when a header named the row. A headerless walk reads it anyway (to find `model`); this makes a
/// header-won one do the same where the body decides something: the row's card refuses image input
/// or tools ([`refused_input`]), or a candidate cannot serve a capability the card advertises
/// ([`unserved`]). Embeddings rows never: their bodies carry neither.
pub fn walk_reads_body(row: &ModelRoute) -> bool {
    Endpoint::of_row(row) != Endpoint::Embeddings
        && (row.card.input & providers::catalog::IN_IMAGE == 0
            || row.card.features & providers::catalog::TOOLS == 0
            || row
                .candidates
                .iter()
                .any(|c| !providers::catalog::serves_structured_outputs(c))
            || (row.card.input & providers::catalog::IN_FILE != 0
                && row
                    .candidates
                    .iter()
                    .any(|c| !providers::catalog::serves_file_input(c))))
}

/// The most tools OpenAI Chat Completions accepts in one request: 400 `array_above_max_length`
/// "Expected an array with maximum length 128" above it. OpenAI's Responses API took 600 (TOOL-1).
pub const CHAT_TOOL_CAP: usize = 128;

/// Whether a translated (Messages) request on `row` walks the row's Responses arm instead of its
/// candidates: a row whose primary is Chat Completions and that has a Responses arm (the GPT rows
/// before 5.4), and a body offering more than [`CHAT_TOOL_CAP`] tools. Chat Completions would refuse
/// them; the arm is translated onto and takes them (D131). A Chat Completions client keeps its
/// own endpoint (a byte relay, OpenAI's own limit and error), and a Responses client already walks
/// the arm.
pub fn tools_need_responses_arm(row: &ModelRoute, client: Option<Endpoint>, body: &[u8]) -> bool {
    tool_count_decides_arm(row, client) && tool_count(body) > CHAT_TOOL_CAP
}

/// Whether a header-won walk on `row` from a client on `path` reads the body to count its tools
/// ([`tools_need_responses_arm`]).
pub fn walk_reads_tools(row: &ModelRoute, path: &str) -> bool {
    tool_count_decides_arm(row, implied_endpoint(path))
}

/// A Messages client on a row whose primary is Chat Completions and that has a Responses arm.
fn tool_count_decides_arm(row: &ModelRoute, client: Option<Endpoint>) -> bool {
    client == Some(Endpoint::Messages)
        && !row.responses.is_empty()
        && row
            .candidates
            .first()
            .is_some_and(|c| providers::catalog::endpoint_of_path(c.path) == "chat/completions")
}

/// Entries in the body's root `tools` array (the last `tools`, which a provider's parser keeps).
/// Zero, with no structural scan, when the body never says `tools`.
fn tool_count(body: &[u8]) -> usize {
    if memchr::memmem::find(body, b"\"tools\"").is_none() {
        return 0;
    }
    let Some(tools) = crate::peek::root_members(body)
        .and_then(|m| m.into_iter().rev().find(|m| m.key_is(body, "tools")))
    else {
        return 0;
    };
    if body.get(tools.value.0) != Some(&b'[') {
        return 0;
    }
    crate::peek::array_elements(body, tools.value.0).map_or(0, |items| items.len())
}

/// What `body` asks of `row` that its card says the row does not accept: `Some("image input")` for
/// an image part on a row without image input, `Some("tools")` for a non-empty `tools` array on a
/// row whose card lists no tools. The walk answers 400 before any upstream sees it. An image would
/// otherwise be ignored and an answer about nothing billed (o3-mini, Together's gpt-oss-120b), or
/// fail with a 500 (gpt-4). Tools would be called with junk arguments by a model the card dropped
/// them from for that (GLM 5.3 Flash, D197), or refused upstream in each provider's own words. A
/// PDF on a row without file input is left to the candidate, because OpenRouter extracts a PDF's
/// text for any model.
pub fn refused_input(row: &ModelRoute, body: &[u8]) -> Option<&'static str> {
    if Endpoint::of_row(row) == Endpoint::Embeddings {
        return None;
    }
    if row.card.input & providers::catalog::IN_IMAGE == 0 && carries_image(body) {
        return Some("image input");
    }
    if row.card.features & providers::catalog::TOOLS == 0 && tool_count(body) > 0 {
        return Some("tools");
    }
    None
}

/// An image content part anywhere in the conversation: Chat Completions `image_url`, Messages
/// `image` (tool results included), Responses `input_image`. The body is parsed only when it
/// contains the bytes `image` at all.
fn carries_image(body: &[u8]) -> bool {
    fn walk(v: &serde_json::Value) -> bool {
        match v {
            serde_json::Value::Array(a) => a.iter().any(walk),
            serde_json::Value::Object(o) => {
                matches!(
                    o.get("type").and_then(serde_json::Value::as_str),
                    Some("image_url" | "image" | "input_image")
                ) || o.values().any(walk)
            }
            _ => false,
        }
    }
    if memchr::memmem::find(body, b"image").is_none() {
        return false;
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return false;
    };
    ["messages", "input", "system"]
        .iter()
        .filter_map(|k| v.get(k))
        .any(walk)
}

/// Catalog indices (bit per index) of `arms` that cannot serve `body`: Bedrock candidates when the
/// body asks for a JSON-schema output (`response_format` / `output_config.format` / `text.format`
/// all name `json_schema`), which Bedrock's Messages surface refuses; and a candidate that reads no
/// PDF (`providers::catalog::serves_file_input`) when the body carries a file part. The walk leaves
/// them out, unless that would leave nothing, in which case the provider's own error is the answer.
/// Zero, with no scan, on a row whose candidates all serve both.
pub fn unserved(arms: &[Candidate], body: &[u8]) -> u8 {
    let mask = |serves: fn(&Candidate) -> bool| {
        arms.iter()
            .take(MAX_CANDIDATES)
            .enumerate()
            .filter(|(_, c)| !serves(c))
            .fold(0u8, |m, (i, _)| m | (1 << i))
    };
    let mut out = 0;
    let no_schema = mask(providers::catalog::serves_structured_outputs);
    if no_schema != 0 && memchr::memmem::find(body, b"\"json_schema\"").is_some() {
        out |= no_schema;
    }
    let no_file = mask(providers::catalog::serves_file_input);
    if no_file != 0 && carries_file(body) {
        out |= no_file;
    }
    out
}

/// A file content part anywhere in the conversation: Chat Completions `file`, Messages
/// `document` (tool results included), Responses `input_file`. The body is parsed only when it
/// contains one of those type names at all.
fn carries_file(body: &[u8]) -> bool {
    fn walk(v: &serde_json::Value) -> bool {
        match v {
            serde_json::Value::Array(a) => a.iter().any(walk),
            serde_json::Value::Object(o) => {
                matches!(
                    o.get("type").and_then(serde_json::Value::as_str),
                    Some("file" | "document" | "input_file")
                ) || o.values().any(walk)
            }
            _ => false,
        }
    }
    let finder = |n: &[u8]| memchr::memmem::find(body, n).is_some();
    if !(finder(b"\"file\"") || finder(b"\"document\"") || finder(b"\"input_file\"")) {
        return false;
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return false;
    };
    ["messages", "input"]
        .iter()
        .filter_map(|k| v.get(k))
        .any(walk)
}

/// One precomputed managed auth value: the formatted secret plus, when the bytes are header-safe,
/// a ready-to-insert [`http::HeaderValue`].
///
/// `insert_header(name, &str)` runs `HeaderValue::from_str`, which validates the bytes and copies
/// them into a fresh `Bytes` — a heap allocation per managed request for a value fixed at boot.
/// Cloning a `HeaderValue` is a refcount bump instead.
///
/// `header` is `None` only if the configured key isn't a legal header value (a stray newline, say),
/// which no key that could ever have worked would be — `insert_header` would have rejected it per
/// request. The caller falls back to the string form so that stays true rather than becoming a
/// silent 503.
///
/// Hygiene note: a `HeaderValue` is not zeroized on drop, so this is one long-lived plaintext copy
/// of the pool key. That is a net *improvement* — `secret.rs` already concedes the key is "copied
/// into Pingora's request headers we don't own", and previously that copy was made and freed
/// thousands of times a second, scattering key bytes across the heap. The `Secret` is kept for the
/// redacting `Debug`.
pub struct PoolAuth {
    pub value: Secret,
    pub header: Option<http::HeaderValue>,
    /// Where the bare key starts in `value` (after the scheme's `Bearer `), for [`Self::key`].
    key_at: usize,
    /// Searchers for [`Self::key`] ([`key_finders`]: the key itself first, then the spellings a JSON
    /// encoder may give it), built once at boot: every managed error response is scanned for an
    /// echo of the key (`proxy::Redact`), and building a searcher per response repeated the key's
    /// preprocessing on each one (D92). Each holds a copy of the key, like `header`.
    finders: Box<[memchr::memmem::Finder<'static>]>,
    /// When this key's last refusal (a 401, a 403 naming the key, or an out-of-credit answer:
    /// D180) cools off, in ms since [`clock_ms`]'s epoch; 0 when it has none. A cooling key is
    /// skipped as a request's *first* key, and a provider whose keys all cool is skipped by a
    /// catalog walk that has another candidate ([`Provider::cooling`]), so traffic stops paying a
    /// round trip to a revoked or unfunded key on every request. Shared across requests; relaxed
    /// ordering, since a stale read costs only one more walk.
    bad_until_ms: AtomicU64,
}

impl PoolAuth {
    /// The bare key, without its scheme: what a provider that echoes its credential would echo,
    /// and so what the gateway scrubs from responses (`proxy::Redact`).
    pub fn key(&self) -> &str {
        self.value.expose().get(self.key_at..).unwrap_or("")
    }

    /// The boot-built searcher for [`Self::key`] as sent.
    pub fn finder(&self) -> &memchr::memmem::Finder<'static> {
        &self.finders[0]
    }

    /// Every boot-built searcher for [`Self::key`]: as sent, then each escaped spelling.
    pub fn finders(&self) -> &[memchr::memmem::Finder<'static>] {
        &self.finders
    }

    fn cooling(&self, now_ms: u64) -> bool {
        self.bad_until_ms.load(Ordering::Relaxed) > now_ms
    }
}

/// Searchers for every spelling of `key` an upstream echo may use: the key as sent first, then, for
/// a key holding `/` or `+` (a Bedrock `ABSK…` key is base64), the spellings a JSON encoder writes
/// them in: `\/` for `/`, and `+` or `+` for `+` (one encoder writes every occurrence
/// the same way). A JSON client decodes each of those to the key, so each is scrubbed (D203). Never
/// empty: an empty key yields one empty searcher, which the scrub skips.
pub fn key_finders(key: &str) -> Box<[memchr::memmem::Finder<'static>]> {
    let slashes: &[&str] = if key.contains('/') {
        &["/", "\\/"]
    } else {
        &["/"]
    };
    let pluses: &[&str] = if key.contains('+') {
        &["+", "\\u002b", "\\u002B"]
    } else {
        &["+"]
    };
    let mut spellings: Vec<String> = Vec::with_capacity(slashes.len() * pluses.len());
    for slash in slashes {
        for plus in pluses {
            let s = key.replace('/', slash).replace('+', plus);
            if !spellings.contains(&s) {
                spellings.push(s);
            }
        }
    }
    spellings
        .iter()
        .map(|s| memchr::memmem::Finder::new(s.as_bytes()).into_owned())
        .collect()
}

/// How long a pool key that drew a 401 (or a 403 naming the key, or an out-of-credit answer) is
/// skipped as a request's first key.
pub const KEY_COOLDOWN: Duration = Duration::from_secs(60);

/// Monotonic milliseconds since the first call, plus one (so 0 stays "never failed"). Coarse
/// enough for a cooldown, and an `AtomicU64` holds it where an `Instant` would need a lock.
fn clock_ms() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    u64::try_from(EPOCH.get_or_init(Instant::now).elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .saturating_add(1)
}

/// A *resolved* provider: static wire facts + the boot-resolved upstream authority/host + (for
/// managed traffic) the precomputed pool auth header values. Built once at boot (see
/// `state::build_providers`); the request hot path holds an `Arc<Provider>` (cheap clone) and
/// borrows these fields, so nothing is re-allocated or re-formatted per request.
pub struct Provider {
    pub name: String,
    /// Upstream `host:port`.
    pub authority: String,
    /// Bare upstream host (SNI / `Host` header) = authority without the port.
    pub host: String,
    /// The provider's wire format (usage parsing + injection eligibility). See [`ProviderSpec::wire`].
    pub dialect: Dialect,
    pub auth: AuthScheme,
    /// Precomputed managed auth values, one per configured pool key, in config order. Empty ⇒ no
    /// pool key is configured for this provider ⇒ managed requests to it are rejected (503). A
    /// managed 429 or 401 walks the next unused entry; the key is never sent to a different
    /// provider. A 401 (or a 403 whose body names the key) also cools the key off for later requests (see [`Self::first_key`]).
    pub pool_auth: Box<[PoolAuth]>,
    /// `host` as a ready-to-insert `HeaderValue` — see [`PoolAuth`].
    pub host_header: Option<http::HeaderValue>,
    /// `name` as a ready-to-insert `HeaderValue`, for the `x-beyond-provider` response header.
    pub name_header: Option<http::HeaderValue>,
    /// Per-provider metric handles, resolved once here so the response path bumps a direct
    /// counter/histogram instead of a string-keyed label lookup per response.
    pub metrics: ProviderMetrics,
    /// Per-provider circuit breaker, shared across all callers to this provider. `None` when the
    /// breaker is disabled (`circuit_breaker_threshold == 0`). Checked before connect and fed the
    /// 5xx/connect outcome — see `proxy`. Lock-free, so the hot path reads it without contention.
    pub breaker: Option<CircuitBreaker>,
}

impl Provider {
    /// Whether at least one pool key is configured — the 503 gate for managed traffic.
    pub fn has_pool_key(&self) -> bool {
        !self.pool_auth.is_empty()
    }

    /// The pool key a new request on this provider starts on: the first one not cooling off from
    /// an auth failure, or key 0 when every key is (a revoked set still has to answer something).
    pub fn first_key(&self) -> u8 {
        if self.pool_auth.len() < 2 {
            return 0;
        }
        let now = clock_ms();
        self.pool_auth
            .iter()
            .position(|k| !k.cooling(now))
            .and_then(|i| u8::try_from(i).ok())
            .unwrap_or(0)
    }

    /// Whether every pool key is cooling off from a refusal (see [`Self::mark_key_bad`]). A catalog
    /// walk leaves such a provider out while another candidate can take the request (D180).
    pub fn cooling(&self) -> bool {
        let now = clock_ms();
        !self.pool_auth.is_empty() && self.pool_auth.iter().all(|k| k.cooling(now))
    }

    /// Record that pool key `i` was refused (a 401, a 403 naming the key, or an out-of-credit
    /// answer): later requests start past it for [`KEY_COOLDOWN`].
    pub fn mark_key_bad(&self, i: u8) {
        if let Some(k) = self.pool_auth.get(usize::from(i)) {
            let cooldown = u64::try_from(KEY_COOLDOWN.as_millis()).unwrap_or(u64::MAX);
            k.bad_until_ms
                .store(clock_ms().saturating_add(cooldown), Ordering::Relaxed);
        }
    }

    /// Resolve a provider from its name, upstream authority, dialect, auth scheme, pool keys, and
    /// pre-resolved per-provider metric handles. Derives the bare host and precomputes each
    /// managed auth header value once. An empty `pool_keys` is the same as none: managed requests
    /// to this provider 503.
    pub fn resolve(
        name: &str,
        authority: String,
        dialect: Dialect,
        auth: AuthScheme,
        pool_keys: &[&str],
        metrics: ProviderMetrics,
        breaker: Option<CircuitBreaker>,
    ) -> Self {
        let host = authority
            .split(':')
            .next()
            .unwrap_or(&authority)
            .to_string();
        let pool_auth = pool_keys
            .iter()
            .map(|k| {
                let value = Secret::new(auth.format(k));
                // Sensitive: HPACK never indexes it (so it cannot be recovered from the
                // compression table) and `Debug` prints `Sensitive`, not the key.
                let header = http::HeaderValue::from_str(value.expose())
                    .ok()
                    .map(|mut h| {
                        h.set_sensitive(true);
                        h
                    });
                let key_at = auth.value_prefix().map_or(0, str::len);
                let finders = key_finders(value.expose().get(key_at..).unwrap_or(""));
                PoolAuth {
                    value,
                    header,
                    key_at,
                    finders,
                    bad_until_ms: AtomicU64::new(0),
                }
            })
            .collect();
        let host_header = http::HeaderValue::from_str(&host).ok();
        let name_header = http::HeaderValue::from_str(name).ok();
        Provider {
            name_header,
            name: name.to_string(),
            authority,
            host,
            dialect,
            auth,
            pool_auth,
            host_header,
            metrics,
            breaker,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model-routed segment is matched only after a provider-table miss, so a provider actually
    /// named `auto` would shadow it. `state::build_providers` rejects that from config; this covers
    /// the other direction — someone adding an `auto` row to the shared provider table.
    #[test]
    fn auto_segment_is_not_a_provider_name() {
        assert!(
            providers::by_name(AUTO_SEGMENT).is_none(),
            "{AUTO_SEGMENT:?} is reserved for model-routed requests and cannot also be a provider",
        );
    }

    /// Every catalog candidate must resolve to a row this gateway actually mints a `Provider` for,
    /// or the route silently loses a failover target at request time.
    #[test]
    fn every_catalog_candidate_is_a_known_provider() {
        for route in providers::catalog::MODEL_ROUTES {
            for c in route.candidates.iter().chain(route.responses.iter()) {
                let spec = providers::by_id(c.provider);
                assert!(
                    known_providers().any(|p| p.id == c.provider),
                    "catalog route {:?} names {}, which the gateway does not route to",
                    route.model,
                    spec.name,
                );
            }
        }
    }

    #[test]
    fn is_default_prefix_boundary_checks() {
        // The real default-prefix shape: exactly "/v1" or "/v1/…".
        assert!(is_default_prefix("/v1"));
        assert!(is_default_prefix("/v1/"));
        assert!(is_default_prefix("/v1/messages"));
        assert!(is_default_prefix("/v1/chat/completions"));
        // Task #7 (pi-parity): Google Gemini's real path shape must NOT be absorbed by a raw
        // `starts_with("/v1")` — before this fix it fell into the bare-default branch, which
        // routes anything that isn't `/v1/messages*` to OpenAI, silently misrouting Gemini.
        assert!(!is_default_prefix(
            "/v1beta/models/gemini-2.5-pro:generateContent"
        ));
        assert!(!is_default_prefix("/v1beta"));
        // Other near-misses that must not match either.
        assert!(!is_default_prefix("/v10"));
        assert!(!is_default_prefix("/v1-legacy"));
        assert!(!is_default_prefix("/v2/messages"));
        assert!(!is_default_prefix(""));
        assert!(!is_default_prefix("/"));
    }

    #[test]
    fn implied_endpoint_is_exact() {
        for (path, want) in [
            ("/v1", None),
            ("/v1/", None),
            ("/auto", None),
            ("/auto/", None),
            ("/v1/messages", Some(Endpoint::Messages)),
            ("/v1/messages/", Some(Endpoint::Messages)),
            ("/auto/v1/messages", Some(Endpoint::Messages)),
            ("/auto/messages", Some(Endpoint::Messages)),
            ("/v1/chat/completions", Some(Endpoint::ChatCompletions)),
            ("/auto/chat/completions", Some(Endpoint::ChatCompletions)),
            ("/v1/responses", Some(Endpoint::Responses)),
            ("/auto/responses", Some(Endpoint::Responses)),
            ("/v1/embeddings", Some(Endpoint::Embeddings)),
            ("/v1/embeddings/", Some(Endpoint::Embeddings)),
            ("/auto/embeddings", Some(Endpoint::Embeddings)),
            // Sub-resources are not the endpoint they hang off.
            ("/v1/messages/count_tokens", None),
            ("/v1/messages/batches", None),
            ("/v1/responses/resp_123", None),
            ("/v1/responses/input_tokens", None),
            ("/v1/chat/completions/abc", None),
            ("/v1/models", None),
            ("/v1/moderations", None),
        ] {
            assert_eq!(implied_endpoint(path), want, "{path}");
        }
    }

    #[test]
    fn is_responses_path_matches_stock_sdk_and_auto_suffixes() {
        assert!(is_responses_path("/v1/responses"));
        assert!(is_responses_path("/auto/v1/responses"));
        assert!(is_responses_path("/auto/responses"));
        assert!(!is_responses_path("/v1/chat/completions"));
        assert!(!is_responses_path("/v1/messages"));
        assert!(!is_responses_path("/v1/embeddings"));
        assert!(!is_responses_path("/v1"));
        assert!(!is_responses_path("/openai/v1/responses"));
    }

    #[test]
    fn catalog_wire_action_translates_chat_completions_versus_messages() {
        assert_eq!(
            catalog_wire_action("/v1/chat/completions", Endpoint::Messages),
            WireAction::Translate {
                client: Endpoint::ChatCompletions
            }
        );
        assert_eq!(
            catalog_wire_action("/auto/chat/completions", Endpoint::Messages),
            WireAction::Translate {
                client: Endpoint::ChatCompletions
            }
        );
        assert_eq!(
            catalog_wire_action("/v1/messages", Endpoint::ChatCompletions),
            WireAction::Translate {
                client: Endpoint::Messages
            }
        );
        assert_eq!(
            catalog_wire_action("/auto/v1/messages", Endpoint::ChatCompletions),
            WireAction::Translate {
                client: Endpoint::Messages
            }
        );
        // Same wire: byte-relay.
        assert_eq!(
            catalog_wire_action("/v1/chat/completions", Endpoint::ChatCompletions),
            WireAction::Relay
        );
        assert_eq!(
            catalog_wire_action("/v1/messages", Endpoint::Messages),
            WireAction::Relay
        );
        // Other OpenAI-shaped paths are still a 400 against an Anthropic row.
        assert_eq!(
            catalog_wire_action("/v1/embeddings", Endpoint::Messages),
            WireAction::Reject
        );
        // Responses is a third inbound dialect: translate onto the row's Chat or Messages endpoint.
        assert_eq!(
            catalog_wire_action("/v1/responses", Endpoint::Messages),
            WireAction::Translate {
                client: Endpoint::Responses
            }
        );
        assert_eq!(
            catalog_wire_action("/v1/responses", Endpoint::ChatCompletions),
            WireAction::Translate {
                client: Endpoint::Responses
            }
        );
        assert_eq!(
            catalog_wire_action("/auto/v1/responses", Endpoint::ChatCompletions),
            WireAction::Translate {
                client: Endpoint::Responses
            }
        );
        // Bare /v1 does not name an endpoint.
        assert_eq!(
            catalog_wire_action("/v1", Endpoint::Messages),
            WireAction::Relay
        );
    }

    /// Embeddings relays only onto an embeddings row. Anything else would forward the body to a
    /// candidate path that is a different endpoint.
    #[test]
    fn embeddings_never_translate() {
        assert_eq!(
            catalog_wire_action("/v1/embeddings", Endpoint::Embeddings),
            WireAction::Relay
        );
        assert_eq!(
            catalog_wire_action("/auto/v1/embeddings", Endpoint::Embeddings),
            WireAction::Relay
        );
        for row in [Endpoint::ChatCompletions, Endpoint::Messages] {
            assert_eq!(
                catalog_wire_action("/v1/embeddings", row),
                WireAction::Reject,
                "{row:?}"
            );
        }
        for path in ["/v1/chat/completions", "/v1/messages", "/v1/responses"] {
            assert_eq!(
                catalog_wire_action(path, Endpoint::Embeddings),
                WireAction::Reject,
                "{path}"
            );
        }
        // A named path the catalog does not serve is a 400 even on a same-wire row. It used to
        // relay, which sent the body to the row's chat path.
        assert_eq!(
            catalog_wire_action("/v1/moderations", Endpoint::ChatCompletions),
            WireAction::Reject
        );
        // `/auto` without `/v1` used to slip past the check (and `/auto/responses` too).
        assert_eq!(
            catalog_wire_action("/auto/embeddings", Endpoint::ChatCompletions),
            WireAction::Reject
        );
        assert_eq!(
            catalog_wire_action("/auto/responses", Endpoint::Embeddings),
            WireAction::Reject
        );
        // Sub-resources ran as billed generations on the row's path.
        for path in [
            "/v1/messages/count_tokens",
            "/v1/responses/input_tokens",
            "/v1/responses/resp_123",
        ] {
            assert_eq!(
                catalog_wire_action(path, Endpoint::Messages),
                WireAction::Reject,
                "{path}"
            );
        }
        assert_eq!(
            catalog_wire_action("/auto", Endpoint::Embeddings),
            WireAction::Relay
        );
    }

    #[test]
    fn sub_resources_are_an_exact_table() {
        for (path, want) in [
            ("/v1/messages/count_tokens", Some(SubResource::CountTokens)),
            (
                "/auto/messages/count_tokens/",
                Some(SubResource::CountTokens),
            ),
            ("/v1/responses/input_tokens", Some(SubResource::InputTokens)),
            ("/v1/responses/compact", Some(SubResource::Compact)),
            ("/auto/v1/responses/compact", Some(SubResource::Compact)),
            ("/v1/responses/resp_123", None),
            ("/v1/messages/batches", None),
            ("/v1/messages", None),
            ("/openai/v1/responses/compact", None),
        ] {
            assert_eq!(SubResource::of_path(path), want, "{path}");
        }
        for (path, want) in [
            ("/v1/messages/count_tokens", Some(SubResource::CountTokens)),
            (
                "/v1/messages/count_tokens/?beta=true",
                Some(SubResource::CountTokens),
            ),
            ("/v1/responses/input_tokens", Some(SubResource::InputTokens)),
            ("/v1/responses/compact", Some(SubResource::Compact)),
            ("/v1/messages", None),
            ("/v1/responses", None),
        ] {
            assert_eq!(SubResource::of_forward_path(path), want, "{path}");
        }
        let claude = providers::for_model("claude-opus-4-8").expect("row");
        let served: Vec<_> = claude
            .candidates
            .iter()
            .filter(|c| SubResource::CountTokens.serves(c))
            .map(|c| c.provider)
            .collect();
        assert_eq!(served, [providers::ProviderId::Anthropic]);
        let gpt = providers::for_model("gpt-4o-mini").expect("row");
        assert!(gpt.responses.iter().any(|c| SubResource::Compact.serves(c)));
        assert!(
            !gpt.candidates
                .iter()
                .any(|c| SubResource::Compact.serves(c))
        );
    }

    #[test]
    fn embedding_paths_are_the_embeddings_endpoint() {
        assert_eq!(
            Endpoint::of_upstream_path("/api/v1/embeddings"),
            Endpoint::Embeddings
        );
        assert_eq!(
            implied_endpoint("/v1/embeddings"),
            Some(Endpoint::Embeddings)
        );
        let row = providers::for_model("text-embedding-3-small").expect("catalog row");
        assert_eq!(Endpoint::of_row(row), Endpoint::Embeddings);
        let chat = providers::for_model("gpt-4o-mini").expect("catalog row");
        assert_eq!(Endpoint::of_row(chat), Endpoint::ChatCompletions);
    }

    #[test]
    fn dialect_defaults() {
        assert_eq!(dialect_default(Dialect::OpenAi), "openai");
        assert_eq!(dialect_default(Dialect::Anthropic), "anthropic");
    }

    /// claim: SEC-4
    /// defect: D53
    #[test]
    fn resolve_derives_host_and_pool_auth() {
        let p = Provider::resolve(
            "openai",
            "api.openai.com:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::Bearer,
            &["sk-x"],
            ProviderMetrics::disconnected(),
            None,
        );
        assert_eq!(p.host, "api.openai.com");
        assert_eq!(p.dialect, Dialect::OpenAi);
        assert_eq!(p.pool_auth[0].value.expose(), "Bearer sk-x");
        let header = p.pool_auth[0].header.as_ref().unwrap();
        assert!(
            header.is_sensitive(),
            "the pool key header is marked sensitive"
        );
        assert!(!format!("{header:?}").contains("sk-x"));

        // No pool key ⇒ no managed auth value (managed requests to it would 503).
        let a = Provider::resolve(
            "anthropic",
            "api.anthropic.com:443".to_string(),
            Dialect::Anthropic,
            AuthScheme::XApiKey,
            &[],
            ProviderMetrics::disconnected(),
            None,
        );
        assert!(a.pool_auth.is_empty());
    }

    #[test]
    fn resolve_holds_every_configured_key() {
        let p = Provider::resolve(
            "openai",
            "api.openai.com:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::Bearer,
            &["sk-a", "sk-b"],
            ProviderMetrics::disconnected(),
            None,
        );
        assert_eq!(p.pool_auth.len(), 2);
        assert_eq!(p.pool_auth[0].value.expose(), "Bearer sk-a");
        assert_eq!(p.pool_auth[1].value.expose(), "Bearer sk-b");
    }

    #[test]
    fn precomputed_header_values_match_the_string_form() {
        // The per-request insert now clones these instead of re-validating and re-copying the
        // string. They must be byte-identical to what `insert_header(name, &str)` would have built,
        // or a managed request goes upstream with a different `Host` or a different pool key.
        for (authority, scheme, key) in [
            ("api.openai.com:443", AuthScheme::Bearer, "sk-test"),
            ("api.anthropic.com:443", AuthScheme::XApiKey, "sk-ant-test"),
            (
                "my-resource.openai.azure.com:443",
                AuthScheme::ApiKey,
                "azure-secret",
            ),
        ] {
            let p = Provider::resolve(
                "p",
                authority.to_string(),
                Dialect::OpenAi,
                scheme,
                &[key],
                ProviderMetrics::disconnected(),
                None,
            );
            assert_eq!(
                p.host_header.as_ref().expect("host is header-safe"),
                &http::HeaderValue::from_str(&p.host).unwrap()
            );
            let auth = p.pool_auth.first().expect("pool key configured");
            assert_eq!(
                auth.header.as_ref().expect("key is header-safe"),
                &http::HeaderValue::from_str(auth.value.expose()).unwrap()
            );
        }

        // No pool key ⇒ no precomputed auth header either (and the 503 path is unchanged).
        let none = Provider::resolve(
            "p",
            "h:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::Bearer,
            &[],
            ProviderMetrics::disconnected(),
            None,
        );
        assert!(none.pool_auth.is_empty());

        // A key that is not a legal header value precomputes header = None, so the caller falls
        // back to the string form and gets the same per-request error it always did — rather than
        // this quietly turning into a 503.
        let bad = Provider::resolve(
            "p",
            "h:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::XApiKey,
            &["has\nnewline"],
            ProviderMetrics::disconnected(),
            None,
        );
        assert_eq!(bad.pool_auth.len(), 1);
        assert!(bad.pool_auth[0].header.is_none());
    }

    #[test]
    fn resolve_azure_config_added_provider_uses_bare_api_key_header() {
        // Task #8 (pi-parity): a config-added Azure provider (`provider_auth_schemes.azure =
        // "api-key"`) must produce a bare key (no `Bearer`) as the managed auth value, sent in
        // `api-key` — matching Azure's real wire (see `AuthScheme::ApiKey`'s doc comment).
        let azure = Provider::resolve(
            "azure",
            "my-resource.openai.azure.com:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::ApiKey,
            &["azure-secret"],
            ProviderMetrics::disconnected(),
            None,
        );
        assert_eq!(azure.auth.header(), "api-key");
        assert_eq!(azure.pool_auth[0].value.expose(), "azure-secret");
    }
}
