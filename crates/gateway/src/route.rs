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
    /// speak Messages. GPT rows also list a parallel `/v1/responses` arm for session state.
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
    let rest = catalog_path_rest(path)?.trim_end_matches('/');
    match rest.strip_prefix("/v1").unwrap_or(rest) {
        "/chat/completions" => Some(Endpoint::ChatCompletions),
        "/messages" => Some(Endpoint::Messages),
        "/responses" => Some(Endpoint::Responses),
        "/embeddings" => Some(Endpoint::Embeddings),
        _ => None,
    }
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
        let rest = catalog_path_rest(path)?.trim_end_matches('/');
        match rest.strip_prefix("/v1").unwrap_or(rest) {
            "/messages/count_tokens" => Some(Self::CountTokens),
            "/responses/input_tokens" => Some(Self::InputTokens),
            "/responses/compact" => Some(Self::Compact),
            _ => None,
        }
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
/// then uses the serving path, so a GPT session walk onto `/v1/responses` is a byte relay.
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
    /// managed 429 walks the next unused entry; the key is never sent to a different provider.
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
                let header = http::HeaderValue::from_str(value.expose()).ok();
                PoolAuth { value, header }
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
