//! Provider routing and per-provider wire details — **data-driven**.
//!
//! The provider is the **first path segment** of the request (`/{provider}/…`); the rest of the path
//! is forwarded to the upstream **verbatim** (native passthrough — the gateway holds no per-provider
//! path knowledge). A path with no provider prefix that starts with `/v1` routes by *dialect* —
//! `/v1/messages*` → `anthropic`, else → `openai` — so an OpenAI/Anthropic client is drop-in by
//! changing only the host. An unrecognized first segment is a 404 (see `proxy::request_filter`).
//!
//! A provider is a *row* in [`known_providers`] (name, upstream authority, wire format, auth scheme) —
//! adding an OpenAI-wire provider (Groq, DeepSeek, Together, …) is one line there, no new code
//! paths. Operators can also add/override providers from config (see `state`/`config`). We do not
//! translate between dialects — that's deliberately out of scope.
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

/// The default API prefix OpenAI/Anthropic clients use. A request with no provider segment whose
/// path is exactly this or begins with this plus `/` (see [`is_default_prefix`]) is routed to a
/// default provider by [`dialect_for_path`](crate::proxy) (the bare-path drop-in case); anything
/// else with an unknown first segment is a 404.
pub const DEFAULT_PREFIX: &str = "/v1";

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

/// A *resolved* provider: static wire facts + the boot-resolved upstream authority/host + (for
/// managed traffic) the precomputed pool auth header value. Built once at boot (see
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
    /// Precomputed managed auth header value (`Bearer <key>` / bare key). `None` ⇒ no pool key is
    /// configured for this provider ⇒ managed requests to it are rejected (503). Kept in `Secret`
    /// for the redacting-`Debug` + zeroize-on-drop hygiene of the underlying key.
    pub pool_auth_value: Option<Secret>,
    /// `pool_auth_value` as a ready-to-insert [`http::HeaderValue`].
    ///
    /// `insert_header(name, &str)` runs `HeaderValue::from_str`, which validates the bytes and
    /// copies them into a fresh `Bytes` — a heap allocation per managed request for a value fixed at
    /// boot. Cloning a `HeaderValue` is a refcount bump instead. Same for [`Self::host_header`].
    ///
    /// `None` only if the configured key isn't a legal header value (a stray newline, say), which no
    /// key that could ever have worked would be — `insert_header` would have rejected it per
    /// request. The caller falls back to the string form so that stays true rather than becoming a
    /// silent 503.
    ///
    /// Hygiene note: a `HeaderValue` is not zeroized on drop, so this is one long-lived plaintext
    /// copy of the pool key. That is a net *improvement* — `secret.rs` already concedes the key is
    /// "copied into Pingora's request headers we don't own", and previously that copy was made and
    /// freed thousands of times a second, scattering key bytes across the heap. The `Secret` is kept
    /// for the redacting `Debug`.
    pub pool_auth_header: Option<http::HeaderValue>,
    /// `host` as a ready-to-insert `HeaderValue` — see [`Self::pool_auth_header`].
    pub host_header: Option<http::HeaderValue>,
    /// Per-provider metric handles, resolved once here so the response path bumps a direct
    /// counter/histogram instead of a string-keyed label lookup per response.
    pub metrics: ProviderMetrics,
    /// Per-provider circuit breaker, shared across all callers to this provider. `None` when the
    /// breaker is disabled (`circuit_breaker_threshold == 0`). Checked before connect and fed the
    /// 5xx/connect outcome — see `proxy`. Lock-free, so the hot path reads it without contention.
    pub breaker: Option<CircuitBreaker>,
}

impl Provider {
    /// Resolve a provider from its name, upstream authority, dialect, auth scheme, (optional) pool
    /// key, and pre-resolved per-provider metric handles. Derives the bare host and precomputes the
    /// managed auth header value once.
    pub fn resolve(
        name: &str,
        authority: String,
        dialect: Dialect,
        auth: AuthScheme,
        pool_key: Option<&str>,
        metrics: ProviderMetrics,
        breaker: Option<CircuitBreaker>,
    ) -> Self {
        let host = authority
            .split(':')
            .next()
            .unwrap_or(&authority)
            .to_string();
        let pool_auth_value = pool_key.map(|k| Secret::new(auth.format(k)));
        let pool_auth_header = pool_auth_value
            .as_ref()
            .and_then(|s| http::HeaderValue::from_str(s.expose()).ok());
        let host_header = http::HeaderValue::from_str(&host).ok();
        Provider {
            name: name.to_string(),
            authority,
            host,
            dialect,
            auth,
            pool_auth_value,
            pool_auth_header,
            host_header,
            metrics,
            breaker,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Some("sk-x"),
            ProviderMetrics::disconnected(),
            None,
        );
        assert_eq!(p.host, "api.openai.com");
        assert_eq!(p.dialect, Dialect::OpenAi);
        assert_eq!(p.pool_auth_value.as_ref().unwrap().expose(), "Bearer sk-x");

        // No pool key ⇒ no managed auth value (managed requests to it would 503).
        let a = Provider::resolve(
            "anthropic",
            "api.anthropic.com:443".to_string(),
            Dialect::Anthropic,
            AuthScheme::XApiKey,
            None,
            ProviderMetrics::disconnected(),
            None,
        );
        assert!(a.pool_auth_value.is_none());
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
                Some(key),
                ProviderMetrics::disconnected(),
                None,
            );
            assert_eq!(
                p.host_header.as_ref().expect("host is header-safe"),
                &http::HeaderValue::from_str(&p.host).unwrap()
            );
            let av = p.pool_auth_value.as_ref().expect("pool key configured");
            assert_eq!(
                p.pool_auth_header.as_ref().expect("key is header-safe"),
                &http::HeaderValue::from_str(av.expose()).unwrap()
            );
        }

        // No pool key ⇒ no precomputed auth header either (and the 503 path is unchanged).
        let none = Provider::resolve(
            "p",
            "h:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::Bearer,
            None,
            ProviderMetrics::disconnected(),
            None,
        );
        assert!(none.pool_auth_value.is_none());
        assert!(none.pool_auth_header.is_none());

        // A key that is not a legal header value precomputes to `None`, so the caller falls back to
        // the string form and gets the same per-request error it always did — rather than this
        // quietly turning into a 503.
        let bad = Provider::resolve(
            "p",
            "h:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::XApiKey,
            Some("has\nnewline"),
            ProviderMetrics::disconnected(),
            None,
        );
        assert!(bad.pool_auth_value.is_some());
        assert!(bad.pool_auth_header.is_none());
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
            Some("azure-secret"),
            ProviderMetrics::disconnected(),
            None,
        );
        assert_eq!(azure.auth.header(), "api-key");
        assert_eq!(
            azure.pool_auth_value.as_ref().unwrap().expose(),
            "azure-secret"
        );
    }
}
