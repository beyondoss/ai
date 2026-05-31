//! Provider routing and per-provider wire details — **data-driven**.
//!
//! Passthrough-first: the ingress *dialect* (which API surface the client called) picks the default
//! provider; an `x-beyond-provider: <name>` header selects any registered provider by name. A
//! provider is a *row* in [`KNOWN_PROVIDERS`] (name, upstream authority, base path, auth scheme) —
//! adding an OpenAI-wire provider (Groq, DeepSeek, Together, …) is one line there, no new code
//! paths, no enum, no match arms. Operators can also add/override providers from config (see
//! `state`/`config`). We do not translate between dialects — that's deliberately out of scope.

use crate::secret::Secret;

/// The path prefix client SDKs use (OpenAI + Anthropic both mount their API under `/v1`). A provider
/// whose `base_path` differs has this leading segment rewritten to its prefix (see
/// [`Provider::upstream_path`]).
pub const CLIENT_PREFIX: &str = "/v1";

/// Which API surface the client called. Drives usage parsing and the default provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    OpenAI,
    Anthropic,
}

/// How the upstream expects the API key. OpenAI-wire providers use `Authorization: Bearer <key>`;
/// Anthropic uses the `x-api-key` header. The gateway swaps the client's virtual key for the real
/// pool key in whichever header the upstream wants (see `proxy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthScheme {
    Bearer,
    XApiKey,
}

impl AuthScheme {
    /// The request header the upstream expects the key in.
    pub fn header(self) -> &'static str {
        match self {
            AuthScheme::Bearer => "authorization",
            AuthScheme::XApiKey => "x-api-key",
        }
    }

    /// Format `key` as the upstream wants it for [`Self::header`].
    pub fn format(self, key: &str) -> String {
        match self {
            AuthScheme::Bearer => format!("Bearer {key}"),
            AuthScheme::XApiKey => key.to_string(),
        }
    }
}

/// Static wire facts for a known provider. Adding a provider = one row in [`KNOWN_PROVIDERS`].
pub struct ProviderSpec {
    pub name: &'static str,
    /// Default upstream `host:port` (TLS:443). Overridable per-provider via config.
    pub authority: &'static str,
    /// Where the provider mounts the OpenAI-wire surface. The client always calls `/v1/…` (its
    /// SDK's fixed prefix); for a provider whose `base_path != "/v1"` the gateway rewrites that
    /// leading segment (e.g. Groq serves under `/openai/v1`, so `/v1/chat/completions` →
    /// `/openai/v1/chat/completions`). Most providers mount at `/v1`, so this is usually `"/v1"`.
    pub base_path: &'static str,
    pub auth: AuthScheme,
}

/// The providers the gateway knows out of the box. All but Anthropic speak the OpenAI wire format
/// (Bearer auth, chat/completions + embeddings); they differ by authority and where they mount that
/// surface (`base_path`) — a new one is a single row here, then reachable via `x-beyond-provider:
/// <name>`. (Config can add further OpenAI-wire providers or override any authority — see
/// `state::build_providers`.)
///
/// `base_path` values are deliberate, not cosmetic: Groq/OpenRouter/Fireworks do **not** mount at
/// `/v1`, so a verbatim path passthrough would 404 against the real provider.
///
/// Every row's `authority`/`base_path`/`auth` was verified against the provider's **official** docs
/// (cited inline) as of 2026-05; the citation is the source of truth if a provider later moves an
/// endpoint. These are static facts, so doc-verification (not a live call) is the right proof.
pub const KNOWN_PROVIDERS: &[ProviderSpec] = &[
    // docs: https://platform.openai.com/docs/api-reference/authentication — base https://api.openai.com/v1, Bearer.
    ProviderSpec {
        name: "openai",
        authority: "api.openai.com:443",
        base_path: "/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.claude.com/en/api/messages — base https://api.anthropic.com, Messages at /v1/messages,
    // auth is `x-api-key` (NOT Bearer). The required `anthropic-version` header is the client's; we pass it through.
    ProviderSpec {
        name: "anthropic",
        authority: "api.anthropic.com:443",
        base_path: "/v1",
        auth: AuthScheme::XApiKey,
    },
    // docs: https://openrouter.ai/docs/quickstart — base https://openrouter.ai/api/v1 (note `/api/v1`, not `/v1`), Bearer.
    ProviderSpec {
        name: "openrouter",
        authority: "openrouter.ai:443",
        base_path: "/api/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.fireworks.ai/tools-sdks/openai-compatibility — base https://api.fireworks.ai/inference/v1, Bearer.
    ProviderSpec {
        name: "fireworks",
        authority: "api.fireworks.ai:443",
        base_path: "/inference/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://console.groq.com/docs/openai — base https://api.groq.com/openai/v1 (note `/openai/v1`), Bearer.
    ProviderSpec {
        name: "groq",
        authority: "api.groq.com:443",
        base_path: "/openai/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://api-docs.deepseek.com/ — base https://api.deepseek.com/v1 (the `/v1` is an OpenAI-compat alias,
    // not API versioning); /v1/chat/completions is officially supported. Bearer.
    ProviderSpec {
        name: "deepseek",
        authority: "api.deepseek.com:443",
        base_path: "/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.together.ai/docs/openai-api-compatibility — base https://api.together.ai/v1, Bearer.
    // Canonical host is `api.together.ai`; the legacy `api.together.xyz` is still live but no longer documented.
    ProviderSpec {
        name: "together",
        authority: "api.together.ai:443",
        base_path: "/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://inference-docs.cerebras.ai/resources/openai — base https://api.cerebras.ai/v1, Bearer.
    ProviderSpec {
        name: "cerebras",
        authority: "api.cerebras.ai:443",
        base_path: "/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.mistral.ai/api/ — base https://api.mistral.ai/v1, Bearer.
    ProviderSpec {
        name: "mistral",
        authority: "api.mistral.ai:443",
        base_path: "/v1",
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.x.ai/docs/api-reference — base https://api.x.ai/v1, Bearer. Reasoning models are slow:
    // the generous read/idle timeouts (see `config`) matter here.
    ProviderSpec {
        name: "xai",
        authority: "api.x.ai:443",
        base_path: "/v1",
        auth: AuthScheme::Bearer,
    },
];

/// The default provider name for a dialect, used when no `x-beyond-provider` override is given.
/// (Model-based auto-routing isn't possible — the body isn't read before peer selection — so the
/// long tail is reached explicitly via the header.)
pub fn dialect_default(d: Dialect) -> &'static str {
    match d {
        Dialect::OpenAI => "openai",
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
    /// Where the provider mounts the OpenAI-wire surface (see [`ProviderSpec::base_path`]).
    pub base_path: String,
    pub auth: AuthScheme,
    /// Precomputed managed auth header value (`Bearer <key>` / bare key). `None` ⇒ no pool key is
    /// configured for this provider ⇒ managed requests to it are rejected (503). Kept in `Secret`
    /// for the redacting-`Debug` + zeroize-on-drop hygiene of the underlying key.
    pub pool_auth_value: Option<Secret>,
}

impl Provider {
    /// Resolve a provider from its name, upstream authority, base path, auth scheme, and (optional)
    /// pool key. Derives the bare host and precomputes the managed auth header value once.
    pub fn resolve(
        name: &str,
        authority: String,
        base_path: &str,
        auth: AuthScheme,
        pool_key: Option<&str>,
    ) -> Self {
        let host = authority
            .split(':')
            .next()
            .unwrap_or(&authority)
            .to_string();
        let pool_auth_value = pool_key.map(|k| Secret::new(auth.format(k)));
        Provider {
            name: name.to_string(),
            authority,
            host,
            base_path: base_path.to_string(),
            auth,
            pool_auth_value,
        }
    }

    /// Map a client request path to the upstream path for this provider. The client's SDK uses the
    /// fixed `/v1` prefix; if this provider mounts elsewhere (`base_path != "/v1"`) the leading
    /// `/v1` is replaced. Returns `None` when no rewrite is needed (the common `/v1` case, or a
    /// path that doesn't start with `/v1`), so the hot path skips reallocating the URI.
    pub fn upstream_path(&self, client_path: &str) -> Option<String> {
        if self.base_path == CLIENT_PREFIX {
            return None;
        }
        // Only remap when the segment is exactly `/v1` (followed by `/` or end), never a prefix
        // match like `/v1beta`. `rest` keeps the remainder (incl. its leading `/`, or empty).
        let rest = client_path.strip_prefix(CLIENT_PREFIX)?;
        if !rest.is_empty() && !rest.starts_with('/') {
            return None;
        }
        Some(format!("{}{}", self.base_path, rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_defaults() {
        assert_eq!(dialect_default(Dialect::OpenAI), "openai");
        assert_eq!(dialect_default(Dialect::Anthropic), "anthropic");
    }

    #[test]
    fn known_provider_names_are_unique() {
        let mut names: Vec<_> = KNOWN_PROVIDERS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(
            before,
            names.len(),
            "duplicate provider name in KNOWN_PROVIDERS"
        );
    }

    #[test]
    fn auth_scheme_formats_and_headers() {
        assert_eq!(AuthScheme::Bearer.header(), "authorization");
        assert_eq!(AuthScheme::XApiKey.header(), "x-api-key");
        assert_eq!(AuthScheme::Bearer.format("k"), "Bearer k");
        // Anthropic wants the bare key (no `Bearer`). Getting this wrong → upstream 401.
        assert_eq!(AuthScheme::XApiKey.format("k"), "k");
    }

    #[test]
    fn resolve_derives_host_and_pool_auth() {
        let p = Provider::resolve(
            "openai",
            "api.openai.com:443".to_string(),
            "/v1",
            AuthScheme::Bearer,
            Some("sk-x"),
        );
        assert_eq!(p.host, "api.openai.com");
        assert_eq!(p.pool_auth_value.as_ref().unwrap().expose(), "Bearer sk-x");

        // No pool key ⇒ no managed auth value (managed requests to it would 503).
        let a = Provider::resolve(
            "anthropic",
            "api.anthropic.com:443".to_string(),
            "/v1",
            AuthScheme::XApiKey,
            None,
        );
        assert!(a.pool_auth_value.is_none());
    }

    #[test]
    fn upstream_path_rewrites_only_non_v1_bases() {
        let v1 = Provider::resolve("openai", "h:443".into(), "/v1", AuthScheme::Bearer, None);
        // `/v1` provider: no rewrite (None) — the hot path passes the client path through verbatim.
        assert_eq!(v1.upstream_path("/v1/chat/completions"), None);

        let groq = Provider::resolve(
            "groq",
            "h:443".into(),
            "/openai/v1",
            AuthScheme::Bearer,
            None,
        );
        assert_eq!(
            groq.upstream_path("/v1/chat/completions").as_deref(),
            Some("/openai/v1/chat/completions")
        );
        // Anthropic-style messages path under a remapped base, and the bare prefix.
        assert_eq!(
            groq.upstream_path("/v1/embeddings").as_deref(),
            Some("/openai/v1/embeddings")
        );
        assert_eq!(groq.upstream_path("/v1").as_deref(), Some("/openai/v1"));
        // A non-`/v1` path (e.g. a health probe) is left alone, as is a `/v1beta`-style false match.
        assert_eq!(groq.upstream_path("/healthz"), None);
        assert_eq!(groq.upstream_path("/v1beta/models"), None);
    }
}
