//! Provider routing and per-provider wire details — **data-driven**.
//!
//! The provider is the **first path segment** of the request (`/{provider}/…`); the rest of the path
//! is forwarded to the upstream **verbatim** (native passthrough — the gateway holds no per-provider
//! path knowledge). A path with no provider prefix that starts with `/v1` routes by *dialect* —
//! `/v1/messages*` → `anthropic`, else → `openai` — so an OpenAI/Anthropic client is drop-in by
//! changing only the host. An unrecognized first segment is a 404 (see `proxy::request_filter`).
//!
//! A provider is a *row* in [`KNOWN_PROVIDERS`] (name, upstream authority, dialect, auth scheme) —
//! adding an OpenAI-wire provider (Groq, DeepSeek, Together, …) is one line there, no new code
//! paths. Operators can also add/override providers from config (see `state`/`config`). We do not
//! translate between dialects — that's deliberately out of scope.

use crate::circuit_breaker::CircuitBreaker;
use crate::metrics::ProviderMetrics;
use crate::secret::Secret;

/// The default API prefix OpenAI/Anthropic clients use. A request with no provider segment that
/// starts with this is routed to a default provider by [`dialect_for_path`](crate::proxy) (the
/// bare-path drop-in case); anything else with an unknown first segment is a 404.
pub const DEFAULT_PREFIX: &str = "/v1";

/// Which API surface the client called. Drives usage parsing and the bare-path default provider.
/// On a provider-prefixed request it's the selected provider's own [`Provider::dialect`]; on a
/// bare-path request it's derived from the path (`proxy::dialect_for_path`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    OpenAI,
    Anthropic,
}

impl Dialect {
    /// Parse a config-supplied dialect name (case-insensitive): `"openai"` or `"anthropic"`. Used
    /// only for a **config-added** provider (`state::build_providers`) — a known provider's dialect
    /// is fixed in [`KNOWN_PROVIDERS`] and never parsed from a string. `None` on anything else, which
    /// the caller turns into a hard boot failure rather than a silent default: real Anthropic-wire
    /// vendors (MiniMax, MiniMax-CN, Kimi-Coding) added via config with a typo'd dialect must not
    /// silently fall back to OpenAI-wire, which zero-bills every request (see
    /// `usage::openai_body`'s dialect-mismatch guard for the runtime backstop).
    pub fn parse_config(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "openai" => Some(Dialect::OpenAI),
            "anthropic" => Some(Dialect::Anthropic),
            _ => None,
        }
    }
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

    /// Parse a config-supplied auth scheme name (case-insensitive, `-`/`_` ignored): `"bearer"` or
    /// `"x-api-key"`/`"xapikey"`. Used only for a **config-added** provider (`state::build_providers`)
    /// — a known provider's scheme is fixed in [`KNOWN_PROVIDERS`]. `None` on anything else, which the
    /// caller turns into a hard boot failure rather than a silent default.
    pub fn parse_config(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "bearer" => Some(AuthScheme::Bearer),
            "xapikey" => Some(AuthScheme::XApiKey),
            _ => None,
        }
    }
}

/// Static wire facts for a known provider. Adding a provider = one row in [`KNOWN_PROVIDERS`].
pub struct ProviderSpec {
    pub name: &'static str,
    /// Default upstream `host:port` (TLS:443). Overridable per-provider via config.
    pub authority: &'static str,
    /// The provider's wire format — drives usage parsing and `stream_options` injection eligibility.
    /// (We forward the client's path verbatim, so the *path* doesn't tell us the wire format; the
    /// provider does.)
    pub dialect: Dialect,
    pub auth: AuthScheme,
}

/// The providers the gateway knows out of the box. All but Anthropic speak the OpenAI wire format
/// (Bearer auth, chat/completions + embeddings); a new one is a single row here, then reachable at
/// `/{name}/…`. (Config can add further providers of either dialect, or override any authority — see
/// `state::build_providers`.)
///
/// We forward the path after `/{name}` **verbatim**, so the gateway carries no per-provider mount
/// path — the client uses the provider's native base path (e.g. `/groq/openai/v1/chat/completions`,
/// `/fireworks/inference/v1/chat/completions`), exactly as it would hitting the provider directly.
/// Each row's `authority`/`auth` is verified against the provider's **official** docs (cited inline)
/// as of 2026-05; the client-facing native path is noted alongside as a convenience.
pub const KNOWN_PROVIDERS: &[ProviderSpec] = &[
    // docs: https://platform.openai.com/docs/api-reference/authentication — base https://api.openai.com/v1, Bearer.
    // Client path: /openai/v1/… (or bare /v1/… as the default).
    ProviderSpec {
        name: "openai",
        authority: "api.openai.com:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.claude.com/en/api/messages — base https://api.anthropic.com, Messages at /v1/messages,
    // auth is `x-api-key` (NOT Bearer). The required `anthropic-version` header is the client's; we pass it through.
    // Client path: /anthropic/v1/messages (or bare /v1/messages as the default).
    ProviderSpec {
        name: "anthropic",
        authority: "api.anthropic.com:443",
        dialect: Dialect::Anthropic,
        auth: AuthScheme::XApiKey,
    },
    // docs: https://openrouter.ai/docs/quickstart — base https://openrouter.ai/api/v1, Bearer.
    // Client path: /openrouter/api/v1/chat/completions.
    ProviderSpec {
        name: "openrouter",
        authority: "openrouter.ai:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.fireworks.ai/tools-sdks/openai-compatibility — base https://api.fireworks.ai/inference/v1, Bearer.
    // Client path: /fireworks/inference/v1/chat/completions.
    ProviderSpec {
        name: "fireworks",
        authority: "api.fireworks.ai:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://console.groq.com/docs/openai — base https://api.groq.com/openai/v1, Bearer.
    // Client path: /groq/openai/v1/chat/completions.
    ProviderSpec {
        name: "groq",
        authority: "api.groq.com:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://api-docs.deepseek.com/ — base https://api.deepseek.com/v1 (the `/v1` is an OpenAI-compat alias,
    // not API versioning); /v1/chat/completions is officially supported. Bearer. Client path: /deepseek/v1/….
    ProviderSpec {
        name: "deepseek",
        authority: "api.deepseek.com:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.together.ai/docs/openai-api-compatibility — base https://api.together.ai/v1, Bearer.
    // Canonical host is `api.together.ai`; the legacy `api.together.xyz` is still live but no longer documented.
    // Client path: /together/v1/….
    ProviderSpec {
        name: "together",
        authority: "api.together.ai:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://inference-docs.cerebras.ai/resources/openai — base https://api.cerebras.ai/v1, Bearer.
    // Client path: /cerebras/v1/….
    ProviderSpec {
        name: "cerebras",
        authority: "api.cerebras.ai:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.mistral.ai/api/ — base https://api.mistral.ai/v1, Bearer. Client path: /mistral/v1/….
    ProviderSpec {
        name: "mistral",
        authority: "api.mistral.ai:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // docs: https://docs.x.ai/docs/api-reference — base https://api.x.ai/v1, Bearer. Reasoning models are slow:
    // the generous read/idle timeouts (see `config`) matter here. Client path: /xai/v1/….
    ProviderSpec {
        name: "xai",
        authority: "api.x.ai:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
    // OpenAI Codex (ChatGPT Plus/Pro subscription backend) — reached only via a stored OpenAI-Codex
    // OAuth credential's own bearer token, never a pool key (no `AI_POOL_KEY_OPENAI_CODEX` exists;
    // this row only ever serves BYO traffic). Base https://chatgpt.com/backend-api, Bearer — a
    // genuinely static host, unlike GitHub Copilot's account-specific proxy endpoint (see
    // `agent_core::client::RouteOverride`'s doc comment for why Copilot instead bypasses the gateway
    // entirely rather than getting a row here). Client path: /openai-codex/backend-api/codex/responses
    // (see `packages/ai/src/api/openai-codex-responses.ts` in pi-mono, `resolveCodexUrl`).
    ProviderSpec {
        name: "openai-codex",
        authority: "chatgpt.com:443",
        dialect: Dialect::OpenAI,
        auth: AuthScheme::Bearer,
    },
];

/// The default provider name for a dialect — used only for the **bare-path** request (no provider
/// segment), where the dialect is derived from the path. A provider-prefixed request names its
/// provider directly.
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
    /// The provider's wire format (usage parsing + injection eligibility). See [`ProviderSpec::dialect`].
    pub dialect: Dialect,
    pub auth: AuthScheme,
    /// Precomputed managed auth header value (`Bearer <key>` / bare key). `None` ⇒ no pool key is
    /// configured for this provider ⇒ managed requests to it are rejected (503). Kept in `Secret`
    /// for the redacting-`Debug` + zeroize-on-drop hygiene of the underlying key.
    pub pool_auth_value: Option<Secret>,
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
        Provider {
            name: name.to_string(),
            authority,
            host,
            dialect,
            auth,
            pool_auth_value,
            metrics,
            breaker,
        }
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
    fn anthropic_is_the_only_anthropic_dialect() {
        // Dialect drives usage parsing + injection; getting Anthropic's wire wrong mis-meters it.
        for spec in KNOWN_PROVIDERS {
            let want = if spec.name == "anthropic" {
                Dialect::Anthropic
            } else {
                Dialect::OpenAI
            };
            assert_eq!(spec.dialect, want, "{} dialect", spec.name);
        }
    }

    #[test]
    fn dialect_parse_config_accepts_known_case_insensitive() {
        assert_eq!(Dialect::parse_config("openai"), Some(Dialect::OpenAI));
        assert_eq!(Dialect::parse_config("OpenAI"), Some(Dialect::OpenAI));
        assert_eq!(Dialect::parse_config("anthropic"), Some(Dialect::Anthropic));
        assert_eq!(Dialect::parse_config("Anthropic"), Some(Dialect::Anthropic));
        assert_eq!(Dialect::parse_config("bogus"), None);
        assert_eq!(Dialect::parse_config(""), None);
    }

    #[test]
    fn auth_scheme_parse_config_accepts_known_variants() {
        assert_eq!(AuthScheme::parse_config("bearer"), Some(AuthScheme::Bearer));
        assert_eq!(AuthScheme::parse_config("Bearer"), Some(AuthScheme::Bearer));
        assert_eq!(
            AuthScheme::parse_config("x-api-key"),
            Some(AuthScheme::XApiKey)
        );
        assert_eq!(
            AuthScheme::parse_config("xapikey"),
            Some(AuthScheme::XApiKey)
        );
        assert_eq!(
            AuthScheme::parse_config("x_api_key"),
            Some(AuthScheme::XApiKey)
        );
        assert_eq!(AuthScheme::parse_config("bogus"), None);
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
            Dialect::OpenAI,
            AuthScheme::Bearer,
            Some("sk-x"),
            ProviderMetrics::disconnected(),
            None,
        );
        assert_eq!(p.host, "api.openai.com");
        assert_eq!(p.dialect, Dialect::OpenAI);
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
}
