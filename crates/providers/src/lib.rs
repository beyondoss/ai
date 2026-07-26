//! The provider table — **one** source of truth for "which upstream is this, and how do I talk to it".
//!
//! Two very different consumers read the same rows:
//!
//! - **The gateway** (`crates/gateway`) routes `/{name}/…` to [`ProviderSpec::authority`], swapping the
//!   client's virtual key for a pool key in whichever header [`ProviderSpec::auth`] names.
//! - **The agent** (`crates/agent`) routes *directly* to [`ProviderSpec::base_url`] when no gateway is
//!   configured, picking up the key from [`ProviderSpec::env_var`] and sending it in that same
//!   [`ProviderSpec::auth`] header. The user never spells the auth scheme out; it falls out of the row.
//!
//! Before this crate existed the same facts were spread across four tables that had to be kept in sync
//! by hand (the gateway's `KNOWN_PROVIDERS`, `agent_core`'s `AggregatorHost`, the agent's
//! `aggregator_host_for_base_url`, and its `default_headers_for_base_url`). They are all rows here now.
//!
//! **Provider is resolved before dialect, never the other way round.** A model id alone cannot tell you
//! the wire format: `anthropic/claude-sonnet-4.5` is an *OpenAI*-wire request when OpenRouter serves it.
//! [`ProviderSpec::wire`] is the authority; see `agent_core::dialect::Dialect::for_model_via_provider`.
//!
//! Adding a provider is one row. Adding a *field* to a row means teaching one of the two consumers
//! something new — think about whether it's really provider knowledge (host, auth, wire) as opposed to
//! *model* knowledge (context window, thinking shape), which belongs in `agent_core::models` instead.

pub mod catalog;

pub use catalog::{Candidate, MAX_CANDIDATES, ModelRoute, for_model};

/// Every upstream this codebase can route to, by either path. The gateway's 11 `/{name}/…` routes and
/// the 5 BYO-only aggregator platforms (reachable only by base-URL override, no gateway-native route)
/// are one enum because they are one *concept* — "which upstream" — and splitting them is what
/// produced the two-tables-out-of-sync problem this crate exists to end.
///
/// `agent_core::models` re-exports this as `AggregatorHost`, its historical name there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderId {
    // --- Gateway-native rows: reachable at `/{name}/…` with a pool key. ---
    OpenAi,
    Anthropic,
    OpenRouter,
    Fireworks,
    Groq,
    DeepSeek,
    Together,
    Cerebras,
    Mistral,
    XAi,
    /// ChatGPT Plus/Pro subscription backend. Gateway-routable, but only ever with a *BYO* OAuth
    /// bearer — there is no `AI_POOL_KEY_OPENAI_CODEX`, hence no [`ProviderSpec::env_var`].
    OpenAiCodex,

    // --- BYO-only rows: no gateway route, named via a `models.json` `base_url` or `AI_PROVIDER`. ---
    HuggingFace,
    Nvidia,
    /// `api.kimi.com/coding` — disambiguated from moonshotai-native by base URL, not by id shape.
    KimiCoding,
    /// `opencode.ai/zen`. Distinct from [`Self::OpenCodeGo`] despite sharing a hostname: the two
    /// genuinely disagree on wire dialect and real numbers for several shared bare ids (`minimax-m3`,
    /// `glm-5.1`), so [`for_host`] splits them on the `/zen` vs `/zen/go` path segment.
    OpenCodeZen,
    /// `opencode.ai/zen/go`. See [`Self::OpenCodeZen`].
    OpenCodeGo,
}

impl ProviderId {
    /// Every variant, in [`Self::index`] order.
    ///
    /// Exists so a consumer can build a fixed-size array keyed by id instead of hashing a name — the
    /// gateway's model-routed path switches providers between connect attempts and wants that lookup
    /// to be an index, not a string compare. Also spares `every_id_has_exactly_one_row` from
    /// hand-maintaining its own copy of the variant list.
    pub const ALL: [ProviderId; 16] = [
        ProviderId::OpenAi,
        ProviderId::Anthropic,
        ProviderId::OpenRouter,
        ProviderId::Fireworks,
        ProviderId::Groq,
        ProviderId::DeepSeek,
        ProviderId::Together,
        ProviderId::Cerebras,
        ProviderId::Mistral,
        ProviderId::XAi,
        ProviderId::OpenAiCodex,
        ProviderId::HuggingFace,
        ProviderId::Nvidia,
        ProviderId::KimiCoding,
        ProviderId::OpenCodeZen,
        ProviderId::OpenCodeGo,
    ];

    /// How many variants there are — the width of an id-keyed array.
    pub const COUNT: usize = Self::ALL.len();

    /// A dense index in `0..COUNT`, for array-keyed lookup.
    ///
    /// The exhaustive `match` is the point: adding a variant fails to compile here rather than
    /// silently landing outside an array built from [`Self::COUNT`].
    pub fn index(self) -> usize {
        match self {
            ProviderId::OpenAi => 0,
            ProviderId::Anthropic => 1,
            ProviderId::OpenRouter => 2,
            ProviderId::Fireworks => 3,
            ProviderId::Groq => 4,
            ProviderId::DeepSeek => 5,
            ProviderId::Together => 6,
            ProviderId::Cerebras => 7,
            ProviderId::Mistral => 8,
            ProviderId::XAi => 9,
            ProviderId::OpenAiCodex => 10,
            ProviderId::HuggingFace => 11,
            ProviderId::Nvidia => 12,
            ProviderId::KimiCoding => 13,
            ProviderId::OpenCodeZen => 14,
            ProviderId::OpenCodeGo => 15,
        }
    }
}

/// The provider's wire format — which API shape its endpoint speaks.
///
/// Two variants, not three: this is the *provider family*, not the request's exact wire shape. An
/// OpenAI-wire provider serves both Chat Completions and Responses models, and which one a given
/// request uses is a property of the **model** (`agent_core::models::capabilities(m).api`), not of the
/// provider. `agent_core::dialect::Dialect` is the three-variant type that names the request shape;
/// this narrows *which of its arms are reachable*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    OpenAi,
    Anthropic,
}

impl WireFormat {
    /// Parse a config-supplied wire name (case-insensitive): `"openai"` or `"anthropic"`. Used only
    /// for a **config-added** gateway provider (`gateway::state::build_providers`) — a known
    /// provider's wire is fixed in [`PROVIDERS`] and never parsed from a string. `None` on anything
    /// else, which the caller turns into a hard boot failure rather than a silent default: real
    /// Anthropic-wire vendors (MiniMax, Kimi-Coding) added via config with a typo'd dialect must not
    /// silently fall back to OpenAI-wire, which zero-bills every request (see
    /// `gateway::usage::openai_body`'s dialect-mismatch guard for the runtime backstop).
    pub fn parse_config(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "openai" => Some(WireFormat::OpenAi),
            "anthropic" => Some(WireFormat::Anthropic),
            _ => None,
        }
    }
}

/// How the upstream expects the API key. OpenAI-wire providers use `Authorization: Bearer <key>`;
/// Anthropic uses the `x-api-key` header; Azure OpenAI uses the bare key in `api-key` (no `Bearer`
/// prefix — same bare-key shape as [`Self::XApiKey`], just a different header name; see
/// `packages/ai/src/api/azure-openai-responses.ts` in pi-mono, whose `AzureOpenAI` SDK client always
/// authenticates this way).
///
/// The gateway swaps the client's virtual key for the real pool key in whichever header this names
/// (`gateway::proxy`). The agent's direct route sends the *user's own* key in that same header —
/// which is why an `ANTHROPIC_API_KEY` direct route gets `x-api-key` without anyone configuring it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthScheme {
    Bearer,
    XApiKey,
    /// Azure OpenAI's `api-key` header (bare key, no prefix). Azure's per-resource host isn't knowable
    /// at compile time, so it is never a [`PROVIDERS`] row — a config-added gateway provider sets
    /// `provider_auth_schemes.<name> = "api-key"`, and the agent reaches it via a `models.json`
    /// `auth_header`.
    ApiKey,
    /// A `Bearer`-prefixed value in a differently-named header — Cloudflare AI Gateway's
    /// `cf-aig-authorization: Bearer <key>` (see `packages/ai/src/providers/cloudflare-auth.ts` in
    /// pi-mono). Added as a low-risk enum shape so the header format is expressible, but **nothing
    /// constructs one yet**: Cloudflare AI Gateway's base URL is templated per account+gateway id
    /// (`https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/{provider}`), which neither the
    /// gateway's one-fixed-authority-per-name model nor a static [`PROVIDERS`] row can express. Not
    /// reachable from [`AuthScheme::parse_config`] either (a `&'static str` header name can't be built
    /// from a runtime config string without leaking).
    CustomHeader(&'static str),
}

impl AuthScheme {
    /// The request header the upstream expects the key in.
    pub fn header(self) -> &'static str {
        match self {
            AuthScheme::Bearer => "authorization",
            AuthScheme::XApiKey => "x-api-key",
            AuthScheme::ApiKey => "api-key",
            AuthScheme::CustomHeader(name) => name,
        }
    }

    /// Format `key` as the upstream wants it for [`Self::header`].
    pub fn format(self, key: &str) -> String {
        match self {
            AuthScheme::Bearer | AuthScheme::CustomHeader(_) => format!("Bearer {key}"),
            AuthScheme::XApiKey | AuthScheme::ApiKey => key.to_string(),
        }
    }

    /// The value prefix [`Self::format`] applies, as a standalone string — the shape
    /// `agent_core::client::DirectRouting::auth_header_prefix` wants. `None` ⇒ the key is sent bare.
    pub fn value_prefix(self) -> Option<&'static str> {
        match self {
            AuthScheme::Bearer | AuthScheme::CustomHeader(_) => Some("Bearer "),
            AuthScheme::XApiKey | AuthScheme::ApiKey => None,
        }
    }

    /// Parse a config-supplied auth scheme name (case-insensitive, `-`/`_` ignored): `"bearer"`,
    /// `"x-api-key"`/`"xapikey"`, or `"api-key"`/`"apikey"` (Azure OpenAI). Used only for a
    /// **config-added** gateway provider — a known provider's scheme is fixed in [`PROVIDERS`]. `None`
    /// on anything else, which the caller turns into a hard boot failure rather than a silent default.
    /// Note `"x-api-key"` and `"api-key"` normalize to *different* strings (`"xapikey"` vs `"apikey"`)
    /// despite both being hyphen-stripped, so they don't collide.
    pub fn parse_config(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "bearer" => Some(AuthScheme::Bearer),
            "xapikey" => Some(AuthScheme::XApiKey),
            "apikey" => Some(AuthScheme::ApiKey),
            _ => None,
        }
    }
}

/// Static wire facts for one provider. Adding a provider = one row in [`PROVIDERS`].
#[derive(Debug, PartialEq, Eq)]
pub struct ProviderSpec {
    pub id: ProviderId,
    /// The gateway's path segment (`/{name}/…`) **and** the value `AI_PROVIDER` takes. One string for
    /// both, so `AI_PROVIDER=groq` and `/groq/…` can never drift apart.
    pub name: &'static str,
    /// Upstream `host:port` for the gateway's proxy dial (TLS:443). Overridable per-provider via
    /// gateway config.
    ///
    /// Deliberately *not* the same field as [`Self::base_url`]: the gateway forwards the client's path
    /// **verbatim** and therefore stores no base path at all, while a direct client needs the full
    /// base including it (`https://openrouter.ai/api/v1`, not just `openrouter.ai`).
    pub authority: &'static str,
    /// Full base URL for a **direct** (gateway-less) client request, base path included. `None` for a
    /// provider with no direct-route story: OAuth-only ([`ProviderId::OpenAiCodex`]) or path-templated
    /// per account ([`ProviderId::KimiCoding`], the OpenCode rows), which a static row can't express.
    ///
    /// Anthropic's is bare (`https://api.anthropic.com`, no `/v1`) because
    /// `agent_core::dialect::Dialect::endpoint_path` appends `/v1/messages`; the OpenAI-wire rows carry
    /// their `/v1` because their endpoint paths are appended bare (see
    /// `agent::gateway_credential::direct_route_path`, which detects exactly this).
    pub base_url: Option<&'static str>,
    /// The provider's wire family. Drives the gateway's usage parsing and `stream_options` injection
    /// eligibility; drives the agent's dialect selection. (We forward the path verbatim, so the *path*
    /// never tells us the wire format — the provider does.)
    pub wire: WireFormat,
    pub auth: AuthScheme,
    /// The conventional env var holding a user's own key for this provider — the whole point of the
    /// direct route. `None` ⇒ no direct route (see [`Self::base_url`]). These names are not invented
    /// here: they are the ones the provider's own SDK/docs use, and the ones
    /// `crates/gateway/tests/smoke.rs` already reads.
    pub env_var: Option<&'static str>,
    /// Model-id prefixes this provider serves **natively** — the signal [`for_model_id`] resolves a
    /// bare `--model` against, so `ANTHROPIC_API_KEY` + `claude-opus-4-8` needs no further config.
    ///
    /// Empty for every *aggregator* (OpenRouter, Groq, Together, Fireworks, Cerebras, HuggingFace,
    /// Nvidia), and that is load-bearing rather than an omission: they serve many vendors' ids, so no
    /// prefix identifies them. `moonshotai/kimi-k2.6` is equally a Fireworks, Together, and OpenRouter
    /// id — guessing between them from the string is exactly the mis-route this crate exists to
    /// prevent. Aggregators are named explicitly (`AI_PROVIDER=openrouter`).
    pub model_id_match: &'static [&'static str],
    /// Hostnames that identify this provider in a user-supplied base URL — the `models.json`
    /// `base_url` override path, where the provider is known only by where it points. See [`for_host`].
    pub host_match: &'static [&'static str],
    /// Narrows [`Self::host_match`] by path prefix, for the one case where a hostname isn't enough:
    /// `opencode.ai` hosts two providers that disagree on wire format. `None` ⇒ hostname alone decides.
    pub path_prefix: Option<&'static str>,
    /// Headers this provider *requires* on every request, independent of auth. Not a convenience: the
    /// NVIDIA NIM poll timeout and Kimi's `User-Agent` allowlist check are both hard upstream
    /// requirements. Seeded under any user-supplied headers, which win.
    pub default_headers: &'static [(&'static str, &'static str)],
}

/// Every provider, gateway-native and BYO-only alike.
///
/// Each row's `authority`/`base_url`/`auth` is verified against the provider's **official** docs (cited
/// inline) as of 2026-05. The gateway-facing client path is noted alongside as a convenience — the
/// gateway mounts no path of its own, so it's the provider's native one.
pub const PROVIDERS: &[ProviderSpec] = &[
    // docs: https://platform.openai.com/docs/api-reference/authentication — base https://api.openai.com/v1, Bearer.
    // Gateway path: /openai/v1/… (or bare /v1/… as the default).
    ProviderSpec {
        id: ProviderId::OpenAi,
        name: "openai",
        authority: "api.openai.com:443",
        base_url: Some("https://api.openai.com/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("OPENAI_API_KEY"),
        // `o1`/`o3`/`o4` are prefix-matched without a trailing `-` on purpose: the family has bare ids
        // (`o3`, `o4-mini`). `codex-` is the standalone id shape; the Codex *subscription* backend is
        // its own row below and is selected by credential, not by id.
        model_id_match: &["gpt-", "chatgpt-", "o1", "o3", "o4", "codex-"],
        host_match: &["api.openai.com"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://docs.claude.com/en/api/messages — base https://api.anthropic.com, Messages at /v1/messages,
    // auth is `x-api-key` (NOT Bearer). The required `anthropic-version` header is the client's own.
    // Gateway path: /anthropic/v1/messages (or bare /v1/messages as the default).
    ProviderSpec {
        id: ProviderId::Anthropic,
        name: "anthropic",
        authority: "api.anthropic.com:443",
        base_url: Some("https://api.anthropic.com"),
        wire: WireFormat::Anthropic,
        auth: AuthScheme::XApiKey,
        env_var: Some("ANTHROPIC_API_KEY"),
        model_id_match: &["claude-"],
        host_match: &["api.anthropic.com"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://openrouter.ai/docs/quickstart — base https://openrouter.ai/api/v1, Bearer.
    // Gateway path: /openrouter/api/v1/chat/completions.
    ProviderSpec {
        id: ProviderId::OpenRouter,
        name: "openrouter",
        authority: "openrouter.ai:443",
        base_url: Some("https://openrouter.ai/api/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("OPENROUTER_API_KEY"),
        // Aggregator: no id prefix identifies it. Its ids are *other vendors'* (`anthropic/claude-…`,
        // `moonshotai/kimi-…`) — matching on those would steal every native route. Named explicitly.
        model_id_match: &[],
        host_match: &["openrouter.ai"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://docs.fireworks.ai/tools-sdks/openai-compatibility — base https://api.fireworks.ai/inference/v1, Bearer.
    // Gateway path: /fireworks/inference/v1/chat/completions.
    ProviderSpec {
        id: ProviderId::Fireworks,
        name: "fireworks",
        authority: "api.fireworks.ai:443",
        base_url: Some("https://api.fireworks.ai/inference/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("FIREWORKS_API_KEY"),
        // Fireworks ids ARE self-identifying by shape (`accounts/fireworks/models/…`), but that's a
        // full-id predicate, not a prefix this table can express — `agent_core::models::is_fireworks_model`
        // owns it, and `client.rs` falls back to it when no provider signal is present.
        model_id_match: &[],
        host_match: &["api.fireworks.ai"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://console.groq.com/docs/openai — base https://api.groq.com/openai/v1, Bearer.
    // Gateway path: /groq/openai/v1/chat/completions.
    ProviderSpec {
        id: ProviderId::Groq,
        name: "groq",
        authority: "api.groq.com:443",
        base_url: Some("https://api.groq.com/openai/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("GROQ_API_KEY"),
        model_id_match: &[],
        host_match: &["api.groq.com"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://api-docs.deepseek.com/ — base https://api.deepseek.com/v1 (the `/v1` is an OpenAI-compat
    // alias, not API versioning); /v1/chat/completions is officially supported. Bearer.
    ProviderSpec {
        id: ProviderId::DeepSeek,
        name: "deepseek",
        authority: "api.deepseek.com:443",
        base_url: Some("https://api.deepseek.com/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("DEEPSEEK_API_KEY"),
        model_id_match: &["deepseek-"],
        host_match: &["api.deepseek.com"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://docs.together.ai/docs/openai-api-compatibility — base https://api.together.ai/v1, Bearer.
    // Canonical host is `api.together.ai`; the legacy `api.together.xyz` is still live but undocumented.
    ProviderSpec {
        id: ProviderId::Together,
        name: "together",
        authority: "api.together.ai:443",
        base_url: Some("https://api.together.ai/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("TOGETHER_API_KEY"),
        model_id_match: &[],
        host_match: &["api.together.ai", "api.together.xyz"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://inference-docs.cerebras.ai/resources/openai — base https://api.cerebras.ai/v1, Bearer.
    ProviderSpec {
        id: ProviderId::Cerebras,
        name: "cerebras",
        authority: "api.cerebras.ai:443",
        base_url: Some("https://api.cerebras.ai/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("CEREBRAS_API_KEY"),
        model_id_match: &[],
        host_match: &["api.cerebras.ai"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://docs.mistral.ai/api/ — base https://api.mistral.ai/v1, Bearer.
    ProviderSpec {
        id: ProviderId::Mistral,
        name: "mistral",
        authority: "api.mistral.ai:443",
        base_url: Some("https://api.mistral.ai/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("MISTRAL_API_KEY"),
        // Mirrors `agent_core::models::is_mistral_model`'s own families.
        model_id_match: &[
            "mistral-",
            "magistral-",
            "devstral-",
            "codestral-",
            "ministral-",
        ],
        host_match: &["api.mistral.ai"],
        path_prefix: None,
        default_headers: &[],
    },
    // docs: https://docs.x.ai/docs/api-reference — base https://api.x.ai/v1, Bearer. Reasoning models are
    // slow: the generous read/idle timeouts (see gateway `config`) matter here.
    ProviderSpec {
        id: ProviderId::XAi,
        name: "xai",
        authority: "api.x.ai:443",
        base_url: Some("https://api.x.ai/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("XAI_API_KEY"),
        model_id_match: &["grok-"],
        host_match: &["api.x.ai"],
        path_prefix: None,
        default_headers: &[],
    },
    // OpenAI Codex (ChatGPT Plus/Pro subscription backend) — reached only via a stored OpenAI-Codex OAuth
    // credential's own bearer token, never a pool key and never an env var (there is no such thing as a
    // "Codex API key"). A genuinely static host, unlike GitHub Copilot's account-specific proxy endpoint —
    // see `agent_core::client::RouteOverride`'s doc comment for why Copilot bypasses the gateway entirely
    // rather than getting a row here. Gateway path: /openai-codex/backend-api/codex/responses.
    ProviderSpec {
        id: ProviderId::OpenAiCodex,
        name: "openai-codex",
        authority: "chatgpt.com:443",
        base_url: None,
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: None,
        model_id_match: &[],
        host_match: &["chatgpt.com"],
        path_prefix: None,
        default_headers: &[],
    },
    // --- BYO-only from here: no gateway route (no `/{name}/…` mount, no pool key). ---
    // docs: https://huggingface.co/docs/inference-providers — base https://router.huggingface.co/v1, Bearer.
    ProviderSpec {
        id: ProviderId::HuggingFace,
        name: "huggingface",
        authority: "router.huggingface.co:443",
        base_url: Some("https://router.huggingface.co/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("HF_TOKEN"),
        model_id_match: &[],
        host_match: &["router.huggingface.co"],
        path_prefix: None,
        default_headers: &[],
    },
    // NVIDIA NIM — base https://integrate.api.nvidia.com/v1, Bearer. `NVCF-POLL-SECONDS` is a hard
    // requirement, not a tuning knob: NIM's cloud functions 202-poll, and without it a long generation is
    // cut short. (Was `agent::settings::default_headers_for_base_url`.)
    ProviderSpec {
        id: ProviderId::Nvidia,
        name: "nvidia",
        authority: "integrate.api.nvidia.com:443",
        base_url: Some("https://integrate.api.nvidia.com/v1"),
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: Some("NVIDIA_API_KEY"),
        model_id_match: &[],
        host_match: &["integrate.api.nvidia.com"],
        path_prefix: None,
        default_headers: &[("NVCF-POLL-SECONDS", "3600")],
    },
    // Kimi-Coding (`api.kimi.com`) — Anthropic-wire, and the `User-Agent` is an upstream allowlist check,
    // not cosmetics. `base_url`/`env_var` are `None`: the coding endpoint is account-path-templated, so it
    // is reached by an explicit `models.json` `base_url` rather than a static row.
    ProviderSpec {
        id: ProviderId::KimiCoding,
        name: "kimi-coding",
        authority: "api.kimi.com:443",
        base_url: None,
        wire: WireFormat::Anthropic,
        auth: AuthScheme::Bearer,
        env_var: None,
        model_id_match: &[],
        host_match: &["api.kimi.com"],
        path_prefix: None,
        default_headers: &[("User-Agent", "KimiCLI/1.5")],
    },
    // OpenCode Zen / Go — same registered domain, two providers. `/zen/go` is listed FIRST so the longest
    // path prefix wins in `for_host` regardless of iteration order (see `path_matches`).
    ProviderSpec {
        id: ProviderId::OpenCodeGo,
        name: "opencode-go",
        authority: "opencode.ai:443",
        base_url: None,
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: None,
        model_id_match: &[],
        host_match: &["opencode.ai"],
        path_prefix: Some("/zen/go"),
        default_headers: &[],
    },
    ProviderSpec {
        id: ProviderId::OpenCodeZen,
        name: "opencode-zen",
        authority: "opencode.ai:443",
        base_url: None,
        wire: WireFormat::OpenAi,
        auth: AuthScheme::Bearer,
        env_var: None,
        model_id_match: &[],
        host_match: &["opencode.ai"],
        path_prefix: Some("/zen"),
        default_headers: &[],
    },
];

impl ProviderSpec {
    /// The path component of [`Self::base_url`] — the mount prefix this provider hangs its API under.
    /// `""` when there is no base URL, and for Anthropic, whose base is deliberately bare.
    ///
    /// The gateway's `/{provider}/…` route never needs this: the client names the provider, so it
    /// also spells the mount, and the path is forwarded verbatim. The **model-routed** (`/auto`)
    /// route does, because there the client cannot know which provider will serve it — and the
    /// providers do not agree (`/v1` for OpenAI, `/openai/v1` for Groq, `/inference/v1` for
    /// Fireworks, `/api/v1` for OpenRouter). So that route strips its own segment, keeps the
    /// remainder, and prepends whichever mount the chosen candidate uses.
    ///
    /// Derived rather than stored so it cannot drift from [`Self::base_url`], which is the field
    /// anyone updating a provider actually edits.
    pub fn base_path(&self) -> &'static str {
        let Some(url) = self.base_url else {
            return "";
        };
        let after_scheme = match url.find("://") {
            Some(i) => &url[i + 3..],
            None => url,
        };
        match after_scheme.find('/') {
            Some(i) => &after_scheme[i..],
            None => "",
        }
    }
}

/// The gateway rows — those with a `/{name}/…` route and a pool key. The BYO-only rows have no
/// upstream the gateway would ever dial on a virtual key's behalf, so including them would mint dead
/// providers (and dead `AI_POOL_KEY_*` env vars) at boot.
pub fn gateway_providers() -> impl Iterator<Item = &'static ProviderSpec> {
    PROVIDERS.iter().filter(|p| {
        !matches!(
            p.id,
            ProviderId::HuggingFace
                | ProviderId::Nvidia
                | ProviderId::KimiCoding
                | ProviderId::OpenCodeZen
                | ProviderId::OpenCodeGo
        )
    })
}

/// Look a provider up by its [`ProviderSpec::name`] — the gateway's path segment, and `AI_PROVIDER`'s
/// value. Case-insensitive, since it's user input on the `AI_PROVIDER` side.
pub fn by_name(name: &str) -> Option<&'static ProviderSpec> {
    PROVIDERS.iter().find(|p| p.name.eq_ignore_ascii_case(name))
}

/// The row for an id. Infallible: every [`ProviderId`] has exactly one row (asserted in tests).
pub fn by_id(id: ProviderId) -> &'static ProviderSpec {
    match PROVIDERS.iter().find(|p| p.id == id) {
        Some(spec) => spec,
        // Unreachable: `every_id_has_exactly_one_row` proves the table is total over `ProviderId`.
        // A `panic!` here would violate the workspace's `clippy::panic = deny`, and there is no
        // sensible `Option` for callers that already hold a valid id — so fall back to the first row,
        // which is only reachable if someone adds a variant and forgets its row *and* ignores CI.
        None => &PROVIDERS[0],
    }
}

/// Whether `path` sits under `prefix` — boundary-checked, so `/zen` does not match `/zenith`, and
/// `/zen` matches both `/zen` itself and `/zen/anything`. A raw `starts_with` would let `/zen` claim
/// `/zen/go`'s requests, which is precisely the two-providers-one-hostname collision this guards.
fn path_matches(prefix: &str, path: &str) -> bool {
    match path.trim_end_matches('/').strip_prefix(prefix) {
        Some(rest) => rest.is_empty() || rest.starts_with('/'),
        None => false,
    }
}

/// Which provider a user-supplied base URL points at — the `models.json` `base_url` path, where the
/// provider is known only by where it points.
///
/// Takes the URL **already split** into a lowercased host and a path, rather than a `&str` URL, so this
/// crate stays dependency-free (see the manifest). Callers with a URL parse it themselves:
/// `agent::gateway_credential::provider_for_base_url`.
///
/// Where two rows share a hostname (`opencode.ai`), the one with the **longest matching**
/// [`ProviderSpec::path_prefix`] wins; a row with a `path_prefix` that doesn't match is not a
/// candidate at all. A bare `https://opencode.ai/` therefore resolves to `None` — correct, since which
/// of the two providers is meant is genuinely unknown.
pub fn for_host(host: &str, path: &str) -> Option<&'static ProviderSpec> {
    let mut best: Option<&'static ProviderSpec> = None;
    for spec in PROVIDERS {
        if !spec.host_match.iter().any(|h| h.eq_ignore_ascii_case(host)) {
            continue;
        }
        match spec.path_prefix {
            Some(prefix) if !path_matches(prefix, path) => continue,
            Some(prefix) => {
                let better = best
                    .and_then(|b| b.path_prefix)
                    .is_none_or(|cur| prefix.len() > cur.len());
                if better {
                    best = Some(spec);
                }
            }
            // A hostname-only row can't be beaten by another hostname-only row (host_match values are
            // unique across rows — asserted in tests), so first match wins.
            None => {
                if best.is_none() {
                    best = Some(spec);
                }
            }
        }
    }
    best
}

/// Which provider natively serves this model id — the zero-config direct route. `claude-opus-4-8` ⇒
/// Anthropic, `gpt-5` ⇒ OpenAI, `grok-4` ⇒ xAI.
///
/// **Aggregators never match here** ([`ProviderSpec::model_id_match`] explains why at length): they
/// serve other vendors' ids, so any prefix that identified them would steal the native route. Reaching
/// an aggregator is an explicit act (`AI_PROVIDER=openrouter`), never an inference.
///
/// `None` means "no native provider recognizes this id" — the caller then wants a clear error naming
/// what to set, not a guess.
pub fn for_model_id(model: &str) -> Option<&'static ProviderSpec> {
    let m = model.to_ascii_lowercase();
    PROVIDERS
        .iter()
        .find(|p| p.model_id_match.iter().any(|pre| m.starts_with(pre)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table must be total over the enum: `by_id` returns a row for every variant, and no variant
    /// has two. This is what lets `by_id` be infallible.
    #[test]
    fn every_id_has_exactly_one_row() {
        assert_eq!(
            ProviderId::ALL.len(),
            PROVIDERS.len(),
            "a row exists with no ProviderId"
        );
        for id in ProviderId::ALL {
            let rows = PROVIDERS.iter().filter(|p| p.id == id).count();
            assert_eq!(rows, 1, "{id:?} has {rows} rows, want exactly 1");
            assert_eq!(by_id(id).id, id);
        }
    }

    /// `index` is used to key a fixed-size array, so it must be a bijection onto `0..COUNT`.
    #[test]
    fn provider_id_indices_are_dense_and_unique() {
        let mut seen = [false; ProviderId::COUNT];
        for (want, id) in ProviderId::ALL.into_iter().enumerate() {
            let got = id.index();
            assert_eq!(got, want, "{id:?} must sit at its position in ALL");
            assert!(!seen[got], "{id:?} reuses index {got}");
            seen[got] = true;
        }
        assert!(seen.into_iter().all(|s| s), "an index is unreachable");
    }

    /// The mount prefixes the model-routed gateway path prepends. Pinned per row because getting one
    /// wrong sends a request to a 404 on a provider that is up.
    #[test]
    fn base_path_is_the_base_urls_path_component() {
        let cases = [
            (ProviderId::OpenAi, "/v1"),
            // Deliberately bare: `base_url` carries no `/v1`, the endpoint path supplies it.
            (ProviderId::Anthropic, ""),
            (ProviderId::OpenRouter, "/api/v1"),
            (ProviderId::Fireworks, "/inference/v1"),
            (ProviderId::Groq, "/openai/v1"),
            (ProviderId::DeepSeek, "/v1"),
            (ProviderId::Together, "/v1"),
            (ProviderId::Cerebras, "/v1"),
            (ProviderId::Mistral, "/v1"),
            (ProviderId::XAi, "/v1"),
            // No direct-route base URL at all.
            (ProviderId::OpenAiCodex, ""),
        ];
        for (id, want) in cases {
            assert_eq!(
                by_id(id).base_path(),
                want,
                "{id:?} base_path from base_url {:?}",
                by_id(id).base_url,
            );
        }
    }

    /// `base_path` must be derivable from `base_url` for *every* row, not just the pinned ones —
    /// a new provider gets the same treatment without anyone remembering to extend the case list.
    #[test]
    fn base_path_agrees_with_base_url_for_every_row() {
        for p in PROVIDERS {
            match p.base_url {
                None => assert_eq!(p.base_path(), "", "{}: no base_url ⇒ no mount", p.name),
                Some(url) => {
                    let path = p.base_path();
                    assert!(
                        path.is_empty() || path.starts_with('/'),
                        "{}: mount {path:?} must be empty or absolute",
                        p.name,
                    );
                    assert!(
                        url.ends_with(path),
                        "{}: base_url {url:?} must end with its mount {path:?}",
                        p.name,
                    );
                    assert!(
                        !path.contains("://"),
                        "{}: mount {path:?} still carries a scheme",
                        p.name,
                    );
                }
            }
        }
    }

    #[test]
    fn provider_names_are_unique() {
        let mut names: Vec<_> = PROVIDERS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate provider name in PROVIDERS");
    }

    /// Wire drives usage parsing on the gateway and body construction in the agent. Getting Anthropic's
    /// wrong mis-meters every request through it; getting an OpenAI-wire provider's wrong POSTs an
    /// Anthropic body to a Chat Completions endpoint.
    #[test]
    fn only_anthropic_and_kimi_coding_speak_anthropic_wire() {
        for spec in PROVIDERS {
            let want = match spec.id {
                ProviderId::Anthropic | ProviderId::KimiCoding => WireFormat::Anthropic,
                _ => WireFormat::OpenAi,
            };
            assert_eq!(spec.wire, want, "{} wire", spec.name);
        }
    }

    /// The direct route is `base_url` + a key from `env_var`. A row with one and not the other is a
    /// half-built route: an `env_var` with no `base_url` reads a key it has nowhere to send, and a
    /// `base_url` with no `env_var` can never be selected by [`for_model_id`].
    #[test]
    fn direct_route_fields_come_in_pairs() {
        for spec in PROVIDERS {
            assert_eq!(
                spec.base_url.is_some(),
                spec.env_var.is_some(),
                "{}: base_url and env_var must both be set or both be None",
                spec.name
            );
        }
    }

    /// A row with no `base_url` has no direct route, so a native id prefix pointing at it would resolve
    /// to a provider that can't be dialed.
    #[test]
    fn model_id_match_implies_a_direct_route() {
        for spec in PROVIDERS {
            if !spec.model_id_match.is_empty() {
                assert!(
                    spec.base_url.is_some(),
                    "{} claims model ids but has no base_url",
                    spec.name
                );
            }
        }
    }

    /// Two rows claiming the same id prefix would make [`for_model_id`] order-dependent — a silent
    /// mis-route that changes when someone reorders the table.
    #[test]
    fn model_id_prefixes_are_disjoint_across_rows() {
        for a in PROVIDERS {
            for b in PROVIDERS {
                if a.id == b.id {
                    continue;
                }
                for pa in a.model_id_match {
                    for pb in b.model_id_match {
                        assert!(
                            !pa.starts_with(pb) && !pb.starts_with(pa),
                            "{} claims {pa:?} but {} claims the overlapping {pb:?}",
                            a.name,
                            b.name
                        );
                    }
                }
            }
        }
    }

    /// Two rows sharing a hostname must disambiguate by path (`for_host`'s longest-prefix rule), or the
    /// lookup is a coin flip.
    #[test]
    fn shared_hostnames_are_path_disambiguated() {
        for a in PROVIDERS {
            for b in PROVIDERS {
                if a.id == b.id {
                    continue;
                }
                let shares_host = a
                    .host_match
                    .iter()
                    .any(|h| b.host_match.iter().any(|o| o.eq_ignore_ascii_case(h)));
                if shares_host {
                    assert!(
                        a.path_prefix.is_some() && b.path_prefix.is_some(),
                        "{} and {} share a hostname without a path_prefix to tell them apart",
                        a.name,
                        b.name
                    );
                    assert_ne!(a.path_prefix, b.path_prefix, "{} vs {}", a.name, b.name);
                }
            }
        }
    }

    #[test]
    fn for_model_id_resolves_native_ids() {
        assert_eq!(
            for_model_id("claude-opus-4-8").map(|p| p.id),
            Some(ProviderId::Anthropic)
        );
        assert_eq!(
            for_model_id("gpt-5").map(|p| p.id),
            Some(ProviderId::OpenAi)
        );
        assert_eq!(for_model_id("o3").map(|p| p.id), Some(ProviderId::OpenAi));
        assert_eq!(for_model_id("grok-4").map(|p| p.id), Some(ProviderId::XAi));
        assert_eq!(
            for_model_id("deepseek-v3").map(|p| p.id),
            Some(ProviderId::DeepSeek)
        );
        assert_eq!(
            for_model_id("CLAUDE-OPUS-4-8").map(|p| p.id),
            Some(ProviderId::Anthropic),
            "model ids are matched case-insensitively"
        );
    }

    /// The heart of it: a vendor-slug id belongs to an *aggregator*, and no aggregator claims prefixes.
    /// If `anthropic/claude-sonnet-4.5` resolved to the Anthropic row we would POST an Anthropic body to
    /// api.anthropic.com with an OpenRouter key — which is the bug this whole design prevents.
    #[test]
    fn vendor_slug_ids_resolve_to_no_native_provider() {
        assert_eq!(for_model_id("anthropic/claude-sonnet-4.5"), None);
        assert_eq!(for_model_id("openai/gpt-5"), None);
        assert_eq!(for_model_id("moonshotai/kimi-k2.6"), None);
        assert_eq!(for_model_id("meta-llama/llama-4"), None);
    }

    #[test]
    fn for_model_id_is_none_for_unknown_ids() {
        assert_eq!(for_model_id("kimi-k2-thinking"), None);
        assert_eq!(for_model_id("glm-5.1"), None);
        assert_eq!(for_model_id(""), None);
    }

    #[test]
    fn for_host_resolves_known_hosts() {
        assert_eq!(
            for_host("openrouter.ai", "/api/v1").map(|p| p.id),
            Some(ProviderId::OpenRouter)
        );
        assert_eq!(
            for_host("api.together.xyz", "/v1").map(|p| p.id),
            Some(ProviderId::Together),
            "the legacy Together host is still live"
        );
        assert_eq!(
            for_host("api.anthropic.com", "/").map(|p| p.id),
            Some(ProviderId::Anthropic)
        );
        assert_eq!(for_host("example.com", "/v1"), None);
    }

    /// `opencode.ai` hosts two providers that disagree on wire and on real numbers for shared bare ids.
    /// The `/zen/go` row must win over `/zen` on its own paths despite both hostnames matching.
    #[test]
    fn for_host_splits_opencode_on_the_path() {
        assert_eq!(
            for_host("opencode.ai", "/zen/go").map(|p| p.id),
            Some(ProviderId::OpenCodeGo)
        );
        assert_eq!(
            for_host("opencode.ai", "/zen/go/v1").map(|p| p.id),
            Some(ProviderId::OpenCodeGo)
        );
        assert_eq!(
            for_host("opencode.ai", "/zen/go/").map(|p| p.id),
            Some(ProviderId::OpenCodeGo)
        );
        assert_eq!(
            for_host("opencode.ai", "/zen").map(|p| p.id),
            Some(ProviderId::OpenCodeZen)
        );
        assert_eq!(
            for_host("opencode.ai", "/zen/v1").map(|p| p.id),
            Some(ProviderId::OpenCodeZen)
        );
        // Neither: which of the two is meant is genuinely unknown, so say so rather than guess.
        assert_eq!(for_host("opencode.ai", "/"), None);
        assert_eq!(for_host("opencode.ai", "/other"), None);
    }

    #[test]
    fn path_matches_is_boundary_checked() {
        assert!(path_matches("/zen", "/zen"));
        assert!(path_matches("/zen", "/zen/"));
        assert!(path_matches("/zen", "/zen/v1"));
        assert!(!path_matches("/zen", "/zenith"));
        assert!(!path_matches("/zen", "/"));
        assert!(!path_matches("/zen/go", "/zen"));
    }

    #[test]
    fn by_name_is_case_insensitive() {
        assert_eq!(
            by_name("openrouter").map(|p| p.id),
            Some(ProviderId::OpenRouter)
        );
        assert_eq!(
            by_name("OpenRouter").map(|p| p.id),
            Some(ProviderId::OpenRouter)
        );
        assert_eq!(
            by_name("openai-codex").map(|p| p.id),
            Some(ProviderId::OpenAiCodex)
        );
        assert_eq!(by_name("nope"), None);
    }

    /// The gateway mints a `Provider` (and expects an `AI_POOL_KEY_*`) per row it sees. BYO-only rows
    /// have no gateway route at all, so they must not appear.
    #[test]
    fn gateway_providers_excludes_byo_only_rows() {
        let names: Vec<_> = gateway_providers().map(|p| p.name).collect();
        assert_eq!(names.len(), 11);
        assert!(names.contains(&"openai-codex"));
        assert!(!names.contains(&"huggingface"));
        assert!(!names.contains(&"opencode-zen"));
        assert!(!names.contains(&"kimi-coding"));
    }

    #[test]
    fn auth_scheme_formats_and_headers() {
        assert_eq!(AuthScheme::Bearer.header(), "authorization");
        assert_eq!(AuthScheme::Bearer.format("k"), "Bearer k");
        assert_eq!(AuthScheme::Bearer.value_prefix(), Some("Bearer "));
        // Anthropic wants the bare key (no `Bearer`). Getting this wrong → upstream 401.
        assert_eq!(AuthScheme::XApiKey.header(), "x-api-key");
        assert_eq!(AuthScheme::XApiKey.format("k"), "k");
        assert_eq!(AuthScheme::XApiKey.value_prefix(), None);
        // Azure OpenAI: bare key (no `Bearer`) in its own `api-key` header.
        assert_eq!(AuthScheme::ApiKey.header(), "api-key");
        assert_eq!(AuthScheme::ApiKey.format("k"), "k");
        assert_eq!(AuthScheme::ApiKey.value_prefix(), None);
        // Cloudflare AI Gateway: a `Bearer`-prefixed value in a differently-named header.
        assert_eq!(
            AuthScheme::CustomHeader("cf-aig-authorization").header(),
            "cf-aig-authorization"
        );
        assert_eq!(
            AuthScheme::CustomHeader("cf-aig-authorization").format("k"),
            "Bearer k"
        );
    }

    /// `value_prefix` and `format` are two views of the same fact and must not drift: the agent's direct
    /// route composes the header value from the prefix, the gateway formats it whole.
    #[test]
    fn value_prefix_agrees_with_format() {
        for scheme in [
            AuthScheme::Bearer,
            AuthScheme::XApiKey,
            AuthScheme::ApiKey,
            AuthScheme::CustomHeader("cf-aig-authorization"),
        ] {
            let composed = format!("{}{}", scheme.value_prefix().unwrap_or(""), "k");
            assert_eq!(composed, scheme.format("k"), "{scheme:?}");
        }
    }

    #[test]
    fn wire_parse_config_accepts_known_case_insensitive() {
        assert_eq!(WireFormat::parse_config("openai"), Some(WireFormat::OpenAi));
        assert_eq!(WireFormat::parse_config("OpenAI"), Some(WireFormat::OpenAi));
        assert_eq!(
            WireFormat::parse_config("anthropic"),
            Some(WireFormat::Anthropic)
        );
        assert_eq!(WireFormat::parse_config("bogus"), None);
        assert_eq!(WireFormat::parse_config(""), None);
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
        assert_eq!(
            AuthScheme::parse_config("api-key"),
            Some(AuthScheme::ApiKey)
        );
        assert_eq!(
            AuthScheme::parse_config("API-KEY"),
            Some(AuthScheme::ApiKey)
        );
        assert_eq!(AuthScheme::parse_config("apikey"), Some(AuthScheme::ApiKey));
        assert_eq!(AuthScheme::parse_config("bogus"), None);
        // Must NOT collide with x-api-key's own normalized spelling.
        assert_ne!(
            AuthScheme::parse_config("api-key"),
            AuthScheme::parse_config("x-api-key")
        );
    }

    /// Env var names are the provider's own convention, not ours — a typo here is a key that silently
    /// never gets picked up. Pinned so a rename is a deliberate act.
    #[test]
    fn env_vars_match_the_providers_own_convention() {
        let want = [
            (ProviderId::Anthropic, "ANTHROPIC_API_KEY"),
            (ProviderId::OpenAi, "OPENAI_API_KEY"),
            (ProviderId::OpenRouter, "OPENROUTER_API_KEY"),
            (ProviderId::Groq, "GROQ_API_KEY"),
            (ProviderId::Fireworks, "FIREWORKS_API_KEY"),
            (ProviderId::DeepSeek, "DEEPSEEK_API_KEY"),
            (ProviderId::Together, "TOGETHER_API_KEY"),
            (ProviderId::Cerebras, "CEREBRAS_API_KEY"),
            (ProviderId::Mistral, "MISTRAL_API_KEY"),
            (ProviderId::XAi, "XAI_API_KEY"),
        ];
        for (id, var) in want {
            assert_eq!(by_id(id).env_var, Some(var), "{id:?}");
        }
    }

    /// Anthropic's base URL is bare on purpose — `Dialect::endpoint_path` appends `/v1/messages` to it,
    /// while the OpenAI-wire rows carry their own `/v1` and get a bare endpoint path appended. Adding a
    /// `/v1` here would produce `https://api.anthropic.com/v1/v1/messages`.
    #[test]
    fn anthropic_base_url_carries_no_v1() {
        assert_eq!(
            by_id(ProviderId::Anthropic).base_url,
            Some("https://api.anthropic.com")
        );
        for spec in PROVIDERS {
            if spec.wire == WireFormat::OpenAi
                && let Some(base) = spec.base_url
            {
                assert!(
                    base.contains("/v1"),
                    "{}: an OpenAI-wire base_url needs its own /v1 base path",
                    spec.name
                );
            }
        }
    }

    /// A base URL must resolve back to the row it came from, or the two lookups disagree and a direct
    /// route gets a different provider's dialect than the one that built it.
    #[test]
    fn base_urls_round_trip_through_for_host() {
        for spec in PROVIDERS {
            let Some(base) = spec.base_url else { continue };
            let rest = base.trim_start_matches("https://");
            let (host, path) = match rest.find('/') {
                Some(i) => (&rest[..i], &rest[i..]),
                None => (rest, "/"),
            };
            assert_eq!(
                for_host(host, path).map(|p| p.id),
                Some(spec.id),
                "{}: base_url {base} does not resolve back to its own row",
                spec.name
            );
        }
    }
}
