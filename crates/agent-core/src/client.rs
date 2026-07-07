//! The default model transport: an HTTP client that speaks provider wire to the Beyond gateway.
//!
//! This is the harness's whole network surface. It never holds a provider key or picks a provider —
//! it sends `Authorization: Bearer <bai_v1…>` to the gateway, which swaps in the pool key, routes to
//! the real provider, and meters usage. The client only picks the *dialect* (by model id), builds
//! the request body, and frames the streaming SSE response back into [`StreamEvent`]s.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde_json::Value;
use zeroize::Zeroize;

use crate::agent::catch_tool_panic;
use crate::dialect::{Dialect, SseEventBuffer, push_sse_line};
use crate::error::{Error, MID_STREAM_NETWORK_ERROR, Result};
use crate::hooks::AgentHooks;
use crate::transport::{EventStream, ModelRequest, ModelTransport};

/// Anthropic's Messages API requires this header; the gateway relays it to the upstream verbatim.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Beta opt-in sent with thinking requests: lets the model interleave thinking between tool calls
/// across a turn (and keeps fine-grained streaming of the thinking blocks).
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
/// Fallback streaming beta for a model whose tool definitions *don't* carry the per-tool
/// `eager_input_streaming: true` marker (see `dialect::anthropic::mark_eager_tool_streaming`) — the two
/// are mutually exclusive; no current model needs this branch, since every current id supports the
/// per-tool marker instead, but it exists for correctness if that ever changes.
const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

/// Anthropic's OAuth (Claude Pro/Max subscription) beta opt-ins — sent only when the attached
/// credential came from an OAuth login (`is_oauth`), never for a plain API key. Anthropic gates this
/// subscription-authenticated endpoint to its own official Claude Code client; these two beta tokens
/// plus [`CLAUDE_CLI_IDENTITY`]/the `x-app`/`anthropic-dangerous-direct-browser-access` headers below
/// are what tell it this request is that client. Presenting this tool as Claude Code is a deliberate,
/// user-confirmed choice (see `crates/agent/src/oauth/anthropic.rs`), not an oversight — full pi
/// parity rather than the "false attribution" this crate previously declined to send.
const CLAUDE_CODE_BETA: &str = "claude-code-20250219";
const OAUTH_BETA: &str = "oauth-2025-04-20";
/// The identity string Anthropic's OAuth gating checks for (`user-agent`). **A moving target, not a
/// one-time constant**: if Anthropic starts rejecting a stale `claude-cli/<version>`, bump this to
/// whatever the real Claude Code CLI currently reports.
const CLAUDE_CLI_IDENTITY: &str = "claude-cli/2.1.75";

/// How many times to re-issue a request that failed transiently (connection refused, timeout, or a
/// retryable status) before giving up. A multi-step agent run re-issues a request every turn, so a
/// single transient gateway hiccup would otherwise vaporize the whole run; the gateway itself is
/// behind a load balancer that sheds load with 429/503 under pressure. Public so a caller overriding
/// only one of the two [`GatewayClient::with_retry`] parameters can default the other to this.
pub const MAX_RETRIES: u32 = 3;
/// Base of the exponential backoff between retries (`BASE · 2^(attempt-1)`). Public — see
/// [`MAX_RETRIES`].
pub const BASE_BACKOFF: Duration = Duration::from_millis(250);
/// Default ceiling on a single backoff wait (exponential or `Retry-After`-derived), so a
/// server-supplied hint can't park a run for minutes. Raised from an earlier 10s toward pi's own
/// default of 60s (`openai-codex-responses.ts`'s `DEFAULT_MAX_RETRY_DELAY_MS`, itself overridable) —
/// at 10s, a 429 with a `Retry-After: 30` hint got retried back into the very rate-limit window it
/// named, capable of exhausting the whole retry budget before that window actually closed. `pub` so a
/// caller overriding only [`GatewayClient::with_retry`]'s two knobs can still reference this default
/// for the third; override the ceiling itself with [`GatewayClient::with_max_backoff`].
pub const MAX_BACKOFF: Duration = Duration::from_secs(60);

/// Cap on the TCP+TLS handshake to the gateway. Mirrors the gateway's own upstream
/// `connect_timeout_secs` (10s): a connection it can't establish in this window is a dead gateway,
/// not a slow one.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Idle timeout *between* reads on the streaming body — **not** a ceiling on total stream duration
/// (an overall `Client::timeout` would wrongly kill long-but-healthy streams). It catches the one
/// failure this layer can't otherwise detect: a gateway holding the connection open but sending no
/// bytes.
///
/// Set to match the gateway's upstream `read_timeout_secs` (600s). We sit *downstream* of the
/// gateway, which applies that same 600s idle timeout to the provider connection — so the gateway
/// can legitimately send us nothing for up to 600s while it waits on the provider (a long
/// extended-thinking gap; gateway TTFT tails to 30s and full-request latency to 600s in its own
/// metrics). A tighter timeout here would sever a stream the gateway still considers healthy: a
/// downstream hop's patience must be at least its upstream's.
const READ_TIMEOUT: Duration = Duration::from_secs(600);

/// A gateway credential that never appears in a `Debug` line and is best-effort scrubbed from memory
/// on drop. A small local newtype rather than depending on `beyond-gateway`'s own `Secret` — this is
/// the wrong dependency direction (agent-core is a lower-level library the gateway sits above) — so
/// this intentionally covers only the one field that needs it, not a general secrets framework.
#[derive(Clone)]
struct ApiKey(String);

impl ApiKey {
    fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the plaintext for the one call site that needs it (the `Authorization` header).
    fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(***)")
    }
}

impl Drop for ApiKey {
    fn drop(&mut self) {
        // Best-effort scrub, not a hard control: the key is necessarily long-lived in RAM for the
        // life of the client, and this only helps at client-drop/rotation time. The workspace forbids
        // `unsafe_code`, ruling out a hand-rolled `as_bytes_mut` overwrite — `zeroize` gives the same
        // effect through a safe API (same crate the gateway's own `Secret` type already uses).
        self.0.zeroize();
    }
}

/// How a [`DirectRouting`]-carrying credential's request must reach its upstream when the default
/// `{GatewayClient::base_url}{dialect.endpoint_path()}` composition is wrong for this provider.
///
/// Both GitHub Copilot and OpenAI Codex OAuth logins hand back a bearer token that's useless against
/// the gateway's *default* provider (bare `/v1/…` always means "whichever provider the gateway's
/// dialect default points at" — Anthropic or OpenAI direct, never Copilot/Codex): the gateway's
/// `route.rs` `KNOWN_PROVIDERS` table is a set of **static** `(name, authority, dialect, auth)` rows,
/// which fits Codex fine (`chatgpt.com` is a fixed host) but cannot express Copilot's endpoint at
/// all — GitHub hands back a *different* proxy host per account/enterprise, embedded in the access
/// token itself (`proxy-ep=…`), not knowable at gateway-boot time. So the two providers need two
/// different fixes, both applied client-side rather than in the gateway's routing table:
#[derive(Debug, Clone)]
pub enum RouteOverride {
    /// Still relayed through the gateway (`GatewayClient::base_url` unchanged) under a
    /// `route::KNOWN_PROVIDERS` row's name as the first path segment (e.g. `"/openai-codex"`), with
    /// `path` appended in place of the dialect's own default endpoint path — Codex's real path
    /// (`/backend-api/codex/responses`) differs from the OpenAI-Responses dialect's usual
    /// `/v1/responses`. Fits because Codex's host (`chatgpt.com`) is genuinely static.
    Prefixed {
        prefix: &'static str,
        path: &'static str,
    },
    /// Bypasses the gateway entirely: `base_url` replaces `GatewayClient::base_url` outright — the
    /// account-specific host baked into this credential's own token — with `path` in place of the
    /// dialect's own default endpoint path (GitHub Copilot's OpenAI-wire endpoints omit the `/v1`
    /// prefix the dialect's default path carries, e.g. `/chat/completions` not
    /// `/v1/chat/completions` — only its Anthropic-wire endpoint matches the dialect default
    /// verbatim). This is the same "forwarded as-is, no gateway involvement" shape the gateway's own
    /// BYO-key concept already describes, just skipping the gateway hop too since there's no static
    /// row that could route it there in the first place.
    Direct {
        base_url: String,
        path: &'static str,
    },
}

/// Extra per-request wiring a [`RouteOverride`]-carrying credential needs beyond the bearer token
/// itself and the URL: static headers the provider requires (Codex's `chatgpt-account-id`; Copilot's
/// fixed editor-identity headers), and/or GitHub Copilot's per-turn dynamic headers (see
/// `copilot_initiator`/`copilot_has_images`).
#[derive(Debug, Clone)]
pub struct DirectRouting {
    pub route: RouteOverride,
    pub static_headers: Vec<(&'static str, String)>,
    /// Attach GitHub Copilot's `X-Initiator`/`Openai-Intent`/`Copilot-Vision-Request` headers,
    /// computed fresh per turn from this request's own messages — `false` for every other provider.
    pub copilot_dynamic_headers: bool,
    /// Send this credential's key in a named header, verbatim (no `Bearer` prefix), instead of the
    /// usual `Authorization: Bearer <key>` — and omit `Authorization` entirely. `None` (every
    /// existing route: Codex, Copilot, a self-hosted `ModelOverride`) preserves the
    /// Authorization-Bearer behavior every provider up to now has needed. `Some("api-key")` is
    /// Azure OpenAI's real wire (Task #8, pi-parity): its `AzureOpenAI` SDK client authenticates via
    /// a bare key in `api-key`, never `Authorization` (see
    /// `packages/ai/src/api/azure-openai-responses.ts` in pi-mono) — sending both would risk Azure
    /// attempting to validate a well-formed-looking but bogus `Authorization` value as an AAD token.
    pub auth_header: Option<String>,
    /// Prepended to the credential's value when sent through [`auth_header`](Self::auth_header) — pi-parity
    /// Fix 4, Round 2: Cloudflare AI Gateway wants `cf-aig-authorization: Bearer <key>`, a *named*
    /// header (so `auth_header` alone, which already covers that) carrying a Bearer-*prefixed* value
    /// (which bare `auth_header` doesn't — it sends the credential verbatim, correct for Azure's
    /// `api-key` scheme but wrong here). `None` (every existing route, including Azure's) preserves
    /// the bare-value behavior `auth_header` already had. Has no effect without `auth_header` also
    /// set — there's no header to prefix a value into otherwise.
    pub auth_header_prefix: Option<String>,
    /// Override the wire dialect (Anthropic/OpenAI Chat Completions/OpenAI Responses) this request
    /// builds and decodes as, instead of inferring it from the model id via
    /// [`Dialect::for_model_via_copilot`] — pi-parity Fix 1, Round 2: a `models.json` `base_url`
    /// override can point at a genuinely Anthropic-wire (or OpenAI-wire) third-party provider whose
    /// model ids don't match [`Dialect::for_model`]'s name heuristic (Kimi-Coding's
    /// `kimi-k2-thinking`, e.g. — no "claude"/"anthropic" substring, so the heuristic would build an
    /// OpenAI-shaped body and POST it to an Anthropic Messages endpoint). `None` (every existing
    /// route) preserves the heuristic. Consulted at the same single call site that already picks the
    /// dialect for every route (gateway-relayed, `Prefixed`, and `Direct` alike) in
    /// [`GatewayClient::stream`], so the override applies consistently to body-building, decoding,
    /// and the Anthropic-specific header/beta logic there — not just the URL.
    pub dialect_override: Option<Dialect>,
    /// Send this instead of the app-level model id (`ModelRequest::model`) as the request body's
    /// wire-level `"model"` field — pi-parity Fix 2, Round 2: Azure OpenAI's deployment name doesn't
    /// have to match the model id used for capability lookups (`AZURE_OPENAI_DEPLOYMENT_NAME_MAP` in
    /// pi's own `azure-openai-responses.ts`). `None` (every existing route) leaves the body's
    /// `"model"` field as `dialect.build_body` already set it. Deliberately *not* a rewrite of
    /// `ModelRequest::model` itself — that field also drives `crate::models::capabilities` lookups
    /// (context window, thinking shape, …), which must stay keyed on the app-level id the operator
    /// actually configured capabilities for, not the wire-level deployment name.
    pub deployment_name: Option<String>,
    /// Extra query string (already escaped, no leading `?`) appended to the built URL — pi-parity Fix
    /// 2, Round 2: an Azure resource pinned to a dated `api-version` (true of most real enterprise
    /// Azure OpenAI deployments) needs `?api-version=2024-08-01-preview` on every request; beyond
    /// previously never sent one at all. Only consulted for [`RouteOverride::Direct`] — a
    /// `Prefixed`-routed request (Codex) has no analogous need and ignores this field. `None`/empty
    /// (every existing route) builds the URL with no query string, unchanged.
    pub query: Option<String>,
    /// Which third-party aggregator platform this BYO override's `base_url` names, if any (pi-parity
    /// pass 20, Task 5) — mirrors how [`deployment_name`](Self::deployment_name) threads Azure's
    /// deployment mapping across the `crates/agent`/`agent-core` crate boundary. Populated by
    /// `crates/agent::gateway_credential`'s `aggregator_host_for_base_url` (which already computes this
    /// from the override's `base_url` for `GatewayCredentialIdentity`'s sake) whenever a `DirectRouting`
    /// is built from a `models.json` `base_url` override; `None` for every route with no BYO override at
    /// all (Codex, GitHub Copilot) — those have no aggregator host of their own to report. Read by
    /// [`GatewayClient::stream`] to set [`crate::transport::ModelRequest::host`], alongside (not instead
    /// of) Fireworks' own id-shape self-identification — see that call site's own comment for the
    /// precedence between the two.
    pub aggregator_host: Option<crate::models::AggregatorHost>,
}

/// A live, ready-to-attach bearer credential for one request, plus the context header construction
/// needs: whether this token came from an OAuth subscription login, which unlocks a provider-specific
/// extra header set (see `send_with_retry`) — distinct from a plain static key — and, for a
/// Copilot/Codex-sourced credential, where the request must actually go (see [`DirectRouting`]).
pub struct Credential {
    key: ApiKey,
    is_oauth: bool,
    direct: Option<DirectRouting>,
}

impl Credential {
    pub fn new(key: impl Into<String>, is_oauth: bool) -> Self {
        Self {
            key: ApiKey::new(key),
            is_oauth,
            direct: None,
        }
    }

    /// Tags this credential with where its request must actually go and what extra headers it needs
    /// — used by a `CredentialSource` wrapper (e.g. `agent::main`'s per-provider OAuth resolution)
    /// that already knows, from which OAuth provider it resolved, that the plain gateway-relative URL
    /// this crate would otherwise build is wrong for this token. See [`RouteOverride`].
    pub fn with_direct_routing(mut self, routing: DirectRouting) -> Self {
        self.direct = Some(routing);
        self
    }
}

/// How [`GatewayClient`] obtains the token attached to every request — resolved fresh immediately
/// before each send rather than fixed at construction, so a token can expire and be refreshed
/// mid-process without this crate knowing anything about *how*. The only implementation shipped in
/// this crate is [`GatewayClient::new`]'s trivial static one; an OAuth-aware implementation (login,
/// local credential storage, locked cross-process refresh) belongs in a higher-level crate that
/// depends on this one, never the reverse — the same "wrong dependency direction" rule already
/// documented on [`ApiKey`], and the reason this crate still never *holds* a provider key of its own,
/// only ever borrows one fetched through this trait for the span of one request.
#[async_trait]
pub trait CredentialSource: Send + Sync {
    async fn credential(&self) -> Result<Credential>;
}

/// The default source: a single string fixed for the client's lifetime — what
/// [`GatewayClient::new`] builds internally.
struct StaticCredential(ApiKey);

#[async_trait]
impl CredentialSource for StaticCredential {
    async fn credential(&self) -> Result<Credential> {
        Ok(Credential {
            key: self.0.clone(),
            is_oauth: false,
            direct: None,
        })
    }
}

/// Build the underlying HTTP client with a given idle-read timeout, connect timeout fixed at
/// [`CONNECT_TIMEOUT`]. Factored out so [`GatewayClient::new`] and
/// [`GatewayClient::with_idle_timeout`] share one construction path.
fn build_http(read_timeout: Duration) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(read_timeout)
        .build()
        .map_err(|e| Error::Transport(e.to_string()))
}

/// An HTTP client pointed at a Beyond gateway base URL, authenticated with a `bai_v1` virtual key (or
/// any other [`CredentialSource`]).
pub struct GatewayClient {
    http: reqwest::Client,
    base_url: String,
    credential_source: Arc<dyn CredentialSource>,
    max_retries: u32,
    base_backoff: Duration,
    max_backoff: Duration,
    /// Extra headers merged onto every outgoing request, applied last so an operator-configured value
    /// wins over anything this client would otherwise set — the plumbing a self-hosted/proxied
    /// endpoint's custom auth/routing header needs (see [`with_extra_headers`](Self::with_extra_headers)).
    /// `Arc` so the (small, deployment-wide, effectively-static) map is cloned by pointer into the
    /// `'static` stream generator in [`ModelTransport::stream`], not copied per request.
    extra_headers: Arc<HashMap<String, String>>,
    /// Optional hook seam for the raw HTTP layer — only [`AgentHooks::after_provider_response`] is
    /// ever called from here (see [`with_hooks`](Self::with_hooks)'s doc comment for why the request-side
    /// half of the pair, `before_provider_request`, is instead called from `Agent::run_turn_once`).
    hooks: Option<Arc<dyn AgentHooks>>,
    /// The live Codex WebSocket transport's connection cache — `Some` for the client's whole lifetime
    /// unless disabled via [`with_codex_websocket`](Self::with_codex_websocket). One per client (not
    /// one per request), so a connection persists and gets reused across turns of the same session —
    /// see `codex_websocket`'s module doc comment.
    codex_websocket: Option<Arc<crate::codex_websocket::CodexWebSocketCache>>,
}

impl GatewayClient {
    /// Build a client for `base_url` (e.g. `http://ai.internal` or `http://127.0.0.1:8080`) using
    /// `api_key` (a `bai_v1…` virtual key, or a BYO provider key the gateway forwards untouched) as a
    /// fixed, non-expiring credential. Pre-first-byte retry defaults to [`MAX_RETRIES`]/
    /// [`BASE_BACKOFF`]; override with [`with_retry`](Self::with_retry) if an operator needs a
    /// different budget for this deployment. Idle-read timeout defaults to [`READ_TIMEOUT`]; override
    /// with [`with_idle_timeout`](Self::with_idle_timeout). Outbound proxy config isn't a client
    /// option here — reqwest already reads `HTTP_PROXY`/`HTTPS_PROXY` from the environment at the
    /// library level, with no code needed on our side.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        Self::with_credential_source(base_url, Arc::new(StaticCredential(ApiKey::new(api_key))))
    }

    /// Build a client whose credential is resolved fresh via `source` immediately before every
    /// request — e.g. an OAuth-aware source that transparently refreshes an expiring token. See
    /// [`CredentialSource`].
    pub fn with_credential_source(
        base_url: impl Into<String>,
        source: Arc<dyn CredentialSource>,
    ) -> Result<Self> {
        let http = build_http(READ_TIMEOUT)?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            credential_source: source,
            max_retries: MAX_RETRIES,
            base_backoff: BASE_BACKOFF,
            max_backoff: MAX_BACKOFF,
            extra_headers: Arc::new(HashMap::new()),
            hooks: None,
            codex_websocket: Some(Arc::new(crate::codex_websocket::CodexWebSocketCache::new())),
        })
    }

    /// Builder-style: override the pre-first-byte retry budget and exponential-backoff base (still
    /// capped at [`MAX_BACKOFF`], or at [`with_max_backoff`](Self::with_max_backoff)'s override).
    pub fn with_retry(mut self, max_retries: u32, base_backoff: Duration) -> Self {
        self.max_retries = max_retries;
        self.base_backoff = base_backoff;
        self
    }

    /// Builder-style: override the ceiling on a single backoff wait (exponential or
    /// `Retry-After`-derived). Defaults to [`MAX_BACKOFF`] (60s) — see that constant's doc comment.
    pub fn with_max_backoff(mut self, max_backoff: Duration) -> Self {
        self.max_backoff = max_backoff;
        self
    }

    /// Builder-style: merge `headers` onto every outgoing request, applied last so an operator's value
    /// wins over anything else this client would otherwise set for the same name. The plumbing a
    /// self-hosted or proxied endpoint's custom auth/routing header needs (pi's `model-registry.ts`
    /// supports the same per-deployment concept) — the CLI/settings surface that lets an operator
    /// actually configure this lives one layer up (`crates/agent/src/settings.rs`), not here.
    pub fn with_extra_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.extra_headers = Arc::new(headers);
        self
    }

    /// Builder-style: install a hook seam for the raw HTTP layer. Only
    /// [`AgentHooks::after_provider_response`] is ever called from here, once a response's status and
    /// headers are known, before its body starts streaming — pi's `afterProviderResponse`. The
    /// request-side half of the pair, [`AgentHooks::before_provider_request`], instead runs one layer
    /// up, in `Agent::run_turn_once` (the one place that already holds both the configured hooks and
    /// the not-yet-sent [`ModelRequest`]) — mutating it there is what actually reaches this client's
    /// own dialect/body construction, so there's nothing left for this layer to do for that half.
    /// A caller wanting both this transport-level observability and the loop's own tool/message hooks
    /// installs the *same* `Arc<dyn AgentHooks>` here and via `Agent::with_hooks`.
    pub fn with_hooks(mut self, hooks: Arc<dyn AgentHooks>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Builder-style: override the idle-read timeout between chunks on the streaming body (default
    /// [`READ_TIMEOUT`] — see that constant's doc comment on why it's an idle timeout, not a ceiling
    /// on total stream duration). Rebuilds the underlying HTTP client, since `reqwest::Client`'s
    /// timeouts are fixed at construction; connect timeout is unaffected.
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Result<Self> {
        self.http = build_http(timeout)?;
        Ok(self)
    }

    /// Builder-style: enable/disable the live Codex WebSocket transport for
    /// `RouteOverride::Prefixed`-routed (OpenAI-Codex-Responses/ChatGPT-subscription) requests — see
    /// `codex_websocket`'s module doc comment. Defaults to enabled: every Codex-routed turn attempts
    /// the WebSocket first, transparently falling back to the existing HTTP/SSE path on any
    /// unavailability or failure (matching pi's own `"auto"` transport default), so an existing caller
    /// is unaffected either way beyond the (best-case) request-size/latency win. Disabling it here
    /// only ever restores the exact pre-existing HTTP/SSE-only behavior.
    ///
    /// This is an escape hatch, not a supported day-to-day setting — **no CLI/settings surface wires
    /// it up yet** (out of scope for this round, which doesn't touch `crates/agent`). A later round
    /// should add one (e.g. a `--codex-transport sse` flag or a `settings.toml` key) only if a real
    /// operator need for pinning this off by default appears; until then every caller gets the
    /// WebSocket attempt by default, same as pi's own shipped client.
    pub fn with_codex_websocket(mut self, enabled: bool) -> Self {
        self.codex_websocket =
            enabled.then(|| Arc::new(crate::codex_websocket::CodexWebSocketCache::new()));
        self
    }
}

#[async_trait]
impl ModelTransport for GatewayClient {
    async fn stream(&self, mut req: ModelRequest) -> Result<EventStream> {
        // Every dialect's wire shape requires a `tool_use` to be immediately followed by its
        // `tool_result` — repair any orphaned one (a hand-edited/externally-loaded session, or a
        // not-yet-discovered code path) before it can reach a request and 400. Cheap no-op when the
        // message list is already well-formed (the overwhelmingly common case).
        if let std::borrow::Cow::Owned(repaired) =
            crate::dialect::repair_orphaned_tool_use(&req.messages)
        {
            req.messages = repaired.into();
        }
        // Every dialect rejects a message with empty/whitespace-only content and no tool calls —
        // reachable via `Message::error()`'s closing record, an immediate-abort turn, or cross-model
        // scrubbing. Fix the wire shape, not the persisted session (see the doc comment).
        if let std::borrow::Cow::Owned(fixed) =
            crate::dialect::ensure_non_empty_content(&req.messages)
        {
            req.messages = fixed.into();
        }
        // Resolved fresh for this turn rather than a field snapshot — the seam that lets a
        // credential expire and be refreshed mid-process (see `CredentialSource`'s doc comment).
        // `send_with_retry`'s own internal transient-failure retries never hit 401, so one fetch per
        // `stream()` call (once per model turn) is all that's needed, not one per retry attempt.
        // Fetched before the dialect is picked (as well as before `url` is built): GitHub Copilot's
        // proxy wants a different wire shape than a model's native classification for at least one id
        // (see `Dialect::for_model_via_copilot`'s doc comment), so dialect selection itself needs to
        // know whether this credential is Copilot-routed.
        let credential = self.credential_source.credential().await?;
        let via_copilot = credential
            .direct
            .as_ref()
            .is_some_and(|d| d.copilot_dynamic_headers);
        // pi-parity pass 20 Task 5: `ModelRequest::host` computed here, *before* dialect selection
        // (rather than down where the other route flags — `is_copilot`/`is_azure`/`is_codex` — are set
        // below), because dialect selection itself now needs it: a handful of bare ids genuinely speak
        // a different wire depending on which aggregator serves them (see
        // `crate::dialect::Dialect::for_model_with_host`'s own doc comment). A BYO override's explicit
        // `DirectRouting::aggregator_host` (populated by `crates/agent::gateway_credential` from the
        // override's own `base_url`) takes precedence over Fireworks' id-shape self-identification —
        // checked first below — though in practice the two never both apply to the same request:
        // Fireworks is its own host, never reached through a BYO override naming a different aggregator.
        req.host = credential
            .direct
            .as_ref()
            .and_then(|d| d.aggregator_host)
            .or_else(|| {
                crate::models::is_fireworks_model(&req.model.to_ascii_lowercase())
                    .then_some(crate::models::AggregatorHost::Fireworks)
            });
        // Pi-parity Fix 1 (Round 2): an operator-configured dialect override (`DirectRouting::
        // dialect_override` — a `models.json` `base_url` override naming a genuinely Anthropic- or
        // OpenAI-wire third-party provider whose model ids don't match `for_model_via_copilot`'s own
        // name heuristic) wins outright. Checked at this single call site — the one place every route
        // (gateway-relayed, `Prefixed`, `Direct`) picks its dialect — so the override applies
        // consistently to the URL below, `build_body`, the decoder, and the Anthropic-specific
        // header/beta logic further down, not just one of them.
        let dialect = credential
            .direct
            .as_ref()
            .and_then(|d| d.dialect_override)
            .unwrap_or_else(|| Dialect::for_model_via_copilot(&req.model, via_copilot, req.host));
        // Pi-parity Fix 2 (Round 2): an Azure resource pinned to a dated `api-version` needs that as a
        // query param on every request (`DirectRouting::query`) — only meaningful for a `Direct` route,
        // which is the only variant that builds a URL from scratch rather than relaying through the
        // gateway's own known-provider table.
        let direct_query = credential.direct.as_ref().and_then(|d| d.query.as_deref());
        let url = match credential.direct.as_ref().map(|d| &d.route) {
            Some(RouteOverride::Prefixed { prefix, path }) => {
                format!("{}{prefix}{path}", self.base_url)
            }
            Some(RouteOverride::Direct { base_url, path }) => match direct_query {
                Some(q) if !q.is_empty() => format!("{base_url}{path}?{q}"),
                _ => format!("{base_url}{path}"),
            },
            None => format!("{}{}", self.base_url, dialect.endpoint_path()),
        };
        // GitHub Copilot's per-turn dynamic headers (see `DirectRouting::copilot_dynamic_headers`'s
        // doc comment) — computed here, before `req` is otherwise consumed, from this turn's own
        // messages. `None` for every credential but a Copilot one.
        let copilot_dynamic = credential
            .direct
            .as_ref()
            .filter(|d| d.copilot_dynamic_headers)
            .map(|_| {
                (
                    copilot_initiator(&req.messages),
                    copilot_has_images(&req.messages),
                )
            });
        let direct_headers: Vec<(&'static str, String)> = credential
            .direct
            .as_ref()
            .map(|d| d.static_headers.clone())
            .unwrap_or_default();
        // A non-Bearer auth header (see `DirectRouting::auth_header`'s doc comment) — `None` for
        // every route but one that explicitly opts out of `Authorization: Bearer` (Azure OpenAI's
        // `api-key`, Task #8).
        let auth_header = credential
            .direct
            .as_ref()
            .and_then(|d| d.auth_header.clone());
        // Pi-parity Fix 4 (Round 2): a prefix prepended to `auth_header`'s value (Cloudflare AI
        // Gateway's `cf-aig-authorization: Bearer <key>` — a named header, like Azure's `api-key`,
        // but carrying a Bearer-prefixed value, unlike Azure's bare one). `None` unless both this and
        // `auth_header` are set — see `DirectRouting::auth_header_prefix`'s doc comment.
        let auth_header_prefix = credential
            .direct
            .as_ref()
            .and_then(|d| d.auth_header_prefix.clone());
        // Distinguishes Copilot- and Azure-routed requests for dialect-body branches that need to
        // know (gpt-5.x reasoning-disable suppression, Azure's prompt_cache_retention suppression) —
        // reuses the same signals already computed above rather than adding a third detection path.
        req.is_copilot = via_copilot;
        // pi-parity (models/dialects pass, Task C): a static-API-key Azure config sends `auth_header:
        // "api-key"` (checked first, unchanged), but Entra ID / Azure AD Bearer-token auth configures
        // `base_url` + `deployment_name` and correctly leaves `auth_header` unset (Bearer *is* the
        // right scheme there) — the narrower check alone missed that shape entirely, so those requests
        // never got Azure's own reasoning-disable/prompt-cache-retention suppression at all. Reuses
        // `DirectRouting::deployment_name` (already populated by `crates/agent/src/gateway_credential
        // .rs`'s BYO-override branch whenever `ModelOverride::deployment_name` is set — read here via
        // `credential.direct`, not a new field) rather than adding a second, independent detection path:
        // `gateway_credential.rs`'s own `is_azure_host`/`over.deployment_name.is_some()` primitive (used
        // only for its `api-version` query-param gating today) was never reconciled with this flag
        // until now — this *is* that reconciliation, entirely on the already-existing `DirectRouting`
        // shape.
        req.is_azure = auth_header.as_deref() == Some("api-key")
            || credential
                .direct
                .as_ref()
                .is_some_and(|d| d.deployment_name.is_some());
        // Same `RouteOverride::Prefixed` signal that already picked Codex's own URL/path above — not
        // a second, independent detection path. See `ModelRequest::is_codex`'s own doc comment for the
        // dialect-body branch this feeds.
        req.is_codex = matches!(
            credential.direct.as_ref().map(|d| &d.route),
            Some(RouteOverride::Prefixed { .. })
        );
        // `req.host` was already computed above, before dialect selection — see that assignment's own
        // comment.
        // Anthropic's own OAuth identity-spoofing headers (`CLAUDE_CODE_BETA`/`CLAUDE_CLI_IDENTITY`
        // etc., below) and body shape (Claude Code's identity system prompt, canonical tool-name
        // casing — `dialect::anthropic::build_body`) only apply to a genuine direct-to-Anthropic
        // request — a Claude-family model routed through a Copilot credential (`direct.is_some()`) is
        // `credential.is_oauth` too, but must not also claim to be the Anthropic-official Claude Code
        // client to GitHub's Copilot proxy. Computed once and fed to both the request body and the
        // decoder (reversing tool_use names back) so the two halves of the round-trip always agree.
        let has_direct_override = credential.direct.is_some();
        let is_oauth = credential.is_oauth && !has_direct_override;
        let mut body = dialect.build_body(&req, is_oauth);
        // Pi-parity Fix 2 (Round 2): Azure OpenAI's deployment name (`DirectRouting::deployment_name`)
        // overwrites just the wire-level `"model"` field `build_body` already set — never
        // `req.model`/`ModelRequest::model` itself, which stays keyed on the app-level id the operator
        // configured capabilities for (`crate::models::capabilities`, thinking budgets, …) rather than
        // whatever wire name the request happens to go out under.
        if let Some(name) = credential
            .direct
            .as_ref()
            .and_then(|d| d.deployment_name.as_deref())
        {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("model".to_string(), Value::String(name.to_string()));
            }
        }
        let hooks = self.hooks.clone();
        // Pi-parity gap: nothing between `ModelRequest` and this dialect-built body ever exposed the
        // *literal* wire-format JSON to a hook — only the abstract, pre-dialect `ModelRequest` via
        // `before_provider_request` above (one layer up, before `build_body` even ran). Mirrors pi's
        // `onPayload`/`beforeProviderPayload` (see `AgentHooks::before_provider_payload`'s own doc
        // comment). Same "fails open" convention as `before_provider_request`: a panicking hook's
        // (possibly partial) mutation is discarded and `body` falls back to exactly what `build_body`
        // produced.
        if let Some(hooks) = &hooks {
            let before_hook = body.clone();
            if catch_tool_panic(hooks.before_provider_payload(&mut body))
                .await
                .is_err()
            {
                body = before_hook;
            }
        }

        // Codex-routed requests get one shot at the live WebSocket transport before falling through
        // to the ordinary HTTP/SSE path below — see `codex_websocket`'s module doc comment for the
        // full design (connection cache, delta diffing, failure classification). Eager (awaited here,
        // not deferred into the lazy `try_stream!` below): unlike the SSE path's own laziness, actually
        // deciding whether to fall back requires knowing the outcome of a real connect/send/first-event
        // attempt, which `ModelTransport::stream`'s own doc comment already allows ("errors here are
        // connection/setup failures"). `dialect == Dialect::OpenAiResponses` is a defensive
        // belt-and-suspenders check alongside `req.is_codex` — every real Codex route is this dialect,
        // and the module hard-depends on its specific wire/decoder shape.
        if req.is_codex && dialect == Dialect::OpenAiResponses {
            if let Some(cache) = self.codex_websocket.clone() {
                if let Some(ws_url) = crate::codex_websocket::to_ws_url(&url) {
                    let request_id = req
                        .cache_key
                        .clone()
                        .unwrap_or_else(crate::codex_websocket::generate_request_id);
                    let ws_headers = crate::codex_websocket::build_headers(
                        credential.key.expose(),
                        &direct_headers,
                        &request_id,
                    );
                    match crate::codex_websocket::try_stream(
                        cache,
                        ws_url,
                        ws_headers,
                        req.cache_key.clone(),
                        body.clone(),
                    )
                    .await
                    {
                        crate::codex_websocket::Attempt::Started(stream) => return Ok(stream),
                        crate::codex_websocket::Attempt::Failed(e) => return Err(e),
                        // Nothing streamed yet — fall through to the existing HTTP/SSE path below,
                        // completely unchanged, exactly as if this block didn't exist.
                        crate::codex_websocket::Attempt::Fallback => {}
                    }
                }
            }
        }

        let tools_for_decoder = req.tools.clone();
        let http = self.http.clone();
        let max_retries = self.max_retries;
        let base_backoff = self.base_backoff;
        let max_backoff = self.max_backoff;
        let extra_headers = self.extra_headers.clone();

        let is_anthropic = dialect == Dialect::Anthropic;
        // pi-parity (models/dialects pass, Task D): gated to Fireworks-hosted models specifically, not
        // every OpenAI-wire-routed request — pi's own `compat.sendSessionAffinityHeaders` is a
        // per-provider catalogue flag that's only ever `true` for Fireworks (and Cloudflare-related
        // catalogues, deliberately out of scope — see `is_fireworks_model`'s own callers), never for
        // native OpenAI/Groq/Together/every other OpenAI-wire provider. Reuses `req.host` (set above
        // from the same `is_fireworks_model` check) rather than a second, independent detection path.
        let is_fireworks = req.host == Some(crate::models::AggregatorHost::Fireworks);
        // Both OpenAI wire dialects use this for connection-level session-affinity routing, distinct
        // from `prompt_cache_key`'s cache-node affinity in the body — matches pi's
        // `openai-responses.ts` (`headers["x-client-request-id"] = sessionId`) and
        // `openai-completions.ts` (`compat.sendSessionAffinityHeaders`), reusing the same
        // per-conversation `cache_key` value pi's own `sessionId` carries. Suppressed when the request
        // opted out of caching (`no_cache`): pinning a one-off request to a specific gateway/cache node
        // makes no sense when it's explicitly not trying to read a cache back, and pi's own dialects
        // gate this the same way `no_cache` gates the body's own `prompt_cache_key`/`cache_control`.
        let session_affinity_header =
            (matches!(dialect, Dialect::OpenAiResponses | Dialect::OpenAi)
                && !req.no_cache
                && is_fireworks)
                .then_some(req.cache_key.as_deref())
                .flatten()
                .map(str::to_string);
        // pi's Chat Completions dialect additionally sends `x-session-affinity` alongside `session_id`
        // and `x-client-request-id` (`openai-completions.ts`'s `compat.sendSessionAffinityHeaders`
        // branch); the Responses dialect never sends it.
        let send_x_session_affinity = dialect == Dialect::OpenAi && is_fireworks;
        let needs_interleaved_beta = needs_interleaved_thinking_beta(&req.model);
        let needs_fine_grained_tool_streaming_beta = !req.tools.is_empty()
            && !crate::models::capabilities_for_route(&req.model, req.is_codex, req.is_azure)
                .supports_eager_tool_streaming;
        // Same condition already used above to decide whether to attempt the WebSocket transport at
        // all — reused here (not re-derived) so the HTTP/SSE fallback path this falls through to can
        // zstd-compress the body the same way the real Codex backend's own client does, without a
        // second, independent "is this Codex" check drifting from the first.
        let is_codex_sse_fallback = req.is_codex && dialect == Dialect::OpenAiResponses;
        let stream = async_stream::try_stream! {
            // Retry the request up to the first byte: a transient failure (refused connection, 429,
            // 503) is re-issued with backoff. We do *not* retry once events have started flowing — a
            // mid-stream drop would replay partial output — so that surfaces as a transport error the
            // loop handles instead (see `Agent::run_events`).
            let resp = send_with_retry(
                &http,
                &url,
                credential.key.expose(),
                &body,
                is_codex_sse_fallback,
                is_anthropic,
                is_oauth,
                needs_interleaved_beta,
                needs_fine_grained_tool_streaming_beta,
                session_affinity_header.as_deref(),
                send_x_session_affinity,
                &direct_headers,
                copilot_dynamic,
                auth_header.as_deref(),
                auth_header_prefix.as_deref(),
                max_retries,
                base_backoff,
                max_backoff,
                &extra_headers,
            )
            .await?;

            // Observability seam (Task #14, pi-parity: pi's real `onResponse`,
            // `packages/coding-agent/src/core/sdk.ts:340-346`, wired to the `"after_provider_response"`
            // extension event — see `AgentHooks::after_provider_response`'s own doc comment for the
            // full citation): fires once the response's status/headers are known, before its body
            // starts streaming. Read-only — the response itself is already normalized into
            // `StreamEvent`s the loop consumes directly, so there's nothing here to rewrite.
            if let Some(hooks) = &hooks {
                let headers: Vec<(String, String)> = resp
                    .headers()
                    .iter()
                    .map(|(name, value)| {
                        (
                            name.to_string(),
                            value.to_str().unwrap_or_default().to_string(),
                        )
                    })
                    .collect();
                // Same "fails open" treatment every other hook call site gets (see
                // `before_provider_payload` above): a panicking hook must not unwind through the
                // in-flight stream it's only supposed to be observing.
                let _ = catch_tool_panic(hooks.after_provider_response(resp.status().as_u16(), &headers))
                    .await;
            }

            // Frame the chunked body line-by-line. A partial trailing line is buffered across chunks
            // until its terminator arrives. The framing (byte buffering + line-terminator split) lives
            // in `LineFramer` — see its doc comment for why it buffers raw *bytes*, not a lossy per-chunk
            // string. `sse_buf` is the separate SSE-level buffer that joins consecutive `data:` lines
            // belonging to one logical event (see `SseEventBuffer`'s doc comment) — real
            // Anthropic/OpenAI never split an event across lines, but a spec-conformant intermediary
            // could, and the SSE spec's own event boundary is a blank line, not a per-`data:`-line one.
            let mut decoder = dialect.decoder(is_oauth, tools_for_decoder);
            let mut framer = LineFramer::new();
            let mut sse_buf = SseEventBuffer::new();
            let mut bytes = resp.bytes_stream();
            while let Some(chunk) = bytes.next().await {
                // A tagged prefix, not `e.to_string()` alone: this is the one call site that turns a
                // live `reqwest::Error` into a mid-stream failure (a connection reset, a read timeout,
                // an unexpected EOF — the body was already flowing, so pre-first-byte retry above
                // never sees it), and `Agent::is_retryable_mid_stream` needs to tell "the network
                // dropped us, safe to restart the turn" apart from "the provider rejected the request"
                // without re-deriving reqwest's classification from its Display text.
                let chunk =
                    chunk.map_err(|e| Error::Transport(format!("{MID_STREAM_NETWORK_ERROR}: {e}")))?;
                framer.extend(&chunk)?;
                while let Some(line) = framer.next_line() {
                    let line = std::str::from_utf8(&line)
                        .map_err(|e| Error::Transport(format!("invalid UTF-8 in SSE stream: {e}")))?;
                    for ev in push_sse_line(decoder.as_mut(), &mut sse_buf, line)? {
                        yield ev;
                    }
                }
            }
            if let Some(line) = framer.take_tail() {
                let line = std::str::from_utf8(&line)
                    .map_err(|e| Error::Transport(format!("invalid UTF-8 in SSE stream: {e}")))?;
                for ev in push_sse_line(decoder.as_mut(), &mut sse_buf, line)? {
                    yield ev;
                }
            }
            // Flush a final event that never got its trailing blank-line terminator (a stream that
            // ends mid-event) — matches pi's own end-of-stream `flushSseEvent` call.
            for ev in push_sse_line(decoder.as_mut(), &mut sse_buf, "")? {
                yield ev;
            }
            for ev in decoder.finish()? {
                yield ev;
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Reassembles a chunked byte stream into whole line-terminated lines — the SSE framing seam.
///
/// It buffers raw *bytes*, not a per-chunk lossy string: a TCP/HTTP chunk boundary can split a
/// multi-byte UTF-8 character, and `from_utf8_lossy` per chunk would replace each half with U+FFFD,
/// silently corrupting non-ASCII tool arguments and prose. Every terminator this framer recognizes
/// (`\n`, `\r\n`, or a bare `\r`) is itself a single-byte ASCII value that never falls inside a UTF-8
/// multi-byte sequence, so every terminated line handed back by [`next_line`](Self::next_line) is
/// guaranteed whole UTF-8; only the unterminated tail — which may split a character — stays buffered
/// for the next chunk (surfaced by [`take_tail`](Self::take_tail) at end-of-stream).
///
/// Public so the streaming decode hot path is benchable in isolation (`benches/decode.rs`), the same
/// way the gateway exposes its request-scan primitives.
///
/// Backed by a [`BytesMut`]: [`next_line`](Self::next_line) finds the terminator with SIMD
/// [`memchr::memchr2`] and hands the line back via `split_to`, an O(1) pointer split that shares the
/// backing allocation — so a line costs neither a per-line heap allocation nor a memmove of the buffer
/// remainder. (A `Vec<u8>` framer paid both on *every* line: `drain(..=nl).collect()` allocates the
/// line and shifts the rest of the buffer down, which is O(lines × remaining) when a chunk carries many
/// coalesced lines.)
#[derive(Default)]
pub struct LineFramer {
    buf: BytesMut,
}

/// Ceiling on a single buffered (unterminated) line. A legitimate SSE `data:` line — even one
/// carrying a large embedded payload — stays orders of magnitude under this; it exists solely to
/// bound how much a malformed or adversarial stream (a line whose `\n` never arrives, or a
/// pathologically long one from a misbehaving provider/gateway) can grow the buffer before the
/// stream errors out instead of the process running out of memory.
const MAX_BUFFERED_LINE_BYTES: usize = 32 * 1024 * 1024;

/// Find the next SSE line terminator in `buf`, returning `(start, len)` where `len` is how many bytes
/// the terminator itself occupies — `1` for a bare `\n` or a bare `\r`, `2` for a `\r\n` pair (treated
/// as one terminator, not two separate lines). `None` if no complete terminator is present yet.
///
/// The SSE spec accepts all three of LF, CR, and CRLF as a line terminator (matches pi's own
/// `nextLineBreakIndex`/`consumeLine`, `anthropic-messages.ts`) — this crate used to only recognize
/// `\n`, so a lone `\r` (no `\n` ever following) sat buffered indefinitely and could merge two logical
/// SSE lines into one once a later, unrelated `\n` finally arrived. A trailing `\r` with nothing after
/// it yet is ambiguous — it might be the first half of a CRLF pair whose `\n` hasn't arrived in a later
/// chunk — so that case waits for more data (or is resolved by [`LineFramer::take_tail`] at
/// end-of-stream) rather than splitting prematurely.
fn find_line_break(buf: &[u8]) -> Option<(usize, usize)> {
    let pos = memchr::memchr2(b'\r', b'\n', buf)?;
    if buf[pos] == b'\n' {
        return Some((pos, 1));
    }
    match buf.get(pos + 1) {
        Some(b'\n') => Some((pos, 2)), // CRLF: one terminator, not two
        Some(_) => Some((pos, 1)),     // bare CR, immediately followed by a non-`\n` byte
        None => None,                  // trailing CR — could still become CRLF; wait for more data
    }
}

impl LineFramer {
    /// A framer with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a freshly-received chunk to the buffer. Errors if the unterminated line being
    /// assembled would exceed [`MAX_BUFFERED_LINE_BYTES`] — see that constant's doc comment.
    pub fn extend(&mut self, chunk: &[u8]) -> std::result::Result<(), Error> {
        self.buf.extend_from_slice(chunk);
        if self.buf.len() > MAX_BUFFERED_LINE_BYTES {
            return Err(Error::Transport(format!(
                "SSE line exceeded {MAX_BUFFERED_LINE_BYTES} bytes without a newline"
            )));
        }
        Ok(())
    }

    /// Pop the next complete line (including its trailing terminator — `\n`, `\r\n`, or a bare `\r`;
    /// see [`find_line_break`]), or `None` if the buffer holds no full line yet — the caller then
    /// awaits the next chunk. The returned [`Bytes`] shares the
    /// framer's backing buffer (no copy); it's dropped as soon as the line is decoded, freeing that
    /// region for the buffer to reclaim.
    pub fn next_line(&mut self) -> Option<Bytes> {
        let (start, len) = find_line_break(&self.buf)?;
        Some(self.buf.split_to(start + len).freeze())
    }

    /// The leftover unterminated tail at end-of-stream (a final line with no trailing newline), or
    /// `None` if the buffer is empty. Consumes the buffer.
    pub fn take_tail(&mut self) -> Option<Bytes> {
        if self.buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buf).freeze())
        }
    }
}

/// Whether an Anthropic-dialect request needs the interleaved-thinking beta opt-in, letting the model
/// weave thinking between tool calls across a turn. Sent by default for every request to a model whose
/// thinking shape isn't already `Adaptive` — matches pi's own default-on gate (`anthropic-messages.ts`:
/// `interleavedThinking ?? true`, `needsInterleavedBeta = interleavedThinking &&
/// model.compat?.forceAdaptiveThinking !== true`), independent of whether *this particular turn*
/// requests new thinking (a later turn with thinking off still benefits from interleaving on a turn
/// that follows it, or from consistent headers across a session's history of requests). `Adaptive`
/// models interleave by default, so sending the opt-in for them would be a harmless no-op at best —
/// skipped to keep the header list accurate to what the request actually needs.
fn needs_interleaved_thinking_beta(model: &str) -> bool {
    crate::models::capabilities(model).thinking != crate::models::ThinkingShape::Adaptive
}

/// Comma-joined `anthropic-beta` opt-ins for a request, or empty when neither applies — prompt
/// caching has been GA for a long time now and no longer needs (or accepts as meaningful) the old
/// `prompt-caching-2024-07-31` opt-in header pi itself has already dropped, so this crate doesn't send
/// it either. The fine-grained tool-streaming beta and each tool definition's own
/// `eager_input_streaming` marker (see `dialect::anthropic::mark_eager_tool_streaming`) are mutually
/// exclusive, so that beta only fires for a model that lacks the per-tool marker.
fn anthropic_betas(
    needs_interleaved: bool,
    needs_fine_grained_tool_streaming: bool,
) -> Vec<&'static str> {
    let mut betas = Vec::new();
    if needs_interleaved {
        betas.push(INTERLEAVED_THINKING_BETA);
    }
    if needs_fine_grained_tool_streaming {
        betas.push(FINE_GRAINED_TOOL_STREAMING_BETA);
    }
    betas
}

/// GitHub Copilot expects `X-Initiator` to say whether this turn is user-initiated or a follow-up
/// after the assistant/a tool already spoke — matches pi's `inferCopilotInitiator`
/// (`packages/ai/src/api/github-copilot-headers.ts`): the *last* message's role decides it, defaulting
/// to `"user"` when there's no history at all.
fn copilot_initiator(messages: &[crate::message::Message]) -> &'static str {
    match messages.last() {
        Some(m) if m.role != crate::message::Role::User => "agent",
        _ => "user",
    }
}

/// GitHub Copilot requires `Copilot-Vision-Request: true` on any turn carrying an image — an
/// `Image` content block (a fresh attachment) or a `ToolResult`'s own `images` (e.g. a screenshot a
/// tool produced) — matches pi's `hasCopilotVisionInput`. Both block shapes only ever appear on a
/// `User`-role message in this crate's model (see `message::ContentBlock`'s own doc comment), so
/// scanning every message's content is equivalent to pi's separate `user`/`toolResult`-role check.
fn copilot_has_images(messages: &[crate::message::Message]) -> bool {
    use crate::message::ContentBlock;
    messages.iter().any(|m| {
        m.content.iter().any(|block| match block {
            ContentBlock::Image { .. } => true,
            ContentBlock::ToolResult { images, .. } => !images.is_empty(),
            _ => false,
        })
    })
}

/// POST the request body, retrying transient failures with exponential backoff until a successful
/// response or the retry budget is exhausted. Honors a `Retry-After` header when the server sends one.
// 11 arguments, all independent inputs a single call site (`GatewayClient::stream`) already has in
// scope from `self`/the request — bundling them into a struct would just be a second place those same
// fields live, not a reduction in what the function needs to know. Private, single-caller helper, not
// a public API shape, so the usual "too many params signals a missing abstraction" concern doesn't
// apply here.
#[allow(clippy::too_many_arguments)]
async fn send_with_retry(
    http: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &Value,
    // Whether this request is Codex's own HTTP/SSE fallback path (the WebSocket transport declined
    // or wasn't attempted) — the one case where the body is zstd-compressed before sending, mirroring
    // pi's own `compressRequestBodyZstd`/`content-encoding: zstd` behavior for that exact backend.
    // `false` for every other route, which keeps sending the plain JSON body unchanged.
    is_codex_sse_fallback: bool,
    is_anthropic: bool,
    is_oauth: bool,
    needs_interleaved_beta: bool,
    needs_fine_grained_tool_streaming_beta: bool,
    session_affinity_header: Option<&str>,
    send_x_session_affinity: bool,
    // Static headers a `DirectRouting`-carrying credential requires (Codex's `chatgpt-account-id`;
    // Copilot's fixed editor-identity headers) — empty for every plain gateway-routed request.
    direct_headers: &[(&'static str, String)],
    // GitHub Copilot's per-turn dynamic headers, precomputed in `GatewayClient::stream` from this
    // turn's own messages: `(X-Initiator value, has an image)` — `None` for every provider but
    // Copilot.
    copilot_dynamic: Option<(&'static str, bool)>,
    // Send `api_key` verbatim in this named header instead of `Authorization: Bearer <api_key>` (and
    // omit `Authorization` entirely) — see `DirectRouting::auth_header`'s doc comment. `None` for
    // every route but Azure OpenAI's `api-key` (Task #8, pi-parity).
    auth_header: Option<&str>,
    // Prepended to `api_key` when sent through `auth_header` above — Cloudflare AI Gateway's
    // `cf-aig-authorization: Bearer <key>` (Fix 4, pi-parity Round 2). `None` for every route but
    // that one, including Azure's (whose `api-key` value stays bare). Has no effect when
    // `auth_header` itself is `None`.
    auth_header_prefix: Option<&str>,
    max_retries: u32,
    base_backoff: Duration,
    max_backoff: Duration,
    // Operator-configured headers merged onto every request (see
    // `GatewayClient::with_extra_headers`) — empty for a client that never set any. Applied *last*,
    // after every other header this function sets, so an operator's own value always wins on a name
    // collision (e.g. overriding `anthropic-version`, or adding a reverse-proxy auth header a
    // self-hosted endpoint needs).
    extra_headers: &HashMap<String, String>,
) -> Result<reqwest::Response> {
    let mut attempt = 0u32;
    loop {
        // Codex's HTTP/SSE fallback accepts a zstd-compressed body (bandwidth optimization only —
        // the endpoint still accepts plain JSON, so a compression failure falls back to that rather
        // than failing the request). Every other route keeps sending `.json(body)` unchanged.
        let mut builder = if is_codex_sse_fallback {
            match serde_json::to_vec(body)
                .ok()
                .and_then(|bytes| crate::codex_websocket::compress_sse_fallback_body(&bytes))
            {
                Some(compressed) => http
                    .post(url)
                    .header(
                        crate::codex_websocket::SSE_FALLBACK_CONTENT_ENCODING.0,
                        crate::codex_websocket::SSE_FALLBACK_CONTENT_ENCODING.1,
                    )
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(compressed),
                None => http.post(url).json(body),
            }
        } else {
            http.post(url).json(body)
        };
        builder = match auth_header {
            // Azure OpenAI shape (no prefix, the ordinary case): the bare key, verbatim, in its own
            // header — never `Authorization` (see `DirectRouting::auth_header`'s doc comment on why
            // sending both is a real risk, not just redundant). Cloudflare AI Gateway shape (`Some`
            // prefix, Fix 4): the same named header, but carrying a Bearer-prefixed value.
            Some(name) => match auth_header_prefix {
                Some(prefix) => builder.header(name, format!("{prefix}{api_key}")),
                None => builder.header(name, api_key),
            },
            None => builder.bearer_auth(api_key),
        };
        for (name, value) in direct_headers {
            builder = builder.header(*name, value.as_str());
        }
        if let Some((initiator, has_images)) = copilot_dynamic {
            // Matches pi's `buildCopilotDynamicHeaders` (`github-copilot-headers.ts`).
            builder = builder
                .header("X-Initiator", initiator)
                .header("Openai-Intent", "conversation-edits");
            if has_images {
                builder = builder.header("Copilot-Vision-Request", "true");
            }
        }
        if let Some(session_id) = session_affinity_header {
            // pi sends `session_id` (`compat.sendSessionIdHeader`, true by default for native OpenAI)
            // and `x-client-request-id` for both OpenAI dialects, both carrying the same value. Missing
            // `session_id` risked cache/session-affinity routing landing on a different backend node
            // per turn even though `x-client-request-id` was already correct.
            builder = builder.header("session_id", session_id);
            builder = builder.header("x-client-request-id", session_id);
            // Chat Completions also gets `x-session-affinity` (`openai-completions.ts`'s
            // `compat.sendSessionAffinityHeaders` branch) — the Responses dialect never sends it.
            if send_x_session_affinity {
                builder = builder.header("x-session-affinity", session_id);
            }
        }
        if is_anthropic {
            builder = builder.header("anthropic-version", ANTHROPIC_VERSION);
            // pi's `anthropic-messages.ts` sends this header in all three of its `createClient`
            // branches (Copilot Bearer auth, OAuth, and plain API-key/header-owned auth) —
            // unconditional here too, not gated on `is_oauth` alone, to match exactly. Believed benign
            // under a headless `reqwest` client with no browser fingerprint to trigger whatever
            // server-side check this header suppresses.
            builder = builder.header("anthropic-dangerous-direct-browser-access", "true");
            // OAuth betas lead the list — matches pi's own exact ordering
            // (`claude-code-20250219,oauth-2025-04-20,[...]`).
            let mut betas = Vec::new();
            if is_oauth {
                betas.push(CLAUDE_CODE_BETA);
                betas.push(OAUTH_BETA);
            }
            betas.extend(anthropic_betas(
                needs_interleaved_beta,
                needs_fine_grained_tool_streaming_beta,
            ));
            // Omit the header entirely when nothing needs it, rather than sending an empty
            // `anthropic-beta:` value — matching pi's own conditional-spread behavior.
            if !betas.is_empty() {
                builder = builder.header("anthropic-beta", betas.join(","));
            }
            if is_oauth {
                // Identity headers Anthropic's OAuth-gated endpoint expects from its own official
                // Claude Code client — see `CLAUDE_CODE_BETA`'s doc comment for why sending these is
                // a deliberate, confirmed choice, not an oversight.
                builder = builder
                    .header("user-agent", CLAUDE_CLI_IDENTITY)
                    .header("x-app", "cli");
            }
        }
        // `RequestBuilder::header` *appends* rather than replaces — calling it for a name this
        // function already set above (an operator overriding `anthropic-version` for a self-hosted/
        // proxied endpoint, say) would send the name twice on the wire instead of the operator's value
        // winning. Building the request, then `HeaderMap::insert`-ing each extra header directly, gives
        // replace semantics instead — matching this builder's own documented "applied last, wins on a
        // name collision" contract.
        let mut request = match builder.build() {
            Ok(request) => request,
            Err(e) => return Err(Error::Transport(e.to_string())),
        };
        for (name, value) in extra_headers {
            if let (Ok(header_name), Ok(header_value)) = (
                reqwest::header::HeaderName::from_bytes(name.as_bytes()),
                reqwest::header::HeaderValue::from_str(value),
            ) {
                request.headers_mut().insert(header_name, header_value);
            }
        }
        match http.execute(request).await {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                if is_retryable_status(status.as_u16()) && attempt < max_retries {
                    let hint = retry_after(&resp, max_backoff);
                    // A 429 needs a quick body peek before committing to a retry: some providers use it
                    // for genuine rate limiting (worth retrying — the request will likely succeed once
                    // the window resets) and others for quota/billing exhaustion (retrying only delays
                    // an unavoidable failure while burning the retry budget on it). The status code
                    // alone can't tell the two apart. Every other retryable status (5xx/529) is a pure
                    // infra hiccup, never a billing signal, so it skips this check.
                    if status.as_u16() == 429 {
                        let detail = read_error_body_capped(resp).await;
                        if is_quota_exhausted(&detail) {
                            // Codex/ChatGPT's own usage-limit body carries a machine code and (often)
                            // a reset time — surface a friendly, human-readable sentence ahead of the
                            // raw upstream detail when it's actually present, rather than only ever
                            // showing the bare JSON. Pure UX polish: the raw detail stays visible
                            // either way.
                            let msg = match codex_friendly_usage_limit_message(&detail) {
                                Some(friendly) => format!(
                                    "{friendly} (gateway returned {status}: {})",
                                    truncate_error_body(detail.trim())
                                ),
                                None => format!(
                                    "gateway returned {status}: {}",
                                    truncate_error_body(detail.trim())
                                ),
                            };
                            return Err(Error::Transport(msg));
                        }
                    }
                    let wait = backoff(attempt, hint, base_backoff, max_backoff);
                    attempt += 1;
                    futures_timer::Delay::new(wait).await;
                    continue;
                }
                // Non-retryable, or out of retries: surface the body so the caller sees *why* — capped,
                // since an upstream can return an arbitrarily large error page (an HTML error document
                // from a misconfigured proxy, say) and this ends up in logs and `AgentEvent::Error`.
                let detail = read_error_body_capped(resp).await;
                // A live 401 (the gateway itself rejected the key, as opposed to `run`/`serve`'s own
                // preflight "no key given at all" check, which this can't be — a request only reaches
                // here with *some* key attached) gets a pointed, actionable message naming the actual
                // cause instead of a bare status code: this crate has no CLI-flag names of its own to
                // reference (agent-core is layered under any consumer's own `--key`/`AI_AGENT_KEY`-style
                // flag), so this stays in terms of "the API key" generically, distinct from — and more
                // specific than — the upstream body alone, which for a 401 is often just `{"error":
                // {"type":"authentication_error"}}` with no hint at what to actually do about it.
                if status.as_u16() == 401 {
                    return Err(Error::Transport(format!(
                        "gateway rejected the request as unauthorized (401) — the API key being used \
                         is missing, invalid, expired, or lacks permission for this model; check the \
                         key and try again. Upstream detail: {}",
                        truncate_error_body(detail.trim())
                    )));
                }
                return Err(Error::Transport(format!(
                    "gateway returned {status}: {}",
                    truncate_error_body(detail.trim())
                )));
            }
            Err(e) => {
                // Connection-level failures (refused, reset, timed out) are exactly the transient
                // class worth retrying; a malformed-request error is not.
                if is_retryable_send_error(&e) && attempt < max_retries {
                    let wait = backoff(attempt, None, base_backoff, max_backoff);
                    attempt += 1;
                    futures_timer::Delay::new(wait).await;
                    continue;
                }
                return Err(Error::Transport(e.to_string()));
            }
        }
    }
}

/// Status codes worth retrying: rate limiting, the Anthropic-specific `529 overloaded`, the transient
/// 5xx gateway/upstream failures, and Cloudflare's `524` ("a connection was established between
/// Cloudflare and the origin server, but the origin did not respond before the connection timed out")
/// — pi's `packages/ai/src/utils/retry.ts:36` treats `"524"` as its only retry signal for that status,
/// matched here too since a `524` is exactly the same transient-infra-hiccup class as the other 5xx
/// entries. A 4xx other than 429 is the caller's fault — don't retry.
///
/// pi-parity fix: `408` (Request Timeout) and `409` (Conflict) were previously included here too, but
/// neither appears anywhere in pi's own retry classifiers (`openai-codex-responses.ts`,
/// `utils/retry.ts`) — there's no pi source justifying either, and both directly contradict this
/// function's own "a 4xx other than 429 is the caller's fault" rule above. Dropping them also restores
/// consistency with the outer whole-run retry layer (`agent::retry::WHOLE_RUN_RETRYABLE_STATUS_DIGITS`
/// in the `agent` crate), which never treated 408/409 as retryable — a persistent 408/409 exhausting
/// this function's pre-connect budget previously had nowhere else to be retried, unlike every other
/// status in this set.
fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 500 | 502 | 503 | 504 | 524 | 529)
}

/// Phrases seen in a 429 body when the rejection is quota/billing exhaustion rather than transient
/// rate limiting — retrying one of these will never succeed until the account itself changes, so it
/// isn't worth spending the retry budget on. Deliberately narrower than `agent::is_context_overflow`'s
/// throttle-exclusion list: this only needs to cover the "don't bother retrying a 429" case, not
/// classify every provider's wording.
const QUOTA_EXHAUSTED_PATTERNS: &[&str] = &[
    "insufficient_quota",
    "quota exceeded",
    "billing",
    "out of budget",
    "exceeded your current quota",
    // Codex/ChatGPT-specific usage-cap phrases — pi's `isTerminalRateLimitError`
    // (`openai-codex-responses.ts:114-118`) recognizes these alongside the generic phrases above.
    // Without them, a 429 carrying one of these Codex-only bodies fell through to ordinary
    // rate-limit handling and retried with full exponential backoff instead of failing fast.
    "gousagelimiterror",
    "freeusagelimiterror",
    "monthly usage limit reached",
    "available balance",
];

/// Whether a 429 response body indicates quota/billing exhaustion (fail fast) rather than ordinary
/// rate limiting (worth retrying). `pub`: `agent::retry::is_retryable_whole_run` (the `agent` crate)
/// reuses this same heuristic for its own raw-status-digit fallback, rather than duplicating the phrase
/// list — a quota-exhausted 429's message text (`"gateway returned 429 …: <body>"`) still contains
/// whichever [`QUOTA_EXHAUSTED_PATTERNS`] phrase the body did, so passing the whole message through
/// works the same as passing just the body does here.
pub fn is_quota_exhausted(body: &str) -> bool {
    let m = body.to_ascii_lowercase();
    QUOTA_EXHAUSTED_PATTERNS.iter().any(|p| m.contains(p))
}

/// A friendlier message for a Codex/ChatGPT-specific usage-limit body, matching pi's Codex-specific
/// `parseErrorResponse` (`openai-codex-responses.ts:1459-1484`): `{"error":{"code":
/// "usage_limit_reached"|"usage_not_included"|"rate_limit_exceeded", "plan_type": "plus", "resets_at":
/// <unix seconds>}}` becomes `"You have hit your ChatGPT usage limit (plus plan). Try again in ~45
/// min."`. `None` for anything that isn't this specific shape — deliberately narrower than pi's own
/// gate (which also fires on bare `response.status === 429` with no code match at all, safe there only
/// because pi's call site is already reached exclusively through its Codex-specific
/// `isTerminalRateLimitError` gate; beyond routes every provider through this same generic retry loop,
/// so requiring the actual `code` match keeps a non-Codex 429's quota body — Anthropic's, OpenAI's own
/// native billing error, DeepSeek's, etc. — showing its own raw detail unchanged instead of an
/// incongruous "ChatGPT usage limit" rewrite). Pure UX polish: the raw body this is derived from stays
/// visible alongside it either way (see the one call site), so a body that fails to parse or doesn't
/// match just yields `None` rather than being treated as an error itself.
fn codex_friendly_usage_limit_message(detail: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(detail).ok()?;
    let err = parsed.get("error")?;
    let code = err
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| err.get("type").and_then(Value::as_str))
        .unwrap_or_default();
    let is_usage_limit_code = [
        "usage_limit_reached",
        "usage_not_included",
        "rate_limit_exceeded",
    ]
    .iter()
    .any(|p| code.eq_ignore_ascii_case(p));
    if !is_usage_limit_code {
        return None;
    }
    let plan = err
        .get("plan_type")
        .and_then(Value::as_str)
        .map(|p| format!(" ({} plan)", p.to_ascii_lowercase()))
        .unwrap_or_default();
    let when = err
        .get("resets_at")
        .and_then(Value::as_i64)
        .map(|resets_at_secs| {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let mins = (resets_at_secs.saturating_sub(now_secs) as f64 / 60.0)
                .round()
                .max(0.0) as i64;
            format!(" Try again in ~{mins} min.")
        })
        .unwrap_or_default();
    Some(
        format!("You have hit your ChatGPT usage limit{plan}.{when}")
            .trim()
            .to_string(),
    )
}

/// Whether a `reqwest` send error is the transient connection class (refused/reset/timed out).
///
/// `is_connect()` only covers a failed TCP *handshake* (refused/unreachable) — it doesn't match a
/// connection that was accepted fine and then reset while the request was being sent, which reqwest
/// reports as the broader `Kind::Request` (`is_request()`) instead. Found live (not by any mock): a
/// fault-injecting proxy that accepts a connection and drops it immediately — a genuine network blip,
/// the same class this function's own name promises to cover — surfaced as a hard, un-retried failure
/// on the very first attempt, despite this exact case ("reset") already being named in this doc comment
/// as something that should be retried. `is_request()` is deliberately broader than a hand-picked
/// reset-specific check: everything in that bucket happens before any response is ever received, so
/// retrying it can't double-apply a change a provider already committed — the same safety property
/// `is_connect()`/`is_timeout()` already rely on.
fn is_retryable_send_error(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect() || e.is_request()
}

/// Cap on an upstream error body surfaced in [`Error::Transport`] — an error page (a misconfigured
/// proxy's HTML, say) can be arbitrarily large, and this text ends up in logs and `AgentEvent::Error`.
const MAX_ERROR_BODY_CHARS: usize = 4_000;

/// Truncate an error body to [`MAX_ERROR_BODY_CHARS`], on a char boundary, noting what was cut —
/// including *how much* was cut, matching pi's `truncateErrorText` (`error-body.ts:115-118`:
/// `` `${text.slice(0, maxChars)}... [truncated ${text.length - maxChars} chars]` ``). The omitted
/// count matters operationally: a bare "[truncated]" marker with no count leaves an on-call engineer
/// unable to tell a 1-char overflow from a multi-megabyte proxy error page from the marker alone.
fn truncate_error_body(s: &str) -> String {
    let total = s.chars().count();
    if total <= MAX_ERROR_BODY_CHARS {
        return s.to_string();
    }
    let kept: String = s.chars().take(MAX_ERROR_BODY_CHARS).collect();
    let omitted = total - MAX_ERROR_BODY_CHARS;
    format!("{kept}... [truncated {omitted} chars]")
}

/// Ceiling on bytes read from a non-2xx response body before giving up on it — the same defensive
/// bound [`crate::dialect::LineFramer`]'s buffer cap applies to the 2xx SSE path, but for the
/// error-body read: an error page (a misconfigured proxy's HTML, a hostile `Direct`-routed third
/// party) can be arbitrarily large, and `Response::text()` has no size limit of its own. Comfortably
/// above [`MAX_ERROR_BODY_CHARS`] (the *displayed* cap after truncation) so any real provider error
/// body is read in full; this only stops reading once a response has already gone far past anything a
/// genuine JSON/text error could need.
const MAX_ERROR_BODY_READ_BYTES: usize = 1024 * 1024;

/// Read a non-2xx response body, stopping at [`MAX_ERROR_BODY_READ_BYTES`] instead of buffering an
/// unbounded error page into memory. Lossily decoded: this is already a truncated, human-facing error
/// string (see [`truncate_error_body`]), never data a caller round-trips.
async fn read_error_body_capped(resp: reqwest::Response) -> String {
    let mut buf = Vec::new();
    let mut stream = resp.bytes_stream();
    while buf.len() < MAX_ERROR_BODY_READ_BYTES {
        match stream.next().await {
            Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
            _ => break,
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Parse a `Retry-After` response header into a duration, capped at `max_backoff`. Checks the
/// non-standard, millisecond-precision `retry-after-ms` header first — mirrors pi's Codex-specific
/// `getRetryAfterDelayMs` (`openai-codex-responses.ts:130-137`), which checks it before falling back to
/// the standard header, but applied generically to every provider's response here rather than gated on
/// a Codex-specific retry path: beyond routes Codex through this same generic retry loop as every other
/// provider, so there's no separate Codex-only call site to special-case instead. A provider that never
/// sends `retry-after-ms` simply never matches the first branch and falls through unchanged.
fn retry_after(resp: &reqwest::Response, max_backoff: Duration) -> Option<Duration> {
    if let Some(ms) = resp
        .headers()
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|raw| parse_retry_after_ms(raw, max_backoff))
    {
        return Some(ms);
    }
    let raw = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?;
    parse_retry_after(raw, max_backoff)
}

/// Parse a `retry-after-ms` header *value* (milliseconds, not seconds) into a wait, capped at
/// `max_backoff`. Mirrors pi's `getRetryAfterDelayMs`: any non-negative finite number is accepted
/// (including a fractional one — `Number.isFinite` in JS accepts floats same as this parses `f64`),
/// clamped up to zero for a negative value (clock skew, a malformed hint), and anything that doesn't
/// parse as a plain number at all is ignored rather than treated as a hard error. Split out from
/// [`retry_after`] so it's testable without a `reqwest::Response`.
fn parse_retry_after_ms(raw: &str, max_backoff: Duration) -> Option<Duration> {
    let millis: f64 = raw.trim().parse().ok()?;
    if !millis.is_finite() {
        return None;
    }
    // Clamp to `max_backoff` *before* converting: `Duration::from_secs_f64` panics if the seconds
    // value doesn't fit a `Duration` (e.g. a header carrying `1e25`), so the value handed to it must
    // already be bounded — clamping the resulting `Duration` afterward, as this used to, is too late.
    let secs = (millis.max(0.0) / 1000.0).min(max_backoff.as_secs_f64());
    Some(Duration::from_secs_f64(secs))
}

/// Parse a `Retry-After` header *value* into a wait, capped at `max_backoff`. RFC 7231 allows two
/// forms: delta-seconds (`120`) and an absolute HTTP-date (`Wed, 21 Oct 2025 07:28:00 GMT`). The
/// date form is converted to a delay from now; a date already in the past (clock skew, a stale hint)
/// yields no extra wait. Split out from [`retry_after`] so it's testable without a `reqwest::Response`.
fn parse_retry_after(raw: &str, max_backoff: Duration) -> Option<Duration> {
    let raw = raw.trim();
    // Delta-seconds: a bare non-negative integer count of seconds.
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(Duration::from_secs(secs).min(max_backoff));
    }
    // HTTP-date: the absolute instant to retry at. Anything we can't parse as either form is ignored.
    let target = httpdate::parse_http_date(raw).ok()?;
    let delay = target
        .duration_since(std::time::SystemTime::now())
        .unwrap_or(Duration::ZERO);
    Some(delay.min(max_backoff))
}

/// Applies +/-20% jitter to a computed exponential-backoff duration so a fleet of concurrent `beyond`
/// processes hitting the same transient failure (429/5xx) don't retry in perfect lockstep against an
/// already-degraded backend (thundering herd). `crates/agent::retry`'s outer, whole-run retry layer
/// already jitters for exactly this reason — but this crate's two *inner* layers ([`backoff`], here,
/// and `agent::mid_stream_backoff`) fire far more often (every turn / every stream, vs. only after a
/// whole run has already exhausted both inner layers), making them the bigger thundering-herd risk of
/// the two. Implemented independently rather than adding a cross-crate dependency on `crates/agent`.
///
/// Applied to the *uncapped* exponential value, before the caller's own `.min(max_backoff)` — a
/// saturated attempt (well past the cap) still collapses to exactly the cap after jitter, same as
/// before jitter existed, since even a -20% jittered value there is still far above the cap.
///
/// No `rand` crate dependency in this crate (checked `Cargo.toml`) — reuses the same zero-dependency
/// salt-plus-counter trick `crates/agent::retry::jitter` uses: a `RandomState`-seeded salt (draws OS
/// entropy once, fixed for the rest of the process's life) plus a monotonic per-call counter, hashed
/// together with `DefaultHasher` (fixed key, so the *only* source of variance is the salt/counter/
/// attempt inputs, not the hasher itself) and mapped onto `[0.8, 1.2]`. Salt gives cross-process
/// variance (two fleet processes at the same attempt number don't jitter identically); the counter
/// gives cross-call variance (the *same* attempt number, computed twice in one process, doesn't
/// jitter identically either — see `backoff_jitter_varies_across_calls` below).
///
/// `pub(crate)`: also reused by `agent::mid_stream_backoff`, this crate's other inner retry layer.
pub(crate) fn jitter(d: Duration, attempt: u32) -> Duration {
    use std::collections::hash_map::{DefaultHasher, RandomState};
    use std::hash::{BuildHasher, Hash, Hasher};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SALT: OnceLock<u64> = OnceLock::new();
    let salt = *SALT.get_or_init(|| RandomState::new().hash_one(0xC0FFEEu64));

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);

    let mut hasher = DefaultHasher::new();
    salt.hash(&mut hasher);
    seq.hash(&mut hasher);
    attempt.hash(&mut hasher);
    let hashed = hasher.finish();

    // Map the hash's top 32 bits onto [0.8, 1.2] (+/-20%, symmetric around the unjittered value).
    let unit = (hashed >> 32) as f64 / u32::MAX as f64; // [0, 1]
    let factor = 0.8 + unit * 0.4;
    d.mul_f64(factor)
}

/// The wait before the next attempt (0-indexed): the larger of the server's `Retry-After` hint and
/// our exponential backoff `base_backoff · 2^attempt` (± [`jitter`]), capped at `max_backoff`. The
/// server's own hint is honored exactly, unjittered — jitter only hedges against *our own* guess.
fn backoff(
    attempt: u32,
    retry_after: Option<Duration>,
    base_backoff: Duration,
    max_backoff: Duration,
) -> Duration {
    // `min(16)` keeps the shift well within `u32` (and `saturating_mul` mops up the rest); by then the
    // result has long since hit `max_backoff`.
    let exp_uncapped = base_backoff.saturating_mul(1u32 << attempt.min(16));
    let exp = jitter(exp_uncapped, attempt).min(max_backoff);
    match retry_after {
        Some(hint) => hint.max(exp).min(max_backoff),
        None => exp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_debug_redacts_the_plaintext() {
        let key = ApiKey::new("bai_v1_supersecret");
        let debug = format!("{key:?}");
        assert_eq!(debug, "ApiKey(***)");
        assert!(!debug.contains("supersecret"), "got: {debug}");
    }

    #[test]
    fn api_key_expose_returns_plaintext() {
        assert_eq!(ApiKey::new("bai_v1_abc").expose(), "bai_v1_abc");
    }

    #[test]
    fn line_framer_splits_whole_lines_and_buffers_the_tail() {
        let mut framer = LineFramer::new();
        framer.extend(b"data: {\"a\":1}\ndata: {\"b").unwrap();
        assert_eq!(framer.next_line().unwrap(), &b"data: {\"a\":1}\n"[..]);
        assert!(framer.next_line().is_none()); // unterminated remainder stays buffered
        framer.extend(b"\":2}\n").unwrap();
        assert_eq!(framer.next_line().unwrap(), &b"data: {\"b\":2}\n"[..]);
    }

    #[test]
    fn line_framer_errors_instead_of_growing_unbounded_on_an_unterminated_line() {
        // A malformed/adversarial stream (or a provider bug) that never sends a `\n` must not be
        // allowed to grow the buffer without bound — this is what caps it.
        let mut framer = LineFramer::new();
        let chunk = vec![b'x'; MAX_BUFFERED_LINE_BYTES];
        framer.extend(&chunk).unwrap(); // exactly at the cap: still fine
        assert!(framer.extend(b"y").is_err()); // one more byte tips it over
    }

    #[test]
    fn line_framer_splits_on_a_bare_cr_line_terminator() {
        // pi-parity fix: the SSE spec (and pi's own `nextLineBreakIndex`/`consumeLine`,
        // `anthropic-messages.ts`) accepts a lone `\r` (not followed by `\n`) as a valid line
        // terminator, same as LF and CRLF. Before this fix, `next_line` only searched for `\n`, so a
        // bare `\r` sat buffered and silently merged two logical SSE lines into one once a later,
        // unrelated `\n` arrived.
        let mut framer = LineFramer::new();
        framer
            .extend(b"data: {\"a\":1}\rdata: {\"b\":2}\n")
            .unwrap();
        assert_eq!(framer.next_line().unwrap(), &b"data: {\"a\":1}\r"[..]);
        assert_eq!(framer.next_line().unwrap(), &b"data: {\"b\":2}\n"[..]);
        assert!(framer.next_line().is_none());
    }

    #[test]
    fn line_framer_treats_crlf_as_one_terminator_not_two() {
        // A `\r\n` pair must yield exactly one line break, not a bare-CR line break immediately
        // followed by an empty LF-terminated line.
        let mut framer = LineFramer::new();
        framer.extend(b"data: 1\r\ndata: 2\n").unwrap();
        assert_eq!(framer.next_line().unwrap(), &b"data: 1\r\n"[..]);
        assert_eq!(framer.next_line().unwrap(), &b"data: 2\n"[..]);
        assert!(framer.next_line().is_none());
    }

    #[test]
    fn line_framer_holds_a_trailing_bare_cr_until_it_can_tell_it_apart_from_crlf() {
        // A `\r` at the very end of the currently-buffered bytes is ambiguous — the next chunk might
        // start with `\n`, making it a CRLF pair instead of a bare-CR-terminated line. Must not split
        // prematurely; the ambiguity resolves once more bytes (or end-of-stream, via `take_tail`)
        // arrive.
        let mut framer = LineFramer::new();
        framer.extend(b"data: 1\r").unwrap();
        assert!(
            framer.next_line().is_none(),
            "a trailing bare CR must wait for more data before splitting"
        );
        // Resolves as a bare CR once a non-`\n` byte follows.
        framer.extend(b"data: 2\n").unwrap();
        assert_eq!(framer.next_line().unwrap(), &b"data: 1\r"[..]);
        assert_eq!(framer.next_line().unwrap(), &b"data: 2\n"[..]);
    }

    #[test]
    fn anthropic_betas_are_mutually_exclusive_with_eager_tool_streaming() {
        // Baseline: neither beta needed — the stale prompt-caching-2024-07-31 opt-in is gone, so this
        // is empty, not a single-element list.
        assert_eq!(anthropic_betas(false, false), Vec::<&str>::new());
        // Interleaved thinking layers on top.
        assert_eq!(
            anthropic_betas(true, false),
            vec![INTERLEAVED_THINKING_BETA]
        );
        // The fine-grained tool-streaming beta only fires when the model lacks the per-tool
        // `eager_input_streaming` marker — never true for any current model, but exercised directly.
        assert_eq!(
            anthropic_betas(false, true),
            vec![FINE_GRAINED_TOOL_STREAMING_BETA]
        );
        assert_eq!(
            anthropic_betas(true, true),
            vec![INTERLEAVED_THINKING_BETA, FINE_GRAINED_TOOL_STREAMING_BETA]
        );
    }

    #[test]
    fn interleaved_thinking_beta_is_sent_regardless_of_this_turns_own_thinking_flag() {
        // pi-parity fix (Task #29): pi sends the interleaved-thinking beta by default for every
        // request to a non-adaptive-thinking model (`interleavedThinking ?? true`), gated only on the
        // model's own thinking *shape* (`model.compat?.forceAdaptiveThinking !== true`) — not on
        // whether this particular turn requested thinking. `claude-sonnet-4-5` is `Budget`-shape
        // (non-Adaptive) — must get the header even with no `thinking` on this turn.
        assert!(needs_interleaved_thinking_beta("claude-sonnet-4-5"));
        // `Adaptive`-shape models (our own default, `claude-opus-4-8`) interleave by default — sending
        // the opt-in would be a no-op, so it's skipped.
        assert!(!needs_interleaved_thinking_beta("claude-opus-4-8"));
    }

    #[test]
    fn retryable_status_classification() {
        for s in [429, 500, 502, 503, 504, 524, 529] {
            assert!(is_retryable_status(s), "{s} should be retryable");
        }
        for s in [200, 400, 401, 403, 404, 408, 409, 422] {
            assert!(!is_retryable_status(s), "{s} should not be retryable");
        }
    }

    #[test]
    fn retryable_status_excludes_408_and_409() {
        // pi-parity fix: pi's own retry classifiers never treat 408 (Request Timeout) or 409
        // (Conflict) as retryable — there's no pi source for either — and including them
        // contradicted this function's own "a 4xx other than 429 is the caller's fault" doc comment.
        // Also restores consistency with `agent::retry::WHOLE_RUN_RETRYABLE_STATUS_DIGITS`, which
        // never included 408/409 either.
        assert!(!is_retryable_status(408));
        assert!(!is_retryable_status(409));
    }

    #[test]
    fn cloudflare_524_is_retryable() {
        // Task #16 (pi-parity): pi's `packages/ai/src/utils/retry.ts:36` treats "524" as its only
        // retry signal for this status — a Cloudflare-fronted origin that timed out responding, the
        // same transient-infra class as the other 5xx entries already covered.
        assert!(is_retryable_status(524));
    }

    #[test]
    fn quota_exhaustion_classification() {
        for body in [
            r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#,
            r#"{"error":"quota exceeded for this billing period"}"#,
            r#"{"error":"please add a payment method — billing required"}"#,
            r#"{"error":"you are out of budget for this key"}"#,
        ] {
            assert!(
                is_quota_exhausted(body),
                "should classify as quota exhaustion: {body}"
            );
        }
        for body in [
            r#"{"error":{"type":"rate_limit_error","message":"Too many requests, please slow down"}}"#,
            "",
            "gateway timeout",
        ] {
            assert!(
                !is_quota_exhausted(body),
                "should NOT classify as quota exhaustion: {body}"
            );
        }
    }

    #[test]
    fn quota_exhaustion_classification_covers_codex_usage_cap_phrases() {
        // pi-parity (pass 17): pi's Codex-specific `isTerminalRateLimitError`
        // (`openai-codex-responses.ts:114-118`) also recognizes these ChatGPT/Codex usage-cap phrases,
        // which previously fell through to ordinary rate-limit handling (full exponential backoff)
        // instead of failing fast.
        for body in [
            r#"{"error":{"type":"GoUsageLimitError","message":"you're out for today"}}"#,
            r#"{"error":{"type":"FreeUsageLimitError","message":"free tier exhausted"}}"#,
            r#"{"error":"Monthly usage limit reached for this account"}"#,
            r#"{"error":"insufficient available balance to complete this request"}"#,
        ] {
            assert!(
                is_quota_exhausted(body),
                "should classify as quota exhaustion: {body}"
            );
        }
    }

    #[test]
    fn codex_friendly_usage_limit_message_matches_pis_parse_error_response() {
        // pi-parity (pass 17, Task 5): pi's Codex-specific `parseErrorResponse`
        // (`openai-codex-responses.ts:1459-1484`) turns this raw shape into a friendly sentence rather
        // than surfacing the bare JSON.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        // +20s of slack so the function's own (slightly later) `now` still rounds to the same 45
        // minutes — avoids flaking right at a rounding boundary.
        let resets_at = now + 45 * 60 + 20;
        let body = format!(
            r#"{{"error":{{"code":"usage_limit_reached","plan_type":"Plus","resets_at":{resets_at}}}}}"#
        );
        let msg = codex_friendly_usage_limit_message(&body).expect("should match");
        assert_eq!(
            msg,
            "You have hit your ChatGPT usage limit (plus plan). Try again in ~45 min."
        );

        // No `plan_type`/`resets_at` at all → still a friendly message, just without those clauses.
        let body = r#"{"error":{"code":"rate_limit_exceeded"}}"#;
        assert_eq!(
            codex_friendly_usage_limit_message(body).unwrap(),
            "You have hit your ChatGPT usage limit."
        );

        // `type` is accepted as a fallback for `code`.
        let body = r#"{"error":{"type":"usage_not_included"}}"#;
        assert_eq!(
            codex_friendly_usage_limit_message(body).unwrap(),
            "You have hit your ChatGPT usage limit."
        );

        // A quota body that ISN'T this specific Codex shape (e.g. plain OpenAI billing, or unparseable
        // text) must not get rewritten — every other provider's raw detail stays exactly as-is.
        assert!(
            codex_friendly_usage_limit_message(
                r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#
            )
            .is_none()
        );
        assert!(codex_friendly_usage_limit_message("not json at all").is_none());
        assert!(
            codex_friendly_usage_limit_message(r#"{"error":"a bare string, no code field"}"#)
                .is_none()
        );
    }

    #[test]
    fn codex_friendly_usage_limit_message_does_not_panic_on_an_out_of_range_resets_at() {
        // A provider/proxy can send an arbitrary `resets_at` — subtracting it from "now" with a raw
        // `-` would panic on underflow (`overflow-checks = true` in the release profile) for a value
        // this far out of range; `saturating_sub` must be used instead.
        let body = r#"{"error":{"code":"usage_limit_reached","resets_at":-9223372036854775808}}"#;
        let msg = codex_friendly_usage_limit_message(body).expect("should still match the shape");
        assert!(msg.starts_with("You have hit your ChatGPT usage limit."));
    }

    /// Asserts `actual` falls within the +/-20% jitter band around `nominal` (the unjittered
    /// exponential value), inclusive. Used everywhere a test previously asserted an exact backoff
    /// value — jitter (added as a pi-parity/consistency fix so this inner retry layer doesn't
    /// thundering-herd the same way the outer whole-run layer's own jitter already guards against)
    /// means the exact value is no longer deterministic, only its range is.
    fn assert_within_jitter(actual: Duration, nominal: Duration) {
        let lo = nominal.mul_f64(0.8);
        let hi = nominal.mul_f64(1.2);
        assert!(
            actual >= lo && actual <= hi,
            "expected {actual:?} within +/-20% of {nominal:?} (i.e. [{lo:?}, {hi:?}])"
        );
    }

    #[test]
    fn backoff_is_exponential_and_capped() {
        assert_within_jitter(backoff(0, None, BASE_BACKOFF, MAX_BACKOFF), BASE_BACKOFF);
        assert_within_jitter(
            backoff(1, None, BASE_BACKOFF, MAX_BACKOFF),
            BASE_BACKOFF * 2,
        );
        assert_within_jitter(
            backoff(2, None, BASE_BACKOFF, MAX_BACKOFF),
            BASE_BACKOFF * 4,
        );
        assert_eq!(backoff(20, None, BASE_BACKOFF, MAX_BACKOFF), MAX_BACKOFF); // saturates, never overflows
        // A server hint wins when larger, but is still capped. The hint itself is never jittered —
        // both hints here are far outside the unjittered exponential's +/-20% band, so the result
        // stays exact regardless of jitter.
        assert_eq!(
            backoff(0, Some(Duration::from_secs(2)), BASE_BACKOFF, MAX_BACKOFF),
            Duration::from_secs(2)
        );
        assert_eq!(
            backoff(
                0,
                Some(Duration::from_secs(3600)),
                BASE_BACKOFF,
                MAX_BACKOFF
            ),
            MAX_BACKOFF
        );
    }

    #[test]
    fn backoff_jitter_varies_across_calls_but_stays_in_range() {
        // Finding #32 (pi-parity/consistency fix): proves jitter is actually applied, not just
        // structurally present — the same attempt number, computed repeatedly, must not always
        // produce the identical duration (the thundering-herd scenario this fix closes), while every
        // value still lands within the documented +/-20% band around the unjittered exponential value.
        let nominal = BASE_BACKOFF * 4; // attempt 2's unjittered value.
        let samples: Vec<_> = (0..50)
            .map(|_| backoff(2, None, BASE_BACKOFF, MAX_BACKOFF))
            .collect();
        for &s in &samples {
            assert_within_jitter(s, nominal);
        }
        assert!(
            samples.windows(2).any(|w| w[0] != w[1]),
            "expected varying backoff durations across repeated calls for the same attempt, got \
             identical values every time: {samples:?}"
        );
    }

    #[test]
    fn backoff_honors_a_custom_base() {
        let custom = Duration::from_millis(1000);
        assert_within_jitter(backoff(0, None, custom, MAX_BACKOFF), custom);
        assert_within_jitter(backoff(1, None, custom, MAX_BACKOFF), custom * 2);
        assert_eq!(backoff(20, None, custom, MAX_BACKOFF), MAX_BACKOFF); // still saturates at the shared cap
    }

    #[test]
    fn backoff_honors_a_custom_max_backoff_ceiling() {
        // Task #21 (pi-parity): the ceiling itself must be an overridable knob, not just the base —
        // a caller who wants a *tighter* cap than the new 60s default (e.g. reverting to the old 10s
        // behavior) must be able to get it via `GatewayClient::with_max_backoff`.
        let tight = Duration::from_secs(10);
        assert_within_jitter(backoff(0, None, BASE_BACKOFF, tight), BASE_BACKOFF);
        assert_eq!(backoff(20, None, BASE_BACKOFF, tight), tight);
        assert_eq!(
            backoff(0, Some(Duration::from_secs(3600)), BASE_BACKOFF, tight),
            tight
        );
    }

    #[test]
    fn max_backoff_default_is_raised_toward_pis_own_default() {
        // Task #21 (pi-parity): raised from an earlier 10s ceiling toward pi's own 60s default
        // (`openai-codex-responses.ts`'s `DEFAULT_MAX_RETRY_DELAY_MS`) — a 429 with a `Retry-After: 30`
        // hint used to get retried back into the very rate-limit window it named.
        assert_eq!(MAX_BACKOFF, Duration::from_secs(60));
    }

    #[test]
    fn truncate_error_body_caps_a_large_upstream_error_page() {
        let short = "gateway timeout";
        assert_eq!(truncate_error_body(short), short);

        // pi-parity fix: the omitted-character count must be included, matching pi's
        // `truncateErrorText` (`error-body.ts:115-118`), not a bare "[truncated]" marker that leaves
        // an on-call engineer unable to tell a 1-char overflow from a multi-megabyte error page.
        let huge = "x".repeat(MAX_ERROR_BODY_CHARS + 500);
        let truncated = truncate_error_body(&huge);
        assert!(
            truncated.ends_with("... [truncated 500 chars]"),
            "got: {truncated}"
        );
        assert_eq!(
            truncated.chars().count(),
            MAX_ERROR_BODY_CHARS + "... [truncated 500 chars]".chars().count()
        );
    }

    #[test]
    fn retry_after_accepts_delta_seconds_and_http_date() {
        // Delta-seconds form, capped at MAX_BACKOFF.
        assert_eq!(
            parse_retry_after("2", MAX_BACKOFF),
            Some(Duration::from_secs(2))
        );
        assert_eq!(
            parse_retry_after(" 3 ", MAX_BACKOFF),
            Some(Duration::from_secs(3))
        );
        assert_eq!(parse_retry_after("99999", MAX_BACKOFF), Some(MAX_BACKOFF));
        // A value that is neither an integer nor an HTTP-date is ignored.
        assert_eq!(parse_retry_after("soon", MAX_BACKOFF), None);
        // HTTP-date already in the past → no extra wait.
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", MAX_BACKOFF),
            Some(Duration::ZERO)
        );
        // HTTP-date in the future → a positive, capped delay.
        let future = std::time::SystemTime::now() + Duration::from_secs(5);
        let delay = parse_retry_after(&httpdate::fmt_http_date(future), MAX_BACKOFF)
            .expect("a parsed delay");
        assert!(
            delay > Duration::ZERO && delay <= MAX_BACKOFF,
            "future http-date should yield a bounded positive delay, got {delay:?}"
        );
    }

    #[test]
    fn parse_retry_after_ms_accepts_milliseconds_and_ignores_garbage() {
        // pi-parity (pass 17): pi's Codex-specific `getRetryAfterDelayMs`
        // (`openai-codex-responses.ts:130-137`) checks a non-standard `retry-after-ms` header before
        // falling back to the standard `Retry-After` (seconds/HTTP-date) one — applied generically here
        // since beyond routes Codex through the same retry path as every other provider.
        assert_eq!(
            parse_retry_after_ms("1500", MAX_BACKOFF),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(
            parse_retry_after_ms(" 250 ", MAX_BACKOFF),
            Some(Duration::from_millis(250))
        );
        // A fractional value is accepted too — JS `Number.isFinite` allows floats same as `f64::parse`.
        assert_eq!(
            parse_retry_after_ms("2500.5", MAX_BACKOFF),
            Some(Duration::from_secs_f64(2.5005))
        );
        // A negative value clamps up to zero rather than yielding a negative/underflowing duration.
        assert_eq!(
            parse_retry_after_ms("-100", MAX_BACKOFF),
            Some(Duration::ZERO)
        );
        // Capped at the caller's `max_backoff`, same as the seconds/date form.
        assert_eq!(
            parse_retry_after_ms("999999999", MAX_BACKOFF),
            Some(MAX_BACKOFF)
        );
        // Not a number at all → ignored, letting the caller fall back to the standard header.
        assert_eq!(parse_retry_after_ms("soon", MAX_BACKOFF), None);
        assert_eq!(parse_retry_after_ms("", MAX_BACKOFF), None);
        // Non-finite (`NaN`/`Infinity` both parse as valid `f64` but must still be rejected).
        assert_eq!(parse_retry_after_ms("NaN", MAX_BACKOFF), None);
        assert_eq!(parse_retry_after_ms("inf", MAX_BACKOFF), None);
    }

    #[test]
    fn parse_retry_after_ms_does_not_panic_on_an_astronomically_large_value() {
        // `Duration::from_secs_f64` panics if the seconds value doesn't fit a `Duration` — a header
        // carrying a value like `1e25` (finite, so it passes the `is_finite` check) must be clamped to
        // `max_backoff` *before* the conversion, not after.
        assert_eq!(parse_retry_after_ms("1e25", MAX_BACKOFF), Some(MAX_BACKOFF));
        assert_eq!(
            parse_retry_after_ms(&f64::MAX.to_string(), MAX_BACKOFF),
            Some(MAX_BACKOFF)
        );
    }

    #[test]
    fn parse_retry_after_honors_a_custom_max_backoff_cap() {
        // A `Retry-After` hint larger than a caller's custom (tighter) ceiling must still be capped to
        // that ceiling, not the module default.
        let tight = Duration::from_secs(5);
        assert_eq!(parse_retry_after("999", tight), Some(tight));
    }

    /// A source that always yields a fixed OAuth-flagged credential — the test double for
    /// [`CredentialSource`] exercising the `is_oauth` header seam without a real credential store.
    struct OauthTestCredential;
    #[async_trait]
    impl CredentialSource for OauthTestCredential {
        async fn credential(&self) -> Result<Credential> {
            Ok(Credential::new("test-oauth-token", true))
        }
    }

    #[tokio::test]
    async fn oauth_credential_adds_anthropics_identity_headers() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            // The test only inspects the request headers the client sent — an empty
            // close-delimited response is enough to let `stream()` complete without needing a real
            // decodable event.
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let client = GatewayClient::with_credential_source(
            format!("http://{addr}"),
            Arc::new(OauthTestCredential),
        )
        .unwrap();
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {} // drain to completion; content doesn't matter here

        let request = server.join().unwrap().to_lowercase();
        assert!(
            request.contains("anthropic-beta: claude-code-20250219,oauth-2025-04-20"),
            "missing/wrong anthropic-beta header, got:\n{request}"
        );
        assert!(
            request.contains("user-agent: claude-cli/"),
            "missing claude-cli user-agent, got:\n{request}"
        );
        assert!(
            request.contains("x-app: cli"),
            "missing x-app header, got:\n{request}"
        );
        assert!(
            request.contains("anthropic-dangerous-direct-browser-access: true"),
            "missing direct-browser-access header, got:\n{request}"
        );
    }

    #[tokio::test]
    async fn a_static_credential_never_sends_oauth_headers() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap().to_lowercase();
        assert!(!request.contains("claude-code-20250219"), "got:\n{request}");
        assert!(!request.contains("oauth-2025-04-20"), "got:\n{request}");
        assert!(!request.contains("x-app:"), "got:\n{request}");
        assert!(
            !request.contains("user-agent: claude-cli/"),
            "got:\n{request}"
        );
        // pi sends this header in every Anthropic auth branch, including plain API-key auth
        // (`anthropic-messages.ts`'s `createClient`) — present here too, unlike the OAuth-only
        // identity headers asserted absent above.
        assert!(
            request.contains("anthropic-dangerous-direct-browser-access: true"),
            "got:\n{request}"
        );
    }

    /// A real TCP peer that answers with valid SSE headers plus one partial event, then vanishes
    /// without a clean shutdown — what an abrupt connection reset or a crashed upstream looks like on
    /// the wire, as opposed to anything this crate constructs itself.
    #[tokio::test]
    async fn a_connection_dropped_mid_body_is_tagged_as_a_mid_stream_network_error() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            // Drain the request so the response write doesn't race a half-closed read side.
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            // `Content-Length` promises far more body than actually arrives before the connection
            // vanishes — unlike a close-delimited (`Connection: close`, no length) body, where EOF
            // *is* the defined end-of-body and reqwest would read this as a normal, if short, success.
            // A length mismatch is what makes the abrupt close an actual framing violation reqwest
            // surfaces as a body-read error, matching what a mid-response connection reset looks like
            // against a real chunked/length-bearing gateway response.
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100000\r\n\r\n\
                      data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
            // No `message_stop`, no clean FIN — the connection just disappears mid-response.
            drop(stream);
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap(); // the request itself succeeds
        let mut tagged = false;
        while let Some(ev) = events.next().await {
            if let Err(Error::Transport(msg)) = ev {
                tagged = msg.contains(MID_STREAM_NETWORK_ERROR);
                break;
            }
        }
        server.join().unwrap();
        assert!(
            tagged,
            "a mid-body connection drop must surface as a MID_STREAM_NETWORK_ERROR-tagged transport error"
        );
    }

    /// A real TCP peer that answers headers plus one partial event, then goes silent (holds the
    /// connection open, sends nothing) well past a deliberately shrunk idle-read timeout — proves
    /// [`GatewayClient::with_idle_timeout`] actually reaches the underlying `reqwest::Client` rather
    /// than being a dead field, since the default [`READ_TIMEOUT`] (600s) would never trip in a test.
    #[tokio::test]
    async fn with_idle_timeout_overrides_the_default_read_timeout() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100000\r\n\r\n\
                      data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
            // Silent for longer than the client's configured idle timeout, but the test itself
            // doesn't wait this long — the client errors out well before this elapses.
            std::thread::sleep(Duration::from_millis(500));
            let _ = stream.shutdown(std::net::Shutdown::Both);
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_idle_timeout(Duration::from_millis(50))
            .unwrap();
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap(); // the request itself succeeds
        let mut tagged = false;
        while let Some(ev) = events.next().await {
            if let Err(Error::Transport(msg)) = ev {
                tagged = msg.contains(MID_STREAM_NETWORK_ERROR);
                break;
            }
        }
        server.join().unwrap();
        assert!(
            tagged,
            "a read past the configured idle timeout must surface as a mid-stream network error"
        );
    }

    /// Task #11 (pi-parity): `with_extra_headers` must merge onto every outgoing request, and win over
    /// a header this client would otherwise set itself for the same name (an operator overriding
    /// `anthropic-version` for a self-hosted/proxied endpoint that speaks a different wire version).
    #[tokio::test]
    async fn extra_headers_are_merged_onto_the_request_and_win_on_a_name_collision() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let captured_clone = captured.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            *captured_clone.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
        });

        let mut extra = HashMap::new();
        extra.insert("anthropic-version".to_string(), "2099-01-01".to_string());
        extra.insert("x-operator-header".to_string(), "custom-value".to_string());
        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_extra_headers(extra);
        // Anthropic dialect, so this client would otherwise send `anthropic-version: 2023-06-01` —
        // the operator's own value must win instead.
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}
        server.join().unwrap();

        let request = captured.lock().unwrap().clone();
        let lower = request.to_lowercase();
        assert!(
            lower.contains("anthropic-version: 2099-01-01"),
            "operator-configured header must win over the client's own default, got:\n{request}"
        );
        assert!(
            !lower.contains("anthropic-version: 2023-06-01"),
            "the client's own default value must not also be present, got:\n{request}"
        );
        assert!(
            lower.contains("x-operator-header: custom-value"),
            "a brand-new operator header must also be attached, got:\n{request}"
        );
    }

    /// Task #14 (pi-parity): a hook installed via `with_hooks` must have `after_provider_response`
    /// called with the real response's status and headers, once they're known — proves the wiring
    /// reaches all the way from `GatewayClient::stream` through to the hook, not just that the trait
    /// method exists.
    #[tokio::test]
    async fn after_provider_response_hook_fires_with_the_real_status_and_headers() {
        type SeenResponse = (u16, Vec<(String, String)>);
        struct RecordsResponse {
            seen: std::sync::Mutex<Option<SeenResponse>>,
        }
        #[async_trait]
        impl AgentHooks for RecordsResponse {
            async fn after_provider_response(&self, status: u16, headers: &[(String, String)]) {
                *self.seen.lock().unwrap() = Some((status, headers.to_vec()));
            }
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Probe: yes\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
            stream.flush().unwrap();
        });

        let hooks = Arc::new(RecordsResponse {
            seen: std::sync::Mutex::new(None),
        });
        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_hooks(hooks.clone());
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}
        server.join().unwrap();

        let (status, headers) = hooks.seen.lock().unwrap().clone().expect(
            "after_provider_response must have fired once the response's status/headers were known",
        );
        assert_eq!(status, 200);
        assert!(
            headers
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("x-probe") && value == "yes"),
            "expected the real response's own headers to reach the hook, got: {headers:?}"
        );
    }

    /// Pi-parity fix: nothing previously exposed the literal, dialect-built wire JSON to a hook — only
    /// the abstract, pre-dialect `ModelRequest` via `before_provider_request` (one layer up, before
    /// `dialect.build_body` even runs). Proves `before_provider_payload` reaches all the way from
    /// `GatewayClient::stream` to the *actual bytes* landing on the wire, not just that the trait method
    /// exists — mirrors the `after_provider_response` test above, but on the request side.
    #[tokio::test]
    async fn before_provider_payload_hook_rewrites_the_literal_wire_body_sent_over_http() {
        struct InjectsPayloadMarker;
        #[async_trait]
        impl AgentHooks for InjectsPayloadMarker {
            async fn before_provider_payload(&self, payload: &mut Value) {
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert(
                        "_wire_marker".to_string(),
                        serde_json::json!("injected-by-hook"),
                    );
                }
            }
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            *captured_clone.lock().unwrap() = buf[..n].to_vec();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
        });

        let hooks = Arc::new(InjectsPayloadMarker);
        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_hooks(hooks);
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}
        server.join().unwrap();

        let request = String::from_utf8_lossy(&captured.lock().unwrap()).to_string();
        assert!(
            request.contains("_wire_marker") && request.contains("injected-by-hook"),
            "expected the hook's mutation to reach the literal wire body actually sent over HTTP, got:\n{request}"
        );
    }

    /// A panicking `before_provider_payload` hook must not corrupt or drop the request — same
    /// "fails open" convention `before_provider_request`'s own panic-safety test uses. The payload sent
    /// over the wire must be exactly what `build_body` produced, with none of the panicking hook's
    /// partial mutation.
    #[tokio::test]
    async fn a_panicking_before_provider_payload_hook_keeps_the_original_body() {
        struct AlwaysPanics;
        #[async_trait]
        impl AgentHooks for AlwaysPanics {
            async fn before_provider_payload(&self, payload: &mut Value) {
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert("_partial".to_string(), serde_json::json!(true));
                }
                panic!("boom: before_provider_payload always panics");
            }
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            *captured_clone.lock().unwrap() = buf[..n].to_vec();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
        });

        let hooks = Arc::new(AlwaysPanics);
        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_hooks(hooks);
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}
        server.join().unwrap();

        let request = String::from_utf8_lossy(&captured.lock().unwrap()).to_string();
        assert!(
            !request.contains("_partial"),
            "a panicking hook's partial mutation must never reach the wire, got:\n{request}"
        );
    }

    /// A panicking `after_provider_response` hook must not unwind through the in-flight stream it's
    /// only supposed to be observing — same "fails open" convention every other hook call site uses
    /// (see the sibling `before_provider_payload` test above). The response must still decode to
    /// completion; only the hook's own panic is contained.
    #[tokio::test]
    async fn a_panicking_after_provider_response_hook_does_not_crash_the_stream() {
        struct AlwaysPanics;
        #[async_trait]
        impl AgentHooks for AlwaysPanics {
            async fn after_provider_response(&self, _status: u16, _headers: &[(String, String)]) {
                panic!("boom: after_provider_response always panics");
            }
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf).unwrap_or(0);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
                      event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n\
                      event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
        });

        let hooks = Arc::new(AlwaysPanics);
        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_hooks(hooks);
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        let mut saw_event = false;
        while let Some(ev) = events.next().await {
            assert!(ev.is_ok(), "unexpected stream error: {ev:?}");
            saw_event = true;
        }
        server.join().unwrap();

        assert!(
            saw_event,
            "the stream must still decode to completion despite the panicking hook"
        );
    }

    /// A real TCP peer that answers with valid SSE headers plus one partial event, then closes the
    /// connection *cleanly* — `Connection: close`, no `Content-Length` mismatch, so EOF is the defined
    /// end of a close-delimited body, not a framing violation (contrast the sibling test above). This is
    /// the wire-level analog of pi's mocked wrapper stream that simply stops yielding events with no
    /// error (`packages/ai/test/openai-responses-terminal-event.test.ts:167-186`, "emits an error final
    /// result when the wrapper stream ends before a terminal response event"). Proves the decoder's own
    /// "ended before…" rejection reaches the caller through the *full* client pipeline — not just the
    /// buffered-decoder unit tests (`dialect::anthropic`'s own `finish()` tests) — and that a clean
    /// close is never mistaken for (or tagged as) a [`MID_STREAM_NETWORK_ERROR`], since nothing about
    /// this shutdown was actually a transport fault.
    #[tokio::test]
    async fn a_clean_early_close_before_the_terminal_event_is_not_tagged_as_a_network_error() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
                      data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
            // A graceful shutdown — no `message_stop`, but nothing about the close itself is abnormal.
            let _ = stream.shutdown(std::net::Shutdown::Both);
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap(); // the request itself succeeds
        let mut last_err = None;
        while let Some(ev) = events.next().await {
            if let Err(Error::Transport(msg)) = ev {
                last_err = Some(msg);
                break;
            }
        }
        server.join().unwrap();
        let msg = last_err.expect("a clean early close must still surface a transport error");
        assert!(
            msg.contains("Anthropic stream ended before message_stop"),
            "expected the decoder's own terminal-event rejection, got: {msg}"
        );
        assert!(
            !msg.contains(MID_STREAM_NETWORK_ERROR),
            "a clean, close-delimited EOF is not a network fault and must not carry the \
             MID_STREAM_NETWORK_ERROR tag: {msg}"
        );
    }

    /// A real TCP peer that always answers 429 with a quota-exhaustion body — proves the client fails
    /// fast (one request, no retry) instead of burning its retry budget on a rejection retrying can
    /// never fix.
    #[tokio::test]
    async fn a_429_with_quota_exhaustion_body_is_not_retried() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = request_count.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut stream, _)) = listener.accept() {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let body = r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#;
                let resp = format!(
                    "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_retry(3, Duration::from_millis(10));
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        let first = events.next().await;
        server.join().unwrap();

        assert!(
            matches!(first, Some(Err(Error::Transport(_)))),
            "expected a transport error, got {first:?}"
        );
        assert_eq!(
            request_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a quota-exhausted 429 must not be retried"
        );
    }

    /// A real TCP peer that answers 429 with an ordinary rate-limit body (no quota/billing phrase) on
    /// the first request, then succeeds on the retry — proves ordinary rate limiting still gets the
    /// normal retry treatment `is_quota_exhausted` doesn't touch.
    #[tokio::test]
    async fn a_429_with_a_plain_rate_limit_body_is_retried() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = request_count.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            for attempt in 0..2 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                if attempt == 0 {
                    let body =
                        r#"{"error":{"type":"rate_limit_error","message":"Too many requests"}}"#;
                    let resp = format!(
                        "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else {
                    let sse = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\ndata: {\"type\":\"message_stop\"}\n\n";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        sse.len(),
                        sse
                    );
                    let _ = stream.write_all(resp.as_bytes());
                }
            }
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_retry(3, Duration::from_millis(10));
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        let _ = events.next().await; // drive the stream far enough to trigger the retry + second request
        server.join().unwrap();

        assert_eq!(
            request_count.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "an ordinary rate-limit 429 must still be retried"
        );
    }

    /// A real TCP peer captures the raw request bytes so the test can inspect headers directly — proves
    /// pi's session-affinity headers (`session_id`/`x-client-request-id`/`x-session-affinity`) go out on
    /// a Chat Completions dialect request too, not just OpenAI Responses — but **only for a
    /// Fireworks-hosted model** (pi-parity, Task D): pi's own `compat.sendSessionAffinityHeaders` is a
    /// per-provider catalogue flag, true only for Fireworks (and Cloudflare-related catalogues,
    /// deliberately out of scope), never for every OpenAI-wire provider indiscriminately — see the
    /// sibling `session_affinity_headers_are_absent_for_a_non_fireworks_chat_completions_request` test
    /// below for the negative case this fix actually closed. Matches `openai-completions.ts`'s
    /// `compat.sendSessionAffinityHeaders` branch, exercised by
    /// `packages/ai/test/openai-completions-prompt-cache.test.ts`'s "sends known session-affinity
    /// headers when compat.sendSessionAffinityHeaders is enabled" case.
    #[tokio::test]
    async fn session_affinity_headers_are_sent_for_a_fireworks_chat_completions_request() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let captured_clone = captured.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            *captured_clone.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
            let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                sse.len(),
                sse
            );
            let _ = stream.write_all(resp.as_bytes());
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        // Fireworks' own "glm-5p2" is the one Fireworks id that stays on the Chat Completions dialect
        // (`Dialect::for_model`'s `is_fireworks_anthropic_wire_model` routes every other current
        // Fireworks id to the Anthropic dialect instead).
        let req = ModelRequest::new("accounts/fireworks/models/glm-5p2", Vec::new(), 100)
            .with_cache_key("session-affinity-test");
        let mut events = client.stream(req).await.unwrap();
        let _ = events.next().await;
        server.join().unwrap();

        let request = captured.lock().unwrap().clone();
        assert!(
            request.contains("session_id: session-affinity-test"),
            "expected a `session_id` header on a Fireworks Chat Completions request, got:\n{request}"
        );
        assert!(
            request.contains("x-client-request-id: session-affinity-test"),
            "expected an `x-client-request-id` header on a Fireworks Chat Completions request, got:\n{request}"
        );
        assert!(
            request.contains("x-session-affinity: session-affinity-test"),
            "expected an `x-session-affinity` header on a Fireworks Chat Completions request, got:\n{request}"
        );
    }

    /// pi-parity (models/dialects pass, Task D): the headers must be gated to Fireworks specifically —
    /// a plain OpenAI-compatible provider (Groq, here) must never get them, even with an otherwise
    /// identical cache key and no `no_cache` opt-out. Before this fix, every Chat-Completions-routed
    /// provider got these headers indiscriminately.
    #[tokio::test]
    async fn session_affinity_headers_are_absent_for_a_non_fireworks_chat_completions_request() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let captured_clone = captured.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            *captured_clone.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
            let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                sse.len(),
                sse
            );
            let _ = stream.write_all(resp.as_bytes());
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        // `llama-3.1-70b` resolves to `Dialect::OpenAi` (Chat Completions) — a third-party
        // OpenAI-compatible provider (Groq), not Fireworks.
        let req = ModelRequest::new("llama-3.1-70b", Vec::new(), 100)
            .with_cache_key("session-affinity-test");
        let mut events = client.stream(req).await.unwrap();
        let _ = events.next().await;
        server.join().unwrap();

        let request = captured.lock().unwrap().clone();
        assert!(
            !request.contains("session_id:"),
            "expected no `session_id` header on a non-Fireworks request, got:\n{request}"
        );
        assert!(
            !request.contains("x-client-request-id:"),
            "expected no `x-client-request-id` header on a non-Fireworks request, got:\n{request}"
        );
        assert!(
            !request.contains("x-session-affinity:"),
            "expected no `x-session-affinity` header on a non-Fireworks request, got:\n{request}"
        );
    }

    /// Pi-parity audit task #40: `no_cache` must suppress the session-affinity headers too, not just
    /// the cache-control blocks — otherwise a request that explicitly opted out of prompt caching still
    /// pins itself to a specific backend session, defeating the purpose of the opt-out. Matches pi's
    /// `compat.sendSessionAffinityHeaders` gating, which never fires when caching is disabled for the
    /// request. Uses a Fireworks-hosted id so this test proves genuine `no_cache` suppression on an
    /// otherwise-eligible route, not just "a non-Fireworks route never gets them anyway" (see the
    /// dedicated negative test above for that).
    #[tokio::test]
    async fn session_affinity_headers_are_suppressed_when_no_cache_is_set() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let captured_clone = captured.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            *captured_clone.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
            let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                sse.len(),
                sse
            );
            let _ = stream.write_all(resp.as_bytes());
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new("accounts/fireworks/models/glm-5p2", Vec::new(), 100)
            .with_cache_key("session-affinity-test")
            .with_no_cache(true);
        let mut events = client.stream(req).await.unwrap();
        let _ = events.next().await;
        server.join().unwrap();

        let request = captured.lock().unwrap().clone();
        assert!(
            !request.contains("session_id:"),
            "expected no `session_id` header when no_cache is set, got:\n{request}"
        );
        assert!(
            !request.contains("x-client-request-id:"),
            "expected no `x-client-request-id` header when no_cache is set, got:\n{request}"
        );
        assert!(
            !request.contains("x-session-affinity:"),
            "expected no `x-session-affinity` header when no_cache is set, got:\n{request}"
        );
    }

    /// Pi-parity audit "cluster deep-dive" pass flagged: cancelling a turn while a `tool_use`'s
    /// arguments are still mid-stream leaves that call orphaned (no matching `tool_result`) in the
    /// persisted session, and — since `repair_cancelled_dispatch` only runs when dispatch itself starts
    /// — the audit's claim was "no repair happens here at all". That claim missed that
    /// `repair_orphaned_tool_use` runs generically on *every* real request through `GatewayClient`,
    /// regardless of how the orphan was produced (see its doc comment: "this covers every other way one
    /// can reach a request"). Proves that end-to-end over the real wire path, not just the isolated
    /// `dialect::tests` unit coverage of `repair_orphaned_tool_use` itself: a session whose last message
    /// is exactly the shape `Accumulator::finish` produces for a tool call cut off mid-argument-stream
    /// (a `ToolUse` block with a best-effort/possibly-empty `input`, no following `tool_result`) must
    /// never reach the wire with that call still unanswered.
    #[tokio::test]
    async fn an_orphaned_tool_use_from_a_cancelled_mid_argument_stream_never_reaches_the_wire() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let captured_clone = captured.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            *captured_clone.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
            let sse = "data: {\"type\":\"message_stop\"}\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                sse.len(),
                sse
            );
            let _ = stream.write_all(resp.as_bytes());
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let messages = vec![
            crate::message::Message::user("edit the file"),
            // Exactly what a cancelled-mid-argument-stream tool call leaves behind: a `ToolUse` block
            // (empty `input`, matching `Accumulator::finish`'s fallback for unparseable/truncated
            // streamed JSON) with no following `tool_result` — the run was aborted before dispatch.
            crate::message::Message::assistant(vec![crate::message::ContentBlock::tool_use(
                "orphan-1",
                "edit",
                serde_json::json!({}),
            )]),
        ];
        let req = ModelRequest::new("claude-opus-4-8", messages, 100);
        let mut events = client.stream(req).await.unwrap();
        let _ = events.next().await;
        server.join().unwrap();

        let request = captured.lock().unwrap().clone();
        let body_start = request.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
        let body = &request[body_start..];
        assert!(
            body.contains("orphan-1") && body.contains("tool_result"),
            "expected a synthetic tool_result for the orphaned tool_use `orphan-1`, got body:\n{body}"
        );
        // The very shape that 400s if unrepaired: a `tool_use` with no immediately-following
        // `tool_result` anywhere in the outgoing body.
        assert!(
            !body.contains("\"content\":[]"),
            "no message should reach the wire with empty content either, got body:\n{body}"
        );
    }

    /// A fixed [`DirectRouting`]-carrying credential — the test double proving the routing mechanism
    /// itself (pi-parity fix: a Copilot/Codex-sourced credential previously always sent its bearer
    /// token to a bare gateway-default path, never its real upstream).
    struct DirectTestCredential(DirectRouting);

    #[async_trait]
    impl CredentialSource for DirectTestCredential {
        async fn credential(&self) -> Result<Credential> {
            Ok(Credential::new("test-oauth-token", true).with_direct_routing(self.0.clone()))
        }
    }

    /// Proves [`RouteOverride::Prefixed`] (the OpenAI-Codex shape): the request still goes to
    /// `GatewayClient::base_url` (through the gateway) but under the provider prefix, with Codex's
    /// static headers attached — and critically, *not* at the dialect's own bare default path
    /// (`/v1/responses`), which is exactly the bug this fix closes (a Codex-credentialed request
    /// silently landing on the gateway's default OpenAI provider instead of `chatgpt.com`).
    #[tokio::test]
    async fn prefixed_route_override_reaches_the_gateway_under_the_provider_prefix_with_static_headers()
     {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Prefixed {
                prefix: "/openai-codex",
                path: "/backend-api/codex/responses",
            },
            static_headers: vec![
                ("chatgpt-account-id", "acct_123".to_string()),
                ("originator", "beyond-ai-agent".to_string()),
                ("OpenAI-Beta", "responses=experimental".to_string()),
            ],
            copilot_dynamic_headers: false,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            format!("http://{addr}"),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap()
        // This test's one-shot mock server only ever accepts a single plain-HTTP connection and
        // asserts on the exact request line/headers it receives — it has nothing to do with the
        // Codex WebSocket transport (a separate, dedicated suite covers that in
        // `tests/codex_websocket_socket.rs`), so disable it here rather than let an eager WS connect
        // attempt consume this listener's only accepted connection before the real HTTP request does.
        .with_codex_websocket(false);
        // `gpt-5-codex` is `Dialect::OpenAiResponses` — its bare default path is `/v1/responses`,
        // which must NOT be what actually goes out.
        let req = ModelRequest::new("gpt-5-codex", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let request_line = request.lines().next().unwrap_or_default();
        assert_eq!(
            request_line, "POST /openai-codex/backend-api/codex/responses HTTP/1.1",
            "expected the prefixed+path-overridden request line, got:\n{request}"
        );
        assert!(
            !request.to_lowercase().contains("post /v1/responses"),
            "must not silently fall back to the dialect's own bare default path, got:\n{request}"
        );
        assert!(
            request.contains("chatgpt-account-id: acct_123"),
            "missing chatgpt-account-id header, got:\n{request}"
        );
        assert!(
            request.contains("originator: beyond-ai-agent"),
            "missing originator header, got:\n{request}"
        );
        assert!(
            request
                .to_lowercase()
                .contains("openai-beta: responses=experimental"),
            "missing OpenAI-Beta header, got:\n{request}"
        );
    }

    /// Proves [`RouteOverride::Direct`] (the GitHub Copilot shape): the request bypasses
    /// `GatewayClient::base_url` entirely (which here points at a closed port that would fail fast if
    /// ever actually dialed) and lands on the credential's own account-specific host, at Copilot's
    /// real path (`/chat/completions`, no `/v1` — distinct from the OpenAI dialect's own default
    /// `/v1/chat/completions`), carrying both Copilot's fixed editor-identity header and its per-turn
    /// dynamic `X-Initiator`/`Openai-Intent` headers — and, despite `is_oauth: true`, none of
    /// Anthropic's own OAuth identity-spoofing headers (this is a Copilot host, not Anthropic's).
    #[tokio::test]
    async fn direct_route_override_bypasses_the_gateway_base_url_and_attaches_copilot_headers() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/chat/completions",
            },
            static_headers: vec![("User-Agent", "GitHubCopilotChat/0.35.0".to_string())],
            copilot_dynamic_headers: true,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            // A closed port: if the client ever mistakenly used this instead of the credential's own
            // `Direct` base_url, the connection would fail fast (refused) rather than this test
            // hanging or flaking on a slow DNS timeout.
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        // `llama-3.1-70b` is `Dialect::OpenAi` (Chat Completions) — its bare default path is
        // `/v1/chat/completions`, which Copilot's real endpoint does not have.
        let messages = vec![crate::message::Message::user("hi")];
        let req = ModelRequest::new("llama-3.1-70b", messages, 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let request_line = request.lines().next().unwrap_or_default();
        assert_eq!(
            request_line, "POST /chat/completions HTTP/1.1",
            "expected Copilot's own path with no /v1 prefix, got:\n{request}"
        );
        let lower = request.to_lowercase();
        assert!(
            lower.contains("user-agent: githubcopilotchat"),
            "missing Copilot's fixed editor-identity header, got:\n{request}"
        );
        assert!(
            lower.contains("x-initiator: user"),
            "missing X-Initiator (no prior assistant/tool turn ⇒ user), got:\n{request}"
        );
        assert!(
            lower.contains("openai-intent: conversation-edits"),
            "missing Openai-Intent header, got:\n{request}"
        );
        assert!(
            !lower.contains("copilot-vision-request"),
            "no image in this turn — Copilot-Vision-Request must be absent, got:\n{request}"
        );
        assert!(
            !request.contains("claude-code-20250219") && !lower.contains("x-app: cli"),
            "an is_oauth Copilot credential must not carry Anthropic's own OAuth identity-spoofing \
             headers, got:\n{request}"
        );
    }

    /// A Claude-family model routed through a Copilot credential is `is_anthropic` but not
    /// `is_oauth` (see `has_direct_override`'s doc comment in `stream()`): it must still get
    /// `anthropic-dangerous-direct-browser-access` (pi sends that header in every Anthropic auth
    /// branch, Copilot included — Task 3, pi-parity), but none of Anthropic's own OAuth-only identity
    /// headers, which would misrepresent this as a direct-to-Anthropic Claude Code request instead of
    /// a Copilot-proxied one.
    #[tokio::test]
    async fn a_copilot_routed_claude_model_gets_the_browser_access_header_but_not_claude_code_identity()
     {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/v1/messages",
            },
            static_headers: vec![("User-Agent", "GitHubCopilotChat/0.35.0".to_string())],
            copilot_dynamic_headers: true,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new(
            "claude-opus-4-8",
            vec![crate::message::Message::user("hi")],
            100,
        );
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let lower = request.to_lowercase();
        assert!(
            lower.contains("anthropic-dangerous-direct-browser-access: true"),
            "got:\n{request}"
        );
        assert!(
            !request.contains("claude-code-20250219") && !lower.contains("oauth-2025-04-20"),
            "got:\n{request}"
        );
        assert!(!lower.contains("x-app: cli"), "got:\n{request}");
        assert!(
            !lower.contains("user-agent: claude-cli/"),
            "got:\n{request}"
        );
    }

    /// pi-parity (Task 2): `gpt-4.1` is `ApiKind::Responses` natively (`models::capabilities`), but
    /// GitHub Copilot's own proxy wants the older Chat Completions wire for that exact id
    /// (`github-copilot.models.ts`: `api: "openai-completions"`, vs `openai.models.ts`'s
    /// `"openai-responses"` for the same id natively). A Copilot-routed request for it must build a
    /// Chat Completions body (`"messages"`), never a Responses body (`"input"`).
    #[tokio::test]
    async fn a_copilot_routed_gpt_4_1_builds_a_chat_completions_body_not_a_responses_body() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/chat/completions",
            },
            static_headers: vec![("User-Agent", "GitHubCopilotChat/0.35.0".to_string())],
            copilot_dynamic_headers: true,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new("gpt-4.1", vec![crate::message::Message::user("hi")], 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "POST /chat/completions HTTP/1.1",
            "expected Copilot's own Chat Completions path, got:\n{request}"
        );
        assert!(
            request.contains("\"messages\":"),
            "expected a Chat Completions body shape, got:\n{request}"
        );
        assert!(
            !request.contains("\"input\":"),
            "must not build a Responses body shape for a Copilot-routed gpt-4.1, got:\n{request}"
        );
    }

    /// A non-Copilot request for the same id must be unaffected: `gpt-4.1` still resolves to the
    /// Responses dialect when reached through any route but Copilot's.
    #[tokio::test]
    async fn a_non_copilot_gpt_4_1_still_builds_a_responses_body() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new("gpt-4.1", vec![crate::message::Message::user("hi")], 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert!(
            request.contains("\"input\":"),
            "expected the native Responses body shape, got:\n{request}"
        );
    }

    /// pi-parity (pass 15): proves `GatewayClient::stream` actually sets [`ModelRequest::is_copilot`]
    /// on the live request — the dialect-side gate (`copilot_routed_gpt5_ids_never_get_an_explicit_
    /// reasoning_disable` in `dialect/openai_responses.rs`) only proves the body-building half; this
    /// proves the wiring from a real Copilot-routed credential through to that flag.
    #[tokio::test]
    async fn copilot_routed_gpt5_2_omits_the_explicit_reasoning_disable_end_to_end() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/responses",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: true,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new("gpt-5.2", vec![crate::message::Message::user("hi")], 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert!(
            !request.contains("\"reasoning\":{\"effort\":\"none\"}"),
            "Copilot-routed gpt-5.2 must not get an explicit reasoning-disable, got:\n{request}"
        );
    }

    /// pi-parity (pass 15): proves `GatewayClient::stream` actually sets [`ModelRequest::is_azure`] on
    /// the live request from the same `auth_header == "api-key"` signal
    /// `direct_route_with_custom_auth_header_sends_bare_key_and_omits_authorization` already exercises
    /// for the header shape — this proves it also suppresses `prompt_cache_retention` end to end.
    #[tokio::test]
    async fn azure_routed_request_omits_prompt_cache_retention_end_to_end() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}/openai/v1"),
                path: "/responses",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: Some("api-key".to_string()),
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new("gpt-4o", vec![crate::message::Message::user("hi")], 100)
            .with_cache_long(true);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert!(
            !request.contains("prompt_cache_retention"),
            "Azure-routed request with --cache-long must not send prompt_cache_retention, got:\n{request}"
        );
    }

    /// pi-parity (models/dialects pass, Task C): an Entra ID / Azure AD Bearer-token config sends a
    /// real Bearer token (`auth_header: None` — Bearer *is* correct there) but still names a
    /// `deployment_name`, unlike the static-API-key Azure shape the sibling test above exercises
    /// (`auth_header: Some("api-key")`). Before this fix, `req.is_azure` stayed `false` for this shape
    /// entirely, so `gpt-5.1` (one of the 7 ids Azure's own catalogue doesn't support an explicit
    /// reasoning-"off" wire value for) would incorrectly send the native `{"effort":"none"}` signal.
    #[tokio::test]
    async fn entra_id_shaped_config_with_only_deployment_name_still_suppresses_reasoning_disable() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}/openai/v1"),
                path: "/responses",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            // Entra ID's real shape: a Bearer token (no `auth_header` override) plus a deployment name
            // — the signal `is_azure` used to miss entirely.
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: Some("my-gpt-5-1-deployment".to_string()),
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        // gpt-5.1 is one of the 7 `NOT_DISABLEABLE_OFF_NATIVE_ROUTE` ids (`models.rs`) — natively
        // disable-capable, but Azure's own catalogue doesn't support an explicit "off" wire value.
        let req = ModelRequest::new("gpt-5.1", vec![crate::message::Message::user("hi")], 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert!(
            !request.contains("\"reasoning\""),
            "Entra-ID-routed gpt-5.1 must not get an explicit reasoning-disable signal at all, got:\n{request}"
        );
        // Sanity check: the Bearer header is still sent as normal (Entra ID's real auth scheme,
        // unaffected by the `is_azure` fix — only `DirectRouting::auth_header` controls that).
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer"),
            "Entra ID auth must still use a normal Bearer header, got:\n{request}"
        );
    }

    /// Proves the [`DirectRouting::auth_header`] mechanism (Task #8, pi-parity: Azure OpenAI
    /// routing support). A route that sets `auth_header: Some("api-key")` must send the credential's
    /// key verbatim in that header, with **no** `Authorization` header at all — Azure's
    /// `AzureOpenAI` SDK client authenticates this way, never via `Authorization: Bearer` (see
    /// `packages/ai/src/api/azure-openai-responses.ts` in pi-mono). Also bypasses the gateway
    /// entirely (`RouteOverride::Direct`, like Copilot) at the "v1"-unified Azure path shape
    /// (`{base_url}/responses`, base_url already carrying `/openai/v1`).
    #[tokio::test]
    async fn direct_route_with_custom_auth_header_sends_bare_key_and_omits_authorization() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}/openai/v1"),
                path: "/responses",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: Some("api-key".to_string()),
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            // A closed port, same guard as the Copilot test: if the client ever fell back to the
            // gateway's own base_url instead of this credential's `Direct` override, the connection
            // would fail fast rather than this test hanging.
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        // `gpt-4o-azure-deployment` is `Dialect::OpenAiResponses` (falls through to Chat Completions
        // only for non-native ids that aren't `ApiKind::Responses` — but any model id works here
        // since the path is fully overridden by `RouteOverride::Direct` regardless of dialect).
        let req = ModelRequest::new("gpt-4o", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let request_line = request.lines().next().unwrap_or_default();
        assert_eq!(
            request_line, "POST /openai/v1/responses HTTP/1.1",
            "expected Azure's v1-unified responses path, got:\n{request}"
        );
        let lower = request.to_lowercase();
        assert!(
            lower.contains("api-key: test-oauth-token"),
            "missing bare key in the api-key header, got:\n{request}"
        );
        assert!(
            !lower.contains("authorization:"),
            "Authorization must be entirely absent when auth_header overrides it, got:\n{request}"
        );
    }

    /// A Copilot turn whose last message is an assistant/tool follow-up (not a fresh user message)
    /// gets `X-Initiator: agent`; one carrying an image gets `Copilot-Vision-Request: true` — both
    /// computed fresh per turn from the request's own messages, matching pi's
    /// `inferCopilotInitiator`/`hasCopilotVisionInput`.
    #[tokio::test]
    async fn copilot_dynamic_headers_reflect_the_turns_own_messages() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/v1/messages",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: true,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let messages = vec![
            crate::message::Message::user("read this screenshot"),
            crate::message::Message::assistant(vec![crate::message::ContentBlock::text(
                "looking now",
            )]),
            crate::message::Message::user_with_images(
                "",
                vec![crate::message::ImageSource::base64("image/png", "QQ==")],
            ),
        ];
        let req = ModelRequest::new("claude-opus-4-8", messages, 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let lower = request.to_lowercase();
        // The last message is the image-carrying user turn, so this is still user-initiated.
        assert!(
            lower.contains("x-initiator: user"),
            "the last message is a fresh user turn, got:\n{request}"
        );
        assert!(
            lower.contains("copilot-vision-request: true"),
            "an Image content block in this turn must set Copilot-Vision-Request, got:\n{request}"
        );
    }

    /// HIGH pi-parity fix: proves `GatewayClient::stream` actually sets [`ModelRequest::is_codex`] on
    /// the live request from the same `RouteOverride::Prefixed` signal that already picks Codex's own
    /// URL/path (`prefixed_route_override_reaches_the_gateway_under_the_provider_prefix_with_static_
    /// headers` proves the routing half; `codex_routed_requests_send_instructions_field_instead_of_
    /// folding_system_into_input` in `dialect/openai_responses.rs` covers the dialect-side gate
    /// directly) — the system prompt must land in a top-level `instructions` field, never folded into
    /// `input`, with `parallel_tool_calls`/`text.verbosity` always present, on the real wire body.
    #[tokio::test]
    async fn codex_routed_request_sends_instructions_and_parallel_tool_calls_end_to_end() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            // Read the raw bytes rather than lossy-UTF8: this Codex-routed HTTP/SSE-fallback request's
            // body is zstd-compressed (`is_codex_sse_fallback`, pi-parity), so the assertions below
            // decompress the body before checking its JSON content. A single `read` is enough here —
            // the (pre-compression) body was already small enough to arrive in one packet before this
            // fix, and compression only shrinks it further.
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let raw = &buf[..n];
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            let split = raw
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .expect("header/body separator");
            let (head, body) = (&raw[..split], &raw[split + 4..]);
            let head = String::from_utf8_lossy(head).to_string();
            let decompressed = zstd::stream::decode_all(body)
                .expect("the Codex fallback body must be a valid zstd frame");
            let json =
                String::from_utf8(decompressed).expect("decompressed body must be UTF-8 JSON");
            (head, json)
        });

        let routing = DirectRouting {
            route: RouteOverride::Prefixed {
                prefix: "/openai-codex",
                path: "/backend-api/codex/responses",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            format!("http://{addr}"),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap()
        // Same reason as `prefixed_route_override_reaches_the_gateway_under_the_provider_prefix_with_
        // static_headers` above: a one-shot mock server with no WS handling of its own, testing the
        // HTTP/SSE wire shape specifically.
        .with_codex_websocket(false);
        let req = ModelRequest::new(
            "gpt-5-codex",
            vec![crate::message::Message::user("hi")],
            100,
        )
        .with_system("be terse");
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let (head, request) = server.join().unwrap();
        assert!(
            head.to_lowercase().contains("content-encoding: zstd"),
            "expected the Codex HTTP/SSE fallback to zstd-compress its body, got headers:\n{head}"
        );
        assert!(
            request.contains("\"instructions\":\"be terse\""),
            "expected the system prompt in a top-level instructions field, got:\n{request}"
        );
        assert!(
            request.contains("\"parallel_tool_calls\":true"),
            "expected parallel_tool_calls:true always sent for a Codex-routed request, got:\n{request}"
        );
        assert!(
            request.contains("\"text\":{\"verbosity\":\"low\"}"),
            "expected text.verbosity always sent for a Codex-routed request, got:\n{request}"
        );
    }

    /// Pi-parity Fix 1 (Round 2): proves [`DirectRouting::dialect_override`] wins over
    /// `Dialect::for_model_via_copilot`'s own name heuristic. `kimi-k2-thinking` matches neither
    /// `for_model`'s "claude"/"anthropic" substring check nor `ApiKind::Responses`, so without the
    /// override it would build an OpenAI Chat Completions body and never send Anthropic's
    /// `anthropic-version` header — exactly the wire-format mismatch a genuinely Anthropic-wire
    /// third-party provider (Kimi-Coding) would hit.
    #[tokio::test]
    async fn dialect_override_forces_anthropic_wire_for_a_model_id_that_fails_the_name_heuristic() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/v1/messages",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: Some(Dialect::Anthropic),
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new(
            "kimi-k2-thinking",
            vec![crate::message::Message::user("hi")],
            100,
        );
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let lower = request.to_lowercase();
        assert!(
            lower.contains("anthropic-version: 2023-06-01"),
            "a model id that fails the name heuristic must still build Anthropic wire when \
             dialect_override says so, got:\n{request}"
        );
    }

    /// A non-overridden request for the same id proves the baseline this fix closes: without
    /// `dialect_override`, `kimi-k2-thinking` falls through to Chat Completions and never sends
    /// Anthropic's `anthropic-version` header.
    #[tokio::test]
    async fn without_a_dialect_override_the_same_model_id_builds_chat_completions_wire() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new(
            "kimi-k2-thinking",
            vec![crate::message::Message::user("hi")],
            100,
        );
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert!(
            !request.to_lowercase().contains("anthropic-version:"),
            "baseline check: an unclassified model id must NOT get Anthropic's own version header \
             without an explicit dialect_override, got:\n{request}"
        );
    }

    /// pi-parity pass 20, Task 5/6: proves the full `DirectRouting::aggregator_host` pipeline —
    /// populated by a BYO `models.json` `base_url` override (`crates/agent::gateway_credential`, not
    /// exercised here directly) and read back out by `GatewayClient::stream` to set
    /// `ModelRequest::host`, which `Dialect::for_model_via_copilot` then consults to resolve one of the
    /// 3 real host-dependent bare-id wire collisions (`crate::dialect::anthropic_wire_bare_id_for_host`).
    /// "minimax-m3" is Anthropic-wire by default (`NATIVE_ANTHROPIC_WIRE_BARE_IDS`, matching native
    /// MiniMax and OpenCode-Go) but genuinely `openai-completions` on OpenCode Zen specifically — with no
    /// `aggregator_host` signal this would silently build an Anthropic-shaped body and send it to an
    /// OpenAI-wire endpoint, a hard 400.
    #[tokio::test]
    async fn aggregator_host_from_a_byo_override_resolves_a_real_bare_id_host_collision() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/v1/chat/completions",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: None,
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: Some(crate::models::AggregatorHost::OpenCodeZen),
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new("minimax-m3", vec![crate::message::Message::user("hi")], 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert!(
            !request.to_lowercase().contains("anthropic-version:"),
            "aggregator_host: Some(OpenCodeZen) must resolve \"minimax-m3\" to the real openai-wire \
             format for that host, not the host-agnostic Anthropic default, got:\n{request}"
        );
    }

    /// The converse of the test above: with no `aggregator_host` signal at all (the common case for a
    /// plain gateway-relayed request), "minimax-m3" keeps the pre-existing host-agnostic default
    /// (Anthropic — correct for native MiniMax and OpenCode-Go, just not OpenCode Zen).
    #[tokio::test]
    async fn without_an_aggregator_host_the_same_bare_id_keeps_the_host_agnostic_anthropic_default()
    {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new("minimax-m3", vec![crate::message::Message::user("hi")], 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        assert!(
            request.to_lowercase().contains("anthropic-version:"),
            "baseline check: with no aggregator host signal, \"minimax-m3\" must keep building the \
             Anthropic wire, got:\n{request}"
        );
    }

    /// Pi-parity Fix 2 (Round 2): proves [`DirectRouting::query`] is appended to a `Direct` route's
    /// URL — an Azure resource pinned to a dated `api-version` needs this on every request.
    #[tokio::test]
    async fn direct_route_query_param_is_appended_to_the_built_url() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}/openai/v1"),
                path: "/responses",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: Some("api-key".to_string()),
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: None,
            query: Some("api-version=2024-08-01-preview".to_string()),
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new("gpt-4o", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let request_line = request.lines().next().unwrap_or_default();
        assert_eq!(
            request_line, "POST /openai/v1/responses?api-version=2024-08-01-preview HTTP/1.1",
            "expected the api-version query param appended to the Direct route's URL, got:\n{request}"
        );
    }

    /// Pi-parity Fix 2 (Round 2): proves [`DirectRouting::deployment_name`] overwrites just the
    /// wire-level `"model"` field the dialect already built, not `ModelRequest::model` itself — an
    /// Azure deployment name doesn't have to match the app-level model id.
    #[tokio::test]
    async fn deployment_name_override_replaces_the_wire_level_model_field() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}/openai/v1"),
                path: "/responses",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: Some("api-key".to_string()),
            auth_header_prefix: None,
            dialect_override: None,
            deployment_name: Some("my-azure-deployment".to_string()),
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        // The app-level id used for capability lookups; the wire-level `"model"` field must instead
        // carry the deployment name.
        let req = ModelRequest::new("gpt-4o", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let body_start = request.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
        let body = &request[body_start..];
        assert!(
            body.contains("\"model\":\"my-azure-deployment\""),
            "expected the deployment name in the wire-level model field, got body:\n{body}"
        );
        assert!(
            !body.contains("\"model\":\"gpt-4o\""),
            "the app-level model id must not leak into the wire-level model field when a deployment \
             name override is set, got body:\n{body}"
        );
    }

    /// Pi-parity Fix 4 (Round 2): proves [`DirectRouting::auth_header_prefix`] prepends the prefix to
    /// the credential value sent through `auth_header` — Cloudflare AI Gateway's
    /// `cf-aig-authorization: Bearer <key>`, a named header (like Azure's bare `api-key`) but carrying
    /// a Bearer-prefixed value (unlike Azure's).
    #[tokio::test]
    async fn auth_header_prefix_is_prepended_to_the_credential_value() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            drop(stream);
            request
        });

        let routing = DirectRouting {
            route: RouteOverride::Direct {
                base_url: format!("http://{addr}"),
                path: "/v1/chat/completions",
            },
            static_headers: Vec::new(),
            copilot_dynamic_headers: false,
            auth_header: Some("cf-aig-authorization".to_string()),
            auth_header_prefix: Some("Bearer ".to_string()),
            dialect_override: None,
            deployment_name: None,
            query: None,
            aggregator_host: None,
        };
        let client = GatewayClient::with_credential_source(
            "http://127.0.0.1:1".to_string(),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
        let req = ModelRequest::new("llama-3.1-70b", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        while events.next().await.is_some() {}

        let request = server.join().unwrap();
        let lower = request.to_lowercase();
        assert!(
            lower.contains("cf-aig-authorization: bearer test-oauth-token"),
            "expected the Bearer-prefixed credential in the named header, got:\n{request}"
        );
        // `\nauthorization:` (not a bare `"authorization:"` substring, which `cf-aig-authorization:`
        // itself also contains) — the real `Authorization` header must still be entirely absent.
        assert!(
            !lower.contains("\nauthorization:"),
            "Authorization must be entirely absent when auth_header overrides it, got:\n{request}"
        );
    }
}
