//! The default model transport: an HTTP client that speaks provider wire to the Beyond gateway.
//!
//! This is the harness's whole network surface. It never holds a provider key or picks a provider —
//! it sends `Authorization: Bearer <bai_v1…>` to the gateway, which swaps in the pool key, routes to
//! the real provider, and meters usage. The client only picks the *dialect* (by model id), builds
//! the request body, and frames the streaming SSE response back into [`StreamEvent`]s.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde_json::Value;
use zeroize::Zeroize;

use crate::dialect::{Dialect, SseEventBuffer, push_sse_line};
use crate::error::{Error, MID_STREAM_NETWORK_ERROR, Result};
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
/// Ceiling on a single backoff wait, so a server-supplied `Retry-After` can't park a run for minutes.
const MAX_BACKOFF: Duration = Duration::from_secs(10);

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
    Direct { base_url: String, path: &'static str },
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
        })
    }

    /// Builder-style: override the pre-first-byte retry budget and exponential-backoff base (still
    /// capped at [`MAX_BACKOFF`]).
    pub fn with_retry(mut self, max_retries: u32, base_backoff: Duration) -> Self {
        self.max_retries = max_retries;
        self.base_backoff = base_backoff;
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
        let dialect = Dialect::for_model(&req.model);
        // Resolved fresh for this turn rather than a field snapshot — the seam that lets a
        // credential expire and be refreshed mid-process (see `CredentialSource`'s doc comment).
        // `send_with_retry`'s own internal transient-failure retries never hit 401, so one fetch per
        // `stream()` call (once per model turn) is all that's needed, not one per retry attempt.
        // Fetched before `url` is built (rather than after, as a plain-key client would): a
        // Copilot/Codex-sourced credential's `direct` routing (see `RouteOverride`) determines the URL
        // itself, not just the bearer value.
        let credential = self.credential_source.credential().await?;
        let url = match credential.direct.as_ref().map(|d| &d.route) {
            Some(RouteOverride::Prefixed { prefix, path }) => {
                format!("{}{prefix}{path}", self.base_url)
            }
            Some(RouteOverride::Direct { base_url, path }) => format!("{base_url}{path}"),
            None => format!("{}{}", self.base_url, dialect.endpoint_path()),
        };
        // GitHub Copilot's per-turn dynamic headers (see `DirectRouting::copilot_dynamic_headers`'s
        // doc comment) — computed here, before `req` is otherwise consumed, from this turn's own
        // messages. `None` for every credential but a Copilot one.
        let copilot_dynamic = credential
            .direct
            .as_ref()
            .filter(|d| d.copilot_dynamic_headers)
            .map(|_| (copilot_initiator(&req.messages), copilot_has_images(&req.messages)));
        let direct_headers: Vec<(&'static str, String)> = credential
            .direct
            .as_ref()
            .map(|d| d.static_headers.clone())
            .unwrap_or_default();
        // Anthropic's own OAuth identity-spoofing headers (`CLAUDE_CODE_BETA`/`CLAUDE_CLI_IDENTITY`
        // etc., below) only apply to a genuine direct-to-Anthropic request — a Claude-family model
        // routed through a Copilot credential (`direct.is_some()`) is `is_oauth` too, but must not
        // also claim to be the Anthropic-official Claude Code client to GitHub's Copilot proxy.
        let has_direct_override = credential.direct.is_some();
        let body = dialect.build_body(&req);
        let http = self.http.clone();
        let max_retries = self.max_retries;
        let base_backoff = self.base_backoff;

        let is_anthropic = dialect == Dialect::Anthropic;
        // Both OpenAI wire dialects use this for connection-level session-affinity routing, distinct
        // from `prompt_cache_key`'s cache-node affinity in the body — matches pi's
        // `openai-responses.ts` (`headers["x-client-request-id"] = sessionId`) and
        // `openai-completions.ts` (`compat.sendSessionAffinityHeaders`), reusing the same
        // per-conversation `cache_key` value pi's own `sessionId` carries. Suppressed when the request
        // opted out of caching (`no_cache`): pinning a one-off request to a specific gateway/cache node
        // makes no sense when it's explicitly not trying to read a cache back, and pi's own dialects
        // gate this the same way `no_cache` gates the body's own `prompt_cache_key`/`cache_control`.
        let session_affinity_header =
            (matches!(dialect, Dialect::OpenAiResponses | Dialect::OpenAi) && !req.no_cache)
                .then_some(req.cache_key.as_deref())
                .flatten()
                .map(str::to_string);
        // pi's Chat Completions dialect additionally sends `x-session-affinity` alongside `session_id`
        // and `x-client-request-id` (`openai-completions.ts`'s `compat.sendSessionAffinityHeaders`
        // branch); the Responses dialect never sends it.
        let send_x_session_affinity = dialect == Dialect::OpenAi;
        let needs_interleaved_beta = needs_interleaved_thinking_beta(&req.model);
        let needs_fine_grained_tool_streaming_beta = !req.tools.is_empty()
            && !crate::models::capabilities(&req.model).supports_eager_tool_streaming;
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
                is_anthropic,
                // Only a genuine direct-to-Anthropic OAuth login gets Anthropic's own identity-
                // spoofing headers — never a Copilot-routed request that merely happens to be
                // Claude-family (see `has_direct_override`'s doc comment above).
                credential.is_oauth && !has_direct_override,
                needs_interleaved_beta,
                needs_fine_grained_tool_streaming_beta,
                session_affinity_header.as_deref(),
                send_x_session_affinity,
                &direct_headers,
                copilot_dynamic,
                max_retries,
                base_backoff,
            )
            .await?;

            // Frame the chunked body line-by-line. A partial trailing line is buffered across chunks
            // until its newline arrives. The framing (byte buffering + newline split) lives in
            // `LineFramer` — see its doc comment for why it buffers raw *bytes*, not a lossy per-chunk
            // string. `sse_buf` is the separate SSE-level buffer that joins consecutive `data:` lines
            // belonging to one logical event (see `SseEventBuffer`'s doc comment) — real
            // Anthropic/OpenAI never split an event across lines, but a spec-conformant intermediary
            // could, and the SSE spec's own event boundary is a blank line, not a per-`data:`-line one.
            let mut decoder = dialect.decoder();
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

/// Reassembles a chunked byte stream into whole newline-terminated lines — the SSE framing seam.
///
/// It buffers raw *bytes*, not a per-chunk lossy string: a TCP/HTTP chunk boundary can split a
/// multi-byte UTF-8 character, and `from_utf8_lossy` per chunk would replace each half with U+FFFD,
/// silently corrupting non-ASCII tool arguments and prose. A `\n` (0x0A) never falls inside a UTF-8
/// multi-byte sequence, so every newline-terminated line handed back by [`next_line`](Self::next_line)
/// is guaranteed whole UTF-8; only the unterminated tail — which may split a character — stays
/// buffered for the next chunk (surfaced by [`take_tail`](Self::take_tail) at end-of-stream).
///
/// Public so the streaming decode hot path is benchable in isolation (`benches/decode.rs`), the same
/// way the gateway exposes its request-scan primitives.
///
/// Backed by a [`BytesMut`]: [`next_line`](Self::next_line) finds the newline with SIMD [`memchr`] and
/// hands the line back via `split_to`, an O(1) pointer split that shares the backing allocation — so a
/// line costs neither a per-line heap allocation nor a memmove of the buffer remainder. (A `Vec<u8>`
/// framer paid both on *every* line: `drain(..=nl).collect()` allocates the line and shifts the rest
/// of the buffer down, which is O(lines × remaining) when a chunk carries many coalesced lines.)
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

    /// Pop the next complete line (including its trailing `\n`), or `None` if the buffer holds no
    /// full line yet — the caller then awaits the next chunk. The returned [`Bytes`] shares the
    /// framer's backing buffer (no copy); it's dropped as soon as the line is decoded, freeing that
    /// region for the buffer to reclaim.
    pub fn next_line(&mut self) -> Option<Bytes> {
        let nl = memchr::memchr(b'\n', &self.buf)?;
        Some(self.buf.split_to(nl + 1).freeze())
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
    max_retries: u32,
    base_backoff: Duration,
) -> Result<reqwest::Response> {
    let mut attempt = 0u32;
    loop {
        let mut builder = http.post(url).bearer_auth(api_key).json(body);
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
                    .header("anthropic-dangerous-direct-browser-access", "true")
                    .header("user-agent", CLAUDE_CLI_IDENTITY)
                    .header("x-app", "cli");
            }
        }
        match builder.send().await {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                if is_retryable_status(status.as_u16()) && attempt < max_retries {
                    let hint = retry_after(&resp);
                    // A 429 needs a quick body peek before committing to a retry: some providers use it
                    // for genuine rate limiting (worth retrying — the request will likely succeed once
                    // the window resets) and others for quota/billing exhaustion (retrying only delays
                    // an unavoidable failure while burning the retry budget on it). The status code
                    // alone can't tell the two apart. Every other retryable status (408/409/5xx/529) is
                    // a pure infra hiccup, never a billing signal, so it skips this check.
                    if status.as_u16() == 429 {
                        let detail = resp.text().await.unwrap_or_default();
                        if is_quota_exhausted(&detail) {
                            return Err(Error::Transport(format!(
                                "gateway returned {status}: {}",
                                truncate_error_body(detail.trim())
                            )));
                        }
                    }
                    let wait = backoff(attempt, hint, base_backoff);
                    attempt += 1;
                    futures_timer::Delay::new(wait).await;
                    continue;
                }
                // Non-retryable, or out of retries: surface the body so the caller sees *why* — capped,
                // since an upstream can return an arbitrarily large error page (an HTML error document
                // from a misconfigured proxy, say) and this ends up in logs and `AgentEvent::Error`.
                let detail = resp.text().await.unwrap_or_default();
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
                    let wait = backoff(attempt, None, base_backoff);
                    attempt += 1;
                    futures_timer::Delay::new(wait).await;
                    continue;
                }
                return Err(Error::Transport(e.to_string()));
            }
        }
    }
}

/// Status codes worth retrying: rate limiting, the Anthropic-specific `529 overloaded`, and the
/// transient 5xx gateway/upstream failures. A 4xx other than 429 is the caller's fault — don't retry.
fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 429 | 500 | 502 | 503 | 504 | 529)
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

/// Truncate an error body to [`MAX_ERROR_BODY_CHARS`], on a char boundary, noting what was cut.
fn truncate_error_body(s: &str) -> String {
    if s.chars().count() <= MAX_ERROR_BODY_CHARS {
        return s.to_string();
    }
    let kept: String = s.chars().take(MAX_ERROR_BODY_CHARS).collect();
    format!("{kept}… [truncated]")
}

/// Parse a `Retry-After` response header into a duration, capped at [`MAX_BACKOFF`].
fn retry_after(resp: &reqwest::Response) -> Option<Duration> {
    let raw = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?;
    parse_retry_after(raw)
}

/// Parse a `Retry-After` header *value* into a wait, capped at [`MAX_BACKOFF`]. RFC 7231 allows two
/// forms: delta-seconds (`120`) and an absolute HTTP-date (`Wed, 21 Oct 2025 07:28:00 GMT`). The
/// date form is converted to a delay from now; a date already in the past (clock skew, a stale hint)
/// yields no extra wait. Split out from [`retry_after`] so it's testable without a `reqwest::Response`.
fn parse_retry_after(raw: &str) -> Option<Duration> {
    let raw = raw.trim();
    // Delta-seconds: a bare non-negative integer count of seconds.
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(Duration::from_secs(secs).min(MAX_BACKOFF));
    }
    // HTTP-date: the absolute instant to retry at. Anything we can't parse as either form is ignored.
    let target = httpdate::parse_http_date(raw).ok()?;
    let delay = target
        .duration_since(std::time::SystemTime::now())
        .unwrap_or(Duration::ZERO);
    Some(delay.min(MAX_BACKOFF))
}

/// The wait before the next attempt (0-indexed): the larger of the server's `Retry-After` hint and
/// our exponential backoff `base_backoff · 2^attempt`, capped at [`MAX_BACKOFF`].
fn backoff(attempt: u32, retry_after: Option<Duration>, base_backoff: Duration) -> Duration {
    // `min(16)` keeps the shift well within `u32` (and `saturating_mul` mops up the rest); by then the
    // result has long since hit `MAX_BACKOFF`.
    let exp = base_backoff
        .saturating_mul(1u32 << attempt.min(16))
        .min(MAX_BACKOFF);
    match retry_after {
        Some(hint) => hint.max(exp).min(MAX_BACKOFF),
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
        for s in [429, 500, 502, 503, 504, 529, 408, 409] {
            assert!(is_retryable_status(s), "{s} should be retryable");
        }
        for s in [200, 400, 401, 403, 404, 422] {
            assert!(!is_retryable_status(s), "{s} should not be retryable");
        }
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
    fn backoff_is_exponential_and_capped() {
        assert_eq!(backoff(0, None, BASE_BACKOFF), BASE_BACKOFF);
        assert_eq!(backoff(1, None, BASE_BACKOFF), BASE_BACKOFF * 2);
        assert_eq!(backoff(2, None, BASE_BACKOFF), BASE_BACKOFF * 4);
        assert_eq!(backoff(20, None, BASE_BACKOFF), MAX_BACKOFF); // saturates, never overflows
        // A server hint wins when larger, but is still capped.
        assert_eq!(
            backoff(0, Some(Duration::from_secs(2)), BASE_BACKOFF),
            Duration::from_secs(2)
        );
        assert_eq!(
            backoff(0, Some(Duration::from_secs(3600)), BASE_BACKOFF),
            MAX_BACKOFF
        );
    }

    #[test]
    fn backoff_honors_a_custom_base() {
        let custom = Duration::from_millis(1000);
        assert_eq!(backoff(0, None, custom), custom);
        assert_eq!(backoff(1, None, custom), custom * 2);
        assert_eq!(backoff(20, None, custom), MAX_BACKOFF); // still saturates at the shared cap
    }

    #[test]
    fn truncate_error_body_caps_a_large_upstream_error_page() {
        let short = "gateway timeout";
        assert_eq!(truncate_error_body(short), short);

        let huge = "x".repeat(MAX_ERROR_BODY_CHARS + 500);
        let truncated = truncate_error_body(&huge);
        assert!(truncated.ends_with("… [truncated]"));
        assert_eq!(
            truncated.chars().count(),
            MAX_ERROR_BODY_CHARS + "… [truncated]".chars().count()
        );
    }

    #[test]
    fn retry_after_accepts_delta_seconds_and_http_date() {
        // Delta-seconds form, capped at MAX_BACKOFF.
        assert_eq!(parse_retry_after("2"), Some(Duration::from_secs(2)));
        assert_eq!(parse_retry_after(" 3 "), Some(Duration::from_secs(3)));
        assert_eq!(parse_retry_after("99999"), Some(MAX_BACKOFF));
        // A value that is neither an integer nor an HTTP-date is ignored.
        assert_eq!(parse_retry_after("soon"), None);
        // HTTP-date already in the past → no extra wait.
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"),
            Some(Duration::ZERO)
        );
        // HTTP-date in the future → a positive, capped delay.
        let future = std::time::SystemTime::now() + Duration::from_secs(5);
        let delay = parse_retry_after(&httpdate::fmt_http_date(future)).expect("a parsed delay");
        assert!(
            delay > Duration::ZERO && delay <= MAX_BACKOFF,
            "future http-date should yield a bounded positive delay, got {delay:?}"
        );
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
            !request.contains("anthropic-dangerous-direct-browser-access"),
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
    /// a Chat Completions dialect request too, not just OpenAI Responses. Matches
    /// `openai-completions.ts`'s `compat.sendSessionAffinityHeaders` branch, exercised by
    /// `packages/ai/test/openai-completions-prompt-cache.test.ts`'s "sends known session-affinity
    /// headers when compat.sendSessionAffinityHeaders is enabled" case.
    #[tokio::test]
    async fn session_affinity_headers_are_sent_for_chat_completions_dialect_too() {
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
        // OpenAI-compatible provider, not native OpenAI Responses.
        let req = ModelRequest::new("llama-3.1-70b", Vec::new(), 100)
            .with_cache_key("session-affinity-test");
        let mut events = client.stream(req).await.unwrap();
        let _ = events.next().await;
        server.join().unwrap();

        let request = captured.lock().unwrap().clone();
        assert!(
            request.contains("session_id: session-affinity-test"),
            "expected a `session_id` header on a Chat Completions request, got:\n{request}"
        );
        assert!(
            request.contains("x-client-request-id: session-affinity-test"),
            "expected an `x-client-request-id` header on a Chat Completions request, got:\n{request}"
        );
        assert!(
            request.contains("x-session-affinity: session-affinity-test"),
            "expected an `x-session-affinity` header on a Chat Completions request, got:\n{request}"
        );
    }

    /// Pi-parity audit task #40: `no_cache` must suppress the session-affinity headers too, not just
    /// the cache-control blocks — otherwise a request that explicitly opted out of prompt caching still
    /// pins itself to a specific backend session, defeating the purpose of the opt-out. Matches pi's
    /// `compat.sendSessionAffinityHeaders` gating, which never fires when caching is disabled for the
    /// request.
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
        let req = ModelRequest::new("llama-3.1-70b", Vec::new(), 100)
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
        };
        let client = GatewayClient::with_credential_source(
            format!("http://{addr}"),
            Arc::new(DirectTestCredential(routing)),
        )
        .unwrap();
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
            request.to_lowercase().contains("openai-beta: responses=experimental"),
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
}
