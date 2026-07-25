//! The Pingora `ProxyHttp` passthrough service.
//!
//! Flow: pick the provider from the **first path segment** (`/{provider}/…`) → verify the virtual
//! key (stateless) → deny-set check (O(1), default-allow) → swap the auth
//! header to the pool key (managed only) → **stream the request body straight through** (never
//! buffered; original framing preserved) while feeding it to a structural scanner that extracts the
//! exact root-level `model` → relay the response **without buffering** → tap usage from a bounded
//! tail → emit a usage fact. Whether the call is streaming is derived from the *response*
//! Content-Type.
//!
//! Verified end-to-end (`tests/e2e.rs`): a real `beyond-ai` binary against real nats-server + a
//! mock upstream — passthrough fidelity, key swap, usage metering (non-streaming + SSE), BYO
//! passthrough, and deny-set propagation all pass.
//!
//! We never read the request body in `request_filter`: Pingora's body-forward phase reads the
//! downstream body itself, so draining it earlier would make Pingora send `Content-Length` bytes
//! with no body and the upstream would hang. We let the body flow through `request_body_filter`
//! (the supported hook), feeding each chunk to a streaming structural scanner (`peek::ModelScanner`,
//! O(1) memory) — never withholding or buffering it.
//!
//! One deliberate exception to the no-buffer rule: a **managed** OpenAI Chat Completions request is
//! buffered and gets `stream_options.include_usage` injected when it streams without it — otherwise
//! OpenAI emits no usage chunk and the request couldn't be metered. We can't set that option in a
//! client SDK we don't control, so the gateway guarantees it, out of the box. Scoped to exactly that
//! path (managed + OpenAI dialect + chat/completions); BYO and everything else stay pure passthrough.
//! The Responses API needs no such injection — it always reports usage on its terminal event — so it
//! stays pure passthrough too (see `is_streamable_path`).
//!
//! Auth branches on key format: `bai_…` is a managed virtual key (verify → deny-check → swap to
//! the pool key); anything else is a **BYO** request — the user's own provider token, passed
//! through unchanged (no swap, no Beyond identity, no deny-set). The key is read from whichever
//! header (or, for Google Gemini, query param) the client's SDK uses — see `extract_virtual_key`.
//!
//! Routing is by the **first path segment** = provider name (`route`, data-driven): `/{provider}/…`
//! selects the provider and the rest of the path is forwarded **verbatim** (the gateway holds no
//! per-provider mount knowledge). A bare path with no provider prefix that is exactly `/v1` or
//! starts with `/v1/` (boundary-checked — see `route::is_default_prefix`, not a raw
//! `starts_with("/v1")`, which would also absorb a lookalike like Google Gemini's `/v1beta/…`) is
//! the drop-in default — dialect picks openai/anthropic (`dialect_for_path`) — so an OpenAI/
//! Anthropic client works by changing only the host. An unknown first segment is a 404. Model isn't
//! used for routing (the body isn't read pre-connect); it's still captured from the body for usage.

use crate::route::{self, Dialect, Provider};
use crate::state::{GatewayState, RequestId};
use crate::{peek, usage};
use async_trait::async_trait;
use bytes::Bytes;
use pingora::http::ResponseHeader;
use pingora_core::Result;
use pingora_core::protocols::ALPN;
use pingora_core::upstreams::peer::HttpPeer;
use pingora_proxy::{ProxyHttp, Session};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Response header carrying the per-request id (`{instance}-{seq}`). Set on both the proxied
/// response and every reject body so a client can quote it and an oncall can grep for it.
const REQUEST_ID_HEADER: &str = "x-beyond-request-id";

/// OpenRouter's dashboard-attribution headers (https://openrouter.ai/docs/quickstart): purely
/// cosmetic on OpenRouter's side (their own cost/usage categorization), no effect on the request
/// or response. Static — this doesn't need to be configurable, just present. Only OpenRouter is in
/// `KNOWN_PROVIDERS` among the providers pi attributes to (NVIDIA NIM, Cloudflare, Vercel AI
/// Gateway aren't routed providers here), so this is the one case worth porting. Header names match
/// pi's current OpenRouter-specific set (`packages/coding-agent/src/core/provider-attribution.ts`):
/// `HTTP-Referer`, `X-OpenRouter-Title` (NOT the generic `X-Title` pi used for the now-removed Vercel
/// AI Gateway route), and `X-OpenRouter-Categories`.
const OPENROUTER_REFERER: &str = "https://beyond.build";
const OPENROUTER_TITLE: &str = "Beyond Gateway";
const OPENROUTER_CATEGORY: &str = "cli-agent";

/// Reject requests whose declared Content-Length exceeds this. The body itself is **not** buffered
/// (it streams straight through); this is purely an abuse guard checked up front via the header.
const MAX_REQUEST_BODY: usize = 100 * 1024 * 1024;

/// Bounded tail of the response kept for usage extraction. The usage event is the final SSE chunk
/// / the whole non-streaming body; keeping a tail means we never buffer a long stream.
const USAGE_TAIL_CAP: usize = 64 * 1024;

/// Bounded **head** of an Anthropic SSE response, kept alongside the tail.
///
/// A tail alone is enough for every other shape — OpenAI puts its usage chunk at the end, and a
/// non-streaming body carries `usage` last. Anthropic streaming is the exception: `input_tokens`
/// and both cache counters ride on `message_start`, the *first* event, while the output count rides
/// on the last `message_delta`. The two facts sit at opposite ends of a stream that can be
/// megabytes long, so a tail-only tap dropped `message_start` for any response past roughly 500
/// output tokens and billed `input_tokens = 0` — and silently, because `saw_any` still went true
/// off the `message_delta`, so the parse never looked like an error.
///
/// 8 KiB is far more than needed (a `message_start` event is a few hundred bytes and is the first
/// thing on the wire) but leaves room for a provider that emits `ping`s or other preamble first.
const USAGE_HEAD_CAP: usize = 8 * 1024;

/// Max upstream **connect** retries before surfacing the failure to the client.
///
/// We retry connect failures only (the idiomatic Pingora pattern, same as edge). Retrying on a
/// received **5xx/429 response** is deliberately *not* done: Pingora 0.8 has no clean
/// post-response retry hook for a streaming passthrough (edge doesn't do it either), the upstream
/// may have started streaming, and the provider SDKs already back off on 429/5xx + `Retry-After`.
const MAX_CONNECT_RETRIES: u8 = 2;

pub struct AiProxy {
    pub state: Arc<GatewayState>,
}

/// Per-request context. `None` until `request_filter` admits the request; short-circuited
/// requests (auth/deny failures) leave it `None`, so later filters no-op.
pub struct RequestCtx {
    tenant_id: u64,
    vpc_id: u64,
    dialect: Dialect,
    /// The resolved upstream provider (authority/host + precomputed managed auth value), shared from
    /// the boot-time registry — a cheap `Arc` clone, nothing re-allocated per request.
    provider: Arc<Provider>,
    /// The path (+ query) to send upstream: the client path with the `/{provider}` segment stripped
    /// (provider-prefixed request) or unchanged (bare-path default). Forwarded **verbatim** — the
    /// gateway does no per-provider path rewriting. Applied as the upstream URI in
    /// `upstream_request_filter`.
    forward_path: String,
    /// Whether this is a **managed** request (`bai_…` key → swap to the pool key). `false` for
    /// **BYO** — we leave the user's own auth header untouched (passthrough).
    managed: bool,
    /// Model the client *requested*, extracted from the request body. This is the billing-log
    /// **fallback** — the authoritative value is the model the provider echoes in its response (see
    /// `resp_model_scanner`), because a client may send an alias (`gpt-4o`) that the provider resolves
    /// to and bills under a pinned id (`gpt-4o-2024-08-06`).
    model: String,
    model_scanner: peek::ModelScanner,
    /// Extracts the model the **provider** reports in its response (the resolved/billed id), fed the
    /// response stream in `response_body_filter`. Preferred over `model` in the `ai.usage` event so
    /// the billed model is authoritative, not the requested alias. Works for SSE too: the scanner
    /// skips the `data: ` prefix and reads the first chunk's root `model`. Falls back to `model` when
    /// the response carries none (e.g. an error body).
    resp_model_scanner: peek::ModelScanner,
    /// Whether the upstream response is an SSE stream — set in `response_filter` from the response
    /// Content-Type (we don't read the request to learn this).
    streaming: bool,
    /// Bounded tail of the response, for the usage tap.
    resp_tail: Vec<u8>,
    /// Bounded head of the response — populated only for an **Anthropic SSE** response, whose input
    /// and cache token counts arrive on the very first event and would otherwise be compacted out of
    /// `resp_tail`. See [`USAGE_HEAD_CAP`]. Empty for every other dialect and for non-streaming
    /// responses, which carry everything the tail already holds.
    resp_head: Vec<u8>,
    /// Running total of request-body bytes seen, to enforce `MAX_REQUEST_BODY` even when the client
    /// uses chunked transfer encoding (no `Content-Length` to check up front).
    body_bytes_fed: usize,
    /// Upstream HTTP status, set in `response_filter` once the response head arrives. Drives the
    /// circuit-breaker outcome recorded once in `logging`: `5xx` → failure, any other response →
    /// success (the provider answered — a `429` is a healthy throttle, not a breaker trip), and a
    /// `None` here with an upstream error → failure (connect/read failed before any response).
    upstream_status: Option<u16>,
    /// Managed OpenAI chat/completions request: buffer the body and inject
    /// `stream_options.include_usage` if it streams without it, so the usage chunk (hence the
    /// billable token count) is guaranteed. The single, deliberate exception to "never buffer the
    /// request body" — scoped to the managed OpenAI chat/completions path (see `is_streamable_path`)
    /// and bounded by `MAX_REQUEST_BODY`. BYO and every other request still stream straight through.
    inject_eligible: bool,
    /// Accumulated request body — populated only when `inject_eligible`; otherwise stays empty and
    /// the body is never buffered.
    req_buf: Vec<u8>,
    start: Instant,
    /// Connect-retry counter (see `fail_to_connect`).
    attempt: u8,
    /// Process-unique id for this request (`{instance}-{seq}`), echoed in the `x-beyond-request-id`
    /// response header and the `ai.usage` event so a client report ties back to a log line.
    request_id: RequestId,
}

impl AiProxy {
    /// Write a small JSON error and signal `request_filter` to short-circuit. The body is built with
    /// `serde_json` (not `format!`) so a `typ`/`msg` containing `"` or `\` can never break out of the
    /// JSON structure — keeps this safe if a future caller passes a non-literal message.
    ///
    /// Every rejection logs one structured `warn` line (the rejection counter only says *how many*,
    /// not *which request* — this is what an oncall greps when a `deny_fraud`/`rate_limit` spike
    /// shows on the dashboard) and echoes the `request_id` in a response header so a client report
    /// quoting that id lands on this line.
    async fn reject(
        session: &mut Session,
        request_id: &str,
        status: u16,
        typ: &str,
        msg: &str,
    ) -> Result<bool> {
        warn!(request_id, status, error_type = typ, "request rejected");
        let body = Bytes::from(
            serde_json::json!({ "error": { "type": typ, "message": msg } }).to_string(),
        );
        let mut resp = ResponseHeader::build(status, None)?;
        resp.insert_header("content-type", "application/json")?;
        resp.insert_header("content-length", body.len().to_string())?;
        resp.insert_header(REQUEST_ID_HEADER, request_id)?;
        session.write_response_header(Box::new(resp), false).await?;
        session.write_response_body(Some(body), true).await?;
        Ok(true)
    }
}

/// Header names carrying a plain static API key (no OAuth/signing), checked in order. Anthropic:
/// `x-api-key`. Azure OpenAI: `api-key`. Google Gemini: `x-goog-api-key`. `Authorization: Bearer`
/// (OpenAI and everyone else) is checked separately below since it needs prefix-stripping.
const STATIC_KEY_HEADERS: [&str; 3] = ["x-api-key", "api-key", "x-goog-api-key"];

/// Extract query-param `name`'s value from a raw query string (`k=v&k2=v2`). Used only for Google
/// Gemini's `?key=` convention — the sole query-param credential shape among recognized providers.
/// No percent-decoding: a real API key is alphanumeric, so a plain split is exact, and decoding would
/// let a crafted query smuggle characters past the literal `name=` match.
fn query_param<'a>(query: &'a str, name: &str) -> Option<&'a str> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == name).then_some(v)
    })
}

/// Extract the presented key (virtual or BYO) from wherever the client's SDK puts it. Every
/// recognized shape is a **plain static key** (no OAuth/signing), so one neutral virtual key works
/// in any of them: Anthropic's `x-api-key`, Azure OpenAI's `api-key`, Google Gemini's
/// `x-goog-api-key` (header, falling back to the `?key=` query param — Gemini accepts either),
/// OpenAI's `Authorization: Bearer`. Header checks first since they're the common case and cheaper
/// (no query parse); query param last since it's the least-preferred shape (keys in a URL end up in
/// proxy/access logs). Borrowed from the request — no per-request copy.
fn extract_virtual_key(req: &pingora::http::RequestHeader) -> Option<&str> {
    for header in STATIC_KEY_HEADERS {
        if let Some(v) = req.headers.get(header).and_then(|v| v.to_str().ok()) {
            return Some(v);
        }
    }
    if let Some(v) = req
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    {
        return Some(v);
    }
    req.uri.query().and_then(|q| query_param(q, "key"))
}

/// Upper bound on a model id we'll record. Real ids are short (`claude-opus-4-8`,
/// `accounts/fireworks/models/…`); anything longer is junk or an attempt to bloat the billing log.
const MAX_MODEL_LEN: usize = 128;

/// Sanitize the model id extracted from the (client-controlled) request body before it lands in the
/// `ai.usage` billing log. `tracing`'s JSON layer escapes the value, but a downstream consumer
/// (logfwd/OTLP → ClickHouse) may re-handle it, so we refuse anything that could break out of a JSON
/// string or a line-oriented log: control bytes, `"`, `\`, `DEL`. A violating or over-long value is
/// recorded as `"unknown"` (matching `peek`'s non-UTF-8 fallback) rather than the raw bytes — a
/// mislabeled-but-safe usage row beats a corrupted or injected one.
fn sanitize_model(model: String) -> Cow<'static, str> {
    let bad = model.len() > MAX_MODEL_LEN
        || model
            .bytes()
            .any(|b| b < 0x20 || b == b'"' || b == b'\\' || b == 0x7f);
    if bad {
        Cow::Borrowed("unknown")
    } else {
        Cow::Owned(model)
    }
}

/// Set OpenRouter's attribution headers when (and only when) `provider_name` is `"openrouter"`
/// **and** the request is managed — every other provider, and every BYO request regardless of
/// provider, is untouched. Task #22 (pi-parity, Medium): pi gates this dashboard-attribution
/// behind a user-controllable telemetry setting (`isInstallTelemetryEnabled`,
/// `packages/coding-agent/src/core/provider-attribution.ts`); this gateway has no per-user
/// telemetry-opt-out setting to consult, but it already has an unambiguous, always-correct proxy
/// for "is this Beyond's own attribution to make, or someone else's traffic passing through us": a
/// BYO request carries the *caller's own* OpenRouter key, not Beyond's — attributing *their*
/// traffic to Beyond's dashboard app would misrepresent whose usage it is, the same harm the
/// telemetry opt-out exists to prevent. Managed traffic (Beyond's own pool key) is the only case
/// these headers describe accurately.
fn apply_provider_attribution(
    upstream_request: &mut pingora::http::RequestHeader,
    provider_name: &str,
    managed: bool,
) -> Result<()> {
    if provider_name == "openrouter" && managed {
        upstream_request.insert_header("HTTP-Referer", OPENROUTER_REFERER)?;
        upstream_request.insert_header("X-OpenRouter-Title", OPENROUTER_TITLE)?;
        upstream_request.insert_header("X-OpenRouter-Categories", OPENROUTER_CATEGORY)?;
    }
    Ok(())
}

fn dialect_for_path(path: &str) -> Dialect {
    // Anthropic Messages vs OpenAI Chat Completions/Embeddings. Embeddings are OpenAI-dialect only.
    if path.starts_with("/v1/messages") {
        Dialect::Anthropic
    } else {
        Dialect::OpenAi
    }
}

/// Resolve the provider name for a request whose first path segment matched no known/config
/// provider: `Some(name)` for the bare-path default — `path` is boundary-checked against
/// [`route::DEFAULT_PREFIX`] (see [`route::is_default_prefix`]) so a lookalike like Google
/// Gemini's `/v1beta/…` doesn't qualify — with the dialect picking openai/anthropic
/// ([`dialect_for_path`]); `None` for anything else, which the caller turns into a 404 rather than
/// silently guessing a provider (Task #7, pi-parity).
fn bare_default_provider_name(path: &str) -> Option<&'static str> {
    route::is_default_prefix(path).then(|| route::dialect_default(dialect_for_path(path)))
}

/// Whether the **forwarded** (provider-native) path targets the OpenAI Chat Completions endpoint.
/// Checked by *suffix*, so it holds regardless of the provider's mount prefix
/// (`/v1/chat/completions`, `/openai/v1/chat/completions`, `/inference/v1/chat/completions`, …). Only
/// this gets buffered for `stream_options.include_usage` injection — **not** `/v1/responses`: the
/// Responses API has no `stream_options` field at all (it always reports usage on the terminal
/// `response.completed` event, streaming or not), so splicing this chat-completions-only fragment into
/// a Responses body would inject a field the API doesn't recognize. Embeddings and everything else
/// never stream, so there's nothing to meter there either.
///
/// **Pass the path only — never a path with a query string.** The match is by suffix, so a trailing
/// `?api-version=2024-10-21` makes it return `false` for a path that plainly *is* chat/completions.
/// Azure OpenAI requires that parameter on every call, so testing this against a path+query silently
/// disabled injection for all managed Azure streams: no `stream_options.include_usage`, therefore no
/// usage chunk from OpenAI, therefore a zero-token billing row. The caller computes this in
/// `request_filter` *before* appending the query for exactly that reason.
fn is_streamable_path(forward_path: &str) -> bool {
    forward_path.ends_with("/chat/completions")
}

/// Splice `stream_options.include_usage` into a buffered OpenAI chat body when it streams without it
/// (see `peek::plan_stream_usage_injection`); otherwise return it unchanged. This is what guarantees
/// a usage chunk — hence a billable token count — from a stock client that never set the option.
fn maybe_inject_stream_usage(body: Vec<u8>) -> Vec<u8> {
    match peek::plan_stream_usage_injection(&body) {
        Some(at) => {
            const FRAG: &[u8] = br#""stream_options":{"include_usage":true},"#;
            let mut out = Vec::with_capacity(body.len() + FRAG.len());
            out.extend_from_slice(&body[..at]);
            out.extend_from_slice(FRAG);
            out.extend_from_slice(&body[at..]);
            out
        }
        None => body,
    }
}

#[async_trait]
impl ProxyHttp for AiProxy {
    type CTX = Option<RequestCtx>;

    fn new_ctx(&self) -> Self::CTX {
        None
    }

    async fn request_filter(&self, session: &mut Session, ctx: &mut Self::CTX) -> Result<bool> {
        self.state.metrics.requests_total.inc();
        let start = Instant::now();
        // One id per request, generated before any reject path so even a 400/401 carries it (in the
        // log line and the `x-beyond-request-id` header). Moved into `ctx` at the end for the
        // admitted path. Cheap: a counter bump + a short `format!` (see `next_request_id`).
        let request_id = self.state.next_request_id();

        // 1. Route by the **first path segment** = provider; forward the rest of the path verbatim
        // (native passthrough — the gateway holds no per-provider mount knowledge). A path with no
        // provider segment that is exactly `/v1` or starts with `/v1/` (boundary-checked — see
        // `bare_default_provider_name`/`route::is_default_prefix`, not a raw prefix test) is the
        // drop-in default: dialect picks openai/anthropic and the path is forwarded as-is. Anything
        // else → unknown provider (404). We resolve before auth (an unknown route is cheap) and
        // compute owned values inside the block so the session borrow ends before any `&mut session`
        // reject below.
        // `forward_streamable` is computed here, on the forwarded **path**, deliberately *before* the
        // query string is appended: `is_streamable_path` matches by suffix, so testing it against a
        // path+query would fail for every provider that requires a query parameter — Azure OpenAI
        // mandates `?api-version=…`, so its managed streams would silently skip `stream_options`
        // injection, emit no usage chunk, and bill zero tokens.
        let (provider_opt, forward_path, forward_streamable) = {
            let uri = &session.req_header().uri;
            let path = uri.path();
            let query = uri.query();
            // `nth(1)`: `/openai/v1/…` → "openai"; `/v1/…` → "v1"; "/" or "" → "".
            let first = path.split('/').nth(1).unwrap_or("");
            let with_query = |p: &str| match query {
                Some(q) => format!("{p}?{q}"),
                None => p.to_string(),
            };
            if let Some(p) = self.state.provider(first) {
                // Provider-prefixed: strip the leading `/{first}` segment, forward the remainder.
                let rest = &path[1 + first.len()..];
                let rest = if rest.is_empty() { "/" } else { rest };
                (Some(p.clone()), with_query(rest), is_streamable_path(rest))
            } else if let Some(name) = bare_default_provider_name(path) {
                // Bare default: dialect picks the provider; forward the path unchanged.
                (
                    self.state.provider(name).cloned(),
                    with_query(path),
                    is_streamable_path(path),
                )
            } else {
                (None, String::new(), false)
            }
        };
        let Some(provider) = provider_opt else {
            return Self::reject(
                session,
                &request_id,
                404,
                "invalid_request_error",
                "unknown provider",
            )
            .await;
        };
        // Dialect now comes from the resolved provider (usage parsing + injection eligibility).
        let dialect = provider.dialect;

        // 2. Extract the presented key — a managed virtual key (`bai_…`) or a raw BYO provider token.
        let Some(raw_key) = extract_virtual_key(session.req_header()) else {
            return Self::reject(
                session,
                &request_id,
                401,
                "authentication_error",
                "missing API key",
            )
            .await;
        };

        // 3. Rate guardrails (see `ratelimit`), charged on the *raw presented key* **before** any
        // verification or upstream connect. Keying on the credential we already hold (rather than the
        // verified tenant id) is what lets this sit ahead of the Ed25519 verify: a single leaked,
        // runaway, or forged key can't drive unbounded crypto work (per-credential tier), and a flood
        // of distinct random BYO tokens can't drive junk-auth connects to providers from our egress
        // IPs (global BYO tier — managed traffic is exempt, see `ratelimit`). The `check` borrow of
        // `raw_key` ends as the call returns, so the `&mut session` reject is free to run on the
        // over-limit path (where `raw_key` is unused afterward).
        if let Some(rl) = &self.state.rate_limit {
            if let Some(reason) = rl.check(raw_key, raw_key.starts_with("bai_")) {
                self.state
                    .metrics
                    .rejections_total
                    .with_label_values(&[reason.label()])
                    .inc();
                return Self::reject(
                    session,
                    &request_id,
                    429,
                    "rate_limit_error",
                    "rate limit exceeded",
                )
                .await;
            }
        }

        // 4. Reject oversized bodies up front (Content-Length) so we never buffer a huge upload.
        let declared_len = session
            .req_header()
            .headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok());
        if let Some(len) = declared_len {
            if len > MAX_REQUEST_BODY {
                return Self::reject(
                    session,
                    &request_id,
                    413,
                    "invalid_request_error",
                    "request body too large",
                )
                .await;
            }
        }

        // 5. Identity + key handling. `bai_…` → managed (stateless verify → deny-check → swap to the
        // pool key). Anything else → BYO: the user's own provider token, passed through unchanged
        // (no Beyond identity, so no deny-set and no per-tenant attribution).
        let (tenant_id, vpc_id, managed) = if raw_key.starts_with("bai_") {
            let Ok(identity) = self.state.keyring.verify(raw_key) else {
                self.state
                    .metrics
                    .rejections_total
                    .with_label_values(&["auth"])
                    .inc();
                return Self::reject(
                    session,
                    &request_id,
                    401,
                    "authentication_error",
                    "invalid API key",
                )
                .await;
            };
            // Deny-set: O(1), default-allow. The gateway never learns *why*, only the reason code.
            if let Some(reason) = self.state.deny.load().reason(identity.tenant_id) {
                // Distinct label per reason — `Unknown` is *not* folded into `deny_fraud`. An
                // `Unknown` arises when the control plane writes a reason string this gateway
                // doesn't recognize (a control-plane deploy ahead of a gateway deploy), which would
                // otherwise spike the fraud counter and mask the real fraud signal. A `deny_unknown`
                // label surfaces it as the deployment-coordination issue it is.
                let label = match reason {
                    crate::deny::DenyReason::Spend => "deny_spend",
                    crate::deny::DenyReason::Fraud => "deny_fraud",
                    crate::deny::DenyReason::Unknown => "deny_unknown",
                };
                self.state
                    .metrics
                    .rejections_total
                    .with_label_values(&[label])
                    .inc();
                return Self::reject(
                    session,
                    &request_id,
                    reason.http_status(),
                    "access_denied",
                    "tenant is over limit or suspended",
                )
                .await;
            }
            // The actual `Bearer …`/`x-api-key` value is precomputed in the provider registry and
            // applied in `upstream_request_filter`; here we only confirm a pool key exists.
            if provider.pool_auth_value.is_none() {
                return Self::reject(
                    session,
                    &request_id,
                    503,
                    "api_error",
                    "no provider key available",
                )
                .await;
            }
            (identity.tenant_id, identity.vpc_id, true)
        } else {
            (0, 0, false)
        };

        // Mark OpenAI managed chat/completions streams for body buffering + `stream_options` injection
        // (handled in `request_body_filter`). Scoped tight: managed only (BYO stays pure
        // passthrough), OpenAI dialect only, streaming-capable paths only — so everything else still
        // streams through untouched. Checked on the forwarded path (suffix), so it's prefix-agnostic.
        let inject_eligible = managed && dialect == Dialect::OpenAi && forward_streamable;

        // Circuit breaker (per provider, all traffic — a down provider is down regardless of whose
        // key is used). Checked here, after every other rejection, so claiming a half-open probe
        // permit corresponds to an *actual* upstream attempt — and balanced by exactly one
        // `record_*` in `logging` (which runs once per admitted request), so a permit can't leak.
        // When open, fast-fail 503 instead of piling the request against `read_timeout_secs` and
        // exhausting connection/in-flight slots for every provider. 5xx/connect failures trip it;
        // 429 never does (that's a healthy provider throttling — see `logging`).
        if let Some(breaker) = &provider.breaker {
            if breaker.allow().is_err() {
                self.state
                    .metrics
                    .rejections_total
                    .with_label_values(&["circuit_open"])
                    .inc();
                return Self::reject(
                    session,
                    &request_id,
                    503,
                    "api_error",
                    "provider temporarily unavailable",
                )
                .await;
            }
        }

        *ctx = Some(RequestCtx {
            tenant_id,
            vpc_id,
            dialect,
            provider,
            forward_path,
            managed,
            model: String::new(),
            model_scanner: peek::ModelScanner::new(),
            resp_model_scanner: peek::ModelScanner::new(),
            streaming: false,
            inject_eligible,
            // Only the inject-eligible path ever buffers the request body (to splice
            // `stream_options` after the root `{`; the `stream` key can appear anywhere in the root
            // object, so the decision needs the whole body — buffering is inherent here, not
            // incidental). When it does, pre-size from the declared Content-Length so accumulation is
            // a single allocation instead of a geometric realloc chain; capped at `MAX_REQUEST_BODY`
            // so a lying header can't pre-allocate unbounded memory. Every other request leaves this
            // empty and never buffers.
            req_buf: match (inject_eligible, declared_len) {
                (true, Some(len)) => Vec::with_capacity(len.min(MAX_REQUEST_BODY)),
                _ => Vec::new(),
            },
            // Grown lazily by the response tap (`response_body_filter`), not pre-reserved: a
            // non-streaming response — the common case — is a few hundred bytes, so reserving the
            // full 64KB cap up front would waste an allocation on every request to hold ~200B. A
            // long stream grows it geometrically to the bounded 2×cap and compacts; that handful of
            // reallocs is lost in the network noise of a stream we're already relaying chunk by chunk.
            resp_tail: Vec::new(),
            // Grown lazily, and only on the one path that needs it (Anthropic SSE) — see
            // `response_body_filter`. Every other response leaves this empty and never allocates.
            resp_head: Vec::new(),
            body_bytes_fed: 0,
            upstream_status: None,
            start,
            attempt: 0,
            request_id,
        });
        // Admitted: count it in-flight. Balanced by the decrement in `logging`, which runs exactly
        // once per admitted request (rejected requests leave `ctx` None and never reach that path,
        // so the gauge can't leak). `active_streams` only covers SSE; this covers every request.
        self.state.metrics.requests_in_flight.inc();
        Ok(false)
    }

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>> {
        // `ctx` is set by `request_filter` for every admitted request; a missing ctx here means an
        // unadmitted request reached `upstream_peer` (a Pingora ordering change or future refactor).
        // Surface it as an error rather than panicking the worker.
        let Some(rc) = ctx.as_ref() else {
            return Err(pingora_core::Error::new_str(
                "upstream_peer reached without request context",
            ));
        };

        // Resolve via the TTL cache (async, non-blocking) rather than `HttpPeer::new`'s eager
        // blocking `getaddrinfo`. SNI/Host = the configured host; TLS on for real providers (the
        // e2e harness flips `upstream_tls=false` for a plaintext mock).
        let addr = match self.state.resolve(&rc.provider.authority).await {
            Ok(a) => a,
            Err(e) => {
                // DNS failures are rare and usually mean a misconfigured `provider_authorities`
                // override — so keep the diagnostic (provider name + authority + the resolver error,
                // already formatted into `e`) instead of discarding it behind an opaque static string.
                // `error_because` chains `e` as the cause so it shows in the Pingora error log.
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.provider.name.as_str(),
                    authority = rc.provider.authority.as_str(),
                    error = %e,
                    "upstream dns resolution failed",
                );
                return Err(pingora_core::Error::because(
                    pingora_core::ErrorType::ConnectError,
                    "upstream dns resolution failed",
                    e,
                ));
            }
        };
        let mut peer = HttpPeer::new(
            addr,
            self.state.config.upstream_tls,
            rc.provider.host.clone(),
        );
        // Prefer HTTP/2 to the provider (config `upstream_http2`, default on), fall back to HTTP/1.1.
        // Every provider in `KNOWN_PROVIDERS` negotiates `h2` over TLS (verified by handshake), and H2
        // multiplexes many concurrent requests/streams over one connection — fewer sockets and TLS
        // handshakes from our egress IPs (which also eases the egress-reputation pressure `ratelimit`
        // guards). `H2H1` is strictly ≥ `H1` on compatibility: ALPN negotiates down to H1 for any host
        // that doesn't offer h2, and a plaintext upstream (the mock, `upstream_tls=false`) has no ALPN
        // at all and stays H1. The negotiated protocol is then visible per-request as
        // `upstream_request.version` (see `upstream_request_filter`), which is what lets the
        // body-injection path frame correctly. The knob lets an operator force all-H1 without a code
        // redeploy, and lets the e2e bench compare the two head-to-head.
        peer.options.alpn = if self.state.config.upstream_http2 {
            ALPN::H2H1
        } else {
            ALPN::H1
        };
        // Cert verification is on everywhere except the bench's self-signed TLS mock (see config).
        if !self.state.config.upstream_verify_cert {
            peer.options.verify_cert = false;
            peer.options.verify_hostname = false;
        }
        peer.options.connection_timeout =
            Some(Duration::from_secs(self.state.config.connect_timeout_secs));
        peer.options.read_timeout = Some(Duration::from_secs(self.state.config.read_timeout_secs));
        peer.options.write_timeout =
            Some(Duration::from_secs(self.state.config.write_timeout_secs));
        peer.options.idle_timeout = Some(Duration::from_secs(self.state.config.idle_timeout_secs));
        Ok(Box::new(peer))
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream_request: &mut pingora::http::RequestHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Some(rc) = ctx.as_ref() else {
            return Ok(());
        };

        // Managed: swap the virtual key for the real pool key (precomputed at boot) in the scheme
        // the upstream wants — removing every inbound static-key header first (see
        // `STATIC_KEY_HEADERS`) so the virtual key never leaks upstream, and so a provider whose own
        // auth header (e.g. Azure's `api-key` — Task #8) happens to be the same header the client
        // presented its virtual key in doesn't end up with two stacked values. BYO (`!managed`):
        // leave the user's own auth header exactly as presented.
        if rc.managed {
            if let Some(av) = &rc.provider.pool_auth_value {
                upstream_request.remove_header("authorization");
                for header in STATIC_KEY_HEADERS {
                    upstream_request.remove_header(header);
                }
                upstream_request.insert_header(rc.provider.auth.header(), av.expose())?;
            }
        }

        // Point Host at the upstream.
        upstream_request.insert_header("host", rc.provider.host.as_str())?;

        // Dashboard-attribution headers (OpenRouter, managed traffic only — Task #22, see
        // `apply_provider_attribution`).
        apply_provider_attribution(upstream_request, rc.provider.name.as_str(), rc.managed)?;

        // Forward the provider-native path (computed in `request_filter`): the client path with the
        // `/{provider}` segment stripped, or unchanged for a bare-path default. We send it verbatim —
        // no per-provider rewriting. Only set the URI when it actually differs from the inbound path
        // (i.e. a `/{provider}` prefix was stripped); the bare-path case needs no change, so we skip
        // the parse + realloc. The body's framing (Content-Length / chunked) is preserved.
        if rc.forward_path
            != upstream_request
                .uri
                .path_and_query()
                .map(|pq| pq.as_str())
                .unwrap_or("")
            && let Ok(uri) = rc.forward_path.parse()
        {
            upstream_request.set_uri(uri);
        }

        // Injection-eligible (OpenAI managed stream): the body is rewritten in `request_body_filter`,
        // changing its length, and we can't know the new length here (headers go out before the body
        // filter runs). So drop the client's `Content-Length`; how the now-unknown length is framed
        // depends on the **negotiated upstream protocol**, which is reliably readable here as
        // `upstream_request.version`: pingora-proxy sets it to HTTP/2 before this filter on the H2 path
        // (`proxy_h2.rs`) and to HTTP/1.1 on the H1 path (`proxy_h1.rs`).
        //
        //   - **H1**: a body with neither `content-length` nor `transfer-encoding` is framed as
        //     *zero-length* by pingora's H1 client (RFC 9112 §6.3) — the injected body would be
        //     silently dropped. So we must set `transfer-encoding: chunked`.
        //   - **H2**: bodies are delimited by `END_STREAM`, and `transfer-encoding` is a forbidden
        //     connection-specific header — the `h2` crate *rejects the whole request*
        //     (`UserError::MalformedHeaders`) if it's present. So we must NOT set it; removing
        //     `content-length` is sufficient and correct.
        if rc.inject_eligible {
            upstream_request.remove_header("content-length");
            if upstream_request.version != http::Version::HTTP_2 {
                upstream_request.insert_header("transfer-encoding", "chunked")?;
            }
        }
        Ok(())
    }

    async fn request_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Some(rc) = ctx.as_mut() else {
            return Ok(());
        };
        // Feed the body through the structural scanner as it passes (never withheld, never
        // buffered) to extract the exact root-level `model`. Body framing is untouched.
        if let Some(chunk) = body.as_ref() {
            // Enforce the body cap on the *streamed* size too: the up-front `Content-Length` check in
            // `request_filter` can't see a chunked-encoded body (no declared length). We don't buffer
            // — we just count — and abort the proxied request once the running total crosses the cap.
            // Aborting (vs. a clean 413) is acceptable here: headers are already away to the upstream,
            // and this is an abuse guard, not a normal client path.
            rc.body_bytes_fed = rc.body_bytes_fed.saturating_add(chunk.len());
            if rc.body_bytes_fed > MAX_REQUEST_BODY {
                self.state
                    .metrics
                    .rejections_total
                    .with_label_values(&["body_too_large"])
                    .inc();
                return Err(pingora_core::Error::new_str("request body exceeds limit"));
            }
            rc.model_scanner.feed(chunk);
            // Eligible requests are buffered so we can splice the root object before any byte reaches
            // the upstream (injection inserts near the front, so we can't have forwarded it already).
            if rc.inject_eligible {
                rc.req_buf.extend_from_slice(chunk);
            }
        }

        if rc.inject_eligible {
            if end_of_stream {
                // Emit the whole (possibly rewritten) body in one shot; `transfer-encoding: chunked`
                // (set in `upstream_request_filter`) makes the changed length fine.
                let buf = std::mem::take(&mut rc.req_buf);
                *body = Some(Bytes::from(maybe_inject_stream_usage(buf)));
            } else {
                // Withhold — the bytes are buffered above; nothing goes upstream until end-of-stream.
                // Use an *empty* chunk, not `None`: pingora derives end-of-body as
                // `end_of_body || data.is_none()` (proxy_h1.rs / proxy_h2.rs), so withholding with
                // `None` would signal end-of-body on the *first* withheld chunk and forward a truncated
                // (empty) body — silently dropping every request body that spans more than one chunk.
                // An empty `Some` is recognized as "nothing to write yet" without ending the body.
                *body = Some(Bytes::new());
            }
        }

        if end_of_stream && rc.model.is_empty() {
            if let Some(m) = rc.model_scanner.take_model() {
                rc.model = sanitize_model(m).into_owned();
            }
        }
        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        if let Some(rc) = ctx.as_mut() {
            // Headers arrived ≈ time-to-first-byte. Per-provider handle resolved once at boot (see
            // `ProviderMetrics`) — first-token latency is per-provider, so an unlabeled histogram
            // can't tell you which one regressed.
            rc.provider
                .metrics
                .ttft_seconds
                .observe(rc.start.elapsed().as_secs_f64());

            // Per-provider response counter, bucketed by status class — the signal that a provider
            // is degrading (429/5xx) before it shows up only as latency or a missing usage event.
            let status = upstream_response.status.as_u16();
            rc.provider.metrics.record_response(status);
            // Remember the status for the circuit-breaker outcome resolved in `logging` (a response
            // arrived, so the provider is reachable — even a 429/5xx is a real answer, not a connect
            // failure). `logging` decides failure-vs-success from this.
            rc.upstream_status = Some(status);

            // Derive streaming from the response, not the request: SSE ⇒ use the streaming usage
            // parser; otherwise the body is a single JSON object.
            rc.streaming = upstream_response
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|ct| ct.contains("event-stream"));
            // Track concurrent SSE streams. Incremented here (response head is in), decremented in
            // `logging` once the stream completes — so the gauge reflects in-flight streams, not a
            // counter that only ever climbs. Non-streaming responses don't touch it.
            if rc.streaming {
                self.state.metrics.active_streams.inc();
            }

            // Echo the request id so a client (or an oncall reading a captured response) can quote it
            // and land on this request's log line. `insert_header` only fails on an invalid value;
            // our id is `[0-9a-f-]`, always valid — but surface a failure rather than silently drop.
            upstream_response.insert_header(REQUEST_ID_HEADER, rc.request_id.as_str())?;
        }
        Ok(())
    }

    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<Option<Duration>>
    where
        Self::CTX: Send + Sync,
    {
        // Passive tap: copy each chunk into a bounded tail for usage parsing, but never withhold it
        // — chunks pass straight through, so the stream is relayed with no added buffering.
        //
        // We let the tail grow to 2× the cap, then compact once with a single `copy_within` that
        // keeps the last cap bytes. This bounds memory the same way the old per-chunk `drain` did,
        // but moves bytes O(stream_len / cap) times instead of once per chunk — for a long stream of
        // small chunks that's the difference between one memmove per 64 KB and one per chunk.
        if let (Some(rc), Some(chunk)) = (ctx.as_mut(), body.as_ref()) {
            // Tap the provider-reported (resolved/billed) model from the response *head* — the
            // scanner stops at the first root `model`, so this is O(1) and cheap (it finds the model
            // in the first chunk and ignores the rest). Kept separate from the tail because the model
            // is at the start of the response while the usage event is at the end.
            rc.resp_model_scanner.feed(chunk);

            // Anthropic SSE only: keep a bounded head so `message_start`'s input + cache token
            // counts survive the tail's compaction. Bounded by `USAGE_HEAD_CAP` and satisfied
            // within the first chunk or two, after which the length check makes this a no-op — so
            // it costs one small allocation on the one path that needs it, and nothing anywhere
            // else. See `USAGE_HEAD_CAP` for why only this dialect needs it.
            if rc.streaming
                && rc.dialect == Dialect::Anthropic
                && rc.resp_head.len() < USAGE_HEAD_CAP
            {
                let want = USAGE_HEAD_CAP - rc.resp_head.len();
                rc.resp_head
                    .extend_from_slice(&chunk[..want.min(chunk.len())]);
            }

            rc.resp_tail.extend_from_slice(chunk);
            if rc.resp_tail.len() > 2 * USAGE_TAIL_CAP {
                let keep_from = rc.resp_tail.len() - USAGE_TAIL_CAP;
                rc.resp_tail.copy_within(keep_from.., 0);
                rc.resp_tail.truncate(USAGE_TAIL_CAP);
            }
        }
        Ok(None)
    }

    fn fail_to_connect(
        &self,
        _session: &mut Session,
        _peer: &HttpPeer,
        ctx: &mut Self::CTX,
        mut e: Box<pingora_core::Error>,
    ) -> Box<pingora_core::Error> {
        if let Some(rc) = ctx.as_mut() {
            // Retry transient connect failures a couple of times (Pingora re-invokes upstream_peer).
            if rc.attempt < MAX_CONNECT_RETRIES {
                rc.attempt += 1;
                // Surface the retry. Without this, a partially-down provider TCP layer (or an
                // egress-IP ban — connect is where that first bites) shows up only as extra latency
                // on `upstream_latency_seconds`, indistinguishable from a slow model. The counter is
                // the dashboard signal; the `warn!` carries the request_id to grep.
                rc.provider.metrics.connect_retries_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.provider.name.as_str(),
                    attempt = rc.attempt,
                    error = %e,
                    "upstream connect failed; retrying",
                );
                e.set_retry(true);
            }
        }
        e
    }

    async fn logging(
        &self,
        _session: &mut Session,
        e: Option<&pingora_core::Error>,
        ctx: &mut Self::CTX,
    ) {
        let Some(rc) = ctx.as_mut() else { return };

        // Balance the in-flight gauge incremented at admission. `logging` runs exactly once per
        // admitted request — including on upstream errors and client disconnects — so the gauge
        // always returns to baseline and can't drift upward.
        self.state.metrics.requests_in_flight.dec();

        // An upstream error (DNS/connect timeout, read timeout, abort) lands here with `Some(e)` but
        // no `ai.usage` row (no parseable body) — and the earlier `warn!` in `upstream_peer` only
        // fires for DNS, not connect/read failures. Log it with the full identity so "why did tenant
        // 42 get 502s for 5 minutes" is one grep on the request_id, not a reconstruction.
        if let Some(e) = e {
            warn!(
                request_id = %rc.request_id,
                tenant_id = rc.tenant_id,
                vpc_id = rc.vpc_id,
                provider = rc.provider.name.as_str(),
                error = %e,
                "upstream request errored",
            );
        }

        // Resolve the circuit-breaker outcome exactly once per admitted request (every request that
        // claimed a permit in `request_filter` records here, so a half-open probe permit can't leak).
        // Failure = the provider is *broken*: a 5xx response, or no response at all paired with an
        // upstream error (connect/read failure). Success = the provider *answered* — 2xx/3xx, and
        // deliberately **4xx/429 too**: a 429 is a healthy provider throttling our pool key, which the
        // rate limiter and the client's `Retry-After` own, NOT a reason to cut all traffic to it.
        if let Some(breaker) = &rc.provider.breaker {
            match rc.upstream_status {
                Some(s) if s >= 500 => breaker.record_failure(),
                Some(_) => breaker.record_success(),
                None if e.is_some() => breaker.record_failure(),
                // No response and no error ⇒ client went away before the upstream answered; don't
                // blame the provider — record success so the probe permit resolves.
                None => breaker.record_success(),
            }
        }

        // The buffer may transiently hold up to 2× the cap before compaction; the usage event is
        // always in the last cap bytes, so slice to that bounded tail before parsing.
        let tail_start = rc.resp_tail.len().saturating_sub(USAGE_TAIL_CAP);
        let tail = &rc.resp_tail[tail_start..];

        // Extract usage facts (shape depends on dialect + streaming). Every case reads the tail;
        // Anthropic streaming *additionally* reads the head, because that's where `message_start`
        // put the input and cache token counts. The two buffers may overlap on a short response —
        // harmless, since every field is assigned rather than accumulated.
        let parsed = match (rc.dialect, rc.streaming) {
            (Dialect::OpenAi, true) => usage::openai_stream(tail),
            (Dialect::OpenAi, false) => usage::openai_body(tail),
            (Dialect::Anthropic, true) => usage::anthropic_stream_parts(&[&rc.resp_head, tail]),
            (Dialect::Anthropic, false) => usage::anthropic_body(tail),
        };
        // A managed 2xx response is *expected* to carry usage; `None` there means the provider's
        // usage block changed shape (a new API version, a wire change) and we're about to emit a
        // zero-token billing row that looks exactly like a (non-existent) legitimate zero-token
        // generation — silently zeroing that tenant's bill. Surface it on a counter + a warn so it
        // can be alerted on. A `None` on a 4xx/5xx (error body has no usage) is normal, not logged.
        if parsed.is_none()
            && rc.managed
            && let Some(s) = rc.upstream_status
            && (200..300).contains(&s)
        {
            self.state.metrics.usage_parse_errors_total.inc();
            warn!(
                request_id = %rc.request_id,
                tenant_id = rc.tenant_id,
                provider = rc.provider.name.as_str(),
                dialect = ?rc.dialect,
                stream = rc.streaming,
                status = s,
                "managed 2xx response carried no parseable usage; emitting a zero-token billing row",
            );
        }
        let usage = parsed.unwrap_or_default();

        let m = &self.state.metrics;
        // Pre-resolved fixed-label children (see `Metrics`) — no per-call `with_label_values` lookup.
        m.tokens_input.inc_by(usage.input_tokens);
        m.tokens_output.inc_by(usage.output_tokens);
        // Cache tokens, too — these are in the `ai.usage` billing log below, but that ships with lag;
        // the counter is the alerting surface for a cache-hit-rate cliff after a deploy.
        m.tokens_cache_read.inc_by(usage.cache_read_tokens);
        m.tokens_cache_write.inc_by(usage.cache_write_tokens);
        rc.provider
            .metrics
            .upstream_latency_seconds
            .observe(rc.start.elapsed().as_secs_f64());
        // Balance the `active_streams` increment from `response_filter`. `logging` runs exactly once
        // per request (including on upstream errors / client disconnects), so a stream that opened is
        // always accounted closed here — the gauge can't leak upward.
        if rc.streaming {
            m.active_streams.dec();
        }

        // Emit the usage *fact* on a dedicated target — **managed only**. The event is an
        // identity-keyed billing record (logfwd/OTLP ships `ai.usage` → ClickHouse → a closed
        // pricing consumer); BYO carries no Beyond identity, so a BYO event would be a billing row
        // with `tenant_id=0` — unbillable, unattributable, and a footgun for any consumer that sums
        // without filtering it out. Aggregate gateway throughput (incl. BYO) is already covered by
        // the Prometheus metrics above, which is the right tool for non-billing observability.
        if rc.managed {
            // Emit BOTH models. `model` is the one the *provider* resolved + billed (echoed in its
            // response) — the key for pricing AND for reconciling against the provider's invoice,
            // which itemizes by the pinned snapshot. `requested_model` is the alias the client sent —
            // product analytics ("what they asked for") and a fallback rate when a snapshot is newer
            // than the downstream price table. They're equal when the response carried no model (e.g.
            // an error body), where `model` falls back to the request alias. Both sanitized.
            let billed = rc.resp_model_scanner.take_model().map(sanitize_model);
            // Borrow the requested model as the fallback rather than cloning it — it's still read as
            // `requested_model` below, so a clone would be pure waste on every managed response.
            let billed_model = billed.as_deref().unwrap_or(&rc.model);
            info!(
                target: "ai.usage",
                request_id = %rc.request_id,
                tenant_id = rc.tenant_id,
                vpc_id = rc.vpc_id,
                provider = rc.provider.name.as_str(),
                model = billed_model,
                requested_model = %rc.model,
                stream = rc.streaming,
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                cache_read_tokens = usage.cache_read_tokens,
                cache_write_tokens = usage.cache_write_tokens,
                // `Some(0)` (reported, none used) vs `None` (not reported at all — an unreasoning
                // model, or a provider that doesn't surface it) matters and is unrecoverable once this
                // line ships, so it's logged as `?` (Debug) rather than collapsed to a bare `0`.
                reasoning_tokens = ?usage.reasoning_tokens,
                latency_ms = rc.start.elapsed().as_millis() as u64,
                "usage"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `RequestHeader` for a given path (+ optional query) with the given headers set —
    /// mirrors the shape `session.req_header()` hands `extract_virtual_key` on the real path.
    fn req_with_headers(
        path: &str,
        headers: &[(&'static str, &'static str)],
    ) -> pingora::http::RequestHeader {
        let mut req =
            pingora::http::RequestHeader::build(http::Method::POST, path.as_bytes(), None).unwrap();
        for (k, v) in headers {
            req.insert_header(*k, *v).unwrap();
        }
        req
    }

    #[test]
    fn extract_virtual_key_recognizes_anthropic_x_api_key() {
        let req = req_with_headers("/v1/messages", &[("x-api-key", "sk-ant-key")]);
        assert_eq!(extract_virtual_key(&req), Some("sk-ant-key"));
    }

    #[test]
    fn extract_virtual_key_recognizes_openai_bearer() {
        let req = req_with_headers(
            "/v1/chat/completions",
            &[("authorization", "Bearer sk-openai-key")],
        );
        assert_eq!(extract_virtual_key(&req), Some("sk-openai-key"));
    }

    #[test]
    fn extract_virtual_key_recognizes_azure_api_key_header() {
        // Task #31: Azure OpenAI authenticates via the bare `api-key` header (no `Bearer` prefix, no
        // OAuth). Before this fix, a client presenting only this header got a 401.
        let req = req_with_headers("/v1/responses", &[("api-key", "azure-secret")]);
        assert_eq!(extract_virtual_key(&req), Some("azure-secret"));
    }

    #[test]
    fn extract_virtual_key_recognizes_google_goog_api_key_header() {
        // Task #31: Google Gemini authenticates via `x-goog-api-key`.
        let req = req_with_headers(
            "/v1beta/models/gemini-2.5-pro:generateContent",
            &[("x-goog-api-key", "goog-secret")],
        );
        assert_eq!(extract_virtual_key(&req), Some("goog-secret"));
    }

    #[test]
    fn extract_virtual_key_recognizes_google_key_query_param() {
        // Task #31: Google Gemini also accepts the key as a `?key=` query param — no header at all.
        let req = req_with_headers(
            "/v1beta/models/gemini-2.5-pro:generateContent?key=goog-query-secret",
            &[],
        );
        assert_eq!(extract_virtual_key(&req), Some("goog-query-secret"));
    }

    #[test]
    fn extract_virtual_key_query_param_only_used_as_last_resort() {
        // A header takes precedence over a `key=` query param if both are somehow present.
        let req = req_with_headers(
            "/v1beta/models/gemini-2.5-pro:generateContent?key=query-secret",
            &[("x-goog-api-key", "header-secret")],
        );
        assert_eq!(extract_virtual_key(&req), Some("header-secret"));
    }

    #[test]
    fn extract_virtual_key_returns_none_when_absent() {
        let req = req_with_headers("/v1/chat/completions", &[]);
        assert_eq!(extract_virtual_key(&req), None);
    }

    #[test]
    fn query_param_finds_key_among_multiple_params() {
        assert_eq!(query_param("a=1&key=abc123&b=2", "key"), Some("abc123"));
        assert_eq!(query_param("key=solo", "key"), Some("solo"));
        assert_eq!(query_param("a=1&b=2", "key"), None);
        assert_eq!(query_param("", "key"), None);
    }

    #[test]
    fn sanitize_model_passes_real_ids() {
        for id in [
            "gpt-4o",
            "claude-opus-4-8",
            "openrouter/meta-llama/llama-3.1",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "gpt-4o-mini-2024-07-18",
        ] {
            assert_eq!(sanitize_model(id.to_string()), id);
        }
    }

    #[test]
    fn sanitize_model_rejects_json_and_log_injection() {
        // A `"` would close the JSON string; `\` could escape; a newline breaks line-oriented log
        // shipping. Any of them ⇒ recorded as "unknown" rather than injected into the billing log.
        for evil in [
            r#"real","injected":"x"#,
            r#"a\b"#,
            "line1\nline2",
            "ctrl\u{0}byte",
        ] {
            assert_eq!(sanitize_model(evil.to_string()), "unknown");
        }
    }

    #[test]
    fn sanitize_model_rejects_overlong() {
        let long = "a".repeat(MAX_MODEL_LEN + 1);
        assert_eq!(sanitize_model(long), "unknown");
        // Exactly at the cap is fine.
        let ok = "a".repeat(MAX_MODEL_LEN);
        assert_eq!(sanitize_model(ok.clone()), ok);
    }

    #[test]
    fn dialect_for_path_selects_anthropic_only_for_messages() {
        // The dialect drives usage parsing *and* stream injection: misclassifying an Anthropic
        // `/v1/messages` request as OpenAI mis-meters its tokens. The rule is a `/v1/messages`
        // prefix ⇒ Anthropic; everything else (chat completions, embeddings, the bare root) is
        // OpenAI-dialect. This locks that mapping so a refactor can't silently flip it.
        assert_eq!(dialect_for_path("/v1/messages"), Dialect::Anthropic);
        assert_eq!(dialect_for_path("/v1/messages/batches"), Dialect::Anthropic);
        assert_eq!(dialect_for_path("/v1/chat/completions"), Dialect::OpenAi);
        assert_eq!(dialect_for_path("/v1/embeddings"), Dialect::OpenAi);
        assert_eq!(dialect_for_path("/"), Dialect::OpenAi);
    }

    #[test]
    fn bare_default_provider_name_rejects_gemini_v1beta_lookalike() {
        // Task #7 (pi-parity, High): a raw `path.starts_with("/v1")` absorbed Google Gemini's real
        // path shape (`/v1beta/models/{model}:generateContent`) into the bare-default branch, which
        // then routed it to OpenAI — a silent misroute that 404s against `api.openai.com` instead of
        // failing with a clear "unknown provider" error. Boundary-checking must reject it.
        assert_eq!(
            bare_default_provider_name("/v1beta/models/gemini-2.5-pro:generateContent"),
            None,
            "/v1beta must NOT be routed to OpenAI (or any provider) via the bare-default path"
        );
        assert_eq!(bare_default_provider_name("/v1beta"), None);

        // The real bare-default shape still resolves correctly, dialect-picked.
        assert_eq!(
            bare_default_provider_name("/v1/messages"),
            Some("anthropic")
        );
        assert_eq!(
            bare_default_provider_name("/v1/chat/completions"),
            Some("openai")
        );
        assert_eq!(bare_default_provider_name("/v1"), Some("openai"));

        // Other near-miss prefixes must also be rejected, not just /v1beta.
        assert_eq!(bare_default_provider_name("/v10/messages"), None);
        assert_eq!(bare_default_provider_name("/v2/messages"), None);
    }

    #[test]
    fn is_streamable_path_matches_generation_suffixes_across_prefixes() {
        // Only chat-completions gets buffered for `stream_options.include_usage` injection. The
        // check is by *suffix* so it holds whatever mount prefix the provider uses; a mismatch here
        // either skips injection on a streamable path (lost usage) or needlessly buffers a
        // non-streaming one.
        assert!(is_streamable_path("/v1/chat/completions"));
        assert!(is_streamable_path("/openai/v1/chat/completions"));
        assert!(is_streamable_path("/inference/v1/chat/completions"));
        // The Responses API must NOT be buffered/injected: it has no `stream_options` field, always
        // reports usage on its terminal event regardless, and splicing this fragment into its body
        // would inject a field the API doesn't recognize.
        assert!(!is_streamable_path("/v1/responses"));
        // Non-streaming endpoints must not be buffered.
        assert!(!is_streamable_path("/v1/embeddings"));
        assert!(!is_streamable_path("/v1/messages"));
        assert!(!is_streamable_path("/v1/models"));
    }

    #[test]
    fn is_streamable_path_must_be_given_the_path_without_the_query() {
        // The suffix match cannot see past a query string. This locks in *why* `request_filter`
        // computes `forward_streamable` before appending the query: Azure OpenAI requires
        // `?api-version=…` on every request, and testing the path+query here returned `false` for a
        // genuine chat/completions call — so managed Azure streams skipped `stream_options`
        // injection, got no usage chunk back, and billed zero tokens with nothing logged.
        assert!(
            !is_streamable_path("/v1/chat/completions?api-version=2024-10-21"),
            "a query string defeats the suffix match — callers must strip it first"
        );
        assert!(!is_streamable_path(
            "/openai/deployments/gpt4o/chat/completions?api-version=2024-10-21"
        ));
        // ...and the same paths, query stripped, are correctly streamable.
        assert!(is_streamable_path("/v1/chat/completions"));
        assert!(is_streamable_path(
            "/openai/deployments/gpt4o/chat/completions"
        ));
    }

    #[test]
    fn openrouter_attribution_present_only_for_openrouter_managed_traffic() {
        let mut openrouter_req = pingora::http::RequestHeader::build(
            http::Method::POST,
            b"/api/v1/chat/completions",
            None,
        )
        .unwrap();
        apply_provider_attribution(&mut openrouter_req, "openrouter", true).unwrap();
        assert_eq!(
            openrouter_req.headers.get("HTTP-Referer").unwrap(),
            OPENROUTER_REFERER
        );
        assert_eq!(
            openrouter_req.headers.get("X-OpenRouter-Title").unwrap(),
            OPENROUTER_TITLE
        );
        assert_eq!(
            openrouter_req
                .headers
                .get("X-OpenRouter-Categories")
                .unwrap(),
            OPENROUTER_CATEGORY
        );

        for other in ["openai", "anthropic", "fireworks", "groq"] {
            let mut req = pingora::http::RequestHeader::build(
                http::Method::POST,
                b"/v1/chat/completions",
                None,
            )
            .unwrap();
            apply_provider_attribution(&mut req, other, true).unwrap();
            assert!(
                req.headers.get("HTTP-Referer").is_none(),
                "{other} should not get HTTP-Referer"
            );
            assert!(
                req.headers.get("X-OpenRouter-Title").is_none(),
                "{other} should not get X-OpenRouter-Title"
            );
            assert!(
                req.headers.get("X-OpenRouter-Categories").is_none(),
                "{other} should not get X-OpenRouter-Categories"
            );
        }
    }

    #[test]
    fn openrouter_attribution_gated_off_for_byo_traffic() {
        // Task #22 (pi-parity, Medium): pi gates its OpenRouter dashboard-attribution headers
        // behind a user-controllable telemetry opt-out (`isInstallTelemetryEnabled`) — before this
        // fix, this gateway injected them unconditionally, including onto a BYO caller's own
        // OpenRouter key, misattributing *their* traffic to Beyond's dashboard app. `managed=false`
        // (BYO) must suppress every one of the three headers, even for the OpenRouter provider.
        let mut byo_req = pingora::http::RequestHeader::build(
            http::Method::POST,
            b"/api/v1/chat/completions",
            None,
        )
        .unwrap();
        apply_provider_attribution(&mut byo_req, "openrouter", false).unwrap();
        assert!(
            byo_req.headers.get("HTTP-Referer").is_none(),
            "BYO OpenRouter traffic must not get HTTP-Referer"
        );
        assert!(
            byo_req.headers.get("X-OpenRouter-Title").is_none(),
            "BYO OpenRouter traffic must not get X-OpenRouter-Title"
        );
        assert!(
            byo_req.headers.get("X-OpenRouter-Categories").is_none(),
            "BYO OpenRouter traffic must not get X-OpenRouter-Categories"
        );
    }
}
