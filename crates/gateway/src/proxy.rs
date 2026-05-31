//! The Pingora `ProxyHttp` passthrough service.
//!
//! Flow: verify the virtual key (stateless) → deny-set check (O(1), default-allow) → pick the
//! provider from the ingress dialect (+ optional `x-beyond-provider` override) → swap the auth
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
//! One deliberate exception to the no-buffer rule: a **managed** OpenAI chat/responses request is
//! buffered and gets `stream_options.include_usage` injected when it streams without it — otherwise
//! OpenAI emits no usage chunk and the request couldn't be metered. We can't set that option in a
//! client SDK we don't control, so the gateway guarantees it, out of the box. Scoped to exactly that
//! path (managed + OpenAI dialect + streaming-capable); BYO and everything else stay pure passthrough.
//!
//! Auth branches on key format: `bai_…` is a managed virtual key (verify → deny-check → swap to
//! the pool key); anything else is a **BYO** request — the user's own provider token, passed
//! through unchanged (no swap, no Beyond identity, no deny-set).
//!
//! Consequence: routing is by **dialect**, not model — the body (hence model) isn't known when
//! `upstream_peer` runs. Any non-default provider is reached via the `x-beyond-provider` header
//! (providers are data — see `route`). Model is still captured (from the streamed body) for usage.

use crate::route::{self, Dialect, Provider};
use crate::state::GatewayState;
use crate::{peek, usage};
use async_trait::async_trait;
use bytes::Bytes;
use pingora::http::ResponseHeader;
use pingora_core::Result;
use pingora_core::upstreams::peer::HttpPeer;
use pingora_proxy::{ProxyHttp, Session};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

/// Reject requests whose declared Content-Length exceeds this. The body itself is **not** buffered
/// (it streams straight through); this is purely an abuse guard checked up front via the header.
const MAX_REQUEST_BODY: usize = 100 * 1024 * 1024;

/// Bounded tail of the response kept for usage extraction. The usage event is the final SSE chunk
/// / the whole non-streaming body; keeping a tail means we never buffer a long stream.
const USAGE_TAIL_CAP: usize = 64 * 1024;

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
    /// Running total of request-body bytes seen, to enforce `MAX_REQUEST_BODY` even when the client
    /// uses chunked transfer encoding (no `Content-Length` to check up front).
    body_bytes_fed: usize,
    /// Managed OpenAI chat/responses request: buffer the body and inject
    /// `stream_options.include_usage` if it streams without it, so the usage chunk (hence the
    /// billable token count) is guaranteed. The single, deliberate exception to "never buffer the
    /// request body" — scoped to the managed OpenAI streaming-capable path and bounded by
    /// `MAX_REQUEST_BODY`. BYO and every other request still stream straight through.
    inject_eligible: bool,
    /// Accumulated request body — populated only when `inject_eligible`; otherwise stays empty and
    /// the body is never buffered.
    req_buf: Vec<u8>,
    start: Instant,
    /// Connect-retry counter (see `fail_to_connect`).
    attempt: u8,
}

impl AiProxy {
    /// Write a small JSON error and signal `request_filter` to short-circuit. The body is built with
    /// `serde_json` (not `format!`) so a `typ`/`msg` containing `"` or `\` can never break out of the
    /// JSON structure — keeps this safe if a future caller passes a non-literal message.
    async fn reject(session: &mut Session, status: u16, typ: &str, msg: &str) -> Result<bool> {
        let body = Bytes::from(
            serde_json::json!({ "error": { "type": typ, "message": msg } }).to_string(),
        );
        let mut resp = ResponseHeader::build(status, None)?;
        resp.insert_header("content-type", "application/json")?;
        resp.insert_header("content-length", body.len().to_string())?;
        session.write_response_header(Box::new(resp), false).await?;
        session.write_response_body(Some(body), true).await?;
        Ok(true)
    }
}

fn extract_virtual_key(session: &Session) -> Option<&str> {
    let h = session.req_header();
    // Anthropic SDK sends `x-api-key`; OpenAI SDK sends `Authorization: Bearer`. One neutral
    // virtual key works in either, so check both. Borrowed from the header — no per-request copy.
    if let Some(v) = h.headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v);
    }
    h.headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
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
fn sanitize_model(model: String) -> String {
    let bad = model.len() > MAX_MODEL_LEN
        || model
            .bytes()
            .any(|b| b < 0x20 || b == b'"' || b == b'\\' || b == 0x7f);
    if bad { "unknown".to_string() } else { model }
}

fn dialect_for_path(path: &str) -> Dialect {
    // Anthropic Messages vs OpenAI Chat Completions/Embeddings. Embeddings are OpenAI-dialect only.
    if path.starts_with("/v1/messages") {
        Dialect::Anthropic
    } else {
        Dialect::OpenAI
    }
}

/// OpenAI **streaming-capable** endpoints: chat completions + the Responses API. These are the only
/// requests we buffer for `stream_options.include_usage` injection — embeddings and every other
/// OpenAI-dialect path never stream, so there's nothing to meter and no reason to buffer them.
fn openai_streamable_path(path: &str) -> bool {
    path.starts_with("/v1/chat/completions") || path.starts_with("/v1/responses")
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

/// The `x-beyond-provider` override value, if present — a provider *name* resolved against the
/// registry in `request_filter`. (An unknown name is rejected there, not silently ignored.)
fn provider_override(session: &Session) -> Option<&str> {
    session
        .req_header()
        .headers
        .get("x-beyond-provider")?
        .to_str()
        .ok()
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

        // 1. Resolve the upstream provider first — from the ingress dialect (the body/model isn't
        // available pre-connect), with an explicit `x-beyond-provider` override. Resolving up front
        // means an unknown provider is a clean 400 before any auth work, and (since it borrows
        // nothing) keeps the borrow checker happy when the key is extracted next. An `Arc` clone of
        // the boot-time registry entry — nothing re-allocated per request.
        let dialect = dialect_for_path(session.req_header().uri.path());
        let provider = match provider_override(session) {
            Some(name) => self.state.provider(name).cloned(),
            None => self
                .state
                .provider(route::dialect_default(dialect))
                .cloned(),
        };
        let Some(provider) = provider else {
            return Self::reject(session, 400, "invalid_request_error", "unknown provider").await;
        };

        // 2. Extract the presented key — a managed virtual key (`bai_…`) or a raw BYO provider token.
        let Some(raw_key) = extract_virtual_key(session) else {
            return Self::reject(session, 401, "authentication_error", "missing API key").await;
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
                return Self::reject(session, 429, "rate_limit_error", "rate limit exceeded").await;
            }
        }

        // 4. Reject oversized bodies up front (Content-Length) so we never buffer a huge upload.
        if let Some(len) = session
            .req_header()
            .headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok())
        {
            if len > MAX_REQUEST_BODY {
                return Self::reject(
                    session,
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
                return Self::reject(session, 401, "authentication_error", "invalid API key").await;
            };
            // Deny-set: O(1), default-allow. The gateway never learns *why*, only the reason code.
            if let Some(reason) = self.state.deny.load().reason(identity.tenant_id) {
                let label = match reason {
                    crate::deny::DenyReason::Spend => "deny_spend",
                    _ => "deny_fraud",
                };
                self.state
                    .metrics
                    .rejections_total
                    .with_label_values(&[label])
                    .inc();
                return Self::reject(
                    session,
                    reason.http_status(),
                    "access_denied",
                    "tenant is over limit or suspended",
                )
                .await;
            }
            // The actual `Bearer …`/`x-api-key` value is precomputed in the provider registry and
            // applied in `upstream_request_filter`; here we only confirm a pool key exists.
            if provider.pool_auth_value.is_none() {
                return Self::reject(session, 503, "api_error", "no provider key available").await;
            }
            (identity.tenant_id, identity.vpc_id, true)
        } else {
            (0, 0, false)
        };

        // Mark OpenAI managed chat/responses streams for body buffering + `stream_options` injection
        // (handled in `request_body_filter`). Scoped tight: managed only (BYO stays pure
        // passthrough), OpenAI dialect only, streaming-capable paths only — so everything else still
        // streams through untouched.
        let inject_eligible = managed
            && dialect == Dialect::OpenAI
            && openai_streamable_path(session.req_header().uri.path());

        *ctx = Some(RequestCtx {
            tenant_id,
            vpc_id,
            dialect,
            provider,
            managed,
            model: String::new(),
            model_scanner: peek::ModelScanner::new(),
            resp_model_scanner: peek::ModelScanner::new(),
            streaming: false,
            inject_eligible,
            req_buf: Vec::new(),
            // Grown lazily by the response tap (`response_body_filter`), not pre-reserved: a
            // non-streaming response — the common case — is a few hundred bytes, so reserving the
            // full 64KB cap up front would waste an allocation on every request to hold ~200B. A
            // long stream grows it geometrically to the bounded 2×cap and compacts; that handful of
            // reallocs is lost in the network noise of a stream we're already relaying chunk by chunk.
            resp_tail: Vec::new(),
            body_bytes_fed: 0,
            start,
            attempt: 0,
        });
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
            Err(_) => {
                return Err(pingora_core::Error::new_str(
                    "upstream dns resolution failed",
                ));
            }
        };
        let mut peer = HttpPeer::new(
            addr,
            self.state.config.upstream_tls,
            rc.provider.host.clone(),
        );
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
        // the upstream wants — removing *both* inbound auth headers first so the virtual key never
        // leaks upstream. BYO (`!managed`): leave the user's own auth header exactly as presented.
        if rc.managed {
            if let Some(av) = &rc.provider.pool_auth_value {
                upstream_request.remove_header("authorization");
                upstream_request.remove_header("x-api-key");
                upstream_request.insert_header(rc.provider.auth.header(), av.expose())?;
            }
        }

        // Point Host at the upstream. The body passes through untouched, so the client's original
        // framing (Content-Length / chunked) is preserved — true passthrough.
        upstream_request.insert_header("host", rc.provider.host.as_str())?;

        // Rewrite the path to the provider's mount point when it isn't `/v1` (e.g. Groq serves the
        // OpenAI surface under `/openai/v1`, Fireworks under `/inference/v1`). Most providers mount
        // at `/v1`, so `upstream_path` returns `None` and the URI is left untouched (no realloc).
        // The query string is preserved.
        if let Some(new_path) = rc.provider.upstream_path(upstream_request.uri.path()) {
            let pq = match upstream_request.uri.query() {
                Some(q) => format!("{new_path}?{q}"),
                None => new_path,
            };
            if let Ok(uri) = pq.parse() {
                upstream_request.set_uri(uri);
            }
        }

        // Injection-eligible (OpenAI managed stream): the body is rewritten in `request_body_filter`,
        // changing its length, and we can't know the new length here (headers go out before the body
        // filter runs). So drop the client's `Content-Length` and frame the buffered body as chunked.
        if rc.inject_eligible {
            upstream_request.remove_header("content-length");
            upstream_request.insert_header("transfer-encoding", "chunked")?;
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
                *body = None;
            }
        }

        if end_of_stream && rc.model.is_empty() {
            if let Some(m) = rc.model_scanner.take_model() {
                rc.model = sanitize_model(m);
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
            // Headers arrived ≈ time-to-first-byte.
            self.state
                .metrics
                .ttft_seconds
                .observe(rc.start.elapsed().as_secs_f64());
            // Derive streaming from the response, not the request: SSE ⇒ use the streaming usage
            // parser; otherwise the body is a single JSON object.
            rc.streaming = upstream_response
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|ct| ct.contains("event-stream"));
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
                e.set_retry(true);
            }
        }
        e
    }

    async fn logging(
        &self,
        _session: &mut Session,
        _e: Option<&pingora_core::Error>,
        ctx: &mut Self::CTX,
    ) {
        let Some(rc) = ctx.as_mut() else { return };

        // The buffer may transiently hold up to 2× the cap before compaction; the usage event is
        // always in the last cap bytes, so slice to that bounded tail before parsing.
        let tail_start = rc.resp_tail.len().saturating_sub(USAGE_TAIL_CAP);
        let tail = &rc.resp_tail[tail_start..];

        // Extract usage facts from the tail (shape depends on dialect + streaming).
        let usage = match (rc.dialect, rc.streaming) {
            (Dialect::OpenAI, true) => usage::openai_stream(tail),
            (Dialect::OpenAI, false) => usage::openai_body(tail),
            (Dialect::Anthropic, true) => usage::anthropic_stream(tail),
            (Dialect::Anthropic, false) => usage::anthropic_body(tail),
        }
        .unwrap_or_default();

        let m = &self.state.metrics;
        m.tokens_total
            .with_label_values(&["input"])
            .inc_by(usage.input_tokens);
        m.tokens_total
            .with_label_values(&["output"])
            .inc_by(usage.output_tokens);
        m.upstream_latency_seconds
            .observe(rc.start.elapsed().as_secs_f64());

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
            let billed_model = rc
                .resp_model_scanner
                .take_model()
                .map(sanitize_model)
                .unwrap_or_else(|| rc.model.clone());
            info!(
                target: "ai.usage",
                tenant_id = rc.tenant_id,
                vpc_id = rc.vpc_id,
                provider = rc.provider.name.as_str(),
                model = %billed_model,
                requested_model = %rc.model,
                stream = rc.streaming,
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                cache_read_tokens = usage.cache_read_tokens,
                cache_write_tokens = usage.cache_write_tokens,
                latency_ms = rc.start.elapsed().as_millis() as u64,
                "usage"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
