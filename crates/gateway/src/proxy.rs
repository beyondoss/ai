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

use crate::metrics::Rejection;
use crate::route::{self, Dialect, Provider};
use crate::state::{GatewayState, RequestId};
use crate::{peek, usage};
use arrayvec::ArrayString;
use async_trait::async_trait;
use bytes::Bytes;
use pingora::http::ResponseHeader;
use pingora_core::Result;
use pingora_core::protocols::ALPN;
use pingora_core::upstreams::peer::HttpPeer;
use pingora_proxy::{ProxyHttp, Session};
use std::borrow::Cow;
use std::fmt::Write as _;
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

/// The bounded window of response bytes kept for usage extraction.
///
/// Grows like a plain `Vec` while the response is small — the common case is a non-streaming body of
/// a few hundred bytes, and reserving the full cap for that would waste an allocation on every
/// request. Once it outgrows the cap it flips to a **ring**: a cap-sized buffer written with
/// wraparound, so every subsequent byte is copied exactly once, with no compaction memmove and no
/// further allocation.
///
/// What it replaced grew to `2 × cap` and then compacted with a `copy_within` that kept the last
/// `cap` bytes. That is bounded, but it re-copies `cap` bytes every `cap` bytes of stream — so a
/// long response was memmoved roughly twice over, on top of the geometric realloc chain from
/// starting at zero capacity. Measured 1.88× the response size in memmove at steady state.
#[derive(Default)]
struct UsageTail {
    buf: Vec<u8>,
    /// Next write index. Meaningful only once `ring` is set.
    head: usize,
    /// Whether `buf` is a wraparound ring of exactly `USAGE_TAIL_CAP` bytes.
    ring: bool,
}

impl UsageTail {
    fn push(&mut self, data: &[u8]) {
        if !self.ring {
            self.buf.extend_from_slice(data);
            if self.buf.len() > USAGE_TAIL_CAP {
                // Outgrown: keep the last cap bytes in order and switch to ring mode. This is the
                // only compaction that ever runs — from here on, writes wrap instead of shifting.
                let start = self.buf.len() - USAGE_TAIL_CAP;
                self.buf.copy_within(start.., 0);
                self.buf.truncate(USAGE_TAIL_CAP);
                self.head = 0;
                self.ring = true;
            }
            return;
        }
        // A chunk at least as large as the whole window: only its last cap bytes can survive, and
        // they land aligned, so the ring resets rather than wrapping.
        if data.len() >= USAGE_TAIL_CAP {
            self.buf
                .copy_from_slice(&data[data.len() - USAGE_TAIL_CAP..]);
            self.head = 0;
            return;
        }
        let first = (USAGE_TAIL_CAP - self.head).min(data.len());
        self.buf[self.head..self.head + first].copy_from_slice(&data[..first]);
        let rest = data.len() - first;
        if rest > 0 {
            self.buf[..rest].copy_from_slice(&data[first..]);
        }
        self.head = (self.head + data.len()) % USAGE_TAIL_CAP;
    }

    /// The retained bytes, oldest first. Rotates the ring into order once, at parse time.
    ///
    /// Stays a ring afterwards, with `head` back at 0 — a subsequent `push` then overwrites the
    /// oldest bytes, which is exactly right. In practice `logging` calls this once, after the body
    /// is complete.
    fn contiguous(&mut self) -> &[u8] {
        if self.ring && self.head != 0 {
            self.buf.rotate_left(self.head);
            self.head = 0;
        }
        &self.buf
    }
}

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
    /// The path (+ query) to send upstream, when it differs from the inbound one: the client path
    /// with the `/{provider}` segment stripped. Forwarded **verbatim** — the gateway does no
    /// per-provider path rewriting. Applied as the upstream URI in `upstream_request_filter`.
    ///
    /// `None` for the bare-path default, whose path already *is* what the upstream should see. The
    /// distinction is in the type rather than discovered by rebuilding the path and comparing it,
    /// so the common route allocates nothing.
    forward_path: Option<String>,
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
    resp_tail: UsageTail,
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
    /// Whether an `allow()` on `provider`'s breaker is outstanding and still owes exactly one
    /// `record_*`.
    ///
    /// The ledger that keeps breaker accounting honest once a request can attempt more than one
    /// provider. Invariant: `breaker_pending` is true **iff** there is exactly one unresolved
    /// `allow()` against whatever `provider` currently points at. `logging` records only when it is
    /// set, so an attempt can never be recorded twice, and a candidate switch resolves the outgoing
    /// candidate before claiming the next one.
    ///
    /// On the `/{provider}/…` path this is simply `breaker.is_some()`, set once in `request_filter`
    /// — exactly the condition `logging` used to test inline — so that path's behaviour is unchanged.
    breaker_pending: bool,
    /// Model-routing state — `Some` only for `/auto`. `None` keeps every provider-routed request on
    /// exactly the code it ran before model routing existed. See [`ModelRouting`] for why it is
    /// boxed rather than inline.
    auto: Option<Box<ModelRouting>>,
    /// Process-unique id for this request (`{instance}-{seq}`), echoed in the `x-beyond-request-id`
    /// response header and the `ai.usage` event so a client report ties back to a log line.
    request_id: RequestId,
}

/// State that exists only for a **model-routed** (`/auto`) request.
///
/// Boxed, and `None` for provider-routed traffic — which is the overwhelming majority. Held inline
/// these fields added 64 bytes to `RequestCtx` (368 → 432), a struct that is touched on every hook
/// and, for a streaming response, once per response chunk. That showed up as a reproducible ~2.5%
/// regression on the `managed_sse_latency` bench, across two independent runs against the same
/// baseline, while the non-streaming case was unaffected — the signature of a per-chunk cost, not a
/// per-request one.
///
/// So the model-routed path pays one small allocation and every other request pays nothing. That is
/// the right way round: `/auto` is opt-in, and the request it serves is about to cross a network.
struct ModelRouting {
    /// The catalog row this request routes over. `&'static`, so it costs a pointer.
    route: &'static route::ModelRoute,
    /// Index into `route.candidates` of the candidate currently being attempted.
    candidate: u8,
    /// Bit `i` ⇒ `route.candidates[i]` is *usable*: this gateway routes to that provider and holds a
    /// pool key for it. Computed once in `request_filter` so `upstream_peer` never re-derives it.
    /// Bounded by [`route::MAX_CANDIDATES`], which is why a `u8` suffices.
    usable: u8,
    /// The client's path (+ query) with the `/auto` segment removed. The forwarded path is this
    /// appended to the chosen candidate's mount, rebuilt into `forward_path` on every attempt.
    suffix: String,
    /// When the current attempt began. Distinct from `RequestCtx::start` (which times the whole
    /// request) so a candidate that burned `connect_timeout_secs` before failing over does not
    /// charge that time to the provider that actually served — which would render an outage at
    /// candidate A as a latency regression at candidate B, inverting the point of the per-provider
    /// label.
    attempt_start: Instant,
}

impl RequestCtx {
    /// Clear the state the request-body phase accumulates, so a **retried** attempt starts from the
    /// same slate as the first one.
    ///
    /// Pingora replays its buffered request body through `request_body_filter` on a retry
    /// (`proxy_h1.rs`'s `send_body_to_pipe`, and the h2 twin), so without this the replayed prefix is
    /// *appended* to whatever the previous attempt already accumulated:
    ///
    /// - `req_buf` would hold the prefix twice, and `peek::scan_buffered` would then plan the splice
    ///   against the **first** copy — handing the upstream a body with a duplicated fragment, which
    ///   it rejects with a `400` that reads like a client error. `logging` even records that `400` as
    ///   a breaker *success* (the provider answered), so nothing surfaces it as our fault.
    /// - `model_scanner`'s brace depth would be permanently offset by the extra `{`, so
    ///   `at_key_level` never matches again and the billing row ships `requested_model = ""`.
    /// - `body_bytes_fed` would double-count the prefix against `MAX_REQUEST_BODY`.
    ///
    /// Called from `upstream_peer`, which pingora invokes exactly once per attempt and always before
    /// any body byte moves — it is the first statement of `proxy_to_upstream`.
    ///
    /// `model` is deliberately **not** cleared: once a complete body has yielded it the value is
    /// correct, and pingora's replay buffer is capped at 64 KiB, so a larger body could not
    /// re-derive it on the next attempt.
    /// When the current upstream attempt began: the per-attempt stamp for a model-routed request,
    /// and simply the request start for everything else — where the two are always equal anyway,
    /// so the common path stores no second `Instant`.
    fn attempt_start(&self) -> Instant {
        self.auto.as_ref().map_or(self.start, |a| a.attempt_start)
    }

    /// Move the candidate cursor past index `i`. No-op for a provider-routed request.
    fn advance_candidate(&mut self, i: u8) {
        if let Some(a) = self.auto.as_mut() {
            a.candidate = i.saturating_add(1);
        }
    }

    /// Whether this request's body is buffered and may leave with a different length than it
    /// arrived — which is one question, not two, because the answer drives *both* the buffering in
    /// `request_body_filter` and the `Content-Length`/`transfer-encoding` re-framing in
    /// `upstream_request_filter`. Splitting them is how you get a body whose framing disagrees with
    /// its bytes.
    ///
    /// Two reasons a body gets rewritten:
    /// - `inject_eligible`: splicing `stream_options` into a managed OpenAI stream.
    /// - `route`: re-spelling `model` for the candidate serving this attempt.
    ///
    /// Both can apply to the same request, in which case both edits are made to the one buffer.
    fn rewrites_body(&self) -> bool {
        self.inject_eligible || self.auto.is_some()
    }

    /// Rebuild the forwarded path for the candidate about to be attempted: its mount prefix plus the
    /// client's suffix.
    ///
    /// Only the model-routed path calls this. Rewritten in place, so a failover costs no allocation
    /// after the first attempt. A no-op for a provider-routed request, which has no `auto_suffix`
    /// and whose `forward_path` was settled once in `request_filter`.
    fn rebuild_forward_path(&mut self, base_path: &str) {
        let Some(auto) = self.auto.as_ref() else {
            return;
        };
        let buf = self.forward_path.get_or_insert_with(String::new);
        write_mounted_path(buf, base_path, &auto.suffix);
    }

    fn reset_request_body_phase(&mut self) {
        // The first attempt has nothing to undo, and that is the only attempt the vast majority of
        // requests ever make — so pay one compare rather than three stores on the hot path. A zero
        // `body_bytes_fed` is an exact witness for "no chunk was ever fed": it is incremented for
        // every chunk that reaches `request_body_filter`, on the same branch that appends to
        // `req_buf` and feeds `model_scanner`, so neither can hold state while it reads zero.
        if self.body_bytes_fed == 0 {
            return;
        }
        // `clear` keeps the capacity, which was pre-sized from `Content-Length` in `request_filter`
        // and which the in-place splice relies on to stay realloc-free.
        self.req_buf.clear();
        self.body_bytes_fed = 0;
        self.model_scanner = peek::ModelScanner::new();
    }
}

/// Every `(error_type, message)` pair the gateway rejects with, paired with its wire body.
///
/// The set is closed: `reject` is only ever called with literals, so the response body is one of
/// these constants and never needs building. Kept as a table rather than scattered `const`s so
/// `reject_bodies_are_valid_json` can walk it and assert each entry parses, carries the `type` and
/// `message` it claims, and is reachable — a hand-written JSON literal is exactly the thing that
/// rots silently otherwise.
pub const REJECT_BODIES: [(&str, &str, &str); 11] = [
    (
        "invalid_request_error",
        "unknown provider",
        r#"{"error":{"message":"unknown provider","type":"invalid_request_error"}}"#,
    ),
    (
        "authentication_error",
        "missing API key",
        r#"{"error":{"message":"missing API key","type":"authentication_error"}}"#,
    ),
    (
        "authentication_error",
        "invalid API key",
        r#"{"error":{"message":"invalid API key","type":"authentication_error"}}"#,
    ),
    (
        "rate_limit_error",
        "rate limit exceeded",
        r#"{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}"#,
    ),
    (
        "invalid_request_error",
        "request body too large",
        r#"{"error":{"message":"request body too large","type":"invalid_request_error"}}"#,
    ),
    (
        "access_denied",
        "tenant is over limit or suspended",
        r#"{"error":{"message":"tenant is over limit or suspended","type":"access_denied"}}"#,
    ),
    (
        "api_error",
        "no provider key available",
        r#"{"error":{"message":"no provider key available","type":"api_error"}}"#,
    ),
    (
        "api_error",
        "provider temporarily unavailable",
        r#"{"error":{"message":"provider temporarily unavailable","type":"api_error"}}"#,
    ),
    (
        "invalid_request_error",
        "unknown model",
        r#"{"error":{"message":"unknown model","type":"invalid_request_error"}}"#,
    ),
    (
        "invalid_request_error",
        "model routing requires a managed key",
        r#"{"error":{"message":"model routing requires a managed key","type":"invalid_request_error"}}"#,
    ),
    (
        "api_error",
        "no provider available for model",
        r#"{"error":{"message":"no provider available for model","type":"api_error"}}"#,
    ),
];

/// The precomputed body for a `(typ, msg)` pair.
///
/// Falls back to building one with `serde_json` for a pair not in [`REJECT_BODIES`]. That branch is
/// unreachable today (a test asserts every call site is covered) and exists so adding a rejection
/// without its table entry degrades to the old allocating behaviour rather than serving a body that
/// contradicts the `error_type` in the log line.
/// `pub` for the bench target (`benches/unit.rs`), which measures it against the `serde_json`
/// construction it replaced. Not part of the crate's intended surface.
pub fn error_body(typ: &str, msg: &str) -> Bytes {
    for (t, m, body) in REJECT_BODIES {
        if t == typ && m == msg {
            return Bytes::from_static(body.as_bytes());
        }
    }
    Bytes::from(serde_json::json!({ "error": { "type": typ, "message": msg } }).to_string())
}

impl AiProxy {
    /// Build the upstream peer for a resolved address + provider.
    ///
    /// Extracted so the provider-routed path and the model-routed candidate walk cannot drift apart
    /// on TLS, ALPN, or timeouts — a fallback candidate connected on different terms than the
    /// primary would be a genuinely nasty thing to debug.
    fn build_peer(&self, addr: std::net::SocketAddr, provider: &Provider) -> HttpPeer {
        let mut peer = HttpPeer::new(addr, self.state.config.upstream_tls, provider.host.clone());
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
        peer
    }

    /// Write a small JSON error and signal `request_filter` to short-circuit. The body is built with
    /// `serde_json` (not `format!`) so a `typ`/`msg` containing `"` or `\` can never break out of the
    /// JSON structure — keeps this safe if a future caller passes a non-literal message.
    ///
    /// Every rejection logs one structured `warn` line (the rejection counter only says *how many*,
    /// not *which request* — this is what an oncall greps when a `deny_fraud`/`rate_limit` spike
    /// shows on the dashboard) and echoes the `request_id` in a response header so a client report
    /// quoting that id lands on this line.
    ///
    /// **Call it through [`Self::reject_boxed`], never `.await` it directly.** `#[async_trait]`
    /// heap-boxes `request_filter`'s future once per request, and this future — 1 136 bytes of it,
    /// measured with `-Zprint-type-sizes` — gets inlined into that state machine at every call site.
    /// Awaiting it inline therefore made *every* request, including every successful one, allocate
    /// room for a rejection it was never going to take.
    async fn reject(
        session: &mut Session,
        request_id: &str,
        status: u16,
        typ: &str,
        msg: &str,
    ) -> Result<bool> {
        warn!(request_id, status, error_type = typ, "request rejected");
        // `typ` and `msg` are always a pair of literals from `RejectBody`, so the body is one of a
        // handful of compile-time constants and `error_body` hands back a `Bytes::from_static` —
        // no JSON DOM, no `String`, no copy. Building it with `serde_json::json!` cost 13
        // allocations and 1 565 bytes per rejected request (measured), which is a poor trade on the
        // one path that a flood drives at full rate. The `json!` was there so a non-literal message
        // couldn't break out of the JSON structure; a closed set of constants gives that for free,
        // and `reject_bodies_are_valid_json` keeps them honest.
        let body = error_body(typ, msg);
        // Content-length formatted into a stack buffer rather than `body.len().to_string()`: still
        // one `HeaderValue` allocation inside pingora, but no `String` of our own.
        let mut len_buf = ArrayString::<20>::new();
        let _ = write!(len_buf, "{}", body.len());
        let mut resp = ResponseHeader::build(status, Some(3))?;
        resp.insert_header("content-type", "application/json")?;
        resp.insert_header("content-length", len_buf.as_str())?;
        resp.insert_header(REQUEST_ID_HEADER, request_id)?;
        session.write_response_header(Box::new(resp), false).await?;
        session.write_response_body(Some(body), true).await?;
        Ok(true)
    }

    /// [`Self::reject`] behind its own allocation, so its state machine is *not* inlined into
    /// `request_filter`'s.
    ///
    /// `request_filter` returns an `#[async_trait]` future that pingora heap-boxes once per request.
    /// With `reject` awaited inline at eight call sites, the largest of them dominated that future:
    /// 1 264 bytes total, of which 1 136 was the reject state machine (`-Zprint-type-sizes`) — for
    /// comparison every other filter's future is 32 bytes and `upstream_peer`'s is 152. A request
    /// that is never rejected still paid for it, because the box has to be big enough for the widest
    /// variant. Boxing here moves that cost onto the requests that actually reject.
    async fn reject_boxed(
        session: &mut Session,
        request_id: &str,
        status: u16,
        typ: &str,
        msg: &str,
    ) -> Result<bool> {
        Box::pin(Self::reject(session, request_id, status, typ, msg)).await
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

/// Whether an error that ended the request should count against the **provider's** circuit breaker.
///
/// `ErrorSource::Downstream` is the *client's* side failing — an aborted request, a broken pipe
/// while we write the response back. Pingora tags those explicitly (`into_down()`), and they carry
/// no information about the upstream's health, so they must not trip a breaker that exists to detect
/// a sick provider.
///
/// Everything else counts: an upstream error (pingora calls `as_up()` before `fail_to_connect`), our
/// own DNS failure returned from `upstream_peer` (which leaves the source `Unset`), or an internal
/// fault. Each is a real failure to complete this request against this provider.
fn is_upstream_failure(e: Option<&pingora_core::Error>) -> bool {
    e.is_some_and(|e| !matches!(e.esource(), pingora_core::ErrorSource::Downstream))
}

/// Write the forwarded path for a model-routed attempt: the candidate's mount prefix, then the
/// client's suffix.
///
/// Reuses `buf`'s allocation, so a failover re-derives the path without allocating again.
///
/// The contract the client sees is "point your SDK's base URL at `…/auto`": an OpenAI-wire SDK then
/// sends `/chat/completions` and gets `/v1/chat/completions` at OpenAI or `/api/v1/chat/completions`
/// at OpenRouter, while an Anthropic SDK sends `/v1/messages` and gets it unchanged, because
/// Anthropic's base URL carries no mount. That asymmetry is not invented here — it is exactly the
/// `base_url` convention already in the shared provider table.
fn write_mounted_path(buf: &mut String, base_path: &str, suffix: &str) {
    buf.clear();
    buf.push_str(base_path);
    buf.push_str(suffix);
}

/// The lowest set bit in `usable` at or after index `from`, or `None` if there is none.
///
/// The candidate walk's only cursor primitive. `from` strictly increases across a request, so the
/// walk always terminates — there is no way to revisit a candidate and claim a second breaker permit
/// against it.
fn first_usable(usable: u8, from: u8) -> Option<u8> {
    if from >= route::MAX_CANDIDATES as u8 {
        return None;
    }
    // Mask off everything below `from`, then take the lowest remaining bit.
    let remaining = usable & !((1u8 << from) - 1);
    (remaining != 0).then(|| remaining.trailing_zeros() as u8)
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

/// The fragment spliced into a streaming OpenAI chat body. Always followed by a comma, since the
/// splice point is just inside a root object that is non-empty by construction (a root `"stream"`
/// key is what made it eligible).
const STREAM_OPTIONS_FRAG: &[u8] = br#""stream_options":{"include_usage":true},"#;

/// Splice `stream_options.include_usage` into a buffered OpenAI chat body at `at`, or return it
/// unchanged when there is nothing to inject. This is what guarantees a usage chunk — hence a
/// billable token count — from a stock client that never set the option.
///
/// Takes the offset rather than computing it: the caller already walked the body once for both the
/// model and this plan (see `peek::scan_buffered`), and re-deriving it here would restore the second
/// traversal that walk exists to remove.
/// Overwrite the `model` value's bytes with the id the chosen candidate uses.
///
/// `span` comes from [`peek::BufferedScan::model_span`] and covers the raw value, quotes excluded.
/// The replacement is a catalog string, so it needs no JSON escaping — the catalog charset test
/// (`model_names_are_lowercase_and_log_safe`, plus the same shape for `upstream_model`) is what
/// makes a raw byte copy safe here.
///
/// Returns the body untouched when the id already matches, which is the common case: candidate 0
/// usually spells the model the way the catalog names it, so the primary path does no memmove at all
/// and only a failover pays for one.
fn apply_model_rewrite(mut body: Vec<u8>, span: (usize, usize), replacement: &[u8]) -> Vec<u8> {
    let (start, end) = span;
    // Defensive: a span outside the buffer would panic on the splice. Unreachable — the span is
    // produced by the same walk over the same bytes — but this runs on every model-routed request.
    if start > end || end > body.len() {
        return body;
    }
    if body[start..end] == *replacement {
        return body;
    }
    body.splice(start..end, replacement.iter().copied());
    body
}

fn apply_stream_usage_injection(mut body: Vec<u8>, at: Option<usize>) -> Vec<u8> {
    let Some(at) = at else { return body };
    // Shift the tail right in place rather than copying the whole body into a second buffer.
    // `req_buf` is pre-sized with `STREAM_OPTIONS_FRAG.len()` of headroom (see `request_filter`), so
    // when the client declared a Content-Length this `resize` is free and the only work is moving
    // the `body.len() - at` bytes after the splice point. The old form allocated a second buffer the
    // size of the whole body and copied every byte into it; at 1 MB that measured 579 µs against
    // 27.8 µs, because the second allocation dominates.
    //
    // Without a declared length (chunked upload) the `resize` may grow once — still a single
    // allocation, i.e. no worse than before.
    let old_len = body.len();
    body.resize(old_len + STREAM_OPTIONS_FRAG.len(), 0);
    body.copy_within(at..old_len, at + STREAM_OPTIONS_FRAG.len());
    body[at..at + STREAM_OPTIONS_FRAG.len()].copy_from_slice(STREAM_OPTIONS_FRAG);
    body
}

/// What the first path segment resolved to.
///
/// An enum rather than a wider tuple because the three failure shapes need *different* rejections
/// (404 unknown provider, 404 unknown model, and — later, once identity is known — 400 for a BYO key
/// on the managed-only model route), and a tuple of `Option`s would encode that in which fields
/// happened to be `None`.
enum Routed {
    /// `/{provider}/…`, or the bare `/v1` default. The provider is named by the request.
    Provider {
        provider: Arc<Provider>,
        /// `None` for the bare default, whose path already is what the upstream should see.
        forward_path: Option<String>,
        streamable: bool,
    },
    /// `/auto/…` with a routing header naming a model the catalog carries.
    Model {
        route: &'static route::ModelRoute,
        /// The client path (+ query) after the `/auto` segment; the mount is prepended per attempt.
        suffix: String,
        streamable: bool,
    },
    /// `/auto/…` with no routing header, or one naming a model we do not serve. Collapsed into one
    /// outcome deliberately: a value we cannot match is a value we do not serve, and splitting it
    /// would leak whether a given name is in the catalog to a caller who has not authenticated.
    UnknownModel,
    /// The first segment matches no provider, is not the bare default, and is not `/auto`.
    UnknownProvider,
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
        let routed = {
            let req = session.req_header();
            let uri = &req.uri;
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
                // `first` is non-empty here (an empty first segment matches no provider), so this
                // always differs from the inbound path and always needs the URI rewritten.
                let rest = &path[1 + first.len()..];
                let rest = if rest.is_empty() { "/" } else { rest };
                Routed::Provider {
                    provider: p.clone(),
                    forward_path: Some(with_query(rest)),
                    streamable: is_streamable_path(rest),
                }
            } else if let Some(name) = bare_default_provider_name(path) {
                // Bare default: dialect picks the provider and the path is forwarded unchanged, so
                // there is nothing to rewrite — `None`. This used to build the path back up with its
                // query appended, hand it to `upstream_request_filter`, get compared equal against
                // the inbound `path_and_query`, and dropped: one wasted allocation per request on
                // the drop-in default route, three when a query string was present.
                match self.state.provider(name) {
                    Some(p) => Routed::Provider {
                        provider: p.clone(),
                        forward_path: None,
                        streamable: is_streamable_path(path),
                    },
                    None => Routed::UnknownProvider,
                }
            } else if first == route::AUTO_SEGMENT {
                // Model-routed. Reached only after a provider-table miss, so the established routes
                // pay nothing for this arm — and `state::build_providers` refuses to boot with a
                // provider named `auto`, so the miss is guaranteed rather than merely likely.
                let rest = &path[1 + first.len()..];
                let rest = if rest.is_empty() { "/" } else { rest };
                // Resolved to a `&'static` row inside the borrow, so nothing borrowed from the
                // session escapes into the rejection paths below.
                let row = req
                    .headers
                    .get(route::MODEL_HEADER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(route::model_route);
                match row {
                    Some(route) => Routed::Model {
                        route,
                        suffix: with_query(rest),
                        // Streamability is a property of the client's suffix, not of whichever mount
                        // gets prepended: every candidate's mount is a prefix, and
                        // `is_streamable_path` matches by suffix. Computed once here for the same
                        // reason it is computed pre-query above.
                        streamable: is_streamable_path(rest),
                    },
                    None => Routed::UnknownModel,
                }
            } else {
                Routed::UnknownProvider
            }
        };

        // Model routing needs identity before it can pick a provider (it is managed-only, and a
        // candidate is only usable if we hold a pool key for it), so the catalog row is carried
        // through the auth gates and resolved to a provider after them. `provider` below is the
        // row's first candidate, which is also what the request will attempt first.
        let (provider, forward_path, forward_streamable, model_route, auto_suffix) = match routed {
            Routed::Provider {
                provider,
                forward_path,
                streamable,
            } => (provider, forward_path, streamable, None, None),
            Routed::Model {
                route,
                suffix,
                streamable,
            } => {
                // Every candidate shares a wire (a catalog invariant), so the first one's dialect —
                // which is all that is read before the real candidate is chosen — is the row's.
                let first = route.candidates.first().and_then(|c| {
                    self.state
                        .provider_by_id(c.provider)
                        .map(|p| (p.clone(), c.upstream_model))
                });
                match first {
                    Some((p, _)) => (p, None, streamable, Some(route), Some(suffix)),
                    // Unreachable in practice: `every_catalog_candidate_is_a_known_provider` proves
                    // every row names a provider the gateway registers. Answer rather than panic.
                    None => {
                        self.state.metrics.rejection(Rejection::NoCandidate).inc();
                        return Self::reject_boxed(
                            session,
                            &request_id,
                            503,
                            "api_error",
                            "no provider available for model",
                        )
                        .await;
                    }
                }
            }
            Routed::UnknownModel => {
                self.state.metrics.rejection(Rejection::UnknownModel).inc();
                return Self::reject_boxed(
                    session,
                    &request_id,
                    404,
                    "invalid_request_error",
                    "unknown model",
                )
                .await;
            }
            Routed::UnknownProvider => {
                return Self::reject_boxed(
                    session,
                    &request_id,
                    404,
                    "invalid_request_error",
                    "unknown provider",
                )
                .await;
            }
        };
        // Dialect now comes from the resolved provider (usage parsing + injection eligibility).
        let dialect = provider.dialect;

        // 2. Extract the presented key — a managed virtual key (`bai_…`) or a raw BYO provider token.
        let Some(raw_key) = extract_virtual_key(session.req_header()) else {
            return Self::reject_boxed(
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
                self.state.metrics.rejection(reason.into()).inc();
                return Self::reject_boxed(
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
                return Self::reject_boxed(
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
                self.state.metrics.rejection(Rejection::Auth).inc();
                return Self::reject_boxed(
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
                    crate::deny::DenyReason::Spend => Rejection::DenySpend,
                    crate::deny::DenyReason::Fraud => Rejection::DenyFraud,
                    crate::deny::DenyReason::Unknown => Rejection::DenyUnknown,
                };
                self.state.metrics.rejection(label).inc();
                return Self::reject_boxed(
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
            //
            // Skipped for a model-routed request: there a pool key is a property of each *candidate*,
            // and the first one lacking a key is a reason to try the next, not to fail the request.
            // The equivalent gate is the usable-candidate check below.
            if model_route.is_none() && provider.pool_auth_value.is_none() {
                return Self::reject_boxed(
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

        // Model routing is **managed-only**, and the first candidate is chosen here.
        let (provider, usable) = match model_route {
            None => (provider, 0u8),
            Some(row) => {
                // A BYO token belongs to exactly one provider. Selecting among candidates would be a
                // guess about which — the guess the `providers` crate exists to refuse — and failing
                // over would send the caller's key for one vendor to a different vendor, which is a
                // credential disclosure, not a degraded experience. 400 rather than 401: their key
                // may be perfectly valid, it is the endpoint that is wrong for it.
                if !managed {
                    self.state
                        .metrics
                        .rejection(Rejection::ByoOnModelRoute)
                        .inc();
                    return Self::reject_boxed(
                        session,
                        &request_id,
                        400,
                        "invalid_request_error",
                        "model routing requires a managed key",
                    )
                    .await;
                }
                // Bit i ⇒ candidate i is registered here *and* has a pool key. Computed once; every
                // later attempt reads this instead of re-deriving it.
                let mut usable = 0u8;
                for (i, c) in row
                    .candidates
                    .iter()
                    .take(route::MAX_CANDIDATES)
                    .enumerate()
                {
                    let keyed = self
                        .state
                        .provider_by_id(c.provider)
                        .is_some_and(|p| p.pool_auth_value.is_some());
                    if keyed {
                        usable |= 1 << i;
                    }
                }
                let Some(first) = first_usable(usable, 0) else {
                    // Every candidate is unkeyed. Distinct from `circuit_open`, which means the
                    // candidates exist and are being skipped while they recover — `doctor`'s
                    // `model_catalog` check exists to catch this configuration at boot instead.
                    self.state.metrics.rejection(Rejection::NoCandidate).inc();
                    return Self::reject_boxed(
                        session,
                        &request_id,
                        503,
                        "api_error",
                        "no provider key available",
                    )
                    .await;
                };
                match row
                    .candidates
                    .get(usize::from(first))
                    .and_then(|c| self.state.provider_by_id(c.provider))
                {
                    Some(p) => (p.clone(), usable),
                    // Unreachable: the bit is only set when `provider_by_id` resolved above.
                    None => {
                        self.state.metrics.rejection(Rejection::NoCandidate).inc();
                        return Self::reject_boxed(
                            session,
                            &request_id,
                            503,
                            "api_error",
                            "no provider key available",
                        )
                        .await;
                    }
                }
            }
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
        //
        // **Not** for a model-routed request: which provider it attempts is not settled until
        // `upstream_peer` picks a candidate, and gating here would claim a permit against candidate
        // 0 and then claim a second one against whichever candidate is actually tried. That path
        // gates per candidate instead, at the same "last thing before the connection" position.
        if model_route.is_none() {
            if let Some(breaker) = &provider.breaker {
                if breaker.allow().is_err() {
                    self.state.metrics.rejection(Rejection::CircuitOpen).inc();
                    return Self::reject_boxed(
                        session,
                        &request_id,
                        503,
                        "api_error",
                        "provider temporarily unavailable",
                    )
                    .await;
                }
            }
        }
        // A permit is now outstanding against this provider (see `RequestCtx::breaker_pending`).
        // `breaker.is_some()` is exactly the condition `logging` used to test inline before the
        // ledger existed, so recording is unchanged for the provider-routed path. The model-routed
        // path starts owing nothing and takes on its first permit in `upstream_peer`.
        let breaker_pending = model_route.is_none() && provider.breaker.is_some();

        *ctx = Some(RequestCtx {
            tenant_id,
            vpc_id,
            dialect,
            provider,
            forward_path,
            managed,
            model: String::new(),
            model_scanner: peek::ModelScanner::new(),
            // `for_response`, not `new`: a response may carry the model nested under `message`
            // (Anthropic's `message_start`), and a root-only scanner would neither find it nor ever
            // stop looking. See `ModelScanner::for_response`.
            resp_model_scanner: peek::ModelScanner::for_response(),
            streaming: false,
            inject_eligible,
            // Only the inject-eligible path ever buffers the request body (to splice
            // `stream_options` after the root `{`; the `stream` key can appear anywhere in the root
            // object, so the decision needs the whole body — buffering is inherent here, not
            // incidental). When it does, pre-size from the declared Content-Length so accumulation is
            // a single allocation instead of a geometric realloc chain; capped at `MAX_REQUEST_BODY`
            // so a lying header can't pre-allocate unbounded memory. Every other request leaves this
            // empty and never buffers.
            //
            // The `+ STREAM_OPTIONS_FRAG.len()` is headroom for the splice, which
            // `apply_stream_usage_injection` performs *in place*: with it the injection never
            // reallocates, so a body arrives, is spliced, and goes upstream on one allocation.
            req_buf: match (inject_eligible || model_route.is_some(), declared_len) {
                (true, Some(len)) => {
                    Vec::with_capacity(len.min(MAX_REQUEST_BODY) + STREAM_OPTIONS_FRAG.len())
                }
                _ => Vec::new(),
            },
            // Grown lazily by the response tap (`response_body_filter`), not pre-reserved: a
            // non-streaming response — the common case — is a few hundred bytes, so reserving the
            // full 64KB cap up front would waste an allocation on every request to hold ~200B. A
            // long stream grows it geometrically to the bounded 2×cap and compacts; that handful of
            // reallocs is lost in the network noise of a stream we're already relaying chunk by chunk.
            resp_tail: UsageTail::default(),
            // Grown lazily, and only on the one path that needs it (Anthropic SSE) — see
            // `response_body_filter`. Every other response leaves this empty and never allocates.
            resp_head: Vec::new(),
            body_bytes_fed: 0,
            upstream_status: None,
            start,
            attempt: 0,
            breaker_pending,
            auto: model_route.map(|route| {
                Box::new(ModelRouting {
                    route,
                    // `first_usable` picked this candidate above; `upstream_peer` re-derives it from
                    // here on.
                    candidate: first_usable(usable, 0).unwrap_or(0),
                    usable,
                    suffix: auto_suffix.unwrap_or_default(),
                    // Overwritten per attempt by `upstream_peer`; seeded so the first attempt is
                    // timed even if it fails before the prologue runs.
                    attempt_start: start,
                })
            }),
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
        let Some(rc) = ctx.as_mut() else {
            return Err(pingora_core::Error::new_str(
                "upstream_peer reached without request context",
            ));
        };

        // Pingora calls this once per attempt, always before a body byte moves, so it is the one
        // place a retry's leftover request-body state can be cleared. No-op on the first attempt.
        rc.reset_request_body_phase();

        // Model-routed: this hook owns the candidate walk *and* the breaker ledger.
        //
        // It is the only place `rc.provider` changes, which is what makes double-recording
        // structurally impossible — and it is the only hook that runs on every attempt, including
        // the ones pingora starts without consulting `fail_to_connect` (its default
        // `error_while_proxy` marks a reused-connection failure retryable on its own). A design that
        // recorded in `fail_to_connect` would miss exactly those.
        if let Some(row) = rc.auto.as_ref().map(|a| a.route) {
            // Reaching here with a permit outstanding means the previous attempt failed before any
            // response arrived, so the candidate we were on earned the failure. Resolve it before
            // touching anything else; `logging` then only ever sees the final candidate's permit.
            if rc.breaker_pending {
                if let Some(b) = &rc.provider.breaker {
                    b.record_failure();
                }
                rc.breaker_pending = false;
            }

            loop {
                // Read the cursor out of the boxed state; `rc` stays mutably borrowable below.
                let (usable, at) = match rc.auto.as_ref() {
                    Some(a) => (a.usable, a.candidate),
                    None => {
                        return Err(pingora_core::Error::new_str("model routing state missing"));
                    }
                };
                let Some(i) = first_usable(usable, at) else {
                    // Out of candidates. `Error::new` defaults `retry` to false, so the proxy loop
                    // stops here rather than spinning; `logging` finds nothing pending to record.
                    return Err(pingora_core::Error::new_str(
                        "no candidate provider available",
                    ));
                };
                if let Some(a) = rc.auto.as_mut() {
                    a.candidate = i;
                }

                let candidate = row.candidates.get(usize::from(i));
                let resolved =
                    candidate.and_then(|c| self.state.provider_by_id(c.provider).cloned());
                let Some(p) = resolved else {
                    // Unreachable: `usable` bits are only set for candidates that resolved.
                    rc.advance_candidate(i);
                    continue;
                };

                // Gate on *this* candidate's breaker. An open one is skipped without claiming a
                // permit and without an attempt — the entire point of holding a candidate list.
                //
                // Deliberately `allow()` rather than reading `state()`: the OPEN → HALF_OPEN
                // transition happens *inside* `allow()`, so a `state()`-based pre-check would report
                // `Open` past the reset timeout, skip a candidate that `allow()` would have admitted
                // as a probe, and leave the breaker with no way to ever close.
                if let Some(b) = &p.breaker {
                    if b.allow().is_err() {
                        self.state.metrics.rejection(Rejection::CircuitOpen).inc();
                        rc.advance_candidate(i);
                        continue;
                    }
                }
                // A permit (if this breaker has one to give) is now outstanding against `p`.
                rc.breaker_pending = p.breaker.is_some();
                rc.provider = p.clone();

                match self.state.resolve(&p.authority).await {
                    Ok(addr) => {
                        // Time this attempt from here, so a candidate that burned its connect
                        // timeout does not charge that to whichever provider ends up serving.
                        if let Some(a) = rc.auto.as_mut() {
                            a.attempt_start = Instant::now();
                        }
                        rc.rebuild_forward_path(p.base_path);
                        return Ok(Box::new(self.build_peer(addr, &p)));
                    }
                    Err(e) => {
                        // DNS failure is handled *here*, inside the walk, rather than by returning
                        // an error: pingora does not call `fail_to_connect` when `upstream_peer`
                        // itself fails (lib.rs returns early), so a returned error would end the
                        // request instead of trying the next candidate — and a provider that has
                        // vanished from DNS is precisely a case failover exists for.
                        warn!(
                            request_id = %rc.request_id,
                            provider = p.name.as_str(),
                            authority = p.authority.as_str(),
                            candidate = i,
                            error = %e,
                            "upstream dns resolution failed; trying the next candidate",
                        );
                        if let Some(b) = &p.breaker {
                            b.record_failure();
                        }
                        rc.breaker_pending = false;
                        rc.advance_candidate(i);
                        continue;
                    }
                }
            }
        }

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
        Ok(Box::new(self.build_peer(addr, &rc.provider)))
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
            // Strip **unconditionally**, before deciding whether there is a pool key to insert.
            //
            // These two used to sit inside the `if let Some(av)` below, which was safe only because
            // a managed request without a pool key is rejected earlier. That is a load-bearing
            // invariant expressed nowhere near here, and if it ever failed the consequence was not a
            // degraded request but a *credential disclosure*: nothing removed, nothing inserted, and
            // the caller's Ed25519 virtual key forwarded verbatim to a third-party provider. Now the
            // worst case is an unauthenticated request the provider rejects with a 401.
            upstream_request.remove_header("authorization");
            for header in STATIC_KEY_HEADERS {
                upstream_request.remove_header(header);
            }
            if let Some(av) = &rc.provider.pool_auth_value {
                // Clone the boot-built `HeaderValue` (a refcount bump) rather than re-validating and
                // re-copying the key out of a `&str` on every managed request. The `&str` path is
                // kept as a fallback for a key that isn't a legal header value, which could never
                // have worked anyway — see `Provider::pool_auth_header`.
                match &rc.provider.pool_auth_header {
                    Some(hv) => {
                        upstream_request.insert_header(rc.provider.auth.header(), hv.clone())?
                    }
                    None => {
                        upstream_request.insert_header(rc.provider.auth.header(), av.expose())?
                    }
                }
            }
        }

        // The routing header is ours, not the provider's. Stripped on every attempt (pingora rebuilds
        // this header from the downstream request each time, so it reappears each time).
        if rc.auto.is_some() {
            upstream_request.remove_header(route::MODEL_HEADER);
        }

        // Point Host at the upstream. Same precomputed-value trick as the pool key above.
        match &rc.provider.host_header {
            Some(hv) => upstream_request.insert_header("host", hv.clone())?,
            None => upstream_request.insert_header("host", rc.provider.host.as_str())?,
        }

        // Dashboard-attribution headers (OpenRouter, managed traffic only — Task #22, see
        // `apply_provider_attribution`).
        apply_provider_attribution(upstream_request, rc.provider.name.as_str(), rc.managed)?;

        // Forward the provider-native path (computed in `request_filter`): the client path with the
        // `/{provider}` segment stripped. Sent verbatim — no per-provider rewriting. The body's
        // framing (Content-Length / chunked) is preserved.
        //
        // `None` means the bare-path default, where the path is already what the upstream should
        // see, so there is nothing to build and nothing to parse. That used to be expressed by
        // reconstructing the path anyway and comparing it against the inbound `path_and_query`;
        // encoding it in the type instead skips the allocation rather than detecting it after
        // the fact.
        if let Some(forward_path) = &rc.forward_path
            && let Ok(uri) = forward_path.parse()
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
        //
        // Keyed on `rewrites_body`, not `inject_eligible`, so the model rewrite gets the same
        // treatment: `openai/gpt-4o-mini` is longer than `gpt-4o-mini`, and forwarding the client's
        // original `Content-Length` alongside a longer body truncates it at the upstream.
        if rc.rewrites_body() {
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
        // buffered) to extract the exact root-level `model` — but only for **managed** traffic,
        // which is the only path that reads it. `rc.model` is used at exactly two places, both
        // inside the `if rc.managed` block in `logging` (the billing-log fallback and
        // `requested_model`). Scanning it for BYO meant walking the whole request body — a
        // structural, depth- and escape-aware pass — to produce a value guaranteed to be discarded.
        if let Some(chunk) = body.as_ref() {
            // Enforce the body cap on the *streamed* size too: the up-front `Content-Length` check in
            // `request_filter` can't see a chunked-encoded body (no declared length). We don't buffer
            // — we just count — and abort the proxied request once the running total crosses the cap.
            // Aborting (vs. a clean 413) is acceptable here: headers are already away to the upstream,
            // and this is an abuse guard, not a normal client path.
            rc.body_bytes_fed = rc.body_bytes_fed.saturating_add(chunk.len());
            if rc.body_bytes_fed > MAX_REQUEST_BODY {
                self.state.metrics.rejection(Rejection::BodyTooLarge).inc();
                return Err(pingora_core::Error::new_str("request body exceeds limit"));
            }
            // Eligible requests are buffered so we can splice the root object before any byte reaches
            // the upstream (injection inserts near the front, so we can't have forwarded it already).
            // When we're buffering anyway, the incremental scan is skipped entirely and both answers
            // come from one walk of the finished buffer below — the body was previously traversed
            // twice, once here for `model` and once by the injection planner, over the same bytes
            // with the same depth/string/escape bookkeeping.
            if rc.rewrites_body() {
                rc.req_buf.extend_from_slice(chunk);
            } else if rc.managed {
                rc.model_scanner.feed(chunk);
            }
        }

        if rc.rewrites_body() {
            if end_of_stream {
                // One structural walk for every answer (see `peek::scan_buffered`).
                let buf = std::mem::take(&mut rc.req_buf);
                let scan = peek::scan_buffered(&buf);
                if rc.model.is_empty() {
                    if let Some(m) = scan.model {
                        // The *client's* id, captured before any rewrite below — this is what
                        // `requested_model` means, and it stays the canonical catalog name on a
                        // model-routed request even though the upstream is about to be told
                        // something else.
                        rc.model = sanitize_model(m).into_owned();
                    }
                }
                // Model-routed: re-spell `model` as the candidate serving *this attempt* spells it.
                // Providers essentially never agree on an id — Anthropic's `claude-opus-4-8` is
                // OpenRouter's `anthropic/claude-opus-4-8` — so without this a failover would ask
                // the fallback for a model it has never heard of.
                //
                // Done before the `stream_options` splice, and safe in that order because
                // `inject_at` points just past the root `{` and so always precedes the model value:
                // rewriting the value cannot move it.
                let buf = match (rc.auto.as_ref(), scan.model_span) {
                    (Some(a), Some(span)) => {
                        match a.route.candidates.get(usize::from(a.candidate)) {
                            Some(c) => apply_model_rewrite(buf, span, c.upstream_model.as_bytes()),
                            None => buf,
                        }
                    }
                    _ => buf,
                };
                // Emit the whole (possibly rewritten) body in one shot; `transfer-encoding: chunked`
                // (set in `upstream_request_filter`) makes the changed length fine.
                *body = Some(Bytes::from(apply_stream_usage_injection(
                    buf,
                    scan.inject_at,
                )));
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

        // The streamed (non-buffered) path; the buffered one resolved `model` above from its single
        // fused walk, and its scanner was never fed.
        if end_of_stream && rc.managed && !rc.inject_eligible && rc.model.is_empty() {
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
            //
            // Timed from `attempt_start`, not `start`. They are the same instant for a request that
            // connects first try; they differ once a candidate has been abandoned, and charging the
            // dead candidate's `connect_timeout_secs` to the provider that actually answered would
            // render an outage at A as a latency regression at B — inverting the one thing the
            // per-provider label is for.
            rc.provider
                .metrics
                .ttft_seconds
                .observe(rc.attempt_start().elapsed().as_secs_f64());

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
            //
            // Managed only, for the same reason as the request-side scanner: the extracted value is
            // read at exactly one place, inside the `if rc.managed` block in `logging`. A BYO
            // request has no Beyond identity and emits no billing row, so scanning its response was
            // pure waste — and *unbounded* waste on any response with no root-level `model`, since
            // the scanner never reaches its `done` short-circuit and walks every byte.
            if rc.managed {
                rc.resp_model_scanner.feed(chunk);
            }

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

            rc.resp_tail.push(chunk);
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
            // Model-routed: one attempt per candidate, in preference order.
            //
            // No same-peer retry here, unlike the provider-routed path below. A *different* provider
            // is strictly a superset of retrying the same one — it recovers from the transient blip
            // the retry exists for and from a provider that is simply down, which the retry cannot.
            // It also bounds the dead air: each attempt can burn `connect_timeout_secs`, so
            // `MAX_CONNECT_RETRIES` attempts per candidate would multiply the client's worst case by
            // three for no additional coverage. And it keeps the ledger trivial — exactly one
            // `allow()`, one attempt, and one `record_*` per candidate.
            if let Some((usable, at)) = rc.auto.as_ref().map(|a| (a.usable, a.candidate)) {
                // The failure itself is recorded by `upstream_peer`'s prologue, when it moves off
                // this candidate. Recording here as well would double-count whenever there is no
                // next candidate, since `logging` would then also resolve the still-pending permit —
                // which would trip the breaker at half its configured threshold on the last
                // candidate, exactly where everything lands once the primaries are sick.
                rc.provider.metrics.connect_retries_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.provider.name.as_str(),
                    candidate = at,
                    error = %e,
                    "upstream connect failed; trying the next candidate",
                );
                // Only signal a retry if there is somewhere to go. Otherwise leave `retry` false so
                // the proxy loop stops and `logging` resolves the outstanding permit against this,
                // the last candidate.
                if first_usable(usable, at.saturating_add(1)).is_some() {
                    self.state.metrics.candidate_failovers_total.inc();
                    rc.advance_candidate(at);
                    e.set_retry(true);
                }
                return e;
            }
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

        // Resolve the outstanding circuit-breaker permit, if this request still owes one.
        //
        // `breaker_pending` is the ledger (see `RequestCtx::breaker_pending`): it is set when a
        // permit is claimed and cleared when it is recorded, so exactly one `record_*` lands per
        // `allow()`. On the model-routed path a candidate switch resolves the outgoing candidate in
        // `upstream_peer` and clears the flag there, which is what stops this from double-recording
        // against whichever candidate happened to be current at the end.
        //
        // Failure = the provider is *broken*: a 5xx response, or no response at all paired with an
        // upstream error (connect/read failure). Success = the provider *answered* — 2xx/3xx, and
        // deliberately **4xx/429 too**: a 429 is a healthy provider throttling our pool key, which the
        // rate limiter and the client's `Retry-After` own, NOT a reason to cut all traffic to it.
        if let Some(breaker) = rc.provider.breaker.as_ref().filter(|_| rc.breaker_pending) {
            match rc.upstream_status {
                Some(s) if s >= 500 => breaker.record_failure(),
                Some(_) => breaker.record_success(),
                // No response head arrived. Blame the provider only when the failure actually came
                // *from* upstream. Pingora tags a client-side abort `ErrorSource::Downstream` (the
                // `into_down()` at proxy_h1.rs's downstream read/write sites), and a user hitting
                // ESC on a slow turn says nothing about the provider's health. Counting those was a
                // live bug: cancellation is routine for a coding agent, and
                // `circuit_breaker_threshold` cancellations inside `circuit_breaker_window_secs`
                // would open the breaker and 503 *everyone*. Worse, `half_open_permits` is 1, so a
                // cancel-prone request drawn as the probe reopened it every time — the breaker could
                // not recover while users were cancelling.
                None if is_upstream_failure(e) => breaker.record_failure(),
                // Client went away, or the request ended with no error at all. The provider is not
                // implicated either way; record a success so a claimed half-open probe permit still
                // resolves rather than being stranded.
                None => breaker.record_success(),
            }
            rc.breaker_pending = false;
        }

        // The last `USAGE_TAIL_CAP` bytes of the response, oldest first (see `UsageTail`). Short
        // responses are the whole body; long ones are rotated into order here, once.
        let tail = rc.resp_tail.contiguous();

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
        // Pre-resolved fixed-label children, and zeros skipped (see `Metrics::record_tokens`). Cache
        // tokens are counted here as well as in the `ai.usage` billing log below, because that log
        // ships with lag — the counter is the alerting surface for a cache-hit-rate cliff after a
        // deploy.
        m.record_tokens(&usage);
        // Read the clock once for both consumers below. Beyond saving a vDSO call, this is a
        // correctness fix: the latency histogram and the `ai.usage` billing line used to call
        // `elapsed()` about forty lines apart, so they reported *different* durations for the same
        // request and could never be reconciled against each other.
        let elapsed = rc.start.elapsed();
        rc.provider
            .metrics
            .upstream_latency_seconds
            .observe(elapsed.as_secs_f64());
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
            //
            // Third link in the fallback chain for a model-routed request: if neither the response
            // nor the body carried an id, the catalog name we routed on is still a true statement
            // about what was asked for, and strictly better than the empty string this used to ship.
            // Derived rather than stored: it is the catalog row's own name.
            let routed_model = rc.auto.as_ref().map(|a| a.route.model);
            let resolved = billed.as_deref().unwrap_or(&rc.model);
            let billed_model = if resolved.is_empty() {
                routed_model.unwrap_or_default()
            } else {
                resolved
            };
            info!(
                target: "ai.usage",
                request_id = %rc.request_id,
                tenant_id = rc.tenant_id,
                vpc_id = rc.vpc_id,
                provider = rc.provider.name.as_str(),
                model = billed_model,
                requested_model = %rc.model,
                // The catalog name the request routed on — `None` (absent from the row) for a
                // provider-routed request. Distinct from `requested_model`, which is whatever the
                // client's *body* asked for: nothing enforces that the two agree, since the routing
                // decision is made from the header before the body is ever read, and a divergence is
                // only visible here, after the fact. `&'static` from the catalog and charset-checked
                // by a catalog test, so it needs no `sanitize_model`.
                routed_model,
                stream = rc.streaming,
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                cache_read_tokens = usage.cache_read_tokens,
                cache_write_tokens = usage.cache_write_tokens,
                // `Some(0)` (reported, none used) vs `None` (not reported at all — an unreasoning
                // model, or a provider that doesn't surface it) matters and is unrecoverable once this
                // line ships, so it's logged as `?` (Debug) rather than collapsed to a bare `0`.
                reasoning_tokens = ?usage.reasoning_tokens,
                latency_ms = elapsed.as_millis() as u64,
                "usage"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RequestCtx` is touched on every hook and, for a streaming response, once per response
    /// chunk, so its size is a real cost rather than bookkeeping. Growing it by 64 bytes to hold the
    /// model-routing fields inline produced a reproducible ~2.5% regression on the
    /// `managed_sse_latency` bench — non-streaming was unaffected, which is the signature of a
    /// per-chunk cost. Boxing that state (see `ModelRouting`) bought it back.
    ///
    /// A ceiling rather than an equality: padding and field order are the compiler's business, and a
    /// few bytes either way is not what this guards. What it guards is someone adding a `String` or
    /// an `Instant` here without noticing that the cost is paid per chunk on every stream.
    #[test]
    fn request_ctx_stays_small_enough_to_be_cheap_per_chunk() {
        let size = std::mem::size_of::<RequestCtx>();
        assert!(
            size <= 384,
            "RequestCtx grew to {size} bytes (ceiling 384). It is touched once per response chunk \
             on a stream — if the new state is only needed on one route, box it the way \
             `ModelRouting` is rather than paying for it on every request.",
        );
    }

    /// The candidate cursor. `from` strictly increases across a request, which is what guarantees
    /// the walk terminates and that no candidate can be revisited to claim a second breaker permit.
    #[test]
    fn first_usable_walks_set_bits_in_order_and_terminates() {
        // 0b1011 ⇒ candidates 0, 1, 3.
        assert_eq!(first_usable(0b1011, 0), Some(0));
        assert_eq!(first_usable(0b1011, 1), Some(1));
        assert_eq!(first_usable(0b1011, 2), Some(3));
        assert_eq!(first_usable(0b1011, 3), Some(3));
        assert_eq!(first_usable(0b1011, 4), None);

        // Nothing usable at all — the "every candidate is unkeyed" case.
        assert_eq!(first_usable(0, 0), None);

        // Past the bitmask width, including values a `saturating_add` could park on.
        assert_eq!(first_usable(0xFF, route::MAX_CANDIDATES as u8), None);
        assert_eq!(first_usable(0xFF, u8::MAX), None);

        // Walking a full mask visits every index exactly once, in order, then stops.
        let mut seen = Vec::new();
        let mut at = 0u8;
        while let Some(i) = first_usable(0xFF, at) {
            seen.push(i);
            at = i.saturating_add(1);
        }
        assert_eq!(seen, (0..route::MAX_CANDIDATES as u8).collect::<Vec<_>>());
    }

    /// The rewrite must replace exactly the value and leave the rest of the body byte-identical,
    /// whether the new id is longer, shorter, or the same. A body that stays valid JSON with a
    /// subtly wrong model is the failure this guards, and nothing downstream would catch it.
    #[test]
    fn model_rewrite_replaces_exactly_the_value() {
        let body = br#"{"model":"gpt-4o-mini","stream":true}"#.to_vec();
        let span = peek::scan_buffered(&body).model_span.unwrap();

        // Longer (the real failover case: OpenRouter prefixes the vendor).
        let grown = apply_model_rewrite(body.clone(), span, b"openai/gpt-4o-mini");
        assert_eq!(
            std::str::from_utf8(&grown).unwrap(),
            r#"{"model":"openai/gpt-4o-mini","stream":true}"#,
        );
        // Shorter.
        let shrunk = apply_model_rewrite(body.clone(), span, b"x");
        assert_eq!(
            std::str::from_utf8(&shrunk).unwrap(),
            r#"{"model":"x","stream":true}"#,
        );
        // Identical ⇒ untouched, and no memmove: the primary candidate's common case.
        let same = apply_model_rewrite(body.clone(), span, b"gpt-4o-mini");
        assert_eq!(same, body);

        // Both rewrites still parse, and the result is still injectable at the same offset — the
        // ordering invariant that lets the model rewrite run before the stream_options splice.
        let rescanned = peek::scan_buffered(&grown);
        assert_eq!(rescanned.model.as_deref(), Some("openai/gpt-4o-mini"));
        assert_eq!(
            rescanned.inject_at,
            peek::scan_buffered(&body).inject_at,
            "the injection offset sits before the model value, so a rewrite cannot move it",
        );
    }

    /// An out-of-range span must be ignored rather than panicking a worker.
    #[test]
    fn model_rewrite_ignores_an_impossible_span() {
        let body = br#"{"model":"a"}"#.to_vec();
        assert_eq!(apply_model_rewrite(body.clone(), (5, 2), b"z"), body);
        assert_eq!(apply_model_rewrite(body.clone(), (0, 999), b"z"), body);
    }

    /// Mount composition for every shape in the provider table. Getting one wrong sends a request
    /// to a 404 on a provider that is perfectly healthy.
    #[test]
    fn mounted_path_prepends_the_candidates_mount() {
        let cases = [
            // (mount, client suffix, forwarded)
            ("/v1", "/chat/completions", "/v1/chat/completions"),
            ("/api/v1", "/chat/completions", "/api/v1/chat/completions"),
            (
                "/inference/v1",
                "/chat/completions",
                "/inference/v1/chat/completions",
            ),
            (
                "/openai/v1",
                "/chat/completions",
                "/openai/v1/chat/completions",
            ),
            // Anthropic's base carries no mount; the SDK's own `/v1/messages` is already complete.
            ("", "/v1/messages", "/v1/messages"),
            // A query string rides along on the suffix.
            (
                "/v1",
                "/chat/completions?api-version=2024",
                "/v1/chat/completions?api-version=2024",
            ),
        ];
        let mut buf = String::new();
        for (mount, suffix, want) in cases {
            write_mounted_path(&mut buf, mount, suffix);
            assert_eq!(buf, want, "mount {mount:?} + suffix {suffix:?}");
        }
        // Reused across attempts without growing: the same buffer served every case above.
        write_mounted_path(&mut buf, "/v1", "/chat/completions");
        assert_eq!(buf, "/v1/chat/completions");
    }

    /// A client-side abort must not count against the provider's breaker; everything else must.
    ///
    /// This is the whole of the cancellation bug: `ErrorSource::Downstream` is a user hitting ESC or
    /// a broken pipe writing the response back, and counting those opened breakers on healthy
    /// providers. `Unset` is our own DNS failure out of `upstream_peer` and `Upstream` is a real
    /// connect/read failure — both are genuine provider failures and must still count.
    #[test]
    fn only_non_downstream_errors_count_against_the_breaker() {
        use pingora_core::{Error, ErrorSource, ErrorType};

        assert!(!is_upstream_failure(None), "no error is not a failure");

        let mut down = *Error::new(ErrorType::WriteError);
        down.esource = ErrorSource::Downstream;
        assert!(
            !is_upstream_failure(Some(&down)),
            "a client abort must not trip the provider's breaker",
        );

        for source in [
            ErrorSource::Upstream,
            ErrorSource::Internal,
            ErrorSource::Unset,
        ] {
            let mut e = *Error::new(ErrorType::ConnectError);
            e.esource = source.clone();
            assert!(
                is_upstream_failure(Some(&e)),
                "{source:?} must count as a provider failure",
            );
        }
    }

    /// The premise behind `RequestCtx::reset_request_body_phase`: a scanner fed a duplicated body
    /// prefix is permanently broken, because the extra `{` offsets its brace depth so the root-level
    /// key check never matches again.
    ///
    /// Pingora replays its retry buffer through `request_body_filter`, so without the reset this is
    /// exactly what the second attempt's scanner sees — and the billing row ships
    /// `requested_model = ""`. The e2e coverage for the real replay path lives in `tests/e2e.rs`.
    #[test]
    fn a_replayed_body_prefix_breaks_a_scanner_that_was_not_reset() {
        // `model` deliberately sits after `messages`, so the scanner is still mid-walk when the
        // replayed prefix arrives — the case a short-circuit on an early `model` would hide.
        let body = br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#;
        let (prefix, rest) = body.split_at(30);

        let mut fresh = peek::ModelScanner::new();
        fresh.feed(body);
        assert_eq!(
            fresh.take_model().as_deref(),
            Some("gpt-4o"),
            "a clean walk finds the root-level model",
        );

        let mut replayed = peek::ModelScanner::new();
        replayed.feed(prefix);
        replayed.feed(prefix); // pingora replays from byte 0 on the retry
        replayed.feed(rest);
        assert_eq!(
            replayed.take_model(),
            None,
            "the duplicated prefix offsets the depth, so the root `model` is never seen — \
             this is why upstream_peer resets the scanner every attempt",
        );
    }

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
    fn usage_tail_retains_exactly_the_last_cap_bytes() {
        // The ring must retain byte-for-byte what the old grow-and-compact buffer did: the whole
        // body while it fits, the last `USAGE_TAIL_CAP` bytes once it doesn't. Getting this wrong
        // silently truncates or misorders the usage event, which is unrecoverable once the request
        // completes — so drive it across chunk sizes that do and don't divide the cap, and across
        // the boundary itself.
        let body: Vec<u8> = (0..(3 * USAGE_TAIL_CAP + 1234))
            .map(|i| (i % 251) as u8)
            .collect();

        for chunk in [1usize, 7, 4096, 8192, USAGE_TAIL_CAP - 1, USAGE_TAIL_CAP] {
            for total in [
                0usize,
                1,
                USAGE_TAIL_CAP - 1,
                USAGE_TAIL_CAP,
                USAGE_TAIL_CAP + 1,
                2 * USAGE_TAIL_CAP + 77,
                body.len(),
            ] {
                let src = &body[..total];
                let mut tail = UsageTail::default();
                for c in src.chunks(chunk.max(1)) {
                    tail.push(c);
                }
                let want = &src[src.len().saturating_sub(USAGE_TAIL_CAP)..];
                assert_eq!(
                    tail.contiguous(),
                    want,
                    "chunk={chunk} total={total}: retained window differs"
                );
                // Idempotent — `contiguous` must not consume or re-rotate.
                assert_eq!(tail.contiguous(), want);
            }
        }

        // A single chunk larger than the whole window keeps only its last cap bytes.
        let mut tail = UsageTail::default();
        tail.push(&body);
        assert_eq!(tail.contiguous(), &body[body.len() - USAGE_TAIL_CAP..]);

        // Memory stays bounded no matter how long the stream runs.
        let mut tail = UsageTail::default();
        for _ in 0..64 {
            tail.push(&body[..USAGE_TAIL_CAP]);
        }
        assert_eq!(tail.contiguous().len(), USAGE_TAIL_CAP);
    }

    #[test]
    fn in_place_splice_produces_the_same_bytes_as_a_copying_one() {
        // The splice moved from "allocate a second buffer and copy everything" to "shift the tail
        // right in place". The wire bytes must be identical, including when the buffer has no spare
        // capacity (a chunked upload, where `resize` has to grow) and when it has exactly the
        // headroom `request_filter` reserves.
        let copying = |body: &[u8], at: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(body.len() + STREAM_OPTIONS_FRAG.len());
            out.extend_from_slice(&body[..at]);
            out.extend_from_slice(STREAM_OPTIONS_FRAG);
            out.extend_from_slice(&body[at..]);
            out
        };

        for src in [
            &br#"{"model":"gpt-4o","stream":true,"messages":[]}"#[..],
            &br#"{"stream":true}"#[..],
            &b"  {  \"stream\" : true , \"model\" : \"m1\" }"[..],
        ] {
            let at = peek::scan_buffered(src).inject_at.expect("streaming body");

            // Exact-fit capacity, as `request_filter` pre-sizes it.
            let mut exact = Vec::with_capacity(src.len() + STREAM_OPTIONS_FRAG.len());
            exact.extend_from_slice(src);
            assert_eq!(
                apply_stream_usage_injection(exact, Some(at)),
                copying(src, at)
            );

            // No spare capacity at all — the chunked case.
            let tight = src.to_vec();
            let out = apply_stream_usage_injection(tight, Some(at));
            assert_eq!(out, copying(src, at));

            // ...and the result is still valid JSON carrying the option.
            let v: serde_json::Value = serde_json::from_slice(&out).expect("valid JSON");
            assert_eq!(
                v["stream_options"]["include_usage"],
                serde_json::json!(true)
            );
        }

        // Nothing to inject ⇒ the body is returned untouched, not grown.
        let untouched = br#"{"model":"gpt-4o"}"#.to_vec();
        assert_eq!(
            apply_stream_usage_injection(untouched.clone(), None),
            untouched
        );
    }

    #[test]
    fn reject_bodies_are_valid_json_and_match_their_type_and_message() {
        // These are hand-written JSON literals standing in for what `serde_json::json!` used to
        // build, so the thing to guard is that they still *say* what the `error_type` in the log
        // line and the metric label claim. A drifting literal would ship a response whose `type`
        // contradicts the reason we rejected for.
        for (typ, msg, body) in REJECT_BODIES {
            let v: serde_json::Value =
                serde_json::from_str(body).unwrap_or_else(|e| panic!("{body} is not JSON: {e}"));
            assert_eq!(v["error"]["type"], typ, "type mismatch in {body}");
            assert_eq!(v["error"]["message"], msg, "message mismatch in {body}");
            // ...and that it is byte-identical to what `json!` would have produced, so switching to
            // the constant changed nothing on the wire.
            let built = serde_json::json!({ "error": { "type": typ, "message": msg } }).to_string();
            assert_eq!(body, built, "constant diverges from the built body");
            // The lookup must find it rather than falling through to the allocating branch.
            assert_eq!(error_body(typ, msg), Bytes::from_static(body.as_bytes()));
        }
    }

    #[test]
    fn every_reject_call_site_has_a_precomputed_body() {
        // One table entry per call site. Adding a rejection without its entry would silently take
        // `error_body`'s allocating fallback forever — correct on the wire, but quietly undoing the
        // point of the table on the one path a flood drives at full rate. Counting is enough to
        // catch it and does not depend on how the arguments happen to be formatted.
        // Only the production half of the file — this test mentions the call form itself, and a
        // test module that grepped its own source would count that too.
        let src = include_str!("proxy.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("split always yields at least one part");
        // Pull each call site's `(type, message)` rather than counting sites. A raw count broke as
        // soon as one message was rejected from more than one place ("no provider key available"
        // now has a provider-routed and a model-routed caller), and relaxing it to `>=` would have
        // stopped catching the thing it exists for: a *new* message with no table entry.
        //
        // The two arguments after the numeric status are the only string literals in the call, and
        // rustfmt keeps them one per line, so scanning forward for the first two quoted literals is
        // stable.
        let mut sites = 0usize;
        for (i, _) in src.match_indices("Self::reject_boxed(") {
            sites += 1;
            let tail = &src[i..];
            let end = tail.find(")\n").map_or(tail.len(), |e| e + 1);
            let literals: Vec<&str> = tail[..end]
                .match_indices('"')
                .collect::<Vec<_>>()
                .chunks(2)
                .filter_map(|c| match c {
                    [(a, _), (b, _)] => Some(&tail[a + 1..*b]),
                    _ => None,
                })
                .collect();
            let (Some(typ), Some(msg)) = (literals.first(), literals.get(1)) else {
                panic!("could not read (type, message) from a reject_boxed call site");
            };
            assert!(
                REJECT_BODIES.iter().any(|(t, m, _)| t == typ && m == msg),
                "reject_boxed({typ:?}, {msg:?}) has no REJECT_BODIES entry — it would take \
                 `error_body`'s allocating fallback on every hit",
            );
        }
        assert!(
            sites >= REJECT_BODIES.len(),
            "{sites} call sites but {} table entries — an entry has no caller",
            REJECT_BODIES.len(),
        );
        // Every tabulated message must also appear at a call site, catching the reverse drift (a
        // table entry left behind after its rejection was removed). Twice: the table and the caller.
        for (_, msg, _) in REJECT_BODIES {
            assert!(
                src.matches(&format!("\"{msg}\"")).count() >= 2,
                "REJECT_BODIES entry {msg:?} has no reject_boxed call site"
            );
        }
    }

    #[test]
    fn rejection_labels_round_trip_and_are_unique() {
        use crate::metrics::Rejection;
        use std::collections::HashSet;
        // `Rejection::ALL` rather than a copy of the variant list: a copy silently stops covering
        // whatever is added next, which is exactly what it happened to do.
        let all = Rejection::ALL;
        let labels: HashSet<&str> = all.iter().map(|r| r.label()).collect();
        assert_eq!(labels.len(), all.len(), "duplicate rejection label");
        for r in all {
            let l = r.label();
            assert!(!l.is_empty(), "{r:?} has an empty label");
            assert!(
                l.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{r:?} label {l:?} must be a lowercase snake_case Prometheus label value",
            );
        }
        // The two dashboards-facing strings must not drift — existing alerts key on them.
        assert_eq!(Rejection::RateLimit.label(), "rate_limit");
        assert_eq!(
            Rejection::RateLimitByoGlobal.label(),
            "rate_limit_byo_global"
        );
        // `Throttled` must map onto the same two.
        assert_eq!(
            Rejection::from(crate::ratelimit::Throttled::PerCredential),
            Rejection::RateLimit
        );
        assert_eq!(
            Rejection::from(crate::ratelimit::Throttled::ByoGlobal),
            Rejection::RateLimitByoGlobal
        );
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
