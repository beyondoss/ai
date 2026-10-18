//! The Pingora `ProxyHttp` passthrough service.
//!
//! Flow: pick the provider from the **first path segment** (`/{provider}/…`) → verify the virtual
//! key (stateless) → deny-set check (O(1), default-allow) → allowance check (O(1), fail-closed
//! until seeded) → swap the auth
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
//! We never drain the request body in `request_filter` and leave Pingora with an empty forward:
//! Pingora's body-forward phase reads the downstream body itself, so consuming it earlier without
//! replaying would make Pingora send `Content-Length` bytes with no body and the upstream would hang.
//! The supported hook is `request_body_filter`, which feeds each chunk to a streaming structural
//! scanner (`peek::ModelScanner`, O(1) memory) — never withholding it on the ordinary path.
//! The exact-match cache may continue that same peek past the first `model` for a small complete
//! body (still inside pingora's retry buffer), so a hit can hash the pre-rewrite bytes; a miss is
//! still replayed, never withheld.
//!
//! One exception reads ahead: a **managed** request on the bare `/v1` default (or `/auto` with no
//! routing header) must resolve a catalog row from the body's root `model` before `upstream_peer`
//! runs. That peek enables pingora's 64 KiB retry buffer and reads the whole body; one that outgrew
//! the buffer is re-run as a pingora subrequest that carries it (`FullBody`), which is also how a
//! large body fails over. Unknown or missing model → 404 naming the miss. Chat Completions ↔ Messages ↔ Responses on a managed catalog walk is
//! translated; inbound Responses with session state walks a GPT row's `/v1/responses` arm (byte
//! relay) or 400s if none remain, naming the field. Any other inbound path vs row endpoint mismatch
//! → 400. A managed `GET /v1/models` lists the keyed catalog; a BYO one relays to its provider.
//!
//! One deliberate exception to the no-buffer rule: a **managed** OpenAI Chat Completions request is
//! buffered and gets `stream_options.include_usage` injected when it streams without it (or forced
//! to `true` when the client sent `stream_options` itself) — otherwise OpenAI emits no usage chunk
//! and the request couldn't be metered. We can't set that option in a
//! client SDK we don't control, so the gateway guarantees it, out of the box. Scoped to exactly that
//! path (managed + OpenAI dialect + chat/completions); BYO and everything else stay pure passthrough.
//! The Responses API needs no such injection — it always reports usage on its terminal event — so it
//! stays pure passthrough too (see `is_streamable_path`).
//!
//! Auth branches on key format: `bai_v1…` is a managed virtual key (verify → deny-check → swap to
//! the pool key; verify failure is **401, never BYO**); anything else is a **BYO** request — the
//! user's own provider token, passed through unchanged (no swap, no Beyond identity, no deny-set).
//! The key is read from whichever header (or, for Google Gemini, query param) the client's SDK
//! uses — see `extract_virtual_key`.
//!
//! Routing is by the **first path segment** = provider name (`route`, data-driven): `/{provider}/…`
//! selects the provider and the rest of the path is forwarded **verbatim** (the gateway holds no
//! per-provider mount knowledge). A bare path with no provider prefix that is exactly `/v1` or
//! starts with `/v1/` (boundary-checked — see `route::is_default_prefix`, not a raw
//! `starts_with("/v1")`, which would also absorb a lookalike like Google Gemini's `/v1beta/…`) is
//! the drop-in default. BYO traffic there still dialect-picks openai/anthropic (`dialect_for_path`).
//! A **managed** request to that default — or to `/auto` — resolves a catalog row from
//! `x-beyond-model` if present, else the body's root `model`; the catalog is the allowlist
//! (unknown/missing → 404 naming the miss) and the request walks that row's same-wire candidates.
//! Candidate spellings are aliases. Same-wire catalog walks are a byte relay; Chat Completions ↔
//! Messages on a managed `/v1` or `/auto` walk is translated so a stock SDK can call the other
//! dialect. Inbound `/v1/responses` with session state (`previous_response_id`, or `store` not
//! explicitly false) walks the row's Responses arm or 400s; `store: false` one-shots may still
//! translate onto Chat Completions. Other inbound-path mismatches are still a 400. `GET /v1/models`
//! lists the keyed catalog to a managed key and relays a BYO key to its provider. `/{provider}/…`
//! is the escape hatch, does not consult the catalog, and never translates. An unknown first
//! segment is a 404.

use crate::cache;
use crate::capture::CaptureBufs;
use crate::circuit_breaker::{Answered, CircuitBreaker, Permit};
use crate::concurrency::TenantSlot;
use crate::drain::{self, Drain};
use crate::key::{self, Identity};
use crate::metrics::{DrainEnd, KeyCooled, Rejection};
use crate::refusal::{self, Refusal, clip_catalog_name};
use crate::retry::{self, Decision, RelayRetry, Retry};
use crate::route::{self, Dialect, Provider, ProviderId};
use crate::signed_id;
use crate::state::{GatewayState, RequestId};
use crate::terminal::TerminalTracker;
use crate::{body_rules, control, peek, pin, remedy, translate, unpriced, usage};
use arrayvec::{ArrayString, ArrayVec};
use async_trait::async_trait;
use bytes::Bytes;
use pingora::http::ResponseHeader;
use pingora_core::Result;
use pingora_core::protocols::ALPN;
use pingora_core::protocols::http::HttpTask;
use pingora_core::protocols::http::subrequest::server::SubrequestHandle;
use pingora_core::upstreams::peer::HttpPeer;
use pingora_proxy::subrequest::{BodyMode, Ctx as SubrequestCtx};
use pingora_proxy::{FailToProxy, ProxyHttp, ProxyWarnLogContext, Session};
use providers::catalog::Caps;
use providers::catalog::strip_snapshot_date;
use std::borrow::Cow;
use std::fmt::Write as _;
use std::num::NonZeroU16;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Response header carrying the per-request id (`{instance}-{seq}`). Set on both the proxied
/// response and every reject body so a client can quote it and an oncall can grep for it.
const REQUEST_ID_HEADER: &str = "x-beyond-request-id";

/// Response headers naming what served the request, so a client can see a failover or a cache
/// replay without a log search. Provider on every upstream response and on a cache replay (the
/// provider that originally served it); the upstream model id on catalog walks, as the gateway
/// sent it to that provider; cache status only on a replay.
const PROVIDER_HEADER: &str = "x-beyond-provider";
const UPSTREAM_MODEL_HEADER: &str = "x-beyond-upstream-model";
const CACHE_STATUS_HEADER: &str = "x-beyond-cache-status";

/// OpenRouter's dashboard-attribution headers (https://openrouter.ai/docs/quickstart): purely
/// cosmetic on OpenRouter's side (their own cost/usage categorization), no effect on the request
/// or response. Static — this doesn't need to be configurable, just present. Only OpenRouter is in
/// `known_providers()` among the providers pi attributes to (NVIDIA NIM, Cloudflare, Vercel AI
/// Gateway aren't routed providers here), so this is the one case worth porting. Header names match
/// pi's current OpenRouter-specific set (`packages/coding-agent/src/core/provider-attribution.ts`):
/// `HTTP-Referer`, `X-OpenRouter-Title` (NOT the generic `X-Title` pi used for the now-removed Vercel
/// AI Gateway route), and `X-OpenRouter-Categories`.
const OPENROUTER_REFERER: &str = "https://beyond.dev";
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
///
/// `buf` is a ring once it holds exactly `USAGE_TAIL_CAP` bytes; below that it is in order and
/// `head` is 0. No separate flag: a full in-order buffer is a ring whose oldest byte is at 0.
struct UsageTail {
    buf: Vec<u8>,
    /// Next write index (the oldest byte). 0 until the buffer is a ring.
    head: usize,
}

impl UsageTail {
    fn ring(&self) -> bool {
        self.buf.len() == USAGE_TAIL_CAP
    }

    fn push(&mut self, data: &[u8]) {
        if !self.ring() {
            self.buf.extend_from_slice(data);
            if self.buf.len() > USAGE_TAIL_CAP {
                // Outgrown: keep the last cap bytes in order and switch to ring mode. This is the
                // only compaction that ever runs — from here on, writes wrap instead of shifting.
                let start = self.buf.len() - USAGE_TAIL_CAP;
                self.buf.copy_within(start.., 0);
                self.buf.truncate(USAGE_TAIL_CAP);
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
        // The wrapped remainder; empty (a no-op copy) when the chunk fit before the end. No `if`:
        // a guard here could only skip an empty copy, so it had no behaviour to test.
        let rest = data.len() - first;
        self.buf[..rest].copy_from_slice(&data[first..]);
        self.head = (self.head + data.len()) % USAGE_TAIL_CAP;
    }

    /// The retained bytes, oldest first. Rotates the ring into order once, at parse time.
    ///
    /// Stays a ring afterwards, with `head` back at 0 — a subsequent `push` then overwrites the
    /// oldest bytes, which is exactly right. In practice `logging` calls this once, after the body
    /// is complete.
    fn contiguous(&mut self) -> &[u8] {
        if self.head != 0 {
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
/// Connect retries stay same-peer / same-key. A received **5xx** is a vendor walk on a catalog
/// walk (`/auto`, or managed `/v1`) only (`upstream_response_filter`). A received **429** is a
/// same-provider key walk when another unused pool key remains — not a connect retry, not a vendor
/// failover, and not a breaker failure.
const MAX_CONNECT_RETRIES: u8 = 2;

/// Concurrent streams the gateway opens on one upstream H2 connection before it opens another
/// (D160). 100 is the least RFC 9113 §6.5.2 recommends a server allow; a provider whose
/// `SETTINGS_MAX_CONCURRENT_STREAMS` is lower lowers it (pingora takes the smaller).
const UPSTREAM_H2_MAX_STREAMS: usize = 100;

pub struct AiProxy {
    /// The gateway state, which lives as long as the process: a `&'static` borrow, so a request
    /// context holds it without an `Arc` clone (a shared-cache-line RMW pair on every request,
    /// fast rejects included; D92). Built with [`AiProxy::new`].
    pub state: &'static GatewayState,
}

impl AiProxy {
    /// A proxy over `state` for the rest of the process. Keeps one reference to it forever (the
    /// gateway builds one proxy at boot, and its state is never torn down before exit).
    pub fn new(state: Arc<GatewayState>) -> Self {
        if state.config.request_max_secs() > 0 {
            crate::deadline::start();
        }
        let state: &'static Arc<GatewayState> = Box::leak(Box::new(state));
        Self { state }
    }
}

/// Pingora's per-request context: the admitted request's state, plus what it holds that must be
/// given back however the request ends.
///
/// `logging` releases everything in the ordinary course. A panic in any proxy phase skips
/// `logging`, but it unwinds through pingora's request future and so drops this context, and
/// [`Drop`] releases whatever `logging` did not: the in-flight gauge, the SSE gauge, the tenant
/// concurrency slot, and an unresolved breaker permit (given back without an outcome, so a
/// half-open breaker's only probe permit cannot be stranded by a gateway bug).
///
/// Derefs to the `Option<RequestCtx>` the hooks were written against.
pub struct Ctx {
    rc: Option<RequestCtx>,
    held: Held,
}

/// The releasable state of one request. Kept beside, not inside, [`RequestCtx`]: none of it but the
/// one-byte `drain` is touched per response chunk, and `RequestCtx`'s size is (see its size test).
struct Held {
    state: &'static GatewayState,
    /// Set at the top of `request_filter`, so an error answered before admission (a body read
    /// failure, say) still carries the request id.
    request_id: Option<RequestId>,
    /// Counted on `ai_requests_in_flight`.
    in_flight: bool,
    /// Counted on `ai_active_streams`.
    active_stream: bool,
    /// One of this tenant's `tenant_max_in_flight` slots, when the request holds one.
    tenant: Option<TenantSlot<'static>>,
    /// Bytes this request holds in the process body budget (`max_buffered_body_bytes`).
    body_bytes: usize,
    /// A `FullBody` re-run: its parent reserved the budget for both copies of the body.
    body_exempt: bool,
    /// The client hung up on an armed stream ([`Drain::Armed`]): set by
    /// `suppress_proxy_warn_log`, which pingora calls with a shared `&Ctx`, hence atomic. Read by
    /// `response_body_filter` only while a drain is armed. While set, the request is counted on
    /// `ai_usage_drains_in_flight`, until `close_drain` clears it. See [`crate::drain`].
    client_left: AtomicBool,
    /// Drain-on-cancel ([`crate::drain`]): armed at a managed 2xx stream's head from a provider
    /// that keeps generating after a disconnect, `Off` everywhere else. The one field here read
    /// per response chunk, a one-byte compare (kept off `RequestCtx`, whose size is tested).
    drain: Drain,
}

impl Held {
    fn admit(&mut self) {
        if !self.in_flight {
            self.in_flight = true;
            self.state.metrics.requests_in_flight.inc();
        }
    }

    fn release_in_flight(&mut self) {
        if std::mem::take(&mut self.in_flight) {
            self.state.metrics.requests_in_flight.dec();
        }
    }

    fn open_stream(&mut self) {
        if !self.active_stream {
            self.active_stream = true;
            self.state.metrics.active_streams.inc();
        }
    }

    fn release_stream(&mut self) {
        if std::mem::take(&mut self.active_stream) {
            self.state.metrics.active_streams.dec();
        }
    }

    /// The client left an armed stream (pingora ignored its downstream error): the drain starts,
    /// counted once. Takes `&self`: pingora reports it through a hook that holds a shared `&Ctx`.
    fn client_left(&self) {
        if !self.client_left.swap(true, Ordering::Relaxed) {
            self.state.metrics.usage_drains_in_flight.inc();
        }
    }

    /// The first chunk after the client left: stop relaying, and give back the tenant's
    /// concurrency slot, which bounds what a client has open, not what the gateway finishes
    /// reading for it.
    fn open_drain(&mut self) {
        self.drain = Drain::Draining;
        self.release_tenant();
    }

    /// End the drain this request started, if any, counting how it ended.
    fn close_drain(&mut self, end: DrainEnd) {
        self.drain = Drain::Off;
        if self.client_left.swap(false, Ordering::Relaxed) {
            self.state.metrics.usage_drains_in_flight.dec();
            self.state.metrics.drain_ended(end);
        }
    }

    /// Make sure at least `total` bytes of buffered body are reserved for this request. `false`
    /// when the process budget cannot cover them (nothing further is reserved). Grows in whole
    /// MiB, so a chunked body costs one shared atomic per MiB rather than per chunk.
    fn reserve_body(&mut self, total: usize) -> bool {
        if self.body_exempt || total <= self.body_bytes {
            return true;
        }
        let Some(budget) = self.state.body_budget.as_ref() else {
            return true;
        };
        const STEP: usize = 1 << 20;
        let rounded = total.div_ceil(STEP).saturating_mul(STEP);
        for target in [rounded, total] {
            if budget.try_reserve(target - self.body_bytes) {
                self.body_bytes = target;
                return true;
            }
        }
        false
    }

    fn release_body(&mut self) {
        let n = std::mem::take(&mut self.body_bytes);
        if n > 0
            && let Some(budget) = self.state.body_budget.as_ref()
        {
            budget.release(n);
        }
    }

    fn release_tenant(&mut self) {
        self.tenant = None;
    }
}

impl std::ops::Deref for Ctx {
    type Target = Option<RequestCtx>;
    fn deref(&self) -> &Self::Target {
        &self.rc
    }
}

impl std::ops::DerefMut for Ctx {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.rc
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        self.held.release_in_flight();
        self.held.release_stream();
        self.held.release_tenant();
        self.held.release_body();
        // `logging` closes a drain with how it ended; only a panic past it lands here.
        self.held.close_drain(DrainEnd::Error);
        // An unresolved breaker permit is given back by its `Leg` as the context goes.
        drop(self.rc.take());
        self.held.state.request_ended();
    }
}

/// Per-request context. `None` until `request_filter` admits the request; short-circuited
/// requests (auth/deny failures) leave it `None`, so later filters no-op.
pub struct RequestCtx {
    /// Who pays: a verified Beyond identity, or the caller's own provider key.
    billing: Billing,
    dialect: Dialect,
    /// The provider this attempt goes to, the pool key it sends, and the breaker permit it owes.
    leg: Leg,
    /// The path (+ query) this attempt sends upstream, applied as the upstream URI in
    /// `upstream_request_filter`. A catalog walk's is set per attempt in `upstream_peer`; before
    /// its first it is `Inbound`, and nothing reads it until a provider was called.
    forward_path: ForwardPath,
    /// Model the client *requested*, extracted from the request body. This is the billing-log
    /// **fallback** — the authoritative value is the model the provider echoes in its response (see
    /// `resp_model_scanner`), because a client may send an alias (`gpt-4o`) that the provider resolves
    /// to and bills under a pinned id (`gpt-4o-2024-08-06`).
    model: String,
    /// What this attempt has made of the request body so far. Rebuilt for every attempt.
    attempt: Attempt,
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
    /// Upstream HTTP status, set in `response_filter` once the response head arrives, which is
    /// also where the breaker permit is resolved (`5xx` → failure, any other response → success:
    /// the provider answered, and a `429` is a healthy throttle). A `None` here at `logging` with an
    /// upstream error → failure (connect/read failed before any response). A `NonZeroU16`, as no
    /// status is `0`: the niche keeps it two bytes beside the other small fields. Read it through
    /// [`RequestCtx::upstream_status`].
    upstream_status: Option<NonZeroU16>,
    /// How this attempt's body goes upstream: streamed through, or buffered and rewritten.
    plan: BodyPlan,
    start: Instant,
    /// When this request must end (`request_max_secs`), in [`crate::deadline`]'s units;
    /// [`crate::deadline::NONE`] when the ceiling is off. A `FullBody` attempt carries its
    /// parent's, so a re-run does not restart the clock.
    deadline: u64,
    /// Model-routing state — `Some` for `/auto` and for a managed bare `/v1` catalog walk. `None`
    /// keeps every provider-routed request on exactly the code it ran before model routing existed.
    /// See [`ModelRouting`] for why it is boxed rather than inline.
    auto: Option<Box<ModelRouting>>,
    /// Control-surface state — `Some` only when the caller sent an `x-beyond-*` header or the tenant
    /// is being captured. Boxed for [`ModelRouting`]'s measured reason: `RequestCtx` is touched once
    /// per response chunk, so inline fields cost streaming latency on *every* request to buy nothing
    /// for the overwhelming majority that use neither feature. `None` is 8 bytes and one discriminant
    /// check per chunk.
    control: Option<Box<RequestControl>>,
    /// Process-unique id for this request (`{instance}-{seq}`), echoed in the `x-beyond-request-id`
    /// response header and the `ai.usage` event so a client report ties back to a log line.
    request_id: RequestId,
    /// The body is fed to the attempt's `input_tally` as it streams: a managed provider-routed body that may
    /// outgrow pingora's 64 KiB retry buffer (a declared length past it, or none), the one case
    /// where no copy of the body is left to read in `logging`. A catalog walk's body is in the
    /// retry buffer or, past it, held by the `FullBody` parent.
    tally_eager: bool,
    /// Response bytes relayed — managed only. Scales the retained tail up to the whole response in
    /// `usage::estimate_stream_output` / `estimate_body_output`. One add per chunk; nothing is
    /// scanned.
    resp_bytes: u32,
    /// How far the current attempt got toward a provider. Lets a billing row tell a request no
    /// provider was ever called for (every breaker open) from one that failed upstream.
    upstream_phase: UpstreamPhase,
    /// Set on a managed response with status >= 400: its body is scrubbed of the pool key this
    /// attempt sent (see [`Redact`]). Boxed: `None` on every other response.
    redact: Option<Box<Redact>>,
    /// Whether the bytes sent to the client so far end with the stream's terminal event (`[DONE]`,
    /// `message_stop`, `response.completed`). Fed on managed streams only; read in `logging` so a
    /// client that closes once it has the whole answer is not a cancel (D120, D122).
    terminal: TerminalTracker,
    /// A managed Responses relay to a provider's store (`signed_id`): the ids in its 2xx are signed
    /// for this tenant, and the ids the client sent were verified and are stripped back in
    /// `request_body_filter`. Boxed: `None` on every other request.
    signed: Option<Box<signed_id::Relay>>,
}

/// What one upstream attempt has made of the request body as `request_body_filter` streams it past.
///
/// `upstream_peer` builds a fresh one for every attempt ([`RequestCtx::begin_attempt`]), because
/// pingora replays its buffered request body through `request_body_filter` on a retry
/// (`proxy_h1.rs`'s `send_body_to_pipe`, and the h2 twin): anything an earlier attempt accumulated
/// would otherwise have the replayed prefix appended to it.
///
/// - `req_buf` would hold the prefix twice, and `peek::scan_buffered` would then plan the splice
///   against the **first** copy — handing the upstream a body with a duplicated fragment, which it
///   rejects with a `400` that reads like a client error. `logging` even records that `400` as a
///   breaker *success* (the provider answered), so nothing surfaces it as our fault.
/// - `model_scanner`'s brace depth would be permanently offset by the extra `{`, so `at_key_level`
///   never matches again and the billing row ships `requested_model = ""`.
/// - `upload` would double-count the prefix against `MAX_REQUEST_BODY`.
///
/// A whole new value per attempt has no field a reset could forget.
struct Attempt {
    /// How much of the request body this attempt has fed through `request_body_filter`.
    upload: UploadProgress,
    /// The body, buffered for a rewrite ([`BodyPlan::rewrites_body`]); empty for every request
    /// that streams its body through.
    req_buf: Vec<u8>,
    /// Scans a body that streams through (managed only) for its root `model`.
    model_scanner: peek::ModelScanner,
    /// Text bytes of the request body (base64 payloads excluded): the input-token estimate for a
    /// billing row the provider's usage did not fill (a stream cut short before its usage block, on
    /// a wire that reports input only at the end). See `usage::InputTally`. Fed as the body streams
    /// past only when `tally_eager`; otherwise `logging` reads the body itself, and only when it
    /// needs an estimate ([`input_estimate`]).
    input_tally: usage::InputTally,
    /// This attempt's upstream read timeout is the time left before `deadline` (shorter than the
    /// silence bound, see `cap_read_at_deadline`), so a read timeout on it is the deadline passing,
    /// not the provider going quiet.
    read_capped: bool,
    /// This is a [`FullBody`] attempt that recorded a [`RelayRetry`]: the parent is discarding its
    /// response and re-running. It still feeds the breaker; it writes no `ai.usage` or
    /// `ai.payload` row (the attempt that serves does). An abandoned attempt is the subrequest's
    /// last: it never asks pingora for another.
    relay_abandoned: bool,
}

impl Attempt {
    /// An attempt that has seen nothing yet, buffering into `req_buf` (empty, perhaps pre-sized).
    fn new(req_buf: Vec<u8>) -> Self {
        Self {
            upload: UploadProgress::default(),
            req_buf,
            model_scanner: peek::ModelScanner::new(),
            input_tally: usage::InputTally::default(),
            read_capped: false,
            relay_abandoned: false,
        }
    }
}

/// How a request's body goes upstream on the current attempt: streamed through untouched, or
/// buffered whole and rewritten. Decided where the route is (`request_filter`) and again per
/// candidate on a catalog walk (`apply_serving_candidate`), never re-derived where the body is
/// handled, so the buffering in `request_body_filter` and the re-framing in
/// `upstream_request_filter` read one answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BodyPlan {
    /// Make a streaming body ask for its usage chunk ([`stream_options_eligible`]): inject
    /// `stream_options.include_usage` if it streams without it, or force a client-sent one to ask,
    /// so the billable token count is guaranteed.
    stream_options: bool,
    /// Managed `/{provider}/…` billable call (every allowlisted endpoint but the free token counts):
    /// held whole so a request whose root `model` names no catalog row ([`price_row`]) is refused
    /// before any byte of it goes upstream, with the catalog walk's 404. A managed key runs priced
    /// models only (D267). BYO keys never set it.
    ///
    /// The same buffer refuses a `/responses` body's `background: true` (D202): that endpoint is
    /// billable, so it is always checked here (a catalog walk reads its body in `request_filter`
    /// and refuses there).
    catalog_check: bool,
    /// See [`Self::rewrites_body`].
    rewrites: bool,
}

impl BodyPlan {
    /// The plan for a request whose body the gateway may edit for these reasons: `stream_options`
    /// and `catalog_check` (see the fields), a catalog walk (`catalog_walk`), and a managed
    /// Responses relay to a provider's store (`signed`).
    fn new(stream_options: bool, catalog_check: bool, catalog_walk: bool, signed: bool) -> Self {
        Self {
            stream_options,
            catalog_check,
            rewrites: stream_options || catalog_check || catalog_walk || signed,
        }
    }

    /// Whether this request's body is buffered and may leave with a different length than it
    /// arrived — which is one question, not two, because the answer drives *both* the buffering in
    /// `request_body_filter` and the `Content-Length`/`transfer-encoding` re-framing in
    /// `upstream_request_filter`. Splitting them is how you get a body whose framing disagrees with
    /// its bytes.
    ///
    /// Why a body gets rewritten:
    /// - `stream_options`: splicing `stream_options` into a managed OpenAI stream.
    /// - a catalog walk: re-spelling `model` for the candidate serving this attempt, translating
    ///   between Chat Completions, Messages and Responses when the wires differ, and the same-wire
    ///   edits a walk makes (the output-limit cap, `max_tokens` respelled for native OpenAI, dropped
    ///   nulls and unreplayable reasoning).
    /// - `catalog_check`: a managed `/{provider}/…` billable body is held whole so a model outside
    ///   the catalog (D267), or a `/responses` body's `background: true` (D202), is refused before
    ///   any byte goes upstream.
    /// - `signed`: a managed Responses relay to a provider's store swaps each tenant-signed id for
    ///   the provider's (`signed_id`, D230).
    ///
    /// Several can apply to the same request, in which case every edit is made to the one buffer.
    fn rewrites_body(self) -> bool {
        self.rewrites
    }
}

/// Who pays for a request, and so whether anything tenant-scoped applies to it.
///
/// The only way to a tenant id is `Billing::Managed`, built from what `Keyring::verify` proved
/// (an [`Identity`]), so no unverified id can reach the deny set, a capture rule or a billing row,
/// and a BYO request has no tenant id at all rather than a zero one.
enum Billing {
    /// A Beyond virtual key: the pool key is swapped in, and the request writes a billing row.
    Managed(Managed),
    /// The caller's own provider key, passed through untouched: no identity, no billing row.
    Byo,
}

/// A managed request's verified identity, and the billing facts its row reads.
struct Managed {
    identity: Identity,
    /// Billing facts tapped beside the usage block ([`BillingTaps`]), made at the first one.
    /// Boxed for [`ModelRouting`]'s reason; `None` until a fact is tapped (a cache hit taps none).
    taps: Option<Box<BillingTaps>>,
}

impl Billing {
    /// Managed for a verified identity, BYO without one.
    fn of(identity: Option<Identity>) -> Self {
        match identity {
            Some(identity) => Billing::Managed(Managed {
                identity,
                taps: None,
            }),
            None => Billing::Byo,
        }
    }

    fn identity(&self) -> Option<&Identity> {
        match self {
            Billing::Managed(m) => Some(&m.identity),
            Billing::Byo => None,
        }
    }

    /// The facts tapped so far; `None` on BYO and before the first.
    fn taps(&self) -> Option<&BillingTaps> {
        match self {
            Billing::Managed(m) => m.taps.as_deref(),
            Billing::Byo => None,
        }
    }

    /// See [`Self::taps`].
    fn taps_mut(&mut self) -> Option<&mut BillingTaps> {
        match self {
            Billing::Managed(m) => m.taps.as_deref_mut(),
            Billing::Byo => None,
        }
    }

    /// A managed request's taps, made here at its first fact; `None` on BYO, which taps nothing.
    fn tap(&mut self) -> Option<&mut BillingTaps> {
        match self {
            Billing::Managed(m) => Some(m.taps.get_or_insert_with(Box::default)),
            Billing::Byo => None,
        }
    }
}

/// One candidate's share of a request: the provider it goes to, the pool key it sends, its connect
/// bookkeeping, and the breaker permit it owes. A candidate switch builds a new leg, so nothing of
/// provider A's (its key index, its address cursor, its refused-stream resend, its permit) can
/// reach provider B.
///
/// The permit is the breaker ledger: `Some` iff one `allow()` against `provider` is unresolved.
/// Every resolution takes it, so an attempt can never be recorded twice; resolving with the permit
/// (not just "the breaker") lets a half-open breaker ignore an attempt whose probe it has since
/// reclaimed (D256). A leg dropped with its permit still out (a panic unwinding the request) gives
/// it back without an outcome, so a half-open breaker's only probe permit is never stranded.
struct Leg {
    /// The resolved upstream provider, shared from the boot-time registry: a cheap `Arc` clone,
    /// nothing re-allocated per request.
    provider: Arc<Provider>,
    /// From `allow()` on `provider`'s breaker. `None` once resolved, and when it has no breaker.
    permit: Option<Permit>,
    /// Index into `provider.pool_auth` of the key this attempt sends. Starts on
    /// `Provider::first_key` (past keys cooling off from a 401) unless a `FullBody` re-run resumes
    /// a key walk; advanced on a managed 429 or 401 when another unused key remains.
    pool_key: u8,
    /// Which of the provider's resolved addresses this attempt dials: advanced by each connect
    /// retry (see `fail_to_connect`).
    addr: u8,
    /// This attempt is the one resend a refused H2 stream gets on its candidate (D72, D91). A
    /// second refusal there is a provider failure: fail over, or end the request.
    refused_resent: bool,
    /// The retry the hook that asked pingora for another attempt declared, read by `upstream_peer`
    /// when that attempt starts. `None` before the first attempt.
    retry: Option<Retry>,
}

impl Leg {
    fn new(provider: Arc<Provider>, permit: Option<Permit>, pool_key: u8) -> Self {
        Self {
            provider,
            permit,
            pool_key,
            addr: 0,
            refused_resent: false,
            retry: None,
        }
    }

    /// The provider's pool key after this one, when it has another.
    fn next_key(&self) -> Option<u8> {
        let next = usize::from(self.pool_key).saturating_add(1);
        (next < self.provider.pool_auth.len())
            .then(|| u8::try_from(next).ok())
            .flatten()
    }

    /// The permit, with the breaker it is owed to.
    fn owed(&mut self) -> Option<(&CircuitBreaker, Permit)> {
        let permit = self.permit.take()?;
        Some((self.provider.breaker.as_ref()?, permit))
    }

    /// The provider answered. The [`Answered`] is for a success its body may still overturn.
    fn succeed(&mut self) -> Option<Answered> {
        self.owed().map(|(b, p)| b.record_success_for(p))
    }

    /// The provider failed this attempt.
    fn fail(&mut self) {
        if let Some((b, p)) = self.owed() {
            b.record_failure_for(p);
        }
    }

    /// The attempt ended with no provider outcome (the client left, or the gateway's own doing).
    fn release(&mut self) {
        if let Some((b, p)) = self.owed() {
            b.release(p);
        }
    }
}

impl Drop for Leg {
    fn drop(&mut self) {
        self.release();
    }
}

/// Billing facts a managed request taps beside its usage block, for the `ai.usage` row.
#[derive(Default)]
struct BillingTaps {
    /// The request's price knobs ([`peek::REQUEST_KNOB_KEYS`]), fed as the body streams only when
    /// no copy of it will be left for `logging` (`tally_eager`); otherwise `logging` scans the copy
    /// (see [`requested_knobs`]).
    knobs: Option<(peek::ModelScanner, peek::Kept)>,
    /// What `resp_model_scanner` keeps beside the model: the response's id and serving host
    /// ([`peek::RESPONSE_EXTRA_KEYS`]).
    resp_kept: Option<peek::Kept>,
    /// Hosted-tool items of a Responses answer (see [`usage::ToolTally`]).
    tools: Option<usage::ToolTally>,
    /// OpenRouter's `X-Generation-Id` response header: the id its generation API takes.
    generation_id: Option<usage::IdStr>,
    /// The vendor's request id header (`request-id`, `x-request-id`, `x-amzn-requestid`): what its
    /// support and logs key on.
    request_id: Option<usage::IdStr>,
    /// The request offers OpenAI's `web_search_preview` tool: its `web_search_call` items are
    /// counted as `web_search_preview`, priced apart (see `unpriced::Inspection`).
    web_search_preview: bool,
}

/// What a request asked for that changes its price, as the client sent it (the served values come
/// from the response). String knobs are kept when they are short tokens; object and array knobs
/// (`provider`, `plugins`, an object `container`) as raw JSON, at most [`peek::RAW_CAPTURE`] bytes.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct RequestedKnobs {
    pub service_tier: Option<usage::ServiceTier>,
    pub speed: Option<usage::ServiceTier>,
    pub inference_geo: Option<usage::ServiceTier>,
    pub provider_routing: Option<String>,
    pub plugins: Option<String>,
    pub container: Option<String>,
}

impl RequestedKnobs {
    fn from_scan(mut s: peek::Kept) -> Self {
        let tier = |v: Option<String>| v.as_deref().and_then(usage::tier_from);
        // A raw value is JSON the scanner copied byte for byte; a string knob arrives unquoted.
        let raw = |v: Option<String>| v.filter(|v| !v.is_empty());
        RequestedKnobs {
            service_tier: tier(s.take(0)),
            speed: tier(s.take(1)),
            inference_geo: tier(s.take(2)),
            provider_routing: raw(s.take(3)),
            plugins: raw(s.take(4)),
            container: raw(s.take(5)),
        }
    }
}

/// The request's price knobs: from the scan fed as the body streamed (`tally_eager`), else from
/// the body `logging` still holds — the `FullBody` parent's copy or pingora's retry buffer, the
/// copies [`input_estimate`] reads. That is the client's body, before any rewrite.
fn requested_knobs(session: &Session, rc: &mut RequestCtx) -> RequestedKnobs {
    if let Some((_, k)) = rc.billing.taps_mut().and_then(|t| t.knobs.take()) {
        return RequestedKnobs::from_scan(k);
    }
    let mut kept = peek::Kept::request_knobs();
    if let Some(body) = full_body_ctx(session)
        .map(|fb| fb.body)
        .or_else(|| session.as_ref().get_retry_buffer())
    {
        peek::ModelScanner::new().feed_keeping(&body, &mut kept);
    }
    RequestedKnobs::from_scan(kept)
}

/// Which price a provider applies to the endpoint that served, where one provider has several:
/// Bedrock's `global.` inference profiles are its base price, and its geographic profiles (`us.`,
/// `eu.`, `jp.`, `au.`, `apac.`) and single-region model ids carry a 10% premium on Claude 4.5
/// and later; OpenRouter's in-region hosts (`us.openrouter.ai`, `eu.openrouter.ai`) pass on the
/// provider's regional surcharge. `None` where the provider has one price.
fn price_variant(provider: &str, upstream_model: &str, host: &str) -> Option<&'static str> {
    match provider {
        "bedrock" => Some(if upstream_model.starts_with("global.") {
            "global"
        } else {
            "regional"
        }),
        "openrouter" => {
            (host != "openrouter.ai" && host.ends_with(".openrouter.ai")).then_some("regional")
        }
        _ => None,
    }
}

/// Providers documented to keep generating, and billing, after the client disconnects
/// mid-stream: a cancelled row's tokens are what was relayed, not what was billed.
const MAY_CONTINUE: [ProviderId; 3] = [
    ProviderId::OpenRouter,
    ProviderId::Bedrock,
    ProviderId::Groq,
];

/// [`MAY_CONTINUE`] by name: a billing row names its provider, and its facts are derived from the
/// row's own fields (as its price is).
const MAY_CONTINUE_PROVIDERS: [&str; 3] = [
    MAY_CONTINUE[0].name(),
    MAY_CONTINUE[1].name(),
    MAY_CONTINUE[2].name(),
];

/// Whether this row's provider may have generated (and billed) past the point the row counts:
/// a stream the client cancelled, or one cut short, on a provider in [`MAY_CONTINUE_PROVIDERS`],
/// unless the gateway read that stream to its end after the client left (`drained`): then the row
/// counts everything the provider generated.
fn upstream_may_continue(
    provider: Option<&str>,
    outcome: &str,
    streaming: bool,
    drained: bool,
) -> bool {
    streaming
        && !drained
        && matches!(outcome, "client_cancelled" | "cut_short")
        && provider.is_some_and(|p| MAY_CONTINUE_PROVIDERS.contains(&p))
}

/// One response chunk of an armed stream ([`Drain::Armed`] or later): `true` when it is drained
/// (its bytes already fed the usage taps, and are not sent), `false` while the client still reads.
/// The first chunk after the client left starts the drain; the stream's end settles it.
fn drain_chunk(rc: &mut RequestCtx, held: &mut Held, end_of_stream: bool) -> bool {
    match held.drain {
        Drain::Off => return false,
        Drain::Armed => {
            if !held.client_left.load(Ordering::Relaxed) {
                return false;
            }
            held.open_drain();
            // The client got part of this answer: the rest is never stored as a cache fill.
            if let Some(a) = rc.auto.as_mut()
                && matches!(a.cache, Some(cache::Pending::Fill { .. }))
            {
                a.cache = None;
            }
        }
        Drain::Draining | Drain::Settled => {}
    }
    if end_of_stream {
        held.drain = Drain::Settled;
    }
    true
}

/// How a drain ended, at `logging`: `None` when none started (the client never left an armed
/// stream). A drain whose upstream ended is settled even if the request then errored, which can
/// only be the dead client's last write (the end of a chunked body); the deadline or an upstream
/// failure first leaves the estimate.
fn drain_end(drain: Drain, client_left: bool, e: Option<&pingora_core::Error>) -> Option<DrainEnd> {
    if !client_left {
        return None;
    }
    Some(match (drain, e) {
        (Drain::Settled, _) => DrainEnd::Settled,
        (_, Some(e)) if is_deadline(e) => DrainEnd::Deadline,
        (_, Some(e)) if e.esource() != &pingora_core::ErrorSource::Downstream => DrainEnd::Error,
        _ => DrainEnd::Settled,
    })
}

/// Whether a response head starts a stream to drain if the client hangs up ([`crate::drain`]): a
/// managed 2xx stream from a provider in [`MAY_CONTINUE`]. BYO writes no billing row, and a
/// provider that stops generating when the client leaves bills only what it relayed.
fn drains_on_cancel(rc: &RequestCtx, status: u16) -> bool {
    rc.managed()
        && rc.streaming
        && (200..300).contains(&status)
        && rc
            .leg
            .provider
            .id
            .is_some_and(|id| MAY_CONTINUE.contains(&id))
}

/// The upstream response header carrying the vendor's request id, in the order they are tried.
const REQUEST_ID_HEADERS: [&str; 3] = ["request-id", "x-request-id", "x-amzn-requestid"];

/// At a managed response head: read the vendor's ids from its headers, and start counting
/// hosted-tool items when a Responses endpoint answered. A few map lookups per response; a fresh
/// tally per head, so an earlier attempt's never leaks into the one that serves.
fn tap_response_head(rc: &mut RequestCtx, resp: &ResponseHeader) {
    let header_id = |name: &str| {
        resp.headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(usage::id_from)
    };
    let generation_id = header_id("x-generation-id");
    let request_id = REQUEST_ID_HEADERS.iter().find_map(|h| header_id(h));
    let responses = match rc.auto.as_ref() {
        Some(a) => catalog_serving_endpoint(a) == Some(route::Endpoint::Responses),
        None => {
            rc.dialect == Dialect::OpenAi
                && rc.forward_path.as_str().is_some_and(|p| {
                    let p = p.split_once('?').map_or(p, |(path, _)| path);
                    p.ends_with("/responses") || p.ends_with("/responses/compact")
                })
        }
    };
    // Always made on a managed response: `resp_model_scanner` keeps the response's id in it.
    let Some(taps) = rc.billing.tap() else {
        return;
    };
    taps.generation_id = generation_id;
    taps.request_id = request_id;
    taps.tools = responses.then(|| usage::ToolTally::new(rc.streaming));
}

/// Fold what the taps saw into the row's usage: the header generation id over the body's (the
/// response scanner's, from the head, over the usage block's), the serving host the head named,
/// and the Responses tool items.
fn merge_taps(rc: &mut RequestCtx, usage: &mut usage::Usage) {
    let Some(t) = rc.billing.taps_mut() else {
        return;
    };
    let head_id = t
        .resp_kept
        .as_ref()
        .and_then(|k| k.get(0))
        .and_then(usage::id_from);
    let head_host = t
        .resp_kept
        .as_ref()
        .and_then(|k| k.get(1))
        .and_then(usage::host_from);
    let (hdr_id, tools, preview) = (t.generation_id, t.tools.take(), t.web_search_preview);
    if let Some(id) = hdr_id.or(head_id) {
        usage.upstream.generation_id = Some(id);
    }
    if head_host.is_some() {
        usage.upstream.served_by = head_host;
    }
    if let Some(mut t) = tools {
        t.finish();
        let mut counted = t.tools;
        if preview {
            counted.web_search_preview = counted.web_search;
            counted.web_search = 0;
        }
        usage.server_tools.merge_max(&counted);
        if usage.upstream.container_id.is_none() {
            usage.upstream.container_id = t.container_id;
        }
        usage.server_tool_calls = usage
            .server_tool_calls
            .max(u64::from(usage.server_tools.web_search));
    }
}

/// Which of a row's counts an estimate replaced (see `logging`), and what the estimate cannot see.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct EstimatedParts {
    input: bool,
    output: bool,
}

impl EstimatedParts {
    /// The row's `usage_estimated_parts`: `input`, `output`, `input,output`, or absent.
    fn parts(self) -> Option<&'static str> {
        match (self.input, self.output) {
            (true, true) => Some("input,output"),
            (true, false) => Some("input"),
            (false, true) => Some("output"),
            (false, false) => None,
        }
    }

    /// The row's `usage_estimate_excludes`: `cache` when the input is estimated (the pre-token
    /// count is the whole prompt as plain input: some of it may have been cache reads, cheaper, or
    /// cache writes, dearer), `reasoning` when the output is estimated and the provider reported no
    /// reasoning count (hidden reasoning is billed as output and invisible in the stream).
    fn excludes(self, u: &usage::Usage) -> Option<&'static str> {
        let reasoning = self.output && u.reasoning_tokens.is_none();
        match (self.input, reasoning) {
            (true, true) => Some("cache,reasoning"),
            (true, false) => Some("cache"),
            (false, true) => Some("reasoning"),
            (false, false) => None,
        }
    }
}

/// Scrubs a managed error body (status >= 400) as it streams past: the pool key the attempt sent
/// (D66), and a provider-account remedy (D174). The one rewrite of a provider's error body, made
/// before translation, capture and the cache see it.
///
/// **Pool key.** A provider, or a proxy in between, that echoes the credential it received in an
/// error message would otherwise hand Beyond's key to the client. Each occurrence is overwritten in
/// place with [`REDACTED`] padded to the key's length, so the body keeps its length. The last
/// `key.len() - 1` bytes of each chunk are held back until the next one, so a key split across
/// chunks is caught too; memory is bounded by the key, not the body. Each JSON-escaped spelling of
/// the key (`\/`, `+`: [`route::key_finders`]) is overwritten the same way, since a JSON client
/// decodes it to the key (D203).
///
/// **Account remedy.** A JSON error (`whole`) is held until its end, at most
/// [`remedy::MAX_BODY`], and handed to [`remedy::neutralize`] after the key is masked: "add your
/// own key", a billing page and the like become a neutral message (`response_filter` dropped the
/// `Content-Length`, since that changes the length). Past the cap it streams as above. An
/// out-of-credit answer sets `unfunded`, which `logging` reads to cool the key (D180).
#[derive(Default)]
struct Redact {
    carry: Vec<u8>,
    whole: bool,
    /// The response's status: whether a remedy reads as a rate limit (see [`remedy::neutralize`]).
    status: u16,
    unfunded: bool,
}

/// What a scrubbed pool key reads as. Same length as the key (padded with `*`), truncated when the
/// key is shorter.
const REDACTED: &[u8] = b"[redacted]";

impl Redact {
    /// Scrub `chunk` (with the bytes held back from the last one) and return what may be relayed
    /// now: everything at `end_of_stream`, nothing while a `whole` body is held, otherwise all but
    /// a key-sized tail. `keys` are the pool key's boot-built searchers
    /// ([`route::PoolAuth::finders`]: the key as sent, then its JSON-escaped spellings), empty when
    /// the attempt sent none.
    fn feed(
        &mut self,
        keys: &[memchr::memmem::Finder<'_>],
        chunk: Option<Bytes>,
        end_of_stream: bool,
    ) -> Option<Bytes> {
        let keys = match keys.first() {
            Some(k) if !k.needle().is_empty() => keys,
            _ => &[],
        };
        if keys.is_empty() && !self.whole && self.carry.is_empty() {
            return chunk;
        }
        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(chunk.as_deref().unwrap_or(&[]));
        if self.whole && !end_of_stream {
            if buf.len() <= remedy::MAX_BODY {
                self.carry = buf;
                // Empty, not `None`: `None` would end the body.
                return Some(Bytes::new());
            }
            self.whole = false;
        }
        for key in keys {
            mask_all(&mut buf, key);
        }
        if self.whole {
            if let Some(n) = remedy::neutralize(&buf, self.status) {
                buf = n.body;
                self.unfunded = n.unfunded;
                // The rewrite decoded every string and wrote it back unescaped: an echo in a
                // spelling no searcher knew (`A` for `A`) is now the key as sent (D203).
                if let Some(key) = keys.first() {
                    mask_all(&mut buf, key);
                }
            } else if let Some(key) = keys.first()
                && memchr::memmem::find(&buf, b"\\u").is_some()
                && let Some(mut plain) = serde_json::from_slice::<serde_json::Value>(&buf)
                    .ok()
                    .and_then(|v| serde_json::to_vec(&v).ok())
                && mask_all(&mut plain, key)
            {
                // Any other `\uXXXX` spelling of the key: a JSON client decodes it to the key, so
                // the body is relayed decoded and masked (D203). Only a body with such an escape
                // and the key behind it is re-serialized.
                buf = plain;
            }
        } else if !end_of_stream {
            let longest = keys.iter().map(|k| k.needle().len()).max().unwrap_or(0);
            let keep = longest.saturating_sub(1).min(buf.len());
            self.carry = buf.split_off(buf.len() - keep);
        }
        Some(Bytes::from(buf))
    }
}

/// Overwrite every occurrence of `key`'s needle in `buf` with [`REDACTED`], padded to its length.
/// Returns whether there was one.
fn mask_all(buf: &mut [u8], key: &memchr::memmem::Finder<'_>) -> bool {
    let len = key.needle().len();
    let mut at = 0;
    let mut found = false;
    while let Some(i) = key.find(&buf[at..]) {
        let hit = &mut buf[at + i..at + i + len];
        for (j, b) in hit.iter_mut().enumerate() {
            *b = REDACTED.get(j).copied().unwrap_or(b'*');
        }
        at += i + len;
        found = true;
    }
    found
}

/// How far a request got toward a provider: what its billing row may claim about who was called.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum UpstreamPhase {
    /// No provider was called: no attempt has been handed a peer.
    #[default]
    None,
    /// `upstream_peer` handed pingora this attempt's peer. A provider was called, even if the
    /// connect then failed.
    Attempted,
    /// The connection is up and the request is going out (`upstream_request_filter` ran).
    Connected,
}

/// Whether the provider has this attempt's whole request: the connection was up, the client's body
/// was read to its end (and so forwarded), and nothing failed writing it upstream. A request in
/// that state is one the provider can bill for even when nothing comes back; one that is not (a
/// connect failure, a reset mid-upload) never reached it.
///
/// The client side stands in for the upstream side: pingora forwards each body chunk as it reads
/// it, and a failure on the way out surfaces as a write error. So this can only over-report by a
/// chunk still in flight when a read-side error lands, never miss a request the provider has.
fn body_delivered(session: &mut Session, rc: &RequestCtx, e: Option<&pingora_core::Error>) -> bool {
    use pingora_core::ErrorType::{WriteError, WriteTimedout};
    rc.upstream_phase == UpstreamPhase::Connected
        && session.as_mut().is_body_done()
        && !e.is_some_and(|e| {
            (e.esource() == &pingora_core::ErrorSource::Upstream
                && (matches!(e.etype(), WriteError | WriteTimedout) || h2_body_unsent(e)))
                || client_body_cut_short(e)
        })
}

/// Whether the client closed its HTTP/1.1 connection before its body's framing-defined end
/// (D259): a `Content-Length` body with bytes still to come, or a chunked one without its
/// terminating chunk. Pingora then marks the body *done* (`is_body_done`), though never complete,
/// so without this the half a provider was sent read as the whole request, and `logging` billed
/// it an estimate. Like [`h2_body_unsent`], the context string is pingora's only handle on it.
fn client_body_cut_short(e: &pingora_core::Error) -> bool {
    if e.esource() != &pingora_core::ErrorSource::Downstream {
        return false;
    }
    let mut at = Some(e);
    while let Some(x) = at {
        if x.context.as_ref().is_some_and(|c| {
            let c = c.as_str();
            c.starts_with("Peer prematurely closed connection with")
                || c.starts_with("Connection prematurely closed without the termination chunk")
        }) {
            return true;
        }
        at = x
            .cause
            .as_deref()
            .and_then(|c| c.downcast_ref::<Box<pingora_core::Error>>())
            .map(|b| &**b);
    }
    false
}

/// Whether writing the request body to an HTTP/2 stream failed because the stream had already
/// closed with body bytes still unsent: pingora's `reserve_and_send` found no capacity to get
/// (`cannot reserve capacity`, `while waiting for capacity`), a write failure it labels `H2Error`
/// rather than `WriteError`. END_STREAM never went out, so the upstream cannot have the whole
/// request. The usual cause is a GOAWAY that closed a stream mid-upload (D248); pingora gives no
/// other handle on it than the context string.
fn h2_body_unsent(e: &pingora_core::Error) -> bool {
    let mut at = Some(e);
    while let Some(x) = at {
        if x.etype() == &pingora_core::ErrorType::H2Error
            && x.context.as_ref().is_some_and(|c| {
                matches!(
                    c.as_str(),
                    "cannot reserve capacity" | "while waiting for capacity"
                )
            })
        {
            return true;
        }
        at = x
            .cause
            .as_deref()
            .and_then(|c| c.downcast_ref::<Box<pingora_core::Error>>())
            .map(|b| &**b);
    }
    false
}

/// Whether `e` ended a request because someone stopped waiting for its response, rather than the
/// provider ending it: the client went away, or the gateway's `read_timeout_secs` expired on an
/// upstream connection that was still up (D130). Once the provider has the whole request
/// ([`body_delivered`]), it is working on, and billing, the prompt in both cases.
///
/// Not a reset or a close from the upstream before any head: that is the peer declining to answer,
/// and real provider edges answer a request they forwarded and lost with an HTTP error (Cloudflare's
/// 52x, Envoy's 503 local reply), so a bare close is billed nothing; an estimate errs low. Not a
/// dead peer either (`ETIMEDOUT` from TCP keepalive or `TCP_USER_TIMEOUT`, a `ReadError`): nobody
/// can say it ever read the request.
///
/// The request deadline passing ([`is_deadline`]) is the gateway giving up too, as a read timeout
/// is: a provider still working on a delivered request bills its prompt.
///
/// Not the gateway's own refusal of a body it withheld (a downstream-tagged status from
/// `request_body_filter`: a catalog miss, a duplicate `model`, `background`, a foreign id, the
/// body budget). The client read in full, but the provider never had the body's last byte, so
/// nothing was waited on and nothing is billed (D267).
fn gave_up_waiting(e: &pingora_core::Error) -> bool {
    match e.esource() {
        pingora_core::ErrorSource::Downstream => !gateway_refusal(e),
        pingora_core::ErrorSource::Upstream => e.etype() == &pingora_core::ErrorType::ReadTimedout,
        _ => is_deadline(e),
    }
}

/// Whether `e` is the gateway's own answer to the client (a `CustomCode` or `HTTPStatus` tagged
/// downstream: a catalog miss, a duplicate `model`, `background`, a foreign id, an oversized body),
/// not the client going away. Before any response head, such a request never reached a provider
/// whole, so it bills nothing and writes no `ai.usage` row (rejections write none, D267).
fn gateway_refusal(e: &pingora_core::Error) -> bool {
    use pingora_core::ErrorType::{CustomCode, HTTPStatus};
    e.esource() == &pingora_core::ErrorSource::Downstream
        && matches!(e.etype(), CustomCode(..) | HTTPStatus(_))
}

/// Whether the upstream refused this request's HTTP/2 stream with a GOAWAY, one shape of
/// [`upstream_refused_stream`]: the connection is retired, so a resend takes another (D248).
fn upstream_goaway(e: &pingora_core::Error) -> bool {
    e.root_cause()
        .downcast_ref::<h2::Error>()
        .is_some_and(|h| h.is_remote() && h.is_go_away())
}

/// Whether the upstream refused this request's HTTP/2 stream before processing any of it, so it
/// is safe to resend whatever was written (D72). Two shapes, both guaranteed by RFC 9113:
///
/// - `RST_STREAM(REFUSED_STREAM)` from the peer (§8.7: "closed prior to any processing").
/// - A remote GOAWAY on the stream. The `h2` client hands a stream that error only when its id is
///   above the GOAWAY's `last_stream_id` (`Streams::recv_go_away`), which is §6.8's "not
///   processed": a stream at or below it that dies later fails with an I/O error instead. Any
///   reason code — a GOAWAY(ENHANCE_YOUR_CALM) refuses as surely as a graceful NO_ERROR one.
fn upstream_refused_stream(e: &pingora_core::Error) -> bool {
    e.root_cause().downcast_ref::<h2::Error>().is_some_and(|h| {
        h.is_remote()
            && (h.is_go_away() || (h.is_reset() && h.reason() == Some(h2::Reason::REFUSED_STREAM)))
    })
}

/// Whether a **reused** HTTP/1.1 connection was reset before any response byte (D80): the
/// provider's kernel answered with RST because the socket was closed with our request unread in it,
/// the idle-close race on a pooled connection, so the request reached no server. Pingora marks this
/// `ReusedOnly`, retryable on a reused connection.
///
/// Not a clean end-of-file: a server that read the request and then closed looks exactly like one
/// that closed first, and the first may be generating (D09). Not a timeout either (`ETIMEDOUT` is a
/// liveness verdict on a dead peer that may have taken the request), nor an HTTP/2 I/O error, which
/// carries no such guarantee ([`upstream_refused_stream`] covers HTTP/2's own refusals).
fn reset_before_reading(e: &pingora_core::Error, client_reused: bool) -> bool {
    client_reused
        && matches!(e.retry, pingora_core::RetryType::ReusedOnly)
        && e.etype() == &pingora_core::ErrorType::ReadError
        && e.root_cause()
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| {
                matches!(
                    io.kind(),
                    std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted
                )
            })
}

/// Whether `e` is the client closing after the stream's terminal event was written to it (D120,
/// D122). openai-python closes at `[DONE]`, Codex at `response.completed`; when that close lands
/// before the provider's end of stream, pingora ends the request with a downstream error although
/// the client read the whole answer.
///
/// "Written" is exact, not a guess at timing: pingora runs `response_body_filter` on a chunk and
/// then awaits its write to the client before it polls the client again, so the terminal bytes
/// [`TerminalTracker`] saw either went out, or their write failed and the error is that write's
/// (`WriteError` / `WriteTimedout`). Any other downstream error (the client's FIN or reset, read
/// after the write returned) came after the client had the terminal event.
fn closed_after_terminal(rc: &RequestCtx, e: &pingora_core::Error) -> bool {
    use pingora_core::ErrorType as T;
    rc.terminal.ended()
        && e.esource() == &pingora_core::ErrorSource::Downstream
        && !matches!(e.etype(), T::WriteError | T::WriteTimedout)
}

/// Who a request's error says ended it, as its billing row reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    /// The client went away: any downstream error but the gateway's own answer.
    Client,
    /// The gateway's own answer to the client ([`gateway_refusal`]). Downstream-tagged, so a
    /// cancel once a response head arrived, and no row at all before one.
    Refusal,
    /// The upstream, or anything else not tagged downstream.
    Upstream,
}

impl Fault {
    fn of(e: &pingora_core::Error) -> Self {
        if gateway_refusal(e) {
            Self::Refusal
        } else if e.esource() == &pingora_core::ErrorSource::Downstream {
            Self::Client
        } else {
            Self::Upstream
        }
    }
}

/// Whose word a billing row's token counts are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Count {
    /// The provider's own report (or none at all, on an error body).
    #[default]
    Reported,
    /// The gateway's estimate of what the provider bills, and which counts it replaced (none
    /// when the provider's partial report already covered what the estimate saw).
    Estimated(EstimatedParts),
}

impl Count {
    fn estimated(self) -> bool {
        matches!(self, Self::Estimated(_))
    }

    /// What the estimate replaced: nothing, for a reported count.
    fn parts(self) -> EstimatedParts {
        match self {
            Self::Reported => EstimatedParts::default(),
            Self::Estimated(p) => p,
        }
    }
}

/// The facts `logging` reads about how a request ended, before it decides how the request bills.
#[derive(Clone, Copy, Debug, Default)]
struct EndFacts {
    /// An abandoned `FullBody` attempt ([`Attempt::relay_abandoned`]).
    relay_abandoned: bool,
    /// A free sub-resource (a token count).
    free: bool,
    /// Answered from the exact-match cache.
    cache_hit: bool,
    /// The provider's status; absent when no response head arrived.
    status: Option<u16>,
    /// Whom the request's error blames; absent when it ended without one.
    fault: Option<Fault>,
    /// The client hung up on a drained stream ([`crate::drain`]) before its terminal event.
    client_left: bool,
    /// A provider was handed the request (`UpstreamPhase` past `None`).
    called: bool,
    /// The drain read the stream to its end after the client left.
    drained: bool,
    /// Whose word the counts are.
    count: Count,
}

/// Every way a request can end, as billing sees it. `logging` builds one from its [`EndFacts`],
/// and [`Ending::row`] is a total match on it, so a new ending does not compile until someone
/// decides how it bills.
///
/// Every ending that writes a row carries its [`Count`], and the row logs it as it is, so an
/// estimate is never written as the provider's word. An estimate needs a 2xx head, or a connected
/// attempt that delivered the whole body ([`gave_up_waiting`]), so a cache hit, an error status
/// and a request no provider was handed carry `Reported`: [`Ending::of`] asserts it in debug
/// builds, where a producer that broke it would show. Only a cancel can be drained: a drain arms
/// only on a managed 2xx stream's head ([`drains_on_cancel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ending {
    /// An abandoned `FullBody` attempt: not the answer the client got. The attempt that serves
    /// writes the one row.
    RelayAbandoned,
    /// A free sub-resource (a token count): not a billable call.
    Free,
    /// Answered from the exact-match cache: the usage stored at fill, no provider called.
    CacheHit(Count),
    /// The gateway's own refusal before any response head (D267): no provider had the request
    /// whole, so it writes no row, as every other rejection writes none.
    Refused,
    /// The provider answered with an error status (4xx/5xx).
    UpstreamStatus(Count),
    /// The client left before the answer was whole, or left a drained stream however the request
    /// then ended. `drained`: the gateway then read the stream to its end.
    ClientCancelled { count: Count, drained: bool },
    /// No provider was called (every candidate's breaker open).
    NoCandidate(Count),
    /// The upstream failed before any response head: a connect, read-timeout or reset failure.
    UpstreamFailed(Count),
    /// A response that started and then died upstream.
    CutShort(Count),
    /// The request completed.
    Completed(Count),
}

/// The fields of an `ai.usage` row its [`Ending`] decides, as logged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EndRow {
    /// What became of the request: a consumer must be able to tell a zero-token row for an
    /// upstream error or a cancel from a real zero-token generation.
    outcome: &'static str,
    /// Whose word the counts are (`usage_estimated`, `usage_estimated_parts`).
    count: Count,
    /// The counts are the cache entry's, stored at fill: no call was made.
    cache_hit: bool,
    /// The client cancelled and the gateway read the stream to its end ([`crate::drain`]).
    drained: bool,
}

impl EndRow {
    /// The row's `usage_settled`: `drained` when the counts after a cancel are the vendor's own.
    fn usage_settled(self) -> Option<&'static str> {
        (self.drained && !self.count.estimated()).then_some("drained")
    }
}

impl Ending {
    /// The ending these facts describe. The order is the precedence: a row-less ending over any
    /// other, an error status over a cancel, a cancel over a missing call.
    fn of(f: &EndFacts) -> Self {
        let count = f.count;
        if f.relay_abandoned {
            return Self::RelayAbandoned;
        }
        if f.free {
            return Self::Free;
        }
        // These three never see an estimate (see `Ending`): one here is a producer's bug.
        let reported = |ending: Self| {
            debug_assert!(!count.estimated(), "{ending:?} carries an estimate");
            ending
        };
        if f.cache_hit {
            return reported(Self::CacheHit(count));
        }
        if f.status.is_none() && f.fault == Some(Fault::Refusal) {
            return Self::Refused;
        }
        if f.status.is_some_and(|s| s >= 400) {
            return reported(Self::UpstreamStatus(count));
        }
        if f.client_left || matches!(f.fault, Some(Fault::Client | Fault::Refusal)) {
            return Self::ClientCancelled {
                count,
                drained: f.drained,
            };
        }
        if !f.called {
            return reported(Self::NoCandidate(count));
        }
        match (f.fault, f.status) {
            (Some(_), None) => Self::UpstreamFailed(count),
            (Some(_), Some(_)) => Self::CutShort(count),
            (None, _) => Self::Completed(count),
        }
    }

    /// How this ending bills; `None` writes no row.
    fn row(self) -> Option<EndRow> {
        let row = |outcome, count| EndRow {
            outcome,
            count,
            cache_hit: false,
            drained: false,
        };
        Some(match self {
            Self::RelayAbandoned | Self::Free | Self::Refused => return None,
            Self::CacheHit(count) => EndRow {
                cache_hit: true,
                ..row("ok", count)
            },
            Self::UpstreamStatus(count) => row("upstream_error", count),
            Self::ClientCancelled { count, drained } => EndRow {
                drained,
                ..row("client_cancelled", count)
            },
            Self::NoCandidate(count) => row("no_candidate", count),
            Self::UpstreamFailed(count) => row("upstream_error", count),
            Self::CutShort(count) => row("cut_short", count),
            Self::Completed(count) => row("ok", count),
        })
    }
}

/// State that exists only when a request uses the `x-beyond-*` control surface — it carried
/// metadata, or it is being captured (by tenant rule or by header).
///
/// Boxed on `RequestCtx` for the same measured reason [`ModelRouting`] is: both members are dead
/// weight on the ordinary request, and `RequestCtx` is on the per-chunk path.
struct RequestControl {
    /// Canonical metadata JSON from `x-beyond-metadata`, already validated and re-serialized by
    /// [`crate::control`]. Emitted on `ai.usage` (and `ai.payload`) verbatim.
    metadata: Option<String>,
    /// Payload buffers — `Some` iff this request is being captured. Independent of `metadata`:
    /// tagging is useful with capture off, which is the point of shipping them together.
    capture: Option<CaptureBufs>,
}

/// State that exists only for a **model-routed** request (`/auto`, or a managed bare `/v1` catalog
/// walk).
///
/// Boxed, and `None` for provider-routed traffic — which is the overwhelming majority. Held inline
/// these fields added 64 bytes to `RequestCtx` (368 → 432), a struct that is touched on every hook
/// and, for a streaming response, once per response chunk. That showed up as a reproducible ~2.5%
/// regression on the `managed_sse_latency` bench, across two independent runs against the same
/// baseline, while the non-streaming case was unaffected — the signature of a per-chunk cost, not a
/// per-request one.
///
/// So the model-routed path pays one small allocation and every other request pays nothing. That is
/// the right way round: catalog walking is opt-in (or the managed `/v1` drop-in), and the request it
/// serves is about to cross a network.
struct ModelRouting {
    /// The catalog row this request routes over. `&'static`, so it costs a pointer.
    route: &'static route::ModelRoute,
    /// A provider endpoint under the parent one (`count_tokens`, `compact`): its suffix is appended
    /// to the serving candidate's path. It walks the row in catalog order (no session pin), and a
    /// free one writes no billing row.
    sub: Option<route::SubResource>,
    /// Walk slot of the candidate currently being attempted. Maps through [`Self::walk`] onto
    /// [`Self::arms`].
    candidate: u8,
    /// Bit `i` ⇒ walk slot `i` is *usable*: this gateway routes to that provider and holds a
    /// pool key for it. Computed once in `request_filter` so `upstream_peer` never re-derives it.
    /// Bounded by [`route::MAX_CANDIDATES`], which is why a `u8` suffices.
    usable: u8,
    /// Catalog indices in walk order. Permuted by `x-beyond-order` / `only` / `split` and, when
    /// those do not fix it, by the caller's computed session pin ([`crate::pin`]).
    /// `first_usable` walks this sequence; failover, breakers, and the 429 key-walk see the same.
    walk: control::Walk,
    /// The candidate slice this walk indexes — [`ModelRoute::candidates`] or
    /// [`ModelRoute::responses`].
    arms: &'static [route::Candidate],
    /// Named session field (`previous_response_id` / `store`) that must not be stripped onto a
    /// non-Responses candidate. `None` is a one-shot (or not a Responses request).
    session_field: Option<&'static str>,
    /// When the current attempt began. Distinct from `RequestCtx::start` (which times the whole
    /// request) so a candidate that burned `connect_timeout_secs` before failing over does not
    /// charge that time to the provider that actually served — which would render an outage at
    /// candidate A as a latency regression at candidate B, inverting the point of the per-provider
    /// label.
    attempt_start: Instant,
    /// Exact-match cache: fill a miss, or a hit already written to the client.
    cache: Option<cache::Pending>,
    /// Inbound endpoint. Always set on a catalog walk so a mixed-row failover can translate
    /// onto the next candidate's path. Same-endpoint attempts skip the mapper (`from == to`).
    translate: Option<translate::TranslateState>,
    /// A 2xx whose health verdict waits on the body's first bytes (see `settle_health`). `None`
    /// once settled, and for every non-2xx.
    health: Option<PendingHealth>,
    /// How many addresses the current candidate resolved to; `RequestCtx::attempt` indexes them.
    addrs: u8,
    /// The walk made at least one upstream attempt (it connected, or tried to). Distinguishes
    /// "every candidate failed" (502) from "every candidate's breaker was open" (503) when the walk
    /// runs out.
    attempted: bool,
    /// Seconds until the soonest skipped open breaker admits a request, for the `Retry-After` on
    /// the 503 when every candidate was skipped. `None` while no breaker skipped one.
    open_retry_after: Option<u16>,
    /// The client asked for priority processing (`route::SpeedAsk::Priority`): an attempt
    /// translated onto a candidate that serves fast mode asks it for fast mode (D266).
    priority: bool,
}

/// A 2xx's held-back health verdict, waiting on the body to say whether it is an answer.
struct PendingHealth {
    /// The body's first bytes, up to [`HEALTH_PREFIX_CAP`].
    prefix: Vec<u8>,
    /// The breaker permit the head resolved as a success. The body's verdict is recorded against it,
    /// in its own generation, so a later failure cannot reopen a probe that another request owns by
    /// then.
    permit: Option<Answered>,
}

impl PendingHealth {
    fn extend_prefix(&mut self, bytes: &[u8]) {
        self.prefix.extend_from_slice(bytes);
    }
}

/// How much of a 2xx body `settle_health` reads before calling it an answer. An error-in-200 says
/// so in its first key (`{"error":` / `{"type":"error"`) or its first SSE event.
const HEALTH_PREFIX_CAP: usize = 1024;

/// A managed status that means the pool key itself was refused (revoked, wrong): a 401. It cools
/// the key off and walks to the next one; on a catalog walk's last key it is a candidate failure.
/// A 403 is not: it is usually about the request (moderation, a model the key's project may not
/// use, Anthropic's `permission_error`), so one tenant's 403 must not move the shared pool off a
/// key (D84). A 403 whose body names the key is cooled from the body ([`body_names_the_key`]).
pub(crate) fn is_pool_key_failure(status: u16) -> bool {
    status == 401
}

/// A managed catalog walk's status that another candidate (holding a different key, at a different
/// vendor) may well not return: refused (401), unfunded (402), or forbidden (403). A candidate
/// failure for this request like a 5xx, but nothing about the provider's health: never a breaker
/// failure. Only a 401 ([`is_pool_key_failure`]) walks keys, and cools its key at the head. A
/// relayed answer can cool its key from `logging` too, read from the body: a 403 that names the
/// key ([`body_names_the_key`]), and an out-of-credit answer under any status (a 402, Anthropic's
/// credit-balance 400, OpenAI's `insufficient_quota` 429: `remedy::Neutralized::unfunded`). A 402
/// the walk fails over on is abandoned at its head, body unread, so its key is cooled there when
/// the status alone says out of credit (`remedy::unfunded_402`, D258). Each is counted on
/// `ai_key_auth_failures_total` by reason (`metrics::KeyCooled`).
pub(crate) fn is_candidate_refusal(status: u16) -> bool {
    (401..=403).contains(&status)
}

/// Whether an error body says the credential itself is bad: OpenAI's `invalid_api_key` code, or
/// Anthropic's `authentication_error` type. Read from a relayed managed 403, which then cools the
/// key for later requests (this one has already answered).
fn body_names_the_key(body: &[u8]) -> bool {
    memchr::memmem::find(body, b"\"invalid_api_key\"").is_some()
        || memchr::memmem::find(body, b"\"authentication_error\"").is_some()
}

/// Whether a 2xx body's first bytes are an error: `Some(true)` an error object, `Some(false)` an
/// answer, `None` not decidable yet. Non-streaming: the root object's first key is `error`, or a
/// first key `type` whose value is `"error"` (Anthropic's shape). Streaming: an `event: error`
/// line, or a first `data:` event whose payload is such an object.
fn body_reports_error(prefix: &[u8], streaming: bool) -> Option<bool> {
    if !streaming {
        return json_is_error_object(prefix);
    }
    let mut rest = prefix;
    loop {
        let (line, tail, complete) = match memchr::memchr(b'\n', rest) {
            Some(i) => (&rest[..i], &rest[i + 1..], true),
            None => (rest, &rest[rest.len()..], false),
        };
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some(event) = line.strip_prefix(b"event:") {
            if !complete {
                return None;
            }
            if event.trim_ascii() == b"error" {
                return Some(true);
            }
        } else if let Some(data) = line.strip_prefix(b"data:") {
            let data = data.trim_ascii_start();
            if data.starts_with(b"[DONE]") {
                return Some(false);
            }
            return match json_is_error_object(data) {
                None if complete => Some(false),
                v => v,
            };
        } else if !complete {
            return None;
        }
        // A blank line, a comment (`: keepalive`), or an `id:`/`retry:` field: keep looking.
        rest = tail;
    }
}

/// [`body_reports_error`] for one JSON value's leading bytes.
fn json_is_error_object(bytes: &[u8]) -> Option<bool> {
    let bytes = bytes.trim_ascii_start();
    let Some(rest) = bytes.strip_prefix(b"{") else {
        return if bytes.is_empty() { None } else { Some(false) };
    };
    let (key, rest) = json_leading_string(rest.trim_ascii_start())?;
    match key {
        b"error" => Some(true),
        b"type" => {
            let rest = rest.trim_ascii_start().strip_prefix(b":")?;
            let (value, _) = json_leading_string(rest.trim_ascii_start())?;
            Some(value == b"error")
        }
        _ => Some(false),
    }
}

/// Whether a whole non-stream body is only an error (D195, D205): a root object with a non-null
/// `error` member and none of the members an answer is carried in (`choices`, `output`,
/// `content`), whatever order its keys come in. OpenRouter's error-in-200 is `{"error": {...}}`,
/// but a provider may write `id`, `object` or `model` first; a Responses answer carries
/// `"error": null` beside its `output`.
fn json_is_error_only(body: &[u8]) -> bool {
    peek::root_members(body).is_some_and(|members| {
        let mut error = false;
        for m in &members {
            if m.key_is(body, "error") {
                error |= &body[m.value.0..m.value.1] != b"null";
            } else if ["choices", "output", "content"]
                .iter()
                .any(|k| m.key_is(body, k))
            {
                return false;
            }
        }
        error
    })
}

/// A leading JSON string's raw bytes and what follows it. `None` when it has not ended yet; an
/// escaped string reads as itself, which matches neither `error` nor `type` (the right answer for
/// a key spelled oddly enough to need one).
fn json_leading_string(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let body = bytes.strip_prefix(b"\"")?;
    let end = memchr::memchr(b'"', body)?;
    Some((&body[..end], &body[end + 1..]))
}

impl ModelRouting {
    /// The catalog candidate at walk slot `i`.
    fn candidate_at(&self, i: u8) -> Option<&'static route::Candidate> {
        self.walk
            .catalog_index(i)
            .and_then(|orig| self.arms.get(usize::from(orig)))
    }
}

/// An HTTP status as [`RequestCtx::upstream_status`] holds it. Every status reaching it was parsed
/// by `http::StatusCode` (100-999) or stored from one, so the `0` that `NonZeroU16` cannot hold
/// never arrives; were one to, it would read as "no response head", hence the assertion.
fn held_status(status: u16) -> Option<NonZeroU16> {
    debug_assert!(status != 0, "an HTTP status is never 0");
    NonZeroU16::new(status)
}

impl RequestCtx {
    /// The upstream HTTP status, once the response head has arrived.
    fn upstream_status(&self) -> Option<u16> {
        self.upstream_status.map(NonZeroU16::get)
    }

    /// Whether this is a **managed** request (`bai_v1…` / `bai_v2…` key → swap to the pool key).
    /// `false` for **BYO** — we leave the user's own auth header untouched (passthrough).
    fn managed(&self) -> bool {
        matches!(self.billing, Billing::Managed(_))
    }

    /// Take one request-body chunk into this attempt's record: the size count, the capture tap and,
    /// for a rewrite, the buffer the rewrite reads. `false` once the body has crossed
    /// `MAX_REQUEST_BODY`. The body filter and the replay tests both feed through here, so a replayed
    /// prefix is accounted the same way each time.
    fn take_request_chunk(&mut self, chunk: &[u8]) -> bool {
        if !self.attempt.upload.feed(chunk.len()) {
            return false;
        }
        // Capture tap. Deliberately on the chunk as it *arrives*, which makes the captured bytes
        // **pre-rewrite** for free: both rewrites happen at end-of-stream over the whole of `req_buf`,
        // long after this ran. A captured body showing a `stream_options` the client's SDK never sent,
        // or a `model` re-spelled for whichever failover candidate served, sends an engineer hunting
        // through their own code for something the gateway put there. What we capture is what the
        // client sent. Bounded and non-withholding: the chunk is forwarded either way.
        if let Some(c) = self.control.as_mut().and_then(|c| c.capture.as_mut()) {
            c.push_req(chunk);
        }
        if self.plan.rewrites_body() {
            self.attempt.req_buf.extend_from_slice(chunk);
        }
        true
    }

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

    /// Whether the bytes sent to the client are watched for the stream's terminal event
    /// (`closed_after_terminal`): a managed stream's, unless it is assembled into one JSON body.
    /// The verdict is read only on a managed request's billing row, and a JSON body has no
    /// terminal event, so watching anything else would scan every chunk for nothing.
    fn tracks_terminal(&self) -> bool {
        self.managed()
            && self.streaming
            && !self.auto.as_ref().is_some_and(|a| catalog_assembling(a))
    }

    /// Anthropic SSE only: keep a bounded head of the upstream response so `message_start`'s input
    /// and cache token counts survive the tail's compaction. At most `USAGE_HEAD_CAP` bytes in all,
    /// satisfied within the first chunk or two, after which this copies nothing: one small
    /// allocation on the one path that needs it, and nothing anywhere else. See `USAGE_HEAD_CAP`
    /// for why only this dialect needs it.
    fn keep_usage_head(&mut self, chunk: &[u8]) {
        if self.streaming && self.dialect == Dialect::Anthropic {
            let want = USAGE_HEAD_CAP.saturating_sub(self.resp_head.len());
            self.resp_head
                .extend_from_slice(&chunk[..want.min(chunk.len())]);
        }
    }

    /// Point the forwarded path at the candidate about to be attempted.
    ///
    /// Taken wholesale from the catalog rather than composed from a mount and the client's suffix:
    /// providers disagree on where an endpoint lives and the disagreement is not a prefix, so there
    /// is no client suffix that is correct for every candidate in a row. Rewritten in place, so a
    /// failover costs no allocation after the first attempt.
    fn set_forward_path(&mut self, path: &str, sub: Option<route::SubResource>) {
        let suffix = sub.map_or("", route::SubResource::suffix);
        match &mut self.forward_path {
            ForwardPath::Catalog(buf) => {
                buf.clear();
                buf.push_str(path);
                buf.push_str(suffix);
            }
            other => *other = ForwardPath::Catalog([path, suffix].concat()),
        }
    }

    /// Clear the state one attempt accumulated, so a **retried** attempt starts from the same slate
    /// as the first: a fresh [`Attempt`]. Called from `upstream_peer`, which pingora invokes exactly
    /// once per attempt and always before any body byte moves — it is the first statement of
    /// `proxy_to_upstream`.
    ///
    /// The per-attempt state boxed elsewhere, to keep this struct small, starts over with it: the
    /// captured request (the tap is fed the same replayed chunks), the price-knob scan, and a
    /// translated walk's response state. A 5xx vendor walk aborts in `upstream_response_filter`
    /// before `response_filter` runs, so that last is empty in practice; it is still cleared, so a
    /// retry cannot append a leftover JSON/SSE buffer onto the candidate that actually serves.
    ///
    /// `model` is deliberately request-level: once a complete body has yielded it the value is
    /// correct, and pingora's replay buffer is capped at 64 KiB, so a larger body could not
    /// re-derive it on the next attempt.
    fn begin_attempt(&mut self) {
        if let Some(t) = self.auto.as_mut().and_then(|a| a.translate.as_mut()) {
            t.sse = None;
            t.json_buf.clear();
            t.assemble = false;
        }
        // An attempt that fed no body chunk is still as `Attempt::new` made it: every body-phase
        // accumulator is fed on the branch that counts the chunk. The first attempt, the only one
        // most requests make, keeps the one `request_filter` built.
        if !self.attempt.upload.started() {
            return;
        }
        if let Some(c) = self.control.as_mut().and_then(|c| c.capture.as_mut()) {
            c.clear_req();
        }
        if let Some(k) = self.billing.taps_mut().and_then(|t| t.knobs.as_mut()) {
            *k = (peek::ModelScanner::new(), peek::Kept::request_knobs());
        }
        // Nothing carried over but the body buffer's allocation, which `request_filter` pre-sized
        // from `Content-Length` and the in-place splice relies on to stay realloc-free.
        let mut req_buf = std::mem::take(&mut self.attempt.req_buf);
        req_buf.clear();
        self.attempt = Attempt::new(req_buf);
    }

    /// Move the walk past candidate `at`, and declare the next attempt its next candidate.
    fn fail_over(&mut self, at: u8) {
        self.advance_candidate(at);
        self.leg.retry = Some(Retry::NextCandidate);
    }

    /// Where the catalog walk can go from the candidate this attempt is on, for
    /// [`retry::decide`]; `None` for a provider-routed request.
    fn walk_facts(&self) -> Option<retry::Walk> {
        let a = self.auto.as_ref()?;
        Some(retry::Walk {
            orig: a.walk.catalog_index(a.candidate),
            next: first_usable(a.usable, a.candidate.saturating_add(1)).is_some(),
            next_addr: self.leg.addr.saturating_add(1) < a.addrs,
        })
    }
}

/// Where a failed attempt's body can be resent from, for [`retry::decide`]. Reads the `FullBody`
/// parent in place: a clone of it is taken only to record a re-run.
fn body_facts(session: &mut Session, walk: Option<retry::Walk>) -> retry::Body {
    let parent = full_body(session).map(|fb| retry::Parent {
        reset_spent: walk
            .and_then(|w| w.orig)
            .is_some_and(|orig| fb.reset & (1 << orig) != 0),
    });
    retry::Body {
        replayable: body_replayable(session),
        truncated: session.as_ref().retry_buffer_truncated(),
        parent,
    }
}

/// Every answer `fail_to_proxy` gives an error that is not a [`Refusal`] (a connect failure, an
/// upstream that failed after it had the whole request, every candidate's breaker open), with its
/// body spelled out at compile time: `(type, message)` as [`failure_response`] and `fail_to_proxy`
/// pair them, so a burst of provider failures is answered without building JSON.
static FAILURES: [refusal::Fixed; 17] = [
    refusal::fixed!("invalid_request_error", "bad request"),
    refusal::fixed!("invalid_request_error", "request body too large"),
    refusal::fixed!("rate_limit_error", "rate limit exceeded"),
    refusal::fixed!("api_error", "upstream unavailable"),
    refusal::fixed!("api_error", "provider temporarily unavailable"),
    refusal::fixed!("api_error", "upstream timed out"),
    refusal::fixed!("invalid_request_error", "request rejected"),
    refusal::fixed!("api_error", "upstream error"),
    refusal::fixed!("invalid_request_error", "request body timed out"),
    refusal::fixed!("api_error", "could not connect to the provider"),
    refusal::fixed!("api_error", "upstream connection failed"),
    refusal::fixed!("api_error", "internal error"),
    refusal::fixed!("api_error", "no provider could be reached"),
    refusal::fixed!("api_error", "no candidate provider available"),
    refusal::fixed!(
        "api_error",
        "the provider refused the stream; it was not processed"
    ),
    refusal::fixed!(
        "api_error",
        "upstream timed out after receiving the request"
    ),
    refusal::fixed!("api_error", "upstream failed after receiving the request"),
];

/// The wire body for a failure's `(typ, msg)`: the static one from [`FAILURES`], else built with
/// `serde_json` (unreachable for any pair the gateway writes: a test walks every one).
/// `pub` for the bench target (`benches/unit.rs`). Not part of the crate's intended surface.
pub fn error_body(typ: &str, msg: &str) -> Bytes {
    match FAILURES.iter().find(|f| f.kind == typ && f.message == msg) {
        Some(f) => Bytes::from_static(f.body.as_bytes()),
        None => refusal::json_body(typ, msg),
    }
}

impl AiProxy {
    /// The bound [`peek_body_model`] puts on each body read itself: `client_read_timeout_secs`,
    /// where the session has no read timeout of its own (HTTP/2; HTTP/1.1 enforces it inside
    /// pingora, so a second timer would be waste).
    fn up_front_read_timeout(&self, session: &Session) -> Option<Duration> {
        let secs = self.state.config.client_read_timeout_secs();
        (secs > 0 && session.as_ref().get_read_timeout().is_none())
            .then(|| Duration::from_secs(secs))
    }

    /// End a request at its deadline (`request_max_secs`): counted, logged, and answered 504 if no
    /// response has started (after one has, pingora cuts the stream as any mid-stream error does).
    fn deadline_cut(&self, request_id: &str) -> Box<pingora_core::Error> {
        let cut = self.refusal_error(Refusal::RequestDeadline);
        warn!(
            request_id,
            limit_secs = self.state.config.request_max_secs(),
            "request reached request_max_secs; ending it",
        );
        cut
    }

    /// Count `refusal` and raise it as pingora's error, for a hook past `request_filter` that
    /// cannot write the answer itself: `fail_to_proxy` answers it, if no response has started.
    fn refusal_error(&self, refusal: Refusal) -> Box<pingora_core::Error> {
        self.state.metrics.rejection(refusal.metric()).inc();
        refusal.into_error()
    }

    /// Cap this attempt's upstream read timeout at the time left before the request's deadline,
    /// when that is the shorter bound, and say so in `read_capped`: pingora re-arms the timeout on
    /// every read and every read starts after this attempt did, so a capped timeout can only fire
    /// at or past the deadline, and `error_while_proxy` turns it into [`Self::deadline_cut`]. This
    /// is what ends a silent stream (an HTTP/2 PING keeps the connection alive, and a `-pro` row
    /// has no silence bound at all); a moving one is cut per chunk. No time left: the request ends
    /// here, before a connection is made that the provider would bill.
    fn cap_read_at_deadline(
        &self,
        rc: &mut RequestCtx,
        mut peer: HttpPeer,
    ) -> Result<Box<HttpPeer>> {
        rc.attempt.read_capped = false;
        if let Some(left) = crate::deadline::remaining(rc.deadline) {
            if left.is_zero() {
                return Err(self.deadline_cut(&rc.request_id));
            }
            if peer.options.read_timeout.is_none_or(|t| left < t) {
                peer.options.read_timeout = Some(left);
                rc.attempt.read_capped = true;
            }
        }
        Ok(Box::new(peer))
    }

    /// Build the upstream peer for a resolved address + provider.
    ///
    /// Extracted so the provider-routed path and the model-routed candidate walk cannot drift apart
    /// on TLS, ALPN, or timeouts — a fallback candidate connected on different terms than the
    /// primary would be a genuinely nasty thing to debug.
    fn build_peer(
        &self,
        addr: std::net::SocketAddr,
        provider: &Provider,
        row: Option<&route::ModelRoute>,
    ) -> HttpPeer {
        let mut peer = HttpPeer::new(
            addr,
            self.state.config.upstream_tls().tls(),
            provider.sni().to_string(),
        );
        // Prefer HTTP/2 to the provider (config `upstream_http2`, default on), fall back to
        // HTTP/1.1. Every provider in `known_providers()` negotiates `h2` over TLS (verified by
        // handshake), and H2 multiplexes many concurrent requests/streams over one connection —
        // fewer sockets and TLS handshakes from our egress IPs (which also eases the
        // egress-reputation pressure `ratelimit` guards). `H2H1` is strictly ≥ `H1` on
        // compatibility: ALPN negotiates down to H1 for any host that doesn't offer h2, and a
        // plaintext upstream (the mock, `upstream_tls=false`) has no ALPN at all and stays H1. The
        // negotiated protocol is then visible per-request as `upstream_request.version` (see
        // `upstream_request_filter`), which is what lets the body-injection path frame correctly.
        // The knob lets an operator force all-H1 without a code redeploy, and lets the e2e bench
        // compare the two head-to-head.
        peer.options.alpn = if self.state.config.upstream_http2() {
            ALPN::H2H1
        } else {
            ALPN::H1
        };
        // Pingora allows one stream per upstream H2 connection unless told otherwise, which made
        // every concurrent request open its own TLS connection: H2 without the multiplexing (D160).
        // The provider's own SETTINGS_MAX_CONCURRENT_STREAMS still caps this (pingora takes the
        // lower), and a full connection makes the next request open another.
        peer.options.max_h2_streams = UPSTREAM_H2_MAX_STREAMS;
        // Cert verification is on everywhere except the bench's self-signed TLS mock (see config).
        if !self.state.config.upstream_tls().verifies() {
            peer.options.verify_cert = false;
            peer.options.verify_hostname = false;
        }
        peer.options.connection_timeout = Some(Duration::from_secs(
            self.state.config.connect_timeout_secs(),
        ));
        // Silence is not a failure signal: a model thinking without emitting is indistinguishable
        // from a stuck provider, so the per-read bound (the wait for the head, and each gap between
        // body reads) is the clients' own: `read_timeout_secs`, 600s, the OpenAI and Anthropic
        // SDKs' default request timeout. A *dead* peer is detected by transport liveness instead.
        // Pingora applies it to every read, so before the head it is the whole wait. A row whose
        // vendor documents requests running "several minutes" and publishes no maximum
        // (`long_running`: OpenAI's `-pro` models) gets no silence deadline at all (D252): a
        // non-stream call there can outlast 600s, and a 504 is still billed by the provider. Its
        // end is the client's own timeout (a cancel the gateway sees and bills, D130) or a dead
        // transport.
        peer.options.read_timeout = (!row.is_some_and(providers::catalog::long_running))
            .then(|| Duration::from_secs(self.state.config.read_timeout_secs()));
        peer.options.h2_ping_interval = self.state.config.h2_ping_interval();
        peer.options.tcp_keepalive = self.state.config.upstream_tcp_keepalive();
        peer.options.write_timeout =
            Some(Duration::from_secs(self.state.config.write_timeout_secs()));
        peer.options.idle_timeout =
            Some(Duration::from_secs(self.state.config.idle_timeout_secs()));
        peer
    }

    /// Count `refusal`, log it, and answer it before any upstream, signalling `request_filter`
    /// to short-circuit.
    ///
    /// Every refusal logs one structured `warn` line (the rejection counter only says *how many*,
    /// not *which request*: this is what an oncall greps when a `deny_fraud`/`rate_limit` spike
    /// shows on the dashboard) and echoes the `request_id` in a response header so a client report
    /// quoting that id lands on this line.
    ///
    /// Not an `async fn`: the refusal moves straight into a boxed write, so the await in
    /// `request_filter` holds one pointer. `#[async_trait]` heap-boxes `request_filter`'s future
    /// once per request, and the box must be big enough for its widest await: with the write
    /// awaited inline at eight call sites it dominated that future (1 264 bytes total, 1 136 of them
    /// the write, `-Zprint-type-sizes`), so a request that is never refused paid for room it never
    /// used.
    ///
    /// For a refusal that may not happen, call this inside the `if let` scrutinee
    /// (`check().map(|r| self.refuse(..))`) and await the write in the block: an `Option<Refusal>`
    /// scrutinee is a temporary the `if let` keeps through its block, and moving the refusal out of
    /// it does not end that, so it would be saved across the await (56 bytes in every request's box).
    fn refuse<'a>(
        &self,
        session: &'a mut Session,
        request_id: &'a str,
        refusal: Refusal,
    ) -> Pin<Box<impl Future<Output = Result<bool>> + Send + 'a>> {
        self.state.metrics.rejection(refusal.metric()).inc();
        Box::pin(write_refusal(session, request_id, refusal))
    }

    /// This tenant's concurrency slot. `Ok(None)` when no cap is configured; `Err` when the tenant
    /// is at its cap.
    fn take_slot(&self, id: &Identity) -> std::result::Result<Option<TenantSlot<'static>>, ()> {
        match self.state.tenant_slots.as_ref() {
            None => Ok(None),
            Some(slots) => slots.take(id).map(Some).ok_or(()),
        }
    }

    /// `body` is [`GatewayState::models_list`], rendered at boot; serving it is a refcount bump.
    async fn reply_models_list(
        session: &mut Session,
        request_id: &str,
        body: Bytes,
    ) -> Result<bool> {
        let head_only = session.req_header().method == http::Method::HEAD;
        let mut len_buf = ArrayString::<20>::new();
        let _ = write!(len_buf, "{}", body.len());
        let mut resp = ResponseHeader::build(200, Some(3))?;
        resp.insert_header("content-type", "application/json")?;
        resp.insert_header("content-length", len_buf.as_str())?;
        resp.insert_header(REQUEST_ID_HEADER, request_id)?;
        session
            .write_response_header(Box::new(resp), head_only)
            .await?;
        if !head_only {
            session.write_response_body(Some(body), true).await?;
        }
        Ok(true)
    }

    async fn reply_models_list_boxed(
        session: &mut Session,
        request_id: &str,
        body: Bytes,
    ) -> Result<bool> {
        Box::pin(Self::reply_models_list(session, request_id, body)).await
    }

    /// `signed_id`: a large catalog body's ids, checked before its first attempt connects (its
    /// `FullBody` re-runs see the body only as it streams in). `Some` when the request is refused.
    fn foreign_ids(
        &self,
        path: &str,
        route: &'static route::ModelRoute,
        body: &[u8],
        tenant_id: u64,
    ) -> Option<Refusal> {
        if !signed_id::catalog_applies(path, route) {
            return None;
        }
        match self.state.id_signer.as_ref() {
            None => Some(Refusal::IdSigningUnset),
            Some(signer) => signer
                .unsign_request(tenant_id, body, true)
                .err()
                .map(Refusal::ForeignId),
        }
    }

    /// Re-run a request whose body the gateway read in full as a pingora subrequest carrying that
    /// body, and pipe its response back; re-run again on the next candidate or pool key when an
    /// attempt asks for it (see [`FullBody`]). Each attempt is a complete request of its own (auth,
    /// deny and allowance checks, the walk, translation), under this request's id; only the attempt
    /// that serves writes `ai.usage`.
    async fn relay_full_body(
        &self,
        session: &mut Session,
        parent: Parent,
        route: &'static route::ModelRoute,
        body: Vec<u8>,
        slot_held: bool,
    ) -> Result<bool> {
        let Parent {
            request_id,
            request_seq,
            deadline,
        } = parent;
        let session_field = if route::is_responses_path(session.req_header().uri.path()) {
            translate::responses_session_field(&body, !route.responses.is_empty())
        } else {
            None
        };
        let facts = body_rules::Facts {
            refused_input: route::refused_input(route, &body).map(|kind| (route.model, kind)),
            background: route::is_responses_path(session.req_header().uri.path())
                && requests_background(&body),
            ..body_rules::Facts::default()
        };
        if let Some(write) = body_rules::refusal(body_rules::Point::Relay, facts)
            .map(|refusal| self.refuse(session, &request_id, refusal))
        {
            return write.await;
        }
        let unserved = route::unserved(route.candidates, &body);
        let responses_tools = route::tools_need_responses_arm(
            route,
            route::implied_endpoint(session.req_header().uri.path()),
            &body,
        );
        let body = Bytes::from(body);
        self.state.metrics.full_body_relays_total.inc();
        let mut skip = 0u8;
        let mut keys = [NO_KEY_WALK; route::MAX_CANDIDATES];
        let mut reset = 0u8;
        let mut resume = None;
        // Every attempt removes a candidate, advances a key, or spends a candidate's one reset
        // retry, so the walk ends on its own; this bound only guards against a bug looping it. The
        // last attempt it allows records no retry, so even then the client gets that attempt's own
        // answer rather than a synthetic one.
        const MAX_ATTEMPTS: usize = route::MAX_CANDIDATES * 18;
        for n in 0..MAX_ATTEMPTS {
            let retry = Arc::new(std::sync::Mutex::new(None));
            let ctx = SubrequestCtx::builder()
                .body_mode(BodyMode::ExpectBody)
                .user_ctx(Box::new(FullBody {
                    route,
                    session_field,
                    skip,
                    unserved,
                    responses_tools,
                    keys,
                    reset,
                    resume,
                    final_attempt: n + 1 == MAX_ATTEMPTS,
                    retry: Arc::clone(&retry),
                    request_id,
                    request_seq,
                    slot_held,
                    deadline,
                    body: body.clone(),
                }))
                .build();
            let Some((subrequest, handle)) = create_full_body_subrequest(session, ctx, body.len())
            else {
                // Unreachable while `allow_spawning_subrequest` returns true; a 500, not a hang.
                return fail_boxed(session, &request_id, 500, "internal error").await;
            };
            let attempt = tokio::spawn(subrequest.run());
            let piped = Box::pin(pipe_full_body(session, handle, body.clone(), &retry)).await;
            let decision = retry.lock().ok().and_then(|mut r| r.take());
            match attempt_end(&piped, decision) {
                AttemptEnd::Done => return Ok(true),
                AttemptEnd::Retry(decision) => {
                    // Let the abandoned attempt finish (its channels are closed, so it aborts its
                    // upstream at once) before the next one starts: it still holds that
                    // candidate's breaker permit, which a half-open breaker has only one of.
                    // Bounded, so an attempt that somehow never notices cannot stall the client.
                    reap_attempt(attempt, &request_id).await;
                    match decision {
                        // The key walk (if any) ended on this candidate: the rest of the walk is
                        // vendor failover, as on a small body (D81).
                        RelayRetry::Candidate(i) => {
                            skip |= 1 << i;
                            resume = None;
                        }
                        // A key walk resumes on this candidate first: the re-run may re-order the
                        // row, and a 429 must never become a vendor switch. The other candidates
                        // stay in the walk, so the next key's 5xx still fails over.
                        RelayRetry::Key { candidate, key } => {
                            if let Some(k) = keys.get_mut(usize::from(candidate)) {
                                *k = key;
                            }
                            resume = Some(candidate);
                        }
                        RelayRetry::Reset(i) => reset |= 1 << i,
                    }
                }
                AttemptEnd::Fail => {
                    // A panicked attempt is logged with the request id before its 502 (D207).
                    let reaped = reap_attempt(attempt, &request_id).await;
                    return Err(match piped {
                        Err(e) => e,
                        // The attempt ended without a response or an error (it panicked, say).
                        // Never leave the client waiting on a connection pingora would keep alive.
                        Ok(_) => reaped.unanswered(),
                    });
                }
            }
        }
        fail_boxed(session, &request_id, 502, "no candidate provider available").await
    }

    /// Cool this attempt's pool key because its account is out of credit (D180): later requests
    /// start past it, and a catalog walk leaves the provider out once all its keys cool. Waiting
    /// does not fix it, so it is cooled as a 401 is.
    fn cool_unfunded_key(&self, request_id: &RequestId, provider: &Provider, key: u8, status: u16) {
        self.state.metrics.key_cooled(KeyCooled::Unfunded);
        warn!(
            %request_id,
            provider = provider.name.as_str(),
            key,
            status,
            "pool key is out of credit; cooling it",
        );
        provider.mark_key_bad(key);
    }

    /// Judge a `402` a catalog walk abandons at its head, unread (`by_status`: `None` for any other
    /// answer, else [`remedy::unfunded_402`]'s verdict). Out of credit by its status alone: cool
    /// the key (D258). OpenRouter's, which may be one request too costly instead: a strike against
    /// the key, and the [`route::OPENROUTER_402_STRIKES`]th in a row with no 2xx between cools it
    /// (D261). A relayed 402 is read by `logging` instead: out of credit cools the key (which spends
    /// its strikes), too costly neither counts nor resets one.
    fn abandon_402(&self, rc: &RequestCtx, by_status: Option<bool>, status: u16) {
        let cool = match by_status {
            Some(true) => true,
            Some(false) => rc.leg.provider.strike_unread_402(rc.leg.pool_key),
            None => false,
        };
        if cool {
            self.cool_unfunded_key(&rc.request_id, &rc.leg.provider, rc.leg.pool_key, status);
        }
    }

    /// Resolve a 2xx's pending health verdict from the first response bytes: count an
    /// error-in-200 against the candidate's breaker.
    fn settle_health(&self, rc: &mut RequestCtx, chunk: &[u8], end_of_stream: bool) {
        let streaming = rc.streaming;
        let Some(a) = rc.auto.as_mut() else { return };
        let Some(pending) = a.health.as_mut() else {
            return;
        };
        let room = HEALTH_PREFIX_CAP.saturating_sub(pending.prefix.len());
        pending.extend_prefix(&chunk[..room.min(chunk.len())]);
        let verdict = match body_reports_error(&pending.prefix, streaming) {
            Some(error) => error,
            // Undecidable within the cap, or the body ended first: an answer we cannot fault.
            None if end_of_stream || pending.prefix.len() >= HEALTH_PREFIX_CAP => false,
            None => return,
        };
        let permit = a.health.take().and_then(|h| h.permit);
        // The head resolved this attempt's breaker permit as a success (see `response_filter`):
        // only the body shows the provider is broken. Count it as the failure it is, so a host that
        // keeps answering errors in 200s opens its breaker and every walk, session pins included,
        // skips it. The windowed rule trips once failures reach the threshold and match the
        // successes, which one success plus one failure per such answer does.
        if verdict
            && let Some(b) = rc.leg.provider.breaker.as_ref()
            && let Some(permit) = permit
        {
            b.record_late_failure(permit);
        }
    }

    /// Replay a cached 2xx. Boxed so its write future is not inlined into `request_filter`.
    async fn reply_cache_hit(
        session: &mut Session,
        request_id: &str,
        hit: &cache::CachedResponse,
    ) -> Result<bool> {
        let mut len_buf = ArrayString::<20>::new();
        let _ = write!(len_buf, "{}", hit.body.len());
        let mut resp = ResponseHeader::build(hit.status, Some(6))?;
        let ct = if hit.content_type.is_empty() {
            "application/json"
        } else {
            hit.content_type.as_ref()
        };
        resp.insert_header("content-type", ct)?;
        resp.insert_header("content-length", len_buf.as_str())?;
        resp.insert_header(REQUEST_ID_HEADER, request_id)?;
        resp.insert_header(CACHE_STATUS_HEADER, "hit")?;
        if !hit.provider.is_empty() {
            resp.insert_header(PROVIDER_HEADER, hit.provider.as_ref())?;
        }
        session.write_response_header(Box::new(resp), false).await?;
        session
            .write_response_body(Some(hit.body.clone()), true)
            .await?;
        Ok(true)
    }

    /// Abort a translated response whose buffer crossed [`translate::MAX_TRANSLATE_BUFFER`].
    ///
    /// Headers are already downstream, so there is no clean error to send — the same trade as the
    /// request-body cap in `request_body_filter`: an abuse guard, not a client path.
    fn translate_overflow(
        &self,
        request_id: &str,
        buffer: &'static str,
    ) -> Box<pingora_core::Error> {
        self.state
            .metrics
            .rejection(Rejection::ResponseTooLarge)
            .inc();
        warn!(
            request_id,
            buffer,
            limit = translate::MAX_TRANSLATE_BUFFER,
            "translated response exceeds the buffer limit; aborting",
        );
        pingora_core::Error::new_str("translated response exceeds buffer limit")
    }

    /// Hold, in the process body budget, the heap translating `body` will take
    /// ([`translate::translation_heap`]) for as long as the hold lives (D216). A body past the
    /// replay buffer only, like every other budget charge: smaller ones are bounded by concurrency.
    /// Never exempt on a `FullBody` re-run, whose parent reserved two copies of the body, not the
    /// `Value`s built from it. A translation that needs more than the whole budget is a 413 naming
    /// translation; one that does not fit now is the budget's retryable 503. With the budget off
    /// there is nothing to hold, and nothing is refused. On a response the status line is already
    /// downstream, so either refusal aborts it.
    fn hold_translation_heap(
        &self,
        body: &[u8],
    ) -> Result<Option<crate::concurrency::BudgetHold<'_>>> {
        let Some(budget) = self.state.body_budget.as_ref() else {
            return Ok(None);
        };
        if !past_replay_buffer(body.len()) {
            return Ok(None);
        }
        let heap = translate::translation_heap(body);
        if heap > budget.limit() {
            return Err(self.refusal_error(Refusal::TranslateTooLarge).into_down());
        }
        match budget.hold(heap) {
            Some(hold) => Ok(Some(hold)),
            None => Err(self.refusal_error(Refusal::BodyMemory).into_down()),
        }
    }

    async fn reply_cache_hit_boxed(
        session: &mut Session,
        request_id: &str,
        hit: &cache::CachedResponse,
    ) -> Result<bool> {
        Box::pin(Self::reply_cache_hit(session, request_id, hit)).await
    }
}

/// Header names carrying a plain static API key (no OAuth/signing), checked in order. Anthropic:
/// `x-api-key`. Azure OpenAI: `api-key`. Google Gemini: `x-goog-api-key`. `Authorization: Bearer`
/// (OpenAI and everyone else) is checked separately below since it needs prefix-stripping.
const STATIC_KEY_HEADERS: [&str; 3] = ["x-api-key", "api-key", "x-goog-api-key"];

/// The client headers a managed request forwards to the provider. Every other client header is
/// dropped in `upstream_request_filter` before the gateway adds its own (pool key, `Host`,
/// `accept-encoding`, OpenRouter attribution, a translated walk's `anthropic-version` /
/// `anthropic-beta`).
///
/// An allowlist rather than a blocklist because the request rides on Beyond's pool key: a header
/// the gateway has never heard of is one it cannot vouch for. `openai-organization` and
/// `openai-project` would switch the org or project the pool key bills to (and 401 for an SDK user
/// with `OPENAI_ORG_ID` set), `cookie` / `proxy-authorization` / `x-goog-user-project` are someone
/// else's credentials or billing selectors, and SDK telemetry (`x-stainless-*`) is noise. Framing
/// headers stay so the body arrives intact, and `expect` stays so a client waiting on
/// `100 Continue` is answered by the provider rather than by its own timeout (unless the gateway
/// already answered it while reading the body, in which case `peek_body_model` removed it).
const MANAGED_FORWARD_HEADERS: [&str; 8] = [
    "content-type",
    "content-length",
    "transfer-encoding",
    "expect",
    "accept",
    "user-agent",
    "anthropic-version",
    "anthropic-beta",
];

/// The beta that turned on fine-grained tool input streaming before each tool's
/// `eager_input_streaming` field replaced it; kept only for a candidate that takes the field.
const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

/// `anthropic-beta` tokens a managed request may send on the pool key. Each changes how a request
/// is parsed or streamed, and none changes what Anthropic charges per token or runs server-side
/// tools the gateway does not meter. Anything else is dropped: `context-1m-*` switches on premium
/// long-context pricing, `mcp-client-*` / `code-execution-*` / `files-api-*` reach servers, sandboxes
/// and storage on Beyond's account, and `oauth-*` is meaningless beside a pool API key. The gateway's
/// own [`translate::THINKING_BINDING_BETA`] is listed too, so a Messages client on a binding model
/// may send it itself. Fast mode's beta is not here: it doubles the per-token price, and is kept
/// only on a request to direct Anthropic (see [`retain_managed_client_headers`]).
const MANAGED_ANTHROPIC_BETAS: [&str; 8] = [
    "claude-code-20250219",
    "prompt-caching-2024-07-31",
    "interleaved-thinking-2025-05-14",
    FINE_GRAINED_TOOL_STREAMING_BETA,
    "context-management-2025-06-27",
    "token-efficient-tools-2025-02-19",
    "output-128k-2025-02-19",
    translate::THINKING_BINDING_BETA,
];

/// Drop every client header a managed request may not forward ([`MANAGED_FORWARD_HEADERS`]), and
/// every `anthropic-beta` token not in [`MANAGED_ANTHROPIC_BETAS`], plus
/// [`translate::FAST_MODE_BETA`] when `anthropic` (the attempt goes to direct Anthropic, the only
/// provider with fast mode). Fast mode is a product Beyond sells at its price: the row records the
/// speed asked and the speed served (`usage.speed`), so it is billed as served (D266). Without
/// `eager_tool_streaming` (the attempt's candidate does not take fine-grained tool streaming,
/// `providers::catalog::serves_eager_input_streaming`), [`FINE_GRAINED_TOOL_STREAMING_BETA`] is
/// dropped too, as `request_body_filter` drops each tool's `eager_input_streaming` (D272). The
/// header sweep allocates nothing (D92: it collected the names into a `Vec` on nearly every
/// managed request); only an `anthropic-beta` value with a token to drop is rebuilt.
fn retain_managed_client_headers(
    req: &mut pingora::http::RequestHeader,
    anthropic: bool,
    eager_tool_streaming: bool,
) -> Result<()> {
    let allowed = |t: &str| {
        (MANAGED_ANTHROPIC_BETAS.contains(&t)
            && (eager_tool_streaming || t != FINE_GRAINED_TOOL_STREAMING_BETA))
            || (anthropic && t == translate::FAST_MODE_BETA)
    };
    remove_headers_where(req, |name| !MANAGED_FORWARD_HEADERS.contains(&name));
    let betas = req.headers.get_all("anthropic-beta");
    let mut values = betas.iter();
    let clean = match (values.next(), values.next()) {
        (None, _) => true,
        (Some(v), None) => v
            .to_str()
            .is_ok_and(|v| v.split(',').map(str::trim).all(allowed)),
        _ => false,
    };
    if !clean {
        let kept = betas
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|v| v.split(','))
            .map(str::trim)
            .filter(|t| allowed(t))
            .collect::<Vec<_>>()
            .join(",");
        req.remove_header("anthropic-beta");
        if !kept.is_empty() {
            req.insert_header("anthropic-beta", kept)?;
        }
    }
    Ok(())
}

/// Drop every `x-beyond-*` header from a request about to go upstream (D67). Allocates nothing.
fn strip_beyond_headers(req: &mut pingora::http::RequestHeader) {
    remove_headers_where(req, |name| name.starts_with("x-beyond-"));
}

/// Remove every header whose (lower-case) name `drop` matches, allocating nothing (D92): names are
/// copied in batches onto the stack, since the map cannot be borrowed while it is edited. A name
/// past the stack slot's length (no real header) is cloned instead.
fn remove_headers_where(req: &mut pingora::http::RequestHeader, drop: impl Fn(&str) -> bool) {
    const BATCH: usize = 16;
    loop {
        let mut names = ArrayVec::<ArrayString<64>, BATCH>::new();
        let mut long = None;
        let mut more = false;
        for k in req.headers.keys().map(http::HeaderName::as_str) {
            if !drop(k) {
                continue;
            }
            if names.is_full() {
                more = true;
                break;
            }
            match ArrayString::from(k) {
                Ok(name) => names.push(name),
                Err(_) => {
                    long = http::HeaderName::from_bytes(k.as_bytes()).ok();
                    more = true;
                    break;
                }
            }
        }
        for name in &names {
            req.remove_header(name.as_str());
        }
        if let Some(name) = long {
            req.remove_header(&name);
        }
        if !more {
            return;
        }
    }
}

/// `s` with its `%XX` escapes decoded, borrowed when it has none. A malformed escape is kept as
/// written. Used only to *recognize* a credential the way a provider would read it (Google decodes
/// `k%65y` as `key`), never to rebuild what is forwarded.
fn percent_decoded(s: &str) -> Cow<'_, str> {
    if !s.contains('%') {
        return Cow::Borrowed(s);
    }
    let hex = |b: u8| char::from(b).to_digit(16);
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(h), Some(l)) = (
                bytes.get(i + 1).copied().and_then(hex),
                bytes.get(i + 2).copied().and_then(hex),
            )
        {
            out.push(u8::try_from(h * 16 + l).unwrap_or(b'%'));
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Cow::Owned(String::from_utf8_lossy(&out).into_owned())
}

/// Whether a raw query pair (`k=v`) names the `key` credential param, under any spelling a provider
/// decodes to `key` (`k%65y`).
fn is_key_param(pair: &str) -> bool {
    let name = pair.split_once('=').map_or(pair, |(k, _)| k);
    name == "key" || percent_decoded(name) == "key"
}

/// Every `key` query param's raw value, in order (Gemini's `?key=` convention — the sole
/// query-param credential shape among recognized providers).
fn key_params(query: &str) -> impl Iterator<Item = &str> {
    query
        .split('&')
        .filter(|p| is_key_param(p))
        .map(|p| p.split_once('=').map_or("", |(_, v)| v))
}

/// The token of one `Authorization` value: `Bearer <token>`, scheme matched case-insensitively and
/// any run of whitespace around the token tolerated (a lenient provider trims it). `None` for any
/// other scheme.
fn bearer_token(v: &str) -> Option<&str> {
    let v = v.trim();
    let scheme = v.get(..6)?;
    let rest = &v[6..];
    (scheme.eq_ignore_ascii_case("bearer") && rest.starts_with(|c: char| c.is_ascii_whitespace()))
        .then(|| rest.trim_start())
}

/// The key a request presents, and whether it is managed.
#[derive(Debug, PartialEq, Eq)]
struct Presented<'a> {
    /// The value to verify (managed) or to rate-guard as a BYO token. Borrowed from the request.
    key: &'a str,
    /// Some credential location carries a virtual key (`bai_v1…`/`bai_v2…`). The request is then
    /// managed whatever the others hold: verified (fail-closed, 401) and every location stripped.
    managed: bool,
}

/// Extract the presented key (virtual or BYO) from wherever the client's SDK puts it. Every
/// recognized shape is a **plain static key** (no OAuth/signing), so one neutral virtual key works
/// in any of them: Anthropic's `x-api-key`, Azure OpenAI's `api-key`, Google Gemini's
/// `x-goog-api-key` (header, falling back to the `?key=` query param — Gemini accepts either),
/// OpenAI's `Authorization: Bearer` (scheme matched case-insensitively). Borrowed from the request —
/// no per-request copy. Empty values count as absent.
///
/// **A managed key anywhere wins** (D29, D65). Every value of every location is read: each line of a
/// repeated header, each `key` param (including a percent-encoded name such as `k%65y`, and a
/// percent-encoded value), and a `Bearer` with extra whitespace. When any of them carries a virtual
/// key the request is managed, and `upstream_request_filter` / `strip_key_param` strip every
/// location. Reading only the first value of each let a junk first line, a second `?key=`, an
/// encoded name or a double space classify the request as BYO, which forwards it untouched — the
/// virtual key included. Otherwise the first location in the order above wins; the query param is
/// last since keys in a URL end up in proxy/access logs.
///
/// A value that is managed only once percent-decoded is returned as written: it fails verification
/// (401), which is the right answer for a key no SDK would encode.
fn extract_virtual_key(req: &pingora::http::RequestHeader) -> Option<Presented<'_>> {
    let headers = STATIC_KEY_HEADERS
        .iter()
        .flat_map(|h| req.headers.get_all(*h))
        .filter_map(|v| v.to_str().ok())
        .map(|v| (v.trim(), false));
    // A scheme-less `Authorization: bai_v1…` is no credential a provider reads, but it is one the
    // gateway must not forward on a BYO request: it counts as managed, never as the BYO key.
    let auth = req
        .headers
        .get_all("authorization")
        .into_iter()
        .filter_map(|v| v.to_str().ok())
        .filter_map(|v| match bearer_token(v) {
            Some(t) => Some((t, false)),
            None => key::is_managed_prefix(v.trim()).then(|| (v.trim(), true)),
        });
    let query = req
        .uri
        .query()
        .into_iter()
        .flat_map(key_params)
        .map(|v| (v, key::is_managed_prefix(&percent_decoded(v))));
    let mut first = None;
    for (v, decoded_managed) in headers.chain(auth).chain(query) {
        if v.is_empty() {
            continue;
        }
        if decoded_managed || key::is_managed_prefix(v) {
            return Some(Presented {
                key: v,
                managed: true,
            });
        }
        first = first.or(Some(v));
    }
    first.map(|key| Presented {
        key,
        managed: false,
    })
}

/// `path_and_query` without its `key` query params, or `None` when it carries none. Other params
/// (Azure's `api-version`) keep their order. Managed requests only: a `?key=` there is the virtual
/// key, which must not reach the provider, while a BYO `?key=` is the caller's own Gemini key. Every
/// spelling a provider decodes to `key` is dropped (`k%65y`), and every repeat of it.
fn strip_key_param(path_and_query: &str) -> Option<String> {
    let (path, query) = path_and_query.split_once('?')?;
    if !query.split('&').any(is_key_param) {
        return None;
    }
    let mut out = String::with_capacity(path_and_query.len());
    out.push_str(path);
    let mut sep = '?';
    for pair in query.split('&').filter(|p| !is_key_param(p)) {
        out.push(sep);
        out.push_str(pair);
        sep = '&';
    }
    Some(out)
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

/// The catalog row a billing row prices against: the provider-echoed id first, then the requested
/// one, each tried as spelled and then with a dated snapshot suffix removed. `None` when neither
/// names a row — an unpriced model, which a consumer must not silently price as something else.
///
/// The catalog lists aliases (`gpt-5`) and vendor slugs (`openai/gpt-5`), never the snapshots a
/// provider echoes (`gpt-5-2025-08-07`, `claude-sonnet-4-5-20250929`), so without the strip a
/// provider-routed row's `model` matches nothing. Runs once per billing row, on strings that are
/// already in hand.
fn price_model(billed: &str, requested: &str) -> Option<&'static str> {
    [billed, requested]
        .into_iter()
        .find_map(|m| price_row(m).map(|r| r.model))
}

/// The catalog row one model id prices at: as spelled (a catalog name, or any candidate's
/// provider-specific spelling of it), then with a dated snapshot suffix removed. The one lookup
/// both [`price_model`] and the managed `/{provider}` refusal ([`BodyPlan::catalog_check`])
/// make on the requested model, so a managed `/{provider}` request is refused exactly when its
/// requested model would leave its billing row without a `price_model`.
fn price_row(m: &str) -> Option<&'static route::ModelRoute> {
    route::model_route(m).or_else(|| strip_snapshot_date(m).and_then(route::model_route))
}

/// Set OpenRouter's attribution headers when (and only when) the provider is OpenRouter
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
    provider: Option<ProviderId>,
    managed: bool,
) -> Result<()> {
    if provider == Some(ProviderId::OpenRouter) && managed {
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
    e.is_some_and(|e| {
        !matches!(e.esource(), pingora_core::ErrorSource::Downstream) && !is_deadline(e)
    })
}

/// How much of the request body one attempt has fed through `request_body_filter`: one count, read
/// by name for what each reader asks of it.
#[derive(Debug, Default)]
struct UploadProgress(usize);

impl UploadProgress {
    /// Count a chunk of `n` bytes. `false` once the body has crossed [`MAX_REQUEST_BODY`]: the cap
    /// holds on the streamed size too, which a chunked body declares nowhere up front.
    fn feed(&mut self, n: usize) -> bool {
        self.0 = self.0.saturating_add(n);
        self.0 <= MAX_REQUEST_BODY
    }

    /// Whether any chunk has reached `request_body_filter` this attempt: what blames a stalled
    /// upload on the client (`client_still_uploading`; none fed means the attempt never got past
    /// connecting), and, false, what tells a retry there is nothing to undo, since every
    /// body-phase accumulator is fed on the branch that counts here.
    fn started(&self) -> bool {
        self.0 > 0
    }
}

/// Whether this attempt had started sending the client's body upstream and the client had not
/// finished it. An upstream error then reflects the client (a stalled or abandoned upload), not the
/// provider. Zero bytes fed means the attempt never got past connecting.
fn client_still_uploading(session: &mut Session, rc: &RequestCtx) -> bool {
    rc.attempt.upload.started() && !session.as_mut().is_body_done()
}

/// Whether the upstream's read timeout fired while the client still owed body bytes (D260): the
/// client stalled mid-upload with its connection open. Pingora's upstream half waits on the
/// response and on the next body chunk at once, re-arming the read timeout on each chunk, and a
/// body write blocked on an upstream that stopped reading is a `WriteTimedout` instead; so this
/// timeout means neither side moved a byte for `read_timeout_secs`, with the provider waiting on
/// the client. Zero bytes fed is excluded as [`client_still_uploading`] excludes it: a client
/// that sent `Expect: 100-continue` waits on the provider's answer before its first byte. Any
/// source: pingora's own downstream body read timeout (HTTP/1.1, 60 s) is already the client's.
fn client_stalled_upload(session: &mut Session, rc: &RequestCtx, e: &pingora_core::Error) -> bool {
    e.etype() == &pingora_core::ErrorType::ReadTimedout && client_still_uploading(session, rc)
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

/// Pingora will only replay a body that has fully arrived and fit in its private 64 KiB buffer.
/// See `upstream_response_filter` — the same gate for a 429 key-walk and a 5xx vendor walk.
fn body_replayable(session: &mut Session) -> bool {
    session.as_mut().is_body_done() && !session.as_ref().retry_buffer_truncated()
}

/// Same cap as pingora's private `BODY_BUF_LIMIT`. The managed `/v1` / headerless `/auto` peek
/// reads at most this many bytes before `upstream_peer`. A root `model` that has not appeared by
/// then is missing — 404. Past this, pingora will not replay the prefix itself (the retry buffer
/// truncates), so [`ModelRouting::replay`] carries our copy for `request_body_filter` to prepend.
const BODY_PEEK_LIMIT: usize = 64 * 1024;

/// Whether a body of `n` bytes is past [`BODY_PEEK_LIMIT`]: pingora cannot replay it from its own
/// buffer, so every copy the gateway holds of it is charged to the process body budget. Below it,
/// concurrency bounds the bodies in memory.
fn past_replay_buffer(n: usize) -> bool {
    n > BODY_PEEK_LIMIT
}

/// [`RequestCtx::tally_eager`]: a managed provider-routed body that may outgrow the retry buffer
/// is the one case where no copy is left for `logging` to tally.
fn tally_eager(managed: bool, catalog: bool, declared_len: Option<usize>) -> bool {
    managed && !catalog && declared_len.is_none_or(past_replay_buffer)
}

/// The `x-beyond-model` header, viewed as a catalog lookup. Present-but-unknown is distinct from
/// absent: the header wins, so an unknown header is a 404 rather than a fall-through to the body.
#[derive(Clone, Copy)]
enum CatalogHeader {
    Absent,
    Known(&'static route::ModelRoute),
    Unknown,
}

fn catalog_from_header(session: &Session) -> CatalogHeader {
    match session
        .req_header()
        .headers
        .get(route::MODEL_HEADER)
        .and_then(|v| v.to_str().ok())
    {
        None => CatalogHeader::Absent,
        Some(name) => match route::model_route(name) {
            Some(r) => CatalogHeader::Known(r),
            None => CatalogHeader::Unknown,
        },
    }
}

/// What `peek_body_model` read before a catalog row could be chosen.
struct BodyPeek {
    model: Option<String>,
    /// The full pre-rewrite body, when the peek read it to the end. Read only once `over_cap` and
    /// `over_budget` are known to be false.
    complete: Option<Vec<u8>>,
    /// The whole body was read and it outgrew pingora's 64 KiB retry buffer: pingora has nothing
    /// to send and nothing left to read. The caller re-runs the request with `complete` as its body
    /// (see [`FullBody`]).
    relay: bool,
    /// [`MAX_REQUEST_BODY`] was reached before the body ended. 413.
    over_cap: bool,
    /// Holding more of the body would cross `max_buffered_body_bytes`. 503.
    over_budget: bool,
}

/// Read the whole body before connecting, scanning it for the root `model`.
///
/// Enables pingora's 64 KiB retry buffer and feeds each *new* chunk to [`peek::ModelScanner`] once
/// (a `model` after a long prompt is linear, not quadratic). `reserve` is the caller's
/// content-length.
///
/// Below [`BODY_PEEK_LIMIT`] pingora replays its own buffer, on the first attempt and on every
/// failover. Once the buffer has truncated it replays nothing, so the caller re-runs the request as
/// a subrequest carrying the body (`relay`), which is also what lets a large body fail over (see
/// [`FullBody`]). The whole body, never a stop at `model`: inbound Responses' `store` /
/// `previous_response_id` can sit after `input`, the cache hashes everything, and stock Python SDKs
/// put `model` *after* `messages` anyway, so a long agent turn reads to the end regardless.
///
/// A body growing past the replay buffer reserves twice its size in the body budget as it grows
/// (this buffer, and the `FullBody` re-run's own copy): a chunked upload has no length to reserve
/// up front.
///
/// A client that stalls here holds its tenant slot while nothing upstream is open, so each read is
/// bounded: by pingora's own downstream read timeout on HTTP/1.1 (`client_read_timeout_secs`, set
/// in `request_filter`), by `read_timeout` here on HTTP/2, whose pingora server has none. A body
/// still arriving at the request's `deadline` ends too. Either is the client's: every error here
/// is tagged downstream, so `fail_to_proxy` answers a timeout 408 and writes nothing to a client
/// that went away.
async fn peek_body_model(
    session: &mut Session,
    reserve: usize,
    held: &mut Held,
    read_timeout: Option<Duration>,
    deadline: u64,
) -> pingora_core::Result<BodyPeek> {
    let down = |mut e: Box<pingora_core::Error>| {
        e.as_down();
        e
    };
    if expects_continue(session) {
        session.write_continue_response().await.map_err(down)?;
        // The client has its `100 Continue`, from us. Forwarding `Expect` would make the provider
        // send a second one, which the gateway relays: two interim responses for one request.
        session
            .req_header_mut()
            .remove_header(&http::header::EXPECT);
    }
    session.as_mut().enable_retry_buffering();
    let mut buf = Vec::with_capacity(reserve.min(MAX_REQUEST_BODY));
    let mut scanner = peek::ModelScanner::new();
    let mut over_cap = false;
    let mut over_budget = false;
    loop {
        if buf.len() >= MAX_REQUEST_BODY {
            over_cap = true;
            break;
        }
        if crate::deadline::expired(deadline) {
            return Err(down(pingora_core::Error::explain(
                pingora_core::ErrorType::ReadTimedout,
                "request body still arriving at request_max_secs",
            )));
        }
        let read = session.read_request_body();
        let read = match read_timeout {
            Some(t) => tokio::time::timeout(t, read).await.unwrap_or_else(|_| {
                pingora_core::Error::e_explain(
                    pingora_core::ErrorType::ReadTimedout,
                    "reading body, client_read_timeout_secs",
                )
            }),
            None => read.await,
        };
        match read.map_err(down)? {
            // An empty chunk feeds nothing, adds nothing, and re-reserves the same total (a no-op).
            Some(chunk) => {
                scanner.feed(&chunk);
                buf.extend_from_slice(&chunk);
                if past_replay_buffer(buf.len()) && !held.reserve_body(buf.len().saturating_mul(2))
                {
                    over_budget = true;
                    break;
                }
            }
            None => break,
        }
    }
    let truncated = session.as_ref().retry_buffer_truncated();
    let done = session.as_mut().is_body_done();
    Ok(BodyPeek {
        model: scanner.take_model(),
        complete: done.then_some(buf),
        relay: truncated && done,
        over_cap: over_cap && !done,
        over_budget,
    })
}

/// Whether the client asked for `100 Continue` before it sends the body. curl does for any body
/// over 1 KiB and waits a second for it; a gateway that reads the body in `request_filter` must
/// answer it, since pingora only does when it streams the body itself.
fn expects_continue(session: &Session) -> bool {
    session
        .req_header()
        .headers
        .get(http::header::EXPECT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
}

/// A subrequest's context when the gateway re-runs a request whose body it read in full.
///
/// Pingora can only send a request body it replays from its 64 KiB retry buffer or reads from the
/// client. Once the gateway has read a larger body neither is left, so
/// [`AiProxy::relay_full_body`] spawns a pingora subrequest that carries the body and pipes its
/// response to the client. The subrequest runs this same proxy with this context: the catalog row
/// is already chosen (no second peek), and the parent already charged the rate guardrail and counted
/// the request. Only the gateway creates it, so a client cannot use it to skip anything.
///
/// It is also how a large body fails over. Inside the subrequest the body still cannot be replayed
/// by pingora, so where an ordinary walk would retry (a 5xx with another candidate left, a 429 with
/// another pool key) the subrequest records the decision in `retry` and relays its response. The
/// parent drops that response before a byte reaches the client and runs a new subrequest that skips
/// the failed candidate (`skip`) or resumes the key walk (`keys`) — the same walk, one attempt per
/// subrequest.
#[derive(Clone)]
struct FullBody {
    route: &'static route::ModelRoute,
    session_field: Option<&'static str>,
    /// Catalog indices (bit per index) an earlier attempt failed over from.
    skip: u8,
    /// Catalog indices (bit per index) that cannot serve this body (`route::unserved`), read by the
    /// parent while it held the whole body.
    unserved: u8,
    /// The walk takes the row's Responses arm for a tool count Chat Completions refuses
    /// (`route::tools_need_responses_arm`), read by the parent while it held the whole body.
    responses_tools: bool,
    /// Per catalog index, the pool key an earlier attempt's key walk moved to, or [`NO_KEY_WALK`]:
    /// start on the provider's first key not cooling off (`Provider::first_key`, D71/D83).
    keys: [u8; route::MAX_CANDIDATES],
    /// Catalog indices (bit per index) that already had their one same-candidate retry after a
    /// stale reused connection failed before the upstream had the body.
    reset: u8,
    /// A key walk in progress on this catalog index: it goes first in the walk, so a re-ordered row
    /// cannot move a 429's key walk onto another vendor. The other candidates stay usable, so a
    /// 5xx (or a last key's auth failure) that ends the key walk still fails over (D81).
    resume: Option<u8>,
    /// The parent's attempt bound is reached: record no retry, relay this attempt's answer.
    final_attempt: bool,
    /// Set by this attempt when it would have retried but could not replay the body.
    retry: Arc<std::sync::Mutex<Option<RelayRetry>>>,
    /// The parent's request id and sequence: every attempt is the same request to the client and
    /// in the logs, and the same `x-beyond-split` draw and probe seed to the walk (a new seq per
    /// attempt could land a 429 key walk on another vendor).
    request_id: RequestId,
    request_seq: u64,
    /// The parent holds this tenant's concurrency slot for the whole request, taken before it read
    /// the body. Attempts neither take nor release one.
    slot_held: bool,
    /// The parent's deadline (`RequestCtx::deadline`): every attempt ends by it.
    deadline: u64,
    /// The parent's copy of the body (a refcount, not a copy): what `logging` tallies for an input
    /// estimate ([`input_estimate`]).
    body: Bytes,
}

/// What every [`FullBody`] attempt inherits from the request that re-runs it.
#[derive(Clone, Copy)]
struct Parent {
    request_id: RequestId,
    request_seq: u64,
    deadline: u64,
}

/// [`FullBody::keys`] for a candidate no earlier attempt walked keys on.
const NO_KEY_WALK: u8 = u8::MAX;

/// Move the walk slot holding catalog index `orig` to the front, keeping the others in order.
/// A no-op when `orig` is not in the walk.
fn walk_front(walk: &mut control::Walk, orig: u8) {
    let len = usize::from(walk.len).min(route::MAX_CANDIDATES);
    if let Some(at) = walk.indices[..len].iter().position(|&i| i == orig) {
        walk.indices[..=at].rotate_right(1);
    }
}

impl FullBody {
    /// Hand a retry back to the parent. Returns whether it was recorded: never on the final
    /// attempt, whose answer is relayed whatever it is.
    fn record(&self, retry: RelayRetry) -> bool {
        if self.final_attempt {
            return false;
        }
        if let Ok(mut slot) = self.retry.lock() {
            *slot = Some(retry);
            return true;
        }
        false
    }
}

/// The input-token estimate for a billing row that needs one (`usage::InputTally`): the tally fed
/// as the body streamed when it was ([`RequestCtx::tally_eager`]), else a tally of the body itself,
/// made now: the `FullBody` parent's copy, or pingora's retry buffer (the whole body, within
/// 64 KiB). Most rows carry the provider's own count and never get here, so the body is not walked
/// for them at all.
fn input_estimate(session: &Session, rc: &RequestCtx) -> u64 {
    if rc.tally_eager {
        return rc.attempt.input_tally.estimate_tokens();
    }
    let body = full_body_ctx(session)
        .map(|fb| fb.body)
        .or_else(|| session.as_ref().get_retry_buffer());
    let mut tally = usage::InputTally::default();
    if let Some(body) = body {
        tally.feed(&body);
    }
    tally.estimate_tokens()
}

fn full_body_ctx(session: &Session) -> Option<FullBody> {
    full_body(session).cloned()
}

/// This attempt's [`FullBody`] context, in place: `None` unless it is a `FullBody` re-run.
fn full_body(session: &Session) -> Option<&FullBody> {
    session
        .subrequest_ctx
        .as_ref()
        .and_then(|c| c.user_ctx())
        .and_then(|u| u.downcast_ref::<FullBody>())
}

/// How long [`AiProxy::relay_full_body`] waits for an abandoned attempt to wind down before it
/// starts the next one anyway.
const ABANDONED_ATTEMPT_GRACE: Duration = Duration::from_secs(2);

/// What [`AiProxy::relay_full_body`] does once an attempt's pipe has ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttemptEnd {
    /// The response reached the client.
    Done,
    /// Run the walk again as the attempt asked.
    Retry(RelayRetry),
    /// End the request with this attempt's error (or a 502 when it had none).
    Fail,
}

/// Decide from how an attempt's pipe ended and the retry (if any) the attempt recorded. The
/// client gone (a downstream error) is the end whatever the attempt recorded: a re-run would send
/// the whole body to another candidate or key for nobody, a generation billed upstream that no
/// client reads (D206).
fn attempt_end(piped: &Result<Piped>, decision: Option<RelayRetry>) -> AttemptEnd {
    match (piped, decision) {
        (Ok(Piped::Written), _) => AttemptEnd::Done,
        (Err(e), _) if e.esource() == &pingora_core::ErrorSource::Downstream => AttemptEnd::Fail,
        (_, Some(d)) => AttemptEnd::Retry(d),
        (_, None) => AttemptEnd::Fail,
    }
}

/// How a finished attempt's task ended, as [`reap_attempt`] saw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reaped {
    Finished,
    Panicked,
    StillRunning,
}

impl Reaped {
    /// The 502 for an attempt that ended with neither a response nor an error, saying whether it
    /// panicked.
    fn unanswered(self) -> Box<pingora_core::Error> {
        pingora_core::Error::explain(
            pingora_core::ErrorType::HTTPStatus(502),
            if self == Reaped::Panicked {
                "full-body attempt panicked"
            } else {
                "full-body attempt ended without a response"
            },
        )
    }
}

/// Wait, at most [`ABANDONED_ATTEMPT_GRACE`], for an attempt's task to end, and say how it did. A
/// panic is logged at error with the request id (D207): the subrequest dies with it, and the
/// client would otherwise get a 502 no log line explains.
async fn reap_attempt<T>(attempt: tokio::task::JoinHandle<T>, request_id: &str) -> Reaped {
    match tokio::time::timeout(ABANDONED_ATTEMPT_GRACE, attempt).await {
        Ok(Ok(_)) => Reaped::Finished,
        Ok(Err(e)) if e.is_panic() => {
            let payload = e.into_panic();
            let msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("non-string panic payload");
            tracing::error!(request_id, panic = msg, "full-body attempt panicked");
            Reaped::Panicked
        }
        Ok(Err(_)) => Reaped::Finished,
        Err(_) => Reaped::StillRunning,
    }
}

/// How one [`FullBody`] attempt's pipe ended.
#[derive(Debug)]
enum Piped {
    /// The response was written to the client.
    Written,
    /// The attempt recorded a retry; its response was dropped at the header.
    Abandoned,
    /// The subrequest closed its channel having written nothing.
    Empty,
}

/// Create a [`FullBody`] subrequest from this request.
///
/// Pingora builds a subrequest by rendering the parent's request header as HTTP/1.1 and parsing it
/// back. An HTTP/2 parent renders as `… HTTP/2`, which that parser rejects with a bare 400, and has
/// no `Host` (it carries `:authority`) and often no `Content-Length`, which would give the
/// subrequest an empty body. So the header is rendered as HTTP/1.1, with the body's real length and
/// a `Host`, and restored afterwards.
fn create_full_body_subrequest(
    session: &mut Session,
    ctx: SubrequestCtx,
    body_len: usize,
) -> Option<(pingora_proxy::PreparedSubrequest, SubrequestHandle)> {
    let original = session.req_header().clone();
    {
        let req = session.req_header_mut();
        req.set_version(http::Version::HTTP_11);
        let _ = req.insert_header(http::header::CONTENT_LENGTH, body_len.to_string());
        req.remove_header(&http::header::TRANSFER_ENCODING);
        if req.headers.get(http::header::HOST).is_none()
            && let Some(authority) = req.uri.authority().map(|a| a.as_str().to_owned())
        {
            let _ = req.insert_header(http::header::HOST, authority);
        }
    }
    let created = session
        .subrequest_spawner
        .as_ref()
        .map(|spawner| spawner.create_subrequest(session.as_downstream(), ctx));
    *session.req_header_mut() = original;
    created
}

/// Run one [`FullBody`] attempt's I/O: hand the subrequest its body, write its response to the
/// client, and watch the client while it runs.
///
/// Pingora's `pipe_subrequest` would do the first two, but with a preset body it never polls the
/// client, so a client that hangs up is only noticed on the next write. A hidden-reasoning model
/// writes nothing for minutes, and the upstream kept generating (and billing) after a cancel that
/// the direct path aborts at once. This loop idle-reads the client the way pingora's own proxy loop
/// does, and returns the moment it goes: dropping the channels is the subrequest's disconnect, so it
/// aborts its upstream and logs a cut-short stream's estimate.
///
/// An attempt that recorded a retry is abandoned at its response header, before a byte reaches the
/// client. Response tasks are drained before a proxy error is looked at (`biased`), so an error
/// cannot cut off the tail of a response already queued.
async fn pipe_full_body(
    session: &mut Session,
    handle: SubrequestHandle,
    body: Bytes,
    retry: &std::sync::Mutex<Option<RelayRetry>>,
) -> Result<Piped> {
    let SubrequestHandle {
        tx,
        mut rx,
        subreq_wants_body,
        subreq_proxy_error,
    } = handle;
    let mut wants_body = std::pin::pin!(subreq_wants_body);
    let mut proxy_error = std::pin::pin!(subreq_proxy_error);
    let mut body = Some(body);
    let (mut body_wait, mut error_wait) = (true, true);
    let mut written = false;
    let mut tasks = Vec::with_capacity(4);
    loop {
        tokio::select! {
            biased;
            task = rx.recv() => {
                let Some(task) = task else {
                    // The subrequest finished. A proxy error it hit is reported alongside.
                    return match proxy_error.try_recv() {
                        Ok(e) => Err(e),
                        Err(_) if written => Ok(Piped::Written),
                        Err(_) => Ok(Piped::Empty),
                    };
                };
                if matches!(task, HttpTask::Header(..))
                    && retry.lock().is_ok_and(|r| r.is_some())
                {
                    return Ok(Piped::Abandoned);
                }
                // Write what is already queued in one go, as pingora's own pipe does.
                tasks.push(task);
                while tasks.len() < 4
                    && let Ok(next) = rx.try_recv()
                {
                    tasks.push(next);
                }
                written = true;
                if session.write_response_tasks(std::mem::take(&mut tasks)).await? {
                    return Ok(Piped::Written);
                }
            }
            wanted = &mut wants_body, if body_wait => {
                body_wait = false;
                if wanted.is_ok() && let Some(b) = body.take() {
                    // The subrequest gone before reading is an error it reports on `proxy_error`.
                    let _ = tx.send(HttpTask::Body(Some(b), true)).await;
                }
            }
            e = &mut proxy_error, if error_wait => {
                error_wait = false;
                if let Ok(e) = e {
                    return Err(e);
                }
            }
            closed = session.downstream_session.read_body_or_idle(true) => {
                // No body is expected (it is already read), so this only returns when the client
                // leaves or errors.
                return Err(match closed {
                    Err(e) => e.into_down(),
                    Ok(_) => pingora_core::Error::new(pingora_core::ErrorType::ConnectionClosed)
                        .into_down(),
                });
            }
        }
    }
}

/// Most `request rejected` lines logged per second, process-wide (D263). A rejection is the one
/// line a client can make the gateway write at will — bad keys, a rate-limited flood — and one line
/// per rejected request made the log the bottleneck of a flood the rejection was meant to shed. A
/// second's first lines still carry their `request_id` for the oncall's grep; past the allowance
/// they are counted, and the next line logged says how many were not (`suppressed`).
/// `ai_rejections_total` still counts every one.
const REJECT_LOG_PER_SEC: u64 = 100;

/// Process-wide, as the log it protects is: every gateway in the process writes the one stdout.
static REJECT_LOG: LogRate = LogRate::new(REJECT_LOG_PER_SEC);

/// A per-second allowance of log lines shared by every worker: relaxed atomics, no lock. Approximate
/// at a second's boundary (a racing line may land in either second), which a log rate can afford.
pub(crate) struct LogRate {
    second: AtomicU64,
    used: AtomicU64,
    suppressed: AtomicU64,
    per_sec: u64,
}

impl LogRate {
    pub(crate) const fn new(per_sec: u64) -> Self {
        Self {
            second: AtomicU64::new(0),
            used: AtomicU64::new(0),
            suppressed: AtomicU64::new(0),
            per_sec,
        }
    }

    /// Whether to log a line at `now` (whole seconds on a monotonic clock). `Some(n)`: log it, and
    /// say `n` lines were suppressed since the last one logged. `None`: suppressed, and counted.
    pub(crate) fn admit(&self, now: u64) -> Option<u64> {
        let seen = self.second.load(Ordering::Relaxed);
        if now > seen
            && self
                .second
                .compare_exchange(seen, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.used.store(0, Ordering::Relaxed);
        }
        if self.used.fetch_add(1, Ordering::Relaxed) < self.per_sec {
            Some(self.suppressed.swap(0, Ordering::Relaxed))
        } else {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

/// The `request rejected` line every gateway-made rejection writes, under [`REJECT_LOG`]'s cap.
fn log_rejection(request_id: &str, status: u16, typ: &str) {
    if let Some(suppressed) = REJECT_LOG.admit(log_second()) {
        warn!(
            request_id,
            status,
            error_type = typ,
            suppressed,
            "request rejected"
        );
    }
}

/// Whole seconds since the first call: [`LogRate`]'s clock.
fn log_second() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs()
}

/// Write `refusal`: its status, body and `Retry-After`, and `x-beyond-request-id`, after the
/// `request rejected` line. See [`AiProxy::refuse`], which counts it and boxes this.
async fn write_refusal(session: &mut Session, request_id: &str, refusal: Refusal) -> Result<bool> {
    let status = refusal.status();
    log_rejection(request_id, status, refusal.kind());
    write_json_body(
        session,
        request_id,
        status,
        refusal.body(),
        refusal.retry_after(),
    )
    .await?;
    Ok(true)
}

/// Answer a request the gateway could not serve but did not refuse (no metric counts it: it is not
/// the client's request that is wrong), logged as a refusal is. Boxed, as [`AiProxy::refuse`] is.
async fn fail_boxed(
    session: &mut Session,
    request_id: &str,
    status: u16,
    msg: &'static str,
) -> Result<bool> {
    log_rejection(request_id, status, "api_error");
    Box::pin(write_json_body(
        session,
        request_id,
        status,
        error_body("api_error", msg),
        None,
    ))
    .await?;
    Ok(true)
}

/// An error the gateway raises for a failure that is no refusal of the request (every candidate
/// failed, or every breaker is open), with the status and message the client should see: carried
/// in `ErrorType::CustomCode` so [`failure_response`] answers with exactly this rather than a
/// generic line for the status.
fn gateway_error(status: u16, msg: &'static str) -> Box<pingora_core::Error> {
    pingora_core::Error::new(pingora_core::ErrorType::CustomCode(msg, status))
}

/// Whether `e` is the request deadline passing ([`AiProxy::deadline_cut`]). A policy limit, never a
/// provider outcome: [`is_upstream_failure`] is false for it, so no breaker hears of it, and
/// `error_while_proxy` passes it through as the gateway's own decision (no resend, no failover).
fn is_deadline(e: &pingora_core::Error) -> bool {
    matches!(Refusal::of(e), Some(Refusal::RequestDeadline))
}

/// The catalog miss for an `x-beyond-model` header that names no row, naming what it sent.
fn header_catalog_miss(session: &Session) -> Refusal {
    let name = session
        .req_header()
        .headers
        .get(route::MODEL_HEADER)
        .and_then(|v| v.to_str().ok());
    Refusal::CatalogMiss {
        name: name.map(Into::into),
        header: true,
    }
}

/// The client-facing `(status, error type, message)` for an error that ended a request, or `None`
/// when the client is already gone (nothing can be written). See `fail_to_proxy`.
fn failure_response(e: &pingora_core::Error) -> Option<(u16, &'static str, &'static str)> {
    use pingora_core::{ErrorSource as Source, ErrorType as T};
    let typ = |status: u16| match status {
        429 => "rate_limit_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    };
    let downstream = matches!(e.esource(), Source::Downstream);
    Some(match e.etype() {
        T::CustomCode(msg, status) => (*status, typ(*status), *msg),
        T::HTTPStatus(status) => {
            let msg = match status {
                400 => "bad request",
                413 => "request body too large",
                429 => "rate limit exceeded",
                502 => "upstream unavailable",
                503 => "provider temporarily unavailable",
                504 => "upstream timed out",
                400..=499 => "request rejected",
                _ => "upstream error",
            };
            (*status, typ(*status), msg)
        }
        // The client's connection is dead: nothing to write to.
        T::ReadError | T::WriteError | T::ConnectionClosed if downstream => return None,
        T::ReadTimedout if downstream => (408, "invalid_request_error", "request body timed out"),
        _ if downstream => (400, "invalid_request_error", "bad request"),
        T::ConnectTimedout | T::TLSHandshakeTimedout | T::ReadTimedout | T::WriteTimedout => {
            (504, "api_error", "upstream timed out")
        }
        T::ConnectRefused
        | T::ConnectNoRoute
        | T::ConnectError
        | T::ConnectProxyFailure
        | T::TLSHandshakeFailure
        | T::TLSWantX509Lookup
        | T::InvalidCert
        | T::HandshakeError
        | T::SocketError
        | T::BindError => (502, "api_error", "could not connect to the provider"),
        _ => match e.esource() {
            Source::Upstream => (502, "api_error", "upstream connection failed"),
            _ => (500, "api_error", "internal error"),
        },
    })
}

/// Write a JSON error `body` with `x-beyond-request-id`, and `Retry-After` when given: every
/// error the gateway itself answers goes out through here, refusals and failures alike.
async fn write_json_body(
    session: &mut Session,
    request_id: &str,
    status: u16,
    body: Bytes,
    retry_after: Option<u64>,
) -> Result<()> {
    // Content-length formatted into a stack buffer rather than `body.len().to_string()`: still one
    // `HeaderValue` allocation inside pingora, but no `String` of our own.
    let mut len_buf = ArrayString::<20>::new();
    let _ = write!(len_buf, "{}", body.len());
    let mut resp = ResponseHeader::build(status, Some(4))?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", len_buf.as_str())?;
    resp.insert_header(REQUEST_ID_HEADER, request_id)?;
    if let Some(secs) = retry_after {
        let mut ra = ArrayString::<20>::new();
        let _ = write!(ra, "{secs}");
        resp.insert_header(http::header::RETRY_AFTER, ra.as_str())?;
    }
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(body), true).await?;
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

/// Stock OpenAI/Anthropic SDKs list models at `GET /v1/models`. A managed request is answered from
/// the keyed catalog before the catalog walk, so an empty GET body is not a missing-model 404. A BYO
/// one relays to its provider like any BYO `/v1` request.
fn is_v1_models_list(session: &Session) -> bool {
    let req = session.req_header();
    let path = req.uri.path();
    (req.method == http::Method::GET || req.method == http::Method::HEAD)
        && (path == "/v1/models" || path == "/v1/models/")
}

/// [`ProxyHttp::request_summary`]'s line: method, path without the query, and host.
fn summary_line(req: &pingora::http::RequestHeader) -> String {
    let host = req
        .headers
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .or_else(|| req.uri.host())
        .unwrap_or("");
    format!("{} {}, Host: {host}", req.method, req.uri.path())
}

/// Resolve the provider name for a request whose first path segment matched no known/config
/// provider: `Some(name)` for the bare-path default — `path` is boundary-checked against
/// [`route::DEFAULT_PREFIX`] (see [`route::is_default_prefix`]) so a lookalike like Google
/// Gemini's `/v1beta/…` doesn't qualify — with the dialect picking openai/anthropic
/// ([`dialect_for_path`]); `None` for anything else, which the caller turns into a 404 rather than
/// silently guessing a provider (Task #7, pi-parity).
///
/// `credential`: the provider the request's BYO credentials belong to
/// ([`byo_credential_dialect`]). It wins over the path, so an Anthropic SDK call to `/v1/files`
/// never reaches OpenAI carrying `sk-ant-…`, and an OpenAI key on `/v1/messages` never reaches
/// Anthropic. With no credential that says, the path picks.
fn bare_default_provider_name(path: &str, credential: Option<Dialect>) -> Option<&'static str> {
    let dialect = credential.unwrap_or_else(|| dialect_for_path(path));
    route::is_default_prefix(path).then(|| route::dialect_default(dialect))
}

/// The provider a BYO key belongs to, from its shape: `sk-ant-…` is Anthropic's, any other `sk-…`
/// OpenAI's. `None` when the shape says nothing.
fn byo_key_dialect(key: &str) -> Option<Dialect> {
    if key.starts_with("sk-ant-") {
        Some(Dialect::Anthropic)
    } else if key.starts_with("sk-") {
        Some(Dialect::OpenAi)
    } else {
        None
    }
}

/// The provider the BYO credentials on a bare `/v1` request belong to, read from every value the
/// gateway would forward (D82): each `x-api-key` line (Anthropic's header, so Anthropic unless the
/// key's shape says otherwise) and each `Authorization: Bearer` token (by its shape only; an
/// opaque token says nothing). Managed and empty values do not count. `Ok(None)`: nothing says.
/// `Err(())`: they name different providers; every one would be forwarded, so whichever provider
/// the request went to would receive the other's key.
fn byo_credential_dialect(req: &pingora::http::RequestHeader) -> Result<Option<Dialect>, ()> {
    let byo = |v: &&str| !v.is_empty() && !key::is_managed_prefix(v);
    let api_keys = req
        .headers
        .get_all("x-api-key")
        .into_iter()
        .filter_map(|v| v.to_str().ok())
        .map(str::trim)
        .filter(byo)
        .map(|v| Some(byo_key_dialect(v).unwrap_or(Dialect::Anthropic)));
    let bearers = req
        .headers
        .get_all("authorization")
        .into_iter()
        .filter_map(|v| v.to_str().ok())
        .filter_map(bearer_token)
        .filter(byo)
        .map(byo_key_dialect);
    let mut vote = None;
    for d in api_keys.chain(bearers).flatten() {
        match vote {
            None => vote = Some(d),
            Some(v) if v == d => {}
            Some(_) => return Err(()),
        }
    }
    Ok(vote)
}

/// Whether a Responses body asks for background mode: a root `background` member (escapes in its
/// name decoded, any copy of it) whose value is `true`. `true` is a literal no escape can spell, so
/// a body without those bytes is not parsed.
fn requests_background(body: &[u8]) -> bool {
    memchr::memmem::find(body, b"true").is_some()
        && peek::root_members(body).is_some_and(|members| {
            members
                .iter()
                .any(|m| m.key_is(body, "background") && &body[m.value.0..m.value.1] == b"true")
        })
}

/// The `/{provider}/…` endpoints a managed key may reach: the metered generation calls and their
/// token-count / compact sub-resources ([`route::ENDPOINT_PATHS`], the table every routing
/// decision reads too, D209), matched by suffix so every provider's mount prefix (`/api/v1`,
/// `/openai/v1`, `/inference/v1`, `/anthropic/v1`, `/backend-api/codex`) works. Everything else on
/// a provider (files, batches, stored responses, fine-tuning, models) is refused for managed keys.
/// A query string is ignored.
fn is_managed_provider_endpoint(path_and_query: &str) -> bool {
    route::forward_endpoint(path_and_query).is_some()
}

/// The wire a `/{provider}/…` forwarded path (query allowed) is answered on: its
/// [`route::ENDPOINT_PATHS`] row's (a sub-resource reads as its parent: `/messages/count_tokens`
/// is Messages). A path in no row (a BYO key's files or models call) falls back to
/// [`route::Endpoint::of_upstream_path`], what a catalog walk applies per candidate.
fn wire_of_forward_path(path_and_query: &str) -> Dialect {
    match route::forward_endpoint(path_and_query) {
        Some(e) => e.endpoint.wire(),
        None => route::Endpoint::of_upstream_path(
            path_and_query
                .split_once('?')
                .map_or(path_and_query, |(p, _)| p),
        )
        .wire(),
    }
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
    route::endpoint_of_path(forward_path)
        .is_some_and(|e| e.endpoint == route::Endpoint::ChatCompletions && e.sub.is_none())
}

/// Overwrite dialect and the body plan from **this** catalog candidate's path. Injection follows
/// the upstream candidate, not the client: a Messages body must not grow `stream_options`, and a
/// Chat Completions failover of a Messages client must.
fn apply_serving_candidate(rc: &mut RequestCtx) {
    let Some(c) = rc.auto.as_ref().and_then(|a| a.candidate_at(a.candidate)) else {
        return;
    };
    rc.dialect = route::Endpoint::of_upstream_path(c.path).wire();
    rc.plan = BodyPlan::new(
        stream_options_eligible(rc.managed(), rc.dialect, c.path),
        false,
        true,
        rc.signed.is_some(),
    );
}

/// Whether a body sent on `wire` to `forward_path` (query allowed) must ask for its usage chunk:
/// a managed OpenAI Chat Completions request, which OpenAI meters only when it streams with
/// `stream_options.include_usage`. The one copy of the rule, for a `/{provider}` route, a catalog
/// walk's first arm, and each candidate it then serves from. BYO and every other endpoint stream
/// straight through. The query is stripped here, so no caller can hand `is_streamable_path` one
/// (Azure's `?api-version=` would silently disable injection).
fn stream_options_eligible(managed: bool, wire: Dialect, forward_path: &str) -> bool {
    managed
        && wire == Dialect::OpenAi
        && is_streamable_path(
            forward_path
                .split_once('?')
                .map_or(forward_path, |(p, _)| p),
        )
}

fn catalog_serving_endpoint(auto: &ModelRouting) -> Option<route::Endpoint> {
    auto.candidate_at(auto.candidate)
        .map(|c| route::Endpoint::of_upstream_path(c.path))
}

/// Whether this attempt, translated onto candidate `c`, asks it for Anthropic fast mode: the client
/// asked for priority processing and `c` serves fast mode (D266). One decision for the attempt's
/// `anthropic-beta` header and its translated body, so the two never disagree.
fn translated_fast(auto: &ModelRouting, c: &route::Candidate) -> bool {
    auto.priority && providers::catalog::serves_fast_mode(c)
}

/// Add `beta` to the request's `anthropic-beta` value, unless it is already there.
fn merge_anthropic_beta(req: &mut pingora::http::RequestHeader, beta: &'static str) -> Result<()> {
    let existing = req
        .headers
        .get("anthropic-beta")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty());
    match existing {
        Some(v) if v.split(',').any(|b| b.trim() == beta) => {}
        Some(v) => {
            let merged = format!("{v},{beta}");
            req.insert_header("anthropic-beta", merged)?;
        }
        None => req.insert_header("anthropic-beta", beta)?,
    }
    Ok(())
}

fn catalog_translating(auto: &ModelRouting) -> bool {
    let Some(t) = auto.translate.as_ref() else {
        return false;
    };
    catalog_serving_endpoint(auto).is_some_and(|up| t.client != up)
}

/// This attempt asked a stream-only candidate for a stream its client did not ask for, and
/// assembles it into one body (D147; see `translate::SseBridge::assembling`).
fn catalog_assembling(auto: &ModelRouting) -> bool {
    auto.translate.as_ref().is_some_and(|t| t.assemble)
}

/// A non-stream error from a same-endpoint candidate, buffered (it is small, and capped by
/// `MAX_TRANSLATE_BUFFER`) so a vendor's own error shape reaches the client in its API's envelope
/// (`translate::response_json_tools`, D100). One already in that envelope is relayed unchanged.
fn catalog_error_relay(auto: &ModelRouting, status: Option<u16>, streaming: bool) -> bool {
    !streaming
        && status.is_some_and(|s| s >= 400)
        && auto.translate.is_some()
        && !catalog_translating(auto)
}

/// A catalog walk (not a sub-resource) served by a Chat Completions candidate of a vendor other
/// than OpenAI, whose stream may repeat what OpenAI's sends once (see `translate::ChatIdentity`).
/// It decides only for a Chat Completions client: any other is translated off that candidate
/// anyway (`catalog_translating`), which builds the same bridge.
fn catalog_chat_relay(auto: &ModelRouting) -> bool {
    auto.translate.is_some()
        && auto.candidate_at(auto.candidate).is_some_and(|c| {
            c.provider != providers::ProviderId::OpenAi
                && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::ChatCompletions
        })
}

/// OpenRouter's context compression is on by default for every endpoint of 8K context or less: a
/// prompt over the window has its middle dropped and is answered, where every other host (and
/// OpenRouter with compression off) answers the context error. The gateway relays what the client
/// sent or a context error, never an answer to a prompt it did not send, so a catalog walk to an
/// OpenRouter Chat Completions candidate turns it off, the way OpenRouter documents
/// (`plugins: [{"id": "context-compression", "enabled": false}]`).
const OPENROUTER_NO_COMPRESSION: &[u8] =
    br#""plugins":[{"id":"context-compression","enabled":false}],"#;

/// An OpenRouter Chat Completions catalog candidate.
fn openrouter_chat(c: &route::Candidate) -> bool {
    c.provider == providers::ProviderId::OpenRouter
        && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::ChatCompletions
}

/// Splice [`OPENROUTER_NO_COMPRESSION`] just inside the root object. A body that already names
/// `plugins` is the client's choice and is left as sent, as is anything that is not a non-empty
/// JSON object (the provider rejects it by name).
fn disable_openrouter_compression(mut body: Vec<u8>) -> Vec<u8> {
    if memchr::memmem::find(&body, br#""plugins""#).is_some() {
        return body;
    }
    let Some(open) = body.iter().position(|b| !b.is_ascii_whitespace()) else {
        return body;
    };
    let next = body
        .get(open + 1..)
        .and_then(|rest| rest.iter().find(|b| !b.is_ascii_whitespace()));
    if body.get(open) != Some(&b'{') || matches!(next, None | Some(b'}')) {
        return body;
    }
    body.splice(
        open + 1..open + 1,
        OPENROUTER_NO_COMPRESSION.iter().copied(),
    );
    body
}

/// The fragment spliced into a streaming OpenAI chat body. Always followed by a comma, since the
/// splice point is just inside a root object that is non-empty by construction (a root `"stream"`
/// key is what made it eligible).
const STREAM_OPTIONS_FRAG: &[u8] = br#""stream_options":{"include_usage":true},"#;

/// Cap each root-level output limit (`peek::OUTPUT_LIMIT_KEYS`) at `max`, the serving row's
/// published maximum (`ModelCard::output_cap`), in place. Never raises one; `max == 0` (no
/// published maximum, or an embeddings row) caps nothing.
/// `true` when a value changed, so the caller re-scans the moved bytes.
fn clamp_output_limits(body: &mut Vec<u8>, spans: &[Option<(usize, usize)>; 3], max: u32) -> bool {
    if max == 0 {
        return false;
    }
    // Back to front, so an earlier span's offsets survive a later splice.
    let mut spans = *spans;
    spans.sort_unstable_by(|a, b| b.cmp(a));
    let mut changed = false;
    for (start, end) in spans.into_iter().flatten() {
        let Some(digits) = body.get(start..end) else {
            continue;
        };
        // Digits only (the scan's own span), so a parse fails only past `u64`: over any cap.
        let over = std::str::from_utf8(digits)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .is_none_or(|n| n > u64::from(max));
        if over {
            body.splice(start..end, max.to_string().into_bytes());
            changed = true;
        }
    }
    changed
}

/// Whether a catalog candidate is native OpenAI Chat Completions, which takes the output limit as
/// `max_completion_tokens` on every model and rejects `max_tokens` on its reasoning models.
fn native_openai_chat(c: &route::Candidate) -> bool {
    c.provider == providers::ProviderId::OpenAi
        && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::ChatCompletions
}

/// Respell a root `max_tokens` as `max_completion_tokens`, in place. `keys` is
/// [`peek::BufferedScan::limit_keys`]. When both are present the explicit `max_completion_tokens`
/// wins and the `max_tokens` member is removed. `true` when the body changed, so the caller
/// re-scans the moved bytes.
fn rename_max_tokens(body: &mut Vec<u8>, keys: &[Option<usize>; 3]) -> bool {
    const KEY: &[u8] = br#""max_tokens""#;
    let Some(at) = keys[0] else {
        return false;
    };
    if body.get(at..at + KEY.len()) != Some(KEY) {
        return false;
    }
    if keys[1].is_none() {
        body.splice(
            at + 1..at + KEY.len() - 1,
            b"max_completion_tokens".iter().copied(),
        );
        return true;
    }
    // Both spellings. Find the end of the `max_tokens` value without building it, then remove
    // the member and one adjacent comma.
    let skip_ws = |mut i: usize| {
        while body.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        i
    };
    let colon = skip_ws(at + KEY.len());
    if body.get(colon) != Some(&b':') {
        return false;
    }
    let value = skip_ws(colon + 1);
    let mut values =
        serde_json::Deserializer::from_slice(&body[value..]).into_iter::<serde::de::IgnoredAny>();
    if !matches!(values.next(), Some(Ok(_))) {
        return false;
    }
    let end = value + values.byte_offset();
    let after = skip_ws(end);
    let range = if body.get(after) == Some(&b',') {
        at..after + 1
    } else {
        // The last member: take the comma before it instead.
        match body[..at].iter().rposition(|b| !b.is_ascii_whitespace()) {
            Some(comma) if body[comma] == b',' => comma..end,
            _ => at..end,
        }
    };
    body.drain(range);
    true
}

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

/// Make a client-sent `stream_options` (value at `at`) ask for usage: `include_usage: false` becomes
/// `true`, an object without it gains it, and a non-object value is replaced by
/// `{"include_usage":true}`. Billing is not the client's to switch off — without the usage chunk the
/// row is an estimate that cannot see hidden reasoning. The client may now receive one extra chunk it
/// did not ask for: OpenAI's usage chunk, `choices: []`, which every SDK already accepts.
///
/// Returns `model_span` moved by the edit when it lay after it. Parses only the `stream_options`
/// value (a few bytes), and only on the rare request that sends one; the value is re-serialized, so
/// its other members keep their meaning but not their spacing.
fn force_include_usage(
    mut body: Vec<u8>,
    at: usize,
    model_span: Option<(usize, usize)>,
) -> (Vec<u8>, Option<(usize, usize)>) {
    use serde_json::Value;
    let Some(rest) = body.get(at..) else {
        return (body, model_span);
    };
    let mut values = serde_json::Deserializer::from_slice(rest).into_iter::<Value>();
    let Some(Ok(value)) = values.next() else {
        return (body, model_span);
    };
    let end = at + values.byte_offset();
    let mut options = match value {
        Value::Object(m) if m.get("include_usage") == Some(&Value::Bool(true)) => {
            return (body, model_span);
        }
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    options.insert("include_usage".to_owned(), Value::Bool(true));
    let Ok(replacement) = serde_json::to_vec(&options) else {
        return (body, model_span);
    };
    let (removed, added) = (end - at, replacement.len());
    body.splice(at..end, replacement);
    let shift = |i: usize| i + added - removed;
    let model_span = model_span.map(|(s, e)| if s > at { (shift(s), shift(e)) } else { (s, e) });
    (body, model_span)
}

/// Splice `stream_options.include_usage` into a buffered OpenAI chat body at `at`, or return it
/// unchanged when there is nothing to inject. This is what guarantees a usage chunk — hence a
/// billable token count — from a stock client that never set the option.
///
/// Takes the offset rather than computing it: the caller already walked the body once for both the
/// model and this plan (see `peek::scan_buffered`), and re-deriving it here would restore the second
/// traversal that walk exists to remove.
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

/// What the first path segment resolved to, before the key says whether the request may walk the
/// catalog.
///
/// Holds only routes that go somewhere: a path that resolves to nothing is a [`RouteMiss`], so no
/// later step has to wonder whether it is looking at one.
enum Route {
    /// `/{provider}/…`. The provider is named by the request. Not a catalog walk — the escape hatch.
    Provider {
        provider: &'static Arc<Provider>,
        /// The inbound path with the `/{provider}` segment stripped, query kept.
        forward_path: String,
    },
    /// Bare `/v1` default. BYO keeps the dialect-picked provider; managed becomes a catalog walk.
    BareDefault { provider: &'static Arc<Provider> },
    /// `/auto/…`. `header` is the catalog row if `x-beyond-model` resolved; `None` means peek the
    /// body's root `model` after identity is known. An *unknown* header is [`RouteMiss::UnknownModel`],
    /// not this — the header wins, and a catalog miss is the allowlist.
    Auto {
        header: Option<&'static route::ModelRoute>,
    },
}

/// A path that resolves to nothing the gateway serves. Each shape gets its own rejection (404
/// unknown provider, 404 unknown model, 400 ambiguous credential), which is why it is a sum type and
/// not an `Option`.
enum RouteMiss {
    /// A routing header (on `/auto`) naming a model we do not serve. Collapsed with a missing body
    /// model into one 404: a value we cannot match is a value we do not serve.
    UnknownModel,
    /// The first segment matches no provider, is not the bare default, and is not `/auto`.
    UnknownProvider,
    /// Bare `/v1` with BYO keys for two different providers (D82): a 400, never a guess.
    AmbiguousCredential,
}

/// What `request_filter` decided a request is, once its key is known: carried whole through
/// admission, so the route is never re-derived from which of several locals happen to be set.
enum Plan {
    /// `/{provider}/…`, or a BYO bare `/v1` on its dialect-default provider.
    Provider {
        provider: &'static Arc<Provider>,
        /// [`ForwardPath::Stripped`] on `/{provider}/…`, [`ForwardPath::Inbound`] on the bare
        /// default.
        forward: ForwardPath,
    },
    /// A managed catalog walk (`/auto`, or a managed bare `/v1`) over a resolved row. Only a
    /// verified key gets one, so it carries the identity it was admitted on.
    Catalog(Walk),
}

/// A request on its way to a [`Plan`] once its key is known: ready, or a managed catalog walk that
/// still has to read its row from `x-beyond-model` or the body's root `model`.
enum Admission {
    Ready(Plan),
    NeedsRow(Identity),
}

/// A catalog walk as `request_filter` admits it.
struct Walk {
    id: Identity,
    row: &'static route::ModelRoute,
    /// The body, read in full, when it outgrew pingora's retry buffer: the request is re-run as a
    /// [`FullBody`] subrequest carrying it rather than proxied here.
    relay: Option<Vec<u8>>,
}

impl Plan {
    /// The catalog row this request walks; `None` on a provider route.
    fn row(&self) -> Option<&'static route::ModelRoute> {
        match self {
            Plan::Catalog(w) => Some(w.row),
            Plan::Provider { .. } => None,
        }
    }

    /// The forwarded `/{provider}` path (query included), when the client named the provider.
    fn stripped(&self) -> Option<&str> {
        match self {
            Plan::Provider {
                forward: ForwardPath::Stripped(p),
                ..
            } => Some(p),
            _ => None,
        }
    }
}

/// What path an attempt sends upstream.
#[derive(Debug)]
enum ForwardPath {
    /// The inbound path and query, as the client sent them (the bare `/v1` default). The common
    /// route builds nothing, so it allocates nothing. On a managed request its `key` params are the
    /// virtual key, stripped as the head goes out.
    Inbound,
    /// The client's path with the `/{provider}` segment stripped, its query kept (minus a managed
    /// request's `key` params). Forwarded **verbatim** — the gateway does no per-provider path
    /// rewriting. Built once, so held as a `Box<str>`: beside `Catalog`'s `String` that keeps the
    /// enum at a `String`'s size.
    Stripped(Box<str>),
    /// The serving candidate's own catalog path (plus a sub-resource's suffix), set as each
    /// attempt dials it; rewritten in place, so a failover allocates nothing after the first.
    /// Taken wholesale from the catalog rather than composed from a mount and the client's suffix:
    /// providers disagree on where an endpoint lives and the disagreement is not a prefix.
    Catalog(String),
}

impl ForwardPath {
    /// The path (and query) to set on the upstream request; `None` keeps the inbound one.
    fn as_str(&self) -> Option<&str> {
        match self {
            ForwardPath::Inbound => None,
            ForwardPath::Stripped(p) => Some(p),
            ForwardPath::Catalog(p) => Some(p),
        }
    }
}

#[async_trait]
impl ProxyHttp for AiProxy {
    type CTX = Ctx;

    fn new_ctx(&self) -> Self::CTX {
        self.state.request_began();
        Ctx {
            rc: None,
            held: Held {
                state: self.state,
                request_id: None,
                in_flight: false,
                active_stream: false,
                tenant: None,
                body_bytes: 0,
                body_exempt: false,
                client_left: AtomicBool::new(false),
                drain: Drain::Off,
            },
        }
    }

    /// Pingora calls this with `DownstreamCache` exactly when it ignores a downstream error to keep
    /// reading the upstream: on this gateway, only for a stream [`drain::arm`]ed it. That is the
    /// client leaving a drainable stream; `response_body_filter` takes it from there. Its warn
    /// line is replaced by `ai_usage_drains_in_flight` / `ai_usage_drains_total`.
    fn suppress_proxy_warn_log(
        &self,
        _session: &Session,
        ctx: &Self::CTX,
        _error: &pingora_core::Error,
        context: ProxyWarnLogContext,
    ) -> bool {
        if matches!(context, ProxyWarnLogContext::DownstreamCache) {
            ctx.held.client_left();
            return true;
        }
        false
    }

    /// The request line pingora prints on its own error lines. Its default prints the path
    /// **with** the query, which is where a Gemini-style `?key=` credential (a virtual key or a
    /// BYO Google key) lives. Logged without the query.
    fn request_summary(&self, session: &Session, _ctx: &Self::CTX) -> String {
        summary_line(session.req_header())
    }

    fn allow_spawning_subrequest(&self, _session: &Session, _ctx: &Self::CTX) -> bool {
        // Needed for `relay_full_body`, which is the only place the gateway spawns one.
        true
    }

    async fn request_filter(&self, session: &mut Session, ctx: &mut Self::CTX) -> Result<bool> {
        // Set only on a subrequest the gateway spawned to re-run a large body (see `FullBody`).
        let full_body = full_body_ctx(session);
        if full_body.is_none() {
            self.state.metrics.requests_total.inc();
            // A client that stops reading is cut off after this long rather than holding its slot,
            // its upstream connection and any breaker permit forever (see the config field).
            let secs = self.state.config.client_write_timeout_secs();
            if secs > 0 {
                session
                    .downstream_session
                    .set_write_timeout(Some(Duration::from_secs(secs)));
            }
            // A client that stops sending its body is bounded the same way: per read, pingora's
            // own timeout on HTTP/1.1 (a no-op on HTTP/2, which `peek_body_model` bounds itself).
            let secs = self.state.config.client_read_timeout_secs();
            session
                .downstream_session
                .set_read_timeout((secs > 0).then(|| Duration::from_secs(secs)));
        }
        let start = Instant::now();
        let deadline = match &full_body {
            Some(fb) => fb.deadline,
            None => crate::deadline::at(start, self.state.config.request_max_secs()),
        };
        // One id per request, generated before any reject path so even a 400/401 carries it (in the
        // log line and the `x-beyond-request-id` header). Moved into `ctx` at the end for the
        // admitted path. Cheap: a counter bump + a short `format!` (see `next_request_id`). A
        // `FullBody` attempt is the parent's request, so it keeps the parent's id and seq.
        let (request_id, request_seq) = match &full_body {
            Some(fb) => (fb.request_id, fb.request_seq),
            None => self.state.next_request_id_seq(),
        };
        ctx.held.request_id = Some(request_id);

        // 1. Route by the **first path segment** = provider; forward the rest of the path verbatim
        // (native passthrough — the gateway holds no per-provider mount knowledge). A path with no
        // provider segment that is exactly `/v1` or starts with `/v1/` (boundary-checked — see
        // `bare_default_provider_name`/`route::is_default_prefix`, not a raw prefix test) is the
        // drop-in default: dialect picks openai/anthropic and the path is forwarded as-is. Anything
        // else → unknown provider (404). We resolve before auth (an unknown route is cheap) and
        // compute owned values inside the block so the session borrow ends before any `&mut session`
        // reject below.
        let route = {
            let req = session.req_header();
            let uri = &req.uri;
            let path = uri.path();
            let query = uri.query();
            // `nth(1)`: `/openai/v1/…` → "openai"; `/v1/…` → "v1"; "/" or "" → "".
            let first = path.split('/').nth(1).unwrap_or("");
            let with_query = |p: &str| match query {
                // Sized exactly, so the `Box<str>` it becomes keeps the allocation.
                Some(q) => [p, "?", q].concat(),
                None => p.to_string(),
            };
            // A config-added provider written with capitals (`MiniMax`) is served at its name, and at
            // `/minimax/…`: the table is lowercase, since config names are lowercased at load. A
            // built-in name is exact (`/OpenAI/…` is no route). The retry runs on a miss only.
            let provider = self.state.provider(first).or_else(|| {
                let lower = first.to_ascii_lowercase();
                (lower != first && providers::by_name(&lower).is_none())
                    .then(|| self.state.provider(&lower))
                    .flatten()
            });
            if let Some(p) = provider {
                // Provider-prefixed: strip the leading `/{first}` segment, forward the remainder.
                // `first` is non-empty here (an empty first segment matches no provider), so this
                // always differs from the inbound path and always needs the URI rewritten.
                let rest = &path[1 + first.len()..];
                let rest = if rest.is_empty() { "/" } else { rest };
                Ok(Route::Provider {
                    provider: p,
                    forward_path: with_query(rest),
                })
            } else if route::is_default_prefix(path) {
                // Bare default: the BYO credential, else the path's dialect, picks the provider;
                // managed traffic becomes a catalog walk after identity. Path is forwarded
                // unchanged for BYO (`None`).
                match byo_credential_dialect(req) {
                    // BYO keys for two providers. A managed key anywhere makes the request managed
                    // (every credential location is stripped), so only BYO is ambiguous.
                    Err(()) if !extract_virtual_key(req).is_some_and(|p| p.managed) => {
                        Err(RouteMiss::AmbiguousCredential)
                    }
                    credential => match bare_default_provider_name(path, credential.ok().flatten())
                        .and_then(|name| self.state.provider(name))
                    {
                        Some(p) => Ok(Route::BareDefault { provider: p }),
                        None => Err(RouteMiss::UnknownProvider),
                    },
                }
            } else if first == route::AUTO_SEGMENT {
                // Model-routed. Reached only after a provider-table miss, so the established routes
                // pay nothing for this arm — and `state::build_providers` refuses to boot with a
                // provider named `auto`, so the miss is guaranteed rather than merely likely.
                // An unknown header 404s here (header wins, catalog miss is the allowlist). An
                // *absent* header waits until after identity so a managed caller can put `model` in
                // the body the way a stock SDK does.
                match catalog_from_header(session) {
                    CatalogHeader::Known(route) => Ok(Route::Auto {
                        header: Some(route),
                    }),
                    CatalogHeader::Absent => Ok(Route::Auto { header: None }),
                    CatalogHeader::Unknown => Err(RouteMiss::UnknownModel),
                }
            } else {
                Err(RouteMiss::UnknownProvider)
            }
        };

        // Catalog resolution needs identity: `/auto` is managed-only, a candidate is only usable if
        // we hold a pool key for it, and managed `/v1` peeks the body only after the key is known
        // (BYO `/v1` stays dialect-default passthrough and must not drain the body). The route is
        // kept whole until then, and becomes a [`Plan`] once the key says which it is.
        let route = match route {
            Ok(route) => route,
            Err(miss) => {
                let refusal = match miss {
                    RouteMiss::UnknownModel => header_catalog_miss(session),
                    RouteMiss::UnknownProvider => Refusal::UnknownProvider,
                    RouteMiss::AmbiguousCredential => Refusal::AmbiguousCredential,
                };
                return self.refuse(session, &request_id, refusal).await;
            }
        };

        // 2. Extract the presented key — a managed virtual key (`bai_v1…`) or a raw BYO provider token.
        let Some(Presented {
            key: raw_key,
            managed: managed_key,
        }) = extract_virtual_key(session.req_header())
        else {
            return self.refuse(session, &request_id, Refusal::MissingKey).await;
        };

        // 3. Rate guardrails (see `ratelimit`), charged on the *raw presented key* **before** any
        // verification or upstream connect. Keying on the credential we already hold (rather than the
        // verified tenant id) is what lets this sit ahead of the Ed25519 verify: a single leaked,
        // runaway, or forged key can't drive unbounded crypto work (per-credential tier), and a flood
        // of distinct random BYO tokens can't drive junk-auth connects to providers from our egress
        // IPs (global BYO tier — managed traffic is exempt, see `ratelimit`). The `check_at` borrow of
        // `raw_key` ends as the call returns, so the `&mut session` reject is free to run on the
        // over-limit path (where `raw_key` is unused afterward).
        if full_body.is_none()
            && let Some(rl) = &self.state.rate_limit
            && let Some(throttled) = rl.check_at(raw_key, managed_key, start)
        {
            let refusal = Refusal::RateLimited(throttled);
            return self.refuse(session, &request_id, refusal).await;
        }

        // 4. Reject oversized bodies up front (Content-Length) so we never buffer a huge upload.
        let declared_len = session
            .req_header()
            .headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<usize>().ok());
        if let Some(len) = declared_len
            && len > MAX_REQUEST_BODY
        {
            return self
                .refuse(session, &request_id, Refusal::BodyTooLarge)
                .await;
        }

        // 5. Identity + key handling. One branch: `bai_v1`/`bai_v2` is fail-closed (prefix match →
        // verify; any verify failure is 401, never BYO). Anything else → BYO: the user's own
        // provider token, passed through unchanged (no Beyond identity, so no deny-set, no
        // allowance, and no per-tenant attribution). A public listener without this split would
        // forward a forged virtual key as BYO — junk-auth egress, and the rate guard already
        // exempted it from the BYO aggregate.
        let identity: Option<Identity> = if managed_key {
            let Ok(identity) = self.state.keyring.verify(raw_key) else {
                return self.refuse(session, &request_id, Refusal::InvalidKey).await;
            };
            // Deny-set: O(1), default-allow. Tenant deny OR (v2) key deny. Tenant wins if both.
            if let Some(reason) = self
                .state
                .deny
                .load()
                .reason_for(identity.tenant_id, identity.key_id)
            {
                // Counted per reason (`Refusal::metric`): an `Unknown` is a control plane writing a
                // reason this build does not know (a deploy ahead of the gateway's), and folded into
                // `deny_fraud` it would spike the fraud signal and mask real fraud.
                return self
                    .refuse(session, &request_id, Refusal::Denied(reason))
                    .await;
            }
            // Allowance: remaining-ok vs exhausted, same WatchedSet shape as deny. Fail-closed
            // while the set has not been read. 402 before cache and before `upstream_peer`.
            if let Some(reason) = self
                .state
                .allowance
                .load()
                .reason_for(identity.tenant_id, identity.key_id)
            {
                let refusal = match reason {
                    crate::allowance::AllowanceReject::Exhausted => Refusal::QuotaExhausted,
                    // Not seeded yet: transient, so a retryable 503 with `Retry-After`, not the
                    // 402 an SDK treats as final ("quota exhausted" above is the final one).
                    crate::allowance::AllowanceReject::Unavailable => Refusal::AllowanceUnavailable,
                };
                return self.refuse(session, &request_id, refusal).await;
            }
            // The actual `Bearer …`/`x-api-key` value is precomputed in the provider registry and
            // applied in `upstream_request_filter`; here we only confirm a pool key exists.
            //
            // Skipped for a catalog walk: there a pool key is a property of each *candidate*,
            // and the first one lacking a key is a reason to try the next, not to fail the request.
            // The equivalent gate is the usable-candidate check below. Also skipped for managed
            // `/v1` before the body peek — the dialect-default provider is not the allowlist.
            if let Route::Provider { provider, .. } = &route
                && !provider.has_pool_key()
            {
                return self.refuse(session, &request_id, Refusal::NoPoolKey).await;
            }
            Some(identity)
        } else {
            None
        };
        let managed = identity.is_some();
        let mut admission = match (route, identity) {
            // A managed `?key=` is the virtual key: never forward it. Catalog walks send the
            // candidate's own path with no query, so only `/{provider}/…` carries one.
            (
                Route::Provider {
                    provider,
                    forward_path,
                },
                _,
            ) => Admission::Ready(Plan::Provider {
                provider,
                forward: ForwardPath::Stripped(
                    match managed {
                        true => strip_key_param(&forward_path).unwrap_or(forward_path),
                        false => forward_path,
                    }
                    .into_boxed_str(),
                ),
            }),
            (Route::BareDefault { provider }, None) => Admission::Ready(Plan::Provider {
                provider,
                forward: ForwardPath::Inbound,
            }),
            // Catalog walk is **managed-only**. BYO on `/auto` 400s before any peek — a BYO token
            // belongs to one vendor, and reading the body to pick among candidates would still be
            // a guess (and a failover would send that key to a different vendor). BYO on `/v1`
            // keeps the dialect-default provider (above) and never consults the catalog.
            (Route::Auto { .. }, None) => {
                return self
                    .refuse(session, &request_id, Refusal::ByoOnModelRoute)
                    .await;
            }
            (Route::Auto { header: Some(row) }, Some(id)) => {
                Admission::Ready(Plan::Catalog(Walk {
                    id,
                    row,
                    relay: None,
                }))
            }
            (Route::BareDefault { .. } | Route::Auto { header: None }, Some(id)) => {
                Admission::NeedsRow(id)
            }
        };

        // Stock SDKs list models at GET /v1/models. A managed caller is answered here, from the
        // catalog rows this deployment's pool keys can serve (built once at boot), so an empty GET
        // is not a missing-model 404. A BYO caller never uses the catalog: like any BYO `/v1`
        // request it relays to the provider its key belongs to, which answers with its own list.
        if managed && is_v1_models_list(session) {
            let body = self.state.models_list.clone();
            return Self::reply_models_list_boxed(session, &request_id, body).await;
        }

        // A managed key spends Beyond's shared pool key, so it reaches only metered generation
        // calls: POST, and on `/{provider}/…` only a generation endpoint. Anything else (listing or
        // reading stored files and responses, batches, fine-tuning, a GET relayed to a generation
        // path) would let one tenant reach another's data on the shared key and run unmetered.
        // BYO keys are the caller's own and pass through untouched.
        if managed && full_body.is_none() {
            let req = session.req_header();
            // A protocol upgrade (WebSocket: OpenAI Realtime, Codex) would turn the request into an
            // opaque relay on the pool key that no usage tap can meter. Refused whatever the path,
            // so the allowlist below is not the only thing standing in its way.
            if req.headers.contains_key(http::header::UPGRADE) {
                return self
                    .refuse(session, &request_id, Refusal::ProtocolUpgrade)
                    .await;
            }
            let method_ok = req.method == http::Method::POST;
            // Only a request that names its provider is held to the allowlist; a catalog walk's
            // paths are the catalog's own.
            let endpoint_ok = match &admission {
                Admission::Ready(plan) => plan.stripped().is_none_or(|p| {
                    is_managed_provider_endpoint(p.split_once('?').map_or(p, |(path, _)| path))
                }),
                Admission::NeedsRow(_) => true,
            };
            if !(method_ok && endpoint_ok) {
                // An endpoint outside the allowlist is 404 whatever the method; a wrong method on an
                // allowed endpoint (or any catalog path) is 405.
                let refusal = Refusal::ManagedEndpoint {
                    method: req.method.clone(),
                    path: clip_catalog_name(req.uri.path()).into(),
                    wrong_method: endpoint_ok,
                };
                return self.refuse(session, &request_id, refusal).await;
            }
        }

        let mut body_complete: Option<Vec<u8>> = None;
        let cache_bypass = !managed || cache::request_bypasses(session.req_header());
        // A `FullBody` re-run walks the row its parent chose.
        if let Some(fb) = &full_body {
            admission = match admission {
                Admission::NeedsRow(id) => Admission::Ready(Plan::Catalog(Walk {
                    id,
                    row: fb.route,
                    relay: None,
                })),
                Admission::Ready(Plan::Catalog(w)) => {
                    Admission::Ready(Plan::Catalog(Walk { row: fb.route, ..w }))
                }
                ready => ready,
            };
        }
        // A headerless catalog walk reads the whole body before choosing a row, rather than
        // stopping at `model`: a large body can only fail over once fully read (see `FullBody`), a
        // small one fits pingora's replay buffer either way and costs nothing more, and with the
        // whole body in hand the gateway can hash it for the cache and read Responses session
        // fields.
        let responses = route::is_responses_path(session.req_header().uri.path());
        let large = declared_len.is_none_or(past_replay_buffer);
        let mut peeked = false;
        // The tenant slot, taken before any full read so `tenant_max_in_flight` bounds the bodies held
        // in memory and not only the requests in flight; moved into the request context below.
        let mut early_slot: Option<TenantSlot<'static>> = None;
        let mut plan = match admission {
            Admission::Ready(plan) => plan,
            Admission::NeedsRow(id) => {
                // Header wins if present (unknown → 404, no fall-through to the body). Absent → peek.
                let (row, relay) = match catalog_from_header(session) {
                    CatalogHeader::Known(route) => (route, None),
                    CatalogHeader::Unknown => {
                        let refusal = header_catalog_miss(session);
                        return self.refuse(session, &request_id, refusal).await;
                    }
                    CatalogHeader::Absent => {
                        if full_body.is_none() {
                            match self.take_slot(&id) {
                                Ok(guard) => early_slot = guard,
                                Err(()) => {
                                    let refusal = Refusal::TenantBusy;
                                    return self.refuse(session, &request_id, refusal).await;
                                }
                            }
                        }
                        // A large body is held twice (here and in its `FullBody` re-run): reserve
                        // both before reading a byte of it.
                        if let Some(n) = declared_len.filter(|&n| past_replay_buffer(n))
                            && !ctx.held.reserve_body(n.saturating_mul(2))
                        {
                            return self.refuse(session, &request_id, Refusal::BodyMemory).await;
                        }
                        let reserve = declared_len.unwrap_or(0).min(MAX_REQUEST_BODY);
                        let peek = Box::pin(peek_body_model(
                            session,
                            reserve,
                            &mut ctx.held,
                            self.up_front_read_timeout(session),
                            deadline,
                        ))
                        .await?;
                        peeked = true;
                        if peek.over_cap {
                            return self
                                .refuse(session, &request_id, Refusal::BodyTooLarge)
                                .await;
                        }
                        if peek.over_budget {
                            return self.refuse(session, &request_id, Refusal::BodyMemory).await;
                        }
                        let Some(name) = peek.model.filter(|n| !n.is_empty()) else {
                            let refusal = Refusal::CatalogMiss {
                                name: None,
                                header: true,
                            };
                            return self.refuse(session, &request_id, refusal).await;
                        };
                        let Some(route) = route::model_route(&name) else {
                            let refusal = Refusal::CatalogMiss {
                                name: Some(name.into()),
                                header: true,
                            };
                            return self.refuse(session, &request_id, refusal).await;
                        };
                        if peek.relay {
                            (route, Some(peek.complete.unwrap_or_default()))
                        } else {
                            body_complete = peek.complete;
                            (route, None)
                        }
                    }
                };
                Plan::Catalog(Walk { id, row, relay })
            }
        };

        // Header-won catalog walks skip the body peek above. The same reasons to read the whole
        // body apply (Responses session state, a body pingora could not replay for failover).
        if let Plan::Catalog(w) = &mut plan
            && full_body.is_none()
            && !peeked
            && (responses
                || large
                || route::walk_reads_body(w.row)
                || route::walk_reads_tools(w.row, session.req_header().uri.path()))
        {
            match self.take_slot(&w.id) {
                Ok(guard) => early_slot = guard,
                Err(()) => return self.refuse(session, &request_id, Refusal::TenantBusy).await,
            }
            if let Some(n) = declared_len.filter(|&n| past_replay_buffer(n))
                && !ctx.held.reserve_body(n.saturating_mul(2))
            {
                return self.refuse(session, &request_id, Refusal::BodyMemory).await;
            }
            let reserve = declared_len.unwrap_or(0).min(MAX_REQUEST_BODY);
            let peek = Box::pin(peek_body_model(
                session,
                reserve,
                &mut ctx.held,
                self.up_front_read_timeout(session),
                deadline,
            ))
            .await?;
            if peek.over_cap {
                return self
                    .refuse(session, &request_id, Refusal::BodyTooLarge)
                    .await;
            }
            if peek.over_budget {
                return self.refuse(session, &request_id, Refusal::BodyMemory).await;
            }
            if peek.relay {
                w.relay = Some(peek.complete.unwrap_or_default());
            } else {
                body_complete = peek.complete;
            }
        }

        // A body read in full past pingora's retry buffer: re-run the request as a `FullBody`
        // subrequest that carries it, rather than proxy it here.
        if let Plan::Catalog(w) = &mut plan
            && let Some(body) = w.relay.take()
        {
            // --- signed ids (`signed_id`) ---
            let path = session.req_header().uri.path();
            if let Some(write) = self
                .foreign_ids(path, w.row, &body, w.id.tenant_id)
                .map(|refusal| self.refuse(session, &request_id, refusal))
            {
                return write.await;
            }
            let slot_held = early_slot.is_some();
            let relayed = self
                .relay_full_body(
                    session,
                    Parent {
                        request_id,
                        request_seq,
                        deadline,
                    },
                    w.row,
                    body,
                    slot_held,
                )
                .await;
            drop(early_slot);
            return relayed;
        }
        let model_route = plan.row();

        // Two root `model` keys (D34), refused before connecting wherever the body is already in
        // hand, with a JSON 400 and the request id (D94). A body streamed through is still refused
        // in `request_body_filter`, after the request headers went upstream.
        // One walk of the body serves both checks below.
        let scan = body_complete
            .as_deref()
            .filter(|_| model_route.is_some())
            .map(peek::scan_buffered);
        // The body rules (`body_rules`) that read its shape are checked here, as soon as it is
        // scanned: a second `model`, and an output limit that is not a whole number on a row with a
        // published cap. What the row accepts is checked once the walk is shaped, below.
        let shape = body_rules::Facts {
            duplicate_model: scan.as_ref().is_some_and(|s| s.duplicate_model),
            inexact_limit: model_route.is_some_and(|r| r.card.output_cap().is_some())
                && scan.as_ref().is_some_and(|s| s.inexact_limit),
            ..body_rules::Facts::default()
        };
        if let Some(write) = body_rules::refusal(body_rules::Point::Connect, shape)
            .map(|refusal| self.refuse(session, &request_id, refusal))
        {
            return write.await;
        }

        // A `FullBody` re-run's body is its parent's, already reserved for both copies.
        ctx.held.body_exempt = full_body.is_some();

        // Per-request control surface (`x-beyond-*`). Managed only: a BYO request carries no verified
        // identity, so a tag on it would be an unattributable row and a capture would be storing
        // prompts we can't attribute to an account that asked us to — the same reason `ai.usage`
        // itself is managed-only.
        //
        // Parsed here, before the candidate walk, so `order` / `only` / `split` can permute the
        // list `first_usable` sees. Nothing here can reject the request: `Control::parse` drops
        // what it can't use and counts it. See `control`'s module docs.
        let parsed_control = if managed {
            let parsed = control::Control::parse(session.req_header());
            if parsed.malformed {
                self.state.metrics.control_header_errors_total.inc();
            }
            if parsed.unknown_provider {
                self.state.metrics.control_unknown_provider_total.inc();
            }
            Some(parsed)
        } else {
            None
        };

        // Model routing is **managed-only**, and the first candidate is chosen here.
        let mut walk = control::Walk::identity(0);
        let mut walk_arms: &'static [route::Candidate] = &[];
        // The client asked for priority processing (`route::SpeedAsk::Priority`, D266).
        let mut priority = false;
        let inbound_responses =
            model_route.is_some() && route::is_responses_path(session.req_header().uri.path());
        let session_field = match &full_body {
            Some(fb) => fb.session_field,
            None if inbound_responses => translate::responses_session_field(
                body_complete.as_deref().unwrap_or(&[]),
                model_route.is_some_and(|r| !r.responses.is_empty()),
            ),
            None => None,
        };
        // `/v1/messages/count_tokens`, `/v1/responses/compact`, … (see `route::SubResource`).
        let sub = model_route.and(route::SubResource::of_path(session.req_header().uri.path()));
        let (provider, usable) = match &plan {
            // Provider-routed, or BYO `/v1` dialect default.
            Plan::Provider { provider, .. } => (Arc::clone(provider), 0u8),
            // Only a managed key walks the catalog (a BYO token belongs to exactly one provider, so
            // selecting among candidates would be a guess, and failing over would send its key to
            // another vendor): BYO was refused before a `Walk` could be made.
            Plan::Catalog(w) => {
                let (row, id) = (w.row, w.id);
                // The rest of the body rules, after the control surface is parsed (and counted)
                // above. `shape` passed, so the order is still `body_rules::Rule::ORDER`.
                let facts = body_rules::Facts {
                    refused_input: body_complete
                        .as_deref()
                        .and_then(|b| route::refused_input(row, b))
                        .map(|kind| (row.model, kind)),
                    background: inbound_responses
                        && body_complete.as_deref().is_some_and(requests_background),
                    ..shape
                };
                if let Some(write) = body_rules::refusal(body_rules::Point::Connect, facts)
                    .map(|refusal| self.refuse(session, &request_id, refusal))
                {
                    return write.await;
                }
                // A Responses request walks the row's Responses arm whenever the row has one,
                // `store: false` one-shots included: there it is a byte relay, and translation onto
                // Chat Completions would lose what only Responses has (Codex's `namespace` tools,
                // `custom` grammars, encrypted reasoning). Only a row without an arm translates,
                // and only a one-shot may (session state there is the 400 below). A failover stays
                // inside the arm it walks, in catalog order: the session pin orders `candidates`
                // only.
                // An embeddings row has no Responses arm either, but "store cannot be honored"
                // would name a field the caller may never have set: it walks its candidates and the
                // wire check below rejects the endpoint, which is what is actually wrong.
                let embeddings_row = route::Endpoint::of_row(row) == route::Endpoint::Embeddings;
                // A sub-resource walks the arms that hold its provider's parent endpoint: OpenAI's
                // `/v1/responses` sits in a GPT row's Responses arm, or first in a Responses-first
                // row's candidates.
                let arms: &'static [route::Candidate] = match sub {
                    Some(route::SubResource::InputTokens | route::SubResource::Compact)
                        if !row.responses.is_empty() =>
                    {
                        row.responses
                    }
                    Some(_) => row.candidates,
                    None if inbound_responses
                        && !embeddings_row
                        && (session_field.is_some() || !row.responses.is_empty()) =>
                    {
                        row.responses
                    }
                    // More tools than Chat Completions takes, from a translated client: the
                    // Responses arm takes them (`route::tools_need_responses_arm`, D131).
                    None if match &full_body {
                        Some(fb) => fb.responses_tools,
                        None => body_complete.as_deref().is_some_and(|b| {
                            route::tools_need_responses_arm(
                                row,
                                route::implied_endpoint(session.req_header().uri.path()),
                                b,
                            )
                        }),
                    } =>
                    {
                        row.responses
                    }
                    None => row.candidates,
                };
                if inbound_responses
                    && let Some(field) = session_field
                    && arms.is_empty()
                {
                    let refusal = Refusal::SessionField {
                        field,
                        model: row.model,
                    };
                    return self.refuse(session, &request_id, refusal).await;
                }
                walk_arms = arms;
                // Permute the row before the usable mask / `first_usable`. Failover, breakers, and
                // the 429 key-walk then see this sequence. Unknown names were already dropped;
                // an `only` filter that left nobody is the same 503 as an unkeyed row.
                //
                // `order` / `split` fix the walk; otherwise the caller's computed session pin orders
                // it (`pin`). `only` filters, then the pin applies to what is left. Responses walks
                // and sub-resources keep catalog order.
                walk = parsed_control.as_ref().map_or_else(
                    || control::Walk::identity(arms.len()),
                    |c| c.catalog_walk(arms, request_seq),
                );
                // Bit `orig` ⇒ `arms[orig]` can be sent to: registered here with a pool key, not
                // skipped by a large-body re-run or held off by a key walk, and serving this
                // sub-resource. Computed before the session pin, which moves what cannot take the
                // request behind what can; the usable mask below is this, by walk slot.
                let mut dispatchable = 0u8;
                // Bit `orig` ⇒ every pool key of that arm's provider is cooling off from a refusal
                // (a 401, an out-of-credit answer): left out below while another arm can serve.
                let mut cooling = 0u8;
                // Each arm's provider, as resolved here: a dispatchable arm has one.
                let mut resolved: [Option<&'static Arc<Provider>>; route::MAX_CANDIDATES] =
                    [None; route::MAX_CANDIDATES];
                for (orig, c) in arms.iter().enumerate().take(route::MAX_CANDIDATES) {
                    let provider = self.state.provider_by_id(c.provider);
                    resolved[orig] = provider;
                    let orig = orig as u8;
                    let keyed = provider.is_some_and(|p| p.has_pool_key());
                    if provider.is_some_and(|p| p.cooling()) {
                        cooling |= 1 << orig;
                    }
                    // A re-run of a large body skips the candidates earlier attempts failed on. A
                    // key walk in progress is not a filter: its candidate goes first below
                    // (`FullBody::resume`), and the rest stay for failover (D81).
                    let failed = full_body
                        .as_ref()
                        .is_some_and(|fb| fb.skip & (1 << orig) != 0);
                    let serves = sub.is_none_or(|sub| sub.serves(c));
                    if keyed && !failed && serves {
                        dispatchable |= 1 << orig;
                    }
                }
                // A candidate that cannot serve this body (`route::unserved`: Bedrock and a
                // JSON-schema output) is not dispatched to or pinned onto, unless nothing else
                // can take the request, when the provider's own error is the answer.
                let unserved = match (&full_body, body_complete.as_deref()) {
                    // The parent computed it against the row's candidates.
                    (Some(fb), _) if std::ptr::eq(arms, row.candidates) => fb.unserved,
                    (Some(_), _) => 0,
                    (None, Some(b)) => route::unserved(arms, b),
                    (None, None) => 0,
                };
                // Anthropic fast mode (D266). A Messages client's `speed: "fast"` is served fast or
                // refused, as Anthropic itself does: only a candidate that serves fast mode may
                // take it, and a walk with none is a 400 here rather than an answer at standard
                // speed (or Bedrock's own refusal) under a fast request. A Chat Completions or
                // Responses client's `service_tier: "priority"` is a best effort and moves nothing:
                // it is mapped to fast mode on whichever attempt reaches a candidate that serves it.
                if sub.is_none() {
                    let scanned = match &full_body {
                        Some(fb) => Some(&fb.body[..]),
                        None => body_complete.as_deref(),
                    };
                    let client = route::implied_endpoint(session.req_header().uri.path());
                    match scanned.map(|b| route::speed_ask(client, b)) {
                        Some(route::SpeedAsk::Fast) => {
                            let slow = route::fast_unserved(arms);
                            if walk.mask() & !slow == 0 {
                                let refusal = Refusal::RefusedInput {
                                    model: row.model,
                                    kind: "fast mode",
                                };
                                return self.refuse(session, &request_id, refusal).await;
                            }
                            dispatchable &= !slow;
                        }
                        Some(route::SpeedAsk::Priority) => priority = true,
                        _ => {}
                    }
                }
                let walked = walk.mask();
                if dispatchable & walked & !unserved != 0 {
                    dispatchable &= !unserved;
                }
                // A provider whose every key was refused within `KEY_COOLDOWN` (revoked, or out of
                // credit: D180) is not sent to while another candidate can take the request; when
                // none can, it is tried anyway, and its own answer is the client's.
                if dispatchable & walked & !cooling != 0 {
                    dispatchable &= !cooling;
                }
                if self.state.config.session_pins()
                    && sub.is_none()
                    && std::ptr::eq(arms, row.candidates)
                    && !parsed_control
                        .as_ref()
                        .is_some_and(control::Control::pins_walk)
                {
                    let affinity = pin::affinity(id.tenant_id, id.vpc_id, id.key_id);
                    if let Some(pinned) = pin::order(walk, row, affinity, dispatchable) {
                        walk = pinned;
                        self.state.metrics.session_pinned_total.inc();
                    }
                }
                // A large body's re-run resumes its key walk on that candidate first; the rest of
                // the walk stays behind it for failover (see `FullBody::resume`).
                if let Some(orig) = full_body.as_ref().and_then(|fb| fb.resume) {
                    walk_front(&mut walk, orig);
                }
                if walk.len == 0 {
                    return self
                        .refuse(session, &request_id, Refusal::NoCandidate)
                        .await;
                }
                // Bit i ⇒ walk slot i is registered here *and* has a pool key. Computed once; every
                // later attempt reads this instead of re-deriving it.
                let mut usable = 0u8;
                for i in 0..walk.len {
                    if walk
                        .catalog_index(i)
                        .is_some_and(|orig| dispatchable & (1 << orig) != 0)
                    {
                        usable |= 1 << i;
                    }
                }
                if let Some(sub) = sub
                    && !arms.iter().any(|c| sub.serves(c))
                {
                    let refusal = Refusal::NoSubResource {
                        model: row.model,
                        provider: sub.provider_name(),
                        path: session.req_header().uri.path().into(),
                    };
                    return self.refuse(session, &request_id, refusal).await;
                }
                // The first usable slot's provider: a usable slot is a dispatchable arm, which
                // resolved one.
                let first = first_usable(usable, 0)
                    .and_then(|i| walk.catalog_index(i))
                    .and_then(|orig| resolved.get(usize::from(orig)).copied().flatten());
                let Some(first) = first else {
                    // Every remaining candidate is unkeyed. Distinct from `circuit_open`, which
                    // means the candidates exist and are being skipped while they recover —
                    // `doctor`'s `model_catalog` check exists to catch this configuration at boot
                    // instead.
                    return self
                        .refuse(session, &request_id, Refusal::NoCandidate)
                        .await;
                };
                (Arc::clone(first), usable)
            }
        };

        // --- signed ids (`signed_id`): Responses state a provider keeps is tenant-bound ---------
        // A managed Responses relay to a store (a GPT row's Responses arm, `/{provider}/…/responses`)
        // must not carry an id another tenant could resolve. The ids the client sent back are
        // checked here, before connecting, wherever the body is in hand (every catalog walk); a
        // `/{provider}` body streams, and is checked as it is stripped in `request_body_filter`.
        let signed = if let Some(id) = identity
            && match model_route {
                Some(row) => signed_id::catalog_applies(session.req_header().uri.path(), row),
                None => plan
                    .stripped()
                    .is_some_and(signed_id::provider_route_applies),
            } {
            let Some(signer) = self.state.id_signer.as_ref() else {
                return self
                    .refuse(session, &request_id, Refusal::IdSigningUnset)
                    .await;
            };
            if let Some(write) = body_complete
                .as_deref()
                .and_then(|b| {
                    signer
                        .unsign_request(id.tenant_id, b, model_route.is_some())
                        .err()
                })
                .map(|refusal| self.refuse(session, &request_id, Refusal::ForeignId(refusal)))
            {
                return write.await;
            }
            Some(Box::new(signed_id::Relay::new(id.tenant_id)))
        } else {
            None
        };
        // --- end signed ids ---

        // Catalog walk: inbound path may name Chat Completions, Messages, or Responses while the
        // serving *candidate* speaks a different one of those three. Always keep the client
        // endpoint so failover can translate onto the next path; embeddings against a generation
        // row (or the reverse) is a 400. Inbound Responses with session state already chose the Responses arm
        // (or 400'd) — same-endpoint is a byte relay (`from == to`). `/{provider}/…` never
        // reaches this — it has no row.
        let mut translate_state = None;
        // A sub-resource is its provider's own API: no wire check, no translation.
        if let Some(row) = model_route.filter(|_| sub.is_none()) {
            let path = session.req_header().uri.path();
            let row_endpoint = route::Endpoint::of_row(row);
            match route::catalog_wire_action(path, row_endpoint) {
                route::WireAction::Reject => {
                    let refusal = Refusal::WrongWire {
                        model: row.model,
                        hint: row_endpoint.post_hint(),
                    };
                    return self.refuse(session, &request_id, refusal).await;
                }
                route::WireAction::Relay => {
                    translate_state = Some(translate::TranslateState::new(
                        route::implied_endpoint(path).unwrap_or(row_endpoint),
                    ));
                }
                route::WireAction::Translate { client } => {
                    translate_state = Some(translate::TranslateState::new(client));
                }
            }
        }

        // Dialect drives usage parsing and injection eligibility.
        //
        // For a model-routed request it is seeded from the **row** here and overwritten in
        // `upstream_peer` from **this candidate's path**. A provider can serve more than one
        // wire (OpenRouter Chat Completions *and* Messages), and a mixed row can list both, so
        // neither the provider table nor the row's declared wire is the serving dialect.
        // Reading the wrong one hands an Anthropic response to the OpenAI usage extractor,
        // which does not error — it trips the dialect-mismatch guard and emits a **zero-token
        // billing row**.
        //
        // A provider-routed request is the same problem without a catalog: `/{provider}/…`
        // forwards a path, and that path — not the provider — says which wire answers. OpenRouter
        // serves Messages at `/api/v1/messages` and Anthropic serves Chat Completions at
        // `/v1/chat/completions`, so the provider's default dialect metered both as zero.
        let dialect = match model_route {
            Some(r) => r.wire,
            None => plan
                .stripped()
                .map_or(provider.dialect, wire_of_forward_path),
        };
        // Mark OpenAI managed chat/completions streams for body buffering + `stream_options` injection
        // (handled in `request_body_filter`). A catalog walk asks of the arm this request will
        // actually walk, not the row's Chat Completions primary: a Responses walk must not inherit
        // `stream_options` injection (each candidate re-decides in `apply_serving_candidate`).
        let stream_options = match model_route {
            Some(_) => walk_arms
                .first()
                .is_some_and(|c| stream_options_eligible(managed, dialect, c.path)),
            None => plan
                .stripped()
                .is_some_and(|p| stream_options_eligible(managed, dialect, p)),
        };
        // Every billable managed `/{provider}/…` body is buffered too, so a model outside the
        // catalog is refused before a byte of it goes upstream: a managed key runs only what its
        // billing row can price (D267). So is a `/responses` body's `background: true` (a
        // generation the provider runs after the request ends, which no usage tap can meter,
        // D202). The free token counts are left to stream: nothing there bills. A catalog walk
        // checked its body above.
        let catalog_check = managed
            && plan
                .stripped()
                .and_then(route::forward_endpoint)
                .is_some_and(|e| e.sub.is_none_or(route::SubResource::billed));
        // That buffer counts against the body budget. A declared large body reserves up front,
        // before the tenant slot and the breaker permit, so a refusal holds neither; a chunked one
        // reserves as it grows (`request_body_filter`).
        if (stream_options || catalog_check)
            && model_route.is_none()
            && let Some(n) = declared_len.filter(|&n| past_replay_buffer(n))
            && !ctx.held.reserve_body(n)
        {
            return self.refuse(session, &request_id, Refusal::BodyMemory).await;
        }

        // Capture decision from the control surface parsed above. The header wins in **both**
        // directions over the tenant's control-plane rule (Cloudflare's `cf-aig-collect-log`
        // semantics), and the two enablers serve different people: the control plane is the
        // operator's, works retroactively, and needs no cooperation from a client we don't control;
        // the header is the caller's, for per-request precision.
        let control = if let Some(parsed) = parsed_control
            && let Some(id) = identity
        {
            let capture = match parsed.capture {
                // Explicit suppression always wins — "not this one, it has PII".
                Some(false) => None,
                // Explicitly requested, so **never sampled away**: a caller who asks to log one
                // trace and silently gets nothing is the single outcome that makes this useless.
                Some(true) => Some(CaptureBufs::new(self.state.capture_defaults.max_bytes)),
                // No opinion — fall back to the tenant's rule. Sparse miss is ~1.5ns (see `deny`'s
                // hasher rationale, which this set shares) and is what the overwhelming majority of
                // managed requests hit.
                None => self
                    .state
                    .capture
                    .load()
                    .rule_for(id.tenant_id)
                    .filter(|rule| rule.samples(request_seq))
                    .map(|rule| CaptureBufs::new(rule.max_bytes)),
            };

            // Allocate the box only when something is actually on. A managed request that used
            // neither feature — nearly all of them — stays at `None`.
            (parsed.metadata.is_some() || capture.is_some()).then(|| {
                Box::new(RequestControl {
                    metadata: parsed.metadata,
                    capture,
                })
            })
        } else {
            None
        };

        // Exact-match cache: lookup only when the pre-rewrite body is already in hand (headerless
        // managed catalog walk). A hit writes the stored 2xx and returns before the breaker, the
        // key-walk, and `upstream_peer`. A miss stays an unbuffered relay; the fill is a tap.
        let cache_look = if let Some(id) = identity
            && !cache_bypass
            && model_route.is_some()
        {
            body_complete.as_deref().and_then(|body| {
                let store = self.state.cache.as_ref()?;
                let req = session.req_header();
                let (ids, n) = walk.provider_ids(walk_arms);
                let ck = cache::key(&cache::KeyParts {
                    tenant_id: id.tenant_id,
                    method: req.method.as_str(),
                    inbound_path: req.uri.path(),
                    model: model_route.map_or("", |r| r.model),
                    headers: &req.headers,
                    body,
                    providers: &ids[..usize::from(n)],
                });
                match store.get(&ck) {
                    Some(hit) => Some(Err(hit)),
                    None => Some(Ok((ck, store.max_bytes()))),
                }
            })
        } else {
            None
        };
        let forward_path = match plan {
            Plan::Provider { forward, .. } => forward,
            Plan::Catalog(_) => ForwardPath::Inbound,
        };
        let mut pending_cache: Option<cache::Pending> = None;
        match cache_look {
            Some(Err(hit)) => {
                Self::reply_cache_hit_boxed(session, &request_id, &hit).await?;
                ctx.rc = Some(RequestCtx {
                    billing: Billing::of(identity),
                    dialect,
                    leg: Leg::new(provider, None, 0),
                    forward_path,
                    control,
                    model: String::new(),
                    attempt: Attempt::new(Vec::new()),
                    resp_model_scanner: peek::ModelScanner::for_response(),
                    // Not an upstream stream — `response_filter` never ran, so `active_streams`
                    // was never incremented. The stored `hit.streaming` is emitted on `ai.usage`.
                    streaming: false,
                    plan: BodyPlan::new(false, false, model_route.is_some(), false),
                    resp_tail: UsageTail::default(),
                    resp_head: Vec::new(),
                    upstream_status: held_status(hit.status),
                    start,
                    deadline,
                    auto: model_route.map(|route| {
                        Box::new(ModelRouting {
                            route,
                            sub,
                            candidate: first_usable(usable, 0).unwrap_or(0),
                            usable,
                            walk,
                            arms: walk_arms,
                            session_field,
                            attempt_start: start,
                            cache: Some(cache::Pending::Hit(Box::new(hit))),
                            translate: None,
                            health: None,
                            addrs: 0,
                            attempted: false,
                            open_retry_after: None,
                            priority,
                        })
                    }),
                    request_id,
                    tally_eager: false,
                    resp_bytes: 0,
                    upstream_phase: UpstreamPhase::None,
                    redact: None,
                    terminal: TerminalTracker::default(),
                    signed: None,
                });
                ctx.held.admit();
                return Ok(true);
            }
            Some(Ok((ck, max_bytes))) => {
                pending_cache = Some(cache::Pending::Fill {
                    key: ck,
                    tap: cache::ResponseTap::new(max_bytes),
                    content_type: None,
                });
            }
            None => {}
        }

        // Per-tenant in-flight cap — the bound on overspend while the allowance-set lags (see
        // `concurrency`). After the cache (a hit costs no provider spend) and before the breaker,
        // so a refused request never holds a half-open probe permit.
        let parent_holds_slot = full_body.as_ref().is_some_and(|fb| fb.slot_held);
        let slot = match (early_slot, identity) {
            // A `FullBody` attempt whose parent holds the slot: neither take nor release one.
            _ if parent_holds_slot => None,
            // Taken before the body was read.
            (Some(slot), _) => Some(slot),
            (None, Some(id)) => match self.take_slot(&id) {
                Ok(slot) => slot,
                Err(()) => {
                    return self.refuse(session, &request_id, Refusal::TenantBusy).await;
                }
            },
            (None, None) => None,
        };

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
        let breaker_permit = match (&model_route, &provider.breaker) {
            (None, Some(breaker)) => breaker.allow().map(Some).map_err(|_| breaker),
            _ => Ok(None),
        };
        if let Err(breaker) = breaker_permit {
            // Given back before the refusal is written: a slow client must not hold it.
            drop(slot);
            let refusal = Refusal::CircuitOpen {
                retry_after: breaker.retry_after_secs(),
            };
            return self.refuse(session, &request_id, refusal).await;
        }
        // The leg owes this permit until an outcome or a release resolves it (see `Leg`). A
        // model-routed request owes none here: it claims each candidate's in `upstream_peer`.
        let permit = breaker_permit.ok().flatten();
        // Past any key cooling off from a 401 (D71); a catalog walk picks per candidate.
        let pool_key = provider.first_key();
        ctx.held.tenant = slot;

        ctx.rc = Some(RequestCtx {
            billing: Billing::of(identity),
            dialect,
            leg: Leg::new(provider, permit, pool_key),
            forward_path,
            control,
            model: String::new(),
            // Its body buffer is pre-sized below, once the context says whether this request
            // rewrites its body.
            attempt: Attempt::new(Vec::new()),
            // `for_response`, not `new`: a response may carry the model nested under `message`
            // (Anthropic's `message_start`), and a root-only scanner would neither find it nor ever
            // stop looking. See `ModelScanner::for_response`.
            resp_model_scanner: peek::ModelScanner::for_response(),
            streaming: false,
            plan: BodyPlan::new(
                stream_options,
                catalog_check,
                model_route.is_some(),
                signed.is_some(),
            ),
            // Grown lazily by the response tap (`response_body_filter`), not pre-reserved: a
            // non-streaming response — the common case — is a few hundred bytes, so reserving the
            // full 64KB cap up front would waste an allocation on every request to hold ~200B. A
            // long stream grows it geometrically to the bounded 2×cap and compacts; that handful of
            // reallocs is lost in the network noise of a stream we're already relaying chunk by chunk.
            resp_tail: UsageTail::default(),
            // Grown lazily, and only on the one path that needs it (Anthropic SSE) — see
            // `response_body_filter`. Every other response leaves this empty and never allocates.
            resp_head: Vec::new(),
            upstream_status: None,
            start,
            deadline,
            auto: model_route.map(|route| {
                Box::new(ModelRouting {
                    route,
                    sub,
                    // `first_usable` picked this candidate above; `upstream_peer` re-derives it from
                    // here on.
                    candidate: first_usable(usable, 0).unwrap_or(0),
                    usable,
                    walk,
                    arms: walk_arms,
                    session_field,
                    // Overwritten per attempt by `upstream_peer`; seeded so the first attempt is
                    // timed even if it fails before the prologue runs.
                    attempt_start: start,
                    cache: pending_cache,
                    translate: translate_state,
                    health: None,
                    addrs: 0,
                    attempted: false,
                    open_retry_after: None,
                    priority,
                })
            }),
            request_id,
            tally_eager: tally_eager(managed, model_route.is_some(), declared_len),
            resp_bytes: 0,
            upstream_phase: UpstreamPhase::None,
            redact: None,
            terminal: TerminalTracker::default(),
            signed,
        });
        // A body buffered for a rewrite (`BodyPlan::rewrites_body`) is pre-sized from the
        // declared Content-Length, so accumulation is a single allocation instead of a geometric
        // realloc chain; capped at `MAX_REQUEST_BODY` so a lying header can't pre-allocate
        // unbounded memory. Every other request leaves it empty and never buffers.
        //
        // The `+ STREAM_OPTIONS_FRAG.len()` is headroom for the splice, which
        // `apply_stream_usage_injection` performs *in place*: with it the injection never
        // reallocates, so a body arrives, is spliced, and goes upstream on one allocation.
        if let Some(rc) = ctx.rc.as_mut()
            && rc.plan.rewrites_body()
            && let Some(len) = declared_len
        {
            rc.attempt.req_buf =
                Vec::with_capacity(len.min(MAX_REQUEST_BODY) + STREAM_OPTIONS_FRAG.len());
        }
        // Admitted: count it in-flight. Released in `logging`, or by `Ctx`'s drop if a panic
        // skipped `logging`, so the gauge cannot leak. `active_streams` only covers SSE; this
        // covers every request.
        ctx.held.admit();
        self.state.fault_point("request_filter");
        Ok(false)
    }

    async fn upstream_peer(
        &self,
        session: &mut Session,
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
        self.state.fault_point("upstream_peer");

        // Pingora calls this once per attempt, always before a body byte moves, so it is where each
        // attempt starts on a fresh body record.
        rc.begin_attempt();
        let retry = rc.leg.retry.take();

        // Model-routed: this hook owns the candidate walk *and* the breaker ledger.
        //
        // It is the only place `rc.leg` changes, which is what makes double-recording structurally
        // impossible — and it is the only hook that runs on every attempt, including the ones
        // pingora starts without consulting `fail_to_connect` (its default `error_while_proxy`
        // marks a reused-connection failure retryable on its own). A design that recorded in
        // `fail_to_connect` would miss exactly those.
        //
        // Not for the same leg again (see `Retry::SameLeg`): that stays on this provider, keeps
        // its outstanding breaker permit, and must not enter the walk.
        if retry != Some(Retry::SameLeg) && rc.auto.is_some() {
            // The hook that moved the walk on declared the outgoing candidate failed (see
            // `Retry::NextCandidate`): resolve its permit before touching anything else, so
            // `logging` only ever sees the final candidate's. A first attempt owes nothing yet.
            // A permit still pending on any other entry is a retry that did not declare itself:
            // its leg would be released here rather than failed.
            debug_assert!(
                rc.leg.permit.is_none() || retry == Some(Retry::NextCandidate),
                "an undeclared retry entered the walk owing a permit ({retry:?})",
            );
            if retry == Some(Retry::NextCandidate) {
                rc.leg.fail();
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
                    // What the client is told depends on why: every candidate was tried and failed
                    // (502), or every one was skipped because its breaker is open (503, retry once
                    // the soonest one half-opens).
                    let attempted = rc.auto.as_ref().is_some_and(|a| a.attempted);
                    return Err(if attempted {
                        gateway_error(502, "no provider could be reached")
                    } else {
                        gateway_error(503, "provider temporarily unavailable")
                    });
                };
                if let Some(a) = rc.auto.as_mut() {
                    a.candidate = i;
                }

                let Some(candidate) = rc.auto.as_ref().and_then(|a| a.candidate_at(i)) else {
                    rc.advance_candidate(i);
                    continue;
                };
                if rc.auto.as_ref().is_some_and(|a| {
                    a.session_field.is_some() && !route::candidate_path_is_responses(candidate.path)
                }) {
                    // Session state must not walk onto Chat Completions / Messages. The walk is
                    // already filtered; this is the belt if a 5xx retry cursor drifted.
                    rc.advance_candidate(i);
                    continue;
                }
                let Some(p) = self.state.provider_by_id(candidate.provider).cloned() else {
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
                let permit = match p.breaker.as_ref() {
                    None => None,
                    Some(b) => match b.allow() {
                        Ok(permit) => Some(permit),
                        Err(_) => {
                            self.state.metrics.rejection(Rejection::CircuitOpen).inc();
                            if let Some(a) = rc.auto.as_mut() {
                                let secs = u16::try_from(b.retry_after_secs()).unwrap_or(u16::MAX);
                                a.open_retry_after =
                                    Some(a.open_retry_after.map_or(secs, |s| s.min(secs)));
                            }
                            rc.advance_candidate(i);
                            continue;
                        }
                    },
                };
                // New vendor ⇒ that vendor's first key not cooling off (D71). A re-run of a large
                // body resumes the key walk an earlier attempt started on this candidate (see
                // `FullBody`); one with no walk starts past the cooling keys like any other request
                // (D83). A new leg starts at the candidate's first address, with its own
                // refused-stream resend, owing the permit (if this breaker has one to give).
                let pool_key = full_body_ctx(session)
                    .zip(rc.auto.as_ref().and_then(|a| a.walk.catalog_index(i)))
                    .and_then(|(fb, orig)| fb.keys.get(usize::from(orig)).copied())
                    .filter(|&k| k != NO_KEY_WALK)
                    .unwrap_or_else(|| p.first_key());
                rc.leg = Leg::new(p.clone(), permit, pool_key);
                apply_serving_candidate(rc);
                if let Some(a) = rc.auto.as_mut() {
                    a.attempted = true;
                }

                match self.state.resolve(&p.authority, 0).await {
                    Ok((addr, addrs)) => {
                        // Time this attempt from here, so a candidate that burned its connect
                        // timeout does not charge that to whichever provider ends up serving.
                        if let Some(a) = rc.auto.as_mut() {
                            a.attempt_start = Instant::now();
                            a.addrs = u8::try_from(addrs).unwrap_or(u8::MAX);
                        }
                        let sub = rc.auto.as_ref().and_then(|a| a.sub);
                        rc.set_forward_path(candidate.path, sub);
                        rc.upstream_phase = UpstreamPhase::Attempted;
                        let peer = self.build_peer(addr, &p, rc.auto.as_ref().map(|a| a.route));
                        return self.cap_read_at_deadline(rc, peer);
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
                        rc.leg.fail();
                        rc.advance_candidate(i);
                        continue;
                    }
                }
            }
        }

        // A provider-routed attempt, or a walk's same leg again, which is timed from here.
        if let Some(a) = rc.auto.as_mut() {
            a.attempt_start = Instant::now();
        }
        // Resolve via the TTL cache (async, non-blocking) rather than `HttpPeer::new`'s eager
        // blocking `getaddrinfo`. SNI/Host = the configured host; TLS on for real providers (the
        // e2e harness flips `upstream_tls=false` for a plaintext mock).
        // Each connect retry (`fail_to_connect`) takes the next resolved address, so a name whose
        // first address is dead (`localhost` → `::1` first) still connects.
        let addr = match self
            .state
            .resolve(&rc.leg.provider.authority, rc.leg.addr)
            .await
        {
            Ok((a, _)) => a,
            Err(e) => {
                // DNS failures are rare and usually mean a misconfigured `provider_authorities`
                // override — so keep the diagnostic (provider name + authority + the resolver error,
                // already formatted into `e`) instead of discarding it behind an opaque static string.
                // `error_because` chains `e` as the cause so it shows in the Pingora error log.
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    authority = rc.leg.provider.authority.as_str(),
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
        rc.upstream_phase = UpstreamPhase::Attempted;
        let peer = self.build_peer(addr, &rc.leg.provider, rc.auto.as_ref().map(|a| a.route));
        self.cap_read_at_deadline(rc, peer)
    }

    /// Fail over — or walk a pool key — before a byte of the error reaches the client.
    ///
    /// This hook runs strictly before anything is written downstream (`h1_response_filter` /
    /// `h2_response_filter` both call it ahead of `write_response_tasks`), so returning a retryable
    /// error here re-enters pingora's retry loop. That is the only place in the response path where
    /// abandoning an answer is still possible.
    ///
    /// Erroring *here* rather than in `response_filter` also keeps the per-attempt state clean for
    /// free: `response_filter` never runs for the abandoned attempt, so nothing increments
    /// `active_streams`, observes TTFT, or sets `upstream_status`, and `response_body_filter` never
    /// feeds the tail, head, or response model scanner. The next attempt starts from the same slate
    /// the first one did.
    ///
    /// Two distinct retries, never mixed:
    ///
    /// - **Managed 429 or 401 → next unused key, same provider.** A 429 is a healthy provider
    ///   throttling *that credential*, not a vendor outage; a 401 is that credential revoked
    ///   (D71), which also cools the key off for later requests
    ///   (`Provider::mark_key_bad`, counted on `ai_key_auth_failures_total`). Walks `/{provider}`
    ///   and `/auto`. BYO does not walk. The last 429 is relayed, `Retry-After` included; the last
    ///   key's 401 is relayed on a provider route and is a candidate failure on a catalog walk.
    ///   Counted on `ai_key_walks_total`. A 403 never walks or cools a key (D84): it is usually
    ///   about the request; one whose body names the key cools it from `logging`.
    /// - **Model-routed 5xx → next candidate.** A provider-routed request named its provider; there
    ///   is nowhere else to go. A 429 is *not* a vendor failover — re-asking a different vendor
    ///   would convert a self-healing throttle into spend somewhere else. Counted on
    ///   `ai_candidate_failovers_total`.
    ///
    /// Both require a **replayable** body. Pingora's replay buffer is a private 64 KiB constant;
    /// past it a retry would send headers describing a body it then never writes. That case is
    /// relayed, not attempted. [`retry::decide`] makes the call; this hook reports the status and
    /// acts on the answer.
    async fn upstream_response_filter(
        &self,
        session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Some(rc) = ctx.as_mut() else {
            return Ok(());
        };
        let status = upstream_response.status.as_u16();
        // A 2xx says this key's account pays: forget its unread OpenRouter 402s (D261).
        if rc.managed() && (200..300).contains(&status) {
            rc.leg.provider.key_served(rc.leg.pool_key);
        }

        // Managed 401: this pool key is revoked (D71). Never the caller's fault. Cool it off so
        // later requests start on a good key, then walk like a 429. Not a 403 (D84, see
        // `is_pool_key_failure`).
        let key_auth = rc.managed() && is_pool_key_failure(status);
        if key_auth {
            self.state.metrics.key_cooled(KeyCooled::Revoked);
            rc.leg.provider.mark_key_bad(rc.leg.pool_key);
        }

        // Below 400 nothing walks: the decision is always to relay, so the hot path reads no
        // further fact.
        if status < 400 {
            return Ok(());
        }
        let walk = rc.walk_facts();
        let decision = retry::decide(&retry::Failed {
            failure: retry::Failure::Status {
                status,
                managed: rc.managed(),
                next_key: rc.leg.next_key(),
            },
            walk,
            body: body_facts(session, walk),
        });
        let at = rc.auto.as_ref().map_or(0, |a| a.candidate);
        // A walk's 401–403 resolves its breaker permit as the success it is (the provider
        // answered); a 5xx's failure is recorded by `upstream_peer`'s prologue, which still sees
        // the permit outstanding, so recording it here as well would double-count.
        let key_failure = rc.managed() && is_candidate_refusal(status);
        // Failing over abandons this response at its head, so `Redact` never reads the body that
        // tells `logging` an account is out of credit (D180), and every later request would pay a
        // round trip to it first. A 402 says so by its status alone except on OpenRouter, whose
        // 402 may be one request too large for the balance (`remedy::unfunded_402`): cool that key
        // here, as `logging` would have, on either abandoning path below (D258). OpenRouter's
        // unread 402 is a strike against its key instead, and enough in a row cool it (D261).
        let unfunded_402 = || {
            (key_failure && status == 402)
                .then(|| rc.auto.as_ref().and_then(|a| a.candidate_at(at)))
                .flatten()
                .map(|c| remedy::unfunded_402(c.provider))
        };
        match decision {
            // The next pool key on this provider.
            Decision::Resend(Retry::SameLeg) => {
                self.state.metrics.key_walks_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    key = rc.leg.pool_key,
                    status,
                    "upstream returned {status}; trying the next pool key",
                );
                if let Some(next) = rc.leg.next_key() {
                    rc.leg.pool_key = next;
                }
                rc.leg.retry = Some(Retry::SameLeg);
                let mut e = pingora_core::Error::new(pingora_core::ErrorType::HTTPStatus(status));
                e.set_retry(true);
                Err(e)
            }
            Decision::Resend(Retry::NextCandidate) => {
                self.state.metrics.candidate_failovers_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    candidate = at,
                    status,
                    "upstream returned {status}; trying the next candidate",
                );
                let unfunded_402 = unfunded_402();
                if key_failure {
                    rc.leg.succeed();
                }
                self.abandon_402(rc, unfunded_402, status);
                rc.fail_over(at);
                let mut e = pingora_core::Error::new(pingora_core::ErrorType::HTTPStatus(status));
                e.set_retry(true);
                Err(e)
            }
            // The parent holds the body: it drops this response and re-runs.
            Decision::Rerun(retry @ RelayRetry::Key { .. }) => {
                self.state.metrics.key_walks_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    key = rc.leg.pool_key,
                    status,
                    "upstream returned {status}; re-running the full body on the next pool key",
                );
                rc.attempt.relay_abandoned =
                    full_body_ctx(session).is_some_and(|fb| fb.record(retry));
                Ok(())
            }
            Decision::Rerun(retry) => {
                // `response_filter` still runs for it, so the breaker sees the failure.
                self.state.metrics.candidate_failovers_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    candidate = at,
                    status,
                    "upstream returned {status}; re-running the full body on the next candidate",
                );
                let unfunded_402 = unfunded_402();
                rc.attempt.relay_abandoned =
                    full_body_ctx(session).is_some_and(|fb| fb.record(retry));
                // Not recorded (the final attempt): this answer is relayed and `logging` reads it.
                if rc.attempt.relay_abandoned {
                    self.abandon_402(rc, unfunded_402, status);
                }
                Ok(())
            }
            // Relayed, `Retry-After` included.
            Decision::Stand(stand) => {
                if stand.key_walk_unreplayable {
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        status,
                        body_done = session.as_mut().is_body_done(),
                        "upstream returned {status} but the request body is not provably replayable; not walking keys",
                    );
                }
                if stand.failover_unreplayable {
                    self.state.metrics.failover_unreplayable_total.inc();
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        status,
                        body_done = session.as_mut().is_body_done(),
                        "upstream failed but the request body is not provably replayable; relaying the error",
                    );
                }
                Ok(())
            }
        }
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream_request: &mut pingora::http::RequestHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Some(rc) = ctx.as_mut() else {
            return Ok(());
        };
        // Pingora runs this once the connection is up, just before the request head goes out.
        rc.upstream_phase = UpstreamPhase::Connected;
        let rc = &*rc;

        // Managed: swap the virtual key for the real pool key (precomputed at boot) in the scheme
        // the upstream wants — removing every inbound static-key header first (see
        // `STATIC_KEY_HEADERS`) so the virtual key never leaks upstream, and so a provider whose own
        // auth header (e.g. Azure's `api-key` — Task #8) happens to be the same header the client
        // presented its virtual key in doesn't end up with two stacked values. BYO (`!managed`):
        // leave the user's own auth header exactly as presented.
        if rc.managed() {
            // Strip **unconditionally**, before deciding whether there is a pool key to insert.
            //
            // These two used to sit inside the `if let Some(av)` below, which was safe only because
            // a managed request without a pool key is rejected earlier. That is a load-bearing
            // invariant expressed nowhere near here, and if it ever failed the consequence was not a
            // degraded request but a *credential disclosure*: nothing removed, nothing inserted, and
            // the caller's Ed25519 virtual key forwarded verbatim to a third-party provider. Now the
            // worst case is an unauthenticated request the provider rejects with a 401.
            //
            // The allowlist would drop these too; they are removed by name so this invariant does
            // not hang on the list's contents.
            let eager_tool_streaming = rc
                .auto
                .as_ref()
                .and_then(|a| a.candidate_at(a.candidate))
                .is_none_or(providers::catalog::serves_eager_input_streaming);
            retain_managed_client_headers(
                upstream_request,
                rc.leg.provider.id == Some(ProviderId::Anthropic),
                eager_tool_streaming,
            )?;
            upstream_request.remove_header("authorization");
            for header in STATIC_KEY_HEADERS {
                upstream_request.remove_header(header);
            }
            // Ask for an uncompressed response. Stock Python and Node SDKs send
            // `Accept-Encoding: gzip`, OpenAI and OpenRouter honor it, and every byte the gateway
            // reads afterwards (the usage tail, the cache fill, translation) assumes plain JSON/SSE:
            // a gzipped body billed zero tokens and a cache hit replayed gzip bytes without their
            // `Content-Encoding`. `identity` rather than removing the header: with no
            // `Accept-Encoding` at all, HTTP allows any coding. Managed only; a BYO response is never
            // parsed for billing.
            upstream_request.insert_header("accept-encoding", "identity")?;
            if let Some(auth) = rc.leg.provider.pool_auth.get(usize::from(rc.leg.pool_key)) {
                // Clone the boot-built `HeaderValue` (a refcount bump) rather than re-validating and
                // re-copying the key out of a `&str` on every managed request. The `&str` path is
                // kept as a fallback for a key that isn't a legal header value, which could never
                // have worked anyway — see `PoolAuth`.
                match &auth.header {
                    Some(hv) => {
                        upstream_request.insert_header(rc.leg.provider.auth.header(), hv.clone())?
                    }
                    None => upstream_request
                        .insert_header(rc.leg.provider.auth.header(), auth.value.expose())?,
                }
            }
        }

        // `x-beyond-*` is Beyond's namespace: the routing header, the control headers, and any name
        // the gateway does not (yet) define. None of it reaches a provider, on any route, managed or
        // BYO (D67) — a provider that rejects unknown headers would turn an opt-in into their 400,
        // and a header the gateway adds later must not have leaked from old clients first. A sweep
        // by prefix rather than a list of names, so a new header cannot be forgotten here. Runs on
        // every attempt (pingora rebuilds the head from the downstream request each time), and
        // before anything below adds a header of its own. Allocates nothing.
        strip_beyond_headers(upstream_request);

        // Point Host at the upstream. Same precomputed-value trick as the pool key above.
        match &rc.leg.provider.host_header {
            Some(hv) => upstream_request.insert_header("host", hv.clone())?,
            None => upstream_request.insert_header("host", rc.leg.provider.host.as_str())?,
        }

        // Dashboard-attribution headers (OpenRouter, managed traffic only — Task #22, see
        // `apply_provider_attribution`).
        apply_provider_attribution(upstream_request, rc.leg.provider.id, rc.managed())?;

        // A stock OpenAI SDK does not send `anthropic-version`. Anthropic (and Bedrock Messages)
        // require it; inject the current version when we translated Chat Completions or Responses
        // → Messages.
        if rc
            .auto
            .as_ref()
            .and_then(|a| a.translate.as_ref())
            .is_some_and(|t| {
                t.client != route::Endpoint::Messages && rc.dialect == Dialect::Anthropic
            })
            && upstream_request.headers.get("anthropic-version").is_none()
        {
            upstream_request.insert_header("anthropic-version", "2023-06-01")?;
        }
        if rc
            .auto
            .as_ref()
            .and_then(|a| a.translate.as_ref())
            .is_some_and(|_| rc.dialect == Dialect::OpenAi)
        {
            upstream_request.remove_header("anthropic-version");
        }
        // Preserved thinking on a translated walk: `translate::request` sets
        // `thinking.block_binding` for Anthropic's own conversation-binding models, a 400 without
        // this beta. Decided from the candidate alone, since headers leave before the body is
        // translated; merged with any beta value the client sent.
        if let Some(a) = rc.auto.as_ref()
            && catalog_translating(a)
            && let Some(c) = a.candidate_at(a.candidate)
            && c.provider == providers::ProviderId::Anthropic
            && route::Endpoint::of_upstream_path(c.path) == route::Endpoint::Messages
        {
            if let Some(beta) = translate::messages_beta(c.upstream_model) {
                merge_anthropic_beta(upstream_request, beta)?;
            }
            // A Chat Completions or Responses client's `service_tier: "priority"`, translated to
            // fast mode on a candidate that serves it (D266). The same decision makes
            // `translate::request_with_tools` add `speed: "fast"` to this attempt's body.
            if translated_fast(a, c) {
                merge_anthropic_beta(upstream_request, translate::FAST_MODE_BETA)?;
            }
        }

        // Forward the provider-native path (computed in `request_filter`): the client path with the
        // `/{provider}` segment stripped. Sent verbatim — no per-provider rewriting. The body's
        // framing (Content-Length / chunked) is preserved.
        //
        // `None` means the bare-path default, where the path is already what the upstream should
        // see, so there is nothing to build and nothing to parse. That used to be expressed by
        // reconstructing the path anyway and comparing it against the inbound `path_and_query`;
        // encoding it in the type instead skips the allocation rather than detecting it after
        // the fact.
        if let Some(forward_path) = rc.forward_path.as_str()
            && let Ok(uri) = forward_path.parse()
        {
            upstream_request.set_uri(uri);
        } else if rc.managed()
            && let Some(stripped) = upstream_request
                .uri
                .path_and_query()
                .and_then(|pq| strip_key_param(pq.as_str()))
            && let Ok(uri) = stripped.parse()
        {
            // The inbound path goes out as-is, and on a managed request its `key` params are the
            // virtual key: `request_filter` strips them from a built `forward_path`, this from the
            // path nobody rebuilt.
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
        // Keyed on `rewrites_body`, not `stream_options`, so the model rewrite gets the same
        // treatment: `openai/gpt-4o-mini` is longer than `gpt-4o-mini`, and forwarding the client's
        // original `Content-Length` alongside a longer body truncates it at the upstream.
        if rc.plan.rewrites_body() {
            upstream_request.remove_header("content-length");
            if upstream_request.version != http::Version::HTTP_2 {
                upstream_request.insert_header("transfer-encoding", "chunked")?;
            }
        }
        Ok(())
    }

    async fn request_body_filter(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Ctx { rc, held } = ctx;
        let Some(rc) = rc.as_mut() else {
            return Ok(());
        };
        // An upload still trickling in at the request's deadline ends there (see
        // `response_body_filter`).
        if crate::deadline::expired(rc.deadline) {
            return Err(self.deadline_cut(&rc.request_id));
        }
        // Feed the body through the structural scanner as it passes (never withheld, never
        // buffered) to extract the exact root-level `model` — but only for **managed** traffic,
        // which is the only path that reads it. `rc.model` is used at exactly two places, both
        // inside the managed (`Billing::Managed`) block in `logging` (the billing-log fallback and
        // `requested_model`). Scanning it for BYO meant walking the whole request body — a
        // structural, depth- and escape-aware pass — to produce a value guaranteed to be discarded.
        if let Some(chunk) = body.as_ref() {
            // Enforce the body cap on the *streamed* size too: the up-front `Content-Length` check in
            // `request_filter` can't see a chunked-encoded body (no declared length). We don't buffer
            // — we just count — and abort the proxied request once the running total crosses the cap.
            // Aborting (vs. a clean 413) is acceptable here: headers are already away to the upstream,
            // and this is an abuse guard, not a normal client path.
            //
            // Tagged `Downstream` because it is the client's fault: an untagged error counted
            // against the provider's breaker, so one caller sending oversized bodies could open it
            // and 503 every tenant. `fail_to_proxy` answers it with the refusal, if it still can.
            if !rc.take_request_chunk(chunk) {
                return Err(self.refusal_error(Refusal::BodyTooLarge).into_down());
            }
            // Eligible requests are buffered so we can splice the root object before any byte reaches
            // the upstream (injection inserts near the front, so we can't have forwarded it already).
            // When we're buffering anyway, the incremental scan is skipped entirely and both answers
            // come from one walk of the finished buffer below — the body was previously traversed
            // twice, once here for `model` and once by the injection planner, over the same bytes
            // with the same depth/string/escape bookkeeping.
            // Only where no copy of the body will be left for `logging` to read (`tally_eager`):
            // everywhere else the estimate is made there, and only when a row needs one.
            if rc.tally_eager {
                rc.attempt.input_tally.feed(chunk);
                // The price knobs, for the same reason: no copy of this body is left for
                // `logging` to read (see `requested_knobs`).
                if let Some(taps) = rc.billing.tap() {
                    let (scan, kept) = taps.knobs.get_or_insert_with(|| {
                        (peek::ModelScanner::new(), peek::Kept::request_knobs())
                    });
                    scan.feed_keeping(chunk, kept);
                }
            }

            if rc.plan.rewrites_body() {
                // A large body buffered for a rewrite (the `/{provider}` usage splice; a catalog
                // walk's large body is a `FullBody` re-run, already reserved) holds budget as it
                // grows. Tagged downstream: the client's size, not the provider's health.
                let held_len = rc.attempt.req_buf.len();
                if past_replay_buffer(held_len) && !held.reserve_body(held_len) {
                    return Err(self.refusal_error(Refusal::BodyMemory).into_down());
                }
            } else if rc.managed() {
                rc.attempt.model_scanner.feed(chunk);
            }
        }

        if rc.plan.rewrites_body() {
            if end_of_stream {
                // One structural walk for every answer (see `peek::scan_buffered`).
                let mut buf = std::mem::take(&mut rc.attempt.req_buf);
                // --- signed ids (`signed_id`): the provider gets its own ids back ---
                // The client's body, per attempt. A catalog walk was checked before connecting; a
                // `/{provider}` body is refused here, before a byte of it goes upstream.
                if rc.signed.is_some()
                    && let Some(signer) = self.state.id_signer.as_ref()
                    && let Some(id) = rc.billing.identity()
                {
                    match signer.unsign_request(id.tenant_id, &buf, rc.auto.is_some()) {
                        Ok(Some(raw)) => buf = raw,
                        Ok(None) => {}
                        Err(_) => {
                            return Err(self.refusal_error(Refusal::ForeignIdLate).into_down());
                        }
                    }
                }
                // --- end signed ids ---
                let mut scan = peek::scan_buffered(&buf);
                // The body rules (`body_rules`), on the client's body: ahead of translation, which
                // re-serializes it and would hide a second `model`, and of every rewrite, so an
                // `:online` model is named for what it is. Each refusal ends the request before a
                // byte of the body goes upstream. Every body buffered here is managed.
                let inspection = if rc.managed() {
                    let model = rc
                        .auto
                        .as_ref()
                        .map(|a| a.route.model)
                        .or(scan.model.as_deref())
                        .unwrap_or(rc.model.as_str());
                    let messages = session.req_header().uri.path().ends_with("/messages");
                    unpriced::inspect(&buf, model, messages)
                } else {
                    unpriced::Inspection::default()
                };
                let refused = {
                    // A managed `/{provider}/…` billable call naming no catalog row (D267): its
                    // billing row would carry no `price_model`. Told by the client's id.
                    let miss = (rc.plan.catalog_check
                        && scan.model.as_deref().and_then(price_row).is_none())
                    .then(|| match rc.model.is_empty() {
                        true => scan.model.take().map_or(Cow::Borrowed(""), sanitize_model),
                        false => Cow::Borrowed(rc.model.as_str()),
                    });
                    let facts = body_rules::Facts {
                        // A catalog walk routed on one `model` and re-spells one, and a managed
                        // `/{provider}` body is billed by the first: the provider's parser picks
                        // its own (usually the last).
                        duplicate_model: (rc.auto.is_some() || rc.plan.catalog_check)
                            && scan.duplicate_model,
                        // A limit with no span (a string, a fraction, a negative): the clamp below
                        // cannot hold it to the row's maximum.
                        inexact_limit: scan.inexact_limit
                            && rc
                                .auto
                                .as_ref()
                                .is_some_and(|a| a.route.card.output_cap().is_some()),
                        refused_input: None,
                        unpriced: inspection.refused,
                        catalog_miss: miss.as_deref(),
                        // Managed `/{provider}/…/responses` asking for `background: true` (D202).
                        background: rc.plan.catalog_check
                            && rc.forward_path.as_str().is_some_and(|p| {
                                route::forward_is_responses(p.split_once('?').map_or(p, |(p, _)| p))
                            })
                            && requests_background(&buf),
                    };
                    body_rules::refusal(body_rules::Point::Buffered, facts)
                };
                if let Some(refusal) = refused {
                    return Err(self.refusal_error(refusal).into_down());
                }
                if inspection.web_search_preview
                    && let Some(taps) = rc.billing.tap()
                {
                    taps.web_search_preview = true;
                }
                // Wire mismatch on this *candidate*: map the inbound JSON onto this path's
                // endpoint *before* the model splice. Keep the original client body in `req_buf`
                // (cleared and replayed per attempt) so a mixed-row failover re-translates rather
                // than forwarding the previous candidate's wire. OpenAI→Anthropic drops
                // `stream_options` here; Anthropic→OpenAI leaves `include_usage` to the inject
                // below, on the translated Chat Completions body. Same-endpoint Responses
                // (session arm) is `from == to` and is a byte relay, less the reasoning items the
                // gateway minted from Claude's thinking (see `translate::strip_gateway_reasoning`).
                if let Some(a) = rc.auto.as_mut() {
                    // An output limit past the row's maximum is a 400 by name; capped before the
                    // translation, so what it derives (a thinking budget) fits under the cap too.
                    // Only a vendor-published maximum: an unpublished one is a placeholder (D85).
                    let mut changed = clamp_output_limits(
                        &mut buf,
                        &scan.limit_spans,
                        a.route.card.output_cap().unwrap_or(0),
                    );
                    let serving = catalog_serving_endpoint(a.as_ref());
                    let candidate = a.candidate_at(a.candidate);
                    let upstream_model = candidate.map_or("", |c| c.upstream_model);
                    let openai_host =
                        candidate.is_some_and(|c| c.provider == providers::ProviderId::OpenAi);
                    // Read once: this attempt's body edits ask the candidate several questions.
                    let caps = candidate.map(route::Candidate::caps);
                    let has = |flag| caps.is_some_and(|c: Caps| c.contains(flag));
                    let stream_only = has(Caps::STREAM_ONLY);
                    let fast = candidate.is_some_and(|c| translated_fast(a, c));
                    let reads_developer = caps.is_none() || has(Caps::DEVELOPER_ROLE);
                    let tool_thinking = candidate.map_or(
                        providers::catalog::ToolThinking::Free,
                        providers::catalog::tool_thinking,
                    );
                    let eager_tool_streaming = caps.is_none() || has(Caps::EAGER_INPUT_STREAMING);
                    let claude = has(Caps::CLAUDE);
                    if let Some(t) = a.translate.as_mut()
                        && let Some(to) = serving
                    {
                        t.tools = translate::ToolNames::default();
                        t.gateway_cache = false;
                        t.cache_1h = false;
                        if t.client != to {
                            // Translation builds `Value`s of the body: its heap is held in the
                            // body budget for the step, or the body is refused (D216).
                            let _heap = self.hold_translation_heap(&buf)?;
                            // The tool names this attempt's response maps calls back through, and
                            // whether its cache breakpoints are the gateway's (per attempt: a
                            // failover candidate on another wire re-decides both).
                            (buf, t.tools, t.gateway_cache, t.cache_1h) =
                                translate::request_with_tools(
                                    t.client,
                                    to,
                                    &buf,
                                    upstream_model,
                                    fast,
                                );
                            changed = true;
                        } else if to == route::Endpoint::Responses {
                            let len = buf.len();
                            buf = translate::strip_gateway_reasoning(buf);
                            // A row with no Responses arm walks its candidates only for a one-shot
                            // (session state there is a 400): onto a Responses candidate (xAI's)
                            // it must not be stored, and xAI stores by default. Nothing there holds
                            // a tool step's `item_reference` either: dropped, as translation drops
                            // it (one for an earlier turn was the 400, D175).
                            changed |= buf.len() != len;
                            if a.route.responses.is_empty() {
                                changed |= translate::store_false(&mut buf);
                                changed |= translate::strip_item_references(&mut buf);
                            }
                        } else if to == route::Endpoint::ChatCompletions && claude {
                            // A Claude model behind Chat Completions (OpenRouter): a tool turn
                            // with no thinking to replay goes without reasoning (D14's rule).
                            let len = buf.len();
                            buf = translate::claude_chat_relay_reasoning(buf, upstream_model);
                            changed |= buf.len() != len;
                        }
                        // Same-wire Chat to another host: an explicit null is "not set", as
                        // translation treats it; OpenRouter 400s `user: null` (D101).
                        if t.client == to && to == route::Endpoint::ChatCompletions && !openai_host
                        {
                            changed |= peek::remove_root_nulls(&mut buf);
                        }
                        if to == route::Endpoint::ChatCompletions {
                            // `developer` is `system` to a host that is not OpenAI's (D173); a
                            // translated Responses body already says so.
                            if t.client == to && !reads_developer {
                                changed |= translate::developer_as_system(&mut buf);
                            }
                            // A candidate whose thinking breaks tools gets them with thinking off
                            // (D171, D172): every candidate of its row refuses or garbles the
                            // combination, so steering around it would not help.
                            changed |= translate::fit_tools_to_thinking(&mut buf, tool_thinking);
                        }
                        // A candidate that answers only streams, for a client that did not ask
                        // for one: ask it for the stream and assemble the answer into the
                        // client's own body (D147). Per attempt, like the tools: a failover
                        // candidate gets the client's body as it came. Its usage is asked for by
                        // the splice below, as on every Chat Completions stream (`inject_at`, or
                        // `force_include_usage` on the client's own `stream_options`); a Messages
                        // body takes no `stream_options`.
                        t.assemble = stream_only
                            && to != route::Endpoint::Responses
                            && translate::force_stream(&mut buf, false);
                        changed |= t.assemble;
                    }
                    // Fine-grained tool streaming only where the candidate takes it (D272): an
                    // older Bedrock stack 400s the field, and a 400 is not failed over. Per
                    // attempt, so direct Anthropic still gets the client's bytes (and its prompt
                    // cache key) unchanged; `upstream_request_filter` drops the matching beta.
                    if serving == Some(route::Endpoint::Messages) && !eager_tool_streaming {
                        changed |= translate::strip_eager_input_streaming(&mut buf);
                    }
                    if changed {
                        scan = peek::scan_buffered(&buf);
                    }
                    // Native OpenAI Chat takes `max_completion_tokens` on every model and 400s
                    // `max_tokens` on its reasoning ones; a translated body already says the former.
                    if candidate.is_some_and(native_openai_chat)
                        && rename_max_tokens(&mut buf, &scan.limit_keys)
                    {
                        scan = peek::scan_buffered(&buf);
                    }
                }
                if rc.model.is_empty()
                    && let Some(m) = scan.model
                {
                    // The *client's* id, captured before any rewrite below — this is what
                    // `requested_model` means, and it stays the canonical catalog name on a
                    // model-routed request even though the upstream is about to be told
                    // something else.
                    rc.model = sanitize_model(m).into_owned();
                }
                // Model-routed: re-spell `model` as the candidate serving *this attempt* spells it.
                // Providers essentially never agree on an id — Anthropic's `claude-opus-4-8` is
                // OpenRouter's `anthropic/claude-opus-4-8` — so without this a failover would ask
                // the fallback for a model it has never heard of.
                //
                // Done before the `stream_options` splice, and safe in that order because
                // `inject_at` points just past the root `{` and so always precedes the model value:
                // rewriting the value cannot move it. A client-sent `stream_options` can sit on
                // either side of `model`, so that rewrite runs first and hands back the span it
                // may have shifted.
                // A duplicate or escaped `stream_options` (OpenAI takes the last, decoded): drop
                // them all, so the injection below adds the one that counts (D88).
                if rc.plan.stream_options
                    && scan.stream_options_ambiguous
                    && peek::remove_root_members(&mut buf, "stream_options")
                {
                    scan = peek::scan_buffered(&buf);
                }
                let (buf, model_span) = match scan.stream_options_at {
                    Some(at) if rc.plan.stream_options => {
                        force_include_usage(buf, at, scan.model_span)
                    }
                    _ => (buf, scan.model_span),
                };
                let buf = match (rc.auto.as_ref(), model_span) {
                    (Some(a), Some(span)) => match a.candidate_at(a.candidate) {
                        Some(c) => apply_model_rewrite(buf, span, c.upstream_model.as_bytes()),
                        None => buf,
                    },
                    _ => buf,
                };
                // Emit the whole (possibly rewritten) body in one shot; `transfer-encoding: chunked`
                // (set in `upstream_request_filter`) makes the changed length fine.
                // `plan.stream_options` is the OpenAI-upstream gate: a translated Anthropic body also
                // carries `"stream":true`, and splicing `stream_options` into it would be a field
                // that API does not recognize.
                let buf = if rc.plan.stream_options {
                    apply_stream_usage_injection(buf, scan.inject_at)
                } else {
                    buf
                };
                let buf = match rc.auto.as_ref().and_then(|a| a.candidate_at(a.candidate)) {
                    Some(c) if openrouter_chat(c) => disable_openrouter_compression(buf),
                    _ => buf,
                };
                *body = Some(Bytes::from(buf));
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
        if end_of_stream
            && rc.managed()
            && !rc.plan.stream_options
            && rc.model.is_empty()
            && let Some(m) = rc.attempt.model_scanner.take_model()
        {
            rc.model = sanitize_model(m).into_owned();
        }
        Ok(())
    }

    async fn response_filter(
        &self,
        session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> Result<()> {
        let Ctx { rc, held } = ctx;
        if let Some(rc) = rc.as_mut() {
            // Headers arrived ≈ time-to-first-byte. Per-provider handle resolved once at boot (see
            // `ProviderMetrics`) — first-token latency is per-provider, so an unlabeled histogram
            // can't tell you which one regressed.
            //
            // Timed from `attempt_start`, not `start`. They are the same instant for a request that
            // connects first try; they differ once a candidate has been abandoned, and charging the
            // dead candidate's `connect_timeout_secs` to the provider that actually answered would
            // render an outage at A as a latency regression at B — inverting the one thing the
            // per-provider label is for.
            // Per-provider response counter, bucketed by status class — the signal that a provider
            // is degrading (429/5xx) before it shows up only as latency or a missing usage event.
            let status = upstream_response.status.as_u16();
            rc.leg
                .provider
                .metrics
                .ttft_seconds
                .observe(rc.attempt_start().elapsed().as_secs_f64());
            rc.leg.provider.metrics.record_response(status);
            rc.upstream_status = held_status(status);
            // Resolve the breaker permit here, at the head, not at the end of the body: the head
            // is the provider's answer (a 5xx is broken, anything else is reachable), and a
            // half-open probe resolved only at end of stream let one long or stalled stream 503
            // the provider for everyone until it ended. `logging` resolves only the attempts that
            // never got a head.
            let answered = if status >= 500 {
                rc.leg.fail();
                None
            } else {
                rc.leg.succeed()
            };
            // A 2xx is not yet known to be healthy: a provider can answer 200 with an error body
            // (OpenRouter's error-in-200, an SSE stream whose first event is an error), so its
            // breaker verdict waits for the body's first bytes (`response_body_filter`).
            if (200..300).contains(&status)
                && let Some(a) = rc.auto.as_mut()
            {
                a.health = Some(PendingHealth {
                    prefix: Vec::new(),
                    permit: answered,
                });
            }

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
                held.open_stream();
            }
            if rc.managed() {
                tap_response_head(rc, upstream_response);
            }
            // Drain-on-cancel, decided per head (a walk's next attempt may be another provider).
            // A drain already under way is left as it is.
            if matches!(held.drain, Drain::Off | Drain::Armed) {
                held.drain = if drains_on_cancel(rc, status) && full_body_ctx(session).is_none() {
                    if drain::arm(&mut session.cache) {
                        Drain::Armed
                    } else {
                        Drain::Off
                    }
                } else {
                    Drain::Off
                };
            }
            self.state.fault_point("response_filter");

            // `x-beyond-*` is the gateway's namespace. A provider (or anything between us and it)
            // that sent its own `x-beyond-provider` or `x-beyond-cache-status` would otherwise reach
            // the client beside, or instead of, ours. Allocates only when one is present.
            let spoofed: Vec<http::HeaderName> = upstream_response
                .headers
                .keys()
                .filter(|k| k.as_str().starts_with("x-beyond-"))
                .cloned()
                .collect();
            for name in &spoofed {
                upstream_response.remove_header(name);
            }

            // The pool key never reaches the client (D66). A provider, or anything between us and
            // it, that echoes the credential it was sent would otherwise hand Beyond's key to a
            // tenant: any header carrying it is dropped, and an error body (>= 400, where an echo
            // lives: "Incorrect API key provided: …") is scrubbed as it streams (`Redact`). A 2xx
            // body is an answer and is not scanned; a streamed answer would pay for it per chunk.
            rc.redact = None;
            if rc.managed()
                && let Some(finder) = rc
                    .leg
                    .provider
                    .pool_auth
                    .get(usize::from(rc.leg.pool_key))
                    .map(route::PoolAuth::finder)
                    .filter(|f| !f.needle().is_empty())
            {
                // Collects nothing (and so allocates nothing) unless a header echoes the key.
                let echoed: Vec<http::HeaderName> = upstream_response
                    .headers
                    .iter()
                    .filter(|(_, v)| finder.find(v.as_bytes()).is_some())
                    .map(|(k, _)| k.clone())
                    .collect();
                for name in &echoed {
                    upstream_response.remove_header(name);
                }
            }
            // A managed JSON error is held whole and checked for a provider-account remedy (D174):
            // the rewrite changes its length, so it loses its `Content-Length` here, on the error
            // path only. A BYO error is the caller's own account talking and is relayed as sent.
            if rc.managed() && status >= 400 {
                let whole = !rc.streaming
                    && upstream_response
                        .headers
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .is_some_and(|ct| ct.contains("json"));
                if whole {
                    upstream_response.remove_header("content-length");
                    if upstream_response.version != http::Version::HTTP_2 {
                        upstream_response.insert_header("transfer-encoding", "chunked")?;
                    }
                }
                let keyed = rc
                    .leg
                    .provider
                    .pool_auth
                    .get(usize::from(rc.leg.pool_key))
                    .is_some_and(|a| !a.finder().needle().is_empty());
                if whole || keyed {
                    rc.redact = Some(Box::new(Redact {
                        whole,
                        status,
                        ..Redact::default()
                    }));
                }
            }

            // Echo the request id so a client (or an oncall reading a captured response) can quote it
            // and land on this request's log line. `insert_header` only fails on an invalid value;
            // our id is `[0-9a-f-]`, always valid — but surface a failure rather than silently drop.
            upstream_response.insert_header(REQUEST_ID_HEADER, rc.request_id.as_str())?;
            if let Some(hv) = rc.leg.provider.name_header.as_ref() {
                upstream_response.insert_header(PROVIDER_HEADER, hv.clone())?;
            }
            if let Some(c) = rc.auto.as_ref().and_then(|a| a.candidate_at(a.candidate)) {
                upstream_response.insert_header(UPSTREAM_MODEL_HEADER, c.upstream_model)?;
            }

            // An assembled answer (D147): the upstream streams, the client gets one JSON body.
            if rc.streaming && rc.auto.as_ref().is_some_and(|a| catalog_assembling(a)) {
                upstream_response.insert_header("content-type", "application/json")?;
            }
            if let Some(cache::Pending::Fill { content_type, .. }) =
                rc.auto.as_mut().and_then(|a| a.cache.as_mut())
            {
                *content_type = upstream_response
                    .headers
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_owned().into_boxed_str());
            }

            // Translate changes the body length (and SSE event count). Drop the upstream
            // Content-Length so the client is not truncated; H1 needs chunked framing.
            // Same-endpoint candidates stay a byte relay, except a Chat Completions stream from a
            // vendor other than OpenAI: it relays through `SseBridge`, which drops the identity
            // fields OpenRouter repeats on every chunk (see `translate::ChatIdentity`).
            if rc.auto.as_ref().is_some_and(|a| {
                catalog_translating(a)
                    || (rc.streaming && (catalog_chat_relay(a) || catalog_assembling(a)))
                    || catalog_error_relay(a, rc.upstream_status(), rc.streaming)
            }) {
                let streaming = rc.streaming;
                let upstream = rc
                    .auto
                    .as_ref()
                    .and_then(|a| catalog_serving_endpoint(a))
                    .unwrap_or_else(|| route::Endpoint::of_wire(rc.dialect));
                if let Some(t) = rc.auto.as_mut().and_then(|a| a.translate.as_mut())
                    && streaming
                {
                    let bridge = if t.assemble {
                        translate::SseBridge::assembling(upstream)
                    } else {
                        translate::SseBridge::new(t.client, upstream)
                    };
                    t.sse = Some(
                        bridge
                            .with_tools(t.tools.clone())
                            .with_gateway_cache(t.gateway_cache),
                    );
                }
                upstream_response.remove_header("content-length");
                if upstream_response.version != http::Version::HTTP_2 {
                    upstream_response.insert_header("transfer-encoding", "chunked")?;
                }
            }
            // --- signed ids (`signed_id`): a 2xx's ids are rewritten, so its length changes ---
            if let Some(s) = rc.signed.as_mut()
                && s.begin(status, rc.streaming)
            {
                upstream_response.remove_header("content-length");
                if upstream_response.version != http::Version::HTTP_2 {
                    upstream_response.insert_header("transfer-encoding", "chunked")?;
                }
            }
            // --- end signed ids ---
        }
        Ok(())
    }

    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<Option<Duration>>
    where
        Self::CTX: Send + Sync,
    {
        // Usage taps read the *upstream* dialect. Cache fill (#68) and capture read the bytes
        // the client sees — post-translate on a wire-mismatched catalog walk, the relayed
        // chunk otherwise. SSE is converted event-by-event; the full stream is never buffered.
        let Ctx { rc, held } = ctx;
        let Some(rc) = rc.as_mut() else {
            return Ok(None);
        };
        self.state.fault_point("response_body_filter");
        // A response still moving at the request's deadline is cut (a silent one is ended by the
        // capped read timeout instead). One atomic load: see `crate::deadline`.
        if crate::deadline::expired(rc.deadline) {
            return Err(self.deadline_cut(&rc.request_id));
        }
        // First, so nothing downstream — translation, capture, the cache, the usage tail — ever
        // holds the pool key (D66).
        if let Some(r) = rc.redact.as_mut() {
            let keys = rc
                .leg
                .provider
                .pool_auth
                .get(usize::from(rc.leg.pool_key))
                .map_or(&[][..], route::PoolAuth::finders);
            *body = r.feed(keys, body.take(), end_of_stream);
        }
        let chunk = body.as_deref().unwrap_or(&[]);
        // A catalog walk's 2xx waits here for its health verdict (see `response_filter`): the
        // first event or JSON key says whether the 200 carries an answer or an error. Bounded, and
        // done after a few hundred bytes.
        if rc.auto.as_ref().is_some_and(|a| a.health.is_some()) {
            self.settle_health(rc, chunk, end_of_stream);
        }
        if !chunk.is_empty() {
            // Tap the provider-reported (resolved/billed) model from the response *head* — the
            // scanner stops at the first root `model`, so this is O(1) and cheap (it finds the model
            // in the first chunk and ignores the rest). Kept separate from the tail because the model
            // is at the start of the response while the usage event is at the end.
            //
            // Managed only, for the same reason as the request-side scanner: the extracted value is
            // read at exactly one place, inside the managed (`Billing::Managed`) block in `logging`. A BYO
            // request has no Beyond identity and emits no billing row, so scanning its response was
            // pure waste — and *unbounded* waste on any response with no root-level `model`, since
            // the scanner never reaches its `done` short-circuit and walks every byte.
            if rc.managed() {
                match rc.billing.taps_mut() {
                    Some(t) => {
                        let kept = t.resp_kept.get_or_insert_with(peek::Kept::response);
                        rc.resp_model_scanner.feed_keeping(chunk, kept);
                        if let Some(tools) = t.tools.as_mut() {
                            tools.feed(chunk);
                        }
                    }
                    None => rc.resp_model_scanner.feed(chunk),
                }
            }

            // Anthropic SSE only: the head that keeps `message_start` (see `keep_usage_head`).
            rc.keep_usage_head(chunk);

            rc.resp_tail.push(chunk);
            // Managed only. Counts *upstream* bytes (pre-translate), like the tail.
            if rc.managed() {
                rc.resp_bytes = rc
                    .resp_bytes
                    .saturating_add(u32::try_from(chunk.len()).unwrap_or(u32::MAX));
            }
        }

        // Drain-on-cancel: once the client has left an armed stream, a chunk feeds the usage taps
        // above and nothing else, and the client is sent nothing more (see `crate::drain`).
        if held.drain != Drain::Off && drain_chunk(rc, held, end_of_stream) {
            *body = None;
            return Ok(None);
        }

        // --- signed ids (`signed_id`): after the usage taps, before anything the client sees ---
        if let (Some(s), Some(signer)) = (rc.signed.as_mut(), self.state.id_signer.as_ref()) {
            match s.feed(signer, body.as_deref().unwrap_or(&[]), end_of_stream) {
                Ok(Some(out)) => *body = Some(Bytes::from(out)),
                Ok(None) => {}
                Err(signed_id::Overflow) => {
                    return Err(self.translate_overflow(&rc.request_id, "signed_id"));
                }
            }
        }
        let chunk = body.as_deref().unwrap_or(&[]);
        // --- end signed ids ---

        // A translation, a Chat Completions relay that `response_filter` gave a bridge, or a
        // same-endpoint error put in the client's envelope.
        let translating = rc.auto.as_ref().is_some_and(|a| {
            catalog_translating(a)
                || a.translate.as_ref().is_some_and(|t| t.sse.is_some())
                || catalog_error_relay(a, rc.upstream_status(), rc.streaming)
        });
        if translating {
            let streaming = rc.streaming;
            let dialect = rc.dialect;
            let status = rc.upstream_status().unwrap_or(200);
            let upstream = rc
                .auto
                .as_ref()
                .and_then(|a| catalog_serving_endpoint(a))
                .unwrap_or_else(|| route::Endpoint::of_wire(dialect));
            let out = if let Some(t) = rc.auto.as_mut().and_then(|a| a.translate.as_mut()) {
                if streaming {
                    let (client, tools, gateway_cache) = (t.client, &t.tools, t.gateway_cache);
                    let sse = t.sse.get_or_insert_with(|| {
                        translate::SseBridge::new(client, upstream)
                            .with_tools(tools.clone())
                            .with_gateway_cache(gateway_cache)
                    });
                    let mut out = sse.feed(chunk, end_of_stream);
                    if let Some(buffer) = sse.overflow() {
                        return Err(self.translate_overflow(&rc.request_id, buffer));
                    }
                    // An assembling bridge writes nothing until the upstream has ended.
                    if t.assemble && end_of_stream {
                        out = sse.assembled(client);
                    }
                    out
                } else {
                    if !t.push_json(chunk) {
                        return Err(self.translate_overflow(&rc.request_id, "json_body"));
                    }
                    if end_of_stream {
                        // The response's `Value`s are held in the body budget like a request's
                        // (D216). Headers are already downstream, so a refusal aborts the response,
                        // as `translate_overflow` does. A same-wire 2xx is relayed as bytes.
                        let _heap = if upstream == t.client && (200..300).contains(&status) {
                            None
                        } else {
                            self.hold_translation_heap(&t.json_buf).inspect_err(|_| {
                                warn!(
                                    request_id = %rc.request_id,
                                    bytes = t.json_buf.len(),
                                    "translated response exceeds the body budget; aborting",
                                );
                            })?
                        };
                        translate::response_json_tools(
                            upstream,
                            t.client,
                            status,
                            &t.json_buf,
                            &t.tools,
                            t.gateway_cache,
                        )
                    } else {
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };
            if !out.is_empty() {
                if let Some(c) = rc.control.as_mut().and_then(|c| c.capture.as_mut()) {
                    c.push_resp(&out);
                }
                if let Some(cache::Pending::Fill { tap, .. }) =
                    rc.auto.as_mut().and_then(|a| a.cache.as_mut())
                {
                    tap.push(&out);
                }
            }
            if rc.tracks_terminal() {
                rc.terminal.feed(&out);
            }
            *body = Some(Bytes::from(out));
        } else if !chunk.is_empty() {
            if rc.tracks_terminal() {
                rc.terminal.feed(chunk);
            }
            // Capture tap — same passive-tap contract as the usage tail above (copy, never withhold),
            // differing only in which end it keeps. `resp_tail` keeps the *last* 64 KB because usage
            // rides the final event; capture keeps the *first* `max_bytes` because that's where the
            // answer starts. On a captured Anthropic SSE response all three buffers coexist, which is
            // fine — each is independently bounded.
            if let Some(c) = rc.control.as_mut().and_then(|c| c.capture.as_mut()) {
                c.push_resp(chunk);
            }

            if let Some(cache::Pending::Fill { tap, .. }) =
                rc.auto.as_mut().and_then(|a| a.cache.as_mut())
            {
                tap.push(chunk);
            }
        }
        Ok(None)
    }

    /// Keep pingora 0.8's retry policy for an error after the connection is up — except that a
    /// body the provider already has is never sent again.
    ///
    /// Pingora 0.9's default refuses to retry any non-idempotent method, and every LLM call is a
    /// `POST` — so the default silently turned off both retries this gateway decides for itself:
    /// the managed 429 key walk and the model-routed 5xx vendor walk. `upstream_response_filter`
    /// only marks those retryable after `body_replayable` has proven the body can be resent, which
    /// is the safety condition the new default approximates with the method.
    ///
    /// A failure with no response (a reset, an early close, a read timeout) is resent only when it
    /// happened before the upstream had the whole body ([`body_delivered`]): a reset mid-upload, a
    /// write error. After that the provider may well be generating, and billing, the answer, so
    /// sending the body again — to the same candidate or the next — duplicates the spend; the
    /// request ends instead, with a JSON 504/502 (`fail_to_proxy`). That holds for pingora's own
    /// reused-connection retry and for the `FullBody` reset retry alike. A 5xx or 429 is an
    /// answer, not a failure, and keeps walking as before.
    ///
    /// The one exception is a stream the provider refused ([`upstream_refused_stream`], D72): an
    /// HTTP/2 GOAWAY that left it above `last_stream_id`, or `RST_STREAM(REFUSED_STREAM)`. That is
    /// the provider's guarantee it processed none of the request, so it is resent — on a fresh
    /// connection to the same candidate — however much of the body went out. Routine H2
    /// connection recycling at a provider would otherwise surface as client 502s.
    ///
    /// [`retry::decide`] makes every one of these calls; this hook reports what the error says
    /// about the body and acts on the answer.
    fn error_while_proxy(
        &self,
        peer: &HttpPeer,
        session: &mut Session,
        e: Box<pingora_core::Error>,
        ctx: &mut Self::CTX,
        client_reused: bool,
    ) -> Box<pingora_core::Error> {
        use pingora_core::ErrorType as T;
        let mut e = e.more_context(format!("Peer: {peer}"));
        // A read timeout capped at the request's deadline is the deadline passing, not the
        // provider going quiet: the gateway's own decision, like those below.
        if e.etype() == &T::ReadTimedout
            && let Some(rc) = ctx.rc.as_ref()
            && rc.attempt.read_capped
        {
            return self.deadline_cut(&rc.request_id);
        }
        // Our own decisions (a 5xx / 401 vendor walk, a 429 key walk, a body cap) are made.
        if matches!(e.etype(), T::HTTPStatus(_) | T::CustomCode(..)) {
            return e;
        }
        // A client that stalled mid-upload timed its own request out (D260): retag the timeout as
        // the client's, so the walk below ends here rather than failing over with a body nobody
        // will finish, `logging` charges no breaker and bills no estimate, and `fail_to_proxy`
        // answers 408 rather than blaming the provider with a 504.
        if ctx
            .rc
            .as_ref()
            .is_some_and(|rc| client_stalled_upload(session, rc, &e))
        {
            e.as_down();
        }
        let Some(rc) = ctx.rc.as_mut() else {
            e.set_retry(false);
            return e;
        };
        // What this failure says about the body (see `retry::decide`): whether the provider may
        // have all of it, or proved it took none (an HTTP/2 refusal, a pooled HTTP/1.1 connection
        // it closed with the request unread).
        let refused = upstream_refused_stream(&e);
        let lost = retry::Lost {
            downstream: *e.esource() == pingora_core::ErrorSource::Downstream,
            responded: session.as_downstream().response_written().is_some(),
            delivered: body_delivered(session, rc, Some(&*e)),
            refused,
            goaway: upstream_goaway(&e),
            reset_unread: reset_before_reading(&e, client_reused),
            reused: client_reused,
            refused_resent: rc.leg.refused_resent,
            h2_body_unsent: h2_body_unsent(&e),
            pingora_retries: {
                let mut verdict = e.retry;
                verdict.decide_reuse(client_reused);
                verdict.retry()
            },
        };
        let walk = rc.walk_facts();
        let decision = retry::decide(&retry::Failed {
            failure: retry::Failure::Lost(lost),
            walk,
            body: body_facts(session, walk),
        });
        if lost.may_resend() && refused {
            if lost.resends_refusal() {
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    error = %e,
                    "upstream refused the stream unprocessed (GOAWAY / REFUSED_STREAM); resending once",
                );
            } else {
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    error = %e,
                    "upstream refused the stream again; treating it as a provider failure",
                );
            }
        }
        match decision {
            // The same candidate on a fresh connection, keeping its breaker permit: it was the
            // connection, not the provider.
            Decision::Resend(Retry::SameLeg) => {
                rc.leg.refused_resent |= lost.spends_resend();
                rc.leg.retry = Some(Retry::SameLeg);
                e.set_retry(true);
            }
            Decision::Resend(Retry::NextCandidate) => {
                self.state.metrics.candidate_failovers_total.inc();
                rc.fail_over(rc.auto.as_ref().map_or(0, |a| a.candidate));
                e.set_retry(true);
            }
            // A large body: pingora cannot resend it, but the parent holds it.
            Decision::Rerun(retry) => {
                e.set_retry(false);
                // A same-candidate retry is the connection's fault (or a stream refused before
                // any processing), not the provider's: give the permit back without an outcome,
                // as a small body keeps it for its own same-candidate retry.
                if matches!(retry, RelayRetry::Reset(_)) {
                    rc.leg.release();
                }
                rc.attempt.relay_abandoned =
                    full_body_ctx(session).is_some_and(|fb| fb.record(retry));
            }
            Decision::Stand(_) => e.set_retry(false),
        }
        e
    }

    /// Answer every request the gateway itself ends, after admission or during the proxy loop,
    /// with the same JSON error envelope a refusal uses: `content-type: application/json`,
    /// `{"error":{"message","type"}}`, and `x-beyond-request-id`. Pingora's default wrote a bare
    /// status with an empty body (a 500 for "every breaker open", which is a 503), which an SDK
    /// cannot parse and an oncall cannot correlate.
    ///
    /// The status follows the cause (see [`failure_response`]): a connect failure on every
    /// candidate is a 502, every candidate's breaker open a 503 with `Retry-After`, an upstream
    /// timeout a 504, a chunked body over the cap a 413. An upstream that failed after it had the
    /// whole request (the walk ends there rather than resending, see `error_while_proxy`) says so:
    /// `upstream timed out after receiving the request` (504) or `upstream failed after receiving
    /// the request` (502). A stream the provider refused twice says it was not processed (502). A
    /// client that is already gone gets nothing, and a response that already started cannot be
    /// replaced.
    async fn fail_to_proxy(
        &self,
        session: &mut Session,
        e: &pingora_core::Error,
        ctx: &mut Self::CTX,
    ) -> FailToProxy
    where
        Self::CTX: Send + Sync,
    {
        let Some((status, typ, mut msg)) = failure_response(e) else {
            return FailToProxy {
                error_code: 0,
                can_reuse_downstream: false,
            };
        };
        if session.response_written().is_some() {
            return FailToProxy {
                error_code: status,
                can_reuse_downstream: false,
            };
        }
        // The provider had the whole request: name that, so a client does not read a resend-safe
        // failure into it (the gateway did not resend; a client retry may run it twice).
        let after_delivery = *e.esource() == pingora_core::ErrorSource::Upstream
            && !matches!(
                e.etype(),
                pingora_core::ErrorType::HTTPStatus(_) | pingora_core::ErrorType::CustomCode(..)
            )
            && ctx
                .rc
                .as_ref()
                .is_some_and(|rc| body_delivered(session, rc, Some(e)));
        // A stream the provider refused (D72) was not processed, however much of the body went
        // out: say exactly that, so a client knows its own retry is safe (D91).
        let refused = upstream_refused_stream(e);
        let status = match (after_delivery && !refused, status) {
            _ if refused => {
                msg = "the provider refused the stream; it was not processed";
                502
            }
            (true, 504) => {
                msg = "upstream timed out after receiving the request";
                504
            }
            (true, _) => {
                msg = "upstream failed after receiving the request";
                502
            }
            (false, status) => status,
        };
        // A refusal raised past `request_filter` (a `CustomCode`, so neither adjustment above
        // applies to it) is answered with its own body; any other error with the static one for
        // its class.
        let body = match Refusal::of(e) {
            Some(refusal) => refusal.body(),
            None => error_body(typ, msg),
        };
        // Pingora closes the client connection after a proxy error; say so (`connection: close`),
        // as its own error responses do, so a pooled client does not send its next request into a
        // socket about to close.
        session.as_downstream_mut().set_keepalive(None);
        // One of the two sources of a gateway `Retry-After`: this one answers every 503 raised past
        // `request_filter` (a refusal there included, such as `BodyMemory`), with the soonest open
        // breaker of the walk, else 1. A refusal written in `request_filter` carries
        // `Refusal::retry_after` instead.
        let retry_after = (status == 503).then(|| {
            ctx.rc
                .as_ref()
                .and_then(|rc| rc.auto.as_ref())
                .and_then(|a| a.open_retry_after)
                .map_or(1, u64::from)
        });
        let request_id = ctx
            .held
            .request_id
            .or_else(|| ctx.rc.as_ref().map(|rc| rc.request_id))
            .unwrap_or_default();
        if let Err(write_err) = Box::pin(write_json_body(
            session,
            &request_id,
            status,
            body,
            retry_after,
        ))
        .await
        {
            warn!(
                request_id = %request_id,
                status,
                error = %write_err,
                "failed to send the error response downstream",
            );
        }
        FailToProxy {
            error_code: status,
            can_reuse_downstream: false,
        }
    }

    fn fail_to_connect(
        &self,
        _session: &mut Session,
        _peer: &HttpPeer,
        ctx: &mut Self::CTX,
        mut e: Box<pingora_core::Error>,
    ) -> Box<pingora_core::Error> {
        let Some(rc) = ctx.as_mut() else { return e };
        let walk = rc.walk_facts();
        // A connect failure moves no body byte, so the body it would resend is never a question.
        let decision = retry::decide(&retry::Failed {
            failure: retry::Failure::Connect {
                retries_left: rc.leg.addr < MAX_CONNECT_RETRIES,
            },
            walk,
            body: retry::Body::default(),
        });
        let at = rc.auto.as_ref().map_or(0, |a| a.candidate);
        match (walk, decision) {
            // The candidate's next address, or a provider-routed request's peer again: same
            // provider, same breaker permit (one address refusing is not the provider failing).
            (_, Decision::Resend(Retry::SameLeg)) => {
                rc.leg.provider.metrics.connect_retries_total.inc();
                rc.leg.addr += 1;
                rc.leg.retry = Some(Retry::SameLeg);
                if walk.is_some() {
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        candidate = at,
                        address = rc.leg.addr,
                        error = %e,
                        "upstream connect failed; trying the candidate's next address",
                    );
                } else {
                    // Without this, a partially-down provider TCP layer (or an egress-IP ban —
                    // connect is where that first bites) shows up only as extra latency on
                    // `upstream_latency_seconds`, indistinguishable from a slow model.
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.leg.provider.name.as_str(),
                        attempt = rc.leg.addr,
                        error = %e,
                        "upstream connect failed; retrying",
                    );
                }
                e.set_retry(true);
            }
            // The failure itself is recorded by `upstream_peer`'s prologue, when it moves off this
            // candidate. Recording here as well would double-count whenever there is no next
            // candidate, since `logging` would then also resolve the still-pending permit — which
            // would trip the breaker at half its configured threshold on the last candidate,
            // exactly where everything lands once the primaries are sick.
            (Some(_), decision) => {
                rc.leg.provider.metrics.connect_retries_total.inc();
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.leg.provider.name.as_str(),
                    candidate = at,
                    error = %e,
                    "upstream connect failed; trying the next candidate",
                );
                // With nowhere to go `retry` stays false, so the proxy loop stops and `logging`
                // resolves the outstanding permit against this, the last candidate.
                if decision == Decision::Resend(Retry::NextCandidate) {
                    self.state.metrics.candidate_failovers_total.inc();
                    rc.fail_over(at);
                    e.set_retry(true);
                }
            }
            // A provider-routed peer out of retries: the error stands, uncounted as a retry.
            (None, _) => {}
        }
        e
    }

    async fn logging(
        &self,
        session: &mut Session,
        e: Option<&pingora_core::Error>,
        ctx: &mut Self::CTX,
    ) {
        let Ctx { rc, held } = ctx;
        // Balance the in-flight gauge and the tenant slot taken at admission. A panic that skips
        // this is covered by `Ctx`'s drop.
        held.release_in_flight();
        held.release_tenant();
        let Some(rc) = rc.as_mut() else { return };
        let managed = rc.managed();
        // Set once the response head arrived (or from the cache hit) and never after, so one read
        // serves the whole hook.
        let upstream_status = rc.upstream_status();
        // A client that closed after its stream's terminal event reached it has the whole answer:
        // the request completed, whatever the provider's end of stream was still doing.
        let e = e.filter(|e| !closed_after_terminal(rc, e));
        // Drain-on-cancel (`crate::drain`): the client left an armed stream before its terminal
        // event, and whether the gateway then read the upstream to its end (`drained`: the usage
        // tail holds the vendor's final usage).
        let left = held.client_left.load(Ordering::Relaxed);
        let client_left = left && !rc.terminal.ended();
        let drain_end = drain_end(held.drain, left, e);
        if let Some(end) = drain_end {
            held.close_drain(end);
        }
        let drained = client_left && drain_end == Some(DrainEnd::Settled);

        // An upstream error (DNS/connect timeout, read timeout, abort) lands here with `Some(e)` but
        // no `ai.usage` row (no parseable body) — and the earlier `warn!` in `upstream_peer` only
        // fires for DNS, not connect/read failures. Log it with the full identity so "why did tenant
        // 42 get 502s for 5 minutes" is one grep on the request_id, not a reconstruction.
        if let Some(e) = e {
            warn!(
                request_id = %rc.request_id,
                tenant_id = rc.billing.identity().map_or(0, |id| id.tenant_id),
                vpc_id = rc.billing.identity().map_or(0, |id| id.vpc_id),
                provider = rc.leg.provider.name.as_str(),
                error = %e,
                "upstream request errored",
            );
        }

        // Resolve the outstanding circuit-breaker permit, if this request still owes one.
        //
        // The leg is the ledger (see `Leg`): it holds the permit from when it is claimed until it
        // is taken to resolve it, so exactly one outcome lands per `allow()`. On the model-routed
        // path a candidate switch resolves the outgoing candidate's leg in `upstream_peer`, which
        // is what stops this from double-recording against whichever candidate happened to be
        // current at the end.
        //
        // Failure = the provider is *broken*: a 5xx response, or no response at all paired with an
        // upstream error (connect/read failure). Success = the provider *answered* — 2xx/3xx, and
        // deliberately **4xx/429 too**: a 429 is a healthy provider throttling our pool key, which the
        // rate limiter and the client's `Retry-After` own, NOT a reason to cut all traffic to it.
        if rc.leg.permit.is_some() {
            match upstream_status {
                // Defensive only, so its mutants are equivalent (excluded in .cargo/mutants.toml):
                // `response_filter` sets `upstream_status` and takes the permit in the same step,
                // and nothing else sets it on a request that holds a permit (a cache hit never
                // claims one), so a pending permit here always comes with `None`.
                Some(s) if s >= 500 => rc.leg.fail(),
                Some(_) => {
                    rc.leg.succeed();
                }
                // No response head arrived. Blame the provider only when the failure actually came
                // *from* upstream. Pingora tags a client-side abort `ErrorSource::Downstream` (the
                // `into_down()` at proxy_h1.rs's downstream read/write sites), and a user hitting
                // ESC on a slow turn says nothing about the provider's health. Counting those was a
                // live bug: cancellation is routine for a coding agent, and
                // `circuit_breaker_threshold` cancellations inside `circuit_breaker_window_secs`
                // would open the breaker and 503 *everyone*. Worse, `half_open_permits` is 1, so a
                // cancel-prone request drawn as the probe reopened it every time — the breaker could
                // not recover while users were cancelling.
                //
                // Nor is a client still uploading when the request died: the upstream was waiting
                // on *our* bytes, so its read timeout or reset says the client stalled, not that
                // the provider is sick. A connect failure moved no body byte, so it still counts.
                None if is_upstream_failure(e) && !client_still_uploading(session, rc) => {
                    rc.leg.fail();
                }
                // Client went away, its upload stalled, or the request ended with no error at all:
                // no provider outcome. Give the permit back without one (D86). A success here closed
                // a half-open breaker on a probe that never heard from the provider, letting every
                // caller flood one that may still be broken; `release` returns the probe permit so
                // the next request probes instead.
                None => rc.leg.release(),
            }
        }

        let cache_hit = if matches!(
            rc.auto.as_ref().and_then(|a| a.cache.as_ref()),
            Some(cache::Pending::Hit(_))
        ) {
            match rc.auto.as_mut().and_then(|a| a.cache.take()) {
                Some(cache::Pending::Hit(h)) => Some(h),
                _ => None,
            }
        } else {
            None
        };

        // The last `USAGE_TAIL_CAP` bytes of the response, oldest first (see `UsageTail`). Short
        // responses are the whole body; long ones are rotated into order here, once. Skipped on a
        // cache hit — there is no tail; tokens come from the stored entry.
        // A free sub-resource (a token count) is not a billable call: it carries no usage block,
        // writes no billing row, and is not a usage-shape regression. On every route: a catalog
        // walk records it, a `/{provider}` route names it in the forwarded path.
        let free = rc
            .auto
            .as_ref()
            .and_then(|a| a.sub)
            .or_else(|| {
                rc.forward_path
                    .as_str()
                    .and_then(route::SubResource::of_forward_path)
            })
            .is_some_and(|sub| !sub.billed());
        // Someone gave up waiting for the response head after the provider had the whole request
        // (a long reasoning turn, a huge prompt): the client, or the gateway's own read timeout
        // with the connection still up (D130). The provider cannot tell who hung up and bills that
        // prompt either way. A request that never reached it (every breaker open, a connect
        // failure) costs nothing and stays 0, and so does a peer that closed or reset the
        // connection before answering: that is a refusal, not a wait (see `gave_up_waiting`).
        let no_head = managed
            && cache_hit.is_none()
            && upstream_status.is_none()
            && e.is_some_and(gave_up_waiting)
            && body_delivered(session, rc, e);
        let (parsed, count) = if let Some(h) = cache_hit.as_ref() {
            (Some(h.usage), Count::Reported)
        } else {
            let tail = rc.resp_tail.contiguous();
            // A relayed managed 403 whose body names the key (not the request) is that key being
            // refused: cool it so later requests start past it, as a 401 does at the head (D84).
            // This request already answered; it does not walk.
            if managed && upstream_status == Some(403) && body_names_the_key(tail) {
                self.state.metrics.key_cooled(KeyCooled::KeyNamed403);
                rc.leg.provider.mark_key_bad(rc.leg.pool_key);
            }
            // A relayed managed error that says the account is out of credit or quota (Anthropic's
            // credit-balance 400, OpenAI's insufficient_quota 429: read from the body by `Redact`)
            // is that key refused until someone pays, which waiting does not fix. Cool it as a 401
            // is cooled: later requests start past it, and a catalog walk leaves the provider out
            // once all its keys cool (D180). This request already answered. Not an abandoned
            // `FullBody` attempt: its body never reaches the client and is read only if it raced
            // the parent's drop, and a walk's 402 was already judged at the head (D258).
            if managed
                && !rc.attempt.relay_abandoned
                && rc.redact.as_ref().is_some_and(|r| r.unfunded)
            {
                let status = upstream_status.unwrap_or(0);
                self.cool_unfunded_key(&rc.request_id, &rc.leg.provider, rc.leg.pool_key, status);
            }
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
            // A managed 2xx stream that ended before its usage block: the client hung up (a
            // cancelled agent turn) or the upstream died mid-stream. The provider bills us for what
            // it generated before it noticed, so bill an estimate rather than the zero this used to
            // emit — which made "stream, then disconnect before the last event" free. Anthropic's
            // `message_start` already carries exact input and cache counts, so only the missing
            // side is estimated. See `usage`'s estimate section: input is a pre-token lower bound,
            // output a measured divisor; both err low.
            let ok_2xx = upstream_status.is_some_and(|s| (200..300).contains(&s));
            let cut_short = managed
                && rc.streaming
                && ok_2xx
                && match rc.dialect {
                    Dialect::OpenAi => parsed.is_none(),
                    // A `message_delta` that arrived but did not parse is as missing as one that
                    // never came. A usage with no token count (only a server tool's count, from a
                    // stream whose message_start was lost) is missing its tokens too: estimated.
                    Dialect::Anthropic => {
                        parsed
                            .as_ref()
                            .is_none_or(|u| u.input_tokens == 0 && u.output_tokens == 0)
                            || !usage::anthropic_stream_finished(tail)
                    }
                };
            // A non-stream 2xx that died before its `usage` (the body is last): the provider
            // generated, and bills, the whole answer; we relayed part of it. Always estimated —
            // the 2xx is the proof the provider took the request.
            let body_cut = managed && !rc.streaming && ok_2xx && parsed.is_none() && e.is_some();
            // A non-stream 2xx that ended cleanly with no usage (OpenRouter's `"usage": null` on
            // an answer that failed mid-generation, a shape change) is a turn the provider took
            // and may bill, as a finished stream without usage is: estimated, never 0/0 (D195).
            // Except a body that is only an error object (OpenRouter's error-in-200, `{"error":
            // {...}}` with no answer), the non-stream form of an error-only stream: not work we
            // were billed for. Read from the root's members, in any order (D205). The tail is the
            // whole body when the body fits in it.
            let error_only_body = !rc.streaming
                && ok_2xx
                && parsed.is_none()
                && u64::from(rc.resp_bytes) <= tail.len() as u64
                && json_is_error_only(tail);
            let body_unmetered = managed
                && !free
                && !rc.streaming
                && ok_2xx
                && parsed.is_none()
                && e.is_none()
                && !error_only_body;
            let output = if cut_short {
                usage::estimate_stream_output(tail, u64::from(rc.resp_bytes))
            } else if body_cut || body_unmetered {
                usage::estimate_body_output(tail, u64::from(rc.resp_bytes))
            } else {
                0
            };
            // The 2xx head is the provider's word that it took the request, and it bills the
            // prompt from there: a stream that went silent or died before its first event is
            // estimated too (D123), like one cut after a few. Only a 200 stream carrying nothing
            // but an error event (`overloaded_error` before any output) is not work we were billed
            // for. A finished stream whose usage we could not read (a shape change, a final event
            // we could not recover) is a turn the provider billed, never a silent 0/0.
            let started = parsed.is_some() || output > 0 || !usage::stream_carried_error(tail);
            if (cut_short && started) || body_cut || body_unmetered || no_head {
                let mut u = parsed.unwrap_or_default();
                let mut parts = EstimatedParts::default();
                if u.input_tokens == 0 {
                    u.input_tokens = input_estimate(session, rc);
                    parts.input = true;
                }
                if output > u.output_tokens {
                    u.output_tokens = output;
                    parts.output = true;
                }
                (Some(u), Count::Estimated(parts))
            } else {
                (parsed, Count::Reported)
            }
        };
        if count.estimated() {
            self.state.metrics.usage_estimated_total.inc();
        }
        // A managed 2xx response is *expected* to carry usage; `None` there means the provider's
        // usage block changed shape (a new API version, a wire change) and we're about to emit a
        // zero-token billing row that looks exactly like a (non-existent) legitimate zero-token
        // generation — silently zeroing that tenant's bill. Surface it on a counter + a warn so it
        // can be alerted on. A `None` on a 4xx/5xx (error body has no usage) is normal, not logged.
        // Cache hits never trip this: they carry the tokens stored from the fill.
        //
        // A stream that ended *cleanly* without usage is that same shape-change signal even though
        // it is now billed an estimate, so it still counts here; one cut short by an error does not.
        if (parsed.is_none() || (count.estimated() && e.is_none()))
            && let Some(id) = rc.billing.identity()
            && !free
            && cache_hit.is_none()
            && let Some(s) = upstream_status
            && (200..300).contains(&s)
        {
            self.state.metrics.usage_parse_errors_total.inc();
            warn!(
                request_id = %rc.request_id,
                tenant_id = id.tenant_id,
                provider = rc.leg.provider.name.as_str(),
                dialect = ?rc.dialect,
                stream = rc.streaming,
                status = s,
                "managed 2xx response ended cleanly without parseable usage; billing an estimate or zero",
            );
        }
        let mut usage = parsed.unwrap_or_default();
        if cache_hit.is_none() {
            merge_taps(rc, &mut usage);
        }
        // Writes caused by breakpoints the gateway added bill as input (see
        // `Usage::bill_gateway_cache_writes`). A cache hit replays the fill's already-billed usage.
        if cache_hit.is_none()
            && rc
                .auto
                .as_ref()
                .and_then(|a| a.translate.as_ref())
                .is_some_and(|t| t.gateway_cache)
        {
            usage.bill_gateway_cache_writes(usage.wire.unwrap_or(rc.dialect));
        }
        // A client's 1-hour breakpoints, carried onto Chat Completions (the OpenRouter failover):
        // billed at the 1-hour rate upstream, reported with no TTL split (see
        // `translate::chat_cache_is_1h`).
        if cache_hit.is_none()
            && rc
                .auto
                .as_ref()
                .and_then(|a| a.translate.as_ref())
                .is_some_and(|t| t.cache_1h)
        {
            usage.attribute_cache_writes_1h();
        }

        let m = &self.state.metrics;
        if cache_hit.is_some() {
            m.cache_hits_total.inc();
        }
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
        if cache_hit.is_none() {
            rc.leg
                .provider
                .metrics
                .upstream_latency_seconds
                .observe(elapsed.as_secs_f64());
            // Balance the `active_streams` increment from `response_filter` (or `Ctx`'s drop does).
            held.release_stream();
        }

        // Emit the priced usage row on a dedicated target — **managed only**. The event is an
        // identity-keyed billing record (logfwd/OTLP ships `ai.usage` → ClickHouse → beyond, which
        // batches it for invoicing); BYO carries no Beyond identity, so a BYO event would be a billing row
        // with `tenant_id=0` — unbillable, unattributable, and a footgun for any consumer that sums
        // without filtering it out. Aggregate gateway throughput (incl. BYO) is already covered by
        // the Prometheus metrics above, which is the right tool for non-billing observability.
        // Whether the request writes a row at all, and what the row says of how it ended, is its
        // `Ending`'s (see `Ending::row`).
        if let Some(&id) = rc.billing.identity()
            && let Some(end) = Ending::of(&EndFacts {
                relay_abandoned: rc.attempt.relay_abandoned,
                free,
                cache_hit: cache_hit.is_some(),
                status: upstream_status,
                fault: e.map(Fault::of),
                client_left,
                called: rc.upstream_phase != UpstreamPhase::None,
                drained,
                count,
            })
            .row()
        {
            // What the client asked for that changes the price (read before the borrows below).
            let requested = requested_knobs(session, rc);
            // Emit BOTH models. `model` is the one the *provider* resolved + billed (echoed in its
            // response) — the key for pricing AND for reconciling against the provider's invoice,
            // which itemizes by the pinned snapshot. `requested_model` is the alias the client sent —
            // product analytics ("what they asked for") and a fallback rate when a snapshot is newer
            // than the downstream price table. They're equal when the response carried no model (e.g.
            // an error body), where `model` falls back to the request alias. Both sanitized.
            let billed = rc.resp_model_scanner.take_model().map(sanitize_model);
            // The catalog name this request routed on — `None` for a provider-routed request.
            // Derived rather than stored: it is the catalog row's own `&'static` name.
            let routed_model = if let Some(h) = cache_hit.as_ref() {
                h.routed_model
            } else {
                rc.auto.as_ref().map(|a| a.route.model)
            };

            // What the client *asked for*.
            //
            // On the model-routed path that is the catalog name from `x-beyond-model` if present,
            // else the body's root `model` — **not** the id we splice for the serving candidate.
            // The body's value is overwritten with that candidate's id before the request leaves,
            // so a discarded spelling determines nothing; reporting it as "requested" was a leftover
            // from an earlier design that did not rewrite bodies. On the provider-routed path the
            // body is untouched and is exactly what was asked for, so it stays the answer there.
            let requested_model = if let Some(h) = cache_hit.as_ref() {
                h.requested_model.as_ref()
            } else {
                routed_model.unwrap_or(rc.model.as_str())
            };

            // A model-routed client should send the same id in the header (when they send one) and
            // the body; nothing enforces it, because the route is chosen from the header before the
            // body is rewritten. It is harmless — the body is overwritten either way — but it means
            // the client believes it asked for something it did not get, which is a client bug
            // worth being able to see. Counted rather than logged per request: a client that always
            // disagrees would otherwise produce one warn line per request forever. Header still
            // wins; a headerless walk has nothing to disagree with.
            if cache_hit.is_none()
                && routed_model.is_some_and(|r| !rc.model.is_empty() && rc.model != r)
            {
                self.state.metrics.model_header_body_mismatch_total.inc();
            }

            // Prefer the id the provider echoed (the pinned snapshot it actually billed); fall back
            // to what was asked for when the response carried none — an error body, say.
            let billed_model = if let Some(h) = cache_hit.as_ref() {
                h.billed_model.as_ref()
            } else {
                billed
                    .as_deref()
                    .filter(|m| !m.is_empty())
                    .unwrap_or(requested_model)
            };
            // The catalog row to price this row at. A model-routed row already names it; a
            // provider-routed one resolves it from what the provider echoed or the client asked for.
            let price_model = routed_model.or_else(|| price_model(billed_model, requested_model));
            // Absent when no provider was called (every candidate's breaker open): `rc.leg.provider`
            // is then just the walk's seed, and naming it would bill a call that never happened.
            let usage_provider = match cache_hit.as_ref() {
                Some(h) => Some(h.provider.as_ref()),
                None => (rc.upstream_phase != UpstreamPhase::None)
                    .then_some(rc.leg.provider.name.as_str()),
            };
            let usage_stream = cache_hit
                .as_ref()
                .map(|h| h.streaming)
                .unwrap_or(rc.streaming);
            // The endpoint that served: absent on a cache hit and when no provider was called.
            let called = cache_hit.is_none() && usage_provider.is_some();
            let upstream_model = called
                .then(|| {
                    rc.auto
                        .as_ref()
                        .and_then(|a| a.candidate_at(a.candidate))
                        .map_or(rc.model.as_str(), |c| c.upstream_model)
                })
                .filter(|m| !m.is_empty())
                .map(|m| sanitize_model(m.to_owned()));
            let upstream_host = called.then_some(rc.leg.provider.host.as_str());
            let upstream_path = called.then(|| {
                let p = rc
                    .forward_path
                    .as_str()
                    .unwrap_or_else(|| session.req_header().uri.path());
                p.split_once('?').map_or(p, |(path, _)| path)
            });
            let price_variant = match (usage_provider, upstream_model.as_deref(), upstream_host) {
                (Some(p), Some(m), Some(h)) if called => price_variant(p, m, h),
                _ => None,
            };
            let upstream_request_id = rc
                .billing
                .taps()
                .and_then(|t| t.request_id)
                .filter(|_| called);
            let server_tools = usage.server_tools.to_row();
            let upstream_may_continue =
                upstream_may_continue(usage_provider, end.outcome, usage_stream, end.drained);
            let estimate_excludes = end.count.parts().excludes(&usage);
            let upstream_cost_usd = usage.upstream.cost_e10.map(usage::e10_to_usd);
            let upstream_inference_cost_usd =
                usage.upstream.inference_cost_e10.map(usage::e10_to_usd);
            let upstream_tool_cost_usd = usage.upstream.tool_cost_e10.map(usage::e10_to_usd);
            // The request's start, whole seconds UTC: what time-of-day tiers (DeepSeek's off-peak)
            // are decided on. Logged, so a repricer reads the same second.
            let start_unix_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
                .saturating_sub(elapsed.as_secs());
            // Price the row from the very facts it logs (the contract is `providers::pricing`;
            // `crates/providers/ARCHITECTURE.md`, "Pricing contract"), typed: the numbers, not
            // their log text, which a repricer parses back to the same inputs (`usage::vendor`
            // tests hold the two equal). Pure and allocation-free: a few hundred nanoseconds,
            // after the response.
            let priced = providers::pricing::price(&providers::pricing::UsageRow {
                price_model,
                provider: usage_provider,
                price_variant,
                usage_estimated: end.count.estimated(),
                upstream_may_continue,
                cache_hit: end.cache_hit,
                unix_secs: start_unix_secs,
                ..usage.pricer_row(rc.dialect)
            });
            let (price_status, price_reason) = match &priced {
                Ok(p) => (p.status.as_str(), None),
                Err(reason) => {
                    self.state.metrics.usage_unpriced_total.inc();
                    ("unpriced", Some(reason.as_str()))
                }
            };
            let priced = priced.ok();
            info!(
                target: "ai.usage",
                request_id = %rc.request_id,
                tenant_id = id.tenant_id,
                vpc_id = id.vpc_id,
                key_id = id.key_id,
                provider = usage_provider,
                model = billed_model,
                requested_model,
                // Present only for a model-routed request, so the billing row says *how* it was
                // routed as well as what ran. Equal to `requested_model` by construction on that
                // path; its value is marking the route, not carrying a second id. `&'static` from
                // the catalog and charset-checked by a catalog test, so it needs no `sanitize_model`.
                routed_model,
                // The catalog row id the row prices at (see `price_model`); absent when unpriced.
                price_model,
                stream = usage_stream,
                cache_hit = end.cache_hit,
                // The provider's HTTP status, absent when no response head arrived (and on a cache
                // hit, which made no call); and what became of the request (see `Ending`).
                upstream_status = cache_hit.is_none().then_some(upstream_status).flatten(),
                outcome = end.outcome,
                // True when the stream was cut short before its usage block and the token counts
                // below are the gateway's estimate, not the provider's report. Estimates err low.
                usage_estimated = end.count.estimated(),
                // `drained`: the client cancelled and the counts are still the provider's own,
                // read from the stream the gateway finished after it left (`crate::drain`).
                // Absent otherwise.
                usage_settled = end.usage_settled(),
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                cache_read_tokens = usage.cache_read_tokens,
                cache_write_tokens = usage.cache_write_tokens,
                // Which convention `input_tokens` follows: `openai` includes cache reads (and
                // OpenRouter's cache writes), `anthropic` excludes both. The same prompt for the same
                // catalog row reports different `input_tokens` on the two wires; a consumer
                // normalizes with this, and the existing fields keep their meaning.
                usage_wire = usage.wire.unwrap_or(rc.dialect).as_str(),
                // Priced variants and per-call fees (see `usage::Usage`). The 1-hour writes are a
                // subset of `cache_write_tokens`, not additional to them.
                cache_write_1h_tokens = usage.cache_write_1h_tokens,
                // Cache writes from breakpoints the gateway added, already in `input_tokens` and
                // not in `cache_write_tokens` (the pricer bills them as the 5-minute writes the vendor charges).
                // For reconciling against the provider's usage, which calls them cache writes.
                gateway_cache_write_tokens = usage.gateway_cache_write_tokens,
                server_tool_calls = usage.server_tool_calls,
                service_tier = usage.service_tier.as_deref(),
                // --- The row contract's additions (ARCHITECTURE.md, "The `ai.usage` row"). ---
                // What the client asked for that changes the price, beside what was served.
                requested_service_tier = requested.service_tier.as_deref(),
                speed = usage.speed.as_deref(),
                requested_speed = requested.speed.as_deref(),
                inference_geo = usage.inference_geo.as_deref(),
                requested_inference_geo = requested.inference_geo.as_deref(),
                requested_provider_routing = requested.provider_routing.as_deref(),
                requested_plugins = requested.plugins.as_deref(),
                requested_container = requested.container.as_deref(),
                // Every server-side tool by kind, `kind=count,…` (nonzero only).
                server_tools = server_tools.as_deref(),
                container_id = usage.upstream.container_id.as_deref(),
                // The concrete endpoint that served, and which of its prices applies.
                upstream_model = upstream_model.as_deref(),
                upstream_host,
                upstream_path,
                price_variant,
                served_by = usage.upstream.served_by.as_deref(),
                // The vendor's own ids and price, for reconciling the row against it.
                upstream_generation_id = usage.upstream.generation_id.as_deref(),
                upstream_request_id = upstream_request_id.as_deref(),
                upstream_cost_usd = upstream_cost_usd.as_deref(),
                upstream_inference_cost_usd = upstream_inference_cost_usd.as_deref(),
                upstream_tool_cost_usd = upstream_tool_cost_usd.as_deref(),
                upstream_byok = usage.upstream.byok,
                upstream_may_continue,
                // --- The price: authoritative (crates/providers/ARCHITECTURE.md, "Pricing
                // contract"). The facts above are kept for audit and repricing. An unpriced row
                // carries `price_status=unpriced` and its reason, never a zero.
                rate_version = providers::rates::RATE_VERSION,
                price_status,
                price_reason,
                cost_micros = priced.map(|p| p.cost.micros),
                price_micros = priced.map(|p| p.price.micros),
                cost_basis = priced.map(|p| p.basis.as_str()),
                cost_detail = priced.map(|p| tracing::field::display(p.cost)),
                price_detail = priced.map(|p| tracing::field::display(p.price)),
                start_unix_secs,
                // Which counts are the gateway's estimate, and what the estimate cannot see.
                usage_estimated_parts = end.count.parts().parts(),
                usage_estimate_excludes = estimate_excludes,
                // `Some(0)` (reported, none used) vs `None` (not reported at all — an unreasoning
                // model, or a provider that doesn't surface it) matters and is unrecoverable once this
                // line ships, so it's logged as `?` (Debug) rather than collapsed to a bare `0`.
                reasoning_tokens = ?usage.reasoning_tokens,
                latency_ms = elapsed.as_millis() as u64,
                // Caller-supplied tags from `x-beyond-metadata`, already validated and canonically
                // re-serialized by `control` — a JSON object, so `GROUP BY metadata['feature']`
                // downstream answers "which feature is burning money" against the same row that
                // carries the tokens. Absent (not `null`) when the caller sent none.
                metadata = rc.control.as_ref().and_then(|c| c.metadata.as_deref()),
                "usage"
            );

            // Payload capture, on its own target so a stalled payload sink can never delay or drop a
            // billing row (see `main::init_tracing`, which gives the two targets different writers).
            // Correlated by `request_id`, which is also on the line above and in the response's
            // `x-beyond-request-id` header — so a user quoting that id resolves straight to their
            // conversation with no join table.
            if cache_hit.is_none()
                && let Some(cap) = rc.control.as_ref().and_then(|c| c.capture.as_ref())
            {
                m.captures_total.inc();
                m.capture_bytes_total.inc_by(cap.bytes() as u64);
                info!(
                    target: "ai.payload",
                    request_id = %rc.request_id,
                    tenant_id = id.tenant_id,
                    provider = rc.leg.provider.name.as_str(),
                    model = billed_model,
                    stream = rc.streaming,
                    status = upstream_status,
                    metadata = rc.control.as_ref().and_then(|c| c.metadata.as_deref()),
                    // What the client sent, before either body rewrite (see `request_body_filter`).
                    request_body = cap.req_str(),
                    response_body = cap.resp_str(),
                    // Truncation and completeness are separate facts and both are load-bearing.
                    // `truncated` means we hit the byte cap; `complete` means the upstream actually
                    // finished. A capture that reads as whole when it isn't produces confident wrong
                    // conclusions during the incident this feature exists to serve — and "the stream
                    // died at token 400" is frequently the answer, so a partial is kept, not dropped.
                    request_truncated = cap.req_truncated(),
                    response_truncated = cap.resp_truncated(),
                    complete = e.is_none() && !client_left,
                    "payload"
                );
            }

            // Fill: complete 2xx only. Client abort, 4xx/5xx, and truncation are all skips — a
            // partial or error body must never be replayed as a success.
            if e.is_none()
                && !end.count.estimated()
                && upstream_status.is_some_and(|s| (200..300).contains(&s))
                && let Some(cache::Pending::Fill {
                    key,
                    tap,
                    content_type,
                }) = rc.auto.as_mut().and_then(|a| a.cache.take())
                && !tap.truncated()
                && let Some(body) = tap.complete_body()
                && let Some(store) = &self.state.cache
            {
                store.insert(
                    key,
                    cache::CachedResponse {
                        status: upstream_status.unwrap_or(200),
                        content_type: content_type.unwrap_or_else(|| "application/json".into()),
                        body: Bytes::copy_from_slice(body),
                        usage: usage.for_cache(),
                        billed_model: billed_model.to_owned().into_boxed_str(),
                        requested_model: requested_model.to_owned().into_boxed_str(),
                        routed_model,
                        provider: rc.leg.provider.name.clone().into_boxed_str(),
                        streaming: rc.streaming,
                    },
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "proxy_row_tests.rs"]
mod row_tests;

/// The `outcome` a request with these facts logs, for the tests that pin each one; `logging`
/// builds its [`Ending`] from every fact at once.
#[cfg(test)]
fn outcome(
    rc: &RequestCtx,
    e: Option<&pingora_core::Error>,
    cache_hit: bool,
    client_left: bool,
) -> &'static str {
    let ending = Ending::of(&EndFacts {
        cache_hit,
        status: rc.upstream_status(),
        fault: e.map(Fault::of),
        client_left,
        called: rc.upstream_phase != UpstreamPhase::None,
        ..EndFacts::default()
    });
    ending.row().map_or("", |r| r.outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::metrics::ProviderMetrics;
    use crate::route::AuthScheme;

    /// `max_tokens` is respelled for native OpenAI Chat; with both spellings the explicit
    /// `max_completion_tokens` stays and the `max_tokens` member goes, whatever its value and
    /// position, leaving valid JSON.
    #[test]
    fn max_tokens_is_respelled_max_completion_tokens() {
        let rename = |body: &str| {
            let mut buf = body.as_bytes().to_vec();
            let scan = peek::scan_buffered(&buf);
            let changed = rename_max_tokens(&mut buf, &scan.limit_keys);
            let out = String::from_utf8(buf).unwrap();
            serde_json::from_str::<serde_json::Value>(&out).expect(&out);
            (out, changed)
        };
        for (body, want, changed) in [
            (
                r#"{"model":"o3","max_tokens":256}"#,
                r#"{"model":"o3","max_completion_tokens":256}"#,
                true,
            ),
            (
                r#"{"max_tokens" : null, "model":"o3"}"#,
                r#"{"max_completion_tokens" : null, "model":"o3"}"#,
                true,
            ),
            (
                r#"{"max_tokens":256, "max_completion_tokens":512}"#,
                r#"{ "max_completion_tokens":512}"#,
                true,
            ),
            (
                r#"{"max_completion_tokens":512, "max_tokens" : {"x":[1,"}"]} }"#,
                r#"{"max_completion_tokens":512 }"#,
                true,
            ),
            (
                r#"{"max_completion_tokens":512,"messages":[{"max_tokens":1}]}"#,
                r#"{"max_completion_tokens":512,"messages":[{"max_tokens":1}]}"#,
                false,
            ),
        ] {
            assert_eq!(rename(body), (want.to_owned(), changed), "{body}");
        }
    }

    /// Every limit over the cap is cut to it, back to front so offsets hold; one under it, or a
    /// cap of zero, is left alone. A value past `u64` is over any cap.
    #[test]
    fn output_limits_are_capped_never_raised() {
        let cap = |body: &str, max: u32| {
            let mut buf = body.as_bytes().to_vec();
            let scan = peek::scan_buffered(&buf);
            let changed = clamp_output_limits(&mut buf, &scan.limit_spans, max);
            (String::from_utf8(buf).unwrap(), changed)
        };
        assert_eq!(
            cap(
                r#"{"max_tokens":64000,"max_completion_tokens":99999999999999999999999}"#,
                16384
            ),
            (
                r#"{"max_tokens":16384,"max_completion_tokens":16384}"#.to_owned(),
                true
            )
        );
        assert_eq!(
            cap(r#"{"max_output_tokens":100}"#, 16384),
            (r#"{"max_output_tokens":100}"#.to_owned(), false)
        );
        assert_eq!(
            cap(r#"{"max_tokens":64000}"#, 0),
            (r#"{"max_tokens":64000}"#.to_owned(), false)
        );
    }

    /// A minimal `RequestCtx` for exercising the body-phase logic without a running proxy.
    pub(super) fn test_ctx(stream_options: bool) -> RequestCtx {
        let provider = Provider::resolve(
            "openai",
            "api.openai.com:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::Bearer,
            &["sk-pool"],
            ProviderMetrics::disconnected(),
            None,
        );
        RequestCtx {
            billing: Billing::of(Some(Identity::unverified_for_test(42, 7, None))),
            dialect: Dialect::OpenAi,
            leg: Leg::new(Arc::new(provider), None, 0),
            forward_path: ForwardPath::Inbound,
            model: String::new(),
            attempt: Attempt::new(Vec::new()),
            resp_model_scanner: peek::ModelScanner::for_response(),
            streaming: false,
            resp_tail: UsageTail::default(),
            resp_head: Vec::new(),
            upstream_status: None,
            plan: BodyPlan::new(stream_options, false, false, false),
            start: Instant::now(),
            deadline: crate::deadline::NONE,
            auto: None,
            control: None,
            request_id: RequestId::new(),
            tally_eager: false,
            resp_bytes: 0,
            upstream_phase: UpstreamPhase::None,
            redact: None,
            terminal: TerminalTracker::default(),
            signed: None,
        }
    }

    /// The Anthropic stream head is bounded at `USAGE_HEAD_CAP` however the stream is chunked: the
    /// chunk that crosses the cap is cut there, and later chunks add nothing. Other responses keep
    /// no head.
    #[test]
    fn the_usage_head_holds_at_most_its_cap() {
        let mut rc = test_ctx(false);
        rc.streaming = true;
        rc.keep_usage_head(&[b'a'; 1000]);
        assert!(rc.resp_head.is_empty(), "an OpenAI stream keeps no head");
        rc.dialect = Dialect::Anthropic;
        rc.keep_usage_head(&[b'a'; 5 * 1024]);
        assert_eq!(rc.resp_head.len(), 5 * 1024);
        rc.keep_usage_head(&[b'b'; 10 * 1024]);
        assert_eq!(rc.resp_head.len(), USAGE_HEAD_CAP);
        assert_eq!(rc.resp_head.last(), Some(&b'b'));
        rc.keep_usage_head(b"more");
        assert_eq!(rc.resp_head.len(), USAGE_HEAD_CAP);
    }

    /// The replay contract, exercised directly: `upstream_peer` starts a fresh attempt before pingora
    /// re-feeds the buffered prefix through `request_body_filter`, so attempt 2 sees the body once.
    ///
    /// This is the bug in miniature. Attempt 1 buffers a prefix and is cut off before end-of-stream,
    /// so `req_buf` is never drained. Pingora then replays that prefix from byte 0. Without the
    /// reset the two concatenate, the splice is planned against the first copy, and the upstream
    /// gets a body with a duplicated fragment — which it rejects with a `400` that `logging` records
    /// as a breaker *success*, so nothing surfaces it as ours.
    ///
    /// Covered here rather than end-to-end because reproducing it over a socket needs the upstream
    /// to die in the window where the gateway holds a *partial* body — a timing-dependent race. The
    /// e2e suite proves the surrounding mechanism (pingora really does retry a reused connection and
    /// really does replay through this filter); this pins what the reset itself must guarantee.
    #[test]
    fn resetting_the_body_phase_makes_a_replayed_prefix_idempotent() {
        let body =
            br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","stream":true}"#;
        let (prefix, rest) = body.split_at(30);

        // Attempt 1: a prefix arrives, then the upstream dies before end-of-stream. The chunk goes
        // through the production append step, as `request_body_filter` feeds it.
        let mut rc = test_ctx(true);
        assert!(rc.take_request_chunk(prefix));
        assert!(
            !rc.attempt.req_buf.is_empty(),
            "precondition: the partial body is still held, since end_of_stream never came",
        );

        // `upstream_peer` runs before any body byte of attempt 2.
        rc.begin_attempt();
        assert_eq!(
            rc.attempt.upload.0, 0,
            "the size guard must not double-count the replayed prefix",
        );

        // Attempt 2: pingora replays from byte 0 through the same filter, then the remainder.
        assert!(rc.take_request_chunk(prefix));
        assert!(rc.take_request_chunk(rest));

        assert_eq!(
            rc.attempt.req_buf, body,
            "the retried attempt must send the body exactly once",
        );
        assert_eq!(
            rc.attempt.upload.0,
            body.len(),
            "the replay counts the body once, as the client sent it",
        );
        let scan = peek::scan_buffered(&rc.attempt.req_buf);
        assert_eq!(
            scan.model.as_deref(),
            Some("gpt-4o"),
            "and the model must still be extractable — a duplicated prefix offsets the scanner's \
             depth permanently, shipping `requested_model` empty",
        );
        assert!(
            scan.inject_at.is_some(),
            "the stream_options splice must still be planned against a well-formed root object",
        );
    }

    /// A retried attempt replays the body prefix through the capture tap, so the reset must clear
    /// the capture as well, or the captured request holds the prefix twice.
    /// claim: O2
    /// defect: D285
    #[test]
    fn a_retried_attempt_captures_the_request_body_once() {
        let body =
            br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","stream":true}"#;
        let (prefix, rest) = body.split_at(30);

        let mut rc = test_ctx(true);
        rc.control = Some(Box::new(RequestControl {
            metadata: None,
            capture: Some(CaptureBufs::new(4096)),
        }));

        // Attempt 1: a prefix arrives through the production append step (captured and buffered),
        // then the upstream dies.
        assert!(rc.take_request_chunk(prefix));
        rc.begin_attempt();

        // Attempt 2: pingora replays from byte 0 through the same step, then the remainder.
        assert!(rc.take_request_chunk(prefix));
        assert!(rc.take_request_chunk(rest));
        let captured = rc.control.as_ref().unwrap().capture.as_ref().unwrap();
        assert_eq!(
            captured.req_str(),
            std::str::from_utf8(body).ok(),
            "the capture holds the replayed prefix twice"
        );
    }

    /// A 2xx whose body turns out to be an error is counted against the breaker permit the head
    /// resolved, so a late verdict cannot reopen a breaker that another request's probe owns by then.
    /// claim: REL-6
    /// defect: D286
    #[test]
    fn a_late_body_verdict_cannot_reopen_a_breaker_another_probe_owns() {
        static NOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1_000);
        fn clock() -> u64 {
            NOW.load(std::sync::atomic::Ordering::Relaxed)
        }
        let breaker = crate::circuit_breaker::CircuitBreaker::with_clock(
            crate::circuit_breaker::CircuitBreakerConfig::consecutive(1)
                .reset_timeout(std::time::Duration::from_secs(30))
                .half_open_permits(1),
            clock,
        );
        // This request was admitted while the breaker was closed, and its head was a 200.
        let admitted = breaker.allow().expect("a closed breaker admits");
        let admitted = breaker.record_success_for(admitted);
        // Another request trips the breaker; after the reset timeout its probe owns the breaker.
        let trip = breaker.allow().expect("a closed breaker admits");
        breaker.record_failure_for(trip);
        NOW.store(1_030, std::sync::atomic::Ordering::Relaxed);
        let _probe = breaker.allow().expect("the reset timeout admits one probe");

        let mut rc = test_ctx(true);
        rc.leg.provider = Arc::new(Provider::resolve(
            "openai",
            "api.openai.com:443".to_string(),
            Dialect::OpenAi,
            AuthScheme::Bearer,
            &["sk-pool"],
            ProviderMetrics::disconnected(),
            Some(breaker),
        ));
        let row = route::model_route("gpt-4o").expect("a catalog row");
        rc.auto = Some(Box::new(ModelRouting {
            route: row,
            sub: None,
            candidate: 0,
            usable: 1,
            walk: control::Walk::identity(row.candidates.len()),
            arms: row.candidates,
            session_field: None,
            attempt_start: Instant::now(),
            cache: None,
            translate: None,
            health: Some(PendingHealth {
                prefix: Vec::new(),
                permit: Some(admitted),
            }),
            addrs: 0,
            attempted: true,
            open_retry_after: None,
            priority: false,
        }));
        let state: &'static Arc<crate::state::GatewayState> = Box::leak(Box::new(
            crate::state::GatewayState::new(
                crate::config::AiConfig::default().validate().unwrap(),
                crate::state::test_metrics(),
            )
            .unwrap(),
        ));

        AiProxy { state }.settle_health(&mut rc, br#"{"error":{"message":"overloaded"}}"#, true);

        assert!(
            matches!(
                rc.leg.provider.breaker.as_ref().unwrap().state(),
                crate::circuit_breaker::CircuitState::HalfOpen { .. }
            ),
            "a late body verdict reopened a breaker owned by another request's probe"
        );
    }

    /// Without the reset, the same sequence corrupts the body — so the test above is not vacuous.
    #[test]
    fn a_replayed_prefix_without_the_reset_corrupts_the_body() {
        let body =
            br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","stream":true}"#;
        let (prefix, rest) = body.split_at(30);

        let mut rc = test_ctx(true);
        rc.attempt.req_buf.extend_from_slice(prefix);
        rc.attempt.upload.feed(prefix.len());
        // ...no reset...
        rc.attempt.req_buf.extend_from_slice(prefix);
        rc.attempt.req_buf.extend_from_slice(rest);

        assert_ne!(
            rc.attempt.req_buf, body,
            "the duplicated prefix must be observable"
        );
        assert!(
            serde_json::from_slice::<serde_json::Value>(&rc.attempt.req_buf).is_err(),
            "and the result is not valid JSON — this is the 400 the provider returns",
        );
        assert_eq!(
            rc.attempt.upload.0,
            prefix.len(),
            "and the size guard has double-counted",
        );
    }

    /// `RequestCtx` is touched on every hook and, for a streaming response, once per response
    /// chunk, so its size is a real cost rather than bookkeeping. Growing it by 64 bytes to hold the
    /// model-routing fields inline produced a reproducible ~2.5% regression on the
    /// `managed_sse_latency` bench — non-streaming was unaffected, which is the signature of a
    /// per-chunk cost. Boxing that state (see `ModelRouting`) bought it back.
    ///
    /// A ceiling rather than an equality: padding and field order are the compiler's business, and a
    /// few bytes either way is not what this guards. What it guards is someone adding a `String` or
    /// an `Instant` here without noticing that the cost is paid per chunk on every stream.
    /// `key_id` (identity, every managed request) is why 384 became 416 — do not spend that slack
    /// on route-specific state.
    #[test]
    fn a_2xx_body_that_is_an_error_object_is_told_apart_from_an_answer() {
        // Non-streaming.
        let json = |b: &str| body_reports_error(b.as_bytes(), false);
        assert_eq!(
            json(r#"{"error":{"message":"overloaded","code":502}}"#),
            Some(true)
        );
        assert_eq!(json(r#" { "type" : "error", "error": {}}"#), Some(true));
        assert_eq!(
            json(r#"{"id":"chatcmpl-1","object":"chat.completion"}"#),
            Some(false)
        );
        assert_eq!(json(r#"{"type":"message","id":"msg_1"}"#), Some(false));
        assert_eq!(json(r#"{"err"#), None, "the first key has not ended");
        assert_eq!(json(""), None);
        assert_eq!(json("not json"), Some(false));
        // Streaming: the first data event decides; comments, ids and blank lines are skipped.
        let sse = |b: &str| body_reports_error(b.as_bytes(), true);
        assert_eq!(sse("data: {\"error\":{\"message\":\"x\"}}\n\n"), Some(true));
        assert_eq!(
            sse("event: error\ndata: {\"type\":\"error\",\"error\":{}}\n\n"),
            Some(true)
        );
        assert_eq!(
            sse(": OPENROUTER PROCESSING\n\ndata: {\"error\":1}\n\n"),
            Some(true)
        );
        assert_eq!(sse("data: {\"id\":\"c\",\"choices\":[]}\n\n"), Some(false));
        assert_eq!(
            sse("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"),
            Some(false)
        );
        assert_eq!(sse("data: [DONE]\n\n"), Some(false));
        assert_eq!(sse(": keepalive\n"), None, "only a comment so far");
        assert_eq!(sse("data: {\"err"), None);
    }

    /// Building a request's context touches no shared reference count: every worker would otherwise
    /// bounce the gateway state's `Arc` cache line on every request, including every fast reject.
    /// claim: S1
    /// defect: D92
    #[test]
    fn new_ctx_pays_no_arc_clone() {
        let state = GatewayState::new(
            crate::config::AiConfig::default().validate().unwrap(),
            crate::state::test_metrics(),
        )
        .unwrap();
        let proxy = AiProxy::new(state.clone());
        let before = Arc::strong_count(&state);
        let ctx = proxy.new_ctx();
        assert_eq!(
            Arc::strong_count(&state),
            before,
            "new_ctx cloned the state Arc"
        );
        drop(ctx);
    }

    /// The header sweep removes every match, past its stack batch and past its name slot, and keeps
    /// everything else.
    #[test]
    fn remove_headers_where_removes_every_match_in_stack_batches() {
        let mut req =
            pingora::http::RequestHeader::build(http::Method::POST, b"/v1/x", None).unwrap();
        let long = format!("x-{}", "l".repeat(80));
        for i in 0..40 {
            req.insert_header(format!("x-drop-{i}"), "v").unwrap();
        }
        req.insert_header(long.clone(), "v").unwrap();
        req.insert_header("content-type", "application/json")
            .unwrap();
        req.insert_header("accept", "*/*").unwrap();
        remove_headers_where(&mut req, |n| n.starts_with("x-"));
        let left: Vec<&str> = req.headers.keys().map(http::HeaderName::as_str).collect();
        assert_eq!(left.len(), 2, "{left:?}");
        assert!(left.contains(&"content-type") && left.contains(&"accept"));
    }

    /// Each pool key's echo searcher is built once at boot, for exactly the bare key a provider
    /// would echo (D92): `response_filter` and `Redact` borrow it rather than rebuild it per
    /// response or per chunk.
    #[test]
    fn a_pool_key_carries_its_boot_built_searcher() {
        let p = Provider::resolve(
            "openai",
            "api.openai.com:443".to_owned(),
            Dialect::OpenAi,
            AuthScheme::Bearer,
            &["sk-pool-a", "sk-pool-b"],
            ProviderMetrics::disconnected(),
            None,
        );
        for (auth, key) in p.pool_auth.iter().zip(["sk-pool-a", "sk-pool-b"]) {
            assert_eq!(auth.key(), key);
            assert_eq!(auth.finder().needle(), key.as_bytes());
        }
    }

    #[test]
    fn request_ctx_stays_small_enough_to_be_cheap_per_chunk() {
        let size = std::mem::size_of::<RequestCtx>();
        assert!(
            // 448: the verified identity is held inline in `Billing::Managed` beside one lazily
            // boxed set of billing taps (`None` on BYO and until a tap first fires), and the
            // per-attempt and per-leg state packs into `Attempt` and `Leg` with the small fields.
            size <= 448,
            "RequestCtx grew to {size} bytes (ceiling 448). It is touched once per response chunk \
             on a stream — if the new state is only needed on one route, box it the way \
             `ModelRouting` is rather than paying for it on every request.",
        );
    }

    /// A model-routed request takes its dialect from **the serving candidate's path**: the row's
    /// wire is the client's default, and a mixed-wire row (Claude → OpenRouter Chat Completions)
    /// would otherwise parse a Chat Completions reply with the Anthropic extractor, or the reverse,
    /// and emit a zero-token billing row. The assertions below check the derivation from the path
    /// only; the provider's own wire is not what they exercise.
    #[test]
    fn a_model_route_takes_its_dialect_from_the_serving_candidate_path() {
        let openrouter = providers::by_id(providers::ProviderId::OpenRouter);
        assert_eq!(openrouter.wire, Dialect::OpenAi, "premise");

        let row = route::model_route("claude-opus-4-8").expect("seed row");
        assert_eq!(row.wire, Dialect::Anthropic, "premise: client default");
        let primary = row.candidates[0];
        let fallback = *row.candidates.last().expect("OpenRouter arm");
        assert_eq!(primary.provider, providers::ProviderId::Anthropic);
        assert_eq!(fallback.provider, providers::ProviderId::OpenRouter);
        assert_eq!(
            route::Endpoint::of_upstream_path(primary.path),
            route::Endpoint::Messages,
        );
        assert_eq!(
            route::Endpoint::of_upstream_path(fallback.path),
            route::Endpoint::ChatCompletions,
            "Claude's OpenRouter arm is Chat Completions, mixed-wire with the row",
        );
        assert_eq!(
            route::Endpoint::of_upstream_path(fallback.path).wire(),
            Dialect::OpenAi,
            "billing for the serving candidate must use the OpenAI extractor",
        );
        // The old derivation (row.wire, or provider.wire) is wrong for this fallback:
        assert_eq!(row.wire, Dialect::Anthropic);
        assert_eq!(openrouter.wire, Dialect::OpenAi);

        // The production derivation: serving the request from each arm sets the dialect from that
        // arm's path. Both attempts start from the row's wire, as `request_filter` leaves them.
        let serving = |candidate: usize| {
            let mut rc = test_ctx(true);
            rc.auto = Some(Box::new(ModelRouting {
                route: row,
                sub: None,
                candidate: u8::try_from(candidate).unwrap(),
                usable: 0b11,
                walk: control::Walk::identity(row.candidates.len()),
                arms: row.candidates,
                session_field: None,
                attempt_start: Instant::now(),
                cache: None,
                translate: None,
                health: None,
                addrs: 0,
                attempted: false,
                open_retry_after: None,
                priority: false,
            }));
            rc.dialect = row.wire;
            apply_serving_candidate(&mut rc);
            rc.dialect
        };
        assert_eq!(serving(0), Dialect::Anthropic, "the Messages primary");
        assert_eq!(
            serving(row.candidates.len() - 1),
            Dialect::OpenAi,
            "the OpenRouter Chat arm of a Claude row"
        );
    }

    /// Only a 401 walks and cools a key; 401, 402 and 403 are all catalog refusals. A 403's body
    /// cools the key only when it names the credential.
    #[test]
    fn only_a_401_or_a_403_naming_the_key_is_a_key_failure() {
        assert!(is_pool_key_failure(401));
        assert!(!is_pool_key_failure(403) && !is_pool_key_failure(402));
        assert!((401..=403).all(is_candidate_refusal));
        assert!(!is_candidate_refusal(429) && !is_candidate_refusal(400));
        assert!(body_names_the_key(
            br#"{"error":{"message":"Incorrect API key provided","code":"invalid_api_key"}}"#
        ));
        assert!(body_names_the_key(
            br#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#
        ));
        for body in [
            &br#"{"type":"error","error":{"type":"permission_error","message":"no access"}}"#[..],
            br#"{"error":{"message":"flagged by moderation","code":403}}"#,
            br#"{"error":{"code":"model_not_found"}}"#,
        ] {
            assert!(!body_names_the_key(body));
        }
    }

    /// A resumed key walk's candidate goes first; everyone else keeps their order behind it.
    #[test]
    fn walk_front_moves_one_candidate_and_keeps_the_rest_in_order() {
        let mut walk = control::Walk::identity(4);
        walk_front(&mut walk, 2);
        assert_eq!(&walk.indices[..4], &[2, 0, 1, 3]);
        walk_front(&mut walk, 2);
        assert_eq!(&walk.indices[..4], &[2, 0, 1, 3]);
        walk_front(&mut walk, 7);
        assert_eq!(
            &walk.indices[..4],
            &[2, 0, 1, 3],
            "not in the walk: unchanged"
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

    /// The premise behind `RequestCtx::begin_attempt`: a scanner fed a duplicated body
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
        assert_eq!(extract_virtual_key(&req).map(|p| p.key), Some("sk-ant-key"));
    }

    #[test]
    fn extract_virtual_key_recognizes_openai_bearer() {
        let req = req_with_headers(
            "/v1/chat/completions",
            &[("authorization", "Bearer sk-openai-key")],
        );
        assert_eq!(
            extract_virtual_key(&req).map(|p| p.key),
            Some("sk-openai-key")
        );
    }

    #[test]
    fn extract_virtual_key_recognizes_azure_api_key_header() {
        // Task #31: Azure OpenAI authenticates via the bare `api-key` header (no `Bearer` prefix, no
        // OAuth). Before this fix, a client presenting only this header got a 401.
        let req = req_with_headers("/v1/responses", &[("api-key", "azure-secret")]);
        assert_eq!(
            extract_virtual_key(&req).map(|p| p.key),
            Some("azure-secret")
        );
    }

    #[test]
    fn extract_virtual_key_recognizes_google_goog_api_key_header() {
        // Task #31: Google Gemini authenticates via `x-goog-api-key`.
        let req = req_with_headers(
            "/v1beta/models/gemini-2.5-pro:generateContent",
            &[("x-goog-api-key", "goog-secret")],
        );
        assert_eq!(
            extract_virtual_key(&req).map(|p| p.key),
            Some("goog-secret")
        );
    }

    #[test]
    fn extract_virtual_key_recognizes_google_key_query_param() {
        // Task #31: Google Gemini also accepts the key as a `?key=` query param — no header at all.
        let req = req_with_headers(
            "/v1beta/models/gemini-2.5-pro:generateContent?key=goog-query-secret",
            &[],
        );
        assert_eq!(
            extract_virtual_key(&req).map(|p| p.key),
            Some("goog-query-secret")
        );
    }

    #[test]
    fn extract_virtual_key_query_param_only_used_as_last_resort() {
        // A header takes precedence over a `key=` query param if both are somehow present.
        let req = req_with_headers(
            "/v1beta/models/gemini-2.5-pro:generateContent?key=query-secret",
            &[("x-goog-api-key", "header-secret")],
        );
        assert_eq!(
            extract_virtual_key(&req).map(|p| p.key),
            Some("header-secret")
        );
    }

    /// claim: SEC-11
    /// defect: D29
    #[test]
    fn a_managed_key_in_any_location_makes_the_request_managed() {
        const VK: &str = "bai_v1.1.payload.sig";
        for headers in [
            &[
                ("x-api-key", "junk"),
                ("authorization", "Bearer bai_v1.1.payload.sig"),
            ][..],
            &[
                ("x-api-key", ""),
                ("authorization", "Bearer bai_v1.1.payload.sig"),
            ][..],
            &[("api-key", "sk-real"), ("x-goog-api-key", VK)][..],
            &[
                ("x-api-key", "sk-ant-real"),
                ("authorization", "bearer bai_v1.1.payload.sig"),
            ][..],
        ] {
            let req = req_with_headers("/openai/v1/chat/completions", headers);
            assert_eq!(
                extract_virtual_key(&req).map(|p| p.key),
                Some(VK),
                "{headers:?}"
            );
        }
        let req = req_with_headers(
            "/openai/v1/chat/completions?key=bai_v1.1.payload.sig",
            &[("authorization", "Bearer sk-byo")],
        );
        assert_eq!(extract_virtual_key(&req).map(|p| p.key), Some(VK));
        // No managed key anywhere: the first non-empty location, as before.
        let req = req_with_headers(
            "/v1/chat/completions",
            &[("x-api-key", ""), ("authorization", "Bearer sk-byo")],
        );
        assert_eq!(extract_virtual_key(&req).map(|p| p.key), Some("sk-byo"));
    }

    #[test]
    fn extract_virtual_key_returns_none_when_absent() {
        let req = req_with_headers("/v1/chat/completions", &[]);
        assert_eq!(extract_virtual_key(&req).map(|p| p.key), None);
    }

    #[test]
    fn managed_client_headers_keep_only_the_allowlist_and_safe_betas() {
        let mut req = req_with_headers(
            "/v1/messages",
            &[
                ("content-type", "application/json"),
                ("user-agent", "claude-cli/2.0"),
                ("anthropic-version", "2023-06-01"),
                ("openai-organization", "org-evil"),
                ("cookie", "session=1"),
                ("x-stainless-lang", "js"),
                (
                    "anthropic-beta",
                    "prompt-caching-2024-07-31, context-1m-2025-08-07,interleaved-thinking-2025-05-14",
                ),
            ],
        );
        retain_managed_client_headers(&mut req, true, true).unwrap();
        let mut names: Vec<&str> = req.headers.keys().map(|k| k.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "anthropic-beta",
                "anthropic-version",
                "content-type",
                "user-agent"
            ]
        );
        assert_eq!(
            req.headers.get("anthropic-beta").unwrap(),
            "prompt-caching-2024-07-31,interleaved-thinking-2025-05-14"
        );

        // A value made only of unlisted tokens leaves no header at all; two header lines are
        // filtered as one list rather than letting the second through unread.
        let mut req = req_with_headers(
            "/v1/messages",
            &[("anthropic-beta", "mcp-client-2025-04-04")],
        );
        retain_managed_client_headers(&mut req, true, true).unwrap();
        assert!(req.headers.get("anthropic-beta").is_none());
        let mut req = req_with_headers("/v1/messages", &[]);
        req.append_header("anthropic-beta", "prompt-caching-2024-07-31")
            .unwrap();
        req.append_header("anthropic-beta", "code-execution-2025-05-22")
            .unwrap();
        retain_managed_client_headers(&mut req, true, true).unwrap();
        assert_eq!(
            req.headers
                .get_all("anthropic-beta")
                .iter()
                .collect::<Vec<_>>(),
            ["prompt-caching-2024-07-31"]
        );
    }

    #[test]
    fn strip_key_param_drops_only_the_credential() {
        assert_eq!(
            strip_key_param("/v1/x?key=bai_v1.a.b.c&api-version=2024").as_deref(),
            Some("/v1/x?api-version=2024")
        );
        assert_eq!(
            strip_key_param("/v1/x?a=1&key=s&b=2&key=t").as_deref(),
            Some("/v1/x?a=1&b=2")
        );
        assert_eq!(strip_key_param("/v1/x?key=s").as_deref(), Some("/v1/x"));
        assert_eq!(strip_key_param("/v1/x?keys=1&monkey=2"), None);
        assert_eq!(strip_key_param("/v1/x"), None);
    }

    #[test]
    fn key_params_finds_every_key_among_multiple_params() {
        let all = |q| key_params(q).collect::<Vec<_>>();
        assert_eq!(all("a=1&key=abc123&b=2"), ["abc123"]);
        assert_eq!(all("key=solo"), ["solo"]);
        assert_eq!(
            all("key=junk&k%65y=two&%6B%65%79=three"),
            ["junk", "two", "three"]
        );
        assert!(all("a=1&b=2&keys=3&monkey=4").is_empty());
        assert!(all("").is_empty());
    }

    /// A full-body attempt that ended with the client gone is the end of the request, whatever
    /// retry the attempt recorded: re-running it would send the whole body to another candidate (or
    /// key) for a client that is no longer there, a generation billed upstream that nobody reads.
    /// An upstream failure with a recorded retry still retries, and nothing written ends the walk.
    /// claim: REL-9, REL-21
    /// defect: D206
    #[test]
    fn a_full_body_attempt_that_lost_its_client_is_never_retried() {
        use pingora_core::{Error, ErrorType};
        let down = || Err(Error::new(ErrorType::ConnectionClosed).into_down());
        let up = || Err(Error::new(ErrorType::ConnectRefused).into_up());
        let retry = Some(RelayRetry::Candidate(0));
        assert_eq!(attempt_end(&down(), retry), AttemptEnd::Fail);
        assert_eq!(
            attempt_end(
                &down(),
                Some(RelayRetry::Key {
                    candidate: 0,
                    key: 1
                })
            ),
            AttemptEnd::Fail
        );
        assert_eq!(attempt_end(&down(), None), AttemptEnd::Fail);
        assert_eq!(
            attempt_end(&up(), retry),
            AttemptEnd::Retry(RelayRetry::Candidate(0))
        );
        assert_eq!(attempt_end(&up(), None), AttemptEnd::Fail);
        assert_eq!(
            attempt_end(&Ok(Piped::Abandoned), retry),
            AttemptEnd::Retry(RelayRetry::Candidate(0))
        );
        assert_eq!(attempt_end(&Ok(Piped::Written), retry), AttemptEnd::Done);
        assert_eq!(attempt_end(&Ok(Piped::Empty), None), AttemptEnd::Fail);
    }

    /// A full-body attempt's task that panicked is reported (the parent logs it with the request
    /// id, then answers 502), not dropped unseen; one that finished or is still running is not.
    /// claim: REL-17, REL-21
    /// defect: D207
    #[tokio::test]
    async fn a_panicked_full_body_attempt_is_reported() {
        let panicked = tokio::spawn(async { panic!("attempt blew up") });
        assert_eq!(reap_attempt(panicked, "rid-1").await, Reaped::Panicked);
        let finished = tokio::spawn(async {});
        assert_eq!(reap_attempt(finished, "rid-2").await, Reaped::Finished);
        let stuck = tokio::spawn(std::future::pending::<()>());
        tokio::time::pause();
        assert_eq!(reap_attempt(stuck, "rid-3").await, Reaped::StillRunning);
    }

    /// An error-in-200 is read from the root's members in any order; an answer beside `"error":
    /// null`, or with partial `choices`, is not one.
    /// claim: BIL-12, BIL-20
    #[test]
    fn json_is_error_only_reads_members_in_any_order() {
        for body in [
            r#"{"error":{"code":502,"message":"x"}}"#,
            r#"{"id":"gen-1","object":"chat.completion","error":{"code":502},"user_id":"u"}"#,
            r#"{"model":"m","error":"upstream failed"}"#,
        ] {
            assert!(json_is_error_only(body.as_bytes()), "{body}");
        }
        for body in [
            r#"{"id":"resp_1","object":"response","error":null,"output":[]}"#,
            r#"{"error":{"code":502},"choices":[{"finish_reason":"error"}]}"#,
            r#"{"type":"message","content":[],"error":{"x":1}}"#,
            r#"{"id":"x","error":null}"#,
            r#"{"error":{"code":502}"#,
            "[]",
        ] {
            assert!(!json_is_error_only(body.as_bytes()), "{body}");
        }
    }

    /// `background: true` is read as a provider's parser reads it: at the root, any copy, its name
    /// escaped or not; never a nested member or a string that says so.
    /// claim: BIL-2
    #[test]
    fn requests_background_reads_the_root_member() {
        for body in [
            r#"{"model":"m","background":true}"#,
            r#"{ "background" : true , "input":"x"}"#,
            r#"{"background":true}"#,
            r#"{"background":false,"background":true}"#,
        ] {
            assert!(requests_background(body.as_bytes()), "{body}");
        }
        for body in [
            r#"{"model":"m","background":false}"#,
            r#"{"model":"m","background":null,"stream":true}"#,
            r#"{"model":"m","background":"true"}"#,
            r#"{"model":"m","metadata":{"background":true}}"#,
            r#"{"model":"m","input":"background\":true"}"#,
            "not json true",
        ] {
            assert!(!requests_background(body.as_bytes()), "{body}");
        }
    }

    /// The pool key is scrubbed wherever it falls in a streamed body — inside one chunk, split
    /// across two, repeated — and the body keeps its length (so its `Content-Length`).
    /// claim: SEC-7
    #[test]
    fn redact_scrubs_a_key_split_across_chunks_and_keeps_the_length() {
        let raw = b"sk-pool-secret-0123456789";
        let key = &memchr::memmem::Finder::new(raw);
        let body = format!(
            r#"{{"error":{{"message":"bad key {k} (again: {k})","note":"{k}"}}}}"#,
            k = std::str::from_utf8(raw).unwrap()
        );
        for split in [1, 7, 20, 30, 47, body.len() - 1] {
            for step in [1, 3, 64] {
                let mut r = Redact::default();
                let mut out = Vec::new();
                let (head, tail) = body.as_bytes().split_at(split);
                let mut chunks: Vec<&[u8]> = head.chunks(step).collect();
                chunks.push(tail);
                for c in chunks {
                    let got = r.feed(
                        std::slice::from_ref(key),
                        Some(Bytes::copy_from_slice(c)),
                        false,
                    );
                    out.extend_from_slice(got.as_deref().unwrap_or(&[]));
                }
                out.extend_from_slice(&r.feed(std::slice::from_ref(key), None, true).unwrap());
                let text = String::from_utf8(out).unwrap();
                assert_eq!(text.len(), body.len(), "{split}/{step}");
                assert!(!text.contains("sk-pool-secret"), "{split}/{step}: {text}");
                assert_eq!(text.matches("[redacted]").count(), 3, "{text}");
            }
        }
        // A JSON error is held whole: nothing until its end, then the key masked and an account
        // remedy neutralized; one past the cap streams with the key masked.
        let remedy = format!(
            r#"{{"error":{{"message":"bad key {k}; add your own key","code":429}}}}"#,
            k = std::str::from_utf8(raw).unwrap()
        );
        let mut r = Redact {
            whole: true,
            status: 429,
            ..Redact::default()
        };
        for c in remedy.as_bytes().chunks(5) {
            let got = r.feed(
                std::slice::from_ref(key),
                Some(Bytes::copy_from_slice(c)),
                false,
            );
            assert!(got.unwrap().is_empty());
        }
        let out = r.feed(std::slice::from_ref(key), None, true).unwrap();
        assert_eq!(
            &out[..],
            format!(
                r#"{{"error":{{"code":429,"message":"{}"}}}}"#,
                remedy::RATE_LIMITED
            )
            .as_bytes()
        );
        let big = format!(
            r#"{{"error":{{"message":"{k} add your own key","pad":"{}"}}}}"#,
            "x".repeat(remedy::MAX_BODY),
            k = std::str::from_utf8(raw).unwrap()
        );
        let mut r = Redact {
            whole: true,
            status: 429,
            ..Redact::default()
        };
        let mut out = r
            .feed(
                std::slice::from_ref(key),
                Some(Bytes::copy_from_slice(big.as_bytes())),
                false,
            )
            .unwrap()
            .to_vec();
        out.extend_from_slice(&r.feed(std::slice::from_ref(key), None, true).unwrap());
        assert_eq!(out.len(), big.len());
        assert!(!out.windows(raw.len()).any(|w| w == raw));

        // A key shorter than the marker is masked with a truncated marker.
        let mut buf = *b"x=abc;";
        assert!(mask_all(&mut buf, &memchr::memmem::Finder::new(b"abc")));
        assert_eq!(&buf, b"x=[re;");
    }

    /// A base64 key (`/`, `+`) echoed in a JSON encoder's spelling (`\/`, `+`, `+`) is
    /// scrubbed on every path: streamed in small chunks, held whole with no remedy, and held whole
    /// and re-serialized by the remedy rewrite, which decodes any escape (`A`) back to the key.
    /// claim: SEC-7
    #[test]
    fn redact_scrubs_every_json_spelling_of_a_base64_key() {
        let key = "ABSKa2V5/cGFy+dA==";
        let keys = route::key_finders(key);
        assert_eq!(keys.len(), 6, "2 slash spellings x 3 plus spellings");
        for echo in [
            key.to_owned(),
            r"ABSKa2V5\/cGFy+dA==".to_owned(),
            r"ABSKa2V5\/cGFy+dA==".to_owned(),
            r"ABSKa2V5/cGFy+dA==".to_owned(),
        ] {
            for (whole, status, remedy) in [
                (false, 401, ""),
                (true, 401, ""),
                (true, 401, " see https://openrouter.ai/settings/keys"),
            ] {
                let body = format!(
                    r#"{{"error":{{"message":"bad key {echo}{remedy}","param":"{echo}","raw":"ABSKa2V5\/cGFy+dA=="}}}}"#
                );
                let mut r = Redact {
                    whole,
                    status,
                    ..Redact::default()
                };
                let mut out = Vec::new();
                for c in body.as_bytes().chunks(3) {
                    let got = r.feed(&keys, Some(Bytes::copy_from_slice(c)), false);
                    out.extend_from_slice(got.as_deref().unwrap_or(&[]));
                }
                out.extend_from_slice(&r.feed(&keys, None, true).unwrap());
                let text = String::from_utf8(out).unwrap();
                let decoded = serde_json::from_str::<serde_json::Value>(&text)
                    .unwrap()
                    .to_string();
                let rewritten = !remedy.is_empty();
                assert!(
                    !text.contains(key) && !text.contains(&echo),
                    "{echo} whole={whole} remedy={rewritten}: {text}"
                );
                // A held JSON body is also clean as a JSON client reads it: the `raw` field's
                // `A` spelling decodes to the key. A streamed one is scrubbed of the
                // spellings an encoder writes.
                if whole {
                    assert!(
                        !decoded.contains(key),
                        "{echo} whole remedy={rewritten}: {text}"
                    );
                }
            }
        }
    }

    /// Every spelling of a credential location is read, and a virtual key in any of them makes the
    /// request managed: a repeated header whose first line is junk, a second `?key=`, an encoded
    /// `key` name or value, extra whitespace after `Bearer`, a scheme-less virtual key.
    /// claim: SEC-3, SEC-11
    #[test]
    fn a_managed_key_in_any_spelling_makes_the_request_managed() {
        const VK: &str = "bai_v1.1.payload.sig";
        let build = |path: &str, headers: &[(&'static str, &str)]| {
            let mut req =
                pingora::http::RequestHeader::build(http::Method::POST, path.as_bytes(), None)
                    .unwrap();
            for (k, v) in headers {
                req.append_header(*k, *v).unwrap();
            }
            req
        };
        let cases: [(&str, &[(&'static str, &str)]); 7] = [
            ("/x?key=junk&key=bai_v1.1.payload.sig", &[]),
            (
                "/x?k%65y=bai_v1.1.payload.sig",
                &[("authorization", "Bearer sk-byo")],
            ),
            ("/x", &[("x-api-key", "junk"), ("x-api-key", VK)]),
            ("/x", &[("authorization", "Bearer  bai_v1.1.payload.sig")]),
            ("/x", &[("authorization", "Bearer	bai_v1.1.payload.sig ")]),
            (
                "/x",
                &[("authorization", "Bearer sk-byo"), ("authorization", VK)],
            ),
            ("/x", &[("api-key", "sk-real"), ("api-key", VK)]),
        ];
        for (path, headers) in cases {
            let req = build(path, headers);
            let p = extract_virtual_key(&req).unwrap();
            assert!(p.managed, "{path} {headers:?}");
            assert_eq!(p.key, VK, "{path} {headers:?}");
        }
        // Managed only once decoded: managed (so stripped, never forwarded), and the encoded
        // value fails verification.
        let req = build("/x?key=%62ai_v1.1.payload.sig", &[]);
        assert!(extract_virtual_key(&req).unwrap().managed);
        // A BYO key in a lenient spelling is still the BYO key.
        let req = build("/x", &[("authorization", "bearer   sk-byo")]);
        assert_eq!(
            extract_virtual_key(&req),
            Some(Presented {
                key: "sk-byo",
                managed: false
            })
        );
        assert_eq!(
            strip_key_param("/x?k%65y=a&b=1&%6b%65%79=c&key=d").as_deref(),
            Some("/x?b=1")
        );
    }

    #[test]
    fn the_gateways_own_refusal_is_not_giving_up_on_a_delivered_request() {
        use pingora_core::{Error, ErrorType as T};
        // A body the gateway withheld and refused: the provider never had it, nothing to bill.
        assert!(!gave_up_waiting(
            &Refusal::CatalogMiss {
                name: None,
                header: false
            }
            .into_error()
            .into_down()
        ));
        assert!(!gave_up_waiting(
            &Error::new(T::HTTPStatus(400)).into_down()
        ));
        // The client going away, and the gateway's read timeout, still are.
        assert!(gave_up_waiting(
            &Error::new(T::ConnectionClosed).into_down()
        ));
        assert!(gave_up_waiting(&Error::new(T::ReadTimedout).into_up()));
    }

    #[test]
    fn price_model_resolves_echoed_snapshots_and_vendor_slugs() {
        assert_eq!(price_model("gpt-5-2025-08-07", "gpt-5"), Some("gpt-5"));
        assert_eq!(price_model("gpt-5-2025-08-07", "x"), Some("gpt-5"));
        assert_eq!(
            price_model("claude-opus-4-8-20260101", "x"),
            Some("claude-opus-4-8")
        );
        assert_eq!(
            price_model("anthropic/claude-opus-4.8", "x"),
            Some("claude-opus-4-8")
        );
        // The echo names nothing priced, the request does.
        assert_eq!(price_model("unknown", "gpt-5"), Some("gpt-5"));
        assert_eq!(price_model("my-finetune-2025-08-07", "nope"), None);
        // The managed `/{provider}` refusal (D267) asks exactly this of the requested model: any
        // candidate's provider spelling counts, as does a dated snapshot; nothing else does.
        let row = |m| price_row(m).map(|r| r.model);
        assert_eq!(
            row("accounts/fireworks/models/gpt-oss-120b"),
            price_model("", "accounts/fireworks/models/gpt-oss-120b")
        );
        assert!(row("accounts/fireworks/models/gpt-oss-120b").is_some());
        assert_eq!(row("claude-sonnet-4-5-20250929"), Some("claude-sonnet-4-5"));
        assert_eq!(row("gpt-4o-mini-2024-07-18"), Some("gpt-4o-mini"));
        for unpriced in ["my-finetune", "ft:gpt-4o:acme", "claude-2.1", "", "unknown"] {
            assert_eq!(row(unpriced), None, "{unpriced}");
            assert_eq!(price_model("", unpriced), None, "{unpriced}");
        }
        // Not a date: left alone.
        assert_eq!(strip_snapshot_date("gpt-4o-mini"), None);
        assert_eq!(strip_snapshot_date("model-12-34"), None);
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
    fn a_provider_route_takes_its_wire_from_the_forwarded_path() {
        for (path, wire) in [
            ("/api/v1/messages", Dialect::Anthropic),
            ("/v1/messages?beta=true", Dialect::Anthropic),
            ("/v1/messages/count_tokens", Dialect::Anthropic),
            ("/v1/messages/", Dialect::Anthropic),
            ("/v1/chat/completions", Dialect::OpenAi),
            (
                "/openai/deployments/x/chat/completions?api-version=1",
                Dialect::OpenAi,
            ),
            ("/v1/responses/compact", Dialect::OpenAi),
            ("/v1/embeddings", Dialect::OpenAi),
        ] {
            assert_eq!(wire_of_forward_path(path), wire, "{path}");
        }
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
            bare_default_provider_name("/v1beta/models/gemini-2.5-pro:generateContent", None),
            None,
            "/v1beta must NOT be routed to OpenAI (or any provider) via the bare-default path"
        );
        assert_eq!(bare_default_provider_name("/v1beta", None), None);

        // The real bare-default shape still resolves correctly, dialect-picked.
        assert_eq!(
            bare_default_provider_name("/v1/messages", None),
            Some("anthropic")
        );
        assert_eq!(
            bare_default_provider_name("/v1/chat/completions", None),
            Some("openai")
        );
        assert_eq!(bare_default_provider_name("/v1", None), Some("openai"));

        // Other near-miss prefixes must also be rejected, not just /v1beta.
        assert_eq!(bare_default_provider_name("/v10/messages", None), None);
        assert_eq!(bare_default_provider_name("/v2/messages", None), None);
    }

    /// claim: SEC-11
    /// defect: D30
    #[test]
    fn a_byo_x_api_key_on_bare_v1_routes_to_anthropic_on_every_path() {
        for path in [
            "/v1/files",
            "/v1/models/claude-opus-4-8",
            "/v1/chat/completions",
            "/v1",
        ] {
            assert_eq!(
                bare_default_provider_name(path, Some(Dialect::Anthropic)),
                Some("anthropic"),
                "{path}"
            );
        }
        assert_eq!(
            bare_default_provider_name("/v1beta/x", Some(Dialect::Anthropic)),
            None
        );
        let req = req_with_headers("/v1/files", &[("x-api-key", "sk-ant-byo")]);
        assert_eq!(byo_credential_dialect(&req), Ok(Some(Dialect::Anthropic)));
        for headers in [
            &[("x-api-key", "bai_v1.1.p.s")][..],
            &[("x-api-key", "")][..],
            &[("authorization", "Bearer opaque-token")][..],
        ] {
            let req = req_with_headers("/v1/files", headers);
            assert_eq!(byo_credential_dialect(&req), Ok(None), "{headers:?}");
        }
    }

    /// The BYO credential that will be forwarded picks the provider (D82): by shape where it has
    /// one, `x-api-key` meaning Anthropic otherwise, every value read; keys for two providers on one
    /// request are ambiguous.
    #[test]
    fn byo_credentials_pick_the_provider_they_belong_to() {
        // Appends, so a repeated header keeps every line.
        let dialect = |headers: &[(&'static str, &'static str)]| {
            let mut req = pingora::http::RequestHeader::build(
                http::Method::POST,
                b"/v1/chat/completions",
                None,
            )
            .unwrap();
            for (k, v) in headers {
                req.append_header(*k, *v).unwrap();
            }
            byo_credential_dialect(&req)
        };
        assert_eq!(
            dialect(&[("authorization", "Bearer sk-proj-abc")]),
            Ok(Some(Dialect::OpenAi))
        );
        assert_eq!(
            dialect(&[("authorization", "Bearer sk-ant-oat01-abc")]),
            Ok(Some(Dialect::Anthropic))
        );
        assert_eq!(
            dialect(&[("x-api-key", ""), ("x-api-key", "sk-ant-api03-x")]),
            Ok(Some(Dialect::Anthropic)),
            "every x-api-key line counts"
        );
        assert_eq!(
            dialect(&[
                ("x-api-key", "sk-ant-a"),
                ("authorization", "Bearer sk-ant-a")
            ]),
            Ok(Some(Dialect::Anthropic)),
            "the same provider twice is not ambiguous"
        );
        assert_eq!(
            dialect(&[
                ("x-api-key", "stray"),
                ("authorization", "Bearer sk-proj-abc")
            ]),
            Err(())
        );
        assert_eq!(
            dialect(&[("x-api-key", "sk-ant-a"), ("x-api-key", "sk-proj-b")]),
            Err(())
        );
        assert_eq!(
            dialect(&[
                ("x-api-key", "bai_v1.1.p.s"),
                ("authorization", "Bearer sk-proj-abc")
            ]),
            Ok(Some(Dialect::OpenAi)),
            "a managed value casts no vote"
        );
        assert_eq!(
            bare_default_provider_name("/v1/messages", Some(Dialect::OpenAi)),
            Some("openai"),
            "an OpenAI key never goes to Anthropic, whatever the path"
        );
    }

    /// The managed allowlist (a security boundary) and every routing decision read one endpoint
    /// table, so they cannot drift: for every endpoint, under every provider's mount prefix, with
    /// a trailing slash or a query, the allowlist admits it, its wire, sub-resource and
    /// streamability follow from its row, and the catalog spelling (`/v1`, `/auto`) names the
    /// same endpoint. A path in no row is refused by the allowlist and names nothing.
    /// claim: SEC-1, CAT-14
    /// defect: D209
    #[test]
    fn the_allowlist_and_the_routing_tables_agree_for_every_endpoint() {
        for e in &route::ENDPOINT_PATHS {
            let generation = e.sub.is_none();
            for prefix in [
                "/v1",
                "/api/v1",
                "/openai/v1",
                "/inference/v1",
                "/anthropic/v1",
                "/backend-api/codex",
                "/openai/deployments/x",
            ] {
                let base = format!("{prefix}{}", e.suffix);
                for path in [
                    base.clone(),
                    format!("{base}/"),
                    format!("{base}?beta=true"),
                ] {
                    let bare = path.split_once('?').map_or(path.as_str(), |(p, _)| p);
                    assert!(is_managed_provider_endpoint(&path), "{path}");
                    assert_eq!(wire_of_forward_path(&path), e.endpoint.wire(), "{path}");
                    assert_eq!(route::SubResource::of_forward_path(&path), e.sub, "{path}");
                    assert_eq!(
                        is_streamable_path(bare),
                        generation && e.endpoint == route::Endpoint::ChatCompletions,
                        "{path}"
                    );
                    assert_eq!(
                        route::forward_is_responses(bare),
                        generation && e.endpoint == route::Endpoint::Responses,
                        "{path}"
                    );
                }
                if generation {
                    assert_eq!(
                        route::Endpoint::of_upstream_path(&base),
                        e.endpoint,
                        "{base}"
                    );
                }
            }
            for path in [
                format!("/v1{}", e.suffix),
                format!("/auto{}", e.suffix),
                format!("/auto/v1{}/", e.suffix),
            ] {
                assert_eq!(
                    route::implied_endpoint(&path),
                    generation.then_some(e.endpoint),
                    "{path}"
                );
                assert_eq!(route::SubResource::of_path(&path), e.sub, "{path}");
            }
        }
        for path in [
            "/v1/files",
            "/v1/models",
            "/v1/batches",
            "/v1/responses/resp_123",
            "/v1/responses/resp_123/cancel",
            "/v1/messages/batches",
            "/v1/chat/completions/chatcmpl-1",
            "/v1/fine_tuning/jobs",
        ] {
            assert!(!is_managed_provider_endpoint(path), "{path}");
            assert_eq!(route::SubResource::of_forward_path(path), None, "{path}");
            assert_eq!(route::implied_endpoint(path), None, "{path}");
            assert!(!route::forward_is_responses(path), "{path}");
        }
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
    fn a_client_sent_stream_options_always_asks_for_usage() {
        for (body, want_span_moves) in [
            (
                r#"{"model":"gpt-4o","stream":true,"stream_options":{"include_usage":false}}"#,
                false,
            ),
            (
                r#"{"stream_options":{},"stream":true,"model":"gpt-4o"}"#,
                true,
            ),
            (
                r#"{"stream":true,"stream_options": null ,"model":"gpt-4o"}"#,
                true,
            ),
            (
                r#"{"stream":true,"stream_options":{"x":1},"model":"gpt-4o"}"#,
                true,
            ),
        ] {
            let scan = peek::scan_buffered(body.as_bytes());
            assert_eq!(scan.inject_at, None, "{body}");
            let at = scan
                .stream_options_at
                .expect("stream_options value located");
            let (out, span) = force_include_usage(body.as_bytes().to_vec(), at, scan.model_span);
            let v: serde_json::Value = serde_json::from_slice(&out).expect("still JSON");
            assert_eq!(v["stream_options"]["include_usage"], true, "{body}");
            let (s, e) = span.expect("span");
            assert_eq!(&out[s..e], b"gpt-4o", "{body}");
            assert_eq!(span != scan.model_span, want_span_moves, "{body}");
        }
        // Already asking: untouched, byte for byte.
        let body = br#"{"stream":true,"stream_options":{ "include_usage" : true }}"#;
        let at = peek::scan_buffered(body)
            .stream_options_at
            .expect("located");
        assert_eq!(force_include_usage(body.to_vec(), at, None).0, body);
        // A non-stream body's stream_options is not ours to touch.
        let off = br#"{"stream":false,"stream_options":{"include_usage":false}}"#;
        assert_eq!(peek::scan_buffered(off).stream_options_at, None);
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
    fn clip_catalog_name_caps_a_long_caller_string() {
        assert_eq!(clip_catalog_name("gpt-4o-mini"), "gpt-4o-mini");
        let long = "x".repeat(200);
        assert_eq!(clip_catalog_name(&long).len(), 128);
        // Byte 128 falls inside a two-byte character: the cut backs off to its start.
        let wide = format!("{}{}", "x".repeat(127), "é".repeat(10));
        assert_eq!(clip_catalog_name(&wide), "x".repeat(127));
    }

    /// Pingora's own error lines print the request without its query, where a `?key=` credential
    /// rides.
    /// claim: SEC-4
    #[test]
    fn the_summary_line_leaves_out_the_query() {
        let mut req = pingora::http::RequestHeader::build(
            http::Method::POST,
            b"/v1beta/models/g:generateContent?key=secret",
            None,
        )
        .unwrap();
        req.insert_header("host", "gw.example").unwrap();
        assert_eq!(
            summary_line(&req),
            "POST /v1beta/models/g:generateContent, Host: gw.example"
        );
    }

    /// The compression opt-out is spliced only just inside a non-empty root object; an empty
    /// object, an array, or a body that already names `plugins` goes as sent.
    /// claim: CAT-3
    /// defect: D109
    #[test]
    fn compression_is_disabled_only_inside_a_non_empty_object() {
        let off = |b: &[u8]| disable_openrouter_compression(b.to_vec());
        for same in [
            &b"{}"[..],
            b" { } ",
            b"[1]",
            br#"[{"a":1}]"#,
            b"",
            b"  ",
            br#"{"plugins":[],"a":1}"#,
        ] {
            assert_eq!(off(same), same, "{}", String::from_utf8_lossy(same));
        }
        let spliced = |pre: &str, rest: &str| {
            let mut v = pre.as_bytes().to_vec();
            v.extend_from_slice(OPENROUTER_NO_COMPRESSION);
            v.extend_from_slice(rest.as_bytes());
            v
        };
        assert_eq!(off(br#"{"a":1}"#), spliced("{", r#""a":1}"#));
        assert_eq!(off(br#" {"a":1}"#), spliced(" {", r#""a":1}"#));
    }

    /// An empty `model` value (a zero-width span) is still rewritten.
    /// claim: R1
    #[test]
    fn an_empty_model_span_is_rewritten() {
        let body = br#"{"model":""}"#.to_vec();
        assert_eq!(
            apply_model_rewrite(body, (10, 10), b"gpt-4o"),
            br#"{"model":"gpt-4o"}"#
        );
    }

    /// A full-body attempt that was cancelled rather than panicked is just finished: only a
    /// panic has a payload to log.
    /// claim: REL-17
    #[tokio::test]
    async fn a_cancelled_full_body_attempt_is_finished_not_a_panic() {
        let cancelled = tokio::spawn(std::future::pending::<()>());
        cancelled.abort();
        assert_eq!(reap_attempt(cancelled, "rid").await, Reaped::Finished);
    }

    #[test]
    fn reject_bodies_are_valid_json_and_match_their_type_and_message() {
        // Every static body is a hand-written JSON literal standing in for what `serde_json::json!`
        // would build, so the thing to guard is that it still *says* what the `error_type` in the
        // log line and the metric label claim, byte for byte. A drifting literal would ship a
        // response whose `type` contradicts the reason the request ended.
        let refusals = crate::refusal::every();
        let fixed = refusals.iter().filter_map(Refusal::fixed);
        for f in fixed.chain(FAILURES) {
            let v: serde_json::Value = serde_json::from_str(f.body)
                .unwrap_or_else(|e| panic!("{} is not JSON: {e}", f.body));
            assert_eq!(v["error"]["type"], f.kind, "type mismatch in {}", f.body);
            assert_eq!(
                v["error"]["message"], f.message,
                "message mismatch in {}",
                f.body
            );
            let built = serde_json::json!({ "error": { "type": f.kind, "message": f.message } })
                .to_string();
            assert_eq!(f.body, built, "constant diverges from the built body");
        }
        // A failure's lookup finds its static body rather than falling through to `serde_json`.
        for f in FAILURES {
            assert_eq!(
                error_body(f.kind, f.message).as_ptr(),
                f.body.as_ptr(),
                "{}: {}",
                f.kind,
                f.message
            );
        }
        // Every pair the old flat table held is still answered with a static body: a fixed
        // refusal's, or a failure's.
        for (typ, msg) in [
            ("invalid_request_error", "unknown provider"),
            ("authentication_error", "missing API key"),
            ("authentication_error", "invalid API key"),
            ("rate_limit_error", "rate limit exceeded"),
            ("invalid_request_error", "request body too large"),
            ("access_denied", "tenant is over limit or suspended"),
            ("api_error", "no provider key available"),
            ("api_error", "provider temporarily unavailable"),
            (
                "invalid_request_error",
                "model routing requires a managed key",
            ),
            ("insufficient_quota", "quota exhausted"),
            ("api_error", "allowance unavailable"),
            ("rate_limit_error", "too many concurrent requests"),
            ("api_error", "too many large request bodies in flight"),
            (
                "api_error",
                "upstream timed out after receiving the request",
            ),
            ("api_error", "upstream failed after receiving the request"),
        ] {
            let as_refusal = refusals
                .iter()
                .find(|r| r.fixed().is_some_and(|f| f.kind == typ && f.message == msg));
            let static_body = match as_refusal {
                Some(r) => {
                    r.body().as_ptr() == r.fixed().map_or(std::ptr::null(), |f| f.body.as_ptr())
                }
                None => FAILURES.iter().any(|f| f.kind == typ && f.message == msg),
            };
            assert!(static_body, "({typ:?}, {msg:?}) lost its static body");
        }
    }

    #[test]
    fn every_reject_call_site_has_a_precomputed_body() {
        // A refusal is typed, so its body cannot miss its table. What can is a failure: each
        // message the production half passes to `fail_boxed` or `gateway_error`, or `fail_to_proxy`
        // assigns, must have its static body in `FAILURES`, under the type it is answered with,
        // and every entry there must have a producer (no entry left behind after its failure
        // went). And no body may be written past both tables: every writer is listed by the
        // function that calls it. Only the production half of the file: this test mentions the call
        // forms itself.
        let src = include_str!("proxy.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("split always yields at least one part");
        let typ = |status: u16| match status {
            429 => "rate_limit_error",
            400..=499 => "invalid_request_error",
            _ => "api_error",
        };
        // The string literals in the call that starts at `at`, up to its closing `)` line.
        let literals = |at: usize| -> Vec<&str> {
            let tail = &src[at..];
            let end = tail.find(")\n").map_or(tail.len(), |e| e + 1);
            tail[..end]
                .match_indices('"')
                .collect::<Vec<_>>()
                .chunks(2)
                .filter_map(|c| match c {
                    [(a, _), (b, _)] => Some(&tail[a + 1..*b]),
                    _ => None,
                })
                .collect()
        };
        // The numeric status argument of `gateway_error(status, "…")`.
        let status_arg = |at: usize| -> u16 {
            let tail = &src[at + "gateway_error(".len()..];
            tail[..tail.find(',').expect("a status argument")]
                .trim()
                .parse()
                .expect("a numeric status")
        };
        // Every call of `fail_boxed` and `gateway_error` but their definitions passes its message as
        // a literal: a site whose message could not be read would escape the pairing below.
        let is_definition = |at: usize| src[..at].ends_with("fn ");
        let mut produced = Vec::new();
        let mut sites = (0usize, 0usize);
        for (i, _) in src.match_indices("fail_boxed(") {
            if is_definition(i) {
                continue;
            }
            let msg = *literals(i)
                .first()
                .expect("a fail_boxed site with a message literal");
            produced.push(("api_error", msg));
            sites.0 += 1;
        }
        for (i, _) in src.match_indices("gateway_error(") {
            if is_definition(i) {
                continue;
            }
            let msg = *literals(i)
                .first()
                .expect("a gateway_error site with a message literal");
            produced.push((typ(status_arg(i)), msg));
            sites.1 += 1;
        }
        assert!(
            sites.0 > 0 && sites.1 > 0,
            "the scan found no sites: {sites:?}"
        );
        // Every path that writes a response body, by the function it is written in: a refusal
        // (`write_refusal`; `fail_to_proxy` for one raised past `request_filter`), a failure
        // (`fail_boxed`, `fail_to_proxy`), the writers' own definitions, and the two bodies that
        // are no error (the models list, a cache hit). A body written anywhere else would pass
        // both tables, so it fails here until it goes through one of these.
        let mut writers = Vec::new();
        let mut current_fn = "";
        for line in src.lines() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            let decl = ["pub(crate) ", "pub ", "const ", "async "]
                .iter()
                .fold(code, |c, p| c.strip_prefix(p).unwrap_or(c));
            if let Some(rest) = decl.strip_prefix("fn ") {
                current_fn = &rest[..rest.find(['(', '<']).unwrap_or(rest.len())];
            }
            for needle in [
                "write_json_body(",
                "error_body(",
                "json_body(",
                "refusal.body()",
                "write_response_body(",
            ] {
                for (at, _) in line.match_indices(needle) {
                    if !line[..at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
                        writers.push((needle, current_fn));
                    }
                }
            }
        }
        writers.sort_unstable();
        let mut want = vec![
            ("error_body(", "error_body"),
            ("error_body(", "fail_boxed"),
            ("error_body(", "fail_to_proxy"),
            ("json_body(", "error_body"),
            ("refusal.body()", "fail_to_proxy"),
            ("refusal.body()", "write_refusal"),
            ("write_json_body(", "fail_boxed"),
            ("write_json_body(", "fail_to_proxy"),
            ("write_json_body(", "write_json_body"),
            ("write_json_body(", "write_refusal"),
            ("write_response_body(", "reply_cache_hit"),
            ("write_response_body(", "reply_models_list"),
            ("write_response_body(", "write_json_body"),
        ];
        want.sort_unstable();
        assert_eq!(
            writers, want,
            "a response body is written outside the tabled paths"
        );
        // `fail_to_proxy` names a delivered request's failure and a refused stream itself, under
        // the type `failure_response` gives the error class reaching that arm: the message is read
        // from its source and the type from the mapping, so either drifting fails here.
        let ftp = src
            .find("async fn fail_to_proxy(")
            .expect("fail_to_proxy is in the production half");
        let ftp = &src[ftp..ftp + src[ftp..].find("\n    }\n").expect("fail_to_proxy ends")];
        for (class, msg) in [
            (
                pingora_core::ErrorType::ReadTimedout,
                "upstream timed out after receiving the request",
            ),
            (
                pingora_core::ErrorType::ConnectionClosed,
                "upstream failed after receiving the request",
            ),
            (
                pingora_core::ErrorType::H2Error,
                "the provider refused the stream; it was not processed",
            ),
        ] {
            assert!(
                ftp.contains(&format!("msg = \"{msg}\";")),
                "fail_to_proxy no longer answers {msg:?}"
            );
            let (_, typ, _) = failure_response(&pingora_core::Error::new_up(class))
                .expect("an upstream error is answered");
            produced.push((typ, msg));
        }
        // Every class `failure_response` maps, by its source and type.
        {
            use pingora_core::{Error, ErrorType as T};
            let mut classes = vec![
                Error::new_down(T::ReadTimedout),
                Error::new_down(T::InvalidHTTPHeader),
                Error::new_up(T::ConnectTimedout),
                Error::new_up(T::ConnectRefused),
                Error::new_up(T::H2Error),
                Error::new(T::InternalError),
            ];
            for status in [400, 401, 413, 429, 500, 502, 503, 504] {
                classes.push(Error::new(T::HTTPStatus(status)));
            }
            for e in &classes {
                let (_, typ, msg) = failure_response(e).expect("answered");
                produced.push((typ, msg));
            }
        }
        // And the other way: every message `fail_to_proxy` assigns, and every message literal in
        // `failure_response`, is one produced above, so an arm added to either without a case here
        // fails rather than going unchecked.
        let quoted = |s: &'static str| -> Vec<&'static str> {
            s.match_indices('"')
                .collect::<Vec<_>>()
                .chunks(2)
                .filter_map(|c| match c {
                    [(a, _), (b, _)] => Some(&s[a + 1..*b]),
                    _ => None,
                })
                .collect()
        };
        for (at, _) in ftp.match_indices("msg = \"") {
            let msg = quoted(&ftp[at..])[0];
            assert!(
                produced.iter().any(|&(_, m)| m == msg),
                "fail_to_proxy answers {msg:?}, which no case here produces"
            );
        }
        let fr = src
            .find("fn failure_response(")
            .expect("failure_response is in the production half");
        let fr = &src[fr..fr + src[fr..].find("\n}\n").expect("failure_response ends")];
        let types = ["rate_limit_error", "invalid_request_error", "api_error"];
        for msg in quoted(fr).into_iter().filter(|m| !types.contains(m)) {
            assert!(
                produced.iter().any(|&(_, m)| m == msg),
                "failure_response answers {msg:?}, which no class here produces"
            );
        }
        for &(typ, msg) in &produced {
            assert!(
                FAILURES.iter().any(|f| f.kind == typ && f.message == msg),
                "({typ:?}, {msg:?}) has no FAILURES entry: it would build its body on every hit",
            );
        }
        for f in FAILURES {
            assert!(
                produced.contains(&(f.kind, f.message)),
                "FAILURES entry ({:?}, {:?}) has no producer",
                f.kind,
                f.message
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
        // The suffix match cannot see past a query string. This locks in *why*
        // `stream_options_eligible` strips the query before asking it: Azure OpenAI requires
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
        apply_provider_attribution(
            &mut openrouter_req,
            Some(providers::ProviderId::OpenRouter),
            true,
        )
        .unwrap();
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
            apply_provider_attribution(
                &mut req,
                Some(providers::by_name(other).expect("a built-in provider").id),
                true,
            )
            .unwrap();
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
        apply_provider_attribution(&mut byo_req, Some(providers::ProviderId::OpenRouter), false)
            .unwrap();
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

/// Behaviors a mutation-testing pass found no test constraining.
#[cfg(test)]
mod mutation_gaps {
    use super::*;
    use crate::refusal::default_retry_after;
    use pingora_core::{Error, ErrorType as T};

    /// Each way a request ends maps to its own `outcome` on the billing row, so a zero-token row
    /// for an error or a cancel can be told from a real zero-token generation.
    /// claim: BIL-12
    #[test]
    fn every_ending_has_its_own_outcome() {
        let mut rc = tests::test_ctx(false);
        assert_eq!(outcome(&rc, None, false, false), "no_candidate");
        assert_eq!(
            outcome(&rc, None, true, false),
            "ok",
            "a cache hit made no call"
        );
        rc.upstream_phase = UpstreamPhase::Attempted;
        let up = Error::new_up(T::ConnectRefused);
        assert_eq!(outcome(&rc, Some(&up), false, false), "upstream_error");
        let down = Error::new_down(T::ReadError);
        assert_eq!(outcome(&rc, Some(&down), false, false), "client_cancelled");
        rc.upstream_phase = UpstreamPhase::Connected;
        rc.upstream_status = NonZeroU16::new(200);
        assert_eq!(outcome(&rc, None, false, false), "ok");
        assert_eq!(outcome(&rc, Some(&up), false, false), "cut_short");
        assert_eq!(outcome(&rc, Some(&down), false, false), "client_cancelled");
        // A drained stream (the client left first) is a cancel however the request then ended.
        assert_eq!(outcome(&rc, None, false, true), "client_cancelled");
        assert_eq!(outcome(&rc, Some(&up), false, true), "client_cancelled");
        rc.upstream_status = NonZeroU16::new(400);
        assert_eq!(outcome(&rc, None, false, false), "upstream_error");
        assert_eq!(outcome(&rc, Some(&down), false, false), "upstream_error");
    }

    /// Only a 401 is a pool key failing (D84): a 402 or 403 is about the request or the account,
    /// a candidate refusal that walks the catalog but never cools or walks keys.
    /// claim: REL-4
    #[test]
    fn pool_key_failures_are_401_only() {
        for (status, key, candidate) in [
            (400, false, false),
            (401, true, true),
            (402, false, true),
            (403, false, true),
            (404, false, false),
            (429, false, false),
            (500, false, false),
        ] {
            assert_eq!(is_pool_key_failure(status), key, "{status}");
            assert_eq!(is_candidate_refusal(status), candidate, "{status}");
        }
    }

    /// A complete first `data:` line the error check cannot read is an answer, not "undecided".
    /// claim: REL-8
    #[test]
    fn an_unreadable_complete_first_event_is_an_answer() {
        assert_eq!(body_reports_error(b"data: {\"err\n\n", true), Some(false));
        assert_eq!(
            body_reports_error(b"data: {\"err\ndata: {\"error\":{}}\n\n", true),
            Some(false),
            "the first event decides"
        );
    }

    /// Only the `Bearer` scheme carries a token; another six-letter scheme does not, and neither
    /// does `Bearer` run into its token.
    /// claim: SEC-11
    #[test]
    fn only_a_spaced_bearer_scheme_carries_a_token() {
        assert_eq!(bearer_token("Bearer bai_v1x"), Some("bai_v1x"));
        assert_eq!(bearer_token("bEaReR \t bai_v1x"), Some("bai_v1x"));
        assert_eq!(bearer_token("Digest bai_v1x"), None);
        assert_eq!(bearer_token("Bearerbai_v1x"), None);
    }

    /// A space is not a control byte: a model id with one is kept on the billing row.
    /// claim: BIL-13
    #[test]
    fn a_model_id_with_a_space_is_kept() {
        assert_eq!(sanitize_model("my model".to_owned()), "my model");
        assert_eq!(sanitize_model("bad\u{1f}model".to_owned()), "unknown");
    }

    /// A dated snapshot prices as its base model, in either spelling the vendors use.
    /// claim: BIL-13, BIL-24
    /// defect: D278
    #[test]
    fn a_dated_snapshot_prices_as_its_base_model() {
        for id in ["gpt-4o-mini-2024-07-18", "gpt-4o-mini-20240718"] {
            assert_eq!(price_row(id).map(|r| r.model), Some("gpt-4o-mini"), "{id}");
        }
    }

    /// Only an eight-digit or `YYYY-MM-DD` suffix is a snapshot date; an all-digit suffix of
    /// another length (`gpt-4-0613`) and an eight-letter one are not.
    /// claim: BIL-13
    #[test]
    fn only_real_snapshot_dates_are_stripped() {
        assert_eq!(
            strip_snapshot_date("claude-sonnet-4-5-20250929"),
            Some("claude-sonnet-4-5")
        );
        assert_eq!(strip_snapshot_date("gpt-5-2025-08-07"), Some("gpt-5"));
        assert_eq!(strip_snapshot_date("gpt-4-0613"), None);
        assert_eq!(strip_snapshot_date("model-abcdefgh"), None);
        assert_eq!(strip_snapshot_date("model-v2-05-12"), None, "no year");
        assert_eq!(
            strip_snapshot_date("model-2025-5-12"),
            None,
            "one-digit month"
        );
    }

    /// `Retry-After` defaults: 5s for an unavailable allowance, 1s for any other 429/503, none
    /// otherwise.
    /// Rejection lines are capped per second: the second's first `per_sec` lines are logged, the
    /// rest suppressed and counted, and the next logged line (in a later second) reports how many.
    /// defect: D263
    #[test]
    fn rejection_lines_are_capped_per_second_and_the_rest_counted() {
        let rate = LogRate::new(3);
        assert_eq!(rate.admit(1), Some(0));
        assert_eq!(rate.admit(1), Some(0));
        assert_eq!(rate.admit(1), Some(0));
        assert_eq!(rate.admit(1), None);
        assert_eq!(rate.admit(1), None);
        // A new second refills the allowance; its first line reports the two suppressed.
        assert_eq!(rate.admit(2), Some(2));
        assert_eq!(rate.admit(2), Some(0));
        // A racing caller with an older second neither rewinds the window nor refills it.
        assert_eq!(rate.admit(1), Some(0));
        assert_eq!(rate.admit(2), None);
        assert_eq!(rate.admit(5), Some(1));
    }

    /// claim: REL-19
    #[test]
    fn default_retry_after_by_status_and_message() {
        assert_eq!(default_retry_after(503, "allowance unavailable"), Some(5));
        assert_eq!(
            default_retry_after(503, "provider temporarily unavailable"),
            Some(1)
        );
        assert_eq!(default_retry_after(429, "allowance unavailable"), Some(1));
        assert_eq!(default_retry_after(429, "rate limit exceeded"), Some(1));
        assert_eq!(default_retry_after(400, "bad request"), None);
        assert_eq!(default_retry_after(502, "upstream unavailable"), None);
    }

    /// The client-facing status, type and message for every class of error that ends a request.
    /// claim: REL-1, REL-2
    #[test]
    fn failure_response_maps_every_error_class() {
        type Want = Option<(u16, &'static str, &'static str)>;
        let cases: [(Box<Error>, Want); 18] = [
            (
                Error::new(T::CustomCode("allowance unavailable", 503)),
                Some((503, "api_error", "allowance unavailable")),
            ),
            (
                Error::new(T::CustomCode("too many", 429)),
                Some((429, "rate_limit_error", "too many")),
            ),
            (
                Error::new(T::CustomCode("nope", 404)),
                Some((404, "invalid_request_error", "nope")),
            ),
            (
                Error::new(T::HTTPStatus(400)),
                Some((400, "invalid_request_error", "bad request")),
            ),
            (
                Error::new(T::HTTPStatus(413)),
                Some((413, "invalid_request_error", "request body too large")),
            ),
            (
                Error::new(T::HTTPStatus(429)),
                Some((429, "rate_limit_error", "rate limit exceeded")),
            ),
            (
                Error::new(T::HTTPStatus(502)),
                Some((502, "api_error", "upstream unavailable")),
            ),
            (
                Error::new(T::HTTPStatus(503)),
                Some((503, "api_error", "provider temporarily unavailable")),
            ),
            (
                Error::new(T::HTTPStatus(504)),
                Some((504, "api_error", "upstream timed out")),
            ),
            (
                Error::new(T::HTTPStatus(401)),
                Some((401, "invalid_request_error", "request rejected")),
            ),
            (
                Error::new(T::HTTPStatus(500)),
                Some((500, "api_error", "upstream error")),
            ),
            (Error::new_down(T::ConnectionClosed), None),
            (
                Error::new_down(T::ReadTimedout),
                Some((408, "invalid_request_error", "request body timed out")),
            ),
            (
                Error::new_down(T::InvalidHTTPHeader),
                Some((400, "invalid_request_error", "bad request")),
            ),
            (
                Error::new_up(T::ReadTimedout),
                Some((504, "api_error", "upstream timed out")),
            ),
            (
                Error::new_up(T::ConnectRefused),
                Some((502, "api_error", "could not connect to the provider")),
            ),
            (
                Error::new_up(T::ConnectionClosed),
                Some((502, "api_error", "upstream connection failed")),
            ),
            (
                Error::new(T::InternalError),
                Some((500, "api_error", "internal error")),
            ),
        ];
        for (e, want) in cases {
            assert_eq!(failure_response(&e), want, "{e}");
        }
    }

    /// Only OpenAI's own Chat Completions takes `max_completion_tokens` in place of `max_tokens`:
    /// not another vendor's Chat endpoint, and not OpenAI's Responses.
    /// claim: TRN-5
    #[test]
    fn only_native_openai_chat_is_respelled() {
        let c = |provider, path| route::Candidate {
            provider,
            upstream_model: "m",
            path,
        };
        use providers::ProviderId::{OpenAi, OpenRouter};
        assert!(native_openai_chat(&c(OpenAi, "/v1/chat/completions")));
        assert!(!native_openai_chat(&c(
            OpenRouter,
            "/api/v1/chat/completions"
        )));
        assert!(!native_openai_chat(&c(OpenAi, "/v1/responses")));
    }

    /// A limit already at the serving row's maximum is not rewritten.
    /// claim: TRN-5
    #[test]
    fn a_limit_equal_to_the_cap_is_left_alone() {
        let mut body = br#"{"max_tokens":16384}"#.to_vec();
        let scan = peek::scan_buffered(&body);
        assert!(!clamp_output_limits(&mut body, &scan.limit_spans, 16384));
        assert_eq!(body, br#"{"max_tokens":16384}"#);
    }

    /// An empty `model` value is still replaced with the serving candidate's id.
    /// claim: R1, BIL-13
    #[test]
    fn an_empty_model_value_is_rewritten() {
        let body = br#"{"model":"","messages":[]}"#.to_vec();
        let scan = peek::scan_buffered(&body);
        let span = scan.model_span.expect("an empty value still has a span");
        let out = apply_model_rewrite(body, span, b"gpt-4o");
        assert_eq!(out, br#"{"model":"gpt-4o","messages":[]}"#);
    }

    /// Forcing `include_usage` keeps the client's other `stream_options` members.
    /// claim: BIL-2
    #[test]
    fn forcing_include_usage_keeps_other_stream_options() {
        let body =
            br#"{"stream":true,"stream_options":{"include_usage":false,"include_obfuscation":false}}"#
                .to_vec();
        let at = peek::scan_buffered(&body)
            .stream_options_at
            .expect("streams with options");
        let (out, _) = force_include_usage(body, at, None);
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["stream_options"]["include_usage"], true);
        assert_eq!(v["stream_options"]["include_obfuscation"], false);
    }

    /// D80 resends only when **every** condition holds: a reused connection, pingora's
    /// `ReusedOnly` verdict, a read error, and a reset or abort at the root. With any one false the
    /// request may have reached a server, so it is never sent twice.
    /// claim: REL-1, REL-11
    #[test]
    fn a_reset_before_reading_needs_every_condition() {
        use std::io::{Error as Io, ErrorKind as K};
        let err = |etype: T, kind: K, reused_only: bool| {
            let mut e = Error::because(etype, "upstream", Io::from(kind));
            if reused_only {
                e.retry = pingora_core::RetryType::ReusedOnly;
            }
            e
        };
        let reset = err(T::ReadError, K::ConnectionReset, true);
        assert!(reset_before_reading(&reset, true));
        let aborted = err(T::ReadError, K::ConnectionAborted, true);
        assert!(reset_before_reading(&aborted, true));
        // A fresh connection: the reset says nothing about an idle-close race.
        assert!(!reset_before_reading(&reset, false));
        // Pingora did not judge it retryable on reuse.
        let decided = err(T::ReadError, K::ConnectionReset, false);
        assert!(!reset_before_reading(&decided, true));
        // Not a read: a write that met the reset may have delivered part of the request.
        let write = err(T::WriteError, K::ConnectionReset, true);
        assert!(!reset_before_reading(&write, true));
        // A timeout is a liveness verdict on a peer that may have taken the request.
        let timed_out = err(T::ReadError, K::TimedOut, true);
        assert!(!reset_before_reading(&timed_out, true));
    }

    /// The usage tail at the exact moment it becomes a ring and across its wrap: a chunk that
    /// lands flush on the end (no remainder, `head` back to 0), then a final usage event that
    /// straddles the end. Any slip here truncates or misorders the event billing reads.
    /// claim: BIL-15
    #[test]
    fn the_usage_tail_wraps_at_exactly_its_capacity() {
        let stream: Vec<u8> = (0..2 * USAGE_TAIL_CAP).map(|i| (i % 253) as u8).collect();
        let mut tail = UsageTail::default();
        tail.push(&stream[..USAGE_TAIL_CAP]);
        assert!(
            tail.ring(),
            "exactly the cap is a ring with nothing dropped"
        );
        assert_eq!(tail.head, 0);
        tail.push(&stream[USAGE_TAIL_CAP..USAGE_TAIL_CAP + 10]);
        tail.push(&stream[USAGE_TAIL_CAP + 10..2 * USAGE_TAIL_CAP - 3]);
        assert_eq!(tail.head, USAGE_TAIL_CAP - 3);
        tail.push(&stream[2 * USAGE_TAIL_CAP - 3..]);
        assert_eq!(tail.head, 0, "flush on the end wraps head to the front");
        assert_eq!(tail.contiguous(), &stream[USAGE_TAIL_CAP..]);

        // Outgrown (compacted to head 0), then filled to 5 bytes short of the end, so the usage
        // event's first 5 bytes land at the end and the rest wrap to the front.
        let mut tail = UsageTail::default();
        tail.push(&stream[..USAGE_TAIL_CAP + 5]);
        tail.push(&stream[USAGE_TAIL_CAP + 5..2 * USAGE_TAIL_CAP - 5]);
        assert_eq!(tail.head, USAGE_TAIL_CAP - 10);
        tail.push(&stream[2 * USAGE_TAIL_CAP - 5..]);
        assert_eq!(tail.head, USAGE_TAIL_CAP - 5);
        let event = br#"data: {"usage":{"total_tokens":7}}"#;
        tail.push(event);
        assert_eq!(tail.head, event.len() - 5);
        let mut whole = stream.clone();
        whole.extend_from_slice(event);
        assert_eq!(tail.contiguous(), &whole[whole.len() - USAGE_TAIL_CAP..]);
    }

    /// A body write that met an HTTP/2 stream already closed (D248) is pingora's `H2Error` with a
    /// capacity context, found anywhere down the cause chain; another type or context is not it.
    /// claim: REL-22
    #[test]
    fn only_an_h2_capacity_failure_is_an_unsent_body() {
        for ctx in ["cannot reserve capacity", "while waiting for capacity"] {
            assert!(h2_body_unsent(&Error::explain(T::H2Error, ctx)), "{ctx}");
            let wrapped = Error::because(
                T::WriteError,
                "writing the body",
                Error::explain(T::H2Error, ctx),
            );
            assert!(h2_body_unsent(&wrapped), "caused by {ctx}");
            assert!(
                !h2_body_unsent(&Error::explain(T::WriteError, ctx)),
                "not H2: {ctx}"
            );
        }
        assert!(!h2_body_unsent(&Error::explain(T::H2Error, "stream reset")));
        assert!(!h2_body_unsent(&Error::new(T::H2Error)));
    }

    /// With no key to scrub (none sent, or an empty one) and no body to hold, a chunk is relayed as
    /// it came, never searched; a body held whole is held whether or not there is a key.
    /// claim: SEC-7
    #[test]
    fn redact_holds_a_whole_body_even_with_no_key() {
        let chunk = Bytes::from_static(br#"{"error":{"message":"x"}}"#);
        let empty = memchr::memmem::Finder::new(b"");
        let mut r = Redact::default();
        let got = r.feed(std::slice::from_ref(&empty), Some(chunk.clone()), false);
        assert_eq!(got.map(|b| b.as_ptr()), Some(chunk.as_ptr()));
        let mut r = Redact {
            whole: true,
            status: 400,
            ..Redact::default()
        };
        assert_eq!(r.feed(&[], Some(chunk.clone()), false), Some(Bytes::new()));
        assert_eq!(r.feed(&[], None, true), Some(chunk));
    }

    /// An attempt that ended unanswered is a 502 that says whether it panicked.
    /// claim: REL-1
    /// defect: D207
    #[test]
    fn an_unanswered_attempt_says_whether_it_panicked() {
        let context = |e: Box<Error>| e.context.as_ref().map(|c| c.as_str().to_owned());
        let panicked = Reaped::Panicked.unanswered();
        assert_eq!(panicked.etype(), &T::HTTPStatus(502));
        assert_eq!(
            context(panicked).as_deref(),
            Some("full-body attempt panicked")
        );
        for quiet in [Reaped::Finished, Reaped::StillRunning] {
            assert_eq!(
                context(quiet.unanswered()).as_deref(),
                Some("full-body attempt ended without a response")
            );
        }
    }

    /// An `anthropic-beta` line whose every token is allowed goes out exactly as the client wrote
    /// it, spacing included: only a value with a token to drop is rebuilt.
    /// claim: SEC-6
    #[test]
    fn an_allowed_beta_line_is_forwarded_untouched() {
        let value = "prompt-caching-2024-07-31,  interleaved-thinking-2025-05-14";
        let mut req = pingora::http::RequestHeader::build("POST", b"/v1/messages", None).unwrap();
        req.insert_header("anthropic-beta", value).unwrap();
        retain_managed_client_headers(&mut req, true, true).unwrap();
        assert_eq!(req.headers.get("anthropic-beta").unwrap(), value);
    }

    /// Fast mode's beta rides on a managed request to direct Anthropic, the only provider with fast
    /// mode, and is dropped on the way to any other (D266).
    /// claim: SEC-6
    /// defect: D266
    #[test]
    fn the_fast_mode_beta_reaches_direct_anthropic_only() {
        let value = "prompt-caching-2024-07-31,fast-mode-2026-02-01";
        let header = |anthropic| {
            let mut req =
                pingora::http::RequestHeader::build("POST", b"/v1/messages", None).unwrap();
            req.insert_header("anthropic-beta", value).unwrap();
            retain_managed_client_headers(&mut req, anthropic, true).unwrap();
            req.headers
                .get("anthropic-beta")
                .map(|v| v.to_str().unwrap().to_owned())
        };
        assert_eq!(header(true).as_deref(), Some(value));
        assert_eq!(header(false).as_deref(), Some("prompt-caching-2024-07-31"));
    }

    /// The fine-grained-tool-streaming beta rides only to a candidate that takes fine-grained tool
    /// streaming; for any other (an older Bedrock stack) it goes with each tool's
    /// `eager_input_streaming` (D272). Alone on the line, the header goes entirely.
    /// claim: SEC-6
    /// defect: D272
    #[test]
    fn the_fine_grained_tool_streaming_beta_reaches_only_a_candidate_that_takes_it() {
        let header = |value: &str, eager_tool_streaming| {
            let mut req =
                pingora::http::RequestHeader::build("POST", b"/v1/messages", None).unwrap();
            req.insert_header("anthropic-beta", value).unwrap();
            retain_managed_client_headers(&mut req, false, eager_tool_streaming).unwrap();
            req.headers
                .get("anthropic-beta")
                .map(|v| v.to_str().unwrap().to_owned())
        };
        let both = "fine-grained-tool-streaming-2025-05-14,prompt-caching-2024-07-31";
        assert_eq!(header(both, true).as_deref(), Some(both));
        assert_eq!(
            header(both, false).as_deref(),
            Some("prompt-caching-2024-07-31")
        );
        assert_eq!(header(FINE_GRAINED_TOOL_STREAMING_BETA, false), None);
    }
}
