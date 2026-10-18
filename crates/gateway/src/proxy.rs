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
use crate::rewrite::{STREAM_OPTIONS_FRAG, ScannedBody};
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

mod billing;
mod full_body;
mod headers;
mod hooks;
mod phases;
mod relay;
mod replies;
mod serving;
mod upstream;

pub use self::replies::error_body;
pub(crate) use self::upstream::{is_candidate_refusal, is_pool_key_failure};
use self::{
    billing::*, full_body::*, headers::*, phases::*, relay::*, replies::*, serving::*, upstream::*,
};

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
mod tests;

/// Behaviors a mutation-testing pass found no test constraining.
#[cfg(test)]
#[path = "proxy/mutation_gaps_tests.rs"]
mod mutation_gaps;

// The body edits are `rewrite::ScannedBody`'s, on a body that carries its own scan. These are
// their raw-offset forms, for the tests that pin one edit's bytes at a time.

/// Cap each root-level output limit at `max`, back to front, on the scan's raw spans
/// (`ScannedBody::clamp_output_limits`). `true` when a value changed.
#[cfg(test)]
fn clamp_output_limits(body: &mut Vec<u8>, spans: &[Option<(usize, usize)>; 3], max: u32) -> bool {
    crate::rewrite::raw::clamp_output_limits(body, spans, max)
}

/// Respell a root `max_tokens` as `max_completion_tokens` (`ScannedBody::rename_max_tokens` on
/// the scan's raw `limit_keys`). `true` when the body changed.
#[cfg(test)]
fn rename_max_tokens(body: &mut Vec<u8>, keys: &[Option<usize>; 3]) -> bool {
    crate::rewrite::raw::rename_max_tokens(body, keys)
}

/// Overwrite the `model` value's bytes at `span` with `replacement`, or leave them when they
/// already match (step 3 of `ScannedBody::into_upstream`).
#[cfg(test)]
fn apply_model_rewrite(body: Vec<u8>, span: (usize, usize), replacement: &[u8]) -> Vec<u8> {
    crate::rewrite::raw::apply_model_rewrite(body, span, replacement)
}

/// Make a client-sent `stream_options` (value at `at`) ask for usage (step 2 of
/// `ScannedBody::into_upstream`), returning `model_span` moved by the edit.
#[cfg(test)]
fn force_include_usage(
    body: Vec<u8>,
    at: usize,
    model_span: Option<(usize, usize)>,
) -> (Vec<u8>, Option<(usize, usize)>) {
    crate::rewrite::raw::force_include_usage(body, at, model_span)
}

/// Splice `stream_options.include_usage` in at `at`, or nothing when there is no point (step 4
/// of `ScannedBody::into_upstream`).
#[cfg(test)]
fn apply_stream_usage_injection(body: Vec<u8>, at: Option<usize>) -> Vec<u8> {
    crate::rewrite::raw::apply_stream_usage_injection(body, at)
}
