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
use crate::circuit_breaker::Permit;
use crate::key;
use crate::metrics::{KeyCooled, Rejection};
use crate::route::{self, Dialect, Provider};
use crate::signed_id;
use crate::state::{GatewayState, RequestId};
use crate::terminal::TerminalTracker;
use crate::{control, peek, pin, remedy, translate, unpriced, usage};
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
use pingora_proxy::{FailToProxy, ProxyHttp, Session};
use std::borrow::Cow;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
/// `KNOWN_PROVIDERS` among the providers pi attributes to (NVIDIA NIM, Cloudflare, Vercel AI
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
        if state.config.request_max_secs > 0 {
            crate::deadline::start();
        }
        let state: &'static Arc<GatewayState> = Box::leak(Box::new(state));
        Self { state }
    }
}

/// Requests that exist right now: incremented when pingora builds a request's context
/// (`new_ctx`, once the request header has been read) and decremented when that context drops,
/// after `logging` has written its billing row. `main`'s shutdown drain exits once this reaches 0.
/// One atomic add and one sub per request, the only shared write `new_ctx` and the context's drop
/// make.
pub static LIVE_REQUESTS: AtomicUsize = AtomicUsize::new(0);

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

/// The releasable state of one request. Kept beside, not inside, [`RequestCtx`]: none of it is
/// touched per response chunk, and `RequestCtx`'s size is (see its size test).
struct Held {
    state: &'static GatewayState,
    /// Set at the top of `request_filter`, so an error answered before admission (a body read
    /// failure, say) still carries the request id.
    request_id: Option<RequestId>,
    /// Counted on `ai_requests_in_flight`.
    in_flight: bool,
    /// Counted on `ai_active_streams`.
    active_stream: bool,
    /// Holds one of this tenant's `tenant_max_in_flight` slots.
    tenant: Option<u64>,
    /// Bytes this request holds in the process body budget (`max_buffered_body_bytes`).
    body_bytes: usize,
    /// A `FullBody` re-run: its parent reserved the budget for both copies of the body.
    body_exempt: bool,
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
        if let Some(tenant) = self.tenant.take()
            && let Some(slots) = self.state.tenant_slots.as_ref()
        {
            slots.release(tenant);
        }
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
        if let Some(rc) = self.rc.as_mut()
            && let Some(permit) = rc.breaker_pending.take()
            && let Some(b) = rc.provider.breaker.as_ref()
        {
            b.release(permit);
        }
        LIVE_REQUESTS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Per-request context. `None` until `request_filter` admits the request; short-circuited
/// requests (auth/deny failures) leave it `None`, so later filters no-op.
pub struct RequestCtx {
    tenant_id: u64,
    vpc_id: u64,
    /// `bai_v2` credential id. `None` for v1 (and BYO). Emitted on `ai.usage`; used with the deny-set.
    key_id: Option<u64>,
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
    /// Whether this is a **managed** request (`bai_v1…` / `bai_v2…` key → swap to the pool key). `false` for
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
    /// Upstream HTTP status, set in `response_filter` once the response head arrives, which is
    /// also where the breaker permit is resolved (`5xx` → failure, any other response → success:
    /// the provider answered, and a `429` is a healthy throttle). A `None` here at `logging` with an
    /// upstream error → failure (connect/read failed before any response).
    upstream_status: Option<u16>,
    /// Managed OpenAI chat/completions request: buffer the body and inject
    /// `stream_options.include_usage` if it streams without it, so the usage chunk (hence the
    /// billable token count) is guaranteed. The single, deliberate exception to "never buffer the
    /// request body" — scoped to the managed OpenAI chat/completions path (see `is_streamable_path`)
    /// and bounded by `MAX_REQUEST_BODY`. BYO and every other request still stream straight through.
    inject_eligible: bool,
    /// Managed `/{provider}/…` billable call (every allowlisted endpoint but the free token
    /// counts): buffer the body (as `inject_eligible` does) so a request whose root `model` names
    /// no catalog row ([`price_row`]) is refused before any byte of it goes upstream, with the
    /// catalog walk's 404. A managed key runs priced models only (D267). BYO keys never set it.
    ///
    /// The same buffer refuses a `/responses` body's `background: true` (D202): that endpoint is
    /// billable, so it is always checked here (a catalog walk reads its body in `request_filter`
    /// and refuses there).
    catalog_check: bool,
    /// Accumulated request body — populated only when `inject_eligible`; otherwise stays empty and
    /// the body is never buffered.
    req_buf: Vec<u8>,
    start: Instant,
    /// When this request must end (`request_max_secs`), in [`crate::deadline`]'s units;
    /// [`crate::deadline::NONE`] when the ceiling is off. A `FullBody` attempt carries its
    /// parent's, so a re-run does not restart the clock.
    deadline: u64,
    /// Connect-retry counter (see `fail_to_connect`).
    attempt: u8,
    /// Index into `provider.pool_auth` of the key used on this attempt. Starts on
    /// `Provider::first_key` (past keys cooling off from a 401). Advanced on a managed 429 or
    /// 401 when another unused key remains. Reset to the new provider's first key when
    /// `provider` changes — never send provider A's key to provider B.
    pool_key: u8,
    /// The previous attempt was a same-provider key walk. `upstream_peer` must not treat that as a
    /// candidate/breaker failure (a 429 is a healthy throttle) and must not pick a new vendor.
    same_provider_retry: bool,
    /// This is a [`FullBody`] attempt that recorded a [`RelayRetry`]: the parent is discarding its
    /// response and re-running. It still feeds the breaker; it writes no `ai.usage`
    /// or `ai.payload` row (the attempt that serves does).
    relay_abandoned: bool,
    /// This attempt is the one resend a refused H2 stream gets on its candidate (D72, D91). A
    /// second refusal there is a provider failure: fail over, or end the request. Cleared when the
    /// walk moves to a new candidate.
    refused_resent: bool,
    /// This attempt's upstream read timeout is the time left before `deadline` (shorter than the
    /// silence bound, see `cap_read_at_deadline`), so a read timeout on it is the deadline passing,
    /// not the provider going quiet.
    read_capped: bool,
    /// The permit an `allow()` on `provider`'s breaker handed out, while it is outstanding and
    /// still owes exactly one `record_*_for` or `release`. Resolving with the permit (not just
    /// "the breaker") is what lets a half-open breaker ignore an attempt whose probe permit it has
    /// since reclaimed (D256).
    ///
    /// The ledger that keeps breaker accounting honest once a request can attempt more than one
    /// provider. Invariant: `breaker_pending` is `Some` **iff** there is exactly one unresolved
    /// `allow()` against whatever `provider` currently points at. `logging` records only when it is
    /// set, so an attempt can never be recorded twice, and a candidate switch resolves the outgoing
    /// candidate before claiming the next one.
    ///
    /// On the `/{provider}/…` path this is the permit `request_filter`'s one `allow()` returned
    /// (`None` when the provider has no breaker).
    breaker_pending: Option<Permit>,
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
    /// Text bytes of the request body (base64 payloads excluded): the input-token estimate for a
    /// billing row the provider's usage did not fill (a stream cut short before its usage block, on
    /// a wire that reports input only at the end). See `usage::InputTally`. Fed as the body streams
    /// past only when `tally_eager`; otherwise `logging` reads the body itself, and only when it
    /// needs an estimate ([`input_estimate`]).
    input_tally: usage::InputTally,
    /// The body is fed to `input_tally` as it streams: a managed provider-routed body that may
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
    /// Billing facts tapped beside the usage block ([`BillingTaps`]): managed only, created at the
    /// first fact. Boxed for [`ModelRouting`]'s reason; `None` on every BYO request.
    taps: Option<Box<BillingTaps>>,
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
    if let Some((_, k)) = rc.taps.as_mut().and_then(|t| t.knobs.take()) {
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
const MAY_CONTINUE_PROVIDERS: [&str; 3] = ["openrouter", "bedrock", "groq"];

/// Whether this row's provider may have generated (and billed) past the point the row counts:
/// a stream the client cancelled, or one cut short, on a provider in [`MAY_CONTINUE_PROVIDERS`].
fn upstream_may_continue(provider: Option<&str>, outcome: &str, streaming: bool) -> bool {
    streaming
        && matches!(outcome, "client_cancelled" | "cut_short")
        && provider.is_some_and(|p| MAY_CONTINUE_PROVIDERS.contains(&p))
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
                && rc.forward_path.as_deref().is_some_and(|p| {
                    let p = p.split_once('?').map_or(p, |(path, _)| path);
                    p.ends_with("/responses") || p.ends_with("/responses/compact")
                })
        }
    };
    // Always made on a managed response: `resp_model_scanner` keeps the response's id in it.
    let taps = rc.taps.get_or_insert_with(Box::default);
    taps.generation_id = generation_id;
    taps.request_id = request_id;
    taps.tools = responses.then(|| usage::ToolTally::new(rc.streaming));
}

/// Fold what the taps saw into the row's usage: the header generation id over the body's (the
/// response scanner's, from the head, over the usage block's), the serving host the head named,
/// and the Responses tool items.
fn merge_taps(rc: &mut RequestCtx, usage: &mut usage::Usage) {
    let Some(t) = rc.taps.as_mut() else { return };
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

/// What became of a request, on its billing row: a consumer must be able to tell a zero-token row
/// for an upstream error or a cancel from a real zero-token generation.
fn outcome(rc: &RequestCtx, e: Option<&pingora_core::Error>, cache_hit: bool) -> &'static str {
    if cache_hit {
        return "ok";
    }
    if rc.upstream_status.is_some_and(|s| s >= 400) {
        return "upstream_error";
    }
    if e.is_some_and(|e| e.esource() == &pingora_core::ErrorSource::Downstream) {
        return "client_cancelled";
    }
    if rc.upstream_phase == UpstreamPhase::None {
        return "no_candidate";
    }
    match (e, rc.upstream_status) {
        // Failed before any response head: a connect, read-timeout or reset failure upstream.
        (Some(_), None) => "upstream_error",
        // A response that started and then died.
        (Some(_), Some(_)) => "cut_short",
        (None, _) => "ok",
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
fn is_pool_key_failure(status: u16) -> bool {
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
fn is_candidate_refusal(status: u16) -> bool {
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

impl RequestCtx {
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
    /// Why a body gets rewritten:
    /// - `inject_eligible`: splicing `stream_options` into a managed OpenAI stream.
    /// - a catalog walk (`auto`): re-spelling `model` for the candidate serving this attempt,
    ///   translating between Chat Completions, Messages and Responses when the wires differ, and
    ///   the same-wire edits a walk makes (the output-limit cap, `max_tokens` respelled for native
    ///   OpenAI, dropped nulls and unreplayable reasoning).
    /// - `catalog_check`: a managed `/{provider}/…` billable body is held whole so a model outside
    ///   the catalog (D267), or a `/responses` body's `background: true` (D202), is refused before
    ///   any byte goes upstream.
    /// - `signed`: a managed Responses relay to a provider's store swaps each tenant-signed id for
    ///   the provider's (`signed_id`, D230).
    ///
    /// Several can apply to the same request, in which case every edit is made to the one buffer.
    fn rewrites_body(&self) -> bool {
        self.inject_eligible || self.catalog_check || self.auto.is_some() || self.signed.is_some()
    }

    /// Whether the bytes sent to the client are watched for the stream's terminal event
    /// (`closed_after_terminal`): a managed stream's, unless it is assembled into one JSON body.
    /// The verdict is read only on a managed request's billing row, and a JSON body has no
    /// terminal event, so watching anything else would scan every chunk for nothing.
    fn tracks_terminal(&self) -> bool {
        self.managed && self.streaming && !self.auto.as_ref().is_some_and(|a| catalog_assembling(a))
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
    fn set_forward_path(&mut self, path: &str) {
        let buf = self.forward_path.get_or_insert_with(String::new);
        buf.clear();
        buf.push_str(path);
    }

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
    fn reset_request_body_phase(&mut self) {
        // Translate response state belongs to *this* attempt. 5xx vendor-walk aborts in
        // `upstream_response_filter` before `response_filter` runs, so these are empty in
        // practice — still clear them so a retry cannot append a leftover JSON/SSE buffer
        // onto the candidate that actually serves.
        if let Some(t) = self.auto.as_mut().and_then(|a| a.translate.as_mut()) {
            t.sse = None;
            t.json_buf.clear();
            t.assemble = false;
        }
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
        self.input_tally = usage::InputTally::default();
        if let Some(k) = self.taps.as_mut().and_then(|t| t.knobs.as_mut()) {
            *k = (peek::ModelScanner::new(), peek::Kept::request_knobs());
        }
    }
}

/// Every flood-path `(error_type, message)` pair, paired with its wire body.
///
/// `reject` is only ever called with these literals, so the body is a compile-time constant.
/// Catalog-walk errors that echo a caller-supplied name go through `reject_message` and allocate.
/// Kept as a table so `reject_bodies_are_valid_json` can walk it and assert each entry parses,
/// carries the `type` and `message` it claims, and is reachable — a hand-written JSON literal is
/// exactly the thing that rots silently otherwise.
pub static REJECT_BODIES: [(&str, &str, &str); 16] = [
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
        "model routing requires a managed key",
        r#"{"error":{"message":"model routing requires a managed key","type":"invalid_request_error"}}"#,
    ),
    (
        "api_error",
        "no provider available for model",
        r#"{"error":{"message":"no provider available for model","type":"api_error"}}"#,
    ),
    (
        "insufficient_quota",
        "quota exhausted",
        r#"{"error":{"message":"quota exhausted","type":"insufficient_quota"}}"#,
    ),
    (
        "api_error",
        "allowance unavailable",
        r#"{"error":{"message":"allowance unavailable","type":"api_error"}}"#,
    ),
    (
        "rate_limit_error",
        "too many concurrent requests",
        r#"{"error":{"message":"too many concurrent requests","type":"rate_limit_error"}}"#,
    ),
    (
        "api_error",
        "too many large request bodies in flight",
        r#"{"error":{"message":"too many large request bodies in flight","type":"api_error"}}"#,
    ),
    (
        "api_error",
        "upstream timed out after receiving the request",
        r#"{"error":{"message":"upstream timed out after receiving the request","type":"api_error"}}"#,
    ),
    (
        "api_error",
        "upstream failed after receiving the request",
        r#"{"error":{"message":"upstream failed after receiving the request","type":"api_error"}}"#,
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
    for &(t, m, body) in &REJECT_BODIES {
        if t == typ && m == msg {
            return Bytes::from_static(body.as_bytes());
        }
    }
    Bytes::from(serde_json::json!({ "error": { "type": typ, "message": msg } }).to_string())
}

impl AiProxy {
    /// The bound [`peek_body_model`] puts on each body read itself: `client_read_timeout_secs`,
    /// where the session has no read timeout of its own (HTTP/2; HTTP/1.1 enforces it inside
    /// pingora, so a second timer would be waste).
    fn up_front_read_timeout(&self, session: &Session) -> Option<Duration> {
        let secs = self.state.config.client_read_timeout_secs;
        (secs > 0 && session.as_ref().get_read_timeout().is_none())
            .then(|| Duration::from_secs(secs))
    }

    /// End a request at its deadline (`request_max_secs`): counted, logged, and answered 504 if no
    /// response has started (after one has, pingora cuts the stream as any mid-stream error does).
    fn deadline_cut(&self, request_id: &str) -> Box<pingora_core::Error> {
        self.state
            .metrics
            .rejection(Rejection::RequestDeadline)
            .inc();
        warn!(
            request_id,
            limit_secs = self.state.config.request_max_secs,
            "request reached request_max_secs; ending it",
        );
        gateway_error(504, REQUEST_DEADLINE)
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
        rc.read_capped = false;
        if let Some(left) = crate::deadline::remaining(rc.deadline) {
            if left.is_zero() {
                return Err(self.deadline_cut(&rc.request_id));
            }
            if peer.options.read_timeout.is_none_or(|t| left < t) {
                peer.options.read_timeout = Some(left);
                rc.read_capped = true;
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
        // Pingora allows one stream per upstream H2 connection unless told otherwise, which made
        // every concurrent request open its own TLS connection: H2 without the multiplexing (D160).
        // The provider's own SETTINGS_MAX_CONCURRENT_STREAMS still caps this (pingora takes the
        // lower), and a full connection makes the next request open another.
        peer.options.max_h2_streams = UPSTREAM_H2_MAX_STREAMS;
        // Cert verification is on everywhere except the bench's self-signed TLS mock (see config).
        if !self.state.config.upstream_verify_cert {
            peer.options.verify_cert = false;
            peer.options.verify_hostname = false;
        }
        peer.options.connection_timeout =
            Some(Duration::from_secs(self.state.config.connect_timeout_secs));
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
            .then(|| Duration::from_secs(self.state.config.read_timeout_secs));
        peer.options.h2_ping_interval = self.state.config.h2_ping_interval();
        peer.options.tcp_keepalive = self.state.config.upstream_tcp_keepalive();
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
        Self::reject_retry_after(
            session,
            request_id,
            status,
            typ,
            msg,
            default_retry_after(status, msg),
        )
        .await
    }

    /// [`Self::reject`] with an explicit `Retry-After` (seconds). `reject` derives one for every
    /// 429 and 503 (see [`default_retry_after`]); the open-breaker 503 knows better, the seconds
    /// left until it half-opens.
    async fn reject_retry_after(
        session: &mut Session,
        request_id: &str,
        status: u16,
        typ: &str,
        msg: &str,
        retry_after: Option<u64>,
    ) -> Result<bool> {
        log_rejection(request_id, status, typ);
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

    /// Take this tenant's concurrency slot before reading a body in full. `Ok(None)` when no cap is
    /// configured; `Err` when the tenant is at its cap.
    fn take_slot_before_read(
        &self,
        tenant_id: u64,
    ) -> std::result::Result<Option<SlotGuard<'_>>, ()> {
        let Some(slots) = self.state.tenant_slots.as_ref() else {
            return Ok(None);
        };
        if !slots.try_acquire(tenant_id) {
            return Err(());
        }
        Ok(Some(SlotGuard {
            slots: Some(slots),
            tenant: tenant_id,
        }))
    }

    async fn reject_tenant_busy(&self, session: &mut Session, request_id: &str) -> Result<bool> {
        self.state
            .metrics
            .rejection(Rejection::TenantConcurrency)
            .inc();
        Self::reject_boxed(
            session,
            request_id,
            429,
            "rate_limit_error",
            "too many concurrent requests",
        )
        .await
    }

    /// The body asks for what the row's card says it does not accept (`route::refused_input`: image
    /// input, tools): a 400 naming the row and the kind, before any upstream sees it.
    async fn reject_refused_input(
        &self,
        session: &mut Session,
        request_id: &str,
        row: &route::ModelRoute,
        kind: &str,
    ) -> Result<bool> {
        self.state.metrics.rejection(Rejection::Modality).inc();
        Self::reject_message_boxed(
            session,
            request_id,
            400,
            "invalid_request_error",
            format!("{} does not accept {kind}", row.model),
        )
        .await
    }

    /// Holding this request's body would cross `max_buffered_body_bytes`: a retryable 503, since
    /// the memory frees as the bodies in flight finish.
    async fn reject_body_memory(&self, session: &mut Session, request_id: &str) -> Result<bool> {
        self.state.metrics.rejection(Rejection::BodyMemory).inc();
        Self::reject_boxed(
            session,
            request_id,
            503,
            "api_error",
            "too many large request bodies in flight",
        )
        .await
    }

    /// A body that reached [`MAX_REQUEST_BODY`] while the gateway was reading it to choose a row.
    async fn reject_too_large(&self, session: &mut Session, request_id: &str) -> Result<bool> {
        self.state.metrics.rejection(Rejection::BodyTooLarge).inc();
        Self::reject_boxed(
            session,
            request_id,
            413,
            "invalid_request_error",
            "request body too large",
        )
        .await
    }

    /// Catalog-walk errors that must echo a caller-supplied name (unknown model, wire mismatch).
    /// Allocating is fine: this is not the flood path the static [`REJECT_BODIES`] table exists for.
    async fn reject_message(
        session: &mut Session,
        request_id: &str,
        status: u16,
        typ: &'static str,
        msg: String,
    ) -> Result<bool> {
        log_rejection(request_id, status, typ);
        let body = Bytes::from(
            serde_json::json!({ "error": { "type": typ, "message": msg } }).to_string(),
        );
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

    /// A managed Responses `background: true` request (D202): the provider answers `queued` with
    /// no usage and generates after the request ends, so nothing the gateway relays could meter
    /// it, and a managed key cannot poll for the result (GET is refused).
    async fn reject_background(&self, session: &mut Session, request_id: &str) -> Result<bool> {
        self.state
            .metrics
            .rejection(Rejection::ManagedEndpoint)
            .inc();
        Self::reject_message_boxed(
            session,
            request_id,
            400,
            "invalid_request_error",
            BACKGROUND_REFUSED.to_owned(),
        )
        .await
    }

    async fn reject_message_boxed(
        session: &mut Session,
        request_id: &str,
        status: u16,
        typ: &'static str,
        msg: String,
    ) -> Result<bool> {
        Box::pin(Self::reject_message(session, request_id, status, typ, msg)).await
    }

    async fn reject_catalog_miss(
        session: &mut Session,
        request_id: &str,
        name: Option<&str>,
    ) -> Result<bool> {
        let msg = catalog_miss_message(name, true);
        Self::reject_message_boxed(session, request_id, 404, "invalid_request_error", msg).await
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
    /// `FullBody` re-runs see the body only as it streams in). `Some` when the request was refused
    /// and the reply is written.
    async fn refuse_foreign_ids(
        &self,
        session: &mut Session,
        request_id: &str,
        route: &'static route::ModelRoute,
        body: &[u8],
        tenant_id: u64,
    ) -> Option<Result<bool>> {
        if !signed_id::catalog_applies(session.req_header().uri.path(), route) {
            return None;
        }
        let (reason, status, typ, msg) = match self.state.id_signer.as_ref() {
            None => (
                Rejection::IdSigningUnset,
                503,
                "api_error",
                "managed Responses ids cannot be issued: the gateway has no id signing key configured"
                    .to_owned(),
            ),
            Some(signer) => {
                let refusal = signer.unsign_request(tenant_id, body, true).err()?;
                (
                    Rejection::ForeignId,
                    400,
                    "invalid_request_error",
                    refusal.to_string(),
                )
            }
        };
        self.state.metrics.rejection(reason).inc();
        Some(Self::reject_message_boxed(session, request_id, status, typ, msg).await)
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
        if let Some(kind) = route::refused_input(route, &body) {
            return self
                .reject_refused_input(session, &request_id, route, kind)
                .await;
        }
        if route::is_responses_path(session.req_header().uri.path()) && requests_background(&body) {
            return self.reject_background(session, &request_id).await;
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
                return Self::reject_message_boxed(
                    session,
                    &request_id,
                    500,
                    "api_error",
                    "internal error".to_owned(),
                )
                .await;
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
        Self::reject_message_boxed(
            session,
            &request_id,
            502,
            "api_error",
            "no candidate provider available".to_owned(),
        )
        .await
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
            Some(false) => rc.provider.strike_unread_402(rc.pool_key),
            None => false,
        };
        if cool {
            self.cool_unfunded_key(&rc.request_id, &rc.provider, rc.pool_key, status);
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
        a.health = None;
        // The head resolved this attempt's breaker permit as a success (see `response_filter`):
        // only the body shows the provider is broken. Count it as the failure it is, so a host that
        // keeps answering errors in 200s opens its breaker and every walk, session pins included,
        // skips it. The windowed rule trips once failures reach the threshold and match the
        // successes, which one success plus one failure per such answer does.
        if verdict && let Some(b) = rc.provider.breaker.as_ref() {
            b.record_failure();
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
            self.state
                .metrics
                .rejection(Rejection::TranslateTooLarge)
                .inc();
            return Err(gateway_error(413, TRANSLATE_TOO_LARGE).into_down());
        }
        match budget.hold(heap) {
            Some(hold) => Ok(Some(hold)),
            None => {
                self.state.metrics.rejection(Rejection::BodyMemory).inc();
                Err(gateway_error(503, "too many large request bodies in flight").into_down())
            }
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
    "fine-grained-tool-streaming-2025-05-14",
    "context-management-2025-06-27",
    "token-efficient-tools-2025-02-19",
    "output-128k-2025-02-19",
    translate::THINKING_BINDING_BETA,
];

/// Drop every client header a managed request may not forward ([`MANAGED_FORWARD_HEADERS`]), and
/// every `anthropic-beta` token not in [`MANAGED_ANTHROPIC_BETAS`], less
/// [`translate::FAST_MODE_BETA`] when `anthropic` (the attempt goes to direct Anthropic, the only
/// provider with fast mode). Fast mode is a product Beyond sells at its price: the row records the
/// speed asked and the speed served (`usage.speed`), so it is billed as served (D266). The header
/// sweep allocates nothing (D92: it collected the names into a `Vec` on nearly every managed
/// request); only an `anthropic-beta` value with a token to drop is rebuilt.
fn retain_managed_client_headers(
    req: &mut pingora::http::RequestHeader,
    anthropic: bool,
) -> Result<()> {
    let allowed = |t: &str| {
        MANAGED_ANTHROPIC_BETAS.contains(&t) || (anthropic && t == translate::FAST_MODE_BETA)
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
/// both [`price_model`] and the managed `/{provider}` refusal ([`RequestCtx::catalog_check`])
/// make on the requested model, so a managed `/{provider}` request is refused exactly when its
/// requested model would leave its billing row without a `price_model`.
fn price_row(m: &str) -> Option<&'static route::ModelRoute> {
    route::model_route(m).or_else(|| strip_snapshot_date(m).and_then(route::model_route))
}

/// `m` without a trailing `-YYYY-MM-DD` (OpenAI) or `-YYYYMMDD` (Anthropic) snapshot date.
fn strip_snapshot_date(m: &str) -> Option<&str> {
    let digits = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());
    let (head, last) = m.rsplit_once('-')?;
    if digits(last, 8) {
        return Some(head);
    }
    if !digits(last, 2) {
        return None;
    }
    let (head, month) = head.rsplit_once('-')?;
    let (head, year) = head.rsplit_once('-')?;
    (digits(month, 2) && digits(year, 4)).then_some(head)
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
    e.is_some_and(|e| {
        !matches!(e.esource(), pingora_core::ErrorSource::Downstream) && !is_deadline(e)
    })
}

/// Whether this attempt had started sending the client's body upstream and the client had not
/// finished it. An upstream error then reflects the client (a stalled or abandoned upload), not the
/// provider. Zero bytes fed means the attempt never got past connecting.
fn client_still_uploading(session: &mut Session, rc: &RequestCtx) -> bool {
    rc.body_bytes_fed > 0 && !session.as_mut().is_body_done()
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

/// A tenant concurrency slot taken before the gateway reads a request body in full, so
/// `tenant_max_in_flight` bounds the bodies held in memory and not only the requests in flight.
/// Released on drop (every early return), or handed to the request context with [`Self::hand_over`]
/// once the request is admitted, after which `logging` releases it.
struct SlotGuard<'a> {
    slots: Option<&'a crate::concurrency::TenantSlots>,
    tenant: u64,
}

impl SlotGuard<'_> {
    fn hand_over(mut self) {
        self.slots = None;
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if let Some(slots) = self.slots {
            slots.release(self.tenant);
        }
    }
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

/// A retry a [`FullBody`] subrequest hands back to its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelayRetry {
    /// A candidate failure (a 5xx, a pool-key 401/402/403, or a connection that failed before the
    /// upstream had the whole body) from this catalog index with another candidate left: skip it.
    Candidate(u8),
    /// A 429 from this catalog index with another pool key left: resume there.
    Key { candidate: u8, key: u8 },
    /// A reused connection failed before the upstream had the whole body (a pooled connection
    /// the provider had already closed): try this candidate once more on a fresh connection.
    Reset(u8),
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
        return rc.input_tally.estimate_tokens();
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
    session
        .subrequest_ctx
        .as_ref()
        .and_then(|c| c.user_ctx())
        .and_then(|u| u.downcast_ref::<FullBody>())
        .cloned()
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

/// The `Retry-After` (seconds) on a gateway-made rejection: every 429 and 503 carries one, so a
/// stock SDK backs off rather than hammering or giving up. A rate limit or a tenant's concurrency
/// cap frees up within a second; an allowance-set not yet read takes a few; anything else gets 1.
fn default_retry_after(status: u16, msg: &str) -> Option<u64> {
    match status {
        503 if msg == "allowance unavailable" => Some(5),
        429 | 503 => Some(1),
        _ => None,
    }
}

/// An error the gateway raises with the status and message the client should see. Carried in
/// `ErrorType::CustomCode` so [`failure_response`] can answer with exactly this rather than a
/// generic line for the status.
/// The 413 for a body whose translation would take more heap than the whole body budget (D216).
const TRANSLATE_TOO_LARGE: &str = "request body is too large to translate onto this model's API \
     within the gateway's memory budget; send it on the model's own API, or send less";

fn gateway_error(status: u16, msg: &'static str) -> Box<pingora_core::Error> {
    pingora_core::Error::new(pingora_core::ErrorType::CustomCode(msg, status))
}

/// The [`gateway_error`] tag for a managed `/{provider}` body whose model names no catalog row
/// (D267). Never shown as is: `fail_to_proxy` answers it with [`catalog_miss_message`] for the
/// model the request carried, the catalog walk's own 404.
const CATALOG_MISS: &str = "model is not in the catalog";

/// The catalog-miss message, for the catalog walk (`reject_catalog_miss`) and for a managed
/// `/{provider}` body (D267) alike: the model by name (clipped), or that none was sent, and where
/// the models a managed key can use are listed.
/// `header`: whether the route reads `x-beyond-model` (a catalog walk does, `/{provider}` does not).
fn catalog_miss_message(name: Option<&str>, header: bool) -> String {
    match name.map(str::trim).filter(|s| !s.is_empty()) {
        None => format!(
            "missing model: set the JSON body's model{} to a catalog name; GET /v1/models lists \
             them",
            if header { " (or x-beyond-model)" } else { "" }
        ),
        Some(n) => format!(
            "model \"{}\" is not in the catalog; a managed key runs catalog models only (GET \
             /v1/models lists them)",
            clip_catalog_name(n)
        ),
    }
}

/// The 504 for a request that outlived `request_max_secs` before its response head.
const REQUEST_DEADLINE: &str = "request exceeded the gateway's maximum duration";

/// Whether `e` is the request deadline passing ([`AiProxy::deadline_cut`]). A policy limit, never a
/// provider outcome: [`is_upstream_failure`] is false for it, so no breaker hears of it, and
/// `error_while_proxy` passes it through as the gateway's own decision (no resend, no failover).
fn is_deadline(e: &pingora_core::Error) -> bool {
    matches!(e.etype(), pingora_core::ErrorType::CustomCode(m, 504) if *m == REQUEST_DEADLINE)
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

/// Write a JSON error (see `fail_to_proxy`). `Retry-After` when given.
async fn write_json_error(
    session: &mut Session,
    request_id: &str,
    status: u16,
    typ: &str,
    msg: &str,
    retry_after: Option<u64>,
) -> Result<()> {
    let body = error_body(typ, msg);
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

/// Cap a caller-supplied model name before echoing it in an error. The peek is 64 KiB; we do not
/// want that in a JSON error body.
fn clip_catalog_name(name: &str) -> &str {
    &name[..name.floor_char_boundary(128)]
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

/// What a managed Responses `background: true` request is told (D202).
const BACKGROUND_REFUSED: &str = "background responses are not available with a managed key: the \
     provider generates them after the request ends, where they cannot be metered; send the request \
     without \"background\": true";

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

/// Overwrite dialect + `stream_options` eligibility from **this** catalog candidate's path.
/// Injection follows the upstream candidate, not the client: a Messages body must not grow
/// `stream_options`, and a Chat Completions failover of a Messages client must.
fn apply_serving_candidate(rc: &mut RequestCtx) {
    let Some(c) = rc.auto.as_ref().and_then(|a| a.candidate_at(a.candidate)) else {
        return;
    };
    let ep = route::Endpoint::of_upstream_path(c.path);
    rc.dialect = ep.wire();
    rc.inject_eligible = rc.managed && is_streamable_path(c.path);
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

/// What the first path segment resolved to.
///
/// An enum rather than a wider tuple because the failure shapes need *different* rejections
/// (404 unknown provider, 404 unknown model, and — later, once identity is known — 400 for a BYO key
/// on a catalog walk), and a tuple of `Option`s would encode that in which fields happened to be
/// `None`.
enum Routed {
    /// `/{provider}/…`. The provider is named by the request. Not a catalog walk — the escape hatch.
    Provider {
        provider: Arc<Provider>,
        /// Always `Some`: the inbound path with the `/{provider}` segment stripped.
        forward_path: Option<String>,
        streamable: bool,
    },
    /// Bare `/v1` default. BYO keeps the dialect-picked provider; managed becomes a catalog walk.
    BareDefault {
        provider: Arc<Provider>,
        streamable: bool,
    },
    /// `/auto/…`. `header` is the catalog row if `x-beyond-model` resolved; `None` means peek the
    /// body's root `model` after identity is known. An *unknown* header is [`Routed::UnknownModel`],
    /// not this — the header wins, and a catalog miss is the allowlist.
    Auto {
        header: Option<&'static route::ModelRoute>,
    },
    /// A routing header (on `/auto`) naming a model we do not serve. Collapsed with a missing body
    /// model into one 404: a value we cannot match is a value we do not serve.
    UnknownModel,
    /// The first segment matches no provider, is not the bare default, and is not `/auto`.
    UnknownProvider,
    /// Bare `/v1` with BYO keys for two different providers (D82): a 400, never a guess.
    AmbiguousCredential,
}

#[async_trait]
impl ProxyHttp for AiProxy {
    type CTX = Ctx;

    fn new_ctx(&self) -> Self::CTX {
        LIVE_REQUESTS.fetch_add(1, Ordering::AcqRel);
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
            },
        }
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
            let secs = self.state.config.client_write_timeout_secs;
            if secs > 0 {
                session
                    .downstream_session
                    .set_write_timeout(Some(Duration::from_secs(secs)));
            }
            // A client that stops sending its body is bounded the same way: per read, pingora's
            // own timeout on HTTP/1.1 (a no-op on HTTP/2, which `peek_body_model` bounds itself).
            let secs = self.state.config.client_read_timeout_secs;
            session
                .downstream_session
                .set_read_timeout((secs > 0).then(|| Duration::from_secs(secs)));
        }
        let start = Instant::now();
        let deadline = match &full_body {
            Some(fb) => fb.deadline,
            None => crate::deadline::at(start, self.state.config.request_max_secs),
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
            } else if route::is_default_prefix(path) {
                // Bare default: the BYO credential, else the path's dialect, picks the provider;
                // managed traffic becomes a catalog walk after identity. Path is forwarded
                // unchanged for BYO (`None`).
                match byo_credential_dialect(req) {
                    // BYO keys for two providers. A managed key anywhere makes the request managed
                    // (every credential location is stripped), so only BYO is ambiguous.
                    Err(()) if !extract_virtual_key(req).is_some_and(|p| p.managed) => {
                        Routed::AmbiguousCredential
                    }
                    credential => match bare_default_provider_name(path, credential.ok().flatten())
                        .and_then(|name| self.state.provider(name))
                    {
                        Some(p) => Routed::BareDefault {
                            provider: p.clone(),
                            streamable: is_streamable_path(path),
                        },
                        None => Routed::UnknownProvider,
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
                    CatalogHeader::Known(route) => Routed::Auto {
                        header: Some(route),
                    },
                    CatalogHeader::Absent => Routed::Auto { header: None },
                    CatalogHeader::Unknown => Routed::UnknownModel,
                }
            } else {
                Routed::UnknownProvider
            }
        };

        // Catalog resolution needs identity: `/auto` is managed-only, a candidate is only usable if
        // we hold a pool key for it, and managed `/v1` peeks the body only after the key is known
        // (BYO `/v1` stays dialect-default passthrough and must not drain the body). `provider` is
        // `None` only for headerless `/auto` — there is no dialect default, and BYO 400s before
        // `RequestCtx` is built.
        let provider: Option<Arc<Provider>>;
        let forward_path: Option<String>;
        let mut forward_streamable: bool;
        let mut model_route: Option<&'static route::ModelRoute>;
        // Peek the body's root `model` after identity (managed `/v1`, headerless `/auto`).
        let resolve_from_body: bool;
        // BYO is 400 — the `/auto` path, with or without a routing header.
        let managed_only: bool;
        // Checked against the managed endpoint allowlist once the key is known to be managed.
        let provider_route = matches!(routed, Routed::Provider { .. });
        match routed {
            Routed::Provider {
                provider: p,
                forward_path: fp,
                streamable,
            } => {
                provider = Some(p);
                forward_path = fp;
                forward_streamable = streamable;
                model_route = None;
                resolve_from_body = false;
                managed_only = false;
            }
            Routed::BareDefault {
                provider: p,
                streamable,
            } => {
                provider = Some(p);
                forward_path = None;
                forward_streamable = streamable;
                model_route = None;
                resolve_from_body = true;
                managed_only = false;
            }
            Routed::Auto {
                header: Some(route),
            } => {
                // Streamability is a property of the *row*: its candidates all serve one wire at
                // matching endpoints (a catalog invariant), so the first candidate's path answers it
                // for all of them.
                let streamable = route
                    .candidates
                    .first()
                    .is_some_and(|c| is_streamable_path(c.path));
                let first = route
                    .candidates
                    .first()
                    .and_then(|c| self.state.provider_by_id(c.provider).cloned());
                match first {
                    Some(p) => {
                        provider = Some(p);
                        forward_path = None;
                        forward_streamable = streamable;
                        model_route = Some(route);
                        resolve_from_body = false;
                        managed_only = true;
                    }
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
            Routed::Auto { header: None } => {
                provider = None;
                forward_path = None;
                forward_streamable = false;
                model_route = None;
                resolve_from_body = true;
                managed_only = true;
            }
            Routed::UnknownModel => {
                self.state.metrics.rejection(Rejection::UnknownModel).inc();
                let name = session
                    .req_header()
                    .headers
                    .get(route::MODEL_HEADER)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned);
                return Self::reject_catalog_miss(session, &request_id, name.as_deref()).await;
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
            Routed::AmbiguousCredential => {
                // Rare and never a flood path, so not in `REJECT_BODIES`.
                return Self::reject_message_boxed(
                    session,
                    &request_id,
                    400,
                    "invalid_request_error",
                    "the request carries API keys for different providers (x-api-key and \
                     Authorization); send only the key for the provider you mean, or name it \
                     with /{provider}/..."
                        .to_owned(),
                )
                .await;
            }
        }

        // 2. Extract the presented key — a managed virtual key (`bai_v1…`) or a raw BYO provider token.
        let Some(Presented {
            key: raw_key,
            managed: managed_key,
        }) = extract_virtual_key(session.req_header())
        else {
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
        // IPs (global BYO tier — managed traffic is exempt, see `ratelimit`). The `check_at` borrow of
        // `raw_key` ends as the call returns, so the `&mut session` reject is free to run on the
        // over-limit path (where `raw_key` is unused afterward).
        if full_body.is_none()
            && let Some(rl) = &self.state.rate_limit
            && let Some(reason) = rl.check_at(raw_key, managed_key, start)
        {
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
            return Self::reject_boxed(
                session,
                &request_id,
                413,
                "invalid_request_error",
                "request body too large",
            )
            .await;
        }

        // 5. Identity + key handling. One branch: `bai_v1`/`bai_v2` is fail-closed (prefix match →
        // verify; any verify failure is 401, never BYO). Anything else → BYO: the user's own
        // provider token, passed through unchanged (no Beyond identity, so no deny-set, no
        // allowance, and no per-tenant attribution). A public listener without this split would
        // forward a forged virtual key as BYO — junk-auth egress, and the rate guard already
        // exempted it from the BYO aggregate.
        let (tenant_id, vpc_id, key_id, managed) = if managed_key {
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
            // Deny-set: O(1), default-allow. Tenant deny OR (v2) key deny. Tenant wins if both.
            if let Some(reason) = self
                .state
                .deny
                .load()
                .reason_for(identity.tenant_id, identity.key_id)
            {
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
            // Allowance: remaining-ok vs exhausted, same WatchedSet shape as deny. Fail-closed
            // while the set has not been read. 402 before cache and before `upstream_peer`.
            if let Some(reason) = self
                .state
                .allowance
                .load()
                .reason_for(identity.tenant_id, identity.key_id)
            {
                match reason {
                    crate::allowance::AllowanceReject::Exhausted => {
                        self.state.metrics.rejection(Rejection::Quota).inc();
                        return Self::reject_boxed(
                            session,
                            &request_id,
                            402,
                            "insufficient_quota",
                            "quota exhausted",
                        )
                        .await;
                    }
                    // Not seeded yet: transient, so a retryable 503 with `Retry-After`, not the
                    // 402 an SDK treats as final ("quota exhausted" above is the final one).
                    crate::allowance::AllowanceReject::Unavailable => {
                        self.state
                            .metrics
                            .rejection(Rejection::AllowanceUnavailable)
                            .inc();
                        return Self::reject_boxed(
                            session,
                            &request_id,
                            503,
                            "api_error",
                            "allowance unavailable",
                        )
                        .await;
                    }
                }
            }
            // The actual `Bearer …`/`x-api-key` value is precomputed in the provider registry and
            // applied in `upstream_request_filter`; here we only confirm a pool key exists.
            //
            // Skipped for a catalog walk: there a pool key is a property of each *candidate*,
            // and the first one lacking a key is a reason to try the next, not to fail the request.
            // The equivalent gate is the usable-candidate check below. Also skipped for managed
            // `/v1` before the body peek — the dialect-default provider is not the allowlist.
            if model_route.is_none()
                && !resolve_from_body
                && !provider.as_ref().is_some_and(|p| p.has_pool_key())
            {
                return Self::reject_boxed(
                    session,
                    &request_id,
                    503,
                    "api_error",
                    "no provider key available",
                )
                .await;
            }
            (identity.tenant_id, identity.vpc_id, identity.key_id, true)
        } else {
            (0, 0, None, false)
        };
        // A managed `?key=` is the virtual key: never forward it. Catalog walks already send the
        // candidate's own path with no query, so only `/{provider}/…` carries one.
        let forward_path = match forward_path {
            Some(fp) if managed => Some(strip_key_param(&fp).unwrap_or(fp)),
            fp => fp,
        };

        // Catalog walk is **managed-only**. BYO on `/auto` 400s before any peek — a BYO token
        // belongs to one vendor, and reading the body to pick among candidates would still be a
        // guess (and a failover would send that key to a different vendor). BYO on `/v1` does not
        // enter this block: it keeps the dialect-default provider and never consults the catalog.
        if !managed && managed_only {
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
                self.state
                    .metrics
                    .rejection(Rejection::ManagedEndpoint)
                    .inc();
                return Self::reject_message_boxed(
                    session,
                    &request_id,
                    400,
                    "invalid_request_error",
                    "protocol upgrades (WebSocket) are not available with a managed key".to_owned(),
                )
                .await;
            }
            let method_ok = req.method == http::Method::POST;
            let endpoint_ok = !provider_route
                || forward_path
                    .as_deref()
                    .map(|p| p.split_once('?').map_or(p, |(path, _)| path))
                    .is_some_and(is_managed_provider_endpoint);
            if !(method_ok && endpoint_ok) {
                let msg = format!(
                    "{} {} is not available with a managed key; managed keys reach POST generation endpoints only (chat/completions, messages, responses, embeddings, count_tokens)",
                    req.method,
                    clip_catalog_name(req.uri.path()),
                );
                // An endpoint outside the allowlist is 404 whatever the method; a wrong method on an
                // allowed endpoint (or any catalog path) is 405.
                let status = if endpoint_ok { 405 } else { 404 };
                self.state
                    .metrics
                    .rejection(Rejection::ManagedEndpoint)
                    .inc();
                return Self::reject_message_boxed(
                    session,
                    &request_id,
                    status,
                    "invalid_request_error",
                    msg,
                )
                .await;
            }
        }

        let mut body_complete: Option<Vec<u8>> = None;
        let cache_bypass = !managed || cache::request_bypasses(session.req_header());
        if let Some(fb) = &full_body {
            model_route = Some(fb.route);
        }
        // A headerless catalog walk reads the whole body before choosing a row, rather than
        // stopping at `model`: a large body can only fail over once fully read (see `FullBody`), a
        // small one fits pingora's replay buffer either way and costs nothing more, and with the
        // whole body in hand the gateway can hash it for the cache and read Responses session
        // fields.
        let responses = route::is_responses_path(session.req_header().uri.path());
        let large = declared_len.is_none_or(past_replay_buffer);
        let mut peeked = false;
        // Taken before any full read (see `SlotGuard`), handed to the request context below.
        let mut early_slot: Option<SlotGuard<'_>> = None;
        if managed && model_route.is_none() && resolve_from_body {
            // Header wins if present (unknown → 404, no fall-through to the body). Absent → peek.
            match catalog_from_header(session) {
                CatalogHeader::Known(route) => model_route = Some(route),
                CatalogHeader::Unknown => {
                    self.state.metrics.rejection(Rejection::UnknownModel).inc();
                    let name = session
                        .req_header()
                        .headers
                        .get(route::MODEL_HEADER)
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_owned);
                    return Self::reject_catalog_miss(session, &request_id, name.as_deref()).await;
                }
                CatalogHeader::Absent => {
                    if full_body.is_none() {
                        match self.take_slot_before_read(tenant_id) {
                            Ok(guard) => early_slot = guard,
                            Err(()) => return self.reject_tenant_busy(session, &request_id).await,
                        }
                    }
                    // A large body is held twice (here and in its `FullBody` re-run): reserve
                    // both before reading a byte of it.
                    if let Some(n) = declared_len.filter(|&n| past_replay_buffer(n))
                        && !ctx.held.reserve_body(n.saturating_mul(2))
                    {
                        return self.reject_body_memory(session, &request_id).await;
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
                        return self.reject_too_large(session, &request_id).await;
                    }
                    if peek.over_budget {
                        return self.reject_body_memory(session, &request_id).await;
                    }
                    let Some(name) = peek.model.filter(|n| !n.is_empty()) else {
                        self.state.metrics.rejection(Rejection::UnknownModel).inc();
                        return Self::reject_catalog_miss(session, &request_id, None).await;
                    };
                    let Some(route) = route::model_route(&name) else {
                        self.state.metrics.rejection(Rejection::UnknownModel).inc();
                        return Self::reject_catalog_miss(session, &request_id, Some(&name)).await;
                    };
                    if peek.relay {
                        let body = peek.complete.unwrap_or_default();
                        // --- signed ids (`signed_id`) ---
                        if let Some(done) = self
                            .refuse_foreign_ids(session, &request_id, route, &body, tenant_id)
                            .await
                        {
                            return done;
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
                                route,
                                body,
                                slot_held,
                            )
                            .await;
                        drop(early_slot);
                        return relayed;
                    }
                    model_route = Some(route);
                    body_complete = peek.complete;
                }
            }
        }

        // Header-won catalog walks skip the body peek above. The same reasons to read the whole
        // body apply (Responses session state, a body pingora could not replay for failover).
        if managed
            && full_body.is_none()
            && !peeked
            && let Some(route) = model_route
            && (responses
                || large
                || route::walk_reads_body(route)
                || route::walk_reads_tools(route, session.req_header().uri.path()))
        {
            match self.take_slot_before_read(tenant_id) {
                Ok(guard) => early_slot = guard,
                Err(()) => return self.reject_tenant_busy(session, &request_id).await,
            }
            if let Some(n) = declared_len.filter(|&n| past_replay_buffer(n))
                && !ctx.held.reserve_body(n.saturating_mul(2))
            {
                return self.reject_body_memory(session, &request_id).await;
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
                return self.reject_too_large(session, &request_id).await;
            }
            if peek.over_budget {
                return self.reject_body_memory(session, &request_id).await;
            }
            if peek.relay {
                let body = peek.complete.unwrap_or_default();
                // --- signed ids (`signed_id`) ---
                if let Some(done) = self
                    .refuse_foreign_ids(session, &request_id, route, &body, tenant_id)
                    .await
                {
                    return done;
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
                        route,
                        body,
                        slot_held,
                    )
                    .await;
                drop(early_slot);
                return relayed;
            }
            body_complete = peek.complete;
        }

        // Two root `model` keys (D34), refused before connecting wherever the body is already in
        // hand, with a JSON 400 and the request id (D94). A body streamed through is still refused
        // in `request_body_filter`, after the request headers went upstream.
        if model_route.is_some()
            && body_complete
                .as_deref()
                .is_some_and(|b| peek::scan_buffered(b).duplicate_model)
        {
            self.state
                .metrics
                .rejection(Rejection::DuplicateModel)
                .inc();
            return Self::reject_message_boxed(
                session,
                &request_id,
                400,
                "invalid_request_error",
                "the request body has more than one root \"model\" key; send exactly one"
                    .to_owned(),
            )
            .await;
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
        let (provider, usable) = match model_route {
            None => {
                // Provider-routed, or BYO `/v1` dialect default. Headerless `/auto` always set
                // `model_route` above (or 400/404'd).
                match provider {
                    Some(p) => (p, 0u8),
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
                if let Some(kind) = body_complete
                    .as_deref()
                    .and_then(|b| route::refused_input(row, b))
                {
                    return self
                        .reject_refused_input(session, &request_id, row, kind)
                        .await;
                }
                if inbound_responses && body_complete.as_deref().is_some_and(requests_background) {
                    return self.reject_background(session, &request_id).await;
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
                    self.state.metrics.rejection(Rejection::WireMismatch).inc();
                    return Self::reject_message_boxed(
                        session,
                        &request_id,
                        400,
                        "invalid_request_error",
                        translate::session_field_refusal(field, row.model),
                    )
                    .await;
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
                for (orig, c) in arms.iter().enumerate().take(route::MAX_CANDIDATES) {
                    let orig = orig as u8;
                    let provider = self.state.provider_by_id(c.provider);
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
                                return self
                                    .reject_refused_input(session, &request_id, row, "fast mode")
                                    .await;
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
                if self.state.config.session_pins
                    && sub.is_none()
                    && std::ptr::eq(arms, row.candidates)
                    && !parsed_control
                        .as_ref()
                        .is_some_and(control::Control::pins_walk)
                {
                    let affinity = pin::affinity(tenant_id, vpc_id, key_id);
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
                    self.state.metrics.rejection(Rejection::WireMismatch).inc();
                    return Self::reject_message_boxed(
                        session,
                        &request_id,
                        400,
                        "invalid_request_error",
                        format!(
                            "{} has no {} upstream for {}",
                            row.model,
                            sub.provider_name(),
                            session.req_header().uri.path()
                        ),
                    )
                    .await;
                }
                let Some(first) = first_usable(usable, 0) else {
                    // Every remaining candidate is unkeyed. Distinct from `circuit_open`, which
                    // means the candidates exist and are being skipped while they recover —
                    // `doctor`'s `model_catalog` check exists to catch this configuration at boot
                    // instead.
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
                match walk
                    .catalog_index(first)
                    .and_then(|orig| arms.get(usize::from(orig)))
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

        // --- signed ids (`signed_id`): Responses state a provider keeps is tenant-bound ---------
        // A managed Responses relay to a store (a GPT row's Responses arm, `/{provider}/…/responses`)
        // must not carry an id another tenant could resolve. The ids the client sent back are
        // checked here, before connecting, wherever the body is in hand (every catalog walk); a
        // `/{provider}` body streams, and is checked as it is stripped in `request_body_filter`.
        let signed = if managed
            && match model_route {
                Some(row) => signed_id::catalog_applies(session.req_header().uri.path(), row),
                None => {
                    provider_route
                        && forward_path
                            .as_deref()
                            .is_some_and(signed_id::provider_route_applies)
                }
            } {
            let Some(signer) = self.state.id_signer.as_ref() else {
                self.state
                    .metrics
                    .rejection(Rejection::IdSigningUnset)
                    .inc();
                return Self::reject_message_boxed(
                    session,
                    &request_id,
                    503,
                    "api_error",
                    "managed Responses ids cannot be issued: the gateway has no id signing key configured"
                        .to_owned(),
                )
                .await;
            };
            if let Some(Err(refusal)) = body_complete
                .as_deref()
                .map(|b| signer.unsign_request(tenant_id, b, model_route.is_some()))
            {
                self.state.metrics.rejection(Rejection::ForeignId).inc();
                return Self::reject_message_boxed(
                    session,
                    &request_id,
                    400,
                    "invalid_request_error",
                    refusal.to_string(),
                )
                .await;
            }
            Some(Box::new(signed_id::Relay::new(tenant_id)))
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
                    self.state.metrics.rejection(Rejection::WireMismatch).inc();
                    return Self::reject_message_boxed(
                        session,
                        &request_id,
                        400,
                        "invalid_request_error",
                        format!("{} is {}", row.model, row_endpoint.post_hint()),
                    )
                    .await;
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
            None if provider_route => forward_path
                .as_deref()
                .map_or(provider.dialect, wire_of_forward_path),
            None => provider.dialect,
        };
        if model_route.is_some() {
            // From the arm this request will actually walk, not the row's Chat Completions primary:
            // a Responses walk must not inherit `stream_options` injection.
            forward_streamable = walk_arms
                .first()
                .is_some_and(|c| is_streamable_path(c.path));
        }

        // Mark OpenAI managed chat/completions streams for body buffering + `stream_options` injection
        // (handled in `request_body_filter`). Scoped tight: managed only (BYO stays pure
        // passthrough), OpenAI dialect only, streaming-capable paths only — so everything else still
        // streams through untouched. Checked on the forwarded path (suffix), so it's prefix-agnostic.
        let inject_eligible = managed && dialect == Dialect::OpenAi && forward_streamable;
        // Every billable managed `/{provider}/…` body is buffered too, so a model outside the
        // catalog is refused before a byte of it goes upstream: a managed key runs only what its
        // billing row can price (D267). So is a `/responses` body's `background: true` (a
        // generation the provider runs after the request ends, which no usage tap can meter,
        // D202). The free token counts are left to stream: nothing there bills. A catalog walk
        // checked its body above.
        let catalog_check = managed
            && provider_route
            && forward_path
                .as_deref()
                .and_then(route::forward_endpoint)
                .is_some_and(|e| e.sub.is_none_or(route::SubResource::billed));
        // That buffer counts against the body budget. A declared large body reserves up front,
        // before the tenant slot and the breaker permit, so a refusal holds neither; a chunked one
        // reserves as it grows (`request_body_filter`).
        if (inject_eligible || catalog_check)
            && model_route.is_none()
            && let Some(n) = declared_len.filter(|&n| past_replay_buffer(n))
            && !ctx.held.reserve_body(n)
        {
            return self.reject_body_memory(session, &request_id).await;
        }

        // Capture decision from the control surface parsed above. The header wins in **both**
        // directions over the tenant's control-plane rule (Cloudflare's `cf-aig-collect-log`
        // semantics), and the two enablers serve different people: the control plane is the
        // operator's, works retroactively, and needs no cooperation from a client we don't control;
        // the header is the caller's, for per-request precision.
        let control = if let Some(parsed) = parsed_control {
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
                    .rule_for(tenant_id)
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
        let cache_look = if !cache_bypass && model_route.is_some() {
            body_complete.as_deref().and_then(|body| {
                let store = self.state.cache.as_ref()?;
                let req = session.req_header();
                let (ids, n) = walk.provider_ids(walk_arms);
                let ck = cache::key(&cache::KeyParts {
                    tenant_id,
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
        let mut pending_cache: Option<cache::Pending> = None;
        match cache_look {
            Some(Err(hit)) => {
                Self::reply_cache_hit_boxed(session, &request_id, &hit).await?;
                ctx.rc = Some(RequestCtx {
                    tenant_id,
                    vpc_id,
                    key_id,
                    dialect,
                    provider,
                    forward_path,
                    managed,
                    control,
                    model: String::new(),
                    model_scanner: peek::ModelScanner::new(),
                    resp_model_scanner: peek::ModelScanner::for_response(),
                    // Not an upstream stream — `response_filter` never ran, so `active_streams`
                    // was never incremented. The stored `hit.streaming` is emitted on `ai.usage`.
                    streaming: false,
                    inject_eligible: false,
                    catalog_check: false,
                    req_buf: Vec::new(),
                    resp_tail: UsageTail::default(),
                    resp_head: Vec::new(),
                    body_bytes_fed: 0,
                    upstream_status: Some(hit.status),
                    start,
                    deadline,
                    attempt: 0,
                    pool_key: 0,
                    same_provider_retry: false,
                    relay_abandoned: false,
                    refused_resent: false,
                    read_capped: false,
                    breaker_pending: None,
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
                    input_tally: usage::InputTally::default(),
                    tally_eager: false,
                    resp_bytes: 0,
                    upstream_phase: UpstreamPhase::None,
                    redact: None,
                    terminal: TerminalTracker::default(),
                    signed: None,
                    taps: None,
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
        let tenant_slot = match self.state.tenant_slots.as_ref() {
            // A `FullBody` attempt whose parent holds the slot: neither take nor release one.
            _ if parent_holds_slot => false,
            // Taken before the body was read; `logging` releases it from here on.
            _ if early_slot.is_some() => {
                if let Some(guard) = early_slot.take() {
                    guard.hand_over();
                }
                true
            }
            Some(slots) if managed => {
                if !slots.try_acquire(tenant_id) {
                    self.state
                        .metrics
                        .rejection(Rejection::TenantConcurrency)
                        .inc();
                    return Self::reject_boxed(
                        session,
                        &request_id,
                        429,
                        "rate_limit_error",
                        "too many concurrent requests",
                    )
                    .await;
                }
                true
            }
            _ => false,
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
            if tenant_slot && let Some(slots) = self.state.tenant_slots.as_ref() {
                slots.release(tenant_id);
            }
            self.state.metrics.rejection(Rejection::CircuitOpen).inc();
            let retry_after = Some(breaker.retry_after_secs());
            return Box::pin(Self::reject_retry_after(
                session,
                &request_id,
                503,
                "api_error",
                "provider temporarily unavailable",
                retry_after,
            ))
            .await;
        }
        // A permit is now outstanding against this provider (see `RequestCtx::breaker_pending`).
        // The model-routed path starts owing nothing and takes on its first permit in
        // `upstream_peer`.
        let breaker_pending = breaker_permit.ok().flatten();
        // Past any key cooling off from a 401 (D71); a catalog walk picks per candidate.
        let pool_key = provider.first_key();
        if tenant_slot {
            ctx.held.tenant = Some(tenant_id);
        }

        ctx.rc = Some(RequestCtx {
            tenant_id,
            vpc_id,
            key_id,
            dialect,
            provider,
            forward_path,
            managed,
            control,
            model: String::new(),
            model_scanner: peek::ModelScanner::new(),
            // `for_response`, not `new`: a response may carry the model nested under `message`
            // (Anthropic's `message_start`), and a root-only scanner would neither find it nor ever
            // stop looking. See `ModelScanner::for_response`.
            resp_model_scanner: peek::ModelScanner::for_response(),
            streaming: false,
            inject_eligible,
            catalog_check,
            // Pre-sized below, once the context says whether this request rewrites its body.
            req_buf: Vec::new(),
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
            deadline,
            attempt: 0,
            pool_key,
            same_provider_retry: false,
            relay_abandoned: false,
            refused_resent: false,
            read_capped: false,
            breaker_pending,
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
            input_tally: usage::InputTally::default(),
            tally_eager: tally_eager(managed, model_route.is_some(), declared_len),
            resp_bytes: 0,
            upstream_phase: UpstreamPhase::None,
            redact: None,
            terminal: TerminalTracker::default(),
            signed,
            taps: None,
        });
        // A body buffered for a rewrite (`RequestCtx::rewrites_body`) is pre-sized from the
        // declared Content-Length, so accumulation is a single allocation instead of a geometric
        // realloc chain; capped at `MAX_REQUEST_BODY` so a lying header can't pre-allocate
        // unbounded memory. Every other request leaves it empty and never buffers.
        //
        // The `+ STREAM_OPTIONS_FRAG.len()` is headroom for the splice, which
        // `apply_stream_usage_injection` performs *in place*: with it the injection never
        // reallocates, so a body arrives, is spliced, and goes upstream on one allocation.
        if let Some(rc) = ctx.rc.as_mut()
            && rc.rewrites_body()
            && let Some(len) = declared_len
        {
            rc.req_buf = Vec::with_capacity(len.min(MAX_REQUEST_BODY) + STREAM_OPTIONS_FRAG.len());
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

        // Pingora calls this once per attempt, always before a body byte moves, so it is the one
        // place a retry's leftover request-body state can be cleared. No-op on the first attempt.
        rc.reset_request_body_phase();

        // Same-provider key walk after a managed 429, a stale pooled connection, or the next
        // address of a candidate whose first one refused the connection. Stay on this provider,
        // keep the outstanding breaker permit (429 is a success, not a failure), and do not enter
        // the candidate walk — that walk would send this provider's remaining keys to a different
        // vendor.
        if rc.same_provider_retry {
            rc.same_provider_retry = false;
            if let Some(a) = rc.auto.as_mut() {
                a.attempt_start = Instant::now();
            }
            let addr = match self.state.resolve(&rc.provider.authority, rc.attempt).await {
                Ok((a, _)) => a,
                Err(e) => {
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
            rc.upstream_phase = UpstreamPhase::Attempted;
            let peer = self.build_peer(addr, &rc.provider, rc.auto.as_ref().map(|a| a.route));
            return self.cap_read_at_deadline(rc, peer);
        }

        // Model-routed: this hook owns the candidate walk *and* the breaker ledger.
        //
        // It is the only place `rc.provider` changes, which is what makes double-recording
        // structurally impossible — and it is the only hook that runs on every attempt, including
        // the ones pingora starts without consulting `fail_to_connect` (its default
        // `error_while_proxy` marks a reused-connection failure retryable on its own). A design that
        // recorded in `fail_to_connect` would miss exactly those.
        if rc.auto.is_some() {
            // Reaching here with a permit outstanding means the previous attempt failed before any
            // response arrived, so the candidate we were on earned the failure. Resolve it before
            // touching anything else; `logging` then only ever sees the final candidate's permit.
            if let Some(permit) = rc.breaker_pending.take()
                && let Some(b) = &rc.provider.breaker
            {
                b.record_failure_for(permit);
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
                let permit = p.breaker.as_ref().map(|b| (b, b.allow()));
                if let Some((b, Err(_))) = permit {
                    self.state.metrics.rejection(Rejection::CircuitOpen).inc();
                    if let Some(a) = rc.auto.as_mut() {
                        let secs = u16::try_from(b.retry_after_secs()).unwrap_or(u16::MAX);
                        a.open_retry_after = Some(a.open_retry_after.map_or(secs, |s| s.min(secs)));
                    }
                    rc.advance_candidate(i);
                    continue;
                }
                // A permit (if this breaker has one to give) is now outstanding against `p`.
                rc.breaker_pending = permit.and_then(|(_, r)| r.ok());
                // New vendor ⇒ that vendor's first key not cooling off (D71). Never carry provider
                // A's index (or secret) onto provider B. A re-run of a large body resumes the key
                // walk an earlier attempt started on this candidate (see `FullBody`); one with no
                // walk starts past the cooling keys like any other request (D83).
                rc.pool_key = full_body_ctx(session)
                    .zip(rc.auto.as_ref().and_then(|a| a.walk.catalog_index(i)))
                    .and_then(|(fb, orig)| fb.keys.get(usize::from(orig)).copied())
                    .filter(|&k| k != NO_KEY_WALK)
                    .unwrap_or_else(|| p.first_key());
                rc.provider = p.clone();
                apply_serving_candidate(rc);
                // A new candidate starts at its first address, with its own refused-stream resend.
                rc.attempt = 0;
                rc.refused_resent = false;
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
                        rc.set_forward_path(candidate.path);
                        if let Some(sub) = rc.auto.as_ref().and_then(|a| a.sub)
                            && let Some(path) = rc.forward_path.as_mut()
                        {
                            path.push_str(sub.suffix());
                        }
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
                        if let Some(b) = &p.breaker
                            && let Some(permit) = rc.breaker_pending.take()
                        {
                            b.record_failure_for(permit);
                        }
                        rc.advance_candidate(i);
                        continue;
                    }
                }
            }
        }

        // Resolve via the TTL cache (async, non-blocking) rather than `HttpPeer::new`'s eager
        // blocking `getaddrinfo`. SNI/Host = the configured host; TLS on for real providers (the
        // e2e harness flips `upstream_tls=false` for a plaintext mock).
        // Each connect retry (`fail_to_connect`) takes the next resolved address, so a name whose
        // first address is dead (`localhost` → `::1` first) still connects.
        let addr = match self.state.resolve(&rc.provider.authority, rc.attempt).await {
            Ok((a, _)) => a,
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
        rc.upstream_phase = UpstreamPhase::Attempted;
        let peer = self.build_peer(addr, &rc.provider, rc.auto.as_ref().map(|a| a.route));
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
    /// relayed, not attempted.
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
        if rc.managed && (200..300).contains(&status) {
            rc.provider.key_served(rc.pool_key);
        }

        // Managed 401: this pool key is revoked (D71). Never the caller's fault. Cool it off so
        // later requests start on a good key, then walk like a 429. Not a 403 (D84, see
        // `is_pool_key_failure`).
        let key_auth = rc.managed && is_pool_key_failure(status);
        if key_auth {
            self.state.metrics.key_cooled(KeyCooled::Revoked);
            rc.provider.mark_key_bad(rc.pool_key);
        }

        // Managed 429 or 401: walk the next unused key on *this* provider. Not a vendor failover
        // (those rules stay — `/auto` 5xx owns that, and a 401 on the last key falls through to it
        // below) and not a breaker failure (the provider answered). A 401 is the provider refusing
        // the credential before it processed anything, so resending the body under the next key
        // is as safe as after a 429.
        if rc.managed && (status == 429 || key_auth) {
            let next = usize::from(rc.pool_key).saturating_add(1);
            if next < rc.provider.pool_auth.len()
                && let Ok(next) = u8::try_from(next)
            {
                if body_replayable(session) {
                    self.state.metrics.key_walks_total.inc();
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.provider.name.as_str(),
                        key = rc.pool_key,
                        status,
                        "upstream returned {status}; trying the next pool key",
                    );
                    rc.pool_key = next;
                    rc.same_provider_retry = true;
                    let mut e =
                        pingora_core::Error::new(pingora_core::ErrorType::HTTPStatus(status));
                    e.set_retry(true);
                    return Err(e);
                }
                if let Some(fb) = full_body_ctx(session)
                    && let Some(orig) = rc
                        .auto
                        .as_ref()
                        .and_then(|a| a.walk.catalog_index(a.candidate))
                {
                    // The parent holds the body: it drops this response and re-runs on the next key.
                    self.state.metrics.key_walks_total.inc();
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.provider.name.as_str(),
                        key = rc.pool_key,
                        status,
                        "upstream returned {status}; re-running the full body on the next pool key",
                    );
                    rc.relay_abandoned = fb.record(RelayRetry::Key {
                        candidate: orig,
                        key: next,
                    });
                    return Ok(());
                }
                warn!(
                    request_id = %rc.request_id,
                    provider = rc.provider.name.as_str(),
                    status,
                    body_done = session.as_mut().is_body_done(),
                    "upstream returned {status} but the request body is not provably replayable; not walking keys",
                );
            }
            // Last key, or unreplayable: relay this 429, Retry-After included. A 401 goes on to
            // the catalog walk's key-failure rule (next candidate), or is relayed.
            if status == 429 {
                return Ok(());
            }
        }

        let Some((usable, at)) = rc.auto.as_ref().map(|a| (a.usable, a.candidate)) else {
            return Ok(());
        };
        // A managed walk's 401/402/403 is this candidate refusing (a revoked or unfunded key, a
        // 403 its vendor may not share), not a provider failure: the next candidate holds a
        // different key, so it is a candidate failure for this request like a 5xx. Unlike a 5xx
        // it says nothing about the provider's health, so it never opens the breaker (resolved
        // as a success below).
        let key_failure = rc.managed && is_candidate_refusal(status);
        if status < 500 && !key_failure {
            return Ok(());
        }
        if first_usable(usable, at.saturating_add(1)).is_none() {
            return Ok(());
        }
        // Failing over abandons this response at its head, so `Redact` never reads the body that
        // tells `logging` an account is out of credit (D180), and every later request would pay a
        // round trip to it first. A 402 says so by its status alone except on OpenRouter, whose
        // 402 may be one request too large for the balance (`remedy::unfunded_402`): cool that key
        // here, as `logging` would have, on either abandoning path below (D258). OpenRouter's
        // unread 402 is a strike against its key instead, and enough in a row cool it (D261).
        let unfunded_402 = (key_failure && status == 402)
            .then(|| rc.auto.as_ref().and_then(|a| a.candidate_at(at)))
            .flatten()
            .map(|c| remedy::unfunded_402(c.provider));
        // Fail over only when the body is **provably** replayable: fully read, and small enough
        // that pingora buffered all of it.
        //
        // `retry_buffer_truncated()` alone is not enough, and the reason is subtler than it looks.
        // It reports on what has been buffered *so far*, so a provider that 5xxes fast — which is
        // what a failing provider does — answers before a large body has finished streaming in, and
        // the check reads `false` simply because the bytes that would truncate it have not arrived.
        //
        // Retrying there is not *unsafe*: pingora replays the buffered prefix with
        // `end_of_body = is_body_done()` and the duplex loop reads the remainder straight from the
        // socket (`proxy_h1.rs`'s retry block), so the next candidate does receive the whole body.
        // What it is, is **timing-dependent** — the same request fails over or does not depending on
        // how quickly the upstream rejected it. For a path that decides which vendor gets billed,
        // a deterministic rule is worth more than the extra failovers the loose one would win.
        //
        // The cost is real and worth naming: a 5xx that arrives while the client is still uploading
        // is relayed rather than retried, even when it would have replayed fine. That is what
        // `ai_failover_unreplayable_total` counts.
        if !body_replayable(session)
            && let Some(fb) = full_body_ctx(session)
            && let Some(orig) = rc.auto.as_ref().and_then(|a| a.walk.catalog_index(at))
        {
            // The parent holds the body: it drops this response and re-runs on the next candidate.
            // `response_filter` still runs for it, so the breaker sees the failure.
            self.state.metrics.candidate_failovers_total.inc();
            warn!(
                request_id = %rc.request_id,
                provider = rc.provider.name.as_str(),
                candidate = at,
                status,
                "upstream returned {status}; re-running the full body on the next candidate",
            );
            rc.relay_abandoned = fb.record(RelayRetry::Candidate(orig));
            // Not recorded (the final attempt): this answer is relayed and `logging` reads it.
            if rc.relay_abandoned {
                self.abandon_402(rc, unfunded_402, status);
            }
            return Ok(());
        }
        if !body_replayable(session) {
            self.state.metrics.failover_unreplayable_total.inc();
            warn!(
                request_id = %rc.request_id,
                provider = rc.provider.name.as_str(),
                status,
                body_done = session.as_mut().is_body_done(),
                "upstream failed but the request body is not provably replayable; relaying the error",
            );
            return Ok(());
        }

        self.state.metrics.candidate_failovers_total.inc();
        warn!(
            request_id = %rc.request_id,
            provider = rc.provider.name.as_str(),
            candidate = at,
            status,
            "upstream returned {status}; trying the next candidate",
        );
        // The outgoing candidate's breaker failure is recorded by `upstream_peer`'s prologue, which
        // still sees `breaker_pending` set. A 5xx is a failure by the breaker's own definition, so
        // that is the right outcome — and recording it here as well would double-count. A key
        // failure is not one: resolve its permit as the success it is (the provider answered).
        if key_failure
            && let Some(permit) = rc.breaker_pending.take()
            && let Some(b) = rc.provider.breaker.as_ref()
        {
            b.record_success_for(permit);
        }
        self.abandon_402(rc, unfunded_402, status);
        rc.advance_candidate(at);
        let mut e = pingora_core::Error::new(pingora_core::ErrorType::HTTPStatus(status));
        e.set_retry(true);
        Err(e)
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
        if rc.managed {
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
            retain_managed_client_headers(upstream_request, rc.provider.name == "anthropic")?;
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
            if let Some(auth) = rc.provider.pool_auth.get(usize::from(rc.pool_key)) {
                // Clone the boot-built `HeaderValue` (a refcount bump) rather than re-validating and
                // re-copying the key out of a `&str` on every managed request. The `&str` path is
                // kept as a fallback for a key that isn't a legal header value, which could never
                // have worked anyway — see `PoolAuth`.
                match &auth.header {
                    Some(hv) => {
                        upstream_request.insert_header(rc.provider.auth.header(), hv.clone())?
                    }
                    None => upstream_request
                        .insert_header(rc.provider.auth.header(), auth.value.expose())?,
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
        match &rc.provider.host_header {
            Some(hv) => upstream_request.insert_header("host", hv.clone())?,
            None => upstream_request.insert_header("host", rc.provider.host.as_str())?,
        }

        // Dashboard-attribution headers (OpenRouter, managed traffic only — Task #22, see
        // `apply_provider_attribution`).
        apply_provider_attribution(upstream_request, rc.provider.name.as_str(), rc.managed)?;

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
        if let Some(forward_path) = &rc.forward_path
            && let Ok(uri) = forward_path.parse()
        {
            upstream_request.set_uri(uri);
        } else if rc.managed
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
        // inside the `if rc.managed` block in `logging` (the billing-log fallback and
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
            // and 503 every tenant. The status is what pingora answers with.
            rc.body_bytes_fed = rc.body_bytes_fed.saturating_add(chunk.len());
            if rc.body_bytes_fed > MAX_REQUEST_BODY {
                self.state.metrics.rejection(Rejection::BodyTooLarge).inc();
                return Err(pingora_core::Error::explain(
                    pingora_core::ErrorType::HTTPStatus(413),
                    "request body exceeds limit",
                )
                .into_down());
            }
            // Eligible requests are buffered so we can splice the root object before any byte reaches
            // the upstream (injection inserts near the front, so we can't have forwarded it already).
            // When we're buffering anyway, the incremental scan is skipped entirely and both answers
            // come from one walk of the finished buffer below — the body was previously traversed
            // twice, once here for `model` and once by the injection planner, over the same bytes
            // with the same depth/string/escape bookkeeping.
            // Capture tap. Deliberately here, on the chunk as it *arrives*, which makes the captured
            // bytes **pre-rewrite** for free: both of the rewrites below happen at end-of-stream over
            // the whole of `req_buf`, long after this ran. That matters twice over — a captured body
            // showing a `stream_options` the client's SDK never sent, or a `model` re-spelled for
            // whichever failover candidate served, sends an engineer hunting through their own code
            // for something the gateway put there. What we capture is what the client sent.
            //
            // Bounded and non-withholding, exactly like the response tap: the chunk is forwarded
            // either way, so this costs a memcpy, never latency.
            if let Some(c) = rc.control.as_mut().and_then(|c| c.capture.as_mut()) {
                c.push_req(chunk);
            }
            // Only where no copy of the body will be left for `logging` to read (`tally_eager`):
            // everywhere else the estimate is made there, and only when a row needs one.
            if rc.tally_eager {
                rc.input_tally.feed(chunk);
                // The price knobs, for the same reason: no copy of this body is left for
                // `logging` to read (see `requested_knobs`).
                if rc.managed {
                    let (scan, kept) = rc
                        .taps
                        .get_or_insert_with(Box::default)
                        .knobs
                        .get_or_insert_with(|| {
                            (peek::ModelScanner::new(), peek::Kept::request_knobs())
                        });
                    scan.feed_keeping(chunk, kept);
                }
            }

            if rc.rewrites_body() {
                rc.req_buf.extend_from_slice(chunk);
                // A large body buffered for a rewrite (the `/{provider}` usage splice; a catalog
                // walk's large body is a `FullBody` re-run, already reserved) holds budget as it
                // grows. Tagged downstream: the client's size, not the provider's health.
                if past_replay_buffer(rc.req_buf.len()) && !held.reserve_body(rc.req_buf.len()) {
                    self.state.metrics.rejection(Rejection::BodyMemory).inc();
                    return Err(
                        gateway_error(503, "too many large request bodies in flight").into_down(),
                    );
                }
            } else if rc.managed {
                rc.model_scanner.feed(chunk);
            }
        }

        if rc.rewrites_body() {
            if end_of_stream {
                // One structural walk for every answer (see `peek::scan_buffered`).
                let mut buf = std::mem::take(&mut rc.req_buf);
                // --- signed ids (`signed_id`): the provider gets its own ids back ---
                // The client's body, per attempt. A catalog walk was checked before connecting; a
                // `/{provider}` body is refused here, before a byte of it goes upstream.
                if rc.signed.is_some()
                    && let Some(signer) = self.state.id_signer.as_ref()
                {
                    match signer.unsign_request(rc.tenant_id, &buf, rc.auto.is_some()) {
                        Ok(Some(raw)) => buf = raw,
                        Ok(None) => {}
                        Err(_) => {
                            self.state.metrics.rejection(Rejection::ForeignId).inc();
                            return Err(gateway_error(400, signed_id::REFUSED).into_down());
                        }
                    }
                }
                // --- end signed ids ---
                let mut scan = peek::scan_buffered(&buf);
                // Two root `model` keys on a catalog walk: the row came from one and the rewrite
                // below edits one, but the provider's parser picks its own (usually the last). Refused
                // rather than guessed, before a body byte goes upstream. Checked on the client's
                // body, ahead of translation, which re-serializes and would hide it.
                // A managed `/{provider}` body is relayed as sent, so the provider serves whichever
                // copy its parser keeps while the billing row names the first: refused the same way.
                if (rc.auto.is_some() || rc.catalog_check) && scan.duplicate_model {
                    self.state
                        .metrics
                        .rejection(Rejection::DuplicateModel)
                        .inc();
                    return Err(pingora_core::Error::explain(
                        pingora_core::ErrorType::HTTPStatus(400),
                        "duplicate root model key",
                    )
                    .into_down());
                }
                // A feature billed outside the usage a row can meter (a container, the image
                // generation tool, an OpenRouter plugin, Anthropic Priority Tier): refused before a
                // byte of the body goes upstream, so nothing runs unpriced (D268). Read on the
                // client's body, ahead of any rewrite and of the catalog check (an `:online` model is named
                // for what it is). Every body buffered here is managed.
                if rc.managed {
                    let model = rc
                        .auto
                        .as_ref()
                        .map(|a| a.route.model)
                        .or(scan.model.as_deref())
                        .unwrap_or(rc.model.as_str());
                    let messages = session.req_header().uri.path().ends_with("/messages");
                    let found = unpriced::inspect(&buf, model, messages);
                    if let Some(why) = found.refused {
                        self.state
                            .metrics
                            .rejection(Rejection::UnpricedFeature)
                            .inc();
                        return Err(gateway_error(400, why).into_down());
                    }
                    if found.web_search_preview {
                        rc.taps.get_or_insert_with(Box::default).web_search_preview = true;
                    }
                }
                // Managed `/{provider}/…` billable call naming no catalog row (D267): its billing
                // row would carry no `price_model`. The client's id is kept for the error and the
                // row, then the request ends before a byte of the body goes upstream.
                if rc.catalog_check && scan.model.as_deref().and_then(price_row).is_none() {
                    if rc.model.is_empty()
                        && let Some(m) = scan.model.take()
                    {
                        rc.model = sanitize_model(m).into_owned();
                    }
                    self.state.metrics.rejection(Rejection::UnknownModel).inc();
                    return Err(gateway_error(404, CATALOG_MISS).into_down());
                }
                // Managed `/{provider}/…/responses` asking for `background: true` (D202).
                if rc.catalog_check
                    && rc.forward_path.as_deref().is_some_and(|p| {
                        route::forward_is_responses(p.split_once('?').map_or(p, |(p, _)| p))
                    })
                    && requests_background(&buf)
                {
                    self.state
                        .metrics
                        .rejection(Rejection::ManagedEndpoint)
                        .inc();
                    return Err(gateway_error(400, BACKGROUND_REFUSED).into_down());
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
                    let stream_only = candidate.is_some_and(providers::catalog::stream_only);
                    let fast = candidate.is_some_and(|c| translated_fast(a, c));
                    let reads_developer =
                        candidate.is_none_or(providers::catalog::reads_developer_role);
                    let tool_thinking = candidate
                        .map_or(providers::catalog::ToolThinking::Free, |c| {
                            providers::catalog::tool_thinking(c)
                        });
                    if let Some(t) = a.translate.as_mut()
                        && let Some(to) = serving
                    {
                        t.tools = translate::ToolNames::default();
                        t.gateway_cache = false;
                        if t.client != to {
                            // Translation builds `Value`s of the body: its heap is held in the
                            // body budget for the step, or the body is refused (D216).
                            let _heap = self.hold_translation_heap(&buf)?;
                            // The tool names this attempt's response maps calls back through, and
                            // whether its cache breakpoints are the gateway's (per attempt: a
                            // failover candidate on another wire re-decides both).
                            (buf, t.tools, t.gateway_cache) = translate::request_with_tools(
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
                        } else if to == route::Endpoint::ChatCompletions
                            && upstream_model.contains("claude")
                        {
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
                            changed |= translate::thinking_off_for_tools(&mut buf, tool_thinking);
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
                if rc.inject_eligible
                    && scan.stream_options_ambiguous
                    && peek::remove_root_members(&mut buf, "stream_options")
                {
                    scan = peek::scan_buffered(&buf);
                }
                let (buf, model_span) = match scan.stream_options_at {
                    Some(at) if rc.inject_eligible => force_include_usage(buf, at, scan.model_span),
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
                // `inject_eligible` is the OpenAI-upstream gate: a translated Anthropic body also
                // carries `"stream":true`, and splicing `stream_options` into it would be a field
                // that API does not recognize.
                let buf = if rc.inject_eligible {
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
            && rc.managed
            && !rc.inject_eligible
            && rc.model.is_empty()
            && let Some(m) = rc.model_scanner.take_model()
        {
            rc.model = sanitize_model(m).into_owned();
        }
        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
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
            rc.provider
                .metrics
                .ttft_seconds
                .observe(rc.attempt_start().elapsed().as_secs_f64());
            // A 2xx is not yet known to be healthy: a provider can answer 200 with an error body
            // (OpenRouter's error-in-200, an SSE stream whose first event is an error), so its
            // breaker verdict waits for the body's first bytes (`response_body_filter`).
            if (200..300).contains(&status)
                && let Some(a) = rc.auto.as_mut()
            {
                a.health = Some(PendingHealth { prefix: Vec::new() });
            }
            rc.provider.metrics.record_response(status);
            rc.upstream_status = Some(status);
            // Resolve the breaker permit here, at the head, not at the end of the body: the head
            // is the provider's answer (a 5xx is broken, anything else is reachable), and a
            // half-open probe resolved only at end of stream let one long or stalled stream 503
            // the provider for everyone until it ended. `logging` resolves only the attempts that
            // never got a head.
            if let Some(permit) = rc.breaker_pending.take()
                && let Some(b) = rc.provider.breaker.as_ref()
            {
                if status >= 500 {
                    b.record_failure_for(permit);
                } else {
                    b.record_success_for(permit);
                }
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
            if rc.managed {
                tap_response_head(rc, upstream_response);
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
            if rc.managed
                && let Some(finder) = rc
                    .provider
                    .pool_auth
                    .get(usize::from(rc.pool_key))
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
            if rc.managed && status >= 400 {
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
                    .provider
                    .pool_auth
                    .get(usize::from(rc.pool_key))
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
            if let Some(hv) = rc.provider.name_header.as_ref() {
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
                    || catalog_error_relay(a, rc.upstream_status, rc.streaming)
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
        let Some(rc) = ctx.as_mut() else {
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
                .provider
                .pool_auth
                .get(usize::from(rc.pool_key))
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
            // read at exactly one place, inside the `if rc.managed` block in `logging`. A BYO
            // request has no Beyond identity and emits no billing row, so scanning its response was
            // pure waste — and *unbounded* waste on any response with no root-level `model`, since
            // the scanner never reaches its `done` short-circuit and walks every byte.
            if rc.managed {
                match rc.taps.as_mut() {
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
            if rc.managed {
                rc.resp_bytes = rc
                    .resp_bytes
                    .saturating_add(u32::try_from(chunk.len()).unwrap_or(u32::MAX));
            }
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
                || catalog_error_relay(a, rc.upstream_status, rc.streaming)
        });
        if translating {
            let streaming = rc.streaming;
            let dialect = rc.dialect;
            let status = rc.upstream_status.unwrap_or(200);
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
            && rc.read_capped
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
        // One rule for every body size (D09, D51): a connection failure is retried only when the
        // upstream cannot have the whole request (`body_delivered` is false: the connection never
        // came up, the body was not read to its end, or writing it failed), and never once the
        // response has started or the client is gone. A provider that received the whole body may
        // be generating, and billing, already; resending it — to the same candidate or the next,
        // pingora's reused-connection retry included — risks running the request twice, so that
        // failure ends the request (`fail_to_proxy` answers it).
        // Except when the provider demonstrably never processed it: an HTTP/2 refusal, or a pooled
        // HTTP/1.1 connection it closed with the request unread. Having the body is not having
        // taken the request.
        let refused = upstream_refused_stream(&e);
        let delivered = !refused
            && !reset_before_reading(&e, client_reused)
            && ctx
                .rc
                .as_ref()
                .is_some_and(|rc| body_delivered(session, rc, Some(&*e)));
        if *e.esource() == pingora_core::ErrorSource::Downstream
            || session.as_downstream().response_written().is_some()
            || delivered
        {
            e.set_retry(false);
            return e;
        }
        let Some(rc) = ctx.rc.as_mut() else {
            e.set_retry(false);
            return e;
        };
        // A refused stream gets one resend on its candidate (D72). Refused again there, the
        // provider is failing to take work (its stream limit, a drain that never ends): a provider
        // failure like any other, so the walk fails over and the breaker hears of it (D91), rather
        // than a tight loop of resends up to pingora's retry limit.
        //
        // Except a GOAWAY on a connection this request found already open (D248): that is the
        // provider draining a connection that carried other streams, one GOAWAY refusing every
        // multiplexed stream above its last id at once (D160), and it retires the connection, so
        // the resend lands on another. Not the provider refusing work, so it does not spend the
        // one resend. A GOAWAY on a connection the request opened itself refused its first stream:
        // that one counts. Each drain resend needs a live connection that served before, and
        // pingora's retry limit bounds them all.
        let drained = refused && client_reused && upstream_goaway(&e);
        let resend_refused = refused && (drained || !rc.refused_resent);
        if resend_refused {
            warn!(
                request_id = %rc.request_id,
                provider = rc.provider.name.as_str(),
                error = %e,
                "upstream refused the stream unprocessed (GOAWAY / REFUSED_STREAM); resending once",
            );
        } else if refused {
            warn!(
                request_id = %rc.request_id,
                provider = rc.provider.name.as_str(),
                error = %e,
                "upstream refused the stream again; treating it as a provider failure",
            );
        }
        let walk = rc
            .auto
            .as_ref()
            .and_then(|a| Some((a.walk.catalog_index(a.candidate)?, a.candidate, a.usable)));
        // A large body: pingora cannot resend it, but the parent holds it, so hand the retry back:
        // a reused connection (a pooled one the provider had closed) gets one more try on the same
        // candidate, otherwise the walk moves to the next candidate, if there is one. With nowhere
        // to go the error stands.
        if let Some(fb) = full_body_ctx(session) {
            e.set_retry(false);
            if let Some((orig, at, usable)) = walk {
                let retry = if (client_reused || refused) && fb.reset & (1 << orig) == 0 {
                    Some(RelayRetry::Reset(orig))
                } else if first_usable(usable, at.saturating_add(1)).is_some() {
                    Some(RelayRetry::Candidate(orig))
                } else {
                    None
                };
                // A same-candidate retry is the connection's fault (or a stream refused before
                // any processing), not the provider's: give the permit back without an outcome,
                // as a small body keeps it for its own same-candidate retry.
                if retry == Some(RelayRetry::Reset(orig))
                    && let Some(permit) = rc.breaker_pending.take()
                    && let Some(b) = rc.provider.breaker.as_ref()
                {
                    b.release(permit);
                }
                if let Some(retry) = retry {
                    rc.relay_abandoned = fb.record(retry);
                }
            }
            return e;
        }
        // A small body pingora can replay from its buffer.
        if session.as_ref().retry_buffer_truncated() {
            e.set_retry(false);
            return e;
        }
        match walk {
            // A reused connection, or a stream the provider refused for the first time: the same
            // candidate again on a fresh one, keeping its breaker permit (it was the connection,
            // not the provider).
            Some(_) if resend_refused || (client_reused && !refused) => {
                rc.refused_resent |= refused && !drained;
                rc.same_provider_retry = true;
                e.set_retry(true);
            }
            Some((_, at, usable)) if first_usable(usable, at.saturating_add(1)).is_some() => {
                self.state.metrics.candidate_failovers_total.inc();
                rc.advance_candidate(at);
                e.set_retry(true);
            }
            Some(_) => e.set_retry(false),
            // Provider-routed: there is no next candidate. A refused stream is resent once; refused
            // again, the error stands (`logging` records it against the breaker). Otherwise
            // pingora's rule: a reused connection is retried once.
            None if resend_refused => {
                rc.refused_resent |= !drained;
                e.set_retry(true);
            }
            None if refused => e.set_retry(false),
            // A body write that met a stream already closed (D248) is a reused connection's
            // failure before delivery, like pingora's own `ReusedOnly` errors.
            None if h2_body_unsent(&e) => {
                e.retry = pingora_core::RetryType::ReusedOnly;
                e.retry.decide_reuse(client_reused);
            }
            None => e.retry.decide_reuse(client_reused),
        }
        e
    }

    /// Answer every request the gateway itself ends, after admission or during the proxy loop,
    /// with the same JSON error envelope the `reject` paths use: `content-type: application/json`,
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
        // A managed `/{provider}` model outside the catalog (D267) names the model, as the catalog
        // walk's 404 does. The one error message built per request, on a refusal path.
        let miss = (msg == CATALOG_MISS)
            .then(|| catalog_miss_message(ctx.rc.as_ref().map(|rc| rc.model.as_str()), false));
        let msg = miss.as_deref().unwrap_or(msg);
        // Pingora closes the client connection after a proxy error; say so (`connection: close`),
        // as its own error responses do, so a pooled client does not send its next request into a
        // socket about to close.
        session.as_downstream_mut().set_keepalive(None);
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
        if let Err(write_err) = Box::pin(write_json_error(
            session,
            &request_id,
            status,
            typ,
            msg,
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
            if let Some((usable, at, addrs)) =
                rc.auto.as_ref().map(|a| (a.usable, a.candidate, a.addrs))
            {
                // This candidate resolved to more than one address and an untried one remains:
                // try it before giving up on the candidate. Same provider, same breaker permit
                // (one address refusing is not the provider failing).
                if rc.attempt.saturating_add(1) < addrs {
                    rc.attempt += 1;
                    rc.same_provider_retry = true;
                    rc.provider.metrics.connect_retries_total.inc();
                    warn!(
                        request_id = %rc.request_id,
                        provider = rc.provider.name.as_str(),
                        candidate = at,
                        address = rc.attempt,
                        error = %e,
                        "upstream connect failed; trying the candidate's next address",
                    );
                    e.set_retry(true);
                    return e;
                }
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
        // A client that closed after its stream's terminal event reached it has the whole answer:
        // the request completed, whatever the provider's end of stream was still doing.
        let e = e.filter(|e| !closed_after_terminal(rc, e));

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
        // `breaker_pending` is the ledger (see `RequestCtx::breaker_pending`): it holds the permit
        // from when it is claimed until it is taken to resolve it, so exactly one `record_*_for`
        // lands per `allow()`. On the model-routed path a candidate switch resolves the outgoing
        // candidate in `upstream_peer` and takes the permit there, which is what stops this from double-recording
        // against whichever candidate happened to be current at the end.
        //
        // Failure = the provider is *broken*: a 5xx response, or no response at all paired with an
        // upstream error (connect/read failure). Success = the provider *answered* — 2xx/3xx, and
        // deliberately **4xx/429 too**: a 429 is a healthy provider throttling our pool key, which the
        // rate limiter and the client's `Retry-After` own, NOT a reason to cut all traffic to it.
        if let Some(breaker) = rc.provider.breaker.as_ref()
            && let Some(permit) = rc.breaker_pending.take()
        {
            match rc.upstream_status {
                // Defensive only, so its mutants are equivalent (excluded in .cargo/mutants.toml):
                // `response_filter` sets `upstream_status` and takes the permit in the same step,
                // and nothing else sets it on a request that holds a permit (a cache hit never
                // claims one), so a pending permit here always comes with `None`.
                Some(s) if s >= 500 => breaker.record_failure_for(permit),
                Some(_) => breaker.record_success_for(permit),
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
                    breaker.record_failure_for(permit);
                }
                // Client went away, its upload stalled, or the request ended with no error at all:
                // no provider outcome. Give the permit back without one (D86). A success here closed
                // a half-open breaker on a probe that never heard from the provider, letting every
                // caller flood one that may still be broken; `release` returns the probe permit so
                // the next request probes instead.
                None => breaker.release(permit),
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
        let mut usage_estimated = false;
        // Which counts an estimate replaced (see `EstimatedParts`).
        let mut estimated = EstimatedParts::default();
        // A free sub-resource (a token count) is not a billable call: it carries no usage block,
        // writes no billing row, and is not a usage-shape regression. On every route: a catalog
        // walk records it, a `/{provider}` route names it in the forwarded path.
        let free = rc
            .auto
            .as_ref()
            .and_then(|a| a.sub)
            .or_else(|| {
                rc.forward_path
                    .as_deref()
                    .and_then(route::SubResource::of_forward_path)
            })
            .is_some_and(|sub| !sub.billed());
        // Someone gave up waiting for the response head after the provider had the whole request
        // (a long reasoning turn, a huge prompt): the client, or the gateway's own read timeout
        // with the connection still up (D130). The provider cannot tell who hung up and bills that
        // prompt either way. A request that never reached it (every breaker open, a connect
        // failure) costs nothing and stays 0, and so does a peer that closed or reset the
        // connection before answering: that is a refusal, not a wait (see `gave_up_waiting`).
        let no_head = rc.managed
            && cache_hit.is_none()
            && rc.upstream_status.is_none()
            && e.is_some_and(gave_up_waiting)
            && body_delivered(session, rc, e);
        let parsed = if cache_hit.is_some() {
            cache_hit.as_ref().map(|h| h.usage)
        } else {
            let tail = rc.resp_tail.contiguous();
            // A relayed managed 403 whose body names the key (not the request) is that key being
            // refused: cool it so later requests start past it, as a 401 does at the head (D84).
            // This request already answered; it does not walk.
            if rc.managed && rc.upstream_status == Some(403) && body_names_the_key(tail) {
                self.state.metrics.key_cooled(KeyCooled::KeyNamed403);
                rc.provider.mark_key_bad(rc.pool_key);
            }
            // A relayed managed error that says the account is out of credit or quota (Anthropic's
            // credit-balance 400, OpenAI's insufficient_quota 429: read from the body by `Redact`)
            // is that key refused until someone pays, which waiting does not fix. Cool it as a 401
            // is cooled: later requests start past it, and a catalog walk leaves the provider out
            // once all its keys cool (D180). This request already answered. Not an abandoned
            // `FullBody` attempt: its body never reaches the client and is read only if it raced
            // the parent's drop, and a walk's 402 was already judged at the head (D258).
            if rc.managed && !rc.relay_abandoned && rc.redact.as_ref().is_some_and(|r| r.unfunded) {
                let status = rc.upstream_status.unwrap_or(0);
                self.cool_unfunded_key(&rc.request_id, &rc.provider, rc.pool_key, status);
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
            let ok_2xx = rc.upstream_status.is_some_and(|s| (200..300).contains(&s));
            let cut_short = rc.managed
                && rc.streaming
                && ok_2xx
                && match rc.dialect {
                    Dialect::OpenAi => parsed.is_none(),
                    // A `message_delta` that arrived but did not parse is as missing as one that
                    // never came.
                    Dialect::Anthropic => {
                        parsed.is_none() || !usage::anthropic_stream_finished(tail)
                    }
                };
            // A non-stream 2xx that died before its `usage` (the body is last): the provider
            // generated, and bills, the whole answer; we relayed part of it. Always estimated —
            // the 2xx is the proof the provider took the request.
            let body_cut = rc.managed && !rc.streaming && ok_2xx && parsed.is_none() && e.is_some();
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
            let body_unmetered = rc.managed
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
                usage_estimated = true;
                let mut u = parsed.unwrap_or_default();
                if u.input_tokens == 0 {
                    u.input_tokens = input_estimate(session, rc);
                    estimated.input = true;
                }
                if output > u.output_tokens {
                    u.output_tokens = output;
                    estimated.output = true;
                }
                Some(u)
            } else {
                parsed
            }
        };
        if usage_estimated {
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
        if (parsed.is_none() || (usage_estimated && e.is_none()))
            && rc.managed
            && !free
            && cache_hit.is_none()
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
            rc.provider
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
        // An abandoned `FullBody` attempt is not the request the client got: the attempt that
        // serves writes the one row (and the one capture).
        // The gateway's own refusal before any response head (a catalog miss on `/{provider}`, a
        // duplicate `model`, `background`): no provider had the request, so no row, as for every
        // other rejection.
        let refused =
            cache_hit.is_none() && rc.upstream_status.is_none() && e.is_some_and(gateway_refusal);
        if rc.managed && !rc.relay_abandoned && !free && !refused {
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
            // Absent when no provider was called (every candidate's breaker open): `rc.provider`
            // is then just the walk's seed, and naming it would bill a call that never happened.
            let usage_provider = match cache_hit.as_ref() {
                Some(h) => Some(h.provider.as_ref()),
                None => {
                    (rc.upstream_phase != UpstreamPhase::None).then_some(rc.provider.name.as_str())
                }
            };
            let outcome = outcome(rc, e, cache_hit.is_some());
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
            let upstream_host = called.then_some(rc.provider.host.as_str());
            let upstream_path = called.then(|| {
                let p = rc
                    .forward_path
                    .as_deref()
                    .unwrap_or_else(|| session.req_header().uri.path());
                p.split_once('?').map_or(p, |(path, _)| path)
            });
            let price_variant = match (usage_provider, upstream_model.as_deref(), upstream_host) {
                (Some(p), Some(m), Some(h)) if called => price_variant(p, m, h),
                _ => None,
            };
            let upstream_request_id = rc
                .taps
                .as_ref()
                .and_then(|t| t.request_id)
                .filter(|_| called);
            let server_tools = usage.server_tools.to_row();
            let upstream_may_continue =
                upstream_may_continue(usage_provider, outcome, usage_stream);
            let estimate_excludes = estimated.excludes(&usage);
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
            // `crates/providers/ARCHITECTURE.md`, "Pricing contract"). `server_tools` is read
            // back from its logged text so a repricer working from the row gets the same counts.
            // Pure and allocation-free: a few hundred nanoseconds, after the response.
            let priced =
                providers::pricing::ToolCounts::parse(server_tools.as_deref().unwrap_or(""))
                    .and_then(|server_tools| {
                        providers::pricing::price(&providers::pricing::UsageRow {
                            price_model,
                            provider: usage_provider,
                            price_variant,
                            usage_wire: usage.wire.unwrap_or(rc.dialect),
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            cache_read_tokens: usage.cache_read_tokens,
                            cache_write_tokens: usage.cache_write_tokens,
                            cache_write_1h_tokens: usage.cache_write_1h_tokens,
                            gateway_cache_write_tokens: usage.gateway_cache_write_tokens,
                            server_tools,
                            service_tier: usage.service_tier.as_deref(),
                            speed: usage.speed.as_deref(),
                            inference_geo: usage.inference_geo.as_deref(),
                            upstream_cost_usd: upstream_cost_usd.as_deref(),
                            upstream_tool_cost_usd: upstream_tool_cost_usd.as_deref(),
                            served_by: usage.upstream.served_by.as_deref(),
                            usage_estimated,
                            upstream_may_continue,
                            cache_hit: cache_hit.is_some(),
                            unix_secs: start_unix_secs,
                        })
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
                tenant_id = rc.tenant_id,
                vpc_id = rc.vpc_id,
                key_id = rc.key_id,
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
                cache_hit = cache_hit.is_some(),
                // The provider's HTTP status, absent when no response head arrived (and on a cache
                // hit, which made no call); and what became of the request (see `outcome`).
                upstream_status = cache_hit.is_none().then_some(rc.upstream_status).flatten(),
                outcome,
                // True when the stream was cut short before its usage block and the token counts
                // below are the gateway's estimate, not the provider's report. Estimates err low.
                usage_estimated,
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
                usage_estimated_parts = estimated.parts(),
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
                    tenant_id = rc.tenant_id,
                    provider = rc.provider.name.as_str(),
                    model = billed_model,
                    stream = rc.streaming,
                    status = rc.upstream_status,
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
                    complete = e.is_none(),
                    "payload"
                );
            }

            // Fill: complete 2xx only. Client abort, 4xx/5xx, and truncation are all skips — a
            // partial or error body must never be replayed as a success.
            if e.is_none()
                && !usage_estimated
                && rc.upstream_status.is_some_and(|s| (200..300).contains(&s))
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
                        status: rc.upstream_status.unwrap_or(200),
                        content_type: content_type.unwrap_or_else(|| "application/json".into()),
                        body: Bytes::copy_from_slice(body),
                        usage: usage.for_cache(),
                        billed_model: billed_model.to_owned().into_boxed_str(),
                        requested_model: requested_model.to_owned().into_boxed_str(),
                        routed_model,
                        provider: rc.provider.name.clone().into_boxed_str(),
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
    pub(super) fn test_ctx(inject_eligible: bool) -> RequestCtx {
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
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
            dialect: Dialect::OpenAi,
            provider: Arc::new(provider),
            forward_path: None,
            managed: true,
            model: String::new(),
            model_scanner: peek::ModelScanner::new(),
            resp_model_scanner: peek::ModelScanner::for_response(),
            streaming: false,
            resp_tail: UsageTail::default(),
            resp_head: Vec::new(),
            body_bytes_fed: 0,
            upstream_status: None,
            inject_eligible,
            catalog_check: false,
            req_buf: Vec::new(),
            start: Instant::now(),
            deadline: crate::deadline::NONE,
            attempt: 0,
            pool_key: 0,
            same_provider_retry: false,
            relay_abandoned: false,
            refused_resent: false,
            read_capped: false,
            breaker_pending: None,
            auto: None,
            control: None,
            request_id: RequestId::new(),
            input_tally: usage::InputTally::default(),
            tally_eager: false,
            resp_bytes: 0,
            upstream_phase: UpstreamPhase::None,
            redact: None,
            terminal: TerminalTracker::default(),
            signed: None,
            taps: None,
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

    /// The replay contract, exercised directly: `upstream_peer` resets the body phase before pingora
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

        // Attempt 1: a prefix arrives, then the upstream dies before end-of-stream.
        let mut rc = test_ctx(true);
        rc.req_buf.extend_from_slice(prefix);
        rc.body_bytes_fed += prefix.len();
        assert!(
            !rc.req_buf.is_empty(),
            "precondition: the partial body is still held, since end_of_stream never came",
        );

        // `upstream_peer` runs before any body byte of attempt 2.
        rc.reset_request_body_phase();

        // Attempt 2: pingora replays from byte 0, then the remainder streams in.
        rc.req_buf.extend_from_slice(prefix);
        rc.req_buf.extend_from_slice(rest);

        assert_eq!(
            rc.req_buf, body,
            "the retried attempt must send the body exactly once",
        );
        assert_eq!(
            rc.body_bytes_fed, 0,
            "the size guard must not double-count the replayed prefix",
        );
        let scan = peek::scan_buffered(&rc.req_buf);
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

    /// Without the reset, the same sequence corrupts the body — so the test above is not vacuous.
    #[test]
    fn a_replayed_prefix_without_the_reset_corrupts_the_body() {
        let body =
            br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","stream":true}"#;
        let (prefix, rest) = body.split_at(30);

        let mut rc = test_ctx(true);
        rc.req_buf.extend_from_slice(prefix);
        rc.body_bytes_fed += prefix.len();
        // ...no reset...
        rc.req_buf.extend_from_slice(prefix);
        rc.req_buf.extend_from_slice(rest);

        assert_ne!(rc.req_buf, body, "the duplicated prefix must be observable");
        assert!(
            serde_json::from_slice::<serde_json::Value>(&rc.req_buf).is_err(),
            "and the result is not valid JSON — this is the 400 the provider returns",
        );
        assert_eq!(
            rc.body_bytes_fed,
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
            crate::config::AiConfig::default(),
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
            // 432: + the boxed `signed` (8 bytes, `None` off the managed Responses relay).
            // 440: + `deadline` (8 bytes; every request has one, and the per-chunk check reads it).
            // 448: + the boxed `taps` (8 bytes, `None` on BYO): the billing facts beside usage.
            size <= 448,
            "RequestCtx grew to {size} bytes (ceiling 448). It is touched once per response chunk \
             on a stream — if the new state is only needed on one route, box it the way \
             `ModelRouting` is rather than paying for it on every request.",
        );
    }

    /// A model-routed request takes its dialect from **this candidate's path**, not the row and
    /// not the provider. Mixed-wire rows (Claude → OpenRouter Chat Completions) would otherwise
    /// parse a Chat Completions reply with the Anthropic extractor — or the reverse — and emit a
    /// zero-token billing row.
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
        retain_managed_client_headers(&mut req, true).unwrap();
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
        retain_managed_client_headers(&mut req, true).unwrap();
        assert!(req.headers.get("anthropic-beta").is_none());
        let mut req = req_with_headers("/v1/messages", &[]);
        req.append_header("anthropic-beta", "prompt-caching-2024-07-31")
            .unwrap();
        req.append_header("anthropic-beta", "code-execution-2025-05-22")
            .unwrap();
        retain_managed_client_headers(&mut req, true).unwrap();
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
            &gateway_error(404, CATALOG_MISS).into_down()
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
        // These are hand-written JSON literals standing in for what `serde_json::json!` used to
        // build, so the thing to guard is that they still *say* what the `error_type` in the log
        // line and the metric label claim. A drifting literal would ship a response whose `type`
        // contradicts the reason we rejected for.
        for &(typ, msg, body) in &REJECT_BODIES {
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
            assert_eq!(error_body(typ, msg).as_ptr(), body.as_ptr(), "{typ}: {msg}");
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
        for &(_, msg, _) in &REJECT_BODIES {
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

/// Behaviors a mutation-testing pass found no test constraining.
#[cfg(test)]
mod mutation_gaps {
    use super::*;
    use pingora_core::{Error, ErrorType as T};

    /// Each way a request ends maps to its own `outcome` on the billing row, so a zero-token row
    /// for an error or a cancel can be told from a real zero-token generation.
    /// claim: BIL-12
    #[test]
    fn every_ending_has_its_own_outcome() {
        let mut rc = tests::test_ctx(false);
        assert_eq!(outcome(&rc, None, false), "no_candidate");
        assert_eq!(outcome(&rc, None, true), "ok", "a cache hit made no call");
        rc.upstream_phase = UpstreamPhase::Attempted;
        let up = Error::new_up(T::ConnectRefused);
        assert_eq!(outcome(&rc, Some(&up), false), "upstream_error");
        let down = Error::new_down(T::ReadError);
        assert_eq!(outcome(&rc, Some(&down), false), "client_cancelled");
        rc.upstream_phase = UpstreamPhase::Connected;
        rc.upstream_status = Some(200);
        assert_eq!(outcome(&rc, None, false), "ok");
        assert_eq!(outcome(&rc, Some(&up), false), "cut_short");
        assert_eq!(outcome(&rc, Some(&down), false), "client_cancelled");
        rc.upstream_status = Some(400);
        assert_eq!(outcome(&rc, None, false), "upstream_error");
        assert_eq!(outcome(&rc, Some(&down), false), "upstream_error");
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
        retain_managed_client_headers(&mut req, true).unwrap();
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
            retain_managed_client_headers(&mut req, anthropic).unwrap();
            req.headers
                .get("anthropic-beta")
                .map(|v| v.to_str().unwrap().to_owned())
        };
        assert_eq!(header(true).as_deref(), Some(value));
        assert_eq!(header(false).as_deref(), Some("prompt-caching-2024-07-31"));
    }
}
