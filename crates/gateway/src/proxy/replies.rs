//! Answers the gateway writes itself: refusals, the models list, a cache hit.

use super::*;

/// Every answer `fail_to_proxy` gives an error that is not a [`Refusal`] (a connect failure, an
/// upstream that failed after it had the whole request, every candidate's breaker open), with its
/// body spelled out at compile time: `(type, message)` as [`failure_response`] and `fail_to_proxy`
/// pair them, so a burst of provider failures is answered without building JSON.
pub(super) static FAILURES: [refusal::Fixed; 17] = [
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
    /// End a request at its deadline (`request_max_secs`): counted, logged, and answered 504 if no
    /// response has started (after one has, pingora cuts the stream as any mid-stream error does).
    pub(super) fn deadline_cut(&self, request_id: &str) -> Box<pingora_core::Error> {
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
    pub(super) fn refusal_error(&self, refusal: Refusal) -> Box<pingora_core::Error> {
        self.state.metrics.rejection(refusal.metric()).inc();
        refusal.into_error()
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
    pub(super) fn refuse<'a>(
        &self,
        session: &'a mut Session,
        request_id: &'a str,
        refusal: Refusal,
    ) -> Pin<Box<impl Future<Output = Result<bool>> + Send + 'a>> {
        self.state.metrics.rejection(refusal.metric()).inc();
        Box::pin(write_refusal(session, request_id, refusal))
    }

    /// `body` is [`GatewayState::models_list`], rendered at boot; serving it is a refcount bump.
    pub(super) async fn reply_models_list(
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

    pub(super) async fn reply_models_list_boxed(
        session: &mut Session,
        request_id: &str,
        body: Bytes,
    ) -> Result<bool> {
        Box::pin(Self::reply_models_list(session, request_id, body)).await
    }

    /// Replay a cached 2xx. Boxed so its write future is not inlined into `request_filter`.
    pub(super) async fn reply_cache_hit(
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

    pub(super) async fn reply_cache_hit_boxed(
        session: &mut Session,
        request_id: &str,
        hit: &cache::CachedResponse,
    ) -> Result<bool> {
        Box::pin(Self::reply_cache_hit(session, request_id, hit)).await
    }
}

/// Most `request rejected` lines logged per second, process-wide (D263). A rejection is the one
/// line a client can make the gateway write at will — bad keys, a rate-limited flood — and one line
/// per rejected request made the log the bottleneck of a flood the rejection was meant to shed. A
/// second's first lines still carry their `request_id` for the oncall's grep; past the allowance
/// they are counted, and the next line logged says how many were not (`suppressed`).
/// `ai_rejections_total` still counts every one.
pub(super) const REJECT_LOG_PER_SEC: u64 = 100;

/// Process-wide, as the log it protects is: every gateway in the process writes the one stdout.
pub(super) static REJECT_LOG: LogRate = LogRate::new(REJECT_LOG_PER_SEC);

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
pub(super) fn log_rejection(request_id: &str, status: u16, typ: &str) {
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
pub(super) fn log_second() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs()
}

/// Write `refusal`: its status, body and `Retry-After`, and `x-beyond-request-id`, after the
/// `request rejected` line. See [`AiProxy::refuse`], which counts it and boxes this.
pub(super) async fn write_refusal(
    session: &mut Session,
    request_id: &str,
    refusal: Refusal,
) -> Result<bool> {
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
pub(super) async fn fail_boxed(
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
pub(super) fn gateway_error(status: u16, msg: &'static str) -> Box<pingora_core::Error> {
    pingora_core::Error::new(pingora_core::ErrorType::CustomCode(msg, status))
}

/// Whether `e` is the request deadline passing ([`AiProxy::deadline_cut`]). A policy limit, never a
/// provider outcome: [`is_upstream_failure`] is false for it, so no breaker hears of it, and
/// `error_while_proxy` passes it through as the gateway's own decision (no resend, no failover).
pub(super) fn is_deadline(e: &pingora_core::Error) -> bool {
    matches!(Refusal::of(e), Some(Refusal::RequestDeadline))
}

/// The catalog miss for an `x-beyond-model` header that names no row, naming what it sent.
pub(super) fn header_catalog_miss(session: &Session) -> Refusal {
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
pub(super) fn failure_response(
    e: &pingora_core::Error,
) -> Option<(u16, &'static str, &'static str)> {
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
pub(super) async fn write_json_body(
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
