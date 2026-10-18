//! Reading a body up front, and re-running one past the retry buffer (`FullBody`).

use super::*;

impl AiProxy {
    /// The bound [`peek_body_model`] puts on each body read itself: `client_read_timeout_secs`,
    /// where the session has no read timeout of its own (HTTP/2; HTTP/1.1 enforces it inside
    /// pingora, so a second timer would be waste).
    pub(super) fn up_front_read_timeout(&self, session: &Session) -> Option<Duration> {
        let secs = self.state.config.client_read_timeout_secs();
        (secs > 0 && session.as_ref().get_read_timeout().is_none())
            .then(|| Duration::from_secs(secs))
    }

    /// Re-run a request whose body the gateway read in full as a pingora subrequest carrying that
    /// body, and pipe its response back; re-run again on the next candidate or pool key when an
    /// attempt asks for it (see [`FullBody`]). Each attempt is a complete request of its own (auth,
    /// deny and allowance checks, the walk, translation), under this request's id; only the attempt
    /// that serves writes `ai.usage`.
    pub(super) async fn relay_full_body(
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
}

/// Same cap as pingora's private `BODY_BUF_LIMIT`. The managed `/v1` / headerless `/auto` peek
/// reads at most this many bytes before `upstream_peer`. A root `model` that has not appeared by
/// then is missing — 404. Past this, pingora will not replay the prefix itself (the retry buffer
/// truncates), so [`ModelRouting::replay`] carries our copy for `request_body_filter` to prepend.
pub(super) const BODY_PEEK_LIMIT: usize = 64 * 1024;

/// Whether a body of `n` bytes is past [`BODY_PEEK_LIMIT`]: pingora cannot replay it from its own
/// buffer, so every copy the gateway holds of it is charged to the process body budget. Below it,
/// concurrency bounds the bodies in memory.
pub(super) fn past_replay_buffer(n: usize) -> bool {
    n > BODY_PEEK_LIMIT
}

/// [`RequestCtx::tally_eager`]: a managed provider-routed body that may outgrow the retry buffer
/// is the one case where no copy is left for `logging` to tally.
pub(super) fn tally_eager(managed: bool, catalog: bool, declared_len: Option<usize>) -> bool {
    managed && !catalog && declared_len.is_none_or(past_replay_buffer)
}

/// What `peek_body_model` read before a catalog row could be chosen.
pub(super) struct BodyPeek {
    pub(super) model: Option<String>,
    /// The full pre-rewrite body, when the peek read it to the end. Read only once `over_cap` and
    /// `over_budget` are known to be false.
    pub(super) complete: Option<Vec<u8>>,
    /// The whole body was read and it outgrew pingora's 64 KiB retry buffer: pingora has nothing
    /// to send and nothing left to read. The caller re-runs the request with `complete` as its body
    /// (see [`FullBody`]).
    pub(super) relay: bool,
    /// [`MAX_REQUEST_BODY`] was reached before the body ended. 413.
    pub(super) over_cap: bool,
    /// Holding more of the body would cross `max_buffered_body_bytes`. 503.
    pub(super) over_budget: bool,
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
pub(super) async fn peek_body_model(
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
pub(super) fn expects_continue(session: &Session) -> bool {
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
pub(super) struct FullBody {
    pub(super) route: &'static route::ModelRoute,
    pub(super) session_field: Option<&'static str>,
    /// Catalog indices (bit per index) an earlier attempt failed over from.
    pub(super) skip: u8,
    /// Catalog indices (bit per index) that cannot serve this body (`route::unserved`), read by the
    /// parent while it held the whole body.
    pub(super) unserved: u8,
    /// The walk takes the row's Responses arm for a tool count Chat Completions refuses
    /// (`route::tools_need_responses_arm`), read by the parent while it held the whole body.
    pub(super) responses_tools: bool,
    /// Per catalog index, the pool key an earlier attempt's key walk moved to, or [`NO_KEY_WALK`]:
    /// start on the provider's first key not cooling off (`Provider::first_key`, D71/D83).
    pub(super) keys: [u8; route::MAX_CANDIDATES],
    /// Catalog indices (bit per index) that already had their one same-candidate retry after a
    /// stale reused connection failed before the upstream had the body.
    pub(super) reset: u8,
    /// A key walk in progress on this catalog index: it goes first in the walk, so a re-ordered row
    /// cannot move a 429's key walk onto another vendor. The other candidates stay usable, so a
    /// 5xx (or a last key's auth failure) that ends the key walk still fails over (D81).
    pub(super) resume: Option<u8>,
    /// The parent's attempt bound is reached: record no retry, relay this attempt's answer.
    final_attempt: bool,
    /// Set by this attempt when it would have retried but could not replay the body.
    retry: Arc<std::sync::Mutex<Option<RelayRetry>>>,
    /// The parent's request id and sequence: every attempt is the same request to the client and
    /// in the logs, and the same `x-beyond-split` draw and probe seed to the walk (a new seq per
    /// attempt could land a 429 key walk on another vendor).
    pub(super) request_id: RequestId,
    pub(super) request_seq: u64,
    /// The parent holds this tenant's concurrency slot for the whole request, taken before it read
    /// the body. Attempts neither take nor release one.
    pub(super) slot_held: bool,
    /// The parent's deadline (`RequestCtx::deadline`): every attempt ends by it.
    pub(super) deadline: u64,
    /// The parent's copy of the body (a refcount, not a copy): what `logging` tallies for an input
    /// estimate ([`input_estimate`]).
    pub(super) body: Bytes,
}

/// What every [`FullBody`] attempt inherits from the request that re-runs it.
#[derive(Clone, Copy)]
pub(super) struct Parent {
    pub(super) request_id: RequestId,
    pub(super) request_seq: u64,
    pub(super) deadline: u64,
}

/// [`FullBody::keys`] for a candidate no earlier attempt walked keys on.
pub(super) const NO_KEY_WALK: u8 = u8::MAX;

/// Move the walk slot holding catalog index `orig` to the front, keeping the others in order.
/// A no-op when `orig` is not in the walk.
pub(super) fn walk_front(walk: &mut control::Walk, orig: u8) {
    let len = usize::from(walk.len).min(route::MAX_CANDIDATES);
    if let Some(at) = walk.indices[..len].iter().position(|&i| i == orig) {
        walk.indices[..=at].rotate_right(1);
    }
}

impl FullBody {
    /// Hand a retry back to the parent. Returns whether it was recorded: never on the final
    /// attempt, whose answer is relayed whatever it is.
    pub(super) fn record(&self, retry: RelayRetry) -> bool {
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

pub(super) fn full_body_ctx(session: &Session) -> Option<FullBody> {
    full_body(session).cloned()
}

/// This attempt's [`FullBody`] context, in place: `None` unless it is a `FullBody` re-run.
pub(super) fn full_body(session: &Session) -> Option<&FullBody> {
    session
        .subrequest_ctx
        .as_ref()
        .and_then(|c| c.user_ctx())
        .and_then(|u| u.downcast_ref::<FullBody>())
}

/// How long [`AiProxy::relay_full_body`] waits for an abandoned attempt to wind down before it
/// starts the next one anyway.
pub(super) const ABANDONED_ATTEMPT_GRACE: Duration = Duration::from_secs(2);

/// What [`AiProxy::relay_full_body`] does once an attempt's pipe has ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttemptEnd {
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
pub(super) fn attempt_end(piped: &Result<Piped>, decision: Option<RelayRetry>) -> AttemptEnd {
    match (piped, decision) {
        (Ok(Piped::Written), _) => AttemptEnd::Done,
        (Err(e), _) if e.esource() == &pingora_core::ErrorSource::Downstream => AttemptEnd::Fail,
        (_, Some(d)) => AttemptEnd::Retry(d),
        (_, None) => AttemptEnd::Fail,
    }
}

/// How a finished attempt's task ended, as [`reap_attempt`] saw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reaped {
    Finished,
    Panicked,
    StillRunning,
}

impl Reaped {
    /// The 502 for an attempt that ended with neither a response nor an error, saying whether it
    /// panicked.
    pub(super) fn unanswered(self) -> Box<pingora_core::Error> {
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
pub(super) async fn reap_attempt<T>(
    attempt: tokio::task::JoinHandle<T>,
    request_id: &str,
) -> Reaped {
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
pub(super) enum Piped {
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
pub(super) fn create_full_body_subrequest(
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
pub(super) async fn pipe_full_body(
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
