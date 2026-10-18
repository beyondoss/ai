//! Upstream attempts: the peer, how an attempt failed, and a pool key's health.

use super::*;

/// Whether the provider has this attempt's whole request: the connection was up, the client's body
/// was read to its end (and so forwarded), and nothing failed writing it upstream. A request in
/// that state is one the provider can bill for even when nothing comes back; one that is not (a
/// connect failure, a reset mid-upload) never reached it.
///
/// The client side stands in for the upstream side: pingora forwards each body chunk as it reads
/// it, and a failure on the way out surfaces as a write error. So this can only over-report by a
/// chunk still in flight when a read-side error lands, never miss a request the provider has.
pub(super) fn body_delivered(
    session: &mut Session,
    rc: &RequestCtx,
    e: Option<&pingora_core::Error>,
) -> bool {
    use pingora_core::ErrorType::{WriteError, WriteTimedout};
    rc.upstream_phase == UpstreamPhase::Connected
        && session.as_mut().is_body_done()
        && !e.is_some_and(|e| {
            (e.esource() == &pingora_core::ErrorSource::Upstream
                && (matches!(e.etype(), WriteError | WriteTimedout) || h2_body_unsent(e)))
                || client_body_cut_short(e)
        })
}

/// Whether the client closed its HTTP/1.1 connection before its body's framing-defined end:
/// a `Content-Length` body with bytes still to come, or a chunked one without its
/// terminating chunk. Pingora then marks the body *done* (`is_body_done`), though never complete,
/// so without this the half a provider was sent read as the whole request, and `logging` billed
/// it an estimate. Like [`h2_body_unsent`], the context string is pingora's only handle on it.
pub(super) fn client_body_cut_short(e: &pingora_core::Error) -> bool {
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
/// request. The usual cause is a GOAWAY that closed a stream mid-upload; pingora gives no
/// other handle on it than the context string.
pub(super) fn h2_body_unsent(e: &pingora_core::Error) -> bool {
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
/// upstream connection that was still up. Once the provider has the whole request
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
/// nothing was waited on and nothing is billed.
pub(super) fn gave_up_waiting(e: &pingora_core::Error) -> bool {
    match e.esource() {
        pingora_core::ErrorSource::Downstream => !gateway_refusal(e),
        pingora_core::ErrorSource::Upstream => e.etype() == &pingora_core::ErrorType::ReadTimedout,
        _ => is_deadline(e),
    }
}

/// Whether `e` is the gateway's own answer to the client (a `CustomCode` or `HTTPStatus` tagged
/// downstream: a catalog miss, a duplicate `model`, `background`, a foreign id, an oversized body),
/// not the client going away. Before any response head, such a request never reached a provider
/// whole, so it bills nothing and writes no `ai.usage` row (rejections write none).
pub(super) fn gateway_refusal(e: &pingora_core::Error) -> bool {
    use pingora_core::ErrorType::{CustomCode, HTTPStatus};
    e.esource() == &pingora_core::ErrorSource::Downstream
        && matches!(e.etype(), CustomCode(..) | HTTPStatus(_))
}

/// Whether the upstream refused this request's HTTP/2 stream with a GOAWAY, one shape of
/// [`upstream_refused_stream`]: the connection is retired, so a resend takes another.
pub(super) fn upstream_goaway(e: &pingora_core::Error) -> bool {
    e.root_cause()
        .downcast_ref::<h2::Error>()
        .is_some_and(|h| h.is_remote() && h.is_go_away())
}

/// Whether the upstream refused this request's HTTP/2 stream before processing any of it, so it
/// is safe to resend whatever was written. Two shapes, both guaranteed by RFC 9113:
///
/// - `RST_STREAM(REFUSED_STREAM)` from the peer (§8.7: "closed prior to any processing").
/// - A remote GOAWAY on the stream. The `h2` client hands a stream that error only when its id is
///   above the GOAWAY's `last_stream_id` (`Streams::recv_go_away`), which is §6.8's "not
///   processed": a stream at or below it that dies later fails with an I/O error instead. Any
///   reason code — a GOAWAY(ENHANCE_YOUR_CALM) refuses as surely as a graceful NO_ERROR one.
pub(super) fn upstream_refused_stream(e: &pingora_core::Error) -> bool {
    e.root_cause().downcast_ref::<h2::Error>().is_some_and(|h| {
        h.is_remote()
            && (h.is_go_away() || (h.is_reset() && h.reason() == Some(h2::Reason::REFUSED_STREAM)))
    })
}

/// Whether a **reused** HTTP/1.1 connection was reset before any response byte: the
/// provider's kernel answered with RST because the socket was closed with our request unread in it,
/// the idle-close race on a pooled connection, so the request reached no server. Pingora marks this
/// `ReusedOnly`, retryable on a reused connection.
///
/// Not a clean end-of-file: a server that read the request and then closed looks exactly like one
/// that closed first, and the first may be generating. Not a timeout either (`ETIMEDOUT` is a
/// liveness verdict on a dead peer that may have taken the request), nor an HTTP/2 I/O error, which
/// carries no such guarantee ([`upstream_refused_stream`] covers HTTP/2's own refusals).
pub(super) fn reset_before_reading(e: &pingora_core::Error, client_reused: bool) -> bool {
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

/// Whether `e` is the client closing after the stream's terminal event was written to it.
/// openai-python closes at `[DONE]`, Codex at `response.completed`; when that close lands
/// before the provider's end of stream, pingora ends the request with a downstream error although
/// the client read the whole answer.
///
/// "Written" is exact, not a guess at timing: pingora runs `response_body_filter` on a chunk and
/// then awaits its write to the client before it polls the client again, so the terminal bytes
/// [`TerminalTracker`] saw either went out, or their write failed and the error is that write's
/// (`WriteError` / `WriteTimedout`). Any other downstream error (the client's FIN or reset, read
/// after the write returned) came after the client had the terminal event.
pub(super) fn closed_after_terminal(rc: &RequestCtx, e: &pingora_core::Error) -> bool {
    use pingora_core::ErrorType as T;
    rc.terminal.ended()
        && e.esource() == &pingora_core::ErrorSource::Downstream
        && !matches!(e.etype(), T::WriteError | T::WriteTimedout)
}

/// A 2xx's held-back health verdict, waiting on the body to say whether it is an answer.
pub(super) struct PendingHealth {
    /// The body's first bytes, up to [`HEALTH_PREFIX_CAP`].
    pub(super) prefix: Vec<u8>,
    /// The breaker permit the head resolved as a success. The body's verdict is recorded against it,
    /// in its own generation, so a later failure cannot reopen a probe that another request owns by
    /// then.
    pub(super) permit: Option<Answered>,
}

impl PendingHealth {
    pub(super) fn extend_prefix(&mut self, bytes: &[u8]) {
        self.prefix.extend_from_slice(bytes);
    }
}

/// How much of a 2xx body `settle_health` reads before calling it an answer. An error-in-200 says
/// so in its first key (`{"error":` / `{"type":"error"`) or its first SSE event.
pub(super) const HEALTH_PREFIX_CAP: usize = 1024;

/// A managed status that means the pool key itself was refused (revoked, wrong): a 401. It cools
/// the key off and walks to the next one; on a catalog walk's last key it is a candidate failure.
/// A 403 is not: it is usually about the request (moderation, a model the key's project may not
/// use, Anthropic's `permission_error`), so one tenant's 403 must not move the shared pool off a
/// key. A 403 whose body names the key is cooled from the body ([`body_names_the_key`]).
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
/// the status alone says out of credit (`remedy::unfunded_402`). Each is counted on
/// `ai_key_auth_failures_total` by reason (`metrics::KeyCooled`).
pub(crate) fn is_candidate_refusal(status: u16) -> bool {
    (401..=403).contains(&status)
}

/// Whether an error body says the credential itself is bad: OpenAI's `invalid_api_key` code, or
/// Anthropic's `authentication_error` type. Read from a relayed managed 403, which then cools the
/// key for later requests (this one has already answered).
pub(super) fn body_names_the_key(body: &[u8]) -> bool {
    memchr::memmem::find(body, b"\"invalid_api_key\"").is_some()
        || memchr::memmem::find(body, b"\"authentication_error\"").is_some()
}

/// Whether a 2xx body's first bytes are an error: `Some(true)` an error object, `Some(false)` an
/// answer, `None` not decidable yet. Non-streaming: the root object's first key is `error`, or a
/// first key `type` whose value is `"error"` (Anthropic's shape). Streaming: an `event: error`
/// line, or a first `data:` event whose payload is such an object.
pub(super) fn body_reports_error(prefix: &[u8], streaming: bool) -> Option<bool> {
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
pub(super) fn json_is_error_object(bytes: &[u8]) -> Option<bool> {
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

/// Whether a whole non-stream body is only an error: a root object with a non-null
/// `error` member and none of the members an answer is carried in (`choices`, `output`,
/// `content`), whatever order its keys come in. OpenRouter's error-in-200 is `{"error": {...}}`,
/// but a provider may write `id`, `object` or `model` first; a Responses answer carries
/// `"error": null` beside its `output`.
pub(super) fn json_is_error_only(body: &[u8]) -> bool {
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
pub(super) fn json_leading_string(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let body = bytes.strip_prefix(b"\"")?;
    let end = memchr::memchr(b'"', body)?;
    Some((&body[..end], &body[end + 1..]))
}

/// Where a failed attempt's body can be resent from, for [`retry::decide`]. Reads the `FullBody`
/// parent in place: a clone of it is taken only to record a re-run.
pub(super) fn body_facts(session: &mut Session, walk: Option<retry::Walk>) -> retry::Body {
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

impl AiProxy {
    /// Cap this attempt's upstream read timeout at the time left before the request's deadline,
    /// when that is the shorter bound, and say so in `read_capped`: pingora re-arms the timeout on
    /// every read and every read starts after this attempt did, so a capped timeout can only fire
    /// at or past the deadline, and `error_while_proxy` turns it into [`Self::deadline_cut`]. This
    /// is what ends a silent stream (an HTTP/2 PING keeps the connection alive, and a `-pro` row
    /// has no silence bound at all); a moving one is cut per chunk. No time left: the request ends
    /// here, before a connection is made that the provider would bill.
    pub(super) fn cap_read_at_deadline(
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
    pub(super) fn build_peer(
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
        // every concurrent request open its own TLS connection: H2 without the multiplexing.
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
        // (`long_running`: OpenAI's `-pro` models) gets no silence deadline at all: a
        // non-stream call there can outlast 600s, and a 504 is still billed by the provider. Its
        // end is the client's own timeout (a cancel the gateway sees and bills) or a dead
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

    /// Cool this attempt's pool key because its account is out of credit: later requests
    /// start past it, and a catalog walk leaves the provider out once all its keys cool. Waiting
    /// does not fix it, so it is cooled as a 401 is.
    pub(super) fn cool_unfunded_key(
        &self,
        request_id: &RequestId,
        provider: &Provider,
        key: u8,
        status: u16,
    ) {
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
    /// the key. OpenRouter's, which may be one request too costly instead: a strike against
    /// the key, and the [`route::OPENROUTER_402_STRIKES`]th in a row with no 2xx between cools it.
    /// A relayed 402 is read by `logging` instead: out of credit cools the key (which spends
    /// its strikes), too costly neither counts nor resets one.
    pub(super) fn abandon_402(&self, rc: &RequestCtx, by_status: Option<bool>, status: u16) {
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
    pub(super) fn settle_health(&self, rc: &mut RequestCtx, chunk: &[u8], end_of_stream: bool) {
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
pub(super) fn is_upstream_failure(e: Option<&pingora_core::Error>) -> bool {
    e.is_some_and(|e| {
        !matches!(e.esource(), pingora_core::ErrorSource::Downstream) && !is_deadline(e)
    })
}

/// How much of the request body one attempt has fed through `request_body_filter`: one count, read
/// by name for what each reader asks of it.
#[derive(Debug, Default)]
pub(super) struct UploadProgress(pub(super) usize);

impl UploadProgress {
    /// Count a chunk of `n` bytes. `false` once the body has crossed [`MAX_REQUEST_BODY`]: the cap
    /// holds on the streamed size too, which a chunked body declares nowhere up front.
    pub(super) fn feed(&mut self, n: usize) -> bool {
        self.0 = self.0.saturating_add(n);
        self.0 <= MAX_REQUEST_BODY
    }

    /// Whether any chunk has reached `request_body_filter` this attempt: what blames a stalled
    /// upload on the client (`client_still_uploading`; none fed means the attempt never got past
    /// connecting), and, false, what tells a retry there is nothing to undo, since every
    /// body-phase accumulator is fed on the branch that counts here.
    pub(super) fn started(&self) -> bool {
        self.0 > 0
    }
}

/// Whether this attempt had started sending the client's body upstream and the client had not
/// finished it. An upstream error then reflects the client (a stalled or abandoned upload), not the
/// provider. Zero bytes fed means the attempt never got past connecting.
pub(super) fn client_still_uploading(session: &mut Session, rc: &RequestCtx) -> bool {
    rc.attempt.upload.started() && !session.as_mut().is_body_done()
}

/// Whether the upstream's read timeout fired while the client still owed body bytes: the
/// client stalled mid-upload with its connection open. Pingora's upstream half waits on the
/// response and on the next body chunk at once, re-arming the read timeout on each chunk, and a
/// body write blocked on an upstream that stopped reading is a `WriteTimedout` instead; so this
/// timeout means neither side moved a byte for `read_timeout_secs`, with the provider waiting on
/// the client. Zero bytes fed is excluded as [`client_still_uploading`] excludes it: a client
/// that sent `Expect: 100-continue` waits on the provider's answer before its first byte. Any
/// source: pingora's own downstream body read timeout (HTTP/1.1, 60 s) is already the client's.
pub(super) fn client_stalled_upload(
    session: &mut Session,
    rc: &RequestCtx,
    e: &pingora_core::Error,
) -> bool {
    e.etype() == &pingora_core::ErrorType::ReadTimedout && client_still_uploading(session, rc)
}

/// Pingora will only replay a body that has fully arrived and fit in its private 64 KiB buffer.
/// See `upstream_response_filter` — the same gate for a 429 key-walk and a 5xx vendor walk.
pub(super) fn body_replayable(session: &mut Session) -> bool {
    session.as_mut().is_body_done() && !session.as_ref().retry_buffer_truncated()
}
