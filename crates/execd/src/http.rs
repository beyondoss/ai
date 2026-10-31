//! The HTTP surface: hyper (HTTP/1.1, plus prior-knowledge HTTP/2) in front of the exec handlers.
//!
//! **Bounds.** One task and one socket per open connection: freed when the client closes, when
//! hyper's header-read timeout closes an idle or stalled one, or when the kernel drops a peer that
//! stopped acknowledging (the socket's user timeout). The number of connections is not capped by
//! execd; the file-descriptor limit ends it, and an accept error pauses the loop briefly instead of
//! spinning. HTTP/2 concurrent streams per connection are hyper's default (200). One request holds
//! at most [`crate::Options::max_request_bytes`] of body (plus the parsed copy of it) until its
//! handler returns; N concurrent cold callers hold N times that, and the callers are the agent's
//! own tool calls, which are bounded by its session.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt as _, Limited};
use hyper::body::{Body as HttpBody, Frame, Incoming, SizeHint};
use hyper::header::{AUTHORIZATION, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE};
use hyper::{Method, Request as HttpRequest, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;

use crate::{Ctx, Route, RouteBody};

/// A response body: one buffer, or a stream of frames fed through a bounded channel.
pub(crate) enum Body {
    Full(Option<Bytes>),
    Stream(mpsc::Receiver<Bytes>),
}

impl HttpBody for Body {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        match self.get_mut() {
            Body::Full(b) => Poll::Ready(b.take().map(|b| Ok(Frame::data(b)))),
            Body::Stream(rx) => rx.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b)))),
        }
    }

    fn is_end_stream(&self) -> bool {
        matches!(self, Body::Full(None))
    }

    fn size_hint(&self) -> SizeHint {
        match self {
            Body::Full(Some(b)) => SizeHint::with_exact(b.len() as u64),
            Body::Full(None) => SizeHint::with_exact(0),
            Body::Stream(_) => SizeHint::default(),
        }
    }
}

pub(crate) type Resp = Response<Body>;

/// A JSON answer with this status.
pub(crate) fn json_response(status: u16, body: &Value) -> Resp {
    let mut r = Response::new(Body::Full(Some(Bytes::from(body.to_string()))));
    *r.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    r.headers_mut().insert(
        CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/json"),
    );
    r
}

pub(crate) fn error_response(status: u16, msg: &str) -> Resp {
    json_response(status, &json!({ "error": msg }))
}

/// A stream's response head, and the sender that feeds its body. The channel is bounded, so a
/// slow client stalls whoever sends (the pipe readers) instead of growing memory.
pub(crate) fn stream_response() -> (Resp, mpsc::Sender<Bytes>) {
    // Depth 1: the least that lets the producer build the next frame while hyper writes the last.
    // Nothing observable depends on it; backpressure holds at any depth.
    let (tx, rx) = mpsc::channel(1);
    let mut r = Response::new(Body::Stream(rx));
    let h = r.headers_mut();
    h.insert(
        CONTENT_TYPE,
        hyper::header::HeaderValue::from_static("application/x-ndjson"),
    );
    h.insert(
        CACHE_CONTROL,
        hyper::header::HeaderValue::from_static("no-store"),
    );
    // Behind nginx: the probe refuses an endpoint whose frames arrive all at once.
    h.insert(
        "x-accel-buffering",
        hyper::header::HeaderValue::from_static("no"),
    );
    (r, tx)
}

/// What the agent sends, minus the job operation, which is parsed once it is known to be one.
#[derive(Deserialize)]
pub(crate) struct Request {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub cwd: Option<String>,
    /// Absent: no deadline, the connection is the command's lifetime.
    pub timeout_ms: Option<u64>,
    /// Standard base64, fed to the command's stdin and then closed.
    pub stdin: Option<String>,
    /// Kept as a value: only `true` streams, and any other type is an unknown shape to ignore.
    pub stream: Option<Value>,
    pub job: Option<Value>,
}

impl Request {
    pub fn wants_stream(&self) -> bool {
        self.stream == Some(Value::Bool(true))
    }
}

/// One line of the request log. Never holds a token or an argument: the command's name only.
pub(crate) struct Line {
    pub started: Instant,
    pub kind: &'static str,
    pub command: String,
}

impl Line {
    pub fn new(kind: &'static str, command: &str) -> Self {
        Self {
            started: Instant::now(),
            kind,
            command: command.rsplit('/').next().unwrap_or(command).to_string(),
        }
    }

    pub fn done(self, status: u16, outcome: &str) {
        crate::log_info!(
            "request kind={} command={:?} status={status} outcome={outcome:?} ms={}",
            self.kind,
            self.command,
            u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
        );
    }
}

/// Constant-time equality: the work done does not depend on where the inputs first differ. The
/// lengths are folded in rather than returned early on.
pub(crate) fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

/// Whether the request carries the bearer. Strict, and fail closed: exactly one `Authorization`
/// header (two are ambiguous, so neither order is believed), whose value is `Bearer`, one space and
/// the token. The scheme is matched case-sensitively, as the agent sends it; `bearer <token>`,
/// `Bearer  <token>` and a token with a prefix, suffix or trailing bytes of its own are all refused.
/// (HTTP/1.1 strips optional whitespace around a header value before this sees it, so a trailing
/// space on the wire is not part of the credential; over HTTP/2 nothing is stripped.) Nothing but
/// `Authorization` is consulted: `Proxy-Authorization` is for a proxy.
fn authorized(ctx: &Ctx, headers: &hyper::HeaderMap) -> bool {
    if let Some(authorizer) = &ctx.opts.authorizer {
        return authorizer(headers);
    }
    let Some(want) = &ctx.opts.bearer else {
        return true;
    };
    let mut all = headers.get_all(AUTHORIZATION).iter();
    let (Some(value), None) = (all.next(), all.next()) else {
        return false;
    };
    value
        .as_bytes()
        .strip_prefix(b"Bearer ")
        .is_some_and(|got| ct_eq(got, want.as_bytes()))
}

/// Accept connections until `stop`.
pub(crate) async fn accept_loop(
    listener: tokio::net::TcpListener,
    ctx: Arc<Ctx>,
    stop: CancellationToken,
) {
    loop {
        tokio::select! {
            () = stop.cancelled() => return,
            accepted = listener.accept() => match accepted {
                Ok((sock, _)) => {
                    tokio::spawn(connection(sock, ctx.clone(), stop.clone()));
                }
                Err(e) => {
                    // Out of descriptors, mostly. Retrying at once would spin; the pause is not a
                    // limit, only the time for one open connection to close.
                    crate::log_warn!("accept failed error={e}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }
}

/// How long a connection may sit without a complete request head (a stalled client, or an idle
/// keep-alive). The agent's pool drops an idle connection after 30 s (`pool_idle_timeout`, and
/// N40), so closing at twice that never beats the client to it.
const HEAD_TIMEOUT: Duration = Duration::from_secs(60);

/// hyper's write-buffer cap: with 16 KiB frames anything past one frame only holds memory for a
/// client that is not reading (see [`crate::frame::READ_BUF`]); 16 KiB is no slower than 400 KiB.
const HTTP_BUF: usize = 16 * 1024;

async fn connection(sock: tokio::net::TcpStream, ctx: Arc<Ctx>, stop: CancellationToken) {
    let _ = sock.set_nodelay(true);
    // A peer that vanished without a FIN is dropped by the kernel once data it was sent has gone
    // unacknowledged this long: the spec's bound on a vanished client, two heartbeat intervals
    // (streaming "Disconnect kills"). The heartbeat keeps data in flight on an idle stream.
    let user_timeout = u32::try_from((ctx.opts.heartbeat * 2).as_millis()).unwrap_or(u32::MAX);
    let _ = nix::sys::socket::setsockopt(
        &sock,
        nix::sys::socket::sockopt::TcpUserTimeout,
        &user_timeout,
    );
    let idle_timeout = ctx.opts.idle_timeout;
    let activity = Arc::new(ConnActivity::new());
    let tracked = activity.clone();
    let ctx_for_svc = ctx.clone();
    let svc = hyper::service::service_fn(move |mut req: HttpRequest<Incoming>| {
        let ctx = ctx_for_svc.clone();
        let tracked = tracked.clone();
        async move {
            // The embedder's route hook sees a request after the authorizer and before anything
            // reads it. A claimed request is not execd's work and is never counted, so a stream
            // it holds open keeps neither the server nor its connection awake. A declined one is
            // counted and handled as ever, with the authorizer's verdict already in hand.
            let mut checked = false;
            if let Some(hook) = ctx.opts.route_hook.as_ref().filter(|_| !is_health(&req)) {
                if !authorized(&ctx, req.headers()) {
                    crate::log_info!("request kind=auth status=401");
                    let guard = InFlight::enter(&tracked, &ctx);
                    let resp = error_response(401, "unauthorized");
                    return Ok::<_, Infallible>(resp.map(|b| Tracked::own(b, Some(guard))));
                }
                match hook(req) {
                    Route::Claimed(answer) => {
                        return Ok(answer.await.map(Tracked::claimed));
                    }
                    Route::Declined(r) => req = r,
                }
                checked = true;
            }
            // The health paths are not work: they neither count nor keep a connection awake.
            let guard = (!is_health(&req)).then(|| InFlight::enter(&tracked, &ctx));
            let resp = route(ctx, req, checked).await;
            // The guard rides the body: a stream is in flight until its last frame is written.
            Ok(resp.map(|b| Tracked::own(b, guard)))
        }
    });
    let mut builder = auto::Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .max_buf_size(HTTP_BUF)
        .header_read_timeout(HEAD_TIMEOUT);
    let conn = builder.serve_connection(TokioIo::new(sock), svc);
    tokio::pin!(conn);
    // With no idle timeout this future never completes and never arms a timer.
    let idle = async {
        match idle_timeout {
            Some(t) => activity.idle_for(t).await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(idle);
    let mut closing = false;
    loop {
        tokio::select! {
            r = conn.as_mut() => {
                if let Err(e) = r {
                    crate::log_debug!("connection ended error={e}");
                }
                return;
            }
            // Dropping the connection drops its in-flight handlers, whose guards kill their groups.
            () = stop.cancelled() => return,
            // Idle: HTTP/1.1 closes once nothing is in flight, h2 sends GOAWAY and closes when its
            // last stream ends. The connection then completes on its own.
            () = &mut idle, if !closing => {
                closing = true;
                conn.as_mut().graceful_shutdown();
            }
        }
    }
}

/// The one unauthenticated answer's paths, read with GET or HEAD: `/livez` and its common name.
fn is_health<B>(req: &HttpRequest<B>) -> bool {
    matches!(req.uri().path(), "/livez" | "/healthz")
        && matches!(*req.method(), Method::GET | Method::HEAD)
}

/// What a connection has in flight, and when it last was not idle: the idle timeout's whole state.
struct ConnActivity {
    in_flight: AtomicUsize,
    /// Set when the count returns to zero; the wait re-checks instead of cancelling a timer.
    last: Mutex<Instant>,
    zero: Notify,
}

impl ConnActivity {
    fn new() -> Self {
        Self {
            in_flight: AtomicUsize::new(0),
            last: Mutex::new(Instant::now()),
            zero: Notify::new(),
        }
    }

    /// Completes once no request has been in flight for `limit`. Sleeps to the deadline the last
    /// activity implies, and again if there was more activity meanwhile: at most one timer at a time,
    /// and none while a request is in flight.
    async fn idle_for(&self, limit: Duration) {
        loop {
            if self.in_flight.load(Ordering::Acquire) > 0 {
                self.zero.notified().await;
                continue;
            }
            let deadline = *self.last.lock().unwrap_or_else(PoisonError::into_inner) + limit;
            if Instant::now() >= deadline {
                return;
            }
            tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
        }
    }
}

/// A request in flight; ends when dropped (the answer fully written, or the connection gone).
struct InFlight(Arc<ConnActivity>, Arc<Ctx>);

impl InFlight {
    /// Counted on the connection (for its idle timeout) and on the server ([`crate::Handle::activity`]),
    /// by the one guard, so the two cannot disagree.
    fn enter(a: &Arc<ConnActivity>, ctx: &Arc<Ctx>) -> Self {
        a.in_flight.fetch_add(1, Ordering::AcqRel);
        ctx.in_flight.fetch_add(1, Ordering::AcqRel);
        ctx.activity_changed();
        Self(a.clone(), ctx.clone())
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        *self.0.last.lock().unwrap_or_else(PoisonError::into_inner) = Instant::now();
        if self.0.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
            // A permit is kept if nobody waits yet, so the wake-up cannot be lost.
            self.0.zero.notify_one();
        }
        self.1.in_flight.fetch_sub(1, Ordering::AcqRel);
        self.1.activity_changed();
    }
}

/// A response body that keeps its request counted as in flight, or the embedder's own, which is not
/// counted at all.
struct Tracked {
    body: Inner,
    _guard: Option<InFlight>,
}

enum Inner {
    Own(Body),
    Claimed(RouteBody),
}

impl Tracked {
    fn own(body: Body, guard: Option<InFlight>) -> Self {
        Self {
            body: Inner::Own(body),
            _guard: guard,
        }
    }

    fn claimed(resp_body: RouteBody) -> Self {
        Self {
            body: Inner::Claimed(resp_body),
            _guard: None,
        }
    }
}

impl HttpBody for Tracked {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, std::io::Error>>> {
        match &mut self.body {
            Inner::Own(b) => Pin::new(b)
                .poll_frame(cx)
                .map(|o| o.map(|r| r.map_err(|e| match e {}))),
            Inner::Claimed(b) => Pin::new(b).poll_frame(cx),
        }
    }

    fn is_end_stream(&self) -> bool {
        match &self.body {
            Inner::Own(b) => b.is_end_stream(),
            Inner::Claimed(b) => b.is_end_stream(),
        }
    }

    fn size_hint(&self) -> SizeHint {
        match &self.body {
            Inner::Own(b) => b.size_hint(),
            Inner::Claimed(b) => b.size_hint(),
        }
    }
}

async fn route(ctx: Arc<Ctx>, req: HttpRequest<Incoming>, checked: bool) -> Resp {
    let (parts, body) = req.into_parts();
    // The one unauthenticated answer, and only to a read: it says the process is up and nothing
    // else. `/healthz` is the same answer under the common name. A POST to either path is an exec request like any other, and is authenticated.
    if matches!(parts.uri.path(), "/livez" | "/healthz")
        && matches!(parts.method, Method::GET | Method::HEAD)
    {
        return json_response(200, &json!({ "ok": true }));
    }
    if parts.method != Method::POST {
        return error_response(405, "POST an exec request");
    }
    if !checked && !authorized(&ctx, &parts.headers) {
        crate::log_info!("request kind=auth status=401");
        return error_response(401, "unauthorized");
    }
    let max = ctx.opts.max_request_bytes;
    let declared = parts
        .headers
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > max) {
        crate::log_info!("request kind=body status=413");
        return error_response(413, "request too large");
    }
    // Read at most `max` bytes, declared or not (chunked bodies declare nothing).
    let bytes = match Limited::new(body, max).collect().await {
        Ok(c) => c.to_bytes(),
        Err(e)
            if e.downcast_ref::<http_body_util::LengthLimitError>()
                .is_some() =>
        {
            crate::log_info!("request kind=body status=413");
            return error_response(413, "request too large");
        }
        Err(_) => return error_response(400, "request body not readable"),
    };
    let req: Request = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            crate::log_info!("request kind=body status=400");
            return error_response(400, &format!("bad request: {e}"));
        }
    };
    if let Some(observer) = &ctx.opts.observer {
        let headers: Vec<(String, String)> = parts
            .headers
            .iter()
            .map(|(n, v)| (n.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        observer.request(&headers, &value);
    }
    drop(bytes);

    if req.job.as_ref().is_some_and(Value::is_object) {
        return crate::jobs::serve(&ctx, &req).await;
    }
    if req.wants_stream() {
        crate::exec::stream(&ctx, req)
    } else {
        crate::exec::envelope(&ctx, req).await
    }
}

/// The parsing a request body goes through once the bearer is accepted, for the fuzz targets and
/// the parser test: the envelope, a `job` object's operation and `notify`, and `stdin`.
/// Whether the body was accepted (the server answers `400` for one that is not). Never panics.
pub(crate) fn fuzz_request(bytes: &[u8]) -> bool {
    use base64::Engine as _;
    let Ok(req) = serde_json::from_slice::<Request>(bytes) else {
        return false;
    };
    if let Some(job) = req.job.as_ref().filter(|j| j.is_object())
        && !crate::jobs::fuzz_job(job)
    {
        return false;
    }
    if let Some(b) = &req.stdin {
        match base64::engine::general_purpose::STANDARD.decode(b) {
            Ok(data) => crate::frame::fuzz(&data),
            Err(_) => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::ct_eq;

    #[test]
    fn equal_only_when_every_byte_and_the_length_match() {
        assert!(ct_eq(b"token", b"token"));
        assert!(ct_eq(b"", b""));
        assert!(!ct_eq(b"token", b"tokeN"));
        assert!(!ct_eq(b"token", b"toke"));
        assert!(!ct_eq(b"toke", b"token"));
        assert!(!ct_eq(b"token\0", b"token"));
        assert!(!ct_eq(b"", b"x"));
    }
}
