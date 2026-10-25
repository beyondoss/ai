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
use std::sync::Arc;
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
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::Ctx;

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

/// A v1.2 stream's response head, and the sender that feeds its body. The channel is bounded, so a
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
    pub stdin_base64: Option<String>,
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
        tracing::info!(
            kind = self.kind,
            command = %self.command,
            status,
            outcome,
            ms = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            "request"
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

fn authorized(ctx: &Ctx, headers: &hyper::HeaderMap) -> bool {
    let Some(want) = &ctx.opts.bearer else {
        return true;
    };
    headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|got| ct_eq(got.as_bytes(), want.as_bytes()))
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
                    tracing::warn!(error = %e, "accept failed");
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

async fn connection(sock: tokio::net::TcpStream, ctx: Arc<Ctx>, stop: CancellationToken) {
    let _ = sock.set_nodelay(true);
    // A peer that vanished without a FIN is dropped by the kernel once data it was sent has gone
    // unacknowledged this long: the spec's bound on a vanished client, two heartbeat intervals
    // (v1.2 "Disconnect kills"). The heartbeat keeps data in flight on an idle stream.
    let user_timeout = u32::try_from((ctx.opts.heartbeat * 2).as_millis()).unwrap_or(u32::MAX);
    let _ = nix::sys::socket::setsockopt(
        &sock,
        nix::sys::socket::sockopt::TcpUserTimeout,
        &user_timeout,
    );
    let svc = hyper::service::service_fn(move |req| handle(ctx.clone(), req));
    let mut builder = auto::Builder::new(TokioExecutor::new());
    builder
        .http1()
        .timer(TokioTimer::new())
        .header_read_timeout(HEAD_TIMEOUT);
    let conn = builder.serve_connection(TokioIo::new(sock), svc);
    tokio::select! {
        r = conn => {
            if let Err(e) = r {
                tracing::debug!(error = %e, "connection ended");
            }
        }
        // Dropping the connection drops its in-flight handlers, whose guards kill their groups.
        () = stop.cancelled() => {}
    }
}

async fn handle(ctx: Arc<Ctx>, req: HttpRequest<Incoming>) -> Result<Resp, Infallible> {
    Ok(route(ctx, req).await)
}

async fn route(ctx: Arc<Ctx>, req: HttpRequest<Incoming>) -> Resp {
    let (parts, body) = req.into_parts();
    if parts.uri.path() == "/livez" {
        return json_response(200, &json!({ "ok": true }));
    }
    if parts.method != Method::POST {
        return error_response(405, "POST an exec request");
    }
    if !authorized(&ctx, &parts.headers) {
        tracing::info!(kind = "auth", status = 401, "request");
        return error_response(401, "unauthorized");
    }
    let max = ctx.opts.max_request_bytes;
    let declared = parts
        .headers
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > max) {
        tracing::info!(kind = "body", status = 413, "request");
        return error_response(413, "request too large");
    }
    // Read at most `max` bytes, declared or not (chunked bodies declare nothing).
    let bytes = match Limited::new(body, max).collect().await {
        Ok(c) => c.to_bytes(),
        Err(e)
            if e.downcast_ref::<http_body_util::LengthLimitError>()
                .is_some() =>
        {
            tracing::info!(kind = "body", status = 413, "request");
            return error_response(413, "request too large");
        }
        Err(_) => return error_response(400, "request body not readable"),
    };
    let req: Request = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            tracing::info!(kind = "body", status = 400, "request");
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

    let job = req.job.as_ref().and_then(Value::as_object);
    if let Some(job) = job {
        if ctx.opts.native_jobs {
            return crate::jobs::serve(&ctx, &req).await;
        }
        // Without native jobs the field is unknown and the command runs, except that an endpoint
        // offering notify for the baseline answers the probe (v1.4, N1).
        if job.get("op").and_then(Value::as_str) == Some("probe") && ctx.opts.advertises_notify() {
            return json_response(200, &json!({ "notify": crate::NOTIFY_VERSION }));
        }
    }
    if req.wants_stream() {
        crate::exec::stream(&ctx, req)
    } else {
        crate::exec::envelope(&ctx, req).await
    }
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
