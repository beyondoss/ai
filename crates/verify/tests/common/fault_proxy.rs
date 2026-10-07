//! The FLT-1 fault proxy: an HTTP/1.1 man-in-the-middle between the gateway and one real provider.
//!
//! The gateway is pointed at it with `provider_authorities.<provider> = "127.0.0.1:<port>"` and
//! `upstream_verify_cert = false`, so it still dials TLS (with its own ALPN, connection pool and
//! error classification) but to a throwaway self-signed cert that only offers `http/1.1`. The
//! proxy terminates that TLS, reads each request in full, and per its [`Script`] either answers
//! itself (a synthetic, provider-shaped `5xx` / `429`), breaks the connection at a chosen point,
//! or re-originates TLS to the real provider (`api.anthropic.com`, ...) and relays the response,
//! optionally slowed, stalled or cut after N server-sent events.
//!
//! Each request becomes a [`Record`]: whether it was forwarded (the provider processed it), the
//! provider's status, how much of the response reached the gateway, and the usage the provider
//! reported. A response the gateway was cut off from is still drained from the provider to its
//! end, so the record holds what the provider billed, not what the gateway happened to see.
//!
//! The upstream leg asks for `accept-encoding: identity` and `connection: close`, so the proxy can
//! read usage from the body and needs no upstream pool. H2 is out of scope: the gateway falls back
//! to HTTP/1.1 by ALPN, exactly as it does for any provider that doesn't offer h2.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _};
use tokio::net::TcpStream;
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// What the proxy does with one request (or, for [`Fault::ResetOnConnect`], one connection).
#[derive(Clone, Debug)]
pub enum Fault {
    /// Relay the provider's response untouched.
    Pass,
    /// Answer with this status and a provider-shaped error body; never forwarded.
    Status {
        status: u16,
        retry_after: Option<u32>,
    },
    /// RST the TCP connection as soon as it is accepted, before the TLS handshake.
    ResetOnConnect,
    /// Read the whole request, then RST; never forwarded (the provider saw nothing).
    ResetBeforeForward,
    /// Forward the request; when the provider's response head arrives, RST the gateway's
    /// connection. The provider processed (and billed) it; the gateway saw no response.
    ResetAfterForward,
    /// Relay the head and the first `events` SSE events, then RST.
    ResetMidStream { events: usize },
    /// Relay every SSE event, each delayed by `per_event`.
    SlowDrip { per_event: Duration },
    /// Forward the request (the provider processes and bills it), hold its whole response back
    /// for `hold`, then close: a provider still working on the request when the gateway's read
    /// timeout expires, the connection up.
    StallBeforeHead { hold: Duration },
    /// Relay the head and the first `events` SSE events, then go silent for `hold`, then close.
    StallMidStream { events: usize, hold: Duration },
}

/// The provider whose error shape a synthetic response imitates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Anthropic,
    OpenAi,
    OpenRouter,
}

/// Faults applied in arrival order: `first[0]` to the first request, and so on; `then` to every
/// request after them.
#[derive(Clone, Debug)]
pub struct Script {
    pub first: Vec<Fault>,
    pub then: Fault,
}

impl Script {
    pub fn pass() -> Self {
        Self {
            first: Vec::new(),
            then: Fault::Pass,
        }
    }
    pub fn once(f: Fault) -> Self {
        Self {
            first: vec![f],
            then: Fault::Pass,
        }
    }
    pub fn always(f: Fault) -> Self {
        Self {
            first: Vec::new(),
            then: f,
        }
    }
}

/// Usage as the provider reported it: `input_total` includes cached tokens on every wire.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_total: u64,
    pub output: u64,
    pub cache_read: u64,
}

/// How much of the provider's response reached the gateway.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Delivery {
    #[default]
    Nothing,
    /// The head only.
    Head,
    /// The head and part of the body.
    Partial,
    /// All of it.
    Full,
}

/// One request (or one reset connection) the proxy saw.
#[derive(Clone, Debug, Default)]
pub struct Record {
    pub fault: String,
    pub path: String,
    /// The request body asked for a stream.
    pub stream: bool,
    /// Sent to the real provider in full: the provider processed it.
    pub forwarded: bool,
    /// The status the real provider (or the synthetic answer) gave.
    pub status: Option<u16>,
    /// What the provider reported, read from the whole response (drained even after a cut).
    pub usage: Option<Usage>,
    pub delivered: Delivery,
    pub events_upstream: usize,
    pub events_sent: usize,
    /// The handler is finished with this record (the provider's response fully drained).
    pub done: bool,
    /// Milliseconds after the proxy started.
    pub at_ms: u64,
    /// A proxy-side failure (the real provider unreachable, ...), never a scripted fault.
    pub error: Option<String>,
    /// Milliseconds from the request's arrival to its response head reaching the gateway's socket
    /// (dial, upstream TLS and the provider's own time to head included).
    pub head_ms: Option<u64>,
    /// Milliseconds from the request's arrival to the first body byte reaching the gateway.
    pub first_byte_ms: Option<u64>,
    /// The provider answered and the proxy held the answer back (`StallBeforeHead`): the gateway
    /// had the request delivered and waited on a live connection.
    pub withheld: bool,
}

struct State {
    upstream_host: String,
    shape: Shape,
    script: Mutex<(VecDeque<Fault>, Fault)>,
    records: Mutex<Vec<Record>>,
    start: Instant,
    acceptor: TlsAcceptor,
    connector: TlsConnector,
}

pub struct FaultProxy {
    pub port: u16,
    state: Arc<State>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for FaultProxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl FaultProxy {
    /// Listen on a free loopback port, relaying to `upstream_host:443`.
    pub async fn start(upstream_host: &str, shape: Shape, script: Script) -> io::Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let ck = rcgen::generate_simple_self_signed(vec!["127.0.0.1".into(), "localhost".into()])
            .map_err(io::Error::other)?;
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(ck.key_pair.serialize_der().into());
        let mut server = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![ck.cert.der().clone()], key)
            .map_err(io::Error::other)?;
        server.alpn_protocols = vec![b"http/1.1".to_vec()];
        let roots =
            rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let mut client = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        client.alpn_protocols = vec![b"http/1.1".to_vec()];

        let listener = beyond_ai_test_support::ports::tokio_listener().await;
        let port = listener.local_addr()?.port();
        let state = Arc::new(State {
            upstream_host: upstream_host.to_owned(),
            shape,
            script: Mutex::new((script.first.into(), script.then)),
            records: Mutex::new(Vec::new()),
            start: Instant::now(),
            acceptor: TlsAcceptor::from(Arc::new(server)),
            connector: TlsConnector::from(Arc::new(client)),
        });
        let st = state.clone();
        let task = tokio::spawn(async move {
            while let Ok((tcp, _)) = listener.accept().await {
                let st = st.clone();
                tokio::spawn(async move { st.connection(tcp).await });
            }
        });
        Ok(Self { port, state, task })
    }

    pub fn records(&self) -> Vec<Record> {
        self.state.records.lock().unwrap().clone()
    }

    /// Wait (up to `timeout`) until every record's handler is done, then return them.
    pub async fn settle(&self, timeout: Duration) -> Vec<Record> {
        let deadline = Instant::now() + timeout;
        loop {
            let recs = self.records();
            if recs.iter().all(|r| r.done) || Instant::now() >= deadline {
                return recs;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// RST rather than FIN: zero linger, then drop.
fn rst(tcp: &TcpStream) {
    let _ = socket2::SockRef::from(tcp).set_linger(Some(Duration::ZERO));
}

impl State {
    fn peek_fault(&self) -> Fault {
        let s = self.script.lock().unwrap();
        s.0.front().cloned().unwrap_or_else(|| s.1.clone())
    }

    fn next_fault(&self) -> Fault {
        let mut s = self.script.lock().unwrap();
        match s.0.pop_front() {
            Some(f) => f,
            None => s.1.clone(),
        }
    }

    fn push(&self, mut rec: Record) -> usize {
        rec.at_ms = self.start.elapsed().as_millis() as u64;
        let mut r = self.records.lock().unwrap();
        r.push(rec);
        r.len() - 1
    }

    fn update(&self, idx: usize, f: impl FnOnce(&mut Record)) {
        f(&mut self.records.lock().unwrap()[idx]);
    }

    async fn connection(self: Arc<Self>, tcp: TcpStream) {
        let _ = tcp.set_nodelay(true);
        if matches!(self.peek_fault(), Fault::ResetOnConnect) {
            let f = self.next_fault();
            self.push(Record {
                fault: format!("{f:?}"),
                done: true,
                ..Record::default()
            });
            rst(&tcp);
            return;
        }
        let Ok(tls) = self.acceptor.accept(tcp).await else {
            return;
        };
        let mut down = Reader::new(tls);
        loop {
            let Ok(Some(head)) = down.head().await else {
                return;
            };
            let Ok(mut req) = parse_request(&head) else {
                return;
            };
            let mut body = Vec::new();
            let mut framing = std::mem::replace(&mut req.framing, Framing::Done);
            loop {
                match framing.next(&mut down).await {
                    Ok(Some(p)) => body.extend_from_slice(&p),
                    Ok(None) => break,
                    Err(_) => return,
                }
            }
            let stream = serde_json::from_slice::<Value>(&body)
                .is_ok_and(|v| v["stream"] == Value::Bool(true));
            let fault = self.next_fault();
            let idx = self.push(Record {
                fault: format!("{fault:?}"),
                path: req.path.clone(),
                stream,
                ..Record::default()
            });
            match fault {
                Fault::Status {
                    status,
                    retry_after,
                } => {
                    let resp = synthetic(self.shape, status, retry_after, idx);
                    let ok = down.s.write_all(&resp).await.is_ok() && down.s.flush().await.is_ok();
                    self.update(idx, |r| {
                        r.status = Some(status);
                        r.done = true;
                    });
                    if !ok {
                        return;
                    }
                }
                Fault::ResetOnConnect | Fault::ResetBeforeForward => {
                    self.update(idx, |r| r.done = true);
                    rst(down.s.get_ref().0);
                    return;
                }
                fault => {
                    let keep = self.forward(idx, &req, &body, fault, &mut down).await;
                    self.update(idx, |r| r.done = true);
                    if !keep {
                        return;
                    }
                }
            }
        }
    }

    /// Relay one request to the real provider. Returns whether the gateway's connection is still
    /// usable for the next request.
    async fn forward(
        &self,
        idx: usize,
        req: &Request,
        body: &[u8],
        fault: Fault,
        down: &mut Reader<tokio_rustls::server::TlsStream<TcpStream>>,
    ) -> bool {
        let fail = |down: &Reader<tokio_rustls::server::TlsStream<TcpStream>>, e: String| {
            self.update(idx, |r| r.error = Some(e));
            rst(down.s.get_ref().0);
            false
        };
        let t0 = Instant::now();
        let up = match self.dial().await {
            Ok(up) => up,
            Err(e) => return fail(down, format!("dial {}: {e}", self.upstream_host)),
        };
        let mut up = Reader::new(up);
        let mut out = format!("{} {} HTTP/1.1\r\n", req.method, req.path).into_bytes();
        for (k, v) in &req.headers {
            if matches!(
                k.to_ascii_lowercase().as_str(),
                "host"
                    | "connection"
                    | "keep-alive"
                    | "proxy-connection"
                    | "te"
                    | "upgrade"
                    | "content-length"
                    | "transfer-encoding"
                    | "accept-encoding"
            ) {
                continue;
            }
            out.extend_from_slice(k.as_bytes());
            out.extend_from_slice(b": ");
            out.extend_from_slice(v);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(
            format!(
                "host: {}\r\nconnection: close\r\naccept-encoding: identity\r\ncontent-length: {}\r\n\r\n",
                self.upstream_host,
                body.len()
            )
            .as_bytes(),
        );
        out.extend_from_slice(body);
        if let Err(e) = up.s.write_all(&out).await {
            return fail(down, format!("upstream write: {e}"));
        }
        let _ = up.s.flush().await;
        self.update(idx, |r| r.forwarded = true);

        let head = match up.head().await {
            Ok(Some(h)) => h,
            Ok(None) => return fail(down, "upstream closed before its head".into()),
            Err(e) => return fail(down, format!("upstream head: {e}")),
        };
        let Ok(resp) = parse_response(&head, &req.method) else {
            return fail(down, "unparseable upstream head".into());
        };
        let sse = resp.content_type.contains("event-stream");
        self.update(idx, |r| r.status = Some(resp.status));
        let mut framing = resp.framing;
        let mut full = Vec::new();
        let mut alive = true;

        if matches!(fault, Fault::ResetAfterForward) {
            rst(down.s.get_ref().0);
            drain(&mut up, &mut framing, &mut full).await;
            self.finish(idx, &full, sse, None);
            return false;
        }
        if let Fault::StallBeforeHead { hold } = fault {
            self.update(idx, |r| r.withheld = true);
            drain(&mut up, &mut framing, &mut full).await;
            self.finish(idx, &full, sse, None);
            tokio::time::sleep(hold.saturating_sub(t0.elapsed())).await;
            return false;
        }

        // The head to the gateway: the provider's, re-framed (a length stays a length; anything
        // else becomes chunked, since the upstream leg is close-delimited or chunked).
        let chunked = !matches!(framing, Framing::Length(_) | Framing::Empty);
        let mut h = format!("HTTP/1.1 {} {}\r\n", resp.status, resp.reason).into_bytes();
        for (k, v) in &resp.headers {
            if matches!(
                k.to_ascii_lowercase().as_str(),
                "connection" | "keep-alive" | "transfer-encoding"
            ) || (chunked && k.eq_ignore_ascii_case("content-length"))
            {
                continue;
            }
            h.extend_from_slice(k.as_bytes());
            h.extend_from_slice(b": ");
            h.extend_from_slice(v);
            h.extend_from_slice(b"\r\n");
        }
        if chunked {
            h.extend_from_slice(b"transfer-encoding: chunked\r\n");
        }
        h.extend_from_slice(b"\r\n");
        if down.s.write_all(&h).await.is_err() || down.s.flush().await.is_err() {
            alive = false;
        } else {
            let ms = t0.elapsed().as_millis() as u64;
            self.update(idx, |r| {
                r.delivered = Delivery::Head;
                r.head_ms = Some(ms);
            });
        }

        let mut pending = Vec::new();
        let mut sent = 0usize;
        let mut seen = 0usize;
        let mut stalled_at: Option<Instant> = None;
        let mut cut = false;
        loop {
            let piece = match framing.next(&mut up).await {
                Ok(Some(p)) => p,
                Ok(None) => break,
                Err(_) => break,
            };
            full.extend_from_slice(&piece);
            if !sse {
                if alive && !cut {
                    alive = send(&mut down.s, chunked, &piece).await;
                }
                continue;
            }
            pending.extend_from_slice(&piece);
            while let Some(end) = event_end(&pending) {
                let event: Vec<u8> = pending.drain(..end).collect();
                seen += 1;
                if !alive || cut || stalled_at.is_some() {
                    continue;
                }
                match fault {
                    Fault::ResetMidStream { events } if sent >= events => {
                        rst(down.s.get_ref().0);
                        cut = true;
                        continue;
                    }
                    Fault::StallMidStream { events, .. } if sent >= events => {
                        stalled_at = Some(Instant::now());
                        continue;
                    }
                    Fault::SlowDrip { per_event } => tokio::time::sleep(per_event).await,
                    _ => {}
                }
                alive = send(&mut down.s, chunked, &event).await;
                if alive {
                    sent += 1;
                }
            }
            let ms = t0.elapsed().as_millis() as u64;
            self.update(idx, |r| {
                r.events_upstream = seen;
                r.events_sent = sent;
                if (sent > 0 || (!sse && alive && !cut)) && r.first_byte_ms.is_none() {
                    r.first_byte_ms = Some(ms);
                }
                if r.first_byte_ms.is_some() && r.delivered < Delivery::Partial {
                    r.delivered = Delivery::Partial;
                }
            });
        }
        let whole = alive && !cut && stalled_at.is_none();
        if whole {
            if !pending.is_empty() {
                alive = send(&mut down.s, chunked, &pending).await;
            }
            if alive && chunked {
                alive = down.s.write_all(b"0\r\n\r\n").await.is_ok();
            }
            alive = alive && down.s.flush().await.is_ok();
        }
        let head_ok = self.records.lock().unwrap()[idx].head_ms.is_some();
        let some_body = self.records.lock().unwrap()[idx].first_byte_ms.is_some();
        let delivered = if !head_ok {
            Delivery::Nothing
        } else if whole && alive {
            Delivery::Full
        } else if some_body {
            Delivery::Partial
        } else {
            Delivery::Head
        };
        self.update(idx, |r| {
            r.events_upstream = seen;
            r.events_sent = sent;
        });
        self.finish(idx, &full, sse, Some(delivered));
        if let (Fault::StallMidStream { hold, .. }, Some(at)) = (&fault, stalled_at) {
            tokio::time::sleep(hold.saturating_sub(at.elapsed())).await;
            return false;
        }
        whole && alive
    }

    fn finish(&self, idx: usize, full: &[u8], sse: bool, delivered: Option<Delivery>) {
        let usage = usage_of(full, sse);
        self.update(idx, |r| {
            r.usage = usage;
            if let Some(d) = delivered {
                r.delivered = d;
            }
        });
    }

    async fn dial(&self) -> io::Result<tokio_rustls::client::TlsStream<TcpStream>> {
        let tcp = TcpStream::connect((self.upstream_host.as_str(), 443)).await?;
        let _ = tcp.set_nodelay(true);
        let name = rustls::pki_types::ServerName::try_from(self.upstream_host.clone())
            .map_err(io::Error::other)?;
        self.connector.connect(name, tcp).await
    }
}

async fn send<W: AsyncWrite + Unpin>(w: &mut W, chunked: bool, data: &[u8]) -> bool {
    let r = if chunked {
        let mut buf = format!("{:x}\r\n", data.len()).into_bytes();
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\r\n");
        w.write_all(&buf).await
    } else {
        w.write_all(data).await
    };
    r.is_ok() && w.flush().await.is_ok()
}

async fn drain<S: AsyncRead + Unpin>(
    up: &mut Reader<S>,
    framing: &mut Framing,
    full: &mut Vec<u8>,
) {
    while let Ok(Some(p)) = framing.next(up).await {
        full.extend_from_slice(&p);
    }
}

/// The end (exclusive) of the first complete SSE event in `buf`.
fn event_end(buf: &[u8]) -> Option<usize> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|i| i + 2);
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// The provider's usage from a whole response body (JSON, or SSE `data:` lines). Counts are
/// cumulative on every wire, so each field keeps its largest value.
pub fn usage_of(body: &[u8], sse: bool) -> Option<Usage> {
    let text = String::from_utf8_lossy(body);
    let values: Vec<Value> = if sse {
        text.lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .filter_map(|d| serde_json::from_str(d.trim()).ok())
            .collect()
    } else {
        serde_json::from_str(&text).into_iter().collect()
    };
    let mut out: Option<Usage> = None;
    let n = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64);
    for v in &values {
        for u in [
            v.get("usage"),
            v.pointer("/message/usage"),
            v.pointer("/response/usage"),
        ]
        .into_iter()
        .flatten()
        .filter(|u| u.is_object())
        {
            let o = out.get_or_insert_with(Usage::default);
            if let Some(p) = n(u, "prompt_tokens") {
                o.input_total = o.input_total.max(p);
                let cached = u
                    .pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                o.cache_read = o.cache_read.max(cached);
            }
            if let Some(c) = n(u, "completion_tokens") {
                o.output = o.output.max(c);
            }
            if let Some(i) = n(u, "input_tokens") {
                let cr = n(u, "cache_read_input_tokens").unwrap_or(0);
                let cw = n(u, "cache_creation_input_tokens").unwrap_or(0);
                o.input_total = o.input_total.max(i + cr + cw);
                o.cache_read = o.cache_read.max(cr);
            }
            if let Some(c) = n(u, "output_tokens") {
                o.output = o.output.max(c);
            }
        }
    }
    out
}

/// A provider-shaped error response: the body, request-id header and retry hints each provider
/// actually sends for that status.
fn synthetic(shape: Shape, status: u16, retry_after: Option<u32>, seq: usize) -> Vec<u8> {
    let reason = match status {
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        529 => "Site Overloaded",
        _ => "Error",
    };
    let id = format!("req_fault_{seq}");
    let (body, mut headers) = match shape {
        Shape::Anthropic => {
            let kind = match status {
                429 => "rate_limit_error",
                529 => "overloaded_error",
                _ => "api_error",
            };
            (
                serde_json::json!({"type": "error", "error": {"type": kind, "message": format!("injected {status}")}, "request_id": id}),
                vec![
                    ("request-id", id.clone()),
                    ("x-should-retry", "true".into()),
                ],
            )
        }
        Shape::OpenAi => {
            let (kind, code) = match status {
                429 => ("requests", Value::from("rate_limit_exceeded")),
                _ => ("server_error", Value::Null),
            };
            (
                serde_json::json!({"error": {"message": format!("injected {status}"), "type": kind, "param": null, "code": code}}),
                vec![("x-request-id", id.clone())],
            )
        }
        Shape::OpenRouter => (
            serde_json::json!({"error": {"code": status, "message": format!("injected {status}")}}),
            vec![],
        ),
    };
    if let Some(s) = retry_after {
        headers.push(("retry-after", s.to_string()));
    }
    let body = body.to_string();
    let mut out = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    out.push_str(&body);
    out.into_bytes()
}

/// A buffered reader over one side of the proxy.
struct Reader<S> {
    s: S,
    buf: Vec<u8>,
    eof: bool,
}

impl<S: AsyncRead + Unpin> Reader<S> {
    fn new(s: S) -> Self {
        Self {
            s,
            buf: Vec::new(),
            eof: false,
        }
    }

    /// Read more bytes; 0 at EOF (a TLS peer that closes without close_notify counts as EOF).
    async fn fill(&mut self) -> io::Result<usize> {
        if self.eof {
            return Ok(0);
        }
        let mut tmp = [0u8; 16384];
        let n = match self.s.read(&mut tmp).await {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => 0,
            Err(e) => return Err(e),
        };
        if n == 0 {
            self.eof = true;
        }
        self.buf.extend_from_slice(&tmp[..n]);
        Ok(n)
    }

    /// A message head through its blank line, or `None` at a clean EOF between messages.
    async fn head(&mut self) -> io::Result<Option<Vec<u8>>> {
        loop {
            if let Some(i) = self.buf.windows(4).position(|w| w == b"\r\n\r\n") {
                return Ok(Some(self.buf.drain(..i + 4).collect()));
            }
            if self.buf.len() > 64 * 1024 {
                return Err(io::Error::other("head too large"));
            }
            if self.fill().await? == 0 {
                return if self.buf.is_empty() {
                    Ok(None)
                } else {
                    Err(io::ErrorKind::UnexpectedEof.into())
                };
            }
        }
    }

    async fn line(&mut self) -> io::Result<Vec<u8>> {
        loop {
            if let Some(i) = self.buf.windows(2).position(|w| w == b"\r\n") {
                let l: Vec<u8> = self.buf.drain(..i + 2).collect();
                return Ok(l[..i].to_vec());
            }
            if self.fill().await? == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
        }
    }

    /// Up to `max` bytes of what is buffered (reading more if nothing is).
    async fn some(&mut self, max: u64) -> io::Result<Vec<u8>> {
        if self.buf.is_empty() && self.fill().await? == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        let n = (self.buf.len() as u64).min(max) as usize;
        Ok(self.buf.drain(..n).collect())
    }
}

/// How a message body is delimited, and how much of it is left.
#[derive(Debug)]
enum Framing {
    Empty,
    Length(u64),
    /// Bytes left in the current chunk; `None` before a chunk-size line.
    Chunked(Option<u64>),
    Close,
    Done,
}

impl Framing {
    /// The next piece of decoded body, or `None` at its end.
    async fn next<S: AsyncRead + Unpin>(
        &mut self,
        r: &mut Reader<S>,
    ) -> io::Result<Option<Vec<u8>>> {
        loop {
            match self {
                Self::Empty | Self::Done | Self::Length(0) => {
                    *self = Self::Done;
                    return Ok(None);
                }
                Self::Length(left) => {
                    let p = r.some(*left).await?;
                    *left -= p.len() as u64;
                    return Ok(Some(p));
                }
                Self::Close => {
                    if r.buf.is_empty() && r.fill().await? == 0 {
                        *self = Self::Done;
                        return Ok(None);
                    }
                    return Ok(Some(std::mem::take(&mut r.buf)));
                }
                Self::Chunked(None) => {
                    let line = r.line().await?;
                    let size = String::from_utf8_lossy(&line);
                    let size = size.split(';').next().unwrap_or("").trim();
                    let n = u64::from_str_radix(size, 16)
                        .map_err(|_| io::Error::other(format!("bad chunk size {size:?}")))?;
                    if n == 0 {
                        while !r.line().await?.is_empty() {}
                        *self = Self::Done;
                        return Ok(None);
                    }
                    *self = Self::Chunked(Some(n));
                }
                Self::Chunked(Some(left)) => {
                    let p = r.some(*left).await?;
                    *left -= p.len() as u64;
                    if *left == 0 {
                        r.line().await?;
                        *self = Self::Chunked(None);
                    }
                    return Ok(Some(p));
                }
            }
        }
    }
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, Vec<u8>)>,
    framing: Framing,
}

struct Response {
    status: u16,
    reason: String,
    headers: Vec<(String, Vec<u8>)>,
    content_type: String,
    framing: Framing,
}

fn header_framing(headers: &[(String, Vec<u8>)]) -> Option<Framing> {
    let get = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| String::from_utf8_lossy(v).to_ascii_lowercase())
    };
    if get("transfer-encoding").is_some_and(|v| v.contains("chunked")) {
        return Some(Framing::Chunked(None));
    }
    get("content-length")
        .and_then(|v| v.trim().parse().ok())
        .map(Framing::Length)
}

fn parse_request(head: &[u8]) -> Result<Request, String> {
    let mut hs = [httparse::EMPTY_HEADER; 96];
    let mut req = httparse::Request::new(&mut hs);
    req.parse(head).map_err(|e| e.to_string())?;
    let headers: Vec<(String, Vec<u8>)> = req
        .headers
        .iter()
        .map(|h| (h.name.to_owned(), h.value.to_vec()))
        .collect();
    let framing = header_framing(&headers).unwrap_or(Framing::Empty);
    Ok(Request {
        method: req.method.unwrap_or("GET").to_owned(),
        path: req.path.unwrap_or("/").to_owned(),
        headers,
        framing,
    })
}

fn parse_response(head: &[u8], method: &str) -> Result<Response, String> {
    let mut hs = [httparse::EMPTY_HEADER; 128];
    let mut resp = httparse::Response::new(&mut hs);
    resp.parse(head).map_err(|e| e.to_string())?;
    let status = resp.code.unwrap_or(0);
    let headers: Vec<(String, Vec<u8>)> = resp
        .headers
        .iter()
        .map(|h| (h.name.to_owned(), h.value.to_vec()))
        .collect();
    let content_type = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        .map(|(_, v)| String::from_utf8_lossy(v).to_ascii_lowercase())
        .unwrap_or_default();
    let framing = if method == "HEAD" || status == 204 || status == 304 || status < 200 {
        Framing::Empty
    } else {
        header_framing(&headers).unwrap_or(Framing::Close)
    };
    Ok(Response {
        status,
        reason: resp.reason.unwrap_or("").to_owned(),
        headers,
        content_type,
        framing,
    })
}
