//! The MCP Events wire: JSON-RPC errors, the per-connection notification router, the connection
//! abstraction (rmcp's own peer for stdio and session-bound HTTP, direct stateless HTTP otherwise),
//! the bounded SSE reader, push-stream handles, and RFC 3339 parsing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rmcp::RoleClient;
use rmcp::model::{
    CancelledNotification, CancelledNotificationParam, ClientRequest, CustomNotification,
    CustomRequest, RequestId,
};
use rmcp::service::{Peer, PeerRequestOptions, ServiceError};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::env_ms;

/// Most bytes one SSE event (one JSON-RPC message on a push stream) may be. The draft keeps
/// delivery bodies at or under 256 KiB; anything far past that is a broken or hostile server.
pub(super) const MAX_SSE_EVENT_BYTES: usize = 1024 * 1024;
/// Most bytes a unary `events/*` response body may be.
pub(super) const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
/// Most bytes a server's webhook JWKS document may be.
pub(super) const MAX_JWKS_BYTES: usize = 64 * 1024;

/// How many notifications one push stream may have queued before its consumer catches up. Past it
/// the stream is marked overflowed and reconnected from the last cursor it delivered.
fn stream_buffer() -> usize {
    (env_ms("BEYOND_AI_AGENT_MCP_EVENTS_STREAM_BUFFER", 256).as_millis() as usize).max(1)
}

// ---------------------------------------------------------------------------------------------
// Notification routing (one per MCP connection)
// ---------------------------------------------------------------------------------------------

/// One `notifications/events/*` message for an open push stream.
#[derive(Debug)]
pub(crate) struct StreamMsg {
    pub(super) method: String,
    pub(super) params: Value,
}

struct Slot {
    tx: mpsc::Sender<StreamMsg>,
    overflowed: Arc<AtomicBool>,
}

#[derive(Default)]
struct RouterInner {
    streams: HashMap<RequestId, Slot>,
}

/// Routes a connection's `notifications/events/*` to the push stream that asked for them, keyed by
/// the `events/stream` request id the server echoes in `_meta`. Lives on the connection's handler;
/// costs a lock only when an events notification actually arrives.
///
/// **Never silently lossy.** A stream whose consumer falls [`stream_buffer`] messages behind is
/// marked overflowed and *unregistered*, so nothing after the first dropped notification is
/// forwarded either. Everything already queued precedes the drop, so once the consumer has drained
/// it, its cursor is the last position it really delivered — it reconnects from there and the
/// server replays the rest. A cursor can never move past an event that was dropped.
#[derive(Clone, Default)]
pub struct NotificationRouter {
    inner: Arc<Mutex<RouterInner>>,
    /// Bumped by `notifications/events/list_changed`; discovery caches compare against it.
    generation: Arc<AtomicU64>,
}

impl NotificationRouter {
    /// Called by the connection's `on_custom_notification`. Anything not `notifications/events/*`
    /// is ignored, as before. `subscription_id` is `_meta["io.modelcontextprotocol/subscriptionId"]`,
    /// which rmcp hands to the handler on the notification context.
    pub(crate) fn route(
        &self,
        notification: CustomNotification,
        subscription_id: Option<RequestId>,
    ) {
        let method = notification.method.as_str();
        if !method.starts_with("notifications/events/") {
            return;
        }
        if method == "notifications/events/list_changed" {
            self.generation.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Some(id) = subscription_id else {
            tracing::debug!(
                method,
                "events notification without a subscriptionId; dropped"
            );
            return;
        };
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let Some(slot) = inner.streams.get(&id) else {
            return;
        };
        let msg = StreamMsg {
            method: method.to_owned(),
            params: notification.params.unwrap_or(Value::Null),
        };
        if slot.tx.try_send(msg).is_err() {
            slot.overflowed.store(true, Ordering::Release);
            inner.streams.remove(&id);
        }
    }

    fn register(&self, id: RequestId, tx: mpsc::Sender<StreamMsg>, overflowed: Arc<AtomicBool>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.streams.insert(id, Slot { tx, overflowed });
        }
    }

    fn unregister(&self, id: &RequestId) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.streams.remove(id);
        }
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------------------------
// JSON-RPC errors
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) struct RpcError {
    pub(super) code: Option<i32>,
    pub(super) message: String,
    pub(super) data: Option<Value>,
}

impl RpcError {
    pub(super) fn local(message: impl Into<String>) -> Self {
        Self {
            code: None,
            message: message.into(),
            data: None,
        }
    }

    fn from_service(e: ServiceError) -> Self {
        match e {
            ServiceError::McpError(data) => Self {
                code: Some(data.code.0),
                message: data.message.into_owned(),
                data: data.data,
            },
            other => Self::local(other.to_string()),
        }
    }

    /// Errors that end a subscription rather than being retried: the request is wrong, the event or
    /// subscription is gone, access is refused, or the mode is unsupported — and Method not found,
    /// which means the server does not speak the extension at all.
    pub(super) fn is_terminal(&self) -> bool {
        matches!(
            self.code,
            Some(-32601 | -32602 | -32011 | -32012 | -32014 | -32015)
        )
    }

    /// The draft's "re-discover and resubscribe" signals: the event type was removed
    /// (`-32011`, `data.kind: "event"`) or its schema changed in place (`-32014`,
    /// `data.reason: "schema_changed"`) — not an authorization failure.
    pub(super) fn wants_rediscovery(&self) -> bool {
        let data = self.data.as_ref();
        match self.code {
            Some(-32011) => data.and_then(|d| d.get("kind")) == Some(&json!("event")),
            Some(-32014) => data.and_then(|d| d.get("reason")) == Some(&json!("schema_changed")),
            _ => false,
        }
    }

    pub(super) fn from_json(v: &Value) -> Self {
        Self {
            code: v.get("code").and_then(Value::as_i64).map(|c| c as i32),
            message: v
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("error")
                .to_owned(),
            data: v.get("data").cloned().filter(|d| !d.is_null()),
        }
    }

    pub(super) fn to_json(&self) -> Value {
        json!({ "code": self.code, "message": self.message, "data": self.data })
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code {
            Some(code) => write!(f, "{} ({code})", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

fn custom_request(method: &str, params: Value) -> ClientRequest {
    ClientRequest::CustomRequest(CustomRequest::new(method, Some(params)))
}

// ---------------------------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------------------------

/// Why a JWKS fetch did not produce a key set.
#[derive(Debug)]
pub(super) enum KeyFetch {
    /// The server publishes Ed25519 keys (possibly none of a usable kind, then empty).
    Keys(Vec<ed25519_dalek::VerifyingKey>),
    /// `404`/`410`: the server publishes no document.
    NotPublished,
    /// A timeout, a connection error, a `5xx`, an oversized or unparseable body.
    Failed(String),
}

/// One server's connection for events requests, plus the HTTP client direct requests go out on.
pub(super) struct Conn {
    pub(super) peer: crate::tools::mcp::EventsPeer,
    /// `Some` exactly when `peer` is HTTP.
    pub(super) http: Option<reqwest::Client>,
}

static NEXT_HTTP_ID: AtomicU64 = AtomicU64::new(1);

/// The deadline on one JWKS fetch, headers and body together.
fn jwks_timeout() -> Duration {
    Duration::from_millis(
        std::env::var("BEYOND_AI_AGENT_MCP_EVENTS_JWKS_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5_000),
    )
}

/// Read a response body, refusing one past `cap` bytes (never trusting `Content-Length` to size a
/// buffer).
async fn read_capped(mut resp: reqwest::Response, cap: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if out.len() + chunk.len() > cap {
            return Err(format!("response body larger than {cap} bytes"));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

impl Conn {
    /// `notifications/events/list_changed` generation, for the discovery cache.
    pub(super) fn generation(&self) -> u64 {
        match &self.peer {
            crate::tools::mcp::EventsPeer::Rmcp { router, .. }
            | crate::tools::mcp::EventsPeer::Http { router, .. } => router.generation(),
        }
    }

    /// The origin this connection's server-identity keys belong to: the MCP URL's scheme, host and
    /// port. `None` for stdio, which has no origin to publish keys on.
    pub(super) fn origin(&self) -> Option<String> {
        let crate::tools::mcp::EventsPeer::Http { url, .. } = &self.peer else {
            return None;
        };
        let u = url::Url::parse(url).ok()?;
        Some(u.origin().ascii_serialization())
    }

    /// A JSON-RPC request body for a direct HTTP call, with the per-request `_meta` a stateless
    /// `2026-07-28` server expects.
    fn http_body(id: u64, method: &str, mut params: Value, protocol_version: &str) -> Value {
        if let Value::Object(p) = &mut params {
            p.insert(
                "_meta".into(),
                json!({
                    "io.modelcontextprotocol/protocolVersion": protocol_version,
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "beyond-ai-agent",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "io.modelcontextprotocol/clientCapabilities": {},
                }),
            );
        }
        json!({ "jsonrpc": "2.0", "id": format!("beyond-events-{id}"), "method": method, "params": params })
    }

    fn http_request(
        http: &reqwest::Client,
        url: &str,
        headers: &[(http::HeaderName, http::HeaderValue)],
        protocol_version: &str,
        method: &str,
        body: &Value,
    ) -> reqwest::RequestBuilder {
        let mut req = http
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", protocol_version)
            .header("Mcp-Method", method);
        for (k, v) in headers {
            req = req.header(k, v);
        }
        req.body(body.to_string())
    }

    /// Send one direct `events/*` POST. With an OAuth login it carries the server's **current**
    /// shared token (not the one these headers were built with — a refresh by any connection since
    /// is used at once), and any 401 refreshes it (once for every concurrent caller — see
    /// `mcp_oauth`) and resends **once**; a failed refresh is the error.
    #[allow(clippy::too_many_arguments)]
    async fn send(
        http: &reqwest::Client,
        url: &str,
        headers: &[(http::HeaderName, http::HeaderValue)],
        auth: Option<&crate::tools::mcp_oauth::ServerAuth>,
        protocol_version: &str,
        method: &str,
        body: &Value,
    ) -> Result<reqwest::Response, RpcError> {
        let failed = |e: reqwest::Error| RpcError::local(format!("POST {method}: {e}"));
        let Some(auth) = auth else {
            return Self::http_request(http, url, headers, protocol_version, method, body)
                .send()
                .await
                .map_err(failed);
        };
        let post = |token: Option<&str>| {
            let mut with: Vec<_> = headers
                .iter()
                .filter(|(k, _)| k != http::header::AUTHORIZATION)
                .cloned()
                .collect();
            if let Some(v) =
                token.and_then(|t| http::HeaderValue::from_str(&format!("Bearer {t}")).ok())
            {
                with.push((http::header::AUTHORIZATION, v));
            }
            Self::http_request(http, url, &with, protocol_version, method, body).send()
        };
        let sent = auth.token().await;
        let resp = post(sent.as_deref()).await.map_err(failed)?;
        if resp.status() != reqwest::StatusCode::UNAUTHORIZED {
            return Ok(resp);
        }
        let fresh = auth
            .after_rejection(sent.as_deref())
            .await
            .map_err(RpcError::local)?;
        post(Some(&fresh)).await.map_err(failed)
    }

    /// Fetch the server's webhook-signing keys from `<server origin>/.well-known/mcp-webhook-jwks.json`
    /// — the standalone-JWKS location the pinned draft names (its alternative, SEP-2127 server
    /// cards, is not published yet). Always from the origin the client already dials for MCP, never
    /// from a delivery. stdio servers have no origin: [`KeyFetch::NotPublished`].
    pub(super) async fn server_signing_keys(&self) -> KeyFetch {
        let (crate::tools::mcp::EventsPeer::Http { url, .. }, Some(http)) =
            (&self.peer, self.http.as_ref())
        else {
            return KeyFetch::NotPublished;
        };
        let Some(jwks) = url::Url::parse(url)
            .ok()
            .and_then(|u| u.join("/.well-known/mcp-webhook-jwks.json").ok())
        else {
            return KeyFetch::Failed("unparseable server URL".into());
        };
        // One deadline over the whole exchange — headers *and* body: a server that sends headers
        // and then stalls must not hold a subscribe (or a refresh) open.
        let fetch = async {
            let resp = http.get(jwks).send().await.map_err(|e| e.to_string())?;
            let status = resp.status().as_u16();
            if status != 200 {
                return Ok((status, Vec::new()));
            }
            read_capped(resp, MAX_JWKS_BYTES)
                .await
                .map(|body| (status, body))
        };
        match tokio::time::timeout(jwks_timeout(), fetch).await {
            Err(_) => KeyFetch::Failed("timed out".into()),
            Ok(Err(e)) => KeyFetch::Failed(e),
            Ok(Ok((200, body))) => match serde_json::from_slice::<Value>(&body) {
                Ok(doc) => KeyFetch::Keys(super::webhook::ed25519_keys_from_jwks(&doc)),
                Err(e) => KeyFetch::Failed(format!("unparseable JWKS: {e}")),
            },
            Ok(Ok((404 | 410, _))) => KeyFetch::NotPublished,
            Ok(Ok((other, _))) => KeyFetch::Failed(format!("HTTP {other}")),
        }
    }

    /// One unary `events/*` request.
    pub(super) async fn call(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, RpcError> {
        match &self.peer {
            crate::tools::mcp::EventsPeer::Rmcp { peer, .. } => {
                let tracked = self.peer.track_request();
                let handle = peer
                    .send_request_with_option(
                        custom_request(method, params),
                        PeerRequestOptions::with_timeout(timeout),
                    )
                    .await
                    .map_err(RpcError::from_service)?;
                if let Some(call) = &tracked {
                    call.bind(handle.id.clone());
                }
                let result = handle
                    .await_response()
                    .await
                    .map_err(RpcError::from_service)?;
                // Untagged union: re-serialize whichever variant matched, then undo a rescue.
                serde_json::to_value(result)
                    .map(crate::tools::mcp_stdio::unwrap_rescued)
                    .map_err(|e| RpcError::local(e.to_string()))
            }
            crate::tools::mcp::EventsPeer::Http {
                url,
                headers,
                auth,
                protocol_version,
                ..
            } => {
                let http = self
                    .http
                    .as_ref()
                    .ok_or_else(|| RpcError::local("no HTTP client"))?;
                let id = NEXT_HTTP_ID.fetch_add(1, Ordering::Relaxed);
                let body = Self::http_body(id, method, params, protocol_version);
                let want = body["id"].clone();
                let fut = async {
                    let resp = Self::send(
                        http,
                        url,
                        headers,
                        auth.as_deref(),
                        protocol_version,
                        method,
                        &body,
                    )
                    .await?;
                    let sse = is_sse(&resp);
                    let status = resp.status();
                    if !sse {
                        let bytes = read_capped(resp, MAX_RESPONSE_BYTES)
                            .await
                            .map_err(|e| RpcError::local(format!("{method}: {e}")))?;
                        let msg: Value = serde_json::from_slice(&bytes).map_err(|_| {
                            RpcError::local(format!("{method}: HTTP {status} with a non-JSON body"))
                        })?;
                        return rpc_outcome(msg);
                    }
                    let mut events = SseReader::new(resp);
                    while let Some(msg) = events.next().await {
                        if msg.get("id") == Some(&want) {
                            return rpc_outcome(msg);
                        }
                    }
                    Err(RpcError::local(format!(
                        "{method}: the stream ended without a response"
                    )))
                };
                tokio::time::timeout(timeout, fut)
                    .await
                    .map_err(|_| RpcError::local(format!("{method}: timed out")))?
            }
        }
    }

    /// Open one `events/stream`. Every message — notifications, then a final `$final`/`$closed` —
    /// arrives on the returned handle's channel, whichever transport carried it.
    pub(super) async fn open_stream(&self, params: Value) -> Result<StreamHandle, RpcError> {
        let (tx, rx) = mpsc::channel::<StreamMsg>(stream_buffer());
        let overflowed = Arc::new(AtomicBool::new(false));
        match &self.peer {
            crate::tools::mcp::EventsPeer::Rmcp { peer, router, .. } => {
                // No request timeout: the draft says clients SHOULD NOT apply one to
                // `events/stream`; the heartbeat is the liveness signal instead.
                let tracked = self.peer.track_request();
                let handle = peer
                    .send_request_with_option(
                        custom_request("events/stream", params),
                        PeerRequestOptions::no_options(),
                    )
                    .await
                    .map_err(RpcError::from_service)?;
                let id = handle.id.clone();
                if let Some(call) = &tracked {
                    call.bind(id.clone());
                }
                router.register(id.clone(), tx.clone(), overflowed.clone());
                let response = handle.rx;
                let finisher = tokio::spawn(async move {
                    // In flight until the stream's response arrives.
                    let _tracked = tracked;
                    let fin = match response.await {
                        Ok(Ok(result)) => {
                            StreamMsg::final_ok(serde_json::to_value(result).unwrap_or(Value::Null))
                        }
                        Ok(Err(e)) => StreamMsg::final_err(RpcError::from_service(e)),
                        Err(_) => StreamMsg::closed("the connection closed"),
                    };
                    let _ = tx.send(fin).await;
                });
                Ok(StreamHandle {
                    rx,
                    overflowed,
                    kind: StreamKind::Rmcp {
                        peer: peer.clone(),
                        router: router.clone(),
                        id,
                        finisher,
                    },
                })
            }
            crate::tools::mcp::EventsPeer::Http {
                url,
                headers,
                auth,
                protocol_version,
                ..
            } => {
                let http = self
                    .http
                    .as_ref()
                    .ok_or_else(|| RpcError::local("no HTTP client"))?;
                let id = NEXT_HTTP_ID.fetch_add(1, Ordering::Relaxed);
                let body = Self::http_body(id, "events/stream", params, protocol_version);
                let want = body["id"].clone();
                // No timeout on the stream itself, for the same reason as above.
                let resp = Self::send(
                    http,
                    url,
                    headers,
                    auth.as_deref(),
                    protocol_version,
                    "events/stream",
                    &body,
                )
                .await?;
                if !is_sse(&resp) {
                    // A JSON answer to a stream request is an immediate error (or a result).
                    let bytes = read_capped(resp, MAX_RESPONSE_BYTES)
                        .await
                        .map_err(|e| RpcError::local(format!("events/stream: {e}")))?;
                    let msg: Value = serde_json::from_slice(&bytes)
                        .map_err(|e| RpcError::local(format!("events/stream: {e}")))?;
                    rpc_outcome(msg)?;
                    return Err(RpcError::local("events/stream answered without a stream"));
                }
                // Backpressure, not loss: the reader awaits channel space, so TCP flow control
                // slows the server rather than anything being dropped.
                let reader = tokio::spawn(async move {
                    let mut events = SseReader::new(resp);
                    while let Some(msg) = events.next().await {
                        if msg.get("id") == Some(&want) {
                            let fin = match rpc_outcome(msg) {
                                Ok(v) => StreamMsg::final_ok(v),
                                Err(e) => StreamMsg::final_err(e),
                            };
                            let _ = tx.send(fin).await;
                            return;
                        }
                        if let Some(method) = msg.get("method").and_then(Value::as_str) {
                            let mut params = msg.get("params").cloned().unwrap_or(Value::Null);
                            if let Value::Object(p) = &mut params {
                                p.remove("_meta");
                            }
                            if tx
                                .send(StreamMsg {
                                    method: method.to_owned(),
                                    params,
                                })
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                    let reason = events
                        .error
                        .take()
                        .unwrap_or_else(|| "the server closed the stream".into());
                    let _ = tx.send(StreamMsg::closed(&reason)).await;
                });
                Ok(StreamHandle {
                    rx,
                    overflowed,
                    kind: StreamKind::Http {
                        reader,
                        cancel: Some((
                            http.clone(),
                            url.clone(),
                            headers.clone(),
                            protocol_version.clone(),
                            body["id"].clone(),
                        )),
                    },
                })
            }
        }
    }
}

fn is_sse(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"))
}

/// A JSON-RPC response object → its result, or its error.
fn rpc_outcome(msg: Value) -> Result<Value, RpcError> {
    if let Some(err) = msg.get("error") {
        return Err(RpcError::from_json(err));
    }
    Ok(msg.get("result").cloned().unwrap_or(Value::Null))
}

/// Minimal `text/event-stream` reader: yields each event's `data:` payload parsed as JSON.
///
/// Bounded and linear: an event larger than [`MAX_SSE_EVENT_BYTES`] ends the stream (with
/// [`Self::error`] saying why) instead of growing the buffer, and the separator scan resumes where
/// it stopped rather than rescanning the whole buffer on every chunk.
pub(super) struct SseReader {
    resp: reqwest::Response,
    buf: Vec<u8>,
    /// Where the next unread event starts in `buf`. Events are parsed in place and the consumed
    /// prefix is dropped only when more bytes are needed — once per chunk, never once per event —
    /// so a chunk holding many events costs time linear in its size.
    start: usize,
    /// How far into `buf` no separator can start — the next scan begins here.
    scanned: usize,
    pub(super) error: Option<String>,
}

impl SseReader {
    pub(super) fn new(resp: reqwest::Response) -> Self {
        Self {
            resp,
            buf: Vec::new(),
            start: 0,
            scanned: 0,
            error: None,
        }
    }

    pub(super) async fn next(&mut self) -> Option<Value> {
        loop {
            if let Some((end, sep_end)) = find_event_end(&self.buf, self.scanned.max(self.start)) {
                let parsed = parse_sse_event(&self.buf[self.start..end]);
                self.start = sep_end;
                self.scanned = sep_end;
                match parsed {
                    Some(v) => return Some(v),
                    None => continue,
                }
            }
            // Need more bytes: drop what has been consumed (at most a partial event remains).
            if self.start > 0 {
                self.buf.drain(..self.start);
                self.scanned = self.scanned.saturating_sub(self.start);
                self.start = 0;
            }
            // A separator is at most four bytes, so one could still start in the last three.
            self.scanned = self.scanned.max(self.buf.len().saturating_sub(3));
            if self.buf.len() > MAX_SSE_EVENT_BYTES {
                self.error = Some(format!("an SSE event exceeded {MAX_SSE_EVENT_BYTES} bytes"));
                return None;
            }
            match self.resp.chunk().await {
                Ok(Some(chunk)) => self.buf.extend_from_slice(&chunk),
                _ => return None,
            }
        }
    }
}

/// One SSE event's `data:` lines, joined and parsed as JSON. `None` for an event with no data or
/// data that is not JSON.
fn parse_sse_event(event: &[u8]) -> Option<Value> {
    let text = String::from_utf8_lossy(event);
    let data: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(|d| d.strip_prefix(' ').unwrap_or(d))
        .collect();
    if data.is_empty() {
        return None;
    }
    serde_json::from_str::<Value>(&data.join("\n")).ok()
}

/// Where the first blank line (`\n\n` or `\r\n\r\n`) at or after `from` ends:
/// `(event_end, separator_end)`.
fn find_event_end(buf: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, i + 2));
        }
        if i + 3 < buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, i + 4));
        }
        i += 1;
    }
    None
}

/// An open push stream. Dropping it stops listening; [`StreamHandle::cancel`] also tells the server.
pub(super) struct StreamHandle {
    pub(super) rx: mpsc::Receiver<StreamMsg>,
    overflowed: Arc<AtomicBool>,
    kind: StreamKind,
}

type HttpCancel = (
    reqwest::Client,
    String,
    Vec<(http::HeaderName, http::HeaderValue)>,
    String,
    Value,
);

enum StreamKind {
    Rmcp {
        peer: Peer<RoleClient>,
        router: NotificationRouter,
        id: RequestId,
        finisher: tokio::task::JoinHandle<()>,
    },
    Http {
        reader: tokio::task::JoinHandle<()>,
        cancel: Option<HttpCancel>,
    },
}

impl StreamHandle {
    /// Whether the router had to drop a notification for this stream (see [`NotificationRouter`]).
    pub(super) fn overflowed(&self) -> bool {
        self.overflowed.load(Ordering::Acquire)
    }

    /// Stop the stream: `notifications/cancelled` on stdio; on HTTP, abort the response (the
    /// draft's signal) and send `notifications/cancelled` too, for servers that only notice that.
    pub(super) async fn cancel(mut self, reason: &str) {
        match &mut self.kind {
            StreamKind::Rmcp { peer, id, .. } => {
                let notification = CancelledNotification::new(CancelledNotificationParam::new(
                    Some(id.clone()),
                    Some(reason.to_owned()),
                ));
                let _ = peer.send_notification(notification.into()).await;
            }
            StreamKind::Http { reader, cancel } => {
                reader.abort();
                if let Some((http, url, headers, version, id)) = cancel.take() {
                    let body = json!({
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": { "requestId": id, "reason": reason },
                    });
                    let mut req = http
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .header("Accept", "application/json, text/event-stream")
                        .header("MCP-Protocol-Version", version)
                        .header("Mcp-Method", "notifications/cancelled");
                    for (k, v) in &headers {
                        req = req.header(k, v);
                    }
                    let _ = tokio::time::timeout(
                        Duration::from_secs(3),
                        req.body(body.to_string()).send(),
                    )
                    .await;
                }
            }
        }
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        match &self.kind {
            StreamKind::Rmcp {
                router,
                id,
                finisher,
                ..
            } => {
                router.unregister(id);
                finisher.abort();
            }
            StreamKind::Http { reader, .. } => reader.abort(),
        }
    }
}

impl StreamMsg {
    pub(super) const FINAL: &'static str = "$final";
    pub(super) const CLOSED: &'static str = "$closed";

    fn final_ok(result: Value) -> Self {
        Self {
            method: Self::FINAL.into(),
            params: json!({ "result": result }),
        }
    }

    fn final_err(e: RpcError) -> Self {
        Self {
            method: Self::FINAL.into(),
            params: json!({ "error": e.to_json() }),
        }
    }

    fn closed(why: &str) -> Self {
        Self {
            method: Self::CLOSED.into(),
            params: json!({ "reason": why }),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// RFC 3339 / ISO 8601 parsing (for `refreshBefore`)
// ---------------------------------------------------------------------------------------------

/// Days since the Unix epoch for a civil date (Howard Hinnant's `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// An ISO 8601 / RFC 3339 date-time → Unix milliseconds. Accepts `T`, `t` or a space between date
/// and time; seconds and a fraction of any length are optional; the zone is `Z`/`z`, `±HH:MM`,
/// `±HHMM` or `±HH`. Out-of-range fields are refused rather than wrapped.
pub(super) fn parse_rfc3339_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    let b = s.as_bytes();
    let num = |r: std::ops::Range<usize>| -> Option<i64> {
        let part = s.get(r)?;
        part.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| part.parse().ok())?
    };
    if b.len() < 16 || b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'T' | b't' | b' ') {
        return None;
    }
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    if b[13] != b':' {
        return None;
    }
    let (h, mi) = (num(11..13)?, num(14..16)?);
    let mut i = 16;
    let mut se = 0;
    if b.get(i) == Some(&b':') {
        se = num(17..19)?;
        i = 19;
    }
    let mut ms = 0i64;
    if b.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        let frac = s.get(start..i)?;
        if frac.is_empty() {
            return None;
        }
        let digits: String = frac.chars().chain("000".chars()).take(3).collect();
        ms = digits.parse().ok()?;
    }
    let offset_min = match b.get(i)? {
        b'Z' | b'z' if i + 1 == b.len() => 0,
        sign @ (b'+' | b'-') => {
            let rest = s.get(i + 1..)?;
            let (oh, om) = match rest.len() {
                2 => (num(i + 1..i + 3)?, 0),
                4 => (num(i + 1..i + 3)?, num(i + 3..i + 5)?),
                5 if rest.as_bytes()[2] == b':' => (num(i + 1..i + 3)?, num(i + 4..i + 6)?),
                _ => return None,
            };
            if oh > 23 || om > 59 {
                return None;
            }
            let v = oh * 60 + om;
            if *sign == b'-' { -v } else { v }
        }
        _ => return None,
    };
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    let days = days_from_civil(y, mo, d);
    let secs = days * 86_400 + h * 3600 + mi * 60 + se - offset_min * 60;
    Some(secs * 1000 + ms)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_parses_zulu_offsets_fractions_and_minute_precision() {
        assert_eq!(parse_rfc3339_ms("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(
            parse_rfc3339_ms("2026-02-19T16:30:00Z"),
            Some(1_771_518_600_000)
        );
        assert_eq!(
            parse_rfc3339_ms("2026-02-19T17:30:00.250+01:00"),
            Some(1_771_518_600_250)
        );
        // No seconds (valid ISO 8601), compact and hour-only offsets.
        assert_eq!(
            parse_rfc3339_ms("2026-02-19T16:30Z"),
            Some(1_771_518_600_000)
        );
        assert_eq!(
            parse_rfc3339_ms("2026-02-19T17:30+0100"),
            Some(1_771_518_600_000)
        );
        assert_eq!(
            parse_rfc3339_ms("2026-02-19T17:30:00+01"),
            Some(1_771_518_600_000)
        );
        for bad in [
            "not a date",
            "2026-13-01T00:00:00Z",
            "2026-02-19T16:30:00",
            "2026-02-19T16:30:00Zjunk",
        ] {
            assert_eq!(parse_rfc3339_ms(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_sse_separator_scan_finds_lf_and_crlf_and_resumes() {
        assert_eq!(find_event_end(b"data: x\n\nrest", 0), Some((7, 9)));
        assert_eq!(find_event_end(b"data: x\r\n\r\n", 0), Some((7, 11)));
        assert_eq!(find_event_end(b"data: x\n", 0), None);
        // Resuming past the start still finds a separator that straddled the previous end.
        assert_eq!(find_event_end(b"abc\n\n", 3), Some((3, 5)));
    }

    #[tokio::test]
    async fn an_oversized_sse_event_ends_the_stream_instead_of_buffering_it() {
        let body = format!("data: {}", "x".repeat(MAX_SSE_EVENT_BYTES + 10));
        let resp = reqwest::Response::from(http::Response::new(body));
        let mut reader = SseReader::new(resp);
        assert!(reader.next().await.is_none());
        assert!(reader.error.unwrap().contains("exceeded"));
        // A normal event still parses.
        let resp = reqwest::Response::from(http::Response::new("data: {\"a\":1}\n\n".to_owned()));
        assert_eq!(SseReader::new(resp).next().await, Some(json!({"a": 1})));
    }

    /// Many events in one chunk are drained in time linear in the chunk, not quadratic: consuming
    /// each event by shifting the rest of the buffer down moves ~700 GB here (seconds), the linear drain a fraction of one.
    #[tokio::test]
    async fn many_events_in_one_chunk_drain_in_linear_time() {
        let n = 300_000;
        let mut body = String::with_capacity(n * 20);
        for i in 0..n {
            body.push_str(&format!("data: {{\"i\":{i}}}\n\n"));
        }
        // All of it already buffered — one large read, as a fast server on a fast link delivers.
        let resp = reqwest::Response::from(http::Response::new(String::new()));
        let mut reader = SseReader::new(resp);
        reader.buf = body.into_bytes();
        let started = std::time::Instant::now();
        let mut seen = 0;
        while let Some(v) = reader.next().await {
            assert_eq!(v["i"], seen);
            seen += 1;
        }
        assert_eq!(seen, n);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "draining {n} events took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_full_stream_is_marked_overflowed_and_forwards_nothing_after() {
        let router = NotificationRouter::default();
        let (tx, mut rx) = mpsc::channel(1);
        let flag = Arc::new(AtomicBool::new(false));
        let id = RequestId::Number(1);
        router.register(id.clone(), tx, flag.clone());
        let n =
            |i: i64| CustomNotification::new("notifications/events/event", Some(json!({"i": i})));
        router.route(n(1), Some(id.clone()));
        router.route(n(2), Some(id.clone())); // full: dropped, stream marked
        assert!(flag.load(Ordering::Acquire));
        assert_eq!(rx.try_recv().unwrap().params["i"], 1);
        router.route(n(3), Some(id)); // never forwarded past the drop
        assert!(rx.try_recv().is_err());
    }

    /// A loopback server answering 401 to any request without `Bearer fresh`, 200 otherwise;
    /// records each request's `Authorization`.
    async fn bearer_server() -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = seen.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let record = record.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
                    let mut buf = vec![0u8; 16384];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
                    let auth = head
                        .lines()
                        .find_map(|l| l.strip_prefix("authorization: "))
                        .unwrap_or_default()
                        .to_owned();
                    record.lock().unwrap().push(auth.clone());
                    let reply: &[u8] = if auth == "bearer fresh" {
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
                    } else {
                        b"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    };
                    let _ = stream.write_all(reply).await;
                });
            }
        });
        (url, seen)
    }

    fn bearer(token: &str) -> Vec<(http::HeaderName, http::HeaderValue)> {
        vec![(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        )]
    }

    #[tokio::test]
    async fn a_direct_events_request_answered_401_refreshes_the_token_and_is_resent_once() {
        agent_core::ensure_provider();
        let (url, seen) = bearer_server().await;
        let auth = crate::tools::mcp_oauth::ServerAuth::fake("stale", || Ok("fresh".into()));
        let resp = Conn::send(
            &reqwest::Client::new(),
            &url,
            &bearer("stale"),
            Some(&auth),
            "2026-07-28",
            "events/list",
            &json!({}),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(*seen.lock().unwrap(), ["bearer stale", "bearer fresh"]);

        // A refresh that fails is the error, naming `agent mcp-login`; nothing is resent.
        let (url, seen) = bearer_server().await;
        let auth = crate::tools::mcp_oauth::ServerAuth::fake("stale", || {
            Err(crate::tools::mcp_oauth::RefreshError::definitive(
                "invalid_grant",
            ))
        });
        let e = Conn::send(
            &reqwest::Client::new(),
            &url,
            &bearer("stale"),
            Some(&auth),
            "2026-07-28",
            "events/list",
            &json!({}),
        )
        .await
        .unwrap_err();
        assert!(e.message.contains("agent mcp-login"), "{}", e.message);
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_direct_events_request_sends_the_current_shared_token_not_the_one_it_was_built_with()
    {
        // The headers were built (at `events_peer`) with the token of that moment; another
        // connection has refreshed it since. The request must not spend a 401 on the old one.
        agent_core::ensure_provider();
        let (url, seen) = bearer_server().await;
        let auth = crate::tools::mcp_oauth::ServerAuth::fake("fresh", || Ok("unused".into()));
        let resp = Conn::send(
            &reqwest::Client::new(),
            &url,
            &bearer("stale"),
            Some(&auth),
            "2026-07-28",
            "events/list",
            &json!({}),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(*seen.lock().unwrap(), ["bearer fresh"]);
    }
}
