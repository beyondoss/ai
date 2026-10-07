//! The streamable-HTTP client an MCP Apps-flavoured connection dials through: `reqwest`, except that
//! an MCP App view's `resources/read` (a `ui://` URI) is sent here and its response read under the
//! view cap ([`crate::tools::mcp_apps::MAX_VIEW_BYTES`]) — refused from its `Content-Length` before a
//! byte of the body is read, or as soon as a streamed body or SSE event passes the cap — instead of
//! being read whole by rmcp's own client and refused after. Every other request goes to rmcp's
//! client (through [`crate::tools::mcp_wire::HttpClient`]) unchanged.
//!
//! A refused read is answered with a JSON-RPC error for its own id, so the caller sees an ordinary
//! failed `resources/read` and the view falls back to text.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;
use http::{HeaderName, HeaderValue};
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::streamable_http_client::{
    SseError, StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
};
use serde_json::{Value, json};

const SESSION_HEADER: &str = "Mcp-Session-Id";

/// `reqwest`, with view reads capped. See the module doc.
#[derive(Clone)]
pub(crate) struct ViewCappedHttp {
    /// Everything that is not a view read goes here (which itself rescues extension results).
    inner: crate::tools::mcp_wire::HttpClient,
    cap: usize,
}

impl ViewCappedHttp {
    pub(crate) fn new(inner: crate::tools::mcp_wire::HttpClient) -> Self {
        Self {
            inner,
            cap: crate::tools::mcp_apps::MAX_VIEW_BYTES,
        }
    }
}

/// The request id of `message` if it is a `resources/read` of a `ui://` view.
fn view_read_id(message: &ClientJsonRpcMessage) -> Option<Value> {
    let v = serde_json::to_value(message).ok()?;
    let is_view = v["method"] == "resources/read"
        && v["params"]["uri"]
            .as_str()
            .is_some_and(crate::tools::mcp_apps::is_ui_uri);
    is_view.then(|| v.get("id").cloned()).flatten()
}

fn refusal(id: &Value, cap: usize) -> Result<ServerJsonRpcMessage, Err> {
    Ok(serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32000, "message": format!("MCP message over {cap} bytes refused by the host") },
    }))?)
}

type Err = StreamableHttpError<reqwest::Error>;

impl ViewCappedHttp {
    async fn capped_view_read(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        id: Value,
        session_id: Option<Arc<str>>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, Err> {
        let mut request = self.inner.0.post(uri.as_ref()).header(
            reqwest::header::ACCEPT,
            "text/event-stream, application/json",
        );
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        for (name, value) in custom_headers {
            // The transport's own headers win, as in rmcp's client.
            if ["accept", "mcp-session-id", "last-event-id"].contains(&name.as_str()) {
                continue;
            }
            request = request.header(name, value);
        }
        let session_was_attached = session_id.is_some();
        if let Some(session) = session_id {
            request = request.header(SESSION_HEADER, session.as_ref());
        }
        let response = request.json(&message).send().await.map_err(Err::Client)?;
        let status = response.status();
        let session = response
            .headers()
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        if status == reqwest::StatusCode::NOT_FOUND && session_was_attached {
            return Err(Err::SessionExpired);
        }
        if !status.is_success() {
            return Err(Err::UnexpectedServerResponse(
                format!("HTTP {status}").into(),
            ));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        // Refused on its declared length: not a byte of the body is read.
        if response
            .content_length()
            .is_some_and(|len| len as usize > self.cap)
        {
            tracing::warn!(
                cap = self.cap,
                "refused an MCP App view by its Content-Length"
            );
            return Ok(StreamableHttpPostResponse::Json(
                refusal(&id, self.cap)?,
                session,
            ));
        }
        let mut body = response.bytes_stream();
        if content_type.starts_with("text/event-stream") {
            // Read events until this request's response, holding no event past the cap.
            let mut pending: Vec<u8> = Vec::new();
            while let Some(chunk) = body.next().await {
                let chunk = chunk.map_err(Err::Client)?;
                pending.extend_from_slice(&chunk);
                while let Some(end) = event_end(&pending) {
                    let event: Vec<u8> = pending.drain(..end).collect();
                    if let Some(msg) = event_message(&event)
                        && message_id(&msg).as_ref() == Some(&id)
                    {
                        return Ok(StreamableHttpPostResponse::Json(msg, session));
                    }
                }
                if pending.len() > self.cap {
                    tracing::warn!(
                        cap = self.cap,
                        "refused an MCP App view streamed over its cap"
                    );
                    return Ok(StreamableHttpPostResponse::Json(
                        refusal(&id, self.cap)?,
                        session,
                    ));
                }
            }
            return Err(Err::UnexpectedEndOfStream);
        }
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(Err::Client)?;
            if buf.len() + chunk.len() > self.cap {
                tracing::warn!(
                    cap = self.cap,
                    "refused an MCP App view streamed over its cap"
                );
                return Ok(StreamableHttpPostResponse::Json(
                    refusal(&id, self.cap)?,
                    session,
                ));
            }
            buf.extend_from_slice(&chunk);
        }
        let msg: ServerJsonRpcMessage = serde_json::from_slice(&buf)?;
        Ok(StreamableHttpPostResponse::Json(msg, session))
    }
}

/// The end (exclusive) of the first complete SSE event in `buf`, after its blank line.
fn event_end(buf: &[u8]) -> Option<usize> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|i| i + 2);
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// The JSON-RPC message an SSE event's `data:` lines carry, if any.
fn event_message(event: &[u8]) -> Option<ServerJsonRpcMessage> {
    let text = std::str::from_utf8(event).ok()?;
    let data: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(|d| d.strip_prefix(' ').unwrap_or(d))
        .collect();
    if data.is_empty() {
        return None;
    }
    serde_json::from_str(&data.join("\n")).ok()
}

fn message_id(msg: &ServerJsonRpcMessage) -> Option<Value> {
    serde_json::to_value(msg).ok()?.get("id").cloned()
}

impl StreamableHttpClient for ViewCappedHttp {
    type Error = reqwest::Error;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, Err> {
        if let Some(id) = view_read_id(&message) {
            return self
                .capped_view_read(uri, message, id, session_id, auth_header, custom_headers)
                .await;
        }
        self.inner
            .post_message(uri, message, session_id, auth_header, custom_headers)
            .await
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, Err> {
        if let Some(id) = view_read_id(&message) {
            return self
                .capped_view_read(uri, message, id, session_id, auth_header, custom_headers)
                .await;
        }
        self.inner
            .post_message_with_max_sse_event_size(
                uri,
                message,
                session_id,
                auth_header,
                custom_headers,
                max_sse_event_size,
            )
            .await
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), Err> {
        self.inner
            .delete_session(uri, session_id, auth_header, custom_headers)
            .await
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<sse_stream::Sse, SseError>>, Err> {
        self.inner
            .get_stream(uri, session_id, last_event_id, auth_header, custom_headers)
            .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<sse_stream::Sse, SseError>>, Err> {
        self.inner
            .get_stream_with_max_sse_event_size(
                uri,
                session_id,
                last_event_id,
                auth_header,
                custom_headers,
                max_sse_event_size,
            )
            .await
    }
}
