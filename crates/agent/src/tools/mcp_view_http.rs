//! The streamable-HTTP client an MCP Apps-flavoured connection dials through: `reqwest`, except that
//! an MCP App view's `resources/read` (a `ui://` URI) is read under the view cap
//! ([`crate::tools::mcp_apps::MAX_VIEW_BYTES`]) — refused from its `Content-Length` before a byte of
//! the body is read, or as soon as a streamed body or one SSE event passes the cap — instead of
//! being read whole by rmcp's own client and refused after. Every other request goes to rmcp's
//! client (through [`crate::tools::mcp_wire::HttpClient`]) unchanged.
//!
//! The view read itself is [`HttpClient::post_bounded`](crate::tools::mcp_wire::HttpClient) — the
//! same request path `skills/*` takes — so it gets everything rmcp's own client does: a 401/403
//! with `WWW-Authenticate` comes back as `AuthRequired`/`InsufficientScope`, a 404 on a session as
//! `SessionExpired`, and an SSE response is handed to rmcp **as a stream**, so every other event on
//! it (progress, a server→client request, a log message) is routed exactly as on any other request.
//! A refused read is answered with a JSON-RPC error for its own id (in place of the oversized event,
//! on a stream), so the caller sees an ordinary failed `resources/read` and the view falls back to
//! text.

use std::collections::HashMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use http::{HeaderName, HeaderValue};
use rmcp::model::ClientJsonRpcMessage;
use rmcp::transport::streamable_http_client::{
    SseError, StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
};
use serde_json::Value;

use crate::tools::mcp_wire::{Limit, OverLimit};

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
    if !matches!(message, ClientJsonRpcMessage::Request(_)) {
        return None;
    }
    let v = serde_json::to_value(message).ok()?;
    let is_view = v["method"] == "resources/read"
        && v["params"]["uri"]
            .as_str()
            .is_some_and(crate::tools::mcp_apps::is_ui_uri);
    is_view.then(|| v.get("id").cloned()).flatten()
}

type Err = StreamableHttpError<reqwest::Error>;

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
                .inner
                .post_bounded(
                    uri,
                    message,
                    session_id,
                    auth_header,
                    custom_headers,
                    Limit {
                        max: self.cap,
                        over: OverLimit::Refuse(id),
                    },
                )
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
                .inner
                .post_bounded(
                    uri,
                    message,
                    session_id,
                    auth_header,
                    custom_headers,
                    Limit {
                        max: self.cap.min(max_sse_event_size),
                        over: OverLimit::Refuse(id),
                    },
                )
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt as _;
    use rmcp::model::ServerJsonRpcMessage;
    use serde_json::json;

    /// One canned HTTP response per connection from a loopback listener, held open afterwards (an
    /// SSE stream the server never ends).
    async fn canned(response: &'static [u8]) -> String {
        canned_in_parts(Box::leak(Box::new([response]))).await
    }

    /// [`canned`], written in `parts` with a pause between, so each reaches the client as a chunk
    /// of its own (an event split across reads, as a large one always is).
    async fn canned_in_parts(parts: &'static [&'static [u8]]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
                    for part in parts {
                        let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, part).await;
                        let _ = tokio::io::AsyncWriteExt::flush(&mut stream).await;
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(600)).await;
                });
            }
        });
        format!("http://{addr}/mcp")
    }

    fn view_read() -> ClientJsonRpcMessage {
        serde_json::from_value(json!({
            "jsonrpc": "2.0", "id": 7, "method": "resources/read", "params": { "uri": "ui://w/v" }
        }))
        .unwrap()
    }

    fn client(cap: usize) -> ViewCappedHttp {
        agent_core::ensure_provider();
        ViewCappedHttp {
            inner: crate::tools::mcp_wire::HttpClient(reqwest::Client::new()),
            cap,
        }
    }

    async fn read(url: &str, cap: usize) -> Result<StreamableHttpPostResponse, Err> {
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            client(cap).post_message(url.into(), view_read(), None, None, HashMap::new()),
        )
        .await
        .expect("the view read must return while the stream is still open")
    }

    fn error_id(msg: &ServerJsonRpcMessage) -> Option<Value> {
        let v = serde_json::to_value(msg).unwrap();
        v.get("error").is_some().then(|| v["id"].clone())
    }

    #[tokio::test]
    async fn a_view_read_answered_401_surfaces_as_auth_required_like_any_other_request() {
        // rmcp's own client turns this into `AuthRequired` carrying the challenge — the error a
        // caller re-authenticates on. The view read must not flatten it into a bare "HTTP 401".
        let url = canned(
            b"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer resource_metadata=\"http://a/.well-known/oauth-protected-resource\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await;
        let Err(e) = read(&url, 1024).await else {
            panic!("a 401 is an error");
        };
        let StreamableHttpError::AuthRequired(auth) = &e else {
            panic!("a 401 view read must surface as AuthRequired, got {e:?}");
        };
        assert!(
            auth.www_authenticate_header
                .contains("oauth-protected-resource"),
            "the challenge rides the error, to authorize against"
        );

        let url = canned(
            b"HTTP/1.1 403 Forbidden\r\nWWW-Authenticate: Bearer error=\"insufficient_scope\", scope=\"ui:read\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await;
        let Err(e) = read(&url, 1024).await else {
            panic!("a 403 is an error");
        };
        let StreamableHttpError::InsufficientScope(scope) = &e else {
            panic!("a 403 view read must surface as InsufficientScope, got {e:?}");
        };
        assert_eq!(scope.get_required_scope(), Some("ui:read"));
    }

    #[tokio::test]
    async fn a_view_reads_sse_stream_is_handed_to_rmcp_with_every_event_on_it() {
        // A progress notification and a server→client request ride the stream before the
        // response: rmcp must see all three (it routes the first two), not just the response.
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n\
              data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{\"progressToken\":1,\"progress\":1}}\n\n\
              data: {\"jsonrpc\":\"2.0\",\"id\":\"s-1\",\"method\":\"ping\"}\n\n\
              data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"contents\":[{\"uri\":\"ui://w/v\",\"mimeType\":\"text/html;profile=mcp-app\",\"text\":\"<p>hi</p>\"}]}}\n\n",
        )
        .await;
        let StreamableHttpPostResponse::Sse(mut events, _) = read(&url, 1024).await.unwrap() else {
            panic!("an SSE view read is handed to rmcp as a stream");
        };
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(events.next().await.unwrap().unwrap().data.unwrap());
        }
        assert!(seen[0].contains("notifications/progress"), "{seen:?}");
        assert!(seen[1].contains("\"method\":\"ping\""), "{seen:?}");
        assert!(seen[2].contains("<p>hi</p>"), "{seen:?}");
    }

    #[tokio::test]
    async fn an_sse_event_over_the_view_cap_becomes_the_refusal_and_ends_the_stream() {
        let url = canned_in_parts(&[
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n\
              data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\",\"data\":\"x\"}}\n\n",
            b"data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"contents\":[{\"uri\":\"ui://w/v\",\"text\":\"vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv",
            b"vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\"}]}}\n\n",
        ])
        .await;
        let StreamableHttpPostResponse::Sse(mut events, _) = read(&url, 128).await.unwrap() else {
            panic!("an SSE view read is handed to rmcp as a stream");
        };
        let first = events.next().await.unwrap().unwrap();
        assert!(first.data.unwrap().contains("notifications/message"));
        let refused = events.next().await.unwrap().unwrap();
        let msg: ServerJsonRpcMessage = serde_json::from_str(&refused.data.unwrap()).unwrap();
        assert_eq!(error_id(&msg), Some(json!(7)), "{msg:?}");
        assert!(
            events.next().await.is_none(),
            "the stream ends at the refusal"
        );
    }

    #[tokio::test]
    async fn a_json_view_over_the_cap_is_refused_by_its_content_length_or_as_it_streams() {
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 999999\r\n\r\n{",
        )
        .await;
        let StreamableHttpPostResponse::Json(msg, _) = read(&url, 1024).await.unwrap() else {
            panic!("a refused JSON view read is answered with a JSON-RPC error");
        };
        assert_eq!(error_id(&msg), Some(json!(7)));

        // No Content-Length (the body runs to the connection's close, which never comes): refused
        // once the body passes the cap, without waiting for the rest.
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n\
              {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"contents\":[{\"uri\":\"ui://w/v\",\"text\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        .await;
        let StreamableHttpPostResponse::Json(msg, _) = read(&url, 64).await.unwrap() else {
            panic!("a refused JSON view read is answered with a JSON-RPC error");
        };
        assert_eq!(error_id(&msg), Some(json!(7)));
    }
}
