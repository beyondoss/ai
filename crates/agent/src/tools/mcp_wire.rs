//! The streamable-HTTP half of keeping extension results intact on their way through `rmcp`.
//!
//! `rmcp` (3.2 through at least 3.5.1) decodes every server result into an *untagged* union, and
//! `CallToolResult` matches any object carrying `_meta` — which every `2026-07-28` server puts on
//! every result — so a `skills/list` answer would arrive as an empty tool result (upstream:
//! modelcontextprotocol/rust-sdk#1197). The workaround is one boundary,
//! [`crate::tools::mcp_stdio::rescue`] / [`crate::tools::mcp_stdio::unwrap_rescued`], applied to the
//! raw bytes before `rmcp` parses them. For stdio that is the one stdio transport
//! ([`crate::tools::mcp_stdio`]); for streamable HTTP it is [`HttpClient`] here, which wraps the
//! `reqwest` client `rmcp` drives and answers `skills/*` POSTs itself (with the exact URI, session,
//! auth and standard headers the transport handed it), delegating everything else untouched.
//!
//! Delete this module when the upstream fix ships.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use http::{HeaderName, HeaderValue};
use rmcp::model::{ClientJsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::streamable_http_client::{
    SseError, StreamableHttpClient, StreamableHttpError, StreamableHttpPostResponse,
};
use serde_json::Value;

/// The `reqwest` client `rmcp`'s streamable-HTTP transport drives, with `skills/*` answered here so
/// their results can be [rescued](crate::tools::mcp_stdio::rescue) before `rmcp` parses them.
#[derive(Clone)]
pub(crate) struct HttpClient(pub(crate) reqwest::Client);

fn is_skills_request(message: &ClientJsonRpcMessage) -> bool {
    // The method is the one thing needed; serializing is how to read it without matching every
    // request variant.
    matches!(message, ClientJsonRpcMessage::Request(_))
        && serde_json::to_value(message)
            .ok()
            .and_then(|v| v.get("method").and_then(Value::as_str).map(str::to_string))
            .is_some_and(|m| m.starts_with("skills/"))
}

/// The largest SSE event a `skills/*` response may carry when the transport names no limit: one
/// listing page or entry of a skill at the spec's per-skill limits, with room to spare.
const DEFAULT_MAX_SKILLS_EVENT: usize = 32 * 1024 * 1024;

/// Headers a configured custom header may not override (`rmcp`'s own reserved set).
const RESERVED_HEADERS: [&str; 3] = ["accept", "mcp-session-id", "last-event-id"];

/// Bound an SSE byte stream: an event (bytes since the last blank line) larger than `max` ends the
/// stream with an error rather than buffering without limit.
fn bounded<S>(
    stream: S,
    max: usize,
) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>>
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>>,
{
    use futures::StreamExt as _;
    stream.scan(EventBoundary::default(), move |state, chunk| {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => return std::future::ready(Some(Err(std::io::Error::other(e)))),
        };
        if state.feed(&chunk) > max {
            return std::future::ready(Some(Err(std::io::Error::other(format!(
                "an SSE event exceeded the maximum size of {max} bytes"
            )))));
        }
        std::future::ready(Some(Ok(chunk)))
    })
}

/// Where SSE events end, tracked byte by byte across chunks: an event ends at a blank line, whatever
/// the line terminators (`\n`, `\r\n` or `\r`) and wherever the chunk boundaries fall.
#[derive(Default)]
struct EventBoundary {
    /// Bytes since the last event boundary — the size of the event being received.
    pending: usize,
    /// Line terminators in a row (a CRLF counts once): two is a blank line, the end of an event.
    terminators: u8,
    /// The previous byte was a `\r`, so a following `\n` completes the same terminator.
    after_cr: bool,
}

impl EventBoundary {
    /// Account for `chunk`; returns the size of the event still open after it.
    fn feed(&mut self, chunk: &[u8]) -> usize {
        for &b in chunk {
            match b {
                b'\n' if self.after_cr => self.after_cr = false,
                b'\n' | b'\r' => {
                    self.after_cr = b == b'\r';
                    self.terminators = self.terminators.saturating_add(1);
                    if self.terminators >= 2 {
                        self.pending = 0;
                        continue;
                    }
                    self.pending += 1;
                }
                _ => {
                    self.after_cr = false;
                    self.terminators = 0;
                    self.pending += 1;
                }
            }
        }
        self.pending
    }
}

/// A non-SSE response body, read up to `max` bytes — a server cannot make the client buffer without
/// limit by sending a huge (or endless) JSON body.
async fn capped_body(
    response: reqwest::Response,
    max: usize,
) -> Result<String, StreamableHttpError<reqwest::Error>> {
    use futures::StreamExt as _;
    let mut body = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(StreamableHttpError::Client)?;
        if body.len() + chunk.len() > max {
            return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                format!("a skills response body exceeded the maximum size of {max} bytes"),
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

impl HttpClient {
    /// One `skills/*` POST, built and its status interpreted the way `rmcp`'s own `reqwest` client
    /// does — reserved headers refused, 401/403 surfaced as the auth errors `rmcp` acts on, a 404 on a
    /// session as `SessionExpired` so it re-initializes — with the result rescued before `rmcp`
    /// parses it. An SSE response is handed back **as a stream**, each event rescued as it
    /// arrives: `rmcp` reads it incrementally, routes any server→client message on it, and stops at
    /// the response, so a server that keeps the stream open costs nothing.
    async fn post_skills(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<reqwest::Error>> {
        use futures::StreamExt as _;
        let mut request = self
            .0
            .post(uri.as_ref())
            .header(http::header::ACCEPT, "text/event-stream, application/json");
        if let Some(token) = auth_header {
            request = request.bearer_auth(token);
        }
        for (name, value) in custom_headers {
            if RESERVED_HEADERS.contains(&name.as_str().to_ascii_lowercase().as_str()) {
                return Err(StreamableHttpError::ReservedHeaderConflict(
                    name.to_string(),
                ));
            }
            request = request.header(name, value);
        }
        let session_was_attached = session_id.is_some();
        if let Some(session) = &session_id {
            request = request.header("Mcp-Session-Id", session.as_ref());
        }
        let response = request
            .json(&message)
            .send()
            .await
            .map_err(StreamableHttpError::Client)?;
        let status = response.status();
        let www_authenticate = response
            .headers()
            .get(http::header::WWW_AUTHENTICATE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if status == reqwest::StatusCode::UNAUTHORIZED
            && let Some(header) = &www_authenticate
        {
            return Err(StreamableHttpError::AuthRequired(
                rmcp::transport::streamable_http_client::AuthRequiredError::new(header.clone()),
            ));
        }
        if status == reqwest::StatusCode::FORBIDDEN
            && let Some(header) = &www_authenticate
        {
            return Err(StreamableHttpError::InsufficientScope(
                rmcp::transport::streamable_http_client::InsufficientScopeError::new(
                    header.clone(),
                    scope_of(header),
                ),
            ));
        }
        if matches!(
            status,
            reqwest::StatusCode::ACCEPTED | reqwest::StatusCode::NO_CONTENT
        ) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == reqwest::StatusCode::NOT_FOUND && session_was_attached {
            return Err(StreamableHttpError::SessionExpired);
        }
        let session = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let content_type = response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if content_type
            .as_deref()
            .is_some_and(|ct| ct.starts_with("text/event-stream"))
            && status.is_success()
        {
            let events = sse_stream::SseStream::from_bytes_stream(bounded(
                response.bytes_stream(),
                max_sse_event_size,
            ))
            .map(|event| {
                event.map(|mut event| {
                    if let Some(data) = &event.data
                        && let Some(rescued) = crate::tools::mcp_stdio::rescue(data.as_bytes())
                    {
                        event.data = Some(String::from_utf8_lossy(&rescued).into_owned());
                    }
                    event
                })
            })
            .boxed();
            return Ok(StreamableHttpPostResponse::Sse(events, session));
        }
        let body = capped_body(response, max_sse_event_size).await?;
        let body = match crate::tools::mcp_stdio::rescue(body.as_bytes()) {
            Some(rescued) => String::from_utf8_lossy(&rescued).into_owned(),
            None => body,
        };
        let parsed = serde_json::from_str::<Value>(&body).ok();
        let Some(value) = parsed.filter(|v| v.get("result").is_some() || v.get("error").is_some())
        else {
            return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                format!("HTTP {status}: {body}"),
            )));
        };
        let message: ServerJsonRpcMessage = serde_json::from_value(value)?;
        Ok(StreamableHttpPostResponse::Json(message, session))
    }
}

/// The `scope=` parameter of a `WWW-Authenticate` value, quoted or not.
fn scope_of(header: &str) -> Option<String> {
    let at = header.to_ascii_lowercase().find("scope=")? + "scope=".len();
    let rest = &header[at..];
    match rest.strip_prefix('"') {
        Some(quoted) => quoted.find('"').map(|end| quoted[..end].to_string()),
        None => Some(
            rest.split(|c: char| c == ',' || c.is_whitespace())
                .next()
                .unwrap_or_default()
                .to_string(),
        ),
    }
}

impl StreamableHttpClient for HttpClient {
    type Error = reqwest::Error;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        if is_skills_request(&message) {
            return self
                .post_skills(
                    uri,
                    message,
                    session_id,
                    auth_header,
                    custom_headers,
                    DEFAULT_MAX_SKILLS_EVENT,
                )
                .await;
        }
        self.0
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
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        if is_skills_request(&message) {
            return self
                .post_skills(
                    uri,
                    message,
                    session_id,
                    auth_header,
                    custom_headers,
                    max_sse_event_size,
                )
                .await;
        }
        self.0
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
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        self.0
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
    ) -> Result<
        BoxStream<'static, Result<sse_stream::Sse, SseError>>,
        StreamableHttpError<Self::Error>,
    > {
        self.0
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
    ) -> Result<
        BoxStream<'static, Result<sse_stream::Sse, SseError>>,
        StreamableHttpError<Self::Error>,
    > {
        self.0
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
    use rmcp::model::{CustomResult, ServerResult};
    use serde_json::json;

    /// The bug this module exists for, pinned: if `rmcp` ever stops shadowing, this fails and the
    /// module (and `mcp_stdio::rescue`) can go. A rescued skills result reaches us whole.
    #[test]
    fn rmcp_still_shadows_a_skills_result_that_carries_meta() {
        let raw = json!({
            "resultType": "complete",
            "skills": [],
            "_meta": { "io.modelcontextprotocol/serverInfo": { "name": "s", "version": "" } },
        });
        let parsed: ServerResult = serde_json::from_value(raw.clone()).unwrap();
        assert!(!matches!(parsed, ServerResult::CustomResult(_)));

        let line = json!({ "jsonrpc": "2.0", "id": 1, "result": raw }).to_string();
        let rescued =
            crate::tools::mcp_stdio::rescue(line.as_bytes()).expect("a skills result is lossy");
        let message: Value = serde_json::from_slice(&rescued).unwrap();
        let parsed: ServerResult = serde_json::from_value(message["result"].clone()).unwrap();
        let ServerResult::CustomResult(CustomResult(value)) = parsed else {
            panic!("a rescued skills result must reach us as a custom result");
        };
        let value = crate::tools::mcp_stdio::unwrap_rescued(value);
        assert!(
            value["skills"].is_array() && value["_meta"].is_object(),
            "{value}"
        );
    }

    #[test]
    fn the_scope_comes_out_of_www_authenticate_quoted_or_not() {
        assert_eq!(
            scope_of(r#"Bearer realm="x", scope="files:read files:write""#).as_deref(),
            Some("files:read files:write")
        );
        assert_eq!(
            scope_of("Bearer scope=read:data, error=x").as_deref(),
            Some("read:data")
        );
        assert_eq!(scope_of("Bearer realm=x"), None);
    }

    /// One canned HTTP response per connection, from a loopback listener; `hold` keeps the
    /// connection open afterwards (an SSE stream the server never ends).
    async fn canned(response: &'static [u8], hold: bool) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
                    let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, response).await;
                    if hold {
                        tokio::time::sleep(std::time::Duration::from_secs(600)).await;
                    }
                });
            }
        });
        format!("http://{addr}/mcp")
    }

    fn skills_list() -> ClientJsonRpcMessage {
        serde_json::from_value(json!({
            "jsonrpc": "2.0", "id": 7, "method": "skills/list", "params": {}
        }))
        .unwrap()
    }

    async fn post(url: &str, session: Option<&str>) -> Result<StreamableHttpPostResponse, String> {
        agent_core::ensure_provider();
        HttpClient(reqwest::Client::new())
            .post_message(
                url.into(),
                skills_list(),
                session.map(Into::into),
                None,
                HashMap::new(),
            )
            .await
            .map_err(|e| format!("{e:?}"))
    }

    #[tokio::test]
    async fn an_sse_response_is_streamed_rescued_and_does_not_wait_for_the_server_to_close() {
        use futures::StreamExt as _;
        // A notification first (routed by rmcp, not dropped), then the response, then nothing — the
        // server holds the stream open.
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n\
              data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\",\"data\":\"hi\"}}\n\n\
              data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"skills\":[],\"_meta\":{\"a\":1}}}\n\n",
            true,
        )
        .await;
        let got = tokio::time::timeout(std::time::Duration::from_secs(10), post(&url, None))
            .await
            .expect("the POST must return while the stream is still open")
            .unwrap();
        let StreamableHttpPostResponse::Sse(mut events, _) = got else {
            panic!("an SSE response is handed to rmcp as a stream");
        };
        let first = events.next().await.unwrap().unwrap();
        assert!(first.data.unwrap().contains("notifications/message"));
        let second = events.next().await.unwrap().unwrap();
        let data: Value = serde_json::from_str(&second.data.unwrap()).unwrap();
        let ServerJsonRpcMessage::Response(response) = serde_json::from_value(data).unwrap() else {
            panic!("the second event is the response");
        };
        let ServerResult::CustomResult(CustomResult(value)) = response.result else {
            panic!("the rescued result must reach rmcp's catch-all");
        };
        let value = crate::tools::mcp_stdio::unwrap_rescued(value);
        assert!(value["skills"].is_array(), "{value}");
    }

    #[tokio::test]
    async fn statuses_map_to_the_errors_rmcp_acts_on() {
        let unauthorized = canned(
            b"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm=\"x\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            false,
        )
        .await;
        let e = post(&unauthorized, None).await.unwrap_err();
        assert!(e.contains("AuthRequired"), "{e}");

        let forbidden = canned(
            b"HTTP/1.1 403 Forbidden\r\nWWW-Authenticate: Bearer scope=\"skills:read\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            false,
        )
        .await;
        let e = post(&forbidden, None).await.unwrap_err();
        assert!(
            e.contains("InsufficientScope") && e.contains("skills:read"),
            "{e}"
        );

        let gone = canned(
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            false,
        )
        .await;
        let e = post(&gone, Some("session-1")).await.unwrap_err();
        assert!(e.contains("SessionExpired"), "{e}");

        let accepted = canned(
            b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            false,
        )
        .await;
        assert!(matches!(
            post(&accepted, None).await.unwrap(),
            StreamableHttpPostResponse::Accepted
        ));
    }

    #[tokio::test]
    async fn a_reserved_custom_header_is_refused() {
        agent_core::ensure_provider();
        let mut headers = HashMap::new();
        headers.insert(
            HeaderName::from_static("mcp-session-id"),
            HeaderValue::from_static("forged"),
        );
        let e = HttpClient(reqwest::Client::new())
            .post_message(
                "http://127.0.0.1:9/mcp".into(),
                skills_list(),
                None,
                None,
                headers,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(e, StreamableHttpError::ReservedHeaderConflict(_)),
            "{e:?}"
        );
    }

    #[test]
    fn event_boundaries_are_found_across_chunks_and_line_endings() {
        // A blank line split across chunks, with CRLF terminators: each event is small.
        let mut b = EventBoundary::default();
        assert_eq!(b.feed(b"data: aaaa\r\n"), 11);
        assert_eq!(b.feed(b"\r\ndata: bb"), 8);
        // `\n` + `\n` split across chunks.
        let mut b = EventBoundary::default();
        b.feed(b"data: x\n");
        assert_eq!(b.feed(b"\ndata: y"), 7);
        // Bare `\r` terminators.
        let mut b = EventBoundary::default();
        assert_eq!(b.feed(b"data: x\r\rdata: yy"), 8);
    }

    #[tokio::test]
    async fn a_small_events_stream_split_with_crlf_is_not_mistaken_for_an_oversized_one() {
        use futures::StreamExt as _;
        let source = futures::stream::iter([
            Ok::<_, reqwest::Error>(bytes::Bytes::from_static(b"data: aaaaaaaaaa\r\n")),
            Ok(bytes::Bytes::from_static(b"\r\ndata: bbbbbbbbbb\r\n")),
            Ok(bytes::Bytes::from_static(b"\r\ndata: cccccccccc\r\n\r\n")),
        ]);
        let mut bounded = std::pin::pin!(bounded(source, 24));
        while let Some(chunk) = bounded.next().await {
            assert!(chunk.is_ok(), "{chunk:?}");
        }
    }

    #[tokio::test]
    async fn a_json_body_over_the_cap_is_refused() {
        agent_core::ensure_provider();
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 64\r\nConnection: close\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"skills\":[]},\"pad\":\"xxxxxxxx\"}",
            false,
        )
        .await;
        let e = HttpClient(reqwest::Client::new())
            .post_message_with_max_sse_event_size(
                url.into(),
                skills_list(),
                None,
                None,
                HashMap::new(),
                16,
            )
            .await
            .unwrap_err();
        assert!(format!("{e}").contains("maximum size of 16 bytes"), "{e}");
    }

    #[tokio::test]
    async fn an_oversized_sse_event_ends_the_stream_with_an_error() {
        use futures::StreamExt as _;
        let source = futures::stream::iter([
            Ok::<_, reqwest::Error>(bytes::Bytes::from_static(b"data: aaaaaaaa")),
            Ok(bytes::Bytes::from_static(b"aaaaaaaaaaa")),
        ]);
        let mut bounded = std::pin::pin!(bounded(source, 16));
        assert!(bounded.next().await.unwrap().is_ok());
        assert!(bounded.next().await.unwrap().is_err());
    }
}
