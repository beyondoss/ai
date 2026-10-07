//! The streamable-HTTP half of keeping extension results intact on their way through `rmcp`.
//!
//! `rmcp` (3.2 through at least 3.5.1) decodes every server result into an *untagged* union, and
//! `CallToolResult` matches any object carrying `_meta` — which every `2026-07-28` server puts on
//! every result — so a `skills/list` answer would arrive as an empty tool result (upstream:
//! modelcontextprotocol/rust-sdk#1197). The workaround is one boundary,
//! [`crate::tools::mcp_stdio::rescue`] / [`crate::tools::mcp_stdio::unwrap_rescued`], applied to the
//! raw bytes before `rmcp` parses them. For stdio that is the one stdio transport
//! ([`crate::tools::mcp_stdio`]); for streamable HTTP it is [`HttpClient`] here, which wraps the
//! `reqwest` client `rmcp` drives and answers every POST itself (with the exact URI, session, auth
//! and standard headers the transport handed it). The same request path
//! ([`HttpClient::post_bounded`]) serves `mcp_view_http`'s capped MCP App view reads.
//!
//! It is also where every streamable-HTTP message is held to the one per-message cap
//! ([`crate::tools::mcp_stdio::max_message_bytes`], `BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES`): a JSON
//! body or SSE event over it answers its request with an error instead of being buffered whole, and
//! the GET stream is capped through `rmcp`'s own SSE-event limit.
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

/// The `reqwest` client `rmcp`'s streamable-HTTP transport drives, answering **every** POST itself
/// ([`HttpClient::post_bounded`]; rmcp's own `post_message` is never called) so results are
/// [rescued](crate::tools::mcp_stdio::rescue) and size-capped before `rmcp` parses them, and so a
/// 401's status is always visible: rmcp turns a 401 whose body is a JSON-RPC error into an ordinary
/// error *response*, the status lost before `mcp_oauth` could see it and refresh. Here any 401 is
/// `AuthRequired`, for every server.
#[derive(Clone)]
pub(crate) struct HttpClient {
    pub(crate) client: reqwest::Client,
}

impl HttpClient {
    #[cfg(test)]
    pub(crate) fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

/// The limit for an ordinary POST: the per-message cap (no larger than `max`, when the transport
/// names one); over it, a request is answered with a JSON-RPC error (its caller sees an ordinary
/// failed request), anything else fails as an undeliverable message does.
fn ordinary_limit(message: &ClientJsonRpcMessage, max: usize) -> Limit {
    // Only the id is needed — read straight off the request, not by serializing the message.
    let id = match message {
        ClientJsonRpcMessage::Request(request) => serde_json::to_value(&request.id).ok(),
        _ => None,
    };
    Limit {
        max: max.min(crate::tools::mcp_stdio::max_message_bytes()),
        over: id.map_or(OverLimit::Fail, OverLimit::Refuse),
    }
}

/// Headers a configured custom header may not override (`rmcp`'s own reserved set).
const RESERVED_HEADERS: [&str; 3] = ["accept", "mcp-session-id", "last-event-id"];

/// Bound an SSE byte stream: an event (bytes since the last blank line) larger than `max` ends the
/// stream with an error rather than buffering without limit, and sets `over`.
fn bounded<S>(
    stream: S,
    max: usize,
    over: Arc<std::sync::atomic::AtomicBool>,
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
            over.store(true, std::sync::atomic::Ordering::Relaxed);
            return std::future::ready(Some(Err(std::io::Error::other(format!(
                "an SSE event exceeded the maximum size of {max} bytes"
            )))));
        }
        std::future::ready(Some(Ok(chunk)))
    })
}

/// How large a response [`HttpClient::post_bounded`] reads, and what one larger becomes.
pub(crate) struct Limit {
    /// The largest JSON body, or SSE event, read.
    pub(crate) max: usize,
    pub(crate) over: OverLimit,
}

/// What a response over its size limit becomes.
#[derive(Clone)]
pub(crate) enum OverLimit {
    /// A transport error: the request fails as any undeliverable one does.
    Fail,
    /// A JSON-RPC error answering the request with this id, so the caller sees an ordinary failed
    /// request (an MCP App view read, which then falls back to text).
    Refuse(Value),
}

/// The JSON-RPC error a refused response is replaced with.
fn refusal(id: &Value, max: usize) -> Result<ServerJsonRpcMessage, serde_json::Error> {
    serde_json::from_value(serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32000,
            "message": format!("MCP message over {max} bytes refused by the host"),
        },
    }))
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
/// limit by sending a huge (or endless) JSON body. `None` once it passes `max` (the rest unread).
async fn capped_body(
    response: reqwest::Response,
    max: usize,
) -> Result<Option<String>, StreamableHttpError<reqwest::Error>> {
    use futures::StreamExt as _;
    let mut body = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(StreamableHttpError::Client)?;
        if body.len() + chunk.len() > max {
            return Ok(None);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Some(String::from_utf8_lossy(&body).into_owned()))
}

impl HttpClient {
    /// One POST this module answers itself — a `skills/*` request, or an MCP App view read
    /// (`mcp_view_http`) — built and its status interpreted the way `rmcp`'s own `reqwest` client
    /// does: reserved headers refused, 401/403 with a `WWW-Authenticate` surfaced as the
    /// `AuthRequired`/`InsufficientScope` errors `rmcp` and its callers act on, a 404 on a session as
    /// `SessionExpired` so it re-initializes. The result is rescued before `rmcp` parses it. An SSE
    /// response is handed back **as a stream**, each event rescued as it arrives: `rmcp` reads it
    /// incrementally, routes every server→client message on it (progress, requests, logs), and stops
    /// at the response, so a server that keeps the stream open costs nothing.
    ///
    /// Nothing is buffered past `limit.max`: a JSON body (refused up front by its `Content-Length`
    /// when it declares one) or one SSE event over it becomes what `limit.over` says.
    pub(crate) async fn post_bounded(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        limit: Limit,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<reqwest::Error>> {
        use futures::StreamExt as _;
        let Limit { max, over } = limit;
        let mut request = self
            .client
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
        // Any 401 — with or without a challenge, whatever its body — is the server refusing the
        // credentials, decided here from the status before a body could be read as a response.
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(StreamableHttpError::AuthRequired(
                rmcp::transport::streamable_http_client::AuthRequiredError::new(
                    www_authenticate.unwrap_or_default(),
                ),
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
        let is_request = matches!(message, ClientJsonRpcMessage::Request(_));
        // As rmcp's client: an empty success for a notification or a reply is an acceptance.
        if status.is_success() && !is_request && response.content_length() == Some(0) {
            return Ok(StreamableHttpPostResponse::Accepted);
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
            let overflowed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let events = sse_stream::SseStream::from_bytes_stream(bounded(
                response.bytes_stream(),
                max,
                overflowed.clone(),
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
            });
            let events = match over {
                OverLimit::Fail => events.boxed(),
                // The event over the limit becomes the refusal, and the stream ends there.
                OverLimit::Refuse(id) => {
                    let refused = sse_stream::Sse {
                        event: None,
                        data: Some(serde_json::to_string(&refusal(&id, max)?)?),
                        id: None,
                        retry: None,
                    };
                    events
                        .scan(false, move |ended, event| {
                            if *ended {
                                return std::future::ready(None);
                            }
                            if event.is_err()
                                && overflowed.load(std::sync::atomic::Ordering::Relaxed)
                            {
                                *ended = true;
                                tracing::warn!(max, "refused an MCP message streamed over its cap");
                                return std::future::ready(Some(Ok(refused.clone())));
                            }
                            std::future::ready(Some(event))
                        })
                        .boxed()
                }
            };
            return Ok(StreamableHttpPostResponse::Sse(events, session));
        }
        let declared_over = response
            .content_length()
            .is_some_and(|len| usize::try_from(len).map_or(true, |len| len > max));
        let body = if declared_over {
            None
        } else {
            capped_body(response, max).await?
        };
        let Some(body) = body else {
            return match over {
                OverLimit::Fail => Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                    format!("a response body exceeded the maximum size of {max} bytes"),
                ))),
                OverLimit::Refuse(id) => {
                    tracing::warn!(max, "refused an MCP message over its cap");
                    Ok(StreamableHttpPostResponse::Json(
                        refusal(&id, max)?,
                        session,
                    ))
                }
            };
        };
        let body = match crate::tools::mcp_stdio::rescue(body.as_bytes()) {
            Some(rescued) => String::from_utf8_lossy(&rescued).into_owned(),
            None => body,
        };
        let parsed = serde_json::from_str::<Value>(&body).ok();
        let Some(value) = parsed.filter(|v| v.get("result").is_some() || v.get("error").is_some())
        else {
            // As rmcp's client: a JSON success that is not a JSON-RPC message, for a notification
            // or a reply, is an acceptance. For a *request* rmcp would call it accepted too and then
            // wait for an answer that never comes, until the request times out; a request must
            // be answered with its response (or an SSE stream carrying it), so here it fails at
            // once instead. A 4xx to `server/discover` is the legacy server's way of saying it
            // has none.
            let json = content_type
                .as_deref()
                .is_some_and(|ct| ct.starts_with("application/json"));
            if status.is_success() && !is_request && json {
                return Ok(StreamableHttpPostResponse::Accepted);
            }
            if status.is_client_error() && !session_was_attached && is_discover(&message) {
                return Ok(StreamableHttpPostResponse::Json(
                    discover_rejected(&message, status, &body)?,
                    None,
                ));
            }
            return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
                format!("HTTP {status}: {body}"),
            )));
        };
        let message: ServerJsonRpcMessage = serde_json::from_value(value)?;
        Ok(StreamableHttpPostResponse::Json(message, session))
    }
}

fn is_discover(message: &ClientJsonRpcMessage) -> bool {
    matches!(
        message,
        ClientJsonRpcMessage::Request(r)
            if matches!(r.request, rmcp::model::ClientRequest::DiscoverRequest(_))
    )
}

/// rmcp's answer to a legacy server's 4xx for `server/discover`: an `invalid_request` error for its
/// id, which sends the handshake on to `initialize`.
fn discover_rejected(
    message: &ClientJsonRpcMessage,
    status: reqwest::StatusCode,
    body: &str,
) -> Result<ServerJsonRpcMessage, serde_json::Error> {
    let id = serde_json::to_value(message)?
        .get("id")
        .cloned()
        .unwrap_or(Value::Null);
    serde_json::from_value(serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32600,
            "message": format!("server/discover rejected with HTTP {status}: {body}"),
        },
    }))
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
        let limit = ordinary_limit(&message, usize::MAX);
        self.post_bounded(uri, message, session_id, auth_header, custom_headers, limit)
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
        let limit = ordinary_limit(&message, max_sse_event_size);
        self.post_bounded(uri, message, session_id, auth_header, custom_headers, limit)
            .await
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        self.client
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
        self.client
            .get_stream_with_max_sse_event_size(
                uri,
                session_id,
                last_event_id,
                auth_header,
                custom_headers,
                crate::tools::mcp_stdio::max_message_bytes(),
            )
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
        self.client
            .get_stream_with_max_sse_event_size(
                uri,
                session_id,
                last_event_id,
                auth_header,
                custom_headers,
                max_sse_event_size.min(crate::tools::mcp_stdio::max_message_bytes()),
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
        HttpClient::new(reqwest::Client::new())
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

    /// Every POST now goes through [`HttpClient::post_bounded`] instead of `rmcp`'s own
    /// `reqwest` client — so for an ordinary request (a tool call) the two must agree on every
    /// response shape `rmcp` acts on: auth required, insufficient scope, an expired session,
    /// accepted, a JSON answer (with its `Mcp-Session-Id`), and an SSE stream. One divergence is
    /// deliberate, and pinned last: a 401 with no challenge is still `AuthRequired` here.
    #[tokio::test]
    async fn an_ordinary_request_is_answered_as_rmcps_own_client_would() {
        use futures::StreamExt as _;
        agent_core::ensure_provider();
        let call = || -> ClientJsonRpcMessage {
            serde_json::from_value(serde_json::json!({
                "jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": { "name": "x" },
            }))
            .unwrap()
        };
        let shape =
            |r: &Result<StreamableHttpPostResponse, StreamableHttpError<reqwest::Error>>| match r {
                Ok(StreamableHttpPostResponse::Accepted) => "accepted".to_owned(),
                Ok(StreamableHttpPostResponse::Json(m, session)) => format!(
                    "json {} session={session:?}",
                    serde_json::to_value(m).unwrap()["id"]
                ),
                Ok(StreamableHttpPostResponse::Sse(_, session)) => {
                    format!("sse session={session:?}")
                }
                Err(StreamableHttpError::AuthRequired(_)) => "auth-required".to_owned(),
                Err(StreamableHttpError::InsufficientScope(_)) => "insufficient-scope".to_owned(),
                Err(StreamableHttpError::SessionExpired) => "session-expired".to_owned(),
                Err(e) => format!("other: {e}"),
                Ok(_) => "other ok".to_owned(),
            };
        let cases: [(&[u8], Option<&str>); 6] = [
            (b"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm=\"x\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", None),
            (b"HTTP/1.1 403 Forbidden\r\nWWW-Authenticate: Bearer scope=\"tools\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", None),
            (b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", Some("s-1")),
            (b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", None),
            (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: s-9\r\nContent-Length: 48\r\nConnection: close\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"content\":[]}}", None),
            (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nMcp-Session-Id: s-9\r\nConnection: close\r\n\r\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"content\":[]}}\n\n", None),
        ];
        for (raw, session) in cases {
            let ours = HttpClient::new(reqwest::Client::new())
                .post_message(
                    canned(raw, false).await.into(),
                    call(),
                    session.map(Into::into),
                    None,
                    HashMap::new(),
                )
                .await;
            let theirs = reqwest::Client::new()
                .post_message(
                    canned(raw, false).await.into(),
                    call(),
                    session.map(Into::into),
                    None,
                    HashMap::new(),
                )
                .await;
            assert_eq!(
                shape(&ours),
                shape(&theirs),
                "for {}",
                String::from_utf8_lossy(&raw[..raw.len().min(40)])
            );
            // An SSE answer streams the same response through both.
            if let (
                Ok(StreamableHttpPostResponse::Sse(mut a, _)),
                Ok(StreamableHttpPostResponse::Sse(mut b, _)),
            ) = (ours, theirs)
            {
                assert_eq!(
                    a.next().await.unwrap().unwrap().data,
                    b.next().await.unwrap().unwrap().data
                );
            }
        }
        // The deliberate divergence: a 401 with no `WWW-Authenticate` whose body is a JSON-RPC
        // error. rmcp hands that back as an ordinary error *response*, losing the status; here any
        // 401 is `AuthRequired`, so `mcp_oauth` can refresh (and any server's caller sees why).
        let raw: &[u8] = b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: 69\r\nConnection: close\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":7,\"error\":{\"code\":-32001,\"message\":\"no token\"}}";
        let ours = HttpClient::new(reqwest::Client::new())
            .post_message(
                canned(raw, false).await.into(),
                call(),
                None,
                None,
                HashMap::new(),
            )
            .await;
        let theirs = reqwest::Client::new()
            .post_message(
                canned(raw, false).await.into(),
                call(),
                None,
                None,
                HashMap::new(),
            )
            .await;
        assert_eq!(shape(&ours), "auth-required");
        assert_ne!(
            shape(&theirs),
            "auth-required",
            "rmcp's own client: {}",
            shape(&theirs)
        );
    }

    /// …and they send the same request: the bearer token, the session id, `Accept`, the protocol
    /// version and custom headers, and the body — so a server's auth and session handling sees
    /// no difference.
    #[tokio::test]
    async fn an_ordinary_request_is_sent_as_rmcps_own_client_would() {
        agent_core::ensure_provider();
        async fn capture() -> (String, tokio::sync::oneshot::Receiver<String>) {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let (tx, rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut got = Vec::new();
                let mut buf = [0u8; 8192];
                loop {
                    let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                        .await
                        .unwrap();
                    got.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&got).into_owned();
                    if let Some((head, body)) = text.split_once("\r\n\r\n") {
                        let len: usize = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse().unwrap())
                            })
                            .unwrap_or(0);
                        if body.len() >= len || n == 0 {
                            let _ = tx.send(text);
                            break;
                        }
                    }
                }
                let _ = tokio::io::AsyncWriteExt::write_all(
                    &mut stream,
                    b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await;
            });
            (format!("http://{addr}/mcp"), rx)
        }
        // The request line, the headers a server acts on (names lowercased, sorted), and the body.
        let normalize = |raw: String| {
            let (head, body) = raw.split_once("\r\n\r\n").unwrap();
            let mut lines = head.lines();
            let request_line = lines.next().unwrap().to_owned();
            let mut headers: Vec<String> = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| format!("{}: {}", k.to_ascii_lowercase(), v.trim()))
                .filter(|l| !l.starts_with("host:"))
                .collect();
            headers.sort();
            (request_line, headers, body.to_owned())
        };
        let message = || -> ClientJsonRpcMessage {
            serde_json::from_value(json!({
                "jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": { "name": "x" },
            }))
            .unwrap()
        };
        let headers = || {
            HashMap::from([
                (
                    HeaderName::from_static("mcp-protocol-version"),
                    HeaderValue::from_static("2025-11-25"),
                ),
                (
                    HeaderName::from_static("x-tenant"),
                    HeaderValue::from_static("t-1"),
                ),
            ])
        };
        let (url, ours) = capture().await;
        HttpClient::new(reqwest::Client::new())
            .post_message(
                url.into(),
                message(),
                Some("s-1".into()),
                Some("tok".into()),
                headers(),
            )
            .await
            .unwrap();
        let (url, theirs) = capture().await;
        reqwest::Client::new()
            .post_message(
                url.into(),
                message(),
                Some("s-1".into()),
                Some("tok".into()),
                headers(),
            )
            .await
            .unwrap();
        let (ours, theirs) = (
            normalize(ours.await.unwrap()),
            normalize(theirs.await.unwrap()),
        );
        assert_eq!(ours, theirs);
        for wanted in [
            "authorization: Bearer tok",
            "mcp-session-id: s-1",
            "x-tenant: t-1",
        ] {
            assert!(
                ours.1.iter().any(|h| h == wanted),
                "{wanted} in {:?}",
                ours.1
            );
        }
    }

    #[tokio::test]
    async fn a_reserved_custom_header_is_refused() {
        agent_core::ensure_provider();
        let mut headers = HashMap::new();
        headers.insert(
            HeaderName::from_static("mcp-session-id"),
            HeaderValue::from_static("forged"),
        );
        let e = HttpClient::new(reqwest::Client::new())
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
        let mut bounded = std::pin::pin!(bounded(source, 24, Default::default()));
        while let Some(chunk) = bounded.next().await {
            assert!(chunk.is_ok(), "{chunk:?}");
        }
    }

    #[tokio::test]
    async fn a_json_success_that_is_not_json_rpc_is_accepted_for_a_notification_but_fails_a_request()
     {
        agent_core::ensure_provider();
        let oauth = HttpClient::new(reqwest::Client::new());
        let not_json_rpc: &'static [u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}";
        // A notification: accepted, as rmcp's client does.
        let notification: ClientJsonRpcMessage = serde_json::from_value(
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
        )
        .unwrap();
        let url = canned(not_json_rpc, false).await;
        let got = oauth
            .post_message(url.into(), notification, None, None, HashMap::new())
            .await
            .unwrap();
        assert!(
            matches!(got, StreamableHttpPostResponse::Accepted),
            "{got:?}"
        );
        // A request: rmcp would also call it accepted, then wait out the request's whole timeout
        // for a response that is not coming. Stricter on purpose: it fails now, naming the body.
        let url = canned(not_json_rpc, false).await;
        let e = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            oauth.post_message(url.into(), skills_list(), None, None, HashMap::new()),
        )
        .await
        .expect("answered at once, not left to time out")
        .unwrap_err();
        assert!(format!("{e}").contains("{\"ok\":true}"), "{e}");
        // Not JSON at all, for a notification: an error, as rmcp's unexpected-content-type is.
        let notification: ClientJsonRpcMessage = serde_json::from_value(
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
        )
        .unwrap();
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 6\r\nConnection: close\r\n\r\n<html>",
            false,
        )
        .await;
        assert!(
            oauth
                .post_message(url.into(), notification, None, None, HashMap::new())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn a_json_body_over_the_cap_answers_the_request_with_an_error() {
        agent_core::ensure_provider();
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 64\r\nConnection: close\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"skills\":[]},\"pad\":\"xxxxxxxx\"}",
            false,
        )
        .await;
        let answer = HttpClient::new(reqwest::Client::new())
            .post_message_with_max_sse_event_size(
                url.into(),
                skills_list(),
                None,
                None,
                HashMap::new(),
                16,
            )
            .await
            .unwrap();
        let StreamableHttpPostResponse::Json(message, _) = answer else {
            panic!("an over-cap answer to a request is that request's error");
        };
        let text = serde_json::to_string(&message).unwrap();
        assert!(text.contains("over 16 bytes refused"), "{text}");
    }

    /// Every POST — not only `skills/*` — is held to the one per-message cap: a tool call's over-cap
    /// answer comes back as that call's error, not read whole.
    #[tokio::test]
    async fn any_request_is_held_to_the_cap() {
        agent_core::ensure_provider();
        let url = canned(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 64\r\nConnection: close\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"content\":[]},\"pad\":\"xxxxxxx\"}",
            false,
        )
        .await;
        let call: ClientJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": { "name": "x" },
        }))
        .unwrap();
        let answer = HttpClient::new(reqwest::Client::new())
            .post_message_with_max_sse_event_size(url.into(), call, None, None, HashMap::new(), 16)
            .await
            .unwrap();
        let StreamableHttpPostResponse::Json(message, _) = answer else {
            panic!("an over-cap answer to a tool call is that call's error");
        };
        assert!(serde_json::to_string(&message).unwrap().contains("refused"));
    }

    #[tokio::test]
    async fn an_oversized_sse_event_ends_the_stream_with_an_error() {
        use futures::StreamExt as _;
        let source = futures::stream::iter([
            Ok::<_, reqwest::Error>(bytes::Bytes::from_static(b"data: aaaaaaaa")),
            Ok(bytes::Bytes::from_static(b"aaaaaaaaaaa")),
        ]);
        let mut bounded = std::pin::pin!(bounded(source, 16, Default::default()));
        assert!(bounded.next().await.unwrap().is_ok());
        assert!(bounded.next().await.unwrap().is_err());
    }
}
