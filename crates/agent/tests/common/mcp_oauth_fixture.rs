//! A real OAuth-protected streamable-HTTP MCP server, and the `agent mcp-login` steps that log in to
//! it, shared by the `mcp_oauth*` suites. Hand-rolled — RFC 8414 metadata, RFC 7591 dynamic client
//! registration, an authorize endpoint that "approves" at once, a token endpoint for both the
//! `authorization_code` and `refresh_token` grants — not a mock of the protocol.
//!
//! Controls ([`OAuthFixture`]) let a test make a token go bad **mid-session** (every issued token
//! revoked after the Nth `tools/call`), make the next refresh fail (`invalid_grant`), hold rejected
//! calls until several have arrived (so concurrent 401s are really concurrent), and count refresh
//! grants and rejected calls.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};

use super::SpawnGuarded;

/// Write `$HOME/.claude/settings.json` with these `mcp_servers`.
pub fn write_global_settings(home: &Path, mcp_servers: Value) {
    let claude_dir = home.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": mcp_servers })).unwrap(),
    )
    .unwrap();
}

/// A parsed raw HTTP request: method, path (no query string), the full raw query string, headers
/// (lower-cased names), and body bytes.
pub struct ParsedRequest {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

pub fn read_request(stream: &mut TcpStream) -> Option<ParsedRequest> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let header_end = loop {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let full_path = parts.next().unwrap_or_default().to_string();
    let (path, query) = full_path
        .split_once('?')
        .map(|(p, q)| (p.to_string(), q.to_string()))
        .unwrap_or((full_path, String::new()));

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = buf[body_start..buf.len().min(body_start + content_length)].to_vec();
    Some(ParsedRequest {
        method,
        path,
        query,
        headers,
        body,
    })
}

pub fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == name).then(|| urlencoding_decode(v))
    })
}

pub fn form_param(body: &[u8], name: &str) -> Option<String> {
    query_param(&String::from_utf8_lossy(body), name)
}

/// Minimal `application/x-www-form-urlencoded` value decoder — `+` for space, `%XX` escapes. Good
/// enough for the plain ASCII values (codes, tokens, grant types) this fixture ever needs to decode.
fn urlencoding_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            '+' => out.push(' '),
            '%' => {
                let hex: String = chars.by_ref().take(2).collect();
                if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                    out.push(byte as char);
                } else {
                    out.push('%');
                    out.push_str(&hex);
                }
            }
            other => out.push(other),
        }
    }
    out
}

pub fn write_response(stream: &mut TcpStream, status: &str, extra_headers: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

pub fn write_json(stream: &mut TcpStream, status: &str, value: &Value) {
    let body = serde_json::to_vec(value).unwrap();
    write_response(stream, status, "Content-Type: application/json\r\n", &body);
}

/// One request's `(Authorization, Mcp-Session-Id)`.
pub type HeaderPair = (Option<String>, Option<String>);

/// A running OAuth-protected MCP server and its controls. See the module doc.
#[derive(Clone)]
pub struct OAuthFixture {
    /// The MCP endpoint (`http://127.0.0.1:<port>/mcp`).
    pub url: String,
    /// Every `Authorization` header an *accepted* MCP request carried.
    pub seen_auth_headers: Arc<Mutex<Vec<String>>>,
    /// `refresh_token` grants the token endpoint has answered (successfully or not).
    pub refresh_grants: Arc<AtomicU32>,
    /// `tools/call` requests rejected with 401.
    pub rejected_calls: Arc<AtomicU32>,
    /// Revoke every issued token right after this many `tools/call`s have been answered (0: never).
    pub revoke_after_calls: Arc<AtomicU32>,
    /// Answer `refresh_token` grants with `400 invalid_grant`.
    pub fail_refresh: Arc<AtomicBool>,
    /// Answer `refresh_token` grants with this `(status line, body)` instead.
    pub refresh_reply: Arc<Mutex<Option<(String, String)>>>,
    /// Hold each rejected `tools/call` until this many have arrived (or 5 s pass), so concurrent
    /// requests with a stale token are all in flight before any is answered.
    pub hold_rejections_until: Arc<AtomicU32>,
    /// Answer a rejected request's 401 with an `application/json` JSON-RPC error body and no
    /// `WWW-Authenticate` challenge (what rmcp's own client reads as an ordinary error response).
    pub reject_with_json_body: Arc<AtomicBool>,
    /// Answer every `tools/call` 401, whatever token it carries — a login the server keeps
    /// refusing (a refresh gets a new token, and that is refused too).
    pub reject_calls: Arc<AtomicBool>,
    /// Speak sessions: `initialize` issues an `Mcp-Session-Id`, every later POST must carry a live
    /// one (`400` without, `404` for an expired one), `GET` opens the standalone SSE stream, and
    /// `DELETE` ends the session.
    pub sessions: Arc<AtomicBool>,
    /// Expire every live session right after this many `tools/call`s have been answered (0: never).
    pub expire_sessions_after_calls: Arc<AtomicU32>,
    /// Answer requests with `text/event-stream` (one `data:` event) instead of JSON.
    pub sse_responses: Arc<AtomicBool>,
    /// Answer `server/discover` with this `(status line, body)` — a legacy server's 4xx.
    pub discover_reply: Arc<Mutex<Option<(String, String)>>>,
    /// Answer notifications with an empty `200 OK` (no content type) instead of `202 Accepted`.
    pub notifications_empty_200: Arc<AtomicBool>,
    /// Never answer a `DELETE` (hold the connection 30 s).
    pub hang_deletes: Arc<AtomicBool>,
    /// What arrived: `initialize`s, `GET` streams, `DELETE`s, POSTs refused for a missing or
    /// expired session, and the `Mcp-Session-Id` each `tools/call` carried.
    pub initializes: Arc<AtomicU32>,
    pub get_streams: Arc<AtomicU32>,
    pub deletes: Arc<AtomicU32>,
    pub session_refusals: Arc<AtomicU32>,
    pub call_sessions: Arc<Mutex<Vec<Option<String>>>>,
    /// Each `DELETE`'s `(Authorization, Mcp-Session-Id)`.
    pub delete_requests: Arc<Mutex<Vec<HeaderPair>>>,
    /// The client's answers to server→client requests (JSON-RPC responses POSTed back), by id.
    pub client_answers: Arc<Mutex<Vec<Value>>>,
    issued: Arc<Mutex<HashSet<String>>>,
    live_sessions: Arc<Mutex<HashSet<String>>>,
}

impl OAuthFixture {
    /// Revoke every token issued so far: the next request with one is answered 401.
    pub fn revoke_all(&self) {
        self.issued.lock().unwrap().clear();
    }

    /// Accept `token` as if issued: for a server configured with a static `Authorization` header
    /// rather than a login.
    pub fn issue(&self, token: &str) {
        self.issued.lock().unwrap().insert(token.to_owned());
    }

    /// Start one. `expires_in_secs` is the lifetime the fixture reports for each token it issues.
    /// `rmcp`'s `AuthorizationManager::get_access_token` refreshes proactively whenever fewer than 30
    /// seconds remain, so anything below 30 is refreshed on the very next read without any waiting;
    /// a test that wants a token to stay put must pass something comfortably above it.
    pub fn spawn(expires_in_secs: u64) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let fixture = Self {
            url: format!("{base}/mcp"),
            seen_auth_headers: Arc::default(),
            refresh_grants: Arc::default(),
            rejected_calls: Arc::default(),
            revoke_after_calls: Arc::default(),
            fail_refresh: Arc::default(),
            refresh_reply: Arc::default(),
            hold_rejections_until: Arc::default(),
            reject_with_json_body: Arc::default(),
            reject_calls: Arc::default(),
            sessions: Arc::default(),
            expire_sessions_after_calls: Arc::default(),
            sse_responses: Arc::default(),
            discover_reply: Arc::default(),
            notifications_empty_200: Arc::default(),
            hang_deletes: Arc::default(),
            initializes: Arc::default(),
            get_streams: Arc::default(),
            deletes: Arc::default(),
            session_refusals: Arc::default(),
            call_sessions: Arc::default(),
            delete_requests: Arc::default(),
            client_answers: Arc::default(),
            issued: Arc::default(),
            live_sessions: Arc::default(),
        };
        let state = Arc::new(Shared {
            fixture: fixture.clone(),
            base,
            expires_in_secs,
            tokens: AtomicU32::new(0),
            calls: AtomicU32::new(0),
            gate: (Mutex::new(0), Condvar::new()),
        });
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let state = state.clone();
                thread::spawn(move || state.serve(stream));
            }
        });
        fixture
    }
}

struct Shared {
    fixture: OAuthFixture,
    base: String,
    expires_in_secs: u64,
    tokens: AtomicU32,
    calls: AtomicU32,
    /// Rejected calls arrived so far, for `hold_rejections_until`.
    gate: (Mutex<u32>, Condvar),
}

impl Shared {
    fn serve(&self, mut stream: TcpStream) {
        let Some(req) = read_request(&mut stream) else {
            return;
        };
        let base = &self.base;
        let f = &self.fixture;
        if req
            .path
            .starts_with("/.well-known/oauth-protected-resource")
        {
            // SEP-985 protected-resource metadata: points at this fixture's own AS so rmcp 3.x
            // discovers the correct issuer (`http://host`) instead of expecting the resource URL.
            return write_json(
                &mut stream,
                "200 OK",
                &json!({
                    "resource": format!("{base}/mcp"),
                    "authorization_servers": [base],
                    "scopes_supported": ["mcp"],
                }),
            );
        }
        if req
            .path
            .starts_with("/.well-known/oauth-authorization-server")
            || req.path.starts_with("/.well-known/openid-configuration")
        {
            return write_json(
                &mut stream,
                "200 OK",
                &json!({
                    "issuer": base,
                    "authorization_endpoint": format!("{base}/authorize"),
                    "token_endpoint": format!("{base}/token"),
                    "registration_endpoint": format!("{base}/register"),
                    "scopes_supported": ["mcp"],
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                }),
            );
        }
        if req.path == "/register" && req.method == "POST" {
            let request: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            let redirect_uris = request
                .get("redirect_uris")
                .cloned()
                .unwrap_or_else(|| json!([]));
            return write_json(
                &mut stream,
                "201 Created",
                &json!({
                    "client_id": "test-dynamically-registered-client",
                    "redirect_uris": redirect_uris,
                }),
            );
        }
        if req.path == "/authorize" && req.method == "GET" {
            let redirect_uri = query_param(&req.query, "redirect_uri").unwrap_or_default();
            let state = query_param(&req.query, "state").unwrap_or_default();
            let sep = if redirect_uri.contains('?') { "&" } else { "?" };
            let location = format!("{redirect_uri}{sep}code=test-authorization-code&state={state}");
            return write_response(
                &mut stream,
                "302 Found",
                &format!("Location: {location}\r\n"),
                b"",
            );
        }
        if req.path == "/token" && req.method == "POST" {
            let grant_type = form_param(&req.body, "grant_type").unwrap_or_default();
            if grant_type == "refresh_token" {
                f.refresh_grants.fetch_add(1, Ordering::SeqCst);
                if f.fail_refresh.load(Ordering::SeqCst) {
                    return write_json(
                        &mut stream,
                        "400 Bad Request",
                        &json!({ "error": "invalid_grant", "error_description": "revoked" }),
                    );
                }
                if let Some((status, body)) = f.refresh_reply.lock().unwrap().clone() {
                    return write_response(
                        &mut stream,
                        &status,
                        "Content-Type: application/json\r\n",
                        body.as_bytes(),
                    );
                }
            } else if grant_type != "authorization_code" {
                return write_json(
                    &mut stream,
                    "400 Bad Request",
                    &json!({ "error": "unsupported_grant_type" }),
                );
            }
            let n = self.tokens.fetch_add(1, Ordering::SeqCst);
            let access_token = format!("access-token-{n}");
            f.issued.lock().unwrap().insert(access_token.clone());
            return write_json(
                &mut stream,
                "200 OK",
                &json!({
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": self.expires_in_secs,
                    "refresh_token": "refresh-token-fixed",
                    "scope": "mcp",
                }),
            );
        }
        if req.path == "/mcp" && req.method == "POST" {
            return self.mcp(&mut stream, &req);
        }
        if req.path == "/mcp" && (req.method == "GET" || req.method == "DELETE") {
            return self.stream_or_delete(&mut stream, &req);
        }
        write_response(&mut stream, "404 Not Found", "", b"");
    }

    /// Whether a request carries a currently-valid bearer token.
    fn authorized(&self, req: &ParsedRequest) -> bool {
        req.headers
            .get("authorization")
            .and_then(|h| h.strip_prefix("Bearer "))
            .is_some_and(|token| self.fixture.issued.lock().unwrap().contains(token))
    }

    /// 401 for a request whose token is not valid: with a `WWW-Authenticate` challenge, or (when
    /// `reject_with_json_body`) an `application/json` JSON-RPC error answering `id` and no challenge.
    fn unauthorized(&self, stream: &mut TcpStream, id: &Value) {
        if self.fixture.reject_with_json_body.load(Ordering::SeqCst) {
            let body = serde_json::to_vec(&json!({ "jsonrpc": "2.0", "id": id,
                "error": { "code": -32001, "message": "unauthorized: token expired" } }))
            .unwrap();
            return write_response(
                stream,
                "401 Unauthorized",
                "Content-Type: application/json\r\n",
                &body,
            );
        }
        write_response(
            stream,
            "401 Unauthorized",
            &format!(
                "WWW-Authenticate: Bearer resource=\"{}/mcp\"\r\n",
                self.base
            ),
            b"",
        );
    }

    /// The standalone SSE stream (`GET`, held open a few seconds) and session end (`DELETE`), for a
    /// fixture that speaks sessions; `405` otherwise, as a session-less server answers.
    fn stream_or_delete(&self, stream: &mut TcpStream, req: &ParsedRequest) {
        let f = &self.fixture;
        if !f.sessions.load(Ordering::SeqCst) {
            return write_response(stream, "405 Method Not Allowed", "", b"");
        }
        if req.method == "DELETE" {
            f.delete_requests.lock().unwrap().push((
                req.headers.get("authorization").cloned(),
                req.headers.get("mcp-session-id").cloned(),
            ));
            if f.hang_deletes.load(Ordering::SeqCst) {
                // A server that never answers the session's end.
                thread::sleep(Duration::from_secs(30));
                return;
            }
        }
        if !self.authorized(req) {
            return self.unauthorized(stream, &Value::Null);
        }
        let live = req
            .headers
            .get("mcp-session-id")
            .is_some_and(|s| f.live_sessions.lock().unwrap().contains(s));
        if !live {
            return write_response(stream, "404 Not Found", "", b"");
        }
        if req.method == "DELETE" {
            f.deletes.fetch_add(1, Ordering::SeqCst);
            if let Some(s) = req.headers.get("mcp-session-id") {
                f.live_sessions.lock().unwrap().remove(s);
            }
            return write_response(stream, "200 OK", "", b"");
        }
        f.get_streams.fetch_add(1, Ordering::SeqCst);
        let _ = stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n: open\n\n",
        );
        let _ = stream.flush();
        thread::sleep(Duration::from_secs(5));
    }

    /// The protected resource: anything without a currently-valid bearer token is answered 401, as
    /// a real OAuth-gated MCP server does — before the session is looked at, as auth comes first.
    fn mcp(&self, stream: &mut TcpStream, req: &ParsedRequest) {
        let f = &self.fixture;
        let request: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let refused = method == "tools/call" && f.reject_calls.load(Ordering::SeqCst);
        if refused || !self.authorized(req) {
            if method == "tools/call" {
                f.rejected_calls.fetch_add(1, Ordering::SeqCst);
                let want = f.hold_rejections_until.load(Ordering::SeqCst);
                let (count, cv) = &self.gate;
                let mut arrived = count.lock().unwrap();
                *arrived += 1;
                cv.notify_all();
                let _ = cv
                    .wait_timeout_while(arrived, Duration::from_secs(5), |n| *n < want)
                    .unwrap();
            }
            return self.unauthorized(stream, &id);
        }
        if let Some(auth) = req.headers.get("authorization") {
            f.seen_auth_headers.lock().unwrap().push(auth.clone());
        }
        let sessions = f.sessions.load(Ordering::SeqCst);
        let session = req.headers.get("mcp-session-id").cloned();
        if method == "server/discover"
            && let Some((status, body)) = f.discover_reply.lock().unwrap().clone()
        {
            return write_response(
                stream,
                &status,
                "Content-Type: text/plain\r\n",
                body.as_bytes(),
            );
        }
        // Every POST but `initialize` (and the discovery probe before it) belongs to a session.
        if sessions && !matches!(method, "initialize" | "server/discover") {
            match &session {
                None => {
                    f.session_refusals.fetch_add(1, Ordering::SeqCst);
                    return write_response(
                        stream,
                        "400 Bad Request",
                        "",
                        b"missing Mcp-Session-Id",
                    );
                }
                Some(s) if !f.live_sessions.lock().unwrap().contains(s) => {
                    f.session_refusals.fetch_add(1, Ordering::SeqCst);
                    return write_response(stream, "404 Not Found", "", b"");
                }
                Some(_) => {}
            }
        }
        // The client answering a server→client request (no `method`, a `result` or `error`).
        if method.is_empty() && (request.get("result").is_some() || request.get("error").is_some())
        {
            f.client_answers.lock().unwrap().push(request.clone());
            let (_, cv) = &self.gate;
            cv.notify_all();
            return write_response(stream, "202 Accepted", "", b"");
        }
        if request.get("id").is_none() {
            return if f.notifications_empty_200.load(Ordering::SeqCst) {
                write_response(stream, "200 OK", "", b"")
            } else {
                write_response(stream, "202 Accepted", "", b"")
            };
        }
        // A pre-`2026-07-28` server: `server/discover` (and anything else unknown) is "method not
        // found", which is what sends rmcp to the legacy `initialize` handshake.
        if !matches!(method, "initialize" | "tools/list" | "tools/call") {
            return write_json(
                stream,
                "200 OK",
                &json!({ "jsonrpc": "2.0", "id": id,
                         "error": { "code": -32601, "message": format!("method not found: {method}") } }),
            );
        }
        let mut extra = String::new();
        let result = match method {
            "initialize" => {
                let n = f.initializes.fetch_add(1, Ordering::SeqCst);
                if sessions {
                    let id = format!("session-{n}");
                    f.live_sessions.lock().unwrap().insert(id.clone());
                    extra = format!("Mcp-Session-Id: {id}\r\n");
                }
                json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "mcp-fixture-oauth-server", "version": "0.0.0" },
                })
            }
            "tools/list" => json!({ "tools": [{
                "name": "echo",
                "description": "Echoes back its `text` argument.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"],
                },
            }, {
                "name": "ask_user",
                "description": "Asks the user for their name, then greets them.",
                "inputSchema": { "type": "object", "properties": {} },
            }] }),
            _ if request.pointer("/params/name").and_then(Value::as_str) == Some("ask_user") => {
                f.call_sessions.lock().unwrap().push(session.clone());
                return self.ask_user(stream, &request, &id, &extra);
            }
            _ => {
                f.call_sessions.lock().unwrap().push(session.clone());
                let text = request
                    .pointer("/params/arguments/text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                json!({ "content": [{ "type": "text", "text": text }], "isError": false })
            }
        };
        let message = json!({ "jsonrpc": "2.0", "id": id, "result": result });
        if f.sse_responses.load(Ordering::SeqCst) {
            let body = format!("data: {message}\n\n");
            write_response(
                stream,
                "200 OK",
                &format!("{extra}Content-Type: text/event-stream\r\n"),
                body.as_bytes(),
            );
        } else {
            let body = serde_json::to_vec(&message).unwrap();
            write_response(
                stream,
                "200 OK",
                &format!("{extra}Content-Type: application/json\r\n"),
                &body,
            );
        }
        if method == "tools/call" {
            let calls = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if calls == f.revoke_after_calls.load(Ordering::SeqCst) {
                f.revoke_all();
            }
            if calls == f.expire_sessions_after_calls.load(Ordering::SeqCst) {
                f.live_sessions.lock().unwrap().clear();
            }
        }
    }

    /// `ask_user`, answered the way a real streamable-HTTP server does it: an SSE stream on the
    /// call's own POST, **flushed event by event** — a progress notification, then an
    /// `elicitation/create` request to the client — and the response only once the client's answer
    /// to that request has arrived (POSTed back on another connection). A client that buffered the
    /// stream to its end would never see the question, and the call would hang to the 10 s cap.
    fn ask_user(&self, stream: &mut TcpStream, request: &Value, id: &Value, extra: &str) {
        let f = &self.fixture;
        let token = request.pointer("/params/_meta/progressToken").cloned();
        let ask_id = format!("srv-elicit-{}", self.calls.fetch_add(1, Ordering::SeqCst));
        let _ = stream.write_all(
            format!("HTTP/1.1 200 OK\r\n{extra}Content-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        );
        let mut event = |message: Value| {
            let _ = stream.write_all(format!("data: {message}\n\n").as_bytes());
            let _ = stream.flush();
        };
        if let Some(token) = token {
            event(
                json!({ "jsonrpc": "2.0", "method": "notifications/progress",
                "params": { "progressToken": token, "progress": 1, "total": 2,
                            "message": "asking the user" } }),
            );
        }
        thread::sleep(Duration::from_millis(50));
        event(
            json!({ "jsonrpc": "2.0", "id": ask_id, "method": "elicitation/create",
            "params": { "message": "What is your name?",
                        "requestedSchema": { "type": "object",
                            "properties": { "name": { "type": "string" } },
                            "required": ["name"] } } }),
        );
        // Wait for the answer, on another connection, before ending this stream.
        let answered =
            |answers: &Vec<Value>| answers.iter().find(|a| a["id"] == json!(ask_id)).cloned();
        let (count, cv) = &self.gate;
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut answer = answered(&f.client_answers.lock().unwrap());
        while answer.is_none() && std::time::Instant::now() < deadline {
            let guard = count.lock().unwrap();
            let _ = cv.wait_timeout(guard, Duration::from_millis(100)).unwrap();
            answer = answered(&f.client_answers.lock().unwrap());
        }
        let text = match answer {
            Some(a) if a["result"]["action"] == "accept" => format!(
                "hello-{}",
                a["result"]["content"]["name"].as_str().unwrap_or("?")
            ),
            Some(a) => format!("not answered: {}", a["result"]["action"]),
            None => "no answer before the stream had to end".to_owned(),
        };
        event(json!({ "jsonrpc": "2.0", "id": id,
            "result": { "content": [{ "type": "text", "text": text }], "isError": false } }));
    }
}

/// The original two-value form: the MCP URL and the accepted `Authorization` headers.
pub fn spawn_oauth_protected_mcp_fixture(
    expires_in_secs: u64,
) -> (String, Arc<Mutex<Vec<String>>>) {
    let f = OAuthFixture::spawn(expires_in_secs);
    (f.url, f.seen_auth_headers)
}

/// A tiny hand-rolled HTTP GET client that follows redirects — simulates "the user's browser visits
/// the authorization URL and is redirected through to the local callback" without pulling in
/// `reqwest`'s `blocking` feature (not otherwise needed anywhere in this crate) just for a test.
pub fn get_following_redirects(url: &str, max_redirects: u8) -> u16 {
    let mut current = url.to_string();
    for _ in 0..=max_redirects {
        let parsed = url::Url::parse(&current).expect("valid URL");
        let host = parsed.host_str().expect("host").to_string();
        let port = parsed.port_or_known_default().unwrap_or(80);
        let path_and_query = match parsed.query() {
            Some(q) => format!("{}?{q}", parsed.path()),
            None => parsed.path().to_string(),
        };
        let mut stream = TcpStream::connect((host.as_str(), port)).expect("connect");
        let request = format!(
            "GET {path_and_query} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        let text = String::from_utf8_lossy(&response);
        let status_line = text.lines().next().unwrap_or_default();
        let status: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if !(300..400).contains(&status) {
            return status;
        }
        let Some(location) = text
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("location:"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
        else {
            return status;
        };
        current = location;
    }
    0
}

/// Read `child`'s stderr line by line until one contains a bare `http://` URL (the exact line
/// `mcp-login` prints), returning it. Panics if the stream closes first — a clear, immediate test
/// failure rather than a hang.
pub fn read_printed_url(stderr: &mut BufReader<std::process::ChildStderr>) -> String {
    let mut line = String::new();
    loop {
        line.clear();
        let n = stderr.read_line(&mut line).unwrap();
        assert!(n != 0, "mcp-login's stderr closed before printing a URL");
        let trimmed = line.trim();
        if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            return trimmed.to_string();
        }
    }
}

/// Run `agent mcp-login <server>` with `$HOME` = `home` through the whole real flow (the browser
/// step simulated by following the printed URL's redirects to the callback), asserting success.
pub fn mcp_login(home: &Path, server: &str) {
    let mut login = Command::new(super::BIN)
        .args(["mcp-login", server])
        .env("HOME", home)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    assert_eq!(get_following_redirects(&auth_url, 3), 200);
    let output = login.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "mcp-login failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
/// A logged-in `$HOME` for a fresh fixture whose tokens stay valid for an hour unless revoked.
pub fn logged_in() -> (tempfile::TempDir, OAuthFixture) {
    let home = tempfile::tempdir().unwrap();
    let fixture = OAuthFixture::spawn(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": fixture.url, "headers": {} }]),
    );
    mcp_login(home.path(), "protected");
    (home, fixture)
}

pub fn echo(id: &str, text: &str) -> String {
    super::turn_tool_use(
        id,
        "mcp__protected__echo",
        &json!({ "text": text }).to_string(),
    )
}

/// One assistant turn calling `echo` once per text, all at once.
pub fn echoes(texts: &[&str]) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
    ];
    for (i, text) in texts.iter().enumerate() {
        events.push(json!({ "type": "content_block_start", "index": i, "content_block": { "type": "tool_use", "id": format!("toolu_p{i}"), "name": "mcp__protected__echo", "input": {} } }));
        events.push(json!({ "type": "content_block_delta", "index": i, "delta": { "type": "input_json_delta", "partial_json": json!({ "text": text }).to_string() } }));
        events.push(json!({ "type": "content_block_stop", "index": i }));
    }
    events.push(json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }));
    events.push(json!({ "type": "message_stop" }));
    super::sse(&events)
}

/// Run one `agent run` against `home` with the scripted model turns; the model request bodies.
pub fn run(home: &std::path::Path, turns: Vec<String>) -> Vec<String> {
    let (base, bodies) = super::spawn_model_server(turns);
    let cwd = tempfile::tempdir().unwrap();
    let out = super::run_cmd(super::BIN)
        .env("HOME", home)
        .args([
            "run",
            "call the protected echo tool",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "8",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    bodies.lock().unwrap().clone()
}

/// The tool results in a recorded model request (headers + body): `(tool_use_id, text, is_error)`.
pub fn tool_results(request: &str) -> Vec<(String, String, bool)> {
    let body = request.split_once("\r\n\r\n").map_or(request, |(_, b)| b);
    let v: Value = serde_json::from_str(body).unwrap();
    let mut out = Vec::new();
    for m in v["messages"].as_array().unwrap() {
        for b in m["content"].as_array().into_iter().flatten() {
            if b["type"] == "tool_result" {
                let text = match &b["content"] {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                out.push((
                    b["tool_use_id"].as_str().unwrap_or_default().to_owned(),
                    text,
                    b["is_error"].as_bool().unwrap_or(false),
                ));
            }
        }
    }
    out
}
