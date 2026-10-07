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
    issued: Arc<Mutex<HashSet<String>>>,
}

impl OAuthFixture {
    /// Revoke every token issued so far: the next request with one is answered 401.
    pub fn revoke_all(&self) {
        self.issued.lock().unwrap().clear();
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
            issued: Arc::default(),
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
        write_response(&mut stream, "404 Not Found", "", b"");
    }

    /// The protected resource: anything without a currently-valid bearer token is answered 401 with
    /// a `WWW-Authenticate` challenge, as a real OAuth-gated MCP server does.
    fn mcp(&self, stream: &mut TcpStream, req: &ParsedRequest) {
        let f = &self.fixture;
        let request: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
        let authorized = req
            .headers
            .get("authorization")
            .and_then(|h| h.strip_prefix("Bearer "))
            .is_some_and(|token| f.issued.lock().unwrap().contains(token));
        if !authorized {
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
            return write_response(
                stream,
                "401 Unauthorized",
                &format!(
                    "WWW-Authenticate: Bearer resource=\"{}/mcp\"\r\n",
                    self.base
                ),
                b"",
            );
        }
        if let Some(auth) = req.headers.get("authorization") {
            f.seen_auth_headers.lock().unwrap().push(auth.clone());
        }
        if request.get("id").is_none() {
            return write_response(stream, "202 Accepted", "", b"");
        }
        let id = request.get("id").cloned().unwrap_or(Value::Null);
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
        let result = match method {
            "initialize" => json!({
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "mcp-fixture-oauth-server", "version": "0.0.0" },
            }),
            "tools/list" => json!({ "tools": [{
                "name": "echo",
                "description": "Echoes back its `text` argument.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"],
                },
            }] }),
            "tools/call" => {
                let text = request
                    .pointer("/params/arguments/text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                json!({ "content": [{ "type": "text", "text": text }], "isError": false })
            }
            other => json!({
                "content": [{ "type": "text", "text": format!("unhandled method {other}") }],
                "isError": true,
            }),
        };
        write_json(
            stream,
            "200 OK",
            &json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        );
        if method == "tools/call" {
            let calls = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if calls == f.revoke_after_calls.load(Ordering::SeqCst) {
                f.revoke_all();
            }
        }
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
