//! MCP OAuth e2e: `agent mcp-login`/`mcp-logout` and `tools::mcp::connect_http`'s OAuth-aware path,
//! driven against a real, hand-rolled OAuth-protected MCP server (metadata discovery, dynamic client
//! registration, an authorize endpoint, a token endpoint) — not a mock of the OAuth protocol. A split
//! sibling of `mcp_client.rs` (which covers the non-OAuth MCP client basics), matching this repo's own
//! "split e2e tests by domain into small files" convention.
//!
//! The "browser" step is simulated by a real, unauthenticated GET (via a small hand-rolled HTTP
//! client — no new `reqwest` feature needed just for tests) to the exact URL `mcp-login` prints,
//! following the fixture's real 302 redirect through to the real local callback listener `mcp-login`
//! itself is running — the whole authorization-code + PKCE + dynamic-client-registration dance runs
//! for real, only the human clicking "Allow" is skipped.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

fn write_global_settings(home: &Path, mcp_servers: Value) {
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
struct ParsedRequest {
    method: String,
    path: String,
    query: String,
    headers: std::collections::HashMap<String, String>,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Option<ParsedRequest> {
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

    let mut headers = std::collections::HashMap::new();
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

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == name).then(|| urlencoding_decode(v))
    })
}

fn form_param(body: &[u8], name: &str) -> Option<String> {
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

fn write_response(stream: &mut TcpStream, status: &str, extra_headers: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn write_json(stream: &mut TcpStream, status: &str, value: &Value) {
    let body = serde_json::to_vec(value).unwrap();
    write_response(stream, status, "Content-Type: application/json\r\n", &body);
}

/// A real OAuth-protected MCP server: serves RFC 8414 authorization-server metadata, RFC 7591 dynamic
/// client registration, an authorize endpoint that immediately "approves" (no real login UI — this is
/// a test fixture, not a security boundary) and redirects back with a code, a token endpoint
/// supporting both `authorization_code` and `refresh_token` grants (issuing a new, distinguishable
/// access token each time — proving a refresh is a *real* new token, not the same one replayed), and
/// the actual MCP JSON-RPC endpoint (one `echo` tool), which records every `Authorization` header it
/// receives so a test can assert exactly which token was actually used to call it.
///
/// 404s `/.well-known/oauth-protected-resource*` on purpose (this fixture doesn't implement SEP-985
/// resource-metadata discovery) so `AuthorizationManager::discover_metadata` cleanly falls through to
/// the plain authorization-server metadata discovery this fixture *does* implement — a deliberate,
/// narrower-but-still-real cut, not a bug: the client-side fallback behavior this exercises is exactly
/// as real either way.
///
/// `expires_in_secs` is the lifetime the fixture reports for each token it issues. Note this is
/// almost always irrelevant in practice: `rmcp`'s own `AuthorizationManager::get_access_token`
/// refreshes proactively whenever fewer than 30 seconds remain (`REFRESH_BUFFER_SECS`), so *any*
/// value below that threshold causes an eager refresh on the very next call regardless of how much
/// wall-clock time has actually passed — tests that want to prove a token reaches the server
/// unmodified should pass something comfortably above 30; tests proving refresh happens can pass
/// anything below it and don't need to sleep at all.
fn spawn_oauth_protected_mcp_fixture(expires_in_secs: u64) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let mcp_url = format!("{base}/mcp");
    let seen_auth_headers = Arc::new(Mutex::new(Vec::new()));
    let seen_auth_headers_writer = seen_auth_headers.clone();
    let token_counter = Arc::new(AtomicU32::new(0));
    let issued_tokens = Arc::new(Mutex::new(std::collections::HashSet::new()));
    let issued_tokens_writer = issued_tokens.clone();

    let base_for_thread = base.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let Some(req) = read_request(&mut stream) else {
                continue;
            };

            if req
                .path
                .starts_with("/.well-known/oauth-protected-resource")
            {
                // SEP-985 protected-resource metadata: points at this fixture's own AS so rmcp 3.x
                // discovers the correct issuer (`http://host`) instead of expecting the resource
                // URL (`http://host/mcp`) when probing AS metadata from the MCP path alone.
                write_json(
                    &mut stream,
                    "200 OK",
                    &json!({
                        "resource": format!("{base_for_thread}/mcp"),
                        "authorization_servers": [base_for_thread],
                        "scopes_supported": ["mcp"],
                    }),
                );
                continue;
            }
            if req
                .path
                .starts_with("/.well-known/oauth-authorization-server")
                || req.path.starts_with("/.well-known/openid-configuration")
            {
                write_json(
                    &mut stream,
                    "200 OK",
                    &json!({
                        "issuer": base_for_thread,
                        "authorization_endpoint": format!("{base_for_thread}/authorize"),
                        "token_endpoint": format!("{base_for_thread}/token"),
                        "registration_endpoint": format!("{base_for_thread}/register"),
                        "scopes_supported": ["mcp"],
                        "response_types_supported": ["code"],
                        "code_challenge_methods_supported": ["S256"],
                    }),
                );
                continue;
            }
            if req.path == "/register" && req.method == "POST" {
                let request: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                let redirect_uris = request
                    .get("redirect_uris")
                    .cloned()
                    .unwrap_or_else(|| json!([]));
                write_json(
                    &mut stream,
                    "201 Created",
                    &json!({
                        "client_id": "test-dynamically-registered-client",
                        "redirect_uris": redirect_uris,
                    }),
                );
                continue;
            }
            if req.path == "/authorize" && req.method == "GET" {
                let redirect_uri = query_param(&req.query, "redirect_uri").unwrap_or_default();
                let state = query_param(&req.query, "state").unwrap_or_default();
                let sep = if redirect_uri.contains('?') { "&" } else { "?" };
                let location =
                    format!("{redirect_uri}{sep}code=test-authorization-code&state={state}");
                write_response(
                    &mut stream,
                    "302 Found",
                    &format!("Location: {location}\r\n"),
                    b"",
                );
                continue;
            }
            if req.path == "/token" && req.method == "POST" {
                let grant_type = form_param(&req.body, "grant_type").unwrap_or_default();
                if grant_type != "authorization_code" && grant_type != "refresh_token" {
                    write_json(
                        &mut stream,
                        "400 Bad Request",
                        &json!({ "error": "unsupported_grant_type" }),
                    );
                    continue;
                }
                let n = token_counter.fetch_add(1, Ordering::SeqCst);
                let access_token = format!("access-token-{n}");
                issued_tokens_writer
                    .lock()
                    .unwrap()
                    .insert(access_token.clone());
                write_json(
                    &mut stream,
                    "200 OK",
                    &json!({
                        "access_token": access_token,
                        "token_type": "Bearer",
                        "expires_in": expires_in_secs,
                        "refresh_token": "refresh-token-fixed",
                        "scope": "mcp",
                    }),
                );
                continue;
            }
            if req.path == "/mcp" && req.method == "POST" {
                // A *real* protected resource: reject anything without a currently-valid bearer
                // token, matching what an actual OAuth-gated MCP server does — without this, the
                // fixture would only ever *advertise* OAuth support without ever enforcing it, and an
                // unauthenticated connect would silently succeed instead of proving the auth path is
                // real.
                let authorized = req
                    .headers
                    .get("authorization")
                    .and_then(|h| h.strip_prefix("Bearer "))
                    .is_some_and(|token| issued_tokens_writer.lock().unwrap().contains(token));
                if !authorized {
                    write_response(
                        &mut stream,
                        "401 Unauthorized",
                        &format!("WWW-Authenticate: Bearer resource=\"{base_for_thread}/mcp\"\r\n"),
                        b"",
                    );
                    continue;
                }
                if let Some(auth) = req.headers.get("authorization") {
                    seen_auth_headers_writer.lock().unwrap().push(auth.clone());
                }
                let request: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
                let is_notification = request.get("id").is_none();
                if is_notification {
                    write_response(&mut stream, "202 Accepted", "", b"");
                    continue;
                }
                let id = request.get("id").cloned().unwrap_or(Value::Null);
                let method = request.get("method").and_then(Value::as_str).unwrap_or("");
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
                    &mut stream,
                    "200 OK",
                    &json!({ "jsonrpc": "2.0", "id": id, "result": result }),
                );
                continue;
            }
            write_response(&mut stream, "404 Not Found", "", b"");
        }
    });
    (mcp_url, seen_auth_headers)
}

/// A tiny hand-rolled HTTP GET client that follows redirects — simulates "the user's browser visits
/// the authorization URL and is redirected through to the local callback" without pulling in
/// `reqwest`'s `blocking` feature (not otherwise needed anywhere in this crate) just for a test.
fn get_following_redirects(url: &str, max_redirects: u8) -> u16 {
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
/// `mcp-login` prints via `eprintln!("  {auth_url}\n")`), returning it. Panics (test-only code) if the
/// stream closes first — a clear, immediate test failure rather than a hang.
fn read_printed_url(stderr: &mut BufReader<std::process::ChildStderr>) -> String {
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

#[test]
fn mcp_login_completes_the_real_oauth_flow_and_the_token_authenticates_a_real_tool_call() {
    let home = tempfile::tempdir().unwrap();
    // A long expiry: this test asserts the *exact* token `mcp-login` obtained reaches the server, so
    // it must not be eagerly refreshed away before the follow-up `run` gets to use it.
    let (url, seen_auth_headers) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);

    // The "user visits the browser" step: a real GET, following the fixture's real 302 redirect
    // through to `mcp-login`'s own real local callback listener.
    let status = get_following_redirects(&auth_url, 3);
    assert_eq!(
        status, 200,
        "the callback must be delivered and answered 200"
    );

    let output = login.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "mcp-login failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("logged in: protected"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );

    // Now prove the persisted credential is actually usable: a real `run` invocation, real tool call,
    // through the real bearer token `mcp-login` just established.
    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__protected__echo",
        &json!({ "text": "oauth-authenticated-marker" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let cwd = tempfile::tempdir().unwrap();
    let run_output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
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
            "6",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    assert!(
        run_output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("oauth-authenticated-marker"),
        "the oauth-authenticated tool call must actually go through: {}",
        bodies[1]
    );
    let seen = seen_auth_headers.lock().unwrap();
    assert!(
        seen.iter().any(|h| h == "Bearer access-token-0"),
        "the mcp server must have received the exact bearer token mcp-login obtained: {seen:?}"
    );
}

#[test]
fn mcp_login_an_unknown_server_name_fails_clearly() {
    let home = tempfile::tempdir().unwrap();
    write_global_settings(home.path(), json!([]));
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "does-not-exist"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no MCP server named `does-not-exist`"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mcp_login_a_stdio_server_is_rejected_with_a_clear_error() {
    let home = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([{ "name": "local", "transport": "stdio", "command": "true", "args": [], "env": {} }]),
    );
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "local"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("stdio transport"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mcp_logout_removes_a_stored_credential_and_is_idempotent() {
    let home = tempfile::tempdir().unwrap();
    let (url, _seen) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    get_following_redirects(&auth_url, 3);
    let output = login.wait_with_output().unwrap();
    assert!(output.status.success());

    let logout = |home: &Path| -> String {
        let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
            .args(["mcp-logout", "protected"])
            .env("HOME", home)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    assert!(logout(home.path()).contains("logged out: protected"));
    // Idempotent: logging out again reports "not logged in", not an error.
    assert!(logout(home.path()).contains("not logged in: protected"));
}

#[test]
fn mcp_connect_to_an_oauth_protected_server_without_ever_logging_in_fails_naming_mcp_login() {
    let home = tempfile::tempdir().unwrap();
    let (url, _seen) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );
    let cwd = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);
    let output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .args([
            "run",
            "just say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    // Fail-soft still applies: the *run* succeeds even though this one MCP server never connected.
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mcp-login protected"),
        "an unauthenticated connect attempt to an oauth-protected server must hint at `agent \
         mcp-login`, not just fail silently: {stderr}"
    );
}

#[test]
fn mcp_get_access_token_transparently_refreshes_an_expired_token_on_the_next_connect() {
    let home = tempfile::tempdir().unwrap();
    // A short (1s) expiry: `rmcp`'s own `AuthorizationManager::get_access_token` refreshes proactively
    // whenever fewer than 30 seconds remain (`REFRESH_BUFFER_SECS`), so any token issued with a
    // sub-30s lifetime is *always* treated as due for refresh on the very next call — no need to
    // actually sleep past the expiry for this to trigger.
    let (url, seen_auth_headers) = spawn_oauth_protected_mcp_fixture(1);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    get_following_redirects(&auth_url, 3);
    assert!(login.wait_with_output().unwrap().status.success());
    assert!(
        seen_auth_headers.lock().unwrap().is_empty(),
        "sanity: mcp-login itself never calls tools/call, so no MCP request (and thus no \
         Authorization header) should have reached the server yet"
    );

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__protected__echo",
        &json!({ "text": "post-refresh-marker" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let cwd = tempfile::tempdir().unwrap();
    let run_output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .args([
            "run",
            "call echo",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "6",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    assert!(
        run_output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    assert!(bodies.lock().unwrap()[1].contains("post-refresh-marker"));

    let seen = seen_auth_headers.lock().unwrap();
    assert!(
        seen.iter().any(|h| h == "Bearer access-token-1"),
        "a genuinely refreshed (not replayed) token must reach the server: {seen:?}"
    );
}

#[test]
fn mcp_login_established_credential_is_honored_by_serve_too_not_just_run() {
    // `tools::mcp::connect_http`'s OAuth-aware path is the same code regardless of which of the two
    // call sites (`run_task`, `ServeConfig::mcp_tools`) invoked `connect_all` — this proves a login
    // established once via `agent mcp-login` (an interactive, CLI-only, `serve`-external command —
    // there's no RPC-triggerable equivalent from *inside* a live `serve` session, and a session already
    // running won't pick up a login established after its own startup either, since MCP servers connect
    // once and aren't reconnected mid-session) is actually honored on `serve`'s *next* startup, not
    // just `run`'s.
    let home = tempfile::tempdir().unwrap();
    let (url, seen_auth_headers) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    get_following_redirects(&auth_url, 3);
    assert!(login.wait_with_output().unwrap().status.success());

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__protected__echo",
        &json!({ "text": "serve-oauth-marker" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let mut child = serve_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"), &base, &session_file)
        .env("HOME", home.path())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "call the protected echo tool" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let response = frames
        .iter()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .unwrap();
    assert_eq!(
        response["success"], true,
        "the prompt turn must succeed: {frames:?}"
    );
    assert!(bodies.lock().unwrap()[1].contains("serve-oauth-marker"));
    assert!(
        seen_auth_headers
            .lock()
            .unwrap()
            .iter()
            .any(|h| h == "Bearer access-token-0"),
        "the mcp-login-established token must authenticate the call made through serve"
    );
}
