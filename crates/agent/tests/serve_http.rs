//! Integration tests for the HTTP POST transport (`serve --listen`).
//!
//! The JSON control protocol is byte-identical to WebSocket/stdio; these assert the *transport*:
//! POST a command without holding a socket, learn the session id from `X-Session-Id`, start a run
//! that outlives the request (202 on `prompt` ack), share that session with a WebSocket attach, and
//! map HTTP-level errors (404/405/400) without breaking the existing upgrade path.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, WS_PATH, free_port, spawn_model_server, turn_text,
    wait_for_port, ws_connect, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

fn serve_ws_child(base: &str, session_dir: &str, port: u16) -> ChildGuard {
    Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir,
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

struct HttpReply {
    status: u16,
    session_id: Option<String>,
    body: Value,
}

/// Raw HTTP/1.1 — the same bytes a `curl` would send. Avoids constructing a `reqwest::Client` in
/// this test crate (that panics unless a rustls crypto provider is installed; production code
/// calls `agent_core::ensure_provider` at every client-construction site, and these tests are not
/// one of those).
async fn http_exchange(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect to serve");
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n");
    if let Some(body) = body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", body.len()));
        req.push_str("\r\n");
        stream
            .write_all(req.as_bytes())
            .await
            .expect("write headers");
        stream.write_all(body).await.expect("write body");
    } else {
        req.push_str("\r\n");
        stream
            .write_all(req.as_bytes())
            .await
            .expect("write headers");
    }
    stream.flush().await.expect("flush");

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.expect("read response");
    let header_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("HTTP header terminator");
    let head = String::from_utf8_lossy(&buf[..header_end]);
    let mut lines = head.split("\r\n");
    let status_line = lines.next().expect("status line");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("status u16");
    let headers = lines
        .filter_map(|line| {
            let (k, v) = line.split_once(':')?;
            Some((k.trim().to_ascii_lowercase(), v.trim().to_string()))
        })
        .collect();
    let body = buf[header_end + 4..].to_vec();
    (status, headers, body)
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

async fn post_cmd(port: u16, session_id: Option<&str>, cmd: Value) -> HttpReply {
    let path = match session_id {
        Some(id) => format!("{WS_PATH}?session_id={id}"),
        None => WS_PATH.to_string(),
    };
    let payload = cmd.to_string();
    let (status, headers, body) =
        http_exchange(port, "POST", &path, Some(payload.as_bytes())).await;
    let session_id = header(&headers, "x-session-id").map(str::to_owned);
    let body: Value = serde_json::from_slice(&body)
        .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&body) }));
    HttpReply {
        status,
        session_id,
        body,
    }
}

async fn wait_for_assistant(port: u16, session_id: &str) -> Value {
    let start = Instant::now();
    loop {
        let reply = post_cmd(
            port,
            Some(session_id),
            json!({ "type": "get_messages", "id": "gm" }),
        )
        .await;
        assert_eq!(reply.status, 200, "get_messages: {}", reply.body);
        if reply.body["success"] == true
            && reply.body["data"]["messages"]
                .as_array()
                .is_some_and(|m| m.iter().any(|msg| msg["role"] == "assistant"))
        {
            return reply.body;
        }
        if start.elapsed() > Duration::from_secs(5) {
            panic!("timed out waiting for assistant message: {:#?}", reply.body);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// POST `get_state` mints a session, returns 200 with the protocol `response`, and names it in
/// `X-Session-Id` so a client that omitted `?session_id=` can address the next request.
#[tokio::test]
async fn http_post_get_state_mints_a_session_and_returns_the_protocol_frame() {
    let (base, _requests) = spawn_model_server(vec![turn_text("unused")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let reply = post_cmd(port, None, json!({ "type": "get_state", "id": "s1" })).await;
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(reply.body["type"], "response");
    assert_eq!(reply.body["command"], "get_state");
    assert_eq!(reply.body["id"], "s1");
    assert_eq!(reply.body["success"], true, "{}", reply.body);
    let sid = reply.session_id.expect("X-Session-Id");
    assert_eq!(
        reply.body["data"]["session_id"].as_str(),
        Some(sid.as_str()),
        "header and get_state payload must name the same session"
    );

    let again = post_cmd(port, Some(&sid), json!({ "type": "get_state", "id": "s2" })).await;
    assert_eq!(again.status, 200, "{}", again.body);
    assert_eq!(again.session_id.as_deref(), Some(sid.as_str()));
    assert_eq!(
        again.body["data"]["session_id"].as_str(),
        Some(sid.as_str())
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// POST `prompt` returns 202 + the ack the moment the turn is queued. The run outlives the request:
/// a later `get_messages` on the same id sees the assistant reply, and a WebSocket attach to that
/// id is the same session.
#[tokio::test]
async fn http_post_prompt_is_accepted_and_the_run_outlives_the_request() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hello from post")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let reply = post_cmd(
        port,
        Some("httprun1"),
        json!({ "type": "prompt", "id": "p1", "message": "hi" }),
    )
    .await;
    assert_eq!(reply.status, 202, "prompt ack is 202: {}", reply.body);
    assert_eq!(reply.session_id.as_deref(), Some("httprun1"));
    assert_eq!(reply.body["type"], "ack");
    assert_eq!(reply.body["command"], "prompt");
    assert_eq!(reply.body["id"], "p1");

    let messages = wait_for_assistant(port, "httprun1").await;
    let dump = messages.to_string();
    assert!(
        dump.contains("hello from post"),
        "POST get_messages must see the run that outlived the prompt request: {messages:#?}"
    );

    let mut ws = ws_connect(port, Some("httprun1")).await;
    ws_send(&mut ws, json!({ "type": "get_state", "id": "w1" })).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    let state = frames.last().expect("get_state response");
    assert_eq!(state["data"]["session_id"], "httprun1");
    assert!(
        state["data"]["message_count"].as_u64().unwrap_or(0) >= 2,
        "websocket attach is the same session the POST ran: {state}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// A `prompt` rejected before the ack (no `message`) is a protocol `response`, so HTTP 200 — not 202.
#[tokio::test]
async fn http_post_prompt_without_message_is_a_protocol_error_not_accepted() {
    let (base, _requests) = spawn_model_server(vec![turn_text("unused")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let reply = post_cmd(
        port,
        Some("badprompt"),
        json!({ "type": "prompt", "id": "p0" }),
    )
    .await;
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(reply.body["type"], "response");
    assert_eq!(reply.body["success"], false);
    assert!(
        reply.body["error"]
            .as_str()
            .unwrap_or("")
            .contains("message"),
        "{}",
        reply.body
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// Supervisor-level `list_daemon_sessions` is answered over POST without spawning a session.
#[tokio::test]
async fn http_post_list_daemon_sessions_does_not_require_a_session() {
    let (base, _requests) = spawn_model_server(vec![turn_text("one")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let created = post_cmd(
        port,
        Some("listed1"),
        json!({ "type": "get_state", "id": "g" }),
    )
    .await;
    assert_eq!(created.status, 200, "{}", created.body);

    let listed = post_cmd(
        port,
        None,
        json!({ "type": "list_daemon_sessions", "id": "L1" }),
    )
    .await;
    assert_eq!(listed.status, 200, "{}", listed.body);
    assert!(
        listed.session_id.is_none(),
        "list_daemon_sessions must not mint a session: {:?}",
        listed.session_id
    );
    assert_eq!(listed.body["command"], "list_daemon_sessions");
    assert_eq!(listed.body["id"], "L1");
    let sessions = listed.body["data"]["sessions"]
        .as_array()
        .expect("data.sessions");
    assert!(
        sessions.iter().any(|s| s["id"] == "listed1"),
        "live session must appear: {sessions:#?}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

#[tokio::test]
async fn http_wrong_path_is_404_and_wrong_method_is_405() {
    let (base, _requests) = spawn_model_server(vec![turn_text("unused")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let (status, _headers, _body) =
        http_exchange(port, "POST", "/nope", Some(b"{\"type\":\"get_state\"}")).await;
    assert_eq!(status, 404);

    let (status, headers, _body) =
        http_exchange(port, "PUT", WS_PATH, Some(b"{\"type\":\"get_state\"}")).await;
    assert_eq!(status, 405);
    let allow = header(&headers, "allow").unwrap_or("");
    assert!(allow.contains("POST"), "Allow: {allow}");
    assert!(allow.contains("GET"), "Allow: {allow}");

    let bad_id = post_cmd(port, Some("../escape"), json!({ "type": "get_state" })).await;
    assert_eq!(bad_id.status, 400, "{}", bad_id.body);

    // The upgrade path still works after the HTTP error mapping — a GET with a WebSocket handshake
    // is not 405.
    let mut ws = ws_connect(port, Some("stillworks")).await;
    ws_send(&mut ws, json!({ "type": "get_state", "id": "ok" })).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true);

    let _ = child.kill();
    let _ = child.wait();
}

#[tokio::test]
async fn http_get_without_upgrade_is_426() {
    let (base, _requests) = spawn_model_server(vec![turn_text("unused")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let (status, headers, _body) = http_exchange(port, "GET", WS_PATH, None).await;
    assert_eq!(status, 426);
    assert!(
        header(&headers, "upgrade")
            .unwrap_or("")
            .eq_ignore_ascii_case("websocket")
    );

    let _ = child.kill();
    let _ = child.wait();
}
