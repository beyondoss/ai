//! A web page must not be able to drive the loopback daemon (`serve --listen` outside service mode),
//! which authenticates no caller and runs `bash`.
//!
//! Each refusal case sends the request a browser would — a `text/plain` "simple" POST (no CORS
//! preflight), a POST with a foreign `Origin`, a POST whose `Host` is a rebound attacker name, a
//! cross-origin WebSocket upgrade — carrying a `bash` command that touches a marker file, and asserts
//! both the status **and** that the marker was never created: the status alone would not prove the
//! command didn't run. Positive controls show a plain local client (no `Origin`, loopback `Host`, JSON)
//! and a loopback-origin page still get in, and that the Unix socket, which a browser cannot reach and
//! a front door splices onto with the browser's own headers, is deliberately not guarded.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use common::{ChildGuard, ISOLATED_HOME, WS_PATH, spawn_listening};
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

/// `serve --listen 127.0.0.1:0 --listen-uds <sock>`. No model is needed: host `bash` runs without one,
/// so the gateway points at a closed port.
fn daemon(dir: &Path) -> (ChildGuard, u16) {
    let sock = dir.join("agent.sock");
    spawn_listening(
        crate::common::agent_cmd()
            .args([
                "serve",
                "--gateway-url",
                "http://127.0.0.1:9",
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                dir.join("sessions").to_str().unwrap(),
                "--listen-uds",
                sock.to_str().unwrap(),
            ])
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )
}

/// Send `head` (header lines, no terminator) and `body`; return the status and the response body.
async fn exchange<S: AsyncRead + AsyncWrite + Unpin>(
    mut stream: S,
    head: &str,
    body: &str,
) -> (u16, String) {
    let req = format!(
        "{head}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), stream.read_to_end(&mut buf))
        .await
        .expect("daemon answered within 20s")
        .unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {text:?}"));
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_owned())
        .unwrap_or_default();
    (status, body)
}

async fn tcp(port: u16) -> TcpStream {
    TcpStream::connect(("127.0.0.1", port)).await.unwrap()
}

fn touch_cmd(marker: &Path) -> String {
    json!({ "type": "bash", "command": format!("touch {}", marker.display()) }).to_string()
}

/// The audit's repro, split so each rule is shown to hold on its own.
#[tokio::test]
async fn browser_shaped_requests_are_refused_before_anything_runs() {
    let dir = tempfile::tempdir().unwrap();
    let (_child, port) = daemon(dir.path());
    let marker = dir.path().join("pwned");
    let cmd = touch_cmd(&marker);
    let post = format!("POST {WS_PATH} HTTP/1.1");

    let cases = [
        // fetch(url, {method:"POST", body}) from any page: text/plain, no preflight.
        (
            "text/plain simple request",
            format!("{post}\r\nHost: 127.0.0.1:{port}\r\nContent-Type: text/plain"),
            415,
        ),
        // A Blob body with no type sends no Content-Type at all — also a simple request.
        (
            "no Content-Type",
            format!("{post}\r\nHost: 127.0.0.1:{port}"),
            415,
        ),
        (
            "foreign Origin",
            format!(
                "{post}\r\nHost: 127.0.0.1:{port}\r\nOrigin: https://evil.example\r\n\
                 Content-Type: application/json"
            ),
            403,
        ),
        // DNS rebinding: same-origin to the browser, so no Origin to catch — Host gives it away.
        (
            "rebound Host",
            format!(
                "{post}\r\nHost: rebound.evil.example:{port}\r\nContent-Type: application/json"
            ),
            403,
        ),
        // The audit's exact request.
        (
            "audit repro",
            format!(
                "{post}\r\nOrigin: https://evil.example\r\nHost: rebound.evil.example\r\n\
                 Content-Type: text/plain"
            ),
            403,
        ),
    ];
    for (name, head, want) in cases {
        let (status, body) = exchange(tcp(port).await, &head, &cmd).await;
        assert_eq!(status, want, "{name}: {body}");
    }

    // A cross-origin WebSocket upgrade: browsers apply no CORS to it, so the server must refuse.
    let upgrade = |origin: &str| {
        format!(
            "GET {WS_PATH} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nOrigin: {origin}\r\n\
             Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n\
             Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ=="
        )
    };
    let (status, body) = exchange(tcp(port).await, &upgrade("https://evil.example"), "").await;
    assert_eq!(status, 403, "cross-origin upgrade: {body}");

    // Give any (wrongly) started command time to land before asserting it never did.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!marker.exists(), "a refused request ran its command");

    // A page served from loopback is local, not foreign: its upgrade gets through.
    let mut ws = tcp(port).await;
    let req = format!(
        "{}\r\nConnection: Upgrade\r\n\r\n",
        upgrade("http://localhost:3000")
    );
    ws.write_all(req.as_bytes()).await.unwrap();
    let mut buf = [0u8; 12];
    ws.read_exact(&mut buf).await.unwrap();
    assert_eq!(&buf, b"HTTP/1.1 101", "loopback-origin upgrade");
}

/// A local non-browser client — no `Origin`, loopback `Host`, JSON — still runs commands over TCP.
#[tokio::test]
async fn a_local_json_client_still_gets_in() {
    let dir = tempfile::tempdir().unwrap();
    let (_child, port) = daemon(dir.path());
    let marker = dir.path().join("ran");
    let head = format!(
        "POST {WS_PATH} HTTP/1.1\r\nHost: localhost:{port}\r\n\
         Content-Type: application/json; charset=utf-8"
    );
    let (status, body) = exchange(tcp(port).await, &head, &touch_cmd(&marker)).await;
    assert_eq!(status, 200, "{body}");
    assert!(marker.exists(), "the command ran: {body}");
}

/// The Unix socket is not guarded: a browser cannot open one, and a front door that splices
/// authenticated browser traffic onto it forwards the browser's own `Origin` and public `Host`.
#[tokio::test]
async fn the_unix_socket_is_not_browser_guarded() {
    let dir = tempfile::tempdir().unwrap();
    let (_child, _port) = daemon(dir.path());
    let marker = dir.path().join("spliced");
    let head = format!(
        "POST {WS_PATH} HTTP/1.1\r\nHost: svc-8787.edge.example\r\n\
         Origin: https://app.edge.example\r\nContent-Type: application/json"
    );
    let sock = dir.path().join("agent.sock");
    let mut stream = None;
    for _ in 0..500 {
        if let Ok(s) = tokio::net::UnixStream::connect(&sock).await {
            stream = Some(s);
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let stream = stream.expect("unix socket came up");
    let (status, body) = exchange(stream, &head, &touch_cmd(&marker)).await;
    assert_eq!(status, 200, "{body}");
    assert!(marker.exists(), "the command ran: {body}");
}
