//! A one-shot answer to a command that runs for a while starts early: the headers go out once the
//! command has been silent for a heartbeat, and the body is the same one JSON document with
//! whitespace in front of it, so a proxy with a response-header timeout (instd's gateway: 295 s)
//! never cuts a long command off. A command that answers before then gets its answer exactly as
//! ever, status included.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::{Duration, Instant};

use base64::Engine as _;
use beyond_test_support::ports;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const BEAT: Duration = Duration::from_millis(200);

async fn start() -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    o.heartbeat = BEAT;
    (execd::serve(ports::tokio_listener().await, o).unwrap(), dir)
}

/// POST `body`; return the head, when it arrived, and the rest of the response read to the end.
async fn exchange(server: &execd::Handle, body: Value) -> (String, Duration, Vec<u8>) {
    let body = body.to_string();
    let mut sock = tokio::net::TcpStream::connect(server.local_addr())
        .await
        .unwrap();
    let req = format!(
        "POST /exec HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let start = Instant::now();
    sock.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    let (mut head_at, mut split) = (None, None);
    loop {
        let n = sock.read(&mut buf).await.unwrap();
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        if split.is_none() {
            split = raw.windows(4).position(|w| w == b"\r\n\r\n");
            if split.is_some() {
                head_at = Some(start.elapsed());
            }
        }
    }
    let split = split.expect("a response head");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    (head, head_at.unwrap(), raw[split + 4..].to_vec())
}

/// The JSON document in a body that may be chunked.
fn document(head: &str, rest: &[u8]) -> String {
    let text = if head.to_ascii_lowercase().contains("chunked") {
        let mut out = Vec::new();
        let mut at = 0;
        loop {
            let eol = rest[at..].windows(2).position(|w| w == b"\r\n").unwrap() + at;
            let size =
                usize::from_str_radix(std::str::from_utf8(&rest[at..eol]).unwrap().trim(), 16)
                    .unwrap();
            if size == 0 {
                break;
            }
            out.extend_from_slice(&rest[eol + 2..eol + 2 + size]);
            at = eol + 2 + size + 2;
        }
        out
    } else {
        rest.to_vec()
    };
    String::from_utf8(text).unwrap()
}

#[tokio::test]
async fn a_long_command_gets_its_headers_before_it_ends() {
    let (server, _dir) = start().await;
    let (head, head_at, rest) = exchange(
        &server,
        json!({ "command": "sh", "args": ["-c", "sleep 1.2; echo done; exit 3"] }),
    )
    .await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(
        head_at < Duration::from_millis(900),
        "the headers waited for the command: {head_at:?}"
    );
    let doc = document(&head, &rest);
    assert!(
        doc.starts_with(char::is_whitespace),
        "the body keeps the connection alive with whitespace: {doc:?}"
    );
    let v: Value = serde_json::from_str(&doc).expect("one JSON document");
    assert_eq!(v["exit_code"], 3);
    let out = base64::engine::general_purpose::STANDARD
        .decode(v["stdout"].as_str().unwrap())
        .unwrap();
    assert_eq!(out, b"done\n");
    server.shutdown().await;
}

#[tokio::test]
async fn a_short_command_is_answered_exactly_as_before() {
    let (server, _dir) = start().await;
    let (head, _, rest) = exchange(
        &server,
        json!({ "command": "sh", "args": ["-c", "echo hi"] }),
    )
    .await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let doc = document(&head, &rest);
    assert!(
        doc.starts_with('{'),
        "no padding when nothing waited: {doc:?}"
    );
    server.shutdown().await;
}

#[tokio::test]
async fn an_error_before_the_command_starts_keeps_its_status() {
    let (server, _dir) = start().await;
    let (head, _, _) = exchange(
        &server,
        json!({ "command": "true", "stdin": "not base64!" }),
    )
    .await;
    assert!(head.starts_with("HTTP/1.1 400"), "{head}");
    server.shutdown().await;
}

#[tokio::test]
async fn a_client_that_leaves_stops_the_command() {
    let (server, dir) = start().await;
    let marker = dir.path().join("ran-to-the-end");
    let script = format!("sleep 1.5; touch {}", marker.display());
    let body = json!({ "command": "sh", "args": ["-c", script] }).to_string();
    let mut sock = tokio::net::TcpStream::connect(server.local_addr())
        .await
        .unwrap();
    let req = format!(
        "POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(req.as_bytes()).await.unwrap();
    // Past the first heartbeat, so the answer has started; then leave.
    tokio::time::sleep(Duration::from_millis(600)).await;
    drop(sock);
    tokio::time::sleep(Duration::from_millis(1800)).await;
    assert!(!marker.exists(), "the command outlived its client");
    server.shutdown().await;
}
