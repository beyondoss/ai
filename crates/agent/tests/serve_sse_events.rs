//! The session as a Server-Sent Events stream (`GET /_beyond/agent/events`, `serve --listen`).
//!
//! The same frames the WebSocket carries, for a client that is not a WebSocket (`curl -N`,
//! `EventSource`). `id:` is the tree id of the newest committed message, on the `catchup` event and
//! on every `committed` event; resuming with `Last-Event-ID` (or `?since=`) catches up only what was
//! committed after it, and an id the session does not have is a `409`, not a silent full or empty
//! catch-up.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;
use std::time::Duration;

use common::{
    ISOLATED_HOME, spawn_listening, spawn_model_server, turn_text, ws_connect,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// One parsed event: `(event, id, data)`.
type Event = (Option<String>, Option<String>, String);

struct Sse {
    stream: TcpStream,
    buf: Vec<u8>,
    status: u16,
}

impl Sse {
    async fn open(port: u16, sid: &str, query: &str, last_event_id: Option<&str>) -> Sse {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let mut req = format!(
            "GET /_beyond/agent/events?session_id={sid}{query} HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: text/event-stream\r\n"
        );
        if let Some(id) = last_event_id {
            req.push_str(&format!("Last-Event-ID: {id}\r\n"));
        }
        req.push_str("\r\n");
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        let head_end = loop {
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break i + 4;
            }
            let mut chunk = [0u8; 1024];
            let n = tokio::time::timeout(Duration::from_secs(20), stream.read(&mut chunk))
                .await
                .expect("the head")
                .unwrap();
            assert!(n > 0, "closed before a head");
            buf.extend_from_slice(&chunk[..n]);
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        buf.drain(..head_end);
        Sse {
            stream,
            buf,
            status,
        }
    }

    /// The next event, skipping comments and the `retry:` line; `None` when the stream closes.
    async fn next(&mut self) -> Option<Event> {
        loop {
            while let Some(i) = self.buf.windows(2).position(|w| w == b"\n\n") {
                let block = String::from_utf8_lossy(&self.buf[..i]).into_owned();
                self.buf.drain(..i + 2);
                let (mut event, mut id, mut data) = (None, None, Vec::new());
                for line in block.lines() {
                    if let Some(v) = line.strip_prefix("event: ") {
                        event = Some(v.to_owned());
                    } else if let Some(v) = line.strip_prefix("id: ") {
                        id = Some(v.to_owned());
                    } else if let Some(v) = line.strip_prefix("data: ") {
                        data.push(v.to_owned());
                    }
                }
                if !data.is_empty() {
                    return Some((event, id, data.join("\n")));
                }
            }
            let mut chunk = [0u8; 8192];
            let n = tokio::time::timeout(Duration::from_secs(30), self.stream.read(&mut chunk))
                .await
                .expect("an event")
                .unwrap_or(0);
            if n == 0 {
                return None;
            }
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }

    /// Events up to and including the first `committed`.
    async fn until_committed(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        while let Some(e) = self.next().await {
            let done = e.0.as_deref() == Some("committed");
            out.push(e);
            if done {
                return out;
            }
        }
        panic!("the stream closed before a commit: {out:?}");
    }
}

fn texts(messages: &Value) -> Vec<String> {
    messages
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m.to_string())
        .collect()
}

#[tokio::test]
async fn the_event_stream_commits_with_ids_and_resumes_after_one() {
    const SID: &str = "ssestream1";
    let dir = tempfile::tempdir().unwrap();
    let (base, _requests) = spawn_model_server(vec![
        turn_text("ANSWER_ONE"),
        turn_text("ANSWER_TWO"),
        turn_text("ANSWER_THREE"),
    ]);
    let (_child, port) = spawn_listening(
        crate::common::agent_cmd()
            .args([
                "serve",
                "--gateway-url",
                &base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                dir.path().to_str().unwrap(),
            ])
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    );

    // Attached before anything is said: a fresh session has no history, so no catch-up.
    let mut sse = Sse::open(port, SID, "", None).await;
    assert_eq!(sse.status, 200);

    let mut ws = ws_connect(port, Some(SID)).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "first" }),
    )
    .await;
    let first = sse.until_committed().await;
    let committed = first.last().unwrap();
    let id1 = committed
        .1
        .clone()
        .expect("a committed event carries the id");
    assert_eq!(
        serde_json::from_str::<Value>(&committed.2).unwrap()["leaf_id"],
        json!(id1)
    );
    assert!(
        first.iter().any(|e| e.2.contains("ANSWER_ONE")),
        "the live frames carry the answer: {first:?}"
    );
    ws_read_until_response(&mut ws, "prompt").await;

    // A second turn, read from the same stream, ends in a later commit.
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p2", "message": "second" }),
    )
    .await;
    let second = sse.until_committed().await;
    let id2 = second.last().unwrap().1.clone().unwrap();
    assert_ne!(id1, id2, "a later commit has a later id");
    ws_read_until_response(&mut ws, "prompt").await;

    // Resume from the first commit: the catch-up is only what came after it, and ends at the second.
    let mut resumed = Sse::open(port, SID, "", Some(&id1)).await;
    assert_eq!(resumed.status, 200);
    let (event, id, data) = resumed.next().await.unwrap();
    assert_eq!(event.as_deref(), Some("catchup"));
    assert_eq!(id.as_deref(), Some(id2.as_str()));
    let frame: Value = serde_json::from_str(&data).unwrap();
    let after = texts(&frame["data"]["messages"]).join("");
    assert!(
        after.contains("second") && after.contains("ANSWER_TWO"),
        "{after}"
    );
    assert!(
        !after.contains("ANSWER_ONE") && !after.contains("first"),
        "nothing already held is sent again: {after}"
    );

    // `?since=` is the same resume.
    let mut by_query = Sse::open(port, SID, &format!("&since={id1}"), None).await;
    let (event, id, _) = by_query.next().await.unwrap();
    assert_eq!(
        (event.as_deref(), id.as_deref()),
        (Some("catchup"), Some(id2.as_str()))
    );

    // A full catch-up (no id) holds everything.
    let mut full = Sse::open(port, SID, "", None).await;
    let (_, _, data) = full.next().await.unwrap();
    let all = texts(&serde_json::from_str::<Value>(&data).unwrap()["data"]["messages"]).join("");
    assert!(
        all.contains("ANSWER_ONE") && all.contains("ANSWER_TWO"),
        "{all}"
    );

    // An id the session does not have is refused, not guessed at.
    let stale = Sse::open(port, SID, "", Some("no-such-message")).await;
    assert_eq!(stale.status, 409);

    // A resumed stream carries on live: a third turn arrives on it, ending in a commit.
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p3", "message": "third" }),
    )
    .await;
    let third = resumed.until_committed().await;
    assert!(
        third.iter().any(|e| e.2.contains("ANSWER_THREE")),
        "{third:?}"
    );
    ws_read_until_response(&mut ws, "prompt").await;
}

#[tokio::test]
async fn the_event_stream_is_get_only() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _requests) = spawn_model_server(vec![turn_text("x")]);
    let (_child, port) = spawn_listening(
        crate::common::agent_cmd()
            .args([
                "serve",
                "--gateway-url",
                &base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                dir.path().to_str().unwrap(),
            ])
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    );
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    stream
        .write_all(b"POST /_beyond/agent/events?session_id=ssepost1 HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 0\r\n\r\n")
        .await
        .unwrap();
    let mut out = String::new();
    let _ = tokio::time::timeout(Duration::from_secs(10), stream.read_to_string(&mut out)).await;
    assert!(out.starts_with("HTTP/1.1 405"), "{out}");
}

/// What the WebSocket checks at the door, the event stream checks too: the same grant, the same
/// statuses, before any stream starts.
#[tokio::test]
async fn the_event_stream_wants_the_same_grant_as_the_websocket() {
    use common::service::Service;
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1", "s2"]).await;
    let status = |grant: Option<String>, sid: &'static str| {
        let port = svc.port;
        async move {
            let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            let auth = grant
                .map(|g| format!("x-beyond-grant: {g}\r\n"))
                .unwrap_or_default();
            stream
                .write_all(
                    format!("GET /_beyond/agent/events?session_id={sid} HTTP/1.1\r\nHost: x\r\n{auth}\r\n")
                        .as_bytes(),
                )
                .await
                .unwrap();
            let mut chunk = [0u8; 512];
            let n = tokio::time::timeout(Duration::from_secs(10), stream.read(&mut chunk))
                .await
                .unwrap()
                .unwrap();
            String::from_utf8_lossy(&chunk[..n])
                .split_whitespace()
                .nth(1)
                .unwrap()
                .to_owned()
        }
    };
    assert_eq!(status(None, "s1.alpha").await, "401", "no grant");
    assert_eq!(
        status(Some(svc.token("t1", "s1.alpha")), "s1.beta").await,
        "403",
        "a grant for another session"
    );
    assert_eq!(
        status(Some(svc.token("t1", "s1.alpha")), "s1.alpha").await,
        "200",
        "the session's own grant"
    );
}
