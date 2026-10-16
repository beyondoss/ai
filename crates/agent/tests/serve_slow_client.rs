//! A connection that cannot keep up with its session's output (`serve --listen`).
//!
//! Each attached connection has its own outbound buffer between the session's broadcast and the
//! socket. Two things must hold when a burst outruns it:
//!
//! - **A client that is reading gets every frame.** A model hands the agent many deltas per network
//!   read, so a single burst can put thousands of frames into the buffer in microseconds — far faster
//!   than any socket drains, however attentive the client. The buffer used to be 1024 frames, and a
//!   3000-delta burst dropped a client that never stopped reading.
//! - **A client that is dropped learns it.** A dropped connection used to stay open: pings kept
//!   arriving and its commands still ran, but it never received another frame, so it never knew to
//!   reconnect, and its session counted as attached forever. Now the socket closes, and the client
//!   re-attaches and is caught up like any reconnect.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::Duration;

use common::{
    ChildGuard, ISOLATED_HOME, TestWs, WS_PATH, spawn_listening, spawn_model_server, sse, ws_send,
};
use futures::StreamExt as _;
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;

fn serve_ws_child(base: &str, session_dir: &str) -> (ChildGuard, u16) {
    spawn_listening(
        Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
            .args([
                "serve",
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
            .stdout(Stdio::null()),
    )
}

/// One model turn of `deltas` text deltas, all in a single response body — the whole burst reaches
/// the agent at once, which is the worst case for the connection buffer.
fn burst_turn(deltas: impl Iterator<Item = String>) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
    ];
    events.extend(deltas.map(|text| {
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } })
    }));
    events.push(json!({ "type": "content_block_stop", "index": 0 }));
    events.push(json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 6 } }));
    events.push(json!({ "type": "message_stop" }));
    sse(&events)
}

/// A client without tungstenite's default 16 MiB frame cap: the catch-up a re-attaching client gets
/// carries the committed turn, which here is larger than that.
async fn connect(port: u16, sid: &str) -> TestWs {
    let mut cfg = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default();
    cfg.max_message_size = None;
    cfg.max_frame_size = None;
    let url = format!("ws://127.0.0.1:{port}{WS_PATH}?session_id={sid}");
    tokio_tungstenite::connect_async_with_config(url, Some(cfg), false)
        .await
        .expect("websocket connect")
        .0
}

enum Next {
    Frame(Value),
    Closed,
    Silent,
}

/// The next JSON frame, or how the socket ended: a close, a reset and EOF all count as `Closed`.
async fn next(ws: &mut TestWs, within: Duration) -> Next {
    loop {
        match tokio::time::timeout(within, ws.next()).await {
            Err(_) => return Next::Silent,
            Ok(None | Some(Err(_)) | Some(Ok(Message::Close(_)))) => return Next::Closed,
            Ok(Some(Ok(Message::Text(t)))) => {
                if let Ok(v) = serde_json::from_str(t.as_str()) {
                    return Next::Frame(v);
                }
            }
            Ok(Some(Ok(_))) => {}
        }
    }
}

fn is_response(f: &Value, command: &str) -> bool {
    f["type"] == "response" && f["command"] == command
}

/// A client that reads continuously receives the whole of a burst far larger than any frame-count
/// buffer would hold, in order, followed by the prompt's response.
#[tokio::test]
async fn a_reading_client_gets_every_frame_of_a_burst() {
    const DELTAS: usize = 3000;
    let expected: String = (0..DELTAS).map(|i| format!("<{i}>")).collect();
    let (base, _requests) =
        spawn_model_server(vec![burst_turn((0..DELTAS).map(|i| format!("<{i}>")))]);
    let dir = tempfile::tempdir().unwrap();
    let (_child, port) = serve_ws_child(&base, dir.path().to_str().unwrap());
    let mut ws = connect(port, "slowclientburst").await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "go" }),
    )
    .await;

    let mut text = String::new();
    loop {
        match next(&mut ws, Duration::from_secs(20)).await {
            Next::Frame(f) if is_response(&f, "prompt") => break,
            Next::Frame(f) => {
                let ev = &f["event"];
                if f["type"] == "event" && ev["kind"] == "stream" && ev["type"] == "text_delta" {
                    text.push_str(ev["text"].as_str().unwrap_or_default());
                }
            }
            Next::Closed => panic!(
                "a reading client was disconnected mid-burst after {} of {} bytes of text",
                text.len(),
                expected.len()
            ),
            Next::Silent => panic!(
                "a reading client stopped receiving mid-burst after {} of {} bytes of text",
                text.len(),
                expected.len()
            ),
        }
    }
    assert!(
        text == expected,
        "the streamed text has a hole or is out of order"
    );
}

/// A client that stops reading while far more output than its buffer streams is disconnected — its
/// socket closes rather than going silent — and a reconnect re-attaches and is answered.
#[tokio::test]
async fn a_client_that_stops_reading_is_disconnected_and_can_reattach() {
    // ~24 MiB of streamed text: well past the connection's backlog cap plus both ends' socket buffers.
    let chunk = "y".repeat(4096);
    let (base, _requests) =
        spawn_model_server(vec![burst_turn((0..6000).map(|i| format!("<{i}>{chunk}")))]);
    let dir = tempfile::tempdir().unwrap();
    let (_child, port) = serve_ws_child(&base, dir.path().to_str().unwrap());
    let sid = "slowclientstall";
    let mut ws = connect(port, sid).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "go" }),
    )
    .await;
    // Read up to the prompt's ack (past the `ready` greeting).
    loop {
        let Next::Frame(f) = next(&mut ws, Duration::from_secs(10)).await else {
            panic!("no ack");
        };
        if f["type"] == "ack" {
            break;
        }
    }

    // Stop reading while the burst streams.
    tokio::time::sleep(Duration::from_secs(6)).await;

    // Resume. What is already buffered arrives, and then the socket must end — not fall silent.
    let mut frames = 0usize;
    let ended = loop {
        match next(&mut ws, Duration::from_secs(10)).await {
            Next::Frame(f) => {
                assert!(
                    !is_response(&f, "prompt"),
                    "the stalled client was never dropped, so this test proves nothing: grow the burst"
                );
                frames += 1;
            }
            Next::Closed => break true,
            Next::Silent => break false,
        }
    };
    assert!(
        ended,
        "a dropped client's socket stayed open and silent after {frames} frames: it can never know to reconnect"
    );

    // The reconnect is caught up and its commands are answered.
    let mut ws2 = connect(port, sid).await;
    ws_send(&mut ws2, json!({ "type": "get_state", "id": "g1" })).await;
    let mut caught_up = false;
    loop {
        match next(&mut ws2, Duration::from_secs(30)).await {
            Next::Frame(f) if is_response(&f, "get_state") => break,
            Next::Frame(f) => caught_up |= f["type"] == "catchup",
            Next::Closed => panic!("the reconnect was closed"),
            Next::Silent => panic!("the reconnect was never answered"),
        }
    }
    assert!(caught_up, "the reconnect was not caught up");
}
