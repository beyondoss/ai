//! `GatewayClient` over a real TCP socket.
//!
//! A bare TCP server writes a canned `text/event-stream` response (connection-close framing, like a
//! provider streaming SSE). This proves the client's chunked-body line framing + dialect decoding
//! end to end over a socket — the unit tests cover decoding from a string; this covers it from the
//! wire. No hyper, no gateway, no network.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use agent_core::message::StreamEvent;
use agent_core::transport::ModelTransport;
use agent_core::{GatewayClient, Message, ModelRequest};
use futures::StreamExt;

/// Spawn a one-shot server that returns `body` as an SSE response, and return its base URL.
fn spawn_sse_server(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            // Drain the request (headers + small JSON body) so the client's write completes.
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });
    format!("http://{addr}")
}

const ANTHROPIC_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi over the wire\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";

#[tokio::test]
async fn gateway_client_decodes_sse_over_socket() {
    let base = spawn_sse_server(ANTHROPIC_SSE);
    let client = GatewayClient::new(base, "bai_v1.test").expect("client");

    let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64);
    let mut stream = client.stream(req).await.expect("stream");

    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("event"));
    }

    assert!(events.contains(&StreamEvent::TextDelta {
        text: "hi over the wire".into()
    }));
    assert!(events.contains(&StreamEvent::Usage {
        input_tokens: 7,
        output_tokens: 5
    }));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::MessageStop { .. })
    ));
}

// Same shape as ANTHROPIC_SSE but the text carries a 4-byte emoji (🌍 = F0 9F 8C 8D) that the
// server will split across two TCP writes.
const EMOJI_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi 🌍 world\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";

/// A regression guard for chunk-boundary UTF-8 corruption: the server flushes the response in two
/// halves split *inside* the emoji's 4-byte sequence, so the client must buffer the partial bytes
/// across chunks. Per-chunk lossy decoding would turn the emoji into two U+FFFD replacements.
#[tokio::test]
async fn gateway_client_reassembles_utf8_split_across_chunks() {
    use std::time::Duration;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{EMOJI_SSE}"
            );
            let bytes = resp.into_bytes();
            // Split two bytes into the emoji's 4-byte sequence (lead byte 0xF0).
            let lead = bytes.iter().position(|&b| b == 0xF0).expect("emoji present");
            let split = lead + 2;
            let _ = stream.write_all(&bytes[..split]);
            let _ = stream.flush();
            thread::sleep(Duration::from_millis(25)); // force a chunk boundary mid-character
            let _ = stream.write_all(&bytes[split..]);
            let _ = stream.flush();
        }
    });

    let client = GatewayClient::new(format!("http://{addr}"), "bai_v1.test").expect("client");
    let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64);
    let mut stream = client.stream(req).await.expect("stream");
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("event"));
    }

    assert!(
        events.contains(&StreamEvent::TextDelta {
            text: "hi 🌍 world".into()
        }),
        "emoji split across chunks must reassemble intact, got: {events:?}"
    );
}

#[tokio::test]
async fn gateway_client_surfaces_http_error() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                b"HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"denied\"}",
            );
        }
    });
    let client = GatewayClient::new(format!("http://{addr}"), "bai_v1.test").expect("client");
    let mut stream = client
        .stream(ModelRequest::new(
            "claude-opus-4-8",
            vec![Message::user("hi")],
            64,
        ))
        .await
        .expect("stream");
    let first = stream.next().await.expect("an item");
    assert!(first.is_err(), "a 403 must surface as a stream error");
}
