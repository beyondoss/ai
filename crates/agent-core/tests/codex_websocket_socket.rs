//! `GatewayClient`'s Codex WebSocket transport (`codex_websocket`) over a real TCP socket, driven by a
//! hand-rolled mock WebSocket server (via `tokio-tungstenite`'s own server-side `accept_async` —
//! already a dependency, see `crates/agent-core/Cargo.toml`). Mirrors `tests/client_socket.rs`'s house
//! style (a bare listener standing in for the real backend) but async throughout, since both the
//! client and the mock server here are genuinely async (unlike that file's blocking `std::net`
//! sockets, which suit the plain-HTTP/SSE path fine).
//!
//! Covers the three behaviors the feature exists for: a first turn sends the full transcript with no
//! `previous_response_id`; a second turn on the same session sends only the new tail, chained off the
//! first turn's response id; and a WebSocket handshake failure falls back to the existing HTTP/SSE
//! path transparently, for the exact same request.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use agent_core::client::{Credential, CredentialSource, DirectRouting, RouteOverride};
use agent_core::message::{ContentBlock, StreamEvent};
use agent_core::transport::ModelTransport;
use agent_core::{GatewayClient, Message, ModelRequest};
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// A `CredentialSource` routing every request through Codex's real `RouteOverride::Prefixed` shape —
/// the same fixture `client.rs`'s own in-crate tests use, reachable here since `CredentialSource`/
/// `Credential`/`DirectRouting`/`RouteOverride` are all public.
struct CodexCredential;

#[async_trait::async_trait]
impl CredentialSource for CodexCredential {
    async fn credential(&self) -> agent_core::Result<Credential> {
        Ok(
            Credential::new("test-codex-token", false).with_direct_routing(DirectRouting {
                route: RouteOverride::Prefixed {
                    prefix: "/openai-codex",
                    path: "/backend-api/codex/responses",
                },
                static_headers: vec![("chatgpt-account-id", "acct_test".to_string())],
                copilot_dynamic_headers: false,
                auth_header: None,
                auth_header_prefix: None,
                dialect_override: None,
                deployment_name: None,
                query: None,
                aggregator_host: None,
            }),
        )
    }
}

/// Send one complete, minimal Codex Responses turn over an already-upgraded mock server socket:
/// `response.created` (carrying `response_id`), an opened+streamed+closed text message item, then
/// `response.completed`.
async fn send_turn(ws: &mut WebSocketStream<tokio::net::TcpStream>, response_id: &str, text: &str) {
    let frames = [
        json!({"type": "response.created", "response": {"id": response_id}}),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_1"},
        }),
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": text}),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            },
        }),
        json!({
            "type": "response.completed",
            "response": {"id": response_id, "status": "completed", "usage": {"input_tokens": 5, "output_tokens": 1}},
        }),
    ];
    for frame in frames {
        ws.send(WsMessage::text(frame.to_string()))
            .await
            .expect("mock server send");
    }
}

/// Read one client text frame and parse it as JSON — the request body the mock server asserts on.
async fn recv_request(ws: &mut WebSocketStream<tokio::net::TcpStream>) -> Value {
    let message = ws
        .next()
        .await
        .expect("a client frame")
        .expect("a well-formed client frame");
    serde_json::from_str(message.into_text().expect("text frame").as_str()).expect("valid json")
}

#[tokio::test]
async fn first_turn_sends_the_full_transcript_with_no_previous_response_id() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let mut ws = tokio_tungstenite::accept_async(stream)
            .await
            .expect("ws handshake");
        let request = recv_request(&mut ws).await;
        send_turn(&mut ws, "resp_1", "hi there").await;
        request
    });

    let client =
        GatewayClient::with_credential_source(format!("http://{addr}"), Arc::new(CodexCredential))
            .expect("client");
    let req = ModelRequest::new("gpt-5-codex", vec![Message::user("hi")], 200)
        .with_cache_key("sess-first-turn");
    let mut stream = client.stream(req).await.expect("stream");
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("event"));
    }

    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "hi there")),
        "expected the mock server's streamed text to reach the caller: {events:?}"
    );
    assert!(matches!(
        events.last(),
        Some(StreamEvent::MessageStop { .. })
    ));

    let request = server.await.expect("server task");
    assert!(
        request.get("previous_response_id").is_none(),
        "a first turn must not send previous_response_id: {request}"
    );
    assert_eq!(
        request["input"].as_array().expect("input array").len(),
        1,
        "a first turn must send the whole (one-message) transcript: {request}"
    );
    assert_eq!(request["type"], "response.create");
}

#[tokio::test]
async fn second_turn_on_the_same_session_sends_only_the_new_delta() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        let mut ws = tokio_tungstenite::accept_async(stream)
            .await
            .expect("ws handshake");

        let turn1 = recv_request(&mut ws).await;
        send_turn(&mut ws, "resp_1", "hi there").await;

        // The connection must be *reused*, not reconnected — a second `listener.accept()` here would
        // hang forever (nothing else ever connects), which is exactly what proves reuse rather than a
        // fresh connection per turn.
        let turn2 = recv_request(&mut ws).await;
        send_turn(&mut ws, "resp_2", "still good").await;

        (turn1, turn2)
    });

    let client =
        GatewayClient::with_credential_source(format!("http://{addr}"), Arc::new(CodexCredential))
            .expect("client");

    let req1 = ModelRequest::new("gpt-5-codex", vec![Message::user("hi")], 200)
        .with_cache_key("sess-delta");
    let mut stream1 = client.stream(req1).await.expect("stream 1");
    while stream1.next().await.is_some() {}

    // Simulates what `Agent::run_events` would have persisted after turn 1: the assistant's own reply,
    // built from the *exact* `id`/text the mock server's `response.output_item.done` event carried
    // (`msg_1`/"hi there") — matching what `dialect::openai_responses::push_assistant_content` would
    // reconstruct from that same `ContentBlock`, which is what the delta comparison's prefix check
    // needs to line up against the connection's cached baseline (see `codex_websocket`'s own module
    // doc comment on why this crate harvests the raw wire item rather than needing a live
    // `AssistantMessage` the transport layer doesn't have).
    let history = vec![
        Message::user("hi"),
        Message::assistant(vec![ContentBlock::Text {
            text: "hi there".to_string(),
            id: Some("msg_1".to_string()),
            phase: None,
        }]),
        Message::user("how are you"),
    ];
    let req2 = ModelRequest::new("gpt-5-codex", history, 200).with_cache_key("sess-delta");
    let mut stream2 = client.stream(req2).await.expect("stream 2");
    let mut events2 = Vec::new();
    while let Some(ev) = stream2.next().await {
        events2.push(ev.expect("event"));
    }
    assert!(
        events2
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "still good")),
        "expected the second turn's own streamed text: {events2:?}"
    );

    let (turn1, turn2) = server.await.expect("server task");
    assert!(turn1.get("previous_response_id").is_none());
    assert_eq!(turn1["input"].as_array().expect("input array").len(), 1);

    assert_eq!(
        turn2["previous_response_id"], "resp_1",
        "the second turn must chain off the first turn's own response id: {turn2}"
    );
    let turn2_input = turn2["input"].as_array().expect("input array");
    assert_eq!(
        turn2_input.len(),
        1,
        "the delta must be exactly the new user message, not the whole transcript: {turn2}"
    );
    assert_eq!(
        turn2_input[0]["content"][0]["type"], "input_text",
        "unexpected delta item shape: {turn2}"
    );
    assert!(
        turn2.to_string().contains("how are you"),
        "the delta must carry the new user turn's text: {turn2}"
    );
}

#[tokio::test]
async fn a_failed_websocket_handshake_falls_back_to_http_sse_transparently() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        // First connection: the WebSocket upgrade attempt — reject it outright with a plain non-101
        // HTTP response, simulating the transport being unavailable.
        let (mut stream, _) = listener.accept().await.expect("accept 1");
        let mut buf = [0u8; 8192];
        let _ = stream.read(&mut buf).await;
        stream
            .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await
            .expect("write rejection");
        let _ = stream.shutdown().await;
        drop(stream);

        // Second connection: the resulting HTTP/SSE fallback request for the *same* turn.
        let (mut stream, _) = listener.accept().await.expect("accept 2");
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).await.expect("read sse request");
        let request = String::from_utf8_lossy(&buf[..n]).to_string();
        let sse_body = "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n\
             data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"fallback ok\"}\n\n\
             data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"fallback ok\"}]}}\n\n\
             data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse_body}"
        );
        stream
            .write_all(resp.as_bytes())
            .await
            .expect("write sse response");
        stream.flush().await.expect("flush");
        request
    });

    let client =
        GatewayClient::with_credential_source(format!("http://{addr}"), Arc::new(CodexCredential))
            .expect("client");
    let req = ModelRequest::new("gpt-5-codex", vec![Message::user("hi")], 200)
        .with_cache_key("sess-fallback");
    let mut stream = client.stream(req).await.expect("stream");
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("event — the fallback must succeed via plain HTTP/SSE"));
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "fallback ok")),
        "expected the HTTP/SSE fallback's own streamed text to reach the caller: {events:?}"
    );
    assert!(matches!(
        events.last(),
        Some(StreamEvent::MessageStop { .. })
    ));

    let request = server.await.expect("server task");
    assert!(
        request.starts_with("POST "),
        "the fallback must be a plain HTTP POST, not another WebSocket upgrade attempt: {request}"
    );
    assert!(
        request.contains("/openai-codex/backend-api/codex/responses"),
        "the fallback must still hit Codex's real path: {request}"
    );
}

/// pi-parity: the Codex HTTP/SSE fallback path zstd-compresses its request body
/// (`codex_websocket::compress_sse_fallback_body`, wired into `client.rs::send_with_retry` gated on
/// `is_codex_sse_fallback`). Reads the raw request bytes (not `String::from_utf8_lossy`, which would
/// corrupt a binary zstd body) to check the header and decompress the body directly.
#[tokio::test]
async fn the_http_sse_fallback_sends_a_zstd_compressed_body_with_the_matching_header() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        // First connection: reject the WebSocket upgrade, forcing the HTTP/SSE fallback.
        let (mut stream, _) = listener.accept().await.expect("accept 1");
        let mut buf = [0u8; 8192];
        let _ = stream.read(&mut buf).await;
        stream
            .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await
            .expect("write rejection");
        let _ = stream.shutdown().await;
        drop(stream);

        // Second connection: read the *raw* fallback request bytes — the body may be binary zstd.
        let (mut stream, _) = listener.accept().await.expect("accept 2");
        let mut raw = Vec::new();
        // The mock server doesn't know the exact body length up front; read until the client stops
        // sending or a generous cap is hit, then respond so `stream()` doesn't hang waiting on a
        // response that never comes.
        let mut chunk = [0u8; 8192];
        loop {
            match tokio::time::timeout(
                std::time::Duration::from_millis(200),
                stream.read(&mut chunk),
            )
            .await
            {
                Ok(Ok(0)) | Err(_) => break,
                Ok(Ok(n)) => raw.extend_from_slice(&chunk[..n]),
                Ok(Err(e)) => panic!("read fallback request: {e}"),
            }
        }
        let sse_body = "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n\
             data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"compressed ok\"}\n\n\
             data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"compressed ok\"}]}}\n\n\
             data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse_body}"
        );
        stream
            .write_all(resp.as_bytes())
            .await
            .expect("write sse response");
        stream.flush().await.expect("flush");
        raw
    });

    let client =
        GatewayClient::with_credential_source(format!("http://{addr}"), Arc::new(CodexCredential))
            .expect("client");
    let req = ModelRequest::new("gpt-5-codex", vec![Message::user("zstd please")], 200)
        .with_cache_key("sess-zstd-fallback");
    let mut stream = client.stream(req).await.expect("stream");
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("event — the compressed fallback must still succeed"));
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "compressed ok")),
        "expected the fallback's own streamed text to reach the caller: {events:?}"
    );

    let raw = server.await.expect("server task");
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a header/body separator");
    let (head, body) = (&raw[..split], &raw[split + 4..]);
    let head = String::from_utf8_lossy(head).to_lowercase();
    assert!(
        head.contains("content-encoding: zstd"),
        "the fallback request must advertise a zstd-compressed body: {head}"
    );
    assert!(
        head.contains("content-type: application/json"),
        "the fallback request must still declare a JSON content type: {head}"
    );

    let decompressed = zstd::stream::decode_all(body).expect("the body must be a valid zstd frame");
    let parsed: Value =
        serde_json::from_slice(&decompressed).expect("decompressed body must be JSON");
    assert!(
        parsed.to_string().contains("zstd please"),
        "the decompressed body must carry the real request content: {parsed}"
    );
}
