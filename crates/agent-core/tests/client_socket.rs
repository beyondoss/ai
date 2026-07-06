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
        index: 0,
        text: "hi over the wire".into()
    }));
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::Usage(u) if u.input_tokens == 7 && u.output_tokens == 5
    )));
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
            let lead = bytes
                .iter()
                .position(|&b| b == 0xF0)
                .expect("emoji present");
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
            index: 0,
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

#[tokio::test]
async fn gateway_client_gives_actionable_guidance_on_a_401() {
    // Pi-parity fix: a live 401 (the gateway itself rejecting the key) used to surface as a bare
    // "gateway returned 401: <body>" — identical formatting to any other 4xx, with no hint at what to
    // actually do about it. Unlike a 403 (still just a generic error, see the test above), 401
    // specifically must name the actual cause and point at the fix.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"type\":\"authentication_error\"}}",
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
    let err = first.expect_err("a 401 must surface as a stream error");
    let msg = err.to_string();
    assert!(
        msg.contains("API key") && (msg.contains("invalid") || msg.contains("expired")),
        "expected actionable 401 guidance naming the key, got: {msg}"
    );
    assert!(
        msg.contains("authentication_error"),
        "the upstream detail must still be included, not just the new guidance: {msg}"
    );
}

#[tokio::test]
async fn gateway_client_bounds_memory_on_an_oversized_error_body() {
    // A misconfigured proxy or hostile upstream can return an arbitrarily large error page; the
    // client must stop reading well before the full body arrives instead of buffering it all into
    // memory ahead of the display truncation. `Content-Length` is set so the body isn't itself framed
    // as an SSE stream — this exercises the non-2xx error path, not `LineFramer`.
    const BODY_LEN: usize = 3 * 1024 * 1024; // comfortably over the 1MB read cap
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let header = format!(
                "HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nContent-Length: {BODY_LEN}\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&vec![b'x'; BODY_LEN]);
            let _ = stream.flush();
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
    let err = first
        .expect_err("a 403 must surface as a stream error")
        .to_string();

    // The displayed message is always capped to a few thousand chars; what distinguishes a bounded
    // read from an unbounded one is the *reported omitted count* — an unbounded read would report
    // omitting the full ~3MB body, a bounded one only the ~1MB it actually buffered.
    let omitted: usize = err
        .rsplit("[truncated ")
        .next()
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("expected a `[truncated N chars]` marker, got: {err}"));
    assert!(
        omitted < 2 * 1024 * 1024,
        "expected the read to stop well short of the full {BODY_LEN}-byte body, omitted count was {omitted}"
    );
}

/// Capture the raw bytes of the first request a one-shot server receives, then answer with a minimal
/// empty-body SSE response. Returns the shared buffer the caller reads after driving the request.
fn spawn_request_capturing_server() -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let captured2 = captured.clone();
    thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            *captured2.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).into_owned();
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            );
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}"), captured)
}

#[tokio::test]
async fn gateway_client_omits_session_affinity_headers_for_a_non_fireworks_responses_request() {
    // pi-parity (models/dialects pass, Task D) correction: these headers are gated to Fireworks-hosted
    // models specifically (pi's own `compat.sendSessionAffinityHeaders` is a per-provider catalogue
    // flag, not a blanket "every OpenAI-wire dialect" rule) — this test used to assert the *opposite*
    // (that native "gpt-5" got the header unconditionally), which was the exact over-broad behavior
    // this fix closed. See `client::tests::session_affinity_headers_are_sent_for_a_fireworks_chat_
    // completions_request`/`..._are_absent_for_a_non_fireworks_chat_completions_request` (`client.rs`)
    // for the Fireworks-positive/negative pair on the Chat Completions dialect — no current Fireworks
    // id reaches the Responses dialect at all (`is_fireworks_anthropic_wire_model` routes almost every
    // Fireworks id to Anthropic instead), so this dialect only ever has the negative case to prove.
    let (base, captured) = spawn_request_capturing_server();
    let client = GatewayClient::new(base, "bai_v1.test").expect("client");
    let req =
        ModelRequest::new("gpt-5", vec![Message::user("hi")], 64).with_cache_key("session-abc");
    let mut stream = client.stream(req).await.expect("stream");
    let _ = stream.next().await;

    let request = captured.lock().unwrap().clone();
    let lower = request.to_ascii_lowercase();
    assert!(
        !lower.contains("x-client-request-id"),
        "native (non-Fireworks) gpt-5 must not get the session-affinity header: {request}"
    );
    assert!(
        !lower.contains("session_id:"),
        "native (non-Fireworks) gpt-5 must not get the session_id header either: {request}"
    );
}

#[tokio::test]
async fn gateway_client_omits_x_client_request_id_without_a_cache_key() {
    // Fireworks-hosted (`glm-5p2` is the one current Fireworks id on the Chat Completions dialect —
    // see the test above) but no `cache_key`: still nothing to route by, so the headers must stay
    // omitted even on an otherwise-eligible route.
    let (base, captured) = spawn_request_capturing_server();
    let client = GatewayClient::new(base, "bai_v1.test").expect("client");
    let req = ModelRequest::new("accounts/fireworks/models/glm-5p2", vec![Message::user("hi")], 64); // no cache_key
    let mut stream = client.stream(req).await.expect("stream");
    let _ = stream.next().await;

    let request = captured.lock().unwrap().clone();
    let lower = request.to_ascii_lowercase();
    assert!(
        !lower.contains("x-client-request-id"),
        "no session id to route by — the header must be omitted: {request}"
    );
    assert!(
        !lower.contains("session_id"),
        "no session id to route by — the header must be omitted: {request}"
    );
}

#[tokio::test]
async fn gateway_client_omits_x_client_request_id_for_anthropic() {
    // The header is Responses-API-specific; Anthropic (and OpenAI Chat Completions) never send it,
    // even with a cache_key set — matches pi's own dialect split.
    let (base, captured) = spawn_request_capturing_server();
    let client = GatewayClient::new(base, "bai_v1.test").expect("client");
    let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64)
        .with_cache_key("session-abc");
    let mut stream = client.stream(req).await.expect("stream");
    let _ = stream.next().await;

    let request = captured.lock().unwrap().clone();
    let lower = request.to_ascii_lowercase();
    assert!(
        !lower.contains("x-client-request-id"),
        "the Anthropic dialect must not send the OpenAI-Responses-specific header: {request}"
    );
    assert!(
        !lower.contains("session_id"),
        "the Anthropic dialect must not send the OpenAI-Responses-specific header: {request}"
    );
}

#[tokio::test]
async fn gateway_client_retries_transient_503_then_succeeds() {
    // The server returns a retryable 503 on the first connection and the real SSE body on the second.
    // The client must transparently retry and deliver the events — a transient gateway hiccup should
    // not vaporize the request.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        // First attempt: 503 with a tiny Retry-After so the backoff stays fast.
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            let _ = stream.flush();
        }
        // Second attempt: the real stream.
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{ANTHROPIC_SSE}"
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });

    let client = GatewayClient::new(format!("http://{addr}"), "bai_v1.test").expect("client");
    let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64);
    let mut stream = client.stream(req).await.expect("stream");
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("event after retry"));
    }
    assert!(
        events.contains(&StreamEvent::TextDelta {
            index: 0,
            text: "hi over the wire".into()
        }),
        "retry must transparently deliver the real stream, got: {events:?}"
    );
}

#[tokio::test]
async fn gateway_client_retries_a_connection_reset_mid_send_then_succeeds() {
    // Found live (a real fault-injecting proxy dropping connections against a genuine Anthropic/OpenAI
    // gateway) before it was ever caught here: `is_retryable_send_error`'s own doc comment ("connection-
    // level failures: refused, reset, timed out... worth retrying") already promised this class was
    // covered, but the code only checked `e.is_timeout() || e.is_connect()` — neither matches a
    // connection that accepted fine and was then reset *while sending the request*, which reqwest
    // reports as `Kind::Request` (`e.is_request()`), a different bucket than `is_connect()`'s narrower
    // "the TCP handshake itself failed" (refused/unreachable). The first connection here is accepted
    // then dropped with zero bytes written back — exactly that reset-after-connect shape, not a 503 or
    // a timeout — so this is real coverage for a genuinely different failure mode than the sibling test
    // above, not a duplicate of it.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    thread::spawn(move || {
        // First attempt: accept, then drop immediately — a connection reset mid-send, not a timeout and
        // not a refused/unreachable connect failure.
        if let Ok((stream, _)) = listener.accept() {
            drop(stream);
        }
        // Second attempt: the real stream.
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{ANTHROPIC_SSE}"
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });

    let client = GatewayClient::new(format!("http://{addr}"), "bai_v1.test").expect("client");
    let req = ModelRequest::new("claude-opus-4-8", vec![Message::user("hi")], 64);
    let mut stream = client.stream(req).await.expect("stream");
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("event after retry"));
    }
    assert!(
        events.contains(&StreamEvent::TextDelta {
            index: 0,
            text: "hi over the wire".into()
        }),
        "a connection reset mid-send must be retried transparently, not surfaced as a hard failure on \
         the very first attempt: {events:?}"
    );
}
