//! End-to-end: a client that gives up must not be counted against the provider.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).
//!
//! Cancellation is routine for a coding agent — a user hits ESC on a slow turn. The gateway used to
//! record every such abort as a provider failure, because `logging` saw an error with no response
//! head and blamed the upstream for it. With `circuit_breaker_threshold` cancellations inside
//! `circuit_breaker_window_secs` that opened the breaker and 503'd *everyone*; and with
//! `half_open_permits` at 1, a cancel-prone request drawn as the recovery probe reopened it every
//! time, so the breaker could not recover while users were cancelling.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code — allow the
// panic-surface restriction lints denied workspace-wide in `[workspace.lints.clippy]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use std::time::Duration;

fn body() -> String {
    r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#.to_string()
}

/// Client aborts, upstream is healthy → the breaker must stay closed.
///
/// The threshold is 2 and the client abandons 5 requests, so a gateway that counted downstream
/// aborts would have opened the breaker several times over and started rejecting.
/// claim: REL-9, REL-6
#[tokio::test]
async fn client_cancellations_do_not_open_the_providers_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    // Slower than the client's patience below, so every one of those requests is abandoned
    // mid-flight — while the upstream itself stays perfectly healthy.
    let mock = MockUpstream::start(Mode::Slow(3_000)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        &sk,
    );

    let impatient = reqwest::Client::builder()
        .timeout(Duration::from_millis(150))
        .build()
        .unwrap();
    for _ in 0..5 {
        let outcome = impatient
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {vkey}"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await;
        // The upstream sleeps for 3s, so the only way the gateway answers within 150ms is by
        // *not asking it* — i.e. the breaker has already opened and is fast-failing. That is
        // precisely the regression under test, so name it rather than reporting a bare `is_err`.
        if let Ok(r) = outcome {
            panic!(
                "gateway answered {} in under 150ms against a 3s upstream — it short-circuited, \
                 which means earlier cancellations were recorded as provider failures and opened \
                 the breaker",
                r.status(),
            );
        }
    }

    let metrics = gw.metrics().await;
    assert_eq!(
        parse_metric(&metrics, "ai_rejections_total", "circuit_open"),
        0.0,
        "client cancellations must not open the breaker:\n{metrics}"
    );

    // And the breaker is genuinely still closed, not merely un-observed: a patient request against
    // the same provider is served rather than fast-failed with a 503.
    let patient = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let resp = patient
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "the provider was healthy throughout; its breaker must still admit traffic",
    );
}

/// The other half of the same rule: a genuinely broken provider must still trip the breaker. Without
/// this, "stop blaming the provider for client aborts" could be satisfied by never blaming it at all.
/// claim: REL-6
#[tokio::test]
async fn upstream_failures_still_open_the_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Status(500)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    for _ in 0..6 {
        let _ = client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {vkey}"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await;
    }

    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_rejections_total", "circuit_open") >= 1.0,
        "sustained 5xx must still open the breaker:\n{metrics}"
    );
}

/// A reused connection that fails after the upstream drained the whole body (the mock reads the
/// request, then closes) is **not** retried. Pingora would resend it on its own, without calling
/// `fail_to_connect`, but the provider may already be generating and billing that request, so the
/// gateway declines the reuse retry once the body was delivered (`error_while_proxy`) and the
/// client gets a JSON 502. This test used to assert the retry; that resend is the duplicate spend
/// D09 removed. The breaker still sees exactly one outcome per attempt.
///
/// The mock kills any request that is not the first on its connection, so the scenario fires the
/// moment pingora reuses a pooled connection — whenever that happens to be, rather than on a fixed
/// request number that may well land on a fresh connection under load.
/// claim: REL-1
#[tokio::test]
async fn a_reused_connection_failure_after_the_body_is_not_resent() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::CloseOnReusedConnection).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    // Streaming + managed + OpenAI chat = the inject-eligible path, buffered and spliced. The mock
    // drains the request before killing the connection, so the failure is on the response-header
    // read, after the body was written.
    let filler = "y".repeat(8 * 1024);
    let body = format!(
        r#"{{"model":"gpt-4o-mini","stream":true,"messages":[{{"role":"user","content":"{filler}"}}]}}"#
    );

    // Sequential, so each request has a pooled connection available to reuse.
    const SENT: usize = 6;
    let mut failed = 0;
    for i in 0..SENT {
        let resp = client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {vkey}"))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap_or_else(|e| panic!("request {i}: {e}; log:\n{}", gw.log()));
        match resp.status().as_u16() {
            200 => {}
            502 => {
                failed += 1;
                assert!(
                    resp.headers().contains_key("x-beyond-request-id"),
                    "request {i}: the 502 carries the request id"
                );
                let text = resp.text().await.unwrap_or_default();
                let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                assert!(v["error"]["message"].is_string(), "request {i}: {text}");
            }
            s => panic!("request {i}: unexpected status {s}"),
        }
    }
    assert!(
        failed > 0,
        "no reused-connection failure occurred, so the scenario did not fire and this test proved \
         nothing"
    );
    assert_eq!(
        mock.hits(),
        SENT,
        "a delivered body was resent ({} upstream requests for {SENT} client requests)",
        mock.hits(),
    );
}

/// An HTTP/1.1 upstream that closes a keep-alive connection with the next request unread in it, the
/// idle-close race: it serves the first request on every connection, then waits for the whole of
/// the second to arrive and closes without reading it, so its kernel answers with a reset. Returns
/// the port and how many times that fired.
async fn closes_reused_connections_unread() -> (u16, std::sync::Arc<std::sync::atomic::AtomicUsize>)
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    const OK: &str = r#"{"id":"c","object":"chat.completion","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;
    let listener = beyond_ai_test_support::ports::tokio_listener().await;
    let port = listener.local_addr().unwrap().port();
    let fired = std::sync::Arc::new(AtomicUsize::new(0));
    let count = fired.clone();
    // Bytes that make up one whole request at the front of `buf`, if they are all there.
    fn whole(buf: &[u8]) -> Option<usize> {
        let head = buf.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
        let text = String::from_utf8_lossy(&buf[..head]).to_ascii_lowercase();
        let len = text
            .lines()
            .find_map(|l| l.strip_prefix("content-length:"))
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        (buf.len() >= head + len).then_some(head + len)
    }
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            let count = count.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 16 * 1024];
                while whole(&buf).is_none() {
                    match s.read(&mut chunk).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buf.extend_from_slice(&chunk[..n]),
                    }
                }
                let reply = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{OK}",
                    OK.len()
                );
                if s.write_all(reply.as_bytes()).await.is_err() {
                    return;
                }
                // The next request: wait until all of it sits in the socket, then close unread.
                let mut peek = vec![0u8; 64 * 1024];
                loop {
                    match s.peek(&mut peek).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) if whole(&peek[..n]).is_some() => break,
                        Ok(_) => tokio::time::sleep(Duration::from_millis(5)).await,
                    }
                }
                count.fetch_add(1, Ordering::SeqCst);
                drop(s);
            });
        }
    });
    (port, fired)
}

/// A pooled connection the provider closed with the request unread (it answers with a reset, before
/// any response byte) never reached a server, so the gateway resends it on a fresh connection, as
/// pingora does for a reused connection. A small body is read straight through to the upstream:
/// nothing reads it ahead of connecting, so "the body was read" is not mistaken for "the provider
/// took it" (D80). A clean end-of-file after the body stays unretried (see the test above, D09).
/// claim: REL-1
/// defect: D80
#[tokio::test]
async fn a_reused_connection_reset_before_reading_is_resent() {
    let (port, fired) = closes_reused_connections_unread().await;
    let (pubkey, _sk) = test_keypair(1);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{port}"),
        &b64(&pubkey),
    )
    .start()
    .await;
    let client = reqwest::Client::new();
    for i in 0..6 {
        let resp = client
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", "Bearer sk-byo-test")
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap_or_else(|e| panic!("request {i}: {e}; log:\n{}", gw.log()));
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        assert_eq!(status, 200, "request {i}: {text}\n{}", gw.log());
    }
    assert!(
        fired.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "no pooled connection was reused, so the scenario did not fire"
    );
}

/// A retry pingora *does* make — the managed 429 key walk — replays its buffered request body
/// through `request_body_filter`, which is what `RequestCtx::reset_request_body_phase` exists to
/// make idempotent. The retried body must be whole and well-formed, not the original with a
/// replayed prefix concatenated onto it, and carry the usage splice exactly once.
#[tokio::test]
async fn a_key_walk_retry_replays_the_body_exactly_once() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::ThrottleKey("sk-replay-a")).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-replay-a", "sk-replay-b"])
        .start()
        .await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        &sk,
    );
    // 8 KiB: a real buffered body, comfortably under pingora's 64 KiB retry buffer, so the retry
    // genuinely replays rather than declining to.
    let filler = "y".repeat(8 * 1024);
    let body = format!(
        r#"{{"model":"gpt-4o-mini","stream":true,"messages":[{{"role":"user","content":"{filler}"}}]}}"#
    );
    let resp = reqwest::Client::new()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(mock.hits(), 2, "the throttled key, then the walked one");

    let cap = mock.captured().expect("the retry reached the upstream");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-replay-b"));
    let received: serde_json::Value = serde_json::from_slice(&cap.body).unwrap_or_else(|e| {
        panic!("retried body is not valid JSON ({e}) — the replay was appended, not reset")
    });
    assert_eq!(received["model"], "gpt-4o-mini");
    assert_eq!(received["messages"][0]["content"], filler);
    assert_eq!(
        received["stream_options"]["include_usage"], true,
        "the usage injection must still be spliced exactly once on the retried attempt",
    );
}

/// The same on a catalog walk: a pooled connection the provider closed with the request unread is
/// the connection's failure, not the candidate's, so the walk retries the same candidate on a fresh
/// connection, even with no other candidate to walk to.
/// claim: REL-1
/// defect: D80
#[tokio::test]
async fn a_catalog_walk_resends_a_reused_connection_reset_before_reading() {
    let (port, fired) = closes_reused_connections_unread().await;
    let (pubkey, sk) = test_keypair(1);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{port}"),
        &b64(&pubkey),
    )
    .providers(&["openai"])
    .start()
    .await;
    let vkey = billing_vkey(&sk, 80);
    let client = reqwest::Client::new();
    for i in 0..6 {
        let resp = client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {vkey}"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap_or_else(|e| panic!("request {i}: {e}; log:\n{}", gw.log()));
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        assert_eq!(status, 200, "request {i}: {text}\n{}", gw.log());
    }
    assert!(
        fired.load(std::sync::atomic::Ordering::SeqCst) > 0,
        "no pooled connection was reused, so the scenario did not fire"
    );
}

/// A listener that accepts connections, reads whatever arrives and never answers; returns its port
/// and how many connections it has accepted.
async fn silent_upstream() -> (u16, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use tokio::io::AsyncReadExt;
    let listener = beyond_ai_test_support::ports::tokio_listener().await;
    let port = listener.local_addr().unwrap().port();
    let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = accepted.clone();
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                while matches!(s.read(&mut buf).await, Ok(n) if n > 0) {}
            });
        }
    });
    (port, accepted)
}

/// A client that resets its connection mid-upload ends its request: the walk does not fail over to
/// the next candidate (or resend to the same one) with a body nobody will finish sending. (A clean
/// close reads as the body's end, so the request counts as delivered and is not resent either.)
/// claim: REL-9
#[tokio::test]
async fn a_client_gone_mid_upload_is_not_failed_over() {
    use std::sync::atomic::Ordering::SeqCst;
    use tokio::io::AsyncWriteExt;
    let (primary, primary_conns) = silent_upstream().await;
    let (fallback, fallback_conns) = silent_upstream().await;
    let (pubkey, sk) = test_keypair(1);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{primary}"),
        &b64(&pubkey),
    )
    .providers(&["openai", "openrouter"])
    .provider_authority("openrouter", &format!("127.0.0.1:{fallback}"))
    .start()
    .await;
    let payload = body();
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let head = format!(
        "POST /auto/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer {}\r\n\
         content-type: application/json\r\nx-beyond-model: gpt-4o-mini\r\n\
         x-beyond-order: openai,openrouter\r\ncontent-length: {}\r\n\r\n",
        billing_vkey(&sk, 81),
        payload.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(&payload.as_bytes()[..payload.len() / 2])
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while primary_conns.load(SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the request reached the primary");
    tokio::time::sleep(Duration::from_millis(200)).await;
    s.set_zero_linger().unwrap();
    drop(s);
    // The abort ends the request in `logging`; nothing is sent anywhere after it.
    gw.wait_for_log_line(&["upstream request errored"]).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        (primary_conns.load(SeqCst), fallback_conns.load(SeqCst)),
        (1, 0),
        "log:\n{}",
        gw.log()
    );
}

/// A provider-routed request whose client stalls mid-upload on a pooled connection ends at the read
/// timeout, and is not resent: the timeout is a liveness verdict on an attempt that may still be
/// running, not a reused connection's failure before delivery (D248's `ReusedOnly` rule is for a
/// body write that met a closed stream only).
/// claim: REL-1
#[tokio::test]
async fn a_read_timeout_on_a_reused_connection_is_not_resent() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mock = ReplyUpstream::start(|_, _| Reply::ok()).await;
    let (pubkey, _sk) = test_keypair(1);
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("read_timeout_secs = 1")
        .start()
        .await;
    // Pools a connection to the upstream.
    let first = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(first.status().as_u16(), 200);
    let payload = body();
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let head = format!(
        "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: gw\r\n\
         authorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\n\
         content-length: {}\r\n\r\n",
        payload.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(&payload.as_bytes()[..payload.len() / 2])
        .await
        .unwrap();
    // The rest never comes: the gateway answers once the upstream read times out.
    let mut buf = [0u8; 4096];
    let n = tokio::time::timeout(Duration::from_secs(10), s.read(&mut buf))
        .await
        .expect("an answer")
        .unwrap_or(0);
    let text = String::from_utf8_lossy(&buf[..n]);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(mock.hits(), 2, "{text}\nlog:\n{}", gw.log());
}

/// Send `head` and then `sent` (the start of a body whose rest never comes) on a raw HTTP/1.1
/// connection that stays open, and return the gateway's answer (status line and all).
async fn h1_stalled_upload(port: u16, head: &str, sent: &str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(sent.as_bytes()).await.unwrap();
    let mut buf = [0u8; 4096];
    let n = tokio::time::timeout(Duration::from_secs(10), s.read(&mut buf))
        .await
        .expect("an answer")
        .unwrap_or(0);
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// The same over h2c: one stream whose headers declare `content-length` and whose single DATA
/// frame carries only `sent`, the stream left open. Returns the status and the response body.
async fn h2_stalled_upload(port: u16, headers: &[(&str, String)], sent: &str) -> (u16, String) {
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let (mut client, conn) = h2::client::handshake(tcp).await.unwrap();
    tokio::spawn(conn);
    let mut req = http::Request::post("http://gw/auto/chat/completions");
    for (k, v) in headers {
        req = req.header(*k, v.as_str());
    }
    let (resp, mut send) = client.send_request(req.body(()).unwrap(), false).unwrap();
    send.send_data(bytes::Bytes::copy_from_slice(sent.as_bytes()), false)
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(10), resp)
        .await
        .expect("an answer")
        .unwrap();
    let status = resp.status().as_u16();
    let mut body = resp.into_body();
    let mut text = Vec::new();
    while let Some(chunk) = body.data().await {
        let Ok(chunk) = chunk else { break };
        let _ = body.flow_control().release_capacity(chunk.len());
        text.extend_from_slice(&chunk);
    }
    drop(send);
    (status, String::from_utf8_lossy(&text).into_owned())
}

/// A managed client that stalls mid-upload, connection open, ends its own request: the upstream's
/// read timeout fires on a provider still waiting for the client's bytes, which is the client's
/// stall, not the provider failing (D260). So the walk does not fail over to the next candidate,
/// the candidate's breaker hears nothing (the next request still reaches it, with a threshold of
/// 1), no pool key is cooled, the client gets a 408, and the half a provider was sent is not
/// billed.
///
/// Over HTTP/1.1 and h2c, each a `Content-Length` body with bytes still to come. h2c is the shape
/// that reaches this in production: pingora's HTTP/2 server has no body read timeout, so the
/// upstream's `read_timeout_secs` is the only clock on a stalled stream. Over HTTP/1.1 pingora's
/// own 60 s body read timeout (a 408) wins against the default 600 s; this one runs at 1 s. A
/// chunked body cannot stall here: with no declared length a catalog walk reads the whole body
/// before it connects (see the chunked test below for one that streams).
/// claim: REL-1, SEC-16, BIL-3
/// defect: D260
#[tokio::test]
async fn a_client_stalled_mid_upload_is_not_failed_over() {
    use std::sync::atomic::Ordering::SeqCst;
    let payload = body();
    let half = &payload[..payload.len() / 2];
    for h2 in [false, true] {
        let what = if h2 { "h2c" } else { "http/1.1" };
        let (primary, primary_conns) = silent_upstream().await;
        let (fallback, fallback_conns) = silent_upstream().await;
        let (pubkey, sk) = test_keypair(1);
        let gw = Gateway::builder(
            unused_nats_port(),
            &format!("127.0.0.1:{primary}"),
            &b64(&pubkey),
        )
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &format!("127.0.0.1:{fallback}"))
        .config_line("read_timeout_secs = 1")
        .config_line("circuit_breaker_threshold = 1")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
        let vkey = billing_vkey(&sk, 260);
        let headers = [
            ("authorization", format!("Bearer {vkey}")),
            ("content-type", "application/json".to_owned()),
            ("x-beyond-model", "gpt-4o-mini".to_owned()),
            ("x-beyond-order", "openai,openrouter".to_owned()),
            ("content-length", payload.len().to_string()),
        ];
        let (status, text) = if h2 {
            h2_stalled_upload(gw.port, &headers, half).await
        } else {
            let mut head = "POST /auto/chat/completions HTTP/1.1\r\nhost: gw\r\n".to_owned();
            for (k, v) in &headers {
                head.push_str(&format!("{k}: {v}\r\n"));
            }
            head.push_str("\r\n");
            let text = h1_stalled_upload(gw.port, &head, half).await;
            let status = text
                .strip_prefix("HTTP/1.1 ")
                .and_then(|t| t.get(..3))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            (status, text)
        };
        assert_eq!(status, 408, "{what}: {text}\nlog:\n{}", gw.log());
        assert!(text.contains("request body timed out"), "{what}: {text}");
        let row = usage_row_of(&gw).await;
        assert_eq!(row["usage_estimated"], false, "{what}: {row}");
        assert_eq!(row["input_tokens"].as_u64(), Some(0), "{what}: {row}");
        assert_eq!(row["outcome"], "client_cancelled", "{what}: {row}");
        // With a threshold of 1, a failure charged to the primary would have opened its breaker
        // and sent this request to the fallback.
        let resp = reqwest::Client::new()
            .post(format!("{}/auto/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {vkey}"))
            .header("content-type", "application/json")
            .header("x-beyond-model", "gpt-4o-mini")
            .header("x-beyond-order", "openai,openrouter")
            .body(payload.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 504, "{what}");
        assert_eq!(
            (primary_conns.load(SeqCst), fallback_conns.load(SeqCst)),
            (2, 0),
            "{what}: log:\n{}",
            gw.log()
        );
        let metrics = gw.metrics().await;
        assert_eq!(
            parse_metric(&metrics, "ai_candidate_failovers_total", ""),
            0.0,
            "{what}"
        );
        for reason in ["revoked", "unfunded", "key_named_403"] {
            assert_eq!(
                parse_metric(&metrics, "ai_key_auth_failures_total", reason),
                0.0,
                "{what}: {reason}"
            );
        }
    }
}

/// A managed chunked upload that stalls mid-stream on a provider route (which streams the body:
/// no catalog walk reads it first) times out as the client's own: a 408, its breaker charged
/// nothing (the next request still reaches the provider, with a threshold of 1), and not billed.
/// claim: REL-1, SEC-16, BIL-3
/// defect: D260
#[tokio::test]
async fn a_client_stalled_mid_chunked_upload_times_out_as_its_own() {
    use std::sync::atomic::Ordering::SeqCst;
    let (upstream, conns) = silent_upstream().await;
    let (pubkey, sk) = test_keypair(1);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{upstream}"),
        &b64(&pubkey),
    )
    .config_line("read_timeout_secs = 1")
    .config_line("circuit_breaker_threshold = 1")
    .config_line("circuit_breaker_window_secs = 60")
    .config_line("circuit_breaker_reset_secs = 60")
    .start()
    .await;
    let vkey = billing_vkey(&sk, 261);
    let payload = body();
    let half = &payload[..payload.len() / 2];
    let head = format!(
        "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer {vkey}\r\n\
         content-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n",
    );
    let text = h1_stalled_upload(gw.port, &head, &format!("{:x}\r\n{half}\r\n", half.len())).await;
    assert!(
        text.starts_with("HTTP/1.1 408"),
        "{text}\nlog:\n{}",
        gw.log()
    );
    let row = usage_row_of(&gw).await;
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["input_tokens"].as_u64(), Some(0), "{row}");
    assert_eq!(row["outcome"], "client_cancelled", "{row}");
    let resp = reqwest::Client::new()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(payload.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        504,
        "the breaker opened on a client's stall"
    );
    assert_eq!(conns.load(SeqCst), 2, "log:\n{}", gw.log());
}
