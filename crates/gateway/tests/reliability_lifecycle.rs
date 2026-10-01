//! Reliability, verify phase 0: readiness, shutdown, stalls, slow readers, body memory and DNS.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

const STREAM_BODY: &str =
    r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

/// NATS is down at boot, so the allowance set never seeds and every managed request is refused
/// with "allowance unavailable". Readiness must say so, or the load balancer keeps sending traffic
/// to a pod that refuses all of it.
/// claim: REL-5
/// defect: D11
#[tokio::test]
async fn readyz_is_not_ready_while_allowance_is_unseeded() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(closed_port(), &mock.authority(), &b64(&pubkey))
        .skip_allowance_ready()
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, 1)))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("allowance unavailable"),
        "precondition: managed traffic is refused ({status} {text})"
    );
    let (ready, body) = gw.admin_get("/readyz").await;
    assert_ne!(
        ready, 200,
        "readyz {ready} {body} while managed requests get {status} allowance unavailable"
    );
}

/// An idle gateway exits promptly on SIGTERM: there is nothing to drain. Default config (600s
/// grace).
/// claim: REL-16, BIL-21
/// defect: D38
#[tokio::test]
async fn an_idle_gateway_exits_promptly_on_sigterm() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let mut gw = Gateway::start(unused_nats_port(), &mock.authority(), &b64(&pubkey)).await;
    gw.sigterm();
    let exited = gw.wait_exit(Duration::from_secs(15)).await;
    assert!(
        exited.is_some_and(|t| t < Duration::from_secs(10)),
        "idle gateway exit after SIGTERM: {exited:?} (None = still running at 15s)"
    );
}

/// The same with a short configured grace: shutdown time tracks the grace, even when idle.
/// claim: REL-16, BIL-21
/// defect: D38
#[tokio::test]
async fn an_idle_gateway_with_a_short_grace_exits_before_the_grace() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let mut gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("shutdown_grace_period_secs = 3")
        .config_line("shutdown_runtime_timeout_secs = 1")
        .start()
        .await;
    gw.sigterm();
    let exited = gw.wait_exit(Duration::from_secs(15)).await;
    assert!(
        exited.is_some_and(|t| t < Duration::from_secs(2)),
        "idle gateway (3s grace) exit after SIGTERM: {exited:?}"
    );
}

/// `read_timeout_secs` is honored: a header stall ends at the configured bound.
/// claim: REL-7
/// defect: D36
#[tokio::test]
async fn a_header_stall_ends_at_the_configured_read_timeout() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = ReplyUpstream::start(|_, _| Reply::HeaderStall).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("read_timeout_secs = 2")
        .start()
        .await;
    let start = Instant::now();
    let status = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(STREAM_BODY)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0);
    let took = start.elapsed();
    assert!(
        status >= 500 && took < Duration::from_secs(6),
        "status {status} after {took:?}"
    );
}

/// With default config a streaming request whose upstream never sends a response head must still
/// fail well under the 600s `read_timeout_secs`: a stall bound below it has to exist. A streaming
/// provider sends its head at once, so 15s is a generous first-byte bound.
/// claim: REL-7
/// defect: D36
#[tokio::test]
#[ignore = "D36 reproduced: default config, streaming header stall still open at 15s (only read_timeout=600s bounds it)"]
async fn a_streaming_header_stall_fails_well_under_read_timeout_by_default() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = ReplyUpstream::start(|_, _| Reply::HeaderStall).await;
    let gw = Gateway::start(unused_nats_port(), &mock.authority(), &b64(&pubkey)).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap();
    let start = Instant::now();
    let result = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(STREAM_BODY)
        .send()
        .await;
    let took = start.elapsed();
    assert!(
        result.is_ok(),
        "no answer from the gateway after {took:?}: {:?}",
        result.err()
    );
}

/// A client that sends a request for a large stream and then never reads must be released (its
/// in-flight slot, upstream connection, gauge) within a bound. 20s is generous for "stopped reading
/// entirely".
/// claim: REL-10
/// defect: D42
#[tokio::test]
#[ignore = "D42 reproduced: a client that never reads holds its in-flight slot past 20s"]
async fn a_client_that_stops_reading_is_released() {
    let (pubkey, _sk) = test_keypair(1);
    let big = {
        let event = "data: {\"choices\":[{\"delta\":{\"content\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}}]}\n\n";
        bytes::Bytes::from(event.repeat((64 << 20) / event.len()))
    };
    let mock = ReplyUpstream::start(move |_, _| Reply::Full {
        status: 200,
        content_type: "text/event-stream",
        body: big.clone(),
    })
    .await;
    let gw = Gateway::start(unused_nats_port(), &mock.authority(), &b64(&pubkey)).await;
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let req = format!(
        "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
         authorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\n\
         content-length: {}\r\n\r\n{STREAM_BODY}",
        STREAM_BODY.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    // Read nothing, ever. Wait for the request to be in flight, then for it to be released.
    wait_for_metric(&gw, "ai_requests_in_flight", "", 1.0).await;
    let start = Instant::now();
    let mut in_flight = 1.0;
    while start.elapsed() < Duration::from_secs(20) {
        in_flight = gw.metric("ai_requests_in_flight", "").await;
        if in_flight == 0.0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    drop(s);
    assert_eq!(
        in_flight,
        0.0,
        "a non-reading client still holds its slot after {:?}",
        start.elapsed()
    );
}

/// Concurrent large uploads must not grow the gateway's memory without bound. Eight managed
/// catalog requests with ~90 MiB bodies, held while the upstream stalls, may not cost more than
/// 512 MiB of gateway RSS between them (the bodies alone are 720 MiB).
/// claim: SEC-19
/// defect: D35
#[tokio::test]
#[ignore = "D35 reproduced: 8 concurrent 90 MiB uploads grow gateway RSS by ~1.4 GiB"]
async fn concurrent_large_uploads_have_bounded_memory() {
    const N: u64 = 8;
    const BODY_MIB: usize = 90;
    let (pubkey, sk) = test_keypair(1);
    let mock = ReplyUpstream::start(|_, _| Reply::HeaderStall).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;
    let base = gw.rss_kib();
    let body = std::sync::Arc::new({
        let mut b = br#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":""#.to_vec();
        b.extend(std::iter::repeat_n(b'x', BODY_MIB << 20));
        b.extend_from_slice(br#""}]}"#);
        b
    });
    let mut conns = Vec::new();
    for i in 0..N {
        let (body, port, key) = (body.clone(), gw.port, vkey(&sk, 100 + i));
        conns.push(tokio::spawn(async move {
            let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .unwrap();
            let head = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
                 authorization: Bearer {key}\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\n\r\n",
                body.len()
            );
            let _ = s.write_all(head.as_bytes()).await;
            let _ = s.write_all(&body).await;
            // Keep the connection open while the test samples memory.
            let mut buf = [0u8; 1024];
            let _ = tokio::time::timeout(Duration::from_secs(20), s.read(&mut buf)).await;
        }));
    }
    let start = Instant::now();
    let mut peak = base;
    while start.elapsed() < Duration::from_secs(12) {
        peak = peak.max(gw.rss_kib());
        if mock.hits() >= N as usize && start.elapsed() > Duration::from_secs(2) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    peak = peak.max(gw.rss_kib());
    let grew_mib = (peak.saturating_sub(base)) / 1024;
    for c in conns {
        c.abort();
    }
    assert!(
        grew_mib < 512,
        "gateway RSS grew {grew_mib} MiB for {N} concurrent {BODY_MIB} MiB uploads (upstream saw {})",
        mock.hits()
    );
}

/// A provider authority whose name resolves to more than one address must be reachable when the
/// first one is dead. `localhost` resolves to `::1` first here and the mock listens only on
/// `127.0.0.1`.
/// claim: REL-20
/// defect: D40
#[tokio::test]
async fn dns_tries_the_next_address_when_the_first_is_dead() {
    let addrs: Vec<_> = tokio::net::lookup_host("localhost:80")
        .await
        .map(|a| a.collect())
        .unwrap_or_default();
    let v6_first = addrs.first().is_some_and(|a| a.is_ipv6());
    if !(v6_first && addrs.iter().any(|a| a.is_ipv4())) {
        eprintln!("SKIP: localhost does not resolve ::1 before 127.0.0.1 here ({addrs:?})");
        return;
    }
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .provider_authority("openai", &format!("localhost:{}", mock.port))
        .start()
        .await;
    let status = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0);
    assert_eq!(
        status,
        200,
        "localhost -> {addrs:?}; upstream hits {}",
        mock.hits()
    );
}

/// A panic inside a proxy phase skips `logging`, which is where a request gives back what it holds.
/// The request context's drop must give it back instead: the in-flight and SSE gauges return to 0
/// and the tenant's only concurrency slot is free for its next request. `AI_FAULT_PANIC` (debug
/// builds only) panics the first request to reach `response_body_filter`, mid-stream.
/// claim: REL-17
/// defect: D39
#[tokio::test]
async fn a_panic_in_a_proxy_phase_releases_what_the_request_held() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Sse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .tenant_max_in_flight(1)
        .env("AI_FAULT_PANIC", "response_body_filter")
        .start()
        .await;
    let key = vkey(&sk, 39);
    let send = || {
        test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(STREAM_BODY)
            .send()
    };
    // The panicking request: the connection dies mid-response, whatever the client makes of it.
    let first = match send().await {
        Ok(r) => r.text().await.map(|_| ()).map_err(|e| e.to_string()),
        Err(e) => Err(e.to_string()),
    };
    assert!(
        gw.log().contains("AI_FAULT_PANIC"),
        "the fault did not fire (first request: {first:?}); log:\n{}",
        gw.log()
    );
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(5)
        && (gw.metric("ai_requests_in_flight", "").await != 0.0
            || gw.metric("ai_active_streams", "").await != 0.0)
    {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(gw.metric("ai_requests_in_flight", "").await, 0.0);
    assert_eq!(gw.metric("ai_active_streams", "").await, 0.0);
    // The tenant's single slot came back: its next request is served, not 429.
    let second = send().await.unwrap();
    assert_eq!(second.status().as_u16(), 200);
}

/// The drain waits for in-flight work: a request already running when SIGTERM lands finishes with
/// its answer and its billing row, and the process exits right after it, not at the grace's end.
/// claim: REL-16, BIL-21
/// defect: D38
#[tokio::test]
async fn sigterm_drains_an_in_flight_request_then_exits() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Slow(1500)).await;
    let mut gw = Gateway::start(unused_nats_port(), &mock.authority(), &b64(&pubkey)).await;
    let (url, key) = (gw.url(), vkey(&sk, 38));
    let held = tokio::spawn(async move {
        test_client()
            .post(format!("{url}/openai/v1/chat/completions"))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#)
            .send()
            .await
            .map(|r| r.status().as_u16())
    });
    wait_for_metric(&gw, "ai_requests_in_flight", "", 1.0).await;
    gw.sigterm();
    let exited = gw.wait_exit(Duration::from_secs(15)).await;
    let status = held.await.unwrap();
    assert_eq!(status.ok(), Some(200), "the in-flight request was cut");
    assert!(
        exited.is_some_and(|t| t < Duration::from_secs(5)),
        "exit after the drain: {exited:?}"
    );
    assert!(
        gw.log().contains("\"target\":\"ai.usage\""),
        "the drained request's billing row was not written"
    );
}

/// The catalog walk does the same: a candidate whose first address is dead is tried on its next
/// address before the walk gives up on it.
/// claim: REL-20
/// defect: D40
#[tokio::test]
async fn a_catalog_candidate_is_tried_on_its_next_address() {
    let addrs: Vec<_> = tokio::net::lookup_host("localhost:80")
        .await
        .map(|a| a.collect())
        .unwrap_or_default();
    let v6_first = addrs.first().is_some_and(|a| a.is_ipv6());
    if !(v6_first && addrs.iter().any(|a| a.is_ipv4())) {
        eprintln!("SKIP: localhost does not resolve ::1 before 127.0.0.1 here ({addrs:?})");
        return;
    }
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openai", &format!("localhost:{}", mock.port))
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, 40)))
        .header("content-type", "application/json")
        .header("x-beyond-model", "gpt-4o-mini")
        .header("x-beyond-order", "openai,openrouter")
        .body(r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(
        (
            resp.status().as_u16(),
            resp.headers()
                .get("x-beyond-provider")
                .and_then(|v| v.to_str().ok())
                .map(String::from)
        ),
        (200, Some("openai".to_string())),
        "upstream hits {}",
        mock.hits()
    );
}
