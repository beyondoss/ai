//! Reliability, verify phase 0: the circuit breaker and the TTFT ranker — what counts as healthy,
//! whether a brownout opens the breaker, and whether a half-open probe can wedge a provider.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const MODEL: &str = "gpt-4o-mini";

fn body() -> String {
    format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#)
}

/// A `bai_v1` key for `tenant_id`. Distinct tenants are distinct callers: no shared session pin.
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

async fn post_auto(client: &reqwest::Client, url: &str, key: &str) -> reqwest::Response {
    client
        .post(format!("{url}/auto/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .body(body())
        .send()
        .await
        .unwrap()
}

/// BYO on the bare `/v1` path: the dialect-default provider (openai), breaker gates all traffic.
async fn post_byo(client: &reqwest::Client, url: &str) -> u16 {
    client
        .post(format!("{url}/v1/chat/completions"))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0)
}

fn provider_of(resp: &reqwest::Response) -> String {
    resp.headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string()
}

/// A row whose primary's pool key is revoked (fast 401) and whose fallback works (a little slower)
/// must keep serving. Thirty callers (distinct tenants, so no session pin hides the row's rank) should
/// mostly succeed on the fallback.
/// claim: REL-4
/// defect: D10
#[tokio::test]
#[ignore = "D10 reproduced: fast 401 primary ranks healthy and fastest; 1/30 callers succeed"]
async fn a_401_candidate_does_not_black_hole_the_row() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(401)).await;
    let fallback = MockUpstream::start(Mode::Slow(30)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let client = test_client();
    let mut ok = 0;
    for i in 0..30u64 {
        let resp = post_auto(&client, &gw.url(), &vkey(&sk, 1000 + i)).await;
        if resp.status().as_u16() == 200 {
            ok += 1;
        }
    }
    assert!(
        ok >= 20,
        "only {ok}/30 requests succeeded; primary hits {} fallback hits {}",
        primary.hits(),
        fallback.hits()
    );
}

/// A 50% 5xx brownout, interleaved with successes, must open the breaker.
/// claim: REL-6
/// defect: D37
#[tokio::test]
async fn a_half_failing_provider_opens_the_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let mock = ReplyUpstream::start(|n, _| {
        if n % 2 == 0 {
            Reply::json(500, r#"{"error":{"message":"mock"}}"#)
        } else {
            Reply::ok()
        }
    })
    .await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 4")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let client = test_client();
    let mut statuses = Vec::new();
    for _ in 0..40 {
        statuses.push(post_byo(&client, &gw.url()).await);
    }
    assert!(
        statuses.contains(&503),
        "a 50% 5xx rate over 40 requests never opened the breaker: {statuses:?}"
    );
}

/// Control for D37: a solid 5xx run does open it (the breaker is wired; only the brownout fails).
/// claim: REL-6
/// defect: D37
#[tokio::test]
async fn a_solid_5xx_run_opens_the_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Status(500)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 4")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let client = test_client();
    let mut statuses = Vec::new();
    for _ in 0..10 {
        statuses.push(post_byo(&client, &gw.url()).await);
    }
    assert!(statuses.contains(&503), "{statuses:?}");
}

/// The half-open probe stalls (headers, one chunk, then silence). Other callers must not see 503
/// for as long as that one stream lives (up to `read_timeout_secs`, 600s by default).
/// claim: REL-6
/// defect: D17
#[tokio::test]
async fn a_stalled_half_open_probe_does_not_wedge_the_provider() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    // 0: 500 opens the breaker. 1: the probe, which stalls. Everything after: healthy.
    let mock = ReplyUpstream::start(|n, _| match n {
        0 => Reply::json(500, r#"{"error":{"message":"mock"}}"#),
        1 => Reply::Stall {
            status: 200,
            content_type: "text/event-stream",
            first: bytes::Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            ),
        },
        _ => Reply::ok(),
    })
    .await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 1")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 1")
        .start()
        .await;
    let client = test_client();
    assert_eq!(post_byo(&client, &gw.url()).await, 500);
    // Past the reset timeout (whole seconds, so wait two), the next request is the probe.
    tokio::time::sleep(Duration::from_millis(2100)).await;
    let probe = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(probe.status().as_u16(), 200, "the probe got its head");
    assert_eq!(mock.hits(), 2);
    // Hold the probe open (unread) while other callers try.
    let start = Instant::now();
    let mut seen = Vec::new();
    while start.elapsed() < Duration::from_secs(8) {
        let s = post_byo(&client, &gw.url()).await;
        seen.push(s);
        if s == 200 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    drop(probe);
    assert_eq!(
        seen.last(),
        Some(&200),
        "with the probe stalled, no request succeeded in 8s ({} tries, upstream hits {}): {seen:?}",
        seen.len(),
        mock.hits()
    );
}

/// A 200 whose body is an error (OpenRouter's error-in-200) is not a healthy answer: it must not
/// pin the caller, and the caller's next requests should reach the working fallback.
/// claim: REL-8
/// defect: D41
#[tokio::test]
#[ignore = "D41 reproduced: 200+error body pins the caller to the failing primary; 0/10 reach fallback"]
async fn a_200_with_an_error_body_is_not_pinned() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Raw(
        200,
        "application/json",
        r#"{"error":{"message":"upstream overloaded","code":502}}"#,
    ))
    .await;
    let fallback = MockUpstream::start(Mode::Slow(30)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let client = test_client();
    let key = vkey(&sk, 5);
    let mut served = Vec::new();
    for _ in 0..10 {
        served.push(provider_of(&post_auto(&client, &gw.url(), &key).await));
    }
    let on_fallback = served.iter().filter(|p| *p == "openrouter").count();
    let pinned = gw.metric("ai_session_pinned_total", "").await;
    assert!(
        on_fallback >= 7,
        "only {on_fallback}/10 reached the working fallback (pinned decisions: {pinned}): {served:?}"
    );
}

/// The SSE twin: a 200 stream whose first event is an error.
/// claim: REL-8
/// defect: D41
#[tokio::test]
#[ignore = "D41 reproduced: SSE error-first 200 pins the caller to the failing primary"]
async fn a_200_stream_with_an_error_first_event_is_not_pinned() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::OpenAiErrorSse).await;
    let fallback = MockUpstream::start(Mode::Slow(30)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let client = test_client();
    let key = vkey(&sk, 6);
    let mut served = Vec::new();
    for _ in 0..10 {
        served.push(provider_of(&post_auto(&client, &gw.url(), &key).await));
    }
    let on_fallback = served.iter().filter(|p| *p == "openrouter").count();
    assert!(
        on_fallback >= 7,
        "only {on_fallback}/10 reached the working fallback: {served:?}"
    );
}

/// Write `head`, then `chunks` 1 MiB chunks of a chunked body, then (if `finish`) the terminator.
/// Returns once the gateway answers or closes. A client that sends neither the terminator nor more
/// bytes is a stalled upload.
async fn chunked_upload(port: u16, head: &str, chunks: usize, finish: bool) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    s.write_all(head.as_bytes()).await.unwrap();
    let frame = {
        let mut f = format!("{:x}\r\n", 1 << 20).into_bytes();
        f.extend_from_slice(&vec![b' '; 1 << 20]);
        f.extend_from_slice(b"\r\n");
        f
    };
    for _ in 0..chunks {
        if s.write_all(&frame).await.is_err() {
            break;
        }
    }
    if finish {
        let _ = s.write_all(b"0\r\n\r\n").await;
    }
    let mut buf = [0u8; 1024];
    let _ = tokio::time::timeout(Duration::from_secs(10), s.read(&mut buf)).await;
}

/// Oversized and stalled uploads are the client's fault. Before the fix each one counted as a
/// provider failure, so any caller (no key needed: BYO counts too) could open the shared breaker
/// and 503 every tenant.
/// claim: SEC-16
/// defect: D32
#[tokio::test]
async fn client_upload_failures_do_not_open_the_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 2")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .config_line("read_timeout_secs = 1")
        .start()
        .await;
    let head = "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
                authorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\n\
                transfer-encoding: chunked\r\n\r\n";
    // Past the 100 MiB cap, with no Content-Length for the up-front check to see.
    for _ in 0..3 {
        chunked_upload(gw.port, head, 101, true).await;
    }
    wait_for_metric(&gw, "ai_rejections_total", "body_too_large", 3.0).await;
    // A body that starts and then stops: the upstream gives up waiting for it.
    let stalls: Vec<_> = (0..3)
        .map(|_| tokio::spawn(chunked_upload(gw.port, head, 1, false)))
        .collect();
    for s in stalls {
        s.await.unwrap();
    }
    let metrics = gw.metrics().await;
    assert!(
        !metrics.contains(r#"ai_rejections_total{reason="circuit_open"} "#)
            || metrics.contains(r#"ai_rejections_total{reason="circuit_open"} 0"#),
        "{metrics}"
    );
    let status = post_byo(&test_client(), &gw.url()).await;
    assert_eq!(status, 200, "the breaker opened on client faults");
}
