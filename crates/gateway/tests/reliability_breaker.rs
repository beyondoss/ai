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

/// The half-open probe is a request whose client stalls mid-upload, so the attempt ends with no
/// provider outcome at all. That says nothing about the provider: the probe permit goes back, and
/// the breaker stays half-open. It must not close (which would let every caller flood a provider
/// that is still broken), so the next callers get exactly one new probe between them.
/// claim: REL-6, SEC-16
/// defect: D86
#[tokio::test]
async fn a_probe_with_no_provider_outcome_leaves_the_breaker_half_open() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    // 0: a 500 opens the breaker. 1: the stalled probe (never answered: its body never ends).
    // Everything after: still broken, and slow enough that concurrent callers overlap.
    let mock = ReplyUpstream::start(|n, _| match n {
        0 => Reply::json(500, r#"{"error":{"message":"mock"}}"#),
        _ => Reply::Delayed(
            Duration::from_millis(800),
            Box::new(Reply::json(500, r#"{"error":{"message":"mock"}}"#)),
        ),
    })
    .await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 1")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 1")
        .config_line("read_timeout_secs = 1")
        .start()
        .await;
    let client = test_client();
    assert_eq!(post_byo(&client, &gw.url()).await, 500);
    tokio::time::sleep(Duration::from_millis(2100)).await;
    let head = "POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
                authorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\n\
                transfer-encoding: chunked\r\n\r\n";
    chunked_upload(gw.port, head, 1, false).await;
    assert_eq!(mock.hits(), 2, "the stalled upload was the probe");
    let callers: Vec<_> = (0..3)
        .map(|_| {
            let (client, url) = (client.clone(), gw.url());
            tokio::spawn(async move { post_byo(&client, &url).await })
        })
        .collect();
    let mut statuses = Vec::new();
    for c in callers {
        statuses.push(c.await.unwrap());
    }
    statuses.sort_unstable();
    assert_eq!(
        (statuses.as_slice(), mock.hits()),
        (&[500, 503, 503][..], 3),
        "a closed breaker let every caller through to the broken provider"
    );
}

/// A pool key failing on every candidate relays the last candidate's own 401, and never opens a
/// breaker: the providers answered, so the next request still reaches them. The one whose every
/// key drew a 401 is cooled, so that request skips it while another candidate remains (D180).
/// claim: REL-4
/// defect: D10
#[tokio::test]
async fn a_401_on_every_candidate_is_relayed_and_opens_no_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(401)).await;
    let fallback = MockUpstream::start(Mode::Status(403)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .config_line("circuit_breaker_threshold = 1")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let client = test_client();
    let first = post_auto(&client, &gw.url(), &vkey(&sk, 77)).await;
    let status = first.status().as_u16();
    assert!(
        status == 401 || status == 403,
        "the last candidate's own status is relayed, got {status}"
    );
    assert_eq!(
        (primary.hits(), fallback.hits()),
        (1, 1),
        "the walk tried both candidates"
    );
    let second = post_auto(&client, &gw.url(), &vkey(&sk, 78)).await;
    assert_ne!(
        second.status().as_u16(),
        503,
        "a key failure opened a breaker"
    );
    assert_eq!(
        (primary.hits(), fallback.hits()),
        (1, 2),
        "the cooled 401 provider is skipped, the other still reached"
    );
}

/// A complete OpenAI chat answer.
const OK_JSON: &str = r#"{"id":"chatcmpl-ok","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

/// An error-in-200 whose first bytes cannot tell yet (`{"err`, then the rest a moment later) is
/// still judged on the bytes that decide it: it must not pin the caller on the strength of a prefix
/// that had not said anything.
/// claim: REL-8, R4
#[tokio::test]
async fn a_200_error_body_split_before_its_first_key_is_not_pinned() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = ScriptedUpstream::start(|_, _| {
        vec![
            Step::Write(http_head(200, "application/json", None)),
            Step::Write(b"{\"err".to_vec()),
            Step::Sleep(Duration::from_millis(150)),
            Step::Write(br#"or":{"message":"upstream overloaded","code":502}}"#.to_vec()),
        ]
    })
    .await;
    let fallback = MockUpstream::start(Mode::Slow(30)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let client = test_client();
    let key = vkey(&sk, 8);
    let mut served = Vec::new();
    for _ in 0..10 {
        let resp = post_auto(&client, &gw.url(), &key).await;
        served.push(provider_of(&resp));
        let _ = resp.bytes().await;
    }
    let on_fallback = served.iter().filter(|p| *p == "openrouter").count();
    assert!(
        on_fallback >= 7,
        "only {on_fallback}/10 reached the working fallback: {served:?}"
    );
}

/// A 2xx whose first KiB says nothing either way (OpenRouter pads a slow non-stream answer with
/// whitespace) is settled as an answer once that KiB is in — while the body is still arriving — so
/// the caller's next request is already pinned to it.
/// claim: R4
#[tokio::test]
async fn an_undecidable_first_kib_settles_as_an_answer_before_the_body_ends() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = ScriptedUpstream::start(|_, _| {
        vec![
            Step::Write(http_head(200, "application/json", None)),
            Step::Write(vec![b' '; 1100]),
            Step::Sleep(Duration::from_secs(4)),
            Step::Write(OK_JSON.as_bytes().to_vec()),
        ]
    })
    .await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let client = test_client();
    let key = vkey(&sk, 9);
    let (c, url, k) = (client.clone(), gw.url(), key.clone());
    let first = tokio::spawn(async move { post_auto(&c, &url, &k).await.bytes().await });
    // The padding is in; the answer is not.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (c, url, k) = (client.clone(), gw.url(), key.clone());
    let second = tokio::spawn(async move { post_auto(&c, &url, &k).await.bytes().await });
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    assert!(
        gw.metric("ai_session_pinned_total", "").await >= 1.0,
        "the second request was not routed by a pin taken on the first's padded prefix"
    );
}

/// A candidate that was answering and then starts failing is demoted after its first failure:
/// the next caller goes to the fallback first instead of paying for the failing attempt again.
/// claim: R7, R1
#[tokio::test]
async fn a_candidate_that_starts_failing_is_demoted_after_one_failure() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = ReplyUpstream::start(|n, _| {
        if n < 2 {
            Reply::json(200, OK_JSON)
        } else {
            Reply::json(500, r#"{"error":{"message":"mock"}}"#)
        }
    })
    .await;
    // Slower than the primary's answers, so only the failure — not raw speed — can put it first.
    let fallback = MockUpstream::start(Mode::Slow(150)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .config_line("circuit_breaker_threshold = 100")
        .start()
        .await;
    let client = test_client();
    // Distinct tenants: no session pin, so only the ranker orders the walk.
    for tenant in 100..105 {
        let resp = post_auto(&client, &gw.url(), &vkey(&sk, tenant)).await;
        assert_eq!(resp.status().as_u16(), 200, "tenant {tenant}");
        let _ = resp.bytes().await;
    }
    assert_eq!(
        primary.hits(),
        3,
        "after its first 5xx the primary kept being tried first"
    );
}

/// A connect failure on a provider route counts against that provider's breaker: no body byte
/// moved, so the failure is the provider's, not a stalled client upload.
/// claim: R6, REL-6
#[tokio::test]
async fn connect_failures_on_a_provider_route_open_the_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 2")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let client = test_client();
    let mut statuses = Vec::new();
    for _ in 0..4 {
        statuses.push(post_byo(&client, &gw.url()).await);
    }
    assert!(statuses.contains(&503), "{statuses:?}");
    // The same with a body still streaming in when the connect fails: no byte of it reached the
    // provider, so it is still the provider's failure, not a stalled upload.
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 2")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let head = "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\n\
                authorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\n\
                transfer-encoding: chunked\r\n\r\n";
    for _ in 0..2 {
        chunked_upload(gw.port, head, 1, true).await;
    }
    assert_eq!(
        post_byo(&client, &gw.url()).await,
        503,
        "two connect failures during uploads did not open the breaker"
    );
}

/// A provider route's connect failure is retried exactly `MAX_CONNECT_RETRIES` (2) times, then
/// answered as a 502 that names the connect failure — never "after receiving the request", since
/// no byte reached the provider — with no `Retry-After`. The breaker's 503 carries one.
/// claim: REL-1, REL-19
#[tokio::test]
async fn a_provider_route_connect_failure_is_retried_twice_then_named() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 2")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let client = test_client();
    let send = || async {
        client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", "Bearer sk-byo-test")
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap()
    };
    let first = send().await;
    assert_eq!(first.status().as_u16(), 502);
    assert!(first.headers().get("retry-after").is_none());
    let text = first.text().await.unwrap();
    assert!(text.contains("could not connect to the provider"), "{text}");
    assert_eq!(
        gw.metric("ai_connect_retries_total", "openai").await,
        2.0,
        "one request, two same-provider retries"
    );
    let mut open = None;
    for _ in 0..3 {
        let resp = send().await;
        if resp.status().as_u16() == 503 {
            open = Some(resp);
            break;
        }
    }
    let open = open.expect("the breaker never opened");
    assert!(open.headers().get("retry-after").is_some());
}

/// A dead single-address candidate is left after one connect failure: there is no other address
/// of it to try, so the walk moves straight to the next candidate.
/// claim: R1
#[tokio::test]
async fn a_dead_single_address_candidate_is_tried_once() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let resp = post_auto(&test_client(), &gw.url(), &vkey(&sk, 120)).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(fallback.hits(), 1);
    assert_eq!(gw.metric("ai_connect_retries_total", "openai").await, 1.0);
}
