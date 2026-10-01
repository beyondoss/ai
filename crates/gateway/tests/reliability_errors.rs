//! Reliability, verify phase 0: errors the gateway makes itself — their status, JSON body,
//! `x-beyond-request-id`, and `Retry-After`.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MODEL: &str = "gpt-4o-mini";

fn body() -> String {
    format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#)
}

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

async fn post(client: &reqwest::Client, url: &str, path: &str, key: &str) -> reqwest::Response {
    client
        .post(format!("{url}{path}"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .body(body())
        .send()
        .await
        .unwrap()
}

/// Everything wrong with a gateway-made error response, empty when it is right.
async fn error_shape_problems(resp: reqwest::Response, want: u16) -> Vec<String> {
    let mut problems = Vec::new();
    let status = resp.status().as_u16();
    if status != want {
        problems.push(format!("status {status}, want {want}"));
    }
    if resp.headers().get("x-beyond-request-id").is_none() {
        problems.push("no x-beyond-request-id".into());
    }
    let text = resp.text().await.unwrap_or_default();
    let json: Option<serde_json::Value> = serde_json::from_str(&text).ok();
    if !json
        .as_ref()
        .is_some_and(|v| v["error"]["message"].is_string())
    {
        problems.push(format!("body is not a JSON error envelope: {text:?}"));
    }
    problems
}

/// Every candidate refuses the connection: a 502 with a JSON error and the request id.
/// claim: REL-2, CAT-11
/// defect: D16
#[tokio::test]
#[ignore = "D16 reproduced: all candidates connect-failing gives an empty 502 with no request id"]
async fn all_candidates_failing_to_connect_is_a_json_502() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;
    let resp = post(
        &test_client(),
        &gw.url(),
        "/auto/chat/completions",
        &vkey(&sk, 1),
    )
    .await;
    let problems = error_shape_problems(resp, 502).await;
    assert!(problems.is_empty(), "{problems:?}");
}

/// Every candidate's breaker is open: the walk has nowhere to go, which is a 503, not a bare 500.
/// claim: REL-2, CAT-11
/// defect: D16
#[tokio::test]
#[ignore = "D16 reproduced: all breakers open on a catalog walk gives a bare 500"]
async fn all_breakers_open_on_a_catalog_walk_is_a_json_503() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Status(500)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .config_line("circuit_breaker_threshold = 1")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let client = test_client();
    let key = vkey(&sk, 1);
    // One walk fails on both candidates and opens both breakers (threshold 1).
    let first = post(&client, &gw.url(), "/auto/chat/completions", &key).await;
    assert_eq!(first.status().as_u16(), 500, "the relayed upstream 500");
    // Logging records the last candidate's failure just after the response; let it land.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let hits = (primary.hits(), fallback.hits());
    let resp = post(&client, &gw.url(), "/auto/chat/completions", &key).await;
    assert_eq!(
        (primary.hits(), fallback.hits()),
        hits,
        "both breakers open: no upstream attempt"
    );
    let problems = error_shape_problems(resp, 503).await;
    assert!(problems.is_empty(), "{problems:?}");
}

/// Write a chunked request with `total` body bytes, then read whatever comes back.
async fn chunked_upload(port: u16, path: &str, total: usize) -> RawResponse {
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let head = format!(
        "POST {path} HTTP/1.1\r\nhost: 127.0.0.1\r\nauthorization: Bearer sk-byo-test\r\n\
         content-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n"
    );
    s.write_all(head.as_bytes()).await.unwrap();
    const CHUNK: usize = 1 << 20;
    let payload = vec![b' '; CHUNK];
    let frame = {
        let mut f = format!("{CHUNK:x}\r\n").into_bytes();
        f.extend_from_slice(&payload);
        f.extend_from_slice(b"\r\n");
        f
    };
    let mut sent = 0;
    let mut write_failed = false;
    while sent < total {
        if s.write_all(&frame).await.is_err() {
            write_failed = true;
            break;
        }
        sent += CHUNK;
    }
    if !write_failed {
        let _ = s.write_all(b"0\r\n\r\n").await;
    }
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(10), s.read_to_end(&mut buf)).await;
    parse_raw_response(&buf)
}

/// A chunked body (no Content-Length, so the up-front check cannot see it) that grows past the
/// 100 MiB cap is a 413 with a JSON error, not a 500 or a reset.
/// claim: REL-2, CAT-11
/// defect: D16
#[tokio::test]
#[ignore = "D16 reproduced: chunked body over 100 MiB on /{provider} gives a bare 413 (no JSON, no request id)"]
async fn a_chunked_body_over_the_cap_is_a_json_413() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats_port, &mock.authority(), &b64(&pubkey)).await;
    let resp = chunked_upload(gw.port, "/openai/v1/chat/completions", 101 << 20).await;
    // The cap fired (not some other failure).
    wait_for_metric(&gw, "ai_rejections_total", "body_too_large", 1.0).await;
    let body = String::from_utf8_lossy(&resp.body).to_string();
    let json_ok = serde_json::from_str::<serde_json::Value>(body.trim())
        .is_ok_and(|v| v["error"]["message"].is_string());
    assert!(
        resp.status == 413 && json_ok && resp.header("x-beyond-request-id").is_some(),
        "status {} headers {:?} body {body:?}",
        resp.status,
        resp.headers
    );
}

/// Control for D16: a chunked over-cap body on the catalog path (read in full to pick a row) is
/// already a clean 413.
/// claim: REL-2, CAT-11
/// defect: D16
#[tokio::test]
async fn a_chunked_body_over_the_cap_on_the_catalog_path_is_a_413() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats_port, &mock.authority(), &b64(&pubkey)).await;
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\nauthorization: Bearer {}\r\n\
         content-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n",
        vkey(&sk, 1)
    );
    s.write_all(head.as_bytes()).await.unwrap();
    let frame = {
        let mut f = format!("{:x}\r\n", 1 << 20).into_bytes();
        f.extend_from_slice(&vec![b' '; 1 << 20]);
        f.extend_from_slice(b"\r\n");
        f
    };
    for _ in 0..101 {
        if s.write_all(&frame).await.is_err() {
            break;
        }
    }
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(10), s.read_to_end(&mut buf)).await;
    let resp = parse_raw_response(&buf);
    assert_eq!(resp.status, 413, "{resp:?}");
    assert!(resp.header("x-beyond-request-id").is_some());
}

/// Gateway-made 429s and 503s carry `Retry-After`, and a transient allowance outage is a retryable
/// 503 rather than a 402 SDKs treat as final. One gateway per case; every problem is reported.
/// claim: REL-19
/// defect: D43
#[tokio::test]
#[ignore = "D43 reproduced: gateway 429/503 carry no Retry-After; allowance unavailable is 402"]
async fn gateway_429_and_503_carry_retry_after() {
    let (pubkey, sk) = test_keypair(1);
    let client = test_client();
    let mut problems: Vec<String> = Vec::new();
    let retry_after = |resp: &reqwest::Response| resp.headers().get("retry-after").is_some();

    // (a) Per-credential rate limit.
    {
        let mock = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .rate_limit_rps(1)
            .byo_rate_limit_rps(0)
            .start()
            .await;
        let mut found = None;
        for _ in 0..20 {
            let resp = post(
                &client,
                &gw.url(),
                "/openai/v1/chat/completions",
                "sk-byo-rl",
            )
            .await;
            if resp.status().as_u16() == 429 {
                found = Some(resp);
                break;
            }
        }
        match found {
            None => problems.push("rate limit: never saw a 429".into()),
            Some(r) if !retry_after(&r) => problems.push("rate limit 429: no Retry-After".into()),
            Some(_) => {}
        }
    }

    // (b) Tenant concurrency cap.
    {
        let mock = MockUpstream::start(Mode::Slow(1500)).await;
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .tenant_max_in_flight(1)
            .start()
            .await;
        let key = vkey(&sk, 3);
        let (url, c, k) = (gw.url(), client.clone(), key.clone());
        let held = tokio::spawn(async move {
            post(&c, &url, "/openai/v1/chat/completions", &k)
                .await
                .status()
        });
        tokio::time::sleep(Duration::from_millis(300)).await;
        let resp = post(&client, &gw.url(), "/openai/v1/chat/completions", &key).await;
        if resp.status().as_u16() != 429 {
            problems.push(format!("tenant cap: got {}, not 429", resp.status()));
        } else if !retry_after(&resp) {
            problems.push("tenant concurrency 429: no Retry-After".into());
        }
        let _ = held.await;
    }

    // (c) Breaker open.
    {
        let mock = MockUpstream::start(Mode::Status(500)).await;
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .config_line("circuit_breaker_threshold = 1")
            .config_line("circuit_breaker_window_secs = 60")
            .config_line("circuit_breaker_reset_secs = 60")
            .start()
            .await;
        let _ = post(
            &client,
            &gw.url(),
            "/openai/v1/chat/completions",
            "sk-byo-cb",
        )
        .await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let resp = post(
            &client,
            &gw.url(),
            "/openai/v1/chat/completions",
            "sk-byo-cb",
        )
        .await;
        if resp.status().as_u16() != 503 {
            problems.push(format!("breaker: got {}, not 503", resp.status()));
        } else if !retry_after(&resp) {
            problems.push("breaker-open 503: no Retry-After".into());
        }
    }

    // (d) Allowance unavailable (NATS down at boot, never seeded).
    {
        let mock = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(closed_port(), &mock.authority(), &b64(&pubkey))
            .skip_allowance_ready()
            .start()
            .await;
        let resp = post(
            &client,
            &gw.url(),
            "/openai/v1/chat/completions",
            &vkey(&sk, 4),
        )
        .await;
        let status = resp.status().as_u16();
        let has_retry_after = retry_after(&resp);
        let text = resp.text().await.unwrap_or_default();
        if !text.contains("allowance unavailable") {
            problems.push(format!("allowance: unexpected answer {status} {text}"));
        } else if status != 503 {
            problems.push(format!(
                "allowance unavailable is {status}, not a retryable 503"
            ));
        } else if !has_retry_after {
            problems.push("allowance-unavailable 503: no Retry-After".into());
        }
    }

    assert!(problems.is_empty(), "{problems:#?}");
}
