//! Claim tests: provider steering headers, an exhausted key pool, and a circuit breaker that opens
//! on a broken provider and closes again once it recovers.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const MODEL: &str = "gpt-4o-mini";

fn chat() -> String {
    format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#)
}

async fn post(gw: &Gateway, key: &str, headers: &[(&str, &str)]) -> reqwest::Response {
    let mut req = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    req.body(chat()).send().await.unwrap()
}

fn provider_of(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

/// Every pool key on the provider throttles. Each key is tried once, and the client gets the 429
/// with the provider's `Retry-After`.
/// claim: R2
#[tokio::test]
async fn every_pool_key_throttled_is_a_429_with_retry_after() {
    let (pubkey, sk) = test_keypair(60);
    let mock = MockUpstream::start(Mode::Status(429)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-walk-a", "sk-walk-b"])
        .start()
        .await;
    let resp = post(&gw, &billing_vkey(&sk, 60), &[]).await;
    assert_eq!(resp.status().as_u16(), 429);
    assert_eq!(
        resp.headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("7"),
        "an exhausted pool keeps the provider's Retry-After"
    );
    assert_eq!(mock.hits(), 2, "each key is asked once");
    assert!(gw.metric("ai_key_walks_total", "").await >= 1.0);
}

/// `x-beyond-order` and `x-beyond-only` decide which provider serves, on every one of N calls, and
/// `x-beyond-provider` says so. Without either, catalog order holds.
/// claim: R3
#[tokio::test]
async fn order_and_only_steer_every_call_to_the_named_provider() {
    let (pubkey, sk) = test_keypair(61);
    let openai = MockUpstream::start(Mode::Json).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 61);
    for (header, value, want) in [
        ("x-beyond-order", "openrouter", "openrouter"),
        ("x-beyond-only", "openrouter", "openrouter"),
        ("x-beyond-order", "openai", "openai"),
        ("x-beyond-only", "openai", "openai"),
    ] {
        for i in 0..5 {
            let resp = post(&gw, &key, &[(header, value)]).await;
            assert_eq!(resp.status().as_u16(), 200, "{header}: {value} #{i}");
            assert_eq!(
                provider_of(&resp).as_deref(),
                Some(want),
                "{header}: {value} #{i}"
            );
        }
    }
    assert_eq!(openai.hits(), 10);
    assert_eq!(openrouter.hits(), 10);
}

/// A provider that 500s opens its breaker: further calls are 503 without reaching it. Once the
/// reset window passes and the provider is healthy again, the half-open probe closes the breaker
/// and traffic flows.
/// claim: R6
#[tokio::test]
async fn the_breaker_opens_on_a_broken_provider_and_recovers() {
    let (pubkey, _sk) = test_keypair(62);
    let broken = Arc::new(AtomicBool::new(true));
    let flag = broken.clone();
    let upstream = ReplyUpstream::start(move |_, _| {
        if flag.load(Ordering::SeqCst) {
            Reply::json(500, r#"{"error":{"message":"down"}}"#)
        } else {
            Reply::ok()
        }
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &upstream.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .circuit_breaker_threshold(3)
        .byo_rate_limit_rps(0)
        .start()
        .await;

    let byo = || post(&gw, "sk-byo-breaker", &[]);
    let mut opened = false;
    for _ in 0..20 {
        if byo().await.status().as_u16() == 503 {
            opened = true;
            break;
        }
    }
    assert!(opened, "the breaker never opened on a run of 500s");
    let hits = upstream.hits();
    assert_eq!(byo().await.status().as_u16(), 503, "open sheds load");
    assert_eq!(
        upstream.hits(),
        hits,
        "an open breaker does not reach the provider"
    );

    broken.store(false, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert_eq!(
        byo().await.status().as_u16(),
        200,
        "the half-open probe reaches the recovered provider"
    );
    for i in 0..3 {
        assert_eq!(byo().await.status().as_u16(), 200, "closed again, #{i}");
    }
}

/// The ranker's probe (every 8th request) measures a fallback the gateway can actually dispatch.
/// A Claude row deployed without Bedrock is anthropic → bedrock (unkeyed) → openrouter: the probe
/// must skip the unkeyed Bedrock arm and promote OpenRouter, or OpenRouter is never measured,
/// ranked or pinned and only ever serves as failover.
/// claim: R7
/// defect: D119
#[tokio::test]
async fn the_probe_skips_an_unkeyed_candidate() {
    let (pubkey, sk) = test_keypair(61);
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .start()
        .await;
    let body = r#"{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"hi"}]}"#;
    // seq 0..=16, each from a fresh tenant so no session pin decides the primary. Seq 8 and 16
    // are probe slots.
    let mut served = Vec::new();
    for t in 0..17 {
        let resp = test_client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header(
                "authorization",
                format!("Bearer {}", billing_vkey(&sk, 7000 + t)),
            )
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        served.push(provider_of(&resp).unwrap_or_default());
    }
    assert_eq!(
        served[8], "openrouter",
        "the seq-8 probe lands on the keyed, unmeasured arm: {served:?}"
    );
    assert_eq!(anthropic.hits() + openrouter.hits(), 17);
}
