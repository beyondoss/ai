//! Reliability, verify phase 0: a body past pingora's 64 KiB replay buffer (the `FullBody`
//! subrequest path) must behave like a small one in the same scenario.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const MODEL: &str = "gpt-4o-mini";

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 11,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

/// A chat body with `pad` bytes of content.
fn body(pad: usize) -> String {
    format!(
        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "x".repeat(pad)
    )
}

const SMALL: usize = 16;
const LARGE: usize = 200 * 1024;

/// What the client saw: status, serving provider header, error message.
#[derive(Debug, PartialEq)]
struct Outcome {
    status: u16,
    provider: Option<String>,
    message: Option<String>,
}

async fn send(gw: &Gateway, key: &str, pad: usize) -> Outcome {
    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .body(body(pad))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let provider = resp
        .headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let text = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(String::from));
    Outcome {
        status,
        provider,
        message,
    }
}

/// The primary throttles every one of its pool keys; the fallback is healthy. A 429 is a key walk
/// on the same vendor, never a vendor switch, so both body sizes must end on the primary's 429.
/// claim: REL-21
/// defect: D51
#[tokio::test]
async fn a_429_key_walk_ends_the_same_for_small_and_large_bodies() {
    let (pubkey, sk) = test_keypair(1);
    let mut outcomes = Vec::new();
    for pad in [SMALL, LARGE] {
        let primary = MockUpstream::start(Mode::Status(429)).await;
        let fallback = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .pool_keys("openai", &["sk-a", "sk-b"])
            .start()
            .await;
        let out = send(&gw, &vkey(&sk), pad).await;
        outcomes.push((out, primary.hits(), fallback.hits()));
    }
    let (small, large) = (&outcomes[0], &outcomes[1]);
    assert_eq!(small.0.status, 429, "small: {small:?}");
    assert_eq!(small, large, "small vs large body diverged");
}

/// Every candidate resets the connection. Whatever the gateway answers, a large body must get the
/// same status as a small one, and must not blame a missing provider key.
/// claim: REL-21
/// defect: D51
#[tokio::test]
#[ignore = "D51 reproduced: all-candidate resets give small empty 502 vs large 503 'no provider key available'"]
async fn resets_on_every_candidate_end_the_same_for_small_and_large_bodies() {
    let (pubkey, sk) = test_keypair(1);
    let mut outcomes = Vec::new();
    for pad in [SMALL, LARGE] {
        let primary = ReplyUpstream::start(|_, _| Reply::Reset).await;
        let fallback = ReplyUpstream::start(|_, _| Reply::Reset).await;
        let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .start()
            .await;
        let out = send(&gw, &vkey(&sk), pad).await;
        outcomes.push((out, primary.hits(), fallback.hits()));
    }
    let (small, large) = (&outcomes[0], &outcomes[1]);
    let misleading = large
        .0
        .message
        .as_deref()
        .is_some_and(|m| m.contains("no provider key"));
    assert!(
        small.0.status == large.0.status && !misleading,
        "small {small:?} vs large {large:?} (outcome, primary hits, fallback hits)"
    );
}
