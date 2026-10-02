//! Reliability, verify phase 0: configuration that silently disables a feature, and env secrets
//! that never reach their provider.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const BODY: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 9,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

/// A managed request on `/{provider}/v1/chat/completions` whose pool key comes only from the
/// environment. Returns (status, the authorization the upstream saw, the response body).
async fn managed_env_key_roundtrip(
    provider: &'static str,
    env: &str,
) -> (u16, Option<String>, String) {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", provider])
        .pool_keys(provider, &[])
        .env(env, "sk-from-env")
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/{provider}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(BODY)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    (status, mock.captured().and_then(|c| c.authorization), text)
}

/// `AI_POOL_KEY_FIREWORKS_ANTHROPIC` must reach the config-added provider `fireworks-anthropic`
/// (an env var name cannot hold `-`).
/// claim: REL-15
/// defect: D52
#[tokio::test]
async fn env_pool_key_reaches_a_hyphenated_provider() {
    let (status, auth, text) =
        managed_env_key_roundtrip("fireworks-anthropic", "AI_POOL_KEY_FIREWORKS_ANTHROPIC").await;
    assert_eq!(
        (status, auth.as_deref()),
        (200, Some("Bearer sk-from-env")),
        "env pool key never reached fireworks-anthropic: {text}"
    );
}

/// Control for D52: the same env path works for a provider name without a hyphen.
/// claim: REL-15
/// defect: D52
#[tokio::test]
async fn env_pool_key_reaches_a_plain_provider() {
    let (status, auth, text) =
        managed_env_key_roundtrip("fireworks", "AI_POOL_KEY_FIREWORKS").await;
    assert_eq!(
        (status, auth.as_deref()),
        (200, Some("Bearer sk-from-env")),
        "{text}"
    );
}

/// `circuit_breaker_window_secs = 0` with a threshold above 1 must either be refused at boot or
/// still trip the breaker. It must not silently disable it.
/// claim: REL-15
/// defect: D52
#[tokio::test]
async fn a_zero_breaker_window_is_refused_or_still_trips() {
    let lines = "circuit_breaker_threshold = 3\ncircuit_breaker_window_secs = 0";
    if boot_refuses(lines, Duration::from_secs(3)).is_some() {
        return;
    }
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Status(500)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .config_line("circuit_breaker_threshold = 3")
        .config_line("circuit_breaker_window_secs = 0")
        .config_line("circuit_breaker_reset_secs = 60")
        .start()
        .await;
    let client = test_client();
    let mut statuses = Vec::new();
    for _ in 0..12 {
        statuses.push(
            client
                .post(format!("{}/openai/v1/chat/completions", gw.url()))
                .header("authorization", "Bearer sk-byo-test")
                .header("content-type", "application/json")
                .body(BODY)
                .send()
                .await
                .map(|r| r.status().as_u16())
                .unwrap_or(0),
        );
    }
    assert!(
        statuses.contains(&503),
        "window 0 booted and 12 straight 500s never opened the breaker: {statuses:?}"
    );
}
