//! Claim tests: `bai_v3` expiry, signing-kid distrust, and signing keys published at runtime, each
//! through the proxy as a client sees it.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::__test::key::{VirtualKey, mint_v3};
use common::*;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

const CHAT: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn v3(sk: &ed25519_dalek::SigningKey, tenant: u64, exp: u64, kid: u32) -> String {
    let id = VirtualKey {
        tenant_id: tenant,
        vpc_id: 1,
        key_id: None,
    };
    mint_v3(&id, tenant * 10, exp, kid, sk)
}

async fn send(gw: &Gateway, key: &str) -> reqwest::Response {
    test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap()
}

async fn status_of(gw: &Gateway, key: &str) -> u16 {
    send(gw, key).await.status().as_u16()
}

/// A `bai_v3` key is served until its `exp` and refused after it with its own 401, and the expired
/// refusal never reaches a provider.
/// claim: A5
#[tokio::test]
async fn an_expired_v3_key_is_a_401_that_says_so() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(61);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let live = v3(&sk, 6101, now() + 3600, 1);
    let stale = v3(&sk, 6102, now() - 3600, 1);
    wait_for_status(200, || async { status_of(&gw, &live).await }).await;

    let hits = mock.hits();
    let resp = send(&gw, &stale).await;
    assert_eq!(resp.status().as_u16(), 401);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["message"], "key expired");
    assert_eq!(body["error"]["type"], "authentication_error");
    assert_eq!(
        mock.hits(),
        hits,
        "an expired key must not reach the provider"
    );

    let metrics = gw.metrics().await;
    assert!(parse_metric(&metrics, "ai_rejections_total", "key_expired") >= 1.0);
    // The same key, one second inside the leeway, is still served.
    let edge = v3(&sk, 6103, now() - 5, 1);
    assert_eq!(status_of(&gw, &edge).await, 200);
}

/// A `blackhole.kid.{kid}` entry refuses every token under that kid with a 401, and deleting the
/// entry restores them.
/// claim: A6
#[tokio::test]
async fn a_distrusted_kid_is_a_401_until_the_entry_is_deleted() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(62);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let key = v3(&sk, 6201, now() + 3600, 1);
    wait_for_status(200, || async { status_of(&gw, &key).await }).await;

    put_kv(nats.port, "blackhole.kid.1", b"fraud").await;
    wait_for_status(401, || async { status_of(&gw, &key).await }).await;
    let body: Value = send(&gw, &key).await.json().await.unwrap();
    assert_eq!(body["error"]["message"], "invalid API key");

    del_kv(nats.port, "blackhole.kid.1").await;
    wait_for_status(200, || async { status_of(&gw, &key).await }).await;
}

/// A signing key the control plane writes to KV is trusted without a restart, and its delete
/// retires it.
/// claim: A7
#[tokio::test]
async fn a_signing_key_in_kv_is_trusted_until_it_is_deleted() {
    let nats = Nats::start().await;
    let (pubkey, _) = test_keypair(63);
    let (rotated_pub, rotated_sk) = test_keypair(64);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let key = v3(&rotated_sk, 6301, now() + 3600, 2);
    assert_eq!(
        status_of(&gw, &key).await,
        401,
        "kid 2 is in neither source yet"
    );

    put_kv(nats.port, "signing_key.2", b64(&rotated_pub).as_bytes()).await;
    wait_for_status(200, || async { status_of(&gw, &key).await }).await;

    del_kv(nats.port, "signing_key.2").await;
    wait_for_status(401, || async { status_of(&gw, &key).await }).await;
}
