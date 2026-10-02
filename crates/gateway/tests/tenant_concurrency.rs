//! End-to-end: the per-tenant in-flight cap (`tenant_max_in_flight`).
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).
//!
//! Spend is enforced after the fact, so the overshoot while the allowance-set's exhaust bit is in
//! transit is `lag × requests in flight × cost per request`. The cap bounds the middle term. What
//! must hold: a tenant at its ceiling gets a 429 without touching the provider, other tenants are
//! unaffected, and every slot comes back — on completion *and* when the client walks away.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code — allow the
// panic-surface restriction lints denied workspace-wide in `[workspace.lints.clippy]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use std::time::Duration;

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

fn body() -> String {
    r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#.to_string()
}

async fn post(client: &reqwest::Client, gw: &Gateway, key: &str) -> reqwest::Response {
    client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap()
}

/// Wait until the mock has seen `n` requests — i.e. that many are genuinely in flight upstream.
async fn wait_for_hits(mock: &MockUpstream, n: usize) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while mock.hits() < n {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("request reached the upstream");
}

/// claim: A3
#[tokio::test]
async fn a_tenant_at_its_ceiling_is_refused_and_recovers() {
    let (pubkey, sk) = test_keypair(51);
    let mock = MockUpstream::start(Mode::Slow(1_000)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .tenant_max_in_flight(1)
        .start()
        .await;
    let client = test_client();
    let (a, b) = (vkey(&sk, 51), vkey(&sk, 52));

    let held = {
        let (client, url, a) = (client.clone(), gw.url(), a.clone());
        tokio::spawn(async move {
            client
                .post(format!("{url}/openai/v1/chat/completions"))
                .header("authorization", format!("Bearer {a}"))
                .header("content-type", "application/json")
                .body(body())
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        })
    };
    wait_for_hits(&mock, 1).await;

    let refused = post(&client, &gw, &a).await;
    assert_eq!(refused.status().as_u16(), 429);
    assert!(
        refused
            .text()
            .await
            .unwrap()
            .contains("too many concurrent requests")
    );
    assert_eq!(
        mock.hits(),
        1,
        "a refused request never reaches the provider"
    );
    wait_for_metric(&gw, "ai_rejections_total", "tenant_concurrency", 1.0).await;

    // Another tenant has its own ceiling.
    assert_eq!(post(&client, &gw, &b).await.status().as_u16(), 200);

    // The held request finishes and gives its slot back.
    assert_eq!(held.await.unwrap(), 200);
    wait_for_status(200, || {
        let (client, a) = (client.clone(), a.clone());
        let gw_url = gw.url();
        async move {
            client
                .post(format!("{gw_url}/openai/v1/chat/completions"))
                .header("authorization", format!("Bearer {a}"))
                .header("content-type", "application/json")
                .body(body())
                .send()
                .await
                .map(|r| r.status().as_u16())
                .unwrap_or(0)
        }
    })
    .await;
}

/// A client that hangs up mid-request must not strand its slot — otherwise a tenant whose users
/// cancel often would ratchet down to zero capacity.
/// claim: REL-9
#[tokio::test]
async fn a_cancelled_request_releases_its_slot() {
    let (pubkey, sk) = test_keypair(53);
    let mock = MockUpstream::start(Mode::Slow(600)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .tenant_max_in_flight(1)
        .start()
        .await;
    let key = vkey(&sk, 53);
    let impatient = reqwest::Client::builder()
        .timeout(Duration::from_millis(100))
        .build()
        .unwrap();
    for _ in 0..3 {
        let _ = impatient
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await;
        // The abandoned request's slot is released when its upstream attempt ends.
        tokio::time::sleep(Duration::from_millis(700)).await;
    }
    let client = test_client();
    assert_eq!(
        post(&client, &gw, &key).await.status().as_u16(),
        200,
        "three cancellations left the tenant with a slot"
    );
}
