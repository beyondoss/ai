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
#[tokio::test]
async fn client_cancellations_do_not_open_the_providers_breaker() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    // Slower than the client's patience below, so every one of those requests is abandoned
    // mid-flight — while the upstream itself stays perfectly healthy.
    let mock = MockUpstream::start(Mode::Slow(3_000)).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
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
#[tokio::test]
async fn upstream_failures_still_open_the_breaker() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Status(500)).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
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
