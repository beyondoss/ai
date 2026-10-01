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
/// claim: REL-9, REL-6
#[tokio::test]
async fn client_cancellations_do_not_open_the_providers_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    // Slower than the client's patience below, so every one of those requests is abandoned
    // mid-flight — while the upstream itself stays perfectly healthy.
    let mock = MockUpstream::start(Mode::Slow(3_000)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
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
/// claim: REL-6
#[tokio::test]
async fn upstream_failures_still_open_the_breaker() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Status(500)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
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

/// A reused connection that fails after the upstream drained the whole body (the mock reads the
/// request, then closes) is **not** retried. Pingora would resend it on its own, without calling
/// `fail_to_connect`, but the provider may already be generating and billing that request, so the
/// gateway declines the reuse retry once the body was delivered (`error_while_proxy`) and the
/// client gets a JSON 502. This test used to assert the retry; that resend is the duplicate spend
/// D09 removed. The breaker still sees exactly one outcome per attempt.
///
/// The mock kills any request that is not the first on its connection, so the scenario fires the
/// moment pingora reuses a pooled connection — whenever that happens to be, rather than on a fixed
/// request number that may well land on a fresh connection under load.
/// claim: REL-1
#[tokio::test]
async fn a_reused_connection_failure_after_the_body_is_not_resent() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::CloseOnReusedConnection).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .start()
        .await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    // Streaming + managed + OpenAI chat = the inject-eligible path, buffered and spliced. The mock
    // drains the request before killing the connection, so the failure is on the response-header
    // read, after the body was written.
    let filler = "y".repeat(8 * 1024);
    let body = format!(
        r#"{{"model":"gpt-4o-mini","stream":true,"messages":[{{"role":"user","content":"{filler}"}}]}}"#
    );

    // Sequential, so each request has a pooled connection available to reuse.
    const SENT: usize = 6;
    let mut failed = 0;
    for i in 0..SENT {
        let resp = client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {vkey}"))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap_or_else(|e| panic!("request {i}: {e}; log:\n{}", gw.log()));
        match resp.status().as_u16() {
            200 => {}
            502 => {
                failed += 1;
                assert!(
                    resp.headers().contains_key("x-beyond-request-id"),
                    "request {i}: the 502 carries the request id"
                );
                let text = resp.text().await.unwrap_or_default();
                let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                assert!(v["error"]["message"].is_string(), "request {i}: {text}");
            }
            s => panic!("request {i}: unexpected status {s}"),
        }
    }
    assert!(
        failed > 0,
        "no reused-connection failure occurred, so the scenario did not fire and this test proved \
         nothing"
    );
    assert_eq!(
        mock.hits(),
        SENT,
        "a delivered body was resent ({} upstream requests for {SENT} client requests)",
        mock.hits(),
    );
}

/// A retry pingora *does* make — the managed 429 key walk — replays its buffered request body
/// through `request_body_filter`, which is what `RequestCtx::reset_request_body_phase` exists to
/// make idempotent. The retried body must be whole and well-formed, not the original with a
/// replayed prefix concatenated onto it, and carry the usage splice exactly once.
#[tokio::test]
async fn a_key_walk_retry_replays_the_body_exactly_once() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::ThrottleKey("sk-replay-a")).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-replay-a", "sk-replay-b"])
        .start()
        .await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        &sk,
    );
    // 8 KiB: a real buffered body, comfortably under pingora's 64 KiB retry buffer, so the retry
    // genuinely replays rather than declining to.
    let filler = "y".repeat(8 * 1024);
    let body = format!(
        r#"{{"model":"gpt-4o-mini","stream":true,"messages":[{{"role":"user","content":"{filler}"}}]}}"#
    );
    let resp = reqwest::Client::new()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(mock.hits(), 2, "the throttled key, then the walked one");

    let cap = mock.captured().expect("the retry reached the upstream");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-replay-b"));
    let received: serde_json::Value = serde_json::from_slice(&cap.body).unwrap_or_else(|e| {
        panic!("retried body is not valid JSON ({e}) — the replay was appended, not reset")
    });
    assert_eq!(received["model"], "gpt-4o-mini");
    assert_eq!(received["messages"][0]["content"], filler);
    assert_eq!(
        received["stream_options"]["include_usage"], true,
        "the usage injection must still be spliced exactly once on the retried attempt",
    );
}
