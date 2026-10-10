//! Which request bodies the gateway buffers, and which it charges to the body budget up front:
//! a managed body it must read whole (the `stream_options` splice, the catalog check), and never a
//! BYO one.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;

/// A BYO stream is relayed as the caller sent it: the managed `stream_options` splice (which
/// guarantees a billing row its usage chunk) is managed-only.
/// claim: SEC-19
/// defect: D267
#[tokio::test]
async fn a_byo_stream_is_not_given_stream_options() {
    let (pubkey, _sk) = test_keypair(170);
    let mock = MockUpstream::start(Mode::Sse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let body =
        r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-caller-own-key")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let _ = resp.bytes().await;
    let sent = mock.captured().unwrap().body;
    assert_eq!(sent, body.as_bytes(), "a BYO body goes as sent");
}

/// A managed `/{provider}` body the catalog check must read whole (here Anthropic Messages, which
/// the `stream_options` splice never touches) reserves its declared length up front: past the
/// budget it is the 503 before any provider is contacted.
/// claim: SEC-19
/// defect: D267
#[tokio::test]
async fn a_large_catalog_checked_body_reserves_before_any_upstream() {
    let (pubkey, sk) = test_keypair(171);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .config_line("max_buffered_body_bytes = 100000")
        .start()
        .await;
    let body = format!(
        r#"{{"model":"claude-opus-4-8","max_tokens":16,"messages":[{{"role":"user","content":"{}"}}]}}"#,
        "x".repeat(200_000)
    );
    let resp = test_client()
        .post(format!("{}/anthropic/v1/messages", gw.url()))
        .header("x-api-key", billing_vkey(&sk, 171))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    assert_eq!(status, 503, "{text}");
    assert_eq!(mock.hits(), 0, "refused before any provider was contacted");
}
