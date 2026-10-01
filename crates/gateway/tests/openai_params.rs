//! End-to-end: request parameters a native OpenAI candidate needs in its own spelling.
//!
//! OpenAI rejects `max_tokens` on its reasoning models ("Use 'max_completion_tokens' instead"),
//! while every OpenAI Chat model accepts `max_completion_tokens`. A stock SDK call that works on a
//! Claude row must work on a GPT row too, so a managed walk onto native OpenAI Chat sends the output
//! limit as `max_completion_tokens`. Other OpenAI-wire hosts keep `max_tokens`, which is what they
//! document.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn managed_key(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 5,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

async fn forwarded_body(providers: &[&'static str], model: &str, body: &str) -> serde_json::Value {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(providers)
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", managed_key(&sk)))
        .header("content-type", "application/json")
        .body(body.replace("MODEL", model))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "{}",
        resp.text().await.unwrap()
    );
    serde_json::from_slice(&mock.captured().expect("forwarded").body).unwrap()
}

/// claim: E1, TRN-5
/// defect: D63
#[tokio::test]
async fn a_stock_max_tokens_reaches_native_openai_as_max_completion_tokens() {
    let body = r#"{"model":"MODEL","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}"#;
    for model in ["gpt-5.1", "gpt-5.2", "gpt-4o-mini"] {
        let sent = forwarded_body(&["openai"], model, body).await;
        assert_eq!(sent["max_completion_tokens"], 1024, "{model}: {sent}");
        assert!(
            sent.get("max_tokens").is_none(),
            "{model}: max_tokens must not reach OpenAI: {sent}"
        );
    }
}

/// A client that already sends `max_completion_tokens` is left alone, and a value it sends in both
/// spellings keeps the explicit one.
///
/// claim: E1, TRN-5
#[tokio::test]
async fn an_explicit_max_completion_tokens_is_kept() {
    let body = r#"{"model":"MODEL","max_completion_tokens":512,"messages":[{"role":"user","content":"hi"}]}"#;
    let sent = forwarded_body(&["openai"], "gpt-5.1", body).await;
    assert_eq!(sent["max_completion_tokens"], 512, "{sent}");
}
