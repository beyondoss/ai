//! Claim tests: what a stock Chat Completions request becomes on its way to the serving model.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;

/// The same Chat request that works on every Claude row, sent to a GPT-5 row on the same wire.
/// OpenAI rejects `max_tokens` on reasoning models ("Use 'max_completion_tokens' instead"), so the
/// limit must reach it as `max_completion_tokens`, as a translated walk already sends it.
/// claim: E1, T1
/// defect: D63
#[tokio::test]
async fn a_chat_max_tokens_reaches_a_gpt5_row_as_max_completion_tokens() {
    let (pubkey, sk) = test_keypair(74);
    let mock = MockUpstream::start(Mode::OpenAiToolJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", billing_vkey(&sk, 74)))
        .header("content-type", "application/json")
        .body(
            r#"{"model":"gpt-5.1","max_tokens":256,"messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{}}}}]}"#,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let sent: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
    assert_eq!(sent["model"], "gpt-5.1", "{sent}");
    assert!(
        sent.get("max_tokens").is_none(),
        "OpenAI 400s max_tokens on a reasoning model: {sent}"
    );
    assert_eq!(sent["max_completion_tokens"], 256, "{sent}");
}
