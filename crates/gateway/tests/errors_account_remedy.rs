//! T6: a provider error reaches the client verbatim except a provider-account remedy — advice
//! about Beyond's own upstream account ("add your own key", a billing page) that a gateway client
//! cannot act on, and which names the upstream behind the gateway.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;

/// OpenRouter's shared-pool 429 (D156): the upstream host's limit on OpenRouter's own key is
/// spent, and OpenRouter tells the caller to bring its own key.
const OPENROUTER_SHARED_POOL_429: &str = r#"{"error":{"message":"mistralai/mistral-large-2512 is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","code":429,"metadata":{"raw":"mistralai/mistral-large-2512 is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Mistral","is_byok":false,"limit_source":"upstream_provider_shared_pool","remedy_hint":"Add your own key at https://openrouter.ai/settings/integrations"}},"user_id":"user_mock"}"#;

/// What must not reach a client from that body.
const ACCOUNT_REMEDY: &[&str] = &[
    "openrouter.ai",
    "add your own key",
    "accumulate your rate limits",
    "is_byok",
    "limit_source",
    "upstream_provider_shared_pool",
    "remedy_hint",
];

fn remedy_leaks(text: &str) -> Vec<&'static str> {
    let lower = text.to_ascii_lowercase();
    ACCOUNT_REMEDY
        .iter()
        .copied()
        .filter(|m| lower.contains(m))
        .collect()
}

/// The 429 reaches a Chat client relayed (the OpenRouter provider route, and a catalog walk whose
/// candidate is OpenRouter's Chat endpoint) and a Messages client translated, with its status, its
/// type or code and its `Retry-After`, and a neutral message in place of OpenRouter's account
/// remedy.
/// claim: T6
/// defect: D174
#[tokio::test]
async fn a_provider_account_remedy_never_reaches_the_client() {
    let (pubkey, sk) = test_keypair(174);
    let mock = MockUpstream::start(Mode::Raw(
        429,
        "application/json",
        OPENROUTER_SHARED_POOL_429,
    ))
    .await;
    let gw = Gateway::builder(
        unused_nats_port(),
        &GatewayBuilder::dead_authority(),
        &b64(&pubkey),
    )
    .providers(&["anthropic", "openrouter"])
    .provider_authority("openrouter", &mock.authority())
    .start()
    .await;
    let key = billing_vkey(&sk, 174);
    let chat = r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#;
    let mut failures = Vec::new();

    for path in ["/openrouter/v1/chat/completions", "/v1/chat/completions"] {
        let resp = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(if path.starts_with("/openrouter") {
                chat.replace("claude-opus-4-8", "anthropic/claude-opus-4.8")
            } else {
                chat.to_owned()
            })
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let retry_after = resp
            .headers()
            .get("retry-after")
            .map(|v| v.to_str().unwrap().to_owned());
        let text = resp.text().await.unwrap();
        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let leaks = remedy_leaks(&text);
        let neutral = v["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("rate-limited upstream") && m.contains("retry later"));
        if status != 429
            || retry_after.as_deref() != Some("7")
            || v["error"]["code"] != 429
            || !neutral
            || !leaks.is_empty()
        {
            failures.push(format!(
                "Chat {path}: {status}, retry-after {retry_after:?}, leaks {leaks:?}: {text}"
            ));
        }
    }

    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", &key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let retry_after = resp
        .headers()
        .get("retry-after")
        .map(|v| v.to_str().unwrap().to_owned());
    let text = resp.text().await.unwrap();
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let leaks = remedy_leaks(&text);
    let neutral = v["error"]["message"]
        .as_str()
        .is_some_and(|m| m.contains("rate-limited upstream") && m.contains("retry later"));
    if status != 429
        || retry_after.as_deref() != Some("7")
        || v["error"]["type"] != "rate_limit_error"
        || !neutral
        || !leaks.is_empty()
    {
        failures.push(format!(
            "Messages /v1/messages: {status}, retry-after {retry_after:?}, leaks {leaks:?}: {text}"
        ));
    }

    assert!(
        failures.is_empty(),
        "{}\n{}",
        failures.join("\n\n"),
        gw.log()
    );
}
