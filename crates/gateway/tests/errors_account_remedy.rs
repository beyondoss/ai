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

/// OpenAI's TPM "Request too large" 429: the request alone is over the organization's
/// tokens-per-minute limit (measured 2026-10-01 on gpt-4.1 with a 1,101,954-word prompt). It names
/// Beyond's org id and rate-limits page, so the message is rewritten, but to what it means: shrink
/// the request. "Retry later" sent a client (and the catalog sweep) round the same 429 until it
/// gave up. Status, type and code stay.
/// claim: T6
/// defect: D196
#[tokio::test]
async fn a_request_over_the_tpm_limit_is_told_to_shrink_not_to_retry() {
    const TOO_LARGE_429: &str = r#"{"error":{"message":"Request too large for gpt-4.1 (for limit gpt-4.1-long-context) in organization org-abc on tokens per min (TPM): Limit 1000000, Requested 1101959. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.","type":"tokens","param":null,"code":"rate_limit_exceeded"}}"#;
    let (pubkey, sk) = test_keypair(196);
    let mock = MockUpstream::start(Mode::Raw(429, "application/json", TOO_LARGE_429)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 196);
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let message = v["error"]["message"].as_str().unwrap_or_default();
    assert_eq!(status, 429, "{text}\n{}", gw.log());
    assert_eq!(v["error"]["type"], "tokens", "{text}");
    assert_eq!(v["error"]["code"], "rate_limit_exceeded", "{text}");
    assert!(
        message.contains("reduce the input or output tokens") && !message.contains("retry later"),
        "the client is told to shrink the request: {text}"
    );
    assert!(
        !text.contains("org-abc") && !text.contains("platform.openai.com"),
        "the account behind the gateway stays out: {text}"
    );
}

/// A streaming request's provider error is a JSON body like any other: the account remedy in it is
/// neutralized the same way, rather than relayed because the client asked for a stream.
/// claim: T6
/// defect: D174
#[tokio::test]
async fn a_streaming_requests_account_remedy_is_neutralized_too() {
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
    .providers(&["openrouter"])
    .provider_authority("openrouter", &mock.authority())
    .start()
    .await;
    let key = billing_vkey(&sk, 174);
    let resp = test_client()
        .post(format!("{}/openrouter/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(r#"{"model":"anthropic/claude-opus-4.8","stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    let leaks = remedy_leaks(&text);
    assert!(
        status == 429 && leaks.is_empty(),
        "{status}, leaks {leaks:?}: {text}\n{}",
        gw.log()
    );
}

/// The neutralized error is held whole and loses its `Content-Length`; on an HTTP/1.1 upstream it
/// goes to the client chunked, so the connection stays open for the client's next request.
/// claim: T6
/// defect: D174
#[tokio::test]
async fn a_rewritten_error_is_chunked_to_an_http1_client() {
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
    .providers(&["openrouter"])
    .provider_authority("openrouter", &mock.authority())
    .start()
    .await;
    let key = billing_vkey(&sk, 174);
    let body =
        r#"{"model":"anthropic/claude-opus-4.8","messages":[{"role":"user","content":"hi"}]}"#;
    let resp = test_client()
        .post(format!("{}/openrouter/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 429);
    assert_eq!(resp.version(), reqwest::Version::HTTP_11);
    let te = resp
        .headers()
        .get("transfer-encoding")
        .map(|v| v.to_str().unwrap().to_owned());
    let length = resp.headers().get("content-length").cloned();
    let text = resp.text().await.unwrap();
    assert_eq!((te.as_deref(), length), (Some("chunked"), None), "{text}");
    assert!(remedy_leaks(&text).is_empty(), "{text}");
}

/// Only a JSON error is held whole for the rewrite: a plain-text one (an edge's HTML or text
/// page) has no remedy to find and is relayed as sent, its `Content-Length` kept.
/// claim: T6
/// defect: D174
#[tokio::test]
async fn a_non_json_error_is_relayed_with_its_length() {
    const PLAIN: &str = "upstream connect error or disconnect/reset before headers";
    let (pubkey, sk) = test_keypair(174);
    let mock = MockUpstream::start(Mode::Raw(400, "text/plain", PLAIN)).await;
    let gw = Gateway::builder(
        unused_nats_port(),
        &GatewayBuilder::dead_authority(),
        &b64(&pubkey),
    )
    .providers(&["openrouter"])
    .provider_authority("openrouter", &mock.authority())
    .start()
    .await;
    let key = billing_vkey(&sk, 174);
    let resp = test_client()
        .post(format!("{}/openrouter/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(
            r#"{"model":"anthropic/claude-opus-4.8","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let length = resp
        .headers()
        .get("content-length")
        .map(|v| v.to_str().unwrap().to_owned());
    let te = resp.headers().get("transfer-encoding").cloned();
    let text = resp.text().await.unwrap();
    assert_eq!(text, PLAIN);
    assert_eq!(
        (length, te),
        (Some(PLAIN.len().to_string()), None),
        "{}",
        gw.log()
    );
}
