//! A pool key that is out of credit is a candidate refusal: later requests go to the next
//! candidate rather than to the unfunded one (D180).
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;

/// Anthropic's answer on an organization with no credit left: a 400, typed as the request's fault.
const ANTHROPIC_NO_CREDIT: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}"#;

/// OpenAI's answer on a project with no quota left: a 429 that waiting does not clear.
const OPENAI_NO_QUOTA: &str = r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#;

/// Each provider's out-of-credit answer reaches the first request with its status and type and a
/// neutral message that neither says "rate limit" nor describes the account; every request after
/// it, for the cooldown, is served by the next candidate (OpenRouter).
/// claim: REL-4
/// defect: D180
#[tokio::test]
async fn an_unfunded_pool_key_sends_later_requests_to_the_next_candidate() {
    let (pubkey, sk) = test_keypair(180);
    let key = billing_vkey(&sk, 180);
    let mut failures = Vec::new();
    for (provider, model, status, body) in [
        ("anthropic", "claude-opus-4-8", 400, ANTHROPIC_NO_CREDIT),
        ("openai", "gpt-4o-mini", 429, OPENAI_NO_QUOTA),
    ] {
        let unfunded = MockUpstream::start(Mode::Raw(status, "application/json", body)).await;
        let fallback = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(unused_nats_port(), &fallback.authority(), &b64(&pubkey))
            .providers(&[provider, "openrouter"])
            .provider_authority(provider, &unfunded.authority())
            .start()
            .await;
        let send = || async {
            let resp = test_client()
                .post(format!("{}/v1/chat/completions", gw.url()))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .header("x-beyond-order", format!("{provider},openrouter"))
                .body(format!(
                    r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#
                ))
                .send()
                .await
                .unwrap();
            let status = resp.status().as_u16();
            let served_by = resp
                .headers()
                .get("x-beyond-provider")
                .map(|v| v.to_str().unwrap().to_owned());
            (status, served_by, resp.text().await.unwrap())
        };

        let (got, by, text) = send().await;
        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        if got != status
            || by.as_deref() != Some(provider)
            || !msg.contains("cannot serve this request right now")
            || msg.contains("rate-limited")
            || text.contains("billing")
            || text.contains("credit")
        {
            failures.push(format!("{provider} first: {got} by {by:?}: {text}"));
        }
        for n in 2..=3 {
            let (got, by, text) = send().await;
            if got != 200 || by.as_deref() != Some("openrouter") {
                failures.push(format!("{provider} request {n}: {got} by {by:?}: {text}"));
            }
        }
        if unfunded.hits() != 1 {
            failures.push(format!(
                "{provider} was sent {} requests after it said it had no credit",
                unfunded.hits()
            ));
        }
        if !failures.is_empty() {
            failures.push(gw.log());
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
