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

/// Providers quote the request in their errors: an unknown model (`The model `X` does not
/// exist`), an unknown parameter (`Unrecognized request argument supplied: X`), an extra field
/// (`X: Extra inputs are not permitted`). A tenant who puts an out-of-credit phrase in one of
/// those (a model named "insufficient balance", a root key named after Anthropic's billing
/// sentence) gets a 400 or 404 that quotes it. That is the tenant's own mistake: it must not cool
/// Beyond's shared pool key for every tenant (the D84 class), and the tenant's error is relayed as
/// the provider wrote it, never replaced with the "cannot serve" text. Every request goes to the
/// provider that answered, and no key is counted as failing.
/// claim: REL-4, SEC-16, T6
/// defect: D200
#[tokio::test]
async fn an_echoed_out_of_credit_phrase_never_cools_the_pool_key() {
    const OPENAI_MODEL_ECHO: &str = r#"{"error":{"message":"The model `insufficient balance` does not exist or you do not have access to it.","type":"invalid_request_error","param":null,"code":"model_not_found"}}"#;
    const OPENAI_PARAM_ECHO: &str = r#"{"error":{"message":"Unrecognized request argument supplied: check your plan and billing details","type":"invalid_request_error","param":null,"code":null}}"#;
    const ANTHROPIC_EXTRA_INPUTS: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.: Extra inputs are not permitted"}}"#;
    const ANTHROPIC_MODEL_ECHO: &str = r#"{"type":"error","error":{"type":"not_found_error","message":"model: purchase more credits at https://openrouter.ai/settings/credits"}}"#;
    let (pubkey, sk) = test_keypair(200);
    let key = billing_vkey(&sk, 200);
    let mut failures = Vec::new();
    for (provider, model, status, body) in [
        ("openai", "gpt-4o-mini", 404, OPENAI_MODEL_ECHO),
        ("openai", "gpt-4o-mini", 400, OPENAI_PARAM_ECHO),
        ("anthropic", "claude-opus-4-8", 400, ANTHROPIC_EXTRA_INPUTS),
        ("anthropic", "claude-opus-4-8", 404, ANTHROPIC_MODEL_ECHO),
    ] {
        let echo = MockUpstream::start(Mode::Raw(status, "application/json", body)).await;
        let fallback = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(unused_nats_port(), &fallback.authority(), &b64(&pubkey))
            .providers(&[provider, "openrouter"])
            .provider_authority(provider, &echo.authority())
            .start()
            .await;
        let want: Value = serde_json::from_str(body).unwrap();
        for n in 1..=3 {
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
            let got = resp.status().as_u16();
            let by = resp
                .headers()
                .get("x-beyond-provider")
                .map(|v| v.to_str().unwrap().to_owned());
            let text = resp.text().await.unwrap();
            let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            if got != status
                || by.as_deref() != Some(provider)
                || v["error"]["message"] != want["error"]["message"]
            {
                failures.push(format!(
                    "{provider} {status} request {n}: {got} by {by:?}: {text}"
                ));
            }
        }
        let cooled = gw
            .metric("ai_key_auth_failures_total", r#"reason="unfunded""#)
            .await;
        if echo.hits() != 3 || fallback.hits() != 0 || cooled != 0.0 {
            failures.push(format!(
                "{provider} {status}: {} hits on the provider, {} on the fallback, {cooled} keys cooled",
                echo.hits(),
                fallback.hits()
            ));
        }
        if !failures.is_empty() {
            failures.push(gw.log());
            break;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Every way a pool key is cooled is counted on `ai_key_auth_failures_total` with its reason, so
/// an operator can tell a key to replace (`revoked`: a 401; `key_named_403`: a 403 whose body names
/// the key) from an account to fund (`unfunded`). Each answer counts once, under its own reason.
/// claim: REL-4, O3
/// defect: D204
#[tokio::test]
async fn each_cooled_key_is_counted_with_its_reason() {
    const BAD_KEY: &str = r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}"#;
    let (pubkey, sk) = test_keypair(204);
    let key = billing_vkey(&sk, 204);
    let mut failures = Vec::new();
    for (status, body, reason) in [
        (401, BAD_KEY, "revoked"),
        (403, BAD_KEY, "key_named_403"),
        (429, OPENAI_NO_QUOTA, "unfunded"),
    ] {
        let mock = MockUpstream::start(Mode::Raw(status, "application/json", body)).await;
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .providers(&["openai"])
            .start()
            .await;
        let resp = test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), status);
        let _ = resp.text().await;
        let metrics = gw.metrics().await;
        for other in ["revoked", "key_named_403", "unfunded"] {
            let label = format!("reason=\"{other}\"");
            let got = parse_metric(&metrics, "ai_key_auth_failures_total{", &label);
            let want = if other == reason { 1.0 } else { 0.0 };
            if got != want {
                failures.push(format!("{status}: {label} = {got}, want {want}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The cooldown is per candidate wherever it sits in the row: an unfunded provider at the row's
/// second catalog slot, walked first by `x-beyond-order`, is passed over for the next requests
/// just as one in the first slot is. (Its out-of-quota 429 is relayed, and cools its key.)
/// claim: REL-4
/// defect: D180
#[tokio::test]
async fn an_unfunded_candidate_cools_off_from_any_catalog_slot() {
    let (pubkey, sk) = test_keypair(180);
    let key = billing_vkey(&sk, 181);
    let unfunded = MockUpstream::start(Mode::Raw(429, "application/json", OPENAI_NO_QUOTA)).await;
    let openai = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &unfunded.authority())
        .start()
        .await;
    let mut served = Vec::new();
    for _ in 0..3 {
        let resp = test_client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .header("x-beyond-order", "openrouter,openai")
            .body(r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#)
            .send()
            .await
            .unwrap();
        let by = resp
            .headers()
            .get("x-beyond-provider")
            .map(|v| v.to_str().unwrap().to_owned());
        served.push((resp.status().as_u16(), by));
    }
    let by = |p: &str| Some(p.to_owned());
    assert_eq!(
        served,
        [
            (429, by("openrouter")),
            (200, by("openai")),
            (200, by("openai"))
        ],
        "{}",
        gw.log()
    );
    assert_eq!(
        unfunded.hits(),
        1,
        "OpenRouter was tried after it said it had no quota"
    );
}

/// A catalog walk fails over on a 402 at the response head (D84), so the body `logging` reads to
/// cool an out-of-credit key (D180) is never read. Anthropic's `billing_error` 402 says out of
/// credit by its status alone, so the walk cools the key there: the first request is served by the
/// next candidate, and every request after it, for the cooldown, goes straight to it. OpenRouter's
/// 402 can instead be one request larger than the balance ("This request requires more credits"),
/// which a smaller request is not: that one fails over too, and cools nothing, so OpenRouter keeps
/// getting the row's requests. Both hold for a body pingora replays and for one past its 64 KiB
/// retry buffer, which fails over by a `FullBody` re-run.
/// claim: REL-4
/// defect: D258
#[tokio::test]
async fn a_402_the_walk_fails_over_on_cools_an_unfunded_key() {
    const ANTHROPIC_BILLING: &str = r#"{"type":"error","error":{"type":"billing_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}"#;
    const OPENROUTER_TOO_COSTLY: &str = r#"{"error":{"message":"This request requires more credits, or fewer max_tokens. You requested up to 64000 tokens, but can only afford 1234. To increase, visit https://openrouter.ai/settings/credits and add more credits","code":402}}"#;
    let (pubkey, sk) = test_keypair(158);
    let key = billing_vkey(&sk, 158);
    let mut failures = Vec::new();
    let cases = [
        // (refusing provider, its 402, the next candidate, its mock, whether the 402 cools)
        (
            "anthropic",
            ANTHROPIC_BILLING,
            "openrouter",
            Mode::Json,
            true,
        ),
        (
            "openrouter",
            OPENROUTER_TOO_COSTLY,
            "anthropic",
            Mode::AnthropicJson,
            false,
        ),
    ];
    // Replayed by pingora, and past its 64 KiB retry buffer (a `FullBody` re-run).
    let large = "x".repeat(100 * 1024);
    for ((first, body, next, mode, cools), prompt) in cases
        .into_iter()
        .flat_map(|c| [(c, "hi"), (c, large.as_str())])
    {
        let size = prompt.len();
        let refusing = MockUpstream::start(Mode::Raw(402, "application/json", body)).await;
        let serving = MockUpstream::start(mode).await;
        let gw = Gateway::builder(unused_nats_port(), &serving.authority(), &b64(&pubkey))
            .providers(&["anthropic", "openrouter"])
            .provider_authority(first, &refusing.authority())
            .provider_authority(next, &serving.authority())
            .start()
            .await;
        for n in 1..=3 {
            let resp = test_client()
                .post(format!("{}/v1/chat/completions", gw.url()))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .header("x-beyond-order", format!("{first},{next}"))
                .body(format!(
                    r#"{{"model":"claude-opus-4-8","messages":[{{"role":"user","content":"{prompt}"}}]}}"#
                ))
                .send()
                .await
                .unwrap();
            let got = resp.status().as_u16();
            let by = resp
                .headers()
                .get("x-beyond-provider")
                .map(|v| v.to_str().unwrap().to_owned());
            let text = resp.text().await.unwrap();
            if got != 200 || by.as_deref() != Some(next) {
                failures.push(format!(
                    "{first} ({size} B) request {n}: {got} by {by:?}: {text}"
                ));
            }
        }
        let cooled = gw
            .metric("ai_key_auth_failures_total", r#"reason="unfunded""#)
            .await;
        let (want_hits, want_cooled) = if cools { (1, 1.0) } else { (3, 0.0) };
        if refusing.hits() != want_hits || serving.hits() != 3 || cooled != want_cooled {
            failures.push(format!(
                "{first} ({size} B): {} hits on its 402 (want {want_hits}), {} on {next}, {cooled} keys cooled (want {want_cooled})",
                refusing.hits(),
                serving.hits(),
            ));
        }
        if !failures.is_empty() {
            failures.push(gw.log());
            break;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
