//! An OpenRouter `402` a catalog walk fails over on is read at the head only, so it cannot say
//! whether the account is out of credit or one request was too costly for the balance left (D258).
//! A run of them on one key with no success between is the account: the key cools (D261).
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::route::OPENROUTER_402_STRIKES;
use common::*;

/// OpenRouter's answer on an account with no credit left.
const OPENROUTER_NO_CREDIT: &str = r#"{"error":{"message":"Insufficient credits. This account never purchased credits. Make sure your key is on the correct account or org, and if so, purchase more at https://openrouter.ai/settings/credits","code":402}}"#;

/// OpenRouter's answer to one request larger than a funded balance can pay for.
const OPENROUTER_TOO_COSTLY: &str = r#"{"error":{"message":"This request requires more credits, or fewer max_tokens. You requested up to 64000 tokens, but can only afford 1234. To increase, visit https://openrouter.ai/settings/credits and add more credits","code":402}}"#;

/// What the too-costly mock refuses: a request whose prompt says so.
const TOO_BIG: &str = "too-big";

/// [`OPENROUTER_402_STRIKES`] as a request count.
fn strikes() -> usize {
    usize::try_from(OPENROUTER_402_STRIKES).unwrap()
}

/// One managed request on claude-opus-4-8 with OpenRouter first: (status, served by, body).
async fn send(gw: &Gateway, key: &str, prompt: &str) -> (u16, Option<String>, String) {
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-order", "openrouter,anthropic")
        .body(format!(
            r#"{{"model":"claude-opus-4-8","messages":[{{"role":"user","content":"{prompt}"}}]}}"#
        ))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let by = resp
        .headers()
        .get("x-beyond-provider")
        .map(|v| v.to_str().unwrap().to_owned());
    (status, by, resp.text().await.unwrap())
}

async fn gateway(openrouter: &MockUpstream, anthropic: &MockUpstream, pubkey: &[u8]) -> Gateway {
    Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .provider_authority("anthropic", &anthropic.authority())
        .start()
        .await
}

/// OpenRouter leads the walk and 402s every request: each fails over to Anthropic and is served,
/// and the [`OPENROUTER_402_STRIKES`]th in a row cools OpenRouter's key, so every request after
/// it goes straight to Anthropic. Holds for a body pingora replays and for one past its 64 KiB
/// retry buffer, which fails over by a `FullBody` re-run.
/// claim: REL-4
/// defect: D261
#[tokio::test]
async fn openrouter_402s_in_a_row_cool_its_key() {
    let (pubkey, sk) = test_keypair(161);
    let key = billing_vkey(&sk, 161);
    let large = "x".repeat(100 * 1024);
    let mut failures = Vec::new();
    for prompt in ["hi", large.as_str()] {
        let size = prompt.len();
        let openrouter =
            MockUpstream::start(Mode::Raw(402, "application/json", OPENROUTER_NO_CREDIT)).await;
        let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
        let gw = gateway(&openrouter, &anthropic, &pubkey).await;
        let requests = strikes() + 3;
        for n in 1..=requests {
            let (status, by, text) = send(&gw, &key, prompt).await;
            if status != 200 || by.as_deref() != Some("anthropic") {
                failures.push(format!(
                    "({size} B) request {n}: {status} by {by:?}: {text}"
                ));
            }
        }
        let cooled = gw
            .metric("ai_key_auth_failures_total", r#"reason="unfunded""#)
            .await;
        if openrouter.hits() != strikes() || anthropic.hits() != requests || cooled != 1.0 {
            failures.push(format!(
                "({size} B): {} hits on OpenRouter (want {}), {} on Anthropic (want {requests}), {cooled} keys cooled (want 1)",
                openrouter.hits(),
                strikes(),
                anthropic.hits(),
            ));
        }
        if !gw.log().contains("pool key is out of credit; cooling it") {
            failures.push(format!("({size} B): no cooling warn line"));
        }
        if !failures.is_empty() {
            failures.push(gw.log());
            break;
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// A funded OpenRouter account refuses only the requests too costly for its balance. Runs of
/// those one short of [`OPENROUTER_402_STRIKES`], each ended by a request OpenRouter serves, never
/// cool the key: every request reaches OpenRouter, the costly ones fail over to Anthropic and the
/// rest are served by OpenRouter.
/// claim: REL-4
/// defect: D261
#[tokio::test]
async fn too_costly_openrouter_402s_between_successes_never_cool_its_key() {
    let (pubkey, sk) = test_keypair(162);
    let key = billing_vkey(&sk, 162);
    let openrouter = MockUpstream::start(Mode::RawWhenBody(
        TOO_BIG,
        402,
        "application/json",
        OPENROUTER_TOO_COSTLY,
    ))
    .await;
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = gateway(&openrouter, &anthropic, &pubkey).await;
    let mut failures = Vec::new();
    let mut sent = 0;
    let mut costly = 0;
    for round in 1..=3 {
        for _ in 1..strikes() {
            let (status, by, text) = send(&gw, &key, TOO_BIG).await;
            sent += 1;
            costly += 1;
            if status != 200 || by.as_deref() != Some("anthropic") {
                failures.push(format!("round {round}, costly: {status} by {by:?}: {text}"));
            }
        }
        let (status, by, text) = send(&gw, &key, "hi").await;
        sent += 1;
        if status != 200 || by.as_deref() != Some("openrouter") {
            failures.push(format!("round {round}, small: {status} by {by:?}: {text}"));
        }
    }
    let cooled = gw
        .metric("ai_key_auth_failures_total", r#"reason="unfunded""#)
        .await;
    if openrouter.hits() != sent || anthropic.hits() != costly || cooled != 0.0 {
        failures.push(format!(
            "{} hits on OpenRouter (want {sent}), {} on Anthropic (want {costly}), {cooled} keys cooled (want 0)",
            openrouter.hits(),
            anthropic.hits(),
        ));
    }
    if !failures.is_empty() {
        failures.push(gw.log());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Many requests 402 on OpenRouter at once: every one is served by Anthropic, and the key is cooled
/// once, not once per [`OPENROUTER_402_STRIKES`] of the burst (a key already cooling does not
/// count the 402s of requests sent before it cooled).
/// claim: REL-4
/// defect: D261
#[tokio::test]
async fn a_burst_of_openrouter_402s_cools_its_key_once() {
    let (pubkey, sk) = test_keypair(163);
    let key = billing_vkey(&sk, 163);
    let openrouter =
        MockUpstream::start(Mode::Raw(402, "application/json", OPENROUTER_NO_CREDIT)).await;
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = gateway(&openrouter, &anthropic, &pubkey).await;
    let burst = 24;
    let mut tasks = Vec::with_capacity(burst);
    for _ in 0..burst {
        let (url, key) = (gw.url(), key.clone());
        tasks.push(tokio::spawn(async move {
            let resp = test_client()
                .post(format!("{url}/v1/chat/completions"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .header("x-beyond-order", "openrouter,anthropic")
                .body(r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#)
                .send()
                .await
                .unwrap();
            resp.status().as_u16()
        }));
    }
    let mut statuses = Vec::with_capacity(burst);
    for t in tasks {
        statuses.push(t.await.unwrap());
    }
    // Then the key is cooling: this one goes straight to Anthropic.
    let before = openrouter.hits();
    let (status, by, text) = send(&gw, &key, "hi").await;
    let cooled = gw
        .metric("ai_key_auth_failures_total", r#"reason="unfunded""#)
        .await;
    assert!(
        statuses.iter().all(|s| *s == 200)
            && status == 200
            && by.as_deref() == Some("anthropic")
            && openrouter.hits() == before
            && openrouter.hits() >= strikes()
            && cooled == 1.0,
        "burst statuses {statuses:?}; after it {status} by {by:?} ({text}); {} hits on OpenRouter ({before} before the last); {cooled} keys cooled (want 1)\n\n{}",
        openrouter.hits(),
        gw.log(),
    );
}
