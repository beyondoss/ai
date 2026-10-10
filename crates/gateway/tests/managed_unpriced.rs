//! A managed key cannot ask for what a provider bills outside the usage the gateway meters
//! (`unpriced::inspect`): each is a 400 before any upstream sees the request, and writes no billing
//! row. A BYO key sends the same body untouched.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;

const OK: &str = r#"{"id":"resp_1","object":"response","model":"gpt-5","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"#;

async fn send(url: String, auth: (&str, String), body: &str) -> (u16, String) {
    let resp = test_client()
        .post(url)
        .header(auth.0, auth.1)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(body.to_owned())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

/// `(provider, path, body, words the refusal names)`. Every model is a catalog one, so only the
/// feature can be what is refused.
const CASES: [(&str, &str, &str, &str); 14] = [
    (
        "openai",
        "/openai/v1/responses",
        r#"{"model":"gpt-5","input":"hi","tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}"#,
        "code_interpreter",
    ),
    (
        "openai",
        "/openai/v1/responses",
        r#"{"model":"gpt-5","input":"hi","tools":[{"type":"shell"}]}"#,
        "shell",
    ),
    (
        "openai",
        "/openai/v1/responses",
        r#"{"model":"gpt-5","input":"hi","tools":[{"type":"image_generation"}]}"#,
        "image_generation",
    ),
    (
        "openai",
        "/openai/v1/responses",
        r#"{"model":"gpt-4o-mini","input":"hi","tools":[{"type":"web_search"}]}"#,
        "gpt-4o-mini",
    ),
    (
        "xai",
        "/xai/v1/responses",
        r#"{"model":"grok-4.3","input":"hi","tools":[{"type":"image_generation"}]}"#,
        "image_generation",
    ),
    (
        "anthropic",
        "/anthropic/v1/messages",
        r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"code_execution_20250825","name":"code_execution"}]}"#,
        "code execution",
    ),
    (
        "anthropic",
        "/anthropic/v1/messages",
        r#"{"model":"claude-opus-4-8","max_tokens":16,"container":"container_011x","messages":[{"role":"user","content":"hi"}]}"#,
        "container",
    ),
    (
        "anthropic",
        "/anthropic/v1/messages",
        r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"advisor_20260301","name":"advisor"}]}"#,
        "advisor",
    ),
    (
        "anthropic",
        "/anthropic/v1/messages",
        r#"{"model":"claude-opus-4-8","max_tokens":16,"service_tier":"priority","messages":[{"role":"user","content":"hi"}]}"#,
        "Priority Tier",
    ),
    (
        "groq",
        "/groq/openai/v1/chat/completions",
        r#"{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"browser_search"}]}"#,
        "browser_search",
    ),
    (
        "openrouter",
        "/openrouter/api/v1/chat/completions",
        r#"{"model":"anthropic/claude-opus-4.8","plugins":[{"id":"web"}],"messages":[{"role":"user","content":"hi"}]}"#,
        "plugins",
    ),
    (
        "openrouter",
        "/openrouter/api/v1/chat/completions",
        r#"{"model":"anthropic/claude-opus-4.8:online","messages":[{"role":"user","content":"hi"}]}"#,
        ":online",
    ),
    (
        "openrouter",
        "/openrouter/api/v1/chat/completions",
        r#"{"model":"anthropic/claude-opus-4.8","web_search_options":{},"messages":[{"role":"user","content":"hi"}]}"#,
        "web_search_options",
    ),
    (
        "openrouter",
        "/openrouter/api/v1/chat/completions",
        r#"{"model":"anthropic/claude-opus-4.8","tools":[{"type":"openrouter:web_search"}],"messages":[{"role":"user","content":"hi"}]}"#,
        "openrouter:web_search",
    ),
];

/// Every unpriced feature on a managed `/{provider}` route: a 400 naming it, no upstream request,
/// no billing row. The same body on a BYO key reaches the provider.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn a_managed_key_cannot_run_an_unpriced_feature() {
    let (pubkey, sk) = test_keypair(160);
    // Counts only requests whose body arrived whole: what a provider would act on.
    let mock = ScriptedUpstream::reply(200, "application/json", OK.to_owned()).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter", "xai", "groq"])
        .start()
        .await;
    let managed = format!("Bearer {}", billing_vkey(&sk, 160));
    let mut failures = Vec::new();
    for (provider, path, body, names) in CASES {
        let before = mock.hits();
        let (status, text) = send(
            format!("{}{path}", gw.url()),
            ("authorization", managed.clone()),
            body,
        )
        .await;
        if status != 400 || !text.contains(names) || mock.hits() != before {
            failures.push(format!(
                "managed {provider} {names}: {status} {text} (upstream hits {})",
                mock.hits() - before
            ));
        }
        let before = mock.hits();
        let (status, text) = send(
            format!("{}{path}", gw.url()),
            ("authorization", "Bearer sk-caller-own-key".to_owned()),
            body,
        )
        .await;
        if status != 200 || mock.hits() != before + 1 {
            failures.push(format!("BYO {provider} {names}: {status} {text}"));
        }
    }
    let metrics = gw.metrics().await;
    if !metrics.contains(&format!(
        r#"ai_rejections_total{{reason="unpriced_feature"}} {}"#,
        CASES.len()
    )) {
        failures.push("every refusal is counted as unpriced_feature".to_owned());
    }
    assert!(
        usage_rows_of(&gw).is_empty(),
        "a refusal writes no billing row: {:?}",
        usage_rows_of(&gw)
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A catalog walk is checked the same way, on the client's body before translation.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn a_catalog_walk_cannot_run_an_unpriced_feature() {
    let (pubkey, sk) = test_keypair(161);
    // Counts only requests whose body arrived whole: what a provider would act on.
    let mock = ScriptedUpstream::reply(200, "application/json", OK.to_owned()).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic"])
        .start()
        .await;
    let managed = format!("Bearer {}", billing_vkey(&sk, 161));
    for (path, body) in [
        (
            "/v1/responses",
            r#"{"model":"gpt-5","input":"hi","tools":[{"type":"image_generation"}]}"#,
        ),
        (
            "/v1/messages",
            r#"{"model":"claude-opus-4-8","max_tokens":16,"service_tier":"priority","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        (
            "/v1/chat/completions",
            r#"{"model":"claude-opus-4-8","plugins":[{"id":"web"}],"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    ] {
        let (status, text) = send(
            format!("{}{path}", gw.url()),
            ("authorization", managed.clone()),
            body,
        )
        .await;
        assert_eq!(status, 400, "{path}: {text}");
        assert!(text.contains("managed key cannot"), "{path}: {text}");
    }
    assert_eq!(mock.hits(), 0, "nothing reached a provider");
    // A priced tool passes.
    let (status, text) = send(
        format!("{}/v1/responses", gw.url()),
        ("authorization", managed.clone()),
        r#"{"model":"gpt-5","input":"hi","tools":[{"type":"web_search"}]}"#,
    )
    .await;
    assert_eq!(status, 200, "{text}");
}
