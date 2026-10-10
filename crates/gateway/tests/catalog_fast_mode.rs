//! End-to-end: Anthropic fast mode on managed traffic (D266).
//!
//! - A Messages client's `speed: "fast"` and its `fast-mode-2026-02-01` beta reach direct
//!   Anthropic for a model with fast mode (Opus 5.5, Opus 5, Opus 4.8), on a catalog walk and on
//!   `/anthropic/…`.
//! - Fast or refused, as Anthropic itself answers it: such a request is never sent to a candidate
//!   without fast mode (Bedrock, OpenRouter), not even as a failover, and a walk with no candidate
//!   that has it is a 400 before any upstream sees it.
//! - A Chat Completions or Responses client's `service_tier: "priority"` (OpenAI's priority
//!   processing, its own fast mode) becomes `speed: "fast"` plus the beta on a direct Anthropic
//!   attempt for such a model, and is a best effort everywhere else: served at standard speed,
//!   with neither field.
//!
//! Asserted at the upstream (what each mock received). Run via `mise run test:integration:rs`
//! (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::{Value, json};

const FAST_BETA: &str = "fast-mode-2026-02-01";

fn provider_of(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

async fn post(
    gw: &Gateway,
    key: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &Value,
) -> reqwest::Response {
    let mut req = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_string());
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    req.send().await.unwrap()
}

fn messages(model: &str) -> Value {
    json!({"model": model, "max_tokens": 16, "speed": "fast",
           "messages": [{"role": "user", "content": "hi"}]})
}

fn body_of(cap: &Captured) -> Value {
    serde_json::from_slice(&cap.body).unwrap()
}

fn has_fast_beta(cap: &Captured) -> bool {
    cap.anthropic_beta
        .as_deref()
        .is_some_and(|v| v.split(',').any(|t| t.trim() == FAST_BETA))
}

/// A Messages client asking fast mode of a model that has it gets its `speed` and its beta to
/// direct Anthropic: headerless, header-won (the row is one whose body the walk reads), and
/// pinned to `/anthropic/…`, where the beta used to be dropped by the managed allowlist.
/// claim: SEC-6, CAT-6
/// defect: D266
#[tokio::test]
async fn fast_mode_reaches_direct_anthropic_with_its_beta() {
    let (pubkey, sk) = test_keypair(81);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .start()
        .await;
    let key = billing_vkey(&sk, 81);
    let beta = (
        "anthropic-beta",
        "prompt-caching-2024-07-31,fast-mode-2026-02-01",
    );
    for (path, model, header_won) in [
        ("/v1/messages", "claude-opus-5-5", false),
        ("/v1/messages", "claude-opus-5", true),
        ("/v1/messages", "claude-opus-4-8", false),
        ("/anthropic/v1/messages", "claude-opus-5-5", false),
    ] {
        let mut headers = vec![beta];
        if header_won {
            headers.push(("x-beyond-model", model));
        }
        let resp = post(&gw, &key, path, &headers, &messages(model)).await;
        assert_eq!(resp.status().as_u16(), 200, "{path} {model}");
        let cap = mock.captured().unwrap();
        assert_eq!(cap.path, "/v1/messages", "{path} {model}");
        assert_eq!(
            cap.anthropic_beta.as_deref(),
            Some("prompt-caching-2024-07-31,fast-mode-2026-02-01"),
            "{path} {model}"
        );
        assert_eq!(body_of(&cap)["speed"], "fast", "{path} {model}");
    }
}

/// Fast mode is fast or an error: a fast request on a row with Bedrock and OpenRouter candidates
/// goes to direct Anthropic whatever the caller's order, and does not fail over when Anthropic
/// fails, where a standard request does. Restricted to candidates without fast mode, it is a 400
/// that no upstream sees.
/// claim: CAT-6
/// defect: D266
#[tokio::test]
async fn a_fast_request_is_never_served_without_fast_mode() {
    let (pubkey, sk) = test_keypair(82);
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let bedrock = MockUpstream::start(Mode::AnthropicJson).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "bedrock", "openrouter"])
        .provider_authority("bedrock", &bedrock.authority())
        .provider_authority("openrouter", &openrouter.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 82);
    let fast = messages("claude-opus-4-8");

    let order = [("x-beyond-order", "bedrock,openrouter")];
    let resp = post(&gw, &key, "/v1/messages", &order, &fast).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(provider_of(&resp).as_deref(), Some("anthropic"));
    assert_eq!(bedrock.hits() + openrouter.hits(), 0);

    let only = [("x-beyond-only", "bedrock,openrouter")];
    let resp = post(&gw, &key, "/v1/messages", &only, &fast).await;
    assert_eq!(resp.status().as_u16(), 400);
    let err: Value = resp.json().await.unwrap();
    assert_eq!(err["error"]["type"], "invalid_request_error");
    assert_eq!(
        err["error"]["message"],
        "claude-opus-4-8 does not accept fast mode"
    );
    assert_eq!(bedrock.hits() + openrouter.hits(), 0);

    // Anthropic failing: the standard request fails over, the fast one gets Anthropic's answer.
    let failing = MockUpstream::start(Mode::AnthropicStatus(500)).await;
    let gw = Gateway::builder(unused_nats_port(), &failing.authority(), &b64(&pubkey))
        .providers(&["anthropic", "bedrock", "openrouter"])
        .provider_authority("bedrock", &bedrock.authority())
        .provider_authority("openrouter", &openrouter.authority())
        .start()
        .await;
    let resp = post(&gw, &key, "/v1/messages", &[], &fast).await;
    assert_ne!(resp.status().as_u16(), 200);
    assert!(failing.hits() >= 1);
    assert_eq!(
        bedrock.hits() + openrouter.hits(),
        0,
        "a fast request failed over"
    );
    let mut standard = fast.clone();
    standard.as_object_mut().unwrap().remove("speed");
    let resp = post(&gw, &key, "/v1/messages", &[], &standard).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(provider_of(&resp).as_deref(), Some("bedrock"));
}

/// A model without fast mode is a 400 for a fast request, before any upstream: Opus 4.7 (which
/// Anthropic answers with an error), Opus 4.6 (which Anthropic would serve at standard speed),
/// Sonnet 5.5, and a GPT row, whose priority processing a Messages client does not ask for with
/// `speed`. `speed: "standard"` is no ask at all.
/// claim: CAT-6
/// defect: D266
#[tokio::test]
async fn fast_mode_on_a_model_without_it_is_a_400() {
    let (pubkey, sk) = test_keypair(83);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;
    let key = billing_vkey(&sk, 83);
    for model in [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5-5",
        "gpt-5",
    ] {
        let resp = post(&gw, &key, "/v1/messages", &[], &messages(model)).await;
        assert_eq!(resp.status().as_u16(), 400, "{model}");
        let err: Value = resp.json().await.unwrap();
        assert_eq!(
            err["error"]["message"],
            format!("{model} does not accept fast mode")
        );
    }
    assert_eq!(mock.hits(), 0);

    let mut standard = messages("claude-sonnet-5-5");
    standard["speed"] = json!("standard");
    let resp = post(&gw, &key, "/v1/messages", &[], &standard).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(body_of(&mock.captured().unwrap())["speed"], "standard");
}

/// OpenAI's `service_tier: "priority"` from a Chat Completions or Responses client becomes fast
/// mode on a direct Anthropic attempt for a model that has it: `speed: "fast"` in the body, the
/// beta on the headers, no `service_tier`. Header-won walks too, since the row's body is read.
/// claim: CAT-6
/// defect: D266
#[tokio::test]
async fn priority_becomes_fast_mode_on_direct_anthropic() {
    let (pubkey, sk) = test_keypair(84);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .start()
        .await;
    let key = billing_vkey(&sk, 84);
    let chat = json!({"model": "claude-opus-5-5", "service_tier": "priority",
                      "messages": [{"role": "user", "content": "hi"}]});
    let responses = json!({"model": "claude-opus-5-5", "service_tier": "priority",
                           "input": "hi"});
    let header_won = [("x-beyond-model", "claude-opus-5-5")];
    for (path, headers, body) in [
        ("/v1/chat/completions", &[][..], &chat),
        ("/v1/chat/completions", &header_won[..], &chat),
        ("/v1/responses", &[][..], &responses),
    ] {
        let resp = post(&gw, &key, path, headers, body).await;
        assert_eq!(resp.status().as_u16(), 200, "{path}");
        let cap = mock.captured().unwrap();
        assert_eq!(cap.path, "/v1/messages", "{path}");
        assert!(has_fast_beta(&cap), "{path}: {:?}", cap.anthropic_beta);
        let sent = body_of(&cap);
        assert_eq!(sent["speed"], "fast", "{path}");
        assert!(sent.get("service_tier").is_none(), "{path}");
    }
}

/// Priority is a best effort: on a model without fast mode, any other tier, or a Bedrock attempt,
/// the request is served at standard speed with neither `speed` nor the beta.
/// claim: CAT-6
/// defect: D266
#[tokio::test]
async fn priority_is_served_at_standard_speed_where_fast_mode_is_not() {
    let (pubkey, sk) = test_keypair(85);
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let bedrock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "bedrock", "openrouter"])
        .provider_authority("bedrock", &bedrock.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 85);
    let chat = |model: &str, tier: &str| {
        json!({"model": model, "service_tier": tier,
               "messages": [{"role": "user", "content": "hi"}]})
    };
    for body in [
        chat("claude-sonnet-5-5", "priority"),
        chat("claude-opus-5-5", "default"),
        chat("claude-opus-5-5", "auto"),
    ] {
        let resp = post(&gw, &key, "/v1/chat/completions", &[], &body).await;
        assert_eq!(resp.status().as_u16(), 200, "{body}");
        let cap = anthropic.captured().unwrap();
        assert!(!has_fast_beta(&cap), "{body}");
        let sent = body_of(&cap);
        assert!(sent.get("speed").is_none(), "{body}");
        assert!(sent.get("service_tier").is_none(), "{body}");
    }

    let order = [("x-beyond-order", "bedrock")];
    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &order,
        &chat("claude-opus-4-8", "priority"),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(provider_of(&resp).as_deref(), Some("bedrock"));
    let cap = bedrock.captured().unwrap();
    assert!(!has_fast_beta(&cap));
    assert!(body_of(&cap).get("speed").is_none());
}
