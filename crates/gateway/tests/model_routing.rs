//! End-to-end: the model-routed (`/auto/…`) path — catalog resolution, the mount prefix, the
//! per-attempt model rewrite, and connect-level failover.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).
//!
//! The topology throughout is the catalog's `gpt-4o-mini` row: `openai` (mount `/v1`, id
//! `gpt-4o-mini`) then `openrouter` (mount `/api/v1`, id `openai/gpt-4o-mini`). Both differ in every
//! dimension the route has to get right, which is why that row was seeded.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code — allow the
// panic-surface restriction lints denied workspace-wide in `[workspace.lints.clippy]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const MODEL: &str = "gpt-4o-mini";

fn body() -> String {
    format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}]}}"#)
}

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
        },
        1,
        sk,
    )
}

/// POST to the model route. `model` goes in the routing header; the body carries it too, exactly as
/// a stock SDK would send it.
async fn post_auto(
    client: &reqwest::Client,
    url: &str,
    key: &str,
    model: Option<&str>,
) -> reqwest::Response {
    let mut req = client
        .post(format!("{url}/auto/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json");
    if let Some(m) = model {
        req = req.header("x-beyond-model", m);
    }
    req.body(body()).send().await.unwrap()
}

/// The primary candidate serves it: OpenAI's mount, OpenAI's pool key, OpenAI's spelling of the id.
#[tokio::test]
async fn routes_by_model_header_to_the_primary_candidate() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let client = reqwest::Client::new();
    let resp = post_auto(&client, &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(resp.status().as_u16(), 200);

    let cap = primary.captured().expect("primary served the request");
    // OpenAI's mount (`/v1`) prepended to the client's suffix. The client never said `/v1`.
    assert_eq!(cap.path, "/v1/chat/completions");
    // The pool key is per-provider in the harness, so this proves *which* provider's key was used.
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));
    // The routing header is ours and must not reach a provider.
    assert_eq!(cap.beyond_model, None);
    // Primary spells it the same as the catalog, so the body is unchanged.
    let body = String::from_utf8(cap.body).unwrap();
    assert!(
        body.contains(r#""model":"gpt-4o-mini""#),
        "primary must receive its own id: {body}"
    );
    assert_eq!(fallback.hits(), 0, "the fallback must not be touched");
}

/// The headline behaviour: primary refuses the connection, and the request still succeeds — served
/// by the fallback, under the fallback's mount, key, and id.
#[tokio::test]
async fn fails_over_to_the_next_candidate_when_the_primary_wont_connect() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let client = reqwest::Client::new();
    let resp = post_auto(&client, &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "a dead primary must be invisible to the client",
    );

    let cap = fallback.captured().expect("fallback served the request");
    // OpenRouter's mount, not OpenAI's — the path is rebuilt for the candidate that serves.
    assert_eq!(cap.path, "/api/v1/chat/completions");
    // The single most important assertion in the feature: the key swap followed the candidate.
    // Forwarding OpenAI's pool key to OpenRouter would be a credential leak, not a failed request.
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-openrouter-pool"),
    );
    assert_eq!(cap.beyond_model, None);
    // And the model was re-spelled the way OpenRouter names it.
    let body = String::from_utf8(cap.body).unwrap();
    assert!(
        body.contains(r#""model":"openai/gpt-4o-mini""#),
        "the fallback must be asked for its own id, not the catalog name: {body}"
    );

    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_candidate_failovers_total", "") >= 1.0,
        "the failover must be visible on its own counter:\n{metrics}"
    );
}

/// The ledger test. If the abandoned candidate's failure were not recorded, its breaker would never
/// open; if the serving candidate's success were recorded against it instead, likewise. One pair of
/// assertions catches both.
#[tokio::test]
async fn records_the_failed_candidates_breaker_not_the_serving_ones() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let client = reqwest::Client::new();
    let key = vkey(&sk);
    // Enough attempts to trip the dead primary's breaker several times over.
    for _ in 0..6 {
        let resp = post_auto(&client, &gw.url(), &key, Some(MODEL)).await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "the fallback keeps serving while the primary's breaker opens",
        );
    }

    let metrics = gw.metrics().await;
    // The primary's breaker opened: once open, its candidate is skipped without an attempt, which
    // is counted as a `circuit_open` rejection even though the request itself succeeded.
    assert!(
        parse_metric(&metrics, "ai_rejections_total", "circuit_open") >= 1.0,
        "the dead primary's breaker must open from its own recorded failures:\n{metrics}"
    );
    // ...and the fallback served every one of them.
    assert_eq!(fallback.hits(), 6, "every request reached the fallback");
}

/// The routing header is required, and only names the catalog serves are routable.
#[tokio::test]
async fn a_missing_or_unknown_model_is_rejected_before_any_upstream() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let client = reqwest::Client::new();
    let key = vkey(&sk);

    let no_header = post_auto(&client, &gw.url(), &key, None).await;
    assert_eq!(no_header.status().as_u16(), 404);
    let unknown = post_auto(&client, &gw.url(), &key, Some("no-such-model")).await;
    assert_eq!(unknown.status().as_u16(), 404);

    assert_eq!(mock.hits(), 0, "neither request may reach an upstream");
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_rejections_total", "unknown_model") >= 2.0,
        "both rejections must be counted:\n{metrics}"
    );
}

/// Model routing is managed-only. A BYO token belongs to one provider, so choosing among candidates
/// would be a guess and failing over would hand one vendor's key to another.
#[tokio::test]
async fn a_byo_key_is_refused_with_400() {
    let nats = Nats::start().await;
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let client = reqwest::Client::new();
    let resp = post_auto(
        &client,
        &gw.url(),
        "sk-someones-own-openai-key",
        Some(MODEL),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 400);
    assert_eq!(
        mock.hits(),
        0,
        "the BYO token must never leave the gateway on this route",
    );
}

/// The billing row names the provider that actually served, and carries the catalog name routed on.
#[tokio::test]
async fn the_usage_row_names_the_candidate_that_served() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let client = reqwest::Client::new();
    let resp = post_auto(&client, &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(resp.status().as_u16(), 200);

    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"openrouter""#])
        .await;
    assert!(
        line.contains(r#""routed_model":"gpt-4o-mini""#),
        "the row must record the catalog name routed on: {line}"
    );
    assert!(
        line.contains(r#""tenant_id":42"#),
        "the row must still attribute the tenant: {line}"
    );
}

/// Every candidate down ⇒ a clean failure, and both candidates were genuinely tried.
#[tokio::test]
async fn every_candidate_down_fails_the_request() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let gw = Gateway::builder(nats.port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;

    let client = reqwest::Client::new();
    let resp = post_auto(&client, &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert!(
        resp.status().is_server_error(),
        "want a 5xx when nothing can serve, got {}",
        resp.status(),
    );

    let metrics = gw.metrics().await;
    for provider in ["openai", "openrouter"] {
        assert!(
            parse_metric(&metrics, "ai_connect_retries_total", provider) >= 1.0,
            "{provider} must have been attempted before giving up:\n{metrics}"
        );
    }
}

/// A body larger than pingora's 64 KiB replay buffer must survive a failover intact.
///
/// This is the case the retry machinery is least able to help with: past that cap pingora cannot
/// replay the body at all. It works only because a *connect* failure happens before any body byte is
/// read from the client, so the retry re-reads from the socket rather than from the buffer.
#[tokio::test]
async fn a_large_body_survives_a_failover_intact() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    // ~256 KiB of message content — comfortably past BODY_BUF_LIMIT.
    let filler = "x".repeat(256 * 1024);
    let big =
        format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{filler}"}}]}}"#);
    let sent = big.len();

    let resp = reqwest::Client::new()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .body(big)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let cap = fallback.captured().expect("fallback served the request");
    let received = String::from_utf8(cap.body).unwrap();
    // The rewrite lengthens the id by exactly the vendor prefix; nothing else may change.
    assert_eq!(
        received.len(),
        sent + "openai/".len(),
        "the body must arrive whole, differing only by the rewritten model id",
    );
    assert!(received.contains(r#""model":"openai/gpt-4o-mini""#));
    assert!(
        received.contains(&filler),
        "the message content must survive the failover byte-for-byte",
    );
}

/// Provider-routed traffic is untouched by any of this: same path, same key, same body, and no
/// routing header involved.
#[tokio::test]
async fn provider_routed_requests_are_unaffected() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = reqwest::Client::new()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let cap = mock.captured().expect("upstream served the request");
    // Forwarded verbatim — no mount logic, because the client named the provider itself.
    assert_eq!(cap.path, "/v1/chat/completions");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));
    let body = String::from_utf8(cap.body).unwrap();
    assert!(
        body.contains(r#""model":"gpt-4o-mini""#),
        "a provider-routed body must not be rewritten: {body}"
    );
}

/// Claude fails over Anthropic → OpenRouter **on the Anthropic wire**, and the usage is parsed with
/// the Anthropic extractor even though OpenRouter is an OpenAI-wire provider in the provider table.
///
/// On the metering half, be clear about what this test can and cannot prove. `rc.dialect` now comes
/// from the row's `wire`; it used to come from `provider.dialect`. For *this* row those agree, since
/// the primary candidate is Anthropic and Anthropic is an Anthropic-wire provider — so reverting the
/// fix leaves this test green (checked, not assumed). What it does prove is that the whole
/// Anthropic-wire path — route, rewrite, relay, extract — meters end to end.
///
/// The case the fix actually guards is a row whose *primary* is an OpenAI-wire provider serving the
/// Anthropic wire (Fireworks does exactly this for its own models). There, `provider.dialect` says
/// `OpenAi`, the Anthropic response hits the dialect-mismatch guard, and the row bills **zero
/// tokens** without erroring. No such row is in the catalog yet, so the derivation is pinned as a
/// unit test in `proxy.rs` instead of contriving one here.
#[tokio::test]
async fn claude_fails_over_on_the_anthropic_wire_and_is_still_metered() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    // The fallback answers in Anthropic shape: `usage.input_tokens`, not `usage.prompt_tokens`.
    let fallback = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats.port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = reqwest::Client::new()
        .post(format!("{}/auto/v1/messages", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", "claude-opus-4-8")
        .body(r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "Anthropic is dead; OpenRouter must serve it"
    );

    let cap = fallback
        .captured()
        .expect("the fallback served the request");
    // OpenRouter's Messages endpoint, which is not reachable by composing Anthropic's `/v1/messages`
    // with any per-provider mount — the catalog states it outright.
    assert_eq!(cap.path, "/api/v1/messages");
    // Auth followed the candidate, and so did its *scheme*: Anthropic wants `x-api-key`, OpenRouter
    // wants Bearer. Sending Anthropic's scheme to OpenRouter would 401.
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-openrouter-pool"),
    );
    assert_eq!(
        cap.x_api_key, None,
        "the Anthropic scheme must not leak to OpenRouter"
    );
    // And the model was re-spelled the way OpenRouter names it — dots, vendor-prefixed.
    let body = String::from_utf8(cap.body).unwrap();
    assert!(
        body.contains(r#""model":"anthropic/claude-opus-4.8""#),
        "the fallback must be asked for its own id: {body}"
    );

    // The billing row must carry real tokens. Zero here means the OpenAI extractor ran against an
    // Anthropic body and the dialect-mismatch guard swallowed it.
    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"openrouter""#])
        .await;
    assert!(
        line.contains(r#""input_tokens":13"#),
        "usage must be parsed with the row's Anthropic dialect, not the provider's OpenAI one: {line}"
    );
    assert!(
        line.contains(r#""routed_model":"claude-opus-4-8""#),
        "{line}"
    );

    let metrics = gw.metrics().await;
    assert_eq!(
        parse_metric(&metrics, "ai_usage_parse_errors_total", ""),
        0.0,
        "no usage should have failed to parse:\n{metrics}"
    );
}

/// `requested_model` on `/auto` is the catalog name from the routing header — not the body's
/// `model`, which the gateway overwrites and which therefore determines nothing.
///
/// The body here deliberately names a *different* model. It runs nothing (the row's primary serves,
/// under the row's id), so reporting it as what the client "requested" would be reporting a
/// discarded input. The disagreement is counted so a client bug is visible.
#[tokio::test]
async fn requested_model_is_the_routed_name_not_the_discarded_body_value() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = reqwest::Client::new()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        // A body naming something else entirely.
        .body(r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    // The body's value never reached the provider: it was overwritten with the row's id.
    let cap = primary.captured().expect("primary served the request");
    let body = String::from_utf8(cap.body).unwrap();
    assert!(
        body.contains(r#""model":"gpt-4o-mini""#) && !body.contains("claude"),
        "the body's model must be replaced by the routed candidate's id: {body}"
    );

    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"openai""#])
        .await;
    assert!(
        line.contains(r#""requested_model":"gpt-4o-mini""#),
        "requested_model must be the routed catalog name: {line}"
    );
    assert!(
        !line.contains("claude"),
        "the discarded body value must not appear anywhere in the billing row: {line}"
    );

    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_model_header_body_mismatch_total", "") >= 1.0,
        "the disagreement must be counted so a client bug is findable:\n{metrics}"
    );
}

/// The ordinary case: header and body agree, and nothing is counted as a mismatch.
#[tokio::test]
async fn agreeing_header_and_body_count_no_mismatch() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = post_auto(&reqwest::Client::new(), &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(resp.status().as_u16(), 200);

    let metrics = gw.metrics().await;
    assert_eq!(
        parse_metric(&metrics, "ai_model_header_body_mismatch_total", ""),
        0.0,
        "a well-formed request must not be counted as a mismatch:\n{metrics}"
    );
}

/// Provider-routed traffic keeps the old meaning: the body is untouched, so the body's `model` *is*
/// what was requested, and no `routed_model` appears at all.
#[tokio::test]
async fn provider_routed_requested_model_still_comes_from_the_body() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = reqwest::Client::new()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(r#"{"model":"some-other-model","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"openai""#])
        .await;
    assert!(
        line.contains(r#""requested_model":"some-other-model""#),
        "a provider-routed body is untouched, so it is what was requested: {line}"
    );
    assert!(
        !line.contains("routed_model"),
        "routed_model marks the model route and must be absent otherwise: {line}"
    );
}
