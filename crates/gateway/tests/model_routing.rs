//! End-to-end: the model-routed path (`/auto/…` and managed `/v1`) — catalog resolution, the
//! mount prefix, the per-attempt model rewrite, and connect-level failover.
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
            key_id: None,
        },
        1,
        sk,
    )
}

/// POST to the model route. `model` goes in the routing header when given; the body always carries
/// it too, the way a stock SDK would.
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

/// Stock OpenAI SDK shape: `POST /v1/chat/completions` with `model` only in the body.
async fn post_v1(
    client: &reqwest::Client,
    url: &str,
    key: &str,
    body: String,
) -> reqwest::Response {
    client
        .post(format!("{url}/v1/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap()
}

/// Two-mock topology used by the catalog-walk tests (openai primary, openrouter fallback).
async fn catalog_gateway(nats_port: u16, pubkey: &str, primary: &str, fallback: &str) -> Gateway {
    Gateway::builder(nats_port, primary, pubkey)
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", fallback)
        .start()
        .await
}

/// The primary candidate serves it: OpenAI's mount, OpenAI's pool key, OpenAI's spelling of the id.
#[tokio::test]
async fn routes_by_model_header_to_the_primary_candidate() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let client = test_client();
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let client = test_client();
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let client = test_client();
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

/// The catalog is the allowlist. An unknown routing header, a missing body `model`, or a body
/// naming something we do not serve, all 404 before any upstream — on `/auto` and on managed `/v1`.
#[tokio::test]
async fn a_missing_or_unknown_model_is_rejected_before_any_upstream() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let client = test_client();
    let key = vkey(&sk);

    let unknown_header = post_auto(&client, &gw.url(), &key, Some("no-such-model")).await;
    assert_eq!(unknown_header.status().as_u16(), 404);
    let unknown_header_body = unknown_header.text().await.unwrap();
    assert!(
        unknown_header_body.contains("no-such-model")
            && unknown_header_body.contains("not in the catalog"),
        "{unknown_header_body}"
    );

    let unknown_body = post_v1(
        &client,
        &gw.url(),
        &key,
        r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#.into(),
    )
    .await;
    assert_eq!(unknown_body.status().as_u16(), 404);
    let unknown_body_text = unknown_body.text().await.unwrap();
    assert!(
        unknown_body_text.contains("no-such-model")
            && unknown_body_text.contains("not in the catalog"),
        "{unknown_body_text}"
    );

    let missing_body = post_v1(
        &client,
        &gw.url(),
        &key,
        r#"{"messages":[{"role":"user","content":"hi"}]}"#.into(),
    )
    .await;
    assert_eq!(missing_body.status().as_u16(), 404);
    let missing_text = missing_body.text().await.unwrap();
    assert!(missing_text.contains("missing model"), "{missing_text}");

    let unknown_auto_body = client
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(unknown_auto_body.status().as_u16(), 404);

    assert_eq!(mock.hits(), 0, "no request may reach an upstream");
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_rejections_total", "unknown_model") >= 4.0,
        "all four rejections must be counted:\n{metrics}"
    );
}

/// Headerless `/auto` is the same walk as managed `/v1`: the body's `model` selects the row.
#[tokio::test]
async fn auto_without_header_routes_from_the_body_model() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = catalog_gateway(
        nats_port,
        &b64(&pubkey),
        &primary.authority(),
        &fallback.authority(),
    )
    .await;

    let resp = post_auto(&test_client(), &gw.url(), &vkey(&sk), None).await;
    assert_eq!(resp.status().as_u16(), 200);

    let cap = primary.captured().expect("primary served the request");
    assert_eq!(cap.path, "/v1/chat/completions");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));
    assert_eq!(fallback.hits(), 0, "the fallback must not be touched");
}

/// Model routing is managed-only. A BYO token belongs to one provider, so choosing among candidates
/// would be a guess and failing over would hand one vendor's key to another.
#[tokio::test]
async fn a_byo_key_is_refused_with_400() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let client = test_client();
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let client = test_client();
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;

    let client = test_client();
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    // ~256 KiB of message content — comfortably past BODY_BUF_LIMIT.
    let filler = "x".repeat(256 * 1024);
    let big =
        format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{filler}"}}]}}"#);
    let sent = big.len();

    let resp = test_client()
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    // The fallback answers in Anthropic shape: `usage.input_tokens`, not `usage.prompt_tokens`.
    let fallback = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = test_client()
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = post_auto(&test_client(), &gw.url(), &vkey(&sk), Some(MODEL)).await;
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
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
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

/// **Status-based failover.** The primary answers `500`; the client still gets a `200` from the
/// fallback, and never sees the error.
///
/// This is the outage that actually happens — a provider that is up and failing, not one that
/// refuses connections — so it is the case the whole feature exists for.
#[tokio::test]
async fn fails_over_when_the_primary_answers_5xx() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = post_auto(&test_client(), &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "a 5xx from the primary must be invisible to the client",
    );

    assert_eq!(primary.hits(), 1, "the primary was asked once");
    let cap = fallback
        .captured()
        .expect("the fallback served the request");
    assert_eq!(cap.path, "/api/v1/chat/completions");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-openrouter-pool"),
        "the key swap must follow the candidate on a status failover too",
    );
    let body = String::from_utf8(cap.body).unwrap();
    assert!(body.contains(r#""model":"openai/gpt-4o-mini""#), "{body}");

    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_candidate_failovers_total", "") >= 1.0,
        "the failover must be counted:\n{metrics}"
    );
}

/// A `429` is a healthy provider throttling *that credential*, not a broken vendor. With one key
/// it is relayed, not failed over — re-asking a different vendor turns a self-healing throttle
/// into spend somewhere else. (Two keys walk on the same provider; see `a_429_walks_keys_not_vendors`.)
#[tokio::test]
async fn a_429_is_relayed_not_failed_over() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(429)).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = post_auto(&test_client(), &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(
        resp.status().as_u16(),
        429,
        "the throttle must reach the client"
    );
    assert_eq!(
        fallback.hits(),
        0,
        "a throttle must not spend at another vendor"
    );
    assert_eq!(
        parse_metric(&gw.metrics().await, "ai_candidate_failovers_total", ""),
        0.0,
        "a 429 must not count as a candidate failover"
    );
}

/// Two keys on the primary: a 429 walks the next key on the *same* provider. The fallback vendor
/// is not asked, and `ai_candidate_failovers_total` stays at zero.
#[tokio::test]
async fn a_429_walks_keys_not_vendors() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::ThrottleKey("sk-walk-a")).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .pool_keys("openai", &["sk-walk-a", "sk-walk-b"])
        .start()
        .await;

    let resp = post_auto(&test_client(), &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "the second key on the primary must serve"
    );
    assert_eq!(
        fallback.hits(),
        0,
        "a throttle must not spend at another vendor"
    );
    let cap = primary
        .captured()
        .expect("the primary served with the second key");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-walk-b"),
        "the retry must present the second secret on the same provider"
    );
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_key_walks_total", "") >= 1.0,
        "the key-walk must be counted:\n{metrics}"
    );
    assert_eq!(
        parse_metric(&metrics, "ai_candidate_failovers_total", ""),
        0.0,
        "walking keys is not a candidate failover:\n{metrics}"
    );
}

/// When every candidate 5xxes, the client gets the last provider's *actual* error rather than a
/// synthetic one — better diagnostics than an exhausted retry loop produces.
#[tokio::test]
async fn every_candidate_5xx_relays_the_last_error() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(503)).await;
    let fallback = MockUpstream::start(Mode::Status(503)).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = post_auto(&test_client(), &gw.url(), &vkey(&sk), Some(MODEL)).await;
    assert_eq!(
        resp.status().as_u16(),
        503,
        "the last provider's own status must reach the client, not a synthetic 502",
    );
    assert_eq!(primary.hits(), 1);
    assert_eq!(fallback.hits(), 1, "both candidates must have been tried");
}

/// A body past pingora's 64 KiB replay buffer is not provably replayable, so the 5xx is relayed
/// rather than retried — and the case is **counted**, which is the number that decides whether
/// covering it is worth the work.
///
/// Note what this does *not* claim. Retrying such a body is not unsafe: pingora would replay the
/// buffered prefix and read the remainder from the socket. The rule exists so the decision is
/// deterministic rather than a race against how fast the upstream rejected the request — an earlier
/// version of this test passed or failed depending on whether the 256 KiB body had finished
/// arriving, because both outcomes were correct behaviour under a looser gate.
#[tokio::test]
async fn an_unreplayable_body_relays_the_5xx_and_is_counted() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    // 256 KiB — comfortably past BODY_BUF_LIMIT.
    let filler = "x".repeat(256 * 1024);
    let big =
        format!(r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{filler}"}}]}}"#);
    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .body(big)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status().as_u16(),
        500,
        "an unreplayable body must relay the error, not attempt a retry it cannot complete",
    );
    assert_eq!(
        fallback.hits(),
        0,
        "the fallback must not be sent headers for a body we cannot resend",
    );
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_failover_unreplayable_total", "") >= 1.0,
        "the uncovered case must be measurable — it is what decides the next investment:\n{metrics}"
    );
}

/// The ledger holds on the **status** failover path too: a candidate that answers 5xx has that
/// failure recorded against its own breaker, while the candidate that served does not.
///
/// The connect-path equivalent is `records_the_failed_candidates_breaker_not_the_serving_ones`.
/// This one matters separately because the two paths resolve the permit through different hooks —
/// `fail_to_connect` versus `upstream_response_filter` — and only `upstream_peer`'s prologue ties
/// them together.
#[tokio::test]
async fn a_5xx_candidates_breaker_opens_while_the_fallback_keeps_serving() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .circuit_breaker_threshold(2)
        .start()
        .await;

    let client = test_client();
    let key = vkey(&sk);
    for i in 0..6 {
        let resp = post_auto(&client, &gw.url(), &key, Some(MODEL)).await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "request {i}: the fallback must keep serving while the primary's breaker opens",
        );
    }

    let metrics = gw.metrics().await;
    // Once the primary's breaker opens, its candidate is skipped without an attempt — counted as a
    // `circuit_open` rejection even though every client request succeeded. If the 5xx failures were
    // not recorded against the primary, this would stay zero; if the fallback's successes were
    // recorded against the primary instead, likewise.
    assert!(
        parse_metric(&metrics, "ai_rejections_total", "circuit_open") >= 1.0,
        "the 5xx candidate's breaker must open from its own recorded failures:\n{metrics}"
    );
    assert_eq!(fallback.hits(), 6, "every request reached the fallback");
    // ...and once the breaker is open the primary stops being asked at all, so it saw fewer than
    // one request per client request.
    assert!(
        primary.hits() < 6,
        "an open breaker must stop the primary being tried; it saw {} of 6",
        primary.hits(),
    );
}

/// Stock OpenAI SDK: `POST /v1/chat/completions` with only `model` in the body hits the primary.
#[tokio::test]
async fn v1_body_model_routes_to_the_primary_candidate() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = catalog_gateway(
        nats_port,
        &b64(&pubkey),
        &primary.authority(),
        &fallback.authority(),
    )
    .await;

    let resp = post_v1(&test_client(), &gw.url(), &vkey(&sk), body()).await;
    assert_eq!(resp.status().as_u16(), 200);

    let cap = primary.captured().expect("primary served the request");
    assert_eq!(cap.path, "/v1/chat/completions");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));
    assert_eq!(cap.beyond_model, None);
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"gpt-4o-mini""#),
        "primary must receive its own id: {got}"
    );
    assert_eq!(fallback.hits(), 0, "the fallback must not be touched");
}

/// Same stock `/v1` shape, dead primary: failover rewrites path, key, and model id.
#[tokio::test]
async fn v1_body_model_fails_over_when_the_primary_wont_connect() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = catalog_gateway(
        nats_port,
        &b64(&pubkey),
        &GatewayBuilder::dead_authority(),
        &fallback.authority(),
    )
    .await;

    let resp = post_v1(&test_client(), &gw.url(), &vkey(&sk), body()).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "a dead primary must be invisible to the client",
    );

    let cap = fallback.captured().expect("fallback served the request");
    assert_eq!(cap.path, "/api/v1/chat/completions");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-openrouter-pool"),
    );
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"openai/gpt-4o-mini""#),
        "the fallback must be asked for its own id: {got}"
    );
    assert!(
        parse_metric(&gw.metrics().await, "ai_candidate_failovers_total", "") >= 1.0,
        "the failover must be counted"
    );
}

/// Stock Anthropic SDK: `POST /v1/messages` with `model` in the body and `x-api-key`.
#[tokio::test]
async fn v1_messages_body_model_fails_over_on_the_anthropic_wire() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("content-type", "application/json")
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
    assert_eq!(cap.path, "/api/v1/messages");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-openrouter-pool"),
    );
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"anthropic/claude-opus-4.8""#),
        "the fallback must be asked for its own id: {got}"
    );
}

/// Header vs body on `/v1`: header wins, disagreement is counted, body is overwritten.
#[tokio::test]
async fn v1_header_wins_over_a_disagreeing_body() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .body(r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let cap = primary.captured().expect("primary served the request");
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"gpt-4o-mini""#) && !got.contains("claude"),
        "the header's row must overwrite the body's model: {got}"
    );
    assert!(
        parse_metric(
            &gw.metrics().await,
            "ai_model_header_body_mismatch_total",
            ""
        ) >= 1.0,
        "the disagreement must be counted"
    );
}

/// BYO on `/v1` is dialect-default passthrough: an unknown catalog name is forwarded, not 404'd.
#[tokio::test]
async fn byo_v1_forwards_an_unknown_model() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = post_v1(
        &test_client(),
        &gw.url(),
        "sk-someones-own-openai-key",
        r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#.into(),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let cap = mock
        .captured()
        .expect("BYO /v1 must reach the dialect default");
    assert_eq!(cap.path, "/v1/chat/completions");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-someones-own-openai-key"),
    );
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"no-such-model""#),
        "BYO /v1 must not rewrite or reject an unknown model: {got}"
    );
}

/// `/{provider}/…` is the escape hatch: the catalog is not consulted.
#[tokio::test]
async fn explicit_provider_path_ignores_the_catalog() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let cap = mock
        .captured()
        .expect("escape hatch must reach the provider");
    assert_eq!(cap.path, "/v1/chat/completions");
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"no-such-model""#),
        "/openai/… must not apply the catalog allowlist: {got}"
    );
}

/// A Claude model posted to Chat Completions is a 400 from us, not a provider 400 on the wrong JSON.
#[tokio::test]
async fn a_wire_mismatch_is_rejected_before_any_upstream() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = post_v1(
        &test_client(),
        &gw.url(),
        &vkey(&sk),
        r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#.into(),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 400);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("claude-opus-4-8") && text.contains("/v1/messages"),
        "{text}"
    );
    assert_eq!(mock.hits(), 0, "the mismatched body must not be forwarded");
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_rejections_total", "wire_mismatch") >= 1.0,
        "{metrics}"
    );
}

/// OpenRouter (and other candidate) spellings are aliases for the catalog row.
#[tokio::test]
async fn v1_accepts_a_candidate_spelling_as_an_alias() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = catalog_gateway(
        nats_port,
        &b64(&pubkey),
        &primary.authority(),
        &fallback.authority(),
    )
    .await;

    let resp = post_v1(
        &test_client(),
        &gw.url(),
        &vkey(&sk),
        r#"{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#.into(),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let cap = primary.captured().expect("alias must resolve to the row");
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"gpt-4o-mini""#),
        "the alias must be rewritten to the serving candidate's id: {got}"
    );
    assert_eq!(fallback.hits(), 0);
}

/// Stock OpenAI/Anthropic SDKs list models at GET /v1/models — the catalog, with each row's wire.
#[tokio::test]
async fn v1_models_lists_the_catalog() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .get(format!("{}/v1/models", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().expect("data array");
    assert!(
        data.len() >= 2,
        "catalog list must include more than a token model: {v}"
    );
    let gpt = data
        .iter()
        .find(|m| m["id"] == "gpt-4o-mini")
        .expect("gpt-4o-mini");
    assert_eq!(gpt["wire"], "openai");
    let claude = data
        .iter()
        .find(|m| m["id"] == "claude-opus-4-8")
        .expect("claude-opus-4-8");
    assert_eq!(claude["wire"], "anthropic");
    assert_eq!(mock.hits(), 0, "listing must not contact an upstream");
}

const CLAUDE: &str = "claude-opus-4-8";

fn claude_body() -> &'static str {
    r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#
}

async fn post_claude(
    client: &reqwest::Client,
    url: &str,
    key: &str,
    extra: &[(&str, &str)],
) -> reqwest::Response {
    let mut req = client
        .post(format!("{url}/auto/v1/messages"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", CLAUDE);
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    req.body(claude_body()).send().await.unwrap()
}

/// Catalog is Anthropic-first; `x-beyond-order: bedrock` must hit Bedrock's mount, key, and id.
#[tokio::test]
async fn order_header_front_loads_bedrock_on_an_anthropic_first_row() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let bedrock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "bedrock", "openrouter"])
        .provider_authority("bedrock", &bedrock.authority())
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;

    let resp = post_claude(
        &test_client(),
        &gw.url(),
        &vkey(&sk),
        &[("x-beyond-order", "bedrock")],
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);

    let cap = bedrock.captured().expect("bedrock served the request");
    assert_eq!(cap.path, "/anthropic/v1/messages");
    assert_eq!(cap.x_api_key.as_deref(), Some("sk-bedrock-pool"));
    assert_eq!(
        cap.authorization, None,
        "Bedrock's scheme is x-api-key, not Bearer"
    );
    assert_eq!(cap.beyond_order, None, "walk header must not leak upstream");
    assert_eq!(cap.beyond_model, None);
    let body = String::from_utf8(cap.body).unwrap();
    assert!(
        body.contains(r#""model":"us.anthropic.claude-opus-4-8""#),
        "bedrock must be asked for its inference-profile id: {body}"
    );
    assert_eq!(anthropic.hits(), 0, "anthropic must not be tried first");
}

/// `only` of a provider the row lists but this gateway has no pool key for is the same 503 as
/// an unkeyed row.
#[tokio::test]
async fn only_of_an_unkeyed_provider_is_503() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .header("x-beyond-only", "openrouter")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 503);
    assert_eq!(primary.hits(), 0, "openai was filtered out of the walk");
}

/// Unparseable walk headers are dropped: catalog order, plus the error counter. Never 4xx.
#[tokio::test]
async fn junk_walk_header_keeps_catalog_order_and_is_counted() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Json).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .header("x-beyond-split", "nope")
        .body(body())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200, "junk must not 4xx the request");
    assert!(primary.hits() >= 1, "default catalog order is openai-first");
    assert_eq!(fallback.hits(), 0);
    let metrics = gw.metrics().await;
    assert!(
        parse_metric(&metrics, "ai_control_header_errors_total", "") >= 1.0,
        "junk walk header must be counted:\n{metrics}"
    );
}

/// Weighted split over many requests hits both named primaries. Leftover is failover, so a live
/// primary is enough — we never need the leftover to fire.
#[tokio::test]
async fn split_over_n_requests_hits_both_primaries() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let bedrock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "bedrock", "openrouter"])
        .provider_authority("bedrock", &bedrock.authority())
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;

    let client = test_client();
    let key = vkey(&sk);
    for _ in 0..40 {
        let resp = post_claude(
            &client,
            &gw.url(),
            &key,
            &[("x-beyond-split", "anthropic=70,bedrock=30")],
        )
        .await;
        assert_eq!(resp.status().as_u16(), 200);
    }
    assert!(
        anthropic.hits() >= 1 && bedrock.hits() >= 1,
        "70/30 split must land on both primaries (anthropic={}, bedrock={})",
        anthropic.hits(),
        bedrock.hits()
    );
}
