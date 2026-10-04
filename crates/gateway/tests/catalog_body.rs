//! End-to-end: the body a catalog walk sends names exactly the model the row chose.
//!
//! A catalog walk routes on one `model` and rewrites one `model` to the serving candidate's id. A
//! body with a second root `model` key leaves the provider to pick between them, and most JSON
//! parsers take the last: `{"model":"cheap",…,"model":"gpt-5.5-pro"}` would route as the cheap row
//! and be served as gpt-5.5-pro. Asserted at the upstream.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn managed_key(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 41,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

/// claim: SEC-21
/// defect: D34
#[tokio::test]
async fn a_body_with_two_root_model_keys_is_refused_on_a_catalog_walk() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let auth = format!("Bearer {}", managed_key(&sk));
    let cases: [(&str, Option<&str>, &str); 3] = [
        // Headerless: the row comes from the first `model`.
        (
            "/v1/chat/completions",
            None,
            r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#,
        ),
        // The header names the row; the body carries two spellings of the key.
        (
            "/auto/v1/chat/completions",
            Some("gpt-4o-mini"),
            r#"{"model":"gpt-4o-mini","model":"gpt-4o","messages":[]}"#,
        ),
        // A translated walk (Messages client, OpenAI row): checked before translation collapses it.
        (
            "/v1/messages",
            Some("gpt-4o-mini"),
            r#"{"model":"gpt-4o-mini","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#,
        ),
    ];
    for (path, row, body) in cases {
        let mut req = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("authorization", &auth)
            .header("content-type", "application/json")
            .body(body);
        if let Some(row) = row {
            req = req.header("x-beyond-model", row);
        }
        let status = req.send().await.unwrap().status().as_u16();
        assert_eq!(status, 400, "{path}: {body}");
        if let Some(cap) = mock.captured() {
            let sent = String::from_utf8_lossy(&cap.body);
            assert!(
                !sent.contains("gpt-4o\""),
                "{path}: the second model reached the provider: {sent}"
            );
        }
    }
    wait_for_metric(&gw, "ai_rejections_total", "duplicate_model", 3.0).await;

    // One `model`, as every real client sends, still walks.
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", &auth)
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let sent: serde_json::Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
    assert_eq!(sent["model"], "gpt-4o-mini");
}

/// Where the body is already in hand before connecting (a headerless walk reads it to choose the
/// row; a header-won large or Responses walk reads it too), two root `model` keys are refused
/// before the upstream gets anything, with the gateway's JSON 400 and its request id, not a bare
/// 400 after the request headers went out.
/// claim: SEC-21, CAT-11
/// defect: D94
#[tokio::test]
async fn a_duplicate_model_is_refused_before_the_upstream_with_a_json_400() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let auth = format!("Bearer {}", managed_key(&sk));
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", &auth)
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    assert!(
        resp.headers().contains_key("x-beyond-request-id"),
        "{:?}",
        resp.headers()
    );
    let text = resp.text().await.unwrap();
    let v: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text:?}"));
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("model")),
        "{v}"
    );
    assert_eq!(mock.hits(), 0, "the upstream got the request");
}

/// POST `body` (sent as written) to `path` with a managed key; the status.
async fn send_raw(gw: &Gateway, sk: &ed25519_dalek::SigningKey, path: &str, body: &str) -> u16 {
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {}", managed_key(sk)))
        .header("x-api-key", managed_key(sk))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(body.to_owned())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let _ = resp.bytes().await;
    status
}

/// What the upstream received, parsed.
fn sent(mock: &MockUpstream) -> serde_json::Value {
    let cap = mock.captured().expect("a request reached the upstream");
    serde_json::from_slice(&cap.body)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&cap.body)))
}

/// A same-wire rewrite that moves bytes ahead of `model` (a `developer` role spelled `system`, a
/// Claude-on-Chat body re-encoded without its reasoning control) is followed by a fresh scan, so
/// the model is still re-spelled where it now is and the body stays valid JSON.
/// claim: SEC-21, TRN-16
/// defect: D173
#[tokio::test]
async fn a_rewrite_ahead_of_the_model_still_respells_the_model() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["deepseek", "openrouter"])
        .provider_authority("openrouter", &mock.authority())
        .start()
        .await;
    // DeepSeek does not read `developer`: the role shrinks by three bytes ahead of `model`.
    let developer = r#"{"messages":[{"role":"developer","content":"be brief"},{"role":"user","content":"hi"}],"model":"deepseek-flash","temperature":0.5}"#;
    assert_eq!(
        send_raw(&gw, &sk, "/v1/chat/completions", developer).await,
        200
    );
    let v = sent(&mock);
    assert_eq!(
        (v["model"].as_str(), v["messages"][0]["role"].as_str()),
        (Some("deepseek-flash"), Some("system")),
        "{v}"
    );
    // A Claude row on OpenRouter's Chat endpoint: a tool turn with no thinking to replay goes
    // without `reasoning_effort`, and the body is re-encoded with `model` after `messages`.
    let claude = r#"{"model":"claude-opus-4-8","reasoning_effort":"high","messages":[{"role":"user","content":"weather?"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"w","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c1","content":"sunny"}],"tools":[{"type":"function","function":{"name":"w","parameters":{"type":"object"}}}]}"#;
    assert_eq!(
        send_raw(&gw, &sk, "/v1/chat/completions", claude).await,
        200
    );
    let v = sent(&mock);
    assert_eq!(v["model"], "anthropic/claude-opus-4.8", "{v}");
    assert!(v.get("reasoning_effort").is_none(), "{v}");
}

/// A same-wire relay is byte for byte except where a host needs otherwise: an assistant turn's
/// `reasoning_details` reaches a non-Claude Chat host as sent (only a Claude model behind Chat
/// loses unreplayable reasoning), and an explicit `null` reaches Anthropic's Messages endpoint as
/// sent (only a Chat host that is not OpenAI's has its root nulls dropped).
/// claim: CAT-9, TRN-16
/// defect: D101
#[tokio::test]
async fn a_same_wire_relay_keeps_what_its_host_reads() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["deepseek", "anthropic"])
        .provider_authority("anthropic", &anthropic.authority())
        .start()
        .await;
    let chat = r#"{"model":"deepseek-flash","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"x","reasoning_details":[{"type":"reasoning.text","text":"t"}]},{"role":"user","content":"more"}]}"#;
    assert_eq!(send_raw(&gw, &sk, "/v1/chat/completions", chat).await, 200);
    let cap = mock.captured().unwrap();
    assert_eq!(String::from_utf8_lossy(&cap.body), chat);

    let messages = r#"{"model":"claude-opus-4-8","max_tokens":16,"temperature":null,"messages":[{"role":"user","content":"hi"}]}"#;
    assert_eq!(send_raw(&gw, &sk, "/v1/messages", messages).await, 200);
    let v = sent(&anthropic);
    assert!(
        v.as_object().unwrap().get("temperature") == Some(&serde_json::Value::Null),
        "{v}"
    );
}

/// `stream_options` is made to ask for usage only on a candidate that streams Chat Completions; a
/// Responses stream's own `stream_options` reaches its Responses upstream as the client sent it.
/// claim: BIL-2, CAT-9
/// defect: D06
#[tokio::test]
async fn a_responses_streams_options_are_not_given_include_usage() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let body = r#"{"model":"gpt-5.4","stream":true,"stream_options":{"include_obfuscation":false},"store":false,"input":"hi"}"#;
    send_raw(&gw, &sk, "/v1/responses", body).await;
    let cap = mock.captured().unwrap();
    assert_eq!(cap.path, "/v1/responses");
    let v: serde_json::Value = serde_json::from_slice(&cap.body).unwrap();
    assert_eq!(
        v["stream_options"],
        serde_json::json!({"include_obfuscation": false}),
        "{v}"
    );
}

/// The reasoning items the gateway minted from Claude's thinking are cut out of a Responses body
/// ahead of `model` on its way to a Responses upstream (here xAI's, on a row with no Responses arm,
/// where no signed-id check makes the cut first); the body is scanned again after the cut, so
/// `model` is still re-spelled where it now is and nothing after it moves.
/// claim: SEC-21, CAT-9
/// defect: D50
#[tokio::test]
async fn a_reasoning_cut_ahead_of_the_model_still_respells_the_model() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["xai"])
        .start()
        .await;
    // Instructions after `model`, longer than the cut: a stale span would land inside them.
    let instructions = "Answer in French. ".repeat(12);
    let body = format!(
        r#"{{"input":[{{"type":"reasoning","id":"rs_gw18f2c3a4b5d60001","encrypted_content":"rs_gw:EqQBsig==","summary":[]}},{{"role":"user","content":"hi"}}],"model":"grok-4.3","instructions":"{instructions}","store":false}}"#
    );
    send_raw(&gw, &sk, "/v1/responses", &body).await;
    let cap = mock.captured().unwrap();
    assert_eq!(cap.path, "/v1/responses");
    let v = sent(&mock);
    assert_eq!(v["model"], "grok-4.3", "{v}");
    assert_eq!(v["instructions"], instructions.as_str(), "{v}");
    assert_eq!(v["input"].as_array().map(Vec::len), Some(1), "{v}");
}
