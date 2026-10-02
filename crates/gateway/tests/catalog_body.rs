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
