//! T6: an upstream error reaches the client in the client's own envelope, whichever provider
//! served the row.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;

/// xAI's own error shape, as it answered a corrupt image on grok-4.3 (live 2026-10-01): a flat
/// object whose `error` is a string.
const XAI_400: &str = r#"{"code":"invalid_image","error":"code: 'Client specified an invalid argument', message: \"Invalid PNG image."}"#;

/// A Chat client on a grok row got xAI's body relayed byte for byte, so `e.body` was a string with
/// no `message`, `type` or `code`; a Messages client got the 400 typed `api_error`. Both now get
/// their envelope, typed from the status when the body names no type.
/// claim: T6
/// defect: D100
#[tokio::test]
async fn an_xai_error_arrives_in_the_clients_envelope() {
    let (pubkey, sk) = test_keypair(135);
    let mock = MockUpstream::start(Mode::Raw(400, "application/json", XAI_400)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["xai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 135);

    let chat = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(r#"{"model":"grok-4.3","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status().as_u16(), 400);
    let v: Value = chat.json().await.unwrap();
    let err = &v["error"];
    assert!(err.is_object(), "an OpenAI error envelope: {v}");
    assert!(
        err["message"]
            .as_str()
            .is_some_and(|m| m.contains("Invalid PNG image")),
        "{v}"
    );
    assert_eq!(err["type"], "invalid_request_error", "{v}");
    assert_eq!(err["code"], "invalid_image", "{v}");

    let messages = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", &key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(r#"{"model":"grok-4.3","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(messages.status().as_u16(), 400);
    let v: Value = messages.json().await.unwrap();
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("Invalid PNG image")),
        "{v}"
    );
}

/// An OpenAI-shaped error from a non-OpenAI Chat vendor is already the client's envelope: it is
/// relayed as it came (parity with calling the provider), not re-encoded.
/// claim: T6
#[tokio::test]
async fn an_error_already_in_the_clients_envelope_is_relayed_untouched() {
    const OPENAI_SHAPED: &str =
        r#"{"error":{"message":"bad","type":"invalid_request_error","param":null,"code":null}}"#;
    let (pubkey, sk) = test_keypair(136);
    let mock = MockUpstream::start(Mode::Raw(400, "application/json", OPENAI_SHAPED)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["deepseek"])
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(&sk, 136)),
        )
        .header("content-type", "application/json")
        .body(r#"{"model":"deepseek-flash","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    assert_eq!(resp.text().await.unwrap(), OPENAI_SHAPED);
}

/// The same-endpoint half of D100: a Chat client on a row whose candidate is a non-OpenAI Chat
/// Completions vendor (DeepSeek) gets that vendor's foreign error shape re-encoded in OpenAI's
/// envelope, with no translation in play.
/// claim: T6
/// defect: D100
#[tokio::test]
async fn a_same_endpoint_vendors_foreign_error_arrives_in_the_clients_envelope() {
    let (pubkey, sk) = test_keypair(137);
    let mock = MockUpstream::start(Mode::Raw(400, "application/json", XAI_400)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["deepseek"])
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(&sk, 137)),
        )
        .header("content-type", "application/json")
        .body(r#"{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert_eq!(v["error"]["code"], "invalid_image", "{v}");
}

/// A sub-resource is its provider's own API and is never re-encoded: Anthropic's error on
/// `/v1/messages/count_tokens` reaches the client byte for byte.
/// claim: T6
#[tokio::test]
async fn a_sub_resource_error_is_relayed_untouched() {
    const ANTHROPIC_400: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages: field required"}}"#;
    let (pubkey, sk) = test_keypair(138);
    let mock = MockUpstream::start(Mode::Raw(400, "application/json", ANTHROPIC_400)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/v1/messages/count_tokens", gw.url()))
        .header("x-api-key", billing_vkey(&sk, 138))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    assert_eq!(resp.text().await.unwrap(), ANTHROPIC_400);
    assert_eq!(
        mock.captured().expect("forwarded").path,
        "/v1/messages/count_tokens"
    );
}
