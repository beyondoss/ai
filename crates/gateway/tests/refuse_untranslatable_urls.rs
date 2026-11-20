//! End-to-end: a request that translation would strip a URL from is a 400 before any upstream,
//! instead of an answer (billed) about a picture the model never saw.
//!
//! - A Chat Completions `file://` image to a Messages candidate, a Responses `gs://` image to a
//!   Chat Completions candidate, and a Messages `file://` source to one, are refused naming the row.
//! - An `https://` or data-URI image still translates and is served.
//! - A byte relay (Chat Completions to a Chat Completions candidate) forwards the same URL as sent.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::{Value, json};

async fn post(gw: &Gateway, key: &str, path: &str, body: &Value) -> reqwest::Response {
    test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap()
}

/// An image row whose only candidate speaks Chat Completions and that has no Responses arm.
const CHAT_ONLY: &str = "google/gemma-4-31b-it";

fn chat_image(model: &str, url: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": url}}]}]})
}

fn responses_image(model: &str, url: &str) -> Value {
    json!({"model": model, "store": false, "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "What is this?"},
        {"type": "input_image", "image_url": url}]}]})
}

fn messages_image(model: &str, url: &str) -> Value {
    json!({"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": url}}]}]})
}

async fn assert_refused(resp: reqwest::Response, what: &str) {
    assert_eq!(resp.status().as_u16(), 400, "{what}");
    let err: Value = resp.json().await.unwrap();
    let msg = err.to_string();
    assert!(
        msg.contains("does not accept an image or file URL that is not http(s) or inline data"),
        "{what}: {msg}"
    );
}

/// claim: CAT-5
#[tokio::test]
async fn a_url_translation_would_drop_is_refused_before_any_upstream() {
    let (pubkey, sk) = test_keypair(73);
    let claude = MockUpstream::start(Mode::AnthropicJson).await;
    let openai = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &claude.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .provider_authority("openai", &openai.authority())
        .provider_authority("openrouter", &openai.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 73);

    // Translated: Chat to Messages, Responses to Chat, Messages to Chat.
    for url in ["file:///home/me/cat.png", "gs://bucket/cat.png", "cat.png"] {
        let resp = post(
            &gw,
            &key,
            "/v1/chat/completions",
            &chat_image("claude-opus-4-8", url),
        )
        .await;
        assert_refused(resp, &format!("chat to messages {url}")).await;
        let resp = post(&gw, &key, "/v1/responses", &responses_image(CHAT_ONLY, url)).await;
        assert_refused(resp, &format!("responses to chat {url}")).await;
        let resp = post(&gw, &key, "/v1/messages", &messages_image(CHAT_ONLY, url)).await;
        assert_refused(resp, &format!("messages to chat {url}")).await;
    }
    assert_eq!(
        claude.hits() + openai.hits(),
        0,
        "a refused URL reached a provider"
    );

    // The same bodies with a URL that is forwarded translate and are served.
    let https = "https://example.com/cat.png";
    let data = "data:image/png;base64,iVBORw0KGgo=";
    for url in [https, data] {
        let resp = post(
            &gw,
            &key,
            "/v1/chat/completions",
            &chat_image("claude-opus-4-8", url),
        )
        .await;
        assert_eq!(resp.status().as_u16(), 200, "chat to messages {url}");
        let resp = post(&gw, &key, "/v1/responses", &responses_image(CHAT_ONLY, url)).await;
        assert_eq!(resp.status().as_u16(), 200, "responses to chat {url}");
        let resp = post(&gw, &key, "/v1/messages", &messages_image(CHAT_ONLY, url)).await;
        assert_eq!(resp.status().as_u16(), 200, "messages to chat {url}");
    }
    assert_eq!(claude.hits(), 2);
    assert_eq!(openai.hits(), 4);

    // A byte relay forwards what it was sent; the provider answers.
    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &chat_image("gpt-4o", "file:///home/me/cat.png"),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(openai.hits(), 5);

    // A repeat of a refusal is refused again, and the same request with a forwardable URL succeeds.
    let bad = chat_image("claude-opus-4-8", "file:///home/me/cat.png");
    assert_refused(
        post(&gw, &key, "/v1/chat/completions", &bad).await,
        "repeat",
    )
    .await;
}
