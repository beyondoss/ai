//! End-to-end: managed catalog walks whose `model` is not in the first 64 KiB of the body.
//!
//! Stock Python SDKs serialize `model` *after* the large field (`messages`, `input`), so an agent
//! turn or an embeddings batch past pingora's 64 KiB retry buffer used to 404 ("missing model") or,
//! when the read that found `model` also finished the body, hang until the client timed out. The
//! gateway now reads on, and re-runs a fully read body as a subrequest (`FullBody` in `proxy.rs`).
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use std::time::Duration;

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

/// A client that gives up well before a hung request would.
fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

/// `{"messages":[…filler…],"model":…}`: the openai-python field order, `model` last.
fn chat_model_last(filler_bytes: usize) -> String {
    let filler = "x".repeat(filler_bytes);
    format!(
        r#"{{"messages":[{{"role":"user","content":"{filler}"}}],"model":"gpt-4o-mini","stream":false}}"#
    )
}

async fn gateway(mode: Mode) -> (MockUpstream, Gateway, ed25519_dalek::SigningKey) {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(mode).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;
    (mock, gw, sk)
}

#[tokio::test]
async fn a_large_chat_body_with_model_last_is_served_and_billed() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let body = chat_model_last(200 * 1024);
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "{}",
        resp.text().await.unwrap()
    );

    let cap = mock.captured().expect("forwarded");
    assert_eq!(cap.path, "/v1/chat/completions");
    let sent = String::from_utf8(cap.body).unwrap();
    assert_eq!(
        sent.len(),
        body.len(),
        "the whole body reached the provider"
    );
    assert!(
        sent.contains(r#""model":"gpt-4o-mini""#),
        "model spliced for the candidate"
    );

    let line = gw
        .wait_for_log_line(&["ai.usage", r#""model":"gpt-4o-2024-08-06""#])
        .await;
    assert!(line.contains(r#""input_tokens":11"#), "{line}");
    let metrics = gw.metrics().await;
    assert_eq!(
        parse_metric(&metrics, "ai_full_body_relays_total", ""),
        1.0,
        "{metrics}"
    );
    // One request as far as the client and the counters are concerned.
    assert_eq!(parse_metric(&metrics, "ai_requests_total", ""), 1.0);
    assert_eq!(mock.hits(), 1);
}

/// The sizes that used to hang: the read that found `model` just past 64 KiB also ended the body.
#[tokio::test]
async fn bodies_just_past_the_replay_buffer_do_not_hang() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    for filler in [
        64 * 1024 - 200,
        64 * 1024,
        64 * 1024 + 100,
        70 * 1024,
        107 * 1024,
    ] {
        let resp = client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk)))
            .header("content-type", "application/json")
            .body(chat_model_last(filler))
            .send()
            .await
            .unwrap_or_else(|e| panic!("filler {filler}: {e}"));
        assert_eq!(resp.status().as_u16(), 200, "filler {filler}");
    }
    assert_eq!(mock.hits(), 5);
}

/// The dominant embeddings shape: a LangChain-style batch, `input` first, well past 64 KiB.
#[tokio::test]
async fn a_large_embeddings_batch_with_input_first_is_served() {
    let (mock, gw, sk) = gateway(Mode::Embeddings).await;
    let chunks: Vec<String> = (0..400)
        .map(|i| format!("\"document chunk {i}: {}\"", "lorem ipsum ".repeat(20)))
        .collect();
    let body = format!(
        r#"{{"input":[{}],"model":"text-embedding-3-small","encoding_format":"base64"}}"#,
        chunks.join(",")
    );
    assert!(body.len() > 64 * 1024);
    let resp = client()
        .post(format!("{}/v1/embeddings", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let cap = mock.captured().expect("forwarded");
    assert_eq!(cap.path, "/v1/embeddings");
    assert_eq!(cap.body.len(), body.len());
    let line = gw
        .wait_for_log_line(&["ai.usage", "text-embedding-3-small"])
        .await;
    assert!(line.contains(r#""input_tokens":5"#), "{line}");
}

/// Responses session state after a large `input` must still pick the Responses arm. Before, only a
/// body under 64 KiB was read for it, so a long Codex-style turn with `previous_response_id` was
/// taken for a one-shot and translated onto Chat Completions, dropping the conversation.
#[tokio::test]
async fn large_responses_session_state_after_input_walks_the_responses_arm() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let filler = "y".repeat(150 * 1024);
    let body = format!(
        r#"{{"input":"{filler}","model":"gpt-4o-mini","previous_response_id":"resp_123","store":true}}"#
    );
    let resp = client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    let _ = resp.status();
    let cap = mock.captured().expect("forwarded");
    assert_eq!(
        cap.path, "/v1/responses",
        "session state walks the Responses arm"
    );
    let sent = String::from_utf8(cap.body).unwrap();
    assert!(
        sent.contains(r#""previous_response_id":"resp_123""#),
        "byte relay keeps session state"
    );
}

#[tokio::test]
async fn a_large_body_without_a_model_is_a_404_not_a_hang() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let filler = "z".repeat(120 * 1024);
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(format!(
            r#"{{"messages":[{{"role":"user","content":"{filler}"}}]}}"#
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
    assert_eq!(mock.hits(), 0);
}

/// `model` first (the Rust and Go SDK order) still streams: no relay, pingora forwards the rest.
#[tokio::test]
async fn a_large_body_with_model_first_is_not_relayed() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let filler = "x".repeat(200 * 1024);
    let body =
        format!(r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{filler}"}}]}}"#);
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(mock.captured().expect("forwarded").body.len(), body.len());
    let metrics = gw.metrics().await;
    assert_eq!(parse_metric(&metrics, "ai_full_body_relays_total", ""), 0.0);
}
