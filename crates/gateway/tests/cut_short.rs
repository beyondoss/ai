//! End-to-end: a managed stream cut short before its usage block is billed an estimate, not zero.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).
//!
//! A stream's usage block is its last event. A client that hangs up first — a cancelled agent turn —
//! used to leave the gateway with nothing to parse, so it emitted a zero-token row while the provider
//! billed us for everything it generated. That made "stream a long answer, disconnect before the last
//! event" free. These tests cancel real streams through the real binary and read the `ai.usage` row.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code — allow the
// panic-surface restriction lints denied workspace-wide in `[workspace.lints.clippy]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use std::time::Duration;

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

/// POST, read the first chunk of the streamed response, and hang up — what an SDK does when the
/// user hits ESC mid-turn.
async fn stream_then_cancel(url: String, key: &str, auth_header: &str, body: String) {
    let client = test_client();
    let mut req = client
        .post(url)
        .header("content-type", "application/json")
        .body(body);
    req = if auth_header == "authorization" {
        req.header("authorization", format!("Bearer {key}"))
    } else {
        req.header(auth_header, key)
            .header("anthropic-version", "2023-06-01")
    };
    let mut resp = req.send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let first = tokio::time::timeout(Duration::from_secs(10), resp.chunk())
        .await
        .expect("the stalled stream still delivers what it sent")
        .unwrap()
        .expect("a first chunk");
    assert!(!first.is_empty());
    drop(resp);
}

/// The one `ai.usage` row, parsed. `tracing`'s JSON layer nests event fields under `fields`.
async fn usage_row(gw: &Gateway) -> serde_json::Value {
    let line = gw.wait_for_log_line(&[r#""target":"ai.usage""#]).await;
    let v: serde_json::Value = serde_json::from_str(&line).expect("usage line is JSON");
    v.get("fields").cloned().unwrap_or(v)
}

/// claim: B2, BIL-3, BIL-20
#[tokio::test]
async fn a_cancelled_openai_stream_is_billed_an_estimate() {
    let (pubkey, sk) = test_keypair(41);
    let mock = MockUpstream::start(Mode::StallSse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .start()
        .await;
    let body = r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"Explain TCP congestion control in detail, please."}]}"#.to_string();

    stream_then_cancel(
        format!("{}/openai/v1/chat/completions", gw.url()),
        &vkey(&sk, 41),
        "authorization",
        body.clone(),
    )
    .await;

    let row = usage_row(&gw).await;
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert_eq!(
        row["output_tokens"].as_u64(),
        Some(STALL_DELTAS),
        "one token per OpenAI delta event relayed: {row}"
    );
    assert_eq!(
        row["input_tokens"].as_u64(),
        Some(body.len() as u64 / 5),
        "input is estimated from the request body at 5 bytes/token: {row}"
    );
    wait_for_metric(&gw, "ai_usage_estimated_total", "", 1.0).await;
}

/// Anthropic reports input and cache tokens on `message_start`, the first event — so those stay
/// exact, and only the output count (which rides the missing `message_delta`) is estimated.
/// claim: B2, BIL-3, BIL-20
#[tokio::test]
async fn a_cancelled_anthropic_stream_keeps_exact_input_and_estimates_output() {
    let (pubkey, sk) = test_keypair(42);
    let mock = MockUpstream::start(Mode::AnthropicStallSse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let body = r#"{"model":"claude-opus-4-8","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#.to_string();

    stream_then_cancel(
        format!("{}/anthropic/v1/messages", gw.url()),
        &vkey(&sk, 42),
        "x-api-key",
        body,
    )
    .await;

    let row = usage_row(&gw).await;
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert_eq!(
        row["input_tokens"].as_u64(),
        Some(STALL_ANTHROPIC_INPUT_TOKENS),
        "message_start's input count is exact, not estimated: {row}"
    );
    assert_eq!(row["output_tokens"].as_u64(), Some(STALL_DELTAS), "{row}");
}

/// An inline image is ~1 MB of base64 and ~1–2K tokens. Counting its bytes as text would bill one
/// picture as a novel, so the base64 payload is excluded from the input estimate.
#[tokio::test]
async fn base64_images_do_not_inflate_the_input_estimate() {
    let (pubkey, sk) = test_keypair(43);
    let mock = MockUpstream::start(Mode::StallSse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .start()
        .await;
    let image = "iVBORw0KGgo".repeat(20_000);
    let body = format!(
        r#"{{"model":"gpt-4o-mini","stream":true,"messages":[{{"role":"user","content":[{{"type":"text","text":"what is this?"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{image}"}}}}]}}]}}"#
    );

    stream_then_cancel(
        format!("{}/openai/v1/chat/completions", gw.url()),
        &vkey(&sk, 43),
        "authorization",
        body.clone(),
    )
    .await;

    let row = usage_row(&gw).await;
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert_eq!(
        row["input_tokens"].as_u64(),
        Some((body.len() - image.len()) as u64 / 5),
        "the base64 payload is not text: {row}"
    );
}

/// A stream that finishes normally is billed exactly what the provider reported — no estimate.
#[tokio::test]
async fn a_finished_stream_is_not_estimated() {
    let (pubkey, sk) = test_keypair(44);
    let mock = MockUpstream::start(Mode::Sse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, 44)))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.text().await.unwrap().contains("[DONE]"));

    let row = usage_row(&gw).await;
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["input_tokens"].as_u64(), Some(5), "{row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(9), "{row}");
}

/// A 200 stream that carries only an error event — Anthropic's `overloaded_error` before any output —
/// is not work the provider billed for, so it is not estimated either.
/// claim: BIL-12
#[tokio::test]
async fn an_error_only_stream_is_not_billed_an_estimate() {
    let (pubkey, sk) = test_keypair(45);
    let mock = MockUpstream::start(Mode::AnthropicErrorSse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/anthropic/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk, 45))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.text().await.unwrap();

    let row = usage_row(&gw).await;
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["input_tokens"].as_u64(), Some(0), "{row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(0), "{row}");
}

/// The over-count this guards: a Messages client's image is `{"type":"base64","data":"…"}` — not a
/// data URI — and when an OpenAI-wire candidate serves it, input is estimated from the body. That
/// payload must not be counted as text either.
#[tokio::test]
async fn anthropic_format_images_do_not_inflate_the_input_estimate() {
    let (pubkey, sk) = test_keypair(46);
    let mock = MockUpstream::start(Mode::StallSse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .start()
        .await;
    let image = "iVBORw0KGgo".repeat(20_000);
    let body = format!(
        r#"{{"model":"gpt-4o-mini","max_tokens":64,"stream":true,"messages":[{{"role":"user","content":[{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{image}"}}}},{{"type":"text","text":"what is this?"}}]}}]}}"#
    );

    stream_then_cancel(
        format!("{}/v1/messages", gw.url()),
        &vkey(&sk, 46),
        "x-api-key",
        body.clone(),
    )
    .await;

    let row = usage_row(&gw).await;
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert_eq!(
        row["input_tokens"].as_u64(),
        Some((body.len() - image.len()) as u64 / 5),
        "an Anthropic base64 source is not text: {row}"
    );
}
