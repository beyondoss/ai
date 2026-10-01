//! Verify phase 0, billing: which extractor, which normalization and which model a row is billed
//! under, by route. Each test asserts the CORRECT behavior; a reproduced defect's test is ignored.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;

async fn post(url: String, key: &str, body: &str, extra: &[(&str, &str)]) -> (u16, String) {
    let mut req = test_client()
        .post(url)
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json");
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    let resp = req.body(body.to_owned()).send().await.unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

/// OpenRouter serves Anthropic Messages at `/api/v1/messages`. A provider-routed request there
/// gets an Anthropic body back, and must be billed with the Anthropic extractor (the forwarded
/// path's wire), not OpenRouter's default OpenAI dialect.
/// claim: BIL-1
/// defect: D05
#[tokio::test]
async fn openrouter_messages_path_is_metered_with_the_anthropic_extractor() {
    let (pubkey, sk) = test_keypair(51);
    let json = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &json.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    let key = billing_vkey(&sk, 51);
    let (status, text) = post(
        format!("{}/openrouter/api/v1/messages", gw.url()),
        &key,
        r#"{"model":"anthropic/claude-opus-4.8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["input_tokens"].as_u64(), Some(13), "non-stream: {row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(7), "non-stream: {row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
}

/// The streaming twin of the test above: an Anthropic SSE stream on OpenRouter's Messages mount.
/// claim: BIL-1
/// defect: D05
#[tokio::test]
async fn openrouter_messages_stream_is_metered_with_the_anthropic_extractor() {
    let (pubkey, sk) = test_keypair(52);
    let sse = MockUpstream::start(Mode::AnthropicSse).await;
    let gw = Gateway::builder(unused_nats_port(), &sse.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    let (status, text) = post(
        format!("{}/openrouter/api/v1/messages", gw.url()),
        &billing_vkey(&sk, 52),
        r#"{"model":"anthropic/claude-opus-4.8","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["input_tokens"].as_u64(), Some(13), "stream: {row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(7), "stream: {row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
}

const ANTHROPIC_CACHED: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":100,"cache_read_input_tokens":900,"cache_creation_input_tokens":0,"output_tokens":7}}"#;

const OPENROUTER_CACHED: &str = r#"{"id":"gen-1","object":"chat.completion","model":"anthropic/claude-opus-4.8","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":7,"total_tokens":1007,"prompt_tokens_details":{"cached_tokens":900}}}"#;

/// One Claude catalog row, the same 1000-token prompt with 900 cached, served once by Anthropic
/// (`input_tokens` excludes cache reads) and once by OpenRouter Chat Completions (`prompt_tokens`
/// includes them). The two rows must agree on what `input_tokens` means.
/// claim: BIL-7
/// defect: D25
#[tokio::test]
async fn a_claude_row_normalizes_input_tokens_the_same_on_both_wires() {
    let (pubkey, sk) = test_keypair(53);
    let anthropic = MockUpstream::start(Mode::Raw(200, "application/json", ANTHROPIC_CACHED)).await;
    let openrouter =
        MockUpstream::start(Mode::Raw(200, "application/json", OPENROUTER_CACHED)).await;
    let via_anthropic = Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;
    let via_openrouter = Gateway::builder(
        unused_nats_port(),
        &GatewayBuilder::dead_authority(),
        &b64(&pubkey),
    )
    .providers(&["anthropic", "openrouter"])
    .provider_authority("openrouter", &openrouter.authority())
    .start()
    .await;
    let body = r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
    for gw in [&via_anthropic, &via_openrouter] {
        let (status, text) = post(
            format!("{}/auto/v1/messages", gw.url()),
            &billing_vkey(&sk, 53),
            body,
            &[("x-beyond-model", "claude-opus-4-8")],
        )
        .await;
        assert_eq!(status, 200, "{text}");
    }
    let a = usage_row_of(&via_anthropic).await;
    let o = usage_row_of(&via_openrouter).await;
    assert_eq!(a["provider"], "anthropic", "{a}");
    assert_eq!(o["provider"], "openrouter", "{o}");
    // The rows keep each wire's own `input_tokens` (downstream consumers rely on it) and say which
    // convention it follows in `usage_wire`. Normalized through it, the two must agree.
    assert_eq!(a["usage_wire"], "anthropic", "{a}");
    assert_eq!(o["usage_wire"], "openai", "{o}");
    assert_eq!(
        (prompt_tokens(&a), a["cache_read_tokens"].as_u64()),
        (prompt_tokens(&o), o["cache_read_tokens"].as_u64()),
        "same prompt, same catalog row, different input_tokens semantics:\nanthropic:  {a}\nopenrouter: {o}"
    );
    assert_eq!(prompt_tokens(&a), Some(1000), "{a}");
}

/// The whole prompt a row reports, cache reads and writes included: `input_tokens` on the OpenAI
/// convention, `input_tokens` plus both cache counts on the Anthropic one.
fn prompt_tokens(row: &serde_json::Value) -> Option<u64> {
    let input = row["input_tokens"].as_u64()?;
    match row["usage_wire"].as_str()? {
        "openai" => Some(input),
        "anthropic" => {
            Some(input + row["cache_read_tokens"].as_u64()? + row["cache_write_tokens"].as_u64()?)
        }
        _ => None,
    }
}

/// OpenRouter reports Claude cache writes on the Chat Completions wire as
/// `prompt_tokens_details.cache_write_tokens`. The row must carry them.
/// claim: BIL-8
/// defect: D21
#[tokio::test]
async fn openrouter_cache_write_tokens_reach_the_row() {
    let (pubkey, sk) = test_keypair(54);
    let mock = MockUpstream::start(Mode::Raw(
        200,
        "application/json",
        r#"{"id":"gen-1","object":"chat.completion","model":"anthropic/claude-opus-4.8","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":7,"total_tokens":127,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":50}}}"#,
    ))
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    let (status, text) = post(
        format!("{}/openrouter/api/v1/chat/completions", gw.url()),
        &billing_vkey(&sk, 54),
        r#"{"model":"anthropic/claude-opus-4.8","messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["input_tokens"].as_u64(), Some(120), "{row}");
    assert_eq!(row["cache_write_tokens"].as_u64(), Some(50), "{row}");
}

const RESPONSES_SNAPSHOT_SSE: &str = "event: response.created\n\
data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"gpt-5.1-2025-11-13\",\"output\":[],\"usage\":null}}\n\n\
event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hi\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-5.1-2025-11-13\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}],\"usage\":{\"input_tokens\":12,\"output_tokens\":5,\"total_tokens\":17,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens_details\":{\"reasoning_tokens\":0}}}}\n\n";

/// A Responses stream echoes the pinned snapshot under `response.model`. The row bills that
/// snapshot, as it does for Chat Completions and Messages streams.
/// claim: BIL-13
/// defect: D27
#[tokio::test]
async fn a_responses_stream_bills_the_echoed_snapshot() {
    let (pubkey, sk) = test_keypair(55);
    let mock =
        MockUpstream::start(Mode::Raw(200, "text/event-stream", RESPONSES_SNAPSHOT_SSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let (status, text) = post(
        format!("{}/openai/v1/responses", gw.url()),
        &billing_vkey(&sk, 55),
        r#"{"model":"gpt-5.1","stream":true,"input":"hi"}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(
        row["input_tokens"].as_u64(),
        Some(12),
        "usage parsed: {row}"
    );
    assert_eq!(row["requested_model"], "gpt-5.1", "{row}");
    assert_eq!(row["model"], "gpt-5.1-2025-11-13", "{row}");
    // A provider-routed row names the catalog row it prices at, through the snapshot suffix.
    assert_eq!(row["price_model"], "gpt-5.1", "{row}");
}
