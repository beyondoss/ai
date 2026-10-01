//! Verify phase 0, billing: exact usage on managed streams — the client cannot opt out of it, the
//! final event can be large, and cumulative updates late in the stream count.
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

/// An OpenAI chat stream the way OpenAI sends it: the usage chunk appears **only** when the request
/// asked for it with `stream_options.include_usage: true`.
fn openai_stream_honoring_include_usage(body: &[u8], _n: usize) -> Vec<Step> {
    let req: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
    let include = req
        .pointer("/stream_options/include_usage")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let mut sse = String::from(
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n\
         data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    );
    if include {
        sse.push_str(
            "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9,\"total_tokens\":14}}\n\n",
        );
    }
    sse.push_str("data: [DONE]\n\n");
    vec![Step::Write(http_response(
        200,
        "text/event-stream",
        sse.as_bytes(),
    ))]
}

/// A managed stream is metered from the provider's own usage chunk even when the client sent
/// `stream_options` that turn it off (`include_usage: false`, or an empty object).
/// claim: BIL-2
/// defect: D06
#[tokio::test]
async fn a_client_cannot_turn_off_exact_stream_metering() {
    let (pubkey, sk) = test_keypair(61);
    let mock = ScriptedUpstream::start(openai_stream_honoring_include_usage).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 61);
    let url = format!("{}/openai/v1/chat/completions", gw.url());
    // Control first: no `stream_options`, so the gateway injects it and the row is exact. Proves
    // the mock and the read below work before the cases under test.
    let bodies = [
        r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        r#"{"model":"gpt-4o","stream":true,"stream_options":{"include_usage":false},"messages":[{"role":"user","content":"hi"}]}"#,
        r#"{"model":"gpt-4o","stream":true,"stream_options":{},"messages":[{"role":"user","content":"hi"}]}"#,
    ];
    for body in bodies {
        let (status, text) = post(url.clone(), &key, body, &[]).await;
        assert_eq!(status, 200, "{text}");
    }
    let rows = wait_usage_rows(&gw, 3, 5).await;
    assert_eq!(rows.len(), 3, "{}", gw.log());
    assert_eq!(
        rows[0]["usage_estimated"], false,
        "control (gateway-injected include_usage) must be exact: {}",
        rows[0]
    );
    for (row, what) in rows[1..].iter().zip(["include_usage:false", "{}"]) {
        assert_eq!(
            (
                row["usage_estimated"].as_bool(),
                row["input_tokens"].as_u64(),
                row["output_tokens"].as_u64()
            ),
            (Some(false), Some(5), Some(9)),
            "stream_options {what}: {row}"
        );
    }
}

/// Usage the tests below assert, on a `response.completed` event bigger than the 64 KiB tail.
fn responses_stream_with_a_huge_final_event() -> &'static str {
    // `response.completed` echoes the request's instructions (and tools, and output) ahead of
    // `usage`. A Codex-sized system prompt is enough to put the event past the 64 KiB tail.
    let instructions = "You are a coding agent. ".repeat(4 * 1024); // ~96 KiB
    let sse = format!(
        "event: response.created\n\
         data: {{\"type\":\"response.created\",\"sequence_number\":0,\"response\":{{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"gpt-5-2025-08-07\",\"output\":[],\"usage\":null}}}}\n\n\
         event: response.output_text.delta\n\
         data: {{\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hi\"}}\n\n\
         event: response.completed\n\
         data: {{\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-5-2025-08-07\",\"instructions\":\"{instructions}\",\"output\":[{{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"hi\"}}]}}],\"usage\":{{\"input_tokens\":24000,\"output_tokens\":5,\"total_tokens\":24005,\"input_tokens_details\":{{\"cached_tokens\":0}},\"output_tokens_details\":{{\"reasoning_tokens\":0}}}}}}}}\n\n"
    );
    Box::leak(sse.into_boxed_str())
}

/// A Responses stream whose final `response.completed` is larger than the usage tail is still
/// billed the provider's exact usage.
/// claim: BIL-15
/// defect: D20
#[tokio::test]
async fn a_responses_stream_with_a_huge_final_event_bills_exact_usage() {
    let (pubkey, sk) = test_keypair(62);
    let mock = MockUpstream::start(Mode::Raw(
        200,
        "text/event-stream",
        responses_stream_with_a_huge_final_event(),
    ))
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let (status, text) = post(
        format!("{}/openai/v1/responses", gw.url()),
        &billing_vkey(&sk, 62),
        r#"{"model":"gpt-5","stream":true,"input":"hi"}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200, "{}", &text[..text.len().min(300)]);
    let row = usage_row_of(&gw).await;
    assert_eq!(
        (
            row["usage_estimated"].as_bool(),
            row["input_tokens"].as_u64(),
            row["output_tokens"].as_u64()
        ),
        (Some(false), Some(24000), Some(5)),
        "{row}"
    );
}

/// With server tools, Anthropic sends updated, cumulative input and cache counts on the final
/// `message_delta.usage`. Those supersede `message_start`'s.
const ANTHROPIC_CUMULATIVE_DELTA_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":2500,\"cache_read_input_tokens\":800,\"cache_creation_input_tokens\":0,\"output_tokens\":300,\"server_tool_use\":{\"web_search_requests\":1}}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

/// claim: BIL-8
/// defect: D22
#[tokio::test]
async fn anthropic_cumulative_message_delta_usage_supersedes_message_start() {
    let (pubkey, sk) = test_keypair(63);
    let mock = MockUpstream::start(Mode::Raw(
        200,
        "text/event-stream",
        ANTHROPIC_CUMULATIVE_DELTA_SSE,
    ))
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/anthropic/v1/messages", gw.url()))
        .header("x-api-key", billing_vkey(&sk, 63))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","max_tokens":512,"stream":true,"messages":[{"role":"user","content":"search the web"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let _ = resp.text().await;
    let row = usage_row_of(&gw).await;
    assert_eq!(
        (
            row["input_tokens"].as_u64(),
            row["cache_read_tokens"].as_u64(),
            row["output_tokens"].as_u64()
        ),
        (Some(2500), Some(800), Some(300)),
        "{row}"
    );
}

/// A Chat Completions stream that finishes cleanly (`[DONE]`) with no generated text and a usage
/// block whose shape the extractor cannot read (`prompt_tokens` as a string): the provider still
/// billed the turn.
const UNPARSEABLE_USAGE_SSE: &str = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n\
data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[],\"usage\":{\"prompt_tokens\":\"5\",\"completion_tokens\":9}}\n\n\
data: [DONE]\n\n";

/// A completed managed 2xx stream whose usage cannot be parsed is billed an estimate, flagged,
/// and counted as a parse error — never a silent zero-token row.
/// claim: BIL-6, BIL-12, BIL-15
/// defect: D56
#[tokio::test]
async fn a_finished_stream_with_unparseable_usage_bills_a_flagged_estimate() {
    let (pubkey, sk) = test_keypair(64);
    let mock =
        MockUpstream::start(Mode::Raw(200, "text/event-stream", UNPARSEABLE_USAGE_SSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let (status, text) = post(
        format!("{}/openai/v1/chat/completions", gw.url()),
        &billing_vkey(&sk, 64),
        r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"Summarize the history of TCP congestion control."}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let row = usage_row_of(&gw).await;
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert!(
        row["input_tokens"].as_u64().unwrap_or(0) > 0,
        "the provider took the prompt; input must be estimated: {row}"
    );
    wait_for_metric(&gw, "ai_usage_parse_errors_total", "", 1.0).await;
}
