//! A client that hangs up once it has read the stream's terminal event, before the provider's end
//! of stream reaches the gateway. openai-python breaks out at `data: [DONE]` and closes; Codex
//! closes at `response.completed`. The answer was delivered whole, so the row says `ok` and the
//! response is cached, the same as a stream the provider ended first.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::time::Duration;

const CHAT_SSE: &str = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"total_tokens\":13}}\n\n\
data: [DONE]\n\n";

const MESSAGES_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5-20250929\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":11,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

const RESPONSES_SSE: &str = "event: response.created\n\
data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"gpt-4o-2024-08-06\",\"output\":[]}}\n\n\
event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hi\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-4o-2024-08-06\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}],\"usage\":{\"input_tokens\":11,\"output_tokens\":2,\"total_tokens\":13}}}\n\n";

/// One HTTP/1.1 chunk.
fn chunk(s: &str) -> Vec<u8> {
    let mut b = format!("{:x}\r\n", s.len()).into_bytes();
    b.extend_from_slice(s.as_bytes());
    b.extend_from_slice(b"\r\n");
    b
}

/// A chunked SSE reply: `sse`, then a pause, then `late` and the end-of-body chunk. With `late`
/// empty the client has the whole stream, terminal event included, before the provider's end of
/// stream.
async fn late_eof_upstream(sse: &'static str, late: &'static str) -> ScriptedUpstream {
    ScriptedUpstream::start(move |_, _| {
        let head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
        let mut tail = if late.is_empty() {
            Vec::new()
        } else {
            chunk(late)
        };
        tail.extend_from_slice(b"0\r\n\r\n");
        vec![
            Step::Write(head.to_vec()),
            Step::Write(chunk(sse)),
            Step::Sleep(Duration::from_millis(1500)),
            Step::Write(tail),
        ]
    })
    .await
}

/// Stream `body` to `path` and hang up as soon as `terminal` (an event's last line) and the blank
/// line ending its event have arrived.
async fn read_to_terminal_then_close(
    gw: &Gateway,
    key: &str,
    path: &str,
    body: &str,
    terminal: &str,
) {
    let mut resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let want = format!("{terminal}\n\n");
    let mut seen = Vec::new();
    while !String::from_utf8_lossy(&seen).contains(&want) {
        let chunk = tokio::time::timeout(Duration::from_secs(10), resp.chunk())
            .await
            .expect("the terminal event arrives before the provider's end of stream")
            .unwrap()
            .expect("the body did not end before the terminal event");
        seen.extend_from_slice(&chunk);
    }
    drop(resp);
}

/// Send the same request again, read it whole, and return its cache status.
async fn replay_cache_status(gw: &Gateway, key: &str, path: &str, body: &str) -> Option<String> {
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    let status = resp
        .headers()
        .get("x-beyond-cache-status")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let _ = resp.bytes().await;
    status
}

async fn closes_after_terminal(
    seed: u8,
    provider: &'static str,
    sse: &'static str,
    path: &str,
    body: &str,
    terminal: &str,
) -> (serde_json::Value, Option<String>) {
    let (pubkey, sk) = test_keypair(seed);
    let mock = late_eof_upstream(sse, "").await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&[provider])
        .cache_ttl_secs(60)
        .start()
        .await;
    let key = billing_vkey(&sk, u64::from(seed));
    read_to_terminal_then_close(&gw, &key, path, body, terminal).await;
    let rows = wait_usage_rows(&gw, 1, 15).await;
    let row = rows
        .first()
        .cloned()
        .unwrap_or_else(|| panic!("no ai.usage row; log:\n{}", gw.log()));
    let cache = replay_cache_status(&gw, &key, path, body).await;
    (row, cache)
}

fn assert_completed(row: &serde_json::Value, cache: Option<String>) {
    assert_eq!(row["outcome"], "ok", "{row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["output_tokens"], 2, "{row}");
    assert_eq!(
        cache.as_deref(),
        Some("hit"),
        "a delivered stream fills the cache: {row}"
    );
}

/// openai-python: closes at `data: [DONE]`.
/// claim: K2, BIL-12
/// defect: D120
#[tokio::test]
async fn a_chat_stream_closed_at_done_is_ok_and_cached() {
    let body = r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let (row, cache) = closes_after_terminal(
        81,
        "openai",
        CHAT_SSE,
        "/v1/chat/completions",
        body,
        "data: [DONE]",
    )
    .await;
    assert_completed(&row, cache);
}

/// A Messages client that closes at `message_stop`.
/// claim: K2, BIL-12
/// defect: D120
#[tokio::test]
async fn a_messages_stream_closed_at_message_stop_is_ok_and_cached() {
    let body = r#"{"model":"claude-sonnet-4-6","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let (row, cache) = closes_after_terminal(
        82,
        "anthropic",
        MESSAGES_SSE,
        "/v1/messages",
        body,
        "data: {\"type\":\"message_stop\"}",
    )
    .await;
    assert_completed(&row, cache);
}

/// Codex: closes at `response.completed`.
/// claim: K2, BIL-12
/// defect: D122
#[tokio::test]
async fn a_responses_stream_closed_at_response_completed_is_ok_and_cached() {
    let body = r#"{"model":"gpt-4o","stream":true,"input":"hi"}"#;
    let (row, cache) = closes_after_terminal(
        83,
        "openai",
        RESPONSES_SSE,
        "/v1/responses",
        body,
        "\"total_tokens\":13}}}",
    )
    .await;
    assert_completed(&row, cache);
}

/// A translated pairing: a Chat client on a Claude row gets the `[DONE]` the gateway wrote for the
/// upstream's `message_stop`, and closes there.
/// claim: K2, BIL-12
/// defect: D120
#[tokio::test]
async fn a_translated_stream_closed_at_done_is_ok_and_cached() {
    let body = r#"{"model":"claude-sonnet-4-6","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let (row, cache) = closes_after_terminal(
        84,
        "anthropic",
        MESSAGES_SSE,
        "/v1/chat/completions",
        body,
        "data: [DONE]",
    )
    .await;
    assert_completed(&row, cache);
}

/// The other side of the line: a client that hangs up before the terminal event reached it
/// cancelled, and its partial stream is not cached.
/// claim: K2, BIL-12
/// defect: D120
#[tokio::test]
async fn a_stream_closed_before_its_terminal_event_is_still_cancelled() {
    let body = r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    let (head, done) = CHAT_SSE.split_at(CHAT_SSE.find("data: [DONE]").unwrap());
    let (pubkey, sk) = test_keypair(85);
    let mock = late_eof_upstream(head, done).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .cache_ttl_secs(60)
        .start()
        .await;
    let key = billing_vkey(&sk, 85);
    // The usage chunk is the last event before `[DONE]`, which the provider has not sent yet.
    read_to_terminal_then_close(
        &gw,
        &key,
        "/v1/chat/completions",
        body,
        "\"total_tokens\":13}}",
    )
    .await;
    let rows = wait_usage_rows(&gw, 1, 15).await;
    let row = rows.first().cloned().unwrap();
    assert_eq!(row["outcome"], "client_cancelled", "{row}");
    assert_ne!(
        replay_cache_status(&gw, &key, "/v1/chat/completions", body)
            .await
            .as_deref(),
        Some("hit"),
        "a stream the client did not get whole is not cached"
    );
}
