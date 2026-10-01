//! Who pays for a cache write: the party that chose to cache.
//!
//! A Chat Completions or Responses client on a Claude row sends no `cache_control`, so the gateway
//! adds default breakpoints (`translate::auto_cache_breakpoints`). The writes those breakpoints
//! cause are the gateway's optimization, not the client's request: they bill at the input rate
//! (folded into `input_tokens`, with `gateway_cache_write_tokens` saying how many), and the client
//! is shown the same thing. A client that sent `cache_control` itself asked for the writes and is
//! billed true cache writes, as calling Anthropic directly would.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;
use std::time::Duration;

const PROVIDERS: &[&str] = &["anthropic", "openai", "openrouter"];

/// 50 uncached, 300 read, 2000 written.
const ANTHROPIC_WRITE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":10,"cache_read_input_tokens":300,"cache_creation_input_tokens":2000,"cache_creation":{"ephemeral_5m_input_tokens":2000,"ephemeral_1h_input_tokens":0}}}"#;

/// The same counts as a stream: input and cache on `message_start`, cumulative on `message_delta`.
const ANTHROPIC_WRITE_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":50,\"output_tokens\":1,\"cache_read_input_tokens\":300,\"cache_creation_input_tokens\":2000}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":50,\"cache_read_input_tokens\":300,\"cache_creation_input_tokens\":2000,\"output_tokens\":10}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

/// A Chat request with a system prompt and no `cache_control`: the gateway marks the system block.
const CHAT_UNMARKED: &str = r#"{"model":"claude-opus-4-8","messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"hi"}]}"#;
const CHAT_UNMARKED_STREAM: &str = r#"{"model":"claude-opus-4-8","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"hi"}]}"#;
/// The same request with the client's own marker on its system prompt.
const CHAT_MARKED: &str = r#"{"model":"claude-opus-4-8","messages":[{"role":"system","content":[{"type":"text","text":"You are terse.","cache_control":{"type":"ephemeral"}}]},{"role":"user","content":"hi"}]}"#;
const RESPONSES_UNMARKED: &str =
    r#"{"model":"claude-opus-4-8","instructions":"You are terse.","input":"hi","store":false}"#;

async fn call(upstream: &ScriptedUpstream, path: &str, body: &str) -> (Value, String, Gateway) {
    let (pubkey, sk) = test_keypair(76);
    let gw = Gateway::builder(unused_nats_port(), &upstream.authority(), &b64(&pubkey))
        .providers(PROVIDERS)
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {}", billing_vkey(&sk, 76)))
        .header("content-type", "application/json")
        .body(body.to_owned())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    wait_usage_rows(&gw, 1, 5).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let rows = usage_rows_of(&gw);
    assert_eq!(rows.len(), 1, "one call, one row: {rows:#?}");
    (rows.into_iter().next().unwrap(), text, gw)
}

/// The last `usage` object the client was shown (a whole body, or the last SSE chunk carrying one).
fn client_usage(text: &str) -> Value {
    let pick = |v: &Value| {
        [&v["usage"], &v["response"]["usage"]]
            .into_iter()
            .find(|u| u.is_object())
            .cloned()
    };
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return pick(&v).expect("a usage block");
    }
    text.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok())
        .filter_map(|v| pick(&v))
        .next_back()
        .expect("a usage chunk")
}

fn n(v: &Value, k: &str) -> Option<u64> {
    v.pointer(k).and_then(Value::as_u64)
}

/// Gateway-added breakpoints: the 2000 written tokens bill as input, the row says the gateway caused
/// them, and the Chat client is shown the same (no cache writes, the same whole prompt). Reads are
/// unaffected.
/// claim: E7, BIL-6
/// defect: D76
#[tokio::test]
async fn gateway_added_breakpoints_bill_their_writes_as_input() {
    for (body, ctype, request) in [
        (ANTHROPIC_WRITE, "application/json", CHAT_UNMARKED),
        (
            ANTHROPIC_WRITE_SSE,
            "text/event-stream",
            CHAT_UNMARKED_STREAM,
        ),
    ] {
        let up = ScriptedUpstream::reply(200, ctype, body.to_owned()).await;
        let (row, text, _gw) = call(&up, "/v1/chat/completions", request).await;
        let sent = String::from_utf8(up.bodies()[0].clone()).unwrap();
        assert!(
            sent.contains("cache_control"),
            "the gateway added a breakpoint: {sent}"
        );
        assert_eq!(row["usage_wire"], "anthropic", "{row}");
        assert_eq!(n(&row, "/input_tokens"), Some(2050), "{ctype}: {row}");
        assert_eq!(n(&row, "/cache_write_tokens"), Some(0), "{ctype}: {row}");
        assert_eq!(n(&row, "/cache_write_1h_tokens"), Some(0), "{ctype}: {row}");
        assert_eq!(
            n(&row, "/gateway_cache_write_tokens"),
            Some(2000),
            "{ctype}: {row}"
        );
        assert_eq!(n(&row, "/cache_read_tokens"), Some(300), "{ctype}: {row}");
        let u = client_usage(&text);
        assert_eq!(n(&u, "/prompt_tokens"), Some(2350), "{ctype}: {u}");
        assert_eq!(
            n(&u, "/prompt_tokens_details/cached_tokens"),
            Some(300),
            "{ctype}: {u}"
        );
        assert_eq!(
            n(&u, "/prompt_tokens_details/cache_write_tokens").unwrap_or(0),
            0,
            "{ctype}: the client must not be shown writes billed as input: {u}"
        );
    }
}

/// The Responses twin: `input_tokens` is the whole prompt and carries no cache writes.
/// claim: E7, BIL-6
/// defect: D76
#[tokio::test]
async fn a_responses_client_sees_gateway_cache_writes_as_input() {
    let up = ScriptedUpstream::reply(200, "application/json", ANTHROPIC_WRITE.to_owned()).await;
    let (row, text, _gw) = call(&up, "/v1/responses", RESPONSES_UNMARKED).await;
    assert_eq!(n(&row, "/input_tokens"), Some(2050), "{row}");
    assert_eq!(n(&row, "/cache_write_tokens"), Some(0), "{row}");
    assert_eq!(n(&row, "/gateway_cache_write_tokens"), Some(2000), "{row}");
    let u = client_usage(&text);
    assert_eq!(n(&u, "/input_tokens"), Some(2350), "{u}");
    assert_eq!(
        n(&u, "/input_tokens_details/cached_tokens"),
        Some(300),
        "{u}"
    );
    assert_eq!(
        n(&u, "/input_tokens_details/cache_write_tokens"),
        Some(0),
        "{u}"
    );
}

/// The client marked its own prompt: it asked for the writes, and is billed and shown true cache
/// writes, exactly as calling Anthropic directly.
/// claim: E7, BIL-6
/// defect: D76
#[tokio::test]
async fn client_cache_control_bills_true_cache_writes() {
    let up = ScriptedUpstream::reply(200, "application/json", ANTHROPIC_WRITE.to_owned()).await;
    let (row, text, _gw) = call(&up, "/v1/chat/completions", CHAT_MARKED).await;
    assert_eq!(n(&row, "/input_tokens"), Some(50), "{row}");
    assert_eq!(n(&row, "/cache_write_tokens"), Some(2000), "{row}");
    assert_eq!(n(&row, "/cache_read_tokens"), Some(300), "{row}");
    assert_eq!(
        n(&row, "/gateway_cache_write_tokens").unwrap_or(0),
        0,
        "{row}"
    );
    let u = client_usage(&text);
    assert_eq!(n(&u, "/prompt_tokens"), Some(2350), "{u}");
    assert_eq!(
        n(&u, "/prompt_tokens_details/cache_write_tokens"),
        Some(2000),
        "{u}"
    );
}
