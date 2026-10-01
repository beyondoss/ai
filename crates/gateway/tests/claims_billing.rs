//! Claim tests: every managed call writes exactly one `ai.usage` row, and the row's tokens are the
//! tokens the client was shown — per dialect, streaming and not, same-wire and translated.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::Value;
use std::time::Duration;

/// Usage as an SDK reads it off the response: `(input, output, cache_read)`.
///
/// Chat Completions, Messages and Responses spell the same three facts differently. For a stream,
/// the last value seen for each field wins, which is how every SDK accumulates (Anthropic's
/// `message_delta` carries the cumulative output count).
fn client_usage(text: &str) -> (u64, u64, u64) {
    let mut acc = (None, None, None);
    let mut read = |v: &Value| {
        let mut take = |usage: &Value| {
            if !usage.is_object() {
                return;
            }
            let num = |k: &str| usage.get(k).and_then(Value::as_u64);
            if let Some(n) = num("prompt_tokens").or_else(|| num("input_tokens")) {
                acc.0 = Some(n);
            }
            if let Some(n) = num("completion_tokens").or_else(|| num("output_tokens")) {
                acc.1 = Some(n);
            }
            let cached = usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .or_else(|| usage["input_tokens_details"]["cached_tokens"].as_u64())
                .or_else(|| num("cache_read_input_tokens"));
            if let Some(n) = cached {
                acc.2 = Some(n);
            }
        };
        take(&v["usage"]);
        take(&v["message"]["usage"]);
        take(&v["response"]["usage"]);
    };
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        read(&v);
    } else {
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data:")
                && let Ok(v) = serde_json::from_str::<Value>(data.trim())
            {
                read(&v);
            }
        }
    }
    (
        acc.0.expect("client saw an input count"),
        acc.1.expect("client saw an output count"),
        acc.2.unwrap_or(0),
    )
}

async fn post(gw: &Gateway, path: &str, auth: (&str, String), body: &str) -> (u16, String) {
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header(auth.0, auth.1)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(body.to_owned())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap())
}

/// The one row a served call wrote. Waits for it, then lets any straggler land before counting.
async fn the_one_row(gw: &Gateway) -> Value {
    wait_usage_rows(gw, 1, 5).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = usage_rows_of(gw);
    assert_eq!(rows.len(), 1, "one call, one row: {rows:#?}");
    rows.into_iter().next().unwrap()
}

/// Boot a gateway on `mode`, make one managed call, and assert the ledger agrees with the client.
async fn one_row_matches_the_client(
    mode: Mode,
    providers: &[&'static str],
    path: &str,
    bearer: bool,
    body: &str,
) {
    let (pubkey, sk) = test_keypair(90);
    let mock = MockUpstream::start(mode).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(providers)
        .start()
        .await;
    let key = billing_vkey(&sk, 90);
    let auth = if bearer {
        ("authorization", format!("Bearer {key}"))
    } else {
        ("x-api-key", key)
    };
    let (status, text) = post(&gw, path, auth, body).await;
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    let (input, output, cache_read) = client_usage(&text);
    let row = the_one_row(&gw).await;
    assert_eq!(row["input_tokens"].as_u64(), Some(input), "{row}\n{text}");
    assert_eq!(row["output_tokens"].as_u64(), Some(output), "{row}\n{text}");
    assert_eq!(
        row["cache_read_tokens"].as_u64(),
        Some(cache_read),
        "{row}\n{text}"
    );
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["tenant_id"].as_u64(), Some(90), "{row}");
}

const GPT_CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;
const GPT_CHAT_STREAM: &str =
    r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
const CLAUDE_CHAT: &str =
    r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#;
const CLAUDE_MESSAGES_STREAM: &str = r#"{"model":"claude-opus-4-8","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
const GPT_MESSAGES: &str =
    r#"{"model":"gpt-4o-mini","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
const GPT_RESPONSES: &str = r#"{"model":"gpt-4o-mini","input":"hi","store":false}"#;

/// claim: E1, B1
#[tokio::test]
async fn chat_on_a_gpt_row_writes_one_row_matching_the_client() {
    one_row_matches_the_client(
        Mode::Json,
        &["openai"],
        "/v1/chat/completions",
        true,
        GPT_CHAT,
    )
    .await;
}

/// claim: E1, B1
#[tokio::test]
async fn a_chat_stream_on_a_gpt_row_writes_one_row_matching_the_client() {
    one_row_matches_the_client(
        Mode::Sse,
        &["openai"],
        "/v1/chat/completions",
        true,
        GPT_CHAT_STREAM,
    )
    .await;
}

/// claim: E1, B1
#[tokio::test]
async fn chat_on_a_claude_row_writes_one_row_matching_the_client() {
    one_row_matches_the_client(
        Mode::AnthropicJson,
        &["anthropic", "openai", "openrouter"],
        "/v1/chat/completions",
        true,
        CLAUDE_CHAT,
    )
    .await;
}

/// claim: E2, B1
#[tokio::test]
async fn a_messages_stream_on_a_claude_row_writes_one_row_matching_the_client() {
    one_row_matches_the_client(
        Mode::AnthropicSse,
        &["anthropic", "openai", "openrouter"],
        "/v1/messages",
        false,
        CLAUDE_MESSAGES_STREAM,
    )
    .await;
}

/// claim: E2, B1
#[tokio::test]
async fn messages_on_a_gpt_row_writes_one_row_matching_the_client() {
    one_row_matches_the_client(
        Mode::Json,
        &["openai", "anthropic", "openrouter"],
        "/v1/messages",
        false,
        GPT_MESSAGES,
    )
    .await;
}

/// claim: E3, B1
#[tokio::test]
async fn one_shot_responses_on_a_gpt_row_writes_one_row_matching_the_client() {
    one_row_matches_the_client(
        Mode::Json,
        &["openai", "openrouter"],
        "/v1/responses",
        true,
        GPT_RESPONSES,
    )
    .await;
}

const ANTHROPIC_CACHED: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":10,"cache_read_input_tokens":4000,"cache_creation_input_tokens":0}}"#;
const OPENAI_CACHED: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o-mini-2024-07-18","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5000,"completion_tokens":10,"total_tokens":5010,"prompt_tokens_details":{"cached_tokens":4000}}}"#;

/// Cache reads the provider reported reach the client, and the row meters the same number.
/// claim: B3
#[tokio::test]
async fn cache_reads_reach_the_client_and_the_row_alike() {
    for (body, providers, path, request, bearer) in [
        (
            ANTHROPIC_CACHED,
            &["anthropic", "openai", "openrouter"][..],
            "/v1/messages",
            r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
            false,
        ),
        (
            OPENAI_CACHED,
            &["openai"][..],
            "/v1/chat/completions",
            GPT_CHAT,
            true,
        ),
    ] {
        let (pubkey, sk) = test_keypair(91);
        let mock = MockUpstream::start(Mode::Raw(200, "application/json", body)).await;
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .providers(providers)
            .start()
            .await;
        let key = billing_vkey(&sk, 91);
        let auth = if bearer {
            ("authorization", format!("Bearer {key}"))
        } else {
            ("x-api-key", key)
        };
        let (status, text) = post(&gw, path, auth, request).await;
        assert_eq!(status, 200, "{path}: {text}");
        let (_, _, client_cached) = client_usage(&text);
        assert_eq!(
            client_cached, 4000,
            "{path}: the client sees the cache read"
        );
        let row = the_one_row(&gw).await;
        assert_eq!(
            row["cache_read_tokens"].as_u64(),
            Some(client_cached),
            "{path}: {row}"
        );
    }
}

/// A BYO caller's token reaches the provider untouched, and the call writes no billing row.
/// claim: A1
#[tokio::test]
async fn byo_is_forwarded_untouched_and_bills_nothing() {
    let (pubkey, _sk) = test_keypair(92);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let (status, text) = post(
        &gw,
        "/v1/chat/completions",
        ("authorization", "Bearer sk-user-own".to_owned()),
        GPT_CHAT,
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let cap = mock.captured().expect("BYO reached the provider");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-user-own"));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        usage_rows_of(&gw).is_empty(),
        "BYO is not ours to bill: {}",
        gw.log()
    );
}
