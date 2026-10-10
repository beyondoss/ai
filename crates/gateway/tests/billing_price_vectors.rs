//! The golden pricing vectors (`verify/pricing_vectors.json`) through the whole gateway: a mock
//! upstream answers with each vector's usage, a real managed request goes through the proxy, and
//! the `ai.usage` row must carry the vector's exact `cost_micros` / `price_micros` (or its
//! `price_status=unpriced` and reason). The offline test (`crates/providers/tests/
//! pricing_vectors.rs`) proves the pricer; this one proves the row feeds it the right facts.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::{Value, json};

fn vectors() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../verify/pricing_vectors.json"
    );
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    doc["vectors"].as_array().unwrap().clone()
}

fn vector(name: &str) -> Value {
    vectors()
        .into_iter()
        .find(|v| v["name"] == name)
        .unwrap_or_else(|| panic!("no vector {name}"))
}

fn n(row: &Value, k: &str) -> u64 {
    row[k].as_u64().unwrap_or(0)
}

fn tool(row: &Value, kind: &str) -> u64 {
    row["server_tools"]
        .as_str()
        .unwrap_or("")
        .split(',')
        .find_map(|kv| kv.strip_prefix(kind)?.strip_prefix('='))
        .map_or(0, |c| c.parse().unwrap())
}

/// The upstream's answer carrying exactly the vector row's usage, in the serving wire's shape.
fn upstream_body(row: &Value, upstream_model: &str) -> String {
    let opt = |k: &str| row.get(k).filter(|v| !v.is_null()).cloned();
    if row["usage_wire"] == "openai" {
        let mut usage = json!({
            "prompt_tokens": n(row, "input_tokens"),
            "completion_tokens": n(row, "output_tokens"),
            "total_tokens": n(row, "input_tokens") + n(row, "output_tokens"),
            "prompt_tokens_details": {
                "cached_tokens": n(row, "cache_read_tokens"),
                "cache_write_tokens": n(row, "cache_write_tokens"),
            },
        });
        let mut body = json!({
            "id": "chatcmpl-vector",
            "object": "chat.completion",
            "model": upstream_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        });
        if let Some(st) = opt("service_tier") {
            body["service_tier"] = st;
        }
        if let Some(host) = opt("served_by") {
            body["provider"] = host;
        }
        if tool(row, "web_search") > 0 {
            usage["server_tool_use"] = json!({"web_search_requests": tool(row, "web_search")});
        }
        let mut text = {
            body["usage"] = usage;
            serde_json::to_string(&body).unwrap()
        };
        // The reported cost goes in as the vector's own number text, exponent and all.
        if let Some(cost) = row["upstream_cost_usd"].as_str() {
            text = text.replacen(
                r#""prompt_tokens":"#,
                &format!(r#""cost":{cost},"prompt_tokens":"#),
                1,
            );
        }
        text
    } else {
        let cw = n(row, "cache_write_tokens");
        let c1 = n(row, "cache_write_1h_tokens");
        let mut usage = json!({
            "input_tokens": n(row, "input_tokens"),
            "output_tokens": n(row, "output_tokens"),
            "cache_read_input_tokens": n(row, "cache_read_tokens"),
            "cache_creation_input_tokens": cw,
            "cache_creation": {"ephemeral_5m_input_tokens": cw - c1, "ephemeral_1h_input_tokens": c1},
            "server_tool_use": {
                "web_search_requests": tool(row, "web_search"),
                "web_fetch_requests": tool(row, "web_fetch"),
                "code_execution_requests": tool(row, "code_execution"),
            },
        });
        for k in ["service_tier", "speed", "inference_geo"] {
            if let Some(v) = opt(k) {
                usage[k] = v;
            }
        }
        serde_json::to_string(&json!({
            "id": "msg_vector",
            "type": "message",
            "role": "assistant",
            "model": upstream_model,
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": usage,
        }))
        .unwrap()
    }
}

/// One vector through a gateway whose `provider` is the mock; the row's price must be the
/// vector's.
async fn through_gateway(name: &str, seed: u8, upstream_model: &str) {
    let v = vector(name);
    let row = &v["row"];
    let provider = row["provider"].as_str().unwrap();
    let body: &'static str = Box::leak(upstream_body(row, upstream_model).into_boxed_str());
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", body)).await;
    let (pubkey, sk) = test_keypair(seed);
    let provider_static: &'static str = Box::leak(provider.to_owned().into_boxed_str());
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&[provider_static])
        .start()
        .await;
    let key = billing_vkey(&sk, u64::from(seed));
    let req = match provider {
        "anthropic" => test_client()
            .post(format!("{}/anthropic/v1/messages", gw.url()))
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .body(
                json!({"model": upstream_model, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
                    .to_string(),
            ),
        "openai" => test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .body(json!({"model": upstream_model, "messages": [{"role": "user", "content": "hi"}]}).to_string()),
        "openrouter" => test_client()
            .post(format!("{}/openrouter/api/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .body(json!({"model": upstream_model, "messages": [{"role": "user", "content": "hi"}]}).to_string()),
        p => panic!("{name}: no request shape for {p}"),
    };
    let resp = req
        .header("content-type", "application/json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200, "{name}");
    let _ = resp.bytes().await;
    let got = usage_row_of(&gw).await;
    let want = &v["expect"];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let start = got["start_unix_secs"].as_u64().unwrap_or(0);
    assert!(
        start <= now && now - start < 60,
        "{name}: start_unix_secs is the request's start: {got}"
    );
    let unpriced_metric = gw
        .metrics()
        .await
        .lines()
        .find(|l| l.starts_with("ai_usage_unpriced_total "))
        .map(|l| l.rsplit(' ').next().unwrap_or("").to_owned());
    let want_metric = if want["unpriced"].is_string() {
        "1"
    } else {
        "0"
    };
    assert_eq!(
        unpriced_metric.as_deref(),
        Some(want_metric),
        "{name}: ai_usage_unpriced_total"
    );
    assert_eq!(
        got["rate_version"],
        providers::rates::RATE_VERSION,
        "{name}: {got}"
    );
    assert_eq!(got["price_model"], row["price_model"], "{name}: {got}");
    if let Some(reason) = want["unpriced"].as_str() {
        assert_eq!(got["price_status"], "unpriced", "{name}: {got}");
        assert_eq!(got["price_reason"], reason, "{name}: {got}");
        assert!(
            got.get("cost_micros").is_none(),
            "{name}: never a zero: {got}"
        );
        assert!(
            got.get("price_micros").is_none(),
            "{name}: never a zero: {got}"
        );
    } else {
        assert_eq!(got["price_status"], want["status"], "{name}: {got}");
        assert_eq!(got["cost_micros"], want["cost_micros"], "{name}: {got}");
        assert_eq!(got["price_micros"], want["price_micros"], "{name}: {got}");
        assert_eq!(got["cost_basis"], want["basis"], "{name}: {got}");
        let detail = got["price_detail"].as_str().unwrap_or("");
        assert!(
            detail.starts_with(&format!("class={}", want["price_class"].as_str().unwrap())),
            "{name}: {got}"
        );
        assert!(got.get("price_reason").is_none(), "{name}: {got}");
    }
}

/// claim: BIL-24
#[tokio::test]
async fn anthropic_basic_prices_through_the_gateway() {
    through_gateway("anthropic_basic", 170, "claude-opus-4-8").await;
}

/// Cache reads, 5-minute and 1-hour writes on the Anthropic wire.
/// claim: BIL-24
#[tokio::test]
async fn anthropic_cache_writes_price_through_the_gateway() {
    through_gateway("anthropic_cache_read_5m_1h", 171, "claude-opus-4-8").await;
}

/// Fast mode × US-only inference × caches × web search, all from the served usage.
/// claim: BIL-24
#[tokio::test]
async fn anthropic_everything_together_prices_through_the_gateway() {
    through_gateway("anthropic_everything_together", 172, "claude-opus-5-5").await;
}

/// A dimension the card does not sell is refused on the row, never a zero.
/// claim: BIL-24
#[tokio::test]
async fn unpriced_row_says_why() {
    through_gateway("anthropic_fast_unsupported", 173, "claude-opus-4-7").await;
}

/// The OpenAI wire: cached tokens inside `prompt_tokens`, counted once.
/// claim: BIL-24
#[tokio::test]
async fn openai_cached_prices_through_the_gateway() {
    through_gateway("openai_wire_cached", 174, "gpt-5.5").await;
}

/// OpenAI's priority (Fast mode) tier, from the served `service_tier`.
/// claim: BIL-24
#[tokio::test]
async fn openai_priority_prices_through_the_gateway() {
    through_gateway("openai_priority", 175, "gpt-5.5").await;
}

/// OpenRouter: the reported `usage.cost` plus the credit fee is the cost; the Anthropic card on
/// OpenRouter's counts (cache writes inside `prompt_tokens`) is the price.
/// claim: BIL-24
#[tokio::test]
async fn openrouter_reported_cost_prices_through_the_gateway() {
    through_gateway("openrouter_reported_cost", 176, "anthropic/claude-opus-4.8").await;
}

/// OpenRouter native web search, counted twice by OpenRouter (`web_search_requests` and
/// `tool_calls`), priced once.
/// claim: BIL-24
#[tokio::test]
async fn openrouter_native_search_prices_through_the_gateway() {
    through_gateway(
        "openrouter_native_search",
        177,
        "anthropic/claude-sonnet-4.6",
    )
    .await;
}

/// `start_unix_secs` is when the request started, not when its row was written: a repricer reads
/// the time-of-day tier (DeepSeek's off-peak) at the start. A slow upstream holds the request
/// for 2.5 s, so the two differ.
/// claim: BIL-24
#[tokio::test]
async fn start_unix_secs_is_the_request_start() {
    let mock = MockUpstream::start(Mode::Slow(2_500)).await;
    let (pubkey, sk) = test_keypair(178);
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let sent = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(&sk, 178)),
        )
        .header("content-type", "application/json")
        .body(
            json!({"model": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        )
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await;
    let row = usage_row_of(&gw).await;
    let start = row["start_unix_secs"].as_u64().unwrap();
    assert!(
        start + 1 >= sent && start <= sent + 1,
        "sent at {sent}, row says the request started at {start}: {row}"
    );
}
