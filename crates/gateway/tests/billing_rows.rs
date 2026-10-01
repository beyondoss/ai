//! Verify phase 0, billing: what an `ai.usage` row carries and whether it is emitted at all —
//! the log filter, priced variants, and the outcome of a zero-cost ending.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;

async fn post(url: String, auth: (&str, String), body: &str, extra: &[(&str, &str)]) -> u16 {
    let mut req = test_client()
        .post(url)
        .header(auth.0, auth.1)
        .header("content-type", "application/json");
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    let resp = req.body(body.to_owned()).send().await.unwrap();
    let status = resp.status().as_u16();
    let _ = resp.bytes().await;
    status
}

/// Billing rows are not diagnostics: an operator turning the log level down to `warn` must not
/// silently stop billing.
/// claim: BIL-4
/// defect: D08
#[tokio::test]
#[ignore = "D08 reproduced: with AI_LOG=warn the global EnvFilter drops every ai.usage row"]
async fn ai_log_warn_still_emits_usage_rows() {
    let (pubkey, sk) = test_keypair(81);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .env("AI_LOG", "warn")
        .start()
        .await;
    let status = post(
        format!("{}/openai/v1/chat/completions", gw.url()),
        ("authorization", format!("Bearer {}", billing_vkey(&sk, 81))),
        r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(mock.hits(), 1);
    let rows = wait_usage_rows(&gw, 1, 5).await;
    assert_eq!(
        rows.len(),
        1,
        "a served managed request must write its billing row; log:\n{}",
        gw.log()
    );
    assert_eq!(rows[0]["input_tokens"].as_u64(), Some(11), "{}", rows[0]);
}

/// A field of `row` whose name contains `needle`, as a number.
fn field_containing(row: &serde_json::Value, needle: &str) -> Option<u64> {
    row.as_object()?
        .iter()
        .find(|(k, _)| k.contains(needle))
        .and_then(|(_, v)| v.as_u64())
}

/// Anthropic prices 1-hour cache writes at 2× input (5-minute ones at 1.25×) and web search per
/// call. Both are on the response; the row must carry them for the bill to be computable.
/// claim: BIL-10, BIL-11
/// defect: D24
#[tokio::test]
#[ignore = "D24 reproduced: the ai.usage row has no field for cache_creation.ephemeral_1h_input_tokens or server_tool_use.web_search_requests"]
async fn priced_variants_and_server_tool_calls_reach_the_row() {
    let (pubkey, sk) = test_keypair(82);
    let mock = MockUpstream::start(Mode::Raw(
        200,
        "application/json",
        r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":10,"cache_read_input_tokens":0,"cache_creation_input_tokens":2000,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":2000},"server_tool_use":{"web_search_requests":3}}}"#,
    ))
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let status = post(
        format!("{}/anthropic/v1/messages", gw.url()),
        ("x-api-key", billing_vkey(&sk, 82)),
        r#"{"model":"claude-opus-4-8","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}"#,
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["cache_write_tokens"].as_u64(), Some(2000), "{row}");
    assert_eq!(
        field_containing(&row, "1h"),
        Some(2000),
        "1-hour cache writes must be distinguishable from 5-minute ones: {row}"
    );
    assert_eq!(
        field_containing(&row, "web_search"),
        Some(3),
        "per-call web search fees need the call count: {row}"
    );
}

/// The upstream outcome carried by a row: any of the plausible field names.
fn outcome(row: &serde_json::Value) -> Option<&serde_json::Value> {
    ["status", "upstream_status", "outcome", "http_status"]
        .iter()
        .find_map(|k| row.get(*k))
}

/// A row for a request that cost nothing (the upstream 5xx'd; then every candidate was refused by
/// an open breaker) must say so, so a consumer can tell it from a real zero-token 2xx call — and it
/// must not name a provider as having served when none was called.
/// claim: BIL-12
/// defect: D26
#[tokio::test]
#[ignore = "D26 reproduced: rows for a 5xx and for breaker exhaustion carry no status/outcome field"]
async fn zero_cost_endings_carry_their_outcome() {
    let (pubkey, sk) = test_keypair(83);
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Status(500)).await;
    let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .circuit_breaker_threshold(1)
        .start()
        .await;
    let key = billing_vkey(&sk, 83);
    let body = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;
    let url = format!("{}/v1/chat/completions", gw.url());
    // 1: both candidates answer 500 → both breakers open (threshold 1).
    let first = post(
        url.clone(),
        ("authorization", format!("Bearer {key}")),
        body,
        &[],
    )
    .await;
    let rows = wait_usage_rows(&gw, 1, 5).await;
    let hits_after_first = (primary.hits(), fallback.hits());
    // 2: every breaker open → no provider is called at all.
    let second = post(url, ("authorization", format!("Bearer {key}")), body, &[]).await;
    let rows2 = wait_usage_rows(&gw, rows.len() + 1, 3).await;
    let hits_after_second = (primary.hits(), fallback.hits());

    let evidence = format!(
        "statuses {first}/{second}; hits {hits_after_first:?}→{hits_after_second:?}; rows:\n{}",
        rows2
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(first >= 500, "{evidence}");
    let r1 = rows2
        .first()
        .unwrap_or_else(|| panic!("no row: {evidence}"));
    assert_eq!(r1["input_tokens"].as_u64(), Some(0), "{evidence}");
    assert!(
        outcome(r1).is_some_and(|s| s.as_u64() == Some(500) || s.is_string()),
        "a 5xx row must carry its upstream status: {evidence}"
    );
    if let Some(r2) = rows2.get(1) {
        assert!(
            outcome(r2).is_some(),
            "a breaker-exhausted row must carry an outcome: {evidence}"
        );
        if hits_after_second == hits_after_first {
            assert!(
                r2.get("provider").is_none_or(|p| p.is_null()),
                "no provider was called, yet the row names one: {evidence}"
            );
        }
    }
}
