//! Verify phase 0, billing: requests that end badly after the provider already has them — a
//! client cancel before the response head, a non-stream body cut off midway, a large body whose
//! upstream goes silent after receiving it.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::time::Duration;

const CHAT_BODY: &str = r#"{"model":"gpt-4o","messages":[{"role":"system","content":"You are a careful assistant."},{"role":"user","content":"Write a long essay about the history of TCP congestion control, please."}]}"#;

/// The provider accepted the request and is thinking (a long o-series turn); the client gives up
/// before the response head. The provider bills that work, so the row must not be zero.
/// claim: BIL-3
/// defect: D07
#[tokio::test]
async fn a_cancel_before_the_response_head_is_billed_an_estimate() {
    let (pubkey, sk) = test_keypair(71);
    // Drains the body (the provider has it), then never answers within the client's patience and
    // finally closes without a response — so no outcome of the gateway's wait yields real usage.
    let mock = ScriptedUpstream::start(|_, _| vec![Step::Sleep(Duration::from_secs(8))]).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(1500))
        .build()
        .unwrap();
    let res = client
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", billing_vkey(&sk, 71)))
        .header("content-type", "application/json")
        .body(CHAT_BODY)
        .send()
        .await;
    assert!(res.is_err(), "the client must time out before the head");
    assert_eq!(mock.hits(), 1, "the upstream received the whole request");

    let rows = wait_usage_rows(&gw, 1, 15).await;
    let row = rows
        .first()
        .unwrap_or_else(|| panic!("no ai.usage row; log:\n{}", gw.log()));
    assert!(
        row["input_tokens"].as_u64().unwrap_or(0) > 0,
        "the provider has the prompt; input must be estimated: {row}"
    );
    assert_eq!(row["usage_estimated"], true, "{row}");
}

/// A non-stream 2xx whose body is cut off midway: the provider generated (and bills) the whole
/// answer, the gateway saw half of it and no usage block.
/// claim: BIL-3
/// defect: D07
#[tokio::test]
async fn a_non_stream_body_cut_off_midway_is_billed_an_estimate() {
    let (pubkey, sk) = test_keypair(72);
    let content = "word ".repeat(2000);
    let full = format!(
        r#"{{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{{"index":0,"message":{{"role":"assistant","content":"{content}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":40,"completion_tokens":2000,"total_tokens":2040}}}}"#
    );
    let mock = ScriptedUpstream::start(move |_, _| {
        let half = &full.as_bytes()[..full.len() / 2];
        vec![
            Step::Write(http_head(200, "application/json", Some(full.len()))),
            Step::Write(half.to_vec()),
        ]
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let res = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", billing_vkey(&sk, 72)))
        .header("content-type", "application/json")
        .body(CHAT_BODY)
        .send()
        .await;
    if let Ok(resp) = res {
        let _ = resp.bytes().await; // errors or truncates; either way the client got half
    }

    let rows = wait_usage_rows(&gw, 1, 10).await;
    let row = rows
        .first()
        .unwrap_or_else(|| panic!("no ai.usage row; log:\n{}", gw.log()));
    assert!(
        row["input_tokens"].as_u64().unwrap_or(0) > 0,
        "input must be estimated: {row}"
    );
    assert!(
        row["output_tokens"].as_u64().unwrap_or(0) > 0,
        "half an answer was relayed; output must be estimated: {row}"
    );
    assert_eq!(row["usage_estimated"], true, "{row}");
}

/// `{"messages":[…200 KiB…],"model":"gpt-4o-mini"}`: past the 64 KiB replay buffer, so the gateway
/// relays it as a `FullBody` subrequest.
fn big_chat() -> String {
    let filler = "x".repeat(200 * 1024);
    format!(r#"{{"messages":[{{"role":"user","content":"{filler}"}}],"model":"gpt-4o-mini"}}"#)
}

/// A large-body request whose upstream received the whole body and then went silent until the read
/// timeout. The provider is (in reality) generating and billing; sending it again duplicates that.
/// At most one request per candidate once the body was delivered — and, by the decided policy, no
/// failover to the next candidate either: the fallback would be a second generation of the same
/// request. The walk ends with a JSON 504.
/// claim: BIL-14, REL-21
/// defect: D09
#[tokio::test]
async fn a_large_body_is_not_resent_to_a_candidate_that_received_it() {
    let (pubkey, sk) = test_keypair(73);
    let silent = || ScriptedUpstream::start(|_, _| vec![Step::Sleep(Duration::from_secs(30))]);
    let primary = silent().await;
    let fallback = silent().await;
    let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .config_line("read_timeout_secs = 2")
        .start()
        .await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(25))
        .build()
        .unwrap();
    let res = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", billing_vkey(&sk, 73)))
        .header("content-type", "application/json")
        .body(big_chat())
        .send()
        .await;
    let (status, text) = match res {
        Ok(r) => (
            Some(r.status().as_u16()),
            r.text().await.unwrap_or_default(),
        ),
        Err(_) => (None, String::new()),
    };
    // (1, 0), stricter than the original "at most once per candidate" (1, 1): the fallback must not
    // be handed a body the primary may already be generating from.
    assert_eq!(
        (primary.hits(), fallback.hits()),
        (1, 0),
        "each candidate must receive the body at most once (client saw {status:?}); log:\n{}",
        gw.log()
    );
    assert_eq!(status, Some(504), "{text}");
    let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    assert!(json["error"]["message"].is_string(), "a JSON error: {text}");
}
