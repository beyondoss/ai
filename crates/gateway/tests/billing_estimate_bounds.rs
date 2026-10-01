//! Billing estimates at their two bounds (BIL-20): never above what the provider counted, and
//! never zero once the provider took the request.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::time::Duration;

/// The prompt of the live fault trials (`verify/clients/py/fault_probe.py`). gpt-4o-mini reports
/// `prompt_tokens` 24 for it as a Chat request, the floor every estimate below must stay under.
const FAULT_PROMPT: &str = "Count from 1 to 20, separated by spaces. Output only the numbers.";
const FAULT_PROMPT_TOKENS: u64 = 24;

/// POST a stream, read its first chunk, hang up.
async fn stream_then_cancel(url: String, headers: &[(&str, String)], body: String) {
    let mut req = test_client()
        .post(url)
        .header("content-type", "application/json")
        .body(body);
    for (k, v) in headers {
        req = req.header(*k, v);
    }
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

fn bearer(key: &str) -> Vec<(&'static str, String)> {
    vec![("authorization", format!("Bearer {key}"))]
}

fn anthropic_headers(key: &str) -> Vec<(&'static str, String)> {
    vec![
        ("x-api-key", key.to_owned()),
        ("anthropic-version", "2023-06-01".to_owned()),
    ]
}

fn one_row(rows: &[serde_json::Value], gw: &Gateway) -> serde_json::Value {
    rows.first()
        .cloned()
        .unwrap_or_else(|| panic!("no ai.usage row; log:\n{}", gw.log()))
}

/// The SDKs' own request bodies for the fault prompt, cut short after a few events on a Chat
/// wire, where input is estimated from the request. The JSON around the prompt (keys, the model
/// id, `max_tokens`, `stream_options`) is not text the provider tokenizes, and counting it put a
/// short prompt's estimate at 40 (Chat) and 32 (Messages translated onto Chat) for 24 real tokens.
/// claim: BIL-20, B2
/// defect: D99
#[tokio::test]
async fn a_short_prompt_estimate_stays_under_the_provider_count() {
    let (pubkey, sk) = test_keypair(130);
    let mock = MockUpstream::start(Mode::StallSse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 130);
    // openai-python's body, then anthropic-python's (served by the Chat candidate, translated).
    let chat = format!(
        r#"{{"messages":[{{"role":"user","content":"{FAULT_PROMPT}"}}],"model":"gpt-4o-mini","max_tokens":80,"stream":true,"stream_options":{{"include_usage":true}}}}"#
    );
    let messages = format!(
        r#"{{"max_tokens":80,"messages":[{{"role":"user","content":"{FAULT_PROMPT}"}}],"model":"gpt-4o-mini","stream":true}}"#
    );
    stream_then_cancel(
        format!("{}/openai/v1/chat/completions", gw.url()),
        &bearer(&key),
        chat,
    )
    .await;
    stream_then_cancel(
        format!("{}/v1/messages", gw.url()),
        &anthropic_headers(&key),
        messages,
    )
    .await;

    let rows = wait_usage_rows(&gw, 2, 15).await;
    assert_eq!(rows.len(), 2, "{rows:?}");
    for row in rows {
        assert_eq!(row["usage_estimated"], true, "{row}");
        let input = row["input_tokens"].as_u64().unwrap();
        assert!(input > 0, "the prompt was sent; input is not zero: {row}");
        assert!(
            input <= FAULT_PROMPT_TOKENS,
            "estimated input {input} exceeds the provider's {FAULT_PROMPT_TOKENS}: {row}"
        );
    }
}

/// A head with chunked framing and no chunk: the provider accepted the stream and then went
/// silent before its first event.
fn accepted_then_silent() -> Vec<Step> {
    vec![
        Step::Write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                .to_vec(),
        ),
        Step::Sleep(Duration::from_secs(8)),
    ]
}

/// The provider answered 200 (it took the request, and bills the prompt) and then stalled before
/// the first body byte. The gateway's read timeout ends the stream; the row was a cut-short 0/0
/// because nothing showed the provider had started. A 2xx head is that proof. Both wires: the
/// Anthropic one gets no `message_start`, so its input is estimated too.
/// claim: B2, BIL-3, BIL-20
/// defect: D123
#[tokio::test]
async fn a_stream_accepted_then_silent_before_its_first_byte_is_billed_an_estimate() {
    for (seed, provider, path) in [
        (131u8, "openai", "/openai/v1/chat/completions"),
        (132u8, "anthropic", "/anthropic/v1/messages"),
    ] {
        let (pubkey, sk) = test_keypair(seed);
        let mock = ScriptedUpstream::start(|_, _| accepted_then_silent()).await;
        let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
            .providers(&[provider])
            .config_line("read_timeout_secs = 2")
            .start()
            .await;
        let key = billing_vkey(&sk, u64::from(seed));
        let headers = if provider == "openai" {
            bearer(&key)
        } else {
            anthropic_headers(&key)
        };
        let body = format!(
            r#"{{"model":"m","max_tokens":80,"stream":true,"messages":[{{"role":"user","content":"{FAULT_PROMPT}"}}]}}"#
        );
        let mut req = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("content-type", "application/json")
            .body(body);
        for (k, v) in &headers {
            req = req.header(*k, v);
        }
        // The head is relayed; the body then breaks off at the read timeout.
        if let Ok(resp) = req.send().await {
            let _ = resp.bytes().await;
        }
        assert_eq!(mock.hits(), 1, "{provider}: the upstream had the request");

        let row = one_row(&wait_usage_rows(&gw, 1, 15).await, &gw);
        assert_eq!(row["upstream_status"], 200, "{provider}: {row}");
        assert_eq!(row["outcome"], "cut_short", "{provider}: {row}");
        assert_eq!(row["usage_estimated"], true, "{provider}: {row}");
        let input = row["input_tokens"].as_u64().unwrap_or(0);
        assert!(
            (1..=FAULT_PROMPT_TOKENS).contains(&input),
            "{provider}: the accepted prompt is billed, under its real count: {row}"
        );
        assert_eq!(
            row["output_tokens"], 0,
            "{provider}: nothing generated was seen: {row}"
        );
    }
}

/// The provider has the whole request and keeps the connection alive, but no head comes before
/// `read_timeout_secs`: a long non-stream generation. The gateway gives up and answers 504, the
/// same moment a client giving up would be billed the prompt (D07): the provider cannot tell who
/// hung up, and bills it either way. The row was 0/0, so the SDK's retry of that 504 hid a billed
/// prompt (BIL-14).
/// claim: BIL-3, BIL-14, BIL-20
/// defect: D130
#[tokio::test]
async fn a_read_timeout_before_the_head_bills_the_delivered_prompt() {
    let (pubkey, sk) = test_keypair(133);
    let mock = ScriptedUpstream::start(|_, _| vec![Step::Sleep(Duration::from_secs(8))]).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .config_line("read_timeout_secs = 2")
        .start()
        .await;
    let body = format!(
        r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{FAULT_PROMPT}"}}]}}"#
    );
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(&sk, 133)),
        )
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 504);
    assert_eq!(mock.hits(), 1, "the upstream had the whole request");

    let row = one_row(&wait_usage_rows(&gw, 1, 15).await, &gw);
    assert_eq!(row["outcome"], "upstream_error", "{row}");
    assert_eq!(row["usage_estimated"], true, "{row}");
    let input = row["input_tokens"].as_u64().unwrap_or(0);
    assert!(
        (1..=FAULT_PROMPT_TOKENS).contains(&input),
        "the delivered prompt is billed, under its real count: {row}"
    );
    assert_eq!(row["output_tokens"], 0, "{row}");
}

/// The other half of D130's rule: a peer that closes the connection before any head, after it had
/// the request, told the gateway it is not answering. Real edges answer a failure they forwarded
/// with an HTTP error (Cloudflare 52x, Envoy's 503 local reply), so a bare close is a refusal and
/// is billed nothing, the side an unverifiable bill errs on (BIL-20).
/// claim: BIL-20
#[tokio::test]
async fn a_close_before_the_head_bills_nothing() {
    let (pubkey, sk) = test_keypair(134);
    let mock = ScriptedUpstream::start(|_, _| Vec::new()).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let body = format!(
        r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{FAULT_PROMPT}"}}]}}"#
    );
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(&sk, 134)),
        )
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);

    let row = one_row(&wait_usage_rows(&gw, 1, 15).await, &gw);
    assert_eq!(row["outcome"], "upstream_error", "{row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["input_tokens"], 0, "{row}");
    assert_eq!(row["output_tokens"], 0, "{row}");
}

/// POST the fault prompt as a non-stream Chat request on the OpenAI provider route and drain it.
async fn post_non_stream(gw: &Gateway, key: &str) {
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(format!(
            r#"{{"model":"m","max_tokens":80,"messages":[{{"role":"user","content":"{FAULT_PROMPT}"}}]}}"#
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await;
}

/// A non-stream 200 that ended cleanly with no usage block: OpenRouter's answer to a request whose
/// generation failed (`"usage": null` beside an empty message; live 2026-10-01 on
/// `x-ai/grok-4.20-multi-agent`, an 8-second answer billed 0/0), or a shape change. OpenRouter
/// documents that the upstream may still bill the prompt, and a stream that finished without usage
/// was already estimated (D123); the non-stream body wrote a zero row. Now it is estimated: input
/// from the prompt, under its real count, and the row says it is an estimate.
/// claim: BIL-20, BIL-12
/// defect: D195
#[tokio::test]
async fn a_non_stream_answer_without_usage_is_billed_an_estimate() {
    for (seed, body) in [
        (
            195u8,
            r#"{"id":"gen-1","object":"chat.completion","model":"x-ai/grok-4.20-multi-agent","choices":[{"index":0,"finish_reason":"error","message":{"role":"assistant","content":""}}],"usage":null}"#,
        ),
        (
            196u8,
            r#"{"id":"gen-2","object":"chat.completion","model":"x-ai/grok-4.20","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Paris is sunny."}}]}"#,
        ),
    ] {
        let (pubkey, sk) = test_keypair(seed);
        let up = ScriptedUpstream::reply(200, "application/json", body.to_owned()).await;
        let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
            .providers(&["openai"])
            .start()
            .await;
        post_non_stream(&gw, &billing_vkey(&sk, u64::from(seed))).await;
        let row = one_row(&wait_usage_rows(&gw, 1, 15).await, &gw);
        assert_eq!(row["upstream_status"], 200, "{row}");
        assert_eq!(row["usage_estimated"], true, "{row}");
        let input = row["input_tokens"].as_u64().unwrap_or(0);
        assert!(
            (1..=FAULT_PROMPT_TOKENS).contains(&input),
            "the prompt is billed, under its real count: {row}"
        );
    }
}

/// OpenRouter's error-in-200 for a non-streaming request: a body holding only an `error` object,
/// no answer (https://openrouter.ai/docs/api-reference/errors). Like a stream that carried only an
/// error event, it is not a generation we were billed for: the row stays 0/0, not an estimate.
/// claim: BIL-12, BIL-20
/// defect: D195
#[tokio::test]
async fn a_non_stream_error_in_200_is_not_billed() {
    let (pubkey, sk) = test_keypair(197);
    let up = ScriptedUpstream::reply(
        200,
        "application/json",
        r#"{"error":{"code":502,"message":"Provider returned error","metadata":{"provider_name":"xAI"}},"user_id":"u"}"#.to_owned(),
    )
    .await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    post_non_stream(&gw, &billing_vkey(&sk, 197)).await;
    let row = one_row(&wait_usage_rows(&gw, 1, 15).await, &gw);
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["input_tokens"], 0, "{row}");
    assert_eq!(row["output_tokens"], 0, "{row}");
}

/// The same error-in-200 whatever order its keys come in: a provider (or OpenRouter quoting one)
/// that writes `id`, `object` and `model` before `error` sends a body that is still only an error,
/// with no `choices`, `output` or `content`. It stays 0/0, not an estimate of the envelope. A body
/// that has an answer beside `"error": null` (a Responses object) is still billed as an answer.
/// claim: BIL-12, BIL-20
/// defect: D205
#[tokio::test]
async fn an_error_in_200_is_not_billed_whatever_its_key_order() {
    let (pubkey, sk) = test_keypair(205);
    let up = ScriptedUpstream::reply(
        200,
        "application/json",
        r#"{"id":"gen-1","object":"chat.completion","created":1,"model":"x-ai/grok-4","error":{"code":502,"message":"Provider returned error","metadata":{"provider_name":"xAI"}},"user_id":"u"}"#.to_owned(),
    )
    .await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    post_non_stream(&gw, &billing_vkey(&sk, 205)).await;
    let row = one_row(&wait_usage_rows(&gw, 1, 15).await, &gw);
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(row["input_tokens"], 0, "{row}");
    assert_eq!(row["output_tokens"], 0, "{row}");
}
