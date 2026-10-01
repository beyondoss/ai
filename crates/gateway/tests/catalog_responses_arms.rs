//! End-to-end: which Responses upstream a catalog walk lands on, and what it tells it.
//!
//! - Grok rows reach xAI over `/v1/responses`, never storing a response the client did not ask to
//!   store (xAI stores every response for 30 days by default).
//! - A Messages client offering more tools than OpenAI Chat Completions takes (128) on a GPT row
//!   walks the row's Responses arm, which takes them.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::{Value, json};

async fn post(
    gw: &Gateway,
    key: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &Value,
) -> reqwest::Response {
    let mut req = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_string());
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    req.send().await.unwrap()
}

/// A Responses body as OpenAI and xAI answer one.
const RESPONSE: &str = r#"{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":30,"input_tokens_details":{"cached_tokens":0},"output_tokens":9,"output_tokens_details":{"reasoning_tokens":8},"total_tokens":39}}"#;

fn sent(mock: &MockUpstream) -> (String, Value) {
    let c = mock.captured().unwrap();
    (c.path, serde_json::from_slice(&c.body).unwrap())
}

/// xAI's Responses API stores every response for 30 days unless the request says `store: false`
/// (https://docs.x.ai/developers/model-capabilities/text/comparison.md: "default: true"). A grok
/// row has no Responses arm, so a Responses client's walk onto xAI is a one-shot (session state
/// there is a 400), and a Chat or Messages client asked for nothing to be kept: each reaches xAI
/// with `store: false`, whatever spelling of "not set" the client used. A Chat client that asked
/// to store keeps its choice; a Responses client's explicit `store: true` is refused.
/// claim: E3, CAT-5
/// defect: D145
#[tokio::test]
async fn a_one_shot_on_a_grok_row_is_not_stored_at_xai() {
    let (pubkey, sk) = test_keypair(75);
    let xai = MockUpstream::start(Mode::Raw(200, "application/json", RESPONSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &xai.authority(), &b64(&pubkey))
        .providers(&["xai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 75);
    for (model, path, body) in [
        (
            "grok-4.3",
            "/v1/responses",
            json!({"model": "grok-4.3", "input": "hi"}),
        ),
        (
            "grok-4.3",
            "/v1/responses",
            json!({"model": "grok-4.3", "store": null, "input": "hi"}),
        ),
        (
            "grok-4.20-multi-agent",
            "/v1/responses",
            json!({"model": "grok-4.20-multi-agent", "input": "hi", "store": false}),
        ),
        (
            "grok-4.6",
            "/v1/chat/completions",
            json!({"model": "grok-4.6", "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (
            "grok-build-0.1",
            "/v1/messages",
            json!({"model": "grok-build-0.1", "max_tokens": 16,
                   "messages": [{"role": "user", "content": "hi"}]}),
        ),
    ] {
        let resp = post(&gw, &key, path, &[], &body).await;
        assert_eq!(resp.status().as_u16(), 200, "{model} {path}");
        let (upstream, v) = sent(&xai);
        assert_eq!(upstream, "/v1/responses", "{model} {path}");
        assert_eq!(v["store"], false, "{model} {path}: {v}");
        assert_eq!(v["model"], model, "{model} {path}: {v}");
        let raw = String::from_utf8(xai.captured().unwrap().body).unwrap();
        assert_eq!(raw.matches("\"store\"").count(), 1, "{path}: {raw}");
    }

    // A Chat client that asked to store keeps its choice.
    let stored = json!({"model": "grok-4.3", "store": true,
                        "messages": [{"role": "user", "content": "hi"}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &[], &stored).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(sent(&xai).1["store"], true);

    // Responses session state on a row without a Responses arm is still a 400, before xAI.
    let hits = xai.hits();
    let resp = post(
        &gw,
        &key,
        "/v1/responses",
        &[],
        &json!({"model": "grok-4.3", "input": "hi", "store": true}),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 400);
    assert_eq!(xai.hits(), hits);

    // So is an `item_reference` that stands for an earlier turn (the AI SDK's default for an
    // earlier answer): sent as `store: false`, nothing at xAI holds the item, and xAI would 422 it
    // (D175).
    let resp = post(
        &gw,
        &key,
        "/v1/responses",
        &[],
        &json!({"model": "grok-4.3", "input": [
            {"role": "user", "content": "hi"},
            {"type": "item_reference", "id": "msg_1"},
            {"role": "user", "content": "again"},
        ]}),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 400);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("item_reference") && text.contains("store: false"),
        "{text}"
    );
    assert_eq!(xai.hits(), hits);

    // A tool step's `item_reference` (the preamble of the step that made the call) is cut out
    // before xAI; the call and its output go as sent (D175).
    let resp = post(
        &gw,
        &key,
        "/v1/responses",
        &[],
        &json!({"model": "grok-4.3", "input": [
            {"role": "user", "content": "weather in Paris?"},
            {"type": "item_reference", "id": "msg_1"},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ]}),
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let (_, v) = sent(&xai);
    assert_eq!(v["store"], false, "{v}");
    assert_eq!(
        v["input"],
        json!([
            {"role": "user", "content": "weather in Paris?"},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ]),
        "{v}"
    );
}

/// `n` small function tools in Anthropic's shape, each `pad` bytes of description long.
fn messages_tools(n: usize, pad: usize) -> Vec<Value> {
    (0..n)
        .map(|i| {
            json!({"name": format!("tool_{i}"), "description": "x".repeat(pad),
                   "input_schema": {"type": "object", "properties": {}}})
        })
        .collect()
}

/// OpenAI Chat Completions refuses more than 128 tools (400 "Expected an array with maximum length
/// 128"); its Responses API takes 600. A Messages client (Claude Code with MCP servers) on a GPT row
/// whose primary is Chat Completions was translated onto Chat and refused. With more than 128 tools
/// it now walks the row's Responses arm, translated onto `/v1/responses`, whether the row came
/// from the body or a header and whether the body is small or past the 64 KiB replay buffer. At
/// 128 it keeps the Chat primary, and a Chat client keeps its own endpoint (OpenAI's own limit).
/// claim: TOOL-1
/// defect: D131
#[tokio::test]
async fn more_tools_than_chat_takes_walk_the_responses_arm() {
    let (pubkey, sk) = test_keypair(76);
    let openai = MockUpstream::start(Mode::Raw(200, "application/json", RESPONSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 76);
    let user = json!([{"role": "user", "content": "Call the last tool."}]);
    let header = [("x-beyond-model", "gpt-5-mini")];
    for (tools, pad, headers, want) in [
        (129, 8, &[][..], "/v1/responses"),
        (129, 8, &header[..], "/v1/responses"),
        // Past the 64 KiB body peek: the full-body relay decides the same way.
        (200, 400, &[][..], "/v1/responses"),
        (128, 8, &[][..], "/v1/chat/completions"),
        (128, 8, &header[..], "/v1/chat/completions"),
    ] {
        let body = json!({"model": "gpt-5-mini", "max_tokens": 64, "messages": user,
                          "tools": messages_tools(tools, pad)});
        let _ = post(&gw, &key, "/v1/messages", headers, &body).await;
        let (path, v) = sent(&openai);
        assert_eq!(
            path,
            want,
            "{tools} tools, {} bytes",
            body.to_string().len()
        );
        if want == "/v1/responses" {
            assert_eq!(v["tools"].as_array().map(Vec::len), Some(tools), "{v}");
            assert_eq!(v["store"], false, "{v}");
        }
    }
    // The translated Responses answer reaches the Messages client.
    let body = json!({"model": "gpt-5-mini", "max_tokens": 64, "messages": user,
                      "tools": messages_tools(150, 8)});
    let resp = post(&gw, &key, "/v1/messages", &[], &body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["type"], "message", "{v}");

    // A Chat client keeps Chat Completions.
    let chat = json!({"model": "gpt-5-mini", "messages": user,
                      "tools": (0..129).map(|i| json!({"type": "function", "function": {
                          "name": format!("tool_{i}"), "parameters": {"type": "object"}}}))
                          .collect::<Vec<_>>()});
    let _ = post(&gw, &key, "/v1/chat/completions", &[], &chat).await;
    assert_eq!(sent(&openai).0, "/v1/chat/completions");
}
