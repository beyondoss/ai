//! End-to-end: `item_reference` input items on `/v1/responses` (D175).
//!
//! The Vercel AI SDK's default Responses model sends each earlier assistant text or reasoning item
//! as `{"type":"item_reference","id":…}` and each tool call in full. A row with no Responses arm
//! has nothing that holds the item, and the gateway stores no customer content. So a reference in
//! a tool step (preamble or reasoning of the step that made the call) is dropped and the call and
//! its output go in full; a reference that stands for an earlier turn is a 400 naming it, before
//! any upstream. A GPT row relays every reference to OpenAI's Responses arm as sent.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use serde_json::{Value, json};

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

async fn post(gw: &Gateway, sk: &ed25519_dalek::SigningKey, body: &Value) -> (u16, String) {
    let resp = test_client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(sk)))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(body).unwrap())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap())
}

async fn claude_gateway(mode: Mode) -> (MockUpstream, Gateway, ed25519_dalek::SigningKey) {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(mode).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;
    (mock, gw, sk)
}

fn reference(id: &str) -> Value {
    json!({"type": "item_reference", "id": id})
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn call(id: &str, city: &str) -> Value {
    json!({"type": "function_call", "call_id": id, "name": "weather",
           "arguments": format!("{{\"city\":\"{city}\"}}")})
}

fn output(id: &str, text: &str) -> Value {
    json!({"type": "function_call_output", "call_id": id, "output": text})
}

fn weather_tool() -> Value {
    json!([{"type": "function", "name": "weather", "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}])
}

/// The Messages body's `tool_use` ids and `tool_result` texts, in order.
fn tool_blocks(got: &Value) -> (Vec<String>, Vec<String>) {
    let (mut uses, mut results) = (Vec::new(), Vec::new());
    for m in got["messages"].as_array().unwrap() {
        for b in m["content"].as_array().into_iter().flatten() {
            match b["type"].as_str() {
                Some("tool_use") => uses.push(b["id"].as_str().unwrap().to_owned()),
                Some("tool_result") => results.push(b["content"].to_string()),
                _ => {}
            }
        }
    }
    (uses, results)
}

/// Step 2 of an AI SDK tool loop whose model wrote a preamble before its tool call: the preamble
/// comes back as an `item_reference`, the call and its output in full. On a Claude row it must
/// succeed, with the call and the output reaching the upstream intact and no reference in sight
/// (under option B it was a 400, breaking every such loop).
/// claim: E3, T1
/// defect: D175
#[tokio::test]
async fn a_tool_step_preamble_reference_is_dropped_and_the_loop_succeeds() {
    let (mock, gw, sk) = claude_gateway(Mode::AnthropicJson).await;
    let body = json!({"model": "claude-opus-4-8", "tools": weather_tool(), "input": [
        user("What's the weather in Paris?"),
        reference("msg_preamble"),
        call("call_1", "Paris"),
        output("call_1", "Sunny in Paris, 31C"),
    ]});
    let (status, text) = post(&gw, &sk, &body).await;
    assert_eq!(status, 200, "{text}");
    let got: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
    let raw = got.to_string();
    assert!(!raw.contains("item_reference"), "{raw}");
    assert!(!raw.contains("msg_preamble"), "{raw}");
    let (uses, results) = tool_blocks(&got);
    assert_eq!(uses, ["call_1"], "{got}");
    assert_eq!(results.len(), 1, "{got}");
    assert!(results[0].contains("Sunny in Paris, 31C"), "{got}");
    let tool_use = got["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().into_iter().flatten())
        .find(|b| b["type"] == "tool_use")
        .unwrap();
    assert_eq!(tool_use["input"], json!({"city": "Paris"}), "{got}");
}

/// A reasoning model's step: the reasoning item comes back as an `item_reference` ahead of the
/// tool call. Dropped like a preamble; the loop succeeds.
/// claim: E3, T1
/// defect: D175
#[tokio::test]
async fn a_reasoning_reference_before_a_function_call_is_dropped() {
    let (mock, gw, sk) = claude_gateway(Mode::AnthropicJson).await;
    let body = json!({"model": "claude-opus-4-8", "tools": weather_tool(), "input": [
        user("What's the weather in Paris?"),
        reference("rs_reasoning"),
        call("call_1", "Paris"),
        output("call_1", "Sunny in Paris, 31C"),
    ]});
    let (status, text) = post(&gw, &sk, &body).await;
    assert_eq!(status, 200, "{text}");
    let got: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
    assert!(!got.to_string().contains("rs_reasoning"), "{got}");
    assert_eq!(tool_blocks(&got).0, ["call_1"], "{got}");
}

/// The shapes a tool step takes: several references in a row, a reference then a reasoning item
/// then the call, parallel calls, text after the call (a reference between the call and its
/// output), and two steps of one loop. Each is dropped, and every call and output arrives.
/// claim: E3, T1
/// defect: D175
#[tokio::test]
async fn every_shape_of_tool_step_reference_is_dropped() {
    let (mock, gw, sk) = claude_gateway(Mode::AnthropicJson).await;
    let reasoning = json!({"type": "reasoning", "id": "rs_1", "summary": []});
    let cases: Vec<(&str, Vec<Value>, Vec<&str>)> = vec![
        (
            "references in a row",
            vec![
                user("q"),
                reference("msg_a"),
                reference("rs_b"),
                call("c1", "Paris"),
                output("c1", "sunny"),
            ],
            vec!["c1"],
        ),
        (
            "reference, reasoning, call",
            vec![
                user("q"),
                reference("msg_a"),
                reasoning.clone(),
                call("c1", "Paris"),
                output("c1", "sunny"),
            ],
            vec!["c1"],
        ),
        (
            "parallel calls",
            vec![
                user("q"),
                reference("msg_a"),
                call("c1", "Paris"),
                call("c2", "Rome"),
                output("c1", "sunny"),
                output("c2", "rain"),
            ],
            vec!["c1", "c2"],
        ),
        (
            "reference before the output",
            vec![
                user("q"),
                call("c1", "Paris"),
                reference("msg_after"),
                output("c1", "sunny"),
            ],
            vec!["c1"],
        ),
        (
            "two steps",
            vec![
                user("q"),
                reference("msg_a"),
                call("c1", "Paris"),
                output("c1", "sunny"),
                reference("msg_b"),
                call("c2", "Rome"),
                output("c2", "rain"),
            ],
            vec!["c1", "c2"],
        ),
    ];
    for (what, input, calls) in cases {
        let outputs = input
            .iter()
            .filter(|i| i["type"] == "function_call_output")
            .count();
        let body = json!({"model": "claude-opus-4-8", "tools": weather_tool(), "input": input});
        let (status, text) = post(&gw, &sk, &body).await;
        assert_eq!(status, 200, "{what}: {text}");
        let got: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
        assert!(!got.to_string().contains("item_reference"), "{what}: {got}");
        let (uses, results) = tool_blocks(&got);
        assert_eq!(uses, calls, "{what}: {got}");
        assert_eq!(results.len(), outputs, "{what}: {got}");
    }
}

/// A reference that stands for an earlier turn (followed by a user message, at the end of input,
/// the whole input, or before a tool output with no call in its step) would be a turn the model
/// answers without: a 400 that names `item_reference` and the remedy, and the upstream is never
/// contacted.
/// claim: E3
/// defect: D175
#[tokio::test]
async fn a_turn_reference_is_refused_before_the_upstream() {
    let (mock, gw, sk) = claude_gateway(Mode::AnthropicJson).await;
    let reasoning = json!({"type": "reasoning", "id": "rs_1", "summary": []});
    let cases: Vec<(&str, Vec<Value>)> = vec![
        (
            "before a user message",
            vec![user("a"), reference("msg_1"), user("b")],
        ),
        ("at the end", vec![user("a"), reference("msg_1")]),
        ("reference only", vec![reference("msg_1")]),
        (
            "reference and reasoning, then a user message",
            vec![user("a"), reference("msg_1"), reasoning, user("b")],
        ),
        (
            "an answer after a tool loop",
            vec![
                user("a"),
                reference("msg_0"),
                call("c1", "Paris"),
                output("c1", "sunny"),
                reference("msg_answer"),
                user("b"),
            ],
        ),
        (
            "before an output with no call in its step",
            vec![user("a"), reference("msg_1"), output("c1", "sunny")],
        ),
    ];
    for (what, input) in cases {
        let body = json!({"model": "claude-opus-4-8", "tools": weather_tool(), "input": input});
        let (status, text) = post(&gw, &sk, &body).await;
        assert_eq!(status, 400, "{what}: {text}");
        assert!(
            text.contains("item_reference") && text.contains("store: false"),
            "{what}: names the item and the remedy: {text}"
        );
        assert_eq!(mock.hits(), 0, "{what}: the upstream was contacted");
    }
}

/// A GPT row relays every `item_reference`, a tool step's and a turn's alike, to OpenAI's
/// Responses arm byte for byte, less the tenant binding on each id (`signed_id.rs`): OpenAI holds
/// the items, and gets its own ids back.
/// claim: E3
/// defect: D175
#[tokio::test]
async fn a_gpt_row_relays_item_references_unchanged() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;
    let issued = dev_id_signer().sign(42, "msg_1");
    let inputs = |id: &str| {
        [
            vec![user("a"), reference(id), user("b")],
            vec![
                user("a"),
                reference(id),
                call("c1", "Paris"),
                output("c1", "sunny"),
            ],
        ]
    };
    for (input, upstream) in inputs(&issued).into_iter().zip(inputs("msg_1")) {
        let body = json!({"model": "gpt-4o", "tools": weather_tool(), "input": input});
        let sent = serde_json::to_vec(
            &json!({"model": "gpt-4o", "tools": weather_tool(), "input": upstream}),
        )
        .unwrap();
        let (status, text) = post(&gw, &sk, &body).await;
        assert_eq!(status, 200, "{text}");
        let cap = mock.captured().unwrap();
        assert_eq!(cap.path, "/v1/responses");
        assert_eq!(
            String::from_utf8(cap.body).unwrap(),
            String::from_utf8(sent).unwrap()
        );
    }
}
