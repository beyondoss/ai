//! End-to-end: what a translated request looks like when it reaches the upstream.
//!
//! `translate.rs`'s unit tests pin each mapping; these drive the stock-SDK flows those mappings
//! exist for through the real proxy, and read what the mock upstream received (body *and* headers,
//! which leave before the body is translated).

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

async fn post(gw: &Gateway, sk: &ed25519_dalek::SigningKey, path: &str, body: &Value) -> String {
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(sk)))
        .header("content-type", "application/json")
        .body(serde_json::to_vec(body).unwrap())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    assert_eq!(status, 200, "{text}");
    text
}

fn captured(mock: &MockUpstream) -> (Captured, Value) {
    let cap = mock.captured().expect("the request reached the upstream");
    let body = serde_json::from_slice(&cap.body).unwrap();
    (cap, body)
}

/// An Anthropic SDK (Claude Code's shape: `max_tokens`, `temperature`, thinking off, a screenshot
/// in a tool result) on a GPT-5 row. OpenAI 400s `max_tokens` and non-default sampling on every
/// reasoning model and `reasoning_effort: "none"` on GPT-5, and a tool message cannot hold the
/// image, so each used to be a 400 or an answer about a picture the model never saw.
#[tokio::test]
async fn anthropic_sdk_on_a_gpt5_row_sends_what_openai_accepts() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let body = json!({
        "model": "gpt-5-nano",
        "max_tokens": 512,
        "temperature": 0.2,
        "thinking": {"type": "disabled"},
        "system": "Be brief.",
        "tools": [{"name": "screenshot", "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {"role": "user", "content": "What is on screen?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "screenshot", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                {"type": "text", "text": "Screenshot:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
            ]}]}
        ]
    });
    let text = post(&gw, &sk, "/v1/messages", &body).await;
    assert!(text.contains(r#""type":"message""#), "{text}");

    let (cap, got) = captured(&mock);
    assert_eq!(cap.path, "/v1/chat/completions");
    assert_eq!(got["model"], "gpt-5-nano");
    assert_eq!(got["max_completion_tokens"], 512, "{got}");
    assert!(got.get("max_tokens").is_none(), "{got}");
    assert!(got.get("temperature").is_none(), "{got}");
    assert_eq!(got["reasoning_effort"], "minimal", "GPT-5's lowest: {got}");
    let roles: Vec<&str> = got["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(
        roles,
        ["system", "user", "assistant", "tool", "user"],
        "{got}"
    );
    assert_eq!(got["messages"][3]["content"], "Screenshot:");
    assert_eq!(
        got["messages"][4]["content"][0]["image_url"]["url"],
        "data:image/png;base64,iVBORw0KGgo="
    );
}

/// A Responses client (store: false) with parallel tool calls on a GPT row. Consecutive
/// `function_call` items became one assistant message each, which OpenAI rejects; a Responses
/// named `tool_choice` went through flat, which Chat Completions rejects.
#[tokio::test]
async fn responses_client_parallel_tool_calls_reach_chat_as_one_turn() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::OpenAiToolJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let body = json!({
        "model": "gpt-4.1-nano",
        "store": false,
        "max_output_tokens": 64,
        "tools": [
            {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
            {"type": "web_search"}
        ],
        "tool_choice": {"type": "function", "name": "get_weather"},
        "input": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"type": "function_call", "call_id": "call_a", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call", "call_id": "call_b", "name": "get_weather", "arguments": "{\"city\":\"Rome\"}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "sunny"},
            {"type": "function_call_output", "call_id": "call_b", "output": "rain"}
        ]
    });
    post(&gw, &sk, "/v1/responses", &body).await;

    let (cap, got) = captured(&mock);
    assert_eq!(cap.path, "/v1/chat/completions");
    let msgs = got["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 4, "{got}");
    assert_eq!(msgs[1]["tool_calls"].as_array().unwrap().len(), 2, "{got}");
    assert_eq!(msgs[2]["tool_call_id"], "call_a");
    assert_eq!(msgs[3]["tool_call_id"], "call_b");
    assert_eq!(
        got["tool_choice"],
        json!({"type": "function", "function": {"name": "get_weather"}})
    );
    assert_eq!(
        got["tools"][1],
        json!({"type": "web_search"}),
        "a hosted tool is forwarded for OpenAI to reject by name, not dropped"
    );
    assert_eq!(got["max_completion_tokens"], 64);
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}})
}

/// Preserved thinking, three turns, a Chat Completions client echoing thinking. Claude Fable 5.1
/// rejects forced tool use, so turn 1's `required` becomes `auto` plus a closing system instruction
/// that only the gateway's request holds. Turn 2 replays turn 1's thinking without it, which an
/// enforced account answers with 400 "bound to a different conversation" unless the request sets
/// `thinking.block_binding.prefix_mismatch_behavior: drop_block`, and that field is itself a 400
/// without the `thinking-binding-controls-2026-08-01` beta header, which only `proxy` can send.
#[tokio::test]
async fn preserved_thinking_survives_a_forced_tool_turn() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicToolJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let system = json!({"role": "system", "content": "You are a weather assistant."});
    let user = json!({"role": "user", "content": "Paris?"});
    let turn1 = json!({
        "model": "claude-fable-5-1",
        "reasoning_effort": "high",
        "tools": [weather_tool()],
        "tool_choice": "required",
        "messages": [system.clone(), user.clone()]
    });
    let text = post(&gw, &sk, "/v1/chat/completions", &turn1).await;
    let reply: Value = serde_json::from_str(&text).unwrap();
    let assistant = reply["choices"][0]["message"].clone();

    let (cap, got) = captured(&mock);
    assert_eq!(cap.path, "/v1/messages");
    assert_eq!(
        cap.anthropic_beta.as_deref(),
        Some("thinking-binding-controls-2026-08-01")
    );
    assert_eq!(
        got["thinking"]["block_binding"],
        json!({"prefix_mismatch_behavior": "drop_block"}),
        "{got}"
    );
    assert_eq!(got["tool_choice"], json!({"type": "auto"}));
    let last = got["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(
        last["role"], "system",
        "the instruction closes turn 1: {got}"
    );

    // Turn 2: the client echoes the assistant turn, as Anthropic returned it (a signed thinking
    // block ahead of the tool call), plus the tool result. The instruction is not in its history.
    let mut echoed = assistant;
    echoed["thinking"] = json!([{"type": "thinking", "thinking": "", "signature": "EqQBsig"}]);
    let tool_msg = json!({"role": "tool", "tool_call_id": "toolu_1", "content": "18C"});
    let turn2 = json!({
        "model": "claude-fable-5-1",
        "reasoning_effort": "high",
        "tools": [weather_tool()],
        "messages": [system.clone(), user.clone(), echoed.clone(), tool_msg.clone()]
    });
    post(&gw, &sk, "/v1/chat/completions", &turn2).await;
    let (cap, got) = captured(&mock);
    assert_eq!(
        cap.anthropic_beta.as_deref(),
        Some("thinking-binding-controls-2026-08-01")
    );
    assert_eq!(
        got["thinking"]["block_binding"]["prefix_mismatch_behavior"], "drop_block",
        "{got}"
    );
    let msgs = got["messages"].as_array().unwrap();
    assert_eq!(msgs[1]["content"][0]["type"], "thinking", "replayed: {got}");
    assert_eq!(msgs[1]["content"][0]["signature"], "EqQBsig");
    assert!(
        msgs.iter().all(|m| m["role"] != "system"),
        "no instruction on an unforced turn: {got}"
    );

    // Turn 3: a mid-conversation system message stays in place, so the prefix every earlier
    // block is bound to is unchanged (it used to be hoisted into top-level `system`).
    let turn3 = json!({
        "model": "claude-fable-5-1",
        "tools": [weather_tool()],
        "messages": [
            system, user, echoed, tool_msg,
            {"role": "assistant", "content": "It is 18C in Paris."},
            {"role": "user", "content": "And Rome?"},
            {"role": "system", "content": "Answer in French."}
        ]
    });
    post(&gw, &sk, "/v1/chat/completions", &turn3).await;
    let (_, got) = captured(&mock);
    assert_eq!(got["system"][0]["text"], "You are a weather assistant.");
    assert_eq!(got["system"].as_array().unwrap().len(), 1, "{got}");
    let last = got["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["role"], "system", "{got}");
    assert_eq!(
        got["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
}

/// The beta header goes only where the field does: a model without the conversation check gets
/// neither, and a Messages client's own walk is a byte relay (its headers are its own).
#[tokio::test]
async fn the_binding_beta_goes_only_with_the_field() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let body = json!({"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]});
    post(&gw, &sk, "/v1/chat/completions", &body).await;
    let (cap, got) = captured(&mock);
    assert_eq!(cap.anthropic_beta, None);
    assert!(got.pointer("/thinking/block_binding").is_none(), "{got}");

    let body = json!({"model": "claude-fable-5-1", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]});
    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let (cap, got) = captured(&mock);
    assert_eq!(cap.anthropic_beta, None, "same-wire Messages is untouched");
    assert!(got.get("thinking").is_none(), "{got}");
}
