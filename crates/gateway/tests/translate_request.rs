//! End-to-end: what a translated request looks like when it reaches the upstream.
//!
//! `translate.rs`'s unit tests pin each mapping; these drive the stock-SDK flows those mappings
//! exist for through the real proxy, and read what the mock upstream received (body *and* headers,
//! which leave before the body is translated) — including a client's next turn built from what the
//! gateway streamed it, and signed thinking bound for OpenRouter.

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

/// Claude streaming signed thinking, then a tool call.
const CLAUDE_THINKING_TOOL_SSE: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Need weather."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBsig=="}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":30}}

event: message_stop
data: {"type":"message_stop"}

"#;

/// A streaming OpenAI SDK on a Claude row with reasoning, two turns. Turn 2 was a 400
/// ("thinking.signature: Field required"): the signature rode a string the request side never read
/// back, and the echoed `reasoning_content` became an unsigned block. The stream now carries each
/// finished block on the `thinking` list, and the echo reaches Anthropic signed.
#[tokio::test]
async fn a_streamed_claude_turn_reaches_anthropic_signed_on_the_next_turn() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Raw(
        200,
        "text/event-stream",
        CLAUDE_THINKING_TOOL_SSE,
    ))
    .await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let user = json!({"role": "user", "content": "Paris?"});
    let turn1 = json!({
        "model": "claude-haiku-4-5", "stream": true, "reasoning_effort": "low",
        "tools": [weather_tool()], "messages": [user.clone()]
    });
    let text = post(&gw, &sk, "/v1/chat/completions", &turn1).await;
    // What an OpenAI SDK accumulates: list entries by `index`, strings concatenated.
    let mut assistant =
        json!({"role": "assistant", "content": null, "tool_calls": [], "thinking": []});
    let mut args = String::new();
    for data in text.lines().filter_map(|l| l.strip_prefix("data: ")) {
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let delta = &chunk["choices"][0]["delta"];
        for entry in delta["thinking"].as_array().into_iter().flatten() {
            let i = usize::try_from(entry["index"].as_u64().expect("an index")).unwrap();
            assert_eq!(i, assistant["thinking"].as_array().unwrap().len(), "{text}");
            assistant["thinking"]
                .as_array_mut()
                .unwrap()
                .push(entry.clone());
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            if call.get("id").is_some() {
                assistant["tool_calls"] = json!([{"id": call["id"], "type": "function",
                    "function": {"name": call["function"]["name"], "arguments": ""}}]);
            }
            args.push_str(call["function"]["arguments"].as_str().unwrap_or(""));
        }
    }
    assistant["tool_calls"][0]["function"]["arguments"] = json!(args);
    assert!(!text.contains("thinking_signature"), "{text}");

    let turn2 = json!({
        "model": "claude-haiku-4-5", "reasoning_effort": "low", "tools": [weather_tool()],
        "messages": [user, assistant, {"role": "tool", "tool_call_id": "toolu_1", "content": "18C"}]
    });
    post(&gw, &sk, "/v1/chat/completions", &turn2).await;
    let (cap, got) = captured(&mock);
    assert_eq!(cap.path, "/v1/messages");
    let turn = &got["messages"][1]["content"];
    assert_eq!(
        turn[0],
        json!({"type": "thinking", "thinking": "Need weather.", "signature": "EqQBsig=="}),
        "{got}"
    );
    assert_eq!(turn[1]["type"], "tool_use");
    assert_eq!(turn[1]["input"], json!({"city": "Paris"}));
}

/// An Anthropic SDK thinking + tool loop on a Claude row only OpenRouter serves. OpenRouter replays
/// Claude's thinking from `reasoning_details` alone; without it turn 2 was a 400 ("a final
/// `assistant` message must start with a thinking block").
#[tokio::test]
async fn anthropic_sdk_thinking_reaches_openrouter_as_reasoning_details() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let body = json!({
        "model": "claude-sonnet-4", "max_tokens": 2000,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tools": [{"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "t", "signature": "EqQBsig=="},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]},
        ]
    });
    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200, "{}", gw.log());
    let (cap, got) = captured(&fallback);
    assert_eq!(cap.path, "/api/v1/chat/completions");
    assert_eq!(
        got["messages"][1]["reasoning_details"],
        json!([{"type": "reasoning.text", "text": "t", "signature": "EqQBsig==",
                "format": "anthropic-claude-v1", "index": 0}]),
        "{got}"
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

/// A stock `client.responses.create(model=..., input=...)` sends no `store` (OpenAI defaults it to
/// true). /v1/models lists `/v1/responses` for a Claude row, so that default call must succeed
/// there, translated onto Messages. Today an omitted `store` counts as session state, and a row
/// with no Responses arm answers 400 "store cannot be honored".
/// claim: TRN-1, CAT-9
/// defect: D12
#[tokio::test]
#[ignore = "D12 reproduced: /v1/responses without store on a Claude row is 400 \"store cannot be honored\""]
async fn stock_responses_create_without_store_works_on_a_claude_row() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    // The catalog advertises the endpoint for this row.
    let models: Value = test_client()
        .get(format!("{}/v1/models", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let row = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "claude-opus-4-8")
        .expect("claude-opus-4-8 is listed");
    assert!(
        row["endpoints"]
            .as_array()
            .unwrap()
            .contains(&json!("/v1/responses")),
        "{row}"
    );

    let text = post(
        &gw,
        &sk,
        "/v1/responses",
        &json!({"model": "claude-opus-4-8", "input": "hi"}),
    )
    .await;
    let resp: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(resp["object"], "response", "{resp}");

    let (cap, got) = captured(&mock);
    assert_eq!(cap.path, "/v1/messages");
    assert_eq!(got["messages"][0]["content"], "hi", "{got}");
    assert!(got.get("store").is_none(), "{got}");
}
