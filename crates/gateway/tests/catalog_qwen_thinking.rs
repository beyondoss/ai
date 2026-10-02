//! End-to-end: Together's (and OpenRouter's) Qwen rows get what their backend can serve.
//!
//! - A forced `tool_choice` on Qwen3.7 Plus or Qwen3.8 Flash goes with thinking off: thinking on,
//!   it is a 400 on every candidate of the row (D172).
//! - Tools on Qwen3.7 Plus go with thinking off unless the client asked for reasoning: thinking,
//!   it sometimes writes the call as text (D171).
//! - A `developer` message reaches Together as `system`, which its Qwen backend reads and
//!   `developer` it refuses (D173). OpenRouter maps the role itself and gets it as sent.
//!
//! The mock Together behaves like the real one: stream only, a 400 on a forced tool while
//! thinking, a 400 on `developer`.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use common::*;
use serde_json::{Value, json};

/// Qwen calling `get_weather`, as Together streams it (abridged).
const QWEN_TOOL_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_7\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]},\"finish_reason\":null}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":{\"completion_tokens\":21,\"prompt_tokens\":80,\"total_tokens\":101}}\n\n",
    "data: [DONE]\n\n",
);

/// Together's answer to a forced tool while thinking (measured 2026-10-01).
const FORCED_WHILE_THINKING: &str = r#"{"id":"p3WVBcE","error":{"message":"<400> InternalError.Algo.InvalidParameter: The tool_choice parameter does not support being set to required or object in thinking mode","type":"invalid_request_error"}}"#;

/// Together's answer to a `developer` message on Qwen (measured 2026-10-01).
const DEVELOPER_REFUSED: &str = r#"{"id":"p3WVHRs","error":{"message":"developer is not one of ['system', 'assistant', 'user', 'tool', 'function']","type":"invalid_request_error"}}"#;

type Seen = Arc<Mutex<Vec<(String, Value)>>>;

/// Together on `/v1/chat/completions`, refusing what the real one refuses; OpenRouter on any
/// other path, answering the stock chat completion.
fn upstream(seen: &Seen) -> impl Fn(usize, &ScriptReq) -> Reply + Send + Sync + 'static {
    let seen = seen.clone();
    move |_, req| {
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        seen.lock().unwrap().push((req.path.clone(), body.clone()));
        if req.path != "/v1/chat/completions" {
            return Reply::ok();
        }
        let thinking = body["reasoning"]["enabled"] != false;
        let forced = body["tool_choice"] == "required" || body["tool_choice"].is_object();
        let developer = body["messages"]
            .as_array()
            .is_some_and(|m| m.iter().any(|m| m["role"] == "developer"));
        if forced && thinking {
            return Reply::json(400, FORCED_WHILE_THINKING);
        }
        if developer {
            return Reply::json(400, DEVELOPER_REFUSED);
        }
        Reply::Full {
            status: 200,
            content_type: "text/event-stream",
            body: Bytes::from_static(QWEN_TOOL_SSE.as_bytes()),
        }
    }
}

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

fn last_sent(seen: &Seen) -> (String, Value) {
    seen.lock().unwrap().last().cloned().unwrap()
}

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}])
}

fn chat(model: &str, extra: Value) -> Value {
    let mut v = json!({"model": model, "tools": tools(),
                       "messages": [{"role": "user", "content": "Weather in Paris?"}]});
    for (k, x) in extra.as_object().unwrap() {
        v[k] = x.clone();
    }
    v
}

async fn gateway(seen: &Seen, seed: u8) -> (ReplyUpstream, Gateway, String) {
    let (pubkey, sk) = test_keypair(seed);
    let up = ReplyUpstream::start(upstream(seen)).await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["together", "openrouter"])
        .provider_authority("openrouter", &up.authority())
        .start()
        .await;
    (up, gw, billing_vkey(&sk, seed.into()))
}

/// Qwen3.7 Plus and Qwen3.8 Flash refuse a forced `tool_choice` while thinking, at Together and at
/// OpenRouter alike, so a forced call (Chat `required` or a named function, Messages `any` or a
/// named tool) goes with `"reasoning":{"enabled":false}` and no `reasoning_effort`, whatever the
/// client asked; it is served. `auto` on Qwen3.8 Flash is sent as it came.
/// claim: T1, E1, E2
/// defect: D172
#[tokio::test]
async fn a_forced_tool_on_a_thinking_qwen_row_goes_with_thinking_off() {
    let seen = Seen::default();
    let (_up, gw, key) = gateway(&seen, 172).await;
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    for (model, extra) in [
        ("qwen/qwen3.8-flash", json!({"tool_choice": named})),
        ("qwen/qwen3.8-flash", json!({"tool_choice": "required"})),
        (
            "qwen/qwen3.8-flash",
            json!({"tool_choice": "required", "reasoning_effort": "high"}),
        ),
        ("qwen/qwen3.7-plus", json!({"tool_choice": named})),
    ] {
        let body = chat(model, extra);
        let resp = post(&gw, &key, "/v1/chat/completions", &[], &body).await;
        assert_eq!(resp.status().as_u16(), 200, "{body}");
        let v: Value = resp.json().await.unwrap();
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        let (_, sent) = last_sent(&seen);
        assert_eq!(sent["reasoning"], json!({"enabled": false}), "{sent}");
        assert!(sent.get("reasoning_effort").is_none(), "{sent}");
        assert_eq!(sent["tool_choice"], body["tool_choice"], "{sent}");
    }

    // A Messages client forcing a tool: translated onto Chat, then the same.
    for choice in [
        json!({"type": "any"}),
        json!({"type": "tool", "name": "get_weather"}),
    ] {
        let body = json!({"model": "qwen/qwen3.8-flash", "max_tokens": 256,
            "tools": [{"name": "get_weather", "input_schema": {"type": "object",
                "properties": {"city": {"type": "string"}}}}],
            "tool_choice": choice,
            "messages": [{"role": "user", "content": "Weather in Paris?"}]});
        let resp = post(&gw, &key, "/v1/messages", &[], &body).await;
        assert_eq!(resp.status().as_u16(), 200, "{body}");
        let (_, sent) = last_sent(&seen);
        assert_eq!(sent["reasoning"], json!({"enabled": false}), "{sent}");
    }

    // OpenRouter's copy is Alibaba's backend too.
    let body = chat("qwen/qwen3.8-flash", json!({"tool_choice": "required"}));
    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &[("x-beyond-only", "openrouter")],
        &body,
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let (path, sent) = last_sent(&seen);
    assert_eq!(path, "/api/v1/chat/completions");
    assert_eq!(sent["reasoning"], json!({"enabled": false}), "{sent}");

    // `auto` on Qwen3.8 Flash thinks as usual.
    let body = chat("qwen/qwen3.8-flash", json!({"tool_choice": "auto"}));
    let resp = post(&gw, &key, "/v1/chat/completions", &[], &body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let (_, sent) = last_sent(&seen);
    assert!(sent.get("reasoning").is_none(), "{sent}");
}

/// Thinking, Qwen3.7 Plus sometimes writes a tool call as content text at Together and at
/// OpenRouter, never with thinking off. A request offering it tools goes with thinking off; a
/// client that asked for reasoning gets it as asked; a request without tools is untouched.
/// claim: T1, CAT-6
/// defect: D171
#[tokio::test]
async fn tools_on_qwen3_7_plus_go_with_thinking_off_unless_reasoning_was_asked() {
    let seen = Seen::default();
    let (_up, gw, key) = gateway(&seen, 171).await;
    for only in [None, Some("openrouter")] {
        let headers: Vec<(&str, &str)> = only.map(|p| ("x-beyond-only", p)).into_iter().collect();
        let body = chat("qwen/qwen3.7-plus", json!({}));
        let resp = post(&gw, &key, "/v1/chat/completions", &headers, &body).await;
        assert_eq!(resp.status().as_u16(), 200, "{only:?}");
        let (_, sent) = last_sent(&seen);
        assert_eq!(sent["reasoning"], json!({"enabled": false}), "{sent}");
    }

    let body = chat("qwen/qwen3.7-plus", json!({"reasoning_effort": "low"}));
    let resp = post(&gw, &key, "/v1/chat/completions", &[], &body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let (_, sent) = last_sent(&seen);
    assert_eq!(sent["reasoning_effort"], "low", "{sent}");
    assert!(sent.get("reasoning").is_none(), "{sent}");

    let plain = json!({"model": "qwen/qwen3.7-plus",
                       "messages": [{"role": "user", "content": "hi"}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &[], &plain).await;
    assert_eq!(resp.status().as_u16(), 200);
    let (_, sent) = last_sent(&seen);
    assert!(sent.get("reasoning").is_none(), "{sent}");
}

/// A Chat client's `developer` message reaches Together as `system` (its Qwen backend refuses
/// `developer`, and its other models do not follow it); every other message is as sent.
/// OpenRouter, which maps the role per upstream itself, gets `developer`.
/// claim: TRN-16
/// defect: D173
#[tokio::test]
async fn a_developer_message_reaches_together_as_system() {
    let seen = Seen::default();
    let (_up, gw, key) = gateway(&seen, 173).await;
    let body = json!({"model": "qwen/qwen3.8-flash", "messages": [
        {"role": "developer", "content": "Answer in French."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "developer"},
        {"role": "developer", "content": [{"type": "text", "text": "Be brief."}]},
        {"role": "user", "content": "again"}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &[], &body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let (path, sent) = last_sent(&seen);
    assert_eq!(path, "/v1/chat/completions");
    let roles: Vec<&str> = sent["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "system", "user"]);
    assert_eq!(sent["messages"][2]["content"], "developer");
    assert_eq!(
        sent["messages"][3]["content"],
        body["messages"][3]["content"]
    );

    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &[("x-beyond-only", "openrouter")],
        &body,
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let (path, sent) = last_sent(&seen);
    assert_eq!(path, "/api/v1/chat/completions");
    assert_eq!(sent["messages"][0]["role"], "developer");
}
