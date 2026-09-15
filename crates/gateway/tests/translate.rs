//! End-to-end: Chat Completions ↔ Messages translation on a managed catalog walk.
//!
//! The existing stream tests in `model_routing.rs` prove the happy path. This file drives the
//! rest of the contract through the real proxy: non-stream JSON (Pingora withholds until EOS),
//! tool_use ↔ tool_calls, error envelopes, cache fill of the *client* dialect, capture of client
//! bytes, `/auto/…` suffixes, and failover-while-translating.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use serde_json::Value;

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

fn claude_chat(stream: bool) -> String {
    format!(
        r#"{{"model":"claude-opus-4-8","stream":{stream},"messages":[{{"role":"user","content":"hi"}}]}}"#
    )
}

fn gpt_messages(stream: bool) -> String {
    format!(
        r#"{{"model":"gpt-4o-mini","max_tokens":16,"stream":{stream},"messages":[{{"role":"user","content":"hi"}}]}}"#
    )
}

/// Non-stream JSON is withheld until EOS and remapped. This is the path a stock SDK takes when
/// `stream` is off — and the one Pingora empty-chunk withholding used to drop.
#[tokio::test]
async fn openai_sdk_nonstream_claude_round_trips_json() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(claude_chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    let v: Value = serde_json::from_str(&text).expect(&text);
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "hi");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 13);
    assert_eq!(v["usage"]["completion_tokens"], 7);
    assert!(
        !text.contains("input_tokens"),
        "Anthropic usage keys must not leak: {text}"
    );

    let cap = mock
        .captured()
        .expect("translated request reaches Anthropic");
    assert_eq!(cap.path, "/v1/messages");
    assert_eq!(
        cap.anthropic_version.as_deref(),
        Some("2023-06-01"),
        "OpenAI→Anthropic must inject anthropic-version"
    );
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""max_tokens":4096"#),
        "Anthropic requires max_tokens; a missing OpenAI value becomes 4096: {got}"
    );
    assert!(
        !got.contains("stream_options"),
        "OpenAI→Anthropic must not inject stream_options: {got}"
    );

    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"anthropic""#])
        .await;
    assert!(
        line.contains(r#""input_tokens":13"#),
        "billing parses the Anthropic JSON: {line}"
    );
}

#[tokio::test]
async fn anthropic_sdk_nonstream_gpt_round_trips_json() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(gpt_messages(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    let v: Value = serde_json::from_str(&text).expect(&text);
    assert_eq!(v["type"], "message");
    assert_eq!(v["content"][0]["text"], "hi");
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["input_tokens"], 11);
    assert_eq!(v["usage"]["output_tokens"], 7);
    assert!(
        !text.contains("chat.completion"),
        "OpenAI JSON must not leak: {text}"
    );

    let cap = mock.captured().expect("translated request reaches OpenAI");
    assert_eq!(cap.path, "/v1/chat/completions");
    assert_eq!(
        cap.anthropic_version, None,
        "anthropic-version must be stripped on Anthropic→OpenAI: {cap:?}"
    );
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        !got.contains("stream_options"),
        "non-stream OpenAI body must not grow include_usage: {got}"
    );

    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"openai""#])
        .await;
    assert!(
        line.contains(r#""input_tokens":11"#),
        "billing parses the OpenAI JSON: {line}"
    );
}

#[tokio::test]
async fn openai_sdk_sees_claude_tool_calls_on_the_stream() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicToolSse).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","stream":true,"messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{}}}}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("chat.completion.chunk"), "{text}");
    assert!(text.contains("\"tool_calls\""), "{text}");
    assert!(text.contains("get_weather"), "{text}");
    assert!(text.contains("toolu_1"), "{text}");
    assert!(text.contains("SF"), "{text}");
    assert!(text.contains(r#""finish_reason":"tool_calls""#), "{text}");
    assert!(text.contains("[DONE]"), "{text}");
    assert!(!text.contains("message_start"), "{text}");

    let cap = mock.captured().expect("tool request reaches Anthropic");
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""name":"get_weather""#) && got.contains("input_schema"),
        "tools must be Messages-shaped upstream: {got}"
    );
}

#[tokio::test]
async fn anthropic_sdk_sees_gpt_tool_use_on_the_stream() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::OpenAiToolSse).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o-mini","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"weather?"}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{}}}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("event: message_start"), "{text}");
    assert!(text.contains("\"type\":\"tool_use\""), "{text}");
    assert!(text.contains("get_weather"), "{text}");
    assert!(text.contains("call_1"), "{text}");
    assert!(text.contains("input_json_delta"), "{text}");
    assert!(text.contains("SF"), "{text}");
    assert!(text.contains(r#""stop_reason":"tool_use""#), "{text}");
    assert!(text.contains("event: message_stop"), "{text}");
    assert!(!text.contains("chat.completion.chunk"), "{text}");
}

#[tokio::test]
async fn openai_sdk_nonstream_claude_tool_calls() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicToolJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{}}}}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let v: Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "toolu_1");
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    let args = v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    assert!(args.contains("SF"), "{args}");
}

#[tokio::test]
async fn anthropic_sdk_nonstream_gpt_tool_use() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::OpenAiToolJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("content-type", "application/json")
        .body(r#"{"model":"gpt-4o-mini","max_tokens":16,"messages":[{"role":"user","content":"weather?"}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{}}}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let v: Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["stop_reason"], "tool_use");
    assert_eq!(v["content"][0]["type"], "tool_use");
    assert_eq!(v["content"][0]["id"], "call_1");
    assert_eq!(v["content"][0]["input"]["city"], "SF");
}

/// The second turn of a stock OpenAI tool loop: assistant `tool_calls` + `role: tool` must become
/// Anthropic `tool_use` + `tool_result` on the wire.
#[tokio::test]
async fn openai_tool_loop_second_turn_reaches_anthropic() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"weather?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"64F"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{}}}}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let cap = mock.captured().expect("second turn reaches Anthropic");
    let got: Value = serde_json::from_slice(&cap.body).unwrap();
    assert_eq!(got["messages"][1]["role"], "assistant");
    assert_eq!(got["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(got["messages"][1]["content"][0]["id"], "call_1");
    assert_eq!(got["messages"][1]["content"][0]["input"]["city"], "SF");
    assert_eq!(got["messages"][2]["role"], "user");
    assert_eq!(got["messages"][2]["content"][0]["type"], "tool_result");
    assert_eq!(got["messages"][2]["content"][0]["tool_use_id"], "call_1");
    assert_eq!(got["messages"][2]["content"][0]["content"], "64F");
}

#[tokio::test]
async fn openai_sdk_sees_mapped_anthropic_error() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicStatus(400)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(claude_chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let v: Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["error"]["message"], "mock");
    assert_eq!(v["error"]["type"], "api_error");
    assert!(v.get("type").is_none() || v["type"] != "error");
}

#[tokio::test]
async fn anthropic_sdk_sees_mapped_openai_error() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Status(400)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("content-type", "application/json")
        .body(gpt_messages(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let v: Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["message"], "mock");
}

#[tokio::test]
async fn openai_sdk_sees_mapped_anthropic_sse_error() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicErrorSse).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(claude_chat(true))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("try again"), "{text}");
    assert!(text.contains("overloaded_error"), "{text}");
    assert!(!text.contains("event: error"), "{text}");
}

#[tokio::test]
async fn anthropic_sdk_sees_mapped_openai_sse_error() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::OpenAiErrorSse).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("content-type", "application/json")
        .body(gpt_messages(true))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("event: error"), "{text}");
    assert!(text.contains("try again"), "{text}");
    assert!(!text.contains("chat.completion"), "{text}");
    assert!(
        !text.contains("event: message_start"),
        "mapped SSE error must not grow a fake message: {text}"
    );
}

/// Cache fill taps *client* bytes. Two identical translated requests: the second is a hit in the
/// inbound dialect and the mock is not asked again.
#[tokio::test]
async fn translated_catalog_walk_fills_the_cache_in_the_client_dialect() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .cache_ttl_secs(60)
        .start()
        .await;
    let key = vkey(&sk);
    let client = test_client();
    let body = claude_chat(false);

    let first = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(first.status().as_u16(), 200);
    let first_body = first.bytes().await.unwrap();
    let first_text = String::from_utf8_lossy(&first_body);
    assert!(
        first_text.contains("chat.completion"),
        "fill is the client dialect: {first_text}"
    );
    let _ = gw.wait_for_log_line(&["ai.usage"]).await;
    let hits_after_fill = mock.hits();
    assert!(hits_after_fill >= 1);

    let second = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status().as_u16(), 200);
    let second_body = second.bytes().await.unwrap();
    assert_eq!(
        second_body, first_body,
        "hit body must be byte-identical to the client-dialect fill"
    );
    assert_eq!(
        mock.hits(),
        hits_after_fill,
        "the second request must not reach the upstream"
    );
    wait_for_metric(&gw, "ai_cache_hits_total", "", 1.0).await;
    let line = gw
        .wait_for_log_line(&["ai.usage", "cache_hit", "true"])
        .await;
    assert!(
        line.contains(r#""input_tokens":13"#),
        "cache hit must emit the stored (upstream) tokens: {line}"
    );
}

/// Capture taps the bytes the client sent and the bytes the client sees — not the upstream dialect.
#[tokio::test]
async fn translate_capture_is_the_client_dialect() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-capture", "on")
        .body(claude_chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);

    let line = gw.wait_for_log_line(&[r#""target":"ai.payload""#]).await;
    let payload: Value = serde_json::from_str(&line).expect(&line);
    let req = payload["fields"]["request_body"]
        .as_str()
        .unwrap_or_else(|| panic!("request_body missing: {payload}"));
    let resp_body = payload["fields"]["response_body"]
        .as_str()
        .unwrap_or_else(|| panic!("response_body missing: {payload}"));
    assert!(
        req.contains("claude-opus-4-8") && req.contains(r#""role":"user""#),
        "request capture is the pre-translate client body: {req}"
    );
    assert!(
        !req.contains(r#""max_tokens":4096"#),
        "request capture must not be the translated Anthropic body: {req}"
    );
    assert!(
        resp_body.contains("chat.completion") && resp_body.contains(r#""prompt_tokens":13"#),
        "response capture is the post-translate client body: {resp_body}"
    );
    assert!(
        !resp_body.contains(r#""type":"message""#) && !resp_body.contains("input_tokens"),
        "response capture must not be the raw Anthropic JSON: {resp_body}"
    );
}

#[tokio::test]
async fn auto_chat_completions_translates_claude() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .header("x-beyond-model", "claude-opus-4-8")
        .body(claude_chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("chat.completion"), "{text}");
    let cap = mock.captured().expect("auto suffix reaches Anthropic");
    assert_eq!(cap.path, "/v1/messages");
}

#[tokio::test]
async fn auto_v1_messages_translates_gpt() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/auto/v1/messages", gw.url()))
        .header("x-api-key", vkey(&sk))
        .header("content-type", "application/json")
        .body(gpt_messages(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains(r#""type":"message""#), "{text}");
    let cap = mock.captured().expect("auto suffix reaches OpenAI");
    assert_eq!(cap.path, "/v1/chat/completions");
}

/// Translate state lives across a candidate walk: Anthropic is dead, OpenRouter serves Messages,
/// the OpenAI client still sees Chat Completions.
#[tokio::test]
async fn failover_while_translating_still_returns_the_client_dialect() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(claude_chat(false))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "Anthropic is dead; OpenRouter must serve the translated walk"
    );
    let text = resp.text().await.unwrap();
    assert!(text.contains("chat.completion"), "{text}");
    assert!(!text.contains(r#""type":"message""#), "{text}");

    let cap = fallback.captured().expect("fallback served");
    assert_eq!(cap.path, "/api/v1/messages");
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""model":"anthropic/claude-opus-4.8""#),
        "failover must splice the OpenRouter candidate id after translate: {got}"
    );
}

fn gpt_responses_session() -> &'static str {
    r#"{"model":"gpt-4o","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"previous_response_id":"resp_abc","include":["reasoning.encrypted_content"],"truncation":"auto"}"#
}

fn gpt_responses_one_shot() -> &'static str {
    r#"{"model":"gpt-4o","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"max_output_tokens":16,"store":false,"include":["file_search_call.results"],"truncation":"auto"}"#
}

/// Managed `/v1/responses` + a GPT row + `previous_response_id` must hit OpenAI `/v1/responses`
/// with the field intact — not Chat Completions with the id stripped.
#[tokio::test]
async fn managed_responses_with_previous_response_id_relays_to_openai_responses() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(gpt_responses_session())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "{}",
        resp.text().await.unwrap()
    );

    let cap = mock
        .captured()
        .expect("session Responses must reach OpenAI");
    assert_eq!(cap.path, "/v1/responses");
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""previous_response_id":"resp_abc""#),
        "previous_response_id must pass through: {got}"
    );
    assert!(
        got.contains(r#""include""#) && got.contains("reasoning.encrypted_content"),
        "include must pass through on same-endpoint Responses: {got}"
    );
    assert!(
        got.contains(r#""truncation":"auto""#),
        "truncation must pass through on same-endpoint Responses: {got}"
    );
    assert!(
        !got.contains("chat/completions"),
        "must not rewrite the path into Chat Completions"
    );
}

/// `store: false` one-shot Responses may still translate onto Chat Completions.
#[tokio::test]
async fn store_false_one_shot_responses_may_translate_onto_chat_completions() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(gpt_responses_one_shot())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "{}",
        resp.text().await.unwrap()
    );

    let cap = mock.captured().expect("one-shot must reach OpenAI");
    assert_eq!(
        cap.path, "/v1/chat/completions",
        "store:false may still land on Chat Completions"
    );
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        !got.contains("previous_response_id"),
        "session fields must not be required on a one-shot: {got}"
    );
    assert!(
        !got.contains(r#""store""#),
        "store is Responses-only and is dropped onto Chat Completions: {got}"
    );
    assert!(
        !got.contains(r#""include""#) && !got.contains("truncation"),
        "include/truncation are dropped when leaving Responses: {got}"
    );
    assert!(
        got.contains(r#""messages""#),
        "input must become messages on Chat Completions: {got}"
    );
}

/// Claude rows have no OpenAI store. Responses + `previous_response_id` is 400, not Messages.
#[tokio::test]
async fn claude_responses_with_previous_response_id_is_400() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","input":"hi","previous_response_id":"resp_1"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("previous_response_id"),
        "400 must name the field: {text}"
    );
    assert!(text.contains("claude-opus-4-8"), "{text}");
    assert_eq!(mock.hits(), 0, "must not become a hollow Messages call");
}

/// `/{provider}/v1/responses` is the escape hatch: byte relay, no catalog.
#[tokio::test]
async fn provider_prefixed_responses_is_still_a_relay() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;

    let resp = test_client()
        .post(format!("{}/openai/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(gpt_responses_session())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "{}",
        resp.text().await.unwrap()
    );

    let cap = mock
        .captured()
        .expect("provider-prefixed Responses must reach the upstream");
    assert_eq!(cap.path, "/v1/responses");
    let got = String::from_utf8(cap.body).unwrap();
    assert!(
        got.contains(r#""previous_response_id":"resp_abc""#),
        "provider path is a relay: {got}"
    );
    assert!(
        got.contains(r#""include""#) && got.contains(r#""truncation":"auto""#),
        "include/truncation pass through on the provider path: {got}"
    );
}
