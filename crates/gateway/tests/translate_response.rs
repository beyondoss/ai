//! End-to-end: what a translated catalog walk hands the client back, through the real proxy.
//!
//! `tests/translate.rs` proves the walk translates at all. This file drives the response contract
//! with provider-shaped fixtures: the full Responses stream lifecycle with tool calls and reasoning,
//! an OpenRouter Claude signature reaching a Messages client on failover, cache-inclusive usage on
//! a Chat client, and upstream errors — including a non-2xx body with no `error` key — keeping
//! their detail.

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

/// Claude with signed thinking, text, and two parallel tool calls; cache reads and writes.
const CLAUDE_TOOLS_SSE: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-opus-4-8","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"cache_read_input_tokens":5000,"cache_creation_input_tokens":2000,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Two cities, two calls."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBsig=="}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking both."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: content_block_start
data: {"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_b","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Rome\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":3}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}

event: message_stop
data: {"type":"message_stop"}

"#;

/// OpenAI streaming text, then two parallel tool calls, `include_usage` on.
const OPENAI_PARALLEL_SSE: &str = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Checking both."},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Rome\"}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1790000000,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":6000,"completion_tokens":40,"total_tokens":6040,"prompt_tokens_details":{"cached_tokens":5000}}}

data: [DONE]

"#;

/// OpenRouter serving Claude over Chat Completions with thinking: the signature arrives only in a
/// later `reasoning_details` entry (shape from a live capture).
const OPENROUTER_CLAUDE_SSE: &str = r#"data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-opus-4.8","choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning":"Two lookups.","reasoning_details":[{"type":"reasoning.text","text":"Two lookups.","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-opus-4.8","choices":[{"index":0,"delta":{"content":"","role":"assistant","reasoning_details":[{"type":"reasoning.text","signature":"EqIFCpwBsig","format":"anthropic-claude-v1","index":0}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-opus-4.8","choices":[{"index":0,"delta":{"content":null,"role":"assistant","tool_calls":[{"index":0,"id":"toolu_bdrk_01","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},"finish_reason":null,"native_finish_reason":null}]}

data: {"id":"gen-1","object":"chat.completion.chunk","created":1790802605,"model":"anthropic/claude-opus-4.8","choices":[{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"tool_calls","native_finish_reason":"tool_use"}],"usage":{"prompt_tokens":597,"completion_tokens":190,"total_tokens":787,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"completion_tokens_details":{"reasoning_tokens":98}}}

data: [DONE]

"#;

/// SSE events as `(event name, data)`; `[DONE]` is a JSON string.
fn events(text: &str) -> Vec<(String, Value)> {
    text.split("\n\n")
        .filter(|e| !e.trim().is_empty())
        .map(|e| {
            let mut name = String::new();
            let mut data = String::new();
            for line in e.lines() {
                if let Some(n) = line.strip_prefix("event: ") {
                    name = n.to_owned();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = d.to_owned();
                }
            }
            let v = if data == "[DONE]" {
                json!("[DONE]")
            } else {
                serde_json::from_str(&data).unwrap_or_else(|e| panic!("{e}: {data}"))
            };
            (name, v)
        })
        .collect()
}

fn names(evs: &[(String, Value)]) -> Vec<&str> {
    evs.iter().map(|(n, _)| n.as_str()).collect()
}

async fn post(gw: &Gateway, path: &str, key: &str, body: String) -> (u16, String) {
    let mut req = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("content-type", "application/json");
    req = if path.ends_with("/messages") {
        req.header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
    } else {
        req.header("authorization", format!("Bearer {key}"))
    };
    // On a transport failure, the gateway's own log says whether it ever saw the request.
    let resp = match req.body(body).send().await {
        Ok(resp) => resp,
        Err(e) => panic!("{e:?}\ngateway log:\n{}", gw.log()),
    };
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap())
}

fn responses_body(model: &str, stream: bool) -> String {
    format!(
        r#"{{"model":"{model}","input":"weather in Paris and Rome?","stream":{stream},"store":false,"tools":[{{"type":"function","name":"get_weather","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}}}}}}]}}"#
    )
}

/// The Responses stream a stock OpenAI SDK needs: every item added before its content, done after,
/// and a terminal `response.completed` whose `output` is those items.
fn assert_lifecycle(evs: &[(String, Value)]) -> Value {
    assert_eq!(evs[0].0, "response.created", "{:?}", names(evs));
    assert_eq!(evs[1].0, "response.in_progress", "{:?}", names(evs));
    for (i, (name, v)) in evs.iter().enumerate() {
        assert_eq!(v["type"], json!(name));
        assert_eq!(v["sequence_number"], json!(i));
    }
    let added = evs
        .iter()
        .filter(|(n, _)| n == "response.output_item.added")
        .count();
    let done: Vec<&Value> = evs
        .iter()
        .filter(|(n, _)| n == "response.output_item.done")
        .map(|(_, v)| &v["item"])
        .collect();
    assert_eq!(added, done.len(), "{:?}", names(evs));
    let (last, v) = evs.last().unwrap();
    assert_eq!(last, "response.completed", "{:?}", names(evs));
    let resp = v["response"].clone();
    assert_eq!(resp["output"].as_array().unwrap().len(), done.len());
    assert!(resp["created_at"].as_u64().unwrap() > 0);
    resp
}

/// A Responses client on a Claude row: thinking, text and both tool calls arrive as items, and
/// `response.completed` carries them. Before this the stream had no items at all and the OpenAI
/// SDK's `responses.stream()` raised on the first delta.
#[tokio::test]
async fn responses_client_on_claude_gets_items_for_reasoning_text_and_calls() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", CLAUDE_TOOLS_SSE)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let (status, text) = post(
        &gw,
        "/v1/responses",
        &vkey(&sk),
        responses_body("claude-opus-4-8", true),
    )
    .await;
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    let evs = events(&text);
    let resp = assert_lifecycle(&evs);
    let out = resp["output"].as_array().unwrap();
    let types: Vec<&str> = out.iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(
        types,
        ["reasoning", "message", "function_call", "function_call"]
    );
    assert_eq!(out[0]["encrypted_content"], "EqQBsig==");
    assert_eq!(out[1]["content"][0]["text"], "Checking both.");
    assert_eq!(out[2]["call_id"], "toolu_a");
    assert_eq!(out[2]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(out[3]["call_id"], "toolu_b");
    assert_eq!(
        resp["usage"]["input_tokens"], 7010,
        "cache reads and writes included"
    );
    assert_eq!(resp["usage"]["input_tokens_details"]["cached_tokens"], 5000);

    let cap = mock.captured().expect("reaches Anthropic");
    assert_eq!(cap.path, "/v1/messages");
    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"anthropic""#])
        .await;
    assert!(
        line.contains(r#""cache_read_tokens":5000"#),
        "billing still reads the upstream Anthropic counts: {line}"
    );
}

#[tokio::test]
async fn responses_client_on_gpt_gets_function_call_items() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", OPENAI_PARALLEL_SSE)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .start()
        .await;

    let (status, text) = post(
        &gw,
        "/v1/responses",
        &vkey(&sk),
        responses_body("gpt-4o-mini", true),
    )
    .await;
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    let resp = assert_lifecycle(&events(&text));
    let out = resp["output"].as_array().unwrap();
    assert_eq!(out.len(), 3, "{resp}");
    assert_eq!(out[0]["type"], "message");
    for (item, city) in out[1..].iter().zip(["Paris", "Rome"]) {
        assert_eq!(item["type"], "function_call");
        assert!(item["id"].as_str().unwrap().starts_with("fc_"));
        let args: Value = serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], city);
    }
    assert_eq!(mock.captured().unwrap().path, "/v1/chat/completions");
}

/// Anthropic is down; OpenRouter serves the Claude row over Chat Completions. The signature it
/// streams in `reasoning_details` must reach the Messages client, or the next turn on Anthropic
/// 400s (`thinking.signature: Field required`) for the rest of the conversation.
#[tokio::test]
async fn messages_client_keeps_the_thinking_signature_across_openrouter_failover() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback =
        MockUpstream::start(Mode::Raw(200, "text/event-stream", OPENROUTER_CLAUDE_SSE)).await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let body = r#"{"model":"claude-opus-4-8","max_tokens":2000,"stream":true,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"weather in Paris?"}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]}"#;
    let (status, text) = post(&gw, "/v1/messages", &vkey(&sk), body.to_owned()).await;
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    let evs = events(&text);
    let starts: Vec<&Value> = evs
        .iter()
        .filter(|(n, _)| n == "content_block_start")
        .map(|(_, v)| &v["content_block"])
        .collect();
    assert_eq!(starts[0]["type"], "thinking", "{text}");
    assert_eq!(starts[1]["type"], "tool_use", "{text}");
    let sig = evs
        .iter()
        .find(|(_, v)| v["delta"]["type"] == "signature_delta")
        .expect("a signature_delta");
    assert_eq!(sig.1["delta"]["signature"], "EqIFCpwBsig");
    assert_eq!(sig.1["index"], 0);
    let delta = &evs.iter().find(|(n, _)| n == "message_delta").unwrap().1;
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(delta["usage"]["input_tokens"], 597);
    assert_eq!(evs.last().unwrap().0, "message_stop");

    let cap = fallback.captured().expect("OpenRouter served");
    assert_eq!(cap.path, "/api/v1/chat/completions");
}

/// A Chat client on a Claude row: `prompt_tokens` is the whole prompt, cache included, and every
/// chunk carries `created` (strictly typed clients reject a chunk without it).
#[tokio::test]
async fn chat_client_on_claude_gets_whole_prompt_usage_and_created() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", CLAUDE_TOOLS_SSE)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let body = r#"{"model":"claude-opus-4-8","stream":true,"messages":[{"role":"user","content":"weather?"}]}"#;
    let (status, text) = post(&gw, "/v1/chat/completions", &vkey(&sk), body.to_owned()).await;
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    let evs = events(&text);
    let chunks: Vec<&Value> = evs
        .iter()
        .map(|(_, v)| v)
        .filter(|v| v.is_object())
        .collect();
    for c in &chunks {
        assert!(c["created"].as_u64().unwrap() > 0, "{c}");
        assert_eq!(c["id"], chunks[0]["id"], "{c}");
    }
    let usage = chunks.iter().find_map(|c| c.get("usage")).unwrap();
    assert_eq!(usage["prompt_tokens"], 7010, "{usage}");
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 5000);
    assert_eq!(usage["prompt_tokens_details"]["cache_write_tokens"], 2000);
    let finish = chunks
        .iter()
        .find_map(|c| c["choices"][0]["finish_reason"].as_str())
        .unwrap();
    assert_eq!(finish, "tool_calls");
    assert_eq!(evs.last().unwrap().1, "[DONE]");
}

/// A truncated Claude answer is `incomplete` to a Responses client, not `completed`.
#[tokio::test]
async fn responses_client_sees_truncation_as_incomplete() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let body = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"cut of"}],"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":16}}"#;
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", body)).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let (status, text) = post(
        &gw,
        "/v1/responses",
        &vkey(&sk),
        responses_body("claude-opus-4-8", false),
    )
    .await;
    assert_eq!(status, 200, "{text}\n{}", gw.log());
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "incomplete");
    assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(v["output"][0]["content"][0]["text"], "cut of");
    assert!(v["created_at"].as_u64().unwrap() > 0);
}

/// Bedrock answers errors as `{"message": …}`, with no `error` key. Mapped as a success, the OpenAI
/// client got an empty completion under a 400; now it gets the message.
#[tokio::test]
async fn a_non_2xx_body_without_an_error_key_reaches_the_client_as_an_error() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Raw(
        400,
        "application/json",
        r#"{"message":"The provided model identifier is invalid."}"#,
    ))
    .await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let body = r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#;
    let (status, text) = post(&gw, "/v1/chat/completions", &vkey(&sk), body.to_owned()).await;
    assert_eq!(status, 400, "{text}\n{}", gw.log());
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        v["error"]["message"], "The provided model identifier is invalid.",
        "{text}"
    );
    assert!(v.get("choices").is_none(), "{text}");
}

/// The Python SDKs send `accept-encoding: gzip, deflate`. Forwarded, Anthropic and OpenAI gzip the
/// body, which the gateway then parses as plain bytes: a translated response reached the client
/// untranslated (or as plain text under `content-encoding: gzip`) and billing parsed zero tokens —
/// on same-wire relays too. The upstream is asked for `identity` on every managed request.
#[tokio::test]
async fn managed_requests_ask_the_upstream_for_an_uncompressed_body() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;

    let key = vkey(&sk);
    for (path, body) in [
        // Translated walk.
        (
            "/v1/chat/completions",
            r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        // Same-wire relay: billing still parses the body.
        (
            "/v1/messages",
            r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    ] {
        let mut req = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("content-type", "application/json")
            .header("accept-encoding", "gzip, deflate");
        req = if path == "/v1/messages" {
            req.header("x-api-key", &key)
                .header("anthropic-version", "2023-06-01")
        } else {
            req.header("authorization", format!("Bearer {key}"))
        };
        let resp = req.body(body).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200, "{path}");
        let cap = mock.captured().expect("reaches the upstream");
        assert_eq!(
            cap.accept_encoding.as_deref(),
            Some("identity"),
            "{path} must not let the upstream compress"
        );
    }
}

/// OpenRouter wraps the provider's error; the Messages client gets the provider's message too.
#[tokio::test]
async fn openrouter_provider_errors_keep_the_provider_message() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let fallback = MockUpstream::start(Mode::Raw(
        400,
        "application/json",
        r#"{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"prompt is too long: 250000 tokens > 200000 maximum\"}}","provider_name":"Anthropic"}},"user_id":"u"}"#,
    ))
    .await;
    let gw = Gateway::builder(nats_port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["anthropic", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;

    let body = r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
    let (status, text) = post(&gw, "/v1/messages", &vkey(&sk), body.to_owned()).await;
    assert_eq!(status, 400, "{text}\n{}", gw.log());
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{text}");
    assert_eq!(
        v["error"]["message"],
        "Provider returned error (Anthropic): prompt is too long: 250000 tokens > 200000 maximum"
    );
}
