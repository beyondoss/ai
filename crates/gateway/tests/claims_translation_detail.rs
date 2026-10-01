//! Claim coverage: translation detail (`TRN-*` in `verify/claims.toml`).
//!
//! Most of these drive the public translation functions directly (`translate::request`,
//! `translate::response_json_status`, `translate::SseBridge`): the mapping tables are pure, so a
//! gateway process adds nothing but time. The one end-to-end test needs the real binary, because
//! what it pins is what leaves the gateway on a same-wire walk, where no translation runs at all.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH for the end-to-end test).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use beyond_ai::route::Endpoint;
use beyond_ai::translate::{SseBridge, request, response_json_status};
use common::*;
use serde_json::{Value, json};

fn translate(from: Endpoint, to: Endpoint, body: &Value, model: &str) -> Value {
    let out = request(from, to, &serde_json::to_vec(body).unwrap(), model);
    serde_json::from_slice(&out)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out)))
}

fn response(up: Endpoint, client: Endpoint, body: &Value) -> Value {
    let out = response_json_status(up, client, 200, &serde_json::to_vec(body).unwrap());
    serde_json::from_slice(&out).unwrap()
}

/// SSE events as `(event name, data)`; a bare `data:` stream has empty names.
fn stream(up: Endpoint, client: Endpoint, src: &str) -> Vec<(String, Value)> {
    let mut bridge = SseBridge::new(client, up);
    let mut out = bridge.feed(src.as_bytes(), false);
    out.extend(bridge.feed(b"", true));
    String::from_utf8(out)
        .unwrap()
        .split("\n\n")
        .filter(|e| !e.trim().is_empty())
        .map(|e| {
            let mut name = String::new();
            let mut data = Value::Null;
            for line in e.lines() {
                if let Some(n) = line.strip_prefix("event: ") {
                    name = n.to_owned();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = serde_json::from_str(d).unwrap_or(Value::String(d.to_owned()));
                }
            }
            (name, data)
        })
        .collect()
}

/// Every key path in a JSON value, for "this field is gone" assertions anywhere in a body.
fn has_key(v: &Value, key: &str) -> bool {
    match v {
        Value::Object(m) => m.iter().any(|(k, x)| k == key || has_key(x, key)),
        Value::Array(a) => a.iter().any(|x| has_key(x, key)),
        _ => false,
    }
}

fn contains_text(v: &Value, needle: &str) -> bool {
    v.to_string().contains(needle)
}

// --- TRN-21 --------------------------------------------------------------------------------------

/// Each mapping ARCHITECTURE.md documents as lossy, asserted as the loss it is: the field is gone
/// on the other wire, not half-translated, and not silently kept where the upstream would reject it.
/// claim: TRN-21
#[test]
fn every_documented_lossy_mapping_drops_what_it_says() {
    use Endpoint::{ChatCompletions as Chat, Messages, Responses};

    // Responses-only fields are dropped when the walk leaves Responses.
    let r = json!({
        "model": "m", "input": "hi", "store": true, "include": ["reasoning.encrypted_content"],
        "truncation": "auto", "background": false,
    });
    for (to, model) in [(Chat, "gpt-4o"), (Messages, "claude-opus-4-8")] {
        let v = translate(Responses, to, &r, model);
        for field in ["store", "include", "truncation", "background"] {
            assert!(v.get(field).is_none(), "{field} survived onto {to:?}: {v}");
        }
    }

    // JSON mode has no schema to give Anthropic.
    let c = json!({
        "model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "json please"}],
        "response_format": {"type": "json_object"},
    });
    let v = translate(Chat, Messages, &c, "claude-opus-4-8");
    assert!(
        v.get("response_format").is_none()
            && v.get("output_config")
                .and_then(|o| o.get("format"))
                .is_none(),
        "{v}"
    );

    // Hints that do not change the response's shape are dropped onto Messages.
    let c = json!({
        "model": "m", "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi", "name": "alice"}],
        "seed": 7, "frequency_penalty": 0.5, "presence_penalty": 0.5, "logit_bias": {"50256": -100},
        "prompt_cache_key": "k", "service_tier": "flex", "verbosity": "low",
    });
    let v = translate(Chat, Messages, &c, "claude-opus-4-8");
    for field in [
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "name",
        "prompt_cache_key",
        "service_tier",
        "verbosity",
    ] {
        assert!(!has_key(&v, field), "{field} survived onto Messages: {v}");
    }
    // `top_k` is Anthropic's, and has no Chat Completions field.
    let m = json!({
        "model": "m", "max_tokens": 100, "top_k": 5,
        "messages": [{"role": "user", "content": "hi"}],
    });
    let v = translate(Messages, Chat, &m, "gpt-4o");
    assert!(!has_key(&v, "top_k"), "{v}");
    // ...while OpenAI's own Chat Completions keeps the three it takes.
    let r = json!({
        "model": "m", "input": "hi", "prompt_cache_key": "k", "service_tier": "flex",
        "text": {"verbosity": "low"},
    });
    let v = translate(Responses, Chat, &r, "gpt-5");
    assert_eq!(v["prompt_cache_key"], "k", "{v}");
    assert_eq!(v["service_tier"], "flex", "{v}");
    let v = translate(Responses, Chat, &r, "deepseek-v4-pro");
    assert!(
        v.get("prompt_cache_key").is_none() && v.get("service_tier").is_none(),
        "{v}"
    );

    // Records of a hosted tool the provider ran itself are dropped; the answer follows as text.
    let r = json!({"model": "m", "input": [
        {"type": "message", "role": "user", "content": "search it"},
        {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "x"}},
        {"type": "mcp_call", "id": "mcp_1", "name": "lookup", "server_label": "s", "arguments": "{}", "output": "SECRET-TOOL-OUTPUT"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "found it"}]},
        {"type": "message", "role": "user", "content": "thanks"}
    ]});
    let v = translate(Responses, Chat, &r, "gpt-4o");
    assert!(
        !contains_text(&v, "web_search_call") && !contains_text(&v, "ws_1"),
        "{v}"
    );
    assert!(!contains_text(&v, "SECRET-TOOL-OUTPUT"), "{v}");
    assert!(contains_text(&v, "found it"), "{v}");

    // A non-http(s) image URL is never forwarded onto Messages (the Responses half is its own
    // reproduction below).
    let v = translate(Chat, Messages, &file_image(), "claude-opus-4-8");
    assert!(
        !contains_text(&v, "/etc/passwd"),
        "file:// URL forwarded onto Messages: {v}"
    );

    // OpenAI's own reasoning items carry nothing Anthropic can verify: they never become thinking.
    let r = json!({"model": "m", "input": [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "reasoning", "id": "rs_openai1", "summary": [{"type": "summary_text", "text": "OPENAI-REASONING"}], "encrypted_content": "gAAAAopaque"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
        {"type": "message", "role": "user", "content": "again"}
    ]});
    let v = translate(Responses, Messages, &r, "claude-opus-4-8");
    assert!(
        !contains_text(&v, "OPENAI-REASONING") && !contains_text(&v, "gAAAAopaque"),
        "{v}"
    );
    assert!(
        !contains_text(&v["messages"], r#""type":"thinking""#),
        "{v}"
    );

    // `redacted_thinking` has no Responses form: a Responses client gets no item for it.
    let msg = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
        "content": [
            {"type": "redacted_thinking", "data": "REDACTED-BLOB"},
            {"type": "text", "text": "answer"}
        ],
        "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2},
    });
    let v = response(Messages, Responses, &msg);
    assert!(!contains_text(&v, "REDACTED-BLOB"), "{v}");
    assert!(contains_text(&v["output"], "answer"), "{v}");

    // Reasoning that never got a signature is dropped for an Anthropic client.
    let chat = json!({
        "id": "c", "object": "chat.completion", "model": "deepseek-v4-pro",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x", "reasoning_content": "UNSIGNED-THOUGHT"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    });
    let v = response(Chat, Messages, &chat);
    assert!(!contains_text(&v, "UNSIGNED-THOUGHT"), "{v}");
}

fn file_image() -> Value {
    json!({"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}
    ]}]})
}

/// "A non-http(s) image URL (`file://`) is never forwarded in any shape" — including onto a
/// Responses-only row, where Chat Completions image parts are otherwise passed through as URLs.
/// claim: TRN-21
/// defect: D68
#[test]
fn a_file_image_url_never_reaches_a_responses_upstream() {
    let v = translate(
        Endpoint::ChatCompletions,
        Endpoint::Responses,
        &file_image(),
        "gpt-5-pro",
    );
    assert!(
        !contains_text(&v, "/etc/passwd"),
        "file:// URL forwarded onto Responses: {v}"
    );
    // The same image from a Responses client onto Chat Completions.
    let r = json!({"model": "m", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "file:///etc/passwd"}
    ]}]});
    let v = translate(Endpoint::Responses, Endpoint::ChatCompletions, &r, "gpt-4o");
    assert!(
        !contains_text(&v, "/etc/passwd"),
        "file:// URL forwarded onto Chat Completions: {v}"
    );
}

// --- TRN-4 ---------------------------------------------------------------------------------------

/// Count of `cache_control` markers anywhere in a body.
fn markers(v: &Value) -> usize {
    match v {
        Value::Object(m) => {
            usize::from(m.contains_key("cache_control")) + m.values().map(markers).sum::<usize>()
        }
        Value::Array(a) => a.iter().map(markers).sum(),
        _ => 0,
    }
}

/// A Chat client's `cache_control` is honoured on every role and at both levels — on a content
/// part, and on the whole message — and lands on the Messages block it describes. A client marker
/// anywhere switches the automatic breakpoints off, so exactly the client's markers go out.
/// claim: TRN-4
#[test]
fn cache_control_is_honoured_on_every_role_at_part_and_message_level() {
    let cc = json!({"type": "ephemeral"});
    let body = json!({"model": "m", "max_tokens": 100, "messages": [
        {"role": "system", "content": [{"type": "text", "text": "SYS-PART", "cache_control": cc}]},
        {"role": "system", "content": "SYS-MSG", "cache_control": cc},
        {"role": "user", "content": [{"type": "text", "text": "USER-PART", "cache_control": cc}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ASSISTANT-PART", "cache_control": cc}]},
        {"role": "user", "content": "USER-MSG", "cache_control": cc},
    ]});
    let v = translate(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &body,
        "claude-haiku-4-5",
    );
    let find = |text: &str| -> Value {
        fn walk(v: &Value, text: &str) -> Option<Value> {
            match v {
                Value::Object(m) if m.get("text").and_then(Value::as_str) == Some(text) => {
                    Some(v.clone())
                }
                Value::Object(m) => m.values().find_map(|x| walk(x, text)),
                Value::Array(a) => a.iter().find_map(|x| walk(x, text)),
                _ => None,
            }
        }
        walk(&v, text).unwrap_or_else(|| panic!("no block for {text}: {v}"))
    };
    for text in [
        "SYS-PART",
        "SYS-MSG",
        "USER-PART",
        "ASSISTANT-PART",
        "USER-MSG",
    ] {
        assert_eq!(
            find(text)["cache_control"],
            cc,
            "{text} lost its marker: {v}"
        );
    }
    assert_eq!(
        markers(&v),
        5,
        "client markers only, no automatic ones: {v}"
    );

    // With no client marker at all, the automatic breakpoints apply instead.
    let plain = json!({"model": "m", "max_tokens": 100, "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]});
    let v = translate(
        Endpoint::ChatCompletions,
        Endpoint::Messages,
        &plain,
        "claude-haiku-4-5",
    );
    assert_eq!(v["system"][0]["cache_control"], cc, "{v}");
    assert_eq!(markers(&v), 2, "the prefix and the conversation: {v}");
}

// --- TRN-22 --------------------------------------------------------------------------------------

/// The stop-reason table toward a Responses client, which the Chat Completions and Messages tables
/// already pinned elsewhere do not reach: what ended the turn becomes `status` and
/// `incomplete_details.reason`, non-stream and streamed (`response.completed` vs
/// `response.incomplete`), from both a Messages and a Chat Completions upstream.
/// claim: TRN-22
#[test]
fn stop_reasons_reach_a_responses_client_from_every_upstream() {
    use Endpoint::{ChatCompletions as Chat, Messages, Responses};

    // (Anthropic stop_reason, Chat finish_reason, Responses status, incomplete reason)
    let table = [
        ("end_turn", "stop", "completed", None),
        ("stop_sequence", "stop", "completed", None),
        (
            "max_tokens",
            "length",
            "incomplete",
            Some("max_output_tokens"),
        ),
        (
            "refusal",
            "content_filter",
            "incomplete",
            Some("content_filter"),
        ),
    ];
    for (stop, finish, status, reason) in table {
        let msg = json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": "partial"}], "stop_reason": stop,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        });
        let chat = json!({
            "id": "c", "object": "chat.completion", "created": 1, "model": "gpt-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "partial"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        });
        for (up, body) in [(Messages, &msg), (Chat, &chat)] {
            let v = response(up, Responses, body);
            assert_eq!(v["status"], status, "{up:?} {stop}/{finish}: {v}");
            assert_eq!(
                v["incomplete_details"]["reason"].as_str(),
                reason,
                "{up:?} {stop}/{finish}: {v}"
            );
        }

        let ant_sse = format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"c\",\"content\":[],\"usage\":{{\"input_tokens\":1}}}}}}\n\n\
             event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
             event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"partial\"}}}}\n\n\
             event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
             event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{stop}\"}},\"usage\":{{\"output_tokens\":5}}}}\n\n\
             event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
        );
        let chat_sse = format!(
            "data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"partial\"}}}}]}}\n\n\
             data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{finish}\"}}]}}\n\n\
             data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5\",\"choices\":[],\"usage\":{{\"prompt_tokens\":1,\"completion_tokens\":5}}}}\n\n\
             data: [DONE]\n\n"
        );
        for (up, src) in [(Messages, &ant_sse), (Chat, &chat_sse)] {
            let evs = stream(up, Responses, src);
            let terminal = if status == "completed" {
                "response.completed"
            } else {
                "response.incomplete"
            };
            let last = evs
                .iter()
                .rev()
                .find(|(n, _)| n.starts_with("response."))
                .unwrap_or_else(|| panic!("{up:?} {stop}: no response.* event: {evs:?}"));
            assert_eq!(last.0, terminal, "{up:?} {stop}/{finish}: {evs:?}");
            assert_eq!(last.1["response"]["status"], status, "{up:?} {stop}");
            assert_eq!(
                last.1["response"]["incomplete_details"]["reason"].as_str(),
                reason,
                "{up:?} {stop}"
            );
        }
    }

    // A tool turn is a completed response whose output holds the call.
    let msg = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 3, "output_tokens": 2},
    });
    let v = response(Messages, Responses, &msg);
    assert_eq!(v["status"], "completed", "{v}");
    assert_eq!(v["output"][0]["type"], "function_call", "{v}");
    assert_eq!(v["output"][0]["name"], "get_weather", "{v}");
}

// --- TRN-5 ---------------------------------------------------------------------------------------

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 55,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

/// A stock Chat Completions request with `max_tokens` on a GPT-5 or o-series row, walked same-wire
/// to OpenAI: OpenAI rejects `max_tokens` on every reasoning model, so what leaves the gateway must
/// carry the limit as `max_completion_tokens` (the translated walks already do).
/// claim: TRN-5
/// defect: D63
#[tokio::test]
#[ignore = "D63 reproduced: a same-wire walk relays max_tokens to OpenAI reasoning models"]
async fn max_tokens_reaches_a_reasoning_model_as_max_completion_tokens() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;
    let mut wrong = Vec::new();
    for model in ["gpt-5-mini", "gpt-5.4", "o3", "o4-mini"] {
        let resp = test_client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk)))
            .header("content-type", "application/json")
            .body(format!(
                r#"{{"model":"{model}","max_tokens":256,"messages":[{{"role":"user","content":"hi"}}]}}"#
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{model}");
        let cap = mock.captured().unwrap();
        assert_eq!(
            cap.path, "/v1/chat/completions",
            "{model}: a same-wire walk"
        );
        let sent: Value = serde_json::from_slice(&cap.body).unwrap();
        if sent.get("max_tokens").is_some() || sent["max_completion_tokens"] != 256 {
            wrong.push(format!("{model}: {sent}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "OpenAI would 400 these:\n{}",
        wrong.join("\n")
    );
}
