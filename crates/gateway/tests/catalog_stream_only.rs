//! End-to-end: a catalog candidate that answers only streams still serves a client that did not
//! ask for one.
//!
//! Together serves Qwen3.6 Plus, Qwen3.7 Plus, Qwen3.7 Max and Qwen3.8 Flash only as streams: a
//! request without `"stream": true` is a 400 `streaming_required`. A stock non-streaming SDK call
//! (Chat, Messages or Responses) on those rows must still get its ordinary JSON answer, billed
//! exactly once from the stream's usage.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use common::*;
use serde_json::{Value, json};

/// Together's answer to a non-streaming request on a stream-only model (measured 2026-10-01).
const TOGETHER_STREAMING_REQUIRED: &str = r#"{
  "id": "p3WJ1pb-2kFHot-a43ce724da45eeaa",
  "error": {
    "message": "This model only supports streaming. Set \"stream\": true.",
    "type": "invalid_request_error",
    "param": "stream",
    "code": "streaming_required"
  }
}"#;

/// Together's Qwen3.8 Flash stream, abridged: reasoning on `delta.reasoning`, then the text, a
/// finish chunk, and the usage chunk it sends whether or not `include_usage` was asked for.
const QWEN_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"logprobs\":null,\"delta\":{\"token_id\":null,\"role\":\"assistant\",\"content\":\"\",\"reasoning\":\"\"},\"finish_reason\":null,\"text\":\"\"}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"delta\":{\"token_id\":null,\"role\":\"assistant\",\"content\":\"\",\"reasoning\":\"We greet\"},\"finish_reason\":null,\"logprobs\":null,\"text\":\"\"}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"delta\":{\"token_id\":null,\"role\":\"assistant\",\"content\":\"\",\"reasoning\":\" them.\"},\"finish_reason\":null,\"logprobs\":null,\"text\":\"\"}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"delta\":{\"token_id\":null,\"role\":\"assistant\",\"content\":\"Hello\",\"reasoning\":\"\"},\"finish_reason\":null,\"logprobs\":null,\"text\":\"Hello\"}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"delta\":{\"token_id\":null,\"role\":\"assistant\",\"content\":\" there!\",\"reasoning\":\"\"},\"finish_reason\":null,\"logprobs\":null,\"text\":\" there!\"}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[{\"index\":0,\"delta\":{\"token_id\":null,\"role\":\"assistant\",\"content\":\"\",\"reasoning\":\"\"},\"finish_reason\":\"stop\",\"logprobs\":null,\"text\":\"\"}],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":null}\n\n",
    "data: {\"id\":\"chatcmpl-qwen\",\"object\":\"chat.completion.chunk\",\"created\":1790872773,\"choices\":[],\"model\":\"Qwen/Qwen3.8-Flash\",\"usage\":{\"completion_tokens\":39,\"completion_tokens_details\":{\"reasoning_tokens\":27,\"text_tokens\":39},\"prompt_tokens\":62,\"prompt_tokens_details\":{\"cached_tokens\":12,\"text_tokens\":62},\"total_tokens\":101}}\n\n",
    "data: [DONE]\n\n",
);

/// Every request the upstream saw: path and body.
type Seen = Arc<Mutex<Vec<(String, Value)>>>;

/// Together on `/v1/chat/completions`, stream-only like the real one: `sse` for a streaming
/// request, the 400 otherwise. Any other path (OpenRouter's) answers the stock chat completion,
/// or `other` when set.
fn together(
    seen: &Seen,
    sse: &'static str,
    other: Option<u16>,
) -> impl Fn(usize, &ScriptReq) -> Reply + Send + Sync + 'static {
    let seen = seen.clone();
    move |_, req| {
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let streaming = body["stream"] == true;
        seen.lock().unwrap().push((req.path.clone(), body));
        if req.path != "/v1/chat/completions" {
            return Reply::ok();
        }
        if let Some(status) = other {
            return Reply::json(
                status,
                r#"{"error":{"message":"Service unavailable","type":"service_unavailable"}}"#,
            );
        }
        if streaming {
            Reply::Full {
                status: 200,
                content_type: "text/event-stream",
                body: Bytes::from_static(sse.as_bytes()),
            }
        } else {
            Reply::json(400, TOGETHER_STREAMING_REQUIRED)
        }
    }
}

async fn post(gw: &Gateway, key: &str, path: &str, body: &Value) -> reqwest::Response {
    test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap()
}

fn content_type(resp: &reqwest::Response) -> String {
    resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned()
}

fn last_sent(seen: &Seen) -> (String, Value) {
    seen.lock().unwrap().last().cloned().unwrap()
}

const MODEL: &str = "qwen/qwen3.8-flash";

fn chat(stream: Option<bool>) -> Value {
    let mut v = json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]});
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

/// A non-streaming Chat, Messages or Responses call on a row whose primary answers only streams
/// gets its ordinary JSON answer: the gateway asks the candidate for a stream (with its usage) and
/// assembles it into the client's own body, reasoning and all. One billing row each, with the
/// stream's exact tokens. A streaming client is relayed as before.
/// claim: CAT-1, CAT-2, E1, E2, E3, B1
/// defect: D147
#[tokio::test]
#[ignore = "D147 reproduced: a non-stream call to a stream-only Together row is Together's 400"]
async fn a_non_stream_call_on_a_stream_only_candidate_gets_a_whole_answer() {
    let (pubkey, sk) = test_keypair(147);
    let seen = Seen::default();
    let up = ReplyUpstream::start(together(&seen, QWEN_SSE, None)).await;
    let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
        .providers(&["together"])
        .start()
        .await;
    let key = billing_vkey(&sk, 147);

    // Chat, with `stream` absent and with `stream: false`.
    for body in [chat(None), chat(Some(false))] {
        let resp = post(&gw, &key, "/v1/chat/completions", &body).await;
        assert_eq!(resp.status().as_u16(), 200, "{body}");
        assert!(
            content_type(&resp).starts_with("application/json"),
            "{}",
            content_type(&resp)
        );
        let v: Value = resp.json().await.unwrap();
        assert_eq!(v["object"], "chat.completion", "{v}");
        let choice = &v["choices"][0];
        assert_eq!(choice["message"]["role"], "assistant", "{v}");
        assert_eq!(choice["message"]["content"], "Hello there!", "{v}");
        assert_eq!(
            choice["message"]["reasoning_content"], "We greet them.",
            "{v}"
        );
        assert_eq!(choice["finish_reason"], "stop", "{v}");
        assert_eq!(v["usage"]["prompt_tokens"], 62, "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 39, "{v}");
        let (path, sent) = last_sent(&seen);
        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(sent["model"], "Qwen/Qwen3.8-Flash", "{sent}");
        assert_eq!(sent["stream"], true, "{sent}");
        assert_eq!(sent["stream_options"]["include_usage"], true, "{sent}");
    }

    // Messages.
    let body = json!({"model": MODEL, "max_tokens": 64,
                      "messages": [{"role": "user", "content": "hi"}]});
    let resp = post(&gw, &key, "/v1/messages", &body).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert!(content_type(&resp).starts_with("application/json"));
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["type"], "message", "{v}");
    assert_eq!(v["stop_reason"], "end_turn", "{v}");
    let text: Vec<&str> = v["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect();
    assert_eq!(text, ["Hello there!"], "{v}");
    assert_eq!(v["usage"]["output_tokens"], 39, "{v}");
    assert_eq!(last_sent(&seen).1["stream"], true);

    // Responses.
    let body = json!({"model": MODEL, "input": "hi"});
    let resp = post(&gw, &key, "/v1/responses", &body).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert!(content_type(&resp).starts_with("application/json"));
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["object"], "response", "{v}");
    assert_eq!(v["status"], "completed", "{v}");
    let message = v["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["type"] == "message")
        .unwrap_or_else(|| panic!("no message item: {v}"));
    assert_eq!(message["content"][0]["text"], "Hello there!", "{v}");
    assert_eq!(v["usage"]["output_tokens"], 39, "{v}");
    assert_eq!(last_sent(&seen).1["stream"], true);

    // A streaming client still gets the stream.
    let resp = post(&gw, &key, "/v1/chat/completions", &chat(Some(true))).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert!(content_type(&resp).starts_with("text/event-stream"));
    let text = resp.text().await.unwrap();
    assert!(text.contains("data: [DONE]"), "{text}");

    // One exact row per call: the stream's own usage, never an estimate.
    let rows = wait_usage_rows(&gw, 5, 10).await;
    assert_eq!(rows.len(), 5, "{rows:?}");
    for row in &rows {
        assert_eq!(row["provider"], "together", "{row}");
        assert_eq!(row["usage_estimated"], false, "{row}");
        assert_eq!(row["input_tokens"], 62, "{row}");
        assert_eq!(row["output_tokens"], 39, "{row}");
        assert_eq!(row["cache_read_tokens"], 12, "{row}");
        assert_eq!(row["upstream_status"], 200, "{row}");
    }
}
