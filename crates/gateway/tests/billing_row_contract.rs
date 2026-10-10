//! The `ai.usage` row contract (ARCHITECTURE.md, "The `ai.usage` row"): every fact a pricer needs
//! reaches the row, on every wire that reports it — Anthropic Messages, OpenAI Chat Completions and
//! Responses, streamed and not, relayed and translated.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::time::Duration;

async fn post(url: String, auth: (&str, String), body: &str, extra: &[(&str, &str)]) -> u16 {
    let mut req = test_client()
        .post(url)
        .header(auth.0, auth.1)
        .header("content-type", "application/json");
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    let resp = req.body(body.to_owned()).send().await.unwrap();
    let status = resp.status().as_u16();
    let _ = resp.bytes().await;
    status
}

fn bearer(sk: &ed25519_dalek::SigningKey, tenant: u64) -> (&'static str, String) {
    (
        "authorization",
        format!("Bearer {}", billing_vkey(sk, tenant)),
    )
}

const ANTHROPIC_FAST_BODY: &str = r#"{"id":"msg_01fast","type":"message","role":"assistant","model":"claude-opus-4-8","container":{"id":"container_011abc","expires_at":"2026-10-10T00:00:00Z"},"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":10,"service_tier":"standard","speed":"fast","inference_geo":"us","server_tool_use":{"web_search_requests":3,"web_fetch_requests":2,"code_execution_requests":1}}}"#;

/// Anthropic Messages, relayed, non-stream: served speed and geography beside the requested ones,
/// every server tool by kind, the container, the message id, the `request-id` header, and the
/// endpoint that served.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn anthropic_relay_records_speed_geo_tools_container_and_ids() {
    let (pubkey, sk) = test_keypair(150);
    let mock = MockUpstream::start(Mode::RawHeaders(
        200,
        "application/json",
        ANTHROPIC_FAST_BODY,
        &[("request-id", "req_011upstream")],
    ))
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let status = post(
        format!("{}/anthropic/v1/messages", gw.url()),
        ("x-api-key", billing_vkey(&sk, 150)),
        r#"{"model":"claude-opus-4-8","max_tokens":64,"speed":"fast","inference_geo":"us","messages":[{"role":"user","content":"hi"}]}"#,
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["speed"], "fast", "{row}");
    assert_eq!(row["requested_speed"], "fast", "{row}");
    assert_eq!(row["inference_geo"], "us", "{row}");
    assert_eq!(row["requested_inference_geo"], "us", "{row}");
    assert_eq!(row["service_tier"], "standard", "{row}");
    assert_eq!(
        row["server_tools"], "web_search=3,web_fetch=2,code_execution=1",
        "{row}"
    );
    assert_eq!(
        row["server_tool_calls"].as_u64(),
        Some(3),
        "old field kept: {row}"
    );
    assert_eq!(row["container_id"], "container_011abc", "{row}");
    // A managed request may not ask for a container (code execution bills container-hours the
    // row cannot see), so the response's is the only one.
    assert!(row.get("requested_container").is_none(), "{row}");
    assert_eq!(row["upstream_generation_id"], "msg_01fast", "{row}");
    assert_eq!(row["upstream_request_id"], "req_011upstream", "{row}");
    assert_eq!(row["upstream_model"], "claude-opus-4-8", "{row}");
    assert_eq!(row["upstream_path"], "/v1/messages", "{row}");
    assert!(row.get("upstream_host").is_some(), "{row}");
    assert!(
        row.get("price_variant").is_none(),
        "one price at Anthropic: {row}"
    );
    assert!(
        row.get("upstream_cost_usd").is_none(),
        "Anthropic reports no cost: {row}"
    );
    assert_eq!(row["upstream_may_continue"], false, "{row}");
}

const ANTHROPIC_FAST_SSE: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01stream\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"usage\":{\"input_tokens\":20,\"output_tokens\":1,\"speed\":\"fast\",\"inference_geo\":\"global\",\"service_tier\":\"standard\"}}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"container\":{\"id\":\"container_s1\"}},\"usage\":{\"output_tokens\":9,\"server_tool_use\":{\"web_search_requests\":1}}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";

/// Anthropic streams `usage.speed` on `message_start` only (measured live): the row still says
/// `fast`, from the retained head, though the final usage block omits it.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn anthropic_stream_records_start_only_speed() {
    let (pubkey, sk) = test_keypair(151);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", ANTHROPIC_FAST_SSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let status = post(
        format!("{}/anthropic/v1/messages", gw.url()),
        ("x-api-key", billing_vkey(&sk, 151)),
        r#"{"model":"claude-opus-4-8","max_tokens":64,"stream":true,"speed":"fast","messages":[{"role":"user","content":"hi"}]}"#,
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["speed"], "fast", "{row}");
    assert_eq!(row["inference_geo"], "global", "{row}");
    assert_eq!(row["requested_speed"], "fast", "{row}");
    assert_eq!(row["upstream_generation_id"], "msg_01stream", "{row}");
    assert_eq!(row["container_id"], "container_s1", "{row}");
    assert_eq!(row["server_tools"], "web_search=1", "{row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(9), "{row}");
}

/// A Chat Completions client walked onto an Anthropic Messages candidate: the row still reads the
/// upstream's speed, tools and id (the usage taps read the upstream wire), and names the candidate
/// that served and its path.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn a_translated_walk_records_the_upstream_facts() {
    let (pubkey, sk) = test_keypair(152);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", ANTHROPIC_FAST_BODY)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let status = post(
        format!("{}/v1/chat/completions", gw.url()),
        bearer(&sk, 152),
        r#"{"model":"claude-opus-4-8","max_tokens":64,"service_tier":"priority","messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["routed_model"], "claude-opus-4-8", "{row}");
    assert_eq!(row["requested_service_tier"], "priority", "{row}");
    assert_eq!(row["speed"], "fast", "{row}");
    assert_eq!(
        row["server_tools"], "web_search=3,web_fetch=2,code_execution=1",
        "{row}"
    );
    assert_eq!(row["upstream_generation_id"], "msg_01fast", "{row}");
    assert_eq!(row["upstream_model"], "claude-opus-4-8", "{row}");
    assert_eq!(row["upstream_path"], "/v1/messages", "{row}");
    assert_eq!(row["usage_wire"], "anthropic", "{row}");
}

/// Bedrock's geographic inference profile carries the regional premium: the row names the profile
/// id that served and `price_variant=regional`.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn a_bedrock_profile_records_its_price_variant() {
    let (pubkey, sk) = test_keypair(153);
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let bedrock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "bedrock"])
        .provider_authority("bedrock", &bedrock.authority())
        .start()
        .await;
    let status = post(
        format!("{}/v1/messages", gw.url()),
        ("x-api-key", billing_vkey(&sk, 153)),
        r#"{"model":"claude-opus-4-8","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}"#,
        &[
            ("anthropic-version", "2023-06-01"),
            ("x-beyond-only", "bedrock"),
        ],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(bedrock.hits(), 1);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["provider"], "bedrock", "{row}");
    assert_eq!(
        row["upstream_model"], "us.anthropic.claude-opus-4-8",
        "{row}"
    );
    assert_eq!(row["price_variant"], "regional", "{row}");
}

const OPENROUTER_CHAT_BODY: &str = r#"{"id":"gen-1760000000-abc","provider":"Amazon Bedrock","model":"anthropic/claude-opus-4.8","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"cost":0.0012,"is_byok":false,"cost_details":{"upstream_inference_cost":null,"upstream_inference_prompt_cost":0.0008,"upstream_inference_completions_cost":0.0004,"server_tool_cost":0.007},"server_tool_use_details":{"tool_calls_executed":1,"tool_calls_requested":1,"web_search_requests":1}}}"#;

/// OpenRouter: its own cost (`usage.cost`, always sent), the serving host, the generation id from
/// `X-Generation-Id`, and the routing preferences and plugins the client asked for.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn openrouter_records_cost_host_generation_id_and_routing() {
    let (pubkey, sk) = test_keypair(154);
    let mock = MockUpstream::start(Mode::RawHeaders(
        200,
        "application/json",
        OPENROUTER_CHAT_BODY,
        &[("x-generation-id", "gen-1760000000-hdr")],
    ))
    .await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    let status = post(
        format!("{}/openrouter/api/v1/chat/completions", gw.url()),
        bearer(&sk, 154),
        r#"{"model":"anthropic/claude-opus-4.8","provider":{"order":["amazon-bedrock"]},"plugins":[{"id":"response-healing"}],"messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["upstream_cost_usd"], "0.0012", "{row}");
    assert_eq!(row["upstream_tool_cost_usd"], "0.007", "{row}");
    assert!(
        row.get("upstream_inference_cost_usd").is_none(),
        "null: {row}"
    );
    assert_eq!(row["upstream_byok"], false, "{row}");
    assert_eq!(row["served_by"], "Amazon Bedrock", "{row}");
    assert_eq!(
        row["upstream_generation_id"], "gen-1760000000-hdr",
        "the header wins over the body: {row}"
    );
    assert_eq!(
        row["requested_provider_routing"], r#"{"order":["amazon-bedrock"]}"#,
        "{row}"
    );
    assert_eq!(
        row["requested_plugins"], r#"[{"id":"response-healing"}]"#,
        "{row}"
    );
    assert_eq!(row["server_tools"], "web_search=1,tool_calls=1", "{row}");
    assert_eq!(row["model"], "anthropic/claude-opus-4.8", "{row}");
}

const OPENROUTER_SSE: &str = "data: {\"id\":\"gen-s1\",\"provider\":\"Anthropic\",\"model\":\"anthropic/claude-opus-4.8\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n\
data: {\"id\":\"gen-s1\",\"provider\":\"Anthropic\",\"model\":\"anthropic/claude-opus-4.8\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4,\"cost\":0.25,\"is_byok\":true,\"cost_details\":{\"upstream_inference_cost\":0.2}}}\n\n\
data: [DONE]\n\n";

/// OpenRouter streamed: cost and BYOK upstream cost from the final chunk, generation id and host
/// from the first.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn openrouter_stream_records_cost_and_ids() {
    let (pubkey, sk) = test_keypair(155);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", OPENROUTER_SSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .start()
        .await;
    let status = post(
        format!("{}/openrouter/api/v1/chat/completions", gw.url()),
        bearer(&sk, 155),
        r#"{"model":"anthropic/claude-opus-4.8","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["upstream_cost_usd"], "0.25", "{row}");
    assert_eq!(row["upstream_inference_cost_usd"], "0.2", "{row}");
    assert_eq!(row["upstream_byok"], true, "{row}");
    assert_eq!(row["upstream_generation_id"], "gen-s1", "{row}");
    assert_eq!(row["served_by"], "Anthropic", "{row}");
}

/// A Responses stream whose hosted-tool items come before more than the retained 64 KiB tail:
/// the per-chunk tally still counts each finished item once.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn responses_tool_items_are_counted_past_the_tail() {
    use std::sync::OnceLock;
    static SSE: OnceLock<String> = OnceLock::new();
    let sse = SSE.get_or_init(|| {
        let mut s = String::from(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_tools\",\"object\":\"response\",\"model\":\"gpt-5-2025-08-07\",\"output\":[]}}\n\n",
        );
        let items = [
            r#"{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"q"}}"#,
            r#"{"id":"ws_2","type":"web_search_call","status":"completed","action":{"type":"open_page","url":"u"}}"#,
            r#"{"id":"ci_1","type":"code_interpreter_call","status":"completed","code":"1","container_id":"cntr_1"}"#,
            r#"{"id":"fs_1","type":"file_search_call","status":"completed"}"#,
            r#"{"id":"ig_1","type":"image_generation_call","status":"completed"}"#,
        ];
        for (i, item) in items.iter().enumerate() {
            s.push_str(&format!(
                "event: response.output_item.added\ndata: {{\"type\":\"response.output_item.added\",\"output_index\":{i},\"item\":{item}}}\n\n\
                 event: response.output_item.done\ndata: {{\"type\":\"response.output_item.done\",\"output_index\":{i},\"item\":{item}}}\n\n"
            ));
        }
        for _ in 0..2000 {
            s.push_str("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"lorem ipsum dolor sit amet \"}\n\n");
        }
        s.push_str(&format!(
            "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_tools\",\"output\":[{}],\"usage\":{{\"input_tokens\":10,\"output_tokens\":4000,\"total_tokens\":4010}}}}}}\n\n",
            items.join(",")
        ));
        s
    });
    let sse: &'static str = sse.as_str();
    let (pubkey, sk) = test_keypair(156);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", sse)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let status = post(
        format!("{}/openai/v1/responses", gw.url()),
        bearer(&sk, 156),
        r#"{"model":"gpt-5","stream":true,"input":"hi","tools":[{"type":"web_search"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(
        row["server_tools"],
        "web_search=1,web_search_page=1,code_execution=1,file_search=1,image_generation=1",
        "{row}"
    );
    assert_eq!(row["server_tool_calls"].as_u64(), Some(1), "{row}");
    assert_eq!(row["container_id"], "cntr_1", "{row}");
    assert_eq!(row["upstream_generation_id"], "resp_tools", "{row}");
    assert_eq!(row["output_tokens"].as_u64(), Some(4000), "{row}");
}

/// A non-stream Responses body lists each item once (pretty-printed, as OpenAI writes it).
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn responses_body_tool_items_are_counted() {
    let body = r#"{
  "id": "resp_body",
  "object": "response",
  "model": "gpt-5-2025-08-07",
  "service_tier": "flex",
  "output": [
    { "id": "ws_1", "type": "web_search_call", "status": "completed", "action": { "type": "search", "query": "q" } },
    { "id": "ws_2", "type": "web_search_call", "status": "completed", "action": { "type": "search", "query": "r" } },
    { "id": "msg_1", "type": "message", "content": [{ "type": "output_text", "text": "hi" }] }
  ],
  "usage": { "input_tokens": 10, "output_tokens": 5, "total_tokens": 15 }
}"#;
    let (pubkey, sk) = test_keypair(157);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", body)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let status = post(
        format!("{}/openai/v1/responses", gw.url()),
        bearer(&sk, 157),
        r#"{"model":"gpt-5","input":"hi","service_tier":"flex"}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["server_tools"], "web_search=2", "{row}");
    assert_eq!(row["service_tier"], "flex", "{row}");
    assert_eq!(row["requested_service_tier"], "flex", "{row}");
    assert_eq!(row["upstream_generation_id"], "resp_body", "{row}");
}

const XAI_BODY: &str = r#"{"id":"xai-1","object":"chat.completion","model":"grok-4.3","service_tier":"priority","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":185,"completion_tokens":5,"total_tokens":190,"num_sources_used":2,"cost_in_usd_ticks":2187000,"server_side_tool_usage_details":{"web_search_calls":1,"x_search_calls":1,"x_posts_fetched":12}}}"#;

/// xAI: requested and served tier (it honours `priority`, at 2x), its exact cost in ticks, and its
/// tool details by kind.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn xai_records_tier_cost_and_tools() {
    let (pubkey, sk) = test_keypair(158);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", XAI_BODY)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["xai"])
        .start()
        .await;
    let status = post(
        format!("{}/xai/v1/chat/completions", gw.url()),
        bearer(&sk, 158),
        r#"{"model":"grok-4.3","service_tier":"priority","messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["service_tier"], "priority", "{row}");
    assert_eq!(row["requested_service_tier"], "priority", "{row}");
    assert_eq!(row["upstream_cost_usd"], "0.0002187", "{row}");
    assert_eq!(
        row["server_tools"], "web_search=1,x_search=1,x_posts=12,sources=2",
        "{row}"
    );
    assert_eq!(row["upstream_generation_id"], "xai-1", "{row}");
}

/// A stream the client cancels on OpenRouter whose drain cannot settle (the upstream goes silent
/// until the read timeout, so its usage never arrives): the row is an estimate, says which parts
/// were estimated and what the estimate cannot see, and flags that the upstream may have kept
/// generating (and billing) past the cut — reconcile with the generation id.
/// claim: BIL-23, BIL-26
/// defect: D268
#[tokio::test]
async fn a_cancelled_openrouter_stream_flags_that_the_upstream_may_continue() {
    let (pubkey, sk) = test_keypair(159);
    let mock = MockUpstream::start(Mode::StallSse).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openrouter"])
        .config_line("read_timeout_secs = 2")
        .start()
        .await;
    let mut resp = test_client()
        .post(format!("{}/openrouter/api/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", billing_vkey(&sk, 159)))
        .header("content-type", "application/json")
        .body(r#"{"model":"anthropic/claude-opus-4.8","stream":true,"messages":[{"role":"user","content":"Explain TCP congestion control."}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let first = tokio::time::timeout(Duration::from_secs(10), resp.chunk())
        .await
        .unwrap()
        .unwrap();
    assert!(first.is_some());
    drop(resp);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["outcome"], "client_cancelled", "{row}");
    assert_eq!(row["usage_estimated"], true, "{row}");
    assert_eq!(row["usage_estimated_parts"], "input,output", "{row}");
    assert_eq!(row["usage_estimate_excludes"], "cache,reasoning", "{row}");
    assert_eq!(row["upstream_may_continue"], true, "{row}");
    assert!(row.get("usage_settled").is_none(), "{row}");
    assert_eq!(row["upstream_generation_id"], "chatcmpl-mock", "{row}");
    wait_for_metric(&gw, "ai_usage_drains_total", "result=\"error\"", 1.0).await;
}

/// `web_search_call` items look the same whichever web search tool ran; the request's
/// `web_search_preview` tool (priced apart) is what tells them apart.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn web_search_preview_calls_are_counted_apart() {
    let body = r#"{"id":"resp_p","object":"response","model":"gpt-5-2025-08-07","output":[{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"q"}}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}"#;
    let (pubkey, sk) = test_keypair(162);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", body)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let status = post(
        format!("{}/openai/v1/responses", gw.url()),
        bearer(&sk, 162),
        r#"{"model":"gpt-5","input":"hi","tools":[{"type":"web_search_preview"}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["server_tools"], "web_search_preview=1", "{row}");
    assert_eq!(row["server_tool_calls"].as_u64(), Some(0), "{row}");
}

const RESPONSES_TOOL_SSE: &str = "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_walk\",\"object\":\"response\",\"model\":\"gpt-5-2025-08-07\",\"output\":[]}}\n\n\
event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fs_1\",\"type\":\"file_search_call\",\"status\":\"completed\"}}\n\n\
event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_walk\",\"output\":[{\"id\":\"fs_1\",\"type\":\"file_search_call\"}],\"usage\":{\"input_tokens\":10,\"output_tokens\":4,\"total_tokens\":14}}}\n\n";

/// A catalog walk onto a Responses candidate counts its tool items too (the walk names the
/// serving endpoint, not a forwarded path).
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn a_responses_walk_counts_tool_items() {
    let (pubkey, sk) = test_keypair(163);
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", RESPONSES_TOOL_SSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let status = post(
        format!("{}/v1/responses", gw.url()),
        bearer(&sk, 163),
        r#"{"model":"gpt-5","stream":true,"input":"hi","tools":[{"type":"file_search","vector_store_ids":["vs_1"]}]}"#,
        &[],
    )
    .await;
    assert_eq!(status, 200);
    let row = usage_row_of(&gw).await;
    assert_eq!(row["routed_model"], "gpt-5", "{row}");
    assert_eq!(row["server_tools"], "file_search=1", "{row}");
}

/// A Chat Completions walk (not Responses) is not tallied: its stream has no output items, and the
/// row's tools come from its usage block alone.
/// claim: BIL-23
/// defect: D268
#[tokio::test]
async fn a_chat_walk_counts_no_responses_items() {
    let (pubkey, sk) = test_keypair(164);
    // A chat body that happens to carry a Responses-looking done event in its text is never read
    // as one: only a Responses endpoint is tallied.
    let mock = MockUpstream::start(Mode::Raw(200, "text/event-stream", RESPONSES_TOOL_SSE)).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let _ = post(
        format!("{}/v1/chat/completions", gw.url()),
        bearer(&sk, 164),
        r#"{"model":"gpt-5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        &[],
    )
    .await;
    let row = usage_row_of(&gw).await;
    assert!(row.get("server_tools").is_none(), "{row}");
}
