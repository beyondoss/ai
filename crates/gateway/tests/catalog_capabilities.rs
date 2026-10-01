//! End-to-end: a catalog walk holds each request to what the row's card says, before and while it
//! picks a candidate.
//!
//! - A JSON-schema output request skips a candidate that cannot honor one (Amazon Bedrock).
//! - A request carrying a PDF skips a candidate that reads none (OpenRouter's grok-build-0.1).
//! - An image on a row whose card lists no image input is a 400 before any upstream sees it.
//! - An OpenRouter candidate is asked not to compress a prompt that overflows its window.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use serde_json::{Value, json};

fn provider_of(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
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

/// An inbound path, extra headers, and a body.
type Case<'a> = (&'a str, &'a [(&'a str, &'a str)], Value);

fn schema() -> Value {
    json!({"type": "object", "properties": {"answer": {"type": "integer"}},
           "required": ["answer"], "additionalProperties": false})
}

/// Bedrock's Messages surface answers `output_config.format` with a 400 (Opus 4.8) or a 404
/// (Haiku 4.5). A structured-output request on a Claude row with a Bedrock candidate is never
/// sent there, whichever client dialect asked and however the walk was ordered; without the
/// constraint Bedrock serves as before. Pinned to Bedrock alone, it is still sent: the provider's
/// own error beats a 503 for a candidate the caller chose.
/// claim: CAT-6
/// defect: D107
#[tokio::test]
async fn a_structured_output_request_skips_a_bedrock_candidate() {
    let (pubkey, sk) = test_keypair(71);
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let bedrock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &anthropic.authority(), &b64(&pubkey))
        .providers(&["anthropic", "bedrock"])
        .provider_authority("bedrock", &bedrock.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 71);
    let order = [("x-beyond-order", "bedrock")];
    let cases: [Case<'_>; 4] = [
        // Chat client, header-won row: the body is read before the walk.
        (
            "/v1/chat/completions",
            &[("x-beyond-model", "claude-haiku-4-5")],
            json!({"model": "claude-haiku-4-5", "max_tokens": 64,
                   "messages": [{"role": "user", "content": "17*23?"}],
                   "response_format": {"type": "json_schema",
                       "json_schema": {"name": "answer", "strict": true, "schema": schema()}}}),
        ),
        // Messages client, headerless: a same-wire relay to Bedrock would carry it verbatim.
        (
            "/v1/messages",
            &[],
            json!({"model": "claude-opus-4-8", "max_tokens": 64,
                   "messages": [{"role": "user", "content": "17*23?"}],
                   "output_config": {"format": {"type": "json_schema", "schema": schema()}}}),
        ),
        // Responses client.
        (
            "/v1/responses",
            &[],
            json!({"model": "claude-haiku-4-5", "input": "17*23?",
                   "text": {"format": {"type": "json_schema", "name": "answer",
                                       "schema": schema()}}}),
        ),
        // Whitespace in the client's JSON does not hide the constraint.
        (
            "/v1/chat/completions",
            &[],
            json!({"model": "claude-haiku-4-5", "max_tokens": 64,
                   "messages": [{"role": "user", "content": "17*23?"}],
                   "response_format": {"type": "json_schema",
                       "json_schema": {"name": "answer", "schema": schema()}}}),
        ),
    ];
    for (path, extra, body) in &cases {
        let mut headers = order.to_vec();
        headers.extend_from_slice(extra);
        let resp = post(&gw, &key, path, &headers, body).await;
        assert_eq!(resp.status().as_u16(), 200, "{path}");
        assert_eq!(provider_of(&resp).as_deref(), Some("anthropic"), "{path}");
    }
    assert_eq!(
        bedrock.hits(),
        0,
        "a structured-output request reached Bedrock"
    );

    // Without the constraint, the caller's order holds.
    let plain = json!({"model": "claude-haiku-4-5", "max_tokens": 64,
                       "messages": [{"role": "user", "content": "hi"}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &order, &plain).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(provider_of(&resp).as_deref(), Some("bedrock"));

    // Bedrock alone: still sent, so the client gets the provider's answer rather than a 503.
    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &[("x-beyond-only", "bedrock")],
        &cases[0].2,
    )
    .await;
    assert_eq!(provider_of(&resp).as_deref(), Some("bedrock"));
    assert_eq!(bedrock.hits(), 2);
}

/// OpenRouter's `z-ai/glm-5.2` accepts a JSON schema and some of its hosts answer outside it
/// (`{\n{\n  "answer": 391\n}`), while Together's GLM 5.2 holds it. A structured-output request on
/// the row is never sent to OpenRouter, even ordered first, from any client dialect; without the
/// constraint the caller's order holds; pinned to OpenRouter alone it is still sent.
/// claim: CAT-6
/// defect: D155
#[tokio::test]
async fn a_structured_output_request_skips_a_candidate_that_does_not_enforce_it() {
    let (pubkey, sk) = test_keypair(75);
    let together = MockUpstream::start(Mode::Json).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &together.authority(), &b64(&pubkey))
        .providers(&["together", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 75);
    let order = [("x-beyond-order", "openrouter")];
    let cases: [Case<'_>; 3] = [
        // Chat client, header-won row: the body is read before the walk.
        (
            "/v1/chat/completions",
            &[("x-beyond-model", "z-ai/glm-5.2")],
            json!({"model": "z-ai/glm-5.2", "max_tokens": 64,
                   "messages": [{"role": "user", "content": "17*23?"}],
                   "response_format": {"type": "json_schema",
                       "json_schema": {"name": "answer", "strict": true, "schema": schema()}}}),
        ),
        // Messages client, headerless, translated onto Chat Completions.
        (
            "/v1/messages",
            &[],
            json!({"model": "z-ai/glm-5.2", "max_tokens": 64,
                   "messages": [{"role": "user", "content": "17*23?"}],
                   "output_config": {"format": {"type": "json_schema", "schema": schema()}}}),
        ),
        // Responses client.
        (
            "/v1/responses",
            &[],
            json!({"model": "z-ai/glm-5.2", "input": "17*23?",
                   "text": {"format": {"type": "json_schema", "name": "answer",
                                       "schema": schema()}}}),
        ),
    ];
    for (path, extra, body) in &cases {
        let mut headers = order.to_vec();
        headers.extend_from_slice(extra);
        let resp = post(&gw, &key, path, &headers, body).await;
        assert_eq!(resp.status().as_u16(), 200, "{path}");
        assert_eq!(provider_of(&resp).as_deref(), Some("together"), "{path}");
    }
    assert_eq!(
        openrouter.hits(),
        0,
        "a structured-output request reached a candidate that does not enforce it"
    );

    // Without the constraint, the caller's order holds.
    let plain = json!({"model": "z-ai/glm-5.2", "max_tokens": 64,
                       "messages": [{"role": "user", "content": "hi"}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &order, &plain).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(provider_of(&resp).as_deref(), Some("openrouter"));

    // OpenRouter alone: still sent, so the client gets the provider's answer rather than a 503.
    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &[("x-beyond-only", "openrouter")],
        &cases[0].2,
    )
    .await;
    assert_eq!(provider_of(&resp).as_deref(), Some("openrouter"));
    assert_eq!(openrouter.hits(), 2);
}

/// A row whose card lists no image input refuses an image part with a 400 naming the row, on
/// every client dialect and whether or not a header named the row, and no upstream is contacted:
/// o3-mini would answer "I can't view images" and bill it, gpt-4 would answer 500. Text on the
/// same row, and an image on a row that reads images, are served.
/// claim: CAT-5
/// defect: D112
#[tokio::test]
async fn an_image_on_a_text_only_row_is_refused_before_any_upstream() {
    let (pubkey, sk) = test_keypair(72);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 72);
    let png = "data:image/png;base64,iVBORw0KGgo=";
    let cases: [Case<'_>; 4] = [
        (
            "/v1/chat/completions",
            &[],
            json!({"model": "gpt-4", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": png}}]}]}),
        ),
        (
            "/v1/chat/completions",
            &[("x-beyond-model", "o3-mini")],
            json!({"model": "o3-mini", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png}}]}]}),
        ),
        (
            "/v1/responses",
            &[],
            json!({"model": "o3-mini", "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": png}]}]}),
        ),
        (
            "/v1/messages",
            &[("x-beyond-model", "gpt-4")],
            json!({"model": "gpt-4", "max_tokens": 16, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "iVBORw0KGgo="}}]}]}),
        ),
    ];
    for (path, headers, body) in &cases {
        let resp = post(&gw, &key, path, headers, body).await;
        assert_eq!(resp.status().as_u16(), 400, "{path}: {body}");
        let err: Value = resp.json().await.unwrap();
        let msg = err.to_string();
        assert!(msg.contains("does not accept image input"), "{path}: {msg}");
    }
    assert_eq!(mock.hits(), 0, "a refused image reached the provider");
    wait_for_metric(&gw, "ai_rejections_total", "modality", 4.0).await;

    let text = json!({"model": "gpt-4", "messages": [{"role": "user",
                      "content": "Describe an image of a cat."}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &[], &text).await;
    assert_eq!(resp.status().as_u16(), 200);
    let seeing = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": png}}]}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &[], &seeing).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(mock.hits(), 2);
}

/// OpenRouter compresses a prompt over an 8K-or-smaller window by default (drops its middle) and
/// answers it. A catalog walk to an OpenRouter Chat Completions candidate turns that off, so an
/// over-window prompt gets the provider's context error; a client that sends its own `plugins`
/// keeps them, and other candidates' bodies are untouched.
/// claim: CAT-3
/// defect: D109
#[tokio::test]
async fn an_openrouter_candidate_is_asked_not_to_compress_the_prompt() {
    let (pubkey, sk) = test_keypair(73);
    let openai = MockUpstream::start(Mode::Json).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 73);
    let only = |p: &'static str| [("x-beyond-only", p)];
    let body = json!({"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}],
                      "stream": true});

    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &only("openrouter"),
        &body,
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    let _ = resp.bytes().await;
    let sent: Value = serde_json::from_slice(&openrouter.captured().unwrap().body).unwrap();
    assert_eq!(
        sent["plugins"],
        json!([{"id": "context-compression", "enabled": false}])
    );
    assert_eq!(sent["model"], "openai/gpt-4");
    assert_eq!(sent["stream_options"]["include_usage"], true);

    // A Messages client translated onto OpenRouter Chat Completions gets the same.
    let messages = json!({"model": "gpt-4", "max_tokens": 16,
                          "messages": [{"role": "user", "content": "hi"}]});
    let resp = post(&gw, &key, "/v1/messages", &only("openrouter"), &messages).await;
    assert_eq!(resp.status().as_u16(), 200);
    let sent: Value = serde_json::from_slice(&openrouter.captured().unwrap().body).unwrap();
    assert_eq!(sent["plugins"][0]["enabled"], false);

    // The client's own plugins are its choice.
    let own = json!({"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}],
                     "plugins": [{"id": "web"}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &only("openrouter"), &own).await;
    assert_eq!(resp.status().as_u16(), 200);
    let sent: Value = serde_json::from_slice(&openrouter.captured().unwrap().body).unwrap();
    assert_eq!(sent["plugins"], json!([{"id": "web"}]));

    // OpenAI's body carries no OpenRouter field.
    let resp = post(&gw, &key, "/v1/chat/completions", &only("openai"), &body).await;
    assert_eq!(resp.status().as_u16(), 200);
    let _ = resp.bytes().await;
    let sent: Value = serde_json::from_slice(&openai.captured().unwrap().body).unwrap();
    assert!(sent.get("plugins").is_none(), "{sent}");
}

/// A Responses body as xAI answers one (reasoning inside `output_tokens`).
const XAI_RESPONSE: &str = r#"{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"grok-build-0.1","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"MARIGOLD","annotations":[]}]}],"usage":{"input_tokens":30,"input_tokens_details":{"cached_tokens":0},"output_tokens":9,"output_tokens_details":{"reasoning_tokens":8},"total_tokens":39},"store":false}"#;

/// Grok rows read PDFs again: every grok row reaches xAI over `/v1/responses`, which reads them
/// (Chat Completions answered 400 "File content is not supported"), so their cards list file input.
/// OpenRouter's `x-ai/grok-build-0.1` reads none (404 "No endpoints found that support file
/// input"): a request carrying a file part, from any client dialect and however the walk was
/// ordered, is never sent there and lands on xAI translated onto Responses. Without a file the
/// caller's order holds; pinned to OpenRouter alone, it is still sent.
/// claim: CAT-5
/// defect: D106
#[tokio::test]
async fn a_pdf_request_skips_a_candidate_that_reads_none() {
    let (pubkey, sk) = test_keypair(74);
    let xai = MockUpstream::start(Mode::Raw(200, "application/json", XAI_RESPONSE)).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &xai.authority(), &b64(&pubkey))
        .providers(&["xai", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .start()
        .await;
    let key = billing_vkey(&sk, 74);
    let pdf = "data:application/pdf;base64,JVBERi0xLjQK";
    let order = [("x-beyond-order", "openrouter")];
    let cases: [Case<'_>; 4] = [
        // Chat client, header-won row: the body is read before the walk.
        (
            "/v1/chat/completions",
            &[("x-beyond-model", "grok-build-0.1")],
            json!({"model": "grok-build-0.1", "messages": [{"role": "user", "content": [
                {"type": "file", "file": {"filename": "a.pdf", "file_data": pdf}},
                {"type": "text", "text": "What is the code word?"}]}]}),
        ),
        // Messages client.
        (
            "/v1/messages",
            &[],
            json!({"model": "grok-build-0.1", "max_tokens": 64, "messages": [{"role": "user",
                "content": [{"type": "document", "source": {"type": "base64",
                    "media_type": "application/pdf", "data": "JVBERi0xLjQK"}},
                    {"type": "text", "text": "What is the code word?"}]}]}),
        ),
        // Responses client: a one-shot, relayed to xAI's Responses.
        (
            "/v1/responses",
            &[],
            json!({"model": "grok-build-0.1", "input": [{"role": "user", "content": [
                {"type": "input_file", "filename": "a.pdf", "file_data": pdf},
                {"type": "input_text", "text": "What is the code word?"}]}]}),
        ),
        // Spacing in the client's JSON does not hide the part.
        (
            "/v1/chat/completions",
            &[],
            json!({"model": "grok-build-0.1", "messages": [{"role": "user", "content": [
                {"type": "file", "file": {"filename": "a.pdf", "file_data": pdf}}]}]}),
        ),
    ];
    for (path, extra, body) in &cases {
        let mut headers = order.to_vec();
        headers.extend_from_slice(extra);
        let resp = post(&gw, &key, path, &headers, body).await;
        assert_eq!(resp.status().as_u16(), 200, "{path}");
        assert_eq!(provider_of(&resp).as_deref(), Some("xai"), "{path}");
        let sent = xai.captured().unwrap();
        assert_eq!(sent.path, "/v1/responses", "{path}");
        let sent: Value = serde_json::from_slice(&sent.body).unwrap();
        assert!(
            sent.to_string().contains("\"input_file\""),
            "{path}: the PDF reached xAI as input_file: {sent}"
        );
    }
    assert_eq!(
        openrouter.hits(),
        0,
        "a PDF reached OpenRouter's grok-build-0.1"
    );

    // Without a file, the caller's order holds.
    let plain = json!({"model": "grok-build-0.1", "messages": [{"role": "user", "content": "hi"}]});
    let resp = post(&gw, &key, "/v1/chat/completions", &order, &plain).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(provider_of(&resp).as_deref(), Some("openrouter"));

    // OpenRouter alone: still sent, so the client gets the provider's answer rather than a 503.
    let resp = post(
        &gw,
        &key,
        "/v1/chat/completions",
        &[("x-beyond-only", "openrouter")],
        &cases[0].2,
    )
    .await;
    assert_eq!(provider_of(&resp).as_deref(), Some("openrouter"));
    assert_eq!(openrouter.hits(), 2);
}
