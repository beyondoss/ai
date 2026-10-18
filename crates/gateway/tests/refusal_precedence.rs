//! End-to-end: which refusal answers a body that breaks two of the gateway's body rules at once.
//!
//! A body is checked at three points: before connecting, where a catalog walk already holds a
//! small body; on a large catalog body before its `FullBody` re-run; and at the end of a body
//! the gateway buffered (a `/{provider}` relay, or that re-run). The rules run in one order, and
//! each point checks only the rules that apply there, so the same two violations can be answered
//! differently by point. Each case pins the winning refusal's status, its exact body, its
//! `ai_rejections_total` label, and that the upstream saw nothing.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;

/// `(path, body, status, exact response body, rejection label)`.
type Case = (&'static str, String, u16, &'static str, &'static str);

/// Filler that takes a body past pingora's 64 KiB replay buffer, so a catalog walk re-runs it as a
/// `FullBody` subrequest.
fn filler() -> String {
    "x".repeat(100 * 1024)
}

const DUPLICATE_MODEL: &str = r#"{"error":{"message":"the request body has more than one root \"model\" key; send exactly one","type":"invalid_request_error"}}"#;
const DUPLICATE_MODEL_STREAMED: &str =
    r#"{"error":{"message":"bad request","type":"invalid_request_error"}}"#;
const OUTPUT_LIMIT: &str = r#"{"error":{"message":"an output limit must be a whole number of tokens","type":"invalid_request_error"}}"#;
const NO_IMAGE: &str = r#"{"error":{"message":"deepseek-v4-pro does not accept image input","type":"invalid_request_error"}}"#;
const BACKGROUND: &str = r#"{"error":{"message":"background responses are not available with a managed key: the provider generates them after the request ends, where they cannot be metered; send the request without \"background\": true","type":"invalid_request_error"}}"#;
const CONTAINERS: &str = r#"{"error":{"message":"a managed key cannot use OpenAI's code_interpreter or hosted shell tool: container time is billed outside the usage the gateway meters; use your own provider key","type":"invalid_request_error"}}"#;
const CATALOG_MISS: &str = r#"{"error":{"message":"model \"not-a-model\" is not in the catalog; a managed key runs catalog models only (GET /v1/models lists them)","type":"invalid_request_error"}}"#;

const IMAGE_CHAT: &str = r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}]"#;
const IMAGE_RESPONSES: &str = r#"[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}]}]"#;

async fn run(cases: Vec<Case>) {
    let (pubkey, sk) = test_keypair(93);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "deepseek"])
        .start()
        .await;
    let auth = format!("Bearer {}", billing_vkey(&sk, 93));
    let mut failures = Vec::new();
    for (path, body, status, want, label) in cases {
        let reason = format!("reason=\"{label}\"");
        let before = parse_metric(&gw.metrics().await, "ai_rejections_total", &reason);
        let resp = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("authorization", &auth)
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        let got_status = resp.status().as_u16();
        let request_id = resp.headers().contains_key("x-beyond-request-id");
        let retry_after = resp.headers().get("retry-after").cloned();
        let text = resp.text().await.unwrap_or_default();
        let after = parse_metric(&gw.metrics().await, "ai_rejections_total", &reason);
        let short = &body[..body.len().min(160)];
        if got_status != status || text != want {
            failures.push(format!(
                "{path} {short}: got {got_status} {text}, want {status} {want}"
            ));
        }
        if !request_id || retry_after.is_some() {
            failures.push(format!(
                "{path} {short}: request id {request_id}, retry-after {retry_after:?}"
            ));
        }
        if after - before != 1.0 {
            failures.push(format!("{path} {short}: {label} went {before} -> {after}"));
        }
    }
    assert_eq!(mock.hits(), 0, "a refused body reached the upstream");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Before connecting, on a small catalog body: a duplicate `model` before a non-integer output
/// limit, that before input the row does not take, that before `background`; and `background`
/// before an unpriced tool, which only the buffered body is checked for.
/// claim: SEC-21, TRN-5
#[tokio::test]
async fn before_connecting_the_first_rule_in_order_answers() {
    run(vec![
        (
            "/v1/chat/completions",
            r#"{"model":"gpt-4o-mini","messages":[],"max_tokens":"9","model":"gpt-4o"}"#.into(),
            400,
            DUPLICATE_MODEL,
            "duplicate_model",
        ),
        (
            "/v1/chat/completions",
            format!(r#"{{"model":"deepseek-v4-pro","max_tokens":1e5,"messages":{IMAGE_CHAT}}}"#),
            400,
            OUTPUT_LIMIT,
            "output_limit",
        ),
        (
            "/v1/responses",
            format!(r#"{{"model":"deepseek-v4-pro","background":true,"input":{IMAGE_RESPONSES}}}"#),
            400,
            NO_IMAGE,
            "modality",
        ),
        (
            "/v1/responses",
            r#"{"model":"gpt-5","background":true,"input":"hi","tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}"#.into(),
            400,
            BACKGROUND,
            "managed_endpoint",
        ),
    ])
    .await;
}

/// A large catalog body is checked for input and `background` before its re-run, which then
/// checks the buffered body: so input the row does not take wins over a duplicate `model` here,
/// unlike on a small body, and the re-run's duplicate `model` is answered as a streamed one.
/// claim: SEC-21, TRN-5
#[tokio::test]
async fn a_large_catalog_body_is_checked_before_and_in_its_re_run() {
    let pad = filler();
    run(vec![
        (
            "/v1/chat/completions",
            format!(
                r#"{{"model":"deepseek-v4-pro","pad":"{pad}","messages":{IMAGE_CHAT},"model":"gpt-4o"}}"#
            ),
            400,
            NO_IMAGE,
            "modality",
        ),
        (
            "/v1/responses",
            format!(
                r#"{{"model":"deepseek-v4-pro","pad":"{pad}","background":true,"input":{IMAGE_RESPONSES}}}"#
            ),
            400,
            NO_IMAGE,
            "modality",
        ),
        (
            "/v1/chat/completions",
            format!(
                r#"{{"model":"gpt-4o-mini","pad":"{pad}","messages":[],"max_tokens":"9","model":"gpt-4o"}}"#
            ),
            400,
            DUPLICATE_MODEL_STREAMED,
            "duplicate_model",
        ),
        (
            "/v1/responses",
            format!(
                r#"{{"model":"gpt-5","pad":"{pad}","input":"hi","max_output_tokens":"9","tools":[{{"type":"code_interpreter","container":{{"type":"auto"}}}}]}}"#
            ),
            400,
            OUTPUT_LIMIT,
            "output_limit",
        ),
    ])
    .await;
}

/// A managed `/{provider}` body is checked once it is buffered: a duplicate `model` first, then
/// an unpriced feature, then a model outside the catalog, then `background`.
/// claim: SEC-21, BIL-23
#[tokio::test]
async fn a_buffered_provider_body_is_answered_by_the_first_rule_in_order() {
    run(vec![
        (
            "/openai/v1/chat/completions",
            r#"{"model":"not-a-model","messages":[],"tools":[{"type":"code_interpreter"}],"model":"gpt-4o"}"#.into(),
            400,
            DUPLICATE_MODEL_STREAMED,
            "duplicate_model",
        ),
        (
            "/openai/v1/responses",
            r#"{"model":"not-a-model","input":"hi","tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}"#.into(),
            400,
            CONTAINERS,
            "unpriced_feature",
        ),
        (
            "/openai/v1/responses",
            r#"{"model":"not-a-model","input":"hi","background":true}"#.into(),
            404,
            CATALOG_MISS,
            "unknown_model",
        ),
        (
            "/openai/v1/responses",
            r#"{"model":"gpt-5","input":"hi","background":true,"tools":[{"type":"code_interpreter","container":{"type":"auto"}}]}"#.into(),
            400,
            CONTAINERS,
            "unpriced_feature",
        ),
    ])
    .await;
}
