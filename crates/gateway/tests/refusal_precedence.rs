//! End-to-end: which refusal answers a body that breaks two of the gateway's body rules at once.
//!
//! A body is checked at three points: before connecting, where a catalog walk already holds a
//! small body; on a large catalog body before its `FullBody` re-run; and at the end of a body
//! the gateway buffered (a `/{provider}` relay, or that re-run). The rules run in one order, and
//! each point checks only the rules that apply there, so the same two violations can be answered
//! differently by point. Each case pins the winning refusal's status, its exact body, its
//! `ai_rejections_total` label, and what the upstream saw ([`Upstream`]): nothing, for a refusal
//! before connecting; never the whole body, for one once the request headers went upstream. A
//! foreign signed id is checked apart from the rules, and its place in the order at each point is
//! pinned too.
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

const FOREIGN_ID: &str = r#"{"error":{"message":"previous_response_id does not belong to this tenant: send back only ids from responses this gateway returned to the same tenant (an id from another tenant, from the provider directly, or altered, is refused)","type":"invalid_request_error"}}"#;
const FOREIGN_ID_STREAMED: &str = r#"{"error":{"message":"an id in this request does not belong to this tenant: send back only ids from responses this gateway returned to the same tenant","type":"invalid_request_error"}}"#;

/// A provider's own response id, as a client that called the provider directly holds it: not
/// signed for any tenant, so a managed key sending it back is refused.
const PROVIDER_ID: &str = "resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26";

const IMAGE_CHAT: &str = r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}]"#;
const IMAGE_RESPONSES: &str = r#"[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}]}]"#;

/// What the upstream may see of a refused request.
#[derive(Clone, Copy, Debug)]
enum Upstream {
    /// Nothing: refused before connecting (a small catalog body, or a large one ahead of its
    /// re-run). Counted by every request head the upstream receives.
    Untouched,
    /// The request head, never the whole body: refused in `request_body_filter`, after the head
    /// went upstream (which is why the refusal there is `DuplicateModelLate` / `ForeignIdLate`),
    /// and before a byte of the body it holds. Whether the head arrives first is a race between
    /// pingora's connect and the body read, so only requests that arrived whole are counted, as
    /// `managed_unpriced.rs` counts them.
    NoWholeBody,
}

async fn run(upstream: Upstream, cases: Vec<Case>) {
    let (pubkey, sk) = test_keypair(93);
    let heads = MockUpstream::start(Mode::Json).await;
    let whole = ScriptedUpstream::reply(200, "application/json", "{}".to_owned()).await;
    let authority = match upstream {
        Upstream::Untouched => heads.authority(),
        Upstream::NoWholeBody => whole.authority(),
    };
    let gw = Gateway::builder(unused_nats_port(), &authority, &b64(&pubkey))
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
    let seen = match upstream {
        Upstream::Untouched => heads.hits(),
        Upstream::NoWholeBody => whole.hits(),
    };
    if seen != 0 {
        failures.push(format!(
            "{upstream:?}: the upstream saw {seen} refused requests"
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Before connecting, on a small catalog body: a duplicate `model` before a non-integer output
/// limit, that before input the row does not take, that before `background`; and `background`
/// before an unpriced tool, which only the buffered body is checked for.
/// claim: SEC-21, TRN-5
#[tokio::test]
async fn before_connecting_the_first_rule_in_order_answers() {
    run(Upstream::Untouched, vec![
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
    run(Upstream::Untouched, vec![
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
    ])
    .await;
    run(Upstream::NoWholeBody, vec![
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
    run(Upstream::NoWholeBody, vec![
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

/// A small catalog body breaking two rules before connecting, past the pairs above: a duplicate
/// `model` before `background`.
/// claim: SEC-21
#[tokio::test]
async fn before_connecting_a_duplicate_model_answers_before_background() {
    run(
        Upstream::Untouched,
        vec![(
            "/v1/responses",
            r#"{"model":"gpt-5","input":"hi","background":true,"model":"gpt-4o"}"#.into(),
            400,
            DUPLICATE_MODEL,
            "duplicate_model",
        )],
    )
    .await;
}

/// The signed ids are checked apart from the body rules (verifying them rewrites the body), so
/// where they fall in the order is pinned here: before connecting, after every body rule; on a
/// large catalog body, before every rule (the ids are checked ahead of the re-run, which checks the
/// rest); in a buffered `/{provider}` body, before every rule. No row with a Responses arm refuses
/// an image or tools, so on a large body a foreign id is paired with `background`, the other rule
/// checked ahead of the re-run, and with the re-run's duplicate `model`.
/// claim: SEC-21, SEC-25
#[tokio::test]
async fn a_foreign_id_answers_in_its_place_in_the_order() {
    let pad = filler();
    run(Upstream::Untouched, vec![
        (
            "/v1/responses",
            format!(
                r#"{{"model":"gpt-5","input":"hi","previous_response_id":"{PROVIDER_ID}","model":"gpt-4o"}}"#
            ),
            400,
            DUPLICATE_MODEL,
            "duplicate_model",
        ),
        (
            "/v1/responses",
            format!(
                r#"{{"model":"gpt-5","input":"hi","background":true,"previous_response_id":"{PROVIDER_ID}"}}"#
            ),
            400,
            BACKGROUND,
            "managed_endpoint",
        ),
        (
            "/v1/responses",
            format!(
                r#"{{"model":"gpt-5","pad":"{pad}","input":"hi","background":true,"previous_response_id":"{PROVIDER_ID}"}}"#
            ),
            400,
            FOREIGN_ID,
            "foreign_id",
        ),
        (
            "/v1/responses",
            format!(
                r#"{{"model":"gpt-5","pad":"{pad}","input":"hi","previous_response_id":"{PROVIDER_ID}","model":"gpt-4o"}}"#
            ),
            400,
            FOREIGN_ID,
            "foreign_id",
        ),
    ])
    .await;
    run(Upstream::NoWholeBody, vec![
        (
            "/openai/v1/responses",
            format!(
                r#"{{"model":"gpt-5","input":"hi","previous_response_id":"{PROVIDER_ID}","model":"gpt-4o"}}"#
            ),
            400,
            FOREIGN_ID_STREAMED,
            "foreign_id",
        ),
        (
            "/openai/v1/responses",
            format!(
                r#"{{"model":"not-a-model","input":"hi","previous_response_id":"{PROVIDER_ID}","tools":[{{"type":"code_interpreter","container":{{"type":"auto"}}}}]}}"#
            ),
            400,
            FOREIGN_ID_STREAMED,
            "foreign_id",
        ),
    ])
    .await;
}
