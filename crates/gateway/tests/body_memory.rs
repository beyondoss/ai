//! End-to-end: what a large request body costs the gateway in memory (SEC-19).
//!
//! A body may be up to `MAX_REQUEST_BODY`, and a catalog walk reads it whole. Each test sends a
//! body of the shape that costs a `serde_json::Value` the most per byte (many tiny objects) and
//! reads the gateway's peak resident set (`VmHWM`) before and after: the admission checks must read
//! such a body by span (D215), and translation, which does build a `Value`, must refuse one it
//! cannot hold within the memory budget rather than take gigabytes (D216).
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 215,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

const MIB: usize = 1 << 20;

/// `n` bytes' worth of `{"a":1}` items, comma-joined: 8 bytes of JSON, a `BTreeMap` node (~0.6 KiB)
/// in a `Value`.
fn tiny_objects(n: usize) -> String {
    vec![r#"{"a":1}"#; n / 8].join(",")
}

async fn gateway(mode: Mode) -> (MockUpstream, Gateway, ed25519_dalek::SigningKey) {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(mode).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "xai"])
        .start()
        .await;
    (mock, gw, sk)
}

/// POST `body` to `path`; the status, the response text, and how far the gateway's peak RSS rose
/// (KiB) while it handled the request.
async fn send(
    gw: &Gateway,
    sk: &ed25519_dalek::SigningKey,
    path: &str,
    body: String,
) -> (u16, String, u64) {
    let before = gw.peak_rss_kib();
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(sk)))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    (status, text, gw.peak_rss_kib().saturating_sub(before))
}

/// The most a request whose body is only read (copied, scanned and relayed, never parsed into a
/// `Value`) may raise the gateway's peak RSS: a handful of copies of the body (the peek buffer, the
/// re-run's copy, the upstream write), with room for allocator slack. A `Value` of these bodies
/// costs ~80x.
fn copies_budget(body: usize) -> u64 {
    (body as u64 * 12) / 1024
}

/// A Responses history of 8 MiB on a row with no Responses arm and a candidate that reads no PDF
/// (grok-build-0.1): `responses_session_field` reads `store`, `previous_response_id`,
/// `conversation` and `input` items, and `route::unserved` looks for a file part (the body says
/// `"file"`, as text). Both used to parse the whole body into a `Value`; now neither does, and the
/// request is relayed with the gateway's peak memory a few copies of the body above where it was.
/// claim: SEC-19
/// defect: D215
#[tokio::test]
async fn a_huge_responses_history_is_admitted_without_a_dom() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let body = format!(
        r#"{{"model":"grok-build-0.1","input":[{{"role":"user","content":[{{"type":"input_text","text":"file"}}]}},{}]}}"#,
        tiny_objects(8 * MIB)
    );
    let len = body.len();
    let (status, text, rise) = send(&gw, &sk, "/v1/responses", body).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(mock.hits(), 1);
    assert!(
        rise <= copies_budget(len),
        "peak RSS rose {rise} KiB for a {} KiB body (budget {} KiB)",
        len / 1024,
        copies_budget(len)
    );
}

/// The same for the image gate: an 8 MiB Chat body on a row without image input (o3-mini) whose
/// text says `"image"`. It carries no image part, so it is relayed, and reading it took no `Value`.
/// claim: SEC-19
/// defect: D215
#[tokio::test]
async fn a_huge_body_that_says_image_is_checked_without_a_dom() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let body = format!(
        r#"{{"model":"o3-mini","messages":[{{"role":"user","content":[{{"type":"text","text":"image"}},{}]}}]}}"#,
        tiny_objects(8 * MIB)
    );
    let len = body.len();
    let (status, text, rise) = send(&gw, &sk, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(mock.hits(), 1);
    assert!(
        rise <= copies_budget(len),
        "peak RSS rose {rise} KiB for a {} KiB body (budget {} KiB)",
        len / 1024,
        copies_budget(len)
    );
}

/// A 4 MiB Responses body of tiny items onto a Claude row must be translated into a Messages body,
/// which takes a `Value`: ~1 GiB for this shape (`translate::translation_heap`), more than the
/// default 512 MiB budget for buffered bodies. It is refused with a 413 that names translation,
/// before any of it reaches an upstream, and the gateway's memory stays where it was.
/// claim: SEC-19
/// defect: D216
#[tokio::test]
async fn a_body_too_costly_to_translate_is_a_413_naming_translation() {
    let (mock, gw, sk) = gateway(Mode::AnthropicJson).await;
    let body = format!(
        r#"{{"model":"claude-opus-4-8","store":false,"input":[{}]}}"#,
        tiny_objects(4 * MIB)
    );
    let len = body.len();
    let (status, text, rise) = send(&gw, &sk, "/v1/responses", body).await;
    assert_eq!(status, 413, "{text}");
    assert!(text.contains("translate"), "{text}");
    // The candidate's request head may have gone out (the body is translated as it is sent); not
    // one byte of a body did.
    assert!(
        mock.captured().is_none_or(|c| c.body.is_empty()),
        "a body reached the upstream"
    );
    assert!(
        rise <= copies_budget(len),
        "peak RSS rose {rise} KiB for a {} KiB body (budget {} KiB)",
        len / 1024,
        copies_budget(len)
    );
}

/// What the translation bound must not refuse: a 16 MiB prompt (one long string, ~5 bytes of heap
/// per byte to translate) and a 2 MiB agent history of 65,536 short messages (~45 bytes per byte),
/// both onto a Claude row from a Chat client, both well inside the budget.
/// claim: SEC-19
/// defect: D216
#[tokio::test]
async fn large_bodies_that_fit_the_budget_still_translate() {
    let (mock, gw, sk) = gateway(Mode::AnthropicJson).await;
    let long = format!(
        r#"{{"model":"claude-opus-4-8","max_tokens":16,"messages":[{{"role":"user","content":"{}"}}]}}"#,
        "a".repeat(16 * MIB)
    );
    let (status, text, _) = send(&gw, &sk, "/v1/chat/completions", long).await;
    assert_eq!(status, 200, "{text}");
    let history = format!(
        r#"{{"model":"claude-opus-4-8","max_tokens":16,"messages":[{}]}}"#,
        vec![r#"{"role":"user","content":"hi"}"#; 65_536].join(",")
    );
    let (status, text, _) = send(&gw, &sk, "/v1/chat/completions", history).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(mock.hits(), 2);
}
