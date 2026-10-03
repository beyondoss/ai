//! End-to-end: `GET`/`HEAD /v1/models` for a BYO caller (D254). A BYO caller never uses the
//! catalog, so its listing is not Beyond's: it relays to the provider its key belongs to (key shape,
//! then the path's default dialect for an opaque key), with the caller's key, and writes no
//! `ai.usage` row. A managed caller keeps the boot-built keyed catalog (D250).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use std::time::Duration;

const OPENAI_LIST: &str = r#"{"object":"list","data":[{"id":"gpt-provider-only","object":"model","created":1,"owned_by":"openai"}]}"#;
const ANTHROPIC_LIST: &str = r#"{"data":[{"type":"model","id":"claude-provider-only","display_name":"Provider Only","created_at":"2026-01-01T00:00:00Z"}],"has_more":false,"first_id":"claude-provider-only","last_id":"claude-provider-only"}"#;

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

struct Fixture {
    openai: MockUpstream,
    anthropic: MockUpstream,
    gw: Gateway,
    sk: ed25519_dalek::SigningKey,
}

/// OpenAI and Anthropic on separate mocks, each answering with a list the catalog does not hold.
async fn fixture() -> Fixture {
    let (pubkey, sk) = test_keypair(1);
    let openai = MockUpstream::start(Mode::Raw(200, "application/json", OPENAI_LIST)).await;
    let anthropic = MockUpstream::start(Mode::Raw(200, "application/json", ANTHROPIC_LIST)).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter"])
        .provider_authority("anthropic", &anthropic.authority())
        .start()
        .await;
    Fixture {
        openai,
        anthropic,
        gw,
        sk,
    }
}

async fn list(gw: &Gateway, headers: &[(&str, &str)]) -> (u16, String) {
    let mut req = test_client().get(format!("{}/v1/models", gw.url()));
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send().await.unwrap();
    (resp.status().as_u16(), resp.text().await.unwrap())
}

async fn assert_no_usage_rows(gw: &Gateway) {
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        usage_rows_of(gw).is_empty(),
        "a BYO listing is not ours to bill: {}",
        gw.log()
    );
}

/// An OpenAI-shaped BYO key lists OpenAI's own models: the request reaches the OpenAI mock with
/// that key, and its body comes back byte for byte.
/// claim: E4, A1
/// defect: D254
#[tokio::test]
async fn byo_openai_key_lists_openais_models() {
    let f = fixture().await;
    let (status, body) = list(&f.gw, &[("authorization", "Bearer sk-proj-caller")]).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, OPENAI_LIST);
    let cap = f.openai.captured().expect("the listing reached OpenAI");
    assert_eq!(cap.path, "/v1/models");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-proj-caller"));
    assert_eq!(f.anthropic.hits(), 0);
    assert_no_usage_rows(&f.gw).await;
}

/// An Anthropic-shaped BYO key lists Anthropic's models, with the caller's key and
/// `anthropic-version` forwarded as sent. `HEAD` relays too.
/// claim: E4, A1
/// defect: D254
#[tokio::test]
async fn byo_anthropic_key_lists_anthropics_models() {
    let f = fixture().await;
    let headers = [
        ("x-api-key", "sk-ant-api03-caller"),
        ("anthropic-version", "2023-06-01"),
    ];
    let (status, body) = list(&f.gw, &headers).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, ANTHROPIC_LIST);
    let cap = f
        .anthropic
        .captured()
        .expect("the listing reached Anthropic");
    assert_eq!(cap.path, "/v1/models");
    assert_eq!(cap.x_api_key.as_deref(), Some("sk-ant-api03-caller"));
    assert_eq!(cap.anthropic_version.as_deref(), Some("2023-06-01"));
    assert_eq!(f.openai.hits(), 0);

    let resp = test_client()
        .head(format!("{}/v1/models", f.gw.url()))
        .header("x-api-key", "sk-ant-api03-caller")
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(f.anthropic.hits(), 2);
    assert_eq!(f.openai.hits(), 0);
    assert_no_usage_rows(&f.gw).await;
}

/// An opaque BYO key says nothing about its provider, so the path's default dialect picks:
/// `/v1/models` is OpenAI's.
/// claim: E4, A1
/// defect: D254
#[tokio::test]
async fn opaque_byo_key_lists_from_the_path_default() {
    let f = fixture().await;
    let (status, body) = list(&f.gw, &[("authorization", "Bearer opaque-caller-token")]).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, OPENAI_LIST);
    let cap = f.openai.captured().expect("the listing reached OpenAI");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer opaque-caller-token")
    );
    assert_eq!(f.anthropic.hits(), 0);
    assert_no_usage_rows(&f.gw).await;
}

/// A managed key still gets the boot-built keyed catalog, and no provider is contacted.
/// claim: E4
/// defect: D254
#[tokio::test]
async fn managed_key_still_lists_the_keyed_catalog() {
    let f = fixture().await;
    let key = format!("Bearer {}", vkey(&f.sk));
    let (status, body) = list(&f.gw, &[("authorization", key.as_str())]).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    let keyed = [
        providers::ProviderId::OpenAi,
        providers::ProviderId::Anthropic,
        providers::ProviderId::OpenRouter,
    ];
    let want: Vec<&str> = providers::catalog::MODEL_ROUTES
        .iter()
        .filter(|r| r.candidates.iter().any(|c| keyed.contains(&c.provider)))
        .map(|r| r.model)
        .collect();
    assert_eq!(ids, want);
    assert!(ids.contains(&"claude-opus-4-8") && ids.contains(&"gpt-4o-mini"));
    assert_eq!((f.openai.hits(), f.anthropic.hits()), (0, 0));
    assert_no_usage_rows(&f.gw).await;
}
