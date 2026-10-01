//! End-to-end: a credential never travels further than it must.
//!
//! A managed virtual key authenticates the caller to the gateway and nothing else: it must not reach
//! a provider in any location (header, query string), and it must not land in the gateway's own
//! error log. A BYO key goes to the provider it belongs to and to no other. Every assertion is made
//! at the upstream (what the mock received) or in the log, not at the status code.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn managed_key(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 21,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;

/// Every header value and the path the upstream saw, joined, for a substring search.
fn everything_forwarded(cap: &Captured) -> String {
    let mut out = cap.path.clone();
    for (k, v) in &cap.headers {
        out.push('\n');
        out.push_str(k.as_str());
        out.push_str(": ");
        out.push_str(&String::from_utf8_lossy(v.as_bytes()));
    }
    out
}

/// A managed `?key=` (Gemini's convention) is the virtual key: the provider gets the path with
/// every other query param intact and no `key=`.
/// claim: SEC-3
/// defect: D03
#[tokio::test]
async fn a_managed_query_key_is_not_forwarded_upstream() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let vk = managed_key(&sk);
    let resp = test_client()
        .post(format!(
            "{}/openai/v1/chat/completions?api-version=2024-10-21&key={vk}&x=1",
            gw.url()
        ))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let cap = mock.captured().expect("the request reached the upstream");
    assert_eq!(cap.path, "/v1/chat/completions?api-version=2024-10-21&x=1");
    let seen = everything_forwarded(&cap);
    assert!(!seen.contains("bai_v"), "virtual key forwarded:\n{seen}");
}

/// Pingora's own error line ("Fail to proxy: …, POST /path?query, Host: …") must not print a
/// credential carried in the query, managed or BYO.
/// claim: SEC-4
/// defect: D03
#[tokio::test]
async fn a_query_key_never_reaches_the_error_log() {
    let (pubkey, sk) = test_keypair(1);
    let gw = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{}", closed_port()),
        &b64(&pubkey),
    )
    .providers(&["openai"])
    .start()
    .await;
    let vk = managed_key(&sk);
    for key in [vk.as_str(), "AIzaByoGoogleSecret"] {
        let resp = test_client()
            .post(format!("{}/openai/v1/chat/completions?key={key}", gw.url()))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_server_error(), "{}", resp.status());
    }
    let log = gw.wait_for_log_line(&["Fail to proxy"]).await;
    assert!(!log.contains("?key="), "{log}");
    let log = gw.log();
    assert!(!log.contains("bai_v"), "virtual key in the log:\n{log}");
    assert!(
        !log.contains("AIzaByoGoogleSecret"),
        "BYO key in the log:\n{log}"
    );
}

/// Credentials that disagree never forward a virtual key: a managed key in any location makes the
/// request managed, and a managed request has every credential location stripped.
/// claim: SEC-11
/// defect: D29
#[tokio::test]
async fn mixed_credentials_never_forward_a_virtual_key() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_key("openai", "sk-pool-openai")
        .start()
        .await;
    let vk = managed_key(&sk);
    let cases: Vec<Vec<(&str, String)>> = vec![
        vec![
            ("x-api-key", "junk".into()),
            ("authorization", format!("Bearer {vk}")),
        ],
        vec![
            ("x-api-key", String::new()),
            ("authorization", format!("Bearer {vk}")),
        ],
        vec![
            ("api-key", "sk-someone-else".into()),
            ("x-goog-api-key", vk.clone()),
        ],
    ];
    for headers in &cases {
        let mut req = test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("content-type", "application/json")
            .body(CHAT);
        for (k, v) in headers {
            req = req.header(*k, v);
        }
        let resp = req.send().await.unwrap();
        assert_eq!(resp.status(), 200, "{headers:?}");
        let cap = mock.captured().unwrap();
        let seen = everything_forwarded(&cap);
        assert!(!seen.contains("bai_"), "{headers:?} forwarded:\n{seen}");
        assert_eq!(
            cap.authorization.as_deref(),
            Some("Bearer sk-pool-openai"),
            "{headers:?}: managed, so the pool key is what goes upstream"
        );
    }
}

/// An Anthropic SDK call to a bare `/v1` path other than `/v1/messages` (files, a model lookup)
/// carries its key in `x-api-key`. It goes to Anthropic, and never to OpenAI.
/// claim: SEC-11
/// defect: D30
#[tokio::test]
async fn a_byo_anthropic_key_on_bare_v1_never_reaches_openai() {
    let (pubkey, _sk) = test_keypair(1);
    let openai = MockUpstream::start(Mode::Json).await;
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic"])
        .provider_authority("anthropic", &anthropic.authority())
        .start()
        .await;
    for (method, path) in [
        (reqwest::Method::GET, "/v1/files"),
        (reqwest::Method::GET, "/v1/models/claude-opus-4-8"),
        (reqwest::Method::POST, "/v1/messages/batches"),
    ] {
        let resp = test_client()
            .request(method.clone(), format!("{}{path}", gw.url()))
            .header("x-api-key", "sk-ant-byo-secret")
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{method} {path}");
        let cap = anthropic.captured().unwrap();
        assert_eq!(cap.path, path);
        assert_eq!(cap.x_api_key.as_deref(), Some("sk-ant-byo-secret"));
    }
    assert_eq!(anthropic.hits(), 3);
    assert_eq!(openai.hits(), 0, "an Anthropic key must never reach OpenAI");

    // A Bearer key on the same paths is still OpenAI's.
    let resp = test_client()
        .get(format!("{}/v1/files", gw.url()))
        .header("authorization", "Bearer sk-openai-byo")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(openai.hits(), 1);
    assert!(openai.captured().unwrap().x_api_key.is_none());
}
