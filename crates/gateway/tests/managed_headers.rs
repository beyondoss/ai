//! End-to-end: which client headers ride on Beyond's pool key.
//!
//! A managed request is sent with Beyond's credentials, so the provider gets only the client
//! headers the gateway can vouch for: framing, `accept`, `user-agent`, Anthropic's version header,
//! and `anthropic-beta` tokens that change nothing about price or server-side execution. Org and
//! project selectors, cookies and other credentials stay behind. BYO requests carry the caller's own
//! key and are forwarded untouched. Asserted at the upstream (what the mock received).
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn managed_key(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 31,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;
const MESSAGES: &str =
    r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;

/// Headers a client may send that must never reach a provider on the pool key.
const HOSTILE: [(&str, &str); 6] = [
    ("openai-organization", "org-someone-else"),
    ("openai-project", "proj_someone_else"),
    ("cookie", "session=abc"),
    ("proxy-authorization", "Basic c2VjcmV0"),
    ("x-goog-user-project", "someone-elses-project"),
    ("x-stainless-lang", "js"),
];

const BETAS: &str = "prompt-caching-2024-07-31, context-1m-2025-08-07,mcp-client-2025-04-04,interleaved-thinking-2025-05-14";

async fn post(
    gw: &Gateway,
    path: &str,
    auth: (&str, String),
    body: &'static str,
    extra: &[(&str, &str)],
) -> u16 {
    let mut req = test_client()
        .post(format!("{}{path}", gw.url()))
        .header(auth.0, auth.1)
        .header("content-type", "application/json")
        .header("user-agent", "test-sdk/1.0")
        .body(body);
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    req.send().await.unwrap().status().as_u16()
}

/// claim: SEC-6
/// defect: D04
#[tokio::test]
async fn managed_requests_forward_only_allowlisted_client_headers() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic"])
        .start()
        .await;
    let mut extra: Vec<(&str, &str)> = HOSTILE.to_vec();
    extra.push(("anthropic-beta", BETAS));
    extra.push(("anthropic-version", "2023-06-01"));

    for (path, auth, body) in [
        (
            "/openai/v1/chat/completions",
            ("authorization", format!("Bearer {}", managed_key(&sk))),
            CHAT,
        ),
        (
            "/anthropic/v1/messages",
            ("x-api-key", managed_key(&sk)),
            MESSAGES,
        ),
        (
            "/v1/chat/completions",
            ("authorization", format!("Bearer {}", managed_key(&sk))),
            CHAT,
        ),
    ] {
        assert_eq!(post(&gw, path, auth, body, &extra).await, 200, "{path}");
        let cap = mock.captured().unwrap();
        for (name, _) in HOSTILE {
            assert!(
                !cap.headers.contains_key(name),
                "{path}: {name} reached the provider on the pool key"
            );
        }
        assert_eq!(
            cap.headers.get("user-agent").unwrap(),
            "test-sdk/1.0",
            "{path}"
        );
        assert_eq!(
            cap.headers.get("content-type").unwrap(),
            "application/json",
            "{path}"
        );
        if path.starts_with("/v1/") {
            // An OpenAI-wire catalog walk drops `anthropic-version` itself; the allowlist is what
            // this checks, not that.
            continue;
        }
        assert_eq!(
            cap.anthropic_version.as_deref(),
            Some("2023-06-01"),
            "{path}"
        );
        assert_eq!(
            cap.anthropic_beta.as_deref(),
            Some("prompt-caching-2024-07-31,interleaved-thinking-2025-05-14"),
            "{path}: only the known-safe beta tokens survive"
        );
    }

    // BYO is the caller's own key: forwarded untouched.
    assert_eq!(
        post(
            &gw,
            "/openai/v1/chat/completions",
            ("authorization", "Bearer sk-byo".into()),
            CHAT,
            &extra
        )
        .await,
        200
    );
    let cap = mock.captured().unwrap();
    // `proxy-authorization` is hop-by-hop: it never crosses the proxy, BYO or not.
    for (name, value) in HOSTILE
        .into_iter()
        .filter(|(n, _)| *n != "proxy-authorization")
    {
        assert_eq!(
            cap.headers.get(name).map(|v| v.to_str().unwrap()),
            Some(value),
            "BYO {name}: {:?}",
            cap.headers
        );
    }
    assert_eq!(cap.anthropic_beta.as_deref(), Some(BETAS));
}

/// The gateway's own beta on a translated walk onto a conversation-binding Claude model still goes
/// out, merged after whatever client tokens survive the allowlist.
/// claim: SEC-6
/// defect: D04
#[tokio::test]
async fn a_translated_walk_merges_its_own_beta_with_the_allowed_client_tokens() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic", "openai", "openrouter"])
        .start()
        .await;
    const FABLE: &str =
        r#"{"model":"claude-fable-5-1","messages":[{"role":"user","content":"hi"}]}"#;
    let auth = ("authorization", format!("Bearer {}", managed_key(&sk)));
    let status = post(
        &gw,
        "/v1/chat/completions",
        auth,
        FABLE,
        &[(
            "anthropic-beta",
            "interleaved-thinking-2025-05-14,code-execution-2025-05-22",
        )],
    )
    .await;
    assert_eq!(status, 200);
    let cap = mock.captured().unwrap();
    assert_eq!(cap.path, "/v1/messages");
    assert_eq!(
        cap.anthropic_beta.as_deref(),
        Some("interleaved-thinking-2025-05-14,thinking-binding-controls-2026-08-01")
    );
}
