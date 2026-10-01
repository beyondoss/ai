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

/// Send one raw HTTP/1.1 request (so `Connection: Upgrade` goes out exactly as written) and read
/// the response.
async fn raw_request(port: u16, request: String) -> RawResponse {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    s.write_all(request.as_bytes()).await.unwrap();
    // Read until the head and its `content-length` body are in: the connection stays open.
    let mut buf = Vec::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let mut chunk = [0u8; 4096];
        let n = match tokio::time::timeout_at(deadline, s.read(&mut chunk)).await {
            Ok(Ok(n)) if n > 0 => n,
            _ => break,
        };
        buf.extend_from_slice(&chunk[..n]);
        let resp = parse_raw_response(&buf);
        let want = resp
            .header("content-length")
            .and_then(|v| v.parse::<usize>().ok());
        if want.is_some_and(|len| resp.body.len() >= len) {
            return resp;
        }
    }
    parse_raw_response(&buf)
}

/// A managed WebSocket upgrade would be an opaque, unmetered relay on the pool key. It is refused
/// with a 400 before any upstream contact, on a generation path as much as on `/realtime`.
/// claim: SEC-20
/// defect: D33
#[tokio::test]
async fn a_managed_upgrade_is_refused_before_the_upstream() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let vk = managed_key(&sk);
    for (method, path, body) in [
        ("POST", "/openai/v1/chat/completions", CHAT),
        ("GET", "/openai/v1/realtime?model=gpt-realtime", ""),
        ("POST", "/v1/chat/completions", CHAT),
    ] {
        let req = format!(
            "{method} {path} HTTP/1.1\r\nhost: 127.0.0.1\r\nauthorization: Bearer {vk}\r\n\
             connection: Upgrade\r\nupgrade: websocket\r\nsec-websocket-version: 13\r\n\
             sec-websocket-key: dGhlIHNhbXBsZSBub25jZQ==\r\ncontent-type: application/json\r\n\
             content-length: {}\r\n\r\n{body}",
            body.len()
        );
        let resp = raw_request(gw.port, req).await;
        assert_eq!(resp.status, 400, "{method} {path}: {resp:?}");
        let text = String::from_utf8_lossy(&resp.body);
        assert!(text.contains("managed key"), "{text}");
        assert!(resp.header("x-beyond-request-id").is_some());
    }
    assert_eq!(mock.hits(), 0, "no upgrade may reach the provider");
}

/// `x-beyond-*` response headers are the gateway's. One sent by the upstream is dropped before the
/// gateway adds its own, so a client never reads a provider's (or a middlebox's) claim about who
/// served it or whether it was a cache replay.
/// claim: SEC-22
/// defect: D53
#[tokio::test]
async fn upstream_x_beyond_headers_never_reach_the_client() {
    let (pubkey, sk) = test_keypair(1);
    let body = br#"{"id":"chatcmpl-mock","object":"chat.completion","model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
    let reply = {
        let mut r = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n\
             x-beyond-provider: evil\r\nx-beyond-cache-status: hit\r\n\
             x-beyond-request-id: forged\r\nx-beyond-anything: 1\r\ncontent-length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        r.extend_from_slice(body);
        r
    };
    let upstream = ScriptedUpstream::start(move |_, _| vec![Step::Write(reply.clone())]).await;
    let gw = Gateway::builder(unused_nats_port(), &upstream.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    for auth in [
        format!("Bearer {}", managed_key(&sk)),
        "Bearer sk-byo".to_owned(),
    ] {
        let resp = test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", &auth)
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let h = resp.headers();
        assert_eq!(h.get_all("x-beyond-provider").iter().count(), 1);
        assert_eq!(h.get("x-beyond-provider").unwrap(), "openai");
        assert_eq!(h.get_all("x-beyond-request-id").iter().count(), 1);
        assert_ne!(h.get("x-beyond-request-id").unwrap(), "forged");
        assert!(h.get("x-beyond-cache-status").is_none());
        assert!(h.get("x-beyond-anything").is_none());
    }
}

/// Pool keys over a plaintext upstream are sent in cleartext on every managed request. Legitimate
/// only against a local mock (as here), so it boots, but loudly.
/// claim: SEC-8
/// defect: D53
#[tokio::test]
async fn pool_keys_over_cleartext_warn_at_boot() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    gw.wait_for_log_line(&["upstream_tls is DISABLED while pool keys are configured"])
        .await;
}
