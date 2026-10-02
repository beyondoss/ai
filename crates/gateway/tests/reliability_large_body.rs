//! Reliability, verify phase 0: a body past pingora's 64 KiB replay buffer (the `FullBody`
//! subrequest path) must behave like a small one in the same scenario.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const MODEL: &str = "gpt-4o-mini";

fn vkey(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 11,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

/// A chat body with `pad` bytes of content.
fn body(pad: usize) -> String {
    format!(
        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "x".repeat(pad)
    )
}

const SMALL: usize = 16;
const LARGE: usize = 200 * 1024;

/// What the client saw: status, serving provider header, error message.
#[derive(Debug, PartialEq)]
struct Outcome {
    status: u16,
    provider: Option<String>,
    message: Option<String>,
}

async fn send(gw: &Gateway, key: &str, pad: usize) -> Outcome {
    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .body(body(pad))
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let provider = resp
        .headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let text = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(String::from));
    Outcome {
        status,
        provider,
        message,
    }
}

/// The primary throttles every one of its pool keys; the fallback is healthy. A 429 is a key walk
/// on the same vendor, never a vendor switch, so both body sizes must end on the primary's 429.
/// claim: REL-21
/// defect: D51
#[tokio::test]
async fn a_429_key_walk_ends_the_same_for_small_and_large_bodies() {
    let (pubkey, sk) = test_keypair(1);
    let mut outcomes = Vec::new();
    for pad in [SMALL, LARGE] {
        let primary = MockUpstream::start(Mode::Status(429)).await;
        let fallback = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .pool_keys("openai", &["sk-a", "sk-b"])
            .start()
            .await;
        let out = send(&gw, &vkey(&sk), pad).await;
        outcomes.push((out, primary.hits(), fallback.hits()));
    }
    let (small, large) = (&outcomes[0], &outcomes[1]);
    assert_eq!(small.0.status, 429, "small: {small:?}");
    assert_eq!(small, large, "small vs large body diverged");
}

/// The primary throttles its first pool key and fails (5xx) on its second. The key walk stays on
/// the primary, and the 5xx that ends it is a candidate failure like any other: both body sizes
/// fail over to the fallback vendor.
/// claim: R1, R5, REL-21
/// defect: D81
#[tokio::test]
async fn a_5xx_after_a_key_walk_fails_over_for_small_and_large_bodies() {
    let (pubkey, sk) = test_keypair(1);
    for pad in [SMALL, LARGE] {
        let primary = ReplyUpstream::start(|_, req| match req.authorization.as_deref() {
            Some("Bearer sk-a") => Reply::json(429, r#"{"error":{"message":"slow down"}}"#),
            _ => Reply::json(500, r#"{"error":{"message":"boom"}}"#),
        })
        .await;
        let fallback = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .pool_keys("openai", &["sk-a", "sk-b"])
            .start()
            .await;
        let out = send(&gw, &vkey(&sk), pad).await;
        assert_eq!(
            (
                out.status,
                out.provider.as_deref(),
                primary.hits(),
                fallback.hits()
            ),
            (200, Some("openrouter"), 2, 1),
            "pad {pad}: {out:?}"
        );
    }
}

/// A revoked first pool key cools off (D71), so later requests start on the next key. A large body
/// re-run as a `FullBody` subrequest must honor that too, rather than paying a 401 and a second
/// upload of the whole body on every request.
/// claim: REL-14, R5
/// defect: D83
#[tokio::test]
async fn a_large_body_starts_past_a_cooling_pool_key() {
    let (pubkey, sk) = test_keypair(1);
    let revoked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = revoked.clone();
    let primary = ReplyUpstream::start(move |_, req| {
        if req.authorization.as_deref() == Some("Bearer sk-a") {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Reply::json(401, r#"{"error":{"message":"invalid api key"}}"#)
        } else {
            Reply::ok()
        }
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-a", "sk-b"])
        .start()
        .await;
    let key = vkey(&sk);
    assert_eq!(send(&gw, &key, SMALL).await.status, 200);
    for i in 0..3 {
        let out = send(&gw, &key, LARGE).await;
        assert_eq!(out.status, 200, "#{i}: {out:?}");
    }
    assert_eq!(
        revoked.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "only the first request pays the revoked key"
    );
}

/// Every candidate resets the connection. Whatever the gateway answers, a large body must get the
/// same status as a small one, and must not blame a missing provider key.
/// claim: REL-21
/// defect: D51
#[tokio::test]
async fn resets_on_every_candidate_end_the_same_for_small_and_large_bodies() {
    let (pubkey, sk) = test_keypair(1);
    let mut outcomes = Vec::new();
    for pad in [SMALL, LARGE] {
        let primary = ReplyUpstream::start(|_, _| Reply::Reset).await;
        let fallback = ReplyUpstream::start(|_, _| Reply::Reset).await;
        let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .start()
            .await;
        let out = send(&gw, &vkey(&sk), pad).await;
        outcomes.push((out, primary.hits(), fallback.hits()));
    }
    let (small, large) = (&outcomes[0], &outcomes[1]);
    let misleading = large
        .0
        .message
        .as_deref()
        .is_some_and(|m| m.contains("no provider key"));
    assert!(
        small.0.status == large.0.status && !misleading,
        "small {small:?} vs large {large:?} (outcome, primary hits, fallback hits)"
    );
}

/// The primary reads the whole request, then drops the connection without answering. It may be
/// generating, and billing, already, so the body is not resent anywhere, whatever its size: one
/// attempt, the fallback untouched, and an accurate JSON 502.
/// claim: REL-1, REL-21
/// defect: D57
#[tokio::test]
async fn a_reset_after_the_whole_body_is_not_resent_for_any_body_size() {
    let (pubkey, sk) = test_keypair(1);
    for pad in [SMALL, LARGE] {
        let primary = ReplyUpstream::start(|_, _| Reply::Reset).await;
        let fallback = MockUpstream::start(Mode::Json).await;
        let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .start()
            .await;
        let out = send(&gw, &vkey(&sk), pad).await;
        assert_eq!(
            (out.status, primary.hits(), fallback.hits()),
            (502, 1, 0),
            "pad {pad}: {out:?}"
        );
        assert!(
            out.message.is_some(),
            "pad {pad}: the 502 must carry a JSON error: {out:?}"
        );
    }
}

/// A provider that reads the request head and the first 256 KiB of the body, so the gateway is
/// mid-write, then resets the connection (no FIN: SO_LINGER 0). Returns its address, how many
/// connections it has reset, and the accept task to abort.
async fn mid_upload_resetter() -> (
    std::net::SocketAddr,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
    tokio::task::JoinHandle<()>,
) {
    use tokio::io::AsyncReadExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let resets = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = resets.clone();
    let task = tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            let mut seen = Vec::new();
            let mut buf = [0u8; 4096];
            while seen.len() < 256 * 1024 {
                match s.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => seen.extend_from_slice(&buf[..n]),
                }
            }
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = s.set_zero_linger();
            drop(s);
        }
    });
    (addr, resets, task)
}

/// An 8 MiB `/auto` walk, openai first, then openrouter.
async fn send_large_walk(gw: &Gateway, key: &str) -> u16 {
    test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", MODEL)
        .header("x-beyond-order", "openai,openrouter")
        .body(body(8 << 20))
        .send()
        .await
        .unwrap()
        .status()
        .as_u16()
}

/// An upstream that resets before it has read the whole body cannot hold the request, so the walk
/// fails over to the next candidate and the client never sees the reset. The gateway holds the body
/// before connecting (a catalog walk reads it ahead), so the reset surfaces as the write of a body
/// too large for the socket buffers failing.
/// claim: REL-1, REL-21
/// defect: D51
#[tokio::test]
async fn a_reset_during_the_upload_fails_over() {
    let (pubkey, sk) = test_keypair(1);
    let (primary, resets, task) = mid_upload_resetter().await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &primary.to_string(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let status = send_large_walk(&gw, &vkey(&sk)).await;
    task.abort();
    assert_eq!(
        (
            status,
            resets.load(std::sync::atomic::Ordering::SeqCst),
            fallback.hits()
        ),
        (200, 1, 1),
        "log:\n{}",
        gw.log()
    );
}

/// A large body's failover to the next candidate still charges the candidate it left: the reset
/// is that provider failing, so its breaker hears of it. Only a same-candidate retry (a reused
/// connection the provider had closed) gives the permit back without an outcome; a failover that
/// did the same would leave a provider resetting every large upload forever closed. With a
/// threshold of 1 the first reset opens the breaker, so the second request skips the primary.
/// claim: REL-21
#[tokio::test]
async fn a_large_body_failover_charges_the_breaker_it_left() {
    let (pubkey, sk) = test_keypair(1);
    let (primary, resets, task) = mid_upload_resetter().await;
    let fallback = MockUpstream::start(Mode::Json).await;
    // A reset far longer than the test: the breaker must still be open for request 2.
    let gw = Gateway::builder(unused_nats_port(), &primary.to_string(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .config_line("circuit_breaker_threshold = 1")
        .config_line("circuit_breaker_window_secs = 60")
        .config_line("circuit_breaker_reset_secs = 600")
        .start()
        .await;
    let key = vkey(&sk);
    let first = send_large_walk(&gw, &key).await;
    let second = send_large_walk(&gw, &key).await;
    task.abort();
    assert_eq!(
        (
            first,
            second,
            resets.load(std::sync::atomic::Ordering::SeqCst),
            fallback.hits()
        ),
        (200, 200, 1, 2),
        "(first, second, primary resets, fallback hits); log:\n{}",
        gw.log()
    );
}
