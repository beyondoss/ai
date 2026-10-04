//! End-to-end: managed catalog walks whose `model` is not in the first 64 KiB of the body.
//!
//! Stock Python SDKs serialize `model` *after* the large field (`messages`, `input`), so an agent
//! turn or an embeddings batch past pingora's 64 KiB retry buffer used to 404 ("missing model") or,
//! when the read that found `model` also finished the body, hang until the client timed out. The
//! gateway now reads on, and re-runs a fully read body as a subrequest (`FullBody` in `proxy.rs`).
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use std::time::Duration;

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

/// A client that gives up well before a hung request would.
fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

/// `{"messages":[…filler…],"model":…}`: the openai-python field order, `model` last.
fn chat_model_last(filler_bytes: usize) -> String {
    let filler = "x".repeat(filler_bytes);
    format!(
        r#"{{"messages":[{{"role":"user","content":"{filler}"}}],"model":"gpt-4o-mini","stream":false}}"#
    )
}

async fn gateway(mode: Mode) -> (MockUpstream, Gateway, ed25519_dalek::SigningKey) {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(mode).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;
    (mock, gw, sk)
}

/// claim: R5, REL-21
#[tokio::test]
async fn a_large_chat_body_with_model_last_is_served_and_billed() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let body = chat_model_last(200 * 1024);
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "{}",
        resp.text().await.unwrap()
    );

    let cap = mock.captured().expect("forwarded");
    assert_eq!(cap.path, "/v1/chat/completions");
    let sent = String::from_utf8(cap.body).unwrap();
    assert_eq!(
        sent.len(),
        body.len(),
        "the whole body reached the provider"
    );
    assert!(
        sent.contains(r#""model":"gpt-4o-mini""#),
        "model spliced for the candidate"
    );

    let line = gw
        .wait_for_log_line(&["ai.usage", r#""model":"gpt-4o-2024-08-06""#])
        .await;
    assert!(line.contains(r#""input_tokens":11"#), "{line}");
    let metrics = gw.metrics().await;
    assert_eq!(
        parse_metric(&metrics, "ai_full_body_relays_total", ""),
        1.0,
        "{metrics}"
    );
    // One request as far as the client and the counters are concerned.
    assert_eq!(parse_metric(&metrics, "ai_requests_total", ""), 1.0);
    assert_eq!(mock.hits(), 1);
}

/// The sizes that used to hang: the read that found `model` just past 64 KiB also ended the body.
#[tokio::test]
async fn bodies_just_past_the_replay_buffer_do_not_hang() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    for filler in [
        64 * 1024 - 200,
        64 * 1024,
        64 * 1024 + 100,
        70 * 1024,
        107 * 1024,
    ] {
        let resp = client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk)))
            .header("content-type", "application/json")
            .body(chat_model_last(filler))
            .send()
            .await
            .unwrap_or_else(|e| panic!("filler {filler}: {e}"));
        assert_eq!(resp.status().as_u16(), 200, "filler {filler}");
    }
    assert_eq!(mock.hits(), 5);
}

/// The dominant embeddings shape: a LangChain-style batch, `input` first, well past 64 KiB.
/// claim: M1
#[tokio::test]
async fn a_large_embeddings_batch_with_input_first_is_served() {
    let (mock, gw, sk) = gateway(Mode::Embeddings).await;
    let chunks: Vec<String> = (0..400)
        .map(|i| format!("\"document chunk {i}: {}\"", "lorem ipsum ".repeat(20)))
        .collect();
    let body = format!(
        r#"{{"input":[{}],"model":"text-embedding-3-small","encoding_format":"base64"}}"#,
        chunks.join(",")
    );
    assert!(body.len() > 64 * 1024);
    let resp = client()
        .post(format!("{}/v1/embeddings", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let cap = mock.captured().expect("forwarded");
    assert_eq!(cap.path, "/v1/embeddings");
    assert_eq!(cap.body.len(), body.len());
    let line = gw
        .wait_for_log_line(&["ai.usage", "text-embedding-3-small"])
        .await;
    assert!(line.contains(r#""input_tokens":5"#), "{line}");
}

/// Responses session state after a large `input` must still pick the Responses arm. Before, only a
/// body under 64 KiB was read for it, so a long Codex-style turn with `previous_response_id` was
/// taken for a one-shot and translated onto Chat Completions, dropping the conversation.
#[tokio::test]
async fn large_responses_session_state_after_input_walks_the_responses_arm() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let filler = "y".repeat(150 * 1024);
    // The id as the gateway issued it to tenant 42 (`signed_id.rs`); the upstream gets `resp_123`.
    let prev = dev_id_signer().sign(42, "resp_123");
    let body = format!(
        r#"{{"input":"{filler}","model":"gpt-4o-mini","previous_response_id":"{prev}","store":true}}"#
    );
    let resp = client()
        .post(format!("{}/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    let _ = resp.status();
    let cap = mock.captured().expect("forwarded");
    assert_eq!(
        cap.path, "/v1/responses",
        "session state walks the Responses arm"
    );
    let sent = String::from_utf8(cap.body).unwrap();
    assert!(
        sent.contains(r#""previous_response_id":"resp_123""#),
        "byte relay keeps session state"
    );
}

#[tokio::test]
async fn a_large_body_without_a_model_is_a_404_not_a_hang() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let filler = "z".repeat(120 * 1024);
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(format!(
            r#"{{"messages":[{{"role":"user","content":"{filler}"}}]}}"#
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
    assert_eq!(mock.hits(), 0);
}

/// `model` first (the Rust and Go SDK order) takes the same path once the body outgrows the replay
/// buffer: it is read in full and re-run, which is what lets it fail over.
#[tokio::test]
async fn a_large_body_with_model_first_is_relayed_too() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let filler = "x".repeat(200 * 1024);
    let body =
        format!(r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{filler}"}}]}}"#);
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(mock.captured().expect("forwarded").body.len(), body.len());
    let metrics = gw.metrics().await;
    assert_eq!(parse_metric(&metrics, "ai_full_body_relays_total", ""), 1.0);
}

/// A small body is still a plain pingora relay: no subrequest.
#[tokio::test]
async fn a_small_body_is_not_relayed() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(r#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o-mini"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(mock.hits(), 1);
    let metrics = gw.metrics().await;
    assert_eq!(parse_metric(&metrics, "ai_full_body_relays_total", ""), 0.0);
}

// ---- Relay audit round 2 ----

fn big_chat(model: &str) -> String {
    let filler = "x".repeat(200 * 1024);
    format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"{filler}"}}]}}"#)
}

/// An HTTP/2 client (the default for the agent's `h2c` serve mode) used to get a bare 400: the
/// subrequest was built by rendering the H2 request line as `HTTP/2`, which its parser rejects.
/// claim: E6, REL-21, REL-11
#[tokio::test]
async fn an_h2c_client_can_send_a_large_body() {
    let (mock, gw, sk) = gateway(Mode::Json).await;
    let h2 = reqwest::Client::builder()
        .http2_prior_knowledge()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    for model_first in [true, false] {
        let body = if model_first {
            big_chat("gpt-4o-mini")
        } else {
            let filler = "x".repeat(200 * 1024);
            format!(
                r#"{{"messages":[{{"role":"user","content":"{filler}"}}],"model":"gpt-4o-mini"}}"#
            )
        };
        let resp = h2
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk)))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.version(), reqwest::Version::HTTP_2);
        assert_eq!(resp.status().as_u16(), 200, "model_first={model_first}");
        assert!(resp.headers().contains_key("x-beyond-request-id"));
        let cap = mock.captured().expect("forwarded");
        assert_eq!(
            cap.body.len(),
            body.len(),
            "the subrequest carried the whole body"
        );
    }
}

/// The tenant cap is taken before the body is read, so it bounds the bodies held in memory: a
/// tenant at its cap gets a 429 at once instead of the gateway buffering a slow upload.
#[tokio::test]
async fn the_tenant_cap_is_checked_before_a_large_body_is_read() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (pubkey, sk) = test_keypair(1);
    // The holder's answer is held until the probe has its 429, so the tenant is at its cap for as
    // long as the probe needs on any runner: a 3 s slow reply and a 300 ms head start lost that
    // race on a starved host.
    let release = std::sync::Arc::new(tokio::sync::Notify::new());
    let held = std::sync::Arc::clone(&release);
    let slow = ReplyUpstream::start(move |_, _| {
        Reply::Held(std::sync::Arc::clone(&held), Box::new(Reply::ok()))
    })
    .await;
    let gw = Gateway::builder(unused_nats_port(), &slow.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .tenant_max_in_flight(1)
        .start()
        .await;
    let key = vkey(&sk);
    let url = format!("{}/v1/chat/completions", gw.url());
    let holder = {
        let (url, key) = (url.clone(), key.clone());
        tokio::spawn(async move {
            client()
                .post(url)
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#)
                .send()
                .await
        })
    };
    // The holder has the tenant's one slot once its request reaches the upstream.
    let deadline = std::time::Instant::now() + CONDITION_BUDGET;
    while slow.hits() == 0 && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(slow.hits(), 1, "the holder reached the upstream");

    // A slow upload: declares 300 KB, sends 2 KB, then waits.
    let addr = gw.url().trim_start_matches("http://").to_owned();
    let mut sock = tokio::net::TcpStream::connect(&addr).await.unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: {addr}\r\nauthorization: Bearer {key}\r\n\
         content-type: application/json\r\ncontent-length: 300000\r\n\r\n"
    );
    sock.write_all(head.as_bytes()).await.unwrap();
    sock.write_all(
        format!(
            r#"{{"messages":[{{"role":"user","content":"{}"#,
            "x".repeat(2000)
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut buf = vec![0u8; 512];
    // The upload never finishes, so any answer at all came before the body was read; a gateway
    // that buffered first could only ever answer with a body-read failure, never this 429. The
    // bound is a stall guard, not the claim, so it is the generous one.
    let n = tokio::time::timeout(CONDITION_BUDGET, sock.read(&mut buf))
        .await
        .expect("answered before the upload finished, not after buffering it")
        .unwrap();
    let text = String::from_utf8_lossy(&buf[..n]);
    assert!(text.starts_with("HTTP/1.1 429"), "{text}");
    release.notify_one();
    let _ = holder.await;
}

/// An abandoned attempt holds nothing the next one needs: with a cap of 1, a large-body failover
/// completes even when the failed upstream never finishes its error body.
/// claim: REL-21
#[tokio::test]
async fn a_large_body_failover_fits_a_tenant_cap_of_one() {
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::StatusThenStall(500)).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    // One worker: the next attempt runs before an unawaited abandoned one could let go.
    let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .tenant_max_in_flight(1)
        .worker_threads(1)
        .start()
        .await;
    // A held slot shows as a 429; an attempt waiting on the stalled error body, as no answer
    // until that upstream's 600s read timeout. Unloaded each takes well under a second, so the
    // bound below is a stall guard many times over, not a speed claim a loaded host could fail.
    let patient = reqwest::Client::builder()
        .timeout(CONDITION_BUDGET * 2)
        .build()
        .unwrap();
    for _ in 0..3 {
        let started = std::time::Instant::now();
        let resp = patient
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk)))
            .header("content-type", "application/json")
            .body(big_chat("gpt-4o-mini"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        assert!(
            started.elapsed() < CONDITION_BUDGET,
            "{:?}",
            started.elapsed()
        );
    }
}

/// One request, one billing row: the abandoned attempt writes none, and every attempt shares the
/// id the client was given.
/// claim: R5, O1, B1, BIL-14, BIL-19, REL-21
#[tokio::test]
async fn a_relayed_failover_writes_one_usage_row_under_the_clients_id() {
    let (pubkey, sk) = test_keypair(1);
    let primary = MockUpstream::start(Mode::Status(500)).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &primary.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    let resp = client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk)))
        .header("content-type", "application/json")
        .body(big_chat("gpt-4o-mini"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let id = resp
        .headers()
        .get("x-beyond-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap()
        .to_owned();
    let line = gw
        .wait_for_log_line(&["\"target\":\"ai.usage\"", r#""provider":"openrouter""#])
        .await;
    assert!(
        line.contains(&id),
        "the served row carries the client's id {id}: {line}"
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    let rows = gw
        .log()
        .lines()
        .filter(|l| l.contains("\"target\":\"ai.usage\""))
        .count();
    assert_eq!(rows, 1, "{}", gw.log());
}

/// A connection that fails before any response header **after the upstream drained the whole
/// body** (the mock reads it, then closes a reused connection) is not resent, to that candidate or
/// the next: the provider may already be generating, and billing, the answer. The request ends
/// with a JSON 502 and the fallback never sees the body.
///
/// This test used to expect the held body to be retried and every request to succeed — the
/// unsafe behavior D09 removed. The mock's shape (drain, then close) is exactly the case the
/// policy forbids resending, so the expectation changed with it.
#[tokio::test]
async fn a_reset_after_the_body_was_delivered_is_not_resent() {
    let (pubkey, sk) = test_keypair(1);
    let flaky = MockUpstream::start(Mode::CloseOnReusedConnection).await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &flaky.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fallback.authority())
        .start()
        .await;
    const SENT: usize = 4;
    let mut failed = 0;
    for _ in 0..SENT {
        let resp = client()
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {}", vkey(&sk)))
            .header("content-type", "application/json")
            .header("x-beyond-order", "openai,openrouter")
            .body(big_chat("gpt-4o-mini"))
            .send()
            .await
            .unwrap();
        match resp.status().as_u16() {
            200 => {}
            502 => {
                failed += 1;
                let text = resp.text().await.unwrap_or_default();
                let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
                assert!(v["error"]["message"].is_string(), "a JSON error: {text}");
            }
            s => panic!("unexpected status {s}; log:\n{}", gw.log()),
        }
    }
    assert!(
        failed > 0,
        "no reused connection was closed, so the scenario did not fire"
    );
    assert_eq!(
        flaky.hits(),
        SENT,
        "a body was sent twice; log:\n{}",
        gw.log()
    );
    assert_eq!(
        fallback.hits(),
        0,
        "the fallback was handed a delivered body"
    );
}
