//! Reliability: an HTTP/2 upstream that answers before it has read the whole request body (an
//! over-window prompt rejected from its first bytes) must have that answer relayed promptly, not
//! held behind a body write the upstream will never take.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const ERROR_BODY: &str = r#"{"error":{"message":"This endpoint's maximum context length is 400000 tokens.","type":"invalid_request_error","code":400}}"#;

/// What the upstream does after sending its complete 400. `StopReading` and `NeverAnswer` reproduce
/// D118 (accepted: the fix belongs in pingora-proxy's h2 loop); they stay so the shapes can be
/// re-checked after a pingora upgrade.
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum AfterAnswer {
    /// Keeps the stream open and never reads (or credits) another byte of the request body.
    StopReading,
    /// RFC 9113 §8.1: `RST_STREAM(NO_ERROR)`, asking the client to stop sending. The client must
    /// not discard the response.
    ResetNoError,
    /// Sends no answer at all and stops reading: a provider stalled mid-upload.
    NeverAnswer,
}

/// A TLS (ALPN `h2`) upstream that answers every request with a complete 400 as soon as its
/// headers arrive, without reading the body. Returns its port and how many requests it got.
async fn early_answer_upstream(after: AfterAnswer) -> (u16, Arc<AtomicUsize>) {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let ck = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()]).unwrap();
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(ck.key_pair.serialize_der().into());
    let mut tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![ck.cert.der().clone()], key)
        .unwrap();
    tls.alpn_protocols = vec![b"h2".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));
    let hits = Arc::new(AtomicUsize::new(0));
    let count = hits.clone();
    tokio::spawn(async move {
        while let Ok((s, _)) = listener.accept().await {
            let (acceptor, count) = (acceptor.clone(), count.clone());
            tokio::spawn(async move {
                let Ok(tls) = acceptor.accept(s).await else {
                    return;
                };
                let Ok(mut conn) = h2::server::handshake(tls).await else {
                    return;
                };
                // Request bodies kept, unread: their flow-control credit is never returned.
                let (mut held, mut pending) = (Vec::new(), Vec::new());
                while let Some(Ok((req, mut respond))) = conn.accept().await {
                    count.fetch_add(1, Ordering::SeqCst);
                    if matches!(after, AfterAnswer::NeverAnswer) {
                        held.push(req.into_body());
                        pending.push(respond);
                        continue;
                    }
                    let resp = http::Response::builder()
                        .status(400)
                        .header("content-type", "application/json")
                        .body(())
                        .unwrap();
                    let Ok(mut send) = respond.send_response(resp, false) else {
                        continue;
                    };
                    let _ = send.send_data(bytes::Bytes::from_static(ERROR_BODY.as_bytes()), true);
                    match after {
                        AfterAnswer::StopReading | AfterAnswer::NeverAnswer => {
                            held.push(req.into_body())
                        }
                        // A reset clears frames still queued, so let the answer go out first
                        // (the accept loop keeps driving the connection meanwhile).
                        AfterAnswer::ResetNoError => {
                            tokio::spawn(async move {
                                tokio::time::sleep(Duration::from_millis(200)).await;
                                send.send_reset(h2::Reason::NO_ERROR);
                            });
                        }
                    }
                }
            });
        }
    });
    (port, hits)
}

/// A chat body of about `len` bytes.
fn big_body(len: usize) -> String {
    format!(
        r#"{{"model":"gpt-4o","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "word ".repeat(len / 5)
    )
}

/// A gateway on an HTTP/2 TLS upstream whose stalled body writes give up after `write_timeout`.
async fn gateway(port: u16, write_timeout: u64) -> Gateway {
    let (pubkey, _sk) = test_keypair(1);
    Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{port}"),
        &b64(&pubkey),
    )
    .tls_upstream()
    .upstream_http2(true)
    .config_line(&format!("write_timeout_secs = {write_timeout}"))
    .start()
    .await
}

/// Send a 2.5 MB body BYO and return the status, body and how long the answer took.
async fn send_big(gw: &Gateway) -> (u16, String, Duration) {
    let start = Instant::now();
    let resp = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()
        .unwrap()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json")
        .body(big_body(2_500_000))
        .send()
        .await;
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            let body = r.text().await.unwrap_or_default();
            (status, body, start.elapsed())
        }
        Err(e) => (0, e.to_string(), start.elapsed()),
    }
}

/// The same answer followed by `RST_STREAM(NO_ERROR)`, the RFC 9113 way to say "stop sending": the
/// client must not lose the response to the reset. The reset unblocks the body write, and pingora
/// then drains the response it already read.
/// claim: CAT-3, REL-1
/// defect: D118
#[tokio::test]
async fn an_h2_answer_then_reset_no_error_is_relayed() {
    let (port, hits) = early_answer_upstream(AfterAnswer::ResetNoError).await;
    // Held behind the body write, the answer would wait out this write timeout: far past the
    // bound below, which is in turn many times what relaying it takes unloaded.
    let gw = gateway(port, 60).await;
    let (status, body, took) = send_big(&gw).await;
    assert_eq!(
        status,
        400,
        "the provider's answer, relayed: {body}\n{}",
        gw.log()
    );
    assert!(body.contains("maximum context length"), "{body}");
    assert!(took < Duration::from_secs(30), "held for {took:?}");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "an answered request is not resent"
    );
}
