//! End-to-end: a client that sends request headers and withholds the body.
//!
//! Pingora connects the upstream before it reads a body byte, and its HTTP/2 server has no body
//! read timeout, so a BYO client (no key to check) that opened h2c streams and never sent a body
//! held a stream and an upstream connection each, bounded only by the upstream's silence timeout
//! (`read_timeout_secs`, 600 s). What must hold: the stream ends within `client_read_timeout_secs`,
//! and no upstream connection is made.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::{Duration, Instant};

/// An upstream that accepts connections and never answers.
struct Silent {
    port: u16,
    accepted: Arc<AtomicUsize>,
}

async fn silent_upstream() -> Silent {
    use tokio::io::AsyncReadExt;
    let listener = beyond_ai_test_support::ports::tokio_listener().await;
    let port = listener.local_addr().unwrap().port();
    let accepted = Arc::new(AtomicUsize::new(0));
    let a = accepted.clone();
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            a.fetch_add(1, SeqCst);
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                while matches!(s.read(&mut buf).await, Ok(n) if n > 0) {}
            });
        }
    });
    Silent { port, accepted }
}

async fn gateway(up: &Silent, lines: &[&str]) -> Gateway {
    let (pubkey, _sk) = test_keypair(70);
    let mut b = Gateway::builder(
        unused_nats_port(),
        &format!("127.0.0.1:{}", up.port),
        &b64(&pubkey),
    );
    for l in lines {
        b = b.config_line(l);
    }
    b.start().await
}

/// What a stalled request is sent as.
#[derive(Clone, Copy)]
struct Stall {
    /// The declared `content-length`, if any.
    declared: Option<usize>,
    /// Body bytes sent before going quiet.
    sent: usize,
}

const NO_BODY: Stall = Stall {
    declared: Some(100),
    sent: 0,
};

/// An h2c stream that has sent its headers (and `stall.sent` bytes) and is left open.
struct H2Stalled {
    resp: h2::client::ResponseFuture,
    _send: h2::SendStream<bytes::Bytes>,
    _client: h2::client::SendRequest<bytes::Bytes>,
}

async fn h2_open(port: u16, stall: Stall) -> H2Stalled {
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let (mut client, conn) = h2::client::handshake(tcp).await.unwrap();
    tokio::spawn(conn);
    let mut req = http::Request::post("http://gw/openai/v1/chat/completions")
        .header("authorization", "Bearer sk-byo-test")
        .header("content-type", "application/json");
    if let Some(n) = stall.declared {
        req = req.header("content-length", n.to_string());
    }
    let (resp, mut send) = client.send_request(req.body(()).unwrap(), false).unwrap();
    if stall.sent > 0 {
        send.send_data(bytes::Bytes::from(vec![b' '; stall.sent]), false)
            .unwrap();
    }
    H2Stalled {
        resp,
        _send: send,
        _client: client,
    }
}

/// Status, `retry-after` and elapsed time of the gateway's answer, or `None` if there was none by
/// `limit`.
async fn h2_answer(s: H2Stalled, limit: Duration) -> Option<(u16, Option<String>, Duration)> {
    let t = Instant::now();
    let r = tokio::time::timeout(limit, s.resp).await.ok()?.ok()?;
    let retry = r
        .headers()
        .get("retry-after")
        .map(|v| v.to_str().unwrap().to_owned());
    Some((r.status().as_u16(), retry, t.elapsed()))
}

async fn h1_answer(port: u16, limit: Duration) -> Option<(u16, Duration)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    s.write_all(
        b"POST /openai/v1/chat/completions HTTP/1.1\r\nhost: gw\r\nauthorization: Bearer sk-byo-test\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\n",
    )
    .await
    .unwrap();
    let t = Instant::now();
    let mut buf = [0u8; 1024];
    let n = tokio::time::timeout(limit, s.read(&mut buf)).await.ok()?;
    let text = String::from_utf8_lossy(&buf[..n.unwrap_or(0)]).into_owned();
    let status = text.get(9..12)?.parse().ok()?;
    Some((status, t.elapsed()))
}

/// A BYO client that sends headers and no body is answered 408 at `client_read_timeout_secs`, with
/// no upstream connection made, over h2c (where pingora has no body read timeout and the
/// upstream's 600 s silence bound used to be the only one) and over HTTP/1.1. Declared length or
/// not, and after a first piece of a small body that never completes.
/// claim: REL-1, SEC-16
/// defect: D300
#[tokio::test]
async fn a_client_that_withholds_its_body_is_cut_at_client_read_timeout() {
    let up = silent_upstream().await;
    // The upstream silence bound is far past the test, so only the client bound can answer.
    let gw = gateway(
        &up,
        &["client_read_timeout_secs = 2", "read_timeout_secs = 60"],
    )
    .await;
    let cases = [
        ("no body", NO_BODY),
        (
            "no content-length",
            Stall {
                declared: None,
                sent: 0,
            },
        ),
        (
            "half a small body",
            Stall {
                declared: Some(100),
                sent: 50,
            },
        ),
    ];
    for (what, stall) in cases {
        let (status, _, took) = h2_answer(h2_open(gw.port, stall).await, Duration::from_secs(20))
            .await
            .unwrap_or_else(|| panic!("h2c {what}: no answer\n{}", gw.log()));
        assert_eq!(status, 408, "h2c {what}");
        assert!(
            took >= Duration::from_millis(1500) && took < Duration::from_secs(10),
            "h2c {what}: {took:?}"
        );
    }
    let (status, took) = h1_answer(gw.port, Duration::from_secs(20))
        .await
        .expect("http/1.1: an answer");
    assert_eq!(status, 408, "http/1.1");
    assert!(took < Duration::from_secs(10), "http/1.1: {took:?}");
    assert_eq!(
        up.accepted.load(SeqCst),
        0,
        "a stalled upload reached the upstream\n{}",
        gw.log()
    );
}

/// A large body's first chunk starts it, so the request is let through to the upstream; a stall
/// after that is bounded by the upstream silence bound (`read_timeout_secs`), which is the
/// documented residual. The slot comes back when it ends.
/// claim: REL-1
/// defect: D300
#[tokio::test]
async fn a_large_upload_that_stalls_after_its_first_chunk_ends_at_the_upstream_bound() {
    let up = silent_upstream().await;
    let gw = gateway(
        &up,
        &["client_read_timeout_secs = 1", "read_timeout_secs = 3"],
    )
    .await;
    let stall = Stall {
        declared: Some(1_000_000),
        sent: 1000,
    };
    let (status, _, took) = h2_answer(h2_open(gw.port, stall).await, Duration::from_secs(20))
        .await
        .expect("an answer");
    assert_eq!(status, 408, "{took:?}\n{}", gw.log());
    assert!(took >= Duration::from_secs(2), "{took:?}");
    assert_eq!(up.accepted.load(SeqCst), 1);
}
