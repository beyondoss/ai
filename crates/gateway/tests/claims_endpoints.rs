//! Claim tests: the client listener's protocol surface — HTTP/1.1 and h2c on one port, and
//! `Expect: 100-continue`.
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;

/// One port answers an HTTP/1.1 client and an h2c (prior-knowledge) client alike.
/// claim: E6
#[tokio::test]
async fn http1_and_h2c_are_served_on_one_port() {
    let (pubkey, sk) = test_keypair(50);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 50);
    let h1 = reqwest::Client::builder()
        .http1_only()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let h2c = reqwest::Client::builder()
        .http2_prior_knowledge()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    for (client, want) in [
        (h1, reqwest::Version::HTTP_11),
        (h2c, reqwest::Version::HTTP_2),
    ] {
        let resp = client
            .post(format!("{}/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.version(), want);
        assert_eq!(resp.status().as_u16(), 200, "{want:?}");
        assert!(
            resp.text().await.unwrap().contains("chatcmpl-mock"),
            "{want:?}"
        );
    }
    assert_eq!(mock.hits(), 2);
}

/// A client that sends `Expect: 100-continue` (curl, and .NET, past 1 KiB) and waits for the
/// interim response before its body gets a final 200, without stalling.
/// claim: E6
#[tokio::test]
async fn expect_100_continue_gets_a_final_200() {
    let (pubkey, sk) = test_keypair(51);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = billing_vkey(&sk, 51);
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer {key}\r\n\
         content-type: application/json\r\ncontent-length: {}\r\nexpect: 100-continue\r\n\
         connection: close\r\n\r\n",
        CHAT.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    // Wait for the interim response the way curl does: up to a second, then send anyway.
    let mut buf = vec![0u8; 64 * 1024];
    let mut got = Vec::new();
    if let Ok(Ok(n)) = tokio::time::timeout(Duration::from_secs(1), s.read(&mut buf)).await {
        got.extend_from_slice(&buf[..n]);
    }
    s.write_all(CHAT.as_bytes()).await.unwrap();
    let read_rest = async {
        loop {
            match s.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => got.extend_from_slice(&buf[..n]),
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), read_rest)
        .await
        .expect("an Expect request must not stall");
    // Drop any interim `100 Continue` head; what follows is the final response.
    let mut rest: &[u8] = &got;
    while rest.starts_with(b"HTTP/1.1 100") {
        let end = rest.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        rest = &rest[end..];
    }
    let resp = parse_raw_response(rest);
    assert_eq!(
        resp.status,
        200,
        "final response: {}",
        String::from_utf8_lossy(&got)
    );
    assert!(String::from_utf8_lossy(&resp.body).contains("chatcmpl-mock"));
}
