//! `serve --service`: a grant carried in `Sec-WebSocket-Protocol: beyond.grant, <grant>`.
//!
//! A browser `WebSocket` cannot set `x-beyond-grant`; the subprotocol list is the one request header
//! it can set. These cases pin the rules: the grant authenticates a real turn, it is never echoed in
//! the 101 (which names the marker only), a list that is not exactly marker-plus-one-token is a 400,
//! and a grant in both carriers is a 400 rather than a silent choice.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use common::service::Service;
use common::{
    spawn_model_server, turn_text, ws_connect_with_headers, ws_read_until_response, ws_send,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn service() -> Service {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    Service::start(&base, &["s1"]).await
}

#[tokio::test]
async fn a_subprotocol_only_client_completes_a_turn() {
    let svc = service().await;
    let token = svc.token("t1", "s1.alpha");
    let proto = format!("beyond.grant, {token}");
    let mut ws = ws_connect_with_headers(
        svc.port,
        Some("s1.alpha"),
        &[("sec-websocket-protocol", proto.as_str())],
    )
    .await
    .expect("subprotocol grant must be accepted");
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}

#[tokio::test]
async fn the_header_path_still_works() {
    let svc = service().await;
    let token = svc.token("t1", "s1.alpha");
    ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .expect("header grant must still be accepted");
}

/// Send a hand-written upgrade and return the raw bytes of the response head.
async fn raw_upgrade(port: u16, session: &str, extra: &str) -> String {
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let req = format!(
        "GET {ws}?session_id={session} HTTP/1.1\r\nHost: x\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\
         Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n{extra}\r\n",
        ws = common::WS_PATH
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    let mut buf = [0u8; 1024];
    while !out.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = s.read(&mut buf).await.unwrap();
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[tokio::test]
async fn the_101_names_the_marker_and_never_the_grant() {
    let svc = service().await;
    let token = svc.token("t1", "s1.alpha");
    let head = raw_upgrade(
        svc.port,
        "s1.alpha",
        &format!("Sec-WebSocket-Protocol: beyond.grant, {token}\r\n"),
    )
    .await;
    assert!(head.starts_with("HTTP/1.1 101"), "{head}");
    assert!(
        head.to_ascii_lowercase()
            .contains("sec-websocket-protocol: beyond.grant\r\n"),
        "{head}"
    );
    assert!(!head.contains(&token), "grant echoed in 101: {head}");
    assert!(
        !head.contains(&token[..16]),
        "part of the grant echoed: {head}"
    );
}

#[tokio::test]
async fn a_malformed_list_is_refused() {
    let svc = service().await;
    let token = svc.token("t1", "s1.alpha");
    for (what, proto) in [
        ("marker without a token", "beyond.grant".to_string()),
        ("trailing comma only", "beyond.grant,".to_string()),
        ("extra token", format!("beyond.grant, {token}, chat")),
        ("token before marker", format!("{token}, beyond.grant")),
        ("marker twice", "beyond.grant, beyond.grant".to_string()),
    ] {
        let status = ws_connect_with_headers(
            svc.port,
            Some("s1.alpha"),
            &[("sec-websocket-protocol", proto.as_str())],
        )
        .await
        .expect_err(what);
        assert_eq!(status, 400, "{what}");
    }
}

#[tokio::test]
async fn a_grant_in_both_carriers_is_ambiguous() {
    let svc = service().await;
    let token = svc.token("t1", "s1.alpha");
    let proto = format!("beyond.grant, {token}");
    let status = ws_connect_with_headers(
        svc.port,
        Some("s1.alpha"),
        &[
            ("x-beyond-grant", token.as_str()),
            ("sec-websocket-protocol", proto.as_str()),
        ],
    )
    .await
    .expect_err("both carriers must be refused");
    assert_eq!(status, 400);
}

#[tokio::test]
async fn an_unrelated_subprotocol_list_is_not_a_grant() {
    let svc = service().await;
    let status = ws_connect_with_headers(
        svc.port,
        Some("s1.alpha"),
        &[("sec-websocket-protocol", "chat, superchat")],
    )
    .await
    .expect_err("no grant");
    assert_eq!(status, 401);
}
