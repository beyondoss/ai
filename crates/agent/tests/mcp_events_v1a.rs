//! MCP Events (draft) webhook **server identity**: a server that publishes an Ed25519 key at
//! `/.well-known/mcp-webhook-jwks.json` gets its deliveries checked for a Standard Webhooks `v1a,`
//! signature as well as the HMAC — a delivery whose `v1a,` is missing or wrong is refused even
//! though its HMAC is valid. A server that publishes no key is unaffected (every other webhook
//! suite). Real `serve --listen`, real fixture, mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::mcp_events_fixture::{emit, spawn_http_fixture, state, write_settings};
use common::{
    BIN, SpawnGuarded, TestWs, free_port, serve_dir_cmd, spawn_model_server_routed, turn_text,
    wait_for_port, ws_connect, ws_next_frame, ws_send,
};
use serde_json::{Value, json};

async fn next(ws: &mut TestWs, timeout: Duration, pred: impl Fn(&Value) -> bool) -> Option<Value> {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, ws_next_frame(ws)).await {
            Ok(Some(f)) if pred(&f) => return Some(f),
            Ok(Some(_)) => {}
            _ => return None,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deliveries_must_carry_a_valid_v1a_signature_when_the_server_publishes_a_key() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_ED25519_SEED", &"07".repeat(32)),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "hooks", "transport": "http", "url": mcp_url,
            "headers": { "Authorization": "Bearer principal-1" },
            "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": "notify" }],
        }]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let _d = serve_dir_cmd(BIN, &base, &home.path().join("s").to_string_lossy())
        .args([
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--mcp-events-callback-url",
            &format!("http://127.0.0.1:{port}"),
        ])
        .env("HOME", home.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded();
    wait_for_port(port);
    let mut ws = ws_connect(port, Some("identity1")).await;
    // The verification challenge itself was v1a-signed and checked: the subscribe succeeds.
    let mut active = false;
    for n in 0..100 {
        let id = format!("l{n}");
        ws_send(&mut ws, json!({ "type": "mcp_events_list", "id": id })).await;
        let l = next(&mut ws, Duration::from_secs(20), |f| f["id"] == id.as_str())
            .await
            .unwrap();
        if l["data"]["subscriptions"][0]["state"] == "active" {
            active = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(active);
    assert_eq!(state(&fixture)["verifications"][0]["ok"], true);

    let r = emit(&fixture, json!({ "event_id": "signed", "data": {} }));
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
    assert!(
        next(&mut ws, Duration::from_secs(10), |f| f["event"]["eventId"]
            == "signed")
        .await
        .is_some()
    );

    // Valid HMAC, forged server signature → refused.
    let r = emit(
        &fixture,
        json!({ "event_id": "forged-v1a", "tamper": "v1a", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 401, "{r:#}");
    // Valid HMAC, no server signature at all → refused (the key is published, so it is required).
    let r = emit(
        &fixture,
        json!({ "event_id": "no-v1a", "tamper": "no_v1a", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 401, "{r:#}");
    assert!(
        next(&mut ws, Duration::from_millis(800), |f| f["type"]
            == "mcp_event")
        .await
        .is_none(),
        "neither refused delivery surfaced"
    );
}
