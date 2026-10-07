//! MCP Events (draft) webhook **server identity**: a server that publishes an Ed25519 key at
//! `/.well-known/mcp-webhook-jwks.json` gets its deliveries checked for a Standard Webhooks `v1a,`
//! signature as well as the HMAC — a delivery whose `v1a,` is missing or wrong is refused even
//! though its HMAC is valid. A server that publishes no key is unaffected (every other webhook
//! suite). Real `serve --listen`, real fixture, mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::mcp_events_fixture::{
    EVENTS_SESSION, control, emit, eventually, spawn_daemon, spawn_http_fixture, state,
    write_settings, ws_next, ws_wait_active,
};
use common::{
    BIN, HeldPort, SpawnGuarded, TestWs, serve_dir_cmd, spawn_model_server_routed, turn_text,
    ws_connect, ws_next_frame, ws_send,
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
    let held = HeldPort::bind();
    let port = held.port();
    let mut cmd = serve_dir_cmd(BIN, &base, &home.path().join("s").to_string_lossy());
    cmd.args([
        "--mcp-events-callback-url",
        &format!("http://127.0.0.1:{port}"),
    ])
    .env("HOME", home.path())
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    held.hand_to(&mut cmd);
    let _d = cmd.spawn_guarded();
    drop((cmd, held));
    let mut ws = ws_connect(port, Some("mcp-events")).await;
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

/// Fail closed, part one: once a key has been seen for an origin, a later `404` (or a failing
/// fetch) on refresh does not switch enforcement off.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn once_keys_are_seen_a_server_that_stops_publishing_is_still_held_to_them() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_ED25519_SEED", &"07".repeat(32)),
        ("MCP_FIXTURE_MAX_TTL_MS", "1500"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon(home.path(), &base, &[]);
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;

    // The server stops publishing its JWKS; let several refreshes go by.
    control(
        &fixture,
        "POST",
        "/control/jwks",
        Some(&json!({ "status": 404 })),
    );
    let refreshes_now = state(&fixture)["hooks"][0]["refreshes"].as_u64().unwrap();
    eventually(Duration::from_secs(20), "two more refreshes", || {
        (state(&fixture)["hooks"][0]["refreshes"].as_u64().unwrap() >= refreshes_now + 2)
            .then_some(())
    });
    let r = emit(
        &fixture,
        json!({ "event_id": "no-v1a-after-404", "tamper": "no_v1a", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 401, "still enforced: {r:#}");
    control(
        &fixture,
        "POST",
        "/control/jwks",
        Some(&json!({ "status": 503 })),
    );
    let refreshes_now = state(&fixture)["hooks"][0]["refreshes"].as_u64().unwrap();
    eventually(
        Duration::from_secs(20),
        "a refresh with a failing fetch",
        || {
            (state(&fixture)["hooks"][0]["refreshes"].as_u64().unwrap() > refreshes_now)
                .then_some(())
        },
    );
    let r = emit(
        &fixture,
        json!({ "event_id": "no-v1a-after-503", "tamper": "no_v1a", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 401, "still enforced: {r:#}");
    let r = emit(
        &fixture,
        json!({ "event_id": "signed-still-fine", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
}

/// Fail closed, part two: with no key known for the origin, a first subscribe whose JWKS fetch
/// fails (not a clear `404`) is refused rather than started without identity checks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failing_jwks_fetch_refuses_the_first_subscribe_instead_of_enforcing_nothing() {
    let (_fx, mcp_url, _fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_ED25519_SEED", &"07".repeat(32)),
        ("MCP_FIXTURE_JWKS_STATUS", "500"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "hooks", "transport": "http", "url": mcp_url,
            "headers": { "Authorization": "Bearer principal-1" },
        }]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon(home.path(), &base, &[]);
    let mut ws = ws_connect(port, Some("identity-refused")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated", "delivery": "webhook" }),
    )
    .await;
    let r = ws_next(
        &mut ws,
        Duration::from_secs(30),
        "the subscribe response",
        |f| f["id"] == "s",
    )
    .await;
    assert_eq!(r["success"], false, "{r:#}");
    assert!(
        r["error"]
            .as_str()
            .unwrap()
            .contains("webhook-signing keys"),
        "{r:#}"
    );
}

/// A JWKS endpoint that sends its headers and then stalls must not hold the subscribe (or leak the
/// task doing it): the whole fetch is under one deadline, the first subscribe is refused once its
/// retries are spent, and nothing is left waiting on the server afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_jwks_body_that_stalls_is_timed_out_and_leaves_nothing_behind() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_ED25519_SEED", &"07".repeat(32)),
    ]);
    control(
        &fixture,
        "POST",
        "/control/jwks",
        Some(&json!({ "stall": true })),
    );
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "hooks", "transport": "http", "url": mcp_url,
            "headers": { "Authorization": "Bearer principal-1" },
        }]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = common::mcp_events_fixture::spawn_daemon_env(
        home.path(),
        &base,
        &[],
        &[("BEYOND_AI_AGENT_MCP_EVENTS_JWKS_TIMEOUT_MS", "300")],
    );
    let mut ws = ws_connect(port, Some("stalled-jwks")).await;
    let started = Instant::now();
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated", "delivery": "webhook" }),
    )
    .await;
    let r = ws_next(
        &mut ws,
        Duration::from_secs(30),
        "the subscribe response",
        |f| f["id"] == "s",
    )
    .await;
    assert_eq!(r["success"], false, "{r:#}");
    assert!(
        r["error"]
            .as_str()
            .unwrap()
            .contains("webhook-signing keys"),
        "refused for want of keys, not by the 20 s ready timeout: {r:#}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    eventually(
        Duration::from_secs(5),
        "every stalled JWKS fetch to have been dropped",
        || (state(&fixture)["jwks_stalled"] == 0).then_some(()),
    );
}

fn hooks(mcp_url: &str) -> Value {
    json!([{
        "name": "hooks", "transport": "http", "url": mcp_url,
        "headers": { "Authorization": "Bearer principal-1" },
        "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": "notify" }],
    }])
}
