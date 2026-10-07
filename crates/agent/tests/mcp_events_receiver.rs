//! MCP Events (draft) webhook receiving under pressure and failure, end to end: the pending queue
//! is bounded in bytes (a `503` past it), a delivery that cannot be made durable is not
//! acknowledged (`503`, and it lands on the server's retry), a body that never finishes does not
//! hold the connection, and the callback survives a crash so deliveries the server retries while
//! the daemon was down still land. Real `serve --listen`, a real fixture, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::mcp_events_fixture::{
    EVENTS_SESSION, control, emit, eventually, raw_request, runs_for_event, spawn_http_fixture,
    state, write_settings, ws_next, ws_wait_active,
};
use common::{
    BIN, ChildGuard, SpawnGuarded, free_port, serve_dir_cmd, spawn_model_server_routed, turn_text,
    wait_for_port, ws_connect, ws_send,
};
use serde_json::{Value, json};

fn hooks(mcp_url: &str, action: &str) -> Value {
    json!([{
        "name": "hooks", "transport": "http", "url": mcp_url,
        "headers": { "Authorization": "Bearer principal-1" },
        "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": action }],
    }])
}

fn daemon(home: &std::path::Path, base: &str, port: u16, env: &[(&str, &str)]) -> ChildGuard {
    let mut cmd = serve_dir_cmd(BIN, base, &home.join("sessions").to_string_lossy());
    cmd.args([
        "--listen",
        &format!("127.0.0.1:{port}"),
        "--mcp-events-callback-url",
        &format!("http://127.0.0.1:{port}"),
    ])
    .env("HOME", home)
    .env("BEYOND_AI_AGENT_MCP_EVENTS_COALESCE_MS", "300")
    .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
    .envs(env.iter().copied())
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::from(
        std::fs::File::create(home.join(format!("serve-{port}.stderr"))).unwrap(),
    ));
    let child = cmd.spawn_guarded();
    wait_for_port(port);
    child
}

/// The events session's state snapshot, once it exists.
fn state_file(home: &std::path::Path) -> std::path::PathBuf {
    eventually(Duration::from_secs(20), "the events state file", || {
        std::fs::read_dir(home.join("sessions"))
            .ok()?
            .map(|e| e.unwrap().path())
            .find(|p| p.to_string_lossy().ends_with("mcp-events.mcp-events.json"))
    })
}

/// Keep the events session busy, so follow-ups stay pending.
async fn busy(ws: &mut common::TestWs, ms: u64) {
    ws_send(
        ws,
        json!({ "type": "prompt", "id": "long", "message": beyond_ai_test_support::stall_prompt(ms) }),
    )
    .await;
    ws_next(ws, Duration::from_secs(10), "the long run's ack", |f| {
        f["type"] == "ack" && f["id"] == "long"
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_pending_queue_is_bounded_in_bytes_and_answers_503_past_it() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let _d = daemon(
        home.path(),
        &base,
        port,
        &[("BEYOND_AI_AGENT_MCP_EVENTS_MAX_PENDING_BYTES", "8000")],
    );
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    busy(&mut ws, 10_000).await;
    let statuses: Vec<Value> = (0..3)
        .map(|n| {
            let r = emit(
                &fixture,
                json!({ "event_id": format!("big-{n}"), "data": { "blob": "x".repeat(3000) } }),
            );
            r["deliveries"][0]["status"].clone()
        })
        .collect();
    assert_eq!(
        statuses,
        [json!(200), json!(200), json!(503)],
        "two ~3 KB events fit 8000 bytes; the third is refused (retryable), not stored"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delivery_that_cannot_be_stored_gets_503_and_lands_on_retry() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let _d = daemon(home.path(), &base, port, &[]);
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    // The pending log cannot be written: its place is taken by a directory.
    let log = state_file(home.path()).with_extension("log");
    std::fs::create_dir(&log).unwrap();
    let r = emit(
        &fixture,
        json!({ "event_id": "unstorable-1", "data": { "summary": "stored on retry" } }),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 503,
        "not durable, not acknowledged: {r:#}"
    );
    std::fs::remove_dir(&log).unwrap();
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "unstorable-1" })),
    );
    assert_eq!(r["deliveries"][0]["status"], 200, "the retry lands: {r:#}");
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "stored on retry") >= 1).then_some(())
    });
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(runs_for_event(&bodies, "stored on retry"), 1, "once");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delivery_body_that_never_finishes_is_timed_out() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "notify"));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let _d = daemon(
        home.path(),
        &base,
        port,
        &[("BEYOND_AI_AGENT_MCP_EVENTS_BODY_TIMEOUT_MS", "500")],
    );
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    let url = state(&fixture)["hooks"][0]["url"]
        .as_str()
        .unwrap()
        .to_owned();
    let path = url.split_once(&format!("127.0.0.1:{port}")).unwrap().1;
    let started = Instant::now();
    let status = raw_request(
        port,
        &format!("POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n\r\n"),
        b"{\"partial\":",
        Duration::from_secs(10),
    );
    assert_eq!(status, 408, "a stalled body is cut off");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

/// A hard crash leaves the server's subscription in place, retrying deliveries to the callback.
/// The callback (token and secret) is kept across restarts, so once the daemon is back a retried
/// delivery finds its route and verifies — instead of `410`, which tells the server to give up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_callback_survives_a_crash_so_retried_deliveries_still_land() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let mut first = daemon(home.path(), &base, port, &[]);
    let url = eventually(Duration::from_secs(20), "the subscription", || {
        state(&fixture)["hooks"][0]["url"]
            .as_str()
            .map(str::to_owned)
    });
    // The callback is on disk before anything depends on it.
    eventually(Duration::from_secs(10), "the persisted callback", || {
        let saved: Value =
            serde_json::from_slice(&std::fs::read(state_file(home.path())).ok()?).ok()?;
        saved["subscriptions"]
            .as_object()?
            .values()
            .any(|s| s["webhook"]["token"].is_string())
            .then_some(())
    });
    first.kill().unwrap();
    let _ = first.wait();
    // While it is down the server's delivery fails.
    let r = emit(
        &fixture,
        json!({ "event_id": "crash-1", "data": { "summary": "retried after the crash" } }),
    );
    assert_ne!(r["deliveries"][0]["status"], 200, "{r:#}");

    let _second = daemon(home.path(), &base, port, &[]);
    eventually(Duration::from_secs(20), "the resubscribe", || {
        let hooks = state(&fixture)["hooks"].as_array().unwrap().clone();
        (hooks.len() == 1 && hooks[0]["refreshes"].as_u64().unwrap_or(0) >= 1
            || state(&fixture)["methods"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|m| *m == "events/subscribe")
                .count()
                >= 2)
            .then_some(())
    });
    assert_eq!(
        state(&fixture)["hooks"][0]["url"].as_str().unwrap(),
        url,
        "the same callback after the restart"
    );
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "crash-1" })),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 200,
        "the retried delivery lands: {r:#}"
    );
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "retried after the crash") >= 1).then_some(())
    });
}

/// After a restart, a delivery the server retries can arrive before the subscription has
/// re-registered its route (here discovery is slow). It must get `503` — retry — not `410`, which
/// tells the server to give up; once the route is back, the retry lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retry_that_beats_the_resubscribe_after_a_restart_is_told_to_retry_not_to_stop() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_LIST_DELAY_MS", "3000"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let mut first = daemon(home.path(), &base, port, &[]);
    eventually(Duration::from_secs(30), "the subscription", || {
        (state(&fixture)["hooks"].as_array().unwrap().len() == 1).then_some(())
    });
    eventually(Duration::from_secs(10), "the persisted callback", || {
        let saved: Value =
            serde_json::from_slice(&std::fs::read(state_file(home.path())).ok()?).ok()?;
        saved["subscriptions"]
            .as_object()?
            .values()
            .any(|s| s["webhook"]["token"].is_string())
            .then_some(())
    });
    first.kill().unwrap();
    let _ = first.wait();
    let r = emit(
        &fixture,
        json!({ "event_id": "early-1", "data": { "summary": "retried early" } }),
    );
    assert_ne!(r["deliveries"][0]["status"], 200, "{r:#}");

    let _second = daemon(home.path(), &base, port, &[]);
    // The state is read at once; discovery takes 3 s, so the route is not back yet.
    tokio::time::sleep(Duration::from_millis(800)).await;
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "early-1" })),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 503,
        "resuming: retry, not stop: {r:#}"
    );
    let landed = eventually(Duration::from_secs(30), "the retry to land", || {
        let r = control(
            &fixture,
            "POST",
            "/control/redeliver",
            Some(&json!({ "event_id": "early-1" })),
        );
        (r["deliveries"][0]["status"] == 200).then_some(r)
    });
    assert_eq!(landed["deliveries"][0]["status"], 200);
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "retried early") >= 1).then_some(())
    });
}

/// Only callbacks that will come back are held after a restart: a configured subscription is
/// subscribed again, a runtime one is not — so a retried delivery to the runtime one's old callback
/// gets `410` (stop) at once, rather than `503` (retry) for minutes on end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn after_a_restart_a_runtime_subscriptions_old_callback_is_gone_not_held() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "notify"));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let mut first = daemon(home.path(), &base, port, &[]);
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated",
                "arguments": { "project": "alpha" }, "delivery": "webhook", "action": "notify" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the subscribe", |f| {
        f["type"] == "response" && f["id"] == "s"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    let runtime_url = eventually(Duration::from_secs(10), "the runtime hook", || {
        state(&fixture)["hooks"]
            .as_array()?
            .iter()
            .find(|h| h["project"] == "alpha")
            .and_then(|h| h["url"].as_str().map(str::to_owned))
    });
    eventually(Duration::from_secs(10), "both callbacks persisted", || {
        let saved: Value =
            serde_json::from_slice(&std::fs::read(state_file(home.path())).ok()?).ok()?;
        (saved["subscriptions"]
            .as_object()?
            .values()
            .filter(|s| s["webhook"]["token"].is_string())
            .count()
            == 2)
            .then_some(())
    });
    drop(ws);
    first.kill().unwrap();
    let _ = first.wait();
    let _second = daemon(home.path(), &base, port, &[]);
    // The configured one comes back on its old callback.
    eventually(
        Duration::from_secs(20),
        "the configured resubscribe",
        || {
            let n = state(&fixture)["methods"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|m| *m == "events/subscribe")
                .count();
            (n >= 3).then_some(())
        },
    );
    let path = runtime_url
        .split_once(&format!("127.0.0.1:{port}"))
        .unwrap()
        .1
        .to_owned();
    let status = raw_request(
        port,
        &format!("POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n"),
        b"{}",
        Duration::from_secs(5),
    );
    assert_eq!(
        status, 410,
        "nothing will resubscribe a runtime subscription: its callback is gone"
    );
}
