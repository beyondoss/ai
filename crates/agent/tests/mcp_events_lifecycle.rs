//! MCP Events (draft) across a session's life: background triggers survive a disconnected client
//! (the idle reaper leaves a session with live subscriptions alone, unless
//! `--mcp-events-reapable`), and cursors + the dedup window persist with the session, so a
//! restarted `serve` resumes from where it was — events emitted while it was down are delivered,
//! and none twice. Real `serve` daemons, a real streamable-HTTP fixture that outlives them, a mock
//! model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Command;
use std::time::{Duration, Instant};

use common::mcp_events_fixture::{
    Frames, emit, eventually, model_requests_with, send, spawn_daemon, spawn_http_fixture, state,
    wait_active, write_settings,
};
use common::{
    BIN, ChildGuard, SpawnGuarded, TestWs, serve_cmd, spawn_model_server_routed, turn_text,
    ws_connect, ws_next_frame, ws_send,
};
use serde_json::{Value, json};

type Bodies = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

fn hooks_server(mcp_url: &str, action: &str) -> Value {
    json!([{
        "name": "hooks",
        "transport": "http",
        "url": mcp_url,
        "headers": { "Authorization": "Bearer principal-1" },
        "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": action }],
    }])
}

async fn next(
    ws: &mut TestWs,
    timeout: Duration,
    what: &str,
    pred: impl Fn(&Value) -> bool,
) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, ws_next_frame(ws)).await {
            Ok(Some(f)) if pred(&f) => return f,
            Ok(Some(_)) => {}
            Ok(None) => panic!("socket closed waiting for {what}"),
            Err(_) => panic!("timed out waiting for {what}"),
        }
    }
}

async fn collect(ws: &mut TestWs, window: Duration, pred: impl Fn(&Value) -> bool) -> Vec<Value> {
    let deadline = Instant::now() + window;
    let mut out = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return out;
        }
        if let Ok(Some(f)) = tokio::time::timeout(left, ws_next_frame(ws)).await
            && pred(&f)
        {
            out.push(f);
        }
    }
}

async fn list(ws: &mut TestWs, id: &str) -> Value {
    ws_send(ws, json!({ "type": "mcp_events_list", "id": id })).await;
    next(ws, Duration::from_secs(20), id, |f| {
        f["type"] == "response" && f["id"] == id
    })
    .await
}

async fn ws_wait_active(ws: &mut TestWs) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    for n in 0.. {
        let l = list(ws, &format!("l{n}")).await;
        let subs = l["data"]["subscriptions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !subs.is_empty() && subs.iter().all(|s| s["state"] == "active") {
            return subs[0].clone();
        }
        assert!(Instant::now() < deadline, "never active: {l:#}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    unreachable!()
}

fn sigterm_and_wait(child: &mut ChildGuard) {
    Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "serve did not exit after SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn model_saw(bodies: &Bodies, needle: &str) -> bool {
    !model_requests_with(bodies, needle).is_empty()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_event_arriving_while_detached_is_delivered_and_seen_on_reattach() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks_server(&mcp_url, "follow_up"));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("handled while away"));
    // A 1 s idle timeout: without the keep-alive this session would be reaped (and unsubscribed)
    // long before the event below arrives.
    let (_d, port) = spawn_daemon(home.path(), &base, &["--session-idle-timeout", "1"]);

    let mut ws = ws_connect(port, Some("mcp-events")).await;
    ws_wait_active(&mut ws).await;
    drop(ws);
    tokio::time::sleep(Duration::from_millis(2500)).await;

    // Still subscribed: nothing unsubscribed, and the delivery lands (a reaped session's route
    // would answer 410).
    assert!(
        state(&fixture)["unsubscribes"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let r = emit(
        &fixture,
        json!({ "data": { "ticket_id": "D-1", "summary": "arrived while detached" } }),
    );
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
    eventually(
        Duration::from_secs(20),
        "the model run the event triggered",
        || model_saw(&bodies, "arrived while detached").then_some(()),
    );
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Re-attach: the catch-up carries the event and the model's answer to it.
    let mut ws = ws_connect(port, Some("mcp-events")).await;
    let catchup = next(&mut ws, Duration::from_secs(20), "the catchup frame", |f| {
        f["type"] == "catchup"
    })
    .await;
    let messages = catchup["data"]["messages"].to_string();
    assert!(messages.contains("arrived while detached"), "{catchup:#}");
    assert!(messages.contains("handled while away"), "{catchup:#}");
    let l = list(&mut ws, "after").await;
    assert_eq!(l["data"]["subscriptions"][0]["delivered"], 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mcp_events_reapable_lets_the_reaper_end_a_detached_sessions_subscriptions() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks_server(&mcp_url, "notify"));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon(
        home.path(),
        &base,
        &["--session-idle-timeout", "1", "--mcp-events-reapable"],
    );
    let mut ws = ws_connect(port, Some("mcp-events")).await;
    ws_wait_active(&mut ws).await;
    drop(ws);
    // Reaped → the session's teardown unsubscribes.
    eventually(Duration::from_secs(20), "the reaper's unsubscribe", || {
        (!state(&fixture)["unsubscribes"]
            .as_array()
            .unwrap()
            .is_empty())
        .then_some(())
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_daemon_resumes_webhook_delivery_from_its_persisted_cursor_without_duplicates()
{
    // Inclusive replay: on resubscribe the fixture also resends the event *at* the cursor, which
    // only the persisted dedup window can drop.
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_REPLAY_INCLUSIVE", "1"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks_server(&mcp_url, "notify"));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));

    let (mut first, port) = spawn_daemon(home.path(), &base, &[]);
    let mut ws = ws_connect(port, Some("mcp-events")).await;
    ws_wait_active(&mut ws).await;
    emit(&fixture, json!({ "event_id": "r-1", "data": { "n": 1 } }));
    next(&mut ws, Duration::from_secs(20), "r-1", |f| {
        f["event"]["eventId"] == "r-1"
    })
    .await;
    drop(ws);
    sigterm_and_wait(&mut first);

    // The position is on disk, beside the transcript.
    let state_file = std::fs::read_dir(home.path().join("sessions"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.to_string_lossy().ends_with("mcp-events.mcp-events.json"))
        .expect("the events session state file beside its's transcript");
    let saved: Value = serde_json::from_slice(&std::fs::read(state_file).unwrap()).unwrap();
    let entry = saved["subscriptions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .clone();
    assert_eq!(entry["cursor"], "1", "{saved:#}");
    assert_eq!(entry["recent"], json!(["r-1"]));

    // Emitted while nothing is listening.
    emit(&fixture, json!({ "event_id": "r-2", "data": { "n": 2 } }));
    emit(&fixture, json!({ "event_id": "r-3", "data": { "n": 3 } }));

    let (_second, port) = spawn_daemon(home.path(), &base, &[]);
    let mut ws = ws_connect(port, Some("mcp-events")).await;
    let got = collect(&mut ws, Duration::from_secs(6), |f| {
        f["type"] == "mcp_event"
    })
    .await;
    let ids: Vec<&str> = got
        .iter()
        .map(|f| f["event"]["eventId"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        ["r-2", "r-3"],
        "the gap is delivered, once, and r-1 is not repeated"
    );
    let l = list(&mut ws, "after").await;
    assert_eq!(
        l["data"]["subscriptions"][0]["duplicates"], 1,
        "the replayed r-1 was dropped: {l:#}"
    );
    let replays = state(&fixture)["deliveries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| d["replay"] == true)
        .count();
    assert_eq!(replays, 3, "the server really did resend r-1, r-2 and r-3");
}

#[test]
fn a_restarted_stdio_serve_resumes_polling_from_its_persisted_cursor() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_REPLAY_INCLUSIVE", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "tickets", "transport": "http", "url": mcp_url,
            "events": [{ "name": "ticket.updated", "delivery": "poll", "action": "notify" }],
        }]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let session = home.path().join("s.jsonl");
    let start = || {
        let mut cmd = serve_cmd(BIN, &base, &session.to_string_lossy());
        cmd.env("HOME", home.path())
            .env("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")
            .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
        let mut child = cmd.spawn_guarded();
        let stdin = child.stdin.take().unwrap();
        let frames = Frames::new(&mut child, None);
        (child, stdin, frames)
    };

    let (mut child, mut stdin, mut frames) = start();
    wait_active(&mut stdin, &mut frames, 1);
    emit(&fixture, json!({ "event_id": "p-1", "data": { "n": 1 } }));
    frames.wait(Duration::from_secs(20), "p-1", |f| {
        f["event"]["eventId"] == "p-1"
    });
    std::thread::sleep(Duration::from_millis(400)); // let the coalesced state write land
    drop(stdin); // EOF: a clean shutdown
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "serve did not exit on EOF");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(home.path().join("s.mcp-events.json").exists());

    emit(&fixture, json!({ "event_id": "p-2", "data": { "n": 2 } }));
    emit(&fixture, json!({ "event_id": "p-3", "data": { "n": 3 } }));

    let (_child, mut stdin, mut frames) = start();
    wait_active(&mut stdin, &mut frames, 1);
    frames.collect(Duration::from_secs(3), |_| false);
    // `seen` holds every frame read so far, including those that arrived while waiting above.
    let ids: Vec<&str> = frames
        .seen
        .iter()
        .filter(|f| f["type"] == "mcp_event")
        .map(|f| f["event"]["eventId"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        ["p-2", "p-3"],
        "resumed from the cursor, nothing repeated"
    );
    send(&mut stdin, json!({ "type": "mcp_events_list", "id": "l" }));
    let l = frames.response("l");
    assert!(
        l["data"]["subscriptions"][0]["duplicates"]
            .as_u64()
            .unwrap()
            >= 1,
        "{l:#}"
    );
}
