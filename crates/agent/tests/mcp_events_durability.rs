//! MCP Events (draft) delivery is durable end to end — to the *model*, not just to the frame:
//!
//! - an event a server was told we received, but the model had not yet seen (held because a run
//!   was in flight), survives a SIGTERM and reaches the model exactly once after the restart;
//! - a poll page the pending queue cannot take does not advance the cursor, so nothing is lost
//!   while the model catches up (and nothing is delivered twice);
//! - the events state follows the session to a new transcript (`new_session`).
//!
//! Real `serve` processes, a real fixture that outlives them, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_events_fixture::{
    EVENTS_SESSION, Frames, emit, eventually, pending_on_disk, runs_for_event, send,
    sigterm_and_wait, spawn_daemon, spawn_http_fixture, state, wait_active, write_settings,
    ws_next, ws_wait_active,
};
use common::{
    BIN, SpawnGuarded, serve_dir_cmd, spawn_model_server_routed, turn_text, ws_connect, ws_send,
};
use serde_json::{Value, json};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_follow_up_held_during_a_run_survives_sigterm_and_reaches_the_model_once_after_restart() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "hooks", "transport": "http", "url": mcp_url,
            "headers": { "Authorization": "Bearer principal-1" },
            "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": "follow_up" }],
        }]),
    );
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("handled"));

    let (mut first, port) = spawn_daemon(home.path(), &base, &[]);
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    // A long run is in flight, so the follow-up is held — not yet seen by the model.
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "long", "message": beyond_ai_test_support::stall_prompt(5000) }),
    )
    .await;
    ws_next(
        &mut ws,
        Duration::from_secs(10),
        "the long run's ack",
        |f| f["type"] == "ack" && f["id"] == "long",
    )
    .await;
    let r = emit(
        &fixture,
        json!({ "event_id": "durable-1", "data": { "summary": "survives sigterm" } }),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 200,
        "acked only once stored: {r:#}"
    );
    ws_next(&mut ws, Duration::from_secs(10), "the frame", |f| {
        f["type"] == "mcp_event"
    })
    .await;
    drop(ws);
    sigterm_and_wait(&mut first);
    assert_eq!(
        runs_for_event(&bodies, "survives sigterm"),
        0,
        "the model had not seen it yet"
    );

    // On disk, pending.
    let state_file = std::fs::read_dir(home.path().join("sessions"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.to_string_lossy().ends_with("mcp-events.mcp-events.json"))
        .expect("the events session's state file");
    let saved = pending_on_disk(&state_file);
    assert_eq!(saved[0]["event"]["eventId"], "durable-1", "{saved:#?}");

    // Restart: the pending event is injected, once.
    let (_second, _) = spawn_daemon(home.path(), &base, &[]);
    eventually(
        Duration::from_secs(30),
        "the model to receive the held event",
        || (runs_for_event(&bodies, "survives sigterm") >= 1).then_some(()),
    );
    // The injected run carries the stalled history, so it takes a while; once it is over the event
    // is delivered and leaves the pending queue on disk.
    eventually(
        Duration::from_secs(30),
        "the pending queue to drain",
        || pending_on_disk(&state_file).is_empty().then_some(()),
    );
    tokio::time::sleep(Duration::from_millis(1000)).await;
    assert_eq!(
        runs_for_event(&bodies, "survives sigterm"),
        1,
        "exactly once"
    );
}

/// With room for one undelivered event and every model run stalled, the first event fills the
/// queue; the next two cannot be accepted, so the poll cursor must stay put until the model has
/// seen the first. Persisting the cursor before delivering (or advancing past a refused event)
/// loses them.
#[test]
fn poll_does_not_advance_its_cursor_past_events_it_could_not_accept() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "tickets", "transport": "http", "url": mcp_url,
            "events": [{
                "name": "ticket.updated", "delivery": "poll", "action": "follow_up",
                // Every injected run stalls, so the one pending slot stays taken a while.
                "instructions": beyond_ai_test_support::stall_prompt(1200),
            }],
        }]),
    );
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let sessions = home.path().join("sessions");
    let mut cmd = serve_dir_cmd(BIN, &base, &sessions.to_string_lossy());
    cmd.env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")
        .env("BEYOND_AI_AGENT_MCP_EVENTS_COALESCE_MS", "100")
        .env("BEYOND_AI_AGENT_MCP_EVENTS_MAX_PENDING", "1")
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(&mut child, None);
    wait_active(&mut stdin, &mut frames, 1);

    for n in 1..=3 {
        emit(
            &fixture,
            json!({ "event_id": format!("bp-{n}"), "data": { "summary": format!("backpressure {n}") } }),
        );
    }
    for n in 1..=3 {
        let needle = format!("backpressure {n}");
        eventually(Duration::from_secs(30), &needle, || {
            (runs_for_event(&bodies, &needle) >= 1).then_some(())
        });
    }
    std::thread::sleep(Duration::from_millis(1500));
    for n in 1..=3 {
        assert_eq!(
            runs_for_event(&bodies, &format!("backpressure {n}")),
            1,
            "event {n}: delivered to the model exactly once"
        );
    }
    send(&mut stdin, json!({ "type": "mcp_events_list", "id": "l" }));
    let l = frames.response("l");
    assert_eq!(l["data"]["pending"], 0, "{l:#}");
    // The refusal really happened: some poll re-read an event it had already been given.
    assert!(
        l["data"]["subscriptions"][0]["duplicates"]
            .as_u64()
            .unwrap()
            >= 1,
        "the cursor was held and the page re-polled: {l:#}"
    );
    let _ = state(&fixture);
}

#[test]
fn the_events_state_follows_the_session_to_a_new_transcript() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "tickets", "transport": "http", "url": mcp_url,
            "events": [{ "name": "ticket.updated", "delivery": "poll", "action": "notify" }],
        }]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let sessions = home.path().join("sessions");
    let mut cmd = serve_dir_cmd(BIN, &base, &sessions.to_string_lossy());
    cmd.env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(&mut child, None);
    wait_active(&mut stdin, &mut frames, 1);
    send(&mut stdin, json!({ "type": "new_session", "id": "ns" }));
    let ns = frames.response("ns");
    let new_id = ns["data"]["session_id"].as_str().unwrap().to_owned();
    // Any command after the switch lets the session notice it; then an event moves the cursor.
    send(&mut stdin, json!({ "type": "get_state", "id": "g" }));
    frames.response("g");
    // Moved, not copied: by the time the next command is answered the state is in the new
    // transcript's files and gone from the old one's — so a later resume of the old session
    // cannot re-inject (or overwrite) what now belongs to this one.
    let sidecars = || -> Vec<String> {
        std::fs::read_dir(&sessions)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".mcp-events."))
            .collect()
    };
    let now = sidecars();
    assert!(
        !now.is_empty() && now.iter().all(|n| n.contains(&new_id)),
        "the events state moved with the session at once: {now:?}"
    );
    emit(&fixture, json!({ "event_id": "moved-1", "data": {} }));
    frames.wait(Duration::from_secs(20), "the event", |f| {
        f["type"] == "mcp_event"
    });
    eventually(
        Duration::from_secs(10),
        "the new session's state file, with the event",
        || {
            let found = std::fs::read_dir(&sessions)
                .ok()?
                .map(|e| e.unwrap().path())
                .find(|p| {
                    let n = p.file_name().unwrap().to_string_lossy().into_owned();
                    n.contains(&new_id) && n.ends_with(".mcp-events.json")
                })?;
            let saved: Value = serde_json::from_slice(&std::fs::read(found).ok()?).ok()?;
            let sub = saved["subscriptions"].as_object()?.values().next()?.clone();
            (sub["recent"] == json!(["moved-1"])).then_some(())
        },
    );
}

/// A stdio `serve` polling one `ticket.updated` subscription with `action`, its model answering
/// from `routes` (else "noted").
fn polling_serve(
    home: &std::path::Path,
    mcp_url: &str,
    action: &str,
    routes: Vec<(String, String)>,
) -> (
    common::ChildGuard,
    std::process::ChildStdin,
    Frames,
    std::sync::Arc<std::sync::Mutex<Vec<String>>>,
) {
    write_settings(
        home,
        json!([{
            "name": "tickets", "transport": "http", "url": mcp_url,
            "events": [{ "name": "ticket.updated", "delivery": "poll", "action": action }],
        }]),
    );
    let (base, bodies) = spawn_model_server_routed(routes, turn_text("noted"));
    let sessions = home.join("sessions");
    let mut cmd = serve_dir_cmd(BIN, &base, &sessions.to_string_lossy());
    cmd.env("HOME", home)
        .env("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")
        .env("BEYOND_AI_AGENT_MCP_EVENTS_COALESCE_MS", "100");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(&mut child, None);
    wait_active(&mut stdin, &mut frames, 1);
    (child, stdin, frames, bodies)
}

/// A `steer` event accepted onto the steer lane mid-run has not reached the model yet; an abort
/// clears that lane. The event must not count as delivered: it is injected again and reaches the
/// model after the abort.
#[test]
fn a_steered_event_an_abort_drops_is_injected_again() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[]);
    let home = tempfile::tempdir().unwrap();
    let (_child, mut stdin, mut frames, bodies) =
        polling_serve(home.path(), &mcp_url, "steer", vec![]);
    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "long", "message": beyond_ai_test_support::stall_prompt(30_000) }),
    );
    frames.wait(Duration::from_secs(10), "the long run's ack", |f| {
        f["type"] == "ack" && f["id"] == "long"
    });
    emit(
        &fixture,
        json!({ "event_id": "steer-1", "data": { "summary": "steered then aborted" } }),
    );
    frames.wait(
        Duration::from_secs(20),
        "the injection queued on the steer lane",
        |f| {
            f["type"] == "response"
                && f["id"]
                    .as_str()
                    .is_some_and(|i| i.starts_with("mcp_events:"))
                && f["data"]["queued_as"] == "steer"
        },
    );
    assert_eq!(runs_for_event(&bodies, "steered then aborted"), 0);
    send(&mut stdin, json!({ "type": "abort", "id": "a" }));
    frames.response("a");
    eventually(
        Duration::from_secs(20),
        "the dropped steer to reach the model after all",
        || (runs_for_event(&bodies, "steered then aborted") >= 1).then_some(()),
    );
}

/// An injected run that fails has not delivered its events: they are injected again — bounded, so
/// an event that always fails the model is dropped (and reported) rather than re-run forever.
#[test]
fn an_event_whose_run_fails_is_injected_again_up_to_a_bound() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[]);
    let home = tempfile::tempdir().unwrap();
    let failing = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"refused by the test\"}}\n\n";
    let (_child, mut stdin, mut frames, bodies) = polling_serve(
        home.path(),
        &mcp_url,
        "follow_up",
        vec![("always fails the model".into(), failing.into())],
    );
    send(
        &mut stdin,
        json!({ "type": "set_auto_retry", "id": "r", "enabled": false }),
    );
    frames.response("r");
    emit(
        &fixture,
        json!({ "event_id": "poison-1", "data": { "summary": "always fails the model" } }),
    );
    frames.wait(
        Duration::from_secs(60),
        "the event dropped as undeliverable",
        |f| f["type"] == "mcp_event_status" && f["kind"] == "dropped",
    );
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(
        runs_for_event(&bodies, "always fails the model"),
        3,
        "a failed run's event is injected again, up to the bound"
    );
}
