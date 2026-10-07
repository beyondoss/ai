//! A server may raise a nested request (here `elicitation/create`) while one of its `events/*`
//! requests is in flight — a unary `events/poll`, or an open `events/stream`. It goes to the session
//! that owns that request, exactly as one raised during a `tools/call` does (the `track_call` rule):
//! that session's client is asked and the server gets the client's answer. Real `serve`, a real
//! stdio fixture, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::Duration;

use common::mcp_events_fixture::{
    Frames, eventually, send, state, stdio_server, wait_control_file, write_settings,
};
use common::{BIN, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::json;

fn nested_during(during: &str, delivery: &str) {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    write_settings(
        home.path(),
        json!([stdio_server(
            "tickets",
            &control_file,
            json!({ "MCP_FIXTURE_NESTED_DURING": during, "MCP_FIXTURE_HEARTBEAT_MS": "200" }),
            json!([{ "name": "ticket.updated", "delivery": delivery, "action": "notify" }]),
        )]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let mut cmd = serve_cmd(BIN, &base, &home.path().join("s.jsonl").to_string_lossy());
    cmd.env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .env("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")
        .stderr(Stdio::null());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(child.stdout.take().unwrap(), None);
    let control = wait_control_file(&control_file);
    let ask = frames.wait(
        Duration::from_secs(30),
        "the nested elicitation, asked of this session's client",
        |f| f["type"] == "elicitation_request",
    );
    assert_eq!(ask["server"], "tickets", "{ask:#}");
    assert!(
        ask.to_string().contains(&format!("events/{during}")),
        "{ask:#}"
    );
    send(
        &mut stdin,
        json!({
            "type": "elicit",
            "request_id": ask["request_id"],
            "action": "accept",
            "content": { "ok": true },
        }),
    );
    let answers = eventually(Duration::from_secs(20), "the server's answer", || {
        let st = state(&control);
        let a = st["nested_answers"].as_array()?.clone();
        (!a.is_empty()).then_some(a)
    });
    assert_eq!(answers[0]["during"], during);
    assert_eq!(
        answers[0]["answer"]["result"]["action"], "accept",
        "the owning session's client answered: {answers:#?}"
    );
    assert_eq!(answers[0]["answer"]["result"]["content"]["ok"], true);
}

#[test]
fn a_nested_elicitation_during_an_events_poll_reaches_the_owning_session() {
    nested_during("poll", "poll");
}

#[test]
fn a_nested_elicitation_during_an_events_stream_reaches_the_owning_session() {
    nested_during("stream", "push");
}

/// In a daemon the server connection is shared by every session, so "this session" is not a given:
/// the nested request raised during the events session's `events/poll` must reach a client attached
/// to *that* session — not the process-wide host, and not another session's client.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_a_daemon_a_nested_elicitation_during_an_events_poll_reaches_the_events_session() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    write_settings(
        home.path(),
        json!([stdio_server(
            "tickets",
            &control_file,
            json!({ "MCP_FIXTURE_NESTED_DURING": "poll", "MCP_FIXTURE_NESTED_DELAY_MS": "1500" }),
            json!([{ "name": "ticket.updated", "delivery": "poll", "action": "notify" }]),
        )]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = common::free_port();
    let _d = common::mcp_events_fixture::spawn_daemon_env(
        home.path(),
        &base,
        port,
        &[],
        &[("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")],
    );
    let control = wait_control_file(&control_file);
    // Another session's client is attached too; it must not be the one asked.
    let mut other = common::ws_connect(port, Some("bystander")).await;
    let mut ws = common::ws_connect(port, Some(common::mcp_events_fixture::EVENTS_SESSION)).await;
    let ask = common::mcp_events_fixture::ws_next(
        &mut ws,
        Duration::from_secs(30),
        "the nested elicitation on the events session",
        |f| f["type"] == "elicitation_request",
    )
    .await;
    common::ws_send(
        &mut ws,
        json!({ "type": "elicit", "request_id": ask["request_id"], "action": "accept", "content": { "ok": true } }),
    )
    .await;
    let answers = eventually(Duration::from_secs(20), "the server's answer", || {
        let a = state(&control)["nested_answers"].as_array()?.clone();
        (!a.is_empty()).then_some(a)
    });
    assert_eq!(
        answers[0]["answer"]["result"]["action"], "accept",
        "{answers:#?}"
    );
    // The bystander saw no elicitation.
    let stray = tokio::time::timeout(Duration::from_millis(500), async {
        while let Some(f) = common::ws_next_frame(&mut other).await {
            if f["type"] == "elicitation_request" {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(!stray, "another session's client was asked");
}

/// Over direct HTTP (a stateless `2026-07-28` server) nothing routes a server→client request to a
/// session, so it is answered at once with an error — never left unanswered, holding the server.
fn nested_over_direct_http(during: &str, delivery: &str) {
    let (_fx, mcp_url, control) = common::mcp_events_fixture::spawn_http_fixture(&[
        ("MCP_FIXTURE_NESTED_DURING", during),
        ("MCP_FIXTURE_HEARTBEAT_MS", "200"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "tickets", "transport": "http", "url": mcp_url,
            "events": [{ "name": "ticket.updated", "delivery": delivery, "action": "notify" }],
        }]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let mut cmd = serve_cmd(BIN, &base, &home.path().join("s.jsonl").to_string_lossy());
    cmd.env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .env("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")
        .stderr(Stdio::null());
    let _child = cmd.spawn_guarded();
    let answers = eventually(Duration::from_secs(15), "the server's answer", || {
        let a = state(&control)["nested_answers"].as_array()?.clone();
        (!a.is_empty()).then_some(a)
    });
    assert_eq!(answers[0]["during"], during);
    assert!(
        answers[0]["answer"]["error"].is_object(),
        "refused at once, not left to time out: {answers:#?}"
    );
}

#[test]
fn over_direct_http_a_nested_request_during_an_events_poll_is_refused_not_ignored() {
    nested_over_direct_http("poll", "poll");
}

#[test]
fn over_direct_http_a_nested_request_during_an_events_stream_is_refused_not_ignored() {
    nested_over_direct_http("stream", "push");
}
