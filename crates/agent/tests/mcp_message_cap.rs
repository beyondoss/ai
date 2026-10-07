//! Every MCP message is held to the one size cap (`BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES`), on every
//! path: a tool result over it — stdio, or streamable HTTP through `mcp_wire::HttpClient` — fails
//! its call with an error (never read whole) and the connection keeps working; and the MCP Events
//! direct-HTTP wire refuses an over-cap unary JSON answer, an over-cap unary SSE answer and an
//! over-cap `events/stream` event. Real `serve`, a real fixture server, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::mcp_events_fixture::{
    Frames, fast_knobs, send, spawn_http_fixture, stdio_server, write_settings,
};
use common::{BIN, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text, turn_tool_use};
use serde_json::{Value, json};
use std::time::Duration;

const LIMIT: &str = "1048576";

/// A `serve` whose model asks for `mcp__big__blob` with `bytes` when the prompt names a size,
/// and answers "done" once it has the tool result.
fn run_blobs(home: &std::path::Path, server: Value) -> Vec<String> {
    write_settings(home, json!([server]));
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("tool_result".into(), turn_text("done")),
            (
                "blob of 2000000".into(),
                turn_tool_use("t_big", "mcp__big__blob", r#"{"bytes":2000000}"#),
            ),
            (
                "blob of 10".into(),
                turn_tool_use("t_small", "mcp__big__blob", r#"{"bytes":10}"#),
            ),
        ],
        turn_text("done"),
    );
    let mut cmd = serve_cmd(BIN, &base, &home.join("s.jsonl").to_string_lossy());
    cmd.env("HOME", home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .env("BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES", LIMIT)
        .stderr(Stdio::null());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(child.stdout.take().unwrap(), None);
    for (id, msg) in [
        ("p1", "make a blob of 2000000"),
        ("p2", "make a blob of 10"),
    ] {
        send(
            &mut stdin,
            json!({ "type": "prompt", "id": id, "message": msg }),
        );
        let r = frames.response(id);
        assert_eq!(r["success"], true, "{r:#}");
        // A fresh transcript, so the next prompt's request carries nothing of this one.
        send(
            &mut stdin,
            json!({ "type": "new_session", "id": format!("ns-{id}") }),
        );
        frames.response(&format!("ns-{id}"));
    }
    let requests = bodies.lock().unwrap().clone();
    drop(stdin);
    requests
}

fn assert_capped(requests: &[String]) {
    let big = requests
        .iter()
        .find(|r| r.contains("t_big") && r.contains("tool_result"))
        .expect("the model got the big call's result");
    assert!(
        big.contains("refused by the host") && !big.contains(&"x".repeat(100_000)),
        "the over-limit result failed the call instead of arriving"
    );
    let small = requests
        .iter()
        .find(|r| r.contains("t_small") && r.contains("tool_result"))
        .expect("the model got the small call's result");
    assert!(
        small.contains("xxxxxxxxxx"),
        "the connection still works after the discarded message"
    );
}

#[test]
fn an_over_limit_message_from_a_stdio_server_fails_its_call_and_the_connection_survives() {
    let home = tempfile::tempdir().unwrap();
    let server = stdio_server("big", &home.path().join("control"), json!({}), json!([]));
    let requests = run_blobs(home.path(), server);
    assert_capped(&requests);
}

#[test]
fn an_over_limit_message_from_an_http_server_fails_its_call_and_the_connection_survives() {
    let (_fx, mcp_url, _control) = spawn_http_fixture(&[]);
    let home = tempfile::tempdir().unwrap();
    let server = json!({ "name": "big", "transport": "http", "url": mcp_url });
    let requests = run_blobs(home.path(), server);
    assert_capped(&requests);
}

/// A `serve` (stdio) on one streamable-HTTP fixture server called `tickets`, with a small cap.
fn events_serve(
    fixture_env: &[(&str, &str)],
    events: Value,
) -> (
    common::ChildGuard,
    std::process::ChildStdin,
    Frames,
    String,
    tempfile::TempDir,
    common::ChildGuard,
) {
    let (fx, mcp_url, control) = spawn_http_fixture(fixture_env);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{ "name": "tickets", "transport": "http", "url": mcp_url, "events": events }]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let mut cmd = serve_cmd(BIN, &base, &home.path().join("s.jsonl").to_string_lossy());
    fast_knobs(&mut cmd)
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES", "65536")
        .stderr(Stdio::null());
    let mut child = cmd.spawn_guarded();
    let stdin = child.stdin.take().unwrap();
    let frames = Frames::new(child.stdout.take().unwrap(), None);
    (child, stdin, frames, control, home, fx)
}

#[test]
fn an_over_cap_unary_json_answer_on_the_events_wire_is_refused() {
    let (_c, mut stdin, mut frames, _control, _home, _fx) =
        events_serve(&[("MCP_FIXTURE_LIST_PAD_BYTES", "200000")], json!([]));
    send(&mut stdin, json!({ "type": "mcp_events_list", "id": "l" }));
    let l = frames.response("l");
    let available = &l["data"]["available"][0];
    assert_eq!(available["supported"], false, "{l:#}");
    assert!(
        available["error"]
            .as_str()
            .unwrap()
            .contains("larger than 65536"),
        "{l:#}"
    );
}

#[test]
fn an_over_cap_unary_sse_answer_on_the_events_wire_is_refused() {
    let (_c, mut stdin, mut frames, _control, _home, _fx) =
        events_serve(&[("MCP_FIXTURE_POLL_SSE_PAD_BYTES", "200000")], json!([]));
    send(
        &mut stdin,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "tickets",
                "name": "ticket.updated", "delivery": "poll", "action": "notify" }),
    );
    let r = frames.response("s");
    assert_eq!(
        r["success"], false,
        "the over-cap poll answer is not read: {r:#}"
    );
}

/// An `events/stream` event over the cap ends the stream with an error rather than being read
/// whole; the subscription retries (with backoff — a server that keeps resending the same giant event
/// keeps failing, it does not take the host's memory), and the session stays responsive.
#[test]
fn an_over_cap_event_on_an_events_stream_is_refused_not_read_whole() {
    let (_c, mut stdin, mut frames, control, _home, _fx) = events_serve(
        &[("MCP_FIXTURE_HEARTBEAT_MS", "200")],
        json!([{ "name": "ticket.updated", "delivery": "push", "action": "notify" }]),
    );
    common::mcp_events_fixture::wait_active(&mut stdin, &mut frames, 1);
    common::mcp_events_fixture::emit(
        &control,
        json!({ "event_id": "huge-1", "data": { "blob": "x".repeat(200_000) } }),
    );
    let status = frames.wait(Duration::from_secs(20), "the stream's refusal", |f| {
        f["type"] == "mcp_event_status"
            && f["kind"] == "error"
            && f.to_string().contains("exceeded")
    });
    assert!(status.to_string().contains("65536"), "{status:#}");
    send(&mut stdin, json!({ "type": "mcp_events_list", "id": "l" }));
    let l = frames.response("l");
    assert_eq!(l["success"], true, "the session stays responsive: {l:#}");
    assert!(
        !frames
            .seen
            .iter()
            .any(|f| f["type"] == "mcp_event" && f["event"]["eventId"] == "huge-1"),
        "the over-cap event was never delivered"
    );
}
