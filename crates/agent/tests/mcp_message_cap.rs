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
    let mut frames = Frames::new(&mut child, None);
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
    let frames = Frames::new(&mut child, None);
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

/// An `events/stream` event over the cap is skipped, not read and not reconnected into: its bounded
/// head names it (and its cursor), it is reported as a gap — to the client and to the model — and the
/// next event on the same stream is delivered. Over streamable HTTP (SSE) and over stdio, and in
/// either key order: JSON does not fix one, so with `params_first` the giant event arrives as
/// `{"params":{"data":…,"_meta":…,"cursor":…},"method":…}` — nothing that names or routes it is
/// within the bounded head, and it must still be skipped and reported, not looped on or lost.
fn an_over_cap_push_event_is_skipped_and_reported(stdio: bool, params_first: bool) {
    let events = json!([{ "name": "ticket.updated", "delivery": "push", "action": "follow_up" }]);
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let order = if params_first { "params_first" } else { "" };
    let http_fixture = (!stdio).then(|| {
        spawn_http_fixture(&[
            ("MCP_FIXTURE_HEARTBEAT_MS", "200"),
            ("MCP_FIXTURE_KEY_ORDER", order),
        ])
    });
    let server = match &http_fixture {
        Some((_fx, mcp_url, _control)) => {
            json!({ "name": "tickets", "transport": "http", "url": mcp_url, "events": events })
        }
        None => stdio_server(
            "tickets",
            &control_file,
            json!({ "MCP_FIXTURE_HEARTBEAT_MS": "200", "MCP_FIXTURE_KEY_ORDER": order }),
            events,
        ),
    };
    write_settings(home.path(), json!([server]));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let mut cmd = serve_cmd(BIN, &base, &home.path().join("s.jsonl").to_string_lossy());
    fast_knobs(&mut cmd)
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES", "65536")
        .stderr(Stdio::null());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(&mut child, None);
    let control = match &http_fixture {
        Some((_, _, control)) => control.clone(),
        None => common::mcp_events_fixture::wait_control_file(&control_file),
    };
    common::mcp_events_fixture::wait_active(&mut stdin, &mut frames, 1);
    let streams = |control: &str| {
        common::mcp_events_fixture::state(control)["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/stream")
            .count()
    };
    let streams_before = streams(&control);
    common::mcp_events_fixture::emit(
        &control,
        json!({ "event_id": "huge-1", "data": { "blob": "x".repeat(200_000) } }),
    );
    common::mcp_events_fixture::emit(
        &control,
        json!({ "event_id": "small-1", "data": { "ok": true } }),
    );
    let gap = frames.wait(
        Duration::from_secs(20),
        "the gap for the dropped event",
        |f| f["type"] == "mcp_event_status" && f["kind"] == "gap",
    );
    assert_eq!(gap["reason"], "oversized", "{gap:#}");
    // In the usual order the fixture serializes `cursor` ahead of the payload (and `eventId` after
    // it): the cursor is kept, the id is not known. Payload first, neither is: the position moves
    // on with the next event or heartbeat instead.
    if !params_first {
        assert!(
            gap["cursor"].is_string(),
            "the skipped event's cursor is kept: {gap:#}"
        );
    }
    // Taken for an event without its method seen, the gap says so rather than asserting it.
    assert_eq!(
        gap["possibly_not_an_event"].as_bool(),
        params_first.then_some(true),
        "{gap:#}"
    );
    let f = frames.wait(Duration::from_secs(20), "the next event", |f| {
        f["type"] == "mcp_event" && f["event"]["eventId"] == "small-1"
    });
    assert_eq!(f["event"]["data"]["ok"], true);
    assert!(
        !frames
            .seen
            .iter()
            .any(|f| f["type"] == "mcp_event" && f["event"]["eventId"] == "huge-1"),
        "the over-cap event was never read"
    );
    common::mcp_events_fixture::eventually(Duration::from_secs(20), "the model told", || {
        bodies
            .lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("larger than the message-size limit"))
            .then_some(())
    });
    assert_eq!(
        bodies
            .lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("possibly not an event")),
        params_first,
        "the model is told when the dropped message may not have been an event"
    );
    assert_eq!(
        streams(&control),
        streams_before,
        "skipped in place: the stream was not reconnected into the same event"
    );
}

#[test]
fn an_over_cap_push_event_over_http_is_skipped_and_reported_and_the_next_is_delivered() {
    an_over_cap_push_event_is_skipped_and_reported(false, false);
}

#[test]
fn an_over_cap_push_event_over_stdio_is_skipped_and_reported_and_the_next_is_delivered() {
    an_over_cap_push_event_is_skipped_and_reported(true, false);
}

#[test]
fn an_over_cap_push_event_with_params_first_over_http_is_skipped_and_reported_not_looped_on() {
    an_over_cap_push_event_is_skipped_and_reported(false, true);
}

#[test]
fn an_over_cap_push_event_with_params_first_over_stdio_is_skipped_and_reported_not_lost() {
    an_over_cap_push_event_is_skipped_and_reported(true, true);
}

/// One over-cap message a stdio server sends, with its routing past the window, reaches every push
/// stream on the connection (which one it was for is unknowable) — two subscriptions here. Each
/// records the gap in its own state, but the model is told once: one dropped message, one notice.
#[test]
fn one_over_cap_message_reaching_two_subscriptions_is_one_notice_to_the_model() {
    let events = json!([
        { "name": "ticket.updated", "arguments": { "project": "alpha" }, "delivery": "push", "action": "follow_up" },
        { "name": "ticket.updated", "arguments": { "project": "beta" }, "delivery": "push", "action": "follow_up" },
    ]);
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    write_settings(
        home.path(),
        json!([stdio_server(
            "tickets",
            &control_file,
            json!({ "MCP_FIXTURE_HEARTBEAT_MS": "200", "MCP_FIXTURE_KEY_ORDER": "params_first" }),
            events,
        )]),
    );
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let mut cmd = serve_cmd(BIN, &base, &home.path().join("s.jsonl").to_string_lossy());
    fast_knobs(&mut cmd)
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_MAX_MESSAGE_BYTES", "65536")
        .stderr(Stdio::null());
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(&mut child, None);
    let control = common::mcp_events_fixture::wait_control_file(&control_file);
    common::mcp_events_fixture::wait_active(&mut stdin, &mut frames, 2);
    common::mcp_events_fixture::emit(
        &control,
        json!({ "project": "alpha", "event_id": "huge-1", "data": { "blob": "x".repeat(200_000) } }),
    );
    common::mcp_events_fixture::emit(
        &control,
        json!({ "project": "alpha", "event_id": "small-1", "data": { "ok": true } }),
    );
    frames.wait(Duration::from_secs(20), "the next event", |f| {
        f["type"] == "mcp_event" && f["event"]["eventId"] == "small-1"
    });
    common::mcp_events_fixture::eventually(Duration::from_secs(20), "the model told", || {
        bodies
            .lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("larger than the message-size limit"))
            .then_some(())
    });
    // Long enough for a second notice to be coalesced and injected, if one were coming.
    frames.collect(Duration::from_secs(3), |_| false);
    let gaps = frames
        .seen
        .iter()
        .filter(|f| f["type"] == "mcp_event_status" && f["kind"] == "gap")
        .count();
    assert_eq!(gaps, 2, "each stream records the gap in its own state");
    let notices = bodies
        .lock()
        .unwrap()
        .iter()
        .map(|b| b.matches("larger than the message-size limit").count())
        .max()
        .unwrap_or(0);
    assert_eq!(notices, 1, "one dropped message, one notice to the model");
}
