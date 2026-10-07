//! MCP Events (draft) over **poll**, end to end: a real stdio `serve`, a real events fixture
//! process, a mock model. Discovery, `events/poll` delivery into an `mcp_event` frame *and* into a
//! model-visible follow-up, and the "server without the extension" cases.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::Duration;

use common::mcp_events_fixture::{
    Frames, emit, eventually, fast_knobs, model_requests_with, send, state, stdio_server,
    wait_active, wait_control_file, write_settings,
};
use common::{BIN, ChildGuard, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::{Value, json};

struct Serve {
    _child: ChildGuard,
    stdin: std::process::ChildStdin,
    frames: Frames,
    bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    _home: tempfile::TempDir,
}

fn start(mcp_servers: Value, home: tempfile::TempDir) -> Serve {
    write_settings(home.path(), mcp_servers);
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let session = home.path().join("s.jsonl");
    let stderr = home.path().join("serve.stderr");
    let mut cmd = serve_cmd(BIN, &base, &session.to_string_lossy());
    fast_knobs(&mut cmd)
        .env("HOME", home.path())
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    let mut child = cmd.spawn_guarded();
    let stdin = child.stdin.take().unwrap();
    let frames = Frames::new(&mut child, Some(stderr));
    Serve {
        _child: child,
        stdin,
        frames,
        bodies,
        _home: home,
    }
}

#[test]
fn poll_delivers_an_event_as_a_frame_and_as_a_model_visible_follow_up() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "tickets",
        &control_file,
        json!({}),
        json!([{
            "name": "ticket.updated",
            "arguments": { "project": "alpha" },
            "delivery": "poll",
            "instructions": "Summarize the ticket change in one line."
        }]),
    );
    let mut s = start(json!([server]), home);
    let control = wait_control_file(&control_file);

    // Discovery: the list command reports what the server offers and the live subscription.
    let listed = wait_active(&mut s.stdin, &mut s.frames, 1);
    let available = &listed["data"]["available"][0];
    assert_eq!(available["server"], "tickets");
    assert_eq!(available["supported"], true);
    let names: Vec<&str> = available["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["ticket.updated", "build.finished"]);
    assert_eq!(listed["data"]["subscriptions"][0]["delivery"], "poll");
    assert_eq!(
        listed["data"]["webhook"], false,
        "stdio serve has no callback URL"
    );

    // A filtered-out event (another project) and a matching one.
    emit(
        &control,
        json!({ "project": "beta", "data": { "ticket_id": "B-9", "summary": "other project" } }),
    );
    emit(
        &control,
        json!({ "project": "alpha", "data": { "ticket_id": "T-1", "summary": "customer replied" } }),
    );

    let frame = s
        .frames
        .wait(Duration::from_secs(20), "an mcp_event frame", |f| {
            f["type"] == "mcp_event"
        });
    assert_eq!(frame["server"], "tickets");
    assert_eq!(frame["name"], "ticket.updated");
    assert_eq!(frame["delivery"], "poll");
    assert_eq!(frame["arguments"], json!({ "project": "alpha" }));
    assert_eq!(frame["event"]["data"]["ticket_id"], "T-1");
    assert_eq!(frame["event"]["eventId"], "evt_2");

    // The injected follow-up runs the model with the event in context.
    let response = s.frames.response("mcp_events:1");
    assert_eq!(response["success"], true, "{response:#}");
    let requests = model_requests_with(&s.bodies, "customer replied");
    assert_eq!(requests.len(), 1, "exactly one model run for one event");
    let body = &requests[0];
    assert!(
        body.contains("untrusted data"),
        "the payload is labelled untrusted"
    );
    assert!(
        body.contains("Summarize the ticket change in one line."),
        "operator instructions reach the model"
    );
    assert!(body.contains("T-1"));
    assert!(
        !body.contains("other project"),
        "the server-side filter held"
    );

    // The cursor advanced: the same event is never polled twice.
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(model_requests_with(&s.bodies, "customer replied").len(), 1);
    let methods = state(&control)["methods"].clone();
    assert!(
        methods
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/poll")
            .count()
            >= 2
    );
}

#[test]
fn poll_only_event_types_negotiate_poll_and_unknown_events_fail_clearly() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server("ci", &control_file, json!({}), json!([]));
    let mut s = start(json!([server]), home);
    let _control = wait_control_file(&control_file);

    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "sub", "server": "ci", "name": "build.finished", "action": "notify" }),
    );
    let r = s.frames.response("sub");
    assert_eq!(r["success"], true, "{r:#}");
    assert_eq!(
        r["data"]["delivery"], "poll",
        "the only offered mode is chosen"
    );
    assert_eq!(r["data"]["action"], "notify");

    // Idempotent upsert: the same subscribe again is the same subscription.
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "sub2", "server": "ci", "name": "build.finished", "action": "notify" }),
    );
    assert_eq!(s.frames.response("sub2")["success"], true);
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_list", "id": "l" }),
    );
    assert_eq!(
        s.frames.response("l")["data"]["subscriptions"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "push", "server": "ci", "name": "build.finished", "delivery": "push" }),
    );
    let r = s.frames.response("push");
    assert_eq!(r["success"], false);
    assert!(
        r["error"]
            .as_str()
            .unwrap()
            .contains("does not offer `push`"),
        "{r:#}"
    );

    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "nope", "server": "ci", "name": "no.such.event" }),
    );
    let r = s.frames.response("nope");
    assert_eq!(r["success"], false);
    assert!(
        r["error"]
            .as_str()
            .unwrap()
            .contains("offers no event `no.such.event`"),
        "{r:#}"
    );

    // `notify` never reaches the model.
    let control = wait_control_file(&s._home.path().join("control"));
    emit(
        &control,
        json!({ "name": "build.finished", "data": { "ok": true } }),
    );
    s.frames
        .wait(Duration::from_secs(20), "the build.finished frame", |f| {
            f["type"] == "mcp_event" && f["name"] == "build.finished"
        });
    std::thread::sleep(Duration::from_millis(800));
    assert!(
        s.bodies.lock().unwrap().is_empty(),
        "a notify subscription never runs the model"
    );

    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_unsubscribe", "id": "u", "server": "ci", "name": "build.finished" }),
    );
    assert_eq!(s.frames.response("u")["data"]["removed"], true);
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_unsubscribe", "id": "u2", "server": "ci", "name": "build.finished" }),
    );
    let again = s.frames.response("u2");
    assert_eq!(again["success"], true, "unsubscribe is idempotent");
    assert_eq!(again["data"]["removed"], false);
}

#[test]
fn a_server_without_the_extension_is_reported_and_its_tools_are_unchanged() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "plain",
        &control_file,
        json!({ "MCP_FIXTURE_NO_EVENTS": "1" }),
        json!([{ "name": "ticket.updated" }]),
    );
    let mut s = start(json!([server]), home);
    let _ = wait_control_file(&control_file);

    let status = s
        .frames
        .wait(Duration::from_secs(20), "the subscribe failure", |f| {
            // Permanent: the server does not speak the extension.
            f["type"] == "mcp_event_status" && f["kind"] == "refused"
        });
    assert!(
        status["error"]
            .as_str()
            .unwrap()
            .contains("does not support the MCP Events extension"),
        "{status:#}"
    );
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_list", "id": "l" }),
    );
    let l = s.frames.response("l");
    assert_eq!(l["data"]["available"][0]["supported"], false);
    assert!(l["data"]["subscriptions"].as_array().unwrap().is_empty());

    // Tools still work exactly as before.
    send(&mut s.stdin, json!({ "type": "get_mcp", "id": "m" }));
    let m = s.frames.response("m");
    assert!(
        m["data"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "mcp__plain__echo"),
        "{m:#}"
    );
}

#[test]
fn a_server_with_no_events_configured_is_never_asked_about_events() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server("quiet", &control_file, json!({}), json!([]));
    let mut s = start(json!([server]), home);
    let control = wait_control_file(&control_file);
    // Give a would-be background subscriber every chance to run.
    send(&mut s.stdin, json!({ "type": "get_state", "id": "g" }));
    s.frames.response("g");
    std::thread::sleep(Duration::from_millis(500));
    let methods = state(&control)["methods"].clone();
    assert!(
        !methods
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m.as_str().unwrap().starts_with("events/")),
        "no events/* request without configuration: {methods}"
    );
}

#[test]
fn a_session_bound_legacy_http_server_is_reached_through_its_own_session_and_gets_max_age() {
    // A pre-2026-07-28 streamable-HTTP server: `initialize` mints an `Mcp-Session-Id` and every
    // later request without it is refused. Events must go through the connection that holds it.
    let (_fx, mcp_url, fixture) =
        common::mcp_events_fixture::spawn_http_fixture(&[("MCP_FIXTURE_LEGACY_SESSION", "1")]);
    let home = tempfile::tempdir().unwrap();
    let server = json!({
        "name": "legacy",
        "transport": "http",
        "url": mcp_url,
        "events": [{ "name": "ticket.updated", "delivery": "poll", "action": "notify", "max_age_ms": 60000 }],
    });
    let mut s = start(json!([server]), home);
    let listed = wait_active(&mut s.stdin, &mut s.frames, 1);
    assert_eq!(
        listed["data"]["available"][0]["supported"], true,
        "{listed:#}"
    );
    emit(
        &fixture,
        json!({ "event_id": "legacy-1", "data": { "n": 1 } }),
    );
    let f = s
        .frames
        .wait(Duration::from_secs(20), "the legacy server's event", |f| {
            f["type"] == "mcp_event"
        });
    assert_eq!(f["event"]["eventId"], "legacy-1");
    let st = state(&fixture);
    assert_eq!(
        st["sessionless_rejections"], 0,
        "nothing went out without the session: {st:#}"
    );
    let poll = st["requests"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["method"] == "events/poll")
        .cloned()
        .unwrap();
    assert_eq!(poll["params"]["maxAgeMs"], 60000, "{poll:#}");
}

/// `hasMore: true` with no events, forever: a broken server. The client must back off to its
/// poll floor instead of spinning.
#[test]
fn has_more_with_no_events_does_not_spin() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "spin",
        &control_file,
        json!({ "MCP_FIXTURE_HASMORE_EMPTY": "1" }),
        json!([{ "name": "ticket.updated", "delivery": "poll", "action": "notify" }]),
    );
    let mut s = start(json!([server]), home);
    let control = wait_control_file(&control_file);
    wait_active(&mut s.stdin, &mut s.frames, 1);
    let polls = |st: &Value| {
        st["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/poll")
            .count()
    };
    let before = polls(&state(&control));
    std::thread::sleep(Duration::from_millis(1500));
    let during = polls(&state(&control)) - before;
    // The floor is 100 ms here: about 15 polls in 1.5 s. A spinning client makes thousands.
    assert!(during <= 25, "{during} polls in 1.5 s");
}

/// `hasMore: true` on a page whose events are all duplicates is no backlog either: the poll floor
/// applies, so a server repeating itself cannot spin the client.
#[test]
fn has_more_with_only_duplicates_does_not_spin() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "spin",
        &control_file,
        json!({ "MCP_FIXTURE_HASMORE_DUPS": "1" }),
        json!([{ "name": "ticket.updated", "delivery": "poll", "action": "notify" }]),
    );
    let mut s = start(json!([server]), home);
    let control = wait_control_file(&control_file);
    wait_active(&mut s.stdin, &mut s.frames, 1);
    let polls = |st: &Value| {
        st["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/poll")
            .count()
    };
    let before = polls(&state(&control));
    std::thread::sleep(Duration::from_millis(1500));
    let during = polls(&state(&control)) - before;
    assert!(during <= 25, "{during} polls in 1.5 s");
}

/// No events configured, nothing subscribed: the session leaves no MCP Events files behind.
#[test]
fn a_session_without_events_writes_no_events_state() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server("tools", &control_file, json!({}), json!([]));
    let mut s = start(json!([server]), home);
    send(
        &mut s.stdin,
        json!({ "type": "prompt", "id": "p", "message": "hi" }),
    );
    s.frames.response("p");
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_list", "id": "l" }),
    );
    s.frames.response("l");
    let dir = s._home.path().to_path_buf();
    drop(s.stdin);
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while s._child.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "serve did not exit");
        std::thread::sleep(Duration::from_millis(50));
    }
    let leftovers: Vec<String> = walk(&dir)
        .into_iter()
        .filter(|n| n.contains(".mcp-events."))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

/// Every file name under `dir`, recursively.
fn walk(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p.to_string_lossy().into_owned());
        }
    }
    out
}

/// A `truncated` page means events may have been skipped: the model is told, not just the frame.
#[test]
fn a_gap_is_explained_to_the_model() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "tickets",
        &control_file,
        json!({}),
        json!([{ "name": "ticket.updated", "delivery": "poll" }]),
    );
    let mut s = start(json!([server]), home);
    let control = wait_control_file(&control_file);
    wait_active(&mut s.stdin, &mut s.frames, 1);
    common::mcp_events_fixture::control(
        &control,
        "POST",
        "/control/truncate_next",
        Some(&json!({})),
    );
    s.frames
        .wait(Duration::from_secs(20), "the gap status", |f| {
            f["type"] == "mcp_event_status" && f["kind"] == "gap"
        });
    eventually(
        Duration::from_secs(20),
        "the model being told about the gap",
        || (!model_requests_with(&s.bodies, "may have been missed").is_empty()).then_some(()),
    );
}

/// Subscribing to a slow server does not hold up subscribing to a fast one: the per-key lock is
/// not a global one, and discovery happens outside it.
#[test]
fn a_slow_server_does_not_serialize_subscriptions_to_others() {
    let home = tempfile::tempdir().unwrap();
    let slow = stdio_server(
        "slow",
        &home.path().join("control-slow"),
        json!({ "MCP_FIXTURE_LIST_DELAY_MS": "3000" }),
        json!([]),
    );
    let fast = stdio_server(
        "fast",
        &home.path().join("control-fast"),
        json!({}),
        json!([]),
    );
    let mut s = start(json!([slow, fast]), home);
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "slow", "server": "slow", "name": "build.finished", "action": "notify" }),
    );
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "fast", "server": "fast", "name": "build.finished", "action": "notify" }),
    );
    let first = s.frames.wait(
        Duration::from_secs(30),
        "the first subscribe response",
        |f| f["type"] == "response" && (f["id"] == "slow" || f["id"] == "fast"),
    );
    assert_eq!(
        first["id"], "fast",
        "the fast server answered first: {first:#}"
    );
    assert_eq!(first["success"], true);
    assert_eq!(s.frames.response("slow")["success"], true);
}
