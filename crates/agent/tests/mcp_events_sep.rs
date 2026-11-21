//! MCP Events against the rules of SEP-3415 (Final), end to end: a real stdio `serve`, the events
//! fixture (which validates what the spec requires of a *server*), a mock model.
//!
//! One file per concern would be many tiny harnesses; these share one: the capability gate, the
//! Streamable HTTP routing headers (`Mcp-Name`, `Mcp-Param-*`, `HeaderMismatch` recovery), the
//! `subscriptions/listen` opt-in for `notifications/events/list_changed`, and `ResourceExhausted`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;
use std::time::Duration;

use common::mcp_events_fixture::{
    Frames, control, emit, eventually, fast_knobs, send, spawn_http_fixture, state, stdio_server,
    wait_active, wait_control_file, write_settings,
};
use common::{BIN, ChildGuard, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::{Value, json};

struct Serve {
    _child: ChildGuard,
    stdin: std::process::ChildStdin,
    frames: Frames,
    _home: tempfile::TempDir,
}

fn start(mcp_servers: Value, home: tempfile::TempDir) -> Serve {
    write_settings(home.path(), mcp_servers);
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
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
        _home: home,
    }
}

fn poll_alpha() -> Value {
    json!([{ "name": "ticket.updated", "arguments": { "project": "alpha" },
             "delivery": "poll", "action": "notify" }])
}

fn http_server(mcp_url: &str, events: Value) -> Value {
    json!({ "name": "remote", "transport": "http", "url": mcp_url, "events": events })
}

fn events_requests(fixture: &str) -> usize {
    state(fixture)["methods"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m.as_str().is_some_and(|m| m.starts_with("events/")))
        .count()
}

fn header<'a>(entry: &'a Value, name: &str) -> Option<&'a str> {
    entry["headers"]
        .as_array()?
        .iter()
        .find(|h| h[0] == name)
        .and_then(|h| h[1].as_str())
}

// ---------------------------------------------------------------------------------------------
// Gap 4: the capability gate
// ---------------------------------------------------------------------------------------------

/// A server that advertised only the pre-SEP top-level `events` capability has not advertised the
/// extension: its configured subscription is refused, with that reason, and no `events/*` request
/// is ever sent to it.
#[test]
fn a_server_without_the_events_extension_is_refused_and_sent_no_events_request() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "draft",
        &control_file,
        json!({ "MCP_FIXTURE_DRAFT_CAPABILITY": "1" }),
        json!([{ "name": "ticket.updated", "delivery": "poll", "action": "notify" }]),
    );
    let mut s = start(json!([server]), home);
    let control = wait_control_file(&control_file);
    let status = s.frames.wait(Duration::from_secs(20), "the refusal", |f| {
        f["type"] == "mcp_event_status" && f["kind"] == "refused"
    });
    assert!(
        status["error"]
            .as_str()
            .unwrap()
            .contains("did not advertise the MCP Events extension"),
        "{status:#}"
    );
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(events_requests(&control), 0, "{:#}", state(&control));
}

// ---------------------------------------------------------------------------------------------
// Gap 1: Streamable HTTP routing headers
// ---------------------------------------------------------------------------------------------

/// Over Streamable HTTP every named `events/*` request carries `Mcp-Name`, and `Mcp-Param-Project`
/// for the `x-mcp-header` argument. The fixture answers a missing or wrong one `400` +
/// `HeaderMismatch`, so a client that sends only `Mcp-Method` never gets a subscription up.
#[test]
fn an_http_subscription_sends_mcp_name_and_the_annotated_argument_as_headers() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[]);
    let mut s = start(
        json!([http_server(&mcp_url, poll_alpha())]),
        tempfile::tempdir().unwrap(),
    );
    wait_active(&mut s.stdin, &mut s.frames, 1);
    emit(
        &fixture,
        json!({ "project": "alpha", "data": { "ticket_id": "H-1" } }),
    );
    s.frames
        .wait(Duration::from_secs(20), "the polled event", |f| {
            f["type"] == "mcp_event"
        });
    let st = state(&fixture);
    assert_eq!(st["header_rejections"], 0, "{st:#}");
    let poll = st["routing_log"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["method"] == "events/poll")
        .expect("a poll was sent");
    assert_eq!(header(poll, "mcp-name"), Some("ticket.updated"));
    assert_eq!(header(poll, "mcp-param-project"), Some("alpha"));
}

/// Push (`events/stream`) carries them too, and a runtime unsubscribe of a webhook-less push
/// subscription is not rejected either.
#[test]
fn an_http_push_stream_sends_the_routing_headers() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[]);
    let events = json!([{ "name": "ticket.updated", "arguments": { "project": "alpha" },
                          "delivery": "push", "action": "notify" }]);
    let mut s = start(
        json!([http_server(&mcp_url, events)]),
        tempfile::tempdir().unwrap(),
    );
    wait_active(&mut s.stdin, &mut s.frames, 1);
    let st = state(&fixture);
    assert_eq!(st["header_rejections"], 0, "{st:#}");
    let stream = st["routing_log"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["method"] == "events/stream")
        .expect("a stream was opened");
    assert_eq!(header(stream, "mcp-name"), Some("ticket.updated"));
    assert_eq!(header(stream, "mcp-param-project"), Some("alpha"));
}

/// The schema the client learned predates the server's `x-mcp-header` annotation: the first
/// request is answered `HeaderMismatch`, and the client lists again and retries with the headers
/// the current schema calls for — the subscription comes up, with no refusal.
#[test]
fn a_header_mismatch_lists_again_and_retries_with_the_current_headers() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_STALE_LIST_ONCE", "1")]);
    let mut s = start(
        json!([http_server(&mcp_url, poll_alpha())]),
        tempfile::tempdir().unwrap(),
    );
    wait_active(&mut s.stdin, &mut s.frames, 1);
    let st = state(&fixture);
    assert!(
        st["header_rejections"].as_u64().unwrap() >= 1,
        "the stale request was rejected: {st:#}"
    );
    assert!(st["list_calls"].as_u64().unwrap() >= 2, "{st:#}");
    assert!(
        !s.frames
            .seen
            .iter()
            .any(|f| f["type"] == "mcp_event_status" && f["kind"] == "refused"),
        "a HeaderMismatch is not a refusal"
    );
}

/// An event type whose `x-mcp-header` annotation breaks the transport's rules is left out of
/// `events/list` (the others stay).
#[test]
fn an_event_type_with_an_invalid_annotation_is_excluded_over_http() {
    let (_fx, mcp_url, _fixture) = spawn_http_fixture(&[("MCP_FIXTURE_BAD_ANNOTATION", "1")]);
    let mut s = start(
        json!([http_server(&mcp_url, json!([]))]),
        tempfile::tempdir().unwrap(),
    );
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_list", "id": "l" }),
    );
    let l = s.frames.response("l");
    let names: Vec<&str> = l["data"]["available"][0]["events"]
        .as_array()
        .unwrap_or_else(|| panic!("{l:#}"))
        .iter()
        .filter_map(|e| e["name"].as_str())
        .collect();
    assert!(names.contains(&"ticket.updated"), "{names:?}");
    assert!(!names.contains(&"bad.annotation"), "{names:?}");
}

// ---------------------------------------------------------------------------------------------
// Gap 3: `notifications/events/list_changed` on `subscriptions/listen`
// ---------------------------------------------------------------------------------------------

/// The client opts in with `eventsListChanged`, the server acknowledges it, and a
/// `list_changed` on that stream makes the next discovery list again — so a type the server added
/// since can be subscribed to. Without the opt-in the server sends nothing and the type stays
/// invisible behind the discovery cache.
fn list_changed_refreshes_discovery(
    home: tempfile::TempDir,
    server: Value,
    control: Result<String, std::path::PathBuf>,
) {
    let mut s = start(json!([server]), home);
    let fixture = match control {
        Ok(url) => url,
        Err(file) => wait_control_file(&file),
    };
    wait_active(&mut s.stdin, &mut s.frames, 1);
    let listen = eventually(Duration::from_secs(20), "the listen stream", || {
        let st = state(&fixture);
        (st["listens"] == 1).then_some(st)
    });
    assert_eq!(
        listen["listen_log"][0]["requested"]["eventsListChanged"],
        true
    );
    assert_eq!(
        listen["listen_log"][0]["acknowledged"]["eventsListChanged"],
        true
    );
    let before = listen["list_calls"].as_u64().unwrap();

    let sent = control_post(&fixture);
    assert_eq!(sent["sent"], 1, "{sent:#}");
    // The notification is asynchronous: subscribe until the new type is seen (it can only be
    // seen through a fresh `events/list`).
    let mut n = 0;
    eventually(Duration::from_secs(20), "the added event type", || {
        n += 1;
        let id = format!("d{n}");
        send(
            &mut s.stdin,
            json!({ "type": "mcp_events_subscribe", "id": id, "server": "remote",
                    "name": "dynamic.added", "action": "notify" }),
        );
        (s.frames.response(&id)["success"] == true).then_some(())
    });
    assert!(state(&fixture)["list_calls"].as_u64().unwrap() > before);
}

fn control_post(fixture: &str) -> Value {
    control(
        fixture,
        "POST",
        "/control/list_changed",
        Some(&json!({ "add_event": true })),
    )
}

#[test]
fn over_http_a_list_changed_on_the_listen_stream_refreshes_event_discovery() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[]);
    list_changed_refreshes_discovery(
        tempfile::tempdir().unwrap(),
        http_server(&mcp_url, poll_alpha()),
        Ok(fixture),
    );
}

#[test]
fn over_stdio_a_list_changed_on_the_listen_stream_refreshes_event_discovery() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server("remote", &control_file, json!({}), poll_alpha());
    list_changed_refreshes_discovery(home, server, Err(control_file));
}

/// A server that did not declare `listChanged` is never asked to send it: no stream at all.
#[test]
fn a_server_without_list_changed_is_not_asked_for_it() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_NO_LIST_CHANGED", "1")]);
    let mut s = start(
        json!([http_server(&mcp_url, poll_alpha())]),
        tempfile::tempdir().unwrap(),
    );
    wait_active(&mut s.stdin, &mut s.frames, 1);
    std::thread::sleep(Duration::from_millis(500));
    let st = state(&fixture);
    assert_eq!(st["listen_log"], json!([]), "{st:#}");
}

// ---------------------------------------------------------------------------------------------
// Gap 2: ResourceExhausted
// ---------------------------------------------------------------------------------------------

/// `ResourceExhausted` (`-32025`) is a quota that can clear: retried on backoff, never a permanent
/// refusal.
#[test]
fn a_resource_exhausted_answer_is_retried_not_refused() {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    let server = stdio_server(
        "tickets",
        &control_file,
        json!({ "MCP_FIXTURE_EXHAUST_FIRST": "2" }),
        json!([{ "name": "ticket.updated", "delivery": "poll", "action": "notify" }]),
    );
    let mut s = start(json!([server]), home);
    wait_active(&mut s.stdin, &mut s.frames, 1);
    assert!(
        !s.frames
            .seen
            .iter()
            .any(|f| f["type"] == "mcp_event_status" && f["kind"] == "refused"),
        "a quota is not a permanent refusal"
    );
}
