//! MCP Events (draft) over **push** (`events/stream`), end to end, on both MCP transports: stdio
//! (notifications interleaved on the server's stdout) and streamable HTTP (an SSE response that
//! stays open). Also: `eventId` dedup under redelivery, explicit unsubscribe cancelling the stream,
//! server-initiated termination, and what an event does while a run is busy.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::Duration;

use common::mcp_events_fixture::{
    Frames, control, emit, eventually, fast_knobs, model_requests_with, send, spawn_http_fixture,
    state, stdio_server, wait_active, wait_control_file, write_settings,
};
use common::{BIN, ChildGuard, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::{Value, json};

struct Serve {
    _child: ChildGuard,
    stdin: std::process::ChildStdin,
    frames: Frames,
    bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    home: tempfile::TempDir,
}

fn start(mcp_servers: Value, extra_env: &[(&str, &str)]) -> Serve {
    let home = tempfile::tempdir().unwrap();
    // Paths inside the settings are relative to this home; callers pass a closure-free value.
    let servers = serde_json::to_string(&mcp_servers)
        .unwrap()
        .replace("$HOME_DIR", &home.path().to_string_lossy());
    write_settings(home.path(), serde_json::from_str(&servers).unwrap());
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let session = home.path().join("s.jsonl");
    let stderr = home.path().join("serve.stderr");
    let mut cmd = serve_cmd(BIN, &base, &session.to_string_lossy());
    fast_knobs(&mut cmd)
        .env("HOME", home.path())
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn_guarded();
    let stdin = child.stdin.take().unwrap();
    let frames = Frames::new(child.stdout.take().unwrap(), Some(stderr));
    Serve {
        _child: child,
        stdin,
        frames,
        bodies,
        home,
    }
}

fn ticket_events(action: &str) -> Value {
    json!([{ "name": "ticket.updated", "arguments": { "project": "alpha" }, "delivery": "push", "action": action }])
}

fn event_frame(f: &Value) -> bool {
    f["type"] == "mcp_event"
}

#[test]
fn push_over_stdio_delivers_dedups_and_unsubscribes() {
    let control_path = "$HOME_DIR/control";
    let server = stdio_server(
        "tickets",
        std::path::Path::new(control_path),
        json!({ "MCP_FIXTURE_HEARTBEAT_MS": "200" }),
        ticket_events("follow_up"),
    );
    let mut s = start(json!([server]), &[]);
    let fixture = wait_control_file(&s.home.path().join("control"));
    wait_active(&mut s.stdin, &mut s.frames, 1);

    let r = emit(
        &fixture,
        json!({ "project": "alpha", "event_id": "evt-push-1", "data": { "ticket_id": "P-1", "summary": "pushed change" } }),
    );
    assert_eq!(r["event_id"], "evt-push-1");
    let frame = s
        .frames
        .wait(Duration::from_secs(20), "the pushed mcp_event", event_frame);
    assert_eq!(frame["delivery"], "push");
    assert_eq!(frame["event"]["eventId"], "evt-push-1");
    assert_eq!(
        frame["event"]["cursor"], "1",
        "push occurrences carry their cursor"
    );
    let resp = s.frames.response("mcp_events:1");
    assert_eq!(resp["success"], true, "{resp:#}");
    assert_eq!(model_requests_with(&s.bodies, "pushed change").len(), 1);

    // Redelivery of the same eventId (an at-least-once retry) is dropped client-side.
    control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "evt-push-1" })),
    );
    let dupes = s.frames.collect(Duration::from_millis(800), event_frame);
    assert!(
        dupes.is_empty(),
        "a redelivered eventId must not surface twice: {dupes:#?}"
    );
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_list", "id": "l" }),
    );
    let l = s.frames.response("l");
    let sub = &l["data"]["subscriptions"][0];
    assert_eq!(sub["duplicates"], 1, "{l:#}");
    assert_eq!(sub["delivered"], 1);
    assert_eq!(
        model_requests_with(&s.bodies, "pushed change").len(),
        1,
        "no second run for a duplicate"
    );

    // Heartbeats keep the stream alive past the idle limit.
    std::thread::sleep(Duration::from_millis(800));
    assert_eq!(state(&fixture)["streams"], 1);

    // Explicit unsubscribe cancels the stream on the server (notifications/cancelled on stdio).
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_unsubscribe", "id": "u", "server": "tickets", "name": "ticket.updated", "arguments": { "project": "alpha" } }),
    );
    assert_eq!(s.frames.response("u")["data"]["removed"], true);
    let st = eventually(
        Duration::from_secs(10),
        "the stream to be cancelled",
        || {
            let st = state(&fixture);
            (st["streams"] == 0 && !st["cancelled"].as_array().unwrap().is_empty()).then_some(st)
        },
    );
    assert_eq!(st["cancelled"].as_array().unwrap().len(), 1);
}

#[test]
fn push_over_streamable_http_reconnects_from_its_cursor_after_idle_and_honours_termination() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_HEARTBEAT_MS", "60000")]);
    let server = json!({
        "name": "remote",
        "transport": "http",
        "url": mcp_url,
        "events": ticket_events("notify"),
    });
    // A 1.5 s idle limit with a server that never heartbeats inside it: the client must presume
    // the stream dead and reconnect with its cursor.
    let mut s = start(
        json!([server]),
        &[("BEYOND_AI_AGENT_MCP_EVENTS_STREAM_IDLE_MS", "1500")],
    );
    wait_active(&mut s.stdin, &mut s.frames, 1);

    emit(
        &fixture,
        json!({ "project": "alpha", "data": { "ticket_id": "H-1" } }),
    );
    let first = s.frames.wait(
        Duration::from_secs(20),
        "the first SSE-delivered event",
        event_frame,
    );
    assert_eq!(first["event"]["data"]["ticket_id"], "H-1");

    // Wait out an idle reconnect, emitting nothing; then an event emitted after it still arrives,
    // exactly once (the replay from the cursor does not resend H-1).
    eventually(Duration::from_secs(15), "a reconnect", || {
        let st = state(&fixture);
        let streams_opened = st["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/stream")
            .count();
        (streams_opened >= 2).then_some(())
    });
    emit(
        &fixture,
        json!({ "project": "alpha", "data": { "ticket_id": "H-2" } }),
    );
    let second = s.frames.wait(
        Duration::from_secs(20),
        "the post-reconnect event",
        event_frame,
    );
    assert_eq!(second["event"]["data"]["ticket_id"], "H-2");
    let extra = s.frames.collect(Duration::from_millis(500), event_frame);
    assert!(extra.is_empty(), "no replayed duplicates: {extra:#?}");

    // Server-initiated termination ends the subscription and is reported.
    control(&fixture, "POST", "/control/terminate", Some(&json!({})));
    let t = s
        .frames
        .wait(Duration::from_secs(20), "the terminated status", |f| {
            f["type"] == "mcp_event_status" && f["kind"] == "terminated"
        });
    assert_eq!(t["error"]["code"], -32012);
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_list", "id": "l" }),
    );
    let l = s.frames.response("l");
    assert_eq!(
        l["data"]["subscriptions"][0]["state"], "terminated",
        "{l:#}"
    );
}

#[test]
fn events_arriving_during_a_busy_run_coalesce_into_one_follow_up_after_it() {
    let control_path = "$HOME_DIR/control";
    let server = stdio_server(
        "tickets",
        std::path::Path::new(control_path),
        json!({}),
        ticket_events("follow_up"),
    );
    let mut s = start(json!([server]), &[]);
    let fixture = wait_control_file(&s.home.path().join("control"));
    wait_active(&mut s.stdin, &mut s.frames, 1);

    // A user prompt whose model call stalls for 2 s: the run is provably in flight meanwhile.
    send(
        &mut s.stdin,
        json!({ "type": "prompt", "id": "user", "message": beyond_ai_test_support::stall_prompt(2000) }),
    );
    s.frames
        .wait(Duration::from_secs(10), "the user's ack", |f| {
            f["type"] == "ack" && f["id"] == "user"
        });
    for n in 1..=3 {
        emit(
            &fixture,
            json!({ "project": "alpha", "data": { "ticket_id": format!("BUSY-{n}") } }),
        );
    }
    // All three frames arrive live, while the run is still going.
    let frames = (0..3)
        .map(|_| {
            s.frames.wait(
                Duration::from_secs(10),
                "a busy-time event frame",
                event_frame,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), 3);

    let user = s.frames.response("user");
    assert_eq!(user["success"], true);
    // Nothing was injected mid-run: the injection's response comes after the user's.
    let injected = s.frames.response("mcp_events:1");
    assert_eq!(injected["success"], true, "{injected:#}");
    let runs = model_requests_with(&s.bodies, "BUSY-1");
    assert_eq!(runs.len(), 1, "one coalesced run for three events");
    for n in 1..=3 {
        assert!(
            runs[0].contains(&format!("BUSY-{n}")),
            "event {n} rides the single follow-up"
        );
    }
    assert!(runs[0].contains("3 event(s) arrived"));
    // And no second injection follows.
    let more = s
        .frames
        .collect(Duration::from_millis(1000), |f| f["id"] == "mcp_events:2");
    assert!(more.is_empty());
}

/// A burst far larger than the stream's notification buffer: the router overflows, the stream
/// reconnects from the last cursor it actually delivered, and the server replays the rest — every
/// event arrives, exactly once.
#[test]
fn a_burst_that_overflows_the_stream_buffer_loses_nothing() {
    let control_path = "$HOME_DIR/control";
    let server = stdio_server(
        "tickets",
        std::path::Path::new(control_path),
        json!({ "MCP_FIXTURE_HEARTBEAT_MS": "200" }),
        json!([{ "name": "ticket.updated", "delivery": "push", "action": "notify" }]),
    );
    let mut s = start(
        json!([server]),
        &[("BEYOND_AI_AGENT_MCP_EVENTS_STREAM_BUFFER", "4")],
    );
    let fixture = wait_control_file(&s.home.path().join("control"));
    wait_active(&mut s.stdin, &mut s.frames, 1);
    let r = control(
        &fixture,
        "POST",
        "/control/emit_burst",
        Some(&json!({ "count": 60 })),
    );
    let want: Vec<String> = r["event_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(want.len(), 60);
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut got: Vec<String> = Vec::new();
    while got.len() < want.len() && std::time::Instant::now() < deadline {
        for f in s.frames.collect(Duration::from_millis(500), event_frame) {
            got.push(f["event"]["eventId"].as_str().unwrap().to_owned());
        }
    }
    let mut sorted = got.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), got.len(), "no event surfaced twice");
    let mut want_sorted = want.clone();
    want_sorted.sort();
    assert_eq!(sorted, want_sorted, "every event in the burst arrived");
    assert!(
        s.frames.seen.iter().any(|f| f["type"] == "mcp_event_status"
            && f["error"]
                .as_str()
                .is_some_and(|e| e.contains("overflowed"))),
        "the buffer really overflowed (the path under test ran)"
    );
}

/// The server changes the event type's schema in place and ends the stream with the draft's
/// `-32014 {reason: schema_changed}`: the client re-fetches `events/list`, resubscribes, and
/// delivery continues.
#[test]
fn a_schema_change_termination_rediscovers_and_resubscribes() {
    let control_path = "$HOME_DIR/control";
    let server = stdio_server(
        "tickets",
        std::path::Path::new(control_path),
        json!({ "MCP_FIXTURE_HEARTBEAT_MS": "200" }),
        json!([{ "name": "ticket.updated", "delivery": "push", "action": "notify" }]),
    );
    let mut s = start(json!([server]), &[]);
    let fixture = wait_control_file(&s.home.path().join("control"));
    wait_active(&mut s.stdin, &mut s.frames, 1);
    let lists_before = count(&state(&fixture)["methods"], "events/list");
    let r = control(&fixture, "POST", "/control/schema_change", Some(&json!({})));
    assert_eq!(r["streams"], 1, "{r:#}");
    s.frames
        .wait(Duration::from_secs(20), "the resubscription", |f| {
            f["type"] == "mcp_event_status" && f["kind"] == "resubscribed"
        });
    let st = state(&fixture);
    assert!(
        count(&st["methods"], "events/list") > lists_before,
        "re-discovered: {st:#}"
    );
    assert!(
        count(&st["methods"], "events/stream") >= 2,
        "a new stream: {st:#}"
    );
    emit(&fixture, json!({ "event_id": "after-change", "data": {} }));
    let f = s.frames.wait(
        Duration::from_secs(20),
        "an event after the change",
        event_frame,
    );
    assert_eq!(f["event"]["eventId"], "after-change");
}

fn count(methods: &Value, name: &str) -> usize {
    methods
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| *m == name)
        .count()
}

/// A runtime subscription the server keeps ending with `schema_changed`: each time it comes back
/// the model is told events may have been missed in between, and the re-discovery budget is
/// restored once the subscription has stayed up for a while — so a long-lived subscription that
/// sees a schema change now and then is not given up on after the fifth.
#[test]
fn rediscovery_tells_the_model_about_the_gap_and_its_budget_recovers() {
    let control_path = "$HOME_DIR/control";
    let server = stdio_server(
        "tickets",
        std::path::Path::new(control_path),
        json!({ "MCP_FIXTURE_HEARTBEAT_MS": "200" }),
        json!([]),
    );
    let mut s = start(
        json!([server]),
        &[("BEYOND_AI_AGENT_MCP_EVENTS_HEALTHY_MS", "300")],
    );
    let fixture = wait_control_file(&s.home.path().join("control"));
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "tickets",
                "name": "ticket.updated", "arguments": { "project": "alpha" },
                "delivery": "push", "action": "follow_up" }),
    );
    let r = s.frames.response("s");
    assert_eq!(r["success"], true, "{r:#}");
    for round in 1..=6 {
        // Up for longer than the healthy period before each change.
        std::thread::sleep(Duration::from_millis(500));
        let r = control(&fixture, "POST", "/control/schema_change", Some(&json!({})));
        assert_eq!(r["streams"], 1, "round {round}: {r:#}");
        s.frames.wait(
            Duration::from_secs(20),
            &format!("resubscription {round}"),
            |f| f["type"] == "mcp_event_status" && f["kind"] == "resubscribed",
        );
    }
    eventually(
        Duration::from_secs(20),
        "the gap notice to reach the model",
        || (!model_requests_with(&s.bodies, "may have been missed").is_empty()).then_some(()),
    );
}
