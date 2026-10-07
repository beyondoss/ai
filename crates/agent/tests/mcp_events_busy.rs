//! MCP Events (draft): an event that arrives while the session is busy with something that is
//! not a prompt — a host `bash` command, a manual `compact` — is queued and delivered to the model
//! afterwards, never refused as busy. Also: `mcp_events_*` commands are answered mid-run, without
//! waiting for the run. Real `serve`, real fixture process, mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::Duration;

use common::mcp_events_fixture::{
    Frames, emit, fast_knobs, model_requests_with, send, stdio_server, wait_active,
    wait_control_file, write_settings,
};
use common::{BIN, ChildGuard, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::{Value, json};

struct Serve {
    _child: ChildGuard,
    stdin: std::process::ChildStdin,
    frames: Frames,
    bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    fixture: String,
    _home: tempfile::TempDir,
}

fn start(extra_args: &[&str]) -> Serve {
    let home = tempfile::tempdir().unwrap();
    let control_file = home.path().join("control");
    write_settings(
        home.path(),
        json!([stdio_server(
            "tickets",
            &control_file,
            json!({}),
            json!([{ "name": "ticket.updated", "delivery": "poll" }]),
        )]),
    );
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let session = home.path().join("s.jsonl");
    let stderr = home.path().join("serve.stderr");
    let mut cmd = serve_cmd(BIN, &base, &session.to_string_lossy());
    fast_knobs(&mut cmd)
        .args(extra_args)
        .env("HOME", home.path())
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = Frames::new(&mut child, Some(stderr));
    let fixture = wait_control_file(&control_file);
    wait_active(&mut stdin, &mut frames, 1);
    Serve {
        _child: child,
        stdin,
        frames,
        bodies,
        fixture,
        _home: home,
    }
}

fn position(frames: &[Value], pred: impl Fn(&Value) -> bool) -> usize {
    frames.iter().position(pred).expect("frame present")
}

#[test]
fn an_event_during_a_host_bash_command_is_queued_and_reaches_the_model_after_it() {
    let mut s = start(&[]);
    send(
        &mut s.stdin,
        json!({ "type": "bash", "id": "sh", "command": "sleep 2" }),
    );
    // The command is provably running before the event is emitted.
    s.frames
        .wait(Duration::from_secs(10), "the bash tool_start", |f| {
            f["type"] == "event" && f["event"]["kind"] == "tool_start"
        });
    emit(
        &s.fixture,
        json!({ "data": { "ticket_id": "B-1", "summary": "during bash" } }),
    );
    s.frames
        .wait(Duration::from_secs(10), "the mcp_event frame", |f| {
            f["type"] == "mcp_event"
        });

    let injected = s.frames.response("mcp_events:1");
    assert_eq!(
        injected["success"], true,
        "the injection must not be refused as busy: {injected:#}"
    );
    let bash_at = position(&s.frames.seen, |f| {
        f["type"] == "response" && f["id"] == "sh"
    });
    let inject_at = position(&s.frames.seen, |f| {
        f["type"] == "response" && f["id"] == "mcp_events:1"
    });
    assert!(
        bash_at < inject_at,
        "the event runs after the bash command, not instead of it"
    );
    assert_eq!(s.frames.seen[bash_at]["success"], true);
    assert!(
        !s.frames
            .seen
            .iter()
            .any(|f| f["id"] == "mcp_events:1" && f["success"] == false),
        "never answered busy"
    );
    assert_eq!(model_requests_with(&s.bodies, "during bash").len(), 1);
}

#[test]
fn an_event_during_a_manual_compact_is_queued_and_reaches_the_model_after_it() {
    let mut s = start(&["--compaction-keep-recent-tokens", "1"]);
    // Two turns of history, the second asking the mock model to stall: every later request that
    // carries the transcript — the compaction's summarization call included — stalls 1.5 s too.
    for (id, msg) in [
        ("p1", "hello".to_owned()),
        ("p2", beyond_ai_test_support::stall_prompt(1500)),
    ] {
        send(
            &mut s.stdin,
            json!({ "type": "prompt", "id": id, "message": msg }),
        );
        assert_eq!(s.frames.response(id)["success"], true);
    }
    send(&mut s.stdin, json!({ "type": "compact", "id": "c" }));
    std::thread::sleep(Duration::from_millis(200));
    emit(
        &s.fixture,
        json!({ "data": { "ticket_id": "C-1", "summary": "during compact" } }),
    );
    let compact = s.frames.response("c");
    assert_eq!(compact["success"], true, "{compact:#}");
    assert_eq!(compact["data"]["compacted"], true, "{compact:#}");
    let injected = s.frames.response("mcp_events:1");
    assert_eq!(injected["success"], true, "{injected:#}");
    // `--compaction-keep-recent-tokens 1` makes the run compact again and continue, so more than
    // one request can carry the transcript; what matters is that the event reached the model, once.
    let reqs = model_requests_with(&s.bodies, "during compact");
    assert!(!reqs.is_empty(), "the model saw the event");
    for r in &reqs {
        assert_eq!(
            r.matches("[MCP events · batch ").count(),
            1,
            "injected exactly once"
        );
    }
}

#[test]
fn mcp_events_commands_are_answered_mid_run_without_waiting_for_it() {
    let mut s = start(&[]);
    send(
        &mut s.stdin,
        json!({ "type": "prompt", "id": "long", "message": beyond_ai_test_support::stall_prompt(3000) }),
    );
    s.frames
        .wait(Duration::from_secs(10), "the prompt's ack", |f| {
            f["type"] == "ack" && f["id"] == "long"
        });
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_list", "id": "mid" }),
    );
    let listed = s.frames.response("mid");
    assert_eq!(listed["success"], true, "{listed:#}");
    assert_eq!(listed["data"]["subscriptions"][0]["state"], "active");
    // Answered while the 3 s run is still going.
    assert!(
        !s.frames
            .seen
            .iter()
            .any(|f| f["type"] == "response" && f["id"] == "long"),
        "the list came back before the run ended"
    );
    // A subscribe mid-run is accepted too.
    send(
        &mut s.stdin,
        json!({ "type": "mcp_events_subscribe", "id": "sub", "server": "tickets", "name": "build.finished", "action": "notify" }),
    );
    assert_eq!(s.frames.response("sub")["success"], true);
    assert_eq!(s.frames.response("long")["success"], true);
}
