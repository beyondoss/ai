//! A reopened service session leaves its sandbox asleep until it is actually used.
//!
//! A sandbox sleeps when idle and any exec request wakes it, so a session that reopens — after a
//! replica restart, a reap, a deploy — must not touch it just because a client looked at the chat.
//! The first life of a session probes and discovers as it always did; what it learned is saved
//! beside the transcript. The next life restores it and makes **no** exec request until the session
//! needs the sandbox: the first tool that runs there (which probes first, exactly once, however many
//! tools ask at the same time) or a question about its jobs. Every test counts the requests the
//! exec endpoint saw, so "it did not wake the box" is a number, not a hope.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::time::Duration;

use common::exec_mock::ExecRequest;
use common::service::Service;
use common::{
    TestWs, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// The home probe every session runs before its first tool: `sh -c 'printf %s "$HOME"'`.
fn is_probe(r: &ExecRequest) -> bool {
    r.command() == "sh" && r.args().iter().any(|a| a.contains(r#"printf %s "$HOME""#))
}

/// Kills, by its unique command line, a sandbox job a test left running.
struct Reap(&'static str);

impl Drop for Reap {
    fn drop(&mut self) {
        let _ = std::process::Command::new("pkill")
            .args(["-KILL", "-f", self.0])
            .status();
    }
}

async fn prompt(ws: &mut TestWs, id: &str, message: &str) -> Vec<Value> {
    ws_send(ws, json!({"type": "prompt", "id": id, "message": message})).await;
    let frames = ws_read_until_response(ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    frames
}

async fn connect(svc: &Service, port: u16, token: &str) -> TestWs {
    ws_connect_with_headers(port, Some("s1.alpha"), &svc.header(token))
        .await
        .unwrap()
}

/// A turn that calls `calls` tools at once, in one assistant message.
fn turn_tool_uses(calls: &[(&str, &str, Value)]) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
    ];
    for (i, (id, name, input)) in calls.iter().enumerate() {
        events.push(json!({ "type": "content_block_start", "index": i, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } }));
        events.push(json!({ "type": "content_block_delta", "index": i, "delta": { "type": "input_json_delta", "partial_json": input.to_string() } }));
        events.push(json!({ "type": "content_block_stop", "index": i }));
    }
    events.push(json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }));
    events.push(json!({ "type": "message_stop" }));
    beyond_ai_test_support::sse(&events)
}

/// The first life of `s1.alpha`: a workspace with a skill and an `AGENTS.md`, one prompt whose
/// turns are `turns`, then the replica is `kill -9`ed. Returns the restarted replica's port and
/// the number of exec requests the first life made.
async fn first_life_then_restart(svc: &mut Service, token: &str) -> (common::service::Peer, usize) {
    let mut ws = connect(svc, svc.port, token).await;
    prompt(&mut ws, "p1", "start").await;
    // The snapshot is written when the loop comes back around after the turn; a second command
    // proves it has.
    ws_send(&mut ws, json!({"type": "get_state", "id": "g0"})).await;
    ws_read_until_response(&mut ws, "get_state").await;
    drop(ws);
    let peer = svc.start_peer(&[]);
    svc.child.kill().unwrap();
    let _ = svc.child.wait();
    (peer, svc.exec.requests().len())
}

fn stock_workspace(svc: &Service) {
    svc.write_in_workspace("AGENTS.md", "LAZY-CONTEXT-MARKER: keep it tidy.");
    svc.write_sandbox_skill(
        ".claude/skills",
        "lazy-skill",
        "LAZY-SKILL-DESCRIPTION",
        "the skill body",
    );
}

/// The headline: reopening a session with a running job and discovered resources, reading its
/// state and leaving makes no exec request at all — not even the probe.
#[tokio::test(flavor = "multi_thread")]
async fn reattaching_and_reading_state_makes_no_exec_request() {
    const JOB: &str = "sleep 3101";
    let _reap = Reap(JOB);
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "bash",
            &json!({"command": JOB, "run_in_background": true}).to_string(),
        ),
        turn_text("started"),
        turn_text("spare"),
        turn_text("spare"),
    ]);
    let mut svc = Service::start(&base, &["s1"]).await;
    stock_workspace(&svc);
    let token = svc.token("t1", "s1.alpha");
    let (peer, before) = first_life_then_restart(&mut svc, &token).await;
    assert!(before > 0, "the first life used its sandbox");

    let mut ws = connect(&svc, peer.port, &token).await;
    ws_send(&mut ws, json!({"type": "get_state", "id": "g1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    ws_send(&mut ws, json!({"type": "get_commands", "id": "c1"})).await;
    let frames = ws_read_until_response(&mut ws, "get_commands").await;
    let commands = serde_json::to_string(&frames.last().unwrap()["data"]).unwrap();
    assert!(
        commands.contains("skill:lazy-skill"),
        "the reopened session still lists its skills: {commands}"
    );
    drop(ws);
    // Anything the attach set off in the background has had time to reach the endpoint.
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let after: Vec<ExecRequest> = svc.exec.requests().split_off(before);
    assert!(
        after.is_empty(),
        "a reattach must not wake the sandbox, but it sent {} request(s): {:#?}",
        after.len(),
        after.iter().map(|r| r.body.clone()).collect::<Vec<_>>()
    );
}

/// The first sandbox tool after a reattach probes first — exactly once — and then runs.
#[tokio::test(flavor = "multi_thread")]
async fn the_first_sandbox_tool_after_a_reattach_probes_once_then_runs() {
    svc_probe_then_tool(false).await;
}

/// Concurrent first calls share one probe.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_first_tools_share_one_probe() {
    svc_probe_then_tool(true).await;
}

async fn svc_probe_then_tool(parallel: bool) {
    let second_turn = if parallel {
        turn_tool_uses(&[
            ("r1", "read", json!({"path": "a.txt"})),
            ("r2", "read", json!({"path": "b.txt"})),
            ("r3", "ls", json!({"path": "."})),
        ])
    } else {
        turn_tool_use("r1", "read", r#"{"path":"a.txt"}"#)
    };
    let (base, requests) = spawn_model_server(vec![
        turn_text("hello"),
        second_turn,
        turn_text("read them"),
    ]);
    let mut svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace("a.txt", "CONTENTS-OF-A");
    svc.write_in_workspace("b.txt", "CONTENTS-OF-B");
    let token = svc.token("t1", "s1.alpha");
    let (peer, before) = first_life_then_restart(&mut svc, &token).await;

    let mut ws = connect(&svc, peer.port, &token).await;
    prompt(&mut ws, "p2", "read the files").await;

    let after: Vec<ExecRequest> = svc.exec.requests().split_off(before);
    let probes = after.iter().filter(|r| is_probe(r)).count();
    assert_eq!(
        probes,
        1,
        "exactly one probe for the first use: {:#?}",
        after.iter().map(|r| r.body.clone()).collect::<Vec<_>>()
    );
    assert!(
        is_probe(&after[0]),
        "the probe runs before any tool's request: {:#?}",
        after.iter().map(|r| r.body.clone()).collect::<Vec<_>>()
    );
    let result = requests.lock().unwrap().last().unwrap().clone();
    assert!(result.contains("CONTENTS-OF-A"), "{result}");
    if parallel {
        assert!(result.contains("CONTENTS-OF-B"), "{result}");
    }
}

/// A probe that fails at first use reaches that tool call with the same words a failed attach
/// used, and nothing runs anywhere else.
#[tokio::test(flavor = "multi_thread")]
async fn a_probe_that_fails_at_first_use_reaches_that_tool_call() {
    let (base, requests) = spawn_model_server(vec![
        turn_text("hello"),
        turn_tool_use("r1", "read", r#"{"path":"a.txt"}"#),
        turn_text("it failed"),
    ]);
    let mut svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace("a.txt", "CONTENTS-OF-A");
    let token = svc.token("t1", "s1.alpha");
    let (peer, before) = first_life_then_restart(&mut svc, &token).await;

    // The sandbox's endpoint is gone by the time the session is reopened.
    let dead = svc.token_with("t1", "s1.alpha", |c| {
        c.exec_url = "http://127.0.0.1:1/exec".into();
    });
    let mut ws = connect(&svc, peer.port, &dead).await;
    prompt(&mut ws, "p2", "read it").await;

    let result = requests.lock().unwrap().last().unwrap().clone();
    assert!(
        result.contains("this session's sandbox is unreachable"),
        "the tool call says why: {result}"
    );
    assert!(!result.contains("CONTENTS-OF-A"), "{result}");
    assert_eq!(
        svc.exec.requests().len(),
        before,
        "the old endpoint saw nothing"
    );
}

/// What a session discovered survives a restart: the reopened session's request carries the same
/// system prompt, byte for byte (so the prompt cache still hits), with no exec request.
#[tokio::test(flavor = "multi_thread")]
async fn discovered_resources_survive_a_restart_without_waking_the_sandbox() {
    let (base, requests) = spawn_model_server(vec![turn_text("one"), turn_text("two")]);
    let mut svc = Service::start(&base, &["s1"]).await;
    stock_workspace(&svc);
    let token = svc.token("t1", "s1.alpha");
    let (peer, before) = first_life_then_restart(&mut svc, &token).await;

    let mut ws = connect(&svc, peer.port, &token).await;
    prompt(&mut ws, "p2", "again").await;
    assert_eq!(
        svc.exec.requests().len(),
        before,
        "a text-only turn on cached resources wakes nothing"
    );

    let bodies = requests.lock().unwrap().clone();
    let system = |raw: &str| common::body_json(raw)["system"].clone();
    let (first, second) = (system(&bodies[0]), system(&bodies[1]));
    let text = serde_json::to_string(&second).unwrap();
    assert!(text.contains("LAZY-CONTEXT-MARKER"), "{text}");
    assert!(text.contains("LAZY-SKILL-DESCRIPTION"), "{text}");
    assert_eq!(
        first, second,
        "the same system prompt before and after the restart"
    );
}

/// A job that ended while the session was closed: the reopen note does not guess — it says the job
/// has not been checked — and the first look at it reconciles with the sandbox and reports how it
/// ended.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_ended_while_detached_is_reported_once_it_is_checked() {
    let (base, requests) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "bash",
            &json!({"command": "sleep 2; echo DONE-WHILE-AWAY", "run_in_background": true})
                .to_string(),
        ),
        turn_text("started"),
        // After the restart:
        turn_text("noted"),
        turn_tool_use("t2", "bash_job", r#"{"job": 1, "timeout_ms": 0}"#),
        turn_text("it finished"),
    ]);
    let mut svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("t1", "s1.alpha");
    let (peer, before) = first_life_then_restart(&mut svc, &token).await;
    let first_life = requests.lock().unwrap().len();
    tokio::time::sleep(Duration::from_secs(3)).await;

    let mut ws = connect(&svc, peer.port, &token).await;
    prompt(&mut ws, "p2", "how is the job?").await;
    let note = requests.lock().unwrap()[first_life].clone();
    assert!(
        note.contains("not yet checked") && note.contains("- job 1: `sleep 2"),
        "the note must not claim a state nobody has checked: {note}"
    );
    assert!(
        !note.contains("(still running)") && !note.contains("(not running)"),
        "{note}"
    );
    assert_eq!(
        svc.exec.requests().len(),
        before,
        "the note checked nothing"
    );

    prompt(&mut ws, "p3", "check it").await;
    let last = requests.lock().unwrap().last().unwrap().clone();
    let from = last
        .find(r#""name":"bash_job""#)
        .expect("the peek was sent");
    let to = last
        .find(r#""tool_use_id":"t2""#)
        .expect("its result was sent");
    let checked = &last[from..to];
    assert!(
        checked.contains("DONE-WHILE-AWAY"),
        "the peek returns the job's final output: {checked}"
    );
    assert!(
        !checked.contains("still running") && !checked.contains("is_error\":true"),
        "a job that ended is reported ended, and successfully: {checked}"
    );
}
