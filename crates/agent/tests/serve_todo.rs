//! `serve` e2e: `get_todos` and the durability of the model's plan.
//!
//! The `todo` tool is stateless and its `tool_progress` frames are ephemeral, so "what is the plan
//! right now?" has to be answerable from somewhere else. Three somewheres, each exercised here against
//! the real binary: the live transcript (idle), a live mirror (mid-run, while `&session` is exclusively
//! borrowed), and `Session::compaction` (after a compaction dropped the `tool_use` block that carried
//! it — including across a process restart).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::ChildStdin;
use std::time::Duration;

use common::{
    ChildGuard, SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn item(content: &str, status: &str) -> Value {
    json!({ "content": content, "activeForm": format!("{content}ing"), "status": status })
}

fn todo_turn(id: &str, todos: &Value) -> String {
    turn_tool_use(id, "todo", &json!({ "todos": todos }).to_string())
}

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

/// Issue `get_todos` and return `data.todos` (`Value::Null` when the model never made a list).
fn get_todos(stdin: &mut ChildStdin, stdout: &mut impl BufRead) -> Value {
    send(stdin, json!({ "type": "get_todos", "id": "gt" }));
    let frames = read_until_response(stdout, "get_todos");
    let resp = frames.last().expect("a response frame");
    assert_eq!(resp["success"], true, "get_todos failed: {resp}");
    resp["data"]["todos"].clone()
}

fn kill(mut child: ChildGuard) {
    let _ = child.kill();
    let _ = child.wait();
}

/// A plan the model committed, and the checklist it renders as.
fn plan() -> Value {
    json!([
        item("Wire the retry loop", "in_progress"),
        item("Add tests", "pending"),
    ])
}

#[test]
fn get_todos_is_null_before_the_model_has_made_a_plan() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    assert_eq!(get_todos(&mut stdin, &mut stdout), Value::Null);
    kill(child);
}

#[test]
fn get_todos_reads_the_last_call_out_of_the_live_transcript_when_idle() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // The model plans, revises, then stops. Full-replace semantics: only the second list survives.
    let first = json!([item("Old plan", "pending")]);
    let (base, _bodies) = spawn_model_server(vec![
        todo_turn("toolu_1", &first),
        todo_turn("toolu_2", &plan()),
        turn_text("planned"),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "plan it" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true);

    assert_eq!(
        get_todos(&mut stdin, &mut stdout),
        plan(),
        "the newest call wins; the superseded one must not resurface"
    );
    kill(child);
}

#[test]
fn get_todos_answers_from_the_live_mirror_while_a_run_is_still_in_flight() {
    // The client this exists for: a phone attaching mid-run, which has already missed every
    // `tool_progress` frame streamed before it connected. `&session` is exclusively borrowed by the
    // in-flight turn, so this can only be answered from `LiveStats`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        todo_turn("toolu_1", &plan()),
        // Keeps the run in flight long enough to poll underneath it.
        turn_tool_use(
            "toolu_2",
            "bash",
            &json!({ "command": "sleep 2" }).to_string(),
        ),
        turn_text("done"),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "plan then work" }),
    );
    // Long enough for the `todo` call to land and the `sleep 2` to start.
    std::thread::sleep(Duration::from_millis(600));

    send(&mut stdin, json!({ "type": "get_todos", "id": "gt" }));
    let frames = read_until_response(&mut stdout, "get_todos");

    // The run must still have been in flight, or this test proves nothing about the mirror: the
    // `prompt` command's own terminal response cannot have arrived before `get_todos`'s did.
    assert!(
        !frames
            .iter()
            .any(|f| f["type"] == "response" && f["command"] == "prompt"),
        "the run finished before get_todos was answered — the busy path was never exercised"
    );

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "get_todos failed: {resp}");
    assert_eq!(
        resp["data"]["todos"],
        plan(),
        "a mid-run get_todos must be served from the live mirror, not rejected as busy"
    );
    kill(child);
}

#[test]
fn a_compaction_carries_the_plan_into_the_summary_and_get_todos_still_finds_it() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        todo_turn("toolu_1", &plan()),
        turn_text("planned, now working"),
        // The summarization call `compact` makes. It never mentions the plan — exactly as a real
        // summarizing model usually wouldn't.
        turn_text("The model planned some work."),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_file)
        // Force an aggressive cut so a short test session actually compacts.
        .args(["--compaction-keep-recent-tokens", "1"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "plan it" }),
    );
    assert_eq!(
        read_until_response(&mut stdout, "prompt").last().unwrap()["success"],
        true
    );

    send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
    let frames = read_until_response(&mut stdout, "compact");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "compact failed: {resp}");
    assert_eq!(
        resp["data"]["compacted"], true,
        "the session must actually compact for this test to mean anything: {resp}"
    );

    // The `tool_use` block that carried the plan is gone from history — `apply_summary` dropped it.
    // Two things must have caught it on the way out.
    // 1. The summary the model will now read carries the plan verbatim.
    let raw = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        raw.contains("<todo_list>"),
        "the summary must carry a deterministic todo block"
    );
    assert!(
        raw.contains("[>] Wire the retry loop"),
        "the summary must carry the plan itself"
    );
    // 2. `get_todos` still answers structurally, now out of `Session::compaction`.
    assert_eq!(get_todos(&mut stdin, &mut stdout), plan());

    kill(child);
}

#[test]
fn the_plan_survives_a_serve_restart_past_a_compaction() {
    // `Session::compaction.todos` is the only surviving copy once the `tool_use` block is folded away.
    // If it didn't round-trip through the session file, every restart would silently lose the plan.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    {
        let (base, _bodies) = spawn_model_server(vec![
            todo_turn("toolu_1", &plan()),
            turn_text("planned"),
            turn_text("A summary with no mention of the plan."),
        ]);
        let mut child = serve_cmd(BIN, &base, &session_file)
            .args(["--compaction-keep-recent-tokens", "1"])
            .spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        send(
            &mut stdin,
            json!({ "type": "prompt", "message": "plan it" }),
        );
        read_until_response(&mut stdout, "prompt");
        send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
        let frames = read_until_response(&mut stdout, "compact");
        assert_eq!(frames.last().unwrap()["data"]["compacted"], true);
        kill(child);
    }

    // A brand-new process, resuming the same session file. Nothing is read from memory.
    let (base2, _b2) = spawn_model_server(vec![]);
    let mut child = serve_cmd(BIN, &base2, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    assert_eq!(
        get_todos(&mut stdin, &mut stdout),
        plan(),
        "the plan must be restored from the persisted compaction record"
    );
    kill(child);
}
