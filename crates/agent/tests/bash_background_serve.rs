//! `serve` e2e: a `run_in_background` job outlives the turn that started it — the start result says it
//! keeps running, so ending the turn earns no reminder — and its exit wakes the idle session. The host `bash` command's `timeout_ms` is
//! the client's own bound, enforced as a cancel.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::ChildStdin;
use std::time::{Duration, Instant};

use common::{
    ChildGuard, SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_agent");

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn close(mut child: ChildGuard, stdin: ChildStdin) {
    drop(stdin);
    let _ = child.wait();
}

#[test]
fn a_background_job_outlives_the_turn_and_its_exit_wakes_the_session() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "bash",
            &json!({ "command": "echo watching; sleep 4; echo bg-done", "run_in_background": true })
                .to_string(),
        ),
        turn_text("started it"),
        // The wake.
        turn_text("noted: it finished"),
    ]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "start it" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert!(
        !frames.iter().any(|f| f["type"] == "bash_jobs_stopped"),
        "the turn's end leaves a background job running: {frames:?}"
    );
    {
        let recorded = bodies.lock().unwrap();
        assert!(
            recorded[1].contains("(it keeps running after your turn ends)"),
            "the start result: {}",
            recorded[1]
        );
        assert_eq!(recorded.len(), 2, "no reminder turn: the job outlives it");
        assert!(
            !recorded.iter().any(|b| b.contains("[background jobs]")),
            "an intended job earns no reminder"
        );
    }

    let woke = read_until_response(&mut stdout, "prompt");
    let event = woke
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "background")
        .expect("a background event identifies the wake");
    assert_eq!(event["event"]["wake"], true);
    assert!(
        event["event"]["text"]
            .as_str()
            .unwrap()
            .starts_with("[job 1 exited with code 0 after")
    );
    let recorded = bodies.lock().unwrap();
    assert_eq!(recorded.len(), 3);
    assert!(
        recorded[2].contains("bg-done"),
        "the notice is the turn's input"
    );
    drop(recorded);
    close(child, stdin);
}

#[test]
fn the_reminder_names_only_the_yielded_job_and_ending_applies_each_kinds_rule() {
    // A background job and a yielded one in the same turn: the reminder is for the one the end
    // stops, and says nothing of the one that keeps running. Reminded once per run, so the next
    // stop ends the prompt — the yielded job stopped with the turn, the background one kept.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "t1",
            "bash",
            &json!({ "command": "echo serving; sleep 96", "run_in_background": true }).to_string(),
        ),
        // A foreground command that outlives its timeout_ms (yielded).
        turn_tool_use(
            "t2",
            "bash",
            &json!({ "command": "echo checking; sleep 97", "timeout_ms": 300 }).to_string(),
        ),
        turn_text("started both"),
        // The reminder's reply.
        turn_text("done"),
    ]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "start it" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true);
    {
        let recorded = bodies.lock().unwrap();
        assert_eq!(
            recorded.len(),
            4,
            "one reminder; the second stop ends the prompt"
        );
        assert_eq!(
            recorded[3]
                .matches("You are ending your turn with jobs")
                .count(),
            1
        );
        let reminder = &recorded[3][recorded[3].find("[background jobs]").unwrap()..];
        assert!(
            reminder.contains("- job 2: `echo checking; sleep 97`")
                && !reminder.contains("- job 1:"),
            "only the yielded job is named: {reminder}"
        );
        assert!(
            reminder.contains("If you end your turn now, job 2 is stopped for you"),
            "{reminder}"
        );
    }
    let stopped = frames
        .iter()
        .find(|f| f["type"] == "bash_jobs_stopped")
        .expect("the yielded job stops with the turn");
    let jobs: Vec<(u64, &str)> = stopped["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| (j["job"].as_u64().unwrap(), j["kind"].as_str().unwrap()))
        .collect();
    assert_eq!(
        jobs,
        vec![(2, "yielded")],
        "the background job keeps running"
    );
    close(child, stdin);
}

#[test]
fn the_host_bash_commands_timeout_ms_cancels_it() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    let started = Instant::now();
    send(
        &mut stdin,
        json!({ "type": "bash", "command": "echo host-started; sleep 60", "timeout_ms": 1000 }),
    );
    let frames = read_until_response(&mut stdout, "bash");
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "cancelled at the deadline, not at the sleep's end: {:?}",
        started.elapsed()
    );
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["is_error"], true, "{data}");
    assert_eq!(data["cancelled"], true, "{data}");
    let result = data["result"].as_str().unwrap();
    assert!(result.contains("host-started"), "{result}");
    assert!(
        result.ends_with("Command timed out after 1 second"),
        "{result}"
    );
    close(child, stdin);
}

#[test]
fn the_host_bash_timeout_message_keeps_a_fractional_second() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    let started = Instant::now();
    send(
        &mut stdin,
        json!({ "type": "bash", "command": "echo host-started; sleep 60", "timeout_ms": 1500 }),
    );
    let frames = read_until_response(&mut stdout, "bash");
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "cancelled at the deadline, not at the sleep's end: {:?}",
        started.elapsed()
    );
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["is_error"], true, "{data}");
    assert_eq!(data["cancelled"], true, "{data}");
    let result = data["result"].as_str().unwrap();
    assert!(result.contains("host-started"), "{result}");
    assert!(
        result.ends_with("Command timed out after 1.5 seconds"),
        "{result}"
    );
    close(child, stdin);
}
