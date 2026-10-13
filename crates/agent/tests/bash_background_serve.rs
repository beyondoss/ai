//! `serve` e2e: a `run_in_background` job outlives the turn that started it — the reminder says it
//! keeps running — and its exit wakes the idle session. The host `bash` command's `timeout_ms` is
//! the client's own bound, enforced as a cancel.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::process::ChildStdin;
use std::time::{Duration, Instant};

use common::{
    ChildGuard, SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

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
        // The one stop reminder.
        turn_text("ok, it can run"),
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
        assert!(
            recorded[2]
                .contains("If you end your turn now: job 1 keeps running, and you will be told"),
            "the serve reminder: {}",
            recorded[2]
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
    assert_eq!(recorded.len(), 4);
    assert!(
        recorded[3].contains("bg-done"),
        "the notice is the turn's input"
    );
    drop(recorded);
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
        result.ends_with("Command timed out after 1 seconds"),
        "{result}"
    );
    close(child, stdin);
}
