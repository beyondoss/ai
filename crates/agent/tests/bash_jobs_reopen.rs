//! e2e: a reopened session's `bash` jobs. The jobs died with the process that ran them, but job
//! numbers used to restart at 1 and nothing told the model — it took a dead server for live, and a
//! new job got id 1 again, so a kill aimed at the old one hit the new one. Now the first request
//! after a reopen lists the jobs the transcript left running as stopped (request-only: never
//! persisted), and the next job is numbered past every id the transcript used. Through a `serve`
//! restart and `run --continue`.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::process::{ChildStdin, Stdio};

use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");
const NOTE: &str = "are no longer running";

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn background(id: &str, command: &str) -> String {
    let input = json!({ "command": command, "run_in_background": true });
    turn_tool_use(id, "bash", &input.to_string())
}

/// One `serve` process on `session_file`: send `prompt`, wait for its response, close stdin (the
/// session's end stops its jobs), and return the model request bodies.
fn serve_once(session_file: &str, turns: Vec<String>, prompt: &str) -> Vec<String> {
    let (base, bodies) = spawn_model_server(turns);
    let mut child = serve_cmd(BIN, &base, session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    send(&mut stdin, json!({ "type": "prompt", "message": prompt }));
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    drop(stdin);
    let _ = child.wait();
    bodies.lock().unwrap().clone()
}

#[test]
fn a_restarted_serve_session_is_told_its_jobs_stopped_and_numbers_past_them() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    serve_once(
        &session_file,
        vec![
            background("t1", "echo up; sleep 90"),
            turn_text("the server is up"),
        ],
        "start the server",
    );

    let bodies = serve_once(
        &session_file,
        vec![
            background("t2", "echo again; sleep 91"),
            turn_text("restarted it"),
            turn_text("ok"),
        ],
        "is the server still up?",
    );
    assert!(
        bodies[0].contains("- job 1: `echo up; sleep 90` (not running)"),
        "the first request after the restart says job 1 stopped: {}",
        bodies[0]
    );
    assert!(bodies[0].contains(NOTE), "{}", bodies[0]);
    assert!(
        bodies[1].contains("Running in the background as job 2 "),
        "the new job does not reuse id 1: {}",
        bodies[1]
    );
    // Request-only: the transcript on disk never holds the note.
    let on_disk = std::fs::read_to_string(&session_file).unwrap();
    assert!(!on_disk.contains(NOTE), "the note was persisted");
}

#[test]
fn run_continue_is_told_the_last_runs_jobs_stopped() {
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions");
    let run = |base: &str, extra: &[&str]| {
        let out = run_cmd(BIN)
            .args([
                "run",
                "--gateway-url",
                base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
            ])
            .arg(&sessions)
            .args(extra)
            .current_dir(dir.path())
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let (base, _) = spawn_model_server(vec![
        background("t1", "echo up; sleep 90"),
        turn_text("the server is up"),
        turn_text("ok"),
    ]);
    run(&base, &["start the server"]);

    let (base, bodies) = spawn_model_server(vec![turn_text("it stopped")]);
    run(&base, &["is the server still up?", "--continue"]);
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("- job 1: `echo up; sleep 90` (not running)"),
        "{}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("New jobs are numbered from 2."),
        "{}",
        bodies[0]
    );
}
