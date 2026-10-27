//! `serve` e2e: reopening a session whose last process was killed mid-turn. The prompt is
//! checkpointed before the model call, so after a SIGKILL the session ends on that unanswered `user`
//! message. A reopened `serve` (a restarted TUI, a daemon, a service replica) used to leave it open:
//! the next prompt stacked a second `user` turn and the model was never told the first was cut off.
//! Opening the session — at start, or by `switch_session` — now closes it, as `run` does.
#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::path::Path;
use std::process::ChildStdin;
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, read_until_response, serve_cmd, serve_dir_cmd, spawn_model_server,
    spawn_model_server_with_stalled_response, turn_text,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_agent");

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn on_disk(path: &Path, needle: &str) -> bool {
    if path.is_dir() {
        return std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| on_disk(&e.path(), needle));
    }
    std::fs::read_to_string(path).is_ok_and(|s| s.contains(needle))
}

/// Start a prompt against a model that never answers, wait until it is on disk, and SIGKILL `serve`.
fn kill_mid_turn(mut cmd: std::process::Command, store: &Path, prompt: &str) {
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    send(&mut stdin, json!({ "type": "prompt", "message": prompt }));
    let t0 = Instant::now();
    while !on_disk(store, prompt) {
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "the prompt was never checkpointed"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(300));
    child.kill().unwrap(); // SIGKILL: nothing gets to close the turn
    let _ = child.wait();
}

fn roles_of(raw: &str) -> (Vec<String>, Vec<Value>) {
    let body = common::body_json(raw);
    let messages = body["messages"].as_array().unwrap().clone();
    let roles = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap_or("?").to_string())
        .collect();
    (roles, messages)
}

#[test]
fn a_session_reopened_after_a_sigkill_closes_the_prompt_it_cut_off() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("s.jsonl");
    let file_str = file.to_string_lossy().into_owned();
    let stalled = spawn_model_server_with_stalled_response(vec![], Duration::from_secs(60), vec![]);
    kill_mid_turn(serve_cmd(BIN, &stalled, &file_str), &file, "FIRST-PROMPT");

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let mut child = serve_cmd(BIN, &base, &file_str).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = common::child_frames(&mut child);
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "SECOND-PROMPT" }),
    );
    let resp = read_until_response(&mut frames, "prompt");
    assert_eq!(resp.last().unwrap()["success"], true, "{resp:#?}");

    let (roles, messages) = roles_of(&bodies.lock().unwrap()[0]);
    assert_eq!(roles, ["user", "assistant", "user"], "{messages:#?}");
    assert!(messages[0].to_string().contains("FIRST-PROMPT"));
    assert!(messages[2].to_string().contains("SECOND-PROMPT"));
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn switching_to_a_session_a_sigkill_cut_off_closes_its_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions");
    let sessions_str = sessions.to_string_lossy().into_owned();
    let stalled = spawn_model_server_with_stalled_response(vec![], Duration::from_secs(60), vec![]);
    let mut cut = serve_dir_cmd(BIN, &stalled, &sessions_str);
    cut.args(["--session-id", "cut-off"]);
    kill_mid_turn(cut, &sessions, "FIRST-PROMPT");

    // A fresh session, then a switch to the one that was cut off.
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let mut child = serve_dir_cmd(BIN, &base, &sessions_str).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = common::child_frames(&mut child);
    send(
        &mut stdin,
        json!({ "type": "switch_session", "session_id": "cut-off" }),
    );
    let resp = read_until_response(&mut frames, "switch_session");
    assert_eq!(resp.last().unwrap()["success"], true, "{resp:#?}");
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "SECOND-PROMPT" }),
    );
    let resp = read_until_response(&mut frames, "prompt");
    assert_eq!(resp.last().unwrap()["success"], true, "{resp:#?}");

    let (roles, messages) = roles_of(&bodies.lock().unwrap()[0]);
    assert_eq!(roles, ["user", "assistant", "user"], "{messages:#?}");
    assert!(messages[0].to_string().contains("FIRST-PROMPT"));
    let _ = child.kill();
    let _ = child.wait();
}
