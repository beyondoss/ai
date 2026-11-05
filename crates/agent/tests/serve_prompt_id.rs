//! `serve` e2e: a prompt's idempotency key makes sending it again safe.
//!
//! The control plane delivers a prompt with the run's id as its idempotency key. A hop that times out after
//! the replica read the command leaves the sender unable to tell whether the turn ran, so it must be able
//! to send the same prompt again without a second turn: a repeat of a taken id is answered as taken (and,
//! while its turn runs, as already running), across a replica restart too.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::{Command, Stdio};
use std::time::Duration;

use common::{
    ChildGuard, ISOLATED_HOME, spawn_listening, spawn_model_server, turn_text, turn_tool_use,
    ws_connect, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

fn serve_ws(base: &str, session_dir: &str) -> (ChildGuard, u16) {
    spawn_listening(
        crate::common::agent_cmd()
            .args([
                "serve",
                "--gateway-url",
                base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                session_dir,
                "--session-idle-timeout",
                "0",
            ])
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )
}

fn terminal(frames: &[Value]) -> &Value {
    frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("a terminal prompt response")
}

fn model_calls(bodies: &std::sync::Mutex<Vec<String>>) -> usize {
    bodies.lock().unwrap().len()
}

#[tokio::test]
async fn a_prompt_sent_again_after_it_ran_is_taken_not_run() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("first"), turn_text("second")]);
    let (_child, port) = serve_ws(&base, dir.path().to_str().unwrap());
    let mut ws = ws_connect(port, Some("pid-1")).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "idempotency_key": "run-1", "message": "hello" }),
    )
    .await;
    let first = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(terminal(&first)["success"], true);
    let calls = model_calls(&bodies);
    assert!(calls >= 1);

    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "idempotency_key": "run-1", "message": "hello" }),
    )
    .await;
    let again = ws_read_until_response(&mut ws, "prompt").await;
    let t = terminal(&again);
    assert_eq!(t["success"], true, "{t}");
    assert_eq!(t["data"]["duplicate"], true, "{t}");
    assert_eq!(
        model_calls(&bodies),
        calls,
        "a repeat of a taken prompt ran a turn"
    );

    // A different id is a different prompt.
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "idempotency_key": "run-2", "message": "again" }),
    )
    .await;
    let other = ws_read_until_response(&mut ws, "prompt").await;
    assert!(terminal(&other)["data"]["duplicate"].is_null());
    assert!(model_calls(&bodies) > calls);
}

#[tokio::test]
async fn a_prompt_sent_again_after_a_restart_is_taken_not_run() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("first"), turn_text("second")]);
    let calls;
    {
        let (_child, port) = serve_ws(&base, dir.path().to_str().unwrap());
        let mut ws = ws_connect(port, Some("pid-2")).await;
        ws_send(
            &mut ws,
            json!({ "type": "prompt", "id": "p", "idempotency_key": "run-1", "message": "hello" }),
        )
        .await;
        ws_read_until_response(&mut ws, "prompt").await;
        calls = model_calls(&bodies);
    } // the replica is gone: the session survives on disk
    let (_child, port) = serve_ws(&base, dir.path().to_str().unwrap());
    let mut ws = ws_connect(port, Some("pid-2")).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "idempotency_key": "run-1", "message": "hello" }),
    )
    .await;
    let again = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(terminal(&again)["data"]["duplicate"], true);
    assert_eq!(
        model_calls(&bodies),
        calls,
        "the takeover ran the turn again"
    );
}

#[tokio::test]
async fn a_prompt_sent_again_while_its_turn_runs_is_already_running() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_1",
            "bash",
            &json!({ "command": "printf started; sleep 3" }).to_string(),
        ),
        turn_text("done"),
    ]);
    let (_child, port) = serve_ws(&base, dir.path().to_str().unwrap());
    let mut ws = ws_connect(port, Some("pid-3")).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "idempotency_key": "run-1", "message": "go" }),
    )
    .await;
    // The turn is in flight once the model has been asked.
    for _ in 0..100 {
        if model_calls(&bodies) >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "idempotency_key": "run-1", "message": "go" }),
    )
    .await;
    // Its answer comes among the running turn's own frames.
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    let dup = frames
        .iter()
        .find(|f| f["type"] == "response" && f["data"]["already_running"] == true)
        .unwrap_or_else(|| panic!("no already-running answer in {frames:?}"));
    assert_eq!(dup["success"], true);
    let _ = Command::new("true");
}
