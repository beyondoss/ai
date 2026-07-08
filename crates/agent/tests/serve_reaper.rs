//! Integration tests for the idle-session reaper (`serve --listen --session-idle-timeout <secs>`). The
//! supervisor reaps a session that has had no attached connection for longer than the timeout and isn't
//! mid-run — dropping its retained `input_tx` so it persists and exits, exactly like a graceful
//! shutdown does per-session. A reconnect to a reaped id transparently respawns and replays from disk.
//! These spawn the real `beyond-ai-agent` binary against the mock Anthropic-SSE gateway.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{
    ISOLATED_HOME, free_port, spawn_model_server, turn_text, wait_for_port, ws_connect,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Spawn a `serve --listen` child with the idle reaper armed at `idle_secs`.
fn serve_reaper_child(base: &str, session_dir: &str, port: u16, idle_secs: u64) -> Child {
    Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir,
            "--session-idle-timeout",
            &idle_secs.to_string(),
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn serve --listen --session-idle-timeout")
}

/// Send `list_daemon_sessions` on a fresh connection and return the `data.sessions` array.
async fn list_daemon_sessions(port: u16) -> Vec<Value> {
    let mut ws = ws_connect(port, None).await;
    ws_send(&mut ws, json!({ "type": "list_daemon_sessions" })).await;
    let frames = ws_read_until_response(&mut ws, "list_daemon_sessions").await;
    frames.last().expect("a response")["data"]["sessions"]
        .as_array()
        .expect("data.sessions array")
        .clone()
}

/// (i) A detached, idle, not-mid-run session is reaped after the timeout — yet the committed turn
/// survives on disk: reconnecting respawns it and `get_messages{since:0}` still returns the reply
/// (persist-on-reap). We prove the reap by observing `live:false` (the id is on disk but no longer has
/// a running task in the map).
#[tokio::test]
async fn idle_session_is_reaped_but_its_turn_persists() {
    const SID: &str = "reapedsession01";
    // Turn 1 for the initial run; turn 2 for the post-reconnect respawn (which replays disk, but a
    // spare turn keeps the mock from starving if anything prompts again).
    let (base, _requests) =
        spawn_model_server(vec![turn_text("committedreply"), turn_text("again")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_reaper_child(&base, dir.path().to_str().unwrap(), port, 1);
    wait_for_port(port);

    // Connect, prompt to completion (persists), then disconnect.
    {
        let mut ws = ws_connect(port, Some(SID)).await;
        ws_send(
            &mut ws,
            json!({ "type": "prompt", "id": "p", "message": "hi" }),
        )
        .await;
        let frames = ws_read_until_response(&mut ws, "prompt").await;
        assert_eq!(frames.last().unwrap()["success"], true);
        drop(ws);
    }

    // Wait comfortably past the timeout (1s) + a reaper tick (~500ms).
    tokio::time::sleep(Duration::from_secs(3)).await;

    // The session was reaped: it's on disk (persisted) but no longer running (`live:false`).
    let sessions = list_daemon_sessions(port).await;
    let entry = sessions
        .iter()
        .find(|s| s["id"] == SID)
        .unwrap_or_else(|| panic!("{SID} should still be listed from disk: {sessions:#?}"));
    assert_eq!(
        entry["live"], false,
        "the idle session must have been reaped (no running task): {entry}"
    );

    // Reconnect to the reaped id: it respawns from disk, and the committed turn replays via
    // `get_messages` (the full transcript from the persisted file — `since` is a message id, so
    // omitting it returns everything from the beginning).
    let mut ws = ws_connect(port, Some(SID)).await;
    ws_send(&mut ws, json!({ "type": "get_messages" })).await;
    let frames = ws_read_until_response(&mut ws, "get_messages").await;
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("committedreply"),
        "the reaped session's committed turn must survive on disk: {dump}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// (ii) Negative: a session that is *still attached* past the timeout is never reaped — the reaper only
/// reclaims detached sessions. The connection is held open across the whole wait; the id stays `live`.
#[tokio::test]
async fn attached_session_past_timeout_is_not_reaped() {
    const SID: &str = "attachedsession1";
    let (base, _requests) = spawn_model_server(vec![turn_text("stillhere")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_reaper_child(&base, dir.path().to_str().unwrap(), port, 1);
    wait_for_port(port);

    // Attach and prompt to completion, but KEEP the connection open.
    let mut ws = ws_connect(port, Some(SID)).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "message": "hi" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true);

    // Wait well past the timeout while still attached.
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Still live — an attached session is never reaped.
    let sessions = list_daemon_sessions(port).await;
    let entry = sessions
        .iter()
        .find(|s| s["id"] == SID)
        .unwrap_or_else(|| panic!("{SID} should still be present: {sessions:#?}"));
    assert_eq!(
        entry["live"], true,
        "an attached session must not be reaped even past the idle timeout: {entry}"
    );

    // The still-attached connection is healthy: a command round-trips.
    ws_send(&mut ws, json!({ "type": "get_state" })).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true);

    let _ = child.kill();
    let _ = child.wait();
}
