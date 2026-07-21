//! Integration tests for the supervisor-level `list_daemon_sessions` command (`serve --listen`). Only
//! the daemon supervisor sees every session, so this command is answered *there* — it unions the live
//! in-memory session map with an on-disk scan of `<session-dir>/*.jsonl`, tagging each entry with a
//! `live` flag. These spawn the real `beyond-ai-agent` binary against the mock Anthropic-SSE gateway
//! and drive it with a real `tokio-tungstenite` client.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, free_port, spawn_model_server, turn_text,
    wait_for_port, ws_connect, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Spawn a `serve --listen 127.0.0.1:<port>` child against `base` (the mock gateway), persisting
/// per-session files under `session_dir`.
fn serve_ws_child(base: &str, session_dir: &str, port: u16) -> ChildGuard {
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
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

/// Prompt a fresh session `id` to completion so it persists under the daemon's `--session-dir`, then
/// drop the connection (the session itself keeps living in the supervisor's map).
async fn prompt_session(port: u16, id: &str) {
    let mut ws = ws_connect(port, Some(id)).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "message": "hi" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "prompt for {id} should succeed: {:?}",
        frames.last()
    );
    drop(ws);
}

/// Send `list_daemon_sessions` and return the `data.sessions` array.
async fn list_daemon_sessions(port: u16) -> Vec<Value> {
    let mut ws = ws_connect(port, None).await;
    ws_send(
        &mut ws,
        json!({ "type": "list_daemon_sessions", "id": "L1" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "list_daemon_sessions").await;
    let resp = frames.last().expect("a list_daemon_sessions response");
    assert_eq!(resp["type"], "response");
    assert_eq!(resp["command"], "list_daemon_sessions");
    assert_eq!(resp["id"], "L1", "the client id is echoed back: {resp}");
    assert_eq!(resp["success"], true, "should succeed: {resp}");
    resp["data"]["sessions"]
        .as_array()
        .expect("data.sessions is an array")
        .clone()
}

/// Find the listing entry for session `id` in a `list_daemon_sessions` result.
fn entry_for<'a>(sessions: &'a [Value], id: &str) -> &'a Value {
    sessions
        .iter()
        .find(|s| s["id"] == id)
        .unwrap_or_else(|| panic!("session {id} not listed: {sessions:#?}"))
}

/// (i) Two persisted sessions, both live: `list_daemon_sessions` lists both with `live:true`, and each
/// entry carries the `to_listing_json` fields (e.g. `message_count`) alongside the `live` flag.
#[tokio::test]
async fn list_daemon_sessions_reports_live_sessions_with_listing_fields() {
    // A turn apiece for the two sessions.
    let (base, _requests) = spawn_model_server(vec![turn_text("one"), turn_text("two")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    prompt_session(port, "aaaaaaaaaaaaaaaa").await;
    prompt_session(port, "bbbbbbbbbbbbbbbb").await;

    let sessions = list_daemon_sessions(port).await;

    for id in ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"] {
        let e = entry_for(&sessions, id);
        assert_eq!(e["live"], true, "session {id} should be live: {e}");
        // Carries the `SessionMeta::to_listing_json` shape, not just `{id, live}`.
        assert!(
            e.get("message_count").is_some(),
            "entry should carry the listing's message_count: {e}"
        );
        assert!(
            e["message_count"].as_u64().unwrap() >= 2,
            "a completed turn commits >=2 messages (user + assistant): {e}"
        );
        assert!(
            e.get("updated_at").is_some(),
            "entry should carry updated_at: {e}"
        );
    }

    let _ = child.kill();
    let _ = child.wait();
}

/// (ii) Cold restart: after the daemon is killed and restarted on the *same* `--session-dir`, the two
/// ids are still listed (from disk) but now `live:false` — nothing is running yet in the fresh process.
#[tokio::test]
async fn list_daemon_sessions_reports_persisted_sessions_as_not_live_after_restart() {
    let (base, _requests) = spawn_model_server(vec![turn_text("one"), turn_text("two")]);
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap().to_string();
    let port = free_port();

    let mut child = serve_ws_child(&base, &dir_str, port);
    wait_for_port(port);
    prompt_session(port, "aaaaaaaaaaaaaaaa").await;
    prompt_session(port, "bbbbbbbbbbbbbbbb").await;

    // While the daemon still runs, both are live.
    let live = list_daemon_sessions(port).await;
    assert_eq!(entry_for(&live, "aaaaaaaaaaaaaaaa")["live"], true);
    assert_eq!(entry_for(&live, "bbbbbbbbbbbbbbbb")["live"], true);

    // Kill the daemon and cold-restart a brand-new process on the same session dir + a new port.
    let _ = child.kill();
    let _ = child.wait();

    let port2 = free_port();
    let mut child2 = serve_ws_child(&base, &dir_str, port2);
    wait_for_port(port2);

    let cold = list_daemon_sessions(port2).await;
    for id in ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"] {
        let e = entry_for(&cold, id);
        assert_eq!(
            e["live"], false,
            "after a cold restart, {id} is on disk but not running: {e}"
        );
        assert!(
            e["message_count"].as_u64().unwrap() >= 2,
            "the persisted listing fields survive the restart: {e}"
        );
    }

    let _ = child2.kill();
    let _ = child2.wait();
}
