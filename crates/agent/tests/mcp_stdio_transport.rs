//! The stdio MCP transport (`tools::mcp_stdio`) — used for **every** stdio server, not only ones
//! with MCP Events — keeps rmcp's behaviour where it matters: a stdout line that is not UTF-8 is
//! skipped rather than fatal, and a server whose connection is dropped (here: reaped idle) gets
//! its grace window after stdin closes before it is killed. Real `serve`, real fixture process.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_events_fixture::{Frames, eventually, send, stdio_server, write_settings};
use common::{BIN, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::json;

fn start(
    home: &std::path::Path,
    extra_env: serde_json::Value,
    idle_secs: &str,
) -> (common::ChildGuard, std::process::ChildStdin, Frames) {
    write_settings(
        home,
        json!([stdio_server(
            "tools",
            &home.join("control"),
            extra_env,
            json!([])
        )]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let mut cmd = serve_cmd(BIN, &base, &home.join("s.jsonl").to_string_lossy());
    cmd.env("HOME", home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", idle_secs);
    let mut child = cmd.spawn_guarded();
    let stdin = child.stdin.take().unwrap();
    let frames = Frames::new(child.stdout.take().unwrap(), None);
    (child, stdin, frames)
}

#[test]
fn a_non_utf8_stdout_line_is_skipped_not_fatal() {
    let home = tempfile::tempdir().unwrap();
    let (_child, mut stdin, mut frames) = start(
        home.path(),
        json!({ "MCP_FIXTURE_GARBAGE_STDOUT": "1" }),
        "0",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    let m = frames.response("m");
    assert!(
        m["data"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "mcp__tools__echo"),
        "the server survived its garbage line and its tools are there: {m:#}"
    );
}

#[test]
fn a_reaped_server_gets_its_grace_window_after_stdin_closes() {
    let home = tempfile::tempdir().unwrap();
    let marker = home.path().join("exited-cleanly");
    let (_child, mut stdin, mut frames) = start(
        home.path(),
        json!({ "MCP_FIXTURE_EXIT_MARKER": marker.to_string_lossy() }),
        "1",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    frames.response("m");
    // Idle for a second → reaped: stdin closes, the server takes 500 ms to "clean up" and writes
    // the marker. Killed on the spot, it never would.
    eventually(Duration::from_secs(15), "the server's clean exit", || {
        marker.exists().then_some(())
    });
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "clean exit");
}
