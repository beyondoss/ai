//! MCP progress → `ToolProgress` e2e: a real MCP stdio fixture emits `notifications/progress`, and
//! the agent must surface those as `tool_progress` event frames *before* `tool_end`.
//!
//! This is the certainty test for the progress gap vs Claude Code / Cursor / the MCP spec: if progress
//! notifications are dropped on the floor, this fails.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::path::Path;

use common::{
    SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

fn fixture_bin() -> &'static str {
    env!("CARGO_BIN_EXE_mcp_fixture_stdio_server")
}

fn write_global_settings(home: &Path, mcp_servers: Value) {
    let claude_dir = home.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": mcp_servers })).unwrap(),
    )
    .unwrap();
}

#[test]
fn serve_streams_mcp_progress_before_tool_end() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(
        &home,
        json!([{
            "name": "prog",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {},
        }]),
    );

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_p",
            "mcp__prog__slow_progress",
            &json!({}).to_string(),
        ),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    // Don't idle-reap mid-test.
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run slow_progress" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let kinds: Vec<&str> = frames
        .iter()
        .filter(|f| f["type"] == "event")
        .filter_map(|f| f["event"]["kind"].as_str())
        .collect();

    let progress: Vec<&Value> = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_progress")
        .collect();

    assert!(
        !progress.is_empty(),
        "MCP slow_progress must emit tool_progress frames; kinds={kinds:?} frames={frames:?}"
    );

    let snapshots: String = progress
        .iter()
        .filter_map(|f| f["event"]["snapshot"].as_str())
        .collect::<Vec<_>>()
        .join("|");
    assert!(
        snapshots.contains("phase-one") && snapshots.contains("phase-two"),
        "progress snapshots must carry the fixture's phase messages, got: {snapshots:?}"
    );

    // Details must include the numeric progress fields from notifications/progress.
    let details_ok = progress.iter().any(|f| {
        f["event"]["details"]["progress"].as_f64().is_some()
            && f["event"]["details"]["message"].as_str().is_some()
    });
    assert!(
        details_ok,
        "tool_progress details must carry progress/message from MCP: {progress:?}"
    );

    let first_progress = kinds.iter().position(|k| *k == "tool_progress").unwrap();
    let tool_end = kinds
        .iter()
        .position(|k| *k == "tool_end")
        .expect("tool_end must appear");
    assert!(
        first_progress < tool_end,
        "MCP progress must arrive before tool_end: {kinds:?}"
    );

    // Final tool result text.
    let tool_end_frame = frames
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .unwrap();
    assert!(
        tool_end_frame["event"]["result"]
            .as_str()
            .unwrap_or("")
            .contains("progress-done"),
        "final result must be progress-done: {tool_end_frame}"
    );
}
