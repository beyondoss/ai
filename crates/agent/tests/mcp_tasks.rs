//! SEP-2663 MCP tasks e2e: fixture returns `resultType: "task"` for `slow_task`; the agent polls
//! `tasks/get`, surfaces status messages as `tool_progress`, and completes with `task-done`.
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
fn serve_polls_mcp_task_to_completion_with_progress() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(
        &home,
        json!([{
            "name": "jobs",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {},
        }]),
    );

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_t", "mcp__jobs__slow_task", &json!({}).to_string()),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run slow_task" })
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
        "SEP-2663 task statusMessage must emit tool_progress; kinds={kinds:?} frames={frames:?}"
    );

    let snapshots: String = progress
        .iter()
        .filter_map(|f| f["event"]["snapshot"].as_str())
        .collect::<Vec<_>>()
        .join("|");
    assert!(
        snapshots.contains("task-phase-one") || snapshots.contains("task-phase-two"),
        "progress must carry fixture task phases, got: {snapshots:?}"
    );

    let details_ok = progress.iter().any(|f| {
        f["event"]["details"]["taskId"].as_str().is_some()
            && f["event"]["details"]["statusMessage"].as_str().is_some()
    });
    assert!(
        details_ok,
        "tool_progress details must carry taskId/statusMessage: {progress:?}"
    );

    let first_progress = kinds.iter().position(|k| *k == "tool_progress").unwrap();
    let tool_end = kinds
        .iter()
        .position(|k| *k == "tool_end")
        .expect("tool_end must appear");
    assert!(
        first_progress < tool_end,
        "task progress must arrive before tool_end: {kinds:?}"
    );

    let tool_end_frame = frames
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .unwrap();
    assert!(
        tool_end_frame["event"]["result"]
            .as_str()
            .unwrap_or("")
            .contains("task-done"),
        "final result must be task-done: {tool_end_frame}"
    );
}
