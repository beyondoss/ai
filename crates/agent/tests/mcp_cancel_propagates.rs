//! Aborting a turn mid-`tools/call` must tell the MCP server (`notifications/cancelled` with the
//! call's request id), so a slow tool stops instead of running to completion unobserved.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::time::{Duration, Instant};

use common::{SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_tool_use};
use serde_json::{Value, json};

#[test]
fn serve_abort_sends_notifications_cancelled_for_a_running_tool_call() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(home.join(".beyond")).unwrap();
    let flag = dir.path().join("cancelled.flag");
    std::fs::write(
        home.join(".beyond/settings.json"),
        json!({ "mcp_servers": [{
            "name": "slow",
            "transport": "stdio",
            "command": env!("CARGO_BIN_EXE_mcp_fixture_stdio_server"),
            "args": [],
            "env": { "MCP_FIXTURE_REQUEST_CANCEL_FLAG": flag.to_string_lossy() },
        }] })
        .to_string(),
    )
    .unwrap();

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_tool_use(
        "toolu_h",
        "mcp__slow__hold_cancellable",
        &json!({}).to_string(),
    )]);
    let mut cmd = serve_cmd(env!("CARGO_BIN_EXE_agent"), &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hold" })).unwrap();
    stdin.flush().unwrap();
    // The tool has started once its `tool_start` event is out.
    common::read_until_event(&mut stdout, |e| e["kind"] == "tool_start");
    std::thread::sleep(Duration::from_millis(300));
    writeln!(stdin, "{}", json!({ "type": "abort", "id": "a1" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "response" && f["command"] == "prompt" && f["success"] == false),
        "aborted prompt reports failure: {frames:?}"
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut seen = None;
    while Instant::now() < deadline {
        if let Ok(id) = std::fs::read_to_string(&flag)
            && !id.is_empty()
        {
            seen = Some(id);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let seen: Value = serde_json::from_str(&seen.expect(
        "abort must send notifications/cancelled for the running tools/call (flag never written)",
    ))
    .unwrap();
    assert!(seen.is_number() || seen.is_string(), "request id: {seen}");

    drop(stdin);
    child.wait().unwrap();
}
