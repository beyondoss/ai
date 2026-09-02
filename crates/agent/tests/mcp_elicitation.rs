//! MCP elicitation e2e: fixture `ask_user` sends nested `elicitation/create`; serve UI answers via
//! `elicit`, and the tool completes with the accepted content.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Duration;

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
fn serve_answers_mcp_elicitation_mid_tool_call() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(
        &home,
        json!([{
            "name": "ask",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {},
        }]),
    );

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_e", "mcp__ask__ask_user", &json!({}).to_string()),
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
        json!({ "type": "prompt", "message": "ask the user" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Wait for the elicitation_request control frame mid-prompt.
    let mut elicit_id = None;
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let mut line = String::new();
    while std::time::Instant::now() < deadline {
        line.clear();
        stdout.read_line(&mut line).unwrap();
        if line.is_empty() {
            break;
        }
        let Ok(frame) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if frame["type"] == "elicitation_request" {
            elicit_id = frame["request_id"].as_str().map(str::to_string);
            assert_eq!(frame["server"], "ask");
            break;
        }
    }
    let elicit_id = elicit_id.expect("timed out waiting for elicitation_request");

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "elicit",
            "request_id": elicit_id,
            "action": "accept",
            "content": { "name": "Ferris" }
        })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let tool_text: String = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .filter_map(|f| f["event"]["result"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        tool_text.contains("hello-Ferris"),
        "elicitation content must reach the tool result; got {tool_text:?} frames={frames:?}"
    );
}
