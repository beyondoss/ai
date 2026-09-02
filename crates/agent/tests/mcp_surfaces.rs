//! MCP resources / prompts / completions e2e — proves the surfaces added beyond tools/list+call.
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

fn response<'a>(frames: &'a [Value], command: &str) -> &'a Value {
    frames
        .iter()
        .find(|f| f["type"] == "response" && f["command"] == command)
        .unwrap_or_else(|| panic!("missing response for {command}: {frames:?}"))
}

fn turn_tool_result_text(frames: &[Value]) -> String {
    frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .filter_map(|f| f["event"]["result"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn get_mcp_lists_resources_and_prompts_and_tools_call_them() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(
        &home,
        json!([{
            "name": "surf",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {},
        }]),
    );

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_r",
            "mcp__surf__resource__doc",
            &json!({}).to_string(),
        ),
        turn_tool_use(
            "toolu_p",
            "mcp__surf__prompt__greet",
            &json!({ "who": "Ferris" }).to_string(),
        ),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "id": "g", "type": "get_mcp" })).unwrap();
    let g_frames = read_until_response(&mut stdout, "get_mcp");
    let g = response(&g_frames, "get_mcp");
    assert_eq!(g["success"], true, "{g}");
    let tools = g["data"]["tools"].as_array().unwrap();
    let tool_names: Vec<&str> = tools.iter().filter_map(|t| t.as_str()).collect();
    assert!(
        tool_names.iter().any(|n| *n == "mcp__surf__resource__doc"),
        "resource tool missing: {tool_names:?}"
    );
    assert!(
        tool_names.iter().any(|n| *n == "mcp__surf__prompt__greet"),
        "prompt tool missing: {tool_names:?}"
    );
    assert!(
        !g["data"]["resources"].as_array().unwrap().is_empty(),
        "resources empty: {g}"
    );
    assert!(
        !g["data"]["prompts"].as_array().unwrap().is_empty(),
        "prompts empty: {g}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "read resource then prompt" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let text = turn_tool_result_text(&frames);
    assert!(
        text.contains("fixture-resource-body"),
        "resource body missing from tool results: {text}"
    );
    assert!(
        text.contains("Please greet Ferris"),
        "prompt expansion missing: {text}"
    );
}

#[test]
fn mcp_complete_returns_fixture_suggestions() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(
        &home,
        json!([{
            "name": "surf",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {},
        }]),
    );

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("idle")]);

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
        json!({
            "id": "c",
            "type": "mcp_complete",
            "server": "surf",
            "params": {
                "ref": { "type": "ref/prompt", "name": "greet" },
                "argument": { "name": "who", "value": "al" }
            }
        })
    )
    .unwrap();
    let frames = read_until_response(&mut stdout, "mcp_complete");
    drop(stdin);
    child.wait().unwrap();

    let c = response(&frames, "mcp_complete");
    assert_eq!(c["success"], true, "{c}");
    let values = c["data"]["completion"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<_>>();
    assert!(
        values.contains(&"alice") && values.contains(&"alex"),
        "unexpected completions: {values:?}"
    );
}
