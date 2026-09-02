//! Session-scoped MCP enable/disable (kit shaping) e2e.
//!
//! Proves `set_mcp_enabled` / `get_mcp` change which MCP tools are advertised to the model without
//! restarting the process — the certainty test for the hot-plug gap vs Claude Code.
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

fn two_servers() -> Value {
    json!([
        {
            "name": "alpha",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {},
        },
        {
            "name": "beta",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {},
        },
    ])
}

fn response<'a>(frames: &'a [Value], command: &str) -> &'a Value {
    frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == command)
        .unwrap_or_else(|| panic!("missing response for {command}: {frames:?}"))
}

fn tool_names(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|t| t.as_str().map(str::to_string))
        .collect()
}

#[test]
fn set_mcp_enabled_changes_advertised_tools_and_model_request() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, two_servers());

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![turn_text("one"), turn_text("two")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Default: both servers enabled.
    writeln!(stdin, "{}", json!({ "id": "g0", "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let _g0_frames = read_until_response(&mut stdout, "get_mcp");
    let g0 = response(&_g0_frames, "get_mcp");
    assert_eq!(g0["success"], true, "{g0}");
    assert_eq!(g0["data"]["mode"], "all");
    let configured = g0["data"]["configured"].as_array().unwrap();
    assert!(
        configured.iter().any(|v| v == "alpha") && configured.iter().any(|v| v == "beta"),
        "both servers configured: {g0}"
    );
    let tools0 = tool_names(&g0["data"]["tools"]);
    assert!(
        tools0.iter().any(|t| t.starts_with("mcp__alpha__"))
            && tools0.iter().any(|t| t.starts_with("mcp__beta__")),
        "default kit advertises both servers: {g0}"
    );

    // Restrict to alpha only.
    writeln!(
        stdin,
        "{}",
        json!({ "id": "s1", "type": "set_mcp_enabled", "servers": ["alpha"] })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _s1_frames = read_until_response(&mut stdout, "set_mcp_enabled");
    let s1 = response(&_s1_frames, "set_mcp_enabled");
    assert_eq!(s1["success"], true, "{s1}");
    assert_eq!(s1["data"]["mode"], "allowlist");
    assert_eq!(s1["data"]["enabled"], json!(["alpha"]));

    writeln!(stdin, "{}", json!({ "id": "g1", "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let _g1_frames = read_until_response(&mut stdout, "get_mcp");
    let g1 = response(&_g1_frames, "get_mcp");
    let tools1 = tool_names(&g1["data"]["tools"]);
    assert!(
        tools1.iter().any(|t| t.starts_with("mcp__alpha__")),
        "alpha tools remain: {g1}"
    );
    assert!(
        tools1.iter().all(|t| !t.starts_with("mcp__beta__")),
        "beta tools must be gone after set_mcp_enabled: {g1}"
    );

    // Prompt — model request body must not advertise beta tools.
    writeln!(
        stdin,
        "{}",
        json!({ "id": "p1", "type": "prompt", "message": "hi" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    {
        let recorded = bodies.lock().unwrap();
        assert!(!recorded.is_empty(), "model must have been called");
        assert!(
            !recorded[0].contains("mcp__beta__"),
            "disabled server's tools must not appear in the model request: {}",
            recorded[0]
        );
        assert!(
            recorded[0].contains("mcp__alpha__"),
            "enabled server's tools must still appear: {}",
            recorded[0]
        );
    }

    // Unknown server name is rejected (no silent empty kit).
    writeln!(
        stdin,
        "{}",
        json!({ "id": "bad", "type": "set_mcp_enabled", "servers": ["nope"] })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _bad_frames = read_until_response(&mut stdout, "set_mcp_enabled");
    let bad = response(&_bad_frames, "set_mcp_enabled");
    assert_eq!(bad["success"], false, "{bad}");
    assert!(
        bad["error"]
            .as_str()
            .unwrap_or("")
            .contains("unknown MCP server"),
        "{bad}"
    );

    // Re-enable all.
    writeln!(
        stdin,
        "{}",
        json!({ "id": "s2", "type": "set_mcp_enabled" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _s2_frames = read_until_response(&mut stdout, "set_mcp_enabled");
    let s2 = response(&_s2_frames, "set_mcp_enabled");
    assert_eq!(s2["success"], true, "{s2}");
    assert_eq!(s2["data"]["mode"], "all");

    writeln!(
        stdin,
        "{}",
        json!({ "id": "p2", "type": "prompt", "message": "again" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    {
        let recorded = bodies.lock().unwrap();
        assert!(recorded.len() >= 2);
        assert!(
            recorded[1].contains("mcp__beta__") && recorded[1].contains("mcp__alpha__"),
            "re-enabling all must restore both servers in the model request: {}",
            recorded[1]
        );
    }

    // Empty allowlist disables every MCP tool.
    writeln!(
        stdin,
        "{}",
        json!({ "id": "s3", "type": "set_mcp_enabled", "servers": [] })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _s3_frames = read_until_response(&mut stdout, "set_mcp_enabled");
    let s3 = response(&_s3_frames, "set_mcp_enabled");
    assert_eq!(s3["success"], true, "{s3}");
    writeln!(stdin, "{}", json!({ "id": "g3", "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let _g3_frames = read_until_response(&mut stdout, "get_mcp");
    let g3 = response(&_g3_frames, "get_mcp");
    assert_eq!(g3["data"]["tools"], json!([]), "{g3}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn disabled_mcp_tool_is_not_callable() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, two_servers());

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_x",
            "mcp__beta__echo",
            &json!({ "text": "should-fail" }).to_string(),
        ),
        turn_text("ok"),
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
        json!({ "type": "set_mcp_enabled", "servers": ["alpha"] })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _frames = read_until_response(&mut stdout, "set_mcp_enabled");
    assert_eq!(response(&_frames, "set_mcp_enabled")["success"], true);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "call beta" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let tool_end = frames
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .expect("tool_end");
    assert_eq!(
        tool_end["event"]["is_error"], true,
        "calling a disabled MCP tool must error: {tool_end}"
    );
    let result = tool_end["event"]["result"].as_str().unwrap_or("");
    assert!(
        result.contains("unknown tool"),
        "error should be unknown tool, got: {result:?}"
    );
}

#[test]
fn new_session_resets_mcp_enablement() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, two_servers());

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("x")]);

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
        json!({ "type": "set_mcp_enabled", "servers": ["alpha"] })
    )
    .unwrap();
    stdin.flush().unwrap();
    let _frames = read_until_response(&mut stdout, "set_mcp_enabled");
    assert_eq!(response(&_frames, "set_mcp_enabled")["success"], true);

    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let _ns_frames = read_until_response(&mut stdout, "new_session");
    let ns = response(&_ns_frames, "new_session");
    assert_eq!(ns["success"], true, "{ns}");

    writeln!(stdin, "{}", json!({ "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let _g_frames = read_until_response(&mut stdout, "get_mcp");
    let g = response(&_g_frames, "get_mcp");
    assert_eq!(
        g["data"]["mode"], "all",
        "new_session must reset MCP kit: {g}"
    );
    let tools = tool_names(&g["data"]["tools"]);
    assert!(
        tools.iter().any(|t| t.starts_with("mcp__beta__")),
        "beta must be back after new_session: {g}"
    );

    drop(stdin);
    child.wait().unwrap();
}
