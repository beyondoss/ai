//! Session-scoped MCP enable/disable (kit shaping) e2e.
//!
//! Proves `set_mcp_enabled` / `get_mcp` change which MCP tools are advertised to the model without
//! restarting the process — the certainty test for the hot-plug gap vs Claude Code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

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

fn spawn_serve(
    home: &Path,
    session_file: &str,
    base: &str,
) -> (SpawnGuarded, impl Write, BufReader<impl std::io::Read>) {
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, base, session_file);
    cmd.env("HOME", home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let stdin = child.stdin.take().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());
    (child, stdin, stdout)
}

fn response(frames: &[Value], command: &str) -> &Value {
    frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == command)
        .unwrap_or_else(|| panic!("missing response for {command}: {frames:?}"))
}

#[test]
fn set_mcp_enabled_changes_advertised_tools_and_model_request() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, two_servers());

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    // Two prompts: first with only alpha enabled, second after re-enabling all.
    let (base, bodies) = spawn_model_server(vec![turn_text("one"), turn_text("two")]);

    let (mut child, mut stdin, mut stdout) = spawn_serve(&home, &session_file, &base);

    // Default: both servers enabled.
    writeln!(stdin, "{}", json!({ "id": "g0", "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let g0 = response(&read_until_response(&mut stdout, "get_mcp"), "get_mcp");
    assert_eq!(g0["success"], true, "{g0}");
    assert_eq!(g0["data"]["mode"], "all");
    let configured = g0["data"]["configured"].as_array().unwrap();
    assert!(
        configured.iter().any(|v| v == "alpha") && configured.iter().any(|v| v == "beta"),
        "both servers configured: {g0}"
    );
    let tools0 = g0["data"]["tools"].as_array().unwrap();
    assert!(
        tools0.iter().any(|t| t.as_str().unwrap().starts_with("mcp__alpha__"))
            && tools0.iter().any(|t| t.as_str().unwrap().starts_with("mcp__beta__")),
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
    let s1 = response(
        &read_until_response(&mut stdout, "set_mcp_enabled"),
        "set_mcp_enabled",
    );
    assert_eq!(s1["success"], true, "{s1}");
    assert_eq!(s1["data"]["mode"], "allowlist");
    assert_eq!(s1["data"]["enabled"], json!(["alpha"]));

    writeln!(stdin, "{}", json!({ "id": "g1", "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let g1 = response(&read_until_response(&mut stdout, "get_mcp"), "get_mcp");
    let tools1 = g1["data"]["tools"].as_array().unwrap();
    assert!(
        tools1.iter().any(|t| t.as_str().unwrap().starts_with("mcp__alpha__")),
        "alpha tools remain: {g1}"
    );
    assert!(
        tools1
            .iter()
            .all(|t| !t.as_str().unwrap().starts_with("mcp__beta__")),
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
    let bad = response(
        &read_until_response(&mut stdout, "set_mcp_enabled"),
        "set_mcp_enabled",
    );
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
    let s2 = response(
        &read_until_response(&mut stdout, "set_mcp_enabled"),
        "set_mcp_enabled",
    );
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
    let s3 = response(
        &read_until_response(&mut stdout, "set_mcp_enabled"),
        "set_mcp_enabled",
    );
    assert_eq!(s3["success"], true, "{s3}");
    writeln!(stdin, "{}", json!({ "id": "g3", "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let g3 = response(&read_until_response(&mut stdout, "get_mcp"), "get_mcp");
    assert_eq!(g3["data"]["tools"], json!([]), "{g3}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn disabled_mcp_tool_is_not_callable() {
    // Even if the model somehow names a disabled tool, it must not succeed as a live call — the
    // tool is absent from the registry (unregistered), so the loop synthesizes an error result.
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, two_servers());

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies): (String, Arc<Mutex<Vec<String>>>) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_x",
            "mcp__beta__echo",
            &json!({ "text": "should-fail" }).to_string(),
        ),
        turn_text("ok"),
    ]);

    let (mut child, mut stdin, mut stdout) = spawn_serve(&home, &session_file, &base);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_mcp_enabled", "servers": ["alpha"] })
    )
    .unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        response(
            &read_until_response(&mut stdout, "set_mcp_enabled"),
            "set_mcp_enabled"
        )["success"],
        true
    );

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

    let session_dir = dir.path().join("sessions");
    std::fs::create_dir_all(&session_dir).unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("x")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, "unused.jsonl");
    // Use a session dir so new_session can mint distinct sessions.
    cmd.args([
        "--session-dir",
        session_dir.to_str().unwrap(),
    ]);
    // serve_cmd already sets --session-file; override by rebuilding... look at serve_cmd
    // Actually serve_cmd takes session_file. For dir mode check how other tests do it.
    drop(cmd);

    // Simpler: same session-file serve, call set_mcp_enabled then new_session if supported with file mode.
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (mut child, mut stdin, mut stdout) = spawn_serve(&home, &session_file, &base);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_mcp_enabled", "servers": ["alpha"] })
    )
    .unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        response(
            &read_until_response(&mut stdout, "set_mcp_enabled"),
            "set_mcp_enabled"
        )["success"],
        true
    );

    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let ns = response(
        &read_until_response(&mut stdout, "new_session"),
        "new_session",
    );
    assert_eq!(ns["success"], true, "{ns}");

    writeln!(stdin, "{}", json!({ "type": "get_mcp" })).unwrap();
    stdin.flush().unwrap();
    let g = response(&read_until_response(&mut stdout, "get_mcp"), "get_mcp");
    assert_eq!(g["data"]["mode"], "all", "new_session must reset MCP kit: {g}");
    let tools = g["data"]["tools"].as_array().unwrap();
    assert!(
        tools.iter().any(|t| t.as_str().unwrap().starts_with("mcp__beta__")),
        "beta must be back after new_session: {g}"
    );

    drop(stdin);
    child.wait().unwrap();
}
