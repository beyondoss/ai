//! Kit shaping reaches the children too: `set_mcp_enabled` decides what a **subagent** may call.
//!
//! A sibling of `mcp_hotplug.rs` (same domain — which MCP servers a session advertises — different
//! surface). The subagent context is built once per session and reused by every child after it, so
//! a child's MCP kit has to be read through the session's live gate rather than snapshotted when
//! the context was built: otherwise disabling a server leaves it reachable by simply delegating.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::path::Path;

use common::{
    SpawnGuarded, advertised_tools, read_until_response, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

fn fixture_bin() -> &'static str {
    env!("CARGO_BIN_EXE_mcp_fixture_stdio_server")
}

fn write_global_settings(home: &Path) {
    let claude_dir = home.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        json!({
            "mcp_servers": [{
                "name": "alpha",
                "transport": "stdio",
                "command": fixture_bin(),
                "args": [],
                "env": {},
            }]
        })
        .to_string(),
    )
    .unwrap();
}

/// An agent with no `tools:` of its own, so it inherits exactly the parent's effective set — which
/// is where an MCP tool would reach it from.
fn write_scout(dir: &Path) {
    let agents = dir.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("scout.md"),
        "---\nname: scout\ndescription: recon\n---\nYou are SCOUT-MARKER.\n",
    )
    .unwrap();
}

fn delegate() -> String {
    turn_tool_use(
        "call-1",
        "subagent",
        &json!({ "agent": "scout", "task": "look around" }).to_string(),
    )
}

#[test]
fn disabling_a_server_takes_its_tools_away_from_a_subagent_too() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home);
    let project = dir.path().join("project");
    std::fs::create_dir_all(&project).unwrap();
    write_scout(&project);
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // Two delegating turns, one before the server is disabled and one after.
    let (base, bodies) = spawn_model_server(vec![
        delegate(),
        turn_text("CHILD-ONE"),
        turn_text("done one"),
        delegate(),
        turn_text("CHILD-TWO"),
        turn_text("done two"),
    ]);

    let mut cmd = serve_cmd(common::BIN, &base, &session_file);
    cmd.env("HOME", &home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .arg("--trust-project")
        .current_dir(&project);
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "id": "p1", "type": "prompt", "message": "explore" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Now take the whole MCP kit away — after the subagent context was already built.
    writeln!(
        stdin,
        "{}",
        json!({ "id": "s1", "type": "set_mcp_enabled", "servers": [] })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_mcp_enabled");
    let disabled = frames
        .iter()
        .rev()
        .find(|f: &&Value| f["command"] == "set_mcp_enabled")
        .unwrap();
    assert_eq!(disabled["success"], true, "{disabled}");

    writeln!(
        stdin,
        "{}",
        json!({ "id": "p2", "type": "prompt", "message": "explore again" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    let _ = child.kill();
    let _ = child.wait();

    let requests = bodies.lock().unwrap().clone();
    assert!(
        requests.len() >= 5,
        "expected two full delegations: {requests:#?}"
    );
    let first_child = advertised_tools(&requests[1]);
    let second_child = advertised_tools(&requests[4]);
    assert!(
        first_child.iter().any(|t| t.starts_with("mcp__alpha__")),
        "a child inherits the parent's MCP kit while the server is enabled: {first_child:?}"
    );
    assert!(
        second_child.iter().all(|t| !t.starts_with("mcp__alpha__")),
        "a disabled server must be unreachable from a subagent, not just from the parent: \
         {second_child:?}"
    );
    // The child really did run both times — the second list isn't empty for some other reason.
    assert!(
        second_child.iter().any(|t| t == "read"),
        "the second child still has its ordinary tools: {second_child:?}"
    );
}
