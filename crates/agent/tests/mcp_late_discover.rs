//! A stdio MCP server too slow to answer `server/discover` before the client gives up on it still
//! connects (`tools::mcp_stdio`, `MessageCaps::stale`).
//!
//! rmcp's handshake sends `server/discover` and, after 10 s of silence, falls back to a legacy
//! `initialize` on the same connection. A slow server (an `npx` cold start on a loaded host) then
//! answers the discover after the initialize went out, and rmcp failed the whole handshake on that
//! stale answer — the server and every tool it had were gone for the run. In one A/B that was 6 of
//! 132 MCP runs, and it read as a broken search (`tools.search("browser")` → `[]`) because the
//! browser server was simply absent. Real binary, real fixture process; takes the 10 s fallback.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{body_json, run_cmd, spawn_model_server, turn_text};
use serde_json::json;

#[test]
fn a_late_answer_to_an_abandoned_discover_does_not_lose_the_server() {
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().join(".claude");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("settings.json"),
        json!({ "mcp_servers": [{
            "name": "slow",
            "transport": "stdio",
            "command": env!("CARGO_BIN_EXE_mcp_fixture_stdio_server"),
            "args": [],
            "env": { "MCP_FIXTURE_LATE_DISCOVER": "1" },
        }] })
        .to_string(),
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let out = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .current_dir(home.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-sonnet-5-5",
            "--no-session-persistence",
            "--mcp-tool-loading",
            "eager",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("handshake"), "{stderr}");
    let first = body_json(&bodies.lock().unwrap()[0]);
    assert!(
        first["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "mcp__slow__add"),
        "the slow server's tools are there: {}",
        first["tools"]
    );
}
