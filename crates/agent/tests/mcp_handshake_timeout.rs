//! A configured MCP server that never finishes its handshake costs its own tools, not startup.
//!
//! Servers connect concurrently, but startup waits for all of them — so one that accepts the
//! connection (or starts) and then says nothing held `run` and `serve` forever, every other server's
//! tools with it. Grant connectors were already bounded; an operator's own servers, stdio and HTTP,
//! now get the same bound (`BEYOND_AI_AGENT_MCP_HANDSHAKE_SECS`, shortened here).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{SpawnGuarded, run_cmd, spawn_model_server, turn_text};
use serde_json::{Value, json};

const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_fixture_stdio_server");

/// A TCP endpoint that accepts connections and never writes a byte.
fn silent_http() -> String {
    let listener = beyond_ai_test_support::ports::listener();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for conn in listener.incoming() {
            held.push(conn);
        }
    });
    format!("http://{addr}/mcp")
}

#[test]
fn a_silent_server_times_out_and_the_others_still_load() {
    let home = tempfile::tempdir().unwrap();
    let claude = home.path().join(".beyond");
    std::fs::create_dir_all(&claude).unwrap();
    let servers = json!([
        { "name": "silent-stdio", "transport": "stdio", "command": FIXTURE,
          "env": { "MCP_FIXTURE_SILENT": "1" } },
        { "name": "silent-http", "transport": "http", "url": silent_http() },
        { "name": "ok", "transport": "stdio", "command": FIXTURE },
    ]);
    std::fs::write(
        claude.join("settings.json"),
        json!({ "mcp_servers": servers }).to_string(),
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let stderr_path = home.path().join("stderr");
    let cwd = tempfile::tempdir().unwrap();
    let mut child = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_HANDSHAKE_SECS", "2")
        .args([
            "run",
            "go",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "2",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .stdout(Stdio::null())
        .stderr(Stdio::from(std::fs::File::create(&stderr_path).unwrap()))
        .spawn_guarded();

    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "`run` never got past connecting its MCP servers; stderr:\n{}",
            std::fs::read_to_string(&stderr_path).unwrap_or_default()
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    let stderr = std::fs::read_to_string(&stderr_path).unwrap();
    assert!(status.success(), "run failed: {stderr}");
    for silent in ["silent-stdio", "silent-http"] {
        assert!(
            stderr
                .lines()
                .any(|l| l.contains(silent) && l.contains("timed out")),
            "the silent server `{silent}` should be reported as timed out: {stderr}"
        );
    }

    let bodies = bodies.lock().unwrap();
    let body: Value = serde_json::from_str(bodies[0].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    let names: Vec<&str> = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert!(
        names.contains(&"mcp__ok__echo"),
        "the healthy server's tools must load: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("mcp__silent")),
        "{names:?}"
    );
}
