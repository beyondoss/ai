//! An MCP server too slow to answer `server/discover` before the client gives up on it still
//! connects — over stdio (`tools::mcp_stdio`, `MessageCaps::stale`) and over streamable HTTP
//! (`tools::mcp_wire`, `HttpClient`).
//!
//! rmcp's handshake sends `server/discover` and, after 10 s of silence, falls back to a legacy
//! `initialize` on the same connection. A slow server (an `npx` cold start on a loaded host, a
//! cold-start serverless MCP host) then answers the discover after the initialize went out, and rmcp
//! failed the whole handshake on that stale answer — the server and every tool it had were gone for
//! the run. In one A/B that was 6 of 132 MCP runs, and it read as a broken search
//! (`tools.search("browser")` → `[]`) because the browser server was simply absent. Real binary,
//! real fixtures; each test takes the 10 s fallback.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_fixture::{
    spawn_http_mcp_fixture_rejecting_discover, spawn_http_mcp_fixture_slow_discover,
};
use common::{body_json, run_cmd, spawn_model_server, turn_text};
use serde_json::{Value, json};

/// Run once with `server` configured; assert it connected and `tool` is advertised.
fn assert_connects(server: Value, tool: &str) {
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().join(".beyond");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("settings.json"),
        json!({ "mcp_servers": [server] }).to_string(),
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
    assert!(!stderr.contains("warning: mcp server"), "{stderr}");
    let first = body_json(&bodies.lock().unwrap()[0]);
    assert!(
        first["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == tool),
        "the slow server's tools are there: {}",
        first["tools"]
    );
}

#[test]
fn a_late_answer_to_an_abandoned_discover_does_not_lose_a_stdio_server() {
    assert_connects(
        json!({
            "name": "slow",
            "transport": "stdio",
            "command": env!("CARGO_BIN_EXE_mcp_fixture_stdio_server"),
            "args": [],
            "env": { "MCP_FIXTURE_LATE_DISCOVER": "1" },
        }),
        "mcp__slow__add",
    );
}

#[test]
fn a_late_answer_to_an_abandoned_discover_does_not_lose_an_http_server() {
    // Past rmcp's 10 s discover timeout, so the legacy `initialize` goes out first.
    let fixture = spawn_http_mcp_fixture_slow_discover(Duration::from_secs(12));
    assert_connects(
        json!({ "name": "slowhttp", "transport": "http", "url": fixture.url }),
        "mcp__slowhttp__echo",
    );
}

#[test]
fn a_server_that_rejects_the_discover_probe_with_http_400_still_connects() {
    // DeepWiki's answer: HTTP 400, code -32600, an id that is not the request's. Any error to the
    // probe means "not a 2026-07-28 server", and the client falls back to `initialize`.
    let fixture = spawn_http_mcp_fixture_rejecting_discover();
    assert_connects(
        json!({ "name": "strict", "transport": "http", "url": fixture.url }),
        "mcp__strict__echo",
    );
}
