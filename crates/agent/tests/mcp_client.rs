//! MCP (Model Context Protocol) client e2e: proves `tools::mcp` end to end through the real compiled
//! `beyond-ai-agent` binary — real subprocess (stdio transport) and real TCP (streamable-HTTP
//! transport), never a mock of the MCP protocol itself. The model side stays mocked (`spawn_model_server`,
//! matching every other e2e test in this directory) so a test can deterministically script "call this
//! MCP tool with these arguments" and inspect exactly what reached the wire.
//!
//! Fixture: `mcp_fixture_stdio_server` (`src/bin/mcp_fixture_stdio_server.rs`, a sibling `[[bin]]` of
//! this same package) is a ~150-line hand-rolled MCP server — no `rmcp` dependency, just newline-
//! delimited JSON-RPC over stdin/stdout — with six tools (`echo`, `add`, `ping`, `fail`, `echo_env`,
//! `image`) each proving a distinct behavior. The streamable-HTTP tests reimplement a small subset of
//! the same dispatch directly in this file (a plain blocking `TcpListener` thread, matching
//! `common::spawn_model_server`'s own idiom) since that transport can't be a subprocess.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use common::{
    read_until_response, run_cmd, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

/// The compiled fixture stdio server's path — a sibling `[[bin]]` target of the `beyond-ai-agent`
/// package (auto-discovered from `src/bin/`), so Cargo builds it before running this test and exposes
/// its path the same way it does for the main binary.
fn fixture_bin() -> &'static str {
    env!("CARGO_BIN_EXE_mcp_fixture_stdio_server")
}

/// One `mcp_servers` entry for a stdio server backed by the fixture binary.
fn stdio_server_config(name: &str, env: Value) -> Value {
    json!({
        "name": name,
        "transport": "stdio",
        "command": fixture_bin(),
        "args": [],
        "env": env,
    })
}

/// Write `~/.claude/settings.json` (the global tier) under `home`, with the given `mcp_servers` array —
/// applies regardless of whether the working directory is trusted (only a *project*-tier settings.json
/// is trust-gated; see `settings::effective_settings_for_cwd`'s doc comment).
fn write_global_settings(home: &Path, mcp_servers: Value) {
    let claude_dir = home.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": mcp_servers })).unwrap(),
    )
    .unwrap();
}

/// Write a *project*-tier `<project>/.claude/settings.json` — only takes effect when `project` is
/// trusted (a persisted `agent trust`, not just present on disk).
fn write_project_settings(project: &Path, mcp_servers: Value) {
    let claude_dir = project.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": mcp_servers })).unwrap(),
    )
    .unwrap();
}

/// Persist a real, on-disk trust grant for `project` via the real `agent trust` CLI subcommand — a
/// project's own `mcp_servers` only ever applies under a *persisted* grant, never a one-off
/// `--trust-project` (see `settings::effective_settings_for_cwd`'s doc comment), so tests that need one
/// must go through this, not just pass `--trust-project` on the `run` invocation itself.
fn trust(home: &Path, project: &Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["trust", project.to_str().unwrap()])
        .env("HOME", home)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "`agent trust` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Run the real `beyond-ai-agent run` binary against a mock model server, with `home` as `$HOME` (so
/// `~/.claude/settings.json` resolves there) and `cwd` as the working directory. Returns
/// `(exit_success, stdout, stderr, recorded_model_request_bodies)`.
fn run_against(
    home: &Path,
    cwd: &Path,
    task: &str,
    responses: Vec<String>,
) -> (bool, String, String, Vec<String>) {
    let (base, bodies) = spawn_model_server(responses);
    let output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home)
        .args([
            "run",
            task,
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "6",
            "--no-session-persistence",
        ])
        .current_dir(cwd)
        .output()
        .expect("spawn binary");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let bodies = bodies.lock().unwrap().clone();
    (output.status.success(), stdout, stderr, bodies)
}

// ============================================================================================
// Stdio transport
// ============================================================================================

#[test]
fn mcp_stdio_tool_is_discovered_and_namespaced_in_the_advertised_tool_defs() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        vec![turn_text("hi there")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        bodies[0].contains("\"mcp__fixture__echo\""),
        "the mcp__<server>__<tool> namespaced name must be advertised: {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("\"mcp__fixture__add\"") && bodies[0].contains("\"mcp__fixture__ping\""),
        "every tool the fixture server lists must be advertised: {}",
        bodies[0]
    );
}

#[test]
fn mcp_stdio_tool_call_round_trips_text_back_to_the_model() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__fixture__echo",
        &json!({ "text": "hello-mcp-e2e-marker-91827" }).to_string(),
    );
    let (ok, stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "call the echo tool",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        stdout.contains("[tool: mcp__fixture__echo]"),
        "should show the tool call preview: {stdout}"
    );
    assert_eq!(bodies.len(), 2, "expected two model requests");
    assert!(
        bodies[1].contains("hello-mcp-e2e-marker-91827"),
        "the 2nd request must feed the real MCP server's response back as the tool_result: {}",
        bodies[1]
    );
}

/// An Anthropic SSE turn calling *two* tools in the same batch (two `tool_use` content blocks, indices
/// 0 and 1) — `common::turn_tool_use` only builds a single-tool-call turn, and the concurrency test
/// below specifically needs two calls landing in the same turn's dispatch batch.
fn turn_two_tool_uses(
    id1: &str,
    name1: &str,
    args1_json: &str,
    id2: &str,
    name2: &str,
    args2_json: &str,
) -> String {
    common::sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id1, "name": name1, "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": args1_json } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "tool_use", "id": id2, "name": name2, "input": {} } }),
        json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "input_json_delta", "partial_json": args2_json } }),
        json!({ "type": "content_block_stop", "index": 1 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// Find the `tool_result` content block addressed to `tool_use_id` in a raw recorded request body
/// (Anthropic wire shape: `messages[].content[] == {"type": "tool_result", "tool_use_id", "content"}`)
/// and return its `content` as a string — precise enough to prove *which* call a result was actually
/// paired with, not just that both results appear somewhere in the body (a plain substring check on the
/// whole body can't tell two results apart, or catch them being swapped between calls).
fn tool_result_text(raw_request: &str, tool_use_id: &str) -> Option<String> {
    // `raw_request` is the *whole* recorded HTTP request (headers + body — see
    // `common::spawn_model_server`'s own doc comment), not bare JSON; skip past the blank-line
    // separator to the actual JSON body before parsing.
    let json_body = raw_request.split("\r\n\r\n").nth(1)?;
    let parsed: Value = serde_json::from_str(json_body).ok()?;
    let block = parsed
        .get("messages")?
        .as_array()?
        .iter()
        .flat_map(|m| {
            m.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .find(|block| {
            block.get("type").and_then(Value::as_str) == Some("tool_result")
                && block.get("tool_use_id").and_then(Value::as_str) == Some(tool_use_id)
        })?;
    block.get("content")?.as_str().map(str::to_string)
}

#[test]
fn mcp_stdio_two_tools_from_the_same_server_called_concurrently_in_one_turn_do_not_cross_wires() {
    // The two `McpTool`s for `echo`/`add` share one `Arc<McpClient>` (one connection to the `fixture`
    // server) — `agent_core::Agent`'s own bounded-concurrent tool dispatch calls both `run()`s at once
    // for a single turn's batch, so this proves `rmcp`'s `Peer<RoleClient>` correctly correlates each
    // concurrent `tools/call` response back to the request that asked for it, rather than a shared
    // client handle risking a race that hands one call's result to the other.
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let turn1 = turn_two_tool_uses(
        "toolu_echo",
        "mcp__fixture__echo",
        &json!({ "text": "concurrent-echo-marker" }).to_string(),
        "toolu_add",
        "mcp__fixture__add",
        &json!({ "a": 700, "b": 7 }).to_string(),
    );
    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "call echo and add together",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert_eq!(bodies.len(), 2, "expected two model requests");
    assert_eq!(
        tool_result_text(&bodies[1], "toolu_echo").as_deref(),
        Some("concurrent-echo-marker"),
        "echo's own result must be paired with echo's own tool_use_id: {}",
        bodies[1]
    );
    assert_eq!(
        tool_result_text(&bodies[1], "toolu_add").as_deref(),
        Some("707"),
        "add's own result must be paired with add's own tool_use_id, not echo's: {}",
        bodies[1]
    );
}

#[test]
fn mcp_stdio_tool_call_with_no_arguments_reaches_the_null_input_path() {
    // `ping` has an empty input schema and is called with no `arguments` field at all — proves
    // `McpTool::run`'s `Value::Null` branch (as opposed to `Value::Object`) is exercised, not just the
    // ordinary object-arguments case every other test here uses.
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let turn1 = turn_tool_use("toolu_1", "mcp__fixture__ping", "{}");
    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "call ping",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        bodies[1].contains("pong"),
        "ping's reply must reach the model: {}",
        bodies[1]
    );
}

#[test]
fn mcp_stdio_numeric_arguments_round_trip() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__fixture__add",
        &json!({ "a": 123450, "b": 6 }).to_string(),
    );
    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "add two numbers",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        bodies[1].contains("123456"),
        "a richer (multi-field, numeric) input schema must round-trip correctly: {}",
        bodies[1]
    );
}

#[test]
fn mcp_stdio_tool_error_propagates_as_an_error_tool_result() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let turn1 = turn_tool_use("toolu_1", "mcp__fixture__fail", "{}");
    let (ok, stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "call fail",
        vec![turn1, turn_text("done")],
    );
    assert!(
        ok,
        "the *agent* must not crash just because a tool call errored: {stderr}"
    );
    assert!(
        bodies[1].contains("intentional failure from fixture server"),
        "the MCP server's error message must reach the model as the tool_result: {}",
        bodies[1]
    );
    assert!(
        stdout.contains("[tool: mcp__fixture__fail]"),
        "the call preview must still show, error or not: {stdout}"
    );
}

#[test]
fn mcp_stdio_configured_env_vars_reach_the_spawned_child_process() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config(
            "fixture",
            json!({ "MCP_FIXTURE_GREETING": "hello-from-settings-json" })
        )]),
    );

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__fixture__echo_env",
        &json!({ "var": "MCP_FIXTURE_GREETING" }).to_string(),
    );
    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "read an env var via mcp",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        bodies[1].contains("hello-from-settings-json"),
        "an `env` entry configured on the mcp server must reach the spawned child's real \
         environment: {}",
        bodies[1]
    );
}

#[test]
fn mcp_stdio_image_content_becomes_an_image_block_in_the_tool_result() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let turn1 = turn_tool_use("toolu_1", "mcp__fixture__image", "{}");
    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "call the image tool",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    const TINY_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
    assert!(
        bodies[1].contains(TINY_PNG_BASE64),
        "an MCP ImageContent block must map to a real image in the tool_result, not be dropped or \
         summarized as text: {}",
        bodies[1]
    );
    assert!(
        bodies[1].contains("image/png"),
        "the image's mime type must be preserved: {}",
        bodies[1]
    );
}

#[test]
fn mcp_stdio_multiple_servers_each_contribute_their_own_namespaced_tools() {
    // Two servers, same underlying fixture binary, different `name`s — proves namespacing scopes by
    // *server* name, not just tool name: both advertise a same-named `echo` tool with zero collision.
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([
            stdio_server_config("alpha", json!({})),
            stdio_server_config("beta", json!({})),
        ]),
    );

    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        vec![turn_text("hi")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(bodies[0].contains("\"mcp__alpha__echo\""), "{}", bodies[0]);
    assert!(bodies[0].contains("\"mcp__beta__echo\""), "{}", bodies[0]);
}

#[test]
fn mcp_a_server_that_fails_to_spawn_is_skipped_fail_soft_while_others_still_work() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([
            {
                "name": "bad",
                "transport": "stdio",
                "command": "/definitely/does/not/exist-mcp-server-binary-xyz",
                "args": [],
                "env": {},
            },
            stdio_server_config("good", json!({})),
        ]),
    );

    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        vec![turn_text("hi")],
    );
    assert!(
        ok,
        "one misconfigured MCP server must not fail the whole run (fail-soft): {stderr}"
    );
    assert!(
        stderr.contains("warning:") && stderr.contains("mcp server `bad`"),
        "a connect failure must be surfaced as a warning naming the server: {stderr}"
    );
    assert!(
        bodies[0].contains("\"mcp__good__echo\""),
        "the other, working server's tools must still be available: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("\"mcp__bad__"),
        "the failed server must contribute zero tools: {}",
        bodies[0]
    );
}

#[test]
fn mcp_multiple_servers_connect_concurrently_not_sequentially() {
    // Regression guard for `tools::mcp::connect_all`'s `futures::future::join_all` fan-out: three
    // servers, each artificially delayed at startup by the same amount — if `connect_all` connected
    // them one at a time, total startup time would scale with the *sum* of the delays; connected
    // concurrently, it tracks the *slowest one*, plus incidental process-spawn/tokio-runtime overhead.
    const DELAY_MS: u64 = 400;
    const SERVER_COUNT: u64 = 3;

    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let delayed = |name: &str| {
        stdio_server_config(
            name,
            json!({ "MCP_FIXTURE_STARTUP_DELAY_MS": DELAY_MS.to_string() }),
        )
    };
    write_global_settings(
        home.path(),
        json!([delayed("one"), delayed("two"), delayed("three")]),
    );

    let start = Instant::now();
    let (ok, _stdout, stderr, _bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        vec![turn_text("hi")],
    );
    let elapsed = start.elapsed();
    assert!(ok, "run failed: {stderr}");
    assert!(
        elapsed < Duration::from_millis(DELAY_MS * SERVER_COUNT - 100),
        "three {DELAY_MS}ms-delayed servers took {elapsed:?} — should track the slowest one \
         (~{DELAY_MS}ms) plus overhead, not their sum (~{}ms), if they connected concurrently",
        DELAY_MS * SERVER_COUNT
    );
}

#[test]
fn mcp_project_settings_json_servers_are_ignored_when_the_project_is_untrusted() {
    let home = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    // No global servers at all — only the project tier configures one, so its absence from the
    // advertised tools is unambiguous evidence trust-gating actually applied.
    write_project_settings(
        project.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );

    let (ok, _stdout, _stderr, bodies) = run_against(
        home.path(),
        project.path(),
        "just say hi",
        vec![turn_text("hi")],
    );
    assert!(ok);
    assert!(
        !bodies[0].contains("mcp__fixture__"),
        "an untrusted project's own settings.json must not contribute any MCP servers at all: {}",
        bodies[0]
    );
}

#[test]
fn mcp_project_settings_json_servers_apply_once_the_project_is_persistently_trusted() {
    let home = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    write_project_settings(
        project.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );
    trust(home.path(), project.path());

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__fixture__echo",
        &json!({ "text": "trusted-project-marker" }).to_string(),
    );
    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        project.path(),
        "call echo",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        bodies[0].contains("\"mcp__fixture__echo\""),
        "a persistently-trusted project's own mcp_servers must apply: {}",
        bodies[0]
    );
    assert!(
        bodies[1].contains("trusted-project-marker"),
        "and must actually be callable, not just advertised: {}",
        bodies[1]
    );
}

// ============================================================================================
// Streamable-HTTP transport
// ============================================================================================

/// A minimal streamable-HTTP MCP fixture: a plain blocking `TcpListener` thread (matching
/// `common::spawn_model_server`'s own idiom — this transport has no subprocess to spawn, so a hand-
/// rolled server has to live somewhere, and duplicating a few lines of dispatch here is simpler than
/// sharing code with the separate `mcp_fixture_stdio_server` *binary* crate). Handles exactly what one
/// client handshake + one `echo` call needs: `initialize`, the `notifications/initialized` notification
/// (replied to with a bodyless `202 Accepted`, matching `StreamableHttpPostResponse::Accepted`),
/// `tools/list`, and `tools/call`. Relies on `rmcp`'s client defaulting to `allow_stateless: true` (no
/// `Mcp-Session-Id` handshake needed).
///
/// Returns the server's URL and a handle recording every `x-test-header` value it ever saw — proving
/// `McpTransport::Http`'s configured headers actually leave the client process and land on the wire.
fn spawn_http_mcp_fixture() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen_headers = Arc::new(Mutex::new(Vec::new()));
    let seen_headers_writer = seen_headers.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let mut header_end = None;
            while header_end.is_none() {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                header_end = buf.windows(4).position(|w| w == b"\r\n\r\n");
            }
            let Some(pos) = header_end else { continue };
            let headers_text = String::from_utf8_lossy(&buf[..pos]).into_owned();
            let content_length: usize = headers_text
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse().unwrap_or(0))
                })
                .unwrap_or(0);
            let body_start = pos + 4;
            while buf.len() < body_start + content_length {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            if let Some(value) = headers_text.lines().find_map(|l| {
                let lower = l.to_ascii_lowercase();
                lower.strip_prefix("x-test-header:").map(|_| {
                    l.split_once(':')
                        .map(|(_, v)| v)
                        .unwrap_or("")
                        .trim()
                        .to_string()
                })
            }) {
                seen_headers_writer.lock().unwrap().push(value);
            }

            let body = &buf[body_start..buf.len().min(body_start + content_length)];
            let request: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
            let is_notification = request.get("id").is_none();
            if is_notification {
                let _ = stream.write_all(
                    b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            }
            let id = request.get("id").cloned().unwrap_or(Value::Null);
            let method = request.get("method").and_then(Value::as_str).unwrap_or("");
            let result = match method {
                "initialize" => json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "mcp-fixture-http-server", "version": "0.0.0" },
                }),
                "tools/list" => json!({ "tools": [
                    {
                        "name": "echo",
                        "description": "Echoes back its `text` argument.",
                        "inputSchema": {
                            "type": "object",
                            "properties": { "text": { "type": "string" } },
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "fail",
                        "description": "Always fails, with an error message.",
                        "inputSchema": { "type": "object", "properties": {} },
                    },
                ] }),
                "tools/call" => {
                    let name = request
                        .pointer("/params/name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if name == "fail" {
                        json!({
                            "content": [{ "type": "text", "text": "intentional http failure" }],
                            "isError": true,
                        })
                    } else {
                        let text = request
                            .pointer("/params/arguments/text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        json!({ "content": [{ "type": "text", "text": text }], "isError": false })
                    }
                }
                other => json!({
                    "content": [{ "type": "text", "text": format!("unhandled method {other}") }],
                    "isError": true,
                }),
            };
            let envelope = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            let encoded = serde_json::to_vec(&envelope).unwrap();
            let http_header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                encoded.len()
            );
            let _ = stream.write_all(http_header.as_bytes());
            let _ = stream.write_all(&encoded);
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}/mcp"), seen_headers)
}

#[test]
fn mcp_http_streamable_tool_is_discovered_and_callable() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let (url, _seen_headers) = spawn_http_mcp_fixture();
    write_global_settings(
        home.path(),
        json!([{
            "name": "remote",
            "transport": "http",
            "url": url,
            "headers": {},
        }]),
    );

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__remote__echo",
        &json!({ "text": "hello-over-http-marker" }).to_string(),
    );
    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "call the remote echo tool",
        vec![turn1, turn_text("done")],
    );
    assert!(ok, "run failed: {stderr}");
    assert!(
        bodies[0].contains("\"mcp__remote__echo\""),
        "the http-transport tool must be advertised: {}",
        bodies[0]
    );
    assert!(
        bodies[1].contains("hello-over-http-marker"),
        "the http-transport tool's reply must reach the model: {}",
        bodies[1]
    );
}

#[test]
fn mcp_http_streamable_configured_headers_reach_the_wire() {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let (url, seen_headers) = spawn_http_mcp_fixture();
    write_global_settings(
        home.path(),
        json!([{
            "name": "remote",
            "transport": "http",
            "url": url,
            "headers": { "X-Test-Header": "configured-header-value" },
        }]),
    );

    let (ok, _stdout, stderr, _bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        vec![turn_text("hi")],
    );
    assert!(ok, "run failed: {stderr}");
    let seen = seen_headers.lock().unwrap();
    assert!(
        seen.iter().any(|v| v == "configured-header-value"),
        "a header configured on an http mcp server must actually be sent on every request: {seen:?}"
    );
}

#[test]
fn mcp_http_streamable_tool_error_propagates_as_an_error_tool_result() {
    // The stdio suite already covers `ToolError::Execution` on `isError: true` (`mcp_stdio_tool_error_
    // propagates_as_an_error_tool_result`) — `McpTool::run`'s error-mapping code is the same regardless
    // of transport, but this proves it end to end over streamable-HTTP too, not just stdio.
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let (url, _seen_headers) = spawn_http_mcp_fixture();
    write_global_settings(
        home.path(),
        json!([{ "name": "remote", "transport": "http", "url": url, "headers": {} }]),
    );

    let turn1 = turn_tool_use("toolu_1", "mcp__remote__fail", "{}");
    let (ok, stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "call the remote fail tool",
        vec![turn1, turn_text("done")],
    );
    assert!(
        ok,
        "the agent must not crash just because an http mcp tool call errored: {stderr}"
    );
    assert!(
        bodies[1].contains("intentional http failure"),
        "the http server's error message must reach the model as the tool_result: {}",
        bodies[1]
    );
    assert!(
        stdout.contains("[tool: mcp__remote__fail]"),
        "the call preview must still show, error or not: {stdout}"
    );
}

#[test]
fn mcp_http_streamable_a_server_that_refuses_the_connection_is_skipped_fail_soft() {
    // Distinct failure mode from stdio's "command not found" (`mcp_a_server_that_fails_to_spawn_is_
    // skipped_fail_soft_while_others_still_work`) — this exercises `connect_http`'s own error path (the
    // handshake itself failing, not a spawn failure), which is otherwise completely untested.
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    // A free port nothing is listening on — connecting to it fails fast with connection-refused, unlike
    // an unroutable address, which would hang until a timeout.
    let dead_port = common::free_port();
    let (url, _seen_headers) = spawn_http_mcp_fixture();
    write_global_settings(
        home.path(),
        json!([
            {
                "name": "bad",
                "transport": "http",
                "url": format!("http://127.0.0.1:{dead_port}/mcp"),
                "headers": {},
            },
            { "name": "good", "transport": "http", "url": url, "headers": {} },
        ]),
    );

    let (ok, _stdout, stderr, bodies) = run_against(
        home.path(),
        cwd.path(),
        "just say hi",
        vec![turn_text("hi")],
    );
    assert!(
        ok,
        "an unreachable http MCP server must not fail the whole run (fail-soft): {stderr}"
    );
    assert!(
        stderr.contains("warning:") && stderr.contains("mcp server `bad`"),
        "a connect failure must be surfaced as a warning naming the server: {stderr}"
    );
    assert!(
        bodies[0].contains("\"mcp__good__echo\""),
        "the other, reachable server's tools must still be available: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("\"mcp__bad__"),
        "the unreachable server must contribute zero tools: {}",
        bodies[0]
    );
}

// ============================================================================================
// `serve`, not just `run`
// ============================================================================================

#[test]
fn mcp_stdio_tool_is_discovered_and_callable_through_serve_too() {
    // Every test above drives `run` — MCP tools are wired into *two* call sites
    // (`main.rs::run_task`'s `run` path, and `ServeConfig::mcp_tools` before `serve`'s own session
    // loop starts), and it's the second one this proves. `serve_cmd` hardcodes `ISOLATED_HOME` (no
    // `.claude/settings.json` at all, by design, for every other serve test); overriding it here is
    // `Command::env`'s documented last-write-wins.
    let home = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([stdio_server_config("fixture", json!({}))]),
    );
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__fixture__echo",
        &json!({ "text": "serve-mcp-marker" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let mut child = serve_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"), &base, &session_file)
        .env("HOME", home.path())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "call the echo tool" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let response = frames
        .iter()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .unwrap();
    assert_eq!(
        response["success"], true,
        "the prompt turn must succeed: {frames:?}"
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "expected two model requests");
    assert!(
        bodies[0].contains("\"mcp__fixture__echo\""),
        "the mcp tool must be advertised through serve too: {}",
        bodies[0]
    );
    assert!(
        bodies[1].contains("serve-mcp-marker"),
        "the real MCP server's response must reach the model through serve too: {}",
        bodies[1]
    );
}
