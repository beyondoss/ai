//! A throwaway, hand-rolled MCP server speaking stdio — a test fixture, not a real MCP server. It
//! exists purely so `crates/agent/tests/mcp_client.rs` can exercise the real `tools::mcp` client code
//! against a real subprocess speaking the real wire protocol (newline-delimited JSON-RPC 2.0), instead
//! of mocking the protocol away.
//!
//! Deliberately dependency-free beyond `tokio`/`serde_json` (already ordinary dependencies of this
//! crate) rather than pulling in `rmcp`'s own server-side machinery: this only ever needs to answer the
//! handful of requests one client makes in a test (`initialize`, `notifications/initialized`,
//! `tools/list`, `tools/call`), so hand-framing them is simpler than standing up a second, heavier
//! dependency surface just for a test double.
//!
//! Six tools, each proving a distinct thing the real `tools::mcp` client code must handle correctly:
//! - `echo`: a one-required-string-argument tool — the ordinary case.
//! - `add`: two-number arguments — proves a richer input schema round-trips.
//! - `ping`: an empty input schema, called with *no* arguments — proves the `Value::Null` input path.
//! - `fail`: always returns `isError: true` — proves error propagation into `ToolError::Execution`.
//! - `echo_env`: returns the value of an env var read from *this process's own environment* — proves
//!   `McpServerConfig`'s configured `env` actually reaches the spawned child.
//! - `image`: returns an `ImageContent` block (a tiny embedded PNG) — proves image content maps to
//!   `ToolOutput::images`, not just text.
//!
//! Also honors `MCP_FIXTURE_STARTUP_DELAY_MS` (an env var, so it's set per-server via `McpServerConfig`'s
//! own `env` map) — sleeps that long before doing anything else, purely so a test can prove
//! `tools::mcp::connect_all` actually connects to multiple configured servers *concurrently* rather than
//! one after another (start N of these with the same delay; total connect time should track the delay
//! once, not N times).
//!
//! Also honors `MCP_FIXTURE_ORPHAN_PIDFILE`: leaves behind a long-lived process that has re-parented to
//! init, and writes its pid to that path. This reproduces what a real browser-driving MCP server does —
//! `rustwright-mcp` double-forks Chromium, so killing the server alone stranded 16 processes holding
//! 322 MB — and lets a test prove the reaper's process-group sweep actually catches such a grandchild
//! rather than only the child it can see.

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// A 1x1 transparent PNG, base64-encoded — small enough to embed literally, real enough to prove image
/// bytes survive the round trip unmodified.
const TINY_PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

#[tokio::main]
async fn main() {
    if let Some(delay_ms) = std::env::var("MCP_FIXTURE_STARTUP_DELAY_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
    // A grandchild that outlives its parent: `sh` backgrounds `sleep` and exits immediately, so the
    // sleep re-parents to init while *staying in this process's group* — exactly Chromium's shape under
    // a double-forking MCP server. Not `setsid`, which would leave the group too and is a different
    // (unsweepable-by-signal) problem.
    if let Ok(pidfile) = std::env::var("MCP_FIXTURE_ORPHAN_PIDFILE") {
        let script = format!("sleep 600 & echo $! > {pidfile}");
        let _ = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .status()
            .await;
    }

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break, // EOF: the client closed the pipe (graceful shutdown).
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            continue; // Not valid JSON — nothing sensible to respond with; skip the line.
        };
        let Some(method) = request.get("method").and_then(Value::as_str) else {
            continue; // A response, not a request/notification — this fixture never sends requests.
        };
        // A notification (no `id`) never gets a reply, per JSON-RPC 2.0.
        let Some(id) = request.get("id").cloned() else {
            continue;
        };

        let response = handle(
            method,
            request.get("params").cloned().unwrap_or(Value::Null),
        );
        let envelope = json!({ "jsonrpc": "2.0", "id": id, "result": response });
        let Ok(mut encoded) = serde_json::to_vec(&envelope) else {
            continue;
        };
        encoded.push(b'\n');
        if stdout.write_all(&encoded).await.is_err() || stdout.flush().await.is_err() {
            break; // The client hung up; nothing left to do.
        }
    }
}

fn handle(method: &str, params: Value) -> Value {
    match method {
        "initialize" => json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "mcp-fixture-stdio-server", "version": "0.0.0" },
        }),
        "tools/list" => json!({ "tools": tool_defs() }),
        "tools/call" => call_tool(params),
        // Anything else this fixture doesn't understand: a JSON-RPC-shaped error result. None of the
        // e2e tests are expected to trigger this — it's here so an unexpected request fails loudly
        // (a clear tool-result-shaped error) rather than the fixture silently hanging.
        other => {
            json!({ "content": [{ "type": "text", "text": format!("fixture: unhandled method `{other}`") }], "isError": true })
        }
    }
}

fn tool_defs() -> Value {
    json!([
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
            "name": "add",
            "description": "Adds two numbers, `a` and `b`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "ping",
            "description": "Takes no arguments; always replies `pong`.",
            "inputSchema": { "type": "object", "properties": {} },
        },
        {
            "name": "fail",
            "description": "Always fails, with an error message — for testing error propagation.",
            "inputSchema": { "type": "object", "properties": {} },
        },
        {
            "name": "echo_env",
            "description": "Returns the value of the env var named by `var`, in this server process's own environment.",
            "inputSchema": {
                "type": "object",
                "properties": { "var": { "type": "string" } },
                "required": ["var"],
            },
        },
        {
            "name": "image",
            "description": "Returns a tiny embedded PNG as image content.",
            "inputSchema": { "type": "object", "properties": {} },
        },
    ])
}

fn call_tool(params: Value) -> Value {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

    match name {
        "echo" => {
            let text = arguments
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            text_result(text, false)
        }
        "add" => {
            let a = arguments.get("a").and_then(Value::as_f64).unwrap_or(0.0);
            let b = arguments.get("b").and_then(Value::as_f64).unwrap_or(0.0);
            text_result(&(a + b).to_string(), false)
        }
        "ping" => text_result("pong", false),
        "fail" => text_result("intentional failure from fixture server", true),
        "echo_env" => {
            let var = arguments
                .get("var")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let value = std::env::var(var).unwrap_or_default();
            text_result(&value, false)
        }
        "image" => json!({
            "content": [{ "type": "image", "data": TINY_PNG_BASE64, "mimeType": "image/png" }],
            "isError": false,
        }),
        other => text_result(&format!("fixture: unknown tool `{other}`"), true),
    }
}

fn text_result(text: &str, is_error: bool) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error,
    })
}
