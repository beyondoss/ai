//! A throwaway, hand-rolled MCP server speaking stdio — a test fixture, not a real MCP server. It
//! exists purely so `crates/agent/tests/mcp_*.rs` can exercise the real `tools::mcp` client code
//! against a real subprocess speaking the real wire protocol (newline-delimited JSON-RPC 2.0), instead
//! of mocking the protocol away.
//!
//! Deliberately dependency-free beyond `tokio`/`serde_json` (already ordinary dependencies of this
//! crate) rather than pulling in `rmcp`'s own server-side machinery.
//!
//! Tools:
//! - `echo` / `add` / `ping` / `fail` / `echo_env` / `image` / `slow_progress` — see prior docs.
//! - `ask_user`: nested `elicitation/create` mid-`tools/call` — classic server→client elicitation.
//! - `ask_user_mrtr`: returns SEP-2322 `input_required` with an elicitation input request; on retry
//!   with `inputResponses` completes — proves MRTR under protocol `2026-07-28`.
//! - `slow_task`: returns SEP-2663 `resultType: "task"`; client polls `tasks/get` until completed
//!   (status messages `task-phase-one` → `task-phase-two` → result `task-done`).
//!
//! Also answers `resources/*`, `prompts/*`, `completion/complete`, `tasks/get` / `tasks/cancel` /
//! `tasks/update`, and `server/discover` (advertising `2026-07-28` + `2025-11-25` + tasks extension).
//! Legacy `initialize` still works for Auto fallback.
//!
//! Env: `MCP_FIXTURE_STARTUP_DELAY_MS`, `MCP_FIXTURE_ORPHAN_PIDFILE` (unchanged).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::io::{Stdin, Stdout};

/// A 1x1 transparent PNG, base64-encoded — small enough to embed literally, real enough to prove image
/// bytes survive the round trip unmodified.
const TINY_PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

const RESOURCE_URI: &str = "fixture://doc";
const RESOURCE_BODY: &str = "fixture-resource-body";

const TASKS_EXTENSION_ID: &str = "io.modelcontextprotocol/tasks";

struct FixtureTask {
    created: Instant,
    complete_after: Duration,
    cancelled: bool,
    poll_interval_ms: u64,
}

fn tasks() -> &'static Mutex<HashMap<String, FixtureTask>> {
    static TASKS: OnceLock<Mutex<HashMap<String, FixtureTask>>> = OnceLock::new();
    TASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn capabilities() -> Value {
    json!({
        "tools": {},
        "resources": {},
        "prompts": {},
        "completions": {},
        "extensions": {
            (TASKS_EXTENSION_ID): {}
        },
    })
}

#[tokio::main]
async fn main() {
    if let Some(delay_ms) = std::env::var("MCP_FIXTURE_STARTUP_DELAY_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
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
            Ok(None) => break,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(method) = request.get("method").and_then(Value::as_str) else {
            continue;
        };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };

        let params = request.get("params").cloned().unwrap_or(Value::Null);

        if method == "tools/call"
            && params.get("name").and_then(Value::as_str) == Some("slow_progress")
        {
            if write_slow_progress(&mut stdout, id, &params).await.is_err() {
                break;
            }
            continue;
        }

        if method == "tools/call" && params.get("name").and_then(Value::as_str) == Some("ask_user")
        {
            if write_ask_user(&mut lines, &mut stdout, id).await.is_err() {
                break;
            }
            continue;
        }

        match handle(method, params) {
            Ok(result) => {
                let envelope = json!({ "jsonrpc": "2.0", "id": id, "result": result });
                if write_json_line(&mut stdout, &envelope).await.is_err() {
                    break;
                }
            }
            Err((code, message)) => {
                let envelope = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": code, "message": message },
                });
                if write_json_line(&mut stdout, &envelope).await.is_err() {
                    break;
                }
            }
        }
    }
}

async fn write_json_line(stdout: &mut Stdout, value: &Value) -> Result<(), std::io::Error> {
    let mut encoded = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    encoded.push(b'\n');
    stdout.write_all(&encoded).await?;
    stdout.flush().await
}

async fn write_slow_progress(
    stdout: &mut Stdout,
    id: Value,
    params: &Value,
) -> Result<(), std::io::Error> {
    let token = params
        .pointer("/_meta/progressToken")
        .cloned()
        .unwrap_or(json!("missing-token"));

    for (progress, message) in [(1.0, "phase-one"), (2.0, "phase-two")] {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {
                "progressToken": token,
                "progress": progress,
                "total": 3.0,
                "message": message,
            }
        });
        write_json_line(stdout, &notification).await?;
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    }

    let result = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{ "type": "text", "text": "progress-done" }],
            "isError": false,
        }
    });
    write_json_line(stdout, &result).await
}

/// Classic nested elicitation: send `elicitation/create`, wait for the client's result, finish the tool.
async fn write_ask_user(
    lines: &mut Lines<BufReader<Stdin>>,
    stdout: &mut Stdout,
    tool_id: Value,
) -> Result<(), std::io::Error> {
    let elicit_id = json!("fixture-elicit-1");
    let elicit_req = json!({
        "jsonrpc": "2.0",
        "id": elicit_id,
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": "What is your name?",
            "requestedSchema": {
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"],
            }
        }
    });
    write_json_line(stdout, &elicit_req).await?;

    let name = loop {
        let line = match lines.next_line().await? {
            Some(line) => line,
            None => return Ok(()),
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if msg.get("id") != Some(&elicit_id) {
            // Ignore unrelated traffic (shouldn't happen mid-call on this fixture).
            continue;
        }
        let action = msg
            .pointer("/result/action")
            .and_then(Value::as_str)
            .unwrap_or("decline");
        if action != "accept" {
            break None;
        }
        break msg
            .pointer("/result/content/name")
            .and_then(Value::as_str)
            .map(str::to_string);
    };

    let text = match name {
        Some(n) => format!("hello-{n}"),
        None => "elicitation-declined".into(),
    };
    let result = json!({
        "jsonrpc": "2.0",
        "id": tool_id,
        "result": {
            "content": [{ "type": "text", "text": text }],
            "isError": false,
        }
    });
    write_json_line(stdout, &result).await
}

fn handle(method: &str, params: Value) -> Result<Value, (i64, String)> {
    match method {
        "server/discover" => Ok(json!({
            "resultType": "complete",
            "supportedVersions": ["2026-07-28", "2025-11-25"],
            "capabilities": capabilities(),
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": "mcp-fixture-stdio-server",
                    "version": "0.0.0",
                }
            }
        })),
        "initialize" => Ok(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": capabilities(),
            "serverInfo": { "name": "mcp-fixture-stdio-server", "version": "0.0.0" },
        })),
        "tools/list" => Ok(json!({ "tools": tool_defs() })),
        "tools/call" => Ok(call_tool(params)),
        "tasks/get" => tasks_get(params),
        "tasks/cancel" => tasks_cancel(params),
        "tasks/update" => {
            // Fixture tasks never enter input_required; ack empty.
            let _ = params;
            Ok(json!({ "resultType": "complete" }))
        }
        "resources/list" => Ok(json!({
            "resources": [{
                "uri": RESOURCE_URI,
                "name": "doc",
                "description": "A fixture text resource.",
                "mimeType": "text/plain",
            }]
        })),
        "resources/read" => {
            let uri = params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if uri != RESOURCE_URI {
                return Err((-32002, format!("unknown resource `{uri}`")));
            }
            Ok(json!({
                "contents": [{
                    "uri": RESOURCE_URI,
                    "mimeType": "text/plain",
                    "text": RESOURCE_BODY,
                }]
            }))
        }
        "prompts/list" => Ok(json!({
            "prompts": [{
                "name": "greet",
                "description": "A fixture prompt that greets `who`.",
                "arguments": [{
                    "name": "who",
                    "description": "Who to greet",
                    "required": true,
                }]
            }]
        })),
        "prompts/get" => {
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if name != "greet" {
                return Err((-32602, format!("unknown prompt `{name}`")));
            }
            let who = params
                .pointer("/arguments/who")
                .and_then(Value::as_str)
                .unwrap_or("world");
            Ok(json!({
                "description": "greet prompt",
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": format!("Please greet {who}.") }
                }]
            }))
        }
        "completion/complete" => {
            let arg = params
                .pointer("/argument/value")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let suggestions: Vec<&str> = ["alice", "alex", "bob"]
                .into_iter()
                .filter(|s| s.starts_with(arg))
                .collect();
            Ok(json!({
                "completion": {
                    "values": suggestions,
                    "total": suggestions.len(),
                    "hasMore": false,
                }
            }))
        }
        // JSON-RPC Method not found — keeps Auto discover→initialize fallback fast for unknown methods.
        other => Err((-32601, format!("Method not found: {other}"))),
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
        {
            "name": "slow_progress",
            "description": "Emits progress notifications, then returns `progress-done`.",
            "inputSchema": { "type": "object", "properties": {} },
        },
        {
            "name": "ask_user",
            "description": "Asks the user their name via elicitation, then greets them.",
            "inputSchema": { "type": "object", "properties": {} },
        },
        {
            "name": "ask_user_mrtr",
            "description": "Asks the user their name via SEP-2322 input_required elicitation (MRTR).",
            "inputSchema": { "type": "object", "properties": {} },
        },
        {
            "name": "slow_task",
            "description": "Returns a SEP-2663 task handle; completes after a short poll with `task-done`.",
            "inputSchema": { "type": "object", "properties": {} },
        },
    ])
}

/// SEP-2322 MRTR: first call returns `input_required`; retry with `inputResponses` completes.
fn call_ask_user_mrtr(params: &Value) -> Value {
    if let Some(responses) = params.get("inputResponses") {
        let action = responses
            .pointer("/name/action")
            .and_then(Value::as_str)
            .unwrap_or("decline");
        let text = if action == "accept" {
            let n = responses
                .pointer("/name/content/name")
                .and_then(Value::as_str)
                .unwrap_or("?");
            format!("mrtr-hello-{n}")
        } else {
            "mrtr-elicitation-declined".into()
        };
        return json!({
            "resultType": "complete",
            "content": [{ "type": "text", "text": text }],
            "isError": false,
        });
    }

    json!({
        "resultType": "input_required",
        "requestState": "fixture-mrtr-state",
        "inputRequests": {
            "name": {
                "method": "elicitation/create",
                "params": {
                    "mode": "form",
                    "message": "What is your name? (MRTR)",
                    "requestedSchema": {
                        "type": "object",
                        "properties": { "name": { "type": "string" } },
                        "required": ["name"],
                    }
                }
            }
        }
    })
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
        "slow_progress" => text_result("progress-done", false),
        "ask_user" => text_result("ask_user should be handled asynchronously", true),
        "ask_user_mrtr" => call_ask_user_mrtr(&params),
        "slow_task" => call_slow_task(),
        other => text_result(&format!("fixture: unknown tool `{other}`"), true),
    }
}

/// SEP-2663: return a task handle; `tasks/get` advances status by wall clock until completed.
fn call_slow_task() -> Value {
    let task_id = {
        let Ok(mut guard) = tasks().lock() else {
            return text_result("tasks lock poisoned", true);
        };
        let task_id = format!("fixture-task-{}", guard.len() + 1);
        guard.insert(
            task_id.clone(),
            FixtureTask {
                created: Instant::now(),
                complete_after: Duration::from_millis(250),
                cancelled: false,
                poll_interval_ms: 40,
            },
        );
        task_id
    };
    json!({
        "resultType": "task",
        "taskId": task_id,
        "status": "working",
        "statusMessage": "task-phase-one",
        "createdAt": "2026-09-02T00:00:00Z",
        "lastUpdatedAt": "2026-09-02T00:00:00Z",
        "ttlMs": null,
        "pollIntervalMs": 40,
    })
}

fn tasks_get(params: Value) -> Result<Value, (i64, String)> {
    let task_id = params
        .get("taskId")
        .and_then(Value::as_str)
        .ok_or((-32602, "tasks/get requires taskId".into()))?
        .to_string();
    let Ok(guard) = tasks().lock() else {
        return Err((-32603, "tasks lock poisoned".into()));
    };
    let Some(task) = guard.get(&task_id) else {
        return Err((-32602, format!("unknown task `{task_id}`")));
    };
    if task.cancelled {
        return Ok(json!({
            "resultType": "complete",
            "taskId": task_id,
            "status": "cancelled",
            "createdAt": "2026-09-02T00:00:00Z",
            "lastUpdatedAt": "2026-09-02T00:00:01Z",
            "ttlMs": null,
            "pollIntervalMs": task.poll_interval_ms,
        }));
    }
    let elapsed = task.created.elapsed();
    if elapsed >= task.complete_after {
        return Ok(json!({
            "resultType": "complete",
            "taskId": task_id,
            "status": "completed",
            "statusMessage": "task-complete",
            "createdAt": "2026-09-02T00:00:00Z",
            "lastUpdatedAt": "2026-09-02T00:00:01Z",
            "ttlMs": null,
            "pollIntervalMs": task.poll_interval_ms,
            "result": {
                "content": [{ "type": "text", "text": "task-done" }],
                "isError": false,
            }
        }));
    }
    // Mid-flight: flip status message after half the wait so the client surfaces progress.
    let status_message = if elapsed >= task.complete_after / 2 {
        "task-phase-two"
    } else {
        "task-phase-one"
    };
    Ok(json!({
        "resultType": "complete",
        "taskId": task_id,
        "status": "working",
        "statusMessage": status_message,
        "createdAt": "2026-09-02T00:00:00Z",
        "lastUpdatedAt": "2026-09-02T00:00:00Z",
        "ttlMs": null,
        "pollIntervalMs": task.poll_interval_ms,
    }))
}

fn tasks_cancel(params: Value) -> Result<Value, (i64, String)> {
    let task_id = params
        .get("taskId")
        .and_then(Value::as_str)
        .ok_or((-32602, "tasks/cancel requires taskId".into()))?
        .to_string();
    let Ok(mut guard) = tasks().lock() else {
        return Err((-32603, "tasks lock poisoned".into()));
    };
    let Some(task) = guard.get_mut(&task_id) else {
        return Err((-32602, format!("unknown task `{task_id}`")));
    };
    task.cancelled = true;
    Ok(json!({ "resultType": "complete" }))
}

fn text_result(text: &str, is_error: bool) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error,
    })
}
