//! A strict SEP-2663 (`io.modelcontextprotocol/tasks`) MCP server over stdio — a test fixture for
//! `crates/agent/tests/mcp_tasks_conformance.rs`, kept apart from `mcp_fixture_stdio_server` so the
//! conformance scenarios can be strict without touching the general-purpose fixture.
//!
//! Strict where the spec puts a MUST on the server, so a client that skips its half fails loudly:
//! - `tools/call` for a task tool, and every `tasks/*` request, is refused with `-32021` unless the
//!   request carries `_meta["io.modelcontextprotocol/clientCapabilities"].extensions` naming the
//!   tasks extension — and always under the legacy `2025-11-25` lifecycle, where the extension is
//!   not defined.
//! - Every task status is derived from how many `tasks/get` polls the task has seen, never from wall
//!   time, so each scenario is deterministic under host load.
//!
//! Every result carries `_meta` (`serverInfo`), as the SEP's reference server's do.
//!
//! Every inbound message is appended to `MCP_TASKS_FIXTURE_LOG` (JSONL: `t_ms`, `method`, `params`)
//! so a test can assert on the client's side of the wire: poll spacing, deduplicated `tasks/update`,
//! `tasks/cancel`, the per-request capability.
//!
//! Tools:
//! - `plain`: an ordinary `CallToolResult` — the client must take either shape.
//! - `poll_task`: completes on the 4th poll; `pollIntervalMs` is 60 at creation and 250 from the
//!   first poll on, so a client honouring the *latest* value is visible in the log.
//! - `lazy_ask_task`: two in-task elicitations (`name`, then `color`). Each answered key stays in
//!   `inputRequests` for three more polls after its `tasks/update` — the eventually-consistent ack the
//!   spec allows — so a client that does not deduplicate keys asks the user twice.
//! - `iserror_task`: `completed` with an `isError: true` tool result (not `failed`).
//! - `vanish_task`: the first poll answers `-32602` (task expired).
//! - `crash_task`: the server process exits on the first poll.
//! - `ttl_task`: `ttlMs: 400`, never leaves `working`.
//! - `blip_task`: over HTTP, polls 2 and 3 have their connection dropped without a response (a
//!   network blip); completes on poll 5 with `blip-done`.
//! - `gated_task`: `working` until the file at `MCP_TASKS_FIXTURE_GATE` exists, then `gated-done`.
//! - `sample_task`: one in-task `sampling/createMessage` (`draft`); completes with `sampled:<text>`.
//! - `ttl_shift_task`: created with `ttlMs: null`; every poll then says `ttlMs: 300`, never finishing.
//!   Only a client honouring the *latest* TTL stops (after 60 polls it completes `ttl-ignored`).
//! - `gated_ttl_task`: `gated_task` with `ttlMs: 1500`; `slow_ttl_task`: with `ttlMs: 4000`.
//! - `drop_update_task`: over HTTP, asks for `name`; the first `tasks/update` has its connection
//!   dropped unapplied; completes with `dropped-update-<name>` once an update lands.
//! - `durable_ask_task`: asks for `name`; after the answer lands, keeps listing `name` for 50 polls
//!   (100 ms apart: an eventually-consistent ack that outlives an agent restart), then completes
//!   with `durable-<name>`.
//! - `crash_once_task`: with `MCP_TASKS_FIXTURE_STATE` set, the task is saved to that file and the
//!   process exits on its first poll; the restarted process (which sleeps
//!   `MCP_TASKS_FIXTURE_RESTART_DELAY_MS` first) still knows the task, which then stays `working`.
//!
//! - `nested_ask` (handled concurrently, stdio or HTTP): after 1.5 s sends a classic nested
//!   `elicitation/create` while its `tools/call` is still open (over HTTP, on that POST's own SSE
//!   stream), then completes with `asked:<the client's response or error>`.
//! - `hold` (handled concurrently): completes with `held` after 4 s, holding a call in flight.
//!
//! Resource `fixture-tasks://doc`: `resources/read` answers a `CreateTaskResult`, which the client
//! MUST treat as an invalid response (tasks are defined for `tools/call` only).
//!
//! Env: `MCP_TASKS_FIXTURE_LOG` (path), `MCP_TASKS_FIXTURE_LEGACY=1` (answer `server/discover` with
//! `-32601`, forcing the client onto legacy `initialize`), `MCP_TASKS_FIXTURE_GATE` (path),
//! `MCP_TASKS_FIXTURE_HTTP_PORT_FILE` (serve Streamable HTTP on `127.0.0.1:<ephemeral>/mcp` instead
//! of stdio, writing the port to this file; each HTTP log line also records the `Mcp-Method` /
//! `Mcp-Name` / `MCP-Protocol-Version` request headers). Over HTTP the server outlives any one
//! client, so a test can restart the agent while a task is in flight.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

const TASKS_EXT: &str = "io.modelcontextprotocol/tasks";
const CAPS_META: &str = "io.modelcontextprotocol/clientCapabilities";
const MISSING_CAPABILITY: i64 = -32021;
/// Not a JSON-RPC code: tells the HTTP transport to drop the connection without answering.
const DROP_CONNECTION: i64 = i64::MIN;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Poll,
    LazyAsk,
    IsError,
    Vanish,
    Crash,
    Ttl,
    Blip,
    Gated,
    Sample,
    TtlShift,
    GatedTtl,
    SlowTtl,
    DropUpdate,
    DurableAsk,
    CrashOnce,
}

struct Task {
    kind: Kind,
    polls: u32,
    /// `lazy_ask_task`: answers received, and the poll count at which each was received.
    answers: HashMap<String, (String, u32)>,
    /// `drop_update_task`: how many `tasks/update`s arrived.
    updates: u32,
}

struct Server {
    started: Instant,
    log: Option<std::fs::File>,
    legacy_only: bool,
    /// Whether the client negotiated through legacy `initialize` (no tasks extension then).
    legacy: bool,
    tasks: HashMap<String, Task>,
    next: u64,
}

type Reply = Result<Value, (i64, String, Option<Value>)>;

fn missing_capability() -> (i64, String, Option<Value>) {
    (
        MISSING_CAPABILITY,
        "Missing required client capability".into(),
        Some(json!({ "requiredCapabilities": { "extensions": { (TASKS_EXT): {} } } })),
    )
}

fn invalid(message: impl Into<String>) -> (i64, String, Option<Value>) {
    (-32602, message.into(), None)
}

impl Server {
    fn log(&mut self, method: &str, params: &Value, headers: Option<&Value>) {
        let t_ms = self.started.elapsed().as_millis() as u64;
        if let Some(file) = self.log.as_mut() {
            let mut line = json!({ "t_ms": t_ms, "method": method, "params": params });
            if let Some(headers) = headers {
                line["headers"] = headers.clone();
            }
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
        }
    }

    fn declares_tasks(&self, params: &Value) -> bool {
        !self.legacy
            && params
                .get("_meta")
                .and_then(|m| m.get(CAPS_META))
                .and_then(|c| c.get("extensions"))
                .and_then(|e| e.get(TASKS_EXT))
                .is_some()
    }

    fn capabilities() -> Value {
        json!({ "tools": {}, "resources": {}, "extensions": { (TASKS_EXT): {} } })
    }

    fn handle(&mut self, method: &str, params: &Value) -> Reply {
        match method {
            "server/discover" if self.legacy_only => Err((-32601, "Method not found".into(), None)),
            "server/discover" => Ok(json!({
                "resultType": "complete",
                "supportedVersions": ["2026-07-28"],
                "capabilities": Self::capabilities(),
                "ttlMs": 0,
                "cacheScope": "private",
                "_meta": { "io.modelcontextprotocol/serverInfo": {
                    "name": "mcp-fixture-tasks-server", "version": "0.0.0" } },
            })),
            "initialize" => {
                self.legacy = true;
                Ok(json!({
                    "protocolVersion": "2025-11-25",
                    "capabilities": { "tools": {}, "resources": {} },
                    "serverInfo": { "name": "mcp-fixture-tasks-server", "version": "0.0.0" },
                }))
            }
            "tools/list" => Ok(json!({ "tools": [
                tool("plain"), tool("poll_task"), tool("lazy_ask_task"), tool("iserror_task"),
                tool("vanish_task"), tool("crash_task"), tool("ttl_task"), tool("blip_task"),
                tool("gated_task"), tool("sample_task"), tool("ttl_shift_task"),
                tool("gated_ttl_task"), tool("drop_update_task"), tool("durable_ask_task"),
                tool("crash_once_task"), tool("nested_ask"), tool("hold"), tool("slow_ttl_task"),
            ] })),
            "resources/list" => Ok(json!({ "resources": [{
                "uri": "fixture-tasks://doc", "name": "doc", "mimeType": "text/plain",
            }] })),
            "resources/read" => {
                if !self.declares_tasks(params) {
                    return Err(missing_capability());
                }
                Ok(self.create(Kind::Poll))
            }
            "tools/call" => self.call(params),
            "tasks/get" | "tasks/update" | "tasks/cancel" => {
                if !self.declares_tasks(params) {
                    return Err(missing_capability());
                }
                let task_id = params
                    .get("taskId")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("taskId is required"))?
                    .to_owned();
                match method {
                    "tasks/get" => self.get(&task_id),
                    "tasks/update" => self.update(&task_id, params),
                    _ => {
                        self.tasks
                            .get(&task_id)
                            .ok_or_else(|| invalid("Failed to cancel task: Task not found"))?;
                        Ok(json!({ "resultType": "complete" }))
                    }
                }
            }
            other => Err((-32601, format!("Method not found: {other}"), None)),
        }
    }

    fn call(&mut self, params: &Value) -> Reply {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let kind = match name {
            "plain" => {
                return Ok(json!({
                    "content": [{ "type": "text", "text": "plain-result" }],
                    "isError": false,
                }));
            }
            "poll_task" => Kind::Poll,
            "lazy_ask_task" => Kind::LazyAsk,
            "iserror_task" => Kind::IsError,
            "vanish_task" => Kind::Vanish,
            "crash_task" => Kind::Crash,
            "ttl_task" => Kind::Ttl,
            "blip_task" => Kind::Blip,
            "gated_task" => Kind::Gated,
            "sample_task" => Kind::Sample,
            "ttl_shift_task" => Kind::TtlShift,
            "gated_ttl_task" => Kind::GatedTtl,
            "slow_ttl_task" => Kind::SlowTtl,
            "drop_update_task" => Kind::DropUpdate,
            "durable_ask_task" => Kind::DurableAsk,
            "crash_once_task" => Kind::CrashOnce,
            other => return Err(invalid(format!("unknown tool `{other}`"))),
        };
        if !self.declares_tasks(params) {
            return Err(missing_capability());
        }
        Ok(self.create(kind))
    }

    fn create(&mut self, kind: Kind) -> Value {
        self.next += 1;
        let task_id = format!(
            "tsk-{:x}-{:x}-{}",
            std::process::id(),
            self.started.elapsed().as_nanos(),
            self.next
        );
        self.tasks.insert(
            task_id.clone(),
            Task {
                kind,
                polls: 0,
                answers: HashMap::new(),
                updates: 0,
            },
        );
        if kind == Kind::CrashOnce {
            save_state(&task_id);
        }
        // A long interval for `ttl_task`, so only a client that caps its wait by the TTL notices
        // the TTL on time.
        let interval = if kind == Kind::Ttl { 10_000 } else { 60 };
        let mut created = task_json(&task_id, "working", ttl_for(kind), interval);
        created["resultType"] = json!("task");
        created["statusMessage"] = json!("created");
        created
    }

    fn get(&mut self, task_id: &str) -> Reply {
        let task = self
            .tasks
            .get_mut(task_id)
            .ok_or_else(|| invalid("Failed to retrieve task: Task not found"))?;
        task.polls += 1;
        let polls = task.polls;
        let ttl = ttl_for(task.kind);
        let mut v = match task.kind {
            Kind::Crash => std::process::exit(3),
            Kind::Vanish => {
                return Err(invalid("Failed to retrieve task: Task has expired"));
            }
            Kind::Ttl => task_json(task_id, "working", ttl, 10_000),
            // Gives up after 60 polls (3 s) so a client ignoring the new TTL fails, not hangs.
            Kind::TtlShift if polls >= 60 => completed(task_id, json!(300), "ttl-ignored"),
            Kind::TtlShift => task_json(task_id, "working", json!(300), 50),
            Kind::CrashOnce if polls == 1 && std::env::var("MCP_TASKS_FIXTURE_STATE").is_ok() => {
                std::process::exit(3)
            }
            Kind::CrashOnce => task_json(task_id, "working", ttl, 50),
            Kind::GatedTtl | Kind::SlowTtl => {
                if gate_open() {
                    completed(task_id, ttl, "gated-done")
                } else {
                    task_json(task_id, "working", ttl, 50)
                }
            }
            Kind::DropUpdate => match task.answers.get("name") {
                Some((name, _)) => completed(task_id, ttl, &format!("dropped-update-{name}")),
                None => ask(
                    task_id,
                    ttl,
                    "name",
                    "What is your name? (dropped update)",
                    30,
                ),
            },
            Kind::DurableAsk => match task.answers.get("name") {
                Some((name, at)) if polls > at + 50 => {
                    completed(task_id, ttl, &format!("durable-{name}"))
                }
                _ => ask(task_id, ttl, "name", "What is your name? (durable)", 100),
            },
            Kind::Blip if polls == 2 || polls == 3 => {
                return Err((DROP_CONNECTION, String::new(), None));
            }
            Kind::Blip if polls < 5 => task_json(task_id, "working", ttl, 40),
            Kind::Blip => completed(task_id, ttl, "blip-done"),
            Kind::Gated => {
                if gate_open() {
                    completed(task_id, ttl, "gated-done")
                } else {
                    task_json(task_id, "working", ttl, 50)
                }
            }
            Kind::Sample => match task.answers.get("draft") {
                Some((text, _)) => completed(task_id, ttl, &format!("sampled:{text}")),
                None => {
                    let mut v = task_json(task_id, "input_required", ttl, 30);
                    v["inputRequests"] = json!({ "draft": {
                        "method": "sampling/createMessage",
                        "params": {
                            "messages": [{
                                "role": "user",
                                "content": { "type": "text", "text": "Write a one-word draft. (task)" },
                            }],
                            "maxTokens": 32,
                        }
                    } });
                    v
                }
            },
            Kind::IsError => {
                let mut v = task_json(task_id, "completed", ttl, 60);
                v["result"] = json!({
                    "content": [{ "type": "text", "text": "tool-level-error" }],
                    "isError": true,
                });
                v
            }
            Kind::Poll if polls < 4 => {
                let mut v = task_json(task_id, "working", ttl, 250);
                v["statusMessage"] = json!(format!("step-{polls}"));
                v
            }
            Kind::Poll => {
                let mut v = task_json(task_id, "completed", ttl, 250);
                v["result"] = json!({
                    "content": [{ "type": "text", "text": "poll-done" }],
                    "isError": false,
                });
                v
            }
            Kind::LazyAsk => {
                // An answered key stays outstanding for three more polls: the ack is eventually
                // consistent, so the client sees the same key again and must not re-ask.
                let settled =
                    |key: &str| matches!(task.answers.get(key), Some((_, at)) if polls > at + 3);
                let ask = if !settled("name") {
                    Some(("name", "What is your name? (task)"))
                } else if !settled("color") {
                    Some(("color", "What is your favourite color? (task)"))
                } else {
                    None
                };
                match ask {
                    Some((key, message)) => {
                        let mut v = task_json(task_id, "input_required", ttl, 30);
                        v["inputRequests"] = json!({ (key): {
                            "method": "elicitation/create",
                            "params": {
                                "mode": "form",
                                "message": message,
                                "requestedSchema": {
                                    "type": "object",
                                    "properties": { (key): { "type": "string" } },
                                    "required": [key],
                                }
                            }
                        } });
                        v
                    }
                    None => {
                        let name = &task.answers["name"].0;
                        let color = &task.answers["color"].0;
                        let mut v = task_json(task_id, "completed", ttl, 30);
                        v["result"] = json!({
                            "content": [{ "type": "text", "text": format!("lazy-{name}-{color}") }],
                            "isError": false,
                        });
                        v
                    }
                }
            }
        };
        v["resultType"] = json!("complete");
        Ok(v)
    }

    fn update(&mut self, task_id: &str, params: &Value) -> Reply {
        let task = self
            .tasks
            .get_mut(task_id)
            .ok_or_else(|| invalid("Failed to update task: Task not found"))?;
        let polls = task.polls;
        task.updates += 1;
        if task.kind == Kind::DropUpdate && task.updates == 1 {
            return Err((DROP_CONNECTION, String::new(), None));
        }
        if let Some(responses) = params.get("inputResponses").and_then(Value::as_object) {
            let known: HashSet<&str> = ["name", "color", "draft"].into_iter().collect();
            for (key, response) in responses {
                // Ignore keys not outstanding (never issued, or already answered), as the spec says.
                if !known.contains(key.as_str()) || task.answers.contains_key(key) {
                    continue;
                }
                // An elicitation answer carries `content.<key>`; a sampling one `content.text`.
                let field = if key == "draft" { "text" } else { key.as_str() };
                let value = response
                    .pointer(&format!("/content/{field}"))
                    .and_then(Value::as_str)
                    .unwrap_or("declined")
                    .to_owned();
                task.answers.insert(key.clone(), (value, polls));
            }
        }
        Ok(json!({ "resultType": "complete" }))
    }
}

fn ttl_for(kind: Kind) -> Value {
    match kind {
        Kind::Ttl => json!(400),
        Kind::GatedTtl => json!(1500),
        Kind::SlowTtl => json!(4000),
        _ => Value::Null,
    }
}

fn gate_open() -> bool {
    std::env::var("MCP_TASKS_FIXTURE_GATE")
        .map(|gate| std::path::Path::new(&gate).exists())
        .unwrap_or(false)
}

/// An `input_required` task asking one elicitation `key`.
fn ask(task_id: &str, ttl: Value, key: &str, message: &str, interval: u64) -> Value {
    let mut v = task_json(task_id, "input_required", ttl, interval);
    v["inputRequests"] = json!({ (key): {
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": message,
            "requestedSchema": {
                "type": "object",
                "properties": { (key): { "type": "string" } },
                "required": [key],
            }
        }
    } });
    v
}

/// `crash_once_task`: remember the task across a process restart.
fn save_state(task_id: &str) {
    if let Ok(path) = std::env::var("MCP_TASKS_FIXTURE_STATE") {
        let _ = std::fs::write(path, task_id);
    }
}

/// The `crash_once_task` a previous process saved, already past its crash.
fn load_state(tasks: &mut HashMap<String, Task>) -> bool {
    let Ok(path) = std::env::var("MCP_TASKS_FIXTURE_STATE") else {
        return false;
    };
    let Ok(task_id) = std::fs::read_to_string(path) else {
        return false;
    };
    tasks.insert(
        task_id,
        Task {
            kind: Kind::CrashOnce,
            polls: 1,
            answers: HashMap::new(),
            updates: 0,
        },
    );
    true
}

fn completed(task_id: &str, ttl: Value, text: &str) -> Value {
    let mut v = task_json(task_id, "completed", ttl, 30);
    v["result"] = json!({ "content": [{ "type": "text", "text": text }], "isError": false });
    v
}

fn tool(name: &str) -> Value {
    json!({
        "name": name,
        "description": format!("SEP-2663 conformance fixture tool `{name}`."),
        "inputSchema": { "type": "object", "properties": {} },
    })
}

fn task_json(task_id: &str, status: &str, ttl: Value, poll_interval_ms: u64) -> Value {
    json!({
        "taskId": task_id,
        "status": status,
        "createdAt": "2026-10-06T00:00:00Z",
        "lastUpdatedAt": "2026-10-06T00:00:00Z",
        "ttlMs": ttl,
        "pollIntervalMs": poll_interval_ms,
    })
}

/// Outstanding nested requests this server sent, by request id, awaiting the client's response.
fn waiters() -> &'static Mutex<HashMap<String, tokio::sync::oneshot::Sender<Value>>> {
    static WAITERS: std::sync::OnceLock<
        Mutex<HashMap<String, tokio::sync::oneshot::Sender<Value>>>,
    > = std::sync::OnceLock::new();
    WAITERS.get_or_init(Default::default)
}

/// A JSON-RPC response from the client: hand it to whoever sent the request.
fn deliver(msg: &Value) {
    let Some(id) = msg.get("id").and_then(Value::as_str) else {
        return;
    };
    let answer = msg
        .get("result")
        .or_else(|| msg.get("error"))
        .cloned()
        .unwrap_or(Value::Null);
    if let Some(tx) = waiters().lock().ok().and_then(|mut w| w.remove(id)) {
        let _ = tx.send(answer);
    }
}

/// The tools answered concurrently, off the request loop.
fn concurrent(params: &Value) -> Option<&'static str> {
    match params.get("name").and_then(Value::as_str) {
        Some("nested_ask") => Some("nested_ask"),
        Some("hold") => Some("hold"),
        _ => None,
    }
}

fn text_reply(id: &Value, text: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": {
        "content": [{ "type": "text", "text": text }], "isError": false,
        "_meta": { "io.modelcontextprotocol/serverInfo": {
            "name": "mcp-fixture-tasks-server", "version": "0.0.0" } },
    } })
}

/// Run one concurrent tool: `send` delivers a message to the client on the call's channel.
async fn run_concurrent(tool: &str, id: Value, send: impl Fn(Value)) {
    match tool {
        "hold" => {
            tokio::time::sleep(std::time::Duration::from_secs(4)).await;
            send(text_reply(&id, "held"));
        }
        _ => {
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            let eid = format!("nested-{id}");
            let (tx, rx) = tokio::sync::oneshot::channel();
            if let Ok(mut w) = waiters().lock() {
                w.insert(eid.clone(), tx);
            }
            send(
                json!({ "jsonrpc": "2.0", "id": eid, "method": "elicitation/create", "params": {
                "mode": "form",
                "message": "nested_ask wants a name",
                "requestedSchema": {
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"],
                },
            } }),
            );
            let answer = tokio::time::timeout(std::time::Duration::from_secs(30), rx)
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or(json!("no answer"));
            send(text_reply(&id, &format!("asked:{answer}")));
        }
    }
}

fn envelope(server: &mut Server, id: Value, method: &str, params: &Value) -> Option<Value> {
    Some(match server.handle(method, params) {
        Ok(mut result) => {
            // Every result carries `_meta`, as real servers' do (the SEP's reference server stamps
            // `serverInfo` on each); a client must not mistake a `_meta`-bearing ack for some other
            // result shape.
            if result.get("_meta").is_none() {
                result["_meta"] = json!({ "io.modelcontextprotocol/serverInfo": {
                    "name": "mcp-fixture-tasks-server", "version": "0.0.0" } });
            }
            json!({ "jsonrpc": "2.0", "id": id, "result": result })
        }
        Err((DROP_CONNECTION, _, _)) => return None,
        Err((code, message, data)) => {
            let mut error = json!({ "code": code, "message": message });
            if let Some(data) = data {
                error["data"] = data;
            }
            json!({ "jsonrpc": "2.0", "id": id, "error": error })
        }
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut tasks = HashMap::new();
    if load_state(&mut tasks)
        && let Some(ms) = std::env::var("MCP_TASKS_FIXTURE_RESTART_DELAY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
    {
        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
    }
    let server = Server {
        started: Instant::now(),
        log: std::env::var("MCP_TASKS_FIXTURE_LOG").ok().and_then(|p| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()
        }),
        legacy_only: std::env::var("MCP_TASKS_FIXTURE_LEGACY").as_deref() == Ok("1"),
        legacy: false,
        tasks,
        next: 0,
    };
    match std::env::var("MCP_TASKS_FIXTURE_HTTP_PORT_FILE") {
        Ok(port_file) => serve_http(Arc::new(Mutex::new(server)), &port_file).await,
        Err(_) => serve_stdio(server).await,
    }
}

async fn serve_stdio(server: Server) {
    let server = Arc::new(Mutex::new(server));
    let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(msg) = out_rx.recv().await {
            let mut bytes = serde_json::to_vec(&msg).unwrap_or_default();
            bytes.push(b'\n');
            if stdout.write_all(&bytes).await.is_err() || stdout.flush().await.is_err() {
                break;
            }
        }
    });
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(method) = msg.get("method").and_then(Value::as_str) else {
            deliver(&msg);
            continue;
        };
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        if let Ok(mut s) = server.lock() {
            s.log(method, &params, None);
        }
        let Some(id) = msg.get("id").cloned() else {
            continue; // a notification
        };
        if method == "tools/call"
            && let Some(tool) = concurrent(&params)
        {
            let out = out_tx.clone();
            tokio::spawn(async move {
                run_concurrent(tool, id, move |m| {
                    let _ = out.send(m);
                })
                .await;
            });
            continue;
        }
        let reply = match server.lock() {
            Ok(mut s) => envelope(&mut s, id, method, &params),
            Err(_) => None,
        };
        if let Some(reply) = reply {
            let _ = out_tx.send(reply);
        }
    }
    drop(out_tx);
    let _ = writer.await;
}

/// Minimal Streamable HTTP: one JSON-RPC message per `POST /mcp`, answered as `application/json`
/// (or `202` for a notification), one request per connection. No SSE stream: `GET` is `405`, which
/// the transport allows for a server that never pushes.
async fn serve_http(server: Arc<Mutex<Server>>, port_file: &str) {
    let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:0").await else {
        return;
    };
    let Ok(addr) = listener.local_addr() else {
        return;
    };
    // Written to a temp name and renamed, so a reader never sees a half-written port.
    let tmp = format!("{port_file}.tmp");
    if std::fs::write(&tmp, addr.port().to_string()).is_err()
        || std::fs::rename(&tmp, port_file).is_err()
    {
        return;
    }
    while let Ok((sock, _)) = listener.accept().await {
        tokio::spawn(http_exchange(sock, server.clone()));
    }
}

async fn http_exchange(mut sock: tokio::net::TcpStream, server: Arc<Mutex<Server>>) {
    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        match sock.read_buf(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let verb = lines
        .next()
        .and_then(|l| l.split(' ').next())
        .unwrap_or("")
        .to_owned();
    let headers: HashMap<String, String> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        match sock.read_buf(&mut body).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
    let respond = |status: &str, body: &[u8]| {
        let mut out = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    };
    if verb != "POST" {
        let _ = sock
            .write_all(&respond("405 Method Not Allowed", b""))
            .await;
        return;
    }
    let Ok(msg) = serde_json::from_slice::<Value>(&body[..len]) else {
        let _ = sock.write_all(&respond("400 Bad Request", b"")).await;
        return;
    };
    let method = msg
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let logged = json!({
        "mcp-method": headers.get("mcp-method"),
        "mcp-name": headers.get("mcp-name"),
        "mcp-protocol-version": headers.get("mcp-protocol-version"),
    });
    if let Ok(mut s) = server.lock() {
        s.log(&method, &params, Some(&logged));
    }
    if method.is_empty() {
        // The client answering a nested request this server sent.
        deliver(&msg);
        let _ = sock.write_all(&respond("202 Accepted", b"")).await;
        return;
    }
    if method == "tools/call"
        && let Some(tool) = concurrent(&params)
    {
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        if tool == "hold" {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            run_concurrent(tool, id, move |m| {
                let _ = tx.send(m);
            })
            .await;
            if let Some(reply) = rx.recv().await {
                let body = serde_json::to_vec(&reply).unwrap_or_default();
                let _ = sock.write_all(&respond("200 OK", &body)).await;
            }
            return;
        }
        // The nested request rides this POST's own SSE stream, then the result closes it.
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if sock.write_all(head.as_bytes()).await.is_err() {
            return;
        }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
        let work = tokio::spawn(run_concurrent(tool, id, move |m| {
            let _ = tx.send(m);
        }));
        while let Some(m) = rx.recv().await {
            let event = format!("event: message\ndata: {m}\n\n");
            if sock.write_all(event.as_bytes()).await.is_err() || sock.flush().await.is_err() {
                break;
            }
        }
        let _ = work.await;
        let _ = sock.shutdown().await;
        return;
    }
    let reply = {
        let Ok(mut server) = server.lock() else {
            return;
        };
        match msg.get("id").cloned() {
            None => Some(None), // a notification or a response: accepted, no body
            Some(id) => envelope(&mut server, id, &method, &params).map(Some),
        }
    };
    let bytes = match reply {
        None => return, // DROP_CONNECTION: close without answering
        Some(None) => respond("202 Accepted", b""),
        Some(Some(envelope)) => {
            respond("200 OK", &serde_json::to_vec(&envelope).unwrap_or_default())
        }
    };
    let _ = sock.write_all(&bytes).await;
    let _ = sock.shutdown().await;
}
