//! SEP-2663 (`io.modelcontextprotocol/tasks`) client conformance, against the strict
//! `mcp_fixture_tasks_server` — which refuses anything a spec MUST says the client has to send, and
//! logs every inbound message so the client's side of the wire can be asserted on directly.
//!
//! Sibling of `mcp_tasks.rs` (happy path, failed/cancelled, abort → `tasks/cancel`); this file holds
//! the spec clauses that suite could not see: per-request capability, the latest `pollIntervalMs`,
//! `inputRequests` deduplication across an eventually-consistent ack, `isError` vs `failed`,
//! task-not-found, server loss mid-task, the `ttlMs` backstop, legacy lifecycle, a task answered to an
//! unsupported request, and the same flow through `run` and a subagent.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const TASKS_EXT: &str = "io.modelcontextprotocol/tasks";

struct Env {
    dir: tempfile::TempDir,
    home: PathBuf,
    log: PathBuf,
}

impl Env {
    fn new(extra_env: Value) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let log = dir.path().join("wire.jsonl");
        let mut env = json!({ "MCP_TASKS_FIXTURE_LOG": log.to_string_lossy() });
        for (k, v) in extra_env.as_object().unwrap() {
            env[k] = v.clone();
        }
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        std::fs::write(
            home.join(".claude/settings.json"),
            json!({ "mcp_servers": [{
                "name": "t",
                "transport": "stdio",
                "command": env!("CARGO_BIN_EXE_mcp_fixture_tasks_server"),
                "args": [],
                "env": env,
            }] })
            .to_string(),
        )
        .unwrap();
        Self { dir, home, log }
    }

    fn session_file(&self) -> String {
        self.dir
            .path()
            .join("s.jsonl")
            .to_string_lossy()
            .into_owned()
    }

    /// Every message the fixture received, in order.
    fn wire(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    fn methods(&self, method: &str) -> Vec<Value> {
        self.wire()
            .into_iter()
            .filter(|m| m["method"] == method)
            .collect()
    }

    /// Run one prompt through `serve`, answering nothing interactively.
    fn serve_prompt(&self, turns: Vec<String>) -> Vec<Value> {
        let (base, _bodies) = spawn_model_server(turns);
        let mut cmd = serve_cmd(common::BIN, &base, &self.session_file());
        cmd.env("HOME", &self.home)
            .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
        let mut child = cmd.spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "prompt");
        drop(stdin);
        child.wait().unwrap();
        frames
    }
}

fn tool_ends(frames: &[Value]) -> Vec<&Value> {
    frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .map(|f| &f["event"])
        .collect()
}

fn only_tool_end(frames: &[Value]) -> &Value {
    let ends = tool_ends(frames);
    assert_eq!(ends.len(), 1, "expected exactly one tool_end: {frames:?}");
    ends[0]
}

fn declares_tasks(params: &Value) -> bool {
    params
        .pointer("/_meta/io.modelcontextprotocol~1clientCapabilities/extensions")
        .and_then(|e| e.get(TASKS_EXT))
        .is_some()
}

/// Headless `run`: an ordinary result and a task handle in the same session (the client MUST take
/// either shape), the per-request capability on every task-related request, and the *latest*
/// `pollIntervalMs` honoured (60 ms at creation, 250 ms from the first poll on).
#[test]
fn run_handles_both_shapes_declares_capability_and_honours_latest_poll_interval() {
    let env = Env::new(json!({}));
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_plain", "mcp__t__plain", "{}"),
        turn_tool_use("toolu_poll", "mcp__t__poll_task", "{}"),
        turn_text("done"),
    ]);
    let mut child = run_cmd(common::BIN)
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
            "4",
            "--no-session-persistence",
        ])
        .env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .current_dir(env.dir.path())
        .stdin(Stdio::null()) // `run` reads a piped stdin as part of the prompt
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    let status = child.wait().unwrap();
    assert!(status.success(), "run failed: {stdout}");

    let requests = bodies.lock().unwrap().clone();
    assert_eq!(requests.len(), 3, "three model turns: {requests:#?}");
    assert!(
        requests[1].contains("plain-result"),
        "plain CallToolResult reaches the model"
    );
    assert!(
        requests[2].contains("poll-done"),
        "completed task result reaches the model: {}",
        requests[2]
    );

    let call = env
        .methods("tools/call")
        .into_iter()
        .find(|m| m["params"]["name"] == "poll_task")
        .expect("tools/call poll_task");
    assert!(
        declares_tasks(&call["params"]),
        "tools/call must carry the per-request tasks capability: {call}"
    );
    assert!(
        call["params"].get("task").is_none(),
        "the legacy per-request `task` opt-in is not part of the extension: {call}"
    );
    let gets = env.methods("tasks/get");
    assert_eq!(gets.len(), 4, "completes on the 4th poll: {gets:?}");
    assert!(
        gets.iter().all(|g| declares_tasks(&g["params"])),
        "{gets:?}"
    );
    let t = |m: &Value| m["t_ms"].as_u64().unwrap();
    assert!(
        t(&gets[0]) >= t(&call) + 55,
        "first poll waits the creation pollIntervalMs (60): call {} get {}",
        t(&call),
        t(&gets[0])
    );
    for pair in gets.windows(2) {
        assert!(
            t(&pair[1]) >= t(&pair[0]) + 240,
            "later polls wait the updated pollIntervalMs (250): {pair:?}"
        );
    }
    assert!(
        env.methods("tasks/cancel").is_empty(),
        "a completed task is not cancelled"
    );
}

/// The ack to `tasks/update` is eventually consistent: the fixture keeps each answered key in
/// `inputRequests` for three more polls. The user must be asked once per key, and each key answered
/// in exactly one `tasks/update`; a new key after that is still asked.
#[test]
fn serve_deduplicates_input_requests_across_polls() {
    let env = Env::new(json!({}));
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_lazy", "mcp__t__lazy_ask_task", "{}"),
        turn_text("done"),
    ]);
    let mut cmd = serve_cmd(common::BIN, &base, &env.session_file());
    cmd.env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();

    let mut asked = Vec::new();
    let mut frames = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut line = String::new();
    loop {
        assert!(Instant::now() < deadline, "timed out: asked={asked:?}");
        line.clear();
        if stdout.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(frame) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if frame["type"] == "elicitation_request" {
            let message = frame
                .pointer("/params/message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let (key, value) = if message.contains("color") {
                ("color", "teal")
            } else {
                ("name", "Ferris")
            };
            asked.push(key);
            writeln!(
                stdin,
                "{}",
                json!({
                    "type": "elicit",
                    "request_id": frame["request_id"],
                    "action": "accept",
                    "content": { (key): value },
                })
            )
            .unwrap();
            stdin.flush().unwrap();
        }
        let done = frame["type"] == "response" && frame["command"] == "prompt";
        frames.push(frame);
        if done {
            break;
        }
    }
    drop(stdin);
    child.wait().unwrap();

    assert_eq!(
        asked,
        ["name", "color"],
        "each inputRequests key is shown to the user exactly once"
    );
    let end = only_tool_end(&frames);
    assert_eq!(end["is_error"], false, "{end}");
    assert!(
        end["result"].as_str().unwrap().contains("lazy-Ferris-teal"),
        "{end}"
    );
    let updates: Vec<Vec<String>> = env
        .methods("tasks/update")
        .iter()
        .map(|u| {
            u["params"]["inputResponses"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        })
        .collect();
    assert_eq!(
        updates,
        vec![vec!["name".to_owned()], vec!["color".to_owned()]],
        "one tasks/update per key, never a re-sent answer"
    );
}

/// `completed` with an `isError: true` result is a tool-level error, not a task failure.
#[test]
fn serve_completed_task_with_is_error_is_a_tool_error_not_a_failure() {
    let env = Env::new(json!({}));
    let frames = env.serve_prompt(vec![
        turn_tool_use("toolu_e", "mcp__t__iserror_task", "{}"),
        turn_text("done"),
    ]);
    let end = only_tool_end(&frames);
    assert_eq!(end["is_error"], true, "{end}");
    let text = end["result"].as_str().unwrap();
    assert!(text.contains("tool-level-error"), "{end}");
    assert!(
        !text.contains("mcp task"),
        "isError is not the failed status: {end}"
    );
}

/// A `-32602` from `tasks/get` (task expired / purged) ends the call with an error naming the cause.
#[test]
fn serve_task_not_found_mid_poll_ends_the_call_with_an_error() {
    let env = Env::new(json!({}));
    let frames = env.serve_prompt(vec![
        turn_tool_use("toolu_v", "mcp__t__vanish_task", "{}"),
        turn_text("done"),
    ]);
    let end = only_tool_end(&frames);
    assert_eq!(end["is_error"], true, "{end}");
    assert!(
        end["result"].as_str().unwrap().contains("Task has expired"),
        "{end}"
    );
    assert_eq!(env.methods("tasks/get").len(), 1, "no retry loop on -32602");
}

/// The server process dies mid-task: the call ends promptly with an error rather than hanging.
#[test]
fn serve_server_loss_mid_task_ends_the_call_promptly() {
    let env = Env::new(json!({}));
    let start = Instant::now();
    let frames = env.serve_prompt(vec![
        turn_tool_use("toolu_x", "mcp__t__crash_task", "{}"),
        turn_text("done"),
    ]);
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "server loss must not hang the call: {:?}",
        start.elapsed()
    );
    let end = only_tool_end(&frames);
    assert_eq!(end["is_error"], true, "{end}");
    // The dead client was dropped and the server redialed: the error is the *new* process's
    // answer for a task it never had, not a give-up after repeated closed-transport failures.
    assert!(
        end["result"].as_str().unwrap().contains("Task not found"),
        "the client must redial a dead stdio server: {end}"
    );
    assert!(
        env.methods("server/discover").len() >= 2,
        "a second handshake (the redial): {:?}",
        env.wire()
    );
}

/// A task that outlives its `ttlMs` is treated as no longer usable: the call errors and the client
/// sends a best-effort `tasks/cancel`. The server suggests 10 s between polls; the wait is capped by
/// the TTL's remainder, so the 400 ms TTL is noticed on time.
#[test]
fn serve_ttl_backstop_ends_a_task_that_outlives_its_ttl() {
    let env = Env::new(json!({}));
    let start = Instant::now();
    let frames = env.serve_prompt(vec![
        turn_tool_use("toolu_ttl", "mcp__t__ttl_task", "{}"),
        turn_text("done"),
    ]);
    assert!(
        start.elapsed() < Duration::from_secs(6),
        "the TTL caps the wait, not the 10 s poll interval: {:?}",
        start.elapsed()
    );
    let end = only_tool_end(&frames);
    assert_eq!(end["is_error"], true, "{end}");
    assert!(end["result"].as_str().unwrap().contains("ttlMs"), "{end}");
    // The cancel is sent from a drop guard; serve has already exited, so it either landed or not —
    // poll the log briefly rather than racing it.
    let deadline = Instant::now() + Duration::from_secs(3);
    while env.methods("tasks/cancel").is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    let cancels = env.methods("tasks/cancel");
    assert_eq!(cancels.len(), 1, "one tasks/cancel: {:?}", env.wire());
    assert!(declares_tasks(&cancels[0]["params"]));
    assert!(
        env.methods("notifications/cancelled").is_empty(),
        "notifications/cancelled MUST NOT be used to cancel a task"
    );
}

/// Under the legacy `2025-11-25` lifecycle the extension does not exist: the strict server refuses
/// the task tool with `-32021`, the client surfaces that as a tool error, and never polls.
#[test]
fn serve_legacy_lifecycle_surfaces_missing_capability_and_never_polls() {
    let env = Env::new(json!({ "MCP_TASKS_FIXTURE_LEGACY": "1" }));
    let frames = env.serve_prompt(vec![
        turn_tool_use("toolu_plain", "mcp__t__plain", "{}"),
        turn_tool_use("toolu_poll", "mcp__t__poll_task", "{}"),
        turn_text("done"),
    ]);
    assert!(
        !env.methods("initialize").is_empty(),
        "fell back to legacy initialize"
    );
    let ends = tool_ends(&frames);
    assert_eq!(ends.len(), 2, "{frames:?}");
    assert_eq!(ends[0]["is_error"], false, "{}", ends[0]);
    assert!(ends[0]["result"].as_str().unwrap().contains("plain-result"));
    assert_eq!(ends[1]["is_error"], true, "{}", ends[1]);
    assert!(
        ends[1]["result"]
            .as_str()
            .unwrap()
            .contains("Missing required client capability"),
        "{}",
        ends[1]
    );
    assert!(env.methods("tasks/get").is_empty(), "never polls");
}

/// Tasks are defined for `tools/call` only: a `CreateTaskResult` answering `resources/read` MUST be
/// treated as an invalid response — an error, not a poll loop and not an empty resource.
#[test]
fn serve_task_answering_an_unsupported_request_is_invalid() {
    let env = Env::new(json!({}));
    let frames = env.serve_prompt(vec![
        turn_tool_use("toolu_r", "mcp__t__resource__doc", "{}"),
        turn_text("done"),
    ]);
    let end = only_tool_end(&frames);
    assert_eq!(end["is_error"], true, "{end}");
    assert!(env.methods("tasks/get").is_empty(), "never polls");
}

fn write_scout(project: &Path) {
    let agents = project.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("scout.md"),
        "---\nname: scout\ndescription: recon\n---\nYou are SCOUT.\n",
    )
    .unwrap();
}

/// A subagent's MCP call drives the same task lifecycle to completion.
#[test]
fn serve_subagent_polls_a_task_to_completion() {
    let env = Env::new(json!({}));
    let project = env.dir.path().join("project");
    write_scout(&project);
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "scout", "task": "run poll_task" }).to_string(),
        ),
        turn_tool_use("toolu_child", "mcp__t__poll_task", "{}"),
        turn_text("CHILD-DONE"),
        turn_text("parent done"),
    ]);
    let mut cmd = serve_cmd(common::BIN, &base, &env.session_file());
    cmd.env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .arg("--trust-project")
        .current_dir(&project);
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let requests = bodies.lock().unwrap().clone();
    assert_eq!(requests.len(), 4, "{requests:#?}");
    assert!(
        requests[2].contains("poll-done"),
        "the child's task result reaches the child's next turn: {}",
        requests[2]
    );
    assert_eq!(env.methods("tasks/get").len(), 4);
}

/// The TTL the client honours is the *latest* one: this task is created with `ttlMs: null` and only
/// later says `300`, so a client that kept the creation value would poll forever.
#[test]
fn serve_latest_ttl_ends_a_task_whose_ttl_appears_mid_task() {
    let env = Env::new(json!({}));
    let start = Instant::now();
    let frames = env.serve_prompt(vec![
        turn_tool_use("toolu_shift", "mcp__t__ttl_shift_task", "{}"),
        turn_text("done"),
    ]);
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
    let end = only_tool_end(&frames);
    assert_eq!(end["is_error"], true, "{end}");
    assert!(
        end["result"].as_str().unwrap().contains("ttlMs (300 ms)"),
        "{end}"
    );
}

/// Headless `run` has no host to ask, so a server's sampling request (here in-task) is refused,
/// and the call ends with that error rather than hanging or inventing a completion.
#[test]
fn run_refuses_in_task_sampling() {
    let env = Env::new(json!({}));
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_s", "mcp__t__sample_task", "{}"),
        turn_text("done"),
    ]);
    let output = run_cmd(common::BIN)
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
            "3",
            "--no-session-persistence",
        ])
        .env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .current_dir(env.dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded()
        .wait_with_output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let requests = bodies.lock().unwrap().clone();
    assert_eq!(requests.len(), 2, "{requests:#?}");
    let body = common::body_json(&requests[1]);
    let result = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().cloned().unwrap_or_default())
        .find(|b| b["type"] == "tool_result" && b["tool_use_id"] == "toolu_s")
        .unwrap();
    assert_eq!(result["is_error"], json!(true), "{result}");
    assert!(result.to_string().contains("sampling"), "{result}");
    assert!(
        env.methods("tasks/update").is_empty(),
        "nothing was sent back as an answer"
    );
}
