//! SEP-2663 tasks that outlive a connection, plus the clauses only Streamable HTTP has. Siblings:
//! `mcp_tasks_conformance.rs` (spec clauses on stdio), `mcp_tasks_resume.rs` (tasks that outlive the
//! agent process), `mcp_tasks_daemon.rs` (routing on a multi-session daemon).
//!
//! - `Mcp-Name` equals `params.taskId` on every `tasks/*` request (spec MUST, SEP-2243 routing).
//! - A request lost in transit is re-sent and the *same* task polled to completion; a lost
//!   `tasks/update` is re-sent, never re-asked.
//! - A call aborted while its connection is being redialed still cancels through the fresh one.
//! - An in-task `sampling/createMessage` reaches the host (`sampling_request` → `sample`), naming
//!   the server, and every view sees it resolved.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
mod mcp_tasks_env;

use std::io::BufReader;
use std::time::Duration;

use common::{read_until_response, spawn_model_server, turn_text, turn_tool_use};
use mcp_tasks_env::{Env, prompt, read_until, send, tool_end, wait_for};
use serde_json::{Value, json};

fn single_prompt(env: &Env, turns: Vec<String>) -> Vec<Value> {
    let (base, _bodies) = spawn_model_server(turns);
    let mut child = env.serve(&base);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    prompt(&mut stdin, "go");
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();
    frames
}

/// Over Streamable HTTP, every `tasks/get` / `tasks/update` / `tasks/cancel` carries
/// `Mcp-Name: <taskId>` and `Mcp-Method: <method>` (spec MUST, so a load balancer can route a
/// task's requests to the instance holding it).
#[test]
fn http_tasks_requests_carry_mcp_name_equal_to_the_task_id() {
    let env = Env::new();
    let _server = env.http();
    let frames = single_prompt(
        &env,
        vec![
            turn_tool_use("toolu_p", "mcp__t__poll_task", "{}"),
            turn_text("done"),
        ],
    );
    let end = tool_end(&frames, "toolu_p");
    assert_eq!(end["is_error"], false, "{end}");
    assert!(end["result"].as_str().unwrap().contains("poll-done"));

    let tasks: Vec<Value> = env
        .wire()
        .into_iter()
        .filter(|m| m["method"].as_str().unwrap_or("").starts_with("tasks/"))
        .collect();
    assert_eq!(tasks.len(), 4, "four polls: {tasks:?}");
    for m in &tasks {
        assert_eq!(
            m["headers"]["mcp-name"], m["params"]["taskId"],
            "Mcp-Name must equal params.taskId: {m}"
        );
        assert_eq!(m["headers"]["mcp-method"], m["method"], "{m}");
    }
    let call = env
        .methods("tools/call")
        .into_iter()
        .find(|m| m["params"]["name"] == "poll_task")
        .unwrap();
    assert_eq!(call["headers"]["mcp-name"], "poll_task", "{call}");
}

/// The fixture drops the connection on two consecutive polls without answering. The task lives on
/// the server (and rmcp's HTTP client survives a failed POST), so the client backs off and polls
/// the same `taskId` again to completion instead of failing the call; nothing is re-invoked.
#[test]
fn http_connection_drop_mid_poll_retries_and_resumes_the_same_task() {
    let env = Env::new();
    let _server = env.http();
    let frames = single_prompt(
        &env,
        vec![
            turn_tool_use("toolu_b", "mcp__t__blip_task", "{}"),
            turn_text("done"),
        ],
    );
    let end = tool_end(&frames, "toolu_b");
    assert_eq!(
        end["is_error"], false,
        "a dropped connection is not the task's failure: {end}"
    );
    assert!(
        end["result"].as_str().unwrap().contains("blip-done"),
        "{end}"
    );

    let calls: Vec<Value> = env
        .methods("tools/call")
        .into_iter()
        .filter(|m| m["params"]["name"] == "blip_task")
        .collect();
    assert_eq!(calls.len(), 1, "the tool is never re-invoked");
    let gets = env.methods("tasks/get");
    assert_eq!(
        gets.len(),
        5,
        "two dropped polls, then polling resumes: {gets:?}"
    );
    let id = &gets[0]["params"]["taskId"];
    assert!(
        gets.iter().all(|g| &g["params"]["taskId"] == id),
        "same task: {gets:?}"
    );
}

/// The first `tasks/update` is dropped unapplied. The answer is re-sent as it was: the user is
/// asked once, and the server sees the same `inputResponses` twice.
#[test]
fn http_dropped_tasks_update_is_resent_not_reasked() {
    let env = Env::new();
    let _server = env.http();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_d", "mcp__t__drop_update_task", "{}"),
        turn_text("done"),
    ]);
    let mut child = env.serve(&base);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    prompt(&mut stdin, "go");
    let mut frames = read_until(&mut stdout, "the elicitation", |f| {
        f["type"] == "elicitation_request"
    });
    let ask = frames.last().unwrap().clone();
    send(
        &mut stdin,
        json!({
            "type": "elicit",
            "request_id": ask["request_id"],
            "action": "accept",
            "content": { "name": "Ferris" },
        }),
    );
    // A second elicitation (the answer re-asked instead of re-sent) fails at once, rather than
    // waiting out the unanswered question's timeout.
    frames.extend(mcp_tasks_env::read_until_or_fail(
        &mut stdout,
        "the prompt response",
        Duration::from_secs(30),
        |f| f["type"] == "elicitation_request",
        |f| f["type"] == "response" && f["command"] == "prompt",
    ));
    drop(stdin);
    child.wait().unwrap();

    let asked = frames
        .iter()
        .filter(|f| f["type"] == "elicitation_request")
        .count();
    assert_eq!(asked, 1, "the user is asked once: {frames:?}");
    let end = tool_end(&frames, "toolu_d");
    assert_eq!(end["is_error"], false, "{end}");
    assert!(
        end["result"]
            .as_str()
            .unwrap()
            .contains("dropped-update-Ferris"),
        "{end}"
    );
    let updates = env.methods("tasks/update");
    assert_eq!(updates.len(), 2, "dropped once, re-sent once: {updates:?}");
    assert_eq!(
        updates[0]["params"]["inputResponses"], updates[1]["params"]["inputResponses"],
        "the re-send carries the same answer"
    );
}

/// The stdio server exits on the first poll; its restart is slow (the fixture sleeps), and the
/// call is aborted while the redial is in flight. The cancel must go out through the connection's
/// fresh client (a server that keeps tasks across its restart does get it), not the dead one.
#[test]
fn abort_during_redial_cancels_through_the_fresh_connection() {
    let env = Env::new();
    env.settings(json!([env.stdio_with(json!({
        "MCP_TASKS_FIXTURE_STATE": env.state.to_string_lossy(),
        "MCP_TASKS_FIXTURE_RESTART_DELAY_MS": "1500",
    }))]));
    let (base, _bodies) = spawn_model_server(vec![turn_tool_use(
        "toolu_c",
        "mcp__t__crash_once_task",
        "{}",
    )]);
    let mut child = env.serve(&base);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    prompt(&mut stdin, "go");
    wait_for("the crashing poll", || !env.methods("tasks/get").is_empty());
    // Past the first backoff (200 ms), so the redial is under way (the restart takes 1.5 s).
    std::thread::sleep(Duration::from_millis(500));
    send(&mut stdin, json!({ "type": "abort", "id": "a1" }));
    read_until_response(&mut stdout, "prompt");
    let task_id = std::fs::read_to_string(&env.state).unwrap();
    wait_for("tasks/cancel at the restarted server", || {
        env.methods("tasks/cancel")
            .iter()
            .any(|c| c["params"]["taskId"] == task_id.as_str())
    });
    drop(stdin);
    child.wait().unwrap();
}

/// An in-task `sampling/createMessage` is treated like the standalone request: it reaches the host
/// as a `sampling_request` naming the server, the host's `sample` answer goes back in
/// `tasks/update`, and a `sampling_resolved` tells every attached view it is settled.
#[test]
fn serve_in_task_sampling_round_trips_through_the_host() {
    let env = Env::new();
    env.stdio();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_s", "mcp__t__sample_task", "{}"),
        turn_text("done"),
    ]);
    let mut child = env.serve(&base);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    prompt(&mut stdin, "go");
    let mut frames = read_until(&mut stdout, "a sampling_request", |f| {
        f["type"] == "sampling_request"
    });
    let ask = frames.last().unwrap().clone();
    assert_eq!(ask["server"], "t", "the asking server is named: {ask}");
    assert!(
        ask.to_string().contains("one-word draft"),
        "the server's sampling params reach the host: {ask}"
    );
    send(
        &mut stdin,
        json!({
            "type": "sample",
            "request_id": ask["request_id"],
            "result": {
                "role": "assistant",
                "content": { "type": "text", "text": "Ferris" },
                "model": "host-model",
                "stopReason": "endTurn",
            },
        }),
    );
    frames.extend(read_until_response(&mut stdout, "prompt"));
    drop(stdin);
    child.wait().unwrap();

    let resolved = frames
        .iter()
        .find(|f| f["type"] == "sampling_resolved")
        .unwrap_or_else(|| panic!("no sampling_resolved: {frames:?}"));
    assert_eq!(resolved["request_id"], ask["request_id"]);
    assert_eq!(resolved["action"], "accept");
    let end = tool_end(&frames, "toolu_s");
    assert_eq!(end["is_error"], false, "{end}");
    assert!(
        end["result"].as_str().unwrap().contains("sampled:Ferris"),
        "{end}"
    );
    let updates = env.methods("tasks/update");
    assert_eq!(updates.len(), 1, "{updates:?}");
    assert_eq!(
        updates[0]["params"]["inputResponses"]["draft"]["content"]["text"], "Ferris",
        "{updates:?}"
    );
}
