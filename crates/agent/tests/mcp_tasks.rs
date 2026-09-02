//! SEP-2663 MCP tasks e2e: happy path + edges (in-task elicitation, failed/cancelled, abort→cancel).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

fn fixture_bin() -> &'static str {
    env!("CARGO_BIN_EXE_mcp_fixture_stdio_server")
}

fn write_global_settings(home: &Path, mcp_servers: Value) {
    let claude_dir = home.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": mcp_servers })).unwrap(),
    )
    .unwrap();
}

fn mcp_server(name: &str) -> Value {
    json!({
        "name": name,
        "transport": "stdio",
        "command": fixture_bin(),
        "args": [],
        "env": {},
    })
}

fn tool_end_text(frames: &[Value]) -> String {
    frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .filter_map(|f| f["event"]["result"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_end_frame(frames: &[Value]) -> &Value {
    frames
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .expect("tool_end must appear")
}

#[test]
fn serve_polls_mcp_task_to_completion_with_progress() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, json!([mcp_server("jobs")]));

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_t", "mcp__jobs__slow_task", &json!({}).to_string()),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run slow_task" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let kinds: Vec<&str> = frames
        .iter()
        .filter(|f| f["type"] == "event")
        .filter_map(|f| f["event"]["kind"].as_str())
        .collect();

    let progress: Vec<&Value> = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_progress")
        .collect();

    assert!(
        !progress.is_empty(),
        "SEP-2663 task statusMessage must emit tool_progress; kinds={kinds:?} frames={frames:?}"
    );

    let snapshots: String = progress
        .iter()
        .filter_map(|f| f["event"]["snapshot"].as_str())
        .collect::<Vec<_>>()
        .join("|");
    assert!(
        snapshots.contains("task-phase-one") || snapshots.contains("task-phase-two"),
        "progress must carry fixture task phases, got: {snapshots:?}"
    );

    let details_ok = progress.iter().any(|f| {
        f["event"]["details"]["taskId"].as_str().is_some()
            && f["event"]["details"]["statusMessage"].as_str().is_some()
    });
    assert!(
        details_ok,
        "tool_progress details must carry taskId/statusMessage: {progress:?}"
    );

    let first_progress = kinds.iter().position(|k| *k == "tool_progress").unwrap();
    let tool_end = kinds
        .iter()
        .position(|k| *k == "tool_end")
        .expect("tool_end must appear");
    assert!(
        first_progress < tool_end,
        "task progress must arrive before tool_end: {kinds:?}"
    );

    assert!(
        tool_end_text(&frames).contains("task-done"),
        "final result must be task-done: {}",
        tool_end_frame(&frames)
    );
}

/// In-task `input_required` elicitation fulfilled via `tasks/update`.
#[test]
fn serve_answers_in_task_elicitation_via_tasks_update() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, json!([mcp_server("jobs")]));

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_ask", "mcp__jobs__ask_task", &json!({}).to_string()),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run ask_task" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let mut elicit_id = None;
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut line = String::new();
    while Instant::now() < deadline {
        line.clear();
        stdout.read_line(&mut line).unwrap();
        if line.is_empty() {
            break;
        }
        let Ok(frame) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if frame["type"] == "elicitation_request" {
            elicit_id = frame["request_id"].as_str().map(str::to_string);
            assert_eq!(frame["server"], "jobs");
            let msg = frame
                .pointer("/params/message")
                .and_then(Value::as_str)
                .unwrap_or_default();
            assert!(
                msg.contains("task"),
                "expected in-task elicitation message, got {msg:?}"
            );
            break;
        }
    }
    let elicit_id = elicit_id.expect("timed out waiting for in-task elicitation_request");

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "elicit",
            "request_id": elicit_id,
            "action": "accept",
            "content": { "name": "Ferris" }
        })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    assert!(
        tool_end_text(&frames).contains("task-ask-hello-Ferris"),
        "in-task elicitation must reach the completed task result; got {} frames={frames:?}",
        tool_end_text(&frames)
    );
}

#[test]
fn serve_maps_failed_mcp_task_to_tool_error() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, json!([mcp_server("jobs")]));

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_f", "mcp__jobs__fail_task", &json!({}).to_string()),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run fail_task" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let end = tool_end_frame(&frames);
    assert_eq!(
        end["event"]["is_error"], true,
        "failed task must surface as tool error: {end}"
    );
    assert!(
        end["event"]["result"]
            .as_str()
            .unwrap_or("")
            .contains("task-failed-on-purpose"),
        "failed task error must carry fixture message: {end}"
    );
}

#[test]
fn serve_maps_server_cancelled_mcp_task_to_tool_error() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    write_global_settings(&home, json!([mcp_server("jobs")]));

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_c",
            "mcp__jobs__server_cancel_task",
            &json!({}).to_string(),
        ),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run server_cancel_task" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let end = tool_end_frame(&frames);
    assert_eq!(
        end["event"]["is_error"], true,
        "cancelled task must surface as tool error: {end}"
    );
    assert!(
        end["event"]["result"]
            .as_str()
            .unwrap_or("")
            .contains("was cancelled"),
        "cancelled task error must say cancelled: {end}"
    );
}

/// Abort mid-poll must send `tasks/cancel` (fixture writes `MCP_FIXTURE_CANCEL_FLAG`).
#[test]
fn serve_abort_sends_tasks_cancel_for_sticky_task() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let cancel_flag = dir.path().join("cancel.flag");
    write_global_settings(
        &home,
        json!([{
            "name": "jobs",
            "transport": "stdio",
            "command": fixture_bin(),
            "args": [],
            "env": {
                "MCP_FIXTURE_CANCEL_FLAG": cancel_flag.to_string_lossy(),
            },
        }]),
    );

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_tool_use(
        "toolu_s",
        "mcp__jobs__sticky_task",
        &json!({}).to_string(),
    )]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.env("HOME", &home);
    cmd.env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run sticky_task" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Wait until polling has started (statusMessage progress), then abort.
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut line = String::new();
    let mut saw_progress = false;
    while Instant::now() < deadline {
        line.clear();
        stdout.read_line(&mut line).unwrap();
        if line.is_empty() {
            break;
        }
        let Ok(frame) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if frame["type"] == "event"
            && frame["event"]["kind"] == "tool_progress"
            && frame["event"]["snapshot"]
                .as_str()
                .unwrap_or("")
                .contains("sticky")
        {
            saw_progress = true;
            break;
        }
    }
    assert!(
        saw_progress,
        "sticky_task must emit tool_progress before abort"
    );

    writeln!(stdin, "{}", json!({ "type": "abort", "id": "a1" })).unwrap();
    stdin.flush().unwrap();

    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "abort must end the sticky task promptly, took {:?}",
        start.elapsed()
    );
    let resp = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("prompt response");
    assert_eq!(
        resp["success"], false,
        "aborted sticky task prompt reports failure: {resp}"
    );
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "response" && f["command"] == "abort" && f["success"] == true),
        "abort must be acknowledged: {frames:?}"
    );

    // Drop guard spawns cancel asynchronously — poll the flag briefly.
    let cancel_deadline = Instant::now() + Duration::from_secs(3);
    let mut cancelled_id = None;
    while Instant::now() < cancel_deadline {
        if cancel_flag.exists() {
            cancelled_id = Some(std::fs::read_to_string(&cancel_flag).unwrap());
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let cancelled_id = cancelled_id
        .expect("abort must send tasks/cancel (fixture MCP_FIXTURE_CANCEL_FLAG was never written)");
    assert!(
        cancelled_id.starts_with("fixture-task-"),
        "cancel flag should carry task id, got {cancelled_id:?}"
    );

    drop(stdin);
    child.wait().unwrap();
}
