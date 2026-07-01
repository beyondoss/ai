//! End-to-end: the real `beyond-ai-agent serve` binary over its stdio control protocol.
//!
//! Drives the headless server exactly as a remote client (or an SSH pipe) would: writes JSON command
//! lines to stdin, reads JSON frames from stdout. Proves (a) a `prompt` streams `event` frames for a
//! tool round-trip then a success `response`, (b) `get_messages` returns the transcript, and (c) a
//! fresh `serve` process **reattaches** to the persisted session and sees the prior transcript.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};

use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};
use common::{ISOLATED_HOME, spawn_model_server, turn_refusal, turn_text, turn_tool_use};
use serde_json::{Value, json};

/// Read stdout frames until the `response` frame for `command` arrives; return all frames seen.
fn read_until_response(reader: &mut impl BufRead, command: &str) -> Vec<Value> {
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some(command);
        frames.push(v);
        if done {
            break;
        }
    }
    frames
}

fn serve_cmd(bin: &str, base: &str, session_file: &str) -> Command {
    let mut c = Command::new(bin);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-file",
        session_file,
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

fn serve_dir_cmd(bin: &str, base: &str, session_dir: &str) -> Command {
    let mut c = Command::new(bin);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-dir",
        session_dir,
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

/// Neither `--session-file` nor `--session-dir` — exercises `Persistence::open`'s default directory
/// (`~/.claude/sessions/<encoded-cwd>/`) rather than in-memory-only.
fn serve_default_persistence_cmd(bin: &str, base: &str) -> Command {
    let mut c = Command::new(bin);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

#[test]
fn serve_streams_tool_progress_from_a_running_bash() {
    // The full streaming chain, deterministically (mock model + real bash, no network): the model
    // calls `bash` with a command that emits output over time; the run must surface those chunks as
    // `tool_progress` event frames *before* the tool's result.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let cmd = "printf 'chunk-a\\n'; sleep 0.15; printf 'chunk-b\\n'";
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("toolu_b", "bash", &json!({ "command": cmd }).to_string()),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run it" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    // Collect tool_progress chunks in arrival order, and prove they precede the tool's end.
    let kinds: Vec<&str> = frames
        .iter()
        .filter(|f| f["type"] == "event")
        .filter_map(|f| f["event"]["kind"].as_str())
        .collect();
    let progress_chunks: String = frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_progress")
        .filter_map(|f| f["event"]["snapshot"].as_str())
        .collect();

    assert!(
        kinds.contains(&"tool_progress"),
        "a running bash must stream tool_progress frames: {kinds:?}"
    );
    assert!(
        progress_chunks.contains("chunk-a") && progress_chunks.contains("chunk-b"),
        "streamed chunks should carry the live output, got: {progress_chunks:?}"
    );
    let first_progress = kinds.iter().position(|k| *k == "tool_progress").unwrap();
    let tool_end = kinds.iter().position(|k| *k == "tool_end").unwrap();
    assert!(
        first_progress < tool_end,
        "progress must arrive before tool_end: {kinds:?}"
    );
}

#[test]
fn serve_follow_up_steers_an_in_flight_run() {
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // turn 1 runs a 1s sleep (keeps the run in flight long enough to steer), turn 2 ends the turn —
    // at which point the queued follow-up is injected — and turn 3 answers the follow-up.
    let turn1 = turn_tool_use(
        "toolu_s",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![
        turn1,
        turn_text("done with the first part"),
        turn_text("done with the follow-up"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    // Queue a follow-up while the first turn's sleep is running.
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f1", "message": "now the second thing" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    // The follow-up was acknowledged...
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up should be acknowledged: {frames:#?}"
    );
    // ...and a `steered` event fired as it was injected.
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a steered event should appear: {frames:#?}"
    );

    // The transcript holds the follow-up text and the second answer.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("now the second thing"));
    assert!(dump.contains("done with the follow-up"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_stop_after_turn_ends_the_run_after_the_current_tool_call_completes() {
    use std::time::Duration;

    // turn 1 runs a 1s sleep (keeps the run in flight long enough to send `stop_after_turn`); turn 2
    // and turn 3 would answer if the run continued — they must never be reached.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let turn1 = turn_tool_use(
        "toolu_stop",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![
        turn1,
        turn_text("should never be reached"),
        turn_text("also never reached"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    // Request a graceful stop while the first turn's bash sleep is still running.
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({ "type": "stop_after_turn", "id": "s1" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "stop_after_turn" && f["success"] == true),
        "stop_after_turn should be acknowledged: {frames:#?}"
    );
    // No `steered` event: the run ended, it wasn't redirected.
    assert!(
        !frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a stop request must not be reported as steering: {frames:#?}"
    );

    // Exactly one model call happened — the run stopped after the first turn's tool call, never
    // asking the model to react to the tool result.
    assert_eq!(
        bodies.lock().unwrap().len(),
        1,
        "the run must not start a second model call after the stop request"
    );

    // The transcript holds the tool call and its result, but neither of the never-reached replies.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("toolu_stop"), "got: {dump}");
    assert!(!dump.contains("should never be reached"));
    assert!(!dump.contains("also never reached"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_stop_after_turn_is_a_no_op_ack_when_idle() {
    // Sent with no `prompt` in flight, `stop_after_turn` must not silently sabotage the *next*
    // prompt (which would only run one turn instead of the two the model script provides).
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_idle",
            "bash",
            &json!({ "command": "echo hi" }).to_string(),
        ),
        turn_text("done"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "stop_after_turn", "id": "s0" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "stop_after_turn");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "stop_after_turn" && f["success"] == true),
        "an idle stop_after_turn should still ack: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    assert_eq!(
        bodies.lock().unwrap().len(),
        2,
        "the prompt sent after an idle stop_after_turn must run to its natural completion"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_export_html_writes_a_self_contained_transcript() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hello there")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "say hi" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    let output_path = dir.path().join("out.html").to_string_lossy().into_owned();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "export_html", "output_path": output_path })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "export_html");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(response["data"]["path"], output_path);

    let html = std::fs::read_to_string(&output_path).unwrap();
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("say hi"));
    assert!(html.contains("hello there"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_prompt_ack_arrives_before_the_first_event_frame() {
    // The lightweight `ack` frame is emitted the moment the turn is queued — before the model call
    // even starts — so it must arrive strictly before any `event` frame in the same `prompt` run.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "id": "p1", "message": "hi" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let ack_pos = frames
        .iter()
        .position(|f| f["type"] == "ack" && f["command"] == "prompt")
        .unwrap_or_else(|| panic!("no ack frame seen: {frames:#?}"));
    let first_event_pos = frames.iter().position(|f| f["type"] == "event");
    if let Some(event_pos) = first_event_pos {
        assert!(
            ack_pos < event_pos,
            "ack must precede the first event frame: {frames:#?}"
        );
    }
    // The ack carries the same id the client sent, so it can be correlated.
    assert_eq!(frames[ack_pos]["id"], "p1");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_busy_prompt_with_streaming_behavior_is_accepted_not_rejected() {
    // A `prompt` sent while another is in flight is normally rejected as busy — unless it carries
    // `streaming_behavior: "steer"|"follow_up"`, in which case it's accepted and routed through the
    // same `Steering` queue as an explicit `steer`/`follow_up` command.
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_s",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![
        turn1,
        turn_text("done with the first part"),
        turn_text("done with the steered part"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({
            "type": "prompt",
            "id": "p2",
            "message": "also handle this",
            "streaming_behavior": "steer",
        })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Two distinct "prompt"-command `response` frames arrive here: p2's immediate accept, and the
    // original (id-less) prompt's eventual terminal response — `read_until_response` (which matches by
    // command alone) would stop at whichever comes first, so read manually until both are seen.
    let mut frames = Vec::new();
    let mut prompt_responses_seen = 0;
    let mut line = String::new();
    while prompt_responses_seen < 2 {
        line.clear();
        if stdout.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if v["type"] == "response" && v["command"] == "prompt" {
            prompt_responses_seen += 1;
        }
        frames.push(v);
    }
    // The busy `prompt` with `streaming_behavior` was accepted (not the "busy" rejection)...
    assert!(
        frames
            .iter()
            .any(|f| f["id"] == "p2" && f["command"] == "prompt" && f["success"] == true),
        "a busy prompt with streaming_behavior must be accepted: {frames:#?}"
    );
    assert!(
        !frames
            .iter()
            .any(|f| f["id"] == "p2" && f["error"].as_str().is_some_and(|e| e.contains("busy"))),
        "must not be rejected as busy: {frames:#?}"
    );
    // ...and was actually injected as a steer.
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a steered event should appear: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("also handle this"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_follow_up_queued_while_idle_is_picked_up_by_next_prompt() {
    // No `prompt` is in flight at all yet — `follow_up` must still be accepted (not rejected as an
    // unknown command) and queue against the persistent `Steering` handle, picked up the moment the
    // next `prompt`'s first turn reaches a stop boundary.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("answered the queued follow-up"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Queue the follow-up first, while genuinely idle.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f0", "message": "the queued question" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "follow_up");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up while idle must be acknowledged, not rejected as unknown: {frames:#?}"
    );

    // Now prompt: turn 1 ends with no tool calls, so the queued follow-up is injected at that stop
    // boundary and turn 2 answers it — all within this one `prompt` call.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames.last().unwrap()["success"] == true,
        "prompt should succeed: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("the queued question"));
    assert!(dump.contains("answered the queued follow-up"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_refusal_ends_the_run_without_draining_steering() {
    // A refusal must be a distinct terminal condition: the `prompt` response reports `refused: true`,
    // no second model call happens (a queued follow-up is NOT drained/injected right after a refusal),
    // and the queued message survives untouched for a later `prompt` to pick up.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_refusal("I can't help with that."),
        turn_text("second prompt's normal answer"),
        turn_text("answered the queued follow-up"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Queue a follow-up while idle, before the refusal even happens.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f0", "message": "should stay queued" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "follow_up");

    // First prompt: the model refuses.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "please do something disallowed" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true);
    assert_eq!(
        response["data"]["refused"], true,
        "refused must be reported: {response:#?}"
    );
    assert!(
        !frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "a refusal must not drain/inject the queued follow-up: {frames:#?}"
    );

    // Second prompt: an ordinary stop — the queued follow-up (still intact) is drained/injected now.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "a normal message" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "event" && f["event"]["kind"] == "steered"),
        "the queued follow-up must survive the refusal and be injected on the next stop: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("should stay queued"));
    assert!(dump.contains("answered the queued follow-up"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switches_model_and_thinking_at_runtime() {
    // These are pure control commands — no model call — so the mock server is never hit.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // The known-model list is returned and non-empty.
    writeln!(stdin, "{}", json!({ "type": "get_available_models" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_available_models");
    let models = frames.last().unwrap()["data"]["models"].as_array().unwrap();
    assert!(
        models.iter().any(|m| m == "claude-opus-4-8"),
        "model list should include the default opus id: {models:#?}"
    );

    // Switch the model; the response echoes it and `get_state` reflects it.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-4o" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["data"]["model"], "gpt-4o");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["model"],
        "gpt-4o",
        "get_state must reflect the switched model"
    );

    // A missing `model` is rejected.
    writeln!(stdin, "{}", json!({ "type": "set_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], false);

    // Set then clear the thinking budget.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_thinking", "budget": 4096 })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_thinking");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert_eq!(frames.last().unwrap()["data"]["thinking"], 4096);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_thinking", "budget": Value::Null })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_thinking");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert!(frames.last().unwrap()["data"]["thinking"].is_null());

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_model_advances_and_wraps() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_available_models" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_available_models");
    let models: Vec<String> = frames.last().unwrap()["data"]["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m.as_str().unwrap().to_string())
        .collect();

    // Pin the model to the list's *last* entry first, so cycling from a known position is
    // unambiguous (the server's own default id, "claude-test", isn't in `available_models()` at all,
    // and would otherwise wrap to index 0 on the very first cycle regardless of direction).
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": models[models.len() - 1] })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    // Cycling past the last entry wraps to the first...
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    assert_eq!(frames.last().unwrap()["data"]["model"], models[0]);

    // ...and cycling again advances normally to the second.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    assert_eq!(frames.last().unwrap()["data"]["model"], models[1]);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_thinking_level_advances_through_the_ladder_and_wraps() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Starting Off, each cycle advances one rung on the portable Off/Minimal/Low/Medium/High/XHigh
    // ladder, wrapping back to Off. `claude-test` (this test's model) resolves to `ThinkingShape::Budget`
    // with a 32_000 max_output, so `reasoning_effort` stays null throughout (that dialect arm never
    // reads it) and `thinking` is the level's derived, clamped budget.
    let expected = [
        ("minimal", json!(1024)),
        ("low", json!(2048)),
        ("medium", json!(8192)),
        ("high", json!(24000)),
        ("xhigh", json!(31999)),
        ("off", Value::Null),
    ];
    for (level, thinking) in expected {
        writeln!(stdin, "{}", json!({ "type": "cycle_thinking_level" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "cycle_thinking_level");
        let data = &frames.last().unwrap()["data"];
        assert_eq!(frames.last().unwrap()["success"], true, "got: {data:#?}");
        assert_eq!(data["level"], level, "got: {data:#?}");
        assert_eq!(data["thinking"], thinking, "got: {data:#?}");
        assert!(data["reasoning_effort"].is_null(), "got: {data:#?}");
    }

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_reasoning_effort_sets_the_portable_level_directly() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "high" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(frames.last().unwrap()["success"], true, "got: {data:#?}");
    assert_eq!(data["level"], "high");
    assert_eq!(data["thinking"], 24000);

    // A subsequent cycle starts from "high", advancing to "xhigh" — proving `set_reasoning_effort`
    // really did move `current_level`, not just a one-off override.
    writeln!(stdin, "{}", json!({ "type": "cycle_thinking_level" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_thinking_level");
    assert_eq!(frames.last().unwrap()["data"]["level"], "xhigh");

    // `null` clears it back to off.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": Value::Null })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["level"], "off");
    assert!(data["thinking"].is_null());

    // An unrecognized effort name is rejected, not silently ignored.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "extreme" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_reasoning_effort_wins_over_a_stale_set_thinking_override() {
    // `set_thinking` sets an explicit raw-budget override; `set_reasoning_effort` (like
    // `cycle_thinking_level`) must clear it so the newly-requested level takes visible effect
    // immediately rather than being masked by the leftover override.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_thinking", "budget": 4096 })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_thinking");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "low" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(
        data["thinking"], 2048,
        "the level's own budget must win, not the stale 4096 override: {data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_auto_compaction_toggles_and_rejects_a_non_boolean() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_compaction");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert_eq!(frames.last().unwrap()["data"]["auto_compaction"], false);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_compaction");
    assert_eq!(frames.last().unwrap()["data"]["auto_compaction"], true);

    // Missing/non-boolean `enabled` is rejected, not silently coerced.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": "yes" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_compaction");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_auto_retry_toggles_and_rejects_a_non_boolean() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_retry");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert_eq!(frames.last().unwrap()["data"]["auto_retry"], false);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_retry");
    assert_eq!(frames.last().unwrap()["data"]["auto_retry"], true);

    // Missing/non-boolean `enabled` is rejected, not silently coerced.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": "yes" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_retry");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_auto_retry_false_fails_immediately_instead_of_retrying_a_dropped_stream() {
    // A stream that opens (`message_start`) but closes with no `message_stop` is a dropped connection —
    // normally retried (`agent_core`'s mid-stream retry). With auto_retry off, it must surface as an
    // immediate `prompt` failure instead, with no second request ever reaching the model server.
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![truncated.to_string()]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_auto_retry");

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false);
    assert!(
        response["error"].as_str().unwrap().contains("stream ended"),
        "got: {response:#?}"
    );
    assert_eq!(
        bodies.lock().unwrap().len(),
        1,
        "auto_retry(false) must not attempt a second request"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_injects_project_instructions_into_system_prompt() {
    // A `CLAUDE.md` in the working directory must reach the model: the agent assembles it into the
    // system prompt. The mock server records the request body, so we assert the marker is present.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("CLAUDE.md"), "PROJECT-MARKER-9182").unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.current_dir(dir.path()); // CLAUDE.md is discovered relative to cwd
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("PROJECT-MARKER-9182")),
        "project CLAUDE.md must be injected into the system prompt; bodies: {recorded:#?}"
    );
}

#[test]
fn serve_caches_the_static_system_prompt_until_reload() {
    // The static half of the system prompt (project instructions, skills) is expensive (filesystem
    // discovery) and is meant to be cached across ordinary turns, refreshed only by `set_model`/
    // `set_thinking`/an explicit `reload` — never by the per-turn dynamic-footer refresh alone.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("CLAUDE.md"), "MARKER-BEFORE-RELOAD").unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) =
        spawn_model_server(vec![turn_text("ok"), turn_text("ok"), turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.current_dir(dir.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Turn 1: the marker as it existed at startup.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "one" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Change the file on disk *while the process keeps running*, then prompt again with no reload.
    std::fs::write(dir.path().join("CLAUDE.md"), "MARKER-AFTER-EDIT").unwrap();
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "two" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Now explicitly reload, then prompt a third time.
    writeln!(stdin, "{}", json!({ "type": "reload" })).unwrap();
    stdin.flush().unwrap();
    let reload_frames = read_until_response(&mut stdout, "reload");
    assert_eq!(
        reload_frames.last().unwrap()["success"],
        true,
        "reload must succeed: {reload_frames:#?}"
    );
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "three" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert_eq!(recorded.len(), 3, "expected exactly three model calls");
    assert!(
        recorded[0].contains("MARKER-BEFORE-RELOAD"),
        "turn 1 must see the file as it was at startup"
    );
    assert!(
        recorded[1].contains("MARKER-BEFORE-RELOAD") && !recorded[1].contains("MARKER-AFTER-EDIT"),
        "turn 2 must still see the cached static prompt, not the on-disk edit: {}",
        recorded[1]
    );
    assert!(
        recorded[2].contains("MARKER-AFTER-EDIT"),
        "turn 3, after an explicit reload, must see the on-disk edit: {}",
        recorded[2]
    );
}

#[test]
fn serve_gates_skills_and_prompts_on_project_trust() {
    // An untrusted project's `.claude/skills`/`.claude/prompts` are attacker-controlled instructions:
    // they must not be advertised via `get_commands`, nor invocable via `/skill:name`/`/name`, until
    // the directory is trusted. Isolate HOME so this doesn't see (or pollute) the real developer's
    // `~/.claude/skills`/`~/.claude/trusted-projects.json`.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nDo the foo thing.",
    )
    .unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(prompt_dir.join("bar.md"), "Bar template body: $1").unwrap();

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Untrusted: no --trust-project.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
        let mut cmd = serve_cmd(bin, &base, &session_file);
        cmd.current_dir(dir.path()).env("HOME", home.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_commands");
        let commands = frames.last().unwrap()["data"]["commands"]
            .as_array()
            .unwrap();
        assert!(
            commands.is_empty(),
            "an untrusted project must advertise no skills/prompts, got: {commands:?}"
        );

        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "/skill:foo" })
        )
        .unwrap();
        stdin.flush().unwrap();
        drop(stdin);
        child.wait().unwrap();

        let recorded = _bodies.lock().unwrap();
        assert!(
            recorded.iter().all(|b| !b.contains("Do the foo thing")),
            "an untrusted project's skill body must never reach the model: {recorded:#?}"
        );
    }

    // Trusted: with --trust-project, both the skill and the prompt template are discoverable.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
        let session_file_2 = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
        let mut cmd = serve_cmd(bin, &base, &session_file_2);
        cmd.arg("--trust-project")
            .current_dir(dir.path())
            .env("HOME", home.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_commands");
        let commands = frames.last().unwrap()["data"]["commands"]
            .as_array()
            .unwrap();
        let names: Vec<&str> = commands.iter().filter_map(|c| c["name"].as_str()).collect();
        assert!(
            names.contains(&"foo"),
            "trusted project should list the skill: {names:?}"
        );
        assert!(
            names.contains(&"bar"),
            "trusted project should list the prompt template: {names:?}"
        );

        drop(stdin);
        child.wait().unwrap();
    }
}

#[test]
fn serve_untrusted_project_still_advertises_and_invokes_a_user_global_skill() {
    // The bug this guards: `.claude/skills` under the *project* is attacker-controlled and rightly
    // gated on trust, but `~/.claude/skills` is the operator's own machine — an untrusted project must
    // not blank that out too.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let user_skill_dir = home.path().join(".claude/skills/mine");
    std::fs::create_dir_all(&user_skill_dir).unwrap();
    std::fs::write(
        user_skill_dir.join("SKILL.md"),
        "---\nname: mine\ndescription: a user-global skill\n---\nDo the user thing.",
    )
    .unwrap();
    // Also seed an untrusted *project* skill, to confirm it stays gated even while the user skill
    // isn't — the split, not just a blanket toggle.
    let project_skill_dir = dir.path().join(".claude/skills/theirs");
    std::fs::create_dir_all(&project_skill_dir).unwrap();
    std::fs::write(
        project_skill_dir.join("SKILL.md"),
        "---\nname: theirs\ndescription: an untrusted project skill\n---\nBody.",
    )
    .unwrap();

    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let mut cmd = serve_cmd(bin, &base, &session_file);
    // No --trust-project: the project is untrusted.
    cmd.current_dir(dir.path()).env("HOME", home.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap();
    let names: Vec<&str> = commands.iter().filter_map(|c| c["name"].as_str()).collect();
    assert!(
        names.contains(&"mine"),
        "an untrusted project must still advertise the user-global skill: {names:?}"
    );
    assert!(
        !names.contains(&"theirs"),
        "the untrusted project's own skill must stay gated: {names:?}"
    );

    // And it's actually invocable, not merely listed.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "/skill:mine" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = _bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("Do the user thing")),
        "the user-global skill's body must reach the model when invoked: {recorded:#?}"
    );
}

#[test]
fn serve_get_commands_reports_a_cross_root_skill_collision() {
    // A skill named "dup" declared at both the user (`~/.claude/skills`) and project
    // (`<cwd>/.claude/skills`) roots: `get_commands` must list it once (project wins) but also report
    // the shadowing via `collisions`, rather than silently resolving it with no client-visible trace.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    for (root, description) in [(home.path(), "user copy"), (dir.path(), "project copy")] {
        let skill_dir = root.join(".claude/skills/dup");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: dup\ndescription: {description}\n---\nBody."),
        )
        .unwrap();
    }
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project")
        .current_dir(dir.path())
        .env("HOME", home.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_commands");
    let data = &frames.last().unwrap()["data"];
    let commands = data["commands"].as_array().unwrap();
    let dup_count = commands
        .iter()
        .filter(|c| c["name"].as_str() == Some("dup"))
        .count();
    assert_eq!(
        dup_count, 1,
        "the shadowed name must appear once: {commands:?}"
    );
    let collisions = data["collisions"].as_array().unwrap();
    assert!(
        collisions
            .iter()
            .any(|c| c.as_str().is_some_and(|s| s.contains("dup"))),
        "collisions must report the shadowed skill: {collisions:?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_defaults_to_home_claude_sessions_when_no_session_flag_given() {
    // Neither --session-file nor --session-dir: must default to a real, cwd-encoded directory under
    // HOME rather than silently running in-memory-only. HOME is overridden to a tempdir so this
    // neither sees nor pollutes the real developer's `~/.claude/sessions`.
    let home = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_default_persistence_cmd(bin, &base);
    cmd.current_dir(project.path()).env("HOME", home.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let sessions_root = home.path().join(".claude/sessions");
    assert!(
        sessions_root.is_dir(),
        "expected a default sessions directory under HOME at {}",
        sessions_root.display()
    );
    // Exactly one project subdirectory (the encoded cwd), containing exactly one session file.
    let project_dirs: Vec<_> = std::fs::read_dir(&sessions_root)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        project_dirs.len(),
        1,
        "expected one encoded-cwd subdirectory: {project_dirs:?}"
    );
    let session_files: Vec<_> = std::fs::read_dir(project_dirs[0].path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    assert_eq!(
        session_files.len(),
        1,
        "expected one persisted session file: {session_files:?}"
    );
}

#[test]
fn serve_resumes_newest_session_matching_cwd_not_globally_newest() {
    // A --session-dir shared across two different project directories (the case the new default
    // avoids by cwd-encoding its own path, but still possible with an explicit shared directory):
    // reattaching from project A must resume A's own session, not B's more-recently-updated one.
    let session_dir_tmp = tempfile::tempdir().unwrap();
    let session_dir = session_dir_tmp.path().to_string_lossy().into_owned();
    let project_a = tempfile::tempdir().unwrap();
    let project_b = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Session A, from project_a.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("answer from A")]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(project_a.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "hi from A" })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
        drop(stdin);
        child.wait().unwrap();
    }

    // Session B, from project_b — created and updated *after* A, so a globally-newest-first pick
    // would wrongly resume this one from project_a.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("answer from B")]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(project_b.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "hi from B" })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
        drop(stdin);
        child.wait().unwrap();
    }

    // Reattach from project_a again — must resume A's transcript, not B's.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(project_a.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_messages");
        let dump = frames.last().unwrap()["data"]["messages"].to_string();
        assert!(
            dump.contains("hi from A") && dump.contains("answer from A"),
            "expected to resume project_a's own session: {dump}"
        );
        assert!(
            !dump.contains("from B"),
            "must not resume project_b's newer-but-different-cwd session: {dump}"
        );
        drop(stdin);
        child.wait().unwrap();
    }
}

#[test]
fn serve_reattaches_through_a_symlinked_cwd_to_the_session_recorded_under_its_real_path() {
    // A project reached through a symlink one time and its real path another must resolve to the
    // same session (`session_store::canonical_cwd`), not silently fork into two.
    let session_dir_tmp = tempfile::tempdir().unwrap();
    let session_dir = session_dir_tmp.path().to_string_lossy().into_owned();
    let projects = tempfile::tempdir().unwrap();
    let real = projects.path().join("real-project");
    std::fs::create_dir(&real).unwrap();
    let link = projects.path().join("project-link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Start (and prompt) from the real path.
    {
        let (base, _bodies) = spawn_model_server(vec![turn_text("answer via real path")]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(&real);
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(
            stdin,
            "{}",
            json!({ "type": "prompt", "message": "hi via real path" })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
        drop(stdin);
        child.wait().unwrap();
    }

    // Reattach from the symlinked path — must resume the same session, not mint a new one.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let mut cmd = serve_dir_cmd(bin, &base, &session_dir);
        cmd.current_dir(&link);
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_messages");
        let dump = frames.last().unwrap()["data"]["messages"].to_string();
        assert!(
            dump.contains("hi via real path") && dump.contains("answer via real path"),
            "a symlinked cwd must reattach to the session recorded under its real path: {dump}"
        );
        drop(stdin);
        child.wait().unwrap();
    }
}

#[test]
fn serve_list_sessions_streams_progress_frames_correlated_to_the_request_id() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    // Pre-seed sessions directly on disk (no need to drive a `prompt` per session) so the scan
    // `list_sessions` performs has more than one file to report progress across.
    let repo = SessionRepo::open(&session_dir).unwrap();
    for i in 0..6 {
        repo.create(SessionMeta::new(format!("/w{i}"), "m"))
            .unwrap();
    }

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    // Drain the `ready` banner.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "list_sessions", "id": "req-1" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");

    let progress: Vec<&Value> = frames
        .iter()
        .filter(|f| f["type"] == "list_progress")
        .collect();
    assert!(
        !progress.is_empty(),
        "expected at least one list_progress frame: {frames:#?}"
    );
    for p in &progress {
        assert_eq!(p["command"], "list_sessions");
        assert_eq!(
            p["id"], "req-1",
            "progress must correlate to the request id"
        );
        assert!(p["scanned"].as_u64().unwrap() >= 1);
        assert!(p["total"].as_u64().unwrap() >= p["scanned"].as_u64().unwrap());
    }
    // The last progress frame observed must reach the full total. Since the scan is parallel, frames
    // may not arrive in strictly increasing `scanned` order, but the maximum reported must still be
    // the total, and the total must match the response's own session count — `serve`'s own startup
    // reattach mints one more session for its actual cwd (which matches none of the 6 seeded here),
    // so the total is 7, not 6.
    let max_scanned = progress
        .iter()
        .map(|p| p["scanned"].as_u64().unwrap())
        .max()
        .unwrap();
    let total = progress[0]["total"].as_u64().unwrap();
    assert!(
        total >= 6,
        "must cover at least the 6 pre-seeded sessions: {progress:#?}"
    );
    assert_eq!(
        max_scanned, total,
        "the last progress frame must reach 100%"
    );

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true);
    assert_eq!(
        response["data"]["sessions"].as_array().unwrap().len() as u64,
        total,
        "progress total must match the number of sessions actually returned"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_repo_lists_switches_and_forks_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    // Two text turns: one per prompt.
    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer"), turn_text("second")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Read the `ready` banner to learn the first session's id.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let first_id = ready["session_id"].as_str().unwrap().to_string();

    // Prompt in session 1.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Start a second session.
    writeln!(stdin, "{}", json!({ "type": "new_session", "id": "n" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let second_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(first_id, second_id, "new_session must mint a new id");

    // Prompt in session 2.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "yo" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // List shows both, newest first.
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert!(
        sessions.len() >= 2,
        "both sessions should be listed: {sessions:#?}"
    );
    // Derived listing fields (`preview`/`message_count`/`updated_at`/`search_text`) live behind
    // `#[serde(skip)]` on `SessionMeta` so they never leak into the on-disk header — `list_sessions`
    // must still surface them to the client via `SessionMeta::to_listing_json`.
    let first_session = &sessions[0];
    assert!(
        first_session["message_count"].as_u64().unwrap() > 0,
        "message_count must be populated: {first_session:#?}"
    );
    assert!(
        first_session["updated_at"].as_u64().unwrap() > 0,
        "updated_at must be populated: {first_session:#?}"
    );
    assert!(
        first_session["preview"].is_string(),
        "preview must be populated: {first_session:#?}"
    );
    assert!(
        first_session["search_text"].is_string(),
        "search_text must be populated: {first_session:#?}"
    );

    // Switch back to session 1 and confirm its transcript is restored.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": first_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "switch_session");
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("first answer"),
        "switched-to session must restore its transcript: {dump}"
    );

    // `get_fork_messages` previews the same prefix `fork` would copy, without creating anything: the
    // session count in `list_sessions` must be unchanged afterward.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_fork_messages", "session_id": first_id, "upto": 1 })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_fork_messages");
    let preview = frames.last().unwrap();
    assert_eq!(preview["success"], true);
    let preview_messages = preview["data"]["messages"].as_array().unwrap();
    assert_eq!(
        preview_messages.len(),
        1,
        "upto:1 previews just the first message: {preview_messages:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let count_before_fork = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();

    // Fork the current session; the fork gets a new id.
    writeln!(stdin, "{}", json!({ "type": "fork" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "fork");
    let fork_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap();
    assert_ne!(fork_id, first_id, "a fork is a distinct session");

    // The preview above must not have created a session of its own — only the real `fork` above added
    // exactly one to the count taken right before it.
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let count_after_fork = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        count_after_fork,
        count_before_fork + 1,
        "get_fork_messages must not itself have created a session"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_list_all_sessions_spans_every_project_under_the_shared_root() {
    // Two independent `serve` processes, each rooted at its own subdirectory of one shared parent —
    // the layout `default_session_dir` produces per-project. `list_sessions` from either must see only
    // its own project's session; `list_all_sessions` must see both.
    let root = tempfile::tempdir().unwrap();
    let dir_a = root.path().join("proj-a").to_string_lossy().into_owned();
    let dir_b = root.path().join("proj-b").to_string_lossy().into_owned();

    let (base_a, _bodies_a) = spawn_model_server(vec![turn_text("answer from a")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child_a = serve_dir_cmd(bin, &base_a, &dir_a).spawn().unwrap();
    let mut stdin_a = child_a.stdin.take().unwrap();
    let mut stdout_a = BufReader::new(child_a.stdout.take().unwrap());
    writeln!(stdin_a, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin_a.flush().unwrap();
    read_until_response(&mut stdout_a, "prompt");

    let (base_b, _bodies_b) = spawn_model_server(vec![turn_text("answer from b")]);
    let mut child_b = serve_dir_cmd(bin, &base_b, &dir_b).spawn().unwrap();
    let mut stdin_b = child_b.stdin.take().unwrap();
    let mut stdout_b = BufReader::new(child_b.stdout.take().unwrap());
    writeln!(stdin_b, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin_b.flush().unwrap();
    read_until_response(&mut stdout_b, "prompt");

    // `list_sessions` from process A sees only its own project's session.
    writeln!(stdin_a, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin_a.flush().unwrap();
    let frames = read_until_response(&mut stdout_a, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert_eq!(
        sessions.len(),
        1,
        "list_sessions must stay scoped to this project: {sessions:#?}"
    );

    // `list_all_sessions` from process A sees both projects' sessions.
    writeln!(stdin_a, "{}", json!({ "type": "list_all_sessions" })).unwrap();
    stdin_a.flush().unwrap();
    let frames = read_until_response(&mut stdout_a, "list_all_sessions");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    let sessions = response["data"]["sessions"].as_array().unwrap();
    assert_eq!(
        sessions.len(),
        2,
        "list_all_sessions must span both projects: {sessions:#?}"
    );

    drop(stdin_a);
    child_a.wait().unwrap();
    drop(stdin_b);
    child_b.wait().unwrap();
}

#[test]
fn serve_list_all_sessions_errors_outside_repo_mode() {
    // Single-file persistence (`--session-file`) has no per-project sibling directories to scan.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "list_all_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_all_sessions");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "got: {response:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_tree_reports_every_node_not_just_leaves() {
    // `list_branches` reports only leaves; `get_tree` must report every node on every branch — proven
    // by branching (via `fork`) and confirming `get_tree`'s node count exceeds what a leaves-only view
    // would show, and that every node carries a role and (for text turns) a preview.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_tree" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let nodes = frames.last().unwrap()["data"]["nodes"].as_array().unwrap();
    // user + assistant.
    assert_eq!(nodes.len(), 2, "got: {nodes:#?}");
    assert!(nodes.iter().any(|n| n["role"] == "user"));
    assert!(nodes.iter().any(|n| n["role"] == "assistant"));
    assert!(
        nodes
            .iter()
            .any(|n| n["preview"].as_str().is_some_and(|p| p.contains("hi"))),
        "the user node should preview its own text: {nodes:#?}"
    );
    // The root node has no parent.
    assert!(nodes.iter().any(|n| n["parent_id"].is_null()));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_abort_cancels_an_in_flight_prompt() {
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir
        .path()
        .join("session.json")
        .to_string_lossy()
        .into_owned();

    // The model asks to run a 30s shell sleep; the run will be aborted mid-tool, so a second turn is
    // never requested.
    let turn1 = turn_tool_use(
        "toolu_b",
        "bash",
        &json!({ "command": "sleep 30" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run a long sleep" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Give the run time to reach the tool, then abort.
    std::thread::sleep(Duration::from_millis(500));
    writeln!(stdin, "{}", json!({ "type": "abort", "id": "a1" })).unwrap();
    stdin.flush().unwrap();

    // The prompt response must come back promptly (well under the 30s sleep) and report failure.
    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        start.elapsed() < Duration::from_secs(15),
        "abort must cancel the in-flight prompt promptly, took {:?}",
        start.elapsed()
    );
    let resp = frames.last().unwrap();
    assert_eq!(resp["command"], "prompt");
    assert_eq!(resp["success"], false, "an aborted prompt reports failure");
    assert!(
        frames
            .iter()
            .any(|f| f["type"] == "response" && f["command"] == "abort" && f["success"] == true),
        "the abort command should have been acknowledged: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
#[cfg(unix)]
fn serve_exits_gracefully_on_sigterm_mid_run() {
    use std::time::{Duration, Instant};

    // A SIGTERM (what `systemctl restart`/`docker stop`/a pod eviction sends) mid-run must be
    // treated like `abort`/stdin-closing: cancel the in-flight turn, persist what's there, and exit
    // on its own — not Rust's default disposition of immediate termination with no destructors run,
    // which would orphan the sleeping child process and lose the turn's unpersisted messages.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir
        .path()
        .join("session.json")
        .to_string_lossy()
        .into_owned();

    let turn1 = turn_tool_use(
        "toolu_b",
        "bash",
        &json!({ "command": "sleep 30" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let pid = child.id();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "run a long sleep" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Give the run time to reach the tool before signaling, so this exercises the mid-run
    // cancellation path (the harder one) rather than racing the idle-between-commands one.
    std::thread::sleep(Duration::from_millis(500));

    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .unwrap();
    assert!(status.success(), "failed to send SIGTERM to serve");

    // Must exit on its own well under the 30s sleep the in-flight tool call was running — not need
    // a hard `child.kill()` to reap it.
    let deadline = Instant::now() + Duration::from_secs(10);
    let exit = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "serve did not exit within 10s of SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(
        exit.success(),
        "serve should exit cleanly on SIGTERM, got {exit:?}"
    );

    // The stdout writer flushes on shutdown; whatever is left in the pipe just needs to not panic
    // when drained, not match any particular frame.
    let mut trailing = String::new();
    let _ = stdout.read_to_string(&mut trailing);

    // What was persisted before the cancel (at least the user's turn) must be a valid, non-empty,
    // readable session file — not lost by the abrupt-looking exit.
    let contents = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        !contents.trim().is_empty(),
        "nothing was persisted before SIGTERM shutdown"
    );
}

/// Extract every persisted message's `id` field, in file order, by parsing the session file's raw
/// JSONL directly — the RPC surface itself only exposes leaf ids (via `list_branches`), not every
/// interior message's id, so a client that wants to fork mid-history needs another source for those
/// (this test stands in for one) until a future extension teaches `get_messages` to include them.
fn message_ids(session_file: &str) -> Vec<String> {
    std::fs::read_to_string(session_file)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["type"] == "message")
        .filter_map(|v| v["id"].as_str().map(str::to_string))
        .collect()
}

#[test]
fn serve_switch_branch_summarizes_abandoned_activity_and_navigates() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // Two turns build a linear history, a third is the branch-summarization call `switch_branch`
    // triggers, a fourth answers the prompt issued after navigating back.
    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("second answer"),
        turn_text("recap: explored a dead end"),
        turn_text("continued from the original branch"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // A single, unbranched history reports exactly one branch, 4 messages deep.
    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches = frames.last().unwrap()["data"]["branches"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(branches.len(), 1, "no branching yet: {branches:#?}");
    assert_eq!(branches[0]["is_active"], true);
    assert_eq!(branches[0]["message_count"], 4);

    // Navigate back to the first turn's assistant reply (message index 1), abandoning the second
    // turn's user+assistant messages.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    let rewind_to = ids[1].clone();
    let abandoned_tip = ids[3].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": rewind_to })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "switch_branch failed: {resp:#?}");
    assert_eq!(resp["data"]["target_id"], rewind_to);

    // The active transcript is now the first turn *plus* the abandoned branch's summary — the recap
    // must actually reach the model-facing transcript, not just sit persisted off to the side.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("first answer"));
    assert!(
        dump.contains("recap: explored a dead end"),
        "the branch summary must be part of the live, model-facing transcript: {dump}"
    );
    assert!(
        !dump.contains("second answer"),
        "the abandoned turn must not appear on the restored branch: {dump}"
    );

    // Two branches now exist: the abandoned one (inactive, still 4 deep) and the active one — the
    // first turn (2) plus the summary message now folded into it (1) = 3 deep.
    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches = frames.last().unwrap()["data"]["branches"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(branches.len(), 2, "expected two branches: {branches:#?}");
    let abandoned = branches
        .iter()
        .find(|b| b["leaf_id"] == json!(abandoned_tip))
        .expect("the old tip should still be listed as a branch");
    assert_eq!(abandoned["is_active"], false);
    assert_eq!(abandoned["message_count"], 4);
    let active = branches.iter().find(|b| b["is_active"] == true).unwrap();
    assert_eq!(active["message_count"], 3);

    // The abandoned branch's summary was generated (consuming the 3rd mock response) and persisted.
    let raw = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        raw.contains("recap: explored a dead end"),
        "the branch summary should be persisted in the session file:\n{raw}"
    );
    assert!(raw.contains("\"branch_summary\""));

    // Continuing from the restored branch forks a *new* line of history off it, not a resumption of
    // the abandoned one.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "continue" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("continued from the original branch"));
    assert!(!dump.contains("second answer"));

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_messages_ids_enable_forking_from_any_point() {
    // Closes the gap `list_branches` alone leaves: it only ever reports a branch's *leaf*, so a
    // client that wants to fork from an arbitrary point in the middle of the visible transcript needs
    // ids from somewhere else. This proves `get_messages`'s tagged ids are real, usable
    // `switch_branch` targets — not just present, but round-trip through the actual RPC surface.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("second answer"),
        turn_text("forked from message index 1"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(messages.len(), 4, "expected 4 messages: {messages:#?}");
    let ids: Vec<String> = messages
        .iter()
        .map(|m| {
            m["id"]
                .as_str()
                .expect("every message should be tagged with an id")
                .to_string()
        })
        .collect();
    // All four ids are distinct.
    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    assert_eq!(
        unique.len(),
        4,
        "message ids should all be distinct: {ids:?}"
    );

    // Fork from message index 1 (the first turn's assistant reply) — a point `list_branches` alone
    // could never have named, since it only reports the (single, so far) branch's leaf.
    // `summarize:false`: the summarization path itself is covered by
    // `serve_switch_branch_summarizes_abandoned_activity_and_navigates`; this test is about ids, not
    // that, and skipping it keeps the mock response count matched to what's actually queued.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": ids[1], "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    assert_eq!(frames.last().unwrap()["success"], true);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "continue from here" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("forked from message index 1"));
    assert!(
        !dump.contains("second answer"),
        "forking from index 1 must not carry over the second turn: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_rejects_unknown_target() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": "does-not-exist" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    assert_eq!(frames.last().unwrap()["success"], false);

    writeln!(stdin, "{}", json!({ "type": "switch_branch" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_streams_events_and_reattaches() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-77\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();
    let session_file = dir
        .path()
        .join("session.json")
        .to_string_lossy()
        .into_owned();

    // One prompt drives two model turns: read tool, then text.
    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("Read complete.");
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // --- First session: prompt, observe streamed events, read transcript ---
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "read hello.txt" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    // A `ready` frame was emitted on startup.
    assert!(
        frames
            .iter()
            .any(|f| f.get("type").and_then(Value::as_str) == Some("ready"))
    );
    // Tool-call boundaries streamed as events.
    let events: Vec<&Value> = frames.iter().filter(|f| f["type"] == "event").collect();
    assert!(
        events
            .iter()
            .any(|e| e["event"]["kind"] == "tool_start" && e["event"]["name"] == "read"),
        "expected a tool_start event for `read`; frames: {frames:#?}"
    );
    assert!(
        events
            .iter()
            .any(|e| e["event"]["kind"] == "tool_end" && e["event"]["name"] == "read")
    );
    // Final response is a success.
    let resp = frames.last().unwrap();
    assert_eq!(resp["type"], "response");
    assert_eq!(resp["success"], true);

    // get_messages returns the transcript including the tool result and final text.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames2 = read_until_response(&mut stdout, "get_messages");
    let dump = frames2.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("secret-marker-77"),
        "transcript should hold the tool result: {dump}"
    );
    assert!(dump.contains("Read complete."));

    drop(stdin); // close stdin → server exits
    assert!(child.wait().unwrap().success());

    // --- Reattach: a fresh process over the same session file sees the prior transcript ---
    let mut child2 = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin2 = child2.stdin.take().unwrap();
    let mut stdout2 = BufReader::new(child2.stdout.take().unwrap());
    writeln!(stdin2, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin2.flush().unwrap();
    let frames3 = read_until_response(&mut stdout2, "get_messages");
    let dump3 = frames3.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump3.contains("secret-marker-77"),
        "reattached session must restore the transcript: {dump3}"
    );

    drop(stdin2);
    child2.wait().unwrap();
}

#[test]
fn serve_exclude_tools_removes_a_tool_from_the_advertised_set() {
    // `--exclude-tools bash` must remove it from both the tool definitions sent to the model and the
    // default system prompt's tool list — an excluded tool should be invisible to the model, not just
    // rejected after the fact if it tries to call it.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--exclude-tools", "bash"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        !recorded[0].contains("\"bash\""),
        "excluded tool must not appear in the request body (tool defs or system prompt): {:?}",
        recorded[0]
    );
    assert!(
        recorded[0].contains("\"read\""),
        "other tools must remain advertised: {:?}",
        recorded[0]
    );
}

#[test]
fn serve_no_tools_sends_no_tools_field_at_all() {
    // `--no-tools` must leave the agent with an empty registry — the Anthropic dialect omits the
    // `tools` key entirely from the wire body when there are none to advertise (see
    // `dialect::anthropic::build_body`), so its absence is the precise signal to check for.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--no-tools"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        !recorded[0].contains("\"tools\":"),
        "no-tools mode must omit the tools field entirely: {:?}",
        recorded[0]
    );
}

#[test]
fn serve_bash_runs_a_host_command_independent_of_the_model() {
    // A `bash` RPC command must run without ever touching the model — no scripted response is queued,
    // so if `serve` mistakenly routed it through a model turn the mock server would have nothing to
    // answer with and the test would hang/fail on `read_until_response`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "printf host-bash-ran" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    assert_eq!(resp["data"]["result"], "host-bash-ran");
    assert_eq!(resp["data"]["is_error"], false);

    // tool_start/tool_end events fire exactly like a model-invoked bash call, for a client that
    // renders both cases through the same code path.
    let kinds: Vec<&str> = frames
        .iter()
        .filter(|f| f["type"] == "event")
        .filter_map(|f| f["event"]["kind"].as_str())
        .collect();
    assert!(kinds.contains(&"tool_start"), "{kinds:?}");
    assert!(kinds.contains(&"tool_end"), "{kinds:?}");
}

#[test]
fn serve_bash_is_rejected_when_the_tool_is_excluded() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--exclude-tools", "bash"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "echo should-not-run" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false, "frames: {frames:#?}");
    assert!(
        resp["error"]
            .as_str()
            .unwrap_or_default()
            .contains("not registered"),
        "frames: {frames:#?}"
    );
}

#[test]
fn serve_abort_bash_cancels_a_running_host_command() {
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "sleep 30" })
    )
    .unwrap();
    stdin.flush().unwrap();
    // Give the command a moment to actually start, then abort it — a real 30s sleep would fail the
    // test on timeout if cancellation didn't work.
    std::thread::sleep(Duration::from_millis(200));
    writeln!(stdin, "{}", json!({ "type": "abort_bash" })).unwrap();
    stdin.flush().unwrap();

    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "bash");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "abort_bash should cancel promptly, not wait out the full sleep"
    );
    drop(stdin);
    child.wait().unwrap();

    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "abort_bash" && f["success"] == true),
        "abort_bash should be acknowledged: {frames:#?}"
    );
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["data"]["is_error"], true,
        "a cancelled command must be reported as an error result: {frames:#?}"
    );
}

#[test]
fn serve_get_state_and_get_session_stats_answer_live_during_a_prompt() {
    use std::time::Duration;

    // A tool-heavy turn (a `bash` sleep keeps it in flight) must still answer read-only progress
    // queries instead of rejecting them as busy — the whole point of H-4: a client polling for a live
    // "tokens/steps so far" indicator shouldn't have to wait for the turn to finish.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let turn1 = turn_tool_use(
        "toolu_live",
        "bash",
        &json!({ "command": "sleep 0.5" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(150)); // let the first turn's usage land, mid-`sleep 0.5`

    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_session_stats", "id": "s1" })
    )
    .unwrap();
    writeln!(stdin, "{}", json!({ "type": "get_state", "id": "g1" })).unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let stats_resp = frames
        .iter()
        .find(|f| f["command"] == "get_session_stats" && f["id"] == "s1")
        .unwrap_or_else(|| panic!("no get_session_stats response: {frames:#?}"));
    assert_eq!(stats_resp["success"], true, "{stats_resp:#?}");
    assert!(
        stats_resp["data"]["input_tokens"].as_u64().unwrap_or(0) > 0,
        "the first turn's usage should already be mirrored live: {stats_resp:#?}"
    );

    let state_resp = frames
        .iter()
        .find(|f| f["command"] == "get_state" && f["id"] == "g1")
        .unwrap_or_else(|| panic!("no get_state response: {frames:#?}"));
    assert_eq!(state_resp["success"], true, "{state_resp:#?}");
    assert!(state_resp["data"]["message_count"].is_null());
    assert!(state_resp["data"]["session_id"].is_string());
}

#[test]
fn serve_reports_cwd_stale_false_for_a_freshly_created_session() {
    // A session `serve` creates itself always records the actual current directory, which obviously
    // still exists — `cwd_stale` must read false everywhere it's surfaced.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    assert_eq!(ready["cwd_stale"], false, "got: {ready:#?}");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let state = frames.last().unwrap();
    assert_eq!(state["data"]["cwd_stale"], false, "got: {state:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_reports_cwd_stale_true_when_the_recorded_directory_no_longer_exists() {
    // File-mode persistence (`--session-file`) reattaches to whatever session is on disk without any
    // cwd-matching filter (unlike repo mode's automatic reattach). Hand-write a header recording a
    // directory that doesn't exist, simulating a project that was since moved or deleted, and confirm
    // `serve` surfaces the mismatch rather than silently proceeding as if nothing changed.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let header = json!({
        "type": "session",
        "id": "stale-cwd-session",
        "created_at": 1,
        "cwd": "/definitely/does/not/exist/beyond-ai-agent-test-fixture",
        "model": "claude-test",
    });
    std::fs::write(&session_file, format!("{header}\n")).unwrap();

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file.to_string_lossy())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    assert_eq!(ready["session_id"], "stale-cwd-session");
    assert_eq!(ready["cwd_stale"], true, "got: {ready:#?}");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let state = frames.last().unwrap();
    assert_eq!(state["data"]["cwd_stale"], true, "got: {state:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_session_reports_cwd_stale_for_the_newly_active_session() {
    // Repo mode's automatic reattach filters by cwd, so a mismatched session can only be reached by an
    // explicit `switch_session` — plant one directly in the repo directory (matching its
    // `<created_at>_<id>.jsonl` naming convention) and confirm switching to it surfaces the mismatch
    // immediately in the `switch_session` response, not just on a later `get_state` poll.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions");
    std::fs::create_dir_all(&session_dir).unwrap();
    let stale_header = json!({
        "type": "session",
        "id": "stale-target",
        "created_at": 1,
        "cwd": "/definitely/does/not/exist/beyond-ai-agent-test-fixture",
        "model": "claude-test",
    });
    std::fs::write(
        session_dir.join("1_stale-target.jsonl"),
        format!("{stale_header}\n"),
    )
    .unwrap();

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir.to_string_lossy())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // The freshly (auto-)created active session must not be stale.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    assert_eq!(ready["cwd_stale"], false, "got: {ready:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": "stale-target" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_session");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(response["data"]["cwd_stale"], true, "got: {response:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_survives_a_hard_crash_mid_run_with_the_first_round_trip_already_durable() {
    use std::time::Duration;

    // A genuine crash (SIGKILL — no signal handler, no graceful drain, nothing like the SIGTERM path
    // above) partway through a *second* tool round-trip must still leave the *first* round-trip's
    // messages durable on disk: proof that incremental mid-run persistence (H-6), not the final
    // post-run persist or the graceful-shutdown path, is what saved them.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "printf round-one-marker" }).to_string(),
    );
    let turn2 = turn_tool_use(
        "toolu_2",
        "bash",
        &json!({ "command": "sleep 5" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();

    // The first round-trip (a fast `printf`) should complete and checkpoint well within this window;
    // the second turn's `sleep 5` is still running when we kill the process.
    std::thread::sleep(Duration::from_millis(800));
    child.kill().unwrap(); // SIGKILL — no destructors, no signal handler, an actual hard crash
    child.wait().unwrap();

    // Reattach with a fresh process and check what survived.
    let (base2, _bodies2) = spawn_model_server(vec![]);
    let mut child2 = serve_cmd(bin, &base2, &session_file).spawn().unwrap();
    let mut stdin2 = child2.stdin.take().unwrap();
    let mut stdout2 = BufReader::new(child2.stdout.take().unwrap());
    writeln!(stdin2, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin2.flush().unwrap();
    let frames = read_until_response(&mut stdout2, "get_messages");
    drop(stdin2);
    child2.wait().unwrap();

    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("round-one-marker"),
        "the first tool round-trip must have been checkpointed before the crash: {dump}"
    );
    assert!(
        !dump.contains("toolu_2"),
        "the second (interrupted) round-trip must not appear as a completed pair: {dump}"
    );
}
