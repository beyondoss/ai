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

/// Like `serve_cmd`, but with an explicit `--model` instead of the hardcoded `"claude-test"` — for
/// tests exercising model-specific reasoning-effort clamping, where the test model itself matters.
fn serve_cmd_with_model(bin: &str, base: &str, session_file: &str, model: &str) -> Command {
    let mut c = Command::new(bin);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        model,
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
fn serve_get_messages_since_returns_only_what_was_appended_after_a_known_id() {
    // Track M21: a client that already has messages through some tree id shouldn't have to
    // re-transfer the whole transcript just to see what's new — pi's own `get_entries({since})`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("second answer")]);

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
    let all = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        all.len(),
        4,
        "expected [first, first answer, second, second answer]: {all:#?}"
    );
    let first_answer_id = all[1]["id"].as_str().unwrap().to_string();

    // Only what's new since the first turn's assistant reply: the second turn's user + assistant.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_messages", "since": first_answer_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "{resp:#?}");
    let since_messages = resp["data"]["messages"].as_array().unwrap();
    assert_eq!(since_messages.len(), 2, "{since_messages:#?}");
    assert!(since_messages[0]["content"].to_string().contains("second"));
    assert!(
        since_messages[1]["content"]
            .to_string()
            .contains("second answer")
    );

    // An unknown `since` id is an error, not a silent full re-fetch.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_messages", "since": "does-not-exist" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false, "{resp:#?}");

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
fn serve_export_html_includes_abandoned_branches_not_just_the_active_path() {
    // Track M19: an abandoned branch (created by rewinding via `switch_branch`) must still show up in
    // the export — the whole point being that the old flat `export_html` silently dropped anything not
    // on the active path.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("second answer")]);

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

    // Rewind to the first turn's assistant reply (message index 1), abandoning the second turn
    // (indices 2, 3) without a summary, so its original text survives verbatim on disk.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": ids[1], "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "switch_branch");

    let output_path = dir.path().join("out.html").to_string_lossy().into_owned();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "export_html", "output_path": output_path })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "export_html");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let html = std::fs::read_to_string(&output_path).unwrap();
    // The abandoned branch renders inline, as a collapsible <details> block positioned right after
    // the message it diverged from — not a separate flat "Other branches" section.
    let split = html
        .find("<details class=\"branch\">")
        .unwrap_or_else(|| panic!("the abandoned branch must get its own <details> block: {html}"));
    let (active_section, branches_section) = html.split_at(split);
    assert!(active_section.contains("first"), "the active path: {html}");
    assert!(
        active_section.contains("first answer"),
        "the active path: {html}"
    );
    assert!(
        !active_section.contains("second"),
        "abandoned by the rewind, must not be on the active path: {active_section}"
    );
    assert!(
        branches_section.contains("second answer"),
        "the abandoned branch's own content must still appear, in its section: {branches_section}"
    );

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
fn serve_follow_up_expands_a_skill_invocation_while_idle() {
    // MEDIUM pi-parity gap (fixed): `follow_up`/`steer` (and `prompt` with `streaming_behavior`) used
    // to push the raw message straight into the steering queue with no `/skill:name`/`/name`
    // expansion — only a fresh top-level `prompt` got that. A `/skill:name` sent through `follow_up`
    // must reach the model as the skill's expanded body, exactly like a fresh `prompt` would, not as
    // the literal unexpanded string.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nSKILL-BODY-MARKER-456",
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("answered the skill"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project").current_dir(dir.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Queue the skill invocation while genuinely idle, via `follow_up`.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "id": "f0", "message": "/skill:foo" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "follow_up");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up while idle must be acknowledged: {frames:#?}"
    );

    // Turn 1 ends with no tool calls, so the queued follow-up is injected at that stop boundary and
    // turn 2 sees it — all within this one `prompt` call.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames.last().unwrap()["success"] == true,
        "prompt should succeed: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("SKILL-BODY-MARKER-456")),
        "the skill's body must be expanded into the follow-up message before it reaches the model: \
         {recorded:#?}"
    );
    assert!(
        recorded.iter().all(|b| !b.contains("/skill:foo")),
        "the raw, unexpanded invocation must never reach the model: {recorded:#?}"
    );
}

#[test]
fn serve_mid_run_steer_expands_a_skill_invocation() {
    // Same gap as `serve_follow_up_expands_a_skill_invocation_while_idle`, but for the *other* code
    // path: a `steer` sent while a run is genuinely in flight (the busy-loop's own command handler,
    // architecturally distinct from the idle handler above) must expand a `/skill:name` invocation too.
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nSKILL-BODY-MARKER-789",
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // turn 1 runs a 1s sleep (keeps the run in flight long enough to steer), turn 2 ends the turn —
    // at which point the steered skill invocation is injected and answered.
    let turn1 = turn_tool_use(
        "toolu_s",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("answered the skill")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.arg("--trust-project").current_dir(dir.path());
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(
        stdin,
        "{}",
        json!({ "type": "steer", "id": "s1", "message": "/skill:foo" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "steer" && f["success"] == true),
        "steer should be acknowledged: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded.iter().any(|b| b.contains("SKILL-BODY-MARKER-789")),
        "the skill's body must be expanded into the steered message before it reaches the model: \
         {recorded:#?}"
    );
    assert!(
        recorded.iter().all(|b| !b.contains("/skill:foo")),
        "the raw, unexpanded invocation must never reach the model: {recorded:#?}"
    );
}

#[test]
fn serve_follow_up_carries_image_attachments_to_the_model() {
    // MEDIUM pi-parity gap (fixed): `follow_up`/`steer` used to have nowhere to put an `images` field
    // at all — a client attaching a screenshot to a queued follow-up had it silently dropped, unlike a
    // fresh `prompt`, which has always supported `images`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("saw the image")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "follow_up",
            "id": "f0",
            "message": "look at this",
            "images": [{ "media_type": "image/png", "data": "aGVsbG8taW1hZ2UtZGF0YQ==" }],
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "follow_up");
    assert!(
        frames
            .iter()
            .any(|f| f["command"] == "follow_up" && f["success"] == true),
        "follow_up with images should be acknowledged: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        frames.last().unwrap()["success"] == true,
        "prompt should succeed: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .any(|b| b.contains("aGVsbG8taW1hZ2UtZGF0YQ==")),
        "the follow-up's image data must reach the model, not be silently dropped: {recorded:#?}"
    );
}

#[test]
fn serve_no_skills_prevents_discovery_and_leaves_an_invocation_unexpanded() {
    // MEDIUM pi-parity gap (fixed): `serve` had no `--no-skills`/`--no-prompt-templates` at all — only
    // `run` did — so an operator wanting a hardened, no-custom-content `serve` deployment had no way to
    // refuse project-supplied skills. Same fixture/assertions as `run`'s own `--no-skills` test.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nSKILL-BODY-MARKER-999",
    )
    .unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--trust-project", "--no-skills"])
        .current_dir(dir.path());
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
        "--no-skills must prevent the skill from being discovered/advertised at all: {commands:?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "/skill:foo" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .all(|b| !b.contains("SKILL-BODY-MARKER-999")),
        "the skill's body must never reach the model when --no-skills is set: {recorded:#?}"
    );
    assert!(
        recorded.iter().any(|b| b.contains("/skill:foo")),
        "the raw invocation must reach the model unexpanded: {recorded:#?}"
    );
}

#[test]
fn serve_no_prompt_templates_prevents_discovery_and_leaves_an_invocation_unexpanded() {
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(prompt_dir.join("bar.md"), "TEMPLATE-BODY-MARKER-999: $1").unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--trust-project", "--no-prompt-templates"])
        .current_dir(dir.path());
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
        "--no-prompt-templates must prevent the template from being discovered/advertised at all: \
         {commands:?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "/bar arg" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .all(|b| !b.contains("TEMPLATE-BODY-MARKER-999")),
        "the template's body must never reach the model when --no-prompt-templates is set: \
         {recorded:#?}"
    );
    assert!(
        recorded.iter().any(|b| b.contains("/bar arg")),
        "the raw invocation must reach the model unexpanded: {recorded:#?}"
    );
}

#[test]
fn serve_default_queue_mode_drains_queued_follow_ups_one_at_a_time() {
    // pi's `PendingMessageQueue` default: several messages queued in quick succession land as
    // *separate* turns, one at a time — not folded into a single injection. Two follow-ups queued
    // while idle must reach the model server as two distinct requests (three total, including the
    // initial prompt), each carrying exactly one of the two follow-up texts as its newest user turn.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        turn_text("first answer"),
        turn_text("answered f1"),
        turn_text("answered f2"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for (id, msg) in [("f1", "first follow-up"), ("f2", "second follow-up")] {
        writeln!(
            stdin,
            "{}",
            json!({ "type": "follow_up", "id": id, "message": msg })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "follow_up");
    }

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        3,
        "initial prompt + one request per follow-up, not one merged request"
    );
    assert!(
        bodies[1].contains("first follow-up") && !bodies[1].contains("second follow-up"),
        "the second request should carry only the first follow-up: {}",
        bodies[1]
    );
    assert!(
        bodies[2].contains("first follow-up") && bodies[2].contains("second follow-up"),
        "the third request replays history, so it sees both by then, but as separate turns: {}",
        bodies[2]
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_queue_mode_all_folds_queued_follow_ups_into_one_injection() {
    // The opt-in `"all"` mode (this crate's original behavior, before pi-parity default flipped it):
    // both queued follow-ups are folded into the *same* next request, not drained one at a time.
    // Follow-ups are governed by `set_follow_up_mode`, not `set_steering_mode` — the two lanes have
    // independent settings.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) =
        spawn_model_server(vec![turn_text("first answer"), turn_text("answered both")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_follow_up_mode", "mode": "all" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_follow_up_mode");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
    assert_eq!(frames.last().unwrap()["data"]["mode"], "all");

    for (id, msg) in [("f1", "first follow-up"), ("f2", "second follow-up")] {
        writeln!(
            stdin,
            "{}",
            json!({ "type": "follow_up", "id": id, "message": msg })
        )
        .unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "follow_up");
    }

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "start" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "initial prompt + one request carrying BOTH follow-ups folded together"
    );
    assert!(
        bodies[1].contains("first follow-up") && bodies[1].contains("second follow-up"),
        "both follow-ups should reach the model in the same request: {}",
        bodies[1]
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_steering_mode_and_follow_up_mode_are_independent_rpc_settings() {
    // Track M12: `set_queue_mode` split into `set_steering_mode`/`set_follow_up_mode` — setting one
    // via RPC must not clobber the other, and `get_state` must report both independently.
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
        json!({ "type": "set_steering_mode", "mode": "all" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_steering_mode");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["steering_mode"], "all", "{data:#?}");
    assert_eq!(
        data["follow_up_mode"], "one_at_a_time",
        "follow_up_mode must be untouched by a steering-mode change: {data:#?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_follow_up_mode", "mode": "all" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_follow_up_mode");
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_steering_mode", "mode": "one_at_a_time" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_steering_mode");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(
        data["steering_mode"], "one_at_a_time",
        "steering_mode must be untouched by a follow-up-mode change: {data:#?}"
    );
    assert_eq!(data["follow_up_mode"], "all", "{data:#?}");

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

    // The known-model list is returned and non-empty, each entry a structured capability object (F-M2:
    // pi's `Model<any>` shape — `id`/`contextWindow`/`reasoning`, minus pricing — not a bare id string).
    writeln!(stdin, "{}", json!({ "type": "get_available_models" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_available_models");
    let models = frames.last().unwrap()["data"]["models"].as_array().unwrap();
    let opus = models
        .iter()
        .find(|m| m["id"] == "claude-opus-4-8")
        .unwrap_or_else(|| panic!("model list should include the default opus id: {models:#?}"));
    assert!(
        opus["context_window"].as_u64().unwrap() > 0,
        "got: {opus:#?}"
    );
    assert!(opus["reasoning"].is_boolean(), "got: {opus:#?}");
    assert_eq!(opus["provider"], "anthropic", "got: {opus:#?}");

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

    // Track L19: an empty/whitespace-only `model` is rejected too — the narrow, unambiguous mistake
    // this process CAN catch on its own (unlike a merely-unrecognized-but-otherwise-well-formed id,
    // which it can't: every id is forwarded verbatim through the gateway, with no local registry to
    // validate a real one against — see the RPC handler's own doc comment). The live model must be
    // left exactly as `gpt-4o` set it above, not reset to empty.
    writeln!(stdin, "{}", json!({ "type": "set_model", "model": "   " })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], false);
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["model"],
        "gpt-4o",
        "a rejected empty model must not disturb the live model: {frames:#?}"
    );

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
        .map(|m| m["id"].as_str().unwrap().to_string())
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
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], models[1]);
    assert_eq!(
        last["data"]["scoped"], false,
        "no --models flag was given, so cycling the full list must report scoped: false"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_model_scoped_by_the_models_flag_cycles_only_the_scope_but_lists_the_full_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-opus-4-8,gpt-4o",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // `get_available_models` is deliberately NOT scoped by `--models` — that flag only narrows
    // `cycle_model`'s own candidate list (asserted below). A client's model *picker* still needs to
    // see the full catalog to offer a "show everything" view, same as pi's own `/model` selector can
    // Tab out of its scope-defaulted view.
    writeln!(stdin, "{}", json!({ "type": "get_available_models" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_available_models");
    let all_models: Vec<String> = frames.last().unwrap()["data"]["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        all_models,
        beyond_ai_agent::serve::available_models()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        "get_available_models must report the full catalog, unaffected by --models scoping"
    );

    // Pin to the scoped list's first entry so cycling from a known position is unambiguous.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "claude-opus-4-8" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], "gpt-4o");
    assert_eq!(last["data"]["scoped"], true);

    // Cycling again must wrap back to the scoped list's *first* entry, not fall through to whatever
    // comes third in the full unscoped list.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    assert_eq!(frames.last().unwrap()["data"]["model"], "claude-opus-4-8");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_models_flag_expands_a_glob_against_the_known_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-*",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Pin to the first claude-* entry in catalog order so cycling from a known position is
    // unambiguous, then walk the whole scoped cycle and confirm it's exactly the claude-* subset of
    // `available_models()`, in catalog order, wrapping back to the start — never a gpt-*/o*-series id.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "claude-opus-4-8" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    let expected: Vec<&str> = beyond_ai_agent::serve::available_models()
        .iter()
        .copied()
        .filter(|id| id.starts_with("claude-"))
        .collect();
    assert!(
        expected.len() >= 2,
        "fixture assumption: the known catalog has multiple claude-* ids"
    );

    let cycle_order: Vec<&str> = expected[1..]
        .iter()
        .chain(expected[..1].iter())
        .copied()
        .collect();
    for want in &cycle_order {
        writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "cycle_model");
        let last = frames.last().unwrap();
        assert_eq!(last["data"]["model"], *want);
        assert_eq!(last["data"]["scoped"], true);
    }

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_models_flag_pattern_level_suffix_pins_that_models_thinking_level_on_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-opus-4-8:high,gpt-4o",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Pin to the unpinned scoped entry first so cycling onto the pinned one is unambiguous.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-4o" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");

    // Cycling onto "claude-opus-4-8:high" must land with reasoning_effort "high", the level its
    // `--models` pattern pinned — not whatever level happened to be active before.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], "claude-opus-4-8");
    assert_eq!(last["data"]["reasoning_effort"], "high");

    // Cycling onto the unpinned "gpt-4o" must not carry the pinned level along — it keeps whatever
    // was already active (still "high" here, since nothing unpins it), matching pi's own
    // "unpinned entries inherit the session's current level" rule rather than resetting to off.
    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], "gpt-4o");
    assert_eq!(last["data"]["reasoning_effort"], "high");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_models_flag_rejects_an_invalid_thinking_level_suffix_as_part_of_the_literal_id() {
    // "claude-opus-4-8:bogus" has no valid thinking-level suffix, so the whole string is kept as a
    // literal id (pi's own scope-mode fallback) rather than silently dropping ":bogus" — since our
    // catalog match is glob-only, a literal that doesn't equal any catalog entry is still forwarded
    // verbatim (see `available_models`'s "hint, not an allowlist" contract), so cycling reaches it too.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--models",
            "claude-opus-4-8:bogus",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "cycle_model");
    assert_eq!(
        frames.last().unwrap()["data"]["model"],
        "claude-opus-4-8:bogus"
    );

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
fn serve_starts_clamped_not_off_for_a_model_that_cannot_disable_reasoning() {
    // The CRITICAL bug this closes: a session on a model with a reasoning mechanism it can't
    // explicitly disable (`gpt-5-codex`: `reasoning_disableable == false`) must never start at the
    // stored level `Off` — that would silently omit the `reasoning` field from every request and let
    // the provider apply its own hidden default effort, with the operator believing reasoning is off.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd_with_model(bin, &base, &session_file, "gpt-5-codex")
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(
        data["thinking_level"], "minimal",
        "gpt-5-codex's floor is minimal; a fresh session with no --reasoning-effort must start \
         there, not at off: got {data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_model_reclamps_off_when_switching_onto_a_non_disableable_model() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    // `claude-test` is disable-capable, so the session starts at a legal "off".
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["data"]["thinking_level"], "off");

    // Switching to a model that can't disable reasoning must bump the still-stored "off" up to that
    // model's own floor, not silently carry an illegal level across the switch.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-5-codex" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["reasoning_effort"], "minimal", "got: {data:#?}");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["thinking_level"],
        "minimal",
        "get_state must reflect the re-clamped level too, not just the set_model response"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_cycle_thinking_level_never_gets_stuck_for_a_model_without_xhigh_or_off() {
    // Regression guard for a bug the naive fix would have introduced: a plain `level.next()` then
    // re-clamp bounces forever between `high` and a re-clamped `xhigh` for a model lacking xhigh
    // support, since `xhigh` always clamps back down to the very `high` it started from. This model
    // additionally can't reach `off` at all (`reasoning_disableable == false`), so the full available
    // ladder is exactly minimal/low/medium/high, wrapping.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd_with_model(bin, &base, &session_file, "gpt-5-codex")
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Starts clamped at "minimal" (see the dedicated startup-clamp test above).
    let expected = ["low", "medium", "high", "minimal", "low", "medium", "high"];
    for level in expected {
        writeln!(stdin, "{}", json!({ "type": "cycle_thinking_level" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "cycle_thinking_level");
        let data = &frames.last().unwrap()["data"];
        assert_eq!(data["level"], level, "got: {data:#?}");
    }

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
fn serve_compact_forwards_custom_instructions_to_the_summarization_call() {
    // Track M14: `compact`'s `custom_instructions` must actually reach the summarization model call —
    // matching pi's own `compact(customInstructions)` — not just be silently ignored.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // Two ordinary conversational turns (4 messages: user/assistant/user/assistant) build up enough
    // history for `find_cut` to find a real cut point once `--compaction-keep-recent-tokens` is tiny;
    // the third response is what the manual `compact` call's own summarization request receives.
    let (base, bodies) = spawn_model_server(vec![
        turn_text("answer one"),
        turn_text("answer two"),
        turn_text("SUMMARY"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--compaction-keep-recent-tokens",
            "1",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["hello", "again"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "compact",
            "custom_instructions": "keep every detail about the auth refactor"
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "compact");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        frames.last().unwrap()["data"]["compacted"],
        true,
        "{frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .any(|b| b.contains("Additional focus: keep every detail about the auth refactor")),
        "the custom instructions must reach the summarization call: {recorded:#?}"
    );
}

#[test]
fn serve_proactively_compacts_a_resumed_large_session_on_its_very_next_prompt() {
    // B-M14 pi-parity gap (fixed): a large session persisted by an *earlier* process (no live
    // `Session` in memory to carry `last_input_tokens` forward) used to resume with that field at
    // its zero default — `SessionStore::open` never restored it from the persisted transcript — so
    // `should_compact` couldn't fire until a fresh turn produced real usage, one whole turn later
    // than it should. Matches pi's own `pre-prompt-compaction-no-continue` regression: the very
    // first prompt sent to a resumed, already-over-threshold session must trigger compaction before
    // that prompt's own answer, not after some wasted extra turn.
    let dir = tempfile::tempdir().unwrap();

    // Seed the session file directly (as an earlier, now-exited process would have left it) with
    // enough text that its char/4 estimate comfortably exceeds the tiny threshold below — four
    // messages so `find_cut` (which declines short conversations) has a real boundary to find.
    let session_file = {
        use agent_core::{ContentBlock, Message, Session};
        use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};

        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let mut seed = Session::new();
        seed.user("u".repeat(400)); // ~100 estimated tokens
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        store.append_new(&seed.messages).unwrap();
        store.path().to_string_lossy().into_owned()
    };

    // Scripted in order: the proactive compaction's own summarization call, then the real answer to
    // the new prompt — if compaction didn't fire before the prompt's own turn, the second scripted
    // response would be consumed as the (unsummarized) prompt's answer instead, and this test's
    // assertions on call count / compacted-event ordering would catch the mismatch.
    let (base, bodies) = spawn_model_server(vec![turn_text("SUMMARY"), turn_text("answered")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--context-window",
            "200",
            "--compaction-reserve-tokens",
            "50",
            "--compaction-keep-recent-tokens",
            "1",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // The very first prompt this (freshly-spawned, freshly-resumed) process ever sends.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "one more thing" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    drop(stdin);
    child.wait().unwrap();

    // Exactly two model calls: the proactive compaction, then the real answer — not three (which
    // would mean compaction was skipped this turn and only caught up reactively on an overflow, or
    // deferred to a second prompt).
    assert_eq!(
        bodies.lock().unwrap().len(),
        2,
        "compaction must fire on this very first prompt, not after it"
    );
    let compacted = frames.iter().any(|f| {
        f.get("type").and_then(Value::as_str) == Some("event")
            && f["event"]["kind"] == json!("compacted")
    });
    assert!(
        compacted,
        "expected a compacted event during this prompt's own processing: {frames:#?}"
    );
}

#[test]
fn serve_compact_preserves_pre_compaction_entries_in_get_tree() {
    // F-M3 (pi: rpc.test.ts:328-340): the storage layer's own non-destructive-compaction guarantee
    // (`session_store.rs`'s `rewrite_compacted_preserves_folded_messages_and_records_provenance`) was
    // never proven end-to-end through the live RPC surface — a `compact` command followed by `get_tree`
    // must still show the folded, pre-compaction messages (matching this codebase's append-only
    // compaction posture: folded away from the *active* path, never deleted from disk).
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("answer one"),
        turn_text("answer two"),
        turn_text("SUMMARY"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--compaction-keep-recent-tokens",
            "1",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["pre-compaction-hello", "pre-compaction-again"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(stdin, "{}", json!({ "type": "compact" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "compact");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        frames.last().unwrap()["data"]["compacted"],
        true,
        "{frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_tree" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let nodes = frames.last().unwrap()["data"]["nodes"].as_array().unwrap();

    // Every pre-compaction message is still readable by its original node, even though none of them
    // are reachable from the new active tip anymore (the compacted summary starts a fresh, detached
    // chain — see `SessionStore::rewrite_compacted`'s own doc comment).
    for text in [
        "pre-compaction-hello",
        "answer one",
        "pre-compaction-again",
        "answer two",
    ] {
        assert!(
            nodes
                .iter()
                .any(|n| n["preview"].as_str().is_some_and(|p| p.contains(text))),
            "folded pre-compaction message {text:?} must still be present in get_tree: {nodes:#?}"
        );
    }

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
fn serve_auto_retries_a_whole_run_after_mid_stream_retry_is_exhausted() {
    // agent-core's own mid-stream retry (`MAX_MID_STREAM_RETRIES = 3`) gives up after 1 initial + 3
    // retried attempts, all against a stream that dies before `message_stop` — 4 requests total,
    // exhausting that layer entirely and returning `Err` to `run_events_steered`. This is the whole-run
    // auto-retry layer's job to pick up from there: automatically re-invoke the run against the same
    // session one more time, which succeeds — a 5th request the model server actually sees, and no
    // second user turn appended (still the same `prompt`).
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_text("recovered"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let auto_retry_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry"))
        .collect();
    assert_eq!(
        auto_retry_frames.len(),
        1,
        "expected exactly one auto_retry notice, got: {frames:#?}"
    );
    assert_eq!(auto_retry_frames[0]["attempt"], 1);
    assert_eq!(auto_retry_frames[0]["max_attempts"], 3);
    assert!(
        auto_retry_frames[0]["error"]
            .as_str()
            .unwrap()
            .contains("stream ended"),
        "got: {auto_retry_frames:#?}"
    );

    // The terminal notice for the retry sequence: the retried attempt succeeded.
    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(
        auto_retry_end_frames.len(),
        1,
        "expected exactly one auto_retry_end notice, got: {frames:#?}"
    );
    assert_eq!(auto_retry_end_frames[0]["success"], true);
    assert_eq!(auto_retry_end_frames[0]["attempt"], 1);
    assert!(
        auto_retry_end_frames[0].get("final_error").is_none(),
        "a successful retry must not carry a final_error: {:?}",
        auto_retry_end_frames[0]
    );

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(
        bodies.lock().unwrap().len(),
        5,
        "4 exhausted mid-stream attempts + 1 successful whole-run retry"
    );

    // The recovered turn's own text must have reached the transcript — proof the retry replayed the
    // *same* user turn rather than silently dropping it or duplicating it.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = &frames.last().unwrap()["data"]["messages"];
    let dump = messages.to_string();
    assert_eq!(
        messages.as_array().unwrap().len(),
        2,
        "exactly one user turn + one assistant turn, no duplicate: {dump}"
    );
    assert!(dump.contains("recovered"), "got: {dump}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_whole_run_retry_recovery_attempt_can_itself_dispatch_tool_calls() {
    // B-L3 pi-parity test gap (fixed): every existing whole-run-retry test recovers into a plain text
    // turn. Nothing proved the retried attempt can continue normally into a *tool-dispatch* turn — a
    // structurally different path through `run_events_steered` (another model round trip after the
    // tool result, real `bash` execution, a second assistant message).
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_tool_use(
            "toolu_retry",
            "bash",
            &json!({ "command": "echo recovered-tool" }).to_string(),
        ),
        turn_text("done after tool"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(
        bodies.lock().unwrap().len(),
        6,
        "4 exhausted mid-stream attempts + the recovered tool-call turn + its follow-up text turn"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("recovered-tool"), "tool actually ran: {dump}");
    assert!(dump.contains("done after tool"), "got: {dump}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_whole_run_retry_succeeds_on_its_second_attempt_not_its_first() {
    // B-L4 pi-parity test gap (fixed): every existing whole-run-retry test hits one of the two
    // boundary cases (first retry attempt succeeds, or all `MAX_RUN_RETRIES` attempts are exhausted).
    // This pins the middle case — the first whole-run retry attempt *also* fails, and the second one
    // recovers — proving the loop actually keeps going past attempt 1 rather than only ever handling
    // "succeeds immediately" or "never succeeds".
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    // Each block of 4 truncated responses exhausts one whole-run attempt's own mid-stream retry
    // budget (`MAX_MID_STREAM_RETRIES = 3`, so 1 initial + 3 retries per attempt) before the whole-run
    // layer re-invokes the entire run from scratch, with a fresh mid-stream budget of its own.
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_text("recovered on the second whole-run attempt"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let auto_retry_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry"))
        .collect();
    assert_eq!(
        auto_retry_frames.len(),
        2,
        "expected two auto_retry notices (attempt 1 failed too, attempt 2 was tried): {frames:#?}"
    );
    assert_eq!(auto_retry_frames[0]["attempt"], 1);
    assert_eq!(auto_retry_frames[1]["attempt"], 2);

    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(
        auto_retry_end_frames.len(),
        1,
        "only one terminal notice, once the sequence actually settles: {frames:#?}"
    );
    assert_eq!(auto_retry_end_frames[0]["success"], true);
    assert_eq!(
        auto_retry_end_frames[0]["attempt"], 2,
        "must report which attempt actually succeeded, not just that one eventually did"
    );

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(bodies.lock().unwrap().len(), 9);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_abort_retry_interrupts_a_pending_whole_run_retry_backoff() {
    // Same failure shape as the auto-retry test above (agent-core's own mid-stream retry exhausts
    // after 4 attempts, handing an `Err` to the whole-run retry layer) — but here the client cancels
    // the pending backoff instead of letting it run its course. Confirms the backoff wait is
    // genuinely interruptible (not a bare `sleep` nothing can touch) and that cancelling it surfaces
    // the real underlying error rather than either hanging for the full delay or silently retrying
    // anyway.
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_text("would have recovered"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();

    // Read frames one at a time until the `auto_retry` notice arrives — sent right before the
    // 2-second whole-run backoff wait starts, *after* agent-core's own mid-stream retry has already
    // spent its own ~1.75s of internal backoff exhausting itself — then immediately cancel it. Timing
    // starts here, not at the `prompt` send, so the assertion below isolates just the whole-run
    // backoff-interruption latency from that unrelated, expected mid-stream delay.
    let mut line = String::new();
    loop {
        line.clear();
        assert!(
            stdout.read_line(&mut line).unwrap() > 0,
            "stdout closed before an auto_retry notice arrived"
        );
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(trimmed).unwrap();
        if v.get("type").and_then(Value::as_str) == Some("auto_retry") {
            break;
        }
    }
    let start = std::time::Instant::now();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "abort_retry", "id": "cancel-1" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let abort_frames = read_until_response(&mut stdout, "abort_retry");
    assert_eq!(
        abort_frames.last().unwrap()["success"],
        true,
        "{abort_frames:#?}"
    );

    // The still-pending `prompt` must resolve right after — not wait out the full 2s backoff — with
    // the *original* mid-stream error, and the 5th (would-be-recovering) request must never fire.
    let frames = read_until_response(&mut stdout, "prompt");
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(1000),
        "abort_retry must interrupt the 2s backoff near-instantly, not wait most/all of it out: \
         took {elapsed:?}"
    );
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "got: {response:#?}");
    assert!(
        response["error"].as_str().unwrap().contains("stream ended"),
        "must surface the real underlying error, not a synthetic cancellation: {response:#?}"
    );
    assert_eq!(
        bodies.lock().unwrap().len(),
        4,
        "the cancelled retry must never have fired the 5th, would-be-recovering request: {:?}",
        bodies.lock().unwrap()
    );

    // The terminal notice for the retry sequence: it never got to retry — the backoff was cancelled.
    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(
        auto_retry_end_frames.len(),
        1,
        "expected exactly one auto_retry_end notice, got: {frames:#?}"
    );
    assert_eq!(auto_retry_end_frames[0]["success"], false);
    assert_eq!(auto_retry_end_frames[0]["attempt"], 1);
    assert_eq!(auto_retry_end_frames[0]["final_error"], "retry cancelled");

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
            names.contains(&"skill:foo"),
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
fn serve_force_untrusted_overrides_a_persisted_trust_grant() {
    // Track L8: `--force-untrusted` must win even over a *persisted* `agent trust <path>` grant, not
    // just the absence of `--trust-project` — the whole point is overriding an operator's standing
    // trust decision for one run (testing untrusted behavior, or extra caution on a checkout that
    // happens to live under an already-trusted parent).
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/foo");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: foo\ndescription: a test skill\n---\nDo the foo thing.",
    )
    .unwrap();

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Persist trust for `dir` via the real `trust` subcommand — the same allowlist `serve` itself
    // reads from, not a hand-rolled fixture.
    let trust_output = Command::new(bin)
        .args(["trust", dir.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust_output.status.success());

    // Control: without --force-untrusted, the persisted trust is honored — the skill is visible.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let session_file = dir.path().join("s1.jsonl").to_string_lossy().into_owned();
        let mut cmd = serve_cmd(bin, &base, &session_file);
        cmd.current_dir(dir.path()).env("HOME", home.path());
        let mut child = cmd.spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());

        writeln!(stdin, "{}", json!({ "type": "get_commands" })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "get_commands");
        let names: Vec<&str> = frames.last().unwrap()["data"]["commands"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c["name"].as_str())
            .collect();
        assert!(
            names.contains(&"skill:foo"),
            "the persisted trust grant should be honored: {names:?}"
        );

        drop(stdin);
        child.wait().unwrap();
    }

    // With --force-untrusted: the same persisted trust grant must be overridden.
    {
        let (base, _bodies) = spawn_model_server(vec![]);
        let session_file = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
        let mut cmd = serve_cmd(bin, &base, &session_file);
        cmd.arg("--force-untrusted")
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
        assert!(
            commands.is_empty(),
            "--force-untrusted must override the persisted trust grant: {commands:?}"
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
        names.contains(&"skill:mine"),
        "an untrusted project must still advertise the user-global skill: {names:?}"
    );
    assert!(
        !names.contains(&"skill:theirs"),
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
        .filter(|c| c["name"].as_str() == Some("skill:dup"))
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
fn serve_get_commands_reports_the_prompt_templates_own_description_not_its_argument_hint() {
    // A prompt template's `get_commands` entry must surface its `description:` frontmatter, not the
    // separate `argument-hint:` field — the two are deliberately distinct (a hint like "<file>" isn't
    // a human-readable summary a client's command palette should display as one).
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(
        prompt_dir.join("fix.md"),
        "---\nargument-hint: <file>\ndescription: Fix a bug in the given file\n---\nFix $1.",
    )
    .unwrap();

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
    let commands = frames.last().unwrap()["data"]["commands"]
        .as_array()
        .unwrap();
    let fix = commands
        .iter()
        .find(|c| c["name"].as_str() == Some("fix"))
        .unwrap_or_else(|| panic!("prompt template \"fix\" not listed: {commands:?}"));
    assert_eq!(
        fix["description"].as_str(),
        Some("Fix a bug in the given file"),
        "description field must be the template's own description, not its argument hint: {fix:?}"
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
    let new_session_data = &frames.last().unwrap()["data"];
    let second_id = new_session_data["session_id"].as_str().unwrap().to_string();
    assert_ne!(first_id, second_id, "new_session must mint a new id");
    assert_eq!(
        new_session_data["parent"], first_id,
        "the fresh session's lineage marker must point back at whatever was active before it: \
         {new_session_data:#?}"
    );

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
    // The lineage marker also persists to disk and survives into `list_sessions`, not just the
    // `new_session` response.
    let second_session = sessions
        .iter()
        .find(|s| s["id"] == second_id)
        .expect("session 2 must be listed");
    assert_eq!(
        second_session["parent"], first_id,
        "list_sessions must surface the persisted lineage marker too: {second_session:#?}"
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
fn serve_clone_forks_the_current_session_at_its_current_tip() {
    // Track L17: pi's own `clone` command — a thin, argument-free alias over `fork` at the session's
    // current tip. This crate's `fork` already defaults to exactly that when called with no
    // `upto`/`target_id`, so `clone` must behave identically to a bare `fork`, just under pi's name.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let source_id = ready["session_id"].as_str().unwrap().to_string();

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "clone" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "clone");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    let clone_id = resp["data"]["session_id"].as_str().unwrap().to_string();
    assert_ne!(clone_id, source_id, "a clone is a distinct session");

    // The clone must carry the source's full transcript (the same active-path-at-current-tip a bare
    // `fork` would copy) — not an empty session.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("first answer") && dump.contains("\"hi\""),
        "clone must carry the source session's full transcript: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_new_session_parent_session_overrides_the_default_lineage() {
    // Track L17: `new_session`'s `parent` previously always pointed at whatever session was active
    // immediately before the call, with no way to link a fresh session to a *different* one instead.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let active_id = ready["session_id"].as_str().unwrap().to_string();

    // A `new_session` with no override still links to whatever was active — the pre-existing default,
    // unchanged by adding the override.
    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let default_data = &frames.last().unwrap()["data"];
    assert_eq!(
        default_data["parent"], active_id,
        "omitting parent_session must keep linking to whatever was active: {default_data:#?}"
    );

    // An explicit `parent_session` naming an unrelated id wins outright, even though it names neither
    // the session that was active before this call nor the one just created above.
    let explicit_parent = "some-other-session-id-entirely";
    writeln!(
        stdin,
        "{}",
        json!({ "type": "new_session", "parent_session": explicit_parent })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let override_data = &frames.last().unwrap()["data"];
    assert_eq!(
        override_data["parent"], explicit_parent,
        "an explicit parent_session must override the default lineage: {override_data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_delete_session_soft_deletes_another_session_but_refuses_the_active_one() {
    // Track L5: `SessionRepo::delete` had no RPC command wired to it at all — genuinely unreachable in
    // production. Deleting a *different* session must remove it from `list_sessions` (moved to
    // `.trash`, not gone outright); deleting the *currently active* one must be refused, since that
    // would move the file out from under the in-memory `Session` a client is still using.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    let first_id = ready["session_id"].as_str().unwrap().to_string();

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // A second session becomes active; the first is now just another entry in the repo.
    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let second_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(first_id, second_id);

    // Refuses to delete the currently active session (session 2).
    writeln!(
        stdin,
        "{}",
        json!({ "type": "delete_session", "session_id": second_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "delete_session");
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["success"], false,
        "must refuse to delete the currently active session: {resp:#?}"
    );

    // Deletes the *other* session (session 1) successfully.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "delete_session", "session_id": first_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "delete_session");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    // No longer listed.
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_sessions");
    let sessions = frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap();
    assert!(
        !sessions.iter().any(|s| s["id"] == first_id),
        "the deleted session must no longer be listed: {sessions:#?}"
    );

    // Idempotent: deleting it again (already gone) is still a success, not an error.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "delete_session", "session_id": first_id })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "delete_session");
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "deleting an already-deleted session must be a no-op success: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_fork_by_target_id_reaches_an_off_active_path_branch() {
    // `fork`'s `upto` count only ever copies a prefix of whatever is *currently* the active path — it
    // has no way to reach a branch the client has since navigated away from. `target_id` does: fork
    // directly from any tree entry, on or off the active path, without first `switch_branch`-ing to it
    // (which would itself mutate the live session just to preview/fork from it).
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("a-reply"), // prompt "a"
        turn_text("b-reply"), // prompt "b"
        turn_text("c-reply"), // prompt "c"
        turn_text("d-reply"), // prompt "d", after switching back to a's leaf
    ]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["a", "b", "c"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = frames.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        messages.len(),
        6,
        "3 user + 3 assistant turns: {messages:#?}"
    );
    let ids: Vec<String> = messages
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    // ids[0] = user "a", ids[1] = assistant "a-reply", ids[2] = user "b", ids[3] = assistant
    // "b-reply", ids[4] = user "c", ids[5] = assistant "c-reply".

    // Switch back to a's own leaf and append "d" — b/c's turns fall off the active path entirely.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": ids[1], "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    assert_eq!(
        read_until_response(&mut stdout, "switch_branch")
            .last()
            .unwrap()["success"],
        true
    );
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "d" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // `get_fork_messages` targeting b's own user turn (off-branch) with `before:true` previews just
    // [a, a-reply] — b's user message excluded (its own parent is a-reply, so excluding just b's
    // message, not "the whole b pair", is exactly what `before` means: fork right before this entry).
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let count_before_preview = read_until_response(&mut stdout, "list_sessions")
        .last()
        .unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_fork_messages", "target_id": ids[2], "before": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let preview = read_until_response(&mut stdout, "get_fork_messages");
    let preview_messages = preview.last().unwrap()["data"]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        preview_messages.len(),
        2,
        "before:true at b's own user turn excludes it, leaving just a's turn: {preview_messages:#?}"
    );

    // The `target_id`-based preview must not have created a session file either — it previously did,
    // permanently polluting `list_sessions` on every branch-point preview (the documented common case).
    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let count_after_preview = read_until_response(&mut stdout, "list_sessions")
        .last()
        .unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        count_after_preview, count_before_preview,
        "get_fork_messages with target_id must not create a session file"
    );

    // A real `fork` at c's assistant reply (off-branch), `before:false` (the default) — includes it,
    // reaching the full original a->b->c line, none of which is the *current* active path (a->d).
    writeln!(stdin, "{}", json!({ "type": "fork", "target_id": ids[5] })).unwrap();
    stdin.flush().unwrap();
    let fork_response = read_until_response(&mut stdout, "fork");
    assert_eq!(fork_response.last().unwrap()["success"], true);

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("a-reply") && dump.contains("b-reply") && dump.contains("c-reply"));
    assert!(
        !dump.contains("d-reply"),
        "the fork reached the off-branch a->b->c line, not the active a->d one: {dump}"
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

/// A model server whose first `fast.len()` requests get an instant response, and whose next request
/// (the summarization call a `switch_branch{summarize:true}` triggers) sends only a partial SSE body —
/// proving the request genuinely reached the server and started streaming — then stalls for `stall`
/// before completing, giving a test a reliable window to `abort` a provably in-flight call instead of
/// racing a near-instant local round trip.
fn spawn_model_server_with_stalled_response(
    fast: Vec<String>,
    stall: std::time::Duration,
) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for resp in fast {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{resp}"
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            }
        }
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let preamble = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
                data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n";
            let _ = stream.write_all(preamble.as_bytes());
            let _ = stream.flush();
            std::thread::sleep(stall);
            // Finishes the turn normally as a fallback safety net in case a test using this doesn't
            // abort before `stall` elapses — a silently-hanging server would fail such a test far more
            // confusingly than a completed-but-too-late response would.
            let rest = "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
                data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"recap\"}}\n\n\
                data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
                data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n\
                data: {\"type\":\"message_stop\"}\n\n";
            let _ = stream.write_all(rest.as_bytes());
            let _ = stream.flush();
        }
    });
    format!("http://{addr}")
}

#[test]
fn serve_switch_branch_abort_cancels_summarization_and_leaves_session_unchanged() {
    use std::time::{Duration, Instant};

    // pi-parity fix (`packages/coding-agent/test/agent-session-tree-navigation.test.ts:175-212`,
    // "should handle abort during summarization"): `switch_branch{summarize:true}`'s LLM call used to
    // run on a fresh, unreachable `CancellationToken` with no way for a client `abort` to ever reach
    // it — the whole RPC loop just blocked until the call finished, however long that took. This
    // proves `abort` actually interrupts a provably in-flight branch summarization promptly, the
    // response reports `cancelled`/`aborted` rather than an error, and the session (branches/leaf) is
    // left completely unchanged, matching pi's `{cancelled:true, aborted:true, summaryEntry:undefined}`
    // contract.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let base = spawn_model_server_with_stalled_response(
        vec![turn_text("first answer"), turn_text("second answer")],
        Duration::from_secs(5),
    );

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

    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches_before = frames.last().unwrap()["data"]["branches"].clone();
    assert_eq!(branches_before.as_array().unwrap().len(), 1);

    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 4, "expected 4 persisted messages: {ids:?}");
    let rewind_to = ids[1].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": rewind_to, "summarize": true })
    )
    .unwrap();
    stdin.flush().unwrap();

    // Give the summarization request time to actually reach the stalled server (it writes a partial
    // body immediately on accept, well before its 5s stall completes) before aborting.
    std::thread::sleep(Duration::from_millis(300));
    let start = Instant::now();
    writeln!(stdin, "{}", json!({ "type": "abort", "id": "a1" })).unwrap();
    stdin.flush().unwrap();

    let abort_frames = read_until_response(&mut stdout, "abort");
    assert_eq!(abort_frames.last().unwrap()["success"], true);

    let frames = read_until_response(&mut stdout, "switch_branch");
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(3),
        "abort must interrupt the stalled summarization promptly, not wait out its 5s stall: {elapsed:?}"
    );
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["success"], true,
        "a cancelled switch is not an RPC-level failure: {resp:#?}"
    );
    assert_eq!(resp["data"]["cancelled"], true);
    assert_eq!(resp["data"]["aborted"], true);

    // The session must be completely unchanged: still one branch, still 4 messages, no partial
    // summary entry anywhere in the persisted file.
    writeln!(stdin, "{}", json!({ "type": "list_branches" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "list_branches");
    let branches_after = frames.last().unwrap()["data"]["branches"].clone();
    assert_eq!(
        branches_after, branches_before,
        "branch structure must be untouched by a cancelled switch"
    );
    assert_eq!(message_ids(&session_file).len(), 4);
    let raw = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        !raw.contains("\"branch_summary\""),
        "a cancelled summarization must not persist a partial summary entry:\n{raw}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_restores_the_model_active_on_that_branch() {
    // `set_model` between two turns records a branch-local change (H6); navigating back to the point
    // right before that change must restore the model that was actually active there, not silently
    // continue with whatever model the process's global setting has since moved on to.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("first answer"),  // prompt "first" on the original model
        turn_text("second answer"), // prompt "second" on the switched-to model
        turn_text("third answer"),  // prompt "third" after switching back
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // The tip right now (after "first answer") is where the upcoming model change is anchored.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 2, "expected 2 persisted messages: {ids:?}");
    let anchor = ids[1].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "claude-test-2" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "second" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Navigate back to right after "first answer" — before the model change ever happened.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": anchor, "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "got: {resp:#?}");
    assert_eq!(
        resp["data"]["model"], "claude-test",
        "switching back before the set_model must restore the original model: {resp:#?}"
    );

    // `get_state` (not just the switch_branch response) must reflect the restored model too.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["data"]["model"], "claude-test");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_branch_resets_thinking_level_instead_of_bleeding_a_sibling_branchs_setting() {
    // Track L4: a `set_reasoning_effort` recorded on one branch must not silently keep applying after
    // switching to a point that never had it — that point genuinely never had a level recorded, so it
    // must resolve to the *process's own starting level* (here, the default "off"), not whatever the
    // global runtime setting happens to still be from the branch just abandoned.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "first" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // The tip right now (after "first answer") never has — and never will have — its own recorded
    // thinking-level change; the upcoming `set_reasoning_effort` is anchored *after* it.
    let ids = message_ids(&session_file);
    assert_eq!(ids.len(), 2, "expected 2 persisted messages: {ids:?}");
    let anchor = ids[1].clone();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_reasoning_effort", "effort": "high" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_reasoning_effort");
    assert_eq!(frames.last().unwrap()["success"], true, "got: {frames:#?}");
    assert_eq!(frames.last().unwrap()["data"]["level"], "high");

    // Switch back to the anchor — a point that predates the level change entirely, and has no
    // ThinkingLevelChange of its own anchored at-or-before it.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_branch", "target_id": anchor, "summarize": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_branch");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "got: {resp:#?}");
    assert_eq!(
        resp["data"]["reasoning_effort"], "off",
        "switching to a point with no recorded level change must reset to the process's own \
         starting level, not bleed the abandoned branch's \"high\": {resp:#?}"
    );

    // `get_state` (not just the switch_branch response) must reflect the reset level too.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["data"]["thinking_level"], "off");

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
fn serve_bash_records_its_result_into_session_context_by_default() {
    // pi-parity fix (M13): the host `bash` RPC command never touched `session` at all — the calling
    // client saw the result, but the model never would on a later turn. Matches pi's own
    // `recordBashResult`, which records by default.
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
        json!({ "type": "bash", "command": "printf host-bash-context-marker" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "bash");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("host-bash-context-marker"),
        "the host bash command's own output must reach session context: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_bash_exclude_from_context_keeps_the_session_untouched() {
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
        json!({
            "type": "bash",
            "command": "printf should-not-reach-context",
            "exclude_from_context": true,
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    let resp = frames.last().unwrap();
    assert_eq!(resp["data"]["result"], "should-not-reach-context");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        !dump.contains("should-not-reach-context"),
        "exclude_from_context: true must keep the session untouched: {dump}"
    );

    drop(stdin);
    child.wait().unwrap();
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
#[cfg(unix)]
fn serve_bash_shell_path_overrides_the_auto_resolved_shell() {
    if !std::path::Path::new("/bin/sh").exists() {
        return; // no alternate shell on this host to prove the override actually took effect
    }
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--bash-shell-path", "/bin/sh"]);
    let mut child = cmd.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // `$BASH_VERSION` is set only by bash itself; `/bin/sh` on this host is dash, which leaves it
    // unset — the POSIX `${VAR:-default}` expansion below works identically under both, so the
    // *value* it prints is the only thing that can differ.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "bash", "command": "echo ${BASH_VERSION:-no-bash}" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "bash");
    drop(stdin);
    child.wait().unwrap();

    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "frames: {frames:#?}");
    assert_eq!(resp["data"]["result"], "no-bash\n", "frames: {frames:#?}");
}

#[test]
fn serve_fails_fast_when_bash_shell_path_does_not_exist() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(["--bash-shell-path", "/no/such/shell-binary"]);
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    drop(child.stdin.take()); // the process must exit before ever trying to read a command line
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success(), "stderr: {stderr}");
    assert!(stderr.contains("--bash-shell-path"), "stderr: {stderr}");
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
fn serve_get_state_reports_pending_tool_ids_while_a_tool_is_running() {
    use std::time::Duration;

    // B-L1 pi-parity gap (fixed): pi's `agent.state.pendingToolCalls` (a live, in-process reactive
    // set) has no RPC equivalent — a client had to reconstruct "which calls are still in flight"
    // itself from the raw event stream. `get_state` now mirrors it directly, live, mid-run.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let turn1 = turn_tool_use(
        "toolu_pending",
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
    std::thread::sleep(Duration::from_millis(150)); // mid-`sleep 0.5`, the bash call is in flight

    writeln!(stdin, "{}", json!({ "type": "get_state", "id": "mid" })).unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");

    let mid_resp = frames
        .iter()
        .find(|f| f["command"] == "get_state" && f["id"] == "mid")
        .unwrap_or_else(|| panic!("no mid-run get_state response: {frames:#?}"));
    assert_eq!(
        mid_resp["data"]["pending_tool_ids"],
        json!(["toolu_pending"]),
        "the running bash call must be reported as pending: {mid_resp:#?}"
    );

    // The whole run has now finished (both turns) on this same live process — a fresh `get_state`
    // proves `tool_ended` actually cleared the id, not just that `tool_started` populated it.
    writeln!(stdin, "{}", json!({ "type": "get_state", "id": "after" })).unwrap();
    stdin.flush().unwrap();
    let after_frames = read_until_response(&mut stdout, "get_state");
    let after_resp = after_frames.last().unwrap();
    assert_eq!(after_resp["id"], "after");
    assert_eq!(
        after_resp["data"]["pending_tool_ids"],
        json!([]),
        "the completed call must no longer be pending: {after_resp:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_state_reports_runtime_settings_and_queue_depth() {
    // `get_state` must carry the runtime-mutable settings (thinking level, auto-compaction, auto-retry,
    // queue mode) and the current queue depth — pi's `get_state` carries the same shape
    // (`thinkingLevel`/`autoCompactionEnabled`/`steeringMode`/`followUpMode`/`pendingMessageCount`), and
    // a client shouldn't need a separate round trip (or its own copy of the defaults) to render them.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Defaults, nothing queued yet.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["thinking_level"], "off", "{data:#?}");
    assert_eq!(data["auto_compaction"], true, "{data:#?}");
    assert_eq!(data["auto_retry"], true, "{data:#?}");
    assert_eq!(data["steering_mode"], "one_at_a_time", "{data:#?}");
    assert_eq!(data["follow_up_mode"], "one_at_a_time", "{data:#?}");
    assert_eq!(data["pending_messages"], 0, "{data:#?}");

    // Queue two follow-ups (idle `follow_up`, no prompt in flight) and flip auto_compaction/queue_mode;
    // `get_state` must reflect all of it without any prompt having run.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "message": "first" })
    )
    .unwrap();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "message": "second" })
    )
    .unwrap();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": false })
    )
    .unwrap();
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["auto_compaction"], false, "{data:#?}");
    assert_eq!(
        data["pending_messages"], 2,
        "two queued follow-ups: {data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
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
fn serve_switch_session_restores_the_reopened_sessions_own_model() {
    // pi-parity fix: `current_model` used to be seeded once from the server's own `--model` startup
    // flag and never reconciled with the session actually being switched to — reattaching to a session
    // last driven on a different model silently kept using the server's startup model instead, with no
    // warning. `switch_branch` already restored this correctly within one session's tree; this proves
    // `switch_session` (a *different* session's own store entirely) now does too.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions");
    std::fs::create_dir_all(&session_dir).unwrap();
    let other_header = json!({
        "type": "session",
        "id": "other-model-target",
        "created_at": 1,
        "cwd": "/definitely/does/not/exist/beyond-ai-agent-test-fixture",
        "model": "claude-test-restored",
    });
    std::fs::write(
        session_dir.join("1_other-model-target.jsonl"),
        format!("{other_header}\n"),
    )
    .unwrap();

    // The server starts on `claude-test` (see `serve_dir_cmd`) — deliberately different from the
    // planted session's own `claude-test-restored`, so a successful restore is unambiguous.
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir.to_string_lossy())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": "other-model-target" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_session");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(
        response["data"]["model"], "claude-test-restored",
        "switch_session must restore the target session's own model, not keep the server's startup \
         model: {response:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let response = frames.last().unwrap();
    assert_eq!(
        response["data"]["model"], "claude-test-restored",
        "the restored model must also be what a subsequent turn would actually use, not just what \
         the switch response reported: {response:#?}"
    );

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

#[test]
fn serve_reports_success_when_a_failed_checkpoint_is_superseded_by_a_successful_final_persist() {
    // LOW pi-parity gap (fixed): `persist_error` used to be set the moment any mid-run checkpoint
    // failed and never cleared, so a checkpoint hiccup early in a run made the terminal `prompt`
    // response report failure even when the run's actual final state was later persisted just fine.
    // Root-only environments can't exercise this (permission bits don't restrict root), so skip there.
    if std::env::var("USER").as_deref() == Ok("root") {
        return;
    }

    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "printf first-round-marker" }).to_string(),
    );
    let turn2 = turn_tool_use(
        "toolu_2",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Consume the ready frame — by now `Persistence::open` has already created the file with normal
    // permissions, so this doesn't race the file's own creation.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();

    // Make the session file read-only *before* sending the prompt, so the first round-trip's
    // mid-run checkpoint (fired right after `toolu_1` completes) fails to append to it.
    std::fs::set_permissions(&session_file, std::fs::Permissions::from_mode(0o444)).unwrap();

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();

    // The first round-trip (a fast `printf`) completes and its checkpoint attempt fails well within
    // this window; the second turn's `sleep 1` is still running when permissions are restored.
    std::thread::sleep(Duration::from_millis(500));
    std::fs::set_permissions(&session_file, std::fs::Permissions::from_mode(0o644)).unwrap();

    // The run ends once `sleep 1` finishes and the model's concluding "done" turn is emitted; the
    // unconditional final persist right after that must now succeed against the writable-again file.
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let response = frames.last().unwrap();
    assert_eq!(
        response["success"], true,
        "the run's true final state was persisted successfully — an earlier, superseded checkpoint \
         failure must not surface as a false failure: {response:#?}"
    );
    assert!(
        response.get("error").is_none() || response["error"].is_null(),
        "got: {response:#?}"
    );

    // And the final persist genuinely did land on disk, not just "no error was reported."
    let on_disk = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        on_disk.contains("first-round-marker"),
        "the final persist must have actually written the transcript: {on_disk}"
    );
}

#[test]
fn serve_stdout_stays_valid_json_even_when_a_load_warning_fires() {
    // A prior bug (output-guard gap): the tracing subscriber defaulted to stdout, the same stream
    // `serve`'s NDJSON protocol writes to. A `tracing::warn!` on a live path (here: `session_store.rs`
    // skipping an unparseable line while loading) would interleave raw log text into the protocol
    // stream, breaking any line-based JSON parser reading it. Plant a session file with exactly that
    // kind of corrupt line, run with `RUST_LOG=warn` so the warning is guaranteed to fire, and assert
    // every single stdout line still parses as JSON — while independently confirming (via captured
    // stderr) that the warning really did fire, so the assertion isn't vacuously true.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let header = json!({
        "type": "session",
        "id": "log-purity-session",
        "created_at": 1,
        "cwd": dir.path().to_string_lossy(),
        "model": "claude-test",
    });
    std::fs::write(
        &session_file,
        format!("{header}\nthis line is not valid JSON at all\n"),
    )
    .unwrap();

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file.to_string_lossy())
        .env("RUST_LOG", "warn")
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut stderr = BufReader::new(child.stderr.take().unwrap());

    // Deliberately NOT `read_until_response` — it silently skips a line that fails to parse as JSON,
    // which would make this test vacuously pass even with the bug reintroduced. Every line must parse.
    let mut line = String::new();
    let mut lines_seen = 0;
    loop {
        line.clear();
        if stdout.read_line(&mut line).unwrap() == 0 {
            panic!("stdout closed before a ready frame arrived");
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| panic!("stdout line was not valid JSON ({e}): {trimmed:?}"));
        lines_seen += 1;
        if v.get("type").and_then(Value::as_str) == Some("ready") {
            break;
        }
    }
    assert!(lines_seen >= 1);

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    loop {
        line.clear();
        if stdout.read_line(&mut line).unwrap() == 0 {
            panic!("stdout closed before the get_state response arrived");
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| panic!("stdout line was not valid JSON ({e}): {trimmed:?}"));
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some("get_state");
        if done {
            break;
        }
    }

    drop(stdin);
    child.wait().unwrap();

    let mut stderr_text = String::new();
    stderr.read_to_string(&mut stderr_text).unwrap();
    assert!(
        stderr_text.contains("skipping unparseable session entry line"),
        "the load warning must actually have fired (on stderr, not stdout) for this test to mean \
         anything: stderr was {stderr_text:?}"
    );
}

#[test]
fn new_session_reports_failure_and_leaves_the_old_session_active_when_persist_fails() {
    // `Persistence::new_session` used to unconditionally return a fresh, empty in-memory `Session`
    // even when the on-disk reset actually failed — reporting RPC success on a session that was never
    // really created, while `SessionStore`'s own persisted-message-count bookkeeping (correctly)
    // stayed untouched. A subsequent real `persist` would then see the small "new" in-memory message
    // count against that stale-large persisted count and silently no-op via `append_new`'s own dedup
    // guard, discarding every message of the "new" session with no error at all. The fix: report
    // failure over RPC, and leave the *old* session active rather than switching to a phantom one.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![turn_text("hello-reply")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hello" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let before = read_until_response(&mut stdout, "get_messages");
    let before_dump = before.last().unwrap()["data"]["messages"].to_string();
    assert!(
        before_dump.contains("hello-reply"),
        "the first prompt must have actually persisted: {before_dump}"
    );

    // Force the *next* write to `session_file` to fail: replace it with a directory of the same name.
    // `new_session`'s single-file-mode reset (`SessionStore::rewrite`) writes a temp file then renames
    // it onto this path — a rename can never land a file onto an existing directory, so this reliably
    // reproduces a real write failure without needing root or a special filesystem.
    std::fs::remove_file(&session_file).unwrap();
    std::fs::create_dir(&session_file).unwrap();

    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let new_session_response = read_until_response(&mut stdout, "new_session");
    let resp = new_session_response.last().unwrap();
    assert_eq!(
        resp["success"], false,
        "a failed on-disk reset must be reported as failure, not silent success: {resp:#?}"
    );
    assert!(
        resp.get("error").is_some(),
        "failure must carry an error message: {resp:#?}"
    );

    // The old session must still be the active one — not silently swapped for an empty one that was
    // never actually persisted.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let after = read_until_response(&mut stdout, "get_messages");
    let after_dump = after.last().unwrap()["data"]["messages"].to_string();
    assert_eq!(
        after_dump, before_dump,
        "a failed new_session must leave the previous session's transcript untouched: {after_dump}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_bare_prompt_while_busy_is_rejected_not_queued() {
    // pi: agent-session-concurrent.test.ts / agent-session-prompt.test.ts — a bare `prompt` (no
    // `streaming_behavior`) sent while one is already in flight must be rejected as busy, distinct
    // from the accepted case (`serve_busy_prompt_with_streaming_behavior_is_accepted_not_rejected`
    // above, which carries `streaming_behavior`).
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_busy",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);

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
        json!({ "type": "prompt", "id": "p2", "message": "also handle this" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let rejection = read_until_response(&mut stdout, "prompt");
    let p2 = rejection
        .iter()
        .find(|f| f["id"] == "p2")
        .expect("p2's own response frame");
    assert_eq!(p2["success"], false, "got: {p2:#?}");
    assert!(
        p2["error"].as_str().unwrap_or_default().contains("busy"),
        "got: {p2:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_unknown_command_type_reports_a_clear_failure_frame() {
    // pi: suite/regressions/5868-rpc-unknown-command-id.test.ts — an unrecognized `type` must still
    // produce a well-formed response frame (echoing both `id` and the unrecognized `command`), not a
    // dropped connection or a malformed/missing reply.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "foobar", "id": "test" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "foobar");
    let resp = frames.last().unwrap();
    assert_eq!(resp["id"], "test");
    assert_eq!(resp["command"], "foobar");
    assert_eq!(resp["success"], false, "got: {resp:#?}");
    assert!(resp.get("error").is_some(), "got: {resp:#?}");

    // The connection must still be alive afterward — a genuinely recognized command works normally.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let ok = read_until_response(&mut stdout, "prompt");
    assert_eq!(ok.last().unwrap()["success"], true);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_jsonl_framing_preserves_u2028_and_u2029_inside_a_payload() {
    // F-L1 (pi: rpc-jsonl.test.ts "splits on LF only and preserves U+2028/U+2029 inside payloads"):
    // U+2028 (LINE SEPARATOR) and U+2029 (PARAGRAPH SEPARATOR) are not ASCII `\n`, so a byte-oriented
    // NDJSON reader that only splits on `0x0A` must pass them straight through as ordinary payload
    // bytes rather than treating them as line breaks. `tokio::io::AsyncBufReadExt::lines()` (this
    // server's stdin reader — see `serve()`) only ever splits on `0x0A`/strips a trailing `\r`, so this
    // is expected to already be correct; proven here rather than assumed.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let title = "a\u{2028}b\u{2029}c";
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_session_name", "title": title })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_session_name");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    // No embedded `\r`/`\n`, so `sanitize_title` leaves it untouched — the separators round-trip
    // byte-for-byte, proving neither the client-side serializer nor this server's line reader treated
    // them as a line break mid-payload.
    assert_eq!(
        frames.last().unwrap()["data"]["title"],
        title,
        "got: {frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let state = read_until_response(&mut stdout, "get_state");
    assert_eq!(state.last().unwrap()["data"]["title"], title, "{state:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_jsonl_framing_handles_crlf_delimited_commands() {
    // F-L1 (pi: rpc-jsonl.test.ts "handles CRLF-delimited input"): a client on Windows, or one that
    // simply writes `\r\n`, must have each command recognized as its own line — `tokio::io::
    // AsyncBufReadExt::lines()` strips a trailing `\r` after splitting on `\n` (see `lines.rs`'s
    // `poll_next_line`), so two `\r\n`-terminated commands in the same write must parse as two clean
    // JSON lines, not one line with a stray `\r` corrupting the trailing `}` or the two commands
    // merging into one.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let cmd_a = json!({ "id": "a", "type": "set_session_name", "title": "first" }).to_string();
    let cmd_b = json!({ "id": "b", "type": "set_session_name", "title": "second" }).to_string();
    write!(stdin, "{cmd_a}\r\n{cmd_b}\r\n").unwrap();
    stdin.flush().unwrap();

    let frames_a = read_until_response(&mut stdout, "set_session_name");
    let resp_a = frames_a
        .iter()
        .find(|f| f["type"] == "response" && f["id"] == "a")
        .unwrap_or_else(|| panic!("expected a response to id \"a\": {frames_a:#?}"));
    assert_eq!(resp_a["success"], true, "{resp_a:#?}");
    assert_eq!(resp_a["data"]["title"], "first", "{resp_a:#?}");

    let frames_b = read_until_response(&mut stdout, "set_session_name");
    let resp_b = frames_b
        .iter()
        .find(|f| f["type"] == "response" && f["id"] == "b")
        .unwrap_or_else(|| panic!("expected a response to id \"b\": {frames_b:#?}"));
    assert_eq!(resp_b["success"], true, "{resp_b:#?}");
    assert_eq!(resp_b["data"]["title"], "second", "{resp_b:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_jsonl_framing_handles_a_final_command_with_no_trailing_newline() {
    // F-L1 (pi: rpc-jsonl.test.ts "emits a final line without trailing LF"): a client that closes
    // stdin right after its last command, with no trailing `\n`, must still have that command
    // processed — `AsyncBufReadExt::lines()` yields the trailing partial line once at EOF (`n == 0`
    // with a non-empty buffer still returns `Some(buf)`; only a truly empty read returns `None`), so
    // this server's `lines.next_line()` loop sees and processes it before observing stdin's close.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // No trailing `\n` at all — `write!`, not `writeln!` — then drop `stdin` to close the pipe.
    write!(
        stdin,
        "{}",
        json!({ "id": "last", "type": "set_session_name", "title": "no newline" })
    )
    .unwrap();
    stdin.flush().unwrap();
    drop(stdin);

    let frames = read_until_response(&mut stdout, "set_session_name");
    let resp = frames.last().unwrap();
    assert_eq!(resp["id"], "last", "{resp:#?}");
    assert_eq!(resp["success"], true, "{resp:#?}");
    assert_eq!(resp["data"]["title"], "no newline", "{resp:#?}");

    child.wait().unwrap();
}

#[test]
fn serve_get_session_stats_reports_context_usage_after_a_real_turn() {
    // Companion e2e proof for `session_stats`'s `context_usage` field (unit-tested directly in
    // `serve.rs`'s own test module) — end-to-end through the real RPC surface: null before any turn,
    // populated with a plausible `percent` after one.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi there")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_session_stats" })).unwrap();
    stdin.flush().unwrap();
    let before = read_until_response(&mut stdout, "get_session_stats");
    assert_eq!(
        before.last().unwrap()["data"]["context_usage"],
        Value::Null,
        "no turn has run yet: {before:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hello" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_session_stats" })).unwrap();
    stdin.flush().unwrap();
    let after = read_until_response(&mut stdout, "get_session_stats");
    let usage = &after.last().unwrap()["data"]["context_usage"];
    assert!(usage["tokens"].as_u64().unwrap() > 0, "got: {usage:#?}");
    assert!(
        usage["context_window"].as_u64().unwrap() > 0,
        "got: {usage:#?}"
    );
    assert!(usage["percent"].as_f64().unwrap() >= 0.0, "got: {usage:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_session_name_and_get_last_assistant_text() {
    // pi: rpc.test.ts "should set and get session name" / "get_last_assistant_text" — both were
    // implemented but had zero e2e coverage: `set_session_name` persists a `session_info` entry
    // reflected by `get_state`, and `get_last_assistant_text` reports the most recent assistant reply.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("the actual reply")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // No name set yet, no assistant reply yet.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let initial = read_until_response(&mut stdout, "get_state");
    assert!(
        initial.last().unwrap()["data"]
            .get("title")
            .is_none_or(Value::is_null),
        "got: {initial:#?}"
    );
    writeln!(stdin, "{}", json!({ "type": "get_last_assistant_text" })).unwrap();
    stdin.flush().unwrap();
    let none_yet = read_until_response(&mut stdout, "get_last_assistant_text");
    let text = &none_yet.last().unwrap()["data"]["text"];
    assert!(
        text.as_str().unwrap_or_default().is_empty() || text.is_null(),
        "got: {none_yet:#?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_session_name", "title": "my-test-session" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let set_resp = read_until_response(&mut stdout, "set_session_name");
    assert_eq!(
        set_resp.last().unwrap()["success"],
        true,
        "got: {set_resp:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let after_name = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        after_name.last().unwrap()["data"]["title"],
        "my-test-session",
        "got: {after_name:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_last_assistant_text" })).unwrap();
    stdin.flush().unwrap();
    let after_reply = read_until_response(&mut stdout, "get_last_assistant_text");
    assert!(
        after_reply.last().unwrap()["data"]["text"]
            .as_str()
            .unwrap_or_default()
            .contains("the actual reply"),
        "got: {after_reply:#?}"
    );

    // The name survives on disk too (a single `session_info` entry, not lost on the next reattach).
    let on_disk = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        on_disk.contains("my-test-session"),
        "session name must be persisted: {on_disk}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_session_name_strips_newlines_and_pushes_a_session_info_changed_frame() {
    // F-M4 (pi: 5996-session-name-newlines.test.ts:14-22): newline sanitization was only proven at the
    // storage-unit level (`session_store.rs`'s `set_title_strips_newlines`), never through the actual
    // `set_session_name` RPC command + `get_state` readback.
    //
    // F-M1 (pi: 3686-session-name-event.test.ts, `session_info_changed` — `rpc-mode.ts:632-639`): the
    // rename response previously carried no `data` at all, and nothing told a client the *sanitized*
    // final name without a follow-up `get_state`. Both are proven together here: the same round trip
    // exercises the sanitization and the new `data`/unsolicited-frame push.
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
        json!({ "id": "n1", "type": "set_session_name", "title": "hello\nworld\r\nagain" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_session_name");

    // The unsolicited `session_info_changed` push frame carries the sanitized name, correlated by id.
    let pushed = frames
        .iter()
        .find(|f| f["type"] == "session_info_changed")
        .unwrap_or_else(|| panic!("expected a session_info_changed frame: {frames:#?}"));
    assert_eq!(pushed["id"], "n1", "got: {pushed:#?}");
    assert_eq!(pushed["title"], "hello world again", "got: {pushed:#?}");

    // The response itself also carries the final sanitized name — no second round trip needed.
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "got: {resp:#?}");
    assert_eq!(resp["data"]["title"], "hello world again", "got: {resp:#?}");

    // And `get_state` reads back the same sanitized value, with no embedded newlines.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let state = read_until_response(&mut stdout, "get_state");
    let title = state.last().unwrap()["data"]["title"].as_str().unwrap();
    assert_eq!(title, "hello world again", "got: {state:#?}");
    assert!(
        !title.contains('\n') && !title.contains('\r'),
        "got: {title:?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_auto_retry_exhausts_all_attempts_and_reports_failure() {
    // Companion to `serve_auto_retries_a_whole_run_after_mid_stream_retry_is_exhausted` above, which
    // only proves attempt 1 recovers — this drives all `MAX_RUN_RETRIES` (3) whole-run attempts to
    // fail, ending in a reported failure with no recovery. Each whole-run attempt itself exhausts
    // agent-core's own mid-stream retry (1 initial + 3 retried, all truncated) before this layer even
    // sees it, so 3 whole-run attempts need 12 truncated stream chunks total, no successful turn at
    // the end. Slow (~15-20s of real backoff sleep) but this is the only way to observe the real
    // exponential-backoff schedule end to end.
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![truncated.to_string(); 12]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let auto_retry_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry"))
        .collect();
    assert_eq!(
        auto_retry_frames.len(),
        3,
        "expected exactly 3 auto_retry notices (attempts 1, 2, 3): {frames:#?}"
    );
    let attempts: Vec<i64> = auto_retry_frames
        .iter()
        .map(|f| f["attempt"].as_i64().unwrap())
        .collect();
    assert_eq!(attempts, vec![1, 2, 3]);

    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(auto_retry_end_frames.len(), 1, "got: {frames:#?}");
    assert_eq!(auto_retry_end_frames[0]["success"], false);
    assert_eq!(auto_retry_end_frames[0]["attempt"], 3);
    assert!(
        auto_retry_end_frames[0].get("final_error").is_some(),
        "an exhausted retry sequence must carry a final_error: {:?}",
        auto_retry_end_frames[0]
    );

    // The prompt command's own terminal response reports failure too — no silent success.
    let prompt_resp = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .unwrap();
    assert_eq!(prompt_resp["success"], false, "got: {prompt_resp:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn get_state_reports_session_file_and_is_streaming_idle_vs_mid_run() {
    // LOW pi-parity gap (fixed): `get_state` used to omit pi's `sessionFile`/`isStreaming`/
    // `isCompacting` fields entirely. This proves `session_file` matches the real `--session-file`
    // path in both the idle and mid-run handlers (architecturally distinct code paths — one keyed off
    // `live_stats`, the other off `session_stats`), and that `is_streaming` correctly flips true only
    // while a `prompt` genuinely has a turn in flight. `is_compacting` isn't forced here (doing so
    // reliably would need a real network delay the shared mock server doesn't support) — its own
    // event-driven state machine (`CompactionStart` before `Compacted`, with the right `reason`) is
    // proven directly at the `agent-core` level instead; this test still confirms it reads back
    // `false` in both idle and this (non-compacting) mid-run case, i.e. it never falsely reports
    // `true` for an ordinary run.
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_gs",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Idle, before any prompt: `is_streaming`/`is_compacting` must both already read `false`, and
    // `session_file` must already resolve to the real on-disk path (`Persistence::open` created it at
    // startup, before any turn ever ran).
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let idle_frames = read_until_response(&mut stdout, "get_state");
    let idle_state = &idle_frames.last().unwrap()["data"];
    assert_eq!(idle_state["is_streaming"], false, "got: {idle_state:#?}");
    assert_eq!(idle_state["is_compacting"], false, "got: {idle_state:#?}");
    assert_eq!(
        idle_state["session_file"].as_str(),
        Some(session_file.as_str()),
        "got: {idle_state:#?}"
    );

    // Mid-run: the sleeping tool call keeps a turn in flight long enough to query `get_state` from
    // the busy-loop's own (architecturally distinct) handler.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    let mid_run_state = &frames
        .iter()
        .find(|f| f["type"] == "response" && f["command"] == "get_state")
        .expect("a get_state response while the prompt is in flight")["data"];
    assert_eq!(
        mid_run_state["is_streaming"], true,
        "got: {mid_run_state:#?}"
    );
    assert_eq!(
        mid_run_state["is_compacting"], false,
        "got: {mid_run_state:#?}"
    );
    assert_eq!(
        mid_run_state["session_file"].as_str(),
        Some(session_file.as_str()),
        "got: {mid_run_state:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_name_flag_sets_the_initial_session_title() {
    // Companion to `run`'s own `--name` e2e coverage (`run_e2e.rs`) — `serve`'s version only applies
    // to a genuinely fresh session (see `ServeConfig::name`'s doc comment), which a brand-new
    // `--session-file` always is.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args(["--name", "my-serve-session"])
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let state = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        state.last().unwrap()["data"]["title"],
        "my-serve-session",
        "got: {state:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}
