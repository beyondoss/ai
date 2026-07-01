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

use common::{spawn_model_server, turn_text, turn_tool_use};
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

    // Fork the current session; the fork gets a new id.
    writeln!(stdin, "{}", json!({ "type": "fork" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "fork");
    let fork_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap();
    assert_ne!(fork_id, first_id, "a fork is a distinct session");

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

    // The active transcript is now just the first turn.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("first answer"));
    assert!(
        !dump.contains("second answer"),
        "the abandoned turn must not appear on the restored branch: {dump}"
    );

    // Two branches now exist: the abandoned one (inactive, still 4 deep) and the active one (2 deep).
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
    assert_eq!(active["message_count"], 2);

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
