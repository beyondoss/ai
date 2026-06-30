//! End-to-end: the real `beyond-ai-agent serve` binary over its stdio control protocol.
//!
//! Drives the headless server exactly as a remote client (or an SSH pipe) would: writes JSON command
//! lines to stdin, reads JSON frames from stdout. Proves (a) a `prompt` streams `event` frames for a
//! tool round-trip then a success `response`, (b) `get_messages` returns the transcript, and (c) a
//! fresh `serve` process **reattaches** to the persisted session and sees the prior transcript.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
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
