//! `serve` e2e: `prompt {output_schema}` — the agent as a callable function over the RPC protocol.
//!
//! The schema is per-prompt, not per-session: one session can answer in prose and the next as typed
//! JSON. That makes the interesting cases the transitions — install, repeat, change, remove — and the
//! guarantee that one prompt's answer can never be reported as another's.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin};

use common::{
    advertised_tools, read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": { "verdict": { "type": "string" } },
        "required": ["verdict"],
        "additionalProperties": false
    })
}

fn so_turn(id: &str, verdict: &str) -> String {
    turn_tool_use(
        id,
        "structured_output",
        &json!({ "verdict": verdict }).to_string(),
    )
}

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

/// Send a `prompt` and return its terminal response frame.
fn prompt(stdin: &mut ChildStdin, stdout: &mut impl BufRead, cmd: Value) -> Value {
    send(stdin, cmd);
    read_until_response(stdout, "prompt")
        .last()
        .unwrap()
        .clone()
}

fn kill(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn a_prompt_with_a_schema_returns_the_validated_payload_on_its_terminal_response() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![so_turn("tu_1", "approved")]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let resp = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "review it", "output_schema": schema() }),
    );
    assert_eq!(resp["success"], true, "{resp}");
    assert_eq!(resp["data"]["structured_output"]["verdict"], "approved");

    // One model call: the tool terminated the run, no wrap-up turn.
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    assert!(
        advertised_tools(&bodies[0])
            .iter()
            .any(|t| t == "structured_output")
    );
    assert!(bodies[0].contains("structured_output_protocol"));
    kill(child);
}

#[test]
fn a_prompt_without_a_schema_carries_neither_the_tool_nor_the_field() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![turn_text("a prose answer")]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let resp = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "just answer" }),
    );
    assert_eq!(resp["success"], true);
    assert!(
        resp["data"].get("structured_output").is_none(),
        "the field must be absent when no schema was asked for, so a client can tell it apart from \
         a null payload: {resp}"
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        !advertised_tools(&bodies[0])
            .iter()
            .any(|t| t == "structured_output")
    );
    assert!(!bodies[0].contains("structured_output_protocol"));
    kill(child);
}

#[test]
fn a_run_that_never_calls_the_tool_reports_a_null_payload_rather_than_omitting_the_field() {
    // `null` means "you asked for typed output and the model didn't produce it"; an absent field means
    // "you never asked". Collapsing the two would hide the failure.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("prose instead")]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let resp = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "review it", "output_schema": schema() }),
    );
    assert_eq!(resp["success"], true, "the run itself completed");
    assert_eq!(resp["data"]["structured_output"], Value::Null, "{resp}");
    kill(child);
}

#[test]
fn a_malformed_schema_is_rejected_before_the_prompt_is_ever_acknowledged() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Not an object schema: a tool's arguments are always an object, so this can never be satisfied.
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "x", "output_schema": { "type": "array" } }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], false);
    assert!(
        !frames.iter().any(|f| f["type"] == "ack"),
        "a prompt rejected for a bad schema must never be acknowledged: {frames:?}"
    );

    // Not a JSON object at all.
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "x", "output_schema": "garbage" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], false);
    assert!(
        resp["error"]
            .as_str()
            .unwrap()
            .contains("JSON Schema object")
    );

    assert!(
        bodies.lock().unwrap().is_empty(),
        "no model call may be billed for an unusable schema"
    );
    kill(child);
}

#[test]
fn a_schema_installed_by_one_prompt_does_not_leak_into_the_next() {
    // Per-prompt, not per-session. The second prompt omits `output_schema`, which must *remove* the tool
    // and its prompt section — not silently keep answering as a function.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        so_turn("tu_1", "approved"),
        turn_text("now just prose"),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let first = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "review it", "output_schema": schema() }),
    );
    assert_eq!(first["data"]["structured_output"]["verdict"], "approved");

    let second = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "and now explain" }),
    );
    assert_eq!(second["success"], true);
    assert!(
        second["data"].get("structured_output").is_none(),
        "the schema must not persist into a prompt that didn't ask for one: {second}"
    );

    let bodies = bodies.lock().unwrap();
    assert!(
        advertised_tools(&bodies[0])
            .iter()
            .any(|t| t == "structured_output")
    );
    assert!(
        !advertised_tools(&bodies[1])
            .iter()
            .any(|t| t == "structured_output"),
        "the tool must be gone from the second turn's advertised set"
    );
    assert!(
        !bodies[1].contains("structured_output_protocol"),
        "and so must its prompt section"
    );
    kill(child);
}

#[test]
fn a_second_prompt_reusing_the_same_schema_still_reports_only_its_own_answer() {
    // The tool (and its compiled validator) is reused across prompts that repeat a schema, so the slot
    // must be cleared between runs. Otherwise a prompt the model answered in prose would inherit the
    // previous prompt's payload — the worst possible failure, since it looks like success.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![
        so_turn("tu_1", "first"),
        turn_text("this time I refuse to use the tool"),
        so_turn("tu_3", "third"),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let ask = json!({ "type": "prompt", "message": "review", "output_schema": schema() });

    let first = prompt(&mut stdin, &mut stdout, ask.clone());
    assert_eq!(first["data"]["structured_output"]["verdict"], "first");

    let second = prompt(&mut stdin, &mut stdout, ask.clone());
    assert_eq!(
        second["data"]["structured_output"],
        Value::Null,
        "a stale payload must not be reported as this prompt's answer: {second}"
    );

    let third = prompt(&mut stdin, &mut stdout, ask);
    assert_eq!(third["data"]["structured_output"]["verdict"], "third");
    kill(child);
}

#[test]
fn changing_the_schema_between_prompts_rebuilds_the_tool() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let other = json!({
        "type": "object",
        "properties": { "score": { "type": "integer" } },
        "required": ["score"],
        "additionalProperties": false
    });
    let (base, bodies) = spawn_model_server(vec![
        so_turn("tu_1", "approved"),
        turn_tool_use(
            "tu_2",
            "structured_output",
            &json!({ "score": 7 }).to_string(),
        ),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let first = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "review", "output_schema": schema() }),
    );
    assert_eq!(first["data"]["structured_output"]["verdict"], "approved");

    let second = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "score it", "output_schema": other }),
    );
    assert_eq!(second["data"]["structured_output"]["score"], 7, "{second}");

    // The advertised `input_schema` *is* the caller's schema, so the tool definition changes with it.
    let bodies = bodies.lock().unwrap();
    let tools_of = |raw: &str| common::body_json(raw)["tools"].clone();
    assert!(tools_of(&bodies[0]).to_string().contains("verdict"));
    assert!(!tools_of(&bodies[0]).to_string().contains("score"));
    assert!(tools_of(&bodies[1]).to_string().contains("score"));
    assert!(!tools_of(&bodies[1]).to_string().contains("verdict"));
    kill(child);
}

#[test]
fn a_schema_violation_is_fed_back_over_the_wire_and_the_model_corrects_itself() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "tu_1",
            "structured_output",
            &json!({ "wrong": true }).to_string(),
        ),
        so_turn("tu_2", "corrected"),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let resp = prompt(
        &mut stdin,
        &mut stdout,
        json!({ "type": "prompt", "message": "review", "output_schema": schema() }),
    );
    assert_eq!(resp["success"], true);
    assert_eq!(resp["data"]["structured_output"]["verdict"], "corrected");

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "the run continued so the model could fix its call"
    );
    assert!(
        bodies[1].contains("verdict"),
        "the model must be told which field was missing: {}",
        bodies[1]
    );
    kill(child);
}
