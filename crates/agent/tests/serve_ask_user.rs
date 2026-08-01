//! `serve` e2e: an agent that stops to ask, and the answer that resumes it.
//!
//! The unit tests in `tools::ask_user` prove the tool validates its input and
//! sets a flag. They cannot prove the thing that matters, which is a claim about
//! the *loop*: that a model calling `ask_user` actually ends its turn, that the
//! question reaches a client without any channel of its own, and that the next
//! prompt continues the same session rather than starting over.
//!
//! These drive the real binary over the real protocol, with a scripted model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::process::ChildStdin;

use common::{
    ChildGuard, SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn kill(mut child: ChildGuard) {
    let _ = child.kill();
    let _ = child.wait();
}

/// A model turn that calls `ask_user` with a question and two options.
fn asking_turn() -> String {
    turn_tool_use(
        "t1",
        "ask_user",
        &json!({
            "question": "Should the retry back off exponentially or linearly?",
            "options": ["Exponential", "Linear"],
        })
        .to_string(),
    )
}

/// The `tool_start` event for `name`, if the run emitted one.
fn tool_start<'a>(frames: &'a [Value], name: &str) -> Option<&'a Value> {
    frames.iter().find(|f| {
        f["type"] == "event"
            && f["event"]["kind"] == "tool_start"
            && f["event"]["name"] == name
    })
}

/// The whole point: a client learns about the question through the event stream
/// it is already reading, with no second channel and nothing to subscribe to.
#[test]
fn the_question_reaches_a_client_on_the_ordinary_event_stream() {
    let (base, _reqs) = spawn_model_server(vec![asking_turn()]);
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("s.jsonl");

    let mut child = serve_cmd(BIN, &base, session.to_str().unwrap()).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "make the uploads retry" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");

    let start = tool_start(&frames, "ask_user")
        .unwrap_or_else(|| panic!("no ask_user tool_start in {frames:#?}"));
    assert_eq!(
        start["event"]["input"]["question"],
        "Should the retry back off exponentially or linearly?",
        "the question rides in the tool call's own arguments"
    );
    assert_eq!(start["event"]["input"]["options"][0], "Exponential");
    assert_eq!(start["event"]["input"]["options"][1], "Linear");

    kill(child);
}

/// Asking ends the turn. The model server is scripted with exactly one response,
/// so a loop that carried on would hang on a second request that never comes —
/// this passing at all is part of the assertion.
#[test]
fn asking_ends_the_turn_rather_than_carrying_on() {
    let (base, reqs) = spawn_model_server(vec![asking_turn()]);
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("s.jsonl");

    let mut child = serve_cmd(BIN, &base, session.to_str().unwrap()).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "make the uploads retry" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");

    let response = frames.last().unwrap();
    assert_eq!(response["type"], "response");
    assert_eq!(response["command"], "prompt");
    assert_eq!(
        response["success"], true,
        "asking is a normal end to a turn, not a failure: {response}"
    );

    // One model round trip. A tool that did not terminate would have sent the
    // result back for another.
    assert_eq!(
        reqs.lock().unwrap().len(),
        1,
        "the turn ended at the question instead of asking the model again"
    );

    kill(child);
}

/// The resume, which is why nothing needed a pause state: the session is idle,
/// not gone, and the answer is simply the next prompt.
#[test]
fn the_answer_continues_the_same_session() {
    let (base, reqs) = spawn_model_server(vec![
        asking_turn(),
        turn_text("Using uploads-api, with exponential backoff."),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("s.jsonl");

    let mut child = serve_cmd(BIN, &base, session.to_str().unwrap()).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "make the uploads retry" }),
    );
    let asked = read_until_response(&mut stdout, "prompt");
    assert!(tool_start(&asked, "ask_user").is_some());

    // The answer. An ordinary prompt — no special command, no correlation id
    // tying it to the question.
    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p2", "message": "Exponential" }),
    );
    let answered = read_until_response(&mut stdout, "prompt");
    let response = answered.last().unwrap();
    assert_eq!(response["success"], true);

    // The second turn saw the first one's transcript, question included, rather
    // than starting from nothing.
    let second = &reqs.lock().unwrap()[1];
    assert!(
        second.contains("Should the retry back off exponentially or linearly?"),
        "the answer arrived without the question it answers"
    );
    assert!(
        second.contains("Exponential"),
        "the answer itself did not reach the model"
    );

    kill(child);
}

/// The tool is offered to the model at all — asserted against the request the
/// model actually received, which is stronger than any listing command: a tool
/// absent from the wire cannot be called no matter what a registry says.
#[test]
fn the_model_is_shown_ask_user() {
    let (base, reqs) = spawn_model_server(vec![turn_text("hello")]);
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("s.jsonl");

    let mut child = serve_cmd(BIN, &base, session.to_str().unwrap()).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(&mut stdin, json!({ "type": "prompt", "id": "p1", "message": "hello" }));
    read_until_response(&mut stdout, "prompt");

    let sent = &reqs.lock().unwrap()[0];
    assert!(
        sent.contains("ask_user"),
        "the model was never offered ask_user"
    );
    assert!(
        sent.contains("end your turn to wait for their answer"),
        "the description has to tell the model what calling it does: {sent}"
    );

    kill(child);
}
