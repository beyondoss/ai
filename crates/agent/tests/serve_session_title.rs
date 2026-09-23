//! `serve` e2e: a session names itself, once.
//!
//! The title is the highest-priority field session search ranks on, and until this existed it was
//! `None` for every session nobody had explicitly named — so the field ranked first was empty almost
//! always. It is generated from the opening exchange by one extra model call, which makes the two
//! properties worth pinning here the ones about *cost*: it happens at most once per session, and it
//! never takes the session down with it when it fails.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};

use common::{
    SpawnGuarded, read_until_response, serve_dir_cmd, spawn_model_server,
    spawn_model_server_routed, turn_text,
};
use serde_json::{Value, json};

/// The one session in the listing.
fn only_session(frames: &[Value]) -> Value {
    frames.last().unwrap()["data"]["sessions"]
        .as_array()
        .unwrap()[0]
        .clone()
}

#[test]
fn a_session_names_itself_from_its_opening_exchange() {
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_string_lossy().into_owned();
    // Two turns: the prompt, then the title call it triggers.
    let (base, _bodies) = spawn_model_server(vec![
        turn_text("here is the answer"),
        turn_text("Fixing the parser"),
    ]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &dir_str).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "the parser drops the last token" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let session = only_session(&read_until_response(&mut stdout, "list_sessions"));
    assert_eq!(
        session["title"], "Fixing the parser",
        "the session must carry the generated title: {session:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn an_unusable_title_is_attempted_once_and_costs_the_session_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_string_lossy().into_owned();
    // Every call gets prose far past the title cap, so `clean_title` rejects it every time — the
    // "model answered the question instead of naming it" case. Unbounded, so nothing here fails for
    // want of a scripted turn.
    let (base, bodies) = spawn_model_server_routed(Vec::new(), turn_text(&"x".repeat(200)));
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &dir_str).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["first", "second"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        let frames = read_until_response(&mut stdout, "prompt");
        assert_eq!(
            frames.last().unwrap()["success"],
            true,
            "a title that cannot be generated must not fail the turn: {frames:#?}"
        );
    }

    writeln!(stdin, "{}", json!({ "type": "list_sessions" })).unwrap();
    stdin.flush().unwrap();
    let session = only_session(&read_until_response(&mut stdout, "list_sessions"));
    assert!(
        session["title"].is_null(),
        "an unusable reply must leave the session untitled, not titled with prose: {session:#?}"
    );

    // Two prompts and **one** title attempt. `title.is_none()` cannot express "already tried" on its
    // own, so without the once-guard a model that reliably declines would charge an extra call on
    // every single turn, forever.
    let recorded = bodies.lock().unwrap().len();
    assert_eq!(
        recorded, 3,
        "expected two prompts and exactly one title attempt, got {recorded} model requests"
    );

    drop(stdin);
    child.wait().unwrap();
}
