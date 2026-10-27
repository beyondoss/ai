//! One writer per single-file session. `run` and `serve` (outside service mode) take the session's
//! writer lock right after opening it and refuse a session another process holds — two writers each
//! chain their next turn onto their own stale tip, and the transcript interleaves into nonsense.
//!
//! The "other process" here is this test process holding the lock: it is an open file description
//! lock, so a child sees it exactly as it would another `run`.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;

use agent_core::{ContentBlock, Message, Session};
use beyond_agent::session_store::{SessionMeta, SessionStore, lock_session_writer};
use common::{SpawnGuarded, run_cmd, serve_cmd, spawn_model_server, turn_text};

/// A two-message session at `path`, ending in a torn line — what a writer mid-append looks like from
/// outside, and so what a refused process must leave exactly as it is.
fn session_with_a_torn_tail(path: &Path) {
    let mut store = SessionStore::create(path.to_path_buf(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    s.push(Message::user("u0"));
    s.push(Message::assistant(vec![ContentBlock::text("a0")]));
    store.append_new(&s.messages).unwrap();
    let mut f = fs::OpenOptions::new().append(true).open(path).unwrap();
    write!(
        f,
        r#"{{"type":"message","id":"in-flight","role":"user","cont"#
    )
    .unwrap();
}

fn run_on(path: &Path, base: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_agent");
    run_cmd(bin)
        .args([
            "run",
            "hello",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            path.to_str().unwrap(),
        ])
        .current_dir(path.parent().unwrap())
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[test]
fn run_refuses_a_session_another_process_is_writing_and_leaves_it_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    session_with_a_torn_tail(&path);
    let before = fs::read(&path).unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("answer")]);

    let held = lock_session_writer(&path).unwrap();
    let out = run_on(&path, &base);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a second writer ran: {stderr}");
    assert!(
        stderr.contains("open in another agent process") && stderr.contains("s.jsonl"),
        "{stderr}"
    );
    assert!(
        bodies.lock().unwrap().is_empty(),
        "the refused run called the model"
    );
    // Not even the torn tail — the holder's append in flight — was touched.
    assert_eq!(fs::read(&path).unwrap(), before);

    // Once the holder lets go, the same run goes through and its turn lands on a clean line.
    drop(held);
    let out = run_on(&path, &base);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (_store, s) = SessionStore::open(path).unwrap();
    let texts: Vec<String> = s
        .messages
        .iter()
        .filter_map(|m| match &m.content[0] {
            ContentBlock::Text { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["u0", "a0", "hello", "answer"]);
}

#[test]
fn serve_refuses_a_session_file_another_process_is_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    session_with_a_torn_tail(&path);
    let before = fs::read(&path).unwrap();
    let (base, _bodies) = spawn_model_server(vec![]);

    let _held = lock_session_writer(&path).unwrap();
    let bin = env!("CARGO_BIN_EXE_agent");
    let child = serve_cmd(bin, &base, path.to_str().unwrap())
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a second writer started: {stderr}");
    assert!(stderr.contains("open in another agent process"), "{stderr}");
    assert_eq!(fs::read(&path).unwrap(), before);
}
