//! The retry after a cut-off reply reaches disk, through `run`'s and `serve`'s real persist wiring.
//!
//! The agent's silent-truncation recovery: a reply stops on `max_tokens`, the context is compacted,
//! the cut-off reply is popped and the turn retried. The compaction's own checkpoint has already
//! written the cut-off reply by then, and the retry lands at the same message count — so a store
//! keyed on count wrote nothing more. A reopen got the cut-off reply back; a retry that called a tool
//! left a `tool_result` on disk with no `tool_use` before it.

// Test target: `.unwrap()`/`panic!` are how a test reports a failed setup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;

use agent_core::{ContentBlock, Message, Session};
use beyond_agent::session_store::{SessionMeta, SessionStore};
use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, sse, turn_text,
    turn_tool_use,
};
use serde_json::json;

/// A reply cut off by `max_tokens`.
fn turn_cut_off(text: &str) -> String {
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 150, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "max_tokens" }, "usage": { "output_tokens": 100 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// What the model is scripted to say: a cut-off reply, the compaction's summary, then the retry.
fn script(retry: Vec<String>) -> Vec<String> {
    let mut turns = vec![
        turn_cut_off("CUT-OFF partial answer"),
        turn_text("## Goal\nsummary of earlier work"),
    ];
    turns.extend(retry);
    turns
}

/// A session with some history for the compaction to fold.
fn seeded_session(dir: &Path) -> String {
    let path = dir.join("s.jsonl");
    let mut store =
        SessionStore::create(path.clone(), SessionMeta::new("/w", "claude-test")).unwrap();
    let mut s = Session::new();
    for i in 0..2 {
        s.push(Message::user(format!("earlier question {i}")));
        s.push(Message::assistant(vec![ContentBlock::text(format!(
            "earlier answer {i}"
        ))]));
    }
    store.append_new(&s.messages).unwrap();
    path.to_string_lossy().into_owned()
}

/// Compaction pinned small enough that a `max_tokens` stop is a hard overflow; session working memory
/// off, so its post-compaction steering turn doesn't add an unscripted model call.
const COMPACTION_ARGS: [&str; 7] = [
    "--context-window",
    "200",
    "--compaction-reserve-tokens",
    "50",
    "--compaction-keep-recent-tokens",
    "1",
    "--no-session-memory",
];

/// The reopened session's active path: it holds the retry and not the cut-off reply, and every
/// `tool_result` follows its `tool_use`.
fn assert_reopens_with_the_retry(path: &str, retried: &str) {
    let (_store, s) = SessionStore::open(path.into()).unwrap();
    let transcript = format!("{:?}", s.messages);
    assert!(
        !transcript.contains("CUT-OFF"),
        "the discarded reply is still on the active path: {transcript}"
    );
    assert!(
        transcript.contains(retried),
        "the retry never reached disk: {transcript}"
    );
    let mut calls = HashSet::new();
    for m in s.messages.iter() {
        for b in &m.content {
            match b {
                ContentBlock::ToolUse { id, .. } => {
                    calls.insert(id.clone());
                }
                ContentBlock::ToolResult { tool_use_id, .. } => assert!(
                    calls.contains(tool_use_id),
                    "a tool_result without its tool_use: {transcript}"
                ),
                _ => {}
            }
        }
    }
}

fn run_session(path: &str, base: &str) {
    let bin = env!("CARGO_BIN_EXE_agent");
    let out = run_cmd(bin)
        .args([
            "run",
            "continue please",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            path,
        ])
        .args(COMPACTION_ARGS)
        .current_dir(Path::new(path).parent().unwrap())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn serve_session(path: &str, base: &str) {
    let bin = env!("CARGO_BIN_EXE_agent");
    let mut child = serve_cmd(bin, base, path)
        .args(COMPACTION_ARGS)
        .current_dir(Path::new(path).parent().unwrap())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "continue please" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn run_persists_a_text_retry_after_a_cut_off_reply() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(script(vec![turn_text("REAL complete answer")]));
    run_session(&path, &base);
    assert_eq!(bodies.lock().unwrap().len(), 3, "cut-off, summary, retry");
    assert_reopens_with_the_retry(&path, "REAL complete answer");
}

#[test]
fn run_persists_a_tool_call_retry_after_a_cut_off_reply() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(script(vec![
        turn_tool_use("call_retry", "ls", r#"{"path":"."}"#),
        turn_text("REAL answer after the tool"),
    ]));
    run_session(&path, &base);
    assert_eq!(
        bodies.lock().unwrap().len(),
        4,
        "cut-off, summary, tool call, answer"
    );
    assert_reopens_with_the_retry(&path, "call_retry");
    assert_reopens_with_the_retry(&path, "REAL answer after the tool");
}

#[test]
fn serve_persists_a_text_retry_after_a_cut_off_reply() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(script(vec![turn_text("REAL complete answer")]));
    serve_session(&path, &base);
    assert_eq!(bodies.lock().unwrap().len(), 3, "cut-off, summary, retry");
    assert_reopens_with_the_retry(&path, "REAL complete answer");
}

#[test]
fn serve_persists_a_tool_call_retry_after_a_cut_off_reply() {
    let dir = tempfile::tempdir().unwrap();
    let path = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(script(vec![
        turn_tool_use("call_retry", "ls", r#"{"path":"."}"#),
        turn_text("REAL answer after the tool"),
    ]));
    serve_session(&path, &base);
    assert_eq!(
        bodies.lock().unwrap().len(),
        4,
        "cut-off, summary, tool call, answer"
    );
    assert_reopens_with_the_retry(&path, "call_retry");
    assert_reopens_with_the_retry(&path, "REAL answer after the tool");
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| match &m.content[0] {
            ContentBlock::Text { text, .. } => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

fn persist(store: &mut SessionStore, s: &Session) {
    store
        .persist(&s.messages, &s.compaction, &s.edited, &s.cuts)
        .unwrap();
}

/// The store side on its own: a cut the store already wrote past is rewound before the append,
/// however the count comes out.
#[test]
fn a_cut_followed_by_a_push_to_the_same_count_is_rewound_and_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    s.push(Message::user("q"));
    s.push(Message::assistant(vec![ContentBlock::text("discarded")]));
    persist(&mut store, &s);
    s.cut_back(1);
    s.push(Message::assistant(vec![ContentBlock::text("kept")]));
    persist(&mut store, &s);
    // Idempotent: the cut is applied once.
    persist(&mut store, &s);
    drop(store);
    let (_store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&reopened.messages), ["q", "kept"]);
}

/// A list shorter than what is on disk, with no compaction behind it, is a cut whether or not one
/// was recorded.
#[test]
fn an_unrecorded_shrink_is_rewound_too() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    s.push(Message::user("q"));
    s.push(Message::assistant(vec![ContentBlock::text("discarded")]));
    persist(&mut store, &s);
    std::sync::Arc::make_mut(&mut s.messages).pop();
    persist(&mut store, &s);
    s.push(Message::assistant(vec![ContentBlock::text("kept")]));
    persist(&mut store, &s);
    drop(store);
    let (_store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&reopened.messages), ["q", "kept"]);
}

/// In-memory differences that are not cuts are never written back: content alone is not a signal.
/// (A cross-model scrub rewrites stored messages in memory only; writing them back would break the
/// original model's replay.)
#[test]
fn a_changed_message_without_a_cut_is_not_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    let mut s = Session::new();
    s.push(Message::user("q"));
    s.push(Message::assistant(vec![ContentBlock::text("as stored")]));
    persist(&mut store, &s);
    std::sync::Arc::make_mut(&mut s.messages)[1] =
        Message::assistant(vec![ContentBlock::text("scrubbed in memory")]);
    s.push(Message::user("next"));
    persist(&mut store, &s);
    drop(store);
    let (_store, reopened) = SessionStore::open(path).unwrap();
    assert_eq!(texts(&reopened.messages), ["q", "as stored", "next"]);
}

/// A cut recorded before the store existed happened to a transcript it never wrote.
#[test]
fn a_cut_from_before_the_store_was_opened_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut s = Session::new();
    s.push(Message::user("q"));
    s.push(Message::assistant(vec![ContentBlock::text("a")]));
    s.push(Message::user("q2"));
    s.cut_back(2);
    let mut store = SessionStore::create(path.clone(), SessionMeta::new("/w", "m")).unwrap();
    persist(&mut store, &s);
    s.push(Message::user("q3"));
    persist(&mut store, &s);
    drop(store);
    let (_store, reopened) = SessionStore::open(path.clone()).unwrap();
    assert_eq!(texts(&reopened.messages), ["q", "a", "q3"]);
    assert!(
        !std::fs::read_to_string(path)
            .unwrap()
            .contains(r#""type":"leaf""#)
    );
}
