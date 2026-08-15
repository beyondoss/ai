//! `serve` e2e: which session a launch opens — `--session-id` addressing, `--continue`, and the fresh
//! default. The contract under test is one sentence: **a session id is an address.** It resolves to the
//! session it names or to a brand-new one under exactly that name, never to some other session that
//! merely shares the directory.
//!
//! The regression behind this file: `--session-id` was consulted only when *no* session yet matched the
//! current `cwd`, so a single pre-existing session in a directory silently swallowed every id pointed at
//! it — every tenant, task, or connection collapsed onto one shared conversation.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::process::{Child, Command, Stdio};

use common::{ISOLATED_HOME, SpawnGuarded, read_until_response, spawn_model_server, turn_text};
use serde_json::{Value, json};

/// A `serve` child bound to `--session-dir`, plus whatever extra selection flags a test wants
/// (`--session-id <id>`, `--continue`).
fn serve_selecting(bin: &str, base: &str, session_dir: &str, extra: &[&str]) -> Command {
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
    .args(extra)
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

/// One command against a live `serve` child, returning the `data` object of its `response` frame.
fn command(child: &mut Child, cmd: Value) -> Value {
    let name = cmd
        .get("type")
        .and_then(Value::as_str)
        .expect("command needs a type")
        .to_string();
    let mut stdin = child.stdin.as_ref().expect("stdin");
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
    let mut stdout = BufReader::new(child.stdout.as_mut().expect("stdout"));
    let frames = read_until_response(&mut stdout, &name);
    frames
        .into_iter()
        .rfind(|f| f.get("type").and_then(Value::as_str) == Some("response"))
        .and_then(|f| f.get("data").cloned())
        .unwrap_or_else(|| panic!("no response data for {name}"))
}

/// `(session_id, message_count)` — the two facts every test here asserts on.
fn identity(child: &mut Child) -> (String, u64) {
    let data = command(child, json!({ "type": "get_state" }));
    (
        data["session_id"].as_str().expect("session_id").to_string(),
        data["message_count"].as_u64().expect("message_count"),
    )
}

/// Drive one prompt to completion, so the session has a real persisted turn behind it.
fn prompt(child: &mut Child, message: &str) {
    command(child, json!({ "type": "prompt", "message": message }));
}

#[test]
fn an_addressed_session_wins_over_an_unrelated_session_in_the_same_directory() {
    // The exact collapse this fixes: a directory that already holds a session for this cwd must not
    // capture a launch that named a *different* session.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let mut squatter = serve_selecting(bin, &base, &session_dir, &[]).spawn_guarded();
    prompt(&mut squatter, "remember the marker: squatter-1");
    let (squatter_id, squatter_count) = identity(&mut squatter);
    assert!(squatter_count > 0, "the squatter session recorded its turn");
    drop(squatter);

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let mut addressed =
        serve_selecting(bin, &base2, &session_dir, &["--session-id", "tenant-a"]).spawn_guarded();
    let (id, count) = identity(&mut addressed);
    assert_eq!(id, "tenant-a", "the id asked for is the id opened");
    assert_ne!(id, squatter_id);
    assert_eq!(
        count, 0,
        "an addressed session must start empty, not inherit the cwd match's transcript"
    );
}

#[test]
fn distinct_session_ids_in_one_directory_are_distinct_sessions() {
    // Multi-tenancy, end to end: N ids sharing one `--session-dir` are N conversations.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    for tenant in ["tenant-a", "tenant-b"] {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ack")]);
        let mut child =
            serve_selecting(bin, &base, &session_dir, &["--session-id", tenant]).spawn_guarded();
        let (id, count) = identity(&mut child);
        assert_eq!(id, tenant);
        assert_eq!(count, 0, "{tenant} must not see the other tenant's history");
        prompt(&mut child, &format!("marker for {tenant}"));
    }

    // Each reopens to its own transcript — the point of an address being stable.
    for tenant in ["tenant-a", "tenant-b"] {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ack")]);
        let mut child =
            serve_selecting(bin, &base, &session_dir, &["--session-id", tenant]).spawn_guarded();
        let (id, count) = identity(&mut child);
        assert_eq!(id, tenant);
        assert_eq!(count, 2, "{tenant} reopens its own one-turn transcript");
    }
}

#[test]
fn an_addressed_session_is_idempotent_across_restarts() {
    // What a supervised (systemd, container) `serve --session-id` depends on: restart, same
    // conversation — deterministically, rather than via "whatever touched this directory most recently".
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let mut first =
        serve_selecting(bin, &base, &session_dir, &["--session-id", "pinned"]).spawn_guarded();
    prompt(&mut first, "remember the marker: pinned-42");
    drop(first);

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let mut restarted =
        serve_selecting(bin, &base2, &session_dir, &["--session-id", "pinned"]).spawn_guarded();
    let (id, count) = identity(&mut restarted);
    assert_eq!(id, "pinned");
    assert_eq!(count, 2, "the restart picked the same conversation back up");

    let sessions = jsonl_count(dir.path());
    assert_eq!(sessions, 1, "and did not stack up a second session file");
}

#[test]
fn a_bare_serve_starts_fresh_while_continue_reattaches() {
    // The default flip. A bare launch owns its own session (two servers in one directory must not
    // silently drive the same on-disk transcript); `--continue` is the one flag that reattaches.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let mut first = serve_selecting(bin, &base, &session_dir, &[]).spawn_guarded();
    prompt(&mut first, "remember the marker: bare-7");
    let (first_id, _) = identity(&mut first);
    drop(first);

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let mut bare = serve_selecting(bin, &base2, &session_dir, &[]).spawn_guarded();
    let (bare_id, bare_count) = identity(&mut bare);
    assert_ne!(bare_id, first_id, "a bare launch starts its own session");
    assert_eq!(bare_count, 0);
    drop(bare);

    let (base3, _bodies3) = spawn_model_server(vec![turn_text("third answer")]);
    let mut continued = serve_selecting(bin, &base3, &session_dir, &["--continue"]).spawn_guarded();
    let (continued_id, continued_count) = identity(&mut continued);
    assert_eq!(
        continued_id, bare_id,
        "--continue reattaches to the most recent session for this cwd"
    );
    assert_eq!(
        continued_count, 0,
        "which is the empty one the bare launch left"
    );
}

#[test]
fn new_session_on_an_addressed_session_keeps_the_id_and_archives_the_old_transcript() {
    // A pinned id is a routing key (`--session-id`, the daemon's `?session_id=`), so `new_session` must
    // not move it out from under whoever holds it. It blanks in place — and the outgoing conversation is
    // snapshotted into a session of its own rather than destroyed, so "start a new session" never
    // doubles as "throw the old one away".
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let mut child =
        serve_selecting(bin, &base, &session_dir, &["--session-id", "routed"]).spawn_guarded();
    prompt(&mut child, "remember the marker: routed-99");
    let (_, before) = identity(&mut child);
    assert_eq!(before, 2);

    let data = command(&mut child, json!({ "type": "new_session" }));
    assert_eq!(
        data["session_id"].as_str(),
        Some("routed"),
        "the address a client routes on must survive new_session"
    );
    let (id, count) = identity(&mut child);
    assert_eq!(id, "routed");
    assert_eq!(count, 0, "…while the conversation itself is blank");

    // The old transcript is still on disk as its own session, with lineage back to the slot.
    assert_eq!(
        jsonl_count(dir.path()),
        2,
        "the outgoing conversation was archived, not overwritten"
    );
    let archived = beyond_ai_agent::session_store::SessionRepo::open(dir.path())
        .unwrap()
        .list()
        .unwrap()
        .into_iter()
        .find(|m| m.id != "routed")
        .expect("an archived sibling session");
    assert_eq!(
        archived.parent.as_deref(),
        Some("routed"),
        "the archive records where it came from"
    );
    assert!(
        archived.message_count > 0,
        "and carries the real transcript"
    );
}

/// How many `.jsonl` sessions the repo directory holds.
fn jsonl_count(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .count()
}
