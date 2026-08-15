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
use std::process::{Command, Stdio};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, read_until_response, spawn_model_server, turn_text,
};
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

/// A live `serve` child together with its stdio.
///
/// The `BufReader` is created **once** and reused for every command. A fresh one per command would be
/// a latent hang: `BufReader` reads ahead, so bytes of the *next* frame routinely land in its buffer,
/// and dropping it discards them — the following read then blocks forever on a response whose bytes
/// were already consumed. A wedged test never drops its [`ChildGuard`], which leaves a `serve` running
/// and (per `.github/workflows/ci.yml`) hangs the whole CI step rather than failing it.
struct Serve {
    child: ChildGuard,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Serve {
    fn start(cmd: &mut Command) -> Self {
        let mut child = cmd.spawn_guarded();
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
        }
    }

    /// Send one command, returning the `data` object of its `response` frame.
    fn command(&mut self, cmd: Value) -> Value {
        let name = cmd
            .get("type")
            .and_then(Value::as_str)
            .expect("command needs a type")
            .to_string();
        writeln!(self.stdin, "{cmd}").unwrap();
        self.stdin.flush().unwrap();
        read_until_response(&mut self.stdout, &name)
            .into_iter()
            .rfind(|f| f.get("type").and_then(Value::as_str) == Some("response"))
            .and_then(|f| f.get("data").cloned())
            .unwrap_or_else(|| panic!("no response data for {name}"))
    }

    /// `(session_id, message_count)` — the two facts every test here asserts on.
    fn identity(&mut self) -> (String, u64) {
        let data = self.command(json!({ "type": "get_state" }));
        (
            data["session_id"].as_str().expect("session_id").to_string(),
            data["message_count"].as_u64().expect("message_count"),
        )
    }

    /// Drive one prompt to completion, so the session has a real persisted turn behind it.
    fn prompt(&mut self, message: &str) {
        self.command(json!({ "type": "prompt", "message": message }));
    }

    /// Close stdin and reap, so the child is gone before the next one starts. `ChildGuard` would kill
    /// it anyway, but an orderly EOF-then-exit is what these tests are actually asserting persisted
    /// state after — and it leaves nothing behind to outlive the test.
    fn shutdown(self) {
        let Self {
            mut child,
            stdin,
            stdout,
        } = self;
        drop(stdin);
        drop(stdout);
        let _ = child.wait();
    }
}

#[test]
fn an_addressed_session_wins_over_an_unrelated_session_in_the_same_directory() {
    // The exact collapse this fixes: a directory that already holds a session for this cwd must not
    // capture a launch that named a *different* session.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let mut squatter = Serve::start(&mut serve_selecting(bin, &base, &session_dir, &[]));
    squatter.prompt("remember the marker: squatter-1");
    let (squatter_id, squatter_count) = squatter.identity();
    assert!(squatter_count > 0, "the squatter session recorded its turn");
    squatter.shutdown();

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let mut addressed = Serve::start(&mut serve_selecting(
        bin,
        &base2,
        &session_dir,
        &["--session-id", "tenant-a"],
    ));
    let (id, count) = addressed.identity();
    assert_eq!(id, "tenant-a", "the id asked for is the id opened");
    assert_ne!(id, squatter_id);
    assert_eq!(
        count, 0,
        "an addressed session must start empty, not inherit the cwd match's transcript"
    );
    addressed.shutdown();
}

#[test]
fn distinct_session_ids_in_one_directory_are_distinct_sessions() {
    // Multi-tenancy, end to end: N ids sharing one `--session-dir` are N conversations.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    for tenant in ["tenant-a", "tenant-b"] {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ack")]);
        let mut child = Serve::start(&mut serve_selecting(
            bin,
            &base,
            &session_dir,
            &["--session-id", tenant],
        ));
        let (id, count) = child.identity();
        assert_eq!(id, tenant);
        assert_eq!(count, 0, "{tenant} must not see the other tenant's history");
        child.prompt(&format!("marker for {tenant}"));
        child.shutdown();
    }

    // Each reopens to its own transcript — the point of an address being stable.
    for tenant in ["tenant-a", "tenant-b"] {
        let (base, _bodies) = spawn_model_server(vec![turn_text("ack")]);
        let mut child = Serve::start(&mut serve_selecting(
            bin,
            &base,
            &session_dir,
            &["--session-id", tenant],
        ));
        let (id, count) = child.identity();
        assert_eq!(id, tenant);
        assert_eq!(count, 2, "{tenant} reopens its own one-turn transcript");
        child.shutdown();
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
    let mut first = Serve::start(&mut serve_selecting(
        bin,
        &base,
        &session_dir,
        &["--session-id", "pinned"],
    ));
    first.prompt("remember the marker: pinned-42");
    first.shutdown();

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let mut restarted = Serve::start(&mut serve_selecting(
        bin,
        &base2,
        &session_dir,
        &["--session-id", "pinned"],
    ));
    let (id, count) = restarted.identity();
    assert_eq!(id, "pinned");
    assert_eq!(count, 2, "the restart picked the same conversation back up");
    restarted.shutdown();

    let sessions = jsonl_count(dir.path());
    assert_eq!(sessions, 1, "and did not stack up a second session file");
}

#[test]
fn a_bare_serve_starts_fresh_while_continue_reattaches() {
    // The default flip. A bare launch owns its own session (two servers in one directory must not
    // silently drive the same on-disk transcript); `--continue` is the one flag that reattaches.
    //
    // Deliberately asserts *which set* `--continue` lands in, not which member. `updated_at` is
    // second-granularity, so two sessions written inside the same second tie, and a tie is broken by
    // directory order — nondeterministic. "Most recent wins" is pinned separately, and unambiguously,
    // by `serve_resumes_newest_session_matching_cwd_not_globally_newest` (which distinguishes its
    // candidates by `cwd` rather than by time).
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().to_string_lossy().into_owned();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let (base, _bodies) = spawn_model_server(vec![turn_text("first answer")]);
    let mut first = Serve::start(&mut serve_selecting(bin, &base, &session_dir, &[]));
    first.prompt("remember the marker: bare-7");
    let (first_id, _) = first.identity();
    first.shutdown();

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let mut bare = Serve::start(&mut serve_selecting(bin, &base2, &session_dir, &[]));
    let (bare_id, bare_count) = bare.identity();
    assert_ne!(bare_id, first_id, "a bare launch starts its own session");
    assert_eq!(
        bare_count, 0,
        "and does not inherit the earlier session's transcript"
    );
    bare.shutdown();
    assert_eq!(
        jsonl_count(dir.path()),
        2,
        "two bare launches, two sessions"
    );

    let (base3, _bodies3) = spawn_model_server(vec![turn_text("third answer")]);
    let mut continued = Serve::start(&mut serve_selecting(
        bin,
        &base3,
        &session_dir,
        &["--continue"],
    ));
    let (continued_id, _) = continued.identity();
    assert!(
        continued_id == bare_id || continued_id == first_id,
        "--continue must reattach to a session already here, not mint one: {continued_id}"
    );
    continued.shutdown();
    assert_eq!(
        jsonl_count(dir.path()),
        2,
        "--continue reattached rather than creating a third session"
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
    let mut child = Serve::start(&mut serve_selecting(
        bin,
        &base,
        &session_dir,
        &["--session-id", "routed"],
    ));
    child.prompt("remember the marker: routed-99");
    let (_, before) = child.identity();
    assert_eq!(before, 2);

    let data = child.command(json!({ "type": "new_session" }));
    assert_eq!(
        data["session_id"].as_str(),
        Some("routed"),
        "the address a client routes on must survive new_session"
    );
    let (id, count) = child.identity();
    assert_eq!(id, "routed");
    assert_eq!(count, 0, "…while the conversation itself is blank");
    child.shutdown();

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
