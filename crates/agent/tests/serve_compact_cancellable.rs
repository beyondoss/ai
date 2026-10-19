//! `serve` e2e: a manual `compact` keeps the session answering.
//!
//! The summary is a model call — up to the 600 s stream idle timeout per call — and the session loop
//! used to wait on it alone: `abort`, `get_state` and a SIGTERM were all seen only once it ended.
//! Now it runs inside a select: `abort` cancels it, read-only commands are answered at once, other
//! commands are refused as busy, and shutdown cancels it.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::time::{Duration, Instant};

use agent_core::{ContentBlock, Message, Session};
use beyond_ai_agent::session_store::{SessionMeta, SessionRepo, SessionStore};
use common::{NoFrame, SpawnGuarded, serve_cmd, spawn_model_server_with_stalled_response};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");
/// How long the summary call stalls: far longer than any assertion below waits.
const STALL: Duration = Duration::from_secs(20);

/// A session with enough history for `compact` to make a real summary call. Returns its file.
fn seeded_session(dir: &std::path::Path) -> String {
    let repo = SessionRepo::open(dir).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
    let mut seed = Session::new();
    seed.user("the task");
    for i in 0..12 {
        seed.push(Message::assistant(vec![ContentBlock::tool_use(
            format!("c{i}"),
            "read",
            json!({ "path": format!("src/f{i}.rs") }),
        )]));
        seed.push(Message::tool_result(
            format!("c{i}"),
            "x".repeat(4_000),
            false,
        ));
    }
    seed.push(Message::assistant(vec![ContentBlock::text(
        "read them all",
    )]));
    store.append_new(&seed.messages).unwrap();
    store.path().to_string_lossy().into_owned()
}

fn spawn(session_file: &str) -> common::ChildGuard {
    let base = spawn_model_server_with_stalled_response(vec![], STALL, vec![]);
    serve_cmd(BIN, &base, session_file)
        .args([
            "--compaction-keep-recent-tokens",
            "1",
            "--no-session-memory",
        ])
        .spawn_guarded()
}

/// Collect `response` frames until `n` have arrived or `within` passes.
fn responses(stdout: &mut common::Frames, n: usize, within: Duration) -> Vec<Value> {
    let deadline = Instant::now() + within;
    let mut seen = Vec::new();
    while seen.len() < n && Instant::now() < deadline {
        match stdout.next_frame(Duration::from_millis(200)) {
            Ok(f) if f["type"] == "response" => seen.push(f),
            Ok(_) | Err(NoFrame::TimedOut) => {}
            Err(NoFrame::Closed) => break,
        }
    }
    seen
}

fn send(stdin: &mut impl Write, v: Value) {
    writeln!(stdin, "{v}").unwrap();
    stdin.flush().unwrap();
}

/// Mid-compaction, `get_state` is answered at once (reporting `is_compacting`), another command is
/// refused as busy, and `abort` cancels the summary call: the compaction ends unapplied and nothing
/// on disk changes.
#[test]
fn abort_cancels_a_manual_compact_and_reads_are_answered_meanwhile() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let mut child = spawn(&session_file);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
    // Let the summary call reach the (stalling) model server.
    std::thread::sleep(Duration::from_millis(700));
    let t0 = Instant::now();
    send(&mut stdin, json!({ "type": "get_state", "id": "g1" }));
    send(
        &mut stdin,
        json!({ "type": "set_session_name", "id": "t1", "title": "x" }),
    );
    send(&mut stdin, json!({ "type": "abort", "id": "a1" }));
    let seen = responses(&mut stdout, 4, Duration::from_secs(10));
    let elapsed = t0.elapsed();
    let by_id = |id: &str| {
        seen.iter()
            .find(|f| f["id"] == id)
            .unwrap_or_else(|| panic!("no response to {id}: {seen:#?}"))
    };
    let state = by_id("g1");
    assert_eq!(state["success"], true, "{state}");
    assert_eq!(state["data"]["is_compacting"], true, "{state}");
    assert_eq!(state["data"]["message_count"], 26, "{state}");
    let busy = by_id("t1");
    assert_eq!(busy["success"], false, "{busy}");
    assert!(
        busy["error"].as_str().unwrap().starts_with("busy"),
        "{busy}"
    );
    assert_eq!(
        by_id("c1")["success"],
        false,
        "the compaction was cancelled"
    );
    assert_eq!(by_id("a1")["success"], true);
    let pos = |id: &str| seen.iter().position(|f| f["id"] == id).unwrap();
    assert!(
        pos("g1") < pos("c1"),
        "get_state waited for the compaction: {seen:#?}"
    );
    assert!(
        pos("c1") < pos("a1"),
        "abort is acknowledged once the compaction stopped"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "abort waited out the summary call: {elapsed:?}"
    );

    drop(stdin);
    child.wait().unwrap();
    let (_, session) = SessionStore::open(session_file.into()).unwrap();
    assert_eq!(
        session.messages.len(),
        26,
        "a cancelled compaction changes nothing on disk"
    );
    assert_eq!(session.compaction.compactions, 0);
}

/// A SIGTERM mid-compaction ends the process promptly instead of waiting out the summary call.
#[test]
fn sigterm_ends_a_manual_compact_promptly() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let mut child = spawn(&session_file);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    let _ = stdout.next_frame(Duration::from_secs(10)); // ready
    send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
    std::thread::sleep(Duration::from_millis(700));
    let t0 = Instant::now();
    let pid = child.id().to_string();
    assert!(
        std::process::Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .unwrap()
            .success()
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "never exited after SIGTERM");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        t0.elapsed() < Duration::from_secs(3),
        "SIGTERM waited out the compaction: {:?}",
        t0.elapsed()
    );
}

/// The reads the prompt busy loop answers without `&mut session` are answered mid-compaction too,
/// not refused as busy.
#[test]
fn the_prompt_loops_other_reads_are_answered_mid_compact() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let mut child = spawn(&session_file);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
    std::thread::sleep(Duration::from_millis(700));
    let reads = [
        "get_commands",
        "get_tree",
        "list_branches",
        "list_sessions",
        "list_all_sessions",
        "get_available_models",
    ];
    for read in reads {
        send(&mut stdin, json!({ "type": read, "id": read }));
    }
    let seen = responses(&mut stdout, reads.len(), Duration::from_secs(10));
    for read in reads {
        let reply = seen
            .iter()
            .find(|f| f["id"] == read)
            .unwrap_or_else(|| panic!("no response to {read}: {seen:#?}"));
        // `list_all_sessions` needs `--session-dir`, which this session lacks: a real answer, an
        // error, but not a busy refusal.
        let busy = reply["error"]
            .as_str()
            .is_some_and(|e| e.starts_with("busy"));
        assert!(!busy, "{read} was refused as busy: {reply}");
        if read != "list_all_sessions" {
            assert_eq!(reply["success"], true, "{read}: {reply}");
        }
    }
    send(&mut stdin, json!({ "type": "abort", "id": "a1" }));
    responses(&mut stdout, 2, Duration::from_secs(10));
}
