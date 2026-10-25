//! `serve` e2e: a host `bash` command or a manual `compact` keeps its daemon session busy.
//!
//! The idle reaper and the drain both read a session's `running` flag. A `prompt` sets it; these two
//! commands used to leave it clear, so a client that started a long host `bash` (or a `compact`) and
//! detached — or an HTTP POST that hit its 30 s reply timeout — had the session reaped or drained out
//! from under the command, which was then cancelled and lost.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use agent_core::{ContentBlock, Message, Session};
use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};
use common::{
    ChildGuard, ISOLATED_HOME, spawn_listening, spawn_model_server,
    spawn_model_server_with_stalled_response, turn_text, ws_connect, ws_next_frame,
    ws_read_until_response, ws_send,
};
use serde_json::json;

fn daemon(base: &str, dir: &str, idle_secs: u64, extra: &[&str]) -> (ChildGuard, u16) {
    spawn_listening(
        crate::common::agent_cmd()
            .args([
                "serve",
                "--gateway-url",
                base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                dir,
                "--session-idle-timeout",
                &idle_secs.to_string(),
            ])
            .args(extra)
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )
}

/// Wait for `path` to exist, up to `secs`.
fn wait_for(path: &std::path::Path, secs: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while !path.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    path.exists()
}

/// Send a host `bash` and wait for its `tool_start`, so the command is provably running.
async fn start_host_bash(ws: &mut common::TestWs, command: String) {
    ws_send(ws, json!({ "type": "bash", "id": "b", "command": command })).await;
    loop {
        let f = ws_next_frame(ws).await.expect("a frame before tool_start");
        if f["event"]["kind"] == "tool_start" {
            return;
        }
    }
}

/// A detached client's host `bash` outlives the idle timeout: the reaper leaves the session alone
/// until the command finishes, and the transcript records it as completed, not cancelled.
#[tokio::test]
async fn reaper_spares_a_detached_host_bash() {
    let tmp = tempfile::tempdir().unwrap();
    let done = tmp.path().join("done");
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let (mut child, port) = daemon(&base, dir.path().to_str().unwrap(), 1, &[]);
    {
        let mut ws = ws_connect(port, Some("hostbashreap01")).await;
        // Well past the 1 s timeout plus a reaper tick.
        start_host_bash(&mut ws, format!("sleep 3; echo ok > {}", done.display())).await;
    }
    assert!(
        wait_for(&done, 15),
        "the reaper cancelled a detached host bash"
    );

    // The file lands a moment before the command's result is recorded; until then the session
    // answers "busy".
    let mut ws = ws_connect(port, Some("hostbashreap01")).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    let messages = loop {
        ws_send(&mut ws, json!({ "type": "get_messages", "id": "m" })).await;
        let frames = ws_read_until_response(&mut ws, "get_messages").await;
        let last = frames.last().unwrap();
        if last["success"] == true {
            break last["data"]["messages"].to_string();
        }
        assert!(
            Instant::now() < deadline,
            "get_messages never succeeded: {last}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert!(
        messages.contains("sleep 3") && messages.contains(r#"\"cancelled\":false"#),
        "the host bash is recorded as completed: {messages}"
    );
    let _ = child.kill();
    let _ = child.wait();
}

/// A drain (SIGTERM with `--drain-grace`) waits for a detached client's host `bash` the way it
/// waits for a prompt, instead of stopping the session at once and cancelling the command.
#[tokio::test]
async fn drain_waits_for_a_host_bash() {
    const SID: &str = "hostbashdrain01";
    let tmp = tempfile::tempdir().unwrap();
    let done = tmp.path().join("done");
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let (mut child, port) = daemon(
        &base,
        dir.path().to_str().unwrap(),
        0,
        &["--drain-grace", "20"],
    );
    {
        let mut ws = ws_connect(port, Some(SID)).await;
        start_host_bash(&mut ws, format!("sleep 2; echo ok > {}", done.display())).await;
    }
    // Let the detach land, so stopping the session would close its input.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let pid = child.id().to_string();
    assert!(
        Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .unwrap()
            .success()
    );
    let _ = child.wait();
    assert!(
        done.exists(),
        "the drain cut the host bash despite its grace"
    );
    let (_, session) = SessionRepo::open(dir.path()).unwrap().open_id(SID).unwrap();
    let recorded = serde_json::to_string(&session.messages).unwrap();
    assert!(
        recorded.contains(r#"\"cancelled\":false"#),
        "the host bash is persisted as completed: {recorded}"
    );
}

/// A session with enough history for `compact` to make a real summary call.
fn seed(dir: &std::path::Path, id: &str) {
    let repo = SessionRepo::open(dir).unwrap();
    let mut store = repo
        .create(SessionMeta::with_id(id, "/w", "claude-test"))
        .unwrap();
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
}

/// A detached client's `compact` outlives the idle timeout: the summary call (stalled past it)
/// completes and the compaction lands on disk. A compaction cancels on EOF, so a reap — which closes
/// the session's input — would cut it short without the busy flag.
#[tokio::test]
async fn reaper_spares_a_detached_compact() {
    const SID: &str = "compactreap01";
    let dir = tempfile::tempdir().unwrap();
    seed(dir.path(), SID);
    // The first model call — the summary — stalls 3 s, well past the 1 s idle timeout.
    let base = spawn_model_server_with_stalled_response(
        vec![],
        Duration::from_secs(3),
        vec![turn_text("spare")],
    );
    let (mut child, port) = daemon(
        &base,
        dir.path().to_str().unwrap(),
        1,
        &[
            "--compaction-keep-recent-tokens",
            "1",
            "--no-session-memory",
        ],
    );
    {
        let mut ws = ws_connect(port, Some(SID)).await;
        ws_send(&mut ws, json!({ "type": "compact", "id": "c" })).await;
        // Let the summary call start before detaching.
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    tokio::time::sleep(Duration::from_secs(5)).await;

    let _ = child.kill();
    let _ = child.wait();
    let (_, session) = SessionRepo::open(dir.path()).unwrap().open_id(SID).unwrap();
    assert!(
        session.messages.len() < 26,
        "the reaper cancelled a detached compact: {} messages still on disk",
        session.messages.len()
    );
}
