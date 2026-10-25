//! `serve` e2e: a compaction's `<background-jobs>` block describes processes, so it lives only as
//! long as they can. It used to be written into the persisted summary, and a restarted process (whose
//! jobs died with the old one) told the model on its first request that a dead job was running. Now
//! the restarted process's first request says no jobs are running, which also corrects a summary
//! whose own prose claims one is.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::ChildStdin;

use common::{
    SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

#[test]
fn a_restarted_session_is_not_told_dead_jobs_are_running() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let start = json!({ "command": "echo up; sleep 90", "run_in_background": true });
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &start.to_string()),
        turn_text("the server is up"),
        // The summarization call `compact` makes.
        turn_text("## Goal\nStarted a server."),
    ]);
    let mut child = serve_cmd(BIN, &base, &session_file)
        .args(["--compaction-keep-recent-tokens", "1"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "start the server" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
    let frames = read_until_response(&mut stdout, "compact");
    assert_eq!(
        frames.last().unwrap()["data"]["compacted"],
        true,
        "{frames:?}"
    );
    // The session's end stops the job.
    drop(stdin);
    let _ = child.wait();

    // A fresh process on the same session: no jobs exist, and none is described.
    let (base, bodies) = spawn_model_server(vec![turn_text("hello again")]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "is the server still up?" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let body = bodies.lock().unwrap()[0].clone();
    assert!(body.contains("Started a server."), "the summary survives");
    assert!(
        !body.contains("- job 1:"),
        "a dead job is described: {body}"
    );
    // The summary's prose may still say the server is up; the first request says it is not.
    assert!(
        body.contains(&format!(
            "{}\\n{}\\n{}",
            agent_core::background::CARRY_OPEN,
            agent_core::background::NO_JOBS,
            agent_core::background::CARRY_CLOSE
        )),
        "the reopened summary is told no jobs are running: {body}"
    );
    drop(stdin);
    let _ = child.wait();
}
