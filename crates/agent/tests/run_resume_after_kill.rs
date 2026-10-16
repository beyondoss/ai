//! `run` e2e: resuming a session whose last run was killed outright. The prompt is checkpointed
//! before the model call, so after a `kill -9` the session's tip is that unanswered `user` message,
//! and `--continue` pushed a second one: `[user, user]`, a shape no dialect accepts (or, where
//! tolerated, two prompts merged into one turn). The resumed run now closes the unanswered prompt with
//! an "interrupted" error record first, as a failed run closes its own.
#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, run_cmd, spawn_model_server, spawn_model_server_with_stalled_response, turn_text,
};

fn prompt_on_disk(dir: &std::path::Path, needle: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|e| {
        let p = e.path();
        if p.is_dir() {
            prompt_on_disk(&p, needle)
        } else {
            std::fs::read_to_string(&p).is_ok_and(|s| s.contains(needle))
        }
    })
}

#[test]
fn continue_after_a_hard_kill_never_sends_two_user_turns() {
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions");
    let stalled = spawn_model_server_with_stalled_response(vec![], Duration::from_secs(30), vec![]);
    let mut child = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "FIRST-PROMPT",
            "--gateway-url",
            &stalled,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
        ])
        .arg(&sessions)
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded();
    // Killed while the model call is in flight: the prompt is on disk, its answer never will be.
    let t0 = Instant::now();
    while !prompt_on_disk(&sessions, "FIRST-PROMPT") {
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "the prompt was never checkpointed"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(300));
    child.kill().unwrap();
    let _ = child.wait();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let out = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "SECOND-PROMPT",
            "--continue",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
        ])
        .arg(&sessions)
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bodies = bodies.lock().unwrap();
    let body = common::body_json(&bodies[0]);
    let messages = body["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap_or("?"))
        .collect();
    assert_eq!(roles, ["user", "assistant", "user"], "{messages:#?}");
    assert!(messages[0].to_string().contains("FIRST-PROMPT"));
    assert!(messages[2].to_string().contains("SECOND-PROMPT"));
}
