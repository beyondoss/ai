//! `run` e2e: a failed invocation must not hijack the next `--continue`. A run that failed validation
//! (an unreadable `--output-schema`, a bad `--memory` backend) had already created its session file,
//! so the next `--continue` resumed that empty session and the model "forgot" the real conversation.
//! Those checks now run before any session is created, and `--continue` passes over header-only
//! sessions in any case.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text};

fn session_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(session_files(&p));
            } else if p.extension().is_some_and(|e| e == "jsonl") {
                out.push(p);
            }
        }
    }
    out
}

#[test]
fn continue_after_a_failed_invocation_resumes_the_real_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions");
    let (base, bodies) = spawn_model_server(vec![turn_text("ok-1"), turn_text("ok-3")]);
    let run = |extra: &[&str]| {
        run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
            .args([
                "run",
                "--gateway-url",
                &base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
            ])
            .arg(&sessions)
            .args(extra)
            .current_dir(dir.path())
            .stdin(Stdio::null())
            .output()
            .unwrap()
    };

    let first = run(&["REMEMBER-THE-FIRST-TASK"]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(session_files(&sessions).len(), 1);

    let bad_schema = run(&["second", "--output-schema", "/nonexistent/schema.json"]);
    assert_eq!(bad_schema.status.code(), Some(2));
    let bad_memory = run(&["second", "--memory", "nosuchscheme://x"]);
    assert_eq!(
        bad_memory.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&bad_memory.stderr)
    );
    assert_eq!(
        session_files(&sessions).len(),
        1,
        "a failed invocation creates no session"
    );

    let third = run(&["third", "--continue"]);
    assert!(
        third.status.success(),
        "{}",
        String::from_utf8_lossy(&third.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "the failed invocations never reached the model"
    );
    assert!(
        bodies[1].contains("REMEMBER-THE-FIRST-TASK"),
        "--continue carries the first run's history"
    );
}
