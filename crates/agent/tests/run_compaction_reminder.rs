//! `run` e2e: the `/session` recall reminder rides the first request after a compaction.
//!
//! A compaction at the top of the loop runs after that turn's steer lane was drained, and the host
//! queues the reminder only when it sees `Compacted`. It used to wait for the next drain: the first
//! post-compaction request went out without it (the model acted on the lossy summary first), and when
//! the model answered without tools the stranded reminder forced a whole extra request.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use agent_core::{ContentBlock, Message, Session};
use beyond_agent::session_store::{SessionMeta, SessionRepo};
use common::{run_cmd, spawn_model_server, turn_text};

const REMINDER: &str = "was just compacted";

/// A session file holding six ~200-token exchanges, comfortably past a 200-token trigger.
fn seeded_session(dir: &std::path::Path) -> String {
    let repo = SessionRepo::open(dir).unwrap();
    let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
    let mut seed = Session::new();
    for i in 0..6 {
        seed.user(format!("question {i} {}", "u".repeat(400)));
        seed.push(Message::assistant(vec![ContentBlock::text(format!(
            "answer {i} {}",
            "a".repeat(400)
        ))]));
    }
    store.append_new(&seed.messages).unwrap();
    store.path().to_string_lossy().into_owned()
}

#[test]
fn the_reminder_rides_the_first_request_after_a_compaction_and_costs_no_extra_one() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(vec![
        turn_text("## Goal\nSUMMARY"),
        turn_text("answered"),
        turn_text("answered again"),
    ]);
    let out = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "continue please",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            &session_file,
            "--compact-at",
            "200",
            "--compaction-keep-recent-tokens",
            "1",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bodies: Vec<String> = bodies
        .lock()
        .unwrap()
        .iter()
        .filter(|b| !b.is_empty())
        .cloned()
        .collect();
    // The summary call, then the first post-compaction request.
    assert!(
        !bodies[0].contains(REMINDER),
        "the summary call carries no reminder"
    );
    assert!(
        bodies[1].contains(REMINDER),
        "the first post-compaction request must carry the reminder ({} requests)",
        bodies.len()
    );
    assert_eq!(
        bodies.len(),
        2,
        "a model that answers without tools ends the run: no extra request for a stranded reminder"
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("answered"));
}
