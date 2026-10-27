//! `run` e2e: a compaction is persisted, and a session resumed past one reloads what the run held.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use agent_core::{ContentBlock, Message, Role, Session};
use beyond_agent::session_store::{SessionMeta, SessionRepo, SessionStore};
use common::{run_cmd, spawn_model_server, turn_text};

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

fn run(base: &str, session_file: &str, prompt: &str) {
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            prompt,
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file,
            "--compact-at",
            "200",
            "--compaction-keep-recent-tokens",
            "1",
            "--no-session-memory",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn text(m: &Message) -> &str {
    match m.content.first() {
        Some(ContentBlock::Text { text, .. }) => text,
        _ => "",
    }
}

#[test]
fn a_compacted_run_reloads_as_the_summary_and_everything_after_it() {
    // Regression: `run` persisted only through the count-keyed `append_new`, which cannot see a
    // compaction's shrink. The reloaded session was the uncompacted history plus the prompt, and the
    // answer after the compaction was never written at all.
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let (base, _) = spawn_model_server(vec![
        turn_text("## Goal\nSUMMARY ONE"),
        turn_text("answered"),
    ]);
    run(&base, &session_file, "continue please");

    let (_, session) = SessionStore::open(session_file.clone().into()).unwrap();
    let m = &session.messages;
    assert!(
        text(&m[0]).contains("SUMMARY ONE"),
        "the compaction must be on disk: {m:#?}"
    );
    assert_eq!(text(m.last().unwrap()), "answered", "{m:#?}");
    assert!(
        m.iter().any(|m| text(m) == "continue please"),
        "the prompt survives the rewrite: {m:#?}"
    );
    assert!(
        m.windows(2).all(|w| w[0].role != w[1].role),
        "roles still alternate: {m:#?}"
    );
    assert_eq!(session.compaction.compactions, 1);
    assert_eq!(m[0].role, Role::User);
    let before = m.len();

    // A second run resumes from the compacted state (the summary, not the folded history, is what
    // reaches the model) and appends after it.
    let (base, bodies) = spawn_model_server(vec![turn_text("again")]);
    run(&base, &session_file, "and once more");
    let body = bodies.lock().unwrap()[0].clone();
    assert!(body.contains("SUMMARY ONE"), "{body}");
    // The user's own words ride the summary's `<user-requests>` carry by design; the folded turns
    // themselves must not come back.
    assert!(!body.contains("answer 0"), "folded history resent: {body}");
    let (_, session) = SessionStore::open(session_file.into()).unwrap();
    let m = &session.messages;
    assert_eq!(m.len(), before + 2, "{m:#?}");
    assert_eq!(text(m.last().unwrap()), "again", "{m:#?}");
    assert_eq!(session.compaction.compactions, 1);
}
