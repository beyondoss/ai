//! `run` and `serve` e2e: masking old tool output before summarizing, and that the masked transcript
//! persists — reloaded from disk by an independent `SessionStore::open`, not read back in memory.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::Stdio;

use agent_core::compaction::MASKED_PREFIX;
use agent_core::{ContentBlock, Message, Session};
use beyond_agent::session_store::{SessionMeta, SessionRepo, SessionStore};
use common::{
    ISOLATED_HOME, SpawnGuarded, read_until_response, run_cmd, spawn_model_server, turn_text,
};
use serde_json::json;

/// A task, twelve `read` turns of 4,000 chars each (about 12K tokens in all), and a closing reply.
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

/// A 10K trigger, two turns kept unmasked, and a 2,500-token minimum, so masking the eleven older
/// results (about 10.7K tokens) is enough on its own.
const FLAGS: [&str; 6] = [
    "--compact-at",
    "10000",
    "--tool-masking-keep-turns",
    "2",
    "--tool-masking-min-tokens",
    "2500",
];

fn masked(m: &Message) -> bool {
    m.content.iter().any(|b| {
        matches!(b, ContentBlock::ToolResult { content, .. } if content.starts_with(MASKED_PREFIX))
    })
}

fn run(base: &str, session_file: &str, extra: &[&str]) {
    let bin = env!("CARGO_BIN_EXE_agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "now fix them",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file,
            "--compaction-keep-recent-tokens",
            "1",
            "--no-session-memory",
        ])
        .args(FLAGS)
        .args(extra)
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

/// The reloaded transcript has the eleven old results masked, the last one intact, and the answer.
fn assert_masked_on_disk(session_file: &str) {
    let (_, session) = SessionStore::open(session_file.into()).unwrap();
    let m = &session.messages;
    assert_eq!(m.len(), 28, "{m:#?}");
    for i in (2..=22).step_by(2) {
        assert!(masked(&m[i]), "result {i} not masked on disk: {:?}", m[i]);
    }
    assert!(!masked(&m[24]), "the kept turn's result was masked");
    assert!(m.windows(2).all(|w| w[0].role != w[1].role));
    assert!(matches!(
        &m[27].content[0], ContentBlock::Text { text, .. } if &**text == "done"
    ));
}

#[test]
fn run_masks_old_tool_output_instead_of_summarizing_and_persists_it() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);
    run(&base, &session_file, &[]);
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1, "no summarization call");
    assert!(
        bodies[0].contains(MASKED_PREFIX),
        "the model saw the placeholders"
    );
    drop(bodies);
    assert_masked_on_disk(&session_file);

    // A resumed run sends the masked transcript as it was stored: the prefix the model already cached.
    let (base, bodies) = spawn_model_server(vec![turn_text("again")]);
    run(&base, &session_file, &[]);
    let body = bodies.lock().unwrap()[0].clone();
    assert_eq!(body.matches(MASKED_PREFIX).count(), 11, "{body}");
}

#[test]
fn run_without_tool_masking_summarizes_instead() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(vec![turn_text("## Goal\nSUMMARY"), turn_text("done")]);
    run(&base, &session_file, &["--no-tool-masking"]);
    assert_eq!(bodies.lock().unwrap().len(), 2, "summary, then the answer");
    let (_, session) = SessionStore::open(session_file.into()).unwrap();
    assert!(session.messages.iter().all(|m| !masked(m)));
    assert_eq!(session.compaction.compactions, 1);
}

#[test]
fn serve_persists_a_masking_pass_for_the_next_process() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);
    let mut child = crate::common::agent_cmd()
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
            "--compaction-keep-recent-tokens",
            "1",
            "--no-session-memory",
        ])
        .args(FLAGS)
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "now fix them" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert!(
        frames
            .iter()
            .any(|f| f["event"]["kind"] == json!("tool_output_masked")),
        "the pass is on the event stream: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();
    assert_eq!(bodies.lock().unwrap().len(), 1, "no summarization call");
    assert_masked_on_disk(&session_file);
}

#[test]
fn run_summarizes_from_the_cached_conversation_unless_told_not_to() {
    // The summary call is the live conversation plus an instruction by default; the flag forces
    // the re-render (its own system prompt over a rendered transcript).
    for (extra, cached) in [(&[][..], true), (&["--no-cached-summary"][..], false)] {
        let dir = tempfile::tempdir().unwrap();
        let session_file = seeded_session(dir.path());
        let (base, bodies) =
            spawn_model_server(vec![turn_text("## Goal\nSUMMARY"), turn_text("done")]);
        let mut args = vec!["--no-tool-masking"];
        args.extend_from_slice(extra);
        run(&base, &session_file, &args);
        let body = bodies.lock().unwrap()[0].clone();
        assert_eq!(body.contains("do not call any tools"), cached, "{extra:?}");
        // The cached call carries the conversation's tools; the re-render carries none.
        assert_eq!(body.contains("\"tools\""), cached, "{extra:?}");
        // Either way the pending prompt waits for the real turn.
        assert!(!body.contains("now fix them") || !cached, "{body}");
    }
}
