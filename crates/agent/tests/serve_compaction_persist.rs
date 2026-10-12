//! `serve` e2e: a compaction mid-run is on disk before the turn ends, so a crash after the list has
//! outgrown its old length reloads the compacted transcript and its provenance, not a corrupt one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use agent_core::{ContentBlock, Message, Role, Session};
use beyond_ai_agent::session_store::{SessionMeta, SessionRepo, SessionStore};
use common::{
    ISOLATED_HOME, SpawnGuarded, spawn_model_server_with_stalled_response, turn_text, turn_tool_use,
};
use serde_json::json;

fn text(m: &Message) -> String {
    m.content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_crash_after_a_mid_run_compaction_reloads_the_compacted_session() {
    let dir = tempfile::tempdir().unwrap();
    let readme = dir.path().join("notes.txt");
    std::fs::write(&readme, "hello").unwrap();
    let readme = readme.to_string_lossy().into_owned();

    // Six messages, about 2K tokens: a `read` round trip (provenance the compaction must carry),
    // then two long exchanges.
    let session_file = {
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let mut seed = Session::new();
        seed.user(format!("question 0 {}", "u".repeat(1_500)));
        seed.push(Message::assistant(vec![ContentBlock::tool_use(
            "seed",
            "read",
            json!({ "path": "src/seeded.rs" }),
        )]));
        seed.push(Message::tool_result("seed", "s".repeat(1_500), false));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(1_500),
        )]));
        seed.user(format!("question 1 {}", "u".repeat(1_500)));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "b".repeat(1_500),
        )]));
        store.append_new(&seed.messages).unwrap();
        store.path().to_string_lossy().into_owned()
    };

    // The summary, then five `read` turns (ten messages: the list outgrows its old seven), then a
    // model call that stalls, so the turn never ends and the process is killed inside it.
    let args = json!({ "path": readme }).to_string();
    let mut fast = vec![turn_text("## Goal\nSUMMARY OF THE SEED")];
    fast.extend((0..5).map(|i| turn_tool_use(&format!("r{i}"), "read", &args)));
    let base = spawn_model_server_with_stalled_response(fast, Duration::from_secs(60), vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
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
            "--compact-at",
            "1000",
            "--compaction-keep-recent-tokens",
            "1",
            "--no-tool-masking",
            "--no-session-memory",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "keep reading" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let (mut compacted, mut tool_ends) = (false, 0);
    while tool_ends < 5 {
        let f = frames
            .next_frame(Duration::from_secs(30))
            .expect("serve went quiet");
        match f["event"]["kind"].as_str() {
            Some("compacted") => compacted = true,
            Some("tool_end") => tool_ends += 1,
            _ => {}
        }
    }
    assert!(compacted, "the run compacted first");
    // The fifth result is checkpointed when the next model call goes out; give it a moment, then
    // kill the process mid-turn: no end-of-turn persist ever runs.
    std::thread::sleep(Duration::from_millis(500));
    child.kill().unwrap();
    child.wait().unwrap();

    let (_, session) = SessionStore::open(session_file.into()).unwrap();
    let m = &session.messages;
    assert!(
        text(&m[0]).contains("SUMMARY OF THE SEED"),
        "the compaction is on disk: {m:#?}"
    );
    assert!(
        !m.iter().any(|m| text(m).contains("question 0")),
        "the folded history is not on the active path: {m:#?}"
    );
    let reads = m
        .iter()
        .flat_map(|m| &m.content)
        .filter(|b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id.starts_with('r')))
        .count();
    assert_eq!(
        reads, 5,
        "every round trip after the compaction survived: {m:#?}"
    );
    assert!(m.windows(2).all(|w| w[0].role != w[1].role), "{m:#?}");
    assert_eq!(m[0].role, Role::User);
    assert!(m.iter().any(|m| text(m) == "keep reading"), "{m:#?}");
    // Provenance came through the crash too.
    assert_eq!(session.compaction.compactions, 1);
    assert!(
        session
            .compaction
            .read_files
            .contains(&"src/seeded.rs".to_string()),
        "{:?}",
        session.compaction
    );
}
