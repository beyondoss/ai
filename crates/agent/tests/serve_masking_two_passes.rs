//! `serve` e2e: a run that masks tool output twice persists both passes.
//!
//! Each masking pass replaces `Session::edited`, so the first pass's edits reach disk only if
//! something writes them before the second pass. `run` does, at every checkpoint; `serve`'s mid-run
//! checkpoints used to pass no edits at all, so only the second pass survived the run — and every
//! reopen then sent the first pass's results unmasked again (~18.7K extra tokens here), missed the
//! prompt cache, and triggered masking a second time.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::process::{Command, Stdio};

use agent_core::compaction::MASKED_PREFIX;
use agent_core::{ContentBlock, Message, Session};
use beyond_ai_agent::session_store::{SessionMeta, SessionRepo, SessionStore};
use common::{
    ISOLATED_HOME, SpawnGuarded, read_until_response, spawn_model_server, sse, turn_text,
};
use serde_json::json;

/// A task, twelve `read` turns of 4,000 chars each, and a closing reply.
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

/// A turn that runs a `bash` printing 16,000 chars — enough to push the prompt back over the
/// trigger — reporting `input_tokens` as its real prompt size.
fn bash_turn(id: &str, input_tokens: u64) -> String {
    let args = json!({ "command": "head -c 16000 /dev/zero | tr '\\0' y" }).to_string();
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": input_tokens, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id, "name": "bash", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": args } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
        json!({ "type": "message_stop" }),
    ])
}

fn masked(m: &Message) -> bool {
    m.content.iter().any(|b| {
        matches!(b, ContentBlock::ToolResult { content, .. } if content.starts_with(MASKED_PREFIX))
    })
}

fn serve(base: &str, session_file: &str) -> std::process::Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    cmd.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-file",
        session_file,
        "--compaction-keep-recent-tokens",
        "1",
        "--no-session-memory",
        "--compact-at",
        "10000",
        "--tool-masking-keep-turns",
        "1",
        "--tool-masking-min-tokens",
        "1000",
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    cmd
}

/// Both passes are on disk after the run, the reloaded transcript is exactly the one the run held
/// (masked content and #261's per-message `prior_turn_tokens` alike), and a reopen sends the model
/// every placeholder it saw before.
#[test]
fn two_masking_passes_in_one_serve_run_both_reach_disk() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = seeded_session(dir.path());
    let (base, bodies) = spawn_model_server(vec![
        bash_turn("b1", 3_500),
        bash_turn("b2", 6_500),
        turn_text("done"),
    ]);
    let mut child = serve(&base, &session_file).spawn_guarded();
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
    let passes = frames
        .iter()
        .filter(|f| f["event"]["kind"] == json!("tool_output_masked"))
        .count();
    assert_eq!(
        passes, 2,
        "the scenario needs two masking passes in one run"
    );
    let seen = bodies
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .matches(MASKED_PREFIX)
        .count();

    writeln!(stdin, "{}", json!({ "type": "get_messages", "id": "m" })).unwrap();
    stdin.flush().unwrap();
    let live = read_until_response(&mut stdout, "get_messages");
    let mut live = live.last().unwrap()["data"]["messages"].clone();
    for m in live.as_array_mut().unwrap() {
        m.as_object_mut().unwrap().remove("id");
    }
    drop(stdin);
    child.wait().unwrap();

    let (_, session) = SessionStore::open(session_file.clone().into()).unwrap();
    let m = &session.messages;
    for i in (2..=22).step_by(2) {
        assert!(masked(&m[i]), "pass-1 result {i} is not masked on disk");
    }
    let on_disk = m.iter().filter(|m| masked(m)).count();
    assert_eq!(on_disk, seen, "every placeholder the model saw is on disk");
    assert!(
        m.iter().any(|m| m.prior_turn_tokens.is_some()),
        "the run measured at least one turn"
    );
    assert_eq!(
        serde_json::to_value(&**m).unwrap(),
        live,
        "the reloaded transcript is the one the run held"
    );

    // A reopen sends the transcript as stored: the prefix the model already cached.
    let (base, bodies) = spawn_model_server(vec![turn_text("again")]);
    let mut child = serve(&base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "and again" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    drop(stdin);
    child.wait().unwrap();
    let body = bodies.lock().unwrap()[0].clone();
    assert_eq!(body.matches(MASKED_PREFIX).count(), seen, "{body}");
}
