//! `run` and `serve` inject the same `/session` steers for the same events. Both feed a prompt's events
//! through one observer (`memory::SessionSteers`); before that each path had its own copy, and they had
//! drifted — `run` rebuilt its observer on every whole-run retry, so a retry re-armed the pressure
//! nudge and the model was told to checkpoint twice in one fill cycle, where `serve` told it once.
//!
//! Each scenario drives the real binary both ways against the same scripted model and compares what the
//! model was sent: the number of requests, and which steers each request carried.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::path::Path;
use std::process::Stdio;

use agent_core::{ContentBlock, Message, Session};
use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};
use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, sse, turn_text,
};
use serde_json::json;

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");
const NUDGE: &str = "getting large";
const REMINDER: &str = "was just compacted";

/// A stream that starts a text reply and then ends without a `message_stop`: a mid-stream drop.
const TRUNCATED: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"PARTIAL-\"}}\n\n";

fn usage_start(input_tokens: u32) -> serde_json::Value {
    json!({ "type": "message_start", "message": { "usage": { "input_tokens": input_tokens, "output_tokens": 1 } } })
}

fn text_with_input(text: &str, input_tokens: u32) -> String {
    sse(&[
        usage_start(input_tokens),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ])
}

fn bash_with_input(id: &str, input_tokens: u32) -> String {
    sse(&[
        usage_start(input_tokens),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id, "name": "bash", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "{\"command\":\"true\"}" } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// Per request: how many pressure nudges and compaction reminders it carried.
fn steers_per_request(bodies: &[String]) -> Vec<(usize, usize)> {
    bodies
        .iter()
        .map(|b| (b.matches(NUDGE).count(), b.matches(REMINDER).count()))
        .collect()
}

/// The recorded model requests, less the empty records of a connection that was opened and closed
/// without a request (a pooled connection the killed `serve` never used).
fn model_requests(recorded: &[String]) -> Vec<String> {
    recorded.iter().filter(|r| !r.is_empty()).cloned().collect()
}

fn via_run(turns: Vec<String>, session_file: &Path, prompt: &str, flags: &[&str]) -> Vec<String> {
    let (base, bodies) = spawn_model_server(turns);
    let session = session_file.to_string_lossy().into_owned();
    let out = run_cmd(BIN)
        .args([
            "run",
            prompt,
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            &session,
            "--json",
        ])
        .args(flags)
        .current_dir(session_file.parent().unwrap())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    model_requests(&bodies.lock().unwrap())
}

fn via_serve(turns: Vec<String>, session_file: &Path, prompt: &str, flags: &[&str]) -> Vec<String> {
    let (base, bodies) = spawn_model_server(turns);
    let mut child = serve_cmd(BIN, &base, &session_file.to_string_lossy())
        .args(flags)
        .current_dir(session_file.parent().unwrap())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = common::child_frames(&mut child);
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": prompt })).unwrap();
    stdin.flush().unwrap();
    let resp = read_until_response(&mut frames, "prompt");
    assert_eq!(resp.last().unwrap()["success"], true, "{resp:#?}");
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    model_requests(&bodies.lock().unwrap())
}

/// A session file holding six ~200-token exchanges, comfortably past a 200-token trigger.
fn seeded_session(dir: &Path) -> std::path::PathBuf {
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
    store.path().to_path_buf()
}

#[test]
fn a_whole_run_retry_does_not_re_arm_the_pressure_nudge_in_either_path() {
    // window 10000, reserve 2000 -> trigger 8000, pressure point 6400. Step 1 crosses it (the nudge
    // fires, riding step 2); step 2 drops until the in-turn retries are spent, so the whole run is
    // retried; the retry crosses the point again, which is still the same fill cycle — no second nudge.
    let script = || {
        let mut turns = vec![bash_with_input("t1", 7000)];
        turns.extend(std::iter::repeat_n(TRUNCATED.to_string(), 4));
        turns.push(text_with_input("done", 7000));
        turns.push(text_with_input("checkpointed", 20));
        turns
    };
    let flags = [
        "--context-window",
        "10000",
        "--compaction-reserve-tokens",
        "2000",
        "--retry-base-delay-ms",
        "10",
    ];
    let run_dir = tempfile::tempdir().unwrap();
    let serve_dir = tempfile::tempdir().unwrap();
    let run = via_run(script(), &run_dir.path().join("s.jsonl"), "go", &flags);
    let serve = via_serve(script(), &serve_dir.path().join("s.jsonl"), "go", &flags);

    assert_eq!(
        steers_per_request(&run),
        steers_per_request(&serve),
        "run and serve must send the same steers for the same events"
    );
    let last = run.last().unwrap();
    assert_eq!(
        last.matches(NUDGE).count(),
        1,
        "one fill cycle, one nudge: {:?}",
        steers_per_request(&run)
    );
}

#[test]
fn a_compaction_reminds_the_same_way_in_both_paths() {
    let script = || {
        vec![
            turn_text("## Goal\nSUMMARY"),
            turn_text("answered"),
            turn_text("answered again"),
        ]
    };
    let flags = [
        "--compact-at",
        "200",
        "--compaction-keep-recent-tokens",
        "1",
    ];
    let run_dir = tempfile::tempdir().unwrap();
    let serve_dir = tempfile::tempdir().unwrap();
    let run = via_run(
        script(),
        &seeded_session(run_dir.path()),
        "continue",
        &flags,
    );
    let serve = via_serve(
        script(),
        &seeded_session(serve_dir.path()),
        "continue",
        &flags,
    );

    assert_eq!(
        steers_per_request(&run),
        steers_per_request(&serve),
        "run and serve must send the same steers for the same events"
    );
    // The summary call, then the answer — which carries the reminder, so no third request.
    assert_eq!(
        steers_per_request(&run),
        [(0, 0), (0, 1)],
        "the reminder rides the first post-compaction request"
    );
}
