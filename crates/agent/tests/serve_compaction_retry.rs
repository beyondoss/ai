//! `serve` e2e: Compaction (manual/proactive) and whole-run auto-retry/backoff.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use common::{
    ISOLATED_HOME, read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

#[test]
fn serve_set_auto_compaction_toggles_and_rejects_a_non_boolean() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_compaction");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert_eq!(frames.last().unwrap()["data"]["auto_compaction"], false);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_compaction");
    assert_eq!(frames.last().unwrap()["data"]["auto_compaction"], true);

    // Missing/non-boolean `enabled` is rejected, not silently coerced.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": "yes" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_compaction");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_no_compaction_flag_starts_with_auto_compaction_disabled() {
    // Track L26 (pi-parity fix): `serve` had no `--no-compaction`/`AI_AGENT_NO_COMPACTION` equivalent
    // of `run`'s identical flag at all — `current_auto_compaction` was hardcoded `true` at startup,
    // only ever changeable afterward via `set_auto_compaction`.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .args(["--no-compaction"])
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["auto_compaction"],
        false,
        "--no-compaction must start the process with auto-compaction already disabled: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_auto_compaction_persists_across_a_restart() {
    // Track L26 (pi-parity fix): `set_auto_compaction` used to mutate only an in-process local — a
    // restarted `serve` (no `--no-compaction`/env var given either time) silently reverted to
    // enabled-by-default, with no way for a client's toggle to survive the restart. Uses a real,
    // writable `HOME` (not the usual `ISOLATED_HOME`) so the persisted `agent settings` file genuinely
    // round-trips across the two separate process spawns below.
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file)
        .env("HOME", home.path())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_compaction");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    drop(stdin);
    child.wait().unwrap();

    // A brand-new process, same HOME, no `--no-compaction` flag and no env var at all — must still
    // start with auto-compaction disabled, purely from the persisted setting the first process wrote.
    let (base2, _bodies2) = spawn_model_server(vec![]);
    let session_file2 = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
    let mut child2 = serve_cmd(bin, &base2, &session_file2)
        .env("HOME", home.path())
        .spawn()
        .unwrap();
    let mut stdin2 = child2.stdin.take().unwrap();
    let mut stdout2 = BufReader::new(child2.stdout.take().unwrap());
    writeln!(stdin2, "{}", json!({ "type": "get_state" })).unwrap();
    stdin2.flush().unwrap();
    let frames2 = read_until_response(&mut stdout2, "get_state");
    assert_eq!(
        frames2.last().unwrap()["data"]["auto_compaction"],
        false,
        "a restarted process must recover the persisted auto-compaction override: {frames2:#?}"
    );

    drop(stdin2);
    child2.wait().unwrap();
}

#[test]
fn serve_compact_forwards_custom_instructions_to_the_summarization_call() {
    // Track M14: `compact`'s `custom_instructions` must actually reach the summarization model call —
    // matching pi's own `compact(customInstructions)` — not just be silently ignored.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    // Two ordinary conversational turns (4 messages: user/assistant/user/assistant) build up enough
    // history for `find_cut` to find a real cut point once `--compaction-keep-recent-tokens` is tiny;
    // the third response is what the manual `compact` call's own summarization request receives.
    let (base, bodies) = spawn_model_server(vec![
        turn_text("answer one"),
        turn_text("answer two"),
        turn_text("SUMMARY"),
    ]);

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
            "--compaction-keep-recent-tokens",
            "1",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["hello", "again"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(
        stdin,
        "{}",
        json!({
            "type": "compact",
            "custom_instructions": "keep every detail about the auth refactor"
        })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "compact");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["compacted"], true, "{frames:#?}");
    // Task #26: `reason` is only populated for a no-op — `null` on a real compaction.
    assert!(data["reason"].is_null(), "{frames:#?}");
    // Pi-parity fix: the response previously carried only `compacted: bool` — pi's own `compact`
    // returns the generated summary text and a post-compaction token estimate too.
    let summary = data["summary"]
        .as_str()
        .expect("a real compaction must carry its generated summary text");
    assert!(
        !summary.is_empty(),
        "summary must not be empty: {frames:#?}"
    );
    assert!(
        data["tokens_before"].as_u64().is_some(),
        "tokens_before must be populated on a real compaction: {frames:#?}"
    );
    assert!(
        data["tokens_after"].as_u64().is_some(),
        "tokens_after must be populated on a real compaction: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert!(
        recorded
            .iter()
            .any(|b| b.contains("Additional focus: keep every detail about the auth refactor")),
        "the custom instructions must reach the summarization call: {recorded:#?}"
    );
}

#[test]
fn serve_compact_reports_too_small_reason_for_a_session_with_nothing_to_compact() {
    // Task #26 (pi-parity fix): a manual `compact` on a session with no worthwhile cut point at all
    // must report `reason: "too_small"` — pi's own "Nothing to compact (session too small)" — not the
    // same undifferentiated `compacted: false` an "already compacted" no-op also used to produce.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    // No turns scripted at all: a real compaction would make a model call the mock has nothing queued
    // for, failing the test loudly rather than silently passing.
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // No `prompt` at all yet — the session is empty, well under `find_split_cut`'s minimum.
    writeln!(stdin, "{}", json!({ "type": "compact" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "compact");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["compacted"], false, "{frames:#?}");
    assert_eq!(data["reason"], "too_small", "{frames:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_compact_reports_already_compacted_reason_when_nothing_new_followed_the_prior_summary() {
    // Task #26 (pi-parity fix): a manual `compact` on a session that already has a compaction summary
    // at its head, with nothing new since that still exceeds the recent-token budget, must report
    // `reason: "already_compacted"` — pi's own "Already compacted" — distinct from the "too_small" case
    // above. Seeded directly (mirrors `agent_core::agent::tests::
    // compact_is_a_no_op_on_a_clean_boundary_when_nothing_new_followed_the_prior_summary`'s exact
    // fixture shape) rather than driven through a real first `compact` call: forcing a *real* first
    // compaction on a small test conversation needs an artificially tiny
    // `--compaction-keep-recent-tokens`, which then makes the post-compaction residual too small for
    // `find_split_cut` to find a second cut point at all (landing on "too_small" instead) — the two
    // conditions are easiest to hit independently, not chained through one process.
    let dir = tempfile::tempdir().unwrap();
    let session_file = {
        use agent_core::compaction::SUMMARY_MARKER;
        use agent_core::{ContentBlock, Message, Session};
        use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};

        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let mut seed = Session::new();
        seed.push(Message::user(format!(
            "{SUMMARY_MARKER}\n\nprior summary body"
        )));
        seed.user("second question");
        seed.push(Message::assistant(vec![ContentBlock::text("second reply")]));
        seed.user("third question");
        seed.push(Message::assistant(vec![ContentBlock::text("third reply")]));
        store.append_new(&seed.messages).unwrap();
        store.path().to_string_lossy().into_owned()
    };

    // No turns scripted at all: this must be an early-return no-op with zero model calls, not a real
    // compaction attempt — a network call here would fail the test loudly rather than silently pass.
    let (base, bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "compact" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "compact");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["compacted"], false, "{frames:#?}");
    assert_eq!(data["reason"], "already_compacted", "{frames:#?}");

    drop(stdin);
    child.wait().unwrap();
    assert_eq!(
        bodies.lock().unwrap().len(),
        0,
        "an already-compacted no-op must make zero model calls"
    );
}

#[test]
fn serve_proactively_compacts_a_resumed_large_session_on_its_very_next_prompt() {
    // B-M14 pi-parity gap (fixed): a large session persisted by an *earlier* process (no live
    // `Session` in memory to carry `last_input_tokens` forward) used to resume with that field at
    // its zero default — `SessionStore::open` never restored it from the persisted transcript — so
    // `should_compact` couldn't fire until a fresh turn produced real usage, one whole turn later
    // than it should. Matches pi's own `pre-prompt-compaction-no-continue` regression: the very
    // first prompt sent to a resumed, already-over-threshold session must trigger compaction before
    // that prompt's own answer, not after some wasted extra turn.
    let dir = tempfile::tempdir().unwrap();

    // Seed the session file directly (as an earlier, now-exited process would have left it) with
    // enough text that its char/4 estimate comfortably exceeds the tiny threshold below — four
    // messages so `find_cut` (which declines short conversations) has a real boundary to find.
    let session_file = {
        use agent_core::{ContentBlock, Message, Session};
        use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};

        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let mut seed = Session::new();
        seed.user("u".repeat(400)); // ~100 estimated tokens
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        store.append_new(&seed.messages).unwrap();
        store.path().to_string_lossy().into_owned()
    };

    // Scripted in order: the proactive compaction's own summarization call, then the real answer to
    // the new prompt — if compaction didn't fire before the prompt's own turn, the second scripted
    // response would be consumed as the (unsummarized) prompt's answer instead, and this test's
    // assertions on call count / compacted-event ordering would catch the mismatch.
    let (base, bodies) = spawn_model_server(vec![turn_text("SUMMARY"), turn_text("answered")]);

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
            "--context-window",
            "200",
            "--compaction-reserve-tokens",
            "50",
            "--compaction-keep-recent-tokens",
            "1",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // The very first prompt this (freshly-spawned, freshly-resumed) process ever sends.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "one more thing" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    drop(stdin);
    child.wait().unwrap();

    // Exactly two model calls: the proactive compaction, then the real answer — not three (which
    // would mean compaction was skipped this turn and only caught up reactively on an overflow, or
    // deferred to a second prompt).
    assert_eq!(
        bodies.lock().unwrap().len(),
        2,
        "compaction must fire on this very first prompt, not after it"
    );
    let compacted = frames.iter().any(|f| {
        f.get("type").and_then(Value::as_str) == Some("event")
            && f["event"]["kind"] == json!("compacted")
    });
    assert!(
        compacted,
        "expected a compacted event during this prompt's own processing: {frames:#?}"
    );
}

#[test]
fn serve_compact_preserves_pre_compaction_entries_in_get_tree() {
    // F-M3 (pi: rpc.test.ts:328-340): the storage layer's own non-destructive-compaction guarantee
    // (`session_store.rs`'s `rewrite_compacted_preserves_folded_messages_and_records_provenance`) was
    // never proven end-to-end through the live RPC surface — a `compact` command followed by `get_tree`
    // must still show the folded, pre-compaction messages (matching this codebase's append-only
    // compaction posture: folded away from the *active* path, never deleted from disk).
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, _bodies) = spawn_model_server(vec![
        turn_text("answer one"),
        turn_text("answer two"),
        turn_text("SUMMARY"),
    ]);

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
            "--compaction-keep-recent-tokens",
            "1",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    for msg in ["pre-compaction-hello", "pre-compaction-again"] {
        writeln!(stdin, "{}", json!({ "type": "prompt", "message": msg })).unwrap();
        stdin.flush().unwrap();
        read_until_response(&mut stdout, "prompt");
    }

    writeln!(stdin, "{}", json!({ "type": "compact" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "compact");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        frames.last().unwrap()["data"]["compacted"],
        true,
        "{frames:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_tree" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_tree");
    let nodes = frames.last().unwrap()["data"]["nodes"].as_array().unwrap();

    // Every pre-compaction message is still readable by its original node, even though none of them
    // are reachable from the new active tip anymore (the compacted summary starts a fresh, detached
    // chain — see `SessionStore::rewrite_compacted`'s own doc comment).
    for text in [
        "pre-compaction-hello",
        "answer one",
        "pre-compaction-again",
        "answer two",
    ] {
        assert!(
            nodes
                .iter()
                .any(|n| n["preview"].as_str().is_some_and(|p| p.contains(text))),
            "folded pre-compaction message {text:?} must still be present in get_tree: {nodes:#?}"
        );
    }

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_auto_retry_toggles_and_rejects_a_non_boolean() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_retry");
    assert_eq!(frames.last().unwrap()["success"], true);
    assert_eq!(frames.last().unwrap()["data"]["auto_retry"], false);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": true })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_retry");
    assert_eq!(frames.last().unwrap()["data"]["auto_retry"], true);

    // Missing/non-boolean `enabled` is rejected, not silently coerced.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": "yes" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_auto_retry");
    assert_eq!(frames.last().unwrap()["success"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_auto_retry_false_fails_immediately_instead_of_retrying_a_dropped_stream() {
    // A stream that opens (`message_start`) but closes with no `message_stop` is a dropped connection —
    // normally retried (`agent_core`'s mid-stream retry). With auto_retry off, it must surface as an
    // immediate `prompt` failure instead, with no second request ever reaching the model server.
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![truncated.to_string()]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_retry", "enabled": false })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_auto_retry");

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false);
    assert!(
        response["error"].as_str().unwrap().contains("stream ended"),
        "got: {response:#?}"
    );
    assert_eq!(
        bodies.lock().unwrap().len(),
        1,
        "auto_retry(false) must not attempt a second request"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_auto_retries_a_whole_run_after_mid_stream_retry_is_exhausted() {
    // agent-core's own mid-stream retry (`MAX_MID_STREAM_RETRIES = 3`) gives up after 1 initial + 3
    // retried attempts, all against a stream that dies before `message_stop` — 4 requests total,
    // exhausting that layer entirely and returning `Err` to `run_events_steered`. This is the whole-run
    // auto-retry layer's job to pick up from there: automatically re-invoke the run against the same
    // session one more time, which succeeds — a 5th request the model server actually sees, and no
    // second user turn appended (still the same `prompt`).
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_text("recovered"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let auto_retry_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_start"))
        .collect();
    assert_eq!(
        auto_retry_frames.len(),
        1,
        "expected exactly one auto_retry notice, got: {frames:#?}"
    );
    assert_eq!(auto_retry_frames[0]["attempt"], 1);
    assert_eq!(auto_retry_frames[0]["max_attempts"], 3);
    assert!(
        auto_retry_frames[0]["error"]
            .as_str()
            .unwrap()
            .contains("stream ended"),
        "got: {auto_retry_frames:#?}"
    );

    // The terminal notice for the retry sequence: the retried attempt succeeded.
    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(
        auto_retry_end_frames.len(),
        1,
        "expected exactly one auto_retry_end notice, got: {frames:#?}"
    );
    assert_eq!(auto_retry_end_frames[0]["success"], true);
    assert_eq!(auto_retry_end_frames[0]["attempt"], 1);
    assert!(
        auto_retry_end_frames[0].get("final_error").is_none(),
        "a successful retry must not carry a final_error: {:?}",
        auto_retry_end_frames[0]
    );

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(
        bodies.lock().unwrap().len(),
        5,
        "4 exhausted mid-stream attempts + 1 successful whole-run retry"
    );

    // The recovered turn's own text must have reached the transcript — proof the retry replayed the
    // *same* user turn rather than silently dropping it or duplicating it.
    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let messages = &frames.last().unwrap()["data"]["messages"];
    let dump = messages.to_string();
    assert_eq!(
        messages.as_array().unwrap().len(),
        2,
        "exactly one user turn + one assistant turn, no duplicate: {dump}"
    );
    assert!(dump.contains("recovered"), "got: {dump}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_whole_run_retry_recovery_attempt_can_itself_dispatch_tool_calls() {
    // B-L3 pi-parity test gap (fixed): every existing whole-run-retry test recovers into a plain text
    // turn. Nothing proved the retried attempt can continue normally into a *tool-dispatch* turn — a
    // structurally different path through `run_events_steered` (another model round trip after the
    // tool result, real `bash` execution, a second assistant message).
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_tool_use(
            "toolu_retry",
            "bash",
            &json!({ "command": "echo recovered-tool" }).to_string(),
        ),
        turn_text("done after tool"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(
        bodies.lock().unwrap().len(),
        6,
        "4 exhausted mid-stream attempts + the recovered tool-call turn + its follow-up text turn"
    );

    writeln!(stdin, "{}", json!({ "type": "get_messages" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_messages");
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(dump.contains("recovered-tool"), "tool actually ran: {dump}");
    assert!(dump.contains("done after tool"), "got: {dump}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_whole_run_retry_succeeds_on_its_second_attempt_not_its_first() {
    // B-L4 pi-parity test gap (fixed): every existing whole-run-retry test hits one of the two
    // boundary cases (first retry attempt succeeds, or all `MAX_RUN_RETRIES` attempts are exhausted).
    // This pins the middle case — the first whole-run retry attempt *also* fails, and the second one
    // recovers — proving the loop actually keeps going past attempt 1 rather than only ever handling
    // "succeeds immediately" or "never succeeds".
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    // Each block of 4 truncated responses exhausts one whole-run attempt's own mid-stream retry
    // budget (`MAX_MID_STREAM_RETRIES = 3`, so 1 initial + 3 retries per attempt) before the whole-run
    // layer re-invokes the entire run from scratch, with a fresh mid-stream budget of its own.
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_text("recovered on the second whole-run attempt"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let auto_retry_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_start"))
        .collect();
    assert_eq!(
        auto_retry_frames.len(),
        2,
        "expected two auto_retry notices (attempt 1 failed too, attempt 2 was tried): {frames:#?}"
    );
    assert_eq!(auto_retry_frames[0]["attempt"], 1);
    assert_eq!(auto_retry_frames[1]["attempt"], 2);

    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(
        auto_retry_end_frames.len(),
        1,
        "only one terminal notice, once the sequence actually settles: {frames:#?}"
    );
    assert_eq!(auto_retry_end_frames[0]["success"], true);
    assert_eq!(
        auto_retry_end_frames[0]["attempt"], 2,
        "must report which attempt actually succeeded, not just that one eventually did"
    );

    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(bodies.lock().unwrap().len(), 9);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_abort_retry_interrupts_a_pending_whole_run_retry_backoff() {
    // Same failure shape as the auto-retry test above (agent-core's own mid-stream retry exhausts
    // after 4 attempts, handing an `Err` to the whole-run retry layer) — but here the client cancels
    // the pending backoff instead of letting it run its course. Confirms the backoff wait is
    // genuinely interruptible (not a bare `sleep` nothing can touch) and that cancelling it surfaces
    // the real underlying error rather than either hanging for the full delay or silently retrying
    // anyway.
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        truncated.to_string(),
        turn_text("would have recovered"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();

    // Read frames one at a time until the `auto_retry` notice arrives — sent right before the
    // 2-second whole-run backoff wait starts, *after* agent-core's own mid-stream retry has already
    // spent its own ~1.75s of internal backoff exhausting itself — then immediately cancel it. Timing
    // starts here, not at the `prompt` send, so the assertion below isolates just the whole-run
    // backoff-interruption latency from that unrelated, expected mid-stream delay.
    let mut line = String::new();
    loop {
        line.clear();
        assert!(
            stdout.read_line(&mut line).unwrap() > 0,
            "stdout closed before an auto_retry notice arrived"
        );
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(trimmed).unwrap();
        if v.get("type").and_then(Value::as_str) == Some("auto_retry_start") {
            break;
        }
    }
    let start = std::time::Instant::now();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "abort_retry", "id": "cancel-1" })
    )
    .unwrap();
    stdin.flush().unwrap();

    let abort_frames = read_until_response(&mut stdout, "abort_retry");
    assert_eq!(
        abort_frames.last().unwrap()["success"],
        true,
        "{abort_frames:#?}"
    );

    // The still-pending `prompt` must resolve right after — not wait out the full 2s backoff — with
    // the *original* mid-stream error, and the 5th (would-be-recovering) request must never fire.
    let frames = read_until_response(&mut stdout, "prompt");
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(1000),
        "abort_retry must interrupt the 2s backoff near-instantly, not wait most/all of it out: \
         took {elapsed:?}"
    );
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "got: {response:#?}");
    assert!(
        response["error"].as_str().unwrap().contains("stream ended"),
        "must surface the real underlying error, not a synthetic cancellation: {response:#?}"
    );
    assert_eq!(
        bodies.lock().unwrap().len(),
        4,
        "the cancelled retry must never have fired the 5th, would-be-recovering request: {:?}",
        bodies.lock().unwrap()
    );

    // The terminal notice for the retry sequence: it never got to retry — the backoff was cancelled.
    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(
        auto_retry_end_frames.len(),
        1,
        "expected exactly one auto_retry_end notice, got: {frames:#?}"
    );
    assert_eq!(auto_retry_end_frames[0]["success"], false);
    assert_eq!(auto_retry_end_frames[0]["attempt"], 1);
    assert_eq!(auto_retry_end_frames[0]["final_error"], "retry cancelled");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_auto_retry_exhausts_all_attempts_and_reports_failure() {
    // Companion to `serve_auto_retries_a_whole_run_after_mid_stream_retry_is_exhausted` above, which
    // only proves attempt 1 recovers — this drives all `MAX_RUN_RETRIES` (3) whole-run attempts to
    // fail, ending in a reported failure with no recovery. Each whole-run attempt itself exhausts
    // agent-core's own mid-stream retry (1 initial + 3 retried, all truncated) before this layer even
    // sees it, so 3 whole-run attempts need 12 truncated stream chunks total, no successful turn at
    // the end. Slow (~15-20s of real backoff sleep) but this is the only way to observe the real
    // exponential-backoff schedule end to end.
    let truncated = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![truncated.to_string(); 12]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let auto_retry_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_start"))
        .collect();
    assert_eq!(
        auto_retry_frames.len(),
        3,
        "expected exactly 3 auto_retry notices (attempts 1, 2, 3): {frames:#?}"
    );
    let attempts: Vec<i64> = auto_retry_frames
        .iter()
        .map(|f| f["attempt"].as_i64().unwrap())
        .collect();
    assert_eq!(attempts, vec![1, 2, 3]);

    let auto_retry_end_frames: Vec<&Value> = frames
        .iter()
        .filter(|f| f.get("type").and_then(Value::as_str) == Some("auto_retry_end"))
        .collect();
    assert_eq!(auto_retry_end_frames.len(), 1, "got: {frames:#?}");
    assert_eq!(auto_retry_end_frames[0]["success"], false);
    assert_eq!(auto_retry_end_frames[0]["attempt"], 3);
    assert!(
        auto_retry_end_frames[0].get("final_error").is_some(),
        "an exhausted retry sequence must carry a final_error: {:?}",
        auto_retry_end_frames[0]
    );

    // The prompt command's own terminal response reports failure too — no silent success.
    let prompt_resp = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .unwrap();
    assert_eq!(prompt_resp["success"], false, "got: {prompt_resp:#?}");

    drop(stdin);
    child.wait().unwrap();
}
