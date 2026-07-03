//! `serve` e2e: Read-only runtime introspection: `get_state`, `get_session_stats`, `cwd_stale`, session naming.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};

use common::{
    read_until_response, serve_cmd, serve_dir_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

#[test]
fn serve_get_state_and_get_session_stats_answer_live_during_a_prompt() {
    use std::time::Duration;

    // A tool-heavy turn (a `bash` sleep keeps it in flight) must still answer read-only progress
    // queries instead of rejecting them as busy — the whole point of H-4: a client polling for a live
    // "tokens/steps so far" indicator shouldn't have to wait for the turn to finish.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let turn1 = turn_tool_use(
        "toolu_live",
        "bash",
        &json!({ "command": "sleep 0.5" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(150)); // let the first turn's usage land, mid-`sleep 0.5`

    writeln!(
        stdin,
        "{}",
        json!({ "type": "get_session_stats", "id": "s1" })
    )
    .unwrap();
    writeln!(stdin, "{}", json!({ "type": "get_state", "id": "g1" })).unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let stats_resp = frames
        .iter()
        .find(|f| f["command"] == "get_session_stats" && f["id"] == "s1")
        .unwrap_or_else(|| panic!("no get_session_stats response: {frames:#?}"));
    assert_eq!(stats_resp["success"], true, "{stats_resp:#?}");
    assert!(
        stats_resp["data"]["input_tokens"].as_u64().unwrap_or(0) > 0,
        "the first turn's usage should already be mirrored live: {stats_resp:#?}"
    );

    let state_resp = frames
        .iter()
        .find(|f| f["command"] == "get_state" && f["id"] == "g1")
        .unwrap_or_else(|| panic!("no get_state response: {frames:#?}"));
    assert_eq!(state_resp["success"], true, "{state_resp:#?}");
    assert!(state_resp["data"]["message_count"].is_null());
    assert!(state_resp["data"]["session_id"].is_string());
}

#[test]
fn serve_get_state_reports_pending_tool_ids_while_a_tool_is_running() {
    use std::time::Duration;

    // B-L1 pi-parity gap (fixed): pi's `agent.state.pendingToolCalls` (a live, in-process reactive
    // set) has no RPC equivalent — a client had to reconstruct "which calls are still in flight"
    // itself from the raw event stream. `get_state` now mirrors it directly, live, mid-run.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let turn1 = turn_tool_use(
        "toolu_pending",
        "bash",
        &json!({ "command": "sleep 0.5" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(150)); // mid-`sleep 0.5`, the bash call is in flight

    writeln!(stdin, "{}", json!({ "type": "get_state", "id": "mid" })).unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");

    let mid_resp = frames
        .iter()
        .find(|f| f["command"] == "get_state" && f["id"] == "mid")
        .unwrap_or_else(|| panic!("no mid-run get_state response: {frames:#?}"));
    assert_eq!(
        mid_resp["data"]["pending_tool_ids"],
        json!(["toolu_pending"]),
        "the running bash call must be reported as pending: {mid_resp:#?}"
    );

    // The whole run has now finished (both turns) on this same live process — a fresh `get_state`
    // proves `tool_ended` actually cleared the id, not just that `tool_started` populated it.
    writeln!(stdin, "{}", json!({ "type": "get_state", "id": "after" })).unwrap();
    stdin.flush().unwrap();
    let after_frames = read_until_response(&mut stdout, "get_state");
    let after_resp = after_frames.last().unwrap();
    assert_eq!(after_resp["id"], "after");
    assert_eq!(
        after_resp["data"]["pending_tool_ids"],
        json!([]),
        "the completed call must no longer be pending: {after_resp:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_state_reports_runtime_settings_and_queue_depth() {
    // `get_state` must carry the runtime-mutable settings (thinking level, auto-compaction, auto-retry,
    // queue mode) and the current queue depth — pi's `get_state` carries the same shape
    // (`thinkingLevel`/`autoCompactionEnabled`/`steeringMode`/`followUpMode`/`pendingMessageCount`), and
    // a client shouldn't need a separate round trip (or its own copy of the defaults) to render them.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // Defaults, nothing queued yet.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["thinking_level"], "off", "{data:#?}");
    assert_eq!(data["auto_compaction"], true, "{data:#?}");
    assert_eq!(data["auto_retry"], true, "{data:#?}");
    assert_eq!(data["steering_mode"], "one_at_a_time", "{data:#?}");
    assert_eq!(data["follow_up_mode"], "one_at_a_time", "{data:#?}");
    assert_eq!(data["pending_messages"], 0, "{data:#?}");

    // Queue two follow-ups (idle `follow_up`, no prompt in flight) and flip auto_compaction/queue_mode;
    // `get_state` must reflect all of it without any prompt having run.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "message": "first" })
    )
    .unwrap();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "follow_up", "message": "second" })
    )
    .unwrap();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_auto_compaction", "enabled": false })
    )
    .unwrap();
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(data["auto_compaction"], false, "{data:#?}");
    assert_eq!(
        data["pending_messages"], 2,
        "two queued follow-ups: {data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_reports_cwd_stale_false_for_a_freshly_created_session() {
    // A session `serve` creates itself always records the actual current directory, which obviously
    // still exists — `cwd_stale` must read false everywhere it's surfaced.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    assert_eq!(ready["cwd_stale"], false, "got: {ready:#?}");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let state = frames.last().unwrap();
    assert_eq!(state["data"]["cwd_stale"], false, "got: {state:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_reports_cwd_stale_true_when_the_recorded_directory_no_longer_exists() {
    // File-mode persistence (`--session-file`) reattaches to whatever session is on disk without any
    // cwd-matching filter (unlike repo mode's automatic reattach). Hand-write a header recording a
    // directory that doesn't exist, simulating a project that was since moved or deleted, and confirm
    // `serve` surfaces the mismatch rather than silently proceeding as if nothing changed.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let header = json!({
        "type": "session",
        "id": "stale-cwd-session",
        "created_at": 1,
        "cwd": "/definitely/does/not/exist/beyond-ai-agent-test-fixture",
        "model": "claude-test",
    });
    std::fs::write(&session_file, format!("{header}\n")).unwrap();

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file.to_string_lossy())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    assert_eq!(ready["session_id"], "stale-cwd-session");
    assert_eq!(ready["cwd_stale"], true, "got: {ready:#?}");

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let state = frames.last().unwrap();
    assert_eq!(state["data"]["cwd_stale"], true, "got: {state:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_session_reports_cwd_stale_for_the_newly_active_session() {
    // Repo mode's automatic reattach filters by cwd, so a mismatched session can only be reached by an
    // explicit `switch_session` — plant one directly in the repo directory (matching its
    // `<created_at>_<id>.jsonl` naming convention) and confirm switching to it surfaces the mismatch
    // immediately in the `switch_session` response, not just on a later `get_state` poll.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions");
    std::fs::create_dir_all(&session_dir).unwrap();
    let stale_header = json!({
        "type": "session",
        "id": "stale-target",
        "created_at": 1,
        "cwd": "/definitely/does/not/exist/beyond-ai-agent-test-fixture",
        "model": "claude-test",
    });
    std::fs::write(
        session_dir.join("1_stale-target.jsonl"),
        format!("{stale_header}\n"),
    )
    .unwrap();

    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir.to_string_lossy())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // The freshly (auto-)created active session must not be stale.
    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();
    let ready: Value = serde_json::from_str(ready.trim()).unwrap();
    assert_eq!(ready["cwd_stale"], false, "got: {ready:#?}");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": "stale-target" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_session");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(response["data"]["cwd_stale"], true, "got: {response:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_switch_session_restores_the_reopened_sessions_own_model() {
    // pi-parity fix: `current_model` used to be seeded once from the server's own `--model` startup
    // flag and never reconciled with the session actually being switched to — reattaching to a session
    // last driven on a different model silently kept using the server's startup model instead, with no
    // warning. `switch_branch` already restored this correctly within one session's tree; this proves
    // `switch_session` (a *different* session's own store entirely) now does too.
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions");
    std::fs::create_dir_all(&session_dir).unwrap();
    let other_header = json!({
        "type": "session",
        "id": "other-model-target",
        "created_at": 1,
        "cwd": "/definitely/does/not/exist/beyond-ai-agent-test-fixture",
        "model": "claude-test-restored",
    });
    std::fs::write(
        session_dir.join("1_other-model-target.jsonl"),
        format!("{other_header}\n"),
    )
    .unwrap();

    // The server starts on `claude-test` (see `serve_dir_cmd`) — deliberately different from the
    // planted session's own `claude-test-restored`, so a successful restore is unambiguous.
    let (base, _bodies) = spawn_model_server(vec![]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir.to_string_lossy())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut ready = String::new();
    stdout.read_line(&mut ready).unwrap();

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": "other-model-target" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "switch_session");
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "got: {response:#?}");
    assert_eq!(
        response["data"]["model"], "claude-test-restored",
        "switch_session must restore the target session's own model, not keep the server's startup \
         model: {response:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let response = frames.last().unwrap();
    assert_eq!(
        response["data"]["model"], "claude-test-restored",
        "the restored model must also be what a subsequent turn would actually use, not just what \
         the switch response reported: {response:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_get_session_stats_reports_context_usage_after_a_real_turn() {
    // Companion e2e proof for `session_stats`'s `context_usage` field (unit-tested directly in
    // `serve.rs`'s own test module) — end-to-end through the real RPC surface: null before any turn,
    // populated with a plausible `percent` after one.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi there")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "get_session_stats" })).unwrap();
    stdin.flush().unwrap();
    let before = read_until_response(&mut stdout, "get_session_stats");
    assert_eq!(
        before.last().unwrap()["data"]["context_usage"],
        Value::Null,
        "no turn has run yet: {before:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hello" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_session_stats" })).unwrap();
    stdin.flush().unwrap();
    let after = read_until_response(&mut stdout, "get_session_stats");
    let usage = &after.last().unwrap()["data"]["context_usage"];
    assert!(usage["tokens"].as_u64().unwrap() > 0, "got: {usage:#?}");
    assert!(
        usage["context_window"].as_u64().unwrap() > 0,
        "got: {usage:#?}"
    );
    assert!(usage["percent"].as_f64().unwrap() >= 0.0, "got: {usage:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_session_name_and_get_last_assistant_text() {
    // pi: rpc.test.ts "should set and get session name" / "get_last_assistant_text" — both were
    // implemented but had zero e2e coverage: `set_session_name` persists a `session_info` entry
    // reflected by `get_state`, and `get_last_assistant_text` reports the most recent assistant reply.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("the actual reply")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // No name set yet, no assistant reply yet.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let initial = read_until_response(&mut stdout, "get_state");
    assert!(
        initial.last().unwrap()["data"]
            .get("title")
            .is_none_or(Value::is_null),
        "got: {initial:#?}"
    );
    writeln!(stdin, "{}", json!({ "type": "get_last_assistant_text" })).unwrap();
    stdin.flush().unwrap();
    let none_yet = read_until_response(&mut stdout, "get_last_assistant_text");
    let text = &none_yet.last().unwrap()["data"]["text"];
    assert!(
        text.as_str().unwrap_or_default().is_empty() || text.is_null(),
        "got: {none_yet:#?}"
    );

    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_session_name", "title": "my-test-session" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let set_resp = read_until_response(&mut stdout, "set_session_name");
    assert_eq!(
        set_resp.last().unwrap()["success"],
        true,
        "got: {set_resp:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let after_name = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        after_name.last().unwrap()["data"]["title"],
        "my-test-session",
        "got: {after_name:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(stdin, "{}", json!({ "type": "get_last_assistant_text" })).unwrap();
    stdin.flush().unwrap();
    let after_reply = read_until_response(&mut stdout, "get_last_assistant_text");
    assert!(
        after_reply.last().unwrap()["data"]["text"]
            .as_str()
            .unwrap_or_default()
            .contains("the actual reply"),
        "got: {after_reply:#?}"
    );

    // The name survives on disk too (a single `session_info` entry, not lost on the next reattach).
    let on_disk = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        on_disk.contains("my-test-session"),
        "session name must be persisted: {on_disk}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_set_session_name_strips_newlines_and_pushes_a_session_info_changed_frame() {
    // F-M4 (pi: 5996-session-name-newlines.test.ts:14-22): newline sanitization was only proven at the
    // storage-unit level (`session_store.rs`'s `set_title_strips_newlines`), never through the actual
    // `set_session_name` RPC command + `get_state` readback.
    //
    // F-M1 (pi: 3686-session-name-event.test.ts, `session_info_changed` — `rpc-mode.ts:632-639`): the
    // rename response previously carried no `data` at all, and nothing told a client the *sanitized*
    // final name without a follow-up `get_state`. Both are proven together here: the same round trip
    // exercises the sanitization and the new `data`/unsolicited-frame push.
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
        json!({ "id": "n1", "type": "set_session_name", "title": "hello\nworld\r\nagain" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_session_name");

    // The unsolicited `session_info_changed` push frame carries the sanitized name, correlated by id.
    let pushed = frames
        .iter()
        .find(|f| f["type"] == "session_info_changed")
        .unwrap_or_else(|| panic!("expected a session_info_changed frame: {frames:#?}"));
    assert_eq!(pushed["id"], "n1", "got: {pushed:#?}");
    assert_eq!(pushed["title"], "hello world again", "got: {pushed:#?}");

    // The response itself also carries the final sanitized name — no second round trip needed.
    let resp = frames.last().unwrap();
    assert_eq!(resp["success"], true, "got: {resp:#?}");
    assert_eq!(resp["data"]["title"], "hello world again", "got: {resp:#?}");

    // And `get_state` reads back the same sanitized value, with no embedded newlines.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let state = read_until_response(&mut stdout, "get_state");
    let title = state.last().unwrap()["data"]["title"].as_str().unwrap();
    assert_eq!(title, "hello world again", "got: {state:#?}");
    assert!(
        !title.contains('\n') && !title.contains('\r'),
        "got: {title:?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn get_state_reports_session_file_and_is_streaming_idle_vs_mid_run() {
    // LOW pi-parity gap (fixed): `get_state` used to omit pi's `sessionFile`/`isStreaming`/
    // `isCompacting` fields entirely. This proves `session_file` matches the real `--session-file`
    // path in both the idle and mid-run handlers (architecturally distinct code paths — one keyed off
    // `live_stats`, the other off `session_stats`), and that `is_streaming` correctly flips true only
    // while a `prompt` genuinely has a turn in flight. `is_compacting` isn't forced here (doing so
    // reliably would need a real network delay the shared mock server doesn't support) — its own
    // event-driven state machine (`CompactionStart` before `Compacted`, with the right `reason`) is
    // proven directly at the `agent-core` level instead; this test still confirms it reads back
    // `false` in both idle and this (non-compacting) mid-run case, i.e. it never falsely reports
    // `true` for an ordinary run.
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let turn1 = turn_tool_use(
        "toolu_gs",
        "bash",
        &json!({ "command": "sleep 1" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // Idle, before any prompt: `is_streaming`/`is_compacting` must both already read `false`, and
    // `session_file` must already resolve to the real on-disk path (`Persistence::open` created it at
    // startup, before any turn ever ran).
    let mut child = serve_cmd(bin, &base, &session_file).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let idle_frames = read_until_response(&mut stdout, "get_state");
    let idle_state = &idle_frames.last().unwrap()["data"];
    assert_eq!(idle_state["is_streaming"], false, "got: {idle_state:#?}");
    assert_eq!(idle_state["is_compacting"], false, "got: {idle_state:#?}");
    assert_eq!(
        idle_state["session_file"].as_str(),
        Some(session_file.as_str()),
        "got: {idle_state:#?}"
    );

    // Mid-run: the sleeping tool call keeps a turn in flight long enough to query `get_state` from
    // the busy-loop's own (architecturally distinct) handler.
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "go" })).unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();

    let frames = read_until_response(&mut stdout, "prompt");
    let mid_run_state = &frames
        .iter()
        .find(|f| f["type"] == "response" && f["command"] == "get_state")
        .expect("a get_state response while the prompt is in flight")["data"];
    assert_eq!(
        mid_run_state["is_streaming"], true,
        "got: {mid_run_state:#?}"
    );
    assert_eq!(
        mid_run_state["is_compacting"], false,
        "got: {mid_run_state:#?}"
    );
    assert_eq!(
        mid_run_state["session_file"].as_str(),
        Some(session_file.as_str()),
        "got: {mid_run_state:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}
