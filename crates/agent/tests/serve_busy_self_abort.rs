//! `serve` e2e: `compact`/`switch_session`/`fork`/`clone`/`new_session` received while a `prompt` is
//! streaming self-abort-and-proceed instead of being rejected as busy (pi-parity Task 4, serve pass
//! 19) — matching pi's real product, which has no busy/idle gating at all for these: `compact()`/
//! `switchSession`/`newSession`/`fork` all call `abort()` unconditionally before replacing the session,
//! with no check for whether a prompt is currently streaming. A client that used to have to send
//! `abort`, wait for its response, then retry the original command now gets the same net effect in one
//! round trip.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::process::{ChildStdin, ChildStdout};
use std::time::Duration;

use common::{
    ChildGuard, SpawnGuarded, read_until_response, serve_dir_cmd,
    spawn_model_server_with_stalled_response, turn_text,
};
use serde_json::{Value, json};

/// Spawn `serve` (repo/multi-session mode) against `base`, and run one quick "warmup" prompt to
/// completion — every test below needs at least one committed turn to exist before it starts the
/// *second* (stalled, mid-run-interrupted) one, so `base`'s model server must answer this first
/// request instantly (the caller's own `fast` list should have exactly one entry for it).
fn spawn_and_warm_up(
    bin: &str,
    base: &str,
    session_dir: &str,
) -> (ChildGuard, ChildStdin, BufReader<ChildStdout>) {
    let mut child = serve_dir_cmd(bin, base, session_dir).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "warmup" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    (child, stdin, stdout)
}

/// Start a second prompt (against the stalled model response `spawn_model_server_with_stalled_response`
/// leaves waiting), give it time to actually reach the server, then send `busy_cmd` while it's
/// provably still streaming. Returns `(prompt's own terminal response, busy_cmd's own response)`, in
/// the order `serve` actually emits them (the interrupted prompt's terminal response first, then the
/// self-abort-and-proceeded command's — see `serve.rs`'s `pending_deferred` doc comment for why).
fn run_busy_then(
    stdin: &mut ChildStdin,
    stdout: &mut BufReader<ChildStdout>,
    busy_cmd: &Value,
) -> (Value, Value) {
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "id": "p1", "message": "go" })
    )
    .unwrap();
    stdin.flush().unwrap();
    // Give the run time to actually reach the stalled server (it writes a partial body immediately on
    // accept, well before its 5s stall completes) before sending the busy command.
    std::thread::sleep(Duration::from_millis(300));
    writeln!(stdin, "{busy_cmd}").unwrap();
    stdin.flush().unwrap();

    let prompt_frames = read_until_response(stdout, "prompt");
    let prompt_resp = prompt_frames
        .into_iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .unwrap();

    let busy_type = busy_cmd["type"].as_str().unwrap();
    let busy_frames = read_until_response(stdout, busy_type);
    let busy_resp = busy_frames
        .into_iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == busy_type)
        .unwrap();

    (prompt_resp, busy_resp)
}

#[test]
fn serve_busy_compact_self_aborts_and_proceeds_in_one_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    // Deliberately no warmup turn here (unlike the other 4 tests below): with only the interrupted
    // "go" prompt's own user message persisted, `compaction::find_split_cut` (which needs at least 4
    // messages to find anything worth cutting) has nothing to do, so `compact` resolves through its
    // own `TooSmall` short-circuit with no model call of its own — deterministic, and avoids the mock
    // server needing to answer a second, genuinely-summarizing request that would otherwise race the
    // busy-time abort. `fast: Vec::new()` — the "go" prompt itself is the very first request, so it's
    // the one that stalls.
    let base =
        spawn_model_server_with_stalled_response(Vec::new(), Duration::from_secs(5), Vec::new());
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_dir_cmd(bin, &base, &session_dir).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let (prompt_resp, compact_resp) = run_busy_then(
        &mut stdin,
        &mut stdout,
        &json!({ "type": "compact", "id": "c1" }),
    );

    assert_eq!(
        prompt_resp["success"], false,
        "the interrupted prompt reports failure, exactly like an explicit abort would: {prompt_resp:#?}"
    );
    assert_eq!(
        compact_resp["success"], true,
        "compact must complete successfully in the same round trip that interrupted the prompt: {compact_resp:#?}"
    );
    assert_eq!(
        compact_resp["data"]["compacted"], false,
        "too little content exists yet to actually compact — this proves the busy-then-proceed round \
         trip itself, not compaction's own threshold logic: {compact_resp:#?}"
    );

    // The server must still be usable afterward — proves the abort-then-proceed sequence left the
    // control loop in a clean, continuable state, not stuck mid-run.
    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(frames.last().unwrap()["data"]["is_streaming"], false);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_busy_new_session_self_aborts_and_proceeds_in_one_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let base = spawn_model_server_with_stalled_response(
        vec![turn_text("warmup")],
        Duration::from_secs(5),
        Vec::new(),
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let (mut child, mut stdin, mut stdout) = spawn_and_warm_up(bin, &base, &session_dir);

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let original_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    let (prompt_resp, new_session_resp) = run_busy_then(
        &mut stdin,
        &mut stdout,
        &json!({ "type": "new_session", "id": "n1" }),
    );

    assert_eq!(prompt_resp["success"], false, "{prompt_resp:#?}");
    assert_eq!(new_session_resp["success"], true, "{new_session_resp:#?}");
    let new_id = new_session_resp["data"]["session_id"].as_str().unwrap();
    assert_ne!(
        new_id, original_id,
        "new_session must actually land on a fresh session id: {new_session_resp:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["session_id"],
        new_id,
        "the process must actually be on the new session afterward: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_busy_switch_session_self_aborts_and_proceeds_in_one_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    // 2 fast (instant) responses: the initial warmup, and a second session's own warmup — the busy
    // "go" prompt below is the *third* model request, so it lands on the stalled one.
    let base = spawn_model_server_with_stalled_response(
        vec![turn_text("warmup-a"), turn_text("warmup-b")],
        Duration::from_secs(5),
        Vec::new(),
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let (mut child, mut stdin, mut stdout) = spawn_and_warm_up(bin, &base, &session_dir);

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let session_a = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Create session B (with its own warmup turn), then switch back to A so the busy prompt below
    // runs there — `switch_session`'s own busy-time target is B.
    writeln!(stdin, "{}", json!({ "type": "new_session" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "new_session");
    let session_b = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "warmup-b" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    writeln!(
        stdin,
        "{}",
        json!({ "type": "switch_session", "session_id": session_a })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "switch_session");

    let (prompt_resp, switch_resp) = run_busy_then(
        &mut stdin,
        &mut stdout,
        &json!({ "type": "switch_session", "id": "s1", "session_id": session_b }),
    );

    assert_eq!(prompt_resp["success"], false, "{prompt_resp:#?}");
    assert_eq!(switch_resp["success"], true, "{switch_resp:#?}");
    assert_eq!(
        switch_resp["data"]["session_id"], session_b,
        "{switch_resp:#?}"
    );
    assert!(
        switch_resp["data"]["reasoning_effort"].is_string(),
        "switch_session's busy-path response must carry the same shape as its idle-path one (Task 3): \
         {switch_resp:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["session_id"],
        session_b,
        "the process must actually be on session B afterward: {frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_busy_fork_self_aborts_and_proceeds_in_one_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let base = spawn_model_server_with_stalled_response(
        vec![turn_text("warmup")],
        Duration::from_secs(5),
        Vec::new(),
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let (mut child, mut stdin, mut stdout) = spawn_and_warm_up(bin, &base, &session_dir);

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let original_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    let (prompt_resp, fork_resp) = run_busy_then(
        &mut stdin,
        &mut stdout,
        &json!({ "type": "fork", "id": "f1" }),
    );

    assert_eq!(prompt_resp["success"], false, "{prompt_resp:#?}");
    assert_eq!(fork_resp["success"], true, "{fork_resp:#?}");
    let fork_id = fork_resp["data"]["session_id"].as_str().unwrap();
    assert_ne!(
        fork_id, original_id,
        "fork must actually land on a new session id: {fork_resp:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["session_id"],
        fork_id,
        "{frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_busy_clone_self_aborts_and_proceeds_in_one_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let session_dir = dir.path().join("sessions").to_string_lossy().into_owned();
    let base = spawn_model_server_with_stalled_response(
        vec![turn_text("warmup")],
        Duration::from_secs(5),
        Vec::new(),
    );
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let (mut child, mut stdin, mut stdout) = spawn_and_warm_up(bin, &base, &session_dir);

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    let original_id = frames.last().unwrap()["data"]["session_id"]
        .as_str()
        .unwrap()
        .to_string();

    let (prompt_resp, clone_resp) = run_busy_then(
        &mut stdin,
        &mut stdout,
        &json!({ "type": "clone", "id": "cl1" }),
    );

    assert_eq!(prompt_resp["success"], false, "{prompt_resp:#?}");
    assert_eq!(clone_resp["success"], true, "{clone_resp:#?}");
    let clone_id = clone_resp["data"]["session_id"].as_str().unwrap();
    assert_ne!(
        clone_id, original_id,
        "clone must actually land on a new session id: {clone_resp:#?}"
    );

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "get_state");
    assert_eq!(
        frames.last().unwrap()["data"]["session_id"],
        clone_id,
        "{frames:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}
