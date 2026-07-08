//! Integration tests for the WebSocket transport (`serve --listen`). Each test spawns the real
//! `beyond-ai-agent` binary in `--listen` mode against the mock Anthropic-SSE gateway
//! (`common::spawn_model_server`), then drives it with a real `tokio-tungstenite` client. The JSON
//! control protocol is byte-identical to stdio mode, so these assert the *transport* — connect,
//! command → frames, and (the point of the feature) that a session survives a dropped connection and a
//! reconnecting client re-attaches to the same still-running run.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{
    ISOLATED_HOME, free_port, spawn_model_server, turn_text, turn_tool_use, wait_for_port,
    ws_connect, ws_next_frame, ws_read_until_response, ws_send,
};
use serde_json::json;

/// Spawn a `serve --listen 127.0.0.1:<port>` child against `base` (the mock gateway), persisting
/// per-session files under `session_dir`. In `--listen` mode stdio is unused, so null it.
fn serve_ws_child(base: &str, session_dir: &str, port: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir,
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn serve --listen")
}

/// (i) DoD: a full session over the socket — `get_state` answers with a `session_id`, and a `prompt`
/// streams `ack` → ≥1 `event` → `response{success:true}` in that order, byte-identical to stdio.
#[tokio::test]
async fn ws_get_state_then_prompt_streams_ack_event_response() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hello there")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let mut ws = ws_connect(port, None).await;

    ws_send(&mut ws, json!({ "type": "get_state" })).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    let state = frames.last().expect("a get_state response");
    assert_eq!(state["type"], "response");
    assert_eq!(state["success"], true, "get_state should succeed: {state}");
    assert!(
        state["data"]["session_id"].as_str().is_some(),
        "get_state must report a session_id: {state}"
    );

    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "hi" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;

    let ack_pos = frames
        .iter()
        .position(|f| f["type"] == "ack" && f["command"] == "prompt")
        .unwrap_or_else(|| panic!("no ack frame: {frames:#?}"));
    assert_eq!(frames[ack_pos]["id"], "p1", "ack echoes the client id");
    let event_pos = frames
        .iter()
        .position(|f| f["type"] == "event")
        .unwrap_or_else(|| panic!("no event frame: {frames:#?}"));
    assert!(ack_pos < event_pos, "ack must precede the first event");
    let resp = frames.last().unwrap();
    assert_eq!(resp["type"], "response");
    assert_eq!(resp["success"], true, "the prompt should succeed: {resp}");

    let _ = child.kill();
    let _ = child.wait();
}

/// (ii) The option-C proof: a run keeps going after its connection drops, and a NEW connection to the
/// same `?session_id` re-attaches to the still-running run — receiving its completion even though it
/// never issued the prompt. The in-flight window is a real `bash sleep` (turn 1 calls the tool; the
/// run is provably mid-flight while it sleeps), so the drop lands squarely inside a live run.
#[tokio::test]
async fn ws_run_survives_dropped_connection_and_reattaches() {
    const SID: &str = "reattachsession1";
    // Turn 1: the model calls `bash sleep 2` (a real 2s in-flight window). Turn 2: it replies "alldone".
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &json!({ "command": "sleep 2" }).to_string()),
        turn_text("alldone"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    // Connection #1 starts the prompt and waits until the run is provably in flight — the `bash`
    // tool_start event means turn 1 finished and the 2s sleep is now running — then drops mid-run.
    let mut ws1 = ws_connect(port, Some(SID)).await;
    ws_send(
        &mut ws1,
        json!({ "type": "prompt", "id": "p1", "message": "go" }),
    )
    .await;
    let mut running = false;
    while let Some(f) = ws_next_frame(&mut ws1).await {
        if f["type"] == "event"
            && f["event"]["kind"] == "tool_start"
            && f["event"]["name"] == "bash"
        {
            running = true;
            break;
        }
    }
    assert!(
        running,
        "connection #1 should see the bash tool start (run in flight)"
    );
    drop(ws1); // simulate a mobile network drop mid-run

    // Connection #2 re-attaches to the same session id. The run is still in flight (bash sleeping).
    let mut ws2 = ws_connect(port, Some(SID)).await;
    ws_send(&mut ws2, json!({ "type": "get_state" })).await;
    let frames = ws_read_until_response(&mut ws2, "get_state").await;
    let state = frames.last().expect("a get_state response");
    assert_eq!(
        state["data"]["session_id"], SID,
        "the reconnecting client sees the same session id: {state}"
    );

    // The run completes on the (now second) connection — proof it was never aborted by the drop. The
    // prompt's terminal `response` arrives here even though ws2 never issued the prompt.
    let frames = ws_read_until_response(&mut ws2, "prompt").await;
    let resp = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .unwrap_or_else(|| panic!("the in-flight prompt should complete on ws2: {frames:#?}"));
    assert_eq!(
        resp["success"], true,
        "the resumed run should succeed: {resp}"
    );

    // The completed assistant turn is committed to history.
    ws_send(&mut ws2, json!({ "type": "get_messages" })).await;
    let frames = ws_read_until_response(&mut ws2, "get_messages").await;
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("alldone"),
        "the run that survived the drop should have committed its assistant reply: {dump}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// (iii) Multi-attach: two connections on ONE session both receive the live stream, and either can
/// drive. A prompt sent on connection A streams to BOTH A and B; a prompt sent on B streams to both
/// too. No eviction — this is the phone + TUI "watch/drive one session together" case.
#[tokio::test]
async fn ws_two_connections_both_receive_the_stream() {
    const SID: &str = "sharedsession1";
    // One turn per prompt (one driven from each connection).
    let (base, _requests) =
        spawn_model_server(vec![turn_text("firstreply"), turn_text("secondreply")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    // Both connect to the SAME session id — they coexist (no eviction).
    let mut ws1 = ws_connect(port, Some(SID)).await;
    let mut ws2 = ws_connect(port, Some(SID)).await;

    // Drive from ws1 → the run must stream to BOTH connections.
    ws_send(
        &mut ws1,
        json!({ "type": "prompt", "id": "p1", "message": "one" }),
    )
    .await;
    let f1 = ws_read_until_response(&mut ws1, "prompt").await;
    let f2 = ws_read_until_response(&mut ws2, "prompt").await; // ws2 receives the broadcast
    for (who, frames) in [("ws1", &f1), ("ws2", &f2)] {
        assert!(
            frames.iter().any(|f| f["type"] == "event"),
            "{who} must receive the streamed events: {frames:#?}"
        );
        let resp = frames
            .iter()
            .rev()
            .find(|f| f["type"] == "response" && f["command"] == "prompt")
            .unwrap_or_else(|| panic!("{who} must receive the prompt response: {frames:#?}"));
        assert_eq!(resp["success"], true, "{who} prompt: {resp}");
    }
    assert!(
        f2.iter()
            .map(|f| f.to_string())
            .collect::<String>()
            .contains("firstreply"),
        "ws2 must see the reply streamed live even though ws1 drove it"
    );

    // Drive from ws2 → both see it too (shared input; neither connection was evicted).
    ws_send(
        &mut ws2,
        json!({ "type": "prompt", "id": "p2", "message": "two" }),
    )
    .await;
    let g2 = ws_read_until_response(&mut ws2, "prompt").await;
    let g1 = ws_read_until_response(&mut ws1, "prompt").await;
    assert_eq!(
        g2.last().unwrap()["success"],
        true,
        "ws2 (driver of the 2nd prompt): {:?}",
        g2.last()
    );
    assert_eq!(
        g1.last().unwrap()["success"],
        true,
        "ws1 must still be attached and see the run ws2 drove"
    );
    assert!(
        g1.iter()
            .map(|f| f.to_string())
            .collect::<String>()
            .contains("secondreply"),
        "ws1 must see ws2's driven reply streamed live"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// Poll for the child to exit within `dur`, without blocking the async runtime. Returns whether it did.
async fn child_exited_within(child: &mut Child, dur: Duration) -> bool {
    let start = std::time::Instant::now();
    loop {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return true;
        }
        if start.elapsed() > dur {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// (iv) Multi-client: two *distinct* session ids run concurrently in one process, each on its own
/// thread and persisted to its own file, fully isolated. Both runs are in flight at once (each holds a
/// real `bash sleep`), and each session's transcript contains only its own prompt.
#[tokio::test]
async fn ws_distinct_sessions_run_concurrently_and_stay_isolated() {
    // Order-independent across the two sessions: both turn-1 requests (whichever arrives first) get a
    // bash-sleep tool call; both turn-2 requests get the final text. So the global-order mock serves
    // either interleaving correctly.
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use("a1", "bash", &json!({ "command": "sleep 1" }).to_string()),
        turn_tool_use("b1", "bash", &json!({ "command": "sleep 1" }).to_string()),
        turn_text("done"),
        turn_text("done"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let mut a = ws_connect(port, Some("alpha")).await;
    let mut b = ws_connect(port, Some("bravo")).await;
    // Distinct, deterministic per-session markers in the user prompt (independent of model responses).
    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "pa", "message": "ALPHAMARKER please run the sleep" }),
    )
    .await;
    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "BRAVOMARKER please run the sleep" }),
    )
    .await;

    // Both complete (their bash sleeps ran concurrently on separate session threads).
    let fa = ws_read_until_response(&mut a, "prompt").await;
    let fb = ws_read_until_response(&mut b, "prompt").await;
    assert_eq!(
        fa.last().unwrap()["success"],
        true,
        "alpha run: {:?}",
        fa.last()
    );
    assert_eq!(
        fb.last().unwrap()["success"],
        true,
        "bravo run: {:?}",
        fb.last()
    );

    // Isolation: each transcript has its own prompt marker and not the other's.
    ws_send(&mut a, json!({ "type": "get_messages" })).await;
    let da = ws_read_until_response(&mut a, "get_messages")
        .await
        .last()
        .unwrap()["data"]["messages"]
        .to_string();
    ws_send(&mut b, json!({ "type": "get_messages" })).await;
    let db = ws_read_until_response(&mut b, "get_messages")
        .await
        .last()
        .unwrap()["data"]["messages"]
        .to_string();
    assert!(
        da.contains("ALPHAMARKER") && !da.contains("BRAVOMARKER"),
        "session alpha must be isolated: {da}"
    );
    assert!(
        db.contains("BRAVOMARKER") && !db.contains("ALPHAMARKER"),
        "session bravo must be isolated: {db}"
    );

    // Each session persisted to its own file, named by id.
    assert!(
        dir.path().join("alpha.jsonl").exists(),
        "alpha.jsonl should exist"
    );
    assert!(
        dir.path().join("bravo.jsonl").exists(),
        "bravo.jsonl should exist"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// (v) Graceful shutdown: a SIGTERM to a `serve --listen` process with a session mid-run must make it
/// exit promptly (not hang) *and* persist the in-flight session before the process goes away — proving
/// the supervisor drives shutdown deterministically (drops each `input_tx`, waits for the session
/// thread to persist) rather than racing `process::exit`.
#[tokio::test]
async fn ws_sigterm_persists_in_flight_session_and_exits() {
    const SID: &str = "sigtermsession1";
    // Turn 1 holds the run open with a long real sleep, so the run is provably in flight at SIGTERM.
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &json!({ "command": "sleep 5" }).to_string()),
        turn_text("done"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);
    let pid = child.id();

    let mut ws = ws_connect(port, Some(SID)).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "SIGMARKER please run the sleep" }),
    )
    .await;
    let mut running = false;
    while let Some(f) = ws_next_frame(&mut ws).await {
        if f["type"] == "event" && f["event"]["kind"] == "tool_start" {
            running = true;
            break;
        }
    }
    assert!(
        running,
        "the run should be in flight (bash sleeping) before SIGTERM"
    );

    // Deliver SIGTERM to the serve process.
    let killed = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .expect("send SIGTERM");
    assert!(killed.success(), "kill -TERM should succeed");

    // It must exit within the shutdown grace period (cancel the sleep, persist, join, exit) — not hang.
    assert!(
        child_exited_within(&mut child, Duration::from_secs(20)).await,
        "serve --listen must exit promptly on SIGTERM, not hang"
    );

    // The in-flight session was persisted before exit (graceful, not killed mid-write).
    let file = dir.path().join(format!("{SID}.jsonl"));
    assert!(
        file.exists(),
        "the session file should exist after graceful shutdown"
    );
    let content = std::fs::read_to_string(&file).unwrap();
    assert!(
        content.contains("SIGMARKER"),
        "the in-flight prompt must be persisted on graceful shutdown, not lost: {content}"
    );

    let _ = child.kill();
    let _ = child.wait();
}
