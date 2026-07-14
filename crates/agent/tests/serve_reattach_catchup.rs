//! Re-attach catch-up ordering (`serve --listen`).
//!
//! `serve_websocket.rs` already proves a run *survives* a dropped connection and that a new connection
//! to the same `?session_id` re-attaches to it. What it never exercises is the thing a re-attaching
//! client actually needs first: **the history it missed while it was gone**, delivered *before* the live
//! output of the turn still in flight.
//!
//! That is the whole point of re-attach — `serve_ws`'s own docs call re-attaching to a still-running run
//! "the entire point" — and it is exactly where the ordering used to break. Attach registers the
//! connection's sink into the `OutFanout` (live frames start flowing at once), while catch-up was a
//! *separate*, client-initiated round trip issued afterwards. Frames carry no sequence number, so
//! delivery order is the only order a client has: whatever arrives before the backlog is rendered before
//! it. Catch-up is now seeded server-side at attach, under the same lock `broadcast` takes.
//!
//! Every test here re-attaches to a session with a run genuinely in flight (a real `bash sleep`, so the
//! window is not a race against a fast local round trip).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Child, Command, Stdio};

use common::{
    ISOLATED_HOME, free_port, spawn_model_server, turn_text, turn_tool_use, wait_for_port,
    ws_connect, ws_next_frame, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Marker text the model commits in turn 1 — the history a re-attaching client must get back.
const COMMITTED: &str = "COMMITTED_BEFORE_DROP";

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

/// Drive a session to the state both tests need: one **committed** turn in history (containing
/// [`COMMITTED`]), then a second prompt parked mid-run inside a real `bash sleep`, then the driving
/// connection dropped. Returns the still-running child and the session id, with the run in flight.
///
/// The mock model answers three requests: turn 1 commits `COMMITTED`; the second prompt's first turn
/// calls `bash sleep 5` (the in-flight window); its second turn ends the run.
async fn session_with_history_and_a_run_in_flight(
    dir: &str,
    port: u16,
    sid: &str,
) -> (Child, String) {
    let (base, _requests) = spawn_model_server(vec![
        turn_text(COMMITTED),
        turn_tool_use("t1", "bash", &json!({ "command": "sleep 5" }).to_string()),
        turn_text("AFTER_THE_SLEEP"),
    ]);
    let child = serve_ws_child(&base, dir, port);
    wait_for_port(port);

    let mut ws1 = ws_connect(port, Some(sid)).await;

    // Turn 1 runs to completion, so `COMMITTED` is genuinely in the session's history — this is the
    // backlog the reconnecting client is owed.
    ws_send(
        &mut ws1,
        json!({ "type": "prompt", "id": "p1", "message": "first" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws1, "prompt").await;
    let done = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .expect("the first prompt should complete");
    assert_eq!(done["success"], true, "first prompt should succeed: {done}");

    // Second prompt: park it mid-run. `tool_start{bash}` proves the run reached the 5s sleep, so the
    // drop below lands squarely inside a live run rather than racing it.
    ws_send(
        &mut ws1,
        json!({ "type": "prompt", "id": "p2", "message": "second" }),
    )
    .await;
    let mut in_flight = false;
    while let Some(f) = ws_next_frame(&mut ws1).await {
        if f["type"] == "event"
            && f["event"]["kind"] == "tool_start"
            && f["event"]["name"] == "bash"
        {
            in_flight = true;
            break;
        }
    }
    assert!(in_flight, "the second prompt should reach the bash sleep");

    drop(ws1); // the mobile network drops mid-run
    (child, sid.to_string())
}

/// A frame that carries the committed history back to the client — whatever shape it arrives in
/// (a `get_messages` response, or a server-pushed catch-up on attach). Keyed on the marker text so the
/// assertion is about *the client got its history back*, not about one wire spelling of it.
fn carries_history(f: &Value) -> bool {
    f["type"] != "event" && f.to_string().contains(COMMITTED)
}

/// A live frame from the turn still in flight (the run the client re-attached *into*).
fn is_live_event(f: &Value) -> bool {
    f["type"] == "event"
}

/// A client that re-attaches mid-run must be able to fetch the history it missed.
///
/// Regression: `get_messages` used to be absent from the busy loop's accepted-command list, so it was
/// rejected as busy for the entire duration of an in-flight run — the client was told to come back
/// later, having *already* been shown that run's live output. It is now answered from the committed-
/// history snapshot the fanout carries, the same source the attach-time catch-up is seeded from.
#[tokio::test]
async fn reattach_midrun_can_fetch_missed_history() {
    const SID: &str = "reattachcatchup1";
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let (mut child, sid) =
        session_with_history_and_a_run_in_flight(dir.path().to_str().unwrap(), port, SID).await;

    let mut ws2 = ws_connect(port, Some(&sid)).await;
    ws_send(&mut ws2, json!({ "type": "get_messages" })).await;
    let frames = ws_read_until_response(&mut ws2, "get_messages").await;
    let resp = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "get_messages")
        .expect("a get_messages response");

    assert_eq!(
        resp["success"], true,
        "a client re-attaching mid-run must be able to read the history it missed, \
         not be rejected as busy: {resp}"
    );
    assert!(
        resp["data"]["messages"].to_string().contains(COMMITTED),
        "the catch-up must contain the turn committed before the drop: {resp}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// The ordering invariant: on re-attach, the missed history must reach the client **before** the live
/// output of the in-flight turn. Frames carry no sequence number, so delivery order is the only order —
/// a backlog that lands after the live stream is rendered after it, which is the out-of-order symptom.
///
/// Regression: the sink used to be registered at attach (live frames flowing at once) with catch-up left
/// to a client round trip that raced them, so a reconnecting client saw the tail of a turn whose earlier
/// messages it had never seen. The history is now queued on the connection *before* its sink goes live,
/// both under the fanout lock `broadcast` takes — so this holds structurally, not by winning a race.
#[tokio::test]
async fn reattach_delivers_history_before_live_frames() {
    const SID: &str = "reattachcatchup2";
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let (mut child, sid) =
        session_with_history_and_a_run_in_flight(dir.path().to_str().unwrap(), port, SID).await;

    // Re-attach and just *listen*, issuing no command at all. Everything that arrives, arrives because
    // the server sent it — which is the point: catch-up must not be a thing the client has to ask for
    // and then hope wins a race.
    let mut ws2 = ws_connect(port, Some(&sid)).await;

    let mut seen: Vec<Value> = Vec::new();
    while let Some(f) = ws_next_frame(&mut ws2).await {
        let stop = carries_history(&f) || is_live_event(&f);
        seen.push(f);
        if stop {
            break;
        }
    }

    let first = seen
        .last()
        .unwrap_or_else(|| panic!("the re-attached client should receive something: {seen:#?}"));
    assert!(
        carries_history(first),
        "on re-attach the client must receive the history it missed BEFORE any live frame of the \
         in-flight turn; instead the first thing it got was live output of a turn whose earlier \
         messages it has never seen: {first}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// Pins the catch-up's wire shape, so the two tests above can't pass on some *other* frame that merely
/// happens to mention the marker. The frame a re-attaching client is seeded with is a `catchup` whose
/// `data` is the same `{messages, leaf_id}` payload `get_messages` answers with.
#[tokio::test]
async fn reattach_catchup_frame_has_the_get_messages_payload_shape() {
    const SID: &str = "reattachcatchup3";
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let (mut child, sid) =
        session_with_history_and_a_run_in_flight(dir.path().to_str().unwrap(), port, SID).await;

    let mut ws2 = ws_connect(port, Some(&sid)).await;
    let first = ws_next_frame(&mut ws2).await.expect("a first frame");

    assert_eq!(
        first["type"], "catchup",
        "the seeded frame is a catchup: {first}"
    );
    let msgs = first["data"]["messages"]
        .as_array()
        .unwrap_or_else(|| panic!("catchup.data.messages must be an array: {first}"));
    assert!(
        msgs.iter().any(|m| m.to_string().contains(COMMITTED)),
        "the catchup carries the turn committed before the drop: {first}"
    );
    assert!(
        first["data"]["leaf_id"].is_string(),
        "the catchup carries the active tip, like get_messages: {first}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// The other half of "picking a session back up": the turn **in flight** at the moment you attach.
///
/// Committed history alone is not enough. The turn currently streaming has not committed yet, so it is
/// not in the catch-up — and the frames it already emitted went out before this connection existed. A
/// client that re-attaches mid-turn would therefore see that turn from the middle: here, a `tool_end`
/// for a `bash` call whose `tool_start` it never received. It renders a truncated turn, which is the
/// same "the transcript is wrong when I reconnect" symptom the ordering bug caused.
///
/// So the in-flight turn's frames so far are replayed on attach too, after the catch-up and before the
/// live stream resumes — leaving a reconnecting client with byte-identical output to one that never
/// dropped.
#[tokio::test]
async fn reattach_replays_the_in_flight_turn_not_just_committed_history() {
    const SID: &str = "reattachcatchup4";
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    // The driving connection saw `tool_start{bash}` before it dropped (that is how the helper knows the
    // run is in flight) — so that frame is, by construction, one this reconnecting client missed.
    let (mut child, sid) =
        session_with_history_and_a_run_in_flight(dir.path().to_str().unwrap(), port, SID).await;

    let mut ws2 = ws_connect(port, Some(&sid)).await;

    // Read until the bash call ends (or the turn does). A client with a coherent view must have seen the
    // call *start* before it sees it end.
    let mut saw_start = false;
    let mut saw_end = false;
    while let Some(f) = ws_next_frame(&mut ws2).await {
        if f["type"] == "event" && f["event"]["name"] == "bash" {
            match f["event"]["kind"].as_str() {
                Some("tool_start") => saw_start = true,
                Some("tool_end") => {
                    saw_end = true;
                    break;
                }
                _ => {}
            }
        }
        if f["type"] == "response" && f["command"] == "prompt" {
            break;
        }
    }

    assert!(
        saw_end,
        "the reconnecting client should see the in-flight bash call finish"
    );
    assert!(
        saw_start,
        "a client that re-attaches mid-turn must be replayed the frames that turn already emitted: \
         it received `tool_end` for a bash call whose `tool_start` it never saw, so it renders a turn \
         starting from the middle"
    );

    let _ = child.kill();
    let _ = child.wait();
}
