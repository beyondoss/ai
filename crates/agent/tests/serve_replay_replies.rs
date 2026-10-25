//! A reply is the asking client's, not the turn's (`serve --listen`).
//!
//! A connection that attaches mid-turn is replayed the turn's frames so far (see
//! `serve_reattach_catchup.rs`). Replies to other clients' commands were recorded into that replay
//! too, each charged a flat 256 bytes — and a `get_messages` reply is the whole transcript. A healthy
//! client polling `get_messages` during a run grew the session's replay buffer by a transcript per
//! call, past the 4 MiB bound that was meant to hold it, and every later attach was handed answers
//! to questions it never asked.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use common::{
    ISOLATED_HOME, spawn_listening, spawn_model_server, turn_text, turn_tool_use, ws_connect,
    ws_next_frame, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Roughly a megabyte of committed transcript, so each `get_messages` reply is that large.
fn big_transcript() -> String {
    format!("COMMITTED {}", "x".repeat(1024 * 1024))
}

fn is_bash(f: &Value, kind: &str) -> bool {
    f["type"] == "event" && f["event"]["kind"] == kind && f["event"]["name"] == "bash"
}

#[tokio::test]
async fn get_messages_replies_during_a_run_are_not_replayed_to_a_later_attach() {
    const SID: &str = "replayreplies1";
    const CALLS: usize = 6;
    let dir = tempfile::tempdir().unwrap();
    let (base, _requests) = spawn_model_server(vec![
        turn_text(&big_transcript()),
        turn_tool_use("t1", "bash", &json!({ "command": "sleep 8" }).to_string()),
        turn_text("AFTER_THE_SLEEP"),
    ]);
    let (mut child, port) = spawn_listening(
        crate::common::agent_cmd()
            .args([
                "serve",
                "--gateway-url",
                &base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                dir.path().to_str().unwrap(),
            ])
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    );

    let mut ws1 = ws_connect(port, Some(SID)).await;
    ws_send(
        &mut ws1,
        json!({ "type": "prompt", "id": "p1", "message": "first" }),
    )
    .await;
    let done = ws_read_until_response(&mut ws1, "prompt").await;
    assert_eq!(done.last().unwrap()["success"], true, "{:?}", done.last());

    // A second prompt, parked inside a real `bash sleep`.
    ws_send(
        &mut ws1,
        json!({ "type": "prompt", "id": "p2", "message": "second" }),
    )
    .await;
    while let Some(f) = ws_next_frame(&mut ws1).await {
        if is_bash(&f, "tool_start") {
            break;
        }
    }

    // A healthy client polls the transcript during the run, reading every reply.
    for i in 0..CALLS {
        ws_send(
            &mut ws1,
            json!({ "type": "get_messages", "id": format!("g{i}") }),
        )
        .await;
        let frames = ws_read_until_response(&mut ws1, "get_messages").await;
        let reply = frames.last().unwrap();
        assert_eq!(reply["success"], true);
        assert!(reply["data"]["messages"].to_string().contains("COMMITTED"));
    }

    // Another connection attaches mid-turn and is caught up: history, then the turn so far.
    let mut ws2 = ws_connect(port, Some(SID)).await;
    let mut seen = Vec::new();
    while let Some(f) = ws_next_frame(&mut ws2).await {
        let end = is_bash(&f, "tool_end");
        seen.push(f);
        if end {
            break;
        }
    }
    let catchup = seen
        .iter()
        .find(|f| f["type"] == "catchup")
        .expect("a catch-up on attach");
    assert_eq!(catchup["turn_in_flight"], true, "{catchup}");
    assert_eq!(
        catchup["turn_truncated"], false,
        "the turn's own frames fit the replay budget"
    );
    assert!(
        seen.iter().any(|f| is_bash(f, "tool_start")),
        "the turn so far is replayed"
    );
    let replayed_replies = seen
        .iter()
        .filter(|f| f["type"] == "response" && f["command"] == "get_messages")
        .count();
    assert_eq!(
        replayed_replies, 0,
        "another client's get_messages replies are not part of the turn replayed on attach"
    );

    // The polling client was never dropped for falling behind: it sees the call finish too.
    let mut finished = false;
    while let Some(f) = ws_next_frame(&mut ws1).await {
        if is_bash(&f, "tool_end") {
            finished = true;
            break;
        }
    }
    assert!(finished, "the polling client stays attached");

    let _ = child.kill();
    let _ = child.wait();
}
