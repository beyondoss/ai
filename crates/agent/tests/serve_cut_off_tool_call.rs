//! `serve` e2e: a tool call the output limit cut off is announced as `not_run` on its `tool_start`
//! frame, and still paired with an error `tool_end`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;

use common::{SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, sse, turn_text};
use serde_json::json;

/// A reply whose `write` call was cut off by `max_tokens` inside its `content`.
fn turn_cut_off_tool_call() -> String {
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": "toolu_cut", "name": "write", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "{\"path\":\"a.txt\",\"content\":\"hello wor" } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "max_tokens" }, "usage": { "output_tokens": 8 } }),
        json!({ "type": "message_stop" }),
    ])
}

#[test]
fn serve_marks_a_cut_off_tool_call_not_run_on_its_tool_start() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_cut_off_tool_call(), turn_text("resent")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = serve_cmd(bin, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "write it" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");

    let kinds = |kind: &str| {
        frames
            .iter()
            .filter(|f| f["type"] == "event" && f["event"]["kind"] == kind)
            .map(|f| f["event"].clone())
            .collect::<Vec<_>>()
    };
    let starts = kinds("tool_start");
    assert_eq!(starts.len(), 1, "{frames:#?}");
    assert_eq!(starts[0]["id"], "toolu_cut");
    assert_eq!(starts[0]["not_run"], true, "{}", starts[0]);
    let ends = kinds("tool_end");
    assert_eq!(ends.len(), 1, "start/end stay paired: {frames:#?}");
    assert_eq!(ends[0]["id"], "toolu_cut");
    assert_eq!(ends[0]["is_error"], true);
    assert!(!dir.path().join("a.txt").exists());

    drop(stdin);
    child.wait().unwrap();
}
