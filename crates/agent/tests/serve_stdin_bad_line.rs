//! `serve` over stdio: a line that can't be taken as a command is answered and skipped — it never
//! ends the session.
//!
//! Reading stdin as text treated a line that isn't UTF-8 as the end of input, so one bad byte ended
//! the session with exit 0 and nothing on stderr; and a line was buffered whole however long it was,
//! so one unterminated 300 MB line cost more than that in memory. Now an undecodable line gets a
//! `parse` error, an oversized one (past the same 8 MiB a WebSocket message may be) gets one too and
//! is discarded as it streams past, and the commands either side of them are answered.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;

use common::{BIN, SpawnGuarded, read_until_response, serve_cmd, spawn_model_server};
use serde_json::{Value, json};

/// Frames up to and including the `get_state` response with `id`, and that response.
fn until_get_state(frames: &mut common::Frames, id: &str) -> (Vec<Value>, Value) {
    let got = read_until_response(frames, "get_state");
    let response = got
        .last()
        .filter(|f| f["command"] == "get_state")
        .unwrap_or_else(|| panic!("no get_state response (the session ended?); frames: {got:?}"))
        .clone();
    assert_eq!(response["id"], id, "{response}");
    assert_eq!(response["success"], true, "{response}");
    (got, response)
}

fn parse_errors(frames: &[Value]) -> Vec<String> {
    frames
        .iter()
        .filter(|f| f["type"] == "response" && f["command"] == "parse")
        .map(|f| {
            assert_eq!(f["success"], false, "{f}");
            f["error"].as_str().unwrap().to_string()
        })
        .collect()
}

#[test]
fn an_undecodable_or_oversized_stdin_line_is_answered_and_the_session_continues() {
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("s.jsonl");
    let mut child = serve_cmd(BIN, &base, session.to_str().unwrap()).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = common::child_frames(&mut child);
    let get_state = |id: &str| format!("{}\n", json!({ "type": "get_state", "id": id }));

    stdin.write_all(get_state("a").as_bytes()).unwrap();
    stdin.flush().unwrap();
    until_get_state(&mut frames, "a");

    // One byte that is not UTF-8, then a command.
    stdin
        .write_all(b"{\"type\":\"get_state\",\"id\":\"\xff\"}\n")
        .unwrap();
    stdin.write_all(get_state("b").as_bytes()).unwrap();
    stdin.flush().unwrap();
    let (got, _) = until_get_state(&mut frames, "b");
    let errors = parse_errors(&got);
    assert!(
        errors.len() == 1 && errors[0].contains("not valid UTF-8"),
        "{errors:?}"
    );

    // A line past the 8 MiB cap, written from another thread since the child reads it as we write.
    let writer = std::thread::spawn(move || {
        let huge = vec![b'x'; 9 * 1024 * 1024];
        stdin.write_all(&huge).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.write_all(get_state("c").as_bytes()).unwrap();
        stdin.flush().unwrap();
        stdin
    });
    let (got, _) = until_get_state(&mut frames, "c");
    let errors = parse_errors(&got);
    assert!(
        errors.len() == 1 && errors[0].contains("exceeds the 8388608-byte limit"),
        "{errors:?}"
    );

    // Still the same live session; closing stdin ends it cleanly.
    drop(writer.join().unwrap());
    assert!(child.wait().unwrap().success());
}
