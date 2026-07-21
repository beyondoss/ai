//! `serve` e2e: per-session `/session` working memory — the compaction-surviving scratchpad — driven
//! through the **real `serve` binary**. This is the money test: a fact the model writes to `/session`
//! survives its own context compaction (which folds the raw transcript into a lossy summary that drops
//! the specific), the model is actively reminded to read it back, and it recovers the exact value. Plus:
//! the store lives in the `<...>.memory/` sibling of the session file, and `--no-session-memory` opts out.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::path::Path;
use std::process::ChildStdin;

use common::{
    ChildGuard, SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, sse, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn mem_turn(id: &str, args: Value) -> String {
    turn_tool_use(id, "memory", &args.to_string())
}

/// A text turn that reports an explicit prompt (`input_tokens`) size, so a test can drive the live
/// prompt across the compaction pressure point without actually building a huge conversation.
fn turn_text_with_input(text: &str, input_tokens: u32) -> String {
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": input_tokens, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ])
}

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn kill(mut child: ChildGuard) {
    let _ = child.kill();
    let _ = child.wait();
}

fn tool_result_texts(frames: &[Value], name: &str) -> Vec<String> {
    frames
        .iter()
        .filter(|f| {
            f["type"] == "event" && f["event"]["kind"] == "tool_end" && f["event"]["name"] == name
        })
        .filter_map(|f| f["event"]["result"].as_str().map(str::to_string))
        .collect()
}

#[test]
fn a_fact_written_to_session_survives_compaction_and_is_recovered_after_the_reminder() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let session_str = session_file.to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        // Prompt 1: the model records a precise fact in its per-session working memory.
        mem_turn(
            "toolu_1",
            json!({ "command": "create", "path": "/session/facts.md", "file_text": "the prod DB port is 5433\n" }),
        ),
        turn_text("noted"),
        // The summarization call the manual `compact` makes — deliberately drops the specific, exactly as
        // a real summarizing model usually would.
        turn_text("The user shared some configuration details."),
        // Prompt 2 (post-compaction): the model reads its working memory back to recover the exact value.
        mem_turn(
            "toolu_2",
            json!({ "command": "view", "path": "/session/facts.md" }),
        ),
        turn_text("the port is 5433"),
    ]);

    let mut child = serve_cmd(BIN, &base, &session_str)
        // Force an aggressive cut so the short session actually compacts (mirrors serve_todo's test).
        .args(["--compaction-keep-recent-tokens", "1"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // --- Prompt 1: write the fact to /session. -------------------------------------------------------
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "save the prod DB port" }),
    );
    assert_eq!(
        read_until_response(&mut stdout, "prompt").last().unwrap()["success"],
        true
    );

    // The working-memory store is the `<...>.memory/` sibling of the session file, and holds the fact.
    let mem_sibling = session_file.with_extension("memory");
    let facts = mem_sibling.join("facts.md");
    assert!(
        facts.exists(),
        "the /session store must live beside the session file at {}",
        mem_sibling.display()
    );
    assert!(
        std::fs::read_to_string(&facts).unwrap().contains("5433"),
        "the working-memory file must hold the exact fact"
    );

    // --- Compact: fold the transcript into a lossy summary. ------------------------------------------
    send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
    let frames = read_until_response(&mut stdout, "compact");
    let resp = frames.last().unwrap();
    assert_eq!(
        resp["data"]["compacted"], true,
        "the session must actually compact for this test to mean anything: {resp}"
    );

    // The working memory itself is untouched by compaction.
    assert!(
        std::fs::read_to_string(&facts).unwrap().contains("5433"),
        "the working memory must be untouched by compaction"
    );

    // Deterministic carry: the summary itself lists the memory the model authored, so even a summarizer
    // that never mentions it can't erase the fact that a `/session` note exists and where.
    let transcript = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        transcript.contains("<memory-notes>") && transcript.contains("/session/facts.md"),
        "the compaction summary must carry a <memory-notes> block naming the note the model wrote"
    );

    // --- Prompt 2: the reminder rides the next request, and the model recovers the value. ------------
    let before = bodies.lock().unwrap().len();
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "what was the prod DB port?" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");

    // The first post-compaction model request: the transcript the model now sees is the lossy summary, so
    // the specific is *gone* from what it's given — and the active recall reminder (pointing at /session)
    // rides that very request to tell it where to look.
    let new_bodies: Vec<String> = bodies.lock().unwrap()[before..].to_vec();
    let first_after = &new_bodies[0];
    assert!(
        !first_after.contains("5433"),
        "compaction must have dropped the specific from what the model is given: {first_after}"
    );
    assert!(
        first_after.contains("compacted") && first_after.contains("/session"),
        "the post-compaction request must carry the working-memory recall reminder: {first_after}"
    );

    // And the model recovered the exact value from /session that the summary had dropped.
    let recovered = tool_result_texts(&frames, "memory");
    assert!(
        recovered.iter().any(|r| r.contains("5433")),
        "view /session/facts.md must return the specific the summary lost: {recovered:?}"
    );

    kill(child);
}

#[test]
fn no_session_memory_omits_the_session_root_but_keeps_durable() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    let session_str = session_file.to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![turn_text("nothing to do")]);
    let mut child = serve_cmd(BIN, &base, &session_str)
        .args(["--no-session-memory"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    send(&mut stdin, json!({ "type": "prompt", "message": "hello" }));
    read_until_response(&mut stdout, "prompt");

    let first = bodies.lock().unwrap()[0].clone();
    // Durable memory is still present; the session working-memory guidance is not.
    assert!(
        first.contains("Memory (durable, cross-session)"),
        "durable /memories must still be advertised"
    );
    assert!(
        !first.contains("Working memory (this session)"),
        "--no-session-memory must drop the /session guidance"
    );
    assert!(
        !session_file.with_extension("memory").exists(),
        "no /session store directory should be created under --no-session-memory"
    );
    kill(child);
}

fn compacted_event(frames: &[Value]) -> bool {
    frames
        .iter()
        .any(|f| f["type"] == "event" && f["event"]["kind"] == json!("compacted"))
}

// window 10000, reserve 2000 -> compaction threshold 8000; pressure point = 8000/5*4 = 6400.
// A 7000-token prompt is past the pressure point but well below the cut, with enough headroom that the
// nudge's own tokens don't tip it over — so the nudge fires and NO compaction happens. The nudge is a
// mid-run steer, so it rides the immediate continuation turn within the same prompt (a "checkpoint now"),
// not a later prompt.
#[test]
fn a_pressure_nudge_fires_before_compaction_within_the_same_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let session_str = dir.path().join("s.jsonl").to_string_lossy().into_owned();

    let (base, bodies) = spawn_model_server(vec![
        turn_text_with_input("ok", 7000), // turn 1: crosses the pressure point, stays below the cut
        turn_text_with_input("done", 20), // turn 2: carries the queued nudge (steer continuation)
    ]);
    let mut child = serve_cmd(BIN, &base, &session_str)
        .args([
            "--context-window",
            "10000",
            "--compaction-reserve-tokens",
            "2000",
        ])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(&mut stdin, json!({ "type": "prompt", "message": "first" }));
    let p1 = read_until_response(&mut stdout, "prompt");
    assert_eq!(p1.last().unwrap()["success"], true, "{p1:#?}");
    assert!(
        !compacted_event(&p1),
        "the nudge must fire BEFORE any compaction"
    );

    // Two requests: the original turn, then the steer continuation the nudge forced.
    let reqs = bodies.lock().unwrap().clone();
    assert_eq!(
        reqs.len(),
        2,
        "the nudge must drive one immediate checkpoint turn: {reqs:#?}"
    );
    assert!(
        !reqs[0].contains("getting large"),
        "the nudge must not appear in the request that triggered it"
    );
    assert!(
        reqs[1].contains("getting large") && reqs[1].contains("/session"),
        "the pressure nudge (checkpoint to /session) must ride the immediate continuation: {}",
        reqs[1]
    );
    kill(child);
}

#[test]
fn below_the_pressure_point_and_under_no_session_memory_no_nudge_fires() {
    // Two guards in one process pair: a prompt that stays *under* the pressure point never nudges, and
    // `--no-session-memory` suppresses the nudge even when the point is crossed.
    for (extra, input_tokens, label) in [
        (
            vec![
                "--context-window",
                "1000",
                "--compaction-reserve-tokens",
                "200",
            ],
            100u32,
            "below point",
        ),
        (
            vec![
                "--context-window",
                "1000",
                "--compaction-reserve-tokens",
                "200",
                "--no-session-memory",
            ],
            700,
            "no-session-memory",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let session_str = dir.path().join("s.jsonl").to_string_lossy().into_owned();
        let (base, bodies) = spawn_model_server(vec![
            turn_text_with_input("ok", input_tokens),
            turn_text_with_input("done", 20),
        ]);
        let mut child = serve_cmd(BIN, &base, &session_str)
            .args(extra)
            .spawn_guarded();
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        send(&mut stdin, json!({ "type": "prompt", "message": "first" }));
        read_until_response(&mut stdout, "prompt");
        let before = bodies.lock().unwrap().len();
        send(&mut stdin, json!({ "type": "prompt", "message": "second" }));
        read_until_response(&mut stdout, "prompt");
        let p2_req = bodies.lock().unwrap()[before].clone();
        assert!(
            !p2_req.contains("getting large"),
            "no pressure nudge expected ({label}): {p2_req}"
        );
        kill(child);
    }
}

// Guard: the session-memory sibling name derivation must match `session_store`'s trash/restore logic.
#[test]
fn the_session_memory_sibling_name_is_the_dot_memory_of_the_session_file() {
    let p = Path::new("/repo/1700_abc.jsonl");
    assert_eq!(
        p.with_extension("memory"),
        Path::new("/repo/1700_abc.memory")
    );
}
