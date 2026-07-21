//! `run` e2e: stdout write robustness. Task #10 (pi-parity fix): `run_turn_once` used raw
//! `println!`/`print!` for both `--json` NDJSON events and plain-text streaming — Rust's `println!`
//! panics internally on any stdout write error, including `EPIPE`, so `agent run --json "task" |
//! head -1` used to crash with "Broken pipe" (exit 101) the moment `head` closed its end, instead of
//! exiting cleanly. These tests reproduce a closed read end without an actual shell pipeline: they
//! read a bounded amount from the child's own stdout pipe, then drop that handle — closing this
//! process's read end exactly like `head` hanging up — and assert the child still exits cleanly
//! rather than panicking.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Read as _};
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{SpawnGuarded, run_cmd, spawn_model_server, sse};
use serde_json::json;

/// An SSE turn whose single text delta is large enough (`len` bytes) that writing its rendered NDJSON
/// line can't complete in one shot once nothing is reading the other end of the pipe — the default
/// Linux pipe buffer is 64KiB, so a delta well past that guarantees the writer blocks (and then, once
/// the reader is gone, fails with `EPIPE`) regardless of exact scheduling between the two processes.
fn turn_with_a_large_text_delta(len: usize) -> String {
    let text = "x".repeat(len);
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// Wait for `child` to exit, or panic after `timeout` — a hang here means the fix regressed into a
/// blocked/deadlocked write instead of a clean exit.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> std::process::ExitStatus {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        assert!(
            Instant::now() < deadline,
            "process did not exit within {timeout:?} of its stdout being closed"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn run_json_mode_exits_cleanly_instead_of_panicking_when_stdout_is_closed_early() {
    let (base, _bodies) = spawn_model_server(vec![turn_with_a_large_text_delta(500_000)]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = run_cmd(bin)
        .args([
            "run",
            "say something long",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--json",
        ])
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();

    // Read exactly the leading `{"kind":"session",...}` header line — proving the write path up to
    // (and including) that line still works normally — then close our end, simulating `head -1`
    // hanging up right after taking its one line.
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut header = String::new();
    stdout.read_line(&mut header).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(header.trim()).unwrap();
    assert_eq!(parsed["kind"], "session");
    drop(stdout);

    let status = wait_with_timeout(&mut child, Duration::from_secs(10));
    let mut stderr = String::new();
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut stderr);
    }
    assert_eq!(
        status.code(),
        Some(0),
        "a closed stdout pipe must exit cleanly (0), not panic on Broken Pipe (101). stderr: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "must not panic on the broken pipe: {stderr}"
    );
}

#[test]
fn run_text_mode_exits_cleanly_instead_of_panicking_when_stdout_is_closed_early() {
    let (base, _bodies) = spawn_model_server(vec![turn_with_a_large_text_delta(500_000)]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = run_cmd(bin)
        .args([
            "run",
            "say something long",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();

    // Text mode has no leading header line — read a small, bounded prefix of the streamed text (well
    // under the 64KiB pipe buffer) so we know the child has started writing, then close our end.
    let mut stdout = child.stdout.take().unwrap();
    let mut prefix = [0u8; 64];
    stdout.read_exact(&mut prefix).unwrap();
    assert!(prefix.iter().all(|&b| b == b'x'));
    drop(stdout);

    let status = wait_with_timeout(&mut child, Duration::from_secs(10));
    let mut stderr = String::new();
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut stderr);
    }
    assert_eq!(
        status.code(),
        Some(0),
        "a closed stdout pipe must exit cleanly (0), not panic on Broken Pipe (101). stderr: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "must not panic on the broken pipe: {stderr}"
    );
}
