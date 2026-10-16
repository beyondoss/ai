//! `run` e2e: text-mode stdout after retries. A stream that drops mid-reply is re-sent, but its partial
//! text had already been printed, so a script capturing stdout read `PARTIAL-PARTIAL-FINAL-ANSWER`
//! with exit 0. When stdout isn't a terminal, each step's text is now written only once the step
//! completes, so a failed attempt's text never reaches it. (`--json` is unchanged: every event,
//! including the failed attempt's deltas, still streams.)
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

/// A stream that starts a text reply and then ends without a `message_stop`: a mid-stream drop.
const TRUNCATED: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"PARTIAL-\"}}\n\n";

fn run(base: &str, extra: &[&str]) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "hi",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--retry-base-delay-ms",
            "10",
        ])
        .args(extra)
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[test]
fn retried_attempts_leave_no_text_on_a_piped_stdout() {
    let (base, bodies) = spawn_model_server(vec![
        TRUNCATED.into(),
        TRUNCATED.into(),
        turn_text("FINAL-ANSWER"),
    ]);
    let out = run(&base, &["--retry-max-retries", "1"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(bodies.lock().unwrap().len(), 3, "both drops were retried");
    assert_eq!(stdout, "FINAL-ANSWER\n", "stderr: {stderr}");
    assert!(
        stderr.contains("retrying"),
        "the retries are still reported on stderr: {stderr}"
    );
}

#[test]
fn a_whole_run_retry_drops_the_failed_attempts_text_but_keeps_completed_steps() {
    // Step 1 completes (a tool call); step 2 drops mid-stream until agent-core's in-turn retries (3)
    // are spent, so the whole run is re-invoked and step 2 is answered. Step 1's output stays; every
    // partial of step 2 goes.
    let mut turns = vec![turn_tool_use(
        "t1",
        "bash",
        &json!({ "command": "true" }).to_string(),
    )];
    turns.extend(std::iter::repeat_n(TRUNCATED.to_string(), 4));
    turns.push(turn_text("FINAL-ANSWER"));
    let (base, _bodies) = spawn_model_server(turns);
    let out = run(&base, &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert!(
        stderr.contains("transient error, retrying"),
        "a whole-run retry: {stderr}"
    );
    assert!(!stdout.contains("PARTIAL"), "stdout: {stdout:?}");
    assert!(
        stdout.contains("[tool: bash]"),
        "the completed step stays: {stdout:?}"
    );
    assert!(stdout.ends_with("FINAL-ANSWER\n"), "stdout: {stdout:?}");
}
