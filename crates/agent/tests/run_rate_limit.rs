//! `run` e2e: a rate-limited provider is waited out, not given up on after three quick retries; and
//! `--retry-rate-limit-secs` bounds that wait.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text};

/// Anthropic's in-stream rate-limit error, as a 200 stream that fails partway.
const RATE_LIMITED: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n\
event: error\n\
data: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Number of request tokens has exceeded your per-minute rate limit\"}}\n\n";

fn run(base: &str, extra: &[&str]) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut args = vec![
        "run",
        "hi",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--no-session-persistence",
    ];
    args.extend_from_slice(extra);
    run_cmd(bin)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary")
}

#[test]
fn run_waits_out_a_rate_limit_that_outlasts_three_quick_retries() {
    // Four rate limits in a row: more than the three retries every transient error used to get, with
    // the whole-run retry off (`--retry-max-retries 0`) so the per-turn rate-limit wait alone has to
    // carry the run.
    let mut responses = vec![RATE_LIMITED.to_string(); 4];
    responses.push(turn_text("made it"));
    let (base, requests) = spawn_model_server(responses);

    let output = run(&base, &["--retry-max-retries", "0"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("made it"), "{stdout}");
    assert_eq!(
        stderr.matches("[rate limited, retrying in").count(),
        4,
        "each wait is announced on stderr: {stderr}"
    );
    assert_eq!(requests.lock().unwrap().len(), 5);
}

#[test]
fn run_fails_clearly_once_the_rate_limit_budget_is_spent() {
    let (base, requests) = spawn_model_server(vec![RATE_LIMITED.to_string(); 4]);

    let output = run(&base, &["--retry-rate-limit-secs", "0"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("rate-limit retry budget exhausted"),
        "the failure names the spent budget: {stderr}"
    );
    assert!(
        !stderr.contains("[transient error, retrying"),
        "a spent budget is not retried again by the whole-run layer: {stderr}"
    );
    assert_eq!(requests.lock().unwrap().len(), 1);
}

/// Anthropic's in-stream overload error: the provider is busy, not throttling this caller.
const OVERLOADED: &str = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n\
event: error\n\
data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";

#[test]
fn run_labels_an_overload_as_an_overload_not_a_rate_limit() {
    let mut responses = vec![OVERLOADED.to_string(); 2];
    responses.push(turn_text("made it"));
    let (base, _requests) = spawn_model_server(responses);

    let output = run(&base, &["--retry-max-retries", "0"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    assert_eq!(
        stderr.matches("[provider overloaded, retrying in").count(),
        2,
        "{stderr}"
    );
    assert!(!stderr.contains("rate limited"), "{stderr}");
}

#[test]
fn run_names_an_overload_when_its_budget_is_spent() {
    let (base, _requests) = spawn_model_server(vec![OVERLOADED.to_string(); 4]);

    let output = run(&base, &["--retry-rate-limit-secs", "0"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(stderr.contains("provider still overloaded"), "{stderr}");
    assert!(!stderr.contains("rate limited"), "{stderr}");
}
