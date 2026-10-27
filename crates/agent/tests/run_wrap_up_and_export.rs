//! `run` e2e: how a run that didn't simply finish says so. A run capped by `--max-steps` ends normally
//! on the model's wrap-up answer (exit 0) but used to say nothing about it in text mode; and
//! `--export` sat after the early returns of a failed or cancelled run, so exactly the transcripts
//! most worth reading were never exported.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};

fn run(base: &str, dir: &std::path::Path, extra: &[&str]) -> std::process::Output {
    run_cmd(env!("CARGO_BIN_EXE_agent"))
        .args([
            "run",
            "do a big task",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .args(extra)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[test]
fn a_run_capped_by_max_steps_says_so_on_stderr_and_still_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", r#"{"command":"echo step1"}"#),
        turn_text("Summary: not done, ran out of steps"),
    ]);
    let out = run(&base, dir.path(), &["--max-steps", "1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(bodies.lock().unwrap().len(), 2, "the wrap-up step ran");
    assert_eq!(out.status.code(), Some(0), "stderr: {stderr}");
    assert!(
        stderr.contains("--max-steps ceiling (1)"),
        "a capped run says so: {stderr}"
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("Summary: not done"));
}

#[test]
fn an_uncapped_run_prints_no_ceiling_notice() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("done")]);
    let out = run(&base, dir.path(), &["--max-steps", "5"]);
    assert!(out.status.success());
    assert!(!String::from_utf8_lossy(&out.stderr).contains("ceiling"));
}

#[test]
fn export_is_written_for_a_run_that_fails() {
    let dir = tempfile::tempdir().unwrap();
    let export = dir.path().join("out.html");
    // No scripted reply: the request fails, with every retry layer off.
    let (base, _bodies) = spawn_model_server(vec![]);
    let out = run(
        &base,
        dir.path(),
        &[
            "--export",
            export.to_str().unwrap(),
            "--retry-max-retries",
            "0",
            "--no-retry",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the run fails: {stderr}");
    let html = std::fs::read_to_string(&export)
        .unwrap_or_else(|e| panic!("no export for a failed run ({e}); stderr: {stderr}"));
    assert!(
        html.contains("do a big task"),
        "the transcript holds the prompt"
    );
}
