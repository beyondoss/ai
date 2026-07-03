//! `agent settings` CLI subcommand and its consumption as a stored default by `run`/`serve`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::process::{Command, Stdio};

use common::{ISOLATED_HOME, run_cmd, spawn_model_server, turn_text};
use serde_json::json;

#[test]
fn settings_with_no_flags_shows_every_default_as_not_set() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = Command::new(bin)
        .arg("settings")
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("default_model: (not set)"), "{stdout}");
    assert!(
        stdout.contains("default_gateway_url: (not set)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("default_session_dir: (not set)"),
        "{stdout}"
    );
}

#[test]
fn settings_set_and_show_round_trips_across_invocations() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let set = Command::new(bin)
        .args(["settings", "--model", "claude-opus-4-8"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(set.status.success());
    let stdout = String::from_utf8_lossy(&set.stdout);
    assert!(stdout.contains("updated settings:"), "{stdout}");
    assert!(
        stdout.contains("default_model: claude-opus-4-8"),
        "{stdout}"
    );

    // A fresh invocation (a different process) must see the same persisted value.
    let show = Command::new(bin)
        .arg("settings")
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(show.status.success());
    let stdout = String::from_utf8_lossy(&show.stdout);
    assert!(
        stdout.contains("default_model: claude-opus-4-8"),
        "{stdout}"
    );
}

#[test]
fn settings_clear_removes_a_field_without_touching_others() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args([
            "settings",
            "--model",
            "claude-opus-4-8",
            "--gateway-url",
            "http://gw.internal",
        ])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let cleared = Command::new(bin)
        .args(["settings", "--clear-model"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(cleared.status.success());
    let stdout = String::from_utf8_lossy(&cleared.stdout);
    assert!(stdout.contains("default_model: (not set)"), "{stdout}");
    assert!(
        stdout.contains("default_gateway_url: http://gw.internal"),
        "clearing one field must not clobber another: {stdout}"
    );
}

#[test]
fn run_binary_uses_the_stored_default_model_when_no_flag_or_env_var_is_given() {
    // Pi-parity fix: no stored-settings layer existed at all — an operator had to retype `--model` (or
    // export `AI_AGENT_MODEL`) on every invocation.
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--model", "claude-test"])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let output = Command::new(bin)
        .args(["run", "hi", "--gateway-url", &base, "--key", "bai_v1.test"])
        .env("HOME", home.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("claude-test") || bodies.iter().any(|b| b.contains("claude-test")),
        "the stored default model must reach the wire request: {bodies:#?}"
    );
}

#[test]
fn run_binary_explicit_model_flag_wins_over_a_stored_default() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--model", "claude-stored-default"])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let output = Command::new(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-explicit-flag",
        ])
        .env("HOME", home.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(output.status.success());
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies.iter().any(|b| b.contains("claude-explicit-flag")),
        "{bodies:#?}"
    );
    assert!(
        !bodies.iter().any(|b| b.contains("claude-stored-default")),
        "{bodies:#?}"
    );
}

#[test]
fn run_binary_with_no_settings_file_at_all_falls_back_to_the_built_in_default() {
    // A missing settings file (the common case — most operators never run `agent settings`) must not
    // error or change behavior from before this feature existed.
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["run", "hi", "--gateway-url", &base, "--key", "bai_v1.test"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn serve_binary_uses_the_stored_default_model_when_no_flag_or_env_var_is_given() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--model", "claude-test"])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--session-file",
            &session_file,
        ])
        .env("HOME", home.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    common::read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies.iter().any(|b| b.contains("claude-test")),
        "{bodies:#?}"
    );
}

#[test]
fn serve_binary_stored_session_dir_default_does_not_override_an_explicit_session_file() {
    // A real precedence hazard this fix had to avoid: `Persistence::open` checks `session_dir` before
    // `session_file`, so blindly filling in a stored `default_session_dir` even when the operator
    // explicitly passed `--session-file` would silently switch them into repo mode instead of the file
    // mode they actually asked for.
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let unrelated_dir = home.path().join("unrelated-repo-dir");

    Command::new(bin)
        .args(["settings", "--session-dir", unrelated_dir.to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
        ])
        .env("HOME", home.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    common::read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    // The explicit `--session-file` must actually have been used — not the stored session-dir default.
    assert!(
        std::path::Path::new(&session_file).exists(),
        "the explicit --session-file must still be honored"
    );
    assert!(
        !unrelated_dir.exists(),
        "the stored default_session_dir must not have been used instead — repo mode would have \
         created this directory"
    );
}

#[test]
fn settings_file_never_exists_under_the_isolated_test_home() {
    // Sanity check underlying every other hermetic test in this suite: no settings file exists under
    // `ISOLATED_HOME`, so `run`/`serve`'s stored-default lookup is always a harmless no-op there,
    // identical to before this feature existed.
    assert!(
        !std::path::Path::new(ISOLATED_HOME)
            .join(".claude/settings.json")
            .exists()
    );
}
