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

#[test]
fn settings_default_project_trust_round_trips_across_invocations() {
    // pi-parity fix: no persisted global default-trust policy existed at all — `always`/`never`/`ask`,
    // consulted only when neither `--trust-project` nor `--force-untrusted` is explicitly given.
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let set = Command::new(bin)
        .args(["settings", "--default-project-trust", "always"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(
        set.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&set.stderr)
    );
    let stdout = String::from_utf8_lossy(&set.stdout);
    assert!(
        stdout.contains("default_project_trust: always"),
        "{stdout}"
    );

    let show = Command::new(bin)
        .arg("settings")
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(show.status.success());
    let stdout = String::from_utf8_lossy(&show.stdout);
    assert!(
        stdout.contains("default_project_trust: always"),
        "a fresh invocation must see the persisted policy: {stdout}"
    );

    let cleared = Command::new(bin)
        .args(["settings", "--clear-default-project-trust"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(cleared.status.success());
    let stdout = String::from_utf8_lossy(&cleared.stdout);
    assert!(
        stdout.contains("default_project_trust: (not set)"),
        "{stdout}"
    );
}

#[test]
fn run_binary_stored_default_project_trust_always_enables_a_project_system_md_with_no_explicit_flag() {
    // The consultation point itself: with neither `--trust-project` nor `--force-untrusted` passed, a
    // stored `always` policy must make the on-disk SYSTEM.md take effect exactly like `--trust-project`
    // would.
    let home = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let set = Command::new(bin)
        .args(["settings", "--default-project-trust", "always"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(set.status.success());

    let claude_dir = project_dir.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(claude_dir.join("SYSTEM.md"), "STORED-ALWAYS-TRUST-MARKER").unwrap();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let output = Command::new(bin)
        .env("HOME", home.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(project_dir.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("STORED-ALWAYS-TRUST-MARKER"),
        "a stored default_project_trust=always must apply the on-disk SYSTEM.md with no explicit \
         --trust-project flag: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_explicit_force_untrusted_wins_over_a_stored_default_project_trust_always() {
    // An explicit per-run flag must always win over the stored global default, the same precedence
    // `default_model`/`default_gateway_url` already have relative to their own explicit flags.
    let home = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let set = Command::new(bin)
        .args(["settings", "--default-project-trust", "always"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(set.status.success());

    let claude_dir = project_dir.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(claude_dir.join("SYSTEM.md"), "SHOULD-NOT-APPLY-MARKER").unwrap();

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let output = Command::new(bin)
        .env("HOME", home.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--force-untrusted",
        ])
        .current_dir(project_dir.path())
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        !bodies[0].contains("SHOULD-NOT-APPLY-MARKER"),
        "an explicit --force-untrusted must win over a stored default_project_trust=always: {}",
        bodies[0]
    );
}

#[test]
fn ai_agent_config_dir_env_var_redirects_settings_and_trust_store_off_of_home() {
    // pi-parity fix: `settings.json`/`trusted-projects.json` previously always lived under
    // `$HOME/.claude/`, with no way to redirect them short of overriding `$HOME` itself (which also
    // moves every other HOME-relative thing this binary or the wider shell environment reads).
    let home = tempfile::tempdir().unwrap();
    let config_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let set = Command::new(bin)
        .args(["settings", "--model", "claude-opus-4-8"])
        .env("HOME", home.path())
        .env("AI_AGENT_CONFIG_DIR", config_dir.path())
        .output()
        .unwrap();
    assert!(
        set.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&set.stderr)
    );
    assert!(
        config_dir.path().join("settings.json").exists(),
        "settings.json must land under AI_AGENT_CONFIG_DIR"
    );
    assert!(
        !home.path().join(".claude/settings.json").exists(),
        "settings.json must not also land under $HOME/.claude when AI_AGENT_CONFIG_DIR overrides it"
    );

    let trust = Command::new(bin)
        .args(["trust", "."])
        .env("HOME", home.path())
        .env("AI_AGENT_CONFIG_DIR", config_dir.path())
        .current_dir(config_dir.path())
        .output()
        .unwrap();
    assert!(
        trust.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&trust.stderr)
    );
    assert!(
        config_dir.path().join("trusted-projects.json").exists(),
        "trusted-projects.json must land under AI_AGENT_CONFIG_DIR too"
    );
}
