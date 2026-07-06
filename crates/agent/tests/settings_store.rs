//! `agent settings` CLI subcommand and its consumption as a stored default by `run`/`serve`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Read, Write};
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
        .args(["run", "hi", "--gateway-url", &base, "--key", "bai_v1.test",
            "--no-session-persistence",
        ])
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
    assert!(stdout.contains("default_project_trust: always"), "{stdout}");

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
fn run_binary_stored_default_project_trust_always_enables_a_project_system_md_with_no_explicit_flag()
 {
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

// --- Round 3 (pi-parity): five more flag/env-only settings gain a persisted stored-default fallback.

#[test]
fn settings_shows_round_3_defaults_as_not_set_and_round_trips_set_show_clear() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let show = Command::new(bin)
        .arg("settings")
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(show.status.success());
    let stdout = String::from_utf8_lossy(&show.stdout);
    for field in [
        "default_bash_shell_path",
        "default_bash_command_prefix",
        "default_compaction_reserve_tokens",
        "default_compaction_keep_recent_tokens",
        "default_retry_max_retries",
        "default_retry_base_delay_ms",
        "default_provider_timeout_ms",
        "default_models_list",
        "default_skill_paths",
        "default_prompt_template_paths",
    ] {
        assert!(
            stdout.contains(&format!("{field}: (not set)")),
            "{field}: {stdout}"
        );
    }

    let set = Command::new(bin)
        .args([
            "settings",
            "--default-bash-shell-path",
            "/bin/zsh",
            "--default-bash-command-prefix",
            "source .venv/bin/activate",
            "--default-compaction-reserve-tokens",
            "8192",
            "--default-compaction-keep-recent-tokens",
            "12000",
            "--default-retry-max-retries",
            "5",
            "--default-retry-base-delay-ms",
            "500",
            "--default-provider-timeout-ms",
            "60000",
            "--default-models",
            "claude-opus-4-8,gpt-5",
            "--default-skill-paths",
            "/opt/skills",
            "--default-prompt-template-paths",
            "/opt/prompts",
        ])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(set.status.success(), "stderr: {}", String::from_utf8_lossy(&set.stderr));
    let stdout = String::from_utf8_lossy(&set.stdout);
    assert!(stdout.contains("default_bash_shell_path: /bin/zsh"), "{stdout}");
    assert!(
        stdout.contains("default_bash_command_prefix: source .venv/bin/activate"),
        "{stdout}"
    );
    assert!(
        stdout.contains("default_compaction_reserve_tokens: 8192"),
        "{stdout}"
    );
    assert!(
        stdout.contains("default_compaction_keep_recent_tokens: 12000"),
        "{stdout}"
    );
    assert!(stdout.contains("default_retry_max_retries: 5"), "{stdout}");
    assert!(stdout.contains("default_retry_base_delay_ms: 500"), "{stdout}");
    assert!(stdout.contains("default_provider_timeout_ms: 60000"), "{stdout}");
    assert!(
        stdout.contains("default_models_list: claude-opus-4-8,gpt-5"),
        "{stdout}"
    );
    assert!(stdout.contains("default_skill_paths: /opt/skills"), "{stdout}");
    assert!(
        stdout.contains("default_prompt_template_paths: /opt/prompts"),
        "{stdout}"
    );

    // A fresh invocation must see the same persisted values.
    let show = Command::new(bin)
        .arg("settings")
        .env("HOME", home.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&show.stdout);
    assert!(stdout.contains("default_bash_shell_path: /bin/zsh"), "{stdout}");
    assert!(stdout.contains("default_models_list: claude-opus-4-8,gpt-5"), "{stdout}");

    let cleared = Command::new(bin)
        .args([
            "settings",
            "--clear-default-bash-shell-path",
            "--clear-default-compaction-reserve-tokens",
            "--clear-default-models",
        ])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(cleared.status.success());
    let stdout = String::from_utf8_lossy(&cleared.stdout);
    assert!(stdout.contains("default_bash_shell_path: (not set)"), "{stdout}");
    assert!(
        stdout.contains("default_compaction_reserve_tokens: (not set)"),
        "{stdout}"
    );
    assert!(stdout.contains("default_models_list: (not set)"), "{stdout}");
    // Clearing those three must not have clobbered an unrelated field left set.
    assert!(
        stdout.contains("default_bash_command_prefix: source .venv/bin/activate"),
        "clearing other fields must not touch this one: {stdout}"
    );
}

#[test]
fn serve_binary_stored_default_bash_shell_path_fails_fast_like_an_explicit_flag_would() {
    // The existing "--bash-shell-path not found" fail-fast validation must still apply to a value that
    // came from a stored default, not just an explicit flag — the check was moved to run *after* the
    // whole fallback chain resolves, precisely so this keeps working.
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--default-bash-shell-path", "/no/such/shell-binary"])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
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
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdin.take()); // the process must exit before ever trying to read a command line
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success(), "stderr: {stderr}");
    assert!(stderr.contains("--bash-shell-path"), "stderr: {stderr}");
}

#[test]
fn serve_binary_explicit_bash_shell_path_flag_wins_over_a_stored_default() {
    if !std::path::Path::new("/bin/sh").exists() {
        return; // no alternate shell on this host to prove the override actually took effect
    }
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // A stored default that, left alone, would fail startup outright.
    Command::new(bin)
        .args(["settings", "--default-bash-shell-path", "/no/such/shell-binary"])
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
            "--session-file",
            &session_file,
            "--bash-shell-path",
            "/bin/sh",
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
    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "an explicit --bash-shell-path must win over a bad stored default and let serve start normally"
    );
}

#[test]
fn run_binary_stored_default_compaction_tokens_force_compaction_with_no_explicit_flag() {
    // Mirrors `run_cli_flags.rs`'s identical flag-driven proof that
    // `--compaction-reserve-tokens`/`--compaction-keep-recent-tokens` force a proactive compaction call
    // on a resumed session crossing the soft threshold — but via a persisted `agent settings` default
    // instead of an explicit flag.
    use agent_core::{ContentBlock, Message, Session};
    use beyond_ai_agent::session_store::{SessionMeta, SessionRepo};

    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args([
            "settings",
            "--default-compaction-reserve-tokens",
            "50",
            "--default-compaction-keep-recent-tokens",
            "1",
        ])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let session_file = {
        let repo = SessionRepo::open(dir.path()).unwrap();
        let mut store = repo.create(SessionMeta::new("/w", "claude-test")).unwrap();
        let mut seed = Session::new();
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        seed.user("u".repeat(400));
        seed.push(Message::assistant(vec![ContentBlock::text(
            "a".repeat(400),
        )]));
        store.append_new(&seed.messages).unwrap();
        store.path().to_string_lossy().into_owned()
    };

    let (base, bodies) = spawn_model_server(vec![turn_text("SUMMARY"), turn_text("answered")]);

    let output = Command::new(bin)
        .args([
            "run",
            "continue please",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            &session_file,
            "--context-window",
            "200",
            "--no-session-persistence",
        ])
        .env("HOME", home.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "a stored compaction-token default must reach the same threshold check the explicit flags do: \
         {bodies:#?}"
    );
}

#[test]
fn serve_binary_stored_default_models_list_scopes_cycle_model_with_no_explicit_flag() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--default-models", "claude-opus-4-8,gpt-4o"])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-opus-4-8",
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

    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = common::read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(last["data"]["model"], "gpt-4o", "frames: {frames:#?}");
    assert_eq!(
        last["data"]["scoped"], true,
        "a stored default --models list must scope cycle_model exactly like the flag does: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_binary_explicit_models_flag_wins_over_a_stored_default_models_list() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    // A stored default scope that does NOT include `claude-opus-4-8` at all.
    Command::new(bin)
        .args(["settings", "--default-models", "gpt-4o,claude-sonnet-4-5"])
        .env("HOME", home.path())
        .output()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-opus-4-8",
            // A single-entry explicit scope containing only the active model — if this correctly wins
            // outright (replacing, not merging with, the stored default above), cycling must wrap right
            // back to this same single entry rather than advancing into the stored default's own list.
            "--models",
            "claude-opus-4-8",
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

    writeln!(stdin, "{}", json!({ "type": "cycle_model" })).unwrap();
    stdin.flush().unwrap();
    let frames = common::read_until_response(&mut stdout, "cycle_model");
    let last = frames.last().unwrap();
    assert_eq!(
        last["data"]["model"], "claude-opus-4-8",
        "an explicit --models list must win outright over a stored default, not merge with it: {frames:#?}"
    );
    drop(stdin);
    child.wait().unwrap();
}

// --- Feature 2 (Round 3, pi-parity): project-level `.claude/settings.json` tier.

#[test]
fn run_binary_project_settings_json_overrides_the_global_default_when_trusted() {
    let home = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--model", "claude-global-model"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    let trust = Command::new(bin)
        .args(["trust", project_dir.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust.status.success());

    let claude_dir = project_dir.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        r#"{"default_model":"claude-project-model"}"#,
    )
    .unwrap();

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
            "--no-session-persistence",
        ])
        .current_dir(project_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies.iter().any(|b| b.contains("claude-project-model")),
        "a trusted project's settings.json must override the global default: {bodies:#?}"
    );
    assert!(
        !bodies.iter().any(|b| b.contains("claude-global-model")),
        "{bodies:#?}"
    );
}

#[test]
fn run_binary_global_value_survives_when_project_settings_json_leaves_it_unset() {
    let home = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--model", "claude-global-model"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    let trust = Command::new(bin)
        .args(["trust", project_dir.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust.status.success());

    let claude_dir = project_dir.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    // Sets an unrelated field only — `default_model` is deliberately left unset at the project tier.
    std::fs::write(
        claude_dir.join("settings.json"),
        r#"{"default_session_dir":"/tmp/project-sessions-should-not-be-used"}"#,
    )
    .unwrap();

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
            "--no-session-persistence",
        ])
        .current_dir(project_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies.iter().any(|b| b.contains("claude-global-model")),
        "a field the project doesn't set must fall through to the global value: {bodies:#?}"
    );
}

#[test]
fn run_binary_untrusted_project_settings_json_is_completely_ignored() {
    let home = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--model", "claude-global-model"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    // Deliberately never trusted (no `agent trust`, no `--trust-project`).

    let claude_dir = project_dir.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(
        claude_dir.join("settings.json"),
        r#"{"default_model":"claude-attacker-model"}"#,
    )
    .unwrap();

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
            "--no-session-persistence",
        ])
        .current_dir(project_dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies.iter().any(|b| b.contains("claude-global-model")),
        "{bodies:#?}"
    );
    assert!(
        !bodies.iter().any(|b| b.contains("claude-attacker-model")),
        "an untrusted project's settings.json must be completely ignored, not partially applied: \
         {bodies:#?}"
    );
}

#[test]
fn run_binary_project_settings_missing_or_malformed_degrades_to_global_only() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    Command::new(bin)
        .args(["settings", "--model", "claude-global-model"])
        .env("HOME", home.path())
        .output()
        .unwrap();

    // Case 1: a trusted project directory with no `.claude/settings.json` at all.
    let project_dir_missing = tempfile::tempdir().unwrap();
    let trust = Command::new(bin)
        .args(["trust", project_dir_missing.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust.status.success());

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
            "--no-session-persistence",
        ])
        .current_dir(project_dir_missing.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "a missing project settings.json must not error: stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies.iter().any(|b| b.contains("claude-global-model")),
        "a missing project settings.json must degrade to global-only: {bodies:#?}"
    );

    // Case 2: a trusted project directory with a malformed `.claude/settings.json`.
    let project_dir_malformed = tempfile::tempdir().unwrap();
    let trust = Command::new(bin)
        .args(["trust", project_dir_malformed.path().to_str().unwrap()])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(trust.status.success());
    let claude_dir = project_dir_malformed.path().join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::write(claude_dir.join("settings.json"), "not valid json at all { [ }").unwrap();

    let (base2, bodies2) = spawn_model_server(vec![turn_text("ok")]);
    let output2 = Command::new(bin)
        .env("HOME", home.path())
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--no-session-persistence",
        ])
        .current_dir(project_dir_malformed.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "a malformed project settings.json must not crash the run: stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output2.stdout),
        String::from_utf8_lossy(&output2.stderr)
    );
    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2.iter().any(|b| b.contains("claude-global-model")),
        "a malformed project settings.json must degrade to global-only: {bodies2:#?}"
    );
}

#[test]
fn serve_binary_stored_steering_mode_and_follow_up_mode_seed_a_fresh_processs_initial_state() {
    // Task 1 (pi-parity fix, pass 19): `steering_mode`/`follow_up_mode` have no `agent settings`
    // CLI-flag setter of their own — they're persisted by `serve`'s own `set_steering_mode`/
    // `set_follow_up_mode` RPC commands (`settings::Settings::steering_mode`'s own doc comment) — but
    // until this pass, nothing on `serve`'s *startup* path ever read that persisted value back: every
    // fresh `serve` process silently started at `QueueMode::default()` (`one_at_a_time` for both lanes)
    // regardless of what a previous session had persisted. Reproduced here as two successive `serve`
    // processes sharing the same `HOME` (so they share the same settings store): the first persists
    // non-default modes for both lanes via the RPC commands; the second, started with neither
    // `--steering-mode` nor `--follow-up-mode`, must start already reflecting them.
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    {
        let dir = tempfile::tempdir().unwrap();
        let session_file = dir.path().join("s1.jsonl").to_string_lossy().into_owned();
        let (base, _bodies) = spawn_model_server(vec![]);
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

        writeln!(
            stdin,
            "{}",
            json!({ "type": "set_steering_mode", "mode": "all" })
        )
        .unwrap();
        stdin.flush().unwrap();
        let frames = common::read_until_response(&mut stdout, "set_steering_mode");
        assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

        writeln!(
            stdin,
            "{}",
            json!({ "type": "set_follow_up_mode", "mode": "all" })
        )
        .unwrap();
        stdin.flush().unwrap();
        let frames = common::read_until_response(&mut stdout, "set_follow_up_mode");
        assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

        drop(stdin);
        child.wait().unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
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

    writeln!(stdin, "{}", json!({ "type": "get_state" })).unwrap();
    stdin.flush().unwrap();
    let frames = common::read_until_response(&mut stdout, "get_state");
    let data = &frames.last().unwrap()["data"];
    assert_eq!(
        data["steering_mode"], "all",
        "a fresh serve process must start from the persisted steering_mode default, not \
         QueueMode::default(): {data:#?}"
    );
    assert_eq!(
        data["follow_up_mode"], "all",
        "a fresh serve process must start from the persisted follow_up_mode default too: {data:#?}"
    );

    drop(stdin);
    child.wait().unwrap();
}
