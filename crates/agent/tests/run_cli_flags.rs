//! `run` e2e: Standalone CLI surface: `--version`/`--help`/`list-models`/`export`, and flags that reach the wire request untouched by a live turn.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

#[test]
fn run_binary_exports_the_transcript_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("all done")]);
    let export_path = dir.path().join("transcript.html");

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--export",
            export_path.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let html = std::fs::read_to_string(&export_path).expect("exported file must exist");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("say hi"));
    assert!(html.contains("all done"));
}

#[test]
fn run_binary_list_models_prints_known_model_ids_with_no_gateway_or_key() {
    // A pure informational query — no `--gateway-url`/`--key` needed, matching `tools`'s own shape.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["list-models"])
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("claude-opus-4-8"), "stdout: {stdout}");
    assert!(stdout.contains("gpt-5"), "stdout: {stdout}");
}

#[test]
fn run_binary_version_flag_prints_only_the_version_to_stdout() {
    // F-L3 (pi: stdout-cleanliness.test.ts "prints --version to stdout when stdout is redirected"):
    // `--version` is clap's own generated flag (`#[command(version, ...)]` on `Cli` — see `main.rs`),
    // and this module's `#![deny(clippy::print_stdout)]` already makes a stray application `println!`
    // a compile error, but neither of those is *proof* that `--version` itself stays clean — this pins
    // it down directly rather than assuming.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .arg("--version")
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.trim() == format!("beyond-ai-agent {}", env!("CARGO_PKG_VERSION")),
        "stdout should be exactly the binary name and version, nothing else: {stdout:?}"
    );
    assert_eq!(
        output.stderr,
        b"",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_binary_help_flag_prints_usage_to_stdout_with_empty_stderr() {
    // F-L3 (pi: stdout-cleanliness.test.ts "prints plain --help to stdout when stdout is redirected"):
    // same idea as the `--version` case above, for clap's generated `--help`.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin).arg("--help").output().expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"), "stdout: {stdout}");
    assert_eq!(
        output.stderr,
        b"",
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_binary_json_help_flag_routes_usage_to_stderr_with_empty_stdout() {
    // F-L3 (pi: stdout-cleanliness.test.ts "keeps stdout empty for --mode json --help ..." / "keeps
    // stdout empty for -p --help ..."): unlike plain `run --help` above, `--json` marks `run`'s stdout
    // as the NDJSON `AgentEvent` stream (see `main.rs`'s `run_turn_once`) — the same one-frame-per-line
    // invariant `serve`'s NDJSON protocol depends on. clap's own `--help` short-circuit runs before any
    // application code, so it can't consult that fact on its own; `main.rs`'s `cli()` helper scans argv
    // for `run` + `--json` + a help flag and redirects clap's rendered help to stderr instead of
    // stdout when it does. No `--gateway-url`/`--key` needed: clap's help short-circuit fires before
    // any of that is ever read.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["run", "--json", "--help"])
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout,
        b"",
        "stdout must stay empty when --json --help combine: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Usage:"), "stderr: {stderr}");
}

#[test]
fn export_subcommand_renders_an_existing_session_file_with_no_gateway_or_key() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");

    // Create a session file the ordinary way, with a real (fake) model server — this part still
    // needs a gateway/key, exactly like any other `run`.
    let (base, _bodies) = spawn_model_server(vec![turn_text("the answer is 42")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let setup = run_cmd(bin)
        .args([
            "run",
            "what is the answer?",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        setup.status.success(),
        "session setup run failed.\nstderr: {}",
        String::from_utf8_lossy(&setup.stderr)
    );
    assert!(session_file.exists());

    // Now export that already-persisted session file directly — no --gateway-url/--key/--model at
    // all, proving the export subcommand is pure offline rendering of what's on disk, unlike `run
    // --export` (which only exports after a live model run completes).
    let export_path = dir.path().join("transcript.html");
    let output = Command::new(bin)
        .args([
            "export",
            session_file.to_str().unwrap(),
            export_path.to_str().unwrap(),
        ])
        .env("HOME", dir.path())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "export subcommand failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let html = std::fs::read_to_string(&export_path).expect("exported file must exist");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("what is the answer?"));
    assert!(html.contains("the answer is 42"));
}

#[test]
fn run_binary_thinking_flag_reaches_the_wire_request() {
    // pi-parity fix (M12): `--thinking` existed on `serve` but not `run` — there was no way to enable
    // extended thinking for a one-shot task at all.
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--thinking",
            "2048",
        ])
        .stdin(Stdio::null())
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
        bodies[0].contains(r#""budget_tokens":2048"#),
        "--thinking must set the wire request's thinking budget: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_cache_long_flag_reaches_the_wire_request() {
    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--cache-long",
        ])
        .stdin(Stdio::null())
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
        bodies[0].contains(r#""ttl":"1h""#),
        "--cache-long must select the 1-hour cache TTL on the wire request: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_bash_shell_path_flag_is_used_for_bash_calls() {
    let dir = tempfile::tempdir().unwrap();
    // A fake "shell" that just echoes a distinctive marker, proving the real command never ran through
    // the auto-resolved shell — only through this one.
    let fake_shell = dir.path().join("fake-shell.sh");
    std::fs::write(
        &fake_shell,
        "#!/bin/sh\necho ran-through-fake-shell-marker\n",
    )
    .unwrap();
    let mut perms = std::fs::metadata(&fake_shell).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
    std::fs::set_permissions(&fake_shell, perms).unwrap();

    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "echo real-command" }).to_string(),
    );
    let turn2 = turn_text("done");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "run a command",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--bash-shell-path",
            fake_shell.to_str().unwrap(),
            "--max-steps",
            "4",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
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
        bodies[1].contains("ran-through-fake-shell-marker"),
        "--bash-shell-path must route the command through the given shell, not the real command's \
         own output: {}",
        bodies[1]
    );
}
