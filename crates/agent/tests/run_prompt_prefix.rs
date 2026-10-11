//! `run` e2e: the first request's prompt prefix — the budgeted skills listing, loading a listed
//! skill by name through `read`, and byte-stability of the system prompt + tools across runs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;
use std::process::{Command, Stdio};

use common::{body_json, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::Value;

/// A HOME laid out like a real Claude Code one: a plain user skill and a synced one under
/// `~/.claude/skills/synced/<uuid>_<uuid>/`, whose absolute path the listing must not carry.
fn claude_home() -> tempfile::TempDir {
    let home = tempfile::tempdir().unwrap();
    let skills = home.path().join(".claude/skills");
    let write = |dir: &Path, manifest: &str| {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), manifest).unwrap();
    };
    write(
        &skills.join("lint"),
        "---\nname: lint\ndescription: Run the project linter and fix what it reports.\n---\nLint body.\n",
    );
    let deploy = skills.join("synced/0e5f-uuid_9b4c-uuid/deploy");
    write(
        &deploy,
        "---\nname: deploy\ndescription: Use when the user asks to ship or deploy a service.\n---\n\
         DEPLOY-SKILL-BODY. See reference.md.\n",
    );
    std::fs::write(deploy.join("reference.md"), "DEPLOY-REFERENCE\n").unwrap();
    home
}

fn run_with_home(home: &Path, cwd: &Path, base: &str, model: &str) -> std::process::Output {
    run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home)
        .args([
            "run",
            "ship it",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            model,
            "--no-session-persistence",
        ])
        .current_dir(cwd)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary")
}

fn system_text(body: &Value) -> String {
    match &body["system"] {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[test]
fn the_model_loads_a_listed_skill_by_name_through_read() {
    // The listing names each skill as `skill:<name>` — never its absolute manifest path — and a
    // model that picks one loads it with `read` at that name: the body, plus the skill's directory
    // so a relative reference (or a script for `bash`) resolves.
    let home = claude_home();
    let cwd = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "read", r#"{"path":"skill:deploy"}"#),
        turn_tool_use("t2", "read", r#"{"path":"skill:deploy/reference.md"}"#),
        turn_text("done"),
    ]);
    let output = run_with_home(home.path(), cwd.path(), &base, "claude-test");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    let system = system_text(&body_json(&bodies[0]));
    assert!(
        system.contains("- deploy: Use when the user asks to ship or deploy a service.\n"),
        "{system}"
    );
    assert!(system.contains("`skill:<name>`"), "{system}");
    let home_str = home.path().to_str().unwrap();
    assert!(
        !system.contains(&format!("{home_str}/.claude/skills")),
        "the listing must not carry absolute skill paths: {system}"
    );
    let deploy_dir = home
        .path()
        .join(".claude/skills/synced/0e5f-uuid_9b4c-uuid/deploy");
    assert!(bodies[1].contains("DEPLOY-SKILL-BODY"), "{}", bodies[1]);
    assert!(
        bodies[1].contains(&format!("[skill directory: {}]", deploy_dir.display())),
        "{}",
        bodies[1]
    );
    assert!(bodies[2].contains("DEPLOY-REFERENCE"), "{}", bodies[2]);
}

#[test]
fn two_runs_send_a_byte_identical_system_prompt_and_tool_set() {
    // The prefix is prompt-cached: anything nondeterministic in it (listing order, a budget choice
    // that depends on more than the inputs) re-bills the whole prefix on every run.
    let home = claude_home();
    let cwd = tempfile::tempdir().unwrap();
    let first_request = || {
        let (base, bodies) = spawn_model_server(vec![turn_text("done")]);
        let output = run_with_home(home.path(), cwd.path(), &base, "claude-test");
        assert!(output.status.success());
        let body = body_json(&bodies.lock().unwrap()[0]);
        (body["system"].to_string(), body["tools"].to_string())
    };
    let (system_a, tools_a) = first_request();
    let (system_b, tools_b) = first_request();
    assert!(system_a.contains("<available_skills>"), "{system_a}");
    assert_eq!(system_a, system_b, "system prompt differs between runs");
    assert_eq!(tools_a, tools_b, "tool set differs between runs");
}

// ---- Live: a real model must choose and load a skill. Bills real requests; ignored in CI. ----

const RELEASE_SKILL: &str = "---\nname: release-train\ndescription: Use when the user asks to \
    cut, prepare or ship a release of a service.\n---\nTo answer, reply with exactly the line \
    RELEASE-TRAIN-7 and nothing else.\n";

fn live(prompt: &str) -> (bool, String, String) {
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    for (name, manifest) in [
        ("release-train", RELEASE_SKILL),
        (
            "pdf",
            "---\nname: pdf\ndescription: Read, merge or split PDF files.\n---\nPDF body.\n",
        ),
    ] {
        let dir = home.path().join(".claude/skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), manifest).unwrap();
    }
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .env("AI_PROVIDER", "anthropic")
        .env("AI_DIRECT", "1")
        .args([
            "run",
            prompt,
            "--model",
            "claude-haiku-4-5",
            "--max-steps",
            "6",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn live agent");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn require_anthropic_key() -> bool {
    std::env::var("ANTHROPIC_API_KEY").is_ok_and(|k| !k.trim().is_empty())
}

#[test]
#[ignore = "live Anthropic; needs ANTHROPIC_API_KEY"]
fn live_model_selects_a_skill_from_its_description() {
    if !require_anthropic_key() {
        eprintln!("ANTHROPIC_API_KEY unset — skipping");
        return;
    }
    let (ok, stdout, stderr) = live("We're ready to ship v2 of the billing service. What's next?");
    assert!(ok, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("RELEASE-TRAIN-7"),
        "stdout: {stdout}\nstderr: {stderr}"
    );
}

#[test]
#[ignore = "live Anthropic; needs ANTHROPIC_API_KEY"]
fn live_model_loads_a_skill_invoked_by_name() {
    if !require_anthropic_key() {
        eprintln!("ANTHROPIC_API_KEY unset — skipping");
        return;
    }
    let (ok, stdout, stderr) = live("Use the release-train skill.");
    assert!(ok, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("RELEASE-TRAIN-7"),
        "stdout: {stdout}\nstderr: {stderr}"
    );
}
