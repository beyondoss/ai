//! `run` e2e: the workspace `AGENTS.md`'s YAML frontmatter configures the agent — its model, its tool
//! allow-list, its turn cap — and never reaches the model as prompt text. An explicit flag beats the
//! file, and an untrusted project's file configures nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;

use common::{advertised_tools, body_json, run_cmd, spawn_model_server, turn_text, turn_tool_use};

const AGENTS_MD: &str = "---\nmodel: claude-frontmatter\ntools: read, grep\nmax-steps: 1\n---\nAlways answer in haiku.\n";

fn run(base: &str, dir: &std::path::Path, extra: &[&str]) -> std::process::Output {
    run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "do a big task",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--no-session-persistence",
        ])
        .args(extra)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("AGENTS.md"), AGENTS_MD).unwrap();
    dir
}

fn script(tool_turns: usize) -> Vec<String> {
    let mut turns: Vec<String> = (0..tool_turns)
        .map(|i| turn_tool_use(&format!("t{i}"), "read", r#"{"path":"AGENTS.md"}"#))
        .collect();
    turns.push(turn_text("wrapped up"));
    turns
}

/// The advertised tools, sorted, without `memory` (which the filter does not govern).
fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.retain(|t| t != "memory");
    v.sort();
    v
}

#[test]
fn frontmatter_picks_the_model_filters_the_tools_and_caps_the_steps() {
    let dir = workspace();
    let (base, bodies) = spawn_model_server(script(1));
    let out = run(&base, dir.path(), &["--trust-project"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");

    let bodies = bodies.lock().unwrap();
    // `max-steps: 1` is one model turn plus the wrap-up turn.
    assert!(
        stderr.contains("--max-steps ceiling (1)"),
        "max-steps: 1 must stop the run: {stderr}"
    );
    assert_eq!(bodies.len(), 2);
    for raw in bodies.iter() {
        assert_eq!(body_json(raw)["model"], "claude-frontmatter");
        assert_eq!(
            sorted(advertised_tools(raw)),
            vec!["grep".to_string(), "read".to_string()]
        );
    }
    // The instructions reach the model; the frontmatter that configured it does not.
    let system = body_json(&bodies[0])["system"].to_string();
    assert!(system.contains("Always answer in haiku."), "{system}");
    assert!(!system.contains("max-steps"), "{system}");
    assert!(!system.contains("claude-frontmatter"), "{system}");
}

#[test]
fn an_explicit_flag_beats_the_frontmatter() {
    let dir = workspace();
    let (base, bodies) = spawn_model_server(script(2));
    let out = run(
        &base,
        dir.path(),
        &[
            "--trust-project",
            "--model",
            "claude-flag",
            "--tools",
            "read",
            "--max-steps",
            "2",
        ],
    );
    assert!(out.status.success());

    let bodies = bodies.lock().unwrap();
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--max-steps ceiling (2)"),
        "--max-steps 2 beats max-steps: 1"
    );
    assert_eq!(bodies.len(), 3);
    for raw in bodies.iter() {
        assert_eq!(body_json(raw)["model"], "claude-flag");
        assert_eq!(sorted(advertised_tools(raw)), vec!["read".to_string()]);
    }
}

#[test]
fn an_untrusted_projects_frontmatter_configures_nothing_but_is_still_not_prompt_text() {
    let dir = workspace();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);
    let out = run(&base, dir.path(), &["--model", "claude-flag"]);
    assert!(out.status.success());

    let bodies = bodies.lock().unwrap();
    let body = body_json(&bodies[0]);
    assert!(
        advertised_tools(&bodies[0]).contains(&"bash".to_string()),
        "the tool allow-list must not apply"
    );
    let system = body["system"].to_string();
    assert!(system.contains("Always answer in haiku."), "{system}");
    assert!(!system.contains("max-steps"), "{system}");
}
