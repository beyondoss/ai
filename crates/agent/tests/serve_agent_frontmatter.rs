//! `serve` e2e: the workspace `AGENTS.md`'s frontmatter narrows the tools and caps the steps of a
//! local session, an explicit flag beats it, and an untrusted project's file configures nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;

use common::{
    SpawnGuarded, advertised_tools, body_json, read_until_response, serve_cmd, spawn_model_server,
    turn_text, turn_tool_use,
};
use serde_json::json;

const AGENTS_MD: &str =
    "---\ntools: read, grep\nmax-steps: 1\n---\nLOCAL-HOUSE-STYLE: answer tersely.\n";

/// Run one prompt against a model that never stops calling tools; returns every request it saw.
fn requests_until_the_run_ends(trust: bool, flags: &[&str]) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("AGENTS.md"), AGENTS_MD).unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let mut turns: Vec<String> = (0..30)
        .map(|i| turn_tool_use(&format!("t{i}"), "read", r#"{"path":"AGENTS.md"}"#))
        .collect();
    turns.push(turn_text("never reached"));
    let (base, requests) = spawn_model_server(turns);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut cmd = serve_cmd(bin, &base, &session_file);
    cmd.args(flags).current_dir(dir.path());
    if trust {
        cmd.arg("--trust-project");
    }
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        json!({"type":"prompt","id":"p1","message":"go"})
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    requests.lock().unwrap().clone()
}

fn without_memory(raw: &str) -> Vec<String> {
    let mut tools = advertised_tools(raw);
    tools.retain(|t| t != "memory");
    tools.sort();
    tools
}

#[test]
fn frontmatter_narrows_the_tools_and_caps_the_steps_of_a_local_session() {
    let requests = requests_until_the_run_ends(true, &[]);
    for raw in &requests {
        assert_eq!(without_memory(raw), vec!["grep", "read"]);
    }
    assert!(
        requests.len() < 10,
        "max-steps: 1 must stop the run, got {} model turns",
        requests.len()
    );
    let system = body_json(&requests[0])["system"].to_string();
    assert!(system.contains("LOCAL-HOUSE-STYLE"), "{system}");
    assert!(!system.contains("max-steps"), "{system}");
}

#[test]
fn an_explicit_flag_beats_the_frontmatter() {
    let requests = requests_until_the_run_ends(true, &["--tools", "grep", "--max-steps", "20"]);
    assert_eq!(without_memory(&requests[0]), vec!["grep"]);
    assert!(
        requests.len() > 10,
        "--max-steps 20 beats max-steps: 1, got {} model turns",
        requests.len()
    );
}

#[test]
fn an_untrusted_projects_frontmatter_configures_nothing() {
    let requests = requests_until_the_run_ends(false, &["--max-steps", "12"]);
    assert!(advertised_tools(&requests[0]).contains(&"bash".to_string()));
    let system = body_json(&requests[0])["system"].to_string();
    assert!(system.contains("LOCAL-HOUSE-STYLE"), "{system}");
    assert!(!system.contains("max-steps"), "{system}");
}
