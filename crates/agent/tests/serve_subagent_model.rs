//! `serve` e2e: a child with no `model:` of its own inherits the model the parent is on **now**.
//! The subagent context outlives every model switch, and it used to carry a copy of the startup
//! model, so after `set_model` every delegated child still ran on the model the session began with.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;

use common::{
    SpawnGuarded, body_json, read_until_response, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::json;

#[test]
fn a_child_dispatched_after_set_model_runs_on_the_new_model() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let agents = dir.path().join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("scout.md"),
        "---\nname: scout\ndescription: recon\ntools: read\n---\nYou are SCOUT-MARKER.\n",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({"agent": "scout", "task": "look"}).to_string(),
        ),
        turn_text("CHILD"),
        turn_text("parent done"),
    ]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = common::serve_cmd(bin, &base, &session_file)
        .arg("--trust-project")
        .current_dir(dir.path())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        json!({"type": "set_model", "model": "claude-test-2"})
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_model");
    writeln!(stdin, "{}", json!({"type": "prompt", "message": "explore"})).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    // (model, is the child's request) for every model request the server saw.
    let models: Vec<(String, bool)> = bodies
        .lock()
        .unwrap()
        .iter()
        .filter(|raw| raw.contains("\"messages\""))
        .map(|raw| {
            (
                body_json(raw)["model"].as_str().unwrap_or("?").to_string(),
                raw.contains("SCOUT-MARKER"),
            )
        })
        .collect();
    let child_models: Vec<&str> = models
        .iter()
        .filter(|(_, c)| *c)
        .map(|(m, _)| m.as_str())
        .collect();
    assert!(!child_models.is_empty(), "the child ran: {models:?}");
    assert!(
        models.iter().all(|(m, _)| m == "claude-test-2"),
        "parent and child must both be on the model the parent switched to: {models:?}"
    );
}

#[test]
fn a_child_dispatched_after_set_thinking_thinks_like_its_parent() {
    // The loop settings ride the same live cell as the model: a `set_thinking` mid-session reaches
    // the next delegated child, which used to run on `Agent::new`'s defaults (no thinking at all).
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let agents = dir.path().join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("scout.md"),
        "---\nname: scout\ndescription: recon\ntools: read\n---\nYou are SCOUT-MARKER.\n",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({"agent": "scout", "task": "look"}).to_string(),
        ),
        turn_text("CHILD"),
        turn_text("parent done"),
    ]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = common::serve_cmd(bin, &base, &session_file)
        .arg("--trust-project")
        .current_dir(dir.path())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    writeln!(stdin, "{}", json!({"type": "set_thinking", "budget": 4096})).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "set_thinking");
    writeln!(stdin, "{}", json!({"type": "prompt", "message": "explore"})).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();

    let bodies = bodies.lock().unwrap();
    let child_body = bodies
        .iter()
        .find(|raw| raw.contains("\"messages\"") && raw.contains("SCOUT-MARKER"))
        .map(|raw| body_json(raw))
        .expect("the child ran");
    assert_eq!(
        child_body["thinking"]["budget_tokens"], 4096,
        "{}",
        child_body["thinking"]
    );
}
