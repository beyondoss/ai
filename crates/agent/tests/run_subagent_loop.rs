//! `run` e2e: a child's loop semantics follow its parent's.
//!
//! - A child whose run ends on a refusal or on an unrecovered output-limit stop has not finished: it
//!   is reported as a failure and its worktree is preserved, never merged.
//! - A child runs under the parent's loop configuration (`--max-steps`, `--thinking`, …), not
//!   `Agent::new`'s defaults.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::Path;
use std::process::{Command, Stdio};

use common::{body_json, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

fn git(dir: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap()
        .status
        .success();
    assert!(ok, "git {args:?} failed");
}

/// A repo with a worktree-isolated `writer`, whose system prompt carries `CHILD-MARKER`.
fn repo() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    git(p, &["init", "--quiet", "-b", "main"]);
    git(p, &["config", "user.name", "t"]);
    git(p, &["config", "user.email", "t@t"]);
    std::fs::write(p.join("target.txt"), "original\n").unwrap();
    let agents = p.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("writer.md"),
        "---\nname: writer\ndescription: edits files\ntools: read,edit,write\nisolation: worktree\n---\nYou are CHILD-MARKER.\n",
    )
    .unwrap();
    git(p, &["add", "-A"]);
    git(p, &["commit", "--quiet", "-m", "init"]);
    dir
}

fn run(dir: &Path, cache: &Path, base: &str, extra: &[&str]) -> std::process::Output {
    run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "go",
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
        ])
        .args(extra)
        .env("XDG_CACHE_HOME", cache)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary")
}

/// The `tool_result` block the parent got back for `call_id`.
fn tool_result(bodies: &[String], call_id: &str) -> Value {
    for raw in bodies.iter().rev() {
        let body = body_json(raw);
        for m in body["messages"].as_array().into_iter().flatten() {
            for c in m["content"].as_array().into_iter().flatten() {
                if c["type"] == "tool_result" && c["tool_use_id"] == call_id {
                    return c.clone();
                }
            }
        }
    }
    panic!("no tool_result for {call_id}");
}

/// A text turn that ends with `stop_reason`, as the provider sends it.
fn turn_ending(text: &str, stop_reason: &str) -> String {
    beyond_ai_test_support::sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// A worktree child writes half its work, then its run ends on `stop_reason`: the result must be a
/// failure that names the preserved checkout, and nothing may reach the parent's tree.
fn assert_unfinished_child_is_a_failure(stop_reason: &str, says: &str) {
    let dir = repo();
    let cache = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({"agent": "writer", "task": "two-step migration"}).to_string(),
        ),
        turn_tool_use(
            "call-2",
            "write",
            &json!({"path": "half.txt", "content": "step 1 of 2\n"}).to_string(),
        ),
        turn_ending("I can't finish the rest of this", stop_reason),
        turn_text("Done."),
    ]);
    let out = run(dir.path(), cache.path(), &base, &[]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let result = tool_result(&bodies.lock().unwrap(), "call-1");
    let text = result["content"].to_string();
    assert_eq!(result["is_error"], true, "{result}");
    assert!(text.contains(says), "{text}");
    assert!(text.contains("worktree preserved at"), "{text}");
    assert!(
        !dir.path().join("half.txt").exists(),
        "an unfinished child's edits must not be merged"
    );
}

#[test]
fn a_child_that_ends_on_a_refusal_is_a_failure_and_is_not_merged() {
    assert_unfinished_child_is_a_failure("refusal", "refused");
}

#[test]
fn a_child_cut_off_at_the_output_limit_is_a_failure_and_is_not_merged() {
    assert_unfinished_child_is_a_failure("max_tokens", "cut off at the output-token limit");
}

/// The requests the child made, as parsed bodies.
fn child_requests(bodies: &[String]) -> Vec<Value> {
    bodies
        .iter()
        .filter(|b| b.contains("\"messages\"") && b.contains("CHILD-MARKER"))
        .map(|b| body_json(b))
        .collect()
}

#[test]
fn a_child_runs_under_the_parents_step_budget() {
    let dir = repo();
    let cache = tempfile::tempdir().unwrap();
    // A child that would read forever. Under the parent's `--max-steps 1` it gets its one step plus
    // the wrap-up grant, and no more.
    let mut script = vec![turn_tool_use(
        "call-1",
        "subagent",
        &json!({"agent": "writer", "task": "read"}).to_string(),
    )];
    for i in 0..12 {
        script.push(turn_tool_use(
            &format!("r-{i}"),
            "read",
            &json!({"path": "target.txt"}).to_string(),
        ));
    }
    script.push(turn_text("done"));
    let (base, bodies) = spawn_model_server(script);
    let _ = run(dir.path(), cache.path(), &base, &["--max-steps", "1"]);
    let child = child_requests(&bodies.lock().unwrap());
    let ceiling = 1 + agent_core::step_budget::WRAP_UP_STEPS as usize;
    assert!(
        !child.is_empty() && child.len() <= ceiling,
        "the child must stop within the parent's step budget ({ceiling} requests), made {}",
        child.len()
    );
}

#[test]
fn a_child_inherits_the_parents_thinking_budget() {
    let dir = repo();
    let cache = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({"agent": "writer", "task": "think"}).to_string(),
        ),
        turn_text("thought"),
        turn_text("Done."),
    ]);
    let out = run(dir.path(), cache.path(), &base, &["--thinking", "2048"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let child = child_requests(&bodies.lock().unwrap());
    assert_eq!(child.len(), 1);
    assert_eq!(
        child[0]["thinking"]["budget_tokens"], 2048,
        "the child must think like its parent: {}",
        child[0]["thinking"]
    );
}
