//! `run` e2e: a failed worktree child's result names where its checkout was preserved, in `single`
//! and `chain` as well as `parallel`. Only `parallel` used to print the note; the other two dropped
//! the path to the only copy of the child's work.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

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
        "---\nname: writer\ndescription: edits files\ntools: read,edit,write\nisolation: worktree\n---\nYou are writer.\n",
    )
    .unwrap();
    git(p, &["add", "-A"]);
    git(p, &["commit", "--quiet", "-m", "init"]);
    dir
}

/// The text of the `tool_result` the parent got back for `call_id`.
fn tool_result(bodies: &[String], call_id: &str) -> String {
    for raw in bodies.iter().rev() {
        let body = body_json(raw);
        for m in body["messages"].as_array().into_iter().flatten() {
            for c in m["content"].as_array().into_iter().flatten() {
                if c["type"] == "tool_result" && c["tool_use_id"] == call_id {
                    return match &c["content"] {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                }
            }
        }
    }
    panic!("no tool_result for {call_id}");
}

/// Run the parent with a `subagent` call of `input`, whose one child writes `half.txt` and then ends
/// in prose although it was asked for structured output — a failure, so its checkout is preserved.
fn failed_child_result(input: Value) -> String {
    let dir = repo();
    let cache = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("call-1", "subagent", &input.to_string()),
        turn_tool_use(
            "call-2",
            "write",
            &json!({"path": "half.txt", "content": "half done\n"}).to_string(),
        ),
        turn_text("I wrote half of it."),
        turn_text("Done."),
    ]);
    let out = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "run",
            "go",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
        ])
        .env("XDG_CACHE_HOME", cache.path())
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let result = tool_result(&bodies.lock().unwrap(), "call-1");
    let path = result
        .split("worktree preserved at ")
        .nth(1)
        .unwrap_or_else(|| panic!("the result must name the preserved checkout: {result}"))
        .split(|c: char| c == ']' || c.is_whitespace())
        .next()
        .unwrap()
        .to_string();
    assert!(
        Path::new(&path).join("half.txt").exists(),
        "the named path must hold the child's work: {path}"
    );
    result
}

fn schema() -> Value {
    json!({"type": "object", "properties": {"done": {"type": "boolean"}}})
}

#[test]
fn a_failed_single_child_names_its_preserved_checkout() {
    failed_child_result(json!({"agent": "writer", "task": "step one", "output_schema": schema()}));
}

#[test]
fn a_failed_chain_step_names_its_preserved_checkout() {
    let result = failed_child_result(json!({"chain": [
        {"agent": "writer", "task": "step one", "output_schema": schema()},
    ]}));
    assert!(result.contains("chain stopped at step 1"), "{result}");
}
