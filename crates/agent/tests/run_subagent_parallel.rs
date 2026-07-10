//! `run` e2e: `subagent` parallel mode through the real binary. Parallel children race to the gateway
//! in a nondeterministic order, so this uses `spawn_model_server_routed` (match by request body) rather
//! than the FIFO `spawn_model_server`. Each child's system prompt carries a unique marker the router
//! keys on.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server_routed, turn_text, turn_tool_use};
use serde_json::json;

fn write_agent(dir: &std::path::Path, name: &str, body: &str) {
    let agents = dir.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join(format!("{name}.md")),
        format!(
            "---\nname: {name}\ndescription: test {name}\ntools: read,grep,find,ls\n---\n{body}\n"
        ),
    )
    .unwrap();
}

#[test]
fn parallel_runs_multiple_children_and_returns_results_in_task_order() {
    let dir = tempfile::tempdir().unwrap();
    // Each agent's body is a unique marker; the router matches a child's request by it.
    write_agent(dir.path(), "alpha", "You are AGENT-ALPHA-MARKER.");
    write_agent(dir.path(), "beta", "You are AGENT-BETA-MARKER.");
    write_agent(dir.path(), "gamma", "You are AGENT-GAMMA-MARKER.");

    // The router serves:
    //  - the parent's first turn (which asks for the subagent fan-out) — matched by the user prompt,
    //  - each child — matched by its system-prompt marker,
    //  - the parent's final turn — matched by a tool-result marker only it carries.
    // Route order matters: routes are first-match-wins, and the parent's ORIGINAL prompt marker stays in
    // its history on every later turn. So the parent's *final* turn (which additionally carries the
    // fan-out results) must be matched by a more-specific route placed EARLIER than the prompt marker —
    // otherwise the parent re-triggers the fan-out forever and hits max-steps.
    let (base, _bodies) = spawn_model_server_routed(
        vec![
            // Parent's final turn: the only request that carries a child's result back as a tool result.
            ("RESULT-ALPHA".to_string(), turn_text("all done")),
            // The three children, matched by their unique system-prompt markers.
            ("AGENT-ALPHA-MARKER".to_string(), turn_text("RESULT-ALPHA")),
            ("AGENT-BETA-MARKER".to_string(), turn_text("RESULT-BETA")),
            ("AGENT-GAMMA-MARKER".to_string(), turn_text("RESULT-GAMMA")),
            // Parent's first turn: has the prompt marker but no results yet, so it falls through to here.
            (
                "PARENT-PROMPT-MARKER".to_string(),
                turn_tool_use(
                    "call-1",
                    "subagent",
                    &json!({ "tasks": [
                        { "agent": "alpha", "task": "task a" },
                        { "agent": "beta", "task": "task b" },
                        { "agent": "gamma", "task": "task c" }
                    ] })
                    .to_string(),
                ),
            ),
        ],
        // Any unmatched request (shouldn't happen) gets a harmless terminal turn.
        turn_text("fallback"),
    );

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "PARENT-PROMPT-MARKER: do the parallel work",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
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

    // The parent's final turn (the request whose body contains all three results) must list them in
    // task order: alpha, beta, gamma — regardless of which child's HTTP request landed first.
    let bodies = _bodies.lock().unwrap();
    let parent_final = bodies
        .iter()
        .find(|b| b.contains("RESULT-ALPHA") && b.contains("RESULT-GAMMA"))
        .expect("the parent's final turn carrying all three results");
    let (pa, pb, pg) = (
        parent_final.find("RESULT-ALPHA").unwrap(),
        parent_final.find("RESULT-BETA").unwrap(),
        parent_final.find("RESULT-GAMMA").unwrap(),
    );
    assert!(
        pa < pb && pb < pg,
        "parallel results must be in task order in the tool result, not completion order:\n{parent_final}"
    );
}
