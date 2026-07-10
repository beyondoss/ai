//! `run` e2e: the `subagent` tool driven through the real binary against a mock gateway. Single and
//! chain modes here (deterministic request order); parallel is in `run_subagent_parallel.rs` (needs a
//! body-matching mock). Security-policy inheritance is in `subagent_policy.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::json;

/// Write an agent definition into `<dir>/.claude/agents/<name>.md`.
fn write_agent(dir: &std::path::Path, name: &str, frontmatter: &str, body: &str) {
    let agents = dir.join(".claude/agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join(format!("{name}.md")),
        format!("---\nname: {name}\n{frontmatter}---\n{body}\n"),
    )
    .unwrap();
}

fn run_bin(dir: &std::path::Path, base: &str, message: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    run_cmd(bin)
        .args([
            "run",
            message,
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-session-persistence",
        ])
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary")
}

fn assert_ok(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn single_mode_runs_a_child_and_returns_its_result_to_the_parent() {
    let dir = tempfile::tempdir().unwrap();
    write_agent(
        dir.path(),
        "scout",
        "description: recon\ntools: read,grep,find,ls\n",
        "You are SCOUT-SYSTEM-MARKER. Report what you find.",
    );

    // Parent turn: call subagent. Child turn: produce a result. Parent turn: finish.
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "scout", "task": "find the entrypoint" }).to_string(),
        ),
        turn_text("CHILD-RECON-RESULT: main.rs"),
        turn_text("Done — the entrypoint is main.rs."),
    ]);

    let output = run_bin(dir.path(), &base, "where is the entrypoint?");
    assert_ok(&output);

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        3,
        "expected parent, child, parent-final requests"
    );

    // The middle request is the child's: it must carry the agent def's body as its system prompt and
    // the delegated task as its user message.
    let child = &bodies[1];
    assert!(
        child.contains("SCOUT-SYSTEM-MARKER"),
        "child system prompt missing the def body: {child}"
    );
    assert!(
        child.contains("find the entrypoint"),
        "child must receive the delegated task: {child}"
    );

    // The parent's final turn must have been handed the child's result as the tool result.
    let parent_final = &bodies[2];
    assert!(
        parent_final.contains("CHILD-RECON-RESULT"),
        "the child's output must return to the parent as a tool result: {parent_final}"
    );
}

#[test]
fn the_subagent_tool_and_available_agents_are_advertised_to_the_model() {
    let dir = tempfile::tempdir().unwrap();
    write_agent(
        dir.path(),
        "scout",
        "description: does recon work\n",
        "You are scout.",
    );

    // One turn: the model just answers. We only care what the FIRST request advertised.
    let (base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let output = run_bin(dir.path(), &base, "hello");
    assert_ok(&output);

    let first = &bodies.lock().unwrap()[0];
    // The tool is registered (its schema is in the request's tool list)…
    assert!(
        first.contains("subagent"),
        "subagent tool must be advertised: {first}"
    );
    // …and the agent is named in the system prompt's <available_agents> block.
    assert!(
        first.contains("available_agents"),
        "the <available_agents> block must be present: {first}"
    );
    assert!(
        first.contains("does recon work"),
        "the agent's description must be listed: {first}"
    );
}

#[test]
fn no_agent_definitions_means_no_subagent_tool() {
    // A project with no `.claude/agents` must not advertise a `subagent` tool the model can't use.
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let output = run_bin(dir.path(), &base, "hello");
    assert_ok(&output);

    let first = &bodies.lock().unwrap()[0];
    assert!(
        !first.contains("available_agents"),
        "no agents ⇒ no block: {first}"
    );
    // The tool name must not appear in the advertised tool schemas. (Guard against a false negative from
    // the word appearing incidentally by checking the tool-definition shape.)
    assert!(
        !first.contains("\"name\":\"subagent\"") && !first.contains("\"name\": \"subagent\""),
        "subagent tool must not be registered when no agents exist: {first}"
    );
}

#[test]
fn chain_mode_feeds_each_step_the_previous_steps_output() {
    let dir = tempfile::tempdir().unwrap();
    write_agent(
        dir.path(),
        "scout",
        "description: recon\n",
        "You are scout.",
    );
    write_agent(
        dir.path(),
        "planner",
        "description: planning\n",
        "You are planner.",
    );

    // Parent: chain[scout, planner]. scout child. planner child. Parent: finish.
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "chain": [
                { "agent": "scout", "task": "recon the repo" },
                { "agent": "planner", "task": "using {previous}, write a plan" }
            ] })
            .to_string(),
        ),
        turn_text("SCOUT-FINDINGS: it's a Rust workspace"),
        turn_text("PLAN: step one, step two"),
        turn_text("Here is the plan."),
    ]);

    let output = run_bin(dir.path(), &base, "plan a change");
    assert_ok(&output);

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 4);
    // The planner (3rd request) must have seen scout's findings substituted for {previous}.
    let planner = &bodies[2];
    assert!(
        planner.contains("using SCOUT-FINDINGS: it's a Rust workspace, write a plan"),
        "planner must receive the substituted task: {planner}"
    );
    // The parent's final turn gets only the LAST step's output.
    let parent_final = &bodies[3];
    assert!(parent_final.contains("PLAN: step one"), "{parent_final}");
    assert!(
        !parent_final.contains("SCOUT-FINDINGS"),
        "chain returns only the last step's output, not intermediates: {parent_final}"
    );
}
