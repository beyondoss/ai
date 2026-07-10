//! `run` e2e: a `subagent` task's own `output_schema` — the callable child.
//!
//! This is the gap the subagent work exposed: a child hands its parent a paragraph when it should hand
//! back `{"files_changed": [...]}`. The schema here is authored by the *parent model*, so the malformed
//! cases matter as much as the happy path: a bad schema must fail that one task with a message the
//! parent can act on, never panic and never take the fan-out down with it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{advertised_tools, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

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

fn scout_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_agent(
        dir.path(),
        "scout",
        "description: recon\ntools: read,grep\n",
        "You are SCOUT. Report what you find.",
    );
    dir
}

fn child_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "files_changed": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["files_changed"],
        "additionalProperties": false
    })
}

/// The parent's `subagent` call, delegating one task with an `output_schema`.
fn delegate(schema: Value) -> String {
    turn_tool_use(
        "call-1",
        "subagent",
        &json!({ "agent": "scout", "task": "find the entrypoint", "output_schema": schema })
            .to_string(),
    )
}

/// The `tool_result` the parent's *next* request carries — i.e. what the child handed back.
fn child_result(parent_final_request: &str) -> String {
    let body = common::body_json(parent_final_request);
    let messages = body["messages"].as_array().expect("messages");
    for m in messages.iter().rev() {
        if let Some(blocks) = m["content"].as_array() {
            for b in blocks {
                if b["type"] == "tool_result" {
                    return match &b["content"] {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                }
            }
        }
    }
    panic!("no tool_result in the parent's final request: {parent_final_request}");
}

#[test]
fn a_child_with_an_output_schema_returns_compact_json_instead_of_prose() {
    let dir = scout_dir();
    let (base, bodies) = spawn_model_server(vec![
        delegate(child_schema()),
        // The child answers through `structured_output` and its run terminates there.
        turn_tool_use(
            "tu_c",
            "structured_output",
            &json!({ "files_changed": ["src/main.rs", "src/lib.rs"] }).to_string(),
        ),
        turn_text("Done."),
    ]);

    let output = run_bin(dir.path(), &base, "what changed?");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 3, "parent, child, parent-final");

    // The child was told the contract, and given the parent's schema as the tool's input schema.
    let child = &bodies[1];
    assert!(
        advertised_tools(child)
            .iter()
            .any(|t| t == "structured_output"),
        "the child must have the tool: {:?}",
        advertised_tools(child)
    );
    assert!(child.contains("structured_output_protocol"));
    assert!(child.contains("files_changed"));

    // And the parent receives machine-parseable JSON, not a paragraph.
    let result = child_result(&bodies[2]);
    let value: Value = serde_json::from_str(&result)
        .unwrap_or_else(|e| panic!("the child's result must be JSON ({e}): {result}"));
    assert_eq!(value["files_changed"][0], "src/main.rs");
    assert_eq!(value["files_changed"][1], "src/lib.rs");
    assert!(value.get("_model_supports_vision").is_none());
}

#[test]
fn a_child_without_an_output_schema_still_returns_its_prose() {
    // The default is unchanged: no schema, no tool, no protocol, and the child's last assistant message
    // is what the parent sees.
    let dir = scout_dir();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "scout", "task": "find the entrypoint" }).to_string(),
        ),
        turn_text("CHILD-RECON-RESULT: main.rs"),
        turn_text("Done."),
    ]);

    let output = run_bin(dir.path(), &base, "where is the entrypoint?");
    assert!(output.status.success());

    let bodies = bodies.lock().unwrap();
    let child = &bodies[1];
    assert!(
        !advertised_tools(child)
            .iter()
            .any(|t| t == "structured_output")
    );
    assert!(!child.contains("structured_output_protocol"));
    assert!(child_result(&bodies[2]).contains("CHILD-RECON-RESULT"));
}

#[test]
fn a_child_that_never_calls_the_tool_fails_its_task_rather_than_returning_prose() {
    // The parent asked for a typed answer; a paragraph is not it. Reporting success here would hand the
    // parent a string where it expected an object.
    let dir = scout_dir();
    let (base, bodies) = spawn_model_server(vec![
        delegate(child_schema()),
        turn_text("I will explain in prose instead."),
        turn_text("Understood."),
    ]);

    let output = run_bin(dir.path(), &base, "what changed?");
    assert!(output.status.success(), "the parent's own run must survive");

    let bodies = bodies.lock().unwrap();
    let result = child_result(&bodies[2]);
    assert!(
        result.contains("without calling `structured_output`"),
        "the parent must be told the child broke its contract: {result}"
    );
    assert!(
        !result.contains("I will explain in prose instead"),
        "the child's prose must not be passed off as the typed answer: {result}"
    );
}

#[test]
fn a_model_authored_schema_that_cannot_compile_fails_only_that_task() {
    // `output_schema` comes from the *parent model*. A schema that won't compile must produce an error
    // the parent can read and retry from — never a panic, and never a dead run.
    let dir = scout_dir();
    // Structurally invalid: `"type"` must be a string or an array of strings.
    let bad = json!({ "type": "object", "properties": { "a": { "type": 123 } } });
    let (base, bodies) = spawn_model_server(vec![
        delegate(bad),
        // No child turn is scripted: the task must fail before a child model call is ever made.
        turn_text("I will fix the schema and retry."),
    ]);

    let output = run_bin(dir.path(), &base, "what changed?");
    assert!(
        output.status.success(),
        "a bad child schema must not kill the parent: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "the child must never have been given a model call"
    );
    let result = child_result(&bodies[1]);
    assert!(
        result.contains("invalid output_schema"),
        "the parent must be told its schema was the problem: {result}"
    );
}

#[test]
fn a_non_object_output_schema_is_rejected_as_invalid_tool_input() {
    // Caught at argument-parse time, before any child is spawned — the parent gets the ordinary
    // malformed-arguments feedback loop.
    let dir = scout_dir();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "agent": "scout", "task": "x", "output_schema": "not an object" }).to_string(),
        ),
        turn_text("I see, I will pass an object."),
    ]);

    let output = run_bin(dir.path(), &base, "what changed?");
    assert!(output.status.success());

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let result = child_result(&bodies[1]);
    assert!(
        result.contains("`output_schema` must be a JSON Schema object"),
        "got: {result}"
    );
}

#[test]
fn each_task_in_a_chain_carries_its_own_output_schema() {
    // Two children, one typed and one not: the schema is per-task, and `{previous}` still substitutes.
    let dir = scout_dir();
    write_agent(
        dir.path(),
        "writer",
        "description: writes\ntools: read\n",
        "You are WRITER.",
    );

    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "call-1",
            "subagent",
            &json!({ "chain": [
                { "agent": "scout", "task": "survey", "output_schema": child_schema() },
                { "agent": "writer", "task": "summarize {previous}" },
            ]})
            .to_string(),
        ),
        turn_tool_use(
            "tu_c1",
            "structured_output",
            &json!({ "files_changed": ["a.rs"] }).to_string(),
        ),
        turn_text("WRITER-PROSE-RESULT"),
        turn_text("Done."),
    ]);

    let output = run_bin(dir.path(), &base, "survey then summarize");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 4, "parent, scout, writer, parent-final");

    // The first child is typed...
    assert!(
        advertised_tools(&bodies[1])
            .iter()
            .any(|t| t == "structured_output")
    );
    // ...the second is not...
    assert!(
        !advertised_tools(&bodies[2])
            .iter()
            .any(|t| t == "structured_output")
    );
    // ...and the first child's JSON was substituted into the second's task via `{previous}`.
    assert!(
        bodies[2].contains("files_changed"),
        "the typed result must flow into the next task: {}",
        bodies[2]
    );
    assert!(child_result(&bodies[3]).contains("WRITER-PROSE-RESULT"));
}
