//! `run` e2e: the `todo` tool driven through the real binary — registration, the system-prompt
//! protocol section, the checklist a call hands back, the structured `details` a UI renders, the
//! validation errors the model is expected to correct, and the `--exclude-tools` opt-out.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn item(content: &str, status: &str) -> Value {
    json!({ "content": content, "activeForm": format!("{content}ing"), "status": status })
}

fn todo_turn(id: &str, todos: Value) -> String {
    turn_tool_use(id, "todo", &json!({ "todos": todos }).to_string())
}

/// Drive `run --json` against a scripted upstream; return (event lines, raw request bodies, exit ok).
fn run_json(turns: Vec<String>, extra_args: &[&str]) -> (Vec<Value>, Vec<String>, bool) {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(turns);

    let mut cmd = run_cmd(BIN);
    cmd.args([
        "run",
        "do the work",
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--max-steps",
        "6",
        "--no-session-persistence",
        "--json",
    ])
    .args(extra_args)
    .current_dir(dir.path());

    let output = cmd.output().expect("spawn binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let events: Vec<Value> = stdout
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let bodies = bodies.lock().unwrap().clone();
    (events, bodies, output.status.success())
}

fn events_of<'a>(events: &'a [Value], kind: &str) -> Vec<&'a Value> {
    events
        .iter()
        .filter(|e| e["kind"] == kind)
        .collect::<Vec<_>>()
}

#[test]
fn a_todo_call_returns_a_checklist_and_streams_the_list_as_structured_details() {
    let todos = json!([
        item("Parse the config", "completed"),
        item("Wire the retry loop", "in_progress"),
        item("Add tests", "pending"),
    ]);
    let (events, _bodies, ok) = run_json(
        vec![todo_turn("toolu_1", todos.clone()), turn_text("all done")],
        &[],
    );
    assert!(ok, "run should exit 0");

    // What the model reads back: a compact checklist, not a JSON echo.
    let ends = events_of(&events, "tool_end");
    let end = ends.iter().find(|e| e["name"] == "todo").expect("tool_end");
    assert_eq!(end["is_error"], false);
    assert_eq!(
        end["result"],
        "Todos updated (3 items):\n\
         [x] Parse the config\n\
         [>] Wire the retry loop\n\
         [ ] Add tests"
    );

    // What a UI renders: the raw array, passed through `ToolProgress.details` verbatim.
    let progress = events_of(&events, "tool_progress");
    let p = progress
        .iter()
        .find(|e| e["name"] == "todo")
        .expect("a tool_progress for todo");
    assert_eq!(p["details"]["todos"], todos);
    assert!(
        p["snapshot"]
            .as_str()
            .unwrap()
            .contains("[>] Wire the retry loop")
    );
}

#[test]
fn an_empty_list_clears_the_plan_rather_than_erroring() {
    let (events, _bodies, ok) = run_json(
        vec![todo_turn("toolu_1", json!([])), turn_text("finished")],
        &[],
    );
    assert!(ok);
    let ends = events_of(&events, "tool_end");
    let end = ends.iter().find(|e| e["name"] == "todo").unwrap();
    assert_eq!(end["is_error"], false);
    assert_eq!(end["result"], "Todo list cleared.");
}

#[test]
fn two_in_progress_items_are_rejected_and_the_model_can_correct_itself() {
    // The whole point of returning `InvalidInput` rather than silently accepting: the loop feeds the
    // reason back as an error `tool_result` and the run continues, so the model fixes its own call.
    let bad = json!([item("First", "in_progress"), item("Second", "in_progress")]);
    let good = json!([item("First", "in_progress"), item("Second", "pending")]);
    let (events, _bodies, ok) = run_json(
        vec![
            todo_turn("toolu_1", bad),
            todo_turn("toolu_2", good),
            turn_text("corrected and done"),
        ],
        &[],
    );
    assert!(ok, "a rejected tool call must not fail the run");

    let ends = events_of(&events, "tool_end");
    let todo_ends: Vec<_> = ends.iter().filter(|e| e["name"] == "todo").collect();
    assert_eq!(todo_ends.len(), 2, "the model retried once");

    assert_eq!(todo_ends[0]["is_error"], true);
    let reason = todo_ends[0]["result"].as_str().unwrap();
    assert!(reason.contains("found 2"), "got: {reason}");
    assert!(reason.contains("First, Second"), "got: {reason}");

    assert_eq!(todo_ends[1]["is_error"], false);
    assert!(
        todo_ends[1]["result"]
            .as_str()
            .unwrap()
            .contains("[>] First")
    );

    // A rejected call emits no progress: the tool validates before it emits, so no UI (and no
    // `serve` live mirror) ever sees a list the model didn't actually commit.
    let progress = events_of(&events, "tool_progress");
    assert_eq!(
        progress.iter().filter(|e| e["name"] == "todo").count(),
        1,
        "only the accepted call should have emitted details"
    );
}

#[test]
fn the_tool_and_its_protocol_are_advertised_to_the_model_by_default() {
    let (_events, bodies, ok) = run_json(vec![turn_text("nothing to do")], &[]);
    assert!(ok);
    let body = &bodies[0];

    // Registered in the default set...
    assert!(body.contains(r#""name":"todo""#), "todo must be advertised");
    // ...and the protocol its schema can't express is in the system prompt.
    assert!(
        body.contains("todo_protocol"),
        "system prompt must explain the tool"
    );
    assert!(body.contains("fully replaces the previous list"));
    assert!(body.contains("exactly one item `in_progress`"));
}

#[test]
fn excluding_the_tool_also_removes_its_prompt_section() {
    // Dead weight discipline: a model with no `todo` tool must not be told how to drive one.
    let (_events, bodies, ok) = run_json(
        vec![turn_text("nothing to do")],
        &["--exclude-tools", "todo"],
    );
    assert!(ok);
    let body = &bodies[0];
    assert!(
        !body.contains(r#""name":"todo""#),
        "todo must not be advertised"
    );
    assert!(
        !body.contains("todo_protocol"),
        "no protocol without the tool"
    );
}
