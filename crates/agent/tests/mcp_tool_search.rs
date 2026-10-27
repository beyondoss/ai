//! Deferred MCP tools e2e (`tools::tool_search`): the real binary, a real MCP server (the stdio
//! fixture), and a scripted model whose requests are inspected on the wire.
//!
//! - a deferred tool is found by `tool_search`, loaded, and called — on Anthropic (natively:
//!   `defer_loading` + `tool_reference`) and on OpenAI Responses (an `additional_tools` item);
//! - the cached prefix (`tools` + `system`) is byte-identical across every turn of a run and across
//!   two runs, a load included;
//! - `--mcp-tool-loading eager` sends the flattened catalog exactly as before.
//!
//! Code Mode taking precedence is in `code_mode_eval.rs`, which needs `--features code-mode`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::Path;

use common::{body_json, run_cmd, spawn_model_server, sse, turn_text, turn_tool_use};
use serde_json::{Value, json};

fn write_settings(home: &Path) {
    let dir = home.join(".beyond");
    std::fs::create_dir_all(&dir).unwrap();
    let server = |name: &str| {
        json!({
            "name": name,
            "transport": "stdio",
            "command": env!("CARGO_BIN_EXE_mcp_fixture_stdio_server"),
            "args": [],
            "env": {},
        })
    };
    std::fs::write(
        dir.join("settings.json"),
        json!({ "mcp_servers": [server("github"), server("jira")] }).to_string(),
    )
    .unwrap();
}

/// Run `task` against the scripted `responses`, returning (success, stderr, request bodies).
fn run(model: &str, loading: &str, responses: Vec<String>) -> (bool, String, Vec<Value>) {
    let home = tempfile::tempdir().unwrap();
    run_in(home.path(), model, loading, responses)
}

/// [`run`] in a given home (and cwd), so two runs can share one — the system prompt names the cwd.
fn run_in(
    home: &Path,
    model: &str,
    loading: &str,
    responses: Vec<String>,
) -> (bool, String, Vec<Value>) {
    write_settings(home);
    let (base, bodies) = spawn_model_server(responses);
    let out = run_cmd(env!("CARGO_BIN_EXE_agent"))
        .env("HOME", home)
        .current_dir(home)
        .args([
            "run",
            "add 2 and 3",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            model,
            "--max-steps",
            "6",
            "--no-session-persistence",
            "--mcp-tool-loading",
            loading,
        ])
        .output()
        .unwrap();
    let bodies = bodies
        .lock()
        .unwrap()
        .iter()
        .map(|r| body_json(r))
        .collect();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        bodies,
    )
}

fn names(tools: &Value) -> Vec<&str> {
    tools
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect()
}

/// The search → load → call script, Anthropic wire.
fn anthropic_script() -> Vec<String> {
    vec![
        turn_tool_use("toolu_1", "tool_search", r#"{"query":"github add"}"#),
        turn_tool_use("toolu_2", "mcp__github__add", r#"{"a":2,"b":3}"#),
        turn_text("5"),
    ]
}

#[test]
fn a_deferred_tool_is_found_loaded_and_called_on_anthropic() {
    let (ok, stderr, bodies) = run("claude-sonnet-5-5", "search", anthropic_script());
    assert!(ok, "{stderr}");
    assert_eq!(bodies.len(), 3, "{stderr}");
    let tools = &bodies[0]["tools"];
    // Advertised: the built-ins and `tool_search`, with the cache breakpoint on the last of them…
    let search = tools
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "tool_search")
        .unwrap();
    assert!(
        search["description"]
            .as_str()
            .unwrap()
            .contains("- github: add, "),
        "{search}"
    );
    // …then every deferred MCP tool, out of the model's context until loaded.
    let deferred: Vec<&Value> = tools
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["defer_loading"] == true)
        .collect();
    assert!(deferred.iter().any(|t| t["name"] == "mcp__github__add"));
    assert!(deferred.iter().all(|t| t.get("cache_control").is_none()));
    assert!(
        names(tools)
            .iter()
            .filter(|n| n.starts_with("mcp__"))
            .all(|n| deferred.iter().any(|t| t["name"] == *n))
    );
    // The search result loads the tool by reference.
    let result = &bodies[1]["messages"].as_array().unwrap().last().unwrap()["content"][0];
    assert_eq!(result["tool_use_id"], "toolu_1");
    assert_eq!(
        result["content"],
        json!([{ "type": "tool_reference", "tool_name": "mcp__github__add" }])
    );
    // The loaded tool really ran against the MCP server.
    let result = &bodies[2]["messages"].as_array().unwrap().last().unwrap()["content"][0];
    assert_eq!(result["tool_use_id"], "toolu_2");
    assert_eq!(result["content"], "5");
    assert_eq!(result["is_error"], false);
}

#[test]
fn the_cached_prefix_is_byte_stable_across_turns_and_runs() {
    let prefix = |b: &Value| format!("{}{}", b["tools"], b["system"]);
    let home = tempfile::tempdir().unwrap();
    let (ok, stderr, first) = run_in(
        home.path(),
        "claude-sonnet-5-5",
        "search",
        anthropic_script(),
    );
    assert!(ok, "{stderr}");
    let (_, _, second) = run_in(
        home.path(),
        "claude-sonnet-5-5",
        "search",
        anthropic_script(),
    );
    let p = prefix(&first[0]);
    // A load does not touch it…
    for b in first.iter().chain(&second) {
        assert_eq!(prefix(b), p);
    }
    // …and no MCP tool name is repeated in the system prompt's tool line.
    assert!(!first[0]["system"].to_string().contains("mcp__github__"));
}

#[test]
fn a_deferred_tool_loads_through_additional_tools_on_openai_responses() {
    let call = |id: &str, name: &str, args: &str| {
        sse(&[
            json!({ "type": "response.output_item.added", "output_index": 0, "item": { "type": "function_call", "id": format!("fc_{id}"), "call_id": format!("call_{id}"), "name": name, "arguments": "" } }),
            json!({ "type": "response.function_call_arguments.delta", "output_index": 0, "delta": args }),
            json!({ "type": "response.output_item.done", "output_index": 0, "item": { "type": "function_call", "id": format!("fc_{id}"), "call_id": format!("call_{id}"), "name": name, "arguments": args } }),
            json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "input_tokens": 1, "output_tokens": 1 } } }),
        ])
    };
    let (ok, stderr, bodies) = run(
        "gpt-6.1-sol",
        "search",
        vec![
            call("1", "tool_search", r#"{"query":"github add"}"#),
            call("2", "mcp__github__add", r#"{"a":2,"b":3}"#),
            common::turn_text_responses("5"),
        ],
    );
    assert!(ok, "{stderr}");
    assert_eq!(bodies.len(), 3, "{stderr}");
    // No MCP schema up front, and `tools` never changes.
    assert!(
        !names(&bodies[0]["tools"])
            .iter()
            .any(|n| n.starts_with("mcp__github__a"))
    );
    assert_eq!(bodies[0]["tools"], bodies[2]["tools"]);
    let input = bodies[1]["input"].as_array().unwrap();
    let at = input
        .iter()
        .position(|i| i["type"] == "function_call_output" && i["call_id"] == "call_1")
        .unwrap();
    assert_eq!(input[at + 1]["type"], "additional_tools");
    assert_eq!(names(&input[at + 1]["tools"]), ["mcp__github__add"]);
    let last = bodies[2]["input"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(last["type"], "function_call_output");
    assert_eq!(last["output"], "5");
}

#[test]
fn eager_loading_sends_the_flattened_catalog_as_before() {
    let (ok, stderr, bodies) = run(
        "claude-sonnet-5-5",
        "eager",
        vec![
            turn_tool_use("toolu_1", "mcp__github__add", r#"{"a":2,"b":3}"#),
            turn_text("5"),
        ],
    );
    assert!(ok, "{stderr}");
    let tools = &bodies[0]["tools"];
    let names = names(tools);
    assert!(!names.contains(&"tool_search"));
    assert!(names.contains(&"mcp__github__add") && names.contains(&"mcp__jira__add"));
    assert!(
        tools
            .as_array()
            .unwrap()
            .iter()
            .all(|t| t.get("defer_loading").is_none())
    );
    let result = &bodies[1]["messages"].as_array().unwrap().last().unwrap()["content"][0];
    assert_eq!(result["content"], "5");
}
