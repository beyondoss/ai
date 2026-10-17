//! MCP names every provider accepts: a server's tool, resource and prompt names are free text
//! (`README.md`, `Resource 1`, `"users" database schema`), but Anthropic and OpenAI require
//! `^[a-zA-Z0-9_-]{1,64}$` of every tool name — one bad name and every turn is refused. These prove
//! the registered names are normalised, that a call still reaches the server under the name *it*
//! knows, and that a server name that can't be split back out of `mcp__<server>__<tool>` is refused.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

fn server(name: &str) -> Value {
    json!({
        "name": name,
        "transport": "stdio",
        "command": env!("CARGO_BIN_EXE_mcp_fixture_stdio_server"),
        "env": { "MCP_FIXTURE_ODD_NAMES": "1" },
    })
}

fn run(servers: Value, responses: Vec<String>) -> (String, Vec<String>) {
    let home = tempfile::tempdir().unwrap();
    let claude = home.path().join(".beyond");
    std::fs::create_dir_all(&claude).unwrap();
    std::fs::write(
        claude.join("settings.json"),
        json!({ "mcp_servers": servers }).to_string(),
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(responses);
    let output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .args([
            "run",
            "go",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
        ])
        .current_dir(Path::new(home.path()))
        .output()
        .expect("spawn binary");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "run failed: {stderr}");
    let bodies = bodies.lock().unwrap().clone();
    (stderr, bodies)
}

/// Every tool name the recorded request (headers, blank line, JSON body) advertised.
fn advertised(raw_request: &str) -> Vec<String> {
    let body = raw_request.split("\r\n\r\n").nth(1).unwrap();
    let body: Value = serde_json::from_str(body).unwrap();
    body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["name"].as_str().map(str::to_owned))
        .collect()
}

fn provider_valid(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

#[test]
fn every_advertised_mcp_name_is_one_providers_accept() {
    let (_stderr, bodies) = run(json!([server("fixture")]), vec![turn_text("hi")]);
    let names = advertised(&bodies[0]);
    for name in &names {
        assert!(
            provider_valid(name),
            "{name:?} would fail every turn: {names:?}"
        );
    }
    for want in [
        "mcp__fixture__get_file",
        "mcp__fixture__resource__README_md",
        "mcp__fixture__prompt__complex_prompt",
    ] {
        assert!(names.iter().any(|n| n == want), "{want} missing: {names:?}");
    }
    assert_eq!(
        names
            .iter()
            .filter(|n| *n == "mcp__fixture__get_file")
            .count(),
        1,
        "`get.file` and `get file` normalise alike: only one may be registered"
    );
}

/// The normalised name is the model's; the server is called by its own.
#[test]
fn a_normalised_name_calls_the_server_by_its_remote_name() {
    let turn = |id: &str, name: &str| turn_tool_use(id, name, "{}");
    let (_stderr, bodies) = run(
        json!([server("fixture")]),
        vec![
            turn("toolu_1", "mcp__fixture__get_file"),
            turn("toolu_2", "mcp__fixture__resource__README_md"),
            turn("toolu_3", "mcp__fixture__prompt__complex_prompt"),
            turn_text("done"),
        ],
    );
    assert_eq!(bodies.len(), 4, "{bodies:#?}");
    // The first listed of the two colliding tools keeps the name.
    assert!(bodies[1].contains("called-as=get.file"), "{}", bodies[1]);
    assert!(bodies[2].contains("readme-body"), "{}", bodies[2]);
    assert!(bodies[3].contains("complex-prompt-body"), "{}", bodies[3]);
}

/// `a__b`'s tool `c` and `a`'s tool `b__c` would both register as `mcp__a__b__c`: a server name with
/// `__` is refused outright, with a warning, and the other server is untouched.
#[test]
fn a_server_name_with_a_double_underscore_is_refused_and_the_rest_load() {
    let (stderr, bodies) = run(json!([server("a__b"), server("a")]), vec![turn_text("hi")]);
    let names = advertised(&bodies[0]);
    assert!(
        names.iter().any(|n| n == "mcp__a__echo"),
        "server `a` must still load: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("mcp__a__b__")),
        "server `a__b` must not register anything: {names:?}"
    );
    assert!(
        stderr.contains("a__b"),
        "the refusal must be reported: {stderr}"
    );
}
