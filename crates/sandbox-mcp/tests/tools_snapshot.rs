//! The tools' names, descriptions and input schemas are the server's contract with every agent
//! that uses it, so they are pinned: a change to any of them fails this test until the snapshot is
//! regenerated on purpose (`UPDATE_SNAPSHOT=1 cargo test -p beyond-sandbox-mcp --test
//! tools_snapshot`) and the diff is read.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rmcp::ServiceExt as _;
use sandbox_mcp::{Config, SandboxMcp};
use serde_json::Value;

const SNAPSHOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tools.snapshot.json");

/// `tools/list` over an in-process MCP session (no sandbox needed: nothing is called).
async fn listed() -> Vec<Value> {
    let config =
        Config::from_env(|k| (k == "SANDBOX_URL").then(|| "http://127.0.0.1:1".to_string()))
            .unwrap();
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let server = tokio::spawn(async move {
        SandboxMcp::new(config)
            .serve(server_io)
            .await
            .unwrap()
            .waiting()
            .await
    });
    let client = ().serve(client_io).await.unwrap();
    let tools = client.peer().list_all_tools().await.unwrap();
    client.cancel().await.unwrap();
    server.abort();
    tools
        .into_iter()
        .map(|t| serde_json::to_value(t).unwrap())
        .collect()
}

#[tokio::test]
async fn the_tool_list_is_the_pinned_one() {
    let mut tools = listed().await;
    tools.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let actual = serde_json::to_string_pretty(&tools).unwrap() + "\n";
    if std::env::var_os("UPDATE_SNAPSHOT").is_some() {
        std::fs::write(SNAPSHOT, &actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(SNAPSHOT)
        .expect("tests/tools.snapshot.json is missing: UPDATE_SNAPSHOT=1 writes it");
    assert!(
        expected == actual,
        "the tool list changed. If that is intended, run with UPDATE_SNAPSHOT=1 and review the diff.\n--- actual ---\n{actual}"
    );
}

#[tokio::test]
async fn the_tools_are_exactly_the_documented_twelve() {
    let mut names: Vec<String> = listed()
        .await
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "edit_file",
            "exec",
            "job_kill",
            "job_list",
            "job_output",
            "job_start",
            "list_dir",
            "mkdir",
            "read_file",
            "search",
            "stat",
            "write_file"
        ]
    );
}

#[tokio::test]
async fn every_tool_and_every_argument_has_a_description() {
    for t in listed().await {
        let name = t["name"].as_str().unwrap();
        assert!(
            t["description"].as_str().is_some_and(|d| d.len() > 40),
            "{name} has no description"
        );
        let props = t["inputSchema"]["properties"].as_object().unwrap();
        for (arg, schema) in props {
            assert!(
                schema["description"].as_str().is_some(),
                "{name}.{arg} has no description"
            );
        }
    }
}
