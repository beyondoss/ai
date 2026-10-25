//! A bundle may carry the tool list of a sandbox-routed MCP server (`.mcp-tools.json`). A bundled
//! session then advertises those tools at open without dialing the server, which is what lets the
//! sandbox behind it sleep (or not exist yet) until the model first calls one. The first call dials
//! the server; a live list that differs replaces the saved one.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use beyond_ai_test_support::bundle::{BundleServer, file, sha256_hex, tar};
use common::grant::Bundle;
use common::mcp_fixture::{HttpMcpFixture, spawn_http_mcp_fixture};
use common::service::{Options, Service};
use common::{
    TestWs, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

async fn open(svc: &Service, server: &BundleServer, bytes: &[u8], mcp: &HttpMcpFixture) -> TestWs {
    let token = svc.token_with("t1", "s1.alpha", |c| {
        c.mcp = vec![("box".into(), mcp.url.clone())];
        c.mcp_sandbox = vec!["box".into()];
        c.bundle = Some(Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(bytes),
        });
    });
    ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap()
}

async fn prompt(ws: &mut TestWs, id: &str, message: &str) {
    ws_send(ws, json!({"type": "prompt", "id": id, "message": message})).await;
    let frames = ws_read_until_response(ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}

async fn mcp_tools(ws: &mut TestWs) -> Vec<String> {
    ws_send(ws, json!({"type": "get_mcp", "id": "m1"})).await;
    let frames = ws_read_until_response(ws, "get_mcp").await;
    let r = frames.last().unwrap();
    r["data"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect()
}

fn saved(tools: &[&str]) -> String {
    let list: Vec<Value> = tools
        .iter()
        .map(|n| {
            json!({
                "remote_name": n,
                "description": format!("saved {n}"),
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
            })
        })
        .collect();
    json!({ "box": list }).to_string()
}

fn bundle(tools_json: &str) -> Vec<u8> {
    tar(&[
        file("AGENTS.md", "be brief\n"),
        file(".mcp-tools.json", tools_json),
    ])
}

async fn start(base: &str) -> (Service, tempfile::TempDir) {
    let cache = tempfile::tempdir().unwrap();
    let svc = Service::start_with(
        base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            env: vec![("TMPDIR".into(), cache.path().to_string_lossy().into_owned())],
            ..Default::default()
        },
    )
    .await;
    (svc, cache)
}

#[tokio::test(flavor = "multi_thread")]
async fn saved_tools_open_with_no_dial_and_the_first_call_dials_once() {
    let bytes = bundle(&saved(&["echo"]));
    let server = BundleServer::start(bytes.clone());
    let mcp = spawn_http_mcp_fixture();
    let (base, requests) = spawn_model_server(vec![
        turn_text("hello"),
        turn_tool_use("t1", "mcp__box__echo", r#"{"text":"SAVED-RAN"}"#),
        turn_text("done"),
    ]);
    let (svc, _cache) = start(&base).await;
    let mut ws = open(&svc, &server, &bytes, &mcp).await;

    assert_eq!(mcp_tools(&mut ws).await, vec!["mcp__box__echo"]);
    prompt(&mut ws, "p1", "hi").await;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(mcp.request_count(), 0, "open must not dial the server");
    assert!(
        svc.exec.requests().is_empty(),
        "open must make no exec request"
    );

    prompt(&mut ws, "p2", "call it").await;
    assert!(
        requests
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .contains("SAVED-RAN"),
        "the saved tool must run"
    );
    let methods = mcp.header_values("x-rpc-method");
    assert_eq!(methods.iter().filter(|m| *m == "tools/call").count(), 1);
    assert_eq!(
        methods.iter().filter(|m| *m == "server/discover").count(),
        1,
        "one handshake, not one per request: {methods:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_live_list_that_differs_replaces_the_saved_one() {
    // The server really lists `echo` and `fail`; the bundle saved `echo` and `ghost`.
    let bytes = bundle(&saved(&["echo", "ghost"]));
    let server = BundleServer::start(bytes.clone());
    let mcp = spawn_http_mcp_fixture();
    let (base, _requests) = spawn_model_server(vec![
        turn_text("hello"),
        turn_tool_use("t1", "mcp__box__echo", r#"{"text":"x"}"#),
        turn_text("done"),
    ]);
    let (svc, _cache) = start(&base).await;
    let mut ws = open(&svc, &server, &bytes, &mcp).await;
    let before = mcp_tools(&mut ws).await;
    assert!(
        before.contains(&"mcp__box__ghost".to_string()),
        "{before:?}"
    );

    prompt(&mut ws, "p1", "hi").await;
    prompt(&mut ws, "p2", "call it").await;
    let after = mcp_tools(&mut ws).await;
    assert!(after.contains(&"mcp__box__fail".to_string()), "{after:?}");
    assert!(!after.contains(&"mcp__box__ghost".to_string()), "{after:?}");
}
