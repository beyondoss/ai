//! End-to-end: **beyond-agent → sandbox tools-mcp → `get_weather`**.
//!
//! Substrate (E2B + tools-mcp) is started by `deploy/demo/tools_agent_e2e.sh`, which exports
//! `TOOLS_MCP_URL` and runs these tests inside the guest.
//!
//! 1. `connect_granted_sandbox_calls_get_weather` — the production sandbox-connector path
//!    (`sandbox: true`, same as the browser usecase grant). This is the agent MCP client dialing
//!    the sandbox; no mock model involved.
//! 2. `serve_service_calls_get_weather_on_sandbox_mcp` — full `serve --service` + scripted model
//!    tool call + `.mcp-tools.json` cold open. Needs the in-process mock model to be reachable
//!    from the agent child (loopback).
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::collections::BTreeMap;
use std::time::Duration;

use beyond_agent::grant::McpConnector;
use beyond_agent::tools::mcp::{self, McpEgress};
use beyond_agent::tools::web::ssrf::EgressPolicy;
use beyond_test_support::bundle::{BundleServer, file, sha256_hex, tar};
use common::grant::Bundle;
use common::service::{Options, Service};
use common::{
    TestWs, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

fn mcp_url() -> Option<String> {
    std::env::var("TOOLS_MCP_URL")
        .ok()
        .filter(|s| !s.is_empty())
}

fn tools_manifest() -> String {
    json!({
        "tools": [{
            "remote_name": "get_weather",
            "description": "Get the weather for a city",
            "input_schema": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }]
    })
    .to_string()
}

fn bundle_bytes() -> Vec<u8> {
    tar(&[
        file(
            "AGENTS.md",
            "---\nmodel: test\ntools: mcp__tools__get_weather, todo\nmax-steps: 4\n---\nbe brief\n",
        ),
        file(".mcp-tools.json", &tools_manifest()),
    ])
}

/// Agent MCP client (`connect_granted`, `sandbox: true`) → sandbox tools-mcp → get_weather.
#[tokio::test(flavor = "multi_thread")]
async fn connect_granted_sandbox_calls_get_weather() {
    let Some(url) = mcp_url() else {
        eprintln!("skip: set TOOLS_MCP_URL (see deploy/demo/tools_agent_e2e.sh)");
        return;
    };

    let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
    let connector = McpConnector {
        name: "tools".into(),
        url,
        sandbox: true,
        managed: false,
        command: None,
    };
    let (live, _catalog, warnings) = mcp::connect_granted(
        &[connector],
        &BTreeMap::new(),
        &egress,
        "",
        mcp::host(),
        Duration::from_secs(30),
        &BTreeMap::new(),
        None,
    )
    .await;
    assert!(
        warnings.is_empty(),
        "connect_granted warnings: {warnings:?}"
    );

    let tool = live
        .snapshot()
        .iter()
        .find(|t| t.name() == "mcp__tools__get_weather")
        .cloned()
        .unwrap_or_else(|| {
            let names: Vec<_> = live
                .snapshot()
                .iter()
                .map(|t| t.name().to_string())
                .collect();
            panic!("get_weather not advertised; have {names:?}");
        });

    let out = tool.run(json!({ "city": "SF" })).await.expect("tools/call");
    assert!(
        out.text.contains("SF") && (out.text.contains("sunny") || out.text.contains("72")),
        "unexpected tool result: {out:?}"
    );
}

async fn open(svc: &Service, server: &BundleServer, bytes: &[u8], mcp_url: &str) -> TestWs {
    let token = svc.token_with("t1", "s1.tools", |c| {
        c.mcp = vec![("tools".into(), mcp_url.to_string())];
        c.mcp_sandbox = vec!["tools".into()];
        c.bundle = Some(Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(bytes),
        });
    });
    ws_connect_with_headers(svc.port, Some("s1.tools"), &svc.header(&token))
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

/// Full replica: saved tools at open (no dial), first model tool call dials the sandbox MCP URL.
#[tokio::test(flavor = "multi_thread")]
async fn serve_service_calls_get_weather_on_sandbox_mcp() {
    let Some(url) = mcp_url() else {
        eprintln!("skip: set TOOLS_MCP_URL (see deploy/demo/tools_agent_e2e.sh)");
        return;
    };

    let bytes = bundle_bytes();
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![
        turn_text("hello"),
        turn_tool_use("t1", "mcp__tools__get_weather", r#"{"city":"SF"}"#),
        turn_text("done"),
    ]);
    // Preflight: the agent child must be able to reach this mock. Fail loud if loopback is broken.
    {
        let port: u16 = base
            .trim_start_matches("http://127.0.0.1:")
            .trim_start_matches("http://[::1]:")
            .parse()
            .unwrap_or_else(|_| panic!("unexpected model base {base}"));
        std::net::TcpStream::connect_timeout(
            &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
            Duration::from_secs(2),
        )
        .unwrap_or_else(|e| panic!("mock model not reachable at {base}: {e}"));
    }
    let cache = tempfile::tempdir().unwrap();
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            env: vec![
                ("TMPDIR".into(), cache.path().to_string_lossy().into_owned()),
                ("NO_PROXY".into(), "*".into()),
                ("no_proxy".into(), "*".into()),
            ],
            ..Default::default()
        },
    )
    .await;

    let mut ws = open(&svc, &server, &bytes, &url).await;
    assert_eq!(
        mcp_tools(&mut ws).await,
        vec!["mcp__tools__get_weather".to_string()]
    );

    prompt(&mut ws, "p1", "hi").await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    prompt(&mut ws, "p2", "weather in SF").await;
    let last = requests
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("model saw a follow-up after the tool call");
    assert!(
        last.contains("SF") && (last.contains("sunny") || last.contains("72")),
        "tool result never reached the model via the agent: {}",
        &last[..last.len().min(800)]
    );
}
