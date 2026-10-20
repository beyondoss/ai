//! A connector that lives inside a sleeping sandbox must not hold a connection open: one held
//! stream keeps the VM awake forever. rmcp opens the standalone `GET` stream as soon as a server
//! hands out a session id, so a sandbox connector has to refuse it locally and pool nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use beyond_ai_agent::grant::McpConnector;
use beyond_ai_agent::tools::mcp::{self, McpEgress};
use beyond_ai_agent::tools::web::ssrf::EgressPolicy;
use common::mcp_session_fixture::{SessionMcpFixture, spawn_session_mcp_fixture};
use serde_json::json;

async fn within(limit: Duration, mut f: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    f()
}

/// Connect `fixture` as a grant connector and make three tool calls with pauses between them.
async fn call_three_times(fixture: &SessionMcpFixture, sandbox: bool) {
    let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
    let connector = McpConnector {
        name: "c".into(),
        url: fixture.url.clone(),
        sandbox,
    };
    let (tools, _catalog, warnings) = mcp::connect_granted(
        &[connector],
        &BTreeMap::new(),
        &egress,
        mcp::host(),
        Duration::ZERO,
    )
    .await;
    assert!(warnings.is_empty(), "{warnings:?}");
    let echo = tools
        .snapshot()
        .iter()
        .find(|t| t.name() == "mcp__c__echo")
        .expect("echo advertised")
        .clone();
    for i in 0..3 {
        let out = echo
            .run(json!({ "text": format!("call {i}") }))
            .await
            .unwrap();
        assert!(out.text.contains(&format!("call {i}")), "{out:?}");
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sandbox_connector_holds_no_stream_and_no_connection_between_calls() {
    let fixture = spawn_session_mcp_fixture().await;
    call_three_times(&fixture, true).await;
    assert_eq!(fixture.gets(), 0, "no standalone stream may be requested");
    assert!(
        within(Duration::from_secs(2), || fixture.open_connections() == 0).await,
        "{} connections still open between calls",
        fixture.open_connections()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_default_connector_still_opens_the_standalone_stream() {
    let fixture = spawn_session_mcp_fixture().await;
    call_three_times(&fixture, false).await;
    assert_eq!(
        fixture.gets(),
        1,
        "tools-changed notifications ride this stream"
    );
    assert!(fixture.open_connections() >= 1);
}
