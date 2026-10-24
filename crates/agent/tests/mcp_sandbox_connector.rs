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
    call_with_idle_gap(fixture, sandbox, Duration::ZERO, Duration::from_millis(150)).await;
}

/// Connect `fixture` with an idle window of `reap_after`, then make three tool calls `gap` apart.
async fn call_with_idle_gap(
    fixture: &SessionMcpFixture,
    sandbox: bool,
    reap_after: Duration,
    gap: Duration,
) {
    let egress = McpEgress::new(EgressPolicy::new(true, &[])).unwrap();
    let connector = McpConnector {
        name: "c".into(),
        url: fixture.url.clone(),
        sandbox,
        managed: false,
    };
    let (tools, _catalog, warnings) = mcp::connect_granted(
        &[connector],
        &BTreeMap::new(),
        &egress,
        "",
        mcp::host(),
        reap_after,
        &BTreeMap::new(),
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
        tokio::time::sleep(gap).await;
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

#[tokio::test(flavor = "multi_thread")]
async fn a_sandbox_connector_keeps_its_session_through_an_idle_pause() {
    let fixture = spawn_session_mcp_fixture().await;
    call_with_idle_gap(
        &fixture,
        true,
        Duration::from_secs(1),
        Duration::from_secs(3),
    )
    .await;
    assert_eq!(
        fixture.deletes(),
        0,
        "the idle reaper ended the MCP session"
    );
    assert_eq!(
        fixture.initializes(),
        1,
        "the idle reaper redialed a new session"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_default_connector_is_still_reaped_when_idle() {
    let fixture = spawn_session_mcp_fixture().await;
    call_with_idle_gap(
        &fixture,
        false,
        Duration::from_secs(1),
        Duration::from_secs(3),
    )
    .await;
    assert!(
        fixture.deletes() >= 1,
        "an idle default connector ends its session"
    );
    assert!(
        fixture.initializes() >= 2,
        "and the next call dials a new one"
    );
}

/// Connect `url` as a connector against the default (private-refusing) egress, with the grant's
/// exec endpoint at `exec_url`; returns whether the tool was advertised and the warnings.
async fn dial_guarded(url: &str, sandbox: bool, exec_url: &str) -> (bool, Vec<String>) {
    let egress = McpEgress::new(EgressPolicy::default()).unwrap();
    let connector = McpConnector {
        name: "c".into(),
        url: url.into(),
        sandbox,
        managed: false,
    };
    let (tools, _catalog, warnings) = mcp::connect_granted(
        &[connector],
        &BTreeMap::new(),
        &egress,
        exec_url,
        mcp::host(),
        Duration::ZERO,
        &BTreeMap::new(),
    )
    .await;
    let found = tools.snapshot().iter().any(|t| t.name() == "mcp__c__echo");
    (found, warnings)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sandbox_connector_on_the_exec_origin_connects_at_a_private_address() {
    let fixture = spawn_session_mcp_fixture().await;
    let exec = fixture.url.replace("/mcp", "/exec");
    let (found, warnings) = dial_guarded(&fixture.url, true, &exec).await;
    assert!(found && warnings.is_empty(), "{warnings:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sandbox_connector_on_another_private_address_is_refused() {
    let fixture = spawn_session_mcp_fixture().await;
    let other = spawn_session_mcp_fixture().await;
    // Same host, different port: a different origin.
    let exec = other.url.replace("/mcp", "/exec");
    let (found, warnings) = dial_guarded(&fixture.url, true, &exec).await;
    assert!(!found);
    assert!(
        warnings.iter().any(|w| w.contains("refusing to dial")),
        "{warnings:?}"
    );
    // And with no exec endpoint at all.
    let (found, _) = dial_guarded(&fixture.url, true, "").await;
    assert!(!found);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_default_connector_on_the_exec_origin_is_still_refused() {
    let fixture = spawn_session_mcp_fixture().await;
    let exec = fixture.url.replace("/mcp", "/exec");
    let (found, warnings) = dial_guarded(&fixture.url, false, &exec).await;
    assert!(!found);
    assert!(
        warnings.iter().any(|w| w.contains("refusing to dial")),
        "{warnings:?}"
    );
}
