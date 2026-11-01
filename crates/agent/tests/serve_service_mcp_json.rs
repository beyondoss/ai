//! Service mode: the tenant's `<workspace>/.mcp.json` adds connectors to a session, and only its
//! `url`s and `command`s count.
//!
//! The file is read from the sandbox, so its content is the tenant's: a `url` becomes a connector
//! dialed through the same SSRF-checked egress as a grant's, and nothing else in it is trusted. A
//! header value, `${VAR}` or `!command` would be a secret lookup or a shell on the replica, and the
//! grant alone supplies headers. A `command` server runs in the sandbox, as a job.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::collections::BTreeMap;

use common::mcp_fixture::spawn_http_mcp_fixture;
use common::service::{Options, Service};
use common::{
    TestWs, spawn_model_server, turn_text, ws_connect_with_headers, ws_read_until_response, ws_send,
};
use serde_json::json;

const SECRET: &str = "replica-only-secret-value";

/// The replica, with loopback egress open for the fixtures and a secret in its own environment.
async fn service(base: &str) -> Service {
    Service::start_with(
        base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            env: vec![("SECRET".into(), SECRET.into())],
            ..Default::default()
        },
    )
    .await
}

async fn open_session(
    svc: &Service,
    session_id: &str,
    connectors: &[(&str, &str)],
    headers: &[(&str, &[(&str, &str)])],
) -> TestWs {
    let mut claims = svc.claims("tenant-a", session_id, &svc.shards[0].0);
    claims.mcp = connectors
        .iter()
        .map(|(n, u)| ((*n).to_string(), (*u).to_string()))
        .collect();
    let mut secrets = svc.secrets();
    secrets.mcp_headers = headers
        .iter()
        .map(|(name, pairs)| {
            (
                (*name).to_string(),
                pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let token = svc.minter.mint(&claims, &secrets);
    ws_connect_with_headers(svc.port, Some(session_id), &svc.header(&token))
        .await
        .unwrap_or_else(|status| panic!("session {session_id} refused with HTTP {status}"))
}

async fn advertised_mcp_tools(ws: &mut TestWs) -> Vec<String> {
    ws_send(ws, json!({ "type": "get_mcp", "id": "m1" })).await;
    let frames = ws_read_until_response(ws, "get_mcp").await;
    let response = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "get_mcp")
        .unwrap_or_else(|| panic!("no get_mcp response: {frames:#?}"));
    assert_eq!(response["success"], true, "{response}");
    response["data"]["tools"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|t| t.as_str().map(str::to_string))
        .collect()
}

/// A `url` entry in the sandbox's `.mcp.json` is a connector of the session, with no grant entry.
#[tokio::test]
async fn a_url_entry_in_the_sandbox_mcp_json_becomes_a_connector() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let fixture = spawn_http_mcp_fixture();
    let svc = service(&base).await;
    svc.write_in_workspace(
        ".mcp.json",
        &json!({"mcpServers": {"docs": {"url": fixture.url}}}).to_string(),
    );

    let mut ws = open_session(&svc, "s1.file", &[], &[]).await;
    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(
        tools.contains(&"mcp__docs__echo".to_string()),
        "the file's connector must be dialed: {tools:?}"
    );
}

/// The file's header says `${SECRET}` and the replica's environment has a `SECRET`: the value must
/// never go out. The grant's own header for the same name is what the server sees.
#[tokio::test]
async fn a_header_in_the_file_is_never_sent_and_a_var_in_it_is_never_expanded() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let with_grant = spawn_http_mcp_fixture();
    let without_grant = spawn_http_mcp_fixture();
    let svc = service(&base).await;
    svc.write_in_workspace(
        ".mcp.json",
        &json!({"mcpServers": {
            "granted": {"url": with_grant.url, "headers": {
                "X-Secret": "${SECRET}", "Authorization": "Bearer ${SECRET}", "X-Cmd": "!echo pwned"}},
            "bare": {"url": without_grant.url, "headers": {
                "X-Secret": "${SECRET}", "Authorization": "Bearer ${SECRET}"}},
        }})
        .to_string(),
    );

    // `granted` has a sealed header in the grant; `bare` has none.
    let mut ws = open_session(
        &svc,
        "s1.hdr",
        &[],
        &[("granted", &[("Authorization", "Bearer from-grant")])],
    )
    .await;
    // `granted` is not in `claims.mcp`, so it connects from the file with the grant's headers for
    // that name.
    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(
        tools.contains(&"mcp__granted__echo".to_string()),
        "{tools:?}"
    );
    assert!(tools.contains(&"mcp__bare__echo".to_string()), "{tools:?}");

    for fixture in [&with_grant, &without_grant] {
        assert!(fixture.request_count() > 0);
        for name in ["x-secret", "x-cmd"] {
            assert!(!fixture.saw_header(name), "{name} came from the file");
        }
        for auth in fixture.header_values("authorization") {
            assert!(
                !auth.contains(SECRET),
                "the replica's secret went out: {auth}"
            );
            assert!(!auth.contains("${"), "the file's template went out: {auth}");
        }
    }
    assert!(
        with_grant
            .header_values("authorization")
            .iter()
            .all(|v| v == "Bearer from-grant"),
        "the grant's header is what goes out: {:?}",
        with_grant.header_values("authorization")
    );
    assert!(
        !without_grant.saw_header("authorization"),
        "a name with no grant headers connects without any"
    );
}

/// A `command` server is a process in the tenant's sandbox, never on the replica: its tools reach
/// the session, and the only way it was started is a job request to the sandbox's exec endpoint.
#[tokio::test]
async fn a_command_entry_runs_in_the_sandbox_and_its_tools_reach_the_session() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = service(&base).await;
    svc.write_in_workspace(
        ".mcp.json",
        &json!({"mcpServers": {"proc": {
            "command": env!("CARGO_BIN_EXE_mcp_fixture_stdio_server")}}})
        .to_string(),
    );

    let mut ws = open_session(&svc, "s1.cmd", &[], &[]).await;
    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(
        tools.contains(&"mcp__proc__echo".to_string()),
        "the server's tools must be advertised: {tools:?}"
    );
    let starts: Vec<_> = svc
        .exec
        .bodies()
        .into_iter()
        .filter(|b| b["job"]["op"] == "start" && b["job"]["label"] == "mcp proc")
        .collect();
    assert_eq!(starts.len(), 1, "the sandbox was asked to run it once");
}

/// The same name in the grant and the file: the grant's connector is the one dialed.
#[tokio::test]
async fn a_grant_connector_wins_over_a_file_entry_of_the_same_name() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let granted = spawn_http_mcp_fixture();
    let filed = spawn_http_mcp_fixture();
    let svc = service(&base).await;
    svc.write_in_workspace(
        ".mcp.json",
        &json!({"mcpServers": {"docs": {"url": filed.url}}}).to_string(),
    );

    let mut ws = open_session(&svc, "s1.clash", &[("docs", &granted.url)], &[]).await;
    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(tools.contains(&"mcp__docs__echo".to_string()), "{tools:?}");
    assert!(granted.request_count() > 0);
    assert_eq!(
        filed.request_count(),
        0,
        "the file's url must not be dialed"
    );
}

/// A file url goes through the same egress policy as a grant's: a private address is refused.
#[tokio::test]
async fn a_file_url_is_held_to_the_egress_policy() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let fixture = spawn_http_mcp_fixture();
    // No `--mcp-allow-private`: loopback is off limits.
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace(
        ".mcp.json",
        &json!({"mcpServers": {"docs": {"url": fixture.url}}}).to_string(),
    );

    let mut ws = open_session(&svc, "s1.egress", &[], &[]).await;
    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(
        tools.iter().all(|t| !t.starts_with("mcp__docs__")),
        "{tools:?}"
    );
    assert_eq!(fixture.request_count(), 0, "a private address was dialed");
}
