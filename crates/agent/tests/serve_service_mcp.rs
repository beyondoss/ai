//! Service mode: a session's MCP connectors come from its **grant**, and stay inside it.
//!
//! The replica's own configured servers are never connected in service mode at all (there are none
//! to connect — stored settings are skipped at startup), so everything a tenant can reach over MCP
//! is in the token it presented. These are the tests that would go quiet if any of that leaked
//! sideways: one tenant's connector reaching another tenant's session, a grant header taking the
//! settings resolution path (`!command`, `$VAR`), an operator's own OAuth login being attached to a
//! connector that merely shares its name, or a connector URL reaching something internal.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;

use common::mcp_fixture::{HttpMcpFixture, spawn_http_mcp_fixture};
use common::service::{Options, Service};
use common::{
    TestWs, spawn_model_server, turn_text, ws_connect_with_headers, ws_read_until_response, ws_send,
};
use serde_json::json;

/// The replica, with loopback egress opened up so the fixtures below are reachable at all. A
/// production replica never passes this: a connector URL must resolve to a public address.
async fn service_with_local_connectors(base: &str) -> Service {
    Service::start_with(
        base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            ..Default::default()
        },
    )
    .await
}

/// Open a session whose grant names `connectors` (`name` → url) and seals `headers` for them.
async fn open_session(
    svc: &Service,
    session_id: &str,
    connectors: &[(&str, &str)],
    headers: &[(&str, &[(&str, &str)])],
) -> TestWs {
    let mut claims = svc.claims("tenant-a", session_id, &svc.shards[0].0);
    claims.mcp = connectors
        .iter()
        .map(|(name, url)| ((*name).to_string(), (*url).to_string()))
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

/// The MCP tool names this session advertises, via the `get_mcp` command.
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

/// Two tenants' sessions on one replica, each with its own connector. Neither learns the other's
/// tools, and neither server is dialed on the other's behalf — the connections are per session, not
/// a process-wide set every session shares.
#[tokio::test]
async fn two_sessions_see_only_the_connectors_their_own_grant_names() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let alpha: HttpMcpFixture = spawn_http_mcp_fixture();
    let beta: HttpMcpFixture = spawn_http_mcp_fixture();
    let svc = service_with_local_connectors(&base).await;

    let mut a = open_session(&svc, "s1.alpha", &[("alpha", &alpha.url)], &[]).await;
    let mut b = open_session(&svc, "s1.beta", &[("beta", &beta.url)], &[]).await;

    let a_tools = advertised_mcp_tools(&mut a).await;
    let b_tools = advertised_mcp_tools(&mut b).await;

    assert!(
        a_tools.contains(&"mcp__alpha__echo".to_string()),
        "session A must see its own connector's tools: {a_tools:?}"
    );
    assert!(
        a_tools.iter().all(|t| !t.starts_with("mcp__beta__")),
        "session A must not see another session's connector: {a_tools:?}"
    );
    assert!(
        b_tools.contains(&"mcp__beta__echo".to_string()),
        "session B must see its own connector's tools: {b_tools:?}"
    );
    assert!(
        b_tools.iter().all(|t| !t.starts_with("mcp__alpha__")),
        "session B must not see another session's connector: {b_tools:?}"
    );
    assert!(
        alpha.request_count() > 0 && beta.request_count() > 0,
        "each connector is dialed by the session that was granted it"
    );
}

/// A grant's header value is a credential, not a template. `settings.json`'s own header syntax runs
/// a `!command` through a shell and expands `$VAR` from the process environment — on a replica that
/// is arbitrary execution and an environment read, for a value that arrived over the network.
#[tokio::test]
async fn a_grant_header_reaches_the_wire_literally_and_is_never_executed_or_expanded() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let fixture = spawn_http_mcp_fixture();
    let svc = service_with_local_connectors(&base).await;

    let mut ws = open_session(
        &svc,
        "s1.headers",
        &[("linear", &fixture.url)],
        &[(
            "linear",
            &[
                ("X-Command", "!echo pwned"),
                ("X-Env", "$HOME"),
                ("Authorization", "Bearer granted-token"),
            ],
        )],
    )
    .await;
    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(
        tools.contains(&"mcp__linear__echo".to_string()),
        "the connector must have connected at all: {tools:?}"
    );

    assert_eq!(
        fixture.header_values("x-command"),
        vec!["!echo pwned"; fixture.request_count()],
        "a `!command` header must arrive as its own eleven bytes — never run, never dropped"
    );
    assert!(
        fixture.header_values("x-env").iter().all(|v| v == "$HOME"),
        "a `$VAR` header must arrive literally: {:?}",
        fixture.header_values("x-env")
    );
    assert!(
        fixture
            .header_values("authorization")
            .iter()
            .all(|v| v == "Bearer granted-token"),
        "the grant's own Authorization header is the one that goes out: {:?}",
        fixture.header_values("authorization")
    );
}

/// The OAuth store is keyed by *server name* and belongs to the replica's operator. A tenant that
/// names its connector after one of the operator's logins must not inherit that token — the grant
/// is the only place a connector's credentials come from.
#[tokio::test]
async fn a_connector_named_like_an_operator_login_gets_no_bearer_token() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let fixture = spawn_http_mcp_fixture();

    // A replica whose operator has logged into an MCP server called `linear`.
    let host_home = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(host_home.path().join(".claude")).unwrap();
    std::fs::write(
        host_home.path().join(".claude/mcp_auth.json"),
        json!({
            "linear": {
                "client_id": "operator-client-id",
                "token_response": {
                    "access_token": "operator-access-token-must-not-leak",
                    "token_type": "bearer",
                    "expires_in": 86400,
                    "refresh_token": "operator-refresh-token",
                },
                "granted_scopes": [],
                "token_received_at": 4_102_444_800u64,
                "issuer": "https://operator.example",
            }
        })
        .to_string(),
    )
    .unwrap();

    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            host_home: Some(host_home.path().to_path_buf()),
            ..Default::default()
        },
    )
    .await;

    // The tenant's connector has the operator's server name and no credentials of its own.
    let mut ws = open_session(&svc, "s1.oauth", &[("linear", &fixture.url)], &[]).await;
    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(
        tools.contains(&"mcp__linear__echo".to_string()),
        "the connector must have connected at all: {tools:?}"
    );
    assert!(
        !fixture.saw_header("authorization"),
        "no Authorization header may be attached to a grant connector: {:?}",
        fixture.header_values("authorization")
    );

    // The credential really was there to be found — otherwise the assertion above would hold for
    // the wrong reason. `mcp-logout` reports exactly what the connect path's `has_credential` sees.
    let logout = std::process::Command::new(common::BIN)
        .args(["mcp-logout", "linear"])
        .env("HOME", host_home.path())
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&logout.stdout).contains("logged out: linear"),
        "the planted operator credential must be a real one: {}{}",
        String::from_utf8_lossy(&logout.stdout),
        String::from_utf8_lossy(&logout.stderr)
    );
}

/// A connector URL is refused unless it resolves to a public address. On a replica the interesting
/// targets are all internal: the instance metadata endpoint, a storage mount target, a peer replica
/// — none of which a tenant may reach by pointing an "MCP server" at them.
#[tokio::test]
async fn a_private_address_connector_is_refused_and_never_dialed() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let fixture = spawn_http_mcp_fixture();
    // No `--mcp-allow-private`: this is how a real replica runs.
    let svc = Service::start(&base, &["s1"]).await;

    let mut ws = open_session(
        &svc,
        "s1.ssrf",
        &[
            ("loopback", &fixture.url),
            ("metadata", "http://169.254.169.254/latest/meta-data/"),
        ],
        &[],
    )
    .await;

    let tools = advertised_mcp_tools(&mut ws).await;
    assert!(
        tools.is_empty(),
        "a connector on a private address must contribute no tools: {tools:?}"
    );
    assert_eq!(
        fixture.request_count(),
        0,
        "and must never be dialed at all"
    );

    // The session itself is unharmed — a refused connector is not a refused session.
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "hello" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
}

/// Sanity: the daemon only dials what a grant names. With no connectors in the grant, the session
/// has no MCP tools at all — not the replica operator's, which service mode never connects.
#[tokio::test]
async fn a_grant_with_no_connectors_has_no_mcp_tools() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = service_with_local_connectors(&base).await;
    let mut ws = open_session(&svc, "s1.none", &[], &[]).await;
    let tools: Vec<String> = advertised_mcp_tools(&mut ws).await;
    assert!(tools.is_empty(), "{tools:?}");
}
