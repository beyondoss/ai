//! An ambient proxy (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`) must not route around the SSRF guard.
//!
//! The guard lives in the clients' DNS resolver (`tools::web::ssrf::SsrfResolver`): every address a
//! connection is about to dial is checked there. Through a proxy, the client never resolves the
//! target at all — it dials the proxy and hands it the hostname — so a grant connector or a `web`
//! fetch naming `localhost` (or anything resolving inward) would be fetched by the proxy on our
//! behalf. Both egress clients ignore the environment's proxies; these prove the proxy sees nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::mcp_fixture::spawn_http_mcp_fixture;
use common::service::{Options, Service};
use common::{
    run_cmd, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers,
    ws_read_until_response, ws_send,
};
use serde_json::json;

/// A listener standing in for a proxy: it counts connections and answers none of them.
fn spawn_proxy() -> (String, Arc<AtomicUsize>) {
    let listener = beyond_ai_test_support::ports::listener();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            counter.fetch_add(1, Ordering::SeqCst);
            // Hold it open briefly, then drop: a request through here never completes.
            drop(conn);
        }
    });
    (format!("http://{addr}"), hits)
}

/// Every proxy variable pointed at `proxy`. `NO_PROXY` exempts only `127.0.0.1`, the literal the
/// mock model, exec endpoint and replica are reached by — so a target named `localhost` is one the
/// environment would send through the proxy.
fn proxy_env(proxy: &str) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = [
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    .into_iter()
    .map(|k| (k.to_string(), proxy.to_string()))
    .collect();
    env.push(("NO_PROXY".into(), "127.0.0.1".into()));
    env.push(("no_proxy".into(), "127.0.0.1".into()));
    env
}

/// A grant connector to `localhost` on a production-shaped replica (no `--mcp-allow-private`):
/// refused by the resolver, never dialed — directly or through the proxy.
#[tokio::test]
async fn a_proxy_in_the_environment_does_not_carry_a_grant_connector_past_the_ssrf_guard() {
    let (proxy, hits) = spawn_proxy();
    let fixture = spawn_http_mcp_fixture();
    let (base, _requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            env: proxy_env(&proxy),
            ..Default::default()
        },
    )
    .await;

    let url = fixture.url.replace("127.0.0.1", "localhost");
    let mut claims = svc.claims("tenant-a", "s1.proxy", &svc.shards[0].0);
    claims.mcp = vec![("loopback".into(), url)];
    let token = svc.minter.mint(&claims, &svc.secrets());
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.proxy"), &svc.header(&token))
        .await
        .unwrap_or_else(|status| panic!("session refused with HTTP {status}"));

    ws_send(&mut ws, json!({ "type": "get_mcp", "id": "m1" })).await;
    let frames = ws_read_until_response(&mut ws, "get_mcp").await;
    let response = frames
        .iter()
        .rev()
        .find(|f| f["type"] == "response" && f["command"] == "get_mcp")
        .unwrap();
    let tools = response["data"]["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        tools.is_empty(),
        "a loopback connector must contribute no tools: {tools:?}"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "the connector went to the proxy, which would have dialed localhost for us"
    );
    assert_eq!(fixture.request_count(), 0);
}

/// The `web` tool, same environment: a fetch of `localhost` is refused, and the proxy sees nothing.
#[test]
fn a_proxy_in_the_environment_does_not_carry_a_web_fetch_past_the_ssrf_guard() {
    let (proxy, hits) = spawn_proxy();
    let target = beyond_ai_test_support::ports::listener();
    let port = target.local_addr().unwrap().port();
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use(
            "tu_1",
            "web",
            &json!({ "url": format!("http://localhost:{port}/") }).to_string(),
        ),
        turn_text("done"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .envs(proxy_env(&proxy))
        .args([
            "run",
            "fetch it",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "3",
            "--no-session-persistence",
            "--tools",
            "web",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "run failed: {stderr}");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "the fetch went to the proxy, which would have dialed localhost for us"
    );
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "{bodies:#?}");
    let body: serde_json::Value =
        serde_json::from_str(bodies[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    let result = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().cloned().unwrap_or_default())
        .find(|b| b["type"] == "tool_result" && b["tool_use_id"] == "tu_1")
        .expect("no tool_result for the fetch");
    assert_eq!(result["is_error"], true, "{result}");
    assert!(
        result.to_string().contains("loopback"),
        "the tool result should be the SSRF refusal: {result}"
    );
}
