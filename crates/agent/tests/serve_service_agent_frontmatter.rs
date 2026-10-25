//! A hosted session reads its agent definition from the tenant's own `AGENTS.md`: the frontmatter
//! picks the model, narrows the tools and caps the steps, exactly as it does locally, and never
//! reaches the model as prompt text.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use common::service::Service;
use common::{
    advertised_tools, body_json, spawn_model_server, turn_text, turn_tool_use,
    ws_connect_with_headers, ws_read_until_response, ws_send,
};
use serde_json::json;

const AGENTS_MD: &str = "---\nmodel: claude-hosted-frontmatter\ntools: read, grep\nmax-steps: 1\n---\nHOSTED-HOUSE-STYLE: answer tersely.\n";

/// How many model turns a run takes when the model never stops calling tools: the cap, plus the
/// wrap-up turns the loop grants past it.
async fn requests_until_the_run_ends(agents_md: &str) -> (Vec<String>, serde_json::Value) {
    let mut turns: Vec<String> = (0..30)
        .map(|i| turn_tool_use(&format!("t{i}"), "read", r#"{"path":"AGENTS.md"}"#))
        .collect();
    turns.push(turn_text("never reached"));
    let (base, requests) = spawn_model_server(turns);
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace("AGENTS.md", agents_md);

    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    ws_send(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    let last = frames.last().unwrap().clone();
    let seen = requests.lock().unwrap().clone();
    (seen, last)
}

#[tokio::test]
async fn the_tenants_agents_md_frontmatter_configures_the_hosted_session() {
    let (requests, _response) = requests_until_the_run_ends(AGENTS_MD).await;

    for raw in &requests {
        assert_eq!(body_json(raw)["model"], "claude-hosted-frontmatter");
        let mut tools = advertised_tools(raw);
        tools.retain(|t| t != "memory");
        tools.sort();
        assert_eq!(tools, vec!["grep".to_string(), "read".to_string()]);
    }
    // `max-steps: 1` stops a model that never stops calling tools after the one step and the wrap-up
    // turns past it; uncapped it would have consumed all 30.
    assert!(
        requests.len() < 10,
        "max-steps: 1 must stop the run, got {} model turns",
        requests.len()
    );
    // The instructions reach the model; the frontmatter that configured it does not.
    let system = body_json(&requests[0])["system"].to_string();
    assert!(system.contains("HOSTED-HOUSE-STYLE"), "{system}");
    assert!(!system.contains("max-steps"), "{system}");
    assert!(!system.contains("claude-hosted-frontmatter"), "{system}");
}

#[tokio::test]
async fn an_agents_md_without_frontmatter_leaves_the_hosted_defaults_alone() {
    let (requests, _response) =
        requests_until_the_run_ends("HOSTED-HOUSE-STYLE: answer tersely.\n").await;

    assert_eq!(body_json(&requests[0])["model"], "claude-test");
    assert!(advertised_tools(&requests[0]).contains(&"bash".to_string()));
    assert!(
        requests.len() > 10,
        "no max-steps means no cap, got {} model turns",
        requests.len()
    );
}
