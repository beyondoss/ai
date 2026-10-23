//! A grant's optional `limits` claim is a ceiling the tenant's own `AGENTS.md` and its client's
//! commands cannot exceed: a model allow-list, a tool allow-list, a step cap. A disallowed model is
//! refused with a clear error and never swapped; tools are intersected; the step cap is the lower of
//! the two. A grant without `limits` behaves as it always did.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::grant::Limits;
use common::service::Service;
use common::{
    advertised_tools, body_json, spawn_model_server, turn_text, turn_tool_use,
    ws_connect_with_headers, ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

fn limits(models: Option<&[&str]>, tools: Option<&[&str]>, max_steps: Option<u32>) -> Limits {
    let owned = |v: &[&str]| v.iter().map(|s| (*s).to_string()).collect();
    Limits {
        models: models.map(owned),
        tools: tools.map(owned),
        max_steps,
        max_mcp_json_servers: None,
    }
}

/// A model that never stops calling tools, so the number of requests is the number of steps taken.
fn endless_tool_calls() -> Vec<String> {
    let mut turns: Vec<String> = (0..30)
        .map(|i| turn_tool_use(&format!("t{i}"), "read", r#"{"path":"AGENTS.md"}"#))
        .collect();
    turns.push(turn_text("never reached"));
    turns
}

fn without_memory(raw: &str) -> Vec<String> {
    let mut tools = advertised_tools(raw);
    tools.retain(|t| t != "memory");
    tools.sort();
    tools
}

async fn connect(svc: &Service, limits: Limits) -> common::TestWs {
    let token = svc.token_with("t1", "s1.alpha", |c| c.limits = Some(limits));
    ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap()
}

async fn command(ws: &mut common::TestWs, message: Value) -> Vec<Value> {
    let name = message["type"].as_str().unwrap().to_string();
    ws_send(ws, message).await;
    ws_read_until_response(ws, &name).await
}

fn error_text(frames: &[Value]) -> String {
    frames
        .iter()
        .filter_map(|f| f["error"].as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

#[tokio::test]
async fn set_model_outside_the_models_ceiling_is_refused_and_leaves_the_model_alone() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    let mut ws = connect(
        &svc,
        limits(Some(&["claude-test", "claude-ok-*"]), None, None),
    )
    .await;

    let refused = command(
        &mut ws,
        json!({"type":"set_model","id":"m1","model":"claude-other"}),
    )
    .await;
    assert_eq!(refused.last().unwrap()["success"], false, "{refused:#?}");
    let why = error_text(&refused);
    assert!(
        why.contains("claude-other") && why.contains("limits"),
        "the refusal must name the model and the ceiling: {why}"
    );

    // A glob entry admits what it matches.
    let allowed = command(
        &mut ws,
        json!({"type":"set_model","id":"m2","model":"claude-ok-9"}),
    )
    .await;
    assert_eq!(allowed.last().unwrap()["success"], true, "{allowed:#?}");
    let frames = command(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        body_json(&requests.lock().unwrap()[0])["model"],
        "claude-ok-9"
    );
}

#[tokio::test]
async fn an_agents_md_model_outside_the_ceiling_refuses_the_session_instead_of_swapping_it() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace(
        "AGENTS.md",
        "---\nmodel: claude-forbidden\n---\nBe brief.\n",
    );
    let mut ws = connect(&svc, limits(Some(&["claude-test"]), None, None)).await;

    let frames = command(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    let why = error_text(&frames);
    assert!(
        why.contains("claude-forbidden") && why.contains("limits"),
        "the session must fail to start with a clear error: {frames:#?}"
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "no model may be called, on either model"
    );
}

#[tokio::test]
async fn an_agents_md_model_inside_the_ceiling_is_used() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace("AGENTS.md", "---\nmodel: claude-ok-fm\n---\nBe brief.\n");
    let mut ws = connect(&svc, limits(Some(&["claude-ok-*"]), None, None)).await;

    let frames = command(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        body_json(&requests.lock().unwrap()[0])["model"],
        "claude-ok-fm"
    );
}

#[tokio::test]
async fn the_replica_default_model_outside_the_ceiling_is_refused_too() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    let mut ws = connect(&svc, limits(Some(&["claude-only-this"]), None, None)).await;

    let frames = command(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    assert!(error_text(&frames).contains("claude-test"), "{frames:#?}");
    assert!(requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_tool_ceiling_is_intersected_with_the_agents_md_allow_list() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace("AGENTS.md", "---\ntools: read, grep\n---\nBe brief.\n");
    let mut ws = connect(&svc, limits(None, Some(&["read", "edit"]), None)).await;

    let frames = command(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(without_memory(&requests.lock().unwrap()[0]), vec!["read"]);
}

#[tokio::test]
async fn the_tool_ceiling_alone_is_the_allow_list() {
    let (base, requests) = spawn_model_server(vec![turn_text("hi")]);
    let svc = Service::start(&base, &["s1"]).await;
    let mut ws = connect(&svc, limits(None, Some(&["read", "grep"]), None)).await;

    let frames = command(
        &mut ws,
        json!({"type":"prompt","id":"p1","message":"hello"}),
    )
    .await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert_eq!(
        without_memory(&requests.lock().unwrap()[0]),
        vec!["grep", "read"]
    );
}

#[tokio::test]
async fn max_steps_is_the_lower_of_the_agents_md_and_the_ceiling() {
    // The file asks for 20 steps, the grant allows 2: the grant wins.
    let (base, requests) = spawn_model_server(endless_tool_calls());
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace("AGENTS.md", "---\nmax-steps: 20\n---\nBe brief.\n");
    let mut ws = connect(&svc, limits(None, None, Some(2))).await;
    command(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    let capped = requests.lock().unwrap().len();
    assert!(
        capped < 12,
        "the ceiling of 2 must win over 20, got {capped} model turns"
    );

    // The file asks for 2, the grant allows 20: the file wins.
    let (base, requests) = spawn_model_server(endless_tool_calls());
    let svc = Service::start(&base, &["s1"]).await;
    svc.write_in_workspace("AGENTS.md", "---\nmax-steps: 2\n---\nBe brief.\n");
    let mut ws = connect(&svc, limits(None, None, Some(20))).await;
    command(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    assert_eq!(
        requests.lock().unwrap().len(),
        capped,
        "both caps are 2, so both runs take the same steps"
    );

    // A ceiling with no file of its own still caps the run.
    let (base, requests) = spawn_model_server(endless_tool_calls());
    let svc = Service::start(&base, &["s1"]).await;
    let mut ws = connect(&svc, limits(None, None, Some(2))).await;
    command(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    assert_eq!(requests.lock().unwrap().len(), capped);
}

#[tokio::test]
async fn a_grant_without_limits_changes_nothing() {
    let (base, requests) = spawn_model_server(endless_tool_calls());
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("t1", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    let set = command(
        &mut ws,
        json!({"type":"set_model","id":"m1","model":"claude-anything"}),
    )
    .await;
    assert_eq!(set.last().unwrap()["success"], true, "{set:#?}");
    command(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    let seen = requests.lock().unwrap();
    assert!(
        seen.len() > 12,
        "no ceiling means no step cap, got {}",
        seen.len()
    );
    assert_eq!(body_json(&seen[0])["model"], "claude-anything");
    assert!(advertised_tools(&seen[0]).contains(&"bash".to_string()));
}
