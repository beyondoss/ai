//! MCP questions on a multi-session `serve --listen` daemon go to the session whose call asked.
//!
//! The operator's MCP servers are connected once and shared by every session of the daemon. Each
//! session used to install its elicitation and sampling gates into one process-wide hub, so the
//! last session to start answered everyone's servers, and once it detached every session's
//! questions were refused. Each session now has its own hub, found by the calls its runs make.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;
mod mcp_tasks_env;

use std::process::{Command, Stdio};
use std::time::Duration;

use common::{
    ChildGuard, SpawnGuarded, TestWs, free_port, spawn_model_server_routed, turn_text,
    turn_tool_use, wait_for_port, ws_connect, ws_next_frame, ws_send,
};
use mcp_tasks_env::Env;
use serde_json::{Value, json};

fn daemon(env: &Env, base: &str, port: u16) -> ChildGuard {
    Command::new(common::BIN)
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            env.dir.path().join("sessions").to_str().unwrap(),
        ])
        .env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

/// Frames until `pred` matches (that frame last), failing after 30 s.
async fn until(ws: &mut TestWs, what: &str, pred: impl Fn(&Value) -> bool) -> Vec<Value> {
    let mut frames = Vec::new();
    let read = async {
        while let Some(frame) = ws_next_frame(ws).await {
            let done = pred(&frame);
            frames.push(frame);
            if done {
                return true;
            }
        }
        false
    };
    let found = tokio::time::timeout(Duration::from_secs(30), read)
        .await
        .unwrap_or(false);
    assert!(found, "never saw {what}: {frames:?}");
    frames
}

fn is_response(frame: &Value, command: &str) -> bool {
    frame["type"] == "response" && frame["command"] == command
}

fn tool_end_result(frames: &[Value]) -> String {
    frames
        .iter()
        .filter(|f| f["type"] == "event" && f["event"]["kind"] == "tool_end")
        .map(|f| f["event"]["result"].as_str().unwrap_or_default().to_owned())
        .collect::<Vec<_>>()
        .join("\n")
}

/// No frame of `kind` arrives on `ws` within a short window.
async fn quiet(ws: &mut TestWs, kind: &str) {
    let seen = tokio::time::timeout(Duration::from_millis(500), async {
        while let Some(frame) = ws_next_frame(ws).await {
            if frame["type"] == kind {
                return Some(frame);
            }
        }
        None
    })
    .await
    .ok()
    .flatten();
    assert!(
        seen.is_none(),
        "a {kind} reached the wrong session: {seen:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_routes_mcp_questions_to_the_session_whose_call_asked() {
    let env = Env::new();
    env.settings(json!([
        env.stdio_with(json!({})),
        {
            "name": "old",
            "transport": "stdio",
            "command": env!("CARGO_BIN_EXE_mcp_fixture_stdio_server"),
            "args": [],
            "env": {},
        },
    ]));
    // Routes are matched first-to-last, so each later state of a conversation comes first.
    let (base, _requests) = spawn_model_server_routed(
        vec![
            ("hello-Ferris".into(), turn_text("alpha-done-2")),
            (
                "MARKER-ALPHA-2".into(),
                turn_tool_use("toolu_a2", "mcp__old__ask_user", "{}"),
            ),
            ("sampled:Ferris".into(), turn_text("alpha-done-1")),
            (
                "MARKER-ALPHA-1".into(),
                turn_tool_use("toolu_a1", "mcp__t__sample_task", "{}"),
            ),
            ("sampled:Bravo".into(), turn_text("bravo-done")),
            (
                "MARKER-BRAVO".into(),
                turn_tool_use("toolu_b", "mcp__t__sample_task", "{}"),
            ),
        ],
        turn_text("fallback"),
    );
    let port = free_port();
    let _daemon = daemon(&env, &base, port);
    wait_for_port(port);

    let mut a = ws_connect(port, Some("mcp-alpha")).await;
    // Bravo starts second: under a process-wide hub its gates would answer for alpha too.
    let mut b = ws_connect(port, Some("mcp-bravo")).await;
    ws_send(&mut b, json!({ "type": "get_state", "id": "s" })).await;
    until(&mut b, "bravo's session to be up", |f| {
        is_response(f, "get_state")
    })
    .await;

    // An in-task sampling request from alpha's call reaches alpha.
    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "p1", "message": "MARKER-ALPHA-1" }),
    )
    .await;
    let frames = until(&mut a, "alpha's sampling_request", |f| {
        f["type"] == "sampling_request"
    })
    .await;
    let ask = frames.last().unwrap().clone();
    quiet(&mut b, "sampling_request").await;
    ws_send(
        &mut a,
        json!({
            "type": "sample",
            "request_id": ask["request_id"],
            "result": {
                "role": "assistant",
                "content": { "type": "text", "text": "Ferris" },
                "model": "host",
                "stopReason": "endTurn",
            },
        }),
    )
    .await;
    let frames = until(&mut a, "alpha's first prompt", |f| is_response(f, "prompt")).await;
    assert!(
        tool_end_result(&frames).contains("sampled:Ferris"),
        "{frames:?}"
    );

    // A classic nested elicitation (sent by the server mid-`tools/call`) from alpha reaches alpha,
    // even after bravo has detached.
    drop(b);
    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "p2", "message": "MARKER-ALPHA-2" }),
    )
    .await;
    let frames = until(&mut a, "alpha's elicitation_request", |f| {
        f["type"] == "elicitation_request"
    })
    .await;
    let ask = frames.last().unwrap().clone();
    ws_send(
        &mut a,
        json!({
            "type": "elicit",
            "request_id": ask["request_id"],
            "action": "accept",
            "content": { "name": "Ferris" },
        }),
    )
    .await;
    let frames = until(&mut a, "alpha's second prompt", |f| {
        is_response(f, "prompt")
    })
    .await;
    assert!(
        tool_end_result(&frames).contains("hello-Ferris"),
        "{frames:?}"
    );

    // And bravo, re-attached, gets its own.
    let mut b = ws_connect(port, Some("mcp-bravo")).await;
    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "MARKER-BRAVO" }),
    )
    .await;
    let frames = until(&mut b, "bravo's sampling_request", |f| {
        f["type"] == "sampling_request"
    })
    .await;
    let ask = frames.last().unwrap().clone();
    quiet(&mut a, "sampling_request").await;
    ws_send(
        &mut b,
        json!({
            "type": "sample",
            "request_id": ask["request_id"],
            "result": {
                "role": "assistant",
                "content": { "type": "text", "text": "Bravo" },
                "model": "host",
                "stopReason": "endTurn",
            },
        }),
    )
    .await;
    let frames = until(&mut b, "bravo's prompt", |f| is_response(f, "prompt")).await;
    assert!(
        tool_end_result(&frames).contains("sampled:Bravo"),
        "{frames:?}"
    );
}

/// All frames for `ms`.
async fn collect(ws: &mut TestWs, ms: u64) -> Vec<Value> {
    let mut frames = Vec::new();
    let _ = tokio::time::timeout(Duration::from_millis(ms), async {
        while let Some(f) = ws_next_frame(ws).await {
            frames.push(f);
        }
    })
    .await;
    frames
}

fn concurrent_routes() -> Vec<(String, String)> {
    vec![
        ("asked:".into(), turn_text("alpha-done")),
        ("held".into(), turn_text("bravo-done")),
        (
            "MARKER-ALPHA".into(),
            turn_tool_use("toolu_a", "mcp__t__nested_ask", "{}"),
        ),
        (
            "MARKER-BRAVO".into(),
            turn_tool_use("toolu_b", "mcp__t__hold", "{}"),
        ),
    ]
}

/// Alpha's call raises a nested elicitation 1.5 s in; bravo's call starts 0.4 s after alpha's and
/// is still in flight then. Returns both sessions' frames for the next 3.5 s.
async fn alpha_asks_while_bravo_holds(env: &Env) -> (TestWs, Vec<Value>, Vec<Value>, ChildGuard) {
    let (base, _requests) = spawn_model_server_routed(concurrent_routes(), turn_text("fallback"));
    let port = free_port();
    let daemon = daemon(env, &base, port);
    wait_for_port(port);
    let mut a = ws_connect(port, Some("route-alpha")).await;
    let mut b = ws_connect(port, Some("route-bravo")).await;
    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "pa", "message": "MARKER-ALPHA" }),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "MARKER-BRAVO" }),
    )
    .await;
    let (fa, fb) = tokio::join!(collect(&mut a, 3500), collect(&mut b, 3500));
    (a, fa, fb, daemon)
}

fn elicitations(frames: &[Value]) -> Vec<&Value> {
    frames
        .iter()
        .filter(|f| f["type"] == "elicitation_request")
        .collect()
}

/// N1 (adopted from the audit's probe): over stdio a nested request carries nothing that names
/// its call. With calls from two sessions in flight on the shared server it is refused: neither
/// session's client sees it, and alpha's call gets the refusal as its answer. Never a guess.
#[tokio::test(flavor = "multi_thread")]
async fn daemon_refuses_an_unattributable_nested_request_rather_than_guessing() {
    let env = Env::new();
    env.stdio();
    let (mut a, mut fa, fb, _daemon) = alpha_asks_while_bravo_holds(&env).await;
    assert!(
        elicitations(&fb).is_empty(),
        "alpha's request reached bravo: {fb:?}"
    );
    assert!(
        elicitations(&fa).is_empty(),
        "an unattributable request must reach nobody: {fa:?}"
    );
    if !fa.iter().any(|f| is_response(f, "prompt")) {
        fa.extend(until(&mut a, "alpha's prompt", |f| is_response(f, "prompt")).await);
    }
    let result = tool_end_result(&fa);
    assert!(
        result.contains("asked:") && result.contains("more than one session"),
        "alpha's call is answered with the refusal: {result}"
    );
}

/// N1: over Streamable HTTP the nested request arrives on the SSE stream of the POST that raised
/// it, so it is attributed to alpha's call even while bravo's is in flight.
#[tokio::test(flavor = "multi_thread")]
async fn daemon_attributes_a_nested_request_by_its_http_stream() {
    let env = Env::new();
    let _server = env.http();
    let (mut a, fa, fb, _daemon) = alpha_asks_while_bravo_holds(&env).await;
    assert!(
        elicitations(&fb).is_empty(),
        "alpha's request reached bravo: {fb:?}"
    );
    let asks = elicitations(&fa);
    assert_eq!(asks.len(), 1, "alpha gets its own request: {fa:?}");
    ws_send(
        &mut a,
        json!({
            "type": "elicit",
            "request_id": asks[0]["request_id"],
            "action": "accept",
            "content": { "name": "Ferris" },
        }),
    )
    .await;
    let frames = until(&mut a, "alpha's prompt", |f| is_response(f, "prompt")).await;
    let result = tool_end_result(&frames);
    assert!(
        result.contains("asked:") && result.contains("Ferris"),
        "{result}"
    );
}
