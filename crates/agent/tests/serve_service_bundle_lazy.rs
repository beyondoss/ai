//! A new session whose grant carries a bundle reads its definition locally, so it sends no exec
//! request until a tool needs the sandbox: the first such call connects (probes once) and runs,
//! and a dead sandbox is that call's error rather than the end of the session. Without a bundle the
//! definition lives in the sandbox, so a new session still probes it at open. Every test counts the
//! requests the exec endpoint saw.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use beyond_ai_test_support::bundle::{BundleServer, file, sha256_hex, tar};
use common::grant::Bundle;
use common::service::{Options, Service};
use common::{
    TestWs, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers, ws_next_frame,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

fn is_probe(r: &common::exec_mock::ExecRequest) -> bool {
    r.command() == "sh" && r.args().iter().any(|a| a.contains(r#"printf %s "$HOME""#))
}

async fn service(base: &str) -> (Service, tempfile::TempDir) {
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

async fn open(
    svc: &Service,
    server: &BundleServer,
    bytes: &[u8],
    exec_url: Option<&str>,
) -> TestWs {
    let token = svc.token_with("t1", "s1.alpha", |c| {
        if let Some(url) = exec_url {
            c.exec_url = url.into();
        }
        c.bundle = Some(Bundle {
            url: server.url.clone(),
            sha256: sha256_hex(bytes),
        });
    });
    ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap()
}

async fn prompt(ws: &mut TestWs, id: &str, message: &str) -> Vec<Value> {
    ws_send(ws, json!({"type": "prompt", "id": id, "message": message})).await;
    let frames = ws_read_until_response(ws, "prompt").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    frames
}

fn bundle_bytes() -> Vec<u8> {
    tar(&[file("AGENTS.md", "BUNDLE-PERSONA: be brief.\n")])
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bundled_session_with_a_sandbox_makes_no_exec_request_until_a_tool_needs_it() {
    let bytes = bundle_bytes();
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![
        turn_text("hello"),
        turn_tool_use("t1", "bash", r#"{"command":"echo LAZY-RAN"}"#),
        turn_text("done"),
    ]);
    let (svc, _cache) = service(&base).await;
    let mut ws = open(&svc, &server, &bytes, None).await;

    prompt(&mut ws, "p1", "hi").await;
    ws_send(&mut ws, json!({"type": "get_state", "id": "g1"})).await;
    ws_read_until_response(&mut ws, "get_state").await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let before = svc.exec.requests();
    assert!(
        before.is_empty(),
        "a text-only turn must not wake the sandbox: {:#?}",
        before.iter().map(|r| r.body.clone()).collect::<Vec<_>>()
    );

    prompt(&mut ws, "p2", "run it").await;
    let after = svc.exec.requests();
    assert_eq!(after.iter().filter(|r| is_probe(r)).count(), 1);
    assert!(is_probe(&after[0]), "the probe comes first");
    let last = requests.lock().unwrap().last().unwrap().clone();
    assert!(last.contains("LAZY-RAN"), "{last}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dead_sandbox_is_the_first_tool_calls_error_not_the_end_of_the_session() {
    let bytes = bundle_bytes();
    let server = BundleServer::start(bytes.clone());
    let (base, requests) = spawn_model_server(vec![
        turn_text("hello"),
        turn_tool_use("t1", "bash", r#"{"command":"echo nope"}"#),
        turn_text("it failed"),
    ]);
    let (svc, _cache) = service(&base).await;
    let mut ws = open(&svc, &server, &bytes, Some("http://127.0.0.1:1/exec")).await;
    prompt(&mut ws, "p1", "hi").await;
    prompt(&mut ws, "p2", "run it").await;
    let last = requests.lock().unwrap().last().unwrap().clone();
    assert!(
        last.contains("this session's sandbox is unreachable"),
        "{last}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_bundle_a_dead_sandbox_still_ends_the_open() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hello")]);
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token_with("t1", "s1.alpha", |c| {
        c.exec_url = "http://127.0.0.1:1/exec".into();
    });
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();
    let mut error = None;
    while let Some(frame) = ws_next_frame(&mut ws).await {
        if frame["type"] == "error" {
            error = frame["error"].as_str().map(str::to_string);
            break;
        }
    }
    assert!(error.is_some_and(|e| e.contains("unreachable")));
}
