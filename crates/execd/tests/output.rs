//! The one-shot answer carries the command's bytes: `stdout` and `stderr` are always standard base64,
//! the empty string for no output, and never a lossy conversion of what the command wrote (the spec,
//! "Bytes").
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use base64::Engine as _;
use beyond_test_support::ports;
use serde_json::{Value, json};
use support::http::post;

const ALL_256: &str =
    r#"i=0; while [ $i -lt 256 ]; do printf "\\$(printf %03o $i)"; i=$((i+1)); done"#;

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

async fn start(opts: impl FnOnce(&mut execd::Options)) -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    opts(&mut o);
    (execd::serve(ports::tokio_listener().await, o).unwrap(), dir)
}

async fn ask(server: &execd::Handle, body: Value) -> Value {
    let a = post(server.local_addr(), &[], &body).await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    a.json()
}

#[tokio::test]
async fn every_byte_value_comes_back_on_both_streams() {
    let (server, _dir) = start(|_| {}).await;
    let all: Vec<u8> = (0..=255u8).collect();
    let a = ask(
        &server,
        json!({
            "command": "sh", "args": ["-c", format!("{ALL_256}; {ALL_256} >&2; exit 5")],
        }),
    )
    .await;
    assert_eq!(a["exit_code"], 5);
    assert_eq!(a["stdout"], b64(&all));
    assert_eq!(a["stderr"], b64(&all));
    server.shutdown().await;
}

#[tokio::test]
async fn both_fields_are_present_even_when_empty() {
    let (server, _dir) = start(|_| {}).await;
    let a = ask(&server, json!({ "command": "true" })).await;
    assert_eq!(a, json!({ "exit_code": 0, "stdout": "", "stderr": "" }));
    server.shutdown().await;
}

#[tokio::test]
async fn a_command_that_cannot_start_is_127_with_its_reason_as_stderr() {
    let (server, _dir) = start(|_| {}).await;
    let a = ask(&server, json!({ "command": "/no/such/command" })).await;
    assert_eq!(a["exit_code"], 127);
    assert_eq!(a["stdout"], "");
    let reason = base64::engine::general_purpose::STANDARD
        .decode(a["stderr"].as_str().unwrap())
        .unwrap();
    assert!(!reason.is_empty());
    server.shutdown().await;
}

#[tokio::test]
async fn the_deadline_note_is_appended_to_the_stderr_bytes() {
    let (server, _dir) = start(|_| {}).await;
    let a = ask(
        &server,
        json!({
            "command": "sh", "args": ["-c", "printf '\\376' >&2; sleep 30"],
            "timeout_ms": 400,
        }),
    )
    .await;
    assert_eq!(a["exit_code"], 124);
    let err = base64::engine::general_purpose::STANDARD
        .decode(a["stderr"].as_str().unwrap())
        .unwrap();
    assert!(err.starts_with(b"\xfe\n"), "{err:?}");
    assert!(err.ends_with(b"command exceeded its timeout"), "{err:?}");
    server.shutdown().await;
}

#[tokio::test]
async fn stdin_bytes_come_back_through_cat() {
    let (server, _dir) = start(|_| {}).await;
    let all: Vec<u8> = (0..=255u8).collect();
    let a = ask(&server, json!({ "command": "cat", "stdin": b64(&all) })).await;
    assert_eq!(a["stdout"], b64(&all));
    server.shutdown().await;
}

#[tokio::test]
async fn the_cap_is_on_the_decoded_bytes() {
    let (server, _dir) = start(|o| o.max_response_bytes = 1000).await;
    let a = ask(
        &server,
        json!({ "command": "sh", "args": ["-c", "head -c 1000 /dev/zero"] }),
    )
    .await;
    // Exactly at the cap is the whole output, though its base64 is longer than the cap.
    assert_eq!(a["stdout"].as_str().unwrap().len(), 1336);
    let over = post(
        server.local_addr(),
        &[],
        &json!({ "command": "sh", "args": ["-c", "head -c 600 /dev/zero; head -c 600 /dev/zero >&2"] }),
    )
    .await;
    assert_eq!(over.status, 422);
    server.shutdown().await;
}

#[tokio::test]
async fn a_stdin_that_is_not_base64_is_a_400() {
    let (server, _dir) = start(|_| {}).await;
    let a = post(
        server.local_addr(),
        &[],
        &json!({ "command": "cat", "stdin": "not base64!" }),
    )
    .await;
    assert_eq!(a.status, 400);
    server.shutdown().await;
}
