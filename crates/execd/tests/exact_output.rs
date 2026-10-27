//! v1.1 `output_base64`: the answer carries the command's bytes as `stdout_base64`/`stderr_base64`
//! and never a lossy conversion of them (the spec, "Exact output").
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
            "output_base64": true,
        }),
    )
    .await;
    assert_eq!(a["exit_code"], 5);
    assert_eq!(a["stdout_base64"], b64(&all));
    assert_eq!(a["stderr_base64"], b64(&all));
    assert!(
        a.get("stdout").is_none() && a.get("stderr").is_none(),
        "instead of the text fields, never beside them: {a}"
    );
    server.shutdown().await;
}

#[tokio::test]
async fn both_fields_are_present_even_when_empty() {
    let (server, _dir) = start(|_| {}).await;
    let a = ask(&server, json!({ "command": "true", "output_base64": true })).await;
    assert_eq!(
        a,
        json!({ "exit_code": 0, "stdout_base64": "", "stderr_base64": "" })
    );
    server.shutdown().await;
}

#[tokio::test]
async fn only_the_json_value_true_asks_for_it() {
    let (server, _dir) = start(|_| {}).await;
    for flag in [
        json!(false),
        json!("true"),
        json!(1),
        json!(null),
        json!([true]),
    ] {
        let a = ask(
            &server,
            json!({ "command": "printf", "args": ["hi"], "output_base64": flag }),
        )
        .await;
        assert_eq!(a["stdout"], "hi", "{flag}: text, as without the field");
        assert!(a.get("stdout_base64").is_none(), "{flag}");
    }
    let a = ask(&server, json!({ "command": "printf", "args": ["hi"] })).await;
    assert_eq!(a["stdout"], "hi");
    server.shutdown().await;
}

#[tokio::test]
async fn text_is_still_lossy_exactly_as_before_when_not_asked() {
    let (server, _dir) = start(|_| {}).await;
    let a = ask(
        &server,
        json!({ "command": "sh", "args": ["-c", r"printf '\377\000\n'"] }),
    )
    .await;
    assert_eq!(a["stdout"], "\u{fffd}\u{0}\n");
    server.shutdown().await;
}

#[tokio::test]
async fn a_command_that_cannot_start_is_127_with_its_reason_as_stderr_bytes() {
    let (server, _dir) = start(|_| {}).await;
    let a = ask(
        &server,
        json!({ "command": "/no/such/command", "output_base64": true }),
    )
    .await;
    assert_eq!(a["exit_code"], 127);
    assert_eq!(a["stdout_base64"], "");
    let reason = base64::engine::general_purpose::STANDARD
        .decode(a["stderr_base64"].as_str().unwrap())
        .unwrap();
    assert!(String::from_utf8(reason).unwrap().contains("No such file"));
    server.shutdown().await;
}

#[tokio::test]
async fn a_deadline_appends_the_note_to_the_decoded_stderr() {
    let (server, _dir) = start(|_| {}).await;
    let a = ask(
        &server,
        json!({
            "command": "sh", "args": ["-c", r"printf '\377' >&2; printf '\376'; sleep 30"],
            "timeout_ms": 400, "output_base64": true,
        }),
    )
    .await;
    assert_eq!(a["exit_code"], 124);
    assert_eq!(a["stdout_base64"], b64(b"\xfe"));
    assert_eq!(
        a["stderr_base64"],
        b64(b"\xff\ncommand exceeded its timeout")
    );
    server.shutdown().await;
}

#[tokio::test]
async fn stdin_and_exact_output_compose() {
    let (server, _dir) = start(|_| {}).await;
    let all: Vec<u8> = (0..=255u8).collect();
    let a = ask(
        &server,
        json!({ "command": "cat", "stdin_base64": b64(&all), "output_base64": true }),
    )
    .await;
    assert_eq!(a["stdout_base64"], b64(&all));
    server.shutdown().await;
}

#[tokio::test]
async fn the_cap_is_on_the_decoded_bytes_and_over_it_is_a_422_never_a_prefix() {
    let (server, _dir) = start(|o| o.max_response_bytes = 1000).await;
    // Exactly the cap: its base64 is 1336 characters, more than the cap, and it is answered.
    let a = ask(
        &server,
        json!({ "command": "sh", "args": ["-c", "head -c 1000 /dev/zero"], "output_base64": true }),
    )
    .await;
    assert_eq!(a["stdout_base64"].as_str().unwrap().len(), 1336);
    // Split across the two streams, together over the cap.
    let over = post(
        server.local_addr(),
        &[],
        &json!({
            "command": "sh", "args": ["-c", "head -c 600 /dev/zero; head -c 600 /dev/zero >&2"],
            "output_base64": true,
        }),
    )
    .await;
    assert_eq!(over.status, 422);
    assert!(String::from_utf8_lossy(&over.body).contains("the command ran"));
    server.shutdown().await;
}
