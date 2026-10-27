//! Raw exec returns the command's bytes: `stdout` and `stderr` are always standard base64 on the
//! wire, so every byte value survives, the cap is on the decoded bytes, and an answer that is not
//! base64 is an error. Against the mock endpoint (execd); the decoding vectors are
//! `tests/fixtures/exec/output.json`, run by the unit tests in `exec_endpoint.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use beyond_sandbox::Sandbox;
use beyond_sandbox::exec::CommandRunner;
use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_test_support::exec_mock::{ExecMock, MockOptions};

const SECS: Duration = Duration::from_secs(20);

/// Writes every byte value, 0x00 to 0xFF.
const ALL_256: &str =
    r#"i=0; while [ $i -lt 256 ]; do printf "\\$(printf %03o $i)"; i=$((i+1)); done"#;

fn sh(script: &str) -> Vec<String> {
    vec!["-c".to_string(), script.to_string()]
}

fn all_bytes() -> Vec<u8> {
    (0..=255u8).collect()
}

async fn mock(options: MockOptions) -> (ExecMock, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    (ExecMock::start_with(dir.path(), options).await, dir)
}

#[tokio::test]
async fn every_byte_value_survives_on_both_streams() {
    let (mock, _dir) = mock(MockOptions::default()).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let out = runner
        .run(
            "sh",
            &sh(&format!("{ALL_256}; {ALL_256} >&2; exit 3")),
            None,
            SECS,
        )
        .await
        .unwrap();
    assert_eq!(out.code, Some(3));
    assert_eq!(out.stdout, all_bytes());
    assert_eq!(out.stderr, all_bytes());
}

#[tokio::test]
async fn stdin_bytes_round_trip_through_cat() {
    let (mock, _dir) = mock(MockOptions::default()).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let out = runner
        .run_with_stdin("cat", &[], None, SECS, &all_bytes())
        .await
        .unwrap();
    assert_eq!(out.stdout, all_bytes());
}

#[tokio::test]
async fn the_request_carries_stdin_and_nothing_about_output() {
    let (mock, _dir) = mock(MockOptions::default()).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    runner
        .run_with_stdin("cat", &[], None, SECS, b"hi")
        .await
        .unwrap();
    let body = mock.bodies().pop().unwrap();
    assert_eq!(body["stdin"], "aGk=");
    let mut keys: Vec<_> = body.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["args", "command", "stdin", "timeout_ms"]);
}

#[tokio::test]
async fn an_answer_that_is_not_base64_is_an_error() {
    let (mock, _dir) = mock(MockOptions {
        corrupts_base64: true,
        ..MockOptions::default()
    })
    .await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let err = runner
        .run("sh", &sh("printf hi"), None, SECS)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not standard base64"), "{err}");
}

#[tokio::test]
async fn an_unreachable_or_refused_endpoint_is_an_error() {
    let (mock, _dir) = mock(MockOptions {
        require_bearer: Some("secret".into()),
        ..MockOptions::default()
    })
    .await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let err = runner
        .run("sh", &sh("printf hi"), None, SECS)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");
    let gone = HttpExecRunner::new("http://127.0.0.1:1/exec").unwrap();
    assert!(gone.run("true", &[], None, SECS).await.is_err());
}

#[tokio::test]
async fn output_past_the_cap_is_an_error_that_says_the_command_ran() {
    let (mock, _dir) = mock(MockOptions::default()).await;
    let runner = HttpExecRunner::new(&mock.url)
        .unwrap()
        .with_max_response_bytes(1000);
    // 600 + 600 decoded bytes, together over a cap of 1000; each alone under it.
    let err = runner
        .run(
            "sh",
            &sh("head -c 600 /dev/zero; head -c 600 /dev/zero >&2"),
            None,
            SECS,
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::FileTooLarge, "{err}");
    assert!(err.to_string().contains("the command ran"), "{err}");
    // At the cap exactly is fine, though its base64 is longer than the cap.
    let ok = runner
        .run("sh", &sh("head -c 1000 /dev/zero"), None, SECS)
        .await
        .unwrap();
    assert_eq!(ok.stdout.len(), 1000);
}

#[tokio::test]
async fn signals_and_deadlines_have_their_shape() {
    let (mock, _dir) = mock(MockOptions::default()).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let killed = runner
        .run("sh", &sh("printf '\\377'; kill -9 $$"), None, SECS)
        .await
        .unwrap();
    assert_eq!(killed.code, Some(137));
    assert_eq!(killed.stdout, [0xff]);
    let slow = runner
        .run(
            "sh",
            &sh("printf '\\377' >&2; sleep 30"),
            None,
            Duration::from_millis(600),
        )
        .await
        .unwrap();
    assert_eq!(slow.code, Some(124));
    assert!(slow.stderr.starts_with(b"\xff\n"), "{:?}", slow.stderr);
    assert!(slow.stderr.ends_with(b"command exceeded its timeout"));
}

#[tokio::test]
async fn the_sandbox_surface_returns_bytes() {
    let (mock, _dir) = mock(MockOptions::default()).await;
    let sandbox = Sandbox::connect(&mock.url, beyond_sandbox::TokenSource::fixed("t"))
        .await
        .unwrap();
    let out = sandbox
        .exec("sh", ["-c", "printf '\\377\\000'"])
        .run()
        .await
        .unwrap();
    assert_eq!(out.stdout, [0xff, 0x00]);
}
