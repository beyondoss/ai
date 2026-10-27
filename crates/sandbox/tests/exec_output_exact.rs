//! Raw exec returns the command's bytes (`run_exact`): over `output_base64` when the endpoint
//! honours it, over a v1.2 stream when it only streams, and as text said to be inexact when it can do
//! neither. Against the mock endpoint in each shape; the decoding vectors are
//! `tests/fixtures/exec/v1.1-output.json`, run by the unit tests in `exec_endpoint/exact.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use beyond_sandbox::Sandbox;
use beyond_sandbox::exec::CommandRunner;
use beyond_sandbox::exec_endpoint::HttpExecRunner;
use beyond_test_support::exec_mock::{ExecMock, MockOptions, Protocol};
use serde_json::Value;

const SECS: Duration = Duration::from_secs(20);

/// Writes every byte value, 0x00 to 0xFF, to the descriptor in `$1`.
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

fn text_only(protocol: Protocol) -> MockOptions {
    MockOptions {
        protocol,
        honors_output_base64: false,
        ..MockOptions::default()
    }
}

/// Which of the endpoint's requests carried which of the optional fields.
fn requests(mock: &ExecMock) -> Vec<(bool, bool)> {
    mock.bodies()
        .iter()
        .map(|b| {
            (
                b["output_base64"] == Value::Bool(true),
                b["stream"] == Value::Bool(true),
            )
        })
        .collect()
}

async fn all_256(runner: &HttpExecRunner) -> beyond_sandbox::ExecOutput {
    runner
        .run_exact(
            "sh",
            &sh(&format!("{ALL_256}; {ALL_256} >&2")),
            None,
            SECS,
            None,
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn an_endpoint_that_honours_output_base64_returns_every_byte_value_exactly() {
    let (mock, _dir) = mock(MockOptions::default()).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let out = all_256(&runner).await;
    assert!(out.exact);
    assert_eq!(out.code, 0);
    assert_eq!(out.stdout, all_bytes());
    assert_eq!(out.stderr, all_bytes());
    // The probe, then the call: both carry the field, neither streams.
    assert_eq!(requests(&mock), [(true, false), (true, false)]);
    // Learned once per connection, not per call.
    all_256(&runner).await;
    assert_eq!(requests(&mock).len(), 3);
}

#[tokio::test]
async fn an_endpoint_that_only_streams_is_exact_over_the_stream() {
    let (mock, _dir) = mock(text_only(Protocol::Streaming)).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let out = all_256(&runner).await;
    assert!(out.exact);
    assert_eq!((out.stdout, out.stderr), (all_bytes(), all_bytes()));
    let seen = requests(&mock);
    assert_eq!(
        seen[0],
        (true, false),
        "the output_base64 probe, answered with text"
    );
    assert!(
        seen.contains(&(false, true)),
        "then the stream probe and the call: {seen:?}"
    );
    assert!(
        !seen.iter().any(|(b64, streamed)| *b64 && *streamed),
        "a stream never carries output_base64"
    );
}

#[tokio::test]
async fn a_plain_v11_endpoint_answers_text_and_says_it_is_not_exact() {
    let (mock, _dir) = mock(text_only(Protocol::V11)).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let out = runner
        .run_exact(
            "sh",
            &sh(r"printf 'caf\303\251 \377'; printf e >&2; exit 4"),
            None,
            SECS,
            None,
        )
        .await
        .unwrap();
    assert!(!out.exact, "the text the endpoint gave is not the bytes");
    assert_eq!(out.code, 4);
    assert_eq!(
        out.stdout_text(),
        "café \u{fffd}",
        "what the endpoint answered, as UTF-8"
    );
    assert_eq!(out.stderr, b"e");
}

#[tokio::test]
async fn a_call_with_stdin_on_an_endpoint_without_output_base64_is_text_not_exact() {
    let (mock, _dir) = mock(text_only(Protocol::Streaming)).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let out = runner
        .run_exact("cat", &[], None, SECS, Some(b"\xff\x00"))
        .await
        .unwrap();
    assert!(!out.exact);
    // Over output_base64 the same call is exact.
    let (mock, _dir) = self::mock(MockOptions::default()).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let out = runner
        .run_exact("cat", &[], None, SECS, Some(b"\xff\x00"))
        .await
        .unwrap();
    assert!(out.exact);
    assert_eq!(out.stdout, b"\xff\x00");
}

#[tokio::test]
async fn a_strict_front_that_refuses_the_field_has_answered_not_supported() {
    let (mock, _dir) = mock(MockOptions {
        refuses_extensions: true,
        ..MockOptions::default()
    })
    .await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    assert!(!runner.outputs_base64().await);
    let out = runner
        .run_exact("sh", &sh("printf hi"), None, SECS, None)
        .await
        .unwrap();
    assert!(!out.exact);
    assert_eq!(out.stdout, b"hi");
}

#[tokio::test]
async fn a_probe_whose_request_failed_is_an_error_never_not_supported() {
    let (mock, _dir) = mock(MockOptions {
        require_bearer: Some("secret".into()),
        ..MockOptions::default()
    })
    .await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let err = runner
        .run_exact("sh", &sh("printf hi"), None, SECS, None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");
    assert!(
        runner.try_outputs_base64().await.is_err(),
        "nothing was cached"
    );
    // Once the endpoint is reachable the same runner learns the truth.
    let authed = HttpExecRunner::new(&mock.url)
        .unwrap()
        .with_header("Authorization", "Bearer secret");
    assert!(authed.outputs_base64().await);

    let gone = HttpExecRunner::new("http://127.0.0.1:1/exec").unwrap();
    assert!(gone.run_exact("true", &[], None, SECS, None).await.is_err());
}

#[tokio::test]
async fn output_past_the_cap_is_an_error_that_says_the_command_ran() {
    for options in [MockOptions::default(), text_only(Protocol::Streaming)] {
        let (mock, _dir) = self::mock(options).await;
        let runner = HttpExecRunner::new(&mock.url)
            .unwrap()
            .with_max_response_bytes(1000);
        // 600 + 600 decoded bytes, together over a cap of 1000; each alone under it.
        let err = runner
            .run_exact(
                "sh",
                &sh("head -c 600 /dev/zero; head -c 600 /dev/zero >&2"),
                None,
                SECS,
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::FileTooLarge, "{err}");
        assert!(err.to_string().contains("the command ran"), "{err}");
        // At the cap exactly is fine, though its base64 is longer than the cap.
        let ok = runner
            .run_exact("sh", &sh("head -c 1000 /dev/zero"), None, SECS, None)
            .await
            .unwrap();
        assert_eq!(ok.stdout.len(), 1000);
    }
}

#[tokio::test]
async fn a_stream_one_shot_is_answered_at_the_shell_exit_not_when_leftovers_end() {
    let (mock, _dir) = mock(text_only(Protocol::Streaming)).await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let started = Instant::now();
    let out = runner
        .run_exact("sh", &sh("sleep 30 & printf hi"), None, SECS, None)
        .await
        .unwrap();
    assert_eq!(
        (out.stdout.as_slice(), out.code, out.exact),
        (&b"hi"[..], 0, true)
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "waited {:?} for a background process",
        started.elapsed()
    );
}

#[tokio::test]
async fn signals_and_deadlines_have_their_v11_shape_on_every_exact_path() {
    for options in [MockOptions::default(), text_only(Protocol::Streaming)] {
        let (mock, _dir) = self::mock(options).await;
        let runner = HttpExecRunner::new(&mock.url).unwrap();
        let killed = runner
            .run_exact("sh", &sh("printf '\\377'; kill -9 $$"), None, SECS, None)
            .await
            .unwrap();
        assert!(killed.exact);
        assert_eq!(killed.code, 137);
        let slow = runner
            .run_exact(
                "sh",
                &sh("printf '\\377' >&2; sleep 30"),
                None,
                Duration::from_millis(600),
                None,
            )
            .await
            .unwrap();
        assert!(slow.exact);
        assert_eq!(slow.code, 124);
        assert!(slow.stderr.starts_with(b"\xff\n"), "{:?}", slow.stderr);
        assert!(slow.stderr.ends_with(b"command exceeded its timeout"));
    }
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
    assert!(out.exact);
    assert_eq!(out.stdout, [0xff, 0x00]);
}
