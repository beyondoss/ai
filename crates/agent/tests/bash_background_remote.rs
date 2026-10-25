//! `run_in_background` and `timeout_ms` over a remote exec endpoint. On a streaming (v1.2) endpoint
//! both behave as on this host: a background start returns with a job whose stream is its lifetime,
//! and a `timeout_ms` returns the call while the command keeps running. A v1.1 endpoint can neither
//! stream nor stop a command, so `run_in_background` is refused before anything runs, and a
//! `timeout_ms` it outlived is said to have had no effect. The wire carries the deadlines `bash`
//! derives: `i32::MAX` ms on a stream, 30 minutes on v1.1.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_core::Background;
use beyond_ai_agent::exec_endpoint::{HttpExecRunner, STREAM_DEADLINE};
use beyond_ai_agent::tools::exec::UNTRACKED_DEADLINE;
use common::exec_mock::{ExecMock, MockOptions, Protocol, StreamEnd, wait_group_gone};
use common::remote_bash::{call, job_number, kill_job, remote_bash};
use serde_json::{Value, json};

async fn endpoint(protocol: Protocol) -> (ExecMock, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(
        dir.path(),
        MockOptions {
            protocol,
            heartbeat: Duration::from_millis(200),
            ..MockOptions::default()
        },
    )
    .await;
    (mock, dir)
}

fn runner(mock: &ExecMock) -> Arc<HttpExecRunner> {
    Arc::new(HttpExecRunner::new(&mock.url).unwrap())
}

/// The `timeout_ms` the endpoint was sent for the command containing `needle`.
fn wire_timeout(mock: &ExecMock, needle: &str) -> Option<u64> {
    mock.bodies()
        .iter()
        .find(|b| b.to_string().contains(needle))
        .and_then(|b| b.get("timeout_ms").and_then(Value::as_u64))
}

#[tokio::test]
async fn a_v12_background_start_returns_a_job_whose_stream_is_its_lifetime() {
    let (mock, _dir) = endpoint(Protocol::Streaming).await;
    let (bash, jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    let started = Instant::now();
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo bg-tag listening; sleep 60", "run_in_background": true }),
    )
    .await
    .unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "returned at the settle: {:?}",
        started.elapsed()
    );
    assert!(out.contains("bg-tag listening"), "first output: {out}");
    assert!(out.contains("Running in the background as job 1"), "{out}");
    let s = mock
        .wait_for_stream("bg-tag", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(common::exec_mock::group_alive(s.pgid));
    assert_eq!(
        wire_timeout(&mock, "bg-tag"),
        Some(STREAM_DEADLINE.as_millis() as u64),
        "a stream carries the session-scale deadline"
    );
    assert_eq!(STREAM_DEADLINE.as_millis(), i32::MAX as u128);
    let n = job_number(&out).unwrap();
    let killed = kill_job(&jobs, n).await.unwrap();
    assert!(
        killed.contains(&format!("Job {n} (`")) && killed.contains(") killed after"),
        "{killed}"
    );
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true })
    );
    assert!(wait_group_gone(s.pgid, Duration::from_secs(5)).await);
}

#[tokio::test]
async fn a_v12_background_command_that_dies_at_once_reports_its_exit() {
    let (mock, _dir) = endpoint(Protocol::Streaming).await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    let err = call(
        bash.as_ref(),
        json!({ "command": "echo 'port 8080 in use' >&2; exit 1", "run_in_background": true }),
    )
    .await
    .unwrap_err();
    assert!(err.contains("port 8080 in use"), "{err}");
    assert!(err.contains("Command exited with code 1"), "{err}");
    assert!(bash.jobs().carry().is_none());
}

#[tokio::test]
async fn a_v12_timeout_returns_the_call_and_the_exit_is_noticed() {
    let (mock, _dir) = endpoint(Protocol::Streaming).await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    let started = Instant::now();
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo to-tag; sleep 1.5; echo to-done", "timeout_ms": 500 }),
    )
    .await
    .unwrap();
    assert!(started.elapsed() < Duration::from_millis(1400), "{out}");
    assert!(
        out.contains("Returned after your timeout_ms of 500 ms"),
        "{out}"
    );
    assert!(!out.contains("to-done"), "{out}");
    let deadline = Instant::now() + Duration::from_secs(10);
    let notice = loop {
        if let Some(n) = bash.jobs().take_notices() {
            break n;
        }
        assert!(Instant::now() < deadline, "no notice");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(
        notice.starts_with("[job 1 exited with code 0 after"),
        "{notice}"
    );
    assert!(notice.contains("to-done"), "{notice}");
}

#[tokio::test]
async fn a_v11_endpoint_refuses_run_in_background_before_running_anything() {
    let (mock, _dir) = endpoint(Protocol::V11).await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    let err = call(
        bash.as_ref(),
        json!({ "command": "echo never-ran-tag; sleep 60", "run_in_background": true }),
    )
    .await
    .unwrap_err();
    assert!(
        err.contains("`run_in_background` is unavailable here"),
        "{err}"
    );
    assert!(err.contains("(protocol v1.1)"), "{err}");
    assert!(
        wire_timeout(&mock, "never-ran-tag").is_none(),
        "only the stream probe reached the endpoint: {:?}",
        mock.bodies()
    );
}

#[tokio::test]
async fn a_v11_endpoint_cannot_hand_back_control_and_the_result_says_so() {
    let (mock, _dir) = endpoint(Protocol::V11).await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo v11-tag; sleep 1; echo after", "timeout_ms": 200 }),
    )
    .await
    .unwrap();
    assert!(out.contains("v11-tag") && out.contains("after"), "{out}");
    assert!(job_number(&out).is_none(), "no job: {out}");
    assert!(
        out.ends_with(
            "This sandbox's exec endpoint cannot return before a command exits, so `timeout_ms` \
             had no effect."
        ),
        "{out}"
    );
    assert_eq!(
        wire_timeout(&mock, "v11-tag"),
        Some(UNTRACKED_DEADLINE.as_millis() as u64),
        "v1.1 carries the internal deadline, not the model's timeout_ms"
    );
    // A command that finished inside its timeout_ms needs no such note.
    let quick = call(
        bash.as_ref(),
        json!({ "command": "echo quick", "timeout_ms": 60_000 }),
    )
    .await
    .unwrap();
    assert_eq!(quick, "quick\n");
}
