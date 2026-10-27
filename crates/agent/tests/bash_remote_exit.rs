//! Remote `bash` jobs and the way out of the process. `std::process::exit` runs no destructors, so
//! the exit path kills every live job explicitly (`kill_live_jobs`, then a bounded wait) — and a job
//! on a streaming exec endpoint must be reached by it exactly as a local process group is: its
//! stream closed, the endpoint killing the group.
//!
//! Needs its own process: `kill_live_jobs` kills every job in the process, which would take down any
//! other suite's jobs running beside it. nextest runs each test in its own process; plain `cargo test`
//! would put every suite in one.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use beyond_agent::exec_endpoint::HttpExecRunner;
use common::exec_mock::{ExecMock, MockOptions, StreamEnd, wait_group_gone};
use common::remote_bash::{call, job_number, remote_bash};
use common::{SpawnGuarded, run_cmd, spawn_model_server, turn_tool_use};
use serde_json::json;

#[tokio::test]
async fn the_exit_path_reaches_a_parked_remote_job() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start(dir.path()).await;
    let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
    let (bash, _jobs) = remote_bash(runner, Duration::from_millis(500));
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo exit-tag; (sleep 60) & sleep 60" }),
    )
    .await
    .unwrap();
    assert!(job_number(&out).is_some(), "{out}");
    let s = mock
        .wait_for_stream("exit-tag", Duration::from_secs(5))
        .await
        .unwrap();

    // Exactly what `main.rs`'s `finish_process_cleanup` does before `process::exit`.
    tokio::task::spawn_blocking(|| {
        beyond_agent::tools::bash::kill_live_jobs();
        beyond_agent::tools::exec::wait_for_pending_group_kills(Duration::from_secs(5));
    })
    .await
    .unwrap();
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true })
    );
    assert!(wait_group_gone(s.pgid, Duration::from_secs(5)).await);
}

/// The real binary, a real signal: `run --exec-url` mid-`bash`, SIGTERM'd. The command — and the
/// grandchild it backgrounded — must die on the endpoint, not run on after the agent is gone.
#[test]
fn sigterm_to_run_kills_the_remote_command_group() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    // A streaming sandbox: this is about the stream a command lives in (a sandbox kills an
    // unkept job the same way when its stream closes, which `sandbox_jobs.rs` covers).
    let mock = rt.block_on(ExecMock::start_with(
        dir.path(),
        MockOptions {
            ..MockOptions::default()
        },
    ));

    let turn = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": "echo sigterm-tag; (sleep 60) & sleep 60" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn]);
    let mut child = run_cmd(common::BIN)
        .args([
            "run",
            "run it remotely",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
            "--exec-url",
            &mock.url,
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded();

    let s = rt
        .block_on(mock.wait_for_stream("sigterm-tag", Duration::from_secs(30)))
        .expect("the bash command reached the endpoint");
    std::process::Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st;
        }
        assert!(Instant::now() < deadline, "run did not exit after SIGTERM");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(143));
    assert_eq!(
        rt.block_on(mock.wait_for_end(s.pgid, Duration::from_secs(5))),
        Some(StreamEnd::Disconnected { killed: true })
    );
    assert!(rt.block_on(wait_group_gone(s.pgid, Duration::from_secs(5))));
}
