//! `bash` over a streaming (v1.2) exec endpoint behaves as it does on this host: a silent command
//! yields as a job — heartbeats are not output — progress streams live, and every way a job ends
//! early (`bash_job` kill, a cancelled call, a dropped tool) kills the command's process group on
//! the far side. Over a v1.1 endpoint the call still waits for the exit, and says so.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_core::ToolProgress;
use agent_core::tool::Tool;
use beyond_agent::exec_endpoint::HttpExecRunner;
use common::exec_mock::{ExecMock, MockOptions, Protocol, StreamEnd, wait_group_gone};
use common::remote_bash::{call, job_number, kill_job, remote_bash};
use serde_json::json;

const IDLE: Duration = Duration::from_millis(600);

/// A conforming endpoint that heartbeats far more often than the idle window: if heartbeats
/// counted as output, nothing would ever yield.
async fn chatty_heartbeats() -> (ExecMock, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(
        dir.path(),
        MockOptions {
            heartbeat: Duration::from_millis(50),
            ..MockOptions::default()
        },
    )
    .await;
    (mock, dir)
}

fn runner(mock: &ExecMock) -> Arc<HttpExecRunner> {
    Arc::new(HttpExecRunner::new(&mock.url).unwrap())
}

#[tokio::test]
async fn a_silent_remote_command_yields_as_a_job_and_bash_job_kill_kills_its_group() {
    let (mock, _dir) = chatty_heartbeats().await;
    let (bash, jobs) = remote_bash(runner(&mock), IDLE);
    let started = Instant::now();
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo silent-tag; (sleep 60) & sleep 60" }),
    )
    .await
    .unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "yielded after {:?}",
        started.elapsed()
    );
    assert!(out.contains("silent-tag"), "{out}");
    let n = job_number(&out).unwrap_or_else(|| panic!("no job in: {out}"));
    let s = mock
        .wait_for_stream("silent-tag", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(
        common::exec_mock::group_alive(s.pgid),
        "a yielded job keeps running"
    );

    let killed = kill_job(&jobs, n).await.unwrap();
    assert!(
        killed.contains(&format!("Job {n} (`")) && killed.contains(") killed after"),
        "{killed}"
    );
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true }),
        "the kill closed the stream"
    );
    assert!(wait_group_gone(s.pgid, Duration::from_secs(5)).await);
}

#[tokio::test]
async fn waiting_on_a_remote_job_returns_its_later_output_and_exit() {
    let (mock, _dir) = chatty_heartbeats().await;
    let (bash, jobs) = remote_bash(runner(&mock), IDLE);
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo first; sleep 1.5; echo second; exit 4" }),
    )
    .await
    .unwrap();
    let n = job_number(&out).unwrap_or_else(|| panic!("no job in: {out}"));
    assert!(out.contains("first") && !out.contains("second"), "{out}");
    // Every wait gets one idle window of silence; the one that sees the exit reports it.
    let mut waits = 0;
    let rest = loop {
        match call(&jobs, json!({ "job": n })).await {
            Ok(still) if waits < 10 => {
                assert!(still.contains("is still running"), "{still}");
                assert!(!still.contains("first"), "{still}");
                waits += 1;
            }
            Ok(still) => panic!("never exited: {still}"),
            Err(rest) => break rest,
        }
    };
    assert!(rest.contains("second"), "{rest}");
    assert!(
        !rest.contains("first"),
        "only output since the last report: {rest}"
    );
    assert!(rest.contains("Command exited with code 4"), "{rest}");
}

#[tokio::test]
async fn remote_progress_streams_live() {
    let (mock, _dir) = chatty_heartbeats().await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    let (tx, mut rx) = futures::channel::mpsc::unbounded();
    let progress = ToolProgress::new(
        tx,
        "id".into(),
        "bash".into(),
        agent_core::CancellationToken::new(),
    );
    let out = bash
        .run_streaming(
            // Past the first progress throttle window (100 ms), as for a local command.
            json!({ "command": "sleep 0.3; echo early; sleep 1; echo late" }),
            &progress,
        )
        .await
        .unwrap();
    assert!(out.text.contains("late"));
    drop(progress);
    let mut snapshots = Vec::new();
    while let Ok(u) = rx.try_recv() {
        if let agent_core::tool::ToolUpdate::Progress { snapshot, .. } = u {
            snapshots.push(snapshot);
        }
    }
    assert!(
        snapshots
            .iter()
            .any(|s| s.contains("early") && !s.contains("late")),
        "a live update must show `early` before `late` exists: {snapshots:?}"
    );
}

#[tokio::test]
async fn cancelling_a_remote_call_kills_its_group() {
    let (mock, _dir) = chatty_heartbeats().await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    tokio::select! {
        r = call(bash.as_ref(), json!({ "command": "echo cancel-tag; (sleep 60) & sleep 60" })) => {
            panic!("the call should have been dropped: {r:?}")
        }
        () = async {
            mock.wait_for_stream("cancel-tag", Duration::from_secs(10)).await.unwrap();
            tokio::time::sleep(Duration::from_millis(200)).await;
        } => {}
    }
    let s = mock.streams_running("cancel-tag").pop().unwrap();
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true })
    );
    assert!(wait_group_gone(s.pgid, Duration::from_secs(5)).await);
}

#[tokio::test]
async fn dropping_the_tool_kills_a_parked_remote_job() {
    let (mock, _dir) = chatty_heartbeats().await;
    let (bash, jobs) = remote_bash(runner(&mock), IDLE);
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo parked-tag; sleep 60" }),
    )
    .await
    .unwrap();
    assert!(job_number(&out).is_some(), "{out}");
    let s = mock
        .wait_for_stream("parked-tag", Duration::from_secs(5))
        .await
        .unwrap();
    drop(jobs);
    drop(bash);
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true })
    );
    assert!(wait_group_gone(s.pgid, Duration::from_secs(5)).await);
}

#[tokio::test]
async fn oversized_streamed_output_is_bounded_and_names_no_host_file() {
    let (mock, _dir) = chatty_heartbeats().await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_secs(30));
    let out = call(
        bash.as_ref(),
        json!({ "command": "head -c 3000000 /dev/zero | tr '\\0' 'a' | fold -w 60" }),
    )
    .await
    .unwrap();
    assert!(out.len() < 100 * 1024, "kept {} bytes", out.len());
    assert!(
        out.contains("Full output not saved"),
        "{}",
        &out[out.len().saturating_sub(300)..]
    );
    assert!(
        !out.contains("Full output: "),
        "no host spill file for a remote command"
    );
}

/// The honest fallback: a v1.1 endpoint streams nothing, so its silence says nothing — the call
/// waits for the command to exit rather than yielding a job it could neither watch nor kill. The
/// description says so.
#[tokio::test]
async fn over_a_v11_endpoint_the_call_waits_for_the_exit() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(
        dir.path(),
        MockOptions {
            protocol: Protocol::V11,
            ..MockOptions::default()
        },
    )
    .await;
    let (bash, _jobs) = remote_bash(runner(&mock), Duration::from_millis(200));
    let out = call(
        bash.as_ref(),
        json!({ "command": "echo before; sleep 1.2; echo after" }),
    )
    .await
    .unwrap();
    assert!(
        out.contains("before") && out.contains("after"),
        "no yield: {out}"
    );
    assert!(job_number(&out).is_none(), "{out}");
    assert!(
        bash.description()
            .contains("cannot stream output as it happens"),
        "{}",
        bash.description()
    );
}
