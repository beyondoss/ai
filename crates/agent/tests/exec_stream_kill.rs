//! Exec protocol v1.2: the connection is the command's lifetime. Hanging up mid-stream — by
//! dropping the call, or through the [`Kill`] a tracked run hands out — must leave the endpoint
//! killing the command's whole process group, a backgrounded grandchild included.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use beyond_ai_agent::exec_endpoint::HttpExecRunner;
use beyond_ai_agent::tools::exec::{CommandRunner, Kill};
use common::exec_mock::{ExecMock, StreamEnd, wait_group_gone};

fn sh(script: &str) -> Vec<String> {
    vec!["-c".to_string(), script.to_string()]
}

/// A command that announces itself, leaves a grandchild in its group, and runs for a long time.
fn long_command(tag: &str) -> Vec<String> {
    sh(&format!("(sleep 60) & echo {tag}; sleep 60"))
}

#[tokio::test]
async fn dropping_the_call_mid_stream_kills_the_process_group() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start(dir.path(), true).await;
    let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
    let (seen_tx, mut seen_rx) = tokio::sync::mpsc::unbounded_channel();
    let task = {
        let runner = runner.clone();
        tokio::spawn(async move {
            let sink = move |_: &[u8]| {
                let _ = seen_tx.send(());
            };
            runner
                .run_streaming(
                    "sh",
                    &long_command("drop-me"),
                    None,
                    Duration::from_secs(120),
                    &sink,
                )
                .await
        })
    };
    tokio::time::timeout(Duration::from_secs(10), seen_rx.recv())
        .await
        .expect("the command's first output streamed");
    let s = mock
        .wait_for_stream("drop-me", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(common::exec_mock::group_alive(s.pgid));

    task.abort();
    let _ = task.await;
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true })
    );
    assert!(
        wait_group_gone(s.pgid, Duration::from_secs(5)).await,
        "the grandchild outlived the connection"
    );
}

#[tokio::test]
async fn a_tracked_runs_kill_closes_the_stream_and_returns_once_it_has() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start(dir.path(), true).await;
    let runner = Arc::new(HttpExecRunner::new(&mock.url).unwrap());
    let kill: Arc<Mutex<Option<Kill>>> = Arc::default();
    let task = {
        let (runner, kill) = (runner.clone(), kill.clone());
        tokio::spawn(async move {
            let on_spawn = move |k: Kill| *kill.lock().unwrap() = Some(k);
            runner
                .run_streaming_tracked(
                    "sh",
                    &long_command("kill-me"),
                    None,
                    Duration::from_secs(120),
                    &|_: &[u8]| {},
                    &on_spawn,
                )
                .await
        })
    };
    let s = mock
        .wait_for_stream("kill-me", Duration::from_secs(10))
        .await
        .unwrap();
    let k = kill
        .lock()
        .unwrap()
        .clone()
        .expect("a streaming run reports how to kill it — the promise that its output is live");
    assert!(matches!(k, Kill::Hook(_)), "no local group to signal");

    tokio::task::spawn_blocking(move || k.kill()).await.unwrap();
    // `kill` returned only after the runner let go of the stream, so its result is already in.
    let res = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("the runner ended when its stream was closed")
        .unwrap();
    assert!(res.is_err(), "a killed stream has no exit status: {res:?}");
    assert_eq!(
        mock.wait_for_end(s.pgid, Duration::from_secs(5)).await,
        Some(StreamEnd::Disconnected { killed: true })
    );
    assert!(wait_group_gone(s.pgid, Duration::from_secs(5)).await);
}

/// A v1.1 endpoint has nothing to kill and no live output to promise, so a tracked run reports no
/// `Kill` — which is what tells `bash` to wait for the exit instead of yielding.
#[tokio::test]
async fn a_v11_tracked_run_reports_no_kill() {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(
        dir.path(),
        common::exec_mock::MockOptions {
            protocol: common::exec_mock::Protocol::V11,
            ..Default::default()
        },
    )
    .await;
    let runner = HttpExecRunner::new(&mock.url).unwrap();
    let reported = Arc::new(Mutex::new(false));
    let r = reported.clone();
    let res = runner
        .run_streaming_tracked(
            "sh",
            &sh("printf hi"),
            None,
            Duration::from_secs(10),
            &|_: &[u8]| {},
            &move |_: Kill| *r.lock().unwrap() = true,
        )
        .await
        .unwrap();
    assert_eq!(res.stdout, "hi");
    assert!(!*reported.lock().unwrap());
}
