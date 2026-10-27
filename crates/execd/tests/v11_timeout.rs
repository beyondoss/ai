//! v1.1 enforces `timeout_ms`: at the deadline the whole group is killed and the request answered
//! `200` with `exit_code` 124 and stderr ending in `command exceeded its timeout`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::{Duration, Instant};

use beyond_test_support::ports;
use serde_json::json;
use support::http::post;

async fn start() -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let server = execd::serve(
        ports::tokio_listener().await,
        execd::Options::new(dir.path()),
    )
    .unwrap();
    (server, dir)
}

#[tokio::test]
async fn a_hung_command_is_killed_with_its_group_and_answered_124() {
    let (server, dir) = start().await;
    let marker = dir.path().join("grandchild");
    let script = format!(
        "echo partial; echo oops >&2; (sleep 2; touch '{}') & sleep 60",
        marker.display()
    );
    let t0 = Instant::now();
    let a = post(
        server.local_addr(),
        &[],
        &json!({ "command": "sh", "args": ["-c", script], "timeout_ms": 500 }),
    )
    .await;
    let took = t0.elapsed();
    assert_eq!(a.status, 200);
    assert!(took < Duration::from_secs(3), "{took:?}");
    let body = a.json();
    assert_eq!(body["exit_code"], 124);
    assert_eq!(body["stdout"], "partial\n");
    let stderr = body["stderr"].as_str().unwrap();
    assert!(
        stderr.starts_with("oops\n") && stderr.ends_with("command exceeded its timeout"),
        "{stderr:?}"
    );
    // The backgrounded grandchild died with the group: its marker never appears.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(!marker.exists(), "the grandchild outlived the timeout");
    server.shutdown().await;
}

#[tokio::test]
async fn a_command_that_finishes_in_time_is_unaffected_and_no_timeout_means_no_deadline() {
    let (server, _dir) = start().await;
    let addr = server.local_addr();
    let a = post(
        addr,
        &[],
        &json!({ "command": "sh", "args": ["-c", "echo hi; exit 3"], "timeout_ms": 5000 }),
    )
    .await;
    assert_eq!(
        a.json(),
        json!({ "exit_code": 3, "stdout": "hi\n", "stderr": "" })
    );
    // No `timeout_ms`: it runs as long as it takes.
    let a = post(
        addr,
        &[],
        &json!({ "command": "sh", "args": ["-c", "sleep 1; echo done"] }),
    )
    .await;
    assert_eq!(a.json()["stdout"], "done\n");
    server.shutdown().await;
}
