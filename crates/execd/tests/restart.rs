//! The job table, the process groups and the exit status survive a restart of execd (R1-R3 of
//! `job_kernel::recover`, driven through the real binary): a crashed execd's jobs are in the next
//! one's table, a live group adopted and killable, an ended job's status exactly what it ended with.
//! Each test is red under the mutation of the kernel the comment names (`--features <mut>`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::Execd;
use support::http::post;

async fn job(addr: SocketAddr, op: Value) -> Value {
    let a = post(addr, &[], &json!({ "command": "true", "job": op })).await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    a.json()
}

async fn start(addr: SocketAddr, id: &str, script: &str) -> i64 {
    let body = json!({
        "command": "sh", "args": ["-c", script],
        "job": { "op": "start", "scope": "s", "id": id, "hold": false },
    });
    let a = post(addr, &[], &body).await;
    assert_eq!(a.status, 200);
    a.json()["started"].as_i64().unwrap()
}

async fn state_of(addr: SocketAddr, id: &str) -> Option<(String, Value)> {
    let l = job(addr, json!({"op":"list","scope":"s"})).await;
    l["jobs"].as_array().and_then(|a| {
        a.iter()
            .find(|j| j["id"] == id)
            .map(|j| (j["state"].as_str().unwrap().to_string(), j["code"].clone()))
    })
}

async fn until(addr: SocketAddr, id: &str, want: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some((s, code)) = state_of(addr, id).await
            && s == want
        {
            return code;
        }
        assert!(Instant::now() < deadline, "{id} never became {want}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn crash(execd: &mut Execd) {
    execd.child.kill().unwrap();
    execd.child.wait().unwrap();
}

#[tokio::test]
async fn a_killed_execds_jobs_are_in_the_next_ones_table_with_their_status() {
    let state = tempfile::tempdir().unwrap();
    let s = state.path().join("ledger");
    let settings = [("EXECD_STATE_DIR", s.to_str().unwrap())];
    let mut first = Execd::start(&settings, &[]);
    let a1 = SocketAddr::V4(first.addr);
    // A kept job running on, one that ended with code 7, and one whose shell exited 3 with a
    // child still running (the group outlives the shell).
    start(a1, "live", "exec sleep 300").await;
    start(a1, "ended", "exit 7").await;
    start(a1, "orphan", "sleep 300 & exit 3").await;
    assert_eq!(until(a1, "ended", "ended").await, json!(7));
    assert_eq!(until(a1, "orphan", "exited").await, json!(3));
    crash(&mut first);

    let second = Execd::start(&settings, &[]);
    let a2 = SocketAddr::V4(second.addr);
    // R2: the live groups are adopted, with what was saved of the shell.
    assert_eq!(state_of(a2, "live").await.unwrap().0, "running");
    assert_eq!(
        state_of(a2, "orphan").await.unwrap(),
        ("exited".to_string(), json!(3))
    );
    // R1: the ended job's status is what it ended with.
    assert_eq!(
        state_of(a2, "ended").await.unwrap(),
        ("ended".to_string(), json!(7))
    );
    // The adopted group is ours to signal: it ends when killed, the shell's status is kept (R3),
    // and a group whose shell status no one saw ends with the lost one.
    job(
        a2,
        json!({"op":"kill","scope":"s","ids":["live","orphan"],"grace_ms":200}),
    )
    .await;
    assert_eq!(until(a2, "orphan", "ended").await, json!(3));
    assert_eq!(until(a2, "live", "ended").await, json!(-1));
}

#[tokio::test]
async fn a_group_that_ended_while_execd_was_down_ends_with_the_status_saved_or_the_lost_one() {
    let state = tempfile::tempdir().unwrap();
    let s = state.path().join("ledger");
    let settings = [("EXECD_STATE_DIR", s.to_str().unwrap())];
    let mut first = Execd::start(&settings, &[]);
    let a1 = SocketAddr::V4(first.addr);
    let p1 = start(a1, "short", "exec sleep 300").await;
    let p2 = start(a1, "shell-gone", "sleep 300 & exit 5").await;
    assert_eq!(until(a1, "shell-gone", "exited").await, json!(5));
    // Find and kill both groups while no execd runs.
    crash(&mut first);
    for pgid in [p1, p2] {
        let status = std::process::Command::new("sh")
            .args(["-c", &format!("kill -s KILL -- -{pgid}")])
            .status()
            .unwrap();
        assert!(status.success(), "could not kill group {pgid}: {status:?}");
    }
    std::thread::sleep(Duration::from_millis(300));
    let second = Execd::start(&settings, &[]);
    let a2 = SocketAddr::V4(second.addr);
    // R3: a gone group is ended at once: the shell's saved status, else the lost one. Never running.
    assert_eq!(
        state_of(a2, "shell-gone").await.unwrap(),
        ("ended".to_string(), json!(5))
    );
    assert_eq!(
        state_of(a2, "short").await.unwrap(),
        ("ended".to_string(), json!(-1))
    );
}
