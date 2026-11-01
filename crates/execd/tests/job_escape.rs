//! A job's processes are the job's wherever they have gone. `timeout` and `setsid` put themselves
//! and their child in a process group (and session) of their own, and a daemon double-forks away;
//! a kill that signalled the job's group alone left them running while `survivors` was empty, and a
//! job whose real work had left the group read as ended. Membership is the group plus everything
//! that carries the job's mark in its environment (`EXECD_JOB_MARK`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::Execd;
use support::http::post;

const SCOPE: &str = "escape";

/// Live (non-zombie) processes whose command line is exactly `cmd`.
fn running(cmd: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let line: Vec<String> = std::fs::read(format!("/proc/{pid}/cmdline"))
            .unwrap_or_default()
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
        let zombie = stat
            .rsplit_once(')')
            .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'));
        if line.join(" ") == cmd && !zombie {
            out.push(pid);
        }
    }
    out
}

/// Kills what a failed assertion would otherwise leave running.
struct Reap(Vec<&'static str>);

impl Drop for Reap {
    fn drop(&mut self) {
        for cmd in &self.0 {
            for pid in running(cmd) {
                let _ = std::process::Command::new("kill")
                    .args(["-KILL", &pid.to_string()])
                    .status();
            }
        }
    }
}

async fn until(what: &str, mut f: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !f() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn addr(e: &Execd) -> SocketAddr {
    SocketAddr::V4(e.addr)
}

async fn job(e: &Execd, op: Value, script: Option<&str>) -> Value {
    let mut op = op;
    op["scope"] = json!(SCOPE);
    let mut body = json!({ "command": "true", "job": op });
    if let Some(s) = script {
        body["command"] = json!("sh");
        body["args"] = json!(["-c", s]);
    }
    let a = post(addr(e), &[], &body).await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    a.json()
}

#[tokio::test]
async fn a_wipe_kills_what_timeout_and_setsid_took_out_of_the_group() {
    let _reap = Reap(vec!["sleep 71", "sleep 72", "sleep 73"]);
    let e = Execd::start(&[], &[]);
    // The shell stays; its children leave its group (timeout, setsid) or its session and parent
    // (a subshell that double-forks).
    let script = "timeout 60 sleep 71 & setsid sleep 72 & (setsid sh -c 'sleep 73 &' &) ; wait";
    job(
        &e,
        json!({ "op": "start", "id": "a", "hold": false, "label": "x" }),
        Some(script),
    )
    .await;
    until("the three sleeps", || {
        ["sleep 71", "sleep 72", "sleep 73"]
            .iter()
            .all(|c| !running(c).is_empty())
    })
    .await;

    let r = job(&e, json!({ "op": "wipe", "grace_ms": 200 }), None).await;
    let alive: Vec<_> = ["sleep 71", "sleep 72", "sleep 73"]
        .iter()
        .filter(|c| !running(c).is_empty())
        .collect();
    assert!(
        alive.is_empty(),
        "a wipe left {alive:?} running; it answered {r}"
    );
    assert_eq!(r["survivors"], json!([]));
}

#[tokio::test]
async fn a_job_is_not_ended_while_its_work_runs_outside_the_group() {
    let _reap = Reap(vec!["sleep 74"]);
    let e = Execd::start(&[], &[]);
    // The shell exits at once; its only child is in a group of its own.
    job(
        &e,
        json!({ "op": "start", "id": "b", "hold": false, "label": "x" }),
        Some("timeout 60 sleep 74 &"),
    )
    .await;
    until("the sleep", || !running("sleep 74").is_empty()).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    let r = job(
        &e,
        json!({ "op": "read", "id": "b", "offset": 0, "max": 1024 }),
        None,
    )
    .await;
    assert_ne!(r["state"], "ended", "the job's work is still running: {r}");

    // And a wipe names it as a survivor if it cannot kill it, or kills it.
    let w = job(&e, json!({ "op": "wipe", "grace_ms": 200 }), None).await;
    assert!(running("sleep 74").is_empty(), "wipe left it running: {w}");
}
