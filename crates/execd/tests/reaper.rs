//! PID 1 duties, on the real binary: adopt and reap orphans, and never steal the exit status of a
//! child it awaits.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::{Duration, Instant};

use serde_json::json;
use support::{Execd, http::post};

/// The pids whose parent is `ppid`.
fn children(ppid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        if rest
            .split_whitespace()
            .nth(1)
            .and_then(|p| p.parse::<u32>().ok())
            == Some(ppid)
        {
            out.push(pid);
        }
    }
    out
}

/// The zombies among `ppid`'s children.
fn zombie_children(ppid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        let mut f = rest.split_whitespace();
        if f.next() == Some("Z") && f.next().and_then(|p| p.parse::<u32>().ok()) == Some(ppid) {
            out.push(pid);
        }
    }
    out
}

#[tokio::test]
async fn orphans_are_reaped_and_awaited_statuses_survive_the_reaper() {
    let execd = Execd::start(&[], &[]);
    let addr = std::net::SocketAddr::V4(execd.addr);
    // An orphan is adopted by execd rather than by the host's init.
    let req =
        json!({ "command": "sh", "args": ["-c", "setsid sleep 1 & exit 0"], "timeout_ms": 30_000 });
    post(addr, &[], &req).await;
    let deadline = Instant::now() + Duration::from_secs(5);
    while children(execd.pid()).is_empty() {
        assert!(Instant::now() < deadline, "execd adopted nothing");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Each command leaves a session leader behind that outlives it (so it is orphaned and adopted
    // by execd, the subreaper) and exits with a code of its own. A reaper that waited on any child
    // would sometimes take a command's status first and the answer would be wrong.
    let mut calls = Vec::new();
    for n in 1..=24u32 {
        calls.push(tokio::spawn(async move {
            let script = format!("setsid sh -c 'sleep 0.3' & exit {n}");
            let req = json!({ "command": "sh", "args": ["-c", script], "timeout_ms": 30_000 });
            let a = post(addr, &[], &req).await;
            (n, a.json()["exit_code"].as_i64())
        }));
    }
    for c in calls {
        let (n, code) = c.await.unwrap();
        assert_eq!(code, Some(i64::from(n)), "command {n}");
    }
    // Once the orphans have exited, none is left under execd, not even as a zombie.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let left = children(execd.pid());
        if left.is_empty() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "unreaped orphans: {left:?} (zombies: {:?})",
            zombie_children(execd.pid())
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
