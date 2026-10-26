//! A test process that dies without running its destructors (`SIGKILL`: a CI timeout, an OOM kill,
//! `nextest` terminating a slow test) must not leave an `execd`, or the commands and jobs under it,
//! running. The victim is this very test binary re-run on one `#[ignore]`d test; the parent kills it
//! mid-run and looks for survivors.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;
use support::{Execd, http::post};

fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| {
        s.rsplit_once(')')
            .and_then(|(_, r)| r.trim_start().chars().next())
            .is_some_and(|st| st != 'Z' && st != 'X')
    })
}

fn pids_in(file: &Path) -> Vec<u32> {
    std::fs::read_to_string(file)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect()
}

/// Not a test of its own: the victim. Starts an execd (as root if `GUARD_ROOT` is set) with a job
/// running, writes the pids of execd, the job's shell and its `sleep` to `GUARD_PIDS`, and waits to
/// be killed.
#[tokio::test]
#[ignore = "run only as the victim of a_killed_test_leaves_no_execd"]
async fn victim() {
    let file = std::env::var("GUARD_PIDS").expect("GUARD_PIDS");
    let execd = if std::env::var_os("GUARD_ROOT").is_some() {
        Execd::start_as_root(&[], &[]).expect("root")
    } else {
        Execd::start(&[], &[])
    };
    let script = format!("echo $$ >> {file}; sleep 600 & echo $! >> {file}; wait");
    let started = post(
        std::net::SocketAddr::V4(execd.addr),
        &[],
        &json!({ "command": "sh", "args": ["-c", script],
                 "job": { "op": "start", "scope": "s", "id": "j", "hold": false } }),
    )
    .await;
    assert_eq!(started.status, 200);
    // Its own file: the job's (root's, under `--run-as`-less root execd) cannot be rewritten by us.
    std::fs::write(format!("{file}.execd"), execd.pid().to_string()).unwrap();
    std::thread::sleep(Duration::from_secs(120));
}

fn run(root: bool) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("pids");
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--ignored", "--exact", "victim", "--nocapture"])
        .env("GUARD_PIDS", &file)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if root {
        cmd.env("GUARD_ROOT", "1");
    }
    let mut victim = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let all = || {
        let mut v = pids_in(&file);
        v.extend(pids_in(&dir.path().join("pids.execd")));
        v
    };
    while all().len() < 3 {
        assert!(
            Instant::now() < deadline,
            "the victim never started its job"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let pids = all();
    assert!(pids.iter().all(|&p| alive(p)), "{pids:?} should be running");
    // The test process dies without a chance to clean up.
    victim.kill().unwrap();
    victim.wait().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while pids.iter().any(|&p| alive(p)) {
        let left: Vec<u32> = pids.iter().copied().filter(|&p| alive(p)).collect();
        assert!(
            Instant::now() < deadline,
            "{left:?} (execd, the job's shell, its sleep) survived the test process"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_killed_test_leaves_no_execd() {
    run(false);
}

#[test]
fn a_killed_test_leaves_no_root_execd() {
    if !Execd::can_be_root() {
        eprintln!("SKIPPED: needs root (or passwordless sudo)");
        return;
    }
    run(true);
}
