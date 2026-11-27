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

// ---------------------------------------------------------------------------------------------
// The end of a job, decided by execd's own child list where it can be.
//
// Every process of a job this server spawned is this process's descendant (execd is the subreaper),
// so a child list the kernel reports empty means nothing of the job is left. The tests below are the
// ones that must fail when that reasoning is applied where it does not hold, or when the scan behind
// it stops looking: a daemon that double-forked out of the group, a grandchild that outlived its
// parent, a job under `EXECD_RUN_AS`, and a marked process that is not execd's descendant at all.

/// execd's log, for the runs that set `EXECD_LOG`.
fn log(e: &Execd) -> String {
    std::fs::read_to_string(e.dir.path().join("execd.log")).unwrap_or_default()
}

/// The job's `state` as `read` answers it.
async fn state(e: &Execd, id: &str) -> String {
    let r = job(
        e,
        json!({ "op": "read", "id": id, "offset": 0, "max": 1 }),
        None,
    )
    .await;
    r["state"].as_str().unwrap_or("gone").to_string()
}

/// Wait for `want`, asserting the job did not get there first when `want` is "not ended".
async fn until_state(e: &Execd, id: &str, want: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let now = state(e, id).await;
        if now == want {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "job {id} is {now:?}, not {want:?}: {}",
            log(e)
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Wake every waiting supervisor the cheap way: the reaper tells all of them when it collects a
/// child, so one short-lived job makes a decision that would otherwise wait for the 5 s tick.
async fn nudge(e: &Execd, id: &str) {
    job(
        e,
        json!({ "op": "start", "id": id, "label": "nudge" }),
        Some("true"),
    )
    .await;
    until_state(e, id, "ended").await;
}

/// A job whose work left the group keeps running, however long it has been: three wakes and the
/// job is still running while its daemon is. Then the daemon dies, the next wake ends the job.
async fn a_stray_holds_the_job_open(e: &Execd, id: &str, stray: &str, first: &str) {
    until("the stray", || !running(stray).is_empty()).await;
    for i in 0..3 {
        nudge(e, &format!("{first}{i}")).await;
        let now = state(e, id).await;
        assert_ne!(
            now,
            "ended",
            "a job read as ended while {stray:?} ran (it is {now:?}): {}",
            log(e)
        );
        assert!(!running(stray).is_empty(), "{stray:?} was killed");
    }
}

/// The other half: once the stray is gone the job ends, so nothing was left holding it open.
async fn the_job_ends_once_the_stray_dies(e: &Execd, id: &str, stray: &str, first: &str) {
    for pid in running(stray) {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status();
    }
    nudge(e, &format!("{first}z")).await;
    until_state(e, id, "ended").await;
}

/// A daemon that double-forked out of the job's group *and* out of its parent chain: the shell exits
/// at once, the intermediate process exits, and the daemon is adopted by execd. The job has not
/// ended while it runs, and it ends when the daemon is killed.
#[tokio::test]
async fn a_job_is_not_ended_while_its_double_forked_daemon_runs() {
    let _reap = Reap(vec!["sleep 81"]);
    let e = Execd::start(&[], &[]);
    job(
        &e,
        json!({ "op": "start", "id": "daemon", "hold": false, "label": "x" }),
        Some("( setsid sh -c 'sleep 81 &' & ) ; true"),
    )
    .await;
    a_stray_holds_the_job_open(&e, "daemon", "sleep 81", "nudge-d").await;
    the_job_ends_once_the_stray_dies(&e, "daemon", "sleep 81", "nudge-d").await;
}

/// A grandchild that outlives its parent: the shell starts a session leader that starts the work and
/// exits, so the only process left is two forks below the job's shell. It keeps the job alive.
#[tokio::test]
async fn a_job_is_not_ended_while_a_grandchild_outlives_its_parent() {
    let _reap = Reap(vec!["sleep 82"]);
    let e = Execd::start(&[], &[]);
    job(
        &e,
        json!({ "op": "start", "id": "grand", "hold": false, "label": "x" }),
        Some("sh -c 'setsid sleep 82 &' ; true"),
    )
    .await;
    a_stray_holds_the_job_open(&e, "grand", "sleep 82", "nudge-g").await;
    the_job_ends_once_the_stray_dies(&e, "grand", "sleep 82", "nudge-g").await;
}

/// The fast path fires when nothing is left, and stands aside when execd has a child of its own: the
/// first end is decided by the child list, the second (with a long-lived job running beside it) by
/// the scan. Both end; the log says which decided.
#[tokio::test]
async fn a_quiet_end_is_decided_by_the_child_list_and_a_busy_one_by_the_scan() {
    let e = Execd::start(&[("EXECD_LOG", "debug")], &[]);
    let answered = |e: &Execd| log(e).matches("execd has no child at all").count();
    job(
        &e,
        json!({ "op": "start", "id": "quiet", "label": "x" }),
        Some("true"),
    )
    .await;
    until_state(&e, "quiet", "ended").await;
    assert_eq!(
        answered(&e),
        1,
        "the quiet end was not answered by the child list"
    );
    // A job of its own that outlives the next one: execd now has a child, so the end of a later job
    // cannot be answered from the child list, and the scan decides it as it always did.
    job(
        &e,
        json!({ "op": "start", "id": "busy", "label": "x" }),
        Some("sleep 85"),
    )
    .await;
    until("the long job", || !running("sleep 85").is_empty()).await;
    job(
        &e,
        json!({ "op": "start", "id": "beside", "label": "x" }),
        Some("true"),
    )
    .await;
    until_state(&e, "beside", "ended").await;
    assert_eq!(
        answered(&e),
        1,
        "the child list answered while execd had a child of its own: {}",
        log(&e)
    );
    job(&e, json!({ "op": "wipe", "grace_ms": 200 }), None).await;
}

/// A client can read a job's mark off the wire and put it on a process of its own, which is not a
/// descendant of execd. That is the one case the child list cannot see, and the documented
/// difference: the job ends as soon as its group is gone. The scan alone — `mut_end_scan_always`,
/// what execd did before this existed — holds it open while that process runs.
#[tokio::test]
async fn a_marked_process_that_is_no_descendant_does_not_hold_the_job_open() {
    let e = Execd::start(&[], &[]);
    let started = job(
        &e,
        json!({ "op": "start", "id": "m", "label": "x" }),
        Some("sleep 0.4"),
    )
    .await;
    let mark = started["mark"].as_str().unwrap().to_string();
    // The test's own process, carrying the job's mark, outside execd's tree.
    let mut forged = std::process::Command::new("env")
        .arg(format!("EXECD_JOB_MARK={mark}"))
        .args(["sleep", "83"])
        .spawn()
        .unwrap();
    if cfg!(feature = "mut_end_scan_always") {
        // The scan is the answer here, and it finds the mark: the job waits for it.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_ne!(
            state(&e, "m").await,
            "ended",
            "the scan did not find a marked process: {}",
            log(&e)
        );
    } else {
        // The child list is the answer, and it cannot see this process: the job ends.
        until_state(&e, "m", "ended").await;
    }
    let _ = forged.kill();
    let _ = forged.wait();
    // And once the marked process is gone, the job ends either way.
    until_state(&e, "m", "ended").await;
}

/// The same property under `EXECD_RUN_AS`, where the job's processes are another uid's: execd (root)
/// reads their environment, the scan finds a stray, and the job does not end while it runs. The
/// child list answers it just as well (the helper execd spawned is the job's process), which is what
/// this pins: another uid changes nothing about the end.
#[tokio::test]
async fn a_job_is_not_ended_while_its_work_runs_as_another_uid() {
    let _reap = Reap(vec!["sleep 84"]);
    let Some(e) = Execd::start_as_root(&[("EXECD_RUN_AS", "nobody")], &[]) else {
        println!("SKIPPED: this host does not let the tests run execd as root");
        return;
    };
    job(
        &e,
        json!({ "op": "start", "id": "uid", "hold": false, "label": "x" }),
        Some("setsid sleep 84 &"),
    )
    .await;
    a_stray_holds_the_job_open(&e, "uid", "sleep 84", "nudge-u").await;
    // The stray is another uid's, so the test's own kill cannot reach it: execd runs as root.
    for pid in running("sleep 84") {
        let _ = std::process::Command::new("sudo")
            .args(["-n", "kill", "-KILL", &pid.to_string()])
            .status();
    }
    nudge(&e, "nudge-uz").await;
    until_state(&e, "uid", "ended").await;
}
