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

async fn read_all(addr: SocketAddr, id: &str) -> (Vec<u8>, Vec<(u64, u64)>) {
    use base64::Engine as _;
    let v = job(
        addr,
        json!({"op":"read","scope":"s","id":id,"offset":0,"max":1 << 20}),
    )
    .await;
    let data = base64::engine::general_purpose::STANDARD
        .decode(v["data"].as_str().unwrap_or(""))
        .unwrap();
    let ranges = v["stderr_ranges"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|r| (r[0].as_u64().unwrap(), r[1].as_u64().unwrap()))
        .collect();
    (data, ranges)
}

async fn read_until(addr: SocketAddr, id: &str, want: &str) -> (Vec<u8>, Vec<(u64, u64)>) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (data, ranges) = read_all(addr, id).await;
        if String::from_utf8_lossy(&data).contains(want) {
            return (data, ranges);
        }
        assert!(
            Instant::now() < deadline,
            "{id} never wrote {want:?}: {:?}",
            String::from_utf8_lossy(&data)
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// S1: a job's output is a file it inherits, not a pipe execd reads. Red under `mut_job_output_pipe`
/// (the job's writes after the restart are `SIGPIPE`: it ends, and nothing is read).
#[tokio::test]
async fn a_job_writes_through_a_restart_of_execd_and_the_new_one_reads_it() {
    let state = tempfile::tempdir().unwrap();
    let s = state.path().join("ledger");
    let flag = state.path().join("go");
    let settings = [("EXECD_STATE_DIR", s.to_str().unwrap())];
    let mut first = Execd::start(&settings, &[]);
    let a1 = SocketAddr::V4(first.addr);
    // Writes before the restart, waits for the flag, writes both streams after it, and stays.
    let script = format!(
        "echo before; while [ ! -e {} ]; do sleep 0.05; done; echo after; echo errafter >&2; exec sleep 300",
        flag.display()
    );
    start(a1, "w", &script).await;
    read_until(a1, "w", "before").await;
    crash(&mut first);

    // While no execd runs, the job writes: into its spool, with no reader.
    std::fs::write(&flag, b"").unwrap();
    std::thread::sleep(Duration::from_millis(400));
    let second = Execd::start(&settings, &[]);
    let a2 = SocketAddr::V4(second.addr);
    // The job lives (a write to a dead pipe would have killed its shell with SIGPIPE), and what it
    // wrote while execd was down, on each stream, is read back, labelled.
    assert_eq!(state_of(a2, "w").await.unwrap().0, "running");
    let (data, ranges) = read_until(a2, "w", "errafter").await;
    let text = String::from_utf8_lossy(&data).to_string();
    assert!(text.contains("after\n"), "{text:?}");
    let at = u64::try_from(text.find("errafter").unwrap()).unwrap();
    assert_eq!(ranges, vec![(at, "errafter\n".len() as u64)], "{text:?}");
    // What it writes now, with this execd, arrives the same way.
    assert_eq!(state_of(a2, "w").await.unwrap().0, "running");
    job(
        a2,
        json!({"op":"kill","scope":"s","ids":["w"],"grace_ms":200}),
    )
    .await;
    until(a2, "w", "ended").await;
}

/// S3: a `SIGHUP` to execd is a restart, not a kill (service-init sends it for a restart of execd): with a
/// ledger every job's group runs on and the next execd adopts it. Red under `mut_shutdown_signals_jobs`.
#[tokio::test]
async fn a_sighup_to_execd_leaves_its_jobs_running() {
    let state = tempfile::tempdir().unwrap();
    let s = state.path().join("ledger");
    let settings = [("EXECD_STATE_DIR", s.to_str().unwrap())];
    let mut first = Execd::start(&settings, &[]);
    let a1 = SocketAddr::V4(first.addr);
    let pgid = start(a1, "live", "exec sleep 300").await;
    start(a1, "done", "exit 4").await;
    until(a1, "done", "ended").await;
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(i32::try_from(first.child.id()).unwrap()),
        nix::sys::signal::Signal::SIGHUP,
    )
    .unwrap();
    first.child.wait().unwrap();
    // The group is there: signal 0 to it.
    let group = nix::unistd::Pid::from_raw(i32::try_from(pgid).unwrap());
    assert!(
        nix::sys::signal::killpg(group, None).is_ok(),
        "execd's shutdown killed the job's group"
    );

    // S4: its record is kept, so the next execd adopts it, and the ended one as it ended.
    let second = Execd::start(&settings, &[]);
    let a2 = SocketAddr::V4(second.addr);
    assert_eq!(state_of(a2, "live").await.unwrap().0, "running");
    assert_eq!(
        state_of(a2, "done").await.unwrap(),
        ("ended".to_string(), json!(4))
    );
    job(
        a2,
        json!({"op":"kill","scope":"s","ids":["live"],"grace_ms":200}),
    )
    .await;
    until(a2, "live", "ended").await;
}

/// S6: a `SIGTERM` to execd is a stop (service-init sends it for a sandbox stop), so with a ledger too
/// every job's group is ended, and the next execd adopts nothing. Red under `mut_term_keeps_jobs`.
#[tokio::test]
async fn a_sigterm_to_execd_ends_every_job_even_with_a_ledger() {
    let state = tempfile::tempdir().unwrap();
    let s = state.path().join("ledger");
    let settings = [("EXECD_STATE_DIR", s.to_str().unwrap())];
    let mut first = Execd::start(&settings, &[]);
    let a1 = SocketAddr::V4(first.addr);
    let pgid = start(a1, "live", "exec sleep 300").await;
    start(a1, "done", "exit 4").await;
    until(a1, "done", "ended").await;
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(i32::try_from(first.child.id()).unwrap()),
        nix::sys::signal::Signal::SIGTERM,
    )
    .unwrap();
    first.child.wait().unwrap();
    let group = nix::unistd::Pid::from_raw(i32::try_from(pgid).unwrap());
    let deadline = Instant::now() + Duration::from_secs(5);
    while nix::sys::signal::killpg(group, None).is_ok() {
        assert!(
            Instant::now() < deadline,
            "execd's stop left the job's group running"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Nothing is in the ledger for the next execd to adopt.
    let second = Execd::start(&settings, &[]);
    let a2 = SocketAddr::V4(second.addr);
    assert!(
        state_of(a2, "live").await.is_none(),
        "a stopped job was adopted"
    );
}

/// X1, X2: a client that resumes through `read` at its offset sees no gap across a crash of execd:
/// what the new execd answers from offset 0 begins with exactly what the old one answered, bytes and
/// stderr labels, and goes on. The job writes both streams in step, past a chunk, so the ring's
/// files and the journal's unfinished chunk are both rebuilt. Red under `mut_resume_cursor_lags`
/// (journalled bytes are read from the spool again: repeated) and `mut_resume_end_drops_journal`
/// (the ring ends before them: a gap).
#[tokio::test]
async fn a_client_resumes_at_its_offset_across_a_crash_with_no_gap() {
    let state = tempfile::tempdir().unwrap();
    let s = state.path().join("ledger");
    let settings = [("EXECD_STATE_DIR", s.to_str().unwrap())];
    let mut first = Execd::start(&settings, &[]);
    let a1 = SocketAddr::V4(first.addr);
    let script = "i=0; while :; do printf 'o%0900d\\n' $i; printf 'e%0900d\\n' $i >&2; i=$((i+1)); sleep 0.005; done";
    start(a1, "w", script).await;
    // Past a chunk (64 KiB), with a tail of the next one unfinished.
    let before = loop {
        let (d, r) = read_all(a1, "w").await;
        if d.len() > 150_000 {
            break (d, r);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    crash(&mut first);
    // Written while no execd runs.
    std::thread::sleep(Duration::from_millis(300));
    let second = Execd::start(&settings, &[]);
    let a2 = SocketAddr::V4(second.addr);
    let (after, ranges) = read_all(a2, "w").await;
    assert!(
        after.len() > before.0.len() && after.starts_with(&before.0),
        "the new execd's answer does not begin with the old one's ({} then {} bytes)",
        before.0.len(),
        after.len()
    );
    let n = before.0.len() as u64;
    let clip = |rs: &[(u64, u64)]| -> Vec<(u64, u64)> {
        rs.iter()
            .filter(|r| r.0 < n)
            .map(|r| (r.0, r.1.min(n - r.0)))
            .collect()
    };
    assert_eq!(clip(&ranges), clip(&before.1), "stderr labels moved");
    job(
        a2,
        json!({"op":"kill","scope":"s","ids":["w"],"grace_ms":200}),
    )
    .await;
    until(a2, "w", "ended").await;
}
