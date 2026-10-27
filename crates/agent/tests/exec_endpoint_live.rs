//! **Live** end-to-end: the real binary driving its whole toolset against a real remote target.
//!
//! Everything else tests the protocol against a mock that runs commands on this machine. That proves
//! wiring; it proves nothing about a genuinely separate, isolated filesystem. This does.
//!
//! ## Vendor-neutral by construction
//!
//! The target is whatever you point it at — there is no provider knowledge here, only the two flags
//! the agent actually ships:
//!
//! ```sh
//! # any transparent transport (docker, ssh, kubectl, podman …)
//! AGENT_LIVE_EXEC_CMD='docker exec my-container {}' \
//!   cargo nextest run -p beyond-agent -E 'test(/^exec_endpoint_live::/)' --no-capture
//!
//! # any HTTP endpoint speaking the exec protocol
//! AGENT_LIVE_EXEC_URL=https://exec.example/run AGENT_LIVE_EXEC_HEADER='Authorization: Bearer …' \
//!   cargo nextest run -p beyond-agent -E 'test(/^exec_endpoint_live::/)' --no-capture
//! ```
//!
//! `AGENT_LIVE_WORKDIR` names a directory that exists **on the target and not on this host** (default
//! `/work`) — that asymmetry is what makes the result meaningful rather than a local no-op.
//!
//! **This runs in CI.** The `exec-live` shard stands up an Alpine container and points the tests at
//! it. Alpine is deliberate: it is busybox, whose `grep` has no `-Z`/`-I`/`--include` and whose
//! `find` has no `-printf`, so it exercises the hardest capability rung — and it is what a large
//! share of real sandboxes actually are. Running against a GNU-flavoured image would quietly skip the
//! code path most likely to be wrong.
//!
//! Skipped when neither variable is set, so a local `cargo test` needs no infrastructure. When one
//! *is* set it fails rather than skips: at that point a target has been asserted to exist, and a
//! silent pass would be worse than no test.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use serde_json::json;

const BIN: &str = env!("CARGO_BIN_EXE_agent");

/// The exec flags to pass through, or `None` to skip.
fn target() -> Option<Vec<String>> {
    if let Ok(url) = std::env::var("AGENT_LIVE_EXEC_URL")
        && !url.is_empty()
    {
        let mut v = vec!["--exec-url".to_string(), url];
        if let Ok(h) = std::env::var("AGENT_LIVE_EXEC_HEADER")
            && !h.is_empty()
        {
            v.push("--exec-header".to_string());
            v.push(h);
        }
        return Some(v);
    }
    if let Ok(cmd) = std::env::var("AGENT_LIVE_EXEC_CMD")
        && !cmd.is_empty()
    {
        return Some(vec!["--exec-cmd".to_string(), cmd]);
    }
    None
}

fn workdir() -> String {
    std::env::var("AGENT_LIVE_WORKDIR").unwrap_or_else(|_| "/work".to_string())
}

macro_rules! target_or_skip {
    () => {
        match target() {
            Some(t) => t,
            None => {
                eprintln!(
                    "skipping: set AGENT_LIVE_EXEC_CMD or AGENT_LIVE_EXEC_URL to run live tests"
                );
                return;
            }
        }
    };
}

/// Drive the real binary for one scripted turn and return `(stdout, stderr)`.
fn run_agent(flags: &[String], turns: Vec<String>) -> (String, String) {
    let (base, _bodies) = common::spawn_model_server(turns);
    let mut cmd = common::run_cmd(BIN);
    cmd.args(["run", "do the task"]);
    cmd.args(flags);
    cmd.args([
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--max-steps",
        "6",
        "--no-session-persistence",
        "--json",
    ]);
    let out = cmd.output().expect("run the agent");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn the_tools_read_a_file_that_exists_only_on_the_target() {
    // The whole point. `AGENT_LIVE_WORKDIR` must not exist on this host, so a passing read cannot
    // have been satisfied locally.
    let flags = target_or_skip!();
    let dir = workdir();
    assert!(
        !std::path::Path::new(&dir).exists(),
        "{dir} exists on this host too, so this test would prove nothing — point \
         AGENT_LIVE_WORKDIR at a path that only exists on the target"
    );

    let path = format!("{dir}/marker.txt");
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use("t1", "read", &json!({ "path": &path }).to_string()),
            common::turn_text("done"),
        ],
    );
    assert!(
        stderr.contains("remote exec endpoint"),
        "the agent should report that it attached; stderr: {stderr}"
    );
    assert!(
        stdout.contains("ONLY"),
        "expected the target-only marker in the tool result.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

#[test]
fn grep_searches_the_targets_filesystem() {
    let flags = target_or_skip!();
    let dir = workdir();
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "grep",
                &json!({ "pattern": "NEEDLE", "path": &dir }).to_string(),
            ),
            common::turn_text("done"),
        ],
    );
    assert!(
        stdout.contains("a.rs"),
        "grep should find the seeded match on the target.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

#[test]
fn bash_runs_on_the_target_not_this_host() {
    // The hole #41 shipped. `bash` must reach the same machine as the filesystem tools, proven by a
    // hostname the host cannot produce.
    let flags = target_or_skip!();
    let (stdout, _stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "bash",
                &json!({ "command": "uname -n; cat /etc/hostname 2>/dev/null || true" })
                    .to_string(),
            ),
            common::turn_text("done"),
        ],
    );
    let host = String::from_utf8_lossy(
        &std::process::Command::new("uname")
            .arg("-n")
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .to_string();
    assert!(
        !stdout.contains(&host),
        "`bash` reported this host's name ({host}) — it ran locally, not on the target.\n{stdout}"
    );
}

#[test]
fn a_write_past_the_argv_limit_round_trips_through_the_target() {
    // A remote write used to ship the whole file as one argv entry, which Linux caps at 128 KiB, so
    // anything over ~96 KiB failed. CI's `docker exec` has no `-i` — the far side gets no stdin, the
    // probe sees that — so this runs the chunked fallback on busybox, the path most likely to trip on
    // a shell difference. Behind `ssh` or `docker exec -i` it runs the stdin path instead.
    use sha2::Digest as _;
    let flags = target_or_skip!();
    let dir = workdir();
    let path = format!("{dir}/big-written-by-agent.txt");
    let line = "chunk boundary check — ünïcödé, a $dollar and a 'quote'\n";
    // Unique per run, so a file left by an earlier run cannot satisfy the digest check.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let content = format!("run {nonce}\n{}", line.repeat(200 * 1024 / line.len() + 1));
    let digest = hex::encode(sha2::Sha256::digest(content.as_bytes()));
    // The check runs on the target, whose shell is POSIX `sh` — Alpine has no `/bin/bash`.
    let flags = [flags, vec!["--bash-shell-path".into(), "/bin/sh".into()]].concat();
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "write",
                &json!({ "path": &path, "content": &content }).to_string(),
            ),
            common::turn_tool_use(
                "t2",
                "bash",
                &json!({ "command": format!("sha256sum < {path}; ls {dir}") }).to_string(),
            ),
            common::turn_text("done"),
        ],
    );
    // The tail only: the head of stdout echoes the 200 KiB `write` input back.
    let tail: String = {
        let chars: Vec<char> = stdout.chars().collect();
        chars[chars.len().saturating_sub(2000)..].iter().collect()
    };
    assert!(
        stdout.contains(&digest),
        "the {} bytes written must be the bytes on the target.\n--- stdout (tail) ---\n{tail}\n\
         --- stderr ---\n{stderr}",
        content.len()
    );
    assert!(
        !stdout.contains(".tmp."),
        "a completed write must not leave its temp file behind.\n--- stdout (tail) ---\n{tail}"
    );
}

#[test]
fn write_then_read_round_trips_through_the_target() {
    let flags = target_or_skip!();
    let dir = workdir();
    let path = format!("{dir}/written-by-agent.txt");
    let body = "round trip — ünïcödé and a $dollar";
    let (stdout, stderr) = run_agent(
        &flags,
        vec![
            common::turn_tool_use(
                "t1",
                "write",
                &json!({ "path": &path, "content": format!("{body}\n") }).to_string(),
            ),
            common::turn_tool_use("t2", "read", &json!({ "path": &path }).to_string()),
            common::turn_text("done"),
        ],
    );
    assert!(
        stdout.contains("ünïcödé") && stdout.contains("$dollar"),
        "content must survive the round trip verbatim.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        !std::path::Path::new(&path).exists(),
        "the file must not appear on this host — that would mean the write never left"
    );
}

/// The runner the live flags describe, built in-process — for a test that needs to hold the job
/// table itself.
fn live_runner() -> Option<std::sync::Arc<dyn beyond_agent::tools::exec::CommandRunner>> {
    use beyond_agent::exec_endpoint::{HttpExecRunner, TemplateRunner};
    if let Ok(url) = std::env::var("AGENT_LIVE_EXEC_URL")
        && !url.is_empty()
    {
        let mut r = HttpExecRunner::new(url).unwrap();
        if let Ok(h) = std::env::var("AGENT_LIVE_EXEC_HEADER")
            && !h.is_empty()
        {
            let (n, v) = HttpExecRunner::parse_header(&h).unwrap();
            r = r.with_header(n, v);
        }
        return Some(std::sync::Arc::new(r));
    }
    let cmd = std::env::var("AGENT_LIVE_EXEC_CMD")
        .ok()
        .filter(|c| !c.is_empty())?;
    Some(std::sync::Arc::new(TemplateRunner::parse(&cmd).unwrap()))
}

/// The kill a `bash_job` reports must be true **on the target**. Against a real `docker exec`
/// (CI's `exec-live` shard), killing the local transport leaves the command running in the
/// container, so this fails unless the kill went through the transport. The target's pid 1 is a
/// plain `sleep` that never reaps, so "dead" is read from `/proc` (a zombie is dead), not `kill -0`.
#[tokio::test]
async fn a_killed_bash_job_dies_on_the_target() {
    use common::remote_bash::{call, job_number, kill_job, remote_bash};
    use std::time::{Duration, Instant};
    let Some(runner) = live_runner() else {
        eprintln!("skipping: set AGENT_LIVE_EXEC_CMD or AGENT_LIVE_EXEC_URL to run live tests");
        return;
    };
    let (bash, jobs) = remote_bash(runner.clone(), Duration::from_secs(2));
    let out = call(
        bash.as_ref(),
        json!({ "command": "(sleep 300) & echo \"pids $$ $!\"; sleep 300" }),
    )
    .await
    .unwrap_or_else(|e| e);
    let n = job_number(&out).unwrap_or_else(|| {
        panic!("a silent command on a streaming target must yield as a job: {out}")
    });
    // The pids arrive when the transport has started the command, which on a loaded runner can be
    // after the 2 s idle window: the first result is then a bare yield. The output is not lost, it
    // comes back on the job's next wait.
    let mut seen = out;
    let give_up = Instant::now() + Duration::from_secs(30);
    while !seen.contains("pids ") && Instant::now() < give_up {
        let more = call(&jobs, json!({ "job": n })).await.unwrap_or_else(|e| e);
        seen.push('\n');
        seen.push_str(&more);
    }
    let pids: Vec<String> = seen
        .lines()
        .find_map(|l| l.strip_prefix("pids "))
        .unwrap_or_else(|| panic!("no pids in: {seen}"))
        .split_whitespace()
        .map(str::to_string)
        .collect();
    let killed = kill_job(&jobs, n).await.unwrap();
    assert!(killed.contains("killed"), "{killed}");

    let alive = format!(
        "alive() {{ s=$(cut -d' ' -f3 \"/proc/$1/stat\" 2>/dev/null); [ -n \"$s\" ] && [ \"$s\" != Z ]; }}; \
         alive {} || alive {}",
        pids[0], pids[1]
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let r = runner
            .run(
                "sh",
                &["-c".to_string(), alive.clone()],
                None,
                Duration::from_secs(10),
            )
            .await
            .unwrap();
        if r.code != Some(0) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the job's processes {pids:?} are still running on the target after bash_job kill"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// What a command leaves running on the target (`nohup server &`) is adopted as a job and stopped
/// with the session — as a local shell's leftovers are. Before, nothing tracked it: it ran (and held
/// its port) until the sandbox died. Read from `/proc` for the same reason as above.
#[tokio::test]
async fn a_leftover_on_the_target_is_adopted_and_stopped_with_the_session() {
    use common::remote_bash::{call, remote_bash};
    use std::time::{Duration, Instant};
    let Some(runner) = live_runner() else {
        eprintln!("skipping: set AGENT_LIVE_EXEC_CMD or AGENT_LIVE_EXEC_URL to run live tests");
        return;
    };
    let (bash, _jobs) = remote_bash(runner.clone(), Duration::from_secs(60));
    let out = call(
        bash.as_ref(),
        json!({ "command": "nohup sleep 301 >/dev/null 2>&1 & echo \"pid $!\"" }),
    )
    .await
    .unwrap_or_else(|e| e);
    assert!(
        out.contains("tracked as job"),
        "the leftover must be adopted: {out}"
    );
    let pid = out
        .lines()
        .find_map(|l| l.strip_prefix("pid "))
        .unwrap_or_else(|| panic!("no pid in: {out}"))
        .trim()
        .to_string();
    let alive = format!(
        "s=$(cut -d' ' -f3 \"/proc/{pid}/stat\" 2>/dev/null); [ -n \"$s\" ] && [ \"$s\" != Z ]"
    );
    let probe = |script: String| {
        let runner = runner.clone();
        async move {
            runner
                .run(
                    "sh",
                    &["-c".to_string(), script],
                    None,
                    Duration::from_secs(10),
                )
                .await
                .unwrap()
                .code
        }
    };
    assert_eq!(
        probe(alive.clone()).await,
        Some(0),
        "the leftover runs on the target"
    );
    let stopped = tokio::task::spawn_blocking({
        let bash = bash.clone();
        move || bash.jobs().stop_all()
    })
    .await
    .unwrap();
    assert_eq!(stopped.len(), 1, "{stopped:?}");
    let deadline = Instant::now() + Duration::from_secs(10);
    while probe(alive.clone()).await == Some(0) {
        assert!(
            Instant::now() < deadline,
            "the leftover {pid} still runs on the target after the session's jobs were stopped"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Wait (bounded) until `cond` holds.
async fn until(what: &str, cond: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !cond() {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Sandbox-owned jobs (`exec_endpoint::jobs`) inside the real target — busybox in CI: a held job
/// streams and dies with its hold, a kept one outlives it and is read back from an offset, a kill
/// stops it and removes its directory, and a job started kept is polled to its end. The baseline's
/// whole job is to work on exactly this image.
#[tokio::test]
async fn a_sandbox_job_lives_in_the_target_and_outlives_its_hold_once_kept() {
    use beyond_agent::exec_endpoint::jobs::{JobEvent, JobId, JobScope, JobSpec, JobStatus};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    let Some(runner) = live_runner() else {
        eprintln!("skipping: set AGENT_LIVE_EXEC_CMD or AGENT_LIVE_EXEC_URL to run live tests");
        return;
    };
    let jobs = runner
        .clone()
        .sandbox_jobs()
        .expect("a remote target has sandbox jobs");
    assert!(
        jobs.available().await,
        "the target lacks a tool the job scripts need"
    );
    let scope = JobScope::for_session(Some("live"), JobId::fresh().as_str());
    let group_alive = |pgid: u32| {
        let runner = runner.clone();
        async move {
            runner
                .run(
                    "sh",
                    &["-c".into(), format!("kill -0 -{pgid} 2>/dev/null")],
                    None,
                    Duration::from_secs(10),
                )
                .await
                .unwrap()
                .code
                == Some(0)
        }
    };
    let spec = |cmd: &str| JobSpec {
        program: "sh".into(),
        args: vec!["-c".into(), cmd.into()],
        cwd: None,
        label: cmd.into(),
    };
    let started = Mutex::new(None::<u32>);
    let output = Mutex::new(Vec::<u8>::new());
    let events = |e: JobEvent<'_>| match e {
        JobEvent::Started(g) => *started.lock().unwrap() = Some(g),
        JobEvent::Output(b) => output.lock().unwrap().extend_from_slice(b),
        _ => {}
    };

    // Held, then dropped: it dies.
    let held = JobId::fresh();
    let s = spec("echo held-up; sleep 300");
    tokio::select! {
        r = jobs.run_held(&scope, &held, &s, &events) => panic!("ended early: {r:?}"),
        () = until("held output", || output.lock().unwrap().ends_with(b"held-up\n")) => {}
    }
    let pgid = started.lock().unwrap().unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    while group_alive(pgid).await {
        assert!(
            Instant::now() < deadline,
            "the held job outlived its hold on the target"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Kept, then dropped: it lives, and reads back from where the model left off. The held job's
    // group is forgotten, so this one's is the only one the checks below can see.
    output.lock().unwrap().clear();
    *started.lock().unwrap() = None;
    let kept = JobId::fresh();
    let s = spec("echo one; sleep 1; echo two; sleep 300");
    let mut run = Box::pin(jobs.run_held(&scope, &kept, &s, &events));
    // Boxed, not stack-pinned: dropping it below must really close the hold.
    tokio::select! {
        r = &mut run => panic!("ended early: {r:?}"),
        () = until("kept output", || output.lock().unwrap().as_slice() == b"one\n") => {}
    }
    // Kept while the hold is up, then the hold goes.
    jobs.keep(&scope, &kept, 1, "background", 4).await.unwrap();
    drop(run);
    let pgid = started.lock().unwrap().unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(
        group_alive(pgid).await,
        "a kept job died with its hold on the target"
    );
    let listed = jobs.list(&scope).await.unwrap();
    let j = listed.iter().find(|j| j.id == kept).unwrap();
    assert_eq!((j.num, j.seen, j.status), (Some(1), 4, JobStatus::Running));
    let c = jobs.read(&scope, &kept, j.seen, 1024).await.unwrap();
    assert_eq!(c.data, b"two\n");
    assert!(
        jobs.kill(&scope, std::slice::from_ref(&kept), true)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(!group_alive(pgid).await);
    assert_eq!(
        jobs.read(&scope, &kept, 0, 1).await.unwrap().status,
        JobStatus::Gone
    );

    // Started kept (no hold) and polled to its end.
    let polled = JobId::fresh();
    jobs.start_kept(&scope, &polled, &spec("echo a; sleep 0.5; echo b; exit 6"))
        .await
        .unwrap();
    output.lock().unwrap().clear();
    assert_eq!(jobs.follow(&scope, &polled, 0, &events).await.unwrap(), 6);
    assert_eq!(output.lock().unwrap().as_slice(), b"a\nb\n");
    jobs.wipe(&scope).await.unwrap();
    assert!(jobs.list(&scope).await.unwrap().is_empty());
}
