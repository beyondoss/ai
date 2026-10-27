//! Stopping remote `bash` jobs never runs on the runtime thread. A remote job's kill runs on that
//! runtime — the sandbox's `kill` operation, or (a stream-owned job) its stream's future letting the
//! connection go — and the stop waits for it; run inline on `run`'s or `serve`'s current-thread
//! runtime, it could never run, so every job cost the kill's full bound and its stream closed only
//! when the process exited. A transcript switch in `serve` also stopped the jobs without telling the
//! client.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::{ChildStdin, Stdio};
use std::time::{Duration, Instant};

use common::exec_mock::{ExecMock, MockOptions, StreamEnd};
use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_agent");

fn background(id: &str, command: &str) -> String {
    turn_tool_use(
        id,
        "bash",
        &json!({ "command": command, "run_in_background": true }).to_string(),
    )
}

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

/// Whether a live (non-zombie) process runs exactly `cmd`.
fn running(cmd: &str) -> bool {
    std::fs::read_dir("/proc").unwrap().flatten().any(|e| {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            return false;
        };
        let line: Vec<String> = std::fs::read(format!("/proc/{pid}/cmdline"))
            .unwrap_or_default()
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        let zombie = std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .unwrap_or_default()
            .rsplit_once(')')
            .is_some_and(|(_, r)| r.trim_start().starts_with('Z'));
        line.join(" ") == cmd && !zombie
    })
}

/// The job's stream ended — the endpoint killed it on a disconnect (a stream-owned job), or it
/// finished because the sandbox's kill ended the job it was holding.
fn stream_ended(end: Option<(StreamEnd, Instant)>) -> bool {
    matches!(
        end,
        Some((
            StreamEnd::Finished | StreamEnd::Disconnected { killed: true },
            _
        ))
    )
}

/// `new_session` in `serve` stops the old transcript's remote job at once — its stream closed, the
/// endpoint killing it — and says so in a `bash_jobs_stopped` frame.
#[test]
fn a_transcript_switch_stops_remote_jobs_promptly_and_says_so() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    // A v1.2-only sandbox: the jobs run on plain exec commands, whose streams the mock records.
    let mock = rt.block_on(ExecMock::start_with(
        dir.path(),
        MockOptions {
            ..MockOptions::default()
        },
    ));
    let (base, _bodies) = spawn_model_server(vec![
        background("t1", "echo switch-tag; sleep 60.21"),
        turn_text("started it"),
        turn_text("ok, it can run"),
    ]);
    let file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let mut cmd = serve_cmd(BIN, &base, &file);
    cmd.args(["--exec-url", &mock.url]);
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "start it" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let s = rt
        .block_on(mock.wait_for_stream("switch-tag", Duration::from_secs(10)))
        .expect("the job's stream is open");

    send(&mut stdin, json!({ "type": "new_session", "id": "n" }));
    let frames = read_until_response(&mut stdout, "new_session");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let switched = Instant::now();
    let stopped = loop {
        let frame = stdout
            .next_frame(Duration::from_secs(5))
            .expect("a bash_jobs_stopped frame after the switch");
        if frame["type"] == "bash_jobs_stopped" {
            break frame;
        }
    };
    assert_eq!(stopped["jobs"][0]["job"], 1, "{stopped}");
    rt.block_on(mock.wait_for_end(s.pgid, Duration::from_secs(5)));
    let record = mock
        .streams()
        .into_iter()
        .find(|r| r.pgid == s.pgid)
        .unwrap();
    assert!(stream_ended(record.end), "{record:?}");
    let deadline = Instant::now() + Duration::from_secs(5);
    while running("sleep 60.21") {
        assert!(Instant::now() < deadline, "the job itself is still running");
        std::thread::sleep(Duration::from_millis(20));
    }
    let (_, ended) = mock
        .streams()
        .into_iter()
        .find(|r| r.pgid == s.pgid)
        .unwrap()
        .end
        .unwrap();
    assert!(
        ended.saturating_duration_since(switched) < Duration::from_secs(1),
        "the stream closed {} ms after the switch: the stop starved the stream's future",
        ended.saturating_duration_since(switched).as_millis()
    );
    drop(stdin);
    let _ = child.wait();
}

/// The end of `run` stops two remote jobs in well under one kill bound, not one bound each.
#[test]
fn the_end_of_a_run_stops_remote_jobs_without_waiting_out_each_kill() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    // A v1.2-only sandbox: the jobs run on plain exec commands, whose streams the mock records.
    let mock = rt.block_on(ExecMock::start_with(
        dir.path(),
        MockOptions {
            ..MockOptions::default()
        },
    ));
    let (base, bodies) = spawn_model_server(vec![
        background("t1", "echo end-a; sleep 60.22"),
        background("t2", "echo end-b; sleep 60.23"),
        turn_text("started them"),
        // The one stop reminder's reply: the last model call.
        turn_text("they can stop"),
    ]);
    let mut child = run_cmd(BIN)
        .args([
            "run",
            "start two",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "8",
            "--no-session-persistence",
            "--exec-url",
            &mock.url,
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded();
    let deadline = Instant::now() + Duration::from_secs(60);
    while bodies.lock().unwrap().len() < 4 {
        assert!(Instant::now() < deadline, "the run never reached its end");
        std::thread::sleep(Duration::from_millis(10));
    }
    let last_call = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st;
        }
        assert!(Instant::now() < deadline, "run did not exit");
        std::thread::sleep(Duration::from_millis(10));
    };
    let took = last_call.elapsed();
    assert!(status.success(), "{status:?}");
    let jobs: Vec<_> = mock
        .streams()
        .into_iter()
        .filter(|r| r.args.iter().any(|a| a.contains("echo end-")))
        .collect();
    assert_eq!(jobs.len(), 2, "{jobs:?}");
    for s in &jobs {
        assert!(stream_ended(s.end), "{s:?}");
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    while running("sleep 60.22") || running("sleep 60.23") {
        assert!(Instant::now() < deadline, "the jobs are still running");
        std::thread::sleep(Duration::from_millis(20));
    }
    // Inline, each v1.2 kill waited out its 2 s bound: two jobs took over 4 s.
    assert!(
        took < Duration::from_millis(2500),
        "stopping two remote jobs took {} ms",
        took.as_millis()
    );
}
