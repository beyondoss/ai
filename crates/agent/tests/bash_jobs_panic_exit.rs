//! A panic out of `serve`'s main future still stops every `bash` job. The panic path used to run
//! only the MCP sweep, not `kill_live_jobs`: a job whose table the unwind never dropped ran on after
//! the process died.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::{ChildStdin, Command};
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_agent");

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn alive(pid: u32) -> bool {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let state = stat
        .rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or("");
    !stat.is_empty() && state != "Z" && state != "X"
}

fn read_pid(path: &std::path::Path) -> u32 {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(pid) = text.trim().parse()
        {
            return pid;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("{} was never written", path.display());
}

/// Kills the job's process if the test fails, so a failure leaks nothing.
struct PidGuard(u32);

impl Drop for PidGuard {
    fn drop(&mut self) {
        if alive(self.0) {
            let _ = Command::new("kill")
                .args(["-9", &self.0.to_string()])
                .status();
        }
    }
}

#[test]
fn a_panic_out_of_serve_stops_its_bash_jobs() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let command = format!(
        "sh -c 'echo $$ > {}; exec sleep 60' & echo launched",
        pidfile.display()
    );
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &json!({ "command": command }).to_string()),
        turn_text("started it"),
        turn_text("ok, it can run"),
    ]);
    let file = tmp.path().join("s.jsonl").to_string_lossy().into_owned();
    let mut cmd = serve_cmd(BIN, &base, &file);
    cmd.env("BEYOND_AI_AGENT_TEST_PANICS", "1");
    let mut child = cmd.spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "start it" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let pid = read_pid(&pidfile);
    let _guard = PidGuard(pid);
    assert!(alive(pid), "the adopted job runs past its turn");

    send(&mut stdin, json!({ "type": "__test_panic", "id": "boom" }));
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "serve hung after a panic");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(101), "{status:?}");
    assert!(
        !alive(pid),
        "serve panicked and left its bash job {pid} running"
    );
    drop(stdin);
}

/// The defect itself: a daemon's sessions run on their own threads, so a panic out of the main
/// future drops none of their job tables — only `kill_live_jobs` reaches those jobs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panic_out_of_a_daemons_main_future_stops_its_sessions_bash_jobs() {
    let home = tempfile::tempdir().unwrap();
    let pidfile = home.path().join("pid");
    let trigger = home.path().join("panic-now");
    let command = format!(
        "sh -c 'echo $$ > {}; exec sleep 60' & echo launched",
        pidfile.display()
    );
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &json!({ "command": command }).to_string()),
        turn_text("started it"),
        turn_text("ok, it can run"),
    ]);
    let trigger_env = trigger.to_string_lossy().into_owned();
    let (mut daemon, port) = common::mcp_events_fixture::spawn_daemon_env(
        home.path(),
        &base,
        &[],
        // A multi-threaded runtime, as a busy daemon (or `--service`) runs: abandoning it drops
        // the session tasks on its workers, if at all, after the exit path has looked.
        &[
            ("BEYOND_AI_AGENT_TEST_PANIC_MAIN_ON", &trigger_env),
            ("BEYOND_AI_AGENT_TOKIO_WORKER_THREADS", "2"),
        ],
    );
    let mut ws = common::ws_connect(port, Some("jobs")).await;
    common::ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "start it" }),
    )
    .await;
    let done = common::mcp_events_fixture::ws_next(
        &mut ws,
        Duration::from_secs(30),
        "the prompt's response",
        |f| f["type"] == "response" && f["id"] == "p1",
    )
    .await;
    assert_eq!(done["success"], true, "{done:#}");
    let pid = read_pid(&pidfile);
    let _guard = PidGuard(pid);
    assert!(alive(pid), "the adopted job runs past its turn");

    std::fs::write(&trigger, b"").unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = daemon.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "the daemon hung after a panic");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(status.code(), Some(101), "{status:?}");
    assert!(
        !alive(pid),
        "the daemon panicked and left a session's bash job {pid} running"
    );
}
