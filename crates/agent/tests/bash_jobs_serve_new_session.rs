//! `serve` e2e: `new_session` stops the old transcript's `bash` jobs even when the session keeps its
//! id — a pinned session (`--session-id`, which every daemon/WebSocket/HTTP/UDS session is) and a
//! `--session-file` session both blank in place. Keyed on the session id, the switch check missed
//! both: the job ran on, and its end woke the blank transcript about a command it never ran.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::{ChildStdin, Command};
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, read_until_response, serve_cmd, serve_dir_cmd, spawn_model_server, turn_text,
    turn_tool_use,
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

/// Kills the job's process if the test fails before `serve` did, so a failure leaks nothing.
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

/// Start an adopted job under `cmd`, send `new_session`, and check the job is stopped and does not
/// wake the blank transcript.
fn new_session_stops_the_jobs(make: impl FnOnce(&str) -> Command) {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let command = format!(
        "sh -c 'echo $$ > {}; exec sleep 60' & echo launched",
        pidfile.display()
    );
    let (base, bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &json!({ "command": command }).to_string()),
        turn_text("started it"),
        // Spares: a wake (the defect) would take one instead of starving.
        turn_text("spare"),
        turn_text("spare"),
    ]);
    let mut child = make(&base).spawn_guarded();
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
    assert!(alive(pid), "the job runs past its turn");
    let calls = bodies.lock().unwrap().len();

    send(&mut stdin, json!({ "type": "new_session", "id": "n" }));
    let frames = read_until_response(&mut stdout, "new_session");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");

    let deadline = Instant::now() + Duration::from_secs(5);
    while alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!alive(pid), "the old transcript's job survived new_session");
    // Nothing wakes the blank transcript: no turn of its own within a wake's settle and more.
    if let Ok(frame) = stdout.next_frame(Duration::from_secs(1)) {
        assert!(
            frame["command"] != "prompt" && frame["event"]["kind"] != "background",
            "the blank transcript woke: {frame}"
        );
    }
    assert_eq!(
        bodies.lock().unwrap().len(),
        calls,
        "no model call after it"
    );
    drop(stdin);
    let _ = child.wait();
}

#[test]
fn a_pinned_session_stops_its_jobs_on_new_session() {
    let dir = tempfile::tempdir().unwrap();
    new_session_stops_the_jobs(|base| {
        let mut cmd = serve_dir_cmd(BIN, base, dir.path().to_str().unwrap());
        cmd.args(["--session-id", "pinned-one"]);
        cmd
    });
}

#[test]
fn a_session_file_stops_its_jobs_on_new_session() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    new_session_stops_the_jobs(|base| serve_cmd(BIN, base, &file));
}
