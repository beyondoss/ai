//! e2e: a session reopened after its process was SIGKILLed stops the `bash` jobs that process left
//! running, and says so truthfully. A SIGKILL (or an OOM kill) runs no exit path, so a background
//! job's processes — and a foreground command's — outlived the process that started them, while the
//! reopen note told the model they "were stopped". Each local job's process group, leader start time
//! and tag are now journaled in the session as it spawns; the next process to open the session finds
//! the survivors by tag, stops them, and the note says "stopped now". Through `run --continue` and a
//! `serve` restart.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server,
    spawn_model_server_with_stalled_response, turn_text, turn_tool_use,
};
use serde_json::json;

const BIN: &str = env!("CARGO_BIN_EXE_agent");

/// The pids of live processes whose command line contains `needle`.
fn running(needle: &str) -> Vec<u32> {
    let me = std::process::id();
    std::fs::read_dir("/proc")
        .unwrap()
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.parse::<u32>().ok())
        .filter(|&pid| pid != me)
        .filter(|pid| {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
            let zombie = stat
                .rsplit_once(')')
                .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'));
            let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            !zombie
                && String::from_utf8_lossy(&cmdline)
                    .replace('\0', " ")
                    .contains(needle)
        })
        .collect()
}

/// Kills whatever a failed test left running, so a failure never leaks a 15-minute `sleep`.
struct Sweep(Vec<String>);

impl Drop for Sweep {
    fn drop(&mut self) {
        for needle in &self.0 {
            for pid in running(needle) {
                let _ = std::process::Command::new("kill")
                    .args(["-KILL", &pid.to_string()])
                    .status();
            }
        }
    }
}

fn count_in_tree(path: &Path, needle: &str) -> usize {
    if path.is_dir() {
        return std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| count_in_tree(&e.path(), needle))
            .sum();
    }
    std::fs::read_to_string(path)
        .map(|s| s.matches(needle).count())
        .unwrap_or(0)
}

/// Unique commands for this test process: a background `sleep` and a foreground one.
fn commands(tag: &str) -> (String, String) {
    let pid = std::process::id();
    (
        format!("sleep 900.{pid}{tag}1"),
        format!("sleep 900.{pid}{tag}2"),
    )
}

/// The model's side of the killed process: start `bg` in the background, then run `fg` in the
/// foreground (which never returns before the kill).
fn killed_turns(bg: &str, fg: &str) -> String {
    spawn_model_server_with_stalled_response(
        vec![
            turn_tool_use(
                "t1",
                "bash",
                &json!({ "command": bg, "run_in_background": true }).to_string(),
            ),
            turn_tool_use("t2", "bash", &json!({ "command": fg }).to_string()),
        ],
        Duration::from_secs(120),
        vec![],
    )
}

/// Wait until both commands run and their processes are journaled (the background job's spawn and
/// its job number, the foreground command's spawn), then SIGKILL `child`.
fn sigkill_once_journaled(mut child: common::ChildGuard, store: &Path, bg: &str, fg: &str) {
    let t0 = Instant::now();
    while running(bg).is_empty()
        || running(fg).is_empty()
        || count_in_tree(store, "bash_job_procs") < 3
    {
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "the jobs never started, or were never journaled"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    child.kill().unwrap(); // SIGKILL: no exit path runs to stop the jobs
    let _ = child.wait();
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !running(bg).is_empty() && !running(fg).is_empty(),
        "the premise: a SIGKILL leaves both commands running"
    );
}

fn assert_stopped_and_said(bg: &str, fg: &str, first_request: &str) {
    assert!(running(bg).is_empty(), "the background job was stopped");
    assert!(running(fg).is_empty(), "the foreground command was stopped");
    assert!(
        first_request.contains(&format!("- job 1: `{bg}` (stopped now)")),
        "the note says job 1 was stopped now: {first_request}"
    );
    assert!(
        first_request.contains(&format!(
            "`{fg}` (a command still running when it closed; stopped now)"
        )),
        "the note names the foreground command: {first_request}"
    );
    assert!(
        !first_request.contains("(not running)"),
        "nothing is reported gone that was running: {first_request}"
    );
}

#[test]
fn run_continue_after_a_sigkill_stops_the_jobs_it_left_and_says_so() {
    let (bg, fg) = commands("1");
    let _sweep = Sweep(vec![bg.clone(), fg.clone()]);
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions");
    let base = killed_turns(&bg, &fg);
    let child = run_cmd(BIN)
        .args([
            "run",
            "start them",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
        ])
        .arg(&sessions)
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded();
    sigkill_once_journaled(child, &sessions, &bg, &fg);

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let out = run_cmd(BIN)
        .args([
            "run",
            "are they still running?",
            "--continue",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
        ])
        .arg(&sessions)
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_stopped_and_said(&bg, &fg, &bodies.lock().unwrap()[0]);
}

#[test]
fn a_serve_restart_after_a_sigkill_stops_the_jobs_it_left_and_says_so() {
    let (bg, fg) = commands("2");
    let _sweep = Sweep(vec![bg.clone(), fg.clone()]);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("s.jsonl");
    let file_str = file.to_string_lossy().into_owned();
    let base = killed_turns(&bg, &fg);
    let mut child = serve_cmd(BIN, &base, &file_str)
        .current_dir(dir.path())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "start them" })
    )
    .unwrap();
    stdin.flush().unwrap();
    sigkill_once_journaled(child, &file, &bg, &fg);

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let mut child = serve_cmd(BIN, &base, &file_str)
        .current_dir(dir.path())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut frames = common::child_frames(&mut child);
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "are they still running?" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let resp = read_until_response(&mut frames, "prompt");
    assert_eq!(resp.last().unwrap()["success"], true, "{resp:#?}");
    assert_stopped_and_said(&bg, &fg, &bodies.lock().unwrap()[0]);
    let _ = child.kill();
    let _ = child.wait();
}
