//! `serve` e2e: what a client's host `bash` command leaves running (`cmd &`) is the client's, not the
//! model's. It used to be adopted into the model's job table, so its end woke the idle session — an
//! unrequested, paid model turn whose input quoted the command's output, even with
//! `exclude_from_context: true`. Now it lives in a table of the session's own that nothing wakes on,
//! and ends with the session.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Write;
use std::process::{ChildStdin, Command};
use std::time::{Duration, Instant};

use common::{SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

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

/// Kills the process if the test fails before `serve` did, so a failure leaks nothing.
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
fn a_host_bash_leftover_never_wakes_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![turn_text("hello"), turn_text("spare")]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({
            "type": "bash",
            "id": "b1",
            "command": "(sleep 1; echo CLIENT-SECRET-OUTPUT) &",
            "exclude_from_context": true,
        }),
    );
    let frames = read_until_response(&mut stdout, "bash");
    let result = frames.last().unwrap()["data"]["result"].to_string();
    assert!(result.contains("runs until the session ends"), "{result}");
    assert!(
        !result.contains("bash_job"),
        "no model-facing instructions in a client's result: {result}"
    );

    // The leftover exits after 1 s: nothing wakes (a wake would settle 200 ms later).
    let deadline = Instant::now() + Duration::from_secs(3);
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        if let Ok(frame) = stdout.next_frame(left) {
            assert!(
                frame["command"] != "prompt" && frame["event"]["kind"] != "background",
                "the client's command woke the model: {frame}"
            );
        }
    }
    assert_eq!(bodies.lock().unwrap().len(), 0, "no model call");

    // Nor does the model's next turn hear of it.
    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "hi" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let body = bodies.lock().unwrap()[0].clone();
    assert!(!body.contains("CLIENT-SECRET-OUTPUT"), "{body}");
    assert!(!body.contains("[job "), "{body}");

    // A leftover still running ends with the session.
    send(
        &mut stdin,
        json!({
            "type": "bash",
            "id": "b2",
            "command": format!("sh -c 'echo $$ > {}; exec sleep 60' &", pidfile.display()),
        }),
    );
    let frames = read_until_response(&mut stdout, "bash");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let pid = read_pid(&pidfile);
    let _guard = PidGuard(pid);
    assert!(alive(pid));
    drop(stdin);
    let _ = child.wait();
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !alive(pid),
        "the session's end leaves the client's leftover running"
    );
}
