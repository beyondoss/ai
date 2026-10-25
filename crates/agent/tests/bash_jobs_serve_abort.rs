//! `serve` e2e: an `abort` while the model waits on a job with `bash_job` cancels the wait, not the
//! job. The model was told a background job "keeps running after your turn ends"; the cancelled wait
//! used to drop the job it had taken out of the table, killing it with no `bash_jobs_stopped` frame,
//! and the next `bash_job` found nothing.
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

#[test]
fn an_abort_during_a_bash_job_wait_keeps_the_job() {
    let dir = tempfile::tempdir().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let start = json!({
        "command": format!("echo $$ > {}; exec sleep 60", pidfile.display()),
        "run_in_background": true,
    });
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &start.to_string()),
        turn_text("started"),
        // p2: wait on it (aborted).
        turn_tool_use("t2", "bash_job", &json!({ "job": 1 }).to_string()),
        // p3: peek at it.
        turn_tool_use(
            "t3",
            "bash_job",
            &json!({ "job": 1, "timeout_ms": 0 }).to_string(),
        ),
        turn_text("peeked"),
        turn_text("spare"),
    ]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
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
    assert!(alive(pid));

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p2", "message": "wait for it" }),
    );
    loop {
        let frame = stdout.next_frame(Duration::from_secs(20)).unwrap();
        if frame["event"]["kind"] == "tool_start" && frame["event"]["name"] == "bash_job" {
            break;
        }
    }
    std::thread::sleep(Duration::from_millis(300));
    send(&mut stdin, json!({ "type": "abort", "id": "a1" }));
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        !frames.iter().any(|f| f["type"] == "bash_jobs_stopped"),
        "nothing was stopped: {frames:?}"
    );
    let wait_end = frames
        .iter()
        .find(|f| f["event"]["kind"] == "tool_end")
        .expect("the cancelled wait ends")
        .to_string();
    assert!(
        wait_end.contains("Wait cancelled; job 1 is still running."),
        "{wait_end}"
    );
    std::thread::sleep(Duration::from_millis(300));
    assert!(alive(pid), "the abort killed the background job");

    // The job is still the model's: a peek finds it running.
    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p3", "message": "peek" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let peek = frames
        .iter()
        .find(|f| f["event"]["kind"] == "tool_end")
        .expect("the peek ends")
        .to_string();
    assert!(!peek.contains("no job 1"), "{peek}");
    assert!(peek.contains("Job 1 is still running"), "{peek}");
    assert!(alive(pid));
    drop(stdin);
    let _ = child.wait();
}
