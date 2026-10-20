//! `serve --listen` e2e: a daemon session's `bash` jobs end with the session, not with the process.
//! A reaped session stops the job it left running; before the fix the job's own task kept the job
//! table alive (task → runner → cell → table → job → task), so it ran on, unreachable, until the
//! daemon exited.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{
    ChildGuard, ISOLATED_HOME, spawn_listening, spawn_model_server, turn_text, turn_tool_use,
    ws_connect, ws_read_until_response, ws_send,
};
use serde_json::json;

fn daemon(base: &str, dir: &str, idle_secs: u64) -> (ChildGuard, u16) {
    spawn_listening(
        crate::common::agent_cmd()
            .args([
                "serve",
                "--gateway-url",
                base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                dir,
                "--session-idle-timeout",
                &idle_secs.to_string(),
            ])
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )
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

/// Kills the job's process if the test fails before the daemon did, so a failure leaks nothing.
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

#[tokio::test]
async fn a_reaped_session_stops_its_jobs() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    // `nohup … &`: the command exits at once, leaving the sleep running as an adopted job.
    let command = format!(
        "nohup sh -c 'echo $$ > {}; exec sleep 89' > /dev/null 2>&1 & echo up",
        pidfile.display()
    );
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t1", "bash", &json!({ "command": command }).to_string()),
        turn_text("it is up"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let (mut child, port) = daemon(&base, dir.path().to_str().unwrap(), 1);
    {
        let mut ws = ws_connect(port, Some("reapjobs01")).await;
        ws_send(
            &mut ws,
            json!({ "type": "prompt", "id": "p", "message": "start it" }),
        )
        .await;
        let frames = ws_read_until_response(&mut ws, "prompt").await;
        assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    }
    let pid = read_pid(&pidfile);
    let _guard = PidGuard(pid);
    assert!(alive(pid), "the job runs past its turn");

    // Detached: the reaper ends the session after 1 s (plus a tick); its job goes with it, while
    // the daemon itself keeps running.
    let deadline = Instant::now() + Duration::from_secs(15);
    while alive(pid) && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        child.try_wait().unwrap().is_none(),
        "the daemon is still up: the session's end, not the process's, stopped the job"
    );
    assert!(!alive(pid), "the reaped session's job is still running");
    let _ = child.kill();
    let _ = child.wait();
}
