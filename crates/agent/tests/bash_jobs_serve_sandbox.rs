//! e2e: a `serve` session's background job lives in its sandbox, not in the process serving it.
//! The process ends (its client went away — what a reap, a deploy or a crash is to the session),
//! the job runs on, and the next process to open the session reports it **still running** under its
//! number, shows its new output, and can kill it. A job that ended in between is reported, once, as
//! an ordinary finish notice. Before, the stream that carried the job closed with the process and
//! the endpoint killed it.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::ChildStdin;
use std::time::{Duration, Instant};

use common::exec_mock::ExecMock;
use common::{
    SpawnGuarded, read_until_response, serve_cmd, spawn_model_server, turn_text, turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_agent");

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn tool(id: &str, name: &str, input: Value) -> String {
    turn_tool_use(id, name, &input.to_string())
}

/// One `serve` process on `session_file`, its tools on `exec_url`: send `prompt`, wait for its
/// response, close stdin (the session closes, not ends), and return the model request bodies.
fn serve_once(session_file: &str, exec_url: &str, turns: Vec<String>, prompt: &str) -> Vec<String> {
    let (base, bodies) = spawn_model_server(turns);
    let mut child = serve_cmd(BIN, &base, session_file)
        .args(["--exec-url", exec_url])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);
    send(&mut stdin, json!({ "type": "prompt", "message": prompt }));
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    drop(stdin);
    let _ = child.wait();
    bodies.lock().unwrap().clone()
}

/// The pid a request body quotes as `jobpid=N` (JSON-escaped newlines around it).
fn quoted_pid(bodies: &[String]) -> u32 {
    bodies
        .iter()
        .find_map(|b| {
            // The command itself says `jobpid=$$`; its output says the number.
            b.match_indices("jobpid=").find_map(|(at, _)| {
                b[at + 7..]
                    .split(|c: char| !c.is_ascii_digit())
                    .next()?
                    .parse()
                    .ok()
            })
        })
        .unwrap_or_else(|| panic!("no pid quoted in {bodies:?}"))
}

fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| s.rsplit_once(')').map(|(_, r)| r.trim_start().to_string()))
        .is_some_and(|r| !r.starts_with('Z'))
}

fn wait_dead(pid: u32) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(pid) {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

/// Kills a job a failed assertion would leave running.
struct Reap(Option<u32>);

impl Drop for Reap {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &format!("-{pid}")])
                .status();
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_background_job_outlives_its_serve_process_and_the_next_one_reports_it_running() {
    let dir = tempfile::tempdir().unwrap();
    let exec = ExecMock::start(dir.path()).await;
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let url = exec.url.clone();
    let first = {
        let (session_file, url) = (session_file.clone(), url.clone());
        tokio::task::spawn_blocking(move || {
            serve_once(
                &session_file,
                &url,
                vec![
                    tool(
                        "t1",
                        "bash",
                        json!({
                            "command": "echo \"jobpid=$$\"; sleep 2; echo while-away; sleep 300",
                            "run_in_background": true,
                        }),
                    ),
                    turn_text("the server is up"),
                    turn_text("ok"),
                    turn_text("spare"),
                ],
                "start the server",
            )
        })
        .await
        .unwrap()
    };
    let pid = quoted_pid(&first);
    let mut reap = Reap(Some(pid));
    assert!(
        first.iter().any(|b| b.contains(
            "it keeps running in the sandbox until it exits, you stop it, or the session ends"
        )),
        "the model was told the sandbox's lifetime: {first:?}"
    );
    std::thread::sleep(Duration::from_secs(3));
    assert!(alive(pid), "the job died with the serve process");

    let second = tokio::task::spawn_blocking(move || {
        serve_once(
            &session_file,
            &url,
            vec![
                tool("t2", "bash_job", json!({ "job": 1, "timeout_ms": 0 })),
                tool("t3", "bash_job", json!({ "job": 1, "kill": true })),
                turn_text("stopped it"),
                turn_text("spare"),
                turn_text("spare"),
            ],
            "is the server still up?",
        )
    })
    .await
    .unwrap();
    assert!(
        second[0].contains("(still running)") && second[0].contains("- job 1: `echo"),
        "the reopen note says job 1 still runs: {}",
        second[0]
    );
    assert!(
        !second[0].contains("(not running)"),
        "nothing was stopped: {}",
        second[0]
    );
    let peek = &second[1];
    assert!(peek.contains("while-away"), "its output went on: {peek}");
    assert!(
        second[2].contains("Job 1 (`echo"),
        "the kill from the new process worked: {}",
        second[2]
    );
    assert!(wait_dead(pid), "the kill reached the sandbox");
    reap.0 = None;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_ended_while_no_process_served_the_session_is_noticed_once() {
    let dir = tempfile::tempdir().unwrap();
    let exec = ExecMock::start(dir.path()).await;
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let url = exec.url.clone();
    {
        let (session_file, url) = (session_file.clone(), url.clone());
        tokio::task::spawn_blocking(move || {
            serve_once(
                &session_file,
                &url,
                vec![
                    tool(
                        "t1",
                        "bash",
                        json!({
                            "command": "sleep 3; echo built-while-away",
                            "run_in_background": true,
                        }),
                    ),
                    turn_text("building"),
                    turn_text("ok"),
                    turn_text("spare"),
                ],
                "build it",
            )
        })
        .await
        .unwrap();
    }
    std::thread::sleep(Duration::from_secs(4));
    let second = tokio::task::spawn_blocking(move || {
        serve_once(
            &session_file,
            &url,
            vec![
                turn_text("noted"),
                turn_text("noted"),
                turn_text("noted"),
                turn_text("noted"),
            ],
            "how did the build go?",
        )
    })
    .await
    .unwrap();
    let last = second.last().unwrap();
    let notices = last.matches("[job 1 exited with code 0").count();
    assert_eq!(
        notices, 1,
        "noticed exactly once in the conversation: {last}"
    );
    assert!(last.contains("built-while-away"), "{last}");
    assert!(
        !last.contains("(not running)"),
        "an ended job is not reported stopped: {last}"
    );
}
