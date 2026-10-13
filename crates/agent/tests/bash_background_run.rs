//! `run` e2e: a `run_in_background` job lives until the run ends — one reminder, then the run's end
//! stops it (recorded with its kind) — and a `timeout_ms` returns the call while the command keeps
//! running to its exit, noticed on a later turn. Nothing is killed on a timer.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn bash_turn(id: &str, input: Value) -> String {
    turn_tool_use(id, "bash", &input.to_string())
}

/// Drive `run --json`; return (event lines, raw request bodies, exit ok).
fn run_json(turns: Vec<String>) -> (Vec<Value>, Vec<String>, bool) {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(turns);
    let output = run_cmd(BIN)
        .args([
            "run",
            "do the work",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "10",
            "--no-session-persistence",
            "--json",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");
    let events = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let bodies = bodies.lock().unwrap().clone();
    (events, bodies, output.status.success())
}

fn alive(pid: u32) -> bool {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let state = stat
        .rsplit_once(')')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or("");
    !stat.is_empty() && state != "Z" && state != "X"
}

fn wait_dead(pid: u32) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if !alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

fn tool_result<'a>(events: &'a [Value], id: &str) -> &'a str {
    events
        .iter()
        .find(|e| e["kind"] == "tool_end" && e["id"] == id)
        .and_then(|e| e["result"].as_str())
        .unwrap_or_else(|| panic!("no result for {id}: {events:?}"))
}

#[test]
fn a_background_job_is_reminded_once_then_stopped_when_the_run_ends() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (events, bodies, ok) = run_json(vec![
        bash_turn(
            "t1",
            json!({
                "command": format!(
                    "echo serving; sh -c 'echo $$ > {}; exec sleep 93'",
                    pidfile.display()
                ),
                "run_in_background": true
            }),
        ),
        turn_text("it is up"),
        turn_text("ending anyway"),
    ]);
    assert!(ok, "run exits 0");
    let started = tool_result(&events, "t1");
    assert!(started.contains("serving"), "{started}");
    assert!(
        started.contains(
            "Running in the background as job 1 (it is stopped for you when the run ends)."
        ),
        "{started}"
    );
    assert_eq!(bodies.len(), 3, "one reminder turn, then the run ends");
    let reminded = &bodies[2];
    assert!(
        reminded.contains("started in the background)"),
        "{reminded}"
    );
    assert!(
        reminded
            .contains("If you end your turn now, job 1 is stopped for you (no need to kill it)."),
        "{reminded}"
    );
    let stopped = events
        .iter()
        .find(|e| e["kind"] == "bash_jobs_stopped")
        .expect("the stopped job is recorded");
    assert_eq!(stopped["jobs"][0]["job"], 1);
    assert_eq!(stopped["jobs"][0]["kind"], "background");
    let pid: u32 = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(wait_dead(pid), "the run's end stops a background job");
}

#[test]
fn after_the_one_reminder_a_peek_and_a_new_job_do_not_earn_another() {
    // The double fire the background A/B found (dev-server): reminded, the model waited on the
    // server, ended again, and was reminded again. Once per run: the second stop ends it, and the
    // run's end stops both jobs.
    let (events, bodies, ok) = run_json(vec![
        bash_turn(
            "t1",
            json!({ "command": "echo serving; sleep 94", "run_in_background": true }),
        ),
        turn_text("it is up"),
        // The reminder's reply: peek at the server, then start another job.
        turn_tool_use(
            "t2",
            "bash_job",
            &json!({ "job": 1, "kill": false, "timeout_ms": 0 }).to_string(),
        ),
        bash_turn(
            "t3",
            json!({ "command": "echo watching; sleep 95", "run_in_background": true }),
        ),
        turn_text("done"),
    ]);
    assert!(ok, "run exits 0");
    let peek = tool_result(&events, "t2");
    assert!(peek.starts_with("Job 1 is still running ("), "{peek}");
    assert_eq!(
        bodies.len(),
        5,
        "one reminder turn; the second stop ends the run"
    );
    assert_eq!(
        bodies
            .iter()
            .filter(|b| b.contains("[background jobs]"))
            .count(),
        bodies.len() - 2,
        "the reminder is in every request after it, and only one was sent"
    );
    assert_eq!(
        bodies[4]
            .matches("You are ending your turn with jobs")
            .count(),
        1
    );
    let stopped = events
        .iter()
        .find(|e| e["kind"] == "bash_jobs_stopped")
        .expect("the stopped jobs are recorded");
    let ids: Vec<u64> = stopped["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["job"].as_u64().unwrap())
        .collect();
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn a_timeout_returns_the_call_and_the_command_runs_on_to_its_exit() {
    let (events, bodies, ok) = run_json(vec![
        bash_turn(
            "t1",
            json!({ "command": "echo first; sleep 1; echo slow-finished", "timeout_ms": 300 }),
        ),
        // The command finishes while this one runs.
        bash_turn("t2", json!({ "command": "sleep 2.5; echo fg" })),
        turn_text("done"),
    ]);
    assert!(ok, "run exits 0");
    let first = tool_result(&events, "t1");
    assert!(
        first.contains("Returned after your timeout_ms of 300 ms."),
        "{first}"
    );
    assert!(!first.contains("slow-finished"), "{first}");
    assert_eq!(bodies.len(), 3, "no reminder: nothing ran at the stop");
    let last = &bodies[2];
    assert_eq!(
        last.matches("[job 1 exited with code 0 after").count(),
        1,
        "{last}"
    );
    assert!(last.contains("slow-finished"), "it ran to its exit: {last}");
    assert!(
        !events.iter().any(|e| e["kind"] == "bash_jobs_stopped"),
        "nothing was left to stop"
    );
}
