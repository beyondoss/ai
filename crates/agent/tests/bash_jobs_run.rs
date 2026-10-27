//! `run` e2e: `bash` jobs through the real binary and a scripted model — a job that ends mid-run is
//! noticed on the next tool-results turn (once), ending with a job still running gets one reminder
//! and then stops it (recorded in the `--json` stream), and no process a command detached outlives
//! the agent, whether the run ends or is signalled.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::time::{Duration, Instant};

use common::{SpawnGuarded, run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_agent");

fn bash_turn(id: &str, command: &str) -> String {
    turn_tool_use(id, "bash", &json!({ "command": command }).to_string())
}

fn run_args(base: &str) -> Vec<String> {
    [
        "run",
        "do the work",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--max-steps",
        "10",
        "--no-session-persistence",
        "--json",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Drive `run --json`; return (event lines, raw request bodies, exit ok).
fn run_json(turns: Vec<String>) -> (Vec<Value>, Vec<String>, bool) {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(turns);
    let output = run_cmd(BIN)
        .args(run_args(&base))
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

#[test]
fn a_job_ending_mid_run_is_noticed_on_the_next_tool_results_turn_once() {
    let (events, bodies, ok) = run_json(vec![
        bash_turn("t1", "(sleep 0.5; echo bg-finished) & echo launched"),
        // The job ends while this one runs.
        bash_turn("t2", "sleep 2; echo foreground-done"),
        turn_text("done"),
    ]);
    assert!(ok, "run exits 0");
    assert_eq!(
        bodies.len(),
        3,
        "no reminder: nothing was running at the stop"
    );
    // The adoption: the first call returned at its shell's exit, with the job number.
    let first = events
        .iter()
        .find(|e| e["kind"] == "tool_end" && e["id"] == "t1")
        .expect("t1's result");
    let first = first["result"].as_str().unwrap();
    assert!(first.contains("tracked as job 1"), "{first}");
    // The notice rides the turn after the second call, once.
    let last = &bodies[2];
    assert_eq!(last.matches("[job 1 ended after").count(), 1, "{last}");
    assert!(last.contains("bg-finished"), "{last}");
    assert!(!bodies[1].contains("[job 1 ended"), "not before it ended");
    let background: Vec<&Value> = events
        .iter()
        .filter(|e| e["kind"] == "background")
        .collect();
    assert_eq!(background.len(), 1, "{background:?}");
    assert_eq!(background[0]["reason"], "finished");
    assert!(
        !events.iter().any(|e| e["kind"] == "bash_jobs_stopped"),
        "nothing was left to stop"
    );
}

#[test]
fn ending_with_a_running_job_reminds_once_then_stops_it_and_records_which() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let (events, bodies, ok) = run_json(vec![
        bash_turn(
            "t1",
            &format!(
                "setsid sh -c 'echo $$ > {}; exec sleep 85' > /dev/null 2>&1 & echo up",
                pidfile.display()
            ),
        ),
        turn_text("done"),
        turn_text("ending anyway"),
    ]);
    assert!(ok, "run exits 0");
    assert_eq!(bodies.len(), 3, "one reminder turn, then the run ends");
    let reminded = &bodies[2];
    assert!(reminded.contains("[background jobs]"), "{reminded}");
    assert!(reminded.contains("- job 1: `setsid sh -c"), "{reminded}");
    assert!(
        reminded
            .contains("If you end your turn now, job 1 is stopped for you (no need to kill it)."),
        "{reminded}"
    );
    let reasons: Vec<&Value> = events
        .iter()
        .filter(|e| e["kind"] == "background")
        .map(|e| &e["reason"])
        .collect();
    assert_eq!(reasons, vec![&json!("still_running")]);
    let stopped = events
        .iter()
        .find(|e| e["kind"] == "bash_jobs_stopped")
        .expect("the stopped jobs are recorded");
    assert_eq!(stopped["jobs"][0]["job"], 1);
    assert!(
        stopped["jobs"][0]["command"]
            .as_str()
            .unwrap()
            .starts_with("setsid sh -c")
    );
    // The setsid'd process left the job's process group; it is stopped all the same.
    let pid = read_pid(&pidfile);
    assert!(wait_dead(pid), "the run's end stops a setsid'd job");
}

#[test]
fn a_signalled_run_stops_every_detached_process_on_its_way_out() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let files: Vec<std::path::PathBuf> = ["nohup", "timeout", "double"]
        .iter()
        .map(|n| tmp.path().join(n))
        .collect();
    let child = |f: &std::path::Path, secs: u32| {
        format!("sh -c 'echo $$ > {}; exec sleep {secs}'", f.display())
    };
    let script = format!(
        "nohup {} > /dev/null 2>&1 &\ntimeout 120 {} > /dev/null 2>&1 &\n( ( {} > /dev/null 2>&1 & ) & )\necho launched",
        child(&files[0], 86),
        child(&files[1], 87),
        child(&files[2], 88),
    );
    let (base, _bodies) = spawn_model_server(vec![
        bash_turn("t1", &script),
        // Still running when the signal lands.
        bash_turn("t2", "sleep 30"),
        turn_text("unreachable"),
    ]);
    let mut agent = run_cmd(BIN)
        .args(run_args(&base))
        .current_dir(dir.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn_guarded();
    let pids: Vec<u32> = files.iter().map(|f| read_pid(f)).collect();
    for &pid in &pids {
        assert!(alive(pid));
    }
    std::process::Command::new("kill")
        .args(["-TERM", &agent.id().to_string()])
        .status()
        .unwrap();
    let _ = agent.wait();
    for pid in pids {
        assert!(wait_dead(pid), "{pid} outlived the agent");
    }
}
