//! `serve` e2e: `bash` jobs outlive the run that started them. An idle session wakes when one ends
//! (a turn whose input is the notice, plus a `background` event saying so); a client prompt racing
//! that wake carries the notice itself and no second turn runs; a compaction carries the job table;
//! and the session's own exit stops what is still running.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::Write;
use std::process::ChildStdin;
use std::time::{Duration, Instant};

use common::{
    ChildGuard, NoFrame, SpawnGuarded, read_until_response, serve_cmd, spawn_model_server,
    turn_text, turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_agent");

fn bash_turn(id: &str, command: &str) -> String {
    turn_tool_use(id, "bash", &json!({ "command": command }).to_string())
}

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

/// End the session the way a client does (stdin closes) and wait for it: its exit path stops every
/// job still running, so nothing a test started outlives it.
fn close(mut child: ChildGuard, stdin: ChildStdin) {
    drop(stdin);
    let _ = child.wait();
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

#[test]
fn an_idle_session_wakes_when_a_job_ends() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        bash_turn("t1", "(sleep 1; echo bg-done) & echo launched"),
        turn_text("started it"),
        // The wake.
        turn_text("noted: it finished"),
    ]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "start it" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true);
    // Ending in serve leaves the job running and its exit wakes the session: no reminder turn.
    {
        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded.len(), 2, "no reminder turn");
        assert!(
            !recorded.iter().any(|b| b.contains("[background jobs]")),
            "an intended job earns no reminder"
        );
    }

    // Nobody sends anything: the job's end starts a turn of its own.
    let woke = read_until_response(&mut stdout, "prompt");
    let event = woke
        .iter()
        .find(|f| f["type"] == "event" && f["event"]["kind"] == "background")
        .expect("a background event identifies the wake");
    assert_eq!(event["event"]["wake"], true);
    assert_eq!(event["event"]["reason"], "finished");
    assert!(
        event["event"]["text"]
            .as_str()
            .unwrap()
            .starts_with("[job 1 ended after")
    );
    assert!(
        woke.iter().any(|f| f["event"]["kind"] == "agent_start"),
        "streamed as a normal turn: {woke:?}"
    );
    assert_eq!(woke.last().unwrap()["success"], true);
    let recorded = bodies.lock().unwrap();
    assert_eq!(recorded.len(), 3);
    assert!(
        recorded[2].contains("[job 1 ended after"),
        "the notice is the turn's input"
    );
    assert!(recorded[2].contains("bg-done"));
    drop(recorded);
    close(child, stdin);
}

#[test]
fn a_client_prompt_racing_a_wake_carries_the_notice_and_no_extra_turn_runs() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        bash_turn("t1", "(sleep 1; echo bg-done) & echo launched"),
        turn_text("started it"),
        // The client's next prompt is in flight when the job ends.
        bash_turn("t2", "sleep 2; echo fg"),
        turn_text("saw both"),
    ]);
    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "start it" }),
    );
    assert_eq!(
        read_until_response(&mut stdout, "prompt").last().unwrap()["success"],
        true
    );
    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p2", "message": "now this" }),
    );
    let second = read_until_response(&mut stdout, "prompt");
    assert_eq!(second.last().unwrap()["id"], "p2");
    assert_eq!(second.last().unwrap()["success"], true);
    // The run that was in flight delivered the notice, so the wake finds nothing to take.
    let deadline = Instant::now() + Duration::from_millis(1500);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match stdout.next_frame(left) {
            Ok(f) => assert!(
                f["event"]["kind"] != "background" && f["command"] != "prompt",
                "no wake turn after the notice was delivered: {f}"
            ),
            Err(NoFrame::TimedOut) => break,
            Err(NoFrame::Closed) => panic!("serve exited"),
        }
    }
    let recorded = bodies.lock().unwrap();
    assert_eq!(recorded.len(), 4, "no extra model call");
    assert_eq!(recorded[3].matches("[job 1 ended after").count(), 1);
    drop(recorded);
    close(child, stdin);
}

#[test]
fn a_compaction_carries_the_job_table_and_the_session_exit_stops_the_job() {
    let dir = tempfile::tempdir().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("pid");
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, bodies) = spawn_model_server(vec![
        bash_turn(
            "t1",
            &format!(
                "nohup sh -c 'echo $$ > {}; exec sleep 89' > /dev/null 2>&1 & echo up",
                pidfile.display()
            ),
        ),
        turn_text("the server is up"),
        // The summarization call `compact` makes; it never mentions the job.
        turn_text("## Goal\nStarted a server."),
        turn_text("it is still up"),
        // The post-compaction memory reminder continues the run once.
        turn_text("still up"),
    ]);
    let mut child = serve_cmd(BIN, &base, &session_file)
        .args(["--compaction-keep-recent-tokens", "1"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = common::child_frames(&mut child);

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "start the server" }),
    );
    assert_eq!(
        read_until_response(&mut stdout, "prompt").last().unwrap()["success"],
        true
    );
    send(&mut stdin, json!({ "type": "compact", "id": "c1" }));
    let resp = read_until_response(&mut stdout, "compact");
    let resp = resp.last().unwrap();
    assert_eq!(resp["data"]["compacted"], true, "{resp}");
    // The table rides the summary in the next request, never in the persisted summary: after a
    // restart those jobs are gone (see `bash_jobs_serve_carry.rs`).
    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "is it up?" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:?}");
    let body = bodies.lock().unwrap().last().unwrap().clone();
    assert!(
        body.contains("<background-jobs>"),
        "the summary carries the table"
    );
    assert!(body.contains("do not start these again"));
    assert!(
        body.contains("- job 1: `nohup sh -c"),
        "the running job is listed"
    );
    let raw = std::fs::read_to_string(&session_file).unwrap();
    assert!(
        raw.contains("Started a server."),
        "the summary is persisted"
    );
    assert!(!raw.contains("<background-jobs>"), "the table is not");

    let pid = read_pid(&pidfile);
    assert!(alive(pid));
    close(child, stdin);
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!alive(pid), "the session's exit stops a nohup'd job");
}
