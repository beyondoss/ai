//! `run` e2e: stdin is read only when asked for. `run "task"` used to wait for stdin to close whenever
//! it wasn't a terminal, so under `ssh host agent run …` without `-t`, `docker exec -i`, or a parent
//! holding a pipe open, it never started; in a `while read …; do agent run …; done` loop it swallowed
//! the rest of the list as its prompt. Stdin is now read only when no message argument is given, or
//! when the message is `-`.
#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::{Read as _, Write as _};
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{SpawnGuarded, run_cmd, spawn_model_server, turn_text};

fn spawn(base: &str, args: &[&str], dir: &std::path::Path) -> common::ChildGuard {
    run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .arg("run")
        .args(args)
        .args([
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded()
}

fn wait(child: &mut common::ChildGuard, within: Duration) -> Option<std::process::ExitStatus> {
    let t0 = Instant::now();
    while t0.elapsed() < within {
        if let Some(s) = child.try_wait().unwrap() {
            return Some(s);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

#[test]
fn a_message_argument_starts_at_once_with_an_open_stdin_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let mut child = spawn(&base, &["hello"], dir.path());
    // Keep the write end open and never write: the parent holds the pipe, like ssh without -t.
    let _stdin = child.stdin.take().unwrap();
    let status = wait(&mut child, Duration::from_secs(20));
    assert!(
        status.is_some_and(|s| s.success()),
        "run with a message must not wait for stdin to close (status {status:?})"
    );
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1);
    let body = common::body_json(&bodies[0]);
    assert_eq!(body["messages"][0]["content"][0]["text"], "hello");
}

#[test]
fn a_dash_message_reads_the_prompt_from_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let mut child = spawn(&base, &["-"], dir.path());
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(b"FROM-STDIN\n").unwrap();
    drop(stdin);
    let status = wait(&mut child, Duration::from_secs(20));
    let mut err = String::new();
    child.stderr.take().unwrap().read_to_string(&mut err).ok();
    assert!(status.is_some_and(|s| s.success()), "{status:?}: {err}");
    let bodies = bodies.lock().unwrap();
    let body = common::body_json(&bodies[0]);
    assert_eq!(body["messages"][0]["content"][0]["text"], "FROM-STDIN");
}

#[test]
fn no_message_argument_still_reads_piped_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let mut child = spawn(&base, &[], dir.path());
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(b"PIPED-PROMPT").unwrap();
    drop(stdin);
    let status = wait(&mut child, Duration::from_secs(20));
    assert!(status.is_some_and(|s| s.success()), "{status:?}");
    let bodies = bodies.lock().unwrap();
    let body = common::body_json(&bodies[0]);
    assert_eq!(body["messages"][0]["content"][0]["text"], "PIPED-PROMPT");
}
