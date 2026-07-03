//! `run` e2e: SIGTERM/SIGINT signal handling. Pi-parity audit H2: `run` registered no signal
//! handler at all, so a shutdown request hit Rust's default kill-immediately disposition — no
//! destructors run, so a `GroupKillGuard` never got to reap a still-running bash child's process
//! group. Reusing `serve`'s `ShutdownSignal` (see `main.rs::run_task`) must fix both halves: `run`
//! exits promptly instead of hanging on its own model/tool timeouts, and it must not orphan any
//! backgrounded grandchild process the tool call itself spawned.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Read;
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{run_cmd, spawn_model_server, turn_tool_use};
use serde_json::json;

/// Spawns `run` against a mock model server whose one turn calls `bash` with `script`, sends
/// `signal` (`"TERM"` or `"INT"`) ~500ms later (enough for the tool call to actually start), and
/// returns the child's `(exit_status, stderr)` once it exits — or panics if it doesn't within 10s.
fn run_and_signal(
    dir: &std::path::Path,
    script: &str,
    signal: &str,
) -> (std::process::ExitStatus, String) {
    let turn1 = turn_tool_use("toolu_1", "bash", &json!({ "command": script }).to_string());
    let (base, _bodies) = spawn_model_server(vec![turn1]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = run_cmd(bin)
        .args([
            "run",
            "run the background sleeper",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
        ])
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Give the bash tool time to actually start (model response received, tool dispatched, shell
    // spawned) before sending the signal — otherwise this would race the tool call's own startup.
    std::thread::sleep(Duration::from_millis(500));

    let pid = child.id();
    let status = std::process::Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(pid.to_string())
        .status()
        .unwrap();
    assert!(
        status.success(),
        "failed to send SIG{signal} to the run process"
    );

    // The process must actually exit promptly (not hang for the backgrounded `sleep 30`) — proves
    // the signal was handled, not ignored (Rust's un-handled default disposition would also exit
    // promptly here, but by *terminating* rather than by unwinding through `GroupKillGuard`, so this
    // check alone doesn't distinguish the two — the marker check the caller does afterward does).
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "run did not exit within 10s of SIG{signal} — signal handling not wired up"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    (status, stderr)
}

#[test]
fn sigterm_during_a_bash_tool_call_kills_the_whole_process_group_not_just_run_itself() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("leaked");
    // Backgrounds a sleeper that writes `marker` after 1s, then the foreground shell itself sleeps
    // long enough that — without a working process-group kill — the command would still be running
    // when the signal arrives, so this test actually exercises the kill path rather than racing a
    // command that was about to finish anyway.
    let script = format!("( sleep 1; touch {} ) & sleep 30", marker.display());
    let (status, stderr) = run_and_signal(dir.path(), &script, "TERM");

    // A cancelled turn is a clean, expected stop — printed as `[cancelled]` (matching the
    // `[refused]` precedent elsewhere in `run_task`), not a raw `Error: Cancelled` Debug dump of
    // the bare enum variant, and a distinct non-zero exit a script/CI caller can key off of.
    assert!(
        stderr.contains("[cancelled]"),
        "expected a clean `[cancelled]` message, got stderr:\n{stderr}"
    );
    assert_eq!(status.code(), Some(1), "got stderr:\n{stderr}");

    // Wait past when the backgrounded sleeper would have written the marker (started at ~0s, fires
    // at ~1s) — it must not exist: the whole process group, not just the direct bash child, must
    // have been killed.
    std::thread::sleep(Duration::from_millis(1500));
    assert!(
        !marker.exists(),
        "backgrounded grandchild survived SIGTERM to `run` — process group not killed"
    );
}

#[test]
fn sigint_during_a_bash_tool_call_kills_the_whole_process_group_too() {
    // Same fixture as the SIGTERM test, but Ctrl-C (SIGINT) — `ShutdownSignal::wait` races
    // `tokio::signal::ctrl_c()` alongside `SIGTERM` (see `serve.rs`), so both must behave alike.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("leaked");
    let script = format!("( sleep 1; touch {} ) & sleep 30", marker.display());
    let (status, stderr) = run_and_signal(dir.path(), &script, "INT");

    assert!(
        stderr.contains("[cancelled]"),
        "expected a clean `[cancelled]` message, got stderr:\n{stderr}"
    );
    assert_eq!(status.code(), Some(1), "got stderr:\n{stderr}");

    std::thread::sleep(Duration::from_millis(1500));
    assert!(
        !marker.exists(),
        "backgrounded grandchild survived SIGINT to `run` — process group not killed"
    );
}

#[test]
#[cfg(unix)]
fn sighup_during_a_bash_tool_call_kills_the_whole_process_group_too() {
    // Pi-parity fix: SIGHUP had no handler at all — its default disposition is the same immediate
    // termination as an unhandled SIGTERM, which would skip `GroupKillGuard` and orphan a backgrounded
    // grandchild. Same fixture as the SIGTERM/SIGINT tests above — `ShutdownSignal::wait` now races
    // `SignalKind::hangup()` alongside them (see `serve.rs`), so all three must behave alike.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("leaked");
    let script = format!("( sleep 1; touch {} ) & sleep 30", marker.display());
    let (status, stderr) = run_and_signal(dir.path(), &script, "HUP");

    assert!(
        stderr.contains("[cancelled]"),
        "expected a clean `[cancelled]` message, got stderr:\n{stderr}"
    );
    assert_eq!(status.code(), Some(1), "got stderr:\n{stderr}");

    std::thread::sleep(Duration::from_millis(1500));
    assert!(
        !marker.exists(),
        "backgrounded grandchild survived SIGHUP to `run` — process group not killed"
    );
}
