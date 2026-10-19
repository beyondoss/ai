//! `run` e2e: SIGTERM/SIGINT signal handling. Pi-parity audit H2: `run` registered no signal
//! handler at all, so a shutdown request hit Rust's default kill-immediately disposition — no
//! destructors run, so a `GroupKillGuard` never got to reap a still-running bash child's process
//! group. Reusing `serve`'s `ShutdownSignal` (see `commands/run.rs::run_task`) must fix both halves: `run`
//! exits promptly instead of hanging on its own model/tool timeouts, and it must not orphan any
//! backgrounded grandchild process the tool call itself spawned.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::Read;
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{SpawnGuarded, run_cmd, spawn_model_server, turn_tool_use};
use serde_json::json;

/// How long the backgrounded grandchild waits before writing its marker. Long enough that a
/// signal sent as soon as the grandchild is up (even on a loaded host) lands well before it fires.
const GRANDCHILD_DELAY_SECS: u64 = 3;

/// Spawns `run` against a mock model server whose one turn calls `bash` with a script that
/// backgrounds a grandchild which writes `dir/leaked` after [`GRANDCHILD_DELAY_SECS`] while the
/// foreground shell sleeps. Waits (bounded) until the grandchild has recorded its pid in
/// `dir/ready`, so the tool call is provably running, then sends `signal` (`"TERM"`, `"INT"` or
/// `"HUP"`) and returns the child's `(exit_status, stderr)` once it exits — or panics if it
/// doesn't within 10s.
fn run_and_signal(dir: &std::path::Path, signal: &str) -> (std::process::ExitStatus, String) {
    let marker = dir.join("leaked");
    let ready = dir.join("ready");
    // The pid is written to a temp name then renamed so a reader never sees a partial file.
    let script = format!(
        "( echo $BASHPID > {tmp} && mv {tmp} {ready}; sleep {GRANDCHILD_DELAY_SECS}; touch {marker} ) & sleep 30",
        tmp = dir.join("ready.tmp").display(),
        ready = ready.display(),
        marker = marker.display(),
    );
    let turn1 = turn_tool_use(
        "toolu_1",
        "bash",
        &json!({ "command": &script }).to_string(),
    );
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
            "--no-session-persistence",
        ])
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn_guarded();

    // Wait for the grandchild to exist (model response received, tool dispatched, shell spawned)
    // before signalling — a fixed sleep can lose that race under load, and then the "grandchild
    // was killed" check passes without a grandchild ever having been started.
    let started = Instant::now();
    let grandchild_pid = loop {
        if let Ok(text) = std::fs::read_to_string(&ready)
            && let Ok(pid) = text.trim().parse::<u32>()
        {
            break pid;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the bash tool never started its grandchild"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(
        std::path::Path::new(&format!("/proc/{grandchild_pid}")).exists(),
        "grandchild {grandchild_pid} was not running when the signal was about to be sent"
    );

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
    let (status, stderr) = run_and_signal(dir.path(), "TERM");

    // A cancelled turn is a clean, expected stop — printed as `[cancelled]` (matching the
    // `[refused]` precedent elsewhere in `run_task`), not a raw `Error: Cancelled` Debug dump of
    // the bare enum variant, and a distinct non-zero exit a script/CI caller can key off of.
    assert!(
        stderr.contains("[cancelled]"),
        "expected a clean `[cancelled]` message, got stderr:\n{stderr}"
    );
    // Task #41 (pi-parity fix): the exit code now names *which* signal actually caused the
    // cancellation (POSIX `128 + signal`, matching pi's own `rpc-mode.ts`/`print-mode.ts` — 143 for
    // SIGTERM) rather than the same undifferentiated `1` every shutdown-signal cancellation used to
    // get regardless of cause.
    assert_eq!(status.code(), Some(143), "got stderr:\n{stderr}");

    // Wait past when the backgrounded sleeper would have written the marker (armed once the grandchild is up, fires
    // after GRANDCHILD_DELAY_SECS) — it must not exist: the whole process group, not just the direct bash child, must
    // have been killed.
    std::thread::sleep(Duration::from_secs(GRANDCHILD_DELAY_SECS) + Duration::from_millis(500));
    assert!(
        !marker.exists(),
        "backgrounded grandchild survived SIGTERM to `run` — process group not killed"
    );
}

#[test]
fn sigint_during_a_bash_tool_call_kills_the_whole_process_group_too() {
    // Same fixture as the SIGTERM test, but Ctrl-C (SIGINT) — `ShutdownSignal::wait` races
    // `tokio::signal::ctrl_c()` alongside `SIGTERM` (see `serve/`), so both must behave alike.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("leaked");
    let (status, stderr) = run_and_signal(dir.path(), "INT");

    assert!(
        stderr.contains("[cancelled]"),
        "expected a clean `[cancelled]` message, got stderr:\n{stderr}"
    );
    // Task #41 (pi-parity fix): `130` (POSIX `128 + 2`) — this crate's own extension of pi's
    // 143/129 SIGTERM/SIGHUP convention to Ctrl-C, since `ShutdownSignal` already treats it as an
    // equal graceful-shutdown trigger.
    assert_eq!(status.code(), Some(130), "got stderr:\n{stderr}");

    std::thread::sleep(Duration::from_secs(GRANDCHILD_DELAY_SECS) + Duration::from_millis(500));
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
    // `SignalKind::hangup()` alongside them (see `serve/`), so all three must behave alike.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("leaked");
    let (status, stderr) = run_and_signal(dir.path(), "HUP");

    assert!(
        stderr.contains("[cancelled]"),
        "expected a clean `[cancelled]` message, got stderr:\n{stderr}"
    );
    // Task #41 (pi-parity fix): `129` (POSIX `128 + 1`), matching pi's own `rpc-mode.ts`/
    // `print-mode.ts` SIGHUP code exactly.
    assert_eq!(status.code(), Some(129), "got stderr:\n{stderr}");

    std::thread::sleep(Duration::from_secs(GRANDCHILD_DELAY_SECS) + Duration::from_millis(500));
    assert!(
        !marker.exists(),
        "backgrounded grandchild survived SIGHUP to `run` — process group not killed"
    );
}

/// A second signal ends a run whose first cancellation is stuck in a wait that doesn't watch it:
/// here an MCP skill fetch (`/skill:docs:...` expands before the turn) from a server that takes a
/// minute to answer. The one-shot handler used to make every signal after the first a no-op.
#[test]
fn a_second_signal_exits_a_run_whose_first_cancellation_is_stuck() {
    let env = common::skills_env::Env::new(json!({ "MCP_SKILLS_FIXTURE_SLOW_READ_MS": "60000" }));
    let (base, _bodies) = spawn_model_server(vec![common::turn_text("unused")]);
    let mut child = run_cmd(common::skills_env::BIN)
        .env("HOME", &env.home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .args([
            "run",
            "/skill:docs:git-workflow",
            "--approve-mcp-skills",
            "--no-session-persistence",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(&env.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let started = Instant::now();
    while !env.log().iter().any(|l| l.contains("git-workflow")) {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the skill was never fetched: {:?}",
            env.log()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let term = || {
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status();
    };
    term();
    std::thread::sleep(Duration::from_millis(300));
    term();
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "a second signal did not end the run"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .ok();
    assert_eq!(status.code(), Some(143), "stderr: {stderr}");
    assert!(stderr.contains("second signal"), "stderr: {stderr}");
}
