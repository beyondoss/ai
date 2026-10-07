//! The stdio MCP transport (`tools::mcp_stdio`) — used for **every** stdio server, not only ones
//! with MCP Events — keeps rmcp's behaviour where it matters: a stdout line that is not UTF-8 is
//! skipped rather than fatal, and a server whose connection is dropped (here: reaped idle) gets
//! its grace window after stdin closes before it is killed. Real `serve`, real fixture process.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_events_fixture::{Frames, eventually, send, stdio_server, write_settings};
use common::{BIN, SpawnGuarded, serve_cmd, spawn_model_server_routed, turn_text};
use serde_json::json;

fn start(
    home: &std::path::Path,
    extra_env: serde_json::Value,
    idle_secs: &str,
) -> (common::ChildGuard, std::process::ChildStdin, Frames) {
    write_settings(
        home,
        json!([stdio_server(
            "tools",
            &home.join("control"),
            extra_env,
            json!([])
        )]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let mut cmd = serve_cmd(BIN, &base, &home.join("s.jsonl").to_string_lossy());
    cmd.env("HOME", home)
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", idle_secs);
    let mut child = cmd.spawn_guarded();
    let stdin = child.stdin.take().unwrap();
    let frames = Frames::new(&mut child, None);
    (child, stdin, frames)
}

#[test]
fn a_non_utf8_stdout_line_is_skipped_not_fatal() {
    let home = tempfile::tempdir().unwrap();
    let (_child, mut stdin, mut frames) = start(
        home.path(),
        json!({ "MCP_FIXTURE_GARBAGE_STDOUT": "1" }),
        "0",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    let m = frames.response("m");
    assert!(
        m["data"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t == "mcp__tools__echo"),
        "the server survived its garbage line and its tools are there: {m:#}"
    );
}

#[test]
fn a_reaped_server_gets_its_grace_window_after_stdin_closes() {
    let home = tempfile::tempdir().unwrap();
    let marker = home.path().join("exited-cleanly");
    let (_child, mut stdin, mut frames) = start(
        home.path(),
        json!({ "MCP_FIXTURE_EXIT_MARKER": marker.to_string_lossy() }),
        "1",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    frames.response("m");
    // Idle for a second → reaped: stdin closes, the server takes 500 ms to "clean up" and writes
    // the marker. Killed on the spot, it never would.
    eventually(Duration::from_secs(15), "the server's clean exit", || {
        marker.exists().then_some(())
    });
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "clean exit");
}

/// Whether `pid` is still a live process (a zombie counts as gone).
fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .map(|s| {
            s.rsplit_once(") ")
                .is_some_and(|(_, rest)| !rest.starts_with('Z'))
        })
        .unwrap_or(false)
}

/// `serve` exiting — stdin EOF or SIGTERM — gives each stdio server its grace window (it sees stdin
/// close and exits cleanly) and then sweeps its process group, so a grandchild it double-forked
/// away does not outlive the agent.
fn exit_sweeps_the_servers_group(sigterm: bool) {
    let home = tempfile::tempdir().unwrap();
    let marker = home.path().join("exited-cleanly");
    let pidfile = home.path().join("orphan.pid");
    let (mut child, mut stdin, mut frames) = start(
        home.path(),
        json!({
            "MCP_FIXTURE_EXIT_MARKER": marker.to_string_lossy(),
            "MCP_FIXTURE_ORPHAN_PIDFILE": pidfile.to_string_lossy(),
        }),
        "0",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    frames.response("m");
    let orphan: u32 = eventually(Duration::from_secs(10), "the orphan's pid", || {
        std::fs::read_to_string(&pidfile).ok()?.trim().parse().ok()
    });
    assert!(alive(orphan), "the grandchild runs before serve exits");

    if sigterm {
        common::mcp_events_fixture::sigterm_and_wait(&mut child);
        drop(stdin);
    } else {
        drop(stdin);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while child.try_wait().unwrap().is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "serve did not exit on EOF"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    // Both are done by the time serve has exited: nothing is left to finish them afterwards.
    assert!(
        !alive(orphan),
        "serve exited and left its stdio server's grandchild {orphan} running"
    );
    assert_eq!(
        std::fs::read_to_string(&marker).ok().as_deref(),
        Some("clean exit"),
        "the server was not given its grace window to exit on its own"
    );
}

#[test]
fn serve_exiting_on_eof_sweeps_its_stdio_servers_groups_after_their_grace() {
    exit_sweeps_the_servers_group(false);
}

#[test]
fn serve_exiting_on_sigterm_sweeps_its_stdio_servers_groups_after_their_grace() {
    exit_sweeps_the_servers_group(true);
}

/// `agent run` ends the same way: when it exits, each stdio server has had its grace and its
/// process group is swept — a grandchild it double-forked does not outlive the run.
#[test]
fn run_exiting_sweeps_its_stdio_servers_groups_after_their_grace() {
    let home = tempfile::tempdir().unwrap();
    let marker = home.path().join("exited-cleanly");
    let pidfile = home.path().join("orphan.pid");
    write_settings(
        home.path(),
        json!([stdio_server(
            "tools",
            &home.path().join("control"),
            json!({
                "MCP_FIXTURE_EXIT_MARKER": marker.to_string_lossy(),
                "MCP_FIXTURE_ORPHAN_PIDFILE": pidfile.to_string_lossy(),
            }),
            json!([])
        )]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("all done"));
    let mut output = common::run_cmd(BIN)
        .args([
            "run",
            "say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .current_dir(home.path())
        // Files, not pipes: a leaked grandchild inherits them, and waiting on a pipe it holds open
        // would turn the regression this catches into a hang.
        .stdin(std::process::Stdio::null())
        .stdout(std::fs::File::create(home.path().join("run.stdout")).unwrap())
        .stderr(std::fs::File::create(home.path().join("run.stderr")).unwrap())
        .spawn_guarded();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = output.try_wait().unwrap() {
            break status;
        }
        assert!(std::time::Instant::now() < deadline, "`run` did not exit");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(
        status.success(),
        "{}",
        std::fs::read_to_string(home.path().join("run.stderr")).unwrap_or_default()
    );
    let orphan: u32 = std::fs::read_to_string(&pidfile)
        .expect("the server started and recorded its grandchild")
        .trim()
        .parse()
        .unwrap();
    assert!(
        !alive(orphan),
        "`run` exited and left its stdio server's grandchild {orphan} running"
    );
    assert_eq!(
        std::fs::read_to_string(&marker).ok().as_deref(),
        Some("clean exit"),
        "the server was not given its grace window to exit on its own"
    );
}

/// A `run` stopped by SIGTERM mid-turn exits from deep inside the turn, with its tools — and so its
/// MCP connections — still held: nothing is dropped. Its stdio servers are still retired
/// (`mcp_stdio::retire_all`) and swept before the process goes.
#[test]
fn a_run_stopped_mid_turn_still_sweeps_its_stdio_servers() {
    let home = tempfile::tempdir().unwrap();
    let marker = home.path().join("exited-cleanly");
    let pidfile = home.path().join("orphan.pid");
    write_settings(
        home.path(),
        json!([stdio_server(
            "tools",
            &home.path().join("control"),
            json!({
                "MCP_FIXTURE_EXIT_MARKER": marker.to_string_lossy(),
                "MCP_FIXTURE_ORPHAN_PIDFILE": pidfile.to_string_lossy(),
            }),
            json!([])
        )]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("all done"));
    let mut child = common::run_cmd(BIN)
        .args([
            "run",
            &beyond_ai_test_support::stall_prompt(60_000),
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .current_dir(home.path())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn_guarded();
    let orphan: u32 = eventually(Duration::from_secs(20), "the orphan's pid", || {
        std::fs::read_to_string(&pidfile).ok()?.trim().parse().ok()
    });
    // Mid-turn: the model is stalling.
    std::thread::sleep(Duration::from_millis(1000));
    common::mcp_events_fixture::sigterm_and_wait(&mut child);
    assert!(
        !alive(orphan),
        "a cancelled `run` left its stdio server's grandchild {orphan} running"
    );
    assert_eq!(
        std::fs::read_to_string(&marker).ok().as_deref(),
        Some("clean exit"),
        "the server was not given its grace window to exit on its own"
    );
}

/// A configuration error `run` only finds after its MCP servers are up (here an unreadable
/// `--output-schema`) exits through the same cleanup: no grandchild is left behind.
#[test]
fn a_run_refused_for_bad_configuration_after_connecting_still_sweeps() {
    let home = tempfile::tempdir().unwrap();
    let pidfile = home.path().join("orphan.pid");
    write_settings(
        home.path(),
        json!([stdio_server(
            "tools",
            &home.path().join("control"),
            json!({ "MCP_FIXTURE_ORPHAN_PIDFILE": pidfile.to_string_lossy() }),
            json!([])
        )]),
    );
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("all done"));
    let mut child = common::run_cmd(BIN)
        .args([
            "run",
            "say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--output-schema",
            "{ this is not json",
        ])
        .env("HOME", home.path())
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
        .current_dir(home.path())
        .stdin(std::process::Stdio::null())
        .stdout(std::fs::File::create(home.path().join("run.stdout")).unwrap())
        .stderr(std::fs::File::create(home.path().join("run.stderr")).unwrap())
        .spawn_guarded();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(std::time::Instant::now() < deadline, "`run` did not exit");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(2), "refused as a usage error");
    let orphan: u32 = std::fs::read_to_string(&pidfile)
        .expect("the server was up before the refusal")
        .trim()
        .parse()
        .unwrap();
    assert!(
        !alive(orphan),
        "`run` refused its configuration and left its stdio server's grandchild {orphan} running"
    );
}
