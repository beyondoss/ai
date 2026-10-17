//! What a stdio MCP server leaves behind, against the real binary: a grandchild that left the
//! server's process group (`setsid`) is re-parented to the agent (a child subreaper), killed with the
//! server — on a reap and on exit — and reaped by the agent rather than left a zombie; and a
//! SIGKILLed agent takes its server down with it (a parent-death signal), even a server that ignores
//! its stdin closing. Linux only: the mechanisms are.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

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

/// `/proc/<pid>/stat`'s state and parent, or `None` once the pid is gone entirely (reaped).
fn stat(pid: u32) -> Option<(char, u32)> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let (_, rest) = s.rsplit_once(") ")?;
    let mut fields = rest.split_whitespace();
    let state = fields.next()?.chars().next()?;
    let ppid = fields.next()?.parse().ok()?;
    Some((state, ppid))
}

/// Whether `pid` is still a live process (a zombie counts as gone).
fn alive(pid: u32) -> bool {
    stat(pid).is_some_and(|(state, _)| state != 'Z' && state != 'X')
}

/// SIGKILLs `pid` on drop, so a regression's survivor does not outlive the test.
struct KillOnDrop(u32);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if alive(self.0) {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &self.0.to_string()])
                .status();
        }
    }
}

/// The pid a fixture wrote to `path`.
fn read_pid(path: &std::path::Path, what: &str) -> u32 {
    eventually(Duration::from_secs(20), what, || {
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    })
}

/// A reaped server's `setsid`'d grandchild: once its launching shell exits it is the agent's child
/// (not init's), the reap kills it although it left the server's group, and the agent reaps its
/// corpse — the pid disappears outright while `serve` keeps running.
#[test]
fn a_reaped_servers_setsid_grandchild_is_adopted_killed_and_reaped() {
    let home = tempfile::tempdir().unwrap();
    let pidfile = home.path().join("orphan.pid");
    let (mut child, mut stdin, mut frames) = start(
        home.path(),
        json!({
            "MCP_FIXTURE_ORPHAN_PIDFILE": pidfile.to_string_lossy(),
            "MCP_FIXTURE_ORPHAN_SETSID": "1",
        }),
        "1",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    frames.response("m");
    let orphan = read_pid(&pidfile, "the grandchild's pid");
    let _cleanup = KillOnDrop(orphan);
    let agent = child.id();
    eventually(
        Duration::from_secs(10),
        "the grandchild re-parented to the agent",
        || (stat(orphan).map(|(_, ppid)| ppid) == Some(agent)).then_some(()),
    );

    // Idle for a second → reaped: the server's grace, then the sweep.
    let deadline = Instant::now() + Duration::from_secs(30);
    while stat(orphan).is_some() {
        assert!(
            Instant::now() < deadline,
            "the reaped server's grandchild {orphan} is still {:?} (state, parent)",
            stat(orphan)
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        child.try_wait().unwrap().is_none(),
        "serve is still running: the grandchild was reaped by it, not by its exit"
    );
}

/// `serve` exiting sweeps a server's `setsid`'d grandchild too, after the server's grace.
#[test]
fn serve_exiting_kills_a_grandchild_that_left_the_servers_group() {
    let home = tempfile::tempdir().unwrap();
    let marker = home.path().join("exited-cleanly");
    let pidfile = home.path().join("orphan.pid");
    let (mut child, mut stdin, mut frames) = start(
        home.path(),
        json!({
            "MCP_FIXTURE_EXIT_MARKER": marker.to_string_lossy(),
            "MCP_FIXTURE_ORPHAN_PIDFILE": pidfile.to_string_lossy(),
            "MCP_FIXTURE_ORPHAN_SETSID": "1",
        }),
        "0",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    frames.response("m");
    let orphan = read_pid(&pidfile, "the grandchild's pid");
    let _cleanup = KillOnDrop(orphan);
    assert!(alive(orphan), "the grandchild runs before serve exits");

    drop(stdin);
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "serve did not exit on EOF");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !alive(orphan),
        "serve exited and left its stdio server's setsid'd grandchild {orphan} running"
    );
    assert_eq!(
        std::fs::read_to_string(&marker).ok().as_deref(),
        Some("clean exit"),
        "the server was not given its grace window to exit on its own"
    );
}

/// SIGKILL runs no cleanup in the agent, and this server ignores its stdin closing — the only other
/// thing that would tell it to go. The kernel's parent-death signal still takes it down.
#[test]
fn a_sigkilled_agent_takes_its_stdio_server_down() {
    let home = tempfile::tempdir().unwrap();
    let pidfile = home.path().join("server.pid");
    let (mut child, mut stdin, mut frames) = start(
        home.path(),
        json!({
            "MCP_FIXTURE_PIDFILE": pidfile.to_string_lossy(),
            "MCP_FIXTURE_IGNORE_EOF": "1",
        }),
        "0",
    );
    send(&mut stdin, json!({ "type": "get_mcp", "id": "m" }));
    frames.response("m");
    let server = read_pid(&pidfile, "the server's pid");
    let _cleanup = KillOnDrop(server);
    assert!(alive(server), "the server runs before the agent is killed");

    child.kill().unwrap();
    child.wait().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(server) {
        assert!(
            Instant::now() < deadline,
            "the SIGKILLed agent's stdio server {server} is still running"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(stdin);
}
