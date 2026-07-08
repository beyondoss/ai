//! Integration tests for the Unix-domain-socket transport (`serve --listen-uds`). Each test spawns the
//! real `beyond-ai-agent` binary bound to a UDS (and, for the cross-transport case, a TCP port too)
//! against the mock Anthropic-SSE gateway, then drives it with a real `tokio-tungstenite` client over a
//! `UnixStream`. The JSON control protocol is byte-identical to the TCP/stdio transports, so these
//! assert the *UDS transport*: a full session over the socket, that a session created over TCP is
//! reachable over UDS by the same id (one shared supervisor), the socket's permission mode, and stale-
//! socket recovery.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::os::unix::fs::PermissionsExt;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::{
    ISOLATED_HOME, free_port, spawn_model_server, turn_text, wait_for_port, ws_connect,
    ws_connect_uds, ws_read_until_response, ws_send,
};
use serde_json::json;

/// Spawn `serve --listen-uds <sock>` (and optionally `--listen <tcp>`) against `base` (the mock
/// gateway), persisting per-session files under `session_dir`. In listener mode stdio is unused, so
/// null it.
fn serve_uds_child(base: &str, session_dir: &str, sock: &str, tcp: Option<u16>) -> Child {
    let mut c = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    c.args([
        "serve",
        "--listen-uds",
        sock,
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-dir",
        session_dir,
    ]);
    if let Some(port) = tcp {
        c.args(["--listen", &format!("127.0.0.1:{port}")]);
    }
    c.env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn serve --listen-uds")
}

/// Block until the UDS at `path` accepts a connection, or panic after ~5s. The socket *file* appearing
/// isn't enough — only a successful connect proves the daemon is listening.
async fn wait_for_uds(path: &std::path::Path) {
    for _ in 0..500 {
        if tokio::net::UnixStream::connect(path).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("unix socket {} never came up", path.display());
}

/// (a) A full session over the socket: `get_state` answers with a `session_id`, and a `prompt` streams
/// `ack` → ≥1 `event` → `response{success:true}` in that order — byte-identical to the TCP transport.
#[tokio::test]
async fn uds_get_state_then_prompt_streams_ack_event_response() {
    let (base, _requests) = spawn_model_server(vec![turn_text("hello there")]);
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("agent.sock");
    let mut child = serve_uds_child(
        &base,
        dir.path().to_str().unwrap(),
        sock.to_str().unwrap(),
        None,
    );
    wait_for_uds(&sock).await;

    let mut ws = ws_connect_uds(&sock, None).await;

    ws_send(&mut ws, json!({ "type": "get_state" })).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    let state = frames.last().expect("a get_state response");
    assert_eq!(state["type"], "response");
    assert_eq!(state["success"], true, "get_state should succeed: {state}");
    assert!(
        state["data"]["session_id"].as_str().is_some(),
        "get_state must report a session_id: {state}"
    );

    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "hi" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;

    let ack_pos = frames
        .iter()
        .position(|f| f["type"] == "ack" && f["command"] == "prompt")
        .unwrap_or_else(|| panic!("no ack frame: {frames:#?}"));
    assert_eq!(frames[ack_pos]["id"], "p1", "ack echoes the client id");
    let event_pos = frames
        .iter()
        .position(|f| f["type"] == "event")
        .unwrap_or_else(|| panic!("no event frame: {frames:#?}"));
    assert!(ack_pos < event_pos, "ack must precede the first event");
    let resp = frames.last().unwrap();
    assert_eq!(resp["type"], "response");
    assert_eq!(resp["success"], true, "the prompt should succeed: {resp}");

    let _ = child.kill();
    let _ = child.wait();
}

/// (b) Cross-transport re-attach: a session created over **TCP** is reachable over **UDS** by the same
/// `?session_id=`, proving both transports front one shared supervisor map. Create + prompt over TCP,
/// drop, reconnect over UDS with the same id, and `get_messages` shows the committed turn.
#[tokio::test]
async fn uds_cross_transport_reattach_shares_supervisor() {
    const SID: &str = "crosstransport01";
    let (base, _requests) = spawn_model_server(vec![turn_text("crossreply")]);
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("agent.sock");
    let port = free_port();
    let mut child = serve_uds_child(
        &base,
        dir.path().to_str().unwrap(),
        sock.to_str().unwrap(),
        Some(port),
    );
    wait_for_port(port);
    wait_for_uds(&sock).await;

    // Create the session over TCP and run a prompt to completion.
    let mut tcp = ws_connect(port, Some(SID)).await;
    ws_send(
        &mut tcp,
        json!({ "type": "prompt", "id": "p1", "message": "over tcp" }),
    )
    .await;
    let frames = ws_read_until_response(&mut tcp, "prompt").await;
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "the TCP prompt should succeed: {:?}",
        frames.last()
    );
    drop(tcp); // detach the TCP connection; the session lives on in the supervisor.

    // Reconnect over UDS to the same id and read history — the committed assistant turn must be there.
    let mut uds = ws_connect_uds(&sock, Some(SID)).await;
    ws_send(&mut uds, json!({ "type": "get_state" })).await;
    let frames = ws_read_until_response(&mut uds, "get_state").await;
    assert_eq!(
        frames.last().unwrap()["data"]["session_id"],
        SID,
        "the UDS client re-attaches to the same session id: {:?}",
        frames.last()
    );

    ws_send(&mut uds, json!({ "type": "get_messages" })).await;
    let frames = ws_read_until_response(&mut uds, "get_messages").await;
    let dump = frames.last().unwrap()["data"]["messages"].to_string();
    assert!(
        dump.contains("crossreply") && dump.contains("over tcp"),
        "the turn created over TCP must be visible over UDS (shared supervisor): {dump}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// (c) The bound socket is `chmod`'d to the default owner-only `0o600` — the local-authz story UDS
/// exists for (loopback TCP has none).
#[tokio::test]
async fn uds_socket_has_owner_only_permissions() {
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("agent.sock");
    let mut child = serve_uds_child(
        &base,
        dir.path().to_str().unwrap(),
        sock.to_str().unwrap(),
        None,
    );
    wait_for_uds(&sock).await;

    let mode = std::fs::metadata(&sock).unwrap().permissions().mode();
    assert_eq!(
        mode & 0o777,
        0o600,
        "the UDS should be chmod 0o600 by default, got {:o}",
        mode & 0o777
    );

    let _ = child.kill();
    let _ = child.wait();
}

/// (d) A stale socket file left at the path (e.g. a crashed prior daemon) must not stop startup: the
/// daemon detects nothing is listening, removes the stale node, rebinds, and serves normally.
#[tokio::test]
async fn uds_stale_socket_file_is_reclaimed() {
    let (base, _requests) = spawn_model_server(vec![turn_text("afterrebind")]);
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("agent.sock");

    // Pre-create a *dead* socket node: bind a UnixListener, then drop it. The path now exists on disk
    // (a socket file) but nothing is listening — exactly the crashed-daemon leftover we must reclaim.
    {
        let stale = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        drop(stale);
    }
    assert!(sock.exists(), "the stale socket node should be on disk");

    let mut child = serve_uds_child(
        &base,
        dir.path().to_str().unwrap(),
        sock.to_str().unwrap(),
        None,
    );
    wait_for_uds(&sock).await;

    // It serves normally after reclaiming the stale node.
    let mut ws = ws_connect_uds(&sock, None).await;
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p1", "message": "hi" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "the daemon should serve after rebinding over a stale socket: {:?}",
        frames.last()
    );

    let _ = child.kill();
    let _ = child.wait();
}
