//! Integration test for systemd socket activation (`serve` adopting a `LISTEN_FDS`-passed socket). We
//! don't fake the protocol: we launch the real `beyond-ai-agent` under the real `systemd-socket-activate`
//! helper, which binds a Unix socket, sets `LISTEN_FDS`/`LISTEN_PID`, and execs us with the socket as
//! fd 3 — exactly what a systemd `.socket` unit does. The agent must adopt that fd (never bind its own)
//! and serve the byte-identical control protocol over it. Skips cleanly where the helper isn't present.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, spawn_model_server, turn_text, ws_connect_uds,
    ws_read_until_response, ws_send,
};
use serde_json::json;

/// Locate `systemd-socket-activate` (part of systemd), or `None` if it isn't installed.
fn socket_activate_bin() -> Option<PathBuf> {
    for p in [
        "/usr/bin/systemd-socket-activate",
        "/usr/lib/systemd/systemd-socket-activate",
        "/lib/systemd/systemd-socket-activate",
        "/bin/systemd-socket-activate",
    ] {
        if Path::new(p).exists() {
            return Some(PathBuf::from(p));
        }
    }
    None
}

/// Block until `sock` accepts a Unix connection (~5s), or panic.
fn wait_for_uds(sock: &Path) {
    for _ in 0..500 {
        if std::os::unix::net::UnixStream::connect(sock).is_ok() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("unix socket {} never came up", sock.display());
}

/// `serve` launched under `systemd-socket-activate` — i.e. the socket is bound by the activator and
/// passed as `LISTEN_FDS` fd 3; `serve` (with NO `--listen`/`--listen-uds`) must adopt it. Drive a real
/// `get_state` + `prompt` over that socket to prove the protocol works over the adopted fd.
#[test]
fn serve_adopts_a_systemd_activated_socket() {
    let Some(activator) = socket_activate_bin() else {
        eprintln!("systemd-socket-activate not installed — skipping socket-activation test");
        return;
    };

    let (base, _requests) = spawn_model_server(vec![turn_text("activated!")]);
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("agent.sock");
    let session_dir = dir.path().join("sessions");

    // `systemd-socket-activate -l <sock> <agent> serve ...` — note: NO --listen/--listen-uds; the agent
    // must pick up the fd from LISTEN_FDS. The activator binds `sock` and execs the agent with it.
    let mut child: ChildGuard = Command::new(&activator)
        .arg("-l")
        .arg(&sock)
        .arg(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .arg("serve")
        .arg("--gateway-url")
        .arg(&base)
        .arg("--key")
        .arg("bai_v1.test")
        .arg("--model")
        .arg("claude-test")
        .arg("--session-dir")
        .arg(&session_dir)
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn_guarded();

    wait_for_uds(&sock);

    // Drive the session over the systemd-activated socket on a small runtime (the harness is sync).
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut ws = ws_connect_uds(&sock, None).await;

        ws_send(&mut ws, json!({ "type": "get_state" })).await;
        let frames = ws_read_until_response(&mut ws, "get_state").await;
        let state = frames.last().expect("a get_state response");
        assert_eq!(
            state["success"], true,
            "get_state over the activated socket: {state}"
        );
        assert!(
            state["data"]["session_id"].as_str().is_some(),
            "get_state must report a session_id over the adopted socket: {state}"
        );

        ws_send(
            &mut ws,
            json!({ "type": "prompt", "id": "p1", "message": "hi" }),
        )
        .await;
        let frames = ws_read_until_response(&mut ws, "prompt").await;
        assert!(
            frames
                .iter()
                .any(|f| f["type"] == "ack" && f["command"] == "prompt"),
            "expected an ack over the activated socket: {frames:#?}"
        );
        assert!(
            frames.iter().any(|f| f["type"] == "event"),
            "expected a streamed event over the activated socket: {frames:#?}"
        );
        let resp = frames.last().unwrap();
        assert_eq!(
            resp["success"], true,
            "the prompt must complete over the systemd-activated socket: {resp}"
        );
    });

    let _ = child.kill();
    let _ = child.wait();
}
