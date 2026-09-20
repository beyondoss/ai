//! `serve` e2e: draining on a signal.
//!
//! A drain is the difference between a rolling deploy and an outage. The contract has four parts and
//! each one is a test here, because each is a separate way to get it wrong:
//!
//! - new sessions are refused, so the edge places them on a replica that is staying;
//! - **sessions this replica already owns keep working**, including a fresh reconnect to one — those
//!   clients cannot be served anywhere else until this replica lets go of the session's lock;
//! - `/readyz` says 503 on the first probe after the signal, which is the whole mechanism by which
//!   the load balancer stops choosing this replica, while `/livez` stays 200 so the orchestrator does
//!   not decide the process is wedged and kill it in the middle of the drain;
//! - an in-flight run is given its grace to finish rather than being cut.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, free_port, spawn_model_server, turn_text,
    turn_tool_use, wait_for_port, ws_connect, ws_next_frame,
};
use futures::SinkExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn serve_ws(base: &str, session_dir: &str, port: u16, drain_grace: &str) -> ChildGuard {
    Command::new(BIN)
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir,
            "--session-idle-timeout",
            "0",
            "--drain-grace",
            drain_grace,
        ])
        .env("HOME", ISOLATED_HOME)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

/// One bare HTTP GET, returning (status, body).
async fn get(port: u16, path: &str) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes())
        .await
        .unwrap();
    let mut out = Vec::new();
    // The replica answers `Connection: close` and closes; a peer that closes while we are still
    // reading surfaces as ECONNRESET rather than EOF, and what it already sent is still valid.
    if let Err(e) = s.read_to_end(&mut out).await
        && out.is_empty()
    {
        panic!("no response for {path}: {e}");
    }
    let text = String::from_utf8_lossy(&out).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
    (status, body.to_owned())
}

fn sigterm(pid: u32) {
    assert!(
        Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .unwrap()
            .success(),
        "failed to signal serve"
    );
}

/// Poll `/readyz` until it reports the drain, bounded. The load balancer's own probe is on a fixed
/// cadence, so "immediately" in the contract means "on the next probe", not "synchronously".
async fn await_draining(port: u16) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (status, body) = get(port, "/readyz").await;
        if status == 503 {
            return body;
        }
        assert!(
            Instant::now() < deadline,
            "/readyz never reported the drain (last status {status})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn a_draining_replica_refuses_new_sessions_but_keeps_serving_the_ones_it_owns() {
    // The heart of it: the same replica, in the same state, must answer these two differently.
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_1",
            "bash",
            &json!({ "command": "printf drain-marker; sleep 0.2; printf ready; sleep 5" })
                .to_string(),
        ),
        turn_text("done"),
    ]);
    let port = free_port();
    let child = serve_ws(&base, dir.path().to_str().unwrap(), port, "30");
    wait_for_port(port);
    let pid = child.id();

    // A session this replica owns, mid-run.
    let mut owned = ws_connect(port, Some("drain-owned")).await;
    owned
        .send(Message::Text(
            json!({ "type": "prompt", "message": "go" })
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    // Wait until the tool is provably running, so the drain has something in flight to protect.
    loop {
        let frame = ws_next_frame(&mut owned).await.expect("socket closed");
        let is_running = frame["type"] == "event"
            && frame["event"]["kind"] == "tool_progress"
            && frame["event"]["snapshot"]
                .as_str()
                .is_some_and(|s| s.contains("drain-marker"));
        if is_running {
            break;
        }
    }

    sigterm(pid);
    let body = await_draining(port).await;
    let reason: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(reason["reason"], "draining", "{body}");

    // `/livez` stays 200 throughout: a drain is not a wedged process, and an orchestrator that
    // cannot tell the difference kills the replica in the middle of the work this exists to save.
    let (livez, _) = get(port, "/livez").await;
    assert_eq!(livez, 200, "/livez must stay 200 while draining");

    // A *new* session is refused, so the edge places it on a replica that is staying.
    let fresh = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let refused = tokio_tungstenite::client_async(
        format!("ws://127.0.0.1:{port}/_beyond/agent?session_id=drain-fresh"),
        fresh,
    )
    .await;
    assert!(
        refused.is_err(),
        "a draining replica must refuse a new session"
    );

    // A *reconnect to the session it already owns* still attaches. This is the case an earlier
    // draft got wrong by checking the closed flag before the map lookup.
    let mut rejoin = ws_connect(port, Some("drain-owned")).await;
    rejoin
        .send(Message::Text(
            json!({ "type": "get_state", "id": "s" }).to_string().into(),
        ))
        .await
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let answered = loop {
        match ws_next_frame(&mut rejoin).await {
            Some(f) if f["type"] == "response" && f["command"] == "get_state" => break true,
            Some(_) => {}
            None => break false,
        }
        assert!(Instant::now() < deadline, "reconnect never answered");
    };
    assert!(
        answered,
        "a reconnect to a session this replica still owns must attach while draining"
    );
}

#[tokio::test]
async fn a_drain_lets_an_in_flight_run_finish_and_persists_it() {
    // The grace is for work already started. With one, the turn completes and is on disk; the
    // process then exits on its own rather than needing to be killed.
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use(
            "toolu_1",
            "bash",
            &json!({ "command": "printf drain-marker; sleep 0.2; printf ready; sleep 1" })
                .to_string(),
        ),
        turn_text("finished after the signal"),
    ]);
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port, "30");
    wait_for_port(port);
    let pid = child.id();

    let mut ws = ws_connect(port, Some("drain-inflight")).await;
    ws.send(Message::Text(
        json!({ "type": "prompt", "message": "go" })
            .to_string()
            .into(),
    ))
    .await
    .unwrap();
    loop {
        let frame = ws_next_frame(&mut ws).await.expect("socket closed");
        if frame["type"] == "event"
            && frame["event"]["kind"] == "tool_progress"
            && frame["event"]["snapshot"]
                .as_str()
                .is_some_and(|s| s.contains("drain-marker"))
        {
            break;
        }
    }

    sigterm(pid);

    // It exits on its own, having waited for the run rather than cutting it.
    let deadline = Instant::now() + Duration::from_secs(45);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "serve did not exit after draining"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // The turn that was in flight when the signal arrived is on disk.
    let file = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .expect("a session file");
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(
        text.contains("finished after the signal"),
        "a drain must let the in-flight run finish and persist: {text}"
    );
}

#[tokio::test]
async fn without_a_grace_a_signal_still_shuts_down_at_once() {
    // The default is zero, and it has to stay behaviour-identical to life before drain existed:
    // Ctrl-C on a laptop daemon ends it now, not in thirty seconds.
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![]);
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port, "0");
    wait_for_port(port);
    let pid = child.id();

    let began = Instant::now();
    sigterm(pid);
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        assert!(Instant::now() < deadline, "serve did not exit");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "a zero grace must not wait: took {:?}",
        began.elapsed()
    );
}
