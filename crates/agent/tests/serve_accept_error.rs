//! A failed `accept` must not leave `serve --listen` deaf.
//!
//! The accept loop used to treat any accept error as the end of that listener: the daemon stayed up
//! but answered nothing more on it (and, on the Unix socket, kept the path so no replacement could
//! bind it). The error injected here is the realistic one — `EMFILE` — produced from outside by
//! lowering the running daemon's descriptor limit (`prlimit`) to just above what it holds at idle,
//! then opening more connections than it can take. Once those clients leave, the daemon has to serve
//! the next connection.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{ChildGuard, ISOLATED_HOME, spawn_listening_logged, spawn_model_server};
use rustix::process::{Pid, Resource, Rlimit, prlimit};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// More connections than the daemon has descriptors left for, by a wide margin.
const FLOOD: usize = 64;

/// Descriptors left over above what the daemon holds at idle: enough to accept a few of the flood,
/// so the next accept is the one that fails.
const HEADROOM: u64 = 4;

fn serve_daemon(
    base: &str,
    dir: &Path,
    uds: Option<&Path>,
) -> (ChildGuard, u16, std::path::PathBuf) {
    let log = dir.join("serve.log");
    let mut cmd = crate::common::agent_cmd();
    cmd.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-dir",
    ])
    .arg(dir.join("sessions"))
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::null())
    .stdout(Stdio::null());
    if let Some(uds) = uds {
        cmd.arg("--listen-uds").arg(uds);
    }
    let (child, port) =
        spawn_listening_logged(&mut cmd, Some(std::fs::File::create(&log).unwrap()));
    (child, port, log)
}

/// Clamp the daemon's descriptor limit (soft and hard) to what it holds now plus [`HEADROOM`].
fn starve_descriptors(pid: u32) {
    let open = std::fs::read_dir(format!("/proc/{pid}/fd"))
        .unwrap()
        .count() as u64;
    let limit = open + HEADROOM;
    prlimit(
        Pid::from_raw(pid as i32),
        Resource::Nofile,
        Rlimit {
            current: Some(limit),
            maximum: Some(limit),
        },
    )
    .expect("lower the daemon's RLIMIT_NOFILE");
}

/// Wait until the daemon has logged a failed accept — proof the error path ran, so the recovery
/// asserted afterwards is not vacuous.
async fn await_accept_failure(log: &Path) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        if text.contains("accept failed") {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the daemon never failed an accept; log:\n{text}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// `GET /livez` over `stream`, bounded so a deaf daemon fails the test instead of hanging it.
async fn livez<S: AsyncRead + AsyncWrite + Unpin>(mut stream: S) -> Option<String> {
    let exchange = async {
        stream
            .write_all(b"GET /livez HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
            .await
            .ok()?;
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.ok()?;
        Some(String::from_utf8_lossy(&buf).into_owned())
    };
    tokio::time::timeout(Duration::from_secs(10), exchange)
        .await
        .ok()
        .flatten()
}

#[tokio::test]
async fn tcp_listener_serves_again_after_running_out_of_descriptors() {
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let (child, port, log) = serve_daemon(&base, dir.path(), None);

    starve_descriptors(child.id());
    let mut flood = Vec::new();
    for _ in 0..FLOOD {
        flood.push(
            tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .unwrap(),
        );
    }
    await_accept_failure(&log).await;
    drop(flood);

    let stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let response = livez(stream).await;
    assert!(
        response
            .as_deref()
            .is_some_and(|r| r.starts_with("HTTP/1.1 200")),
        "the TCP listener stopped serving after an accept error: {response:?}; log:\n{}",
        std::fs::read_to_string(&log).unwrap_or_default()
    );
}

#[tokio::test]
async fn unix_listener_serves_again_after_running_out_of_descriptors() {
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("agent.sock");
    let (child, _port, log) = serve_daemon(&base, dir.path(), Some(&sock));
    // Bound before the TCP announcement, but don't depend on that ordering.
    let deadline = Instant::now() + Duration::from_secs(10);
    while tokio::net::UnixStream::connect(&sock).await.is_err() {
        assert!(Instant::now() < deadline, "the unix socket never came up");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    starve_descriptors(child.id());
    let mut flood = Vec::new();
    for _ in 0..FLOOD {
        flood.push(tokio::net::UnixStream::connect(&sock).await.unwrap());
    }
    await_accept_failure(&log).await;
    drop(flood);

    let stream = tokio::net::UnixStream::connect(&sock).await.unwrap();
    let response = livez(stream).await;
    assert!(
        response
            .as_deref()
            .is_some_and(|r| r.starts_with("HTTP/1.1 200")),
        "the unix listener stopped serving after an accept error: {response:?}; log:\n{}",
        std::fs::read_to_string(&log).unwrap_or_default()
    );
}
