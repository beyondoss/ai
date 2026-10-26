//! After a mixed workload has finished, execd is back where it started.
//!
//! The real binary runs a representative mix: v1.1 commands (with stdin, failing, leaving a
//! background process), streams that finish, streams the client drops mid-way, and jobs (one that
//! writes past the 8 MiB the spool keeps, running ones killed, a held one dropped unkept, a kept one,
//! ended ones removed, a final `wipe`). Then every connection is closed and every job gone, and
//! `/proc` must show what it showed before the first request: the same open descriptors, the same
//! threads, no zombie child, and nothing in the jobs directory. A descriptor leaked per stream, a
//! job directory not deleted, a child not reaped each fail it. Everything is read from `/proc` and
//! the filesystem, so it holds whatever execd's internals look like.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::baseline::Baseline;
use support::{Execd, http::post};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

const SCOPE: &str = "baseline";

async fn run(addr: SocketAddr, body: Value) -> support::http::Answer {
    post(addr, &[], &body).await
}

async fn post_ok(addr: SocketAddr, body: Value) -> Value {
    let a = run(addr, body).await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    a.json()
}

async fn job(addr: SocketAddr, mut job: Value) -> Value {
    job["scope"] = json!(SCOPE);
    let a = run(
        addr,
        json!({ "command": "true", "args": [], "timeout_ms": 30_000, "job": job }),
    )
    .await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    a.json()
}

/// Open a stream and read until `until` has appeared (or the stream ends), then drop the socket:
/// a client that leaves. `None` reads to the end.
async fn stream(addr: SocketAddr, body: &Value, until: Option<&str>) -> String {
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let b = body.to_string();
    let head = format!(
        "POST /exec HTTP/1.1\r\nHost: execd\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        b.len()
    );
    sock.write_all(head.as_bytes()).await.unwrap();
    sock.write_all(b.as_bytes()).await.unwrap();
    let mut seen = Vec::new();
    let mut buf = vec![0u8; 65536];
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let text = String::from_utf8_lossy(&seen).into_owned();
        if until.is_some_and(|u| text.contains(u)) {
            return text;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, sock.read(&mut buf)).await {
            Ok(Ok(n)) if n > 0 => seen.extend_from_slice(&buf[..n]),
            Ok(_) => return text,
            Err(_) => panic!("stream timed out:\n{text}"),
        }
    }
}

async fn wait_ended(addr: SocketAddr, id: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let r = job(
            addr,
            json!({ "op": "read", "id": id, "offset": 0, "max": 16 }),
        )
        .await;
        if r["state"] == "ended" {
            return;
        }
        assert!(Instant::now() < deadline, "job {id} never ended: {r}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn sh(script: &str) -> Value {
    json!({ "command": "sh", "args": ["-c", script], "timeout_ms": 30_000 })
}

fn start(id: &str, script: &str, hold: bool) -> Value {
    let mut b = sh(script);
    b["job"] = json!({ "op": "start", "scope": SCOPE, "id": id, "hold": hold, "label": id });
    if hold {
        b["stream"] = json!(true);
    }
    b
}

/// One round of the mix. Rounds run side by side.
async fn round(addr: SocketAddr, n: usize) {
    // v1.1: stdin, a failure, and a command that leaves a process behind.
    let a = run(
        addr,
        json!({ "command": "cat", "args": [], "timeout_ms": 30_000, "stdin_base64": "aGVsbG8=" }),
    )
    .await;
    assert_eq!(a.json()["stdout"], "hello");
    let a = run(addr, sh("echo e >&2; exit 3")).await;
    assert_eq!(a.json()["exit_code"], 3);
    let a = run(addr, sh("sleep 60 & echo left")).await;
    assert_eq!(a.json()["exit_code"], 0);

    // Streams that finish: small, past the 64 KiB frame, and one that leaves a process behind.
    for script in [
        "seq 1 100",
        "head -c 300000 /dev/zero",
        "sleep 0.3 & echo hi",
    ] {
        let mut b = sh(script);
        b["stream"] = json!(true);
        let text = stream(addr, &b, None).await;
        assert!(text.contains("exit_code"), "{text}");
    }
    // Streams the client drops mid-way: the command is killed.
    for _ in 0..3 {
        let mut b = sh("while :; do echo x; sleep 0.02; done");
        b["stream"] = json!(true);
        stream(addr, &b, Some("stdout")).await;
    }

    // Jobs. One writes past the 8 MiB the spool keeps: the oldest chunks are evicted.
    let big = format!("big{n:x}");
    post_ok(addr, start(&big, "head -c 12582912 /dev/zero", false)).await;
    wait_ended(addr, &big).await;
    let r = job(
        addr,
        json!({ "op": "read", "id": big, "offset": 0, "max": 65536 }),
    )
    .await;
    assert!(
        r["offset"].as_u64().unwrap() > 0,
        "nothing was evicted: {r}"
    );
    let mut at = json!({ "op": "attach", "id": big, "offset": 0 });
    at["scope"] = json!(SCOPE);
    let mut b = json!({ "command": "true", "args": [], "timeout_ms": 30_000, "job": at });
    b["stream"] = json!(true);
    stream(addr, &b, Some("gap")).await;
    job(
        addr,
        json!({ "op": "kill", "ids": [big], "remove": true, "grace_ms": 200 }),
    )
    .await;

    // A running job, attached to and left, then killed and removed.
    let run_id = format!("run{n:x}");
    post_ok(
        addr,
        start(&run_id, "while :; do echo tick; sleep 0.05; done", false),
    )
    .await;
    let mut at = json!({ "op": "attach", "id": run_id, "offset": 0 });
    at["scope"] = json!(SCOPE);
    let mut b = json!({ "command": "true", "args": [], "timeout_ms": 30_000, "job": at });
    b["stream"] = json!(true);
    stream(addr, &b, Some("stdout")).await;
    let k = job(
        addr,
        json!({ "op": "kill", "ids": [run_id], "remove": true, "grace_ms": 200 }),
    )
    .await;
    assert_eq!(k["survivors"], json!([]));

    // A held job whose stream is dropped unkept dies with it.
    let held = format!("held{n:x}");
    stream(
        addr,
        &start(&held, "while :; do echo h; sleep 0.05; done", true),
        Some("stdout"),
    )
    .await;

    // A job that ended on its own is removed by `remove`.
    let quick = format!("quick{n:x}");
    post_ok(addr, start(&quick, "echo done", false)).await;
    wait_ended(addr, &quick).await;
    job(addr, json!({ "op": "remove", "ids": [quick] })).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn execd_returns_to_its_starting_state_after_a_mixed_workload() {
    let jobs = tempfile::tempdir().unwrap();
    let execd = Execd::start(&[], &[("EXECD_JOBS_DIR", jobs.path().to_str().unwrap())]);
    let addr = SocketAddr::V4(execd.addr);
    let before = Baseline::take(&execd);

    let rounds: Vec<_> = (0..4).map(|n| tokio::spawn(round(addr, n))).collect();
    for r in rounds {
        r.await.unwrap();
    }
    // A kept job and a held-and-kept one are still there; `wipe` takes them and everything else.
    let kept = "kept";
    post_ok(addr, start(kept, "sleep 600", false)).await;
    let w = job(addr, json!({ "op": "wipe", "grace_ms": 200 })).await;
    assert_eq!(w["survivors"], json!([]));

    before
        .restored(jobs.path(), Duration::from_secs(15))
        .unwrap_or_else(|e| panic!("execd did not return to its starting state: {e}"));
    assert!(!execd.log().contains("panicked"));
}
