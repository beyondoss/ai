//! `Activity::callbacks_owed`: a v1.4 callback is owed from the moment it becomes owed until it is
//! settled, abandoned at the attempt limit, or its job is removed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use beyond_test_support::ports;
use serde_json::json;
use support::http::post;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

/// A callback receiver on loopback: answers the nth request with `status(n)`; returns its address
/// and the number of requests seen.
async fn receiver(
    status: impl Fn(usize) -> u16 + Send + Sync + 'static,
) -> (String, Arc<AtomicUsize>) {
    let listener = ports::tokio_listener().await;
    let url = format!("http://{}/cb", listener.local_addr().unwrap());
    let seen = Arc::new(AtomicUsize::new(0));
    let count = seen.clone();
    let status = Arc::new(status);
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let (count, status) = (count.clone(), status.clone());
            tokio::spawn(async move {
                // The request: head, then the body its Content-Length names.
                let mut raw = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).to_lowercase();
                    if let Some(head) = text
                        .split("\r\n\r\n")
                        .next()
                        .filter(|_| text.contains("\r\n\r\n"))
                    {
                        let len: usize = head
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length:")
                                    .map(|v| v.trim().parse().unwrap())
                            })
                            .unwrap_or(0);
                        if raw.len() >= head.len() + 4 + len {
                            break;
                        }
                    }
                }
                let n = count.fetch_add(1, Ordering::SeqCst) + 1;
                let s = status(n);
                let _ = sock
                    .write_all(
                        format!("HTTP/1.1 {s} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                            .as_bytes(),
                    )
                    .await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (url, seen)
}

async fn kept_job_with_notify(addr: std::net::SocketAddr, url: &str, command: &[&str]) {
    let start = json!({ "command": command[0], "args": &command[1..], "job":
        { "op": "start", "scope": "s", "id": "j", "hold": false, "label": "" } });
    assert_eq!(post(addr, &[], &start).await.status, 200);
    let keep = json!({ "command": "true", "job": { "op": "keep", "scope": "s", "id": "j",
        "num": 1, "kind": "background", "seen": 0, "notify": { "url": url, "token": "tok-123" } } });
    assert_eq!(post(addr, &[], &keep).await.status, 200);
}

async fn until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn start() -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    o.kept_jobs_survive_idle = true;
    let server = execd::serve(ports::tokio_listener().await, o).unwrap();
    (server, dir)
}

/// 500, 500, then 200: owed through both attempts and both backoffs, and not after the 200.
#[tokio::test]
async fn a_callback_is_owed_through_its_retries_and_not_after_the_200() {
    let (server, _dir) = start().await;
    let (url, seen) = receiver(|n| if n <= 2 { 500 } else { 200 }).await;
    let mut rx = server.activity_changed();
    assert_eq!(server.activity().callbacks_owed, 0);
    // A long-running job: its end is not what is owed here, the proof at `keep` is.
    kept_job_with_notify(server.local_addr(), &url, &["sleep", "30"]).await;
    assert_eq!(server.activity().callbacks_owed, 1);
    let mut moved = 0;
    while seen.load(Ordering::SeqCst) < 3 {
        // Through the attempts and the backoffs between them it never drops to zero.
        assert_eq!(
            server.activity().callbacks_owed,
            1,
            "after {} attempts",
            seen.load(Ordering::SeqCst)
        );
        if rx.has_changed().unwrap() {
            moved += 1;
            rx.mark_unchanged();
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    until("the 200 to settle it", || {
        server.activity().callbacks_owed == 0
    })
    .await;
    assert!(moved > 0, "the view never moved");
    assert!(rx.has_changed().unwrap(), "settling did not bump the view");
    assert_eq!(seen.load(Ordering::SeqCst), 3);
    server.shutdown().await;
}

/// A receiver that says never (404) settles it; so does removing the job while a retry waits.
#[tokio::test]
async fn a_stop_answer_or_a_removal_settles_it() {
    let (server, _dir) = start().await;
    let addr = server.local_addr();
    let (url, _) = receiver(|_| 404).await;
    kept_job_with_notify(addr, &url, &["sleep", "30"]).await;
    until("the 404 to settle it", || {
        server.activity().callbacks_owed == 0
    })
    .await;

    // Removal during a backoff.
    let (server2, _dir2) = start().await;
    let addr2 = server2.local_addr();
    let (url, seen) = receiver(|_| 500).await;
    kept_job_with_notify(addr2, &url, &["sleep", "30"]).await;
    until("the first attempt", || seen.load(Ordering::SeqCst) >= 1).await;
    assert_eq!(server2.activity().callbacks_owed, 1);
    let kill = json!({ "command": "true", "job":
        { "op": "kill", "scope": "s", "ids": ["j"], "remove": true, "grace_ms": 100 } });
    assert_eq!(post(addr2, &[], &kill).await.status, 200);
    assert_eq!(server2.activity().callbacks_owed, 0);
    server.shutdown().await;
    server2.shutdown().await;
}
