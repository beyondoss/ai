//! The embedder's job-op hook, and `Activity::jobs_alive`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use beyond_ai_test_support::ports;
use serde_json::{Value, json};
use support::http::post;
use tokio::io::AsyncWriteExt as _;

async fn start(opts: impl FnOnce(&mut execd::Options)) -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    opts(&mut o);
    let server = execd::serve(ports::tokio_listener().await, o).unwrap();
    (server, dir)
}

fn job(op: Value) -> Value {
    json!({ "command": "sleep", "args": ["30"], "timeout_ms": 60_000, "job": op })
}

#[tokio::test]
async fn an_unknown_job_op_is_a_400_without_a_hook() {
    let (server, _dir) = start(|_| {}).await;
    let a = post(
        server.local_addr(),
        &[],
        &job(json!({ "op": "frobnicate" })),
    )
    .await;
    assert_eq!(a.status, 400);
    assert!(a.json()["error"].as_str().unwrap().contains("frobnicate"));
    server.shutdown().await;
}

#[tokio::test]
async fn the_hook_answers_an_unknown_op_and_only_that() {
    let seen: Arc<Mutex<Vec<(String, Value)>>> = Arc::default();
    let s = seen.clone();
    let (server, _dir) = start(|o| {
        o.job_op_hook = Some(Arc::new(move |op, whole| {
            s.lock().unwrap().push((op.to_string(), whole.clone()));
            match op {
                "frobnicate" => Some((202, json!({ "hooked": whole["arg"] }))),
                _ => None,
            }
        }));
    })
    .await;
    let addr = server.local_addr();
    let a = post(addr, &[], &job(json!({ "op": "frobnicate", "arg": 7 }))).await;
    assert_eq!((a.status, a.json()), (202, json!({ "hooked": 7 })));
    // The hook declining is the usual 400.
    let a = post(addr, &[], &job(json!({ "op": "other" }))).await;
    assert_eq!(a.status, 400);
    // Known ops never reach it.
    for op in [
        json!({ "op": "probe" }),
        json!({ "op": "list", "scope": "s" }),
        json!({ "op": "read", "scope": "s", "id": "x", "offset": 0, "max": 1 }),
        json!({ "op": "start", "scope": "s", "id": "k", "hold": false, "label": "" }),
        json!({ "op": "kill", "scope": "s", "ids": ["k"], "remove": true, "grace_ms": 100 }),
    ] {
        post(addr, &[], &job(op)).await;
    }
    let ops: Vec<String> = seen
        .lock()
        .unwrap()
        .iter()
        .map(|(o, _)| o.clone())
        .collect();
    assert_eq!(ops, ["frobnicate", "other"]);
    assert_eq!(
        seen.lock().unwrap()[0].1,
        json!({ "op": "frobnicate", "arg": 7 })
    );
    server.shutdown().await;
}

#[tokio::test]
async fn an_unauthenticated_request_never_reaches_the_hook() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let (server, _dir) = start(|o| {
        o.bearer = Some("tok".into());
        o.job_op_hook = Some(Arc::new(move |_, _| {
            c.fetch_add(1, Ordering::SeqCst);
            Some((200, json!({})))
        }));
    })
    .await;
    let addr = server.local_addr();
    let req = job(json!({ "op": "frobnicate" }));
    assert_eq!(post(addr, &[], &req).await.status, 401);
    assert_eq!(
        post(addr, &[("Authorization", "Bearer tok")], &req)
            .await
            .status,
        200
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    server.shutdown().await;
}

async fn until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn bumped(rx: &mut tokio::sync::watch::Receiver<u64>) -> bool {
    let moved = tokio::time::timeout(Duration::from_secs(2), rx.changed())
        .await
        .is_ok();
    rx.mark_unchanged();
    moved
}

/// Held and kept jobs both count, in every scope, from start to end; a held job's start and end
/// move the change counter.
#[tokio::test]
async fn jobs_alive_counts_held_and_kept_jobs_until_they_end() {
    let (server, _dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let mut rx = server.activity_changed();
    assert_eq!(server.activity().jobs_alive, 0);

    // A kept job in one scope.
    let kept = json!({ "command": "sleep", "args": ["30"], "job":
        { "op": "start", "scope": "a", "id": "kept", "hold": false, "label": "" } });
    assert_eq!(post(addr, &[], &kept).await.status, 200);
    until("the kept job", || server.activity().jobs_alive == 1).await;

    // A held job in another: its stream is the request.
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = json!({ "command": "sleep", "args": ["1"], "job":
        { "op": "start", "scope": "b", "id": "held", "hold": true, "label": "" } })
    .to_string();
    sock.write_all(
        format!(
            "POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    until("the held job", || server.activity().jobs_alive == 2).await;
    let a = server.activity();
    assert_eq!(a.running_kept.len(), 1, "a held job is not kept");

    // The held job ends on its own (and with it its stream): the view moves.
    rx.mark_unchanged();
    until("the held job to end", || server.activity().jobs_alive == 1).await;
    assert!(rx.has_changed().unwrap());

    // Removing the kept job takes it out.
    let kill = json!({ "command": "true", "job":
        { "op": "kill", "scope": "a", "ids": ["kept"], "remove": true, "grace_ms": 100 } });
    rx.mark_unchanged();
    assert_eq!(post(addr, &[], &kill).await.status, 200);
    assert!(bumped(&mut rx).await);
    until("the removal", || server.activity().jobs_alive == 0).await;
    drop(sock);
    server.shutdown().await;
}

/// The change counter moves at a held job's start and again at its end, with nothing else in flight.
#[tokio::test]
async fn a_held_jobs_start_and_end_each_bump_the_view() {
    let (server, _dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let mut rx = server.activity_changed();
    rx.mark_unchanged();
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = json!({ "command": "sleep", "args": ["1"], "job":
        { "op": "start", "scope": "s", "id": "h", "hold": true, "label": "" } })
    .to_string();
    sock.write_all(
        format!(
            "POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    until("the held job", || server.activity().jobs_alive == 1).await;
    assert!(bumped(&mut rx).await, "start");
    until("the job to end", || server.activity().jobs_alive == 0).await;
    assert!(bumped(&mut rx).await, "end");
    drop(sock);
    server.shutdown().await;
}
