//! What an embedding host relies on: its own authorizer, the idle-connection timeout, and the
//! activity view. All in-process, through `execd::serve`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use beyond_agent::exec_endpoint::conformance::{Expectations, Verdict, verify};
use beyond_test_support::ports;
use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::{Value, json};
use support::http::{post, send};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

async fn start(opts: impl FnOnce(&mut execd::Options)) -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    opts(&mut o);
    let server = execd::serve(ports::tokio_listener().await, o).unwrap();
    (server, dir)
}

fn accept_header(value: &'static str) -> execd::Authorizer {
    Arc::new(move |h: &execd::HeaderMap| {
        h.get("x-embed-key")
            .is_some_and(|v| v.as_bytes() == value.as_bytes())
    })
}

fn run(cmd: &str, args: &[&str]) -> Value {
    json!({ "command": cmd, "args": args, "timeout_ms": 60_000 })
}

fn stream(cmd: &str, args: &[&str]) -> Value {
    let mut r = run(cmd, args);
    r["stream"] = json!(true);
    r
}

// --- the authorizer -------------------------------------------------------------------------------

/// An authorizer that accepts one header value passes every check the verifier has.
#[tokio::test]
async fn an_authorizer_passes_the_full_conformance_verifier() {
    let (server, _dir) = start(|o| {
        o.authorizer = Some(accept_header("let-me-in"));
        o.kept_jobs_survive_idle = true;
    })
    .await;
    let url = format!("http://{}/exec", server.local_addr());
    let headers = [("x-embed-key".to_string(), "let-me-in".to_string())];
    let report = verify(&url, &headers, &Expectations::default()).await;
    println!("{}", report.render());
    assert!(
        report.streams && report.jobs && report.notify,
        "{}",
        report.render()
    );
    let not_passed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| (c.name, c.verdict))
        .collect();
    assert!(not_passed.is_empty(), "{not_passed:?}\n{}", report.render());
    // And the same server refuses a caller without it.
    let no = post(server.local_addr(), &[], &run("true", &[])).await;
    assert_eq!(no.status, 401);
    assert_eq!(no.body, br#"{"error":"unauthorized"}"#);
    server.shutdown().await;
}

#[tokio::test]
async fn an_authorizer_decides_and_the_health_paths_skip_it() {
    let (server, _dir) = start(|o| o.authorizer = Some(accept_header("k"))).await;
    let addr = server.local_addr();
    for bad in [
        &[][..],
        &[("x-embed-key", "K")][..],
        &[("x-embed-key", "")][..],
    ] {
        assert_eq!(post(addr, bad, &run("true", &[])).await.status, 401);
    }
    assert_eq!(
        post(addr, &[("x-embed-key", "k")], &run("true", &[]))
            .await
            .status,
        200
    );
    for path in ["/livez", "/healthz"] {
        assert_eq!(send(addr, "GET", path, &[], b"").await.status, 200);
    }
    server.shutdown().await;
}

#[tokio::test]
async fn an_authorizer_runs_once_per_request_before_the_body_is_read() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c = calls.clone();
    let (server, _dir) = start(|o| {
        o.authorizer = Some(Arc::new(move |_| {
            c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            false
        }));
    })
    .await;
    // A body that is never sent: the refusal comes first.
    let mut sock = tokio::net::TcpStream::connect(server.local_addr())
        .await
        .unwrap();
    sock.write_all(b"POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n\r\n")
        .await
        .unwrap();
    let mut buf = [0u8; 64];
    let n = tokio::time::timeout(Duration::from_secs(2), sock.read(&mut buf))
        .await
        .expect("answered before the body")
        .unwrap();
    assert!(buf[..n].starts_with(b"HTTP/1.1 401"));
    send(server.local_addr(), "GET", "/livez", &[], b"").await;
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    server.shutdown().await;
}

#[tokio::test]
async fn setting_the_bearer_and_an_authorizer_is_a_configuration_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    o.bearer = Some("tok".into());
    o.authorizer = Some(accept_header("k"));
    let err = execd::serve(ports::tokio_listener().await, o)
        .err()
        .expect("both were accepted");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

/// An authorizer is authentication: a listener on every interface is allowed with it.
#[tokio::test]
async fn an_authorizer_counts_as_auth_on_a_network_listener() {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    o.authorizer = Some(accept_header("k"));
    execd::serve(ports::tokio_listener_all().await, o)
        .unwrap()
        .shutdown()
        .await;
}

// --- the idle timeout -----------------------------------------------------------------------------

/// One request on a kept-alive HTTP/1.1 connection; `None` if the server had closed it.
async fn h1_exec(sock: &mut tokio::net::TcpStream) -> Option<u16> {
    let body = run("true", &[]).to_string();
    let req = format!(
        "POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(req.as_bytes()).await.ok()?;
    let mut raw = Vec::new();
    let mut byte = [0u8; 1];
    while !raw.ends_with(b"\r\n\r\n") {
        if sock.read(&mut byte).await.ok()? == 0 {
            return None;
        }
        raw.push(byte[0]);
    }
    let head = String::from_utf8_lossy(&raw).into_owned();
    let len: usize = head
        .lines()
        .find_map(|l| {
            l.to_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap())
        })
        .unwrap();
    let mut rest = vec![0u8; len];
    sock.read_exact(&mut rest).await.ok()?;
    head.split(' ').nth(1).and_then(|s| s.parse().ok())
}

/// Wait for the server to close `sock`; how long that took.
async fn closed_after(sock: &mut tokio::net::TcpStream, from: Instant) -> Duration {
    let mut buf = [0u8; 16];
    let n = tokio::time::timeout(Duration::from_secs(5), sock.read(&mut buf))
        .await
        .expect("the server never closed the idle connection")
        .unwrap_or(0);
    assert_eq!(n, 0, "unexpected bytes");
    from.elapsed()
}

async fn h2_connect(
    addr: SocketAddr,
) -> (
    hyper::client::conn::http2::SendRequest<Full<Bytes>>,
    tokio::task::JoinHandle<Instant>,
) {
    let sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (send, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(sock))
            .await
            .unwrap();
    // The task ends, and says when, once the server has closed the connection.
    let done = tokio::spawn(async move {
        let _ = conn.await;
        Instant::now()
    });
    (send, done)
}

fn h2_request(addr: SocketAddr, body: &Value) -> hyper::Request<Full<Bytes>> {
    hyper::Request::post(format!("http://{addr}/exec"))
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap()
}

#[tokio::test]
async fn an_idle_http1_connection_is_closed_by_the_timeout() {
    let (server, _dir) = start(|o| o.idle_timeout = Some(Duration::from_secs(1))).await;
    let mut sock = tokio::net::TcpStream::connect(server.local_addr())
        .await
        .unwrap();
    let t0 = Instant::now();
    let took = closed_after(&mut sock, t0).await;
    assert!(
        (Duration::from_millis(900)..Duration::from_millis(2500)).contains(&took),
        "{took:?}"
    );
    server.shutdown().await;
}

#[tokio::test]
async fn an_idle_h2_connection_gets_a_goaway_by_the_timeout() {
    let (server, _dir) = start(|o| o.idle_timeout = Some(Duration::from_secs(1))).await;
    let addr = server.local_addr();
    let (mut send, done) = h2_connect(addr).await;
    let t0 = Instant::now();
    let resp = send
        .send_request(h2_request(addr, &run("true", &[])))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    resp.into_body().collect().await.unwrap();
    let closed = tokio::time::timeout(Duration::from_secs(5), done)
        .await
        .expect("no GOAWAY")
        .unwrap();
    let took = closed - t0;
    assert!(
        (Duration::from_millis(900)..Duration::from_millis(2500)).contains(&took),
        "{took:?}"
    );
    server.shutdown().await;
}

/// A request on a kept-alive connection starts the idle clock over.
#[tokio::test]
async fn a_request_resets_the_idle_clock() {
    let (server, _dir) = start(|o| o.idle_timeout = Some(Duration::from_secs(1))).await;
    let mut sock = tokio::net::TcpStream::connect(server.local_addr())
        .await
        .unwrap();
    // Three requests 0.7 s apart span 1.4 s: without the reset the connection would have closed at 1 s.
    for _ in 0..3 {
        assert_eq!(h1_exec(&mut sock).await, Some(200));
        tokio::time::sleep(Duration::from_millis(700)).await;
    }
    let last = Instant::now() - Duration::from_millis(700);
    let took = closed_after(&mut sock, last).await;
    assert!(
        (Duration::from_millis(900)..Duration::from_millis(2500)).contains(&took),
        "{took:?}"
    );
    server.shutdown().await;
}

/// A stream is a request in flight however long it lasts, even when it only heartbeats.
#[tokio::test]
async fn a_five_second_stream_with_heartbeats_is_not_cut_over_http1() {
    let (server, _dir) = start(|o| {
        o.idle_timeout = Some(Duration::from_secs(1));
        o.heartbeat = Duration::from_millis(300);
    })
    .await;
    let t0 = Instant::now();
    let a = post(server.local_addr(), &[], &stream("sleep", &["5"])).await;
    assert!(t0.elapsed() >= Duration::from_secs(5));
    let frames = a.frames();
    assert_eq!(frames.last().unwrap(), &json!({ "exit_code": 0 }));
    assert!(frames.iter().any(|f| f == &json!({})), "no heartbeat");
    server.shutdown().await;
}

#[tokio::test]
async fn a_five_second_stream_with_heartbeats_is_not_cut_over_h2() {
    let (server, _dir) = start(|o| {
        o.idle_timeout = Some(Duration::from_secs(1));
        o.heartbeat = Duration::from_millis(300);
    })
    .await;
    let addr = server.local_addr();
    let (mut send, done) = h2_connect(addr).await;
    let resp = send
        .send_request(h2_request(addr, &stream("sleep", &["5"])))
        .await
        .unwrap();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let last = String::from_utf8_lossy(&body)
        .lines()
        .last()
        .unwrap()
        .to_string();
    assert_eq!(last, r#"{"exit_code":0}"#);
    // Then the connection is idle and goes.
    tokio::time::timeout(Duration::from_secs(5), done)
        .await
        .expect("no GOAWAY after the stream")
        .unwrap();
    server.shutdown().await;
}

// --- activity -------------------------------------------------------------------------------------

async fn until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Whether the counter moved within a second, consuming the change.
async fn bumped(rx: &mut tokio::sync::watch::Receiver<u64>) -> bool {
    let moved = tokio::time::timeout(Duration::from_secs(1), rx.changed())
        .await
        .is_ok();
    rx.mark_unchanged();
    moved
}

#[tokio::test]
async fn in_flight_counts_a_v11_call_a_stream_and_a_client_that_leaves() {
    let (server, _dir) = start(|_| {}).await;
    let addr = server.local_addr();
    assert_eq!(server.activity(), execd::Activity::default());

    // A v1.1 call.
    let call = tokio::spawn(async move { post(addr, &[], &run("sleep", &["0.5"])).await });
    until("the call to count", || server.activity().in_flight == 1).await;
    assert_eq!(call.await.unwrap().status, 200);
    until("the call to end", || server.activity().in_flight == 0).await;

    // A stream.
    let call = tokio::spawn(async move { post(addr, &[], &stream("sleep", &["0.5"])).await });
    until("the stream to count", || server.activity().in_flight == 1).await;
    call.await.unwrap();
    until("the stream to end", || server.activity().in_flight == 0).await;

    // A client that disconnects mid-stream.
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = stream("sleep", &["30"]).to_string();
    sock.write_all(
        format!(
            "POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    until("the stream to count", || server.activity().in_flight == 1).await;
    drop(sock);
    until("the abandoned stream to end", || {
        server.activity().in_flight == 0
    })
    .await;
    server.shutdown().await;
}

#[tokio::test]
async fn the_health_paths_never_count() {
    let (server, _dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let mut rx = server.activity_changed();
    rx.mark_unchanged();
    for path in ["/livez", "/healthz"] {
        for method in ["GET", "HEAD"] {
            assert_eq!(send(addr, method, path, &[], b"").await.status, 200);
        }
    }
    assert_eq!(server.activity().in_flight, 0);
    assert!(!rx.has_changed().unwrap(), "a health check bumped the view");
    // And not on top of real work either.
    let call = tokio::spawn(async move { post(addr, &[], &run("sleep", &["0.5"])).await });
    until("the call to count", || server.activity().in_flight == 1).await;
    assert_eq!(send(addr, "GET", "/healthz", &[], b"").await.status, 200);
    assert_eq!(server.activity().in_flight, 1);
    call.await.unwrap();
    server.shutdown().await;
}

fn job_req(op: Value, command: Option<(&str, &[&str])>) -> Value {
    let mut r = match command {
        Some((c, a)) => run(c, a),
        None => run("true", &[]),
    };
    r["job"] = op;
    r
}

#[tokio::test]
async fn running_kept_lists_a_kept_job_until_it_ends_or_is_removed() {
    let (server, _dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let mut rx = server.activity_changed();

    // A kept job (not held) that ends on its own.
    rx.mark_unchanged();
    let start = job_req(
        json!({ "op": "start", "scope": "s", "id": "short", "hold": false, "label": "" }),
        Some(("sleep", &["1"])),
    );
    let a = post(addr, &[], &start).await;
    assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
    let pgid = i32::try_from(a.json()["started"].as_i64().unwrap()).unwrap();
    assert!(bumped(&mut rx).await, "a kept job did not bump the view");
    assert_eq!(
        server.activity().running_kept,
        vec![execd::KeptJob {
            pgid,
            scope: "s".into(),
            id: "short".into()
        }]
    );
    until("the job to end", || {
        server.activity().running_kept.is_empty()
    })
    .await;

    // A held job is not kept until `keep` says so.
    rx.mark_unchanged();
    let held = job_req(
        json!({ "op": "start", "scope": "s", "id": "held", "hold": true, "label": "" }),
        Some(("sleep", &["30"])),
    );
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    let body = held.to_string();
    sock.write_all(
        format!(
            "POST /exec HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    until("the held start to count", || {
        server.activity().in_flight == 1
    })
    .await;
    assert!(server.activity().running_kept.is_empty());
    rx.mark_unchanged();
    let keep = job_req(
        json!({ "op": "keep", "scope": "s", "id": "held", "num": 1, "kind": "background", "seen": 0 }),
        None,
    );
    assert_eq!(post(addr, &[], &keep).await.status, 200);
    assert!(bumped(&mut rx).await, "keep did not bump the view");
    assert_eq!(server.activity().running_kept.len(), 1);
    drop(sock);
    // Kept, so it survives the hold going away; removing it takes it out of the view.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(server.activity().running_kept.len(), 1);
    rx.mark_unchanged();
    let kill = job_req(
        json!({ "op": "kill", "scope": "s", "ids": ["held"], "remove": true, "grace_ms": 200 }),
        None,
    );
    assert_eq!(post(addr, &[], &kill).await.status, 200);
    assert!(bumped(&mut rx).await, "remove did not bump the view");
    until("the removed job to leave the view", || {
        server.activity().running_kept.is_empty()
    })
    .await;
    server.shutdown().await;
}

/// Each transition moves the counter: request start, request end, job kept, kept job ended, job
/// removed.
#[tokio::test]
async fn activity_changed_fires_for_each_transition() {
    let (server, _dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let mut rx = server.activity_changed();
    rx.mark_unchanged();

    // Request start, then end.
    let call = tokio::spawn(async move { post(addr, &[], &run("sleep", &["0.6"])).await });
    assert!(bumped(&mut rx).await, "request start");
    assert_eq!(server.activity().in_flight, 1);
    call.await.unwrap();
    until("the end", || server.activity().in_flight == 0).await;
    rx.mark_unchanged();
    // The end itself bumped (the count dropped after the answer): observed by the settled state
    // above, and by a fresh pair below with nothing else running.
    let start_then_end = tokio::spawn(async move { post(addr, &[], &run("true", &[])).await });
    start_then_end.await.unwrap();
    until("the second end", || server.activity().in_flight == 0).await;
    assert!(rx.has_changed().unwrap(), "a whole request bumped nothing");

    // Kept, ended, removed.
    rx.mark_unchanged();
    let start = job_req(
        json!({ "op": "start", "scope": "s", "id": "j", "hold": false, "label": "" }),
        Some(("sleep", &["0.6"])),
    );
    assert_eq!(post(addr, &[], &start).await.status, 200);
    assert!(bumped(&mut rx).await, "job kept");
    assert_eq!(server.activity().running_kept.len(), 1);
    // Nothing is in flight now, so the only thing left to move the counter is the job's end.
    until("the start to finish", || server.activity().in_flight == 0).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    rx.mark_unchanged();
    assert!(bumped(&mut rx).await, "kept job ended");
    assert!(server.activity().running_kept.is_empty());
    let remove = job_req(json!({ "op": "remove", "scope": "s", "ids": ["j"] }), None);
    rx.mark_unchanged();
    assert_eq!(post(addr, &[], &remove).await.status, 200);
    assert!(bumped(&mut rx).await, "job removed");
    server.shutdown().await;
}
