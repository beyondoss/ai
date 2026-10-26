//! The endpoint's own behaviour, embedded in-process: what the conformance verifier does not reach
//! (authentication, request limits, backpressure, shutdown, h2c, the observer).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use beyond_ai_test_support::ports;
use serde_json::{Value, json};
use support::http::{post, send};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn start(opts: impl FnOnce(&mut execd::Options)) -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    opts(&mut o);
    let server = execd::serve(ports::tokio_listener().await, o).unwrap();
    (server, dir)
}

fn run(cmd: &str, args: &[&str]) -> Value {
    json!({ "command": cmd, "args": args, "timeout_ms": 30_000 })
}

/// The library fails closed as the binary does: no bearer is served only on loopback.
#[tokio::test]
async fn serve_refuses_no_bearer_on_a_network_listener() {
    let dir = tempfile::tempdir().unwrap();
    let open = |bearer: Option<&str>, insecure: bool| {
        let mut o = execd::Options::new(dir.path());
        o.bearer = bearer.map(String::from);
        o.insecure_no_auth = insecure;
        o
    };
    let all = ports::tokio_listener_all;
    let refused = execd::serve(all().await, open(None, false))
        .err()
        .expect("served open on all interfaces");
    assert_eq!(refused.kind(), std::io::ErrorKind::PermissionDenied);
    for (bearer, insecure) in [(Some("tok"), false), (None, true)] {
        execd::serve(all().await, open(bearer, insecure))
            .unwrap()
            .shutdown()
            .await;
    }
    execd::serve(ports::tokio_listener().await, open(None, false))
        .unwrap()
        .shutdown()
        .await;
}

#[tokio::test]
async fn livez_answers_without_auth_and_everything_else_wants_the_bearer() {
    let (server, _dir) = start(|o| o.bearer = Some("open-sesame".into())).await;
    let addr = server.local_addr();
    assert_eq!(send(addr, "GET", "/livez", &[], b"").await.status, 200);
    let req = run("echo", &["hi"]);
    assert_eq!(post(addr, &[], &req).await.status, 401);
    assert_eq!(
        post(addr, &[("Authorization", "Bearer nope")], &req)
            .await
            .status,
        401
    );
    // A prefix of the right token, and the right token plus more, are both wrong.
    assert_eq!(
        post(addr, &[("Authorization", "Bearer open-ses")], &req)
            .await
            .status,
        401
    );
    assert_eq!(
        post(addr, &[("Authorization", "Bearer open-sesame!")], &req)
            .await
            .status,
        401
    );
    let ok = post(addr, &[("Authorization", "Bearer open-sesame")], &req).await;
    assert_eq!(ok.status, 200);
    assert_eq!(ok.json()["stdout"], "hi\n");
    server.shutdown().await;
}

#[tokio::test]
async fn the_envelope_runs_a_command_with_stdin_and_reports_signals() {
    let (server, dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let mut req = run("cat", &[]);
    req["stdin_base64"] = json!(base64::engine::general_purpose::STANDARD.encode(b"stdin-probe"));
    let a = post(addr, &[], &req).await;
    assert_eq!(a.json()["stdout"], "stdin-probe");
    assert_eq!(a.json()["exit_code"], 0);
    // cwd defaults to the root.
    let a = post(addr, &[], &run("pwd", &[])).await;
    let want = dir.path().canonicalize().unwrap();
    assert_eq!(
        a.json()["stdout"].as_str().unwrap().trim(),
        want.to_str().unwrap()
    );
    // A command that kills itself is 128 + the signal, as a shell reports it.
    let a = post(addr, &[], &run("sh", &["-c", "kill -9 $$"])).await;
    assert_eq!(a.json()["exit_code"], 137);
    // `timeout_ms` is optional: without it the connection is the command's lifetime.
    let a = post(addr, &[], &json!({ "command": "true" })).await;
    assert_eq!(a.json()["exit_code"], 0);
    // One that cannot start is an ordinary result.
    let a = post(addr, &[], &run("/no/such/command", &[])).await;
    assert_eq!(a.status, 200);
    assert_eq!(a.json()["exit_code"], 127);
    server.shutdown().await;
}

#[tokio::test]
async fn v11_answers_when_the_command_exits_and_kills_what_it_left_behind() {
    let (server, dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let marker = dir.path().join("alive");
    // A backgrounded child holding the pipes open must not hold up the answer, and must not survive it.
    let script = format!("(sleep 0.3; touch {}) & echo done", marker.display());
    let t0 = Instant::now();
    let a = post(addr, &[], &run("sh", &["-c", &script])).await;
    assert_eq!(a.json()["stdout"], "done\n");
    assert!(t0.elapsed() < Duration::from_secs(5));
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert!(!marker.exists(), "the leftover outlived the v1.1 answer");
    server.shutdown().await;
}

#[tokio::test]
async fn an_oversize_request_is_refused_before_it_is_read() {
    let (server, _dir) = start(|o| o.max_request_bytes = 1024).await;
    let addr = server.local_addr();
    let mut req = run("true", &[]);
    req["stdin_base64"] = json!("A".repeat(4096));
    assert_eq!(post(addr, &[], &req).await.status, 413);
    // Malformed JSON and a missing command are the client's fault.
    assert_eq!(send(addr, "POST", "/exec", &[], b"{").await.status, 400);
    assert_eq!(send(addr, "POST", "/exec", &[], b"{}").await.status, 400);
    assert_eq!(send(addr, "GET", "/exec", &[], b"").await.status, 405);
    server.shutdown().await;
}

#[test]
fn the_request_limit_is_the_largest_file_the_agent_writes_in_base64_plus_an_argv() {
    // 64 MiB (`MAX_EDIT_BYTES`) as base64, plus the 2 MiB `ARG_MAX` envelope.
    assert_eq!(execd::MAX_REQUEST_BYTES, 89_478_488 + 2 * 1024 * 1024);
}

#[tokio::test]
async fn a_stream_frames_its_output_and_ends_with_one_final_frame() {
    let (server, _dir) = start(|_| {}).await;
    let mut req = run("sh", &["-c", "printf out; printf err >&2; exit 3"]);
    req["stream"] = json!(true);
    let a = post(server.local_addr(), &[], &req).await;
    assert_eq!(a.status, 200);
    assert_eq!(a.header("content-type"), Some("application/x-ndjson"));
    assert_eq!(a.header("cache-control"), Some("no-store"));
    let frames = a.frames();
    assert_eq!(frames.last().unwrap(), &json!({ "exit_code": 3 }));
    let text = |key: &str| -> Vec<u8> {
        frames
            .iter()
            .filter_map(|f| f[key].as_str())
            .flat_map(|b| base64::engine::general_purpose::STANDARD.decode(b).unwrap())
            .collect()
    };
    assert_eq!(text("stdout"), b"out");
    assert_eq!(text("stderr"), b"err");
    server.shutdown().await;
}

/// A client that stops reading stalls the command; memory does not grow to hold what it wrote.
#[tokio::test]
async fn a_slow_client_stalls_the_command_instead_of_growing_memory() {
    let (server, dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let counter = dir.path().join("blocks");
    // 4000 blocks of 64 KiB (256 MiB), each counted once its write has completed.
    let script = format!(
        "i=0; while [ $i -lt 4000 ]; do head -c 65536 /dev/zero; i=$((i+1)); echo $i > {}; done",
        counter.display()
    );
    let mut req = run("sh", &["-c", &script]);
    req["stream"] = json!(true);
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(&support::http::request(
        "POST",
        "/exec",
        &[],
        req.to_string().as_bytes(),
    ))
    .await
    .unwrap();
    let blocks = || -> u64 {
        std::fs::read_to_string(&counter)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    };
    // Not reading: the command writes until the pipe, the channels and the socket's kernel
    // buffers are full, then blocks.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let stalled_at = blocks();
    assert!(
        stalled_at > 0 && stalled_at < 4000,
        "stalled at {stalled_at}"
    );
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(
        blocks(),
        stalled_at,
        "the command went on writing to a client that was not reading"
    );
    // Reading resumes it.
    let mut sink = vec![0u8; 1 << 20];
    let drain = tokio::time::timeout(Duration::from_secs(20), async {
        let mut total = 0usize;
        loop {
            match sock.read(&mut sink).await {
                Ok(0) | Err(_) => break total,
                Ok(n) => total += n,
            }
        }
    })
    .await
    .expect("the stream ends once it is read");
    assert!(drain > 4000 * 65536, "read {drain} bytes");
    assert_eq!(blocks(), 4000);
    server.shutdown().await;
}

#[tokio::test]
async fn a_client_that_leaves_mid_stream_loses_its_command() {
    let (server, dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let pidfile = dir.path().join("pid");
    let script = format!("echo $$ > {}; exec sleep 300", pidfile.display());
    let mut req = run("sh", &["-c", &script]);
    req["stream"] = json!(true);
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(&support::http::request(
        "POST",
        "/exec",
        &[],
        req.to_string().as_bytes(),
    ))
    .await
    .unwrap();
    let pid: i32 = loop {
        if let Some(p) = std::fs::read_to_string(&pidfile)
            .ok()
            .and_then(|s| s.trim().parse().ok())
        {
            break p;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(alive(pid));
    drop(sock);
    wait_gone(pid).await;
    server.shutdown().await;
}

fn alive(pid: i32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| {
            s.rsplit_once(')')
                .map(|(_, r)| r.trim_start().starts_with(|c| c != 'Z'))
        })
        .unwrap_or(false)
}

async fn wait_gone(pid: i32) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(pid) {
        assert!(
            Instant::now() < deadline,
            "process {pid} outlived its client"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn shutdown_kills_every_running_group() {
    let (server, dir) = start(|_| {}).await;
    let addr = server.local_addr();
    let pidfile = dir.path().join("job.pid");
    let script = format!("echo $$ > {}; exec sleep 300", pidfile.display());
    // A kept job: it outlives its request, so only a shutdown will end it.
    let mut req = run("sh", &["-c", &script]);
    req["job"] = json!({ "op": "start", "scope": "s", "id": "ab", "hold": false, "label": "t" });
    assert_eq!(post(addr, &[], &req).await.status, 200);
    let pid: i32 = loop {
        if let Some(p) = std::fs::read_to_string(&pidfile)
            .ok()
            .and_then(|s| s.trim().parse().ok())
        {
            break p;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(alive(pid));
    assert_eq!(server.shutdown().await, 1);
    wait_gone(pid).await;
}

#[tokio::test]
async fn http2_prior_knowledge_is_served_on_the_same_port() {
    use http_body_util::{BodyExt as _, Full};
    use hyper_util::rt::{TokioExecutor, TokioIo};
    let (server, _dir) = start(|_| {}).await;
    let sock = tokio::net::TcpStream::connect(server.local_addr())
        .await
        .unwrap();
    let (mut sender, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(sock))
            .await
            .unwrap();
    tokio::spawn(conn);
    let req = hyper::Request::post("http://execd/exec")
        .body(Full::new(bytes::Bytes::from(
            run("echo", &["h2"]).to_string(),
        )))
        .unwrap();
    let resp = sender.send_request(req).await.unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["stdout"], "h2\n");
    server.shutdown().await;
}

#[derive(Default)]
struct Seen(Mutex<Vec<String>>);

impl execd::Observer for Seen {
    fn request(&self, headers: &[(String, String)], body: &Value) {
        let auth = headers.iter().any(|(n, _)| n == "authorization");
        self.0.lock().unwrap().push(format!(
            "{} auth={auth}",
            body["command"].as_str().unwrap_or("")
        ));
    }
}

#[tokio::test]
async fn an_observer_sees_authenticated_requests_and_an_unset_one_costs_nothing() {
    let seen = Arc::new(Seen::default());
    let o2 = seen.clone();
    let (server, _dir) = start(move |o| {
        o.bearer = Some("b".into());
        o.observer = Some(o2);
    })
    .await;
    let addr = server.local_addr();
    post(addr, &[], &run("true", &[])).await;
    post(addr, &[("Authorization", "Bearer b")], &run("true", &[])).await;
    assert_eq!(*seen.0.lock().unwrap(), vec!["true auth=true".to_string()]);
    server.shutdown().await;
}

/// The job table keeps what the retention says and reads the rest back as a gap.
#[tokio::test]
async fn job_output_past_the_retention_is_a_gap() {
    let (server, _dir) = start(|o| o.job_output_retention = 100_000).await;
    let addr = server.local_addr();
    let mut req = run("sh", &["-c", "head -c 1000000 /dev/zero"]);
    req["job"] = json!({ "op": "start", "scope": "s", "id": "01", "hold": false, "label": "" });
    post(addr, &[], &req).await;
    let end = loop {
        let mut r = run("true", &[]);
        r["job"] = json!({ "op": "read", "scope": "s", "id": "01", "offset": 0, "max": 10 });
        let v = post(addr, &[], &r).await.json();
        if v["state"] == "ended" {
            break v;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(end["end"], 1_000_000);
    let from = end["offset"].as_u64().unwrap();
    // At least the retention is kept, and at most that plus one 64 KiB chunk.
    let kept = 1_000_000 - from;
    assert!((100_000..100_000 + 65_536).contains(&kept), "kept {kept}");
    server.shutdown().await;
}

/// Output past the cap is never answered as the result: a 422 that says the command ran, and its
/// group is gone. Exactly at the cap is a 200 with everything.
#[tokio::test]
async fn output_over_the_cap_is_an_error_and_exactly_at_it_is_the_whole_output() {
    let (server, dir) = start(|o| o.max_response_bytes = 1000).await;
    let addr = server.local_addr();
    let a = post(
        addr,
        &[],
        &run("sh", &["-c", "head -c 1000 /dev/zero | tr '\\0' x"]),
    )
    .await;
    assert_eq!(a.status, 200);
    assert_eq!(a.json()["stdout"].as_str().unwrap().len(), 1000);
    let pidfile = dir.path().join("pid");
    let script = format!(
        "echo $$ > {}; head -c 1001 /dev/zero | tr '\\0' x; (sleep 300 &) ; echo done",
        pidfile.display()
    );
    let a = post(addr, &[], &run("sh", &["-c", &script])).await;
    assert_eq!(a.status, 422);
    let msg = a.json()["error"].as_str().unwrap().to_string();
    assert!(
        msg.contains("the command ran") && msg.contains("1000-byte"),
        "{msg}"
    );
    assert!(!a.json().as_object().unwrap().contains_key("stdout"));
    // The group, backgrounded sleep included, was killed.
    let pid: i32 = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let leftovers = std::fs::read_dir("/proc")
        .unwrap()
        .flatten()
        .filter_map(|e| std::fs::read_to_string(e.path().join("stat")).ok())
        .filter(|st| {
            st.rsplit_once(')').is_some_and(|(_, r)| {
                let f: Vec<&str> = r.split_whitespace().collect();
                f.first() != Some(&"Z") && f.get(2).and_then(|g| g.parse::<i32>().ok()) == Some(pid)
            })
        })
        .count();
    assert_eq!(leftovers, 0);
    server.shutdown().await;
}
