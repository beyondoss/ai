//! P3: stdin throughput over ONE HTTP/2 (prior knowledge) connection, 8 requests in flight, against
//! the same run's B3 (a one-shot `cat >/dev/null` with 16 MiB of base64 stdin, same connection type).
//!
//!     cargo bench -p beyond-execd --bench jobs_stdin_h2 -- /path/to/execd [GIB=1] [INFLIGHT=8]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout
)]

use std::sync::Arc;
use std::time::Instant;

use base64::Engine as _;
use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper::Request;
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::{Value, json};

async fn call(
    tx: &hyper::client::conn::http2::SendRequest<Full<Bytes>>,
    body: Value,
) -> (u16, Bytes) {
    let mut tx = tx.clone();
    let req = Request::post("http://execd/exec")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap();
    let resp = tx.send_request(req).await.unwrap();
    let status = resp.status().as_u16();
    (status, resp.into_body().collect().await.unwrap().to_bytes())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    let bin = args.get(1).expect("path to execd");
    let gib: u64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(1);
    let inflight: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(8);
    let dir = tempfile::tempdir().unwrap();
    let port = {
        let l = beyond_test_support::ports::listener();
        l.local_addr().unwrap().port()
    };
    let mut child = std::process::Command::new(bin)
        .env("EXECD_LISTEN", format!("127.0.0.1:{port}"))
        .env("EXECD_ROOT", dir.path())
        .env("EXECD_JOBS_DIR", dir.path())
        .env_remove("EXECD_BEARER")
        .env_remove("EXECD_BEARER_FILE")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let sock = loop {
        match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
            Ok(s) => break s,
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
        }
    };
    let (tx, conn) =
        hyper::client::conn::http2::handshake(TokioExecutor::new(), TokioIo::new(sock))
            .await
            .unwrap();
    tokio::spawn(conn);

    // B3: one-shot stdin, 16 MiB.
    let big = base64::engine::general_purpose::STANDARD.encode(vec![b'x'; 16 << 20]);
    let t = Instant::now();
    let (st, _) = call(
        &tx,
        json!({"command":"sh","args":["-c","cat >/dev/null"],"stdin":big,"timeout_ms":60000}),
    )
    .await;
    assert_eq!(st, 200);
    let b3 = 16.0 / t.elapsed().as_secs_f64();

    // P3: a job that reads and discards, 128 KiB writes at their positions, `inflight` at a time.
    let (st, _) = call(
        &tx,
        json!({"command":"sh","args":["-c","cat >/dev/null"],
               "job":{"op":"start","scope":"b","id":"t1","hold":false,"stdin_open":true}}),
    )
    .await;
    assert_eq!(st, 200);
    let chunk = base64::engine::general_purpose::STANDARD.encode(vec![b'x'; 128 * 1024]);
    let n = gib * 8192;
    let sem = Arc::new(tokio::sync::Semaphore::new(inflight));
    let t = Instant::now();
    let mut tasks = Vec::new();
    for i in 0..n {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let (tx, chunk) = (tx.clone(), chunk.clone());
        tasks.push(tokio::spawn(async move {
            let (st, _) = call(
                &tx,
                json!({"command":"true","timeout_ms":60000,
                       "job":{"op":"write","scope":"b","id":"t1","at":i * 131072,"data":chunk}}),
            )
            .await;
            assert_eq!(st, 200);
            drop(permit);
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    let p3 = (gib * 1024) as f64 / t.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"B3_oneshot_stdin_MiBps_h2": b3, "P3_MiBps_h2": p3, "ratio": p3 / b3, "inflight": inflight, "gib": gib})
    );
    child.kill().unwrap();
    child.wait().unwrap();
}
