//! The exec endpoint's response cap: an oversized response must neither exhaust memory nor pass for
//! the command's complete output.
//!
//! The runner used to buffer the whole body with no bound, so one `cat` of a huge file could take down
//! a process serving many sessions. It now reads incrementally and stops at the cap. Past the cap the
//! call is an **error**: the body is one JSON object, so a prefix of it cannot be parsed into an
//! honest partial result, and the error says the command ran so it is narrowed rather than repeated.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::SpawnGuarded as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use beyond_ai_agent::exec_endpoint::{DEFAULT_MAX_RESPONSE_BYTES, HttpExecRunner};
use beyond_ai_agent::tools::exec::CommandRunner;
use beyond_ai_agent::tools::{ToolConfig, default_registry_with_config};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MIB: usize = 1024 * 1024;

/// Accept connections forever, read each request's head, and hand the socket to `respond`.
async fn server<F, Fut>(respond: F) -> String
where
    F: Fn(tokio::net::TcpStream) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/exec", listener.local_addr().unwrap());
    let respond = Arc::new(respond);
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            let respond = respond.clone();
            tokio::spawn(async move {
                // The requests here are small; one read gets the head and body.
                let mut buf = vec![0u8; 64 * 1024];
                let _ = sock.read(&mut buf).await;
                respond(sock).await;
            });
        }
    });
    url
}

/// An endpoint whose response never ends: a valid JSON prefix, then `stdout` bytes forever, chunked.
/// Records how much it managed to send before the client hung up.
async fn endless(sent: Arc<AtomicUsize>) -> String {
    server(move |mut sock| {
        let sent = sent.clone();
        async move {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                        Transfer-Encoding: chunked\r\n\r\n";
            if sock.write_all(head.as_bytes()).await.is_err() {
                return;
            }
            let prefix = r#"{"exit_code":0,"stderr":"","stdout":""#;
            let mut frame = format!("{:x}\r\n{prefix}\r\n", prefix.len()).into_bytes();
            let body = vec![b'A'; 64 * 1024];
            loop {
                if sock.write_all(&frame).await.is_err() {
                    return; // the client hung up — the whole point
                }
                sent.fetch_add(frame.len(), Ordering::Relaxed);
                frame = format!("{:x}\r\n", body.len()).into_bytes();
                frame.extend_from_slice(&body);
                frame.extend_from_slice(b"\r\n");
            }
        }
    })
    .await
}

#[tokio::test]
async fn an_endless_response_is_cut_off_at_the_cap_and_reported_as_an_error() {
    let sent = Arc::new(AtomicUsize::new(0));
    let url = endless(sent.clone()).await;
    let runner = HttpExecRunner::new(&url)
        .unwrap()
        .with_max_response_bytes(MIB);

    // Unbounded buffering would never return here (or would return only after exhausting memory).
    let err = tokio::time::timeout(
        Duration::from_secs(60),
        runner.run("cat", &["/dev/zero".into()], None, Duration::from_secs(10)),
    )
    .await
    .expect("the read must stop at the cap rather than drain an endless body")
    .expect_err("an oversized response must be an error, not a partial result");

    let msg = err.to_string();
    assert!(
        msg.contains(&format!("exceeded the {MIB}-byte cap")),
        "{msg}"
    );
    assert!(
        msg.contains("the command ran"),
        "the caller must learn the command's side effects happened: {msg}"
    );
    // The client hung up rather than draining: the server gets only as far as the cap plus what the
    // socket buffers absorb before the close lands.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let total = sent.load(Ordering::Relaxed);
    assert!(
        total < 64 * MIB,
        "the server sent {total} bytes — the client kept reading"
    );
}

#[tokio::test]
async fn a_declared_oversized_length_fails_without_reading_the_body() {
    // A Content-Length over the cap, then silence. Reading any of the body would hang on bytes that
    // never come; the cap must be enforced from the header alone.
    let url = server(|mut sock| async move {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{{",
            64 * MIB
        );
        let _ = sock.write_all(head.as_bytes()).await;
        tokio::time::sleep(Duration::from_secs(120)).await;
        drop(sock);
    })
    .await;
    let runner = HttpExecRunner::new(&url)
        .unwrap()
        .with_max_response_bytes(MIB);
    let err = tokio::time::timeout(
        Duration::from_secs(30),
        runner.run("true", &[], None, Duration::from_secs(10)),
    )
    .await
    .expect("the declared length alone must decide it")
    .expect_err("must be the cap error");
    assert!(err.to_string().contains("-byte cap"), "{err}");
}

#[tokio::test]
async fn a_large_response_under_the_default_cap_is_returned_whole() {
    // 4 MiB of stdout: past anything a single read returns, well under the 16 MiB default.
    let stdout = "B".repeat(4 * MIB);
    let url = server(move |mut sock| {
        let body = json!({ "exit_code": 3, "stdout": stdout, "stderr": "tail" }).to_string();
        async move {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
        }
    })
    .await;
    let res = HttpExecRunner::new(&url)
        .unwrap()
        .run("true", &[], None, Duration::from_secs(10))
        .await
        .unwrap();
    const { assert!(4 * MIB < DEFAULT_MAX_RESPONSE_BYTES) };
    assert_eq!(res.code, Some(3));
    assert_eq!(res.stdout.len(), 4 * MIB);
    assert_eq!(res.stderr, "tail");
    assert!(!res.truncated, "nothing was dropped");
}

#[tokio::test]
async fn the_model_sees_an_oversized_bash_result_as_an_error_not_as_output() {
    let url = endless(Arc::new(AtomicUsize::new(0))).await;
    let runner: Arc<dyn CommandRunner> = Arc::new(
        HttpExecRunner::new(&url)
            .unwrap()
            .with_max_response_bytes(MIB),
    );
    let reg = default_registry_with_config(&ToolConfig {
        command_runner: Some(runner),
        ..ToolConfig::new()
    });
    let err = tokio::time::timeout(
        Duration::from_secs(60),
        reg.get("bash")
            .unwrap()
            .run(json!({ "command": "cat /dev/zero" })),
    )
    .await
    .expect("bash must not hang on an endless response")
    .expect_err("a response that never fit must not read as the command's output");
    assert!(err.to_string().contains("-byte cap"), "{err}");
}

#[tokio::test]
async fn run_applies_the_exec_max_response_bytes_flag() {
    // The flag, end to end through the real binary: an odd cap that no default could produce must be
    // the one the model's `bash` result reports.
    let url = endless(Arc::new(AtomicUsize::new(0))).await;
    let (base, _bodies) = common::spawn_model_server(vec![
        common::turn_tool_use("t1", "bash", &json!({ "command": "yes" }).to_string()),
        common::turn_text("done"),
    ]);
    let mut cmd = common::run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    cmd.args(["run", "do the task", "--exec-url", &url])
        .args(["--exec-max-response-bytes", "777777"])
        .args(["--gateway-url", &base, "--key", "bai_v1.test"])
        .args(["--model", "claude-test", "--max-steps", "4"])
        .args(["--no-session-persistence", "--json"]);
    // Output to files and a polled wait, not a blocking `output()`: the mock endpoint runs on this
    // test's runtime and must keep serving meanwhile, and the guard kills the child if we panic.
    let (out, err) = (
        tempfile::NamedTempFile::new().unwrap(),
        tempfile::NamedTempFile::new().unwrap(),
    );
    cmd.stdout(out.reopen().unwrap())
        .stderr(err.reopen().unwrap());
    let mut child = cmd.spawn_guarded();
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    while child.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "the run must finish");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let stdout = std::fs::read_to_string(out.path()).unwrap();
    assert!(
        stdout.contains("exceeded the 777777-byte cap"),
        "stdout: {stdout}\nstderr: {}",
        std::fs::read_to_string(err.path()).unwrap()
    );
}
