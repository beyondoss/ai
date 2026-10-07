//! The `serve` suites' frame readers fail fast instead of hanging. A run that stalls — a scripted
//! reply taken by the wrong request, a question nobody will answer — used to leave a test blocked in
//! a read until the runner killed it, minutes later and with nothing said about where it stopped.
//! Every reader now has one shared deadline (`common::FRAME_DEADLINE`): stdout through
//! `common::serve_frames`, WebSockets through `common::ws_next_frame`, and `skills_env::Serve`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A stdout that stays open and silent fails the read at the deadline.
#[test]
fn a_silent_stdout_fails_the_read_at_its_deadline() {
    let mut child = common::ChildGuard::spawn(
        Command::new("sleep")
            .arg("30")
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    );
    let mut frames =
        common::frames_with_deadline(child.stdout.take().unwrap(), Duration::from_millis(300));
    let started = Instant::now();
    let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        common::read_until_response(&mut frames, "prompt")
    }));
    let message = *read.unwrap_err().downcast::<String>().unwrap();
    assert!(message.contains("stalled"), "{message}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
}

/// A WebSocket that stays open and silent fails the read at the deadline.
#[tokio::test]
async fn a_silent_websocket_fails_the_read_at_its_deadline() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        tokio::time::sleep(Duration::from_secs(30)).await;
        drop(ws);
    });
    let mut ws = common::ws_connect(port, None).await;
    let started = Instant::now();
    let read = tokio::spawn(async move {
        common::ws_next_frame_within(&mut ws, Duration::from_millis(300)).await
    })
    .await;
    let message = *read.unwrap_err().into_panic().downcast::<String>().unwrap();
    assert!(message.contains("stalled"), "{message}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    server.abort();
}

/// Every `serve_*` suite reads its child's stdout through `serve_frames`, and `skills_env` through
/// the same `Frames` — none through a bare `BufReader`, which has no deadline.
#[test]
fn every_serve_suite_reads_frames_with_the_shared_deadline() {
    let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut readers = 0;
    let mut bare = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(&tests)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            // This file names the pattern it looks for.
            name.starts_with("serve_")
                && name.ends_with(".rs")
                && name != "serve_harness_deadlines.rs"
        })
        .collect();
    files.push(tests.join("common/skills_env.rs"));
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        readers += text.matches("serve_frames(").count();
        for (n, line) in text.lines().enumerate() {
            if line.contains("BufReader::new(") && line.contains("stdout") {
                bare.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
            }
        }
    }
    assert!(
        bare.is_empty(),
        "stdout read without a deadline:\n{}",
        bare.join("\n")
    );
    assert!(
        readers > 200,
        "the suites read through serve_frames ({readers} readers)"
    );
}
