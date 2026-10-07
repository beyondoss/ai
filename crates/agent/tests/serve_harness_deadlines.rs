//! The `serve` suites' frame readers fail fast instead of hanging. A run that stalls — a scripted
//! reply taken by the wrong request, a question nobody will answer — used to leave a test blocked in
//! a read until the runner killed it, minutes later and with nothing said about where it stopped.
//! Every reader now has one shared deadline (`common::FRAME_DEADLINE`): stdout through
//! `common::child_frames`, WebSockets through `common::ws_next_frame`, and `skills_env::Serve`.
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
    let mut frames = common::child_frames_within(&mut child, Duration::from_millis(300));
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

/// No test reads a child's stdout except through `common::child_frames`: in the `serve_*` and `mcp_*`
/// suites and `tests/common`, the only code that touches a child's stdout pipe is that one helper. A
/// reader without the deadline is not discouraged but unreachable — there is no `ChildStdout` to wrap.
///
/// Matched on code with comments dropped and all whitespace removed, so line breaks and formatting
/// cannot hide a use.
#[test]
fn only_child_frames_touches_a_childs_stdout() {
    const FORBIDDEN: &[&str] = &[
        "ChildStdout",
        ".stdout.take(",
        ".stdout.as_mut(",
        ".stdout.as_ref(",
        ".stdout.unwrap(",
        ".stdout.expect(",
        ".stdout=",
    ];
    let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut files: Vec<_> = std::fs::read_dir(&tests)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            (name.starts_with("serve_") || name.starts_with("mcp_"))
                && name.ends_with(".rs")
                // This file names what it looks for.
                && name != "serve_harness_deadlines.rs"
        })
        .collect();
    files.extend(
        std::fs::read_dir(tests.join("common"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "rs")),
    );
    assert!(
        files.len() > 80,
        "the scan found the suites: {}",
        files.len()
    );
    let mut readers = 0;
    let mut found = Vec::new();
    for path in &files {
        let code: String = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<String>()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        readers += code.matches("child_frames(").count();
        // The helper itself takes the pipe, once.
        let allowed = usize::from(path.ends_with("common/mod.rs"));
        for pattern in FORBIDDEN {
            let n = code.matches(pattern).count();
            let n = if *pattern == ".stdout.take(" {
                n.saturating_sub(allowed)
            } else {
                n
            };
            if n > 0 {
                found.push(format!("{}: `{pattern}` x{n}", path.display()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a child's stdout read other than through common::child_frames:\n{}",
        found.join("\n")
    );
    assert!(
        readers > 250,
        "the suites read through child_frames ({readers})"
    );
}
