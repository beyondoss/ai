// Test target: `.unwrap()` asserts preconditions; that's the point.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Opening a session must not pin the runtime thread every *other* session is on.
//!
//! `serve --listen` `tokio::spawn`s every session onto one process runtime, `current_thread` by
//! default (`main.rs::build_runtime`). `Persistence::open` — a directory resolve plus a full replay
//! of the transcript — used to run inline on that thread, so one session starting up stopped every
//! other session's command loop, the accept loop and the idle reaper for its whole duration. That
//! matters because starts are *correlated*: a deploy or a load-balancer rehash lands N reconnects on
//! one replica at once, and inline they serialize.
//!
//! The same shape as `tool_reactor_stall.rs`, one level up: there a tool must not pin the executor
//! its own session is on; here a session must not pin the executor every other session is on. The
//! stall is injected (`BEYOND_AI_AGENT_TEST_SLOW_SESSION_OPEN_MS`, a debug-build-only seam — see
//! `serve.rs`'s `simulated_slow_open`) because what is being pinned is *where* the open runs, not how
//! long it takes: a real transcript big enough to cost whole seconds is neither cheap to build nor
//! stable across machines, and a tight timing threshold is how a suite gets flaky.

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, TestWs, free_port, spawn_model_server, wait_for_port,
    ws_connect, ws_read_until_response, ws_send,
};
use serde_json::json;

/// How long every `Persistence::open` in the child daemon is made to take. Big enough that the gap
/// between "offloaded" (~1 ms) and "inline" (the rest of this stall) is two orders of magnitude, so
/// neither assertion below depends on a fine timing margin.
const SLOW_OPEN_MS: u64 = 3_000;

/// The ceiling for an RPC answered by an *already-started* session while another session is opening.
/// Measured here at `SLOW_OPEN_MS = 3000`: **731 µs** with the open on `spawn_blocking`, **2.70 s**
/// with it inline. This threshold sits three orders of magnitude clear of the first and a third of
/// the way to the second — generous to ordinary scheduler jitter on a loaded runner, and still
/// nowhere near the failure it guards.
const MAX_RPC: Duration = Duration::from_millis(1_000);

/// Long enough for the second connection to be accepted and its session task to reach its open,
/// short enough that the open is still in flight when the measured RPC lands.
const OPEN_IN_FLIGHT: Duration = Duration::from_millis(300);

fn serve_ws_child(base: &str, session_dir: &str, port: u16) -> ChildGuard {
    Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir,
        ])
        .env("HOME", ISOLATED_HOME)
        .env(
            "BEYOND_AI_AGENT_TEST_SLOW_SESSION_OPEN_MS",
            SLOW_OPEN_MS.to_string(),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

/// Round-trip one command and return how long the reply took. `get_todos` is answered straight out of
/// the session's in-memory state — no file I/O, no subprocess — so what it measures is purely how long
/// the session's own loop took to get a turn on the runtime.
async fn timed_rpc(ws: &mut TestWs, id: &str) -> Duration {
    let started = Instant::now();
    ws_send(ws, json!({ "type": "get_todos", "id": id })).await;
    let frames = ws_read_until_response(ws, "get_todos").await;
    let elapsed = started.elapsed();
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "get_todos {id} should succeed: {:?}",
        frames.last()
    );
    elapsed
}

#[tokio::test]
async fn one_session_opening_does_not_stall_another_sessions_rpc() {
    let (base, _requests) = spawn_model_server(vec![]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let _child = serve_ws_child(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    // The observer session, driven to a reply first so its own (equally stalled) open is behind us —
    // what this measures is one session answering *while another opens*, not a session opening.
    let mut observer = ws_connect(port, Some("bbbbbbbbbbbbbbbb")).await;
    let first = timed_rpc(&mut observer, "b0").await;
    assert!(
        first >= Duration::from_millis(SLOW_OPEN_MS / 2),
        "the injected stall is not in effect (this session's own open returned in {first:?}), so \
         nothing below would be measuring anything — a release build compiles the seam out, so run \
         this in a debug profile"
    );

    // The opener, spawned and deliberately not awaited: its `Persistence::open` is stalling right now.
    let opener = tokio::spawn(async move {
        let mut ws = ws_connect(port, Some("aaaaaaaaaaaaaaaa")).await;
        timed_rpc(&mut ws, "a0").await
    });
    tokio::time::sleep(OPEN_IN_FLIGHT).await;

    let during = timed_rpc(&mut observer, "b1").await;
    let opener = opener.await.unwrap();

    assert!(
        opener >= Duration::from_millis(SLOW_OPEN_MS / 2),
        "sanity: the other session really was mid-open for the whole measurement, but its first \
         reply came back in {opener:?}"
    );
    assert!(
        during < MAX_RPC,
        "a session's RPC waited {during:?} while a *different* session opened — session start-up is \
         back on the shared runtime thread, so one reconnect stalls every other session on the \
         replica (and the accept loop, and the reaper). `Persistence::open` belongs on \
         `spawn_blocking`, via `serve.rs`'s `open_persistence_blocking`."
    );
}
