// Test target: `.unwrap()` asserts preconditions; that's the point.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! A tool must not pin the async runtime for the duration of its file I/O and CPU work.
//!
//! `serve_ws` gives every session its own **`current_thread`** runtime, so a tool that does its work
//! inline — rather than handing it to `spawn_blocking` — stops that session's executor dead for as
//! long as it runs. Nothing else on that runtime gets polled: not the outbound event pump, and not the
//! stdin/WebSocket command loop that carries `abort` and `steer`. `serve.rs`'s `persist_blocking`
//! already moved `sync_all` off the reactor for exactly this reason, and its doc comment says so; the
//! file tools are the ones that hadn't followed.
//!
//! These tests pin that invariant where it's cheapest to check: a ticker task that wants to wake every
//! millisecond, running concurrently with one tool call on a `current_thread` runtime. If the tool
//! yields, the ticker keeps ticking and the worst gap stays small. If the tool blocks, the worst gap is
//! the whole tool call.
//!
//! The subject files are deliberately large (a few MB), so a blocking implementation stalls for tens of
//! milliseconds and the assertion has a wide, un-flaky margin between "yielded" and "blocked" — rather
//! than depending on a tight timing threshold.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use agent_core::Tool;
use beyond_ai_agent::tools::{edit, ls, write};
use serde_json::json;

/// The worst gap the ticker observed between its own wakeups while `work` ran. A yielding tool leaves
/// this near the tick interval; a blocking one leaves it at roughly the tool's own duration.
async fn worst_ticker_gap<F, T>(work: F) -> Duration
where
    F: std::future::Future<Output = T>,
{
    let worst = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (w, s) = (worst.clone(), stop.clone());
    let ticker = tokio::spawn(async move {
        let mut last = Instant::now();
        while !s.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(1)).await;
            let now = Instant::now();
            w.fetch_max(
                now.duration_since(last).as_micros() as u64,
                Ordering::Relaxed,
            );
            last = now;
        }
    });
    // Let the ticker settle so its own startup isn't counted as a stall.
    tokio::time::sleep(Duration::from_millis(20)).await;
    worst.store(0, Ordering::Relaxed);

    work.await;

    stop.store(true, Ordering::Relaxed);
    let _ = ticker.await;
    Duration::from_micros(worst.load(Ordering::Relaxed))
}

/// Generous enough that ordinary scheduler jitter on a loaded CI box can't trip it, yet far below the
/// tens of milliseconds an inline implementation stalls for on these inputs.
const MAX_STALL: Duration = Duration::from_millis(15);

fn big_ascii_source(lines: usize) -> String {
    let mut s = String::with_capacity(lines * 56);
    for i in 0..lines {
        s.push_str(&format!("    let x_{i} = compute(i, {i}) + adjust();\n"));
    }
    s
}

#[tokio::test(flavor = "current_thread")]
async fn edit_does_not_stall_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subject.rs");
    // ~4 MB: an inline `edit` spends tens of ms here (read + normalize + match + splice + write).
    let src = big_ascii_source(80_000);
    std::fs::write(&path, &src).unwrap();

    let tool = edit::Edit::new(dir.path());
    let p = path.to_str().unwrap().to_string();
    let gap = worst_ticker_gap(async {
        tool.run(json!({
            "path": p,
            "old_string": "    let x_40001 = compute(i, 40001) + adjust();",
            "new_string": "    let x_40001 = compute(i, 40001) + adjust(); // edited",
        }))
        .await
        .unwrap()
    })
    .await;

    assert!(
        gap < MAX_STALL,
        "`edit` pinned the current_thread runtime for {gap:?} — nothing else on this session's \
         executor could be polled for that long, including its abort/steer command loop. Its file I/O \
         and matching belong on `spawn_blocking`, like `read`/`grep`/`find` already are."
    );
}

#[tokio::test(flavor = "current_thread")]
async fn write_does_not_stall_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.rs");
    let src = big_ascii_source(80_000);

    let tool = write::Write::new(dir.path());
    let p = path.to_str().unwrap().to_string();
    let gap = worst_ticker_gap(async {
        tool.run(json!({ "path": p, "content": src }))
            .await
            .unwrap()
    })
    .await;

    assert!(
        gap < MAX_STALL,
        "`write` pinned the current_thread runtime for {gap:?} — its file I/O belongs on \
         `spawn_blocking`."
    );
}

#[tokio::test(flavor = "current_thread")]
async fn ls_does_not_stall_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    // `ls` stats every entry it collects; a wide directory is where that adds up.
    for i in 0..5_000 {
        std::fs::create_dir(dir.path().join(format!("sub_{i:05}"))).unwrap();
    }

    let d = dir.path().to_str().unwrap().to_string();
    let gap = worst_ticker_gap(async {
        ls::Ls::default()
            .run(json!({ "path": d, "limit": 5000 }))
            .await
            .unwrap()
    })
    .await;

    assert!(
        gap < MAX_STALL,
        "`ls` pinned the current_thread runtime for {gap:?} — its `read_dir` + per-entry `metadata` \
         belong on `spawn_blocking`."
    );
}
