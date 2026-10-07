// Test target: `.unwrap()` asserts preconditions; that's the point.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! A tool must not pin the async runtime for the duration of its file I/O and CPU work.
//!
//! `serve_ws` runs every session as a task on one **shared** runtime, so a tool that does its work
//! inline — rather than handing it to `spawn_blocking` — stalls a worker that other sessions also
//! use. Nothing else on that worker gets polled meanwhile: not the outbound event pump, and not the
//! stdin/WebSocket command loop that carries `abort` and `steer`. `serve.rs`'s `persist_blocking`
//! already moved `sync_all` off the reactor for exactly this reason, and its doc comment says so.
//!
//! **The probe measures CPU, not wall time.** Each tool call is polled on a `current_thread`
//! runtime, and the probe adds up the *thread CPU time* the runtime thread spends inside the tool
//! future's `poll`s: the time the tool holds the executor doing its own work. Work handed to
//! `spawn_blocking` is charged to a pool thread, not to this one. The bar is relative to a baseline
//! taken in the same run: the CPU the runtime thread spends doing the same work inline. An inline
//! tool holds the executor for about the whole baseline; a yielding one for a small fraction of it.
//!
//! Thread CPU time does not advance while the thread is preempted. So a loaded host can make a run
//! slower, but cannot make a yielding tool look like a blocking one. A wall-clock probe (a ticker's
//! worst gap) could, and did: it counts every preemption anywhere in the call.
//! `the_probe_catches_a_tool_that_works_inline` keeps the probe's teeth honest.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use agent_core::Tool;
use beyond_ai_agent::tools::{edit, ls, write};
use serde_json::json;

/// CPU time consumed by the calling thread so far.
fn thread_cpu() -> Duration {
    let ts = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// What the runtime thread spent inside one future's `poll`s.
struct Held {
    /// Thread CPU time inside `poll`.
    cpu: Duration,
    /// How many `poll`s returned `Pending`: a tool that never yields returns `Ready` from its first.
    pendings: u32,
}

struct Probe<F> {
    inner: Pin<Box<F>>,
    held: Held,
}

impl<F: Future> Future for Probe<F> {
    type Output = (F::Output, Duration, u32);

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let start = thread_cpu();
        let polled = self.inner.as_mut().poll(cx);
        let spent = thread_cpu() - start;
        self.held.cpu += spent;
        match polled {
            Poll::Ready(out) => Poll::Ready((out, self.held.cpu, self.held.pendings)),
            Poll::Pending => {
                self.held.pendings += 1;
                Poll::Pending
            }
        }
    }
}

/// Run `work` on this (`current_thread`) runtime, returning how long it held the executor.
async fn held<F: Future>(work: F) -> Held {
    let (_, cpu, pendings) = Probe {
        inner: Box::pin(work),
        held: Held {
            cpu: Duration::ZERO,
            pendings: 0,
        },
    }
    .await;
    Held { cpu, pendings }
}

/// The thread CPU time `work` takes, run inline on this thread: the baseline a tool's hold is
/// measured against. The cheapest of three runs, so a cold cache doesn't inflate it.
fn inline_cpu(mut work: impl FnMut()) -> Duration {
    (0..3)
        .map(|_| {
            let start = thread_cpu();
            work();
            thread_cpu() - start
        })
        .min()
        .unwrap()
}

/// A yielding tool may hold the executor for at most this fraction of the inline baseline (its own
/// argument parsing, the hand-off, and assembling the result). An inline one holds about all of it.
///
/// That per-call overhead is fixed (a few hundred microseconds in a debug build) while the baseline
/// scales with the input, so the inputs are sized for a baseline of several milliseconds even on a
/// fast machine: a 4 MB `write` was under a millisecond on a CI runner, and the fixed overhead alone
/// came to a third of it.
const MAX_HELD_FRACTION: f64 = 0.25;

/// Lines of [`big_ascii_source`] for the `edit` and `write` subjects: ~36 MB.
const SUBJECT_LINES: usize = 640_000;

fn assert_yields(tool: &str, held: &Held, baseline: Duration, why: &str) {
    eprintln!(
        "PROBE {tool}: held {:?} of {baseline:?}, {} pendings",
        held.cpu, held.pendings
    );
    assert!(
        held.pendings > 0,
        "`{tool}` never yielded: its first poll ran to completion, all on the runtime thread. {why}"
    );
    assert!(
        held.cpu.as_secs_f64() < baseline.as_secs_f64() * MAX_HELD_FRACTION,
        "`{tool}` held the current_thread runtime for {:?} of thread CPU, against {baseline:?} \
         for the same work done inline: nothing else on this session's executor could be polled \
         meanwhile, including its abort/steer command loop. {why}",
        held.cpu
    );
}

fn big_ascii_source(lines: usize) -> String {
    let mut s = String::with_capacity(lines * 56);
    for i in 0..lines {
        s.push_str(&format!("    let x_{i} = compute(i, {i}) + adjust();\n"));
    }
    s
}

const OLD: &str = "    let x_40001 = compute(i, 40001) + adjust();";
const NEW: &str = "    let x_40001 = compute(i, 40001) + adjust(); // edited";

#[tokio::test(flavor = "current_thread")]
async fn edit_does_not_stall_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subject.rs");
    // An inline `edit` spends tens of ms here (read + normalize + match + splice + write).
    let src = big_ascii_source(SUBJECT_LINES);
    std::fs::write(&path, &src).unwrap();
    let scratch = dir.path().join("baseline.rs");
    let baseline = inline_cpu(|| {
        let text = std::fs::read_to_string(&path).unwrap();
        let at = text.find(OLD).unwrap();
        let mut out = String::with_capacity(text.len() + NEW.len());
        out.push_str(&text[..at]);
        out.push_str(NEW);
        out.push_str(&text[at + OLD.len()..]);
        std::fs::write(&scratch, out).unwrap();
    });

    let tool = edit::Edit::new(dir.path());
    let p = path.to_str().unwrap().to_string();
    let held = held(async {
        tool.run(json!({ "path": p, "old_string": OLD, "new_string": NEW }))
            .await
            .unwrap()
    })
    .await;
    assert!(std::fs::read_to_string(&path).unwrap().contains(NEW));
    assert_yields(
        "edit",
        &held,
        baseline,
        "Its file I/O and matching belong on `spawn_blocking`, like `read`/`grep`/`find` already are.",
    );
}

#[tokio::test(flavor = "current_thread")]
async fn write_does_not_stall_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.rs");
    let src = big_ascii_source(SUBJECT_LINES);
    let scratch = dir.path().join("baseline.rs");
    let baseline = inline_cpu(|| std::fs::write(&scratch, &src).unwrap());

    let tool = write::Write::new(dir.path());
    // Built before the probe starts: `json!` copies `src`, and that copy is the test's, not the tool's.
    let input = json!({ "path": path.to_str().unwrap(), "content": src });
    let held = held(async { tool.run(input).await.unwrap() }).await;
    assert_eq!(std::fs::metadata(&path).unwrap().len(), src.len() as u64);
    assert_yields(
        "write",
        &held,
        baseline,
        "Its file I/O belongs on `spawn_blocking`.",
    );
}

/// What an inline `ls` would do with `dir`: read it and stat every entry.
fn scan(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().is_dir())
        .filter(|d| *d)
        .count()
}

fn wide_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    // `ls` stats every entry it collects; a wide directory is where that adds up.
    for i in 0..5_000 {
        std::fs::create_dir(dir.path().join(format!("sub_{i:05}"))).unwrap();
    }
    dir
}

#[tokio::test(flavor = "current_thread")]
async fn ls_does_not_stall_the_runtime() {
    let dir = wide_dir();
    let baseline = inline_cpu(|| assert_eq!(scan(dir.path()), 5_000));

    let d = dir.path().to_str().unwrap().to_string();
    let held = held(async {
        ls::Ls::default()
            .run(json!({ "path": d, "limit": 5000 }))
            .await
            .unwrap()
    })
    .await;
    assert_yields(
        "ls",
        &held,
        baseline,
        "Its `read_dir` + per-entry `metadata`, and the sort and render over them, belong on \
         `spawn_blocking`.",
    );
}

/// The probe has teeth: the same scan done inline in a future holds the executor for about the
/// whole baseline, and never yields, so `assert_yields` would fail it.
#[tokio::test(flavor = "current_thread")]
async fn the_probe_catches_a_tool_that_works_inline() {
    let dir = wide_dir();
    let baseline = inline_cpu(|| assert_eq!(scan(dir.path()), 5_000));
    let held = held(async { scan(dir.path()) }).await;
    assert_eq!(held.pendings, 0);
    assert!(
        held.cpu.as_secs_f64() >= baseline.as_secs_f64() * MAX_HELD_FRACTION * 2.0,
        "an inline scan held {:?} against a {baseline:?} baseline",
        held.cpu
    );
}
