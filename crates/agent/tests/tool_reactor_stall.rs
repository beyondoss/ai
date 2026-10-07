// Test target: `.unwrap()` asserts preconditions; that's the point.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! A tool must not pin the async runtime: not with its CPU work, and not by blocking the thread.
//!
//! `serve_ws` runs every session as a task on one **shared** runtime, so a tool that does its work
//! inline — rather than handing it to `spawn_blocking` — stalls a worker that other sessions also
//! use. Nothing else on that worker gets polled meanwhile: not the outbound event pump, and not the
//! stdin/WebSocket command loop that carries `abort` and `steer`. `serve.rs`'s `persist_blocking`
//! already moved `sync_all` off the reactor for exactly this reason, and its doc comment says so.
//!
//! Two probes, each on a `current_thread` runtime, check two different things:
//!
//! - **CPU held** ([`held`]): the *thread CPU time* the runtime thread spends inside the tool
//!   future's `poll`s, against a baseline of the same work done inline in the same run. It catches
//!   a tool doing its CPU work on the executor. It cannot see a tool that blocks the thread without
//!   using CPU — `std::thread::sleep`, blocking disk or NFS I/O, a contended `std` mutex — since a
//!   blocked thread accrues no CPU time.
//! - **Responsiveness** ([`stalls`]): a ticker task on the same runtime, waking every millisecond
//!   while the tool runs; the worst gap between its ticks is how long the runtime could not poll
//!   anything else, by CPU work or by blocking alike. Each gap has the time the thread spent
//!   *waiting on a run queue* (preempted: `/proc/thread-self/schedstat`) taken out, so a loaded host
//!   cannot make a yielding tool look like a blocking one — the trap a plain wall-clock ticker fell
//!   into before. What is left is time the thread ran, or blocked, without returning to the
//!   executor; it must stay under [`MAX_UNRESPONSIVE`].
//!
//! Every tool in `beyond_ai_agent::tools` that runs without an external service is put through the
//! responsiveness probe with a workload large enough that inline work would show
//! (`*_keeps_the_runtime_responsive`); the CPU probe covers the file tools. MCP tools are
//! network clients over rmcp's async transports, and `subagent` runs a nested agent loop on the same
//! runtime: neither is probed here. `the_probe_catches_a_tool_that_works_inline` and
//! `the_responsiveness_probe_catches_a_tool_that_blocks_without_cpu` keep both probes' teeth
//! honest.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use agent_core::Tool;
use beyond_ai_agent::tools::{bash, edit, find, grep, ls, memory, read, todo, web, write};
use serde_json::json;

/// The longest the runtime may go without polling anything else while a tool runs, preemption
/// excluded. A tool's own poll (parsing its input, handing work off, assembling the result) takes
/// well under a millisecond; every workload below would block for far longer if done inline.
const MAX_UNRESPONSIVE: Duration = Duration::from_millis(20);

/// How long this thread has waited on a run queue (preempted, runnable but not running) so far —
/// the second field of `/proc/thread-self/schedstat`. `None` where the kernel does not report it
/// (off Linux, or without scheduler statistics): the responsiveness probe is then **skipped**, not
/// run on plain wall clock, which a loaded host would make flake.
fn run_queue_wait() -> Option<Duration> {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(Duration::from_nanos)
}

/// One ticker sample: when, and the thread's run-queue wait so far (if the kernel reports it).
type Sample = (Instant, Option<Duration>);

/// The worst gap between consecutive ticker samples with the run-queue wait between them taken out;
/// `None` when any sample had no run-queue figure (see [`run_queue_wait`]).
fn worst_stretch(samples: &[Sample]) -> Option<Duration> {
    samples
        .windows(2)
        .map(|w| {
            let (waited_then, waited_now) = (w[0].1?, w[1].1?);
            Some((w[1].0 - w[0].0).saturating_sub(waited_now.saturating_sub(waited_then)))
        })
        .try_fold(Duration::ZERO, |worst, gap| Some(worst.max(gap?)))
}

/// Why a responsiveness probe did not run, said once per probe.
fn skip_without_schedstat(tool: &str) {
    eprintln!(
        "SKIP {tool}: /proc/thread-self/schedstat is unavailable here, so the responsiveness probe \
         cannot take preemption out of its measurement and is not run (the CPU probe still is)"
    );
}

/// Held for each test's whole run, setup included. Under `cargo test` the tests share one process,
/// and one test's large allocations and frees (building a tree of thousands of files, a 4 MB source)
/// make another's runtime thread wait in the kernel on the shared address space — time that is
/// neither preemption nor the tool's, and would read as a stall. (`nextest`, which CI uses, runs each
/// test in its own process anyway.)
static ONE_TEST_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn one_at_a_time() -> tokio::sync::MutexGuard<'static, ()> {
    ONE_TEST_AT_A_TIME.lock().await
}

/// The worst stretch, preemption excluded, that the runtime thread went without polling a ticker
/// task while `work` ran on the same `current_thread` runtime; `None` without schedstat.
async fn stalls<F: Future>(work: F) -> (F::Output, Option<Duration>) {
    let samples: Arc<Mutex<Vec<Sample>>> = Arc::default();
    let stop = Arc::new(AtomicBool::new(false));
    let ticker = tokio::spawn({
        let samples = samples.clone();
        let stop = stop.clone();
        async move {
            while !stop.load(Ordering::Relaxed) {
                samples
                    .lock()
                    .unwrap()
                    .push((Instant::now(), run_queue_wait()));
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    });
    // Let the ticker take its first sample before the work starts.
    tokio::task::yield_now().await;
    let out = work.await;
    samples
        .lock()
        .unwrap()
        .push((Instant::now(), run_queue_wait()));
    stop.store(true, Ordering::Relaxed);
    ticker.abort();
    let worst = worst_stretch(&samples.lock().unwrap());
    (out, worst)
}

fn assert_responsive(tool: &str, worst: Duration, why: &str) {
    eprintln!("PROBE {tool}: worst unresponsive stretch {worst:?}");
    assert!(
        worst < MAX_UNRESPONSIVE,
        "`{tool}` kept the current_thread runtime from polling anything else for {worst:?} \
         (preemption excluded): every other task on this worker — other sessions, the abort/steer \
         command loop — waited that long. {why}"
    );
}

/// [`assert_responsive`] on the least of up to three runs of `$attempt` (an expression evaluated
/// afresh for each run, so any per-run setup in it happens before the measurement starts). A tool
/// that works or blocks inline stalls the runtime on every run; the host's own one-off stalls of
/// the runtime thread (a page fault served from disk on a busy machine) do not repeat.
macro_rules! assert_responsive_least {
    ($tool:expr, $why:expr, $attempt:expr) => {{
        let mut least = Duration::MAX;
        for _ in 0..3 {
            let (_, worst) = stalls($attempt).await;
            let Some(worst) = worst else {
                skip_without_schedstat($tool);
                return;
            };
            least = least.min(worst);
            if least < MAX_UNRESPONSIVE {
                break;
            }
        }
        assert_responsive($tool, least, $why);
    }};
}

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
    let _serial = one_at_a_time().await;
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
    let _serial = one_at_a_time().await;
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
    let _serial = one_at_a_time().await;
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
    let _serial = one_at_a_time().await;
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

/// The responsiveness probe has teeth where the CPU probe has none: a tool that yields once and
/// then blocks the thread without using CPU (here `std::thread::sleep`, standing in for blocking
/// disk or NFS I/O or a contended `std` mutex) holds almost no CPU, so the CPU probe passes it —
/// and the runtime still could not poll anything for the whole sleep.
#[tokio::test(flavor = "current_thread")]
async fn the_responsiveness_probe_catches_a_tool_that_blocks_without_cpu() {
    let _serial = one_at_a_time().await;
    let blocking = || async {
        tokio::task::yield_now().await;
        std::thread::sleep(Duration::from_millis(150));
    };
    let cpu = held(blocking()).await;
    assert!(cpu.pendings > 0);
    assert!(
        cpu.cpu < Duration::from_millis(20),
        "the CPU probe cannot see it: {:?}",
        cpu.cpu
    );
    let ((), worst) = stalls(blocking()).await;
    let Some(worst) = worst else {
        return skip_without_schedstat("the probe's own teeth");
    };
    assert!(
        worst >= Duration::from_millis(100),
        "the responsiveness probe must: worst stretch {worst:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn edit_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subject.rs");
    let src = big_ascii_source(80_000);
    let tool = edit::Edit::new(dir.path());
    let p = path.to_str().unwrap().to_string();
    assert_responsive_least!("edit", "Its file I/O belongs on `spawn_blocking`.", {
        // Fresh for each run (an edit changes the file), and before the measurement starts.
        std::fs::write(&path, &src).unwrap();
        async {
            tool.run(json!({ "path": p, "old_string": OLD, "new_string": NEW }))
                .await
                .unwrap()
        }
    });
}

#[tokio::test(flavor = "current_thread")]
async fn write_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.rs");
    let input = json!({ "path": path.to_str().unwrap(), "content": big_ascii_source(80_000) });
    let tool = write::Write::new(dir.path());
    assert_responsive_least!("write", "Its file I/O belongs on `spawn_blocking`.", {
        let input = input.clone();
        async { tool.run(input).await.unwrap() }
    });
}

#[tokio::test(flavor = "current_thread")]
async fn ls_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = wide_dir();
    let d = dir.path().to_str().unwrap().to_string();
    assert_responsive_least!(
        "ls",
        "Its directory scan belongs on `spawn_blocking`.",
        async {
            ls::Ls::default()
                .run(json!({ "path": d, "limit": 5000 }))
                .await
                .unwrap()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn read_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.rs");
    std::fs::write(&path, big_ascii_source(80_000)).unwrap();
    let p = path.to_str().unwrap().to_string();
    let tool = read::Read::new(dir.path());
    assert_responsive_least!("read", "Its file I/O belongs on `spawn_blocking`.", async {
        tool.run(json!({ "path": p, "offset": 70_000, "limit": 2000 }))
            .await
            .unwrap()
    });
}

/// A tree of `files` source files, for the search tools.
fn source_tree(files: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..files {
        let sub = dir.path().join(format!("d{}", i % 50));
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join(format!("f{i}.rs")), big_ascii_source(200)).unwrap();
    }
    dir
}

#[tokio::test(flavor = "current_thread")]
async fn grep_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = source_tree(2_000);
    let tool = grep::Grep::new(dir.path());
    assert_responsive_least!(
        "grep",
        "Its walk and match belong on `spawn_blocking`.",
        async {
            tool.run(json!({ "pattern": "x_199 = compute", "limit": 5000 }))
                .await
                .unwrap()
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn find_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = source_tree(2_000);
    let tool = find::Find::new(dir.path());
    assert_responsive_least!("find", "Its walk belongs on `spawn_blocking`.", async {
        tool.run(json!({ "pattern": "**/*.rs", "limit": 5000 }))
            .await
            .unwrap()
    });
}

#[tokio::test(flavor = "current_thread")]
async fn bash_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = tempfile::tempdir().unwrap();
    let tool = bash::Bash::real().with_root(dir.path());
    assert_responsive_least!(
        "bash",
        "Waiting on the child, reading its output and spilling it must not block the runtime.",
        async {
            let out = tool
                .run(json!({ "command": "sleep 0.3; seq 1 20000" }))
                .await
                .unwrap();
            assert!(out.text.contains("20000"), "{}", out.text);
        }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn todo_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let tool = todo::Todo::new();
    let todos: Vec<_> = (0..500)
        .map(|i| json!({ "content": format!("task {i}"), "activeForm": format!("doing task {i}"), "status": "pending" }))
        .collect();
    assert_responsive_least!(
        "todo",
        "It is in-memory; nothing here should block.",
        async { tool.run(json!({ "todos": todos })).await.unwrap() }
    );
}

#[tokio::test(flavor = "current_thread")]
async fn memory_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let dir = tempfile::tempdir().unwrap();
    // Every line a hit: a search returning hundreds of thousands of lines, whose rendering (and
    // freeing) inline would hold the runtime for tens of milliseconds.
    for i in 0..2_000 {
        std::fs::write(
            dir.path().join(format!("note_{i}.md")),
            format!(
                "# note {i}\n{}",
                "some remembered fact about the project\n".repeat(200)
            ),
        )
        .unwrap();
    }
    let backend = Arc::new(beyond_ai_agent::memory::file::FileBackend::at(
        dir.path().to_path_buf(),
    ));
    std::fs::write(dir.path().join("big.md"), big_ascii_source(80_000)).unwrap();
    let tool = memory::Memory::new(backend);
    assert_responsive_least!(
        "memory search",
        "Its file I/O, and rendering a large hit list, belong on `spawn_blocking`.",
        async {
            tool.run(json!({ "command": "search", "query": "remembered fact" }))
                .await
                .unwrap()
        }
    );
    assert_responsive_least!(
        "memory view",
        "Its file I/O, and rendering a large document, belong on `spawn_blocking`.",
        async {
            tool.run(json!({ "command": "view", "path": "/memories/big.md" }))
                .await
                .unwrap()
        }
    );
}

/// A local HTTP server answering every request with `body` as HTML.
async fn html_server(body: String) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener.into_std().unwrap()).unwrap();
            while let Ok((mut stream, _)) = listener.accept().await {
                let body = body.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes()).await;
                    let _ = stream.write_all(body.as_bytes()).await;
                });
            }
        });
    });
    format!("http://{addr}/")
}

/// One `markdown` fetch of `url`, which must have rendered the page.
async fn web_markdown(tool: &web::Web, url: &str) {
    let out = tool
        .run(json!({ "url": url, "mode": "markdown" }))
        .await
        .unwrap();
    assert!(
        out.text.contains("paragraph"),
        "{}",
        &out.text[..out.text.len().min(300)]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn web_keeps_the_runtime_responsive() {
    let _serial = one_at_a_time().await;
    let mut html = String::from("<html><body>");
    for i in 0..20_000 {
        html.push_str(&format!(
            "<p>paragraph {i} with <a href='/x{i}'>a link</a></p>"
        ));
    }
    html.push_str("</body></html>");
    let url = html_server(html).await;
    // The isolated parse runs in the agent binary's `__web-parse` child, not in this test binary.
    let fresh =
        || web::Web::new(true, &[], None).with_parser_binary(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    let why = "Fetching is async; building the client, and parsing in (and waiting on) the \
               isolated child, belong on the blocking pool.";
    // A first call builds the client (TLS provider, root certificates): a fresh tool each run.
    let tools: Vec<web::Web> = (0..3).map(|_| fresh()).collect();
    let mut cold = tools.iter();
    assert_responsive_least!(
        "web (first call)",
        why,
        web_markdown(cold.next().unwrap(), &url)
    );
    // Every call parses the page in an isolated child and waits on it.
    let warm = fresh();
    web_markdown(&warm, &url).await;
    assert_responsive_least!("web", why, web_markdown(&warm, &url));
}

/// `execute` (Code Mode) runs QuickJS's synchronous evaluation: a busy script must hold a
/// blocking-pool thread, never the runtime thread — whatever else this worker runs (other sessions,
/// their progress and aborts) keeps being polled while it spins.
#[cfg(feature = "code-mode")]
#[tokio::test(flavor = "current_thread")]
async fn code_mode_keeps_the_runtime_responsive_while_a_script_spins() {
    let _serial = one_at_a_time().await;
    let tool = beyond_ai_agent::tools::code_mode::Execute::new(vec![]);
    let spin_of = |n: u64| json!({ "code": format!("let x = 0; for (let i = 0; i < {n}; i++) {{ x += i % 7; }} return x;") });
    // Long enough to show, whatever this build's QuickJS speed: doubled until a run takes 200 ms.
    let mut n = 100_000u64;
    let ran = loop {
        let started = Instant::now();
        tool.run(spin_of(n)).await.unwrap();
        let ran = started.elapsed();
        if ran >= Duration::from_millis(200) || n >= 1 << 30 {
            break ran;
        }
        n *= 2;
    };
    eprintln!("PROBE execute spins {n} iterations in {ran:?}");
    let spin = spin_of(n);
    assert_responsive_least!(
        "execute",
        "QuickJS evaluates synchronously; it belongs on the blocking pool.",
        async { tool.run(spin.clone()).await.unwrap() }
    );
    let held = held(async { tool.run(spin).await.unwrap() }).await;
    assert!(
        held.cpu < Duration::from_millis(20),
        "`execute` spent {:?} of the runtime thread's CPU on a script that spins for {ran:?}",
        held.cpu
    );
}

/// An abort still stops a spinning script promptly — by its cancellation token, or by the call
/// being dropped — and frees the process's one JS slot, so the next `execute` runs at once rather
/// than after the 30 s deadline.
#[cfg(feature = "code-mode")]
#[tokio::test(flavor = "current_thread")]
async fn code_mode_abort_interrupts_a_spinning_script_promptly() {
    use agent_core::tool::ToolProgress;
    let _serial = one_at_a_time().await;
    let tool = beyond_ai_agent::tools::code_mode::Execute::new(vec![]);
    let forever = json!({ "code": "while (true) {}" });
    let quick = json!({ "code": "return 'next';" });

    // Cancelled through the call's token, from another task on this runtime.
    let token = tokio_util::sync::CancellationToken::new();
    let (tx, _rx) = futures::channel::mpsc::unbounded();
    let progress = ToolProgress::new(tx, "x".into(), "execute".into(), token.clone());
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        token.cancel();
    });
    let started = Instant::now();
    let err = tool
        .run_streaming(forever.clone(), &progress)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("cancelled"), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    canceller.await.unwrap();
    let next = tokio::time::timeout(Duration::from_secs(5), tool.run(quick.clone()))
        .await
        .expect("the slot is free again: the cancelled script was interrupted")
        .unwrap();
    assert_eq!(next.text, "next");

    // Dropped mid-run, as an abort that drops the tool future does.
    let dropped = tokio::time::timeout(Duration::from_millis(200), tool.run(forever)).await;
    assert!(dropped.is_err(), "still spinning when dropped");
    let next = tokio::time::timeout(Duration::from_secs(5), tool.run(quick))
        .await
        .expect("the slot is free again: the dropped script was interrupted")
        .unwrap();
    assert_eq!(next.text, "next");
}

/// All the CPU this (`current_thread`) runtime's thread spends while `work` runs — in `work`'s own
/// polls and in any task it spawns onto this runtime — so per-byte work moved into a spawned task
/// is seen too.
async fn runtime_thread_cpu<F: Future>(work: F) -> Duration {
    let start = thread_cpu();
    work.await;
    thread_cpu() - start
}

/// `edit`'s cost on the runtime thread does not grow with the file: validating it as UTF-8 and
/// handing its new contents to the writer are per-byte work, done on the blocking pool (the writer
/// takes the buffer, it does not copy it here).
#[tokio::test(flavor = "current_thread")]
async fn edit_does_no_per_byte_work_on_the_runtime_thread() {
    let _serial = one_at_a_time().await;
    let dir = tempfile::tempdir().unwrap();
    let tool = edit::Edit::new(dir.path());
    let mut cost = Vec::new();
    for lines in [80_000usize, 640_000] {
        let path = dir.path().join(format!("subject_{lines}.rs"));
        let src = big_ascii_source(lines);
        let p = path.to_str().unwrap().to_string();
        // The least of three, each on a fresh copy: per-byte work shows on every run.
        let mut least = Duration::MAX;
        for _ in 0..3 {
            std::fs::write(&path, &src).unwrap();
            let spent = runtime_thread_cpu(async {
                tool.run(json!({ "path": p, "old_string": "let x_40001 = compute(i, 40001)", "new_string": "let x_40001 = compute(i, 40001) /* edited */" }))
                    .await
                    .unwrap();
            })
            .await;
            least = least.min(spent);
        }
        eprintln!(
            "PROBE edit {} MB: runtime thread {least:?}",
            src.len() >> 20
        );
        cost.push(least);
    }
    assert!(
        cost[1] < cost[0] + Duration::from_micros(600),
        "an 8x larger file cost the runtime thread {:?} against {:?}: per-byte work is on it",
        cost[1],
        cost[0]
    );
}

/// Without scheduler statistics the responsiveness probe reports "unavailable" (and its tests
/// skip, saying so) rather than measuring plain wall clock, which counts every preemption and would
/// make a loaded host's run flake.
#[test]
fn the_responsiveness_probe_is_unavailable_without_schedstat_not_wall_clock() {
    let t = Instant::now();
    let later = t + Duration::from_millis(50);
    assert_eq!(
        worst_stretch(&[(t, None), (later, None)]),
        None,
        "no run-queue figure: no measurement"
    );
    assert_eq!(
        worst_stretch(&[(t, Some(Duration::ZERO)), (later, None)]),
        None
    );
    // With it, the run-queue wait is taken out.
    assert_eq!(
        worst_stretch(&[
            (t, Some(Duration::from_millis(1))),
            (later, Some(Duration::from_millis(31)))
        ]),
        Some(Duration::from_millis(20))
    );
}
