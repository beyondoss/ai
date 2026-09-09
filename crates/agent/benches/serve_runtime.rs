// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Process-level Tokio runtime A/B for `serve --listen`.
//!
//! Every WebSocket session is a task on this process runtime (`serve_ws` `tokio::spawn`s
//! `serve_session`; the future is `Send`). Extra work-stealing workers therefore run real session
//! work, not just the accept loop — they also cost per-thread stacks and mimalloc heaps. This bench
//! A/Bs `current_thread` (the production default in `main.rs::build_runtime`) against multi-thread.
//!
//! It spawns the real `beyond-ai-agent` binary twice — once per flavor, selected by
//! `BEYOND_AI_AGENT_TOKIO_WORKER_THREADS` — and drives concurrent WebSocket sessions against a
//! loopback mock. The two children are the same binary, same flags, same load; the scheduler is
//! the only variable. Metrics are process-level (the child's `/proc/<pid>/status` + `/stat`), not
//! in-process allocator counts: thread stacks and per-thread heaps never show up in a
//! `divan::AllocProfiler` of the bench itself.
//!
//! ```text
//! cargo bench -p beyond-ai-agent --bench serve_runtime
//! ```
//!
//! Overrides (optional): `SERVE_RUNTIME_BENCH_CONCURRENCY` (default 32),
//! `SERVE_RUNTIME_BENCH_DURATION_SECS` (default 4).
//!
//! Flavors:
//! - `current_thread` — `BEYOND_AI_AGENT_TOKIO_WORKER_THREADS=1` (production default)
//! - `multi_thread`   — `BEYOND_AI_AGENT_TOKIO_WORKER_THREADS=0` (one worker per core, the
//!   previous default)

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;

const WS_PATH: &str = "/_beyond/agent";
const HOME: &str = "/nonexistent-beyond-ai-agent-bench-home";

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let concurrency = env_u64("SERVE_RUNTIME_BENCH_CONCURRENCY", 32) as usize;
    let duration = Duration::from_secs(env_u64("SERVE_RUNTIME_BENCH_DURATION_SECS", 4));
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    eprintln!(
        "serve_runtime: bin={bin} cores={cores} concurrency={concurrency} duration={duration:?}"
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(run(bin, concurrency, duration, cores));
}

async fn run(bin: &str, concurrency: usize, duration: Duration, cores: usize) {
    let tmp = tempfile::tempdir().unwrap();
    let gateway = spawn_mock_gateway();

    let mut rows = Vec::new();
    for flavor in [Flavor::MultiThread, Flavor::CurrentThread] {
        eprintln!("serve_runtime: measuring {flavor}…");
        let row = measure(bin, flavor, &gateway, tmp.path(), concurrency, duration).await;
        eprintln!(
            "  {flavor}: idle_rss={:.2} MiB  attached_rss={:.2} MiB  peak_rss={:.2} MiB  \
             threads={}  rpc_rps={:.0}  rpc_p95={:.2} ms  prompt_rps={:.1}  prompt_p95={:.1} ms  \
             cpu={:.1}%",
            row.idle_rss_mi,
            row.attached_rss_mi,
            row.peak_rss_mi,
            row.attached_threads,
            row.rpc.rps,
            row.rpc.p95_ms,
            row.prompt.rps,
            row.prompt.p95_ms,
            row.rpc_cpu_pct,
        );
        rows.push(row);
    }

    print_table(&rows, cores);
    print_verdict(&rows);
}

#[derive(Clone, Copy)]
enum Flavor {
    CurrentThread,
    MultiThread,
}

impl Flavor {
    fn env_value(self) -> &'static str {
        match self {
            Self::CurrentThread => "1",
            Self::MultiThread => "0",
        }
    }
}

impl std::fmt::Display for Flavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::CurrentThread => "current_thread",
            Self::MultiThread => "multi_thread",
        })
    }
}

struct Load {
    ok: u64,
    err: u64,
    rps: f64,
    p50_ms: f64,
    p95_ms: f64,
}

struct Row {
    flavor: Flavor,
    idle_rss_mi: f64,
    attached_rss_mi: f64,
    peak_rss_mi: f64,
    attached_threads: u64,
    rpc: Load,
    prompt: Load,
    rpc_cpu_pct: f64,
}

async fn measure(
    bin: &str,
    flavor: Flavor,
    gateway: &str,
    cwd: &Path,
    concurrency: usize,
    duration: Duration,
) -> Row {
    let serve = spawn_serve(bin, flavor, gateway, cwd);
    let pid = serve.child.id();
    // Let the runtime finish starting workers / parking them before the idle sample.
    tokio::time::sleep(Duration::from_millis(250)).await;
    let idle = proc_snap(pid).unwrap_or_default();

    let mut sockets = connect_all(serve.port, concurrency).await;
    let attached = proc_snap(pid).unwrap_or_default();

    let sampler = RssSampler::start(pid);
    let cpu_before = proc_snap(pid).unwrap_or_default();
    let rpc = drive(&mut sockets, duration, Workload::GetState).await;
    let cpu_after = proc_snap(pid).unwrap_or_default();
    let prompt = drive(&mut sockets, duration, Workload::Prompt).await;
    let peak_kb = sampler.stop().max(cpu_after.rss_kb).max(attached.rss_kb);

    // Drop kills the child; do it before the next flavor starts so ports/RSS don't overlap.
    drop(sockets);
    drop(serve);

    let wall = duration.as_secs_f64().max(0.001);
    let cpu_ticks = cpu_after.cpu_ticks.saturating_sub(cpu_before.cpu_ticks);
    Row {
        flavor,
        idle_rss_mi: kb_to_mi(idle.rss_kb),
        attached_rss_mi: kb_to_mi(attached.rss_kb),
        peak_rss_mi: kb_to_mi(peak_kb.max(cpu_after.hwm_kb)),
        attached_threads: attached.threads,
        rpc_cpu_pct: ticks_to_cpu_pct(cpu_ticks, wall),
        rpc,
        prompt,
    }
}

#[derive(Clone, Copy, Debug)]
enum Workload {
    GetState,
    Prompt,
}

async fn connect_all(port: u16, concurrency: usize) -> Vec<Ws> {
    let mut sockets = Vec::with_capacity(concurrency);
    for i in 0..concurrency {
        let sid = format!("b{i:04}");
        sockets.push(
            ws_connect(port, &sid)
                .await
                .unwrap_or_else(|_| panic!("websocket connect to session {sid}")),
        );
    }
    sockets
}

async fn drive(sockets: &mut Vec<Ws>, duration: Duration, workload: Workload) -> Load {
    // One untimed round-trip per socket so session spawn / first-turn setup isn't in the RPS.
    // Parallel: 32 sequential prompts would dominate the wall clock before the timed window starts.
    {
        let handles: Vec<_> = sockets
            .drain(..)
            .enumerate()
            .map(|(i, mut ws)| {
                tokio::spawn(async move {
                    let id = format!("warm-{i}");
                    let (cmd, command) = command_for(workload, &id);
                    tokio::time::timeout(Duration::from_secs(20), roundtrip(&mut ws, cmd, command))
                        .await
                        .unwrap_or_else(|_| panic!("{workload:?} warmup timed out on socket {i}"))
                        .unwrap_or_else(|_| panic!("{workload:?} warmup failed on socket {i}"));
                    ws
                })
            })
            .collect();
        for h in handles {
            sockets.push(h.await.expect("warmup task panicked"));
        }
    }

    let latencies = Arc::new(Mutex::new(Vec::<Duration>::new()));
    let ok = Arc::new(AtomicU64::new(0));
    let err = Arc::new(AtomicU64::new(0));
    let deadline = Instant::now() + duration;
    let start = Instant::now();

    // Own each socket on its own task so the load generator can use the parent multi-thread
    // runtime; a single-task `join_all` would serialize client readiness on one worker.
    let handles: Vec<_> = sockets
        .drain(..)
        .enumerate()
        .map(|(i, mut ws)| {
            let latencies = latencies.clone();
            let ok = ok.clone();
            let err = err.clone();
            tokio::spawn(async move {
                worker(&mut ws, i, deadline, workload, latencies, ok, err).await;
                ws
            })
        })
        .collect();
    let mut back = Vec::with_capacity(handles.len());
    for h in handles {
        back.push(h.await.expect("load worker task panicked"));
    }
    *sockets = back;

    let wall = start.elapsed().as_secs_f64().max(0.001);
    let mut samples = latencies.lock().unwrap().clone();
    samples.sort_unstable();
    Load {
        ok: ok.load(Ordering::Relaxed),
        err: err.load(Ordering::Relaxed),
        rps: ok.load(Ordering::Relaxed) as f64 / wall,
        p50_ms: percentile_ms(&samples, 0.50),
        p95_ms: percentile_ms(&samples, 0.95),
    }
}

async fn worker(
    ws: &mut Ws,
    idx: usize,
    deadline: Instant,
    workload: Workload,
    latencies: Arc<Mutex<Vec<Duration>>>,
    ok: Arc<AtomicU64>,
    err: Arc<AtomicU64>,
) {
    let mut n = 0u64;
    while Instant::now() < deadline {
        let id = format!("{:x}-{n}", idx);
        let (cmd, command) = command_for(workload, &id);
        let t0 = Instant::now();
        match roundtrip(ws, cmd, command).await {
            Ok(()) => {
                latencies.lock().unwrap().push(t0.elapsed());
                ok.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                err.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
        n += 1;
    }
}

fn command_for(workload: Workload, id: &str) -> (Value, &'static str) {
    match workload {
        Workload::GetState => (json!({ "type": "get_state", "id": id }), "get_state"),
        Workload::Prompt => (
            json!({ "type": "prompt", "id": id, "message": "hi" }),
            "prompt",
        ),
    }
}

type Ws =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn ws_connect(port: u16, session_id: &str) -> Result<Ws, ()> {
    let url = format!("ws://127.0.0.1:{port}{WS_PATH}?session_id={session_id}");
    let (ws, _) = tokio_tungstenite::connect_async(url)
        .await
        .map_err(|_| ())?;
    Ok(ws)
}

async fn roundtrip(ws: &mut Ws, cmd: Value, command: &str) -> Result<(), ()> {
    ws.send(Message::Text(cmd.to_string().into()))
        .await
        .map_err(|_| ())?;
    let want_id = cmd.get("id").and_then(Value::as_str).map(str::to_owned);
    loop {
        let msg = match tokio::time::timeout(Duration::from_secs(20), ws.next()).await {
            Ok(Some(Ok(m))) => m,
            _ => return Err(()),
        };
        let Message::Text(text) = msg else { continue };
        let v: Value = serde_json::from_str(&text).map_err(|_| ())?;
        if v["type"] != "response" {
            continue;
        }
        if v["command"] != command {
            continue;
        }
        if let Some(id) = &want_id
            && v.get("id").is_some()
            && v["id"] != *id
        {
            continue;
        }
        if v["success"] == false {
            return Err(());
        }
        return Ok(());
    }
}

struct ServeProc {
    child: Child,
    port: u16,
}

impl Drop for ServeProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_serve(bin: &str, flavor: Flavor, gateway: &str, cwd: &Path) -> ServeProc {
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--listen",
            "127.0.0.1:0",
            "--gateway-url",
            gateway,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--no-memory",
            "--no-tools",
        ])
        .env("HOME", HOME)
        .env("BEYOND_AI_AGENT_TOKIO_WORKER_THREADS", flavor.env_value())
        .env_remove("AI_GATEWAY_URL")
        .env_remove("AI_AGENT_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {bin}: {e}"));

    match wait_for_listen(&mut child) {
        Ok(port) => ServeProc { child, port },
        Err(log) => {
            let _ = child.kill();
            let status = child.wait();
            panic!("serve ({flavor}) failed to listen ({status:?}):\n{log}");
        }
    }
}

fn wait_for_listen(child: &mut Child) -> Result<u16, String> {
    let stderr = child.stderr.take().expect("piped stderr");
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let _ = tx.send(line);
        }
    });

    let deadline = Instant::now() + Duration::from_secs(15);
    let mut log = String::new();
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remaining.min(Duration::from_millis(50))) {
            Ok(line) => {
                log.push_str(&line);
                log.push('\n');
                if let Some(port) = parse_listen_port(&line) {
                    return Ok(port);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(status)) = child.try_wait() {
                    return Err(format!("exited {status} before listen:\n{log}"));
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.try_wait();
                return Err(format!("stderr closed before listen:\n{log}"));
            }
        }
    }
    Err(format!("timed out waiting for listen line:\n{log}"))
}

struct RssSampler {
    stop: Arc<std::sync::atomic::AtomicBool>,
    peak_kb: Arc<AtomicU64>,
    join: Option<thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start(pid: u32) -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak_kb = Arc::new(AtomicU64::new(0));
        let join = thread::spawn({
            let stop = stop.clone();
            let peak_kb = peak_kb.clone();
            move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Some(snap) = proc_snap(pid) {
                        peak_kb.fetch_max(snap.rss_kb, Ordering::Relaxed);
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            }
        });
        Self {
            stop,
            peak_kb,
            join: Some(join),
        }
    }

    fn stop(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
        self.peak_kb.load(Ordering::Relaxed)
    }
}

fn parse_listen_port(line: &str) -> Option<u16> {
    let rest = line.strip_prefix("serve: websocket listening on ")?;
    let addr = rest.split_whitespace().next()?;
    addr.rsplit_once(':')?.1.parse().ok()
}

/// Unbounded Anthropic-SSE mock: every request gets the same one-turn text reply, concurrently.
/// Prompt load would serialize behind a FIFO mock and measure the mock, not `serve`.
fn spawn_mock_gateway() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    let body = anthropic_text_turn("ok");
    thread::spawn(move || {
        listener.set_nonblocking(false).unwrap();
        for conn in listener.incoming() {
            let Ok(stream) = conn else { break };
            let body = body.clone();
            thread::spawn(move || {
                use std::io::Write;
                let mut stream = stream;
                drain_http_request(&mut stream);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
                );
                let _ = stream.write_all(http.as_bytes());
            });
        }
    });
    format!("http://{addr}")
}

/// Read headers + `Content-Length` body so a large `prompt` request isn't reset mid-write — that
/// showed up as exactly `concurrency` prompt errors (one per session, once history exceeded one
/// `read()` of 4 KiB).
fn drain_http_request(stream: &mut std::net::TcpStream) {
    use std::io::Read;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
            let len = headers
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")
                        .and_then(|v| v.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            let need = pos + 4 + len;
            while buf.len() < need {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            return;
        }
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn anthropic_text_turn(text: &str) -> String {
    let events = [
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ];
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

#[derive(Default, Clone, Copy)]
struct ProcSnap {
    rss_kb: u64,
    hwm_kb: u64,
    threads: u64,
    cpu_ticks: u64,
}

fn proc_snap(pid: u32) -> Option<ProcSnap> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let mut rss_kb = 0;
    let mut hwm_kb = 0;
    let mut threads = 0;
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            rss_kb = parse_kb(v);
        } else if let Some(v) = line.strip_prefix("VmHWM:") {
            hwm_kb = parse_kb(v);
        } else if let Some(v) = line.strip_prefix("Threads:") {
            threads = v.split_whitespace().next()?.parse().ok()?;
        }
    }
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // After `comm`, fields start at kernel field 3 (`state`). utime=14, stime=15.
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(ProcSnap {
        rss_kb,
        hwm_kb,
        threads,
        cpu_ticks: utime + stime,
    })
}

fn parse_kb(v: &str) -> u64 {
    v.split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

fn kb_to_mi(kb: u64) -> f64 {
    kb as f64 / 1024.0
}

fn clk_tck() -> f64 {
    Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(100.0)
}

fn ticks_to_cpu_pct(ticks: u64, wall_secs: f64) -> f64 {
    if wall_secs <= 0.0 {
        return 0.0;
    }
    (ticks as f64 / clk_tck()) / wall_secs * 100.0
}

fn percentile_ms(sorted: &[Duration], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)].as_secs_f64() * 1e3
}

fn print_table(rows: &[Row], cores: usize) {
    println!();
    println!("serve --listen Tokio runtime A/B  (nproc={cores}, same binary, same load)");
    println!(
        "{:<16} {:>10} {:>12} {:>10} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "flavor",
        "idle_MiB",
        "attach_MiB",
        "peak_MiB",
        "threads",
        "rpc_rps",
        "rpc_p50ms",
        "rpc_p95ms",
        "prmpt_rps",
        "prmpt_p95",
        "cpu%"
    );
    for r in rows {
        println!(
            "{:<16} {:>10.2} {:>12.2} {:>10.2} {:>8} {:>8.0} {:>10.2} {:>10.2} {:>10.1} {:>10.1} {:>7.1}",
            r.flavor,
            r.idle_rss_mi,
            r.attached_rss_mi,
            r.peak_rss_mi,
            r.attached_threads,
            r.rpc.rps,
            r.rpc.p50_ms,
            r.rpc.p95_ms,
            r.prompt.rps,
            r.prompt.p95_ms,
            r.rpc_cpu_pct,
        );
        if r.rpc.err > 0 || r.prompt.err > 0 {
            println!("  errors: rpc={} prompt={}", r.rpc.err, r.prompt.err);
        }
    }
}

fn print_verdict(rows: &[Row]) {
    let multi = rows
        .iter()
        .find(|r| matches!(r.flavor, Flavor::MultiThread));
    let current = rows
        .iter()
        .find(|r| matches!(r.flavor, Flavor::CurrentThread));
    let (Some(multi), Some(current)) = (multi, current) else {
        return;
    };
    let rss_delta = multi.idle_rss_mi - current.idle_rss_mi;
    let attach_delta = multi.attached_rss_mi - current.attached_rss_mi;
    let peak_delta = multi.peak_rss_mi - current.peak_rss_mi;
    let rps_ratio = if multi.rpc.rps > 0.0 {
        current.rpc.rps / multi.rpc.rps
    } else {
        0.0
    };
    let p95_ratio = if multi.rpc.p95_ms > 0.0 {
        current.rpc.p95_ms / multi.rpc.p95_ms
    } else {
        0.0
    };
    let prompt_rps_ratio = if multi.prompt.rps > 0.0 {
        current.prompt.rps / multi.prompt.rps
    } else {
        0.0
    };
    let prompt_p95_ratio = if multi.prompt.p95_ms > 0.0 {
        current.prompt.p95_ms / multi.prompt.p95_ms
    } else {
        0.0
    };
    println!();
    println!(
        "current_thread vs multi_thread: idle RSS {rss_delta:+.2} MiB, attached RSS {attach_delta:+.2} MiB, \
         peak RSS {peak_delta:+.2} MiB, rpc rps ×{rps_ratio:.2}, rpc p95 ×{p95_ratio:.2}, \
         prompt rps ×{prompt_rps_ratio:.2}, prompt p95 ×{prompt_p95_ratio:.2}"
    );
    // "Meaningful regression": >10% less throughput or >15% worse p95 on either load.
    // Memory win is peak RSS under load — parked workers barely fault their stacks, so idle
    // RSS understates the mimalloc-per-thread cost that shows up once frames are moving.
    let regression =
        rps_ratio < 0.90 || p95_ratio > 1.15 || prompt_rps_ratio < 0.90 || prompt_p95_ratio > 1.15;
    if peak_delta > 0.5 && !regression {
        println!(
            "verdict: current_thread saves memory without meaningful regression — production default."
        );
    } else if regression {
        println!(
            "verdict: current_thread regresses load (rpc rps ×{rps_ratio:.2} p95 ×{p95_ratio:.2}, \
             prompt rps ×{prompt_rps_ratio:.2} p95 ×{prompt_p95_ratio:.2}); \
             inspect before keeping it as the default."
        );
    } else {
        println!(
            "verdict: peak RSS delta is noise ({peak_delta:+.2} MiB); either flavor is fine on this host."
        );
    }
    assert!(
        current.rpc.ok > 0 && multi.rpc.ok > 0,
        "rpc load produced no successful get_state round-trips"
    );
    assert!(
        current.prompt.ok > 0 && multi.prompt.ok > 0,
        "prompt load produced no successful prompt round-trips"
    );
}
