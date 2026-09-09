//! Per-session packing density of the multi-tenant `serve --listen` daemon.
//!
//! Each session is a task on the process-wide Tokio runtime (not a dedicated OS thread +
//! current-thread executor) and, by default, shares one upstream HTTP pool. The load-bearing CI
//! assertion is that **thread count does not grow one-for-one with sessions**. RSS / CPU / p95
//! `get_state` latency across a sweep of session counts are printed by the ignored bench so a
//! regression is visible without baking machine-specific RSS floors into CI.
//!
//! Run the sweep:
//! ```text
//! cargo test -p beyond-ai-agent --test serve_session_density density_sweep -- --ignored --nocapture
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, TestWs, free_port, spawn_model_server, turn_text,
    wait_for_port, ws_connect, ws_read_until_response, ws_send,
};
use serde_json::json;

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

fn serve_ws(base: &str, session_dir: &str, port: u16) -> ChildGuard {
    Command::new(BIN)
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
            // Keep every session for the measurement window; we hold the sockets anyway.
            "--session-idle-timeout",
            "0",
        ])
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

#[derive(Clone, Copy, Debug)]
struct ProcSnap {
    rss_kb: u64,
    threads: u64,
    cpu_ticks: u64,
}

fn proc_snap(pid: u32) -> ProcSnap {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let mut rss_kb = 0;
    let mut threads = 0;
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            rss_kb = v
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("Threads:") {
            threads = v
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
        }
    }
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    // /proc/pid/stat: comm can contain spaces/parens. Split on the last `)` then fields
    // 12/13 of the remainder are utime/stime (1-indexed fields 14/15 of the whole record).
    let cpu_ticks = stat
        .rsplit_once(')')
        .map(|(_, rest)| rest)
        .unwrap_or("")
        .split_whitespace()
        .nth(11)
        .and_then(|u| u.parse::<u64>().ok())
        .unwrap_or(0)
        + stat
            .rsplit_once(')')
            .map(|(_, rest)| rest)
            .unwrap_or("")
            .split_whitespace()
            .nth(12)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
    ProcSnap {
        rss_kb,
        threads,
        cpu_ticks,
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

async fn attach_idle(port: u16, n: usize) -> Vec<TestWs> {
    let mut sockets = Vec::with_capacity(n);
    for i in 0..n {
        sockets.push(ws_connect(port, Some(&format!("dens-{i}"))).await);
    }
    sockets
}

async fn ping_all(sockets: &mut [TestWs], rounds: usize) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(sockets.len() * rounds);
    for _ in 0..rounds {
        for (i, ws) in sockets.iter_mut().enumerate() {
            let t0 = Instant::now();
            ws_send(ws, json!({ "type": "get_state", "id": format!("g{i}") })).await;
            let frames = ws_read_until_response(ws, "get_state").await;
            samples.push(t0.elapsed());
            assert_eq!(
                frames.last().unwrap()["success"],
                true,
                "get_state failed: {:?}",
                frames.last()
            );
        }
    }
    samples
}

/// Idle sessions must not each take an OS thread. A reversion to thread-per-session would add ~N
/// threads for N sessions; the shared runtime adds only the (fixed) worker/blocking pool.
#[tokio::test]
async fn idle_sessions_do_not_take_a_thread_each() {
    let (base, _r) = spawn_model_server(vec![turn_text("unused")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);

    let pid = child.id();
    // One throwaway connection so the daemon has completed its first accept/handshake path.
    let warmup = ws_connect(port, Some("dens-warmup")).await;
    drop(warmup);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let baseline = proc_snap(pid);

    const N: usize = 8;
    let sockets = attach_idle(port, N).await;
    // Let session tasks finish Persistence::open / agent build.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let loaded = proc_snap(pid);

    let extra_threads = loaded.threads.saturating_sub(baseline.threads);
    assert!(
        extra_threads < N as u64,
        "session count grew threads one-for-one (shared-runtime regression): \
         baseline_threads={} loaded_threads={} extra={extra_threads} N={N} rss_kb {}→{}",
        baseline.threads,
        loaded.threads,
        baseline.rss_kb,
        loaded.rss_kb
    );

    drop(sockets);
    let _ = child.kill();
    let _ = child.wait();
}

/// Print RSS / CPU / throughput / p95 `get_state` latency across session counts. Ignored in CI:
/// the numbers are the measurement, and RSS floors are host-dependent. The thread invariant above
/// is the regression gate.
#[tokio::test]
#[ignore = "density sweep: cargo test -p beyond-ai-agent --test serve_session_density density_sweep -- --ignored --nocapture"]
async fn density_sweep() {
    let (base, _r) = spawn_model_server(vec![turn_text("unused")]);
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let mut child = serve_ws(&base, dir.path().to_str().unwrap(), port);
    wait_for_port(port);
    let pid = child.id();

    let ticks_per_sec: u64 = 100; // Linux USER_HZ; RSS/thread counts are the load-bearing columns.

    println!("pid={pid} ticks_per_sec={ticks_per_sec}");
    println!("N rss_kb rss_per_session_kb threads cpu_ms get_state_n get_state_qps p50_us p95_us");

    let mut sockets: Vec<TestWs> = Vec::new();
    let mut last_cpu = proc_snap(pid).cpu_ticks;
    let mut last_cpu_at = Instant::now();
    for n in [1usize, 8, 16, 32] {
        while sockets.len() < n {
            let i = sockets.len();
            sockets.push(ws_connect(port, Some(&format!("dens-{i}"))).await);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;

        const ROUNDS: usize = 8;
        let mut samples = ping_all(&mut sockets, ROUNDS).await;
        samples.sort();
        let wall = samples.iter().sum::<Duration>();
        let qps = if wall.is_zero() {
            0.0
        } else {
            samples.len() as f64 / wall.as_secs_f64()
        };

        let snap = proc_snap(pid);
        let cpu_dt = last_cpu_at.elapsed().as_secs_f64().max(1e-6);
        let cpu_ms =
            (snap.cpu_ticks.saturating_sub(last_cpu) as f64 / ticks_per_sec as f64) * 1000.0;
        // cpu_ms is since the previous N; report utilization over that window as well via the
        // printed cpu_ms (absolute ticks converted). Reset for the next step.
        let _ = cpu_dt;
        last_cpu = snap.cpu_ticks;
        last_cpu_at = Instant::now();

        println!(
            "{n} {} {:.1} {} {:.1} {} {:.1} {} {}",
            snap.rss_kb,
            snap.rss_kb as f64 / n as f64,
            snap.threads,
            cpu_ms,
            samples.len(),
            qps,
            percentile(&samples, 0.50).as_micros(),
            percentile(&samples, 0.95).as_micros(),
        );
    }

    drop(sockets);
    let _ = child.kill();
    let _ = child.wait();
}
