//! swarm_load — agent-swarm packing-density load test.
//!
//! Quantifies the PEAK RSS cost of the per-turn history deep-clone when many agent sessions are
//! mid-turn at the same instant — the OOM-risk / packing-density metric for densely-packed swarms in
//! cloud VMs.
//!
//! It reproduces the exact per-turn hot pattern of the agent loop, with no model call and no network:
//! a request snapshot of the history is held alive (`session.messages.clone()`) while the session
//! `push`es the next turn. Because the snapshot keeps the `Arc<Vec<Message>>` shared, `Session::push`'s
//! `Arc::make_mut` must clone the whole history vec. On `main` that clone deep-copies every message's
//! multi-KB payload bytes; on `jared/perf-audit` those payloads are `Arc<str>`, so the clone is a
//! refcount bump and the bytes are shared. The transient coexistence of N such copies — one per
//! concurrent mid-turn session — is what this measures via `VmHWM` (the kernel's peak-RSS high-water
//! mark).
//!
//! Uses ONLY the public agent-core API, so the SAME source compiles on both branches.
//!
//! Usage: `swarm_load <N> [WAVES]`  (N = concurrent sessions; WAVES defaults to 5)

// Load-test example, not production code: thread spawn/join and env/proc parsing use
// `expect`/`unwrap` on setup that failing means the measurement can't run anyway.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use agent_core::message::{ContentBlock, ImageSource, Message};
use agent_core::session::Session;

// --- History profile: the shape a real coding-agent session accumulates -------------------------
const HISTORY_MESSAGES: usize = 40; // total messages pre-grown per session
const TOOL_RESULT_BYTES: usize = 8 * 1024; // each multi-KB ToolResult body (read/bash/grep output)
const IMAGE_B64_BYTES: usize = 200 * 1024; // one ~200 KB base64 image (a screenshot / read-on-image)
const DEFAULT_WAVES: usize = 5; // synchronized push waves (peak coexistence, repeated)

/// Build one session pre-grown to the realistic profile: ~40 messages, mostly 8 KB tool-result
/// bodies, plus one ~200 KB base64 image — ~0.5 MB of content, the shape a real session carries.
fn build_session() -> Session {
    let mut s = Session::new();
    s.user("You are a coding agent. Begin the task.");
    // One image turn (~200 KB base64).
    let img = "A".repeat(IMAGE_B64_BYTES);
    s.push(Message::user_with_images(
        "",
        vec![ImageSource::base64("image/png", img)],
    ));
    // Fill the rest with alternating assistant text / 8 KB tool-result turns.
    while s.messages.len() < HISTORY_MESSAGES {
        s.push(Message::assistant(vec![ContentBlock::text(
            "Calling a tool to inspect the repository.",
        )]));
        if s.messages.len() >= HISTORY_MESSAGES {
            break;
        }
        let body = "x".repeat(TOOL_RESULT_BYTES);
        s.push(Message::tool_result(
            format!("toolu_{}", s.messages.len()),
            body,
            false,
        ));
    }
    s
}

/// The next big turn to push in a wave — a realistic ~8 KB tool result.
fn next_turn(wave: usize) -> Message {
    let body = "y".repeat(TOOL_RESULT_BYTES);
    Message::tool_result(format!("toolu_wave_{wave}"), body, false)
}

/// (VmHWM, VmRSS) in kB from /proc/self/status.
fn read_rss_kb() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut hwm = 0;
    let mut rss = 0;
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            hwm = v
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("VmRSS:") {
            rss = v
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
        }
    }
    (hwm, rss)
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(64);
    let waves: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(DEFAULT_WAVES);

    // Two rendezvous points shared by main + all N workers:
    //  `start` releases everyone into a wave together;
    //  `peak`  blocks until every worker is holding BOTH its snapshot and its freshly-pushed vec —
    //          the instant of true N-way coexistence, when peak RSS is real.
    let start = Arc::new(Barrier::new(n + 1));
    let peak = Arc::new(Barrier::new(n + 1));

    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Arc::clone(&start);
        let peak = Arc::clone(&peak);
        // Small stacks: the payloads live on the heap; this keeps thread overhead from polluting the
        // RSS signal we care about (the history clones), and keeps N up to a few thousand cheap.
        let builder = thread::Builder::new().stack_size(512 * 1024);
        let handle = builder
            .spawn(move || {
                let mut session = build_session();
                let mut push_nanos: Vec<u128> = Vec::with_capacity(waves);
                for wave in 0..waves {
                    let big = next_turn(wave);
                    start.wait();
                    // The exact per-turn hot pattern: hold a request snapshot alive across the push,
                    // so make_mut sees a shared Arc — deep clone (BEFORE) / refcount bump (AFTER).
                    let snapshot = session.messages.clone();
                    let t0 = Instant::now();
                    session.push(big);
                    push_nanos.push(t0.elapsed().as_nanos());
                    black_box(&snapshot);
                    black_box(&session.messages);
                    // Rendezvous at peak coexistence: snapshot (old vec) and session (new vec) are
                    // both alive here, across all N workers simultaneously.
                    peak.wait();
                    // snapshot dropped at end of iteration.
                    drop(snapshot);
                }
                black_box(&session);
                push_nanos
            })
            .expect("spawn worker");
        handles.push(handle);
    }

    // Drive the waves and capture peak RSS at each coexistence rendezvous.
    let mut peak_hwm_kb = 0;
    for _ in 0..waves {
        start.wait();
        peak.wait();
        let (hwm, _rss) = read_rss_kb();
        peak_hwm_kb = peak_hwm_kb.max(hwm);
    }

    // Collect per-push latencies from every (worker × wave).
    let mut all_nanos: Vec<u128> = Vec::with_capacity(n * waves);
    for h in handles {
        all_nanos.extend(h.join().expect("join worker"));
    }
    all_nanos.sort_unstable();

    let (final_hwm, final_rss) = read_rss_kb();
    peak_hwm_kb = peak_hwm_kb.max(final_hwm);

    let peak_rss_mb = peak_hwm_kb as f64 / 1024.0;
    let rss_per_session_kb = peak_hwm_kb as f64 / n as f64;
    let p50_us = percentile(&all_nanos, 0.50) as f64 / 1000.0;
    let p99_us = percentile(&all_nanos, 0.99) as f64 / 1000.0;

    // One machine-readable line.
    println!(
        "N={n} peakRSS_MB={peak_rss_mb:.1} rssPerSession_KB={rss_per_session_kb:.1} \
         push_p50_us={p50_us:.2} push_p99_us={p99_us:.2} steadyRSS_MB={:.1} waves={waves}",
        final_rss as f64 / 1024.0
    );
}
