// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Concurrency-scaling probe for the SSE decode path — the evidence gate for whether the decoder's
//! per-event `serde_json::Value` allocations are worth eliminating.
//!
//! Per *single* session the decode cost is negligible against seconds of model generation; the only
//! way it could gate real throughput is **allocator contention** when many decode streams run at once
//! in one process (each event mallocs/frees ~12 times, and a contended global allocator serializes
//! that). This bench spawns `N` OS threads (real parallelism — an async executor on one thread
//! wouldn't expose allocator lock contention), each decoding a full transcript `REPS` times, and
//! reports aggregate throughput.
//!
//! **Reading it.** The box has 16 cores, so `N ≤ 16` is the diagnostic window (work fits in real
//! parallelism). With no contention, aggregate MB/s should scale ~linearly with `N` there — 2×, 4×,
//! 8×, 16× the `N=1` number — and per-thread wall-time stay flat. If allocator contention bites, the
//! aggregate curve **bends** (sublinear) well before the core count, and that's the evidence that
//! justifies the typed-parse rewrite. `N=32` (past the core count) is included only to show CPU
//! saturation — a linear-in-N wall-time climb there is expected and is *not* a contention signal.
//!
//! Deliberately **no `AllocProfiler`**: this runs on the plain system allocator (the production one,
//! whose contention is the whole question). Divan's profiler adds per-alloc atomic counters that are
//! themselves a global contention point — installing it here would manufacture the very signal we're
//! trying to measure. Allocation *counts* come from the sibling `decode` bench instead.
//!
//! Run with `mise run bench:decode:concurrent` (or `cargo bench -p beyond-ai-agent-core --bench
//! decode_concurrent`).

use std::hint::black_box;
use std::sync::Barrier;
use std::thread;

use agent_core::dialect::{StreamDecoder, anthropic, decode_sse};
use divan::Bencher;
use divan::counter::BytesCount;

// Control-run allocator (see the `bench-mimalloc` feature). When enabled, the identical bench runs on
// a sharded low-contention allocator instead of glibc malloc — if the concurrency curve flattens vs.
// the default run, the bend was allocator-lock contention (and reducing per-event allocs would help);
// if it doesn't move, the bend is CPU-side (SMT/turbo/bandwidth) and the typed rewrite wouldn't help.
#[cfg(feature = "bench-mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    divan::main();
}

/// Streamed text deltas per transcript — a long tool-heavy turn (mirrors the `decode` bench).
const TEXT_DELTAS: usize = 800;
/// Full-transcript decodes each worker thread performs per sample. Sized so per-thread work
/// (~REPS × 250 µs ≈ 10 ms) dwarfs thread-spawn + barrier overhead, so the measurement is decode
/// throughput, not scheduling noise.
const REPS: usize = 40;

/// Build the Anthropic tool-heavy transcript (the default Claude wire, and the heaviest decode).
fn anthropic_transcript(deltas: usize) -> String {
    let mut s = String::new();
    let ev = |s: &mut String, name: &str, data: &str| {
        s.push_str("event: ");
        s.push_str(name);
        s.push('\n');
        s.push_str("data: ");
        s.push_str(data);
        s.push_str("\n\n");
    };
    ev(
        &mut s,
        "message_start",
        r#"{"type":"message_start","message":{"usage":{"input_tokens":1200,"output_tokens":0,"cache_read_input_tokens":1100,"cache_creation_input_tokens":0}}}"#,
    );
    ev(
        &mut s,
        "content_block_start",
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
    );
    for _ in 0..deltas {
        ev(
            &mut s,
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"the quick brown fox "}}"#,
        );
    }
    ev(
        &mut s,
        "content_block_stop",
        r#"{"type":"content_block_stop","index":0}"#,
    );
    ev(
        &mut s,
        "message_delta",
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":180}}"#,
    );
    ev(&mut s, "message_stop", r#"{"type":"message_stop"}"#);
    s
}

/// `N` threads each decode the transcript `REPS` times, started together via a barrier so the timed
/// region is steady-state parallel decode (not staggered spawn). The `BytesCount` counter is the
/// **total** bytes decoded (`N × REPS × len`), so divan's throughput column is the aggregate — the
/// number whose scaling with `N` answers the contention question.
#[divan::bench(args = [1, 2, 4, 8, 16, 32])]
fn concurrent_decode(bencher: Bencher, threads: usize) {
    let transcript = anthropic_transcript(TEXT_DELTAS);
    let body_len = transcript.len();
    bencher
        .counter(BytesCount::new(threads * REPS * body_len))
        .bench(|| {
            let barrier = Barrier::new(threads);
            let transcript = &transcript;
            thread::scope(|scope| {
                for _ in 0..threads {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        // Align the start so the measured region is all-threads-parallel, not a
                        // spawn ramp — the point is steady-state allocator pressure.
                        barrier.wait();
                        let mut sink = 0usize;
                        for _ in 0..REPS {
                            let mut decoder = anthropic::Decoder::default();
                            let events =
                                decode_sse(&mut decoder as &mut dyn StreamDecoder, transcript)
                                    .expect("valid transcript");
                            sink += black_box(&events).len();
                        }
                        black_box(sink);
                    });
                }
            });
        });
}
