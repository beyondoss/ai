// Bench target: `.unwrap()` sets up fixtures; not production code.
#![allow(clippy::unwrap_used)]

//! Bench: the two-pass `AgentEvent` → `Value` → `String` serialization `serve.rs`'s old `event_frame`
//! used, versus the single-pass `AgentEvent` → `String` it uses now. This is `serve`'s streaming
//! control loop's own hot path — once per model-generated delta, not once per tool call — so it never
//! had a benchmark of its own before this fix; these numbers are what motivated it.
//!
//! `serve.rs`'s real `event_frame`/`OutFrame`/`write_frame` aren't reachable from a bench target
//! (they're private — deliberately so, `serve` is a CLI implementation, not a library surface). This
//! replicates both serialization strategies against the same public `agent_core::AgentEvent` type
//! instead, so the comparison is honest without widening `serve.rs`'s API just to make it benchable.
//! Run: `cargo bench -p beyond-ai-agent --bench serve_events`.

use std::hint::black_box;

use agent_core::{AgentEvent, StreamEvent};
use bytes::Bytes;
use divan::Bencher;
use serde::Serialize;
use serde_json::{Map, Value, json};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// A short model-generated text chunk — the common case for `AgentEvent::Stream(StreamEvent::
/// TextDelta)`, which streams once per delta during generation, far more often than any RPC frame.
fn small_delta() -> AgentEvent {
    AgentEvent::Stream(StreamEvent::TextDelta {
        index: 0,
        text: "Let me check that file before making the change.".to_string(),
    })
}

/// A larger event — a `ToolEnd` carrying a realistic multi-KB tool result (e.g. a `read`/`grep`
/// call's output) — to show the gap scales with payload size, not just with fire frequency.
fn large_tool_end() -> AgentEvent {
    AgentEvent::ToolEnd {
        id: "tool_01abc".to_string(),
        name: "read".to_string(),
        result: "    1\tuse std::io;\n".repeat(150),
        is_error: false,
    }
}

fn event(which: &str) -> AgentEvent {
    match which {
        "small_delta" => small_delta(),
        "large_tool_end" => large_tool_end(),
        other => unreachable!("unknown bench arg: {other}"),
    }
}

/// The old `event_frame` strategy: serialize `ev` into an owned `Value` tree, wrap it in a `{type,
/// event}` envelope `Value`, then serialize *that* tree to the final string — two full passes.
#[divan::bench(args = ["small_delta", "large_tool_end"])]
fn value_then_string(bencher: Bencher, which: &str) {
    let ev = event(which);
    bencher.bench_local(|| {
        let event = serde_json::to_value(black_box(&ev)).unwrap();
        let mut m = Map::new();
        m.insert("type".into(), json!("event"));
        m.insert("event".into(), event);
        let line = serde_json::to_string(&Value::Object(m)).unwrap();
        black_box(line.len());
    });
}

/// The current `event_frame` strategy: one direct serialize of a borrowing envelope straight to the
/// final string — no intermediate `Value` tree at all.
#[divan::bench(args = ["small_delta", "large_tool_end"])]
fn direct_string(bencher: Bencher, which: &str) {
    let ev = event(which);
    #[derive(Serialize)]
    struct EventFrame<'a> {
        r#type: &'static str,
        event: &'a AgentEvent,
    }
    bencher.bench_local(|| {
        let line = serde_json::to_string(&EventFrame {
            r#type: "event",
            event: black_box(&ev),
        })
        .unwrap();
        black_box(line.len());
    });
}

/// The already-serialized event line `event_frame` hands to an `OutFrame::Raw` — the `{type,event}`
/// envelope as its final text.
fn frame_line(which: &str) -> String {
    #[derive(Serialize)]
    struct EventFrame<'a> {
        r#type: &'static str,
        event: &'a AgentEvent,
    }
    serde_json::to_string(&EventFrame {
        r#type: "event",
        event: &event(which),
    })
    .unwrap()
}

// ── `OutFanout::broadcast`'s per-event clone cost: `Raw(String)` (before) vs `Raw(Bytes)` (after) ──
//
// `broadcast` clones the frame once for the in-flight-turn recording, then once per *extra* attached
// sink (the single-sink case moves, so it clones zero times for the sink). `serve.rs`'s real
// `OutFanout` is private and unreachable from a bench target, so — exactly as the serialize-strategy
// benches above do — this replicates its clone pattern against the same serialized line. With
// `String` each clone is a full heap allocation + `memcpy` of the line; with `Bytes` it is a refcount
// bump. divan's `AllocProfiler` reports the alloc-count gap directly. `1sink` = recording only (1
// clone); `3sink` = recording + 3 sinks (4 clones).

#[divan::bench(args = ["small_delta", "large_tool_end"])]
fn fanout_string_1sink(bencher: Bencher, which: &str) {
    let line = frame_line(which);
    bencher.bench_local(|| {
        let rec = black_box(line.clone());
        black_box(&rec);
    });
}

#[divan::bench(args = ["small_delta", "large_tool_end"])]
fn fanout_bytes_1sink(bencher: Bencher, which: &str) {
    let line = Bytes::from(frame_line(which));
    bencher.bench_local(|| {
        let rec = black_box(line.clone());
        black_box(&rec);
    });
}

#[divan::bench(args = ["small_delta", "large_tool_end"])]
fn fanout_string_3sink(bencher: Bencher, which: &str) {
    let line = frame_line(which);
    bencher.bench_local(|| {
        let rec = line.clone();
        let s1 = line.clone();
        let s2 = line.clone();
        let s3 = line.clone();
        black_box((&rec, &s1, &s2, &s3));
    });
}

#[divan::bench(args = ["small_delta", "large_tool_end"])]
fn fanout_bytes_3sink(bencher: Bencher, which: &str) {
    let line = Bytes::from(frame_line(which));
    bencher.bench_local(|| {
        let rec = line.clone();
        let s1 = line.clone();
        let s2 = line.clone();
        let s3 = line.clone();
        black_box((&rec, &s1, &s2, &s3));
    });
}
