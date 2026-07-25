// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Codex WebSocket delta-transport per-turn work. The whole point of this transport is to *avoid*
//! resending the transcript each turn (send only the new-item delta + a `previous_response_id`), yet
//! the pre-M7 code deep-cloned that transcript 3-4× per turn on the way to building the tiny delta
//! frame. This bench measures that waste directly: timing **and** allocations come from `divan`'s
//! `AllocProfiler` (installed as the global allocator below), so alloc count + bytes-cloned per turn
//! sit beside ns/turn in one table. Run:
//! `cargo bench -p beyond-ai-agent-core --bench codex_ws`.
//!
//! `codex_websocket`'s `body_fingerprint`/`input_items`/`build_wire_body`/`wire_frame` and its inbound
//! frame-parse are all module-private (it's a live transport, not a library surface). Rather than
//! widen that API purely to bench it, this replicates each clone pattern — the pre-M7 `before_*` and
//! the post-M7 `after_*` — faithfully against the same public `serde_json::Value` body shape the real
//! functions operate on. This is the sanctioned pattern `crates/agent/benches/serve_events.rs` uses
//! for `serve`'s private `event_frame`. The `after_*` replicas are line-for-line the shipped code;
//! the correctness of the shipped code itself is gated by the in-module `codex_websocket` unit tests.
//!
//! Groups map to the findings:
//! - `fingerprint` — [T4-F1]: `body_fingerprint`, clone-whole-then-remove vs clone-retained-only.
//! - `build_wire_body` — [T4-F3, T4-F4]: the delta-frame build (fingerprint compare + input
//!   length/prefix check + object-minus-input clone), full transcript clone vs borrow + delta tail.
//! - `wire_frame` — [T4-F4]: the `response.create` envelope, clone-map vs move-map.
//! - `frame_parse` — [T4-F2]: one inbound stream-event frame, `to_string()`-then-parse vs parse-off
//!   the borrowed frame bytes.
//! - `harvest_item` — [T4-F11]: pulling one `response.output_item.done`'s `item` for the replay
//!   baseline, `get(..).clone()` vs `remove(..)` out of the soon-dropped frame.
//!
//! The `size` axis (transcript item count) is the independent variable: the gap is meant to grow with
//! the conversation, which is exactly the cost the delta transport exists to eliminate.

use std::hint::black_box;

use divan::Bencher;
use serde_json::{Map, Value, json};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// Transcript sizes to bracket the range: an early turn vs. a grown conversation. The per-turn clone
/// cost the delta transport is supposed to avoid scales with this count.
const SIZES: [usize; 2] = [20, 100];

/// The fields `body_fingerprint` excludes — kept in lockstep with `codex_websocket::DIFF_EXCLUDED_FIELDS`.
const DIFF_EXCLUDED_FIELDS: [&str; 3] = ["input", "previous_response_id", "max_output_tokens"];

/// One realistic Codex input item — alternating user message / assistant function_call, the shapes a
/// real transcript is full of. Deliberately non-trivial (nested content, multiple string fields) so a
/// deep clone of the array is a meaningful amount of work, as it is on the wire.
fn input_item(i: usize) -> Value {
    if i.is_multiple_of(2) {
        json!({
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "please investigate the failing test and explain the root cause in detail"
            }]
        })
    } else {
        json!({
            "type": "function_call",
            "name": "bash",
            "call_id": format!("call_{i:08x}"),
            "arguments": "{\"command\":\"grep -rn TODO crates/ && cargo test -p beyond-ai-agent-core\"}"
        })
    }
}

/// A full request body of `n` input items — the shape `dialect::openai_responses::build_body` emits
/// and this transport diffs: a handful of scalar/object fields plus the big `input` transcript array.
fn body(n: usize) -> Value {
    json!({
        "model": "gpt-5-codex",
        "instructions": "You are a coding agent. Be precise and terse.",
        "tools": [{"type": "function", "name": "bash", "description": "run a shell command"}],
        "max_output_tokens": 32000,
        "stream": true,
        "input": (0..n).map(input_item).collect::<Vec<_>>(),
    })
}

// ============================================================================
// [T4-F1] body_fingerprint
// ============================================================================

/// Pre-M7: deep-clone the whole body (incl. the `input` transcript), then drop the excluded fields.
fn before_fingerprint(body: &Value) -> Value {
    let mut body = body.clone();
    if let Some(obj) = body.as_object_mut() {
        for key in DIFF_EXCLUDED_FIELDS {
            obj.remove(key);
        }
    }
    body
}

/// Post-M7 (shipped): clone only the retained fields into a fresh object.
fn after_fingerprint(body: &Value) -> Value {
    match body.as_object() {
        Some(obj) => Value::Object(
            obj.iter()
                .filter(|(key, _)| !DIFF_EXCLUDED_FIELDS.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
        None => body.clone(),
    }
}

mod fingerprint {
    use super::*;

    #[divan::bench(args = SIZES)]
    fn before(bencher: Bencher, n: usize) {
        let body = body(n);
        bencher.bench(|| before_fingerprint(black_box(&body)));
    }

    #[divan::bench(args = SIZES)]
    fn after(bencher: Bencher, n: usize) {
        let body = body(n);
        bencher.bench(|| after_fingerprint(black_box(&body)));
    }
}

// ============================================================================
// [T4-F3, T4-F4] build_wire_body — the delta-frame build (the hot per-turn path)
// ============================================================================

/// A second-turn scenario: the continuation's baseline is the first `n` items, and the current body
/// has one new item appended — so the prefix matches and the delta path fires (the interesting case
/// the whole transport exists for). Returns `(full_body, baseline_input, fingerprint, response_id)`.
fn delta_scenario(n: usize) -> (Value, Vec<Value>, Value, String) {
    let baseline: Vec<Value> = (0..n).map(input_item).collect();
    let full = body(n + 1); // baseline prefix + one new tail item
    let fp = after_fingerprint(&full);
    (full, baseline, fp, "resp_prev_0123456789".to_string())
}

/// Pre-M7: `input_items` clones the whole array for the length/prefix check, and every full-resend
/// branch — plus the delta branch — deep-clones the whole body.
fn before_build_wire_body(
    full_body: &Value,
    baseline: &[Value],
    fingerprint: &Value,
    response_id: &str,
) -> Value {
    if response_id.is_empty() {
        return full_body.clone();
    }
    if before_fingerprint(full_body) != *fingerprint {
        return full_body.clone();
    }
    let current_input: Vec<Value> = full_body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if current_input.len() < baseline.len() {
        return full_body.clone();
    }
    if current_input[..baseline.len()] != baseline[..] {
        return full_body.clone();
    }
    let delta = &current_input[baseline.len()..];
    let mut trimmed = full_body.clone();
    if let Some(obj) = trimmed.as_object_mut() {
        obj.insert("input".to_string(), Value::Array(delta.to_vec()));
        obj.insert(
            "previous_response_id".to_string(),
            Value::String(response_id.to_string()),
        );
    }
    trimmed
}

/// Post-M7 (shipped): borrow the input for the checks, clone only the delta tail, and clone
/// object-minus-input for the delta frame.
fn after_build_wire_body(
    full_body: &Value,
    baseline: &[Value],
    fingerprint: &Value,
    response_id: &str,
) -> Value {
    if response_id.is_empty() {
        return full_body.clone();
    }
    if after_fingerprint(full_body) != *fingerprint {
        return full_body.clone();
    }
    let current_input: &[Value] = full_body
        .get("input")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if current_input.len() < baseline.len() {
        return full_body.clone();
    }
    if current_input[..baseline.len()] != baseline[..] {
        return full_body.clone();
    }
    let delta = &current_input[baseline.len()..];
    let mut trimmed: Map<String, Value> = full_body
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter(|(key, _)| key.as_str() != "input")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default();
    trimmed.insert("input".to_string(), Value::Array(delta.to_vec()));
    trimmed.insert(
        "previous_response_id".to_string(),
        Value::String(response_id.to_string()),
    );
    Value::Object(trimmed)
}

mod build_wire_body {
    use super::*;

    #[divan::bench(args = SIZES)]
    fn before(bencher: Bencher, n: usize) {
        let (full, baseline, fp, id) = delta_scenario(n);
        bencher.bench(|| {
            before_build_wire_body(black_box(&full), black_box(&baseline), black_box(&fp), &id)
        });
    }

    #[divan::bench(args = SIZES)]
    fn after(bencher: Bencher, n: usize) {
        let (full, baseline, fp, id) = delta_scenario(n);
        bencher.bench(|| {
            after_build_wire_body(black_box(&full), black_box(&baseline), black_box(&fp), &id)
        });
    }
}

// ============================================================================
// [T4-F4] wire_frame — the response.create envelope
// ============================================================================

/// Pre-M7: `&Value` in, clone the whole map to add the `type` field.
fn before_wire_frame(body: &Value) -> String {
    let mut obj = match body {
        Value::Object(map) => map.clone(),
        _ => Map::new(),
    };
    obj.insert(
        "type".to_string(),
        Value::String("response.create".to_string()),
    );
    Value::Object(obj).to_string()
}

/// Post-M7 (shipped): `Value` in, move the map.
fn after_wire_frame(body: Value) -> String {
    let mut obj = match body {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    obj.insert(
        "type".to_string(),
        Value::String("response.create".to_string()),
    );
    Value::Object(obj).to_string()
}

mod wire_frame {
    use super::*;

    // The real caller owns a freshly-built wire body per turn, so `before` must clone one per sample
    // to have a `&Value` to pass (its whole cost is that it can't consume the owned value it's given),
    // while `after` moves the owned value straight in. Both rebuild the body per sample so neither
    // times body construction; the honest comparison is total per-turn work at the call site.
    #[divan::bench(args = SIZES)]
    fn before(bencher: Bencher, n: usize) {
        bencher.with_inputs(|| body(n)).bench_values(|body| {
            let borrowed = &body;
            black_box(before_wire_frame(black_box(borrowed)))
        });
    }

    #[divan::bench(args = SIZES)]
    fn after(bencher: Bencher, n: usize) {
        bencher
            .with_inputs(|| body(n))
            .bench_values(|body| black_box(after_wire_frame(black_box(body))));
    }
}

// ============================================================================
// [T4-F2] inbound frame parse — once per stream event
// ============================================================================

/// A single streamed frame the size and shape of a real `response.output_text.delta` event.
fn stream_frame() -> String {
    json!({
        "type": "response.output_text.delta",
        "output_index": 0,
        "item_id": "item_0123456789abcdef",
        "delta": "the quick brown fox jumps over the lazy dog "
    })
    .to_string()
}

mod frame_parse {
    use super::*;

    /// Pre-M7: copy the frame into an owned `String`, then parse from that.
    #[divan::bench]
    fn before(bencher: Bencher) {
        let frame = stream_frame();
        bencher.bench(|| {
            let text = black_box(frame.as_str()).to_string();
            serde_json::from_str::<Value>(&text).unwrap()
        });
    }

    /// Post-M7 (shipped): parse straight off the borrowed frame bytes — no owned copy.
    #[divan::bench]
    fn after(bencher: Bencher) {
        let frame = stream_frame();
        bencher.bench(|| serde_json::from_str::<Value>(black_box(frame.as_str())).unwrap());
    }
}

// ============================================================================
// [T4-F11] harvest one output item off a soon-dropped frame
// ============================================================================

/// A `response.output_item.done` frame carrying a realistic assistant `item` (the replay baseline
/// source). The `value` it's parsed into is dropped at the end of the iteration either way.
fn output_item_done_frame() -> Value {
    json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message",
            "role": "assistant",
            "id": "msg_0123456789abcdef",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "Here is a fairly long assistant reply that becomes part of the replay baseline for the next turn's prefix check, so its size matters."
            }]
        }
    })
}

mod harvest_item {
    use super::*;

    /// Pre-M7: clone the `item` out of a frame that's about to be dropped.
    #[divan::bench]
    fn before(bencher: Bencher) {
        bencher
            .with_inputs(output_item_done_frame)
            .bench_values(|value| {
                let item = value.get("item").cloned();
                black_box(item)
            });
    }

    /// Post-M7 (shipped): move the `item` out of the soon-dropped frame.
    #[divan::bench]
    fn after(bencher: Bencher) {
        bencher
            .with_inputs(output_item_done_frame)
            .bench_values(|mut value| {
                let item = value.as_object_mut().and_then(|obj| obj.remove("item"));
                black_box(item)
            });
    }
}
