// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Tool-call dispatch bench: the *per-tool-call* hot path the M9/M10 findings target. A `write`/`edit`
//! argument **is** the whole file body, so every superfluous clone of a coerced tool-input `Value` on
//! the dispatch path copies (and reallocates) a multi-KB — sometimes multi-hundred-KB — payload once
//! per call. `divan`'s `AllocProfiler` (installed below) reports alloc count + bytes per sample beside
//! ns/iter, so those copies are visible directly, not inferred.
//!
//! Run with `cargo bench -p beyond-ai-agent-core --bench dispatch`.
//!
//! Three groups:
//! - `coerce` — the real public [`coerce_tool_arguments`] API over a realistic `write` argument
//!   (a ~100 KB `content` field). Exercises `coerce_object_properties` ([T4-F5]): the per-property
//!   `remove` + `key.clone()` + `insert` churn vs. the in-place `get_mut` + `Value::take`. The 100 KB
//!   body itself rides through *by move* either way (a correctly-typed string is a pass-through), so
//!   what this isolates is the per-property key-clone / double-tree-op overhead the fix removes.
//! - `outcome_dispatch` — a faithful replica of the phase-2 execution loop's per-call outcome read
//!   ([T1-F2]): `outcomes[i].clone()` off an `Arc`-shared slot vec (deep-copies the coerced `Value`,
//!   i.e. the whole file body, per call) vs. single-owner `.take()` (moves it). This is the dominant
//!   per-call copy the audit calls out.
//! - `registry_fanout` — a faithful replica of the per-group registry capture ([T1-F1]): cloning the
//!   whole `HashMap<String, Arc<…>>` tool registry into each concurrent group vs. sharing one borrow.
//!
//! Replica groups (`outcome_dispatch`, `registry_fanout`) carry both the *before* and *after* shapes as
//! sibling benches, so one run shows the delta; the internal dispatch loop isn't reachable as a public
//! API to bench directly (see `benches/serve_events.rs` for the same sanctioned-replica approach). The
//! `coerce` group calls the real API, so its before/after is captured by re-running across the change.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use agent_core::validation::coerce_tool_arguments;
use divan::Bencher;
use divan::counter::BytesCount;
use serde_json::{Value, json};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// A file body large enough that copying it dominates any per-call fixed cost — the `write`/`edit`
/// argument regime the dispatch findings are about. ~100 KB, matching the audit's cited case.
const BODY_BYTES: usize = 100 * 1024;

/// The `write` tool's own argument schema shape (an object with a `path` and a `content` string) — the
/// schema `coerce_object_properties` walks on every `write` dispatch.
fn write_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "content": { "type": "string" },
        },
        "required": ["path", "content"],
    })
}

/// A realistic, already-correctly-typed `write` argument: `content` is a ~100 KB string. Coercion is a
/// pure pass-through here (the common case), so this measures the coercion *machinery*, not any actual
/// primitive conversion.
fn write_arg() -> Value {
    json!({
        "path": "/workspace/src/main.rs",
        "content": "x".repeat(BODY_BYTES),
    })
}

// --- coerce: the real coerce_tool_arguments API over a large write arg ([T4-F5]) ------------------

mod coerce {
    use super::*;

    #[divan::bench]
    fn write_arg_passthrough(bencher: Bencher) {
        let schema = write_schema();
        bencher
            .counter(BytesCount::new(BODY_BYTES))
            // Fresh owned arg per sample (coercion consumes it); generation is untimed.
            .with_inputs(write_arg)
            .bench_values(|arg| coerce_tool_arguments(black_box(&schema), black_box(arg)).unwrap());
    }
}

// --- outcome_dispatch: per-call outcome read, clone vs take ([T1-F2]) -----------------------------

mod outcome_dispatch {
    use super::*;

    /// Mirror of the loop's phase-1 gate outcome: a call resolved without running (`Immediate`) or one
    /// gated and ready to run carrying its coerced arguments `Value` (`Ready`) — the file body.
    enum GateOutcome {
        #[allow(dead_code)]
        Immediate(String),
        Ready(Value),
    }

    /// How many tool calls in the batch. A model batching a handful of file writes in one turn.
    const CALLS: usize = 4;

    fn outcomes() -> Vec<Option<GateOutcome>> {
        (0..CALLS)
            .map(|_| Some(GateOutcome::Ready(write_arg())))
            .collect()
    }

    /// Consume one gated outcome the way the group future does (extract the coerced value, hand it to
    /// the tool). Returns a byte count so the value can't be optimized away.
    fn consume(outcome: Option<GateOutcome>) -> usize {
        match outcome {
            Some(GateOutcome::Ready(v)) => v
                .get("content")
                .and_then(Value::as_str)
                .map_or(0, |s| s.len()),
            Some(GateOutcome::Immediate(s)) => s.len(),
            None => 0,
        }
    }

    /// BEFORE: `outcomes` is shared behind an `Arc`, so each slot must be `clone()`d out — deep-copying
    /// the whole coerced `Value` (the file body) per call.
    #[divan::bench]
    fn clone_from_shared(bencher: Bencher) {
        bencher
            .counter(BytesCount::new(BODY_BYTES * CALLS))
            .with_inputs(|| Arc::new(outcomes()))
            .bench_values(|outcomes| {
                let mut n = 0usize;
                for i in 0..CALLS {
                    // The exact `outcomes[i].clone()` the shared-Arc slot vec forces.
                    n += consume(black_box(match &outcomes[i] {
                        Some(GateOutcome::Ready(v)) => Some(GateOutcome::Ready(v.clone())),
                        Some(GateOutcome::Immediate(s)) => Some(GateOutcome::Immediate(s.clone())),
                        None => None,
                    }));
                }
                n
            });
    }

    /// AFTER: single-owner slots — each outcome is `take()`n exactly once by the one group that owns it,
    /// so the coerced `Value` is moved, never copied.
    #[divan::bench]
    fn take_owned(bencher: Bencher) {
        bencher
            .counter(BytesCount::new(BODY_BYTES * CALLS))
            .with_inputs(outcomes)
            .bench_values(|mut outcomes| {
                let mut n = 0usize;
                for slot in outcomes.iter_mut() {
                    n += consume(black_box(slot.take()));
                }
                n
            });
    }
}

// --- registry_fanout: per-group registry capture, clone vs borrow ([T1-F1]) -----------------------

mod registry_fanout {
    use super::*;

    /// Stand-in for `Arc<dyn Tool>` — an `Arc` payload whose clone is a refcount bump, so the cost the
    /// registry-map clone actually pays is the map's own `String` keys + bucket rehashing, matching the
    /// real `HashMap<String, Arc<dyn Tool>>`.
    type ToolStub = Arc<u64>;

    /// A representative tool set (read/write/edit/bash/grep/find/ls + a few more).
    const TOOLS: &[&str] = &[
        "read", "write", "edit", "bash", "grep", "find", "ls", "web", "memory", "todo",
    ];
    /// Concurrent tool-call groups in a turn (each currently clones the whole registry).
    const GROUPS: usize = 4;

    fn registry() -> HashMap<String, ToolStub> {
        TOOLS
            .iter()
            .enumerate()
            .map(|(i, name)| ((*name).to_string(), Arc::new(i as u64)))
            .collect()
    }

    /// BEFORE: every group clones the whole registry map (all keys + Arc bumps).
    #[divan::bench]
    fn clone_per_group(bencher: Bencher) {
        bencher.with_inputs(registry).bench_values(|registry| {
            let mut n = 0usize;
            for _ in 0..GROUPS {
                let current_tools = black_box(registry.clone());
                n += current_tools.get("write").map_or(0, |t| **t as usize);
            }
            n
        });
    }

    /// AFTER: every group shares one borrow of the registry — no per-group map copy.
    #[divan::bench]
    fn borrow_per_group(bencher: Bencher) {
        bencher.with_inputs(registry).bench_values(|registry| {
            let registry = &registry;
            let mut n = 0usize;
            for _ in 0..GROUPS {
                let current_tools = black_box(registry);
                n += current_tools.get("write").map_or(0, |t| **t as usize);
            }
            n
        });
    }
}
