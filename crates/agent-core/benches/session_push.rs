// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! `Session::push` bench: the end-of-turn append onto a history the in-flight `ModelRequest` still
//! shares. During a turn the agent loop holds `req.messages = session.messages.clone()` — an `Arc`
//! bump — so when the turn ends and `push` calls `Arc::make_mut`, the `Arc` is **shared** and
//! `make_mut` **deep-clones the entire history**: every payload in every prior message, once per
//! turn, growing with history. The M8/[T4-F8] change makes the large immutable `ContentBlock`
//! payloads (`Text.text`, `ToolResult.content`, base64 `ImageSource.data`, thinking) `Arc<str>`, so
//! each per-message deep clone becomes a refcount bump instead of a fresh heap copy of the bytes.
//!
//! `divan`'s `AllocProfiler` (installed below) reports alloc **count** + **bytes** per sample beside
//! ns/iter — the whole point of the change is that both drop, so both must be visible. This bench is
//! written purely against the public constructor APIs (`ContentBlock::text`, `Message::tool_result`,
//! `ImageSource::base64`, …), which take `impl Into<_>` and so compile unchanged before (`String`)
//! and after (`Arc<str>`); the before/after delta is captured by re-running across the change.
//!
//! Run with `cargo bench -p beyond-ai-agent-core --bench session_push`.
//!
//! Two groups:
//! - `push_shared` — the real hot path: rebuild the realistic history, take `let _req =
//!   session.messages.clone()` to force the shared-`Arc` state exactly as an in-flight turn does,
//!   then `session.push(msg)` — measuring the `make_mut` deep-clone the shared `Arc` triggers.
//! - `push_owned` — the same `push` when the `Arc` is *not* shared (the between-turns steady state),
//!   as a control: here `make_mut` is already just an in-place `Vec::push`, so this group should be
//!   flat across the change (nothing to deep-clone), isolating that the win is specifically the
//!   shared-clone case, not `push` itself.
//!
//! History sizes bracket a short and a long conversation so the "cost grows with history" claim is
//! visible as a slope, not a single point.

use std::hint::black_box;

use agent_core::message::{ContentBlock, ImageSource, Message};
use agent_core::session::Session;
use divan::Bencher;
use divan::counter::BytesCount;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// History lengths to bench at — a short exchange and a long one. `make_mut`'s deep clone is O(total
/// bytes across all messages), so the cost should scale roughly with this.
const HISTORY_SIZES: &[usize] = &[20, 60];

/// A multi-KB tool-result body — the dominant per-message payload in a real agent transcript (a
/// `read`/`bash`/`grep` result). ~4 KB, the regime where copying the bytes dominates fixed per-message
/// overhead.
const TOOL_RESULT_BYTES: usize = 4 * 1024;

/// A single base64 image blob (a screenshot / `read` on an image) — the single biggest immutable
/// payload a message can carry. ~48 KB of base64, planted once in the history.
const IMAGE_BYTES: usize = 48 * 1024;

/// Build a realistic large conversation: alternating user prompt / assistant text / tool-call /
/// tool-result turns, with one image message planted in the middle. `n` is the target message count.
fn build_history(n: usize) -> Session {
    let tool_body: String = "x".repeat(TOOL_RESULT_BYTES);
    let image_data: String = "A".repeat(IMAGE_BYTES);
    let prose = "The quick brown fox jumps over the lazy dog. ".repeat(8);

    let mut session = Session::new();
    for i in 0..n {
        let msg = match i % 4 {
            0 => Message::user(format!("user prompt number {i}: {prose}")),
            1 => Message::assistant(vec![
                ContentBlock::text(format!("assistant reply {i}: {prose}")),
                ContentBlock::tool_use(
                    format!("tu_{i}"),
                    "read",
                    serde_json::json!({ "path": format!("src/file_{i}.rs") }),
                ),
            ]),
            2 => Message::tool_result(format!("tu_{}", i - 1), tool_body.clone(), false),
            // One image message planted mid-history (the biggest single blob); plain text elsewhere.
            _ if i == n / 2 => Message::user_with_images(
                format!("look at this, step {i}"),
                vec![ImageSource::base64("image/png", image_data.clone())],
            ),
            _ => Message::assistant(vec![ContentBlock::text(format!("summary {i}: {prose}"))]),
        };
        session.push(msg);
    }
    session
}

/// Approximate total immutable-payload bytes in a history of `n` messages — the amount `make_mut`
/// deep-copies on a shared `Arc`. Used as the `BytesCount` so divan reports throughput (bytes/s) and
/// the per-sample byte count reflects what the clone actually moves.
fn approx_history_bytes(n: usize) -> usize {
    // ~1 tool-result body + a fraction of an image + small prose per 4 messages.
    let quarters = n / 4;
    quarters * TOOL_RESULT_BYTES + IMAGE_BYTES + n * 400
}

/// The message we append each iteration — a fresh tool result, the common end-of-turn append.
fn make_push_msg() -> Message {
    Message::tool_result("tu_next", "y".repeat(TOOL_RESULT_BYTES), false)
}

/// The hot path: `Arc` shared with an in-flight request → `push`'s `make_mut` deep-clones the history.
#[divan::bench(args = HISTORY_SIZES)]
fn push_shared(bencher: Bencher, n: usize) {
    bencher
        .counter(BytesCount::new(approx_history_bytes(n)))
        .with_inputs(|| {
            let session = build_history(n);
            // Force the shared-`Arc` state the agent loop creates: `req.messages` holds a second
            // strong ref, so the next `push` cannot mutate in place and must deep-clone.
            let req = session.messages.clone();
            (session, req)
        })
        .bench_local_values(|(mut session, req)| {
            session.push(make_push_msg());
            black_box(&req);
            black_box(&session);
        });
}

/// Control: single-owner `Arc` → `push` is an in-place `Vec::push`, no deep clone. Should be flat
/// across the change.
#[divan::bench(args = HISTORY_SIZES)]
fn push_owned(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| build_history(n))
        .bench_local_values(|mut session| {
            session.push(make_push_msg());
            black_box(&session);
        });
}
