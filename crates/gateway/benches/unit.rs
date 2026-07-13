// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code. See tests/e2e.rs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Unit bench: the pure, IO-free hot paths. Timing **and** allocations come from `divan` — its
//! `AllocProfiler` (installed as the global allocator below) reports alloc count + bytes per
//! sample right beside ns/iter, so the design's allocation claims are visible in one table.
//! Run with `mise run bench:unit` (or `cargo bench --bench unit`).
//!
//! The headline invariant to watch: managed-key **verify** is 0 allocs — it decodes onto the
//! stack (see `key.rs`). `peek` should hold a flat, tiny alloc count independent of body size
//! (the O(1)-memory claim). A regression shows up as a non-zero / grown number in the alloc
//! columns the moment this runs.
//!
//! Fixtures are built *outside* the closure handed to `Bencher::bench` (or in `args`), so only the
//! measured call is timed and counted — setup allocations don't pollute the numbers.

use std::hint::black_box;

use divan::Bencher;
use divan::counter::BytesCount;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

mod key {
    use super::*;
    use beyond_ai::key::{Keyring, VirtualKey, mint};
    use ed25519_dalek::SigningKey;

    const ID: VirtualKey = VirtualKey {
        tenant_id: 42,
        vpc_id: 7,
    };

    /// Stateless verify — must not touch the heap (stack-only base64 decode + signature check).
    #[divan::bench]
    fn verify(bencher: Bencher) {
        let sk = SigningKey::from_bytes(&[1u8; 32]);
        let mut ring = Keyring::new();
        ring.insert(1, sk.verifying_key());
        let token = mint(&ID, 1, &sk);
        bencher.bench(|| ring.verify(black_box(&token)));
    }

    /// Reference mint path (allocates the output string + base64 segments) — tracked so the Go
    /// control-plane parity implementation has a baseline.
    #[divan::bench]
    fn mint_key(bencher: Bencher) {
        let sk = SigningKey::from_bytes(&[1u8; 32]);
        bencher.bench(|| mint(black_box(&ID), 1, &sk));
    }
}

mod route {
    use super::*;
    use beyond_ai::route::{Dialect, dialect_default};

    // Dialect → default provider name: the per-request routing decision (sans override). 0-alloc.
    #[divan::bench(args = [Dialect::OpenAi, Dialect::Anthropic])]
    fn dialect_default_name(bencher: Bencher, dialect: Dialect) {
        bencher.bench(|| dialect_default(black_box(dialect)));
    }
}

mod deny {
    use super::*;
    use beyond_ai::deny::{self, DenyReason, DenySet};

    // --- ingest path: parse a watched NATS key/value into the set (off the request hot path) ---

    #[divan::bench]
    fn parse_key() -> Option<u64> {
        deny::parse_key(black_box("blackhole.123456789"))
    }

    #[divan::bench]
    fn parse_reason_bare() -> beyond_ai::deny::DenyReason {
        deny::parse_reason(black_box(b"spend"))
    }

    #[divan::bench]
    fn parse_reason_json() -> beyond_ai::deny::DenyReason {
        deny::parse_reason(black_box(br#"{"reason":"fraud","exp":123}"#))
    }

    // --- request hot path: the lookup run on EVERY managed request (`proxy::request_filter`) ---

    /// Build a deny-set holding `n` cut-off tenants (ids `0..n`). Built outside the timed closure.
    fn populated(n: u64) -> DenySet {
        (0..n).map(|t| (t, DenyReason::Spend)).collect()
    }

    /// The common case: tenant **absent** from the set (default-allow). The headline invariant is
    /// that this is O(1) and **0-alloc regardless of set size** — so the args span an empty set and
    /// a large one (1M cut-off tenants); the ns/iter and the (absent) alloc columns must stay flat.
    /// A regression to anything size-dependent shows up as the big-`n` row diverging from the small.
    #[divan::bench(args = [0, 1_000_000])]
    fn reason_miss(bencher: Bencher, n: u64) {
        let set = populated(n);
        // A tenant id past the populated range → guaranteed miss (the allow path).
        bencher.bench(|| set.reason(black_box(n + 1)));
    }

    /// The deny case: tenant present. Same O(1) hash lookup, returning the reason — proves the
    /// enforce path costs the same as the allow path (no surprise on the rejection branch).
    #[divan::bench(args = [1, 1_000_000])]
    fn reason_hit(bencher: Bencher, n: u64) {
        let set = populated(n);
        bencher.bench(|| set.reason(black_box(n / 2)));
    }
}

mod ratelimit {
    use super::*;
    use beyond_ai::ratelimit::RateLimit;

    /// Guardrail charged on **every request before verify** (`proxy::request_filter`). Managed: a
    /// seeded hash of the raw credential + the per-credential sketch `observe` (the BYO global tier is
    /// skipped). Fixed memory regardless of key cardinality, so this must be flat and low-alloc.
    #[divan::bench]
    fn check_managed(bencher: Bencher) {
        let rl = RateLimit::new(1_000_000, 1_000_000).expect("enabled");
        let cred = "bai_v1.1.AAAAAAAAAAAAAAAAAAAAAA.signature-base64url-payload-here";
        bencher.bench(|| rl.check(black_box(cred), black_box(true)));
    }

    /// A longer BYO provider token — exercises both tiers (global BYO bucket + per-credential sketch)
    /// against a realistic raw token length: the full per-request BYO cost.
    #[divan::bench]
    fn check_byo(bencher: Bencher) {
        let rl = RateLimit::new(1_000_000, 1_000_000).expect("enabled");
        let token = "sk-some-byo-provider-token-of-realistic-length-abcdef0123456789";
        bencher.bench(|| rl.check(black_box(token), black_box(false)));
    }
}

mod usage {
    use super::*;
    use beyond_ai::usage::{self, Usage};

    const OAI: &[u8] = br#"{"usage":{"prompt_tokens":12,"completion_tokens":34,"prompt_tokens_details":{"cached_tokens":4}}}"#;
    const ANT: &[u8] = br#"{"usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":10,"cache_creation_input_tokens":7}}"#;
    const OAI_SSE: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";
    const ANT_SSE: &[u8] = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":15}}\n\n";

    #[divan::bench]
    fn openai_body() -> Option<Usage> {
        usage::openai_body(black_box(OAI))
    }

    #[divan::bench]
    fn anthropic_body() -> Option<Usage> {
        usage::anthropic_body(black_box(ANT))
    }

    #[divan::bench]
    fn openai_stream() -> Option<Usage> {
        usage::openai_stream(black_box(OAI_SSE))
    }

    #[divan::bench]
    fn anthropic_stream() -> Option<Usage> {
        usage::anthropic_stream(black_box(ANT_SSE))
    }
}

mod peek {
    use super::*;
    use beyond_ai::peek::ModelScanner;

    /// A realistic chat body with `padding` bytes of message content, the root `model` placed
    /// **last** so the scanner must walk the whole body (worst case for the streaming scan).
    fn body_with_model_last(padding: usize) -> Vec<u8> {
        let content = "x".repeat(padding);
        format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"stream":true,"model":"claude-opus-4-8"}}"#)
            .into_bytes()
    }

    /// Sizes span a tiny request, a typical prompt, and a large one (e.g. a pasted document /
    /// base64 image) that exercises the SIMD fast-skip over uninteresting string content. The
    /// `BytesCount` makes divan report bytes/sec; the alloc columns should stay flat across sizes.
    #[divan::bench(args = [0, 4 * 1024, 256 * 1024])]
    fn scan_model_last(bencher: Bencher, padding: usize) {
        let body = body_with_model_last(padding);
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            let mut scanner = ModelScanner::new();
            scanner.feed(black_box(&body));
            scanner.take_model()
        });
    }

    use beyond_ai::peek::plan_stream_usage_injection;

    /// A streaming body whose large `content` value precedes the root `stream` field — the worst
    /// case for the injection planner: it must walk past `padding` bytes of uninteresting string
    /// content (the SIMD fast-skip target) before it can decide.
    fn streaming_body(padding: usize) -> Vec<u8> {
        let content = "x".repeat(padding);
        format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"model":"gpt-4o","stream":true}}"#)
            .into_bytes()
    }

    /// The common case: a non-streaming body (no `stream` field). The planner must prove absence,
    /// which today means a full structural walk — the case the `memmem` pre-filter short-circuits.
    fn non_streaming_body(padding: usize) -> Vec<u8> {
        let content = "x".repeat(padding);
        format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"model":"gpt-4o"}}"#)
            .into_bytes()
    }

    /// Plan injection on a **streaming** body (must walk past the big content value to find `stream`).
    #[divan::bench(args = [0, 4 * 1024, 256 * 1024])]
    fn plan_inject_streaming(bencher: Bencher, padding: usize) {
        let body = streaming_body(padding);
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| plan_stream_usage_injection(black_box(&body)));
    }

    /// Plan injection on a **non-streaming** body (no `stream` key — the majority case).
    #[divan::bench(args = [0, 4 * 1024, 256 * 1024])]
    fn plan_inject_non_streaming(bencher: Bencher, padding: usize) {
        let body = non_streaming_body(padding);
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| plan_stream_usage_injection(black_box(&body)));
    }
}
