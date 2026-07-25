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

mod circuit_breaker {
    use super::*;
    use beyond_ai::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use std::time::Duration;

    /// Production shape: the windowed policy `config.rs` always builds, with the default threshold.
    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(CircuitBreakerConfig::windowed(20, Duration::from_secs(10)))
    }

    /// The per-provider gate charged on **every** request (`proxy::request_filter`), measured on the
    /// state it is in ~100% of the time: CLOSED. One acquire load + an unpack + a branch — no CAS, no
    /// clock read, 0 allocs. A regression here (a clock call or a write creeping onto the closed
    /// path) shows up as ns/iter climbing off single digits.
    #[divan::bench]
    fn allow_closed(bencher: Bencher) {
        let cb = breaker();
        bencher.bench(|| black_box(&cb).allow());
    }

    /// Charged on every successful upstream response (`proxy::logging`). The early return for a
    /// healthy CLOSED breaker keeps this a single load with **no write**, so a hot provider's breaker
    /// cache line stays Shared across every worker instead of ping-ponging Modified once per
    /// response. It must therefore cost about the same as `allow_closed`; if it ever measures like a
    /// CAS, the early return has been lost.
    #[divan::bench]
    fn record_success_healthy(bencher: Bencher) {
        let cb = breaker();
        bencher.bench(|| black_box(&cb).record_success());
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

    // --- realistic tail sizes -------------------------------------------------------------------
    //
    // The four benches above run on 100-200 byte fixtures: two `data:` lines. `proxy::logging`
    // actually hands these parsers a bounded tail of up to `USAGE_TAIL_CAP` (64 KiB) — ~450 lines on
    // a real stream. That gap is why a full JSON parse of every line, and a scalar byte-at-a-time
    // line split, both went unnoticed: at two lines neither is measurable. Size is the variable that
    // matters here, so sweep it, exactly as the `peek` module below already does.

    /// A realistic OpenAI chat stream of ~`bytes` with the usage chunk on the penultimate line
    /// (where OpenAI actually puts it), then `[DONE]`.
    fn openai_sse_tail(bytes: usize) -> Vec<u8> {
        let mut s = String::with_capacity(bytes + 256);
        while s.len() < bytes {
            s.push_str("data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" tok\"}}]}\n\n");
        }
        s.push_str("data: {\"id\":\"chatcmpl-x\",\"choices\":[],\"usage\":{\"prompt_tokens\":5000,\"completion_tokens\":2500}}\n\n");
        s.push_str("data: [DONE]\n\n");
        s.into_bytes()
    }

    /// A realistic Anthropic stream of ~`bytes`: `message_start` (input + cache tokens), a long run
    /// of `content_block_delta`, then the terminal `message_delta` (output tokens).
    fn anthropic_sse_tail(bytes: usize) -> Vec<u8> {
        let mut s = String::with_capacity(bytes + 256);
        s.push_str("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":5000,\"output_tokens\":1,\"cache_read_input_tokens\":4000}}}\n\n");
        while s.len() < bytes {
            s.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" tok\"}}\n\n");
        }
        s.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2500}}\n\n");
        s.into_bytes()
    }

    /// Reverse scan + `memmem` pre-filter: the answer is on the penultimate line, so cost should be
    /// ~flat in tail size. If this starts scaling with the argument, the early return is gone.
    #[divan::bench(args = [4 * 1024, 64 * 1024])]
    fn openai_stream_tail(bencher: Bencher, bytes: usize) {
        let sse = openai_sse_tail(bytes);
        bencher
            .counter(BytesCount::of_slice(&sse))
            .bench(|| usage::openai_stream(black_box(&sse)));
    }

    /// Genuinely a full pass (input tokens are at the head, output at the tail), so this *does*
    /// scale with size — but only over the `memchr` line split and the substring pre-filter, not a
    /// JSON parse per line. The alloc columns should stay at zero.
    #[divan::bench(args = [4 * 1024, 64 * 1024])]
    fn anthropic_stream_tail(bencher: Bencher, bytes: usize) {
        let sse = anthropic_sse_tail(bytes);
        bencher
            .counter(BytesCount::of_slice(&sse))
            .bench(|| usage::anthropic_stream(black_box(&sse)));
    }

    /// The worst case for the reverse scan: a well-formed stream that never carries usage, so it
    /// cannot stop early and must walk the whole tail. Guards the claim that the pre-filter, not the
    /// early return, is what keeps this cheap.
    #[divan::bench(args = [4 * 1024, 64 * 1024])]
    fn openai_stream_tail_no_usage(bencher: Bencher, bytes: usize) {
        let mut sse = openai_sse_tail(bytes);
        // Drop the usage chunk + [DONE], leaving only content deltas.
        let cut = sse
            .windows(7)
            .position(|w| w == b"\"usage\"")
            .expect("fixture carries a usage chunk");
        sse.truncate(cut);
        bencher
            .counter(BytesCount::of_slice(&sse))
            .bench(|| usage::openai_stream(black_box(&sse)));
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

mod store_watch {
    use super::*;
    use beyond_ai::deny::{DenyReason, DenySet};
    use beyond_ai::store_watch::apply_batch;
    use store::{KvEntry, KvUpdate, VersionToken};

    /// A realistically-sized live deny-set. The set is O(denied), not O(tenants), so a few thousand
    /// cut-off tenants is a busy day — but it's the *size of the map that gets copied* on every
    /// `rcu`, which is exactly what batching is about.
    const DENIED: u64 = 1_024;

    fn populated(n: u64) -> DenySet {
        (0..n).map(|t| (t, DenyReason::Spend)).collect()
    }

    /// A burst of `k` `Put` deltas for tenants outside the existing set (a control-plane sweep).
    fn burst(k: usize) -> Vec<KvUpdate> {
        (0..k)
            .map(|i| {
                KvUpdate::Put(KvEntry {
                    key: format!("blackhole.{}", DENIED + i as u64),
                    value: b"spend".to_vec(),
                    version: VersionToken::from_u64(100 + i as u64),
                })
            })
            .collect()
    }

    /// The watcher's apply step, batched: `k` deltas land under **one** `rcu` clone of the map.
    /// The alloc column is the claim — it must stay at 1 map allocation regardless of `k`, and the
    /// `k = 1` row must match `apply_one_at_a_time`'s (batching costs nothing in the steady state).
    #[divan::bench(args = [1, 8, 64, 256])]
    fn apply_batched(bencher: Bencher, k: usize) {
        let set = populated(DENIED);
        let updates = burst(k);
        bencher.bench(|| apply_batch(black_box(&set), black_box(&updates)));
    }

    /// The pre-batching shape, kept as the control: one `rcu` — i.e. one full clone of the map —
    /// per delta, which is what the watch loop used to do. O(k·N) copies against the batched
    /// O(N + k).
    #[divan::bench(args = [1, 8, 64, 256])]
    fn apply_one_at_a_time(bencher: Bencher, k: usize) {
        let set = populated(DENIED);
        let updates = burst(k);
        bencher.bench(|| {
            let mut cur = apply_batch(black_box(&set), &updates[..1]);
            for u in &updates[1..] {
                cur = apply_batch(&cur, std::slice::from_ref(u));
            }
            cur
        });
    }

    // --- cold-boot / `CursorExpired` rescan: turning a scan into snapshot `Put` records ---

    fn scanned(n: usize) -> Vec<KvEntry> {
        (0..n)
            .map(|i| KvEntry {
                key: format!("blackhole.{i}"),
                value: br#"{"reason":"spend","exp":1750000000}"#.to_vec(),
                version: VersionToken::from_u64(i as u64 + 1),
            })
            .collect()
    }

    /// `rebuild_snapshot` wraps each scanned entry in a `KvUpdate::Put` for `write_update`. The file
    /// I/O is identical either way and is deliberately excluded — the only difference is whether
    /// each entry's `String` key and `Vec<u8>` value are copied or moved, so the alloc column is the
    /// whole story: 2 allocations per entry vs none.
    ///
    /// Both variants take the scan result **by value** (as the real closure does) so each pays for
    /// dropping it; otherwise the cloning variant would look artificially cheap by leaving the
    /// originals alive past the timed region.
    #[divan::bench(args = [128, 4096])]
    fn rebuild_puts_cloned(bencher: Bencher, n: usize) {
        bencher.with_inputs(|| scanned(n)).bench_values(|entries| {
            for e in &entries {
                black_box(KvUpdate::Put(e.clone()));
            }
        });
    }

    /// The same loop consuming the `Vec` it owns (`for e in entries`) — 0 allocations.
    #[divan::bench(args = [128, 4096])]
    fn rebuild_puts_moved(bencher: Bencher, n: usize) {
        bencher.with_inputs(|| scanned(n)).bench_values(|entries| {
            for e in entries {
                black_box(KvUpdate::Put(e));
            }
        });
    }
}
