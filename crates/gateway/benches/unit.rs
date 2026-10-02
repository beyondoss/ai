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
//! Also on a request we actually serve, and measured here for that reason: `allowance::reason_for`
//! (every managed request, same shape as deny), `pin::order` (every managed default
//! walk), `cache::key` + `ResponseCache::get` (every catalog walk **when the cache is enabled** —
//! it is off by default), and `translate` (only when the client wire and the candidate wire
//! differ). Circuit-open and a hung failover are not here: neither is steady-state gateway CPU.
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

/// One process-wide registry for every bench that touches metrics.
///
/// `Metrics::new` registers on the default registry, which rejects a second registration. `admin`
/// and `reject` each used to build their own, so a full `divan` run — `admin` is alphabetical,
/// so it always goes first — panicked in `reject` with "registered once". The provider child
/// series are materialised here too: that cross-product is what makes a scrape body tens of KiB
/// from the first sample, and it has to happen exactly once regardless of which module runs first.
fn bench_metrics() -> &'static std::sync::Arc<beyond_ai::metrics::Metrics> {
    use std::sync::{Arc, OnceLock};

    use beyond_ai::metrics::{Metrics, ProviderMetrics};

    static M: OnceLock<Arc<Metrics>> = OnceLock::new();
    M.get_or_init(|| {
        let m = Metrics::new().expect("register metrics once");
        for spec in beyond_ai::route::known_providers() {
            let pm = ProviderMetrics::resolve(&m, spec.name);
            // Non-zero observations so the histogram buckets encode realistic values, not zeros.
            pm.ttft_seconds.observe(0.42);
            pm.upstream_latency_seconds.observe(3.5);
            pm.record_response(200);
            pm.record_response(429);
        }
        m
    })
}

mod key {
    use super::*;
    use beyond_ai::key::{Keyring, VirtualKey, mint, mint_v2};
    use ed25519_dalek::SigningKey;

    const ID: VirtualKey = VirtualKey {
        tenant_id: 42,
        vpc_id: 7,
        key_id: None,
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

    /// Same stack-only path for v2 (24-byte payload). Must stay 0 allocs.
    #[divan::bench]
    fn verify_v2(bencher: Bencher) {
        let sk = SigningKey::from_bytes(&[1u8; 32]);
        let mut ring = Keyring::new();
        ring.insert(1, sk.verifying_key());
        let token = mint_v2(&ID, 99, 1, &sk);
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

mod admin {
    use super::*;
    use beyond_ai::admin::{AdminApp, HEALTH_OK};

    fn registry() -> &'static std::sync::Arc<beyond_ai::metrics::Metrics> {
        bench_metrics()
    }

    /// `/metrics`: gather the default registry and text-encode it. The `BytesCount` reports the real
    /// encoded body size, which is the number the buffer pre-sizing has to track — a stale constant
    /// shows up here as reallocs (bytes-copied) rather than as an obviously wrong line.
    #[divan::bench]
    fn scrape(bencher: Bencher) {
        let _ = registry();
        let len = AdminApp::metrics().body().len();
        bencher
            .counter(BytesCount::new(len))
            .bench(AdminApp::metrics);
    }

    /// `/livez` (and `/readyz`): the per-probe health body. Cheap and infrequent, but it is the one
    /// response the orchestrator hits forever, so the alloc columns are worth pinning.
    #[divan::bench]
    fn health(bencher: Bencher) {
        bencher.bench(|| AdminApp::health(black_box(200), black_box(HEALTH_OK)));
    }
}

mod reject {
    use super::*;
    use beyond_ai::metrics::Rejection;
    use beyond_ai::proxy::{REJECT_BODIES, error_body};

    fn metrics() -> &'static std::sync::Arc<beyond_ai::metrics::Metrics> {
        bench_metrics()
    }

    /// The rejection response body. This is the flood path: `ratelimit`'s whole reason for existing
    /// is that a leaked or forged credential can be shed cheaply, so it runs at full request rate
    /// exactly when the gateway can least afford it. Must be **0 allocs** — the body is one of a
    /// closed set of constants, returned as `Bytes::from_static`.
    #[divan::bench]
    fn body_precomputed() -> bytes::Bytes {
        let (typ, msg, _) = REJECT_BODIES[2];
        error_body(black_box(typ), black_box(msg))
    }

    /// What it replaced: a `serde_json` DOM (a `Map` plus an owned `String` per key and per string
    /// value) serialized into a fresh `String`, for output that was always one of eight constants.
    #[divan::bench]
    fn body_via_serde_json() -> bytes::Bytes {
        let (typ, msg, _) = REJECT_BODIES[2];
        bytes::Bytes::from(
            serde_json::json!({ "error": { "type": black_box(typ), "message": black_box(msg) } })
                .to_string(),
        )
    }

    /// The four token counters for a typical OpenAI response: input and output non-zero, cache-read
    /// zero (no cache hit), cache-write **always** zero on the OpenAI wire. Guarding on non-zero
    /// turns two of the four contended read-modify-writes into a branch.
    #[divan::bench]
    fn record_tokens_guarded(bencher: Bencher) {
        let m = metrics();
        let usage = beyond_ai::usage::Usage {
            input_tokens: black_box(5000),
            output_tokens: black_box(2500),
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            reasoning_tokens: None,
            ..Default::default()
        };
        bencher.bench(|| m.record_tokens(black_box(&usage)));
    }

    /// What it replaced: all four bumped unconditionally. `inc_by` is an unguarded `fetch_add`, so
    /// adding zero still takes the cache line Exclusive and invalidates it on every other worker.
    #[divan::bench]
    fn record_tokens_unguarded(bencher: Bencher) {
        let m = metrics();
        let (i, o, cr, cw) = (5000u64, 2500u64, 0u64, 0u64);
        bencher.bench(|| {
            m.tokens_input.inc_by(black_box(i));
            m.tokens_output.inc_by(black_box(o));
            m.tokens_cache_read.inc_by(black_box(cr));
            m.tokens_cache_write.inc_by(black_box(cw));
        });
    }

    /// The rejection counter, resolved once at boot: a direct `fetch_add`.
    #[divan::bench]
    fn counter_preresolved(bencher: Bencher) {
        let m = metrics();
        bencher.bench(|| m.rejection(black_box(Rejection::Auth)).inc());
    }

    /// What it replaced: an FNV hash of the label bytes, a `RwLock` read, a `HashMap` lookup, an
    /// `Arc` clone and an `Arc` drop — per rejected request, on every worker, against one lock.
    #[divan::bench]
    fn counter_via_label_lookup(bencher: Bencher) {
        let m = metrics();
        bencher.bench(|| {
            m.rejections_total
                .with_label_values(&[black_box(Rejection::Auth.label())])
                .inc()
        });
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

    // --- forward-path construction on the bare-default route ------------------------------------
    //
    // `/v1/…` with no provider prefix is the drop-in route (an OpenAI or Anthropic SDK pointed at
    // the gateway by changing only the host), so it is the common shape. Its forwarded path is the
    // inbound path, unchanged — `request_filter` now says so with `None` instead of rebuilding the
    // string, handing it to `upstream_request_filter`, comparing it equal, and dropping it.
    //
    // These two rows are the work that no longer happens on that route, not a before/after of code
    // that still runs. Kept because the cost is entirely in the allocation, which is invisible
    // end-to-end (a loopback round-trip is ~120 µs) but shows plainly in the alloc columns.

    /// The two header values a managed request sets on every upstream request: the pool key and
    /// `Host`. Both are fixed at boot, but `insert_header(name, &str)` re-validates and re-copies
    /// the bytes into a fresh `Bytes` each time.
    #[divan::bench]
    fn header_value_from_str(bencher: Bencher) {
        let auth = "Bearer sk-proj-0123456789abcdef0123456789abcdef0123456789abcdef";
        bencher.bench(|| http::HeaderValue::from_str(black_box(auth)).unwrap());
    }

    /// Cloning the boot-built value instead: a `Bytes` refcount bump, no allocation.
    #[divan::bench]
    fn header_value_clone(bencher: Bencher) {
        let auth = "Bearer sk-proj-0123456789abcdef0123456789abcdef0123456789abcdef";
        let pre = http::HeaderValue::from_str(auth).unwrap();
        let _ = pre.clone(); // force the one-time promotion to a shared representation
        bencher.bench(|| black_box(&pre).clone());
    }

    /// What the bare-default route used to build, with no query string.
    #[divan::bench]
    fn forward_path_rebuilt_no_query() -> String {
        let path = black_box("/v1/chat/completions");
        let query: Option<&str> = black_box(None);
        match query {
            Some(q) => format!("{path}?{q}"),
            None => path.to_string(),
        }
    }

    /// The same, with a query string — Azure OpenAI requires `?api-version=…` on every call, so
    /// this is not a rare shape.
    #[divan::bench]
    fn forward_path_rebuilt_with_query() -> String {
        let path = black_box("/v1/chat/completions");
        let query: Option<&str> = black_box(Some("api-version=2024-10-21"));
        match query {
            Some(q) => format!("{path}?{q}"),
            None => path.to_string(),
        }
    }
}

mod resp_tail {
    use super::*;

    const CAP: usize = 64 * 1024;

    /// What the response tap did: grow to `2 × CAP`, then `copy_within` the last `CAP` bytes back to
    /// the front and truncate. Bounded, but it re-copies `CAP` bytes every `CAP` bytes of stream, so
    /// a long response gets memmoved about twice over — plus the geometric realloc chain from
    /// starting at zero capacity.
    #[divan::bench(args = [64 * 1024, 512 * 1024, 4 * 1024 * 1024])]
    fn grow_and_compact(bencher: Bencher, total: usize) {
        let chunk = vec![b'x'; 8 * 1024];
        let chunks = total / chunk.len();
        bencher.counter(BytesCount::new(total)).bench(|| {
            let mut tail: Vec<u8> = Vec::new();
            for _ in 0..chunks {
                tail.extend_from_slice(black_box(&chunk));
                if tail.len() > 2 * CAP {
                    let keep = tail.len() - CAP;
                    tail.copy_within(keep.., 0);
                    tail.truncate(CAP);
                }
            }
            tail
        });
    }

    /// The ring: grows normally until it outgrows `CAP`, then writes with wraparound. Every byte is
    /// copied exactly once, there is no compaction memmove, and after the one switchover there is no
    /// further allocation. `proxy::UsageTail` is the real implementation; this mirrors it so the two
    /// rows are comparable (the type is private, and its behaviour is pinned by
    /// `usage_tail_retains_exactly_the_last_cap_bytes`).
    #[divan::bench(args = [64 * 1024, 512 * 1024, 4 * 1024 * 1024])]
    fn ring(bencher: Bencher, total: usize) {
        let chunk = vec![b'x'; 8 * 1024];
        let chunks = total / chunk.len();
        bencher.counter(BytesCount::new(total)).bench(|| {
            let mut buf: Vec<u8> = Vec::new();
            let (mut head, mut is_ring) = (0usize, false);
            for _ in 0..chunks {
                let data = black_box(&chunk);
                if !is_ring {
                    buf.extend_from_slice(data);
                    if buf.len() > CAP {
                        let start = buf.len() - CAP;
                        buf.copy_within(start.., 0);
                        buf.truncate(CAP);
                        head = 0;
                        is_ring = true;
                    }
                    continue;
                }
                let first = (CAP - head).min(data.len());
                buf[head..head + first].copy_from_slice(&data[..first]);
                let rest = data.len() - first;
                if rest > 0 {
                    buf[..rest].copy_from_slice(&data[first..]);
                }
                head = (head + data.len()) % CAP;
            }
            if is_ring {
                buf.rotate_left(head);
            }
            buf
        });
    }
}

mod state {
    use super::*;
    use beyond_ai::config::AiConfig;
    use beyond_ai::state::GatewayState;
    use std::sync::{Arc, OnceLock};

    fn state() -> &'static Arc<GatewayState> {
        static S: OnceLock<Arc<GatewayState>> = OnceLock::new();
        S.get_or_init(|| {
            GatewayState::new(AiConfig::default(), bench_metrics().clone()).expect("build state")
        })
    }

    /// Minted for **every** request, including every fast reject, so it is the most-executed line in
    /// the crate. Must stay 0-alloc (an `ArrayString` on the stack) and off `core::fmt`: the
    /// instance half is boot-constant and is copied, only the counter is rendered.
    #[divan::bench]
    fn next_request_id(bencher: Bencher) {
        let s = state();
        bencher.bench(|| black_box(s).next_request_id());
    }
}

mod deny {
    use super::*;
    use beyond_ai::deny::{self, DenyReason, DenySet};

    // --- ingest path: parse a watched NATS key/value into the set (off the request hot path) ---

    #[divan::bench]
    fn parse_key() -> Option<beyond_ai::deny::DenyTarget> {
        deny::parse_key(black_box("blackhole.123456789"))
    }

    #[divan::bench]
    fn parse_reason_bare() -> beyond_ai::deny::DenyReason {
        deny::parse_reason(black_box(b"spend"))
    }

    /// The shape the control plane actually writes (note the extra `exp` field, which must be
    /// skipped without ever being materialized). Must be **0 allocs**: the reason deserializes as a
    /// `Cow` borrowed straight out of the input. A non-zero alloc column here means someone
    /// reintroduced an owning parse — on a seed/rescan this runs once per entry.
    #[divan::bench]
    fn parse_reason_json() -> beyond_ai::deny::DenyReason {
        deny::parse_reason(black_box(br#"{"reason":"fraud","exp":123}"#))
    }

    /// Same, but the reason is `\u`-escaped (`\u0066` is `f`), so it cannot be borrowed out of
    /// the input. This is the case that forces the field to be a `Cow` and not a `&str`: a
    /// `&str` field cannot represent an escaped string, so it would fail the parse outright and
    /// silently degrade to `Unknown`.
    /// Costs **2 allocs / 13 B** — serde_json's unescape scratch buffer (8 B) plus the resulting
    /// `String` (5 B). The control plane doesn't write escapes, so this is the rare path; it is
    /// benched to keep the borrowed case above honest about what it is avoiding.
    #[divan::bench]
    fn parse_reason_json_escaped() -> beyond_ai::deny::DenyReason {
        deny::parse_reason(black_box(br#"{"reason":"\u0066raud","exp":123}"#))
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

/// The per-request cost of the capture/control surface **when nobody is using it** — which is what
/// almost every request pays and the only number that has to be free.
mod capture {
    use super::*;
    use beyond_ai::capture::{CaptureBufs, CaptureRule, CaptureSet};
    use beyond_ai::control::Control;
    use pingora::http::RequestHeader;

    const DEFAULTS: CaptureRule = CaptureRule {
        sample_n: 1,
        max_bytes: 256 * 1024,
    };

    fn populated(n: u64) -> CaptureSet {
        (0..n).map(|t| (t, DEFAULTS)).collect()
    }

    // --- request hot path: what a request that isn't being captured pays ---

    /// **The headline "capture costs nothing when off" measurement.** Sparse miss, O(1), 0-alloc
    /// regardless of set size — same shape and same `FxHasher` as `deny::reason_miss`, so the two
    /// rows should read alike. `n = 0` is the realistic production case (nobody capturing); the 1M
    /// row exists so a regression to anything size-dependent shows up as divergence between them.
    #[divan::bench(args = [0, 1_000_000])]
    fn rule_for_miss(bencher: Bencher, n: u64) {
        let set = populated(n);
        bencher.bench(|| set.rule_for(black_box(n + 1)));
    }

    #[divan::bench(args = [1, 1_000_000])]
    fn rule_for_hit(bencher: Bencher, n: u64) {
        let set = populated(n);
        bencher.bench(|| set.rule_for(black_box(n / 2)));
    }

    /// Parsing the control surface on a request that sent **no** `x-beyond-*` headers — the other
    /// half of the always-paid cost. Two absent-header lookups and nothing else; must be 0-alloc.
    #[divan::bench]
    fn control_parse_absent(bencher: Bencher) {
        let req = RequestHeader::build("POST", b"/v1/chat/completions", None).expect("header");
        bencher.bench(|| Control::parse(black_box(&req)));
    }

    /// The same parse when the caller *did* send tags. Allocates (the canonical re-serialization is
    /// the point — see `control`), so this row is here to keep that cost visible and bounded rather
    /// than to be free.
    #[divan::bench]
    fn control_parse_metadata(bencher: Bencher) {
        let mut req = RequestHeader::build("POST", b"/v1/chat/completions", None).expect("header");
        req.insert_header(
            "x-beyond-metadata",
            r#"{"feature":"summarizer","org":"acme"}"#,
        )
        .expect("insert");
        bencher.bench(|| Control::parse(black_box(&req)));
    }

    // --- the tap itself: what a request that IS being captured pays, per chunk ---

    /// Bounded append across a body delivered in 16 KiB chunks. The `total > cap` row is the one
    /// that matters for a long SSE stream: once the cap is reached the tap must stop copying
    /// entirely rather than keep growing, so its cost per chunk should collapse to a length check.
    #[divan::bench(args = [8 * 1024, 256 * 1024, 1024 * 1024])]
    fn push_bounded(bencher: Bencher, total: usize) {
        let chunk = vec![b'x'; 16 * 1024];
        bencher.bench(|| {
            let mut bufs = CaptureBufs::new(black_box(total) as u32);
            let mut written = 0;
            while written < 1024 * 1024 {
                bufs.push_resp(black_box(&chunk));
                written += chunk.len();
            }
            bufs
        });
    }
}

mod ratelimit {
    use super::*;
    use beyond_ai::ratelimit::RateLimit;
    use std::cell::Cell;
    use std::sync::LazyLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Guardrail charged on **every request before verify** (`proxy::request_filter`). Managed: a
    /// seeded hash of the raw credential + the per-credential sketch `observe` (the BYO global tier is
    /// skipped). Fixed memory regardless of key cardinality, so this must be flat and low-alloc.
    ///
    /// Single-threaded, one credential: everything the sketch touches stays in L1. That is the *best*
    /// case and it is deliberately kept — it isolates hashing + the arithmetic from the memory system.
    /// It cannot see contention, cache misses, or the window reset; `flood_*` below covers those.
    #[divan::bench]
    fn check_managed(bencher: Bencher) {
        let rl = RateLimit::new(1_000_000, 1_000_000).expect("enabled");
        let cred = "bai_v1.1.AAAAAAAAAAAAAAAAAAAAAA.signature-base64url-payload-here";
        bencher.bench(|| rl.check(black_box(cred), black_box(true)));
    }

    /// A longer BYO provider token — exercises both tiers (global BYO bucket + per-credential sketch)
    /// against a realistic raw token length: the full per-request BYO cost. Single-threaded/hot-cache,
    /// same caveat as `check_managed`.
    #[divan::bench]
    fn check_byo(bencher: Bencher) {
        let rl = RateLimit::new(1_000_000, 1_000_000).expect("enabled");
        let token = "sk-some-byo-provider-token-of-realistic-length-abcdef0123456789";
        bencher.bench(|| rl.check(black_box(token), black_box(false)));
    }

    // --- the load-shape this guardrail actually exists for: many workers, many distinct creds ---

    /// Distinct credentials in the flood corpus. Power of two so the cursor wrap is a mask. Sized to
    /// the per-credential sketch's `SLOTS` so a run touches the *whole* counter array, not a hot
    /// corner of it — that is what makes the sketch's cache footprint visible.
    const FLOOD: usize = 65_536;

    /// Realistic ~70-byte credentials, built once outside every timed region.
    static CREDS: LazyLock<Vec<String>> = LazyLock::new(|| {
        (0..FLOOD)
            .map(|i| format!("bai_v1.1.AAAAAAAAAAAAAAAA{i:08}.signature-base64url-payload-here"))
            .collect()
    });

    static NEXT_THREAD: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        /// Per-thread cursor into `CREDS`. Seeded to a different region per thread (odd stride, so it
        /// is a permutation) so the threads don't march in lockstep over the same counters — and it is
        /// thread-local rather than a shared atomic so the harness itself contributes no contention.
        static CURSOR: Cell<usize> =
            Cell::new(NEXT_THREAD.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E37_79B9));
    }

    #[inline]
    fn next_cred() -> &'static str {
        let i = CURSOR.with(|c| {
            let v = c.get();
            c.set(v.wrapping_add(1));
            v
        });
        &CREDS[i & (FLOOD - 1)]
    }

    /// The real shape of the load: N Pingora workers charging *distinct* credentials concurrently.
    /// Three things this shows that the hot-cache benches structurally cannot:
    ///
    /// 1. **Contention** — `threads = 16` puts every worker on the same shared counters.
    /// 2. **Cache footprint** — `FLOOD` distinct keys spread the accesses over the entire sketch, so
    ///    the per-request cost includes the cache misses the sizing constant buys.
    /// 3. **The window reset** — `min_time = 3` guarantees the run spans ≥ 2 window rotations, so the
    ///    reset walk lands inside a sample. It is ~1 in a million requests, so watch the **slowest**
    ///    column, not the median: the median is throughput, the max is the tail this costs.
    #[divan::bench(threads = [1, 16], min_time = 3, sample_size = 100)]
    fn check_flood_managed(bencher: Bencher) {
        let rl = RateLimit::new(u32::MAX, u32::MAX).expect("enabled");
        LazyLock::force(&CREDS);
        bencher.bench(|| rl.check(black_box(next_cred()), black_box(true)));
    }

    /// Same, on the BYO path — so it charges the global BYO tier *and* the per-credential sketch. The
    /// global tier is a single shared bucket, i.e. the most contended thing in the module.
    #[divan::bench(threads = [1, 16], min_time = 3, sample_size = 100)]
    fn check_flood_byo(bencher: Bencher) {
        let rl = RateLimit::new(u32::MAX, u32::MAX).expect("enabled");
        LazyLock::force(&CREDS);
        bencher.bench(|| rl.check(black_box(next_cred()), black_box(false)));
    }

    /// The tail the flood benches can only hint at: the cost of the *one* request per window that
    /// wins the rotation and pays for zeroing a whole sketch inline, on its own request thread.
    ///
    /// `check_at` makes this measurable instead of statistical — every iteration steps the clock a
    /// full window, so every iteration rotates. Read it as a per-rotation cost, not a per-request
    /// one: at 1 Hz and a plausible peak of 100k rps it is one request in ~100k that pays it, i.e.
    /// around p99.999 for the ceiling on `SLOTS`, not a throughput term.
    #[divan::bench(sample_size = 20)]
    fn rotate_window(bencher: Bencher) {
        let rl = RateLimit::new(u32::MAX, 0).expect("enabled");
        let cred = "bai_v1.1.AAAAAAAAAAAAAAAAAAAAAA.signature-base64url-payload-here";
        let t0 = std::time::Instant::now();
        let mut window = 0u32;
        bencher.bench_local(|| {
            window += 1;
            rl.check_at(
                black_box(cred),
                black_box(true),
                t0 + std::time::Duration::from_secs(1) * window,
            )
        });
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

    /// An OpenAI embeddings response: a large `data` array, then `model`, then `usage` last.
    fn embeddings_body(vectors: usize, dims: usize) -> Vec<u8> {
        let vec_json = (0..dims)
            .map(|i| format!("{}", 0.0123456 + i as f64 * 1e-7))
            .collect::<Vec<_>>()
            .join(",");
        let items = (0..vectors)
            .map(|i| format!(r#"{{"object":"embedding","index":{i},"embedding":[{vec_json}]}}"#))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"object":"list","data":[{items}],"model":"text-embedding-3-large","usage":{{"prompt_tokens":812,"total_tokens":812}}}}"#
        )
        .into_bytes()
    }

    /// A non-streaming body that *fits* in the tail: the ordinary path, a full document parse that
    /// structurally skips everything before `usage`. Kept as the baseline for the row below.
    #[divan::bench]
    fn openai_body_whole(bencher: Bencher) {
        let body = embeddings_body(1, 3072);
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| usage::openai_body(black_box(&body)));
    }

    /// The same shape *oversized*, sliced to the 64 KiB tail `proxy::logging` actually passes — so
    /// the buffer starts mid-value and the whole-document parse fails. This used to meter nothing at
    /// all; it now recovers from the trailing `usage`, and does so faster than parsing a whole body,
    /// because the anchored parse skips straight to the object instead of walking the array.
    #[divan::bench]
    fn openai_body_truncated_tail(bencher: Bencher) {
        let body = embeddings_body(8, 3072);
        let tail = &body[body.len() - 64 * 1024..];
        bencher
            .counter(BytesCount::of_slice(tail))
            .bench(|| usage::openai_body(black_box(tail)));
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

    /// The **response**-side scan (`proxy::response_body_filter`), fed the relayed stream chunk by
    /// chunk to recover the model the provider actually billed under.
    ///
    /// Two very different shapes, which is the whole point of benching it:
    ///
    /// - `openai`: the first SSE chunk carries a root-level `model`, so the scanner sets `done` and
    ///   every later chunk short-circuits. Flat in stream size — this is the design working.
    /// - `anthropic`: the model is nested at `message.model` inside `message_start`, i.e. depth 2,
    ///   so a root-only scanner never matches, never sets `done`, and byte-walks the *entire*
    ///   stream to return `None`. Linear in stream size, for nothing.
    ///
    /// Both are now skipped outright for BYO traffic, whose extracted model is never read.
    #[divan::bench(args = [64 * 1024, 1024 * 1024])]
    fn scan_response_stream(bencher: Bencher, bytes: usize) {
        let openai = {
            let mut s = String::from(
                "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            );
            while s.len() < bytes {
                s.push_str("data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" tok\"}}]}\n\n");
            }
            s.into_bytes()
        };
        let anthropic = {
            let mut s = String::from(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":5000}}}\n\n",
            );
            while s.len() < bytes {
                s.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" tok\"}}\n\n");
            }
            s.into_bytes()
        };
        // Fed in 8 KiB chunks, the way pingora hands the relay to `response_body_filter`.
        let scan = |body: &[u8]| {
            let mut scanner = ModelScanner::for_response();
            for chunk in body.chunks(8 * 1024) {
                scanner.feed(chunk);
            }
            scanner.take_model()
        };
        bencher
            .counter(BytesCount::of_slice(&openai))
            .bench(|| (scan(black_box(&openai)), scan(black_box(&anthropic))));
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

    use beyond_ai::peek::scan_buffered;

    /// What a managed OpenAI chat request used to cost: the body walked once by `ModelScanner`
    /// chunk-by-chunk for the root `model`, then walked end-to-end *again* by the injection planner
    /// for root `stream`/`stream_options`. Same traversal, same bytes, different needles.
    #[divan::bench(args = [4 * 1024, 64 * 1024, 512 * 1024])]
    fn buffered_two_walks(bencher: Bencher, padding: usize) {
        let body = streaming_body(padding);
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            let mut scanner = ModelScanner::new();
            for chunk in black_box(&body).chunks(8 * 1024) {
                scanner.feed(chunk);
            }
            (
                scanner.take_model(),
                plan_stream_usage_injection(black_box(&body)),
            )
        });
    }

    /// The same two answers from one pass. Cross-checked for equivalence against both originals in
    /// `peek::tests::fused_scan_matches_the_two_walks_it_replaces`.
    #[divan::bench(args = [4 * 1024, 64 * 1024, 512 * 1024])]
    fn buffered_fused_walk(bencher: Bencher, padding: usize) {
        let body = streaming_body(padding);
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| scan_buffered(black_box(&body)));
    }

    const FRAG: &[u8] = br#""stream_options":{"include_usage":true},"#;

    // Both splice rows measure **accumulate + inject**, which is what a buffered request actually
    // costs end to end. Timing the injection alone would be misleading in either direction: the old
    // form's cost is dominated by its second allocation (and that buffer's page faults), which only
    // shows up next to an accumulation that had to happen regardless.

    /// As it was: `req_buf` sized to the body, then a *second* buffer of `len + FRAG` allocated and
    /// every byte copied across.
    #[divan::bench(args = [4 * 1024, 64 * 1024, 1024 * 1024])]
    fn splice_copying(bencher: Bencher, padding: usize) {
        let body = streaming_body(padding);
        let at = 1usize;
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            let b = black_box(&body);
            let mut buf = Vec::with_capacity(b.len());
            buf.extend_from_slice(b);
            let mut out = Vec::with_capacity(buf.len() + FRAG.len());
            out.extend_from_slice(&buf[..at]);
            out.extend_from_slice(FRAG);
            out.extend_from_slice(&buf[at..]);
            out
        });
    }

    /// As it is now: `req_buf` sized with the fragment's worth of headroom up front, so the splice
    /// shifts the tail right in place and the whole path is one allocation.
    #[divan::bench(args = [4 * 1024, 64 * 1024, 1024 * 1024])]
    fn splice_in_place(bencher: Bencher, padding: usize) {
        let body = streaming_body(padding);
        let at = 1usize;
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            let b = black_box(&body);
            let mut buf = Vec::with_capacity(b.len() + FRAG.len());
            buf.extend_from_slice(b);
            let old = buf.len();
            buf.resize(old + FRAG.len(), 0);
            buf.copy_within(at..old, at + FRAG.len());
            buf[at..at + FRAG.len()].copy_from_slice(FRAG);
            buf
        });
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

/// Allowance check on every managed request, after verify (`proxy::request_filter`). Same hasher
/// and same claim as `deny::reason`: O(1), 0-alloc, flat from an empty set to a large one. v2 pays
/// two lookups (key, then tenant); v1 pays the tenant lookup only.
mod allowance {
    use super::*;
    use beyond_ai::allowance::{AllowanceSet, AllowanceTarget};

    /// `n` exhausted tenants and `n` exhausted keys. Built outside the timed closure.
    fn populated(n: u64) -> AllowanceSet {
        let mut set = AllowanceSet::from_ready();
        for id in 0..n {
            set.insert_target(AllowanceTarget::Tenant(id));
            set.insert_target(AllowanceTarget::Key(id));
        }
        set
    }

    /// Steady state: ready, and neither id is exhausted. `Some(key_id)` is the v2 shape — both
    /// maps are probed. Must stay 0-alloc and flat across set size.
    #[divan::bench(args = [0, 1_000_000])]
    fn reason_miss_v2(bencher: Bencher, n: u64) {
        let set = populated(n);
        bencher.bench(|| set.reason_for(black_box(n + 1), black_box(Some(n + 1))));
    }

    /// v1 token: no `key_id`, so only the tenant map is probed.
    #[divan::bench(args = [0, 1_000_000])]
    fn reason_miss_v1(bencher: Bencher, n: u64) {
        let set = populated(n);
        bencher.bench(|| set.reason_for(black_box(n + 1), black_box(None)));
    }

    /// The 402 branch. Same hash lookup as the miss — the rejection must not cost more.
    #[divan::bench(args = [1, 1_000_000])]
    fn reason_hit_key(bencher: Bencher, n: u64) {
        let set = populated(n);
        bencher.bench(|| set.reason_for(black_box(0), black_box(Some(n / 2))));
    }
}

/// Catalog-walk order: `order` is every managed default walk, the computed session pin
/// (rendezvous hash of the first-party lead, usable first). No allocation, no lock, no state.
mod pin_order {
    use super::*;
    use beyond_ai::control::Walk;
    use beyond_ai::pin::{affinity, order};

    #[divan::bench]
    fn order_pinned(bencher: Bencher) {
        let row = providers::for_model("claude-opus-4-8").expect("catalog row");
        let aff = affinity(42, 7, None);
        let walk = Walk::identity(row.candidates.len());
        bencher.bench(|| {
            order(
                black_box(walk),
                black_box(row),
                black_box(aff),
                black_box(u8::MAX),
            )
        });
    }
}

/// Exact-match response cache. Off unless `cache_ttl_secs > 0`. When it is on, every managed
/// catalog walk fingerprints the pre-rewrite body and takes the process-wide mutex for `get` —
/// a miss still pays both. The fingerprint hashes the body twice (two seeds → 128 bits).
mod response_cache {
    use super::*;
    use std::sync::LazyLock;
    use std::time::Duration;

    use beyond_ai::cache::{self, CacheKey, CachedResponse, ResponseCache};
    use beyond_ai::usage::Usage;
    use bytes::Bytes;

    fn entry(body: &[u8]) -> CachedResponse {
        CachedResponse {
            status: 200,
            content_type: "application/json".into(),
            body: Bytes::copy_from_slice(body),
            usage: Usage::default(),
            billed_model: "claude-opus-4-8".into(),
            requested_model: "claude-opus-4-8".into(),
            routed_model: Some("claude-opus-4-8"),
            provider: "anthropic".into(),
            streaming: false,
        }
    }

    fn body_of(n: usize) -> Vec<u8> {
        let content = "x".repeat(n);
        format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"model":"gpt-4o"}}"#)
            .into_bytes()
    }

    /// The miss-path CPU that does not take the lock: two SipHash passes over the whole body.
    /// Bytes-counter is the body once; the hasher walks it twice.
    #[divan::bench(args = [0, 4 * 1024, 64 * 1024, 256 * 1024])]
    fn fingerprint(bencher: Bencher, padding: usize) {
        let body = body_of(padding);
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| cache::key(black_box(&parts(42, &body, &[1, 2, 3]))));
    }

    static NO_HEADERS: LazyLock<http::HeaderMap> = LazyLock::new(http::HeaderMap::new);

    fn parts<'a>(tenant_id: u64, body: &'a [u8], providers: &'a [u8]) -> cache::KeyParts<'a> {
        cache::KeyParts {
            tenant_id,
            method: "POST",
            inbound_path: "/auto",
            model: "gpt-4o",
            headers: &NO_HEADERS,
            body,
            providers,
        }
    }

    fn fill(entries: usize) -> ResponseCache {
        let cache = ResponseCache::new(Duration::from_secs(3600), entries.max(1) + 8, 64);
        for i in 0..entries {
            let body = (i as u64).to_le_bytes();
            let key = cache::key(&parts(i as u64, &body, &[1]));
            cache.insert(key, entry(&body));
        }
        cache
    }

    /// Common case once the cache is on and traffic does not repeat: lock, probe, miss.
    /// `entries` is 0 and the default cap (1024) so a scan-shaped lookup would show up as a gap.
    #[divan::bench(args = [0, 1024])]
    fn get_miss(bencher: Bencher, entries: usize) {
        let cache = fill(entries);
        let missing = cache::key(&parts(u64::MAX, b"nope", &[9]));
        bencher.bench(|| cache.get(black_box(&missing)));
    }

    /// Hit clones the stored response. The body is `Bytes` (refcount); the alloc column is the
    /// four owned strings and must stay flat when the body grows from a few bytes to 64 KiB.
    #[divan::bench(args = [16, 64 * 1024])]
    fn get_hit(bencher: Bencher, body_len: usize) {
        let body = vec![b'y'; body_len];
        let cache = ResponseCache::new(Duration::from_secs(3600), 8, body_len.max(1));
        let key = cache::key(&parts(7, &body, &[1, 2]));
        cache.insert(key, entry(&body));
        bencher.bench(|| cache.get(black_box(&key)));
    }

    struct SharedHit {
        cache: ResponseCache,
        key: CacheKey,
    }

    fn shared_hit() -> &'static SharedHit {
        static HIT: LazyLock<SharedHit> = LazyLock::new(|| {
            let body: &[u8] = b"{\"ok\":true}";
            let cache = ResponseCache::new(Duration::from_secs(3600), 8, 1024);
            let key = cache::key(&parts(7, body, &[1, 2]));
            cache.insert(key, entry(body));
            SharedHit { cache, key }
        });
        &HIT
    }

    /// Same hit from 1 and 16 threads. The cache is one `Mutex` for the whole process; this is
    /// what a hot key costs once more than one worker is in `get` at once.
    #[divan::bench(threads = [1, 16])]
    fn get_hit_shared(bencher: Bencher) {
        let hit = shared_hit();
        bencher.bench(|| hit.cache.get(black_box(&hit.key)));
    }
}

/// Cross-wire translation. Called only when the inbound endpoint and the candidate endpoint
/// differ (`proxy` skips it on a same-wire walk). A full `serde_json` parse and rebuild of the
/// body — once per request for JSON, once per SSE event for a stream. This is the CPU that can
/// actually sit next to `key/verify`.
mod translate {
    use super::*;
    use beyond_ai::route::Endpoint;
    use beyond_ai::translate::{self, SseBridge};

    fn chat_request(padding: usize) -> Vec<u8> {
        let content = "x".repeat(padding);
        format!(
            r#"{{"model":"claude-opus-4-8","messages":[{{"role":"user","content":"{content}"}}],"tools":[{{"type":"function","function":{{"name":"get_weather","parameters":{{"type":"object","properties":{{}}}}}}}}]}}"#
        )
        .into_bytes()
    }

    fn anthropic_message(text_len: usize) -> Vec<u8> {
        let text = "y".repeat(text_len);
        format!(
            r#"{{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{{"type":"text","text":"{text}"}}],"stop_reason":"end_turn","usage":{{"input_tokens":12,"output_tokens":34}}}}"#
        )
        .into_bytes()
    }

    /// OpenAI SDK → Claude, the agent path. Once, at end of the request body.
    #[divan::bench(args = [0, 64 * 1024])]
    fn request_chat_to_messages(bencher: Bencher, padding: usize) {
        let body = chat_request(padding);
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            translate::request(
                black_box(Endpoint::ChatCompletions),
                black_box(Endpoint::Messages),
                black_box(&body),
                black_box("claude-opus-4-8"),
            )
        });
    }

    /// Responses → Messages is two maps (Responses → Chat → Messages). Small body: the point is
    /// the extra pass, not the payload.
    #[divan::bench]
    fn request_responses_to_messages(bencher: Bencher) {
        let body = br#"{"model":"claude-opus-4-8","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"max_output_tokens":16}"#;
        bencher.bench(|| {
            translate::request(
                black_box(Endpoint::Responses),
                black_box(Endpoint::Messages),
                black_box(body),
                black_box("claude-opus-4-8"),
            )
        });
    }

    /// Non-stream response, withheld until end-of-stream and mapped in one shot.
    #[divan::bench(args = [16, 64 * 1024])]
    fn response_messages_to_chat(bencher: Bencher, text_len: usize) {
        let body = anthropic_message(text_len);
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            translate::response_json(
                black_box(Endpoint::Messages),
                black_box(Endpoint::ChatCompletions),
                black_box(&body),
            )
        });
    }

    /// Tool arguments with only integers, or with a decimal (the D127 path: their text is kept).
    const TOOL_ARGS: [&str; 2] = [
        r#"{"path":"src/main.rs","line":42,"limit":200}"#,
        r#"{"lat":48.858370,"lon":2.294481,"zoom":12}"#,
    ];

    /// A Chat client's tool loop onto Claude: 20 replayed calls whose arguments are re-parsed.
    #[divan::bench(args = [0, 1])]
    fn request_chat_tool_history_to_messages(bencher: Bencher, which: usize) {
        let args = serde_json::to_string(TOOL_ARGS[which]).unwrap();
        let mut messages = vec![r#"{"role":"user","content":"go"}"#.to_owned()];
        for i in 0..20 {
            messages.push(format!(
                r#"{{"role":"assistant","content":null,"tool_calls":[{{"id":"call_{i}","type":"function","function":{{"name":"f","arguments":{args}}}}}]}}"#
            ));
            messages.push(format!(
                r#"{{"role":"tool","tool_call_id":"call_{i}","content":"ok"}}"#
            ));
        }
        let body = format!(
            r#"{{"model":"claude-opus-4-8","messages":[{}]}}"#,
            messages.join(",")
        )
        .into_bytes();
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            translate::request(
                black_box(Endpoint::ChatCompletions),
                black_box(Endpoint::Messages),
                black_box(&body),
                black_box("claude-opus-4-8"),
            )
        });
    }

    /// A Claude tool call onto a Chat client: `tool_use.input` becomes an `arguments` string.
    #[divan::bench(args = [0, 1])]
    fn response_messages_tool_use_to_chat(bencher: Bencher, which: usize) {
        let body = format!(
            r#"{{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{{"type":"text","text":"Looking."}},{{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}],"stop_reason":"tool_use","usage":{{"input_tokens":12,"output_tokens":34}}}}"#,
            TOOL_ARGS[which]
        )
        .into_bytes();
        bencher.bench(|| {
            translate::response_json(
                black_box(Endpoint::Messages),
                black_box(Endpoint::ChatCompletions),
                black_box(&body),
            )
        });
    }

    /// One Anthropic `text_delta` rewritten into an OpenAI chat chunk. This is the per-token cost
    /// of a translated stream; a completion pays it once per event, not once per request.
    #[divan::bench]
    fn sse_text_delta(bencher: Bencher) {
        let event = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n";
        let mut bridge = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
        bencher.bench_local(|| bridge.feed(black_box(event), false));
    }
}

/// `usage::InputTally`: the input estimate of a billing row the provider's usage left empty (a
/// managed `/{provider}` body past the retry buffer is tallied as it streams; the rest only when a
/// row needs it).
mod input_tally {
    use super::*;
    use beyond_ai::usage::InputTally;

    /// A coding agent's prompt: ~100 KiB of source as message content.
    fn code_body() -> Vec<u8> {
        let src = include_str!("../src/proxy.rs");
        let mut end = src.len().min(100 * 1024);
        while !src.is_char_boundary(end) {
            end -= 1;
        }
        let text = serde_json::to_string(&src[..end]).unwrap();
        format!(r#"{{"model":"gpt-5","messages":[{{"role":"user","content":{text}}}]}}"#)
            .into_bytes()
    }

    /// A ~1 MiB inline image and a line of text: nearly every byte is a skipped payload.
    fn image_body() -> Vec<u8> {
        let b64 = "iVBORw0KGgo".repeat(100_000);
        format!(
            r#"{{"model":"gpt-5","messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{b64}"}}}},{{"type":"text","text":"what is this?"}}]}}]}}"#
        )
        .into_bytes()
    }

    /// Fed in 16 KiB chunks, the way a body streams past `request_body_filter`.
    fn feed(bencher: Bencher, body: Vec<u8>) {
        bencher.counter(BytesCount::new(body.len())).bench(|| {
            let mut t = InputTally::default();
            for chunk in black_box(&body).chunks(16 * 1024) {
                t.feed(chunk);
            }
            t.estimate_tokens()
        });
    }

    #[divan::bench]
    fn code(bencher: Bencher) {
        feed(bencher, code_body());
    }

    /// ~1 MiB of source (a large agent turn): a `\n` escape every ~40 bytes.
    #[divan::bench]
    fn code_1mb(bencher: Bencher) {
        let src = [
            include_str!("../src/proxy.rs"),
            include_str!("../src/translate.rs"),
        ]
        .concat();
        let mut end = src.len().min(1024 * 1024);
        while !src.is_char_boundary(end) {
            end -= 1;
        }
        let text = serde_json::to_string(&src[..end]).unwrap();
        feed(
            bencher,
            format!(r#"{{"model":"gpt-5","messages":[{{"role":"user","content":{text}}}]}}"#)
                .into_bytes(),
        );
    }

    /// ~108 KiB of English with no escapes: one long text segment.
    #[divan::bench]
    fn prose(bencher: Bencher) {
        let text = "The quick brown fox jumps over the lazy dog, it's 42. ".repeat(2000);
        feed(
            bencher,
            format!(r#"{{"messages":[{{"role":"user","content":"{text}"}}]}}"#).into_bytes(),
        );
    }

    #[divan::bench]
    fn image(bencher: Bencher) {
        feed(bencher, image_body());
    }
}

/// Tenant-bound Responses ids (`signed_id`): paid only by managed Responses relays to a store.
/// Per SSE event, the cost a GPT-row Responses stream adds: a delta event names its item id (a
/// memo hit after the first), `created` / `completed` name the response id, and an event with no
/// id is copied after one `memmem`.
mod signed_id {
    use super::*;
    use beyond_ai::signed_id::{Relay, Signer};

    const RESP: &str = "resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26";
    const MSG: &str = "msg_0750331520328311006abeead0270c87d0b8128a81f8474fea";

    fn signer() -> Signer {
        Signer::new(&[(b'1', &[7u8; 32])], b'1').unwrap()
    }

    #[divan::bench]
    fn sign(bencher: Bencher) {
        let s = signer();
        bencher.bench(|| s.sign(black_box(42), black_box(RESP)));
    }

    #[divan::bench]
    fn verify(bencher: Bencher) {
        let s = signer();
        let t = s.sign(42, RESP);
        bencher.bench(|| s.verify(black_box(42), black_box(&t)));
    }

    /// One event through a relay already streaming (`begin` once, outside the loop).
    fn event(bencher: Bencher, ev: String) {
        let s = signer();
        let mut relay = Relay::new(42);
        relay.begin(200, true);
        let _ = relay.feed(&s, ev.as_bytes(), false);
        bencher
            .counter(BytesCount::new(ev.len()))
            .bench_local(|| relay.feed(&s, black_box(ev.as_bytes()), false));
    }

    #[divan::bench]
    fn delta_event(bencher: Bencher) {
        event(
            bencher,
            format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":7,\"item_id\":\"{MSG}\",\"output_index\":0,\"content_index\":0,\"delta\":\" world\",\"logprobs\":[],\"obfuscation\":\"Xq3k\"}}\n\n"
            ),
        );
    }

    #[divan::bench]
    fn completed_event(bencher: Bencher) {
        event(
            bencher,
            format!(
                "event: response.completed\ndata: {{\"type\":\"response.completed\",\"sequence_number\":40,\"response\":{{\"id\":\"{RESP}\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-5.1\",\"output\":[{{\"id\":\"{MSG}\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"{}\",\"annotations\":[]}}]}}],\"usage\":{{\"input_tokens\":12,\"output_tokens\":300,\"total_tokens\":312}}}}}}\n\n",
                "lorem ipsum ".repeat(100)
            ),
        );
    }

    #[divan::bench]
    fn event_without_id(bencher: Bencher) {
        event(
            bencher,
            "event: response.in_progress\ndata: {\"type\":\"keepalive\",\"sequence_number\":3}\n\n"
                .to_owned(),
        );
    }

    /// The same delta event through a relay that is off (a non-2xx): the floor `delta_event` adds to.
    #[divan::bench]
    fn delta_event_off(bencher: Bencher) {
        let s = signer();
        let mut relay = Relay::new(42);
        relay.begin(500, true);
        let ev = format!(
            "data: {{\"type\":\"response.output_text.delta\",\"item_id\":\"{MSG}\",\"delta\":\" world\"}}\n\n"
        );
        bencher.bench_local(|| relay.feed(&s, black_box(ev.as_bytes()), false));
    }

    /// An AI SDK turn sent back: `previous_response_id` and twenty items, half of them references.
    #[divan::bench]
    fn unsign_request(bencher: Bencher) {
        let s = signer();
        let mut items = Vec::new();
        for i in 0..20 {
            if i % 2 == 0 {
                items.push(format!(
                    r#"{{"role":"user","content":"question {i} about the weather in Paris and Lyon"}}"#
                ));
            } else {
                items.push(format!(
                    r#"{{"type":"item_reference","id":"{}"}}"#,
                    s.sign(42, MSG)
                ));
            }
        }
        let body = format!(
            r#"{{"model":"gpt-5.1","previous_response_id":"{}","input":[{}]}}"#,
            s.sign(42, RESP),
            items.join(",")
        );
        bencher
            .counter(BytesCount::new(body.len()))
            .bench(|| s.unsign_request(42, black_box(body.as_bytes()), true));
    }
}

/// Prose that mentions `image`, `file`, `document`, `reasoning` and `thinking` as words, never as
/// JSON keys or type values: the shape of a coding agent's long prompt, which loose substring
/// gates took for a reason to parse the whole body.
fn prose_body(len: usize) -> Vec<u8> {
    let line = "We keep thinking about the image file and the document; the reasoning is in the input_file note. ";
    let text = line.repeat(len / line.len() + 1);
    format!(
        r#"{{"model":"m","stream":true,"messages":[{{"role":"system","content":"You are helpful."}},{{"role":"user","content":"{text}"}}]}}"#
    )
    .into_bytes()
}

/// Peak heap of `translate::request` per shape of JSON: the measurement behind
/// `translate::translation_heap`'s constants. Read the `max alloc` bytes column against the body
/// size (`BytesCount`). Run with `cargo bench --bench unit -- translate_heap`.
mod translate_heap {
    use super::*;
    use beyond_ai::route::Endpoint;
    use beyond_ai::translate;

    const N: usize = 4 << 20;

    fn rep(s: &str, k: usize) -> String {
        vec![s; k].join(",")
    }

    fn tools(x: &str) -> String {
        format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"f","parameters":{{"type":"object","x":[{x}]}}}}}}]}}"#
        )
    }

    /// (name, client wire, upstream wire, body). Each is the body shape that costs a `Value` the
    /// most per byte on its path: many tiny containers or scalars, which `serde_json` stores at
    /// 32+ bytes each (a `BTreeMap` leaf node per non-empty object), copied again into the
    /// translated `Value`.
    fn shapes() -> Vec<(&'static str, Endpoint, Endpoint, String)> {
        use Endpoint::{ChatCompletions as Chat, Messages, Responses};
        vec![
            (
                "chat_messages",
                Chat,
                Messages,
                format!(
                    r#"{{"model":"m","messages":[{}]}}"#,
                    rep(r#"{"role":"user","content":"hi"}"#, N / 32)
                ),
            ),
            (
                "chat_text_parts",
                Chat,
                Messages,
                format!(
                    r#"{{"model":"m","messages":[{{"role":"user","content":[{}]}}]}}"#,
                    rep(r#"{"type":"text","text":"a"}"#, N / 27)
                ),
            ),
            (
                "chat_unknown_parts",
                Chat,
                Messages,
                format!(
                    r#"{{"model":"m","messages":[{{"role":"user","content":[{}]}}]}}"#,
                    rep(r#"{"a":1}"#, N / 8)
                ),
            ),
            (
                "chat_schema_objects",
                Chat,
                Messages,
                tools(&rep(r#"{"a":1}"#, N / 8)),
            ),
            (
                "chat_schema_empty_objects",
                Chat,
                Messages,
                tools(&rep("{}", N / 3)),
            ),
            (
                "chat_schema_empty_arrays",
                Chat,
                Messages,
                tools(&rep("[]", N / 3)),
            ),
            (
                "chat_schema_numbers",
                Chat,
                Messages,
                tools(&rep("1", N / 2)),
            ),
            (
                "chat_schema_empty_strings",
                Chat,
                Messages,
                tools(&rep(r#""""#, N / 3)),
            ),
            (
                "chat_schema_wide_object",
                Chat,
                Messages,
                format!(
                    r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"f","parameters":{{{}}}}}}}]}}"#,
                    (0..N / 12)
                        .map(|i| format!(r#""{i:x}":0"#))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ),
            (
                "chat_long_string",
                Chat,
                Messages,
                format!(
                    r#"{{"model":"m","messages":[{{"role":"user","content":"{}"}}]}}"#,
                    "a".repeat(N)
                ),
            ),
            (
                "responses_unknown_items",
                Responses,
                Chat,
                format!(
                    r#"{{"model":"m","store":false,"input":[{}]}}"#,
                    rep(r#"{"a":1}"#, N / 8)
                ),
            ),
            (
                "responses_unknown_items_to_messages",
                Responses,
                Messages,
                format!(
                    r#"{{"model":"m","store":false,"input":[{}]}}"#,
                    rep(r#"{"a":1}"#, N / 8)
                ),
            ),
            (
                "responses_empty_items_to_messages",
                Responses,
                Messages,
                format!(
                    r#"{{"model":"m","store":false,"input":[{}]}}"#,
                    rep("{}", N / 3)
                ),
            ),
            (
                "responses_messages_to_messages",
                Responses,
                Messages,
                format!(
                    r#"{{"model":"m","store":false,"input":[{}]}}"#,
                    rep(r#"{"role":"user","content":"hi"}"#, N / 32)
                ),
            ),
            (
                "messages_tool_input_objects",
                Messages,
                Chat,
                format!(
                    r#"{{"model":"m","max_tokens":5,"messages":[{{"role":"assistant","content":[{{"type":"tool_use","id":"t","name":"f","input":{{"x":[{}]}}}}]}}]}}"#,
                    rep(r#"{"a":1}"#, N / 8)
                ),
            ),
            (
                "messages_tool_input_floats",
                Messages,
                Chat,
                format!(
                    r#"{{"model":"m","max_tokens":5,"messages":[{{"role":"assistant","content":[{{"type":"tool_use","id":"t","name":"f","input":{{"x":[{}]}}}}]}}]}}"#,
                    rep("1.5", N / 4)
                ),
            ),
            (
                "messages_system_parts",
                Messages,
                Chat,
                format!(
                    r#"{{"model":"m","max_tokens":5,"system":[{}],"messages":[{{"role":"user","content":"hi"}}]}}"#,
                    rep(r#"{"type":"text","text":"a"}"#, N / 27)
                ),
            ),
            (
                "messages_unknown_blocks_to_responses",
                Messages,
                Responses,
                format!(
                    r#"{{"model":"m","max_tokens":5,"messages":[{{"role":"user","content":[{}]}}]}}"#,
                    rep(r#"{"a":1}"#, N / 8)
                ),
            ),
        ]
    }

    const NAMES: [&str; 18] = [
        "chat_messages",
        "chat_text_parts",
        "chat_unknown_parts",
        "chat_schema_objects",
        "chat_schema_empty_objects",
        "chat_schema_empty_arrays",
        "chat_schema_numbers",
        "chat_schema_empty_strings",
        "chat_schema_wide_object",
        "chat_long_string",
        "responses_unknown_items",
        "responses_unknown_items_to_messages",
        "responses_empty_items_to_messages",
        "responses_messages_to_messages",
        "messages_tool_input_objects",
        "messages_tool_input_floats",
        "messages_system_parts",
        "messages_unknown_blocks_to_responses",
    ];

    #[divan::bench(args = NAMES, sample_count = 1, sample_size = 1)]
    fn request(bencher: Bencher, name: &str) {
        let (_, from, to, body) = shapes().into_iter().find(|s| s.0 == name).unwrap();
        let body = body.into_bytes();
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| translate::request(from, to, black_box(&body), "claude-sonnet-4-5"));
    }
}

/// Admission-path body checks (`route::refused_input`, `route::unserved`,
/// `translate::responses_session_field`) and the same-wire Claude relay's reasoning gate.
mod body_gates {
    use super::*;
    use beyond_ai::{route, translate};
    use providers::catalog::{IN_IMAGE, MODEL_ROUTES, serves_file_input};

    fn no_image_row() -> &'static route::ModelRoute {
        MODEL_ROUTES
            .iter()
            .find(|r| {
                r.card.input & IN_IMAGE == 0
                    && route::Endpoint::of_row(r) != route::Endpoint::Embeddings
            })
            .expect("a row without image input")
    }

    fn file_gap_row() -> &'static route::ModelRoute {
        MODEL_ROUTES
            .iter()
            .find(|r| r.candidates.iter().any(|c| !serves_file_input(c)))
            .expect("a row with a candidate that reads no PDF")
    }

    #[divan::bench]
    fn refused_input_prose_500k(bencher: Bencher) {
        let body = prose_body(500 * 1024);
        let row = no_image_row();
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| route::refused_input(black_box(row), black_box(&body)));
    }

    #[divan::bench]
    fn unserved_prose_500k(bencher: Bencher) {
        let body = prose_body(500 * 1024);
        let row = file_gap_row();
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| route::unserved(black_box(row.candidates), black_box(&body)));
    }

    #[divan::bench]
    fn claude_chat_relay_reasoning_prose_500k(bencher: Bencher) {
        let body = prose_body(500 * 1024);
        bencher
            .counter(BytesCount::of_slice(&body))
            .with_inputs(|| body.clone())
            .bench_local_values(|b| {
                translate::claude_chat_relay_reasoning(b, black_box("anthropic/claude-sonnet-4.5"))
            });
    }

    /// A Responses body with `n` small input items: a long agent history.
    fn responses_items(n: usize) -> Vec<u8> {
        let items = vec![r#"{"type":"message","role":"user","content":"hi"}"#; n].join(",");
        format!(r#"{{"model":"m","store":false,"input":[{items}]}}"#).into_bytes()
    }

    #[divan::bench(args = [1_000, 100_000])]
    fn responses_session_field_items(bencher: Bencher, n: usize) {
        let body = responses_items(n);
        bencher
            .counter(BytesCount::of_slice(&body))
            .bench(|| translate::responses_session_field(black_box(&body), black_box(false)));
    }
}

/// A same-wire Chat Completions stream relayed from a host other than OpenAI (`SseBridge`'s
/// relay, which drops the identity fields OpenRouter repeats on every chunk).
mod chat_relay {
    use super::*;
    use beyond_ai::route::Endpoint;
    use beyond_ai::translate::SseBridge;

    const FIRST: &[u8] = b"data: {\"id\":\"gen-1\",\"provider\":\"Anthropic\",\"model\":\"anthropic/claude-sonnet-4.5\",\"object\":\"chat.completion.chunk\",\"created\":1790000000,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null,\"native_finish_reason\":null,\"logprobs\":null}]}\n\n";

    /// OpenRouter's every later chunk: `role` again.
    const REPEAT_ROLE: &[u8] = b"data: {\"id\":\"gen-1\",\"provider\":\"Anthropic\",\"model\":\"anthropic/claude-sonnet-4.5\",\"object\":\"chat.completion.chunk\",\"created\":1790000000,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\" world, how are you\"},\"finish_reason\":null,\"native_finish_reason\":null,\"logprobs\":null}]}\n\n";

    /// A host that sends `role` once: nothing to drop.
    const PLAIN: &[u8] = b"data: {\"id\":\"gen-1\",\"model\":\"m\",\"object\":\"chat.completion.chunk\",\"created\":1790000000,\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world, how are you\"},\"finish_reason\":null}]}\n\n";

    fn bridge() -> SseBridge {
        let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::ChatCompletions);
        let _ = b.feed(FIRST, false);
        b
    }

    #[divan::bench]
    fn repeated_role_event(bencher: Bencher) {
        let mut b = bridge();
        bencher.bench_local(|| b.feed(black_box(REPEAT_ROLE), false));
    }

    #[divan::bench]
    fn plain_event(bencher: Bencher) {
        let mut b = bridge();
        bencher.bench_local(|| b.feed(black_box(PLAIN), false));
    }
}

/// `SseBridge::feed` handed many events in one chunk (a provider that batches, or a reader that
/// fell behind): each event used to drain the buffer from the front.
mod sse_feed {
    use super::*;
    use beyond_ai::route::Endpoint;
    use beyond_ai::translate::SseBridge;

    #[divan::bench(args = [16, 1024])]
    fn messages_to_chat_batch(bencher: Bencher, n: usize) {
        let start = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n";
        let delta = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n";
        let batch = delta.repeat(n).into_bytes();
        bencher
            .counter(BytesCount::of_slice(&batch))
            .with_inputs(|| {
                let mut b = SseBridge::new(Endpoint::ChatCompletions, Endpoint::Messages);
                let _ = b.feed(start.as_bytes(), false);
                b
            })
            .bench_local_refs(|b| b.feed(black_box(&batch), false));
    }

    /// A Chat upstream onto a Messages client: call 0 is live with 64 KiB of arguments that closed
    /// but are not valid JSON, and each delta of queued call 1 asks whether call 0 is whole.
    #[divan::bench]
    fn queued_call_delta_behind_invalid_live_call(bencher: Bencher) {
        let args =
            serde_json::to_string(&format!("{{\"a\":\"{}\"}}}}", "x".repeat(64 * 1024))).unwrap();
        let open0 = format!(
            "data: {{\"id\":\"c\",\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{{\"name\":\"f\",\"arguments\":{args}}}}}]}}}}]}}\n\n"
        );
        let open1 = "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"g\",\"arguments\":\"\"}}]}}]}\n\n";
        let delta1 = b"data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"x\"}}]}}]}\n\n";
        let mut b = SseBridge::new(Endpoint::Messages, Endpoint::ChatCompletions);
        let _ = b.feed(open0.as_bytes(), false);
        let _ = b.feed(open1.as_bytes(), false);
        bencher.bench_local(|| b.feed(black_box(delta1), false));
    }
}
