//! A-1 end-to-end bench: the real `beyond-ai` binary + real `nats-server` + a mock upstream,
//! driven over real HTTP. Run with `mise run bench:e2e` (needs `nats-server` on PATH — mise
//! provides it). This is the macro counterpart to `unit.rs`: it measures the *whole* request path
//! (TCP accept → Pingora filters → key verify → key swap → body stream → upstream → usage tap),
//! not a single function.
//!
//! Reuses the e2e test harness (`tests/common`) verbatim so the bench and the integration tests
//! exercise the same stack. Allocations are deliberately *not* measured here — the gateway is a
//! separate process, so its heap is invisible to this binary; allocation regressions belong to the
//! in-process `unit` bench.
//!
//! The stack starts **once** and stays warm for the whole run; each iteration is one (or, for the
//! throughput group, N concurrent) HTTP round-trip(s) against that live gateway.

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::Runtime;
use tokio::task::JoinSet;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const MANAGED_BODY: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

/// A plausible BYO provider token (anything not starting with `bai_` is BYO — passed through
/// unchanged, no verify/deny/swap). The mock upstream accepts any token.
const BYO_KEY: &str = "sk-byo-provider-token-1234567890";

/// Concurrency level for the throughput group — enough in-flight requests to expose per-request
/// overhead and connection-pool behavior without saturating a laptop.
const CONCURRENCY: u64 = 32;

/// A live, warmed-up stack. Field order matters only for drop (children are killed on drop); we
/// hold every piece so nothing is torn down mid-bench.
struct Stack {
    // RAII guards: held only so their `Drop` (kill subprocess / abort task / clean tempdir) fires
    // when the bench ends. Never read directly — the requests go through `url`/`client`.
    #[allow(dead_code)]
    gw: Gateway,
    #[allow(dead_code)]
    mock: MockUpstream,
    #[allow(dead_code)]
    nats: Nats,
    client: reqwest::Client,
    vkey: String,
    url: String,
}

async fn start_stack() -> Stack {
    start_stack_with(Mode::Json).await
}

async fn start_stack_with(mode: Mode) -> Stack {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(mode).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    let url = gw.url();

    // Warm until the gateway answers 200 — the watcher connects to NATS and the DNS cache fills on
    // the first call, neither of which we want inside the timed loop.
    {
        let (c, u, k) = (client.clone(), url.clone(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                c.post(format!("{u}/v1/chat/completions"))
                    .header("authorization", format!("Bearer {k}"))
                    .header("content-type", "application/json")
                    .body(MANAGED_BODY)
                    .send()
                    .await
                    .map(|r| r.status().as_u16())
                    .unwrap_or(0)
            }
        })
        .await;
    }

    Stack {
        gw,
        mock,
        nats,
        client,
        vkey,
        url,
    }
}

/// One full managed round-trip: key swap + body relay + non-streaming usage tap. Drains the
/// response body so the connection is returned to the pool (otherwise reqwest would open a new
/// socket every iteration and we'd be benching `connect`, not the gateway).
async fn managed_roundtrip(s: &Stack) {
    let resp = s
        .client
        .post(format!("{}/v1/chat/completions", s.url))
        .header("authorization", format!("Bearer {}", s.vkey))
        .header("content-type", "application/json")
        .body(MANAGED_BODY)
        .send()
        .await
        .expect("request");
    debug_assert_eq!(resp.status().as_u16(), 200);
    let _ = resp.bytes().await.expect("body");
}

/// One **BYO** round-trip: a non-`bai_` token, passed straight through — no key verify, no deny-set
/// check, no key swap. Isolates the passthrough path's overhead from the managed path's auth work.
async fn byo_roundtrip(s: &Stack) {
    let resp = s
        .client
        .post(format!("{}/v1/chat/completions", s.url))
        .header("authorization", format!("Bearer {BYO_KEY}"))
        .header("content-type", "application/json")
        .body(MANAGED_BODY)
        .send()
        .await
        .expect("request");
    debug_assert_eq!(resp.status().as_u16(), 200);
    let _ = resp.bytes().await.expect("body");
}

/// One **rejected** request: no API key ⇒ 401, short-circuited in `request_filter` **before** any
/// upstream connection. Benched to prove a flood of rejects costs far less than a proxied request —
/// the rate-guardrail/flood rationale (a reject must not consume the upstream-connection
/// constraint). The gap between this and `managed_json_latency` is the gateway's reject headroom.
async fn reject_roundtrip(s: &Stack) {
    let resp = s
        .client
        .post(format!("{}/v1/chat/completions", s.url))
        .header("content-type", "application/json")
        .body(MANAGED_BODY)
        .send()
        .await
        .expect("request");
    debug_assert_eq!(resp.status().as_u16(), 401);
    let _ = resp.bytes().await.expect("body");
}

fn bench_e2e(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let stack = rt.block_on(start_stack());
    // A second stack whose mock streams SSE, so the response-tap (tail buffer + compaction) hot path
    // is actually exercised — it's a near no-op for the single-shot JSON body.
    let sse_stack = rt.block_on(start_stack_with(Mode::Sse));

    let mut group = c.benchmark_group("e2e");
    // Real round-trips are sub-millisecond on loopback but still ~100× a micro-bench; trim the
    // sample count so the suite stays in the seconds, not minutes.
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // Single-request latency through the full proxy: managed (verify + deny + key swap), BYO
    // (pure passthrough), SSE relay (exercises the streaming response tap), and the reject
    // fast-path (401, no upstream). Compared against each other these isolate where time goes.
    group.bench_function("managed_json_latency", |b| {
        b.to_async(&rt).iter(|| managed_roundtrip(&stack));
    });
    group.bench_function("byo_json_latency", |b| {
        b.to_async(&rt).iter(|| byo_roundtrip(&stack));
    });
    group.bench_function("managed_sse_latency", |b| {
        b.to_async(&rt).iter(|| managed_roundtrip(&sse_stack));
    });
    group.bench_function("reject_missing_key_latency", |b| {
        b.to_async(&rt).iter(|| reject_roundtrip(&stack));
    });

    // Throughput: CONCURRENCY requests in flight per iteration. `Throughput::Elements` makes
    // criterion report requests/sec.
    group.throughput(Throughput::Elements(CONCURRENCY));
    group.bench_function("managed_json_throughput", |b| {
        b.to_async(&rt).iter(|| async {
            let mut set = JoinSet::new();
            for _ in 0..CONCURRENCY {
                let client = stack.client.clone();
                let url = stack.url.clone();
                let vkey = stack.vkey.clone();
                set.spawn(async move {
                    let resp = client
                        .post(format!("{url}/v1/chat/completions"))
                        .header("authorization", format!("Bearer {vkey}"))
                        .header("content-type", "application/json")
                        .body(MANAGED_BODY)
                        .send()
                        .await
                        .expect("request");
                    let _ = resp.bytes().await.expect("body");
                });
            }
            while let Some(r) = set.join_next().await {
                r.expect("task");
            }
        });
    });

    group.finish();

    // Keep the stacks alive until every bench has run, then tear them down explicitly.
    drop(stack);
    drop(sse_stack);
}

criterion_group!(benches, bench_e2e);
criterion_main!(benches);
