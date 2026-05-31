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
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
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

fn bench_e2e(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let stack = rt.block_on(start_stack());

    let mut group = c.benchmark_group("e2e");
    // Real round-trips are sub-millisecond on loopback but still ~100× a micro-bench; trim the
    // sample count so the suite stays in the seconds, not minutes.
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // Single-request latency through the full proxy.
    group.bench_function("managed_json_latency", |b| {
        b.to_async(&rt).iter(|| managed_roundtrip(&stack));
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

    // Keep the stack alive until every bench has run, then tear it down explicitly.
    drop(stack);
}

criterion_group!(benches, bench_e2e);
criterion_main!(benches);
