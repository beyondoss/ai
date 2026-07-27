// Bench target: `.unwrap()`/`.expect()` set up the harness; not production code. See tests/e2e.rs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::Runtime;
use tokio::task::JoinSet;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

const MANAGED_BODY: &str = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;

/// Model-routed body. Same size as `MANAGED_BODY` so `auto_json_latency` minus
/// `managed_json_latency` is the model route's own cost and not a payload difference.
const AUTO_BODY: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;

/// A plausible BYO provider token (anything not starting with `bai_` is BYO — passed through
/// unchanged, no verify/deny/swap). The mock upstream accepts any token.
const BYO_KEY: &str = "sk-byo-provider-token-1234567890";

/// A realistically-sized chat body (~64 KiB of message content, `model` last so a structural scan
/// must walk the whole thing). `MANAGED_BODY` is 60 bytes, which makes every body-proportional cost
/// in the request path invisible — the same blind spot that hid the response-side findings.
fn large_body() -> String {
    let content = "x".repeat(64 * 1024);
    format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"model":"gpt-4o"}}"#)
}

/// The `large_body` shape on the model route: same size, `model` named as the catalog row so the
/// rewrite path runs. This is where `/auto`'s whole-body buffering has to show up if anywhere.
fn large_auto_body() -> String {
    let content = "x".repeat(64 * 1024);
    format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"model":"gpt-4o-mini"}}"#)
}

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
    // Rate limits OFF, both tiers. criterion drives one credential at thousands of requests per
    // second — far past the 100 rps default — and a throttled request 429s in `request_filter`
    // *before* the upstream, so with the ceiling in place these benches were timing the reject path
    // and reporting it as proxy latency. `bench_concurrency` already did this; `bench_e2e` did not.
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .rate_limit_rps(0)
        .byo_rate_limit_rps(0)
        .start()
        .await;
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

/// A stack wired for the **model route**: `openai` and `openrouter` both point at the mock, so the
/// catalog's `gpt-4o-mini` row resolves and both candidates are usable.
async fn start_auto_stack() -> Stack {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .rate_limit_rps(0)
        .byo_rate_limit_rps(0)
        .start()
        .await;
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
    {
        let (c, u, k) = (client.clone(), url.clone(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                c.post(format!("{u}/auto/chat/completions"))
                    .header("authorization", format!("Bearer {k}"))
                    .header("content-type", "application/json")
                    .header("x-beyond-model", "gpt-4o-mini")
                    .body(AUTO_BODY)
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

/// A model-routed stack whose **primary candidate is dead**, so every request walks to the fallback.
/// The delta against `auto_json_latency` is what a failover costs the client.
async fn start_auto_failover_stack() -> Stack {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &GatewayBuilder::dead_authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &mock.authority())
        .rate_limit_rps(0)
        .byo_rate_limit_rps(0)
        .start()
        .await;
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
    {
        let (c, u, k) = (client.clone(), url.clone(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                c.post(format!("{u}/auto/chat/completions"))
                    .header("authorization", format!("Bearer {k}"))
                    .header("content-type", "application/json")
                    .header("x-beyond-model", "gpt-4o-mini")
                    .body(AUTO_BODY)
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

/// One model-routed round-trip. Same work as `managed_roundtrip` plus: a catalog lookup, the
/// candidate walk, whole-body buffering, and the `model` splice. The delta against
/// `managed_json_latency` is the price of the model route.
async fn auto_roundtrip(s: &Stack, body: &str) {
    let resp = s
        .client
        .post(format!("{}/auto/chat/completions", s.url))
        .header("authorization", format!("Bearer {}", s.vkey))
        .header("content-type", "application/json")
        .header("x-beyond-model", "gpt-4o-mini")
        .body(body.to_string())
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "bench measured a non-200; it is timing the wrong path"
    );
    let _ = resp.bytes().await.expect("body");
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
    assert_eq!(
        resp.status().as_u16(),
        200,
        "bench measured a non-200; it is timing the wrong path"
    );
    let _ = resp.bytes().await.expect("body");
}

/// A stack configured for the **Anthropic** dialect: the `anthropic` provider, reached at
/// `/v1/messages` with the key in `x-api-key`. Needed because the default stack configures
/// openai/fireworks and drives `/v1/chat/completions`, which would parse an Anthropic stream with
/// the OpenAI parser and never touch the Anthropic-specific response path at all.
async fn start_anthropic_stack(mode: Mode) -> Stack {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(mode).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .rate_limit_rps(0)
        .byo_rate_limit_rps(0)
        .start()
        .await;
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
    {
        let (c, u, k) = (client.clone(), url.clone(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                c.post(format!("{u}/v1/messages"))
                    .header("x-api-key", &k)
                    .header("content-type", "application/json")
                    .body(r#"{"model":"claude-opus-4-8","messages":[]}"#)
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

/// One managed Anthropic round-trip: `/v1/messages`, key in `x-api-key`, Anthropic-dialect usage
/// parsing over a head **and** tail window.
async fn anthropic_roundtrip(s: &Stack) {
    let resp = s
        .client
        .post(format!("{}/v1/messages", s.url))
        .header("x-api-key", &s.vkey)
        .header("content-type", "application/json")
        .body(r#"{"model":"claude-opus-4-8","messages":[]}"#)
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status().as_u16(), 200);
    let body = resp.bytes().await.expect("body");
    assert!(
        body.len() > 128 * 1024,
        "expected the large stream, got {} B",
        body.len()
    );
}

/// One round-trip with an arbitrary key and body, so a bench can vary either. `key` decides the
/// path: a `bai_…` virtual key is managed, anything else is BYO.
async fn roundtrip_with(s: &Stack, key: &str, body: &str) {
    let resp = s
        .client
        .post(format!("{}/v1/chat/completions", s.url))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "bench measured a non-200; it is timing the wrong path"
    );
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
    assert_eq!(
        resp.status().as_u16(),
        200,
        "bench measured a non-200; it is timing the wrong path"
    );
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
    assert_eq!(
        resp.status().as_u16(),
        401,
        "reject bench must actually reject"
    );
    let _ = resp.bytes().await.expect("body");
}

fn bench_e2e(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let stack = rt.block_on(start_stack());
    // A second stack whose mock streams SSE. `Mode::Sse` is a *three-line* canned stream, so despite
    // what this comment used to claim it does not exercise the response tap's bounded window at all
    // — the tail never fills, nothing is ever evicted, and the usage parser sees two `data:` lines.
    // It is kept as the cheap "is a stream slower than a body" datapoint; the two `*_large_sse_*`
    // stacks below are what actually cover the tap.
    let sse_stack = rt.block_on(start_stack_with(Mode::Sse));
    // >128 KiB OpenAI stream: fills and wraps the response tail, so the usage chunk has to survive
    // eviction. This fixture existed but was only ever used by one integration test, never benched —
    // which is why an O(response-size) memmove and a per-line JSON parse in the tail went unnoticed.
    let large_sse_stack = rt.block_on(start_stack_with(Mode::SseLarge));
    // ~600 KiB Anthropic stream: the shape that splits its usage facts across the *first* and
    // *last* events, so it covers the head buffer as well as the tail.
    let anthropic_sse_stack = rt.block_on(start_anthropic_stack(Mode::AnthropicSseLarge));

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

    // Streams large enough to actually drive the response tap: the tail wraps, the model scanner has
    // to terminate early rather than walk the whole body, and the usage parser has to find its event
    // in a full window. Read these against `managed_sse_latency` — the gap between a 3-line stream
    // and these is the response path's size sensitivity, which nothing measured before.
    group.bench_function("managed_large_sse_latency", |b| {
        b.to_async(&rt).iter(|| managed_roundtrip(&large_sse_stack));
    });
    group.bench_function("managed_large_anthropic_sse_latency", |b| {
        b.to_async(&rt)
            .iter(|| anthropic_roundtrip(&anthropic_sse_stack));
    });

    // The same two paths at a realistic body size, since `MANAGED_BODY` is 60 bytes and hides every
    // body-proportional cost in the request path.
    //
    // Do not read a managed-vs-BYO gap out of these two rows: measured at 64 KiB they land inside
    // each other's confidence intervals (120.2 µs vs 124.2 µs), because a loopback round-trip is
    // dominated by HTTP and syscall time and swamps a few microseconds of body scanning. The
    // per-request scan cost is isolated properly by `peek::scan_model_last` /
    // `peek::scan_response_stream` in the unit bench; these exist to catch a *gross* regression in
    // the whole path at a size the suite otherwise never exercises.
    let big = large_body();
    group.bench_function("managed_large_body_latency", |b| {
        b.to_async(&rt)
            .iter(|| roundtrip_with(&stack, &stack.vkey, &big));
    });
    group.bench_function("byo_large_body_latency", |b| {
        b.to_async(&rt)
            .iter(|| roundtrip_with(&stack, BYO_KEY, &big));
    });

    // The model route. `auto_json_latency` against `managed_json_latency` isolates its fixed cost
    // (catalog lookup, candidate walk, the `Box<ModelRouting>`); `auto_large_body_latency` against
    // `managed_large_body_latency` isolates the part that scales with the body, which is where
    // `/auto`'s whole-body buffering and the `model` splice live. Neither had ever been measured.
    let auto_stack = rt.block_on(start_auto_stack());
    let big_auto = large_auto_body();
    group.bench_function("auto_json_latency", |b| {
        b.to_async(&rt)
            .iter(|| auto_roundtrip(&auto_stack, AUTO_BODY));
    });
    group.bench_function("auto_large_body_latency", |b| {
        b.to_async(&rt)
            .iter(|| auto_roundtrip(&auto_stack, &big_auto));
    });
    // Every request here loses its primary to a refused connection and is served by the fallback,
    // so the delta against `auto_json_latency` is what a failover costs the client.
    let auto_failover_stack = rt.block_on(start_auto_failover_stack());
    group.bench_function("auto_failover_latency", |b| {
        b.to_async(&rt)
            .iter(|| auto_roundtrip(&auto_failover_stack, AUTO_BODY));
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

    // Throughput for the shape this gateway actually carries in production: a long Anthropic
    // stream, under concurrency. `managed_json_throughput` measures a 60-byte body and a 250-byte
    // response, which is the cheapest possible request and so mostly measures accept/parse; this one
    // measures the response path doing real work on many streams at once.
    group.throughput(Throughput::Elements(CONCURRENCY));
    group.bench_function("managed_large_anthropic_sse_throughput", |b| {
        b.to_async(&rt).iter(|| async {
            let mut set = JoinSet::new();
            for _ in 0..CONCURRENCY {
                let (client, url, vkey) = (
                    anthropic_sse_stack.client.clone(),
                    anthropic_sse_stack.url.clone(),
                    anthropic_sse_stack.vkey.clone(),
                );
                set.spawn(async move {
                    let resp = client
                        .post(format!("{url}/v1/messages"))
                        .header("x-api-key", &vkey)
                        .header("content-type", "application/json")
                        .body(r#"{"model":"claude-opus-4-8","messages":[]}"#)
                        .send()
                        .await
                        .expect("request");
                    assert_eq!(resp.status().as_u16(), 200);
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
    drop(large_sse_stack);
    drop(anthropic_sse_stack);
}

/// Concurrency levels swept by `bench_concurrency`. Spans below and above hyper's default
/// `SETTINGS_MAX_CONCURRENT_STREAMS` (200) so an H2 stream-concurrency cliff (if any) shows up against
/// H1's connection pool.
const SWEEP: &[u64] = &[1, 8, 32, 128, 512];

/// Fire `conc` managed requests at `url` concurrently and drain each body (returns the connection to
/// the pool). This is one bench iteration; `Throughput::Elements(conc)` makes criterion report req/s.
async fn drive(client: &reqwest::Client, url: &str, vkey: &str, conc: u64) {
    let mut set = JoinSet::new();
    for _ in 0..conc {
        let (c, u, k) = (client.clone(), url.to_string(), vkey.to_string());
        set.spawn(async move {
            let resp = c
                .post(format!("{u}/v1/chat/completions"))
                .header("authorization", format!("Bearer {k}"))
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
}

/// Warm a gateway until it answers 200, then return the protocol it used to reach the upstream — read
/// from the `x-mock-proto` header the TLS mock stamps and the gateway relays. This is the proof the
/// "h2"/"h1" bench labels reflect what actually negotiated, not just what we configured.
async fn warm_and_proto(client: &reqwest::Client, url: &str, vkey: &str) -> String {
    {
        let (c, u, k) = (client.clone(), url.to_string(), vkey.to_string());
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
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(MANAGED_BODY)
        .send()
        .await
        .expect("warm request");
    resp.headers()
        .get("x-mock-proto")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string()
}

/// H2-vs-H1 to the upstream, under concurrency. One TLS+H2 mock; two gateways against it — one with
/// `upstream_http2 = true` (ALPN H2H1 → h2), one `false` (ALPN H1). Same client→gateway transport
/// (plain H1) for both, so the only variable is the gateway→upstream protocol. The sweep exposes
/// whether H2 multiplexing wins or hits its stream-concurrency cap vs H1's connection pool.
fn bench_concurrency(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let nats = rt.block_on(Nats::start());
    let mock = rt.block_on(MockUpstream::start_tls(Mode::Json));
    let (pubkey, sk) = test_keypair(1);
    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
        },
        1,
        &sk,
    );

    // Two gateways at the same self-signed TLS mock; ALPN is the only difference. Rate limits OFF
    // (both tiers): the sweep drives one credential well past the 100 rps default, and a rate-limited
    // 429 short-circuits *before* the upstream — it would measure the reject path, not H2-vs-H1.
    let gw_h2 = rt.block_on(
        Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
            .tls_upstream()
            .upstream_http2(true)
            .rate_limit_rps(0)
            .byo_rate_limit_rps(0)
            .start(),
    );
    let gw_h1 = rt.block_on(
        Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
            .tls_upstream()
            .upstream_http2(false)
            .rate_limit_rps(0)
            .byo_rate_limit_rps(0)
            .start(),
    );
    let client = reqwest::Client::new();
    let (url_h2, url_h1) = (gw_h2.url(), gw_h1.url());

    // Prove the gateways actually negotiated what we asked for before trusting the labels.
    let proto_h2 = rt.block_on(warm_and_proto(&client, &url_h2, &vkey));
    let proto_h1 = rt.block_on(warm_and_proto(&client, &url_h1, &vkey));
    assert_eq!(
        proto_h2, "h2",
        "upstream_http2=true should negotiate h2 to the mock"
    );
    assert_eq!(
        proto_h1, "http/1.1",
        "upstream_http2=false should stay http/1.1 to the mock"
    );
    eprintln!("e2e_concurrency: confirmed gw_h2→upstream=h2, gw_h1→upstream=http/1.1");

    let mut group = c.benchmark_group("e2e_concurrency");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(6));
    for &conc in SWEEP {
        group.throughput(Throughput::Elements(conc));
        group.bench_with_input(BenchmarkId::new("h2", conc), &conc, |b, &conc| {
            b.to_async(&rt)
                .iter(|| drive(&client, &url_h2, &vkey, conc));
        });
        group.bench_with_input(BenchmarkId::new("h1", conc), &conc, |b, &conc| {
            b.to_async(&rt)
                .iter(|| drive(&client, &url_h1, &vkey, conc));
        });
    }
    group.finish();

    drop(gw_h2);
    drop(gw_h1);
    drop(mock);
    drop(nats);
}

/// Proxy worker-thread scaling: one gateway pinned to a single worker (Pingora's own
/// `ServerConf::default()`, which is what this crate silently inherited until `worker_threads` was
/// wired up in `main.rs`) against one sized per core. Same mock, same client transport, same config
/// otherwise — the thread count is the only variable.
///
/// This is the bench that would have caught it: a service that leaves its `threads` as `None` gets
/// `conf.threads` = 1, so *every* request filter, the Ed25519 verify, both body scanners and the
/// usage tap ran on one core regardless of box size. Nothing in the suite measured above one
/// connection's worth of concurrency, so a hard 1-core ceiling looked exactly like a fast gateway.
///
/// Read it at the high end of the sweep: at concurrency 1 the two are identical by construction
/// (one in-flight request cannot use two cores), and the gap only opens once there is real parallel
/// work to place.
fn bench_worker_threads(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");
    let nats = rt.block_on(Nats::start());
    let mock = rt.block_on(MockUpstream::start(Mode::Json));
    let (pubkey, sk) = test_keypair(1);
    let vkey = mint(
        &VirtualKey {
            tenant_id: 42,
            vpc_id: 7,
        },
        1,
        &sk,
    );

    // Rate limits off on both, for the same reason as `bench_concurrency`: the sweep drives one
    // credential far past the 100 rps default, and a 429 short-circuits before the work we're
    // measuring — it would compare reject paths, not proxy throughput.
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let gw_one = rt.block_on(
        Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
            .worker_threads(1)
            .rate_limit_rps(0)
            .byo_rate_limit_rps(0)
            .start(),
    );
    let gw_many = rt.block_on(
        Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
            .worker_threads(cores)
            .rate_limit_rps(0)
            .byo_rate_limit_rps(0)
            .start(),
    );
    let client = reqwest::Client::new();
    let (url_one, url_many) = (gw_one.url(), gw_many.url());
    rt.block_on(warm_and_proto(&client, &url_one, &vkey));
    rt.block_on(warm_and_proto(&client, &url_many, &vkey));
    eprintln!("e2e_worker_threads: comparing 1 worker vs {cores} workers");

    let mut group = c.benchmark_group("e2e_worker_threads");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(6));
    for &conc in SWEEP {
        group.throughput(Throughput::Elements(conc));
        group.bench_with_input(BenchmarkId::new("threads_1", conc), &conc, |b, &conc| {
            b.to_async(&rt)
                .iter(|| drive(&client, &url_one, &vkey, conc));
        });
        group.bench_with_input(
            BenchmarkId::new(format!("threads_{cores}"), conc),
            &conc,
            |b, &conc| {
                b.to_async(&rt)
                    .iter(|| drive(&client, &url_many, &vkey, conc));
            },
        );
    }
    group.finish();

    drop(gw_one);
    drop(gw_many);
    drop(mock);
    drop(nats);
}

criterion_group!(benches, bench_e2e, bench_concurrency, bench_worker_threads);
criterion_main!(benches);
