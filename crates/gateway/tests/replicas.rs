//! End-to-end: several gateway replicas sharing one NATS, as production runs them.
//!
//! Every other test runs one gateway. Here two `beyond-ai` processes share one `nats-server` (and
//! one mock upstream), and a client alternates between them, the way a load balancer spreads one
//! caller's requests. What is shared and what is per pod is stated in ARCHITECTURE.md ("Running
//! several replicas"); these tests back it:
//!
//! - **Shared:** the NATS deny/allowance sets (a write lands on every replica within the bound) and
//!   the id signing keys (an id one replica signs verifies on another; rotation is two-phase).
//! - **Computed, so the same everywhere:** session pins. No pod keeps any pin state; each computes
//!   the same provider for a caller every turn from the caller and the row (D253, fixed). The one
//!   residual divergence is an open breaker on one pod during a real outage.
//! - **Per pod:** `tenant_max_in_flight` and the per-credential rate limit (N replicas admit N × the
//!   limit).
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};

const TENANT: u64 = 901;

const RESP: &str = "resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26";

/// An OpenAI Responses answer whose response id is `RESP`.
const RESPONSE: &str = r#"{"id":"resp_0750331520328311006abeeacfb62c87d0bdb6cbe1c41eea26","object":"response","created_at":1759300000,"status":"completed","model":"gpt-4o-2024-08-06","previous_response_id":null,"output":[{"type":"message","id":"msg_0750331520328311006abeead0270c87d0b8128a81f8474fea","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello.","annotations":[]}]}],"usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":7}}"#;

const OK_JSON: &str = r#"{"id":"chatcmpl-ok","object":"chat.completion","model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;

const NEW_SECRET: [u8; 32] = [9; 32];

/// A replica of the Responses deployment: GPT rows on the shared mock, the given id signing keys
/// (kid → secret) and the kid that signs.
async fn responses_replica(
    nats_port: u16,
    mock: &MockUpstream,
    pubkey: &[u8],
    keys: &[(char, &[u8])],
    kid: char,
) -> Gateway {
    Gateway::builder(nats_port, &mock.authority(), &b64(pubkey))
        .providers(&["openai", "openrouter", "anthropic"])
        .id_signing_keys(keys, kid)
        .start()
        .await
}

async fn respond(
    gw: &Gateway,
    sk: &ed25519_dalek::SigningKey,
    previous: Option<&str>,
) -> (u16, String) {
    let mut body = json!({"model": "gpt-4o", "input": "go on"});
    if let Some(id) = previous {
        body["previous_response_id"] = json!(id);
    }
    let resp = test_client()
        .post(format!("{}/v1/responses", gw.url()))
        .header(
            "authorization",
            format!("Bearer {}", billing_vkey(sk, TENANT)),
        )
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&body).unwrap())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap())
}

/// One turn of a chain on `gw`: it must accept `previous` (signed by another replica), hand the
/// upstream the provider's own id, and return the next id, signed under `want_kid`.
async fn turn(
    gw: &Gateway,
    mock: &MockUpstream,
    sk: &ed25519_dalek::SigningKey,
    previous: &str,
    want_kid: char,
    step: &str,
) -> String {
    let (status, text) = respond(gw, sk, Some(previous)).await;
    assert_eq!(status, 200, "{step}: the chain broke: {text}");
    let sent: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
    assert_eq!(
        sent["previous_response_id"], RESP,
        "{step}: the upstream gets the provider id"
    );
    let id = serde_json::from_str::<Value>(&text).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        id.starts_with(&format!("resp_x{want_kid}")),
        "{step}: new ids carry kid {want_kid}: {id}"
    );
    id
}

/// An id replica A signs verifies on replica B (same keys), and back: a conversation that
/// alternates replicas keeps its chain.
/// claim: SEC-25
#[tokio::test]
async fn an_id_signed_by_one_replica_verifies_on_another() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(41);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", RESPONSE)).await;
    let keys: &[(char, &[u8])] = &[('1', &DEV_ID_SECRET)];
    let a = responses_replica(nats_port, &mock, &pubkey, keys, '1').await;
    let b = responses_replica(nats_port, &mock, &pubkey, keys, '1').await;

    let (status, text) = respond(&a, &sk, None).await;
    assert_eq!(status, 200, "{text}");
    let mut id = serde_json::from_str::<Value>(&text).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    for (i, gw) in [&b, &a, &b, &a].into_iter().enumerate() {
        id = turn(gw, &mock, &sk, &id, '1', &format!("turn {}", i + 2)).await;
    }
}

/// The documented two-phase rotation (ARCHITECTURE.md, "Rotation"), done as a rolling deploy of
/// two replicas: phase 1 adds the new key to every replica (still signing with the old kid), phase
/// 2 switches `id_signing_kid`. One conversation alternates replicas throughout, and its chain
/// continues at every step, including the steps where the two replicas sign with different kids.
/// claim: SEC-25
#[tokio::test]
async fn a_two_phase_rotation_keeps_a_chain_across_a_rolling_deploy() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(42);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", RESPONSE)).await;
    let old: &[(char, &[u8])] = &[('1', &DEV_ID_SECRET)];
    let both: &[(char, &[u8])] = &[('1', &DEV_ID_SECRET), ('2', &NEW_SECRET)];

    let mut a = responses_replica(nats_port, &mock, &pubkey, old, '1').await;
    let mut b = responses_replica(nats_port, &mock, &pubkey, old, '1').await;
    let (status, text) = respond(&a, &sk, None).await;
    assert_eq!(status, 200, "{text}");
    let first = serde_json::from_str::<Value>(&text).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let mut id = turn(&b, &mock, &sk, &first, '1', "before").await;

    // Phase 1, replica by replica: add kid 2, keep signing with kid 1.
    a = responses_replica(nats_port, &mock, &pubkey, both, '1').await;
    id = turn(&a, &mock, &sk, &id, '1', "phase 1, A rolled").await;
    id = turn(&b, &mock, &sk, &id, '1', "phase 1, A rolled").await;
    b = responses_replica(nats_port, &mock, &pubkey, both, '1').await;
    id = turn(&a, &mock, &sk, &id, '1', "phase 1 done").await;
    id = turn(&b, &mock, &sk, &id, '1', "phase 1 done").await;

    // Phase 2, replica by replica: switch the signing kid. The replicas now sign with different
    // kids, and each verifies the other's.
    a = responses_replica(nats_port, &mock, &pubkey, both, '2').await;
    id = turn(&a, &mock, &sk, &id, '2', "phase 2, A rolled").await;
    id = turn(&b, &mock, &sk, &id, '1', "phase 2, A rolled").await;
    id = turn(&a, &mock, &sk, &id, '2', "phase 2, A rolled").await;
    b = responses_replica(nats_port, &mock, &pubkey, both, '2').await;
    id = turn(&b, &mock, &sk, &id, '2', "phase 2 done").await;
    turn(&a, &mock, &sk, &id, '2', "phase 2 done").await;

    // An id issued before the rotation still verifies everywhere while kid 1 stays listed.
    turn(&a, &mock, &sk, &first, '2', "old id on A").await;
    turn(&b, &mock, &sk, &first, '2', "old id on B").await;
}

/// The warning behind the order: switch `id_signing_kid` on one replica before every replica has
/// the new key, and an id that replica signs is refused by a replica without it, with the named 400
/// (`previous_response_id does not belong to this tenant`), before any upstream call.
/// claim: SEC-25
#[tokio::test]
async fn switching_the_kid_before_every_replica_has_the_key_breaks_chains() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(43);
    let mock = MockUpstream::start(Mode::Raw(200, "application/json", RESPONSE)).await;
    let old = responses_replica(nats_port, &mock, &pubkey, &[('1', &DEV_ID_SECRET)], '1').await;
    let early = responses_replica(
        nats_port,
        &mock,
        &pubkey,
        &[('1', &DEV_ID_SECRET), ('2', &NEW_SECRET)],
        '2',
    )
    .await;

    let (status, text) = respond(&early, &sk, None).await;
    assert_eq!(status, 200, "{text}");
    let id = serde_json::from_str::<Value>(&text).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(id.starts_with("resp_x2"), "{id}");
    let hits = mock.hits();
    let (status, text) = respond(&old, &sk, Some(&id)).await;
    assert_eq!(status, 400, "{text}");
    assert!(
        text.contains("previous_response_id")
            && text.contains("does not belong to this tenant")
            && text.contains("invalid_request_error"),
        "{text}"
    );
    assert_eq!(mock.hits(), hits, "a refused id never reaches the upstream");
    assert!(
        parse_metric(&old.metrics().await, "ai_rejections_total", "foreign_id") >= 1.0,
        "the old replica counts it as a foreign id"
    );
}

async fn chat(gw: &Gateway, key: &str) -> u16 {
    test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0)
}

/// Poll `gw` until it answers `want` for `key`, or the bound passes. Returns how long it took.
async fn lands_within(gw: &Gateway, key: &str, want: u16, bound: Duration, what: &str) {
    let start = Instant::now();
    let mut last = 0;
    while start.elapsed() < bound {
        last = chat(gw, key).await;
        if last == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("{what}: still {last} after {bound:?}, want {want}");
}

/// A tenant deny or an exhausted allowance written once to the shared NATS KV refuses that tenant
/// on **both** replicas within the documented 2s bound (D121, SEC-17), and the delete restores
/// both. Another tenant is untouched.
/// claim: SEC-17, A2
#[tokio::test]
async fn a_deny_and_an_allowance_land_on_every_replica_within_the_bound() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(44);
    let mock = MockUpstream::start(Mode::Json).await;
    let a = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let b = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let (key, other) = (billing_vkey(&sk, 4401), billing_vkey(&sk, 4402));
    for gw in [&a, &b] {
        assert_eq!(chat(gw, &key).await, 200);
    }
    let bound = Duration::from_secs(2);

    put_kv(nats.port, "blackhole.4401", b"spend").await;
    for (gw, name) in [(&a, "A"), (&b, "B")] {
        lands_within(gw, &key, 402, bound, &format!("deny on {name}")).await;
        assert_eq!(chat(gw, &other).await, 200, "another tenant on {name}");
    }
    del_kv(nats.port, "blackhole.4401").await;
    for (gw, name) in [(&a, "A"), (&b, "B")] {
        lands_within(gw, &key, 200, bound, &format!("deny lifted on {name}")).await;
    }

    put_kv(nats.port, "allowance.4401", b"exhausted").await;
    for (gw, name) in [(&a, "A"), (&b, "B")] {
        lands_within(gw, &key, 402, bound, &format!("allowance on {name}")).await;
        assert_eq!(chat(gw, &other).await, 200, "another tenant on {name}");
    }
    del_kv(nats.port, "allowance.4401").await;
    for (gw, name) in [(&a, "A"), (&b, "B")] {
        lands_within(
            gw,
            &key,
            200,
            bound,
            &format!("allowance restored on {name}"),
        )
        .await;
    }
}

async fn post_auto(gw: &Gateway, key: &str) -> String {
    let resp = test_client()
        .post(format!("{}/auto/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", "gpt-4o-mini")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let provider = resp
        .headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let _ = resp.bytes().await;
    provider
}

/// Two replicas whose outcomes differ for one key: replica A saw the primary fail once and failed
/// over to the fallback for that request; replica B never saw the failure and served the primary.
/// Both providers are healthy from then on. Before D253's fix each pod remembered its own pin (A
/// the fallback, B the primary) for up to an hour.
async fn split_pins() -> (ReplyUpstream, MockUpstream, Gateway, Gateway, String) {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(45);
    // One transient 5xx (the first request anywhere), then healthy.
    let primary = ReplyUpstream::start(|n, _| {
        if n == 0 {
            Reply::json(500, r#"{"error":{"message":"transient"}}"#)
        } else {
            Reply::json(200, OK_JSON)
        }
    })
    .await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let replica = || {
        Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .config_line("circuit_breaker_threshold = 100")
            .start()
    };
    let a = replica().await;
    let b = replica().await;
    let key = billing_vkey(&sk, 4501);
    assert_eq!(
        post_auto(&a, &key).await,
        "openrouter",
        "A fails over to the fallback for that request"
    );
    assert_eq!(post_auto(&b, &key).await, "openai", "B serves the primary");
    (primary, fallback, a, b, key)
}

async fn gw_pinned(gw: &Gateway) -> f64 {
    gw.metric("ai_session_pinned_total", "").await
}

/// What R4 promises a conversation spread over replicas: one provider every turn, whichever pod
/// takes it, while that provider is usable. The pin is computed from the caller and the row every
/// turn, and A's failure moved only the request that met it, so A is back on the primary at once.
/// claim: R4
/// defect: D253
#[tokio::test]
async fn a_pinned_conversation_stays_on_one_provider_across_replicas() {
    let (_primary, _fallback, a, b, key) = split_pins().await;
    let mut served = Vec::new();
    for _ in 0..5 {
        served.push(post_auto(&a, &key).await);
        served.push(post_auto(&b, &key).await);
    }
    assert!(
        served.windows(2).all(|w| w[0] == w[1]),
        "a healthy pinned conversation changed provider between replicas: {served:?}"
    );
    assert_eq!(served[0], "openai", "the catalog primary, not the failover");
    assert!(
        gw_pinned(&a).await >= 5.0 && gw_pinned(&b).await >= 5.0,
        "both pods routed the turns by the pin"
    );
}

async fn post_claude(gw: &Gateway, key: &str) -> String {
    let resp = test_client()
        .post(format!("{}/auto/v1/messages", gw.url()))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .header("x-beyond-model", "claude-opus-4-8")
        .body(r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let provider = resp
        .headers()
        .get("x-beyond-provider")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let _ = resp.bytes().await;
    provider
}

/// A pooled row (Anthropic + Bedrock, OpenRouter as failover) over two replicas: every session
/// reaches one provider on both pods for 20 alternating turns, its very first included, although
/// the pods' TTFT samples favour Bedrock (the Anthropic mock is slower). Sessions spread over both
/// first-party hosts by hash, and none lands on the aggregator while they are usable.
/// claim: R4
/// defect: D253
#[tokio::test]
async fn a_pooled_session_reaches_one_provider_on_every_replica() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(48);
    let anthropic = MockUpstream::start(Mode::Slow(20)).await;
    let bedrock = MockUpstream::start(Mode::AnthropicJson).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let replica = || {
        Gateway::builder(nats_port, &anthropic.authority(), &b64(&pubkey))
            .providers(&["anthropic", "bedrock", "openrouter"])
            .provider_authority("bedrock", &bedrock.authority())
            .provider_authority("openrouter", &openrouter.authority())
            .start()
    };
    let a = replica().await;
    let b = replica().await;
    let mut by_provider = std::collections::BTreeMap::<String, usize>::new();
    for tenant in 4800..4810 {
        let key = billing_vkey(&sk, tenant);
        let mut served = Vec::new();
        for _ in 0..10 {
            served.push(post_claude(&a, &key).await);
            served.push(post_claude(&b, &key).await);
        }
        assert!(
            served.windows(2).all(|w| w[0] == w[1]),
            "session {tenant} changed provider between replicas: {served:?}"
        );
        *by_provider.entry(served[0].clone()).or_default() += 1;
    }
    assert!(
        !by_provider.contains_key("openrouter"),
        "a pin was hashed onto the aggregator: {by_provider:?}"
    );
    assert_eq!(
        by_provider.len(),
        2,
        "sessions did not spread over both first-party hosts: {by_provider:?}"
    );
}

/// A failure fails over for that request only: while the primary keeps failing each turn fails
/// over in-gateway (the client never sees it), and the first turn after the primary answers again
/// is back on it, on both replicas.
/// claim: R4, R1
/// defect: D253
#[tokio::test]
async fn a_failover_pin_returns_to_the_primary_once_it_is_healthy() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(49);
    // The primary's first three attempts fail, then it is healthy.
    let primary = ReplyUpstream::start(|n, _| {
        if n < 3 {
            Reply::json(500, r#"{"error":{"message":"outage"}}"#)
        } else {
            Reply::json(200, OK_JSON)
        }
    })
    .await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let replica = || {
        Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .config_line("circuit_breaker_threshold = 100")
            .start()
    };
    let a = replica().await;
    let b = replica().await;
    let key = billing_vkey(&sk, 4901);
    let mut served = Vec::new();
    for _ in 0..4 {
        served.push(post_auto(&a, &key).await);
        served.push(post_auto(&b, &key).await);
    }
    assert_eq!(
        served[..3],
        ["openrouter"; 3],
        "the outage turns fail over: {served:?}"
    );
    assert!(
        served[3..].iter().all(|p| p == "openai"),
        "the conversation returns to the primary once it answers: {served:?}"
    );
}

/// The residual divergence, stated: a replica whose breaker for the preferred host is open routes
/// the session to the next candidate, and only while the breaker is open. The other replica, whose
/// breaker never opened, keeps serving the preferred host; once the open breaker's probe succeeds,
/// both are back on it.
/// claim: R4, R6
/// defect: D253
#[tokio::test]
async fn a_replica_with_an_open_breaker_routes_away_only_while_it_is_open() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(50);
    let failing = Arc::new(AtomicBool::new(true));
    let primary = ReplyUpstream::start({
        let failing = failing.clone();
        move |_, _| {
            if failing.load(Ordering::Relaxed) {
                Reply::json(500, r#"{"error":{"message":"outage"}}"#)
            } else {
                Reply::json(200, OK_JSON)
            }
        }
    })
    .await;
    let fallback = MockUpstream::start(Mode::Json).await;
    let replica = || {
        Gateway::builder(nats_port, &primary.authority(), &b64(&pubkey))
            .providers(&["openai", "openrouter"])
            .provider_authority("openrouter", &fallback.authority())
            .config_line("circuit_breaker_threshold = 2")
            .config_line("circuit_breaker_window_secs = 60")
            .config_line("circuit_breaker_reset_secs = 2")
            .start()
    };
    let a = replica().await;
    let b = replica().await;
    let key = billing_vkey(&sk, 5001);
    // Only A meets the outage: two failed-over turns open its breaker for the primary.
    for _ in 0..2 {
        assert_eq!(post_auto(&a, &key).await, "openrouter");
    }
    failing.store(false, Ordering::Relaxed);
    let hits = primary.hits();
    for turn in 0..3 {
        assert_eq!(
            post_auto(&a, &key).await,
            "openrouter",
            "turn {turn}: A's breaker is open, so A skips the primary"
        );
        assert_eq!(
            post_auto(&b, &key).await,
            "openai",
            "turn {turn}: B's breaker is closed, so B keeps the preferred host"
        );
    }
    assert_eq!(primary.hits(), hits + 3, "only B reached the primary");
    // Past the reset the half-open probe succeeds and A is back on the primary.
    let start = Instant::now();
    let mut on_a = String::new();
    while start.elapsed() < Duration::from_secs(10) {
        on_a = post_auto(&a, &key).await;
        if on_a == "openai" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert_eq!(on_a, "openai", "A returns once its breaker closes");
    for _ in 0..5 {
        assert_eq!(post_auto(&a, &key).await, "openai");
        assert_eq!(post_auto(&b, &key).await, "openai");
    }
}

/// `tenant_max_in_flight` is per process: with a cap of 1 on each of two replicas, a tenant holding
/// its one slot on A is refused a second there (429) and still admitted on B. N replicas admit
/// N × the cap, so the overspend bound scales with the replica count.
/// claim: A3
#[tokio::test]
async fn the_tenant_in_flight_cap_is_per_replica() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(46);
    let mock = MockUpstream::start(Mode::Slow(1_500)).await;
    let replica = || {
        Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
            .tenant_max_in_flight(1)
            .start()
    };
    let a = replica().await;
    let b = replica().await;
    let key = billing_vkey(&sk, 4601);

    let held_a = tokio::spawn({
        let (url, key) = (a.url(), key.clone());
        async move { chat_url(&url, &key).await }
    });
    wait_hits(&mock, 1).await;
    assert_eq!(chat(&a, &key).await, 429, "A is at the tenant's cap");
    let held_b = tokio::spawn({
        let (url, key) = (b.url(), key.clone());
        async move { chat_url(&url, &key).await }
    });
    wait_hits(&mock, 2).await;
    assert_eq!(chat(&b, &key).await, 429, "B has its own cap of 1");
    assert_eq!(held_a.await.unwrap(), 200);
    assert_eq!(held_b.await.unwrap(), 200);
    assert_eq!(mock.hits(), 2, "two in flight at once: one per replica");
}

async fn chat_url(url: &str, key: &str) -> u16 {
    test_client()
        .post(format!("{url}/openai/v1/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0)
}

async fn wait_hits(mock: &MockUpstream, n: usize) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while mock.hits() < n {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("request reached the upstream");
}

/// `rate_limit_rps` is per process: a credential refused (429) by replica A is still admitted by
/// replica B, which has counted none of its requests. N replicas admit N × the rate.
/// claim: A3
#[tokio::test]
async fn the_per_credential_rate_limit_is_per_replica() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(47);
    let mock = MockUpstream::start(Mode::Json).await;
    const RPS: u32 = 3;
    let replica = || {
        Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
            .rate_limit_rps(RPS)
            .start()
    };
    let a = replica().await;
    let b = replica().await;
    let key = billing_vkey(&sk, 4701);

    let mut refused = false;
    for _ in 0..(RPS * 10) {
        if chat(&a, &key).await == 429 {
            refused = true;
            break;
        }
    }
    assert!(refused, "A refuses the credential past {RPS} req/s");
    // B has seen none of these requests: its window holds 0, so its own RPS are admitted now.
    for i in 0..RPS {
        assert_eq!(chat(&b, &key).await, 200, "B request {i} after A's 429");
    }
}
