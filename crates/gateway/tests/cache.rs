//! End-to-end: exact-match response cache on managed `/v1`.
//!
//! Off by default (`cache_ttl_secs = 0`). These tests turn it on. Lookup only happens where the
//! client body is already in hand before `upstream_peer` (headerless managed catalog walk). A hit
//! must be byte-identical to the stored 2xx and must not touch the upstream, the breaker, or the
//! key-walk.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn body() -> String {
    r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#.to_string()
}

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 7,
            key_id: None,
        },
        1,
        sk,
    )
}

async fn post(
    client: &reqwest::Client,
    url: &str,
    key: &str,
    body: String,
    extra: &[(&str, &str)],
) -> reqwest::Response {
    let mut req = client
        .post(format!("{url}/v1/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json");
    for (k, v) in extra {
        req = req.header(*k, *v);
    }
    req.body(body).send().await.unwrap()
}

#[tokio::test]
async fn two_identical_managed_v1_requests_hit_the_cache() {
    let nats = unused_nats_port();
    let (pubkey, sk) = test_keypair(11);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .cache_ttl_secs(60)
        .start()
        .await;
    let key = vkey(&sk, 11);
    let client = test_client();
    {
        let (c, u, k) = (client.clone(), gw.url(), key.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move { post(&c, &u, &k, body(), &[]).await.status().as_u16() }
        })
        .await;
    }
    // logging inserts after the response is fully written; wait so the next request can hit.
    let _ = gw.wait_for_log_line(&["ai.usage"]).await;

    let first = post(&client, &gw.url(), &key, body(), &[]).await;
    let first_status = first.status().as_u16();
    let first_body = first.bytes().await.unwrap();
    let hits_after_fill = mock.hits();
    assert!(hits_after_fill >= 1, "fill must have reached upstream");

    let second = post(&client, &gw.url(), &key, body(), &[]).await;
    let second_status = second.status().as_u16();
    let second_body = second.bytes().await.unwrap();

    assert_eq!(first_status, 200);
    assert_eq!(
        second_status, first_status,
        "hit status must match the fill"
    );
    assert_eq!(
        second_body, first_body,
        "hit body must be byte-identical to the fill"
    );
    assert_eq!(
        mock.hits(),
        hits_after_fill,
        "the second request must not reach the upstream (got {} hits, fill was {})",
        mock.hits(),
        hits_after_fill
    );
    wait_for_metric(&gw, "ai_cache_hits_total", "", 1.0).await;
    let line = gw
        .wait_for_log_line(&["ai.usage", "cache_hit", "true"])
        .await;
    assert!(
        line.contains("\"input_tokens\":11") || line.contains("input_tokens\":11"),
        "cache hit must emit the stored tokens: {line}"
    );
}

#[tokio::test]
async fn different_tenant_is_a_cache_miss() {
    let nats = unused_nats_port();
    let (pubkey, sk) = test_keypair(12);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .cache_ttl_secs(60)
        .start()
        .await;
    let a = vkey(&sk, 12);
    let b = vkey(&sk, 13);
    let client = test_client();
    {
        let (c, u, k) = (client.clone(), gw.url(), a.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move { post(&c, &u, &k, body(), &[]).await.status().as_u16() }
        })
        .await;
    }
    let _ = gw.wait_for_log_line(&["ai.usage"]).await;
    let hits_after_a = mock.hits();
    let resp = post(&client, &gw.url(), &b, body(), &[]).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert!(
        mock.hits() > hits_after_a,
        "a different tenant must miss and go upstream (hits {} → {})",
        hits_after_a,
        mock.hits()
    );
}

#[tokio::test]
async fn x_beyond_cache_off_always_goes_upstream() {
    let nats = unused_nats_port();
    let (pubkey, sk) = test_keypair(14);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .cache_ttl_secs(60)
        .start()
        .await;
    let key = vkey(&sk, 14);
    let client = test_client();
    {
        let (c, u, k) = (client.clone(), gw.url(), key.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                post(&c, &u, &k, body(), &[("x-beyond-cache", "off")])
                    .await
                    .status()
                    .as_u16()
            }
        })
        .await;
    }
    let _ = gw.wait_for_log_line(&["ai.usage"]).await;
    let hits_after_first = mock.hits();
    let resp = post(
        &client,
        &gw.url(),
        &key,
        body(),
        &[("x-beyond-cache", "off")],
    )
    .await;
    assert_eq!(resp.status().as_u16(), 200);
    assert!(
        mock.hits() > hits_after_first,
        "`x-beyond-cache: off` must not lookup or store (hits {} → {})",
        hits_after_first,
        mock.hits()
    );
    // A later request without the header still cannot hit: the off requests never filled.
    let before = mock.hits();
    let _ = post(&client, &gw.url(), &key, body(), &[]).await;
    assert!(
        mock.hits() > before,
        "off requests must not have filled the cache"
    );
}

#[tokio::test]
async fn a_429_then_200_is_cacheable_under_the_client_body_hash() {
    // The key is the client body, not the pool key that served. A first attempt that 429s and
    // walks to a second key still fills from the eventual 2xx; the next identical request hits.
    const WALK_A: &str = "sk-cache-walk-a";
    const WALK_B: &str = "sk-cache-walk-b";
    let nats = unused_nats_port();
    let (pubkey, sk) = test_keypair(15);
    let mock = MockUpstream::start(Mode::ThrottleKey(WALK_A)).await;
    let gw = Gateway::builder(nats, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &[WALK_A, WALK_B])
        .cache_ttl_secs(60)
        .start()
        .await;
    let key = vkey(&sk, 15);
    let client = test_client();
    {
        let (c, u, k) = (client.clone(), gw.url(), key.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move { post(&c, &u, &k, body(), &[]).await.status().as_u16() }
        })
        .await;
    }
    let _ = gw.wait_for_log_line(&["ai.usage"]).await;
    let hits_after_fill = mock.hits();
    assert!(
        hits_after_fill >= 2,
        "the fill must have walked 429 then 200 (got {} hits)",
        hits_after_fill
    );

    let resp = post(&client, &gw.url(), &key, body(), &[]).await;
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        mock.hits(),
        hits_after_fill,
        "the replayed 2xx must be served from cache, not by walking keys again"
    );
    wait_for_metric(&gw, "ai_cache_hits_total", "", 1.0).await;
}
