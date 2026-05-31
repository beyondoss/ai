//! End-to-end: real `beyond-ai` binary + real nats-server + mock upstream.
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).
//!
//! Signing key + pool key come from the gateway's *config*; NATS carries only the deny-set.

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code — allow the
// panic-surface restriction lints denied workspace-wide in `[workspace.lints.clippy]`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn body_for(model: &str) -> String {
    format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#)
}

async fn post_status(client: &reqwest::Client, url: &str, key: &str, body: String) -> u16 {
    client
        .post(format!("{url}/v1/chat/completions"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0)
}

/// POST to an arbitrary gateway path with a Bearer key — exercises provider routing by the first
/// path segment (`/{provider}/…`) vs the bare-path default.
async fn post_path_status(
    client: &reqwest::Client,
    url: &str,
    path: &str,
    key: &str,
    body: String,
) -> u16 {
    client
        .post(format!("{url}{path}"))
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0)
}

/// POST with the virtual key in the `x-api-key` header (Anthropic-SDK style) instead of `Bearer`.
async fn post_status_xapikey(
    client: &reqwest::Client,
    url: &str,
    path: &str,
    key: &str,
    body: String,
) -> u16 {
    client
        .post(format!("{url}{path}"))
        .header("x-api-key", key)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .map(|r| r.status().as_u16())
        .unwrap_or(0)
}

/// Send a hand-written HTTP/1.1 request and return the response status. Used to declare a
/// Content-Length the body guard must reject *without* actually transferring that many bytes
/// (the guard fires on the header, before any body is read).
async fn raw_status(port: u16, request: &str) -> u16 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    s.write_all(request.as_bytes()).await.unwrap();
    s.flush().await.unwrap();
    let mut buf = vec![0u8; 256];
    let n = s.read(&mut buf).await.unwrap();
    String::from_utf8_lossy(&buf[..n])
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}

#[tokio::test]
async fn managed_swaps_key_relays_body_and_meters_usage() {
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

    {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move { post_status(&c, &u, &k, body_for("gpt-4o")).await }
        })
        .await;
    }

    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body_for("gpt-4o"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.text().await.unwrap().contains("\"chatcmpl-mock\""));

    // Managed: the mock saw the real pool key, never the virtual key.
    let cap = mock.captured().expect("mock received a request");
    assert_eq!(cap.path, "/v1/chat/completions");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));
    assert!(!cap.body.is_empty());

    wait_for_metric(&gw, "ai_tokens_total", "input", 11.0).await;

    // Bad managed key → 401.
    assert_eq!(
        post_status(
            &client,
            &gw.url(),
            "bai_v1.1.bogus.bogus",
            body_for("gpt-4o")
        )
        .await,
        401
    );
}

#[tokio::test]
async fn byo_passes_user_token_through_unchanged() {
    let nats = Nats::start().await;
    let (pubkey, _sk) = test_keypair(1); // gateway still needs a signing key in config to boot
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let client = reqwest::Client::new();
    // A raw provider token (not `bai_`) → BYO → forwarded verbatim.
    {
        let (c, u) = (client.clone(), gw.url());
        wait_for_status(200, move || {
            let (c, u) = (c.clone(), u.clone());
            async move { post_status(&c, &u, "sk-user-byo", body_for("gpt-4o")).await }
        })
        .await;
    }
    let cap = mock.captured().expect("mock received a request");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-user-byo"),
        "BYO token must pass through unchanged (no swap)"
    );
}

#[tokio::test]
async fn fireworks_path_prefix_strips_and_swaps_pool_key() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(4);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 5,
            vpc_id: 6,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    // Fireworks is selected by the `/fireworks` path segment; the client uses its native base path
    // (`/inference/v1`). The gateway strips `/fireworks` and forwards the rest VERBATIM, and a
    // managed key swaps to the Fireworks-specific pool key.
    {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                post_path_status(
                    &c,
                    &u,
                    "/fireworks/inference/v1/chat/completions",
                    &k,
                    body_for("accounts/fireworks/models/llama-v3p1-70b-instruct"),
                )
                .await
            }
        })
        .await;
    }

    let cap = mock.captured().expect("mock received a request");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-fireworks-pool"),
        "managed Fireworks request must swap to the Fireworks pool key"
    );
    // The `/fireworks` segment is stripped; the provider-native remainder is forwarded verbatim
    // (the gateway does no per-provider path rewriting).
    assert_eq!(
        cap.path, "/inference/v1/chat/completions",
        "first segment (provider) stripped; remainder forwarded verbatim"
    );
}

#[tokio::test]
async fn openai_prefix_matches_bare_default() {
    // `/openai/v1/chat/completions` (explicit prefix) must reach OpenAI identically to bare
    // `/v1/chat/completions` (dialect default): same pool-key swap, same upstream path after strip.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(8);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 1,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                post_path_status(
                    &c,
                    &u,
                    "/openai/v1/chat/completions",
                    &k,
                    body_for("gpt-4o"),
                )
                .await
            }
        })
        .await;
    }
    let cap = mock.captured().expect("mock received a request");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));
    assert_eq!(
        cap.path, "/v1/chat/completions",
        "`/openai` stripped → same upstream path as the bare `/v1` default"
    );
}

#[tokio::test]
async fn unknown_provider_segment_returns_404() {
    // An unrecognized first path segment that isn't the bare `/v1` default is a routing miss — 404
    // from the gateway (before any auth), not a confusing upstream error. Provider resolution is the
    // very first step, so this fires regardless of the key.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(9);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 1,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
    wait_for_status(404, move || {
        let (c, u, k) = (c.clone(), u.clone(), k.clone());
        async move {
            post_path_status(&c, &u, "/bogus/v1/chat/completions", &k, body_for("gpt-4o")).await
        }
    })
    .await;
}

#[tokio::test]
async fn streaming_relays_sse_and_meters_usage() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(3);
    let mock = MockUpstream::start(Mode::Sse).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 7,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    let body = r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#
        .to_string();

    {
        let (c, u, k, b) = (client.clone(), gw.url(), vkey.clone(), body.clone());
        wait_for_status(200, move || {
            let (c, u, k, b) = (c.clone(), u.clone(), k.clone(), b.clone());
            async move { post_status(&c, &u, &k, b).await }
        })
        .await;
    }

    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.text().await.unwrap().contains("[DONE]"));
    wait_for_metric(&gw, "ai_tokens_total", "input", 5.0).await;

    // The client streamed without `stream_options`, so the managed OpenAI path must have buffered the
    // body and spliced `stream_options.include_usage` in before forwarding — otherwise OpenAI emits no
    // usage chunk and the request is unbillable. The metric above can't prove this (the mock returns a
    // usage chunk unconditionally), so assert the *forwarded body* the mock actually received carries
    // the injected fragment. This is the only coverage that the splice in `request_body_filter` ran.
    let cap = mock.captured().expect("mock received a request");
    let needle = br#""stream_options":{"include_usage":true}"#;
    assert!(
        cap.body.windows(needle.len()).any(|w| w == needle),
        "managed OpenAI streaming body must have stream_options.include_usage injected; got: {}",
        String::from_utf8_lossy(&cap.body)
    );
}

#[tokio::test]
async fn blackhole_denies_then_restores() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(2);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 99,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();

    let probe = |want: u16| {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        async move {
            wait_for_status(want, move || {
                let (c, u, k) = (c.clone(), u.clone(), k.clone());
                async move { post_status(&c, &u, &k, body_for("gpt-4o")).await }
            })
            .await
        }
    };

    probe(200).await; // ready + allowed
    put_kv(nats.port, "blackhole.99", b"spend").await;
    probe(402).await; // denied once the watch delta lands
    del_kv(nats.port, "blackhole.99").await;
    probe(200).await; // restored
}

#[tokio::test]
async fn blackhole_fraud_returns_403() {
    // The spend path (402) is covered above; fraud takes the separate `DenyReason::Fraud` branch
    // and must surface as 403 (not 402, not 200) end-to-end.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(20);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let vkey = mint(
        &VirtualKey {
            tenant_id: 1234,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();

    let probe = |want: u16| {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        async move {
            wait_for_status(want, move || {
                let (c, u, k) = (c.clone(), u.clone(), k.clone());
                async move { post_status(&c, &u, &k, body_for("gpt-4o")).await }
            })
            .await
        }
    };

    probe(200).await; // ready + allowed
    put_kv(nats.port, "blackhole.1234", b"fraud").await;
    probe(403).await; // fraud → forbidden
}

#[tokio::test]
async fn oversized_content_length_is_rejected_413() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(21);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 1,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();

    // Wait until the gateway is serving.
    {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move { post_status(&c, &u, &k, body_for("gpt-4o")).await }
        })
        .await;
    }

    // Declare a body of 200 MiB + 1 (> the 100 MiB guard) but send no body — the guard rejects on
    // the Content-Length header in request_filter before any body is read.
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: x\r\n\
         Authorization: Bearer {vkey}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: 209715201\r\n\
         Connection: close\r\n\r\n"
    );
    assert_eq!(raw_status(gw.port, &req).await, 413);
}

#[tokio::test]
async fn per_credential_rate_limit_returns_429() {
    // Every other rejection code is covered e2e (401/402/403/413/503) — 429 was the gap. A
    // misconfigured ceiling (e.g. `rate_limit_rps` env typo'd to 0) would silently disable the
    // guardrail, so prove the full enforcement path: a burst on one credential trips 429, charged on
    // the raw key in `request_filter` before any verify/upstream connect. BYO (so no key material
    // needed); the global BYO tier is disabled so this isolates the per-credential ceiling.
    let nats = Nats::start().await;
    let (pubkey, _sk) = test_keypair(40);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .rate_limit_rps(5)
        .byo_rate_limit_rps(0)
        .start()
        .await;
    let client = reqwest::Client::new();

    // Wait until the gateway serves, using a *different* credential so the flood token's budget is
    // untouched by readiness probing.
    {
        let (c, u) = (client.clone(), gw.url());
        wait_for_status(200, move || {
            let (c, u) = (c.clone(), u.clone());
            async move { post_status(&c, &u, "sk-byo-warmup", body_for("gpt-4o")).await }
        })
        .await;
    }

    // Burst one credential well past its 5 rps ceiling within a single window. The first few are
    // served (200); once the ceiling is crossed the rest are throttled (429).
    let mut saw_200 = false;
    let mut saw_429 = false;
    for _ in 0..50 {
        match post_status(&client, &gw.url(), "sk-byo-flood", body_for("gpt-4o")).await {
            200 => saw_200 = true,
            429 => saw_429 = true,
            other => panic!("unexpected status under rate limit: {other}"),
        }
    }
    assert!(
        saw_200,
        "the first requests under the ceiling must be served"
    );
    assert!(
        saw_429,
        "a burst past the per-credential ceiling must yield 429"
    );
    wait_for_metric(&gw, "ai_rejections_total", "rate_limit", 1.0).await;
}

#[tokio::test]
async fn managed_key_via_x_api_key_header_is_accepted() {
    // Anthropic SDKs present the key in `x-api-key`, not `Authorization: Bearer`. A managed virtual
    // key must be extracted from either header; here it arrives via x-api-key on the OpenAI path and
    // must still swap to the OpenAI pool key in the Bearer scheme the upstream wants.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(22);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 8,
            vpc_id: 8,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                post_status_xapikey(&c, &u, "/v1/chat/completions", &k, body_for("gpt-4o")).await
            }
        })
        .await;
    }
    let cap = mock.captured().expect("mock received a request");
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-pool-secret"));
}

#[tokio::test]
async fn managed_key_for_unconfigured_provider_returns_503() {
    // The default gateway configures OpenAI + Fireworks pool keys, but NOT Anthropic. A managed key
    // routed to Anthropic (via the `/anthropic` path segment) has no pool key → 503, surfaced in
    // request_filter before any upstream connect.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(23);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 11,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
    wait_for_status(503, move || {
        let (c, u, k) = (c.clone(), u.clone(), k.clone());
        async move {
            post_path_status(
                &c,
                &u,
                "/anthropic/v1/messages",
                &k,
                body_for("claude-opus-4-8"),
            )
            .await
        }
    })
    .await;
}

#[tokio::test]
async fn anthropic_dialect_swaps_key_relays_and_meters() {
    // The Anthropic path (`/v1/messages`) drives a different dialect, a different auth scheme
    // (x-api-key, not Bearer), and a different usage parser than the OpenAI tests above.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(24);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["anthropic"])
        .start()
        .await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 77,
            vpc_id: 2,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    {
        let (c, u, k) = (client.clone(), gw.url(), vkey.clone());
        wait_for_status(200, move || {
            let (c, u, k) = (c.clone(), u.clone(), k.clone());
            async move {
                post_status_xapikey(&c, &u, "/v1/messages", &k, body_for("claude-opus-4-8")).await
            }
        })
        .await;
    }

    let resp = client
        .post(format!("{}/v1/messages", gw.url()))
        .header("x-api-key", &vkey)
        .header("content-type", "application/json")
        .body(body_for("claude-opus-4-8"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let cap = mock.captured().expect("mock received a request");
    assert_eq!(cap.path, "/v1/messages");
    // Anthropic wants the key in x-api-key, and the inbound virtual key must not leak upstream.
    assert_eq!(cap.x_api_key.as_deref(), Some("sk-anthropic-pool"));
    assert_eq!(cap.authorization, None);

    wait_for_metric(&gw, "ai_tokens_total", "input", 13.0).await;
}

#[tokio::test]
async fn missing_api_key_returns_401() {
    // A request with neither Authorization nor x-api-key takes the "missing API key" branch — a
    // different path than the malformed-key (invalid) branch the managed test exercises.
    let nats = Nats::start().await;
    let (pubkey, _sk) = test_keypair(25);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let client = reqwest::Client::new();

    let (c, u) = (client.clone(), gw.url());
    wait_for_status(401, move || {
        let (c, u) = (c.clone(), u.clone());
        async move {
            c.post(format!("{u}/v1/chat/completions"))
                .header("content-type", "application/json")
                .body(body_for("gpt-4o"))
                .send()
                .await
                .map(|r| r.status().as_u16())
                .unwrap_or(0)
        }
    })
    .await;
}

#[tokio::test]
async fn deny_set_is_fail_open_when_nats_drops() {
    // After NATS goes away the last-known deny-set must be *retained* (fail-open), and auth/keys —
    // which come from config, not NATS — must keep working.
    let mut nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(26);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;

    let denied = mint(
        &VirtualKey {
            tenant_id: 555,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let allowed = mint(
        &VirtualKey {
            tenant_id: 556,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();

    let probe = |key: String, want: u16| {
        let (c, u) = (client.clone(), gw.url());
        async move {
            wait_for_status(want, move || {
                let (c, u, k) = (c.clone(), u.clone(), key.clone());
                async move { post_status(&c, &u, &k, body_for("gpt-4o")).await }
            })
            .await
        }
    };

    probe(denied.clone(), 200).await; // ready + allowed
    put_kv(nats.port, "blackhole.555", b"spend").await;
    probe(denied.clone(), 402).await; // deny delta landed

    nats.stop(); // NATS disappears

    probe(denied.clone(), 402).await; // stale deny retained, not cleared
    probe(allowed.clone(), 200).await; // un-denied tenant still served without NATS
}

#[tokio::test]
async fn streaming_tail_compaction_preserves_usage_event() {
    // The usage chunk trails 130 KiB of content, forcing the proxy's response-tail compaction
    // (resp_tail grows past 2× the 64 KiB cap). The usage event must survive in the retained tail.
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(27);
    let mock = MockUpstream::start(Mode::SseLarge).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 21,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();
    let body = r#"{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}"#
        .to_string();

    {
        let (c, u, k, b) = (client.clone(), gw.url(), vkey.clone(), body.clone());
        wait_for_status(200, move || {
            let (c, u, k, b) = (c.clone(), u.clone(), k.clone(), b.clone());
            async move { post_status(&c, &u, &k, b).await }
        })
        .await;
    }

    let resp = client
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap(); // drain the >128 KiB stream

    wait_for_metric(&gw, "ai_tokens_total", "input", 5.0).await;
}

#[tokio::test]
async fn on_disk_snapshot_enforces_across_restart_without_nats() {
    // With a configured snapshot path, the deny-set is persisted to disk as deltas arrive. A restart
    // must seed from that file and enforce immediately — even with NATS unreachable — proving the
    // gateway reads the snapshot rather than re-scanning NATS on every boot.
    let mut nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(28);
    let mock = MockUpstream::start(Mode::Json).await;
    let snap = std::env::temp_dir().join(format!("beyond-ai-snap-{}.log", nats.port));
    let _ = std::fs::remove_file(&snap);
    let snap_str = snap.to_str().unwrap().to_string();

    let denied = mint(
        &VirtualKey {
            tenant_id: 8800,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let allowed = mint(
        &VirtualKey {
            tenant_id: 8801,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    let client = reqwest::Client::new();

    let probe = |gw_url: String, key: String, want: u16| {
        let c = client.clone();
        async move {
            wait_for_status(want, move || {
                let (c, u, k) = (c.clone(), gw_url.clone(), key.clone());
                async move { post_status(&c, &u, &k, body_for("gpt-4o")).await }
            })
            .await
        }
    };

    // --- First run: blackhole the tenant; the delta is persisted to the snapshot. ---
    {
        let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
            .snapshot_path(&snap_str)
            .start()
            .await;
        probe(gw.url(), denied.clone(), 200).await; // ready + allowed
        put_kv(nats.port, "blackhole.8800", b"fraud").await;
        probe(gw.url(), denied.clone(), 403).await; // applied in-memory AND appended to the snapshot
        // Let the watcher's apply→persist step flush the checkpoint to disk before we kill it.
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        // gw drops here → process killed.
    }

    // NATS goes away entirely: a restart has nothing to scan and must rely on the snapshot.
    nats.stop();

    // --- Restart against the same snapshot path, NATS down. ---
    let gw2 = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .snapshot_path(&snap_str)
        .start()
        .await;
    // Seeded from disk: the fraud hold is enforced even though NATS is unreachable.
    probe(gw2.url(), denied, 403).await;
    // And an un-denied tenant is still served (auth/keys are from config, not NATS).
    probe(gw2.url(), allowed, 200).await;

    let _ = std::fs::remove_file(&snap);
}

#[tokio::test]
async fn health_endpoints_report_ready_on_the_metrics_listener() {
    // /livez and /readyz live on the metrics listener (alongside /metrics) and must both 200 with a
    // `{status:"ok"}` body once the process is up. Readiness is intentionally *not* gated on NATS —
    // the gateway is fail-open, so it can serve from config alone. We stop NATS before probing to
    // prove readiness doesn't depend on it: a NATS-less gateway is still ready.
    let mut nats = Nats::start().await;
    let (pubkey, _sk) = test_keypair(30);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::start(nats.port, &mock.authority(), &b64(&pubkey)).await;
    nats.stop();

    let (live_status, live_body) = gw.admin_get("/livez").await;
    assert_eq!(
        live_status, 200,
        "livez should be 200 once the process answers"
    );
    assert!(
        live_body.contains("\"status\":\"ok\""),
        "livez body: {live_body}"
    );

    let (ready_status, ready_body) = gw.admin_get("/readyz").await;
    assert_eq!(
        ready_status, 200,
        "readyz should be 200 even with NATS down (fail-open): {ready_body}"
    );
    assert!(
        ready_body.contains("\"status\":\"ok\""),
        "readyz body: {ready_body}"
    );

    // An unknown admin path is a clean 404, not a hang or a 200.
    let (nf_status, _) = gw.admin_get("/nope").await;
    assert_eq!(nf_status, 404);
}
