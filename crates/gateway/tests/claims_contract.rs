//! Claim coverage: the client-facing contract (`CAT-10`, `CAT-11`, `CAT-12`, `CAT-14`, `CAT-15` in
//! `verify/claims.toml`).
//!
//! Model-name resolution, the rejection table and its precedence, every documented header, the path
//! and method table, and a mechanical lint of what the README, ARCHITECTURE.md and the example
//! config say.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use beyond_ai::key::{VirtualKey, mint};
use common::*;
use serde_json::Value;

fn vkey(sk: &ed25519_dalek::SigningKey, tenant_id: u64) -> String {
    mint(
        &VirtualKey {
            tenant_id,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

fn chat(model: &str) -> String {
    format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#)
}

const MESSAGES: &str =
    r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;

/// The JSON error envelope every gateway-made rejection carries: `{"error":{"type","message"}}`,
/// which both stock SDKs read their exception message from.
fn envelope(body: &str) -> (String, String) {
    let v: Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body:?}"));
    let typ = v["error"]["type"].as_str().unwrap_or_default().to_owned();
    let msg = v["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert!(
        !typ.is_empty() && !msg.is_empty(),
        "incomplete envelope: {body}"
    );
    (typ, msg)
}

// --- CAT-10 --------------------------------------------------------------------------------------

/// The catalog resolves a name exactly, then case-insensitively, then by any candidate's own
/// upstream spelling (OpenRouter's slug, Bedrock's inference profile, a dated snapshot), and the
/// serving candidate always receives its own spelling. Nothing fuzzier: a name padded with
/// whitespace inside the JSON body is a catalog miss that names what was sent.
/// claim: CAT-10
#[tokio::test]
async fn aliases_case_and_snapshot_ids_resolve_and_padding_does_not() {
    let (pubkey, sk) = test_keypair(1);
    let openai = MockUpstream::start(Mode::Json).await;
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter", "bedrock"])
        .provider_authority("anthropic", &anthropic.authority())
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .provider_authority("bedrock", &GatewayBuilder::dead_authority())
        .start()
        .await;
    let key = vkey(&sk, 10);
    let post = |path: &'static str, body: String, header: Option<&'static str>| {
        let (url, key) = (gw.url(), key.clone());
        async move {
            let mut req = test_client()
                .post(format!("{url}{path}"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(body);
            if let Some(h) = header {
                req = req.header("x-beyond-model", h);
            }
            let resp = req.send().await.unwrap();
            let status = resp.status().as_u16();
            let upstream_model = resp
                .headers()
                .get("x-beyond-upstream-model")
                .map(|v| v.to_str().unwrap().to_owned());
            (status, upstream_model, resp.text().await.unwrap())
        }
    };

    // (sent in the body, upstream that serves, spelling it must receive)
    for (sent, to_anthropic, want) in [
        ("gpt-4o-mini", false, "gpt-4o-mini"),
        ("GPT-4o-Mini", false, "gpt-4o-mini"),
        ("openai/gpt-4o-mini", false, "gpt-4o-mini"),
        ("OpenAI/GPT-4o-mini", false, "gpt-4o-mini"),
        ("claude-haiku-4-5", true, "claude-haiku-4-5"),
        ("anthropic/claude-haiku-4.5", true, "claude-haiku-4-5"),
        (
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            true,
            "claude-haiku-4-5",
        ),
    ] {
        let path = if to_anthropic {
            "/v1/messages"
        } else {
            "/v1/chat/completions"
        };
        let body = if to_anthropic {
            format!(
                r#"{{"model":"{sent}","max_tokens":16,"messages":[{{"role":"user","content":"hi"}}]}}"#
            )
        } else {
            chat(sent)
        };
        let (status, upstream, text) = post(path, body, None).await;
        assert_eq!(status, 200, "{sent}: {text}");
        assert_eq!(upstream.as_deref(), Some(want), "{sent}");
        let mock = if to_anthropic { &anthropic } else { &openai };
        let got: Value = serde_json::from_slice(&mock.captured().unwrap().body).unwrap();
        assert_eq!(
            got["model"], want,
            "{sent}: the candidate gets its own spelling"
        );
    }

    // The routing header resolves the same way, and HTTP strips its surrounding whitespace.
    let (status, upstream, _) = post(
        "/auto/chat/completions",
        chat("whatever"),
        Some("  OPENAI/gpt-4o-mini  "),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(upstream.as_deref(), Some("gpt-4o-mini"));

    // No trimming or fuzzy matching inside the body: a miss, named.
    let hits = openai.hits();
    for padded in [
        " gpt-4o-mini",
        "gpt-4o-mini ",
        "gpt-4o-mini\t",
        "gpt 4o mini",
    ] {
        let (status, _, text) = post("/v1/chat/completions", chat(padded), None).await;
        assert_eq!(status, 404, "{padded:?}: {text}");
        let (typ, msg) = envelope(&text);
        assert_eq!(typ, "invalid_request_error");
        assert!(msg.contains("is not in the catalog"), "{msg}");
    }
    assert_eq!(openai.hits(), hits, "a miss never reaches a provider");
}

// --- CAT-11 --------------------------------------------------------------------------------------

/// Every rejection the gateway writes itself: its status, the error `type` both SDKs key on, a
/// non-empty message, `application/json`, and an `x-beyond-request-id` — and, where two reasons
/// apply at once, the one the documented request flow checks first.
/// claim: CAT-11
#[tokio::test]
async fn rejections_carry_the_contract_status_envelope_and_precedence() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter"])
        .start()
        .await;
    let ok = vkey(&sk, 1101);
    let denied = vkey(&sk, 1102);
    let exhausted = vkey(&sk, 1103);
    put_kv(nats.port, "blackhole.1102", b"spend").await;
    put_kv(nats.port, "allowance.1103", b"exhausted").await;
    wait_for_metric(&gw, "ai_deny_set_size", "", 1.0).await;
    wait_for_metric(&gw, "ai_allowance_set_size", "", 1.0).await;

    struct Case {
        what: &'static str,
        method: reqwest::Method,
        path: &'static str,
        auth: Option<String>,
        headers: Vec<(&'static str, String)>,
        body: String,
        status: u16,
        typ: &'static str,
    }
    let bearer = |k: &str| Some(format!("Bearer {k}"));
    let forged = "bai_v1.1.AAAAAAAAAAAAAAAAAAAAAA.AAAA".to_owned();
    let huge = (100 * 1024 * 1024 + 1).to_string();
    let cases = vec![
        Case {
            what: "unknown provider beats a missing key",
            method: reqwest::Method::POST,
            path: "/nope/v1/chat/completions",
            auth: None,
            headers: vec![],
            body: chat("gpt-4o-mini"),
            status: 404,
            typ: "invalid_request_error",
        },
        Case {
            what: "unknown x-beyond-model on /auto beats a missing key",
            method: reqwest::Method::POST,
            path: "/auto/chat/completions",
            auth: None,
            headers: vec![("x-beyond-model", "no-such-model".into())],
            body: chat("gpt-4o-mini"),
            status: 404,
            typ: "invalid_request_error",
        },
        Case {
            what: "a missing key beats an oversized body",
            method: reqwest::Method::POST,
            path: "/openai/v1/chat/completions",
            auth: None,
            headers: vec![("content-length", huge.clone())],
            body: String::new(),
            status: 401,
            typ: "authentication_error",
        },
        Case {
            what: "an oversized declared body beats a forged key",
            method: reqwest::Method::POST,
            path: "/openai/v1/chat/completions",
            auth: bearer(&forged),
            headers: vec![("content-length", huge)],
            body: String::new(),
            status: 413,
            typ: "invalid_request_error",
        },
        Case {
            what: "a forged key",
            method: reqwest::Method::POST,
            path: "/v1/chat/completions",
            auth: bearer(&forged),
            headers: vec![],
            body: chat("no-such-model"),
            status: 401,
            typ: "authentication_error",
        },
        Case {
            what: "a denied tenant beats a catalog miss",
            method: reqwest::Method::POST,
            path: "/v1/chat/completions",
            auth: bearer(&denied),
            headers: vec![],
            body: chat("no-such-model"),
            status: 402,
            typ: "access_denied",
        },
        Case {
            what: "an exhausted allowance beats a missing pool key",
            method: reqwest::Method::POST,
            path: "/groq/openai/v1/chat/completions",
            auth: bearer(&exhausted),
            headers: vec![],
            body: chat("llama"),
            status: 402,
            typ: "insufficient_quota",
        },
        Case {
            what: "a provider without a pool key",
            method: reqwest::Method::POST,
            path: "/groq/openai/v1/chat/completions",
            auth: bearer(&ok),
            headers: vec![],
            body: chat("llama"),
            status: 503,
            typ: "api_error",
        },
        Case {
            what: "BYO on the managed-only route",
            method: reqwest::Method::POST,
            path: "/auto/chat/completions",
            auth: Some("Bearer sk-byo".into()),
            headers: vec![("x-beyond-model", "gpt-4o-mini".into())],
            body: chat("gpt-4o-mini"),
            status: 400,
            typ: "invalid_request_error",
        },
        Case {
            what: "a managed GET on a generation path",
            method: reqwest::Method::GET,
            path: "/v1/chat/completions",
            auth: bearer(&ok),
            headers: vec![],
            body: String::new(),
            status: 405,
            typ: "invalid_request_error",
        },
        Case {
            what: "a managed call outside the generation allowlist",
            method: reqwest::Method::POST,
            path: "/openai/v1/files",
            auth: bearer(&ok),
            headers: vec![],
            body: "{}".into(),
            status: 404,
            typ: "invalid_request_error",
        },
        Case {
            what: "a catalog miss",
            method: reqwest::Method::POST,
            path: "/v1/chat/completions",
            auth: bearer(&ok),
            headers: vec![],
            body: chat("no-such-model"),
            status: 404,
            typ: "invalid_request_error",
        },
        Case {
            what: "a missing model",
            method: reqwest::Method::POST,
            path: "/v1/chat/completions",
            auth: bearer(&ok),
            headers: vec![],
            body: r#"{"messages":[]}"#.into(),
            status: 404,
            typ: "invalid_request_error",
        },
        Case {
            what: "a chat row on the embeddings endpoint",
            method: reqwest::Method::POST,
            path: "/v1/embeddings",
            auth: bearer(&ok),
            headers: vec![],
            body: r#"{"model":"gpt-4o-mini","input":"hi"}"#.into(),
            status: 400,
            typ: "invalid_request_error",
        },
    ];
    for c in cases {
        let mut req = test_client()
            .request(c.method.clone(), format!("{}{}", gw.url(), c.path))
            .header("content-type", "application/json");
        if let Some(a) = &c.auth {
            req = req.header("authorization", a);
        }
        for (k, v) in &c.headers {
            req = req.header(*k, v);
        }
        if !c.body.is_empty() {
            req = req.body(c.body.clone());
        }
        let resp = req.send().await.unwrap();
        assert_eq!(resp.status().as_u16(), c.status, "{}", c.what);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json",
            "{}",
            c.what
        );
        assert!(
            resp.headers().get("x-beyond-request-id").is_some(),
            "{}",
            c.what
        );
        let (typ, _) = envelope(&resp.text().await.unwrap());
        assert_eq!(typ, c.typ, "{}", c.what);
    }

    // Rate limiting is checked before the key is verified: a forged-key flood gets 429, not 401.
    let limited = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .rate_limit_rps(2)
        .start()
        .await;
    let mut seen = BTreeSet::new();
    for _ in 0..8 {
        let resp = test_client()
            .post(format!("{}/openai/v1/chat/completions", limited.url()))
            .header("authorization", format!("Bearer {forged}x"))
            .header("content-type", "application/json")
            .body(chat("gpt-4o-mini"))
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        if status == 429 {
            assert_eq!(envelope(&resp.text().await.unwrap()).0, "rate_limit_error");
        }
        seen.insert(status);
    }
    assert!(
        seen.contains(&429),
        "a forged-key flood is rate limited: {seen:?}"
    );
    assert_eq!(mock.hits(), 0, "no rejection reached a provider");
}

// --- CAT-12 --------------------------------------------------------------------------------------

/// Every documented header does what the docs say, and a malformed value is dropped and counted —
/// never a 4xx. Request side: `x-beyond-model`, `-metadata`, `-capture`, `-cache`, `-order`, `-only`,
/// `-split`, `Cache-Control: no-store`. Response side: `x-beyond-request-id` on every response,
/// `x-beyond-provider` on upstream answers and replays, `x-beyond-upstream-model` on catalog walks
/// only, `x-beyond-cache-status: hit` on replays only.
/// claim: CAT-12
#[tokio::test]
async fn every_documented_header_behaves_and_malformed_values_are_dropped() {
    let (pubkey, sk) = test_keypair(1);
    let openai = MockUpstream::start(Mode::Json).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .cache_ttl_secs(60)
        .config_line("smart_router = false")
        .start()
        .await;
    let key = vkey(&sk, 12);
    let send = |path: &'static str, body: String, headers: Vec<(&'static str, String)>| {
        let (url, key) = (gw.url(), key.clone());
        async move {
            let mut req = test_client()
                .post(format!("{url}{path}"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(body);
            for (k, v) in headers {
                req = req.header(k, v);
            }
            let resp = req.send().await.unwrap();
            let h = resp.headers().clone();
            (resp.status().as_u16(), h)
        }
    };
    let errors = |m: &str| parse_metric(m, "ai_control_header_errors_total", "");

    // x-beyond-model wins over the body; unknown is a 404 naming it.
    let (status, h) = send(
        "/auto/chat/completions",
        chat("gpt-4o"),
        vec![("x-beyond-model", "gpt-4o-mini".into())],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(h.get("x-beyond-upstream-model").unwrap(), "gpt-4o-mini");
    assert_eq!(h.get("x-beyond-provider").unwrap(), "openai");
    assert!(h.get("x-beyond-request-id").is_some());
    let (status, h) = send(
        "/auto/chat/completions",
        chat("gpt-4o-mini"),
        vec![("x-beyond-model", "nope".into())],
    )
    .await;
    assert_eq!(status, 404);
    assert!(
        h.get("x-beyond-request-id").is_some(),
        "a rejection has an id too"
    );

    // A headerless catalog walk (the body is in hand) fills the cache; the same request again is
    // a replay: provider kept, cache-status hit.
    let (status, h) = send("/v1/chat/completions", chat("gpt-4o-mini"), vec![]).await;
    assert_eq!(status, 200);
    assert!(
        h.get("x-beyond-cache-status").is_none(),
        "a fill is not a replay"
    );
    let before = openai.hits();
    let (status, h) = send("/v1/chat/completions", chat("gpt-4o-mini"), vec![]).await;
    assert_eq!(status, 200);
    assert_eq!(openai.hits(), before, "served from the cache");
    assert_eq!(h.get("x-beyond-cache-status").unwrap(), "hit");
    assert_eq!(h.get("x-beyond-provider").unwrap(), "openai");

    // `x-beyond-cache: off` and `Cache-Control: no-store` skip the lookup.
    for skip in [
        ("x-beyond-cache", "off".to_owned()),
        ("cache-control", "no-store".to_owned()),
    ] {
        let before = openai.hits();
        let (status, h) = send(
            "/v1/chat/completions",
            chat("gpt-4o-mini"),
            vec![skip.clone()],
        )
        .await;
        assert_eq!(status, 200, "{skip:?}");
        assert_eq!(openai.hits(), before + 1, "{skip:?} went upstream");
        assert!(h.get("x-beyond-cache-status").is_none(), "{skip:?}");
    }

    // A provider-routed answer names its provider but has no upstream-model header.
    let (status, h) = send("/openai/v1/chat/completions", chat("gpt-4o-mini"), vec![]).await;
    assert_eq!(status, 200);
    assert_eq!(h.get("x-beyond-provider").unwrap(), "openai");
    assert!(h.get("x-beyond-upstream-model").is_none());

    // Walk headers: order and only move the walk; split picks a weighted primary.
    for (hdr, value) in [
        ("x-beyond-order", "openrouter"),
        ("x-beyond-only", "openrouter"),
        ("x-beyond-split", "openrouter=1,openai=0"),
    ] {
        let before = openrouter.hits();
        let (status, h) = send(
            "/v1/chat/completions",
            chat("gpt-4o-mini"),
            vec![(hdr, value.into()), ("x-beyond-cache", "off".into())],
        )
        .await;
        assert_eq!(status, 200, "{hdr}");
        assert_eq!(openrouter.hits(), before + 1, "{hdr}: {value}");
        assert_eq!(h.get("x-beyond-provider").unwrap(), "openrouter", "{hdr}");
        assert_eq!(
            h.get("x-beyond-upstream-model").unwrap(),
            "openai/gpt-4o-mini",
            "{hdr}"
        );
    }

    // Malformed values: served normally, counted, as if absent (catalog order: openai first).
    let mut count = errors(&gw.metrics().await);
    let too_long = format!(r#"{{"k":"{}"}}"#, "v".repeat(2000));
    let seventeen = format!(
        "{{{}}}",
        (0..17)
            .map(|i| format!(r#""k{i}":"v""#))
            .collect::<Vec<_>>()
            .join(",")
    );
    for (hdr, value) in [
        ("x-beyond-metadata", "not json".to_owned()),
        ("x-beyond-metadata", r#"{"nested":{"a":1}}"#.to_owned()),
        ("x-beyond-metadata", "[1,2]".to_owned()),
        ("x-beyond-metadata", too_long),
        ("x-beyond-metadata", seventeen),
        ("x-beyond-capture", "yes".to_owned()),
        ("x-beyond-cache", "1".to_owned()),
        ("x-beyond-order", " , ,".to_owned()),
        ("x-beyond-only", "".to_owned()),
        ("x-beyond-split", "openrouter=heavy".to_owned()),
        ("x-beyond-split", "openrouter=0".to_owned()),
    ] {
        let before = openai.hits();
        let (status, _) = send(
            "/v1/chat/completions",
            chat("gpt-4o-mini"),
            vec![(hdr, value.clone()), ("x-beyond-cache", "off".into())],
        )
        .await;
        // `x-beyond-cache: 1` replaces the `off` above, so it may be a replay; both are served.
        assert_eq!(status, 200, "{hdr}: {value:?} must not fail the request");
        if hdr != "x-beyond-cache" {
            assert_eq!(
                openai.hits(),
                before + 1,
                "{hdr}: {value:?} kept catalog order"
            );
        }
        let now = errors(&gw.metrics().await);
        assert!(now > count, "{hdr}: {value:?} was not counted");
        count = now;
    }
    // An `x-beyond-model` that is not UTF-8 is treated as absent: the body's model routes.
    let (status, h) = send(
        "/v1/chat/completions",
        chat("gpt-4o-mini"),
        vec![("x-beyond-cache", "off".into())],
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(h.get("x-beyond-upstream-model").unwrap(), "gpt-4o-mini");
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer {key}\r\n\
         content-type: application/json\r\nx-beyond-cache: off\r\nx-beyond-model: XX\r\ncontent-length: {}\r\n\
         connection: close\r\n\r\n{}",
        chat("gpt-4o-mini").len(),
        chat("gpt-4o-mini")
    );
    // `format!` cannot carry raw bytes; patch the header value in after.
    let mut bytes = raw.into_bytes();
    let marker = "x-beyond-model: ".as_bytes();
    let at = bytes
        .windows(marker.len())
        .position(|w| w == marker)
        .unwrap()
        + marker.len();
    let end = at + bytes[at..].windows(2).position(|w| w == b"\r\n").unwrap();
    bytes.splice(at..end, [0xffu8, 0xfe]);
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", gw.port))
        .await
        .unwrap();
    s.write_all(&bytes).await.unwrap();
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(3), s.read_to_end(&mut buf)).await;
    let resp = parse_raw_response(&buf);
    assert_eq!(resp.status, 200, "{}", String::from_utf8_lossy(&buf));
    assert_eq!(resp.header("x-beyond-upstream-model"), Some("gpt-4o-mini"));
}

// --- CAT-14 --------------------------------------------------------------------------------------

/// The path and method table, pinned: every `/auto` spelling the docs name, the bare `/v1` default
/// with its trailing-slash and case rules, the managed method rules, `GET`/`HEAD /v1/models`, and
/// CORS as a non-goal (no preflight answer, no `Access-Control-*` header from the gateway).
/// claim: CAT-14
#[tokio::test]
async fn the_path_and_method_table_is_pinned() {
    let (pubkey, sk) = test_keypair(1);
    let openai = MockUpstream::start(Mode::Json).await;
    let anthropic = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter"])
        .provider_authority("anthropic", &anthropic.authority())
        .start()
        .await;
    let managed = format!("Bearer {}", vkey(&sk, 14));
    let byo = "Bearer sk-byo".to_owned();
    let none = String::new();

    // (method, path, auth, model header, body, status, upstream path the provider must see)
    type Row<'a> = (
        &'a str,
        &'a str,
        &'a String,
        Option<&'a str>,
        String,
        u16,
        Option<&'a str>,
    );
    let gpt = chat("gpt-4o-mini");
    let rows: Vec<Row<'_>> = vec![
        // /auto spellings: with and without /v1, trailing slash ignored.
        (
            "POST",
            "/auto/chat/completions",
            &managed,
            Some("gpt-4o-mini"),
            gpt.clone(),
            200,
            Some("/v1/chat/completions"),
        ),
        (
            "POST",
            "/auto/v1/chat/completions",
            &managed,
            Some("gpt-4o-mini"),
            gpt.clone(),
            200,
            Some("/v1/chat/completions"),
        ),
        (
            "POST",
            "/auto/v1/chat/completions/",
            &managed,
            Some("gpt-4o-mini"),
            gpt.clone(),
            200,
            Some("/v1/chat/completions"),
        ),
        (
            "POST",
            "/auto/messages",
            &managed,
            Some("claude-opus-4-8"),
            MESSAGES.into(),
            200,
            Some("/v1/messages"),
        ),
        (
            "POST",
            "/auto/v1/messages",
            &managed,
            None,
            MESSAGES.into(),
            200,
            Some("/v1/messages"),
        ),
        // Bare /v1: managed walks the catalog; BYO is the dialect default, path as sent.
        (
            "POST",
            "/v1/chat/completions",
            &managed,
            None,
            gpt.clone(),
            200,
            Some("/v1/chat/completions"),
        ),
        (
            "POST",
            "/v1/messages",
            &managed,
            None,
            MESSAGES.into(),
            200,
            Some("/v1/messages"),
        ),
        (
            "POST",
            "/v1/chat/completions",
            &byo,
            None,
            gpt.clone(),
            200,
            Some("/v1/chat/completions"),
        ),
        (
            "GET",
            "/v1/files",
            &byo,
            None,
            String::new(),
            200,
            Some("/v1/files"),
        ),
        // The provider segment and /auto are case-sensitive; near-miss prefixes are not /v1.
        (
            "POST",
            "/OpenAI/v1/chat/completions",
            &managed,
            None,
            gpt.clone(),
            404,
            None,
        ),
        (
            "POST",
            "/AUTO/chat/completions",
            &managed,
            Some("gpt-4o-mini"),
            gpt.clone(),
            404,
            None,
        ),
        (
            "POST",
            "/V1/chat/completions",
            &managed,
            None,
            gpt.clone(),
            404,
            None,
        ),
        (
            "POST",
            "/v1beta/models/x:generateContent",
            &byo,
            None,
            "{}".into(),
            404,
            None,
        ),
        ("POST", "/", &managed, None, gpt.clone(), 404, None),
        // Managed: POST only; GET/HEAD /v1/models answered from the catalog for anyone with a key.
        (
            "PUT",
            "/v1/chat/completions",
            &managed,
            None,
            gpt.clone(),
            405,
            None,
        ),
        (
            "DELETE",
            "/openai/v1/chat/completions",
            &managed,
            None,
            String::new(),
            405,
            None,
        ),
        (
            "GET",
            "/v1/models",
            &managed,
            None,
            String::new(),
            200,
            None,
        ),
        (
            "HEAD",
            "/v1/models",
            &managed,
            None,
            String::new(),
            200,
            None,
        ),
        ("GET", "/v1/models/", &byo, None, String::new(), 200, None),
        ("GET", "/v1/models", &none, None, String::new(), 401, None),
        // CORS is a non-goal: a preflight is just another request.
        (
            "OPTIONS",
            "/v1/chat/completions",
            &none,
            None,
            String::new(),
            401,
            None,
        ),
        (
            "OPTIONS",
            "/v1/chat/completions",
            &managed,
            None,
            String::new(),
            405,
            None,
        ),
    ];
    for (method, path, auth, model, body, status, upstream) in rows {
        let before = (openai.hits(), anthropic.hits());
        let mut req = test_client()
            .request(
                reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
                format!("{}{path}", gw.url()),
            )
            .header("content-type", "application/json")
            .header("origin", "https://app.example")
            .header("access-control-request-method", "POST");
        if !auth.is_empty() {
            req = req.header("authorization", auth.as_str());
        }
        if let Some(m) = model {
            req = req.header("x-beyond-model", m);
        }
        if !body.is_empty() {
            req = req.body(body);
        }
        let resp = req.send().await.unwrap();
        let who = if auth.contains("bai_") {
            "managed"
        } else if auth.is_empty() {
            "no key"
        } else {
            "BYO"
        };
        assert_eq!(resp.status().as_u16(), status, "{method} {path} ({who})");
        for (k, _) in resp.headers() {
            assert!(
                !k.as_str().starts_with("access-control-"),
                "{method} {path}: the gateway answered CORS ({k})"
            );
        }
        let after = (openai.hits(), anthropic.hits());
        match upstream {
            Some(want) => {
                let cap = if after.1 > before.1 {
                    anthropic.captured().unwrap()
                } else {
                    assert!(after.0 > before.0, "{method} {path}: no upstream call");
                    openai.captured().unwrap()
                };
                assert_eq!(cap.path, want, "{method} {path}");
            }
            None => assert_eq!(after, before, "{method} {path} ({who}) reached a provider"),
        }
    }
}

// --- CAT-15 --------------------------------------------------------------------------------------

fn repo_file(rel: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// Every top-level field `AiConfig` deserializes, with its default value.
fn config_defaults() -> BTreeMap<String, Value> {
    let v = serde_json::to_value(beyond_ai::config::AiConfig::default()).unwrap();
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// The rows of the first Markdown table after `heading`: each row's cells, trimmed.
fn table_after(doc: &str, heading: &str) -> Vec<Vec<String>> {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("no {heading:?} in doc"));
    doc[start..]
        .lines()
        .skip_while(|l| !l.starts_with('|'))
        .take_while(|l| l.starts_with('|'))
        .skip(2)
        .map(|l| {
            l.trim_matches('|')
                .split(" | ")
                .map(|c| c.trim().to_owned())
                .collect()
        })
        .collect()
}

/// The first backticked span in a cell, if any.
fn ticked(cell: &str) -> Option<&str> {
    let a = cell.find('`')?;
    let b = cell[a + 1..].find('`')?;
    Some(&cell[a + 1..a + 1 + b])
}

/// A documented literal (`100`, `"openai"`, `nats://…`, `true`) as the JSON value config holds.
fn literal(s: &str) -> Value {
    let s = s.trim();
    if let Ok(n) = s.parse::<u64>() {
        return Value::from(n);
    }
    match s {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        _ => {}
    }
    Value::String(s.trim_matches('"').to_owned())
}

/// Mechanically checkable statements in the README and the example config: the provider count
/// and list, the config keys the README names, its relative links, and every key the example config
/// sets (or shows commented out) being a real field at the value it claims is the default.
/// claim: CAT-15
#[test]
fn readme_and_example_config_statements_hold() {
    let readme = repo_file("README.md");
    let providers: BTreeSet<&str> = providers::gateway_providers().map(|p| p.name).collect();

    // "**12 providers, zero config**: openai, anthropic, …"
    let line = readme
        .lines()
        .find(|l| l.contains("providers, zero config"))
        .expect("the README's provider line");
    let n: usize = line
        .split("**")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("a provider count");
    assert_eq!(n, providers.len(), "README provider count: {line}");
    let listed = line.split_once(':').unwrap().1;
    let listed = listed.split(". ").next().unwrap();
    let names: BTreeSet<&str> = listed
        .split([',', ' '])
        .map(|s| s.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-'))
        .filter(|s| providers.contains(s))
        .collect();
    assert_eq!(names, providers, "README provider list: {line}");

    // Config keys the README names exist.
    let fields = config_defaults();
    for key in [
        "signing_keys",
        "snapshot_path",
        "rate_limit_rps",
        "provider_authorities",
        "provider_dialects",
        "provider_auth_schemes",
    ] {
        assert!(readme.contains(key), "README no longer names {key}");
        assert!(
            fields.contains_key(key),
            "README names {key}, not a config field"
        );
    }
    for row in table_after(&readme, "Optional:") {
        let key = ticked(&row[0]).unwrap().trim_matches(['[', ']']);
        assert!(fields.contains_key(key), "README table names {key}");
        if let Some(d) = ticked(&row[1]) {
            assert_eq!(fields[key], literal(d), "README default for {key}");
        }
    }

    // Relative links resolve.
    for (i, _) in readme.match_indices("](") {
        let target = &readme[i + 2..];
        let target = &target[..target.find(')').unwrap()];
        if target.starts_with("http") || target.starts_with('#') {
            continue;
        }
        let path = target.split('#').next().unwrap();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        assert!(root.join(path).exists(), "README links to missing {path}");
    }

    // Every key in config.example.toml, set or commented out, is a field; the set ones and the
    // commented-out scalars are the defaults the file says they are.
    let example = repo_file("crates/gateway/config.example.toml");
    let mut in_table = false;
    for raw in example.lines() {
        let line = raw.trim();
        let (commented, body) = match line.strip_prefix('#') {
            Some(rest) => (true, rest.trim()),
            None => (false, line),
        };
        if body.starts_with('[') && body.ends_with(']') {
            in_table = true;
            let table = body.trim_matches(['[', ']']);
            assert!(fields.contains_key(table), "config.example table [{table}]");
            continue;
        }
        if in_table || body.is_empty() {
            continue;
        }
        let Some((k, v)) = body.split_once(" = ") else {
            continue;
        };
        if !k.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            continue;
        }
        if commented && v.starts_with('"') && v.contains('<') {
            // A placeholder value (`"<base64 .creds>"`), not a default.
            assert!(fields.contains_key(k), "config.example key {k}");
            continue;
        }
        assert!(
            fields.contains_key(k),
            "config.example key {k} is not a config field"
        );
        let value = v.split(" #").next().unwrap().trim();
        if fields[k].is_null() {
            continue; // optional, no default (snapshot_path, nats_creds_file)
        }
        assert_eq!(
            fields[k],
            literal(value),
            "config.example says {k} = {value}"
        );
    }
}

/// ARCHITECTURE.md's tables: every module in the module map is a source file, every metric in the
/// metrics table is registered, and every field in the configuration table is a real config field
/// at the documented default. The example config's list of zero-config providers is the gateway's.
/// claim: CAT-15
/// defect: D55
#[test]
fn architecture_tables_name_real_modules_metrics_and_fields() {
    let arch = repo_file("crates/gateway/ARCHITECTURE.md");
    let mut wrong = Vec::new();

    for row in table_after(&arch, "## Modules") {
        let module = ticked(&row[0]).unwrap();
        let path = format!("crates/gateway/src/{module}.rs");
        if !std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(&path)
            .exists()
        {
            wrong.push(format!("module `{module}`: no {path}"));
        }
    }

    let metrics_src = repo_file("crates/gateway/src/metrics.rs");
    for row in table_after(&arch, "## Metrics") {
        let name = ticked(&row[0]).unwrap();
        if !metrics_src.contains(&format!("\"{name}\"")) {
            wrong.push(format!("metric `{name}`: not registered in metrics.rs"));
        }
    }

    let fields = config_defaults();
    for row in table_after(&arch, "## Configuration") {
        let name = ticked(&row[0]).unwrap();
        let (field, dotted) = match name.split_once('.') {
            Some((f, _)) => (f, true),
            None => (name, false),
        };
        let Some(default) = fields.get(field) else {
            wrong.push(format!("config table names `{name}`: no such field"));
            continue;
        };
        if dotted {
            continue;
        }
        if let Some(d) = ticked(&row[1])
            && *default != literal(d)
        {
            wrong.push(format!(
                "config table: `{name}` default `{d}`, config has {default}"
            ));
        }
    }

    let example = repo_file("crates/gateway/config.example.toml");
    let line = example
        .lines()
        .skip_while(|l| !l.contains("Known providers (zero-config"))
        .take(3)
        .map(|l| l.trim_start_matches('#'))
        .collect::<String>();
    let line = line.split_once("defaults)").map_or("", |(_, rest)| rest);
    let listed: BTreeSet<&str> = line
        .split([',', ' ', ':', '.'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let providers: BTreeSet<&str> = providers::gateway_providers().map(|p| p.name).collect();
    if listed != providers {
        wrong.push(format!(
            "config.example zero-config providers {listed:?} != gateway providers {providers:?}"
        ));
    }

    assert!(wrong.is_empty(), "false statements:\n{}", wrong.join("\n"));
}

/// The `///` lines directly above `item` (e.g. `fn clamp_output_limits(`) in `src`, joined.
fn doc_above(src: &str, item: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let at = lines
        .iter()
        .position(|l| l.trim_start().starts_with(item) || l.contains(&format!(" {item}")))
        .unwrap_or_else(|| panic!("{item} not found"));
    let mut doc: Vec<&str> = lines[..at]
        .iter()
        .rev()
        .take_while(|l| l.trim_start().starts_with("///") || l.trim_start().starts_with("#["))
        .copied()
        .collect();
    doc.reverse();
    doc.join("\n")
}

/// Doc comments sit on the item they describe, and the docs do not contradict the code: the
/// parallel merges left `proxy.rs` doc blocks fused onto the wrong functions, ARCHITECTURE.md
/// saying BYO forwards every header untouched (D67 sweeps `x-beyond-*`) and is pure passthrough,
/// and `config.rs` naming a Codex pool key that `providers` says cannot exist.
/// claim: CAT-15
/// defect: D98
#[test]
#[ignore = "D98 reproduced: misplaced doc comments and contradicting docs"]
fn doc_comments_sit_on_their_items_and_the_docs_agree() {
    let proxy = repo_file("crates/gateway/src/proxy.rs");
    let arch = repo_file("crates/gateway/ARCHITECTURE.md");
    let config = repo_file("crates/gateway/src/config.rs");
    let mut wrong = Vec::new();
    let mut check = |ok: bool, what: &str| {
        if !ok {
            wrong.push(what.to_owned());
        }
    };
    let clamp = doc_above(&proxy, "fn clamp_output_limits(");
    check(
        !clamp.contains("stream_options") && !clamp.contains("Overwrite the `model`"),
        "clamp_output_limits carries the injection / model-rewrite docs",
    );
    check(
        doc_above(&proxy, "fn apply_stream_usage_injection(").contains("Splice `stream_options"),
        "apply_stream_usage_injection has no doc",
    );
    check(
        doc_above(&proxy, "fn apply_model_rewrite(").contains("Overwrite the `model`"),
        "apply_model_rewrite has no doc",
    );
    check(
        !doc_above(&proxy, "fn is_managed_provider_endpoint(")
            .contains("Whether the **forwarded**"),
        "is_managed_provider_endpoint carries is_streamable_path's doc",
    );
    check(
        doc_above(&proxy, "fn is_streamable_path(").contains("Whether the **forwarded**"),
        "is_streamable_path has no doc",
    );
    check(
        !doc_above(&proxy, "fn attempt_start(").contains("Clear the state"),
        "attempt_start carries reset_request_body_phase's doc",
    );
    check(
        doc_above(&proxy, "fn reset_request_body_phase(").contains("Clear the state"),
        "reset_request_body_phase has no doc",
    );
    check(
        !proxy.contains("Two reasons a body gets rewritten"),
        "rewrites_body says two reasons and lists three",
    );
    check(
        !arch.contains("forward every client header untouched")
            && !arch.contains("headers are forwarded untouched"),
        "ARCHITECTURE.md: BYO headers untouched (D67 sweeps x-beyond-*)",
    );
    check(
        !arch.contains("remain pure passthrough") && !arch.contains("(pure passthrough)"),
        "ARCHITECTURE.md: BYO is pure passthrough",
    );
    check(
        !config.contains("`AI_POOL_KEY_OPENAI_CODEX` reaches"),
        "config.rs names a Codex pool key; providers says there is none",
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
