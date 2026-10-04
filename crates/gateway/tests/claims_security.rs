//! Claim coverage: security and isolation (`SEC-*` in `verify/claims.toml`).
//!
//! Each test asserts one claim end to end against the real binary and a recording upstream. The
//! assertions are made where the claim lives: at the upstream (what left the gateway), at the client
//! (what came back), or in the log and metrics.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use beyond_ai::key::{VirtualKey, mint, mint_v2};
use bytes::Bytes;
use common::*;
use http_body_util::{BodyExt, Full};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;
const MESSAGES: &str =
    r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;

/// The path and every header the upstream saw, joined, for a substring search.
fn everything_forwarded(cap: &Captured) -> String {
    let mut out = cap.path.clone();
    for (k, v) in &cap.headers {
        out.push('\n');
        out.push_str(k.as_str());
        out.push_str(": ");
        out.push_str(&String::from_utf8_lossy(v.as_bytes()));
    }
    out
}

/// Every credential-bearing header the upstream received, by name.
fn credential_headers(cap: &Captured) -> Vec<(String, String)> {
    ["authorization", "x-api-key", "api-key", "x-goog-api-key"]
        .into_iter()
        .flat_map(|name| {
            cap.headers
                .get_all(name)
                .iter()
                .map(move |v| (name.to_owned(), v.to_str().unwrap_or("").to_owned()))
        })
        .collect()
}

/// Write `request` on a fresh connection and read until the peer closes or `wait` elapses.
async fn raw_exchange(port: u16, request: &[u8], wait: Duration) -> Vec<u8> {
    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    // A peer that rejects early may reset before the whole request is written; that is an answer.
    let _ = s.write_all(request).await;
    let mut buf = Vec::new();
    let deadline = tokio::time::Instant::now() + wait;
    loop {
        let mut chunk = [0u8; 8192];
        match tokio::time::timeout_at(deadline, s.read(&mut chunk)).await {
            Ok(Ok(n)) if n > 0 => buf.extend_from_slice(&chunk[..n]),
            _ => break,
        }
    }
    buf
}

/// Every HTTP status line in a raw response stream (a pipelined connection can answer twice).
fn statuses(raw: &[u8]) -> Vec<u16> {
    let text = String::from_utf8_lossy(raw);
    text.match_indices("HTTP/1.1 ")
        .filter_map(|(i, _)| text[i + 9..].get(..3).and_then(|s| s.parse().ok()))
        .collect()
}

// --- an upstream that answers with whatever credential it was sent -------------------------------

/// What [`EchoUpstream`] does with the credential it received.
#[derive(Clone, Copy)]
enum Echo {
    /// 401 with the credential in an OpenAI error message and an `x-echo-auth` header.
    OpenAi401,
    /// 401 with the credential in an Anthropic error message and an `x-echo-auth` header.
    Anthropic401,
    /// 200 with the credential in a response header only.
    HeaderOn200,
}

/// A plaintext H1 upstream that echoes the presented credential back, the way a careless provider
/// (or a debugging proxy between us and it) might. Proves what the gateway relays, not what a
/// well-behaved provider sends.
struct EchoUpstream {
    port: u16,
    task: tokio::task::JoinHandle<()>,
}

impl EchoUpstream {
    async fn start(mode: Echo) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<hyper::body::Incoming>| async move {
                        let h = req.headers();
                        let cred = ["authorization", "x-api-key", "api-key"]
                            .into_iter()
                            .find_map(|n| h.get(n).and_then(|v| v.to_str().ok()))
                            .unwrap_or("")
                            .to_owned();
                        let _ = req.into_body().collect().await;
                        let (status, body) = match mode {
                            Echo::OpenAi401 => (
                                401,
                                format!(
                                    r#"{{"error":{{"message":"Incorrect API key provided: {cred}","type":"invalid_request_error","code":"invalid_api_key"}}}}"#
                                ),
                            ),
                            Echo::Anthropic401 => (
                                401,
                                format!(
                                    r#"{{"type":"error","error":{{"type":"authentication_error","message":"invalid x-api-key {cred}"}}}}"#
                                ),
                            ),
                            Echo::HeaderOn200 => (
                                200,
                                r#"{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#.to_owned(),
                            ),
                        };
                        Ok::<_, std::io::Error>(
                            Response::builder()
                                .status(status)
                                .header("content-type", "application/json")
                                .header("x-echo-auth", cred)
                                .body(Full::new(Bytes::from(body)))
                                .unwrap(),
                        )
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), svc)
                        .await;
                });
            }
        });
        EchoUpstream { port, task }
    }

    fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }
}

impl Drop for EchoUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// --- SEC-5 ---------------------------------------------------------------------------------------

/// Wherever a managed request carries its virtual key, and whatever else it carries beside it,
/// exactly one credential leaves the gateway: the provider's pool key, in that provider's own
/// header and scheme. Covers Bearer (OpenAI), `x-api-key` (Anthropic) and a config-added provider
/// whose scheme is Azure's `api-key`.
/// claim: SEC-5
#[tokio::test]
async fn every_credential_location_is_stripped_and_one_pool_key_goes_out() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "azure"])
        .config_line(r#"provider_auth_schemes = { azure = "api-key" }"#)
        .start()
        .await;
    let vk = vkey(&sk, 5);

    // (location of the virtual key, extra credentials beside it)
    let placements: Vec<(Vec<(&str, String)>, &str)> = vec![
        (vec![("authorization", format!("Bearer {vk}"))], ""),
        (vec![("x-api-key", vk.clone())], ""),
        (vec![("api-key", vk.clone())], ""),
        (vec![("x-goog-api-key", vk.clone())], ""),
        (vec![], "key"),
        (
            vec![
                ("authorization", format!("Bearer {vk}")),
                ("x-api-key", "sk-ant-someone".into()),
                ("api-key", "azure-someone".into()),
                ("x-goog-api-key", "AIzaSomeone".into()),
            ],
            "",
        ),
    ];
    // (route, body, the header and value the provider must receive)
    let routes: [(&str, &str, &str, &str); 3] = [
        (
            "/openai/v1/chat/completions",
            CHAT,
            "authorization",
            "Bearer sk-pool-secret",
        ),
        (
            "/anthropic/v1/messages",
            MESSAGES,
            "x-api-key",
            "sk-anthropic-pool",
        ),
        (
            "/azure/openai/deployments/d/chat/completions",
            CHAT,
            "api-key",
            "sk-unknown-pool",
        ),
    ];
    for (route, body, want_name, want_value) in routes {
        for (headers, query) in &placements {
            let url = if query.is_empty() {
                format!("{}{route}?api-version=2024-10-21", gw.url())
            } else {
                format!("{}{route}?api-version=2024-10-21&key={vk}", gw.url())
            };
            let mut req = test_client()
                .post(url)
                .header("content-type", "application/json")
                .body(body);
            for (k, v) in headers {
                req = req.header(*k, v);
            }
            let resp = req.send().await.unwrap();
            assert_eq!(resp.status(), 200, "{route} {headers:?} {query}");
            let cap = mock.captured().unwrap();
            assert_eq!(
                credential_headers(&cap),
                vec![(want_name.to_owned(), want_value.to_owned())],
                "{route} {headers:?} {query}: exactly the pool key, in the provider's scheme"
            );
            let seen = everything_forwarded(&cap);
            assert!(!seen.contains("bai_v"), "{route}: forwarded\n{seen}");
            assert!(!seen.contains("someone"), "{route}: forwarded\n{seen}");
            assert!(
                cap.path.ends_with("?api-version=2024-10-21"),
                "{route}: {}",
                cap.path
            );
        }
    }
}

// --- SEC-3 / SEC-11 ------------------------------------------------------------------------------

/// The spellings a credential arrives in that the first-location classifier was not written for: a
/// repeated header or query param whose first value is junk, a percent-encoded `key`, a `Bearer`
/// with two spaces. None of them may carry a virtual key to a provider.
/// claim: SEC-3, SEC-11
/// defect: D65
#[tokio::test]
async fn repeated_or_respelled_credentials_never_forward_a_virtual_key() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let vk = vkey(&sk, 6);
    let base = format!("{}/openai/v1/chat/completions", gw.url());
    let cases: Vec<(String, Vec<(&str, String)>)> = vec![
        // Two query keys, the first junk: the classifier sees `junk` and forwards the query.
        (format!("{base}?key=junk&key={vk}"), vec![]),
        // A percent-encoded `key` name, which Google decodes as `key`.
        (
            format!("{base}?k%65y={vk}"),
            vec![("authorization", format!("Bearer {vk}"))],
        ),
        // Two `x-api-key` lines, the first junk.
        (
            base.clone(),
            vec![("x-api-key", "junk".into()), ("x-api-key", vk.clone())],
        ),
        // Two spaces after `Bearer`: a lenient provider trims and reads the virtual key.
        (
            base.clone(),
            vec![("authorization", format!("Bearer  {vk}"))],
        ),
    ];
    let mut leaks = Vec::new();
    for (url, headers) in &cases {
        let before = mock.hits();
        let mut req = test_client()
            .post(url)
            .header("content-type", "application/json")
            .body(CHAT);
        for (k, v) in headers {
            req = req.header(*k, v);
        }
        let _ = req.send().await.unwrap();
        if mock.hits() > before {
            let seen = everything_forwarded(&mock.captured().unwrap());
            if seen.contains("bai_v") {
                leaks.push(format!("{url} {headers:?} forwarded:\n{seen}"));
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "virtual key reached the provider:\n{}",
        leaks.join("\n\n")
    );
}

// --- SEC-7 ---------------------------------------------------------------------------------------

/// Every response the gateway writes itself — rejections, routing errors, a dead upstream — and
/// every relayed success carries no pool key in its body or headers.
/// claim: SEC-7
#[tokio::test]
async fn gateway_made_responses_never_carry_a_pool_key() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter"])
        .provider_authority("anthropic", &GatewayBuilder::dead_authority())
        .provider_authority("openrouter", &GatewayBuilder::dead_authority())
        .start()
        .await;
    let vk = vkey(&sk, 7);
    let bearer = format!("Bearer {vk}");
    let cases: Vec<(reqwest::Method, &str, String, String)> = vec![
        // relayed success
        (
            reqwest::Method::POST,
            "/openai/v1/chat/completions",
            bearer.clone(),
            CHAT.into(),
        ),
        // managed endpoint refusal (404) and method refusal (405)
        (
            reqwest::Method::POST,
            "/openai/v1/files",
            bearer.clone(),
            "{}".into(),
        ),
        (
            reqwest::Method::GET,
            "/v1/chat/completions",
            bearer.clone(),
            String::new(),
        ),
        // catalog miss, unknown provider
        (
            reqwest::Method::POST,
            "/v1/chat/completions",
            bearer.clone(),
            r#"{"model":"no-such-model","messages":[]}"#.into(),
        ),
        (
            reqwest::Method::POST,
            "/nope/v1/chat/completions",
            bearer.clone(),
            CHAT.into(),
        ),
        // forged key, missing key
        (
            reqwest::Method::POST,
            "/openai/v1/chat/completions",
            "Bearer bai_v1.1.AAAA.BBBB".into(),
            CHAT.into(),
        ),
        (
            reqwest::Method::POST,
            "/openai/v1/chat/completions",
            String::new(),
            CHAT.into(),
        ),
        // every candidate down (the Claude row's Anthropic and OpenRouter arms are dead)
        (
            reqwest::Method::POST,
            "/v1/messages",
            bearer.clone(),
            MESSAGES.into(),
        ),
        // a provider with no pool key
        (
            reqwest::Method::POST,
            "/groq/openai/v1/chat/completions",
            bearer,
            CHAT.into(),
        ),
    ];
    for (method, path, auth, body) in cases {
        let mut req = test_client()
            .request(method.clone(), format!("{}{path}", gw.url()))
            .header("content-type", "application/json")
            .body(body);
        if !auth.is_empty() {
            req = req.header("authorization", auth);
        }
        let resp = req.send().await.unwrap();
        let status = resp.status();
        let mut all = String::new();
        for (k, v) in resp.headers() {
            all.push_str(&format!("{k}: {}\n", String::from_utf8_lossy(v.as_bytes())));
        }
        all.push_str(&resp.text().await.unwrap_or_default());
        for pool in ["sk-pool-secret", "sk-anthropic-pool", "sk-openrouter-pool"] {
            assert!(
                !all.contains(pool),
                "{method} {path} ({status}) returned a pool key:\n{all}"
            );
        }
    }
}

/// A provider (or anything between us and it) that echoes the credential it was sent must not hand
/// Beyond's pool key to the client — not in an error body, not in a response header, not on a
/// translated walk.
/// claim: SEC-7
/// defect: D66
#[tokio::test]
async fn an_upstream_echo_of_the_pool_key_never_reaches_the_client() {
    let (pubkey, sk) = test_keypair(1);
    let vk = vkey(&sk, 8);
    let mut leaks = Vec::new();
    for (echo, provider, path, body) in [
        (
            Echo::OpenAi401,
            "openai",
            "/openai/v1/chat/completions",
            CHAT,
        ),
        (
            Echo::HeaderOn200,
            "openai",
            "/openai/v1/chat/completions",
            CHAT,
        ),
        (Echo::OpenAi401, "openai", "/v1/chat/completions", CHAT),
        (
            Echo::Anthropic401,
            "anthropic",
            "/v1/chat/completions",
            r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    ] {
        let up = EchoUpstream::start(echo).await;
        let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
            .providers(&[provider, "openrouter", "bedrock"])
            .provider_authority("openrouter", &GatewayBuilder::dead_authority())
            .provider_authority("bedrock", &GatewayBuilder::dead_authority())
            .start()
            .await;
        let resp = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("authorization", format!("Bearer {vk}"))
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .unwrap();
        let mut all = String::new();
        for (k, v) in resp.headers() {
            all.push_str(&format!("{k}: {}\n", String::from_utf8_lossy(v.as_bytes())));
        }
        all.push_str(&resp.text().await.unwrap_or_default());
        if all.contains("sk-pool-secret") || all.contains("sk-anthropic-pool") {
            leaks.push(format!("{path} on {provider}:\n{all}"));
        }
    }
    assert!(
        leaks.is_empty(),
        "pool key reached the client:\n{}",
        leaks.join("\n\n")
    );
}

/// A Bedrock API key (`ABSK…`) is base64, so it carries `/` and `+`, and a JSON encoder may write
/// those as `\/` and `+`. An echo spelled so is the same key to any JSON client, and the
/// remedy rewrite used to decode it (`\/` to `/`) when it re-serialized the body. Neither the raw
/// key nor an escaped spelling reaches the client, in a 401 whose JSON carries no remedy (relayed
/// as sent), one with a remedy (rewritten and re-serialized), and a non-JSON one (streamed).
/// claim: SEC-7
/// defect: D203
#[tokio::test]
async fn a_json_escaped_echo_of_the_pool_key_never_reaches_the_client() {
    const KEY: &str = "ABSKQmVkcm9ja0FQSUtleS1h/YmNk+ZWZn/aGlq";
    const PLAIN: &str = r#"{"message":"The security token included in the request is invalid: ABSKQmVkcm9ja0FQSUtleS1h\/YmNk+ZWZn\/aGlq"}"#;
    const REMEDY: &str = r#"{"error":{"message":"Invalid key; manage keys at https://openrouter.ai/settings/keys","code":401,"param":"ABSKQmVkcm9ja0FQSUtleS1h\/YmNk+ZWZn\/aGlq"}}"#;
    const TEXT: &str = r#"invalid credential "ABSKQmVkcm9ja0FQSUtleS1h\/YmNk+ZWZn\/aGlq""#;
    let (pubkey, sk) = test_keypair(203);
    let vk = vkey(&sk, 203);
    let escaped = KEY.replace('/', "\\/");
    let mut leaks = Vec::new();
    for (content_type, body) in [
        ("application/json", PLAIN),
        ("application/json", REMEDY),
        ("text/plain", TEXT),
    ] {
        let up = MockUpstream::start(Mode::Raw(401, content_type, body)).await;
        let gw = Gateway::builder(unused_nats_port(), &up.authority(), &b64(&pubkey))
            .providers(&["openai"])
            .pool_keys("openai", &[KEY])
            .start()
            .await;
        let resp = test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", format!("Bearer {vk}"))
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        // What a JSON client reads: every string value, decoded.
        let decoded = serde_json::from_str::<serde_json::Value>(&text)
            .map(|v| v.to_string().replace("\\/", "/"))
            .unwrap_or_default();
        let fragments = ["ABSKQmVkcm9ja0FQSUtleS1h", "YmNk", "aGlq"];
        if status != 401
            || text.contains(KEY)
            || text.contains(&escaped)
            || decoded.contains(KEY)
            || fragments.iter().all(|f| text.contains(f))
        {
            leaks.push(format!("{content_type} {status}: {text}"));
        }
    }
    assert!(
        leaks.is_empty(),
        "pool key reached the client:\n{}",
        leaks.join("\n\n")
    );
}

// --- SEC-9 ---------------------------------------------------------------------------------------

/// Every `x-beyond-*` request header is the gateway's own namespace, and none reaches a provider on
/// any route: managed or BYO, provider-prefixed, bare `/v1`, or `/auto`. That includes names the
/// gateway does not (yet) define.
/// claim: SEC-9
/// defect: D67
#[tokio::test]
async fn client_x_beyond_headers_never_reach_the_upstream_on_any_route() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter"])
        .start()
        .await;
    let managed = format!("Bearer {}", vkey(&sk, 9));
    let byo = "Bearer sk-byo-openai".to_owned();
    let control: [(&str, &str); 8] = [
        ("x-beyond-model", "gpt-4o-mini"),
        ("x-beyond-metadata", r#"{"feature":"x"}"#),
        ("x-beyond-capture", "off"),
        ("x-beyond-cache", "off"),
        ("x-beyond-order", "openai"),
        ("x-beyond-only", "openai,openrouter"),
        ("x-beyond-split", "openai=1"),
        ("x-beyond-trace", "future-header"),
    ];
    let mut leaks = Vec::new();
    for (path, auth) in [
        ("/openai/v1/chat/completions", &managed),
        ("/v1/chat/completions", &managed),
        ("/auto/v1/chat/completions", &managed),
        ("/openai/v1/chat/completions", &byo),
        ("/v1/chat/completions", &byo),
    ] {
        let mut req = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("authorization", auth.as_str())
            .header("content-type", "application/json")
            .body(CHAT);
        for (k, v) in control {
            req = req.header(k, v);
        }
        let resp = req.send().await.unwrap();
        assert_eq!(resp.status(), 200, "{path} {auth}");
        let cap = mock.captured().unwrap();
        let forwarded: Vec<&str> = cap
            .headers
            .keys()
            .map(|k| k.as_str())
            .filter(|k| k.starts_with("x-beyond-"))
            .collect();
        if !forwarded.is_empty() {
            let who = if auth.contains("bai_") {
                "managed"
            } else {
                "BYO"
            };
            leaks.push(format!("{who} {path}: {forwarded:?}"));
        }
    }
    assert!(
        leaks.is_empty(),
        "x-beyond-* reached the provider:\n{}",
        leaks.join("\n")
    );
}

/// The managed half of SEC-9, which holds today: every route a managed key can take strips the
/// whole namespace.
/// claim: SEC-9
#[tokio::test]
async fn managed_x_beyond_headers_never_reach_the_upstream() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter"])
        .start()
        .await;
    let managed = format!("Bearer {}", vkey(&sk, 10));
    for path in [
        "/openai/v1/chat/completions",
        "/v1/chat/completions",
        "/auto/v1/chat/completions",
        "/auto/chat/completions",
    ] {
        let resp = test_client()
            .post(format!("{}{path}", gw.url()))
            .header("authorization", managed.as_str())
            .header("content-type", "application/json")
            .header("x-beyond-model", "gpt-4o-mini")
            .header("x-beyond-metadata", r#"{"feature":"x"}"#)
            .header("x-beyond-capture", "nonsense")
            .header("x-beyond-cache", "off")
            .header("x-beyond-order", "openai")
            .header("x-beyond-only", "openai,openrouter")
            .header("x-beyond-split", "openai=1")
            .header("x-beyond-trace", "future-header")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{path}");
        let cap = mock.captured().unwrap();
        let forwarded: Vec<&str> = cap
            .headers
            .keys()
            .map(|k| k.as_str())
            .filter(|k| k.starts_with("x-beyond-"))
            .collect();
        assert!(forwarded.is_empty(), "{path}: {forwarded:?}");
    }
}

// --- SEC-10 --------------------------------------------------------------------------------------

/// A BYO key is the caller's own: it never draws a pool key, never walks the catalog (or another
/// vendor), is never served from or stored in the response cache, is never captured, and its
/// `x-beyond-*` headers change nothing.
/// claim: SEC-10
#[tokio::test]
async fn byo_gets_no_pool_key_walk_cache_capture_or_control() {
    let nats = Nats::start().await;
    let (pubkey, _sk) = test_keypair(1);
    let openai = MockUpstream::start(Mode::Json).await;
    let openrouter = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &openrouter.authority())
        .cache_ttl_secs(60)
        .start()
        .await;
    let send = |path: &'static str, extra: Vec<(&'static str, &'static str)>| {
        let url = format!("{}{path}", gw.url());
        async move {
            let mut req = test_client()
                .post(url)
                .header("authorization", "Bearer sk-byo-own-key")
                .header("content-type", "application/json")
                .body(CHAT);
            for (k, v) in extra {
                req = req.header(k, v);
            }
            req.send().await.unwrap()
        }
    };

    // The catalog route is refused outright.
    let resp = send(
        "/auto/v1/chat/completions",
        vec![("x-beyond-model", "gpt-4o-mini")],
    )
    .await;
    assert_eq!(resp.status(), 400);
    assert_eq!(openai.hits() + openrouter.hits(), 0);

    // Bare `/v1` with walk headers: the dialect default, the caller's own key, the body untouched.
    for _ in 0..2 {
        let resp = send(
            "/v1/chat/completions",
            vec![
                ("x-beyond-order", "openrouter"),
                ("x-beyond-only", "openrouter"),
                ("x-beyond-capture", "on"),
                ("x-beyond-cache", "on"),
            ],
        )
        .await;
        assert_eq!(resp.status(), 200);
        assert!(resp.headers().get("x-beyond-upstream-model").is_none());
        assert!(resp.headers().get("x-beyond-cache-status").is_none());
    }
    assert_eq!(
        openai.hits(),
        2,
        "both requests reached the provider: no cache replay"
    );
    assert_eq!(openrouter.hits(), 0, "no walk header moved a BYO request");
    let cap = openai.captured().unwrap();
    assert_eq!(cap.authorization.as_deref(), Some("Bearer sk-byo-own-key"));
    assert_eq!(cap.body, CHAT.as_bytes(), "a BYO body is relayed as sent");

    // A BYO 429 is the caller's own throttle: no pool-key walk.
    let throttled = MockUpstream::start(Mode::Status(429)).await;
    let gw2 = Gateway::builder(nats.port, &throttled.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .pool_keys("openai", &["sk-pool-a", "sk-pool-b"])
        .start()
        .await;
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw2.url()))
        .header("authorization", "Bearer sk-byo-own-key")
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 429);
    assert_eq!(throttled.hits(), 1);
    assert_eq!(
        throttled.captured().unwrap().authorization.as_deref(),
        Some("Bearer sk-byo-own-key")
    );

    // Nothing above was captured or billed.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let log = gw.log() + &gw2.log();
    assert!(!log.contains(r#""target":"ai.payload""#), "{log}");
    assert!(!log.contains(r#""target":"ai.usage""#), "{log}");
}

// --- SEC-12 --------------------------------------------------------------------------------------

/// The per-credential limit counts one identity, wherever the request presents it: the same key
/// sent as `Bearer`, `x-api-key` and `?key=` shares one bucket. A respelled kid (`01`) is not a
/// second spelling of the key: it does not verify.
/// claim: SEC-12
#[tokio::test]
async fn the_rate_limit_counts_one_identity_across_credential_locations() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .rate_limit_rps(2)
        .start()
        .await;
    let vk = vkey(&sk, 12);
    let url = format!("{}/openai/v1/chat/completions", gw.url());
    // Two requests per location, six in all. Separate buckets would see two each and never trip a
    // limit of two; one bucket sees six inside at most two windows, so at least one window has 3.
    let mut statuses = Vec::new();
    for round in 0..2 {
        for loc in 0..3 {
            let mut req = test_client()
                .post(if loc == 2 {
                    format!("{url}?key={vk}")
                } else {
                    url.clone()
                })
                .header("content-type", "application/json")
                .body(CHAT);
            req = match loc {
                0 => req.header("authorization", format!("Bearer {vk}")),
                1 => req.header("x-api-key", vk.clone()),
                _ => req,
            };
            statuses.push((round, loc, req.send().await.unwrap().status().as_u16()));
        }
    }
    assert!(
        statuses.iter().any(|(_, _, s)| *s == 429),
        "one identity in three locations must share one bucket: {statuses:?}"
    );

    let respelled = vk.replacen("bai_v1.1.", "bai_v1.01.", 1);
    let resp = test_client()
        .post(&url)
        .header("authorization", format!("Bearer {respelled}"))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "a non-canonical kid does not verify");
}

// --- SEC-13 --------------------------------------------------------------------------------------

/// Every way to forge a managed key — tampered payload or signature, a key from another signer, an
/// unknown or respelled kid, a truncated or padded token, a version swap — gets the same 401 with
/// the same body, on every route and in every credential location, and none reaches a provider.
/// claim: SEC-13
#[tokio::test]
async fn every_forged_key_gets_one_byte_identical_401_and_no_upstream_call() {
    let (pubkey, sk) = test_keypair(1);
    let (_, other) = test_keypair(2);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic"])
        .start()
        .await;
    let good = vkey(&sk, 13);
    let parts: Vec<&str> = good.split('.').collect();
    let (payload, sig) = (parts[2], parts[3]);
    let flip = |s: &str| {
        let mut b = s.as_bytes().to_vec();
        b[3] = if b[3] == b'A' { b'B' } else { b'A' };
        String::from_utf8(b).unwrap()
    };
    let id = VirtualKey {
        tenant_id: 13,
        vpc_id: 1,
        key_id: None,
    };
    let v2 = mint_v2(&id, 77, 1, &sk);
    let v2_parts: Vec<&str> = v2.split('.').collect();
    let forged = vec![
        format!("bai_v1.1.{}.{sig}", flip(payload)),
        format!("bai_v1.1.{payload}.{}", flip(sig)),
        vkey(&other, 13),
        format!("bai_v1.9.{payload}.{sig}"),
        format!("bai_v1.01.{payload}.{sig}"),
        format!("bai_v1.1.{payload}"),
        format!("bai_v1.1.{payload}.{sig}="),
        format!("bai_v1.1.{payload}.{}", &sig[..sig.len() - 2]),
        format!("bai_v2.1.{payload}.{sig}"),
        format!("bai_v1.1.{}.{}", v2_parts[2], v2_parts[3]),
        "bai_v1".to_owned(),
        "bai_v2.1.aaaa.bbbb".to_owned(),
    ];
    let mut bodies = std::collections::BTreeSet::new();
    for key in &forged {
        for (path, header, value) in [
            (
                "/openai/v1/chat/completions",
                "authorization",
                format!("Bearer {key}"),
            ),
            (
                "/v1/chat/completions",
                "authorization",
                format!("Bearer {key}"),
            ),
            (
                "/auto/v1/chat/completions",
                "authorization",
                format!("Bearer {key}"),
            ),
            ("/anthropic/v1/messages", "x-api-key", key.clone()),
            ("/v1/messages", "x-api-key", key.clone()),
        ] {
            let resp = test_client()
                .post(format!("{}{path}", gw.url()))
                .header(header, value)
                .header("content-type", "application/json")
                .header("x-beyond-model", "gpt-4o-mini")
                .body(CHAT)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 401, "{key} on {path}");
            let mut names: Vec<String> = resp
                .headers()
                .keys()
                .map(|k| k.as_str().to_owned())
                .filter(|k| k != "date" && k != "connection")
                .collect();
            names.sort();
            assert_eq!(
                names,
                ["content-length", "content-type", "x-beyond-request-id"],
                "{key} on {path}: the 401 says nothing else"
            );
            bodies.insert(resp.bytes().await.unwrap().to_vec());
        }
    }
    assert_eq!(
        bodies.len(),
        1,
        "every forgery must get the same body: {:?}",
        bodies
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect::<Vec<_>>()
    );
    assert_eq!(mock.hits(), 0, "no forged key may reach a provider");
}

// --- SEC-15 --------------------------------------------------------------------------------------

/// Per-caller state stays with its caller. One tenant's session pin does not move another tenant
/// (or another credential of the same tenant); one tenant at its rate limit or concurrency cap does
/// not throttle another.
/// claim: SEC-15
/// defect: D58
#[tokio::test]
async fn pins_limits_and_slots_do_not_cross_tenants() {
    let (pubkey, sk) = test_keypair(1);
    let slow = MockUpstream::start(Mode::Slow(60)).await;
    let fast = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &slow.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .provider_authority("openrouter", &fast.authority())
        .start()
        .await;
    let client = test_client();
    let post = |key: String| {
        let (c, url) = (client.clone(), gw.url());
        async move {
            c.post(format!("{url}/auto/chat/completions"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .header("x-beyond-model", "gpt-4o-mini")
                .body(CHAT)
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        }
    };
    let tenant = |t: u64, key_id: Option<u64>| {
        let id = VirtualKey {
            tenant_id: t,
            vpc_id: 1,
            key_id: None,
        };
        match key_id {
            Some(k) => mint_v2(&id, k, 1, &sk),
            None => mint(&id, 1, &sk),
        }
    };

    // A session pin is computed from the caller's own verified identity every turn and keeps no
    // state, so nothing another caller does can move it. Tenant B steering its own requests at the
    // fast arm (an order header), and a sibling credential of A doing the same, leave tenant A on
    // the catalog primary, turn after turn.
    let ordered = |key: String| {
        let (c, url) = (client.clone(), gw.url());
        async move {
            c.post(format!("{url}/auto/chat/completions"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .header("x-beyond-model", "gpt-4o-mini")
                .header("x-beyond-order", "openrouter")
                .body(CHAT)
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        }
    };
    for _ in 0..4 {
        assert_eq!(ordered(tenant(300, None)).await, 200);
        assert_eq!(ordered(tenant(100, Some(5))).await, 200);
        assert_eq!(post(tenant(100, None)).await, 200);
    }
    assert_eq!(slow.hits(), 4, "tenant A stayed on its pin, the primary");
    assert_eq!(
        fast.hits(),
        8,
        "the other callers' steering moved only them"
    );

    // Rate limit: tenant C flooding its own bucket leaves tenant D untouched.
    let mock = MockUpstream::start(Mode::Json).await;
    let gw2 = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .rate_limit_rps(2)
        .byo_rate_limit_rps(0)
        .start()
        .await;
    let direct = |key: String| {
        let (c, url) = (client.clone(), gw2.url());
        async move {
            c.post(format!("{url}/openai/v1/chat/completions"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(CHAT)
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        }
    };
    let mut c_statuses = Vec::new();
    for _ in 0..7 {
        c_statuses.push(direct(tenant(400, None)).await);
    }
    assert!(
        c_statuses.contains(&429),
        "tenant C hit its own limit: {c_statuses:?}"
    );
    assert_eq!(
        direct(tenant(401, None)).await,
        200,
        "tenant D has its own bucket"
    );

    // Concurrency: tenant E holding its one slot leaves tenant F a slot of its own.
    let held = MockUpstream::start(Mode::Slow(1500)).await;
    let gw3 = Gateway::builder(unused_nats_port(), &held.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .tenant_max_in_flight(1)
        .start()
        .await;
    let busy = {
        let (c, url, key) = (client.clone(), gw3.url(), tenant(500, None));
        tokio::spawn(async move {
            c.post(format!("{url}/openai/v1/chat/completions"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(CHAT)
                .send()
                .await
                .unwrap()
                .status()
                .as_u16()
        })
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while held.hits() == 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let second_e = client
        .post(format!("{}/openai/v1/chat/completions", gw3.url()))
        .header("authorization", format!("Bearer {}", tenant(500, None)))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap()
        .status()
        .as_u16();
    assert_eq!(second_e, 429, "tenant E is at its cap");
    let f = client
        .post(format!("{}/openai/v1/chat/completions", gw3.url()))
        .header("authorization", format!("Bearer {}", tenant(501, None)))
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap()
        .status()
        .as_u16();
    assert_eq!(f, 200, "tenant F is not throttled by E's slot");
    assert_eq!(busy.await.unwrap(), 200);
}

// --- SEC-17 --------------------------------------------------------------------------------------

/// The stated bound: a deny written to the control plane refuses the tenant's next request within
/// two seconds on a connected gateway. The pinned policy for a stream already in flight is that it
/// runs to completion: the deny-set gates admission, it does not cut a paid-for generation.
/// claim: SEC-17
#[tokio::test]
async fn a_deny_lands_within_the_bound_and_in_flight_streams_finish() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let up = ReplyUpstream::start(|n, _| {
        if n == 0 {
            // The in-flight request: its answer arrives after the deny has landed.
            Reply::Delayed(Duration::from_millis(1500), Box::new(Reply::sse()))
        } else {
            Reply::sse()
        }
    })
    .await;
    let gw = Gateway::builder(nats.port, &up.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = vkey(&sk, 1717);
    let post = {
        let (url, key) = (gw.url(), key.clone());
        move || {
            let (url, key) = (url.clone(), key.clone());
            async move {
                test_client()
                    .post(format!("{url}/openai/v1/chat/completions"))
                    .header("authorization", format!("Bearer {key}"))
                    .header("content-type", "application/json")
                    .body(r#"{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}"#)
                    .send()
                    .await
                    .unwrap()
            }
        }
    };
    let in_flight = tokio::spawn({
        let post = post.clone();
        async move {
            let resp = post().await;
            (
                resp.status().as_u16(),
                resp.text().await.unwrap_or_default(),
            )
        }
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while up.hits() == 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(up.hits(), 1, "the stream is in flight before the deny");

    put_kv(nats.port, "blackhole.1717", b"spend").await;
    let written = Instant::now();
    let mut status = 0;
    while written.elapsed() < Duration::from_secs(2) {
        status = post().await.status().as_u16();
        if status == 402 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(status, 402, "the deny did not land within 2s");

    let (status, body) = in_flight.await.unwrap();
    assert_eq!(
        status, 200,
        "the in-flight stream was admitted before the deny"
    );
    assert!(body.contains("[DONE]"), "and runs to completion: {body}");
    // Billed exactly: the in-flight stream's row (the process's first request, seq 0) is the
    // provider's own usage, not refused or estimated.
    let deadline = Instant::now() + Duration::from_secs(10);
    let row = loop {
        let first = usage_rows_of(&gw).into_iter().find(|r| {
            r["request_id"]
                .as_str()
                .is_some_and(|id| id.ends_with("-0"))
        });
        match first {
            Some(row) => break row,
            None if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            None => panic!("no row for the in-flight stream; log:\n{}", gw.log()),
        }
    };
    assert_eq!(row["outcome"], "ok", "{row}");
    assert_eq!(row["usage_estimated"], false, "{row}");
    assert_eq!(
        (row["input_tokens"].as_u64(), row["output_tokens"].as_u64()),
        (Some(5), Some(9)),
        "{row}"
    );
}

/// The bound and the in-flight policy the test above pins are the documented contract, and the
/// numbers ARCHITECTURE.md derives them from are the code's: a control-plane change is pushed (no
/// gateway-side timer), applied before the next admission, bounded at 2s while the watch is
/// connected; after a NATS outage it lands once the watcher's reconnect backoff (capped at
/// `RECONNECT_BACKOFF_MAX`) reattaches and replays. A stream admitted before it runs to completion
/// and is billed exactly.
/// claim: SEC-17, TEN-1
/// defect: D121
#[test]
fn architecture_states_the_revocation_bound_and_in_flight_policy() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let arch = std::fs::read_to_string(root.join("ARCHITECTURE.md")).unwrap();
    let watch = std::fs::read_to_string(root.join("src/store_watch.rs")).unwrap();
    let heading = "### Revocation: how fast a change lands, and streams in flight";
    let start = arch
        .find(heading)
        .unwrap_or_else(|| panic!("ARCHITECTURE.md has no {heading:?} section"));
    let section = &arch[start + heading.len()..];
    let section = &section[..section.find("\n### ").unwrap_or(section.len())];

    let cap = watch
        .lines()
        .find_map(|l| {
            l.strip_prefix("const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(")?
                .strip_suffix(");")
        })
        .expect("store_watch.rs defines RECONNECT_BACKOFF_MAX in whole seconds");
    for needle in [
        "no gateway-side timer",
        "within 2s",
        &format!("{cap}s"),
        "`request_filter`",
        "runs to completion",
        "billed exactly",
        "`blackhole.",
        "`allowance.",
    ] {
        assert!(
            section.contains(needle),
            "the revocation section does not state {needle:?}:\n{section}"
        );
    }
}

// --- SEC-18 --------------------------------------------------------------------------------------

/// Framing an intermediary and the gateway could read differently is refused, and nothing hidden in
/// a body ever reaches the provider as a second request.
/// claim: SEC-18
#[tokio::test]
async fn smuggled_and_malformed_framing_is_rejected() {
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let head = "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: x\r\n\
                authorization: Bearer sk-byo\r\ncontent-type: application/json\r\n";
    let smuggled =
        "GET /openai/v1/files HTTP/1.1\r\nhost: x\r\nauthorization: Bearer sk-byo\r\n\r\n";

    // Rejected outright: no upstream contact at all.
    for (what, extra, body) in [
        (
            "two different content-lengths",
            "content-length: 2\r\ncontent-length: 4\r\n",
            "{}{}",
        ),
        (
            "non-numeric content-length",
            "content-length: abc\r\n",
            "{}",
        ),
        ("signed content-length", "content-length: +2\r\n", "{}"),
        (
            "unknown transfer-encoding",
            "transfer-encoding: gzip\r\n",
            "{}",
        ),
    ] {
        let before = mock.hits();
        let raw = raw_exchange(
            gw.port,
            format!("{head}{extra}\r\n{body}").as_bytes(),
            Duration::from_secs(2),
        )
        .await;
        let got = statuses(&raw);
        assert!(
            got.first().is_none_or(|s| (400..500).contains(s)),
            "{what}: {got:?} {}",
            String::from_utf8_lossy(&raw)
        );
        assert_eq!(mock.hits(), before, "{what}: reached the provider");
    }

    // CL.TE / TE.CL desync attempts: whatever the gateway answers, the provider sees at most the
    // one request, and never the smuggled one.
    for (what, extra, body) in [
        (
            "content-length beside chunked",
            "content-length: 4\r\ntransfer-encoding: chunked\r\n".to_owned(),
            format!("0\r\n\r\n{smuggled}"),
        ),
        (
            "chunked beside content-length",
            "transfer-encoding: chunked\r\ncontent-length: 50\r\n".to_owned(),
            format!("2\r\n{{}}\r\n0\r\n\r\n{smuggled}"),
        ),
        (
            "chunked, obfuscated",
            "transfer-encoding: chunked\r\ntransfer-encoding: identity\r\n".to_owned(),
            format!("0\r\n\r\n{smuggled}"),
        ),
    ] {
        let before = mock.hits();
        let raw = raw_exchange(
            gw.port,
            format!("{head}{extra}\r\n{body}").as_bytes(),
            Duration::from_secs(2),
        )
        .await;
        let new = mock.hits() - before;
        assert!(new <= 1, "{what}: the provider saw {new} requests");
        if new == 1 {
            let cap = mock.captured().unwrap();
            assert_eq!(cap.path, "/v1/chat/completions", "{what}");
            assert!(
                !String::from_utf8_lossy(&cap.body).contains("GET /"),
                "{what}: the smuggled request rode in the body upstream"
            );
        }
        assert!(
            statuses(&raw).len() <= 1,
            "{what}: the connection answered a second (smuggled) request: {}",
            String::from_utf8_lossy(&raw)
        );
    }
}

// --- SEC-19 --------------------------------------------------------------------------------------

/// Oversized request heads, absurd paths and pathologically nested JSON are refused or survived:
/// none reaches the provider as-is when over a bound, none crashes the gateway, and the next
/// ordinary request is still served.
/// claim: SEC-19
#[tokio::test]
async fn oversized_headers_paths_and_deep_json_are_bounded() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::AnthropicJson).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic", "openrouter", "bedrock"])
        .start()
        .await;
    let vk = vkey(&sk, 19);

    // One 1 MiB header and a 1 MiB path: refused before any upstream contact.
    for (what, req) in [
        (
            "1 MiB header",
            format!(
                "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer sk-byo\r\n\
                 x-filler: {}\r\ncontent-length: 2\r\n\r\n{{}}",
                "a".repeat(1 << 20)
            ),
        ),
        (
            "1 MiB path",
            format!(
                "POST /openai/v1/{}/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer sk-byo\r\n\
                 content-length: 2\r\n\r\n{{}}",
                "a".repeat(1 << 20)
            ),
        ),
        (
            "10k headers",
            format!(
                "POST /openai/v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer sk-byo\r\n\
                 {}content-length: 2\r\n\r\n{{}}",
                (0..10_000)
                    .map(|i| format!("x-h{i}: v\r\n"))
                    .collect::<String>()
            ),
        ),
    ] {
        let before = mock.hits();
        let raw = raw_exchange(gw.port, req.as_bytes(), Duration::from_secs(3)).await;
        let got = statuses(&raw);
        assert!(
            got.first().is_none_or(|s| (400..500).contains(s)),
            "{what}: {got:?}"
        );
        assert_eq!(mock.hits(), before, "{what}: reached the provider");
    }

    // 100k-deep JSON, routed (peek) and translated (Chat Completions → Messages): an answer, not a
    // crash, and no stack blown.
    let deep = format!(
        r#"{{"model":"claude-opus-4-8","messages":[{{"role":"user","content":"hi"}}],"metadata":{}{}}}"#,
        "[".repeat(100_000),
        "]".repeat(100_000)
    );
    let resp = test_client()
        .post(format!("{}/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {vk}"))
        .header("content-type", "application/json")
        .body(deep)
        .send()
        .await
        .expect("the gateway answers a deeply nested body");
    assert!(
        resp.status().as_u16() < 500 || resp.status().as_u16() == 502,
        "deep JSON: {}",
        resp.status()
    );

    // Still serving.
    let resp = test_client()
        .post(format!("{}/anthropic/v1/messages", gw.url()))
        .header("x-api-key", vk)
        .header("content-type", "application/json")
        .body(MESSAGES)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "the gateway survived every bound probe");
}

// --- SEC-22 --------------------------------------------------------------------------------------

/// With verification on (the default), an upstream whose certificate does not verify gets no
/// request: the handshake fails, the client gets a gateway error, and the pool key never leaves.
/// claim: SEC-22
#[tokio::test]
async fn an_unverifiable_upstream_cert_fails_closed() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start_tls(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .tls_upstream()
        // The harness writes `upstream_verify_cert = false` for its self-signed mock; production's
        // default is on, and the env override restores it.
        .env("AI_UPSTREAM_VERIFY_CERT", "true")
        .start()
        .await;
    for auth in [format!("Bearer {}", vkey(&sk, 22)), "Bearer sk-byo".into()] {
        let resp = test_client()
            .post(format!("{}/openai/v1/chat/completions", gw.url()))
            .header("authorization", auth)
            .header("content-type", "application/json")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        assert!(
            resp.status().is_server_error(),
            "a self-signed upstream must not be trusted: {}",
            resp.status()
        );
    }
    assert_eq!(
        mock.hits(),
        0,
        "no request crossed an unverified TLS session"
    );
}

// --- SEC-23 --------------------------------------------------------------------------------------

/// `/metrics`, `/livez` and `/readyz` live on the internal listener only. On the client listener
/// they are unknown routes. And the scrape carries no tenant or key identifier, even after managed
/// traffic tagged with metadata.
/// claim: SEC-23
#[tokio::test]
async fn admin_endpoints_are_not_on_the_client_listener_and_metrics_name_no_tenant() {
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    for path in [
        "/metrics",
        "/livez",
        "/readyz",
        "/metrics/",
        "/v1/../metrics",
    ] {
        let resp = test_client()
            .get(format!("{}{path}", gw.url()))
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        assert!(
            (400..500).contains(&status),
            "{path} on the client listener: {status}"
        );
        assert!(!body.contains("ai_requests_total"), "{path}: {body}");
    }
    let tenant = 918_273_645;
    let resp = test_client()
        .post(format!("{}/openai/v1/chat/completions", gw.url()))
        .header("authorization", format!("Bearer {}", vkey(&sk, tenant)))
        .header("content-type", "application/json")
        .header("x-beyond-metadata", r#"{"customer":"acme-secret-co"}"#)
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    wait_for_metric(&gw, "ai_requests_total", "", 1.0).await;
    let scrape = gw.metrics().await;
    let tenant = tenant.to_string();
    for needle in [
        tenant.as_str(),
        "acme-secret-co",
        "bai_v",
        "tenant=",
        "tenant_id",
        "vpc",
    ] {
        assert!(!scrape.contains(needle), "/metrics carries {needle}");
    }
}

// --- SEC-24 --------------------------------------------------------------------------------------

/// A caller's `x-beyond-capture` reaches only its own request: `off` suppresses that one request
/// and leaves the operator's tenant-wide capture on for the next; `on` cannot exceed the per-request
/// byte cap the operator set, so a forced capture is no bigger than an operator one.
/// claim: SEC-24
#[tokio::test]
async fn capture_off_is_per_request_and_forced_capture_is_bounded() {
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats.port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .capture_max_bytes(256)
        .start()
        .await;
    let operator_tenant = 2424;
    put_kv(nats.port, &format!("aicapture.{operator_tenant}"), b"{}").await;
    wait_for_metric(&gw, "ai_capture_set_size", "", 1.0).await;
    let payloads = |gw: &Gateway| gw.log().matches(r#""target":"ai.payload""#).count();
    let send = |tenant: u64, header: Option<&'static str>, content: String| {
        let (url, key) = (gw.url(), vkey(&sk, tenant));
        async move {
            let mut req = test_client()
                .post(format!("{url}/openai/v1/chat/completions"))
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(format!(
                    r#"{{"model":"gpt-4o-mini","messages":[{{"role":"user","content":"{content}"}}]}}"#
                ));
            if let Some(h) = header {
                req = req.header("x-beyond-capture", h);
            }
            assert_eq!(req.send().await.unwrap().status(), 200);
        }
    };
    let wait_payloads = |gw: &Gateway, n: usize| {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut got = payloads(gw);
        while got < n && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(25));
            got = payloads(gw);
        }
        got
    };

    send(operator_tenant, Some("off"), "private".into()).await;
    send(operator_tenant, None, "after-the-opt-out".into()).await;
    assert_eq!(wait_payloads(&gw, 1), 1);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let log = gw.log();
    assert_eq!(
        payloads(&gw),
        1,
        "only the request without the opt-out: {log}"
    );
    assert!(
        log.contains("after-the-opt-out"),
        "operator capture is still on"
    );
    assert!(!log.contains("private"), "the opted-out request stayed out");

    // A caller forcing capture on its own tenant gets the operator's byte cap.
    send(5151, Some("on"), "x".repeat(10_000)).await;
    assert_eq!(wait_payloads(&gw, 2), 2);
    let line = gw
        .log()
        .lines()
        .filter(|l| l.contains(r#""target":"ai.payload""#))
        .find(|l| l.contains("xxxx"))
        .expect("the forced capture")
        .to_owned();
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    let body = v["fields"]["request_body"].as_str().unwrap_or("");
    assert!(
        body.len() <= 256,
        "forced capture exceeded the cap: {} bytes",
        body.len()
    );
}

/// Send a BYO request to bare `/v1{path}` with these credential headers (repeats kept) and return
/// the status.
async fn bare_byo(gw: &Gateway, path: &str, headers: &[(&'static str, &str)]) -> u16 {
    let mut map = reqwest::header::HeaderMap::new();
    for (k, v) in headers {
        map.append(
            reqwest::header::HeaderName::from_static(k),
            reqwest::header::HeaderValue::from_str(v).unwrap(),
        );
    }
    let resp = test_client()
        .post(format!("{}{path}", gw.url()))
        .headers(map)
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let _ = resp.bytes().await;
    status
}

/// Bare `/v1` routes a BYO request by the credential it will forward, never by a stray header
/// beside it. An OpenAI-shaped `Bearer sk-…` never reaches Anthropic, whatever the path; every
/// `x-api-key` value counts, not just the first; and credentials for two different providers on one
/// request are a 400, since the gateway forwards them all and cannot know which one the caller
/// meant.
/// claim: SEC-11
/// defect: D82
#[tokio::test]
async fn bare_v1_byo_routes_by_the_forwarded_credential() {
    let openai = MockUpstream::start(Mode::Json).await;
    let anthropic = MockUpstream::start(Mode::Json).await;
    let (pubkey, _sk) = test_keypair(82);
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic"])
        .provider_authority("anthropic", &anthropic.authority())
        .start()
        .await;
    // A stray x-api-key beside an OpenAI key: ambiguous, and nothing leaves the gateway.
    let status = bare_byo(
        &gw,
        "/v1/chat/completions",
        &[
            ("x-api-key", "stray"),
            ("authorization", "Bearer sk-proj-openai-byo"),
        ],
    )
    .await;
    assert_eq!(
        (status, openai.hits(), anthropic.hits()),
        (400, 0, 0),
        "mixed BYO credentials"
    );
    // An OpenAI key on the Messages path goes to OpenAI (which will refuse it), not to Anthropic.
    let status = bare_byo(
        &gw,
        "/v1/messages",
        &[("authorization", "Bearer sk-proj-openai-byo")],
    )
    .await;
    assert_eq!(
        (status, openai.hits(), anthropic.hits()),
        (200, 1, 0),
        "an OpenAI key reached Anthropic"
    );
    // The Anthropic key is the second x-api-key line: it still routes to Anthropic.
    let status = bare_byo(
        &gw,
        "/v1/chat/completions",
        &[("x-api-key", ""), ("x-api-key", "sk-ant-api03-byo")],
    )
    .await;
    assert_eq!(
        (status, openai.hits(), anthropic.hits()),
        (200, 1, 1),
        "an Anthropic key reached OpenAI"
    );
}

/// A managed key anywhere makes a bare `/v1` request managed: BYO credentials for two providers
/// beside it are not ambiguous, because every credential location is stripped and the pool key
/// goes instead. The request is served, and no client credential reaches the provider.
/// claim: SEC-11
/// defect: D82
#[tokio::test]
async fn a_managed_key_beside_mixed_byo_credentials_is_not_ambiguous() {
    let openai = MockUpstream::start(Mode::Json).await;
    let (pubkey, sk) = test_keypair(82);
    let gw = Gateway::builder(unused_nats_port(), &openai.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let managed = format!("Bearer {}", billing_vkey(&sk, 821));
    let status = bare_byo(
        &gw,
        "/v1/chat/completions",
        &[
            ("authorization", &managed),
            ("x-api-key", "stray"),
            ("x-api-key", "sk-proj-openai-byo"),
        ],
    )
    .await;
    assert_eq!((status, openai.hits()), (200, 1), "{}", gw.log());
    let cap = openai.captured().unwrap();
    assert!(cap.x_api_key.is_none(), "{:?}", cap.x_api_key);
    assert_ne!(cap.authorization.as_deref(), Some(managed.as_str()));
}

/// A BYO `?key=` query parameter is the caller's own credential on its own account: forwarded with
/// the rest of the path, untouched. Only a managed (virtual) key is stripped from the query.
/// claim: SEC-11
#[tokio::test]
async fn a_byo_key_query_parameter_is_forwarded() {
    let mock = MockUpstream::start(Mode::Json).await;
    let (pubkey, _sk) = test_keypair(82);
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let resp = test_client()
        .post(format!(
            "{}/openai/v1/chat/completions?key=byo-secret&x=1",
            gw.url()
        ))
        .header("authorization", "Bearer sk-proj-openai-byo")
        .header("content-type", "application/json")
        .body(CHAT)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        mock.captured().unwrap().path,
        "/v1/chat/completions?key=byo-secret&x=1"
    );
}
