//! End-to-end: what a managed key may reach.
//!
//! A managed key spends Beyond's shared pool key, so it reaches only metered generation calls:
//! `POST` to a generation endpoint (or its count/compact sub-resources). Anything else on the
//! shared key would let one tenant reach data another tenant stored with the provider, and would
//! run unmetered. Every refusal here is asserted at the upstream (zero hits), not just at the
//! status code. BYO keys are the caller's own and pass through untouched.
//!
//! Run via `mise run test:integration:rs` (needs `nats-server` on PATH).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

fn managed_key(sk: &ed25519_dalek::SigningKey) -> String {
    mint(
        &VirtualKey {
            tenant_id: 11,
            vpc_id: 1,
            key_id: None,
        },
        1,
        sk,
    )
}

async fn send(
    gw: &Gateway,
    method: reqwest::Method,
    path: &str,
    auth: (&str, String),
    body: Option<&str>,
) -> (u16, String, Option<String>) {
    let mut req = test_client()
        .request(method, format!("{}{path}", gw.url()))
        .header(auth.0, auth.1);
    if let Some(b) = body {
        req = req
            .header("content-type", "application/json")
            .body(b.to_owned());
    }
    let resp = req.send().await.unwrap();
    let status = resp.status().as_u16();
    let rid = resp
        .headers()
        .get("x-beyond-request-id")
        .map(|v| v.to_str().unwrap().to_owned());
    (status, resp.text().await.unwrap(), rid)
}

const CHAT: &str = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}"#;

/// claim: SEC-1
/// defect: D01
#[tokio::test]
async fn a_managed_key_reaches_only_generation_endpoints_on_a_provider_route() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic"])
        .start()
        .await;
    let bearer = || ("authorization", format!("Bearer {}", managed_key(&sk)));
    let x_api_key = || ("x-api-key", managed_key(&sk));

    use reqwest::Method as M;
    // Not generation calls: stored data, batches, fine-tuning, model admin, and generation paths
    // reached with the wrong method.
    let refused: Vec<(M, &str, u16)> = vec![
        (M::GET, "/openai/v1/files", 404),
        (M::GET, "/openai/v1/files/file-abc/content", 404),
        (M::DELETE, "/openai/v1/files/file-abc", 404),
        (M::GET, "/openai/v1/responses/resp_abc", 404),
        (M::DELETE, "/openai/v1/responses/resp_abc", 404),
        (M::GET, "/openai/v1/chat/completions", 405),
        (M::GET, "/openai/v1/chat/completions/chatcmpl-abc", 404),
        (M::POST, "/openai/v1/batches", 404),
        (M::POST, "/openai/v1/files", 404),
        (M::POST, "/openai/v1/fine_tuning/jobs", 404),
        (M::POST, "/openai/v1/images/generations", 404),
        (M::POST, "/openai/v1/responses/resp_abc/cancel", 404),
        (M::GET, "/openai/v1/models", 404),
        (M::POST, "/openai/v1/vector_stores", 404),
        (M::POST, "/openai/v1/completions", 404),
    ];
    for (method, path, want) in &refused {
        let body = (method == M::POST).then_some(CHAT);
        let (status, text, rid) = send(&gw, method.clone(), path, bearer(), body).await;
        assert_eq!(status, *want, "{method} {path}: {text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error", "{text}");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("managed key"),
            "{text}"
        );
        assert!(rid.is_some(), "{method} {path}: no x-beyond-request-id");
    }
    for (method, path) in [
        (M::GET, "/anthropic/v1/messages/batches"),
        (M::POST, "/anthropic/v1/messages/batches"),
        (M::GET, "/anthropic/v1/files"),
        (M::GET, "/anthropic/v1/models"),
    ] {
        let body = (method == M::POST).then_some(CHAT);
        let (status, text, _) = send(&gw, method.clone(), path, x_api_key(), body).await;
        assert!(
            matches!(status, 404 | 405),
            "{method} {path}: {status} {text}"
        );
    }
    assert_eq!(
        mock.hits(),
        0,
        "no refused request may reach the provider with the pool key"
    );

    // The generation calls themselves still work, on every allowed endpoint.
    for path in [
        "/openai/v1/chat/completions",
        "/openai/v1/responses",
        "/openai/v1/embeddings",
        "/openai/v1/responses/input_tokens",
        "/openai/v1/chat/completions?api-version=2024-10-21",
    ] {
        let (status, text, _) = send(&gw, M::POST, path, bearer(), Some(CHAT)).await;
        assert_eq!(status, 200, "{path}: {text}");
    }
    for path in [
        "/anthropic/v1/messages",
        "/anthropic/v1/messages/count_tokens",
    ] {
        let (status, text, _) = send(&gw, M::POST, path, x_api_key(), Some(CHAT)).await;
        assert_eq!(status, 200, "{path}: {text}");
    }
    assert_eq!(mock.hits(), 7);
    let metrics = gw.metrics().await;
    assert!(
        metrics.contains(r#"ai_rejections_total{reason="managed_endpoint"} 19"#),
        "every refusal is counted:\n{metrics}"
    );
}

/// claim: SEC-2
/// defect: D02
#[tokio::test]
async fn catalog_paths_take_post_only_from_a_managed_key() {
    let nats_port = unused_nats_port();
    let (pubkey, sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "openrouter"])
        .start()
        .await;
    let bearer = || ("authorization", format!("Bearer {}", managed_key(&sk)));

    use reqwest::Method as M;
    for (method, path) in [
        (M::GET, "/auto/v1/chat/completions"),
        (M::GET, "/auto/chat/completions"),
        (M::DELETE, "/auto/v1/chat/completions"),
        (M::GET, "/v1/chat/completions"),
        (M::GET, "/v1/responses/resp_abc"),
        (M::DELETE, "/v1/responses/resp_abc"),
        (M::GET, "/v1/files"),
        (M::PUT, "/v1/chat/completions"),
    ] {
        let resp = test_client()
            .request(method.clone(), format!("{}{path}", gw.url()))
            .header(bearer().0, bearer().1)
            .header("x-beyond-model", "gpt-4o-mini")
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        assert_eq!(status, 405, "{method} {path}: {text}");
        assert!(text.contains("managed key"), "{text}");
    }
    assert_eq!(
        mock.hits(),
        0,
        "no refused catalog request reaches a candidate"
    );

    // Listing models stays open, and POST still walks the catalog.
    let (status, text, _) = send(&gw, M::GET, "/v1/models", bearer(), None).await;
    assert_eq!(status, 200, "{text}");
    let resp = test_client()
        .head(format!("{}/v1/models", gw.url()))
        .header(bearer().0, bearer().1)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let (status, text, _) = send(&gw, M::POST, "/v1/chat/completions", bearer(), Some(CHAT)).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(mock.hits(), 1);
}

/// BYO keys belong to the caller: their own files and stored responses are theirs to read, so the
/// managed allowlist must not touch them.
///
/// claim: SEC-1, SEC-10
#[tokio::test]
async fn a_byo_key_still_reaches_any_provider_endpoint() {
    let nats_port = unused_nats_port();
    let (pubkey, _sk) = test_keypair(1);
    let mock = MockUpstream::start(Mode::Json).await;
    let gw = Gateway::builder(nats_port, &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let (status, text, _) = send(
        &gw,
        reqwest::Method::GET,
        "/openai/v1/files",
        ("authorization", "Bearer sk-caller-own-key".to_owned()),
        None,
    )
    .await;
    assert_eq!(status, 200, "{text}");
    let cap = mock.captured().expect("BYO request reached the provider");
    assert_eq!(cap.path, "/v1/files");
    assert_eq!(
        cap.authorization.as_deref(),
        Some("Bearer sk-caller-own-key"),
        "BYO auth passes through untouched"
    );
}

/// What OpenAI answers a Responses `background: true` request: queued, no usage. It generates
/// afterwards and bills the account, and nothing the gateway relays ever carries that usage.
const QUEUED: &str = r#"{"id":"resp_bg","object":"response","created_at":1,"status":"queued","background":true,"model":"gpt-4.1","output":[],"usage":null}"#;

/// A Responses `background: true` request is answered `200 {"status":"queued","usage":null}` and
/// generated asynchronously, so a managed key would run it on the pool key unmetered (D195 would
/// even bill an estimate of the envelope), and a managed key cannot poll for it (GET is refused).
/// So a managed key gets a named 400 and no provider receives the request whole: on a catalog walk
/// (small body, and a body past 64 KiB that goes through the full-body re-run) the refusal comes
/// before any upstream; on the `/openai` provider route (small and large) the body is held and the
/// request aborted before its last byte, as a duplicate `model` is (D34). `background: false` is
/// served, and a BYO key is relayed as sent: the caller's own account can poll.
/// claim: BIL-2, SEC-1
/// defect: D202
#[tokio::test]
async fn background_responses_are_refused_on_managed_keys_only() {
    let (pubkey, sk) = test_keypair(202);
    // Counts only requests whose body arrived in full: what a provider would act on.
    let mock = ScriptedUpstream::reply(200, "application/json", QUEUED.to_owned()).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai"])
        .start()
        .await;
    let key = managed_key(&sk);
    let pad = "x".repeat(80 * 1024);
    let small = r#"{"model":"gpt-4.1","input":"hi","background":true}"#.to_owned();
    // `model` after `input`, past pingora's replay buffer: the catalog walk's full-body re-run.
    let large = format!(r#"{{"input":"{pad}","background":true,"model":"gpt-4.1"}}"#);
    let mut failures = Vec::new();
    for path in ["/v1/responses", "/openai/v1/responses"] {
        for body in [&small, &large] {
            let before = mock.hits();
            let (status, text, rid) = send(
                &gw,
                reqwest::Method::POST,
                path,
                ("authorization", format!("Bearer {key}")),
                Some(body),
            )
            .await;
            let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
            let named = v["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("background"));
            if status != 400 || !named || rid.is_none() || mock.hits() != before {
                failures.push(format!(
                    "{path} ({} bytes): {status}, {} upstream hits: {text}",
                    body.len(),
                    mock.hits() - before
                ));
            }
        }
    }
    let (status, text, _) = send(
        &gw,
        reqwest::Method::POST,
        "/v1/responses",
        ("authorization", format!("Bearer {key}")),
        Some(r#"{"model":"gpt-4.1","input":"hi","background":false}"#),
    )
    .await;
    if status != 200 {
        failures.push(format!("background: false: {status}: {text}"));
    }
    let before = mock.hits();
    let (status, text, _) = send(
        &gw,
        reqwest::Method::POST,
        "/openai/v1/responses",
        ("authorization", "Bearer sk-caller-own-key".to_owned()),
        Some(&small),
    )
    .await;
    if status != 200 || mock.hits() != before + 1 || !text.contains("queued") {
        failures.push(format!("BYO: {status}: {text}"));
    }
    assert!(failures.is_empty(), "{}\n{}", failures.join("\n"), gw.log());
}

/// A billable OpenAI answer, so a served request writes a priced row.
const CHAT_OK: &str = r#"{"id":"c1","object":"chat.completion","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;

fn chat_body(model: &str) -> String {
    format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#)
}

/// A managed key on `/{provider}/…` runs only what its billing row can price: a billable call whose
/// root `model` resolves to no catalog row (the lookup `price_model` makes on the requested model:
/// a catalog name or any candidate's provider-specific spelling, either with a dated snapshot
/// suffix) is the catalog walk's 404, naming the model and pointing at `/v1/models`, and no
/// provider receives the body (the mock counts only requests that arrived whole). Small and large
/// bodies, `model` before and after the prompt, every billable endpoint. A missing `model` is
/// refused too, and a duplicate one is a 400 (the provider serves the last, the row names the
/// first). The free token counts write no billing row and stay open, and a BYO key runs any slug.
/// claim: SEC-1, BIL-2
/// defect: D267
#[tokio::test]
async fn managed_provider_routes_run_catalog_models_only() {
    let (pubkey, sk) = test_keypair(66);
    let mock = ScriptedUpstream::reply(200, "application/json", CHAT_OK.to_owned()).await;
    let gw = Gateway::builder(unused_nats_port(), &mock.authority(), &b64(&pubkey))
        .providers(&["openai", "anthropic"])
        .start()
        .await;
    let key = managed_key(&sk);
    let bearer = || ("authorization", format!("Bearer {key}"));
    let x_api_key = || ("x-api-key", key.clone());
    let pad = "x".repeat(80 * 1024);
    let large = |model: &str| {
        format!(r#"{{"messages":[{{"role":"user","content":"{pad}"}}],"model":"{model}"}}"#)
    };
    let mut failures = Vec::new();

    // Priced: a catalog name, a candidate's own spelling, and a dated snapshot of either; then the
    // free token counts, whatever their model.
    let served: Vec<(&str, bool, String)> = vec![
        (
            "/openai/v1/chat/completions",
            true,
            chat_body("gpt-4o-mini"),
        ),
        (
            "/openai/v1/chat/completions",
            true,
            chat_body("gpt-4o-mini-2024-07-18"),
        ),
        (
            "/openai/v1/responses",
            true,
            r#"{"model":"gpt-4.1","input":"hi"}"#.to_owned(),
        ),
        (
            "/openai/v1/embeddings",
            true,
            r#"{"model":"text-embedding-3-small","input":"hi"}"#.to_owned(),
        ),
        (
            "/anthropic/v1/messages",
            false,
            chat_body("claude-opus-4-8"),
        ),
        (
            "/anthropic/v1/messages",
            false,
            chat_body("claude-sonnet-4-5-20250929"),
        ),
        ("/anthropic/v1/messages", false, large("claude-haiku-4-5")),
        (
            "/anthropic/v1/messages/count_tokens",
            false,
            chat_body("not-a-catalog-model"),
        ),
        (
            "/openai/v1/responses/input_tokens",
            true,
            r#"{"model":"not-a-catalog-model","input":"hi"}"#.to_owned(),
        ),
    ];
    let served_billable = 7;
    for (path, bearer_auth, body) in &served {
        let before = mock.hits();
        let auth = if *bearer_auth { bearer() } else { x_api_key() };
        let (status, text, _) = send(&gw, reqwest::Method::POST, path, auth, Some(body)).await;
        if status != 200 || mock.hits() != before + 1 {
            failures.push(format!(
                "served {path} ({} bytes): {status}: {text}",
                body.len()
            ));
        }
    }

    // Unpriced: refused before the provider has the body.
    let refused: Vec<(&str, bool, String, u16, &str)> = vec![
        (
            "/openai/v1/chat/completions",
            true,
            chat_body("my-finetune"),
            404,
            "\"my-finetune\"",
        ),
        (
            "/openai/v1/chat/completions",
            true,
            large("ft:gpt-4o:acme"),
            404,
            "\"ft:gpt-4o:acme\"",
        ),
        (
            "/openai/v1/responses",
            true,
            r#"{"model":"o9-secret","input":"hi"}"#.to_owned(),
            404,
            "\"o9-secret\"",
        ),
        (
            "/openai/v1/responses/compact",
            true,
            r#"{"model":"o9-secret","input":"hi"}"#.to_owned(),
            404,
            "\"o9-secret\"",
        ),
        (
            "/openai/v1/embeddings",
            true,
            r#"{"model":"text-embedding-ada-002","input":"hi"}"#.to_owned(),
            404,
            "\"text-embedding-ada-002\"",
        ),
        (
            "/anthropic/v1/messages",
            false,
            chat_body("claude-2.1"),
            404,
            "\"claude-2.1\"",
        ),
        (
            "/anthropic/v1/messages",
            false,
            large("claude-2.1"),
            404,
            "\"claude-2.1\"",
        ),
        (
            "/openai/v1/chat/completions",
            true,
            r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
            404,
            "missing model",
        ),
        (
            "/openai/v1/chat/completions",
            true,
            r#"{"model":"gpt-4o-mini","messages":[],"model":"my-finetune"}"#.to_owned(),
            400,
            "",
        ),
    ];
    let mut refused_ids = Vec::new();
    for (path, bearer_auth, body, want, named) in &refused {
        let before = mock.hits();
        let auth = if *bearer_auth { bearer() } else { x_api_key() };
        let (status, text, rid) = send(&gw, reqwest::Method::POST, path, auth, Some(body)).await;
        let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
        let msg = v["error"]["message"].as_str().unwrap_or("");
        let named_ok = *want != 404
            || (msg.contains(named)
                && msg.contains("/v1/models")
                && v["error"]["type"] == "invalid_request_error");
        if status != *want || !named_ok || rid.is_none() || mock.hits() != before {
            failures.push(format!(
                "refused {path} ({} bytes): {status}, {} upstream hits: {text}",
                body.len(),
                mock.hits() - before
            ));
        }
        refused_ids.extend(rid);
    }

    // A BYO key is the caller's own: any slug, relayed as sent.
    for (path, auth) in [
        (
            "/openai/v1/chat/completions",
            ("authorization", "Bearer sk-caller-own-key".to_owned()),
        ),
        (
            "/anthropic/v1/messages",
            ("x-api-key", "sk-ant-caller-own-key".to_owned()),
        ),
    ] {
        let before = mock.hits();
        let body = chat_body("my-finetune");
        let (status, text, _) = send(&gw, reqwest::Method::POST, path, auth, Some(&body)).await;
        if status != 200 || mock.hits() != before + 1 {
            failures.push(format!("BYO {path}: {status}: {text}"));
        }
    }

    // A refusal writes no billing row, and every served managed row is priced.
    let rows = wait_usage_rows(&gw, served_billable, 5).await;
    for row in &rows {
        let refused_row = refused_ids
            .iter()
            .any(|r| Some(r.as_str()) == row["request_id"].as_str());
        let tokens =
            row["input_tokens"].as_u64().unwrap_or(0) + row["output_tokens"].as_u64().unwrap_or(0);
        if refused_row {
            failures.push(format!(
                "a refused request wrote a billing row (tokens {tokens}): {row}"
            ));
        }
        if !refused_row && row["price_model"].as_str().is_none() {
            failures.push(format!("a served managed row is unpriced: {row}"));
        }
    }
    let metrics = gw.metrics().await;
    if !metrics.contains(r#"ai_rejections_total{reason="unknown_model"} 8"#) {
        failures.push("the eight catalog misses are counted as unknown_model".to_owned());
    }
    assert!(
        failures.is_empty(),
        "{}\n{}",
        failures.join("\n"),
        log_tail(&gw.log())
    );
}
