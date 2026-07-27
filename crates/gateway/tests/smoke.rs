//! Live smoke tests against **real** providers — the proof docs and the mock can't give:
//! a real TLS/SNI handshake to the provider host, the base-path rewrite landing on a real mount
//! (200, not 404), the **managed** path (verify → deny-check → pool-key swap), and a real
//! (non-canned) response body.
//!
//! These exercise the **production** path, not BYO: the test generates an Ed25519 keypair, configures
//! the *real* provider key (from the env var) as the gateway's pool key, mints a `bai_…` virtual key,
//! and sends that. So the gateway verifies the virtual key, runs the deny-set check, and swaps in the
//! real provider key before forwarding — the same flow a real managed tenant takes. The real key only
//! ever lives in the gateway's config; the client presents the minted virtual key.
//!
//! Two safety layers so this never runs — or bills — by accident:
//!   1. Every test is `#[ignore]`, so a plain `cargo test` skips the whole file.
//!   2. When explicitly run, each test still **skips** (early-returns) unless its provider's API
//!      key env var is set — so you only ever hit the providers you have keys for.
//!
//! Run them:
//!   ANTHROPIC_API_KEY=sk-ant-… mise run test:smoke
//!   # or directly:
//!   ANTHROPIC_API_KEY=sk-ant-… cargo test -p beyond-ai --test smoke -- --ignored --nocapture
//!
//! Model ids are the cheapest small model per provider as of 2026-05; adjust if a provider retires
//! one (a model-not-found is a stale id here, not a gateway bug).

// Test target: `.unwrap()`/`.expect()`/`panic!` are assertions, not production code. See e2e.rs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai::key::{VirtualKey, mint};
use common::*;

/// The provider's API key from the environment, or `None` (→ the test logs a skip and returns).
fn env_key(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.trim().is_empty())
}

/// A gateway wired to the **real** provider hosts over TLS, with `provider`'s pool key set to the
/// caller's real key and a signing key installed — so a minted virtual key for `provider` verifies
/// and swaps to the real key. Returns the gateway plus the minted `bai_…` key to present as a client.
/// (Its own nats-server backs the deny-set, empty here — this tenant isn't denied.)
async fn managed_gateway(nats: &Nats, provider: &str, real_key: &str) -> (Gateway, String) {
    let (pubkey, sk) = test_keypair(7);
    let gw = Gateway::builder(nats.port, "unused", &b64(&pubkey))
        .real_upstreams()
        .pool_key(provider, real_key)
        .start()
        .await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 1,
            vpc_id: 1,
        },
        1,
        &sk,
    );
    (gw, vkey)
}

/// Drive one OpenAI-wire provider through the gateway as a managed request. The provider is selected
/// by the first path segment; `chat_path` is the full gateway path — `/{provider}/{native-base}/
/// chat/completions` (the provider's own base path after the selector, forwarded verbatim).
async fn smoke_openai_wire(provider: &str, key_env: &str, model: &str, chat_path: &str) {
    let Some(key) = env_key(key_env) else {
        eprintln!("smoke[{provider}]: {key_env} unset — skipping");
        return;
    };
    let nats = Nats::start().await;
    let (gw, vkey) = managed_gateway(&nats, provider, &key).await;
    let client = reqwest::Client::new();

    let body = format!(
        r#"{{"model":"{model}","max_tokens":16,"messages":[{{"role":"user","content":"Reply with the single word: ping"}}]}}"#
    );
    let resp = client
        .post(format!("{}{chat_path}", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .expect("request to gateway");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    assert!(
        status.is_success(),
        "smoke[{provider}] model={model} path={chat_path}: expected 2xx, got {status}.\n\
         404 ⇒ wrong native path / provider segment; 401 ⇒ pool-key swap/verify; 403 ⇒ deny-set; \
         a model error ⇒ stale model id. body: {text}"
    );
    assert!(
        text.contains("\"choices\""),
        "smoke[{provider}]: {status} but no `choices` in body: {text}"
    );
    eprintln!("smoke[{provider}]: OK ({status}) — verified, swapped, real 2xx");
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_anthropic() {
    let Some(key) = env_key("ANTHROPIC_API_KEY") else {
        eprintln!("smoke[anthropic]: ANTHROPIC_API_KEY unset — skipping");
        return;
    };
    let nats = Nats::start().await;
    let (gw, vkey) = managed_gateway(&nats, "anthropic", &key).await;
    let client = reqwest::Client::new();

    // `/anthropic/v1/messages` → provider `anthropic` (selected by the path segment, stripped to
    // `/v1/messages` upstream). The minted virtual key is presented in `x-api-key` (the Anthropic
    // SDK's header); the gateway verifies it and swaps in the real key — again in `x-api-key` (not
    // Bearer). The required `anthropic-version` header passes through. This is the *only* test
    // covering the x-api-key auth scheme + a real TLS handshake to api.anthropic.com via the full
    // managed path.
    let body = r#"{"model":"claude-haiku-4-5","max_tokens":16,"messages":[{"role":"user","content":"Reply with the single word: ping"}]}"#;
    let resp = client
        .post(format!("{}/anthropic/v1/messages", gw.url()))
        .header("x-api-key", &vkey)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .expect("request to gateway");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    assert!(
        status.is_success(),
        "smoke[anthropic]: expected 2xx, got {status}. body: {text}"
    );
    assert!(
        text.contains("\"content\""),
        "smoke[anthropic]: {status} but no `content` in body: {text}"
    );
    eprintln!("smoke[anthropic]: OK ({status}) — verified, swapped to x-api-key, real 2xx");
}

// --- OpenAI-wire providers. Same code path; testing more than one confirms each host/base-path/auth
// row in `route::KNOWN_PROVIDERS` against the real endpoint. ---

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_openai() {
    smoke_openai_wire(
        "openai",
        "OPENAI_API_KEY",
        "gpt-4o-mini",
        "/openai/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_openai_responses_usage_is_metered() {
    // Track B ported a gateway-side fix for the Responses API's nested usage shape
    // (`response.completed.response.usage`, not Chat Completions' top-level `usage` field). The unit
    // test for that parser is real, but nothing had watched the *actual* gateway meter a *real*
    // Responses-routed call end to end — this is that proof: a real call must produce a non-zero
    // token count on the gateway's own metrics, not a silent zero-token row (which
    // `ai_usage_parse_errors_total` would also flag, since a managed 2xx with no parseable usage is
    // exactly the failure mode this guards against).
    let Some(key) = env_key("OPENAI_API_KEY") else {
        eprintln!("smoke[openai-responses]: OPENAI_API_KEY unset — skipping");
        return;
    };
    let nats = Nats::start().await;
    let (gw, vkey) = managed_gateway(&nats, "openai", &key).await;
    let client = reqwest::Client::new();

    let before = gw.metrics().await;
    let before_output = parse_metric(&before, "ai_tokens_total", "output");
    let before_errors = parse_metric(&before, "ai_usage_parse_errors_total", "");

    // A real Responses API request (not Chat Completions): flat `input` array, `max_output_tokens`,
    // streamed — the exact shape `dialect::openai_responses::build_body` sends.
    let body = r#"{"model":"gpt-4o-mini","input":[{"role":"user","content":[{"type":"input_text","text":"Reply with the single word: ping"}]}],"max_output_tokens":16,"stream":true,"store":false}"#;
    let resp = client
        .post(format!("{}/openai/v1/responses", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .expect("request to gateway");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    assert!(
        status.is_success(),
        "smoke[openai-responses]: expected 2xx, got {status}. body: {text}"
    );
    assert!(
        text.contains("response.completed"),
        "smoke[openai-responses]: {status} but no terminal event in body: {text}"
    );

    // The gateway must have parsed a non-zero output-token count from the *nested* usage shape — the
    // whole point of the fix under test.
    wait_for_metric(&gw, "ai_tokens_total", "output", before_output + 1.0).await;
    let after_errors = parse_metric(&gw.metrics().await, "ai_usage_parse_errors_total", "");
    assert_eq!(
        after_errors, before_errors,
        "a real Responses call must not log a usage-parse failure"
    );
    eprintln!(
        "smoke[openai-responses]: OK ({status}) — real Responses API call, usage metered correctly"
    );
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_groq() {
    // Groq mounts under `/openai/v1`; the client sends `/groq/openai/v1/...` and the gateway strips
    // `/groq` and forwards the rest verbatim. The highest-value non-`/v1` native-path case.
    smoke_openai_wire(
        "groq",
        "GROQ_API_KEY",
        "llama-3.1-8b-instant",
        "/groq/openai/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_fireworks() {
    // Fireworks mounts under `/inference/v1`: client sends `/fireworks/inference/v1/...`.
    smoke_openai_wire(
        "fireworks",
        "FIREWORKS_API_KEY",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "/fireworks/inference/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_openrouter() {
    // OpenRouter mounts under `/api/v1`: client sends `/openrouter/api/v1/...`.
    smoke_openai_wire(
        "openrouter",
        "OPENROUTER_API_KEY",
        "openai/gpt-4o-mini",
        "/openrouter/api/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_deepseek() {
    smoke_openai_wire(
        "deepseek",
        "DEEPSEEK_API_KEY",
        "deepseek-chat",
        "/deepseek/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_together() {
    smoke_openai_wire(
        "together",
        "TOGETHER_API_KEY",
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "/together/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_cerebras() {
    smoke_openai_wire(
        "cerebras",
        "CEREBRAS_API_KEY",
        "llama3.1-8b",
        "/cerebras/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_mistral() {
    smoke_openai_wire(
        "mistral",
        "MISTRAL_API_KEY",
        "mistral-small-latest",
        "/mistral/v1/chat/completions",
    )
    .await;
}

#[tokio::test]
#[ignore = "live provider smoke; run via `mise run test:smoke` with API keys set"]
async fn smoke_xai() {
    smoke_openai_wire(
        "xai",
        "XAI_API_KEY",
        "grok-3-mini",
        "/xai/v1/chat/completions",
    )
    .await;
}

/// The `AI_POOL_KEY_*` env var a provider's key conventionally lives in — mirrors the shared table's
/// `env_var`, which is what the catalog smoke below reads.
fn provider_env_var(id: providers::ProviderId) -> Option<&'static str> {
    providers::by_id(id).env_var
}

/// **Every catalog row, against the real providers, through the real gateway.**
///
/// The catalog is product data: a wrong model id or path does not fail loudly, it routes to a 404
/// that reads like the client's fault. Unit tests can check the table's shape but not its *truth* —
/// only the provider can say whether `anthropic/claude-opus-4.8` is still a thing. This is the test
/// that keeps the table honest.
///
/// Drives each `(row, candidate)` pair individually via the model route, forcing that candidate to
/// be the only usable one, so a green run means every entry is independently servable rather than
/// every row merely having *one* working primary.
///
/// Skips any candidate whose key is absent, so a partial keyring smokes what it can.
#[tokio::test]
#[ignore = "hits real providers and bills tiny requests; run via `mise run test:smoke`"]
async fn catalog_rows_are_servable() {
    let mut checked = 0usize;
    let mut skipped: Vec<String> = Vec::new();

    for route in providers::catalog::MODEL_ROUTES {
        for candidate in route.candidates {
            let spec = providers::by_id(candidate.provider);
            let Some(var) = provider_env_var(candidate.provider) else {
                skipped.push(format!("{} ({}): no env var", route.model, spec.name));
                continue;
            };
            let Some(key) = env_key(var) else {
                skipped.push(format!("{} ({}): {var} unset", route.model, spec.name));
                continue;
            };

            let nats = Nats::start().await;
            // Only this candidate holds a pool key, so the gateway has nowhere else to go — a 200
            // here is this exact (provider, path, model id) triple answering, not a fallback
            // quietly covering for it.
            let (gw, vkey) = managed_gateway(&nats, spec.name, &key).await;

            // The body is the row's own wire. `max_tokens` is required by Anthropic and harmless to
            // OpenAI, and 1 token keeps the bill to a fraction of a cent.
            let body = format!(
                r#"{{"model":"{}","max_tokens":1,"messages":[{{"role":"user","content":"hi"}}]}}"#,
                route.model,
            );
            let mut req = reqwest::Client::new()
                .post(format!("{}/auto/x", gw.url()))
                .header("authorization", format!("Bearer {vkey}"))
                .header("content-type", "application/json")
                .header("x-beyond-model", route.model);
            if route.wire == providers::WireFormat::Anthropic {
                req = req.header("anthropic-version", "2023-06-01");
            }
            let resp = req.body(body).send().await.expect("request sent");
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            assert_eq!(
                status,
                200,
                "catalog row {:?} → {} {} (model {:?}) returned {status}: {}",
                route.model,
                spec.name,
                candidate.path,
                candidate.upstream_model,
                text.chars().take(300).collect::<String>(),
            );
            // The provider echoes the id it actually ran, which is how a silently-rewritten or
            // aliased model shows up.
            eprintln!(
                "smoke[catalog]: {:<18} → {:<11} {:<24} ok",
                route.model, spec.name, candidate.upstream_model,
            );
            checked += 1;
        }
    }

    for s in &skipped {
        eprintln!("smoke[catalog]: skipped {s}");
    }
    assert!(
        checked > 0,
        "no catalog candidate was checked — set at least one provider key",
    );
    eprintln!(
        "smoke[catalog]: {checked} candidate(s) verified, {} skipped",
        skipped.len()
    );
}

/// **A live failover.** The primary candidate is pointed at a dead port; the fallback is the *real*
/// provider. Proves the whole chain against something that can actually say no: catalog → candidate
/// walk → mount → key swap in the fallback's own scheme → model-id rewrite → real 200.
///
/// The mocked failover tests prove the mechanism. This proves the mechanism against a provider that
/// has its own opinions about paths, ids, and auth headers — which is where a catalog is wrong in
/// practice.
#[tokio::test]
#[ignore = "hits a real provider and bills a tiny request; run via `mise run test:smoke`"]
async fn model_route_fails_over_to_a_real_provider() {
    let Some(key) = env_key("OPENROUTER_API_KEY") else {
        eprintln!("smoke[failover]: OPENROUTER_API_KEY unset — skipping");
        return;
    };
    // `claude-opus-4-8` is an Anthropic-wire row: Anthropic first, OpenRouter second.
    let model = "claude-opus-4-8";
    let nats = Nats::start().await;
    let (pubkey, sk) = test_keypair(7);
    let gw = Gateway::builder(nats.port, "unused", &b64(&pubkey))
        .real_upstreams()
        // Only OpenRouter is keyed, so Anthropic is not even a usable candidate...
        .pool_key("openrouter", &key)
        // ...and it is unreachable besides, so nothing can quietly serve from it.
        .provider_authority("anthropic", &GatewayBuilder::dead_authority())
        .start()
        .await;
    let vkey = mint(
        &VirtualKey {
            tenant_id: 1,
            vpc_id: 1,
        },
        1,
        &sk,
    );

    let resp = reqwest::Client::new()
        .post(format!("{}/auto/x", gw.url()))
        .header("authorization", format!("Bearer {vkey}"))
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .header("x-beyond-model", model)
        .body(format!(
            r#"{{"model":"{model}","max_tokens":1,"messages":[{{"role":"user","content":"hi"}}]}}"#
        ))
        .send()
        .await
        .expect("request sent");
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    assert_eq!(
        status,
        200,
        "live failover to OpenRouter returned {status}: {}",
        text.chars().take(400).collect::<String>(),
    );
    // The reply is Anthropic-shaped and names OpenRouter's spelling of the model — proof the rewrite
    // landed and the response came from the fallback, not from somewhere unexpected.
    assert!(
        text.contains(r#""type":"message""#),
        "want an Anthropic Messages reply: {text}"
    );
    assert!(
        text.contains("anthropic/claude-opus-4.8"),
        "the fallback must have been asked for its own id: {text}"
    );

    // And it metered: a zero-token row here would mean the Anthropic extractor never ran.
    let line = gw
        .wait_for_log_line(&["ai.usage", r#""provider":"openrouter""#])
        .await;
    assert!(
        !line.contains(r#""input_tokens":0"#),
        "the live failover must still be metered: {line}"
    );
    eprintln!("smoke[failover]: live Anthropic→OpenRouter failover ok — {line}");
}
