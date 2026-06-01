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
