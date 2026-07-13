#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! The direct route, on the wire.
//!
//! `gateway_credential`'s unit tests prove the *resolver* picks the right base URL, path, and auth
//! header for a given environment. This proves the bytes that actually go out — and that the gateway is
//! genuinely optional: the base URL handed to `GatewayClient` here is deliberately unroutable, so a
//! request arriving at the listener at all is a request the direct route redirected.
//!
//! The provider rows' real hosts are compile-time constants, so these drive through `AI_BASE_URL`, which
//! can be pointed at a socket. It is the same `registry_direct_routing` code path either way.

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use agent_core::transport::ModelTransport;
use agent_core::{GatewayClient, Message, ModelRequest};
use beyond_ai_agent::gateway_credential::{
    GatewayCredential, ProviderEnv, resolve_gateway_credential,
};
use futures::StreamExt;

/// A one-shot listener that records the raw request text and answers with an empty SSE 200.
fn capture_request() -> (String, Arc<Mutex<String>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured = Arc::new(Mutex::new(String::new()));
    let sink = Arc::clone(&captured);
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap_or(0);
        *sink.lock().expect("lock") = String::from_utf8_lossy(&buf[..n]).to_string();
        let _ = stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        );
        let _ = stream.flush();
    });
    (format!("http://{addr}"), captured)
}

/// Resolve a credential from `env` and drive one turn through it.
async fn drive(env: &ProviderEnv, model: &str) {
    let credential =
        resolve_gateway_credential(None, model, env).expect("the credential must resolve");
    // `gateway.invalid` does not resolve. In direct mode the credential carries a `RouteOverride::Direct`
    // that replaces this base URL outright, so it is never dialed — which is what the listener receiving
    // anything at all proves.
    let client = match credential {
        GatewayCredential::Static(key) => {
            GatewayClient::new("http://gateway.invalid".to_string(), key).expect("client")
        }
        GatewayCredential::Oauth(source) => {
            GatewayClient::with_credential_source("http://gateway.invalid".to_string(), source)
                .expect("client")
        }
    };
    // The listener answers with an empty SSE body, so the stream ends immediately; a connect error is
    // equally fine to swallow here — the assertion is on what reached the socket, not on the response.
    if let Ok(mut stream) = client
        .stream(ModelRequest::new(model, vec![Message::user("hi")], 64))
        .await
    {
        while stream.next().await.is_some() {}
    }
}

/// A Claude model builds an Anthropic-wire request — `/v1/messages`, with the version header the
/// Messages API requires. The dialect came from the model; the route came from the environment.
#[tokio::test]
async fn an_anthropic_wire_route_posts_v1_messages_with_the_version_header() {
    let (base, captured) = capture_request();
    let env = ProviderEnv::from_vars(
        &[
            ("AI_BASE_URL", base.as_str()),
            ("AI_API_KEY", "sk-ant-test"),
        ],
        false,
    );
    drive(&env, "claude-opus-4-8").await;
    let request = captured.lock().expect("lock").clone();
    let lower = request.to_ascii_lowercase();

    assert!(
        lower.contains("post /v1/messages"),
        "a Claude model builds an Anthropic-wire request; got:\n{request}"
    );
    assert!(
        lower.contains("anthropic-version:"),
        "the Messages API requires anthropic-version; got:\n{request}"
    );
}

/// An OpenAI-wire route sends `Authorization: Bearer` and no `x-api-key`, and appends its endpoint path
/// to a `/v1` base without doubling the segment.
#[tokio::test]
async fn an_openai_wire_route_sends_authorization_bearer_and_no_x_api_key() {
    let (base, captured) = capture_request();
    let env = ProviderEnv::from_vars(
        &[
            ("AI_BASE_URL", &format!("{base}/v1")),
            ("AI_API_KEY", "sk-test"),
        ],
        false,
    );
    drive(&env, "deepseek-v3").await;
    let request = captured.lock().expect("lock").clone();
    let lower = request.to_ascii_lowercase();

    assert!(
        lower.contains("authorization: bearer sk-test"),
        "an OpenAI-wire provider takes the key as a Bearer; got:\n{request}"
    );
    assert!(
        !lower.contains("x-api-key:"),
        "…and must not also send x-api-key; got:\n{request}"
    );
    assert!(
        lower.contains("post /v1/chat/completions"),
        "a Chat-Completions model POSTs /v1/chat/completions, with no doubled /v1; got:\n{request}"
    );
}

/// The headline: with **no gateway configured**, a key in the environment is enough to produce a real
/// request. Before this, the same invocation failed with `no gateway key`.
#[tokio::test]
async fn with_no_gateway_configured_a_key_alone_produces_a_request() {
    let (base, captured) = capture_request();
    let env = ProviderEnv::from_vars(
        &[("AI_BASE_URL", base.as_str()), ("AI_API_KEY", "k")],
        false,
    );
    drive(&env, "gpt-5").await;
    assert!(
        captured.lock().expect("lock").contains("HTTP/1.1"),
        "a request must have reached the listener rather than the (unroutable) gateway"
    );
}
