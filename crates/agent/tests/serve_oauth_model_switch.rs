//! `serve` e2e: switching the active model across two different OAuth providers must re-derive the
//! gateway credential/routing for the new provider on the very next request — not keep sending
//! whichever provider's bearer token/routing was resolved for the model active at process start.
//!
//! Regression test for the "OAuth routing/credentials frozen at process start" bug: `resolve_gateway_
//! credential` used to run exactly once, before `serve` even started, and the resulting
//! `Arc<GatewayClient>` was reused verbatim by every later `set_model`/`cycle_model` rebuild — with no
//! check that the new model's provider still matched whichever OAuth credential/routing was resolved
//! at startup.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::process::{Command, Stdio};

use common::{read_until_response, spawn_model_server, turn_text, turn_text_responses};
use serde_json::json;

fn write_auth_json(home: &std::path::Path, body: &str) {
    let dir = home.join(".claude");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("auth.json"), body).unwrap();
}

/// Anthropic OAuth (no `--key`/`AI_AGENT_KEY`) → `set_model` to an OpenAI-Codex-OAuth model, with
/// neither logged in via a shared credential — the two providers' stored logins are entirely
/// independent, so the second turn can only carry the right bearer token/headers if `serve` actually
/// re-resolved credentials for the new model instead of reusing the client built for the first one.
#[test]
fn set_model_across_oauth_providers_rederives_credential_and_routing() {
    let home = tempfile::tempdir().unwrap();
    write_auth_json(
        home.path(),
        r#"{
            "anthropic": {
                "type": "oauth",
                "access": "anthropic-test-token",
                "refresh": "anthropic-refresh",
                "expires": 4000000000000
            },
            "openai-codex": {
                "type": "oauth",
                "access": "codex-test-token",
                "refresh": "codex-refresh",
                "expires": 4000000000000,
                "account_id": "acct-123"
            }
        }"#,
    );

    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    // The first turn is Anthropic-dialect SSE (`claude-test`); the second, after switching to
    // `gpt-5-codex` (an OpenAI-Responses-dialect model), must be shaped for that dialect instead —
    // crossing dialects mid-test needs a dialect-matched mock response, same as
    // `serve_max_tokens_flag_reaches_the_wire_request_and_survives_a_model_switch`'s own note.
    //
    // A Codex-OAuth-routed turn now makes TWO physical connections, not one: `gpt-5-codex` is
    // `is_codex`-routed, so beyond always attempts the Codex WebSocket transport first (matching pi's
    // real "auto" default). This plain mock server never speaks WebSocket, so that attempt's upgrade
    // handshake gets a bare `200 OK` back — not the `101 Switching Protocols` it needs — which is
    // correctly classified as a connect failure and falls through to the existing HTTP/SSE path for
    // the same turn, but the fallback is a *second*, separate connection. The mock's `responses` list
    // is consumed one-per-`accept()`, so it needs a third entry to cover that extra connection.
    let (base, bodies) = spawn_model_server(vec![
        turn_text("first"),
        turn_text_responses("ws-preflight-rejected"),
        turn_text_responses("second"),
    ]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = Command::new(bin)
        .args([
            "serve",
            "--gateway-url",
            &base,
            "--model",
            "claude-test",
            "--session-file",
            &session_file,
        ])
        .env("HOME", home.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // First turn: the Anthropic-OAuth-resolved model, no `--key` given at all — proves the initial
    // resolution still works with no explicit credential (matches `oauth_e2e.rs`'s existing coverage).
    writeln!(stdin, "{}", json!({ "type": "prompt", "message": "hi" })).unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    // Switch to a model only the `openai-codex` OAuth login can serve.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "set_model", "model": "gpt-5-codex" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "set_model");
    assert_eq!(
        frames.last().unwrap()["success"],
        true,
        "set_model to a Codex-OAuth model should succeed: {frames:#?}"
    );

    // Second turn: must now be routed/authenticated as the Codex OAuth login, not the stale Anthropic
    // client frozen at startup.
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "hi again" })
    )
    .unwrap();
    stdin.flush().unwrap();
    read_until_response(&mut stdout, "prompt");

    drop(stdin);
    child.wait().unwrap();

    let recorded = bodies.lock().unwrap();
    assert_eq!(recorded.len(), 3, "{recorded:#?}");
    let first = recorded[0].to_lowercase();
    let ws_preflight = recorded[1].to_lowercase();
    let second = recorded[2].to_lowercase();

    assert!(
        first.contains("authorization: bearer anthropic-test-token"),
        "the first turn must use the Anthropic OAuth credential: {first}"
    );
    assert!(
        first.starts_with("post /v1/messages"),
        "the first turn must hit the default Anthropic path: {first}"
    );

    // The rejected WebSocket pre-flight for the second turn already carries the re-derived Codex
    // credential/routing — it's a real attempted connection, not a stale reuse of the Anthropic
    // client, it just doesn't speak this mock's plain-HTTP dialect.
    assert!(
        ws_preflight.contains("connection: upgrade") && ws_preflight.contains("upgrade: websocket"),
        "the second turn's first connection must be the Codex WebSocket pre-flight attempt: \
         {ws_preflight}"
    );
    assert!(
        ws_preflight.contains("authorization: bearer codex-test-token"),
        "the Codex websocket pre-flight must already carry the re-derived codex OAuth credential: \
         {ws_preflight}"
    );
    assert!(
        ws_preflight.starts_with("get /openai-codex/backend-api/codex/responses"),
        "the Codex websocket pre-flight must target the Codex backend path: {ws_preflight}"
    );

    assert!(
        second.contains("authorization: bearer codex-test-token"),
        "switching to a codex model must use the codex OAuth credential, not the stale anthropic \
         bearer token: {second}"
    );
    assert!(
        second.contains("chatgpt-account-id: acct-123"),
        "switching to a codex model must attach its own routing headers: {second}"
    );
    assert!(
        second.starts_with("post /openai-codex/backend-api/codex/responses"),
        "switching to a codex model must route to its own backend path, not the default dialect \
         path: {second}"
    );
}
