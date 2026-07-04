//! End-to-end proof that a stored OAuth credential actually drives a real `run` invocation: no
//! `--key`/`AI_AGENT_KEY` at all, just a seeded `~/.claude/auth.json` and an Anthropic-dialect
//! `--model` — `resolve_gateway_credential` must infer the provider, `OAuthCredentialSource` must
//! hand a live token to `GatewayClient`, and the request that reaches the wire must carry Anthropic's
//! OAuth identity headers (the full chain `client.rs`'s own unit tests exercise with a hand-built test
//! double, exercised here instead through every real layer).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::Stdio;

use common::{run_cmd, spawn_model_server, turn_text};

fn write_auth_json(home: &std::path::Path, body: &str) {
    let dir = home.join(".claude");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("auth.json"), body).unwrap();
}

#[test]
fn a_stored_anthropic_credential_is_used_with_no_key_flag_and_carries_the_oauth_headers() {
    let home = tempfile::tempdir().unwrap();
    write_auth_json(
        home.path(),
        r#"{"anthropic":{"type":"oauth","access":"test-oauth-access-token","refresh":"test-refresh","expires":4000000000000}}"#,
    );

    let (base, bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .env("HOME", home.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let recorded = bodies.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    let request = recorded[0].to_lowercase();
    assert!(
        request.contains("authorization: bearer test-oauth-access-token"),
        "must send the stored OAuth access token, not fail with \"no gateway key\": {request}"
    );
    assert!(
        request.contains("anthropic-beta: claude-code-20250219,oauth-2025-04-20"),
        "{request}"
    );
    assert!(request.contains("user-agent: claude-cli/"), "{request}");
    assert!(request.contains("x-app: cli"), "{request}");
    assert!(
        request.contains("anthropic-dangerous-direct-browser-access: true"),
        "{request}"
    );
}

#[test]
fn no_key_and_no_stored_credential_is_a_clean_error_naming_agent_login() {
    let home = tempfile::tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");

    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            "http://127.0.0.1:1",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .env("HOME", home.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("agent login"), "{stderr}");
}
