//! MCP OAuth e2e: `agent mcp-login`/`mcp-logout` and `tools::mcp::connect_http`'s OAuth-aware path,
//! driven against a real, hand-rolled OAuth-protected MCP server (metadata discovery, dynamic client
//! registration, an authorize endpoint, a token endpoint) — not a mock of the OAuth protocol. A split
//! sibling of `mcp_client.rs` (which covers the non-OAuth MCP client basics), matching this repo's own
//! "split e2e tests by domain into small files" convention.
//!
//! The "browser" step is simulated by a real, unauthenticated GET (via a small hand-rolled HTTP
//! client — no new `reqwest` feature needed just for tests) to the exact URL `mcp-login` prints,
//! following the fixture's real 302 redirect through to the real local callback listener `mcp-login`
//! itself is running — the whole authorization-code + PKCE + dynamic-client-registration dance runs
//! for real, only the human clicking "Allow" is skipped.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use common::mcp_oauth_fixture::{
    get_following_redirects, read_printed_url, spawn_oauth_protected_mcp_fixture,
    write_global_settings,
};
use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, turn_text,
    turn_tool_use,
};
use serde_json::json;

#[test]
fn mcp_login_completes_the_real_oauth_flow_and_the_token_authenticates_a_real_tool_call() {
    let home = tempfile::tempdir().unwrap();
    // A long expiry: this test asserts the *exact* token `mcp-login` obtained reaches the server, so
    // it must not be eagerly refreshed away before the follow-up `run` gets to use it.
    let (url, seen_auth_headers) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);

    // The "user visits the browser" step: a real GET, following the fixture's real 302 redirect
    // through to `mcp-login`'s own real local callback listener.
    let status = get_following_redirects(&auth_url, 3);
    assert_eq!(
        status, 200,
        "the callback must be delivered and answered 200"
    );

    let output = login.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "mcp-login failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("logged in: protected"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );

    // Now prove the persisted credential is actually usable: a real `run` invocation, real tool call,
    // through the real bearer token `mcp-login` just established.
    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__protected__echo",
        &json!({ "text": "oauth-authenticated-marker" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let cwd = tempfile::tempdir().unwrap();
    let run_output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .args([
            "run",
            "call the protected echo tool",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "6",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    assert!(
        run_output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[1].contains("oauth-authenticated-marker"),
        "the oauth-authenticated tool call must actually go through: {}",
        bodies[1]
    );
    let seen = seen_auth_headers.lock().unwrap();
    assert!(
        seen.iter().any(|h| h == "Bearer access-token-0"),
        "the mcp server must have received the exact bearer token mcp-login obtained: {seen:?}"
    );
}

#[test]
fn mcp_login_an_unknown_server_name_fails_clearly() {
    let home = tempfile::tempdir().unwrap();
    write_global_settings(home.path(), json!([]));
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "does-not-exist"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no MCP server named `does-not-exist`"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mcp_login_a_stdio_server_is_rejected_with_a_clear_error() {
    let home = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([{ "name": "local", "transport": "stdio", "command": "true", "args": [], "env": {} }]),
    );
    let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "local"])
        .env("HOME", home.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("stdio transport"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mcp_logout_removes_a_stored_credential_and_is_idempotent() {
    let home = tempfile::tempdir().unwrap();
    let (url, _seen) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    get_following_redirects(&auth_url, 3);
    let output = login.wait_with_output().unwrap();
    assert!(output.status.success());

    let logout = |home: &Path| -> String {
        let output = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
            .args(["mcp-logout", "protected"])
            .env("HOME", home)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    assert!(logout(home.path()).contains("logged out: protected"));
    // Idempotent: logging out again reports "not logged in", not an error.
    assert!(logout(home.path()).contains("not logged in: protected"));
}

#[test]
fn mcp_connect_to_an_oauth_protected_server_without_ever_logging_in_fails_naming_mcp_login() {
    let home = tempfile::tempdir().unwrap();
    let (url, _seen) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );
    let cwd = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hi")]);
    let output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .args([
            "run",
            "just say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    // Fail-soft still applies: the *run* succeeds even though this one MCP server never connected.
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mcp-login protected"),
        "an unauthenticated connect attempt to an oauth-protected server must hint at `agent \
         mcp-login`, not just fail silently: {stderr}"
    );
}

#[test]
fn mcp_get_access_token_transparently_refreshes_an_expired_token_on_the_next_connect() {
    let home = tempfile::tempdir().unwrap();
    // A short (1s) expiry: `rmcp`'s own `AuthorizationManager::get_access_token` refreshes proactively
    // whenever fewer than 30 seconds remain (`REFRESH_BUFFER_SECS`), so any token issued with a
    // sub-30s lifetime is *always* treated as due for refresh on the very next call — no need to
    // actually sleep past the expiry for this to trigger.
    let (url, seen_auth_headers) = spawn_oauth_protected_mcp_fixture(1);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    get_following_redirects(&auth_url, 3);
    assert!(login.wait_with_output().unwrap().status.success());
    assert!(
        seen_auth_headers.lock().unwrap().is_empty(),
        "sanity: mcp-login itself never calls tools/call, so no MCP request (and thus no \
         Authorization header) should have reached the server yet"
    );

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__protected__echo",
        &json!({ "text": "post-refresh-marker" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let cwd = tempfile::tempdir().unwrap();
    let run_output = run_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .env("HOME", home.path())
        .args([
            "run",
            "call echo",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "6",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    assert!(
        run_output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    assert!(bodies.lock().unwrap()[1].contains("post-refresh-marker"));

    let seen = seen_auth_headers.lock().unwrap();
    assert!(
        seen.iter().any(|h| h == "Bearer access-token-1"),
        "a genuinely refreshed (not replayed) token must reach the server: {seen:?}"
    );
}

#[test]
fn mcp_login_established_credential_is_honored_by_serve_too_not_just_run() {
    // `tools::mcp::connect_http`'s OAuth-aware path is the same code regardless of which of the two
    // call sites (`run_task`, `ServeConfig::mcp_tools`) invoked `connect_all` — this proves a login
    // established once via `agent mcp-login` (an interactive, CLI-only, `serve`-external command —
    // there's no RPC-triggerable equivalent from *inside* a live `serve` session, and a session already
    // running won't pick up a login established after its own startup either, since MCP servers connect
    // once and aren't reconnected mid-session) is actually honored on `serve`'s *next* startup, not
    // just `run`'s.
    let home = tempfile::tempdir().unwrap();
    let (url, seen_auth_headers) = spawn_oauth_protected_mcp_fixture(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );

    let mut login = Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    get_following_redirects(&auth_url, 3);
    assert!(login.wait_with_output().unwrap().status.success());

    let turn1 = turn_tool_use(
        "toolu_1",
        "mcp__protected__echo",
        &json!({ "text": "serve-oauth-marker" }).to_string(),
    );
    let (base, bodies) = spawn_model_server(vec![turn1, turn_text("done")]);
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let mut child = serve_cmd(env!("CARGO_BIN_EXE_beyond-ai-agent"), &base, &session_file)
        .env("HOME", home.path())
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    writeln!(
        stdin,
        "{}",
        json!({ "type": "prompt", "message": "call the protected echo tool" })
    )
    .unwrap();
    stdin.flush().unwrap();
    let frames = read_until_response(&mut stdout, "prompt");
    drop(stdin);
    child.wait().unwrap();

    let response = frames
        .iter()
        .find(|f| f["type"] == "response" && f["command"] == "prompt")
        .unwrap();
    assert_eq!(
        response["success"], true,
        "the prompt turn must succeed: {frames:?}"
    );
    assert!(bodies.lock().unwrap()[1].contains("serve-oauth-marker"));
    assert!(
        seen_auth_headers
            .lock()
            .unwrap()
            .iter()
            .any(|h| h == "Bearer access-token-0"),
        "the mcp-login-established token must authenticate the call made through serve"
    );
}
