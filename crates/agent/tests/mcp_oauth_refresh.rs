//! Mid-session MCP OAuth refresh (`tools::mcp_oauth`): a server that starts answering 401 while a
//! session is running — its token revoked or expired on the server's side — gets one refresh,
//! through the `agent mcp-login` store, and the request is retried once. Driven against the real
//! OAuth-protected fixture (`common::mcp_oauth_fixture`), the token going bad between two tool calls
//! of one `run`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::atomic::Ordering;

use common::mcp_oauth_fixture::{OAuthFixture, mcp_login, write_global_settings};
use common::{run_cmd, spawn_model_server, sse, turn_text, turn_tool_use};
use serde_json::{Value, json};

/// A logged-in `$HOME` for a fresh fixture whose tokens stay valid for an hour unless revoked.
fn logged_in() -> (tempfile::TempDir, OAuthFixture) {
    let home = tempfile::tempdir().unwrap();
    let fixture = OAuthFixture::spawn(3600);
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": fixture.url, "headers": {} }]),
    );
    mcp_login(home.path(), "protected");
    (home, fixture)
}

fn echo(id: &str, text: &str) -> String {
    turn_tool_use(
        id,
        "mcp__protected__echo",
        &json!({ "text": text }).to_string(),
    )
}

/// One assistant turn calling `echo` once per text, all at once.
fn echoes(texts: &[&str]) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
    ];
    for (i, text) in texts.iter().enumerate() {
        events.push(json!({ "type": "content_block_start", "index": i, "content_block": { "type": "tool_use", "id": format!("toolu_p{i}"), "name": "mcp__protected__echo", "input": {} } }));
        events.push(json!({ "type": "content_block_delta", "index": i, "delta": { "type": "input_json_delta", "partial_json": json!({ "text": text }).to_string() } }));
        events.push(json!({ "type": "content_block_stop", "index": i }));
    }
    events.push(json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }));
    events.push(json!({ "type": "message_stop" }));
    sse(&events)
}

/// Run one `agent run` against `home` with the scripted model turns; the model request bodies.
fn run(home: &std::path::Path, turns: Vec<String>) -> Vec<String> {
    let (base, bodies) = spawn_model_server(turns);
    let cwd = tempfile::tempdir().unwrap();
    let out = run_cmd(common::BIN)
        .env("HOME", home)
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
            "8",
            "--no-session-persistence",
        ])
        .current_dir(cwd.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    bodies.lock().unwrap().clone()
}

/// The tool results in a recorded model request (headers + body): `(tool_use_id, text, is_error)`.
fn tool_results(request: &str) -> Vec<(String, String, bool)> {
    let body = request.split_once("\r\n\r\n").map_or(request, |(_, b)| b);
    let v: Value = serde_json::from_str(body).unwrap();
    let mut out = Vec::new();
    for m in v["messages"].as_array().unwrap() {
        for b in m["content"].as_array().into_iter().flatten() {
            if b["type"] == "tool_result" {
                let text = match &b["content"] {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                out.push((
                    b["tool_use_id"].as_str().unwrap_or_default().to_owned(),
                    text,
                    b["is_error"].as_bool().unwrap_or(false),
                ));
            }
        }
    }
    out
}

fn stored_access_token(home: &std::path::Path) -> String {
    let store: Value =
        serde_json::from_str(&std::fs::read_to_string(home.join(".claude/mcp_auth.json")).unwrap())
            .unwrap();
    store["protected"]["token_response"]["access_token"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn a_token_revoked_mid_session_is_refreshed_once_and_the_call_succeeds() {
    let (home, fixture) = logged_in();
    assert_eq!(stored_access_token(home.path()), "access-token-0");
    // The first call succeeds; then every token the fixture issued is revoked.
    fixture.revoke_after_calls.store(1, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first-call"),
            echo("toolu_2", "after-revocation"),
            turn_text("done"),
        ],
    );
    let results = tool_results(&bodies[2]);
    let second = results.iter().find(|(id, ..)| id == "toolu_2").unwrap();
    assert!(
        !second.2 && second.1.contains("after-revocation"),
        "the call after the revocation must go through on a refreshed token: {second:?}"
    );
    assert_eq!(fixture.rejected_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.refresh_grants.load(Ordering::SeqCst), 1);
    let seen = fixture.seen_auth_headers.lock().unwrap().clone();
    assert_eq!(
        seen.last().map(String::as_str),
        Some("Bearer access-token-1"),
        "the retry carries the refreshed token: {seen:?}"
    );
    assert_eq!(
        stored_access_token(home.path()),
        "access-token-1",
        "the refreshed token is persisted for the next process"
    );
}

#[test]
fn a_refresh_that_fails_names_mcp_login_and_is_not_retried() {
    let (home, fixture) = logged_in();
    fixture.revoke_after_calls.store(1, Ordering::SeqCst);
    fixture.fail_refresh.store(true, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first-call"),
            echo("toolu_2", "after-revocation"),
            echo("toolu_3", "again"),
            turn_text("done"),
        ],
    );
    for (body, id) in [(&bodies[2], "toolu_2"), (&bodies[3], "toolu_3")] {
        let results = tool_results(body);
        let (_, text, is_error) = results.iter().find(|(i, ..)| i == id).unwrap();
        assert!(
            *is_error && text.contains("agent mcp-login protected"),
            "a failed refresh must tell the user to log in again: {text}"
        );
    }
    assert_eq!(
        fixture.refresh_grants.load(Ordering::SeqCst),
        1,
        "one refresh attempt for the rejected token — no loop, and not again for the next call"
    );
}

#[test]
fn concurrent_rejections_refresh_once() {
    let (home, fixture) = logged_in();
    fixture.revoke_after_calls.store(1, Ordering::SeqCst);
    // Hold each rejected call until all three are in flight with the revoked token.
    fixture.hold_rejections_until.store(3, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first-call"),
            echoes(&["p-a", "p-b", "p-c"]),
            turn_text("done"),
        ],
    );
    let results = tool_results(&bodies[2]);
    for text in ["p-a", "p-b", "p-c"] {
        assert!(
            results.iter().any(|(_, t, e)| !e && t.contains(text)),
            "every concurrent call must succeed after the refresh: {results:?}"
        );
    }
    assert_eq!(
        fixture.rejected_calls.load(Ordering::SeqCst),
        3,
        "all three calls were in flight with the revoked token"
    );
    assert_eq!(
        fixture.refresh_grants.load(Ordering::SeqCst),
        1,
        "three concurrent 401s cost one refresh"
    );
}
