//! Mid-session MCP OAuth refresh (`tools::mcp_oauth`): a server that starts answering 401 while a
//! session is running — its token revoked or expired on the server's side — gets one refresh,
//! through the `agent mcp-login` store, and the request is retried once. Driven against the real
//! OAuth-protected fixture (`common::mcp_oauth_fixture`), the token going bad between two tool calls
//! of one `run`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::atomic::Ordering;

use common::mcp_oauth_fixture::{echo, echoes, logged_in, run, tool_results};
use common::turn_text;
use serde_json::{Value, json};

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

/// A refresh the token endpoint answers with `(status, body)`: the call fails naming `agent
/// mcp-login` — a permanent refusal, not something to back off and retry — and the endpoint is
/// asked exactly once for the rejected token.
fn a_permanent_refresh_refusal(status: &str, body: Value) {
    let (home, fixture) = logged_in();
    fixture.revoke_after_calls.store(1, Ordering::SeqCst);
    *fixture.refresh_reply.lock().unwrap() = Some((status.to_owned(), body.to_string()));
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first-call"),
            echo("toolu_2", "after-revocation"),
            turn_text("done"),
        ],
    );
    let results = tool_results(&bodies[2]);
    let (_, text, is_error) = results.iter().find(|(i, ..)| i == "toolu_2").unwrap();
    assert!(
        *is_error && text.contains("agent mcp-login protected"),
        "{status} {body}: a permanent refusal must tell the user to log in again: {text}"
    );
    assert_eq!(fixture.refresh_grants.load(Ordering::SeqCst), 1);
}

#[test]
fn every_rfc_6749_refusal_is_definitive_and_names_mcp_login() {
    // RFC 6749 §5.2's codes (`invalid_grant` is the test above); `invalid_client` comes as a 401.
    for (status, code) in [
        ("401 Unauthorized", "invalid_client"),
        ("400 Bad Request", "unauthorized_client"),
        ("400 Bad Request", "unsupported_grant_type"),
        ("400 Bad Request", "invalid_scope"),
        ("400 Bad Request", "invalid_request"),
    ] {
        a_permanent_refresh_refusal(status, json!({ "error": code }));
    }
}

#[test]
fn a_token_endpoint_that_is_not_there_is_definitive_and_names_mcp_login() {
    a_permanent_refresh_refusal("404 Not Found", json!({ "message": "no such route" }));
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
