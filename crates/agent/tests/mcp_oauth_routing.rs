//! An MCP server with an `agent mcp-login` is spoken to over this crate's own POST path
//! (`mcp_wire::HttpClient::post_bounded`, which answers every streamable-HTTP POST) instead of
//! rmcp's client — so that path has to carry everything a streamable-HTTP server does: sessions
//! (`Mcp-Session-Id`, a 404 for an expired one), SSE responses, the standalone `GET` stream, the
//! legacy handshake's cues. Each test here drives a real dial (`agent run`) against the OAuth fixture
//! speaking that feature, and pins one piece of the routing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::atomic::Ordering;

use common::mcp_oauth_fixture::{echo, logged_in, run, tool_results};
use common::turn_text;

/// `toolu_<n>`'s result in the model request that followed it: `(text, is_error)`.
fn result_of(bodies: &[String], request: usize, id: &str) -> (String, bool) {
    let results = tool_results(&bodies[request]);
    let (_, text, is_error) = results
        .into_iter()
        .find(|(i, ..)| i == id)
        .unwrap_or_else(|| panic!("no result for {id} in request {request}"));
    (text, is_error)
}

#[test]
fn a_401_carrying_a_json_rpc_error_body_refreshes_through_a_real_dial() {
    // rmcp's own client reads this 401 as an ordinary error *response*; only `post_bounded`
    // (every POST's path) sees the status, so the OAuth layer refreshes.
    let (home, fixture) = logged_in();
    fixture.reject_with_json_body.store(true, Ordering::SeqCst);
    fixture.revoke_after_calls.store(1, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first"),
            echo("toolu_2", "after-revocation"),
            turn_text("done"),
        ],
    );
    let (text, is_error) = result_of(&bodies, 2, "toolu_2");
    assert!(
        !is_error && text.contains("after-revocation"),
        "the call after the revocation must succeed on a refreshed token: {text}"
    );
    assert_eq!(fixture.rejected_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.refresh_grants.load(Ordering::SeqCst), 1);
}

#[test]
fn a_legacy_servers_4xx_to_server_discover_falls_back_to_initialize() {
    // A pre-2026-07-28 server may answer the discovery probe with a plain 400; rmcp reads that as
    // "no discover, use initialize" — so must the OAuth path, or the handshake fails.
    let (home, fixture) = logged_in();
    *fixture.discover_reply.lock().unwrap() =
        Some(("400 Bad Request".into(), "unknown method".into()));
    let bodies = run(
        home.path(),
        vec![echo("toolu_1", "legacy-handshake"), turn_text("done")],
    );
    let (text, is_error) = result_of(&bodies, 1, "toolu_1");
    assert!(
        !is_error && text.contains("legacy-handshake"),
        "the server must be connected through the legacy handshake: {text}"
    );
    assert_eq!(fixture.initializes.load(Ordering::SeqCst), 1);
}

#[test]
fn an_empty_200_to_a_notification_is_an_acceptance() {
    // `notifications/initialized` answered by an empty `200 OK` (no content type) rather than 202:
    // rmcp accepts it, and so must the OAuth path, or the handshake fails.
    let (home, fixture) = logged_in();
    fixture
        .notifications_empty_200
        .store(true, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![echo("toolu_1", "empty-200"), turn_text("done")],
    );
    let (text, is_error) = result_of(&bodies, 1, "toolu_1");
    assert!(
        !is_error && text.contains("empty-200"),
        "the server must be connected: {text}"
    );
}

#[test]
fn sessions_sse_responses_and_the_standalone_stream_work_for_an_oauth_server() {
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    fixture.sse_responses.store(true, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "over-sse-1"),
            echo("toolu_2", "over-sse-2"),
            turn_text("done"),
        ],
    );
    for (request, id, text) in [(1, "toolu_1", "over-sse-1"), (2, "toolu_2", "over-sse-2")] {
        let (got, is_error) = result_of(&bodies, request, id);
        assert!(!is_error && got.contains(text), "{id}: {got}");
    }
    assert_eq!(
        fixture.session_refusals.load(Ordering::SeqCst),
        0,
        "every POST after initialize carried the live session"
    );
    let sessions = fixture.call_sessions.lock().unwrap().clone();
    assert_eq!(
        sessions,
        [Some("session-0".to_owned()), Some("session-0".to_owned())]
    );
    assert!(
        fixture.get_streams.load(Ordering::SeqCst) >= 1,
        "the standalone SSE stream was opened, with the bearer and the session"
    );
}

#[test]
fn an_expired_session_is_reinitialized_and_the_call_retried() {
    // The 404 for an expired session is rmcp's `SessionExpired`: it re-initializes and retries.
    let (home, fixture) = logged_in();
    fixture.sessions.store(true, Ordering::SeqCst);
    fixture
        .expire_sessions_after_calls
        .store(1, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "first"),
            echo("toolu_2", "after-expiry"),
            turn_text("done"),
        ],
    );
    let (text, is_error) = result_of(&bodies, 2, "toolu_2");
    assert!(
        !is_error && text.contains("after-expiry"),
        "the call after the session expired must succeed on a new session: {text}"
    );
    assert_eq!(fixture.initializes.load(Ordering::SeqCst), 2);
    assert!(fixture.session_refusals.load(Ordering::SeqCst) >= 1);
    let sessions = fixture.call_sessions.lock().unwrap().clone();
    assert_eq!(
        sessions.last().cloned().flatten().as_deref(),
        Some("session-1")
    );
}
