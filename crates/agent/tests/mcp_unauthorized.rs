//! What a tool call that a streamable-HTTP server answers 401 tells the model. Every POST goes
//! through `mcp_wire::HttpClient::post_bounded`, which decides a 401 from its status (so a login can
//! be refreshed) — and the server's own message must survive that: an "invalid API key" from a
//! server with a static key is the tool error, and a login the server keeps refusing says to run
//! `agent mcp-login`, on the tool call and not only at connect.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::atomic::Ordering;

use common::mcp_oauth_fixture::{
    OAuthFixture, echo, logged_in, run, tool_results, write_global_settings,
};
use common::turn_text;
use serde_json::json;

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
fn a_401_from_a_server_with_a_static_key_is_the_tool_error_with_the_servers_message() {
    let fixture = OAuthFixture::spawn(3600);
    fixture.issue("static-key");
    let home = tempfile::tempdir().unwrap();
    write_global_settings(
        home.path(),
        json!([{ "name": "protected", "transport": "http", "url": fixture.url,
                 "headers": { "Authorization": "Bearer static-key" } }]),
    );
    // The key works for one call, then the server stops accepting it — answering with a JSON-RPC
    // error that says why.
    fixture.reject_with_json_body.store(true, Ordering::SeqCst);
    fixture.revoke_after_calls.store(1, Ordering::SeqCst);
    let bodies = run(
        home.path(),
        vec![
            echo("toolu_1", "one"),
            echo("toolu_2", "two"),
            turn_text("done"),
        ],
    );
    let (first, failed) = result_of(&bodies, 1, "toolu_1");
    assert!(!failed && first.contains("one"), "{first}");
    let (text, is_error) = result_of(&bodies, 2, "toolu_2");
    assert!(is_error, "{text}");
    assert!(
        text.contains("unauthorized: token expired"),
        "the server's own message: {text}"
    );
    assert!(!text.contains("mcp-login"), "no login to run: {text}");
}

fn refused_login(json_body: bool) -> String {
    let (home, fixture) = logged_in();
    fixture
        .reject_with_json_body
        .store(json_body, Ordering::SeqCst);
    fixture.reject_calls.store(true, Ordering::SeqCst);
    let bodies = run(home.path(), vec![echo("toolu_1", "one"), turn_text("done")]);
    assert_eq!(
        fixture.refresh_grants.load(Ordering::SeqCst),
        1,
        "the 401 was refreshed once, and the retry refused too"
    );
    let (text, is_error) = result_of(&bodies, 1, "toolu_1");
    assert!(is_error, "{text}");
    assert!(
        text.contains("agent mcp-login protected"),
        "the tool error says how to fix it: {text}"
    );
    text
}

#[test]
fn a_login_the_server_keeps_refusing_names_mcp_login_on_the_tool_error_and_keeps_its_message() {
    let text = refused_login(true);
    assert!(
        text.contains("unauthorized: token expired"),
        "the server's own message: {text}"
    );
}

#[test]
fn a_login_refused_with_a_bare_challenge_names_mcp_login_on_the_tool_error() {
    refused_login(false);
}
