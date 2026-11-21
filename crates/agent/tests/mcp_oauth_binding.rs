//! MCP OAuth credentials belong to one server, and `mcp-login` follows the authorization spec's
//! discovery rules: a login is sent only to the URL it was made for; PKCE must be advertised;
//! the 401 challenge's `resource_metadata` and `scope` are used; endpoints are never guessed.
//! Driven against the real OAuth fixture (`common::mcp_oauth_fixture`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::io::BufReader;
use std::process::Stdio;
use std::sync::atomic::Ordering;

use common::SpawnGuarded;
use common::mcp_oauth_fixture::{
    OAuthFixture, echo, logged_in, read_printed_url, run, tool_results, write_global_settings,
};
use common::turn_text;
use serde_json::{Value, json};

fn point_protected_at(home: &std::path::Path, url: &str) {
    write_global_settings(
        home,
        json!([{ "name": "protected", "transport": "http", "url": url, "headers": {} }]),
    );
}

/// A server on another URL that would accept the token the first one issued, were it sent.
fn lookalike() -> OAuthFixture {
    let other = OAuthFixture::spawn(3600);
    other.issue("access-token-0");
    other
}

fn call_echo(home: &std::path::Path) {
    let bodies = run(home, vec![echo("toolu_1", "bound-call"), turn_text("done")]);
    let results = tool_results(&bodies[1]);
    assert_eq!(results.len(), 1, "{results:?}");
}

#[test]
fn a_login_is_not_sent_to_another_url_under_the_same_name() {
    let (home, a) = logged_in();
    // A project's `.mcp.json` redefines the same name at another server.
    let b = lookalike();
    point_protected_at(home.path(), &b.url);
    call_echo(home.path());
    assert!(
        b.seen_auth_headers.lock().unwrap().is_empty(),
        "the login made for another server must not reach this one: {:?}",
        b.seen_auth_headers.lock().unwrap()
    );
    // Back at the server it was made for, the same login is sent.
    point_protected_at(home.path(), &a.url);
    call_echo(home.path());
    assert!(
        a.seen_auth_headers
            .lock()
            .unwrap()
            .iter()
            .any(|h| h == "Bearer access-token-0")
    );
}

#[test]
fn a_stored_login_with_no_resource_is_not_sent() {
    let (home, a) = logged_in();
    let path = home.path().join(".beyond/mcp_auth.json");
    let mut store: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    store["protected"]
        .as_object_mut()
        .unwrap()
        .remove("resource");
    std::fs::write(&path, store.to_string()).unwrap();
    let before = a.seen_auth_headers.lock().unwrap().len();
    call_echo(home.path());
    let seen = a.seen_auth_headers.lock().unwrap().clone();
    assert!(
        !seen[before..].iter().any(|h| h.starts_with("Bearer ")),
        "an unbound login must be refused, not sent: {seen:?}"
    );
}

/// `agent mcp-login protected`, which must stop before printing an authorization URL: its stderr.
fn login_refused(home: &std::path::Path) -> String {
    let out = common::agent_cmd()
        .args(["mcp-login", "protected"])
        .env("HOME", home)
        .output()
        .unwrap();
    assert!(!out.status.success(), "login must be refused");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn logged_out_home(fixture: &OAuthFixture) -> tempfile::TempDir {
    let home = tempfile::tempdir().unwrap();
    point_protected_at(home.path(), &fixture.url);
    home
}

#[test]
fn login_is_refused_when_the_authorization_server_omits_pkce() {
    let f = OAuthFixture::spawn(3600);
    f.omit_pkce.store(true, Ordering::SeqCst);
    let home = logged_out_home(&f);
    let err = login_refused(home.path());
    assert!(err.contains("PKCE"), "{err}");
}

#[test]
fn login_is_refused_when_discovery_finds_nothing_rather_than_guessing_endpoints() {
    let f = OAuthFixture::spawn(3600);
    *f.prm_only_at.lock().unwrap() = Some("/.well-known/oauth-protected-resource/none".into());
    f.no_as_metadata.store(true, Ordering::SeqCst);
    let home = logged_out_home(&f);
    let err = login_refused(home.path());
    assert!(err.contains("not guessing endpoints"), "{err}");
}

#[test]
fn login_follows_the_challenges_resource_metadata_and_scope() {
    let f = OAuthFixture::spawn(3600);
    let base = f.url.trim_end_matches("/mcp").to_owned();
    let prm = "/.well-known/oauth-protected-resource/elsewhere";
    *f.prm_only_at.lock().unwrap() = Some(prm.into());
    *f.challenge_extra.lock().unwrap() = Some(format!(
        ", resource_metadata=\"{base}{prm}\", scope=\"files:read files:write\""
    ));
    let home = logged_out_home(&f);
    let mut login = common::agent_cmd()
        .args(["mcp-login", "protected"])
        .env("HOME", home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn_guarded();
    let mut stderr = BufReader::new(login.stderr.take().unwrap());
    let auth_url = read_printed_url(&mut stderr);
    let scope = url::Url::parse(&auth_url)
        .unwrap()
        .query_pairs()
        .find(|(k, _)| k == "scope")
        .map(|(_, v)| v.into_owned())
        .unwrap_or_default();
    assert!(
        scope.starts_with("files:read files:write"),
        "the challenge's scope comes first: {auth_url}"
    );
    assert_eq!(
        common::mcp_oauth_fixture::get_following_redirects(&auth_url, 3),
        200
    );
    assert!(login.wait_with_output().unwrap().status.success());
}
