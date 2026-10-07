//! Integration test for W3's shared upstream client (`serve --listen --upstream-http2 <mode>`). With
//! a shared `reqwest::Client`, every daemon session drives its prompts through *one* connection pool
//! instead of opening its own. The mock gateway (`common::spawn_model_server`) speaks HTTP/1.1, so
//! `--upstream-http2 auto` (no prior-knowledge) pools over h1 and requests succeed — exactly what a
//! real deployment does until the gateway gains h2c. On-the-wire h2c multiplexing (one TCP connection
//! for many sessions) is asserted later by a live smoke against the h2c gateway, out of scope here;
//! this proves the shared client is wired end-to-end and two concurrent sessions both complete through
//! it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};

use common::{
    ChildGuard, ISOLATED_HOME, spawn_listening, spawn_model_server, turn_text, ws_connect,
    ws_read_until_response, ws_send,
};
use serde_json::json;

/// Spawn a `serve --listen` child with a shared upstream client in `--upstream-http2 <mode>`.
fn serve_ws_child_shared(base: &str, session_dir: &str, http2_mode: &str) -> (ChildGuard, u16) {
    spawn_listening(
        Command::new(env!("CARGO_BIN_EXE_beyond-ai-agent"))
            .args([
                "serve",
                "--upstream-http2",
                http2_mode,
                "--gateway-url",
                base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--session-dir",
                session_dir,
            ])
            .env("HOME", ISOLATED_HOME)
            .stdin(Stdio::null())
            .stdout(Stdio::null()),
    )
}

/// Two distinct sessions, driven concurrently through the one shared `--upstream-http2 auto` client,
/// each completing a `prompt` (`response{success:true}`). The mock gateway is h1, so `auto` pools over
/// HTTP/1.1 and both succeed — proving the shared client is injected into every session's gateway
/// client rather than each session building (and tearing down) its own.
#[tokio::test]
async fn shared_client_two_concurrent_sessions_both_complete() {
    // Two prompts across the two sessions → two upstream requests → two turns (global order).
    let (base, _requests) =
        spawn_model_server(vec![turn_text("alpha reply"), turn_text("bravo reply")]);
    let dir = tempfile::tempdir().unwrap();
    let (mut child, port) = serve_ws_child_shared(&base, dir.path().to_str().unwrap(), "auto");

    let mut a = ws_connect(port, Some("alpha")).await;
    let mut b = ws_connect(port, Some("bravo")).await;

    ws_send(
        &mut a,
        json!({ "type": "prompt", "id": "pa", "message": "hi from alpha" }),
    )
    .await;
    ws_send(
        &mut b,
        json!({ "type": "prompt", "id": "pb", "message": "hi from bravo" }),
    )
    .await;

    // Both prompts complete through the single shared connection pool.
    let fa = ws_read_until_response(&mut a, "prompt").await;
    let fb = ws_read_until_response(&mut b, "prompt").await;
    assert_eq!(
        fa.last().unwrap()["success"],
        true,
        "alpha's prompt should succeed through the shared client: {:?}",
        fa.last()
    );
    assert_eq!(
        fb.last().unwrap()["success"],
        true,
        "bravo's prompt should succeed through the shared client: {:?}",
        fb.last()
    );

    let _ = child.kill();
    let _ = child.wait();
}
