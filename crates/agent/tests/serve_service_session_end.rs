//! What a client is told when its session ends underneath it.
//!
//! A `serve --service` session can end on its own, before or after it goes live: its sandbox is
//! unreachable, its sealed storage will not open with the key in the grant, an internal error ends
//! the loop. The replica logs why and broadcasts an `error` frame — and for a long time that was all
//! it did. The socket stayed open, attached to a session that no longer existed.
//!
//! That is worse than it sounds, because the connection's *only* liveness check was its own next
//! `send` into the session's input channel. A client that has asked a question and is waiting for the
//! answer never sends anything else, so it never trips the check: it holds a live, silent socket for
//! as long as it is willing to wait, and the replica holds the connection task and its buffers for
//! exactly as long. The fleet simulator found this the hard way — a scenario that presented a wrong
//! data key hung indefinitely rather than failing, and the run that was supposed to prove tenant
//! isolation instead proved that a refusal is not a refusal if nobody is told.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::service::{Options, Service};
use common::{TestWs, spawn_model_server, ws_connect_with_headers, ws_send};
use futures::StreamExt as _;
use serde_json::{Value, json};

/// Read until the socket closes, or give up and hand back what was seen.
///
/// `Err` is the regression: a socket still open long after the session behind it ended.
async fn read_until_closed(ws: &mut TestWs, within: Duration) -> Result<Vec<Value>, Vec<Value>> {
    let deadline = tokio::time::Instant::now() + within;
    let mut seen = Vec::new();
    loop {
        match tokio::time::timeout_at(deadline, ws.next()).await {
            Err(_) => return Err(seen),
            // Closed cleanly, or the peer went away — either way the client is no longer waiting on
            // something that will never come, which is the property under test.
            Ok(None) | Ok(Some(Err(_))) => return Ok(seen),
            Ok(Some(Ok(msg))) => {
                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg
                    && let Ok(v) = serde_json::from_str::<Value>(text.as_str())
                {
                    seen.push(v);
                }
            }
        }
    }
}

/// A session whose sandbox isn't there fails during startup. The client must learn that from the
/// socket, not from a timeout it chose itself.
#[tokio::test]
async fn a_session_that_fails_to_start_closes_the_socket_instead_of_going_quiet() {
    failed_start_is_reported_before_the_close(&[]).await;
}

/// The same failure with the reason late: the session's loop has ended (its input is gone) but the
/// error that says why is broadcast a second later. The connection must still deliver it before it
/// closes — it waits for the session task to finish, not just for its input to drop. Flushing at the
/// first sign raced the broadcast, and under load the client saw a bare close and no reason.
#[tokio::test]
async fn the_reason_reaches_the_client_even_when_it_is_broadcast_after_the_loop_ends() {
    failed_start_is_reported_before_the_close(&[(
        "BEYOND_AI_AGENT_TEST_SLOW_SESSION_END_MS",
        "1000",
    )])
    .await;
}

/// The same failure with the connection late: the session fails and says why before this
/// connection's output is even registered (its upgrade was answered while the session was starting).
/// The reason is kept and handed to a connection that registers afterwards, so it is still told why.
#[tokio::test]
async fn the_reason_reaches_a_connection_that_registers_after_the_session_ended() {
    failed_start_is_reported_before_the_close(&[("BEYOND_AI_AGENT_TEST_SLOW_ATTACH_MS", "1500")])
        .await;
}

async fn failed_start_is_reported_before_the_close(env: &[(&str, &str)]) {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            env: env
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
            ..Default::default()
        },
    )
    .await;

    // Port 1 is not an exec endpoint. The upgrade still succeeds — it is answered once the session's
    // storage lock is held, before the body runs — so the client is attached to a session that is
    // about to end.
    let token = svc.token_with("t1", "s1.no-sandbox", |c| {
        c.exec_url = "http://127.0.0.1:1/".into();
    });
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.no-sandbox"), &svc.header(&token))
        .await
        .expect("the upgrade is answered before the session body runs");

    // The shape that matters: ask, then wait. Nothing else is ever sent on this socket, so a
    // connection that only notices a dead session when it next writes never notices at all.
    ws_send(&mut ws, json!({"type": "get_messages", "id": "q"})).await;

    let frames = read_until_closed(&mut ws, Duration::from_secs(30))
        .await
        .unwrap_or_else(|seen| {
            panic!(
                "the socket was still open 30s after the session ended, with no answer to \
             `get_messages`; saw {seen:#?}"
            )
        });
    assert!(
        frames.iter().any(|f| f["type"] == "error"),
        "the client must be told why it is being closed, not merely disconnected: {frames:#?}"
    );
}
