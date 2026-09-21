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

use common::service::Service;
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
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;

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
