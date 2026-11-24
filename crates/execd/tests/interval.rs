//! A request may name its own heartbeat: `Keepalive-Ping-Interval`, whole seconds from 1 to 240
//! (E2B's envd takes the same header). Anything else, or no header, runs at the server's beat. The
//! decision is `execd_wire::beat_ms` (W1b); this runs it through the real server.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::Duration;

use beyond_test_support::ports;
use serde_json::json;
use support::http::post;

/// The server's own beat: 20 times faster than the shortest a client may ask for.
const BEAT: Duration = Duration::from_millis(50);

async fn start() -> (execd::Handle, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut o = execd::Options::new(dir.path());
    o.heartbeat = BEAT;
    (execd::serve(ports::tokio_listener().await, o).unwrap(), dir)
}

/// The heartbeats (`{}`) a stream of a command that is silent for 1.5 s carries when it asks for
/// `interval` (None: no header).
async fn beats(server: &execd::Handle, interval: Option<&str>) -> usize {
    let headers: Vec<(&str, &str)> = interval
        .map(|v| vec![("Keepalive-Ping-Interval", v)])
        .unwrap_or_default();
    let a = post(
        server.local_addr(),
        &headers,
        &json!({ "command": "sh", "args": ["-c", "sleep 1.5"], "stream": true }),
    )
    .await;
    assert_eq!(a.status, 200);
    a.frames()
        .iter()
        .filter(|f| f.as_object().is_some_and(|o| o.is_empty()))
        .count()
}

/// A client's interval is its beat: one second is one heartbeat in a 1.5 s silence, where the
/// server's own would send about thirty. Red without the header read (`mut_interval_off` and
/// `mut_interval_unbounded` are red in the kernel's own tests).
#[tokio::test]
async fn a_client_chosen_interval_sets_the_streams_beat() {
    let (server, _dir) = start().await;
    let own = beats(&server, None).await;
    assert!(own >= 20, "the server's own beat sent {own}");
    let chosen = beats(&server, Some("1")).await;
    assert_eq!(chosen, 1, "one beat in 1.5 s at a 1 s interval");
}

/// Zero (there is no "off"), a negative, text, a float and a number over 240 all take the server's
/// beat: the stream is never left silent.
#[tokio::test]
async fn an_interval_that_is_not_one_to_240_seconds_is_the_default() {
    let (server, _dir) = start().await;
    for bad in ["0", "-1", "abc", "1.5", "", "241", "99999999999999999999"] {
        let n = beats(&server, Some(bad)).await;
        assert!(n >= 20, "{bad:?}: {n} beats, the default's about thirty");
    }
}

/// A one-shot's committed answer keeps the chosen beat too: headers at the first interval, a space at
/// each after, then the answer.
#[tokio::test]
async fn a_one_shot_keeps_the_chosen_beat() {
    let (server, _dir) = start().await;
    let a = post(
        server.local_addr(),
        &[("Keepalive-Ping-Interval", "1")],
        &json!({ "command": "sh", "args": ["-c", "sleep 2.5"] }),
    )
    .await;
    assert_eq!(a.status, 200);
    let spaces = a.body.iter().take_while(|b| **b == b' ').count();
    assert_eq!(
        spaces, 1,
        "headers at 1 s, one space at 2 s, the answer at 2.5 s"
    );
    assert_eq!(a.json()["exit_code"], 0);
}
