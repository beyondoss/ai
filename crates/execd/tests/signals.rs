//! A launcher that starts execd with `cmd &` or `nohup` leaves `SIGINT`, `SIGQUIT` and `SIGHUP`
//! ignored, and an ignored signal survives `exec`: every command and job inherited it, so a customer's
//! `Ctrl-C` to a job did nothing and a shell could not `trap` it. execd hands every command the
//! default disposition, whatever it was started with.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};
use support::Execd;
use support::http::post;

fn text(v: &Value) -> String {
    String::from_utf8(
        base64::engine::general_purpose::STANDARD
            .decode(v.as_str().unwrap_or(""))
            .unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn commands_get_default_dispositions_whatever_execd_was_launched_with() {
    let execd = Execd::start_ignoring("HUP INT QUIT USR1 USR2 TERM ALRM");
    let addr = std::net::SocketAddr::V4(execd.addr);
    // A one-shot that signals itself: with the signal ignored it would print `alive`.
    for sig in ["INT", "QUIT", "HUP", "USR1", "USR2", "TERM"] {
        let a = post(
            addr,
            &[],
            &json!({"command":"sh","args":["-c",format!("kill -{sig} $$; sleep 1; echo alive")],"timeout_ms":10000}),
        )
        .await;
        let v = a.json();
        assert!(
            !text(&v["stdout"]).contains("alive"),
            "SIG{sig} was ignored by a one-shot: {v}"
        );
    }
    // A job whose shell traps each signal prints it when the signal op reaches it: a trap on a signal
    // that was ignored on entry never runs.
    let script = ["INT", "QUIT", "HUP", "USR1", "USR2", "TERM"]
        .iter()
        .map(|s| format!("trap 'printf {s}.' {s}"))
        .collect::<Vec<_>>()
        .join("; ")
        + "; printf ready; while :; do (trap '' INT QUIT HUP USR1 USR2 TERM; sleep 0.1); done";
    let a = post(
        addr,
        &[],
        &json!({"command":"sh","args":["-c",script],
                "job":{"op":"start","scope":"s","id":"a1","hold":false}}),
    )
    .await;
    assert_eq!(a.status, 200);
    let read = || async {
        let a = post(
            addr,
            &[],
            &json!({"command":"true","job":{"op":"read","scope":"s","id":"a1","offset":0,"max":4096}}),
        )
        .await;
        text(&a.json()["data"])
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while read().await != "ready" {
        assert!(Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    let mut want = String::from("ready");
    for sig in ["INT", "QUIT", "HUP", "USR1", "USR2", "TERM"] {
        post(
            addr,
            &[],
            &json!({"command":"true","job":{"op":"signal","scope":"s","ids":["a1"],"signal":format!("SIG{sig}")}}),
        )
        .await;
        want.push_str(&format!("{sig}."));
        let deadline = Instant::now() + Duration::from_secs(5);
        while read().await != want {
            assert!(
                Instant::now() < deadline,
                "SIG{sig} never ran the job's trap: it printed {:?}, want {want:?}",
                read().await
            );
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }
    post(
        addr,
        &[],
        &json!({"command":"true","job":{"op":"kill","scope":"s","ids":["a1"],"remove":true,"grace_ms":200}}),
    )
    .await;
}
