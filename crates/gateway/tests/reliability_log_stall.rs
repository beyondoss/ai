//! Reliability: a log pipeline that stops draining stdout must not stall request handling (D263).
//!
//! Run via `mise run test:integration:rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::*;

/// With stdout never read, the pipe fills within a few hundred log lines. Diagnostics used to be
/// written with a blocking `write(2)` on the Tokio worker that emitted them, so from then on every
/// worker parked in the write and every request hung with its tenant slot held. Now they go
/// through a bounded, lossy queue on their own thread: requests keep being answered, and the lines
/// the stalled pipe cannot take are counted on `ai_log_dropped_total`.
///
/// `AI_LOG=debug` makes every request write several diagnostic lines (pingora's own, through
/// `LogTracer`), and each is a rejection (unknown provider), which writes no `ai.usage` row — those
/// keep their blocking, lossless writer by design.
/// defect: D263
#[tokio::test]
async fn a_stalled_stdout_does_not_stall_requests() {
    let (pubkey, _) = test_keypair(1);
    let gw = Gateway::builder(
        unused_nats_port(),
        &GatewayBuilder::dead_authority(),
        &b64(&pubkey),
    )
    .worker_threads(2)
    .env("AI_LOG", "debug")
    .stall_stdout()
    .start()
    .await;
    reject_until_logs_drop(&gw, true).await;
}

/// Send rounds of 16 concurrent rejections, each of which must be answered within 5 s, until the
/// stdout pipe and the diagnostic queue behind it are both full (`ai_log_dropped_total` moves).
/// With `then_as_many_again`, keep going for as many rounds again once lines are being dropped.
async fn reject_until_logs_drop(gw: &Gateway, then_as_many_again: bool) {
    let client = test_client();
    let url = format!("{}/no-such-provider/v1/chat/completions", gw.url());
    let mut full_at = None;
    for round in 0..2000 {
        let mut sends = tokio::task::JoinSet::new();
        for _ in 0..16 {
            let req = client
                .post(&url)
                .header("content-type", "application/json")
                .body(r#"{"model":"x","messages":[]}"#)
                .timeout(Duration::from_secs(5))
                .send();
            sends.spawn(async move { req.await.map(|r| r.status().as_u16()) });
        }
        while let Some(result) = sends.join_next().await {
            match result.expect("task") {
                Ok(status) => assert_eq!(status, 404, "round {round}"),
                Err(e) => panic!(
                    "round {round}: a request stalled behind the undrained stdout: {e}\nstderr:\n{}",
                    gw.log()
                ),
            }
        }
        match full_at {
            None if round % 10 == 9 => {
                let dropped = parse_metric(&gw.metrics().await, "ai_log_dropped_total", "");
                if dropped > 0.0 {
                    if !then_as_many_again {
                        return;
                    }
                    full_at = Some(round);
                }
            }
            Some(at) if round >= 2 * at + 10 => return,
            _ => {}
        }
    }
    panic!(
        "the diagnostic sink never dropped a line, so the stall never reached the queue's bound \
         (full_at = {full_at:?})"
    );
}

/// A rejection flood writes a capped number of `request rejected` lines a second, not one per
/// request, and every line left out is counted on a later line's `suppressed`.
/// defect: D263
#[tokio::test]
async fn a_rejection_flood_logs_a_capped_number_of_lines() {
    let (pubkey, _) = test_keypair(2);
    let gw = Gateway::builder(
        unused_nats_port(),
        &GatewayBuilder::dead_authority(),
        &b64(&pubkey),
    )
    .start()
    .await;
    let client = test_client();
    let url = format!("{}/no-such-provider/v1/chat/completions", gw.url());
    let reject = || async {
        let resp = client.post(&url).body("{}").send().await.unwrap();
        assert_eq!(resp.status(), 404);
    };
    let flood: u64 = 600;
    let started = std::time::Instant::now();
    for _ in 0..flood {
        reject().await;
    }
    let seconds = started.elapsed().as_secs() + 1;
    // A fresh second, so this line is admitted and reports what the flood left out.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    reject().await;
    // Wait until every rejection is accounted for: lines are written off the worker thread.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let (logged, suppressed) = loop {
        let log = gw.log();
        let rows: Vec<serde_json::Value> = log
            .lines()
            .filter(|l| l.contains("request rejected"))
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let suppressed: u64 = rows
            .iter()
            .filter_map(|v| v["fields"]["suppressed"].as_u64())
            .sum();
        let logged = rows.len() as u64;
        if logged + suppressed >= flood + 1 || std::time::Instant::now() > deadline {
            break (logged, suppressed);
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert!(
        logged <= 100 * (seconds + 1),
        "{logged} rejection lines for {flood} rejections in ~{seconds}s: not capped"
    );
    assert!(
        suppressed > 0,
        "{flood} rejections in ~{seconds}s suppressed none"
    );
    assert_eq!(
        logged + suppressed,
        flood + 1,
        "every rejection is either logged or counted on a later line"
    );
}

/// The diagnostic sink is drained at shutdown: the drain's own `exiting` line, written just before
/// `exit`, reaches stdout rather than dying in the queue.
/// defect: D263
#[tokio::test]
async fn shutdown_flushes_the_diagnostic_sink() {
    let (pubkey, _) = test_keypair(3);
    let gw = Gateway::builder(
        unused_nats_port(),
        &GatewayBuilder::dead_authority(),
        &b64(&pubkey),
    )
    .env("AI_LOG", "info")
    .start()
    .await;
    gw.sigterm();
    gw.wait_for_log_line(&["drained: no request in flight; exiting"])
        .await;
}

/// A wedged stdout cannot hold shutdown up: the flush of the queued log lines gives up at its
/// bound, says so on stderr (the one stream left), and the process exits.
/// defect: D263
#[tokio::test]
async fn a_stalled_stdout_does_not_hold_shutdown() {
    let (pubkey, _) = test_keypair(4);
    let mut gw = Gateway::builder(
        unused_nats_port(),
        &GatewayBuilder::dead_authority(),
        &b64(&pubkey),
    )
    .env("AI_LOG", "debug")
    .stall_stdout()
    .config_line("shutdown_grace_period_secs = 30")
    .start()
    .await;
    reject_until_logs_drop(&gw, false).await;
    gw.sigterm();
    let exited = gw.wait_exit(Duration::from_secs(15)).await;
    assert!(
        exited.is_some(),
        "a wedged stdout held the process past its log-flush bound:\n{}",
        gw.log()
    );
    gw.wait_for_log_line(&["diagnostic log sink did not drain before exit"])
        .await;
}
