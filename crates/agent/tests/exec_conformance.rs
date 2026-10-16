//! The exec endpoint conformance verifier (`exec_endpoint::conformance`), two ways.
//!
//! **Against an endpoint you are building** — the reason it exists. The other side of the exec
//! contract lives outside this repo; this is how it proves it conforms:
//!
//! ```sh
//! AGENT_CONFORMANCE_EXEC_URL=https://exec.example/run \
//! AGENT_CONFORMANCE_EXEC_HEADER='Authorization: Bearer …' \
//!   cargo test -p beyond-ai-agent --test exec_conformance -- --nocapture
//! ```
//!
//! It requires v1.2; set `AGENT_CONFORMANCE_ALLOW_V11=1` to accept a v1.1 endpoint (whose streaming
//! checks are then skipped). Skipped when the URL is unset.
//!
//! **Against the in-repo mock endpoints**, always: the verifier must pass a conforming endpoint and
//! fail each of the ways an endpoint gets v1.2 wrong — otherwise a green run against a real endpoint
//! would mean nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use beyond_ai_agent::exec_endpoint::conformance::{self, Expectations, Verdict, verify};
use common::exec_mock::{ExecMock, MockOptions, Protocol};

/// The mocks heartbeat fast so the verifier's silent command need not wait out the spec's 5 s.
fn quick() -> (MockOptions, Expectations) {
    (
        MockOptions {
            heartbeat: Duration::from_millis(300),
            ..MockOptions::default()
        },
        Expectations {
            heartbeat: Duration::from_millis(300),
            kill_within: Duration::from_secs(1),
        },
    )
}

async fn verify_mock(opts: MockOptions, expect: &Expectations) -> conformance::Report {
    let dir = tempfile::tempdir().unwrap();
    let mock = ExecMock::start_with(dir.path(), opts).await;
    let report = verify(&mock.url, &[], expect).await;
    println!("{}", report.render());
    // A non-conforming endpoint leaves its command running; this suite must not.
    for s in mock.streams() {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", s.pgid)])
            .stderr(std::process::Stdio::null())
            .status();
    }
    report
}

#[tokio::test]
async fn a_conforming_endpoint_passes_every_check() {
    let (opts, expect) = quick();
    let report = verify_mock(opts, &expect).await;
    assert!(report.streams);
    assert!(report.passed(), "{}", report.render());
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
    assert_eq!(report.checks.len(), 11);
}

#[tokio::test]
async fn a_v11_endpoint_passes_with_its_streaming_checks_skipped() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            protocol: Protocol::V11,
            ..opts
        },
        &expect,
    )
    .await;
    assert!(!report.streams);
    assert!(report.passed(), "{}", report.render());
    assert_eq!(
        report.check(conformance::ENVELOPE).unwrap().verdict,
        Verdict::Pass
    );
    assert_eq!(
        report.check(conformance::DISCONNECT).unwrap().verdict,
        Verdict::Skip
    );
}

#[tokio::test]
async fn a_buffering_endpoint_fails_incremental_delivery_and_heartbeats() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            protocol: Protocol::Buffered,
            ..opts
        },
        &expect,
    )
    .await;
    assert!(!report.passed());
    for name in [conformance::INCREMENTAL, conformance::HEARTBEAT] {
        assert_eq!(
            report.check(name).unwrap().verdict,
            Verdict::Fail,
            "{name}: {}",
            report.render()
        );
    }
}

#[tokio::test]
async fn an_endpoint_that_ends_the_stream_with_the_shell_fails_the_leftover_check() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            tracks_leftovers: false,
            ..opts
        },
        &expect,
    )
    .await;
    let check = report.check(conformance::LEFTOVERS).unwrap();
    assert_eq!(check.verdict, Verdict::Fail, "{}", report.render());
    assert!(check.detail.contains("shell_exit"), "{}", check.detail);
    assert_eq!(
        report
            .checks
            .iter()
            .filter(|c| c.verdict == Verdict::Fail)
            .count(),
        1,
        "{}",
        report.render()
    );
}

#[tokio::test]
async fn an_endpoint_that_survives_a_disconnect_fails_the_kill_check() {
    let (opts, expect) = quick();
    let report = verify_mock(
        MockOptions {
            kills_on_disconnect: false,
            ..opts
        },
        &expect,
    )
    .await;
    let check = report.check(conformance::DISCONNECT).unwrap();
    assert_eq!(check.verdict, Verdict::Fail, "{}", report.render());
    assert!(check.detail.contains("still running"), "{}", check.detail);
    // Everything else about it is fine: the failures are pinned on the one requirement it breaks —
    // kill on disconnect, which the leftover check measures a second time, after `shell_exit`.
    let failed: Vec<&str> = report
        .checks
        .iter()
        .filter(|c| c.verdict == Verdict::Fail)
        .map(|c| c.name)
        .collect();
    assert_eq!(
        failed,
        vec![conformance::DISCONNECT, conformance::LEFTOVERS],
        "{}",
        report.render()
    );
}

#[tokio::test]
async fn a_live_endpoint_conforms() {
    let Ok(url) = std::env::var("AGENT_CONFORMANCE_EXEC_URL") else {
        eprintln!("skipping: set AGENT_CONFORMANCE_EXEC_URL to verify an endpoint");
        return;
    };
    let headers: Vec<(String, String)> = std::env::var("AGENT_CONFORMANCE_EXEC_HEADER")
        .ok()
        .filter(|h| !h.is_empty())
        .map(|h| beyond_ai_agent::exec_endpoint::HttpExecRunner::parse_header(&h).unwrap())
        .into_iter()
        .collect();
    let report = verify(&url, &headers, &Expectations::default()).await;
    println!("{}", report.render());
    assert!(report.passed(), "{}", report.render());
    if std::env::var("AGENT_CONFORMANCE_ALLOW_V11").as_deref() != Ok("1") {
        assert!(
            report.streams,
            "a v1.1 endpoint: `bash` cannot yield or kill its commands (set \
             AGENT_CONFORMANCE_ALLOW_V11=1 to accept it)"
        );
    }
}
