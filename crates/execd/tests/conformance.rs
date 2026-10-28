//! The real `execd` binary against the agent's conformance verifier, the whole protocol.
//!
//! The verifier is the agent's, the endpoint's only client: it checks framing, the probe, exit,
//! signal and timeout frames, heartbeats, the 64 KiB frame cap, kill on disconnect (measured
//! against a grandchild's marker file), leftover processes, sandbox jobs (a held job's live output
//! and end, a dropped hold killing an unkept job, a kept job surviving its hold, `kill` reaching a
//! job that ignores `SIGTERM`, scopes) and job-end notify (a callback receiver in this process, the
//! retry and stop rules, no redirects, TLS verified, the token kept out of argv and answers). It
//! runs with the spec's own numbers: the 5 s heartbeat and a 30 s idle window, so it takes a minute.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use beyond_agent::exec_endpoint::conformance::{self, Expectations, Verdict, verify};
use support::Execd;

/// A deployment that says its kept jobs survive idling advertises notify, and passes every check
/// the verifier has, behind a bearer.
#[tokio::test]
async fn the_binary_passes_every_check_v11_to_v14() {
    // At the most verbose level, so the log holds everything execd would say.
    let execd = Execd::start(
        &[("EXECD_LOG", "debug")],
        &[("EXECD_BEARER", "s3cret-bearer")],
    );
    let headers = [(
        "Authorization".to_string(),
        "Bearer s3cret-bearer".to_string(),
    )];
    let report = verify(&execd.url(), &headers, &Expectations::default()).await;
    println!("{}", report.render());
    let not_passed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| (c.name, c.verdict))
        .collect();
    assert!(not_passed.is_empty(), "{not_passed:?}\n{}", report.render());
    assert_eq!(report.checks.len(), 20 + conformance::notify::CHECKS.len());
    // The whole run's log, at its most verbose, never held the bearer; and it is a real log, not an empty one.
    let log = execd.log();
    assert!(
        log.lines().count() > 100,
        "a verbose log is not this short: {}",
        log.lines().count()
    );
    assert!(!log.contains("s3cret-bearer"), "the bearer is in the log");
    assert!(
        !log.to_lowercase().contains("bearer "),
        "an authorization value is in the log"
    );
}

/// The verifier cannot tell it is talking to an endpoint whose commands run as another user: every
/// check passes under `EXECD_RUN_AS`, with the bearer in the environment. Needs root (or `sudo -n`).
#[tokio::test]
async fn the_binary_passes_every_check_under_run_as() {
    let Some(execd) = Execd::start_as_root(
        &[("EXECD_RUN_AS", "nobody")],
        &[("EXECD_BEARER", "s3cret-run-as-bearer")],
    ) else {
        return;
    };
    let headers = [(
        "Authorization".to_string(),
        "Bearer s3cret-run-as-bearer".to_string(),
    )];
    let report = verify(&execd.url(), &headers, &Expectations::default()).await;
    println!("{}", report.render());
    let not_passed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| (c.name, c.verdict))
        .collect();
    assert!(not_passed.is_empty(), "{not_passed:?}\n{}", report.render());
    assert_eq!(report.checks.len(), 20 + conformance::notify::CHECKS.len());
    assert!(!execd.log().contains("s3cret-run-as-bearer"));
}
