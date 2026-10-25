//! The real `execd` binary against the agent's conformance verifier, v1.1 through v1.4.
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

use beyond_ai_agent::exec_endpoint::conformance::{self, Expectations, Verdict, verify};
use support::Execd;

/// A deployment that says its kept jobs survive idling advertises v1.4, and passes every check
/// the verifier has, behind a bearer.
#[tokio::test]
async fn the_binary_passes_every_check_v11_to_v14() {
    let execd = Execd::start(
        &["--kept-jobs-survive-idle"],
        &[("EXECD_BEARER", "s3cret-bearer")],
    );
    let headers = [(
        "Authorization".to_string(),
        "Bearer s3cret-bearer".to_string(),
    )];
    let report = verify(&execd.url(), &headers, &Expectations::default()).await;
    println!("{}", report.render());
    assert!(report.streams, "{}", report.render());
    assert!(report.jobs, "{}", report.render());
    assert!(report.notify, "{}", report.render());
    let not_passed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| (c.name, c.verdict))
        .collect();
    assert!(not_passed.is_empty(), "{not_passed:?}\n{}", report.render());
    assert_eq!(report.checks.len(), 17 + conformance::notify::CHECKS.len());
}

/// With the deployment setting off (the default) notify is not advertised, and everything else
/// still passes; the v1.4 checks are skipped, not failed.
#[tokio::test]
async fn without_the_idle_setting_it_is_a_conforming_v13_endpoint() {
    let execd = Execd::start(&[], &[]);
    let report = verify(&execd.url(), &[], &Expectations::default()).await;
    println!("{}", report.render());
    assert!(report.streams && report.jobs && !report.notify);
    assert!(report.passed(), "{}", report.render());
}
