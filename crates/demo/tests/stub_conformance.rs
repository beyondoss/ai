//! The sandbox the demo deploys passes the repo's exec conformance suite, job notify included.
//!
//! It runs the verifier against the exact options the `exec-stub` binary serves
//! (`stub::options`), on a loopback port the test holds, so the sandbox's callback receiver and the
//! verifier's share a host. A full run is about 50 s: the suite waits out a deliberately silent
//! command, a grandchild's timer, and the idle window a kept job must survive.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use beyond_ai_agent::exec_endpoint::conformance::{Expectations, Verdict, notify, verify};
use beyond_ai_demo::stub;
use beyond_ai_test_support::exec_mock::ExecMock;

#[tokio::test]
async fn the_demo_sandbox_passes_every_conformance_check() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mock = ExecMock::start_with(dir.path(), stub::options(home)).await;

    let report = verify(
        &mock.url,
        &[],
        &Expectations {
            notify_idle: Duration::from_secs(5),
            ..Expectations::default()
        },
    )
    .await;
    println!("{}", report.render());

    assert!(report.streams, "{}", report.render());
    assert!(report.jobs, "{}", report.render());
    assert!(
        report.notify,
        "the sandbox must advertise job notify (v1.4)"
    );
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
    for name in notify::CHECKS {
        assert!(
            report.check(name).is_some(),
            "the v1.4 check {name} did not run"
        );
    }
}
