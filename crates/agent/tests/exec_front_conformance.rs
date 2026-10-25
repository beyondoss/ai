//! The exec front / edge conformance kit (`exec_endpoint::conformance::front`), two ways.
//!
//! **Against the front you are building** — the reason it exists (`BEYOND.md`, "Conformance"):
//!
//! ```sh
//! AGENT_CONFORMANCE_FRONT_URL='https://front.example/exec/{session}' \
//! AGENT_CONFORMANCE_FRONT_HOOK=/path/to/operator-hook \
//! AGENT_CONFORMANCE_FRONT_TENANT_CAP=3 \
//!   cargo test -p beyond-ai-agent --test exec_front_conformance -- --nocapture
//! ```
//!
//! The module doc of `exec_endpoint::conformance::front` lists every variable and the hook's ops.
//! The run fails if any check fails; a skipped check says why. Skipped when the URL is unset.
//!
//! **Against the in-repo reference front**, always: the kit must pass a conforming front and fail
//! each way a front gets a requirement wrong, by the check's name. Otherwise a green run against a
//! real front would mean nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use beyond_ai_agent::exec_endpoint::conformance::Verdict;
use beyond_ai_agent::exec_endpoint::conformance::front::{self, CommandHooks, FrontTarget, verify};
use common::front_mock::{Fault, FrontMock};

const CAP: usize = 2;

async fn run(fault: Fault) -> front::FrontReport {
    let mock = FrontMock::start(fault, CAP).await;
    let mut target = FrontTarget::new(mock.exec_url());
    target.health_url = Some(mock.health_url());
    target.tenant_cap = Some(CAP);
    target.settle = Duration::from_millis(400);
    if fault == Fault::SlowCreate {
        target.probe = Duration::from_millis(500);
    }
    let report = verify(&target, &mock).await;
    println!("{fault:?}\n{}", report.render());
    report
}

#[tokio::test]
async fn a_conforming_front_passes_every_check() {
    let report = run(Fault::None).await;
    let not_passed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| c.name)
        .collect();
    assert_eq!(not_passed, [front::EXEC_PROTOCOL], "{}", report.render());
    assert_eq!(report.checks.len(), front::CHECKS.len());
    assert!(report.passed());
}

/// Each broken front fails exactly the check it breaks, and no other.
#[tokio::test]
async fn each_broken_front_fails_its_check_by_name() {
    let cases = [
        (Fault::NoSingleFlight, front::SINGLE_FLIGHT),
        (Fault::LeaksOnFailedCreate, front::FAILED_CREATE),
        (Fault::NoCap, front::TENANT_CAP),
        (Fault::CapWithoutRetryAfter, front::TENANT_CAP),
        (Fault::StaleRoute, front::GONE_VM),
        (Fault::ExtensionIs500, front::EXTENSIONS),
        (Fault::Placeholder, front::NO_PLACEHOLDER),
        (Fault::SlowCreate, front::LATENCY),
        (Fault::AcceptsMissingToken, front::TOKEN_MISSING),
        (Fault::AcceptsGarbageToken, front::TOKEN_GARBAGE),
        (Fault::IgnoresSandboxScope, front::TOKEN_OTHER_SANDBOX),
        (Fault::IgnoresEpoch, front::TOKEN_STALE_EPOCH),
        (Fault::HealthIsOpen, front::TOKEN_HEALTH),
    ];
    let reports = futures::future::join_all(cases.iter().map(|(fault, _)| run(*fault))).await;
    for ((fault, broken), report) in cases.iter().zip(reports) {
        let failed: Vec<_> = report
            .checks
            .iter()
            .filter(|c| c.verdict == Verdict::Fail)
            .map(|c| c.name)
            .collect();
        assert!(
            failed.contains(broken),
            "{fault:?} must fail {broken:?}:\n{}",
            report.render()
        );
    }
}

/// A front that creates a VM for a request it then refuses is caught by the token checks that
/// refuse a token naming the tenant.
#[tokio::test]
async fn a_refused_request_that_still_creates_a_vm_fails() {
    let report = run(Fault::CreatesBeforeAuth).await;
    for name in [front::TOKEN_OTHER_SANDBOX, front::TOKEN_STALE_EPOCH] {
        assert_eq!(
            report.check(name).unwrap().verdict,
            Verdict::Fail,
            "{}",
            report.render()
        );
    }
}

/// A check whose input is not configured says so instead of passing.
#[tokio::test]
async fn unconfigured_checks_are_skipped_not_passed() {
    let mock = FrontMock::start(Fault::None, CAP).await;
    let report = verify(&FrontTarget::new(mock.exec_url()), &mock).await;
    for name in [front::TENANT_CAP, front::TOKEN_HEALTH] {
        assert_eq!(
            report.check(name).unwrap().verdict,
            Verdict::Skip,
            "{}",
            report.render()
        );
    }
}

/// Against a real front, configured from the environment.
#[tokio::test]
async fn the_configured_front_conforms() {
    let Ok(url) = std::env::var("AGENT_CONFORMANCE_FRONT_URL") else {
        eprintln!("skipping: set AGENT_CONFORMANCE_FRONT_URL to verify a front");
        return;
    };
    let hook = std::env::var("AGENT_CONFORMANCE_FRONT_HOOK").expect(
        "AGENT_CONFORMANCE_FRONT_HOOK: the operator hook command (see the kit's module doc)",
    );
    let env = |k: &str| std::env::var(format!("AGENT_CONFORMANCE_FRONT_{k}")).ok();
    let secs = |k: &str| env(k).and_then(|v| v.parse().ok()).map(Duration::from_secs);
    let mut target = FrontTarget::new(url);
    target.health_url = env("HEALTH_URL");
    target.tenant_cap = env("TENANT_CAP").and_then(|v| v.parse().ok());
    target.exec_protocol = env("EXEC").as_deref() == Some("1");
    target.probe = secs("PROBE_SECS").unwrap_or(target.probe);
    target.silence = secs("SILENCE_SECS").unwrap_or(target.silence);
    target.settle = secs("SETTLE_SECS").unwrap_or(target.settle);
    let report = verify(&target, &CommandHooks(hook)).await;
    println!("{}", report.render());
    assert!(report.passed(), "{}", report.render());
}
