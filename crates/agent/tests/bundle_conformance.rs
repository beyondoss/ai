//! The bundle control plane and store conformance kit (`conformance::bundle`), two ways.
//!
//! **Against the deploy path you are building** — the reason it exists (`BEYOND.md`, "Conformance"):
//!
//! ```sh
//! AGENT_CONFORMANCE_BUNDLE_HOOK=/path/to/operator-hook \
//!   cargo nextest run -p beyond-ai-agent -E 'test(/^bundle_conformance::/)' --no-capture
//! ```
//!
//! The module doc of `conformance::bundle` lists the variables and the hook's ops. A full run waits
//! out a bundle URL's life plus the fetch window (about two minutes). Skipped when the hook is unset.
//!
//! **Against the in-repo reference deploy and store**, always: the kit must pass a conforming one and
//! fail each way one gets a requirement wrong, by the check's name.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use std::time::Duration;

use beyond_ai_agent::conformance::bundle::{self, BundleTarget, verify};
use beyond_ai_agent::conformance::{CommandHook, Report};
use beyond_ai_agent::exec_endpoint::conformance::Verdict;
use common::bundle_mock::{BundleMock, Fault};

fn target() -> BundleTarget {
    BundleTarget {
        fetch_window: BundleMock::WINDOW,
        slack: Duration::from_millis(700),
        wait: true,
        loopback_http: true,
    }
}

async fn run(fault: Fault) -> Report {
    let mock = BundleMock::start(fault).await;
    let report = verify(&target(), &mock).await;
    println!("{fault:?}\n{}", report.render());
    report
}

#[tokio::test]
async fn a_conforming_store_passes_every_check_but_the_one_loopback_cannot() {
    let report = run(Fault::None).await;
    assert_eq!(report.checks.len(), bundle::CHECKS.len());
    let not_passed: Vec<_> = report
        .checks
        .iter()
        .filter(|c| c.verdict != Verdict::Pass)
        .map(|c| c.name)
        .collect();
    assert_eq!(not_passed, [bundle::URL_HTTPS], "{}", report.render());
}

/// Each broken deploy or store fails the check it breaks, by name.
#[tokio::test(flavor = "multi_thread")]
async fn each_broken_store_fails_its_check_by_name() {
    let cases = [
        (Fault::NoValidation, bundle::UNSAFE_REFUSED),
        (Fault::NamedByLength, bundle::HASH_IS_NAME),
        (Fault::UnsignedUrl, bundle::URL_SIGNED),
        (Fault::ImmortalUrl, bundle::URL_SHORT_LIVED),
        (Fault::NoWindow, bundle::URL_SHORT_LIVED),
        (Fault::DropsMcpTools, bundle::SAVED_TOOLS),
    ];
    let reports = futures::future::join_all(cases.iter().map(|(f, _)| run(*f))).await;
    for ((fault, name), report) in cases.iter().zip(reports) {
        assert!(
            report.failed().contains(name),
            "{fault:?} must fail {name:?}:\n{}",
            report.render()
        );
    }
}

/// A plain-http URL fails the scheme check unless the target is a loopback reference.
#[tokio::test]
async fn a_plain_http_url_fails_the_https_check() {
    let mock = BundleMock::start(Fault::None).await;
    let strict = BundleTarget {
        loopback_http: false,
        wait: false,
        ..target()
    };
    let report = verify(&strict, &mock).await;
    assert!(
        report.failed().contains(&bundle::URL_HTTPS),
        "{}",
        report.render()
    );
    assert_eq!(
        report.check(bundle::URL_SHORT_LIVED).unwrap().verdict,
        Verdict::Skip
    );
}

/// Against a real deploy path, configured from the environment.
#[tokio::test]
async fn the_configured_store_conforms() {
    let Ok(hook) = std::env::var("AGENT_CONFORMANCE_BUNDLE_HOOK") else {
        eprintln!("skipping: set AGENT_CONFORMANCE_BUNDLE_HOOK to verify a deploy path");
        return;
    };
    let mut t = BundleTarget::default();
    if let Some(s) = std::env::var("AGENT_CONFORMANCE_BUNDLE_FETCH_WINDOW_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        t.fetch_window = Duration::from_secs(s);
    }
    t.wait = std::env::var("AGENT_CONFORMANCE_BUNDLE_WAIT").as_deref() != Ok("0");
    let report = verify(&t, &CommandHook(hook)).await;
    println!("{}", report.render());
    assert!(report.passed(), "{}", report.render());
}
