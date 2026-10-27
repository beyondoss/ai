//! The connector broker conformance kit (`conformance::broker`), two ways.
//!
//! **Against the broker you are building** — the reason it exists (`BEYOND.md`, "Conformance"):
//!
//! ```sh
//! AGENT_CONFORMANCE_BROKER_URL='https://broker.example/managed/notes/mcp' \
//! AGENT_CONFORMANCE_BROKER_HOOK=/path/to/operator-hook \
//! AGENT_CONFORMANCE_BROKER_TOOL=my_notes \
//!   cargo nextest run -p beyond-agent -E 'test(/^broker_conformance::/)' --no-capture
//! ```
//!
//! The module doc of `conformance::broker` lists every variable and the hook's ops. Skipped when the
//! URL is unset.
//!
//! **Against the in-repo reference broker**, always: the kit must pass a conforming broker and fail
//! each way a broker gets a requirement wrong, by the check's name.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use beyond_agent::conformance::broker::{self, BrokerTarget, verify};
use beyond_agent::conformance::{CommandHook, Report};
use beyond_agent::exec_endpoint::conformance::Verdict;
use common::broker_mock::{BrokerMock, Fault};
use serde_json::json;

async fn run(fault: Fault) -> Report {
    let mock = BrokerMock::start(fault).await;
    let target = BrokerTarget {
        url: mock.url(),
        tool: Some(("whoami".into(), json!({}))),
    };
    let report = verify(&target, &mock).await;
    println!("{fault:?}\n{}", report.render());
    report
}

#[tokio::test]
async fn a_conforming_broker_passes_every_check() {
    let report = run(Fault::None).await;
    assert!(
        report.checks.iter().all(|c| c.verdict == Verdict::Pass),
        "{}",
        report.render()
    );
    assert_eq!(report.checks.len(), broker::CHECKS.len());
}

/// Each broken broker fails the check it breaks, by name.
#[tokio::test]
async fn each_broken_broker_fails_its_check_by_name() {
    let cases = [
        (Fault::MintsWrongBytes, broker::FIXTURE),
        (Fault::HonoursNamedConnection, broker::ISOLATION),
        (Fault::IgnoresTenant, broker::ISOLATION),
        (Fault::AcceptsExpired, broker::EXPIRED),
        (Fault::RevokeLags, broker::REVOKED),
        (Fault::DeleteKeepsBinding, broker::SESSION_DELETED),
        (Fault::SharedCache, broker::CACHE),
        (Fault::LeaksUpstreamToken, broker::NO_UPSTREAM_TOKEN),
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

/// Without a user tool the checks that need a call say so instead of passing.
#[tokio::test]
async fn checks_that_need_the_user_tool_are_skipped_without_it() {
    let mock = BrokerMock::start(Fault::None).await;
    let report = verify(
        &BrokerTarget {
            url: mock.url(),
            tool: None,
        },
        &mock,
    )
    .await;
    for name in [broker::ISOLATION, broker::REVOKED, broker::SESSION_DELETED] {
        assert_eq!(
            report.check(name).unwrap().verdict,
            Verdict::Skip,
            "{}",
            report.render()
        );
    }
}

/// Against a real broker, configured from the environment.
#[tokio::test]
async fn the_configured_broker_conforms() {
    let Ok(url) = std::env::var("AGENT_CONFORMANCE_BROKER_URL") else {
        eprintln!("skipping: set AGENT_CONFORMANCE_BROKER_URL to verify a broker");
        return;
    };
    let hook = std::env::var("AGENT_CONFORMANCE_BROKER_HOOK").expect(
        "AGENT_CONFORMANCE_BROKER_HOOK: the operator hook command (see the kit's module doc)",
    );
    let tool = std::env::var("AGENT_CONFORMANCE_BROKER_TOOL")
        .ok()
        .map(|name| {
            let args = std::env::var("AGENT_CONFORMANCE_BROKER_TOOL_ARGS")
                .ok()
                .and_then(|a| serde_json::from_str(&a).ok())
                .unwrap_or(json!({}));
            (name, args)
        });
    let report = verify(&BrokerTarget { url, tool }, &CommandHook(hook)).await;
    println!("{}", report.render());
    assert!(report.passed(), "{}", report.render());
}
