//! The harness catches a bad client: a driver that corrupts one op's answer must fail the checks that
//! use that op, and no others; the same driver unbroken fails nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use sandbox_conformance::backends::Binaries;
use sandbox_conformance::matrix::{self, Driver, Plan};

const GOOD: &str = env!("CARGO_BIN_EXE_sandbox-driver-rust");
const BROKEN: &str = env!("CARGO_BIN_EXE_sandbox-driver-broken");

/// Every check id of the scenarios `files.*`, with the op that step sends.
fn ops_by_check() -> std::collections::BTreeMap<String, String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sdk/conformance/scenarios");
    let mut out = std::collections::BTreeMap::new();
    for (_, file) in sandbox_conformance::load_scenarios(&dir).unwrap() {
        for sc in file["scenarios"].as_array().unwrap() {
            let sid = sc["id"].as_str().unwrap();
            for st in sc["steps"].as_array().unwrap() {
                out.insert(
                    format!("{sid}/{}", st["id"].as_str().unwrap()),
                    st["op"].as_str().unwrap().to_string(),
                );
            }
        }
    }
    out
}

async fn run(cmd: String) -> sandbox_conformance::report::Cell {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../sdk/conformance/scenarios");
    let plan = Plan {
        drivers: vec![Driver {
            name: "d".into(),
            cmd,
        }],
        // execd's own library behind a loopback listener, with the shipped endpoint's behaviour.
        backends: vec!["embedded".into()],
        scenarios: Arc::new(matrix::load(&dir).unwrap()),
        filter: Some("files.*".into()),
        bins: Binaries::default(),
        parallel: 1,
    };
    matrix::run(plan).await.remove(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_unbroken_driver_passes() {
    let cell = run(GOOD.to_string()).await;
    assert!(cell.passes(), "{:#?}", cell.tally.failing);
    assert!(cell.tally.passed > 100);
}

async fn broken_op_fails_exactly_its_checks(op: &str) {
    let cell = run(format!("env BROKEN_INNER={GOOD} BROKEN_OP={op} {BROKEN}")).await;
    let ops = ops_by_check();
    let failing: BTreeSet<&String> = cell.tally.failing.iter().collect();
    assert!(
        failing.len() >= 5,
        "a driver that corrupts `{op}` must fail the checks that read it (failed {})",
        failing.len()
    );
    for id in &failing {
        assert_eq!(
            ops.get(*id).map(String::as_str),
            Some(op),
            "{id} failed but does not use `{op}`"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_driver_that_corrupts_read_fails_the_read_checks_only() {
    broken_op_fails_exactly_its_checks("read").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_driver_that_corrupts_stat_fails_the_stat_checks_only() {
    broken_op_fails_exactly_its_checks("stat").await;
}
