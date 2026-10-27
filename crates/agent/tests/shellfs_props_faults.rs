//! `ShellFs` over a transport that fails on purpose (`shellfs_props_env::faults`).
//!
//! The promise under test is the spec's, and no stronger: a write whose transport failed leaves the
//! file **either unchanged or fully written, never partial, and never a temp** (a temp may survive
//! only when the cleanup command itself was the request the fault hit: the cleanup is one best-effort
//! command). Reads, stats, listings and searches may fail; when they answer, they answer correctly.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use crate::shellfs_props::run_machine;
use crate::shellfs_props_env::faults::FaultRunner;
use crate::shellfs_props_env::{self as env, Config, PLAIN};
use beyond_agent::tools::exec::CommandRunner;
use hegel::TestCase;
use hegel::generators as gs;

#[hegel::test(env::settings(100, 0x5e11_f511))]
fn a_faulty_transport(tc: TestCase) {
    run_machine(
        tc,
        Config {
            fault_pct: 25,
            ..PLAIN
        },
    );
}

/// What the first operation's probe may do over a transport that fails: fail, or learn exactly what a
/// healthy one shows. Never cache a degraded answer (no `dd`, no `rg`) learned from a failed probe:
/// FILES-OVER-EXEC section 4.2 (D3).
#[hegel::test(env::settings(40, 0x5e11_f513))]
fn the_probe_never_caches_what_a_failed_request_says(tc: TestCase) {
    let seed = tc.draw(gs::integers::<u64>());
    let pct = tc.draw(gs::sampled_from(vec![20u64, 60, 100]));
    let clean = env::rt()
        .block_on(async {
            env::connect(env::http_runner(None))
                .await
                .expect("connect sends nothing")
                .capabilities()
                .await
        })
        .expect("a healthy loopback execd answers the probe");
    let faults = FaultRunner::new(env::http_runner(None), seed, pct);
    faults.enable(true);
    let runner: Arc<dyn CommandRunner> = faults.clone();
    let got = env::rt().block_on(async {
        env::connect(runner)
            .await
            .expect("connect sends nothing")
            .capabilities()
            .await
    });
    let report = faults.take();
    tc.note(&format!(
        "pct={pct} seed={seed}: probe requests {}, failed by a fault: {}",
        report.requests, report.errored
    ));
    match got {
        Ok(caps) => assert_eq!(
            caps, clean,
            "the probe succeeded with capabilities learned from failed requests (errored={})",
            report.errored
        ),
        Err(_) => assert!(
            report.errored,
            "the probe failed over a transport no fault touched"
        ),
    }
}
