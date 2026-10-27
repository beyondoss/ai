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

#[hegel::test(env::settings(40, 0x5e11_f511))]
fn a_faulty_transport_with_stdin_writes(tc: TestCase) {
    run_machine(
        tc,
        Config {
            fault_pct: 25,
            ..PLAIN
        },
    );
}

#[hegel::test(env::settings(30, 0x5e11_f512))]
fn a_faulty_transport_with_chunked_writes(tc: TestCase) {
    run_machine(
        tc,
        Config {
            fault_pct: 25,
            stdin: false,
            ..PLAIN
        },
    );
}

/// What `connect` may do over a transport that fails: fail, or learn exactly what a healthy one shows.
/// Never cache a degraded answer (no stdin, no `dd`, no `rg`) learned from failed probes: FILES-OVER-EXEC
/// section 4.2 (D3).
#[hegel::test(env::settings(40, 0x5e11_f513))]
fn connect_never_caches_what_failed_probes_say(tc: TestCase) {
    let seed = tc.draw(gs::integers::<u64>());
    let pct = tc.draw(gs::sampled_from(vec![20u64, 60, 100]));
    let clean = env::rt()
        .block_on(env::connect(env::http_runner(None)))
        .expect("a healthy loopback execd answers the probe")
        .capabilities();
    let faults = FaultRunner::new(env::http_runner(None), seed, pct);
    faults.enable(true);
    let runner: Arc<dyn CommandRunner> = faults.clone();
    let got = env::rt().block_on(env::connect(runner));
    let report = faults.take();
    tc.note(&format!(
        "pct={pct} seed={seed}: probe requests {}, failed by a fault: {}",
        report.requests, report.errored
    ));
    match got {
        Some(fs) => assert_eq!(
            fs.capabilities(),
            clean,
            "connect succeeded with capabilities learned from failed probes (errored={})",
            report.errored
        ),
        None => assert!(
            report.errored,
            "connect failed over a transport no fault touched"
        ),
    }
}
