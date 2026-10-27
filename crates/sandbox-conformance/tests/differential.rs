//! Randomized differential testing across every client.
//!
//! Hegel generates a sequence of operations (writes, guarded writes, reads, stats, listings,
//! searches, mkdirs, raw exec, and changes made behind the client's back). Each sequence is run in
//! lockstep through EVERY driver in `sdk/conformance/drivers.toml`, each in its own fresh root on the
//! same backend, and every answer is checked two ways:
//!
//! 1. against every other driver's answer, byte for byte after `norm::canon`'s listed changes;
//! 2. on a local backend, against the filesystem itself (`oracle`): `std::fs` snapshots and a
//!    brute-force search, never another client.
//!
//! A difference is shrunk by Hegel to a minimal sequence and printed as a scenario file. The four
//! defects already known (N1..N4) are tolerated by id, as the `ShellFs` property test does.
//!
//! CI runs fixed seeds and case counts. Locally: `DIFFERENTIAL_CASES=N`, `DIFFERENTIAL_SEED=S`,
//! `DIFFERENTIAL_STEPS=N` (ops per sequence), `DIFFERENTIAL_BACKENDS=a,b` and
//! `DIFFERENTIAL_DRIVERS=rust,ts` widen, move or narrow the search. `mise run test:differential`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

mod differential {
    pub mod known;
    pub mod norm;
    pub mod ops;
    pub mod oracle;
    pub mod world;
}

use std::sync::atomic::Ordering;
use std::time::Instant;

use differential::{ops, world::World};
use hegel::TestCase;

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn settings(default_cases: u64, seed: u64) -> hegel::Settings {
    // `DIFFERENTIAL_SHRINK=0` reports the first failing sequence unshrunk, for a quick look.
    let mut phases = vec![
        hegel::Phase::Explicit,
        hegel::Phase::Reuse,
        hegel::Phase::Generate,
        hegel::Phase::Target,
    ];
    if std::env::var("DIFFERENTIAL_SHRINK").as_deref() != Ok("0") {
        phases.push(hegel::Phase::Shrink);
    }
    hegel::Settings::new()
        .phases(phases)
        .test_cases(env_u64("DIFFERENTIAL_CASES").unwrap_or(default_cases))
        .seed(Some(env_u64("DIFFERENTIAL_SEED").unwrap_or(seed)))
        .database(None)
        .suppress_health_check(hegel::HealthCheck::all())
}

fn drive(backend: &str, default_cases: u64, seed: u64) {
    if let Ok(only) = std::env::var("DIFFERENTIAL_BACKENDS")
        && !only.split(',').any(|b| b == backend)
    {
        return;
    }
    if matches!(backend, "alpine" | "debian-runas") && !differential::world::docker_available() {
        eprintln!("differential {backend}: skipped, docker is not available");
        return;
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let world = rt
        .block_on(World::start(backend))
        .unwrap_or_else(|e| panic!("differential {backend}: {e}"));
    if let Err(e) = rt.block_on(world.capabilities_agree()) {
        panic!("differential {backend}: {e}");
    }
    let steps = env_u64("DIFFERENTIAL_STEPS").unwrap_or(30) as usize;
    let started = Instant::now();
    hegel::Hegel::new(|tc: TestCase| {
        let seq = ops::sequence(&tc, steps);
        if let Err(f) = rt.block_on(differential::world::run_case(&world, &seq)) {
            panic!("DIVERGENCE\n{}", f.text);
        }
    })
    .settings(settings(default_cases, seed))
    .run();
    eprintln!(
        "differential {backend}: {} drivers ({}), {} ops in {:.1}s, known open bugs tolerated: {:?}",
        world.drivers.len(),
        world
            .drivers
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>()
            .join(","),
        world.ops_run.load(Ordering::Relaxed),
        started.elapsed().as_secs_f64(),
        world.known.lock().unwrap(),
    );
}

#[test]
fn local_plain() {
    drive("local-plain", 80, 0xd1ff_0001);
}

#[test]
fn local_without_stdin() {
    drive("local-no-stdin", 60, 0xd1ff_0002);
}

#[test]
fn local_posix_grep() {
    drive("local-grep", 60, 0xd1ff_0003);
}

#[test]
fn local_busybox_rungs() {
    drive("local-busybox-grep", 60, 0xd1ff_0004);
}

#[test]
fn alpine() {
    drive("alpine", 40, 0xd1ff_0005);
}

#[test]
fn debian_runas() {
    drive("debian-runas", 40, 0xd1ff_0006);
}
