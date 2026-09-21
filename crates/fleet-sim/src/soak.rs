//! The soak: hours of randomized chaos, then the checker over everything that happened.
//!
//! The matrix proves the cases someone thought of. This is the other half — it runs sessions and
//! breaks replicas on a schedule nobody designed, and then asks whether the promises still hold. A
//! violation here is one no scenario was written for, which is the only kind left to find once the
//! matrix is green.
//!
//! **Deterministic by seed.** The schedule is drawn from a seeded PRNG, so a failing soak is a
//! failing soak you can run again. A chaos test you cannot reproduce reports a ghost.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::json;

use crate::check;
use crate::edge::{self, Placement};
use crate::history::History;
use crate::scenarios::Fleet;
use crate::substrate::Kind;
use crate::workload;

/// xorshift64*, so a run is reproducible from its seed without a dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // A zero state is a fixed point for xorshift; anything else is fine.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

/// Run the soak. Returns true if every invariant held.
pub async fn run(kind: Kind, duration: Duration, seed: u64, sessions: usize) -> bool {
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("soak: {e}");
            return false;
        }
    };
    let history_path = dir.path().join("soak.jsonl");
    let mut fleet = match Fleet::start(kind, 3, &history_path).await {
        Ok(f) => f,
        Err(e) => {
            eprintln!("soak: {e}");
            return false;
        }
    };
    println!(
        "soak · substrate {kind:?} · {duration:?} · seed {seed} · {sessions} sessions · 3 replicas"
    );

    // The workers' promises go in their own file: the fleet's history is the chaos schedule, and
    // keeping them apart means the reckoning reads acknowledgements without filtering past events it
    // is not asking about.
    let history = match History::create(&dir.path().join("worker.jsonl")) {
        Ok(h) => Arc::new(h),
        Err(e) => {
            eprintln!("soak: {e}");
            return false;
        }
    };
    let stop = Arc::new(AtomicBool::new(false));
    let committed = Arc::new(AtomicU64::new(0));
    let refused = Arc::new(AtomicU64::new(0));

    // One task per session, each looping: place through the edge, commit a turn, occasionally drop
    // the connection and come back. Every acknowledgement is recorded as a promise.
    let mut workers = Vec::new();
    for i in 0..sessions {
        let session = format!("s1.soak-{i}");
        let exec_url = fleet.exec.url.clone();
        let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec_url);
        let edge_targets: Vec<_> = fleet.edge.targets().to_vec();
        let (history, stop, committed, refused) = (
            history.clone(),
            stop.clone(),
            committed.clone(),
            refused.clone(),
        );
        let mut rng = Rng::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
        workers.push(tokio::spawn(async move {
            let mut turn = 0u64;
            while !stop.load(Ordering::Relaxed) {
                // The worker routes itself: a real client only knows the edge's rule, and during
                // chaos the replica it lands on changes underneath it.
                let placed =
                    edge::place_among(&edge_targets, &session, &grant, Duration::from_secs(20))
                        .await;
                let Placement::Served { port, .. } = placed else {
                    refused.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    continue;
                };
                let Ok(mut ws) = workload::connect(port, &session, &grant).await else {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    continue;
                };
                let marker = format!("{session}-turn-{turn}");
                match workload::prompt(&mut ws, &marker).await {
                    Ok(resp) if resp["success"] == true => {
                        history.record(
                            "message_committed",
                            json!({ "text": marker, "session": session }),
                        );
                        committed.fetch_add(1, Ordering::Relaxed);
                        turn += 1;
                    }
                    _ => {
                        // A turn lost to a replica dying mid-flight is allowed: the contract is that
                        // an *acknowledged* turn survives, not that every attempt succeeds.
                        history.record("turn_failed", json!({ "session": session, "turn": turn }));
                    }
                }
                drop(ws);
                tokio::time::sleep(Duration::from_millis(50 + rng.below(150))).await;
            }
        }));
    }

    // Chaos: drain a replica, wait for it to go, bring it back.
    let began = Instant::now();
    let mut rng = Rng::new(seed);
    while began.elapsed() < duration {
        tokio::time::sleep(Duration::from_millis(1500 + rng.below(2500))).await;
        let victim = rng.below(fleet.replicas.len() as u64) as usize;
        if !fleet.replicas[victim].is_running() {
            continue;
        }
        let name = fleet.replicas[victim].name.clone();
        fleet
            .history
            .record("chaos_drain", json!({ "replica": name }));
        let _ = fleet.replicas[victim].signal_term();
        let _ = fleet.replicas[victim].wait_for_exit(Duration::from_secs(45));
        fleet.retarget_excluding(&name);
        if let Err(e) = fleet.restart(victim).await {
            eprintln!("soak: could not restart {name}: {e}");
        }
        fleet
            .history
            .record("chaos_restored", json!({ "replica": name }));
    }

    stop.store(true, Ordering::Relaxed);
    for w in workers {
        let _ = w.await;
    }

    // The reckoning: every acknowledged turn has to be in the transcript its session replays now.
    let promised = History::read(history.path()).unwrap_or_default();
    let mut findings = Vec::new();
    for dir in fleet.substrate.session_dirs("s1", "t1") {
        findings.push(check::one_writer_per_session(&dir));
    }
    let exec_url = fleet.exec.url.clone();
    for i in 0..sessions {
        let session = format!("s1.soak-{i}");
        let grant = fleet.edge.grant("t1", &session, "s1", "/", &exec_url);
        // Two routings, deliberately, because the difference between them is a finding.
        //
        // A strict edge sends every attempt to the session's hash target and retries there, which is
        // what the design says: the hash is deterministic, so the retry waits for that session's
        // lock rather than wandering. But a session that failed over is live on the *substitute*,
        // and its hash target answers 503 until the substitute's copy is idle-reaped — an hour, by
        // default. An edge that walks the ring after repeated 503s finds it at once.
        let strict = edge::place(&fleet.edge, &session, &grant, Duration::from_secs(10)).await;
        let port = match strict {
            Placement::Served { port, .. } => {
                findings.push(check::reachable_by_hash(&session));
                port
            }
            _ => {
                let walked = edge::place_among(
                    fleet.edge.targets(),
                    &session,
                    &grant,
                    Duration::from_secs(20),
                )
                .await;
                match walked {
                    Placement::Served { port, .. } => {
                        findings.push(check::stranded_from_its_hash_target(&session));
                        port
                    }
                    _ => {
                        findings.push(check::unreachable_after_soak(&session));
                        continue;
                    }
                }
            }
        };
        match workload::connect(port, &session, &grant).await {
            Ok(mut ws) => match workload::transcript(&mut ws).await {
                Ok(replayed) => {
                    let mine: Vec<_> = promised
                        .iter()
                        .filter(|e| e["detail"]["session"] == session.as_str())
                        .cloned()
                        .collect();
                    findings.push(check::no_acknowledged_message_lost(&mine, &replayed));
                }
                Err(_) => findings.push(check::unreachable_after_soak(&session)),
            },
            Err(_) => findings.push(check::unreachable_after_soak(&session)),
        }
    }

    println!(
        "\n{} turns committed · {} placements refused · {} chaos events",
        committed.load(Ordering::Relaxed),
        refused.load(Ordering::Relaxed),
        History::read(&history_path)
            .unwrap_or_default()
            .iter()
            .filter(|e| e["kind"] == "chaos_drain")
            .count()
    );
    check::report(&format!("soak (seed {seed})"), &findings)
}
