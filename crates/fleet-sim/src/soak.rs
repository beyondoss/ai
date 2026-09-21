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

/// One tenant in the soak: its name, its data key, and the sessions it owns.
///
/// Separate keys are the point. A single-tenant soak exercises the storage layout but says nothing
/// about the property the whole sealing design exists for, and the failure it would miss is not a
/// subtle one — a session written under the wrong tenant's subtree, or sealed under a key its owner
/// does not hold, looks exactly like a working session until somebody else can read it.
struct Tenant {
    name: String,
    dek: [u8; 32],
}

/// What one worker session is: who owns it, where it lives, and what it is called.
struct Assignment {
    tenant: usize,
    shard: String,
    session: String,
}

/// Spread `sessions` over `tenants` × `shards`, so both dimensions are populated at any count.
///
/// Round-robin rather than random: a soak is reproducible from its seed, and which tenant owns which
/// session should not change when the seed does — only what happens to them should.
fn assign(tenants: &[Tenant], shards: &[String], sessions: usize) -> Vec<Assignment> {
    (0..sessions)
        .map(|i| {
            let tenant = i % tenants.len();
            let shard = shards[(i / tenants.len()) % shards.len()].clone();
            Assignment {
                session: format!("{shard}.soak-{i}"),
                tenant,
                shard,
            }
        })
        .collect()
}

/// Run the soak. Returns true if every invariant held.
pub async fn run(
    kind: Kind,
    duration: Duration,
    seed: u64,
    sessions: usize,
    tenant_count: usize,
    shard_count: usize,
    reconnect_every: u64,
    chaos: bool,
) -> bool {
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("soak: {e}");
            return false;
        }
    };
    let history_path = dir.path().join("soak.jsonl");
    let mut fleet = match Fleet::start_sharded(kind, 3, shard_count.max(1), &history_path).await {
        Ok(f) => f,
        Err(e) => {
            eprintln!("soak: {e}");
            return false;
        }
    };
    let shard_names = fleet.shard_names();
    let tenants: Vec<Tenant> = (0..tenant_count.max(1))
        .map(|i| Tenant {
            name: format!("t{}", i + 1),
            // A distinct key per tenant, which is the only thing the isolation claim is about.
            dek: [(i as u8).wrapping_add(1); 32],
        })
        .collect();
    let plan = assign(&tenants, &shard_names, sessions);
    println!(
        "soak · substrate {kind:?} · {duration:?} · seed {seed} · {sessions} session(s) over {} \
         tenant(s) × {} shard(s) · 3 replicas",
        tenants.len(),
        shard_names.len()
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

    // The edge's routable set, shared rather than copied. A worker that captured the target list at
    // start would keep dialling a replica that a deploy has since replaced — and a restart brings the
    // replacement up on a *new* port, so those sessions would be unreachable for the rest of the run
    // through no fault of the thing under test. A real client asks the edge every time; so does this.
    let targets = Arc::new(std::sync::Mutex::new(fleet.edge.targets().to_vec()));

    // One task per session, each looping: place through the edge, commit a turn, occasionally attach
    // twice or drop the connection and come back. Every acknowledgement is recorded as a promise.
    let mut workers = Vec::new();
    for (i, a) in plan.iter().enumerate() {
        let tenant = &tenants[a.tenant];
        let session = a.session.clone();
        let exec_url = fleet.exec.url.clone();
        let grant =
            fleet
                .edge
                .grant_with_dek(&tenant.name, &session, &a.shard, "/", &exec_url, tenant.dek);
        let (history, stop, committed, refused, targets) = (
            history.clone(),
            stop.clone(),
            committed.clone(),
            refused.clone(),
            targets.clone(),
        );
        let tenant_name = tenant.name.clone();
        let mut rng = Rng::new(seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
        workers.push(tokio::spawn(async move {
            let mut turn = 0u64;
            // The socket is kept **across turns**, which is what a client actually does: a phone or
            // a TUI opens one connection and talks over it for as long as the conversation lasts.
            // Reconnecting per turn made every turn pay a TCP handshake, a WebSocket upgrade, a
            // grant verification and a session attach, and charged all of it to what looked like
            // storage throughput.
            let mut held: Option<workload::Ws> = None;
            while !stop.load(Ordering::Relaxed) {
                // Reattach on a cadence anyway, because detach and re-attach is the path the
                // connection-is-a-view design exists for and a soak that never exercised it would
                // be testing a client nobody has.
                if reconnect_every != 0 && turn % reconnect_every == 0 {
                    held = None;
                }
                if held.is_none() {
                    // The worker routes itself, against the edge's *current* view: a real client
                    // only knows the edge's rule, and during chaos the replica it lands on changes
                    // underneath it.
                    let now: Vec<_> = targets
                        .lock()
                        .map(|t| t.clone())
                        .unwrap_or_else(|e| e.into_inner().clone());
                    let placed =
                        edge::place_among(&now, &session, &grant, Duration::from_secs(20)).await;
                    let Placement::Served { port, .. } = placed else {
                        refused.fetch_add(1, Ordering::Relaxed);
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        continue;
                    };
                    let Ok(ws) = workload::connect(port, &session, &grant).await else {
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        continue;
                    };
                    // A second connection to the same session, sometimes: the phone and the laptop,
                    // which the fan-out design is for. It must not disturb the first — and it is a
                    // second *view*, never a second writer.
                    if rng.below(4) == 0
                        && let Ok(second) = workload::connect(port, &session, &grant).await
                    {
                        drop(second);
                    }
                    held = Some(ws);
                }
                let Some(ws) = held.as_mut() else { continue };
                let marker = format!("{session}-turn-{turn}");
                match workload::prompt(ws, &marker).await {
                    Ok(resp) if resp["success"] == true => {
                        history.record(
                            "message_committed",
                            json!({ "text": marker, "session": session, "tenant": tenant_name }),
                        );
                        committed.fetch_add(1, Ordering::Relaxed);
                        turn += 1;
                    }
                    _ => {
                        // A turn lost to a replica dying mid-flight is allowed: the contract is that
                        // an *acknowledged* turn survives, not that every attempt succeeds. The
                        // socket goes with it — whatever went wrong, this one is not trusted again.
                        history.record("turn_failed", json!({ "session": session, "turn": turn }));
                        held = None;
                    }
                }
                tokio::time::sleep(Duration::from_millis(50 + rng.below(150))).await;
            }
        }));
    }

    // Chaos: take a replica out and bring it back. Two ways, because they are not the same event and
    // the common one is not the violent one — a fleet is redeployed far more often than a machine
    // dies, and a deploy is the case a drain exists to make safe. One in three is a hard kill.
    let began = Instant::now();
    let mut rng = Rng::new(seed);
    let (mut deploys, mut kills) = (0u32, 0u32);
    // Chaos off is not a weaker soak, it is the **control**. Under continuous chaos every session a
    // dying replica held must re-place, re-acquire its lock and replay its whole transcript from
    // disk — so a throughput number measured with chaos on is partly a measurement of recovery, and
    // there is no way to tell how much without the arm that has none.
    if !chaos {
        tokio::time::sleep(duration).await;
    }
    while chaos && began.elapsed() < duration {
        tokio::time::sleep(Duration::from_millis(1500 + rng.below(2500))).await;
        let victim = rng.below(fleet.replicas.len() as u64) as usize;
        if !fleet.replicas[victim].is_running() {
            continue;
        }
        let name = fleet.replicas[victim].name.clone();
        let hard = rng.below(3) == 0;
        fleet
            .history
            .record("chaos_drain", json!({ "replica": name, "hard": hard }));
        if hard {
            kills += 1;
            let _ = fleet.replicas[victim].kill_hard();
        } else {
            deploys += 1;
            let _ = fleet.replicas[victim].signal_term();
            let _ = fleet.replicas[victim].wait_for_exit(Duration::from_secs(45));
        }
        fleet.retarget_excluding(&name);
        if let Ok(mut t) = targets.lock() {
            *t = fleet.edge.targets().to_vec();
        }
        if let Err(e) = fleet.restart(victim).await {
            eprintln!("soak: could not restart {name}: {e}");
        }
        // Published the moment the replacement is listening, which is what an edge does when a task
        // passes its health check.
        if let Ok(mut t) = targets.lock() {
            *t = fleet.edge.targets().to_vec();
        }
        fleet
            .history
            .record("chaos_restored", json!({ "replica": name }));
    }

    stop.store(true, Ordering::Relaxed);
    for w in workers {
        let _ = w.await;
    }

    // The reckoning: every acknowledged turn has to be in the transcript its session replays now,
    // every session has to be where its own tenant and shard say it is, and nothing anyone said may
    // be readable on the mount.
    let promised = History::read(history.path()).unwrap_or_default();
    // A few things to look for on the mount, not every marker in the run — see
    // [`check::nothing_readable_on_the_shard`] for why that difference is twenty-two minutes of CPU.
    // The two fragments are the *shape* every marker has, so either appearing anywhere on a shard is
    // a plaintext leak; the sampled whole markers make the check specific as well as sensitive.
    let mut needles = vec!["-turn-".to_owned(), ".soak-".to_owned()];
    needles.extend(
        promised
            .iter()
            .filter(|e| e["kind"] == "message_committed")
            .filter_map(|e| e["detail"]["text"].as_str().map(str::to_owned))
            .step_by(1 + promised.len() / 8)
            .take(8),
    );
    let mut findings = Vec::new();

    for (shard, root) in &fleet.substrate.shards {
        findings.push(check::nothing_readable_on_the_shard(root, &needles));
        for tenant in &tenants {
            for d in fleet.substrate.session_dirs(shard, &tenant.name) {
                findings.push(check::one_writer_per_session(&d));
                findings.push(check::session_is_in_its_own_subtree(
                    &d,
                    shard,
                    &tenant.name,
                ));
            }
        }
    }

    let exec_url = fleet.exec.url.clone();
    // How many sessions the hash alone could not reach. Printed because it is the measure of how
    // load-bearing the edge's ring-walk fallback is, and after a rolling deploy it is most of them.
    let mut needed_the_walk = 0u32;
    for a in &plan {
        let tenant = &tenants[a.tenant];
        let grant = fleet.edge.grant_with_dek(
            &tenant.name,
            &a.session,
            &a.shard,
            "/",
            &exec_url,
            tenant.dek,
        );
        // Two routings, deliberately, because the difference between them is a finding.
        //
        // A strict edge sends every attempt to the session's hash target and retries there, which is
        // what the design says: the hash is deterministic, so the retry waits for that session's
        // lock rather than wandering. But a session that failed over is live on the *substitute*,
        // and its hash target answers 503 until the substitute's copy is idle-reaped.
        let strict = edge::place(&fleet.edge, &a.session, &grant, Duration::from_secs(10)).await;
        let port = match strict {
            Placement::Served { port, .. } => {
                findings.push(check::reachable_by_hash(&a.session));
                port
            }
            _ => {
                let walked = edge::place_among(
                    fleet.edge.targets(),
                    &a.session,
                    &grant,
                    Duration::from_secs(20),
                )
                .await;
                match walked {
                    Placement::Served { port, .. } => {
                        needed_the_walk += 1;
                        findings.push(check::stranded_from_its_hash_target(&a.session));
                        port
                    }
                    _ => {
                        findings.push(check::unreachable_after_soak(&a.session));
                        continue;
                    }
                }
            }
        };
        match workload::connect(port, &a.session, &grant).await {
            Ok(mut ws) => match workload::transcript(&mut ws).await {
                Ok(replayed) => {
                    let mine: Vec<_> = promised
                        .iter()
                        .filter(|e| e["detail"]["session"] == a.session.as_str())
                        .cloned()
                        .collect();
                    findings.push(check::no_acknowledged_message_lost(&mine, &replayed));
                }
                Err(_) => findings.push(check::unreachable_after_soak(&a.session)),
            },
            Err(_) => findings.push(check::unreachable_after_soak(&a.session)),
        }
    }

    println!(
        "\n{} turns committed · {} placements refused · {deploys} deploy(s) · {kills} hard kill(s) \
         · {needed_the_walk}/{} session(s) needed the ring walk",
        committed.load(Ordering::Relaxed),
        refused.load(Ordering::Relaxed),
        plan.len(),
    );
    check::report(&format!("soak (seed {seed})"), &findings)
}
