//! The conformance matrix: one scenario per claim the design makes.
//!
//! Each scenario says which substrate it needs. A scenario that depends on behaviour only a shared
//! network filesystem exhibits **skips** on a local directory rather than passing — a green result
//! that exercised nothing is worse than a skip that explains itself, because it is the one a reader
//! will trust.

use std::time::{Duration, Instant};

use beyond_ai_test_support::exec_mock::ExecMock;
use beyond_ai_test_support::{
    spawn_model_server, spawn_model_server_routed_unrecorded,
    spawn_model_server_with_stalled_response, turn_text,
};
use serde_json::json;

use crate::check::{self, Finding};
use crate::edge::{Addr, Edge, Placement, Target};
use crate::faults::{Fault, Faults};
use crate::history::History;
use crate::replica::Replica;
use crate::substrate::{Kind, Substrate};
use crate::workload;

/// What a scenario did.
pub enum Outcome {
    Passed,
    Failed,
    /// Ran nothing, and said why.
    Skipped(String),
}

pub struct Scenario {
    pub name: &'static str,
    pub claims: &'static str,
    pub needs_shared_fs: bool,
}

pub const ALL: &[Scenario] = &[
    Scenario {
        name: "drain-keeps-serving-what-it-owns",
        claims: "C7",
        needs_shared_fs: false,
    },
    Scenario {
        name: "live-session-cap-refuses",
        claims: "C9",
        needs_shared_fs: false,
    },
    Scenario {
        name: "unmounted-shard-is-misdirected",
        claims: "C5",
        needs_shared_fs: false,
    },
    Scenario {
        name: "one-tenant-cannot-read-another",
        claims: "C10",
        needs_shared_fs: false,
    },
    Scenario {
        name: "metrics-name-no-tenant",
        claims: "C11",
        needs_shared_fs: false,
    },
    Scenario {
        name: "owner-refuses-non-owner",
        claims: "C4, C1",
        needs_shared_fs: false,
    },
    Scenario {
        name: "takeover-after-hard-kill",
        claims: "C1, C2, C6",
        needs_shared_fs: true,
    },
    Scenario {
        name: "hung-mount-is-reported-not-leaked",
        claims: "C8",
        needs_shared_fs: true,
    },
    Scenario {
        name: "fenced-owner-stops-and-says-so",
        claims: "C3",
        needs_shared_fs: true,
    },
];

impl Drop for Fleet {
    fn drop(&mut self) {
        // **Replicas first, explicitly.** Fields drop in declaration order, and `substrate` is
        // declared first — so without this the mounts, veth pairs and network namespaces went away
        // while the replicas were still running on them. A replica then sat in an uninterruptible
        // NFS RPC whose network namespace no longer existed, and the harness hung in `wait()` for
        // the child it had just killed. Nothing about that is visible in the scenario that
        // triggers it, which is exactly why it is spelled out here rather than left to field order.
        for r in &mut self.replicas {
            let _ = r.kill_hard();
        }
    }
}

/// Marks an error that is really "this scenario cannot run against *this* fleet" rather than a
/// failure. The matrix turns it into a skip with the reason attached, so a run never reports a claim
/// as proved when the fleet could not host the scenario that proves it.
pub const ATTACHED_SKIP: &str = "cannot-attach: ";

/// An attached fleet's address, set once from the command line.
///
/// A global rather than a parameter threaded through nine scenario signatures, and that is a
/// deliberate trade: the spec is parsed once from `argv`, is immutable afterwards, and every
/// scenario reaches `Fleet::start_against` eventually — so threading it would add an argument to
/// nine functions to reach one decision point. What a scenario *does* still differs by substrate,
/// and that stays explicit via `Kind::is_shared_filesystem`.
static ATTACHED: std::sync::OnceLock<Attachment> = std::sync::OnceLock::new();

/// Where an attached fleet lives. See [`Fleet::attach`].
#[derive(Clone)]
pub struct Attachment {
    pub shards: Vec<(String, std::path::PathBuf)>,
    pub replicas: Vec<Addr>,
    pub fault_cmd: String,
    /// Where the mock model server binds, and — on the next port up — the exec double.
    ///
    /// Fixed rather than ephemeral because the replicas are started **before** this process: they
    /// carry `--gateway-url` from their task definition and the exec URL inside every grant, so both
    /// addresses have to be predictable and reachable from another host. `127.0.0.1:0` is neither.
    pub mock_listen: Addr,
}

/// Name the fleet every scenario in this process will attach to. Called once, from `main`.
pub fn attach_to(spec: Attachment) {
    let _ = ATTACHED.set(spec);
}

/// What an attached fleet cannot provide, if a scenario asked for it.
///
/// A replica's live-session cap and metrics listener are set when the *task* starts, which happened
/// before this process did. Rather than quietly ignoring the request and grading a scenario that
/// tested something else, say so — the caller turns it into a skip with a reason, which is the same
/// contract `needs_shared_fs` already uses.
fn attached_cannot(max_live_sessions: Option<usize>, with_metrics: bool) -> Option<String> {
    if max_live_sessions.is_some() {
        return Some(
            "an attached replica's --max-live-sessions was fixed when its task started; the fleet \
             would have to be provisioned with the cap this scenario needs"
                .to_owned(),
        );
    }
    if with_metrics {
        return Some(
            "an attached replica's metrics listener is loopback-only inside its own task, so it is \
             only scrapeable through a sidecar the fleet must be provisioned with"
                .to_owned(),
        );
    }
    None
}

/// Everything a scenario needs standing up: shards, an edge, replicas, a history.
pub struct Fleet {
    pub substrate: Substrate,
    pub edge: Edge,
    pub replicas: Vec<Replica>,
    pub history: History,
    pub gateway_url: String,
    /// The sandbox every session's tools run in. One per fleet: the scenarios here are about storage
    /// and ownership, not isolation between sandboxes, which the service suites already cover.
    pub exec: ExecMock,
    /// Each replica's metrics listener, in the same order as `replicas`.
    pub metrics_ports: Vec<Option<u16>>,
    /// How this fleet's faults are performed. Local mechanisms for a fleet it spawned; a
    /// caller-supplied command for one it attached to — see [`crate::faults`].
    pub faults: Faults,
    _keys: tempfile::TempDir,
    _sandbox: tempfile::TempDir,
}

impl Fleet {
    /// Stand up `replicas` replicas over one shard, all mounting it.
    pub async fn start(
        kind: Kind,
        replicas: usize,
        history_path: &std::path::Path,
    ) -> Result<Self, String> {
        Self::start_with(kind, replicas, history_path, None, false).await
    }

    /// Attach to a fleet **somebody else is running**, on storage somebody else provisioned.
    ///
    /// The inverse of every other constructor: nothing is spawned, nothing is exported, nothing is
    /// mounted, and `Drop` takes nothing down. The shards are paths that already exist, the replicas
    /// are addresses that already answer, and every fault goes through `fault_cmd` because the
    /// simulator owns neither the processes nor the network.
    ///
    /// This is the path the EFS proof runs on — and it is worth running against the *local* NFS
    /// substrate first, with a fault script that drives `ip link`, because then the attach path is
    /// proved by the same nine scenarios before any of it depends on AWS.
    ///
    /// The grant and seal keys are deterministic (`Minter::new`'s seeds are fixed so a failure
    /// reproduces), so the replicas can be started with the matching `--grant-key`/`--seal-key`
    /// before the simulator ever runs — see the `keys` subcommand. That also means they are **not
    /// secret**: an attached fleet must be somewhere private.
    pub async fn attach(
        shards: Vec<(String, std::path::PathBuf)>,
        replicas: Vec<Addr>,
        fault_cmd: String,
        spec_bind: Addr,
        history_path: &std::path::Path,
    ) -> Result<Self, String> {
        if replicas.is_empty() {
            return Err("an attached fleet needs at least one --replica host:port".into());
        }
        let substrate = Substrate::attach(shards)?;
        let keys = tempfile::tempdir().map_err(|e| format!("keys: {e}"))?;
        let sandbox = tempfile::tempdir().map_err(|e| format!("sandbox: {e}"))?;
        let sandbox_home = sandbox.path().join("home");
        std::fs::create_dir_all(&sandbox_home).map_err(|e| format!("sandbox home: {e}"))?;
        // Bound where the caller said, and reachable by that same address: a replica dialing
        // `127.0.0.1` would reach itself, not the driver.
        let model_at = format!("{}:{}", spec_bind.host, spec_bind.port);
        let exec_at = format!("{}:{}", spec_bind.host, spec_bind.port + 1);
        let exec = ExecMock::start_on(&exec_at, sandbox.path(), Some(&sandbox_home), true).await;
        let gateway_url = beyond_ai_test_support::spawn_model_server_routed_unrecorded_on(
            &model_at,
            Vec::new(),
            turn_text("ok"),
        );

        let attached: Vec<Replica> = replicas
            .iter()
            .enumerate()
            .map(|(i, addr)| Replica::attach(&format!("r{}", i + 1), addr.clone()))
            .collect();
        let targets = attached
            .iter()
            .map(|r| Target {
                name: r.name.clone(),
                addr: r.addr.clone(),
            })
            .collect();
        let mut edge = Edge::new(keys.path(), Vec::new());
        edge.set_targets(targets);

        // Every replica is reachable before a scenario starts, or the first failure is a routing
        // mystery rather than a claim.
        for r in &attached {
            match workload::http_get(&r.addr, "/livez").await {
                Ok((200, _)) => {}
                other => {
                    return Err(format!(
                        "replica {} at {} did not answer /livez with 200: {other:?}",
                        r.name, r.addr
                    ));
                }
            }
        }

        let metrics_ports = vec![None; attached.len()];
        let history = History::create(history_path)?;
        Ok(Self {
            substrate,
            edge,
            replicas: attached,
            history,
            gateway_url,
            exec,
            metrics_ports,
            faults: Faults::external(fault_cmd),
            _keys: keys,
            _sandbox: sandbox,
        })
    }

    /// As [`start`](Self::start), over `shards` shards rather than one.
    ///
    /// A real slice mounts up to ten, and which shard a session lives on is carried in its own id —
    /// so a fleet with one shard cannot show a session being addressed to the wrong place, or two
    /// tenants' sessions landing in separate subtrees of separate mounts.
    pub async fn start_sharded(
        kind: Kind,
        replicas: usize,
        shards: usize,
        history_path: &std::path::Path,
    ) -> Result<Self, String> {
        // A soak runs for as long as it is asked to, so its model has to as well.
        //
        // `spawn_model_server` answers a fixed list of replies **and then stops accepting**, which
        // silently capped every run at exactly as many turns as there were canned replies — two
        // soaks in a row reported "64 turns committed", which is the length of that list rather
        // than anything about the fleet. Every chaos event after the cap landed on sessions that
        // could no longer write, and the reckoning graded promises all made in the first minute.
        // The routed server answers an unbounded number of requests from its fallback, and serves
        // each connection on its own thread instead of one at a time.
        // Unrecorded: the recorder keeps every raw request, and a request carries the whole
        // transcript, so its memory grows with the square of the turn count. A soak issues hundreds
        // of thousands — it reached 23.9 GB and took the host out of memory, while looking for all
        // the world like an agent-side leak. Nothing here ever reads those bodies.
        let gateway_url = spawn_model_server_routed_unrecorded(Vec::new(), turn_text("ok"));
        Self::start_against(
            kind,
            replicas,
            shards,
            history_path,
            None,
            false,
            gateway_url,
        )
        .await
    }

    /// The shard names this fleet's replicas mount, in `--shard` order.
    pub fn shard_names(&self) -> Vec<String> {
        self.substrate
            .shards
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// As [`start`](Self::start), with the two per-replica knobs some scenarios need: a live-session
    /// cap, and a metrics listener to scrape.
    pub async fn start_with(
        kind: Kind,
        replicas: usize,
        history_path: &std::path::Path,
        max_live_sessions: Option<usize>,
        with_metrics: bool,
    ) -> Result<Self, String> {
        // Enough plain turns for every session a scenario runs; the mock replays them in order.
        let turns: Vec<String> = (0..64).map(|i| turn_text(&format!("turn-{i}"))).collect();
        Self::start_with_turns(
            kind,
            replicas,
            history_path,
            max_live_sessions,
            with_metrics,
            turns,
        )
        .await
    }

    /// As [`start_with`](Self::start_with), with the model's replies chosen by the caller — for a
    /// scenario that needs a turn to still be *running* when something happens to the replica.
    pub async fn start_with_turns(
        kind: Kind,
        replicas: usize,
        history_path: &std::path::Path,
        max_live_sessions: Option<usize>,
        with_metrics: bool,
        turns: Vec<String>,
    ) -> Result<Self, String> {
        let (gateway_url, _bodies) = spawn_model_server(turns);
        Self::start_against(
            kind,
            replicas,
            1,
            history_path,
            max_live_sessions,
            with_metrics,
            gateway_url,
        )
        .await
    }

    /// As above, against a model server the caller already has — for a scenario that needs the model
    /// to *stall*, which is the only way to hold a run in flight without depending on how fast a
    /// sandbox happens to be.
    pub async fn start_against(
        kind: Kind,
        replicas: usize,
        shards: usize,
        history_path: &std::path::Path,
        max_live_sessions: Option<usize>,
        with_metrics: bool,
        gateway_url: String,
    ) -> Result<Self, String> {
        if let Some(spec) = ATTACHED.get() {
            if let Some(why) = attached_cannot(max_live_sessions, with_metrics) {
                return Err(format!("{ATTACHED_SKIP}{why}"));
            }
            if spec.replicas.len() < replicas {
                return Err(format!(
                    "{ATTACHED_SKIP}this scenario needs {replicas} replicas and the attached fleet \
                     has {}",
                    spec.replicas.len()
                ));
            }
            return Self::attach(
                spec.shards.clone(),
                spec.replicas[..replicas].to_vec(),
                spec.fault_cmd.clone(),
                spec.mock_listen.clone(),
                history_path,
            )
            .await;
        }
        let substrate = Substrate::prepare(kind, shards, replicas)?;
        let keys = tempfile::tempdir().map_err(|e| format!("keys: {e}"))?;
        let sandbox = tempfile::tempdir().map_err(|e| format!("sandbox: {e}"))?;
        let sandbox_home = sandbox.path().join("home");
        std::fs::create_dir_all(&sandbox_home).map_err(|e| format!("sandbox home: {e}"))?;
        // A real endpoint speaking the v1.1 exec protocol, rooted at a directory standing in for the
        // tenant's box. Service mode probes it at session start, so a session whose endpoint does not
        // work never comes up at all — which is the correct behaviour, and makes a broken double look
        // exactly like a broken sandbox.
        let exec = ExecMock::start_with_home(sandbox.path(), Some(&sandbox_home), true).await;

        let mut edge = Edge::new(keys.path(), Vec::new());
        let agent = agent_binary()?;

        let mut started = Vec::new();
        let mut targets = Vec::new();
        let mut metrics_ports = Vec::new();
        for i in 0..replicas {
            let name = format!("r{}", i + 1);
            let port = free_port()?;
            let metrics_port = if with_metrics {
                Some(free_port()?)
            } else {
                None
            };
            let replica = Replica::start(&crate::replica::Launch {
                name: &name,
                bin: &agent,
                gateway_url: &gateway_url,
                port,
                grant_key_flag: &edge.grant_key_flag(),
                seal_key: edge.seal_key(),
                shards: &substrate.shard_args_for(i),
                drain_grace: Some(30),
                max_live_sessions,
                metrics_port,
            })?;
            targets.push(Target {
                name,
                addr: Addr::local(port),
            });
            metrics_ports.push(metrics_port);
            started.push(replica);
        }
        edge.set_targets(targets);

        let history = History::create(history_path)?;
        Ok(Self {
            substrate,
            edge,
            replicas: started,
            history,
            gateway_url,
            exec,
            metrics_ports,
            faults: Faults::local(),
            _keys: keys,
            _sandbox: sandbox,
        })
    }

    /// Kill replica `idx` the way a machine failure does — no signal handler, no destructors, no
    /// chance to release a lock or seal a segment. The case the epoch fence exists for.
    ///
    /// One dispatch point rather than a choice at each call site: a scenario should not have to know
    /// whether this fleet owns its replicas, and a scenario that asked the wrong way would silently
    /// test a different failure than the one it is named for.
    pub fn kill_hard(&mut self, idx: usize) -> Result<(), String> {
        if self.faults.is_external() {
            return self.faults.run(Fault::Kill, &self.replicas[idx].name);
        }
        self.replicas[idx].kill_hard()
    }

    /// `SIGTERM` replica `idx` and **return immediately** — the deploy case, as a drain scenario
    /// needs it: the interesting window is while the replica is still up and draining, so waiting
    /// for it to exit here would skip the very thing under test. (It did, once: every drain
    /// assertion came back `Connection refused` because the replica was already gone.)
    pub fn signal_term(&mut self, idx: usize) -> Result<(), String> {
        if self.faults.is_external() {
            return self.faults.run(Fault::Term, &self.replicas[idx].name);
        }
        self.replicas[idx].signal_term()
    }

    /// `SIGTERM` replica `idx` and wait for it to actually go — what chaos wants, where the next
    /// event should not start until this one finished.
    pub fn stop_gracefully(&mut self, idx: usize, within: Duration) -> Result<(), String> {
        self.signal_term(idx)?;
        if self.faults.is_external() {
            // Whoever owns the replica owns how long stopping takes; the fault command is expected
            // to have returned only once it was done.
            return Ok(());
        }
        self.replicas[idx].wait_for_exit(within)
    }

    /// Cut replica `idx` off from its storage while its peers keep serving.
    pub fn partition(&self, idx: usize) -> Result<(), String> {
        if self.faults.is_external() {
            return self.faults.run(Fault::Partition, &self.replicas[idx].name);
        }
        self.substrate.partition(idx)
    }

    /// Give replica `idx` its storage back. Best-effort: a scenario that has already failed still
    /// has to leave the fleet usable for the next one.
    pub fn heal(&self, idx: usize) {
        if self.faults.is_external() {
            let _ = self.faults.run(Fault::Heal, &self.replicas[idx].name);
            return;
        }
        self.substrate.heal(idx);
    }

    /// Bring replica `idx` back on a fresh port, and put it back in the ring — a deploy replacing a
    /// task, not a process resurrecting.
    ///
    /// An attached replica comes back at the *same* address: whoever owns it replaces the task, and
    /// the simulator has no port to choose. So the ring does not change and there is nothing to
    /// retarget.
    pub async fn restart(&mut self, idx: usize) -> Result<(), String> {
        if self.faults.is_external() {
            return self.faults.run(Fault::Restart, &self.replicas[idx].name);
        }
        let name = self.replicas[idx].name.clone();
        let port = free_port()?;
        let shard_args = self.substrate.shard_args_for(idx);
        let replica = Replica::start(&crate::replica::Launch {
            name: &name,
            bin: &agent_binary()?,
            gateway_url: &self.gateway_url,
            port,
            grant_key_flag: &self.edge.grant_key_flag(),
            seal_key: self.edge.seal_key(),
            shards: &shard_args,
            drain_grace: Some(30),
            max_live_sessions: None,
            metrics_port: None,
        })?;
        self.replicas[idx] = replica;
        let targets = self
            .replicas
            .iter()
            .filter(|r| r.is_running())
            .map(|r| Target {
                name: r.name.clone(),
                addr: r.addr.clone(),
            })
            .collect();
        self.edge.set_targets(targets);
        Ok(())
    }

    /// Drop a replica from the edge's ring — what an orchestrator does when a task dies.
    pub fn retarget_excluding(&mut self, name: &str) {
        let targets = self
            .replicas
            .iter()
            .filter(|r| r.name != name && r.is_running())
            .map(|r| Target {
                name: r.name.clone(),
                addr: r.addr.clone(),
            })
            .collect();
        self.edge.set_targets(targets);
    }
}

/// The agent binary this simulator drives. Built by the same `cargo` invocation, so it is the code
/// under test rather than whatever happens to be on `$PATH`.
fn agent_binary() -> Result<String, String> {
    // `CARGO_BIN_EXE_*` only exists for binaries of *this* crate, so the agent is located relative
    // to this binary: both land in the same target profile directory.
    let mut path = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    path.pop();
    // `target/<profile>/deps/fleet-sim` → `target/<profile>/beyond-ai-agent`
    if path.ends_with("deps") {
        path.pop();
    }
    let agent = path.join("beyond-ai-agent");
    if !agent.exists() {
        return Err(format!(
            "{} not found — build it first: cargo build -p beyond-ai-agent",
            agent.display()
        ));
    }
    Ok(agent.display().to_string())
}

fn free_port() -> Result<u16, String> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| format!("bind: {e}"))?;
    l.local_addr()
        .map(|a| a.port())
        .map_err(|e| format!("addr: {e}"))
}

/// **C4** — a replica that does not own a session refuses it rather than serving a second copy, and
/// **C1** — the session it does own has exactly one epoch, one file.
///
/// This is the everyday case, not a failure case: two replicas mount one shard, a session is pinned
/// on whichever the hash chose, and the *other* one must never serve it. It runs on a local directory
/// because nothing here depends on cross-client filesystem behaviour — the refusal comes from the
/// session lock and the supervisor's table, both of which are real on any filesystem.
pub async fn owner_refuses_non_owner(kind: Kind, history_path: &std::path::Path) -> Outcome {
    let fleet = match Fleet::start(kind, 2, history_path).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let tenant = "t1";
    let session = "s1.alpha";
    // The workspace is in the sandbox, not on the replica — the whole point of service mode.
    let exec_url = fleet.exec.url.clone();
    let grant = fleet.edge.grant(tenant, session, "s1", "/", &exec_url);

    // Place it the way the edge would, and hold it open so the second replica sees a live owner.
    let placed = crate::edge::place(&fleet.edge, session, &grant, Duration::from_secs(20)).await;
    let Placement::Served { addr, .. } = placed else {
        return Outcome::failed_with(format!("the edge could not place the session: {placed:?}"));
    };
    fleet.history.record(
        "placed",
        json!({ "session": session, "addr": addr.to_string() }),
    );

    let mut ws = match workload::connect(&addr, session, &grant).await {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("owner connect: {e}")),
    };
    if let Err(e) = workload::command(&mut ws, json!({ "type": "get_state" }), "get_state").await {
        return Outcome::failed_with(format!("owner get_state: {e}"));
    }
    fleet
        .history
        .record("session_live", json!({ "addr": addr.to_string() }));

    // Now the other replica. It must refuse — not serve, not corrupt.
    let other = fleet
        .replicas
        .iter()
        .find(|r| r.addr != addr)
        .map(|r| r.addr.clone());
    let mut findings = Vec::new();
    match other {
        None => findings.push(Finding {
            claim: "C4",
            ok: false,
            detail: "only one replica: nothing to refuse the session".into(),
        }),
        Some(other) => match workload::probe_session(&other, session, &grant).await {
            Ok(503) => findings.push(Finding {
                claim: "C4",
                ok: true,
                detail: "the non-owner answered 503, as the edge's retry expects".into(),
            }),
            Ok(status) => findings.push(Finding {
                claim: "C4",
                ok: false,
                detail: format!("the non-owner answered {status}; only 503 is safe here"),
            }),
            Err(e) => findings.push(Finding {
                claim: "C4",
                ok: false,
                detail: format!("probe failed: {e}"),
            }),
        },
    }

    let dirs = fleet.substrate.session_dirs("s1", tenant);
    match dirs.first() {
        Some(dir) => findings.push(check::one_writer_per_session(dir)),
        None => findings.push(Finding {
            claim: "C1",
            ok: false,
            detail: "the session wrote no directory".into(),
        }),
    }

    if check::report("owner-refuses-non-owner", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

impl Outcome {
    /// Report why a scenario could not even run, then fail it. A scenario that fell over during
    /// setup has not disproved anything, and saying so is more useful than a bare FAIL.
    /// A scenario that could not proceed.
    ///
    /// A [`ATTACHED_SKIP`]-prefixed reason becomes a **skip**, not a failure: the fleet could not
    /// host the scenario, which is a different statement from the claim being false, and grading it
    /// as a failure would be as misleading as grading it as a pass. Every scenario gets this for
    /// free because they all report a failed start through here.
    fn failed_with(msg: String) -> Self {
        if let Some(why) = msg.strip_prefix(ATTACHED_SKIP) {
            return Self::Skipped(why.to_owned());
        }
        println!("  ✗ {msg}");
        Self::Failed
    }
}

/// **C1, C2, C6** — a hard kill, then a takeover.
///
/// The scenario the whole storage design exists for. A session is live on A and its turn has been
/// acknowledged to the client; A is killed with no chance to release its lock or seal its segment;
/// B must eventually take the session over, seal what A wrote, open a new epoch, and replay a
/// transcript that still contains the acknowledged turn.
///
/// Needs a shared filesystem, and skips without one rather than passing: on a local directory the
/// kernel releases a dead process's lock immediately, so the takeover is instant and the scenario
/// would be measuring the kernel instead of a lease. The interesting number — **how long B waits**,
/// and whether that fits inside the edge's retry budget — only exists across clients.
pub async fn takeover_after_hard_kill(kind: Kind, history_path: &std::path::Path) -> Outcome {
    if !kind.is_shared_filesystem() {
        return Outcome::Skipped(
            "needs `--substrate nfs`: on a local directory a dead process's lock is released \
             immediately, so this would measure the kernel rather than a lease"
                .to_string(),
        );
    }

    let mut fleet = match Fleet::start(kind, 2, history_path).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let (tenant, session) = ("t1", "s1.takeover");
    let exec_url = fleet.exec.url.clone();
    let grant = fleet.edge.grant(tenant, session, "s1", "/", &exec_url);

    // Place it, and commit a turn the client is told about. That acknowledgement is the promise the
    // rest of this scenario has to keep.
    let placed = crate::edge::place(&fleet.edge, session, &grant, Duration::from_secs(30)).await;
    let Placement::Served { addr, .. } = placed else {
        return Outcome::failed_with(format!("could not place the session: {placed:?}"));
    };
    let mut ws = match workload::connect(&addr, session, &grant).await {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("owner connect: {e}")),
    };
    let marker = "committed-before-the-kill";
    if let Err(e) = workload::prompt(&mut ws, marker).await {
        return Outcome::failed_with(format!("prompt: {e}"));
    }
    fleet
        .history
        .record("message_committed", json!({ "text": marker }));

    let dirs = fleet.substrate.session_dirs("s1", tenant);
    let Some(session_dir) = dirs.first().cloned() else {
        return Outcome::failed_with("the session wrote no directory".into());
    };
    let epoch_before = check::segments(&session_dir)
        .last()
        .map(|(e, _)| *e)
        .unwrap_or(0);

    // Kill the owner outright: no signal handler, no destructors, nothing released. Its lock now
    // survives on the server's lease, which is the whole point.
    let owner_name = match fleet.replicas.iter_mut().find(|r| r.addr == addr) {
        Some(r) => {
            let name = r.name.clone();
            if let Err(e) = r.kill_hard() {
                return Outcome::failed_with(format!("kill: {e}"));
            }
            name
        }
        None => return Outcome::failed_with("the placed port matches no replica".into()),
    };
    drop(ws);
    fleet
        .history
        .record("owner_killed", json!({ "replica": owner_name }));
    fleet.retarget_excluding(&owner_name);

    // The edge re-places, and keeps retrying: every attempt is refused until the dead owner's lease
    // lapses and the survivor can take the lock.
    let began = std::time::Instant::now();
    let replaced =
        crate::edge::place(&fleet.edge, session, &grant, crate::edge::RETRY_BUDGET).await;
    let waited = began.elapsed();
    let mut findings = Vec::new();

    let Placement::Served { addr: new_addr, .. } = replaced else {
        findings.push(Finding {
            claim: "C6",
            ok: false,
            detail: format!(
                "no replica took the session over inside the edge's {:?} retry budget ({replaced:?}). \
                 The fence means nothing was lost — but for that long, the tenant saw an outage.",
                crate::edge::RETRY_BUDGET
            ),
        });
        return if check::report("takeover-after-hard-kill", &findings) {
            Outcome::Passed
        } else {
            Outcome::Failed
        };
    };
    fleet.history.record(
        "taken_over",
        json!({ "addr": new_addr.to_string(), "waited_ms": waited.as_millis() }),
    );
    findings.push(Finding {
        claim: "C6",
        ok: true,
        detail: format!(
            "a survivor took the session over after {waited:?} — the lock cost a wait, not a line \
             of history"
        ),
    });

    // Make the new owner *write* before asking whether it opened its own epoch. A takeover that has
    // only read has nothing to seal yet: the roll happens when the new owner first persists, so
    // checking before that measures the scenario's own impatience rather than the fence.
    match workload::connect(&new_addr, session, &grant).await {
        Ok(mut ws2) => {
            if let Err(e) = workload::prompt(&mut ws2, "after-the-takeover").await {
                findings.push(Finding {
                    claim: "C2",
                    ok: false,
                    detail: format!("the new owner could not commit a turn: {e}"),
                });
            }
        }
        Err(e) => findings.push(Finding {
            claim: "C2",
            ok: false,
            detail: format!("could not reach the new owner to make it write: {e}"),
        }),
    }

    // Now it must have opened its own epoch, leaving A's sealed behind it.
    findings.push(check::takeover_sealed_the_previous_segment(
        &session_dir,
        epoch_before,
    ));
    findings.push(check::one_writer_per_session(&session_dir));

    // And the promise: what the client was told committed is still in the transcript it replays.
    match workload::connect(&new_addr, session, &grant).await {
        Ok(mut ws2) => match workload::transcript(&mut ws2).await {
            Ok(replayed) => {
                let history =
                    crate::history::History::read(fleet.history.path()).unwrap_or_default();
                findings.push(check::no_acknowledged_message_lost(&history, &replayed));
            }
            Err(e) => findings.push(Finding {
                claim: "no-lost-write",
                ok: false,
                detail: format!("could not read the transcript back: {e}"),
            }),
        },
        Err(e) => findings.push(Finding {
            claim: "no-lost-write",
            ok: false,
            detail: format!("could not reattach after the takeover: {e}"),
        }),
    }

    if check::report("takeover-after-hard-kill", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// **C7** — a draining replica refuses new sessions, keeps the ones it owns, and says so on
/// `/readyz` while staying alive on `/livez`.
///
/// The contract a rolling deploy depends on, checked from the outside: the load balancer learns to
/// stop choosing this replica, the orchestrator learns not to kill it, and a client whose session
/// lives here — which cannot be served anywhere else until the lock is released — keeps working.
pub async fn drain_keeps_serving_what_it_owns(
    kind: Kind,
    history_path: &std::path::Path,
) -> Outcome {
    // A run that is still in flight when the signal lands. Without one there is nothing to drain:
    // the replica has no work to finish, shuts down at once, and every property below is
    // unobservable — correct behaviour, and a drain scenario with an idle session is a test of
    // nothing.
    //
    // The *model* stalls rather than a tool, because a tool's duration depends on the sandbox and a
    // request/response exec endpoint streams nothing while it runs, so "the command has started" is
    // not observable from here. A model that has not answered keeps the run in flight by definition.
    let gateway = spawn_model_server_with_stalled_response(
        Vec::new(),
        Duration::from_secs(6),
        vec![turn_text("finished after the signal")],
    );
    let mut fleet = match Fleet::start_against(kind, 1, 1, history_path, None, false, gateway).await
    {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let (tenant, session) = ("t1", "s1.draining");
    let exec_url = fleet.exec.url.clone();
    let grant = fleet.edge.grant(tenant, session, "s1", "/", &exec_url);
    let addr = fleet.replicas[0].addr.clone();

    let mut ws = match workload::connect(&addr, session, &grant).await {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("connect: {e}")),
    };
    // Start the run and wait until the tool is provably executing, so the drain has work to protect.
    if let Err(e) = workload::send(&mut ws, json!({ "type": "prompt", "message": "go" })).await {
        return Outcome::failed_with(format!("prompt: {e}"));
    }
    // The first event proves the run has begun; the model will not answer for several seconds, so
    // the run is in flight for the whole of the window this scenario asserts on.
    if let Err(e) = workload::read_until(&mut ws, |f| f["type"] == "event").await {
        return Outcome::failed_with(format!("waiting for the run to start: {e}"));
    }

    if let Err(e) = fleet.signal_term(0) {
        return Outcome::failed_with(format!("SIGTERM: {e}"));
    }
    fleet
        .history
        .record("drain_started", json!({ "addr": addr.to_string() }));

    let mut findings = Vec::new();

    // `/readyz` has to flip within a probe interval — that answer *is* the mechanism by which the
    // load balancer stops sending new sessions here.
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut readyz = 0;
    while std::time::Instant::now() < deadline {
        if let Ok((status, _)) = workload::http_get(&addr, "/readyz").await {
            readyz = status;
            if status == 503 {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    findings.push(Finding {
        claim: "C7",
        ok: readyz == 503,
        detail: format!("/readyz answered {readyz} while draining (503 takes it out of the pool)"),
    });

    let livez = workload::http_get(&addr, "/livez").await.map(|(s, _)| s);
    findings.push(Finding {
        claim: "C7",
        ok: livez.as_ref().is_ok_and(|s| *s == 200),
        detail: format!(
            "/livez answered {:?} (200 keeps the orchestrator from killing it mid-drain)",
            livez.unwrap_or(0)
        ),
    });

    // A *new* session is refused; the one this replica owns still answers.
    let fresh = fleet
        .edge
        .grant(tenant, "s1.brand-new", "s1", "/", &exec_url);
    let refused = workload::probe_session(&addr, "s1.brand-new", &fresh).await;
    findings.push(Finding {
        claim: "C7",
        ok: matches!(refused, Ok(503)),
        detail: format!(
            "a new session got {refused:?} (503 sends the edge to a replica that stays)"
        ),
    });

    // Asked on a *fresh* connection to the same session rather than on the socket already draining
    // the run's event stream: a reconnect is the case the contract is about, and it is the one an
    // earlier draft of the drain got wrong by checking the closed flag before the map lookup.
    let owned = match workload::connect(&addr, session, &grant).await {
        Ok(mut again) => {
            workload::command(&mut again, json!({ "type": "get_state" }), "get_state").await
        }
        Err(e) => Err(format!("reconnect refused: {e}")),
    };
    findings.push(Finding {
        claim: "C7",
        ok: owned.is_ok(),
        detail: match &owned {
            Ok(_) => {
                "the session it already owns still answers — the outage a drain exists to avoid"
                    .to_string()
            }
            Err(e) => format!("the owned session stopped answering: {e}"),
        },
    });

    if !findings.iter().all(|f| f.ok) {
        let said = fleet.replicas[0].said();
        if !said.trim().is_empty() {
            println!("  replica said: {}", said.trim());
        }
    }
    if check::report("drain-keeps-serving-what-it-owns", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// **C9** — a replica at `--max-live-sessions` refuses the next one rather than exceeding its
/// descriptor and lock budget.
pub async fn live_session_cap_refuses(kind: Kind, history_path: &std::path::Path) -> Outcome {
    // One live session allowed, so the second is the interesting one.
    let fleet = match Fleet::start_with(kind, 1, history_path, Some(1), false).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let exec_url = fleet.exec.url.clone();
    let addr = fleet.replicas[0].addr.clone();
    let first = fleet.edge.grant("t1", "s1.first", "s1", "/", &exec_url);
    let second = fleet.edge.grant("t1", "s1.second", "s1", "/", &exec_url);

    let mut held = match workload::connect(&addr, "s1.first", &first).await {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("first session: {e}")),
    };
    if let Err(e) = workload::command(&mut held, json!({ "type": "get_state" }), "get_state").await
    {
        return Outcome::failed_with(format!("first get_state: {e}"));
    }

    let over = workload::probe_session(&addr, "s1.second", &second).await;
    let findings = vec![Finding {
        claim: "C9",
        ok: matches!(over, Ok(503)),
        detail: format!(
            "at the cap, the next session got {over:?} — 503 so the edge places it elsewhere rather \
             than this replica running past its lock and descriptor budget"
        ),
    }];

    if check::report("live-session-cap-refuses", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// **C11** — nothing in the scrape names a tenant, a session, a shard or a workspace.
///
/// Checked against a **live replica that has actually served a session**, not against a freshly
/// constructed registry: a unit test can only prove the labels defined today are clean, while this
/// proves nothing leaked into one at runtime.
pub async fn metrics_name_no_tenant(kind: Kind, history_path: &std::path::Path) -> Outcome {
    let fleet = match Fleet::start_with(kind, 1, history_path, None, true).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let Some(Some(metrics_port)) = fleet.metrics_ports.first().copied() else {
        return Outcome::failed_with("the replica has no metrics listener".into());
    };
    let exec_url = fleet.exec.url.clone();
    let grant = fleet.edge.grant(
        "tenant-should-not-appear",
        "s1.secret-session",
        "s1",
        "/",
        &exec_url,
    );
    let mut ws = match workload::connect(&fleet.replicas[0].addr, "s1.secret-session", &grant).await
    {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("connect: {e}")),
    };
    let _ = workload::command(&mut ws, json!({ "type": "get_state" }), "get_state").await;

    let scrape = match workload::http_get(&Addr::local(metrics_port), "/metrics").await {
        Ok((200, body)) => body,
        other => return Outcome::failed_with(format!("scrape: {other:?}")),
    };

    let mut findings = Vec::new();
    for needle in [
        "tenant-should-not-appear",
        "secret-session",
        "tenant=",
        "session_id=",
        "shard=",
        "workspace=",
    ] {
        findings.push(Finding {
            claim: "C11",
            ok: !scrape.contains(needle),
            detail: format!("{needle:?} does not appear in the scrape"),
        });
    }
    // And the scrape is real, not empty — an empty body would pass every check above.
    findings.push(Finding {
        claim: "C11",
        ok: scrape.contains("agent_sessions_live"),
        detail: "the scrape carries the replica's own instruments".into(),
    });

    if check::report("metrics-name-no-tenant", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// **C5** — a replica asked for a shard it does not mount answers `421`, never a cross-slice hop.
///
/// The distinction is the whole reason the shard is written into the session id: `421` tells the edge
/// "you sent this to the wrong slice", which is a routing fact it can act on, while a `404` would say
/// "no such session" about a session that exists perfectly well somewhere else. A replica that
/// forwarded instead would be doing service discovery, which is the thing this design deliberately
/// keeps out of the agent.
pub async fn unmounted_shard_is_misdirected(
    kind: Kind,
    // No history: this scenario asserts on two immediate answers rather than on anything that
    // accumulated over a run, so there is nothing to compare against afterwards.
    _history_path: &std::path::Path,
) -> Outcome {
    // Two shards, and a replica that mounts only the first.
    let substrate = match Substrate::prepare(kind, 2, 1) {
        Ok(s) => s,
        Err(e) => return Outcome::failed_with(e),
    };
    let keys = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => return Outcome::failed_with(format!("keys: {e}")),
    };
    let sandbox = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => return Outcome::failed_with(format!("sandbox: {e}")),
    };
    let exec = ExecMock::start_with_home(sandbox.path(), Some(sandbox.path()), true).await;
    let (gateway_url, _bodies) = spawn_model_server(vec![turn_text("unused")]);

    let mut edge = Edge::new(keys.path(), Vec::new());
    let agent = match agent_binary() {
        Ok(a) => a,
        Err(e) => return Outcome::failed_with(e),
    };
    let port = match free_port() {
        Ok(p) => p,
        Err(e) => return Outcome::failed_with(e),
    };
    let all = substrate.shard_args_for(0);
    let only_first = &all[..1];
    let replica = match Replica::start(&crate::replica::Launch {
        name: "r1",
        bin: &agent,
        gateway_url: &gateway_url,
        port,
        grant_key_flag: &edge.grant_key_flag(),
        seal_key: edge.seal_key(),
        shards: only_first,
        drain_grace: None,
        max_live_sessions: None,
        metrics_port: None,
    }) {
        Ok(r) => r,
        Err(e) => return Outcome::failed_with(e),
    };
    let addr = replica.addr.clone();
    edge.set_targets(vec![Target {
        name: "r1".into(),
        addr: addr.clone(),
    }]);

    // A session homed on the shard this replica does *not* mount.
    let elsewhere = edge.grant("t1", "s2.somewhere-else", "s2", "/", &exec.url);
    let answered = workload::probe_session(&addr, "s2.somewhere-else", &elsewhere).await;

    // And the control: the same replica serves its own shard, so a 421 above is about the shard and
    // not about the replica being broken.
    let mine = edge.grant("t1", "s1.mine", "s1", "/", &exec.url);
    let served = workload::probe_session(&addr, "s1.mine", &mine).await;

    let findings = vec![
        Finding {
            claim: "C5",
            ok: matches!(answered, Ok(421)),
            detail: format!(
                "a shard this replica does not mount got {answered:?} — 421 is a routing fact the \
                 edge can act on, where 404 would claim the session does not exist"
            ),
        },
        Finding {
            claim: "C5",
            ok: matches!(served, Ok(101)),
            detail: format!("the shard it does mount was served ({served:?})"),
        },
    ];

    if check::report("unmounted-shard-is-misdirected", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// One complete read of a session by a client presenting `grant`: connect, replay, give up.
///
/// Bounded and timed, and both matter. What C10 asks is whether a wrong key produces a **refusal**,
/// and an attempt that simply never returns is a different answer to that question — one a patient
/// scenario would report as a hung simulator rather than as the finding it is. The elapsed time is
/// printed because "refused" and "refused after ninety seconds" are not the same contract.
async fn read_session(
    at: &Addr,
    session: &str,
    grant: &str,
) -> (Result<String, String>, std::time::Duration) {
    let began = std::time::Instant::now();
    let attempt = async {
        let mut ws = workload::connect(at, session, grant).await?;
        let msgs = workload::transcript(&mut ws).await?;
        Ok::<_, String>(serde_json::to_string(&msgs).unwrap_or_default())
    };
    let out = match tokio::time::timeout(std::time::Duration::from_secs(90), attempt).await {
        Ok(r) => r,
        Err(_) => Err("the attempt never returned, and never failed, within 90s".to_owned()),
    };
    (out, began.elapsed())
}

/// **C10** — per-tenant sealing, checked three ways that fail for three different reasons.
///
/// The claim is not "tenants are in different directories", which any bug in path handling would
/// undo silently. It is that a tenant's lines are **sealed under its own key**, so the data is
/// useless to anyone else even if every other control were bypassed. So this checks the paths, then
/// the plaintext, then the key itself — by handing a replica the right session with the wrong DEK
/// and requiring that it cannot produce the content.
pub async fn one_tenant_cannot_read_another(kind: Kind, history_path: &std::path::Path) -> Outcome {
    let mut fleet = match Fleet::start(kind, 1, history_path).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let exec_url = fleet.exec.url.clone();
    let addr = fleet.replicas[0].addr.clone();

    // Two tenants, two keys, two markers. Same shard, deliberately: sharing storage is the condition
    // the sealing exists for, and putting them on different shards would prove nothing.
    let one_dek = [0x11u8; 32];
    let two_dek = [0x22u8; 32];
    let one_marker = "tenant-one-private-text";
    let two_marker = "tenant-two-private-text";

    for (tenant, session, dek, marker) in [
        ("t-one", "s1.one", one_dek, one_marker),
        ("t-two", "s1.two", two_dek, two_marker),
    ] {
        let grant = fleet
            .edge
            .grant_with_dek(tenant, session, "s1", "/", &exec_url, dek);
        let mut ws = match workload::connect(&addr, session, &grant).await {
            Ok(ws) => ws,
            Err(e) => return Outcome::failed_with(format!("{tenant} connect: {e}")),
        };
        if let Err(e) = workload::prompt(&mut ws, marker).await {
            return Outcome::failed_with(format!("{tenant} prompt: {e}"));
        }
    }

    eprintln!("  … both tenants wrote");
    let mut findings = Vec::new();

    // 1. Paths. Each tenant's sessions live under its own subtree.
    let one_dirs = fleet.substrate.session_dirs("s1", "t-one");
    let two_dirs = fleet.substrate.session_dirs("s1", "t-two");
    findings.push(Finding {
        claim: "C10",
        ok: !one_dirs.is_empty() && !two_dirs.is_empty(),
        detail: format!(
            "each tenant wrote its own subtree ({} and {} session dir(s))",
            one_dirs.len(),
            two_dirs.len()
        ),
    });

    eprintln!("  … checking plaintext on the shard");
    // 2. Plaintext. Neither tenant's text appears anywhere on the shard in the clear — not in the
    // other's tree, and not in its own either. A directory boundary is an access control; sealing is
    // what survives one being wrong.
    let mut leaked = Vec::new();
    for (name, path) in &fleet.substrate.shards {
        for marker in [one_marker, two_marker] {
            if grep_tree(path, marker) {
                leaked.push(format!("{marker:?} readable in shard {name}"));
            }
        }
    }
    findings.push(Finding {
        claim: "C10",
        ok: leaked.is_empty(),
        detail: if leaked.is_empty() {
            "neither tenant's text is readable on the shard — every line is sealed".to_string()
        } else {
            leaked.join("; ")
        },
    });

    // 3. The key — and the session has to be **re-opened from disk** for this to mean anything.
    //
    // A live session is already open, holding the codec it was opened with. A second connection
    // naming the same tenant and id attaches to that open session and reads its in-memory
    // transcript, and the DEK in the grant is never consulted. The first version of this scenario
    // did exactly that and reported a key failure that was really its own impatience.
    //
    // Restarting the replica clears the map, so the next connection must open the sealed segments
    // from the shard — which is the only path on which the DEK is used at all.
    eprintln!(
        "  … restarting the replica (was pid {:?} port {})",
        fleet.replicas[0].pid(),
        fleet.replicas[0].addr
    );
    if let Err(e) = fleet.kill_hard(0) {
        return Outcome::failed_with(format!("could not stop the replica: {e}"));
    }
    if let Err(e) = fleet.restart(0).await {
        return Outcome::failed_with(format!("could not restart the replica: {e}"));
    }
    let addr = fleet.replicas[0].addr.clone();

    eprintln!(
        "  … restarted (now pid {:?} port {}); trying the wrong key",
        fleet.replicas[0].pid(),
        fleet.replicas[0].addr
    );
    let wrong_key = fleet
        .edge
        .grant_with_dek("t-two", "s1.two", "s1", "/", &exec_url, one_dek);
    let (with_wrong_key, wrong_took) = read_session(&addr, "s1.two", &wrong_key).await;
    eprintln!("  … the wrong key finished in {wrong_took:?}: {with_wrong_key:?}");
    if with_wrong_key
        .as_ref()
        .err()
        .is_some_and(|e| e.contains("never returned"))
    {
        eprintln!(
            "  ── the replica said ──\n{}\n  ──────────────────────",
            fleet.replicas[0].said().trim()
        );
    }

    // And the control: the *right* key, on the same restarted replica, must still read it. Without
    // this, a replica that simply failed to reopen anything would pass the check above.
    eprintln!("  … trying the right key (control)");
    let right_key = fleet
        .edge
        .grant_with_dek("t-two", "s1.two", "s1", "/", &exec_url, two_dek);
    let (with_right_key, right_took) = read_session(&addr, "s1.two", &right_key).await;
    eprintln!("  … the right key finished in {right_took:?}");
    let exposed = with_wrong_key
        .as_ref()
        .is_ok_and(|text| text.contains(two_marker));
    findings.push(Finding {
        claim: "C10",
        ok: !exposed,
        detail: if exposed {
            "a grant bearing the WRONG tenant key read the session's content".to_string()
        } else {
            format!(
                "the wrong key could not produce the content ({})",
                match &with_wrong_key {
                    Ok(_) => "session opened but the text was not there",
                    Err(_) => "the session refused to open at all",
                }
            )
        },
    });

    let recovered = with_right_key
        .as_ref()
        .is_ok_and(|text| text.contains(two_marker));
    findings.push(Finding {
        claim: "C10",
        ok: recovered,
        detail: if recovered {
            "and the right key still reads it after the restart — the refusal above is the key, \
             not a replica that reopened nothing"
                .to_string()
        } else {
            format!("the correct key could not read it either: {with_right_key:?}")
        },
    });

    if check::report("one-tenant-cannot-read-another", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// Does `needle` appear, in the clear, in any file under `root`?
fn grep_tree(root: &std::path::Path, needle: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if grep_tree(&path, needle) {
                return true;
            }
        } else if let Ok(bytes) = std::fs::read(&path)
            && bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
        {
            return true;
        }
    }
    false
}

/// How many OS threads a process has right now.
///
/// The measurement C8 is actually about. A `spawn_blocking` probe is **uncancellable**: once it is
/// stuck in a `stat` on a hard mount that stopped answering, it stays stuck, and the thread it is on
/// is gone until the mount comes back. Whether that costs one thread or one per request is the whole
/// difference between a replica that reports itself unhealthy and a replica that exhausts tokio's
/// 512-thread blocking pool and takes every other `spawn_blocking` in the process down with it.
fn threads(pid: u32) -> usize {
    std::fs::read_dir(format!("/proc/{pid}/task"))
        .map(|d| d.flatten().count())
        .unwrap_or(0)
}

/// A bounded `GET`, because the point of the scenario is that the replica answers.
async fn get_within(at: &Addr, path: &str, within: Duration) -> Result<u16, String> {
    match tokio::time::timeout(within, workload::http_get(at, path)).await {
        Ok(Ok((status, _))) => Ok(status),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(format!("{path} did not answer within {within:?}")),
    }
}

/// **C8 — a hung mount is reported, not leaked.**
///
/// The failure this exists for is specific and quiet. `/readyz` is unauthenticated by design — an
/// orchestrator deciding whether a replica may hold sessions has no grant and could never obtain
/// one — and its probe is filesystem I/O on a hard mount. A probe with no deadline and no
/// single-flight guard spawns a blocking thread per request, each one stuck forever on a mount that
/// stopped answering, while `/livez` keeps returning 200 so nothing restarts the replica. With a
/// 10-second container health check that is a thread every ten seconds against tokio's 512-thread
/// blocking pool, after which *every* `spawn_blocking` in the process — every session's persistence —
/// queues behind dead threads.
///
/// So the mount is really taken away: packets to this replica's NFS server address are dropped, which
/// is what a lost mount target looks like to a `hard` client. Its peers keep their own mounts, which
/// is why each replica mounts separately.
pub async fn hung_mount_is_reported_not_leaked(
    kind: Kind,
    history_path: &std::path::Path,
) -> Outcome {
    if !kind.is_shared_filesystem() {
        return Outcome::Skipped(
            "a local directory cannot be partitioned from itself: there is no client to cut off, \
             and a `stat` on it cannot block. Needs `--substrate nfs`."
                .into(),
        );
    }
    let fleet = match Fleet::start(kind, 1, history_path).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let addr = fleet.replicas[0].addr.clone();
    let Some(pid) = fleet.replicas[0].pid() else {
        return Outcome::failed_with("the replica has no pid".to_owned());
    };

    // Healthy first, or the rest proves nothing.
    match get_within(&addr, "/readyz", Duration::from_secs(10)).await {
        Ok(200) => {}
        other => return Outcome::failed_with(format!("/readyz before the partition: {other:?}")),
    }
    let baseline = threads(pid);

    if let Err(e) = fleet.partition(0) {
        return Outcome::failed_with(format!("could not take the mount away: {e}"));
    }

    // Past the readiness cache before asking anything. `/readyz` memoizes its last *completed*
    // answer for two seconds, and the healthy probe above filled that cache — so a burst of quick
    // requests issued now would all be answered "ready" from a snapshot taken while the mount still
    // worked. The first version of this scenario did exactly that and reported forty healthy probes
    // against a mount that was gone, which is a harness measuring its own cache.
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Then hammer it the way a container health check would, only faster. Sequential on purpose: a
    // concurrent burst would also pass a per-request-thread implementation if the burst happened to
    // be smaller than the pool, whereas a probe that is *not* single-flight leaks on every one of
    // these.
    let mut statuses = Vec::new();
    let mut slowest = Duration::ZERO;
    for _ in 0..20 {
        let began = Instant::now();
        let status = get_within(&addr, "/readyz", Duration::from_secs(15)).await;
        slowest = slowest.max(began.elapsed());
        statuses.push(status);
    }
    let after = threads(pid);
    let livez = get_within(&addr, "/livez", Duration::from_secs(10)).await;

    fleet.heal(0);

    let mut findings = Vec::new();

    let not_ready = statuses
        .iter()
        .filter(|s| matches!(s, Ok(code) if *code != 200))
        .count();
    findings.push(Finding {
        claim: "C8",
        ok: not_ready == statuses.len(),
        detail: if not_ready == statuses.len() {
            format!(
                "every one of {} probes reported not-ready, the slowest in {slowest:?} — the \
                 orchestrator is told, and told promptly",
                statuses.len()
            )
        } else {
            format!(
                "{not_ready}/{} probes reported not-ready (slowest {slowest:?}); a replica that \
                 cannot reach its shard must not claim it can: {statuses:?}",
                statuses.len()
            )
        },
    });

    findings.push(Finding {
        claim: "C8",
        ok: matches!(livez, Ok(200)),
        detail: match livez {
            Ok(200) => {
                "/livez stayed 200 — the process is alive, so the orchestrator takes it out \
                        of the pool rather than killing it mid-session"
                    .to_string()
            }
            other => format!("/livez answered {other:?}; a wedged mount is not a dead process"),
        },
    });

    // A deadline on the *answer*, separately from the status. An orchestrator that waits an
    // unbounded time for `/readyz` is an orchestrator that never takes the replica out of the pool,
    // whatever the eventual answer would have been.
    findings.push(Finding {
        claim: "C8",
        ok: slowest < Duration::from_secs(5),
        detail: format!("the slowest probe answered in {slowest:?}"),
    });

    // The bound: the probe is single-flight, so one thread is stuck on the mount however many times
    // it is asked. Four is slack for tokio growing its pool for unrelated work while this ran; the
    // failure being caught is a thread *per request*.
    let grew = after.saturating_sub(baseline);
    findings.push(Finding {
        claim: "C8",
        ok: grew <= 4,
        detail: format!(
            "{} thread(s) while the mount was gone ({baseline} → {after}) across {} probes — \
             single-flight holds; a probe per request would be {}",
            grew,
            statuses.len(),
            statuses.len()
        ),
    });

    // And it recovers: not-ready has to be a report on the mount, not a latch.
    let mut recovered = false;
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if let Ok(200) = get_within(&addr, "/readyz", Duration::from_secs(10)).await {
            recovered = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    findings.push(Finding {
        claim: "C8",
        ok: recovered,
        detail: if recovered {
            "and /readyz went back to 200 once the mount answered again — the replica rejoins the \
             pool by itself"
                .to_string()
        } else {
            "/readyz never recovered after the mount came back: not-ready latched".to_string()
        },
    });

    if check::report("hung-mount-is-reported-not-leaked", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

/// **C3 — a fenced owner stops, and says so.**
///
/// The only scenario in the matrix that stages the failure the epoch fence was actually designed
/// for, and it cannot be staged any other way. Everywhere else a replica stops owning a session
/// because it died or was asked to stop. Here it goes on believing it owns one while another replica
/// takes it: the owner is partitioned from its storage, its NFSv4 lease lapses, the server revokes
/// its open state and releases its lock, and the successor takes over and seals the segment the old
/// owner still has open.
///
/// What must then happen, in order, when the old owner comes back:
///
/// 1. Its append fails — the server expired the state that descriptor referred to.
/// 2. The failed append seals its own segment locally and forces a roll.
/// 3. The roll's `O_EXCL` create of the next epoch finds the successor already there, which is the
///    fence: the store is poisoned and never writes again.
/// 4. The session ends with `session_superseded`, so the client is told rather than left talking to
///    a replica whose writes go nowhere.
///
/// A replica that skipped any of those would be a second writer on one session — silently, because
/// its own appends would keep succeeding locally. That is the failure this whole storage design
/// exists to make impossible, and until now nothing ran it.
pub async fn fenced_owner_stops_and_says_so(kind: Kind, history_path: &std::path::Path) -> Outcome {
    if !kind.is_shared_filesystem() {
        return Outcome::Skipped(
            "needs a lock held on a lease rather than by a process, and a client that can be cut \
             off from the server. On a local directory the lock dies with the owner and there is \
             nothing to fence. Needs `--substrate nfs`."
                .into(),
        );
    }
    let fleet = match Fleet::start(kind, 2, history_path).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let (a, b) = (
        fleet.replicas[0].addr.clone(),
        fleet.replicas[1].addr.clone(),
    );
    let session = "s1.fenced";
    let grant = fleet
        .edge
        .grant("t1", session, "s1", "/", &fleet.exec.url.clone());

    // The old owner, addressed directly rather than through the ring: which replica owns this
    // session is the whole subject, so it is chosen here and not by a hash.
    let mut owner = match workload::connect(&a, session, &grant).await {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("the owner could not open it: {e}")),
    };
    match workload::prompt(&mut owner, "before-the-partition").await {
        Ok(r) if r["success"] == true => {}
        other => return Outcome::failed_with(format!("the owner's first turn: {other:?}")),
    }
    eprintln!("  … the owner holds it; cutting it off from its storage");

    if let Err(e) = fleet.partition(0) {
        return Outcome::failed_with(format!("could not partition the owner: {e}"));
    }

    // Now wait out the lease. Nothing here can shorten it: the successor is refused with a 503 for
    // as long as the *server* still considers the old owner's lock held, which is the honest cost of
    // a takeover from a partitioned owner and worth reporting as a number.
    let began = Instant::now();
    let deadline = began + Duration::from_secs(240);
    let mut successor = None;
    let mut refusals = 0u32;
    while Instant::now() < deadline {
        match workload::connect(&b, session, &grant).await {
            Ok(ws) => {
                successor = Some(ws);
                break;
            }
            Err(e) => {
                refusals += 1;
                if refusals % 10 == 1 {
                    eprintln!("  … still refused after {:?}: {e}", began.elapsed());
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
    let took_over_in = began.elapsed();
    let Some(mut successor) = successor else {
        fleet.heal(0);
        return Outcome::failed_with(format!(
            "the successor never got the session: still refused {refusals} times after \
             {took_over_in:?}, so a partitioned owner strands its sessions indefinitely"
        ));
    };
    eprintln!("  … the successor took it over after {took_over_in:?} ({refusals} refusals)");

    // It has to *write*, not just open: the seal is recorded by the roll, not by the replay.
    let successor_turn = match workload::prompt(&mut successor, "after-the-takeover").await {
        Ok(r) if r["success"] == true => true,
        other => {
            eprintln!("  … the successor could not write: {other:?}");
            false
        }
    };

    fleet.heal(0);
    eprintln!("  … storage back; the old owner is about to find out");

    // The old owner, still attached, still believing it owns this session, now tries to write.
    if let Err(e) = workload::send(
        &mut owner,
        json!({ "type": "prompt", "message": "after-the-fence" }),
    )
    .await
    {
        return Outcome::failed_with(format!("could not prompt the fenced owner: {e}"));
    }
    // Everything it says, not just the first thing: the supersession is reported on the way out of
    // the command loop, which is *after* the prompt's own response.
    let (said_superseded, frames) =
        workload::collect_until(&mut owner, Duration::from_secs(120), |f| {
            f["type"] == "session_superseded"
        })
        .await;

    let mut findings = Vec::new();

    findings.push(Finding {
        claim: "C3",
        ok: successor_turn,
        detail: if successor_turn {
            format!(
                "the successor took the session over {took_over_in:?} after the owner was cut off \
                 ({refusals} refusals meanwhile) and committed a turn — the lease is the cost, the \
                 takeover is not in doubt"
            )
        } else {
            "the successor took the lock but could not write, so nothing sealed the old owner's \
             segment and the fence was never armed"
                .to_string()
        },
    });

    findings.push(Finding {
        claim: "C3",
        ok: said_superseded,
        detail: if said_superseded {
            "the fenced owner said `session_superseded` — the client is told to reconnect, rather \
             than going on talking to a replica whose writes go nowhere"
                .to_string()
        } else {
            format!(
                "the fenced owner never said `session_superseded`; it said: {:?}",
                frames
                    .iter()
                    .map(|f| f["type"].as_str().unwrap_or("?"))
                    .collect::<Vec<_>>()
            )
        },
    });

    // The half that decides whether this is a nuisance or a broken promise: if the fenced owner
    // acknowledged that turn, it has to be in the transcript. A reader bounds each segment at the
    // offset its successor sealed it to, so anything the old owner appended past that point is
    // ignored — an acknowledgement for a line no reader will ever replay is precisely the
    // "no acknowledged line is lost" invariant failing.
    let acknowledged = frames
        .iter()
        .any(|f| f["type"] == "response" && f["command"] == "prompt" && f["success"] == true);
    let replayed = workload::transcript(&mut successor)
        .await
        .map(|msgs| serde_json::to_string(&msgs).unwrap_or_default())
        .unwrap_or_default();
    let survived = replayed.contains("after-the-fence");
    findings.push(Finding {
        claim: "C3",
        ok: !acknowledged || survived,
        detail: if !acknowledged {
            "the fenced owner acknowledged nothing, so nothing was promised".to_string()
        } else if survived {
            "the fenced owner's turn was acknowledged and is in the transcript".to_string()
        } else {
            "the fenced owner acknowledged a turn that is **not** in the replayed transcript: it \
             was appended past the offset its segment was sealed at, so every reader ignores it. \
             The client was told the turn committed and it did not."
                .to_string()
        },
    });

    // And the structural half, read from the server's own tree rather than through either client.
    let dirs = fleet.substrate.session_dirs("s1", "t1");
    for dir in &dirs {
        findings.push(check::one_writer_per_session(dir));
    }
    findings.push(Finding {
        claim: "C3",
        ok: !dirs.is_empty(),
        detail: format!("{} session director(ies) on the shard", dirs.len()),
    });

    if check::report("fenced-owner-stops-and-says-so", &findings) {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}
