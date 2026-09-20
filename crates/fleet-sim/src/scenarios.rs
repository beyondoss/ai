//! The conformance matrix: one scenario per claim the design makes.
//!
//! Each scenario says which substrate it needs. A scenario that depends on behaviour only a shared
//! network filesystem exhibits **skips** on a local directory rather than passing — a green result
//! that exercised nothing is worse than a skip that explains itself, because it is the one a reader
//! will trust.

use std::time::Duration;

use beyond_ai_test_support::exec_mock::ExecMock;
use beyond_ai_test_support::{spawn_model_server, turn_text};
use serde_json::json;

use crate::check::{self, Finding};
use crate::edge::{Edge, Placement, Target};
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
        name: "owner-refuses-non-owner",
        claims: "C4, C1",
        needs_shared_fs: false,
    },
    Scenario {
        name: "takeover-after-hard-kill",
        claims: "C1, C2, C6",
        needs_shared_fs: true,
    },
];

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
        let substrate = Substrate::prepare(kind, 1)?;
        let keys = tempfile::tempdir().map_err(|e| format!("keys: {e}"))?;
        let sandbox = tempfile::tempdir().map_err(|e| format!("sandbox: {e}"))?;
        let sandbox_home = sandbox.path().join("home");
        std::fs::create_dir_all(&sandbox_home).map_err(|e| format!("sandbox home: {e}"))?;
        // A real endpoint speaking the v1.1 exec protocol, rooted at a directory standing in for the
        // tenant's box. Service mode probes it at session start, so a session whose endpoint does not
        // work never comes up at all — which is the correct behaviour, and makes a broken double look
        // exactly like a broken sandbox.
        let exec = ExecMock::start_with_home(sandbox.path(), Some(&sandbox_home), true).await;

        // Enough turns for every session any scenario runs; the mock replays them in order.
        let turns: Vec<String> = (0..64).map(|i| turn_text(&format!("turn-{i}"))).collect();
        let (gateway_url, _bodies) = spawn_model_server(turns);

        let mut edge = Edge::new(keys.path(), Vec::new());
        let agent = agent_binary()?;

        let shard_args = substrate.shard_args();
        let mut started = Vec::new();
        let mut targets = Vec::new();
        for i in 0..replicas {
            let name = format!("r{}", i + 1);
            let port = free_port()?;
            let replica = Replica::start(
                &name,
                &agent,
                &gateway_url,
                port,
                &edge.grant_key_flag(),
                edge.seal_key(),
                &shard_args,
                Some(30),
            )?;
            targets.push(Target { name, port });
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
            _keys: keys,
            _sandbox: sandbox,
        })
    }

    /// Drop a replica from the edge's ring — what an orchestrator does when a task dies.
    pub fn retarget_excluding(&mut self, name: &str) {
        let targets = self
            .replicas
            .iter()
            .filter(|r| r.name != name && r.is_running())
            .map(|r| Target {
                name: r.name.clone(),
                port: r.port,
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
    let Placement::Served { port, .. } = placed else {
        return Outcome::failed_with(format!("the edge could not place the session: {placed:?}"));
    };
    fleet
        .history
        .record("placed", json!({ "session": session, "port": port }));

    let mut ws = match workload::connect(port, session, &grant).await {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("owner connect: {e}")),
    };
    if let Err(e) = workload::command(&mut ws, json!({ "type": "get_state" }), "get_state").await {
        return Outcome::failed_with(format!("owner get_state: {e}"));
    }
    fleet
        .history
        .record("session_live", json!({ "port": port }));

    // Now the other replica. It must refuse — not serve, not corrupt.
    let other = fleet
        .replicas
        .iter()
        .find(|r| r.port != port)
        .map(|r| r.port);
    let mut findings = Vec::new();
    match other {
        None => findings.push(Finding {
            claim: "C4",
            ok: false,
            detail: "only one replica: nothing to refuse the session".into(),
        }),
        Some(other_port) => match workload::probe_session(other_port, session, &grant).await {
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
    fn failed_with(msg: String) -> Self {
        println!("  ✗ {msg}");
        Self::Failed
    }
}

/// **C1, C2, C6** — a hard kill, then a takeover.
///
/// Needs a shared filesystem, and skips without one for a reason worth stating: on a local directory
/// a dead process's lock is released by the kernel the instant it dies, so a takeover is immediate
/// and the scenario would prove the opposite of what it claims to. The whole question — how long a
/// lease keeps a dead owner's lock, and whether the edge's retry budget covers it — only exists
/// across clients.
pub async fn takeover_after_hard_kill(kind: Kind, _history: &std::path::Path) -> Outcome {
    if !kind.is_shared_filesystem() {
        return Outcome::Skipped(
            "needs `--substrate nfs`: on a local directory a dead process's lock is released \
             immediately, so this would measure the kernel rather than a lease"
                .to_string(),
        );
    }
    Outcome::Skipped("the nfs substrate is not wired up yet".to_string())
}
