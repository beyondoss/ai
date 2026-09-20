//! The conformance matrix: one scenario per claim the design makes.
//!
//! Each scenario says which substrate it needs. A scenario that depends on behaviour only a shared
//! network filesystem exhibits **skips** on a local directory rather than passing — a green result
//! that exercised nothing is worse than a skip that explains itself, because it is the one a reader
//! will trust.

use std::time::Duration;

use beyond_ai_test_support::exec_mock::ExecMock;
use beyond_ai_test_support::{
    spawn_model_server, spawn_model_server_with_stalled_response, turn_text,
};
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
    /// Each replica's metrics listener, in the same order as `replicas`.
    pub metrics_ports: Vec<Option<u16>>,
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
        history_path: &std::path::Path,
        max_live_sessions: Option<usize>,
        with_metrics: bool,
        gateway_url: String,
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

        let mut edge = Edge::new(keys.path(), Vec::new());
        let agent = agent_binary()?;

        let shard_args = substrate.shard_args();
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
                shards: &shard_args,
                drain_grace: Some(30),
                max_live_sessions,
                metrics_port,
            })?;
            targets.push(Target { name, port });
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
    let Placement::Served { port, .. } = placed else {
        return Outcome::failed_with(format!("could not place the session: {placed:?}"));
    };
    let mut ws = match workload::connect(port, session, &grant).await {
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
    let owner_name = match fleet.replicas.iter_mut().find(|r| r.port == port) {
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

    let Placement::Served { port: new_port, .. } = replaced else {
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
        json!({ "port": new_port, "waited_ms": waited.as_millis() }),
    );
    findings.push(Finding {
        claim: "C6",
        ok: true,
        detail: format!(
            "a survivor took the session over after {waited:?} — the lock cost a wait, not a line \
             of history"
        ),
    });

    // The new owner must have opened its own epoch, leaving A's sealed behind it.
    findings.push(check::takeover_sealed_the_previous_segment(
        &session_dir,
        epoch_before,
    ));
    findings.push(check::one_writer_per_session(&session_dir));

    // And the promise: what the client was told committed is still in the transcript it replays.
    match workload::connect(new_port, session, &grant).await {
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
    let mut fleet = match Fleet::start_against(kind, 1, history_path, None, false, gateway).await {
        Ok(f) => f,
        Err(e) => return Outcome::failed_with(e),
    };
    let (tenant, session) = ("t1", "s1.draining");
    let exec_url = fleet.exec.url.clone();
    let grant = fleet.edge.grant(tenant, session, "s1", "/", &exec_url);
    let port = fleet.replicas[0].port;

    let mut ws = match workload::connect(port, session, &grant).await {
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

    if let Err(e) = fleet.replicas[0].signal_term() {
        return Outcome::failed_with(format!("SIGTERM: {e}"));
    }
    fleet
        .history
        .record("drain_started", json!({ "port": port }));

    let mut findings = Vec::new();

    // `/readyz` has to flip within a probe interval — that answer *is* the mechanism by which the
    // load balancer stops sending new sessions here.
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut readyz = 0;
    while std::time::Instant::now() < deadline {
        if let Ok((status, _)) = workload::http_get(port, "/readyz").await {
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

    let livez = workload::http_get(port, "/livez").await.map(|(s, _)| s);
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
    let refused = workload::probe_session(port, "s1.brand-new", &fresh).await;
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
    let owned = match workload::connect(port, session, &grant).await {
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
    let port = fleet.replicas[0].port;
    let first = fleet.edge.grant("t1", "s1.first", "s1", "/", &exec_url);
    let second = fleet.edge.grant("t1", "s1.second", "s1", "/", &exec_url);

    let mut held = match workload::connect(port, "s1.first", &first).await {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("first session: {e}")),
    };
    if let Err(e) = workload::command(&mut held, json!({ "type": "get_state" }), "get_state").await
    {
        return Outcome::failed_with(format!("first get_state: {e}"));
    }

    let over = workload::probe_session(port, "s1.second", &second).await;
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
    let mut ws = match workload::connect(fleet.replicas[0].port, "s1.secret-session", &grant).await
    {
        Ok(ws) => ws,
        Err(e) => return Outcome::failed_with(format!("connect: {e}")),
    };
    let _ = workload::command(&mut ws, json!({ "type": "get_state" }), "get_state").await;

    let scrape = match workload::http_get(metrics_port, "/metrics").await {
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
    let substrate = match Substrate::prepare(kind, 2) {
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
    let all = substrate.shard_args();
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
    edge.set_targets(vec![Target {
        name: "r1".into(),
        port: replica.port,
    }]);

    // A session homed on the shard this replica does *not* mount.
    let elsewhere = edge.grant("t1", "s2.somewhere-else", "s2", "/", &exec.url);
    let answered = workload::probe_session(port, "s2.somewhere-else", &elsewhere).await;

    // And the control: the same replica serves its own shard, so a 421 above is about the shard and
    // not about the replica being broken.
    let mine = edge.grant("t1", "s1.mine", "s1", "/", &exec.url);
    let served = workload::probe_session(port, "s1.mine", &mine).await;

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
