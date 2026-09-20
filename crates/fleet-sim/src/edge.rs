//! The edge, as a double.
//!
//! Routing is the edge's job and the agent has no part in it — that is a decision the design makes
//! explicitly, and it means a simulator that wants to exercise the real failure modes has to *be* an
//! edge. This is that: it mints `bsg_v1` grants, hashes a session id to a replica, and honours the
//! `503` + `Retry-After` a replica answers when it does not own a session.
//!
//! Because it implements a contract this repo documents rather than one it imports, it doubles as a
//! conformance reference: if the control plane's edge behaves like this one, the fleet behaves like
//! this simulator says it does.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use beyond_ai_test_support::grant::{Claims, Minter, Secrets};

/// How long a grant this edge mints is good for. Long enough that no scenario trips over expiry, and
/// finite so a grant is never mistaken for a credential.
const GRANT_TTL: Duration = Duration::from_secs(3600);

/// How long to keep retrying a `503` before calling a session unreachable.
///
/// This is the edge's **retry budget**, and whether it exceeds a dead owner's lock lease is the
/// question the whole failover story turns on: the fence is in the data, so history is safe either
/// way, but a budget shorter than the lease is an outage the tenant sees.
pub const RETRY_BUDGET: Duration = Duration::from_secs(120);

/// One replica the edge can route to.
#[derive(Clone)]
pub struct Target {
    pub name: String,
    pub port: u16,
}

/// Virtual nodes per replica on the ring. Enough that removing one replica spreads its keys across
/// the others rather than dumping them all on its neighbour.
const VNODES_PER_TARGET: usize = 64;

/// FNV-1a. Small, stable across releases, and the ring's positions have to be reproducible — a hash
/// that changed between builds would move every session on deploy, which is the failure this ring
/// exists to avoid.
fn hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub struct Edge {
    pub minter: Minter,
    targets: Vec<Target>,
}

impl Edge {
    pub fn new(key_dir: &Path, targets: Vec<Target>) -> Self {
        Self {
            minter: Minter::new(key_dir),
            targets,
        }
    }

    pub fn grant_key_flag(&self) -> String {
        self.minter.grant_key_flag()
    }

    pub fn seal_key(&self) -> &Path {
        self.minter.seal_key()
    }

    /// The replicas currently in the ring.
    pub fn targets(&self) -> &[Target] {
        &self.targets
    }

    /// Replace the routable set — a deploy, a scale-in, a replica that died.
    pub fn set_targets(&mut self, targets: Vec<Target>) {
        self.targets = targets;
    }

    /// Which replica this session id hashes to.
    ///
    /// **Consistent hashing, not `hash % len`** — and the difference is not a refinement, it is the
    /// property the design depends on. With a modulo the ring is renumbered whenever the replica set
    /// changes size, so *every* session moves when one replica leaves or rejoins. A session that
    /// moves while its previous owner still holds the lock is answered 503 by its new hash target,
    /// and the retry — which is deterministic, and so returns to that same target — waits for a lock
    /// that will not free until the old owner's session is idle-reaped. The soak found exactly that:
    /// sessions unreachable after a replica came back, because the arithmetic had reshuffled them.
    ///
    /// A ring of virtual nodes moves only the keys belonging to the replica that changed, which is
    /// what keeps a session on the replica that owns it across a scale-out.
    pub fn route(&self, session_id: &str) -> Option<&Target> {
        let point = hash(session_id.as_bytes());
        // The first virtual node at or after this point, wrapping — the textbook ketama lookup.
        let mut best: Option<(u64, usize)> = None;
        let mut lowest: Option<(u64, usize)> = None;
        for (idx, target) in self.targets.iter().enumerate() {
            for vnode in 0..VNODES_PER_TARGET {
                let at = hash(format!("{}#{vnode}", target.name).as_bytes());
                if lowest.is_none_or(|(low, _)| at < low) {
                    lowest = Some((at, idx));
                }
                if at >= point && best.is_none_or(|(b, _)| at < b) {
                    best = Some((at, idx));
                }
            }
        }
        best.or(lowest).and_then(|(_, idx)| self.targets.get(idx))
    }

    /// Mint a grant for `tenant`'s `session_id`, homed on `home_shard` and pointed at `exec_url`.
    pub fn grant(
        &self,
        tenant: &str,
        session_id: &str,
        home_shard: &str,
        workspace_root: &str,
        exec_url: &str,
    ) -> String {
        let exp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
            + GRANT_TTL.as_secs();
        let claims = Claims {
            tenant: tenant.to_string(),
            session_id: session_id.to_string(),
            home_shard: home_shard.to_string(),
            workspace_root: workspace_root.to_string(),
            exec_url: exec_url.to_string(),
            mcp: Vec::new(),
            exp,
        };
        let secrets = Secrets {
            exec_headers: Vec::new(),
            mcp_headers: BTreeMap::new(),
            gateway_key: "bai_v1.test".into(),
            // One tenant in these scenarios, so one key. A multi-tenant scenario derives a distinct
            // one per tenant — the isolation claim is that no line opens under another tenant's.
            dek: [7u8; 32],
        };
        self.minter.mint(&claims, &secrets)
    }
}

/// What happened when the edge tried to place a connection.
#[derive(Debug)]
pub enum Placement {
    /// A replica accepted it, and this is the port that did.
    Served { port: u16, waited: Duration },
    /// Every attempt inside the budget was refused.
    Exhausted { last_status: u16, waited: Duration },
    /// Nothing to route to.
    NoTarget,
}

/// Place a connection, honouring `Retry-After` exactly as a real edge must.
///
/// The *deterministic* part matters more than the backoff: the retry returns to the replica the hash
/// chose, so it queues behind that session's lock rather than being served a second copy elsewhere.
/// A simulator that retried a different replica would be testing a different — and unsafe —
/// architecture.
pub async fn place(edge: &Edge, session_id: &str, grant: &str, budget: Duration) -> Placement {
    let Some(target) = edge.route(session_id) else {
        return Placement::NoTarget;
    };
    let port = target.port;
    let began = Instant::now();
    let mut last_status = 0;
    while began.elapsed() < budget {
        match crate::workload::probe_session(port, session_id, grant).await {
            Ok(status) if status < 400 => {
                return Placement::Served {
                    port,
                    waited: began.elapsed(),
                };
            }
            Ok(status) => {
                last_status = status;
                // A 421 means this replica does not mount the session's shard: retrying it is
                // pointless, and in a real fleet the edge would re-place onto the slice that does.
                if status == 421 {
                    break;
                }
            }
            Err(_) => {
                // Connection refused — the replica is gone. A real edge drops it from the ring; the
                // scenario that killed it re-places the targets itself.
                last_status = 0;
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Placement::Exhausted {
        last_status,
        waited: began.elapsed(),
    }
}

/// Place a connection against a fixed target set, for a caller that holds its own copy of the ring.
///
/// A soak worker routes itself: it knows the edge's *rule*, not the edge's current state, which is
/// exactly the position a real client is in while a deploy moves replicas underneath it.
pub async fn place_among(
    targets: &[Target],
    session_id: &str,
    grant: &str,
    budget: Duration,
) -> Placement {
    if targets.is_empty() {
        return Placement::NoTarget;
    }
    let began = Instant::now();
    let mut last_status = 0;
    while began.elapsed() < budget {
        // Walk the ring from the hashed position: during chaos the chosen replica may be gone, and a
        // real edge's health checks would have dropped it. Starting from the hash keeps a session
        // sticky to one replica whenever that replica is up.
        for candidate in ring_order(targets, session_id) {
            let port = candidate.port;
            match crate::workload::probe_session(port, session_id, grant).await {
                Ok(status) if status < 400 => {
                    return Placement::Served {
                        port,
                        waited: began.elapsed(),
                    };
                }
                Ok(status) => last_status = status,
                Err(_) => last_status = 0,
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    Placement::Exhausted {
        last_status,
        waited: began.elapsed(),
    }
}

/// The targets in ring order for `session_id`: its owner first, then the rest.
///
/// A client walks this during chaos because the replica it belongs on may be down. Starting at the
/// ring position keeps it sticky to one replica whenever that replica is up, which is what makes the
/// 503-and-retry behaviour converge instead of oscillating.
fn ring_order<'a>(targets: &'a [Target], session_id: &str) -> Vec<&'a Target> {
    let point = hash(session_id.as_bytes());
    let mut scored: Vec<(u64, &Target)> = targets
        .iter()
        .map(|t| {
            let best = (0..VNODES_PER_TARGET)
                .map(|v| hash(format!("{}#{v}", t.name).as_bytes()))
                .map(|at| at.wrapping_sub(point))
                .min()
                .unwrap_or(u64::MAX);
            (best, t)
        })
        .collect();
    scored.sort_by_key(|(d, _)| *d);
    scored.into_iter().map(|(_, t)| t).collect()
}
