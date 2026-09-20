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

    /// Replace the routable set — a deploy, a scale-in, a replica that died.
    pub fn set_targets(&mut self, targets: Vec<Target>) {
        self.targets = targets;
    }

    /// Which replica this session id hashes to.
    ///
    /// A ring hash, not round-robin: the retry after a `503` has to come back to the **same**
    /// replica, so it waits for that session's lock to free rather than wandering the group and
    /// collecting a 503 from each member in turn. Deterministic on (id, target set) alone — no
    /// shared state, no lookup, which is exactly why the agent needs to know nothing about it.
    pub fn route(&self, session_id: &str) -> Option<&Target> {
        if self.targets.is_empty() {
            return None;
        }
        // FNV-1a over the id, then modulo the ring. Small and stable; a production edge would use
        // ketama for smoother movement when the set changes, which affects *how many* keys move on a
        // deploy, not whether the retry is deterministic.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in session_id.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let idx = (h % self.targets.len() as u64) as usize;
        self.targets.get(idx)
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
