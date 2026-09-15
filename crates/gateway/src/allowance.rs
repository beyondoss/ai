//! Sparse remaining-ok / exhausted set — the gateway's quota bit, not a price table.
//!
//! Design: the control plane writes when a tenant or `bai_v2` credential is **exhausted**.
//! Membership is the bit; we do not interpret a remaining counter. After the watcher has seeded,
//! absence = remaining-ok. Until then the set is **unready** and every managed request 402s
//! (fail-closed). That is the opposite of [`crate::deny`], which default-allows on a miss so a
//! NATS blip cannot take the fleet down. Quota must not serve through an unread store.
//!
//! Two memberships, one set: `allowance.{tenant}` exhausts every key for that tenant;
//! `allowance.key.{id}` exhausts one `bai_v2` credential. A request is blocked if **either**
//! matches. v1 tokens have no `key_id`, so only the tenant grain can exhaust them.
//!
//! The hasher is the same FxHasher as the deny-set, for the same reason: ids only reach this map
//! after Ed25519 verify, so SipHash buys nothing on a lookup that runs on every managed request.

use std::collections::HashSet;
use std::hash::BuildHasherDefault;

type AllowanceHasher = BuildHasherDefault<rustc_hash::FxHasher>;

/// Who an `allowance.*` entry names. Parsed from the KV key; the value is ignored (presence =
/// exhausted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowanceTarget {
    Tenant(u64),
    Key(u64),
}

/// Why a managed request did not pass the allowance check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowanceReject {
    /// Control plane marked this tenant or key exhausted.
    Exhausted,
    /// The set has not been read yet (no successful scan or snapshot). Fail-closed.
    Unavailable,
}

/// Sparse exhausted-id set plus the boot-ready flag.
///
/// `ready == false` is the cold-boot default: we have not yet stored a scan or snapshot, so we
/// cannot tell remaining-ok from "the store is unread". `from_entries` (even an empty scan)
/// flips it to `true`.
#[derive(Debug, Clone)]
pub struct AllowanceSet {
    ready: bool,
    tenants: HashSet<u64, AllowanceHasher>,
    keys: HashSet<u64, AllowanceHasher>,
}

impl Default for AllowanceSet {
    fn default() -> Self {
        Self {
            ready: false,
            tenants: HashSet::with_hasher(AllowanceHasher::default()),
            keys: HashSet::with_hasher(AllowanceHasher::default()),
        }
    }
}

impl AllowanceSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed constructor: a successful scan or snapshot, including an empty one.
    pub fn from_ready() -> Self {
        Self {
            ready: true,
            tenants: HashSet::with_hasher(AllowanceHasher::default()),
            keys: HashSet::with_hasher(AllowanceHasher::default()),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// A successful store read. Empty membership after this is remaining-ok, not unready.
    pub fn mark_ready(&mut self) {
        self.ready = true;
    }

    /// Remaining-ok if ready and neither id is a member. Unready → [`AllowanceReject::Unavailable`].
    pub fn reason_for(&self, tenant_id: u64, key_id: Option<u64>) -> Option<AllowanceReject> {
        if !self.ready {
            return Some(AllowanceReject::Unavailable);
        }
        if key_id.is_some_and(|id| self.keys.contains(&id)) {
            return Some(AllowanceReject::Exhausted);
        }
        if self.tenants.contains(&tenant_id) {
            return Some(AllowanceReject::Exhausted);
        }
        None
    }

    pub fn insert_target(&mut self, target: AllowanceTarget) {
        match target {
            AllowanceTarget::Tenant(t) => {
                self.tenants.insert(t);
            }
            AllowanceTarget::Key(k) => {
                self.keys.insert(k);
            }
        }
    }

    pub fn remove_target(&mut self, target: AllowanceTarget) {
        match target {
            AllowanceTarget::Tenant(t) => {
                self.tenants.remove(&t);
            }
            AllowanceTarget::Key(k) => {
                self.keys.remove(&k);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tenants.len() + self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tenants.is_empty() && self.keys.is_empty()
    }
}

/// Parse a slipstream allowance key.
///
/// - `allowance.{tenant_id}` → [`AllowanceTarget::Tenant`]
/// - `allowance.key.{id}` → [`AllowanceTarget::Key`]
///
/// Both live under the `allowance.` watch prefix. `None` for anything else. The `key.` form is
/// checked first — otherwise `allowance.key.42` would be a failed tenant parse and silently dropped.
pub fn parse_key(key: &str) -> Option<AllowanceTarget> {
    let rest = key.strip_prefix("allowance.")?;
    if let Some(id) = rest.strip_prefix("key.") {
        return Some(AllowanceTarget::Key(id.parse().ok()?));
    }
    Some(AllowanceTarget::Tenant(rest.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unready_blocks_everyone() {
        let set = AllowanceSet::new();
        assert!(!set.is_ready());
        assert_eq!(set.reason_for(1, None), Some(AllowanceReject::Unavailable));
        assert_eq!(
            set.reason_for(1, Some(9)),
            Some(AllowanceReject::Unavailable)
        );
    }

    #[test]
    fn empty_ready_set_is_remaining_ok() {
        let set = AllowanceSet::from_ready();
        assert!(set.is_ready());
        assert!(set.is_empty());
        assert_eq!(set.reason_for(1, None), None);
        assert_eq!(set.reason_for(1, Some(9)), None);
    }

    #[test]
    fn key_or_tenant_exhaust() {
        let mut set = AllowanceSet::from_ready();
        set.insert_target(AllowanceTarget::Key(100));
        // Same tenant, two credentials: only the named key is cut off.
        assert_eq!(
            set.reason_for(1, Some(100)),
            Some(AllowanceReject::Exhausted)
        );
        assert_eq!(set.reason_for(1, Some(101)), None);
        assert_eq!(set.reason_for(1, None), None); // v1: no key_id, tenant not exhausted
        set.insert_target(AllowanceTarget::Tenant(1));
        // Tenant exhaust kills every key for that tenant, including one already key-exhausted.
        assert_eq!(
            set.reason_for(1, Some(100)),
            Some(AllowanceReject::Exhausted)
        );
        assert_eq!(
            set.reason_for(1, Some(101)),
            Some(AllowanceReject::Exhausted)
        );
        assert_eq!(set.reason_for(1, None), Some(AllowanceReject::Exhausted));
        assert_eq!(
            set.reason_for(2, Some(100)),
            Some(AllowanceReject::Exhausted)
        ); // key still exhausted
        set.remove_target(AllowanceTarget::Key(100));
        assert_eq!(set.reason_for(2, Some(100)), None);
        set.remove_target(AllowanceTarget::Tenant(1));
        assert_eq!(set.reason_for(1, Some(101)), None);
    }

    #[test]
    fn key_parsing() {
        assert_eq!(parse_key("allowance.42"), Some(AllowanceTarget::Tenant(42)));
        assert_eq!(parse_key("allowance.key.7"), Some(AllowanceTarget::Key(7)));
        assert_eq!(parse_key("allowance.notanumber"), None);
        assert_eq!(parse_key("allowance.key.notanumber"), None);
        assert_eq!(parse_key("allowance.key"), None);
        assert_eq!(parse_key("blackhole.42"), None);
        assert_eq!(parse_key("aicapture.1"), None);
    }
}
