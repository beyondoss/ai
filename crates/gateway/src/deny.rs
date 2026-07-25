//! Sparse per-tenant deny-set — the gateway's *entire* spend/fraud surface.
//!
//! Design (deliberate, see plan): the gateway only ever asks "is this tenant cut off?" and
//! default-**allows** on a miss. We hold **only the exceptions** (the cut-off tenants), so memory
//! is `O(denied)`, not `O(tenants)` — this scales to millions of tenants because `denied` stays a
//! tiny slice (a few MB even at 1M entries; a tenant id is 8 bytes). The gateway never decides
//! *why* a tenant is denied — the control plane writes/removes entries; we just enforce + log.
//!
//! TTL/auto-restore is handled by slipstream, not here: spend holds are written with a TTL to the
//! next budget reset, so they expire into a `Del` event that removes them; fraud holds have no TTL
//! (sticky). This struct only reflects current membership.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// The deny map's hasher. **Deliberately not** std's `RandomState` (SipHash-1-3).
///
/// HashDoS resistance exists to stop an attacker from *choosing* keys that collide. Here the key is
/// `identity.tenant_id`, which only ever comes out of `keyring.verify(raw_key)` — an Ed25519
/// signature check over a token minted by the control plane, which holds the only signing key. And
/// the ordering is load-bearing: `proxy::request_filter` verifies **first** and only then looks the
/// tenant up (`proxy.rs`, step 5), so an unforgeable id is the *only* thing that ever reaches this
/// map. An attacker cannot pick their tenant id, cannot enumerate ids to farm collisions, and never
/// gets an unverified id into the table at all — so SipHash's collision resistance buys nothing and
/// costs ~4x on a lookup that runs on **every managed request**.
///
/// Measured on the dev host (`benches/unit.rs`, `deny::reason_*`, 1M entries): SipHash 6.21 ns vs
/// Fx 1.49 ns single-threaded; 13.44 ns vs 3.40 ns at 16 threads (the gateway is many-core, so the
/// contended number is the one that matters).
///
/// **Do not "fix" this back to `RandomState`** without first breaking the property above — i.e. not
/// unless some path starts inserting or looking up a tenant id that a caller supplied and we did not
/// verify. If that ever happens, the hasher is the *second* thing to fix; the unverified id is the
/// first.
type DenyHasher = BuildHasherDefault<rustc_hash::FxHasher>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyReason {
    /// Over budget. Typically written with a TTL to the next reset → auto-restores.
    Spend,
    /// Abuse / fraud. Sticky (no TTL) until a human clears it.
    Fraud,
    /// Reason not recognized in the entry value — still denied (fail safe on the enforce side).
    Unknown,
}

impl DenyReason {
    /// HTTP status to return. 402 Payment Required for spend, 403 Forbidden for fraud/other —
    /// gives the client (and our own dashboards) a meaningful signal without leaking detail.
    pub fn http_status(self) -> u16 {
        match self {
            DenyReason::Spend => 402,
            DenyReason::Fraud | DenyReason::Unknown => 403,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct DenySet {
    denied: HashMap<u64, DenyReason, DenyHasher>,
}

impl DenySet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-size for a known entry count — for the seed/rescan path, which knows how many entries it
    /// is about to insert and would otherwise rehash the whole table a dozen times on the way up.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            denied: HashMap::with_capacity_and_hasher(n, DenyHasher::default()),
        }
    }

    /// Default-allow: absence from the set = allowed. This is the safe-for-availability default —
    /// a tenant we've never heard of is served, not blocked.
    pub fn is_denied(&self, tenant_id: u64) -> bool {
        self.denied.contains_key(&tenant_id)
    }

    pub fn reason(&self, tenant_id: u64) -> Option<DenyReason> {
        self.denied.get(&tenant_id).copied()
    }

    pub fn insert(&mut self, tenant_id: u64, reason: DenyReason) {
        self.denied.insert(tenant_id, reason);
    }

    pub fn remove(&mut self, tenant_id: u64) {
        self.denied.remove(&tenant_id);
    }

    pub fn len(&self) -> usize {
        self.denied.len()
    }

    pub fn is_empty(&self) -> bool {
        self.denied.is_empty()
    }
}

/// The seed/rescan constructor (`store_watch::denyset_from_entries` builds the whole set with
/// `.collect()`). The hasher is an implementation detail of `DenySet`, deliberately not a type
/// parameter, so this signature — and every call site — is unaffected by the choice above.
impl FromIterator<(u64, DenyReason)> for DenySet {
    fn from_iter<I: IntoIterator<Item = (u64, DenyReason)>>(iter: I) -> Self {
        Self {
            denied: iter.into_iter().collect(),
        }
    }
}

/// Parse a slipstream deny key `blackhole.{tenant_id}` → tenant id. Returns `None` for keys that
/// don't match (so an unrelated watched key never corrupts the set).
pub fn parse_key(key: &str) -> Option<u64> {
    key.strip_prefix("blackhole.")?.parse().ok()
}

/// Parse the entry value into a reason. Accepts either a bare token (`spend`/`fraud`) or a JSON
/// object `{"reason":"spend", ...}`. Anything else → `Unknown` (still denied — fail safe).
pub fn parse_reason(value: &[u8]) -> DenyReason {
    let s = std::str::from_utf8(value).unwrap_or("").trim();
    // The JSON branch must own its extracted reason (it's borrowed from a temporary `Value`); the
    // bare-token branch matches the borrowed `&str` directly — no allocation on the common path.
    let json_reason: Option<String>;
    let token: &str = if s.starts_with('{') {
        json_reason = serde_json::from_slice::<serde_json::Value>(value)
            .ok()
            .and_then(|v| v.get("reason").and_then(|r| r.as_str()).map(str::to_owned));
        json_reason.as_deref().unwrap_or("")
    } else {
        s
    };
    match token {
        "spend" => DenyReason::Spend,
        "fraud" => DenyReason::Fraud,
        _ => DenyReason::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_allows_unknown_tenants() {
        let set = DenySet::new();
        assert!(!set.is_denied(12345));
    }

    #[test]
    fn insert_remove_and_reason() {
        let mut set = DenySet::new();
        set.insert(1, DenyReason::Spend);
        set.insert(2, DenyReason::Fraud);
        assert!(set.is_denied(1));
        assert_eq!(set.reason(1), Some(DenyReason::Spend));
        assert_eq!(set.reason(2).unwrap().http_status(), 403);
        set.remove(1);
        assert!(!set.is_denied(1)); // restored
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn key_parsing() {
        assert_eq!(parse_key("blackhole.42"), Some(42));
        assert_eq!(parse_key("blackhole.notanumber"), None);
        assert_eq!(parse_key("signkey.1"), None);
    }

    #[test]
    fn reason_parsing_bare_and_json() {
        assert_eq!(parse_reason(b"spend"), DenyReason::Spend);
        assert_eq!(parse_reason(b" fraud "), DenyReason::Fraud);
        assert_eq!(
            parse_reason(br#"{"reason":"spend","exp":123}"#),
            DenyReason::Spend
        );
        assert_eq!(parse_reason(b"weird"), DenyReason::Unknown);
    }

    #[test]
    fn spend_is_402_fraud_is_403() {
        assert_eq!(DenyReason::Spend.http_status(), 402);
        assert_eq!(DenyReason::Fraud.http_status(), 403);
        // Unknown is fail-safe: an unrecognized reason still denies, and maps to 403 like fraud
        // (not 402) — so a control-plane reason we don't yet parse can't be mistaken for a billing
        // block or, worse, leak through as an allow.
        assert_eq!(DenyReason::Unknown.http_status(), 403);
    }
}
