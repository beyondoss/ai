//! Per-key request-rate guardrail — a blast-radius circuit breaker, **not** a spend control.
//!
//! The deny-set (see `deny`) is the spend/fraud authority, but it's *cumulative* and reacts on a
//! lag: it only learns of spend after usage facts round-trip through the control plane, and it's
//! structurally blind to request floods that never bill — auth failures (rejected here, never reach
//! upstream), provider 4xx, and BYO traffic (on the caller's own key, no Beyond identity). This caps
//! the *velocity* a single key can drive, which bounds two things the deny-set can't: (1) spend from
//! a leaked/runaway managed key during the deny-set's reaction lag, and (2) the gateway-resource cost
//! (verifies, sockets, upstream connections) of a failure flood — the classic internal-service
//! incident: a buggy client in a retry storm.
//!
//! It is deliberately generous: a ceiling well above any legitimate single-tenant steady state, so
//! it never trips in normal operation. Tune it from `ai_rejections_total{reason="rate_limit"}`.
//!
//! Backed by pingora-limits' `Rate`: a count-min-sketch estimator with **fixed memory regardless of
//! key cardinality** (no per-tenant entry, no background GC), matching the deny-set's O(denied)
//! ethos. The sketch can *over*estimate a key's rate on hash collision but never under, so the cap
//! is always enforced; `SLOTS` is sized wide enough that overestimation stays negligible at our
//! active-key counts.

use pingora_limits::rate::Rate;
use std::hash::Hash;
use std::time::Duration;

/// Count-min sketch dimensions. Wider than `Rate::new`'s 1024-slot default because our key
/// cardinality (active tenants + BYO callers within a 1s window) is high; more slots keeps
/// collision-driven overestimation negligible. ~8192·4·2 atomic counters — a few hundred KB, fixed.
const SLOTS: usize = 8192;
const HASHES: usize = 4;

/// The rate window. The ceiling is expressed per this interval, i.e. requests/second.
const WINDOW: Duration = Duration::from_secs(1);

/// What a single request is charged against. Managed traffic is keyed by tenant, so one tenant's
/// runaway can't throttle another; BYO has no Beyond identity, so it's keyed by a hash of the
/// caller's own token. One key space — the enum discriminant keeps a `tenant_id` from colliding with
/// a BYO token hash that happens to share its value.
#[derive(Hash)]
pub enum RlKey {
    Tenant(u64),
    Byo(u64),
}

impl RlKey {
    /// Key a BYO request by a hash of its raw token (we have no tenant identity for BYO). The token
    /// itself is never stored — only this digest, which the sketch hashes again into its slots.
    pub fn byo(raw_token: &str) -> Self {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        raw_token.hash(&mut h);
        RlKey::Byo(h.finish())
    }
}

pub struct RateLimit {
    rate: Rate,
    /// Max requests per `WINDOW` for a single key before we start rejecting.
    max_per_window: isize,
}

impl RateLimit {
    /// `rps` is the per-key requests/second ceiling. `rps == 0` disables the limiter (`None`).
    pub fn new(rps: u32) -> Option<Self> {
        if rps == 0 {
            return None;
        }
        Some(Self {
            rate: Rate::new_with_estimator_config(WINDOW, HASHES, SLOTS),
            max_per_window: rps as isize,
        })
    }

    /// Charge one request to `key`. Returns `true` when it's within budget, `false` once the key has
    /// exceeded its ceiling in the current window. `observe` counts the event and returns the running
    /// total for the window, so the very request that crosses the line is the first one rejected.
    pub fn check(&self, key: &RlKey) -> bool {
        self.rate.observe(key, 1) <= self.max_per_window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rps_disables() {
        assert!(RateLimit::new(0).is_none());
    }

    #[test]
    fn allows_up_to_ceiling_then_rejects() {
        let rl = RateLimit::new(5).unwrap();
        let k = RlKey::Tenant(1);
        for _ in 0..5 {
            assert!(rl.check(&k));
        }
        // 6th request in the same 1s window crosses the ceiling.
        assert!(!rl.check(&k));
    }

    #[test]
    fn keys_have_independent_budgets() {
        let rl = RateLimit::new(2).unwrap();
        assert!(rl.check(&RlKey::Tenant(1)));
        assert!(rl.check(&RlKey::Tenant(1)));
        assert!(!rl.check(&RlKey::Tenant(1))); // tenant 1 exhausted
        assert!(rl.check(&RlKey::Tenant(2))); // a different tenant is unaffected
        // Same numeric value, different variant ⇒ different key (discriminant disambiguates).
        assert!(rl.check(&RlKey::Byo(1)));
    }
}
