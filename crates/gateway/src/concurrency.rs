//! Per-tenant in-flight cap: the bound on how far a tenant can overspend.
//!
//! Spend is enforced after the fact. A request's `ai.usage` row ships through logfwd → ClickHouse →
//! the control plane, which writes the tenant's `allowance.*` exhaust bit back through NATS; until
//! that bit lands, the tenant keeps being served. The overshoot is therefore
//! `lag × in-flight requests × cost per request`, and of those three only in-flight is something the
//! gateway can bound. This is that bound — a ceiling on requests one tenant holds open at once — so
//! the overshoot is a number an operator can price rather than whatever a runaway agent fleet
//! happens to reach.
//!
//! It is **per process**, like the cache and the TTFT ranker: N replicas admit up to N × the limit.
//! It is not a rate limit either — a tenant making short requests at a high rate never hits it,
//! and should not, because short requests are not where overspend comes from.
//!
//! State is exact, not a sketch: two tenants must never share a ceiling. Sparse — a tenant with
//! nothing in flight has no entry — and sharded so tenants on different shards never contend.

use rustc_hash::FxHashMap;
use std::sync::{Mutex, MutexGuard};

/// Shard count. A power of two so the shard is a shift of a multiplicative hash. 64 shards keep
/// contention negligible at any realistic core count; each is one small map.
const SHARDS: usize = 64;

pub struct TenantSlots {
    limit: u32,
    shards: [Mutex<FxHashMap<u64, u32>>; SHARDS],
}

impl TenantSlots {
    /// `None` when `limit == 0` — the operator's off switch, and the default.
    pub fn new(limit: u32) -> Option<Self> {
        (limit > 0).then(|| Self {
            limit,
            shards: std::array::from_fn(|_| Mutex::new(FxHashMap::default())),
        })
    }

    /// Claim a slot for `tenant`. `false` means the tenant is at its ceiling and nothing was
    /// claimed. Every `true` must be paired with exactly one [`Self::release`].
    pub fn try_acquire(&self, tenant: u64) -> bool {
        let mut shard = self.shard(tenant);
        let n = shard.entry(tenant).or_insert(0);
        if *n >= self.limit {
            return false;
        }
        *n += 1;
        true
    }

    pub fn release(&self, tenant: u64) {
        let mut shard = self.shard(tenant);
        if let Some(n) = shard.get_mut(&tenant) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                shard.remove(&tenant);
            }
        }
    }

    /// Requests `tenant` currently holds open. For tests and diagnostics.
    pub fn in_flight(&self, tenant: u64) -> u32 {
        self.shard(tenant).get(&tenant).copied().unwrap_or(0)
    }

    fn shard(&self, tenant: u64) -> MutexGuard<'_, FxHashMap<u64, u32>> {
        // Fibonacci hashing: tenant ids are often sequential, and the top bits of the product
        // spread them evenly where the low bits would not.
        let i =
            (tenant.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - SHARDS.trailing_zeros())) as usize;
        // A poisoned shard only means another thread panicked mid-update of a counter map; the
        // map itself is still a valid map. Refusing every later request for those tenants would
        // turn one bug into an outage.
        match self.shards[i].lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_off() {
        assert!(TenantSlots::new(0).is_none());
    }

    #[test]
    fn a_tenant_is_capped_and_recovers_on_release() {
        let slots = TenantSlots::new(2).expect("on");
        assert!(slots.try_acquire(7));
        assert!(slots.try_acquire(7));
        assert!(
            !slots.try_acquire(7),
            "third concurrent request is over the cap"
        );
        assert_eq!(slots.in_flight(7), 2, "a refused acquire claims nothing");
        slots.release(7);
        assert!(slots.try_acquire(7));
    }

    #[test]
    fn tenants_do_not_share_a_ceiling() {
        let slots = TenantSlots::new(1).expect("on");
        for tenant in 0..10_000 {
            assert!(
                slots.try_acquire(tenant),
                "tenant {tenant} has its own slot"
            );
        }
    }

    #[test]
    fn released_tenants_leave_no_entry() {
        let slots = TenantSlots::new(3).expect("on");
        assert!(slots.try_acquire(42));
        slots.release(42);
        slots.release(42); // an extra release is a no-op, not an underflow
        assert_eq!(slots.in_flight(42), 0);
        let entries: usize = slots
            .shards
            .iter()
            .map(|s| s.lock().map(|m| m.len()).unwrap_or(0))
            .sum();
        assert_eq!(entries, 0, "sparse: nothing in flight means nothing stored");
    }
}
