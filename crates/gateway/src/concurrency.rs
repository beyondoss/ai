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
//! It is **per process**, like the cache and the circuit breaker: N replicas admit up to N × the limit.
//! It is not a rate limit either — a tenant making short requests at a high rate never hits it,
//! and should not, because short requests are not where overspend comes from.
//!
//! State is exact, not a sketch: two tenants must never share a ceiling. Sparse — a tenant with
//! nothing in flight has no entry — and sharded so tenants on different shards never contend.

use crate::key::Identity;
use crate::sync::{AtomicUsize, Mutex, MutexGuard, Ordering};
use rustc_hash::FxHashMap;

/// The process-wide budget for request bodies the gateway holds in memory (`max_buffered_body_bytes`).
///
/// A body past pingora's 64 KiB replay buffer is read in full before a catalog walk (and buffered
/// again by the `FullBody` re-run that carries it), and a managed OpenAI chat body is buffered for
/// the `stream_options` splice. Each is capped at 100 MiB, but nothing capped how many: eight
/// concurrent 90 MiB uploads grew the process by ~1.4 GiB. Every such buffer reserves its bytes
/// here first, and a request that would cross the budget is refused with a 503 + `Retry-After`
/// before its body is read (or as soon as a chunked one grows past it) rather than letting the
/// process grow until the kernel kills it. Small bodies (≤ the replay buffer) are not counted: they
/// are bounded by concurrency and are the common case, which then pays no shared atomic.
pub struct BodyBudget {
    limit: usize,
    used: AtomicUsize,
}

impl BodyBudget {
    /// `None` when `limit == 0`: the budget is off.
    pub fn new(limit: usize) -> Option<Self> {
        (limit > 0).then(|| Self {
            limit,
            used: AtomicUsize::new(0),
        })
    }

    /// Reserve `n` bytes. `false` (and nothing reserved) when that would cross the limit. Every
    /// `true` must be paired with a [`Self::release`] of the same `n`.
    pub fn try_reserve(&self, n: usize) -> bool {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(n).filter(|total| *total <= self.limit)
            })
            .is_ok()
    }

    pub fn release(&self, n: usize) {
        let _ = self
            .used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                Some(used.saturating_sub(n))
            });
    }

    /// Bytes reserved right now. For tests and diagnostics.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// The whole budget: a reservation larger than this can never be granted.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Reserve `n` bytes for as long as the returned hold lives; `None` (nothing reserved) when that
    /// would cross the limit. For memory held across one synchronous step, such as the `Value`s a
    /// translation builds (`translate::translation_heap`): the hold releases on drop, a panic
    /// included.
    pub fn hold(&self, n: usize) -> Option<BudgetHold<'_>> {
        self.try_reserve(n)
            .then_some(BudgetHold { budget: self, n })
    }
}

/// Bytes reserved in a [`BodyBudget`] until dropped. See [`BodyBudget::hold`].
pub struct BudgetHold<'a> {
    budget: &'a BodyBudget,
    n: usize,
}

impl Drop for BudgetHold<'_> {
    fn drop(&mut self) {
        self.budget.release(self.n);
    }
}

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

    /// One of `id`'s slots, held until the returned guard drops. `None` when the tenant is at its
    /// ceiling and nothing was claimed.
    pub fn take(&self, id: &Identity) -> Option<TenantSlot<'_>> {
        self.try_acquire(id.tenant_id).then_some(TenantSlot {
            slots: self,
            tenant: id.tenant_id,
        })
    }

    /// Claim a slot for `tenant`. `false` means the tenant is at its ceiling and nothing was
    /// claimed. Every `true` must be paired with exactly one [`Self::release`]: [`TenantSlot`] is
    /// what pairs them.
    fn try_acquire(&self, tenant: u64) -> bool {
        let mut shard = self.shard(tenant);
        let n = shard.entry(tenant).or_insert(0);
        if *n >= self.limit {
            return false;
        }
        *n += 1;
        true
    }

    fn release(&self, tenant: u64) {
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

/// One of a tenant's `tenant_max_in_flight` slots: taken by [`TenantSlots::take`] and given back
/// when dropped, so whichever way a request ends (an early refusal, `logging`, a drain that frees
/// it early, a panic unwinding the context) it is released exactly once. Move-only; whoever holds
/// it owns the slot.
#[must_use = "a slot dropped at once is released at once"]
pub struct TenantSlot<'a> {
    slots: &'a TenantSlots,
    tenant: u64,
}

impl Drop for TenantSlot<'_> {
    fn drop(&mut self) {
        self.slots.release(self.tenant);
    }
}

#[cfg(all(test, not(beyond_loom)))]
mod tests {
    use super::*;

    #[test]
    fn zero_is_off() {
        assert!(TenantSlots::new(0).is_none());
        assert!(BodyBudget::new(0).is_none());
    }

    #[test]
    fn the_body_budget_refuses_past_its_limit_and_recovers() {
        let budget = BodyBudget::new(100).expect("on");
        assert!(budget.try_reserve(60));
        assert!(!budget.try_reserve(41), "would cross the limit");
        assert_eq!(budget.used(), 60, "a refused reserve claims nothing");
        assert!(budget.try_reserve(40));
        budget.release(60);
        assert!(budget.try_reserve(60));
        budget.release(1_000);
        assert_eq!(
            budget.used(),
            0,
            "an over-release saturates, never underflows"
        );
    }

    /// claim: A3
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

/// Permutation models of the tenant slot and the body budget (`cargo test --lib loom` under
/// `--cfg beyond_loom`; `mise run test:loom`). Each explores every interleaving of two or three threads.
#[cfg(all(test, beyond_loom))]
mod loom_models {
    use super::*;
    use loom::sync::Arc;
    use loom::thread;

    /// However the acquires and releases interleave, a tenant never holds more than its ceiling,
    /// and when everyone is done nothing is held and nothing is stored.
    #[test]
    fn a_tenant_never_holds_more_than_its_ceiling() {
        loom::model(|| {
            let slots = Arc::new(TenantSlots::new(2).expect("on"));
            let held = Arc::new(AtomicUsize::new(0));
            let hs: Vec<_> = (0..3)
                .map(|_| {
                    let (slots, held) = (slots.clone(), held.clone());
                    thread::spawn(move || {
                        if slots.try_acquire(7) {
                            let now = held.fetch_add(1, Ordering::SeqCst) + 1;
                            assert!(now <= 2, "{now} slots held under a ceiling of 2");
                            assert!(slots.in_flight(7) <= 2);
                            held.fetch_sub(1, Ordering::SeqCst);
                            slots.release(7);
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            assert_eq!(slots.in_flight(7), 0, "every release freed its slot");
            assert!(
                slots.try_acquire(7) && slots.try_acquire(7),
                "no slot was lost"
            );
        });
    }

    /// A release racing an acquire at the ceiling either frees the slot in time or does not, but
    /// the slot is never lost: once the holder is done, the tenant can take it again.
    #[test]
    fn a_release_racing_an_acquire_loses_no_slot() {
        loom::model(|| {
            let slots = Arc::new(TenantSlots::new(1).expect("on"));
            assert!(slots.try_acquire(1));
            let releaser = {
                let slots = slots.clone();
                thread::spawn(move || slots.release(1))
            };
            let taker = {
                let slots = slots.clone();
                thread::spawn(move || {
                    if slots.try_acquire(1) {
                        slots.release(1);
                    }
                })
            };
            releaser.join().unwrap();
            taker.join().unwrap();
            assert_eq!(slots.in_flight(1), 0);
            assert!(slots.try_acquire(1), "the freed slot can be taken");
        });
    }

    /// The reserved total never passes the limit, and what is held at the end is exactly what the
    /// granted reservations asked for.
    #[test]
    fn the_body_budget_never_over_reserves() {
        loom::model(|| {
            let budget = Arc::new(BodyBudget::new(100).expect("on"));
            let hs: Vec<_> = [60usize, 50, 40]
                .into_iter()
                .map(|n| {
                    let budget = budget.clone();
                    thread::spawn(move || {
                        let ok = budget.try_reserve(n);
                        assert!(budget.used() <= 100);
                        ok.then_some(n)
                    })
                })
                .collect();
            let granted: usize = hs.into_iter().filter_map(|h| h.join().unwrap()).sum();
            assert!(granted <= 100, "granted {granted} of 100");
            assert_eq!(budget.used(), granted, "a refusal reserved nothing");
        });
    }

    /// `reserve_body`'s shape: try a rounded-up target, fall back to the exact one, release exactly
    /// what was reserved. With two requests racing, the budget is never crossed and ends empty.
    #[test]
    fn a_reservation_released_returns_exactly_what_it_took() {
        loom::model(|| {
            let budget = Arc::new(BodyBudget::new(100).expect("on"));
            let hs: Vec<_> = [(80usize, 55usize), (70, 45)]
                .into_iter()
                .map(|(rounded, exact)| {
                    let budget = budget.clone();
                    thread::spawn(move || {
                        let mut held = 0;
                        for target in [rounded, exact] {
                            if budget.try_reserve(target) {
                                held = target;
                                break;
                            }
                        }
                        assert!(budget.used() <= 100);
                        if held > 0 {
                            budget.release(held);
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            assert_eq!(
                budget.used(),
                0,
                "release returned exactly what was reserved"
            );
        });
    }

    /// A hold frees its bytes on drop.
    #[test]
    fn a_dropped_hold_frees_its_bytes() {
        loom::model(|| {
            let budget = Arc::new(BodyBudget::new(100).expect("on"));
            let hs: Vec<_> = (0..2)
                .map(|_| {
                    let budget = budget.clone();
                    thread::spawn(move || {
                        if let Some(hold) = budget.hold(70) {
                            assert!(budget.used() <= 100);
                            drop(hold);
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            assert_eq!(budget.used(), 0);
            assert!(budget.hold(100).is_some(), "the whole budget is free again");
        });
    }
}
