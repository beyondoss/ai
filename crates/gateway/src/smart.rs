//! In-process catalog-walk ranking from observed time-to-first-byte.
//!
//! **Per-pod, not fleet-wide.** EWMA samples never leave this process. Two replicas of the same
//! catalog row can walk candidates in different orders; `x-beyond-split` is the only cross-replica
//! pin (hash of the request counter, not a shared rank). There is no Redis on this path.
//! `ai_smart_rank_scope{kind="process"}` is the metric that says so.
//!
//! A managed `/auto` or `/v1` request still walks one catalog row — no provider the row does not
//! already name. What changes is the **order** of that walk when the caller did not pin it
//! (`x-beyond-order` / `x-beyond-split`).
//!
//! Default with no samples is the row's static order. After a candidate answers, its EWMA TTFT is
//! the sort key: measured candidates go first (fastest first), unmeasured stay failover in catalog
//! order. A connect failure or 5xx is recorded as a penalty so a fast error does not outrank a
//! slower 2xx. A 429 is a real answer, not a penalty — same distinction the breaker already makes.
//!
//! A candidate whose **latest** attempt failed ranks behind every other candidate, unmeasured ones
//! included, until it answers again or its sample goes stale. This is what makes a client's own
//! retry a failover. A 5xx on a request body past pingora's 64 KiB replay buffer cannot be retried
//! in-gateway (see ARCHITECTURE.md), so it is relayed; the stock OpenAI/Anthropic SDKs retry 5xx
//! and 529 on their own, and that retry is a fresh request with a fresh body. Before this, the
//! penalty alone left a failing primary *measured* and so still ahead of a never-tried fallback —
//! the retry went straight back to the provider that had just failed.
//!
//! Unmeasured candidates would otherwise never run if the primary always succeeds, so every
//! [`PROBE_EVERY`]th request (skipping seq `0`) promotes the first unmeasured slot to primary.
//! Once every candidate on the row has a sample, the walk is pure EWMA. A sample older than
//! [`STALE`] is treated as unmeasured so a previously-slow arm can be retried.
//!
//! Integer EWMA (`7/8` previous + `1/8` sample) in microseconds, atomics only — the request path
//! does not allocate or take a lock. Per `(catalog row, candidate index)`, not per provider: Opus
//! on Bedrock is not Haiku on Bedrock.
//!
//! # Session pins
//!
//! Ranking decides where a **new** caller goes; a pin keeps an existing one there. Provider prompt
//! caches are per provider, so a walk that re-ranks every request moves an agent loop from
//! Anthropic to Bedrock and back, and every move re-reads the whole prefix at full input price
//! plus a cache write (1.25× vs 0.1× on Claude). Worse, the probe above lands on whichever request
//! happens to draw a multiple of [`PROBE_EVERY`] — usually someone mid-session.
//!
//! The pin key is the verified identity (`tenant_id`, `vpc_id`, `key_id`) plus the catalog row:
//! one virtual key is one app, and an app's sessions share system prompt and tool prefixes, so
//! keeping the whole key on one provider is what the provider cache wants. After a 2xx, the
//! serving candidate is pinned. While the pin is live, [`Router::rank`] puts that candidate first
//! and does not probe. A pin yields when its candidate's latest attempt failed (the walk then fails
//! over and the next 2xx re-pins), after [`PIN_IDLE_S`] without a 2xx (the provider cache has
//! expired anyway), and after [`PIN_MAX_AGE_S`] so a pin taken during an outage drifts back to the
//! ranked primary. An open breaker or a missing pool key needs no check here: `upstream_peer`
//! skips the candidate, the next one serves, and that 2xx re-pins.
//!
//! One fixed table of [`PIN_SLOTS`] packed `AtomicU64`s, direct-mapped by hash: no allocation per
//! request, no lock, no eviction pass. A collision overwrites; the cost is one re-rank. Pins are
//! per pod like the EWMA. A key whose requests land on two pods can hold two pins, but each pod
//! still stops bouncing it, and with a healthy primary both pods rank the same way.

use crate::control::Walk;
use providers::{MAX_CANDIDATES, ModelRoute, catalog::MODEL_ROUTES};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Promote the first unmeasured candidate to primary on `seq % PROBE_EVERY == 0` (and `seq != 0`).
/// Seq `0` is the first request of the process; leaving it on catalog order keeps a two-request
/// cache fill/hit on the same walk.
pub(crate) const PROBE_EVERY: u64 = 8;

/// A sample this old is ignored for ranking (the cell is still kept; the next observe continues
/// the EWMA). Long enough that a burst of traffic exploits, short enough that a recovered arm is
/// retried without an operator flipping a header.
const STALE: Duration = Duration::from_secs(30);

/// [`STALE`] in nanoseconds, so a rank does not re-convert it once per candidate.
const STALE_NS: u128 = STALE.as_nanos();

/// Cap on a stored sample. Matches the top [`crate::metrics`] TTFT bucket.
const MAX_US: u64 = 30_000_000;

/// Floor applied to a failed attempt (connect / 5xx) so a 2ms 500 cannot beat a 200ms 200.
const PENALTY_US: u64 = 5_000_000;

/// Session-pin table size. 16384 × 8 B = 128 KiB, fixed at boot. Direct-mapped, so this bounds how
/// many (key, model) pairs one pod can keep pinned before collisions start re-ranking some of them.
const PIN_SLOTS: usize = 1 << 14;

/// A pin with no 2xx for this long is dropped. Matches the default provider prompt-cache TTL
/// (Anthropic's 5-minute ephemeral cache; OpenAI's in-memory prefix cache is the same order): once
/// the cache is cold, sticking to its provider buys nothing.
pub(crate) const PIN_IDLE_S: u64 = 300;

/// A pin older than this is dropped even while in use, so a key pinned to a fallback during an
/// outage returns to the ranked primary. One re-rank per key per hour is one cache miss.
pub(crate) const PIN_MAX_AGE_S: u64 = 3600;

/// Packed pin word: `tag:24 | candidate:4 | created_s:18 | last_s:18`. Seconds are since [`BASE`],
/// modulo 2^18 (~72 h); ages are taken with wrapping subtraction, so they are exact for anything
/// under 72 h, far past both limits. A word of `0` is an empty slot (the tag is never zero).
const PIN_TIME_BITS: u32 = 18;
const PIN_TIME_MASK: u64 = (1 << PIN_TIME_BITS) - 1;
const PIN_IDX_SHIFT: u32 = 2 * PIN_TIME_BITS;
const PIN_TAG_SHIFT: u32 = PIN_IDX_SHIFT + 4;

static BASE: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Per-row EWMA table, parallel to [`MODEL_ROUTES`], plus the session-pin table.
pub struct Router {
    rows: Box<[Row]>,
    pins: Box<[AtomicU64]>,
}

struct Row {
    ewma_us: [AtomicU64; MAX_CANDIDATES],
    last_ns: [AtomicU64; MAX_CANDIDATES],
    /// The latest attempt against this candidate failed (connect failure or 5xx). Cleared by the
    /// next success; ignored once the sample is [`STALE`].
    failed: [AtomicBool; MAX_CANDIDATES],
}

impl Row {
    fn new() -> Self {
        Self {
            ewma_us: std::array::from_fn(|_| AtomicU64::new(0)),
            last_ns: std::array::from_fn(|_| AtomicU64::new(0)),
            failed: std::array::from_fn(|_| AtomicBool::new(false)),
        }
    }
}

/// How a candidate ranks: healthy measured (fastest first), then unmeasured (catalog order), then
/// candidates whose latest attempt failed (least bad first).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Tier {
    Healthy(u64),
    Unmeasured,
    Failed(u64),
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Router {
    pub fn new() -> Self {
        Self {
            rows: MODEL_ROUTES.iter().map(|_| Row::new()).collect(),
            pins: (0..PIN_SLOTS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Record an attempt against `route.candidates[catalog_idx]`.
    ///
    /// `ok` is a response the provider *answered* with (2xx / 3xx / 4xx, including 429). Connect
    /// failure and 5xx pass `ok = false` and take the penalty floor.
    pub fn observe(&self, route: &ModelRoute, catalog_idx: u8, elapsed_us: u64, ok: bool) {
        let Some((_, row)) = self.row(route) else {
            return;
        };
        let i = usize::from(catalog_idx);
        if i >= MAX_CANDIDATES {
            return;
        }
        let sample = if ok {
            elapsed_us.min(MAX_US)
        } else {
            elapsed_us.clamp(PENALTY_US, MAX_US)
        };
        let cell = &row.ewma_us[i];
        loop {
            let old = cell.load(Ordering::Relaxed);
            let new = ewma(old, sample);
            if cell
                .compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
        row.failed[i].store(!ok, Ordering::Relaxed);
        row.last_ns[i].store(now_ns(), Ordering::Relaxed);
    }

    /// Reorder `walk`: a live session pin first, else EWMA order, else probe an unmeasured arm.
    /// Identity when the row is unknown or `walk` is empty. The `bool` is whether a pin decided
    /// the primary. `affinity` is [`affinity`]'s hash of the caller, `None` for no pinning.
    pub fn rank(
        &self,
        walk: Walk,
        route: &ModelRoute,
        seed: u64,
        affinity: Option<u64>,
    ) -> (Walk, bool) {
        if walk.len == 0 {
            return (walk, false);
        }
        let Some((row_i, row)) = self.row(route) else {
            return (walk, false);
        };
        // One clock read for every candidate. `effective` used to call `now_ns` itself, so a
        // row with three arms took three vDSO reads to answer one ranking.
        let now = now_ns();
        let score = |orig| self.effective(row, orig, now);
        if let Some(aff) = affinity
            && let Some(pinned) = self.pinned(pin_key(aff, row_i), now)
            && walk.indices[..usize::from(walk.len)].contains(&pinned)
            && !matches!(score(pinned), Tier::Failed(_))
        {
            // The rest keeps its EWMA order as failover. No probe: that would move this caller.
            return (to_front(sort_measured(walk, score), pinned), true);
        }
        if seed != 0 && seed.is_multiple_of(PROBE_EVERY) {
            return (probe(walk, score), false);
        }
        (sort_measured(walk, score), false)
    }

    /// Pin `affinity`'s caller on `route` to `route.candidates[catalog_idx]`. Called after a 2xx.
    /// A live pin to the same candidate only refreshes its last-used second (and skips the store
    /// entirely when that second has not changed, so a hot key does not bounce the cache line on
    /// every response).
    pub fn pin(&self, route: &ModelRoute, affinity: u64, catalog_idx: u8) {
        if usize::from(catalog_idx) >= MAX_CANDIDATES {
            return;
        }
        let Some((row_i, _)) = self.row(route) else {
            return;
        };
        let key = pin_key(affinity, row_i);
        let slot = &self.pins[slot_of(key)];
        let tag = tag_of(key);
        let now = secs(now_ns());
        let old = slot.load(Ordering::Relaxed);
        let created =
            if old >> PIN_TAG_SHIFT == tag && pin_idx(old) == catalog_idx && pin_live(old, now) {
                if old & PIN_TIME_MASK == now {
                    return;
                }
                (old >> PIN_TIME_BITS) & PIN_TIME_MASK
            } else {
                now
            };
        slot.store(pack(tag, catalog_idx, created, now), Ordering::Relaxed);
    }

    /// The live pin for `key`, if any.
    fn pinned(&self, key: u64, now_ns: u64) -> Option<u8> {
        let w = self.pins[slot_of(key)].load(Ordering::Relaxed);
        (w >> PIN_TAG_SHIFT == tag_of(key) && pin_live(w, secs(now_ns))).then(|| pin_idx(w))
    }

    fn row(&self, route: &ModelRoute) -> Option<(usize, &Row)> {
        // Name search, not the row's address. `MODEL_ROUTES` is a `const` slice, so each use
        // site can hold its own copy — pointer arithmetic against `as_ptr()` does not land on
        // the `&'static` row `for_model` returned.
        let i = MODEL_ROUTES
            .binary_search_by(|r| r.model.cmp(route.model))
            .ok()?;
        Some((i, self.rows.get(i)?))
    }

    fn effective(&self, row: &Row, catalog_idx: u8, now: u64) -> Tier {
        let i = usize::from(catalog_idx);
        if i >= MAX_CANDIDATES {
            return Tier::Unmeasured;
        }
        let ewma = row.ewma_us[i].load(Ordering::Relaxed);
        if ewma == 0 {
            return Tier::Unmeasured;
        }
        let last = row.last_ns[i].load(Ordering::Relaxed);
        if last != 0 && u128::from(now.saturating_sub(last)) > STALE_NS {
            return Tier::Unmeasured;
        }
        if row.failed[i].load(Ordering::Relaxed) {
            Tier::Failed(ewma)
        } else {
            Tier::Healthy(ewma)
        }
    }
}

fn ewma(old: u64, sample: u64) -> u64 {
    if old == 0 {
        sample
    } else {
        // 7/8 previous + 1/8 sample. Shift is exact for the integer contract and cheaper than a
        // float on a path that runs once per attempt, not once per chunk.
        old - (old >> 3) + (sample >> 3)
    }
}

fn now_ns() -> u64 {
    BASE.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// The pin identity for a verified managed caller. A virtual key is deterministic per
/// `(tenant, app)`, so `(tenant_id, vpc_id, key_id)` is one app's key. Not keyed: tenant ids are
/// verified before this runs, so nobody can aim collisions at a slot, and a collision only costs a
/// re-rank anyway.
pub fn affinity(tenant_id: u64, vpc_id: u64, key_id: Option<u64>) -> u64 {
    let mut h = mix(tenant_id ^ 0x243F_6A88_85A3_08D3);
    h = mix(h ^ vpc_id);
    mix(h ^ key_id.map_or(0, |k| k ^ 0x1319_8A2E_0370_7344))
}

/// `splitmix64`'s finalizer: every input bit reaches every output bit.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn pin_key(affinity: u64, row_i: usize) -> u64 {
    mix(affinity ^ (row_i as u64).wrapping_mul(0xA076_1D64_78BD_642F))
}

/// Low bits pick the slot; the tag comes from the high bits, so the two are independent.
fn slot_of(key: u64) -> usize {
    (key as usize) & (PIN_SLOTS - 1)
}

/// 24-bit tag, never zero (zero marks an empty slot).
fn tag_of(key: u64) -> u64 {
    (key >> PIN_TAG_SHIFT).max(1)
}

fn secs(now_ns: u64) -> u64 {
    (now_ns / 1_000_000_000) & PIN_TIME_MASK
}

fn pack(tag: u64, idx: u8, created: u64, last: u64) -> u64 {
    (tag << PIN_TAG_SHIFT)
        | (u64::from(idx) << PIN_IDX_SHIFT)
        | ((created & PIN_TIME_MASK) << PIN_TIME_BITS)
        | (last & PIN_TIME_MASK)
}

fn pin_idx(w: u64) -> u8 {
    ((w >> PIN_IDX_SHIFT) & 0xF) as u8
}

fn pin_live(w: u64, now: u64) -> bool {
    let last = w & PIN_TIME_MASK;
    let created = (w >> PIN_TIME_BITS) & PIN_TIME_MASK;
    let idle = now.wrapping_sub(last) & PIN_TIME_MASK;
    let age = now.wrapping_sub(created) & PIN_TIME_MASK;
    w != 0 && idle <= PIN_IDLE_S && age <= PIN_MAX_AGE_S
}

/// Sort by [`Tier`]: healthy measured fastest-first, then unmeasured, then failed. Ties keep the
/// order the walk already had, so unmeasured candidates stay in catalog (or header) order.
fn sort_measured(walk: Walk, score: impl Fn(u8) -> Tier) -> Walk {
    let n = usize::from(walk.len);
    let mut keyed = [(Tier::Unmeasured, 0u8, 0u8); MAX_CANDIDATES];
    for (slot, (key, &orig)) in keyed.iter_mut().zip(&walk.indices[..n]).enumerate() {
        *key = (score(orig), slot as u8, orig);
    }
    keyed[..n].sort_unstable_by_key(|&(tier, slot, _)| (tier, slot));
    let mut out = Walk {
        indices: [0u8; MAX_CANDIDATES],
        len: walk.len,
    };
    for (dst, &(_, _, orig)) in out.indices.iter_mut().zip(&keyed[..n]) {
        *dst = orig;
    }
    out
}

/// Move the first unmeasured catalog index to slot 0; leave the rest in order. No-op when every
/// slot is already measured (exploit-only) or the unmeasured one is already primary.
fn probe(walk: Walk, score: impl Fn(u8) -> Tier) -> Walk {
    let n = usize::from(walk.len);
    match walk.indices[..n]
        .iter()
        .find(|&&orig| score(orig) == Tier::Unmeasured)
    {
        Some(&orig) => to_front(walk, orig),
        None => walk,
    }
}

/// Move catalog index `orig` to slot 0; the others keep their relative order.
fn to_front(walk: Walk, orig: u8) -> Walk {
    let n = usize::from(walk.len);
    let Some(pos) = walk.indices[..n].iter().position(|&i| i == orig) else {
        return walk;
    };
    let mut out = walk;
    out.indices.copy_within(0..pos, 1);
    out.indices[0] = orig;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Walk;
    use providers::by_id;

    fn opus() -> &'static ModelRoute {
        providers::for_model("claude-opus-4-8").expect("catalog row")
    }

    fn names(walk: Walk, row: &ModelRoute) -> Vec<&'static str> {
        (0..walk.len)
            .map(|i| {
                let orig = walk.indices[i as usize];
                by_id(row.candidates[orig as usize].provider).name
            })
            .collect()
    }

    #[test]
    fn no_samples_keeps_catalog_order() {
        let r = Router::new();
        let row = opus();
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1, None).0;
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    /// claim: R7
    #[test]
    fn faster_measured_candidate_is_tried_first() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 2_000_000, true);
        r.observe(row, 1, 100_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1, None).0;
        assert_eq!(names(walk, row), ["bedrock", "anthropic", "openrouter"]);
    }

    #[test]
    fn unmeasured_stay_failover_so_a_working_primary_is_not_abandoned() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1, None).0;
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    #[test]
    fn failed_attempt_takes_the_penalty_floor() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 1_000, false);
        r.observe(row, 1, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1, None).0;
        assert_eq!(names(walk, row)[0], "bedrock");
    }

    /// The client-retry failover: a primary that just 5xx'd must not stay ahead of a fallback that
    /// has never been tried, or the SDK's retry goes straight back to it.
    #[test]
    fn a_failed_primary_ranks_behind_an_unmeasured_fallback() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        r.observe(row, 0, 50_000, false);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1, None).0;
        assert_eq!(names(walk, row), ["bedrock", "openrouter", "anthropic"]);
    }

    #[test]
    fn a_success_restores_a_failed_candidate() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 50_000, false);
        r.observe(row, 0, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1, None).0;
        assert_eq!(names(walk, row)[0], "anthropic");
    }

    #[test]
    fn when_everything_failed_the_least_bad_goes_first() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 9_000_000, false);
        r.observe(row, 1, 6_000_000, false);
        r.observe(row, 2, 7_000_000, false);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1, None).0;
        assert_eq!(names(walk, row), ["bedrock", "openrouter", "anthropic"]);
    }

    #[test]
    fn probe_seed_promotes_the_first_unmeasured() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, PROBE_EVERY, None)
            .0;
        assert_eq!(names(walk, row), ["bedrock", "anthropic", "openrouter"]);
    }

    #[test]
    fn seq_zero_does_not_probe() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 0, None).0;
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    #[test]
    fn a_route_outside_the_catalog_slice_is_not_ranked() {
        // A name the catalog does not have has no EWMA cells. Ranking leaves the walk alone.
        let r = Router::new();
        let route = ModelRoute {
            model: "not-a-catalog-row",
            wire: providers::WireFormat::OpenAi,
            candidates: &[],
            responses: &[],
            // Not a catalog row — the ranker must ignore it. The price and card are unused.
            price: providers::ListPrice {
                input: "0",
                output: "0",
                cache_read: "0",
                cache_write: "0",
            },
            card: providers::catalog::MODEL_ROUTES[0].card,
        };
        let walk = Walk::identity(3);
        let ranked = r.rank(walk, &route, 1, None).0;
        assert_eq!(ranked.indices, walk.indices);
        assert_eq!(ranked.len, walk.len);
    }

    // ---- Session pins ----

    const APP: u64 = 0xA11CE;

    fn aff() -> u64 {
        affinity(42, 7, Some(APP))
    }

    /// claim: R4
    #[test]
    fn a_pin_keeps_the_caller_on_a_slower_candidate() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 2_000_000, true);
        r.observe(row, 1, 100_000, true);
        // Without a pin, Bedrock is faster and goes first.
        let (walk, pinned) = r.rank(Walk::identity(row.candidates.len()), row, 1, Some(aff()));
        assert!(!pinned);
        assert_eq!(names(walk, row)[0], "bedrock");
        // This caller was last served by Anthropic: it stays there, Bedrock is its failover.
        r.pin(row, aff(), 0);
        let (walk, pinned) = r.rank(Walk::identity(row.candidates.len()), row, 1, Some(aff()));
        assert!(pinned);
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    /// The probe is what bounced sessions most: it moved whichever request drew the seed.
    #[test]
    fn a_pinned_caller_is_never_the_probe() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        r.pin(row, aff(), 0);
        let (walk, pinned) = r.rank(
            Walk::identity(row.candidates.len()),
            row,
            PROBE_EVERY,
            Some(aff()),
        );
        assert!(pinned);
        assert_eq!(names(walk, row)[0], "anthropic");
        // An unpinned caller on the same seed still probes.
        let (walk, _) = r.rank(Walk::identity(row.candidates.len()), row, PROBE_EVERY, None);
        assert_eq!(names(walk, row)[0], "bedrock");
    }

    #[test]
    fn a_pin_yields_when_its_candidate_just_failed() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        r.pin(row, aff(), 0);
        r.observe(row, 0, 50_000, false);
        let (walk, pinned) = r.rank(Walk::identity(row.candidates.len()), row, 1, Some(aff()));
        assert!(!pinned);
        assert_eq!(names(walk, row), ["bedrock", "openrouter", "anthropic"]);
        // The failover's 2xx re-pins, and the caller stays there after Anthropic recovers.
        r.pin(row, aff(), 1);
        r.observe(row, 0, 200_000, true);
        let (walk, pinned) = r.rank(Walk::identity(row.candidates.len()), row, 1, Some(aff()));
        assert!(pinned);
        assert_eq!(names(walk, row)[0], "bedrock");
    }

    #[test]
    fn pins_are_per_caller_and_per_model() {
        let r = Router::new();
        let row = opus();
        let other_row = providers::for_model("claude-haiku-4-5").expect("catalog row");
        r.pin(row, aff(), 1);
        let (_, pinned) = r.rank(
            Walk::identity(row.candidates.len()),
            row,
            1,
            Some(affinity(42, 7, Some(APP + 1))),
        );
        assert!(!pinned, "another key must not inherit the pin");
        let (_, pinned) = r.rank(
            Walk::identity(other_row.candidates.len()),
            other_row,
            1,
            Some(aff()),
        );
        assert!(!pinned, "another model must not inherit the pin");
    }

    #[test]
    fn a_pin_outside_the_filtered_walk_is_ignored() {
        let r = Router::new();
        let row = opus();
        r.pin(row, aff(), 2);
        // `x-beyond-only` left Anthropic and Bedrock; the pinned OpenRouter is not in the walk.
        let walk = Walk {
            indices: [0, 1, 0, 0, 0, 0, 0, 0],
            len: 2,
        };
        let (out, pinned) = r.rank(walk, row, 1, Some(aff()));
        assert!(!pinned);
        assert_eq!(names(out, row), ["anthropic", "bedrock"]);
    }

    #[test]
    fn refreshing_a_pin_keeps_its_creation_time() {
        let r = Router::new();
        let row = opus();
        let (row_i, _) = r.row(row).expect("row");
        let slot = &r.pins[slot_of(pin_key(aff(), row_i))];
        r.pin(row, aff(), 0);
        let first = slot.load(Ordering::Relaxed);
        r.pin(row, aff(), 0);
        let second = slot.load(Ordering::Relaxed);
        assert_eq!(
            (first >> PIN_TIME_BITS) & PIN_TIME_MASK,
            (second >> PIN_TIME_BITS) & PIN_TIME_MASK
        );
        // Moving to another candidate starts a new pin.
        r.pin(row, aff(), 1);
        assert_eq!(pin_idx(slot.load(Ordering::Relaxed)), 1);
    }

    #[test]
    fn a_pin_expires_when_idle_or_old() {
        let tag = 0xABCDEF;
        let now = 10_000;
        assert!(pin_live(pack(tag, 0, now - 100, now - 10), now));
        assert!(!pin_live(
            pack(tag, 0, now - 400, now - PIN_IDLE_S - 1),
            now
        ));
        assert!(!pin_live(
            pack(tag, 0, now - PIN_MAX_AGE_S - 1, now - 1),
            now
        ));
        assert!(!pin_live(0, now));
    }

    #[test]
    fn pin_ages_survive_the_seconds_counter_wrapping() {
        let tag = 0xABCDEF;
        // Taken 20s before the 18-bit seconds counter wrapped; read 10s after.
        let taken = PIN_TIME_MASK - 19;
        let now = 10;
        assert!(pin_live(pack(tag, 3, taken, taken), now));
        assert_eq!(pin_idx(pack(tag, 3, taken, taken)), 3);
    }

    #[test]
    fn the_tag_is_never_zero() {
        assert_eq!(tag_of(0), 1);
        assert_ne!(pack(tag_of(0), 0, 0, 0), 0);
    }
}
