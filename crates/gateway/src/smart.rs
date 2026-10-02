//! In-process catalog-walk ranking from observed time-to-first-byte.
//!
//! **Per-pod, not fleet-wide.** EWMA samples never leave this process. Two replicas of the same
//! catalog row can walk new callers' candidates in different orders. What agrees across replicas
//! is computed, never shared: a session pin (hash of the caller, see below) and `x-beyond-split`
//! (hash of the request counter). There is no Redis on this path.
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
//! Ranking orders a walk for nobody in particular; a session pin keeps one caller on one provider.
//! Provider prompt caches are per provider, so a walk that re-ranks every request moves an agent
//! loop from Anthropic to Bedrock and back, and every move re-reads the whole prefix at full input
//! price plus a cache write (1.25× vs 0.1× on Claude). Worse, the probe above lands on whichever
//! request happens to draw a multiple of [`PROBE_EVERY`] — usually someone mid-session.
//!
//! The pin key is the verified identity (`tenant_id`, `vpc_id`, `key_id`) plus the catalog row:
//! one virtual key is one app, and an app's sessions share system prompt and tool prefixes, so
//! keeping the whole key on one provider is what the provider cache wants.
//!
//! **The pin is computed every turn, never remembered.** A caller's walk is [`preferred`]: the
//! row's leading run of equally-preferred first-party hosts (the vendor, and clouds reselling it
//! under its own ids: Anthropic + Bedrock) ordered by rendezvous hash of the pin key and provider,
//! then the rest in catalog order. Candidates that cannot be used on this pod (no pool key, every
//! key cooling, cannot serve the body) move to the back. No TTFT, no probe, no failure history:
//! nothing a pod observes moves the pin, so every replica computes the same walk and there is no
//! per-pod table to disagree (D253: a remembered per-pod pin let pod A keep a key on its failover
//! after one transient 5xx while pod B served the primary, and alternating turns re-bought the
//! prefix every time). An aggregator is never hashed in; it serves only when no first-party host
//! before it can.
//!
//! A failure fails over for that request only: the walk moves on in-gateway, and the next turn
//! starts from the preferred host again. An open breaker is skipped by `upstream_peer`, on that
//! pod, while it is open, and the next candidate in the computed walk is the same on every pod.
//! That is the one place replicas can disagree: during a real outage one pod's breaker can be open
//! while another's is closed, and a session alternating between them then hops between the
//! preferred host (where it still answers) and its failover until the breakers agree. Rendezvous
//! hashing moves only the keys hashed to the sick host.

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

static BASE: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Per-row EWMA table, parallel to [`MODEL_ROUTES`].
pub struct Router {
    rows: Box<[Row]>,
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

    /// Reorder `walk`: a caller with an identity gets its computed session pin ([`preferred`]);
    /// a walk with none gets EWMA order, or a probe of an unmeasured arm. Identity when the row is
    /// unknown or `walk` is empty. The `bool` is whether a pin decided the order. `affinity` is
    /// [`affinity`]'s hash of the caller, `None` for no pinning.
    ///
    /// `dispatchable` has bit `i` set when `route.candidates[i]` can be sent to (it has a pool key
    /// here). Only those can be probed: an unkeyed candidate never gets a sample, so it would stay
    /// unmeasured forever, win every probe, be skipped by `upstream_peer`, and starve a keyed
    /// unmeasured arm behind it of its only chance to be measured.
    pub fn rank(
        &self,
        walk: Walk,
        route: &ModelRoute,
        seed: u64,
        affinity: Option<u64>,
        dispatchable: u8,
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
        if let Some(aff) = affinity {
            // Only what cannot be used here moves. No TTFT, probe or failure history: anything
            // this pod observed would let two pods disagree.
            let usable = |orig: u8| dispatchable & (1 << orig) != 0;
            return (preferred(walk, route, pin_key(aff, row_i), usable), true);
        }
        if seed != 0 && seed.is_multiple_of(PROBE_EVERY) {
            return (probe(walk, score, dispatchable), false);
        }
        (sort_measured(walk, score), false)
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
/// verified before this runs, and the hash only orders providers the caller's own row names.
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

/// A first-party host for its row: the vendor's own API, or a cloud reselling the model under the
/// vendor's ids (Bedrock). An aggregator claims no id prefix (`ProviderSpec::model_id_match`
/// explains why), and it is the costlier failover a pin must not be hashed onto while a
/// first-party host can serve.
fn first_party(route: &ModelRoute, orig: u8) -> bool {
    route
        .candidates
        .get(usize::from(orig))
        .is_some_and(|c| !providers::by_id(c.provider).model_id_match.is_empty())
}

/// FNV-1a of the provider's name: its identity in the rendezvous hash. The name, not the
/// `ProviderId` discriminant, so two builds that order the enum differently (a rolling deploy)
/// still agree.
fn provider_hash(name: &str) -> u64 {
    name.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The computed session pin: the walk's leading run of [`first_party`] candidates ordered by
/// rendezvous hash of (`key`, provider), then the rest in the walk's (catalog) order; candidates
/// `usable` rejects go behind the others, keeping that order. A pure function of the key, the row
/// and the usable set, so every replica computes the same walk. Hashing the provider, not the
/// slot, keeps a key's choice when an `only` filter or an unusable host removes another candidate: only
/// the keys hashed to the removed one move.
fn preferred(walk: Walk, route: &ModelRoute, key: u64, usable: impl Fn(u8) -> bool) -> Walk {
    let n = usize::from(walk.len);
    let lead = walk.indices[..n]
        .iter()
        .take_while(|&&orig| first_party(route, orig))
        .count();
    // (sick, past the lead, inverted hash, slot, orig): usable first, the lead highest hash first.
    let mut keyed = [(false, false, 0u64, 0u8, 0u8); MAX_CANDIDATES];
    for (slot, (k, &orig)) in keyed.iter_mut().zip(&walk.indices[..n]).enumerate() {
        let rest = slot >= lead;
        let hash = match route.candidates.get(usize::from(orig)) {
            Some(c) if !rest => !mix(key ^ provider_hash(providers::by_id(c.provider).name)),
            _ => 0,
        };
        *k = (!usable(orig), rest, hash, slot as u8, orig);
    }
    keyed[..n].sort_unstable();
    let mut out = Walk {
        indices: [0u8; MAX_CANDIDATES],
        len: walk.len,
    };
    for (dst, k) in out.indices.iter_mut().zip(&keyed[..n]) {
        *dst = k.4;
    }
    out
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

/// Move the first unmeasured, dispatchable catalog index to slot 0; leave the rest in order. No-op
/// when every dispatchable slot is already measured (exploit-only) or the unmeasured one is
/// already primary.
fn probe(walk: Walk, score: impl Fn(u8) -> Tier, dispatchable: u8) -> Walk {
    let n = usize::from(walk.len);
    match walk.indices[..n]
        .iter()
        .find(|&&orig| dispatchable & (1 << orig) != 0 && score(orig) == Tier::Unmeasured)
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

    /// Every candidate keyed.
    const ALL: u8 = u8::MAX;

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
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 1, None, ALL)
            .0;
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    /// claim: R7
    #[test]
    fn faster_measured_candidate_is_tried_first() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 2_000_000, true);
        r.observe(row, 1, 100_000, true);
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 1, None, ALL)
            .0;
        assert_eq!(names(walk, row), ["bedrock", "anthropic", "openrouter"]);
    }

    #[test]
    fn unmeasured_stay_failover_so_a_working_primary_is_not_abandoned() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 1, None, ALL)
            .0;
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    #[test]
    fn failed_attempt_takes_the_penalty_floor() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 1_000, false);
        r.observe(row, 1, 200_000, true);
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 1, None, ALL)
            .0;
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
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 1, None, ALL)
            .0;
        assert_eq!(names(walk, row), ["bedrock", "openrouter", "anthropic"]);
    }

    #[test]
    fn a_success_restores_a_failed_candidate() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 50_000, false);
        r.observe(row, 0, 200_000, true);
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 1, None, ALL)
            .0;
        assert_eq!(names(walk, row)[0], "anthropic");
    }

    #[test]
    fn when_everything_failed_the_least_bad_goes_first() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 9_000_000, false);
        r.observe(row, 1, 6_000_000, false);
        r.observe(row, 2, 7_000_000, false);
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 1, None, ALL)
            .0;
        assert_eq!(names(walk, row), ["bedrock", "openrouter", "anthropic"]);
    }

    #[test]
    fn probe_seed_promotes_the_first_unmeasured() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r
            .rank(
                Walk::identity(row.candidates.len()),
                row,
                PROBE_EVERY,
                None,
                ALL,
            )
            .0;
        assert_eq!(names(walk, row), ["bedrock", "anthropic", "openrouter"]);
    }

    /// claim: R7
    /// defect: D119
    #[test]
    fn the_probe_skips_a_candidate_that_cannot_be_dispatched() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        // Bedrock (index 1) has no pool key: OpenRouter is the arm to measure.
        let walk = r
            .rank(
                Walk::identity(row.candidates.len()),
                row,
                PROBE_EVERY,
                None,
                0b101,
            )
            .0;
        assert_eq!(names(walk, row), ["openrouter", "anthropic", "bedrock"]);
    }

    #[test]
    fn seq_zero_does_not_probe() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r
            .rank(Walk::identity(row.candidates.len()), row, 0, None, ALL)
            .0;
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
        let ranked = r.rank(walk, &route, 1, None, ALL).0;
        assert_eq!(ranked.indices, walk.indices);
        assert_eq!(ranked.len, walk.len);
    }

    // ---- Session pins ----

    const APP: u64 = 0xA11CE;

    fn aff() -> u64 {
        affinity(42, 7, Some(APP))
    }

    fn mini() -> &'static ModelRoute {
        providers::for_model("gpt-4o-mini").expect("catalog row")
    }

    fn ranked(
        r: &Router,
        row: &ModelRoute,
        seed: u64,
        a: u64,
        dispatchable: u8,
    ) -> (Vec<&'static str>, bool) {
        let (walk, pinned) = r.rank(
            Walk::identity(row.candidates.len()),
            row,
            seed,
            Some(a),
            dispatchable,
        );
        (names(walk, row), pinned)
    }

    /// The walk a fresh pod computes for `a` on `row`, every candidate usable.
    fn computed(row: &ModelRoute, a: u64) -> Vec<&'static str> {
        ranked(&Router::new(), row, 1, a, ALL).0
    }

    fn index_of(row: &ModelRoute, name: &str) -> u8 {
        row.candidates
            .iter()
            .position(|c| by_id(c.provider).name == name)
            .expect("candidate") as u8
    }

    /// claim: R4
    #[test]
    fn a_pinned_caller_ignores_ttft() {
        let r = Router::new();
        let row = opus();
        let want = computed(row, aff());
        let first = index_of(row, want[0]);
        // Make the computed primary the slow one: a walk with no caller goes around it.
        r.observe(row, first, 2_000_000, true);
        r.observe(row, 1 - first, 100_000, true);
        let (walk, _) = r.rank(Walk::identity(row.candidates.len()), row, 1, None, ALL);
        assert_ne!(names(walk, row)[0], want[0]);
        let (walk, pinned) = ranked(&r, row, 1, aff(), ALL);
        assert!(pinned);
        assert_eq!(walk, want);
        assert_eq!(walk[2], "openrouter", "the aggregator stays last");
    }

    /// The probe is what bounced sessions most: it moved whichever request drew the seed.
    #[test]
    fn a_pinned_caller_is_never_the_probe() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let (walk, pinned) = ranked(&r, row, PROBE_EVERY, aff(), ALL);
        assert!(pinned);
        assert_eq!(walk, computed(row, aff()));
        // A walk with no caller on the same seed still probes.
        let (walk, _) = r.rank(
            Walk::identity(row.candidates.len()),
            row,
            PROBE_EVERY,
            None,
            ALL,
        );
        assert_eq!(names(walk, row)[0], "bedrock");
    }

    /// D253: two pods with different samples, probes and failures compute the same walk for one
    /// caller. Across callers the pooled first-party hosts share the load, and the aggregator never
    /// leads while one of them is usable.
    /// claim: R4
    /// defect: D253
    #[test]
    fn every_pod_computes_the_same_pin() {
        let row = opus();
        let (a, b) = (Router::new(), Router::new());
        a.observe(row, 0, 50_000, false);
        a.observe(row, 1, 90_000, false);
        a.observe(row, 2, 10_000, true);
        b.observe(row, 0, 300_000, true);
        let mut leads = [0usize; 2];
        for app in 0..64 {
            let k = affinity(42, 7, Some(app));
            for seed in [1, PROBE_EVERY] {
                let on_a = ranked(&a, row, seed, k, ALL).0;
                assert_eq!(on_a, ranked(&b, row, seed, k, ALL).0, "app {app}");
                assert_eq!(on_a[2], "openrouter");
                leads[usize::from(on_a[0] == "bedrock")] += 1;
            }
        }
        assert!(
            leads[0] > 16 && leads[1] > 16,
            "rendezvous spreads the lead: {leads:?}"
        );
    }

    /// A failure fails over for that request only: the next turn starts from the preferred host
    /// again, wherever it runs.
    /// claim: R4
    /// defect: D253
    #[test]
    fn a_failure_does_not_move_the_next_turn() {
        let r = Router::new();
        let row = mini();
        r.observe(row, 0, 50_000, false);
        assert_eq!(ranked(&r, row, 1, aff(), ALL).0, ["openai", "openrouter"]);
        // A walk with no caller still goes around the failed primary.
        let (walk, _) = r.rank(Walk::identity(row.candidates.len()), row, 1, None, ALL);
        assert_eq!(names(walk, row)[0], "openrouter");
    }

    #[test]
    fn a_pin_skips_a_candidate_that_cannot_be_used_here() {
        let row = opus();
        let want = computed(row, aff());
        let lead = index_of(row, want[0]);
        let walk = ranked(&Router::new(), row, 1, aff(), ALL & !(1 << lead)).0;
        assert_eq!(walk, [want[1], want[2], want[0]]);
    }

    /// The pin is a function of the caller's own verified identity: another key, tenant or model
    /// hashes on its own and cannot be steered by this one.
    /// claim: SEC-15
    #[test]
    fn pins_are_per_caller_and_per_model() {
        let row = opus();
        let k = |t, app| pin_key(affinity(t, 7, Some(app)), 0);
        assert_ne!(k(42, APP), k(42, APP + 1), "another key");
        assert_ne!(k(42, APP), k(43, APP), "another tenant");
        assert_ne!(pin_key(aff(), 0), pin_key(aff(), 1), "another model");
        // Over many tenants sharing one key id, the lead follows each tenant's own hash.
        let leads: std::collections::BTreeSet<_> = (0..32)
            .map(|t| computed(row, affinity(t, 7, Some(APP)))[0])
            .collect();
        assert_eq!(leads.len(), 2, "{leads:?}");
    }

    /// `x-beyond-only` filters before ranking; the pin is computed over what is left, and the
    /// removed candidate moves no other (rendezvous).
    #[test]
    fn a_filtered_walk_keeps_the_pin_among_what_is_left() {
        let r = Router::new();
        let row = opus();
        let want = computed(row, aff());
        let walk = Walk {
            indices: [0, 1, 0, 0, 0, 0, 0, 0],
            len: 2,
        };
        let (out, pinned) = r.rank(walk, row, 1, Some(aff()), ALL);
        assert!(pinned);
        assert_eq!(names(out, row), want[..2]);
        let walk = Walk {
            indices: [2, 0, 0, 0, 0, 0, 0, 0],
            len: 1,
        };
        assert_eq!(
            names(r.rank(walk, row, 1, Some(aff()), ALL).0, row),
            ["openrouter"]
        );
    }

    /// Only the leading run of first-party hosts is hashed: a row whose primary is an aggregator
    /// keeps the catalog's order.
    #[test]
    fn an_aggregator_primary_keeps_catalog_order() {
        const CANDIDATES: &[providers::Candidate] = &[
            providers::Candidate {
                provider: providers::ProviderId::OpenRouter,
                upstream_model: "x",
                path: "/api/v1/chat/completions",
            },
            providers::Candidate {
                provider: providers::ProviderId::OpenAi,
                upstream_model: "x",
                path: "/v1/chat/completions",
            },
            providers::Candidate {
                provider: providers::ProviderId::Anthropic,
                upstream_model: "x",
                path: "/v1/messages",
            },
        ];
        let route = ModelRoute {
            candidates: CANDIDATES,
            ..*opus()
        };
        for app in 0..16 {
            let k = pin_key(affinity(42, 7, Some(app)), 0);
            let out = preferred(Walk::identity(3), &route, k, |_| true);
            assert_eq!(out.indices[..3], [0, 1, 2]);
        }
    }
}
