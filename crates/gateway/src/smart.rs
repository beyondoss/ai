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
//! Unmeasured candidates would otherwise never run if the primary always succeeds, so every
//! [`PROBE_EVERY`]th request (skipping seq `0`) promotes the first unmeasured slot to primary.
//! Once every candidate on the row has a sample, the walk is pure EWMA. A sample older than
//! [`STALE`] is treated as unmeasured so a previously-slow arm can be retried.
//!
//! Integer EWMA (`7/8` previous + `1/8` sample) in microseconds, atomics only — the request path
//! does not allocate or take a lock. Per `(catalog row, candidate index)`, not per provider: Opus
//! on Bedrock is not Haiku on Bedrock.

use crate::control::Walk;
use providers::{MAX_CANDIDATES, ModelRoute, catalog::MODEL_ROUTES};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Promote the first unmeasured candidate to primary on `seq % PROBE_EVERY == 0` (and `seq != 0`).
/// Seq `0` is the first request of the process; leaving it on catalog order keeps a two-request
/// cache fill/hit on the same walk.
pub(crate) const PROBE_EVERY: u64 = 8;

/// A sample this old is ignored for ranking (the cell is still kept; the next observe continues
/// the EWMA). Long enough that a burst of traffic exploits, short enough that a recovered arm is
/// retried without an operator flipping a header.
const STALE: Duration = Duration::from_secs(30);

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
}

impl Row {
    fn new() -> Self {
        Self {
            ewma_us: std::array::from_fn(|_| AtomicU64::new(0)),
            last_ns: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
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
        let Some(row) = self.row(route) else {
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
        row.last_ns[i].store(now_ns(), Ordering::Relaxed);
    }

    /// Reorder `walk` by EWMA, or probe an unmeasured arm. Identity when the row is unknown or
    /// `walk` is empty.
    pub fn rank(&self, walk: Walk, route: &ModelRoute, seed: u64) -> Walk {
        if walk.len == 0 {
            return walk;
        }
        let Some(row) = self.row(route) else {
            return walk;
        };
        if seed != 0 && seed.is_multiple_of(PROBE_EVERY) {
            return probe(walk, |orig| self.effective(row, orig));
        }
        sort_measured(walk, |orig| self.effective(row, orig))
    }

    fn row(&self, route: &ModelRoute) -> Option<&Row> {
        let i = MODEL_ROUTES
            .binary_search_by(|r| r.model.cmp(route.model))
            .ok()?;
        self.rows.get(i)
    }

    fn effective(&self, row: &Row, catalog_idx: u8) -> u64 {
        let i = usize::from(catalog_idx);
        if i >= MAX_CANDIDATES {
            return 0;
        }
        let ewma = row.ewma_us[i].load(Ordering::Relaxed);
        if ewma == 0 {
            return 0;
        }
        let last = row.last_ns[i].load(Ordering::Relaxed);
        if last != 0 && u128::from(now_ns().saturating_sub(last)) > STALE.as_nanos() {
            return 0;
        }
        ewma
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

/// Measured slots (ewma > 0) sorted fastest-first, then unmeasured in the order they already sat.
fn sort_measured(walk: Walk, score: impl Fn(u8) -> u64) -> Walk {
    let n = walk.len;
    let mut measured = [(0u64, 0u8, 0u8); MAX_CANDIDATES];
    let mut unmeasured = [0u8; MAX_CANDIDATES];
    let mut nm = 0usize;
    let mut nu = 0usize;
    for slot in 0..n {
        let orig = walk.indices[slot as usize];
        let ewma = score(orig);
        if ewma == 0 {
            unmeasured[nu] = orig;
            nu += 1;
        } else {
            measured[nm] = (ewma, slot, orig);
            nm += 1;
        }
    }
    measured[..nm].sort_unstable_by_key(|&(ewma, slot, _)| (ewma, slot));
    let mut out = Walk {
        indices: [0u8; MAX_CANDIDATES],
        len: n,
    };
    for (slot, orig) in out.indices.iter_mut().zip(
        measured[..nm]
            .iter()
            .map(|t| t.2)
            .chain(unmeasured[..nu].iter().copied()),
    ) {
        *slot = orig;
    }
    out
}

/// Move the first unmeasured catalog index to slot 0; leave the rest in order. No-op when every
/// slot is already measured (exploit-only) or the unmeasured one is already primary.
fn probe(walk: Walk, score: impl Fn(u8) -> u64) -> Walk {
    let n = walk.len;
    let mut pick = None;
    for i in 0..n {
        if score(walk.indices[i as usize]) == 0 {
            pick = Some(i);
            break;
        }
    }
    let Some(pick) = pick else {
        return walk;
    };
    if pick == 0 {
        return walk;
    }
    let mut out = walk;
    let primary = out.indices[pick as usize];
    let mut i = pick;
    while i > 0 {
        out.indices[i as usize] = out.indices[i as usize - 1];
        i -= 1;
    }
    out.indices[0] = primary;
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
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1);
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    #[test]
    fn faster_measured_candidate_is_tried_first() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 2_000_000, true);
        r.observe(row, 1, 100_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1);
        assert_eq!(names(walk, row), ["bedrock", "anthropic", "openrouter"]);
    }

    #[test]
    fn unmeasured_stay_failover_so_a_working_primary_is_not_abandoned() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1);
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }

    #[test]
    fn failed_attempt_takes_the_penalty_floor() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 1_000, false);
        r.observe(row, 1, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 1);
        assert_eq!(names(walk, row)[0], "bedrock");
    }

    #[test]
    fn probe_seed_promotes_the_first_unmeasured() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, PROBE_EVERY);
        assert_eq!(names(walk, row), ["bedrock", "anthropic", "openrouter"]);
    }

    #[test]
    fn seq_zero_does_not_probe() {
        let r = Router::new();
        let row = opus();
        r.observe(row, 0, 200_000, true);
        let walk = r.rank(Walk::identity(row.candidates.len()), row, 0);
        assert_eq!(names(walk, row), ["anthropic", "bedrock", "openrouter"]);
    }
}
