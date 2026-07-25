//! Request-rate guardrails — blast-radius circuit breakers, **not** a spend control.
//!
//! The deny-set (see `deny`) is the spend/fraud authority, but it's *cumulative* and reacts on a
//! lag: it only learns of spend after usage facts round-trip through the control plane, and it's
//! structurally blind to request floods that never bill — auth failures (rejected here, never reach
//! upstream), provider 4xx, and BYO traffic (on the caller's own key, no Beyond identity). Two tiers
//! cap velocity, both charged in `proxy::request_filter` *before* the Ed25519 verify and the upstream
//! connect, so a flood can't drive unbounded crypto/socket work:
//!
//! 1. **Per-credential** — keyed by the raw presented credential (the whole `bai_…` virtual key or
//!    BYO token). Catches a single leaked/runaway key. Granularity is per-credential: managed virtual
//!    keys are deterministic per `(tenant, app)`, so this is effectively a per-(tenant, app) ceiling —
//!    one credential's runaway can't throttle another. A flood of *distinct* credentials slips past
//!    it (every random string is its own bucket), which is what tier 2 exists for.
//!
//! 2. **Global BYO aggregate** — a single bucket for *all* BYO traffic combined. BYO is unverified
//!    and upstream-bound: a flood of distinct random BYO tokens would otherwise open junk-auth
//!    connections to providers from our egress IPs, getting them rate-limited or banned (we put
//!    ourselves in the firing line). This bounds that aggregate regardless of how the tokens vary.
//!    **Managed traffic is exempt** — it's Ed25519-verified before any upstream connect and can't be
//!    forged (the signing key lives only in the control plane), so a random `bai_` flood fails verify
//!    and never reaches a provider (CPU only, no egress impact). Exempting it means this shared bucket
//!    only ever sheds BYO load under a flood, never the core managed tenants.
//!
//! Both tiers are deliberately generous: ceilings well above legitimate steady state, so they never
//! trip in normal operation. Tune from `ai_rejections_total{reason="rate_limit"}` (per-credential)
//! and `{reason="rate_limit_byo_global"}` (BYO aggregate).
//!
//! ## Design decision: why a global BYO cap and not per-source-IP (READ BEFORE CHANGING)
//!
//! The threat that shaped tier 2 is **egress-IP reputation**, not gateway CPU. We are an egress proxy:
//! BYO requests connect outward to OpenAI/Anthropic/OpenRouter/… *from our IPs* carrying the caller's
//! token. A flood of distinct **junk** BYO tokens makes those providers see a torrent of failed-auth
//! connections from us and rate-limit or ban our egress IPs — taking down BYO for *everyone*, and
//! degrading managed traffic that shares the same egress. That blast radius is why this lives here and
//! is on by default, rather than being pushed entirely to the mesh/ingress.
//!
//! **Per-source-IP limiting was considered and rejected** as the primary control. It's the surgical
//! answer in principle (throttle only the noisy source), but it depends on the calling task's real IP
//! being visible here — and in production we front this with ECS Service Connect, where it is unclear
//! whether the peer address is the client task or a collapsed mesh/proxy hop. If it's collapsed,
//! per-IP keying is worse than nothing: it either does nothing (all sources share one IP, so no single
//! key trips) or throttles every tenant at once. We refused to hinge an egress-protection control on
//! an unverified topology assumption. The global BYO cap is **topology-independent** — it bounds the
//! aggregate no matter how source identity is mangled. (If/when we confirm real per-task IPs reach us,
//! a per-IP tier is a reasonable *addition* in front of this — not a replacement.)
//!
//! ## What this deliberately does NOT cover (the residual — don't assume it's solved)
//!
//! - **The BYO cap is a shared bucket.** A flood large enough to hit `byo_rate_limit_rps` *does* shed
//!   legitimate BYO callers along with the attacker — they're indistinguishable at admit time (we
//!   reject before we know a token is junk). The trust segmentation (managed exempt) bounds the blast
//!   radius to BYO only; it does not make the BYO shedding selective.
//! - **The default ceiling is a guess.** `byo_rate_limit_rps = 1000` was picked without real BYO
//!   traffic numbers — high enough to clear plausible legitimate use, low enough that a junk flood
//!   can't realistically get us banned. It is meant to be tuned from the metric, not trusted as-is.
//! - **A more selective control is the next step, not this.** The surgical fix for egress reputation
//!   is a **provider-feedback circuit breaker**: watch upstream responses and back BYO off a provider
//!   when we see a burst of `401`s (junk auth) from it, instead of capping all BYO blindly. That reacts
//!   to the actual signal (providers rejecting us) and spares legitimate BYO. It's a real feature, not
//!   a guardrail, so it's intentionally out of scope here. If you're here because the blunt cap hurt,
//!   build that — don't just raise the number.
//!
//! The per-credential tier is a pair of pingora-limits count-min-sketch `Estimator`s rotated at the
//! window boundary (`WindowedRate`), giving **fixed memory regardless of key cardinality** — no
//! per-credential entry, no background GC — which matches the deny-set's O(denied) ethos. A sketch
//! can *over*estimate a key's rate on hash collision but never under, so a cap is always enforced;
//! `SLOTS` is sized wide enough that overestimation stays negligible. `pingora_limits::rate::Rate`
//! itself is deliberately **not** used — see the note on `WindowedRate` for the reason.
//!
//! The global BYO tier is a *single* bucket, so it needs no sketch at all: it is one cacheline of
//! atomic (`ByoCounter`) packing the window index with the count, which makes it both exact and
//! reset-free. Only the per-credential tier, where key cardinality is unbounded, pays for a sketch.

// `ahash::RandomState` carries its own inherent `hash_one` (it does not need `BuildHasher` in
// scope), which is also the specialised, faster path — see the `hasher` field's note.
use ahash::RandomState;
use pingora_limits::estimator::Estimator;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Count-min sketch dimensions for the per-credential tier.
///
/// ## Accuracy: where 65536 comes from
///
/// The sketch can only *over*estimate a key's rate, never under, so the cap always holds and the
/// single failure mode is **false throttling** — an innocent caller whose counter was inflated by
/// other people's traffic colliding into its slots. Two numbers bound that:
///
/// - the guarantee is `Pr[error > e·N / SLOTS] < e^-HASHES`, i.e. an additive error of at most
///   `e·N/SLOTS` on all but ≈ 0.7% of checks at `HASHES = 5`;
/// - typical error is far lower — one row carries ≈ `N/SLOTS` of foreign mass and the estimate is
///   the *min* over `HASHES` independently-hashed rows.
///
/// `N` is **not** the node's request rate. It is the rate that reaches *this sketch*, which is
/// smaller, and asymmetrically so:
///
/// - BYO contributes at most `byo_rate_limit_rps` (default 1000), because `check_at` returns on the
///   global BYO ceiling **before** it touches the sketch. An unbounded junk-token flood is already
///   shed a tier earlier and never lands here.
/// - Managed traffic is the unbounded term. It is exempt from tier 2 by design, and a flood of
///   forged `bai_…` strings is charged here in full — it is only rejected later, at verify.
///
/// So `peak_N` ≈ the peak *managed* rps a single node can accept. **Assume 100k rps.** A flood of
/// distinct forged keys cannot outrun the node's Ed25519 verify throughput (22.9 µs/verify measured
/// — `bench unit -- key::verify` — so ≈ 44k/s/core); a same-key flood is rejected here before verify
/// and is bounded instead by L7 parsing. 100k sits between those and far above any plausible
/// legitimate volume.
///
/// The error must stay small against the ceiling it could push a caller over — and that ceiling is
/// `config.rate_limit_rps`, **default 100 rps**, not an abstract large number. (The original sizing
/// said "far under the per-credential ceiling" without naming it; against 100, "≤ 5" is 5%, which
/// is the actual claim being made.) Budgeting 5%:
///
/// ```text
/// SLOTS ≥ e × peak_N / tolerable_error = 2.718 × 100_000 / 5 = 54_366  →  65_536
/// ```
///
/// To re-derive: `SLOTS ≈ e × peak_N / (0.05 × rate_limit_rps)`, rounded up to a power of two. Note
/// it is a function of the *ratio* — raise `rate_limit_rps` to 1000 and 8192 slots would do.
///
/// ## Cost: the side the original sizing never accounted for
///
/// Memory is `2 × HASHES × SLOTS × 8 B` = **5.24 MB, fixed** regardless of credential cardinality
/// (no per-key entry, no GC), of which 2.62 MB is live and touched at `HASHES` pseudo-random
/// offsets per request. Measured against `SLOTS = 8192` at the same `HASHES`, on a 16-thread Zen
/// box (1 MB L2/core, 16 MB shared L3), medians over three runs:
///
/// ```text
///                              SLOTS = 8192   SLOTS = 65536
///   check, 1 thread               47.0 ns        55.2 ns      +8.2 ns (+17%)
///   check, 16 threads             87.4 ns        87.6 ns      no measurable difference
///   window rotation                4.7 µs        40.7 µs      8.7×
/// ```
///
/// The per-request cache cost is real but small, and it *disappears* under concurrency: 2.62 MB is
/// L3-resident here and the `HASHES` loads issue in parallel, so the latency hides behind
/// throughput. That will not hold on a node with a smaller or busier L3 — re-measure, don't assume.
///
/// What the width actually costs is paid at the **rotation**: 8× the slots is 8× the zeroing walk,
/// and that walk runs inline on one unlucky request thread (see `WindowedRate::rotate`). At 1 Hz
/// against 100k rps that is one request in ~100k — a p99.999 tail term, not a throughput one. Given
/// the accuracy argument above needs the width and the throughput argument does not object, the
/// width stays. If that tail ever matters more than the accuracy does, shrink `SLOTS` *and* raise
/// `rate_limit_rps` together: as the formula shows, they are one knob, not two.
const SLOTS: usize = 65536;
const HASHES: usize = 5;

/// The rate window. Every ceiling is expressed per this interval, i.e. requests/second.
const WINDOW: Duration = Duration::from_secs(1);

/// `WINDOW` in milliseconds — the unit the window index is counted in. A `const` so `check_at`'s
/// divide is a multiply-shift rather than a hardware `div`.
const WINDOW_MS: u64 = WINDOW.as_millis() as u64;

/// The whole state of the global BYO tier: one cacheline-isolated atomic packing
/// `(window_index: u32) << 32 | (count: u32)`.
///
/// Tier 2 is a **single shared bucket**, so a count-min sketch buys it nothing — there is only one
/// key, so there is nothing to collide with and nothing to estimate. It was nevertheless a
/// `Rate::new`, i.e. the default 4 × 1024 estimator, which meant every BYO request paid four fully
/// contended read-modify-writes across four cachelines, a clock read, and a share of a 64 KB
/// two-slot reset walk — all to maintain one number that one atomic holds exactly.
///
/// Packing the window index *with* the count is what makes the rotation free: opening a new window
/// is the same compare-and-swap as counting a request, so there is no reset step at all and no
/// interval in which the count and the window it belongs to can disagree.
#[repr(align(64))]
struct ByoCounter {
    /// `(window_index: u32) << 32 | (count: u32)`.
    packed: AtomicU64,
    /// The ceiling for this tier. Never written after construction, so sharing the line with
    /// `packed` costs nothing — a thread that just took the line exclusive for its CAS reads the
    /// ceiling out of the same line for free.
    max: u32,
}

impl ByoCounter {
    fn new(max: u32) -> Self {
        Self {
            packed: AtomicU64::new(0),
            max,
        }
    }

    /// Charge one request to `window`, returning the running count for that window.
    ///
    /// The window index is truncated to 32 bits. At a 1 s window counted from this limiter's
    /// construction off a monotonic `Instant`, that is ~136 years of process uptime — unreachable
    /// rather than merely unlikely.
    fn observe(&self, window: u64) -> u32 {
        let window = window as u32;
        let mut cur = self.packed.load(Ordering::Relaxed);
        loop {
            let next = if window > (cur >> 32) as u32 {
                // First request of a new window: the previous count is simply overwritten. That
                // *is* the reset — no walk to amortise, no slot to rotate, no memory to clear.
                (u64::from(window) << 32) | 1
            } else {
                // Same window — or this thread's clock read lost the race to one that already
                // opened the next window, in which case charge the live window rather than dragging
                // it backwards and handing out a second budget for a window that just closed.
                // Saturating: a count that wrapped to 0 would read as "under the ceiling", so a
                // flood big enough to overflow must stay pinned above it (and the workspace builds
                // with `overflow-checks`, so a plain `+ 1` would be a panic).
                (cur & !(u32::MAX as u64)) | u64::from((cur as u32).saturating_add(1))
            };
            // Relaxed on both sides: this counter publishes no other memory, and a single location
            // still has a total modification order, which is all the "never move backwards"
            // argument above needs.
            match self
                .packed
                .compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return next as u32,
                Err(seen) => cur = seen,
            }
        }
    }
}

/// A fixed-window event counter over a count-min sketch: "how many events has `key` been charged in
/// the window that is currently open". Two sketches are rotated at each window boundary so the reset
/// walk can never race an increment.
///
/// ## Why not `pingora_limits::rate::Rate` (READ BEFORE REVERTING)
///
/// `Rate::maybe_reset` computes `past_ms` as
///
/// ```text
/// let now = Instant::now().duration_since(self.start).as_millis() as u64;   // (A)
/// let last_reset = self.last_reset_time.load(Ordering::SeqCst);             // (B)
/// let past_ms = now - last_reset;                                           // unchecked
/// ```
///
/// — a subtraction of two values read at two different instants. A thread that takes its clock
/// reading at (A) and is then descheduled, while a thread whose reading was *later* wins the reset
/// CAS and stores that larger timestamp, resumes to find `now < last_reset` and underflows. No
/// memory ordering fixes this: the clock read strictly precedes the atomic load, so their relative
/// age can always be inverted. Cargo profile settings apply to dependencies and this workspace sets
/// `[profile.release] overflow-checks = true`, so the underflow is a **panicking Pingora worker**,
/// not a silent wrap. (`rate.rs`'s `assert!(new >= now - 1000)` on the CAS-loser path is the same
/// hazard, unreachable today only because `WINDOW` happens to be exactly 1 s.)
///
/// This type keeps the same red/blue rotation but never subtracts two independently-read values.
/// The caller turns its single `Instant` into a monotonically non-decreasing **window index** once
/// (`RateLimit::check_at`) and rotation is decided by comparing indices, so there is nothing left to
/// underflow — the bug is designed out rather than clamped away. Threading the index in also means
/// the clock is read once per *request* instead of once per tier.
struct WindowedRate {
    /// The two sketches. The live window index's parity picks the one `observe` increments; the
    /// other is zeroed *before* the flip, so a rotation neither exposes stale counts nor wipes
    /// increments still in flight for the window being counted.
    slots: [Estimator; 2],
    /// The highest window index some thread has taken responsibility for rotating into. CAS'd, so
    /// exactly one thread performs each rotation and the ~36 µs zeroing walk is never duplicated.
    claimed: AtomicU64,
    /// The window index that is *live*: `& 1` selects the slot `observe` increments. Advanced only
    /// once the incoming slot holds zeros, with release/acquire so a thread that sees the new index
    /// is guaranteed to see them. `fetch_max` keeps it monotonic, so a rotation descheduled past a
    /// later one cannot drag the live slot backwards onto counts that are already being taken.
    live: AtomicU64,
}

impl WindowedRate {
    fn new(hashes: usize, slots: usize) -> Self {
        Self {
            slots: [Estimator::new(hashes, slots), Estimator::new(hashes, slots)],
            claimed: AtomicU64::new(0),
            live: AtomicU64::new(0),
        }
    }

    /// Charge one event to `key` in `window`, returning the running count for that window.
    ///
    /// A `window` *older* than the live one — this thread's clock read lost a race — is charged to
    /// the live window rather than reopening a closed one. Counts only ever move forward.
    fn observe(&self, key: u64, window: u64) -> isize {
        let claimed = self.claimed.load(Ordering::Relaxed);
        // The overwhelmingly common case is `window == claimed`: one predictable compare, no RMW.
        if window > claimed
            && self
                .claimed
                .compare_exchange(claimed, window, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            self.rotate(claimed, window);
        }
        let live = (self.live.load(Ordering::Acquire) & 1) as usize;
        self.slots[live].incr(key, 1)
    }

    /// Retire window `from` and open `to`. The `claimed` CAS elects exactly one request thread per
    /// rotation to run this, so the zeroing walk is paid once per window, not once per request.
    ///
    /// Known residual, written down rather than hidden: if a rotation is descheduled long enough
    /// for the *next* window to open before it publishes, that next rotation can zero a slot the
    /// stalled one has left live. The stall has to exceed a whole `WINDOW` (1 s) to happen at all,
    /// against a walk measured at ~39 µs, and the consequence is bounded — some counts in one
    /// window are dropped, so a handful of requests are admitted that would have been shed. Never a
    /// panic, never a ceiling that stops being enforced, and self-correcting at the next window.
    #[cold]
    fn rotate(&self, from: u64, to: u64) {
        let incoming = (to & 1) as usize;
        // Zero *before* publishing. Until `live` moves, every other thread is still charging the
        // outgoing slot, so nothing is counting into the one being cleared and no in-flight
        // increment can be lost. (When `to - from` is even the incoming slot *is* the outgoing one;
        // reaching that needs two or more elapsed windows, i.e. a window that saw no traffic at
        // all, so there is nothing there to lose either.)
        self.slots[incoming].reset();
        self.live.fetch_max(to, Ordering::AcqRel);
        // Two or more windows elapsed, so the other slot is stale too. Cleared *after* publishing,
        // both so the fresh slot goes live as early as possible and because nothing increments this
        // one once it is no longer live.
        if to - from >= 2 {
            self.slots[incoming ^ 1].reset();
        }
    }
}

/// Why a request was throttled — carried out so the caller can label the rejection metric and an
/// operator can tell *which* ceiling tripped (and thus which knob to tune).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Throttled {
    /// A single credential exceeded its per-credential ceiling.
    PerCredential,
    /// Aggregate BYO traffic exceeded the global ceiling.
    ByoGlobal,
}

impl Throttled {
    /// The `ai_rejections_total{reason=…}` label. `PerCredential` keeps the original `"rate_limit"`
    /// label so existing dashboards/alerts are unbroken.
    pub fn label(self) -> &'static str {
        crate::metrics::Rejection::from(self).label()
    }
}

impl From<Throttled> for crate::metrics::Rejection {
    /// So the caller can bump a pre-resolved counter instead of a string-keyed label lookup — the
    /// rate limiter is precisely the path a flood drives at full request rate.
    fn from(t: Throttled) -> Self {
        match t {
            Throttled::PerCredential => crate::metrics::Rejection::RateLimit,
            Throttled::ByoGlobal => crate::metrics::Rejection::RateLimitByoGlobal,
        }
    }
}

pub struct RateLimit {
    /// `(sketch, max_per_window)` for the per-credential tier. `None` disables it.
    per_cred: Option<(WindowedRate, isize)>,
    /// The global BYO aggregate tier — one bucket, so one atomic. `None` disables it.
    byo_global: Option<ByoCounter>,
    /// Epoch the window index is measured from. Fixed at construction, so every tier counts the
    /// same windows off one clock read per request rather than each re-reading the clock.
    start: Instant,
    /// Process-random hash state. The raw credential is reduced to the per-credential sketch key
    /// through this, so the hash is keyed by a per-process secret. Without it the digest would be
    /// precomputable (`DefaultHasher` keys on zeros), letting an attacker craft two tokens that
    /// collide into the same slots and inflate another caller's counter — false throttling. Random
    /// seeding makes that collision search infeasible.
    ///
    /// **`ahash`, not std's SipHash-1-3** — 4.5 ns vs 19.1 ns on a ~70-byte credential, on a path
    /// charged before verify on *every* request. The seeding argument above is unchanged, not
    /// waived: `ahash::RandomState::new()` pulls its key material from the OS RNG once per process
    /// (its default `runtime-rng` feature calls `getrandom`) and permutes it with a per-instance
    /// counter — the same "secret per process" property `std::hash::RandomState` gives. So the
    /// digest is still unpredictable off-process and a colliding token pair still cannot be
    /// precomputed, which is the threat that motivated seeding it in the first place.
    ///
    /// What ahash does *not* carry over is SipHash's proof against an **adaptive** attacker who
    /// searches for a collision online. That search needs an oracle, and this module exposes none:
    /// the sketch's estimate is never returned, and the only observable is a 429 — which requires
    /// already sustaining the per-credential ceiling, i.e. being the flood this limiter is there to
    /// shed. Also note that pingora-limits hashes *again* internally with its own per-row
    /// `ahash::RandomState`s, so a collision here is necessary but not sufficient.
    hasher: RandomState,
}

impl RateLimit {
    /// `per_cred_rps` is the per-credential ceiling; `byo_global_rps` is the aggregate BYO ceiling.
    /// Either tier is disabled by passing `0`. Returns `None` (no limiter at all) only when both are
    /// `0`, so the hot path can skip it entirely.
    pub fn new(per_cred_rps: u32, byo_global_rps: u32) -> Option<Self> {
        if per_cred_rps == 0 && byo_global_rps == 0 {
            return None;
        }
        Some(Self {
            per_cred: (per_cred_rps != 0)
                .then(|| (WindowedRate::new(HASHES, SLOTS), per_cred_rps as isize)),
            byo_global: (byo_global_rps != 0).then(|| ByoCounter::new(byo_global_rps)),
            start: Instant::now(),
            // `RandomState::new()` draws fresh key material from the OS RNG per process.
            hasher: RandomState::new(),
        })
    }

    /// The index of the window `now` falls in — the only thing either tier needs from the clock.
    ///
    /// `saturating_duration_since`, not `-`: with `check_at` the caller supplies `now`, and a
    /// caller that captured it before this limiter was constructed (or a test that hands one back)
    /// must clamp to window 0, not panic under `overflow-checks`.
    #[inline]
    fn window_of(&self, now: Instant) -> u64 {
        (now.saturating_duration_since(self.start).as_millis() as u64) / WINDOW_MS
    }

    /// Charge one request. `managed` is `true` for a verified-path (`bai_…`) credential, `false` for
    /// BYO. Returns `None` when within budget, or `Some(reason)` once a ceiling is crossed — the very
    /// request that crosses the line is the first one rejected (`observe` returns the running total).
    /// The credential itself is never stored; only its seeded digest feeds the per-credential sketch.
    ///
    /// `#[must_use]`: `observe` has already incremented the counters by the time this returns, so a
    /// caller that drops the result has *charged* the request but skipped enforcement — the limiter is
    /// silently bypassed. The crate's `#![deny(unused_must_use)]` only bites with this attribute
    /// present, so it's load-bearing, not decorative.
    #[must_use = "the throttle decision must be enforced — dropping it charges the request but lets it through"]
    pub fn check(&self, raw_credential: &str, managed: bool) -> Option<Throttled> {
        self.check_at(raw_credential, managed, Instant::now())
    }

    /// `check`, with the clock supplied by the caller.
    ///
    /// `request_filter` already takes an `Instant::now()` at the top of every request for its
    /// latency histogram; handing that same reading in makes the limiter's clock read free instead
    /// of duplicating a vDSO `clock_gettime`. It is also what makes the window boundaries testable:
    /// a test pins `now` and gets a deterministic window rather than sleeping for one.
    #[must_use = "the throttle decision must be enforced — dropping it charges the request but lets it through"]
    pub fn check_at(&self, raw_credential: &str, managed: bool, now: Instant) -> Option<Throttled> {
        // One clock read, one divide, shared by both tiers.
        let window = self.window_of(now);
        // Global BYO backstop first: BYO is unverified and upstream-bound, so this is the ceiling that
        // protects our egress IPs from a distinct-token flood. Managed traffic skips it (verified,
        // can't be forged, already bounded per-credential) so it never shares this bucket.
        if !managed {
            if let Some(byo) = &self.byo_global {
                if byo.observe(window) > byo.max {
                    return Some(Throttled::ByoGlobal);
                }
            }
        }
        // Per-credential ceiling: a single leaked/runaway key (managed or BYO), capped before verify.
        if let Some((rate, max)) = &self.per_cred {
            let key = self.hasher.hash_one(raw_credential);
            if rate.observe(key, window) > *max {
                return Some(Throttled::PerCredential);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn both_zero_disables() {
        assert!(RateLimit::new(0, 0).is_none());
    }

    #[test]
    fn per_credential_allows_up_to_ceiling_then_rejects() {
        let rl = RateLimit::new(5, 0).unwrap();
        let cred = "bai_v1.1.payload.sig";
        for _ in 0..5 {
            assert_eq!(rl.check(cred, true), None);
        }
        // 6th request in the same 1s window crosses the per-credential ceiling.
        assert_eq!(rl.check(cred, true), Some(Throttled::PerCredential));
    }

    #[test]
    fn credentials_have_independent_budgets() {
        let rl = RateLimit::new(2, 0).unwrap();
        assert_eq!(rl.check("token-1", false), None);
        assert_eq!(rl.check("token-1", false), None);
        assert_eq!(rl.check("token-1", false), Some(Throttled::PerCredential)); // token-1 exhausted
        assert_eq!(rl.check("token-2", false), None); // a different credential is unaffected
    }

    #[test]
    fn byo_global_caps_distinct_tokens_but_exempts_managed() {
        // Per-credential disabled, global BYO ceiling = 3. A flood of *distinct* BYO tokens (which
        // would each slip past per-credential keying) is still bounded by the shared bucket.
        let rl = RateLimit::new(0, 3).unwrap();
        assert_eq!(rl.check("byo-aaaa", false), None);
        assert_eq!(rl.check("byo-bbbb", false), None);
        assert_eq!(rl.check("byo-cccc", false), None);
        assert_eq!(rl.check("byo-dddd", false), Some(Throttled::ByoGlobal)); // 4th distinct token

        // Managed traffic is exempt from the BYO bucket — a distinct `bai_…` flood is never throttled
        // here (it's bounded by verify failing, not by this ceiling).
        for i in 0..10 {
            assert_eq!(rl.check(&format!("bai_v1.1.p{i}.s{i}"), true), None);
        }
    }

    /// Window rotation, pinned to an explicit clock so it is deterministic and instant: exhaust the
    /// budget, step into the next window, get the whole budget back. The tail also covers the
    /// two-or-more-windows-elapsed path, where *both* slots are stale and have to be cleared.
    #[test]
    fn per_credential_budget_is_restored_by_the_next_window() {
        let rl = RateLimit::new(2, 0).unwrap();
        // Taken *after* construction, so it is at or after the limiter's epoch → window 0.
        let t0 = Instant::now();
        let cred = "bai_v1.1.payload.sig";

        assert_eq!(rl.check_at(cred, true, t0), None);
        assert_eq!(rl.check_at(cred, true, t0), None);
        assert_eq!(rl.check_at(cred, true, t0), Some(Throttled::PerCredential));

        // Window 1: a fresh budget, and the same ceiling at the same place.
        let t1 = t0 + WINDOW;
        assert_eq!(rl.check_at(cred, true, t1), None);
        assert_eq!(rl.check_at(cred, true, t1), None);
        assert_eq!(rl.check_at(cred, true, t1), Some(Throttled::PerCredential));

        // Window 6: an idle gap of more than two windows, so neither slot holds anything live.
        let t6 = t0 + WINDOW * 6;
        assert_eq!(rl.check_at(cred, true, t6), None);
        assert_eq!(rl.check_at(cred, true, t6), None);
        assert_eq!(rl.check_at(cred, true, t6), Some(Throttled::PerCredential));
    }

    /// A thread whose clock read predates a rotation must not reopen the window it closed. In
    /// pingora's `Rate` this is the case that underflows `now - last_reset` (a worker panic under
    /// the workspace's `overflow-checks`); here it must simply charge the live window.
    #[test]
    fn a_stale_clock_read_cannot_reopen_a_closed_window() {
        let rl = RateLimit::new(2, 0).unwrap();
        let t0 = Instant::now();
        let t1 = t0 + WINDOW;
        let cred = "bai_v1.1.payload.sig";

        assert_eq!(rl.check_at(cred, true, t1), None); // rotates into window 1
        // Arriving late with a window-0 reading: charged to window 1, not given a new budget.
        assert_eq!(rl.check_at(cred, true, t0), None);
        assert_eq!(rl.check_at(cred, true, t0), Some(Throttled::PerCredential));
        // And the rotation state is still sane afterwards — window 2 starts clean.
        assert_eq!(rl.check_at(cred, true, t0 + WINDOW * 2), None);
    }

    /// The concurrency regression for `WindowedRate`. Every thread walks its own skewed sequence of
    /// window indices, so rotations are continuously raced and threads routinely arrive with a
    /// reading older than the winner's. Against `pingora_limits::rate::Rate` this is the shape that
    /// underflows `past_ms`; it must run clean here, and the counts must stay bounded by the
    /// ceiling within each window (the sketch is exact for a single key).
    #[test]
    fn racing_rotations_with_out_of_order_clock_reads_are_safe() {
        const THREADS: u64 = 8;
        const STEPS: u64 = 400;

        let rl = RateLimit::new(1_000_000, 0).unwrap();
        let base = Instant::now();
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let rl = &rl;
                s.spawn(move || {
                    for i in 0..STEPS {
                        // 400 ms of simulated time per step (so ~160 rotations), each thread offset
                        // by a different sub-window skew — threads land on either side of a
                        // boundary in a different order every time.
                        let skew = (i + t * 37) % 5;
                        let at = base + Duration::from_millis(i * 400 + skew * 150);
                        let _ = rl.check_at("byo-token", false, at);
                    }
                });
            }
        });
    }

    /// The rotation must not lose counts under contention: with the window pinned, `THREADS × EACH`
    /// concurrent charges against one key have to leave the sketch holding all of them.
    #[test]
    fn concurrent_charges_in_one_window_are_all_counted() {
        const THREADS: usize = 8;
        const EACH: usize = 2_000;

        // Ceiling just under the total, so the last charge must be the one that trips it.
        let rl = RateLimit::new((THREADS * EACH) as u32 - 1, 0).unwrap();
        let now = Instant::now();
        let cred = "bai_v1.1.payload.sig";
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let rl = &rl;
                s.spawn(move || {
                    for _ in 0..EACH {
                        let _ = rl.check_at(cred, true, now);
                    }
                });
            }
        });
        // If a single increment had been dropped the running total would still be under the
        // ceiling and this would come back `None`.
        assert_eq!(rl.check_at(cred, true, now), Some(Throttled::PerCredential));
    }

    /// The BYO bucket rotates on the same window index as the per-credential tier, and its rotation
    /// *is* the counting CAS — so a new window has to hand back the whole budget with no reset step
    /// in between. The tail covers a multi-window idle gap, which for this tier is not a special
    /// case at all (there is no second slot to go stale).
    #[test]
    fn byo_global_budget_is_restored_by_the_next_window() {
        let rl = RateLimit::new(0, 2).unwrap();
        let t0 = Instant::now();

        assert_eq!(rl.check_at("byo-a", false, t0), None);
        assert_eq!(rl.check_at("byo-b", false, t0), None);
        assert_eq!(rl.check_at("byo-c", false, t0), Some(Throttled::ByoGlobal));

        let t1 = t0 + WINDOW;
        assert_eq!(rl.check_at("byo-d", false, t1), None);
        assert_eq!(rl.check_at("byo-e", false, t1), None);
        assert_eq!(rl.check_at("byo-f", false, t1), Some(Throttled::ByoGlobal));

        let t9 = t0 + WINDOW * 9;
        assert_eq!(rl.check_at("byo-g", false, t9), None);
        assert_eq!(rl.check_at("byo-h", false, t9), None);
        assert_eq!(rl.check_at("byo-i", false, t9), Some(Throttled::ByoGlobal));
    }

    /// Same stale-reading hazard as the per-credential tier: a thread whose clock read predates the
    /// rotation must be charged to the live window, not reopen the closed one with a fresh budget.
    #[test]
    fn byo_global_stale_clock_read_cannot_reopen_a_closed_window() {
        let rl = RateLimit::new(0, 2).unwrap();
        let t0 = Instant::now();
        let t1 = t0 + WINDOW;

        assert_eq!(rl.check_at("byo-a", false, t1), None); // opens window 1
        assert_eq!(rl.check_at("byo-b", false, t0), None); // late arrival, charged to window 1
        assert_eq!(rl.check_at("byo-c", false, t0), Some(Throttled::ByoGlobal));
        // Window 2 still opens cleanly afterwards.
        assert_eq!(rl.check_at("byo-d", false, t0 + WINDOW * 2), None);
    }

    /// Unlike the sketch, one bucket in one atomic is *exactly* countable: with the window pinned,
    /// `THREADS × EACH` concurrent charges against a ceiling of half that must admit exactly the
    /// ceiling. Any lost or duplicated increment in the CAS loop moves this number.
    #[test]
    fn byo_global_ceiling_is_exact_under_concurrency() {
        const THREADS: usize = 8;
        const EACH: usize = 4_000;
        const CEILING: u32 = (THREADS * EACH / 2) as u32;

        let rl = RateLimit::new(0, CEILING).unwrap();
        let now = Instant::now();
        let admitted = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let (rl, admitted) = (&rl, &admitted);
                s.spawn(move || {
                    let mut n = 0;
                    for _ in 0..EACH {
                        if rl.check_at("byo-token", false, now).is_none() {
                            n += 1;
                        }
                    }
                    admitted.fetch_add(n, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(admitted.load(Ordering::Relaxed), CEILING as usize);
    }

    #[test]
    fn byo_global_does_not_touch_managed_budget() {
        // With only the global BYO tier on, managed requests pass freely while BYO is being capped.
        let rl = RateLimit::new(0, 1).unwrap();
        assert_eq!(rl.check("byo-1", false), None);
        assert_eq!(rl.check("byo-2", false), Some(Throttled::ByoGlobal)); // BYO bucket exhausted
        assert_eq!(rl.check("bai_v1.1.p.s", true), None); // managed unaffected
    }
}
