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
//! Backed by pingora-limits' `Rate`: count-min-sketch estimators with **fixed memory regardless of
//! key cardinality** (no per-credential entry, no background GC), matching the deny-set's O(denied)
//! ethos. A sketch can *over*estimate a key's rate on hash collision but never under, so a cap is
//! always enforced; `SLOTS` is sized wide enough that overestimation stays negligible.

use pingora_limits::rate::Rate;
use std::hash::{BuildHasher, RandomState};
use std::time::Duration;

/// Count-min sketch dimensions for the per-credential tier. The estimator can only *over*estimate a
/// key's rate (never under — so the cap always holds); the additive error is bounded by
/// `(e / SLOTS) × N`, where `N` is total req/s across *all* credentials on the node. Sized for a
/// single high-volume node: at `SLOTS = 65536` that error stays ≤ ~5 even at ~100k req/s aggregate —
/// far under the per-credential ceiling, so a legitimate caller near its limit isn't false-throttled.
/// `HASHES = 5` sets the tail confidence (≈ `e^-5` ≈ 0.7% of checks may exceed that bound; the
/// estimate is the min over the 5 rows). Memory is `2 × HASHES × SLOTS × 8 B` ≈ **5 MB, fixed**
/// regardless of credential cardinality (no per-key entry, no GC). To resize: `SLOTS ≈ e × peak_N /
/// tolerable_error`.
const SLOTS: usize = 65536;
const HASHES: usize = 5;

/// The rate window. Every ceiling is expressed per this interval, i.e. requests/second.
const WINDOW: Duration = Duration::from_secs(1);

/// The single sketch key the global BYO tier counts everything under (one shared bucket).
const BYO_GLOBAL_KEY: u8 = 0;

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
        match self {
            Throttled::PerCredential => "rate_limit",
            Throttled::ByoGlobal => "rate_limit_byo_global",
        }
    }
}

pub struct RateLimit {
    /// `(sketch, max_per_window)` for the per-credential tier. `None` disables it.
    per_cred: Option<(Rate, isize)>,
    /// `(sketch, max_per_window)` for the global BYO aggregate tier. `None` disables it.
    byo_global: Option<(Rate, isize)>,
    /// Process-random hash state. The raw credential is reduced to the per-credential sketch key
    /// through this, so the SipHash key is per-process and secret. Without it the digest would be
    /// precomputable (`DefaultHasher` keys on zeros), letting an attacker craft two tokens that
    /// collide into the same slots and inflate another caller's counter — false throttling. Random
    /// seeding makes that collision search infeasible.
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
            per_cred: (per_cred_rps != 0).then(|| {
                (
                    Rate::new_with_estimator_config(WINDOW, HASHES, SLOTS),
                    per_cred_rps as isize,
                )
            }),
            // One bucket, so the default estimator is plenty — no need for the wide sketch.
            byo_global: (byo_global_rps != 0).then(|| (Rate::new(WINDOW), byo_global_rps as isize)),
            // `RandomState::new()` draws a fresh SipHash key from the OS RNG per process.
            hasher: RandomState::new(),
        })
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
        // Global BYO backstop first: BYO is unverified and upstream-bound, so this is the ceiling that
        // protects our egress IPs from a distinct-token flood. Managed traffic skips it (verified,
        // can't be forged, already bounded per-credential) so it never shares this bucket.
        if !managed {
            if let Some((rate, max)) = &self.byo_global {
                if rate.observe(&BYO_GLOBAL_KEY, 1) > *max {
                    return Some(Throttled::ByoGlobal);
                }
            }
        }
        // Per-credential ceiling: a single leaked/runaway key (managed or BYO), capped before verify.
        if let Some((rate, max)) = &self.per_cred {
            let key = self.hasher.hash_one(raw_credential);
            if rate.observe(&key, 1) > *max {
                return Some(Throttled::PerCredential);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn byo_global_does_not_touch_managed_budget() {
        // With only the global BYO tier on, managed requests pass freely while BYO is being capped.
        let rl = RateLimit::new(0, 1).unwrap();
        assert_eq!(rl.check("byo-1", false), None);
        assert_eq!(rl.check("byo-2", false), Some(Throttled::ByoGlobal)); // BYO bucket exhausted
        assert_eq!(rl.check("bai_v1.1.p.s", true), None); // managed unaffected
    }
}
