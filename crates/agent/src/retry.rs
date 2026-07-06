//! Whole-run retry policy, shared by `serve.rs` (the `"prompt"` command's auto-retry loop, toggleable
//! via `set_auto_retry`) and `main.rs`'s one-shot `run`/`run --json` (always on — a scripted/cron
//! invocation has no interactive way to disable it, and nothing about retrying transparently is
//! output-visible harm).
//!
//! Complements `agent_core`'s own *within-turn* retry (pre-first-byte in `client.rs`, mid-stream in
//! `agent.rs`) rather than duplicating it: this wraps a whole run that already exhausted those layers
//! and still ended in what looks like a transient error — re-invoking the run from scratch against the
//! same session, up to [`MAX_RUN_RETRIES`] times, with exponential backoff. Matches pi's own default
//! (`agent-session.ts`'s `maxRetries: 3`, `baseDelayMs: 2000`).

/// Ceiling on automatic whole-run retries after a run ends in a transient-looking error.
pub const MAX_RUN_RETRIES: u32 = 3;
pub const RUN_RETRY_BASE_BACKOFF: std::time::Duration = std::time::Duration::from_secs(2);
pub const RUN_RETRY_MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);

/// Exponential backoff for a whole-run retry: `RUN_RETRY_BASE_BACKOFF · 2^(attempt-1)` (± [`jitter`]),
/// capped at `RUN_RETRY_MAX_BACKOFF`. `attempt` is 1-based. Coarser than `agent_core`'s own
/// `mid_stream_backoff` (250ms base/5s cap) — this gates a whole extra model turn's worth of retry, not
/// a resumed stream, so pi's slower 2/4/8s cadence is the better fit.
pub fn backoff(attempt: u32) -> std::time::Duration {
    let exp = RUN_RETRY_BASE_BACKOFF.saturating_mul(1u32 << attempt.saturating_sub(1).min(16));
    jitter(exp, attempt).min(RUN_RETRY_MAX_BACKOFF)
}

/// Safety-audit fix (low severity): applies +/-20% jitter to a computed exponential-backoff duration so
/// that a fleet of concurrent agent processes hitting the same transient failure (503/429) don't retry
/// in perfect lockstep against an already-degraded backend (thundering herd). Applied to the *uncapped*
/// exponential value, before the caller's own `.min(RUN_RETRY_MAX_BACKOFF)` — a saturated attempt (well
/// past the cap) still collapses to exactly the cap after jitter, same as before jitter existed, since
/// even a -20% jittered value there is still far above the cap. That's an accepted, documented
/// trade-off rather than an oversight: jitter's effect is on the non-saturated attempts that matter in
/// practice ([`MAX_RUN_RETRIES`] is only 3 by default, so the cap is rarely even reached), and keeping
/// `RUN_RETRY_MAX_BACKOFF` a true, never-exceeded ceiling is simpler to reason about for callers than a
/// "usually-a-cap-but-up-to-20%-over" one.
///
/// No `rand` crate is a direct (or even workspace) dependency anywhere in this repo — it only ever
/// shows up transitively (checked both this crate's `Cargo.toml` and the workspace root's) — so this
/// reuses the same zero-dependency salt-plus-counter trick `session_store::new_id`/
/// `tools::mod::temp_suffix` already use for "just need per-process/per-call variance, not
/// cryptographic quality": a `RandomState`-seeded salt (draws OS entropy once, fixed for the rest of
/// the process's life) plus a monotonic per-call counter, hashed together with `DefaultHasher` (fixed
/// key, so the *only* source of variance is the salt/counter/attempt inputs, not the hasher itself) and
/// mapped onto `[0.8, 1.2]`. Salt gives cross-process variance (two fleet processes at the same attempt
/// number don't jitter identically); the counter gives cross-call variance (the *same* attempt number,
/// computed twice in one process, doesn't jitter identically either — see
/// `backoff_jitter_varies_across_calls` below).
fn jitter(d: std::time::Duration, attempt: u32) -> std::time::Duration {
    use std::collections::hash_map::{DefaultHasher, RandomState};
    use std::hash::{BuildHasher, Hash, Hasher};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SALT: OnceLock<u64> = OnceLock::new();
    let salt = *SALT.get_or_init(|| RandomState::new().hash_one(0xB0FF0Fu64));

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);

    let mut hasher = DefaultHasher::new();
    salt.hash(&mut hasher);
    seq.hash(&mut hasher);
    attempt.hash(&mut hasher);
    let hashed = hasher.finish();

    // Map the hash's top 32 bits onto [0.8, 1.2] (+/-20%, symmetric around the unjittered value).
    let unit = (hashed >> 32) as f64 / u32::MAX as f64; // [0, 1]
    let factor = 0.8 + unit * 0.4;
    d.mul_f64(factor)
}

/// Raw HTTP status-code digit patterns worth retrying at the whole-run level specifically — a
/// deliberately broader net than `agent_core::agent::is_retryable_mid_stream`'s in-band error-*type*
/// matching. That function excludes these on purpose (a mid-stream message's "500" is more likely an
/// unrelated number — a token count, a byte size — than an actual status code, since the connection
/// already returned 200 by the time it's checking). At the whole-run level an error reaching here has
/// already exhausted every narrower classification `is_retryable_mid_stream` offers (including its own
/// pre-first-byte retry in `client.rs`, on the *real* status code), so pi's broader net
/// (`RETRYABLE_PROVIDER_ERROR_PATTERN`, `packages/ai/src/utils/retry.ts`) is the better fit here — it's
/// pi's *only* layer for a plain HTTP-status failure at all, since SDK-level retry is disabled. A
/// plain-text/HTML error page from a flaky proxy (no recognized JSON `error.type` field) wouldn't match
/// any narrower pattern but plausibly still deserves a retry.
const WHOLE_RUN_RETRYABLE_STATUS_DIGITS: &[&str] = &["429", "500", "502", "503", "504", "524"];

/// Reqwest's own literal, stable wrapper text for a pre-response send failure (`Kind::Request`'s
/// `Display` impl) — refused/reset/timed-out-while-connecting, DNS, TLS. Found live (a real
/// fault-injecting proxy, not a mock): `client.rs`'s pre-first-byte retry already recognizes this exact
/// class as transient (`is_retryable_send_error`) and retries it up to its own budget — but when a
/// connection blip outlasts *that* tight budget too, the resulting `Error::Transport` carries only this
/// generic reqwest string (never a real HTTP status, since no response was ever received), which neither
/// `is_retryable_mid_stream`'s provider-error-text vocabulary nor the raw status-digit list above was
/// ever going to match, silently downgrading "a network blip client.rs already tried and failed to ride
/// out" into a hard, un-retried whole-run failure — exactly the case this whole layer exists to catch.
const TRANSPORT_SEND_FAILURE_MARKER: &str = "error sending request for url";

/// Runtime-configurable variant of the module's own const policy — `run_turn` (`main.rs`) and `serve`'s
/// `"prompt"` auto-retry loop each build one from whichever `--retry-max-retries`/`--retry-base-delay-ms`
/// (`AI_AGENT_RETRY_MAX_RETRIES`/`AI_AGENT_RETRY_BASE_DELAY_MS`) the operator gave — the same two flags
/// that previously configured only `agent_core::client`'s pre-connect/mid-stream layer, leaving this
/// whole-run layer pinned to [`MAX_RUN_RETRIES`]/[`RUN_RETRY_BASE_BACKOFF`] regardless of what the
/// operator asked for. `max_backoff` has no CLI flag of its own (matching the two that exist today), so
/// it stays fixed at [`RUN_RETRY_MAX_BACKOFF`] either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunRetryPolicy {
    pub max_retries: u32,
    pub base_backoff: std::time::Duration,
}

impl Default for RunRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: MAX_RUN_RETRIES,
            base_backoff: RUN_RETRY_BASE_BACKOFF,
        }
    }
}

impl RunRetryPolicy {
    /// Build from the same two optional overrides `--retry-max-retries`/`--retry-base-delay-ms` already
    /// carry through `main.rs`/`serve.rs` — `None` for either falls back to this module's own constant,
    /// exactly like `agent_core::client`'s identical `unwrap_or(MAX_RETRIES)`/`unwrap_or(BASE_BACKOFF)`.
    pub fn from_overrides(
        max_retries: Option<u32>,
        base_backoff: Option<std::time::Duration>,
    ) -> Self {
        Self {
            max_retries: max_retries.unwrap_or(MAX_RUN_RETRIES),
            base_backoff: base_backoff.unwrap_or(RUN_RETRY_BASE_BACKOFF),
        }
    }

    /// Same exponential-backoff-plus-jitter shape as the module-level [`backoff`], just off this
    /// policy's own `base_backoff` instead of the fixed [`RUN_RETRY_BASE_BACKOFF`] constant.
    pub fn backoff(&self, attempt: u32) -> std::time::Duration {
        let exp = self
            .base_backoff
            .saturating_mul(1u32 << attempt.saturating_sub(1).min(16));
        jitter(exp, attempt).min(RUN_RETRY_MAX_BACKOFF)
    }

    /// Like [`from_overrides`](Self::from_overrides), but `enabled: false` wins outright — pi's
    /// `RetrySettings.enabled: false` (`settings-manager.ts:28`) gates its own whole-run
    /// auto-retry-after-error loop entirely, independent of whatever retry *count* is separately
    /// configured. This is the concrete backing for `main.rs`'s `--no-retry` CLI flag (Task #52,
    /// pi-parity fix): a clearer, more discoverable alias for `max_retries: 0` — which already fully
    /// disables this loop today (`run_turn`'s own `attempt < max_retries` gate never lets a single
    /// retry through when the ceiling is zero — see `run_retry_policy_disabling_forces_zero_max_retries_
    /// regardless_of_max_retries_override` below) — rather than a wholly separate boolean the rest of
    /// this module, and every caller reading `max_retries`, would need to keep in sync with it.
    ///
    /// `max_retries` is still consulted when `enabled` is `true` (ordinary `from_overrides` behavior);
    /// `base_backoff` is always honored either way, since a disabled policy's `backoff` is never
    /// actually called (nothing here ever retries), but keeping it set is simpler than special-casing a
    /// bogus value nothing reads.
    pub fn from_overrides_with_enabled(
        enabled: bool,
        max_retries: Option<u32>,
        base_backoff: Option<std::time::Duration>,
    ) -> Self {
        if !enabled {
            return Self {
                max_retries: 0,
                base_backoff: base_backoff.unwrap_or(RUN_RETRY_BASE_BACKOFF),
            };
        }
        Self::from_overrides(max_retries, base_backoff)
    }
}

/// Whether a whole run that ended in `Err` is worth automatically re-invoking from scratch — a superset
/// of `agent_core::agent::is_retryable_mid_stream` (every mid-stream-worth-retrying error is also worth
/// a whole-run retry) plus [`WHOLE_RUN_RETRYABLE_STATUS_DIGITS`] and
/// [`TRANSPORT_SEND_FAILURE_MARKER`], appropriate only at this outer layer. Excludes a context-overflow
/// rejection either way (`is_context_overflow`, `pub` for exactly this) — that's compact-and-retry's
/// signal, not this one's; retrying it blindly here would just fail identically again without
/// compacting first.
pub fn is_retryable_whole_run(e: &agent_core::Error) -> bool {
    if agent_core::agent::is_retryable_mid_stream(e) {
        return true;
    }
    let agent_core::Error::Transport(msg) = e else {
        return false;
    };
    if agent_core::agent::is_context_overflow(e) {
        return false;
    }
    // A quota/billing-exhaustion 429 (`"gateway returned 429 …: insufficient_quota…"`) still contains
    // the raw digits `WHOLE_RUN_RETRYABLE_STATUS_DIGITS` matches on below, but retrying it can never
    // succeed until the account itself changes — the same reason `client.rs`'s pre-first-byte 429
    // handling and `is_retryable_mid_stream`'s allowlist both already exclude it. Reuse `client.rs`'s
    // own heuristic rather than duplicating its phrase list here.
    if agent_core::client::is_quota_exhausted(msg) {
        return false;
    }
    msg.contains(TRANSPORT_SEND_FAILURE_MARKER)
        || WHOLE_RUN_RETRYABLE_STATUS_DIGITS
            .iter()
            .any(|d| msg.contains(d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::Error;

    /// Asserts `actual` falls within the +/-20% jitter band around `nominal` (the unjittered
    /// exponential value), inclusive. Used everywhere a test previously asserted an exact backoff
    /// value — jitter means the exact value is no longer deterministic, only its range is.
    fn assert_within_jitter(actual: std::time::Duration, nominal: std::time::Duration) {
        let lo = nominal.mul_f64(0.8);
        let hi = nominal.mul_f64(1.2);
        assert!(
            actual >= lo && actual <= hi,
            "expected {actual:?} within +/-20% of {nominal:?} (i.e. [{lo:?}, {hi:?}])"
        );
    }

    #[test]
    fn backoff_doubles_then_caps() {
        assert_within_jitter(backoff(1), std::time::Duration::from_secs(2));
        assert_within_jitter(backoff(2), std::time::Duration::from_secs(4));
        assert_within_jitter(backoff(3), std::time::Duration::from_secs(8));
        // Attempt 10's unjittered value (2s · 2^9 = 1024s) is so far past the cap that even a -20%
        // jittered value is still far above it, so this stays an exact assertion — see `jitter`'s doc
        // comment on why a saturated attempt isn't expected to show variance.
        assert_eq!(backoff(10), RUN_RETRY_MAX_BACKOFF);
    }

    #[test]
    fn backoff_jitter_varies_across_calls_but_stays_in_range() {
        // Safety-audit fix: proves jitter is actually applied, not just structurally present — the
        // same attempt number, computed repeatedly, must not always produce the identical duration
        // (the pre-fix thundering-herd bug), while every value still lands within the documented
        // +/-20% band around the unjittered exponential value.
        let nominal = std::time::Duration::from_secs(4); // attempt 2's unjittered value.
        let samples: Vec<_> = (0..50).map(|_| backoff(2)).collect();
        for &s in &samples {
            assert_within_jitter(s, nominal);
        }
        assert!(
            samples.windows(2).any(|w| w[0] != w[1]),
            "expected varying backoff durations across repeated calls for the same attempt, got \
             identical values every time: {samples:?}"
        );
    }

    #[test]
    fn run_retry_policy_backoff_jitter_varies_across_calls_but_stays_in_range() {
        // Same regression as `backoff_jitter_varies_across_calls_but_stays_in_range`, for the
        // runtime-configurable `RunRetryPolicy` variant.
        let policy = RunRetryPolicy::default();
        let nominal = std::time::Duration::from_secs(4); // attempt 2's unjittered value.
        let samples: Vec<_> = (0..50).map(|_| policy.backoff(2)).collect();
        for &s in &samples {
            assert_within_jitter(s, nominal);
        }
        assert!(
            samples.windows(2).any(|w| w[0] != w[1]),
            "expected varying backoff durations across repeated calls for the same attempt, got \
             identical values every time: {samples:?}"
        );
    }

    #[test]
    fn run_retry_policy_defaults_match_the_bare_constants() {
        let policy = RunRetryPolicy::default();
        assert_eq!(policy.max_retries, MAX_RUN_RETRIES);
        assert_eq!(policy.base_backoff, RUN_RETRY_BASE_BACKOFF);
        assert_within_jitter(policy.backoff(1), std::time::Duration::from_secs(2));
        assert_within_jitter(policy.backoff(3), std::time::Duration::from_secs(8));
    }

    #[test]
    fn run_retry_policy_from_overrides_honors_operator_supplied_values() {
        // Task #50: `--retry-max-retries`/`--retry-base-delay-ms` must reach this whole-run layer, not
        // just `agent_core::client`'s pre-connect/mid-stream one.
        let policy =
            RunRetryPolicy::from_overrides(Some(7), Some(std::time::Duration::from_millis(100)));
        assert_eq!(policy.max_retries, 7);
        assert_eq!(policy.base_backoff, std::time::Duration::from_millis(100));
        assert_within_jitter(policy.backoff(1), std::time::Duration::from_millis(100));
        assert_within_jitter(policy.backoff(2), std::time::Duration::from_millis(200));
        // `max_backoff` has no override of its own — still caps at the fixed module constant even with
        // a custom base.
        assert_eq!(policy.backoff(20), RUN_RETRY_MAX_BACKOFF);
    }

    #[test]
    fn run_retry_policy_from_overrides_falls_back_to_defaults_when_absent() {
        let policy = RunRetryPolicy::from_overrides(None, None);
        assert_eq!(policy, RunRetryPolicy::default());
    }

    #[test]
    fn from_overrides_with_enabled_true_behaves_exactly_like_from_overrides() {
        // Task #52: `--no-retry` absent (`enabled: true`) must be a complete no-op over the existing
        // `from_overrides` behavior — no new precedence or defaulting quirks introduced for the common
        // case.
        assert_eq!(
            RunRetryPolicy::from_overrides_with_enabled(true, Some(5), None),
            RunRetryPolicy::from_overrides(Some(5), None)
        );
        assert_eq!(
            RunRetryPolicy::from_overrides_with_enabled(true, None, None),
            RunRetryPolicy::default()
        );
    }

    #[test]
    fn from_overrides_with_enabled_false_forces_zero_max_retries_regardless_of_max_retries_override() {
        // Task #52 (pi-parity feature): `--no-retry` must win outright even if `--retry-max-retries 7`
        // (or a stored `default_retry_max_retries`) was *also* given — matching pi's own
        // `RetrySettings.enabled: false` precedence, an explicit "off" always means off.
        let disabled = RunRetryPolicy::from_overrides_with_enabled(
            false,
            Some(7),
            Some(std::time::Duration::from_millis(500)),
        );
        assert_eq!(disabled.max_retries, 0);
    }

    #[test]
    fn a_disabled_run_retry_policy_never_lets_run_turns_own_gate_retry_once() {
        // Confirms the mechanism `from_overrides_with_enabled(false, ..)`'s doc comment relies on:
        // `main.rs::run_turn`'s own retry loop gates on `attempt < policy.max_retries` starting from
        // `attempt == 0` — a zero ceiling means that check is `0 < 0`, always false, so the loop's
        // first (and only) attempt is never retried, for any error, retryable-looking or not.
        let disabled = RunRetryPolicy::from_overrides_with_enabled(false, Some(99), None);
        let attempt = 0u32;
        assert!(attempt >= disabled.max_retries);
    }

    #[test]
    fn whole_run_retries_a_plain_http_status_digit_mid_stream_never_would() {
        // Deliberately no retryable *prose* here (contrast the free-text fallback tests in
        // `agent_core::agent`'s test module) — this message's only retryable-looking signal is the raw
        // "503" digit, which is the one thing the mid-stream layer excludes on purpose.
        let e = Error::Transport("gateway returned 503: unexpected response from upstream".into());
        assert!(
            !agent_core::agent::is_retryable_mid_stream(&e),
            "mid-stream layer deliberately excludes raw status digits"
        );
        assert!(
            is_retryable_whole_run(&e),
            "whole-run layer must catch what the narrower one misses"
        );
    }

    #[test]
    fn whole_run_retries_a_cloudflare_524_gateway_timeout() {
        // pi-parity: pi's whole-run classifier includes the literal "524" (Cloudflare's own gateway
        // timeout code, distinct from the origin's own 502/503/504) alongside the other raw status
        // digits — same rationale as the 503 case just above, ported verbatim rather than left out.
        // Deliberately avoids the word "timeout" in the message (unlike Cloudflare's real body text)
        // since that free-text phrase is itself independently mid-stream-retryable — this fixture
        // isolates the raw "524" digit as the *only* retryable-looking signal, same as the 503 case.
        let e = Error::Transport("gateway returned 524: unexpected response from upstream".into());
        assert!(
            !agent_core::agent::is_retryable_mid_stream(&e),
            "mid-stream layer deliberately excludes raw status digits"
        );
        assert!(
            is_retryable_whole_run(&e),
            "whole-run layer must retry a Cloudflare 524 gateway timeout"
        );
    }

    #[test]
    fn whole_run_retries_client_rs_own_exhausted_transport_send_failure() {
        // Found live (a real fault-injecting proxy repeatedly dropping connections): client.rs's own
        // pre-first-byte retry (`is_retryable_send_error`) already recognizes a reset-mid-send as
        // transient and retries it — but its own budget is tight (4 attempts), and once that's
        // exhausted the resulting `Error::Transport` carries only reqwest's generic
        // `TRANSPORT_SEND_FAILURE_MARKER` text, with no HTTP status digit at all (no response was ever
        // received) and no provider-error prose either — matching neither `is_retryable_mid_stream`'s
        // vocabulary nor the raw-status-digit fallback, silently turning a network blip that outlasted
        // the fast inner retry into a hard, un-retried whole-run failure.
        let msg = "error sending request for url (http://127.0.0.1:9/v1/messages)";
        let e = Error::Transport(msg.into());
        assert!(
            !agent_core::agent::is_retryable_mid_stream(&e),
            "this is our own client's generic reqwest text, not provider error prose"
        );
        assert!(
            !WHOLE_RUN_RETRYABLE_STATUS_DIGITS
                .iter()
                .any(|d| msg.contains(d)),
            "no HTTP status digit is present at all — no response was ever received"
        );
        assert!(
            is_retryable_whole_run(&e),
            "an exhausted pre-connect transport failure must still be retried at the whole-run layer"
        );
    }

    #[test]
    fn whole_run_does_not_retry_a_quota_exhausted_429_even_though_it_contains_the_digit() {
        // A-M8: mirrors `client.rs::a_429_with_quota_exhaustion_body_is_not_retried`'s exact message
        // shape (`"gateway returned {status}: {body}"`) — this must fail the same way `client.rs`'s own
        // pre-first-byte 429 handling and `is_retryable_mid_stream`'s allowlist already do, instead of
        // slipping through `WHOLE_RUN_RETRYABLE_STATUS_DIGITS`'s raw "429" substring match, which
        // otherwise can't tell a quota rejection from ordinary rate limiting.
        let e = Error::Transport(
            "gateway returned 429 Too Many Requests: {\"error\":{\"type\":\"insufficient_quota\",\"message\":\"You exceeded your current quota\"}}".into(),
        );
        assert!(
            !agent_core::agent::is_retryable_mid_stream(&e),
            "mid-stream layer's allowlist already excludes this"
        );
        assert!(
            !is_retryable_whole_run(&e),
            "a quota-exhausted 429 must not be retried at the whole-run layer either — retrying it can \
             never succeed until the account itself changes"
        );
    }

    #[test]
    fn whole_run_still_retries_a_plain_429_without_quota_language() {
        // The flip side of the test above: don't regress
        // `whole_run_retries_a_plain_http_status_digit_mid_stream_never_would`'s guarantee — an ordinary
        // rate-limit 429 (no quota/billing phrase) must still fall through to the raw-status-digit
        // fallback and be retried.
        let e = Error::Transport(
            "gateway returned 429 Too Many Requests: {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Too many requests, please slow down\"}}".into(),
        );
        assert!(
            is_retryable_whole_run(&e),
            "an ordinary rate-limit 429 without quota language must still be retried"
        );
    }

    #[test]
    fn whole_run_still_retries_everything_mid_stream_does() {
        let e = Error::Transport("Anthropic stream ended before message_stop".into());
        assert!(agent_core::agent::is_retryable_mid_stream(&e));
        assert!(is_retryable_whole_run(&e), "must be a strict superset");
    }

    #[test]
    fn whole_run_does_not_retry_a_context_overflow_even_if_it_contains_a_retryable_digit() {
        let e = Error::Transport(
            "prompt is too long: 503000 tokens exceeds the 500000 token context window".into(),
        );
        assert!(
            !is_retryable_whole_run(&e),
            "an overflow error must not be retried blindly just because its message happens to \
             contain a status-code-shaped substring"
        );
    }

    #[test]
    fn whole_run_does_not_retry_a_non_retryable_error() {
        let e = Error::Transport("invalid_request_error: missing required field".into());
        assert!(!is_retryable_whole_run(&e));
    }

    #[test]
    fn whole_run_inherits_the_mid_stream_free_text_fallback() {
        // LOW pi-parity gap (fixed): `is_retryable_mid_stream`'s new free-text prose fallback (for a
        // provider error with no recognized `error.type`) must reach this outer layer too, the same
        // way every other mid-stream-retryable shape already does — this isn't a separate pattern list
        // to keep in sync, just confirming the inheritance actually holds for the new addition.
        let e = Error::Transport("provider returned error: Service Unavailable, try later".into());
        assert!(agent_core::agent::is_retryable_mid_stream(&e));
        assert!(is_retryable_whole_run(&e));
    }
}
