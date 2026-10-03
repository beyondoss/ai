//! Lock-free circuit breaker for protecting external service calls.
//!
//! This implementation is provably race-free through:
//! 1. Atomic words for all mutable state (no multi-variable coordination)
//! 2. Compare-and-swap loops for all state transitions
//! 3. Monotonic timestamps for timeout detection — `CLOCK_MONOTONIC`, never the wall clock, so an
//!    NTP step can't invert a timeout decision (see `CircuitBreaker::system_clock`)
//!
//! # States
//!
//! ```text
//!                 failure_threshold reached
//!     ┌─────────┐ ──────────────────────────► ┌────────┐
//!     │ Closed  │                             │  Open  │
//!     └─────────┘ ◄────────────────────────── └────────┘
//!          ▲        success in half-open           │
//!          │                                       │ reset_timeout elapsed
//!          │        ┌─────────────┐                │
//!          └─────── │  Half-Open  │ ◄──────────────┘
//!            success└─────────────┘
//!                         │
//!                         │ failure
//!                         ▼
//!                    back to Open
//! ```
//!
//! # Failure Policies
//!
//! Two failure detection policies are supported:
//!
//! - **Consecutive**: Opens after N failures in a row. Any success resets the count.
//!   Good for detecting complete backend failures.
//!
//! - **Windowed**: Opens when, within a time window, at least N failures have been seen **and**
//!   failures are at least half of the outcomes recorded since the window's first failure. A
//!   success does not erase the count, so a backend failing every other request (a partial
//!   brownout) opens it; a busy healthy backend with background errors does not, because its
//!   successes outnumber them. The window is fixed, not rolling: it starts at its first failure,
//!   and the first failure after it expires starts a new one with both counts reset. The two
//!   counts saturate at the same value, so a failure majority can always trip it.
//!
//! # Example
//!
//! ```rust
//! use beyond_ai::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, FailurePolicy};
//! use std::time::Duration;
//!
//! // Consecutive failures (default)
//! let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
//!
//! // Windowed failures (better for edge proxies)
//! let cb = CircuitBreaker::new(
//!     CircuitBreakerConfig::windowed(3, Duration::from_secs(10))
//!         .reset_timeout(Duration::from_secs(30))
//! );
//!
//! // Before calling external service
//! if cb.allow().is_err() {
//!     // return Err("service temporarily unavailable");
//! }
//!
//! // match call_external_service().await {
//! //     Ok(result) => {
//! //         cb.record_success();
//! //         Ok(result)
//! //     }
//! //     Err(e) if is_connectivity_error(&e) => {
//! //         cb.record_failure();
//! //         Err(e)
//! //     }
//! //     Err(e) => Err(e), // Don't count business logic errors
//! // }
//! ```

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How failures are counted before opening the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailurePolicy {
    /// N consecutive failures opens the circuit. Any success resets the count.
    Consecutive {
        /// Number of consecutive failures before opening.
        threshold: u32,
    },
    /// N failures within the window, making up at least half of the window's outcomes, opens the
    /// circuit. Outcomes older than the window are forgotten.
    Windowed {
        /// Number of failures within the window before opening.
        threshold: u32,
        /// Time window for counting failures.
        window: Duration,
    },
}

impl Default for FailurePolicy {
    fn default() -> Self {
        FailurePolicy::Consecutive { threshold: 5 }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// How failures are counted.
    pub failure_policy: FailurePolicy,
    /// Time to wait in open state before transitioning to half-open.
    pub reset_timeout: Duration,
    /// Number of probe requests allowed in half-open state.
    pub half_open_permits: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_policy: FailurePolicy::default(),
            reset_timeout: Duration::from_secs(30),
            half_open_permits: 3,
        }
    }
}

impl CircuitBreakerConfig {
    /// Create a config with consecutive failure detection.
    pub fn consecutive(threshold: u32) -> Self {
        Self {
            failure_policy: FailurePolicy::Consecutive { threshold },
            ..Default::default()
        }
    }

    /// Create a config with windowed failure detection.
    pub fn windowed(threshold: u32, window: Duration) -> Self {
        Self {
            failure_policy: FailurePolicy::Windowed { threshold, window },
            ..Default::default()
        }
    }

    /// Set the reset timeout (time in open state before half-open).
    pub fn reset_timeout(mut self, timeout: Duration) -> Self {
        self.reset_timeout = timeout;
        self
    }

    /// Set the number of half-open permits.
    pub fn half_open_permits(mut self, permits: u32) -> Self {
        self.half_open_permits = permits;
        self
    }
}

/// Lock-free circuit breaker.
///
/// All state is packed into a single 64-bit atomic:
/// - Bits 62-63: State (0=closed, 1=open, 2=half-open)
/// - Bits 48-61: Failure count (14 bits, max 16383) in CLOSED. In OPEN and HALF_OPEN, the probe
///   **generation**: bumped on every OPEN → HALF_OPEN and every stalled-probe reclaim, and carried
///   in each [`Permit`] `allow()` hands out, so a permit that was reclaimed (or issued before the
///   breaker half-opened) can no longer release a probe permit or decide the probe (D256).
/// - Bits 32-47: Half-open permits remaining (16 bits) in HALF_OPEN. In CLOSED windowed mode, the
///   successes recorded since the current window's first failure (saturating at the failure count's
///   14-bit cap, so failures can always catch up), which is what makes
///   the trip decision a failure *rate*.
/// - Bits 0-31: Timestamp (seconds since a process-wide **monotonic** base — see
///   `CircuitBreaker::system_clock` — so it would take 136 years of process uptime to wrap). In
///   OPEN it's when the circuit opened (drives the reset timeout); in CLOSED windowed mode it
///   doubles as the current failure window's start. Because the timestamp lives in the same word, a
///   windowed failure can reset-the-window and increment-the-count in a **single** CAS — so
///   concurrent failures at a window boundary can never each independently reset to 1 and drop one
///   another.
///
/// This packing ensures all state transitions are atomic via single CAS operations.
pub struct CircuitBreaker {
    /// Packed state word.
    state: AtomicU64,
    /// Configuration (immutable after construction).
    config: CircuitBreakerConfig,
    /// Clock function for getting current time in seconds. Must be non-decreasing; production uses
    /// [`CircuitBreaker::system_clock`], tests inject a fixed or hand-stepped one. A clock that does
    /// move backwards is still *safe* — every elapsed computation saturates at zero — it just makes
    /// the breaker hold its current state until the clock catches back up.
    clock: fn() -> u64,
}

/// Process-wide monotonic base for [`CircuitBreaker::system_clock`].
///
/// Initialized once, on first use (the first breaker construction), so every packed timestamp in the
/// counts seconds from the *same* origin and stays directly comparable across per-provider breakers.
/// Keeping it a static (rather than an `Instant` per breaker) is what lets `clock` remain a bare
/// `fn() -> u64`: the injection point tests rely on needs no per-instance state behind it.
static MONOTONIC_BASE: LazyLock<Instant> = LazyLock::new(Instant::now);

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("state", &self.state)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

// State encoding constants
const STATE_CLOSED: u64 = 0;
const STATE_OPEN: u64 = 1;
const STATE_HALF_OPEN: u64 = 2;

const STATE_SHIFT: u32 = 62;
const STATE_MASK: u64 = 0b11;

const FAILURE_SHIFT: u32 = 48;
const FAILURE_MASK: u64 = 0x3FFF; // 14 bits

const PERMIT_SHIFT: u32 = 32;
const PERMIT_MASK: u64 = 0xFFFF; // 16 bits

const TIMESTAMP_MASK: u64 = 0xFFFF_FFFF; // 32 bits

/// Largest failure threshold the packed 14-bit count can reach. A higher one is clamped here: past
/// it the count would wrap to 0 on `pack` and the breaker could never open — silently disabled.
pub const MAX_FAILURE_THRESHOLD: u32 = FAILURE_MASK as u32;

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self::with_clock(config, Self::system_clock)
    }

    /// Create a circuit breaker with a custom clock (for testing).
    pub fn with_clock(config: CircuitBreakerConfig, clock: fn() -> u64) -> Self {
        let initial = Self::pack(STATE_CLOSED, 0, 0, clock());
        Self {
            state: AtomicU64::new(initial),
            config,
            clock,
        }
    }

    /// Monotonic clock returning seconds since the process-wide [`MONOTONIC_BASE`].
    ///
    /// Deliberately **not** wall-clock time. `SystemTime` reads `CLOCK_REALTIME`, which is steppable:
    /// when NTP corrects a host whose clock had drifted forward (cloud instances do step), `now`
    /// jumps *backwards* past timestamps already packed into the state word, and both time-based
    /// decisions invert at exactly the moment the fleet is least healthy — every open breaker is
    /// forced half-open at once, and a windowed failure count can never accrue past 1 so the breaker
    /// stops tripping at all. `Instant` reads `CLOCK_MONOTONIC`: the same vDSO cost, but it cannot
    /// step or go backwards. The trade is that the 32-bit field now counts seconds since process
    /// start rather than since the epoch — which is strictly better, since wrapping it would take
    /// 136 years of *uptime* instead of landing on a fixed calendar date.
    #[inline]
    fn system_clock() -> u64 {
        MONOTONIC_BASE.elapsed().as_secs() & TIMESTAMP_MASK
    }

    /// Get current time from the configured clock.
    #[inline]
    fn now_secs(&self) -> u64 {
        (self.clock)()
    }

    /// Pack state components into a single u64.
    #[inline]
    fn pack(state: u64, failures: u64, permits: u64, timestamp: u64) -> u64 {
        ((state & STATE_MASK) << STATE_SHIFT)
            | ((failures & FAILURE_MASK) << FAILURE_SHIFT)
            | ((permits & PERMIT_MASK) << PERMIT_SHIFT)
            | (timestamp & TIMESTAMP_MASK)
    }

    /// Unpack a u64 into state components.
    #[inline]
    fn unpack(packed: u64) -> (u64, u64, u64, u64) {
        let state = (packed >> STATE_SHIFT) & STATE_MASK;
        let failures = (packed >> FAILURE_SHIFT) & FAILURE_MASK;
        let permits = (packed >> PERMIT_SHIFT) & PERMIT_MASK;
        let timestamp = packed & TIMESTAMP_MASK;
        (state, failures, permits, timestamp)
    }

    /// Check if a request should be allowed through the circuit.
    ///
    /// Returns `Ok(())` if the request is allowed, `Err(CircuitOpen)` if the
    /// circuit is open and the request should be rejected.
    ///
    /// In half-open state, this atomically decrements the permit count. The returned [`Permit`]
    /// is what the attempt later resolves with [`record_success_for`](Self::record_success_for),
    /// [`record_failure_for`](Self::record_failure_for) or [`release`](Self::release).
    pub fn allow(&self) -> Result<Permit, CircuitOpen> {
        loop {
            let packed = self.state.load(Ordering::Acquire);
            let (state, failures, permits, timestamp) = Self::unpack(packed);

            match state {
                STATE_CLOSED => return Ok(Permit(NOT_A_PROBE)),

                STATE_OPEN => {
                    let now = self.now_secs();
                    // `saturating_sub`, not `wrapping_sub`: `system_clock` is monotonic so
                    // `now >= timestamp` always holds, but an injected clock must not be able to
                    // turn a backwards step into a near-2^32 `elapsed` that trivially clears any
                    // reset timeout and forces every open breaker half-open in lockstep. Saturating
                    // at zero fails in the safe direction — the circuit simply stays open until the
                    // clock genuinely advances past the timeout. Nothing is lost to the missing
                    // wrap: the field can only wrap after 136 years of process uptime.
                    let elapsed = now.saturating_sub(timestamp);

                    if elapsed >= self.config.reset_timeout.as_secs() {
                        // Timeout elapsed, try to transition to half-open, in a new generation:
                        // a permit from an earlier half-open (or from CLOSED) is not this probe.
                        let new_packed = Self::pack(
                            STATE_HALF_OPEN,
                            failures + 1,
                            u64::from(self.config.half_open_permits),
                            now,
                        );

                        match self.state.compare_exchange_weak(
                            packed,
                            new_packed,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => continue,  // Transitioned, retry allow()
                            Err(_) => continue, // Someone else modified, retry
                        }
                    }
                    return Err(CircuitOpen);
                }

                STATE_HALF_OPEN => {
                    if permits == 0 {
                        // Every probe permit is out. One that has gone unresolved for a whole
                        // reset timeout belongs to an attempt that stalled (a header that never
                        // came): reclaim it and admit a fresh probe, rather than shed this
                        // provider's traffic for as long as that one attempt lives. Every permit
                        // out is that old (the timestamp is the newest handout), so all of them
                        // are reclaimed at once by bumping the generation: the stalled attempts'
                        // later `release` or outcome is then a no-op, never a second probe (D256).
                        let now = self.now_secs();
                        let reset = self.config.reset_timeout.as_secs();
                        if reset > 0 && now.saturating_sub(timestamp) >= reset {
                            let generation = (failures + 1) & FAILURE_MASK;
                            let new_packed = Self::pack(
                                STATE_HALF_OPEN,
                                generation,
                                u64::from(self.config.half_open_permits.saturating_sub(1)),
                                now,
                            );
                            if self
                                .state
                                .compare_exchange_weak(
                                    packed,
                                    new_packed,
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                )
                                .is_ok()
                            {
                                return Ok(Permit::probe(generation));
                            }
                            continue;
                        }
                        return Err(CircuitOpen);
                    }

                    // Try to claim a permit, stamping when it was handed out: the stalled-probe
                    // reclaim above measures from the newest handout, not from when the breaker
                    // half-opened. Otherwise a permit given back by `release` and claimed again
                    // looks a whole reset timeout old at once, and a second probe is admitted
                    // beside it.
                    let new_packed =
                        Self::pack(STATE_HALF_OPEN, failures, permits - 1, self.now_secs());

                    match self.state.compare_exchange_weak(
                        packed,
                        new_packed,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => return Ok(Permit::probe(failures)),
                        Err(_) => continue, // CAS failed, retry
                    }
                }

                _ => {
                    // Invalid state, reset to closed
                    let new_packed = Self::pack(STATE_CLOSED, 0, 0, self.now_secs());
                    let _ = self.state.compare_exchange(
                        packed,
                        new_packed,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    return Ok(Permit(NOT_A_PROBE));
                }
            }
        }
    }

    /// Record a successful request.
    ///
    /// In closed state, resets the failure counter (and window for windowed mode).
    /// In half-open state, closes the circuit (service is healthy again).
    ///
    /// An outcome not tied to a permit (an error found in a 200 body after its head resolved the
    /// permit). An attempt resolving the permit `allow()` gave it uses
    /// [`record_success_for`](Self::record_success_for).
    pub fn record_success(&self) {
        self.success(None);
    }

    /// Resolve `permit` as a success. In HALF_OPEN only the current generation's probe decides:
    /// a reclaimed probe's late answer, or one admitted before the breaker half-opened, is a no-op.
    pub fn record_success_for(&self, permit: Permit) {
        self.success(Some(permit));
    }

    fn success(&self, scope: Option<Permit>) {
        // Fast path: a healthy CLOSED breaker with no accrued failures is the overwhelmingly common
        // case — every successful response calls this. Bail before any write so high-throughput
        // success traffic to one provider doesn't bounce its breaker cache line across every worker
        // thread on each response. The CLOSED timestamp is only read as a window anchor by the *next*
        // failure (which re-anchors a fresh window when failures==0 regardless), so not refreshing it
        // on an already-clean breaker changes no observable behavior.
        let (state, failures, _, _) = Self::unpack(self.state.load(Ordering::Acquire));
        if state == STATE_CLOSED && failures == 0 {
            return;
        }

        let now = self.now_secs();
        loop {
            let packed = self.state.load(Ordering::Acquire);
            let (state, failures, successes, ts) = Self::unpack(packed);

            let new_packed = match (state, &self.config.failure_policy) {
                // Windowed: a success inside a window that has seen failures is counted, not a
                // reset — the trip decision is a failure rate (see `record_failure_windowed`).
                // Once the window has expired the slate is wiped.
                (STATE_CLOSED, FailurePolicy::Windowed { window, .. })
                    if now.saturating_sub(ts) < window.as_secs() =>
                {
                    // Saturate at the failure count's own cap, not the field's: past it the
                    // failures (14 bits) could never catch up, and the rate rule never trip.
                    if successes >= FAILURE_MASK {
                        return;
                    }
                    Self::pack(STATE_CLOSED, failures, successes + 1, ts)
                }
                (STATE_HALF_OPEN, _) if Self::stale(scope, failures) => return,
                // Consecutive, an expired window, or HALF_OPEN (a probe succeeded → close the
                // circuit): reset the counts and re-anchor the window.
                (STATE_CLOSED | STATE_HALF_OPEN, _) => Self::pack(STATE_CLOSED, 0, 0, now),
                // OPEN records no success; any other word is invalid.
                _ => return,
            };

            match self.state.compare_exchange_weak(
                packed,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(_) => continue,
            }
        }
    }

    /// Record a failed request.
    ///
    /// In closed state, increments the failure counter and opens the circuit
    /// if the threshold is reached.
    /// In half-open state, reopens the circuit immediately.
    ///
    /// An outcome not tied to a permit; see [`record_success`](Self::record_success).
    pub fn record_failure(&self) {
        self.failure(None);
    }

    /// Resolve `permit` as a failure. In HALF_OPEN only the current generation's probe decides,
    /// as for [`record_success_for`](Self::record_success_for).
    pub fn record_failure_for(&self, permit: Permit) {
        self.failure(Some(permit));
    }

    fn failure(&self, scope: Option<Permit>) {
        match &self.config.failure_policy {
            FailurePolicy::Consecutive { threshold } => {
                self.record_failure_consecutive(*threshold, scope);
            }
            FailurePolicy::Windowed { threshold, window } => {
                self.record_failure_windowed(*threshold, window.as_secs(), scope);
            }
        }
    }

    /// Whether `scope` is a permit from another generation than the HALF_OPEN word's `generation`.
    /// An unscoped outcome (`None`) is never stale.
    #[inline]
    fn stale(scope: Option<Permit>, generation: u64) -> bool {
        scope.is_some_and(|p| u64::from(p.0) != generation)
    }

    /// The OPEN word for a breaker opening at `now`. Its generation is seeded from the clock, not
    /// 0, so a permit that outlives a close and a later re-open is unlikely to match the new one.
    #[inline]
    fn open_word(now: u64) -> u64 {
        Self::pack(STATE_OPEN, now, 0, now)
    }

    /// Record failure with consecutive failure tracking.
    fn record_failure_consecutive(&self, threshold: u32, scope: Option<Permit>) {
        let threshold = threshold.min(MAX_FAILURE_THRESHOLD);
        // Read the clock once, above the retry loop — the timestamp this failure stamps is the time
        // the failure happened, not the time its CAS finally landed, and re-reading a vDSO clock on
        // every contended retry is pure waste. Matches `record_failure_windowed`.
        let now = self.now_secs();

        loop {
            let packed = self.state.load(Ordering::Acquire);
            let (state, failures, _, _) = Self::unpack(packed);

            let new_packed = match state {
                STATE_CLOSED => {
                    let new_failures = failures + 1;
                    if new_failures >= u64::from(threshold) {
                        Self::open_word(now)
                    } else {
                        Self::pack(STATE_CLOSED, new_failures, 0, now)
                    }
                }
                STATE_HALF_OPEN if Self::stale(scope, failures) => return,
                // Keep the generation; the next half-open bumps it.
                STATE_HALF_OPEN => Self::pack(STATE_OPEN, failures, 0, now),
                // OPEN is already open; any other word is invalid.
                _ => return,
            };

            match self.state.compare_exchange_weak(
                packed,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(_) => continue,
            }
        }
    }

    /// Record failure with windowed failure tracking.
    ///
    /// The current window's start lives in the packed timestamp (CLOSED state), so the
    /// reset-the-window-and-set-to-1 decision and the within-window increment are made from the
    /// **same** `packed` value and committed in the **same** CAS. That single-CAS atomicity is what
    /// fixes the old two-atomic race: when many failures land at a window boundary, they retry the
    /// CAS and each sees the latest count, so they linearize into a correct running total instead of
    /// each independently resetting to 1 and dropping the others (which could pin the count at 1 and
    /// keep a genuinely-broken provider's breaker stuck closed).
    fn record_failure_windowed(&self, threshold: u32, window_secs: u64, scope: Option<Permit>) {
        let threshold = threshold.min(MAX_FAILURE_THRESHOLD);
        let now = self.now_secs();

        loop {
            let packed = self.state.load(Ordering::Acquire);
            let (state, failures, successes, ts) = Self::unpack(packed);

            let new_packed = match state {
                STATE_CLOSED => {
                    // `failures == 0` ⇒ fresh window (cold start, or just after a success reset);
                    // `now - ts >= window` ⇒ the window this failure falls into has expired. Either
                    // way this failure starts a new window at count 1; otherwise it accrues into the
                    // current one. The anchor (window start) is carried in the timestamp field.
                    //
                    // `saturating_sub` for the same reason as `allow()`: a clock that steps
                    // backwards must read as "no time has passed", not as a wrapped ~2^32 elapsed
                    // that would expire the window on *every* failure and pin the count at 1 — a
                    // breaker that can never trip. Saturating keeps the anchor slightly in the
                    // future until the clock catches up, which only ever counts more failures into
                    // one window (fail-closed).
                    let window_expired = failures == 0 || now.saturating_sub(ts) >= window_secs;
                    let (new_failures, successes, anchor) = if window_expired {
                        (1, 0, now)
                    } else {
                        ((failures + 1).min(FAILURE_MASK), successes, ts)
                    };
                    // A failure *rate*: enough failures to mean something, and at least as many
                    // failures as successes since the window's first failure. Requiring both keeps
                    // a busy healthy provider's background errors from tripping it, while a
                    // provider failing every other request still opens.
                    if new_failures >= u64::from(threshold) && new_failures >= successes {
                        Self::open_word(now)
                    } else {
                        Self::pack(STATE_CLOSED, new_failures, successes, anchor)
                    }
                }
                STATE_HALF_OPEN if Self::stale(scope, failures) => return,
                STATE_HALF_OPEN => Self::pack(STATE_OPEN, failures, 0, now),
                // OPEN is already open; any other word is invalid.
                _ => return,
            };

            match self.state.compare_exchange_weak(
                packed,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(_) => continue,
            }
        }
    }

    /// Get the current circuit state for observability.
    pub fn state(&self) -> CircuitState {
        let packed = self.state.load(Ordering::Acquire);
        let (state, failures, permits, _) = Self::unpack(packed);

        match state {
            STATE_CLOSED => CircuitState::Closed {
                failure_count: failures as u32,
            },
            STATE_OPEN => CircuitState::Open,
            STATE_HALF_OPEN => CircuitState::HalfOpen {
                permits_remaining: permits as u32,
            },
            _ => CircuitState::Closed { failure_count: 0 },
        }
    }

    /// Give back a permit `allow()` handed out, without an outcome: the attempt never reached the
    /// provider as far as its health is concerned (the gateway itself failed). In HALF_OPEN that
    /// returns the probe permit so another request can probe; in any other state a permit costs
    /// nothing, so this is a no-op. So is a permit from another generation (reclaimed from a
    /// stalled probe, or handed out before the breaker half-opened): it no longer holds a probe
    /// permit, and returning one would admit a probe beside the live one (D256).
    pub fn release(&self, permit: Permit) {
        loop {
            let packed = self.state.load(Ordering::Acquire);
            let (state, failures, permits, ts) = Self::unpack(packed);
            if state != STATE_HALF_OPEN
                || u64::from(permit.0) != failures
                || permits >= u64::from(self.config.half_open_permits)
            {
                return;
            }
            let new_packed = Self::pack(STATE_HALF_OPEN, failures, permits + 1, ts);
            if self
                .state
                .compare_exchange_weak(packed, new_packed, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Whole seconds until this breaker would admit a request again, for the `Retry-After` on the
    /// `503` it causes. At least 1.
    pub fn retry_after_secs(&self) -> u64 {
        let (state, _, permits, ts) = Self::unpack(self.state.load(Ordering::Acquire));
        let reset = self.config.reset_timeout.as_secs();
        let waiting = match state {
            STATE_OPEN => reset.saturating_sub(self.now_secs().saturating_sub(ts)),
            STATE_HALF_OPEN if permits == 0 => {
                reset.saturating_sub(self.now_secs().saturating_sub(ts))
            }
            _ => 0,
        };
        waiting.max(1)
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset(&self) {
        let packed = Self::pack(STATE_CLOSED, 0, 0, self.now_secs());
        self.state.store(packed, Ordering::Release);
    }

    /// Force the circuit to a specific state (for testing/admin).
    #[cfg(test)]
    pub fn force_state(&self, new_state: CircuitState) {
        let now = self.now_secs();
        let packed = match new_state {
            CircuitState::Closed { failure_count } => {
                Self::pack(STATE_CLOSED, u64::from(failure_count), 0, now)
            }
            CircuitState::Open => Self::open_word(now),
            CircuitState::HalfOpen { permits_remaining } => {
                Self::pack(STATE_HALF_OPEN, 0, u64::from(permits_remaining), now)
            }
        };
        self.state.store(packed, Ordering::Release);
    }
}

/// What `allow()` hands out: the right to resolve one attempt. In HALF_OPEN it carries the probe
/// generation it was claimed in; from CLOSED it is no probe at all, so it can never resolve or
/// release one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permit(u16);

/// A permit claimed while CLOSED. Outside the 14-bit generation range, so it matches none.
const NOT_A_PROBE: u16 = u16::MAX;

impl Permit {
    /// A probe permit of `generation` (a 14-bit field value, so it always fits).
    #[inline]
    fn probe(generation: u64) -> Self {
        Self((generation & FAILURE_MASK) as u16)
    }
}

/// Error returned when the circuit is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitOpen;

impl std::fmt::Display for CircuitOpen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "circuit breaker is open")
    }
}

impl std::error::Error for CircuitOpen {}

/// Observable circuit state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow through normally.
    Closed {
        /// Number of failures since last success/reset.
        failure_count: u32,
    },
    /// Circuit is open, requests are rejected immediately.
    Open,
    /// Circuit is half-open, limited probe requests allowed.
    HalfOpen {
        /// Number of probe requests still allowed.
        permits_remaining: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // =========================================================================
    // Consecutive mode tests
    // =========================================================================

    #[test]
    fn test_initial_state_is_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
    }

    #[test]
    fn test_allow_when_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert!(cb.allow().is_ok());
    }

    #[test]
    fn test_consecutive_failures_increment() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::consecutive(5));

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 1 });

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 3 });
    }

    #[test]
    fn test_consecutive_success_resets_failures() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::consecutive(5));

        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 3 });

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
    }

    #[test]
    fn test_consecutive_opens_at_threshold() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::consecutive(3));

        cb.record_failure();
        cb.record_failure();
        assert!(matches!(cb.state(), CircuitState::Closed { .. }));

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_rejects_when_open() {
        let cb = CircuitBreaker::new(
            CircuitBreakerConfig::consecutive(1).reset_timeout(Duration::from_secs(3600)),
        );

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.allow().is_err());
    }

    #[test]
    fn test_half_open_after_timeout() {
        let cb = CircuitBreaker::new(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::from_millis(1))
                .half_open_permits(2),
        );

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(10));

        assert!(cb.allow().is_ok());
        assert!(matches!(cb.state(), CircuitState::HalfOpen { .. }));
    }

    #[test]
    fn test_half_open_permits_decrement() {
        let cb = CircuitBreaker::new(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::from_millis(1))
                .half_open_permits(3),
        );

        cb.record_failure();
        thread::sleep(Duration::from_millis(10));

        assert!(cb.allow().is_ok());
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 2
            }
        );

        assert!(cb.allow().is_ok());
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 1
            }
        );

        assert!(cb.allow().is_ok());
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 0
            }
        );

        assert!(cb.allow().is_err());
    }

    /// claim: R6
    #[test]
    fn test_half_open_success_closes() {
        let cb = CircuitBreaker::new(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::from_millis(1))
                .half_open_permits(3),
        );

        cb.record_failure();
        thread::sleep(Duration::from_millis(10));
        let _ = cb.allow();

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
    }

    #[test]
    fn test_half_open_failure_reopens() {
        let cb = CircuitBreaker::new(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::from_millis(1))
                .half_open_permits(3),
        );

        cb.record_failure();
        thread::sleep(Duration::from_millis(10));
        let _ = cb.allow();

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    // =========================================================================
    // Windowed mode tests
    // =========================================================================

    #[test]
    fn test_windowed_opens_at_threshold() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::windowed(3, Duration::from_secs(10)));

        cb.record_failure();
        cb.record_failure();
        assert!(matches!(cb.state(), CircuitState::Closed { .. }));

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_windowed_resets_after_window() {
        // Note: window uses second-level precision, so use 1 second window
        let cb = CircuitBreaker::new(CircuitBreakerConfig::windowed(3, Duration::from_secs(1)));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 2 });

        // Wait for window to expire (1 second + buffer)
        thread::sleep(Duration::from_millis(1100));

        // This failure starts a new window
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 1 });

        // Two more to hit threshold
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_windowed_success_does_not_erase_failures() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::windowed(3, Duration::from_secs(10)));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 2 });

        // A success is counted against the rate, not a reset.
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 2 });

        // 3 failures to 1 success: past the threshold and at least half the outcomes.
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    /// claim: R6
    #[test]
    fn test_windowed_half_failing_provider_opens() {
        // The brownout a success-resets rule missed: every other request fails.
        let cb = CircuitBreaker::new(CircuitBreakerConfig::windowed(4, Duration::from_secs(60)));
        for _ in 0..4 {
            cb.record_failure();
            cb.record_success();
        }
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_windowed_background_errors_on_a_busy_provider_stay_closed() {
        // Far more failures than the threshold, but a tenth of the traffic.
        let cb = CircuitBreaker::new(CircuitBreakerConfig::windowed(4, Duration::from_secs(60)));
        for _ in 0..50 {
            cb.record_failure();
            for _ in 0..9 {
                cb.record_success();
            }
        }
        assert!(matches!(cb.state(), CircuitState::Closed { .. }));
    }

    #[test]
    fn test_a_stalled_probe_permit_is_reclaimed_after_the_reset_timeout() {
        static NOW: AtomicU64 = AtomicU64::new(100);
        fn clock() -> u64 {
            NOW.load(Ordering::Relaxed)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(1, Duration::from_secs(60))
                .reset_timeout(Duration::from_secs(5))
                .half_open_permits(1),
            clock,
        );
        cb.record_failure();
        NOW.store(105, Ordering::Relaxed);
        let stalled = cb.allow().expect("the probe");
        assert!(cb.allow().is_err(), "the only permit is out");
        NOW.store(109, Ordering::Relaxed);
        assert!(cb.allow().is_err(), "not yet a whole reset timeout");
        assert_eq!(cb.retry_after_secs(), 1);
        NOW.store(110, Ordering::Relaxed);
        let live = cb.allow().expect("the stalled probe's permit is reclaimed");
        assert!(cb.allow().is_err());
        // The stalled probe no longer decides: its late answer, either way, is a no-op.
        cb.record_success_for(stalled);
        cb.record_failure_for(stalled);
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 0
            }
        );
        // The live probe does.
        cb.record_success_for(live);
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
    }

    /// D256: a permit claimed in an earlier half-open, or while CLOSED, is not this half-open's
    /// probe: it can neither release a probe permit nor decide the probe.
    /// claim: REL-6
    /// defect: D256
    #[test]
    fn a_permit_from_another_generation_neither_releases_nor_decides() {
        static NOW: AtomicU64 = AtomicU64::new(100);
        fn clock() -> u64 {
            NOW.load(Ordering::Relaxed)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::from_secs(5))
                .half_open_permits(1),
            clock,
        );
        let before_trip = cb.allow().expect("closed");
        cb.record_failure();
        NOW.store(105, Ordering::Relaxed);
        let first = cb.allow().expect("first half-open's probe");
        cb.record_failure_for(first);
        assert_eq!(cb.state(), CircuitState::Open);
        NOW.store(110, Ordering::Relaxed);
        let probe = cb.allow().expect("second half-open's probe");
        for stale in [before_trip, first] {
            cb.release(stale);
            cb.record_success_for(stale);
            cb.record_failure_for(stale);
        }
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 0
            }
        );
        cb.release(probe);
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 1
            }
        );
    }

    /// Concurrent stale releases racing a reclaim never leave more permits than configured, and
    /// never more than one probe out with `half_open_permits = 1`.
    /// claim: REL-6
    /// defect: D256
    #[test]
    fn concurrent_stale_releases_never_mint_permits() {
        static NOW: AtomicU64 = AtomicU64::new(100);
        fn clock() -> u64 {
            NOW.load(Ordering::Relaxed)
        }
        let cb = Arc::new(CircuitBreaker::with_clock(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::from_secs(5))
                .half_open_permits(1),
            clock,
        ));
        cb.record_failure();
        let mut stale = Vec::new();
        for t in 1..=32 {
            NOW.store(100 + 5 * t, Ordering::Relaxed);
            stale.push(cb.allow().expect("reclaimed every reset timeout"));
        }
        let live = stale.pop().unwrap();
        let handles: Vec<_> = stale
            .into_iter()
            .map(|p| {
                let cb = Arc::clone(&cb);
                thread::spawn(move || cb.release(p))
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(cb.allow().is_err(), "the live probe is the only one");
        cb.release(live);
        cb.release(live);
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 1
            }
        );
    }

    /// A probe permit given back by `release` (an attempt with no provider outcome) and claimed
    /// again is a fresh probe: the stalled-probe reclaim counts from that handout, so a second
    /// probe is not admitted beside it.
    #[test]
    fn a_released_probe_permit_claimed_again_is_not_reclaimable_at_once() {
        static NOW: AtomicU64 = AtomicU64::new(100);
        fn clock() -> u64 {
            NOW.load(Ordering::Relaxed)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(1, Duration::from_secs(60))
                .reset_timeout(Duration::from_secs(5))
                .half_open_permits(1),
            clock,
        );
        cb.record_failure();
        NOW.store(105, Ordering::Relaxed);
        let probe = cb.allow().expect("the probe");
        NOW.store(112, Ordering::Relaxed);
        cb.release(probe);
        assert!(cb.allow().is_ok(), "the returned permit");
        assert!(cb.allow().is_err(), "one probe at a time");
        NOW.store(117, Ordering::Relaxed);
        assert!(
            cb.allow().is_ok(),
            "reclaimed a reset timeout after its handout"
        );
    }

    /// D256: a stalled probe's permit is reclaimed for a new probe. When the stalled attempt later
    /// ends with no provider outcome, its `release` must not hand back a permit it no longer holds:
    /// with `half_open_permits = 1` that would admit a second probe beside the live one.
    /// claim: REL-6
    /// defect: D256
    #[test]
    fn a_reclaimed_probe_releasing_late_does_not_mint_a_second_probe() {
        static NOW: AtomicU64 = AtomicU64::new(100);
        fn clock() -> u64 {
            NOW.load(Ordering::Relaxed)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(1, Duration::from_secs(60))
                .reset_timeout(Duration::from_secs(5))
                .half_open_permits(1),
            clock,
        );
        cb.record_failure();
        NOW.store(105, Ordering::Relaxed);
        let stalled = cb.allow().expect("the probe");
        NOW.store(110, Ordering::Relaxed);
        let _live = cb.allow().expect("the stalled probe's permit is reclaimed");
        cb.release(stalled);
        assert!(cb.allow().is_err(), "one probe at a time");
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 0
            }
        );
    }

    /// D256: a reclaim moves the generation *forward*, past every generation already handed out.
    /// The reclaimed probe's permit must stay stale through the live probe failing, the breaker
    /// reopening and half-opening again: if the reclaim stepped the generation back, the next
    /// half-open's bump would land on the stalled permit's generation, and its late success would
    /// close a breaker whose current probe has not answered.
    /// claim: REL-6
    /// defect: D256
    #[test]
    fn a_reclaimed_probe_stays_stale_through_a_reopen() {
        static NOW: AtomicU64 = AtomicU64::new(100);
        fn clock() -> u64 {
            NOW.load(Ordering::Relaxed)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(1, Duration::from_secs(60))
                .reset_timeout(Duration::from_secs(5))
                .half_open_permits(1),
            clock,
        );
        cb.record_failure();
        NOW.store(105, Ordering::Relaxed);
        let stalled = cb.allow().expect("the probe");
        NOW.store(110, Ordering::Relaxed);
        let live = cb.allow().expect("the stalled probe's permit is reclaimed");
        cb.record_failure_for(live);
        assert_eq!(cb.state(), CircuitState::Open, "the live probe failed");
        NOW.store(115, Ordering::Relaxed);
        let _next = cb.allow().expect("half-open again");
        cb.record_success_for(stalled);
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 0
            },
            "the stalled probe's late success decides nothing"
        );
    }

    #[test]
    fn test_retry_after_counts_down_the_reset_timeout() {
        static NOW: AtomicU64 = AtomicU64::new(100);
        fn clock() -> u64 {
            NOW.load(Ordering::Relaxed)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(1, Duration::from_secs(60))
                .reset_timeout(Duration::from_secs(30)),
            clock,
        );
        assert_eq!(cb.retry_after_secs(), 1, "closed");
        cb.record_failure();
        assert_eq!(cb.retry_after_secs(), 30);
        NOW.store(120, Ordering::Relaxed);
        assert_eq!(cb.retry_after_secs(), 10);
    }

    #[test]
    fn test_release_returns_a_probe_permit() {
        let cb = CircuitBreaker::new(
            CircuitBreakerConfig::windowed(1, Duration::from_secs(60))
                .reset_timeout(Duration::from_millis(1))
                .half_open_permits(1),
        );
        cb.record_failure();
        thread::sleep(Duration::from_millis(10));
        let probe = cb.allow().expect("the probe");
        cb.release(probe);
        let again = cb.allow().expect("the released permit probes again");
        // Releasing more than was handed out cannot mint permits.
        cb.release(again);
        cb.release(again);
        assert_eq!(
            cb.state(),
            CircuitState::HalfOpen {
                permits_remaining: 1
            }
        );
    }

    #[test]
    fn test_windowed_half_open_recovery() {
        let cb = CircuitBreaker::new(
            CircuitBreakerConfig::windowed(2, Duration::from_secs(10))
                .reset_timeout(Duration::from_millis(1)),
        );

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(10));

        assert!(cb.allow().is_ok());
        assert!(matches!(cb.state(), CircuitState::HalfOpen { .. }));

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
    }

    // =========================================================================
    // Concurrency tests
    // =========================================================================

    #[test]
    fn test_concurrent_failures_open_exactly_once() {
        for _ in 0..100 {
            let cb = Arc::new(CircuitBreaker::new(
                CircuitBreakerConfig::consecutive(10).reset_timeout(Duration::from_secs(3600)),
            ));

            let handles: Vec<_> = (0..20)
                .map(|_| {
                    let cb = Arc::clone(&cb);
                    thread::spawn(move || {
                        cb.record_failure();
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(cb.state(), CircuitState::Open);
        }
    }

    #[test]
    fn test_concurrent_allow_in_half_open_respects_permits() {
        for _ in 0..100 {
            let cb = Arc::new(CircuitBreaker::new(
                CircuitBreakerConfig::consecutive(1)
                    .reset_timeout(Duration::from_millis(1))
                    .half_open_permits(5),
            ));

            cb.record_failure();
            thread::sleep(Duration::from_millis(10));

            let allowed = Arc::new(std::sync::atomic::AtomicU32::new(0));

            let handles: Vec<_> = (0..20)
                .map(|_| {
                    let cb = Arc::clone(&cb);
                    let allowed = Arc::clone(&allowed);
                    thread::spawn(move || {
                        if cb.allow().is_ok() {
                            allowed.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            let total_allowed = allowed.load(Ordering::SeqCst);
            assert!(
                total_allowed <= 5,
                "allowed {} requests but only 5 permits",
                total_allowed
            );
        }
    }

    /// A fixed clock pins every failure into the same window, so this isolates concurrent *counting*
    /// from window timing — the exact condition the old two-atomic design dropped increments under.
    fn fixed_clock() -> u64 {
        1000
    }

    #[test]
    fn test_concurrent_windowed_counts_every_failure() {
        // Regression for the window-boundary race: with all 20 failures inside one window and the
        // threshold well above 20, every concurrent failure must be counted — none silently dropped.
        // The old code computed the window-reset decision outside the state CAS, so losers of the
        // window_start race could reset a peer's increment back to 1, pinning the count and keeping a
        // genuinely-failing provider's breaker stuck closed. The single-CAS anchor makes that
        // impossible: the count must land at exactly 20.
        for _ in 0..200 {
            let cb = Arc::new(CircuitBreaker::with_clock(
                CircuitBreakerConfig::windowed(1000, Duration::from_secs(60)),
                fixed_clock,
            ));

            let handles: Vec<_> = (0..20)
                .map(|_| {
                    let cb = Arc::clone(&cb);
                    thread::spawn(move || cb.record_failure())
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(
                cb.state(),
                CircuitState::Closed { failure_count: 20 },
                "every concurrent in-window failure must be counted (no lost increments)"
            );
        }
    }

    #[test]
    fn test_concurrent_windowed_failures() {
        for _ in 0..50 {
            let cb = Arc::new(CircuitBreaker::new(CircuitBreakerConfig::windowed(
                10,
                Duration::from_secs(60),
            )));

            let handles: Vec<_> = (0..20)
                .map(|_| {
                    let cb = Arc::clone(&cb);
                    thread::spawn(move || {
                        cb.record_failure();
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(cb.state(), CircuitState::Open);
        }
    }

    // =========================================================================
    // Clock tests
    // =========================================================================

    #[test]
    fn test_system_clock_is_monotonic_not_wall_clock() {
        let first = CircuitBreaker::system_clock();
        thread::sleep(Duration::from_millis(5));
        let second = CircuitBreaker::system_clock();

        assert!(second >= first, "monotonic clock must never go backwards");
        // Seconds since process start, not since 1970 — a `SystemTime` reading would be ~1.7e9 and
        // climbing. This is what makes the packed 32-bit timestamp unwrappable in practice (it needs
        // 136 years of *uptime*) and what makes it immune to an NTP step.
        assert!(
            first < 60 * 60 * 24,
            "clock must count from a process-local base, got {first}"
        );
    }

    /// Hand-driven clock for the backward-step regressions below. Each test owns its own static so
    /// the two can run concurrently in the same test binary without stepping on each other.
    static BACKSTEP_OPEN_CLOCK: AtomicU64 = AtomicU64::new(0);
    fn backstep_open_clock() -> u64 {
        BACKSTEP_OPEN_CLOCK.load(Ordering::SeqCst)
    }

    #[test]
    fn test_backward_clock_step_does_not_force_half_open() {
        // Regression: `allow()` computed `now.wrapping_sub(timestamp)`. When the clock stepped
        // backwards — routine when NTP corrects a host that had drifted forward — `now < timestamp`
        // wrapped to a near-2^32 value that clears *any* reset timeout, so the very next request
        // forced the breaker half-open and put traffic back on a provider that is very likely still
        // broken. Worse, it did this to every open breaker in the process at once, right when the
        // fleet was least able to absorb it.
        BACKSTEP_OPEN_CLOCK.store(10_000, Ordering::SeqCst);
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::consecutive(1).reset_timeout(Duration::from_secs(30)),
            backstep_open_clock,
        );

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // NTP steps the clock one hour backwards.
        BACKSTEP_OPEN_CLOCK.store(10_000 - 3_600, Ordering::SeqCst);
        assert!(
            cb.allow().is_err(),
            "a backwards clock step must not admit traffic through an open circuit"
        );
        assert_eq!(
            cb.state(),
            CircuitState::Open,
            "a backwards clock step must not transition the circuit to half-open"
        );

        // ...and the fix must not strand it: once the clock genuinely advances past the reset
        // timeout, the normal half-open transition still happens.
        BACKSTEP_OPEN_CLOCK.store(10_000 + 30, Ordering::SeqCst);
        assert!(cb.allow().is_ok());
        assert!(matches!(cb.state(), CircuitState::HalfOpen { .. }));
    }

    static BACKSTEP_WINDOW_CLOCK: AtomicU64 = AtomicU64::new(0);
    fn backstep_window_clock() -> u64 {
        BACKSTEP_WINDOW_CLOCK.load(Ordering::SeqCst)
    }

    #[test]
    fn test_backward_clock_step_does_not_reset_failure_window() {
        // The same wrap in `record_failure_windowed`'s `window_expired` check made every failure
        // after a backwards step look like it landed in an expired window, so the count reset to 1
        // each time and the breaker could never reach its threshold — it simply stopped protecting
        // anything, silently, while the clock was behind.
        BACKSTEP_WINDOW_CLOCK.store(10_000, Ordering::SeqCst);
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(3, Duration::from_secs(60)),
            backstep_window_clock,
        );

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 1 });

        // NTP steps the clock one hour backwards mid-window.
        BACKSTEP_WINDOW_CLOCK.store(10_000 - 3_600, Ordering::SeqCst);

        cb.record_failure();
        assert_eq!(
            cb.state(),
            CircuitState::Closed { failure_count: 2 },
            "failures must keep accruing across a backwards clock step"
        );

        cb.record_failure();
        assert_eq!(
            cb.state(),
            CircuitState::Open,
            "the breaker must still trip at its threshold while the clock is behind"
        );
    }

    /// A threshold past the 14-bit count used to wrap the count to 0 on `pack`, so the breaker
    /// could never open. It is clamped to the largest count the word can hold instead.
    #[test]
    fn a_threshold_past_the_packed_count_still_opens() {
        for config in [
            CircuitBreakerConfig::consecutive(u32::MAX),
            CircuitBreakerConfig::windowed(MAX_FAILURE_THRESHOLD + 1, Duration::from_secs(60)),
        ] {
            let cb = CircuitBreaker::with_clock(
                config.reset_timeout(Duration::from_secs(3600)),
                fixed_clock,
            );
            for _ in 0..MAX_FAILURE_THRESHOLD - 1 {
                cb.record_failure();
            }
            assert!(matches!(cb.state(), CircuitState::Closed { .. }));
            cb.record_failure();
            assert_eq!(
                cb.state(),
                CircuitState::Open,
                "must open, not wrap back to 0"
            );
        }
    }

    /// The windowed rate rule is "failures ≥ threshold and failures ≥ successes". A busy window can
    /// record more successes than the failure count can hold; failures must still be able to catch
    /// up, so the two counts saturate at the same value and a majority of failures always opens.
    /// claim: REL-6
    /// defect: D97
    #[test]
    fn a_failure_majority_opens_a_window_with_more_successes_than_the_failure_count_holds() {
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(3, Duration::from_secs(60))
                .reset_timeout(Duration::from_secs(3600)),
            fixed_clock,
        );
        cb.record_failure();
        for _ in 0..20_000 {
            cb.record_success();
        }
        for _ in 0..20_001 {
            cb.record_failure();
            if cb.state() == CircuitState::Open {
                return;
            }
        }
        panic!(
            "20001 failures against 20000 successes in one window never opened: {:?}",
            cb.state()
        );
    }

    // =========================================================================
    // Pack/unpack tests
    // =========================================================================

    #[test]
    fn test_pack_unpack_roundtrip() {
        let test_cases = [
            (STATE_CLOSED, 0, 0, 0),
            (STATE_OPEN, 0, 0, 12345),
            (STATE_HALF_OPEN, 100, 50, 999999),
            (STATE_CLOSED, FAILURE_MASK, PERMIT_MASK, TIMESTAMP_MASK),
        ];

        for (state, failures, permits, timestamp) in test_cases {
            let packed = CircuitBreaker::pack(state, failures, permits, timestamp);
            let (s, f, p, t) = CircuitBreaker::unpack(packed);
            assert_eq!(s, state, "state mismatch");
            assert_eq!(f, failures, "failures mismatch");
            assert_eq!(p, permits, "permits mismatch");
            assert_eq!(t, timestamp, "timestamp mismatch");
        }
    }

    // =========================================================================
    // Builder API tests
    // =========================================================================

    #[test]
    fn test_builder_consecutive() {
        let config = CircuitBreakerConfig::consecutive(5)
            .reset_timeout(Duration::from_secs(60))
            .half_open_permits(10);

        assert_eq!(
            config.failure_policy,
            FailurePolicy::Consecutive { threshold: 5 }
        );
        assert_eq!(config.reset_timeout, Duration::from_secs(60));
        assert_eq!(config.half_open_permits, 10);
    }

    #[test]
    fn test_builder_windowed() {
        let config = CircuitBreakerConfig::windowed(3, Duration::from_secs(10))
            .reset_timeout(Duration::from_secs(30))
            .half_open_permits(5);

        assert_eq!(
            config.failure_policy,
            FailurePolicy::Windowed {
                threshold: 3,
                window: Duration::from_secs(10)
            }
        );
        assert_eq!(config.reset_timeout, Duration::from_secs(30));
        assert_eq!(config.half_open_permits, 5);
    }
}

/// Behaviors a mutation-testing pass found no test constraining.
#[cfg(test)]
mod mutation_gaps {
    use super::*;

    /// `allow()` on a closed breaker is a read: it must not wipe the failures counted so far, or a
    /// provider failing every request between admissions would never trip.
    /// claim: R6
    #[test]
    fn allow_on_a_closed_breaker_keeps_its_failure_count() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::consecutive(3));
        cb.record_failure();
        cb.record_failure();
        assert!(cb.allow().is_ok());
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 2 });
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    /// A windowed breaker's success after its window has expired wipes the slate, at the boundary
    /// itself too; inside the window it only counts toward the rate.
    /// claim: R6
    #[test]
    fn windowed_success_resets_once_the_window_has_expired() {
        static NOW: AtomicU64 = AtomicU64::new(1_000);
        fn clock() -> u64 {
            NOW.load(Ordering::SeqCst)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::windowed(5, Duration::from_secs(10)),
            clock,
        );
        cb.record_failure();
        cb.record_failure();
        NOW.store(1_009, Ordering::SeqCst);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 2 });
        // Exactly one window after the first failure: expired.
        NOW.store(1_010, Ordering::SeqCst);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
    }

    /// A failed probe reopens a windowed breaker, as it does a consecutive one.
    /// claim: R6, REL-6
    #[test]
    fn windowed_half_open_failure_reopens() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::windowed(3, Duration::from_secs(10)));
        cb.force_state(CircuitState::HalfOpen {
            permits_remaining: 1,
        });
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    /// `Retry-After` while half-open: 1 when a probe permit is free (the next request is admitted),
    /// and the rest of the reset timeout when every permit is out.
    /// claim: R6, REL-19
    #[test]
    fn half_open_retry_after_depends_on_free_permits() {
        static NOW: AtomicU64 = AtomicU64::new(500);
        fn clock() -> u64 {
            NOW.load(Ordering::SeqCst)
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::from_secs(30))
                .half_open_permits(1),
            clock,
        );
        cb.force_state(CircuitState::HalfOpen {
            permits_remaining: 1,
        });
        NOW.store(510, Ordering::SeqCst);
        assert_eq!(cb.retry_after_secs(), 1, "a free permit admits now");
        cb.force_state(CircuitState::HalfOpen {
            permits_remaining: 0,
        });
        NOW.store(510, Ordering::SeqCst);
        assert_eq!(
            cb.retry_after_secs(),
            30,
            "no permit: wait out the reset timeout"
        );
    }

    /// `reset()` closes an open breaker and clears its count.
    /// claim: R6
    #[test]
    fn reset_closes_an_open_breaker() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::consecutive(1));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
        assert!(cb.allow().is_ok());
    }

    /// A probe generation wraps through 0 (the 14-bit field), and a probe of generation 0 still
    /// closes the breaker on success: the no-op fast path is for a clean CLOSED word only.
    /// claim: R6, REL-6
    #[test]
    fn a_generation_zero_probe_success_closes_the_breaker() {
        // Opening at t = 16383 seeds the generation at the field's max, so half-opening wraps it
        // to 0.
        fn clock() -> u64 {
            FAILURE_MASK
        }
        let cb = CircuitBreaker::with_clock(
            CircuitBreakerConfig::consecutive(1)
                .reset_timeout(Duration::ZERO)
                .half_open_permits(1),
            clock,
        );
        cb.record_failure();
        let probe = cb.allow().expect("half-open probe");
        assert_eq!(probe, Permit::probe(0), "the generation wrapped to 0");
        cb.record_success_for(probe);
        assert_eq!(cb.state(), CircuitState::Closed { failure_count: 0 });
    }

    /// The breaker's `Debug` prints its config, not an empty struct.
    /// claim: R6
    #[test]
    fn debug_names_the_breaker_and_its_config() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::consecutive(7));
        let s = format!("{cb:?}");
        assert!(
            s.starts_with("CircuitBreaker {") && s.contains("threshold: 7"),
            "{s}"
        );
    }
}
