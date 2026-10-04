//! The request deadline's clock (`request_max_secs`).
//!
//! A deadline is milliseconds since a process-wide base [`Instant`]. Two checks read it, and
//! neither takes a timer:
//!
//! - **Where the upstream read timeout is armed** (`upstream_peer`, once per attempt), the read
//!   timeout is capped at the time left ([`remaining`], one exact clock read). Pingora re-arms it on
//!   every read, and every read starts after the attempt did, so a capped timeout can only fire at
//!   or past the deadline. That is what ends a silent stream, one HTTP/2 PINGs keep alive.
//! - **Per body chunk** (`response_body_filter`, `request_body_filter`), [`expired`] compares
//!   against [`now_ms`], a coarse clock: one relaxed atomic load, no syscall. A ticker thread
//!   advances it once a second, so a stream that keeps moving is cut within about a second of its
//!   deadline. That is what ends a stream kept alive by its own bytes (SSE keepalive comments).
//!
//! The ticker starts with the first [`start`] (the proxy's construction when the ceiling is on);
//! with it off no request carries a deadline and nothing reads the clock.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// No deadline: a request with this never expires (`request_max_secs = 0`).
pub const NONE: u64 = u64::MAX;

/// How often the coarse clock advances: the lateness bound of a per-chunk cut.
const TICK: Duration = Duration::from_secs(1);

static BASE: OnceLock<Instant> = OnceLock::new();
static NOW_MS: AtomicU64 = AtomicU64::new(0);
static TICKER: OnceLock<()> = OnceLock::new();

fn base() -> Instant {
    *BASE.get_or_init(Instant::now)
}

fn ms_since_base(at: Instant) -> u64 {
    u64::try_from(at.saturating_duration_since(base()).as_millis()).unwrap_or(NONE - 1)
}

/// Start the coarse clock's ticker. Idempotent: one thread per process, however many proxies.
pub fn start() {
    TICKER.get_or_init(|| {
        NOW_MS.store(ms_since_base(Instant::now()), Ordering::Relaxed);
        let spawned = std::thread::Builder::new()
            .name("deadline-clock".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(TICK);
                    NOW_MS.store(ms_since_base(Instant::now()), Ordering::Relaxed);
                }
            });
        if let Err(e) = spawned {
            // Without the ticker the coarse clock stands still: per-chunk cuts never fire, and
            // the capped read timeout still ends a silent stream at its deadline.
            tracing::error!(error = %e, "could not start the request deadline clock");
        }
    });
}

/// The deadline of a request that started at `start` and may live `max_secs`; [`NONE`] for 0.
pub fn at(start: Instant, max_secs: u64) -> u64 {
    if max_secs == 0 {
        return NONE;
    }
    ms_since_base(start).saturating_add(max_secs.saturating_mul(1000))
}

/// The coarse clock, in the deadline's units. Lags real time by up to [`TICK`].
#[inline]
pub fn now_ms() -> u64 {
    NOW_MS.load(Ordering::Relaxed)
}

/// Whether `deadline` has passed by the coarse clock. Never for [`NONE`].
#[inline]
pub fn expired(deadline: u64) -> bool {
    now_ms() >= deadline
}

/// The exact time left before `deadline`, zero once it has passed; `None` for [`NONE`].
pub fn remaining(deadline: u64) -> Option<Duration> {
    (deadline != NONE)
        .then(|| Duration::from_millis(deadline.saturating_sub(ms_since_base(Instant::now()))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_max_is_no_deadline() {
        assert_eq!(at(Instant::now(), 0), NONE);
        assert_eq!(remaining(NONE), None);
        assert!(!expired(NONE));
    }

    #[test]
    fn remaining_counts_down_to_zero() {
        let start = Instant::now();
        let d = at(start, 2);
        let left = remaining(d).unwrap();
        assert!(left <= Duration::from_secs(2), "{left:?}");
        assert!(left > Duration::from_millis(1500), "{left:?}");
        assert_eq!(
            remaining(0),
            Some(Duration::ZERO),
            "a deadline already passed"
        );
    }

    #[test]
    fn the_coarse_clock_ticks_and_expires_deadlines() {
        start();
        let d = at(Instant::now(), 1);
        assert!(!expired(d));
        std::thread::sleep(Duration::from_millis(2300));
        assert!(expired(d), "now {} deadline {d}", now_ms());
    }
}
