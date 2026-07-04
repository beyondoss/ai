//! Shared device-code polling loop. GitHub Copilot's flow and OpenAI Codex's own (RFC-8628-shaped
//! but not RFC-8628-compliant — see `openai_codex.rs`) device flow differ only in their per-attempt
//! request/response shape; the interval/backoff/deadline/cancellation semantics are identical enough
//! to share one generic loop, ported from pi's `pollOAuthDeviceCodeFlow`.

use std::future::Future;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

use super::error::{OAuthError, Result};

/// A minimum enforced poll interval, regardless of what a server claims — guards against a
/// misbehaving/malicious server naming a near-zero interval and turning this into a tight loop.
const MIN_INTERVAL: Duration = Duration::from_millis(1000);
/// RFC 8628 §3.5: a `slow_down` response must increase the poll interval by at least 5 seconds, and
/// the increase persists for every subsequent poll, not just the next one.
const SLOW_DOWN_INCREMENT: Duration = Duration::from_millis(5000);

const TIMEOUT_MESSAGE: &str = "Device flow timed out";
/// A `slow_down` this close to a timeout is the signature of clock drift inside a WSL/VM guest —
/// worth telling the user directly (a real UX nicety carried over from pi) rather than leaving them
/// to guess why a normally-quick device flow timed out.
const SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down \
     responses. This is often caused by clock drift in WSL or VM environments. Please sync or \
     restart the VM clock and try again.";

/// One poll attempt's outcome.
#[derive(Debug)]
pub enum DevicePollStep<T> {
    Pending,
    /// The server asked the client to slow down, without naming its own required interval — apply
    /// RFC 8628 §3.5's plain fallback (increase the local interval by at least
    /// [`SLOW_DOWN_INCREMENT`]).
    SlowDown,
    /// Like [`SlowDown`](Self::SlowDown), but the server named its own required minimum poll
    /// interval in the response body (GitHub Copilot's `slow_down` responses carry this in an
    /// `interval` field) — prefer it over the local +5s increment. A client that only tracks its own
    /// locally-incremented interval can fall behind the server's actual requirement under clock
    /// drift (WSL/VM guests — see microsoft/WSL#10006), and keep hitting the rate limit on every
    /// subsequent poll forever. Mirrors pi's `pollOAuthDeviceCodeFlow`'s `intervalSeconds` handling.
    SlowDownWithInterval(Duration),
    Complete(T),
}

/// Poll `poll_once` on an interval until it reports [`DevicePollStep::Complete`], the deadline
/// (`expires_in` from now, or never if `None`) passes, or `cancel` fires.
///
/// `wait_before_first_poll`: GitHub Copilot's device flow sleeps one interval *before* its first
/// poll; OpenAI Codex's polls immediately, then waits between subsequent attempts — replicate
/// whichever the caller's spec says, there is no shared default.
pub async fn poll_device_code<T, F, Fut>(
    mut interval: Duration,
    expires_in: Option<Duration>,
    wait_before_first_poll: bool,
    cancel: &CancellationToken,
    mut poll_once: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<DevicePollStep<T>>>,
{
    interval = interval.max(MIN_INTERVAL);
    let deadline = expires_in.map(|d| Instant::now() + d);
    let mut saw_slow_down = false;
    let mut first = true;

    loop {
        if let Some(deadline) = deadline {
            if Instant::now() >= deadline {
                let message = if saw_slow_down {
                    SLOW_DOWN_TIMEOUT_MESSAGE
                } else {
                    TIMEOUT_MESSAGE
                };
                return Err(OAuthError::DeviceFlowTimedOut(message.to_string()));
            }
        }

        if !first || wait_before_first_poll {
            tokio::select! {
                _ = cancel.cancelled() => return Err(OAuthError::LoginCancelled),
                _ = tokio::time::sleep(interval) => {}
            }
        }
        first = false;

        if cancel.is_cancelled() {
            return Err(OAuthError::LoginCancelled);
        }

        match poll_once().await? {
            DevicePollStep::Complete(value) => return Ok(value),
            DevicePollStep::Pending => {}
            DevicePollStep::SlowDown => {
                saw_slow_down = true;
                interval = next_slow_down_interval(interval, None);
            }
            DevicePollStep::SlowDownWithInterval(server_interval) => {
                saw_slow_down = true;
                interval = next_slow_down_interval(interval, Some(server_interval));
            }
        }
    }
}

/// The next poll interval after a `slow_down`, given the interval in effect just before it and
/// whichever server-provided interval (if any) came back with it. Split out from the match arms
/// above purely so the choice between "trust the server" and "apply the local +5s increment" is
/// unit-testable without driving the whole async loop through real time.
///
/// Mirrors pi's `pollOAuthDeviceCodeFlow`: `server_interval` wins outright when present (still
/// floored at [`MIN_INTERVAL`]) — trusting only a locally-tracked increment risks the client falling
/// behind the server's actual requirement under clock drift (WSL/VM guests) and re-triggering
/// `slow_down` forever. Only when the server didn't name one does RFC 8628 §3.5's plain increment
/// apply.
fn next_slow_down_interval(current: Duration, server_interval: Option<Duration>) -> Duration {
    match server_interval {
        Some(server_interval) => server_interval.max(MIN_INTERVAL),
        None => (current + SLOW_DOWN_INCREMENT).max(MIN_INTERVAL),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn completes_on_first_successful_poll_without_waiting_when_not_told_to() {
        let started = Instant::now();
        let result = poll_device_code(
            Duration::from_millis(50),
            Some(Duration::from_secs(5)),
            false, // Codex-style: poll immediately
            &CancellationToken::new(),
            || async { Ok(DevicePollStep::Complete(42)) },
        )
        .await
        .unwrap();
        assert_eq!(result, 42);
        assert!(started.elapsed() < Duration::from_millis(40), "should not have waited at all");
    }

    #[tokio::test]
    async fn waits_before_the_first_poll_when_told_to() {
        let calls = Arc::new(AtomicUsize::new(0));
        let started = Instant::now();
        let calls_clone = calls.clone();
        poll_device_code(
            Duration::from_millis(30),
            Some(Duration::from_secs(5)),
            true, // Copilot-style: wait, then poll
            &CancellationToken::new(),
            move || {
                let calls = calls_clone.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(DevicePollStep::Complete(()))
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(
            started.elapsed() >= Duration::from_millis(30),
            "must wait one interval before the very first poll"
        );
    }

    #[tokio::test]
    async fn slow_down_increases_the_interval_and_persists() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let result = poll_device_code(
            Duration::from_millis(10),
            Some(Duration::from_secs(30)),
            false,
            &CancellationToken::new(),
            move || {
                let calls = calls_clone.clone();
                async move {
                    let n = calls.fetch_add(1, Ordering::SeqCst);
                    match n {
                        0 => Ok(DevicePollStep::Pending),
                        1 => Ok(DevicePollStep::SlowDown),
                        _ => Ok(DevicePollStep::Complete(n)),
                    }
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(result, 2); // completed on the 3rd call (index 2)
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn slow_down_with_interval_is_recognized_by_the_full_poll_loop_and_clamped_to_the_minimum() {
        // A malicious/misbehaving server naming an absurdly small interval must still be floored at
        // MIN_INTERVAL, exactly like the no-interval fallback path already is.
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let started = Instant::now();
        let result = poll_device_code(
            Duration::from_millis(10),
            Some(Duration::from_secs(5)),
            false,
            &CancellationToken::new(),
            move || {
                let calls = calls_clone.clone();
                async move {
                    let n = calls.fetch_add(1, Ordering::SeqCst);
                    match n {
                        0 => Ok(DevicePollStep::SlowDownWithInterval(Duration::from_millis(1))),
                        _ => Ok(DevicePollStep::Complete(n)),
                    }
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(result, 1);
        assert!(
            started.elapsed() >= MIN_INTERVAL,
            "a tiny server-provided interval must still be floored at MIN_INTERVAL"
        );
    }

    #[test]
    fn next_slow_down_interval_prefers_the_servers_larger_interval_over_the_local_increment() {
        let current = Duration::from_millis(1000);
        // The local +5s increment would only produce 6s; the server's own reported interval is
        // larger and must win outright — not get compared against or blended with the local value.
        let local_would_produce = current + SLOW_DOWN_INCREMENT;
        let server_interval = local_would_produce + Duration::from_secs(2);
        let next = next_slow_down_interval(current, Some(server_interval));
        assert_eq!(next, server_interval);
        assert!(
            next > local_would_produce,
            "the larger server-provided interval must win over the local increment"
        );
    }

    #[test]
    fn next_slow_down_interval_falls_back_to_the_local_increment_when_the_server_gives_none() {
        let current = Duration::from_millis(1000);
        assert_eq!(
            next_slow_down_interval(current, None),
            current + SLOW_DOWN_INCREMENT
        );
    }

    #[test]
    fn next_slow_down_interval_still_floors_a_tiny_server_interval_at_the_minimum() {
        assert_eq!(
            next_slow_down_interval(Duration::from_secs(1), Some(Duration::from_millis(1))),
            MIN_INTERVAL
        );
    }

    #[tokio::test]
    async fn times_out_when_the_deadline_passes_without_completion() {
        let err = poll_device_code(
            Duration::from_millis(10),
            Some(Duration::from_millis(25)),
            false,
            &CancellationToken::new(),
            || async { Ok(DevicePollStep::<()>::Pending) },
        )
        .await
        .unwrap_err();
        match err {
            OAuthError::DeviceFlowTimedOut(msg) => assert!(!msg.contains("slow_down")),
            other => panic!("expected DeviceFlowTimedOut, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn timeout_message_mentions_clock_drift_after_a_slow_down() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let err = poll_device_code(
            Duration::from_millis(10),
            Some(Duration::from_millis(35)),
            false,
            &CancellationToken::new(),
            move || {
                let calls = calls_clone.clone();
                async move {
                    let n = calls.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        Ok(DevicePollStep::<()>::SlowDown)
                    } else {
                        Ok(DevicePollStep::<()>::Pending)
                    }
                }
            },
        )
        .await
        .unwrap_err();
        match err {
            OAuthError::DeviceFlowTimedOut(msg) => {
                assert!(msg.contains("clock drift"), "got: {msg}")
            }
            other => panic!("expected DeviceFlowTimedOut, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancellation_stops_the_poll_immediately() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = poll_device_code(
            Duration::from_millis(10),
            None,
            true,
            &cancel,
            || async { Ok(DevicePollStep::<()>::Pending) },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, OAuthError::LoginCancelled));
    }

    #[tokio::test]
    async fn a_pending_response_never_returned_by_poll_once_is_treated_as_an_error_passthrough() {
        let err = poll_device_code(
            Duration::from_millis(5),
            Some(Duration::from_secs(5)),
            false,
            &CancellationToken::new(),
            || async { Err::<DevicePollStep<()>, _>(OAuthError::DeviceFlowFailed("boom".to_string())) },
        )
        .await
        .unwrap_err();
        assert!(matches!(err, OAuthError::DeviceFlowFailed(_)));
    }
}
