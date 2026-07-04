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
pub enum DevicePollStep<T> {
    Pending,
    SlowDown,
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
                interval = (interval + SLOW_DOWN_INCREMENT).max(MIN_INTERVAL);
            }
        }
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
