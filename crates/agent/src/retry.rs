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

/// Exponential backoff for a whole-run retry: `RUN_RETRY_BASE_BACKOFF · 2^(attempt-1)`, capped at
/// `RUN_RETRY_MAX_BACKOFF`. `attempt` is 1-based. Coarser than `agent_core`'s own `mid_stream_backoff`
/// (250ms base/5s cap) — this gates a whole extra model turn's worth of retry, not a resumed stream, so
/// pi's slower 2/4/8s cadence is the better fit.
pub fn backoff(attempt: u32) -> std::time::Duration {
    RUN_RETRY_BASE_BACKOFF
        .saturating_mul(1u32 << attempt.saturating_sub(1).min(16))
        .min(RUN_RETRY_MAX_BACKOFF)
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
const WHOLE_RUN_RETRYABLE_STATUS_DIGITS: &[&str] = &["429", "500", "502", "503", "504"];

/// Whether a whole run that ended in `Err` is worth automatically re-invoking from scratch — a superset
/// of `agent_core::agent::is_retryable_mid_stream` (every mid-stream-worth-retrying error is also worth
/// a whole-run retry) plus [`WHOLE_RUN_RETRYABLE_STATUS_DIGITS`], appropriate only at this outer layer.
/// Excludes a context-overflow rejection either way (`is_context_overflow`, `pub` for exactly this) —
/// that's compact-and-retry's signal, not this one's; retrying it blindly here would just fail
/// identically again without compacting first.
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
    WHOLE_RUN_RETRYABLE_STATUS_DIGITS
        .iter()
        .any(|d| msg.contains(d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::Error;

    #[test]
    fn backoff_doubles_then_caps() {
        assert_eq!(backoff(1), std::time::Duration::from_secs(2));
        assert_eq!(backoff(2), std::time::Duration::from_secs(4));
        assert_eq!(backoff(3), std::time::Duration::from_secs(8));
        assert_eq!(backoff(10), RUN_RETRY_MAX_BACKOFF);
    }

    #[test]
    fn whole_run_retries_a_plain_http_status_digit_mid_stream_never_would() {
        let e = Error::Transport("gateway returned 503: <html>Service Unavailable</html>".into());
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
}
