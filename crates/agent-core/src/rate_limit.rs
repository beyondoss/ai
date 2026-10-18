//! Rate-limit retry: which failures are the provider saying "slow down", how long it asked us to
//! wait, and how patiently a turn waits it out.
//!
//! A rate limit is not a fault to retry a fixed number of times. A tokens-per-minute window shared
//! with other runs can stay full for many seconds, and three retries half a second apart (what every
//! transient error used to get) give up long before it drains. So [`crate::Agent`]'s per-turn retry
//! gives this class a **time budget** instead of a count: it keeps re-sending, waiting the larger of
//! the server's own hint and an exponential backoff, until the budget would be overrun, and then fails
//! with [`BUDGET_EXHAUSTED`] in the message so no outer layer retries it again.
//!
//! The hint can arrive two ways, and both end up in the error text: an HTTP 429's headers (the client
//! writes the wait it derived into its final error as `retry-after-ms: N`), or an error event *inside*
//! a 200 stream, where OpenAI Responses carries the headers in the event body (`"headers":
//! {"retry-after-ms":"243",…}`) and the prose says "Please try again in 243ms".

use std::time::Duration;

use crate::Error;

/// Default time budget for one turn's rate-limit retries. Claude Code retries 10 times by default
/// (500ms base, doubling) — about 2.5 minutes of waiting — and Codex's `stream_max_retries`/
/// `request_max_retries` of 5/4 at a 200ms base give up in seconds. An unattended agent run losing its
/// progress to a minute-long TPM window is the worse failure, so this sits above Claude Code's.
pub const DEFAULT_BUDGET: Duration = Duration::from_secs(300);
/// First rate-limit wait when the server gives no longer hint; doubles per attempt. Matches the
/// Anthropic SDK / Claude Code's 500ms initial retry delay.
pub const DEFAULT_BASE_BACKOFF: Duration = Duration::from_millis(500);
/// Ceiling on a single wait, hint or not — the same 60s as `client::MAX_BACKOFF`, so a server asking
/// for minutes is re-asked once a minute rather than trusted to park the run that long.
pub const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(60);

/// Marker in the error a turn fails with once its rate-limit budget is spent. Classifiers check it so
/// the whole-run retry layer above doesn't start the same wait all over again.
pub const BUDGET_EXHAUSTED: &str = "rate-limit retry budget exhausted";

/// How patiently a turn waits out rate limits — see the module doc. `budget: 0` turns the patient
/// retry off: the first rate limit fails the turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitRetry {
    pub budget: Duration,
    pub base_backoff: Duration,
    pub max_wait: Duration,
}

impl Default for RateLimitRetry {
    fn default() -> Self {
        Self {
            budget: DEFAULT_BUDGET,
            base_backoff: DEFAULT_BASE_BACKOFF,
            max_wait: DEFAULT_MAX_WAIT,
        }
    }
}

impl RateLimitRetry {
    /// The wait before rate-limit retry `attempt` (1-based): the larger of the server's `hint` and
    /// `base_backoff · 2^(attempt-1)` (± jitter), capped at `max_wait`. The hint is a floor, not the
    /// answer: under contention ("try again in 243ms" while seven other runs drain the same window)
    /// re-sending on the hint alone would just be rejected again.
    pub fn wait(&self, attempt: u32, hint: Option<Duration>) -> Duration {
        let exp = self
            .base_backoff
            .saturating_mul(1u32 << attempt.saturating_sub(1).min(16));
        let exp = crate::client::jitter(exp, attempt).min(self.max_wait);
        hint.map_or(exp, |h| h.min(self.max_wait).max(exp))
    }
}

/// Why the provider asked us to wait. Both kinds get the same time budget and backoff; only the
/// label differs, so a person reading "rate limited" is not sent off to check their quota when the
/// provider is simply busy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Throttle {
    /// A 429 or an in-stream rate-limit error: this caller (or its shared window) is over a limit.
    RateLimited,
    /// A 529 or `overloaded_error`: the provider is out of capacity, whoever is asking.
    Overloaded,
}

/// Whether `e` is the provider (or the gateway) saying "slow down" ([`classify`] is `Some`).
pub fn is_rate_limited(e: &Error) -> bool {
    classify(e).is_some()
}

/// Which kind of "slow down" `e` is, if it is one: a 429 or an in-stream rate-limit error from either
/// dialect (Anthropic `rate_limit_error`, OpenAI `rate_limit_exceeded`) is [`Throttle::RateLimited`];
/// an overload (Anthropic's 529 `overloaded_error`, a bare 529) is [`Throttle::Overloaded`] -- the
/// same 429-and-529 pair Claude Code's retry watchdog waits out. Never a quota/billing 429 (retrying
/// can't fix an empty account), a context overflow, a request too large for the limit itself
/// (OpenAI's TPM "Request too large ... must be reduced", which no wait admits), or an error that
/// already spent its budget.
pub fn classify(e: &Error) -> Option<Throttle> {
    let Error::Transport(msg) = e else {
        return None;
    };
    if msg.contains(BUDGET_EXHAUSTED)
        || crate::client::is_quota_exhausted(msg)
        || crate::agent::is_context_overflow(e)
    {
        return None;
    }
    let m = msg.to_ascii_lowercase();
    // A bare 529 (no `overloaded_error` body) is still Anthropic's overload status.
    if m.contains(" returned 529") {
        return Some(Throttle::Overloaded);
    }
    if ["request too large", "must be reduced", "will not succeed"]
        .iter()
        .any(|p| m.contains(p))
    {
        return None;
    }
    if [
        "rate_limit",
        "rate limit",
        "rate-limit",
        "too many requests",
    ]
    .iter()
    .any(|p| m.contains(p))
    {
        return Some(Throttle::RateLimited);
    }
    m.contains("overloaded").then_some(Throttle::Overloaded)
}

/// The wait the server asked for, read out of an error message, in order of precision:
/// `retry-after-ms`, `retry-after` (seconds), OpenAI's prose "try again in 1.5s", and — only when
/// none of those is present — the longest `x-ratelimit-reset-*`. The reset headers say when the
/// bucket is *full* again (59s in the TPM error that motivated this), not when this request fits
/// (243ms in the same error), so they are the weakest hint and never override a real one.
pub fn retry_hint(msg: &str) -> Option<Duration> {
    let m = msg.to_ascii_lowercase();
    if let Some(ms) = value_after(&m, "retry-after-ms").and_then(parse_number) {
        return Some(secs(ms / 1000.0));
    }
    let mut from = 0;
    while let Some(i) = m[from..].find("retry-after") {
        let rest = &m[from + i + "retry-after".len()..];
        from += i + "retry-after".len();
        if rest.starts_with('-') {
            continue;
        }
        if let Some(s) = parse_number(skip_separators(rest)) {
            return Some(secs(s));
        }
    }
    if let Some(d) = m
        .find("try again in")
        .and_then(|i| parse_duration(m[i + "try again in".len()..].trim_start()))
    {
        return Some(d);
    }
    ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"]
        .iter()
        .filter_map(|k| value_after(&m, k).and_then(parse_duration))
        .max()
}

/// The text right after `key` and its separator (`": "`, `": \"`, `=`), for a key that appears once.
fn value_after<'a>(m: &'a str, key: &str) -> Option<&'a str> {
    let i = m.find(key)?;
    Some(skip_separators(&m[i + key.len()..]))
}

fn skip_separators(s: &str) -> &str {
    s.trim_start_matches(['"', '\\', '\'', ':', '=', ' '])
}

/// A leading non-negative decimal number, e.g. `243` or `1.5`.
fn parse_number(s: &str) -> Option<f64> {
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(s.len());
    s[..end].parse::<f64>().ok().filter(|n| n.is_finite())
}

/// A Go-style duration as OpenAI writes them (`243ms`, `59.032s`, `6m0s`, `1h2m3s`), also accepting
/// spelled-out units with a space (`20 seconds`). Needs at least one number-with-unit.
fn parse_duration(mut s: &str) -> Option<Duration> {
    let mut total = 0.0f64;
    let mut any = false;
    while let Some(n) = parse_number(s) {
        let after_num = s.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        let after_num = after_num.trim_start_matches(' ');
        let unit_len = after_num
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(after_num.len());
        let scale = match &after_num[..unit_len] {
            "ms" | "msec" | "milliseconds" | "millisecond" => 0.001,
            "s" | "sec" | "secs" | "second" | "seconds" => 1.0,
            "m" | "min" | "mins" | "minute" | "minutes" => 60.0,
            "h" | "hr" | "hour" | "hours" => 3600.0,
            _ => break,
        };
        total += n * scale;
        any = true;
        s = &after_num[unit_len..];
    }
    any.then(|| secs(total))
}

/// Seconds to a `Duration`, clamped to a day so an absurd hint can't panic `from_secs_f64`.
fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s.clamp(0.0, 86_400.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact in-stream error from the eval run that lost its progress (OpenAI Responses, TPM).
    const OPENAI_IN_STREAM: &str = r#"provider stream error: {"type":"error","error":{"type":"tokens","code":"rate_limit_exceeded","headers":{"retry-after":"1","retry-after-ms":"243","x-ratelimit-limit-requests":"10000","x-ratelimit-limit-tokens":"10000000","x-ratelimit-remaining-requests":"9997","x-ratelimit-remaining-tokens":"161194","x-ratelimit-reset-requests":"18ms","x-ratelimit-reset-tokens":"59.032s"},"message":"Rate limit reached for gpt-6-luna in organization org-x on tokens per min (TPM): Limit 10000000, Used 9838806, Requested 201732. Please try again in 243ms. Visit https://platform.openai.com/account/rate-limits to learn more.","param":null},"sequence_number":2}"#;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn classifies_both_dialects_in_stream_and_http_shapes() {
        for msg in [
            OPENAI_IN_STREAM,
            // Anthropic in-stream (`dialect::sse_error` renders `{type}: {message}`).
            "provider stream error: rate_limit_error: Number of request tokens has exceeded your per-minute rate limit",
            "provider stream error: overloaded_error: Overloaded",
            // `client.rs`'s final error after its own pre-first-byte retries, both providers.
            r#"api.openai.com returned 429 Too Many Requests (retry-after-ms: 2000): {"error":{"type":"requests","code":"rate_limit_exceeded","message":"Rate limit reached"}}"#,
            r#"api.anthropic.com returned 429 Too Many Requests: {"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed your rate limit"}}"#,
            r#"api.anthropic.com returned 529 <unknown status code>: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            "api.anthropic.com returned 529 <unknown status code>: ",
            // The gateway's rewrite of a managed key's upstream rate limit (`remedy::RATE_LIMITED`).
            r#"gateway.example returned 429 Too Many Requests: {"error":{"type":"rate_limit_error","message":"The provider is rate-limited upstream; retry later."}}"#,
            // The gateway's own tenant limit.
            r#"gateway.example returned 429 Too Many Requests (retry-after-ms: 1000): {"error":{"type":"rate_limit_error","message":"rate limit exceeded"}}"#,
        ] {
            assert!(
                is_rate_limited(&Error::Transport(msg.into())),
                "should be a rate limit: {msg}"
            );
        }
    }

    #[test]
    fn an_overload_is_told_apart_from_a_rate_limit() {
        let kind = |m: &str| classify(&Error::Transport(m.into()));
        for m in [
            "provider stream error: overloaded_error: Overloaded",
            r#"api.anthropic.com returned 529 <unknown status code>: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            "api.anthropic.com returned 529 <unknown status code>: ",
        ] {
            assert_eq!(kind(m), Some(Throttle::Overloaded), "{m}");
        }
        for m in [
            OPENAI_IN_STREAM,
            r#"api.anthropic.com returned 429 Too Many Requests: {"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed your rate limit"}}"#,
        ] {
            assert_eq!(kind(m), Some(Throttle::RateLimited), "{m}");
        }
    }

    #[test]
    fn quota_overflow_spent_budget_and_other_errors_are_not_rate_limits() {
        for msg in [
            r#"api.openai.com returned 429 Too Many Requests: {"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#,
            "prompt is too long: 250000 tokens > 200000 maximum context length",
            "Anthropic stream ended before message_stop",
            "provider stream error: server_error: boom",
            r#"api.openai.com returned 429 Too Many Requests: {"error":{"message":"Request too large for gpt-4.1 on tokens per min (TPM): Limit 30000, Requested 50000. The input or output tokens must be reduced in order to run successfully.","type":"tokens","code":"rate_limit_exceeded"}}"#,
        ] {
            assert!(
                !is_rate_limited(&Error::Transport(msg.into())),
                "should not be a rate limit: {msg}"
            );
        }
        let spent = format!("{BUDGET_EXHAUSTED} after 9 attempts over 300s: {OPENAI_IN_STREAM}");
        assert!(!is_rate_limited(&Error::Transport(spent)));
        assert!(!is_rate_limited(&Error::Cancelled));
    }

    #[test]
    fn hint_prefers_retry_after_ms_then_seconds_then_prose_then_reset() {
        assert_eq!(retry_hint(OPENAI_IN_STREAM), Some(ms(243)));
        assert_eq!(
            retry_hint(r#"{"headers":{"retry-after":"7"},"message":"x"}"#),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            retry_hint("host returned 429 Too Many Requests (retry-after-ms: 1500): {}"),
            Some(ms(1500))
        );
        assert_eq!(
            retry_hint("Rate limit reached. Please try again in 1.5s. Visit"),
            Some(ms(1500))
        );
        assert_eq!(
            retry_hint("Rate limit reached. Please try again in 6m0s."),
            Some(Duration::from_secs(360))
        );
        assert_eq!(
            retry_hint("Please try again in 20 seconds"),
            Some(Duration::from_secs(20))
        );
        // Only reset headers: the longest one (the bucket is full again by then).
        assert_eq!(
            retry_hint(
                r#""x-ratelimit-reset-requests":"18ms","x-ratelimit-reset-tokens":"1m2.5s""#
            ),
            Some(ms(62_500))
        );
        // An escaped JSON rendering (`Debug` of the error) still parses.
        assert_eq!(retry_hint(r#"{\"retry-after-ms\":\"472\"}"#), Some(ms(472)));
        assert_eq!(retry_hint("rate_limit_error: slow down"), None);
        assert_eq!(retry_hint("retry-after: soon"), None);
        // Absurd values clamp instead of panicking.
        assert_eq!(
            retry_hint("retry-after: 99999999999999999999"),
            Some(Duration::from_secs(86_400))
        );
    }

    #[test]
    fn wait_floors_at_the_hint_and_backs_off_exponentially_to_the_cap() {
        let p = RateLimitRetry::default();
        // A short hint is a floor under the exponential backoff, not a replacement for it.
        let w = p.wait(1, Some(ms(243)));
        assert!(w >= ms(400) && w <= ms(600), "{w:?}");
        // A long hint wins over a short backoff.
        assert_eq!(
            p.wait(1, Some(Duration::from_secs(20))),
            Duration::from_secs(20)
        );
        // ... but never past the per-wait cap.
        assert_eq!(p.wait(1, Some(Duration::from_secs(600))), DEFAULT_MAX_WAIT);
        let w4 = p.wait(4, None);
        assert!(w4 >= ms(3200) && w4 <= ms(4800), "{w4:?}");
        assert_eq!(p.wait(30, None), DEFAULT_MAX_WAIT);
    }
}
