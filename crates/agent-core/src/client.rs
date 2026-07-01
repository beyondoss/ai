//! The default model transport: an HTTP client that speaks provider wire to the Beyond gateway.
//!
//! This is the harness's whole network surface. It never holds a provider key or picks a provider —
//! it sends `Authorization: Bearer <bai_v1…>` to the gateway, which swaps in the pool key, routes to
//! the real provider, and meters usage. The client only picks the *dialect* (by model id), builds
//! the request body, and frames the streaming SSE response back into [`StreamEvent`]s.

use std::time::Duration;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde_json::Value;

use crate::dialect::{Dialect, push_sse_line};
use crate::error::{Error, Result};
use crate::transport::{EventStream, ModelRequest, ModelTransport};

/// Anthropic's Messages API requires this header; the gateway relays it to the upstream verbatim.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Beta opt-in sent with thinking requests: lets the model interleave thinking between tool calls
/// across a turn (and keeps fine-grained streaming of the thinking blocks).
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
/// Prompt-caching beta opt-in. GA for current models, but sent explicitly so caching engages
/// regardless of the `anthropic-version` default.
const PROMPT_CACHING_BETA: &str = "prompt-caching-2024-07-31";

/// How many times to re-issue a request that failed transiently (connection refused, timeout, or a
/// retryable status) before giving up. A multi-step agent run re-issues a request every turn, so a
/// single transient gateway hiccup would otherwise vaporize the whole run; the gateway itself is
/// behind a load balancer that sheds load with 429/503 under pressure. Public so a caller overriding
/// only one of the two [`GatewayClient::with_retry`] parameters can default the other to this.
pub const MAX_RETRIES: u32 = 3;
/// Base of the exponential backoff between retries (`BASE · 2^(attempt-1)`). Public — see
/// [`MAX_RETRIES`].
pub const BASE_BACKOFF: Duration = Duration::from_millis(250);
/// Ceiling on a single backoff wait, so a server-supplied `Retry-After` can't park a run for minutes.
const MAX_BACKOFF: Duration = Duration::from_secs(10);

/// Cap on the TCP+TLS handshake to the gateway. Mirrors the gateway's own upstream
/// `connect_timeout_secs` (10s): a connection it can't establish in this window is a dead gateway,
/// not a slow one.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Idle timeout *between* reads on the streaming body — **not** a ceiling on total stream duration
/// (an overall `Client::timeout` would wrongly kill long-but-healthy streams). It catches the one
/// failure this layer can't otherwise detect: a gateway holding the connection open but sending no
/// bytes.
///
/// Set to match the gateway's upstream `read_timeout_secs` (600s). We sit *downstream* of the
/// gateway, which applies that same 600s idle timeout to the provider connection — so the gateway
/// can legitimately send us nothing for up to 600s while it waits on the provider (a long
/// extended-thinking gap; gateway TTFT tails to 30s and full-request latency to 600s in its own
/// metrics). A tighter timeout here would sever a stream the gateway still considers healthy: a
/// downstream hop's patience must be at least its upstream's.
const READ_TIMEOUT: Duration = Duration::from_secs(600);

/// An HTTP client pointed at a Beyond gateway base URL, authenticated with a `bai_v1` virtual key.
pub struct GatewayClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    max_retries: u32,
    base_backoff: Duration,
}

impl GatewayClient {
    /// Build a client for `base_url` (e.g. `http://ai.internal` or `http://127.0.0.1:8080`) using
    /// `api_key` (a `bai_v1…` virtual key, or a BYO provider key the gateway forwards untouched).
    /// Pre-first-byte retry defaults to [`MAX_RETRIES`]/[`BASE_BACKOFF`]; override with
    /// [`with_retry`](Self::with_retry) if an operator needs a different budget for this deployment.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .build()
            .map_err(|e| Error::Transport(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            max_retries: MAX_RETRIES,
            base_backoff: BASE_BACKOFF,
        })
    }

    /// Builder-style: override the pre-first-byte retry budget and exponential-backoff base (still
    /// capped at [`MAX_BACKOFF`]).
    pub fn with_retry(mut self, max_retries: u32, base_backoff: Duration) -> Self {
        self.max_retries = max_retries;
        self.base_backoff = base_backoff;
        self
    }
}

#[async_trait]
impl ModelTransport for GatewayClient {
    async fn stream(&self, req: ModelRequest) -> Result<EventStream> {
        let dialect = Dialect::for_model(&req.model);
        let url = format!("{}{}", self.base_url, dialect.endpoint_path());
        let body = dialect.build_body(&req);
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let max_retries = self.max_retries;
        let base_backoff = self.base_backoff;

        let is_anthropic = dialect == Dialect::Anthropic;
        // Interleaved thinking lets the model weave thinking between tool calls across a turn — but
        // it's only meaningful for the `Budget` shape; `Adaptive` models interleave by default, and
        // sending the beta opt-in for them is a harmless no-op at best, so skip it to keep the header
        // list accurate to what the request actually needs.
        let needs_interleaved_beta = req.thinking.is_some()
            && crate::models::capabilities(&req.model).thinking
                != crate::models::ThinkingShape::Adaptive;
        let stream = async_stream::try_stream! {
            // Retry the request up to the first byte: a transient failure (refused connection, 429,
            // 503) is re-issued with backoff. We do *not* retry once events have started flowing — a
            // mid-stream drop would replay partial output — so that surfaces as a transport error the
            // loop handles instead (see `Agent::run_events`).
            let resp = send_with_retry(
                &http,
                &url,
                &api_key,
                &body,
                is_anthropic,
                needs_interleaved_beta,
                max_retries,
                base_backoff,
            )
            .await?;

            // Frame the chunked body line-by-line. SSE for both providers carries one JSON object
            // per `data:` line, so a line splitter suffices; a partial trailing line is buffered
            // across chunks until its newline arrives. The framing (byte buffering + newline split)
            // lives in `LineFramer` — see its doc comment for why it buffers raw *bytes*, not a
            // lossy per-chunk string.
            let mut decoder = dialect.decoder();
            let mut framer = LineFramer::new();
            let mut bytes = resp.bytes_stream();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| Error::Transport(e.to_string()))?;
                framer.extend(&chunk);
                while let Some(line) = framer.next_line() {
                    let line = std::str::from_utf8(&line)
                        .map_err(|e| Error::Transport(format!("invalid UTF-8 in SSE stream: {e}")))?;
                    for ev in push_sse_line(decoder.as_mut(), line)? {
                        yield ev;
                    }
                }
            }
            if let Some(line) = framer.take_tail() {
                let line = std::str::from_utf8(&line)
                    .map_err(|e| Error::Transport(format!("invalid UTF-8 in SSE stream: {e}")))?;
                for ev in push_sse_line(decoder.as_mut(), line)? {
                    yield ev;
                }
            }
            for ev in decoder.finish()? {
                yield ev;
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Reassembles a chunked byte stream into whole newline-terminated lines — the SSE framing seam.
///
/// It buffers raw *bytes*, not a per-chunk lossy string: a TCP/HTTP chunk boundary can split a
/// multi-byte UTF-8 character, and `from_utf8_lossy` per chunk would replace each half with U+FFFD,
/// silently corrupting non-ASCII tool arguments and prose. A `\n` (0x0A) never falls inside a UTF-8
/// multi-byte sequence, so every newline-terminated line handed back by [`next_line`](Self::next_line)
/// is guaranteed whole UTF-8; only the unterminated tail — which may split a character — stays
/// buffered for the next chunk (surfaced by [`take_tail`](Self::take_tail) at end-of-stream).
///
/// Public so the streaming decode hot path is benchable in isolation (`benches/decode.rs`), the same
/// way the gateway exposes its request-scan primitives.
///
/// Backed by a [`BytesMut`]: [`next_line`](Self::next_line) finds the newline with SIMD [`memchr`] and
/// hands the line back via `split_to`, an O(1) pointer split that shares the backing allocation — so a
/// line costs neither a per-line heap allocation nor a memmove of the buffer remainder. (A `Vec<u8>`
/// framer paid both on *every* line: `drain(..=nl).collect()` allocates the line and shifts the rest
/// of the buffer down, which is O(lines × remaining) when a chunk carries many coalesced lines.)
#[derive(Default)]
pub struct LineFramer {
    buf: BytesMut,
}

impl LineFramer {
    /// A framer with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a freshly-received chunk to the buffer.
    pub fn extend(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Pop the next complete line (including its trailing `\n`), or `None` if the buffer holds no
    /// full line yet — the caller then awaits the next chunk. The returned [`Bytes`] shares the
    /// framer's backing buffer (no copy); it's dropped as soon as the line is decoded, freeing that
    /// region for the buffer to reclaim.
    pub fn next_line(&mut self) -> Option<Bytes> {
        let nl = memchr::memchr(b'\n', &self.buf)?;
        Some(self.buf.split_to(nl + 1).freeze())
    }

    /// The leftover unterminated tail at end-of-stream (a final line with no trailing newline), or
    /// `None` if the buffer is empty. Consumes the buffer.
    pub fn take_tail(&mut self) -> Option<Bytes> {
        if self.buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buf).freeze())
        }
    }
}

/// POST the request body, retrying transient failures with exponential backoff until a successful
/// response or the retry budget is exhausted. Honors a `Retry-After` header when the server sends one.
#[allow(clippy::too_many_arguments)]
async fn send_with_retry(
    http: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &Value,
    is_anthropic: bool,
    needs_interleaved_beta: bool,
    max_retries: u32,
    base_backoff: Duration,
) -> Result<reqwest::Response> {
    let mut attempt = 0u32;
    loop {
        let mut builder = http.post(url).bearer_auth(api_key).json(body);
        if is_anthropic {
            builder = builder.header("anthropic-version", ANTHROPIC_VERSION);
            // Comma-separated beta opt-ins. Prompt caching is GA but harmless to opt into explicitly;
            // interleaved thinking is added only for `Budget`-shape thinking requests.
            let mut betas = vec![PROMPT_CACHING_BETA];
            if needs_interleaved_beta {
                betas.push(INTERLEAVED_THINKING_BETA);
            }
            builder = builder.header("anthropic-beta", betas.join(","));
        }
        match builder.send().await {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                if is_retryable_status(status.as_u16()) && attempt < max_retries {
                    let wait = backoff(attempt, retry_after(&resp), base_backoff);
                    attempt += 1;
                    futures_timer::Delay::new(wait).await;
                    continue;
                }
                // Non-retryable, or out of retries: surface the body so the caller sees *why* — capped,
                // since an upstream can return an arbitrarily large error page (an HTML error document
                // from a misconfigured proxy, say) and this ends up in logs and `AgentEvent::Error`.
                let detail = resp.text().await.unwrap_or_default();
                return Err(Error::Transport(format!(
                    "gateway returned {status}: {}",
                    truncate_error_body(detail.trim())
                )));
            }
            Err(e) => {
                // Connection-level failures (refused, reset, timed out) are exactly the transient
                // class worth retrying; a malformed-request error is not.
                if is_retryable_send_error(&e) && attempt < max_retries {
                    let wait = backoff(attempt, None, base_backoff);
                    attempt += 1;
                    futures_timer::Delay::new(wait).await;
                    continue;
                }
                return Err(Error::Transport(e.to_string()));
            }
        }
    }
}

/// Status codes worth retrying: rate limiting, the Anthropic-specific `529 overloaded`, and the
/// transient 5xx gateway/upstream failures. A 4xx other than 429 is the caller's fault — don't retry.
fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 429 | 500 | 502 | 503 | 504 | 529)
}

/// Whether a `reqwest` send error is the transient connection class (refused/reset/timed out).
fn is_retryable_send_error(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect()
}

/// Cap on an upstream error body surfaced in [`Error::Transport`] — an error page (a misconfigured
/// proxy's HTML, say) can be arbitrarily large, and this text ends up in logs and `AgentEvent::Error`.
const MAX_ERROR_BODY_CHARS: usize = 4_000;

/// Truncate an error body to [`MAX_ERROR_BODY_CHARS`], on a char boundary, noting what was cut.
fn truncate_error_body(s: &str) -> String {
    if s.chars().count() <= MAX_ERROR_BODY_CHARS {
        return s.to_string();
    }
    let kept: String = s.chars().take(MAX_ERROR_BODY_CHARS).collect();
    format!("{kept}… [truncated]")
}

/// Parse a `Retry-After` response header into a duration, capped at [`MAX_BACKOFF`].
fn retry_after(resp: &reqwest::Response) -> Option<Duration> {
    let raw = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?;
    parse_retry_after(raw)
}

/// Parse a `Retry-After` header *value* into a wait, capped at [`MAX_BACKOFF`]. RFC 7231 allows two
/// forms: delta-seconds (`120`) and an absolute HTTP-date (`Wed, 21 Oct 2025 07:28:00 GMT`). The
/// date form is converted to a delay from now; a date already in the past (clock skew, a stale hint)
/// yields no extra wait. Split out from [`retry_after`] so it's testable without a `reqwest::Response`.
fn parse_retry_after(raw: &str) -> Option<Duration> {
    let raw = raw.trim();
    // Delta-seconds: a bare non-negative integer count of seconds.
    if let Ok(secs) = raw.parse::<u64>() {
        return Some(Duration::from_secs(secs).min(MAX_BACKOFF));
    }
    // HTTP-date: the absolute instant to retry at. Anything we can't parse as either form is ignored.
    let target = httpdate::parse_http_date(raw).ok()?;
    let delay = target
        .duration_since(std::time::SystemTime::now())
        .unwrap_or(Duration::ZERO);
    Some(delay.min(MAX_BACKOFF))
}

/// The wait before the next attempt (0-indexed): the larger of the server's `Retry-After` hint and
/// our exponential backoff `base_backoff · 2^attempt`, capped at [`MAX_BACKOFF`].
fn backoff(attempt: u32, retry_after: Option<Duration>, base_backoff: Duration) -> Duration {
    // `min(16)` keeps the shift well within `u32` (and `saturating_mul` mops up the rest); by then the
    // result has long since hit `MAX_BACKOFF`.
    let exp = base_backoff
        .saturating_mul(1u32 << attempt.min(16))
        .min(MAX_BACKOFF);
    match retry_after {
        Some(hint) => hint.max(exp).min(MAX_BACKOFF),
        None => exp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_status_classification() {
        for s in [429, 500, 502, 503, 504, 529, 408, 409] {
            assert!(is_retryable_status(s), "{s} should be retryable");
        }
        for s in [200, 400, 401, 403, 404, 422] {
            assert!(!is_retryable_status(s), "{s} should not be retryable");
        }
    }

    #[test]
    fn backoff_is_exponential_and_capped() {
        assert_eq!(backoff(0, None, BASE_BACKOFF), BASE_BACKOFF);
        assert_eq!(backoff(1, None, BASE_BACKOFF), BASE_BACKOFF * 2);
        assert_eq!(backoff(2, None, BASE_BACKOFF), BASE_BACKOFF * 4);
        assert_eq!(backoff(20, None, BASE_BACKOFF), MAX_BACKOFF); // saturates, never overflows
        // A server hint wins when larger, but is still capped.
        assert_eq!(
            backoff(0, Some(Duration::from_secs(2)), BASE_BACKOFF),
            Duration::from_secs(2)
        );
        assert_eq!(
            backoff(0, Some(Duration::from_secs(3600)), BASE_BACKOFF),
            MAX_BACKOFF
        );
    }

    #[test]
    fn backoff_honors_a_custom_base() {
        let custom = Duration::from_millis(1000);
        assert_eq!(backoff(0, None, custom), custom);
        assert_eq!(backoff(1, None, custom), custom * 2);
        assert_eq!(backoff(20, None, custom), MAX_BACKOFF); // still saturates at the shared cap
    }

    #[test]
    fn truncate_error_body_caps_a_large_upstream_error_page() {
        let short = "gateway timeout";
        assert_eq!(truncate_error_body(short), short);

        let huge = "x".repeat(MAX_ERROR_BODY_CHARS + 500);
        let truncated = truncate_error_body(&huge);
        assert!(truncated.ends_with("… [truncated]"));
        assert_eq!(
            truncated.chars().count(),
            MAX_ERROR_BODY_CHARS + "… [truncated]".chars().count()
        );
    }

    #[test]
    fn retry_after_accepts_delta_seconds_and_http_date() {
        // Delta-seconds form, capped at MAX_BACKOFF.
        assert_eq!(parse_retry_after("2"), Some(Duration::from_secs(2)));
        assert_eq!(parse_retry_after(" 3 "), Some(Duration::from_secs(3)));
        assert_eq!(parse_retry_after("99999"), Some(MAX_BACKOFF));
        // A value that is neither an integer nor an HTTP-date is ignored.
        assert_eq!(parse_retry_after("soon"), None);
        // HTTP-date already in the past → no extra wait.
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"),
            Some(Duration::ZERO)
        );
        // HTTP-date in the future → a positive, capped delay.
        let future = std::time::SystemTime::now() + Duration::from_secs(5);
        let delay = parse_retry_after(&httpdate::fmt_http_date(future)).expect("a parsed delay");
        assert!(
            delay > Duration::ZERO && delay <= MAX_BACKOFF,
            "future http-date should yield a bounded positive delay, got {delay:?}"
        );
    }
}
