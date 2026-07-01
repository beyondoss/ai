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
use crate::error::{Error, MID_STREAM_NETWORK_ERROR, Result};
use crate::transport::{EventStream, ModelRequest, ModelTransport};

/// Anthropic's Messages API requires this header; the gateway relays it to the upstream verbatim.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Beta opt-in sent with thinking requests: lets the model interleave thinking between tool calls
/// across a turn (and keeps fine-grained streaming of the thinking blocks).
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
/// Fallback streaming beta for a model whose tool definitions *don't* carry the per-tool
/// `eager_input_streaming: true` marker (see `dialect::anthropic::mark_eager_tool_streaming`) — the two
/// are mutually exclusive; no current model needs this branch, since every current id supports the
/// per-tool marker instead, but it exists for correctness if that ever changes.
const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";

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
        let needs_fine_grained_tool_streaming_beta = !req.tools.is_empty()
            && !crate::models::capabilities(&req.model).supports_eager_tool_streaming;
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
                needs_fine_grained_tool_streaming_beta,
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
                // A tagged prefix, not `e.to_string()` alone: this is the one call site that turns a
                // live `reqwest::Error` into a mid-stream failure (a connection reset, a read timeout,
                // an unexpected EOF — the body was already flowing, so pre-first-byte retry above
                // never sees it), and `Agent::is_retryable_mid_stream` needs to tell "the network
                // dropped us, safe to restart the turn" apart from "the provider rejected the request"
                // without re-deriving reqwest's classification from its Display text.
                let chunk =
                    chunk.map_err(|e| Error::Transport(format!("{MID_STREAM_NETWORK_ERROR}: {e}")))?;
                framer.extend(&chunk)?;
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

/// Ceiling on a single buffered (unterminated) line. A legitimate SSE `data:` line — even one
/// carrying a large embedded payload — stays orders of magnitude under this; it exists solely to
/// bound how much a malformed or adversarial stream (a line whose `\n` never arrives, or a
/// pathologically long one from a misbehaving provider/gateway) can grow the buffer before the
/// stream errors out instead of the process running out of memory.
const MAX_BUFFERED_LINE_BYTES: usize = 32 * 1024 * 1024;

impl LineFramer {
    /// A framer with an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a freshly-received chunk to the buffer. Errors if the unterminated line being
    /// assembled would exceed [`MAX_BUFFERED_LINE_BYTES`] — see that constant's doc comment.
    pub fn extend(&mut self, chunk: &[u8]) -> std::result::Result<(), Error> {
        self.buf.extend_from_slice(chunk);
        if self.buf.len() > MAX_BUFFERED_LINE_BYTES {
            return Err(Error::Transport(format!(
                "SSE line exceeded {MAX_BUFFERED_LINE_BYTES} bytes without a newline"
            )));
        }
        Ok(())
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

/// Comma-joined `anthropic-beta` opt-ins for a request, or empty when neither applies — prompt
/// caching has been GA for a long time now and no longer needs (or accepts as meaningful) the old
/// `prompt-caching-2024-07-31` opt-in header pi itself has already dropped, so this crate doesn't send
/// it either. Interleaved thinking is added only for `Budget`-shape thinking requests; the
/// fine-grained tool-streaming beta and each tool definition's own `eager_input_streaming` marker (see
/// `dialect::anthropic::mark_eager_tool_streaming`) are mutually exclusive, so the beta only fires for a
/// model that lacks the per-tool marker.
fn anthropic_betas(
    needs_interleaved: bool,
    needs_fine_grained_tool_streaming: bool,
) -> Vec<&'static str> {
    let mut betas = Vec::new();
    if needs_interleaved {
        betas.push(INTERLEAVED_THINKING_BETA);
    }
    if needs_fine_grained_tool_streaming {
        betas.push(FINE_GRAINED_TOOL_STREAMING_BETA);
    }
    betas
}

/// POST the request body, retrying transient failures with exponential backoff until a successful
/// response or the retry budget is exhausted. Honors a `Retry-After` header when the server sends one.
// 8 arguments, all independent inputs a single call site (`GatewayClient::stream`) already has in
// scope from `self`/the request — bundling them into a struct would just be a second place those same
// fields live, not a reduction in what the function needs to know. Private, single-caller helper, not
// a public API shape, so the usual "too many params signals a missing abstraction" concern doesn't
// apply here.
#[allow(clippy::too_many_arguments)]
async fn send_with_retry(
    http: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &Value,
    is_anthropic: bool,
    needs_interleaved_beta: bool,
    needs_fine_grained_tool_streaming_beta: bool,
    max_retries: u32,
    base_backoff: Duration,
) -> Result<reqwest::Response> {
    let mut attempt = 0u32;
    loop {
        let mut builder = http.post(url).bearer_auth(api_key).json(body);
        if is_anthropic {
            builder = builder.header("anthropic-version", ANTHROPIC_VERSION);
            let betas = anthropic_betas(
                needs_interleaved_beta,
                needs_fine_grained_tool_streaming_beta,
            );
            // Omit the header entirely when nothing needs it, rather than sending an empty
            // `anthropic-beta:` value — matching pi's own conditional-spread behavior.
            if !betas.is_empty() {
                builder = builder.header("anthropic-beta", betas.join(","));
            }
        }
        match builder.send().await {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                if is_retryable_status(status.as_u16()) && attempt < max_retries {
                    let hint = retry_after(&resp);
                    // A 429 needs a quick body peek before committing to a retry: some providers use it
                    // for genuine rate limiting (worth retrying — the request will likely succeed once
                    // the window resets) and others for quota/billing exhaustion (retrying only delays
                    // an unavoidable failure while burning the retry budget on it). The status code
                    // alone can't tell the two apart. Every other retryable status (408/409/5xx/529) is
                    // a pure infra hiccup, never a billing signal, so it skips this check.
                    if status.as_u16() == 429 {
                        let detail = resp.text().await.unwrap_or_default();
                        if is_quota_exhausted(&detail) {
                            return Err(Error::Transport(format!(
                                "gateway returned {status}: {}",
                                truncate_error_body(detail.trim())
                            )));
                        }
                    }
                    let wait = backoff(attempt, hint, base_backoff);
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

/// Phrases seen in a 429 body when the rejection is quota/billing exhaustion rather than transient
/// rate limiting — retrying one of these will never succeed until the account itself changes, so it
/// isn't worth spending the retry budget on. Deliberately narrower than `agent::is_context_overflow`'s
/// throttle-exclusion list: this only needs to cover the "don't bother retrying a 429" case, not
/// classify every provider's wording.
const QUOTA_EXHAUSTED_PATTERNS: &[&str] = &[
    "insufficient_quota",
    "quota exceeded",
    "billing",
    "out of budget",
    "exceeded your current quota",
];

/// Whether a 429 response body indicates quota/billing exhaustion (fail fast) rather than ordinary
/// rate limiting (worth retrying).
fn is_quota_exhausted(body: &str) -> bool {
    let m = body.to_ascii_lowercase();
    QUOTA_EXHAUSTED_PATTERNS.iter().any(|p| m.contains(p))
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
    fn line_framer_splits_whole_lines_and_buffers_the_tail() {
        let mut framer = LineFramer::new();
        framer.extend(b"data: {\"a\":1}\ndata: {\"b").unwrap();
        assert_eq!(framer.next_line().unwrap(), &b"data: {\"a\":1}\n"[..]);
        assert!(framer.next_line().is_none()); // unterminated remainder stays buffered
        framer.extend(b"\":2}\n").unwrap();
        assert_eq!(framer.next_line().unwrap(), &b"data: {\"b\":2}\n"[..]);
    }

    #[test]
    fn line_framer_errors_instead_of_growing_unbounded_on_an_unterminated_line() {
        // A malformed/adversarial stream (or a provider bug) that never sends a `\n` must not be
        // allowed to grow the buffer without bound — this is what caps it.
        let mut framer = LineFramer::new();
        let chunk = vec![b'x'; MAX_BUFFERED_LINE_BYTES];
        framer.extend(&chunk).unwrap(); // exactly at the cap: still fine
        assert!(framer.extend(b"y").is_err()); // one more byte tips it over
    }

    #[test]
    fn anthropic_betas_are_mutually_exclusive_with_eager_tool_streaming() {
        // Baseline: neither beta needed — the stale prompt-caching-2024-07-31 opt-in is gone, so this
        // is empty, not a single-element list.
        assert_eq!(anthropic_betas(false, false), Vec::<&str>::new());
        // Interleaved thinking layers on top.
        assert_eq!(
            anthropic_betas(true, false),
            vec![INTERLEAVED_THINKING_BETA]
        );
        // The fine-grained tool-streaming beta only fires when the model lacks the per-tool
        // `eager_input_streaming` marker — never true for any current model, but exercised directly.
        assert_eq!(
            anthropic_betas(false, true),
            vec![FINE_GRAINED_TOOL_STREAMING_BETA]
        );
        assert_eq!(
            anthropic_betas(true, true),
            vec![INTERLEAVED_THINKING_BETA, FINE_GRAINED_TOOL_STREAMING_BETA]
        );
    }

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
    fn quota_exhaustion_classification() {
        for body in [
            r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#,
            r#"{"error":"quota exceeded for this billing period"}"#,
            r#"{"error":"please add a payment method — billing required"}"#,
            r#"{"error":"you are out of budget for this key"}"#,
        ] {
            assert!(
                is_quota_exhausted(body),
                "should classify as quota exhaustion: {body}"
            );
        }
        for body in [
            r#"{"error":{"type":"rate_limit_error","message":"Too many requests, please slow down"}}"#,
            "",
            "gateway timeout",
        ] {
            assert!(
                !is_quota_exhausted(body),
                "should NOT classify as quota exhaustion: {body}"
            );
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

    /// A real TCP peer that answers with valid SSE headers plus one partial event, then vanishes
    /// without a clean shutdown — what an abrupt connection reset or a crashed upstream looks like on
    /// the wire, as opposed to anything this crate constructs itself.
    #[tokio::test]
    async fn a_connection_dropped_mid_body_is_tagged_as_a_mid_stream_network_error() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            // Drain the request so the response write doesn't race a half-closed read side.
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            // `Content-Length` promises far more body than actually arrives before the connection
            // vanishes — unlike a close-delimited (`Connection: close`, no length) body, where EOF
            // *is* the defined end-of-body and reqwest would read this as a normal, if short, success.
            // A length mismatch is what makes the abrupt close an actual framing violation reqwest
            // surfaces as a body-read error, matching what a mid-response connection reset looks like
            // against a real chunked/length-bearing gateway response.
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100000\r\n\r\n\
                      data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
                )
                .unwrap();
            stream.flush().unwrap();
            // No `message_stop`, no clean FIN — the connection just disappears mid-response.
            drop(stream);
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key").unwrap();
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap(); // the request itself succeeds
        let mut tagged = false;
        while let Some(ev) = events.next().await {
            if let Err(Error::Transport(msg)) = ev {
                tagged = msg.contains(MID_STREAM_NETWORK_ERROR);
                break;
            }
        }
        server.join().unwrap();
        assert!(
            tagged,
            "a mid-body connection drop must surface as a MID_STREAM_NETWORK_ERROR-tagged transport error"
        );
    }

    /// A real TCP peer that always answers 429 with a quota-exhaustion body — proves the client fails
    /// fast (one request, no retry) instead of burning its retry budget on a rejection retrying can
    /// never fix.
    #[tokio::test]
    async fn a_429_with_quota_exhaustion_body_is_not_retried() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = request_count.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut stream, _)) = listener.accept() {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let body = r#"{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}"#;
                let resp = format!(
                    "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_retry(3, Duration::from_millis(10));
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        let first = events.next().await;
        server.join().unwrap();

        assert!(
            matches!(first, Some(Err(Error::Transport(_)))),
            "expected a transport error, got {first:?}"
        );
        assert_eq!(
            request_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a quota-exhausted 429 must not be retried"
        );
    }

    /// A real TCP peer that answers 429 with an ordinary rate-limit body (no quota/billing phrase) on
    /// the first request, then succeeds on the retry — proves ordinary rate limiting still gets the
    /// normal retry treatment `is_quota_exhausted` doesn't touch.
    #[tokio::test]
    async fn a_429_with_a_plain_rate_limit_body_is_retried() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = request_count.clone();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            for attempt in 0..2 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                if attempt == 0 {
                    let body =
                        r#"{"error":{"type":"rate_limit_error","message":"Too many requests"}}"#;
                    let resp = format!(
                        "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else {
                    let sse = "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\ndata: {\"type\":\"message_stop\"}\n\n";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        sse.len(),
                        sse
                    );
                    let _ = stream.write_all(resp.as_bytes());
                }
            }
        });

        let client = GatewayClient::new(format!("http://{addr}"), "test-key")
            .unwrap()
            .with_retry(3, Duration::from_millis(10));
        let req = ModelRequest::new("claude-test", Vec::new(), 100);
        let mut events = client.stream(req).await.unwrap();
        let _ = events.next().await; // drive the stream far enough to trigger the retry + second request
        server.join().unwrap();

        assert_eq!(
            request_count.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "an ordinary rate-limit 429 must still be retried"
        );
    }
}
