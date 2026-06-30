//! The default model transport: an HTTP client that speaks provider wire to the Beyond gateway.
//!
//! This is the harness's whole network surface. It never holds a provider key or picks a provider —
//! it sends `Authorization: Bearer <bai_v1…>` to the gateway, which swaps in the pool key, routes to
//! the real provider, and meters usage. The client only picks the *dialect* (by model id), builds
//! the request body, and frames the streaming SSE response back into [`StreamEvent`]s.

use std::time::Duration;

use async_trait::async_trait;
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
/// behind a load balancer that sheds load with 429/503 under pressure.
const MAX_RETRIES: u32 = 3;
/// Base of the exponential backoff between retries (`BASE · 2^(attempt-1)`).
const BASE_BACKOFF: Duration = Duration::from_millis(250);
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
}

impl GatewayClient {
    /// Build a client for `base_url` (e.g. `http://ai.internal` or `http://127.0.0.1:8080`) using
    /// `api_key` (a `bai_v1…` virtual key, or a BYO provider key the gateway forwards untouched).
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
        })
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

        let is_anthropic = dialect == Dialect::Anthropic;
        let thinking = req.thinking.is_some();
        let stream = async_stream::try_stream! {
            // Retry the request up to the first byte: a transient failure (refused connection, 429,
            // 503) is re-issued with backoff. We do *not* retry once events have started flowing — a
            // mid-stream drop would replay partial output — so that surfaces as a transport error the
            // loop handles instead (see `Agent::run_events`).
            let resp = send_with_retry(&http, &url, &api_key, &body, is_anthropic, thinking).await?;

            // Frame the chunked body line-by-line. SSE for both providers carries one JSON object
            // per `data:` line, so a line splitter suffices; a partial trailing line is buffered
            // across chunks until its newline arrives.
            //
            // We buffer raw *bytes*, not a lossy string: a chunk boundary can split a multi-byte
            // UTF-8 character, and `from_utf8_lossy` per chunk would replace each half with U+FFFD
            // — silently corrupting non-ASCII tool arguments and prose. A `\n` (0x0A) never falls
            // inside a UTF-8 character, so every newline-terminated line is whole UTF-8; only the
            // unterminated tail — which may split a character — stays buffered for the next chunk.
            let mut decoder = dialect.decoder();
            let mut buf: Vec<u8> = Vec::new();
            let mut bytes = resp.bytes_stream();
            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|e| Error::Transport(e.to_string()))?;
                buf.extend_from_slice(&chunk);
                while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=nl).collect();
                    let line = std::str::from_utf8(&line)
                        .map_err(|e| Error::Transport(format!("invalid UTF-8 in SSE stream: {e}")))?;
                    for ev in push_sse_line(decoder.as_mut(), line)? {
                        yield ev;
                    }
                }
            }
            if !buf.is_empty() {
                let line = std::str::from_utf8(&buf)
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

/// POST the request body, retrying transient failures with exponential backoff until a successful
/// response or the retry budget is exhausted. Honors a `Retry-After` header when the server sends one.
async fn send_with_retry(
    http: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &Value,
    is_anthropic: bool,
    thinking: bool,
) -> Result<reqwest::Response> {
    let mut attempt = 0u32;
    loop {
        let mut builder = http.post(url).bearer_auth(api_key).json(body);
        if is_anthropic {
            builder = builder.header("anthropic-version", ANTHROPIC_VERSION);
            // Comma-separated beta opt-ins. Prompt caching is GA but harmless to opt into explicitly;
            // interleaved thinking is added only when thinking is on.
            let mut betas = vec![PROMPT_CACHING_BETA];
            if thinking {
                betas.push(INTERLEAVED_THINKING_BETA);
            }
            builder = builder.header("anthropic-beta", betas.join(","));
        }
        match builder.send().await {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                if is_retryable_status(status.as_u16()) && attempt < MAX_RETRIES {
                    let wait = backoff(attempt, retry_after(&resp));
                    attempt += 1;
                    futures_timer::Delay::new(wait).await;
                    continue;
                }
                // Non-retryable, or out of retries: surface the body so the caller sees *why*.
                let detail = resp.text().await.unwrap_or_default();
                return Err(Error::Transport(format!(
                    "gateway returned {status}: {}",
                    detail.trim()
                )));
            }
            Err(e) => {
                // Connection-level failures (refused, reset, timed out) are exactly the transient
                // class worth retrying; a malformed-request error is not.
                if is_retryable_send_error(&e) && attempt < MAX_RETRIES {
                    let wait = backoff(attempt, None);
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

/// Parse a `Retry-After` header (delta-seconds form) into a duration, capped at [`MAX_BACKOFF`].
fn retry_after(resp: &reqwest::Response) -> Option<Duration> {
    let secs: u64 = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(secs).min(MAX_BACKOFF))
}

/// The wait before the next attempt (0-indexed): the larger of the server's `Retry-After` hint and
/// our exponential backoff `BASE · 2^attempt`, capped at [`MAX_BACKOFF`].
fn backoff(attempt: u32, retry_after: Option<Duration>) -> Duration {
    // `min(16)` keeps the shift well within `u32` (and `saturating_mul` mops up the rest); by then the
    // result has long since hit `MAX_BACKOFF`.
    let exp = BASE_BACKOFF
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
        assert_eq!(backoff(0, None), BASE_BACKOFF);
        assert_eq!(backoff(1, None), BASE_BACKOFF * 2);
        assert_eq!(backoff(2, None), BASE_BACKOFF * 4);
        assert_eq!(backoff(20, None), MAX_BACKOFF); // saturates, never overflows
        // A server hint wins when larger, but is still capped.
        assert_eq!(
            backoff(0, Some(Duration::from_secs(2))),
            Duration::from_secs(2)
        );
        assert_eq!(backoff(0, Some(Duration::from_secs(3600))), MAX_BACKOFF);
    }
}
