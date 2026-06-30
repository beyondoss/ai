//! The default model transport: an HTTP client that speaks provider wire to the Beyond gateway.
//!
//! This is the harness's whole network surface. It never holds a provider key or picks a provider —
//! it sends `Authorization: Bearer <bai_v1…>` to the gateway, which swaps in the pool key, routes to
//! the real provider, and meters usage. The client only picks the *dialect* (by model id), builds
//! the request body, and frames the streaming SSE response back into [`StreamEvent`]s.

use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;

use crate::dialect::{Dialect, push_sse_line};
use crate::error::{Error, Result};
use crate::transport::{EventStream, ModelRequest, ModelTransport};

/// Anthropic's Messages API requires this header; the gateway relays it to the upstream verbatim.
const ANTHROPIC_VERSION: &str = "2023-06-01";

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

        let stream = async_stream::try_stream! {
            let mut builder = http.post(&url).bearer_auth(&api_key).json(&body);
            if dialect == Dialect::Anthropic {
                builder = builder.header("anthropic-version", ANTHROPIC_VERSION);
            }

            let resp = builder.send().await.map_err(|e| Error::Transport(e.to_string()))?;
            let status = resp.status();
            if !status.is_success() {
                let detail = resp.text().await.unwrap_or_default();
                Err(Error::Transport(format!("gateway returned {status}: {}", detail.trim())))?;
            } else {
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
                for ev in decoder.finish() {
                    yield ev;
                }
            }
        };

        Ok(Box::pin(stream))
    }
}
