//! The default model transport: an HTTP client that speaks provider wire to the Beyond gateway.
//!
//! This is the harness's whole network surface. It never holds a provider key or picks a provider —
//! it sends `Authorization: Bearer <bai_v1…>` to the gateway, which swaps in the pool key, routes to
//! the real provider, and meters usage. The client only picks the *dialect* (by model id), builds
//! the request body, and frames the streaming SSE response back into [`StreamEvent`]s.

use async_trait::async_trait;
use futures::StreamExt;

use crate::dialect::{Dialect, push_sse_line};
use crate::error::{Error, Result};
use crate::transport::{EventStream, ModelRequest, ModelTransport};

/// Anthropic's Messages API requires this header; the gateway relays it to the upstream verbatim.
const ANTHROPIC_VERSION: &str = "2023-06-01";

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
                let mut decoder = dialect.decoder();
                let mut buf = String::new();
                let mut bytes = resp.bytes_stream();
                while let Some(chunk) = bytes.next().await {
                    let chunk = chunk.map_err(|e| Error::Transport(e.to_string()))?;
                    buf.push_str(&String::from_utf8_lossy(&chunk));
                    while let Some(nl) = buf.find('\n') {
                        let line: String = buf.drain(..=nl).collect();
                        for ev in push_sse_line(decoder.as_mut(), &line)? {
                            yield ev;
                        }
                    }
                }
                if !buf.is_empty() {
                    for ev in push_sse_line(decoder.as_mut(), &buf)? {
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
