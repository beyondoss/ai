//! Admin / observability HTTP surface served on the metrics listener: `/livez`, `/readyz`,
//! `/metrics`.
//!
//! Matches the Beyond service convention (cf. `auth`, `objects`): the body is `{"status",
//! "version"}` and there are two probes. **Both always return HTTP 200** once the process is
//! answering, because the gateway is **fail-open by design** — auth + key swap come from boot
//! config, and a NATS outage degrades only the (stale) deny-set, never the ability to serve. So
//! readiness must *not* gate on NATS: a cold boot with NATS down can still serve correctly, and a
//! non-200 would pull a healthy gateway out of the load balancer for no reason.
//!
//! `readyz` does, however, carry a distinct *body* signal that `livez` doesn't: when the deny-set
//! watcher is disconnected from NATS, `readyz` reports `"status":"degraded"` (still 200). This lets
//! an operator alert on "readyz has been degraded for >N minutes" — the spend/fraud enforcement is
//! stale — without ever risking an LB eviction. `livez` is pure liveness: 200/`"ok"` whenever the
//! process can answer. (The `ai_nats_connected` gauge is the same signal in Prometheus; the body
//! flag is for orchestrators that probe HTTP but don't scrape.)
//!
//! Implemented as a Pingora `ServeHttp` app so all three paths share the one (internal) metrics
//! port — Pingora's built-in prometheus service only serves `/metrics`, so we hand-route all three.

use crate::metrics::Metrics;
use async_trait::async_trait;
use http::Response;
use pingora_core::apps::http_app::ServeHttp;
use pingora_core::protocols::http::ServerSession;
use prometheus::{Encoder, TextEncoder};
use std::sync::Arc;

/// Compile-time service version, surfaced in every health body (matches the sibling services).
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct AdminApp {
    /// Read-only handle to the metric gauges. Used by `/readyz` to reflect NATS connectivity in the
    /// health body (never to gate the HTTP status — see module docs).
    pub metrics: Arc<Metrics>,
}

impl AdminApp {
    /// Build a `{"status","version"}` JSON health response. `status` is `"ok"`/`"degraded"` so a
    /// human or a probe can read intent without parsing the code. Header values are all static or
    /// integer, so the builder can't fail — `expect` documents that invariant.
    #[allow(clippy::expect_used)] // builder inputs are all static/integer; cannot fail
    fn health(status: u16, health: &str) -> Response<Vec<u8>> {
        let body = serde_json::json!({ "status": health, "version": VERSION })
            .to_string()
            .into_bytes();
        Response::builder()
            .status(status)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::CONTENT_LENGTH, body.len())
            .body(body)
            .expect("static health response is always valid")
    }

    /// Encode the default Prometheus registry as text (same output as Pingora's built-in service).
    #[allow(clippy::expect_used)] // builder inputs are encoder-derived/integer; cannot fail
    fn metrics() -> Response<Vec<u8>> {
        let encoder = TextEncoder::new();
        // Pre-size for a typical scrape: the gateway's fixed metric set renders to a few KiB of
        // text, so one allocation up front avoids the handful of reallocs `Vec::new` would incur as
        // the encoder appends. 8 KiB comfortably covers the current set with headroom.
        let mut buffer = Vec::with_capacity(8 * 1024);
        // `encode` only errors if the writer fails; a `Vec` never does, so the result is infallible
        // here — discard it explicitly (the crate denies `unused_must_use`).
        let _ = encoder.encode(&prometheus::gather(), &mut buffer);
        Response::builder()
            .status(200)
            .header(http::header::CONTENT_TYPE, encoder.format_type())
            .header(http::header::CONTENT_LENGTH, buffer.len())
            .body(buffer)
            .expect("metrics response is always valid")
    }
}

#[async_trait]
impl ServeHttp for AdminApp {
    async fn response(&self, session: &mut ServerSession) -> Response<Vec<u8>> {
        match session.req_header().uri.path() {
            // Pure liveness: 200/ok whenever the process can answer.
            "/livez" => Self::health(200, "ok"),
            // Readiness: always 200 (fail-open — never pull a serving gateway from the LB), but the
            // body reports `degraded` when the deny-set watcher is disconnected from NATS, so an
            // operator can alert on stale spend/fraud enforcement without an eviction.
            "/readyz" => {
                let health = if self.metrics.nats_connected.get() == 1 {
                    "ok"
                } else {
                    "degraded"
                };
                Self::health(200, health)
            }
            "/metrics" => Self::metrics(),
            _ => Self::health(404, "not_found"),
        }
    }
}
