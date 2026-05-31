//! Admin / observability HTTP surface served on the metrics listener: `/livez`, `/readyz`,
//! `/metrics`.
//!
//! Matches the Beyond service convention (cf. `auth`, `objects`): the body is `{"status",
//! "version"}` and there are two probes. Both return 200 once the process is answering, because
//! the gateway is **fail-open by design** — auth + key swap come from boot config, and a NATS
//! outage degrades only the (stale) deny-set, never the ability to serve. So readiness must *not*
//! gate on NATS: a cold boot with NATS down can still serve correctly, and reporting not-ready
//! would pull a healthy gateway out of the load balancer for no reason. Readiness here therefore
//! means "listeners up + boot config loaded" — which is true the instant we can answer this
//! request (state is built before the server starts; a build failure `exit`s in `main`).
//! `readyz` is kept distinct from `livez` only to honor the orchestrator's two-probe convention.
//!
//! Implemented as a Pingora `ServeHttp` app so all three paths share the one (internal) metrics
//! port — Pingora's built-in prometheus service only serves `/metrics`, so we hand-route all three.

use async_trait::async_trait;
use http::Response;
use pingora_core::apps::http_app::ServeHttp;
use pingora_core::protocols::http::ServerSession;
use prometheus::{Encoder, TextEncoder};

/// Compile-time service version, surfaced in every health body (matches the sibling services).
const VERSION: &str = env!("CARGO_PKG_VERSION");

pub struct AdminApp;

impl AdminApp {
    /// Build a `{"status","version"}` JSON health response. `status` is `"ok"`/`"degraded"` so a
    /// human or a probe can read intent without parsing the code. Header values are all static or
    /// integer, so the builder can't fail — `expect` documents that invariant.
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
    fn metrics() -> Response<Vec<u8>> {
        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
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
            // Liveness + readiness are the same signal here (see module docs): the gateway is
            // fail-open, so "can answer" ⇒ "can serve". Both 200 once the process is up.
            "/livez" | "/readyz" => Self::health(200, "ok"),
            "/metrics" => Self::metrics(),
            _ => Self::health(404, "not_found"),
        }
    }
}
