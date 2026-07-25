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
use std::sync::atomic::{AtomicUsize, Ordering};

/// Render one `{"status","version"}` body at compile time. The shape is fixed and the version is a
/// compile-time constant (`CARGO_PKG_VERSION`), so there is nothing to build per probe — `concat!`
/// folds each body into a `&'static str`. It takes literals only, hence `env!` here rather than a
/// `const VERSION`; keeping the shape in one macro is what stops the bodies from drifting apart.
macro_rules! health_body {
    ($status:literal) => {
        concat!(
            r#"{"status":""#,
            $status,
            r#"","version":""#,
            env!("CARGO_PKG_VERSION"),
            r#""}"#
        )
    };
}

/// The health bodies, one per status word this surface reports (the set is closed — adding one means
/// adding a constant here). Matches the sibling services' `{"status","version"}` shape.
pub const HEALTH_OK: &str = health_body!("ok");
pub const HEALTH_DEGRADED: &str = health_body!("degraded");
pub const HEALTH_NOT_FOUND: &str = health_body!("not_found");

/// Floor for the `/metrics` buffer hint, and its value before any scrape has been seen. Only the
/// *cold* scrape depends on it being in the right ballpark — see [`AdminApp::metrics`].
const MIN_SCRAPE_BUF: usize = 8 * 1024;

/// Encoded length of the last scrape (plus headroom), used to pre-size the next one.
static LAST_SCRAPE: AtomicUsize = AtomicUsize::new(MIN_SCRAPE_BUF);

pub struct AdminApp {
    /// Read-only handle to the metric gauges. Used by `/readyz` to reflect NATS connectivity in the
    /// health body (never to gate the HTTP status — see module docs).
    pub metrics: Arc<Metrics>,
}

impl AdminApp {
    /// Build a health response from one of the pre-rendered `HEALTH_*` bodies, whose `status` field
    /// (`"ok"`/`"degraded"`/`"not_found"`) lets a human or a probe read intent without parsing the
    /// code. Header values are all static or integer, so the builder can't fail — `expect` documents
    /// that invariant.
    #[allow(clippy::expect_used)] // builder inputs are all static/integer; cannot fail
    pub fn health(status: u16, health: &'static str) -> Response<Vec<u8>> {
        let body = health.as_bytes().to_vec();
        Response::builder()
            .status(status)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::CONTENT_LENGTH, body.len())
            .body(body)
            .expect("static health response is always valid")
    }

    /// Encode the default Prometheus registry as text (same output as Pingora's built-in service).
    ///
    /// The buffer is pre-sized from the *previous* scrape rather than from a constant. The constant
    /// this replaces (8 KiB, "the fixed metric set renders to a few KiB") stopped being true when the
    /// per-provider labels landed: `ProviderMetrics::resolve` materialises every child series at boot,
    /// so the full provider × (histogram buckets + status classes) cross-product is in the body from
    /// the first scrape — tens of KiB, i.e. two reallocs and a ~24 KiB memcpy every time. A larger
    /// constant would go stale the same way; the last encoded length is a measurement, and it tracks
    /// providers added by config with nobody remembering to update it.
    #[allow(clippy::expect_used)] // builder inputs are encoder-derived/integer; cannot fail
    pub fn metrics() -> Response<Vec<u8>> {
        let encoder = TextEncoder::new();
        // Relaxed on both ends: this is a size hint, not a shared invariant. Concurrent scrapes
        // racing here cost at worst one of them a realloc (or a few unused bytes), never correctness.
        let mut buffer = Vec::with_capacity(LAST_SCRAPE.load(Ordering::Relaxed));
        // `encode` only errors if the writer fails; a `Vec` never does, so the result is infallible
        // here — discard it explicitly (the crate denies `unused_must_use`).
        let _ = encoder.encode(&prometheus::gather(), &mut buffer);
        // A little headroom over what we just measured, so the steady drift of counter values into
        // more digits doesn't cost a realloc on the next scrape. Derived from the fresh length each
        // time (not compounded from the hint), so it converges instead of creeping upward.
        LAST_SCRAPE.store(
            (buffer.len() + buffer.len() / 8).max(MIN_SCRAPE_BUF),
            Ordering::Relaxed,
        );
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
            "/livez" => Self::health(200, HEALTH_OK),
            // Readiness: always 200 (fail-open — never pull a serving gateway from the LB), but the
            // body reports `degraded` when the deny-set watcher is disconnected from NATS, so an
            // operator can alert on stale spend/fraud enforcement without an eviction.
            "/readyz" => {
                let health = if self.metrics.nats_connected.get() == 1 {
                    HEALTH_OK
                } else {
                    HEALTH_DEGRADED
                };
                Self::health(200, health)
            }
            "/metrics" => Self::metrics(),
            _ => Self::health(404, HEALTH_NOT_FOUND),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_bodies_keep_the_documented_json_shape() {
        // The bodies are hand-rolled `concat!` now rather than serde output, so a typo would ship a
        // malformed probe body. Pin that they still parse and carry the two fields probes match on.
        for (body, status) in [
            (HEALTH_OK, "ok"),
            (HEALTH_DEGRADED, "degraded"),
            (HEALTH_NOT_FOUND, "not_found"),
        ] {
            let v: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(v["status"], status, "body: {body}");
            assert_eq!(v["version"], env!("CARGO_PKG_VERSION"), "body: {body}");
        }
    }

    #[test]
    fn scrape_hint_covers_the_length_it_just_measured() {
        // The self-tuning hint is only useful if it never undersizes the next scrape: a hint below
        // the body it was derived from would reintroduce the realloc it exists to avoid.
        let len = AdminApp::metrics().body().len();
        let hint = LAST_SCRAPE.load(Ordering::Relaxed);
        assert!(hint >= len, "hint {hint} must cover a {len}-byte body");
        assert!(hint >= MIN_SCRAPE_BUF, "hint {hint} fell below the floor");
    }
}
