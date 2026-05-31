//! Prometheus metrics (PATTERNS.md: `Arc<Metrics>`).
//!
//! Registered on the **default** registry so Pingora's built-in `prometheus_http_service`
//! exposes them with no extra wiring. `Metrics::new` is called exactly once (in `main`).

use prometheus::{
    Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, default_registry,
};
use std::sync::Arc;

pub struct Metrics {
    pub requests_total: IntCounter,
    /// Labeled by reason ("auth", "deny_spend", "deny_fraud") so we can see *why* we rejected.
    pub rejections_total: IntCounterVec,
    /// Labeled by kind: input|output.
    pub tokens_total: IntCounterVec,
    pub ttft_seconds: Histogram,
    pub upstream_latency_seconds: Histogram,
    pub active_streams: IntGauge,
}

impl Metrics {
    /// Build and register every metric on the default registry. Fallible: registering a name that
    /// already exists (a second `Metrics::new()` against the process-wide default registry) returns
    /// `AlreadyRegisteredError` rather than panicking, so a double-init surfaces as an error the
    /// caller can report instead of crashing the process.
    pub fn new() -> prometheus::Result<Arc<Self>> {
        let r = default_registry();

        let requests_total =
            IntCounter::with_opts(Opts::new("ai_requests_total", "Total requests handled"))?;
        let rejections_total = IntCounterVec::new(
            Opts::new("ai_rejections_total", "Requests rejected before upstream"),
            &["reason"],
        )?;
        let tokens_total =
            IntCounterVec::new(Opts::new("ai_tokens_total", "Tokens metered"), &["kind"])?;
        let ttft_seconds = Histogram::with_opts(HistogramOpts::new(
            "ai_ttft_seconds",
            "Time to first byte from upstream",
        ))?;
        let upstream_latency_seconds = Histogram::with_opts(HistogramOpts::new(
            "ai_upstream_latency_seconds",
            "Full upstream request duration",
        ))?;
        let active_streams = IntGauge::with_opts(Opts::new(
            "ai_active_streams",
            "In-flight streaming responses",
        ))?;

        r.register(Box::new(requests_total.clone()))?;
        r.register(Box::new(rejections_total.clone()))?;
        r.register(Box::new(tokens_total.clone()))?;
        r.register(Box::new(ttft_seconds.clone()))?;
        r.register(Box::new(upstream_latency_seconds.clone()))?;
        r.register(Box::new(active_streams.clone()))?;

        Ok(Arc::new(Self {
            requests_total,
            rejections_total,
            tokens_total,
            ttft_seconds,
            upstream_latency_seconds,
            active_streams,
        }))
    }
}
