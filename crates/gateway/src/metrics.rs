//! Prometheus metrics (PATTERNS.md: `Arc<Metrics>`).
//!
//! Registered on the **default** registry so Pingora's built-in `prometheus_http_service`
//! exposes them with no extra wiring. `Metrics::new` is called exactly once (in `main`).

use prometheus::{
    HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts, default_registry,
};
use std::sync::Arc;

pub struct Metrics {
    pub requests_total: IntCounter,
    /// Labeled by reason ("auth", "deny_spend", "deny_fraud") so we can see *why* we rejected.
    pub rejections_total: IntCounterVec,
    /// Upstream responses by provider + status class ("2xx"/"4xx"/"5xx"). A provider degrading
    /// (429/5xx) is otherwise invisible until it surfaces as latency or missing usage events —
    /// this is the per-provider error-rate signal an oncall pages on.
    pub upstream_responses_total: IntCounterVec,
    /// Upstream **connect** retries by provider (see `proxy::fail_to_connect`). A partially-down
    /// provider TCP layer (or an egress-IP ban) silently retries up to `MAX_CONNECT_RETRIES` times
    /// per request; without this, the extra latency looks like a slow provider, not a connect
    /// problem. Pairs with a `warn!` on the same path so the dashboard spike has a log to grep.
    pub connect_retries_total: IntCounterVec,
    /// Labeled by kind: input|output|cache_read|cache_write. Cache tokens are also in the `ai.usage`
    /// billing log, but that ships with lag — the Prometheus counter is the alerting surface for
    /// "cache hit rate fell off a cliff after a deploy" (cache write ≈ 3× input, cache read ≈ 0.1×,
    /// so a regression is a real cost event, not just a latency one).
    pub tokens_total: IntCounterVec,
    /// Labeled by provider: TTFT varies by an order of magnitude across providers (Groq/Cerebras
    /// <100ms vs. a large Anthropic/xAI model at seconds), so an unlabeled histogram can't tell you
    /// *which* provider's first-token time regressed.
    pub ttft_seconds: HistogramVec,
    /// Labeled by provider, same rationale as `ttft_seconds`: full-request duration is dominated by
    /// the model's generation time, which is per-provider.
    pub upstream_latency_seconds: HistogramVec,
    pub active_streams: IntGauge,
    /// Total in-flight requests (streaming + non-streaming), incremented once a request is admitted
    /// in `request_filter` and decremented in `logging`. `active_streams` only covers SSE; under a
    /// burst or a stalled upstream this is what distinguishes "high rps, fast upstreams" from
    /// "connections piling up" — the difference between a perf blip and a connection-exhaustion
    /// incident.
    pub requests_in_flight: IntGauge,
    /// Current deny-set cardinality (denied tenants). The set is `O(denied)` and fed from NATS; a
    /// fraud event or a control-plane bug that mass-denies tenants would otherwise grow it invisibly
    /// until it shows up as memory pressure. Updated on every seed and every applied delta.
    pub deny_set_size: IntGauge,
    /// NATS connectivity for the deny-set watcher (1 = connected, 0 = disconnected). The gateway is
    /// fail-open — it serves on the last-known set when NATS is down — so staleness is otherwise
    /// silent; this is the metric to alert "deny-set has been stale for >N minutes" on.
    pub nats_connected: IntGauge,
}

/// TTFT buckets (seconds). Tuned for LLM latency: sub-second prompts up through the multi-second
/// first-token times of large models. The default prometheus buckets top out at 10s, but TTFT for a
/// busy model can exceed that, so the tail goes to 30s.
const TTFT_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

/// Full-request duration buckets (seconds). A streaming completion runs far longer than the
/// default 10s ceiling (`read_timeout_secs` defaults to 600), so the tail reaches 300s — without
/// these, every long stream lands in `+Inf` and the p99/p999 tail is unrecoverable.
const LATENCY_BUCKETS: &[f64] = &[
    0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

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
        let upstream_responses_total = IntCounterVec::new(
            Opts::new(
                "ai_upstream_responses_total",
                "Upstream responses by provider and status class",
            ),
            &["provider", "status"],
        )?;
        let connect_retries_total = IntCounterVec::new(
            Opts::new(
                "ai_connect_retries_total",
                "Upstream connect retries by provider",
            ),
            &["provider"],
        )?;
        let tokens_total =
            IntCounterVec::new(Opts::new("ai_tokens_total", "Tokens metered"), &["kind"])?;
        let ttft_seconds = HistogramVec::new(
            HistogramOpts::new("ai_ttft_seconds", "Time to first byte from upstream")
                .buckets(TTFT_BUCKETS.to_vec()),
            &["provider"],
        )?;
        let upstream_latency_seconds = HistogramVec::new(
            HistogramOpts::new(
                "ai_upstream_latency_seconds",
                "Full upstream request duration",
            )
            .buckets(LATENCY_BUCKETS.to_vec()),
            &["provider"],
        )?;
        let active_streams = IntGauge::with_opts(Opts::new(
            "ai_active_streams",
            "In-flight streaming responses",
        ))?;
        let requests_in_flight = IntGauge::with_opts(Opts::new(
            "ai_requests_in_flight",
            "In-flight requests (streaming + non-streaming)",
        ))?;
        let deny_set_size =
            IntGauge::with_opts(Opts::new("ai_deny_set_size", "Currently denied tenants"))?;
        let nats_connected = IntGauge::with_opts(Opts::new(
            "ai_nats_connected",
            "Deny-set watcher NATS connectivity (1=connected, 0=disconnected)",
        ))?;

        r.register(Box::new(requests_total.clone()))?;
        r.register(Box::new(rejections_total.clone()))?;
        r.register(Box::new(upstream_responses_total.clone()))?;
        r.register(Box::new(connect_retries_total.clone()))?;
        r.register(Box::new(tokens_total.clone()))?;
        r.register(Box::new(ttft_seconds.clone()))?;
        r.register(Box::new(upstream_latency_seconds.clone()))?;
        r.register(Box::new(active_streams.clone()))?;
        r.register(Box::new(requests_in_flight.clone()))?;
        r.register(Box::new(deny_set_size.clone()))?;
        r.register(Box::new(nats_connected.clone()))?;

        Ok(Arc::new(Self {
            requests_total,
            rejections_total,
            upstream_responses_total,
            connect_retries_total,
            tokens_total,
            ttft_seconds,
            upstream_latency_seconds,
            active_streams,
            requests_in_flight,
            deny_set_size,
            nats_connected,
        }))
    }
}
