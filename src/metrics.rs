//! Prometheus metrics (PATTERNS.md: `Arc<Metrics>`).
//!
//! Registered on the **default** registry so Pingora's built-in `prometheus_http_service`
//! exposes them with no extra wiring. `Metrics::new` is called exactly once (in `main`).

use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts,
    default_registry,
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
    /// The four `tokens_total` children, resolved once at boot. The label set (`input`/`output`/
    /// `cache_read`/`cache_write`) is fixed and known at compile time, so we pay the
    /// `with_label_values` map lookup once here instead of four times per metered response.
    pub tokens_input: IntCounter,
    pub tokens_output: IntCounter,
    pub tokens_cache_read: IntCounter,
    pub tokens_cache_write: IntCounter,
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
        // Resolve the fixed-label children once. Created against the (about-to-be-registered) vec, so
        // they export normally; the hot path then bumps a direct handle, no per-call label lookup.
        let tokens_input = tokens_total.with_label_values(&["input"]);
        let tokens_output = tokens_total.with_label_values(&["output"]);
        let tokens_cache_read = tokens_total.with_label_values(&["cache_read"]);
        let tokens_cache_write = tokens_total.with_label_values(&["cache_write"]);
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
            tokens_input,
            tokens_output,
            tokens_cache_read,
            tokens_cache_write,
            ttft_seconds,
            upstream_latency_seconds,
            active_streams,
            requests_in_flight,
            deny_set_size,
            nats_connected,
        }))
    }
}

/// Per-provider metric handles, resolved once at boot and held on the [`Provider`](crate::route::Provider).
///
/// Every per-provider metric (`ttft_seconds`, `upstream_latency_seconds`, `upstream_responses_total`,
/// `connect_retries_total`) is keyed on the provider name — a label known at boot from the provider
/// registry. Resolving the child handles here lets the response path bump a direct counter/histogram
/// instead of doing a string-keyed `with_label_values` map lookup on every response.
pub struct ProviderMetrics {
    pub ttft_seconds: Histogram,
    pub upstream_latency_seconds: Histogram,
    pub connect_retries_total: IntCounter,
    /// Responses by status class, indexed `[1xx, 2xx, 3xx, 4xx, 5xx]` (see [`Self::record_response`]).
    responses: [IntCounter; 5],
}

impl ProviderMetrics {
    /// Resolve the child handles for `provider` from the shared label vecs. Called once per provider
    /// at boot (see `state::build_providers`).
    pub fn resolve(m: &Metrics, provider: &str) -> Self {
        ProviderMetrics {
            ttft_seconds: m.ttft_seconds.with_label_values(&[provider]),
            upstream_latency_seconds: m.upstream_latency_seconds.with_label_values(&[provider]),
            connect_retries_total: m.connect_retries_total.with_label_values(&[provider]),
            responses: [
                m.upstream_responses_total
                    .with_label_values(&[provider, "1xx"]),
                m.upstream_responses_total
                    .with_label_values(&[provider, "2xx"]),
                m.upstream_responses_total
                    .with_label_values(&[provider, "3xx"]),
                m.upstream_responses_total
                    .with_label_values(&[provider, "4xx"]),
                m.upstream_responses_total
                    .with_label_values(&[provider, "5xx"]),
            ],
        }
    }

    /// Count one upstream response, bucketed by status class (`1xx`/`2xx`/`3xx`/`4xx`/`5xx`).
    /// A `1xx` (e.g. `100 Continue`, `101 Switching Protocols`) gets its own bucket rather than
    /// falling through to `5xx` — providers don't normally emit it, but a misbucketed informational
    /// status would otherwise read as a phantom upstream-error spike on the dashboard.
    pub fn record_response(&self, status: u16) {
        let idx = match status {
            100..=199 => 0,
            200..=299 => 1,
            300..=399 => 2,
            400..=499 => 3,
            _ => 4,
        };
        self.responses[idx].inc();
    }

    /// Standalone, **unregistered** handles for tests that build a `Provider` without a live registry.
    #[cfg(test)]
    pub fn disconnected() -> Self {
        let counter = || IntCounter::new("t", "t").expect("valid counter opts");
        let hist =
            || Histogram::with_opts(HistogramOpts::new("t", "t")).expect("valid histogram opts");
        ProviderMetrics {
            ttft_seconds: hist(),
            upstream_latency_seconds: hist(),
            connect_retries_total: counter(),
            responses: [counter(), counter(), counter(), counter(), counter()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_response_buckets_by_status_class() {
        // Lock the index mapping: a 1xx must land in its own bucket, never the 5xx fallback (which
        // would read as a phantom upstream-error spike on the provider dashboard).
        let pm = ProviderMetrics::disconnected();
        pm.record_response(100); // 1xx
        pm.record_response(204); // 2xx
        pm.record_response(301); // 3xx
        pm.record_response(404); // 4xx
        pm.record_response(503); // 5xx
        for (idx, status) in [100u16, 204, 301, 404, 503].iter().enumerate() {
            assert_eq!(
                pm.responses[idx].get(),
                1,
                "status {status} landed in the wrong class bucket"
            );
        }
    }
}
