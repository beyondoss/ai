//! Prometheus metrics (PATTERNS.md: `Arc<Metrics>`).
//!
//! Registered on the **default** registry so Pingora's built-in `prometheus_http_service`
//! exposes them with no extra wiring. `Metrics::new` is called exactly once (in `main`).

use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts,
    default_registry,
};
use std::sync::Arc;

/// Why a request was rejected — the closed label set of `ai_rejections_total`.
///
/// An enum rather than bare `&str`s so the children can be resolved once at boot (see
/// [`Metrics::rejection`]). This is the flood path by construction: `proxy`'s rate guardrail exists
/// precisely so a leaked or forged credential can be shed cheaply, so it runs at full request rate
/// exactly when the gateway is least able to afford a map lookup and a lock on every hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// Virtual key failed Ed25519 verification.
    Auth,
    /// Deny-set hit, reason "spend".
    DenySpend,
    /// Deny-set hit, reason "fraud".
    DenyFraud,
    /// Deny-set hit with a reason string this build doesn't recognize — kept distinct from
    /// `DenyFraud` so a control-plane deploy running ahead of a gateway deploy shows up as the
    /// deployment-coordination issue it is instead of spiking the fraud signal.
    DenyUnknown,
    /// Provider's circuit breaker is open.
    CircuitOpen,
    /// Streamed request body crossed `MAX_REQUEST_BODY`.
    BodyTooLarge,
    /// Per-credential rate ceiling.
    RateLimit,
    /// Aggregate BYO rate ceiling.
    RateLimitByoGlobal,
}

impl Rejection {
    /// Every variant, in `as_index` order. The array in `Metrics` is built from this, so adding a
    /// variant without adding it here fails the exhaustive `match` in `as_index`.
    const ALL: [Rejection; 8] = [
        Rejection::Auth,
        Rejection::DenySpend,
        Rejection::DenyFraud,
        Rejection::DenyUnknown,
        Rejection::CircuitOpen,
        Rejection::BodyTooLarge,
        Rejection::RateLimit,
        Rejection::RateLimitByoGlobal,
    ];

    /// The `reason=` label value. `RateLimit` keeps the original `"rate_limit"` string so existing
    /// dashboards and alerts are unbroken.
    pub fn label(self) -> &'static str {
        match self {
            Rejection::Auth => "auth",
            Rejection::DenySpend => "deny_spend",
            Rejection::DenyFraud => "deny_fraud",
            Rejection::DenyUnknown => "deny_unknown",
            Rejection::CircuitOpen => "circuit_open",
            Rejection::BodyTooLarge => "body_too_large",
            Rejection::RateLimit => "rate_limit",
            Rejection::RateLimitByoGlobal => "rate_limit_byo_global",
        }
    }

    fn as_index(self) -> usize {
        match self {
            Rejection::Auth => 0,
            Rejection::DenySpend => 1,
            Rejection::DenyFraud => 2,
            Rejection::DenyUnknown => 3,
            Rejection::CircuitOpen => 4,
            Rejection::BodyTooLarge => 5,
            Rejection::RateLimit => 6,
            Rejection::RateLimitByoGlobal => 7,
        }
    }
}

pub struct Metrics {
    pub requests_total: IntCounter,
    /// Labeled by reason ("auth", "deny_spend", "deny_fraud") so we can see *why* we rejected.
    pub rejections_total: IntCounterVec,
    /// The `rejections_total` children, resolved once at boot — same rationale as `tokens_*` below,
    /// and for the same reason it matters more here: `with_label_values` is an FNV hash of the label
    /// bytes plus a `RwLock` read, a `HashMap` lookup, an `Arc` clone and an `Arc` drop, and this
    /// path fires at full request rate under a credential-stuffing flood. Measured 14.3 ns vs 1.3 ns
    /// single-threaded, and 808 ns vs 151 ns with 16 threads contending the same lock.
    /// Indexed by [`Rejection::as_index`]; read it through [`Metrics::rejection`].
    rejections: [IntCounter; 8],
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
    /// Managed `2xx` responses that carried no parseable usage block. Such a response still emits a
    /// zero-token `ai.usage` row, which is indistinguishable downstream from a (non-existent)
    /// legitimate zero-token generation — so a provider changing its usage wire shape would silently
    /// zero out billing. This counter (paired with a `warn!`) is the alerting surface for that.
    pub usage_parse_errors_total: IntCounter,
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
        // Resolve all eight children up front. Also means every reason is present in a scrape from
        // process start, so a dashboard shows an explicit 0 rather than a missing series.
        let rejections = Rejection::ALL.map(|r| rejections_total.with_label_values(&[r.label()]));
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
        let usage_parse_errors_total = IntCounter::with_opts(Opts::new(
            "ai_usage_parse_errors_total",
            "Managed 2xx responses with no parseable usage (emitted as a zero-token billing row)",
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
        r.register(Box::new(usage_parse_errors_total.clone()))?;

        Ok(Arc::new(Self {
            requests_total,
            rejections_total,
            rejections,
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
            usage_parse_errors_total,
        }))
    }

    /// The pre-resolved `ai_rejections_total` child for `reason` — a direct `fetch_add`, with no
    /// label hash, no registry lock, and no `Arc` churn on a path that runs at full request rate
    /// under a flood.
    #[inline]
    pub fn rejection(&self, reason: Rejection) -> &IntCounter {
        &self.rejections[reason.as_index()]
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
