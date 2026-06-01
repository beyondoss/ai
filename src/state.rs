//! Shared gateway state.
//!
//! Only the **deny-set** is dynamic (watched from NATS, behind `ArcSwap` for lock-free reads).
//! Everything else — the signing keyring and the resolved provider registry (upstreams + pool auth
//! values) — is built once at boot from config (SSM/env), so the auth + key paths have **no runtime
//! dependency on NATS**.

use crate::config::AiConfig;
use crate::deny::DenySet;
use crate::error::{GatewayError, Result};
use crate::key::Keyring;
use crate::metrics::{Metrics, ProviderMetrics};
use crate::ratelimit::RateLimit;
use crate::route::{self, AuthScheme, Dialect, Provider};
use arc_swap::ArcSwap;
use arrayvec::ArrayString;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::warn;

/// How long a resolved upstream address is reused before re-resolving.
const DNS_TTL: Duration = Duration::from_secs(60);

/// A process-unique request id, `{instance:x}-{seq:x}`. Two `u64`s in hex (≤16 chars each) plus the
/// `-` separator never exceed 33 bytes, so it lives inline on the stack — no per-request heap
/// allocation on the admitted path (it's minted for every request, including fast rejects).
pub type RequestId = ArrayString<33>;

/// Build the resolved provider registry from the static known set + config: every known provider
/// (its authority overridable by `provider_authorities`), plus any config-only OpenAI-wire provider
/// (a `provider_authorities` entry whose name isn't known). Each provider's pool key (if any) is
/// looked up by name and its managed auth header value precomputed.
fn build_providers(config: &AiConfig, metrics: &Metrics) -> HashMap<String, Arc<Provider>> {
    // One independent breaker per provider, all built from the same config (the breaker holds
    // atomics so it can't be cloned — we mint a fresh one per provider). `None` ⇒ breaker disabled.
    let cb_config = config.circuit_breaker_config();
    let breaker = || {
        cb_config
            .clone()
            .map(crate::circuit_breaker::CircuitBreaker::new)
    };

    let mut providers = HashMap::new();
    for spec in route::KNOWN_PROVIDERS {
        let authority = config
            .provider_authorities
            .get(spec.name)
            .cloned()
            .unwrap_or_else(|| spec.authority.to_string());
        let pool_key = config.pool_keys.get(spec.name).map(|s| s.expose());
        providers.insert(
            spec.name.to_string(),
            Arc::new(Provider::resolve(
                spec.name,
                authority,
                spec.dialect,
                spec.auth,
                pool_key,
                ProviderMetrics::resolve(metrics, spec.name),
                breaker(),
            )),
        );
    }
    // Config-only providers (name not in the known set): assume OpenAI-wire (Bearer). A non-OpenAI
    // wire format would need real code, so we don't pretend to support it from config alone.
    for (name, authority) in &config.provider_authorities {
        if !providers.contains_key(name) {
            let pool_key = config.pool_keys.get(name).map(|s| s.expose());
            providers.insert(
                name.clone(),
                Arc::new(Provider::resolve(
                    name,
                    authority.clone(),
                    Dialect::OpenAI,
                    AuthScheme::Bearer,
                    pool_key,
                    ProviderMetrics::resolve(metrics, name),
                    breaker(),
                )),
            );
        }
    }
    providers
}

pub struct GatewayState {
    pub config: AiConfig,
    pub metrics: Arc<Metrics>,

    /// Trusted Ed25519 public keys by kid — from config (rotate via redeploy). Static for life.
    pub keyring: Keyring,
    /// Resolved providers by name (upstream authority/host + precomputed managed auth value). Built
    /// once at boot from `route::KNOWN_PROVIDERS` + config; the request path clones the `Arc`.
    providers: HashMap<String, Arc<Provider>>,

    /// Sparse deny-set — the ONE thing watched from NATS. Default-allow on miss; fail-open.
    pub deny: ArcSwap<DenySet>,

    /// Per-key request-rate guardrail (see `ratelimit`). `None` when `rate_limit_rps == 0`. Fixed
    /// memory regardless of tenant count, so it lives in the static state with no GC.
    pub rate_limit: Option<RateLimit>,

    /// TTL cache of resolved upstream addresses, so `upstream_peer` neither blocks on a synchronous
    /// `getaddrinfo` nor re-resolves the same provider host every request. `ArcSwap` so the common
    /// case — a cache hit, on every admitted request after warmup — is a lock-free atomic load; the
    /// only writes are the ~10 providers' entries refreshed once per `DNS_TTL`, applied via `rcu`.
    dns_cache: ArcSwap<HashMap<String, (SocketAddr, Instant)>>,

    /// Per-process instance token (8 OS-random bytes), the high half of every `request_id`.
    /// Random rather than a uuid dep, so log lines from two gateways don't collide when aggregated —
    /// and random rather than the boot wall-clock, which collides when a rapid scale-up boots several
    /// instances within the same nanosecond.
    instance_id: u64,
    /// Monotonic per-request counter, the low half of `request_id`. A relaxed `fetch_add` — the only
    /// requirement is uniqueness within the process, not cross-request ordering.
    request_seq: AtomicU64,
}

impl GatewayState {
    pub fn new(config: AiConfig, metrics: Arc<Metrics>) -> Result<Arc<Self>> {
        let keyring = config.build_keyring()?;
        // No signing keys ⇒ every `bai_…` fails verify and falls through to BYO treatment: no key
        // swap, no deny-set, no `ai.usage` billing. That's a *valid* mode (a BYO-only deployment),
        // but a far more common cause is a missing/typo'd `signing_keys` (SSM param, env) — which
        // looks healthy while silently dropping all billing. A managed deployment sets
        // `require_signing_keys = true` so this mis-deploy is a hard, visible boot failure; otherwise
        // we warn loudly and continue (BYO-only is legitimate and the test/e2e harnesses run keyless).
        if config.signing_keys.is_empty() {
            if config.require_signing_keys {
                return Err(GatewayError::Config(
                    "require_signing_keys is set but no signing_keys are configured — refusing to \
                     boot into silent BYO-only mode (no key swap, no deny-set, no billing). Check \
                     the signing_keys config / SSM param."
                        .to_string(),
                ));
            }
            warn!(
                "no signing_keys configured — all managed (bai_) traffic will be treated as BYO \
                 (no key swap, no deny-set, no billing). Expected only for a BYO-only deployment."
            );
        }
        let providers = build_providers(&config, &metrics);
        let rate_limit = RateLimit::new(config.rate_limit_rps, config.byo_rate_limit_rps);

        // 8 OS-random bytes as the instance token, so two gateways' request_ids never collide when
        // aggregated — including when a rapid scale-up boots several instances within the same
        // nanosecond (which a wall-clock token can't distinguish). If the OS RNG is somehow
        // unavailable, fall back to the boot wall-clock rather than panicking — a degraded-uniqueness
        // id beats failing to start.
        let instance = {
            let mut buf = [0u8; 8];
            match getrandom::fill(&mut buf) {
                Ok(()) => u64::from_le_bytes(buf),
                Err(_) => SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0),
            }
        };

        Ok(Arc::new(Self {
            metrics,
            keyring,
            providers,
            deny: ArcSwap::from_pointee(DenySet::new()),
            rate_limit,
            dns_cache: ArcSwap::from_pointee(HashMap::new()),
            instance_id: instance,
            request_seq: AtomicU64::new(0),
            config,
        }))
    }

    /// A process-unique request id (`{instance}-{seq}`) for log correlation and the
    /// `x-beyond-request-id` response header. Deliberately *not* a uuid: a per-process instance
    /// token (computed once at boot) plus a relaxed atomic counter is unique across the fleet, costs
    /// one `fetch_add` + a hex format into a stack buffer (no heap allocation), and needs no
    /// randomness per request.
    pub fn next_request_id(&self) -> RequestId {
        let seq = self.request_seq.fetch_add(1, Ordering::Relaxed);
        let mut id = RequestId::new();
        // Can't overflow: two `u64`s in hex + `-` is ≤33 bytes, exactly the buffer's capacity. The
        // `write!` is infallible here, but if a future format change ever exceeded the cap we'd
        // rather emit a truncated id than panic on a correlation aid — so swallow the result.
        let _ = write!(id, "{:x}-{seq:x}", self.instance_id);
        id
    }

    /// The resolved provider for `name` (the request's first path segment, or the bare-path dialect
    /// default), or `None` if no such provider is registered — which `request_filter` turns into a
    /// 404.
    pub fn provider(&self, name: &str) -> Option<&Arc<Provider>> {
        self.providers.get(name)
    }

    /// Resolve an `host:port` authority to a `SocketAddr`, cached for `DNS_TTL`. Uses
    /// `tokio::net::lookup_host` (runs `getaddrinfo` on the blocking pool — async-safe) instead of
    /// `HttpPeer::new`'s eager blocking resolve.
    pub async fn resolve(&self, authority: &str) -> Result<SocketAddr> {
        // Cache hit (the common case after warmup): a lock-free `ArcSwap` load — no mutex, no
        // syscall — so concurrent workers never serialize on a DNS lookup that's already resolved.
        if let Some((addr, at)) = self.dns_cache.load().get(authority) {
            if at.elapsed() < DNS_TTL {
                return Ok(*addr);
            }
        }
        let addr = tokio::net::lookup_host(authority)
            .await
            .map_err(|e| GatewayError::Dns(format!("{authority}: {e}")))?
            .next()
            .ok_or_else(|| GatewayError::Dns(format!("{authority}: no addresses")))?;
        // rcu the new/refreshed entry in. Two concurrent misses for the same host may both resolve
        // and both rcu; that's harmless (same answer, last writer wins) and far cheaper than holding
        // a lock across `getaddrinfo`. The clone-on-write copies a ~10-entry map — trivial, and only
        // on the rare miss/refresh path, never on a hit.
        //
        // Sweep entries that are long dead while we're already paying for the clone. The cache keys
        // are provider authorities, which come entirely from the boot-time registry (so in practice
        // the map is bounded by the provider count, not by traffic) — this sweep is belt-and-
        // suspenders against authorities ever becoming dynamic, and it's a *TTL* drop, not an
        // eviction *policy*: there's no capacity contest here, so LRU/SIEVE would be machinery for a
        // problem we don't have. We keep anything within `2 × DNS_TTL` so a still-live provider whose
        // entry just expired (and is about to be refreshed) is never dropped out from under a
        // concurrent resolve.
        let now = Instant::now();
        self.dns_cache.rcu(|cur| {
            let mut next = HashMap::clone(cur);
            next.retain(|_, (_, at)| now.duration_since(*at) < DNS_TTL * 2);
            next.insert(authority.to_string(), (addr, now));
            next
        });
        Ok(addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::AuthScheme;
    use crate::secret::Secret;

    /// One process-wide `Metrics` (it registers on the default Prometheus registry, which rejects a
    /// second registration), shared by every test that needs a `GatewayState`.
    fn test_metrics() -> Arc<Metrics> {
        use std::sync::OnceLock;
        static M: OnceLock<Arc<Metrics>> = OnceLock::new();
        M.get_or_init(|| Metrics::new().expect("register metrics once"))
            .clone()
    }

    #[test]
    fn registry_resolves_known_overrides_and_additions() {
        let config = AiConfig {
            // Override a known provider's authority + give it a pool key; add a config-only one.
            // `custom2` is a config-only provider with **no** pool key — the condition that makes a
            // managed request to it 503 (no managed auth value to swap in).
            provider_authorities: HashMap::from([
                ("openai".to_string(), "127.0.0.1:9".to_string()),
                ("custom".to_string(), "llm.internal:8443".to_string()),
                ("custom2".to_string(), "other.internal:8443".to_string()),
            ]),
            pool_keys: HashMap::from([
                ("openai".to_string(), Secret::new("sk-openai")),
                ("custom".to_string(), Secret::new("sk-custom")),
            ]),
            ..Default::default()
        };
        let providers = build_providers(&config, &test_metrics());

        // Known provider: authority overridden, pool auth precomputed in the right scheme.
        let openai = providers.get("openai").unwrap();
        assert_eq!(openai.authority, "127.0.0.1:9");
        assert_eq!(openai.auth, AuthScheme::Bearer);
        assert_eq!(
            openai.pool_auth_value.as_ref().unwrap().expose(),
            "Bearer sk-openai"
        );

        // Known provider, no override: built-in default + no pool key ⇒ no managed auth value.
        let anthropic = providers.get("anthropic").unwrap();
        assert_eq!(anthropic.authority, "api.anthropic.com:443");
        assert_eq!(anthropic.auth, AuthScheme::XApiKey);
        assert!(anthropic.pool_auth_value.is_none());

        // Config-only provider: added as OpenAI-wire (Bearer), reachable by name.
        let custom = providers.get("custom").unwrap();
        assert_eq!(custom.host, "llm.internal");
        assert_eq!(
            custom.pool_auth_value.as_ref().unwrap().expose(),
            "Bearer sk-custom"
        );

        // Config-only provider with no pool key: registered (reachable by name) but with no managed
        // auth value — this `None` is exactly what `request_filter` turns into a 503 for a managed
        // request. (BYO to it still works; it just can't serve the pooled path.)
        let custom2 = providers.get("custom2").unwrap();
        assert!(
            custom2.pool_auth_value.is_none(),
            "a provider with no configured pool key must have no managed auth value (→ 503)"
        );
    }

    #[tokio::test]
    async fn resolve_caches_hit_and_errors_on_bad_host() {
        // `resolve` is on the request hot path (every admitted request hits `upstream_peer`). Cover
        // the three outcomes: a successful resolve, a cache hit returning the same address without a
        // fresh lookup, and a lookup failure surfacing as `GatewayError::Dns` (not a panic/hang).
        let config = AiConfig::default();
        let state = GatewayState::new(config, test_metrics()).unwrap();

        // An IP literal resolves through `lookup_host` without real DNS — deterministic, offline-safe.
        let addr = state.resolve("127.0.0.1:9").await.unwrap();
        assert_eq!(addr, "127.0.0.1:9".parse().unwrap());

        // Second call is served from the TTL cache: same answer, and the entry is now present.
        assert_eq!(state.resolve("127.0.0.1:9").await.unwrap(), addr);
        assert!(state.dns_cache.load().contains_key("127.0.0.1:9"));

        // A guaranteed-NXDOMAIN host (RFC 6761 reserves `.invalid`) → a Dns error, never a panic.
        assert!(matches!(
            state.resolve("nonexistent.invalid:80").await,
            Err(GatewayError::Dns(_))
        ));
    }
}
