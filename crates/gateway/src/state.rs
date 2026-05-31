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
use crate::metrics::Metrics;
use crate::ratelimit::RateLimit;
use crate::route::{self, AuthScheme, Provider};
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a resolved upstream address is reused before re-resolving.
const DNS_TTL: Duration = Duration::from_secs(60);

/// Build the resolved provider registry from the static known set + config: every known provider
/// (its authority overridable by `provider_authorities`), plus any config-only OpenAI-wire provider
/// (a `provider_authorities` entry whose name isn't known). Each provider's pool key (if any) is
/// looked up by name and its managed auth header value precomputed.
fn build_providers(config: &AiConfig) -> HashMap<String, Arc<Provider>> {
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
                spec.base_path,
                spec.auth,
                pool_key,
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
                    route::CLIENT_PREFIX,
                    AuthScheme::Bearer,
                    pool_key,
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
    /// `getaddrinfo` nor re-resolves the same provider host every request.
    dns_cache: Mutex<HashMap<String, (SocketAddr, Instant)>>,
}

impl GatewayState {
    pub fn new(config: AiConfig, metrics: Arc<Metrics>) -> Result<Arc<Self>> {
        let keyring = config.build_keyring()?;
        let providers = build_providers(&config);
        let rate_limit = RateLimit::new(config.rate_limit_rps);

        Ok(Arc::new(Self {
            metrics,
            keyring,
            providers,
            deny: ArcSwap::from_pointee(DenySet::new()),
            rate_limit,
            dns_cache: Mutex::new(HashMap::new()),
            config,
        }))
    }

    /// The resolved provider for `name` (`x-beyond-provider` value or dialect default), or `None`
    /// if no such provider is registered.
    pub fn provider(&self, name: &str) -> Option<&Arc<Provider>> {
        self.providers.get(name)
    }

    /// Resolve an `host:port` authority to a `SocketAddr`, cached for `DNS_TTL`. Uses
    /// `tokio::net::lookup_host` (runs `getaddrinfo` on the blocking pool — async-safe) instead of
    /// `HttpPeer::new`'s eager blocking resolve.
    pub async fn resolve(&self, authority: &str) -> Result<SocketAddr> {
        // Scope the guard so the `std::sync::Mutex` is provably released before the `.await` below —
        // a `std` guard is not `Send` and must never be held across an await. The explicit block
        // makes that invariant local and obvious (a stray log/borrow added before the await would
        // otherwise either deadlock or fail to compile). A miss may let two concurrent callers both
        // resolve; that's harmless (same answer, last writer wins) and not worth a lock across DNS.
        {
            // Recover from a poisoned lock (a prior holder panicked) rather than propagating the
            // panic: the cache holds only transient `SocketAddr` entries, so a poisoned-but-readable
            // map is safe to use. Without this, one panic would wedge every later DNS lookup.
            let cache = self.dns_cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some((addr, at)) = cache.get(authority) {
                if at.elapsed() < DNS_TTL {
                    return Ok(*addr);
                }
            }
        }
        let addr = tokio::net::lookup_host(authority)
            .await
            .map_err(|e| GatewayError::Dns(format!("{authority}: {e}")))?
            .next()
            .ok_or_else(|| GatewayError::Dns(format!("{authority}: no addresses")))?;
        self.dns_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(authority.to_string(), (addr, Instant::now()));
        Ok(addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::AuthScheme;
    use crate::secret::Secret;

    #[test]
    fn registry_resolves_known_overrides_and_additions() {
        let config = AiConfig {
            // Override a known provider's authority + give it a pool key; add a config-only one.
            provider_authorities: HashMap::from([
                ("openai".to_string(), "127.0.0.1:9".to_string()),
                ("custom".to_string(), "llm.internal:8443".to_string()),
            ]),
            pool_keys: HashMap::from([
                ("openai".to_string(), Secret::new("sk-openai")),
                ("custom".to_string(), Secret::new("sk-custom")),
            ]),
            ..Default::default()
        };
        let providers = build_providers(&config);

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
    }
}
