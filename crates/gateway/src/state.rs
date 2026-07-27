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

/// The `{instance:x}-` half of every request id: 16 hex digits plus the separator.
type InstancePrefix = ArrayString<17>;

/// Append `v` as lower-case hex, no allocation and no `core::fmt`.
///
/// `write!(.., "{v:x}")` goes through the whole formatting machinery — a `Formatter`, a vtable, and
/// padding/width logic none of which applies here — for two integers on a path that runs on every
/// request including every fast reject. A nibble loop measured 10.55 ns against 28.7 ns for the
/// `write!` form. Neither allocates.
///
/// A `try_push` that would overflow is dropped rather than panicking, matching the existing
/// preference for a truncated correlation id over a downed worker. It cannot happen: 16 hex digits
/// plus a 17-byte prefix is exactly the 33-byte capacity.
fn push_lower_hex(out: &mut RequestId, mut v: u64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    if v == 0 {
        let _ = out.try_push('0');
        return;
    }
    let mut buf = [0u8; 16];
    let mut i = buf.len();
    while v > 0 {
        i -= 1;
        buf[i] = HEX[(v & 0xf) as usize];
        v >>= 4;
    }
    for &b in &buf[i..] {
        let _ = out.try_push(b as char);
    }
}

/// Build the resolved provider registry from the static known set + config: every known provider
/// (its authority overridable by `provider_authorities`), plus any config-only provider (a
/// `provider_authorities` entry whose name isn't known), whose dialect/auth scheme come from
/// `provider_dialects`/`provider_auth_schemes` (default OpenAI/Bearer, for backward compatibility).
/// Each provider's pool key (if any) is looked up by name and its managed auth header value
/// precomputed. An unrecognized dialect/auth-scheme string is a hard boot failure (`Err`) rather than
/// a silent default — see `Dialect::parse_config`/`AuthScheme::parse_config`.
fn build_providers(config: &AiConfig, metrics: &Metrics) -> Result<HashMap<String, Arc<Provider>>> {
    // One independent breaker per provider, all built from the same config (the breaker holds
    // atomics so it can't be cloned — we mint a fresh one per provider). `None` ⇒ breaker disabled.
    let cb_config = config.circuit_breaker_config();
    let breaker = || {
        cb_config
            .clone()
            .map(crate::circuit_breaker::CircuitBreaker::new)
    };

    // `/auto` is the model-routed segment (`route::AUTO_SEGMENT`). Provider lookup runs *first* in
    // `request_filter`, so a provider registered under that name would shadow the whole feature —
    // silently, and only for requests that were meant to be model-routed. Refuse to boot instead.
    if let Some(authority) = config.provider_authorities.get(route::AUTO_SEGMENT) {
        return Err(GatewayError::Config(format!(
            "provider_authorities.{} = {authority:?} uses a reserved name: {}/… is the \
             model-routed route, which a provider of that name would shadow",
            route::AUTO_SEGMENT,
            route::AUTO_SEGMENT,
        )));
    }

    let mut providers = HashMap::new();
    for spec in route::known_providers() {
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
                spec.wire,
                spec.auth,
                pool_key,
                ProviderMetrics::resolve(metrics, spec.name),
                breaker(),
            )),
        );
    }
    // Config-only providers (name not in the known set): dialect/auth scheme come from
    // `provider_dialects`/`provider_auth_schemes` (default OpenAI/Bearer, preserved for backward
    // compatibility), so adding an Anthropic-wire vendor (MiniMax, MiniMax-CN, Kimi-Coding, …) is a
    // config line — `provider_authorities.minimax = "…"` + `provider_dialects.minimax = "anthropic"`
    // (+ `provider_auth_schemes.minimax = "x-api-key"` if the default Bearer is wrong too) — not a
    // code change. A value that doesn't parse is a boot failure, not a silent fallback: silently
    // defaulting a typo'd "anthropic" to "openai" is exactly the zero-billing bug this field exists to
    // prevent (see `usage::openai_body`'s dialect-mismatch guard for the runtime backstop).
    for (name, authority) in &config.provider_authorities {
        if !providers.contains_key(name) {
            let pool_key = config.pool_keys.get(name).map(|s| s.expose());
            let dialect = match config.provider_dialects.get(name) {
                Some(s) => Dialect::parse_config(s).ok_or_else(|| {
                    GatewayError::Config(format!(
                        "provider_dialects.{name} = {s:?} is not a recognized dialect \
                         (expected \"openai\" or \"anthropic\")"
                    ))
                })?,
                None => Dialect::OpenAi,
            };
            let auth = match config.provider_auth_schemes.get(name) {
                Some(s) => AuthScheme::parse_config(s).ok_or_else(|| {
                    GatewayError::Config(format!(
                        "provider_auth_schemes.{name} = {s:?} is not a recognized auth scheme \
                         (expected \"bearer\", \"x-api-key\", or \"api-key\")"
                    ))
                })?,
                None => AuthScheme::Bearer,
            };
            providers.insert(
                name.clone(),
                Arc::new(Provider::resolve(
                    name,
                    authority.clone(),
                    dialect,
                    auth,
                    pool_key,
                    ProviderMetrics::resolve(metrics, name),
                    breaker(),
                )),
            );
        }
    }
    Ok(providers)
}

/// Index the resolved providers by [`providers::ProviderId`], for the model-routed path's
/// per-attempt candidate lookup.
///
/// Built from the shared table rather than by walking the map, so the array can only ever hold a
/// provider under its own id. A config-added provider has no id and is deliberately absent — the
/// catalog names candidates by id, so it could never reference one.
fn index_by_id(
    resolved: &HashMap<String, Arc<Provider>>,
) -> [Option<Arc<Provider>>; providers::ProviderId::COUNT] {
    let mut by_id: [Option<Arc<Provider>>; providers::ProviderId::COUNT] = Default::default();
    for spec in route::known_providers() {
        by_id[spec.id.index()] = resolved.get(spec.name).cloned();
    }
    by_id
}

pub struct GatewayState {
    pub config: AiConfig,
    pub metrics: Arc<Metrics>,

    /// Trusted Ed25519 public keys by kid — from config (rotate via redeploy). Static for life.
    pub keyring: Keyring,
    /// Resolved providers by name (upstream authority/host + precomputed managed auth value). Built
    /// once at boot from `route::KNOWN_PROVIDERS` + config; the request path clones the `Arc`.
    providers: HashMap<String, Arc<Provider>>,
    /// The same providers, indexed by [`providers::ProviderId`] — the model-routed path switches
    /// candidates between connect attempts and holds ids, not names, so this makes that an array
    /// index instead of hashing a string on a path that can run several times per request.
    ///
    /// `None` for an id the gateway does not route to (the BYO-only rows), which is also why a
    /// config-added provider is absent: it has no `ProviderId`, and a catalog row can only name one.
    by_id: [Option<Arc<Provider>>; providers::ProviderId::COUNT],

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

    /// The per-process instance token (8 OS-random bytes) already rendered as `{:x}-` — the high
    /// half of every `request_id`. It is constant for the life of the process, so it is formatted
    /// once here rather than re-derived on a path that runs for every request.
    ///
    /// Random rather than a uuid dep, so log lines from two gateways don't collide when aggregated —
    /// and random rather than the boot wall-clock, which collides when a rapid scale-up boots several
    /// instances within the same nanosecond.
    instance_prefix: InstancePrefix,
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
        // Disabling upstream cert verification turns every provider connection into an unverified
        // channel — a transparent-MitM opening. It's legitimate *only* for the local self-signed TLS
        // mock (e2e/bench). Warn loudly at boot so it can never be a silent production misconfig (an
        // `AI_UPSTREAM_VERIFY_CERT=false` copied out of a bench env) that looks healthy.
        if config.upstream_tls && !config.upstream_verify_cert {
            warn!(
                "upstream TLS certificate verification is DISABLED — connections to providers are \
                 unauthenticated and vulnerable to interception. This is valid ONLY for a local \
                 test/bench mock; never set upstream_verify_cert=false against a real provider."
            );
        }

        let providers = build_providers(&config, &metrics)?;
        let by_id = index_by_id(&providers);
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
            by_id,
            deny: ArcSwap::from_pointee(DenySet::new()),
            rate_limit,
            dns_cache: ArcSwap::from_pointee(HashMap::new()),
            instance_prefix: {
                // Rendered once. Infallible: 16 hex digits + `-` is exactly the capacity.
                let mut p = InstancePrefix::new();
                let _ = write!(p, "{instance:x}-");
                p
            },
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
        // The instance half is boot-constant, so it is copied rather than re-formatted; only the
        // counter is rendered, and by a nibble loop rather than `core::fmt`. Can't overflow: 17-byte
        // prefix + ≤16 hex digits is exactly the buffer's capacity. A `try_push` that somehow did
        // overflow is dropped — a truncated correlation id beats panicking a worker over one.
        let _ = id.try_push_str(&self.instance_prefix);
        push_lower_hex(&mut id, seq);
        id
    }

    /// The resolved provider for `name` (the request's first path segment, or the bare-path dialect
    /// default), or `None` if no such provider is registered — which `request_filter` turns into a
    /// 404.
    pub fn provider(&self, name: &str) -> Option<&Arc<Provider>> {
        self.providers.get(name)
    }

    /// The resolved provider for a catalog candidate's id, or `None` if this gateway does not route
    /// to it. One array index — the model-routed path calls this once per connect attempt.
    pub fn provider_by_id(&self, id: providers::ProviderId) -> Option<&Arc<Provider>> {
        self.by_id[id.index()].as_ref()
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

    /// Provider lookup runs before the model-routed segment is even considered, so registering a
    /// provider named `auto` from config would silently disable model routing. Boot must refuse.
    #[test]
    fn reserved_auto_provider_name_fails_boot() {
        let config = AiConfig {
            provider_authorities: HashMap::from([(
                route::AUTO_SEGMENT.to_string(),
                "llm.internal:8443".to_string(),
            )]),
            ..Default::default()
        };
        let err = build_providers(&config, &test_metrics())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains(route::AUTO_SEGMENT) && err.contains("reserved"),
            "boot must fail naming the reserved segment, got {err:?}",
        );
    }

    /// The id-keyed array is what the model-routed path indexes per connect attempt. It must agree
    /// with the by-name map for every known provider, and must have no entry for a config-only one
    /// (which has no `ProviderId` for a catalog row to name).
    #[test]
    fn provider_by_id_resolves_known_rows_and_skips_config_only_ones() {
        let config = AiConfig {
            provider_authorities: HashMap::from([(
                "custom".to_string(),
                "llm.internal:8443".to_string(),
            )]),
            ..Default::default()
        };
        let resolved = build_providers(&config, &test_metrics()).unwrap();
        let by_id = index_by_id(&resolved);

        for spec in route::known_providers() {
            let via_id = by_id[spec.id.index()].as_ref();
            assert!(via_id.is_some(), "{} missing from the id index", spec.name);
            assert_eq!(
                via_id.map(|p| p.name.as_str()),
                Some(spec.name),
                "{} is indexed under another provider's id",
                spec.name,
            );
        }

        assert!(
            resolved.contains_key("custom"),
            "config-only provider is still reachable by name",
        );
        let indexed = by_id.iter().flatten().count();
        assert_eq!(
            indexed,
            route::known_providers().count(),
            "the id index must hold exactly the known rows — no config-only providers",
        );
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
        let providers = build_providers(&config, &test_metrics()).unwrap();

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

    #[test]
    fn config_added_provider_honors_dialect_and_auth_scheme_overrides() {
        // Task #30: a config-added Anthropic-wire vendor (MiniMax, MiniMax-CN, Kimi-Coding in the
        // real pi fleet) must be reachable with the correct dialect + auth scheme from config alone —
        // no code change. Before this fix every config-added provider was hardcoded OpenAI+Bearer.
        let config = AiConfig {
            provider_authorities: HashMap::from([(
                "minimax".to_string(),
                "api.minimax.io:443".to_string(),
            )]),
            provider_dialects: HashMap::from([("minimax".to_string(), "Anthropic".to_string())]),
            provider_auth_schemes: HashMap::from([(
                "minimax".to_string(),
                "x-api-key".to_string(),
            )]),
            pool_keys: HashMap::from([("minimax".to_string(), Secret::new("mm-key"))]),
            ..Default::default()
        };
        let providers = build_providers(&config, &test_metrics()).unwrap();
        let minimax = providers.get("minimax").unwrap();
        assert_eq!(minimax.dialect, Dialect::Anthropic);
        assert_eq!(minimax.auth, AuthScheme::XApiKey);
        assert_eq!(minimax.pool_auth_value.as_ref().unwrap().expose(), "mm-key");

        // Unset dialect/auth_scheme still defaults to OpenAI/Bearer (backward compatible).
        let default_config = AiConfig {
            provider_authorities: HashMap::from([(
                "custom".to_string(),
                "llm.internal:8443".to_string(),
            )]),
            ..Default::default()
        };
        let providers = build_providers(&default_config, &test_metrics()).unwrap();
        let custom = providers.get("custom").unwrap();
        assert_eq!(custom.dialect, Dialect::OpenAi);
        assert_eq!(custom.auth, AuthScheme::Bearer);
    }

    #[test]
    fn config_added_provider_rejects_unrecognized_dialect_or_auth_scheme() {
        // A typo'd dialect ("anthropc") must fail boot loudly — silently falling back to OpenAI-wire
        // is exactly the zero-billing bug this config field exists to prevent.
        let bad_dialect = AiConfig {
            provider_authorities: HashMap::from([(
                "minimax".to_string(),
                "api.minimax.io:443".to_string(),
            )]),
            provider_dialects: HashMap::from([("minimax".to_string(), "anthropc".to_string())]),
            ..Default::default()
        };
        assert!(matches!(
            build_providers(&bad_dialect, &test_metrics()),
            Err(GatewayError::Config(_))
        ));

        let bad_auth = AiConfig {
            provider_authorities: HashMap::from([(
                "minimax".to_string(),
                "api.minimax.io:443".to_string(),
            )]),
            provider_auth_schemes: HashMap::from([("minimax".to_string(), "bogus".to_string())]),
            ..Default::default()
        };
        assert!(matches!(
            build_providers(&bad_auth, &test_metrics()),
            Err(GatewayError::Config(_))
        ));
    }

    #[test]
    fn request_ids_match_the_format_they_replaced() {
        // The id is what an oncall greps and what a client quotes back, so the hand-rolled hex must
        // render exactly what `write!("{:x}-{:x}")` did — including the boundaries a nibble loop is
        // most likely to get wrong.
        for instance in [0u64, 1, 0xf, 0x10, 0xdead_beef_cafe_f00d, u64::MAX] {
            let mut prefix = InstancePrefix::new();
            let _ = write!(prefix, "{instance:x}-");
            for seq in [0u64, 1, 0xf, 0x10, 0xff, 12345, u64::MAX] {
                let mut got = RequestId::new();
                let _ = got.try_push_str(&prefix);
                push_lower_hex(&mut got, seq);
                assert_eq!(
                    got.as_str(),
                    format!("{instance:x}-{seq:x}"),
                    "instance={instance:x} seq={seq:x}"
                );
            }
        }
    }

    #[test]
    fn request_ids_are_unique_and_fit_the_buffer() {
        let state = GatewayState::new(AiConfig::default(), test_metrics()).unwrap();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10_000 {
            let id = state.next_request_id();
            // The widest possible id is 16 hex + `-` + 16 hex; the buffer is exactly that, so an
            // id that had been truncated would show up as a short/duplicated value here.
            assert!(id.len() <= 33);
            assert!(seen.insert(id), "request id repeated");
        }
        // All ids share the one boot-constant instance prefix.
        let prefix = state.instance_prefix.as_str().to_string();
        assert!(state.next_request_id().starts_with(&prefix));
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
