//! Shared gateway state.
//!
//! The deny-set, capture-set, and allowance-set are dynamic (watched from NATS, behind `ArcSwap`
//! for lock-free reads). Everything else — the signing keyring and the resolved provider registry
//! (upstreams + pool auth values) — is built once at boot from config (SSM/env), so the auth +
//! key paths have **no runtime dependency on NATS**. Allowance is fail-closed until its watcher
//! stores a snapshot (empty = remaining-ok); deny is fail-open on the same unread store.

use crate::allowance::AllowanceSet;
use crate::cache::{self, ResponseCache};
use crate::capture::{CaptureRule, CaptureSet};
use crate::concurrency::{BodyBudget, TenantSlots};
use crate::config::AiConfig;
use crate::deny::DenySet;
use crate::error::{GatewayError, Result};
use crate::key::Keyring;
use crate::metrics::{Metrics, ProviderMetrics};
use crate::ratelimit::RateLimit;
use crate::route::{self, AuthScheme, Dialect, Provider};
use crate::smart;
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

/// A lookup that has not answered in this long has failed. `getaddrinfo` has no deadline of its
/// own, and a resolver that hangs would otherwise stall every request to that provider.
const DNS_TIMEOUT: Duration = Duration::from_secs(5);

/// When re-resolution fails, the last good answer is served for up to this long after it was
/// resolved (serve-stale): a resolver outage should not take down providers whose addresses have
/// not changed.
const DNS_STALE_MAX: Duration = Duration::from_secs(600);

/// While serving stale, how long before the next re-resolution attempt, so an outage does not
/// cost every request a lookup timeout.
const DNS_RETRY: Duration = Duration::from_secs(5);

/// One cached resolution: every address the lookup returned, in its order.
#[derive(Clone)]
struct DnsEntry {
    addrs: Arc<[SocketAddr]>,
    /// When the lookup that produced `addrs` succeeded. Bounds serve-stale.
    resolved_at: Instant,
    /// When to look the name up again.
    refresh_at: Instant,
}

/// A lookup's answer as its waiters see it: `None` until it is known (or `DNS_TIMEOUT` passed).
type DnsAnswer = Option<std::result::Result<Arc<[SocketAddr]>, String>>;

/// The DNS cache, and the lookups in flight for it (D89).
#[derive(Default)]
struct DnsCache {
    entries: ArcSwap<HashMap<String, DnsEntry>>,
    /// The one lookup running per authority (single flight). Its answer is published on the channel
    /// when it lands, or as a timeout after `DNS_TIMEOUT`; the entry stays until the lookup itself
    /// returns, so a `getaddrinfo` hung on the blocking pool is joined, never repeated. At most one
    /// blocking lookup per provider authority is ever outstanding.
    inflight: std::sync::Mutex<HashMap<String, tokio::sync::watch::Receiver<DnsAnswer>>>,
}

impl DnsCache {
    /// Join the lookup in flight for `authority`, or start one in the background.
    fn lookup_once<F, Fut>(
        self: &Arc<Self>,
        authority: &str,
        lookup: F,
    ) -> tokio::sync::watch::Receiver<DnsAnswer>
    where
        F: FnOnce(String) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = std::io::Result<Vec<SocketAddr>>> + Send + 'static,
    {
        let mut inflight = self
            .inflight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(answer) = inflight.get(authority) {
            return answer.clone();
        }
        let (tx, rx) = tokio::sync::watch::channel(None);
        inflight.insert(authority.to_owned(), rx.clone());
        drop(inflight);
        let (dns, authority) = (Arc::clone(self), authority.to_owned());
        tokio::spawn(async move { dns.run_lookup(authority, lookup, tx).await });
        rx
    }

    async fn run_lookup<F, Fut>(
        self: Arc<Self>,
        authority: String,
        lookup: F,
        tx: tokio::sync::watch::Sender<DnsAnswer>,
    ) where
        F: FnOnce(String) -> Fut,
        Fut: std::future::Future<Output = std::io::Result<Vec<SocketAddr>>>,
    {
        /// Ends the flight however the task ends (a panicking lookup included), so the next
        /// request past `refresh_at` starts a new one.
        struct Landed<'a>(&'a DnsCache, &'a str);
        impl Drop for Landed<'_> {
            fn drop(&mut self) {
                self.0
                    .inflight
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(self.1);
            }
        }
        let _landed = Landed(&self, &authority);
        let pending = lookup(authority.clone());
        tokio::pin!(pending);
        let looked_up = match tokio::time::timeout(DNS_TIMEOUT, &mut pending).await {
            Ok(r) => r.map_err(|e| format!("{authority}: {e}")),
            Err(_) => {
                let timed_out = format!(
                    "{authority}: lookup timed out after {}s",
                    DNS_TIMEOUT.as_secs()
                );
                tx.send_replace(Some(self.settle(&authority, Err(timed_out))));
                // Still in flight: wait the hung lookup out rather than start another beside it.
                // A late answer still fills the cache.
                if let Ok(addrs) = pending.await {
                    let _ = self.settle(&authority, Ok(addrs));
                }
                return;
            }
        };
        tx.send_replace(Some(self.settle(&authority, looked_up)));
    }

    /// Store a lookup's outcome and return the answer its waiters get. A failure keeps a still
    /// servable entry (serve-stale), looked up again in `DNS_RETRY` rather than on every request.
    fn settle(
        &self,
        authority: &str,
        looked_up: std::result::Result<Vec<SocketAddr>, String>,
    ) -> std::result::Result<Arc<[SocketAddr]>, String> {
        let now = Instant::now();
        let e = match looked_up {
            Ok(addrs) if !addrs.is_empty() => {
                let addrs: Arc<[SocketAddr]> = addrs.into();
                let entry = DnsEntry {
                    addrs: addrs.clone(),
                    resolved_at: now,
                    refresh_at: now + DNS_TTL,
                };
                // Sweep entries that are long dead while we're already paying for the clone.
                // The keys are provider authorities from the boot-time registry, so the map is
                // bounded by the provider count; this is belt-and-suspenders, a TTL drop, not an
                // eviction policy. Anything still servable as stale is kept.
                self.entries.rcu(|cur| {
                    let mut next = HashMap::clone(cur);
                    next.retain(|_, e| now.duration_since(e.resolved_at) < DNS_STALE_MAX);
                    next.insert(authority.to_string(), entry.clone());
                    next
                });
                return Ok(addrs);
            }
            Ok(_) => format!("{authority}: no addresses"),
            Err(e) => e,
        };
        if let Some(entry) = self.entries.load().get(authority)
            && now.duration_since(entry.resolved_at) < DNS_STALE_MAX
        {
            warn!(
                authority,
                error = %e,
                age_s = now.duration_since(entry.resolved_at).as_secs(),
                "upstream dns re-resolution failed; serving the last good answer",
            );
            self.entries.rcu(|cur| {
                let mut next = HashMap::clone(cur);
                if let Some(entry) = next.get_mut(authority) {
                    entry.refresh_at = now + DNS_RETRY;
                }
                next
            });
        }
        Err(e)
    }
}

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
        let pool_keys: Vec<&str> = config
            .pool_keys
            .get(spec.name)
            .map(|k| k.iter().map(|s| s.expose()).collect())
            .unwrap_or_default();
        providers.insert(
            spec.name.to_string(),
            Arc::new(Provider::resolve(
                spec.name,
                authority,
                spec.wire,
                spec.auth,
                &pool_keys,
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
            let pool_keys: Vec<&str> = config
                .pool_keys
                .get(name)
                .map(|k| k.iter().map(|s| s.expose()).collect())
                .unwrap_or_default();
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
                    &pool_keys,
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

/// The managed `GET /v1/models` body for a deployment whose providers are `by_id`.
///
/// A row is listed when at least one of its candidates has a pool key here, so a row whose primary
/// is unkeyed but whose fallback is keyed stays listed: a managed request for it is served by the
/// fallback. A row none of whose candidates is keyed would only 503 a managed caller, so it is not
/// advertised (a deployment with no pool keys lists none). Only managed callers get this body: a BYO
/// caller never uses the catalog, and its listing relays to its own provider (D254).
pub(crate) fn models_list_body(
    by_id: &[Option<Arc<Provider>>; providers::ProviderId::COUNT],
) -> bytes::Bytes {
    let keyed =
        |id: providers::ProviderId| by_id[id.index()].as_ref().is_some_and(|p| p.has_pool_key());
    bytes::Bytes::from(providers::catalog::models_list_json(|r| {
        r.candidates.iter().any(|c| keyed(c.provider))
    }))
}

pub struct GatewayState {
    pub config: AiConfig,
    pub metrics: Arc<Metrics>,

    /// Trusted Ed25519 public keys by kid — from config (rotate via redeploy). Static for life.
    pub keyring: Keyring,
    /// Signs and verifies tenant-bound Responses ids (`signed_id.rs`). `None` without
    /// `id_signing_keys`: a managed Responses relay to a store is then a 503.
    pub id_signer: Option<crate::signed_id::Signer>,
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

    /// The managed `GET /v1/models` body, rendered once at boot by [`models_list_body`]: the
    /// catalog rows this deployment's pool keys can serve.
    pub models_list: bytes::Bytes,

    /// Sparse deny-set — watched from NATS. Default-allow on miss; fail-open.
    pub deny: ArcSwap<DenySet>,

    /// Sparse allowance-set — watched from NATS. Membership = exhausted. Unready until the first
    /// successful scan or snapshot (empty scan is remaining-ok). Fail-closed: a managed request
    /// 402s while unready, unlike deny.
    pub allowance: ArcSwap<AllowanceSet>,

    /// Sparse capture-set — watched from NATS under its own prefix and its own watcher. Default-off
    /// on miss, so a NATS outage degrades to "capturing nothing", which is the correct thing to lose.
    pub capture: ArcSwap<CaptureSet>,
    /// Per-tenant capture defaults resolved from config once at boot, applied to any control-plane
    /// entry that doesn't override them (and to a capture requested by header, which has no entry).
    pub capture_defaults: CaptureRule,

    /// Exact-match response cache. `None` when `cache_ttl_secs == 0`. Process-local: another replica
    /// does not share this table, and a miss does not consult a shared store.
    pub cache: Option<ResponseCache>,

    /// Per-candidate TTFT EWMA used to rank catalog walks. Always allocated; [`AiConfig::smart_router`]
    /// gates whether `rank` runs. Observing while the flag is off is wasted work, so the proxy
    /// skips both. Process-local: replicas do not share samples.
    pub smart: smart::Router,

    /// Per-tenant in-flight cap (see `concurrency`). `None` when `tenant_max_in_flight == 0`.
    pub tenant_slots: Option<TenantSlots>,

    /// Process-wide budget for buffered request bodies (see `concurrency::BodyBudget`). `None` when
    /// `max_buffered_body_bytes == 0`.
    pub body_budget: Option<BodyBudget>,

    /// Per-key request-rate guardrail (see `ratelimit`). `None` when `rate_limit_rps == 0`. Fixed
    /// memory regardless of tenant count, so it lives in the static state with no GC.
    pub rate_limit: Option<RateLimit>,

    /// TTL cache of resolved upstream addresses, so `upstream_peer` neither blocks on a synchronous
    /// `getaddrinfo` nor re-resolves the same provider host every request. `ArcSwap` so the common
    /// case — a cache hit, on every admitted request after warmup — is a lock-free atomic load; the
    /// only writes are the ~10 providers' entries refreshed once per `DNS_TTL`, applied via `rcu`.
    /// `Arc` so a refresh can run in a task of its own (see [`DnsCache`]).
    dns_cache: Arc<DnsCache>,

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

    /// Debug builds only: a proxy phase to panic in, once (see [`Self::fault_point`]).
    #[cfg(debug_assertions)]
    fault: FaultPanic,
}

/// A test-only fault: `AI_FAULT_PANIC=<phase>` makes the first request to reach that phase panic.
///
/// Compiled into debug builds only (the test binaries the integration tests spawn); a release
/// build has neither the field nor the env read, so production cannot be made to panic this way.
/// Exists to prove that a panic in a proxy phase releases what the request held (`proxy::Ctx`).
#[cfg(debug_assertions)]
struct FaultPanic {
    phase: Option<String>,
    fired: std::sync::atomic::AtomicBool,
}

#[cfg(debug_assertions)]
impl FaultPanic {
    fn from_env() -> Self {
        Self {
            phase: std::env::var("AI_FAULT_PANIC")
                .ok()
                .filter(|p| !p.is_empty()),
            fired: std::sync::atomic::AtomicBool::new(false),
        }
    }

    #[allow(clippy::panic)] // the whole point: a deliberate, test-only panic
    fn hit(&self, phase: &str) {
        if self.phase.as_deref() == Some(phase) && !self.fired.swap(true, Ordering::AcqRel) {
            panic!("AI_FAULT_PANIC: injected panic in {phase}");
        }
    }
}

impl GatewayState {
    pub fn new(config: AiConfig, metrics: Arc<Metrics>) -> Result<Arc<Self>> {
        let keyring = config.build_keyring()?;
        let id_signer = config.build_id_signer()?;
        // A deployment that verifies `bai_` keys serves managed traffic, and managed Responses ids
        // sit in one provider org every tenant shares (`signed_id.rs`): without an id signing key
        // they can't be tenant-bound, so this is a hard boot failure, not a warning. A BYO-only
        // deployment (no `signing_keys`) stores nothing on Beyond's accounts and needs none.
        if id_signer.is_none() && !config.signing_keys.is_empty() {
            return Err(GatewayError::Config(
                "signing_keys are configured (managed traffic) but no id_signing_keys are — refusing \
                 to boot: managed Responses ids would be resolvable by every tenant. Set \
                 AI_ID_SIGNING_KEY_1 to the base64 of 32 random bytes (e.g. `openssl rand -base64 \
                 32`)."
                    .to_string(),
            ));
        }
        // No signing keys ⇒ every `bai_v1…` fails verify and 401s (fail-closed). BYO still works.
        // That's a *valid* mode (a BYO-only deployment), but a far more common cause is a
        // missing/typo'd `signing_keys` (SSM param, env) — which 401s every managed tenant. A
        // managed deployment sets `require_signing_keys = true` so this mis-deploy is a hard,
        // visible boot failure; otherwise we warn loudly and continue (BYO-only is legitimate and
        // the test/e2e harnesses run keyless).
        if config.signing_keys.is_empty() {
            if config.require_signing_keys {
                return Err(GatewayError::Config(
                    "require_signing_keys is set but no signing_keys are configured — refusing to \
                     boot into a mode where every bai_v1 token 401s. Check the signing_keys \
                     config / SSM param."
                        .to_string(),
                ));
            }
            warn!(
                "no signing_keys configured — all managed (bai_v1) traffic will 401 (fail-closed); \
                 only BYO works. Expected only for a BYO-only deployment."
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
        // Pool keys over cleartext: every managed request would carry Beyond's provider key across
        // the network unencrypted. Legitimate only against the local plaintext mock (e2e/bench).
        if !config.upstream_tls && providers.values().any(|p| p.has_pool_key()) {
            warn!(
                "upstream_tls is DISABLED while pool keys are configured — Beyond's provider keys \
                 are sent in cleartext on every managed request. Valid ONLY for a local test/bench \
                 mock; never set upstream_tls=false against a real provider."
            );
        }
        let by_id = index_by_id(&providers);
        let models_list = models_list_body(&by_id);
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

        // Resolved once here rather than read per-request: these are boot config, and the capture
        // decision runs on every managed request.
        if config.capture_max_bytes > crate::capture::MAX_CAPTURE_BYTES {
            warn!(
                configured = config.capture_max_bytes,
                ceiling = crate::capture::MAX_CAPTURE_BYTES,
                "capture_max_bytes exceeds the ceiling; clamping",
            );
        }
        let capture_defaults = CaptureRule {
            sample_n: config.capture_default_sample_n.max(1),
            max_bytes: config
                .capture_max_bytes
                .min(crate::capture::MAX_CAPTURE_BYTES),
        };

        let cache = (config.cache_ttl_secs > 0).then(|| {
            cache::ResponseCache::new(
                Duration::from_secs(config.cache_ttl_secs),
                config.cache_max_entries,
                config.cache_max_bytes,
            )
        });

        Ok(Arc::new(Self {
            metrics,
            keyring,
            id_signer,
            providers,
            by_id,
            models_list,
            deny: ArcSwap::from_pointee(DenySet::new()),
            allowance: ArcSwap::from_pointee(AllowanceSet::new()),
            capture: ArcSwap::from_pointee(CaptureSet::new()),
            capture_defaults,
            cache,
            smart: smart::Router::new(),
            tenant_slots: TenantSlots::new(config.tenant_max_in_flight),
            body_budget: BodyBudget::new(config.max_buffered_body_bytes),
            rate_limit,
            dns_cache: Arc::new(DnsCache::default()),
            instance_prefix: {
                // Rendered once. Infallible: 16 hex digits + `-` is exactly the capacity.
                let mut p = InstancePrefix::new();
                let _ = write!(p, "{instance:x}-");
                p
            },
            request_seq: AtomicU64::new(0),
            #[cfg(debug_assertions)]
            fault: FaultPanic::from_env(),
            config,
        }))
    }

    /// A process-unique request id (`{instance}-{seq}`) for log correlation and the
    /// `x-beyond-request-id` response header. Deliberately *not* a uuid: a per-process instance
    /// token (computed once at boot) plus a relaxed atomic counter is unique across the fleet, costs
    /// one `fetch_add` + a hex format into a stack buffer (no heap allocation), and needs no
    /// randomness per request.
    pub fn next_request_id(&self) -> RequestId {
        self.next_request_id_seq().0
    }

    /// As [`Self::next_request_id`], but also returns the raw counter value behind it.
    ///
    /// The capture sampler needs a per-request number and this counter already is one, drawn from
    /// the same single `fetch_add` that mints the id — so sampling costs nothing extra, and "1 in N"
    /// is exactly 1 in N rather than a probabilistic approximation of it. Returning the pair keeps
    /// that to one atomic; a separate `fetch_add` for sampling would both cost more and desynchronize
    /// the two, making "why wasn't request X captured?" unanswerable from its id.
    pub fn next_request_id_seq(&self) -> (RequestId, u64) {
        let seq = self.request_seq.fetch_add(1, Ordering::Relaxed);
        let mut id = RequestId::new();
        // The instance half is boot-constant, so it is copied rather than re-formatted; only the
        // counter is rendered, and by a nibble loop rather than `core::fmt`. Can't overflow: 17-byte
        // prefix + ≤16 hex digits is exactly the buffer's capacity. A `try_push` that somehow did
        // overflow is dropped — a truncated correlation id beats panicking a worker over one.
        let _ = id.try_push_str(&self.instance_prefix);
        push_lower_hex(&mut id, seq);
        (id, seq)
    }

    /// The resolved provider for `name` (the request's first path segment, or the bare-path dialect
    /// default), or `None` if no such provider is registered — which `request_filter` turns into a
    /// 404.
    /// A named point in the proxy phases where a debug build can be told to panic (see
    /// `FaultPanic`). Compiles to nothing in a release build.
    #[inline(always)]
    pub fn fault_point(&self, _phase: &'static str) {
        #[cfg(debug_assertions)]
        self.fault.hit(_phase);
    }

    pub fn provider(&self, name: &str) -> Option<&Arc<Provider>> {
        self.providers.get(name)
    }

    /// The resolved provider for a catalog candidate's id, or `None` if this gateway does not route
    /// to it. One array index — the model-routed path calls this once per connect attempt.
    pub fn provider_by_id(&self, id: providers::ProviderId) -> Option<&Arc<Provider>> {
        self.by_id[id.index()].as_ref()
    }

    /// Resolve an `host:port` authority and pick the address for connect attempt `attempt`: the
    /// lookup's addresses in order, wrapping. Returns that address and how many there are, so a
    /// caller whose connect failed can try the next one. Cached for `DNS_TTL`, refreshed by one
    /// background lookup per authority while callers keep the cached answer; a lookup is bounded
    /// by `DNS_TIMEOUT`; when re-resolution fails the last good answer is served for up to
    /// `DNS_STALE_MAX`. Uses `tokio::net::lookup_host` (runs `getaddrinfo` on the blocking pool —
    /// async-safe) instead of `HttpPeer::new`'s eager blocking resolve.
    pub async fn resolve(&self, authority: &str, attempt: u8) -> Result<(SocketAddr, usize)> {
        let addrs = self
            .resolve_all(authority, |a| async move {
                tokio::net::lookup_host(a)
                    .await
                    .map(|it| it.collect::<Vec<_>>())
            })
            .await?;
        let addr = addrs[usize::from(attempt) % addrs.len()];
        Ok((addr, addrs.len()))
    }

    /// [`Self::resolve`]'s cache, single flight and serve-stale over an injectable `lookup`, so a
    /// test can fail or hang the resolver.
    ///
    /// A due refresh never makes a caller wait: the cached answer is served while one background
    /// lookup runs (D89). Only a caller with nothing servable (cold, or past `DNS_STALE_MAX`) waits,
    /// and concurrent ones share that one lookup and its answer, bounded by `DNS_TIMEOUT`.
    async fn resolve_all<F, Fut>(&self, authority: &str, lookup: F) -> Result<Arc<[SocketAddr]>>
    where
        F: FnOnce(String) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = std::io::Result<Vec<SocketAddr>>> + Send + 'static,
    {
        // Cache hit (the common case after warmup): a lock-free `ArcSwap` load — no mutex, no
        // syscall — so concurrent workers never serialize on a DNS lookup that's already resolved.
        let cached = self.dns_cache.entries.load().get(authority).cloned();
        let now = Instant::now();
        if let Some(entry) = &cached
            && now < entry.refresh_at
        {
            return Ok(entry.addrs.clone());
        }
        let mut answer = self.dns_cache.lookup_once(authority, lookup);
        if let Some(entry) = cached
            && now.duration_since(entry.resolved_at) < DNS_STALE_MAX
        {
            return Ok(entry.addrs);
        }
        let landed = match answer.wait_for(Option::is_some).await {
            Ok(a) => (*a).clone(),
            Err(_) => None,
        };
        match landed {
            Some(Ok(addrs)) => Ok(addrs),
            Some(Err(e)) => Err(GatewayError::Dns(e)),
            None => Err(GatewayError::Dns(format!(
                "{authority}: lookup ended without an answer"
            ))),
        }
    }
}

/// One process-wide `Metrics` (it registers on the default Prometheus registry, which rejects a
/// second registration), shared by every unit test that needs a `GatewayState`.
#[cfg(test)]
pub(crate) fn test_metrics() -> Arc<Metrics> {
    use std::sync::OnceLock;
    static M: OnceLock<Arc<Metrics>> = OnceLock::new();
    M.get_or_init(|| Metrics::new().expect("register metrics once"))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::AuthScheme;

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
                ("openai".to_string(), "sk-openai".into()),
                ("custom".to_string(), "sk-custom".into()),
            ]),
            ..Default::default()
        };
        let providers = build_providers(&config, &test_metrics()).unwrap();

        // Known provider: authority overridden, pool auth precomputed in the right scheme.
        let openai = providers.get("openai").unwrap();
        assert_eq!(openai.authority, "127.0.0.1:9");
        assert_eq!(openai.auth, AuthScheme::Bearer);
        assert_eq!(openai.pool_auth[0].value.expose(), "Bearer sk-openai");

        // Known provider, no override: built-in default + no pool key ⇒ no managed auth value.
        let anthropic = providers.get("anthropic").unwrap();
        assert_eq!(anthropic.authority, "api.anthropic.com:443");
        assert_eq!(anthropic.auth, AuthScheme::XApiKey);
        assert!(anthropic.pool_auth.is_empty());

        // Config-only provider: added as OpenAI-wire (Bearer), reachable by name.
        let custom = providers.get("custom").unwrap();
        assert_eq!(custom.host, "llm.internal");
        assert_eq!(custom.pool_auth[0].value.expose(), "Bearer sk-custom");

        // Config-only provider with no pool key: registered (reachable by name) but with no managed
        // auth value — this `None` is exactly what `request_filter` turns into a 503 for a managed
        // request. (BYO to it still works; it just can't serve the pooled path.)
        let custom2 = providers.get("custom2").unwrap();
        assert!(
            custom2.pool_auth.is_empty(),
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
            pool_keys: HashMap::from([("minimax".to_string(), "mm-key".into())]),
            ..Default::default()
        };
        let providers = build_providers(&config, &test_metrics()).unwrap();
        let minimax = providers.get("minimax").unwrap();
        assert_eq!(minimax.dialect, Dialect::Anthropic);
        assert_eq!(minimax.auth, AuthScheme::XApiKey);
        assert_eq!(minimax.pool_auth[0].value.expose(), "mm-key");

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
        let (addr, n) = state.resolve("127.0.0.1:9", 0).await.unwrap();
        assert_eq!(addr, "127.0.0.1:9".parse().unwrap());
        assert_eq!(n, 1);

        // Second call is served from the TTL cache: same answer, and the entry is now present.
        assert_eq!(state.resolve("127.0.0.1:9", 0).await.unwrap().0, addr);
        assert!(state.dns_cache.entries.load().contains_key("127.0.0.1:9"));

        // A guaranteed-NXDOMAIN host (RFC 6761 reserves `.invalid`) → a Dns error, never a panic.
        assert!(matches!(
            state.resolve("nonexistent.invalid:80", 0).await,
            Err(GatewayError::Dns(_))
        ));
    }

    fn addrs(list: &[&str]) -> Vec<SocketAddr> {
        list.iter().map(|a| a.parse().unwrap()).collect()
    }

    /// Every address is kept, in the lookup's order, and connect attempts walk them.
    #[tokio::test]
    async fn resolve_keeps_every_address_in_order() {
        let state = GatewayState::new(AiConfig::default(), test_metrics()).unwrap();
        let got = state
            .resolve_all("multi.test:443", |_| async {
                Ok(addrs(&["[::1]:443", "127.0.0.1:443"]))
            })
            .await
            .unwrap();
        assert_eq!(&*got, &addrs(&["[::1]:443", "127.0.0.1:443"])[..]);
    }

    /// A lookup that hangs is cut at `DNS_TIMEOUT`, and with no earlier answer it is an error.
    #[tokio::test(start_paused = true)]
    async fn a_hung_lookup_times_out() {
        let state = GatewayState::new(AiConfig::default(), test_metrics()).unwrap();
        let start = tokio::time::Instant::now();
        let got = state
            .resolve_all("hang.test:443", |_| std::future::pending())
            .await;
        assert!(matches!(got, Err(GatewayError::Dns(ref e)) if e.contains("timed out")));
        assert_eq!(start.elapsed(), DNS_TIMEOUT);
    }

    /// When re-resolution fails or hangs, the last good answer is served, within `DNS_STALE_MAX`.
    #[tokio::test]
    async fn a_failed_re_resolution_serves_the_last_good_answer() {
        let state = GatewayState::new(AiConfig::default(), test_metrics()).unwrap();
        let good = addrs(&["10.0.0.1:443"]);
        let primed = good.clone();
        let first = state
            .resolve_all("stale.test:443", move |_| async move { Ok(primed) })
            .await
            .unwrap();
        assert_eq!(&*first, &good[..]);
        // Expire the entry without waiting a TTL out.
        state.dns_cache.entries.rcu(|cur| {
            let mut next = HashMap::clone(cur);
            if let Some(e) = next.get_mut("stale.test:443") {
                e.refresh_at = Instant::now();
            }
            next
        });
        let stale = state
            .resolve_all("stale.test:443", |_| async {
                Err(std::io::Error::other("resolver down"))
            })
            .await
            .unwrap();
        assert_eq!(&*stale, &good[..], "served stale");
        // Past the stale bound it is an error again.
        state.dns_cache.entries.rcu(|cur| {
            let mut next = HashMap::clone(cur);
            if let Some(e) = next.get_mut("stale.test:443") {
                e.refresh_at = Instant::now();
                e.resolved_at = Instant::now() - DNS_STALE_MAX;
            }
            next
        });
        assert!(
            state
                .resolve_all("stale.test:443", |_| async {
                    Err(std::io::Error::other("resolver down"))
                })
                .await
                .is_err()
        );
    }

    /// Run `n` concurrent resolves of `authority` whose lookup counts itself and then hangs.
    /// Returns each caller's answer and how long it waited, and the number of lookups started.
    async fn hung_resolves(
        state: &Arc<GatewayState>,
        authority: &'static str,
        n: usize,
    ) -> (Vec<(Option<Vec<SocketAddr>>, Duration)>, usize) {
        use std::sync::atomic::AtomicUsize;
        let lookups = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..n {
            let (state, lookups) = (state.clone(), lookups.clone());
            tasks.push(tokio::spawn(async move {
                let start = tokio::time::Instant::now();
                let got = state
                    .resolve_all(authority, move |_| {
                        lookups.fetch_add(1, Ordering::SeqCst);
                        std::future::pending()
                    })
                    .await;
                (got.ok().map(|a| a.to_vec()), start.elapsed())
            }));
        }
        let mut out = Vec::new();
        for t in tasks {
            out.push(t.await.unwrap());
        }
        (out, lookups.load(Ordering::SeqCst))
    }

    /// A due refresh is one lookup, run in the background while every caller is served the cached
    /// answer at once. A hung resolver then costs nobody its timeout, and the number of `getaddrinfo`
    /// calls stuck on the blocking pool stays bounded.
    /// claim: REL-20
    /// defect: D89
    #[tokio::test(start_paused = true)]
    async fn a_due_refresh_is_one_background_lookup() {
        let state = GatewayState::new(AiConfig::default(), test_metrics()).unwrap();
        let good = addrs(&["10.0.0.1:443"]);
        let primed = good.clone();
        state
            .resolve_all("refresh.test:443", move |_| async move { Ok(primed) })
            .await
            .unwrap();
        state.dns_cache.entries.rcu(|cur| {
            let mut next = HashMap::clone(cur);
            if let Some(e) = next.get_mut("refresh.test:443") {
                e.refresh_at = Instant::now();
            }
            next
        });
        let (answers, lookups) = hung_resolves(&state, "refresh.test:443", 16).await;
        for (got, waited) in answers {
            assert_eq!(got.as_deref(), Some(&good[..]));
            assert_eq!(waited, Duration::ZERO, "a caller waited on the refresh");
        }
        assert_eq!(lookups, 1, "one refresh, not one per caller");
    }

    /// With nothing cached, concurrent callers share one lookup and its answer (here, its timeout).
    /// claim: REL-20
    /// defect: D89
    #[tokio::test(start_paused = true)]
    async fn concurrent_cold_resolves_share_one_lookup() {
        let state = GatewayState::new(AiConfig::default(), test_metrics()).unwrap();
        let (answers, lookups) = hung_resolves(&state, "cold.test:443", 16).await;
        for (got, waited) in answers {
            assert_eq!(got, None);
            assert_eq!(waited, DNS_TIMEOUT);
        }
        assert_eq!(lookups, 1, "one lookup, not one per caller");
    }

    /// A lookup that outlives its timeout stays the one in flight: later callers get its timeout at
    /// once instead of starting another `getaddrinfo` beside the hung one.
    #[tokio::test(start_paused = true)]
    async fn a_hung_lookup_is_never_joined_by_a_second() {
        let state = GatewayState::new(AiConfig::default(), test_metrics()).unwrap();
        let (_, lookups) = hung_resolves(&state, "hung.test:443", 1).await;
        assert_eq!(lookups, 1);
        let (answers, lookups) = hung_resolves(&state, "hung.test:443", 4).await;
        for (got, waited) in answers {
            assert_eq!((got, waited), (None, Duration::ZERO));
        }
        assert_eq!(lookups, 0, "the hung lookup is still the one in flight");
    }
}
