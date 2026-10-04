//! Layered configuration (PATTERNS.md: Figment defaults → TOML → `AI_`-prefixed env).
//!
//! Auth + key material come from config (signing public keys, managed pool keys). Deny is
//! fail-open without NATS; allowance is fail-closed until its watcher stores a scan or snapshot,
//! so managed traffic 402s until that read. BYO still serves from boot config alone.

use crate::error::{GatewayError, Result};
use crate::key::{Keyring, Kid};
use crate::secret::Secret;
use figment::Figment;
use figment::providers::{Env, Format, Toml};
use pingora_core::protocols::TcpKeepalive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::path::Path;
use std::time::Duration;

/// One provider's pool keys.
///
/// A TOML string or an `AI_POOL_KEY_*` env value is a list of one; a TOML array is N keys, walked
/// in order on a managed 429 (see `proxy::upstream_response_filter`). An empty list is the same as
/// a missing entry: managed traffic to that provider 503s.
#[derive(Clone, Default, Debug)]
pub struct PoolKeyList(Vec<Secret>);

impl PoolKeyList {
    pub fn as_slice(&self) -> &[Secret] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Deref for PoolKeyList {
    type Target = [Secret];
    fn deref(&self) -> &[Secret] {
        &self.0
    }
}

impl From<Secret> for PoolKeyList {
    fn from(s: Secret) -> Self {
        Self(vec![s])
    }
}

impl From<&str> for PoolKeyList {
    fn from(s: &str) -> Self {
        Self(vec![Secret::new(s)])
    }
}

impl From<Vec<Secret>> for PoolKeyList {
    fn from(v: Vec<Secret>) -> Self {
        Self(v)
    }
}

impl Serialize for PoolKeyList {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl<'de> Deserialize<'de> for PoolKeyList {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = PoolKeyList;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a pool key string or an array of pool keys")
            }
            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(PoolKeyList(vec![Secret::new(v)]))
            }
            fn visit_string<E: serde::de::Error>(
                self,
                v: String,
            ) -> std::result::Result<Self::Value, E> {
                Ok(PoolKeyList(vec![Secret::new(v)]))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut keys = Vec::new();
                while let Some(s) = seq.next_element::<Secret>()? {
                    keys.push(s);
                }
                Ok(PoolKeyList(keys))
            }
        }
        d.deserialize_any(V)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
// `default` so every field is optional. We deliberately do NOT set serde's `deny_unknown_fields`:
// config is merged from `Env::prefixed("AI_")`, a namespace shared with foreign variables the
// platform injects (e.g. `AI_AGENT`, `AI_LOG`), so rejecting unknown keys at the serde layer would
// fail load on a valid environment. Typo protection is instead enforced one layer down, against the
// *TOML file only* (`reject_unknown_toml_keys`): the file is ours alone — not a shared namespace —
// so an unrecognized key there is unambiguously a mistake, and a silent one (it loads its default
// and the setting does nothing), worth a hard, visible boot failure.
#[serde(default)]
pub struct AiConfig {
    /// Downstream listener for client (app) traffic. Internal-only in production (Service Connect
    /// fronts it as `ai.internal`); no public ingress, so plain HTTP here is fine.
    pub listen: String,
    /// Prometheus metrics listener.
    pub metrics_listen: String,

    /// Tokio worker threads for the **proxy** service — the threads that run every request filter,
    /// the Ed25519 verify, the body scanners and the usage tap. `0` ⇒ one per available core.
    ///
    /// This has to be set explicitly because Pingora's `ServerConf::default()` is `threads: 1`, and
    /// a service that leaves its own `threads` as `None` inherits it (`server/mod.rs:705` →
    /// `Runtime::new_steal(threads, …)` → `worker_threads(1)`). Unset, the entire proxy therefore
    /// runs on a **single** core no matter how large the box: measured at 100% of one core with the
    /// other fifteen idle. It is applied to the proxy service alone rather than to `conf.threads`,
    /// which would also hand the (idle) admin listener a full thread pool.
    ///
    /// Note this counts *async worker* threads only. Tokio names its blocking-pool threads after the
    /// same runtime, so `ps` shows extra `Pingora HTTP Pr` threads once `lookup_host` runs — those
    /// never execute a filter, and only per-thread CPU accounting tells them apart.
    ///
    /// Set it explicitly under a CPU quota: `available_parallelism` reports the host's cores, not the
    /// cgroup's share, so an over-provisioned pool on a throttled container just adds scheduler churn.
    pub worker_threads: usize,

    /// Accept HTTP/2 cleartext (h2c) on the downstream (`listen`) listener, in addition to
    /// HTTP/1.1. Backward-compatible: Pingora peeks the connection preface and serves h2c only when
    /// the client sends the H2 preface, otherwise it transparently falls back to HTTP/1.1 — so
    /// existing h1 clients are unaffected. Lets an on-VM agent multiplex all its sessions over a
    /// single h2c connection to the gateway (`--upstream-http2 h2c` on the agent side). Default
    /// `true`; set `false` to force h1-only downstream.
    pub downstream_h2c: bool,

    /// NATS / slipstream connection (cf. `_envcommon/ecs-service.hcl`: `tls://connect.ngs.global`).
    /// Used for the watched deny-set (`blackhole.*`), allowance-set (`allowance.*`), and
    /// capture-set (`aicapture.*`).
    pub nats_url: String,
    /// Base64 `.creds` (ECS via SOPS) — takes priority over `nats_creds_file`. Held in `Secret` so
    /// it can't leak through the `Debug`/`Serialize` this struct derives (a stray `?config` log).
    pub nats_creds: Option<Secret>,
    pub nats_creds_file: Option<String>,
    /// slipstream bucket holding `blackhole.*`, `allowance.*`, and `aicapture.*`.
    pub config_bucket: String,

    /// Optional path to an on-disk deny-set snapshot (slipstream's append-log + resume cursor). When
    /// set **and on durable storage** (the edge/tunnel deployment model), a restart seeds the
    /// deny-set from this file and *resumes the NATS watch from the saved revision* — skipping the
    /// boot scan and surviving a restart with enforcement intact even before NATS reconnects. The
    /// allowance-set uses `{path}.allowance` so the two files cannot be loaded as each other. Unset
    /// (the default, e.g. ephemeral/Fargate) ⇒ seed from a NATS scan each boot, unchanged. The file
    /// is a pure cache: delete it (or point at scratch) and the gateway falls back to scanning.
    pub snapshot_path: Option<String>,

    /// Trusted Ed25519 signing **public** keys: `kid` (as string — TOML/JSON map keys are strings)
    /// → base64 public key. Multiple allowed for zero-downtime rotation. Config, not NATS.
    pub signing_keys: HashMap<String, String>,

    /// Fail the boot if `signing_keys` is empty, instead of serving BYO-only with every `bai_v1`
    /// token 401ing (fail-closed). Empty signing keys is a *legitimate* mode (a BYO-only
    /// deployment) but is far more often a mis-deploy — a typo'd/absent SSM param — that 401s
    /// every managed client. A managed deployment should set this `true` so a bad deploy fails
    /// fast and visibly at boot. Default `false` to keep BYO-only and the test/e2e harnesses
    /// (which run keyless) working out of the box.
    pub require_signing_keys: bool,

    /// Secrets that bind provider-held Responses ids to a tenant (`signed_id.rs`): kid (one
    /// character, `[0-9A-Za-z]`) → base64 of at least 32 random bytes. From the
    /// `[id_signing_keys]` table or `AI_ID_SIGNING_KEY_<KID>` env. Empty ⇒ a managed Responses
    /// relay to a store (a GPT row's Responses arm, a managed `/{provider}/…/responses`) is a 503
    /// (fail-closed); everything else is unaffected. More than one ⇒ rotation (see
    /// `id_signing_kid`).
    pub id_signing_keys: HashMap<String, Secret>,

    /// The kid in `id_signing_keys` that signs new ids. Optional with exactly one key. Ids carry
    /// the kid that signed them, so ids from a previous key verify while that key stays listed.
    pub id_signing_kid: String,

    /// Managed Beyond pool keys, **by provider name** (`openai`, `anthropic`, `fireworks`, …).
    /// From the `[pool_keys]` TOML table (an array of keys, or a string = list of one) or
    /// SSM-injected `AI_POOL_KEY_<NAME>` env (the env form is the production path and stays one
    /// key = list of one — see `load_with_path`). A provider with no keys / an empty list can't
    /// serve managed traffic (→ 503); BYO is unaffected. On a managed 429 the gateway walks the
    /// next unused key for that same provider; it never sends provider A's key to provider B.
    pub pool_keys: HashMap<String, PoolKeyList>,

    /// Per-provider upstream authority (`host:port`), **by provider name**. For a known provider
    /// (see `route::KNOWN_PROVIDERS`) this *overrides* its default; for an unknown name it *adds* a
    /// new provider, then reachable at `/{name}/…` (the provider is the request's first path
    /// segment). Empty = every known provider uses its built-in default. (The e2e harness points
    /// providers at a mock here.)
    pub provider_authorities: HashMap<String, String>,

    /// Wire dialect for a **config-added** provider (a `provider_authorities` entry whose name isn't
    /// in `route::KNOWN_PROVIDERS`), by provider name: `"openai"` or `"anthropic"` (case-insensitive;
    /// see `Dialect::parse_config`). Has no effect on a known provider — its dialect is fixed in code.
    /// Missing ⇒ `"openai"` (the long-standing default, kept for backward compatibility). Real pi
    /// vendors like MiniMax, MiniMax-CN, and Kimi-Coding speak the **Anthropic** wire (`x-api-key`,
    /// `usage.input_tokens`/`output_tokens`) — defaulting them to OpenAI-wire silently zero-bills
    /// every request (`usage::openai_body` deserializes the mismatched shape into all-zero fields
    /// instead of failing). An unrecognized value is a **hard boot failure**, not a silent fallback to
    /// `"openai"` — that fallback is exactly the bug this field exists to let an operator opt out of.
    pub provider_dialects: HashMap<String, String>,

    /// Managed auth scheme for a **config-added** provider, by provider name: `"bearer"`,
    /// `"x-api-key"`, or `"api-key"` (Azure OpenAI's bare-key header — see `AuthScheme::ApiKey`;
    /// case-insensitive, `-`/`_` ignored; see `AuthScheme::parse_config`). Has no effect on a known
    /// provider. Missing ⇒ `"bearer"` (backward-compatible default). An unrecognized value is a hard
    /// boot failure.
    pub provider_auth_schemes: HashMap<String, String>,

    /// Upstream timeouts (seconds). Streaming responses are long, so read/idle are generous.
    pub connect_timeout_secs: u64,
    /// The longest the provider may stay silent: before its response head, or between body reads
    /// (pingora has one per-read upstream timeout). 600 s, the OpenAI and Anthropic SDKs' default
    /// request timeout, so the gateway never gives up on a slow-but-alive provider before the
    /// client itself would. Silence is not a failure signal: a model thinking without emitting
    /// looks exactly like a stuck provider. A *dead* provider connection is caught sooner, by
    /// transport liveness (`h2_ping_interval_secs`, `tcp_keepalive_*`).
    pub read_timeout_secs: u64,
    pub write_timeout_secs: u64,
    pub idle_timeout_secs: u64,

    /// Downstream write timeout (seconds): a write to the client that cannot make progress for this
    /// long ends the request. A client that stops reading (a hung SDK, a paused process, a
    /// half-dead connection) otherwise held its in-flight slot, its tenant slot, its upstream
    /// connection and possibly a half-open probe permit forever, since nothing else times out a
    /// blocked write. Per write: a slow but steady reader never trips it. `0` disables it.
    ///
    /// Not a guess at model behavior, and not replaceable by keepalive: it judges a client that
    /// stopped consuming bytes the gateway already holds. A live process that stops reading still
    /// ACKs at the kernel and advertises a zero window, so TCP keepalive (which probes only an idle
    /// connection) reports it healthy forever, and its own timeout cannot fire because it is not
    /// waiting on anything. A client that reads at all drains a full socket buffer within a few
    /// round trips; 60 s of zero progress is hundreds of them.
    pub client_write_timeout_secs: u64,

    /// HTTP/2 PING interval on upstream connections (seconds); `0` disables. A provider that stops
    /// acknowledging (a dead host, a partition, a wedged edge) fails the connection and every
    /// stream on it within this interval plus pingora's fixed 5 s ACK deadline, however silent the
    /// model is meant to be: a PING is answered by the peer's HTTP/2 stack, not by the model. 15 s:
    /// the bound it buys (≤ 20 s) is far under any client's patience, and one 17-byte frame and its
    /// ACK per connection per interval is negligible next to a token stream. The 5 s ACK deadline is
    /// over ten worst-case intercontinental round trips (~300 ms) and fits several TCP
    /// retransmissions, so a live but distant peer never misses it.
    pub h2_ping_interval_secs: u64,
    /// TCP keepalive on upstream and client connections: probe after this many idle seconds, every
    /// `tcp_keepalive_interval_secs`, and drop the peer after `tcp_keepalive_count` unanswered
    /// probes; `0` disables. Probes are answered by the peer's kernel, so a live process that is
    /// merely quiet always passes, and a vanished host fails within idle + interval × count
    /// (15 + 5 × 3 = 30 s). Upstream it covers HTTP/1.1 providers, which have no PING, and the
    /// connection is also given `TCP_USER_TIMEOUT` of the same 30 s, so data the provider never
    /// acknowledges (a partition mid-request, where keepalive does not probe) fails as fast.
    /// Toward clients it frees a vanished client's slot during a silent model turn, when the
    /// gateway has nothing to write. The interval is 15 × a 300 ms worst-case round trip and well
    /// past Linux's 200 ms minimum retransmission timeout; three probes ride out two lost ones.
    /// The kernel's own default (2 h idle, 9 × 75 s) is far too slow to matter.
    pub tcp_keepalive_idle_secs: u64,
    pub tcp_keepalive_interval_secs: u64,
    pub tcp_keepalive_count: u32,

    /// Most request-body bytes this process holds in memory at once, across every request. A body
    /// past pingora's 64 KiB replay buffer is read in full for a catalog walk (twice over while its
    /// `FullBody` re-run buffers its own copy), and a managed OpenAI chat body is buffered for the
    /// usage splice; each is capped at 100 MiB but nothing bounded how many. A request that would
    /// cross this gets a 503 with `Retry-After` (`ai_rejections_total{reason="body_memory"}`)
    /// before its body is read, or as soon as a chunked one grows past it. Bodies within the replay
    /// buffer are not counted. `0` disables.
    pub max_buffered_body_bytes: usize,

    /// Graceful-shutdown drain window (seconds): after SIGTERM, how long Pingora lets **in-flight
    /// requests finish** before tearing the runtimes down. Maps to Pingora's `grace_period_seconds`
    /// (left unset, Pingora silently defaults to 300s — this knob makes the window explicit).
    ///
    /// **Default to `read_timeout_secs` so we never truncate a response.** The gateway is a
    /// transparent man-in-the-middle: cutting an in-flight stream on deploy corrupts a generation the
    /// caller is paying for and can't cleanly retry (a half-delivered SSE isn't idempotent). The
    /// longest a request can live is `read_timeout_secs`, so a drain window of at least that
    /// guarantees every accepted request finishes — Pingora stops *accepting* new connections the
    /// instant SIGTERM lands, so this only ever waits out the existing longest stream, not new work.
    /// Slower rollouts are the deliberate price of not mangling responses. It is an upper bound,
    /// not a wait: the process exits as soon as no request is left in flight (see `main`'s drain).
    ///
    /// **The orchestrator must grant the same window**, or it caps us: the platform SIGKILLs at its
    /// own stop timeout regardless of this value. Set k8s `terminationGracePeriodSeconds` (or the EC2
    /// agent's `ECS_CONTAINER_STOP_TIMEOUT`) to match. Note **ECS Fargate caps `stopTimeout` at 120s**
    /// — there, full coverage of a 600s stream is impossible and the longest streams will still be
    /// cut at 120s; that's a Fargate limitation, not a reason to default to truncating.
    pub shutdown_grace_period_secs: u64,
    /// Final runtime-teardown timeout (seconds) **after** the drain window: how long Pingora waits for
    /// the tokio runtimes to exit before forcing the process down. Maps to Pingora's
    /// `graceful_shutdown_timeout_seconds` (unset ⇒ a silent 5s default). A few seconds is enough to
    /// flush logs/metrics; this is a backstop against a wedged runtime hanging shutdown forever, not a
    /// second drain window (that's `shutdown_grace_period_secs`).
    pub shutdown_runtime_timeout_secs: u64,

    /// TLS to the upstream provider. Real providers are HTTPS (true); the e2e harness sets false
    /// to talk to a plaintext mock.
    pub upstream_tls: bool,

    /// Prefer HTTP/2 (with HTTP/1.1 fallback) to the upstream. `true` ⇒ peer ALPN `H2H1`: every
    /// provider that offers `h2` over TLS is reached over a multiplexed H2 connection (fewer sockets
    /// and TLS handshakes from our egress IPs), and any host that doesn't offer it negotiates down to
    /// H1. `false` ⇒ ALPN `H1` (one connection per in-flight request, pooled). The knob exists so an
    /// operator can fall back to H1 without a code redeploy if a provider's h2 stack misbehaves, and
    /// so the e2e concurrency bench can compare the two. Only consulted over TLS — a plaintext upstream
    /// (the mock) has no ALPN and is always H1 regardless.
    pub upstream_http2: bool,

    /// Verify the upstream's TLS certificate (and that it matches the SNI). `true` everywhere in
    /// production. The **only** intended `false` is the e2e concurrency bench, whose TLS mock presents
    /// a self-signed cert — turning verification off there lets us exercise the real TLS+ALPN+H2 path
    /// against a local mock without a CA. Never set this `false` against a real provider.
    pub upstream_verify_cert: bool,

    /// Per-credential request-rate ceiling (requests/sec). A blast-radius guardrail (see `ratelimit`),
    /// not a spend control: it caps how fast a single credential (managed virtual key ≈ a `(tenant,
    /// app)`, or a BYO token) can drive the gateway, bounding a leaked/runaway key during the
    /// deny-set's reaction lag and a failure flood that never bills. `0` disables it. The default is
    /// generous — a circuit breaker, not a quota; tune from `ai_rejections_total{reason="rate_limit"}`.
    pub rate_limit_rps: u32,

    /// Aggregate request-rate ceiling (requests/sec) for **all BYO traffic combined** — a single
    /// shared bucket. BYO is unverified and upstream-bound, so a flood of *distinct* random BYO tokens
    /// slips past the per-credential ceiling and would open junk-auth connections to providers from
    /// our egress IPs (getting them rate-limited or banned). This bounds that aggregate regardless of
    /// token variation. Managed traffic is **exempt** (it's Ed25519-verified before any upstream
    /// connect and can't be forged), so this shared bucket never sheds core tenant load. `0` disables
    /// it. Generous by default; tune from `ai_rejections_total{reason="rate_limit_byo_global"}`.
    ///
    /// Before changing this (or reaching for per-IP limiting), read the **design-decision** block in
    /// the `ratelimit` module docs: it records why this is a global cap and not per-source-IP, what it
    /// deliberately doesn't cover, and why the real fix for egress-reputation pain is a
    /// provider-feedback circuit breaker rather than a bigger number here.
    pub byo_rate_limit_rps: u32,

    /// Per-provider circuit breaker: number of upstream **failures within `circuit_breaker_window_secs`**
    /// that trips the breaker open for that provider. A failure is a **5xx response or a connect
    /// failure** — i.e. the *provider is broken*. A `429` is deliberately **not** a failure: it means
    /// the provider is healthy and throttling that credential (a velocity/spend signal the rate
    /// limiter, the same-provider key walk, and the client's `Retry-After` backoff own), so tripping
    /// on it would convert a self-healing throttle into a self-inflicted outage. While open, requests to that provider fast-fail with a
    /// `503` (`ai_rejections_total{reason="circuit_open"}`) instead of piling up against
    /// `read_timeout_secs` and exhausting connection/in-flight slots for *every* provider. After
    /// `circuit_breaker_reset_secs` a probe request is allowed; success closes it, failure reopens it.
    /// Applies to **all** traffic to the provider (managed + BYO) — a down provider is down regardless
    /// of whose key is used. `0` disables the breaker entirely. Default is generous so normal
    /// background 5xx noise never trips it.
    pub circuit_breaker_threshold: u32,
    /// Fixed window (seconds) over which `circuit_breaker_threshold` failures are counted. A window
    /// starts at its first failure; the first failure after it ends starts a new one with every count
    /// back at zero — so it trips on a *burst* of failures, not on a slow trickle spread across a
    /// healthy day.
    pub circuit_breaker_window_secs: u64,
    /// How long the breaker stays open before allowing a half-open probe request (seconds). Long enough
    /// to let a provider recover, short enough that recovery is detected promptly.
    pub circuit_breaker_reset_secs: u64,

    /// Per-direction byte cap on a captured payload, before truncation. Applies to the request body
    /// and the response body independently, and is the default a control-plane capture entry
    /// overrides per tenant.
    ///
    /// This is really a bound on what the **log pipeline** will carry: a captured payload rides the
    /// same logfwd/OTLP path as `ai.usage`, so the cap has to sit under whatever per-record limit
    /// that path enforces. Raise it only alongside that limit, or captures will be silently dropped
    /// downstream where this gateway can't see it happen.
    pub capture_max_bytes: u32,
    /// Default sampling for control-plane-enabled capture: keep 1 request in N. `1` captures every
    /// request. Ignored for a capture explicitly requested via `x-beyond-capture: on`, which is
    /// never sampled away.
    pub capture_default_sample_n: u32,
    /// Depth of the bounded queue feeding the capture sink. When it fills, captures are **dropped**
    /// (counted on `ai_capture_dropped_total`) rather than blocking — a stalled log sink must never
    /// be able to backpressure the data plane. Deeper absorbs longer sink stalls at the cost of
    /// holding more payload bytes in memory.
    pub capture_queue_depth: usize,
    /// Bound on the bytes held by queued capture lines (`0`: none). A line count alone is not a
    /// memory bound: one line carries up to two `capture_max_bytes` bodies, more after JSON
    /// escaping, so a full queue of 1 024 could hold gigabytes while the log pipeline stalls. A
    /// line that would cross this bound is **dropped** (counted on `ai_capture_dropped_total`),
    /// never waited for (D262).
    pub capture_queue_bytes: usize,

    /// Exact-match response cache TTL (seconds). `0` disables it (the default). Only managed
    /// catalog walks whose client body is already in hand before `upstream_peer` (`/auto`, managed
    /// `/v1`) look up or fill. A hit replays the stored 2xx and skips the provider; a miss stays an
    /// unbuffered relay and fills via a tap. BYO and `/{provider}` passthrough are not cached.
    ///
    /// The store is **this process**. Another replica does not see the entry; a miss never consults
    /// Redis or any shared backend.
    pub cache_ttl_secs: u64,
    /// Cap on stored entries. Oldest insertion is dropped when a new one would exceed it.
    pub cache_max_entries: usize,
    /// Cap on a single stored response body (bytes). Oversize 2xxs are relayed but not stored.
    pub cache_max_bytes: usize,

    /// Order a managed default walk by the caller's computed session pin (`crate::pin`): the row's
    /// leading first-party hosts by rendezvous hash of the caller, then catalog order. Off is the
    /// row's static order, still subject to `x-beyond-order` / `only` / `split`. A pure function of
    /// the caller and the row, so every replica agrees. The name is from when this also ranked by
    /// latency; it is kept so deployed configs (and `reject_unknown_toml_keys`) still load.
    pub session_pins: bool,

    /// Most requests one tenant may hold open on this process at once. `0` disables it (the
    /// default). Spend is enforced after the fact — the allowance-set's exhaust bit lands only once
    /// usage has shipped and been summed — so this is what bounds a tenant's overshoot in that
    /// window: at most this many requests' worth per replica. Over the cap → `429`
    /// (`ai_rejections_total{reason="tenant_concurrency"}`). Managed traffic only.
    pub tenant_max_in_flight: u32,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".to_string(),
            metrics_listen: "0.0.0.0:9090".to_string(),
            // One proxy worker per core. Pingora's own default is 1, which silently pins the whole
            // gateway to a single core; scaling with the box is the least surprising default for an
            // L7 proxy. Override under a CPU quota — see the field docs.
            worker_threads: 0,
            // Accept downstream h2c by default; it's backward-compatible (h1 clients fall back
            // transparently) and lets the on-VM agent share one multiplexed connection.
            downstream_h2c: true,
            nats_url: "nats://localhost:4222".to_string(),
            nats_creds: None,
            nats_creds_file: None,
            config_bucket: "ai-gateway".to_string(),
            snapshot_path: None,
            signing_keys: HashMap::new(),
            require_signing_keys: false,
            id_signing_keys: HashMap::new(),
            id_signing_kid: String::new(),
            pool_keys: HashMap::new(),
            provider_authorities: HashMap::new(),
            provider_dialects: HashMap::new(),
            provider_auth_schemes: HashMap::new(),
            connect_timeout_secs: 10,
            // Generous: LLM streams can run for minutes; a tight read timeout would kill them.
            read_timeout_secs: 600,
            write_timeout_secs: 60,
            idle_timeout_secs: 90,
            client_write_timeout_secs: 60,
            h2_ping_interval_secs: 15,
            tcp_keepalive_idle_secs: 15,
            tcp_keepalive_interval_secs: 5,
            tcp_keepalive_count: 3,
            max_buffered_body_bytes: 512 * 1024 * 1024,
            // Drain for the full request lifetime (= read_timeout_secs) so a deploy never truncates
            // an in-flight stream — we're a transparent proxy and must not mangle a paid-for
            // generation. Pingora stops accepting new connections at SIGTERM, so this only waits out
            // the longest existing stream. The orchestrator's stop timeout must match (see field
            // docs; ECS Fargate's 120s cap is a hard limit there). Then a short teardown backstop.
            shutdown_grace_period_secs: 600,
            shutdown_runtime_timeout_secs: 10,
            upstream_tls: true,
            // Prefer H2 to providers by default (all of `KNOWN_PROVIDERS` offer it; H1 fallback is
            // automatic). Flip to false for an all-H1 upstream without recompiling.
            upstream_http2: true,
            // Verify upstream certs by default; only the bench's self-signed TLS mock turns this off.
            upstream_verify_cert: true,
            // Generous per-credential circuit breaker, on by default. Won't touch legitimate
            // steady-state traffic; caps a runaway/leaked key or a retry-storm flood. Set 0 to disable.
            rate_limit_rps: 100,
            // Generous aggregate BYO ceiling, on by default — well above any expected legitimate BYO
            // throughput, low enough that a junk-auth flood can't get our egress IPs flagged by the
            // providers. Tune from the metric; set 0 to disable. (Managed traffic is exempt.)
            byo_rate_limit_rps: 1_000,
            // Per-provider breaker: trip after 20 upstream failures (5xx/connect) within 10s, stay
            // open 30s, then probe. Generous enough that a provider's occasional background 5xx never
            // trips it — only a sustained brownout does. Set threshold 0 to disable.
            circuit_breaker_threshold: 20,
            circuit_breaker_window_secs: 10,
            circuit_breaker_reset_secs: 30,
            // 256 KiB per direction. Comfortably holds a large system prompt plus a long
            // conversation — the shapes this feature exists to explain — while staying well under
            // the per-record size a log/OTLP pipeline will carry without complaint.
            capture_max_bytes: 256 * 1024,
            // Capture everything for a tenant that's been switched on. Capture is a targeted
            // debugging tool enabled for one tenant at a time, so the default that answers "what did
            // the agent do" is *all of it*; sampling is the escape hatch for a tenant whose volume
            // makes that impractical, set per-tenant on the control-plane entry.
            capture_default_sample_n: 1,
            // Absorbs a multi-second sink stall at a healthy capture rate. Past that we drop rather
            // than block — see the field docs and `ai_capture_dropped_total`.
            capture_queue_depth: 1024,
            // 64 MiB: dozens of full-size payloads, a small fraction of `max_buffered_body_bytes`.
            capture_queue_bytes: 64 * 1024 * 1024,
            // Off. A TTL of 0 is the disable knob; turning it on is an operator choice, not a
            // surprise change in what the gateway talks to.
            cache_ttl_secs: 0,
            cache_max_entries: 1024,
            // 64 KiB: the same bound as the catalog-walk peek, so a cached response is no larger
            // than the request that produced it was allowed to be while still being "in hand".
            cache_max_bytes: 64 * 1024,
            session_pins: true,
            // Off. The right ceiling depends on how many parallel agents a tenant legitimately runs
            // and how long the allowance pipeline lags; that is an operator's call.
            tenant_max_in_flight: 0,
        }
    }
}

impl AiConfig {
    pub fn load_with_path(path: Option<&Path>) -> Result<Self> {
        let toml_path = path.unwrap_or_else(|| Path::new("config.toml"));

        // Serialize the defaults once and read the file once, then reuse both.
        //
        // figment's `Data::data()` re-reads and re-parses the file on *every* call and does not
        // cache, so validating the key set and then merging the same `Toml::file(..)` opened, read
        // and parsed the config twice — and, more than a wasted parse, read it at two different
        // instants, so a file rewritten in between yielded a validated-then-different config. The
        // same applied to `AiConfig::default()`, serialized once for the known-key set and again
        // for the merge.
        let defaults = pre_read(figment::providers::Serialized::defaults(AiConfig::default()))?;
        // Catch a typo'd key in the operator's own TOML *before* any of it merges — a misspelled
        // `require_signing_keys` would otherwise load its default and 401 every managed client
        // while the gateway looks healthy. Only the TOML file is checked (see the
        // `deny_unknown_fields` note on `AiConfig`); the env layer must stay lenient.
        let toml = read_toml(toml_path)?;
        reject_unknown_toml_keys(toml_path, &defaults, &toml)?;

        let mut fig = Figment::from(defaults).merge(toml);
        // Flat mapping: `AI_READ_TIMEOUT_SECS` → `read_timeout_secs`. (No `.split('_')` — these are
        // flat fields, not nested tables.) Unknown `AI_*` vars are tolerated (see the
        // `deny_unknown_fields` note on `AiConfig`) — which is also why pool keys are collected
        // separately below rather than via this flat merge.
        fig = fig.merge(Env::prefixed("AI_"));
        let mut cfg: AiConfig = fig
            .extract()
            .map_err(|e| GatewayError::Config(e.to_string()))?;
        // One pass over the environment for both secret prefixes, and via `vars_os` because
        // `std::env::vars()` *panics* on a variable that isn't valid UTF-8 — a hostile or merely odd
        // environment should not be able to kill the boot before it starts.
        cfg.merge_secret_env(env_pairs());
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject nonsensical values that would otherwise fail silently at runtime. A `0` connect/read
    /// timeout (a typo'd SSM param) becomes a `Duration::from_secs(0)` deadline that fails every
    /// upstream call immediately — surfacing only as a 502 cascade, not a loud boot failure. Catch it
    /// here so a mis-deploy fails fast and visibly. Write/idle are not load-bearing for correctness
    /// (Pingora treats them as best-effort), so they're left unconstrained.
    fn validate(&self) -> Result<()> {
        if self.connect_timeout_secs == 0 {
            return Err(GatewayError::Config(
                "connect_timeout_secs must be > 0 (a 0 connect timeout fails every upstream connect)"
                    .to_string(),
            ));
        }
        if self.read_timeout_secs == 0 {
            return Err(GatewayError::Config(
                "read_timeout_secs must be > 0 (a 0 read timeout aborts every response before it arrives)"
                    .to_string(),
            ));
        }
        // The kernel rejects a zero probe interval or count (EINVAL), which would fail every
        // upstream connect and leave every accepted client socket without keepalive.
        if self.tcp_keepalive_idle_secs > 0
            && (self.tcp_keepalive_interval_secs == 0 || self.tcp_keepalive_count == 0)
        {
            return Err(GatewayError::Config(
                "tcp_keepalive_interval_secs and tcp_keepalive_count must be > 0 when                  tcp_keepalive_idle_secs is set; set tcp_keepalive_idle_secs = 0 to disable keepalive"
                    .to_string(),
            ));
        }
        // A breaker with a zero window counts every failure into an already-expired window, so the
        // count never passes 1 and the breaker never opens; a zero reset re-admits a probe at once,
        // so an open breaker sheds nothing. Both read as "breaker on" while doing nothing.
        if self.circuit_breaker_threshold > 0 && self.circuit_breaker_window_secs == 0 {
            return Err(GatewayError::Config(
                "circuit_breaker_window_secs must be > 0 (a 0 window never accrues a second \
                 failure, so the breaker never opens); set circuit_breaker_threshold = 0 to disable it"
                    .to_string(),
            ));
        }
        if self.circuit_breaker_threshold > 0 && self.circuit_breaker_reset_secs == 0 {
            return Err(GatewayError::Config(
                "circuit_breaker_reset_secs must be > 0 (a 0 reset half-opens the breaker the \
                 instant it opens); set circuit_breaker_threshold = 0 to disable it"
                    .to_string(),
            ));
        }
        if self.circuit_breaker_threshold > crate::circuit_breaker::MAX_FAILURE_THRESHOLD {
            return Err(GatewayError::Config(format!(
                "circuit_breaker_threshold = {} exceeds the maximum of {} (the breaker's packed \
                 failure count cannot hold more)",
                self.circuit_breaker_threshold,
                crate::circuit_breaker::MAX_FAILURE_THRESHOLD,
            )));
        }
        // A malformed id signing key is a boot failure, not a 503 on every managed Responses turn.
        self.build_id_signer()?;
        Ok(())
    }

    /// The signer for tenant-bound Responses ids (`signed_id.rs`), or `None` when no
    /// `id_signing_keys` are set.
    pub fn build_id_signer(&self) -> Result<Option<crate::signed_id::Signer>> {
        crate::signed_id::Signer::from_config(&self.id_signing_keys, &self.id_signing_kid)
            .map_err(GatewayError::Config)
    }

    /// The per-provider circuit-breaker config, or `None` when disabled (`circuit_breaker_threshold
    /// == 0`). Windowed policy: a degrading backend trips on a *burst* of failures, not a slow
    /// trickle (see `circuit_breaker` crate docs). Each provider gets its own breaker built from this
    /// (see `state::build_providers`).
    pub fn circuit_breaker_config(&self) -> Option<crate::circuit_breaker::CircuitBreakerConfig> {
        if self.circuit_breaker_threshold == 0 {
            return None;
        }
        Some(
            crate::circuit_breaker::CircuitBreakerConfig::windowed(
                self.circuit_breaker_threshold,
                std::time::Duration::from_secs(self.circuit_breaker_window_secs),
            )
            .reset_timeout(std::time::Duration::from_secs(
                self.circuit_breaker_reset_secs,
            ))
            // Admit exactly **one** half-open probe (matching ARCHITECTURE.md: "a probe is
            // admitted"). The library default is 3, but for an egress breaker a single probe is the
            // conservative choice — one request tests a possibly-still-broken provider, and a success
            // closes it immediately for full traffic; 3 concurrent probes would send 3× the load at a
            // provider we believe is down. Deliberately not a config knob: minimum effective surface.
            .half_open_permits(1),
        )
    }

    /// Fold the secret-carrying env prefixes into the config, in a single pass.
    ///
    /// `AI_POOL_KEY_<NAME>` → `pool_keys[name]` (provider name lowercased),
    /// `AI_SIGNING_KEY_<KID>` → `signing_keys[kid]` and `AI_ID_SIGNING_KEY_<KID>` →
    /// `id_signing_keys[kid]` (key ids verbatim). This is the production secret path: all are map
    /// fields a flat figment env merge can't target, the ECS container has no
    /// mounted config file, and env must win over anything baked into one.
    ///
    /// `std::env::vars()` allocates a `(String, String)` for *every* variable in the environment,
    /// so folding the two prefixes separately walked and heap-copied the whole environment twice —
    /// including the long base64 pool keys and `.creds` blobs SSM injects, into `String`s that are
    /// then dropped **un-zeroized**. One pass halves both the allocations and the number of
    /// plaintext credential copies left in freed heap.
    fn merge_secret_env(&mut self, vars: impl Iterator<Item = (String, String)>) {
        for (k, v) in vars {
            if let Some(name) = k.strip_prefix("AI_POOL_KEY_") {
                let name = self.pool_key_env_provider(name);
                self.pool_keys.insert(name, Secret::new(v).into());
            } else if let Some(kid) = k.strip_prefix("AI_SIGNING_KEY_") {
                self.signing_keys.insert(kid.to_string(), v);
            } else if let Some(kid) = k.strip_prefix("AI_ID_SIGNING_KEY_") {
                self.id_signing_keys.insert(kid.to_string(), Secret::new(v));
            }
        }
    }

    /// The provider an `AI_POOL_KEY_<NAME>` variable is for.
    ///
    /// An environment variable name cannot hold `-`, so `<NAME>` is lowercased, and when that names
    /// no provider but its `_` → `-` spelling does, the hyphenated provider wins:
    /// `AI_POOL_KEY_FIREWORKS_ANTHROPIC` reaches a config-added `fireworks-anthropic`. A name that
    /// matches exactly always wins, so a provider really called `my_vendor` keeps its key. (The
    /// mapping is mechanical; it does not make a provider poolable. `openai-codex` is reached only
    /// with a client's own ChatGPT bearer and has no pool key: see `providers::ProviderId`.)
    fn pool_key_env_provider(&self, env_name: &str) -> String {
        let name = env_name.to_ascii_lowercase();
        let known = |n: &str| {
            crate::route::known_providers().any(|p| p.name == n)
                || self.provider_authorities.contains_key(n)
        };
        if known(&name) || !name.contains('_') {
            return name;
        }
        let hyphenated = name.replace('_', "-");
        if known(&hyphenated) { hyphenated } else { name }
    }

    /// Build the trusted keyring from the configured signing public keys.
    pub fn build_keyring(&self) -> Result<Keyring> {
        let mut ring = Keyring::new();
        for (kid_str, b64) in &self.signing_keys {
            let kid: Kid = kid_str
                .parse()
                .map_err(|_| GatewayError::Config(format!("invalid signing key id {kid_str}")))?;
            let vk = crate::key::verifying_key_from_value(b64.as_bytes()).ok_or_else(|| {
                GatewayError::Config(format!("invalid signing public key for kid {kid}"))
            })?;
            ring.insert(kid, vk);
        }
        Ok(ring)
    }
}

/// Environment variables as `(String, String)`, skipping any that aren't valid UTF-8.
///
/// `std::env::vars()` **panics** on a non-UTF-8 variable. Nothing we read could be non-UTF-8 and be
/// meaningful, so skipping is right — but panicking during boot because some unrelated variable in
/// the container's environment has odd bytes is not.
fn env_pairs() -> impl Iterator<Item = (String, String)> {
    std::env::vars_os().filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
}

/// A figment provider whose data has already been read and parsed.
///
/// figment's own providers re-do their work on every `data()` call — `Data::data()` calls
/// `F::from_path` each time, with no caching — and figment calls `data()` once per merge. Wrapping
/// the parsed result lets the same bytes be inspected *and* merged without reading the file, or
/// re-serializing the defaults, a second time.
struct PreRead {
    meta: figment::Metadata,
    data: figment::value::Map<figment::Profile, figment::value::Dict>,
}

impl PreRead {
    /// The top-level keys across every profile.
    fn keys(&self) -> std::collections::BTreeSet<String> {
        self.data
            .values()
            .flat_map(|dict| dict.keys().cloned())
            .collect()
    }
}

impl figment::Provider for PreRead {
    fn metadata(&self) -> figment::Metadata {
        self.meta.clone()
    }
    fn data(
        &self,
    ) -> figment::error::Result<figment::value::Map<figment::Profile, figment::value::Dict>> {
        Ok(self.data.clone())
    }
}

/// Read a provider's data once, up front.
fn pre_read<P: figment::Provider>(p: P) -> Result<PreRead> {
    let data = p.data().map_err(|e| GatewayError::Config(e.to_string()))?;
    Ok(PreRead {
        meta: p.metadata(),
        data,
    })
}

/// Read and parse the TOML config once.
///
/// `data()` errors two ways we must distinguish: a *missing* file is benign (the gateway runs on
/// defaults + env), but a *malformed* file is a hard error we must surface here with the file named
/// — otherwise the syntax error only reappears later as an opaque Figment `extract()` failure with
/// no path attribution. A missing file yields no keys and passes; any other error (parse failure,
/// permission denied) fails the load loudly.
fn read_toml(path: &Path) -> Result<PreRead> {
    use figment::Provider as _;
    let provider = Toml::file(path);
    match provider.data() {
        Ok(data) => Ok(PreRead {
            meta: provider.metadata(),
            data,
        }),
        Err(e) if path.exists() => Err(GatewayError::Config(format!(
            "failed to parse {}: {e}",
            path.display()
        ))),
        // No file (or it vanished between checks): nothing to validate — defaults + env apply.
        Err(_) => Ok(PreRead {
            meta: provider.metadata(),
            data: figment::value::Map::new(),
        }),
    }
}

impl AiConfig {
    /// The upstream HTTP/2 PING interval (see `h2_ping_interval_secs`).
    pub fn h2_ping_interval(&self) -> Option<Duration> {
        (self.h2_ping_interval_secs > 0).then(|| Duration::from_secs(self.h2_ping_interval_secs))
    }

    /// TCP keepalive for an upstream connection: the probes, plus `TCP_USER_TIMEOUT` at the same
    /// bound so unacknowledged request bytes (where keepalive does not probe) fail as fast.
    pub fn upstream_tcp_keepalive(&self) -> Option<TcpKeepalive> {
        self.tcp_keepalive(true)
    }

    /// TCP keepalive for an accepted client connection. No `TCP_USER_TIMEOUT`: a live client that
    /// stops reading (zero window) is `client_write_timeout_secs`'s to judge, not the kernel's.
    pub fn downstream_tcp_keepalive(&self) -> Option<TcpKeepalive> {
        self.tcp_keepalive(false)
    }

    fn tcp_keepalive(&self, user_timeout: bool) -> Option<TcpKeepalive> {
        if self.tcp_keepalive_idle_secs == 0 {
            return None;
        }
        let idle = Duration::from_secs(self.tcp_keepalive_idle_secs);
        let interval = Duration::from_secs(self.tcp_keepalive_interval_secs);
        let count = self.tcp_keepalive_count as usize;
        #[cfg(not(target_os = "linux"))]
        let _ = user_timeout;
        Some(TcpKeepalive {
            idle,
            interval,
            count,
            #[cfg(target_os = "linux")]
            user_timeout: if user_timeout {
                idle + interval * self.tcp_keepalive_count
            } else {
                Duration::ZERO
            },
        })
    }
}

/// Fail the load if the config file carries any key that isn't an `AiConfig` field.
///
/// `known` is derived from `AiConfig` itself by serializing its defaults, so it tracks the struct
/// automatically and can never drift from the field list. See the `deny_unknown_fields` note on
/// `AiConfig` for why this is scoped to the TOML file and not the env layer.
fn reject_unknown_toml_keys(path: &Path, defaults: &PreRead, toml: &PreRead) -> Result<()> {
    let known = defaults.keys();
    let unknown: Vec<String> = toml
        .keys()
        .into_iter()
        .filter(|k| !known.contains(k))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(GatewayError::Config(format!(
        "unknown key(s) in {}: {} — check for a typo (known keys: {})",
        path.display(),
        unknown.join(", "),
        known.into_iter().collect::<Vec<_>>().join(", "),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = AiConfig::default();
        // Read timeout must comfortably exceed a long stream.
        assert!(c.read_timeout_secs >= 300);
        assert_eq!(c.config_bucket, "ai-gateway");
    }

    #[test]
    fn loads_without_a_file() {
        let c = AiConfig::load_with_path(None).unwrap();
        assert_eq!(c.listen, "0.0.0.0:8080");
    }

    #[test]
    fn validate_rejects_a_breaker_threshold_the_count_cannot_hold() {
        let at = |t| AiConfig {
            circuit_breaker_threshold: t,
            ..Default::default()
        };
        let max = crate::circuit_breaker::MAX_FAILURE_THRESHOLD;
        assert!(at(max).validate().is_ok());
        assert!(at(max + 1).validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_connect_and_read_timeouts() {
        // A 0 connect/read timeout (a typo'd SSM param) must fail boot loudly, not degrade into a
        // 502 cascade at runtime.
        assert!(
            AiConfig {
                connect_timeout_secs: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            AiConfig {
                read_timeout_secs: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        // Defaults are valid.
        assert!(AiConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_zero_breaker_window_or_reset() {
        let cfg = |window, reset, threshold| AiConfig {
            circuit_breaker_window_secs: window,
            circuit_breaker_reset_secs: reset,
            circuit_breaker_threshold: threshold,
            ..Default::default()
        };
        assert!(cfg(0, 30, 20).validate().is_err());
        assert!(cfg(10, 0, 20).validate().is_err());
        // A disabled breaker has no window to get wrong.
        assert!(cfg(0, 0, 0).validate().is_ok());
    }

    #[test]
    fn pool_key_env_reaches_a_hyphenated_provider() {
        let mut c = AiConfig {
            provider_authorities: HashMap::from([
                ("fireworks-anthropic".to_string(), "h:443".to_string()),
                ("my_vendor".to_string(), "h:443".to_string()),
            ]),
            ..Default::default()
        };
        c.merge_secret_env(
            [
                (
                    "AI_POOL_KEY_FIREWORKS_ANTHROPIC".to_string(),
                    "a".to_string(),
                ),
                ("AI_POOL_KEY_OPENAI_CODEX".to_string(), "b".to_string()),
                ("AI_POOL_KEY_MY_VENDOR".to_string(), "c".to_string()),
                ("AI_POOL_KEY_UNKNOWN_THING".to_string(), "d".to_string()),
            ]
            .into_iter(),
        );
        assert_eq!(c.pool_keys["fireworks-anthropic"][0].expose(), "a");
        assert_eq!(c.pool_keys["openai-codex"][0].expose(), "b");
        assert_eq!(
            c.pool_keys["my_vendor"][0].expose(),
            "c",
            "an exact match wins"
        );
        assert_eq!(c.pool_keys["unknown_thing"][0].expose(), "d");
    }

    /// Write `body` to a uniquely-named temp TOML file (the literal `label` keeps parallel tests
    /// from colliding) and return its path; the caller removes it.
    fn temp_toml(label: &str, body: &str) -> std::path::PathBuf {
        use std::io::Write as _;
        let path = std::env::temp_dir().join(format!("beyond-ai-cfg-{label}.toml"));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    #[test]
    fn rejects_typod_toml_key() {
        // A misspelled key in the operator's own TOML is a silent footgun (loads its default, the
        // setting does nothing) — load must fail loudly and name the offending key, not boot healthy.
        let path = temp_toml(
            "typo",
            "listen = \"0.0.0.0:1234\"\nreqiure_signing_keys = true\n",
        );
        let err = AiConfig::load_with_path(Some(&path)).unwrap_err();
        let _ = std::fs::remove_file(&path);
        match err {
            GatewayError::Config(msg) => assert!(
                msg.contains("reqiure_signing_keys"),
                "error must name the typo'd key, got: {msg}"
            ),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_malformed_toml_with_path_attribution() {
        // A syntax error in the operator's TOML must fail the load *here*, naming the file — not
        // silently pass this check and resurface later as an opaque Figment extract() error.
        let path = temp_toml("malformed", "listen = \"unterminated\nrate_limit_rps = 7\n");
        let err = AiConfig::load_with_path(Some(&path)).unwrap_err();
        let _ = std::fs::remove_file(&path);
        match err {
            GatewayError::Config(msg) => assert!(
                msg.contains("malformed") || msg.contains("parse"),
                "error must indicate a parse failure and name the file, got: {msg}"
            ),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn accepts_known_toml_keys() {
        // Every key here is a real `AiConfig` field (including the `[signing_keys]` table) — load
        // must succeed and apply the values.
        let path = temp_toml(
            "known",
            "listen = \"0.0.0.0:1234\"\nrequire_signing_keys = true\nrate_limit_rps = 7\n\n[signing_keys]\n1 = \"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\"\n",
        );
        let c = AiConfig::load_with_path(Some(&path)).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(c.listen, "0.0.0.0:1234");
        assert!(c.require_signing_keys);
        assert_eq!(c.rate_limit_rps, 7);
        assert!(c.signing_keys.contains_key("1"));
    }

    #[test]
    fn build_keyring_rejects_non_numeric_kid() {
        // `kid` is parsed as `u32`; a non-numeric map key must fail boot (loud) rather than
        // silently drop a trusted signing key (which would 401 every token under it).
        let c = AiConfig {
            signing_keys: HashMap::from([("not-a-number".to_string(), "AAAA".to_string())]),
            ..Default::default()
        };
        assert!(c.build_keyring().is_err());
    }

    #[test]
    fn build_keyring_rejects_invalid_public_key() {
        // A value that is neither raw 32 bytes nor base64 of 32 bytes must fail boot, not install a
        // bogus key that can never verify anything.
        let c = AiConfig {
            signing_keys: HashMap::from([("1".to_string(), "!!! not base64 !!!".to_string())]),
            ..Default::default()
        };
        assert!(c.build_keyring().is_err());
    }

    #[test]
    fn config_file_is_read_exactly_once() {
        // Validating the key set and then merging used to open, read and parse the file twice —
        // and, worse than the wasted parse, at two different instants, so a file rewritten in
        // between produced a config that had been validated against different content than it
        // loaded. Prove single-read by making the *second* read return something the first didn't:
        // load, then delete the file mid-flight is untestable, so instead assert the parsed data is
        // carried through rather than re-fetched — a second read of a now-unknown-key file would
        // fail, and a second read of a now-missing file would silently drop the values.
        let dir = std::env::temp_dir().join(format!("ai-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "read_timeout_secs = 1234\nconfig_bucket = \"from-file\"\n",
        )
        .unwrap();

        let defaults =
            pre_read(figment::providers::Serialized::defaults(AiConfig::default())).unwrap();
        let toml = read_toml(&path).unwrap();
        // Everything downstream now works off `toml`, so removing the file must not change the
        // outcome — which is only true if nothing re-reads it.
        std::fs::remove_file(&path).unwrap();
        reject_unknown_toml_keys(&path, &defaults, &toml).unwrap();
        let cfg: AiConfig = Figment::from(defaults).merge(toml).extract().unwrap();
        assert_eq!(cfg.read_timeout_secs, 1234);
        assert_eq!(cfg.config_bucket, "from-file");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pool_key_env_merges_and_overrides() {
        // `AI_POOL_KEY_<NAME>` → `pool_keys[name]` (lowercased), and env wins over a config-file
        // value (the production secret path). A non-pool `AI_*` var is ignored.
        let mut c = AiConfig {
            pool_keys: HashMap::from([("openai".to_string(), "from-file".into())]),
            ..Default::default()
        };
        c.merge_secret_env(
            [
                ("AI_POOL_KEY_OPENAI".to_string(), "from-env".to_string()),
                ("AI_POOL_KEY_GROQ".to_string(), "gsk-x".to_string()),
                ("AI_LOG".to_string(), "debug".to_string()),
            ]
            .into_iter(),
        );
        assert_eq!(c.pool_keys.get("openai").unwrap()[0].expose(), "from-env");
        assert_eq!(c.pool_keys.get("groq").unwrap()[0].expose(), "gsk-x");
        assert_eq!(c.pool_keys.get("openai").unwrap().len(), 1);
        assert!(!c.pool_keys.contains_key("log"));
    }

    #[test]
    fn pool_keys_toml_array_or_string() {
        // TOML array is N keys; a TOML string is a list of one (same as env).
        let dir = std::env::temp_dir().join(format!("ai-pool-keys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            "[pool_keys]\n\
             openai = [\"sk-a\", \"sk-b\"]\n\
             anthropic = \"sk-ant\"\n\
             empty = []\n",
        )
        .unwrap();
        let defaults =
            pre_read(figment::providers::Serialized::defaults(AiConfig::default())).unwrap();
        let toml = read_toml(&path).unwrap();
        let cfg: AiConfig = Figment::from(defaults).merge(toml).extract().unwrap();
        let openai = cfg.pool_keys.get("openai").unwrap();
        assert_eq!(openai.len(), 2);
        assert_eq!(openai[0].expose(), "sk-a");
        assert_eq!(openai[1].expose(), "sk-b");
        assert_eq!(
            cfg.pool_keys.get("anthropic").unwrap()[0].expose(),
            "sk-ant"
        );
        assert!(cfg.pool_keys.get("empty").unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn signing_key_env_merges_and_overrides() {
        // `AI_SIGNING_KEY_<KID>` → `signing_keys[kid]`, env wins over a config-file value (the prod
        // ECS path where the gateway has no mounted config). Non-signing `AI_*` vars are ignored.
        let mut c = AiConfig {
            signing_keys: HashMap::from([("1".to_string(), "from-file".to_string())]),
            ..Default::default()
        };
        c.merge_secret_env(
            [
                ("AI_SIGNING_KEY_1".to_string(), "from-env".to_string()),
                ("AI_SIGNING_KEY_2".to_string(), "second-kid".to_string()),
                ("AI_POOL_KEY_OPENAI".to_string(), "sk-x".to_string()),
            ]
            .into_iter(),
        );
        assert_eq!(c.signing_keys.get("1").unwrap(), "from-env");
        assert_eq!(c.signing_keys.get("2").unwrap(), "second-kid");
        assert!(!c.signing_keys.contains_key("OPENAI"));
    }
}
