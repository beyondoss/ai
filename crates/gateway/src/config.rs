//! Layered configuration (PATTERNS.md: Figment defaults → TOML → `AI_`-prefixed env).
//!
//! Auth + key material come from config (signing public keys, managed pool keys), so the gateway
//! is fully functional from boot config alone — NATS is only needed for the deny-set.

use crate::error::{GatewayError, Result};
use crate::key::{Keyring, Kid};
use crate::secret::Secret;
use figment::Figment;
use figment::providers::{Env, Format, Toml};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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

    /// NATS / slipstream connection (cf. `_envcommon/ecs-service.hcl`: `tls://connect.ngs.global`).
    /// Used only for the watched deny-set (`blackhole.*`).
    pub nats_url: String,
    /// Base64 `.creds` (ECS via SOPS) — takes priority over `nats_creds_file`. Held in `Secret` so
    /// it can't leak through the `Debug`/`Serialize` this struct derives (a stray `?config` log).
    pub nats_creds: Option<Secret>,
    pub nats_creds_file: Option<String>,
    /// slipstream bucket holding `blackhole.*` (the deny-set — the only thing in NATS).
    pub config_bucket: String,

    /// Optional path to an on-disk deny-set snapshot (slipstream's append-log + resume cursor). When
    /// set **and on durable storage** (the edge/tunnel deployment model), a restart seeds the
    /// deny-set from this file and *resumes the NATS watch from the saved revision* — skipping the
    /// boot scan and surviving a restart with enforcement intact even before NATS reconnects. Unset
    /// (the default, e.g. ephemeral/Fargate) ⇒ seed from a NATS scan each boot, unchanged. The file
    /// is a pure cache: delete it (or point at scratch) and the gateway falls back to scanning.
    pub snapshot_path: Option<String>,

    /// Trusted Ed25519 signing **public** keys: `kid` (as string — TOML/JSON map keys are strings)
    /// → base64 public key. Multiple allowed for zero-downtime rotation. Config, not NATS.
    pub signing_keys: HashMap<String, String>,

    /// Fail the boot if `signing_keys` is empty, instead of degrading to BYO-only. Empty signing
    /// keys is a *legitimate* mode (a BYO-only deployment) but is far more often a mis-deploy — a
    /// typo'd/absent SSM param — that looks healthy while silently dropping **all** managed billing
    /// and deny-set enforcement. A managed deployment should set this `true` so a bad deploy fails
    /// fast and visibly at boot rather than serving for free. Default `false` to keep BYO-only and
    /// the test/e2e harnesses (which run keyless) working out of the box.
    pub require_signing_keys: bool,

    /// Managed Beyond pool keys, **by provider name** (`openai`, `anthropic`, `fireworks`, …).
    /// From the `[pool_keys]` TOML table or SSM-injected `AI_POOL_KEY_<NAME>` env (the env form is
    /// the production path — see `load_with_path`). A provider with no pool key here can't serve
    /// managed traffic (→ 503); BYO is unaffected. Values are `Secret` so a key can't leak through
    /// the `Debug`/`Serialize` this struct derives; read the plaintext via `expose` at the use site.
    pub pool_keys: HashMap<String, Secret>,

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
    pub read_timeout_secs: u64,
    pub write_timeout_secs: u64,
    pub idle_timeout_secs: u64,

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
    /// Slower rollouts are the deliberate price of not mangling responses.
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
    /// the provider is healthy and throttling our pool key (a velocity/spend signal the rate limiter
    /// and the client's `Retry-After` backoff own), so tripping on it would convert a self-healing
    /// throttle into a self-inflicted outage. While open, requests to that provider fast-fail with a
    /// `503` (`ai_rejections_total{reason="circuit_open"}`) instead of piling up against
    /// `read_timeout_secs` and exhausting connection/in-flight slots for *every* provider. After
    /// `circuit_breaker_reset_secs` a probe request is allowed; success closes it, failure reopens it.
    /// Applies to **all** traffic to the provider (managed + BYO) — a down provider is down regardless
    /// of whose key is used. `0` disables the breaker entirely. Default is generous so normal
    /// background 5xx noise never trips it.
    pub circuit_breaker_threshold: u32,
    /// Rolling window (seconds) over which `circuit_breaker_threshold` failures are counted. Failures
    /// older than the window are forgotten — so it trips on a *burst* of failures, not on a slow trickle
    /// spread across a healthy day.
    pub circuit_breaker_window_secs: u64,
    /// How long the breaker stays open before allowing a half-open probe request (seconds). Long enough
    /// to let a provider recover, short enough that recovery is detected promptly.
    pub circuit_breaker_reset_secs: u64,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:8080".to_string(),
            metrics_listen: "0.0.0.0:9090".to_string(),
            nats_url: "nats://localhost:4222".to_string(),
            nats_creds: None,
            nats_creds_file: None,
            config_bucket: "ai-gateway".to_string(),
            snapshot_path: None,
            signing_keys: HashMap::new(),
            require_signing_keys: false,
            pool_keys: HashMap::new(),
            provider_authorities: HashMap::new(),
            provider_dialects: HashMap::new(),
            provider_auth_schemes: HashMap::new(),
            connect_timeout_secs: 10,
            // Generous: LLM streams can run for minutes; a tight read timeout would kill them.
            read_timeout_secs: 600,
            write_timeout_secs: 60,
            idle_timeout_secs: 90,
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
        }
    }
}

impl AiConfig {
    pub fn load_with_path(path: Option<&Path>) -> Result<Self> {
        let toml_path = path.unwrap_or_else(|| Path::new("config.toml"));
        // Catch a typo'd key in the operator's own TOML *before* any of it merges — a misspelled
        // `require_signing_keys` would otherwise load its default and silently drop all managed
        // billing while the gateway looks healthy. Only the TOML file is checked (see the
        // `deny_unknown_fields` note on `AiConfig`); the env layer must stay lenient.
        reject_unknown_toml_keys(toml_path)?;

        let mut fig = Figment::from(figment::providers::Serialized::defaults(AiConfig::default()));
        fig = fig.merge(Toml::file(toml_path));
        // Flat mapping: `AI_READ_TIMEOUT_SECS` → `read_timeout_secs`. (No `.split('_')` — these are
        // flat fields, not nested tables.) Unknown `AI_*` vars are tolerated (see the
        // `deny_unknown_fields` note on `AiConfig`) — which is also why pool keys are collected
        // separately below rather than via this flat merge.
        fig = fig.merge(Env::prefixed("AI_"));
        let mut cfg: AiConfig = fig
            .extract()
            .map_err(|e| GatewayError::Config(e.to_string()))?;
        cfg.merge_pool_key_env(std::env::vars());
        cfg.merge_signing_key_env(std::env::vars());
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
        Ok(())
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

    /// Fold `AI_POOL_KEY_<NAME>` environment variables into `pool_keys` (provider name lowercased).
    /// This is the production secret path (SSM-injected env); a flat figment merge can't target a
    /// map field, and env must win over any `[pool_keys]` value baked into a config file.
    fn merge_pool_key_env(&mut self, vars: impl Iterator<Item = (String, String)>) {
        for (k, v) in vars {
            if let Some(name) = k.strip_prefix("AI_POOL_KEY_") {
                self.pool_keys
                    .insert(name.to_ascii_lowercase(), Secret::new(v));
            }
        }
    }

    /// Fold `AI_SIGNING_KEY_<KID>` environment variables into `signing_keys` (`kid -> base64 pubkey`).
    /// The prod (ECS) secret path: `[signing_keys]` is a TOML table a flat figment env merge can't
    /// target, and the container has no mounted config file — so the trusted pubkey(s) arrive as env,
    /// the same shape as `AI_POOL_KEY_<NAME>`. The `<KID>` suffix is the key id verbatim (e.g.
    /// `AI_SIGNING_KEY_1`); env wins over any `[signing_keys]` baked into a config file.
    fn merge_signing_key_env(&mut self, vars: impl Iterator<Item = (String, String)>) {
        for (k, v) in vars {
            if let Some(kid) = k.strip_prefix("AI_SIGNING_KEY_") {
                self.signing_keys.insert(kid.to_string(), v);
            }
        }
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

/// The set of top-level keys a config file may set, derived from `AiConfig` itself by serializing
/// its defaults — so it tracks the struct automatically and can never drift from the field list.
fn known_config_keys() -> std::collections::BTreeSet<String> {
    use figment::Provider as _;
    figment::providers::Serialized::defaults(AiConfig::default())
        .data()
        .map(|profiles| {
            profiles
                .into_values()
                .flat_map(|dict| dict.into_keys())
                .collect()
        })
        .unwrap_or_default()
}

/// Fail the load if the TOML file at `path` carries any key that isn't an `AiConfig` field. A
/// missing file is fine (the gateway runs on defaults + env), so an unreadable/absent file yields no
/// keys and passes. See the `deny_unknown_fields` note on `AiConfig` for why this is scoped to the
/// TOML file and not the env layer.
fn reject_unknown_toml_keys(path: &Path) -> Result<()> {
    use figment::Provider as _;
    let known = known_config_keys();
    // `data()` errors two ways we must distinguish: a *missing* file is benign (the gateway runs on
    // defaults + env), but a *malformed* file is a hard error we must surface here with the file
    // named — otherwise the syntax error only reappears later as an opaque Figment `extract()`
    // failure with no path attribution. A missing file yields no keys and passes; any other error
    // (parse failure, permission denied) fails the load loudly.
    let unknown: std::collections::BTreeSet<String> = match Toml::file(path).data() {
        Ok(profiles) => profiles
            .into_values()
            .flat_map(|dict| dict.into_keys())
            .filter(|k| !known.contains(k))
            .collect(),
        Err(e) if path.exists() => {
            return Err(GatewayError::Config(format!(
                "failed to parse {}: {e}",
                path.display()
            )));
        }
        // No file (or it vanished between checks): nothing to validate — defaults + env apply.
        Err(_) => return Ok(()),
    };
    if unknown.is_empty() {
        return Ok(());
    }
    let unknown: Vec<String> = unknown.into_iter().collect();
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
    fn pool_key_env_merges_and_overrides() {
        // `AI_POOL_KEY_<NAME>` → `pool_keys[name]` (lowercased), and env wins over a config-file
        // value (the production secret path). A non-pool `AI_*` var is ignored.
        let mut c = AiConfig {
            pool_keys: HashMap::from([("openai".to_string(), Secret::new("from-file"))]),
            ..Default::default()
        };
        c.merge_pool_key_env(
            [
                ("AI_POOL_KEY_OPENAI".to_string(), "from-env".to_string()),
                ("AI_POOL_KEY_GROQ".to_string(), "gsk-x".to_string()),
                ("AI_LOG".to_string(), "debug".to_string()),
            ]
            .into_iter(),
        );
        assert_eq!(c.pool_keys.get("openai").unwrap().expose(), "from-env");
        assert_eq!(c.pool_keys.get("groq").unwrap().expose(), "gsk-x");
        assert!(!c.pool_keys.contains_key("log"));
    }

    #[test]
    fn signing_key_env_merges_and_overrides() {
        // `AI_SIGNING_KEY_<KID>` → `signing_keys[kid]`, env wins over a config-file value (the prod
        // ECS path where the gateway has no mounted config). Non-signing `AI_*` vars are ignored.
        let mut c = AiConfig {
            signing_keys: HashMap::from([("1".to_string(), "from-file".to_string())]),
            ..Default::default()
        };
        c.merge_signing_key_env(
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
