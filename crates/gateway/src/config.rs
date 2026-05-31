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
// `default` so every field is optional. We deliberately do NOT set `deny_unknown_fields`: config is
// merged from `Env::prefixed("AI_")`, a namespace shared with foreign variables the platform injects
// (e.g. `AI_AGENT`, `AI_LOG`), so rejecting unknown keys would fail load on a valid environment
// rather than catch a typo.
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

    /// Managed Beyond pool keys, **by provider name** (`openai`, `anthropic`, `fireworks`, …).
    /// From the `[pool_keys]` TOML table or SSM-injected `AI_POOL_KEY_<NAME>` env (the env form is
    /// the production path — see `load_with_path`). A provider with no pool key here can't serve
    /// managed traffic (→ 503); BYO is unaffected. Values are `Secret` so a key can't leak through
    /// the `Debug`/`Serialize` this struct derives; read the plaintext via `expose` at the use site.
    pub pool_keys: HashMap<String, Secret>,

    /// Per-provider upstream authority (`host:port`), **by provider name**. For a known provider
    /// (see `route::KNOWN_PROVIDERS`) this *overrides* its default; for an unknown name it *adds* a
    /// new OpenAI-wire provider reachable via `x-beyond-provider`. Empty = every known provider uses
    /// its built-in default. (The e2e harness points providers at a mock here.)
    pub provider_authorities: HashMap<String, String>,

    /// Upstream timeouts (seconds). Streaming responses are long, so read/idle are generous.
    pub connect_timeout_secs: u64,
    pub read_timeout_secs: u64,
    pub write_timeout_secs: u64,
    pub idle_timeout_secs: u64,

    /// TLS to the upstream provider. Real providers are HTTPS (true); the e2e harness sets false
    /// to talk to a plaintext mock.
    pub upstream_tls: bool,

    /// Per-key request-rate ceiling (requests/sec). A blast-radius guardrail (see `ratelimit`), not
    /// a spend control: it caps how fast a single tenant (managed) or BYO caller can drive the
    /// gateway, bounding a leaked/runaway key during the deny-set's reaction lag and a failure flood
    /// that never bills. `0` disables it. The default is generous — a circuit breaker, not a quota;
    /// tune from `ai_rejections_total{reason="rate_limit"}`.
    pub rate_limit_rps: u32,
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
            pool_keys: HashMap::new(),
            provider_authorities: HashMap::new(),
            connect_timeout_secs: 10,
            // Generous: LLM streams can run for minutes; a tight read timeout would kill them.
            read_timeout_secs: 600,
            write_timeout_secs: 60,
            idle_timeout_secs: 90,
            upstream_tls: true,
            // Generous per-key circuit breaker, on by default. Won't touch legitimate steady-state
            // traffic; caps a runaway/leaked key or a retry-storm flood. Set 0 to disable.
            rate_limit_rps: 100,
        }
    }
}

impl AiConfig {
    pub fn load_with_path(path: Option<&Path>) -> Result<Self> {
        let mut fig = Figment::from(figment::providers::Serialized::defaults(AiConfig::default()));
        fig = fig.merge(Toml::file(path.unwrap_or_else(|| Path::new("config.toml"))));
        // Flat mapping: `AI_READ_TIMEOUT_SECS` → `read_timeout_secs`. (No `.split('_')` — these are
        // flat fields, not nested tables.) Unknown `AI_*` vars are tolerated (see the
        // `deny_unknown_fields` note on `AiConfig`) — which is also why pool keys are collected
        // separately below rather than via this flat merge.
        fig = fig.merge(Env::prefixed("AI_"));
        let mut cfg: AiConfig = fig
            .extract()
            .map_err(|e| GatewayError::Config(e.to_string()))?;
        cfg.merge_pool_key_env(std::env::vars());
        Ok(cfg)
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
}
