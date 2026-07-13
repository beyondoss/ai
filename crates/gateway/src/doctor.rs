//! Diagnostics (PATTERNS.md `doctor` pattern): fast prerequisite checks, exit 0/1.
//!
//! The point is to catch a misconfiguration *before* traffic lands on the instance, where it would
//! otherwise surface as a first-request failure (a 401 from an empty keyring, a 503 from a missing
//! pool key, a 502 from an unresolvable provider). We check the things boot does lazily or never:
//! NATS reachability, the signing keyring, managed pool keys, and provider DNS.

use crate::config::AiConfig;
use crate::route;
use std::collections::BTreeMap;
use std::time::Duration;

pub struct CheckResult {
    pub name: &'static str,
    pub passed: bool,
    pub message: String,
    pub hint: Option<String>,
}

fn pass(name: &'static str, message: impl Into<String>) -> CheckResult {
    CheckResult {
        name,
        passed: true,
        message: message.into(),
        hint: None,
    }
}

fn fail(name: &'static str, message: impl Into<String>, hint: &str) -> CheckResult {
    CheckResult {
        name,
        passed: false,
        message: message.into(),
        hint: Some(hint.to_string()),
    }
}

pub async fn run_checks(config: &AiConfig) -> Vec<CheckResult> {
    let mut out = Vec::new();

    // NATS / slipstream reachability — without it we can't load signing keys or the deny-set.
    match store::nats_connect(
        &config.nats_url,
        config.nats_creds.as_ref().map(|s| s.expose()),
        config.nats_creds_file.as_deref(),
    )
    .await
    {
        Ok(_) => out.push(pass("nats", format!("connected to {}", config.nats_url))),
        Err(e) => out.push(fail(
            "nats",
            e.to_string(),
            "check AI_NATS_URL and credentials",
        )),
    }

    out.push(check_signing_keys(config));
    out.push(check_pool_keys(config));
    out.extend(check_provider_dns(config).await);

    out
}

/// The signing keyring is what authenticates managed traffic. An empty or invalid keyring isn't a
/// hard boot failure (the gateway still serves BYO), but it silently turns *every* `bai_…` key into a
/// 401 — a footgun worth surfacing loudly here. `build_keyring` already rejects a non-numeric kid or
/// an unparseable public key, so a success means every configured key installed.
fn check_signing_keys(config: &AiConfig) -> CheckResult {
    match config.build_keyring() {
        Ok(ring) if ring.is_empty() => fail(
            "signing_keys",
            "no signing keys configured — all managed (bai_…) traffic will 401, only BYO works",
            "set [signing_keys] (kid → base64 Ed25519 public key) in config or AI_ env",
        ),
        Ok(ring) => pass(
            "signing_keys",
            format!("{} signing key(s) loaded", ring.len()),
        ),
        Err(e) => fail(
            "signing_keys",
            e.to_string(),
            "every kid must be numeric and every value a base64 (or raw 32-byte) Ed25519 public key",
        ),
    }
}

/// Pool keys back managed traffic (swapped in per provider). Cross-check against the keyring: if
/// signing keys are present the operator *intends* to serve managed traffic, so zero pool keys means
/// every managed request 503s — a real misconfiguration. A pure-BYO deployment (no signing keys) with
/// no pool keys is legitimate, so that case passes with a note instead of failing.
fn check_pool_keys(config: &AiConfig) -> CheckResult {
    let mut names: Vec<&str> = config.pool_keys.keys().map(String::as_str).collect();
    names.sort_unstable();
    let managed_intended = !config.signing_keys.is_empty();
    match (names.is_empty(), managed_intended) {
        (true, true) => fail(
            "pool_keys",
            "signing keys are configured (managed traffic expected) but no pool keys are set — \
             every managed request will 503",
            "set AI_POOL_KEY_<PROVIDER> (e.g. AI_POOL_KEY_OPENAI) for each provider you serve",
        ),
        (true, false) => pass(
            "pool_keys",
            "none configured (BYO-only deployment — no signing keys either)",
        ),
        (false, _) => pass("pool_keys", format!("pool keys for: {}", names.join(", "))),
    }
}

/// Resolve every provider authority the gateway might dial (known providers + config overrides/adds),
/// so a DNS or typo'd-authority problem shows up here rather than as a 502 on the first request. Each
/// lookup is bounded so one black-holed host can't hang the doctor. We don't connect (no auth, no TLS
/// handshake) — reachability of the *name* is the prerequisite; live auth is proven by the smoke test.
async fn check_provider_dns(config: &AiConfig) -> Vec<CheckResult> {
    // Effective authority per provider name: the known default unless config overrides it, plus any
    // config-only provider. A BTreeMap dedups and keeps the output stable/ordered.
    let mut authorities: BTreeMap<&str, String> = BTreeMap::new();
    for spec in route::known_providers() {
        authorities.insert(spec.name, spec.authority.to_string());
    }
    for (name, authority) in &config.provider_authorities {
        authorities.insert(name.as_str(), authority.clone());
    }

    let mut results = Vec::with_capacity(authorities.len());
    for (name, authority) in authorities {
        // `CheckResult.name` is `&'static str`: a known provider lends its static name; a config-only
        // provider (non-'static) reports under a generic label, with the real name in the message.
        let check_name: &'static str = route::known_providers()
            .find(|s| s.name == name)
            .map_or("provider_dns", |s| s.name);
        let lookup = tokio::time::timeout(
            Duration::from_secs(3),
            tokio::net::lookup_host(authority.clone()),
        )
        .await;
        let res = match lookup {
            Ok(Ok(mut addrs)) => match addrs.next() {
                Some(addr) => pass(check_name, format!("{name} → {authority} ({addr})")),
                None => fail(
                    check_name,
                    format!("{name}: {authority} resolved to no addresses"),
                    "check the provider authority (host:port) in provider_authorities",
                ),
            },
            Ok(Err(e)) => fail(
                check_name,
                format!("{name}: {authority}: {e}"),
                "check the provider authority (host:port) and DNS",
            ),
            Err(_) => fail(
                check_name,
                format!("{name}: {authority}: DNS lookup timed out (>3s)"),
                "the upstream host may be unreachable or DNS is slow",
            ),
        };
        results.push(res);
    }
    results
}

pub fn print_results(title: &str, results: &[CheckResult]) {
    println!("== {title} ==");
    for r in results {
        let mark = if r.passed { "ok" } else { "FAIL" };
        println!("[{mark}] {}: {}", r.name, r.message);
        if let (false, Some(hint)) = (r.passed, &r.hint) {
            println!("       hint: {hint}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::Secret;
    use std::collections::HashMap;

    #[test]
    fn signing_keys_empty_fails() {
        // No keys ⇒ every managed token 401s; doctor must flag it, not pass silently.
        let c = AiConfig::default();
        assert!(!check_signing_keys(&c).passed);
    }

    #[test]
    fn signing_keys_valid_passes() {
        let c = AiConfig {
            // 32 zero bytes, base64 — a structurally valid Ed25519 public key.
            signing_keys: HashMap::from([(
                "1".to_string(),
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_string(),
            )]),
            ..Default::default()
        };
        assert!(check_signing_keys(&c).passed);
    }

    #[test]
    fn pool_keys_missing_with_signing_keys_fails() {
        // Signing keys present (managed intended) but no pool keys ⇒ every managed request 503s.
        let c = AiConfig {
            signing_keys: HashMap::from([(
                "1".to_string(),
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_string(),
            )]),
            ..Default::default()
        };
        assert!(!check_pool_keys(&c).passed);
    }

    #[test]
    fn pool_keys_absent_byo_only_passes() {
        // No signing keys and no pool keys is a legitimate BYO-only deployment — must not fail.
        assert!(check_pool_keys(&AiConfig::default()).passed);
    }

    #[test]
    fn pool_keys_present_passes() {
        let c = AiConfig {
            pool_keys: HashMap::from([("openai".to_string(), Secret::new("sk-x"))]),
            ..Default::default()
        };
        assert!(check_pool_keys(&c).passed);
    }
}
