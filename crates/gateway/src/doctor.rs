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
use tokio::task::JoinSet;

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

/// How long we wait on any one provider's DNS before calling it unreachable. Because the lookups
/// overlap (see below), this is the *total* the doctor can spend on DNS, not the per-provider budget.
const DNS_TIMEOUT: Duration = Duration::from_secs(3);

/// Resolve every provider authority the gateway might dial (known providers + config overrides/adds),
/// so a DNS or typo'd-authority problem shows up here rather than as a 502 on the first request. Each
/// lookup is bounded so one black-holed host can't hang the doctor. We don't connect (no auth, no TLS
/// handshake) — reachability of the *name* is the prerequisite; live auth is proven by the smoke test.
///
/// The lookups run **concurrently**. Sequentially the bound is `providers × DNS_TIMEOUT` — ~30s of
/// nothing when DNS is black-holed, which is precisely the situation you run `doctor` in. `lookup_host`
/// hands the blocking `getaddrinfo` to the runtime's blocking pool, so they genuinely overlap even on
/// the current-thread runtime `main` builds, and the wall clock collapses to one timeout. Results are
/// sorted back into provider order on the way out: the output is read by a human (and asserted in
/// tests), so it must not depend on which lookup happened to finish first.
async fn check_provider_dns(config: &AiConfig) -> Vec<CheckResult> {
    // Effective authority per provider name: the known default unless config overrides it, plus any
    // config-only provider. A BTreeMap dedups and keeps the output stable/ordered.
    //
    // `CheckResult.name` is `&'static str`: a known provider lends its static name — carried in the
    // value straight off the registry row, so we never re-scan the registry per provider — while a
    // config-only provider (non-'static) reports under a generic label, real name in the message.
    let mut authorities: BTreeMap<&str, (&'static str, &str)> = BTreeMap::new();
    for spec in route::known_providers() {
        authorities.insert(spec.name, (spec.name, spec.authority));
    }
    for (name, authority) in &config.provider_authorities {
        authorities
            .entry(name.as_str())
            .and_modify(|slot| slot.1 = authority.as_str())
            .or_insert(("provider_dns", authority.as_str()));
    }

    let mut lookups = JoinSet::new();
    for (name, (check_name, authority)) in authorities {
        // A spawned task is `'static`, so it owns its strings: one allocation per provider, which is
        // what buys the overlap. Everything above this point stays borrowed.
        let (name, authority) = (name.to_string(), authority.to_string());
        lookups.spawn(async move {
            let lookup =
                tokio::time::timeout(DNS_TIMEOUT, tokio::net::lookup_host(authority.as_str())).await;
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
                    format!(
                        "{name}: {authority}: DNS lookup timed out (>{}s)",
                        DNS_TIMEOUT.as_secs()
                    ),
                    "the upstream host may be unreachable or DNS is slow",
                ),
            };
            (name, res)
        });
    }

    let mut results = Vec::with_capacity(lookups.len());
    while let Some(joined) = lookups.join_next().await {
        results.push(match joined {
            Ok(entry) => entry,
            // Nothing aborts the set, so a join error means the task panicked. Report it instead of
            // silently dropping a provider — a doctor that under-reports is worse than one that says
            // something went wrong.
            Err(e) => (
                String::new(),
                fail(
                    "provider_dns",
                    format!("a provider DNS check did not run: {e}"),
                    "this is a gateway bug — please report it",
                ),
            ),
        });
    }
    // `join_next` yields in completion order; restore the provider order the BTreeMap defined.
    results.sort_by(|(a, _), (b, _)| a.cmp(b));
    results.into_iter().map(|(_, res)| res).collect()
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

    /// Two properties of `check_provider_dns` at once:
    ///
    /// 1. **Order is deterministic** — the report follows provider-name order no matter which lookup
    ///    finished first. That's the part the concurrency rewrite could break silently.
    /// 2. **The lookups overlap** — N of them cost about one lookup, not N.
    ///
    /// Every authority is overridden to a `.invalid` name (RFC 2606: reserved, never resolves), so the
    /// run is all negative lookups: no live provider DNS in a unit test, and no chance of a name whose
    /// success path (RFC 3484 address sorting) serializes inside glibc and muddies the timing.
    ///
    /// (2) is measured against a baseline taken in the same test, because a negative lookup costs
    /// anywhere from ~20ms (real resolver round trip) to microseconds (no resolver reachable — the
    /// offline-sandbox case). Below `MIN_BASELINE` the sequential and concurrent regimes are
    /// indistinguishable from scheduling noise, so we skip the timing claim rather than ship a flaky
    /// assertion; (1) still runs.
    #[tokio::test]
    async fn provider_dns_lookups_overlap_and_report_in_order() {
        use std::time::Instant;
        const MIN_BASELINE: Duration = Duration::from_millis(2);

        let mut authorities: HashMap<String, String> = route::known_providers()
            .map(|s| (s.name.to_string(), format!("zz-{}.invalid:443", s.name)))
            .collect();
        // Plus a few config-only providers, so the run covers both `CheckResult.name` branches.
        authorities.extend((0..4).map(|i| (format!("zz-extra-{i}"), format!("zz-{i}.invalid:443"))));
        let expected = authorities.len();
        let c = AiConfig {
            provider_authorities: authorities,
            ..Default::default()
        };

        let t = Instant::now();
        let _ = tokio::net::lookup_host("zz-baseline.invalid:443").await;
        let baseline = t.elapsed();

        let t = Instant::now();
        let results = check_provider_dns(&c).await;
        let elapsed = t.elapsed();

        // Every provider reported exactly once, in name order. The name leads each message, ahead of
        // either ` →` (pass) or `:` (fail).
        assert_eq!(results.len(), expected);
        let names: Vec<&str> = results
            .iter()
            .map(|r| r.message.split([' ', ':']).next().unwrap_or_default())
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "doctor output must be in provider order");

        if baseline >= MIN_BASELINE {
            let sequential = baseline * expected as u32;
            assert!(
                elapsed < sequential / 2,
                "{expected} lookups took {elapsed:?}; sequential would be ≈{sequential:?} \
                 (one lookup = {baseline:?}) — they are not overlapping",
            );
        }
    }
}
