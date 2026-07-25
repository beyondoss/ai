//! Egress safety for the `web` tool — the layer ax (a local human-run CLI) never needs and we cannot
//! do without (we fetch URLs the *model* chose, from a server).
//!
//! Because the `web` tool is read-only and therefore never approval-gated, **this is the sole boundary
//! between the model and an internal fetch.** Its correctness is load-bearing, so it is deliberately
//! explicit — every blocked range is an auditable octet/segment check, not a call to a std helper whose
//! semantics might drift (and one of the ones we'd want, `Ipv6Addr::is_unicast_link_local`, is still
//! unstable anyway).
//!
//! Two layers, both here:
//!
//! 1. [`SsrfResolver`] implements [`reqwest::dns::Resolve`]. reqwest connects to exactly what the
//!    resolver returns, so validating every resolved `SocketAddr` here checks the *actual* connection
//!    IP — for the initial request and for every redirect hop — closing the resolve-then-reconnect
//!    (DNS-rebinding) TOCTOU gap a bare hostname check leaves wide open.
//! 2. [`validate_url`] rejects non-`http(s)` schemes and, for a URL whose host is a literal IP, the IP
//!    directly (so `http://10.0.0.1/` is refused without even a DNS round trip). The tool re-runs this
//!    on every redirect hop, which also stops an `http → file:` scheme downgrade.
//!
//! The one thing an IP blocklist can't express is "let me reach *this* private host in dev," so
//! [`EgressPolicy`] carries an explicit host allowlist (`--web-allow-host`) and a blanket
//! `--web-allow-private` escape hatch. The allowlist is also what makes a loopback test fixture
//! reachable at all — see `tests/web_tool.rs`.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

/// Why a fetch was refused before it ever left the process. Rendered into a `ToolError::InvalidInput`
/// (the caller's fault: the model chose the URL) so the model gets a clear reason back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Blocked {
    /// Scheme other than http/https (e.g. `file:`, `ftp:`, `gopher:`).
    Scheme(String),
    /// The URL had no host at all.
    NoHost,
    /// The host was (or resolved to) a blocked address.
    Address {
        host: String,
        addr: IpAddr,
        class: &'static str,
    },
}

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scheme(s) => {
                write!(
                    f,
                    "refusing to fetch a `{s}:` URL — only http and https are allowed"
                )
            }
            Self::NoHost => write!(f, "the URL has no host to fetch"),
            Self::Address { host, addr, class } => write!(
                f,
                "refusing to fetch `{host}`: it resolves to {addr}, a {class} address \
                 (blocked to prevent access to internal services; pass --web-allow-host {host} \
                 or --web-allow-private to override)"
            ),
        }
    }
}

impl std::error::Error for Blocked {}

/// What egress the operator permits. Deny-by-default; the two knobs open it back up.
#[derive(Debug, Clone, Default)]
pub struct EgressPolicy {
    /// `--web-allow-private`: disable the private/loopback/link-local blocklist entirely. For a trusted
    /// local-dev box.
    pub allow_private: bool,
    /// `--web-allow-host`: hostnames (as they appear in the URL, case-insensitive) that bypass the
    /// blocklist even when `allow_private` is off — a specific internal service, or `127.0.0.1` for a
    /// test fixture.
    pub allow_hosts: Vec<String>,
}

impl EgressPolicy {
    pub fn new(allow_private: bool, allow_hosts: &[String]) -> Self {
        Self {
            allow_private,
            allow_hosts: allow_hosts.iter().map(|h| h.to_ascii_lowercase()).collect(),
        }
    }

    fn host_allowed(&self, host: &str) -> bool {
        // `allow_hosts` is already lowercased at construction, so an ASCII-case-insensitive compare
        // matches without allocating a lowercased copy of `host` per check.
        self.allow_hosts
            .iter()
            .any(|h| h.eq_ignore_ascii_case(host))
    }

    /// Whether `addr` is safe to connect to for a request to `host`, given this policy.
    pub fn addr_permitted(&self, host: &str, addr: IpAddr) -> Result<(), Blocked> {
        if self.allow_private || self.host_allowed(host) {
            return Ok(());
        }
        match blocked_class(addr) {
            Some(class) => Err(Blocked::Address {
                host: host.to_string(),
                addr,
                class,
            }),
            None => Ok(()),
        }
    }
}

/// Classify a blocked address, or `None` if it is a routable public address. Every arm is an explicit
/// range so the policy can be read and audited without chasing std's stability guarantees.
///
/// IPv4-mapped and IPv4-compatible IPv6 addresses are unwrapped and re-checked as their embedded v4 —
/// `http://[::ffff:169.254.169.254]/` must not sneak a metadata fetch past the v4 checks.
pub fn blocked_class(addr: IpAddr) -> Option<&'static str> {
    match addr {
        IpAddr::V4(v4) => blocked_v4(v4),
        IpAddr::V6(v6) => {
            // `::ffff:a.b.c.d` — check as its embedded v4.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return blocked_v4(v4);
            }
            // v6-native ranges *first*: `::1`/`::` must be classified here, before the v4-compatible
            // fallback below — `Ipv6Addr::to_ipv4()` maps `::1` to the *public-looking* `0.0.0.1` and
            // would otherwise wave loopback straight through.
            if let Some(class) = blocked_v6(v6) {
                return Some(class);
            }
            // The deprecated v4-compatible `::a.b.c.d` form (only reached when the v6-native checks
            // didn't already block it).
            if let Some(v4) = v6.to_ipv4() {
                return blocked_v4(v4);
            }
            None
        }
    }
}

fn blocked_v4(ip: Ipv4Addr) -> Option<&'static str> {
    let [a, b, ..] = ip.octets();
    if ip.is_unspecified() {
        return Some("unspecified (0.0.0.0)");
    }
    if ip.is_loopback() {
        return Some("loopback");
    }
    if ip.is_link_local() {
        // 169.254.0.0/16 — includes cloud metadata 169.254.169.254.
        return Some("link-local / cloud-metadata");
    }
    if ip.is_private() {
        // 10/8, 172.16/12, 192.168/16.
        return Some("private (RFC 1918)");
    }
    if ip.is_broadcast() {
        return Some("broadcast");
    }
    // 100.64.0.0/10 — carrier-grade NAT (RFC 6598), reachable-but-internal.
    if a == 100 && (64..=127).contains(&b) {
        return Some("carrier-grade NAT (RFC 6598)");
    }
    None
}

fn blocked_v6(ip: Ipv6Addr) -> Option<&'static str> {
    let seg0 = ip.segments()[0];
    if ip.is_unspecified() {
        return Some("unspecified (::)");
    }
    if ip.is_loopback() {
        return Some("loopback (::1)");
    }
    // fc00::/7 — unique local addresses (the v6 analogue of RFC 1918).
    if (seg0 & 0xfe00) == 0xfc00 {
        return Some("unique-local (fc00::/7)");
    }
    // fe80::/10 — link-local unicast.
    if (seg0 & 0xffc0) == 0xfe80 {
        return Some("link-local (fe80::/10)");
    }
    None
}

/// Validate a URL before any network activity: scheme must be http/https, and if the host is a literal
/// IP it is checked directly (no DNS needed). A hostname passes here and is validated later at the
/// resolver seam. Called on the initial URL and on every redirect hop.
pub fn validate_url(url: &reqwest::Url, policy: &EgressPolicy) -> Result<(), Blocked> {
    match url.scheme() {
        "http" | "https" => {}
        other => return Err(Blocked::Scheme(other.to_string())),
    }
    let host = url.host_str().ok_or(Blocked::NoHost)?;
    match url.host() {
        Some(url::Host::Ipv4(v4)) => policy.addr_permitted(host, IpAddr::V4(v4)),
        Some(url::Host::Ipv6(v6)) => policy.addr_permitted(host, IpAddr::V6(v6)),
        _ => Ok(()), // a name — validated at resolution time
    }
}

/// A [`reqwest::dns::Resolve`] that resolves normally, then refuses the whole request if **any**
/// resolved address is blocked. Installed via `ClientBuilder::dns_resolver`, so it runs for the initial
/// connection and every redirect hop reqwest makes.
pub struct SsrfResolver {
    /// Held behind an `Arc` so each `resolve` clones a pointer into the async task, not the whole
    /// `EgressPolicy` (which owns a `Vec<String>` of allow-hosts).
    policy: Arc<EgressPolicy>,
}

impl SsrfResolver {
    pub fn new(policy: EgressPolicy) -> Arc<Self> {
        Arc::new(Self {
            policy: Arc::new(policy),
        })
    }
}

impl Resolve for SsrfResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let policy = Arc::clone(&self.policy);
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?
                .collect();
            for sa in &addrs {
                policy
                    .addr_permitted(&host, sa.ip())
                    .map_err(|b| Box::new(b) as Box<dyn std::error::Error + Send + Sync>)?;
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn public_v4_addresses_pass() {
        for s in ["1.1.1.1", "8.8.8.8", "93.184.216.34", "140.82.112.3"] {
            assert_eq!(blocked_class(ip(s)), None, "{s} should be public");
        }
    }

    #[test]
    fn the_internal_v4_ranges_are_blocked() {
        let cases = [
            ("127.0.0.1", "loopback"),
            ("127.5.6.7", "loopback"),
            ("0.0.0.0", "unspecified"),
            ("169.254.169.254", "link-local"), // cloud metadata
            ("169.254.0.1", "link-local"),
            ("10.0.0.1", "private"),
            ("172.16.0.1", "private"),
            ("172.31.255.255", "private"),
            ("192.168.1.1", "private"),
            ("100.64.0.1", "carrier-grade NAT"),
            ("255.255.255.255", "broadcast"),
        ];
        for (s, needle) in cases {
            let class = blocked_class(ip(s)).unwrap_or_else(|| panic!("{s} must be blocked"));
            assert!(
                class.contains(needle),
                "{s}: got {class:?}, wanted {needle:?}"
            );
        }
        // A public-looking neighbor of a private range is NOT blocked.
        assert_eq!(blocked_class(ip("172.32.0.1")), None);
        assert_eq!(blocked_class(ip("11.0.0.1")), None);
    }

    #[test]
    fn the_internal_v6_ranges_are_blocked() {
        for s in ["::1", "::", "fc00::1", "fd12:3456::1", "fe80::1"] {
            assert!(blocked_class(ip(s)).is_some(), "{s} must be blocked");
        }
        // Public v6 passes.
        assert_eq!(blocked_class(ip("2606:4700:4700::1111")), None);
    }

    #[test]
    fn ipv4_mapped_v6_cannot_smuggle_an_internal_v4_past_the_check() {
        // The classic bypass: dress a metadata/loopback address as v6.
        assert!(blocked_class(ip("::ffff:169.254.169.254")).is_some());
        assert!(blocked_class(ip("::ffff:127.0.0.1")).is_some());
        assert!(blocked_class(ip("::ffff:10.0.0.1")).is_some());
        // A mapped *public* v4 still passes.
        assert_eq!(blocked_class(ip("::ffff:8.8.8.8")), None);
    }

    fn url(s: &str) -> reqwest::Url {
        s.parse().unwrap()
    }

    #[test]
    fn validate_url_rejects_non_http_schemes() {
        let p = EgressPolicy::default();
        assert!(matches!(
            validate_url(&url("file:///etc/passwd"), &p),
            Err(Blocked::Scheme(_))
        ));
        assert!(matches!(
            validate_url(&url("ftp://example.com/x"), &p),
            Err(Blocked::Scheme(_))
        ));
        assert!(validate_url(&url("https://example.com/"), &p).is_ok());
    }

    #[test]
    fn validate_url_blocks_a_literal_internal_ip_without_dns() {
        let p = EgressPolicy::default();
        assert!(validate_url(&url("http://169.254.169.254/latest/meta-data/"), &p).is_err());
        assert!(validate_url(&url("http://[::1]:8080/"), &p).is_err());
        assert!(validate_url(&url("http://10.0.0.5/admin"), &p).is_err());
        assert!(validate_url(&url("https://93.184.216.34/"), &p).is_ok());
    }

    #[test]
    fn allow_private_disables_the_blocklist() {
        let p = EgressPolicy::new(true, &[]);
        assert!(p.addr_permitted("anything", ip("10.0.0.1")).is_ok());
        assert!(p.addr_permitted("anything", ip("127.0.0.1")).is_ok());
    }

    #[test]
    fn allow_host_reopens_exactly_one_host_case_insensitively() {
        let p = EgressPolicy::new(false, &["Localhost".to_string()]);
        assert!(p.addr_permitted("localhost", ip("127.0.0.1")).is_ok());
        assert!(p.addr_permitted("LOCALHOST", ip("127.0.0.1")).is_ok());
        // A different internal host is still blocked.
        assert!(p.addr_permitted("evil.internal", ip("127.0.0.1")).is_err());
    }
}
