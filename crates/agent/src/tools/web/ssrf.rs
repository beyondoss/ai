//! Egress safety for the `web` tool — the layer ax (a local human-run CLI) never needs and we cannot
//! do without (we fetch URLs the *model* chose, from a server).
//!
//! Because a plain `web` fetch is never approval-gated, **this is the sole boundary
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
/// The table follows the IANA IPv4 and IPv6 Special-Purpose Address Registries: every block that is
/// not globally reachable is refused. IPv6 forms that *embed* an IPv4 address — IPv4-mapped
/// (`::ffff:a.b.c.d`), IPv4-compatible (`::a.b.c.d`), NAT64 well-known prefix (`64:ff9b::a.b.c.d`,
/// RFC 6052) and 6to4 (`2002:aabb:ccdd::`, RFC 3056) — are unwrapped and re-checked as their embedded
/// v4, so `http://[64:ff9b::a9fe:a9fe]/` can't sneak a metadata fetch past the v4 checks through a
/// DNS64/NAT64 gateway. A public embedded v4 still passes: that is how an IPv6-only host reaches the
/// v4 internet.
///
/// This one classifier is shared by the `web` tool and MCP's HTTP egress (`tools::mcp::McpEgress`),
/// both through [`EgressPolicy::addr_permitted`].
pub fn blocked_class(addr: IpAddr) -> Option<&'static str> {
    match addr {
        IpAddr::V4(v4) => blocked_v4(v4),
        IpAddr::V6(v6) => {
            // `::ffff:a.b.c.d` — check as its embedded v4.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return blocked_v4(v4);
            }
            let seg = v6.segments();
            // 64:ff9b::/96 — NAT64 well-known prefix (RFC 6052): the low 32 bits are the v4 the
            // translator will connect to.
            if seg[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
                return blocked_v4(embedded_v4(seg[6], seg[7]));
            }
            // 2002::/16 — 6to4 (RFC 3056): bits 16..48 are the v4 of the relay-reachable site.
            if seg[0] == 0x2002 {
                return blocked_v4(embedded_v4(seg[1], seg[2]));
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

fn embedded_v4(hi: u16, lo: u16) -> Ipv4Addr {
    let [a, b] = hi.to_be_bytes();
    let [c, d] = lo.to_be_bytes();
    Ipv4Addr::new(a, b, c, d)
}

fn blocked_v4(ip: Ipv4Addr) -> Option<&'static str> {
    let [a, b, c, _] = ip.octets();
    if ip.is_unspecified() {
        return Some("unspecified (0.0.0.0)");
    }
    // 0.0.0.0/8 — "this network" (RFC 791). Linux routes `0.x.y.z` to the local host.
    if a == 0 {
        return Some("this-network (0.0.0.0/8)");
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
    // 192.0.0.0/24 — IETF protocol assignments (RFC 6890): DS-Lite B4/AFTR, NAT64 discovery, …
    if (a, b, c) == (192, 0, 0) {
        return Some("IETF protocol assignment (192.0.0.0/24)");
    }
    // 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24 — documentation (RFC 5737).
    if (a, b, c) == (192, 0, 2) || (a, b, c) == (198, 51, 100) || (a, b, c) == (203, 0, 113) {
        return Some("documentation (RFC 5737)");
    }
    // 192.88.99.0/24 — deprecated 6to4 relay anycast (RFC 7526).
    if (a, b, c) == (192, 88, 99) {
        return Some("6to4 relay anycast (192.88.99.0/24)");
    }
    // 198.18.0.0/15 — benchmarking (RFC 2544).
    if a == 198 && (b == 18 || b == 19) {
        return Some("benchmarking (198.18.0.0/15)");
    }
    // 224.0.0.0/4 — multicast.
    if (224..=239).contains(&a) {
        return Some("multicast (224.0.0.0/4)");
    }
    // 240.0.0.0/4 — reserved (RFC 1112); 255.255.255.255 was caught above.
    if a >= 240 {
        return Some("reserved (240.0.0.0/4)");
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
    // fec0::/10 — deprecated site-local (RFC 3879); still routed internally on old networks.
    if (seg0 & 0xffc0) == 0xfec0 {
        return Some("site-local (fec0::/10)");
    }
    // ff00::/8 — multicast.
    if (seg0 & 0xff00) == 0xff00 {
        return Some("multicast (ff00::/8)");
    }
    let seg = ip.segments();
    // 64:ff9b:1::/48 — local-use NAT64 (RFC 8215): a site's own translator, internal by definition.
    if seg[..3] == [0x64, 0xff9b, 0x1] {
        return Some("local-use NAT64 (64:ff9b:1::/48)");
    }
    // 100::/64 — discard-only (RFC 6666).
    if seg[..4] == [0x100, 0, 0, 0] {
        return Some("discard-only (100::/64)");
    }
    // 2001::/23 — IETF protocol assignments, including Teredo 2001::/32 (a tunnel whose far end is
    // an obfuscated IPv4 we can't vet), benchmarking 2001:2::/48 and ORCHID.
    if seg0 == 0x2001 && seg[1] < 0x200 {
        return Some("IETF protocol assignment / Teredo (2001::/23)");
    }
    // 2001:db8::/32 and 3fff::/20 — documentation (RFC 3849, RFC 9637).
    if (seg0 == 0x2001 && seg[1] == 0xdb8) || (seg0 == 0x3fff && seg[1] < 0x1000) {
        return Some("documentation (2001:db8::/32, 3fff::/20)");
    }
    // 5f00::/16 — SRv6 segment identifiers (RFC 9602), not globally reachable.
    if seg0 == 0x5f00 {
        return Some("SRv6 SID (5f00::/16)");
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
            // port-0: a DNS lookup, not a bind — only the addresses it returns are used.
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

    /// Every IANA special-purpose block that is not globally reachable, one representative address
    /// (or more, at the edges) per row.
    #[test]
    fn every_special_purpose_v4_range_is_blocked() {
        let cases = [
            ("0.0.0.1", "this-network"),
            ("0.255.255.255", "this-network"),
            ("192.0.0.1", "IETF protocol"),
            ("192.0.0.170", "IETF protocol"), // NAT64 discovery
            ("192.0.2.1", "documentation"),
            ("198.51.100.7", "documentation"),
            ("203.0.113.9", "documentation"),
            ("192.88.99.1", "6to4 relay"),
            ("198.18.0.1", "benchmarking"),
            ("198.19.255.255", "benchmarking"),
            ("224.0.0.1", "multicast"),
            ("239.255.255.250", "multicast"), // SSDP
            ("240.0.0.1", "reserved"),
            ("254.1.2.3", "reserved"),
            ("100.127.255.255", "carrier-grade NAT"),
        ];
        for (s, needle) in cases {
            let class = blocked_class(ip(s)).unwrap_or_else(|| panic!("{s} must be blocked"));
            assert!(
                class.contains(needle),
                "{s}: got {class:?}, wanted {needle:?}"
            );
        }
        // The neighbors are public.
        for s in [
            "1.0.0.1",
            "100.128.0.1",
            "192.0.1.1",
            "192.0.3.1",
            "198.17.255.255",
            "198.20.0.1",
            "223.255.255.254",
        ] {
            assert_eq!(blocked_class(ip(s)), None, "{s} is public");
        }
    }

    #[test]
    fn nat64_embeds_are_checked_as_their_v4() {
        // 64:ff9b::/96 — a DNS64/NAT64 gateway connects to the embedded v4, so an internal one is
        // an internal fetch.
        for s in [
            "64:ff9b::a9fe:a9fe", // 169.254.169.254
            "64:ff9b::7f00:1",    // 127.0.0.1
            "64:ff9b::a00:1",     // 10.0.0.1
            "64:ff9b::1",         // 0.0.0.1
            "64:ff9b::c0a8:101",  // 192.168.1.1
        ] {
            assert!(blocked_class(ip(s)).is_some(), "{s} must be blocked");
        }
        assert!(
            blocked_class(ip("64:ff9b::a9fe:a9fe"))
                .unwrap()
                .contains("link-local"),
            "the class is the embedded v4's"
        );
        // A public v4 through NAT64 is how an IPv6-only host reaches the v4 internet.
        assert_eq!(blocked_class(ip("64:ff9b::808:808")), None); // 8.8.8.8
        // 64:ff9b:1::/48 — local-use NAT64: internal whatever it embeds.
        for s in ["64:ff9b:1::808:808", "64:ff9b:1:ffff::1"] {
            assert!(
                blocked_class(ip(s)).unwrap().contains("local-use NAT64"),
                "{s}"
            );
        }
    }

    #[test]
    fn six_to_four_embeds_are_checked_as_their_v4() {
        assert!(blocked_class(ip("2002:a9fe:a9fe::1")).is_some()); // 169.254.169.254
        assert!(blocked_class(ip("2002:7f00:1::")).is_some()); // 127.0.0.1
        assert!(blocked_class(ip("2002:c0a8:101::1")).is_some()); // 192.168.1.1
        assert_eq!(blocked_class(ip("2002:808:808::1")), None); // 8.8.8.8
    }

    #[test]
    fn every_special_purpose_v6_range_is_blocked() {
        let cases = [
            ("fec0::1", "site-local"),
            ("feff::1", "site-local"),
            ("ff02::1", "multicast"),
            ("ff05::2", "multicast"),
            ("100::1", "discard-only"),
            ("2001::1", "Teredo"),
            ("2001:0:4136:e378:8000:63bf:3fff:fdd2", "Teredo"),
            ("2001:2::1", "IETF protocol"), // benchmarking
            ("2001:1ff:ffff::1", "IETF protocol"),
            ("2001:db8::1", "documentation"),
            ("3fff::1", "documentation"),
            ("3fff:fff::1", "documentation"),
            ("5f00::1", "SRv6"),
            ("fdff:ffff::1", "unique-local"),
            ("febf::1", "link-local"),
        ];
        for (s, needle) in cases {
            let class = blocked_class(ip(s)).unwrap_or_else(|| panic!("{s} must be blocked"));
            assert!(
                class.contains(needle),
                "{s}: got {class:?}, wanted {needle:?}"
            );
        }
        // The neighbors are public.
        for s in [
            "2001:200::1", // the first real RIR allocation after the /23
            "2001:4860:4860::8888",
            "2001:db9::1",
            "3ffe:ffff::1",
            "3fff:1000::1",
            "4000::1",
            "5f01::1",
            "fe00::1",
            "101::1",
        ] {
            assert_eq!(blocked_class(ip(s)), None, "{s} is public");
        }
    }

    #[test]
    fn mcp_egress_uses_this_classifier() {
        // MCP's HTTP egress builds on `EgressPolicy::addr_permitted`, which is this table.
        let p = EgressPolicy::default();
        for s in ["64:ff9b::a9fe:a9fe", "0.1.2.3", "198.18.0.1", "ff02::1"] {
            assert!(p.addr_permitted("h", ip(s)).is_err(), "{s}");
        }
        assert!(
            validate_url(&url("http://[64:ff9b::7f00:1]/"), &p).is_err(),
            "a literal NAT64 loopback URL is refused without DNS"
        );
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
