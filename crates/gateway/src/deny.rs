//! Sparse per-tenant deny-set — the gateway's *entire* spend/fraud surface.
//!
//! Design (deliberate, see plan): the gateway only ever asks "is this tenant cut off?" and
//! default-**allows** on a miss. We hold **only the exceptions** (the cut-off tenants), so memory
//! is `O(denied)`, not `O(tenants)` — this scales to millions of tenants because `denied` stays a
//! tiny slice (a few MB even at 1M entries; a tenant id is 8 bytes). The gateway never decides
//! *why* a tenant is denied — the control plane writes/removes entries; we just enforce + log.
//!
//! TTL/auto-restore is handled by slipstream, not here: spend holds are written with a TTL to the
//! next budget reset, so they expire into a `Del` event that removes them; fraud holds have no TTL
//! (sticky). This struct only reflects current membership.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// The deny map's hasher. **Deliberately not** std's `RandomState` (SipHash-1-3).
///
/// HashDoS resistance exists to stop an attacker from *choosing* keys that collide. Here the key is
/// `identity.tenant_id`, which only ever comes out of `keyring.verify(raw_key)` — an Ed25519
/// signature check over a token minted by the control plane, which holds the only signing key. And
/// the ordering is load-bearing: `proxy::request_filter` verifies **first** and only then looks the
/// tenant up (`proxy.rs`, step 5), so an unforgeable id is the *only* thing that ever reaches this
/// map. An attacker cannot pick their tenant id, cannot enumerate ids to farm collisions, and never
/// gets an unverified id into the table at all — so SipHash's collision resistance buys nothing and
/// costs ~4x on a lookup that runs on **every managed request**.
///
/// Measured on the dev host (`benches/unit.rs`, `deny::reason_*`, 1M entries): SipHash 6.21 ns vs
/// Fx 1.49 ns single-threaded; 13.44 ns vs 3.40 ns at 16 threads (the gateway is many-core, so the
/// contended number is the one that matters).
///
/// **Do not "fix" this back to `RandomState`** without first breaking the property above — i.e. not
/// unless some path starts inserting or looking up a tenant id that a caller supplied and we did not
/// verify. If that ever happens, the hasher is the *second* thing to fix; the unverified id is the
/// first.
type DenyHasher = BuildHasherDefault<rustc_hash::FxHasher>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyReason {
    /// Over budget. Typically written with a TTL to the next reset → auto-restores.
    Spend,
    /// Abuse / fraud. Sticky (no TTL) until a human clears it.
    Fraud,
    /// Reason not recognized in the entry value — still denied (fail safe on the enforce side).
    Unknown,
}

impl DenyReason {
    /// HTTP status to return. 402 Payment Required for spend, 403 Forbidden for fraud/other —
    /// gives the client (and our own dashboards) a meaningful signal without leaking detail.
    pub fn http_status(self) -> u16 {
        match self {
            DenyReason::Spend => 402,
            DenyReason::Fraud | DenyReason::Unknown => 403,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct DenySet {
    denied: HashMap<u64, DenyReason, DenyHasher>,
}

impl DenySet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-size for a known entry count — for the seed/rescan path, which knows how many entries it
    /// is about to insert and would otherwise rehash the whole table a dozen times on the way up.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            denied: HashMap::with_capacity_and_hasher(n, DenyHasher::default()),
        }
    }

    /// Default-allow: absence from the set = allowed. This is the safe-for-availability default —
    /// a tenant we've never heard of is served, not blocked.
    pub fn is_denied(&self, tenant_id: u64) -> bool {
        self.denied.contains_key(&tenant_id)
    }

    pub fn reason(&self, tenant_id: u64) -> Option<DenyReason> {
        self.denied.get(&tenant_id).copied()
    }

    pub fn insert(&mut self, tenant_id: u64, reason: DenyReason) {
        self.denied.insert(tenant_id, reason);
    }

    pub fn remove(&mut self, tenant_id: u64) {
        self.denied.remove(&tenant_id);
    }

    pub fn len(&self) -> usize {
        self.denied.len()
    }

    pub fn is_empty(&self) -> bool {
        self.denied.is_empty()
    }
}

/// The seed/rescan constructor (`store_watch::denyset_from_entries` builds the whole set with
/// `.collect()`). The hasher is an implementation detail of `DenySet`, deliberately not a type
/// parameter, so this signature — and every call site — is unaffected by the choice above.
impl FromIterator<(u64, DenyReason)> for DenySet {
    fn from_iter<I: IntoIterator<Item = (u64, DenyReason)>>(iter: I) -> Self {
        Self {
            denied: iter.into_iter().collect(),
        }
    }
}

/// Parse a slipstream deny key `blackhole.{tenant_id}` → tenant id. Returns `None` for keys that
/// don't match (so an unrelated watched key never corrupts the set).
pub fn parse_key(key: &str) -> Option<u64> {
    key.strip_prefix("blackhole.")?.parse().ok()
}

/// The only field of the control plane's JSON entry value we care about; everything else it writes
/// (`exp`, …) is skipped by the parser without ever being materialized.
///
/// `Cow`, not `&str`: a `&str` field can only deserialize from an *unescaped* string, so a reason
/// containing a `\`-escape would fail the whole parse and silently degrade to `Unknown`. `Cow`
/// borrows in the common case and falls back to an owned `String` only for an escaped value.
///
/// Two non-obvious requirements, both load-bearing — drop either and this silently allocates again:
///   * `#[serde(borrow)]` — without it serde uses the blanket `impl Deserialize for Cow<'_, T>`,
///     which *always* yields `Cow::Owned`.
///   * a **bare** `Cow`, not `Option<Cow>` — serde_derive only rewrites a field to its borrowing
///     deserializer when the type is literally `Cow<'a, _>` (`internals::attr::is_cow`); wrapped in
///     an `Option` the attribute still compiles but quietly falls back to the always-owned impl.
///     `#[serde(default)]` covers the absent case instead, and `String::new()` doesn't allocate.
#[derive(serde::Deserialize)]
struct ReasonOnly<'a> {
    #[serde(borrow, default)]
    reason: std::borrow::Cow<'a, str>,
}

/// Parse the entry value into a reason. Accepts either a bare token (`spend`/`fraud`) or a JSON
/// object `{"reason":"spend", ...}`. Anything else → `Unknown` (still denied — fail safe).
pub fn parse_reason(value: &[u8]) -> DenyReason {
    let s = std::str::from_utf8(value).unwrap_or("").trim();
    // Both branches are zero-alloc. JSON is what the control plane actually writes, so it is *not*
    // the cold branch: it runs once per delta, and once per entry on a full seed/rescan — at a large
    // deny-set, allocating here is millions of allocations per cold boot. Deserializing into a
    // borrowed `ReasonOnly` instead of a `serde_json::Value` skips building a `Map` plus a `String`
    // per key and per string value, and hands back a `&str` pointing into `value` itself.
    let parsed;
    let token: &str = if s.starts_with('{') {
        // `from_str` on the trimmed `s`, not `from_slice` on the raw bytes: identical input to the
        // parser (JSON ignores surrounding whitespace) but it skips a second UTF-8 validation of a
        // slice we just validated.
        parsed = serde_json::from_str::<ReasonOnly<'_>>(s).ok();
        parsed.as_ref().map_or("", |r| &r.reason)
    } else {
        s
    };
    match token {
        "spend" => DenyReason::Spend,
        "fraud" => DenyReason::Fraud,
        _ => DenyReason::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_allows_unknown_tenants() {
        let set = DenySet::new();
        assert!(!set.is_denied(12345));
    }

    #[test]
    fn insert_remove_and_reason() {
        let mut set = DenySet::new();
        set.insert(1, DenyReason::Spend);
        set.insert(2, DenyReason::Fraud);
        assert!(set.is_denied(1));
        assert_eq!(set.reason(1), Some(DenyReason::Spend));
        assert_eq!(set.reason(2).unwrap().http_status(), 403);
        set.remove(1);
        assert!(!set.is_denied(1)); // restored
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn key_parsing() {
        assert_eq!(parse_key("blackhole.42"), Some(42));
        assert_eq!(parse_key("blackhole.notanumber"), None);
        assert_eq!(parse_key("signkey.1"), None);
    }

    #[test]
    fn reason_parsing_bare_and_json() {
        assert_eq!(parse_reason(b"spend"), DenyReason::Spend);
        assert_eq!(parse_reason(b" fraud "), DenyReason::Fraud);
        assert_eq!(
            parse_reason(br#"{"reason":"spend","exp":123}"#),
            DenyReason::Spend
        );
        assert_eq!(parse_reason(b"weird"), DenyReason::Unknown);
    }

    #[test]
    fn reason_parsing_tolerates_escapes_and_junk() {
        // The reason arrives escaped (`\u0073` is `s`, so this decodes to "spend"). A borrowed
        // `&str` field cannot represent an escaped string and would fail the *whole* parse, silently
        // degrading to `Unknown`; `Cow` falls back to an owned copy for exactly this input. This is
        // the case that dictates the field type — see `ReasonOnly`.
        assert_eq!(
            parse_reason(br#"{"reason":"\u0073pend"}"#),
            DenyReason::Spend
        );
        // Fail-safe on every malformed shape: still denied, never an allow.
        assert_eq!(parse_reason(br#"{"reason":123}"#), DenyReason::Unknown);
        assert_eq!(parse_reason(br#"{"exp":123}"#), DenyReason::Unknown); // no `reason` field
        assert_eq!(parse_reason(b"{not json"), DenyReason::Unknown);
        assert_eq!(parse_reason(b""), DenyReason::Unknown);
        assert_eq!(parse_reason(&[0xff, 0xfe]), DenyReason::Unknown); // invalid UTF-8
        // Whitespace around a JSON object still reaches the JSON branch (we parse the trimmed str).
        assert_eq!(
            parse_reason(b"  {\"reason\":\"fraud\"}  "),
            DenyReason::Fraud
        );
    }

    #[test]
    fn spend_is_402_fraud_is_403() {
        assert_eq!(DenyReason::Spend.http_status(), 402);
        assert_eq!(DenyReason::Fraud.http_status(), 403);
        // Unknown is fail-safe: an unrecognized reason still denies, and maps to 403 like fraud
        // (not 402) — so a control-plane reason we don't yet parse can't be mistaken for a billing
        // block or, worse, leak through as an allow.
        assert_eq!(DenyReason::Unknown.http_status(), 403);
    }
}
