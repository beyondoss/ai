//! Sparse deny-set — the gateway's *entire* spend/fraud surface.
//!
//! Design (deliberate, see plan): the gateway only ever asks "is this tenant or credential cut
//! off?" and default-**allows** on a miss. We hold **only the exceptions**, so memory is
//! `O(denied)`, not `O(tenants)` — this scales to millions of tenants because `denied` stays a
//! tiny slice of them. The gateway never decides *why* — the control plane writes/removes
//! entries; we just enforce + log.
//!
//! Two memberships, one set: `blackhole.{tenant}` cuts off every key for that tenant;
//! `blackhole.key.{id}` cuts off one `bai_v2` credential. A request is denied if **either**
//! matches. v1 tokens have no `key_id`, so only the tenant map can deny them.
//!
//! A third membership, `blackhole.kid.{kid}`, distrusts a **signing key id**: every token under that
//! kid is refused at verify (401), whichever source — config or the live ring — supplies its key.
//! It is not a tenant or credential cut-off and `reason_for` never sees it; `Keyring::verify_with`
//! asks [`DenySet::kid_denied`] before any signature math. The kid it is asked about comes straight
//! from an *unverified* token, but the set is only ever read with it (callers cannot add to it), so
//! it shares the id table; it is skipped when empty, and never capped, because refusing to record a
//! distrust would fail unsafe. Its size is the number of kids the control plane has distrusted — a
//! handful — and a KV delete of the entry frees it.
//!
//! Storage and updates: each membership is an [`IdMap`], mutated in place by the single watch loop
//! (an update costs O(batch), not O(set)) and read lock-free by the request path. Its table is 10
//! bytes a slot at most three-quarters full: 1M denied tenants is `2^21` slots ≈ **20 MiB**, and a
//! seed sizes it up front. `million_entry_footprint_is_tens_of_megabytes` pins that arithmetic.
//!
//! TTL/auto-restore is handled by slipstream, not here: spend holds are written with a TTL to the
//! next budget reset, so they expire into a `Del` event that removes them; fraud holds have no TTL
//! (sticky). This struct only reflects current membership.

use crate::id_map::{Full, IdMap};
use crate::key::Kid;

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

/// Who a `blackhole.*` entry names. Parsed from the KV key; the value is still just a reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenyTarget {
    Tenant(u64),
    Key(u64),
    /// A signing key id, not a tenant or credential: see the module doc.
    Kid(Kid),
}

/// Reason codes as stored in an [`IdMap`] (0 is "absent" there, so they start at 1).
impl DenyReason {
    fn code(self) -> u8 {
        match self {
            DenyReason::Spend => 1,
            DenyReason::Fraud => 2,
            DenyReason::Unknown => 3,
        }
    }

    fn from_code(c: u8) -> Self {
        match c {
            1 => DenyReason::Spend,
            2 => DenyReason::Fraud,
            _ => DenyReason::Unknown,
        }
    }
}

/// Mutated in place through `&self` by the one watch loop (see [`IdMap`]); a rescan builds a fresh
/// set and replaces the whole thing atomically.
#[derive(Debug, Default)]
pub struct DenySet {
    denied: IdMap,
    keys: IdMap,
    /// Distrusted signing key ids. Membership only; the entry's value is not read.
    kids: IdMap,
}

impl DenySet {
    pub fn new() -> Self {
        Self::default()
    }

    /// The seed/rescan constructor: a set holding `targets`, each map sized for its own entries.
    pub fn from_targets(targets: &[(DenyTarget, DenyReason)]) -> Self {
        let (mut denied, mut keys, mut kids) = (Vec::new(), Vec::new(), Vec::new());
        for &(t, r) in targets {
            match t {
                DenyTarget::Tenant(t) => denied.push((t, r.code())),
                DenyTarget::Key(k) => keys.push((k, r.code())),
                DenyTarget::Kid(k) => kids.push((u64::from(k), 1)),
            }
        }
        Self {
            denied: IdMap::from_pairs(&denied),
            keys: IdMap::from_pairs(&keys),
            kids: IdMap::from_pairs(&kids),
        }
    }

    /// A copy with room for `extra` more of each kind. The watch loop publishes it in place of
    /// `self` when an in-place insert reports [`Full`].
    pub fn grown(&self, extra: usize) -> Self {
        Self {
            denied: self.denied.grown(extra),
            keys: self.keys.grown(extra),
            kids: self.kids.grown(extra),
        }
    }

    /// Pre-size the tenant map for `n` entries.
    #[cfg(test)]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            denied: IdMap::with_capacity(n),
            ..Self::default()
        }
    }

    /// Whether signing key id `kid` is distrusted. Hot path (every managed request): an empty set
    /// answers without hashing.
    pub fn kid_denied(&self, kid: Kid) -> bool {
        !self.kids.is_empty() && self.kids.contains(u64::from(kid))
    }

    /// Default-allow: absence from the set = allowed. This is the safe-for-availability default —
    /// a tenant we've never heard of is served, not blocked.
    pub fn is_denied(&self, tenant_id: u64) -> bool {
        self.denied.contains(tenant_id)
    }

    /// Tenant-only reason. Prefer [`Self::reason_for`] on the request path so a key-level deny
    /// is visible too.
    pub fn reason(&self, tenant_id: u64) -> Option<DenyReason> {
        self.denied.get(tenant_id).map(DenyReason::from_code)
    }

    /// Denied if the tenant is in the set **or** (when present) the credential's `key_id` is.
    /// Tenant deny wins when both match — it is the broader cut-off.
    pub fn reason_for(&self, tenant_id: u64, key_id: Option<u64>) -> Option<DenyReason> {
        if let Some(r) = self.reason(tenant_id) {
            return Some(r);
        }
        key_id.and_then(|id| self.keys.get(id).map(DenyReason::from_code))
    }

    /// The inserts are in place and `Err(Full)` leaves the set unchanged: grow it (see `IdMap`).
    pub fn insert(&self, tenant_id: u64, reason: DenyReason) -> Result<(), Full> {
        self.denied.insert(tenant_id, reason.code())
    }

    pub fn insert_key(&self, key_id: u64, reason: DenyReason) -> Result<(), Full> {
        self.keys.insert(key_id, reason.code())
    }

    pub fn insert_target(&self, target: DenyTarget, reason: DenyReason) -> Result<(), Full> {
        match target {
            DenyTarget::Tenant(t) => self.insert(t, reason),
            DenyTarget::Key(k) => self.insert_key(k, reason),
            DenyTarget::Kid(k) => self.kids.insert(u64::from(k), 1),
        }
    }

    pub fn remove(&self, tenant_id: u64) {
        self.denied.remove(tenant_id);
    }

    pub fn remove_key(&self, key_id: u64) {
        self.keys.remove(key_id);
    }

    pub fn remove_target(&self, target: DenyTarget) {
        match target {
            DenyTarget::Tenant(t) => self.remove(t),
            DenyTarget::Key(k) => self.remove_key(k),
            DenyTarget::Kid(k) => self.kids.remove(u64::from(k)),
        }
    }

    pub fn len(&self) -> usize {
        self.denied.len() + self.keys.len() + self.kids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FromIterator<(u64, DenyReason)> for DenySet {
    fn from_iter<I: IntoIterator<Item = (u64, DenyReason)>>(iter: I) -> Self {
        let pairs: Vec<_> = iter.into_iter().map(|(t, r)| (t, r.code())).collect();
        Self {
            denied: IdMap::from_pairs(&pairs),
            ..Self::default()
        }
    }
}

/// Parse a slipstream deny key.
///
/// - `blackhole.{tenant_id}` → [`DenyTarget::Tenant`]
/// - `blackhole.key.{id}` → [`DenyTarget::Key`]
/// - `blackhole.kid.{kid}` → [`DenyTarget::Kid`]
///
/// Both live under the `blackhole.` watch prefix, so one watcher sees both. `None` for anything
/// else (so an unrelated watched key never corrupts the set). The `key.` form is checked first —
/// otherwise `blackhole.key.42` would be a failed tenant parse and silently dropped.
pub fn parse_key(key: &str) -> Option<DenyTarget> {
    let rest = key.strip_prefix("blackhole.")?;
    if let Some(id) = rest.strip_prefix("key.") {
        return Some(DenyTarget::Key(id.parse().ok()?));
    }
    if let Some(kid) = rest.strip_prefix("kid.") {
        return Some(DenyTarget::Kid(kid.parse().ok()?));
    }
    Some(DenyTarget::Tenant(rest.parse().ok()?))
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
        let set = DenySet::new();
        set.insert(1, DenyReason::Spend).unwrap();
        set.insert(2, DenyReason::Fraud).unwrap();
        assert!(set.is_denied(1));
        assert_eq!(set.reason(1), Some(DenyReason::Spend));
        assert_eq!(set.reason(2).unwrap().http_status(), 403);
        set.remove(1);
        assert!(!set.is_denied(1)); // restored
        assert_eq!(set.len(), 1);
    }

    /// The module doc quotes a concrete footprint for a 1M-entry set. Pin the arithmetic to the
    /// actual allocation rather than to a comment nobody re-derives.
    #[test]
    fn million_entry_footprint_is_tens_of_megabytes() {
        let set = DenySet::with_capacity(1_000_000);
        // 1M entries at most three-quarters full rounds up to 2^21 slots of 10 bytes each.
        assert_eq!(set.denied.heap_bytes(), (1 << 21) * 10);
    }

    #[test]
    fn key_parsing() {
        assert_eq!(parse_key("blackhole.42"), Some(DenyTarget::Tenant(42)));
        assert_eq!(parse_key("blackhole.key.7"), Some(DenyTarget::Key(7)));
        assert_eq!(parse_key("blackhole.notanumber"), None);
        assert_eq!(parse_key("blackhole.key.notanumber"), None);
        assert_eq!(parse_key("blackhole.key"), None);
        assert_eq!(parse_key("blackhole.kid.3"), Some(DenyTarget::Kid(3)));
        assert_eq!(parse_key("blackhole.kid.x"), None);
        assert_eq!(parse_key("blackhole.kid"), None);
        assert_eq!(parse_key("signkey.1"), None);
    }

    #[test]
    fn key_or_tenant_deny() {
        let set = DenySet::new();
        set.insert_key(100, DenyReason::Fraud).unwrap();
        // Same tenant, two credentials: only the named key is cut off.
        assert_eq!(set.reason_for(1, Some(100)), Some(DenyReason::Fraud));
        assert_eq!(set.reason_for(1, Some(101)), None);
        assert_eq!(set.reason_for(1, None), None); // v1: no key_id, tenant not denied
        set.insert(1, DenyReason::Spend).unwrap();
        // Tenant deny kills every key for that tenant, including one already key-denied.
        assert_eq!(set.reason_for(1, Some(100)), Some(DenyReason::Spend));
        assert_eq!(set.reason_for(1, Some(101)), Some(DenyReason::Spend));
        assert_eq!(set.reason_for(1, None), Some(DenyReason::Spend));
        assert_eq!(set.reason_for(2, Some(100)), Some(DenyReason::Fraud)); // key still denied
        set.remove_key(100);
        assert_eq!(set.reason_for(2, Some(100)), None);
    }

    #[test]
    fn kid_distrust_is_its_own_membership() {
        let set = DenySet::new();
        assert!(!set.kid_denied(3));
        set.insert_target(DenyTarget::Kid(3), DenyReason::Fraud)
            .unwrap();
        assert!(set.kid_denied(3));
        assert!(!set.kid_denied(4));
        // A kid is neither a tenant nor a credential id of the same number.
        assert_eq!(set.reason_for(3, Some(3)), None);
        assert_eq!(set.len(), 1);
        set.remove_target(DenyTarget::Kid(3));
        assert!(!set.kid_denied(3));
        assert!(set.is_empty());
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
