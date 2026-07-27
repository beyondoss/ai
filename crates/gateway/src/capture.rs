//! Sparse per-tenant capture-set — which tenants have payload logging switched on.
//!
//! Structurally a twin of [`crate::deny`], and for the same reason: the gateway only ever asks "is
//! this tenant capturing?" and defaults to **no** on a miss, so we hold **only the exceptions**.
//! Memory is `O(capturing)`, not `O(tenants)` — capture is a debugging tool enabled for a handful of
//! tenants at a time, so that set stays tiny while the tenant table does not.
//!
//! The one behavioural difference from the deny-set is the direction it fails. A stale deny-set
//! keeps *enforcing* (we never clear it), because under-enforcing costs money. A stale capture-set
//! just keeps capturing what it was already capturing, and a missing one captures nothing — losing
//! debug payloads is the correct thing to lose during a NATS outage, which is why this set needs no
//! on-disk snapshot to survive a cold start (see `store_watch::WatchedSet::snapshot_path`).
//!
//! Enablement expiry is **not** implemented here: the control plane writes a capture entry with a
//! slipstream TTL, and its expiry arrives as an ordinary `Delete`/`Purge` delta that removes the
//! tenant. "Capture tenant 42 for the next hour" therefore costs this crate zero lines of code.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// Same `FxHasher` choice, and the same justification, as [`crate::deny`]: the key is a `tenant_id`
/// that only ever arrives from `keyring.verify()`, so it is unforgeable and un-enumerable by an
/// attacker and SipHash's collision resistance buys nothing on a lookup that runs on every managed
/// request. See `deny.rs`'s `DenyHasher` for the measured numbers and the conditions under which
/// this choice would have to be revisited.
type CaptureHasher = BuildHasherDefault<rustc_hash::FxHasher>;

/// Fraction of a capturing tenant's requests to actually capture, as `1` in `sample_n`.
///
/// Exists because "capture tenant 42" can mean 10k req/s. Note this bounds only *operator*-enabled
/// capture: a caller who explicitly asks for a request with `x-beyond-capture: on` is never sampled
/// away (see `proxy`'s capture decision), because silently dropping a specifically-requested trace
/// is the one outcome that makes the feature useless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRule {
    /// Capture 1 request in `sample_n`. Always ≥ 1; `1` means every request.
    pub sample_n: u32,
    /// Per-direction byte cap for this tenant, overriding the config default.
    pub max_bytes: u32,
}

impl CaptureRule {
    /// Is the request at `seq` in this rule's sample?
    ///
    /// Deterministic on the gateway's existing monotonic request counter rather than an RNG: no
    /// per-tenant state to keep, no lock, and a reproducible answer when reasoning after the fact
    /// about why a given request was or wasn't captured.
    pub fn samples(&self, seq: u64) -> bool {
        // `sample_n` is normalized to ≥ 1 at parse time; guard anyway so a future construction path
        // can't turn a bad control-plane value into a division by zero on the request path.
        self.sample_n <= 1 || seq.is_multiple_of(u64::from(self.sample_n.max(1)))
    }
}

#[derive(Debug, Default, Clone)]
pub struct CaptureSet {
    capturing: HashMap<u64, CaptureRule, CaptureHasher>,
}

impl CaptureSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Default-off: absence from the set = not capturing. The safe default for a feature whose whole
    /// cost is storing customer prompts — a tenant we've never heard of is never captured.
    pub fn rule_for(&self, tenant_id: u64) -> Option<CaptureRule> {
        self.capturing.get(&tenant_id).copied()
    }

    pub fn insert(&mut self, tenant_id: u64, rule: CaptureRule) {
        self.capturing.insert(tenant_id, rule);
    }

    pub fn remove(&mut self, tenant_id: u64) {
        self.capturing.remove(&tenant_id);
    }

    pub fn len(&self) -> usize {
        self.capturing.len()
    }

    pub fn is_empty(&self) -> bool {
        self.capturing.is_empty()
    }
}

impl FromIterator<(u64, CaptureRule)> for CaptureSet {
    fn from_iter<I: IntoIterator<Item = (u64, CaptureRule)>>(iter: I) -> Self {
        Self {
            capturing: iter.into_iter().collect(),
        }
    }
}

/// Head-bounded copies of one request's bodies, held only while that request is being captured.
///
/// **Head, not tail** — deliberately the inverse of `proxy::UsageTail`, which keeps the *last* 64 KB.
/// The two want opposite ends because they answer opposite questions: usage lives at the end of a
/// response (`message_delta`, the final usage chunk), while *meaning* lives at the start of a request
/// (the system prompt and opening messages). Truncating a capture from the front would throw away
/// the part that explains what the agent was told to do.
///
/// Appends stop at `max_bytes` and set the truncation flag; they never reallocate past the cap and
/// never withhold a byte from the relay — this is a passive tap, exactly like the usage tap it sits
/// beside (see `proxy::response_body_filter`).
#[derive(Debug)]
pub struct CaptureBufs {
    req: Vec<u8>,
    resp: Vec<u8>,
    req_truncated: bool,
    resp_truncated: bool,
    max_bytes: usize,
}

impl CaptureBufs {
    pub fn new(max_bytes: u32) -> Self {
        Self {
            // Not pre-allocated to `max_bytes`: most captured bodies are far smaller than the cap,
            // and reserving 256 KiB twice per captured request would dwarf the bytes actually used.
            req: Vec::new(),
            resp: Vec::new(),
            req_truncated: false,
            resp_truncated: false,
            max_bytes: max_bytes as usize,
        }
    }

    pub fn push_req(&mut self, chunk: &[u8]) {
        Self::push(
            &mut self.req,
            &mut self.req_truncated,
            chunk,
            self.max_bytes,
        );
    }

    pub fn push_resp(&mut self, chunk: &[u8]) {
        Self::push(
            &mut self.resp,
            &mut self.resp_truncated,
            chunk,
            self.max_bytes,
        );
    }

    fn push(buf: &mut Vec<u8>, truncated: &mut bool, chunk: &[u8], max: usize) {
        let room = max.saturating_sub(buf.len());
        if room == 0 {
            // Already full: mark truncated only if there were actually more bytes to drop, so a body
            // that lands exactly on the cap isn't mislabelled as incomplete.
            *truncated |= !chunk.is_empty();
            return;
        }
        let take = room.min(chunk.len());
        buf.extend_from_slice(&chunk[..take]);
        *truncated |= take < chunk.len();
    }

    /// The captured request body as UTF-8, or `None` if it isn't valid UTF-8.
    ///
    /// These are JSON and SSE bodies, so valid UTF-8 is the norm; the `None` path exists because a
    /// *truncated* capture can end mid-multibyte-character even when the full body was fine.
    pub fn req_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.req).ok()
    }

    pub fn resp_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.resp).ok()
    }

    pub fn req_truncated(&self) -> bool {
        self.req_truncated
    }

    pub fn resp_truncated(&self) -> bool {
        self.resp_truncated
    }

    /// Total bytes retained, for the cost counter.
    pub fn bytes(&self) -> usize {
        self.req.len() + self.resp.len()
    }
}

/// Parse a slipstream capture key `aicapture.{tenant_id}` → tenant id. `None` for anything else, so
/// an unrelated watched key can never enable capture for a tenant that wasn't named.
pub fn parse_key(key: &str) -> Option<u64> {
    key.strip_prefix("aicapture.")?.parse().ok()
}

/// The entry value the control plane writes. Both fields optional so `{}` — or a bare non-JSON
/// value — still means "capture this tenant with the defaults", which is what an operator typing
/// the minimum possible entry during an incident expects to happen.
#[derive(serde::Deserialize)]
struct RuleValue {
    #[serde(default)]
    sample_n: Option<u32>,
    #[serde(default)]
    max_bytes: Option<u32>,
}

/// Parse an entry value into a rule, falling back to `defaults` for anything absent or unusable.
///
/// **Presence of the key is the enablement signal; the value only tunes it.** A malformed value
/// therefore still captures (at defaults) rather than silently doing nothing — an operator who
/// fat-fingers the JSON during an incident gets capture, which is what they were reaching for, and
/// the mistake is visible in the payloads rather than in an absence they'd have to go hunting for.
pub fn parse_rule(value: &[u8], defaults: CaptureRule) -> CaptureRule {
    let s = std::str::from_utf8(value).unwrap_or("").trim();
    if !s.starts_with('{') {
        return defaults;
    }
    let Ok(parsed) = serde_json::from_str::<RuleValue>(s) else {
        return defaults;
    };
    CaptureRule {
        // A `sample_n` of 0 would mean "capture one in zero requests", which is either a division by
        // zero or silently nothing depending on how it's read. Clamp to 1 (capture everything) —
        // consistent with treating the key's presence as the enablement signal.
        sample_n: parsed.sample_n.unwrap_or(defaults.sample_n).max(1),
        max_bytes: parsed.max_bytes.unwrap_or(defaults.max_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULTS: CaptureRule = CaptureRule {
        sample_n: 1,
        max_bytes: 262_144,
    };

    #[test]
    fn default_is_not_capturing() {
        assert_eq!(CaptureSet::new().rule_for(12345), None);
    }

    #[test]
    fn insert_remove_and_lookup() {
        let mut set = CaptureSet::new();
        set.insert(42, DEFAULTS);
        assert_eq!(set.rule_for(42), Some(DEFAULTS));
        assert_eq!(set.rule_for(43), None);
        set.remove(42);
        assert_eq!(set.rule_for(42), None);
        assert!(set.is_empty());
    }

    #[test]
    fn parse_key_accepts_only_its_own_prefix() {
        assert_eq!(parse_key("aicapture.42"), Some(42));
        // The deny-set's keys share a bucket with ours — mistaking one for the other would enable
        // payload capture for every blackholed tenant, which is both wrong and expensive.
        assert_eq!(parse_key("blackhole.42"), None);
        assert_eq!(parse_key("aicapture.notanumber"), None);
        assert_eq!(parse_key("aicapture."), None);
    }

    #[test]
    fn a_malformed_value_still_captures_at_defaults() {
        // Presence of the key is the enablement signal. An operator mid-incident who writes junk
        // (or an empty value, or `{}`) gets capture at defaults rather than silence.
        for value in [b"".as_slice(), b"garbage", b"{}", b"{\"nope\":1}", b"null"] {
            assert_eq!(parse_rule(value, DEFAULTS), DEFAULTS, "{value:?}");
        }
    }

    #[test]
    fn value_fields_override_defaults_independently() {
        assert_eq!(
            parse_rule(br#"{"sample_n":10}"#, DEFAULTS),
            CaptureRule {
                sample_n: 10,
                max_bytes: DEFAULTS.max_bytes
            }
        );
        assert_eq!(
            parse_rule(br#"{"max_bytes":1024}"#, DEFAULTS),
            CaptureRule {
                sample_n: DEFAULTS.sample_n,
                max_bytes: 1024
            }
        );
    }

    #[test]
    fn zero_sample_n_clamps_to_capture_everything() {
        // Reading `sample_n: 0` literally is either a division by zero or "capture nothing"; both
        // are worse than the honest reading of an entry that exists at all, which is "capture".
        let r = parse_rule(br#"{"sample_n":0}"#, DEFAULTS);
        assert_eq!(r.sample_n, 1);
        assert!(r.samples(0) && r.samples(1) && r.samples(7));
    }

    #[test]
    fn bufs_keep_the_head_and_flag_the_drop() {
        let mut b = CaptureBufs::new(8);
        b.push_req(b"abcd");
        b.push_req(b"efghIJKL"); // 4 fit, 4 dropped
        assert_eq!(
            b.req_str(),
            Some("abcdefgh"),
            "keeps the *head*, not the tail"
        );
        assert!(b.req_truncated());
        // Directions are independent — a truncated request must not imply a truncated response.
        assert!(!b.resp_truncated());
        assert_eq!(b.bytes(), 8);
    }

    #[test]
    fn a_body_landing_exactly_on_the_cap_is_not_truncated() {
        // The distinction matters: a capture flagged `truncated` when it is in fact complete sends
        // whoever is reading it hunting for bytes that were never dropped.
        let mut b = CaptureBufs::new(4);
        b.push_resp(b"abcd");
        assert_eq!(b.resp_str(), Some("abcd"));
        assert!(!b.resp_truncated());
        // A further empty chunk still isn't a drop; a non-empty one is.
        b.push_resp(b"");
        assert!(!b.resp_truncated());
        b.push_resp(b"e");
        assert!(b.resp_truncated());
    }

    #[test]
    fn truncation_mid_character_yields_none_rather_than_mojibake() {
        // Cutting a body at a byte boundary can land inside a multi-byte character even when the
        // whole body was valid UTF-8. Reporting `None` (and skipping the field) beats emitting a
        // mangled string into the log.
        let mut b = CaptureBufs::new(2);
        b.push_req("é".as_bytes()); // 2 bytes, fits exactly
        assert_eq!(b.req_str(), Some("é"));

        let mut cut = CaptureBufs::new(1);
        cut.push_req("é".as_bytes()); // 1 of 2 bytes
        assert_eq!(cut.req_str(), None);
        assert!(cut.req_truncated());
    }

    #[test]
    fn a_zero_cap_captures_nothing_but_still_reports_truncation() {
        let mut b = CaptureBufs::new(0);
        b.push_req(b"anything");
        assert_eq!(b.req_str(), Some(""));
        assert!(b.req_truncated());
        assert_eq!(b.bytes(), 0);
    }

    #[test]
    fn sampling_is_one_in_n_and_deterministic() {
        let r = CaptureRule {
            sample_n: 4,
            max_bytes: 1024,
        };
        let hits: Vec<u64> = (0..12).filter(|&s| r.samples(s)).collect();
        assert_eq!(hits, vec![0, 4, 8]);
        // Same sequence number, same answer — the property that makes "why wasn't this captured?"
        // answerable after the fact.
        assert_eq!(r.samples(5), r.samples(5));
    }
}
