//! Per-request control headers — the `x-beyond-*` namespace.
//!
//! One parse seam for everything a caller can say about a single request, read once in
//! `proxy::request_filter` after identity is verified and stripped before the request leaves for the
//! provider. Members today:
//!
//! | Header | Value | Effect |
//! |---|---|---|
//! | `x-beyond-metadata` | flat JSON object of scalars | tagged onto `ai.usage` + `ai.payload` |
//! | `x-beyond-capture` | `on` / `off` | enable or suppress payload capture for this request |
//! | `x-beyond-cache` | `on` / `off` | enable or skip the exact-match response cache for this request |
//! | `x-beyond-order` | `bedrock,anthropic` | those providers first (stable), then the rest of the row |
//! | `x-beyond-only` | `bedrock,openrouter` | drop anyone on the row not named |
//! | `x-beyond-split` | `anthropic=70,bedrock=30` | pick the primary by those weights; leftover stay failover |
//!
//! Walk headers permute a catalog row's candidate list. They never add a provider the row does not
//! already list, they do not change the wire, and they run **before** `first_usable` / breaker skip
//! so failover, breakers, and the 429 key-walk see the permuted sequence. Names must match
//! [`providers::ProviderSpec::name`] on that row; unknown names are dropped. An `only` filter that
//! leaves nothing usable is a 503 from routing (the same as no pool-keyed candidate) — not a 4xx
//! from this module.
//!
//! `order` and `split` **pin** the walk: the TTFT ranker in [`crate::smart`] does not run. `only`
//! is a filter, then the ranker still applies. No walk header at all is also ranked.
//!
//! **Nothing here can fail a request.** Every malformed, oversize, or unrecognized value is dropped
//! and counted, and the request proceeds exactly as if the header were absent. An observability
//! header that can 400 a customer's inference call is a worse bug than the missing observability —
//! and this is a *proxy*, where the client's own SDK is what generated the header we'd be rejecting.
//! Unparseable walk headers are that case: default catalog order, plus the error counter.
//!
//! **Metadata is re-serialized, never passed through.** We parse the client's JSON, validate it, and
//! emit JSON we build ourselves from the parsed values. That makes log injection structurally
//! impossible rather than filtered-for: no arrangement of client bytes can escape the field, because
//! the client's bytes are never what we write. Keys are sorted so the same tags always render the
//! same way — deterministic to test against, and cheap to dedup downstream.
//!
//! Managed traffic only. A BYO request carries no verified `tenant_id`, so a tag on it would be an
//! unattributable row — the same reason `ai.usage` itself is managed-only (see `proxy::logging`).

use http::header::HeaderName;
use pingora::http::RequestHeader;
use providers::{Candidate, MAX_CANDIDATES, by_id};
use std::sync::LazyLock;

/// Tag set for cost attribution: `{"feature":"summarizer","org":"acme"}`.
pub const METADATA_HEADER: &str = "x-beyond-metadata";

/// Per-request capture override: `on` or `off`.
pub const CAPTURE_HEADER: &str = "x-beyond-capture";

/// Per-request exact-match cache override: `on` or `off`. `off` skips lookup and store.
pub const CACHE_HEADER: &str = "x-beyond-cache";

/// Catalog-walk preference: named providers first, then the rest of the row, catalog-relative.
pub const ORDER_HEADER: &str = "x-beyond-order";

/// Catalog-walk filter: keep only the named providers that already sit on the row.
pub const ONLY_HEADER: &str = "x-beyond-only";

/// Catalog-walk weighted primary: `name=weight` pairs; leftover candidates stay failover.
pub const SPLIT_HEADER: &str = "x-beyond-split";

/// The same names as pre-parsed [`HeaderName`]s, which is what [`Control::parse`] actually looks
/// up with.
///
/// `HeaderMap::get(&str)` re-hashes the name on every call; `get(&HeaderName)` uses the hash the
/// name already carries. Both of these are read on **every managed request**, including the vast
/// majority that sent neither — so the difference is pure overhead on the hot path. Measured on the
/// dev host (`benches/unit.rs`, `capture::control_parse_absent`): **39.8 ns → 19.8 ns** fastest,
/// 39.8 → 29.8 ns median. Same reasoning as the boot-built `HeaderValue`s in
/// `proxy::upstream_request_filter`.
static METADATA_NAME: LazyLock<HeaderName> =
    LazyLock::new(|| HeaderName::from_static(METADATA_HEADER));
static CAPTURE_NAME: LazyLock<HeaderName> =
    LazyLock::new(|| HeaderName::from_static(CAPTURE_HEADER));
static CACHE_NAME: LazyLock<HeaderName> = LazyLock::new(|| HeaderName::from_static(CACHE_HEADER));
static ORDER_NAME: LazyLock<HeaderName> = LazyLock::new(|| HeaderName::from_static(ORDER_HEADER));
static ONLY_NAME: LazyLock<HeaderName> = LazyLock::new(|| HeaderName::from_static(ONLY_HEADER));
static SPLIT_NAME: LazyLock<HeaderName> = LazyLock::new(|| HeaderName::from_static(SPLIT_HEADER));

/// Every header this module consumes. Stripped in `upstream_request_filter` so a provider never
/// sees a Beyond control header — they're ours, they'd be meaningless upstream, and a provider that
/// rejects unknown headers would turn our observability feature into their 400.
pub const CONTROL_HEADERS: [&str; 6] = [
    METADATA_HEADER,
    CAPTURE_HEADER,
    CACHE_HEADER,
    ORDER_HEADER,
    ONLY_HEADER,
    SPLIT_HEADER,
];

/// Longest metadata header we'll even attempt to parse. Checked **before** parsing so a caller
/// can't make us walk a multi-megabyte JSON document on the request path.
const MAX_METADATA_LEN: usize = 1024;

/// Most tag pairs we'll keep. Cost attribution wants a handful of stable dimensions (feature, org,
/// plan, env); anything past this is either a mistake or an attempt to bloat the billing log.
const MAX_METADATA_PAIRS: usize = 16;

/// Longest single key or rendered value.
const MAX_METADATA_FIELD: usize = 128;

/// What the caller asked for on this request. Each member is `None` when the header was absent
/// *or* unusable — the two are deliberately indistinguishable to callers of this module, because
/// the handling is identical.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Control {
    /// Canonical JSON object, built by us from validated pairs. `None` when absent or rejected.
    pub metadata: Option<String>,
    /// `Some(true)` = capture this request, `Some(false)` = don't, `None` = no opinion (fall back to
    /// the tenant's control-plane rule).
    pub capture: Option<bool>,
    /// `Some(true)` = cache this request (the default when the store is on), `Some(false)` = skip
    /// lookup and store. `None` = no opinion. `Cache-Control: no-store` is checked separately and
    /// also skips.
    pub cache: Option<bool>,
    /// Named providers to front-load, in the order written. `None` when absent or unusable.
    pub order: Option<Vec<String>>,
    /// Named providers to keep. `None` when absent or unusable. An empty-after-apply list is a
    /// routing 503, not a parse failure.
    pub only: Option<Vec<String>>,
    /// Weighted primary: `(ProviderSpec::name, weight)`. `None` when absent or unusable.
    pub split: Option<Vec<(String, u32)>>,
    /// A header was present but unusable. Drives `control_header_errors_total` — without it a
    /// client whose tags silently never appear has no signal to debug against.
    pub malformed: bool,
}

/// Catalog indices in the order this request will walk them.
///
/// Bit `i` of the gateway's `usable` mask refers to walk slot `i`, which maps to
/// `row.candidates[indices[i]]`. Identity (`indices[i] = i`) when no walk header applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Walk {
    pub indices: [u8; MAX_CANDIDATES],
    pub len: u8,
}

impl Walk {
    /// Catalog order, truncated to [`MAX_CANDIDATES`].
    pub fn identity(n: usize) -> Self {
        let len = n.min(MAX_CANDIDATES) as u8;
        let mut indices = [0u8; MAX_CANDIDATES];
        for i in 0..len {
            indices[i as usize] = i;
        }
        Self { indices, len }
    }

    /// Original catalog index for walk slot `i`, if that slot exists.
    pub fn catalog_index(self, i: u8) -> Option<u8> {
        (i < self.len).then_some(self.indices[i as usize])
    }

    /// `ProviderId::index` bytes in walk order — the exact-match cache's per-arm discriminator.
    pub fn provider_ids(self, candidates: &[Candidate]) -> ([u8; MAX_CANDIDATES], u8) {
        let mut ids = [0u8; MAX_CANDIDATES];
        let mut n = 0u8;
        for i in 0..self.len {
            let Some(c) = candidates.get(usize::from(self.indices[i as usize])) else {
                continue;
            };
            ids[n as usize] = c.provider.index() as u8;
            n += 1;
        }
        (ids, n)
    }
}

impl Control {
    /// Read and validate the `x-beyond-*` headers. Never fails; see the module docs.
    pub fn parse(req: &RequestHeader) -> Self {
        let mut out = Control::default();

        // A header whose bytes aren't UTF-8 is unusable — `to_str` failing is itself a rejection,
        // not an "absent", so it counts toward `malformed`.
        if let Some(raw) = req.headers.get(&*METADATA_NAME) {
            match raw.to_str().ok().and_then(parse_metadata) {
                Some(m) => out.metadata = Some(m),
                None => out.malformed = true,
            }
        }

        if let Some(raw) = req.headers.get(&*CAPTURE_NAME) {
            match raw.to_str().ok().and_then(parse_on_off) {
                Some(c) => out.capture = Some(c),
                None => out.malformed = true,
            }
        }

        if let Some(raw) = req.headers.get(&*CACHE_NAME) {
            match raw.to_str().ok().and_then(parse_on_off) {
                Some(c) => out.cache = Some(c),
                None => out.malformed = true,
            }
        }

        if let Some(raw) = req.headers.get(&*ORDER_NAME) {
            match raw.to_str().ok().and_then(parse_name_list) {
                Some(v) => out.order = Some(v),
                None => out.malformed = true,
            }
        }

        if let Some(raw) = req.headers.get(&*ONLY_NAME) {
            match raw.to_str().ok().and_then(parse_name_list) {
                Some(v) => out.only = Some(v),
                None => out.malformed = true,
            }
        }

        if let Some(raw) = req.headers.get(&*SPLIT_NAME) {
            match raw.to_str().ok().and_then(parse_split) {
                Some(v) => out.split = Some(v),
                None => out.malformed = true,
            }
        }

        out
    }

    /// True when this request named an explicit walk (`order` or `split`). The TTFT ranker must
    /// not override a caller who already picked. `only` is a filter, not a pin.
    pub fn pins_walk(&self) -> bool {
        self.order.is_some() || self.split.is_some()
    }

    /// Catalog indices in the order this request will walk them.
    ///
    /// `only` filters, then `order` front-loads named providers, then `split` picks the primary
    /// from its weighted names (deterministic in `seed`, not `rand`). Unknown names are dropped.
    /// Does not add a provider the row does not already list.
    pub fn catalog_walk(&self, candidates: &[Candidate], seed: u64) -> Walk {
        permute(
            candidates,
            self.only.as_deref(),
            self.order.as_deref(),
            self.split.as_deref(),
            seed,
        )
    }
}

/// `on` / `off`, case-insensitively. Deliberately not accepting `true`/`1`/`yes`: a narrow spelling
/// makes a typo visible on the error counter instead of silently meaning the opposite of what the
/// caller intended. Shared by `x-beyond-capture` and `x-beyond-cache`.
fn parse_on_off(raw: &str) -> Option<bool> {
    match raw.trim() {
        v if v.eq_ignore_ascii_case("on") => Some(true),
        v if v.eq_ignore_ascii_case("off") => Some(false),
        _ => None,
    }
}

/// Comma-separated `ProviderSpec::name` tokens. Empty / whitespace-only / no tokens → unusable.
fn parse_name_list(raw: &str) -> Option<Vec<String>> {
    let mut names = Vec::new();
    for part in raw.split(',') {
        let name = part.trim();
        if name.is_empty() {
            continue;
        }
        if names.len() == MAX_CANDIDATES {
            break;
        }
        names.push(name.to_string());
    }
    (!names.is_empty()).then_some(names)
}

/// Comma-separated `name=weight` pairs. Weights are unsigned integers; a zero-weight arm is kept
/// so apply can skip it. No valid pair → unusable (counted, default order).
fn parse_split(raw: &str) -> Option<Vec<(String, u32)>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, weight) = part.split_once('=')?;
        let name = name.trim();
        let weight = weight.trim();
        if name.is_empty() || weight.is_empty() || !weight.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let w: u32 = weight.parse().ok()?;
        if out.len() == MAX_CANDIDATES {
            break;
        }
        out.push((name.to_string(), w));
    }
    out.iter().any(|(_, w)| *w > 0).then_some(out)
}

/// Stable mix so adjacent `seed` values (a request counter) do not all land on the same side of a
/// 70/30 cut. Not `std`'s `DefaultHasher`: that carries a per-process key, which is the
/// rand-per-replica chaos a split header exists to avoid.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn provider_name(c: &Candidate) -> &'static str {
    by_id(c.provider).name
}

fn permute(
    candidates: &[Candidate],
    only: Option<&[String]>,
    order: Option<&[String]>,
    split: Option<&[(String, u32)]>,
    seed: u64,
) -> Walk {
    let n_orig = candidates.len().min(MAX_CANDIDATES) as u8;
    let mut idx = [0u8; MAX_CANDIDATES];
    let mut n = 0u8;

    if let Some(only) = only {
        for i in 0..n_orig {
            if only
                .iter()
                .any(|s| s == provider_name(&candidates[i as usize]))
            {
                idx[n as usize] = i;
                n += 1;
            }
        }
    } else {
        for i in 0..n_orig {
            idx[i as usize] = i;
        }
        n = n_orig;
    }

    if let Some(order) = order {
        let mut out = [0u8; MAX_CANDIDATES];
        let mut k = 0u8;
        let mut used = 0u8;
        for want in order {
            for j in 0..n {
                if used & (1 << j) != 0 {
                    continue;
                }
                if provider_name(&candidates[idx[j as usize] as usize]) == want {
                    out[k as usize] = idx[j as usize];
                    k += 1;
                    used |= 1 << j;
                    break;
                }
            }
        }
        for j in 0..n {
            if used & (1 << j) == 0 {
                out[k as usize] = idx[j as usize];
                k += 1;
            }
        }
        idx = out;
        n = k;
    }

    if let Some(split) = split {
        let mut arms = [(0u8, 0u32); MAX_CANDIDATES];
        let mut nw = 0u8;
        let mut total = 0u32;
        for (want, w) in split {
            if *w == 0 {
                continue;
            }
            for j in 0..n {
                if provider_name(&candidates[idx[j as usize] as usize]) == want
                    && !arms[..nw as usize].iter().any(|(s, _)| *s == j)
                {
                    arms[nw as usize] = (j, *w);
                    total = total.saturating_add(*w);
                    nw += 1;
                    break;
                }
            }
        }
        if nw > 0 && total > 0 {
            let r = (mix64(seed) % u64::from(total)) as u32;
            let mut acc = 0u32;
            let mut pick_slot = arms[0].0;
            for &(slot, w) in &arms[..nw as usize] {
                acc += w;
                if r < acc {
                    pick_slot = slot;
                    break;
                }
            }
            let primary = idx[pick_slot as usize];
            let mut out = [0u8; MAX_CANDIDATES];
            out[0] = primary;
            let mut k = 1u8;
            for j in 0..n {
                if idx[j as usize] != primary {
                    out[k as usize] = idx[j as usize];
                    k += 1;
                }
            }
            idx = out;
            n = k;
        }
    }

    Walk {
        indices: idx,
        len: n,
    }
}

/// Parse, validate, and canonically re-serialize the metadata object. `None` rejects the whole
/// header rather than keeping the valid subset: a half-applied tag set is worse than no tags, since
/// a `GROUP BY` over it silently under-counts instead of visibly missing.
fn parse_metadata(raw: &str) -> Option<String> {
    if raw.len() > MAX_METADATA_LEN {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_str(raw).ok()?;
    let obj = parsed.as_object()?;
    if obj.is_empty() || obj.len() > MAX_METADATA_PAIRS {
        return None;
    }

    // `serde_json::Map` iterates in sorted order under the default `preserve_order = false` feature
    // set, but that's a build-time property of a dependency rather than a promise to us — sort
    // explicitly so canonical output can't quietly depend on how the workspace resolves features.
    let mut pairs: Vec<(&str, String)> = Vec::with_capacity(obj.len());
    for (key, value) in obj {
        if !is_valid_field(key) {
            return None;
        }
        let rendered = match value {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            // Null, arrays, and nested objects are out: the target is a flat string→string map, and
            // silently flattening or stringifying them would put a shape in the column that the
            // caller didn't ask for and can't predict.
            _ => return None,
        };
        if !is_valid_field(&rendered) {
            return None;
        }
        pairs.push((key.as_str(), rendered));
    }
    pairs.sort_unstable_by(|a, b| a.0.cmp(b.0));

    // Re-serialize from the validated pairs. `serde_json` owns the escaping, so the result is
    // well-formed JSON by construction no matter what the client sent.
    let canonical: serde_json::Map<String, serde_json::Value> = pairs
        .into_iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::String(v)))
        .collect();
    serde_json::to_string(&canonical).ok()
}

/// A key or rendered value we're willing to put in the billing log.
///
/// Non-empty, bounded, and free of the bytes that break line-oriented log shipping — control
/// characters and `DEL`. Quotes and backslashes are *allowed* here, unlike `proxy::sanitize_model`:
/// that function hands its output to `tracing` as a bare field, whereas everything here is escaped
/// by `serde_json` on the way out, so the injection those characters enable can't happen. Rejecting
/// them anyway would break ordinary tag values like `He said "hi"`.
fn is_valid_field(s: &str) -> bool {
    !s.is_empty() && s.len() <= MAX_METADATA_FIELD && !s.bytes().any(|b| b < 0x20 || b == 0x7f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pingora::http::RequestHeader;

    fn req(headers: &[(&str, &str)]) -> RequestHeader {
        let mut r = RequestHeader::build("POST", b"/v1/messages", None).expect("build header");
        for (k, v) in headers {
            r.insert_header(k.to_string(), *v).expect("insert header");
        }
        r
    }

    #[test]
    fn absent_headers_yield_no_opinion_and_no_error() {
        let c = Control::parse(&req(&[]));
        assert_eq!(c, Control::default());
        assert!(!c.malformed);
    }

    #[test]
    fn metadata_is_canonicalized_and_key_sorted() {
        // Same tags in a different order must render identically — downstream dedup and test
        // assertions both depend on it.
        let a = Control::parse(&req(&[(
            METADATA_HEADER,
            r#"{"org":"acme","feature":"summarizer"}"#,
        )]));
        let b = Control::parse(&req(&[(
            METADATA_HEADER,
            r#"{"feature":"summarizer","org":"acme"}"#,
        )]));
        assert_eq!(
            a.metadata.as_deref(),
            Some(r#"{"feature":"summarizer","org":"acme"}"#)
        );
        assert_eq!(a.metadata, b.metadata);
        assert!(!a.malformed);
    }

    #[test]
    fn scalar_values_are_stringified() {
        // Numbers and bools are legitimate tags (`{"plan":3,"beta":true}`); the target column is
        // string→string, so they render as their JSON text rather than being rejected.
        let c = Control::parse(&req(&[(METADATA_HEADER, r#"{"beta":true,"plan":3}"#)]));
        assert_eq!(c.metadata.as_deref(), Some(r#"{"beta":"true","plan":"3"}"#));
    }

    #[test]
    fn log_injection_is_structurally_impossible() {
        // The whole point of re-serializing: a value that would close the JSON string and inject a
        // sibling field comes back escaped, not filtered — and the field count is still one.
        let c = Control::parse(&req(&[(
            METADATA_HEADER,
            r#"{"feature":"real\",\"tenant_id\":\"999"}"#,
        )]));
        let out = c.metadata.expect("value is legal, just adversarial");
        let back: serde_json::Value = serde_json::from_str(&out).expect("valid JSON out");
        let obj = back.as_object().expect("object");
        assert_eq!(obj.len(), 1, "no injected sibling field: {out}");
        assert_eq!(obj["feature"], r#"real","tenant_id":"999"#);
    }

    #[test]
    fn rejects_non_objects_nested_values_and_empty() {
        for bad in [
            "[1,2,3]",            // array at the root
            r#""just a string""#, // scalar at the root
            "{}",                 // nothing to tag with
            r#"{"a":{"b":1}}"#,   // nested object
            r#"{"a":[1]}"#,       // array value
            r#"{"a":null}"#,      // null value
            r#"{"feature":"x""#,  // truncated / not JSON
        ] {
            let c = Control::parse(&req(&[(METADATA_HEADER, bad)]));
            assert!(c.metadata.is_none(), "should reject: {bad}");
            assert!(c.malformed, "should count as malformed: {bad}");
        }
    }

    #[test]
    fn rejects_control_bytes_smuggled_in_as_json_escapes() {
        // The escaped form is the entire threat model here. A *raw* control byte can never reach
        // this function: `HeaderValue` rejects bytes below 0x20 at HTTP parse time, so the request
        // dies before `Control::parse` ever runs. (Trying to build one in this test panics inside
        // `insert_header` \u{2014} which is how this test found its own bug.)
        //
        // A JSON escape walks straight through that guard. Every payload below is printable ASCII
        // on the wire and only becomes a newline/tab/NUL/DEL once `serde_json` decodes it. A
        // decoded newline in a tag would split one billing row into two lines for a line-oriented
        // shipper \u{2014} which is what `is_valid_field` exists to stop, and why it has to run on the
        // *decoded* value rather than on the header bytes.
        for bad in [
            r#"{"a":"line1\nline2"}"#, // decodes to a real newline in the value
            r#"{"a\tb":"v"}"#,         // ...and in the key
            r#"{"a":"\u0000"}"#,       // NUL, which a raw byte could never have delivered
            r#"{"a":"\u007f"}"#,       // DEL, likewise
        ] {
            assert!(
                bad.is_ascii(),
                "must be legal header bytes on the wire: {bad}"
            );
            let c = Control::parse(&req(&[(METADATA_HEADER, bad)]));
            assert!(c.metadata.is_none(), "should reject: {bad}");
            assert!(c.malformed, "should count as malformed: {bad}");
        }
    }

    #[test]
    fn rejects_oversize_header_without_parsing_it() {
        // Bounded before `from_str` is ever called — the guard is against the parse cost, so
        // testing it via a payload that is *only* too long (and otherwise perfectly valid) is the
        // case that matters.
        let big = format!(r#"{{"a":"{}"}}"#, "x".repeat(MAX_METADATA_LEN));
        assert!(big.len() > MAX_METADATA_LEN);
        let c = Control::parse(&req(&[(METADATA_HEADER, &big)]));
        assert!(c.metadata.is_none());
        assert!(c.malformed);
    }

    #[test]
    fn rejects_too_many_pairs_and_overlong_fields() {
        let many: String = {
            let inner: Vec<String> = (0..=MAX_METADATA_PAIRS)
                .map(|i| format!(r#""k{i}":"v""#))
                .collect();
            format!("{{{}}}", inner.join(","))
        };
        assert!(
            Control::parse(&req(&[(METADATA_HEADER, &many)]))
                .metadata
                .is_none()
        );

        let long_value = format!(r#"{{"a":"{}"}}"#, "x".repeat(MAX_METADATA_FIELD + 1));
        assert!(
            Control::parse(&req(&[(METADATA_HEADER, &long_value)]))
                .metadata
                .is_none()
        );
    }

    #[test]
    fn capture_accepts_on_off_case_insensitively() {
        for (raw, want) in [
            ("on", true),
            ("ON", true),
            (" On ", true),
            ("off", false),
            ("OFF", false),
        ] {
            let c = Control::parse(&req(&[(CAPTURE_HEADER, raw)]));
            assert_eq!(c.capture, Some(want), "{raw}");
            assert!(!c.malformed, "{raw}");
        }
    }

    #[test]
    fn capture_rejects_other_spellings_rather_than_guessing() {
        // `true`/`1`/`yes` are *not* accepted: silently guessing turns a typo into the opposite of
        // what the caller meant, where a rejection shows up on the error counter.
        for raw in ["true", "1", "yes", "enabled", ""] {
            let c = Control::parse(&req(&[(CAPTURE_HEADER, raw)]));
            assert_eq!(c.capture, None, "{raw}");
            assert!(c.malformed, "{raw}");
        }
    }

    #[test]
    fn one_bad_header_does_not_discard_the_other() {
        // Independent members: a junk capture value must not cost the caller their tags.
        let c = Control::parse(&req(&[
            (METADATA_HEADER, r#"{"feature":"chat"}"#),
            (CAPTURE_HEADER, "maybe"),
        ]));
        assert_eq!(c.metadata.as_deref(), Some(r#"{"feature":"chat"}"#));
        assert_eq!(c.capture, None);
        assert!(c.malformed);
    }

    #[test]
    fn cache_accepts_on_off_case_insensitively() {
        for (raw, want) in [("on", true), ("OFF", false), (" Off ", false)] {
            let c = Control::parse(&req(&[(CACHE_HEADER, raw)]));
            assert_eq!(c.cache, Some(want), "{raw}");
            assert!(!c.malformed, "{raw}");
        }
    }

    #[test]
    fn cache_rejects_other_spellings_rather_than_guessing() {
        for raw in ["true", "1", "yes", ""] {
            let c = Control::parse(&req(&[(CACHE_HEADER, raw)]));
            assert_eq!(c.cache, None, "{raw}");
            assert!(c.malformed, "{raw}");
        }
    }

    fn claude_row() -> &'static providers::ModelRoute {
        providers::catalog::for_model("claude-opus-4-8").expect("catalog row")
    }

    fn walk_names(walk: Walk, candidates: &[Candidate]) -> Vec<&'static str> {
        (0..walk.len)
            .map(|i| provider_name(&candidates[usize::from(walk.indices[i as usize])]))
            .collect()
    }

    #[test]
    fn order_and_only_parse_comma_lists() {
        let c = Control::parse(&req(&[
            (ORDER_HEADER, " bedrock, anthropic "),
            (ONLY_HEADER, "bedrock,openrouter"),
        ]));
        assert_eq!(
            c.order.as_deref(),
            Some(["bedrock".to_string(), "anthropic".to_string()].as_slice())
        );
        assert_eq!(
            c.only.as_deref(),
            Some(["bedrock".to_string(), "openrouter".to_string()].as_slice())
        );
        assert!(!c.malformed);
    }

    #[test]
    fn split_parses_name_weight_pairs() {
        let c = Control::parse(&req(&[(SPLIT_HEADER, "anthropic=70, bedrock=30")]));
        assert_eq!(
            c.split.as_deref(),
            Some([("anthropic".to_string(), 70), ("bedrock".to_string(), 30)].as_slice())
        );
        assert!(!c.malformed);
    }

    #[test]
    fn junk_walk_headers_are_dropped_and_counted() {
        for (header, raw) in [
            (ORDER_HEADER, ""),
            (ORDER_HEADER, "   ,  ,"),
            (ONLY_HEADER, ""),
            (SPLIT_HEADER, "nope"),
            (SPLIT_HEADER, "anthropic"),
            (SPLIT_HEADER, "anthropic="),
            (SPLIT_HEADER, "=70"),
            (SPLIT_HEADER, "anthropic=-1"),
            (SPLIT_HEADER, "anthropic=70.5"),
            (SPLIT_HEADER, "anthropic=0,bedrock=0"),
        ] {
            let c = Control::parse(&req(&[(header, raw)]));
            assert!(
                c.order.is_none() && c.only.is_none() && c.split.is_none(),
                "{header}={raw}"
            );
            assert!(c.malformed, "{header}={raw}");
        }
    }

    #[test]
    fn order_front_loads_named_providers_then_the_rest() {
        let row = claude_row();
        let c = Control::parse(&req(&[(ORDER_HEADER, "bedrock")]));
        let walk = c.catalog_walk(row.candidates, 0);
        assert_eq!(
            walk_names(walk, row.candidates),
            ["bedrock", "anthropic", "openrouter"]
        );
    }

    #[test]
    fn only_drops_anyone_not_named() {
        let row = claude_row();
        let c = Control::parse(&req(&[(ONLY_HEADER, "bedrock,openrouter")]));
        let walk = c.catalog_walk(row.candidates, 0);
        assert_eq!(walk_names(walk, row.candidates), ["bedrock", "openrouter"]);
    }

    #[test]
    fn unknown_names_are_dropped_without_adding_off_row_providers() {
        let row = claude_row();
        let c = Control::parse(&req(&[(ORDER_HEADER, "openai,bedrock,not-a-provider")]));
        let walk = c.catalog_walk(row.candidates, 0);
        // openai is a real provider but not on this row — must not appear.
        assert_eq!(
            walk_names(walk, row.candidates),
            ["bedrock", "anthropic", "openrouter"]
        );
        assert!(!c.malformed);
    }

    #[test]
    fn only_of_unknown_names_leaves_an_empty_walk() {
        let row = claude_row();
        let c = Control::parse(&req(&[(ONLY_HEADER, "openai")]));
        let walk = c.catalog_walk(row.candidates, 0);
        assert_eq!(walk.len, 0);
    }

    #[test]
    fn split_picks_a_weighted_primary_and_keeps_leftover_in_catalog_order() {
        let row = claude_row();
        let c = Control::parse(&req(&[(SPLIT_HEADER, "anthropic=70,bedrock=30")]));
        let mut saw_anthropic = false;
        let mut saw_bedrock = false;
        for seed in 0..256 {
            let walk = c.catalog_walk(row.candidates, seed);
            let names = walk_names(walk, row.candidates);
            assert_eq!(names.len(), 3, "{names:?}");
            match names[0] {
                "anthropic" => {
                    saw_anthropic = true;
                    assert_eq!(names, ["anthropic", "bedrock", "openrouter"]);
                }
                "bedrock" => {
                    saw_bedrock = true;
                    assert_eq!(names, ["bedrock", "anthropic", "openrouter"]);
                }
                other => panic!("split must pick a named arm, not {other}"),
            }
        }
        assert!(
            saw_anthropic && saw_bedrock,
            "a 70/30 split over many seeds must hit both primaries"
        );
    }

    #[test]
    fn default_walk_is_catalog_order() {
        let row = claude_row();
        let walk = Control::default().catalog_walk(row.candidates, 0);
        assert_eq!(
            walk_names(walk, row.candidates),
            ["anthropic", "bedrock", "openrouter"]
        );
        assert_eq!(walk, Walk::identity(row.candidates.len()));
    }

    #[test]
    fn order_and_split_pin_the_walk_only_does_not() {
        let order = Control::parse(&req(&[(ORDER_HEADER, "bedrock")]));
        assert!(order.pins_walk());
        let split = Control::parse(&req(&[(SPLIT_HEADER, "anthropic=1")]));
        assert!(split.pins_walk());
        let only = Control::parse(&req(&[(ONLY_HEADER, "bedrock")]));
        assert!(!only.pins_walk());
        assert!(!Control::default().pins_walk());
    }
}
