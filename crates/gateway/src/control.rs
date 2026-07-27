//! Per-request control headers — the `x-beyond-*` namespace.
//!
//! One parse seam for everything a caller can say about a single request, read once in
//! `proxy::request_filter` after identity is verified and stripped before the request leaves for the
//! provider. Two members today:
//!
//! | Header | Value | Effect |
//! |---|---|---|
//! | `x-beyond-metadata` | flat JSON object of scalars | tagged onto `ai.usage` + `ai.payload` |
//! | `x-beyond-capture` | `on` / `off` | enable or suppress payload capture for this request |
//!
//! **Nothing here can fail a request.** Every malformed, oversize, or unrecognized value is dropped
//! and counted, and the request proceeds exactly as if the header were absent. An observability
//! header that can 400 a customer's inference call is a worse bug than the missing observability —
//! and this is a *proxy*, where the client's own SDK is what generated the header we'd be rejecting.
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
use std::sync::LazyLock;

/// Tag set for cost attribution: `{"feature":"summarizer","org":"acme"}`.
pub const METADATA_HEADER: &str = "x-beyond-metadata";

/// Per-request capture override: `on` or `off`.
pub const CAPTURE_HEADER: &str = "x-beyond-capture";

/// The same two names as pre-parsed [`HeaderName`]s, which is what [`Control::parse`] actually looks
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

/// Every header this module consumes. Stripped in `upstream_request_filter` so a provider never
/// sees a Beyond control header — they're ours, they'd be meaningless upstream, and a provider that
/// rejects unknown headers would turn our observability feature into their 400.
pub const CONTROL_HEADERS: [&str; 2] = [METADATA_HEADER, CAPTURE_HEADER];

/// Longest metadata header we'll even attempt to parse. Checked **before** parsing so a caller
/// can't make us walk a multi-megabyte JSON document on the request path.
const MAX_METADATA_LEN: usize = 1024;

/// Most tag pairs we'll keep. Cost attribution wants a handful of stable dimensions (feature, org,
/// plan, env); anything past this is either a mistake or an attempt to bloat the billing log.
const MAX_METADATA_PAIRS: usize = 16;

/// Longest single key or rendered value.
const MAX_METADATA_FIELD: usize = 128;

/// What the caller asked for on this request. Both fields are `None` when the header was absent
/// *or* unusable — the two are deliberately indistinguishable to callers of this module, because
/// the handling is identical.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Control {
    /// Canonical JSON object, built by us from validated pairs. `None` when absent or rejected.
    pub metadata: Option<String>,
    /// `Some(true)` = capture this request, `Some(false)` = don't, `None` = no opinion (fall back to
    /// the tenant's control-plane rule).
    pub capture: Option<bool>,
    /// A header was present but unusable. Drives `control_header_errors_total` — without it a
    /// client whose tags silently never appear has no signal to debug against.
    pub malformed: bool,
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
            match raw.to_str().ok().and_then(parse_capture) {
                Some(c) => out.capture = Some(c),
                None => out.malformed = true,
            }
        }

        out
    }
}

/// `on` / `off`, case-insensitively. Deliberately not accepting `true`/`1`/`yes`: a narrow spelling
/// makes a typo visible on the error counter instead of silently meaning the opposite of what the
/// caller intended.
fn parse_capture(raw: &str) -> Option<bool> {
    match raw.trim() {
        v if v.eq_ignore_ascii_case("on") => Some(true),
        v if v.eq_ignore_ascii_case("off") => Some(false),
        _ => None,
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
}
