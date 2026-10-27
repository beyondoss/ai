//! The body rewrites as they stood before `rewrite::ScannedBody` tied their offsets to the bytes,
//! as the oracle of the `prop_scanned_body_edits_*` properties (`props_lex.rs`, `props_peek.rs`).
//! The free functions below `upstream` are verbatim copies of the old `proxy.rs` ones.
//! `upstream` is not a copy: it re-derives, step for step, the order `request_body_filter` ran
//! them in, without the opaque edits (translation, the strips) between the clamp and the rename.
//!
//! Do not fix anything here. A deliberate change of behaviour changes both sides in one commit.
#![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use beyond_gateway::__test::peek;

/// The old buffered path's edits, in its order, offsets adjusted by hand: the clamp, then (on a
/// native OpenAI Chat attempt) the `max_tokens` rename, each re-scanning when it changed the body;
/// the ambiguous `stream_options` removal, re-scanning too; `force_include_usage` handing back the
/// moved model span; the model rewrite; the injection at the latest scan's `inject_at`, which the
/// two edits after that scan never moved.
pub fn upstream(
    mut buf: Vec<u8>,
    cap: u32,
    rename: bool,
    stream_usage: bool,
    model: Option<&[u8]>,
) -> Vec<u8> {
    let mut scan = peek::scan_buffered(&buf);
    if clamp_output_limits(&mut buf, &scan.limit_spans, cap) {
        scan = peek::scan_buffered(&buf);
    }
    if rename && rename_max_tokens(&mut buf, &scan.limit_keys) {
        scan = peek::scan_buffered(&buf);
    }
    if stream_usage
        && scan.stream_options_ambiguous
        && peek::remove_root_members(&mut buf, "stream_options")
    {
        scan = peek::scan_buffered(&buf);
    }
    let (buf, model_span) = match scan.stream_options_at {
        Some(at) if stream_usage => force_include_usage(buf, at, scan.model_span),
        _ => (buf, scan.model_span),
    };
    let buf = match (model, model_span) {
        (Some(m), Some(span)) => apply_model_rewrite(buf, span, m),
        _ => buf,
    };
    if stream_usage {
        apply_stream_usage_injection(buf, scan.inject_at)
    } else {
        buf
    }
}

/// The fragment spliced into a streaming OpenAI chat body. Always followed by a comma, since the
/// splice point is just inside a root object that is non-empty by construction (a root `"stream"`
/// key is what made it eligible).
pub const STREAM_OPTIONS_FRAG: &[u8] = br#""stream_options":{"include_usage":true},"#;

/// Cap each root-level output limit (`peek::OUTPUT_LIMIT_KEYS`) at `max`, the serving row's
/// published maximum (`ModelCard::output_cap`), in place. Never raises one; `max == 0` (no
/// published maximum, or an embeddings row) caps nothing.
/// `true` when a value changed, so the caller re-scans the moved bytes.
pub fn clamp_output_limits(
    body: &mut Vec<u8>,
    spans: &[Option<(usize, usize)>; 3],
    max: u32,
) -> bool {
    if max == 0 {
        return false;
    }
    // Back to front, so an earlier span's offsets survive a later splice.
    let mut spans = *spans;
    spans.sort_unstable_by(|a, b| b.cmp(a));
    let mut changed = false;
    for (start, end) in spans.into_iter().flatten() {
        let Some(digits) = body.get(start..end) else {
            continue;
        };
        // Digits only (the scan's own span), so a parse fails only past `u64`: over any cap.
        let over = std::str::from_utf8(digits)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .is_none_or(|n| n > u64::from(max));
        if over {
            body.splice(start..end, max.to_string().into_bytes());
            changed = true;
        }
    }
    changed
}

/// Respell a root `max_tokens` as `max_completion_tokens`, in place. `keys` is
/// [`beyond_gateway::peek::BufferedScan::limit_keys`]. When both are present the explicit `max_completion_tokens`
/// wins and the `max_tokens` member is removed. `true` when the body changed, so the caller
/// re-scans the moved bytes.
pub fn rename_max_tokens(body: &mut Vec<u8>, keys: &[Option<usize>; 3]) -> bool {
    const KEY: &[u8] = br#""max_tokens""#;
    let Some(at) = keys[0] else {
        return false;
    };
    if body.get(at..at + KEY.len()) != Some(KEY) {
        return false;
    }
    if keys[1].is_none() {
        body.splice(
            at + 1..at + KEY.len() - 1,
            b"max_completion_tokens".iter().copied(),
        );
        return true;
    }
    // Both spellings. Find the end of the `max_tokens` value without building it, then remove
    // the member and one adjacent comma.
    let skip_ws = |mut i: usize| {
        while body.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        i
    };
    let colon = skip_ws(at + KEY.len());
    if body.get(colon) != Some(&b':') {
        return false;
    }
    let value = skip_ws(colon + 1);
    let mut values =
        serde_json::Deserializer::from_slice(&body[value..]).into_iter::<serde::de::IgnoredAny>();
    if !matches!(values.next(), Some(Ok(_))) {
        return false;
    }
    let end = value + values.byte_offset();
    let after = skip_ws(end);
    let range = if body.get(after) == Some(&b',') {
        at..after + 1
    } else {
        // The last member: take the comma before it instead.
        match body[..at].iter().rposition(|b| !b.is_ascii_whitespace()) {
            Some(comma) if body[comma] == b',' => comma..end,
            _ => at..end,
        }
    };
    body.drain(range);
    true
}

/// Overwrite the `model` value's bytes with the id the chosen candidate uses.
///
/// `span` comes from [`beyond_gateway::peek::BufferedScan::model_span`] and covers the raw value, quotes excluded.
/// The replacement is a catalog string, so it needs no JSON escaping — the catalog charset test
/// (`model_names_are_lowercase_and_log_safe`, plus the same shape for `upstream_model`) is what
/// makes a raw byte copy safe here.
///
/// Returns the body untouched when the id already matches, which is the common case: candidate 0
/// usually spells the model the way the catalog names it, so the primary path does no memmove at all
/// and only a failover pays for one.
pub fn apply_model_rewrite(mut body: Vec<u8>, span: (usize, usize), replacement: &[u8]) -> Vec<u8> {
    let (start, end) = span;
    // Defensive: a span outside the buffer would panic on the splice. Unreachable — the span is
    // produced by the same walk over the same bytes — but this runs on every model-routed request.
    if start > end || end > body.len() {
        return body;
    }
    if body[start..end] == *replacement {
        return body;
    }
    body.splice(start..end, replacement.iter().copied());
    body
}

/// Make a client-sent `stream_options` (value at `at`) ask for usage: `include_usage: false` becomes
/// `true`, an object without it gains it, and a non-object value is replaced by
/// `{"include_usage":true}`. Billing is not the client's to switch off — without the usage chunk the
/// row is an estimate that cannot see hidden reasoning. The client may now receive one extra chunk it
/// did not ask for: OpenAI's usage chunk, `choices: []`, which every SDK already accepts.
///
/// Returns `model_span` moved by the edit when it lay after it. Parses only the `stream_options`
/// value (a few bytes), and only on the rare request that sends one; the value is re-serialized, so
/// its other members keep their meaning but not their spacing.
pub fn force_include_usage(
    mut body: Vec<u8>,
    at: usize,
    model_span: Option<(usize, usize)>,
) -> (Vec<u8>, Option<(usize, usize)>) {
    use serde_json::Value;
    let Some(rest) = body.get(at..) else {
        return (body, model_span);
    };
    let mut values = serde_json::Deserializer::from_slice(rest).into_iter::<Value>();
    let Some(Ok(value)) = values.next() else {
        return (body, model_span);
    };
    let end = at + values.byte_offset();
    let mut options = match value {
        Value::Object(m) if m.get("include_usage") == Some(&Value::Bool(true)) => {
            return (body, model_span);
        }
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    options.insert("include_usage".to_owned(), Value::Bool(true));
    let Ok(replacement) = serde_json::to_vec(&options) else {
        return (body, model_span);
    };
    let (removed, added) = (end - at, replacement.len());
    body.splice(at..end, replacement);
    let shift = |i: usize| i + added - removed;
    let model_span = model_span.map(|(s, e)| if s > at { (shift(s), shift(e)) } else { (s, e) });
    (body, model_span)
}

/// Splice `stream_options.include_usage` into a buffered OpenAI chat body at `at`, or return it
/// unchanged when there is nothing to inject. This is what guarantees a usage chunk — hence a
/// billable token count — from a stock client that never set the option.
///
/// Takes the offset rather than computing it: the caller already walked the body once for both the
/// model and this plan (see `peek::scan_buffered`), and re-deriving it here would restore the second
/// traversal that walk exists to remove.
pub fn apply_stream_usage_injection(mut body: Vec<u8>, at: Option<usize>) -> Vec<u8> {
    let Some(at) = at else { return body };
    // Shift the tail right in place rather than copying the whole body into a second buffer.
    // `req_buf` is pre-sized with `STREAM_OPTIONS_FRAG.len()` of headroom (see `request_filter`), so
    // when the client declared a Content-Length this `resize` is free and the only work is moving
    // the `body.len() - at` bytes after the splice point. The old form allocated a second buffer the
    // size of the whole body and copied every byte into it; at 1 MB that measured 579 µs against
    // 27.8 µs, because the second allocation dominates.
    //
    // Without a declared length (chunked upload) the `resize` may grow once — still a single
    // allocation, i.e. no worse than before.
    let old_len = body.len();
    body.resize(old_len + STREAM_OPTIONS_FRAG.len(), 0);
    body.copy_within(at..old_len, at + STREAM_OPTIONS_FRAG.len());
    body[at..at + STREAM_OPTIONS_FRAG.len()].copy_from_slice(STREAM_OPTIONS_FRAG);
    body
}
