//! The outgoing body's byte edits, on a body that carries its own scan.
//!
//! A buffered body is walked once ([`peek::scan_buffered`]) for the offsets its rewrites splice at:
//! the `model` value, each output limit's digits and key, where `stream_options` goes or already
//! sits. An edit that moves bytes moves every offset past it, so an offset held across an edit
//! points into the wrong bytes, and a splice there sends a corrupted body upstream. [`ScannedBody`]
//! owns the bytes and their scan as one value: each edit is a method that consumes it and hands it
//! back with every offset moved by that edit (`Splice`), and no offset leaves it. An edit it cannot
//! see into goes through [`Edit`], which gives the bytes back only re-scanned. So no code outside
//! this module can hold an offset across an edit: the borrow or the move that the edit takes ends it.
//!
//! Every function here indexes client bytes, so a slip is a panic on a hostile body: indexing and
//! unchecked arithmetic are denied, as in `peek`.
#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::peek::{self, BufferedScan};

/// The fragment spliced into a streaming OpenAI chat body. Always followed by a comma, since the
/// splice point is just inside a root object that is non-empty by construction (a root `"stream"`
/// key is what made it eligible).
pub(crate) const STREAM_OPTIONS_FRAG: &[u8] = br#""stream_options":{"include_usage":true},"#;

/// A buffered request body and the scan of exactly these bytes.
///
/// The scan's offsets are private and move with every edit, so they always index the bytes beside
/// them. What is readable outside is what no edit here invalidates: the bytes themselves (borrowed,
/// so not across an edit) and the facts the body rules read.
pub struct ScannedBody {
    buf: Vec<u8>,
    scan: BufferedScan,
}

impl ScannedBody {
    /// Scan `buf`: the one walk every offset below comes from.
    pub fn new(buf: Vec<u8>) -> Self {
        let scan = peek::scan_buffered(&buf);
        Self { buf, scan }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.buf
    }

    /// The root `model`, unescaped ([`BufferedScan::model`]).
    pub(crate) fn model(&self) -> Option<&str> {
        self.scan.model.as_deref()
    }

    /// Move the root `model` out; later reads see `None` until a re-scan.
    pub(crate) fn take_model(&mut self) -> Option<String> {
        self.scan.model.take()
    }

    /// [`BufferedScan::duplicate_model`].
    pub(crate) fn duplicate_model(&self) -> bool {
        self.scan.duplicate_model
    }

    /// [`BufferedScan::inexact_limit`].
    pub(crate) fn inexact_limit(&self) -> bool {
        self.scan.inexact_limit
    }

    /// Open the bytes to an edit this type cannot see into (translation, the strips). The scan
    /// stays sealed inside [`Edit`] until [`Edit::finish`] says whether the bytes changed.
    pub(crate) fn edit(self) -> Edit {
        Edit { body: self }
    }

    /// Cap each root-level output limit (`peek::OUTPUT_LIMIT_KEYS`) at `max`, the serving row's
    /// published maximum (`ModelCard::output_cap`), in place. Never raises one; `max == 0` (no
    /// published maximum, or an embeddings row) caps nothing. Each cut moves every offset past it,
    /// so the limits can be cut in any order and the body needs no re-scan.
    pub fn clamp_output_limits(mut self, max: u32) -> Self {
        if max == 0 {
            return self;
        }
        for k in 0..peek::OUTPUT_LIMIT_KEYS.len() {
            // Read per limit, not up front: an earlier cut may have moved it.
            let Some(span) = self.scan.limit_spans.get(k).copied().flatten() else {
                continue;
            };
            if let Some(splice) = clamp_limit(&mut self.buf, span, max) {
                self.moved(splice);
            }
        }
        self
    }

    /// Respell a root `max_tokens` as `max_completion_tokens` (see `rename_max_tokens`),
    /// re-scanned when it changed: a rename re-indexes the limits, and a removal takes a member out.
    /// Only on an attempt onto native OpenAI Chat, so the second walk is off every other path.
    pub fn rename_max_tokens(mut self) -> Self {
        match rename_max_tokens(&mut self.buf, &self.scan.limit_keys) {
            true => Self::new(self.buf),
            false => self,
        }
    }

    /// The body as it goes upstream: the last edits, in one fixed order.
    ///
    /// `stream_usage` is the OpenAI-upstream gate (`BodyPlan::stream_options`); `model` is the
    /// serving candidate's id, on a model-routed request. In order:
    ///
    /// 1. An ambiguous `stream_options` (duplicated, or escaped: OpenAI takes the last, decoded) is
    ///    removed and the body re-scanned, so step 4 adds the one that counts (D88).
    /// 2. A client-sent `stream_options` is made to ask for usage (`force_include_usage`).
    /// 3. `model` is overwritten with the candidate's id (`overwrite`).
    /// 4. A stream that sent no `stream_options` gets [`STREAM_OPTIONS_FRAG`] just inside the root.
    ///
    /// Steps 2 and 4 exclude each other (a root `stream_options` is what makes step 2 apply and step
    /// 4 not), and every offset moves with each step, so the order is the behaviour's, not the
    /// offsets': the model is re-spelled before the injection because that is what was always sent.
    pub fn into_upstream(mut self, stream_usage: bool, model: Option<&[u8]>) -> Vec<u8> {
        if stream_usage
            && self.scan.stream_options_ambiguous
            && peek::remove_root_members(&mut self.buf, "stream_options")
        {
            self = Self::new(self.buf);
        }
        debug_assert!(
            self.scan.inject_at.is_none() || self.scan.stream_options_at.is_none(),
            "a body either sent stream_options or gets one, never both"
        );
        if stream_usage
            && let Some(at) = self.scan.stream_options_at
            && let Some(splice) = force_include_usage(&mut self.buf, at)
        {
            self.moved(splice);
        }
        if let Some(id) = model
            && let Some(span) = self.scan.model_span
            && let Some(splice) = overwrite(&mut self.buf, span, id)
        {
            self.moved(splice);
        }
        if stream_usage && let Some(at) = self.scan.inject_at {
            inject_stream_options(&mut self.buf, at);
        }
        self.buf
    }

    /// Move every offset in the scan by `splice`, the edit just made to `buf`.
    fn moved(&mut self, splice: Splice) {
        let s = &mut self.scan;
        s.inject_at = s.inject_at.and_then(|p| splice.point(p));
        s.stream_options_at = s.stream_options_at.and_then(|p| splice.point(p));
        s.model_span = s.model_span.and_then(|r| splice.span(r));
        for r in &mut s.limit_spans {
            *r = r.and_then(|r| splice.span(r));
        }
        for p in &mut s.limit_keys {
            *p = p.and_then(|p| splice.point(p));
        }
        debug_assert!(self.offsets_in_bounds(), "an offset left the body");
    }

    fn offsets_in_bounds(&self) -> bool {
        let s = &self.scan;
        let len = self.buf.len();
        let points = [s.inject_at, s.stream_options_at]
            .into_iter()
            .chain(s.limit_keys);
        let spans = s.limit_spans.into_iter().chain([s.model_span]);
        points.flatten().all(|p| p <= len) && spans.flatten().all(|(a, b)| a <= b && b <= len)
    }
}

/// A [`ScannedBody`]'s bytes, open to an edit the scan cannot follow. The scan is sealed until
/// [`Self::finish`]: there is no way to read an offset while the bytes may be changing.
pub(crate) struct Edit {
    body: ScannedBody,
}

impl Edit {
    /// Replace the bytes by a function of them, for the edits that take the body by value.
    pub(crate) fn map(&mut self, f: impl FnOnce(Vec<u8>) -> Vec<u8>) {
        self.body.buf = f(std::mem::take(&mut self.body.buf));
    }

    /// Close the edit: re-scanned when `changed`, else the scan as it was (the bytes are the
    /// ones it indexes).
    ///
    /// `changed` is the edits' own report, and a false `false` would keep offsets into bytes that
    /// moved. Debug builds re-scan to check it: a held scan that differs from a fresh one is an
    /// edit that changed what the scan reads and said it did not, same length or not. A taken
    /// `model` is not compared (`take_model` emptied it on purpose).
    pub(crate) fn finish(self, changed: bool) -> ScannedBody {
        if changed {
            return ScannedBody::new(self.body.buf);
        }
        if cfg!(debug_assertions) {
            let mut fresh = peek::scan_buffered(&self.body.buf);
            if self.body.scan.model.is_none() {
                fresh.model = None;
            }
            assert_eq!(
                self.body.scan, fresh,
                "an edit changed the body and reported no change"
            );
        }
        self.body
    }
}

impl std::ops::Deref for Edit {
    type Target = Vec<u8>;
    fn deref(&self) -> &Vec<u8> {
        &self.body.buf
    }
}

impl std::ops::DerefMut for Edit {
    fn deref_mut(&mut self) -> &mut Vec<u8> {
        &mut self.body.buf
    }
}

/// One edit's effect on offsets: `removed` bytes at `at` became `added` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Splice {
    at: usize,
    removed: usize,
    added: usize,
}

impl Splice {
    /// Where the byte at `p` is now. A byte before the edit stays; one past it moves by the length
    /// change; the start of a replaced run is the replacement's start; any other replaced byte is
    /// gone. An insertion (`removed == 0`) moves the byte at `at` too: it lands after the new bytes.
    fn point(self, p: usize) -> Option<usize> {
        let end = self.at.checked_add(self.removed)?;
        if p < self.at || (p == self.at && self.removed > 0) {
            Some(p)
        } else if p < end {
            None
        } else {
            p.checked_sub(self.removed)?.checked_add(self.added)
        }
    }

    /// Where the range `(a, b)` is now. The edited range itself is the replacement; a range before
    /// the edit stays, one past it moves.
    ///
    /// A range the edit cut into means the scan and the edit read the bytes differently: the only
    /// edit that can meet one is `force_include_usage`, whose `serde_json` parse ends the
    /// `stream_options` value, against the scan's `model` span. Debug builds stop there. A release
    /// build moves it as the hand-adjusted code did (shifted when it starts past the edit's start,
    /// else kept) so the body sent is the one it always was; a shift below zero is `None`, as the
    /// old shift's wrapped span missed the buffer and left the model unrewritten.
    fn span(self, (a, b): (usize, usize)) -> Option<(usize, usize)> {
        let end = self.at.checked_add(self.removed)?;
        let shift = |i: usize| i.checked_add(self.added)?.checked_sub(self.removed);
        if (a, b) == (self.at, end) {
            Some((self.at, self.at.checked_add(self.added)?))
        } else if b <= self.at {
            Some((a, b))
        } else if a >= end {
            Some((shift(a)?, shift(b)?))
        } else {
            debug_assert!(false, "{self:?} cut into {a}..{b}");
            match a > self.at {
                true => Some((shift(a)?, shift(b)?)),
                false => Some((a, b)),
            }
        }
    }
}

/// Splice `with` over `buf[range]`, as the [`Splice`] it was. `None` (and `buf` untouched) for a
/// range outside `buf`.
fn splice(buf: &mut Vec<u8>, (start, end): (usize, usize), with: &[u8]) -> Option<Splice> {
    if start > end || end > buf.len() {
        return None;
    }
    buf.splice(start..end, with.iter().copied());
    Some(Splice {
        at: start,
        removed: end.wrapping_sub(start),
        added: with.len(),
    })
}

/// Cut one limit's digits (`span`) to `max` when over it.
fn clamp_limit(buf: &mut Vec<u8>, span: (usize, usize), max: u32) -> Option<Splice> {
    let digits = buf.get(span.0..span.1)?;
    // Digits only (the scan's own span), so a parse fails only past `u64`: over any cap.
    let over = std::str::from_utf8(digits)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .is_none_or(|n| n > u64::from(max));
    if !over {
        return None;
    }
    splice(buf, span, max.to_string().as_bytes())
}

/// Overwrite the `model` value's bytes (`span`, quotes excluded) with `id`, the serving candidate's.
///
/// `id` is a catalog string, so it needs no JSON escaping — the catalog charset test
/// (`model_names_are_lowercase_and_log_safe`, plus the same shape for `upstream_model`) is what
/// makes a raw byte copy safe here. `None` when the id already matches, which is the common case:
/// candidate 0 usually spells the model the way the catalog names it, so the primary path does no
/// memmove at all and only a failover pays for one.
fn overwrite(buf: &mut Vec<u8>, span: (usize, usize), id: &[u8]) -> Option<Splice> {
    // A span outside the buffer is unreachable (the span is produced by the walk over these bytes),
    // but this runs on every model-routed request, so it is refused, not a panic.
    if buf.get(span.0..span.1)? == id {
        return None;
    }
    splice(buf, span, id)
}

/// Make a client-sent `stream_options` (value at `at`) ask for usage: `include_usage: false` becomes
/// `true`, an object without it gains it, and a non-object value is replaced by
/// `{"include_usage":true}`. Billing is not the client's to switch off — without the usage chunk the
/// row is an estimate that cannot see hidden reasoning. The client may now receive one extra chunk it
/// did not ask for: OpenAI's usage chunk, `choices: []`, which every SDK already accepts.
///
/// Parses only the `stream_options` value (a few bytes), and only on the rare request that sends
/// one; the value is re-serialized, so its other members keep their meaning but not their spacing.
fn force_include_usage(buf: &mut Vec<u8>, at: usize) -> Option<Splice> {
    use serde_json::Value;
    let mut values = serde_json::Deserializer::from_slice(buf.get(at..)?).into_iter::<Value>();
    let Some(Ok(value)) = values.next() else {
        return None;
    };
    let end = at.checked_add(values.byte_offset())?;
    let mut options = match value {
        Value::Object(m) if m.get("include_usage") == Some(&Value::Bool(true)) => return None,
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    options.insert("include_usage".to_owned(), Value::Bool(true));
    let replacement = serde_json::to_vec(&options).ok()?;
    splice(buf, (at, end), &replacement)
}

/// Splice [`STREAM_OPTIONS_FRAG`] in at `at`. This is what guarantees a usage chunk — hence a
/// billable token count — from a stock client that never set the option.
///
/// Shifts the tail right in place rather than copying the whole body into a second buffer.
/// `req_buf` is pre-sized with `STREAM_OPTIONS_FRAG.len()` of headroom (see `request_filter`), so
/// when the client declared a Content-Length this `resize` is free and the only work is moving the
/// `len - at` bytes after the splice point. A copy into a second buffer measured 579 µs against
/// 27.8 µs at 1 MB, because the second allocation dominates. Without a declared length (chunked
/// upload) the `resize` may grow once: still a single allocation.
fn inject_stream_options(buf: &mut Vec<u8>, at: usize) -> Option<Splice> {
    let old = buf.len();
    let to = at.checked_add(STREAM_OPTIONS_FRAG.len())?;
    if at > old {
        return None;
    }
    buf.resize(old.checked_add(STREAM_OPTIONS_FRAG.len())?, 0);
    buf.copy_within(at..old, to);
    buf.get_mut(at..to)?.copy_from_slice(STREAM_OPTIONS_FRAG);
    Some(Splice {
        at,
        removed: 0,
        added: STREAM_OPTIONS_FRAG.len(),
    })
}

/// Respell a root `max_tokens` as `max_completion_tokens`, in place. `keys` is
/// [`BufferedScan::limit_keys`]. When both are present the explicit `max_completion_tokens` wins and
/// the `max_tokens` member is removed. `true` when the body changed.
fn rename_max_tokens(body: &mut Vec<u8>, keys: &[Option<usize>; 3]) -> bool {
    const KEY: &[u8] = br#""max_tokens""#;
    let [Some(at), second, _] = *keys else {
        return false;
    };
    let Some(key_end) = at.checked_add(KEY.len()) else {
        return false;
    };
    if body.get(at..key_end) != Some(KEY) {
        return false;
    }
    if second.is_none() {
        // Inside the quotes: `at + 1 ..= key_end - 1` exist, the key matched above.
        let inner = (at.saturating_add(1), key_end.saturating_sub(1));
        return splice(body, inner, b"max_completion_tokens").is_some();
    }
    // Both spellings. Find the end of the `max_tokens` value without building it, then remove
    // the member and one adjacent comma.
    let skip_ws = |i: usize| {
        let n = body
            .get(i..)
            .unwrap_or_default()
            .iter()
            .take_while(|b| b.is_ascii_whitespace());
        i.saturating_add(n.count())
    };
    let colon = skip_ws(key_end);
    if body.get(colon) != Some(&b':') {
        return false;
    }
    let value = skip_ws(colon.saturating_add(1));
    let Some(rest) = body.get(value..) else {
        return false;
    };
    let mut values =
        serde_json::Deserializer::from_slice(rest).into_iter::<serde::de::IgnoredAny>();
    if !matches!(values.next(), Some(Ok(_))) {
        return false;
    }
    let end = value.saturating_add(values.byte_offset());
    let after = skip_ws(end);
    let range = if body.get(after) == Some(&b',') {
        (at, after.saturating_add(1))
    } else {
        // The last member: take the comma before it instead.
        let before = body.get(..at).unwrap_or_default();
        match before.iter().rposition(|b| !b.is_ascii_whitespace()) {
            Some(comma) if before.get(comma) == Some(&b',') => (comma, end),
            _ => (at, end),
        }
    };
    splice(body, range, b"").is_some()
}

/// The edits on raw offsets, for the tests that pin their bytes one edit at a time. Test-only, so
/// nothing that ships can hold an offset outside [`ScannedBody`].
#[cfg(test)]
pub(crate) mod raw {
    use super::*;

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
        for span in spans.into_iter().flatten() {
            changed |= clamp_limit(body, span, max).is_some();
        }
        changed
    }

    pub fn rename_max_tokens(body: &mut Vec<u8>, keys: &[Option<usize>; 3]) -> bool {
        super::rename_max_tokens(body, keys)
    }

    pub fn apply_model_rewrite(mut body: Vec<u8>, span: (usize, usize), id: &[u8]) -> Vec<u8> {
        overwrite(&mut body, span, id);
        body
    }

    /// The edit, and `model_span` moved by it.
    pub fn force_include_usage(
        mut body: Vec<u8>,
        at: usize,
        model_span: Option<(usize, usize)>,
    ) -> (Vec<u8>, Option<(usize, usize)>) {
        let span = match super::force_include_usage(&mut body, at) {
            Some(splice) => model_span.and_then(|r| splice.span(r)),
            None => model_span,
        };
        (body, span)
    }

    pub fn apply_stream_usage_injection(mut body: Vec<u8>, at: Option<usize>) -> Vec<u8> {
        if let Some(at) = at {
            inject_stream_options(&mut body, at);
        }
        body
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    /// Every offset after each edit is where a fresh scan of the edited bytes puts it: moving them
    /// is a re-scan without the walk.
    #[test]
    fn moved_offsets_match_a_rescan() {
        let offsets = |s: &BufferedScan| {
            (
                s.inject_at,
                s.model_span,
                s.limit_spans,
                s.limit_keys,
                s.stream_options_at,
            )
        };
        for (body, max, model) in [
            (
                r#"{"max_tokens":64000,"model":"gpt-4o","max_completion_tokens":99999999999999999999999,"stream":true}"#,
                16384,
                "openai/gpt-4o",
            ),
            (
                r#"{"model":"a","stream":true,"stream_options":{"include_usage":false},"max_output_tokens":900000}"#,
                100,
                "much-longer-id",
            ),
            (
                r#"{"stream_options":null, "max_tokens" : 5 ,"stream":true,"model":"long-model-name"}"#,
                1,
                "x",
            ),
            (r#"{"model":"","max_tokens":20}"#, 10, "gpt-4o"),
            // An insertion (an empty model span) at a member boundary, with an injection point
            // ahead of it and a limit key right after it.
            (
                r#"{"stream":true,"model":"","max_tokens":70000}"#,
                5,
                "gpt-4o",
            ),
            (r#"{"model":"m","stream":true}"#, 0, "openai/m"),
        ] {
            let mut b = ScannedBody::new(body.as_bytes().to_vec()).clamp_output_limits(max);
            assert_eq!(
                offsets(&b.scan),
                offsets(&peek::scan_buffered(&b.buf)),
                "clamp {body}"
            );
            if let Some(at) = b.scan.stream_options_at
                && let Some(splice) = force_include_usage(&mut b.buf, at)
            {
                b.moved(splice);
                assert_eq!(
                    offsets(&b.scan),
                    offsets(&peek::scan_buffered(&b.buf)),
                    "force {body}"
                );
            }
            let span = b.scan.model_span.expect("model");
            let splice = overwrite(&mut b.buf, span, model.as_bytes()).expect("a new id");
            b.moved(splice);
            assert_eq!(
                offsets(&b.scan),
                offsets(&peek::scan_buffered(&b.buf)),
                "model {body}"
            );
        }
    }

    /// An opaque edit that moves bytes and says so is re-scanned: the held scan is a fresh one,
    /// for a splice through `Edit::map` and for a translation onto another wire.
    #[test]
    fn a_reported_opaque_edit_is_rescanned() {
        use crate::route::Endpoint;
        let body = br#"{"model":"gpt-4o","max_tokens":64000,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let fresh = |b: &ScannedBody| peek::scan_buffered(&b.buf);

        let mut e = ScannedBody::new(body.to_vec())
            .clamp_output_limits(100)
            .edit();
        e.map(|b| [&b"{\"user\":\"u\","[..], &b[1..]].concat());
        let b = e.finish(true);
        assert_eq!(b.scan, fresh(&b));
        assert_eq!(b.scan.model_span.map(|(s, _)| s), Some(21));

        let mut e = ScannedBody::new(body.to_vec()).edit();
        let (translated, ..) = crate::translate::request_with_tools(
            Endpoint::ChatCompletions,
            Endpoint::Messages,
            &e,
            "claude-sonnet-4-5",
            false,
        );
        *e = translated;
        let b = e.finish(true);
        assert_eq!(b.scan, fresh(&b));
        assert_ne!(b.buf, body, "the translation changed the bytes");
    }

    /// An edit that changes what the scan reads and reports no change is caught in a debug build,
    /// even at the same length.
    #[test]
    #[should_panic(expected = "reported no change")]
    fn an_unreported_same_length_edit_is_caught() {
        let mut e = ScannedBody::new(br#"{"model":"gpt-4o"}"#.to_vec()).edit();
        e[10] = b'G';
        let _ = e.finish(false);
    }

    /// An unchanged body keeps its scan, a taken `model` included.
    #[test]
    fn an_unchanged_edit_keeps_the_scan() {
        let mut b = ScannedBody::new(br#"{"model":"gpt-4o","stream":true}"#.to_vec());
        assert_eq!(b.take_model().as_deref(), Some("gpt-4o"));
        let b = b.edit().finish(false);
        assert_eq!(b.model(), None);
        assert_eq!(b.scan.inject_at, Some(1));
    }

    /// An insertion moves the byte at its point; a replacement keeps its start; a byte it replaced
    /// is gone. A span it cut into is a bug (`span`'s debug assert).
    #[test]
    fn splice_moves_points_and_spans() {
        let ins = Splice {
            at: 4,
            removed: 0,
            added: 3,
        };
        assert_eq!(
            (ins.point(3), ins.point(4), ins.point(9)),
            (Some(3), Some(7), Some(12))
        );
        assert_eq!(
            ins.span((4, 4)),
            Some((4, 7)),
            "the edited empty span is the insertion"
        );
        assert_eq!(ins.span((1, 4)), Some((1, 4)));
        let rep = Splice {
            at: 4,
            removed: 2,
            added: 5,
        };
        assert_eq!(
            (rep.point(4), rep.point(5), rep.point(6)),
            (Some(4), None, Some(9))
        );
        assert_eq!(rep.span((4, 6)), Some((4, 9)));
        assert_eq!(rep.span((6, 8)), Some((9, 11)));
    }
}
