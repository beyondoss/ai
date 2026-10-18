//! Streaming, 100%-accurate extraction of the **root-level `model`** from a JSON request body.
//!
//! Both OpenAI and Anthropic require `model` as a top-level field of the request object. We extract
//! it with a structural state machine fed the body chunks *as they stream through* — the body is
//! never buffered or reordered. This is exact (not a byte-heuristic): it tracks nesting depth and
//! string/escape state, so a `"model"` appearing inside a nested object (e.g. a message) or inside
//! a string value is correctly ignored, and field order is irrelevant. Memory is O(1): only
//! root-level *keys* and the `model` value are accumulated, each capped at [`MAX_CAPTURE`] bytes. Large uninteresting string content
//! (system prompts, base64 images) is skipped with a SIMD-accelerated `memchr2` search to the next
//! `"`/`\`, not inspected byte-by-byte — so even a multi-MB request is walked cheaply.
//!
//! Every function here indexes client bytes, so a slip is a panic on a hostile body: indexing and
//! unchecked arithmetic are denied in this module, and each place that keeps one (a hot loop whose
//! bound is its own condition) says why it cannot fail.
#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::lex::{self, Flow, Lexer, Visit, string_end};

/// Most bytes of a root-level key or `model` value we will hold. Real model ids are well under
/// 128 bytes (`proxy::sanitize_model` records anything longer as `unknown`), and the only keys we
/// compare against are `model` and `message`, so anything past this is junk — and without a bound a
/// managed request carrying a multi-megabyte `model` string would be materialized in full just to be
/// discarded. A capped value is a prefix: longer than any catalog name or loggable id, so it can only
/// ever miss the catalog or log as `unknown`, never match something it isn't.
pub const MAX_CAPTURE: usize = 256;

/// Append to a capture, up to [`MAX_CAPTURE`] bytes.
#[inline]
fn push_capped(buf: &mut Vec<u8>, bytes: &[u8]) {
    let room = MAX_CAPTURE.saturating_sub(buf.len());
    buf.extend_from_slice(bytes.get(..room.min(bytes.len())).unwrap_or_default());
}

/// The text of a captured JSON string, with its escapes as sent: decoded the way a provider's parser
/// decodes them, so `"gpt-5"` is `gpt-5`. `None` for bytes that are not text. An escaped capture
/// at the cap is `None` too: its last escape may be cut, and the decoded prefix can then read as a
/// shorter, real name. `escaped` is whether the scan saw a backslash in the capture, so no rescan.
fn capture_text(raw: Vec<u8>, escaped: bool) -> Option<String> {
    if !escaped {
        return String::from_utf8(raw).ok();
    }
    if raw.len() >= MAX_CAPTURE {
        return None;
    }
    lex::unescape(&raw)
}

/// A captured key's text as [`capture_text`] decodes it. Unescaped keys are compared as captured, so
/// the common key costs no more than it did before escapes were decoded.
fn key_bytes(raw: &[u8], escaped: bool) -> Option<std::borrow::Cow<'_, [u8]>> {
    if escaped {
        capture_text(raw.to_vec(), true).map(|s| std::borrow::Cow::Owned(s.into_bytes()))
    } else {
        Some(std::borrow::Cow::Borrowed(raw))
    }
}

#[derive(Clone, Copy, PartialEq, Default)]
enum Cap {
    #[default]
    No,
    Key,
    ModelValue,
    /// The string value of the key [`Kept::last_key_extra`] names.
    Extra,
}

/// The key-level members a response scanner also keeps when they come before `model`: the
/// provider's id for this generation (every wire puts `id` first: Anthropic `message.id`, Chat
/// Completions' root `id`, Responses `response.id`, OpenRouter's `gen-…`) and OpenRouter's
/// serving host (`provider`, which it writes between `id` and `model`).
pub const RESPONSE_EXTRA_KEYS: &[&[u8]] = &[b"id", b"provider"];

/// The root request knobs a billing row records as *requested* (the served value comes from the
/// response): OpenAI/OpenRouter/xAI `service_tier`, Anthropic/OpenRouter `speed`, Anthropic
/// `inference_geo`.
///
/// Also the knobs whose value is an object or array, kept as raw JSON (at most [`RAW_CAPTURE`]
/// bytes): OpenRouter's `provider` routing preferences and `plugins` (web search, file parsing),
/// and Anthropic's `container` (a code-execution container to reuse; an id string or an object).
pub const REQUEST_KNOB_KEYS: &[&[u8]] = &[
    b"service_tier",
    b"speed",
    b"inference_geo",
    b"provider",
    b"plugins",
    b"container",
];

/// Most bytes of an object or array knob value kept. Past it the value is cut (and then not valid
/// JSON), at a UTF-8 boundary.
pub const RAW_CAPTURE: usize = 512;

/// How many extra keys a scan can keep.
const MAX_EXTRAS: usize = 6;

#[derive(Default)]
pub struct ModelScanner {
    model: Option<String>,
    done: bool,
    /// Strings, escapes and nesting depth (root object contents are at depth 1), across chunks.
    lex: Lexer,
    root_is_object: bool,
    /// Whether the next key-level string is a key (`{`/`,` → key; `:` → value).
    expect_key: bool,
    /// The most recent key-level key was exactly `model`.
    last_key_is_model: bool,
    /// Also accept `model` nested one level under a root `message` key — see
    /// [`ModelScanner::for_response`]. Off for request bodies, whose `model` is always root-level.
    accept_message_nesting: bool,
    /// The most recent **root** key was exactly `message` (only tracked when
    /// `accept_message_nesting`).
    last_key_is_message: bool,
    /// We are inside the object that root-level `message` maps to, so its keys count as key-level.
    in_message: bool,
    /// What (if anything) we're accumulating into `cur` for the current string.
    cap: Cap,
    cur: Vec<u8>,
}

/// What a scan keeps beside `model` ([`ModelScanner::feed_keeping`]): the values of a fixed set of
/// key-level members. Held outside the scanner, by the one caller that wants them, so the scanner
/// inline on every request stays the size it was (`RequestCtx`'s per-chunk size budget).
pub struct Kept {
    keys: &'static [&'static [u8]],
    /// Walk past `model` until every key is found (the request-knob scan, whose keys may follow
    /// `model`). A response scan stops at `model`: what precedes it is all it keeps.
    past_model: bool,
    /// The first value seen for each key, by index.
    values: [Option<String>; MAX_EXTRAS],
    /// The most recent key-level key was `keys[i]`.
    last_key_extra: Option<u8>,
    /// An object or array value of `keys[i]` is being captured raw.
    raw: Option<u8>,
    /// The depth outside that value: the capture ends when the depth returns to it.
    raw_depth: u32,
    raw_buf: Vec<u8>,
}

impl Kept {
    fn with(keys: &'static [&'static [u8]], past_model: bool) -> Self {
        Kept {
            keys,
            past_model,
            values: Default::default(),
            last_key_extra: None,
            raw: None,
            raw_depth: 0,
            raw_buf: Vec::new(),
        }
    }

    /// [`RESPONSE_EXTRA_KEYS`], for a [`ModelScanner::for_response`] scan.
    pub fn response() -> Self {
        Self::with(RESPONSE_EXTRA_KEYS, false)
    }

    /// [`REQUEST_KNOB_KEYS`], over a whole request body: fed to a [`ModelScanner::new`] scan,
    /// which then walks the whole body (see `proxy::requested_knobs`).
    pub fn request_knobs() -> Self {
        Self::with(REQUEST_KNOB_KEYS, true)
    }

    /// The value captured for `keys[i]`, if one was seen.
    pub fn get(&self, i: usize) -> Option<&str> {
        self.values.get(i).and_then(|v| v.as_deref())
    }

    /// Take the value captured for `keys[i]`.
    pub fn take(&mut self, i: usize) -> Option<String> {
        self.values.get_mut(i).and_then(Option::take)
    }

    /// Keep `v` as the value of the key [`Self::last_key_extra`] names, or of the raw capture,
    /// unless one was kept already.
    fn keep(&mut self, k: u8, v: String) {
        if let Some(slot) = self.values.get_mut(usize::from(k))
            && slot.is_none()
        {
            *slot = Some(v);
        }
    }

    fn all_found(&self) -> bool {
        self.values
            .iter()
            .take(self.keys.len())
            .all(Option::is_some)
    }
}

/// Append to a raw capture, up to [`RAW_CAPTURE`] bytes.
fn push_raw(buf: &mut Vec<u8>, bytes: &[u8]) {
    let room = RAW_CAPTURE.saturating_sub(buf.len());
    buf.extend_from_slice(bytes.get(..room.min(bytes.len())).unwrap_or_default());
}

impl ModelScanner {
    /// Strict: only a **root-level** `model`. Correct for request bodies, where both OpenAI and
    /// Anthropic require `model` as a top-level field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Also accepts `model` nested one level under a root-level `message` key.
    ///
    /// That is where Anthropic puts it on the streaming wire: `message_start` is
    /// `{"type":"message_start","message":{…,"model":"claude-…",…}}`, so the model sits at depth 2.
    /// A root-only scanner never matches it, which means it also never reaches its `done`
    /// short-circuit — so it byte-walked the *entire* stream to return `None`, at ~1.4 GB/s, on
    /// every Anthropic streaming response. Measured 1.77 ms of pure waste on a 2.5 MB stream against
    /// 1.7 µs for the OpenAI shape, which finds its root `model` in the first chunk and stops.
    ///
    /// Recovering the value is the point, though; the speedup is a consequence. The provider-echoed
    /// model is the id a response is actually *billed* under (a client may send an alias like
    /// `claude-opus-4-8` that resolves to a pinned snapshot), so without this every Anthropic stream
    /// fell back to the requested alias in the billing log — the exact reconciliation gap
    /// `resp_model_scanner` exists to close, silently open for one of the two dialects.
    ///
    /// A Responses stream has the same shape under a different key: `response.created` (and
    /// `response.completed`) carry the pinned snapshot as `response.model`, and no event has a root
    /// `model` at all. So a root `response` key opens the same one-level nested scan; without it a
    /// Responses stream billed under the requested alias.
    ///
    /// Kept opt-in rather than always-on so request-body scanning cannot start picking up a nested
    /// `model` that isn't the one the client asked for.
    pub fn for_response() -> Self {
        Self {
            accept_message_nesting: true,
            ..Self::default()
        }
    }

    /// Take the extracted model, if found. (Available as soon as the value is seen.)
    pub fn take_model(&mut self) -> Option<String> {
        self.model.take()
    }

    /// Whether the scan has already closed a root `model` value.
    ///
    /// Further [`Self::feed`] calls are no-ops once this is set. Callers that only need to
    /// know *whether* to stop reading — the catalog peek, which may still have to drain a
    /// small body — check this instead of re-walking bytes already fed.
    pub fn found(&self) -> bool {
        self.done
    }

    /// Whether `depth` is one whose keys we inspect: the root object always, plus the object
    /// under a root `message` key when [`Self::for_response`] enabled it.
    #[inline]
    fn at_key_level(&self, depth: u32) -> bool {
        (depth == 1 && self.root_is_object) || (depth == 2 && self.in_message)
    }

    /// Feed the next bytes of the body.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.feed_inner(bytes, None);
    }

    /// Feed the next bytes, keeping the values of `kept`'s keys too. Every chunk of one body goes
    /// to the same `kept`.
    pub fn feed_keeping(&mut self, bytes: &[u8], kept: &mut Kept) {
        self.feed_inner(bytes, Some(kept));
    }

    fn feed_inner(&mut self, bytes: &[u8], kept: Option<&mut Kept>) {
        if self.done {
            return;
        }
        // Where the raw capture in progress starts in this chunk.
        let raw_from = kept.as_ref().and_then(|k| k.raw).map(|_| 0);
        let mut lex = self.lex;
        let mut walk = Walk {
            s: self,
            kept,
            bytes,
            raw_from,
        };
        let stopped = lex.feed(bytes, &mut walk);
        let Walk {
            s, kept, raw_from, ..
        } = walk;
        s.lex = lex;
        // A stop is the scan finishing, which ends any raw capture first.
        if stopped.is_none()
            && let Some(k) = kept
            && k.raw.is_some()
            && let Some(from) = raw_from
        {
            push_raw(&mut k.raw_buf, bytes.get(from..).unwrap_or_default());
        }
    }
}

/// One [`ModelScanner::feed_inner`] call: the scanner's key-level grammar over the lexer's events.
struct Walk<'a, 'k> {
    s: &'a mut ModelScanner,
    kept: Option<&'k mut Kept>,
    bytes: &'a [u8],
    /// Where the raw capture in progress starts in this chunk.
    raw_from: Option<usize>,
}

impl Walk<'_, '_> {
    /// The scan is over: `feed` is a no-op from here.
    fn finish(&mut self) -> Flow {
        self.s.done = true;
        Flow::Break(())
    }
}

impl Visit for Walk<'_, '_> {
    #[inline(always)]
    fn open(&mut self, _: usize, depth: u32) {
        let extra = self.kept.as_ref().and_then(|k| k.last_key_extra);
        let s = &mut *self.s;
        s.cur.clear();
        // Decide whether this string is worth accumulating — only key-level keys and the `model`
        // value matter.
        s.cap = if s.at_key_level(depth) {
            if s.expect_key {
                Cap::Key
            } else if s.last_key_is_model {
                Cap::ModelValue
            } else if extra.is_some() {
                Cap::Extra
            } else {
                Cap::No
            }
        } else {
            Cap::No
        };
    }

    #[inline(always)]
    fn content(&mut self, raw: &[u8]) {
        let s = &mut *self.s;
        if s.cap != Cap::No {
            push_capped(&mut s.cur, raw);
        }
    }

    fn close(&mut self, _: usize, depth: u32, escaped: bool) -> Flow {
        let s = &mut *self.s;
        match s.cap {
            Cap::Key => {
                let key = key_bytes(&s.cur, escaped);
                s.last_key_is_model = key.as_deref() == Some(b"model".as_slice());
                if let Some(k) = self.kept.as_deref_mut() {
                    let at = key
                        .as_deref()
                        .and_then(|key| k.keys.iter().position(|want| *want == key));
                    k.last_key_extra = at.and_then(|i| u8::try_from(i).ok());
                }
                // Only a *root* `message` opens the nested scan — a `message` key inside the
                // message object itself must not re-arm it.
                if s.accept_message_nesting && depth == 1 {
                    s.last_key_is_message =
                        matches!(key.as_deref(), Some(b"message" | b"response"));
                }
            }
            Cap::ModelValue => {
                // A valid JSON string value is UTF-8; if a malformed/adversarial body smuggles
                // non-UTF-8 bytes here we record "unknown" rather than emitting a
                // `U+FFFD`-corrupted model into the billing log. Either way we're done.
                let v = capture_text(std::mem::take(&mut s.cur), escaped)
                    .unwrap_or_else(|| "unknown".to_string());
                match self.kept.as_deref() {
                    // A knob scan keeps walking; the first `model` is the one kept.
                    Some(k) if k.past_model => {
                        s.model.get_or_insert(v);
                        if k.all_found() {
                            return self.finish();
                        }
                    }
                    _ => {
                        s.model = Some(v);
                        return self.finish();
                    }
                }
            }
            Cap::Extra => {
                if let Some(k) = self.kept.as_deref_mut()
                    && let Some(idx) = k.last_key_extra
                {
                    // Not UTF-8: dropped, never written to a row as a corrupted value.
                    if let Some(v) = capture_text(std::mem::take(&mut s.cur), escaped) {
                        k.keep(idx, v);
                    }
                    if k.past_model && k.all_found() {
                        return self.finish();
                    }
                }
            }
            Cap::No => {}
        }
        let s = &mut *self.s;
        s.cap = Cap::No;
        s.cur.clear();
        Flow::Continue(())
    }

    #[inline(always)]
    fn byte(&mut self, at: usize, b: u8, depth: u32) -> Flow {
        let extra = self.kept.as_ref().and_then(|k| k.last_key_extra);
        let s = &mut *self.s;
        match b {
            b'{' | b'['
                if extra.is_some()
                    && !s.expect_key
                    && s.at_key_level(depth)
                    && self.kept.as_ref().is_some_and(|k| k.raw.is_none()) =>
            {
                // An object or array value of a kept key: captured raw until its depth closes.
                if let Some(k) = self.kept.as_deref_mut() {
                    k.raw = extra;
                    k.raw_depth = depth;
                    k.raw_buf.clear();
                }
                self.raw_from = Some(at);
            }
            b'{' => {
                if depth == 0 {
                    s.root_is_object = true;
                    s.expect_key = true;
                } else if depth == 1 && s.last_key_is_message {
                    // Descending into the root `message` object: its keys become key-level too.
                    s.in_message = true;
                    s.expect_key = true;
                }
            }
            b'[' => {
                if depth == 0 {
                    s.root_is_object = false;
                }
            }
            b'}' | b']' => {
                // Leaving the `message` object returns key-level scanning to the root.
                if depth == 2 && s.in_message {
                    s.in_message = false;
                    s.last_key_is_message = false;
                }
                if let Some(k) = self.kept.as_deref_mut()
                    && let Some(idx) = k.raw
                    && depth.saturating_sub(1) == k.raw_depth
                {
                    let from = self.raw_from.take().unwrap_or(0);
                    let to = at.saturating_add(1);
                    push_raw(&mut k.raw_buf, self.bytes.get(from..to).unwrap_or_default());
                    k.raw = None;
                    let v = utf8_prefix(std::mem::take(&mut k.raw_buf));
                    k.keep(idx, v);
                    if k.past_model && k.all_found() {
                        return self.finish();
                    }
                }
            }
            // Unguarded by depth: a `:` or `,` inside a nested value is always followed by the
            // key-level `,` or `}` that closes it before the next key-level string, and that one
            // sets `expect_key` again.
            b':' => s.expect_key = false,
            b',' => {
                s.expect_key = true;
                s.last_key_is_model = false;
                if let Some(k) = self.kept.as_deref_mut() {
                    k.last_key_extra = None;
                }
                if depth == 1 {
                    s.last_key_is_message = false;
                }
            }
            _ => {}
        }
        Flow::Continue(())
    }
}

/// The whole UTF-8 characters at the front of `buf`: a capture cut inside a character keeps the
/// ones before it, never a corrupted value.
fn utf8_prefix(buf: Vec<u8>) -> String {
    match String::from_utf8(buf) {
        Ok(v) => v,
        Err(e) => {
            let up_to = e.utf8_error().valid_up_to();
            let mut b = e.into_bytes();
            b.truncate(up_to);
            String::from_utf8(b).unwrap_or_default()
        }
    }
}

/// Both answers the injection path needs from a **fully buffered** request body.
///
/// Its offsets index exactly the bytes scanned and go stale at the first edit, so what edits a body
/// holds them inside `rewrite::ScannedBody`, which moves them with each edit.
pub struct BufferedScan {
    /// The root-level `model`, as [`ModelScanner`] would have extracted it.
    pub model: Option<String>,
    /// Where to splice `"stream_options":{"include_usage":true},`: just past the root object's
    /// opening `{`, **only** when the root has `"stream": true` (the last root `stream` decides, as
    /// the provider keeps the last duplicate) and **no** root `stream_options`. OpenAI emits a
    /// usage chunk on a stream only when the request asks for it, and a stock client that omits it
    /// could not be metered, so the gateway asks on its behalf. The offset is always inside a
    /// non-empty object, so the fragment is always followed by a comma.
    pub inject_at: Option<usize>,
    /// Where the value of a root `stream_options` begins, when the body streams **and** already
    /// carries one. The client may have turned usage off (`include_usage: false`) or left it out
    /// (`{}`); either way OpenAI would send no usage chunk, so the caller rewrites that value (see
    /// `rewrite::ScannedBody::into_upstream`). `None` whenever `inject_at` is `Some`, and for a non-stream.
    pub stream_options_at: Option<usize>,
    /// Byte range of the root-level `model` **value**, exclusive of its quotes — the span to
    /// overwrite when a model-routed request has to be re-spelled for the candidate serving it.
    ///
    /// Raw, still-escaped bytes, unlike [`Self::model`] which is unescaped: a rewrite replaces the
    /// span wholesale, so it wants the extent of what is actually on the wire, not the decoded
    /// value's length. `None` whenever [`Self::model`] is `None`, and the two always describe the
    /// same occurrence — the first root-level `model`.
    pub model_span: Option<(usize, usize)>,
    /// The root object has more than one `model` key. JSON parsers disagree on which one wins
    /// (most take the last), so a catalog walk that routed on one and rewrote one cannot know which
    /// the provider would serve. A key spelled with escapes (`"mod\u0065l"`) counts: it decodes to
    /// `model` at the provider.
    pub duplicate_model: bool,
    /// Byte range of each root-level output limit's digits, indexed like [`OUTPUT_LIMIT_KEYS`]
    /// (first occurrence of each): the span a catalog walk caps at the serving model's maximum.
    /// A value that is not a plain non-negative integer is not recorded.
    pub limit_spans: [Option<(usize, usize)>; 3],
    /// A root output limit is present with a value that is no plain integer and not `null`: a
    /// string, a fraction, an exponent, a negative, an object. It has no span, so a clamp cannot
    /// cap it, and a catalog walk refuses the body rather than send it on.
    pub inexact_limit: bool,
    /// Offset of each root-level output limit key's opening quote, indexed like
    /// [`OUTPUT_LIMIT_KEYS`] (first occurrence of each, whatever its value): where a walk onto
    /// native OpenAI Chat Completions renames `max_tokens`.
    pub limit_keys: [Option<usize>; 3],
    /// The root carries more than one `stream_options`, or one spelled with escapes. OpenAI takes
    /// the last and decodes escapes, so neither [`Self::stream_options_at`] (the first raw one) nor
    /// [`Self::inject_at`] (which a later escaped key would override) can be trusted to decide
    /// whether usage is on. The caller removes every such member and injects a fresh one.
    pub stream_options_ambiguous: bool,
}

/// The fields that cap a response's length, whichever wire the body speaks.
pub const OUTPUT_LIMIT_KEYS: [&[u8]; 3] = [
    b"max_tokens",
    b"max_completion_tokens",
    b"max_output_tokens",
];

/// One structural walk producing both answers, for the path that already has the whole body.
///
/// A managed OpenAI chat request is buffered anyway (the injection point is near the front, so the
/// decision needs the whole body), and one walk answers both what `ModelScanner` would find (the
/// root `model`) and where `stream_options` goes: same bytes, different needles.
///
/// Only for the buffered path. A request that streams through un-buffered still uses
/// `ModelScanner`, which is incremental and cannot be replaced by this.
///
/// Semantics are exactly the two walks it fused, including the details that look incidental:
/// `stream_options` anywhere at root wins regardless of `stream`, a key spelled with escapes is
/// decoded and the last `stream` decides (as the provider's parser does; a duplicate or escaped
/// `stream_options` is flagged in `stream_options_ambiguous` instead, D88), and the model value
/// *is* unescaped because `ModelScanner` unescapes it. `fused_scan_matches_the_two_walks_it_replaces` cross-checks a corpus against both.
pub fn scan_buffered(body: &[u8]) -> BufferedScan {
    // Not a JSON object ⇒ nothing to inject into. A non-object root also has no root-level `model`,
    // so both answers are `None` and there is nothing to walk for.
    let Some(open) = root_open(body) else {
        return BufferedScan {
            model: None,
            inject_at: None,
            stream_options_at: None,
            model_span: None,
            duplicate_model: false,
            limit_spans: [None; 3],
            inexact_limit: false,
            limit_keys: [None; 3],
            stream_options_ambiguous: false,
        };
    };
    let mut walk = Fused {
        body,
        open: Open::None,
        expect_key: false,
        last_key_is_stream: false,
        last_key_is_model: false,
        stream_true: false,
        saw_stream_options: false,
        stream_options_ambiguous: false,
        stream_options_key_end: 0,
        model: None,
        model_span: None,
        model_keys: 0,
        limit_key: None,
        limit_spans: [None; 3],
        inexact_limit: false,
        limit_value_next: false,
        limit_keys: [None; 3],
    };
    // Only whitespace precedes `open`, which no rule below reads, so the whole body is lexed.
    Lexer::new().feed(body, &mut walk);

    // The `stream_options` value: past the key's closing quote, whitespace, `:`, whitespace.
    let stream_options_at = (walk.stream_true && walk.saw_stream_options)
        .then(|| {
            let colon = skip_ws(body, walk.stream_options_key_end.saturating_add(1));
            (body.get(colon) == Some(&b':')).then_some(())?;
            let value = skip_ws(body, colon.saturating_add(1));
            (value < body.len()).then_some(value)
        })
        .flatten();
    BufferedScan {
        model: walk.model,
        inject_at: (walk.stream_true && !walk.saw_stream_options).then_some(open.saturating_add(1)),
        stream_options_at,
        model_span: walk.model_span,
        duplicate_model: walk.model_keys > 1,
        limit_spans: walk.limit_spans,
        inexact_limit: walk.inexact_limit,
        limit_keys: walk.limit_keys,
        stream_options_ambiguous: walk.stream_options_ambiguous,
    }
}

/// The string [`scan_buffered`] is inside, and where its content starts (just past the quote).
#[derive(Clone, Copy)]
enum Open {
    /// One it reads nothing from.
    None,
    /// A root key, sliced out of the body at its closing quote.
    Key(usize),
    /// The value of the first root `model`.
    Model(usize),
}

/// [`scan_buffered`]'s walk: the root members' grammar over the lexer's events.
struct Fused<'a> {
    body: &'a [u8],
    open: Open,
    expect_key: bool,
    last_key_is_stream: bool,
    last_key_is_model: bool,
    stream_true: bool,
    saw_stream_options: bool,
    stream_options_ambiguous: bool,
    /// The closing quote of the root `stream_options` key; its value follows the next `:`.
    stream_options_key_end: usize,
    model: Option<String>,
    model_span: Option<(usize, usize)>,
    model_keys: u32,
    /// The output limit whose value comes next, as an index into `OUTPUT_LIMIT_KEYS`.
    limit_key: Option<usize>,
    limit_spans: [Option<(usize, usize)>; 3],
    inexact_limit: bool,
    /// The byte after a limit key's colon is still to come.
    limit_value_next: bool,
    limit_keys: [Option<usize>; 3],
}

/// Whether a key's raw bytes name `want`: as sent, or decoded when it held a backslash.
fn key_names(raw: &[u8], escaped: bool, want: &str) -> bool {
    if escaped {
        lex::unescapes_to(raw, want)
    } else {
        raw == want.as_bytes()
    }
}

impl Fused<'_> {
    /// The first byte of a limit's value decides it: a digit starts a plain integer, the literal
    /// `null` is no value, and whitespace or the colon before the value is skipped. Anything else
    /// (an `n` that is not `null`, a string, a fraction) is inexact.
    #[inline(always)]
    fn limit_value(&mut self, at: usize, b: u8) {
        if self.limit_value_next && !(b.is_ascii_whitespace() || b == b':') {
            self.limit_value_next = false;
            let null = self.body.get(at..).is_some_and(|r| r.starts_with(b"null"));
            if !(b.is_ascii_digit() || null) {
                self.inexact_limit = true;
            }
        }
    }

    fn root_key(&mut self, start: usize, end: usize, escaped: bool) {
        let key = self.body.get(start..end).unwrap_or_default();
        // Unlike a planner that only wants `stream` we cannot stop on `stream_options`: the model
        // may still be ahead of us. Record it and keep walking.
        if key == b"stream_options" {
            self.stream_options_ambiguous |= self.saw_stream_options;
            if !self.saw_stream_options {
                self.saw_stream_options = true;
                self.stream_options_key_end = end;
            }
        } else if escaped && lex::unescapes_to(key, "stream_options") {
            self.stream_options_ambiguous = true;
        }
        // The provider decodes an escaped key and keeps the last duplicate, so each key that
        // decodes to `stream` decides afresh (D88).
        self.last_key_is_stream = key_names(key, escaped, "stream");
        if self.last_key_is_stream {
            self.stream_true = false;
        }
        self.last_key_is_model = key_names(key, escaped, "model");
        self.model_keys = self
            .model_keys
            .saturating_add(u32::from(self.last_key_is_model));
        // A key spelled with escapes names the limit it decodes to (`"max\u005ftokens"`).
        self.limit_key = OUTPUT_LIMIT_KEYS
            .iter()
            .position(|k| std::str::from_utf8(k).is_ok_and(|want| key_names(key, escaped, want)));
        if let Some(k) = self.limit_key
            && let Some(seen) = self.limit_keys.get_mut(k)
        {
            // A second root spelling of the same limit: a parser keeps one of them, and only the
            // first is spanned, so it cannot be capped reliably. Refused.
            if seen.is_some() {
                self.inexact_limit = true;
            }
            seen.get_or_insert(start.saturating_sub(1));
            self.limit_value_next = true;
        }
    }

    /// A root limit's digits at `at`: spanned when they are the whole of a plain integer.
    fn limit_digits(&mut self, at: usize) {
        let body = self.body;
        let end = body
            .get(at..)
            .and_then(|r| r.iter().position(|c| !c.is_ascii_digit()))
            .map_or(body.len(), |rel| at.saturating_add(rel));
        // A plain integer ends its member: `,` or `}`, past any whitespace. A fraction or an
        // exponent ends in `.`, `e` or `E` instead, and is no integer.
        let ends_member = body
            .get(end..)
            .and_then(|r| r.iter().find(|c| !c.is_ascii_whitespace()))
            .is_some_and(|c| matches!(c, b',' | b'}'));
        if let Some(k) = self.limit_key.take() {
            if !ends_member {
                self.inexact_limit = true;
            } else if let Some(span @ None) = self.limit_spans.get_mut(k) {
                *span = Some((at, end));
            }
        }
    }
}

impl Visit for Fused<'_> {
    #[inline(always)]
    fn byte(&mut self, at: usize, b: u8, depth: u32) -> Flow {
        self.limit_value(at, b);
        match b {
            b'{' if depth == 0 => self.expect_key = true,
            // Unguarded by depth, as in the walk this one fused: a nested `:` is always followed
            // by the root `,` or `}` that closes its value before the next root key.
            b':' => self.expect_key = false,
            b',' if depth == 1 => {
                self.expect_key = true;
                self.last_key_is_stream = false;
                self.last_key_is_model = false;
                self.limit_key = None;
            }
            // The digits after it are no limit start: the first one took `limit_key`.
            b'0'..=b'9' if depth == 1 && !self.expect_key && self.limit_key.is_some() => {
                self.limit_digits(at);
            }
            // A negative limit is no limit to cap. (A string or structured value never puts a
            // digit at depth 1 before the `,` that ends it, so a nested `-` may clear it too.)
            b'-' => self.limit_key = None,
            b't' if depth == 1 && self.last_key_is_stream => {
                if self.body.get(at..).is_some_and(|r| r.starts_with(b"true")) {
                    self.stream_true = true;
                }
                self.last_key_is_stream = false;
            }
            _ => {}
        }
        Flow::Continue(())
    }

    #[inline(always)]
    fn open(&mut self, at: usize, depth: u32) {
        self.limit_value(at, b'"');
        let start = at.saturating_add(1);
        self.open = if depth == 1 && self.expect_key {
            Open::Key(start)
        } else if depth == 1 && self.last_key_is_model && self.model.is_none() {
            Open::Model(start)
        } else {
            Open::None
        };
    }

    #[inline(always)]
    fn close(&mut self, at: usize, depth: u32, escaped: bool) -> Flow {
        match std::mem::replace(&mut self.open, Open::None) {
            Open::Key(start) if depth == 1 => self.root_key(start, at, escaped),
            Open::Model(start) => {
                self.last_key_is_model = false;
                // `at` is the closing quote, `start` the byte after the opening one, so the span
                // covers exactly the value and neither quote.
                self.model_span = Some((start, at));
                let kept = at.min(start.saturating_add(MAX_CAPTURE));
                let raw = self.body.get(start..kept).unwrap_or_default().to_vec();
                // Same non-UTF-8 fallback as `ModelScanner`: record "unknown" rather than emit a
                // U+FFFD-corrupted model into the billing log.
                self.model =
                    Some(capture_text(raw, escaped).unwrap_or_else(|| "unknown".to_string()));
            }
            _ => {}
        }
        Flow::Continue(())
    }
}

/// Remove every root member whose value is `null`, by span: how OpenAI SDKs send an unset option,
/// which OpenAI reads as "not set" and other Chat Completions hosts may reject (OpenRouter 400s
/// `user: null`). One `memmem` keeps a body without `null` unwalked. `true` when the body changed.
pub fn remove_root_nulls(body: &mut Vec<u8>) -> bool {
    if memchr::memmem::find(body, b"null").is_none() {
        return false;
    }
    let Some(members) = root_members(body) else {
        return false;
    };
    let spans: Vec<(usize, usize)> = members.iter().map(Member::span).collect();
    let null: Vec<bool> = members
        .iter()
        .map(|m| body.get(m.value.0..m.value.1) == Some(b"null"))
        .collect();
    remove_items(body, &spans, |k| null.get(k).copied().unwrap_or(false))
}

/// Remove every root member whose key decodes to `key` (escaped spellings included), by span. `true`
/// when the body changed.
pub fn remove_root_members(body: &mut Vec<u8>, key: &str) -> bool {
    let Some(members) = root_members(body) else {
        return false;
    };
    let spans: Vec<(usize, usize)> = members.iter().map(Member::span).collect();
    let hit: Vec<bool> = members.iter().map(|m| m.key_is(body, key)).collect();
    remove_items(body, &spans, |k| hit.get(k).copied().unwrap_or(false))
}

/// Whether a quoted JSON key containing escapes decodes to `want`. Only reached for a root key with a
/// backslash in it, which real clients never send, so the allocation is off every ordinary path.
fn escaped_key_is(quoted: &[u8], want: &str) -> bool {
    quoted
        .strip_prefix(b"\"")
        .and_then(|q| q.strip_suffix(b"\""))
        .is_some_and(|raw| lex::unescapes_to(raw, want))
}

// ---- Span edits ------------------------------------------------------------------------------------
//
// A few rewrites (dropping a root member, dropping items from an array) must leave every other byte
// of the body as the client sent it: member order (strict structured outputs, schema property
// order), spacing, and number spellings are all part of what the provider sees and what a prompt
// cache keys on. A `serde_json::Value` round-trip loses all three. These helpers find spans with the
// same string/escape-aware walk as the scanners above and splice them out; nothing is re-encoded.

/// One member of a JSON object: the key's raw bytes (inside its quotes, escapes as sent) and the
/// value's extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    pub key: (usize, usize),
    pub value: (usize, usize),
}

impl Member {
    /// The member's extent: the key's opening quote through the value's last byte.
    pub fn span(&self) -> (usize, usize) {
        (self.key.0.saturating_sub(1), self.value.1)
    }

    /// Whether the key decodes to `want`. A raw compare, unless the key carries an escape
    /// (`"stream_options"`), which a provider's parser decodes and so must we.
    pub fn key_is(&self, body: &[u8], want: &str) -> bool {
        let Some(raw) = body.get(self.key.0..self.key.1) else {
            return false;
        };
        if raw.contains(&b'\\') {
            let quoted = body.get(self.key.0.saturating_sub(1)..self.key.1.saturating_add(1));
            quoted.is_some_and(|q| escaped_key_is(q, want))
        } else {
            raw == want.as_bytes()
        }
    }
}

fn skip_ws(b: &[u8], i: usize) -> usize {
    let rest = b.get(i..).unwrap_or_default();
    let n = rest.iter().take_while(|c| c.is_ascii_whitespace()).count();
    i.saturating_add(n)
}

/// The end (exclusive) of the JSON value starting at `i` (no leading whitespace).
pub fn value_end(b: &[u8], i: usize) -> Option<usize> {
    match *b.get(i)? {
        b'"' => string_end(b, i),
        b'{' | b'[' => {
            /// Stops at the bracket that closes the first one.
            struct Close;
            impl Visit for Close {
                #[inline(always)]
                fn byte(&mut self, _: usize, c: u8, depth: u32) -> Flow {
                    if matches!(c, b'}' | b']') && depth == 1 {
                        Flow::Break(())
                    } else {
                        Flow::Continue(())
                    }
                }
            }
            let end = Lexer::new().feed(b.get(i..)?, &mut Close)?;
            i.checked_add(end)
        }
        _ => {
            let n = b
                .get(i..)?
                .iter()
                .position(|c| matches!(c, b',' | b'}' | b']') || c.is_ascii_whitespace())
                .map_or(b.len(), |k| i.saturating_add(k));
            (n > i).then_some(n)
        }
    }
}

/// The members of the object whose `{` is at `open`, in order. `None` for malformed JSON.
pub fn object_members(b: &[u8], open: usize) -> Option<Vec<Member>> {
    members(b, open).collect()
}

/// The members of the body's root object. `None` when the root is not a well-formed object.
pub fn root_members(b: &[u8]) -> Option<Vec<Member>> {
    let open = skip_ws(b, 0);
    (b.get(open) == Some(&b'{')).then_some(())?;
    object_members(b, open)
}

/// The elements of the array whose `[` is at `open`, as spans, in order. `None` for malformed
/// JSON.
pub fn array_elements(b: &[u8], open: usize) -> Option<Vec<(usize, usize)>> {
    elements(b, open).collect()
}

/// Remove the items `drop` selects from one container's item spans (members or elements, in
/// order), with exactly the separators that keep it valid JSON. Every other byte is untouched.
/// `true` when anything was removed.
pub fn remove_items(
    body: &mut Vec<u8>,
    items: &[(usize, usize)],
    drop: impl Fn(usize) -> bool,
) -> bool {
    let mut cuts = Vec::new();
    item_cuts(items, drop, &mut cuts);
    splice_out(body, &cuts);
    !cuts.is_empty()
}

/// Remove every member whose key decodes to `key` from each object element of the array whose `[`
/// is at `open` (an Anthropic `tools` list's per-tool fields), with the separators that keep each
/// object valid JSON, in one pass over the body. Every other byte is untouched. `true` when
/// anything was removed; `false`, untouched, for malformed JSON.
pub fn remove_element_members(body: &mut Vec<u8>, open: usize, key: &str) -> bool {
    let Some(elements) = array_elements(body, open) else {
        return false;
    };
    let mut cuts = Vec::new();
    for (start, _) in elements {
        if body.get(start) != Some(&b'{') {
            continue;
        }
        let Some(members) = object_members(body, start) else {
            return false;
        };
        let spans: Vec<(usize, usize)> = members.iter().map(Member::span).collect();
        item_cuts(
            &spans,
            |k| members.get(k).is_some_and(|m| m.key_is(body, key)),
            &mut cuts,
        );
    }
    splice_out(body, &cuts);
    !cuts.is_empty()
}

/// The spans [`remove_items`] cuts for one container's items, appended to `cuts` in order.
pub(crate) fn item_cuts(
    items: &[(usize, usize)],
    drop: impl Fn(usize) -> bool,
    cuts: &mut Vec<(usize, usize)>,
) {
    let (Some(&(_, end)), Some(last)) = (items.last(), items.len().checked_sub(1)) else {
        return;
    };
    let last_kept = (0..=last).rev().find(|&k| !drop(k));
    for (k, &(start, _)) in items.iter().enumerate() {
        if !drop(k) {
            continue;
        }
        match last_kept.and_then(|l| Some((l, items.get(l)?))) {
            // Ahead of a kept item: the item and the separator after it, up to the next item.
            Some((l, _)) if k < l => {
                let next = items.get(k.saturating_add(1)).map_or(end, |n| n.0);
                cuts.push((start, next));
            }
            // The trailing run: from the last kept item's end, separators included.
            Some((_, kept)) => {
                cuts.push((kept.1, end));
                break;
            }
            // Nothing kept: the whole run, leaving an empty container.
            None => {
                cuts.push((start, end));
                break;
            }
        }
    }
}

/// Cut every `(start, end)` span (ascending, disjoint, in bounds) out of `body` in one pass: each
/// kept byte moves at most once, so a body with many cuts (a long history of references, every
/// `null` option) costs O(len), not a tail move per cut.
fn splice_out(body: &mut Vec<u8>, cuts: &[(usize, usize)]) {
    let mut write = match cuts.first() {
        Some(&(s, _)) => s,
        None => return,
    };
    for (k, &(_, end)) in cuts.iter().enumerate() {
        let next = cuts.get(k.saturating_add(1)).map_or(body.len(), |c| c.0);
        // Spans are ascending and disjoint (`remove_items` builds them in order): `end <= next`,
        // and two touching spans copy nothing.
        body.copy_within(end..next, write);
        write = write.saturating_add(next.saturating_sub(end));
    }
    body.truncate(write);
}

// ---- Span reads without a DOM ----------------------------------------------------------------------
//
// The admission checks (session state, image and file parts, item references) read a handful of
// fields out of bodies up to `MAX_REQUEST_BODY`. A `serde_json::Value` of such a body costs 10-80
// bytes of heap per byte of JSON (a `BTreeMap` node per object), so one 90 MiB request could take
// gigabytes. These walk spans instead: no allocation beyond the caller's own, O(len) time.

/// The members of the object whose `{` is at `open`, in order, one at a time: `Some(member)` per
/// member, then `None` once if the object is malformed (after which the iterator ends). See
/// [`members`].
pub struct Members<'a> {
    b: &'a [u8],
    at: usize,
    first: bool,
    done: bool,
    close: Option<usize>,
}

/// Iterate the members of the object whose `{` is at `open` without collecting them.
pub fn members(b: &[u8], open: usize) -> Members<'_> {
    Members {
        b,
        at: open.saturating_add(1),
        first: true,
        done: b.get(open) != Some(&b'{'),
        close: None,
    }
}

impl Members<'_> {
    /// Offset just past the closing `}`, once the iterator has reached it.
    pub fn end(&self) -> Option<usize> {
        self.close
    }

    /// The next member; `Ok(None)` at the closing brace, `Err(())` on malformed JSON.
    fn step(&mut self) -> Result<Option<Member>, ()> {
        let b = self.b;
        let mut i = skip_ws(b, self.at);
        if std::mem::take(&mut self.first) {
            if b.get(i) == Some(&b'}') {
                self.close = Some(i.saturating_add(1));
                return Ok(None);
            }
        } else {
            match b.get(i) {
                Some(b',') => i = skip_ws(b, i.saturating_add(1)),
                Some(b'}') => {
                    self.close = Some(i.saturating_add(1));
                    return Ok(None);
                }
                _ => return Err(()),
            }
        }
        if b.get(i) != Some(&b'"') {
            return Err(());
        }
        let key_end = string_end(b, i).ok_or(())?;
        let key = (i.saturating_add(1), key_end.saturating_sub(1));
        i = skip_ws(b, key_end);
        if b.get(i) != Some(&b':') {
            return Err(());
        }
        let start = skip_ws(b, i.saturating_add(1));
        let end = value_end(b, start).ok_or(())?;
        self.at = end;
        Ok(Some(Member {
            key,
            value: (start, end),
        }))
    }
}

impl Iterator for Members<'_> {
    type Item = Option<Member>;

    fn next(&mut self) -> Option<Option<Member>> {
        if self.done {
            return None;
        }
        match self.step() {
            Ok(Some(m)) => Some(Some(m)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(()) => {
                self.done = true;
                Some(None)
            }
        }
    }
}

/// The elements of the array whose `[` is at `open`, one span at a time, as [`Members`] yields
/// members: `Some(span)` per element, then `None` once if the array is malformed.
pub struct Elements<'a> {
    b: &'a [u8],
    at: usize,
    first: bool,
    done: bool,
}

/// Iterate the elements of the array whose `[` is at `open` without collecting them.
pub fn elements(b: &[u8], open: usize) -> Elements<'_> {
    Elements {
        b,
        at: open.saturating_add(1),
        first: true,
        done: b.get(open) != Some(&b'['),
    }
}

impl Elements<'_> {
    fn step(&mut self) -> Result<Option<(usize, usize)>, ()> {
        let b = self.b;
        let mut i = skip_ws(b, self.at);
        if std::mem::take(&mut self.first) {
            if b.get(i) == Some(&b']') {
                return Ok(None);
            }
        } else {
            match b.get(i) {
                Some(b',') => i = skip_ws(b, i.saturating_add(1)),
                Some(b']') => return Ok(None),
                _ => return Err(()),
            }
        }
        let end = value_end(b, i).ok_or(())?;
        self.at = end;
        Ok(Some((i, end)))
    }
}

impl Iterator for Elements<'_> {
    type Item = Option<(usize, usize)>;

    fn next(&mut self) -> Option<Option<(usize, usize)>> {
        if self.done {
            return None;
        }
        match self.step() {
            Ok(Some(span)) => Some(Some(span)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(()) => {
                self.done = true;
                Some(None)
            }
        }
    }
}

/// Offset of the root object's `{`, when the body (past leading whitespace) is an object.
pub fn root_open(b: &[u8]) -> Option<usize> {
    let open = skip_ws(b, 0);
    (b.get(open) == Some(&b'{')).then_some(open)
}

/// The last member keyed `key` (as a provider's parser keeps it) of the object whose `{` is at
/// `open`: `Some(None)` when it has none, `None` when `open` is not a well-formed object.
pub fn last_member(b: &[u8], open: usize, key: &str) -> Option<Option<Member>> {
    if b.get(open) != Some(&b'{') {
        return None;
    }
    let mut last = None;
    for m in members(b, open) {
        let m = m?;
        if m.key_is(b, key) {
            last = Some(m);
        }
    }
    Some(last)
}

/// The JSON string at `span` (its quotes included), decoded: borrowed when it carries no escape,
/// otherwise decoded once. `None` when `span` is not a string.
pub fn str_value(b: &[u8], span: (usize, usize)) -> Option<std::borrow::Cow<'_, str>> {
    match b.get(span.0..span.1)? {
        [b'"', inner @ .., b'"'] if !inner.contains(&b'\\') => std::str::from_utf8(inner)
            .ok()
            .map(std::borrow::Cow::Borrowed),
        [b'"', inner @ .., b'"'] => lex::unescape(inner).map(std::borrow::Cow::Owned),
        _ => None,
    }
}

/// Whether the JSON string at `span` (its quotes included) decodes to `want`. A raw compare, unless
/// the string carries an escape, which a provider's parser decodes and so must we; only then is it
/// decoded, and only when its length could still match.
pub fn str_is(b: &[u8], span: (usize, usize), want: &str) -> bool {
    let Some(quoted) = b.get(span.0..span.1) else {
        return false;
    };
    match quoted {
        [b'"', inner @ .., b'"'] if !inner.contains(&b'\\') => inner == want.as_bytes(),
        [b'"', .., b'"'] => escaped_key_is(quoted, want),
        _ => false,
    }
}

/// Whether a `"type"` member anywhere inside `span` has a string value that is one of `types`: an
/// image or file content part, at any depth (tool results included).
///
/// No walk of the structure: in valid JSON every `"` not escaped by an odd run of backslashes is a
/// string delimiter, and string contents never hold one, so the bytes `"type"`, `:`, `"<value>"`
/// with an unescaped opening quote are exactly a member `type: <value>`. The scan is one `memmem`
/// pass plus O(1) per hit. A key or value spelled with escapes (`"type"`) is not read as one;
/// no client sends them. A part that names its type twice counts if either is listed: refused, or
/// steered, rather than guessed at.
pub fn has_typed_member(b: &[u8], span: (usize, usize), types: &[&str]) -> bool {
    const KEY: &[u8] = b"\"type\"";
    let Some(region) = b.get(span.0..span.1) else {
        return false;
    };
    memchr::memmem::find_iter(region, KEY).any(|at| {
        let at = span.0.saturating_add(at);
        if escaped_at(b, at) {
            return false;
        }
        let colon = skip_ws(b, at.saturating_add(KEY.len()));
        if b.get(colon) != Some(&b':') {
            return false;
        }
        let open = skip_ws(b, colon.saturating_add(1));
        if b.get(open) != Some(&b'"') || open >= span.1 {
            return false;
        }
        string_end(b, open).is_some_and(|end| types.iter().any(|t| str_is(b, (open, end), t)))
    })
}

/// Whether the byte at `at` is escaped: preceded by an odd run of backslashes.
fn escaped_at(b: &[u8], at: usize) -> bool {
    let before = b.get(..at).unwrap_or_default();
    before.iter().rev().take_while(|&&c| c == b'\\').count() % 2 == 1
}

/// Replace every root member keyed `key` (escaped spellings included) with one `"key":value`, placed
/// first; every other byte stays as sent. `key` must need no escaping, and `value` must be one JSON
/// value. `false`, untouched, when the root is not a well-formed object.
pub fn set_root_member(body: &mut Vec<u8>, key: &str, value: &[u8]) -> bool {
    let Some(found) = root_members(body) else {
        return false;
    };
    let spans: Vec<(usize, usize)> = found.iter().map(Member::span).collect();
    let hit: Vec<bool> = found.iter().map(|m| m.key_is(body, key)).collect();
    let had_others = hit.iter().any(|h| !h);
    remove_items(body, &spans, |k| hit.get(k).copied().unwrap_or(false));
    // Still an object: only whole members were cut, never its braces.
    let Some(open) = root_open(body) else {
        return false;
    };
    let mut member = Vec::with_capacity(key.len().saturating_add(value.len()).saturating_add(4));
    member.push(b'"');
    member.extend_from_slice(key.as_bytes());
    member.extend_from_slice(b"\":");
    member.extend_from_slice(value);
    if had_others {
        member.push(b',');
    }
    let at = open.saturating_add(1);
    body.splice(at..at, member);
    true
}

/// What a `serde_json::Value` of a body would hold, counted from its bytes without parsing it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Structure {
    /// Objects with at least one member: each costs a `BTreeMap` node in a `Value`.
    pub objects: usize,
    /// Every `{`, `[`, `,` and `:` outside a string. Each value past the first is announced by one
    /// (a member's key and value by the `,` or `{` before it and its `:`), so this bounds the
    /// values, keys and strings the `Value` would hold.
    pub tokens: usize,
}

/// [`Structure`] of `b`, in one pass: string contents are skipped with `memchr`.
pub fn structure(b: &[u8]) -> Structure {
    struct Count<'a> {
        b: &'a [u8],
        s: Structure,
    }
    impl Visit for Count<'_> {
        #[inline(always)]
        fn byte(&mut self, at: usize, c: u8, _: u32) -> Flow {
            if matches!(c, b'{' | b'[' | b',' | b':') {
                self.s.tokens = self.s.tokens.saturating_add(1);
                // An object is empty when only whitespace stands between its braces.
                let next = skip_ws(self.b, at.saturating_add(1));
                if c == b'{' && self.b.get(next) != Some(&b'}') {
                    self.s.objects = self.s.objects.saturating_add(1);
                }
            }
            Flow::Continue(())
        }
    }
    // An unterminated string ends the count: nothing after it is outside a string.
    let mut count = Count {
        b,
        s: Structure::default(),
    };
    Lexer::new().feed(b, &mut count);
    count.s
}

#[cfg(test)]
#[path = "peek_kept_tests.rs"]
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
mod kept_tests;

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    /// Where `scan_buffered` would splice the `stream_options` fragment.
    fn inject_at(body: &[u8]) -> Option<usize> {
        scan_buffered(body).inject_at
    }

    /// The stand-alone injection planner `scan_buffered` fused, kept as the oracle of
    /// [`fused_scan_matches_the_two_walks_it_replaces`]. Do not fix anything here.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    pub(super) fn plan_stream_usage_injection(body: &[u8]) -> Option<usize> {
        // Proof for the allow: every index is `i`/`j` with `i < n` (the loop condition, or a `memchr`
        // offset inside `bytes[i..]`), or a key span bounded by the closing quote just found; counters
        // (`i`, `j`, `depth`, `model_keys`) count bytes of a body that fits in memory, so they cannot
        // overflow. This loop runs on every managed request body, so it keeps the unchecked forms.
        let n = body.len();
        // Cheap pre-filter: injection is only ever needed when a root-level `"stream"` key is present.
        // If the quoted token `"stream"` doesn't occur *anywhere*, the structural answer is
        // unconditionally `None`, so a single SIMD `memmem` pass lets us skip the whole walk — the
        // common case, since most requests aren't streaming. (Note `"stream_options"` does NOT contain
        // the needle: the byte after `stream` is `_`, not a closing quote — so a body carrying only
        // `stream_options` fails this pre-filter and returns `None` here, which is the correct answer.)
        // An escaped key (`"str\u0065am"`) decodes to `stream` at the provider without spelling it,
        // and such a key always carries `\u`; real clients never send one, so the second search runs
        // only on a body with no `"stream"` at all.
        if memchr::memmem::find(body, b"\"stream\"").is_none()
            && memchr::memmem::find(body, b"\\u").is_none()
        {
            return None;
        }
        // Must be a JSON object; anything else (array, scalar, garbage) we never rewrite.
        let i = body.iter().position(|b| !b.is_ascii_whitespace())?;
        if body[i] != b'{' {
            return None;
        }
        let insert_at = i + 1;

        let mut depth = 0u32;
        let mut in_string = false;
        let mut escaped = false;
        let mut expect_key = false;
        let mut capturing_key = false;
        // Start index (just past the opening `"`) of the root-level key currently being scanned. The
        // body is fully in hand, so we slice the key out of it at the closing quote — no accumulation
        // buffer, zero-copy. A key with an escape in it is decoded, as the provider's parser does.
        let mut key_start = 0usize;
        // The current root-level key is exactly `stream` (so the next literal is its value).
        let mut last_key_is_stream = false;
        let mut stream_true = false;

        let mut j = i;
        while j < n {
            if in_string {
                // Fast path: jump straight to the next `"`/`\` with a SIMD search instead of
                // inspecting every byte, a root key included (it is sliced out of the body at its
                // closing quote). Mirrors the skip in `ModelScanner::feed`.
                if !escaped {
                    match memchr::memchr2(b'"', b'\\', &body[j..]) {
                        Some(rel) => j += rel,
                        None => break, // rest of the body is skippable string content
                    }
                }
                let b = body[j];
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                    if capturing_key {
                        capturing_key = false;
                        // Only root-level (`depth == 1`) keys matter.
                        if depth == 1 {
                            let key = &body[key_start..j];
                            // A root `stream_options` means the client already controls usage — the
                            // answer is `None` regardless of anything else in the body, so stop now
                            // rather than walking the remainder for a result we already know.
                            let escaped_key = key.contains(&b'\\');
                            if key == b"stream_options"
                                || (escaped_key
                                    && escaped_key_is(&body[key_start - 1..=j], "stream_options"))
                            {
                                return None;
                            }
                            // The provider keeps the last of duplicate keys, so each `stream` key
                            // decides afresh.
                            last_key_is_stream = key == b"stream"
                                || (escaped_key
                                    && escaped_key_is(&body[key_start - 1..=j], "stream"));
                            if last_key_is_stream {
                                stream_true = false;
                            }
                        }
                    }
                }
                j += 1;
                continue;
            }
            let b = body[j];
            match b {
                b'"' => {
                    // A root-level key starts only where one is expected (just after `{` or `,`).
                    if depth == 1 && expect_key {
                        capturing_key = true;
                        key_start = j + 1; // first key byte is just past this opening quote
                    } else {
                        capturing_key = false;
                    }
                    in_string = true;
                }
                b'{' => {
                    depth += 1;
                    if depth == 1 {
                        expect_key = true;
                    }
                }
                b'[' => depth += 1,
                b'}' | b']' => depth = depth.saturating_sub(1),
                // Unguarded by depth, as in `ModelScanner::feed`: a nested `:` or `,` is always
                // followed by the root `,` or `}` that closes its value before the next root key.
                b':' => expect_key = false,
                b',' => {
                    expect_key = true;
                    last_key_is_stream = false;
                }
                // The value of a root-level `stream` key: a bare `true` literal.
                b't' if depth == 1 && last_key_is_stream => {
                    if body[j..].starts_with(b"true") {
                        stream_true = true;
                    }
                    last_key_is_stream = false;
                }
                _ => {}
            }
            j += 1;
        }

        // `stream_options` would have already returned `None` above, so reaching here means it's absent.
        if stream_true { Some(insert_at) } else { None }
    }

    /// The iterators agree with the collecting walks, end at the closing bracket, and report a
    /// malformed container once.
    #[test]
    fn members_and_elements_walk_without_collecting() {
        let b = br#" { "a" : [1, {"b":2}, "x\"]"] , "c":{} } "#;
        let open = root_open(b).unwrap();
        let mut it = members(b, open);
        let got: Vec<Member> = it.by_ref().map(Option::unwrap).collect();
        assert_eq!(got, object_members(b, open).unwrap());
        assert_eq!(it.end(), Some(b.len() - 1));
        let arr = got[0].value.0;
        let spans: Vec<(usize, usize)> = elements(b, arr).map(Option::unwrap).collect();
        assert_eq!(spans, array_elements(b, arr).unwrap());
        assert_eq!(members(b"{}", 0).count(), 0);
        assert_eq!(elements(b"[ ]", 0).count(), 0);
        for bad in [
            &br#"{"a":1 "b":2}"#[..],
            br#"{"a"}"#,
            br#"{"a":1,"#,
            b"{1:2}",
        ] {
            let items: Vec<Option<Member>> = members(bad, 0).collect();
            assert_eq!(
                items.last(),
                Some(&None),
                "{}",
                String::from_utf8_lossy(bad)
            );
        }
        assert_eq!(elements(b"[1 2]", 0).last(), Some(None));
        assert_eq!(members(b"[1]", 0).count(), 0, "not an object");
    }

    #[test]
    fn last_member_and_str_reads() {
        let b = br#"{"type":"a","type":"image","n":null}"#;
        let m = last_member(b, 0, "type").unwrap().unwrap();
        assert!(str_is(b, m.value, "image"));
        assert_eq!(str_value(b, m.value).as_deref(), Some("image"));
        assert_eq!(last_member(b, 0, "none"), Some(None));
        assert_eq!(last_member(b"[1]", 0, "type"), None);
        let n = last_member(b, 0, "n").unwrap().unwrap();
        assert!(!str_is(b, n.value, "null"));
        assert_eq!(str_value(b, n.value), None);
        // An escaped value is decoded, as the provider's parser decodes it.
        let e = br#"{"type":"im\u0061ge"}"#;
        let m = last_member(e, 0, "type").unwrap().unwrap();
        assert_eq!(str_value(e, m.value).as_deref(), Some("image"));
        // A string far longer than the name is not decoded to be compared.
        let long = format!(r#"{{"k":"{}"}}"#, "\\u0041".repeat(10_000));
        let m = last_member(long.as_bytes(), 0, "k").unwrap().unwrap();
        assert!(!str_is(long.as_bytes(), m.value, "A"));
    }

    /// A `"type"` member is found at any depth inside the span, never inside a string, and only
    /// with one of the listed values.
    #[test]
    fn has_typed_member_reads_only_real_type_members() {
        let types = ["image_url", "image"];
        let hit = |b: &[u8]| has_typed_member(b, (0, b.len()), &types);
        assert!(hit(
            br#"[{"type":"text"},{"content":[{"type" : "image", "x":1}]}]"#
        ));
        assert!(hit(br#"{"type":"image_url"}"#));
        assert!(!hit(br#"[{"type":"text","text":"\"type\":\"image\""}]"#));
        assert!(!hit(
            br#"[{"type":"text","text":"say \\"},{"type":"input_image"}]"#
        ));
        assert!(!hit(br#"[{"name":"type","value":"image"}]"#));
        assert!(!hit(br#"[{"type":"images"}]"#));
        // A string ending in an escaped backslash leaves the next quote unescaped.
        assert!(hit(br#"[{"text":"a\\","type":"image"}]"#));
    }

    /// The one root-member splice behind `store_false`, `force_stream` and
    /// `fit_tools_to_thinking`: every member under the key goes, one goes first, the rest of the
    /// bytes stay as sent.
    #[test]
    fn set_root_member_replaces_every_spelling_and_keeps_the_rest() {
        let set = |body: &str, key: &str, value: &str| {
            let mut b = body.as_bytes().to_vec();
            let changed = set_root_member(&mut b, key, value.as_bytes());
            (changed, String::from_utf8(b).unwrap())
        };
        assert_eq!(
            set(
                r#" {"a":1, "store":true,"b" : [2],"store":null}"#,
                "store",
                "false"
            ),
            (true, r#" {"store":false,"a":1, "b" : [2]}"#.to_owned())
        );
        assert_eq!(
            set(r#"{"store":true}"#, "store", "false"),
            (true, r#"{"store":false}"#.to_owned())
        );
        assert_eq!(set("{}", "s", "true"), (true, r#"{"s":true}"#.to_owned()));
        assert_eq!(set("[1]", "s", "true"), (false, "[1]".to_owned()));
        assert_eq!(set(r#"{"a":"#, "s", "true"), (false, r#"{"a":"#.to_owned()));
    }

    /// Many cuts cost one pass, and land exactly where `remove_items` asked.
    #[test]
    fn remove_items_cuts_many_items_in_one_pass() {
        let items: Vec<String> = (0..2000).map(|i| format!(r#"{{"k":{i}}}"#)).collect();
        let mut body = format!("[{}]", items.join(",")).into_bytes();
        let spans = array_elements(&body, 0).unwrap();
        assert!(remove_items(&mut body, &spans, |k| k % 3 != 1));
        let want: Vec<&String> = items.iter().skip(1).step_by(3).collect();
        let want = format!(
            "[{}]",
            want.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
        assert_eq!(String::from_utf8(body).unwrap(), want);
    }

    #[test]
    fn structure_counts_objects_and_tokens_outside_strings() {
        assert_eq!(
            structure(br#"{"a":[1,{"b":"{[,:]}"},{ },[]]}"#),
            Structure {
                objects: 2,
                tokens: 10
            }
        );
        assert_eq!(structure(b""), Structure::default());
        assert_eq!(structure(br#""unterminated {"#), Structure::default());
    }

    /// Only a root member whose value is the literal `null` is removed (D101): not a string that
    /// starts with `n`, not a nested `null`, not a role named user.
    #[test]
    fn remove_root_nulls_removes_only_root_nulls() {
        fn strip(body: &[u8]) -> (bool, Vec<u8>) {
            let mut b = body.to_vec();
            let changed = remove_root_nulls(&mut b);
            (changed, b)
        }
        let (changed, b) = strip(br#"{"model":"m","user" : null}"#);
        assert!(changed);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&b).unwrap(),
            serde_json::json!({"model": "m"})
        );
        assert!(!strip(br#"{"user":"nobody","model":"m"}"#).0);
        let nested = br#"{"messages":[{"role":"user","user":null}],"metadata":{"user":null}}"#;
        assert_eq!(strip(nested), (false, nested.to_vec()));
        let (changed, b) = strip(br#"{"user":"u","x":null}"#);
        assert!(changed);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&b).unwrap(),
            serde_json::json!({"user": "u"})
        );
    }

    /// Only root-level, non-negative integer output limits are spanned: the digits exactly, the
    /// first of each key, never one nested in a message or tool schema.
    #[test]
    fn output_limit_spans_cover_root_integers_only() {
        let body = br#"{"messages":[{"max_tokens":7}],"max_tokens" : 64000,"max_output_tokens":-1,"max_completion_tokens":"9","model":"m","max_tokens":5}"#;
        let scan = scan_buffered(body);
        let text = |s: Option<(usize, usize)>| s.map(|(a, b)| &body[a..b]);
        assert_eq!(text(scan.limit_spans[0]), Some(&b"64000"[..]));
        assert_eq!(scan.limit_spans[1], None, "a string is no limit");
        assert_eq!(scan.limit_spans[2], None, "a negative is no limit");
        let key = |at: Option<usize>| at.map(|a| &body[a..a + 12]);
        assert_eq!(key(scan.limit_keys[0]), Some(&br#""max_tokens""#[..]));
        assert_eq!(
            scan.limit_keys[1].map(|a| &body[a..a + 23]),
            Some(&br#""max_completion_tokens""#[..]),
            "a key is recorded whatever its value"
        );
        assert_eq!(scan.model.as_deref(), Some("m"));
    }

    /// A pathological `model` value is held to a bounded prefix, fed in small chunks the way a
    /// streamed body arrives, and the buffered scan agrees — while the span still covers the whole
    /// raw value, so a model-routed rewrite replaces all of it.
    #[test]
    fn an_oversized_model_value_is_capped_not_materialized() {
        let huge = "a".repeat(MAX_CAPTURE * 64);
        let body = format!(r#"{{"model":"{huge}","messages":[]}}"#);
        let mut s = ModelScanner::new();
        for chunk in body.as_bytes().chunks(1024) {
            s.feed(chunk);
            assert!(
                s.cur.capacity() <= MAX_CAPTURE * 2,
                "accumulation must stay bounded"
            );
        }
        let m = s.take_model().expect("value still closes");
        assert_eq!(m.len(), MAX_CAPTURE);

        let fused = scan_buffered(body.as_bytes());
        assert_eq!(fused.model.as_deref(), Some(m.as_str()));
        let (start, end) = fused.model_span.expect("span");
        assert_eq!(end - start, huge.len());
    }

    /// An overlong key is capped too, and a capped key can never be mistaken for `model`.
    #[test]
    fn an_oversized_key_is_capped_and_does_not_match() {
        let key = format!("model{}", "x".repeat(MAX_CAPTURE * 4));
        let body = format!(r#"{{"{key}":"nope","model":"gpt-4o"}}"#);
        assert_eq!(scan(body.as_bytes()).as_deref(), Some("gpt-4o"));
    }

    /// claim: SEC-21
    /// defect: D273
    #[test]
    fn an_escaped_model_key_is_found_by_the_streaming_scanner() {
        let body = format!(r#"{{"mod{}u0065l":"x"}}"#, '\\');
        let oracle: serde_json::Value = serde_json::from_slice(body.as_bytes()).unwrap();
        assert_eq!(oracle["model"], "x");
        assert_eq!(scan(body.as_bytes()).as_deref(), Some("x"));
        assert_eq!(scan_buffered(body.as_bytes()).model.as_deref(), Some("x"));
    }

    /// claim: SEC-21, BIL-13
    /// defect: D274
    #[test]
    fn a_unicode_escape_in_a_model_value_is_decoded() {
        let body = format!(r#"{{"model":"gpt-{}u0035"}}"#, '\\');
        let oracle: serde_json::Value = serde_json::from_slice(body.as_bytes()).unwrap();
        assert_eq!(oracle["model"], "gpt-5");
        assert_eq!(scan(body.as_bytes()).as_deref(), Some("gpt-5"));
        assert_eq!(
            scan_buffered(body.as_bytes()).model.as_deref(),
            Some("gpt-5")
        );
    }

    /// claim: SEC-21, BIL-13
    /// defect: D274
    #[test]
    fn a_newline_escape_in_a_model_value_is_decoded() {
        let body = br#"{"model":"a\nb"}"#;
        let oracle: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(oracle["model"], "a\nb");
        assert_eq!(scan(body).as_deref(), Some("a\nb"));
        assert_eq!(scan_buffered(body).model.as_deref(), Some("a\nb"));
    }

    /// claim: SEC-21, BIL-13
    /// defect: D274
    #[test]
    fn a_slash_escape_in_a_model_value_is_decoded() {
        let body = br#"{"model":"openai\/gpt-5"}"#;
        assert_eq!(scan(body).as_deref(), Some("openai/gpt-5"));
        assert_eq!(scan_buffered(body).model.as_deref(), Some("openai/gpt-5"));
    }

    /// claim: TRN-5
    /// defect: D275
    #[test]
    fn a_fractional_output_limit_has_no_span() {
        let body = br#"{"max_tokens":1.5,"model":"m"}"#;
        let scan = scan_buffered(body);
        assert_eq!(scan.limit_spans[0], None);
        assert!(scan.inexact_limit);
    }

    /// claim: TRN-5
    /// defect: D275
    #[test]
    fn an_exponent_output_limit_has_no_span() {
        let body = br#"{"max_tokens":1e5,"model":"m"}"#;
        let scan = scan_buffered(body);
        assert_eq!(scan.limit_spans[0], None);
        assert!(scan.inexact_limit);
    }

    /// A limit key spelled with escapes is the limit it decodes to, so it is spanned and capped.
    /// claim: TRN-5
    /// defect: D275
    #[test]
    fn an_escaped_output_limit_key_is_spanned() {
        let bs = '\\';
        let body = format!(r#"{{"max{bs}u005ftokens":99999,"model":"m"}}"#);
        let oracle: serde_json::Value = serde_json::from_slice(body.as_bytes()).unwrap();
        assert_eq!(oracle["max_tokens"], 99999);
        let scan = scan_buffered(body.as_bytes());
        let (a, b) = scan.limit_spans[0].expect("the escaped key is max_tokens");
        assert_eq!(&body.as_bytes()[a..b], b"99999");
        assert!(!scan.inexact_limit);
    }

    /// A second root spelling of one limit cannot be capped reliably (a parser keeps one of them), so
    /// it is inexact, whatever its values.
    /// claim: TRN-5
    /// defect: D275
    #[test]
    fn a_duplicate_output_limit_key_is_inexact() {
        let scan = scan_buffered(br#"{"max_tokens":8,"max_tokens":99999,"model":"m"}"#);
        assert!(scan.inexact_limit);
        let other = scan_buffered(br#"{"max_tokens":8,"max_completion_tokens":9,"model":"m"}"#);
        assert!(
            !other.inexact_limit,
            "two different limits are not a duplicate"
        );
    }

    /// Any first byte that is not a digit, `null` or whitespace makes the limit inexact: a leading
    /// `+` or `.` is no integer either.
    /// claim: TRN-5
    /// defect: D275
    #[test]
    fn a_limit_starting_with_plus_or_dot_is_inexact() {
        for body in [
            r#"{"max_tokens":+5,"model":"m"}"#,
            r#"{"max_tokens":.5,"model":"m"}"#,
        ] {
            assert!(scan_buffered(body.as_bytes()).inexact_limit, "{body}");
        }
        assert!(!scan_buffered(br#"{"max_tokens": null ,"model":"m"}"#).inexact_limit);
        // An `n` that is not the literal `null` is no null: inexact.
        for body in [
            r#"{"max_tokens":nope,"model":"m"}"#,
            r#"{"max_tokens":nul,"model":"m"}"#,
        ] {
            assert!(scan_buffered(body.as_bytes()).inexact_limit, "{body}");
        }
    }

    /// claim: TRN-5
    /// defect: D275
    #[test]
    fn only_a_non_integer_output_limit_is_inexact() {
        let flagged = |body: &str| scan_buffered(body.as_bytes()).inexact_limit;
        assert!(flagged(r#"{"max_tokens":"100","model":"m"}"#), "a string");
        assert!(flagged(r#"{"max_tokens":-1,"model":"m"}"#), "a negative");
        assert!(flagged(r#"{"max_tokens":true,"model":"m"}"#), "a bool");
        assert!(
            flagged(r#"{"max_tokens":{"n":1},"model":"m"}"#),
            "an object"
        );
        assert!(
            flagged(r#"{"max_tokens":1 .5,"model":"m"}"#),
            "a fraction past whitespace"
        );
        assert!(
            !flagged(r#"{"max_tokens":null,"model":"m"}"#),
            "null is no value"
        );
        assert!(
            !flagged(r#"{"max_tokens" : 64000 ,"model":"m"}"#),
            "a plain integer"
        );
        assert!(
            !flagged(r#"{"messages":[{"max_tokens":"x"}],"max_tokens":8}"#),
            "a nested one"
        );
    }

    fn scan(body: &[u8]) -> Option<String> {
        let mut s = ModelScanner::new();
        s.feed(body);
        s.take_model()
    }

    #[test]
    fn found_flips_only_when_the_model_value_closes() {
        let body = br#"{"messages":[{"content":"hello"}],"model":"gpt-4o"}"#;
        let at = body
            .windows(7)
            .position(|w| w == b"\"model\"")
            .expect("fixture has a model key");
        let mut s = ModelScanner::new();
        s.feed(&body[..at]);
        assert!(!s.found(), "the value has not closed yet");
        s.feed(&body[at..]);
        assert!(s.found());
        assert_eq!(s.take_model().as_deref(), Some("gpt-4o"));
        // `done` stays set: a later feed must stay a no-op even after the string is taken.
        assert!(s.found());
    }

    fn scan_response(body: &[u8]) -> Option<String> {
        let mut s = ModelScanner::for_response();
        s.feed(body);
        s.take_model()
    }

    /// `model_span` must bracket exactly the value's raw bytes — neither quote, nothing else. The
    /// model-routed path overwrites this range in place, so an off-by-one produces a body that is
    /// still valid JSON with a subtly wrong model, which no downstream check would catch.
    #[test]
    fn model_span_brackets_exactly_the_value_bytes() {
        let cases: &[&[u8]] = &[
            br#"{"model":"gpt-4o-mini"}"#,
            br#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o-mini"}"#,
            br#"{"model":"gpt-4o-mini","stream":true}"#,
            // A nested "model" must not be the one located.
            br#"{"tools":[{"model":"decoy"}],"model":"gpt-4o-mini"}"#,
            // Whitespace around the value's colon, and a leading-whitespace root.
            b"  {\n  \"model\" : \"gpt-4o-mini\"\n}",
        ];
        for body in cases {
            let scan = scan_buffered(body);
            let span = scan.model_span;
            assert!(
                span.is_some(),
                "no span for {:?}",
                String::from_utf8_lossy(body)
            );
            if let Some((start, end)) = span {
                assert_eq!(
                    &body[start..end],
                    b"gpt-4o-mini",
                    "span covers the wrong bytes in {:?}",
                    String::from_utf8_lossy(body),
                );
                // The bytes immediately outside the span are the delimiting quotes.
                assert_eq!(body[start - 1], b'"', "span must start inside the quote");
                assert_eq!(body[end], b'"', "span must end before the quote");
            }
            assert_eq!(scan.model.as_deref(), Some("gpt-4o-mini"));
        }
    }

    /// No root-level model ⇒ no span. The two answers always agree about whether one was found.
    #[test]
    fn model_span_is_absent_exactly_when_the_model_is() {
        let cases: &[&[u8]] = &[
            br#"{"messages":[]}"#,
            br#"[]"#,
            br#"{"tools":[{"model":"nested-only"}]}"#,
            b"",
        ];
        for body in cases {
            let scan = scan_buffered(body);
            assert_eq!(
                scan.model.is_some(),
                scan.model_span.is_some(),
                "model and model_span disagree on {:?}",
                String::from_utf8_lossy(body),
            );
        }
    }

    /// claim: SEC-21
    /// defect: D34
    #[test]
    fn a_second_root_model_key_is_flagged_however_it_is_spelled() {
        for body in [
            br#"{"model":"cheap","messages":[],"model":"o1-pro"}"#.as_slice(),
            br#"{"model":"cheap","mod\u0065l":"o1-pro"}"#,
            br#"{"mod\u0065l":"o1-pro","model":"cheap"}"#,
        ] {
            let scan = scan_buffered(body);
            assert!(scan.duplicate_model, "{}", String::from_utf8_lossy(body));
        }
        for body in [
            br#"{"model":"cheap","messages":[{"model":"nested"}],"metadata":{"model":"x"}}"#
                .as_slice(),
            br#"{"model":"cheap","system":"\"model\":\"o1-pro\""}"#,
            br#"{"models":["a"],"model":"cheap"}"#,
        ] {
            assert!(
                !scan_buffered(body).duplicate_model,
                "{}",
                String::from_utf8_lossy(body)
            );
        }
        // An escaped spelling alone is still the model: its value is what gets rewritten.
        let scan = scan_buffered(br#"{"mod\u0065l":"o1-pro"}"#);
        assert_eq!(scan.model.as_deref(), Some("o1-pro"));
        assert!(scan.model_span.is_some());
    }

    #[test]
    fn fused_scan_matches_the_two_walks_it_replaces() {
        // `scan_buffered` exists only to produce the same two answers in one pass. Anything it does
        // differently is a bug, so assert equivalence against both originals over the corpus —
        // including the cases where the two walks disagree with each other's shortcuts (the
        // planner returns early on `stream_options`; the fused walk must keep going for the model).
        let big = "x".repeat(64 * 1024);
        let cases: Vec<Vec<u8>> = vec![
            br#"{"model":"gpt-4o","stream":true,"messages":[]}"#.to_vec(),
            br#"{"model":"gpt-4o","messages":[]}"#.to_vec(),
            br#"{"stream":true,"stream_options":{"include_usage":false},"model":"after-opts"}"#
                .to_vec(),
            br#"{"stream_options":{},"stream":true,"model":"x"}"#.to_vec(),
            br#"{"messages":[{"role":"u","stream":true,"model":"NESTED"}],"model":"real"}"#
                .to_vec(),
            br#"{"system":"set stream:true please","model":"x"}"#.to_vec(),
            br#"{"messages":[],"stream":true,"model":"claude-opus-4-8"}"#.to_vec(),
            br#"{"messages":[],"stream":true}"#.to_vec(),
            br#"{"model":"openrouter/meta-llama/llama-3.1"}"#.to_vec(),
            br#"{"model":"gpt-4\"o","stream":true}"#.to_vec(), // escaped value must be unescaped
            br#"{"model":"a","model":"b","stream":true}"#.to_vec(), // first root model wins
            b"  {  \"stream\" : true , \"model\" : \"m1\" }".to_vec(),
            b"[1,2,3]".to_vec(),
            b"not json".to_vec(),
            b"".to_vec(),
            b"{}".to_vec(),
            // Not JSON: a key with no `:` before its nested value. Both walks read the nested `:`
            // as the root's, so neither takes the next root string for a key.
            br#"{"a" {"b":1} "model":"x"}"#.to_vec(),
            br#"{"a" {"b":1} "stream":true}"#.to_vec(),
            format!(r#"{{"messages":[{{"content":"{big}"}}],"stream":true,"model":"gpt-4o"}}"#)
                .into_bytes(),
            format!(r#"{{"system":"{big} \"stream\":true","model":"x"}}"#).into_bytes(),
        ];

        for body in &cases {
            let want_model = scan(body);
            let want_inject = plan_stream_usage_injection(body);
            let got = scan_buffered(body);
            let shown = String::from_utf8_lossy(&body[..body.len().min(90)]);
            assert_eq!(
                got.model, want_model,
                "model diverged from ModelScanner for {shown}"
            );
            assert_eq!(
                got.inject_at, want_inject,
                "inject_at diverged from plan_stream_usage_injection for {shown}"
            );
        }
    }

    #[test]
    fn response_scanner_reads_anthropics_nested_message_start_model() {
        // Anthropic's streaming wire nests the model under a root `message` key, so a root-only
        // scanner never matched it — and therefore never set `done`, so it walked the entire stream
        // to return `None`. Two costs: the billing log fell back to the *requested* alias instead of
        // the id the response was billed under, and every Anthropic stream paid a full byte-walk.
        let message_start = br#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8","content":[],"usage":{"input_tokens":5000}}}"#;
        assert_eq!(
            scan_response(message_start).as_deref(),
            Some("claude-opus-4-8")
        );
        // The strict scanner used on request bodies must NOT pick it up — that is the whole reason
        // the nesting is opt-in.
        assert_eq!(scan(message_start), None);
    }

    #[test]
    fn response_scanner_reads_a_responses_streams_nested_model() {
        let created = b"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"model\":\"gpt-5-2025-08-07\",\"output\":[]}}\n\n";
        assert_eq!(scan_response(created).as_deref(), Some("gpt-5-2025-08-07"));
        // Still exactly one level: a model inside the response's tools is not the response's.
        assert_eq!(
            scan_response(br#"{"response":{"tools":[{"model":"NESTED"}]}}"#),
            None
        );
    }

    #[test]
    fn response_scanner_stops_early_on_a_long_anthropic_stream() {
        // The point of finding it is that the scanner can then stop. Feed a stream whose deltas
        // continue long past `message_start`, and confirm the model came out — `feed` short-circuits
        // on `done`, so a scanner that had to reach the end could not have produced this.
        let mut sse = String::from(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":1}}}\n\n",
        );
        while sse.len() < 256 * 1024 {
            sse.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\" tok\"}}\n\n");
        }
        let mut s = ModelScanner::for_response();
        for chunk in sse.as_bytes().chunks(8 * 1024) {
            s.feed(chunk);
        }
        assert_eq!(s.take_model().as_deref(), Some("claude-opus-4-8"));
    }

    #[test]
    fn response_scanner_prefers_a_root_model_and_ignores_deeper_nesting() {
        // OpenAI's shape is unchanged: a root `model` still wins, and is found first.
        assert_eq!(
            scan_response(br#"{"id":"c","model":"gpt-4o-2024-08-06","choices":[]}"#).as_deref(),
            Some("gpt-4o-2024-08-06")
        );
        // Anthropic non-streaming also carries a root `model`.
        assert_eq!(
            scan_response(br#"{"id":"msg_1","model":"claude-opus-4-8","content":[]}"#).as_deref(),
            Some("claude-opus-4-8")
        );
        // Only `message` opens the nested scan — an arbitrary nested object must stay ignored, or
        // the billed model could be read out of an unrelated field.
        assert_eq!(
            scan_response(br#"{"choices":[{"model":"NESTED"}],"other":{"model":"ALSO-NESTED"}}"#),
            None
        );
        // ...and nesting stops at one level: `message.tool.model` is not `message.model`.
        assert_eq!(
            scan_response(br#"{"message":{"tool":{"model":"TOO-DEEP"}}}"#),
            None
        );
        // A `message` key whose value is not an object must not arm anything.
        assert_eq!(scan_response(br#"{"message":"hi","x":1}"#), None);
        // Leaving the message object restores root-level scanning.
        assert_eq!(
            scan_response(br#"{"message":{"id":"m"},"model":"root-wins"}"#).as_deref(),
            Some("root-wins")
        );
    }

    #[test]
    fn extracts_model_from_sse_first_chunk() {
        // The response-side model tap feeds SSE through this same scanner. `data: ` is non-structural
        // noise at depth 0, so the scanner reads the first chunk's root `model` — the provider's
        // resolved/billed id — and stops. This is what makes the billing model authoritative.
        let sse = b"data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[]}\n\n";
        assert_eq!(scan(sse).as_deref(), Some("gpt-4o-2024-08-06"));
    }

    /// Apply the injection `scan_buffered` plans and return the rewritten body (or unchanged if none),
    /// so tests assert the *resulting* JSON — the thing the upstream actually receives.
    fn inject(body: &str) -> String {
        match inject_at(body.as_bytes()) {
            Some(at) => {
                let frag = br#""stream_options":{"include_usage":true},"#;
                let mut out = Vec::with_capacity(body.len() + frag.len());
                out.extend_from_slice(&body.as_bytes()[..at]);
                out.extend_from_slice(frag);
                out.extend_from_slice(&body.as_bytes()[at..]);
                String::from_utf8(out).unwrap()
            }
            None => body.to_string(),
        }
    }

    #[test]
    fn injects_when_streaming_and_absent() {
        let out = inject(r#"{"model":"gpt-4o","stream":true,"messages":[]}"#);
        assert_eq!(
            out,
            r#"{"stream_options":{"include_usage":true},"model":"gpt-4o","stream":true,"messages":[]}"#
        );
        // The result must be valid JSON with the option set.
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            v["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn stream_can_be_the_only_or_last_key() {
        assert!(inject_at(br#"{"stream":true}"#).is_some());
        let v: serde_json::Value =
            serde_json::from_str(&inject(r#"{"model":"x","stream":true}"#)).unwrap();
        assert_eq!(
            v["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn skips_when_options_already_present() {
        // Client already asked for usage (in any form) — never touch it.
        assert_eq!(
            inject_at(br#"{"stream":true,"stream_options":{"include_usage":false}}"#),
            None
        );
        // Order-independent: options before stream.
        assert_eq!(inject_at(br#"{"stream_options":{},"stream":true}"#), None);
    }

    #[test]
    fn skips_when_not_streaming() {
        assert_eq!(inject_at(br#"{"model":"x","stream":false}"#), None);
        assert_eq!(inject_at(br#"{"model":"x"}"#), None);
    }

    #[test]
    fn ignores_nested_or_in_string_stream() {
        // `stream` inside a message object is not the root field.
        assert_eq!(
            inject_at(br#"{"messages":[{"role":"u","stream":true}],"model":"x"}"#),
            None
        );
        // `stream` mentioned inside a string value must not trigger.
        assert_eq!(
            inject_at(br#"{"system":"set stream:true please","model":"x"}"#),
            None
        );
    }

    #[test]
    fn injects_with_large_content_before_stream() {
        // Exercises the SIMD fast-skip in the planner: a large content value must be skipped, and
        // the genuine root `stream` after it still triggers injection.
        let big = "x".repeat(64 * 1024);
        let body = format!(r#"{{"messages":[{{"content":"{big}"}}],"stream":true}}"#);
        let v: serde_json::Value = serde_json::from_str(&inject(&body)).unwrap();
        assert_eq!(
            v["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn skips_word_stream_inside_large_value() {
        // The word `stream` (even `"stream"`) buried in a big string value must not trigger — the
        // memmem pre-filter passes, but the structural walk correctly skips over the string content.
        let big = "x".repeat(64 * 1024);
        let body = format!(r#"{{"system":"{big} \"stream\":true","model":"x"}}"#);
        assert_eq!(inject_at(body.as_bytes()), None);
    }

    #[test]
    fn stream_options_after_large_content_suppresses() {
        // The early-return-on-stream_options path: stream_options appearing (in any order, after a
        // big value) must suppress injection even though `stream:true` is also present.
        let big = "x".repeat(64 * 1024);
        let body = format!(
            r#"{{"content":"{big}","stream":true,"stream_options":{{"include_usage":false}}}}"#
        );
        assert_eq!(inject_at(body.as_bytes()), None);
    }

    #[test]
    fn tolerates_whitespace_and_non_objects() {
        assert!(inject_at(b"  {  \"stream\" : true }").is_some());
        assert_eq!(inject_at(b"[1,2,3]"), None);
        assert_eq!(inject_at(b"not json"), None);
    }

    #[test]
    fn simple() {
        assert_eq!(
            scan(br#"{"model":"gpt-4o","messages":[]}"#).as_deref(),
            Some("gpt-4o")
        );
    }

    #[test]
    fn model_last_after_huge_array() {
        let body = br#"{"messages":[{"role":"user","content":"...lots of text..."}],"stream":true,"model":"claude-opus-4-8"}"#;
        assert_eq!(scan(body).as_deref(), Some("claude-opus-4-8"));
    }

    #[test]
    fn nested_model_is_ignored() {
        // `"model"` inside a message object must NOT win over the real root-level one.
        let body = br#"{"messages":[{"role":"x","model":"NESTED"}],"model":"real"}"#;
        assert_eq!(scan(body).as_deref(), Some("real"));
    }

    #[test]
    fn model_word_inside_a_string_value_is_ignored() {
        let body = br#"{"system":"use the model called \"gpt\" please","model":"real"}"#;
        assert_eq!(scan(body).as_deref(), Some("real"));
    }

    #[test]
    fn whitespace_tolerant() {
        assert_eq!(
            scan(br#"{  "model" :  "m1" , "x":1 }"#).as_deref(),
            Some("m1")
        );
    }

    #[test]
    fn vendor_prefixed_value() {
        assert_eq!(
            scan(br#"{"model":"openrouter/meta-llama/llama-3.1"}"#).as_deref(),
            Some("openrouter/meta-llama/llama-3.1")
        );
    }

    #[test]
    fn split_across_feeds() {
        let mut s = ModelScanner::new();
        for part in [
            &b"{\"messages\":[],\"mod"[..],
            &b"el\":\"gp"[..],
            &b"t-4o\"}"[..],
        ] {
            s.feed(part);
        }
        assert_eq!(s.take_model().as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn absent_is_none() {
        assert_eq!(scan(br#"{"messages":[]}"#), None);
        assert_eq!(scan(b"not json"), None);
    }

    #[test]
    fn large_skipped_value_then_model() {
        // Exercises the SIMD fast-skip: a ~256KB content string (with an escaped quote) then the
        // real model. Must skip the bulk and still find the root model exactly.
        let big = "x".repeat(256 * 1024);
        let body =
            format!(r#"{{"messages":[{{"content":"{big}\"still in string"}}],"model":"gpt-4o"}}"#);
        assert_eq!(scan(body.as_bytes()).as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn nested_object_value_then_root_model() {
        // A root key whose value is an object, followed by the real model.
        let body = br#"{"response_format":{"type":"json_object"},"model":"gpt-4o"}"#;
        assert_eq!(scan(body).as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn escaped_quote_inside_model_value_does_not_terminate_it() {
        // An escaped `"` *inside the model value itself* exercises the `Cap::ModelValue` escape
        // path (line ~72): the backslash-escaped quote must be kept in the accumulated value rather
        // than ending the string early. (Model ids never really contain quotes, but a structural
        // regression here would truncate the model — and thus mislabel usage — for any value that
        // happens to contain an escape.)
        assert_eq!(
            scan(br#"{"model":"gpt-4\"o"}"#).as_deref(),
            Some("gpt-4\"o")
        );
    }

    /// Every subset of root members removed by span leaves valid JSON equal to the object without
    /// them, and an empty subset leaves the bytes identical. Same for array elements.
    #[test]
    fn span_removal_matches_value_removal_for_every_subset() {
        use serde_json::Value;
        let corpus = [
            r#"{"a":1,"b" : "x\"}y" ,"c":[1,{"d":null}], "e":{"f":[]},"g":true}"#,
            "{\n  \"only\": null\n}",
            r#" { "k\u0065y":"v", "z": -1.5e3 } "#,
        ];
        for body in corpus {
            let b = body.as_bytes();
            let members = root_members(b).expect(body);
            let spans: Vec<_> = members.iter().map(Member::span).collect();
            let want: Value = serde_json::from_str(body).unwrap();
            let keys: Vec<String> = want.as_object().unwrap().keys().cloned().collect();
            assert_eq!(members.len(), keys.len(), "{body}");
            for mask in 0u32..(1 << spans.len()) {
                let mut out = b.to_vec();
                let changed = remove_items(&mut out, &spans, |k| mask & (1 << k) != 0);
                assert_eq!(changed, mask != 0);
                if mask == 0 {
                    assert_eq!(out, b);
                }
                let got: Value = serde_json::from_slice(&out)
                    .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out)));
                let mut expect = want.clone();
                for (k, m) in members.iter().enumerate() {
                    if mask & (1 << k) != 0 {
                        let key: String =
                            serde_json::from_slice(&b[m.key.0 - 1..=m.key.1]).unwrap();
                        expect.as_object_mut().unwrap().remove(&key);
                    }
                }
                assert_eq!(got, expect, "{body} mask {mask:b}");
            }
        }
        let arr = br#"[ 1, "two" ,{"x":[3]},[],null ]"#;
        let elems = array_elements(arr, 0).unwrap();
        let all: Vec<Value> = serde_json::from_slice::<Vec<Value>>(arr).unwrap();
        for mask in 0u32..(1 << elems.len()) {
            let mut out = arr.to_vec();
            remove_items(&mut out, &elems, |k| mask & (1 << k) != 0);
            let got: Vec<Value> = serde_json::from_slice(&out).unwrap();
            let expect: Vec<Value> = all
                .iter()
                .enumerate()
                .filter(|(k, _)| mask & (1 << k) == 0)
                .map(|(_, v)| v.clone())
                .collect();
            assert_eq!(got, expect, "mask {mask:b}");
        }
        let esc = br#"{"k\u0065y":1}"#;
        assert!(root_members(esc).unwrap()[0].key_is(esc, "key"));
    }
}

/// Behaviors a mutation-testing pass found no test constraining.
#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
mod mutation_gaps {
    use super::*;

    fn inject_at(body: &[u8]) -> Option<usize> {
        scan_buffered(body).inject_at
    }

    fn scan(body: &[u8]) -> Option<String> {
        let mut s = ModelScanner::new();
        s.feed(body);
        s.take_model()
    }

    /// A root `model` whose value is not a string names no model: a string nested inside it is not
    /// the value, so it can't pick a catalog row or a billing model.
    /// claim: SEC-21
    #[test]
    fn a_non_string_root_model_names_no_model() {
        assert_eq!(scan(br#"{"model":["gpt-4o"],"messages":[]}"#), None);
        assert_eq!(scan(br#"{"model":{"id":"gpt-4o"},"messages":[]}"#), None);
        let mut s = ModelScanner::for_response();
        s.feed(br#"{"type":"message_start","message":{"model":["claude-x"],"id":"m"}}"#);
        assert_eq!(s.take_model(), None);
    }

    /// Escapes inside a skipped string (a prompt) are stepped over one byte at a time, so an
    /// escape other than `\"` never swallows the closing quote, and the root `model` after it is
    /// still found.
    /// claim: SEC-21
    #[test]
    fn escapes_in_skipped_strings_do_not_derail_the_scan() {
        let body = br#"{"messages":[{"role":"user","content":"line\nnext \"model\":\"evil\" \\ end"}],"model":"gpt-4o"}"#;
        assert_eq!(scan(body).as_deref(), Some("gpt-4o"));
        // The same bytes split at every point agree.
        for cut in 1..body.len() {
            let mut s = ModelScanner::new();
            s.feed(&body[..cut]);
            s.feed(&body[cut..]);
            assert_eq!(s.take_model().as_deref(), Some("gpt-4o"), "cut at {cut}");
        }
        // A root string whose escape is followed by its own closing quote: swallowing that quote
        // would shift every quote after it.
        assert_eq!(
            scan(br#"{"system":"a\nb","model":"gpt-4o"}"#).as_deref(),
            Some("gpt-4o")
        );
        // The buffered walks step over the same escapes.
        for streamed in [
            &br#"{"messages":[{"content":"line\nnext \"stream\":false"}],"model":"gpt-4o","stream":true}"#[..],
            &br#"{"system":"a\nb","model":"gpt-4o","stream":true}"#[..],
        ] {
            let scan = scan_buffered(streamed);
            assert_eq!(scan.model.as_deref(), Some("gpt-4o"));
            assert_eq!(scan.inject_at, Some(1));
            assert_eq!(inject_at(streamed), Some(1));
        }
    }

    /// A response's `model` after a nested object inside `message` is still the message's model:
    /// leaving that nested object does not end the message.
    /// claim: BIL-13
    #[test]
    fn a_response_model_after_a_nested_object_in_message_is_found() {
        let mut s = ModelScanner::for_response();
        s.feed(br#"{"type":"message_start","message":{"usage":{"input_tokens":1},"content":[],"model":"claude-x"}}"#);
        assert_eq!(s.take_model().as_deref(), Some("claude-x"));
    }

    /// Only a root `"stream": true` streams, and a string *value* spelled like a key is not one:
    /// another key's `true`, or a value reading `stream_options`, never changes the answer.
    /// claim: BIL-2
    #[test]
    fn only_the_root_stream_literal_decides_streaming() {
        let not_streaming = br#"{"stream":false,"store":true,"model":"m"}"#;
        assert_eq!(inject_at(not_streaming), None);
        assert_eq!(scan_buffered(not_streaming).inject_at, None);

        let value_spelled_like_a_key = br#"{"model":"stream_options","stream":true}"#;
        assert_eq!(inject_at(value_spelled_like_a_key), Some(1));
        assert_eq!(scan_buffered(value_spelled_like_a_key).inject_at, Some(1));
    }

    /// Where a client's own `stream_options` value starts, with and without whitespace around the
    /// colon — the span `force_include_usage` rewrites.
    /// claim: BIL-2
    #[test]
    fn stream_options_value_offset_is_exact() {
        for body in [
            &br#"{"stream":true,"stream_options":{"include_usage":false}}"#[..],
            &br#"{"stream":true,"stream_options" :  {"include_usage":false}}"#[..],
        ] {
            let at = scan_buffered(body)
                .stream_options_at
                .expect("streams with options");
            assert!(
                body[at..].starts_with(br#"{"include_usage":false}"#),
                "{at}"
            );
        }
    }

    /// An output limit nested in an array is no limit to clamp.
    /// claim: TRN-5
    #[test]
    fn a_nested_output_limit_is_not_spanned() {
        let scan = scan_buffered(br#"{"max_tokens":[64000],"model":"m"}"#);
        assert_eq!(scan.limit_spans[0], None);
    }

    /// A key with an escape counts as `model` only when it decodes to `model`.
    /// claim: SEC-21
    #[test]
    fn an_escaped_key_is_model_only_when_it_decodes_to_model() {
        let other = scan_buffered(br#"{"a\u0062":"x","model":"m"}"#);
        assert_eq!(other.model.as_deref(), Some("m"));
        assert!(!other.duplicate_model);
        let model = scan_buffered(br#"{"mod\u0065l":"x","model":"m"}"#);
        assert!(model.duplicate_model);
    }

    /// Only a body whose root is an object is ever rewritten: one that merely contains an object
    /// after a leading value (trailing garbage a provider rejects) gets no splice, which would
    /// otherwise land inside the leading value.
    /// claim: BIL-2
    #[test]
    fn a_body_that_does_not_open_with_an_object_is_never_planned() {
        assert_eq!(inject_at(br#"{"stream":true}"#), Some(1));
        assert_eq!(inject_at(br#"[]{"stream":true}"#), None);
        assert_eq!(inject_at(br#" 0{"stream":true}"#), None);
    }

    /// The response scanner reads raw provider bytes: a root `message` whose value was not an
    /// object arms no nesting once a `,` ends it, so a malformed body's bare object at the root is
    /// not read as the `message` object and cannot supply the billed model.
    /// claim: BIL-13
    #[test]
    fn a_comma_disarms_a_root_message_that_was_not_an_object() {
        let mut s = ModelScanner::for_response();
        s.feed(br#"{"message":"x",{"model":"not-billed"}}"#);
        assert_eq!(s.take_model(), None);
        // The armed path, for contrast.
        let mut s = ModelScanner::for_response();
        s.feed(br#"{"message":{"model":"billed"}}"#);
        assert_eq!(s.take_model().as_deref(), Some("billed"));
    }

    /// A `"type"` whose opening quote is escaped (one backslash) sits inside a string, here a key
    /// spelled `\"type`, so it is no member; three backslashes are still an odd run.
    /// claim: T3
    #[test]
    fn an_escaped_type_quote_is_not_a_type_member() {
        let hit = |b: &[u8]| has_typed_member(b, (0, b.len()), &["image"]);
        assert!(hit(br#"{"type":"image"}"#));
        assert!(!hit(br#"{"\"type":"image"}"#));
        assert!(!hit(br#"{"\\\"type":"image"}"#));
    }

    /// A value cannot be empty: a separator where a value should start ends none, so `{"a":}` and
    /// `{"a":,"b":1}` are malformed rather than members with an empty value.
    /// claim: SEC-21
    #[test]
    fn a_separator_where_a_value_starts_is_no_value() {
        for b in [&b","[..], b"}", b"]", b" "] {
            assert_eq!(value_end(b, 0), None, "{:?}", b);
        }
        assert_eq!(value_end(b"12,", 0), Some(2));
        assert_eq!(root_members(br#"{"a":}"#), None);
        assert_eq!(root_members(br#"{"a":,"b":1}"#), None);
    }

    /// A string value spelled with an escape is decoded before it is compared, as the provider's
    /// parser decodes it.
    /// claim: T3
    #[test]
    fn an_escaped_string_value_is_compared_decoded() {
        // `"image"`, spelled out so the escape survives as written.
        let quoted = format!(r#""ima{}u0067e""#, '\\');
        let b = quoted.as_bytes();
        assert!(str_is(b, (0, b.len()), "image"));
        assert!(!str_is(b, (0, b.len()), "imag"));
    }

    /// A `"type"` member counts only when its value starts inside the span: a span that ends at
    /// the key reads no value past it.
    /// claim: T3
    #[test]
    fn a_type_value_past_the_span_is_not_read() {
        let b = br#"{"type":"image"}"#;
        assert!(has_typed_member(b, (0, b.len()), &["image"]));
        assert!(!has_typed_member(b, (1, 7), &["image"]));
    }

    /// A root `stream_options` spelled with an escape decodes to `stream_options` at the provider,
    /// so the client already controls usage there too: nothing to inject. The planner the
    /// buffered scan fused said so with no offset; the buffered scan says so by flagging the key
    /// ambiguous, which the caller resolves by removing it and injecting the one that counts.
    /// claim: BIL-2
    #[test]
    fn an_escaped_stream_options_key_means_no_injection() {
        use super::tests::plan_stream_usage_injection;
        let body = format!(r#"{{"stream":true,"stream{}u005foptions":{{}}}}"#, '\\');
        assert_eq!(plan_stream_usage_injection(body.as_bytes()), None);
        assert_eq!(plan_stream_usage_injection(br#"{"stream":true}"#), Some(1));
        assert!(scan_buffered(body.as_bytes()).stream_options_ambiguous);
        assert_eq!(inject_at(br#"{"stream":true}"#), Some(1));
    }
}
