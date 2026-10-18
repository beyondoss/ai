//! `peek`'s structural walkers as they stood before they were ported onto `lex`, kept verbatim as
//! the behavioural oracle of the differential properties in `props_lex.rs`: the ported scanners
//! must answer exactly as these do on any bytes, malformed ones included, at any chunking. The
//! semantic oracle is `serde_json`; this one pins what JSON alone does not decide.
//!
//! Do not fix anything here. A deliberate change of behaviour changes both sides in one commit.
#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

/// Most bytes of a root-level key or `model` value we will hold. Real model ids are well under
/// 128 bytes (`proxy::sanitize_model` records anything longer as `unknown`), and the only keys we
/// compare against are `model` and `message`, so anything past this is junk — and without a bound a
/// managed request carrying a multi-megabyte `model` string would be materialized in full just to be
/// discarded. A capped value is a prefix: longer than any catalog name or loggable id, so it can only
/// ever miss the catalog or log as `unknown`, never match something it isn't.
pub const MAX_CAPTURE: usize = 256;

#[inline]
fn push_capped(buf: &mut Vec<u8>, b: u8) {
    if buf.len() < MAX_CAPTURE {
        buf.push(b);
    }
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
    let mut quoted = Vec::with_capacity(raw.len().saturating_add(2));
    quoted.push(b'"');
    quoted.extend(raw);
    quoted.push(b'"');
    serde_json::from_slice::<String>(&quoted).ok()
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
    /// Nesting depth: number of currently-open `{`/`[`. Root object contents are at depth 1.
    depth: u32,
    root_is_object: bool,
    in_string: bool,
    escaped: bool,
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
    /// The capture in `cur` holds a backslash, so it is decoded when it closes.
    cur_escaped: bool,
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

    /// Whether the current depth is one whose keys we inspect: the root object always, plus the
    /// object under a root `message` key when [`Self::for_response`] enabled it.
    #[inline]
    fn at_key_level(&self) -> bool {
        (self.depth == 1 && self.root_is_object) || (self.depth == 2 && self.in_message)
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

    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
    fn feed_inner(&mut self, bytes: &[u8], mut kept: Option<&mut Kept>) {
        // Proof for the allow: every index is `i`/`j` with `i < n` (the loop condition, or a `memchr`
        // offset inside `bytes[i..]`), or a key span bounded by the closing quote just found; counters
        // (`i`, `j`, `depth`, `model_keys`) count bytes of a body that fits in memory, so they cannot
        // overflow. This loop runs on every managed request body, so it keeps the unchecked forms.
        if self.done {
            return;
        }
        let mut i = 0;
        let n = bytes.len();
        // Where the raw capture in progress starts in this chunk.
        let mut raw_from = kept.as_ref().and_then(|k| k.raw).map(|_| 0);
        while i < n {
            if self.in_string {
                // Fast path: the content of a string we don't accumulate (a big base64 image, a long
                // prompt, anything nested) — jump straight to the next `"` or `\` with a
                // SIMD-accelerated search instead of inspecting every byte.
                if self.cap == Cap::No && !self.escaped {
                    match memchr::memchr2(b'"', b'\\', &bytes[i..]) {
                        Some(rel) => i += rel,
                        // The rest of this chunk is skippable string content.
                        None => break,
                    }
                }
                let b = bytes[i];
                i += 1;
                if self.escaped {
                    self.escaped = false;
                    if self.cap != Cap::No {
                        push_capped(&mut self.cur, b);
                    }
                } else if b == b'\\' {
                    self.escaped = true;
                    if self.cap != Cap::No {
                        push_capped(&mut self.cur, b);
                        self.cur_escaped = true;
                    }
                } else if b == b'"' {
                    self.in_string = false;
                    match self.cap {
                        Cap::Key => {
                            let key = key_bytes(&self.cur, self.cur_escaped);
                            self.last_key_is_model = key.as_deref() == Some(b"model".as_slice());
                            if let Some(k) = kept.as_deref_mut() {
                                let at = key
                                    .as_deref()
                                    .and_then(|key| k.keys.iter().position(|want| *want == key));
                                k.last_key_extra = at.and_then(|i| u8::try_from(i).ok());
                            }
                            // Only a *root* `message` opens the nested scan — a `message` key inside
                            // the message object itself must not re-arm it.
                            if self.accept_message_nesting && self.depth == 1 {
                                self.last_key_is_message =
                                    matches!(key.as_deref(), Some(b"message" | b"response"));
                            }
                        }
                        Cap::ModelValue => {
                            // A valid JSON string value is UTF-8; if a malformed/adversarial body
                            // smuggles non-UTF-8 bytes here we record "unknown" rather than emitting
                            // a `U+FFFD`-corrupted model into the billing log. Either way we're done.
                            let v = capture_text(std::mem::take(&mut self.cur), self.cur_escaped)
                                .unwrap_or_else(|| "unknown".to_string());
                            match kept.as_deref() {
                                // A knob scan keeps walking; the first `model` is the one kept.
                                Some(k) if k.past_model => {
                                    self.model.get_or_insert(v);
                                    if k.all_found() {
                                        self.done = true;
                                        return;
                                    }
                                }
                                _ => {
                                    self.model = Some(v);
                                    self.done = true;
                                    return;
                                }
                            }
                        }
                        Cap::Extra => {
                            if let Some(k) = kept.as_deref_mut()
                                && let Some(idx) = k.last_key_extra
                            {
                                // Not UTF-8: dropped, never written to a row as a corrupted value.
                                if let Some(v) =
                                    capture_text(std::mem::take(&mut self.cur), self.cur_escaped)
                                {
                                    k.keep(idx, v);
                                }
                                if k.past_model && k.all_found() {
                                    self.done = true;
                                    return;
                                }
                            }
                        }
                        Cap::No => {}
                    }
                    self.cap = Cap::No;
                    self.cur.clear();
                } else if self.cap != Cap::No {
                    push_capped(&mut self.cur, b);
                }
                continue;
            }

            let b = bytes[i];
            i += 1;
            let extra = kept.as_ref().and_then(|k| k.last_key_extra);
            match b {
                b'"' => {
                    self.in_string = true;
                    self.cur.clear();
                    self.cur_escaped = false;
                    // Decide whether this string is worth accumulating — only key-level keys and
                    // the `model` value matter.
                    self.cap = if self.at_key_level() {
                        if self.expect_key {
                            Cap::Key
                        } else if self.last_key_is_model {
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
                b'{' | b'['
                    if extra.is_some()
                        && !self.expect_key
                        && self.at_key_level()
                        && kept.as_ref().is_some_and(|k| k.raw.is_none()) =>
                {
                    // An object or array value of a kept key: captured raw until its depth closes.
                    if let Some(k) = kept.as_deref_mut() {
                        k.raw = extra;
                        k.raw_depth = self.depth;
                        k.raw_buf.clear();
                    }
                    raw_from = Some(i - 1);
                    self.depth += 1;
                }
                b'{' => {
                    if self.depth == 0 {
                        self.root_is_object = true;
                        self.expect_key = true;
                    } else if self.depth == 1 && self.last_key_is_message {
                        // Descending into the root `message` object: its keys become key-level too.
                        self.in_message = true;
                        self.expect_key = true;
                    }
                    self.depth += 1;
                }
                b'[' => {
                    if self.depth == 0 {
                        self.root_is_object = false;
                    }
                    self.depth += 1;
                }
                b'}' | b']' => {
                    // Leaving the `message` object returns key-level scanning to the root.
                    if self.depth == 2 && self.in_message {
                        self.in_message = false;
                        self.last_key_is_message = false;
                    }
                    self.depth = self.depth.saturating_sub(1);
                    if let Some(k) = kept.as_deref_mut()
                        && let Some(idx) = k.raw
                        && self.depth == k.raw_depth
                    {
                        push_raw(&mut k.raw_buf, &bytes[raw_from.unwrap_or(0)..i]);
                        raw_from = None;
                        k.raw = None;
                        let v = utf8_prefix(std::mem::take(&mut k.raw_buf));
                        k.keep(idx, v);
                        if k.past_model && k.all_found() {
                            self.done = true;
                            return;
                        }
                    }
                }
                // Unguarded by depth: a `:` or `,` inside a nested value is always followed by
                // the key-level `,` or `}` that closes it before the next key-level string, and
                // that one sets `expect_key` again.
                b':' => self.expect_key = false,
                b',' => {
                    self.expect_key = true;
                    self.last_key_is_model = false;
                    if let Some(k) = kept.as_deref_mut() {
                        k.last_key_extra = None;
                    }
                    if self.depth == 1 {
                        self.last_key_is_message = false;
                    }
                }
                _ => {}
            }
        }
        if let Some(k) = kept
            && k.raw.is_some()
            && let Some(from) = raw_from
        {
            push_raw(&mut k.raw_buf, &bytes[from..n]);
        }
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

/// Decide whether an OpenAI **chat** request body needs `stream_options.include_usage` injected,
/// and where. Returns `Some(offset)` — the byte index just after the root object's opening `{`, where
/// the caller splices `"stream_options":{"include_usage":true},` — **only** when the body is a JSON
/// object with a root-level `"stream": true` and **no** root-level `"stream_options"` key. Otherwise
/// `None` (not a stream, options already set, or not an object) → forward unchanged.
///
/// Why this exists: OpenAI only emits a usage chunk on a stream when the request carries
/// `stream_options.include_usage = true`. A stock client that omits it would stream with no usage,
/// so managed traffic couldn't be metered. We can't ask for it via a header and can't set it in a
/// client SDK we don't control, so the gateway injects it — for every OpenAI streaming chat request,
/// out of the box.
///
/// Structural (depth + string + escape aware), so a `"stream"` inside a message object or inside a
/// string value never triggers injection — only the genuine root-level field. The returned offset is
/// always inside a non-empty object (a root `"stream"` is present), so the caller always follows the
/// fragment with a comma.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
pub fn plan_stream_usage_injection(body: &[u8]) -> Option<usize> {
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
                            || (escaped_key && escaped_key_is(&body[key_start - 1..=j], "stream"));
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

/// Both answers the injection path needs from a **fully buffered** request body.
pub struct BufferedScan {
    /// The root-level `model`, as [`ModelScanner`] would have extracted it.
    pub model: Option<String>,
    /// Where to splice the `stream_options` fragment, as [`plan_stream_usage_injection`] would have
    /// reported it.
    pub inject_at: Option<usize>,
    /// Where the value of a root `stream_options` begins, when the body streams **and** already
    /// carries one. The client may have turned usage off (`include_usage: false`) or left it out
    /// (`{}`); either way OpenAI would send no usage chunk, so the caller rewrites that value (see
    /// `proxy::force_include_usage`). `None` whenever `inject_at` is `Some`, and for a non-stream.
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
/// decision needs the whole body), and it was then walked **twice**: once chunk-by-chunk by
/// `ModelScanner` looking for a root `model`, then again end-to-end by
/// [`plan_stream_usage_injection`] looking for root `stream`/`stream_options`. Same traversal, same
/// bytes, same depth/string/escape bookkeeping — just different needles. Measured 16.3 µs vs 4.4 µs
/// on a 512 KiB body.
///
/// Only for the buffered path. A request that streams through un-buffered still uses
/// `ModelScanner`, which is incremental and cannot be replaced by this.
///
/// Semantics are exactly the two functions it replaces, including the details that look incidental:
/// `stream_options` anywhere at root wins regardless of `stream`, a key spelled with escapes is
/// decoded and the last `stream` decides (as the provider's parser does; a duplicate or escaped
/// `stream_options` is flagged in `stream_options_ambiguous` instead, D88), and the model value
/// *is* unescaped because `ModelScanner` unescapes it. `fused_scan_matches_the_two_walks_it_replaces` cross-checks a corpus against both.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
pub fn scan_buffered(body: &[u8]) -> BufferedScan {
    // Proof for the allow: every index is `i`/`j` with `i < n` (the loop condition, or a `memchr`
    // offset inside `bytes[i..]`), or a key span bounded by the closing quote just found; counters
    // (`i`, `j`, `depth`, `model_keys`) count bytes of a body that fits in memory, so they cannot
    // overflow. This loop runs on every managed request body, so it keeps the unchecked forms.
    let n = body.len();
    let mut i = 0;
    while i < n && body[i].is_ascii_whitespace() {
        i += 1;
    }
    // Not a JSON object ⇒ nothing to inject into. A non-object root also has no root-level `model`,
    // so both answers are `None` and there is nothing to walk for.
    if i >= n || body[i] != b'{' {
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
    }
    let insert_at = i + 1;

    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;
    let mut expect_key = false;
    let mut capturing_key = false;
    let mut key_start = 0usize;
    let mut last_key_is_stream = false;
    let mut last_key_is_model = false;
    let mut stream_true = false;
    let mut saw_stream_options = false;
    let mut stream_options_ambiguous = false;
    // The closing quote of the root `stream_options` key; its value follows the next `:`.
    let mut stream_options_key_end = 0usize;
    // Accumulated (unescaped) `model` value, and whether we're inside it. A `Vec` rather than a
    // slice because escapes must be resolved, exactly as `ModelScanner` does.
    let mut capturing_model = false;
    let mut model_buf: Vec<u8> = Vec::new();
    let mut model_escaped = false;
    let mut model: Option<String> = None;
    // Where the value's raw bytes begin (just past the opening quote), and the finished span.
    let mut model_start = 0usize;
    let mut model_span: Option<(usize, usize)> = None;
    let mut model_keys = 0u32;
    // The output limit whose value comes next, as an index into `OUTPUT_LIMIT_KEYS`.
    let mut limit_key: Option<usize> = None;
    let mut limit_spans: [Option<(usize, usize)>; 3] = [None; 3];
    let mut inexact_limit = false;
    // The byte after a limit key's colon is still to come.
    let mut limit_value_next = false;
    let mut limit_keys: [Option<usize>; 3] = [None; 3];

    let mut j = i;
    while j < n {
        if in_string {
            // SIMD skip over any string whose bytes we don't accumulate — message content, system
            // prompts, base64 images, and keys too, which are sliced out of the body at their
            // closing quote.
            if !capturing_model && !escaped {
                match memchr::memchr2(b'"', b'\\', &body[j..]) {
                    Some(rel) => j += rel,
                    None => break, // rest of the body is skippable string content
                }
            }
            let b = body[j];
            if escaped {
                escaped = false;
                if capturing_model {
                    push_capped(&mut model_buf, b);
                }
            } else if b == b'\\' {
                escaped = true;
                if capturing_model {
                    push_capped(&mut model_buf, b);
                    model_escaped = true;
                }
            } else if b == b'"' {
                in_string = false;
                if capturing_key {
                    capturing_key = false;
                    if depth == 1 {
                        let key = &body[key_start..j];
                        // Unlike the planner we cannot return early on `stream_options`: the model
                        // may still be ahead of us. Record it and keep walking.
                        if key == b"stream_options" {
                            stream_options_ambiguous |= saw_stream_options;
                            if !saw_stream_options {
                                saw_stream_options = true;
                                stream_options_key_end = j;
                            }
                        } else if key.contains(&b'\\')
                            && escaped_key_is(&body[key_start - 1..=j], "stream_options")
                        {
                            stream_options_ambiguous = true;
                        }
                        // The provider decodes an escaped key and keeps the last duplicate,
                        // so each key that decodes to `stream` decides afresh (D88).
                        last_key_is_stream = key == b"stream"
                            || (key.contains(&b'\\')
                                && escaped_key_is(&body[key_start - 1..=j], "stream"));
                        if last_key_is_stream {
                            stream_true = false;
                        }
                        last_key_is_model = key == b"model"
                            || (key.contains(&b'\\')
                                && escaped_key_is(&body[key_start - 1..=j], "model"));
                        model_keys += u32::from(last_key_is_model);
                        // A key spelled with escapes names the limit it decodes to (`"max\u005ftokens"`).
                        let quoted = &body[key_start - 1..=j];
                        limit_key = OUTPUT_LIMIT_KEYS.iter().position(|k| {
                            *k == key
                                || (key.contains(&b'\\')
                                    && std::str::from_utf8(k)
                                        .is_ok_and(|want| escaped_key_is(quoted, want)))
                        });
                        if let Some(k) = limit_key {
                            // A second root spelling of the same limit: a parser keeps one of them, and
                            // only the first is spanned, so it cannot be capped reliably. Refused.
                            if limit_keys[k].is_some() {
                                inexact_limit = true;
                            }
                            limit_keys[k].get_or_insert(key_start - 1);
                            limit_value_next = true;
                        }
                    }
                } else if capturing_model {
                    capturing_model = false;
                    last_key_is_model = false;
                    // `j` is the closing quote, `model_start` the byte after the opening one, so the
                    // span covers exactly the value and neither quote.
                    model_span = Some((model_start, j));
                    // Same non-UTF-8 fallback as `ModelScanner`: record "unknown" rather than emit a
                    // U+FFFD-corrupted model into the billing log.
                    model = Some(
                        capture_text(std::mem::take(&mut model_buf), model_escaped)
                            .unwrap_or_else(|| "unknown".to_string()),
                    );
                }
            } else if capturing_model {
                push_capped(&mut model_buf, b);
            }
            j += 1;
            continue;
        }
        let b = body[j];
        // The first byte of a limit's value decides it: a digit starts a plain integer, the literal
        // `null` is no value, and whitespace or the colon before the value is skipped. Anything else
        // (an `n` that is not `null`, a string, a fraction) is inexact.
        if limit_value_next && !(b.is_ascii_whitespace() || b == b':') {
            limit_value_next = false;
            if !(b.is_ascii_digit() || body[j..].starts_with(b"null")) {
                inexact_limit = true;
            }
        }
        match b {
            b'"' => {
                capturing_key = false;
                capturing_model = false;
                if depth == 1 && expect_key {
                    capturing_key = true;
                    key_start = j + 1;
                } else if depth == 1 && last_key_is_model && model.is_none() {
                    capturing_model = true;
                    model_buf.clear();
                    model_escaped = false;
                    model_start = j + 1;
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
            // Unguarded by depth, as in both walks this one replaces: a nested `:` is always
            // followed by the root `,` or `}` that closes its value before the next root key.
            b':' => expect_key = false,
            b',' if depth == 1 => {
                expect_key = true;
                last_key_is_stream = false;
                last_key_is_model = false;
                limit_key = None;
            }
            b'0'..=b'9' if depth == 1 && !expect_key && limit_key.is_some() => {
                let end = body[j..]
                    .iter()
                    .position(|c| !c.is_ascii_digit())
                    .map_or(n, |rel| j + rel);
                // A plain integer ends its member: `,` or `}`, past any whitespace. A fraction or an
                // exponent ends in `.`, `e` or `E` instead, and is no integer.
                let ends_member = body[end..]
                    .iter()
                    .find(|c| !c.is_ascii_whitespace())
                    .is_some_and(|c| matches!(c, b',' | b'}'));
                if let Some(k) = limit_key.take() {
                    if !ends_member {
                        inexact_limit = true;
                    } else if limit_spans[k].is_none() {
                        limit_spans[k] = Some((j, end));
                    }
                }
                j = end;
                continue;
            }
            // A negative limit is no limit to cap. (A string or structured value never puts a digit
            // at depth 1 before the `,` that ends it, so a nested `-` may clear it too.)
            b'-' => limit_key = None,
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

    // The `stream_options` value: past the key's closing quote, whitespace, `:`, whitespace.
    let stream_options_at = (stream_true && saw_stream_options)
        .then(|| {
            let rest = &body[stream_options_key_end + 1..];
            let colon = rest.iter().position(|b| !b.is_ascii_whitespace())?;
            (rest[colon] == b':').then_some(())?;
            let value = rest[colon + 1..]
                .iter()
                .position(|b| !b.is_ascii_whitespace())?;
            Some(stream_options_key_end + 1 + colon + 1 + value)
        })
        .flatten();
    BufferedScan {
        model,
        inject_at: (stream_true && !saw_stream_options).then_some(insert_at),
        stream_options_at,
        model_span,
        duplicate_model: model_keys > 1,
        limit_spans,
        inexact_limit,
        limit_keys,
        stream_options_ambiguous,
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
    // A decoded byte takes at most six raw ones (`\u00XX`), plus the two quotes: a longer string
    // cannot decode to `want`, and is not decoded (it may be a whole prompt).
    let most = want.len().saturating_mul(6).saturating_add(2);
    quoted.len() <= most && serde_json::from_slice::<String>(quoted).is_ok_and(|k| k == want)
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

/// The end (exclusive) of the string whose opening quote is at `open`.
fn string_end(b: &[u8], open: usize) -> Option<usize> {
    let mut i = open.saturating_add(1);
    loop {
        let at = i.saturating_add(memchr::memchr2(b'"', b'\\', b.get(i..)?)?);
        if b.get(at) == Some(&b'\\') {
            i = at.saturating_add(2);
        } else {
            return Some(at.saturating_add(1));
        }
    }
}

/// The end (exclusive) of the JSON value starting at `i` (no leading whitespace).
pub fn value_end(b: &[u8], i: usize) -> Option<usize> {
    match *b.get(i)? {
        b'"' => string_end(b, i),
        b'{' | b'[' => {
            let mut depth = 0u32;
            let mut j = i;
            while let Some(&c) = b.get(j) {
                match c {
                    b'"' => {
                        j = string_end(b, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth = depth.saturating_add(1),
                    b'}' | b']' => {
                        // The first byte opened a container, so `depth` is at least 1 here.
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            return Some(j.saturating_add(1));
                        }
                    }
                    _ => {}
                }
                j = j.saturating_add(1);
            }
            None
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
fn item_cuts(
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
        quoted @ [b'"', .., b'"'] => serde_json::from_slice::<String>(quoted)
            .ok()
            .map(std::borrow::Cow::Owned),
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
    let mut s = Structure::default();
    let mut i = 0usize;
    while let Some(rest) = b.get(i..) {
        let q = memchr::memchr(b'"', rest).map_or(b.len(), |q| i.saturating_add(q));
        for (at, &c) in b.get(i..q).unwrap_or_default().iter().enumerate() {
            if !matches!(c, b'{' | b'[' | b',' | b':') {
                continue;
            }
            s.tokens = s.tokens.saturating_add(1);
            // An object is empty when only whitespace stands between its braces.
            let next = skip_ws(b, i.saturating_add(at).saturating_add(1));
            if c == b'{' && b.get(next) != Some(&b'}') {
                s.objects = s.objects.saturating_add(1);
            }
        }
        match string_end(b, q) {
            Some(end) => i = end,
            None => break,
        }
    }
    s
}
