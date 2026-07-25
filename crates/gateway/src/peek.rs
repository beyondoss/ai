//! Streaming, 100%-accurate extraction of the **root-level `model`** from a JSON request body.
//!
//! Both OpenAI and Anthropic require `model` as a top-level field of the request object. We extract
//! it with a structural state machine fed the body chunks *as they stream through* — the body is
//! never buffered or reordered. This is exact (not a byte-heuristic): it tracks nesting depth and
//! string/escape state, so a `"model"` appearing inside a nested object (e.g. a message) or inside
//! a string value is correctly ignored, and field order is irrelevant. Memory is O(1): only short
//! root-level *keys* and the `model` value are accumulated. Large uninteresting string content
//! (system prompts, base64 images) is skipped with a SIMD-accelerated `memchr2` search to the next
//! `"`/`\`, not inspected byte-by-byte — so even a multi-MB request is walked cheaply.

#[derive(Clone, Copy, PartialEq, Default)]
enum Cap {
    #[default]
    No,
    Key,
    ModelValue,
}

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

    /// Whether the current depth is one whose keys we inspect: the root object always, plus the
    /// object under a root `message` key when [`Self::for_response`] enabled it.
    #[inline]
    fn at_key_level(&self) -> bool {
        (self.depth == 1 && self.root_is_object) || (self.depth == 2 && self.in_message)
    }

    /// The depth whose `:` / `,` punctuation drives `expect_key`. Mirrors [`Self::at_key_level`].
    #[inline]
    fn key_depth(&self) -> u32 {
        if self.in_message { 2 } else { 1 }
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        if self.done {
            return;
        }
        let mut i = 0;
        let n = bytes.len();
        while i < n {
            if self.in_string {
                // Fast path: the content of a string we don't accumulate (a big base64 image, a long
                // prompt, anything nested) — jump straight to the next `"` or `\` with a
                // SIMD-accelerated search instead of inspecting every byte.
                if self.cap == Cap::No && !self.escaped {
                    match memchr::memchr2(b'"', b'\\', &bytes[i..]) {
                        Some(rel) => i += rel,
                        None => return, // rest of this chunk is skippable string content
                    }
                }
                let b = bytes[i];
                i += 1;
                if self.escaped {
                    self.escaped = false;
                    if self.cap != Cap::No {
                        self.cur.push(b);
                    }
                } else if b == b'\\' {
                    self.escaped = true;
                } else if b == b'"' {
                    self.in_string = false;
                    match self.cap {
                        Cap::Key => {
                            self.last_key_is_model = self.cur == b"model";
                            // Only a *root* `message` opens the nested scan — a `message` key inside
                            // the message object itself must not re-arm it.
                            if self.accept_message_nesting && self.depth == 1 {
                                self.last_key_is_message = self.cur == b"message";
                            }
                        }
                        Cap::ModelValue => {
                            // A valid JSON string value is UTF-8; if a malformed/adversarial body
                            // smuggles non-UTF-8 bytes here we record "unknown" rather than emitting
                            // a `U+FFFD`-corrupted model into the billing log. Either way we're done.
                            self.model = Some(
                                String::from_utf8(std::mem::take(&mut self.cur))
                                    .unwrap_or_else(|_| "unknown".to_string()),
                            );
                            self.done = true;
                            return;
                        }
                        Cap::No => {}
                    }
                    self.cap = Cap::No;
                    self.cur.clear();
                } else if self.cap != Cap::No {
                    self.cur.push(b);
                }
                continue;
            }

            let b = bytes[i];
            i += 1;
            match b {
                b'"' => {
                    self.in_string = true;
                    self.cur.clear();
                    // Decide whether this string is worth accumulating — only key-level keys and
                    // the `model` value matter.
                    self.cap = if self.at_key_level() {
                        if self.expect_key {
                            Cap::Key
                        } else if self.last_key_is_model {
                            Cap::ModelValue
                        } else {
                            Cap::No
                        }
                    } else {
                        Cap::No
                    };
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
                }
                b':' if self.depth == self.key_depth() => self.expect_key = false,
                b',' if self.depth == self.key_depth() => {
                    self.expect_key = true;
                    self.last_key_is_model = false;
                    if self.depth == 1 {
                        self.last_key_is_message = false;
                    }
                }
                _ => {}
            }
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
pub fn plan_stream_usage_injection(body: &[u8]) -> Option<usize> {
    let n = body.len();
    // Cheap pre-filter: injection is only ever needed when a root-level `"stream"` key is present.
    // If the quoted token `"stream"` doesn't occur *anywhere*, the structural answer is
    // unconditionally `None`, so a single SIMD `memmem` pass lets us skip the whole walk — the
    // common case, since most requests aren't streaming. (Note `"stream_options"` does NOT contain
    // the needle: the byte after `stream` is `_`, not a closing quote — so a body carrying only
    // `stream_options` fails this pre-filter and returns `None` here, which is the correct answer.)
    memchr::memmem::find(body, b"\"stream\"")?;
    let mut i = 0;
    while i < n && body[i].is_ascii_whitespace() {
        i += 1;
    }
    // Must be a JSON object; anything else (array, scalar, garbage) we never rewrite.
    if i >= n || body[i] != b'{' {
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
    // buffer, zero-copy. (Escaped keys are sliced raw; since neither `stream` nor `stream_options`
    // contains an escape, an escaped key simply doesn't match either needle — the correct answer.)
    let mut key_start = 0usize;
    // The current root-level key is exactly `stream` (so the next literal is its value).
    let mut last_key_is_stream = false;
    let mut stream_true = false;

    let mut j = i;
    while j < n {
        if in_string {
            // Fast path: inside a string we're not capturing (any non-root-key string — message
            // content, system prompts, base64 images), jump straight to the next `"`/`\` with a
            // SIMD search instead of inspecting every byte. Mirrors the skip in `ModelScanner::feed`.
            if !capturing_key && !escaped {
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
                        if key == b"stream_options" {
                            return None;
                        }
                        last_key_is_stream = key == b"stream";
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
            b':' if depth == 1 => expect_key = false,
            b',' if depth == 1 => {
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
}

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
/// `stream_options` anywhere at root wins regardless of `stream`, an escaped key matches neither
/// needle (so it is sliced raw), and the model value *is* unescaped because `ModelScanner`
/// unescapes it. `fused_scan_matches_the_two_walks_it_replaces` cross-checks a corpus against both.
pub fn scan_buffered(body: &[u8]) -> BufferedScan {
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
    // Accumulated (unescaped) `model` value, and whether we're inside it. A `Vec` rather than a
    // slice because escapes must be resolved, exactly as `ModelScanner` does.
    let mut capturing_model = false;
    let mut model_buf: Vec<u8> = Vec::new();
    let mut model: Option<String> = None;

    let mut j = i;
    while j < n {
        if in_string {
            // SIMD skip over any string we aren't capturing — message content, system prompts,
            // base64 images. Mirrors both functions this replaces.
            if !capturing_key && !capturing_model && !escaped {
                match memchr::memchr2(b'"', b'\\', &body[j..]) {
                    Some(rel) => j += rel,
                    None => break, // rest of the body is skippable string content
                }
            }
            let b = body[j];
            if escaped {
                escaped = false;
                if capturing_model {
                    model_buf.push(b);
                }
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
                if capturing_key {
                    capturing_key = false;
                    if depth == 1 {
                        let key = &body[key_start..j];
                        // Unlike the planner we cannot return early on `stream_options`: the model
                        // may still be ahead of us. Record it and keep walking.
                        if key == b"stream_options" {
                            saw_stream_options = true;
                        }
                        last_key_is_stream = key == b"stream";
                        last_key_is_model = key == b"model";
                    }
                } else if capturing_model {
                    capturing_model = false;
                    last_key_is_model = false;
                    // Same non-UTF-8 fallback as `ModelScanner`: record "unknown" rather than emit a
                    // U+FFFD-corrupted model into the billing log.
                    model = Some(
                        String::from_utf8(std::mem::take(&mut model_buf))
                            .unwrap_or_else(|_| "unknown".to_string()),
                    );
                }
            } else if capturing_model {
                model_buf.push(b);
            }
            j += 1;
            continue;
        }
        let b = body[j];
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
            b':' if depth == 1 => expect_key = false,
            b',' if depth == 1 => {
                expect_key = true;
                last_key_is_stream = false;
                last_key_is_model = false;
            }
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

    BufferedScan {
        model,
        inject_at: (stream_true && !saw_stream_options).then_some(insert_at),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(body: &[u8]) -> Option<String> {
        let mut s = ModelScanner::new();
        s.feed(body);
        s.take_model()
    }

    fn scan_response(body: &[u8]) -> Option<String> {
        let mut s = ModelScanner::for_response();
        s.feed(body);
        s.take_model()
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

    /// Apply `plan_stream_usage_injection` and return the rewritten body (or unchanged if no plan),
    /// so tests assert the *resulting* JSON — the thing the upstream actually receives.
    fn inject(body: &str) -> String {
        match plan_stream_usage_injection(body.as_bytes()) {
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
        assert!(plan_stream_usage_injection(br#"{"stream":true}"#).is_some());
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
            plan_stream_usage_injection(
                br#"{"stream":true,"stream_options":{"include_usage":false}}"#
            ),
            None
        );
        // Order-independent: options before stream.
        assert_eq!(
            plan_stream_usage_injection(br#"{"stream_options":{},"stream":true}"#),
            None
        );
    }

    #[test]
    fn skips_when_not_streaming() {
        assert_eq!(
            plan_stream_usage_injection(br#"{"model":"x","stream":false}"#),
            None
        );
        assert_eq!(plan_stream_usage_injection(br#"{"model":"x"}"#), None);
    }

    #[test]
    fn ignores_nested_or_in_string_stream() {
        // `stream` inside a message object is not the root field.
        assert_eq!(
            plan_stream_usage_injection(
                br#"{"messages":[{"role":"u","stream":true}],"model":"x"}"#
            ),
            None
        );
        // `stream` mentioned inside a string value must not trigger.
        assert_eq!(
            plan_stream_usage_injection(br#"{"system":"set stream:true please","model":"x"}"#),
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
        assert_eq!(plan_stream_usage_injection(body.as_bytes()), None);
    }

    #[test]
    fn stream_options_after_large_content_suppresses() {
        // The early-return-on-stream_options path: stream_options appearing (in any order, after a
        // big value) must suppress injection even though `stream:true` is also present.
        let big = "x".repeat(64 * 1024);
        let body = format!(
            r#"{{"content":"{big}","stream":true,"stream_options":{{"include_usage":false}}}}"#
        );
        assert_eq!(plan_stream_usage_injection(body.as_bytes()), None);
    }

    #[test]
    fn tolerates_whitespace_and_non_objects() {
        assert!(plan_stream_usage_injection(b"  {  \"stream\" : true }").is_some());
        assert_eq!(plan_stream_usage_injection(b"[1,2,3]"), None);
        assert_eq!(plan_stream_usage_injection(b"not json"), None);
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
}
