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
    /// Whether the next root-level string is a key (`{`/`,` → key; `:` → value).
    expect_key: bool,
    /// The most recent root-level key was exactly `model`.
    last_key_is_model: bool,
    /// What (if anything) we're accumulating into `cur` for the current string.
    cap: Cap,
    cur: Vec<u8>,
}

impl ModelScanner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the extracted model, if found. (Available as soon as the value is seen.)
    pub fn take_model(&mut self) -> Option<String> {
        self.model.take()
    }

    #[inline]
    fn at_root_object(&self) -> bool {
        self.depth == 1 && self.root_is_object
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
                        Cap::Key => self.last_key_is_model = self.cur == b"model",
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
                    // Decide whether this string is worth accumulating — only root-object keys and
                    // the `model` value matter.
                    self.cap = if self.at_root_object() {
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
                    }
                    self.depth += 1;
                }
                b'[' => {
                    if self.depth == 0 {
                        self.root_is_object = false;
                    }
                    self.depth += 1;
                }
                b'}' | b']' => self.depth = self.depth.saturating_sub(1),
                b':' if self.depth == 1 => self.expect_key = false,
                b',' if self.depth == 1 => {
                    self.expect_key = true;
                    self.last_key_is_model = false;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(body: &[u8]) -> Option<String> {
        let mut s = ModelScanner::new();
        s.feed(body);
        s.take_model()
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
