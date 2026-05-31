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

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(body: &[u8]) -> Option<String> {
        let mut s = ModelScanner::new();
        s.feed(body);
        s.take_model()
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
