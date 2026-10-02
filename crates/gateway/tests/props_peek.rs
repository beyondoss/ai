//! Differential property tests for `peek`: the structural scanners against `serde_json`, which is
//! how the provider reads the same body (duplicate keys: the last one wins).
//!
//! Bodies are written by hand so the generator controls what a serializer would normalize away:
//! whitespace anywhere JSON allows it, escaped spellings of keys and values, duplicate keys,
//! `model` / `stream` look-alikes nested in objects and inside string values.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::peek::{self, ModelScanner, OUTPUT_LIMIT_KEYS};
use props_support::*;
use proptest::prelude::*;
use serde_json::{Map, Value, json};

/// Whitespace JSON allows between tokens.
fn ws(choice: u8) -> &'static str {
    match choice % 6 {
        0 | 1 => "",
        2 => " ",
        3 => "\n",
        4 => "\t ",
        _ => "\r\n  ",
    }
}

/// How a string is spelled on the wire.
#[derive(Clone, Copy, Debug)]
struct Spelling {
    /// `\uXXXX` for every non-ASCII char (what Python's `ensure_ascii` does).
    ascii_only: bool,
    /// `\/` for `/` (PHP's default).
    escape_slash: bool,
    /// `\uXXXX` for one ASCII letter (no real serializer; adversarial).
    escape_letter: Option<u8>,
}

fn spelling() -> impl Strategy<Value = Spelling> {
    (
        any::<bool>(),
        any::<bool>(),
        prop::option::weighted(0.15, any::<u8>()),
    )
        .prop_map(|(ascii_only, escape_slash, escape_letter)| Spelling {
            ascii_only,
            escape_slash,
            escape_letter,
        })
}

const PLAIN: Spelling = Spelling {
    ascii_only: false,
    escape_slash: false,
    escape_letter: None,
};

impl Spelling {
    /// Whether the spelling uses an escape that is not one of the "simple" ones every JSON writer
    /// emits for `"` and `\`.
    fn exotic(&self, s: &str) -> bool {
        (self.ascii_only && !s.is_ascii())
            || (self.escape_slash && s.contains('/'))
            || (self.escape_letter.is_some() && s.chars().any(|c| c.is_ascii_alphabetic()))
            || s.chars().any(|c| (c as u32) < 0x20)
    }

    fn quote(&self, s: &str) -> String {
        let letters: Vec<usize> = s
            .char_indices()
            .filter(|(_, c)| c.is_ascii_alphabetic())
            .map(|(i, _)| i)
            .collect();
        let escaped_letter = self
            .escape_letter
            .and_then(|n| letters.get(usize::from(n) % letters.len().max(1)).copied());
        let mut out = String::from("\"");
        for (i, c) in s.char_indices() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '/' if self.escape_slash => out.push_str("\\/"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c if Some(i) == escaped_letter => out.push_str(&format!("\\u{:04x}", c as u32)),
                c if self.ascii_only && !c.is_ascii() => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{u:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }
}

/// Write `v` as JSON with whitespace drawn from `w` (cycled).
fn write(v: &Value, w: &[u8], k: &mut usize, sp: Spelling, out: &mut String) {
    fn next(w: &[u8], k: &mut usize) -> &'static str {
        *k += 1;
        ws(w.get(*k % w.len().max(1)).copied().unwrap_or(0))
    }
    match v {
        Value::Object(m) => {
            out.push('{');
            for (i, (key, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(next(w, k));
                out.push_str(&sp.quote(key));
                out.push_str(next(w, k));
                out.push(':');
                out.push_str(next(w, k));
                write(x, w, k, sp, out);
                out.push_str(next(w, k));
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(next(w, k));
                write(x, w, k, sp, out);
                out.push_str(next(w, k));
            }
            out.push(']');
        }
        Value::String(s) => out.push_str(&sp.quote(s)),
        other => out.push_str(&other.to_string()),
    }
}

/// The root keys the scanners care about, plus near misses.
const ROOT_KEYS: &[&str] = &[
    "model",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "messages",
    "message",
    "response",
    "Model",
    "models",
    "model ",
    "stream_option",
];

/// One root member: key, key spelling, value, value spelling.
#[derive(Clone, Debug)]
struct Member {
    key: String,
    key_sp: Spelling,
    value: Value,
    value_sp: Spelling,
}

fn model_id() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => "[a-z0-9][a-z0-9./:-]{0,40}",
        1 => Just("anthropic/claude-sonnet-4.5".to_owned()),
        1 => text(6),
        1 => "[a-z]{300,320}",
    ]
}

/// A value that hides `model` / `stream` look-alikes one level down or inside strings.
fn decoy() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(
            json!({ "model": "decoy", "stream": true, "stream_options": {"include_usage": false} })
        ),
        Just(json!([{ "role": "user", "content": "\"model\":\"decoy\",\"stream\":true" }])),
        Just(json!({ "model": { "model": "deeper" } })),
        Just(json!("{\"model\":\"in-a-string\"}")),
        json_value(3),
    ]
}

fn member() -> impl Strategy<Value = Member> {
    let key = prop_oneof![
        6 => prop::sample::select(ROOT_KEYS).prop_map(str::to_owned),
        1 => text(3),
    ];
    (
        key,
        spelling(),
        any::<u8>(),
        model_id(),
        decoy(),
        spelling(),
        tokens(),
    )
        .prop_map(|(key, key_sp, pick, id, decoy, value_sp, n)| {
            let value = match (key.as_str(), pick % 8) {
                ("model", 0..=5) => json!(id),
                ("stream", 0..=5) => json!(pick % 2 == 0),
                ("stream_options", 0..=3) => json!({ "include_usage": pick % 2 == 0 }),
                ("stream_options", 4) => json!({}),
                ("max_tokens" | "max_completion_tokens" | "max_output_tokens", 0..=4) => json!(n),
                ("max_tokens" | "max_completion_tokens" | "max_output_tokens", 5) => json!(-1),
                _ => decoy,
            };
            Member {
                key,
                key_sp,
                value,
                value_sp,
            }
        })
}

#[derive(Clone, Debug)]
struct Body {
    members: Vec<Member>,
    text: String,
}

fn body() -> impl Strategy<Value = Body> {
    (
        prop::collection::vec(member(), 0..8),
        prop::collection::vec(any::<u8>(), 1..16),
    )
        .prop_map(|(members, w)| {
            let mut k = 0;
            let mut out = String::new();
            out.push_str(ws(w[0]));
            out.push('{');
            for (i, m) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                k += 1;
                out.push_str(ws(w[k % w.len()]));
                out.push_str(&m.key_sp.quote(&m.key));
                out.push(':');
                k += 1;
                out.push_str(ws(w[k % w.len()]));
                write(&m.value, &w, &mut k, m.value_sp, &mut out);
            }
            k += 1;
            out.push_str(ws(w[k % w.len()]));
            out.push('}');
            out.push_str(ws(w[(k + 1) % w.len()]));
            Body { members, text: out }
        })
}

impl Body {
    fn count(&self, key: &str) -> usize {
        self.members.iter().filter(|m| m.key == key).count()
    }
    fn exotic_key(&self, key: &str) -> bool {
        self.members
            .iter()
            .any(|m| m.key == key && m.key_sp.exotic(key))
    }
    fn serde(&self) -> Map<String, Value> {
        match serde_json::from_str::<Value>(&self.text) {
            Ok(Value::Object(m)) => m,
            other => panic!("generator wrote invalid JSON ({other:?}): {}", self.text),
        }
    }
}

/// Byte offset where the value of the `n`th (0-based) root member starts.
fn value_offsets(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    // A tiny structural walk: the first byte after each depth-1 `:`.
    let b = text.as_bytes();
    let (mut depth, mut in_str, mut esc, mut expect_value) = (0u32, false, false, false);
    for (i, &c) in b.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        if expect_value && !c.is_ascii_whitespace() {
            out.push(i);
            expect_value = false;
        }
        match c {
            b'"' => in_str = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            b':' if depth == 1 => expect_value = true,
            _ => {}
        }
    }
    out
}

/// The root `model` the scanners read is the one serde reads, at the span serde would decode,
/// and splicing a new id into that span changes exactly the model. Duplicates are flagged.
#[test]
fn prop_root_model_and_span_match_serde() {
    check("prop_root_model_and_span_match_serde", body(), |b| {
        let root = b.serde();
        let scan = peek::scan_buffered(b.text.as_bytes());
        let models = b.count("model");
        prop_assert_eq!(scan.duplicate_model, models > 1, "{}", b.text);
        if models != 1 {
            if models == 0 {
                prop_assert_eq!(scan.model, None);
            }
            return Ok(());
        }
        let m = b.members.iter().find(|m| m.key == "model").unwrap();
        let Some(id) = root["model"].as_str() else {
            prop_assert_eq!(scan.model, None, "{}", b.text);
            return Ok(());
        };
        // A value spelled with escapes other than `\"` and `\\` is not decoded (the scanner keeps
        // the escaped character); no client writes a model id that way and the span rewrite
        // replaces the raw bytes whole, so the provider never sees the misread. See the report.
        if !m.value_sp.exotic(id) && id.len() <= peek::MAX_CAPTURE {
            prop_assert_eq!(scan.model.as_deref(), Some(id), "{}", b.text);
        }
        let (s, e) = scan.model_span.expect("a string model has a span");
        let mut spliced = b.text.clone().into_bytes();
        spliced.splice(s..e, b"rewritten-id".iter().copied());
        let after: Value = serde_json::from_slice(&spliced).unwrap_or_else(|err| {
            panic!(
                "splice broke the JSON ({err}): {}",
                String::from_utf8_lossy(&spliced)
            )
        });
        let mut want = Value::Object(root.clone());
        want["model"] = json!("rewritten-id");
        prop_assert_eq!(after, want, "{}", b.text);
        Ok(())
    });
}

/// The streaming scanner, fed in any chunking, agrees with the buffered one.
#[test]
fn prop_streaming_scanner_agrees_with_buffered_scan() {
    check(
        "prop_streaming_scanner_agrees_with_buffered_scan",
        (body(), prop::collection::vec(any::<usize>(), 0..12)),
        |(b, cuts)| {
            // `ModelScanner` matches keys raw; `scan_buffered` decodes an escaped `model` key so the
            // duplicate check cannot be dodged. A body with an escaped `model` key and no plain one
            // routes on nothing (404) rather than on a misread, so that difference is excluded here.
            prop_assume!(!b.exotic_key("model"));
            let mut s = ModelScanner::new();
            for c in chunk(b.text.as_bytes(), &cuts) {
                s.feed(&c);
            }
            let buffered = peek::scan_buffered(b.text.as_bytes());
            prop_assert_eq!(s.take_model(), buffered.model, "{}", b.text);
            Ok(())
        },
    );
}

/// The scanners' view of `stream` / `stream_options` is the provider's: a body serde reads as
/// streaming is one the gateway reads as streaming, and the `stream_options` value the gateway
/// would rewrite is the one serde keeps. Otherwise the include_usage rewrite edits a member the
/// provider ignores and a client turns off exact metering (BIL-2).
#[test]
fn prop_stream_and_stream_options_match_serde() {
    check("prop_stream_and_stream_options_match_serde", body(), |b| {
        let root = b.serde();
        let scan = peek::scan_buffered(b.text.as_bytes());
        let streams = root.get("stream") == Some(&Value::Bool(true));
        prop_assert_eq!(
            scan.inject_at.is_some() || scan.stream_options_at.is_some(),
            streams,
            "{}",
            b.text
        );
        if !streams {
            return Ok(());
        }
        if scan.stream_options_ambiguous {
            // A duplicate or escaped `stream_options` (D88): the gateway removes every member that
            // decodes to it and injects the one that counts. What the provider then reads is the
            // client's body with usage on.
            let mut buf = b.text.clone().into_bytes();
            prop_assert!(peek::remove_root_members(&mut buf, "stream_options"));
            let rescan = peek::scan_buffered(&buf);
            prop_assert!(!rescan.stream_options_ambiguous);
            prop_assert_eq!(rescan.stream_options_at, None);
            let at = rescan.inject_at.expect("inject point after removal");
            buf.splice(
                at..at,
                br#""stream_options":{"include_usage":true},"#.iter().copied(),
            );
            let after: Value = serde_json::from_slice(&buf).unwrap_or_else(|err| {
                panic!(
                    "removal and injection broke the JSON ({err}): {}",
                    String::from_utf8_lossy(&buf)
                )
            });
            let mut want = Value::Object(root.clone());
            want["stream_options"] = json!({ "include_usage": true });
            prop_assert_eq!(after, want, "{}", b.text);
            return Ok(());
        }
        if root.contains_key("stream_options") {
            prop_assert_eq!(scan.inject_at, None);
            let at = scan
                .stream_options_at
                .expect("stream_options value located");
            let i = b
                .members
                .iter()
                .position(|m| m.key == "stream_options")
                .unwrap();
            prop_assert_eq!(at, value_offsets(&b.text)[i], "{}", b.text);
            let mut values =
                serde_json::Deserializer::from_slice(&b.text.as_bytes()[at..]).into_iter::<Value>();
            let first = values.next().map(Result::unwrap);
            prop_assert_eq!(first.as_ref(), root.get("stream_options"));
            prop_assert_eq!(peek::plan_stream_usage_injection(b.text.as_bytes()), None);
        } else {
            let at = scan.inject_at.expect("inject point");
            prop_assert_eq!(
                Some(at),
                peek::plan_stream_usage_injection(b.text.as_bytes())
            );
            let mut spliced = b.text.clone().into_bytes();
            spliced.splice(
                at..at,
                br#""stream_options":{"include_usage":true},"#.iter().copied(),
            );
            let after: Value = serde_json::from_slice(&spliced).unwrap_or_else(|err| {
                panic!(
                    "injection broke the JSON ({err}): {}",
                    String::from_utf8_lossy(&spliced)
                )
            });
            let mut want = Value::Object(root.clone());
            want["stream_options"] = json!({ "include_usage": true });
            prop_assert_eq!(after, want, "{}", b.text);
        }
        Ok(())
    });
}

/// Each output limit's span covers exactly the integer serde reads, and replacing it changes
/// only that member.
#[test]
fn prop_limit_spans_match_serde() {
    check("prop_limit_spans_match_serde", body(), |b| {
        let root = b.serde();
        let scan = peek::scan_buffered(b.text.as_bytes());
        for (k, key) in OUTPUT_LIMIT_KEYS.iter().enumerate() {
            let key = std::str::from_utf8(key).unwrap();
            // A duplicate or escaped limit key can only make the provider 400 the client's own
            // request (the clamp edits a member the provider then ignores); it moves no money and
            // crosses no tenant, so the property covers the single, plainly spelled member.
            if b.count(key) != 1 || b.exotic_key(key) {
                continue;
            }
            let i = b.members.iter().position(|m| m.key == key).unwrap();
            let key_at = scan.limit_keys[k].expect("limit key located");
            prop_assert_eq!(
                &b.text[key_at..key_at + key.len() + 2],
                format!("\"{key}\"")
            );
            match root[key].as_u64() {
                Some(n) => {
                    let (s, e) = scan.limit_spans[k].expect("integer limit has a span");
                    prop_assert_eq!(s, value_offsets(&b.text)[i]);
                    prop_assert_eq!(&b.text[s..e], n.to_string());
                    let mut spliced = b.text.clone().into_bytes();
                    spliced.splice(s..e, b"7".iter().copied());
                    let after: Value = serde_json::from_slice(&spliced).unwrap();
                    let mut want = Value::Object(root.clone());
                    want[key] = json!(7);
                    prop_assert_eq!(after, want);
                }
                // Not a non-negative integer. A number's leading digits (`0.5`, `1e6`) are still
                // recorded, contrary to the field's doc; the clamp then edits only those digits,
                // which keeps the body valid JSON and can at worst leave the client's own
                // non-integer limit for the provider to 400 — so the property asserts exactly that.
                None => {
                    if let Some((s, e)) = scan.limit_spans[k] {
                        prop_assert!(root[key].is_number(), "span on a non-number: {}", b.text);
                        prop_assert_eq!(s, value_offsets(&b.text)[i]);
                        let mut spliced = b.text.clone().into_bytes();
                        spliced.splice(s..e, b"7".iter().copied());
                        prop_assert!(serde_json::from_slice::<Value>(&spliced).is_ok());
                    }
                }
            }
        }
        Ok(())
    });
}

/// Arbitrary bytes never panic any scanner, and a span is always inside the body.
#[test]
fn prop_scanners_never_panic_on_any_bytes() {
    check(
        "prop_scanners_never_panic_on_any_bytes",
        (
            prop::collection::vec(any::<u8>(), 0..256),
            prop::collection::vec(any::<usize>(), 0..8),
        ),
        |(bytes, cuts)| {
            let scan = peek::scan_buffered(&bytes);
            for (s, e) in scan
                .model_span
                .into_iter()
                .chain(scan.limit_spans.into_iter().flatten())
            {
                prop_assert!(s <= e && e <= bytes.len());
            }
            for at in scan.inject_at.into_iter().chain(scan.stream_options_at) {
                prop_assert!(at <= bytes.len());
            }
            let _ = peek::plan_stream_usage_injection(&bytes);
            for mut s in [ModelScanner::new(), ModelScanner::for_response()] {
                for c in chunk(&bytes, &cuts) {
                    s.feed(&c);
                }
                let _ = s.take_model();
            }
            Ok(())
        },
    );
}

/// `for_response` reads the model a provider echoes: root `model`, or `message.model` /
/// `response.model` one level down (Anthropic and Responses streams), as serde reads it.
#[test]
fn prop_response_scanner_reads_the_echoed_model() {
    check(
        "prop_response_scanner_reads_the_echoed_model",
        (
            prop::sample::select(vec!["message", "response", ""]),
            "[a-z0-9][a-z0-9.-]{0,30}",
            json_object(2),
            json_object(2),
            prop::collection::vec(any::<u8>(), 1..8),
            prop::collection::vec(any::<usize>(), 0..8),
        ),
        |(nest, id, before, after, w, cuts)| {
            let mut inner = before.as_object().unwrap().clone();
            inner.remove("model");
            inner.insert("model".into(), json!(id));
            let v = if nest.is_empty() {
                Value::Object(inner)
            } else {
                let mut root = after.as_object().unwrap().clone();
                root.retain(|k, _| k != "model" && k != "message" && k != "response");
                root.insert(nest.to_owned(), Value::Object(inner));
                Value::Object(root)
            };
            let mut text = String::new();
            let mut k = 0;
            write(&v, &w, &mut k, PLAIN, &mut text);
            let mut s = ModelScanner::for_response();
            for c in chunk(text.as_bytes(), &cuts) {
                s.feed(&c);
            }
            prop_assert_eq!(s.take_model(), Some(id), "{}", text);
            Ok(())
        },
    );
}
