//! Property tests for `usage`: the billing parsers and the cut-short estimators.
//!
//! Properties:
//! - no parser or estimator panics on any bytes, any usage-shaped JSON, any SSE framing;
//! - a stream and its non-stream body bill the same tokens, per dialect;
//! - the proxy's partial views bill like the whole stream: any tail that still holds the usage
//!   event (OpenAI), any head + tail that hold `message_start` and `message_delta` (Anthropic);
//! - generated text cannot forge an Anthropic finish or a stream error;
//! - `InputTally` counts the same whatever the chunking.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod props_support;

use beyond_ai::route::Endpoint;
use beyond_ai::usage::{self, InputTally};
use props_support::requests::chat_request;
use props_support::script::script;
use props_support::*;
use proptest::prelude::*;
use serde_json::{Value, json};

/// Every parser and estimator, on one input. Panics are the failure.
fn run_all(bytes: &[u8], split: usize, total: u64) {
    let _ = usage::openai_body(bytes);
    let _ = usage::anthropic_body(bytes);
    let _ = usage::openai_stream(bytes);
    let _ = usage::anthropic_stream(bytes);
    let at = split.min(bytes.len());
    let _ = usage::anthropic_stream_parts(&[&bytes[..at], &bytes[at..]]);
    let _ = usage::anthropic_stream_finished(bytes);
    let _ = usage::stream_carried_error(bytes);
    let _ = usage::estimate_stream_output(bytes, total);
    let _ = usage::estimate_body_output(bytes, total);
    let mut t = InputTally::default();
    t.feed(&bytes[..at]);
    t.feed(&bytes[at..]);
    let _ = t.estimate_tokens();
}

/// A relayed byte count: the tail plus up to a few GiB before it.
fn total_for(len: usize, extra: u64) -> u64 {
    len as u64 + extra % (4 << 30)
}

/// A count anywhere in `u64`, weighted to the edges.
fn any_count() -> impl Strategy<Value = Value> {
    prop_oneof![
        3 => (0u64..1_000_000).prop_map(Value::from),
        1 => Just(json!(0)),
        1 => Just(json!(u64::MAX)),
        1 => Just(json!(u64::MAX / 2)),
        1 => Just(json!(-1)),
        1 => Just(json!(1.5)),
        1 => Just(json!(1e30)),
        1 => Just(json!("12")),
        1 => Just(Value::Null),
    ]
}

/// A usage block with every metered field, each at any count.
fn usage_block() -> impl Strategy<Value = Value> {
    const FIELDS: &[&str] = &[
        "/prompt_tokens",
        "/completion_tokens",
        "/total_tokens",
        "/prompt_tokens_details/cached_tokens",
        "/prompt_tokens_details/cache_write_tokens",
        "/completion_tokens_details/reasoning_tokens",
        "/prompt_cache_hit_tokens",
        "/input_tokens",
        "/output_tokens",
        "/cache_read_input_tokens",
        "/cache_creation_input_tokens",
        "/cache_creation/ephemeral_1h_input_tokens",
        "/output_tokens_details/thinking_tokens",
        "/output_tokens_details/reasoning_tokens",
        "/input_tokens_details/cached_tokens",
        "/input_tokens_details/cache_write_tokens",
        "/server_tool_use/web_search_requests",
        "/service_tier",
    ];
    prop::collection::vec((prop::sample::select(FIELDS), any_count()), 0..10).prop_map(|kv| {
        let mut v = json!({});
        for (ptr, x) in kv {
            let mut cur = &mut v;
            let parts: Vec<&str> = ptr.trim_start_matches('/').split('/').collect();
            for (i, p) in parts.iter().enumerate() {
                if !cur.is_object() {
                    *cur = json!({});
                }
                if i + 1 == parts.len() {
                    cur[*p] = x.clone();
                } else {
                    cur = cur.as_object_mut().unwrap().entry(*p).or_insert(json!({}));
                }
            }
        }
        v
    })
}

#[test]
fn prop_parsers_never_panic_on_any_bytes() {
    check(
        "prop_parsers_never_panic_on_any_bytes",
        (
            prop::collection::vec(any::<u8>(), 0..512),
            any::<usize>(),
            any::<u64>(),
        ),
        |(bytes, split, extra)| {
            run_all(&bytes, split, total_for(bytes.len(), extra));
            Ok(())
        },
    );
}

/// Usage blocks at boundary counts, in every place each dialect carries one, body and stream.
#[test]
fn prop_parsers_never_panic_on_boundary_counts() {
    check(
        "prop_parsers_never_panic_on_boundary_counts",
        (usage_block(), any::<usize>(), any::<u64>(), framing()),
        |(u, split, extra, fr)| {
            let bodies = [
                json!({ "usage": u }),
                json!({ "type": "message_start", "message": { "usage": u } }),
                json!({ "type": "message_delta", "usage": u }),
                json!({ "type": "response.completed", "response": { "usage": u } }),
            ];
            let mut sse = Vec::new();
            for b in &bodies {
                let text = b.to_string();
                run_all(text.as_bytes(), split, total_for(text.len(), extra));
                fr.event(&mut sse, None, &text);
            }
            run_all(&sse, split, total_for(sse.len(), extra));
            Ok(())
        },
    );
}

/// Arbitrary usage-vocabulary events, framed as SSE: no panic anywhere.
#[test]
fn prop_parsers_never_panic_on_vocabulary_streams() {
    check(
        "prop_parsers_never_panic_on_vocabulary_streams",
        (
            prop::collection::vec(vocab_event(), 0..16),
            framing(),
            any::<usize>(),
            any::<u64>(),
        ),
        |(events, fr, split, extra)| {
            let mut sse = Vec::new();
            for (name, v) in &events {
                fr.event(&mut sse, *name, &v.to_string());
            }
            run_all(&sse, split, total_for(sse.len(), extra));
            Ok(())
        },
    );
}

/// A stream and its non-stream body bill the same tokens, for each dialect.
#[test]
fn prop_streams_bill_like_their_bodies() {
    check(
        "prop_streams_bill_like_their_bodies",
        (
            prop::sample::select(props_support::script::DIALECTS.to_vec()),
            script(),
        ),
        |(d, s)| {
            let (body, stream) = (s.body(d), s.stream(d, None, None));
            let (b, st) = match d {
                Endpoint::Messages => (
                    usage::anthropic_body(&body),
                    usage::anthropic_stream(&stream),
                ),
                _ => (usage::openai_body(&body), usage::openai_stream(&stream)),
            };
            prop_assert!(b.is_some() && st.is_some());
            let (b, st) = (b.unwrap(), st.unwrap());
            prop_assert_eq!(normalize(&b), normalize(&st));
            prop_assert_eq!(
                b.reasoning_tokens.unwrap_or(0),
                st.reasoning_tokens.unwrap_or(0)
            );
            let want = Billed {
                prompt: s.usage.prompt(),
                cache_read: s.usage.cache_read,
                cache_write: s.usage.cache_write,
                output: s.usage.output,
            };
            prop_assert_eq!(normalize(&b), want);
            Ok(())
        },
    );
}

/// The proxy keeps a bounded tail (OpenAI wire) or head and tail (Anthropic): any such view that
/// still holds the events carrying usage bills exactly what the whole stream does.
#[test]
fn prop_partial_views_bill_like_the_whole_stream() {
    check(
        "prop_partial_views_bill_like_the_whole_stream",
        (
            prop::sample::select(props_support::script::DIALECTS.to_vec()),
            script(),
            any::<usize>(),
            any::<usize>(),
        ),
        |(d, s, a, b)| {
            let stream = s.stream(d, None, None);
            let n = stream.len();
            match d {
                Endpoint::Messages => {
                    let whole = usage::anthropic_stream(&stream);
                    // Head: through the end of `message_start`'s event at least.
                    let at = find(&stream, b"message_start", 0).unwrap();
                    let start_end = end_of_event(&stream, at);
                    let delta_at = rfind(&stream, b"message_delta").unwrap();
                    let delta_start = start_of_event(&stream, delta_at);
                    let head = start_end + a % (n - start_end + 1);
                    let tail = b % (delta_start + 1);
                    let parts = usage::anthropic_stream_parts(&[&stream[..head], &stream[tail..]]);
                    prop_assert_eq!(parts, whole);
                }
                _ => {
                    let whole = usage::openai_stream(&stream);
                    let at = rfind(&stream, b"\"usage\":{").unwrap();
                    let start = start_of_event(&stream, at);
                    let tail = a % (start + 1);
                    prop_assert_eq!(usage::openai_stream(&stream[tail..]), whole);
                }
            }
            Ok(())
        },
    );
}

fn find(h: &[u8], n: &[u8], from: usize) -> Option<usize> {
    h.get(from..)?
        .windows(n.len())
        .position(|w| w == n)
        .map(|p| p + from)
}
fn rfind(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).rposition(|w| w == n)
}
/// Start of the SSE line holding byte `at` — the event's first line is at or before it.
fn start_of_event(h: &[u8], at: usize) -> usize {
    // Back to the blank line that ends the previous event.
    let mut i = at;
    while i > 0 {
        if h[i - 1] == b'\n'
            && i >= 2
            && (h[i - 2] == b'\n' || (h[i - 2] == b'\r' && i >= 4 && h[i - 3] == b'\n'))
        {
            return i;
        }
        i -= 1;
    }
    0
}
fn end_of_event(h: &[u8], at: usize) -> usize {
    let mut i = at;
    while i + 1 < h.len() {
        if h[i] == b'\n'
            && (h[i + 1] == b'\n' || (h[i + 1] == b'\r' && h.get(i + 2) == Some(&b'\n')))
        {
            return (i + 3).min(h.len());
        }
        i += 1;
    }
    h.len()
}

/// Generated text can say anything; it can never make a cut-short Anthropic stream look finished,
/// nor any stream look like it carried an error.
#[test]
fn prop_generated_text_cannot_forge_a_finish_or_an_error() {
    check(
        "prop_generated_text_cannot_forge_a_finish_or_an_error",
        (
            prop::sample::select(props_support::script::DIALECTS.to_vec()),
            script(),
            any::<usize>(),
        ),
        |(d, s, cut)| {
            let full = s.stream(d, None, None);
            prop_assert!(
                !usage::stream_carried_error(&full),
                "{}",
                String::from_utf8_lossy(&full)
            );
            if d == Endpoint::Messages {
                // Cut anywhere before the `message_delta` event.
                let n = s.event_count(d);
                let before_delta = cut % (n - 2);
                let cut_stream = s.stream(d, Some(before_delta), None);
                prop_assert!(
                    !usage::anthropic_stream_finished(&cut_stream),
                    "{}",
                    String::from_utf8_lossy(&cut_stream)
                );
                prop_assert!(
                    usage::anthropic_stream_finished(&full)
                        || s.framing.no_event_lines && !full_has_typed_delta(&full)
                );
            }
            Ok(())
        },
    );
}

fn full_has_typed_delta(b: &[u8]) -> bool {
    find(b, br#""type":"message_delta""#, 0).is_some()
}

/// `InputTally` counts the same however the request body is chunked.
#[test]
fn prop_input_tally_ignores_chunking() {
    check(
        "prop_input_tally_ignores_chunking",
        (chat_request(), prop::collection::vec(any::<usize>(), 0..24)),
        |(r, cuts)| {
            let body = serde_json::to_vec(&r.body).unwrap();
            let mut whole = InputTally::default();
            whole.feed(&body);
            // Any chunking, down to a byte at a time (D125).
            let mut small = InputTally::default();
            for c in chunk(&body, &cuts) {
                small.feed(&c);
            }
            prop_assert_eq!(whole.estimate_tokens(), small.estimate_tokens());
            Ok(())
        },
    );
}
