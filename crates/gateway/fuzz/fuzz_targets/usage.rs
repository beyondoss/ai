//! The billing parsers (`usage`: JSON bodies and SSE streams in the OpenAI and Anthropic dialects)
//! on arbitrary bytes, whole and split in two: no panic (`props_usage::prop_parsers_never_panic_on_any_bytes`).
//!
//! Input: 8 bytes of relayed-byte count, 2 bytes of split point, the rest is the body.
#![no_main]

use beyond_gateway::__test::usage::{self, InputTally};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }
    let (head, bytes) = data.split_at(10);
    let extra = u64::from_le_bytes(head[..8].try_into().unwrap_or_default());
    let split = u16::from_le_bytes([head[8], head[9]]) as usize;
    let total = bytes.len() as u64 + extra % (4 << 30);
    let at = split.min(bytes.len());

    let _ = usage::openai_body(bytes);
    let _ = usage::anthropic_body(bytes);
    let _ = usage::openai_stream(bytes);
    let _ = usage::anthropic_stream(bytes);
    let _ = usage::anthropic_stream_parts(&[&bytes[..at], &bytes[at..]]);
    let _ = usage::anthropic_stream_finished(bytes);
    let _ = usage::stream_carried_error(bytes);
    let _ = usage::estimate_stream_output(bytes, total);
    let _ = usage::estimate_body_output(bytes, total);
    let mut t = InputTally::default();
    t.feed(&bytes[..at]);
    t.feed(&bytes[at..]);
    let _ = t.estimate_tokens();
});
