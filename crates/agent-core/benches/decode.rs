// Bench target: `.unwrap()`/`.expect()` set up fixtures; not production code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Streaming-decode bench: the crate's only *delta-frequency* hot path — the SSE pipeline run once
//! per streamed token, thousands of times per turn. Timing **and** allocations come from `divan`; its
//! `AllocProfiler` (installed as the global allocator below) reports alloc count + bytes per sample
//! beside ns/iter, so the decode findings' allocation claims are visible in one table, not inferred.
//! Run with `mise run bench:decode` (or `cargo bench -p beyond-ai-agent-core --bench decode`).
//!
//! Three groups, mapping to the two audit findings:
//! - `framing` — [`LineFramer`] alone: the byte-buffer newline split (finding #1). Two chunking
//!   strategies bracket the real range: one line per chunk (empty remainder, pure scan+alloc) vs.
//!   the whole body in bulk chunks (many lines per buffer — the `drain`-from-front memmove case the
//!   `memchr`+`split_to` fix targets).
//! - `decode` — [`decode_sse`] per dialect: the `serde_json::Value` parse + field extraction + event
//!   `Vec` build (finding #2), isolated from framing.
//! - `pipeline` — framer + `push_sse_line` + decoder end-to-end, exactly as `GatewayClient::stream`
//!   drives it, at a realistic chunk size — the number that actually matters.
//!
//! Fixtures are built *outside* the closure handed to `Bencher::bench`, so only the measured call is
//! timed and counted — transcript construction doesn't pollute the numbers.

use std::hint::black_box;

use agent_core::client::LineFramer;
use agent_core::dialect::{SseEventBuffer, decode_sse, push_sse_line};
use agent_core::dialect::{anthropic, openai, openai_responses};
use divan::Bencher;
use divan::counter::BytesCount;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// A representative tool-heavy assistant turn: this many streamed text deltas before the tool call.
/// ~800 short deltas ≈ a long prose answer — enough events that per-event cost dominates the fixed
/// message_start/stop framing, which is the regime a real streaming turn runs in.
const TEXT_DELTAS: usize = 800;
/// Fragments the tool call's JSON arguments are streamed in (the model dribbles `input_json_delta`s).
const ARG_FRAGMENTS: usize = 24;
/// A realistic-ish TCP/HTTP read size for the end-to-end pipeline bench — several KB, so many SSE
/// lines land in one buffer (the coalesced-delivery case, worst for drain-from-front framing).
const BULK_CHUNK: usize = 8 * 1024;

/// Build an Anthropic (`/v1/messages`) streaming turn: `message_start`, a text block of `deltas`
/// `text_delta`s, then a `bash` tool call whose arguments arrive in `ARG_FRAGMENTS` fragments, then
/// the terminal `message_delta`/`message_stop`. Includes the `event:` lines the real wire sends
/// (which the framer must still split and `push_sse_line` skips) so the framing workload is honest.
fn anthropic_transcript(deltas: usize) -> String {
    let mut s = String::new();
    let ev = |s: &mut String, name: &str, data: &str| {
        s.push_str("event: ");
        s.push_str(name);
        s.push('\n');
        s.push_str("data: ");
        s.push_str(data);
        s.push_str("\n\n");
    };
    ev(
        &mut s,
        "message_start",
        r#"{"type":"message_start","message":{"usage":{"input_tokens":1200,"output_tokens":0,"cache_read_input_tokens":1100,"cache_creation_input_tokens":0}}}"#,
    );
    ev(
        &mut s,
        "content_block_start",
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
    );
    for _ in 0..deltas {
        ev(
            &mut s,
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"the quick brown fox "}}"#,
        );
    }
    ev(
        &mut s,
        "content_block_stop",
        r#"{"type":"content_block_stop","index":0}"#,
    );
    ev(
        &mut s,
        "content_block_start",
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01ABC","name":"bash"}}"#,
    );
    for _ in 0..ARG_FRAGMENTS {
        ev(
            &mut s,
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"grep -rn "}}"#,
        );
    }
    ev(
        &mut s,
        "content_block_stop",
        r#"{"type":"content_block_stop","index":1}"#,
    );
    ev(
        &mut s,
        "message_delta",
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":180}}"#,
    );
    ev(&mut s, "message_stop", r#"{"type":"message_stop"}"#);
    s
}

/// Build an OpenAI Chat Completions (`/v1/chat/completions`) streaming turn: `deltas` content chunks,
/// a `bash` tool call with fragmented `arguments`, a `finish_reason`, and the trailing usage-only chunk.
fn openai_transcript(deltas: usize) -> String {
    let mut s = String::new();
    let data = |s: &mut String, d: &str| {
        s.push_str("data: ");
        s.push_str(d);
        s.push_str("\n\n");
    };
    for _ in 0..deltas {
        data(
            &mut s,
            r#"{"choices":[{"index":0,"delta":{"content":"the quick brown fox "}}]}"#,
        );
    }
    data(
        &mut s,
        r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_01ABC","function":{"name":"bash","arguments":""}}]}}]}"#,
    );
    for _ in 0..ARG_FRAGMENTS {
        data(
            &mut s,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"grep -rn "}}]}}]}"#,
        );
    }
    data(
        &mut s,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    );
    data(
        &mut s,
        r#"{"choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":180,"prompt_tokens_details":{"cached_tokens":1100}}}"#,
    );
    data(&mut s, "[DONE]");
    s
}

/// Build an OpenAI Responses (`/v1/responses`) streaming turn: an output text item of `deltas`
/// `output_text.delta`s, then the terminal `response.completed` carrying usage.
fn responses_transcript(deltas: usize) -> String {
    let mut s = String::new();
    let ev = |s: &mut String, name: &str, data: &str| {
        s.push_str("event: ");
        s.push_str(name);
        s.push('\n');
        s.push_str("data: ");
        s.push_str(data);
        s.push_str("\n\n");
    };
    ev(
        &mut s,
        "response.output_item.added",
        r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}"#,
    );
    for _ in 0..deltas {
        ev(
            &mut s,
            "response.output_text.delta",
            r#"{"type":"response.output_text.delta","output_index":0,"delta":"the quick brown fox "}"#,
        );
    }
    ev(
        &mut s,
        "response.output_item.done",
        r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}"#,
    );
    ev(
        &mut s,
        "response.completed",
        r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1200,"output_tokens":180,"input_tokens_details":{"cached_tokens":1100},"output_tokens_details":{"reasoning_tokens":0}}}}"#,
    );
    s
}

/// The three dialects' transcripts, labeled — the `args` axis for the decode/pipeline benches.
fn transcripts() -> [(&'static str, String); 3] {
    [
        ("anthropic", anthropic_transcript(TEXT_DELTAS)),
        ("openai", openai_transcript(TEXT_DELTAS)),
        ("responses", responses_transcript(TEXT_DELTAS)),
    ]
}

// --- framing: LineFramer alone (finding #1) -------------------------------------------------------

mod framing {
    use super::*;

    /// Best case: each SSE line delivered as its own chunk, so the buffer's remainder after popping a
    /// line is always empty — the `drain`-from-front memmove is trivial and this measures pure
    /// newline scan + per-line allocation.
    #[divan::bench]
    fn one_line_per_chunk(bencher: Bencher) {
        let body = anthropic_transcript(TEXT_DELTAS).into_bytes();
        // Pre-split into line chunks (each includes its trailing '\n'), outside the timed closure.
        let chunks: Vec<&[u8]> = split_inclusive_newline(&body);
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            let mut framer = LineFramer::new();
            let mut n = 0usize;
            for chunk in &chunks {
                framer.extend(black_box(chunk)).unwrap();
                while let Some(line) = framer.next_line() {
                    n += black_box(&line).len();
                }
            }
            if let Some(tail) = framer.take_tail() {
                n += tail.len();
            }
            n
        });
    }

    /// Worst case for drain-from-front framing: the whole body delivered in a few large (`BULK_CHUNK`)
    /// buffers, so each buffer holds many lines and every `next_line` memmoves the remaining bytes
    /// down. This is the coalesced-fast-stream regime the `memchr`+`split_to` fix targets.
    #[divan::bench]
    fn bulk_chunks(bencher: Bencher) {
        let body = anthropic_transcript(TEXT_DELTAS).into_bytes();
        let chunks: Vec<&[u8]> = body.chunks(BULK_CHUNK).collect();
        bencher.counter(BytesCount::of_slice(&body)).bench(|| {
            let mut framer = LineFramer::new();
            let mut n = 0usize;
            for chunk in &chunks {
                framer.extend(black_box(chunk)).unwrap();
                while let Some(line) = framer.next_line() {
                    n += black_box(&line).len();
                }
            }
            if let Some(tail) = framer.take_tail() {
                n += tail.len();
            }
            n
        });
    }

    /// Split a byte buffer into chunks each ending at (and including) a `\n`, with any unterminated
    /// tail as a final chunk — the "one SSE line per TCP read" delivery pattern.
    fn split_inclusive_newline(body: &[u8]) -> Vec<&[u8]> {
        let mut out = Vec::new();
        let mut start = 0;
        for (i, &b) in body.iter().enumerate() {
            if b == b'\n' {
                out.push(&body[start..=i]);
                start = i + 1;
            }
        }
        if start < body.len() {
            out.push(&body[start..]);
        }
        out
    }
}

// --- decode: decode_sse per dialect (finding #2) --------------------------------------------------

mod decode {
    use super::*;

    #[divan::bench(args = ["anthropic", "openai", "responses"])]
    fn decode_sse_dialect(bencher: Bencher, which: &str) {
        let transcripts = transcripts();
        let (_, body) = transcripts
            .iter()
            .find(|(name, _)| *name == which)
            .expect("known dialect");
        bencher
            .counter(BytesCount::of_slice(body.as_bytes()))
            .bench(|| {
                let mut decoder = new_decoder(which);
                decode_sse(decoder.as_mut(), black_box(body)).expect("valid transcript")
            });
    }
}

// --- pipeline: framer + push_sse_line + decoder, exactly as the client drives it (both findings) ---

mod pipeline {
    use super::*;

    #[divan::bench(args = ["anthropic", "openai", "responses"])]
    fn framer_and_decode(bencher: Bencher, which: &str) {
        let transcripts = transcripts();
        let (_, body) = transcripts
            .iter()
            .find(|(name, _)| *name == which)
            .expect("known dialect");
        let bytes = body.as_bytes();
        let chunks: Vec<&[u8]> = bytes.chunks(BULK_CHUNK).collect();
        bencher.counter(BytesCount::of_slice(bytes)).bench(|| {
            let mut decoder = new_decoder(which);
            let mut framer = LineFramer::new();
            let mut sse_buf = SseEventBuffer::new();
            let mut events = 0usize;
            for chunk in &chunks {
                framer.extend(black_box(chunk)).unwrap();
                while let Some(line) = framer.next_line() {
                    let line = std::str::from_utf8(&line).expect("whole-line utf8");
                    events += push_sse_line(decoder.as_mut(), &mut sse_buf, line)
                        .expect("valid line")
                        .len();
                }
            }
            if let Some(tail) = framer.take_tail() {
                let line = std::str::from_utf8(&tail).expect("utf8 tail");
                events += push_sse_line(decoder.as_mut(), &mut sse_buf, line)
                    .expect("valid tail")
                    .len();
            }
            // Flush any event that never got its trailing blank-line terminator — matches
            // `GatewayClient::stream`'s own final flush.
            events += push_sse_line(decoder.as_mut(), &mut sse_buf, "")
                .expect("valid flush")
                .len();
            events += decoder.finish().expect("clean finish").len();
            events
        });
    }
}

/// A fresh decoder for the named dialect (each bench sample decodes from a clean state).
fn new_decoder(which: &str) -> Box<dyn agent_core::dialect::StreamDecoder> {
    match which {
        "anthropic" => Box::<anthropic::Decoder>::default(),
        "openai" => Box::<openai::Decoder>::default(),
        "responses" => Box::<openai_responses::Decoder>::default(),
        other => panic!("unknown dialect {other}"),
    }
}
