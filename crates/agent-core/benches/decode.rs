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
use agent_core::message::{ContentBlock, Message, ToolDef};
use agent_core::transport::ModelRequest;
use divan::Bencher;
use divan::counter::BytesCount;
use serde_json::json;

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
/// Every non-terminal chunk carries `"finish_reason":null`, matching real OpenAI (and every
/// OpenAI-compatible) traffic — omitting it here previously hid the fast-path regression where
/// `FastTextChoice` didn't declare `finish_reason` and so rejected every real-shaped chunk.
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
            r#"{"choices":[{"index":0,"delta":{"content":"the quick brown fox "},"finish_reason":null}]}"#,
        );
    }
    data(
        &mut s,
        r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_01ABC","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#,
    );
    for _ in 0..ARG_FRAGMENTS {
        data(
            &mut s,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"grep -rn "}}]},"finish_reason":null}]}"#,
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

// --- caps: per-request capability resolution (M14 / findings T2-F2, T2-F3) ------------------------

/// The route-capabilities chain runs on every outbound request (the `client.rs::stream` betas gates),
/// and the audit's claim is that it lowercases the model id 4+ times and rescans the family table
/// twice. This bench calls the public resolution entry point (`capabilities_for_route_with_host`) for a
/// representative Anthropic id and an OpenRouter-fronted vendor-slug id (exercises the `rfind('/')`
/// family-id path) in a loop, so allocs/call and ns/call are visible before vs. after threading a
/// lowercase-once `&str` through the chain.
mod caps {
    use super::*;
    use agent_core::models::capabilities_for_route_with_host;

    #[divan::bench(args = ["claude-opus-4-8", "anthropic/claude-sonnet-4-5"])]
    fn resolve(bencher: Bencher, model: &str) {
        bencher.bench(|| {
            capabilities_for_route_with_host(
                black_box(model),
                black_box(false),
                black_box(false),
                black_box(false),
                black_box(None),
            )
        });
    }
}

// --- encode: dialect build_body over a synthetic multi-turn history (M6) --------------------------
//
// The request-encode counterpart to the decode benches above: `build_body` translates the full
// conversation history into a provider's wire shape once per request (per attempt, per retry), so its
// cost grows with history length — the opposite frequency class from the per-token decode path, but
// on the same turn. Measures allocs/request + ns/request for building the body **and** serializing it
// to bytes, exactly what the client does (`GatewayClient::stream` hands the built body to reqwest's
// `.json(..)`, i.e. `serde_json::to_vec`). Serializing is included so the audit's "materialize a
// `Value` tree the caller immediately serializes" round-trip is visible end to end.

/// Assistant/user/tool-result round-trips in the synthetic history — ~`3 * ROUNDS` messages plus the
/// leading system prompt, i.e. a mid-length agent transcript (36 messages here), the regime where
/// per-message encode cost dominates the fixed request scaffolding.
const ROUNDS: usize = 12;

/// A JSON reasoning signature exactly as a dialect decoder stores it — `serde_json`'s own canonical
/// compact form (`item.to_string()`; see `openai_responses::Decoder`). Both the Responses dialect's
/// `Thinking` replay and the Chat Completions dialect's `reasoning_details` passthrough re-parse a
/// value of this shape, so a realistic history must carry it.
fn reasoning_signature(tag: usize) -> String {
    json!({
        "type": "reasoning",
        "id": format!("rs_{tag}"),
        "summary": [{ "type": "summary_text", "text": "weighing the options before answering" }],
    })
    .to_string()
}

/// A representative mid-length agent transcript for `model`: repeated `user → assistant(text +
/// tool_use) → tool_result` rounds, with a signed thinking block every 4th assistant turn — enough
/// text/tool/thinking history to exercise every per-message encode branch across all three dialects.
///
/// This is the **common** shape a real turn encodes: a single-model, already-clean transcript.
/// `model_id` is stamped with `model` on every assistant turn, so the cross-dialect normalization
/// passes (openai `normalize_cross_model_tool_id`, anthropic `normalize_cross_model_tool_ids`) hit
/// their same-model fast path exactly as in production, rather than the exotic multi-model-fan-out
/// path a unit test constructs. No `Text::id`/`phase` and no `ToolUse::thought_signature` (both set
/// only by a *different* dialect on a transcript that has crossed wires) for the same reason.
fn synthetic_history(model: &str) -> Vec<Message> {
    let mut msgs = Vec::with_capacity(ROUNDS * 3);
    for i in 0..ROUNDS {
        msgs.push(Message::user(format!(
            "Question {i}: what does the code in module_{i} do, and how could I make its hot loop faster \
             without changing behavior?"
        )));

        let mut blocks: Vec<ContentBlock> = Vec::new();
        if i % 4 == 0 {
            blocks.push(ContentBlock::Thinking {
                text: format!(
                    "Module {i} looks allocation-heavy; let me grep for the hot path first."
                ),
                signature: reasoning_signature(i),
            });
        }
        blocks.push(ContentBlock::text(format!(
            "I'll start by searching module_{i} for the relevant function, then read it."
        )));
        blocks.push(ContentBlock::tool_use(
            format!("call_{i}"),
            "bash",
            json!({ "command": format!("grep -rn 'fn hot' module_{i}/ && wc -l module_{i}/*.rs") }),
        ));
        let mut assistant = Message::assistant(blocks);
        assistant.model_id = Some(model.to_string());
        msgs.push(assistant);

        msgs.push(Message::tool_result(
            format!("call_{i}"),
            format!(
                "module_{i}/lib.rs:42: fn hot(x: &[u8]) -> usize\nmatched 3 lines\n\
                 module_{i}/lib.rs 210\nmodule_{i}/util.rs 88\ntotal 298 lines"
            ),
            false,
        ));
    }
    msgs
}

/// A `ModelRequest` over `synthetic_history()` for `model`, with a system prompt and, when `tools`,
/// the usual tool set. The no-tools variant (`tools: false`) is the `--no-tools` follow-up shape whose
/// `tools: []` fallback used to trigger a whole separate `has_tool_history` rescan of the history
/// (openai [T3-F9]).
fn encode_request(model: &str, tools: bool) -> ModelRequest {
    let req = ModelRequest::new(model, synthetic_history(model), 4096).with_system(
        "You are a meticulous performance engineer. Prefer the minimum effective change, keep \
         behavior byte-identical, and measure before and after.",
    );
    if !tools {
        return req;
    }
    req.with_tools(vec![
        ToolDef {
            name: "bash".into(),
            description: "Run a shell command and return combined stdout/stderr.".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"],
            }),
        },
        ToolDef {
            name: "read".into(),
            description: "Read a file from the workspace.".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" }, "offset": { "type": "integer" } },
                "required": ["path"],
            }),
        },
    ])
}

fn model_for(which: &str) -> &'static str {
    match which {
        "anthropic" => "claude-opus-4-8",
        "openai" => "gpt-4o",
        "responses" => "gpt-5",
        other => panic!("unknown dialect {other}"),
    }
}

fn build(which: &str, req: &ModelRequest) -> serde_json::Value {
    match which {
        "anthropic" => anthropic::build_body(black_box(req), false),
        "openai" => openai::build_body(black_box(req)),
        "responses" => openai_responses::build_body(black_box(req)),
        _ => unreachable!(),
    }
}

mod encode {
    use super::*;

    /// The `ModelRequest` → `serde_json::Value` translation alone — the tree-building pass this task's
    /// finding cluster targets, isolated from serialization so the per-message walk savings (openai
    /// [T3-F7]/[T3-F9], anthropic [T3-F2]) aren't drowned out by `to_vec`'s own cost. `tools=false`
    /// exercises the `has_tool_history`/`tools: []` fallback path.
    #[divan::bench(args = ["anthropic", "openai", "responses"])]
    fn build_tree(bencher: Bencher, which: &str) {
        let req = encode_request(model_for(which), true);
        bencher.bench(|| build(which, &req));
    }

    /// Same, on the no-tools follow-up shape (`--no-tools` over a tool-carrying history) — the path
    /// openai's [T3-F9] flag replaces a whole extra history rescan on.
    #[divan::bench(args = ["anthropic", "openai", "responses"])]
    fn build_tree_no_tools(bencher: Bencher, which: &str) {
        let req = encode_request(model_for(which), false);
        bencher.bench(|| build(which, &req));
    }

    /// Build **and serialize** to wire bytes — the whole `ModelRequest` → bytes cost the client pays
    /// (`GatewayClient::stream` → reqwest `.json(..)` → `serde_json::to_vec`), allocs + ns per request.
    #[divan::bench(args = ["anthropic", "openai", "responses"])]
    fn build_and_serialize(bencher: Bencher, which: &str) {
        let req = encode_request(model_for(which), true);
        bencher.bench(|| serde_json::to_vec(&build(which, &req)).expect("serialize body"));
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
