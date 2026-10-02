//! Shared generators and checkers for the property suites (`props_*.rs`).
//!
//! Every property draws realistic-but-adversarial wire shapes: arbitrary unicode and escapes in
//! text and tool arguments, nested JSON with boundary numbers, every content kind a response can
//! carry, and SSE framings (CRLF, comments, keepalives, multi-line data, split at any byte).
//!
//! Scale: each property runs `PROPTEST_CASES` cases (proptest's own variable; `PROPS_CASES` is an
//! alias; default 2000), and stops drawing new ones after `PROPS_SECS` seconds (default 60), so a
//! run stays bounded on a loaded machine and under nextest's `ci` terminate-after (180 s). The
//! weekly deep job (`.github/workflows/deep.yml`: `PROPTEST_CASES=100000 PROPS_SECS=5400 cargo
//! nextest run -p beyond-ai --profile deep -E 'test(/prop/)'`) raises the box and the profile's
//! timeouts so every property runs all of its cases. Every property's name starts with `prop_`. A failure prints its
//! `PROPS_SEED`; set it to replay the same cases.

#![allow(dead_code)]

use proptest::prelude::*;
use proptest::test_runner::{
    Config, FileFailurePersistence, RngSeed, TestCaseResult, TestError, TestRunner,
};
use serde_json::{Map, Value, json};
use std::time::{Duration, Instant};

pub mod requests;
pub mod script;

pub fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Proptest config for every property: `PROPS_CASES` cases, no regression files written into the
/// source tree (every counterexample worth keeping becomes a named regression test instead).
pub fn config() -> Config {
    Config {
        cases: u32::try_from(env_u64("PROPTEST_CASES", env_u64("PROPS_CASES", 2000)))
            .unwrap_or(u32::MAX),
        failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
        max_shrink_iters: 20_000,
        max_shrink_time: 60_000,
        // Rejections are counted by `check`, not capped: a filter that rejects most draws is a
        // generator to fix, and the case count shows it.
        max_global_rejects: u32::MAX,
        ..Config::default()
    }
}

/// Wall-clock budget for one property: `PROPS_SECS`, default 60.
pub struct Budget(Instant, Duration);

impl Budget {
    pub fn start() -> Self {
        Budget(
            Instant::now(),
            Duration::from_secs(env_u64("PROPS_SECS", 60)),
        )
    }
    pub fn spent(&self) -> bool {
        self.0.elapsed() > self.1
    }
}

/// Run one property: up to `PROPTEST_CASES` generated cases, stopping early once `PROPS_SECS` is
/// spent. A failure is shrunk to a minimal input and reported with it.
pub fn check<S>(name: &str, strategy: S, test: impl Fn(S::Value) -> TestCaseResult)
where
    S: Strategy,
    S::Value: std::fmt::Debug,
{
    // A fixed seed per run, printed on failure, so any counterexample can be replayed exactly with
    // `PROPS_SEED=<seed>`.
    let seed = std::env::var("PROPS_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| d.as_nanos() as u64)
        });
    let cfg = Config {
        rng_seed: RngSeed::Fixed(seed),
        ..config()
    };
    let cases = cfg.cases;
    let mut runner = TestRunner::new(cfg);
    let budget = Budget::start();
    let (mut ran, mut rejected) = (0u32, 0u32);
    while ran < cases && !budget.spent() {
        let tree = match strategy.new_tree(&mut runner) {
            Ok(t) => t,
            Err(why) => panic!("{name}: could not generate a case: {why}"),
        };
        match runner.run_one(tree, &test) {
            Ok(true) => ran += 1,
            Ok(false) => rejected += 1,
            Err(TestError::Fail(why, input)) => {
                panic!(
                    "{name} failed after {ran} cases (PROPS_SEED={seed}): {why}\nminimal failing input: {input:#?}"
                )
            }
            Err(TestError::Abort(why)) => panic!("{name} aborted after {ran} cases: {why}"),
        }
    }
    eprintln!(
        "{name}: {ran} cases ({rejected} filtered) in {:.1?}",
        budget.0.elapsed()
    );
}

// --- text -------------------------------------------------------------------------------------

/// One adversarial piece of text: plain words, every JSON escape, SSE framing look-alikes, control
/// characters, multi-byte and astral code points, combining marks, RTL marks, the BOM.
pub fn atom() -> impl Strategy<Value = String> {
    prop_oneof![
        6 => "[a-zA-Z0-9 ,.]{1,8}",
        1 => Just("\"".to_owned()),
        1 => Just("\\".to_owned()),
        1 => Just("\n".to_owned()),
        1 => Just("\r\n".to_owned()),
        1 => Just("\n\n".to_owned()),
        1 => Just("\t".to_owned()),
        1 => Just("\u{0}".to_owned()),
        1 => Just("\u{1b}[0m".to_owned()),
        1 => Just("\u{2028}\u{2029}".to_owned()),
        1 => Just("\u{feff}".to_owned()),
        1 => Just("e\u{301}".to_owned()),
        1 => Just("\u{202e}abc".to_owned()),
        1 => Just("😀".to_owned()),
        1 => Just("日本語".to_owned()),
        1 => Just("data: [DONE]".to_owned()),
        1 => Just("event: error".to_owned()),
        1 => Just("\"usage\":{\"input_tokens\":1}".to_owned()),
        1 => Just("\"error\":{".to_owned()),
        1 => Just("\"type\":\"message_delta\"".to_owned()),
        1 => Just("{\"".to_owned()),
        1 => Just("\\u0041".to_owned()),
        1 => any::<char>().prop_map(String::from),
    ]
}

/// Text made of 0..n atoms.
pub fn text(max: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(atom(), 0..max).prop_map(|v| v.concat())
}

/// Non-empty text.
pub fn text1(max: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(atom(), 1..max.max(2)).prop_map(|v| v.concat())
}

/// Text pre-split into the deltas a stream carries it in (every piece non-empty).
pub fn pieces(max: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(text1(4), 1..max.max(2))
}

/// A tool name as a real tool loop would declare it.
pub fn tool_name() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => "[a-z][a-z0-9_]{0,15}",
        1 => "[a-zA-Z_][a-zA-Z0-9_-]{0,63}",
        1 => Just("mcp__server__tool".to_owned()),
    ]
}

/// An id an upstream hands out (never one the gateway would mint).
pub fn up_id(prefix: &'static str) -> impl Strategy<Value = String> {
    "[A-Za-z0-9]{4,24}".prop_map(move |s| format!("{prefix}{s}"))
}

// --- JSON -------------------------------------------------------------------------------------

/// A JSON number exactly representable after a serde_json round trip (integers, and floats whose
/// shortest repr re-parses to themselves).
pub fn json_number() -> impl Strategy<Value = Value> {
    prop_oneof![
        4 => (-1000i64..1000).prop_map(Value::from),
        1 => Just(Value::from(0)),
        1 => Just(Value::from(i64::MIN)),
        1 => Just(Value::from(i64::MAX)),
        1 => Just(Value::from(u64::MAX)),
        1 => Just(json!(0.5)),
        1 => Just(json!(-1.25e-7)),
        1 => Just(json!(1e300)),
    ]
}

pub fn json_leaf() -> impl Strategy<Value = Value> {
    prop_oneof![
        1 => Just(Value::Null),
        1 => any::<bool>().prop_map(Value::Bool),
        2 => json_number(),
        4 => text(6).prop_map(Value::String),
    ]
}

/// An arbitrary JSON value, nested up to `depth`.
pub fn json_value(depth: u32) -> impl Strategy<Value = Value> {
    json_leaf().prop_recursive(depth, 48, 6, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
            prop::collection::vec((text(3), inner), 0..5)
                .prop_map(|kv| Value::Object(kv.into_iter().collect::<Map<_, _>>())),
        ]
    })
}

/// Tool arguments: always a JSON object (a function tool's arguments are one).
pub fn json_object(depth: u32) -> impl Strategy<Value = Value> {
    prop::collection::vec((text(3), json_value(depth)), 0..5)
        .prop_map(|kv| Value::Object(kv.into_iter().collect::<Map<_, _>>()))
}

/// Split `s` at char boundaries near the given fractions (each in 0..1000), dropping empties.
pub fn split_at(s: &str, cuts: &[u16]) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    let bounds: Vec<usize> = s.char_indices().map(|(i, _)| i).skip(1).collect();
    let mut at: Vec<usize> = cuts
        .iter()
        .filter_map(|c| {
            if bounds.is_empty() {
                None
            } else {
                Some(bounds[usize::from(*c) % bounds.len()])
            }
        })
        .collect();
    at.sort_unstable();
    at.dedup();
    let mut out = Vec::new();
    let mut prev = 0;
    for a in at {
        out.push(s[prev..a].to_owned());
        prev = a;
    }
    out.push(s[prev..].to_owned());
    out
}

// --- usage ------------------------------------------------------------------------------------

/// A token count as a provider reports it: mostly ordinary, sometimes zero, sometimes huge.
///
/// Huge stays at or below `u64::MAX / 8`, so no sum of a usage block's five counts passes
/// `u64::MAX`. That is not a D87 exclusion: these properties compare the usage a client is shown
/// with the usage billed, and a saturated sum loses the information either side would need (an
/// OpenAI-shaped block whose `prompt + completion` and `prompt + completion + reasoning` both
/// saturate reads as xAI's reasoning-outside convention). Counts at `u64::MAX` are covered for
/// panics by `props_usage::usage_block` and `props_regressions::usage_with_boundary_counts_never_panics`.
pub fn tokens() -> impl Strategy<Value = u64> {
    prop_oneof![
        8 => 0u64..200_000,
        2 => Just(0u64),
        1 => Just(u64::from(u32::MAX)),
        1 => Just(u64::MAX / 8),
    ]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tokens {
    /// Prompt tokens neither read from nor written to the cache.
    pub uncached: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub output: u64,
    pub reasoning: Option<u64>,
}

impl Tokens {
    pub fn prompt(&self) -> u64 {
        self.uncached
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }
}

pub fn usage_tokens() -> impl Strategy<Value = Tokens> {
    (
        tokens(),
        tokens(),
        tokens(),
        tokens(),
        prop::option::of(tokens()),
    )
        .prop_map(
            |(uncached, cache_read, cache_write, output, reasoning)| Tokens {
                uncached,
                cache_read,
                cache_write,
                output,
                // OpenAI counts reasoning inside `completion_tokens`, so it never exceeds it.
                reasoning: reasoning.map(|r| r.min(output)),
            },
        )
}

/// What a billing row says, normalized through its wire (`usage_wire`): the whole prompt, the
/// cache split, output, reasoning (None and 0 are the same count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Billed {
    pub prompt: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub output: u64,
}

pub fn normalize(u: &beyond_ai::usage::Usage) -> Billed {
    let anthropic = matches!(u.wire, Some(beyond_ai::route::Dialect::Anthropic));
    let prompt = if anthropic {
        u.input_tokens
            .saturating_add(u.cache_read_tokens)
            .saturating_add(u.cache_write_tokens)
    } else {
        u.input_tokens
    };
    Billed {
        prompt,
        cache_read: u.cache_read_tokens,
        cache_write: u.cache_write_tokens,
        output: u.output_tokens,
    }
}

// --- SSE --------------------------------------------------------------------------------------

/// How an upstream frames its events: line ending, comments, keepalives, extra spacing.
#[derive(Clone, Copy, Debug)]
pub struct Framing {
    pub crlf: bool,
    /// `: keepalive` comment events between events.
    pub comments: bool,
    /// `data:` with no space, or with several.
    pub data_spaces: u8,
    /// Omit the `event:` line where the dialect names its type inside the data too.
    pub no_event_lines: bool,
    /// Carry an object's JSON on two `data:` lines (legal SSE: a client joins them with `\n`,
    /// which is JSON whitespace).
    pub multiline: bool,
}

/// How often a stream spreads its JSON over two `data:` lines.
pub const MULTILINE_WEIGHT: f64 = 0.15;

pub fn framing() -> impl Strategy<Value = Framing> {
    (
        any::<bool>(),
        any::<bool>(),
        0u8..3,
        any::<bool>(),
        prop::bool::weighted(MULTILINE_WEIGHT),
    )
        .prop_map(
            |(crlf, comments, data_spaces, no_event_lines, multiline)| Framing {
                crlf,
                comments,
                data_spaces,
                no_event_lines,
                multiline,
            },
        )
}

impl Framing {
    pub const PLAIN: Framing = Framing {
        crlf: false,
        comments: false,
        data_spaces: 1,
        no_event_lines: false,
        multiline: false,
    };

    pub fn event(&self, out: &mut Vec<u8>, name: Option<&str>, data: &str) {
        let nl: &[u8] = if self.crlf { b"\r\n" } else { b"\n" };
        if self.comments {
            out.extend_from_slice(b": keepalive");
            out.extend_from_slice(nl);
            out.extend_from_slice(nl);
        }
        if let Some(name) = name.filter(|_| !self.no_event_lines) {
            out.extend_from_slice(b"event: ");
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(nl);
        }
        let line = |out: &mut Vec<u8>, payload: &str| {
            out.extend_from_slice(b"data:");
            for _ in 0..self.data_spaces {
                out.push(b' ');
            }
            out.extend_from_slice(payload.as_bytes());
            out.extend_from_slice(nl);
        };
        match data.strip_prefix('{').filter(|_| self.multiline) {
            Some(rest) => {
                line(out, "{");
                line(out, rest);
            }
            None => line(out, data),
        }
        out.extend_from_slice(nl);
    }
}

/// One SSE event of a client-facing stream, as the bridge wrote it.
#[derive(Debug, Clone)]
pub struct Event {
    pub name: Option<String>,
    pub data: String,
}

/// Parse a well-formed SSE byte stream into events. `Err` names the first framing problem.
pub fn parse_events(bytes: &[u8]) -> Result<Vec<Event>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("client SSE is not UTF-8: {e}"))?;
    let mut events = Vec::new();
    if text.is_empty() {
        return Ok(events);
    }
    let body = text
        .strip_suffix("\n\n")
        .ok_or_else(|| format!("client SSE does not end with a blank line: {text:?}"))?;
    for raw in body.split("\n\n") {
        // A comment-only event (the bridge's keep-alive, D251) is no event to any SSE parser.
        if raw.split('\n').all(|l| l.starts_with(':')) {
            continue;
        }
        let mut name = None;
        let mut data: Option<String> = None;
        for line in raw.split('\n') {
            if let Some(n) = line.strip_prefix("event: ") {
                name = Some(n.to_owned());
            } else if let Some(d) = line.strip_prefix("data: ") {
                if data.is_some() {
                    return Err(format!("two data lines in one event: {raw:?}"));
                }
                data = Some(d.to_owned());
            } else {
                return Err(format!("unexpected SSE line {line:?} in {raw:?}"));
            }
        }
        let data = data.ok_or_else(|| format!("event without data: {raw:?}"))?;
        events.push(Event { name, data });
    }
    Ok(events)
}

/// Replace every id the gateway minted (`{prefix}_gw{hex}`) with a placeholder numbered by first
/// appearance, and every timestamp with 0, so two runs of the same stream compare equal.
pub fn normalize_minted(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut ids: Vec<String> = Vec::new();
    let mut out = String::new();
    for chunk in text.split_inclusive("\n\n") {
        // Keep-alive comments (D251) follow the feed's chunking, not the stream: no parser sees
        // them, so two feeds of one stream compare without them.
        if chunk
            .trim_end_matches('\n')
            .split('\n')
            .all(|l| l.starts_with(':'))
        {
            continue;
        }
        let mut lines = String::new();
        for line in chunk.split_inclusive('\n') {
            if let Some(d) = line.strip_prefix("data: ") {
                let (d, nl) = d.strip_suffix('\n').map_or((d, ""), |x| (x, "\n"));
                match serde_json::from_str::<Value>(d) {
                    Ok(mut v) => {
                        scrub(&mut v, &mut ids);
                        lines.push_str("data: ");
                        lines.push_str(&v.to_string());
                        lines.push_str(nl);
                    }
                    Err(_) => lines.push_str(line),
                }
            } else {
                lines.push_str(line);
            }
        }
        out.push_str(&lines);
    }
    out
}

/// A JSON value with minted ids and timestamps scrubbed (see [`normalize_minted`]).
pub fn scrub(v: &mut Value, ids: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m.iter_mut() {
                if k == "created" || k == "created_at" {
                    *x = json!(0);
                } else {
                    scrub(x, ids);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| scrub(x, ids)),
        Value::String(s) => {
            if let Some(at) = s.find("_gw") {
                let n = match ids.iter().position(|i| i == s) {
                    Some(n) => n,
                    None => {
                        ids.push(s.clone());
                        ids.len() - 1
                    }
                };
                *s = format!("{}_MINTED{n}", &s[..at]);
            }
        }
        _ => {}
    }
}

// --- structured fuzz --------------------------------------------------------------------------

/// Every key the three dialects' stream events and bodies use, so random objects land on real
/// code paths instead of being ignored as unknown.
pub const KEYS: &[&str] = &[
    "type",
    "index",
    "id",
    "model",
    "role",
    "content",
    "text",
    "delta",
    "content_block",
    "message",
    "usage",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "cached_tokens",
    "cache_write_tokens",
    "completion_tokens_details",
    "reasoning_tokens",
    "output_tokens_details",
    "input_tokens_details",
    "thinking_tokens",
    "choices",
    "finish_reason",
    "native_finish_reason",
    "tool_calls",
    "function",
    "custom",
    "name",
    "arguments",
    "input",
    "partial_json",
    "thinking",
    "signature",
    "data",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "format",
    "summary",
    "refusal",
    "stop_reason",
    "stop_details",
    "explanation",
    "error",
    "code",
    "param",
    "metadata",
    "raw",
    "provider_name",
    "response",
    "item",
    "item_id",
    "output_index",
    "content_index",
    "summary_index",
    "output",
    "status",
    "incomplete_details",
    "reason",
    "call_id",
    "encrypted_content",
    "sequence_number",
    "created",
    "created_at",
    "object",
    "part",
    "parts",
    "detail",
];

/// String values that steer dispatch: event types, block types, finish reasons, roles.
pub const TYPES: &[&str] = &[
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
    "text",
    "text_delta",
    "tool_use",
    "input_json_delta",
    "thinking",
    "thinking_delta",
    "signature_delta",
    "redacted_thinking",
    "server_tool_use",
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.output_item.done",
    "response.output_text.delta",
    "response.refusal.delta",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.custom_tool_call_input.delta",
    "response.custom_tool_call_input.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_text.delta",
    "response.reasoning_summary_part.added",
    "response.completed",
    "response.incomplete",
    "response.failed",
    "function_call",
    "custom_tool_call",
    "reasoning",
    "message",
    "output_text",
    "refusal",
    "reasoning.text",
    "reasoning.summary",
    "reasoning.encrypted",
    "anthropic-claude-v1",
    "openai-responses-v1",
    "function",
    "assistant",
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "end_turn",
    "max_tokens",
    "tool_use",
    "refusal",
    "pause_turn",
    "completed",
    "incomplete",
    "failed",
    "max_output_tokens",
    "[DONE]",
    "chat.completion.chunk",
    "chat.completion",
    "response",
];

pub fn vocab_leaf() -> impl Strategy<Value = Value> {
    prop_oneof![
        1 => Just(Value::Null),
        1 => any::<bool>().prop_map(Value::Bool),
        2 => json_number(),
        2 => (0u64..4).prop_map(Value::from),
        3 => prop::sample::select(TYPES).prop_map(|s| json!(s)),
        2 => text(4).prop_map(Value::String),
        1 => json_object(1).prop_map(|o| json!(o.to_string())),
    ]
}

/// A JSON value whose keys and dispatch strings come from the dialects' own vocabulary.
pub fn vocab_value(depth: u32) -> impl Strategy<Value = Value> {
    vocab_leaf().prop_recursive(depth, 64, 6, |inner| {
        prop_oneof![
            1 => prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            3 => prop::collection::vec((prop::sample::select(KEYS), inner), 0..6).prop_map(|kv| {
                Value::Object(kv.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
            }),
        ]
    })
}

/// A vocabulary object with a dispatching `type`.
pub fn vocab_event() -> impl Strategy<Value = (Option<&'static str>, Value)> {
    (
        prop::option::of(prop::sample::select(TYPES)),
        prop::sample::select(TYPES),
        prop::collection::vec((prop::sample::select(KEYS), vocab_value(3)), 0..6),
    )
        .prop_map(|(name, typ, kv)| {
            let mut m: Map<String, Value> =
                kv.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
            m.insert("type".into(), json!(typ));
            (name, Value::Object(m))
        })
}

/// Cut `bytes` into chunks at the given positions (each taken modulo the length).
pub fn chunk(bytes: &[u8], cuts: &[usize]) -> Vec<Vec<u8>> {
    if bytes.is_empty() {
        return vec![Vec::new()];
    }
    let mut at: Vec<usize> = cuts.iter().map(|c| c % bytes.len()).collect();
    at.sort_unstable();
    at.dedup();
    let mut out = Vec::new();
    let mut prev = 0;
    for a in at {
        out.push(bytes[prev..a].to_vec());
        prev = a;
    }
    out.push(bytes[prev..].to_vec());
    out
}
