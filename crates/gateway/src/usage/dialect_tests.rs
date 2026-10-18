//! [`body`] and [`stream`] against the four per-dialect parsers they replaced, frozen in
//! `legacy` as they were (each reporting which spelling it billed), over a corpus and over
//! generated input. The contract is exact: on every dialect and both framings, the new path bills
//! what the old one did, byte for byte, refusals included. And [`OpenAiUsage::spelling`], which
//! both routes the OpenAI mapping and guards the dialect, reads every `usage` object the old
//! parser judged as the old rules did (`legacy::spelling`).

use super::*;
use proptest::prelude::*;
use serde_json::{Value, json};

/// The parsers as they were, verbatim apart from reporting the spelling each billed usage was in,
/// and from logging, at each guard, the old rules' spelling beside the new classifier's.
///
/// The one deliberate change since is D287: a `usage` naming no input or output count of the
/// wire's spelling is no usage. Its OpenAI half lives in the live `counts_tokens` both sides call,
/// so the proptests here cannot see a change to it; the corpus pins it directly instead. Its
/// Anthropic half (a `message_start` without `input_tokens` assigns nothing) is the one edit to
/// this copy, marked where it is made.
mod legacy {
    use super::super::*;
    use std::cell::RefCell;

    type Read = Option<(Usage, Spelling)>;

    thread_local! {
        /// Every top-level OpenAI `usage` the old parser judged: (old rules, `OpenAiUsage::spelling`).
        pub static JUDGED: RefCell<Vec<(Option<Spelling>, Option<Spelling>)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// The old rules, in the order the old code applied them (the guard, then `from_chat`'s
    /// routing): the oracle for [`OpenAiUsage::spelling`].
    pub fn spelling(u: &OpenAiUsage) -> Option<Spelling> {
        if !u.counts_tokens() {
            None
        } else if looks_anthropic_shaped(u) {
            Some(Spelling::Anthropic)
        } else if is_responses_shaped(u) {
            Some(Spelling::Responses)
        } else {
            Some(Spelling::Chat)
        }
    }

    /// The old guard, logging both classifiers' reading of `u`.
    fn accepts(u: &OpenAiUsage) -> bool {
        JUDGED.with(|j| j.borrow_mut().push((spelling(u), u.spelling())));
        u.counts_tokens() && !looks_anthropic_shaped(u)
    }

    fn is_responses_shaped(u: &OpenAiUsage) -> bool {
        u.input_tokens.is_some() && u.output_tokens.is_some() && u.total_tokens.is_some()
    }

    fn from_openai(mut u: OpenAiUsage) -> (Usage, Spelling) {
        let vendor = u.vendor();
        let (mut usage, wire) = if is_responses_shaped(&u) {
            let r = OpenAiResponsesUsage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
                input_tokens_details: u.input_tokens_details,
                output_tokens_details: u.output_tokens_details,
                ..OpenAiResponsesUsage::default()
            };
            (Usage::from(r), Spelling::Responses)
        } else {
            (Usage::from_chat(u), Spelling::Chat)
        };
        vendor.apply(&mut usage);
        (usage, wire)
    }

    fn looks_anthropic_shaped(u: &OpenAiUsage) -> bool {
        u.input_tokens.is_some() && u.output_tokens.is_some() && u.total_tokens.is_none()
    }

    pub fn openai_body(body: &[u8]) -> Read {
        #[derive(Deserialize)]
        struct Body {
            usage: Option<OpenAiUsage>,
            #[serde(default, deserialize_with = "de_service_tier")]
            service_tier: Option<ServiceTier>,
            #[serde(default, deserialize_with = "vendor::de_id")]
            id: Option<IdStr>,
            #[serde(default, deserialize_with = "vendor::de_host")]
            provider: Option<HostStr>,
        }
        let (usage, service_tier, id, provider) = match serde_json::from_slice::<Body>(body) {
            Ok(b) => (b.usage?, b.service_tier, b.id, b.provider),
            Err(_) => (
                recover_trailing_usage::<OpenAiUsage>(body)?,
                None,
                None,
                None,
            ),
        };
        if !accepts(&usage) {
            return None;
        }
        let (u, wire) = from_openai(usage);
        let mut u = Usage { service_tier, ..u };
        u.upstream.generation_id = id;
        u.upstream.served_by = provider;
        Some((u, wire))
    }

    pub fn openai_stream(sse: &[u8]) -> Read {
        let finder = memchr::memmem::Finder::new(b"usage");
        let mut broken = false;
        for line in sse_data_lines_rev(sse) {
            if !might_carry_usage(&finder, line) {
                continue;
            }
            match openai_chunk_usage(line) {
                Some(Some(u)) => return Some(u),
                Some(None) => {}
                None => broken = true,
            }
        }
        if broken {
            let mut found = None;
            for_each_event(sse, |data| {
                if let Some(u) = event_payloads(data, &finder, openai_chunk_usage).flatten() {
                    found = Some(u);
                }
            });
            if found.is_some() {
                return found;
            }
        }
        let first = &sse[..memchr::memchr(b'\n', sse).unwrap_or(sse.len())];
        recover_trailing_usage::<OpenAiUsage>(first)
            .filter(accepts)
            .map(from_openai)
    }

    fn openai_chunk_usage(payload: &[u8]) -> Option<Read> {
        #[derive(Deserialize)]
        struct ResponsesEnvelope {
            usage: Option<OpenAiResponsesUsage>,
            #[serde(default, deserialize_with = "de_service_tier")]
            service_tier: Option<ServiceTier>,
            #[serde(default, deserialize_with = "vendor::de_id")]
            id: Option<IdStr>,
        }
        #[derive(Deserialize)]
        struct Chunk {
            usage: Option<OpenAiUsage>,
            response: Option<ResponsesEnvelope>,
            #[serde(default, deserialize_with = "de_service_tier")]
            service_tier: Option<ServiceTier>,
            #[serde(default, deserialize_with = "vendor::de_id")]
            id: Option<IdStr>,
            #[serde(default, deserialize_with = "vendor::de_host")]
            provider: Option<HostStr>,
        }
        let chunk = serde_json::from_slice::<Chunk>(payload).ok()?;
        Some(if let Some(u) = chunk.usage {
            accepts(&u).then(|| {
                let (u, wire) = from_openai(u);
                let mut u = Usage {
                    service_tier: chunk.service_tier,
                    ..u
                };
                u.upstream.generation_id = chunk.id;
                u.upstream.served_by = chunk.provider;
                (u, wire)
            })
        } else {
            chunk.response.and_then(|r| {
                r.usage.filter(|u| u.counts_tokens()).map(|u| {
                    let mut u = Usage {
                        service_tier: r.service_tier,
                        ..Usage::from(u)
                    };
                    u.upstream.generation_id = r.id;
                    (u, Spelling::Responses)
                })
            })
        })
    }

    pub fn anthropic_body(body: &[u8]) -> Option<Usage> {
        #[derive(Deserialize)]
        struct Body {
            usage: Option<AnthropicUsage>,
            #[serde(default, deserialize_with = "vendor::de_id")]
            id: Option<IdStr>,
            #[serde(default)]
            container: Option<Container>,
            #[serde(default, deserialize_with = "vendor::de_host")]
            provider: Option<HostStr>,
        }
        let (u, id, container, provider) = match serde_json::from_slice::<Body>(body) {
            Ok(b) => (b.usage?, b.id, b.container, b.provider),
            Err(_) => (
                recover_trailing_usage::<AnthropicUsage>(body)?,
                None,
                None,
                None,
            ),
        };
        if !u.counts_tokens() || u.looks_openai_shaped() {
            return None;
        }
        let mut u = Usage::from(u);
        u.upstream.generation_id = id;
        u.upstream.container_id = container.and_then(|c| c.id);
        u.upstream.served_by = provider;
        Some(u)
    }

    pub fn anthropic_stream_parts(parts: &[&[u8]]) -> Option<Usage> {
        let finder = memchr::memmem::Finder::new(b"usage");
        let mut usage = Usage::default();
        let mut saw_any = false;
        let mut broken = false;
        for part in parts {
            for line in sse_lines(part) {
                let Some(line) = strip_sse_data(line) else {
                    continue;
                };
                if !might_carry_usage(&finder, line) {
                    continue;
                }
                broken |= !anthropic_apply(&mut usage, &mut saw_any, line);
            }
        }
        if broken {
            usage = Usage::default();
            saw_any = false;
            for part in parts {
                for_each_event(part, |data| {
                    event_payloads(data, &finder, |p| {
                        anthropic_apply(&mut usage, &mut saw_any, p).then_some(())
                    });
                });
            }
        }
        saw_any.then_some(Usage {
            wire: Some(Dialect::Anthropic),
            ..usage
        })
    }

    fn anthropic_apply(usage: &mut Usage, saw_any: &mut bool, payload: &[u8]) -> bool {
        #[derive(Deserialize)]
        struct Message {
            usage: Option<AnthropicUsage>,
            #[serde(default, deserialize_with = "vendor::de_id")]
            id: Option<IdStr>,
        }
        #[derive(Deserialize)]
        struct Delta {
            #[serde(default)]
            container: Option<Container>,
        }
        #[derive(Deserialize)]
        struct Chunk {
            message: Option<Message>,
            usage: Option<AnthropicUsage>,
            delta: Option<Delta>,
        }
        let Ok(chunk) = serde_json::from_slice::<Chunk>(payload) else {
            return false;
        };
        if let Some(c) = chunk.delta.and_then(|d| d.container).and_then(|c| c.id) {
            usage.upstream.container_id = Some(c);
        }
        if let Some(mut m) = chunk.message
            && let Some(mut u) = m.usage.take()
            && !u.looks_openai_shaped()
        {
            // D287, the one edit to this frozen copy: a `message_start` naming no
            // `input_tokens` assigns nothing and is no usage (it once assigned zeros).
            if u.input_tokens.is_some() {
                usage.input_tokens = u.input_tokens.unwrap_or(0);
                usage.cache_read_tokens = u.cache_read_input_tokens.unwrap_or(0);
                usage.cache_write_tokens = u.cache_creation_input_tokens.unwrap_or(0);
                usage.cache_write_1h_tokens = u.cache_creation.ephemeral_1h_input_tokens;
                usage.service_tier = u.service_tier;
                if m.id.is_some() {
                    usage.upstream.generation_id = m.id;
                }
                *saw_any = true;
            }
            u.vendor().apply(usage);
            usage.server_tool_calls = u64::from(usage.server_tools.web_search);
        }
        if let Some(mut u) = chunk.usage
            && !u.looks_openai_shaped()
        {
            let counted = u.counts_tokens();
            if let Some(n) = u.output_tokens.filter(|&n| n > 0) {
                usage.output_tokens = n;
            }
            if let Some(n) = u.input_tokens.filter(|&n| n > 0) {
                usage.input_tokens = n;
            }
            if let Some(n) = u.cache_read_input_tokens.filter(|&n| n > 0) {
                usage.cache_read_tokens = n;
            }
            if let Some(n) = u.cache_creation_input_tokens.filter(|&n| n > 0) {
                usage.cache_write_tokens = n;
            }
            if u.cache_creation.ephemeral_1h_input_tokens > 0 {
                usage.cache_write_1h_tokens = u.cache_creation.ephemeral_1h_input_tokens;
            }
            if u.service_tier.is_some() {
                usage.service_tier = u.service_tier;
            }
            u.vendor().apply(usage);
            usage.server_tool_calls = u64::from(usage.server_tools.web_search);
            if let Some(rt) = u.output_tokens_details.thinking_tokens {
                usage.reasoning_tokens = Some(rt);
            }
            let server_counted = usage.server_tools != ServerTools::default();
            *saw_any |= counted || server_counted;
        }
        true
    }
}

#[derive(Clone, Copy, Debug)]
enum Framing {
    Body,
    Stream,
}

/// What the old path billed in `dialect`, and in which spelling (the OpenAI parser's inference;
/// the Anthropic parser never inferred).
fn old(dialect: Dialect, framing: Framing, head: &[u8], tail: &[u8]) -> Option<(Usage, Spelling)> {
    match (dialect, framing) {
        (Dialect::OpenAi, Framing::Body) => legacy::openai_body(tail),
        (Dialect::OpenAi, Framing::Stream) => legacy::openai_stream(tail),
        (Dialect::Anthropic, Framing::Body) => {
            legacy::anthropic_body(tail).map(|u| (u, Spelling::Anthropic))
        }
        (Dialect::Anthropic, Framing::Stream) => {
            legacy::anthropic_stream_parts(&[head, tail]).map(|u| (u, Spelling::Anthropic))
        }
    }
}

fn new(dialect: Dialect, framing: Framing, head: &[u8], tail: &[u8]) -> Option<Usage> {
    match framing {
        Framing::Body => body(dialect, tail),
        Framing::Stream => stream(dialect, head, tail),
    }
}

/// What one input read as: the spelling the OpenAI parser billed, whether the Anthropic parser
/// billed, and the old-rules spellings of every top-level OpenAI `usage` judged on the way.
struct Read {
    openai: Option<Spelling>,
    anthropic: bool,
    judged: Vec<Option<Spelling>>,
}

/// The contract, on both dialects: the new path bills what the old one did, and the new
/// classifier spells every object the old guard judged as the old rules did.
fn check_equivalent(framing: Framing, head: &[u8], tail: &[u8]) -> Read {
    legacy::JUDGED.with(|j| j.borrow_mut().clear());
    let mut read = Read {
        openai: None,
        anthropic: false,
        judged: Vec::new(),
    };
    for dialect in [Dialect::OpenAi, Dialect::Anthropic] {
        let was = old(dialect, framing, head, tail);
        assert_eq!(
            new(dialect, framing, head, tail),
            was.map(|(u, _)| u),
            "{dialect:?} {framing:?}\nhead: {}\ntail: {}",
            String::from_utf8_lossy(head),
            String::from_utf8_lossy(tail)
        );
        match dialect {
            Dialect::OpenAi => read.openai = was.map(|(_, s)| s),
            Dialect::Anthropic => read.anthropic = was.is_some(),
        }
    }
    for (old, new) in legacy::JUDGED.with(|j| std::mem::take(&mut *j.borrow_mut())) {
        assert_eq!(
            new,
            old,
            "spelling, {framing:?}: {}",
            String::from_utf8_lossy(tail)
        );
        read.judged.push(old);
    }
    read
}

/// A corpus case: name, framing, head, tail, the spelling the OpenAI parser billed, and whether
/// the Anthropic parser billed.
type Case = (
    &'static str,
    Framing,
    Vec<u8>,
    Vec<u8>,
    Option<Spelling>,
    bool,
);

/// One `data:` event per payload, each followed by a blank line.
fn sse(events: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for e in events {
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(e.as_bytes());
        out.extend_from_slice(b"\n\n");
    }
    out
}

/// Both dialects, both framings, the mismatches, empty usage, server-tool-only deltas, Responses'
/// `total_tokens`, cache and reasoning fields: each case names what each old parser billed, so the
/// corpus provably reaches every branch.
#[test]
fn the_corpus_bills_as_the_inferring_parsers_did() {
    use Framing::{Body, Stream};
    use Spelling::{Chat, Responses};
    let embeddings_tail = {
        let mut b = String::from(r#"{"object":"list","data":[{"embedding":["#);
        while b.len() < 4096 {
            b.push_str("0.0123456,");
        }
        b.push_str(r#"0.1]}],"usage":{"prompt_tokens":812,"total_tokens":812}}"#);
        b.into_bytes()[2048..].to_vec()
    };
    let compact_tail = {
        let mut b = String::from(r#"{"object":"response.compaction","output":[{"text":""#);
        b.push_str(&"x".repeat(4096));
        b.push_str(r#""}],"usage":{"input_tokens":900,"output_tokens":40,"total_tokens":940,"input_tokens_details":{"cached_tokens":800}}}"#);
        b.into_bytes()[1024..].to_vec()
    };
    let responses_cut_stream = {
        let mut s =
            String::from(r#"data: {"type":"response.completed","response":{"instructions":""#);
        s.push_str(&"y".repeat(2048));
        s.push_str(r#"","usage":{"input_tokens":5000,"output_tokens":30,"total_tokens":5030,"output_tokens_details":{"reasoning_tokens":12}}}}"#);
        s.push_str("\n\n");
        s.into_bytes()[512..].to_vec()
    };
    let chat_stream = sse(&[
        r#"{"id":"c1","choices":[{"delta":{"content":"hi"}}]}"#,
        r#"{"id":"c1","service_tier":"default","provider":"OpenAI","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":5},"completion_tokens_details":{"reasoning_tokens":3}}}"#,
        "[DONE]",
    ]);
    let chat_multiline = b"data: {\"choices\":[],\ndata: \"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\n".to_vec();
    let messages_head = sse(&[
        r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":40,"output_tokens":1,"cache_read_input_tokens":30,"cache_creation_input_tokens":20,"cache_creation":{"ephemeral_1h_input_tokens":10},"service_tier":"standard"}}}"#,
    ]);
    let messages_tail = sse(&[
        r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"usage"}}"#,
        r#"{"type":"message_delta","delta":{"container":{"id":"cont_1"}},"usage":{"output_tokens":90,"output_tokens_details":{"thinking_tokens":60},"server_tool_use":{"web_search_requests":2}}}"#,
    ]);
    let server_tool_only =
        sse(&[r#"{"type":"message_delta","usage":{"server_tool_use":{"web_search_requests":1}}}"#]);

    #[rustfmt::skip]
    let cases: Vec<Case> = vec![
        ("chat body", Body, vec![], br#"{"id":"chatcmpl-1","service_tier":"priority","usage":{"prompt_tokens":12,"completion_tokens":34,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2},"completion_tokens_details":{"reasoning_tokens":9}}}"#.to_vec(), Some(Chat), false),
        ("chat body, xAI reasoning outside", Body, vec![], br#"{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":25,"completion_tokens_details":{"reasoning_tokens":10}}}"#.to_vec(), Some(Chat), false),
        ("chat body, DeepSeek cache hits", Body, vec![], br#"{"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_cache_hit_tokens":60}}"#.to_vec(), Some(Chat), false),
        ("chat body, OpenRouter cost", Body, vec![], br#"{"provider":"Anthropic","usage":{"prompt_tokens":1,"completion_tokens":2,"cost":0.0012,"server_tool_use_details":{"web_search_requests":1}}}"#.to_vec(), Some(Chat), false),
        ("embeddings body", Body, vec![], br#"{"object":"list","data":[],"usage":{"prompt_tokens":8,"total_tokens":8}}"#.to_vec(), Some(Chat), false),
        ("embeddings tail, front-truncated", Body, vec![], embeddings_tail, Some(Chat), false),
        ("responses body, total_tokens", Body, vec![], br#"{"id":"resp_1","service_tier":"flex","usage":{"input_tokens":20,"output_tokens":8,"total_tokens":28,"input_tokens_details":{"cached_tokens":6},"output_tokens_details":{"reasoning_tokens":4}}}"#.to_vec(), Some(Responses), true),
        ("responses body, xAI reasoning outside", Body, vec![], br#"{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":25,"output_tokens_details":{"reasoning_tokens":10},"cost_in_usd_ticks":1500}}"#.to_vec(), Some(Responses), true),
        ("compact tail, front-truncated", Body, vec![], compact_tail, Some(Responses), true),
        ("anthropic body", Body, vec![], br#"{"id":"msg_1","container":{"id":"cont_9"},"usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":10,"cache_creation_input_tokens":7,"cache_creation":{"ephemeral_1h_input_tokens":3},"output_tokens_details":{"thinking_tokens":20},"service_tier":"priority","server_tool_use":{"web_search_requests":2}}}"#.to_vec(), None, true),
        ("chat body on the Anthropic wire", Body, vec![], br#"{"usage":{"prompt_tokens":1,"completion_tokens":1,"input_tokens":1}}"#.to_vec(), Some(Chat), false),
        ("empty usage", Body, vec![], br#"{"usage":{}}"#.to_vec(), None, false),
        ("null usage", Body, vec![], br#"{"usage":null}"#.to_vec(), None, false),
        ("no usage", Body, vec![], br#"{"error":{"message":"x"}}"#.to_vec(), None, false),
        ("chat stream", Stream, vec![], chat_stream, Some(Chat), false),
        ("chat stream, trailing empty usage", Stream, vec![], sse(&[r#"{"usage":{"prompt_tokens":4,"completion_tokens":2}}"#, r#"{"usage":{}}"#]), Some(Chat), false),
        ("chat stream, trailing Anthropic-shaped usage", Stream, vec![], sse(&[r#"{"usage":{"prompt_tokens":4,"completion_tokens":2}}"#, r#"{"usage":{"input_tokens":1,"output_tokens":1}}"#]), Some(Chat), true),
        ("chat stream, one event over two data lines", Stream, vec![], chat_multiline, Some(Chat), false),
        ("responses stream", Stream, vec![], sse(&[r#"{"type":"response.output_text.delta","delta":"usage"}"#, r#"{"type":"response.completed","response":{"id":"resp_2","service_tier":"default","usage":{"input_tokens":30,"output_tokens":9,"total_tokens":39,"input_tokens_details":{"cached_tokens":10,"cache_write_tokens":4},"output_tokens_details":{"reasoning_tokens":5}}}}"#]), Some(Responses), false),
        ("responses stream, envelope without total_tokens", Stream, vec![], sse(&[r#"{"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":2}}}"#]), Some(Responses), false),
        ("responses stream, final event cut at the front", Stream, vec![], responses_cut_stream, Some(Responses), false),
        ("responses stream, empty envelope usage", Stream, vec![], sse(&[r#"{"type":"response.completed","response":{"usage":{}}}"#]), None, false),
        // D287, pinned here because both sides call the live `counts_tokens`: a usage naming no
        // input or output count of the OpenAI spelling is no OpenAI usage.
        ("output_tokens-only body", Body, vec![], br#"{"usage":{"output_tokens":42}}"#.to_vec(), None, true),
        ("output_tokens-only stream", Stream, vec![], sse(&[r#"{"type":"message_delta","usage":{"output_tokens":42}}"#]), None, true),
        ("total_tokens-only body", Body, vec![], br#"{"usage":{"total_tokens":9}}"#.to_vec(), None, false),
        ("total_tokens-only stream", Stream, vec![], sse(&[r#"{"choices":[],"usage":{"total_tokens":9}}"#]), None, false),
        ("responses stream, output_tokens-only envelope", Stream, vec![], sse(&[r#"{"type":"response.completed","response":{"usage":{"output_tokens":42}}}"#]), None, false),
        // D287, a deliberate change: these two once billed in Chat's spelling (0/0) on the OpenAI
        // wire. A `message_delta` whose `usage` names `output_tokens` without `input_tokens` names
        // no OpenAI count, so it is no usage there, not a zero.
        ("messages stream, head and tail (a message_delta with no input_tokens is no OpenAI usage)", Stream, messages_head.clone(), messages_tail.clone(), None, true),
        ("messages stream, server-tool-only delta", Stream, vec![], server_tool_only, None, true),
        ("messages stream, whole", Stream, vec![], [messages_head, messages_tail].concat(), None, true),
        ("chat stream on the Anthropic wire", Stream, vec![], sse(&[r#"{"type":"message_delta","usage":{"prompt_tokens":1,"completion_tokens":1}}"#]), Some(Chat), false),
        ("empty stream", Stream, vec![], vec![], None, false),
        ("[DONE] only", Stream, vec![], b"data: [DONE]\n\n".to_vec(), None, false),
    ];
    let mut judged = Vec::new();
    for (name, framing, head, tail, openai, anthropic) in &cases {
        let read = check_equivalent(*framing, head, tail);
        assert_eq!(
            (read.openai, read.anthropic),
            (*openai, *anthropic),
            "{name}: the case reaches the branch it names"
        );
        judged.extend(read.judged);
    }
    // The spelling oracle saw every answer the classifier can give.
    for s in [
        None,
        Some(Spelling::Chat),
        Some(Spelling::Responses),
        Some(Spelling::Anthropic),
    ] {
        assert!(judged.contains(&s), "the corpus never judged {s:?}");
    }
}

/// A count anywhere in `u64`, and the values that fail to deserialize as one.
fn any_count() -> impl Strategy<Value = Value> {
    prop_oneof![
        4 => (0u64..1_000_000).prop_map(Value::from),
        2 => Just(json!(0)),
        1 => Just(json!(u64::MAX)),
        1 => Just(json!(-1)),
        1 => Just(json!("12")),
        1 => Just(Value::Null),
    ]
}

/// A usage object over every dialect's vocabulary, so each draw may be spelled for any wire, for
/// none, or for two at once.
fn usage_object() -> impl Strategy<Value = Value> {
    const FIELDS: &[&str] = &[
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "prompt_cache_hit_tokens",
    ];
    let nested = prop_oneof![
        Just(("prompt_tokens_details", "cached_tokens")),
        Just(("completion_tokens_details", "reasoning_tokens")),
        Just(("input_tokens_details", "cached_tokens")),
        Just(("output_tokens_details", "reasoning_tokens")),
        Just(("output_tokens_details", "thinking_tokens")),
        Just(("server_tool_use", "web_search_requests")),
    ];
    (
        prop::collection::vec((prop::sample::select(FIELDS), any_count()), 0..6),
        prop::collection::vec((nested, any_count()), 0..3),
    )
        .prop_map(|(flat, nested)| {
            let mut v = json!({});
            for (k, x) in flat {
                v[k] = x;
            }
            for ((outer, inner), x) in nested {
                if !v[outer].is_object() {
                    v[outer] = json!({});
                }
                v[outer][inner] = x;
            }
            v
        })
}

/// Every place a wire carries a usage object, as a body or as an SSE stream.
fn carried(u: &Value, at: u8) -> (Framing, Vec<u8>) {
    let text = |v: Value| v.to_string();
    match at % 6 {
        0 => (
            Framing::Body,
            text(json!({ "id": "x", "usage": u })).into_bytes(),
        ),
        1 => (
            Framing::Stream,
            sse(&[&text(json!({ "choices": [], "usage": u }))]),
        ),
        2 => (
            Framing::Stream,
            sse(&[&text(
                json!({ "type": "response.completed", "response": { "usage": u } }),
            )]),
        ),
        3 => (
            Framing::Stream,
            sse(&[&text(
                json!({ "type": "message_start", "message": { "usage": u } }),
            )]),
        ),
        4 => (
            Framing::Stream,
            sse(&[&text(json!({ "type": "message_delta", "usage": u }))]),
        ),
        _ => (
            Framing::Stream,
            sse(&[
                &text(json!({ "usage": u })),
                &text(json!({ "type": "response.completed", "response": { "usage": u } })),
            ]),
        ),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, ..ProptestConfig::default() })]

    /// Generated usage objects, in every carrier, front-truncated or not.
    #[test]
    fn generated_usage_bills_as_the_inferring_parsers_did(
        u in usage_object(),
        at in any::<u8>(),
        cut in any::<usize>(),
    ) {
        // The classifier against the old rules on the generated object itself, wherever it lands.
        if let Ok(o) = serde_json::from_value::<OpenAiUsage>(u.clone()) {
            prop_assert_eq!(o.spelling(), legacy::spelling(&o), "{}", u);
        }
        let (framing, bytes) = carried(&u, at);
        check_equivalent(framing, &[], &bytes);
        let cut = cut % (bytes.len() + 1);
        check_equivalent(framing, &[], &bytes[cut..]);
        check_equivalent(framing, &bytes[..cut], &bytes[cut..]);
    }

    /// Arbitrary bytes, as body and as stream.
    #[test]
    fn arbitrary_bytes_bill_as_the_inferring_parsers_did(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
        if let Ok(o) = serde_json::from_slice::<OpenAiUsage>(&bytes) {
            prop_assert_eq!(o.spelling(), legacy::spelling(&o));
        }
        check_equivalent(Framing::Body, &[], &bytes);
        check_equivalent(Framing::Stream, &[], &bytes);
    }
}
