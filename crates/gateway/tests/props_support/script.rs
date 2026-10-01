//! One model response as dialect-neutral content ([`Script`]), rendered the way each upstream
//! dialect sends it (non-stream body and SSE stream), and read back from what a client of each
//! dialect receives ([`Seen`]).

use super::*;
use beyond_ai::route::Endpoint;

pub const DIALECTS: [Endpoint; 3] = [
    Endpoint::ChatCompletions,
    Endpoint::Messages,
    Endpoint::Responses,
];

#[derive(Clone, Debug)]
pub enum Block {
    /// Text, as the deltas it streams in.
    Text(Vec<String>),
    /// A thinking block: its text in deltas, and Anthropic's signature (when signed).
    Thinking {
        text: Vec<String>,
        signature: Option<String>,
    },
    Redacted(String),
    /// A server tool Anthropic ran itself (web search): a `server_tool_use` block, whose input
    /// streams as `input_json_delta` like a client call's, then its result block. No other dialect
    /// carries it, and it must never surface as a client tool call.
    Server {
        id: String,
        query: String,
    },
    /// A function call. `cuts` split the argument JSON into deltas.
    Tool {
        id: String,
        name: String,
        args: Value,
        cuts: Vec<u16>,
    },
}

impl Block {
    pub fn text(&self) -> Option<String> {
        match self {
            Block::Text(p) => Some(p.concat()),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    End,
    MaxTokens,
    Sequence,
}

#[derive(Clone, Debug)]
pub struct Script {
    pub id: String,
    pub model: String,
    pub blocks: Vec<Block>,
    pub stop: Stop,
    pub usage: Tokens,
    /// Anthropic `message_delta` repeats the input and cache counts (the 2025+ API does).
    pub delta_repeats_input: bool,
    /// Usage rides the finish chunk itself rather than a trailing usage-only chunk (Chat).
    pub usage_on_finish: bool,
    pub framing: Framing,
    /// Chat: parallel calls open together and their argument deltas interleave round-robin.
    pub interleave: bool,
    /// Chat: every chunk carries `"usage": null` (what OpenAI sends under `include_usage`) and
    /// tool-call deltas carry `"content": null`.
    pub chat_nulls: bool,
}

pub fn block() -> impl Strategy<Value = Block> {
    prop_oneof![
        4 => pieces(6).prop_map(Block::Text),
        2 => (pieces(4), prop::option::of("[A-Za-z0-9+/=]{8,64}")).prop_map(|(text, signature)| {
            Block::Thinking { text, signature }
        }),
        1 => "[A-Za-z0-9+/=]{8,64}".prop_map(Block::Redacted),
        1 => (up_id("srvtoolu_"), text1(3)).prop_map(|(id, query)| Block::Server { id, query }),
        3 => (
            up_id("toolu_"),
            tool_name(),
            json_object(3),
            prop::collection::vec(any::<u16>(), 0..6)
        )
            .prop_map(|(id, name, args, cuts)| Block::Tool {
                id,
                name,
                args,
                cuts
            }),
    ]
}

pub fn script() -> impl Strategy<Value = Script> {
    (
        up_id("id"),
        "[a-z0-9.-]{1,24}",
        prop::collection::vec(block(), 0..6),
        prop_oneof![
            4 => Just(Stop::End),
            1 => Just(Stop::MaxTokens),
            1 => Just(Stop::Sequence)
        ],
        usage_tokens(),
        any::<bool>(),
        any::<bool>(),
        framing(),
        any::<(bool, bool)>(),
    )
        .prop_map(
            |(
                id,
                model,
                blocks,
                stop,
                usage,
                delta_repeats_input,
                usage_on_finish,
                framing,
                (interleave, chat_nulls),
            )| {
                Script {
                    id,
                    model,
                    blocks,
                    stop,
                    usage,
                    delta_repeats_input,
                    usage_on_finish,
                    framing,
                    interleave,
                    chat_nulls,
                }
            },
        )
}

impl Script {
    pub fn has_tools(&self) -> bool {
        self.blocks.iter().any(|b| matches!(b, Block::Tool { .. }))
    }

    pub fn text(&self) -> String {
        self.blocks.iter().filter_map(Block::text).collect()
    }

    pub fn tools(&self) -> Vec<(String, Value)> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Tool { name, args, .. } => Some((name.clone(), args.clone())),
                _ => None,
            })
            .collect()
    }

    /// Signed and redacted thinking, as Anthropic blocks, in order.
    pub fn replayable_thinking(&self) -> Vec<Value> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Thinking {
                    text,
                    signature: Some(sig),
                } => {
                    Some(json!({ "type": "thinking", "thinking": text.concat(), "signature": sig }))
                }
                Block::Redacted(d) => Some(json!({ "type": "redacted_thinking", "data": d })),
                _ => None,
            })
            .collect()
    }

    fn anthropic_stop(&self) -> &'static str {
        if self.has_tools() && self.stop == Stop::End {
            return "tool_use";
        }
        match self.stop {
            Stop::End => "end_turn",
            Stop::MaxTokens => "max_tokens",
            Stop::Sequence => "stop_sequence",
        }
    }

    fn chat_finish(&self) -> &'static str {
        if self.has_tools() && self.stop == Stop::End {
            return "tool_calls";
        }
        match self.stop {
            Stop::End | Stop::Sequence => "stop",
            Stop::MaxTokens => "length",
        }
    }

    // --- usage per dialect --------------------------------------------------------------------

    fn anthropic_usage(&self, start: bool) -> Value {
        let u = self.usage;
        let mut m = Map::new();
        if start || self.delta_repeats_input {
            m.insert("input_tokens".into(), json!(u.uncached));
            m.insert("cache_creation_input_tokens".into(), json!(u.cache_write));
            m.insert("cache_read_input_tokens".into(), json!(u.cache_read));
        }
        m.insert(
            "output_tokens".into(),
            json!(if start { 1.min(u.output) } else { u.output }),
        );
        if !start && let Some(r) = u.reasoning {
            m.insert(
                "output_tokens_details".into(),
                json!({ "thinking_tokens": r }),
            );
        }
        Value::Object(m)
    }

    fn chat_usage(&self) -> Value {
        let u = self.usage;
        let mut m = Map::new();
        m.insert("prompt_tokens".into(), json!(u.prompt()));
        m.insert("completion_tokens".into(), json!(u.output));
        m.insert(
            "total_tokens".into(),
            json!(u.prompt().saturating_add(u.output)),
        );
        m.insert(
            "prompt_tokens_details".into(),
            json!({ "cached_tokens": u.cache_read, "cache_write_tokens": u.cache_write }),
        );
        if let Some(r) = u.reasoning {
            m.insert(
                "completion_tokens_details".into(),
                json!({ "reasoning_tokens": r }),
            );
        }
        Value::Object(m)
    }

    fn responses_usage(&self) -> Value {
        let u = self.usage;
        json!({
            "input_tokens": u.prompt(),
            "input_tokens_details": { "cached_tokens": u.cache_read, "cache_write_tokens": u.cache_write },
            "output_tokens": u.output,
            "output_tokens_details": { "reasoning_tokens": u.reasoning.unwrap_or(0) },
            "total_tokens": u.prompt().saturating_add(u.output),
        })
    }

    // --- non-stream bodies --------------------------------------------------------------------

    pub fn body(&self, d: Endpoint) -> Vec<u8> {
        let v = match d {
            Endpoint::Messages => self.anthropic_body(),
            Endpoint::Responses => self.responses_body(),
            _ => self.chat_body(),
        };
        serde_json::to_vec(&v).unwrap()
    }

    fn anthropic_body(&self) -> Value {
        let content: Vec<Value> = self
            .blocks
            .iter()
            .flat_map(|b| match b {
                Block::Text(p) => vec![json!({ "type": "text", "text": p.concat() })],
                Block::Thinking { text, signature } => {
                    vec![json!({ "type": "thinking", "thinking": text.concat(), "signature": signature.clone().unwrap_or_default() })]
                }
                Block::Redacted(d) => vec![json!({ "type": "redacted_thinking", "data": d })],
                Block::Server { id, query } => vec![
                    json!({ "type": "server_tool_use", "id": id, "name": "web_search", "input": { "query": query } }),
                    json!({ "type": "web_search_tool_result", "tool_use_id": id, "content": [] }),
                ],
                Block::Tool { id, name, args, .. } => {
                    vec![json!({ "type": "tool_use", "id": id, "name": name, "input": args })]
                }
            })
            .collect();
        let mut usage = self.anthropic_usage(true);
        usage["output_tokens"] = json!(self.usage.output);
        if let Some(r) = self.usage.reasoning {
            usage["output_tokens_details"] = json!({ "thinking_tokens": r });
        }
        json!({
            "id": self.id, "type": "message", "role": "assistant", "model": self.model,
            "content": content, "stop_reason": self.anthropic_stop(), "stop_sequence": null,
            "usage": usage,
        })
    }

    /// OpenRouter's shape: reasoning text plus `reasoning_details` carrying Claude's signatures.
    fn chat_reasoning_details(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            match b {
                Block::Thinking { text, signature } => {
                    let mut d = json!({ "type": "reasoning.text", "text": text.concat(), "format": "anthropic-claude-v1", "index": i });
                    if let Some(sig) = signature {
                        d["signature"] = json!(sig);
                    }
                    out.push(d);
                }
                Block::Redacted(data) => out.push(json!({
                    "type": "reasoning.encrypted", "data": data, "format": "anthropic-claude-v1", "index": i,
                })),
                _ => {}
            }
        }
        out
    }

    fn chat_body(&self) -> Value {
        let text = self.text();
        let calls: Vec<Value> = self
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Tool { id, name, args, .. } => Some(json!({
                    "id": id, "type": "function",
                    "function": { "name": name, "arguments": args.to_string() },
                })),
                _ => None,
            })
            .collect();
        let mut msg = Map::new();
        msg.insert("role".into(), json!("assistant"));
        msg.insert(
            "content".into(),
            if text.is_empty() && !calls.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
        if !calls.is_empty() {
            msg.insert("tool_calls".into(), Value::Array(calls));
        }
        let details = self.chat_reasoning_details();
        if !details.is_empty() {
            let reasoning: String = self
                .blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Thinking { text, .. } => Some(text.concat()),
                    _ => None,
                })
                .collect();
            msg.insert("reasoning".into(), json!(reasoning));
            msg.insert("reasoning_details".into(), Value::Array(details));
        }
        json!({
            "id": self.id, "object": "chat.completion", "created": 1_700_000_000, "model": self.model,
            "choices": [{ "index": 0, "message": Value::Object(msg), "finish_reason": self.chat_finish() }],
            "usage": self.chat_usage(),
        })
    }

    fn responses_items(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            match b {
                Block::Text(p) => out.push(json!({
                    "type": "message", "id": format!("msg_{i}"), "status": "completed", "role": "assistant",
                    "content": [{ "type": "output_text", "text": p.concat(), "annotations": [] }],
                })),
                Block::Thinking { text, .. } => out.push(json!({
                    "type": "reasoning", "id": format!("rs_{i}"),
                    "summary": [{ "type": "summary_text", "text": text.concat() }],
                })),
                // No Responses form.
                Block::Redacted(_) | Block::Server { .. } => {}
                Block::Tool { id, name, args, .. } => out.push(json!({
                    "type": "function_call", "id": format!("fc_{i}"), "call_id": id, "name": name,
                    "arguments": args.to_string(), "status": "completed",
                })),
            }
        }
        out
    }

    fn responses_status(&self) -> (&'static str, Value) {
        match self.stop {
            Stop::MaxTokens => ("incomplete", json!({ "reason": "max_output_tokens" })),
            _ => ("completed", Value::Null),
        }
    }

    fn responses_body(&self) -> Value {
        let (status, incomplete) = self.responses_status();
        json!({
            "id": self.id, "object": "response", "created_at": 1_700_000_000, "status": status,
            "error": null, "incomplete_details": incomplete, "model": self.model,
            "output": self.responses_items(), "usage": self.responses_usage(),
        })
    }

    // --- streams ------------------------------------------------------------------------------

    /// The full upstream stream. `cut` stops it after that many events (a stream cut short);
    /// `error_at` puts an error event before the event with that index.
    pub fn stream(&self, d: Endpoint, cut: Option<usize>, error_at: Option<usize>) -> Vec<u8> {
        let events = match d {
            Endpoint::Messages => self.anthropic_events(),
            Endpoint::Responses => self.responses_events(),
            _ => self.chat_events(),
        };
        let mut out = Vec::new();
        for (i, (name, data)) in events.iter().enumerate() {
            if cut.is_some_and(|c| i >= c) {
                break;
            }
            if error_at == Some(i) {
                let (n, e) = error_event(d);
                self.framing.event(&mut out, n, &e);
            }
            self.framing.event(&mut out, name.as_deref(), data);
        }
        out
    }

    pub fn event_count(&self, d: Endpoint) -> usize {
        match d {
            Endpoint::Messages => self.anthropic_events().len(),
            Endpoint::Responses => self.responses_events().len(),
            _ => self.chat_events().len(),
        }
    }

    fn anthropic_events(&self) -> Vec<(Option<String>, String)> {
        let mut ev = Vec::new();
        let mut push = |name: &str, v: Value| ev.push((Some(name.to_owned()), v.to_string()));
        push(
            "message_start",
            json!({ "type": "message_start", "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": [], "stop_reason": null, "stop_sequence": null, "usage": self.anthropic_usage(true),
            }}),
        );
        push("ping", json!({ "type": "ping" }));
        let mut i = 0usize;
        for b in &self.blocks {
            let start = |block: Value| json!({ "type": "content_block_start", "index": i, "content_block": block });
            let delta = |d: Value| json!({ "type": "content_block_delta", "index": i, "delta": d });
            match b {
                Block::Server { id, query } => {
                    push(
                        "content_block_start",
                        start(
                            json!({ "type": "server_tool_use", "id": id, "name": "web_search", "input": {} }),
                        ),
                    );
                    push(
                        "content_block_delta",
                        delta(
                            json!({ "type": "input_json_delta", "partial_json": json!({ "query": query }).to_string() }),
                        ),
                    );
                    push(
                        "content_block_stop",
                        json!({ "type": "content_block_stop", "index": i }),
                    );
                    i += 1;
                    push(
                        "content_block_start",
                        json!({ "type": "content_block_start", "index": i, "content_block": {
                            "type": "web_search_tool_result", "tool_use_id": id, "content": [] } }),
                    );
                }
                Block::Text(p) => {
                    push(
                        "content_block_start",
                        start(json!({ "type": "text", "text": "" })),
                    );
                    for t in p {
                        push(
                            "content_block_delta",
                            delta(json!({ "type": "text_delta", "text": t })),
                        );
                    }
                }
                Block::Thinking { text, signature } => {
                    push(
                        "content_block_start",
                        start(json!({ "type": "thinking", "thinking": "", "signature": "" })),
                    );
                    for t in text {
                        push(
                            "content_block_delta",
                            delta(json!({ "type": "thinking_delta", "thinking": t })),
                        );
                    }
                    if let Some(sig) = signature {
                        push(
                            "content_block_delta",
                            delta(json!({ "type": "signature_delta", "signature": sig })),
                        );
                    }
                }
                Block::Redacted(d) => {
                    push(
                        "content_block_start",
                        start(json!({ "type": "redacted_thinking", "data": d })),
                    );
                }
                Block::Tool {
                    id,
                    name,
                    args,
                    cuts,
                } => {
                    push(
                        "content_block_start",
                        start(json!({ "type": "tool_use", "id": id, "name": name, "input": {} })),
                    );
                    for p in split_at(&args.to_string(), cuts) {
                        push(
                            "content_block_delta",
                            delta(json!({ "type": "input_json_delta", "partial_json": p })),
                        );
                    }
                }
            }
            push(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": i }),
            );
            i += 1;
        }
        push(
            "message_delta",
            json!({ "type": "message_delta",
                "delta": { "stop_reason": self.anthropic_stop(), "stop_sequence": null },
                "usage": self.anthropic_usage(false) }),
        );
        push("message_stop", json!({ "type": "message_stop" }));
        ev
    }

    fn chat_events(&self) -> Vec<(Option<String>, String)> {
        let mut ev = Vec::new();
        let nulls = self.chat_nulls;
        let chunk = |delta: Value, finish: Option<&str>| {
            let mut c = json!({ "id": self.id, "object": "chat.completion.chunk", "created": 1_700_000_000, "model": self.model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
            if nulls {
                c["usage"] = Value::Null;
            }
            c
        };
        let mut push = |v: Value| ev.push((None, v.to_string()));
        push(chunk(json!({ "role": "assistant", "content": "" }), None));
        let mut tool = 0u32;
        let mut i = 0;
        while i < self.blocks.len() {
            match &self.blocks[i] {
                Block::Text(p) => {
                    for t in p {
                        push(chunk(json!({ "content": t }), None));
                    }
                }
                Block::Thinking { text, signature } => {
                    for t in text {
                        push(chunk(
                            json!({ "reasoning": t, "reasoning_details": [{
                                "type": "reasoning.text", "text": t, "format": "anthropic-claude-v1", "index": i,
                            }] }),
                            None,
                        ));
                    }
                    if let Some(sig) = signature {
                        push(chunk(
                            json!({ "reasoning_details": [{
                                "type": "reasoning.text", "signature": sig, "format": "anthropic-claude-v1", "index": i,
                            }] }),
                            None,
                        ));
                    }
                }
                Block::Server { .. } => {}
                Block::Redacted(d) => push(chunk(
                    json!({ "reasoning_details": [{
                        "type": "reasoning.encrypted", "data": d, "format": "anthropic-claude-v1", "index": i,
                    }] }),
                    None,
                )),
                Block::Tool { .. } => {
                    // The run of consecutive calls starting here: opened, then their arguments,
                    // one call after another or interleaved round-robin.
                    let mut run = Vec::new();
                    while let Some(Block::Tool {
                        id,
                        name,
                        args,
                        cuts,
                    }) = self.blocks.get(i)
                    {
                        run.push((tool, id, name, split_at(&args.to_string(), cuts)));
                        tool += 1;
                        i += 1;
                    }
                    i -= 1;
                    let open = |(index, id, name): (u32, &String, &String)| {
                        let mut d = json!({ "tool_calls": [{ "index": index, "id": id, "type": "function",
                            "function": { "name": name, "arguments": "" } }] });
                        if nulls {
                            d["content"] = Value::Null;
                        }
                        d
                    };
                    let piece = |index: u32, p: &str| {
                        let mut d = json!({ "tool_calls": [{ "index": index, "function": { "arguments": p } }] });
                        if nulls {
                            d["content"] = Value::Null;
                        }
                        d
                    };
                    if self.interleave {
                        for (index, id, name, _) in &run {
                            push(chunk(open((*index, id, name)), None));
                        }
                        let most = run.iter().map(|r| r.3.len()).max().unwrap_or(0);
                        for k in 0..most {
                            for (index, _, _, pieces) in &run {
                                if let Some(p) = pieces.get(k) {
                                    push(chunk(piece(*index, p), None));
                                }
                            }
                        }
                    } else {
                        for (index, id, name, pieces) in &run {
                            push(chunk(open((*index, id, name)), None));
                            for p in pieces {
                                push(chunk(piece(*index, p), None));
                            }
                        }
                    }
                }
            }
            i += 1;
        }
        let mut finish = chunk(json!({}), Some(self.chat_finish()));
        if self.usage_on_finish {
            finish["usage"] = self.chat_usage();
            push(finish);
        } else {
            push(finish);
            push(
                json!({ "id": self.id, "object": "chat.completion.chunk", "created": 1_700_000_000,
                "model": self.model, "choices": [], "usage": self.chat_usage() }),
            );
        }
        ev.push((None, "[DONE]".to_owned()));
        ev
    }

    fn responses_events(&self) -> Vec<(Option<String>, String)> {
        let mut ev: Vec<(Option<String>, String)> = Vec::new();
        let mut seq = 0u64;
        let mut push = |name: &str, mut v: Value| {
            v["type"] = json!(name);
            v["sequence_number"] = json!(seq);
            seq += 1;
            ev.push((Some(name.to_owned()), v.to_string()));
        };
        let shell = json!({ "id": self.id, "object": "response", "created_at": 1_700_000_000,
            "status": "in_progress", "model": self.model, "output": [], "usage": null });
        push("response.created", json!({ "response": shell }));
        push("response.in_progress", json!({ "response": shell }));
        let items = self.responses_items();
        let mut k = 0usize;
        for (i, b) in self.blocks.iter().enumerate() {
            if matches!(b, Block::Redacted(_) | Block::Server { .. }) {
                continue;
            }
            let item = items[k].clone();
            let item_id = item["id"].as_str().unwrap().to_owned();
            match b {
                Block::Text(p) => {
                    push(
                        "response.output_item.added",
                        json!({ "output_index": k, "item": {
                        "type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": [] }}),
                    );
                    push(
                        "response.content_part.added",
                        json!({ "item_id": item_id, "output_index": k,
                        "content_index": 0, "part": { "type": "output_text", "text": "", "annotations": [] }}),
                    );
                    for t in p {
                        push(
                            "response.output_text.delta",
                            json!({ "item_id": item_id, "output_index": k,
                            "content_index": 0, "delta": t }),
                        );
                    }
                    push(
                        "response.output_text.done",
                        json!({ "item_id": item_id, "output_index": k,
                        "content_index": 0, "text": p.concat() }),
                    );
                    push(
                        "response.content_part.done",
                        json!({ "item_id": item_id, "output_index": k,
                        "content_index": 0, "part": { "type": "output_text", "text": p.concat(), "annotations": [] }}),
                    );
                }
                Block::Thinking { text, .. } => {
                    push(
                        "response.output_item.added",
                        json!({ "output_index": k, "item": {
                        "type": "reasoning", "id": item_id, "summary": [] }}),
                    );
                    push(
                        "response.reasoning_summary_part.added",
                        json!({ "item_id": item_id, "output_index": k,
                        "summary_index": 0, "part": { "type": "summary_text", "text": "" }}),
                    );
                    for t in text {
                        push(
                            "response.reasoning_summary_text.delta",
                            json!({ "item_id": item_id,
                            "output_index": k, "summary_index": 0, "delta": t }),
                        );
                    }
                    push(
                        "response.reasoning_summary_text.done",
                        json!({ "item_id": item_id,
                        "output_index": k, "summary_index": 0, "text": text.concat() }),
                    );
                }
                Block::Tool {
                    id,
                    name,
                    args,
                    cuts,
                } => {
                    push(
                        "response.output_item.added",
                        json!({ "output_index": k, "item": {
                        "type": "function_call", "id": item_id, "status": "in_progress", "arguments": "",
                        "call_id": id, "name": name }}),
                    );
                    for p in split_at(&args.to_string(), cuts) {
                        push(
                            "response.function_call_arguments.delta",
                            json!({ "item_id": item_id,
                            "output_index": k, "delta": p }),
                        );
                    }
                    push(
                        "response.function_call_arguments.done",
                        json!({ "item_id": item_id,
                        "output_index": k, "arguments": args.to_string() }),
                    );
                }
                Block::Redacted(_) | Block::Server { .. } => unreachable!(),
            }
            let _ = i;
            push(
                "response.output_item.done",
                json!({ "output_index": k, "item": item }),
            );
            k += 1;
        }
        let (status, incomplete) = self.responses_status();
        let terminal = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        push(
            terminal,
            json!({ "response": {
            "id": self.id, "object": "response", "created_at": 1_700_000_000, "status": status,
            "error": null, "incomplete_details": incomplete, "model": self.model,
            "output": items, "usage": self.responses_usage() }}),
        );
        ev
    }
}

/// An upstream error event in dialect `d`'s own shape.
pub fn error_event(d: Endpoint) -> (Option<&'static str>, String) {
    match d {
        Endpoint::Messages => (
            Some("error"),
            json!({ "type": "error", "error": { "type": "overloaded_error", "message": "Overloaded" } })
                .to_string(),
        ),
        Endpoint::Responses => (
            Some("error"),
            json!({ "type": "error", "code": "server_error", "message": "boom", "param": null, "sequence_number": 999 })
                .to_string(),
        ),
        _ => (
            None,
            json!({ "error": { "message": "boom", "type": "server_error", "code": "server_error" } })
                .to_string(),
        ),
    }
}

// --- reading what a client was sent -----------------------------------------------------------

/// What a client of one dialect was shown, gathered from its stream or body.
#[derive(Debug, Default, Clone)]
pub struct Seen {
    pub text: String,
    /// Tool calls in order: name and the argument JSON as sent.
    pub tools: Vec<(String, String)>,
    /// Replayable thinking blocks in Anthropic shape (Chat `thinking` list, Messages blocks).
    pub thinking: Vec<Value>,
    /// Responses reasoning items' `encrypted_content`.
    pub encrypted: Vec<String>,
    pub error: bool,
    /// The response said how it ended (finish reason / message_stop / terminal event).
    pub finished: bool,
}

pub fn seen_body(client: Endpoint, body: &[u8]) -> Result<Seen, String> {
    let v: Value =
        serde_json::from_slice(body).map_err(|e| format!("client body is not JSON: {e}"))?;
    let mut s = Seen::default();
    if v.get("error").is_some_and(|e| e.is_object()) || v.get("type") == Some(&json!("error")) {
        s.error = true;
        return Ok(s);
    }
    match client {
        Endpoint::ChatCompletions => {
            let m = &v["choices"][0]["message"];
            s.text = m["content"].as_str().unwrap_or("").to_owned();
            for c in m["tool_calls"].as_array().into_iter().flatten() {
                s.tools.push((
                    c["function"]["name"].as_str().unwrap_or("").to_owned(),
                    c["function"]["arguments"]
                        .as_str()
                        .unwrap_or("<not a string>")
                        .to_owned(),
                ));
            }
            for b in m["thinking"].as_array().into_iter().flatten() {
                s.thinking.push(b.clone());
            }
        }
        Endpoint::Messages => {
            for b in v["content"].as_array().into_iter().flatten() {
                match b["type"].as_str() {
                    Some("text") => s.text.push_str(b["text"].as_str().unwrap_or("")),
                    Some("tool_use") => s.tools.push((
                        b["name"].as_str().unwrap_or("").to_owned(),
                        b["input"].to_string(),
                    )),
                    Some("thinking" | "redacted_thinking") => s.thinking.push(b.clone()),
                    _ => {}
                }
            }
        }
        _ => {
            for item in v["output"].as_array().into_iter().flatten() {
                match item["type"].as_str() {
                    Some("message") => {
                        for p in item["content"].as_array().into_iter().flatten() {
                            s.text.push_str(p["text"].as_str().unwrap_or(""));
                        }
                    }
                    Some("function_call") => s.tools.push((
                        item["name"].as_str().unwrap_or("").to_owned(),
                        item["arguments"]
                            .as_str()
                            .unwrap_or("<not a string>")
                            .to_owned(),
                    )),
                    Some("reasoning") => {
                        if let Some(e) = item["encrypted_content"].as_str() {
                            s.encrypted.push(e.to_owned());
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    s.finished = true;
    Ok(s)
}

/// Read a client stream, checking its dialect's framing and lifecycle on the way. `Err` names the
/// first violation.
pub fn seen_stream(client: Endpoint, bytes: &[u8]) -> Result<Seen, String> {
    let events = parse_events(bytes)?;
    match client {
        Endpoint::ChatCompletions => seen_chat_stream(&events),
        Endpoint::Messages => seen_messages_stream(&events),
        _ => seen_responses_stream(&events),
    }
}

fn json_of(e: &Event) -> Result<Value, String> {
    serde_json::from_str(&e.data).map_err(|err| format!("event data is not JSON ({err}): {e:?}"))
}

fn seen_chat_stream(events: &[Event]) -> Result<Seen, String> {
    let mut s = Seen::default();
    let mut done = false;
    // Tool calls by index.
    let mut calls: Vec<(u64, String, String)> = Vec::new();
    let mut thinking: Vec<(u64, Value)> = Vec::new();
    for e in events {
        if done {
            return Err(format!("event after [DONE]: {e:?}"));
        }
        if e.name.is_some() {
            return Err(format!("named event on a Chat stream: {e:?}"));
        }
        if e.data == "[DONE]" {
            done = true;
            continue;
        }
        if s.error {
            return Err(format!("chunk after an error: {e:?}"));
        }
        let v = json_of(e)?;
        if v.get("error").is_some_and(|x| !x.is_null()) {
            s.error = true;
            continue;
        }
        if v["choices"][0]["finish_reason"].is_string() {
            s.finished = true;
        }
        let d = &v["choices"][0]["delta"];
        if let Some(t) = d["content"].as_str() {
            s.text.push_str(t);
        }
        for c in d["tool_calls"].as_array().into_iter().flatten() {
            let i = c["index"]
                .as_u64()
                .ok_or_else(|| format!("tool call delta without index: {c}"))?;
            let at = match calls.iter().position(|x| x.0 == i) {
                Some(at) => at,
                None => {
                    calls.push((i, String::new(), String::new()));
                    calls.len() - 1
                }
            };
            if let Some(n) = c["function"]["name"].as_str() {
                calls[at].1.push_str(n);
            }
            if let Some(a) = c["function"]["arguments"].as_str() {
                calls[at].2.push_str(a);
            }
        }
        for t in d["thinking"].as_array().into_iter().flatten() {
            let i = t["index"]
                .as_u64()
                .ok_or_else(|| format!("thinking entry without index: {t}"))?;
            let mut b = t.clone();
            b.as_object_mut().unwrap().remove("index");
            thinking.push((i, b));
        }
    }
    if !done {
        return Err("Chat stream without [DONE]".into());
    }
    s.tools = calls.into_iter().map(|(_, n, a)| (n, a)).collect();
    s.thinking = thinking.into_iter().map(|(_, b)| b).collect();
    Ok(s)
}

fn seen_messages_stream(events: &[Event]) -> Result<Seen, String> {
    let mut s = Seen::default();
    let mut started = false;
    let mut stopped = false;
    let mut delta_seen = false;
    let mut next_index = 0u64;
    // The open block: index and its content so far.
    let mut open: Option<(u64, Value)> = None;
    for e in events {
        let v = json_of(e)?;
        let name = e
            .name
            .as_deref()
            .ok_or_else(|| format!("unnamed event on a Messages stream: {e:?}"))?;
        if v["type"].as_str() != Some(name) {
            return Err(format!("event name {name} disagrees with its type: {e:?}"));
        }
        if stopped || s.error {
            return Err(format!("event after the stream ended: {e:?}"));
        }
        if name == "error" {
            s.error = true;
            continue;
        }
        if !started && name != "message_start" {
            return Err(format!("first event is {name}, not message_start"));
        }
        match name {
            "message_start" => {
                if started {
                    return Err("two message_start events".into());
                }
                started = true;
            }
            "content_block_start" => {
                if delta_seen {
                    return Err("content block after message_delta".into());
                }
                if let Some((i, _)) = &open {
                    return Err(format!("block {next_index} opened while block {i} is open"));
                }
                let i = v["index"].as_u64().ok_or("block start without index")?;
                if i != next_index {
                    return Err(format!("block index {i}, expected {next_index}"));
                }
                next_index += 1;
                open = Some((i, v["content_block"].clone()));
            }
            "content_block_delta" => {
                let i = v["index"].as_u64().ok_or("block delta without index")?;
                let Some((oi, block)) = open.as_mut() else {
                    return Err(format!("delta for block {i} with no block open"));
                };
                if *oi != i {
                    return Err(format!("delta for block {i} while block {oi} is open"));
                }
                let d = &v["delta"];
                match (block["type"].as_str(), d["type"].as_str()) {
                    (Some("text"), Some("text_delta")) => {
                        let t = d["text"].as_str().ok_or("text_delta without text")?;
                        s.text.push_str(t);
                    }
                    (Some("tool_use"), Some("input_json_delta")) => {
                        let p = d["partial_json"]
                            .as_str()
                            .ok_or("input_json_delta without partial_json")?;
                        let mut acc = block
                            .get("_args")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned();
                        acc.push_str(p);
                        block["_args"] = json!(acc);
                    }
                    (Some("thinking"), Some("thinking_delta")) => {
                        let t = d["thinking"]
                            .as_str()
                            .ok_or("thinking_delta without thinking")?;
                        let mut acc = block["thinking"].as_str().unwrap_or("").to_owned();
                        acc.push_str(t);
                        block["thinking"] = json!(acc);
                    }
                    (Some("thinking"), Some("signature_delta")) => {
                        let t = d["signature"]
                            .as_str()
                            .ok_or("signature_delta without signature")?;
                        let mut acc = block["signature"].as_str().unwrap_or("").to_owned();
                        acc.push_str(t);
                        block["signature"] = json!(acc);
                    }
                    (b, dt) => return Err(format!("delta {dt:?} inside a {b:?} block")),
                }
            }
            "content_block_stop" => {
                let i = v["index"].as_u64().ok_or("block stop without index")?;
                let Some((oi, block)) = open.take() else {
                    return Err(format!("stop for block {i} with no block open"));
                };
                if oi != i {
                    return Err(format!("stop for block {i} while block {oi} is open"));
                }
                match block["type"].as_str() {
                    Some("tool_use") => {
                        let args = block["_args"].as_str().unwrap_or("").to_owned();
                        s.tools
                            .push((block["name"].as_str().unwrap_or("").to_owned(), args));
                    }
                    Some("thinking") => {
                        if block["signature"].as_str().unwrap_or("").is_empty() {
                            return Err(format!(
                                "unsigned thinking block sent to a Messages client: {block}"
                            ));
                        }
                        s.thinking.push(block);
                    }
                    Some("redacted_thinking") => s.thinking.push(block),
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some((i, _)) = &open {
                    return Err(format!("message_delta while block {i} is open"));
                }
                if delta_seen {
                    return Err("two message_delta events".into());
                }
                delta_seen = true;
                if !v["delta"]["stop_reason"].is_string() {
                    return Err(format!("message_delta without a stop_reason: {v}"));
                }
            }
            "message_stop" => {
                if !delta_seen {
                    return Err("message_stop without message_delta".into());
                }
                stopped = true;
                s.finished = true;
            }
            "ping" => {}
            other => return Err(format!("unknown Messages event {other}")),
        }
    }
    if !stopped && !s.error {
        return Err("Messages stream ends without message_stop or error".into());
    }
    Ok(s)
}

fn seen_responses_stream(events: &[Event]) -> Result<Seen, String> {
    let mut s = Seen::default();
    let mut seq = 0u64;
    let mut terminal = false;
    // Items by output_index: id, type, done.
    let mut items: Vec<(String, String, bool)> = Vec::new();
    let mut args: Vec<String> = Vec::new();
    let mut errored_event = false;
    for (n, e) in events.iter().enumerate() {
        let v = json_of(e)?;
        let name = e
            .name
            .as_deref()
            .ok_or_else(|| format!("unnamed event on a Responses stream: {e:?}"))?;
        if v["type"].as_str() != Some(name) {
            return Err(format!("event name {name} disagrees with its type: {e:?}"));
        }
        if terminal {
            return Err(format!("event after the terminal event: {e:?}"));
        }
        match v["sequence_number"].as_u64() {
            Some(x) if x == seq => seq += 1,
            other => return Err(format!("sequence_number {other:?}, expected {seq}: {e:?}")),
        }
        match (n, name) {
            (0, "response.created") | (1, "response.in_progress") => continue,
            (0, _) => return Err(format!("first event is {name}")),
            (1, _) => return Err(format!("second event is {name}")),
            _ => {}
        }
        if errored_event && name != "response.failed" {
            return Err(format!("{name} after an error event"));
        }
        match name {
            "error" => {
                errored_event = true;
                s.error = true;
            }
            "response.failed" => {
                if !errored_event {
                    return Err("response.failed without an error event".into());
                }
                terminal = true;
            }
            "response.completed" | "response.incomplete" => {
                terminal = true;
                s.finished = true;
                let out = v["response"]["output"]
                    .as_array()
                    .ok_or("terminal response without output")?;
                if let Some((i, _)) = items.iter().enumerate().find(|(_, it)| !it.2) {
                    return Err(format!("{name} with item {i} never done"));
                }
                let ids: Vec<&str> = out.iter().map(|i| i["id"].as_str().unwrap_or("")).collect();
                let want: Vec<&str> = items.iter().map(|i| i.0.as_str()).collect();
                if ids != want {
                    return Err(format!("terminal output ids {ids:?}, items were {want:?}"));
                }
            }
            "response.output_item.added" => {
                let i = v["output_index"]
                    .as_u64()
                    .ok_or("item added without output_index")?;
                if i != items.len() as u64 {
                    return Err(format!("item added at {i}, expected {}", items.len()));
                }
                let id = v["item"]["id"].as_str().ok_or("item added without an id")?;
                items.push((
                    id.to_owned(),
                    v["item"]["type"].as_str().unwrap_or("").to_owned(),
                    false,
                ));
                args.push(String::new());
            }
            "response.output_item.done" => {
                let i = v["output_index"]
                    .as_u64()
                    .ok_or("item done without output_index")? as usize;
                let it = items
                    .get_mut(i)
                    .ok_or_else(|| format!("done for unknown item {i}"))?;
                if it.2 {
                    return Err(format!("item {i} done twice"));
                }
                if v["item"]["id"].as_str() != Some(it.0.as_str()) {
                    return Err(format!("item {i} done under another id: {v}"));
                }
                it.2 = true;
                let item = &v["item"];
                match item["type"].as_str() {
                    Some("function_call") => {
                        let a = item["arguments"]
                            .as_str()
                            .ok_or("function_call without arguments")?;
                        if a != args[i] {
                            return Err(format!(
                                "item {i} done with arguments {a:?}, deltas said {:?}",
                                args[i]
                            ));
                        }
                        s.tools
                            .push((item["name"].as_str().unwrap_or("").to_owned(), a.to_owned()));
                    }
                    Some("reasoning") => {
                        if let Some(e) = item["encrypted_content"].as_str() {
                            s.encrypted.push(e.to_owned());
                        }
                    }
                    _ => {}
                }
            }
            _ => {
                let i = v["output_index"]
                    .as_u64()
                    .ok_or_else(|| format!("{name} without output_index"))?
                    as usize;
                let id = v["item_id"]
                    .as_str()
                    .ok_or_else(|| format!("{name} without item_id"))?;
                let it = items
                    .get(i)
                    .ok_or_else(|| format!("{name} for unknown item {i}"))?;
                if it.0 != id {
                    return Err(format!(
                        "{name} names item {id}, output_index {i} is {}",
                        it.0
                    ));
                }
                if it.2 {
                    return Err(format!("{name} after item {i} was done"));
                }
                match name {
                    "response.output_text.delta" => {
                        s.text.push_str(v["delta"].as_str().unwrap_or(""))
                    }
                    "response.function_call_arguments.delta" => {
                        args[i].push_str(v["delta"].as_str().unwrap_or(""))
                    }
                    _ => {}
                }
            }
        }
    }
    if !terminal {
        return Err("Responses stream without a terminal event".into());
    }
    Ok(s)
}
