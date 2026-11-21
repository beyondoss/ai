//! Stream: Chat Completions chunks → Messages events.

use super::*;

/// How far a translator has written its client's stream. `start` opens it (`message_start`,
/// `response.created`); the terminal event or an error closes it, and nothing follows a close.
/// A Responses client assembled into one body can close straight from `Unstarted`: its error
/// writes no event, so there is nothing to open.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Phase {
    #[default]
    Unstarted,
    Open,
    Closed,
}

impl Phase {
    /// `Unstarted` → `Open` only: a closed stream never reopens, whoever calls `start`.
    pub(super) fn open(&mut self) {
        debug_assert!(
            *self != Self::Closed,
            "start after the client's terminal event"
        );
        if *self == Self::Unstarted {
            *self = Self::Open;
        }
    }
}

/// Chat Completions items → Anthropic Messages events.
#[derive(Default)]
pub(super) struct OaiToAnt {
    pub(super) phase: Phase,
    id: String,
    model: String,
    next_block: usize,
    /// The open text block's index. Anthropic blocks are sequential: anything else closes it.
    text_block: Option<usize>,
    pub(super) calls: ToolCalls,
    /// The call whose `tool_use` block is open (an index into `calls`). Parallel calls stream one
    /// block at a time: Anthropic never opens a block before closing the last.
    live: Option<usize>,
    /// Calls ready to show while another is live, in order. Their arguments gather in
    /// [`ChatCall::args`] until the live block closes, which it does only once its arguments are
    /// complete JSON (nothing more can follow) or the stream ends: an upstream may interleave
    /// argument deltas across calls.
    queued: Vec<usize>,
    /// How far the live call's arguments have been scanned for their end (see [`ArgsEnd`]).
    live_end: ArgsEnd,
    /// Thinking held back until its signature arrives. See [`is_replayable_thinking`].
    thinking: Gather,
    stop: Option<&'static str>,
    /// The model refused in words (`refusal`), which Anthropic reports as a stop reason. Not a
    /// `phase`: it is a fact about the answer, read when the stream closes.
    refused: bool,
    usage: Option<Usage>,
}

/// Where a streamed call's JSON arguments end, found incrementally: each step scans only the bytes
/// that arrived since the last, so a live call with queued calls behind it costs O(len) overall,
/// not a full re-parse per step. The one full parse runs when the outer value closes, to
/// confirm it is whole JSON.
#[derive(Default)]
pub(in crate::translate) struct ArgsEnd {
    scanned: usize,
    lex: crate::lex::Lexer,
    /// The outer object, array or string has closed.
    closed: bool,
    /// It closed and the parse said it is not JSON: no byte after a closed value can fix that.
    pub(in crate::translate) invalid: bool,
    /// The argument length at which the last full parse failed. Asked again with nothing new (a
    /// queued call's delta asks about the live call), the answer is the same, unparsed.
    pub(in crate::translate) failed_at: Option<usize>,
}

/// Stops the lex where the outer value closes: its last bracket, or a string's closing quote.
struct OuterClose;

impl crate::lex::Visit for OuterClose {
    #[inline(always)]
    fn byte(&mut self, _: usize, c: u8, depth: u32) -> crate::lex::Flow {
        // The first byte opened the value, so the bracket back at depth 1 is the one closing it.
        if matches!(c, b'}' | b']') && depth == 1 {
            crate::lex::Flow::Break(())
        } else {
            crate::lex::Flow::Continue(())
        }
    }

    #[inline(always)]
    fn close(&mut self, _: usize, depth: u32, _: bool) -> crate::lex::Flow {
        // A string argument closes with its quote.
        if depth == 0 {
            crate::lex::Flow::Break(())
        } else {
            crate::lex::Flow::Continue(())
        }
    }
}

impl ArgsEnd {
    /// Whether `args` (the live call's whole argument text so far) is complete JSON.
    pub(in crate::translate) fn whole(&mut self, args: &str) -> bool {
        if self.invalid || self.failed_at == Some(args.len()) {
            return false;
        }
        let b = args.as_bytes();
        let first = b.iter().position(|c| !c.is_ascii_whitespace());
        match first.and_then(|i| b.get(i)) {
            None => return false,
            // A number or literal: whole once it parses (`tru` is not yet, `true` is).
            Some(&c) if !matches!(c, b'{' | b'[' | b'"') => return self.parses(args),
            Some(_) => {}
        }
        if !self.closed {
            let rest = b.get(self.scanned..).unwrap_or_default();
            match self.lex.feed(rest, &mut OuterClose) {
                // Once closed, `scanned` is never read again.
                Some(_) => self.closed = true,
                None => self.scanned = b.len(),
            }
        }
        if !self.closed {
            return false;
        }
        let whole = self.parses(args);
        self.invalid = !whole;
        whole
    }

    /// One full parse of `args`, remembering a failure for its length.
    fn parses(&mut self, args: &str) -> bool {
        let ok = serde_json::from_str::<serde::de::IgnoredAny>(args).is_ok();
        if !ok {
            self.failed_at = Some(args.len());
        }
        ok
    }
}

fn ant_event(out: &mut Vec<u8>, typ: &str, body: &Value) {
    out.extend(sse_named(typ, &value_string(body)));
}

impl OaiToAnt {
    pub(super) fn chunk(&mut self, v: &Value, out: &mut Vec<u8>) {
        if self.phase == Phase::Closed {
            return;
        }
        if self.phase == Phase::Unstarted {
            if let Some(id) = non_empty_str(v, "id") {
                id.clone_into(&mut self.id);
            }
            if let Some(model) = non_empty_str(v, "model") {
                model.clone_into(&mut self.model);
            }
            self.start(out);
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|a| a.first());
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            let mut blocks = Vec::new();
            // OpenRouter repeats `reasoning_details` text in `reasoning`: read one, not both.
            if let Some(details) = delta
                .get("reasoning_details")
                .and_then(Value::as_array)
                .filter(|d| !d.is_empty())
            {
                for d in details {
                    self.thinking.detail(d, &mut blocks);
                }
            } else if let Some(r) = non_empty_str(delta, "reasoning_content")
                .or_else(|| non_empty_str(delta, "reasoning"))
            {
                self.thinking.text(None, r, &mut blocks);
            }
            // Whole blocks on the `thinking` list (a Messages upstream read by `AntToOai`): the
            // reasoning text streamed just before is the same block's, now signed.
            for b in delta
                .get("thinking")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                let block = anthropic_thinking_block(b);
                if is_replayable_thinking(&block) {
                    self.thinking.discard();
                    blocks.push(block);
                }
            }
            for b in blocks.iter().filter(|b| is_replayable_thinking(b)) {
                self.thinking_block(b, out);
            }
            if let Some(t) = non_empty_str(delta, "content") {
                self.text(t, out);
            }
            if let Some(r) = non_empty_str(delta, "refusal") {
                self.refused = true;
                self.text(r, out);
            }
            for c in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
            {
                self.tool(c, out);
            }
        }
        if let Some(finish) = choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str)
            .filter(|f| !f.is_empty())
        {
            let native = choice
                .and_then(|c| c.get("native_finish_reason"))
                .and_then(Value::as_str)
                .and_then(anthropic_stop_reason);
            self.stop = Some(native.unwrap_or_else(|| map_stop_to_anthropic(Some(finish))));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(Usage::from_chat(u));
        }
        if self.stop.is_some() && self.usage.is_some() {
            self.finish(out);
        }
    }

    fn start(&mut self, out: &mut Vec<u8>) {
        self.phase.open();
        if self.id.is_empty() {
            self.id = fresh_id("msg");
        }
        ant_event(
            out,
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 },
                }
            }),
        );
    }

    fn next_index(&mut self) -> usize {
        let idx = self.next_block;
        self.next_block = self.next_block.saturating_add(1);
        idx
    }

    fn close_text(&mut self, out: &mut Vec<u8>) {
        if let Some(idx) = self.text_block.take() {
            block_stop(out, idx);
        }
    }

    /// Close the live block and show every queued call after it, each start to stop.
    fn close_calls(&mut self, out: &mut Vec<u8>) {
        self.close_live(out);
        for at in std::mem::take(&mut self.queued) {
            self.show_call(at, out);
            self.close_live(out);
        }
    }

    fn close_live(&mut self, out: &mut Vec<u8>) {
        if let Some(call) = self.live.take().and_then(|at| self.calls.calls.get_mut(at)) {
            call.closed = true;
            block_stop(out, call.slot);
        }
    }

    /// Open `at`'s block with every argument byte gathered so far; it becomes the live call.
    fn show_call(&mut self, at: usize, out: &mut Vec<u8>) {
        let idx = self.next_index();
        let Some(call) = self.calls.calls.get_mut(at) else {
            return;
        };
        call.slot = idx;
        block_start(
            out,
            idx,
            &json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": {} }),
        );
        if !call.args.is_empty() {
            block_delta(
                out,
                idx,
                &json!({ "type": "input_json_delta", "partial_json": call.args }),
            );
        }
        self.live = Some(at);
        self.live_end = ArgsEnd::default();
    }

    /// While calls wait, close the live block once its arguments are whole and show the next.
    fn advance_calls(&mut self, out: &mut Vec<u8>) {
        while !self.queued.is_empty() {
            if !self.live_whole() {
                return;
            }
            self.close_live(out);
            let at = self.queued.remove(0);
            self.show_call(at, out);
        }
    }

    /// Whether the live call can take no more arguments. Its JSON has closed; or it has none and
    /// the upstream is already streaming a queued call's arguments, so it is a call that takes no
    /// arguments, not one whose arguments are late.
    fn live_whole(&mut self) -> bool {
        let Some(c) = self.live.and_then(|at| self.calls.calls.get(at)) else {
            return true;
        };
        if c.args.trim_ascii().is_empty() {
            return self
                .queued
                .iter()
                .any(|&q| self.calls.calls.get(q).is_some_and(|q| !q.args.is_empty()));
        }
        self.live_end.whole(&c.args)
    }

    /// A whole thinking block, signed or redacted, emitted at once.
    fn thinking_block(&mut self, b: &Value, out: &mut Vec<u8>) {
        self.close_text(out);
        self.close_calls(out);
        let idx = self.next_index();
        if b.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
            block_start(out, idx, b);
            block_stop(out, idx);
            return;
        }
        block_start(
            out,
            idx,
            &json!({ "type": "thinking", "thinking": "", "signature": "" }),
        );
        if let Some(text) = non_empty_str(b, "thinking") {
            block_delta(
                out,
                idx,
                &json!({ "type": "thinking_delta", "thinking": text }),
            );
        }
        if let Some(sig) = non_empty_str(b, "signature") {
            block_delta(
                out,
                idx,
                &json!({ "type": "signature_delta", "signature": sig }),
            );
        }
        block_stop(out, idx);
    }

    fn text(&mut self, text: &str, out: &mut Vec<u8>) {
        // Text ends any thinking block still waiting for a signature: it never gets one now.
        self.thinking.discard();
        self.close_calls(out);
        let idx = match self.text_block {
            Some(idx) => idx,
            None => {
                let idx = self.next_index();
                self.text_block = Some(idx);
                block_start(out, idx, &json!({ "type": "text", "text": "" }));
                idx
            }
        };
        block_delta(out, idx, &json!({ "type": "text_delta", "text": text }));
    }

    fn tool(&mut self, c: &Value, out: &mut Vec<u8>) {
        self.thinking.discard();
        self.close_text(out);
        let mut steps = Vec::new();
        self.calls.feed(c, &mut steps);
        self.call_steps(steps, out);
    }

    fn call_steps(&mut self, steps: Vec<CallStep>, out: &mut Vec<u8>) {
        for step in steps {
            match step {
                CallStep::Open(at) => {
                    if self.live.is_none() {
                        self.show_call(at, out);
                    } else {
                        self.queued.push(at);
                    }
                }
                // A queued call's arguments are already in its `args`, shown when it opens.
                CallStep::Args(at, args) => {
                    if self.live == Some(at)
                        && let Some(call) = self.calls.calls.get(at)
                    {
                        block_delta(
                            out,
                            call.slot,
                            &json!({ "type": "input_json_delta", "partial_json": args }),
                        );
                    }
                }
            }
            self.advance_calls(out);
        }
    }

    pub(super) fn finish(&mut self, out: &mut Vec<u8>) {
        if self.phase == Phase::Closed {
            return;
        }
        // A usage-only / `[DONE]`-only upstream stream never opened a block; the client still
        // needs `message_start` before `message_delta` / `message_stop`.
        if self.phase == Phase::Unstarted {
            self.start(out);
        }
        self.phase = Phase::Closed;
        self.thinking.discard();
        let mut steps = Vec::new();
        self.calls.open_rest(&mut steps);
        if !steps.is_empty() {
            self.close_text(out);
        }
        self.call_steps(steps, out);
        self.close_text(out);
        self.close_calls(out);
        let stop = finish_anthropic_stop(
            self.stop.unwrap_or("end_turn"),
            self.refused,
            !self.calls.is_empty(),
        );
        ant_event(
            out,
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop, "stop_sequence": Value::Null },
                "usage": self.usage.unwrap_or_default().to_anthropic(),
            }),
        );
        ant_event(out, "message_stop", &json!({ "type": "message_stop" }));
    }
}

fn block_start(out: &mut Vec<u8>, index: usize, block: &Value) {
    ant_event(
        out,
        "content_block_start",
        &json!({ "type": "content_block_start", "index": index, "content_block": block }),
    );
}

fn block_delta(out: &mut Vec<u8>, index: usize, delta: &Value) {
    ant_event(
        out,
        "content_block_delta",
        &json!({ "type": "content_block_delta", "index": index, "delta": delta }),
    );
}

fn block_stop(out: &mut Vec<u8>, index: usize) {
    ant_event(
        out,
        "content_block_stop",
        &json!({ "type": "content_block_stop", "index": index }),
    );
}
