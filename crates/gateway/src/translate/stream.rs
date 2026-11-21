//! SSE: the bridge that translates a streamed response, and the framing it reads and writes.

use crate::translate::*;

mod ant_to_oai;
mod oai_to_ant;
mod oai_to_resp;
mod resp_to_oai;
mod tool_calls;

pub(super) use self::ant_to_oai::*;
pub(super) use self::oai_to_ant::*;
pub(super) use self::oai_to_resp::*;
use self::resp_to_oai::*;
pub(super) use self::tool_calls::*;

/// Event-by-event SSE translator. Incomplete events stay in `buf`; complete events are mapped
/// immediately. Does not wait for `[DONE]` before forwarding deltas.
///
/// Every pairing meets in Chat Completions chunks: a Messages or Responses upstream is read into
/// [`ChatItem`]s, and a Messages or Responses client is written from them. The middle is values,
/// not bytes, so a composed pairing (Messages → Responses) parses each event once.
///
/// With `client == upstream` it is a relay: events pass through as they came, except a Chat
/// Completions chunk that repeats an identity field (see [`ChatIdentity`]).
pub struct SseBridge {
    client: Endpoint,
    upstream: Endpoint,
    identity: ChatIdentity,
    pub(super) buf: Vec<u8>,
    /// Bytes of `buf` already searched for an event end, so a long unterminated event is scanned
    /// once rather than from the start on every chunk.
    scanned: usize,
    ant_to_oai: AntToOai,
    resp_to_oai: RespToOai,
    oai_to_ant: OaiToAnt,
    pub(super) oai_to_resp: OaiToResp,
    lifecycle: Lifecycle,
    /// A Chat client's `[DONE]` went out. Apart from `lifecycle` because it follows both of its
    /// ends: the upstream's own `[DONE]` after a clean end, and flush's after an error.
    done: bool,
}

/// How far the upstream's account of the response has got. The order is one-way: an error is
/// itself an end, so nothing returns to `Open` and nothing leaves `Errored`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lifecycle {
    Open,
    /// The upstream said how the response ended: a finish reason or its own end marker (or an error
    /// that came after the client's terminal event, so never reached it). A stream that ends while
    /// `Open` was cut short, and flush reports that instead of inventing a clean `end_turn` /
    /// `completed`.
    Ended,
    /// An error went to the client. Nothing after it is forwarded, and flush must not invent a
    /// success close (`message_stop`, `response.completed`); a Chat client still gets its `[DONE]`.
    Errored,
}

impl Lifecycle {
    /// The upstream said how the response ends; an error already sent stays the outcome.
    fn end(&mut self) {
        if *self == Self::Open {
            *self = Self::Ended;
        }
    }
}

/// One Chat Completions stream item, the dialect every pairing meets in.
enum ChatItem {
    Chunk(Value),
    /// The upstream's error body as it came; each client shapes its own envelope from it.
    Error(Value),
    Done,
}

impl ChatItem {
    /// Whether this item tells the client how the response ended.
    fn ends(&self) -> bool {
        match self {
            Self::Chunk(v) => v
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
                .is_some_and(|f| !f.is_empty()),
            Self::Error(_) | Self::Done => true,
        }
    }
}

/// The identity fields of a Chat Completions stream: `delta.role`, a `reasoning_details` entry's
/// `id` and `format`, a tool call's `id` and name. OpenAI sends each once, in the first chunk that
/// carries its choice or entry, and stock SDKs accumulate every other string as a delta
/// (openai-python's `accumulate_delta` concatenates all but `index` and `type`). OpenRouter
/// repeats `role` on every chunk and `format` on every reasoning entry, so a relayed stream left
/// the client a message whose role was "assistantassistant…" for its next turn to send back. This
/// drops a repeat.
#[derive(Default)]
pub(in crate::translate) struct ChatIdentity {
    /// One packed key per field already sent: choice, list, entry index, field.
    sent: Vec<u64>,
}

impl ChatIdentity {
    const ROLE: u64 = 0;
    const REASONING_ID: u64 = 1;
    const REASONING_FORMAT: u64 = 2;
    const CALL_ID: u64 = 3;
    const CALL_NAME: u64 = 4;

    /// Drop the identity fields this stream already sent. `true` when anything was dropped.
    pub(in crate::translate) fn strip(&mut self, chunk: &mut Value) -> bool {
        let Some(choices) = chunk.get_mut("choices").and_then(Value::as_array_mut) else {
            return false;
        };
        let mut dropped = false;
        for (pos, choice) in choices.iter_mut().enumerate() {
            let c = index_of(choice, pos);
            let Some(delta) = choice.get_mut("delta").and_then(Value::as_object_mut) else {
                continue;
            };
            dropped |= self.once(delta, "role", c, 0, Self::ROLE);
            if let Some(entries) = delta
                .get_mut("reasoning_details")
                .and_then(Value::as_array_mut)
            {
                for (pos, entry) in entries.iter_mut().enumerate() {
                    let i = index_of(entry, pos);
                    if let Some(entry) = entry.as_object_mut() {
                        dropped |= self.once(entry, "id", c, i, Self::REASONING_ID);
                        dropped |= self.once(entry, "format", c, i, Self::REASONING_FORMAT);
                    }
                }
            }
            if let Some(calls) = delta.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for (pos, call) in calls.iter_mut().enumerate() {
                    let i = index_of(call, pos);
                    let Some(call) = call.as_object_mut() else {
                        continue;
                    };
                    dropped |= self.once(call, "id", c, i, Self::CALL_ID);
                    // A custom tool's name rides `custom`, the way a function's rides `function`.
                    for body in ["function", "custom"] {
                        if let Some(body) = call.get_mut(body).and_then(Value::as_object_mut) {
                            dropped |= self.once(body, "name", c, i, Self::CALL_NAME);
                        }
                    }
                }
            }
        }
        dropped
    }

    /// (choice, entry, kind) packed into disjoint bit ranges: 24 bits, 32 bits, 8 bits.
    fn key(choice: u64, entry: u64, kind: u64) -> u64 {
        (choice & 0xff_ffff) << 40 | (entry & 0xffff_ffff) << 8 | kind
    }

    /// The packed key for (choice, entry, kind), when neither `sent` nor `pending` holds it yet:
    /// the field is new and should be kept (and recorded). `None` for a repeat.
    fn seen(sent: &[u64], pending: &[u64], choice: u64, entry: u64, kind: u64) -> Option<u64> {
        let key = Self::key(choice, entry, kind);
        (!sent.contains(&key) && !pending.contains(&key)).then_some(key)
    }

    /// Keep `obj[field]` the first time this stream carries it for (choice, entry); drop it after.
    fn once(
        &mut self,
        obj: &mut Map<String, Value>,
        field: &str,
        choice: u64,
        entry: u64,
        kind: u64,
    ) -> bool {
        if !obj.get(field).is_some_and(Value::is_string) {
            return false;
        }
        let key = Self::key(choice, entry, kind);
        if self.sent.contains(&key) {
            obj.remove(field);
            return true;
        }
        self.sent.push(key);
        false
    }
}

/// An item's own `index`, else its position in the list.
fn index_of(item: &Value, pos: usize) -> u64 {
    item.get("index")
        .and_then(Value::as_u64)
        .unwrap_or(pos as u64)
}

/// An SSE comment line, sent to a translated stream's client in place of upstream events that
/// map to nothing for it: Anthropic's `event: ping`, OpenRouter's `: OPENROUTER PROCESSING`, and
/// thinking or a reasoning summary the client's dialect does not carry, or holds until its block
/// is whole. Dropping them left the client no byte for as long as the model thought, and a
/// load balancer's or proxy's idle timeout (often 60–100 s) cut the stream. Every SSE parser
/// ignores a line that starts with `:`, so the client's events are unchanged. At most one per
/// [`SseBridge::feed`] call that consumed an event and wrote nothing, so the cadence is the
/// provider's own (Anthropic pings every few seconds while it thinks) and needs no timer.
pub const KEEP_ALIVE: &[u8] = b": keep-alive\n\n";

/// What a client is told when the upstream stream ended without saying how.
fn truncated_stream_error() -> Value {
    json!({ "error": {
        "message": "The upstream stream ended before the response was complete.",
        "type": "api_error",
        "code": "stream_truncated",
    }})
}

impl SseBridge {
    pub fn new(client: Endpoint, upstream: Endpoint) -> Self {
        Self {
            client,
            upstream,
            identity: ChatIdentity::default(),
            buf: Vec::new(),
            scanned: 0,
            ant_to_oai: AntToOai::default(),
            resp_to_oai: RespToOai::default(),
            oai_to_ant: OaiToAnt::default(),
            oai_to_resp: OaiToResp::default(),
            lifecycle: Lifecycle::Open,
            done: false,
        }
    }

    /// Name each tool call the way the request offered it to a Responses client (see
    /// [`ToolNames`]).
    #[must_use]
    pub fn with_tools(mut self, tools: ToolNames) -> Self {
        self.oai_to_resp.tools = tools;
        self
    }

    /// Show the response's cache writes as input: the request's breakpoints were the gateway's
    /// (see [`request_with_tools`]). Only a Messages upstream reports cache writes this way.
    #[must_use]
    pub fn with_gateway_cache(mut self, on: bool) -> Self {
        self.ant_to_oai.gateway_cache = on;
        self
    }

    /// A bridge that assembles an upstream stream into one non-streaming answer instead of
    /// relaying it, for a client that did not ask for a stream from a candidate that answers only
    /// streams (`providers::catalog::stream_only`). It reads the stream the way a Responses
    /// client's bridge does, since that already builds the whole response (`response.completed`
    /// repeats every output item and the usage), but writes no event: [`Self::feed`] returns
    /// nothing, and [`Self::assembled`] gives the body once the upstream has ended. Reasoning,
    /// refusals, tool calls (arguments as the model wrote them) and the finish reason come through
    /// as a Responses client's stream would carry them. Not for a Responses upstream, which a
    /// Responses client's bridge relays.
    pub fn assembling(upstream: Endpoint) -> Self {
        debug_assert_ne!(upstream, Endpoint::Responses);
        let mut bridge = Self::new(Endpoint::Responses, upstream);
        bridge.oai_to_resp.quiet = true;
        bridge
    }

    /// The `client`'s non-streaming body for an [`Self::assembling`] bridge whose upstream has
    /// ended: the finished response mapped like any non-stream Responses body, or an error in the
    /// client's envelope (the upstream's own, or `stream_truncated` for a stream that ended without
    /// saying how). The status line has already gone out as the upstream's 200, so an error comes
    /// with it, as an error-in-200 from a non-streaming upstream does.
    pub fn assembled(&mut self, client: Endpoint) -> Vec<u8> {
        match self.oai_to_resp.whole.take() {
            Some(Whole::Done(resp)) if client == Endpoint::Responses => encode(&resp),
            Some(Whole::Done(resp)) => encode(&map_response(
                Endpoint::Responses,
                client,
                &resp,
                &ToolNames::default(),
                false,
            )),
            Some(Whole::Failed(err)) => encode(&map_error(&err, client, 200)),
            None => encode(&map_error(&truncated_stream_error(), client, 200)),
        }
    }

    /// Bytes held for an event whose terminating blank line has not arrived yet.
    pub fn pending_len(&self) -> usize {
        self.buf.len()
    }

    /// Output this bridge has gathered so far and holds to the end of the stream: text, reasoning
    /// and tool arguments for a Responses client's closing event, and tool calls waiting their
    /// turn onto Messages. Counted as it is gathered; see [`MAX_STREAM_OUTPUT`].
    pub fn accumulated(&self) -> usize {
        self.oai_to_ant
            .calls
            .held
            .saturating_add(self.oai_to_resp.held)
            .saturating_add(self.oai_to_resp.calls.held)
            .saturating_add(self.resp_to_oai.held)
    }

    /// Which of this bridge's bounds the stream has crossed, if any: an unterminated event past
    /// [`MAX_TRANSLATE_BUFFER`] (`"sse_event"`), or gathered output past [`MAX_STREAM_OUTPUT`]
    /// (`"sse_output"`).
    pub fn overflow(&self) -> Option<&'static str> {
        if self.pending_len() > MAX_TRANSLATE_BUFFER {
            Some("sse_event")
        } else if self.accumulated() > MAX_STREAM_OUTPUT {
            Some("sse_output")
        } else {
            None
        }
    }

    /// Feed upstream SSE bytes. Returns client-dialect SSE bytes (possibly empty).
    ///
    /// Each complete event is mapped where it lies in the buffer, and the buffer is compacted once
    /// per call: draining it per event moved the unread tail once per event, O(events x bytes)
    /// for a chunk that carries many.
    pub fn feed(&mut self, data: &[u8], end: bool) -> Vec<u8> {
        let mut buf = std::mem::take(&mut self.buf);
        buf.extend_from_slice(data);
        let mut out = Vec::new();
        let mut start = 0;
        let mut consumed = false;
        while let Some((from, to)) = next_event(&buf, &mut start, &mut self.scanned) {
            consumed = true;
            self.map_event(buf.get(from..to).unwrap_or_default(), &mut out);
        }
        if consumed && !end && out.is_empty() && self.keeps_alive() {
            out.extend_from_slice(KEEP_ALIVE);
        }
        buf.drain(..start);
        self.scanned = self.scanned.saturating_sub(start);
        if end && !buf.is_empty() {
            self.scanned = 0;
            self.map_event(&buf, &mut out);
            buf.clear();
        }
        self.buf = buf;
        if end {
            out.extend(self.flush());
        }
        out
    }

    /// Whether a translated stream owes its client a [`KEEP_ALIVE`] for upstream events that
    /// mapped to nothing. A same-wire relay forwards the provider's own pings and comments
    /// as they came; an assembling bridge writes no event; and nothing follows an error or the
    /// client's terminal event.
    fn keeps_alive(&self) -> bool {
        self.upstream != self.client
            && !self.oai_to_resp.quiet
            && self.lifecycle != Lifecycle::Errored
            && !self.client_ended()
    }

    fn map_event(&mut self, raw: &[u8], out: &mut Vec<u8>) {
        if self.upstream == self.client {
            self.relay(raw, out);
            return;
        }
        let (event, data) = parse_sse(raw);
        if data.is_empty() && event.is_empty() {
            return;
        }
        let mut items = Vec::new();
        match self.upstream {
            Endpoint::Messages => self.ant_to_oai.event(&event, &data, &mut items),
            Endpoint::Responses => self.resp_to_oai.event(&event, &data, &mut items),
            Endpoint::ChatCompletions => chat_items(&data, &mut items),
            Endpoint::Embeddings => {}
        }
        for item in items {
            if item.ends() {
                self.lifecycle.end();
            }
            self.deliver(item, out);
        }
    }

    /// A relayed event, as it came unless it is a chunk that repeats an identity field.
    ///
    /// Only the fields [`ChatIdentity`] edits can change an event, so one that names none of them
    /// as a key (`"role"`, `"tool_calls"`, `"reasoning_details"`) is copied through unparsed, as
    /// the relay was before it learned to drop repeats; one whose only such field is `role` (every
    /// OpenRouter chunk) has a repeat cut by span ([`Self::relay_role`]); anything else goes
    /// through a `Value`. A key spelled with escapes is not seen; no provider sends one.
    fn relay(&mut self, raw: &[u8], out: &mut Vec<u8>) {
        let has = |needle: &[u8]| memchr::memmem::find(raw, needle).is_some();
        let lists = has(b"\"tool_calls\"") || has(b"\"reasoning_details\"");
        if !lists && !has(b"\"role\"") {
            out.extend_from_slice(raw);
            return;
        }
        if !lists && self.relay_role(raw, out).is_some() {
            return;
        }
        let (event, data) = parse_sse(raw);
        if event.is_empty()
            && let Ok(mut v) = serde_json::from_str::<Value>(&data)
            && self.identity.strip(&mut v)
        {
            out.extend(sse_data(&value_string(&v)));
            return;
        }
        out.extend_from_slice(raw);
    }

    /// [`ChatIdentity::strip`] for a chunk whose only identity field is `delta.role`, by span: the
    /// first `role` per choice is recorded, a repeat is cut out of the event's own bytes, and
    /// every other byte goes to `out` as sent. `None`, with nothing written or recorded, for any
    /// shape this does not handle (more than one `data:` line, another field line, a duplicate
    /// key, malformed JSON): the caller takes the `Value` path.
    fn relay_role(&mut self, raw: &[u8], out: &mut Vec<u8>) -> Option<()> {
        let rest = raw.strip_prefix(b"data:")?;
        let line_end = memchr::memchr(b'\n', rest).unwrap_or(rest.len());
        let (line, tail) = rest.split_at(line_end);
        if !tail.iter().all(|&c| c == b'\n' || c == b'\r') {
            return None;
        }
        let skip = line.iter().take_while(|c| c.is_ascii_whitespace()).count();
        let base = raw.len() - rest.len() + skip;
        let json = raw.get(base..base + line_end - skip)?;
        let json = json.strip_suffix(b"\r").unwrap_or(json);
        let open = peek::root_open(json)?;
        let mut choices = None;
        let mut root = peek::members(json, open);
        for m in root.by_ref() {
            let m = m?;
            if m.key_is(json, "choices") {
                choices.is_none().then_some(())?;
                choices = Some(m.value);
            }
        }
        if !json.get(root.end()?..)?.iter().all(u8::is_ascii_whitespace) {
            return None;
        }
        let Some(choices) = choices.filter(|c| json.get(c.0) == Some(&b'[')) else {
            out.extend_from_slice(raw);
            return Some(());
        };
        // Each repeated `role` member's cut: the member and the separator that keeps the object
        // valid (as `peek::remove_items` cuts one item).
        let mut cuts: Vec<(usize, usize)> = Vec::new();
        let mut record = Vec::new();
        for (pos, choice) in peek::elements(json, choices.0).enumerate() {
            let (start, _) = choice?;
            if json.get(start) != Some(&b'{') {
                continue;
            }
            let (mut index, mut delta) = (None, None);
            for m in peek::members(json, start) {
                let m = m?;
                if m.key_is(json, "delta") {
                    delta.is_none().then_some(())?;
                    delta = Some(m.value);
                } else if m.key_is(json, "index") {
                    index = Some(m.value);
                }
            }
            let Some(delta) = delta.filter(|d| json.get(d.0) == Some(&b'{')) else {
                continue;
            };
            // `index_of`: a non-negative integer `index`, else the position.
            let c = index
                .and_then(|i| {
                    std::str::from_utf8(json.get(i.0..i.1)?)
                        .ok()?
                        .parse::<u64>()
                        .ok()
                })
                .unwrap_or(pos as u64);
            let (mut prev_end, mut role, mut next_start) = (None, None, None);
            for m in peek::members(json, delta.0) {
                let m = m?;
                let (from, to) = m.span();
                if role.is_some() && next_start.is_none() {
                    next_start = Some(from);
                }
                if m.key_is(json, "role") {
                    role.is_none().then_some(())?;
                    role = Some((m, prev_end));
                }
                if role.is_none() {
                    prev_end = Some(to);
                }
            }
            let Some((role, prev_end)) = role else {
                continue;
            };
            if json.get(role.value.0) != Some(&b'"') {
                continue;
            }
            let (from, to) = role.span();
            match ChatIdentity::seen(&self.identity.sent, &record, c, 0, ChatIdentity::ROLE) {
                Some(key) => record.push(key),
                None => cuts.push(match (next_start, prev_end) {
                    (Some(next), _) => (from, next),
                    (None, Some(prev)) => (prev, to),
                    (None, None) => (from, to),
                }),
            }
        }
        self.identity.sent.extend(record);
        let Some(&(first, _)) = cuts.first() else {
            out.extend_from_slice(raw);
            return Some(());
        };
        out.reserve(raw.len());
        out.extend_from_slice(raw.get(..base + first)?);
        for (k, &(_, to)) in cuts.iter().enumerate() {
            let next = cuts.get(k + 1).map_or(json.len(), |c| c.0);
            out.extend_from_slice(json.get(to..next)?);
        }
        out.extend_from_slice(raw.get(base + json.len()..)?);
        Some(())
    }

    fn deliver(&mut self, item: ChatItem, out: &mut Vec<u8>) {
        if self.lifecycle == Lifecycle::Errored || self.client_ended() {
            return;
        }
        match (self.client, item) {
            (client, ChatItem::Error(v)) => {
                self.lifecycle = Lifecycle::Errored;
                match client {
                    Endpoint::ChatCompletions => out.extend(sse_data(&value_string(&map_error(
                        &v,
                        Endpoint::ChatCompletions,
                        200,
                    )))),
                    Endpoint::Messages => out.extend(sse_named(
                        "error",
                        &value_string(&map_error(&v, Endpoint::Messages, 200)),
                    )),
                    Endpoint::Responses => self.oai_to_resp.error(&v, out),
                    Endpoint::Embeddings => {}
                }
            }
            (Endpoint::ChatCompletions, ChatItem::Chunk(v)) => {
                out.extend(sse_data(&value_string(&v)));
            }
            (Endpoint::ChatCompletions, ChatItem::Done) => self.chat_done(out),
            (Endpoint::Messages, ChatItem::Chunk(v)) => self.oai_to_ant.chunk(&v, out),
            (Endpoint::Messages, ChatItem::Done) => self.oai_to_ant.finish(out),
            (Endpoint::Responses, ChatItem::Chunk(v)) => self.oai_to_resp.chunk(&v, out),
            (Endpoint::Responses, ChatItem::Done) => self.oai_to_resp.finish(out),
            (Endpoint::Embeddings, _) => {}
        }
    }

    /// The client was sent its terminal event (a Chat `[DONE]`, a Messages `message_stop`, a
    /// Responses `response.completed`). Nothing follows it: an upstream that writes past its own
    /// end (an error after the usage chunk, a chunk after `message_stop`) must not hand the client
    /// a second ending, and `proxy`'s [`crate::terminal::TerminalTracker`] reads the end of
    /// what the client was sent to tell a close after the answer from a cancel.
    fn client_ended(&self) -> bool {
        match self.client {
            Endpoint::ChatCompletions => self.done,
            Endpoint::Messages => self.oai_to_ant.phase == Phase::Closed,
            Endpoint::Responses => self.oai_to_resp.phase == Phase::Closed,
            Endpoint::Embeddings => false,
        }
    }

    /// A Chat client's `[DONE]`, sent once.
    fn chat_done(&mut self, out: &mut Vec<u8>) {
        if !self.done {
            self.done = true;
            out.extend(sse_data("[DONE]"));
        }
    }

    /// End of the upstream body. `proxy` reaches this only when the upstream ended its body cleanly
    /// (an upstream that drops a chunked, sized or HTTP/2 body mid-way fails the request instead,
    /// and the client's connection is cut). A clean end with no terminal event is still a cut-short
    /// response — a close-delimited body, or a provider that stopped writing — so the client gets
    /// an error, never a `message_stop` / `response.completed` for a half-written answer.
    fn flush(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.upstream == self.client {
            return out;
        }
        if self.lifecycle == Lifecycle::Open {
            // `Ended` stands if the client already had its terminal event and the error is dropped.
            self.lifecycle.end();
            self.deliver(ChatItem::Error(truncated_stream_error()), &mut out);
        }
        match self.client {
            Endpoint::ChatCompletions => self.chat_done(&mut out),
            Endpoint::Messages if self.lifecycle != Lifecycle::Errored => {
                self.oai_to_ant.finish(&mut out);
            }
            // After an error this is a no-op: `OaiToResp::error` closed its phase.
            Endpoint::Responses => self.oai_to_resp.finish(&mut out),
            // Embeddings never stream and never translate.
            Endpoint::Messages | Endpoint::Embeddings => {}
        }
        out
    }
}

/// A Chat Completions upstream event.
fn chat_items(data: &str, items: &mut Vec<ChatItem>) {
    if data == "[DONE]" {
        items.push(ChatItem::Done);
        return;
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return;
    };
    items.push(if looks_like_error(&v) {
        ChatItem::Error(v)
    } else {
        ChatItem::Chunk(v)
    });
}

/// The fields every Chat Completions chunk repeats.
#[derive(Default)]
struct ChunkMeta {
    id: String,
    model: String,
    created: u64,
}

impl ChunkMeta {
    fn fill(&mut self) {
        if self.id.is_empty() {
            self.id = fresh_id("chatcmpl");
        }
        if self.created == 0 {
            self.created = unix_now();
        }
    }

    fn chunk(&mut self, delta: Value, finish: Option<&str>) -> ChatItem {
        self.fill();
        ChatItem::Chunk(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        }))
    }

    /// The trailing usage-only chunk, as `stream_options.include_usage` shapes it.
    fn usage(&mut self, usage: Value) -> ChatItem {
        self.fill();
        ChatItem::Chunk(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage,
        }))
    }
}

fn reasoning_delta(text: &str) -> Value {
    json!({ "reasoning_content": text, "reasoning": text })
}

/// The next complete event in `buf` from `*start`: its span, with `*start` moved past it.
/// `scanned` is how far into `buf` an earlier call already searched without finding an end; the
/// search resumes there, not from the event's first byte. Nothing is copied or moved.
pub(in crate::translate) fn next_event(
    buf: &[u8],
    start: &mut usize,
    scanned: &mut usize,
) -> Option<(usize, usize)> {
    let rest = buf.get(*start..)?;
    // A terminator is judged at its first `\n`, which needs up to two bytes after it — so the last
    // two positions searched before may have been undecidable and are searched again.
    let from = scanned.saturating_sub(2).saturating_sub(*start);
    match find_event_end(rest, from) {
        Some(len) => {
            let event = (*start, *start + len);
            *start += len;
            *scanned = *start;
            Some(event)
        }
        None => {
            *scanned = buf.len();
            None
        }
    }
}

/// End (exclusive) of the first blank line: a line ending (`\n`, or the `\n` of `\r\n`) followed by
/// an empty line ended by `\n` or `\r\n`, judging only `\n`s at or after `from`. That covers
/// `\n\n` and `\r\n\r\n`, and the mixed endings SSE allows too.
///
/// One forward pass over newlines. The earlier two-`windows` form ran both searches to completion,
/// so a stream using only one line ending walked the whole buffer for the other on every call.
pub(in crate::translate) fn find_event_end(buf: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    while let Some(off) = memchr::memchr(b'\n', buf.get(at..)?) {
        let nl = at + off;
        match buf.get(nl + 1) {
            Some(b'\n') => return Some(nl + 2),
            Some(b'\r') if buf.get(nl + 2) == Some(&b'\n') => return Some(nl + 3),
            _ => {}
        }
        at = nl + 1;
    }
    None
}

fn parse_sse(raw: &[u8]) -> (String, String) {
    let text = String::from_utf8_lossy(raw);
    let mut event = String::new();
    let mut data = String::new();
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(rest) = line.strip_prefix("event:") {
            event = rest.trim().to_owned();
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        }
    }
    (event, data)
}

fn sse_data(data: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() + 8);
    o.extend_from_slice(b"data: ");
    o.extend_from_slice(data.as_bytes());
    o.extend_from_slice(b"\n\n");
    o
}

fn sse_named(event: &str, data: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(event.len() + data.len() + 16);
    o.extend_from_slice(b"event: ");
    o.extend_from_slice(event.as_bytes());
    o.extend_from_slice(b"\ndata: ");
    o.extend_from_slice(data.as_bytes());
    o.extend_from_slice(b"\n\n");
    o
}
