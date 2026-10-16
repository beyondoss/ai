//! Per-run agent state.
//!
//! A [`Session`] is the evolving state of one agent run: the full message history plus token/step
//! counters. It's `serde`-serializable so a headless run (`serve`) can persist it and a client can
//! reattach to a running session later — the foundation for the attach-later remote-control model.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::compaction::CompactionProvenance;
use crate::message::{ContentBlock, Message, Role, TokenUsage};

/// The state of one agent run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Session {
    /// Full conversation, oldest first. Held behind an `Arc` so each model request shares the history
    /// by cloning a pointer instead of deep-copying every message every turn (the history grows each
    /// step, so a per-turn deep copy is quadratic over a run). [`Session::push`] mutates via
    /// `Arc::make_mut` — copy-on-write, so it appends in place once the prior turn's request (the only
    /// other holder of the pointer) has been dropped. Serialized as a plain message array, so the
    /// shared representation is invisible on the wire and persisted sessions are byte-identical.
    #[serde(
        serialize_with = "serialize_messages",
        deserialize_with = "deserialize_messages"
    )]
    pub messages: Arc<Vec<Message>>,
    /// Completed loop iterations (model turn + tool dispatch).
    pub steps: u32,
    /// Cumulative uncached input tokens reported by the model.
    pub input_tokens: u64,
    /// Cumulative output tokens reported by the model.
    pub output_tokens: u64,
    /// Cumulative input tokens served from the prompt cache. The signal that the cache breakpoints
    /// the Anthropic dialect stamps are actually hitting.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Cumulative input tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
    /// Of `cache_write_tokens`, how many used the 1-hour TTL specifically — see
    /// [`TokenUsage::cache_write_1h_tokens`].
    #[serde(default)]
    pub cache_write_1h_tokens: u64,
    /// Cumulative reasoning/thinking tokens billed separately by the provider.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Input tokens of the most recent turn (cached + uncached) — the live context size, which the
    /// cumulative totals above cannot express. The compaction trigger reads this against the model's
    /// context window.
    #[serde(default)]
    pub last_input_tokens: u32,
    /// `messages.len()` at the moment `last_input_tokens` was last set. Messages appended after this
    /// point (the very turn that usage snapshot came from, plus anything since) aren't reflected in
    /// `last_input_tokens` yet — `compaction::trailing_tokens` estimates them separately so the
    /// compaction trigger isn't comparing the window against a stale, undercounted size.
    #[serde(default)]
    pub last_usage_message_count: usize,
    /// Output tokens the provider reported for the turn `last_usage_message_count` was snapshotted at
    /// — i.e. the exact size of `messages[last_usage_message_count]` (that turn's own assistant
    /// message), once appended. `compaction::trailing_tokens` falls back on this for that one message
    /// when it carries no usage of its own (the message's own `usage` is the same figure, and is
    /// what it reads first).
    #[serde(default)]
    pub last_output_tokens: u32,
    /// What compaction has recorded about this session so far — folded forward across every round
    /// (see [`CompactionProvenance`]), since `apply_summary` physically replaces the summarized
    /// messages and anything not captured here is lost with them. Default (all-empty, `compactions:
    /// 0`) for a session that's never been compacted, so older persisted sessions round-trip unchanged.
    #[serde(default)]
    pub compaction: CompactionProvenance,
    /// Blocks a host attached to one message **for the model only** — added to that message in every
    /// request ([`Session::request_messages`]) but never to `messages` itself, so they are not
    /// persisted, not shown in the transcript, and not read by anything that reads the conversation
    /// (a title request, an export). Never serialized. Each is pinned to the message it was attached
    /// to by value, so it rides the same message in every later request — keeping the request's
    /// history byte-stable for the prompt cache — and silently lapses if that message is ever
    /// rewritten or dropped.
    #[serde(skip)]
    pub request_blocks: Vec<RequestBlock>,
    /// Indices of messages the last tool-output masking pass rewrote in place
    /// ([`crate::compaction::mask_tool_results`]), for a persistence layer to write as edits: an
    /// append-only store keyed on message count never sees a change that keeps the count. Writing an
    /// edit must be idempotent (skip a message already stored as it is), so a persistence layer reads
    /// this freely and never drains it. Replaced by each masking pass; cleared when a compaction
    /// replaces the list, whose own rewrite carries the masked content. Never serialized.
    #[serde(skip)]
    pub edited: Vec<usize>,
    /// Where the transcript was cut back, or rewritten from, in place since this session was built
    /// ([`Session::cut_back`], [`Session::rewriting_from`]) — for a persistence layer to rewind to
    /// before it appends: one keyed on message count can't see a pop followed by a push (the count is
    /// back where it was) and would keep the discarded message on disk while never writing its
    /// replacement. Cleared when a compaction replaces the list (its own rewrite carries the current
    /// one). Never serialized.
    #[serde(skip)]
    pub cuts: Cuts,
    /// The last real prompt size [`Session::record_usage`] recorded, and the message boundary it
    /// belongs to, for [`Session::push`] to turn into the next assistant message's
    /// [`Message::prior_turn_tokens`]. Only valid while nothing has rewritten the history since:
    /// compaction, masking and a cross-model scrub clear it ([`Session::forget_anchor`]). Never
    /// serialized, so a reopened session measures its first turn afresh.
    #[serde(skip)]
    anchor: Option<Anchor>,
}

/// Process-wide source of [`Cuts`] ids: increasing across every session in the process, so a store
/// that has applied cut `n` can tell any later cut — even one recorded on a session rebuilt since —
/// from one it has already seen.
static CUT_IDS: AtomicU64 = AtomicU64::new(0);

/// The highest cut id issued in this process so far. A persistence layer opened now starts here:
/// every cut that exists already happened to a transcript it never wrote.
pub fn cut_watermark() -> u64 {
    CUT_IDS.load(Ordering::Relaxed)
}

/// Where a [`Session`]'s transcript was cut back, or rewritten from, in place — see
/// [`Session::cuts`]. Each cut is `(id, length)`: the messages before `length` are untouched, the
/// rest changed or went away. Ids come from [`cut_watermark`]'s counter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cuts(Vec<(u64, usize)>);

impl Cuts {
    fn record(&mut self, len: usize) {
        let id = CUT_IDS.fetch_add(1, Ordering::Relaxed) + 1;
        self.0.push((id, len));
    }

    /// Over the cuts after id `seen`: the newest id, and the shortest length cut back to — how much
    /// of what a store wrote since `seen` still stands. `None` when there are none.
    pub fn since(&self, seen: u64) -> Option<(u64, usize)> {
        self.0
            .iter()
            .filter(|(id, _)| *id > seen)
            .fold(None, |acc, &(id, len)| match acc {
                None => Some((id, len)),
                Some((newest, floor)) => Some((newest.max(id), floor.min(len))),
            })
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

/// One request's real prompt size, keyed to the message boundary it saw — see [`Session::anchor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Anchor {
    /// `messages.len()` when the request was sent: the index its reply lands at.
    index: usize,
    /// [`TokenUsage::prompt_tokens`] of that request.
    prompt: u32,
    /// The prompt's growth since the previous request, when that one's reply landed exactly two
    /// messages earlier: the real size of the turn in between.
    prior_turn: Option<u32>,
}

/// One block attached for the model to a message of a [`Session`] — see
/// [`Session::request_blocks`].
#[derive(Debug, Clone, PartialEq)]
pub struct RequestBlock {
    /// Where the message sat when the block was attached.
    pub index: usize,
    /// The message itself, as it was — the block applies only while `messages[index]` still is it.
    pub anchor: Message,
    pub block: ContentBlock,
}

/// Serialize the shared history as a plain `[Message]` array — the `Arc`/`Vec` wrapper is an
/// in-memory sharing detail, not part of the persisted shape.
fn serialize_messages<S>(messages: &Arc<Vec<Message>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    messages.as_slice().serialize(serializer)
}

/// Load a message array back into a freshly-owned `Arc<Vec<Message>>`.
fn deserialize_messages<'de, D>(deserializer: D) -> Result<Arc<Vec<Message>>, D::Error>
where
    D: Deserializer<'de>,
{
    Vec::<Message>::deserialize(deserializer).map(Arc::new)
}

impl Session {
    /// A fresh, empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach `text` to the newest message, if it is a user turn, as a block of its own for the
    /// model — never in the user's own text, never persisted (see [`Self::request_blocks`]). Returns
    /// whether it was attached.
    pub fn attach_request_block(&mut self, text: impl Into<Arc<str>>) -> bool {
        let Some(last) = self.messages.last() else {
            return false;
        };
        if last.role != Role::User {
            return false;
        }
        self.attach_request_block_at(self.messages.len() - 1, text)
    }

    /// [`Self::attach_request_block`] onto `messages[index]`, whatever its role. Returns whether
    /// there is such a message.
    pub fn attach_request_block_at(&mut self, index: usize, text: impl Into<Arc<str>>) -> bool {
        let Some(anchor) = self.messages.get(index).cloned() else {
            return false;
        };
        // Drop blocks whose message is gone, so this list can't outgrow the history.
        let messages = &self.messages;
        self.request_blocks
            .retain(|b| messages.get(b.index).is_some_and(|m| *m == b.anchor));
        self.request_blocks.push(RequestBlock {
            index,
            anchor,
            block: ContentBlock::text(text),
        });
        true
    }

    /// The history as a model request sends it: `messages`, with every still-valid
    /// [request block](Self::request_blocks) appended to its message. A pointer clone when there are
    /// none (the common case).
    pub fn request_messages(&self) -> Arc<Vec<Message>> {
        if self.request_blocks.is_empty() {
            return self.messages.clone();
        }
        let mut out: Vec<Message> = (*self.messages).clone();
        for b in &self.request_blocks {
            if self.messages.get(b.index) == Some(&b.anchor)
                && let Some(m) = out.get_mut(b.index)
            {
                m.content.push(b.block.clone());
            }
        }
        Arc::new(out)
    }

    /// Append a message to the history. Copy-on-write via `Arc::make_mut`: in place when the history
    /// is solely owned here (the steady state between turns), cloning only if a request still holds
    /// the prior snapshot.
    ///
    /// An assistant message landing where the last [`record_usage`](Self::record_usage) said its
    /// request's reply would gets that request's measured prior turn
    /// ([`Message::prior_turn_tokens`]), when the turn's earlier assistant message came from the same
    /// model.
    pub fn push(&mut self, mut message: Message) {
        if let Some(Anchor {
            index,
            prior_turn: Some(turn),
            ..
        }) = self.anchor
            && index == self.messages.len()
            && message.role == Role::Assistant
            && message.prior_turn_tokens.is_none()
            && index
                .checked_sub(2)
                .and_then(|i| self.messages.get(i))
                .is_some_and(|m| m.role == Role::Assistant && m.model_id == message.model_id)
        {
            message.prior_turn_tokens = Some(turn);
        }
        Arc::make_mut(&mut self.messages).push(message);
    }

    /// Drop the recorded prompt size, after something rewrote the history it measured (compaction,
    /// masking, a scrub): the next request's growth over it would mean nothing.
    pub fn forget_anchor(&mut self) {
        self.anchor = None;
    }

    /// Seed (or continue) the conversation with a user text turn.
    pub fn user(&mut self, text: impl Into<Arc<str>>) -> &mut Self {
        self.push(Message::user(text));
        self
    }

    /// Close a prompt that was never answered with a synthetic [`Message::error`] record, the same
    /// closing record a failed run leaves (see [`Self::pop_error_record`]). Returns whether one was
    /// added.
    ///
    /// The prompt is checkpointed before the model call, so a process killed outright (OOM, `kill -9`,
    /// a CI cancel) leaves a session whose tip is that `user` message, with no closing record — no
    /// `Err` path ever ran. A host resuming such a session calls this before adding its own prompt, so
    /// the conversation doesn't send two consecutive `user` turns: the model sees that the earlier
    /// request was interrupted rather than two prompts merged into one turn.
    pub fn close_unanswered_prompt(&mut self, why: impl Into<String>) -> bool {
        if self.messages.last().is_some_and(|m| m.role == Role::User) {
            self.push(Message::error(why));
            true
        } else {
            false
        }
    }

    /// Remove the trailing message if (and only if) it's a synthetic [`Message::error`] record —
    /// called by a whole-run auto-retry layer right before re-invoking the same run on the same
    /// session. `run_events_steered` persists this record on every `Err`-ending run (so a client's own
    /// retry via a fresh `prompt` never stacks a second consecutive `user` turn on a truly abandoned
    /// run), but an *automatic* retry re-runs the exact same attempt from scratch — it must be
    /// genuinely invisible in the session's history, "no trace of the failed attempt in loop-visible
    /// state" (mirroring how `agent_core`'s own mid-stream retry layer, one level down, never persists
    /// anything for a failed attempt in the first place). Leaving the record in place here would let
    /// the retry's own real response land right after it, producing two consecutive assistant turns —
    /// or, on final exhaustion, a spurious extra history entry no client asked for.
    pub fn pop_error_record(&mut self) {
        if self
            .messages
            .last()
            .is_some_and(|m| m.error_message.is_some())
        {
            self.cut_back(self.messages.len() - 1);
        }
    }

    /// Cut the transcript back to its first `len` messages, recording the cut in [`Self::cuts`] so a
    /// persistence layer that already wrote what went away rewinds past it rather than keeping it.
    /// Every in-place shrink goes through here. A no-op when there is nothing past `len`.
    pub fn cut_back(&mut self, len: usize) {
        if len < self.messages.len() {
            Arc::make_mut(&mut self.messages).truncate(len);
            self.cuts.record(len);
        }
    }

    /// Record that the messages from `index` on are about to be rewritten in place — the count-keeping
    /// counterpart of [`Self::cut_back`], for a persistence layer that already wrote the old ones.
    pub fn rewriting_from(&mut self, index: usize) {
        if index < self.messages.len() {
            self.cuts.record(index);
        }
    }

    /// Scrub state that doesn't survive a model switch, ahead of resuming the conversation on
    /// `new_model`. Two things are per-producing-model, not portable:
    ///
    /// - Signed `Thinking` blocks and encrypted `RedactedThinking` blocks — Anthropic can reject a
    ///   later turn that replays a *signed* one to a different model than produced it, and a redacted
    ///   block is opaque ciphertext meaningless to any other model. A signed `Thinking` block's own
    ///   *text*, though, is ordinary prose — pi downgrades it to a plain `Text` block instead of
    ///   discarding it outright, preserving the visible reasoning trace as context for the new model
    ///   rather than silently erasing every prior turn's chain of thought on a model switch. An
    ///   *empty*-text `Thinking` block (nothing worth preserving) is dropped like `RedactedThinking`,
    ///   not downgraded into a useless empty `Text` block.
    /// - Combined OpenAI-Responses tool-call ids (`"call_id|item_id"`) — the `item_id` half only pairs
    ///   with a `reasoning` item on the *same* model/dialect; replayed to a foreign model it's at best
    ///   dead weight, at worst rejected. Truncating a `ToolUse.id` this way would silently break its
    ///   pairing with the matching `ToolResult.tool_use_id` — possibly in a later message — so every
    ///   rewrite is recorded and replayed onto the paired `ToolResult` in a second pass below.
    ///
    /// Applied per-message, gated on `Message::model_id`: a message stamped with `new_model` itself is
    /// untouched (still valid to replay), everything else — including any message with `model_id: None`
    /// (persisted before this field existed, or from a source that never stamped it) — is treated as
    /// foreign and scrubbed. `split_once('|')`-based truncation is a no-op for Anthropic-native tool-use
    /// ids (never contain `|`), so this is safe to run unconditionally regardless of dialect.
    pub fn scrub_cross_model_state(&mut self, new_model: &str) {
        self.anchor = None;
        let messages = Arc::make_mut(&mut self.messages);
        // A measured turn (`Message::prior_turn_tokens`) whose messages this rewrites no longer
        // describes them: drop it wherever the turn's assistant message is foreign.
        let foreign = |m: &Message| m.model_id.as_deref() != Some(new_model);
        for i in 2..messages.len() {
            if foreign(&messages[i]) || foreign(&messages[i - 2]) {
                messages[i].prior_turn_tokens = None;
            }
        }

        // First pass: downgrade/drop thinking blocks and truncate foreign-model tool-call ids,
        // recording old -> new id for every `ToolUse` rewrite.
        let mut id_remap: HashMap<String, String> = HashMap::new();
        for message in messages.iter_mut() {
            if message.model_id.as_deref() == Some(new_model) {
                continue;
            }
            message.content.retain_mut(|block| match block {
                ContentBlock::Thinking { text, .. } if !text.is_empty() => {
                    let text = std::mem::take(text);
                    *block = ContentBlock::text(text);
                    true
                }
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => false,
                ContentBlock::ToolUse { id, .. } => {
                    let new_id = crate::dialect::openai_responses::call_id_only(id);
                    if new_id != *id {
                        id_remap.insert(std::mem::replace(id, new_id.clone()), new_id);
                    }
                    true
                }
                _ => true,
            });
        }

        // Second pass: replay each rewrite onto its paired `ToolResult` — which may live in a later
        // message, so it's out of scope in the loop above. `ToolResult` blocks live on `User`
        // messages, which never carry a `model_id` (see `Message::model_id`), so there's no
        // same-model shortcut to take here: any match against the remap is by definition a pairing
        // that must follow its `ToolUse` to stay intact, regardless of which message it's on.
        if !id_remap.is_empty() {
            for message in messages.iter_mut() {
                for block in &mut message.content {
                    if let ContentBlock::ToolResult { tool_use_id, .. } = block
                        && let Some(new_id) = id_remap.get(tool_use_id)
                    {
                        new_id.clone_into(tool_use_id);
                    }
                }
            }
        }
    }

    /// Whether this session holds any state [`scrub_cross_model_state`](Self::scrub_cross_model_state)
    /// would actually change, were the conversation resumed on `new_model`.
    ///
    /// A read-only pre-check, and the reason the scrub can sit unconditionally on the run path
    /// ([`crate::agent::Agent::run_events_steered`]) without costing anything on the overwhelmingly
    /// common same-model turn: the scrub itself calls `Arc::make_mut` on the message vec, which
    /// **deep-clones the whole transcript** whenever the `Arc` is shared (it is — persistence holds a
    /// handle). Paying that on every turn of a long session to almost always change nothing would be a
    /// real regression, so the mutation is gated behind this scan, which allocates nothing.
    ///
    /// Deliberately narrower than "does any message have a foreign `model_id`". Every `User` message has
    /// `model_id: None` and is therefore foreign by the scrub's own rule, so that question is `true` for
    /// essentially every session and would gate nothing. What actually matters is whether a *foreign*
    /// message carries state that doesn't survive the crossing: a `Thinking`/`RedactedThinking` block
    /// (signature or ciphertext bound to the model that produced it) or an OpenAI-Responses combined
    /// `"call_id|item_id"` tool-call id. Nothing else the scrub touches can differ.
    pub fn needs_cross_model_scrub(&self, new_model: &str) -> bool {
        self.messages
            .iter()
            .filter(|m| m.model_id.as_deref() != Some(new_model))
            .flat_map(|m| m.content.iter())
            .any(|block| match block {
                ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => true,
                ContentBlock::ToolUse { id, .. } => id.contains('|'),
                _ => false,
            })
    }

    /// Fold a turn's token usage into the running totals and record the live context size.
    pub fn record_usage(&mut self, usage: TokenUsage) {
        self.input_tokens += u64::from(usage.input_tokens);
        self.output_tokens += u64::from(usage.output_tokens);
        self.cache_read_tokens += u64::from(usage.cache_read_tokens);
        self.cache_write_tokens += u64::from(usage.cache_write_tokens);
        self.cache_write_1h_tokens += u64::from(usage.cache_write_1h_tokens);
        self.reasoning_tokens += u64::from(usage.reasoning_tokens);
        // The whole prompt the model just saw = uncached input + everything served from / written to
        // cache. This is the current context size (the cumulative sums above can't express it) and is
        // what the compaction trigger compares against the model's context window. `saturating_add`,
        // not `+`: these three fields come straight from parsed model-API usage data with no
        // upper-bound validation (a non-standard proxy could report anomalously large values), and
        // `overflow-checks = true` in the release profile turns a raw `u32` overflow into a panic.
        let live_input = usage.prompt_tokens();
        // A genuinely all-zero report means no real `Usage` event ever arrived this turn (the loop's
        // `Aborted`-turn path calls this unconditionally, before checking `stop_reason`, and a stream
        // cancelled before its `Usage` event carries `TokenUsage::default()`) — not a provider telling
        // us the live context just became empty. Holding the last real snapshot instead of zeroing it
        // out keeps `get_state`'s `context_usage` (and the compaction trigger) reporting the session's
        // actual last-known size through an aborted turn, rather than spuriously reporting "no context
        // at all" the instant a client cancels mid-stream. A message pushed since the last real
        // snapshot (including the very turn this call is for) is still accounted for — just via
        // `compaction::trailing_tokens`'s calibrated estimate instead of this exact figure, the same
        // fallback every message ahead of `last_usage_message_count` already uses.
        if live_input > 0 {
            let index = self.messages.len();
            self.last_input_tokens = live_input;
            self.last_usage_message_count = index;
            self.last_output_tokens = usage.output_tokens;
            // The anchor: this request's real prompt, and — when the previous request's reply landed
            // two messages back and nothing rewrote the history since — the real size of the turn
            // between them. A prompt that shrank without a rewrite (a smaller system prompt or tool
            // set) measures nothing.
            let prior_turn = self
                .anchor
                .filter(|a| a.index + 2 == index && live_input >= a.prompt)
                .map(|a| live_input - a.prompt);
            self.anchor = Some(Anchor {
                index,
                prompt: live_input,
                prior_turn,
            });
        }
    }

    /// Rebuild the live-size snapshot ([`Session::last_input_tokens`] and its boundary) for a session
    /// opened from disk, from the persisted messages: the most recent message carrying usage gives
    /// the exact prompt size its request carried, as a still-running process would have it.
    ///
    /// Except when that usage predates the last compaction: the kept suffix keeps its messages'
    /// usage, which measured the prompt *before* the summary replaced the prefix. A session reopened
    /// between a compaction and the next reply would report the pre-compaction size. It is
    /// recognized by the summary's own record of the size it compacted from
    /// ([`crate::compaction::summary_tokens_before`]), which is that very usage's prompt size. Then,
    /// and when no message carries usage at all, the size is the whole conversation's
    /// [`crate::token_size::message_sizes`] — every measured turn exact, the rest estimated.
    pub fn restore_usage_snapshot(&mut self, family: crate::token_size::TokenFamily) {
        let last = self
            .messages
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, m)| m.usage.filter(|u| u.prompt_tokens() > 0).map(|u| (i, u)));
        let compacted_from = self
            .messages
            .first()
            .and_then(crate::compaction::summary_tokens_before);
        match last {
            Some((i, usage)) if compacted_from != Some(usage.prompt_tokens()) => {
                self.last_usage_message_count = i;
                self.last_output_tokens = usage.output_tokens;
                self.last_input_tokens = usage.prompt_tokens();
            }
            _ => {
                let sizes = crate::token_size::message_sizes(&self.messages, family, None);
                self.last_input_tokens = crate::token_size::total(&sizes);
                self.last_usage_message_count = self.messages.len();
                self.last_output_tokens = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn close_unanswered_prompt_closes_only_a_user_tip() {
        let mut session = Session::new();
        assert!(!session.close_unanswered_prompt("x"), "nothing to close");
        session.user("first");
        assert!(session.close_unanswered_prompt("interrupted"));
        let tip = session.messages.last().unwrap();
        assert_eq!(tip.role, Role::Assistant);
        assert_eq!(tip.error_message.as_deref(), Some("interrupted"));
        assert!(!session.close_unanswered_prompt("again"), "already closed");
        assert_eq!(session.messages.len(), 2);
    }

    use super::*;

    /// Every in-place shrink is recorded, with an id newer than anything before it; `since` reports
    /// the newest id and the shortest length over the cuts a store hasn't seen.
    #[test]
    fn cuts_record_every_shrink_and_since_folds_the_unseen_ones() {
        let mut s = Session::new();
        for i in 0..4 {
            s.user(format!("u{i}"));
        }
        let before = cut_watermark();
        assert_eq!(s.cuts.since(before), None);
        s.cut_back(3);
        s.cut_back(9); // nothing past 9: not a cut
        s.push(Message::error("boom"));
        s.pop_error_record();
        let (newest, floor) = s.cuts.since(before).unwrap();
        assert_eq!(floor, 3);
        assert!(newest > before);
        assert_eq!(s.cuts.since(newest), None);
        s.rewriting_from(1);
        assert_eq!(s.cuts.since(newest).map(|(_, f)| f), Some(1));
    }

    #[test]
    fn scrub_cross_model_state_downgrades_signed_thinking_to_text_and_drops_redacted() {
        let mut s = Session::new();
        s.push(Message::user("think then answer"));
        s.push(
            Message::assistant(vec![
                ContentBlock::Thinking {
                    text: "let me reason".into(),
                    signature: "sig-abc".into(),
                },
                ContentBlock::RedactedThinking {
                    data: "opaque".into(),
                },
                ContentBlock::text("answer"),
            ])
            .with_model_id("claude-opus-4-8"),
        );
        s.scrub_cross_model_state("gpt-5");
        assert_eq!(s.messages[0].content.len(), 1); // user text untouched
        // The redacted (opaque ciphertext) block is dropped; the signed thinking block's own text
        // survives, downgraded to a plain Text block — visible reasoning context for the new model,
        // not silently erased.
        assert_eq!(
            s.messages[1].content.len(),
            2,
            "{:?}",
            s.messages[1].content
        );
        assert_eq!(
            s.messages[1].content[0],
            ContentBlock::text("let me reason")
        );
        assert_eq!(s.messages[1].content[1], ContentBlock::text("answer"));
    }

    #[test]
    fn scrub_cross_model_state_drops_an_empty_thinking_block_instead_of_downgrading_it() {
        // Nothing worth preserving — must not become a useless empty Text block, which some
        // providers (Anthropic in particular) reject outright as an invalid content block.
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::Thinking {
                text: String::new().into(),
                signature: String::new(),
            }])
            .with_model_id("claude-opus-4-8"),
        );
        s.scrub_cross_model_state("gpt-5");
        assert!(
            s.messages[0].content.is_empty(),
            "{:?}",
            s.messages[0].content
        );
    }

    #[test]
    fn scrub_cross_model_state_leaves_a_message_stamped_with_the_new_model_untouched() {
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::Thinking {
                text: "reasoning".into(),
                signature: "sig".into(),
            }])
            .with_model_id("gpt-5"),
        );
        s.scrub_cross_model_state("gpt-5");
        assert_eq!(s.messages[0].content.len(), 1); // thinking block survives — same model
    }

    #[test]
    fn scrub_cross_model_state_treats_a_message_with_no_model_id_as_foreign() {
        // Persisted before this field existed (or from a source that never stamped it) — always
        // scrubbed (downgraded, for a non-empty signed Thinking block — see the two tests above).
        let mut s = Session::new();
        s.push(Message::assistant(vec![ContentBlock::Thinking {
            text: "reasoning".into(),
            signature: "sig".into(),
        }]));
        s.scrub_cross_model_state("gpt-5");
        assert_eq!(s.messages[0].content, vec![ContentBlock::text("reasoning")]);
    }

    #[test]
    fn scrub_cross_model_state_truncates_combined_tool_call_ids_from_a_foreign_model() {
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::tool_use(
                "call_1|fc_1",
                "read",
                serde_json::json!({}),
            )])
            .with_model_id("gpt-5"),
        );
        s.scrub_cross_model_state("claude-opus-4-8");
        let ContentBlock::ToolUse { id, .. } = &s.messages[0].content[0] else {
            panic!("expected a ToolUse block");
        };
        assert_eq!(id, "call_1");
    }

    #[test]
    fn scrub_cross_model_state_keeps_a_tool_use_and_its_later_tool_result_paired() {
        // Realistic multi-message shape: the ToolUse lives on the assistant turn that requested it,
        // its ToolResult on a later user turn (with unrelated turns in between) — truncating the
        // combined id on one side without the other would leave the pairing broken.
        let mut s = Session::new();
        s.push(
            Message::assistant(vec![ContentBlock::tool_use(
                "call_1|fc_1",
                "read",
                serde_json::json!({}),
            )])
            .with_model_id("gpt-5"),
        );
        s.push(
            Message::assistant(vec![ContentBlock::text("still working...")]).with_model_id("gpt-5"),
        );
        s.push(Message::tool_result("call_1|fc_1", "file contents", false));

        s.scrub_cross_model_state("claude-opus-4-8");

        let ContentBlock::ToolUse {
            id: tool_use_id, ..
        } = &s.messages[0].content[0]
        else {
            panic!("expected a ToolUse block");
        };
        let ContentBlock::ToolResult {
            tool_use_id: result_id,
            ..
        } = &s.messages[2].content[0]
        else {
            panic!("expected a ToolResult block");
        };
        assert_eq!(tool_use_id, "call_1");
        assert_eq!(result_id, "call_1");
        assert_eq!(
            tool_use_id, result_id,
            "ToolUse/ToolResult pairing must survive the scrub"
        );
    }

    #[test]
    fn round_trips_through_json() {
        let mut s = Session::new();
        s.user("hello");
        s.steps = 2;
        s.record_usage(TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_tokens: 100,
            cache_write_tokens: 20,
            reasoning_tokens: 4,
            ..Default::default()
        });
        s.record_usage(TokenUsage {
            input_tokens: 3,
            output_tokens: 7,
            cache_read_tokens: 200,
            ..Default::default()
        });
        let json = serde_json::to_string(&s).unwrap();
        let back: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.steps, 2);
        assert_eq!(back.input_tokens, 13);
        assert_eq!(back.output_tokens, 12);
        assert_eq!(back.cache_read_tokens, 300);
        assert_eq!(back.cache_write_tokens, 20);
        assert_eq!(back.reasoning_tokens, 4);
        // Live context size = last turn's input + cache read + cache write (3 + 200 + 0).
        assert_eq!(back.last_input_tokens, 203);
        // The second (most recent) call's own reported output size, for `trailing_tokens` to use in
        // place of a rough estimate for that turn's assistant message.
        assert_eq!(back.last_output_tokens, 7);
    }

    #[test]
    fn record_usage_saturates_instead_of_panicking_on_u32_overflow() {
        // A non-standard proxy/gateway reporting anomalously large usage fields must not crash a
        // long-running `serve` session — `overflow-checks = true` in the release profile turns a raw
        // `u32 + u32 + u32` overflow into a hard panic.
        let mut s = Session::new();
        s.record_usage(TokenUsage {
            input_tokens: u32::MAX - 10,
            cache_read_tokens: 100,
            cache_write_tokens: 100,
            ..Default::default()
        });
        assert_eq!(s.last_input_tokens, u32::MAX);
    }

    #[test]
    fn record_usage_holds_the_last_real_snapshot_through_a_genuinely_zero_usage_turn() {
        // pi-parity fix: a stream cancelled before its `Usage` event ever arrives (`Agent`'s `Aborted`
        // turn path) calls `record_usage(TokenUsage::default())` unconditionally — previously this
        // zeroed `last_input_tokens`, making `get_state`'s `context_usage` (and the compaction trigger)
        // report "no context at all" the instant a client cancelled mid-stream, even with a real,
        // sizable prior conversation still live in the session.
        let mut s = Session::new();
        s.record_usage(TokenUsage {
            input_tokens: 5_000,
            cache_read_tokens: 200,
            ..Default::default()
        });
        assert_eq!(s.last_input_tokens, 5_200);
        let count_before = s.last_usage_message_count;

        s.push(Message::assistant(vec![ContentBlock::text("")]));
        s.record_usage(TokenUsage::default());

        assert_eq!(
            s.last_input_tokens, 5_200,
            "a zero-usage report must not overwrite the last real live-context snapshot"
        );
        assert_eq!(
            s.last_usage_message_count, count_before,
            "the snapshot's message-count boundary must also hold, not silently advance past the \
             unaccounted-for message that triggered this call"
        );
    }

    /// The gate must fire on exactly the state the scrub would change — and on nothing else, or every
    /// turn of a long same-model session pays a full transcript deep-clone to change nothing.
    #[test]
    fn needs_cross_model_scrub_fires_only_on_state_that_does_not_survive_a_switch() {
        // A same-model transcript with a signed thinking block: replayable as-is, nothing to do.
        let mut s = Session::new();
        s.push(Message {
            model_id: Some("claude-opus-4-8".to_string()),
            ..Message::assistant(vec![ContentBlock::Thinking {
                text: "hmm".into(),
                signature: "sig".into(),
            }])
        });
        assert!(!s.needs_cross_model_scrub("claude-opus-4-8"));
        // …the same block, now facing a different model: bound to the model that produced it.
        assert!(s.needs_cross_model_scrub("gpt-5"));

        // Plain text from a foreign model is perfectly replayable — this must NOT trigger a scrub, or
        // the gate is just "is there any foreign message", which is true of essentially every session
        // (every `User` message has `model_id: None`) and would gate nothing.
        let mut plain = Session::new();
        plain.push(Message::user("hi"));
        plain.push(Message {
            model_id: Some("gpt-5".to_string()),
            ..Message::assistant(vec![ContentBlock::text("hello")])
        });
        assert!(!plain.needs_cross_model_scrub("claude-opus-4-8"));

        // An OpenAI-Responses combined tool id from a foreign model does need truncating.
        let mut combined = Session::new();
        combined.push(Message {
            model_id: Some("gpt-5".to_string()),
            ..Message::assistant(vec![ContentBlock::ToolUse {
                id: "call_1|item_1".into(),
                name: "read".into(),
                input: serde_json::json!({}),
                thought_signature: None,
            }])
        });
        assert!(combined.needs_cross_model_scrub("claude-opus-4-8"));
        // An Anthropic-native id contains no `|`, so a switch away from it is a no-op.
        let mut native = Session::new();
        native.push(Message {
            model_id: Some("claude-opus-4-8".to_string()),
            ..Message::assistant(vec![ContentBlock::ToolUse {
                id: "toolu_abc".into(),
                name: "read".into(),
                input: serde_json::json!({}),
                thought_signature: None,
            }])
        });
        assert!(!native.needs_cross_model_scrub("gpt-5"));
    }

    #[test]
    fn a_request_block_rides_its_message_in_every_request_and_is_never_persisted() {
        let mut s = Session::new();
        s.user("first");
        assert!(s.attach_request_block("ctx-block"));
        s.push(Message::assistant(vec![ContentBlock::text("reply")]));
        s.user("second");
        let req = s.request_messages();
        // On the message it was attached to, as a block of its own; the user's text untouched.
        assert_eq!(req[0].content.len(), 2);
        assert_eq!(req[0].content[0], ContentBlock::text("first"));
        assert_eq!(req[0].content[1], ContentBlock::text("ctx-block"));
        assert_eq!(req[2].content.len(), 1);
        // Never in the history, never serialized.
        assert_eq!(s.messages[0].content.len(), 1);
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("ctx-block"), "{json}");
        // Not on an assistant turn.
        s.push(Message::assistant(vec![ContentBlock::text("r2")]));
        assert!(!s.attach_request_block("nope"));
    }

    #[test]
    fn a_request_block_lapses_when_its_message_is_rewritten_and_follows_compaction() {
        let mut s = Session::new();
        s.user("one");
        s.push(Message::assistant(vec![ContentBlock::text("a1")]));
        s.user("two");
        s.attach_request_block("ctx-two");
        // Compaction keeping messages[2..]: the block moves with "two".
        crate::compaction::apply_summary(&mut s, 2, "summary", 10);
        let req = s.request_messages();
        assert_eq!(req[1].content.last(), Some(&ContentBlock::text("ctx-two")));
        // Rewritten: the anchor no longer matches, so the block lapses rather than land elsewhere.
        let mut messages = (*s.messages).clone();
        messages[1] = Message::user("rewritten");
        s.messages = Arc::new(messages);
        let req = s.request_messages();
        assert!(
            req.iter()
                .all(|m| !m.content.contains(&ContentBlock::text("ctx-two")))
        );
    }

    fn prompt(n: u32) -> TokenUsage {
        TokenUsage {
            input_tokens: n / 10,
            cache_read_tokens: n - n / 10,
            output_tokens: 7,
            ..Default::default()
        }
    }

    fn reply(model: &str, usage: TokenUsage) -> Message {
        Message::assistant(vec![ContentBlock::text("r")])
            .with_model_id(model)
            .with_usage(usage)
    }

    /// One loop turn the way `run_events_steered` records it: usage first, then the reply.
    fn turn(s: &mut Session, model: &str, prompt_tokens: u32) {
        s.record_usage(prompt(prompt_tokens));
        s.push(reply(model, prompt(prompt_tokens)));
        s.push(Message::tool_result("t", "out", false));
    }

    #[test]
    fn each_reply_carries_the_real_size_of_the_turn_before_it() {
        let mut s = Session::new();
        s.user("go");
        turn(&mut s, "m", 3_000);
        turn(&mut s, "m", 7_500);
        turn(&mut s, "m", 8_000);
        // The first reply has no earlier request to measure against.
        assert_eq!(s.messages[1].prior_turn_tokens, None);
        // Each later one: its prompt less the previous prompt — the earlier reply plus its tool
        // results, as the provider counted them.
        assert_eq!(s.messages[3].prior_turn_tokens, Some(4_500));
        assert_eq!(s.messages[5].prior_turn_tokens, Some(500));
        // User messages never carry one.
        assert!(
            s.messages
                .iter()
                .filter(|m| m.role == Role::User)
                .all(|m| m.prior_turn_tokens.is_none())
        );
    }

    #[test]
    fn a_rewrite_or_a_gap_between_two_requests_measures_nothing() {
        let mut s = Session::new();
        s.user("go");
        turn(&mut s, "m", 3_000);
        // A rewrite (compaction, masking, a scrub) between the two requests: the growth would be the
        // rewrite's, not the turn's.
        s.forget_anchor();
        turn(&mut s, "m", 9_000);
        assert_eq!(s.messages[3].prior_turn_tokens, None);
        // An error turn in between: the previous measured reply is four messages back, not two.
        s.push(Message::error("boom"));
        s.user("again");
        turn(&mut s, "m", 9_600);
        assert_eq!(s.messages[7].prior_turn_tokens, None);
        // Measuring resumes on the next clean pair.
        turn(&mut s, "m", 9_900);
        assert_eq!(s.messages[9].prior_turn_tokens, Some(300));
        // A model switch between the two replies: different tokenizers, nothing comparable.
        turn(&mut s, "other", 12_000);
        assert_eq!(s.messages[11].prior_turn_tokens, None);
        // A prompt that shrank with no rewrite (a smaller tool set) measures nothing either.
        turn(&mut s, "other", 11_000);
        assert_eq!(s.messages[13].prior_turn_tokens, None);
        // A zero-usage report (a cancelled stream) leaves the last anchor alone.
        s.record_usage(TokenUsage::default());
        s.push(Message::assistant(vec![ContentBlock::text("")]));
        assert_eq!(s.messages[15].prior_turn_tokens, None);
    }

    #[test]
    fn a_measured_turn_persists_and_old_sessions_still_load() {
        let mut s = Session::new();
        s.user("go");
        turn(&mut s, "m", 3_000);
        turn(&mut s, "m", 4_000);
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"prior_turn_tokens\":1000"), "{json}");
        let back: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages[3].prior_turn_tokens, Some(1_000));
        // The anchor itself is not persisted: a reopened session measures its first turn afresh.
        assert_eq!(back.anchor, None);
        // A message persisted before the field existed reads as unmeasured.
        let old: Message =
            serde_json::from_str(r#"{"role":"assistant","content":[{"type":"text","text":"hi"}]}"#)
                .unwrap();
        assert_eq!(old.prior_turn_tokens, None);
        // And an unmeasured message writes no field at all: byte-identical to before.
        assert!(
            !serde_json::to_string(&old)
                .unwrap()
                .contains("prior_turn_tokens")
        );
    }

    #[test]
    fn compaction_and_a_scrub_drop_the_anchor_and_the_turns_they_invalidate() {
        let mut s = Session::new();
        s.user("go");
        turn(&mut s, "a", 3_000);
        turn(&mut s, "a", 4_000);
        crate::compaction::apply_summary(&mut s, 1, "summary", 4_000);
        assert_eq!(s.anchor, None);
        // The kept turn keeps its measurement: its content did not change.
        assert_eq!(s.messages[3].prior_turn_tokens, Some(1_000));
        // A scrub for another model rewrites that model's turns: their measurements go.
        let mut t = Session::new();
        t.user("go");
        turn(&mut t, "a", 3_000);
        turn(&mut t, "a", 4_000);
        t.scrub_cross_model_state("b");
        assert_eq!(t.anchor, None);
        assert_eq!(t.messages[3].prior_turn_tokens, None);
    }

    #[test]
    fn restore_uses_the_last_usage_unless_it_predates_the_last_compaction() {
        let family = crate::token_size::TokenFamily::Claude;
        let mut s = Session::new();
        s.user("go");
        turn(&mut s, "m", 3_000);
        turn(&mut s, "m", 50_000);
        // Reopened mid-run: the last reply's request size, exactly.
        let mut reopened = Session {
            messages: s.messages.clone(),
            ..Session::new()
        };
        reopened.restore_usage_snapshot(family);
        assert_eq!(reopened.last_input_tokens, 50_000);
        assert_eq!(reopened.last_usage_message_count, 3);

        // Compacted (from that very size), then reopened before the next reply: the kept reply's
        // usage measured the prompt before the summary replaced its prefix. The summary records the
        // size it compacted from, which identifies it.
        crate::compaction::apply_summary(&mut s, 3, "the summary", 50_000);
        let mut reopened = Session {
            messages: s.messages.clone(),
            ..Session::new()
        };
        reopened.restore_usage_snapshot(family);
        assert!(
            reopened.last_input_tokens < 50_000,
            "a pre-compaction size must not survive the reopen: {}",
            reopened.last_input_tokens
        );
        assert_eq!(reopened.last_usage_message_count, reopened.messages.len());
        let sizes = crate::token_size::message_sizes(&reopened.messages, family, None);
        assert_eq!(reopened.last_input_tokens, crate::token_size::total(&sizes));
    }
}
