//! Context compaction — keeping a long run under the model's context window.
//!
//! A loop re-sends an ever-growing transcript every turn. Left unbounded it eventually exceeds the
//! model's context window and every further request fails — the session effectively dies. Compaction
//! replaces the older prefix of the conversation with an LLM-generated summary while keeping the most
//! recent turns verbatim, so the run can continue indefinitely.
//!
//! This module is the pure, network-free half: the trigger ([`should_compact`]), the cut-point
//! search ([`find_cut`]), the summary-prompt construction ([`render_prefix`]/[`summary_request`]),
//! and file-op extraction. [`Agent`](crate::Agent) owns the one piece that needs the network — making
//! the summarization model call — and stitches the result back into the [`Session`].

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::message::{ContentBlock, Message, Role, StopReason};
use crate::session::Session;
use crate::transport::ModelRequest;

/// Why a compaction fired — carried on [`crate::agent::AgentEvent::Compacted`] and folded into
/// [`CompactionProvenance`], so a persisted session (or a client watching the event stream) can tell
/// *why* a round happened, not just that one did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    /// The live prompt crossed the configured threshold (`should_compact`'s automatic trigger).
    Threshold,
    /// The provider rejected a request for exceeding the context window; this is the recovery.
    Overflow,
    /// An explicit request (a headless client's manual `compact` command).
    Manual,
}

/// What compaction has recorded about this session so far, folded forward across every round.
/// [`Session::compaction`] carries one of these; each successful [`crate::agent::Agent::compact`] call
/// replaces it with the merged result of [`merge_file_ops`].
///
/// This exists because `apply_summary` is *deliberately* destructive — it physically replaces the
/// summarized prefix with a summary message, matching this project's flat-history simplification — so
/// once a round's raw messages are gone, anything not captured here is gone with them. Without folding
/// file-ops forward, `extract_file_ops` on the next round only ever sees *new* activity (the old
/// messages, and their `ToolUse` blocks, no longer exist to scan), so a second or third compaction
/// would silently lose the first round's file awareness entirely.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompactionProvenance {
    /// Files read (or listed) across every compaction round so far, deduped, oldest-first.
    #[serde(default)]
    pub read_files: Vec<String>,
    /// Files written or edited across every compaction round so far, deduped, oldest-first.
    #[serde(default)]
    pub modified_files: Vec<String>,
    /// How many times this session has been compacted.
    #[serde(default)]
    pub compactions: u32,
    /// Why the most recent compaction fired. `None` until the first one.
    #[serde(default)]
    pub last_reason: Option<CompactionReason>,
}

/// Fold `previous`'s file-ops forward with whatever [`extract_file_ops`] finds in `messages` (the
/// current round's prefix), deduping by path and keeping first-seen order, and record `reason` as the
/// new `last_reason`. `messages` is the *whole* current-round prefix, including a leading
/// prior-summary message if there is one — that message carries no `ToolUse` blocks (it's prose), so
/// scanning it contributes nothing and this naturally reduces to "previous provenance + this round's
/// new activity" without needing to slice it off.
pub fn merge_file_ops(
    previous: &CompactionProvenance,
    messages: &[Message],
    reason: CompactionReason,
) -> CompactionProvenance {
    let (new_read, new_modified) = extract_file_ops(messages);
    let mut read = previous.read_files.clone();
    for path in new_read {
        if !read.contains(&path) {
            read.push(path);
        }
    }
    let mut modified = previous.modified_files.clone();
    for path in new_modified {
        if !modified.contains(&path) {
            modified.push(path);
        }
    }
    CompactionProvenance {
        read_files: read,
        modified_files: modified,
        compactions: previous.compactions.saturating_add(1),
        last_reason: Some(reason),
    }
}

/// Tunables for automatic compaction.
#[derive(Debug, Clone, Copy)]
pub struct CompactionConfig {
    /// The model's context window, in tokens. The trigger compares the live prompt size against it.
    pub context_window: u32,
    /// Headroom left below the window: compaction fires once the prompt crosses
    /// `context_window - reserve_tokens`, leaving room for the next turn's output + slack.
    pub reserve_tokens: u32,
    /// Roughly how many tokens of recent conversation to keep verbatim after a compaction.
    pub keep_recent_tokens: u32,
    /// Output ceiling for the summarization call itself.
    pub summary_max_tokens: u32,
    /// Whether automatic (threshold-triggered) compaction is on. Manual/overflow compaction ignores it.
    pub enabled: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        // Defaults sized for a 200k-context Claude model; override per deployment.
        Self {
            context_window: 200_000,
            reserve_tokens: 24_000,
            keep_recent_tokens: 40_000,
            summary_max_tokens: 4_096,
            enabled: true,
        }
    }
}

/// System prompt for the summarization call. The trailing guardrail matches pi's own
/// `SUMMARIZATION_SYSTEM_PROMPT` (`compaction.ts`): the rendered transcript embeds real past user
/// questions, and without an explicit instruction not to, the model can slip into answering one of
/// them instead of emitting the structured checkpoint format this call actually needs.
pub const SUMMARY_SYSTEM: &str = "You compact a long agent transcript so the agent can keep working \
with far fewer tokens but no loss of essential context. Be precise, concrete, and information-dense; \
preserve file paths, identifiers, commands, and decisions exactly. Do NOT continue the conversation. \
Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

/// The instruction appended after the rendered transcript in the summarization call. Matches pi's own
/// `SUMMARIZATION_PROMPT` (`compaction.ts`) structurally: a bracketed exemplar per section (telling the
/// model *what kind of content* goes there, not just a bare heading), an explicit "(none)" fallback for
/// the two sections that can legitimately be empty, checkbox conventions for Done/In Progress
/// (`- [x]`/`- [ ]`) matching a to-do-list shape the model has abundant training signal for, and a
/// numbered (not bulleted) Next Steps list, since order matters there and nowhere else in the template.
pub const SUMMARY_INSTRUCTION: &str = "The messages above are a conversation to summarize. Create a \
structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT \
format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items if the session covers \
different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements \
mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed \
tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, \
if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of \
what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to \
continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, \
function names, and error messages.";

/// The instruction used when a *previous* summary already exists — an incremental update rather than a
/// from-scratch re-summarization. Without this, each compaction re-summarizes the prior summary as raw
/// transcript, compounding information loss and re-billing those tokens every cycle. Matches pi's own
/// `UPDATE_SUMMARIZATION_PROMPT`: explicit per-field update rules (move Done↔In-Progress items, drop
/// resolved Blocked entries, append rather than replace Key Decisions) rather than just "keep it
/// current" — the same granularity gap [`SUMMARY_INSTRUCTION`]'s doc comment describes.
pub const UPDATE_INSTRUCTION: &str = "The messages above are NEW conversation messages to incorporate \
into the existing summary provided in <previous-summary> tags.\n\nUpdate the existing structured \
summary with new information. RULES:\n- PRESERVE all existing information from the previous summary\n- \
ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: move \
items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was \
accomplished\n- PRESERVE exact file paths, function names, and error messages\n- If something is no \
longer relevant, you may remove it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add \
new ones if the task expanded]\n\n## Constraints & Preferences\n- [Preserve existing, add new ones \
discovered]\n\n## Progress\n### Done\n- [x] [Include previously done items AND newly completed \
items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- [Current \
blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all \
previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- \
[Preserve important context, add new if needed]\n\nKeep each section concise. Preserve exact file \
paths, function names, and error messages.";

/// The instruction used when the cut point falls *inside* an in-progress turn (see [`is_split_turn`]):
/// the prefix being summarized ends mid-tool-dispatch, so its concluding response — and the context
/// for it — lives in the kept suffix, not here. A generic "summarize this closed-off history" framing
/// (`SUMMARY_INSTRUCTION`) would ask the model to describe a turn as finished when it isn't; this asks
/// it to preserve the *original request* and *progress so far* instead, framed as still-open context
/// for whatever the kept suffix does next.
pub const SPLIT_TURN_INSTRUCTION: &str = "This is the PREFIX of a turn that was too large to keep. \
The SUFFIX (recent work) is retained.\n\nSummarize the prefix to provide context for the retained \
suffix:\n\n## Original Request\n[What did the user ask for in this turn?]\n\n## Early Progress\n- \
[Key decisions and work done in the prefix]\n\n## Context for Suffix\n- [Information needed to \
understand the retained recent work]\n\nBe concise. Focus on what's needed to understand the kept \
suffix.";

/// Prefix marking a message as a compaction summary, so a later compaction recognizes it and updates
/// it incrementally instead of re-summarizing it. Created by [`apply_summary`], detected by
/// [`previous_summary`].
pub const SUMMARY_MARKER: &str = "[Earlier conversation compacted to save context]";

/// The turn-prefix summarization call's budget, as a fraction of [`CompactionConfig::reserve_tokens`]
/// directly — matching pi's own `generateTurnPrefixSummary` (`Math.floor(0.5 * reserveTokens)`,
/// `compaction.ts`), not a fraction of the history call's already-scaled
/// [`CompactionConfig::summary_max_tokens`] (itself `0.8 * reserve_tokens`), which would instead net an
/// effective ~0.4 * `reserve_tokens`. A split-turn's prefix is a *partial* turn (see [`is_split_turn`]),
/// inherently shorter than the closed-off history it's summarized alongside, so it needs
/// proportionally less room.
pub const SPLIT_TURN_PREFIX_SCALE: f64 = 0.5;

/// Tool-result content is truncated to this many characters when rendered into the summary prompt —
/// tool output is usually the bulk of the transcript and the least useful to re-summarize in full.
const TOOL_RESULT_MAX_CHARS: usize = 2_000;

/// A visible thinking block's text is truncated to this many characters when rendered into the
/// summary prompt — the reasoning trace can run long, and it's supplementary color for *why* a
/// decision was made, not primary content worth spending as much of the prompt's own budget on as an
/// actual user/assistant exchange.
const THINKING_MAX_CHARS: usize = 1_000;

/// Estimate a text's token count. The cheap, dependency-free char/4 heuristic — good enough for the
/// trigger and the cut-point search, which only need to be in the right ballpark.
pub fn estimate_tokens(text: &str) -> u32 {
    (text.chars().count() / 4) as u32
}

/// Estimate one message's token cost from its blocks. `pub(crate)`: also used by
/// [`crate::branch_summary`] to window a branch's transcript by token budget.
///
/// Sums raw character counts across every block first, then applies one `ceil(chars / 4)` — matching
/// pi's own `estimateTokens` (`estimate.ts`), which sums a whole message's chars before its single
/// division. Flooring (or ceiling) *per block* and summing the results instead systematically
/// under-counts a message built of several short blocks (e.g. a tool call's name and its JSON input,
/// each too short alone to clear a whole token at chars/4 floor division, but not once combined).
pub(crate) fn estimate_message_tokens(m: &Message) -> u32 {
    let chars: usize = m
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text, .. } => text.chars().count(),
            ContentBlock::Thinking { text, .. } => text.chars().count(),
            ContentBlock::RedactedThinking { data } => data.chars().count(),
            ContentBlock::ToolUse { name, input, .. } => {
                name.chars().count() + input.to_string().chars().count()
            }
            ContentBlock::ToolResult { content, .. } => content.chars().count(),
            // Matches pi's flat `ESTIMATED_IMAGE_CHARS` (4800, chars/4 = 1200 tokens) for an image
            // (Anthropic bills vision roughly there); the exact figure depends on resolution, but the
            // trigger only needs the right ballpark, and folding it into the same char sum as every
            // other block keeps the single ceiling division below the one true source of rounding.
            ContentBlock::Image { .. } => 4_800,
        })
        .sum();
    chars.div_ceil(4) as u32
}

/// Total estimated tokens across every message in `messages` — pi's own `estimateMessagesTokens`,
/// summing [`estimate_message_tokens`] per message. Unlike [`trailing_tokens`] (a *delta* since the last
/// real usage snapshot), this estimates the whole list from scratch — used right after
/// [`apply_summary`] rebuilds the message list for a post-compaction estimate, a moment at which
/// `trailing_tokens` would just report 0 (`apply_summary` resets its snapshot to point past the
/// freshly-shrunk list's end, so there's no "since last snapshot" delta to speak of yet).
pub(crate) fn estimate_messages_tokens(messages: &[Message]) -> u32 {
    messages
        .iter()
        .map(estimate_message_tokens)
        .fold(0u32, |acc, n| acc.saturating_add(n))
}

/// Estimated tokens in every message appended since [`Session::last_input_tokens`] was last set — the
/// assistant turn that usage snapshot itself came from, plus anything since (tool results, further
/// turns). `should_compact` adds this to `last_input_tokens` so the trigger reflects the prompt's
/// *actual* current size rather than a size that's already stale by the time it's checked (the check
/// runs at the top of the next loop iteration, after that turn's own messages were pushed).
///
/// The very first trailing message — `messages[last_usage_message_count]`, when it's the assistant
/// turn that snapshot was taken for (see [`Session::last_output_tokens`]'s doc comment) — uses the
/// provider's own reported output size instead of the usual char/4 estimate: it's the one message
/// here whose real size is already known exactly, not guessed. Guarded on `Role::Assistant` so this
/// never misapplies to an ordinary user/tool-result message at that same index (e.g. before any turn
/// has completed yet, when both fields are still their zero defaults).
pub fn trailing_tokens(session: &Session) -> u32 {
    let start = session.last_usage_message_count.min(session.messages.len());
    let mut total = 0u32;
    for (i, m) in session.messages[start..].iter().enumerate() {
        let tokens = if i == 0 && m.role == Role::Assistant && session.last_output_tokens > 0 {
            session.last_output_tokens
        } else {
            estimate_message_tokens(m)
        };
        total = total.saturating_add(tokens);
    }
    total
}

/// Whether the live prompt has crossed the compaction threshold. Uses the *last turn's* input size
/// ([`Session::last_input_tokens`]) plus [`trailing_tokens`] (messages appended since that snapshot) —
/// the real, current context size — not the cumulative totals.
///
/// Strict `>`, matching pi's own `shouldCompact` (`compaction.ts`): landing exactly on
/// `context_window - reserve_tokens` still leaves the full reserve intact, so it isn't yet a reason to
/// fire — only actually exceeding it is.
pub fn should_compact(session: &Session, cfg: &CompactionConfig) -> bool {
    session.last_input_tokens > 0
        && session
            .last_input_tokens
            .saturating_add(trailing_tokens(session))
            > cfg.context_window.saturating_sub(cfg.reserve_tokens)
}

/// A hard overflow signal, independent of [`should_compact`]'s (possibly-disabled) soft threshold: the
/// live prompt has already reached — or a turn's own shape strongly implies it's about to reach — the
/// model's *raw* `context_window`, not just the reserve-adjusted budget `should_compact` guards.
/// Disabling [`CompactionConfig::enabled`] is a preference about when to compact *proactively*, with
/// headroom to spare; it isn't license to keep sending requests that are already guaranteed to overflow
/// on the very next turn, so callers check this regardless of that flag.
///
/// Two ways this fires — both driven by the provider's own reported usage, not by parsing an error:
/// - The live prompt (last turn's usage + anything appended since) already meets or exceeds the raw
///   window on its own — a silent overflow no error was ever raised for (the request that pushed it
///   there still succeeded).
/// - `stop_reason` is [`StopReason::MaxTokens`] *and* the live prompt plus the full output budget the
///   next turn would request together would meet or exceed the window — a turn getting cut off by the
///   output-token ceiling is normally unremarkable (a model that simply wanted to keep writing), but
///   when there's this little room left for output, the window itself — not `max_tokens` — is plausibly
///   the actual constraint, and it can only get tighter as the conversation continues to grow.
pub fn is_hard_overflow(
    session: &Session,
    cfg: &CompactionConfig,
    stop_reason: StopReason,
    next_max_tokens: u32,
) -> bool {
    let live_prompt = session
        .last_input_tokens
        .saturating_add(trailing_tokens(session));
    live_prompt >= cfg.context_window
        || (stop_reason == StopReason::MaxTokens
            && live_prompt.saturating_add(next_max_tokens) >= cfg.context_window)
}

/// Choose the first message index to keep verbatim. Walks back from the end accumulating estimated
/// tokens until ~`keep_recent_tokens` is covered, then snaps to the nearest assistant message —
/// **forward** (at or after that point) when one exists, so the kept suffix never retains *more*
/// than the budget, only ever less. (An earlier version of this always snapped backward, which could
/// pull in a whole extra message's worth of tokens ahead of the accumulation point — unbounded if
/// that message happened to be a large tool result.)
///
/// Forward-snapping can run off the end of the array when the accumulation point lands on the very
/// last message and it isn't an assistant message (routinely true: `should_compact` is checked right
/// after a tool-dispatch turn appends a `user` tool-result message, before the next assistant turn
/// exists yet). Rather than decline to compact entirely in that case, this falls back to snapping
/// backward instead — some valid boundary is required for the post-compaction alternation to hold,
/// and retaining slightly more than the budget for one round beats not compacting at all.
///
/// Either direction is equally safe against splitting a `tool_use`/`tool_result` pair *in this
/// codebase's message model specifically*: roles strictly alternate (the loop never emits two
/// consecutive same-role messages — Anthropic rejects that), so a pair always spans exactly
/// `messages[k]` (assistant) / `messages[k+1]` (user), and landing the boundary on the assistant
/// message immediately before *or* immediately after a tool-result message never lands mid-pair. The
/// post-compaction history is `[summary(user), assistant, user, …]` — valid alternation either way.
///
/// Returns `None` when there isn't a worthwhile prefix to summarize (too short, or no clean
/// boundary at all) — the caller then leaves the conversation untouched.
pub fn find_cut(messages: &[Message], keep_recent_tokens: u32) -> Option<usize> {
    let n = messages.len();
    // Need at least a couple of turns to bother — a summary plus a kept suffix.
    if n < 4 {
        return None;
    }
    let mut kept = 0u32;
    let mut idx = n - 1;
    while idx > 0 {
        kept = kept.saturating_add(estimate_message_tokens(&messages[idx]));
        if kept >= keep_recent_tokens {
            break;
        }
        idx -= 1;
    }
    let mut forward = idx;
    while forward < n && messages[forward].role != Role::Assistant {
        forward += 1;
    }
    let first_kept = if forward < n {
        forward
    } else {
        let mut backward = idx;
        while backward > 0 && messages[backward].role != Role::Assistant {
            backward -= 1;
        }
        backward
    };
    // Leave a non-trivial prefix to summarize, and require a real assistant boundary.
    if first_kept == 0 || first_kept >= n || messages[first_kept].role != Role::Assistant {
        return None;
    }
    Some(first_kept)
}

/// [`find_cut`]'s boundary, plus — when the prefix being summarized ends mid-turn (see
/// [`is_split_turn`]) — where that in-progress turn actually *started*, so the caller can summarize
/// the closed-off history and the split turn's own prefix separately instead of collapsing both under
/// the split-turn template (which is written for a partial turn, not a whole conversation's worth of
/// completed ones).
pub struct CutPoint {
    pub first_kept: usize,
    /// `Some(index)` when `messages[..first_kept]` ends mid-turn: the index of the most recent
    /// message that actually starts a turn (real user content, not a bare tool-result reply).
    /// `messages[..turn_start]` is the closed-off history; `messages[turn_start..first_kept]` is the
    /// in-progress turn's own prefix. `None` on a clean boundary — same single-call path as today.
    pub turn_start: Option<usize>,
}

/// Backstop on how far back [`find_split_cut`]'s turn-start scan will look before giving up and
/// treating the whole scanned window as the in-progress turn's own prefix. Deliberately large and
/// independent of `keep_recent_tokens` (which callers may set small, e.g. in tests, to force a
/// specific cut point for the unrelated suffix-selection budget in [`find_cut`]) — a real turn almost
/// never approaches this, so it only bounds worst-case cost on a pathologically long single turn
/// (hundreds of tool round-trips with no intervening genuine user turn) without changing the result
/// of any realistic one.
const MAX_SPLIT_TURN_SCAN_TOKENS: u32 = 200_000;

/// [`find_cut`], plus the split-turn boundary described on [`CutPoint`].
pub fn find_split_cut(messages: &[Message], keep_recent_tokens: u32) -> Option<CutPoint> {
    let first_kept = find_cut(messages, keep_recent_tokens)?;
    let turn_start = is_split_turn(&messages[..first_kept]).then(|| {
        // Bounded by `MAX_SPLIT_TURN_SCAN_TOKENS`: stop once that much of the in-progress turn has
        // been scanned, even without a genuine turn-start yet, rather than walking the entire
        // (potentially session-spanning) prefix. `is_split_turn`'s gate is common — true after any
        // multi-tool-call turn — so an unbounded scan here would run on most real compactions.
        // Landing on the cap instead of the turn's true start is a safe approximation, not an
        // incorrect one: everything from the cap forward is still treated as the in-progress turn's
        // own prefix (summarized with the split-turn template), and anything before it as ordinary
        // closed-off history — just a coarser split than scanning further back would have found.
        let mut scanned_tokens = 0u32;
        let mut cut = 0;
        for i in (0..first_kept).rev() {
            cut = i;
            if is_turn_start(&messages[i]) {
                break;
            }
            scanned_tokens = scanned_tokens.saturating_add(estimate_message_tokens(&messages[i]));
            if scanned_tokens >= MAX_SPLIT_TURN_SCAN_TOKENS {
                break;
            }
        }
        // The scan bottomed out at index 0 without finding any genuine turn-start in between — the
        // whole window back to the start of the session is one still-open turn. If that first message
        // is a prior compaction's summary (always `messages[0]`; see `apply_summary`), it's closed-off
        // history, not part of the open turn — bump past it so the summary itself becomes (all of) the
        // history side of the split, and `compact`'s incremental-update path folds it forward instead
        // of misreading `turn_start == 0` as "no prior history exists" and burying the summary inside
        // the split-turn template meant for a partial turn's own activity.
        if cut == 0 && previous_summary(&messages[..1]).is_some() {
            cut = 1;
        }
        cut
    });
    Some(CutPoint {
        first_kept,
        turn_start,
    })
}

/// Whether `m` is a genuine turn-start: a user message carrying real content (text, an image — anything
/// that isn't purely a reply to a prior tool dispatch). Distinguishes "the user asked something new"
/// from the `ToolResult`-only user messages the loop appends between rounds of one still-open turn —
/// and from a prior compaction's own summary message (see [`SUMMARY_MARKER`]): it's `Role::User` with
/// real text too, but it's closed-off history recording *past* activity, not the user asking something
/// new right now.
fn is_turn_start(m: &Message) -> bool {
    m.role == Role::User
        && m.content
            .iter()
            .any(|b| !matches!(b, ContentBlock::ToolResult { .. }))
        && !matches!(
            m.content.first(),
            Some(ContentBlock::Text { text, .. }) if text.starts_with(SUMMARY_MARKER)
        )
}

/// Render the prefix `messages` into a plain-text transcript for the summarization prompt. Tool
/// results — usually the bulk — are truncated, as is a visible thinking block's text (supplementary
/// context on *why* a decision was made, not primary content); `RedactedThinking` is always dropped —
/// its `data` is opaque ciphertext the provider chose not to expose, never human-readable regardless
/// of length.
pub fn render_prefix(messages: &[Message]) -> String {
    render_prefix_impl(messages, true)
}

/// Same rendering as [`render_prefix`], but drops `ToolResult` content entirely instead of rendering a
/// truncated body — used only by [`crate::branch_summary`]. Matches pi's own `getMessageFromEntry`
/// (`branch-summarization.ts`), which returns `undefined` for any tool-result-role entry so
/// `prepareBranchEntries` never includes tool output in a branch summary's rendered transcript at all;
/// unlike a from-scratch compaction summary (which stays live conversation context), a branch summary
/// is read once on return, so tool output's terseness matters more than its (already-truncated) detail.
/// Tool-call *names* (via `ToolUse`) still survive, same as ordinary compaction.
pub(crate) fn render_prefix_without_tool_results(messages: &[Message]) -> String {
    render_prefix_impl(messages, false)
}

fn render_prefix_impl(messages: &[Message], include_tool_results: bool) -> String {
    let mut out = String::new();
    for m in messages {
        for b in &m.content {
            match b {
                ContentBlock::Text { text, .. } => {
                    let who = match m.role {
                        Role::User => "User",
                        Role::Assistant => "Assistant",
                        Role::System => "System",
                    };
                    out.push_str(who);
                    out.push_str(": ");
                    out.push_str(text);
                    out.push('\n');
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    out.push_str(&format!("Assistant called tool `{name}`({input})\n"));
                }
                ContentBlock::ToolResult {
                    content, is_error, ..
                } => {
                    if !include_tool_results {
                        continue;
                    }
                    let body = truncate_chars(content, TOOL_RESULT_MAX_CHARS);
                    let tag = if *is_error {
                        "tool error"
                    } else {
                        "tool result"
                    };
                    out.push_str(&format!("[{tag}] {body}\n"));
                }
                ContentBlock::Image { .. } => out.push_str("[image]\n"),
                ContentBlock::Thinking { text, .. } => {
                    let body = truncate_chars(text, THINKING_MAX_CHARS);
                    out.push_str(&format!("[thinking] {body}\n"));
                }
                // Encrypted thinking the provider chose not to expose in clear text — opaque
                // ciphertext, never worth spending prompt budget on regardless of length.
                ContentBlock::RedactedThinking { .. } => {}
            }
        }
    }
    out
}

/// Truncate a string to at most `max` characters, appending a marker that notes the elision.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max).collect();
    let dropped = s.chars().count() - max;
    format!("{kept}… [{dropped} more characters truncated]")
}

/// Files the prefix read vs. modified, extracted from `read`/`write`/`edit` tool calls. Appended to
/// the summary so the model keeps file awareness across the cut.
pub fn extract_file_ops(messages: &[Message]) -> (Vec<String>, Vec<String>) {
    let mut read = Vec::new();
    let mut modified = Vec::new();
    for m in messages {
        for b in &m.content {
            if let ContentBlock::ToolUse { name, input, .. } = b {
                let Some(path) = input.get("path").and_then(|p| p.as_str()) else {
                    continue;
                };
                let bucket = match name.as_str() {
                    "read" | "ls" => &mut read,
                    "write" | "edit" => &mut modified,
                    _ => continue,
                };
                let path = path.to_string();
                if !bucket.contains(&path) {
                    bucket.push(path);
                }
            }
        }
    }
    (read, modified)
}

/// If `prefix` begins with a prior compaction summary (a user message tagged with [`SUMMARY_MARKER`]),
/// return the summary body (without the marker line, or `apply_summary`'s leading token-count line).
/// This is what lets compaction be *incremental*: the previous summary is fed forward as-is and
/// updated, not re-summarized as transcript — so the token-count line (informational metadata for an
/// exported transcript, not part of the summary itself) is stripped before it would otherwise leak into
/// the next summarization prompt.
pub fn previous_summary(prefix: &[Message]) -> Option<&str> {
    let first = prefix.first()?;
    if first.role != Role::User {
        return None;
    }
    let ContentBlock::Text { text, .. } = first.content.first()? else {
        return None;
    };
    let body = text.strip_prefix(SUMMARY_MARKER)?.trim_start();
    let body = strip_leading_tokens_before_line(body);
    Some(body)
}

/// Strip an optional leading `"Compacted from {N} tokens\n\n"` line (see `apply_summary`) if present;
/// otherwise returns `body` unchanged.
fn strip_leading_tokens_before_line(body: &str) -> &str {
    body.strip_prefix("Compacted from ")
        .and_then(|rest| rest.split_once(" tokens\n\n"))
        .map(|(_, rest)| rest)
        .unwrap_or(body)
}

/// Whether `prefix` (what's about to be summarized) ends *mid-turn* — its cut point falls inside an
/// in-progress tool-dispatch sequence rather than at a genuine turn boundary. `find_cut` always snaps
/// to an assistant message, but that assistant message can be a **continuation** of dispatching
/// several tool calls for one earlier user request, not necessarily the first response to a fresh one
/// — in which case the request that motivated `prefix`'s trailing work is itself inside `prefix`,
/// summarized away, while the *conclusion* of that same turn lives in the kept suffix.
///
/// Detected by shape: a mid-turn tool round-trip always ends with a user message carrying at least one
/// `ToolResult` block — the loop appends exactly that after dispatching a batch of tool calls, before
/// the assistant's next (possibly still-mid-turn) response. LOW pi-parity gap (fixed): this used to
/// require *every* block to be a `ToolResult` (no free text at all), which missed a message that's
/// still fundamentally a tool-dispatch continuation but also carries a mid-run steer's injected text
/// block (`Agent::run_events_steered` folds a queued steer message onto this same turn, see
/// `steering.rs`) — a message model with no genuine-fresh-turn path that could ever produce a
/// `ToolResult` block at all, so its mere presence is unambiguous regardless of what else rides along.
pub fn is_split_turn(prefix: &[Message]) -> bool {
    match prefix.last() {
        Some(m) => {
            m.role == Role::User
                && m.content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
        }
        None => false,
    }
}

/// Combine a [`find_split_cut`] round's two independently-generated summaries — the closed-off
/// history and the in-progress turn's own prefix — into the one summary spliced into the session.
pub fn merge_split_summary(history: &str, turn_prefix: &str) -> String {
    format!("{history}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix}")
}

/// Build the (network-free) summarization request: the rendered prefix plus the structured
/// instruction and file-op tags, as a single user turn. One self-contained message sidesteps any
/// role-alternation constraints, and capping tool results keeps the prompt well under the window.
///
/// When `prefix` already starts with a prior summary, this builds an *incremental update* instead:
/// the previous summary is presented verbatim, only the activity since it is rendered as transcript,
/// and the model is asked to merge them — so successive compactions don't re-summarize (and lose) the
/// same early context over and over.
///
/// `file_ops` is the *already-merged* provenance (see [`merge_file_ops`]) — the read/modified-file
/// tags embedded in the prompt reflect every round so far, not just this one's new activity, so the
/// model doesn't lose file awareness the previous round recorded.
///
/// `custom_instructions`, when given, is appended to whichever instruction template applies —
/// "Additional focus: {custom_instructions}" — matching pi's own `generateSummary`. This is how a
/// manual (client-triggered) compaction can steer *what* the summary emphasizes (e.g. "keep every
/// detail about the auth refactor") without replacing the structured Markdown format itself.
pub fn summary_request(
    model: &str,
    prefix: &[Message],
    max_tokens: u32,
    file_ops: &CompactionProvenance,
    custom_instructions: Option<&str>,
) -> ModelRequest {
    let (read, modified) = (&file_ops.read_files, &file_ops.modified_files);

    let mut prompt = String::new();
    let instruction = match previous_summary(prefix) {
        Some(prev) => {
            // Incremental: previous summary verbatim, then only the new activity (prefix[1..]).
            prompt.push_str("<previous-summary>\n");
            prompt.push_str(prev);
            prompt.push_str("\n</previous-summary>\n\n<new-activity>\n");
            prompt.push_str(&render_prefix(&prefix[1..]));
            prompt.push_str("</new-activity>\n\n");
            UPDATE_INSTRUCTION
        }
        None => {
            // Matches pi's own `generateSummary`/`generateTurnPrefixSummary`: the rendered transcript
            // is wrapped in `<conversation>` tags, the same delimiting convention the incremental path
            // just below already applies to its own new-activity transcript via `<new-activity>`.
            prompt.push_str("<conversation>\n");
            prompt.push_str(&render_prefix(prefix));
            prompt.push_str("\n</conversation>\n\n");
            if is_split_turn(prefix) {
                SPLIT_TURN_INSTRUCTION
            } else {
                SUMMARY_INSTRUCTION
            }
        }
    };
    if !read.is_empty() {
        prompt.push_str(&format!(
            "<read-files>\n{}\n</read-files>\n",
            read.join("\n")
        ));
    }
    if !modified.is_empty() {
        prompt.push_str(&format!(
            "<modified-files>\n{}\n</modified-files>\n",
            modified.join("\n")
        ));
    }
    prompt.push_str(instruction);
    if let Some(custom) = custom_instructions {
        prompt.push_str("\n\nAdditional focus: ");
        prompt.push_str(custom);
    }

    ModelRequest::new(model, Arc::new(vec![Message::user(prompt)]), max_tokens)
        .with_system(SUMMARY_SYSTEM)
}

/// Render `read_files`/`modified_files` as the same `<read-files>`/`<modified-files>` tags
/// [`summary_request`] embeds in its *prompt* — but meant to be appended to the summarization model's
/// *output* instead, so the file list survives into the live conversation deterministically rather
/// than depending on the summarizing model choosing to mention exact paths in its own prose. Matches
/// pi's `formatFileOperations` (`compaction/utils.ts`), including the leading `"\n\n"` separator and
/// returning `""` when both lists are empty (so appending this to a summary is always safe,
/// unconditionally). Callers: [`crate::agent::Agent::compact`], [`crate::agent::Agent::summarize_branch`].
pub fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections = Vec::new();
    if !read_files.is_empty() {
        sections.push(format!(
            "<read-files>\n{}\n</read-files>",
            read_files.join("\n")
        ));
    }
    if !modified_files.is_empty() {
        sections.push(format!(
            "<modified-files>\n{}\n</modified-files>",
            modified_files.join("\n")
        ));
    }
    if sections.is_empty() {
        String::new()
    } else {
        format!("\n\n{}", sections.join("\n\n"))
    }
}

/// Splice a summary into `session`, replacing the prefix before `first_kept` with one summary user
/// message. Resets [`Session::last_input_tokens`] so the freshly-shrunk context doesn't immediately
/// re-trigger the threshold (the true size is recomputed after the next turn). `tokens_before` (the
/// pre-compaction size the caller already computed) is embedded as a leading `"Compacted from {N}
/// tokens\n\n"` line in the summary body — read back out by `export.rs`'s `parse_compaction_tokens_before`
/// to show a token count in an exported transcript's compaction block, matching pi's own
/// `entry.tokensBefore` (`template.js:1294-1299`). `previous_summary` strips this line back out before
/// handing the body to an incremental re-summarization prompt.
pub fn apply_summary(session: &mut Session, first_kept: usize, summary: &str, tokens_before: u32) {
    let kept = &session.messages[first_kept..];
    let mut new_messages = Vec::with_capacity(1 + kept.len());
    new_messages.push(Message::user(format!(
        "{SUMMARY_MARKER}\n\nCompacted from {tokens_before} tokens\n\n{summary}"
    )));
    new_messages.extend_from_slice(kept);
    session.messages = Arc::new(new_messages);
    session.last_input_tokens = 0;
    // Rebase against the freshly-shrunk message list — a stale index from before the splice would
    // otherwise either panic (out of bounds) or, worse, silently under/over-count `trailing_tokens`
    // against messages that no longer exist at those positions.
    session.last_usage_message_count = session.messages.len();
    // No message lives at the new `last_usage_message_count` yet (it points one past the end), so
    // there's nothing this could correctly describe until the next real `record_usage` sets it fresh —
    // stale, reset alongside the index it's paired with rather than left to (harmlessly, since the
    // trailing slice is empty either way) dangle.
    session.last_output_tokens = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn convo() -> Vec<Message> {
        vec![
            Message::user("the original task: refactor foo"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "src/foo.rs" }),
            )]),
            Message::tool_result("1", "fn foo() {}", false),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "src/foo.rs" }),
            )]),
            Message::tool_result("2", "edited", false),
            Message::assistant(vec![ContentBlock::text("done")]),
        ]
    }

    #[test]
    fn estimate_message_tokens_ceils_once_over_the_whole_message_not_per_block() {
        // pi-parity gap (fixed): pi's `estimateTokens` (`estimate.ts`) sums a whole message's chars
        // before its single `Math.ceil(chars / 4)`. Flooring (the old per-block behavior, via
        // `estimate_tokens`'s chars/4) each block separately then summing systematically under-counts a
        // message built of several short blocks — three 3-char text blocks (9 chars total, ceil(9/4) =
        // 3 tokens) each individually floor to 0 tokens alone (3/4 = 0), summing to 0 instead.
        let m = Message::assistant(vec![
            ContentBlock::text("abc"),
            ContentBlock::text("def"),
            ContentBlock::text("ghi"),
        ]);
        assert_eq!(
            estimate_message_tokens(&m),
            3,
            "9 total chars across 3 blocks must ceil-divide once as a whole (ceil(9/4) = 3), not \
             floor-divide per block and sum (0 + 0 + 0 = 0)"
        );
    }

    #[test]
    fn summary_instruction_templates_give_pi_parity_structural_guidance() {
        // pi-parity gap: the templates used to be bare heading names with one line of framing — no
        // bracketed exemplars (what kind of content goes in each section), no "(none)" fallback for
        // sections that can legitimately be empty, no checkbox/numbering conventions. Matches pi's own
        // `SUMMARIZATION_PROMPT`/`UPDATE_SUMMARIZATION_PROMPT` (`compaction.ts`) structurally.
        for (name, template) in [
            ("SUMMARY_INSTRUCTION", SUMMARY_INSTRUCTION),
            ("UPDATE_INSTRUCTION", UPDATE_INSTRUCTION),
        ] {
            assert!(
                template.contains("- [x]") && template.contains("- [ ]"),
                "{name} must use checkbox conventions for Done/In Progress: {template}"
            );
            assert!(
                template.contains("1. ["),
                "{name}'s Next Steps must be a numbered (not bulleted) list: {template}"
            );
            assert!(
                template.contains("## Goal\n["),
                "{name} must give a bracketed exemplar right after the Goal heading, not a bare \
                 heading: {template}"
            );
        }
        // Only the from-scratch template has optional sections that can legitimately be empty (the
        // update template's rules are phrased as edits to what's already there, so it has no
        // equivalent "or (none)" fallback — matching pi's own asymmetry between the two prompts).
        assert!(
            SUMMARY_INSTRUCTION.contains("\"(none)\""),
            "SUMMARY_INSTRUCTION must give an explicit \"(none)\" fallback for optional sections: \
             {SUMMARY_INSTRUCTION}"
        );
    }

    #[test]
    fn should_compact_tracks_last_input_against_window() {
        let cfg = CompactionConfig {
            context_window: 1000,
            reserve_tokens: 200,
            ..Default::default()
        };
        let mut s = Session::new();
        assert!(!should_compact(&s, &cfg)); // no usage yet
        s.last_input_tokens = 700;
        assert!(!should_compact(&s, &cfg)); // below 1000-200
        s.last_input_tokens = 850;
        assert!(should_compact(&s, &cfg));
    }

    #[test]
    fn should_compact_does_not_fire_exactly_at_the_threshold() {
        // pi-parity gap (fixed): pi's `shouldCompact` fires on strict `>`, not `>=` — landing exactly
        // on `context_window - reserve_tokens` still leaves the full reserve intact, so it isn't yet a
        // reason to fire. One token past it must.
        let cfg = CompactionConfig {
            context_window: 1000,
            reserve_tokens: 200, // threshold: 800
            ..Default::default()
        };
        let mut s = Session::new();
        s.last_input_tokens = 800;
        assert!(
            !should_compact(&s, &cfg),
            "must not fire when the live prompt merely meets the threshold"
        );
        s.last_input_tokens = 801;
        assert!(
            should_compact(&s, &cfg),
            "must fire once the live prompt actually exceeds the threshold"
        );
    }

    #[test]
    fn should_compact_counts_trailing_messages_appended_since_last_usage_snapshot() {
        // The last recorded usage snapshot sits just under the threshold (gap of 50 tokens) — but
        // messages appended *since* that snapshot (a turn's tool results, further turns) aren't
        // reflected in `last_input_tokens` yet. Without accounting for them, the trigger would stay
        // stuck below threshold until the *next* model call reports a fresh (and by then already-over-
        // budget) size — a full extra round late. `trailing_tokens` closes that gap immediately.
        let cfg = CompactionConfig {
            context_window: 1000,
            reserve_tokens: 200, // threshold: 800
            ..Default::default()
        };
        let mut s = Session::new();
        s.last_input_tokens = 750; // 50 tokens under threshold
        s.last_usage_message_count = s.messages.len(); // snapshot taken with no messages yet
        assert!(!should_compact(&s, &cfg));

        // ~75 estimated tokens' worth of message appended since the snapshot — comfortably over the
        // 50-token gap.
        s.push(Message::user("x".repeat(300)));
        assert!(
            should_compact(&s, &cfg),
            "trailing tokens since the last usage snapshot must push the trigger over threshold"
        );
    }

    #[test]
    fn trailing_tokens_uses_the_precise_reported_output_size_for_the_just_completed_turn() {
        // The assistant's own reply text is trivially short (~0 estimated tokens via the char/4
        // heuristic), but the provider reported a much larger real `output_tokens` figure for that
        // same turn. `trailing_tokens` must use the precise reported number for that one message, not
        // re-derive a rough (and here, badly wrong) estimate from its rendered text.
        let mut s = Session::new();
        s.user("go");
        s.last_usage_message_count = s.messages.len(); // snapshot taken right before the reply
        s.last_output_tokens = 5_000;
        s.push(Message::assistant(vec![ContentBlock::text("ok")])); // ~0 tokens by char/4
        assert_eq!(trailing_tokens(&s), 5_000);
    }

    #[test]
    fn trailing_tokens_still_estimates_messages_after_the_just_completed_turn() {
        // Only the message at exactly `last_usage_message_count` gets the precise substitution —
        // anything appended *after* it (a further turn, tool results) still falls back to the usual
        // char/4 heuristic, since there's no provider-reported figure for those.
        let mut s = Session::new();
        s.user("go");
        s.last_usage_message_count = s.messages.len();
        s.last_output_tokens = 100;
        s.push(Message::assistant(vec![ContentBlock::text("ok")]));
        s.push(Message::user("x".repeat(400))); // ~100 estimated tokens
        assert_eq!(trailing_tokens(&s), 200);
    }

    #[test]
    fn trailing_tokens_ignores_last_output_tokens_when_the_message_at_the_snapshot_is_not_assistant()
     {
        // Defensive guard: `last_output_tokens` only ever describes an assistant turn's own output
        // (set by `record_usage`, which always runs right before that message is pushed) — if the
        // message actually sitting at `last_usage_message_count` isn't an assistant message (e.g. the
        // session's initial state, before any turn has completed), the field must not be misapplied
        // to it.
        let mut s = Session::new();
        s.last_output_tokens = 5_000; // stale/default-adjacent value, must not leak in here
        s.push(Message::user("x".repeat(40))); // ~10 estimated tokens
        assert_eq!(trailing_tokens(&s), 10);
    }

    #[test]
    fn should_compact_fires_one_turn_earlier_using_the_precise_output_token_count() {
        // Threshold = context_window - reserve_tokens = 5_000. The last recorded prompt size alone
        // (100) is nowhere near it, and the assistant's own reply text is trivially short — the old
        // char/4-only estimate would have stayed far under threshold, deferring the trigger a full
        // extra turn. The provider's real reported output size for that same turn (5_000) closes the
        // gap immediately.
        let cfg = CompactionConfig {
            context_window: 6_000,
            reserve_tokens: 1_000,
            ..Default::default()
        };
        let mut s = Session::new();
        s.user("go");
        s.last_input_tokens = 100;
        s.last_usage_message_count = s.messages.len();
        s.last_output_tokens = 5_000;
        s.push(Message::assistant(vec![ContentBlock::text("ok")]));
        assert!(
            should_compact(&s, &cfg),
            "100 (last prompt) + 5_000 (this turn's real output) meets the 5_000 threshold"
        );
    }

    #[test]
    fn apply_summary_resets_last_output_tokens_alongside_the_usage_snapshot_it_pairs_with() {
        let mut s = Session::new();
        s.messages = Arc::new(convo());
        s.last_output_tokens = 12345;
        apply_summary(&mut s, 3, "SUMMARY", 9999);
        assert_eq!(s.last_output_tokens, 0);
    }

    #[test]
    fn is_hard_overflow_fires_when_the_live_prompt_already_meets_the_raw_window() {
        // No error was ever raised — the turn that pushed usage this high still succeeded — so this is
        // exactly the "silent" overflow case: the raw window (not the softer reserve-adjusted
        // threshold `should_compact` checks) has already been met.
        let cfg = CompactionConfig {
            context_window: 1000,
            reserve_tokens: 200,
            ..Default::default()
        };
        let mut s = Session::new();
        s.last_input_tokens = 1000;
        assert!(is_hard_overflow(&s, &cfg, StopReason::EndTurn, 0));
    }

    #[test]
    fn is_hard_overflow_is_false_well_under_the_window_even_on_a_max_tokens_stop() {
        // A `MaxTokens` stop reason alone must not force compaction — that's just a model that wanted
        // to keep writing, extremely common and unrelated to context pressure, when there's plenty of
        // window left for the next turn's full output budget too.
        let cfg = CompactionConfig {
            context_window: 100_000,
            reserve_tokens: 10_000,
            ..Default::default()
        };
        let mut s = Session::new();
        s.last_input_tokens = 10_000;
        assert!(!is_hard_overflow(&s, &cfg, StopReason::MaxTokens, 4_096));
    }

    #[test]
    fn is_hard_overflow_fires_on_a_max_tokens_stop_when_the_output_budget_would_meet_the_window() {
        // The live prompt alone is under the raw window, but a `MaxTokens` stop plus the *next* turn's
        // requested output budget would meet it — a sign the window itself, not `max_tokens`, is the
        // actual constraint on how much room the model had to keep writing.
        let cfg = CompactionConfig {
            context_window: 10_000,
            reserve_tokens: 1_000,
            ..Default::default()
        };
        let mut s = Session::new();
        s.last_input_tokens = 9_500;
        assert!(!is_hard_overflow(&s, &cfg, StopReason::EndTurn, 1_000));
        assert!(
            is_hard_overflow(&s, &cfg, StopReason::MaxTokens, 1_000),
            "9_500 + 1_000 meets the 10_000 window under a MaxTokens stop"
        );
    }

    #[test]
    fn is_hard_overflow_counts_trailing_messages_since_the_last_usage_snapshot() {
        // Same "messages appended since the snapshot aren't in `last_input_tokens` yet" gap
        // `should_compact` accounts for via `trailing_tokens` — this check must too.
        let cfg = CompactionConfig {
            context_window: 1_000,
            ..Default::default()
        };
        let mut s = Session::new();
        s.last_input_tokens = 700;
        s.last_usage_message_count = s.messages.len();
        assert!(!is_hard_overflow(&s, &cfg, StopReason::EndTurn, 0));
        s.push(Message::user("x".repeat(2_000))); // ~500 estimated tokens
        assert!(is_hard_overflow(&s, &cfg, StopReason::EndTurn, 0));
    }

    #[test]
    fn find_cut_snaps_to_assistant_and_leaves_both_sides() {
        let messages = convo();
        // Keep a tiny budget so the cut lands late, then snaps to an assistant boundary.
        let cut = find_cut(&messages, 1).expect("a cut");
        assert_eq!(messages[cut].role, Role::Assistant);
        assert!(cut > 0 && cut < messages.len());
    }

    #[test]
    fn find_cut_snaps_forward_never_retaining_more_than_the_budget() {
        // `convo()`: [0 user, 1 assistant(tool_use read), 2 user(tool_result), 3 assistant(tool_use
        // edit), 4 user(tool_result "edited"), 5 assistant(text "done")]. Budget 2: accumulating from
        // the end, message 5 ("done", ~1 token) isn't enough alone, so the walk continues to message 4
        // ("edited", ~1 token) — cumulative 2, stop at idx=4, a *user* (tool_result) message.
        let messages = convo();
        let cut = find_cut(&messages, 2).expect("a cut");
        // Forward-snap lands on message 5 (the next assistant boundary *after* idx=4) — the
        // tool_use/tool_result pair at [3, 4] is fully summarized, not partially retained. A prior
        // backward-snapping implementation would have landed on message 3 instead, retaining both
        // messages 3 and 4 — strictly more than the 2-token budget asked for.
        assert_eq!(cut, 5);
        assert_eq!(messages[cut].role, Role::Assistant);
        // The kept suffix (from `cut` onward) never exceeds the budget by more than the one boundary
        // message itself — unlike backward-snap, it can't additionally pull in an *earlier* sibling
        // message (here, message 3) ahead of the accumulation point.
        let kept_tokens: u32 = messages[cut..].iter().map(estimate_message_tokens).sum();
        assert!(kept_tokens <= 2, "kept {kept_tokens} tokens, budget was 2");
    }

    #[test]
    fn find_split_cut_is_none_on_clean_boundary() {
        // Two fully closed text-only turns; the cut lands right after the second one ends cleanly on
        // an assistant reply — no split-turn boundary anywhere in sight.
        let messages = vec![
            Message::user(
                "first request, long enough that its token estimate is comfortably nonzero",
            ),
            Message::assistant(vec![ContentBlock::text(
                "first done, a fairly long response so the estimate is nontrivial too",
            )]),
            Message::user(
                "second request, also long enough for a nonzero token estimate here as well",
            ),
            Message::assistant(vec![ContentBlock::text(
                "second done, another long enough response for the estimate to register",
            )]),
        ];
        let cut = find_split_cut(&messages, 1).expect("a cut");
        assert_eq!(cut.first_kept, 3);
        assert_eq!(
            cut.turn_start, None,
            "a clean boundary must not report a split-turn start"
        );
    }

    #[test]
    fn find_split_cut_reports_turn_start_on_mid_turn_boundary() {
        // Turn 1 (closed): user -> assistant(text). Turn 2 (still open): user -> assistant(tool_use)
        // -> user(tool_result) -> assistant(tool_use) -> user(tool_result) — mid-dispatch, no
        // concluding assistant reply yet.
        let messages = vec![
            Message::user("first request"), // 0
            Message::assistant(vec![ContentBlock::text("first done")]),
            Message::user("second request"), // 2 — the split turn's real start
            Message::assistant(vec![ContentBlock::tool_use(
                // 3
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("1", "contents of a.rs", false), // 4
            Message::assistant(vec![ContentBlock::tool_use(
                // 5
                "2",
                "edit",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("2", "edited", false), // 6
        ];
        // A tiny budget forces the cut to land as late as possible; the trailing tool_result-only
        // message at index 6 makes it snap (forward has nowhere to go, so it snaps backward) to the
        // assistant message at index 5 — a split-turn boundary.
        let cut = find_split_cut(&messages, 1).expect("a cut");
        assert_eq!(cut.first_kept, 5);
        assert_eq!(
            cut.turn_start,
            Some(2),
            "must find the split turn's own start (the 'second request' message), not the first one"
        );
    }

    #[test]
    fn find_split_cut_scan_is_bounded_and_does_not_walk_the_whole_prefix() {
        // A closed turn, then a pathologically long single turn (many tool round-trips, each with a
        // huge tool_result) with no intervening genuine user turn until the very start. Without a
        // bound, the turn-start scan would walk the entire chain back to index 2 (the "second
        // request" message, the turn's true start); `MAX_SPLIT_TURN_SCAN_TOKENS` must stop it well
        // before that instead.
        let big = "x".repeat(400_000); // ~100,000 estimated tokens (chars/4) per message
        let mut messages = vec![
            Message::user("first request"), // 0 — a fully closed turn
            Message::assistant(vec![ContentBlock::text("first done")]), // 1
            Message::user("second request"), // 2 — the split turn's real (but far-back) start
        ];
        for i in 0..10 {
            messages.push(Message::assistant(vec![ContentBlock::tool_use(
                i.to_string(),
                "read",
                json!({ "path": "a.rs" }),
            )]));
            messages.push(Message::tool_result(i.to_string(), big.clone(), false));
        }
        let cut = find_split_cut(&messages, 1).expect("a cut");
        let turn_start = cut
            .turn_start
            .expect("a trailing bare tool_result is a split-turn boundary");
        assert_ne!(
            turn_start, 2,
            "the bound must stop the scan before reaching the true (far-back) turn start"
        );
        assert_ne!(
            turn_start, 0,
            "must not fall all the way back to index 0 either"
        );
    }

    #[test]
    fn find_split_cut_treats_a_prior_summary_as_history_not_as_the_turn_start() {
        // After one compaction, `messages[0]` is always the summary marker message (`apply_summary`
        // always splices it in at the front). If the entire conversation since then is one continuous,
        // still-open turn — no genuine new user message anywhere — the backward scan would otherwise
        // walk all the way back to the summary and misidentify it as the split turn's own start: it's
        // `Role::User` with real text, the same shape `is_turn_start` looks for. That would report
        // `turn_start == 0`, which `Agent::compact` reads as "nothing precedes the split turn" — losing
        // the fact that a real prior summary exists. It must instead land at 1, folding the summary
        // into the closed-off history side of the split.
        let messages = vec![
            Message::user(format!("{SUMMARY_MARKER}\n\nprior summary body")), // 0
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]), // 1
            Message::tool_result("1", "contents of a.rs", false),             // 2
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "a.rs" }),
            )]), // 3
            Message::tool_result("2", "edited", false),                       // 4
        ];
        let cut = find_split_cut(&messages, 1).expect("a cut");
        assert_eq!(cut.first_kept, 3);
        assert_eq!(
            cut.turn_start,
            Some(1),
            "the summary at index 0 must be excluded from the split turn's own prefix"
        );
    }

    #[test]
    fn is_turn_start_rejects_a_summary_marker_message() {
        let summary = Message::user(format!("{SUMMARY_MARKER}\n\nprior summary body"));
        assert!(
            !is_turn_start(&summary),
            "a prior compaction's summary is closed-off history, not a fresh user turn"
        );
    }

    #[test]
    fn find_cut_never_splits_a_tool_use_tool_result_pair() {
        // A longer conversation with several tool round-trips — for every budget size, the chosen cut
        // must never leave a `tool_result` on the kept side without its originating `tool_use` (or
        // vice versa), regardless of which side of a pair the raw accumulation index landed on.
        let mut messages = vec![Message::user("start")];
        for i in 0..8 {
            messages.push(Message::assistant(vec![ContentBlock::tool_use(
                i.to_string(),
                "read",
                json!({ "path": format!("f{i}.rs") }),
            )]));
            messages.push(Message::tool_result(
                i.to_string(),
                format!("contents of f{i}"),
                false,
            ));
        }
        messages.push(Message::assistant(vec![ContentBlock::text("done")]));

        for budget in [1u32, 3, 5, 8, 12, 20, 50] {
            let Some(cut) = find_cut(&messages, budget) else {
                continue;
            };
            assert_eq!(
                messages[cut].role,
                Role::Assistant,
                "budget {budget}: cut must land on an assistant boundary"
            );
            // If a tool_result is kept, its tool_use (the immediately preceding message) must be too.
            for (i, m) in messages.iter().enumerate().skip(cut) {
                if matches!(m.content.first(), Some(ContentBlock::ToolResult { .. })) {
                    assert!(
                        i > 0 && i > cut,
                        "budget {budget}: tool_result at {i} is kept but its tool_use at {} is not",
                        i.saturating_sub(1)
                    );
                }
            }
        }
    }

    #[test]
    fn find_cut_declines_short_conversations() {
        assert!(find_cut(&convo()[..3], 1).is_none());
    }

    #[test]
    fn extract_file_ops_buckets_reads_and_writes() {
        let (read, modified) = extract_file_ops(&convo());
        assert_eq!(read, vec!["src/foo.rs"]);
        assert_eq!(modified, vec!["src/foo.rs"]);
    }

    #[test]
    fn summary_request_truncates_tool_results_and_tags_files() {
        let big = "x".repeat(5000);
        let messages = vec![
            Message::user("task"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("1", &big, false),
            // A concluding assistant response — without this the prefix ends on a bare tool_result,
            // which `is_split_turn` (correctly) reads as an in-progress turn and would swap in
            // `SPLIT_TURN_INSTRUCTION` instead of the generic template this test is exercising.
            Message::assistant(vec![ContentBlock::text("done")]),
        ];
        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &messages,
            CompactionReason::Manual,
        );
        let req = summary_request("claude-test", &messages, 512, &file_ops, None);
        let text = match &req.messages[0].content[0] {
            ContentBlock::Text { text, .. } => text,
            other => panic!("expected text, got {other:?}"),
        };
        assert!(text.contains("characters truncated"));
        assert!(text.contains("<read-files>"));
        assert!(text.contains("a.rs"));
        assert!(text.contains("## Goal"));
        assert_eq!(req.system.as_deref(), Some(SUMMARY_SYSTEM));
    }

    #[test]
    fn summary_request_wraps_the_fresh_transcript_in_conversation_tags() {
        // pi-parity gap (fixed): pi's `generateSummary`/`generateTurnPrefixSummary` always wrap the
        // rendered transcript in `<conversation>` tags — matching our own incremental-update path's
        // `<new-activity>` wrapper for internal consistency.
        let messages = convo();
        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &messages,
            CompactionReason::Threshold,
        );
        let req = summary_request("claude-test", &messages, 512, &file_ops, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("<conversation>"));
        assert!(text.contains("</conversation>"));
        let open = text.find("<conversation>").unwrap();
        let close = text.find("</conversation>").unwrap();
        assert!(open < close);
        assert!(
            text[open..close].contains("refactor foo"),
            "the rendered transcript must live inside the conversation tags: {text}"
        );
    }

    #[test]
    fn summary_request_does_not_wrap_the_incremental_update_path_in_conversation_tags() {
        // The incremental-update path keeps its own existing `<new-activity>` wrapper unchanged —
        // `<conversation>` is only added to the from-scratch/split-turn path.
        let prefix = vec![
            Message::user(format!("{SUMMARY_MARKER}\n\nPrevious summary body")),
            Message::assistant(vec![ContentBlock::text("new work since")]),
        ];
        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &prefix,
            CompactionReason::Manual,
        );
        let req = summary_request("claude-test", &prefix, 512, &file_ops, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(!text.contains("<conversation>"));
        assert!(text.contains("<new-activity>"));
    }

    #[test]
    fn render_prefix_includes_a_visible_thinking_blocks_text() {
        let messages = vec![
            Message::user("why did the build fail?"),
            Message::assistant(vec![
                ContentBlock::Thinking {
                    text: "the error mentions a missing semicolon on line 12".into(),
                    signature: "sig".into(),
                },
                ContentBlock::text("there's a missing semicolon"),
            ]),
        ];
        let rendered = render_prefix(&messages);
        assert!(
            rendered.contains("[thinking] the error mentions a missing semicolon on line 12"),
            "got: {rendered}"
        );
        assert!(rendered.contains("there's a missing semicolon"));
    }

    #[test]
    fn render_prefix_truncates_a_long_thinking_block() {
        let long = "x".repeat(THINKING_MAX_CHARS + 500);
        let messages = vec![Message::assistant(vec![ContentBlock::Thinking {
            text: long.clone(),
            signature: "sig".into(),
        }])];
        let rendered = render_prefix(&messages);
        assert!(rendered.contains("characters truncated"));
        assert!(
            !rendered.contains(&long),
            "the full untruncated text must not appear"
        );
    }

    #[test]
    fn render_prefix_omits_redacted_thinking_entirely() {
        // Opaque ciphertext the provider chose not to expose — never human-readable, so it must never
        // be spent on, unlike a visible `Thinking` block.
        let messages = vec![Message::assistant(vec![ContentBlock::RedactedThinking {
            data: "opaque-ciphertext-blob".into(),
        }])];
        let rendered = render_prefix(&messages);
        assert!(!rendered.contains("opaque-ciphertext-blob"));
        assert_eq!(
            rendered, "",
            "a redacted-thinking-only message renders as nothing"
        );
    }

    #[test]
    fn render_prefix_without_tool_results_drops_tool_result_content_but_keeps_tool_call_names() {
        // pi-parity gap (fixed): pi's `getMessageFromEntry` (branch-summarization.ts) returns
        // `undefined` for any tool-result-role entry, so a branch summary's rendered transcript never
        // includes tool output at all — only user/assistant text, thinking, and tool-call *names*.
        let messages = vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "src/x.rs" }),
            )]),
            Message::tool_result("1", "SECRET FILE CONTENTS should not appear", false),
            Message::assistant(vec![ContentBlock::text("done")]),
        ];
        let rendered = render_prefix_without_tool_results(&messages);
        assert!(rendered.contains("try approach X"));
        assert!(rendered.contains("Assistant called tool `read`"));
        assert!(rendered.contains("done"));
        assert!(
            !rendered.contains("SECRET FILE CONTENTS"),
            "tool result content must not survive: {rendered}"
        );
        assert!(!rendered.contains("[tool result]"));

        // The ordinary renderer, by contrast, still includes it — this is a branch-summarization-only
        // exclusion, not a change to compaction's own summarization path.
        assert!(render_prefix(&messages).contains("SECRET FILE CONTENTS"));
    }

    #[test]
    fn second_compaction_updates_the_previous_summary_incrementally() {
        // Simulate a prefix that begins with a prior summary (as `apply_summary` would produce),
        // followed by new activity. The request must update the previous summary, not re-summarize it.
        let prefix = vec![
            Message::user(format!(
                "{SUMMARY_MARKER}\n\nPrevious summary body about task X"
            )),
            Message::assistant(vec![ContentBlock::text("new work since")]),
            Message::tool_result("9", "new tool output", false),
        ];
        // The prior summary is detected and handed back verbatim.
        assert_eq!(
            previous_summary(&prefix),
            Some("Previous summary body about task X")
        );

        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &prefix,
            CompactionReason::Manual,
        );
        let req = summary_request("claude-test", &prefix, 512, &file_ops, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        // Uses the incremental update framing, not the from-scratch instruction.
        assert!(text.contains("<previous-summary>"));
        assert!(text.contains("Previous summary body about task X"));
        assert!(text.contains("<new-activity>"));
        assert!(text.contains("NEW conversation messages to incorporate"));
        // The marker line itself isn't fed back into the transcript as raw text.
        assert!(!text.contains(SUMMARY_MARKER));
    }

    #[test]
    fn file_ops_fold_forward_across_successive_compactions() {
        // Round 1: the model read/edited round1.rs. Round 1's provenance records that.
        let round1_prefix = vec![
            Message::user("task"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "round1.rs" }),
            )]),
            Message::tool_result("1", "contents", false),
        ];
        let round1_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &round1_prefix,
            CompactionReason::Threshold,
        );
        assert_eq!(round1_ops.read_files, vec!["round1.rs"]);
        assert_eq!(round1_ops.compactions, 1);

        // Round 2's prefix — as `apply_summary` would leave it — starts with round 1's summary
        // message (prose, no `ToolUse` blocks) followed by *new* activity referencing a *different*
        // file, round2.rs. If file-ops didn't fold forward, this round's provenance would only ever
        // contain round2.rs — round1.rs would have silently vanished the moment its raw `ToolUse`
        // message was replaced by the round-1 summary.
        let round2_prefix = vec![
            Message::user(format!("{SUMMARY_MARKER}\n\nRound 1 summary")),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "round2.rs" }),
            )]),
            Message::tool_result("2", "edited", false),
        ];
        let round2_ops = merge_file_ops(&round1_ops, &round2_prefix, CompactionReason::Threshold);

        // Both rounds' files survive, in first-seen order — round1.rs isn't lost, and round2.rs isn't
        // just tacked on without it.
        assert_eq!(round2_ops.read_files, vec!["round1.rs"]);
        assert_eq!(round2_ops.modified_files, vec!["round2.rs"]);
        assert_eq!(round2_ops.compactions, 2);

        // The summarization prompt itself carries the *merged* tags, not just this round's activity —
        // the model doing the summarizing sees the full file history, not a truncated one.
        let req = summary_request("claude-test", &round2_prefix, 512, &round2_ops, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("round1.rs"),
            "round 1's file must survive: {text}"
        );
        assert!(text.contains("round2.rs"));
    }

    #[test]
    fn merge_file_ops_dedupes_a_path_seen_again_in_a_later_round() {
        let previous = CompactionProvenance {
            read_files: vec!["a.rs".into()],
            modified_files: vec![],
            compactions: 1,
            last_reason: Some(CompactionReason::Threshold),
        };
        let messages = vec![Message::assistant(vec![ContentBlock::tool_use(
            "1",
            "read",
            json!({ "path": "a.rs" }), // the same file, read again
        )])];
        let merged = merge_file_ops(&previous, &messages, CompactionReason::Manual);
        assert_eq!(merged.read_files, vec!["a.rs"]); // not duplicated
        assert_eq!(merged.last_reason, Some(CompactionReason::Manual));
    }

    #[test]
    fn previous_summary_is_none_for_a_normal_prefix() {
        assert!(previous_summary(&convo()).is_none());
    }

    #[test]
    fn is_split_turn_detects_a_prefix_ending_mid_tool_dispatch() {
        // `convo()` ends with `assistant(Text "done")` — a genuine turn conclusion, not mid-dispatch.
        assert!(!is_split_turn(&convo()));

        // Cut the same conversation off one message earlier — right after a tool_result, before the
        // assistant's concluding response. That's exactly the split-turn shape: the trailing message
        // is a tool-results-only user message with the turn's actual conclusion (and the original
        // request that motivated it) elsewhere.
        let messages = convo();
        assert!(is_split_turn(&messages[..messages.len() - 1]));

        // An empty prefix, or one ending on a genuine user text turn, isn't split.
        assert!(!is_split_turn(&[]));
        assert!(!is_split_turn(&[Message::user("hello")]));
    }

    #[test]
    fn is_split_turn_still_detects_a_tool_result_message_with_a_steered_text_block_mixed_in() {
        // LOW pi-parity gap (fixed): a mid-run steer message rides on the same tool-results turn
        // (`Agent::run_events_steered`), so this message is *not* purely `ToolResult` blocks — but
        // it's still unambiguously a tool-dispatch continuation, since a genuine fresh user turn can
        // never carry a `ToolResult` block at all. The old `all(...)` check missed this shape entirely.
        let steered_with_tool_result = Message::tool_results(vec![
            ContentBlock::ToolResult {
                tool_use_id: "1".into(),
                content: "ok".into(),
                is_error: false,
                images: Vec::new(),
            },
            ContentBlock::text("keep going, also check the other file"),
        ]);
        assert!(is_split_turn(&[steered_with_tool_result]));
    }

    #[test]
    fn summary_request_uses_the_split_turn_instruction_when_the_cut_is_mid_turn() {
        let messages = convo();
        let prefix = &messages[..messages.len() - 1]; // ends on a bare tool_result — split-turn shape
        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            prefix,
            CompactionReason::Threshold,
        );
        let req = summary_request("claude-test", prefix, 512, &file_ops, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("## Original Request"));
        assert!(text.contains("## Early Progress"));
        assert!(text.contains("## Context for Suffix"));
        assert!(!text.contains("## Goal")); // the generic template's heading must not also appear
    }

    #[test]
    fn summary_request_uses_the_generic_instruction_when_the_cut_is_a_clean_boundary() {
        let messages = convo(); // ends on a genuine assistant conclusion — not split
        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &messages,
            CompactionReason::Threshold,
        );
        let req = summary_request("claude-test", &messages, 512, &file_ops, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(text.contains("## Goal"));
        assert!(!text.contains("## Original Request"));
    }

    #[test]
    fn summary_request_appends_custom_instructions_as_additional_focus() {
        // Matches pi's own `generateSummary`: "Additional focus: {customInstructions}" appended after
        // the structured instruction template, not replacing it — a manual compaction's client-supplied
        // steering, not a wholesale prompt override.
        let messages = convo();
        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &messages,
            CompactionReason::Manual,
        );
        let req = summary_request(
            "claude-test",
            &messages,
            512,
            &file_ops,
            Some("keep every detail about the auth refactor"),
        );
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("## Goal"),
            "the structured template survives: {text}"
        );
        assert!(
            text.contains("Additional focus: keep every detail about the auth refactor"),
            "the custom instructions must be appended: {text}"
        );
    }

    #[test]
    fn summary_request_omits_additional_focus_when_no_custom_instructions_given() {
        let messages = convo();
        let file_ops = merge_file_ops(
            &CompactionProvenance::default(),
            &messages,
            CompactionReason::Threshold,
        );
        let req = summary_request("claude-test", &messages, 512, &file_ops, None);
        let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(!text.contains("Additional focus"));
    }

    #[test]
    fn apply_summary_replaces_prefix_and_resets_size() {
        let mut s = Session::new();
        s.messages = Arc::new(convo());
        s.last_input_tokens = 9999;
        let n_before = s.messages.len();
        apply_summary(&mut s, 3, "SUMMARY", 9999);

        // summary(user) + kept suffix [3..]
        assert_eq!(s.messages.len(), 1 + (n_before - 3));
        assert_eq!(s.messages[0].role, Role::User);
        assert!(matches!(
            &s.messages[0].content[0],
            ContentBlock::Text { text, .. } if text.contains("SUMMARY")
        ));
        // Alternation: summary(user) then the kept assistant message.
        assert_eq!(s.messages[1].role, Role::Assistant);
        assert_eq!(s.last_input_tokens, 0);
    }
}
