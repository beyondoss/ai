//! Context compaction — keeping a long run under the model's context window.
//!
//! A loop re-sends an ever-growing transcript every turn. Left unbounded it eventually exceeds the
//! model's context window and every further request fails — the session effectively dies. Compaction
//! replaces the older prefix of the conversation with an LLM-generated summary while keeping the most
//! recent turns verbatim, so the run can continue indefinitely.
//!
//! This module is the pure, network-free half: the trigger ([`should_compact`]), the cut-point
//! search ([`find_cut`]), the summary-prompt construction ([`render_prefix`]/[`summary_request`]),
//! and provenance extraction. [`Agent`](crate::Agent) owns the one piece that needs the network —
//! making the summarization model call — and stitches the result back into the [`Session`].
//!
//! **The deterministic-carry channel.** A summarizing model cannot be trusted to preserve specific
//! facts in free prose, so anything that *must* survive a cut rides [`CompactionProvenance`] instead:
//! it is folded forward every round by [`merge_provenance`], persisted with the session, restored on
//! reopen, and re-appended to each new summary verbatim by host code in
//! [`Agent::compact`](crate::agent::Agent::compact). Two things ride it today — the read/modified file
//! lists ([`format_file_operations`]) and the model's `todo` list ([`format_todo_list`]) — and both are
//! there for the same reason: they are the run's working state, and losing them silently degrades the
//! model rather than failing loudly. Anything else added here should clear the same bar.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

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

/// Which summarization call produced a compaction's summary — carried on
/// [`crate::agent::AgentEvent::Compacted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummarySource {
    /// The cached call: the live conversation plus a closing instruction.
    Cached,
    /// The re-render ([`summary_request`]), with no cached call before it: the cached summary is
    /// off, the reason is an overflow, or the conversation didn't fit.
    Rerender,
    /// The re-render, after the cached call ran and its reply was turned down: a tool call, a blank
    /// reply, an error, a reply that is not a checkpoint ([`checkpoint_body`]), or one cut off at the
    /// output limit or looping ([`is_runaway`]).
    CachedRejected,
}

/// What compaction has recorded about this session so far, folded forward across every round.
/// [`Session::compaction`] carries one of these; each successful [`crate::agent::Agent::compact`] call
/// replaces it with the merged result of [`merge_provenance`].
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
    /// The most recent `todo` list any compacted prefix carried — the raw `todos` array the model last
    /// sent, verbatim. `None` when the session has never compacted away a `todo` call.
    ///
    /// Unlike the file lists, this is *last-wins*, not accumulate-and-dedupe: the tool's contract is a
    /// full replace, so folding two rounds' lists together would resurrect steps the model deliberately
    /// dropped. An explicitly cleared list (`[]`) is therefore meaningful and must not fall back to an
    /// older one — see [`merge_provenance`].
    #[serde(default)]
    pub todos: Option<Value>,
    /// Logical paths the model has authored via the `memory` tool (a `create`/`str_replace`/`insert`),
    /// deduped, oldest-first, across every round so far. Carried for the same reason as the file lists:
    /// once `apply_summary` drops the prefix, the fact that the model *wrote itself a note* — and where —
    /// is gone with it, so a model reminded to consult its working memory post-compaction would have no
    /// idea what it saved or that it saved anything. Rendered into the summary by [`format_memory_notes`],
    /// it becomes a guaranteed, zero-effort "here is what you recorded, recall the relevant ones" pointer.
    #[serde(default)]
    pub memory_notes: Vec<String>,
    /// The user's own requests from compacted prefixes, verbatim — each genuine user turn's text,
    /// in order, deduped, each capped at [`USER_REQUEST_MAX_CHARS`]; past [`USER_REQUESTS_KEPT`] the
    /// first and the latest are kept. Folded forward by [`merge_user_requests`], rendered by
    /// [`format_user_requests`].
    ///
    /// Why: without it, 23 of 52 later-round eval summaries re-inferred the request ("The user's exact
    /// request is not visible in this prefix. From the work, the task is to…"), and only 34–40% kept
    /// the "don't edit tests" constraint. Carried, the user's words sit outside anything a summarizer
    /// can paraphrase.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub user_requests: Vec<String>,
}

/// A carried user request longer than this many characters is truncated (with a marker).
pub const USER_REQUEST_MAX_CHARS: usize = 4_000;

/// At most this many user requests are carried: the first (the task as originally set) and the
/// latest ones.
pub const USER_REQUESTS_KEPT: usize = 4;

/// `previous`'s carried requests plus every genuine user turn in `messages` (text only — the words
/// the user typed or a steer injected, not tool results or a prior summary), deduped, capped per
/// request at [`USER_REQUEST_MAX_CHARS`] and overall at [`USER_REQUESTS_KEPT`]: the first plus the
/// latest. `messages` is the round's prefix; the leading summary message is skipped by
/// `is_turn_start`.
pub fn merge_user_requests(previous: &[String], messages: &[Message]) -> Vec<String> {
    let mut out = previous.to_vec();
    for m in messages.iter().filter(|m| is_turn_start(m)) {
        let text: String = m
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(&**text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let text = truncate_chars(text.trim(), USER_REQUEST_MAX_CHARS);
        if !text.is_empty() && !out.contains(&text) {
            out.push(text);
        }
    }
    if out.len() > USER_REQUESTS_KEPT {
        out.drain(1..out.len() + 1 - USER_REQUESTS_KEPT);
    }
    out
}

/// Render the carried user requests as a `<user-requests>` block appended to a summary — the same
/// deterministic-carry contract as [`format_file_operations`]. `""` when there are none.
pub fn format_user_requests(requests: &[String]) -> String {
    if requests.is_empty() {
        return String::new();
    }
    format!(
        "\n\n<user-requests>\nThe user's own words, verbatim (the first request, then the latest):\n{}\n</user-requests>",
        requests.join("\n---\n")
    )
}

/// Fold `previous`'s provenance forward with whatever the current round's prefix (`messages`) adds, and
/// record `reason` as the new `last_reason`.
///
/// File paths accumulate (deduped by path, first-seen order); the `todo` list is replaced whenever this
/// round's prefix contains one at all, and otherwise carried forward untouched; the user's requests
/// accumulate by [`merge_user_requests`]. `messages` is the
/// *whole* current-round prefix, including a leading prior-summary message if there is one — that
/// message carries no `ToolUse` blocks (it's prose), so scanning it contributes nothing and this
/// naturally reduces to "previous provenance + this round's new activity" without needing to slice it
/// off.
pub fn merge_provenance(
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
    let mut memory_notes = previous.memory_notes.clone();
    for path in extract_memory_notes(messages) {
        if !memory_notes.contains(&path) {
            memory_notes.push(path);
        }
    }
    CompactionProvenance {
        read_files: read,
        modified_files: modified,
        compactions: previous.compactions.saturating_add(1),
        last_reason: Some(reason),
        // `or_else`, not `or`: a round whose prefix cleared the list (`todos: []`) yields `Some([])` and
        // must win over `previous`, or a cleared plan would silently reappear at the next compaction.
        todos: extract_todos(messages).or_else(|| previous.todos.clone()),
        memory_notes,
        user_requests: merge_user_requests(&previous.user_requests, messages),
    }
}

/// Tunables for automatic compaction.
#[derive(Debug, Clone, Copy)]
pub struct CompactionConfig {
    /// The model's context window, in tokens: prompt plus output. The trigger compares the live
    /// prompt size against [`Self::input_budget`], which this bounds.
    pub context_window: u32,
    /// The most input tokens one request may carry, when the model caps input below the window
    /// ([`crate::models::ModelCaps::input_limit`]: OpenAI's GPT-5 family, 922K of a 1.05M window).
    /// `None`: the window alone bounds input.
    pub input_limit: Option<u32>,
    /// Fire no later than this many prompt tokens, even when the window would allow more
    /// ([`crate::models::ModelCaps::compact_at`]: a quality or price line, not a hard limit). `None`:
    /// the input budget alone sets the trigger. See [`Self::trigger_tokens`].
    pub compact_at: Option<u32>,
    /// Headroom left below the input budget: compaction fires once the prompt crosses
    /// `input_budget() - reserve_tokens`, leaving room for the next turn's growth + slack.
    pub reserve_tokens: u32,
    /// Roughly how many tokens of recent conversation to keep verbatim after a compaction.
    pub keep_recent_tokens: u32,
    /// Output ceiling for the summarization call itself.
    pub summary_max_tokens: u32,
    /// Whether automatic (threshold-triggered) compaction is on. Manual/overflow compaction ignores it.
    pub enabled: bool,
    /// Whether an automatic compaction first tries [`mask_tool_results`]: elide old tool output and
    /// summarize only if that doesn't free enough. See [`MaskPlan`].
    pub mask_tool_output: bool,
    /// Tool results in the last this-many turns (assistant messages) are never masked...
    pub mask_keep_turns: usize,
    /// ...unless those turns hold more than this many tokens: then only the most recent this-many
    /// tokens are kept. `None`: a quarter of the trigger ([`Self::mask_keep_tokens`]), so the window
    /// scales with the model — on a 90K trigger the last ten turns can be most of the context, and a
    /// fixed turn count left nothing to mask.
    pub mask_keep_tokens: Option<u32>,
    /// A masking pass is applied only when it frees at least this many tokens, or a tenth of the
    /// trigger when that is less ([`Self::mask_min_free`]), so the bar scales with the model. Every pass rewrites
    /// history and so invalidates the prompt cache from its first edit on; a pass must buy back enough
    /// context to pay for that. Our `clear_at_least`.
    pub mask_min_free_tokens: u32,
    /// Whether the summarization call is the live conversation plus a closing instruction (its
    /// prefix read from the prompt cache; see `Agent::conversation_summary_request`), falling back to
    /// the [`summary_request`] re-render when that doesn't fit, errors, or its reply is not a
    /// checkpoint ([`checkpoint_body`]), is cut off at the output limit, or loops ([`is_runaway`]).
    /// `false`: always re-render.
    pub cached_summary: bool,
    /// Changes under evaluation, all off by default. See [`CompactionCandidates`].
    pub candidates: CompactionCandidates,
}

/// Compaction changes under A/B evaluation, each off by default so the shipped behavior is what
/// [`Default`] gives. One value is one arm; the CLI sets them by name (`--compaction-candidates`).
/// Each field names the eval finding it answers, with the replay evidence behind it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionCandidates {
    /// Every summary is one current-state document: the [`STATE_FORMAT`] template on both paths, with
    /// [`STATE_RULES`] (state as of the newest messages, in-flight step included; one document, never
    /// appended to; no pasted code; about 1,500 words; the user credited only with the user's words),
    /// a previous summary rewritten rather than reused, and no split-turn decomposition.
    ///
    /// Why: a single-prompt agentic run is one long "turn", and `find_cut` always leaves the prefix
    /// ending on a tool result, so 55 of 87 eval summaries took the split-turn path (51 opening "No
    /// prior history.") and the structured template and [`UPDATE_INSTRUCTION`] almost never ran. Each
    /// later round then reused the previous summary verbatim (`turn_start == 1`) and appended another
    /// split-turn section: 30 summaries stack several request sections, re-split sizes grow
    /// 6.4K → 13K → 22.9K → 26.7K chars by round, and the largest is 49K chars with 4 sections (Sol,
    /// 98 code fences across its 16). The judge rated 93% of split-template summaries stale against the
    /// newest messages, 41% of structured ones.
    pub state_template: bool,
    /// The re-render also shows the summarizer the kept suffix, read-only, in `<recent-context>`, so
    /// its summary is current as of the newest messages rather than as of the cut. The cached call
    /// already sees it (it sends the whole conversation): judged stale 40% of the time against 93%
    /// for the re-render's split template, which sees only the prefix.
    pub sees_tail: bool,
    /// A masking pass is applied only when it pays for its cache rewrite before the freed room
    /// refills: `0.1 · freed · (freed / growth) ≥ (w − 0.1) · rewrite` ([`mask_net_saving`]), with
    /// `rewrite` the tokens from the first edited message on, `growth` the transcript's recent
    /// tokens per turn ([`growth_per_turn`]) and `w` the cache-write price (1.25×, 2× at the 1-hour
    /// TTL). In the eval, 7 of 12 passes earned back their rewrite; this rule agrees with the
    /// outcome on 9 of the 12.
    pub mask_breakeven: bool,
    /// Like [`Self::mask_breakeven`], but the pass may start later: of the plan's edits, it keeps those
    /// from the start that maximizes the same net saving ([`best_mask_start`]), so an early result that
    /// frees little stops forcing a rewrite of everything after it.
    pub mask_best_start: bool,
    /// The re-render keeps a long tool result's head *and* tail ([`head_tail_chars`]): its first
    /// [`HEAD_CHARS`] and last [`TAIL_CHARS`] characters, plus every line from the elided middle that
    /// is a test tally or an error ([`is_signal_line`]), instead of only its first
    /// [`TOOL_RESULT_MAX_CHARS`].
    ///
    /// Why: test runners print their tally last. In the eval, 66% of the long test results had their
    /// pass/fail tally only after character 2,000, so the summarizer never saw it.
    pub head_tail_results: bool,
}

impl CompactionCandidates {
    /// Every candidate's CLI name, in the order `--help` lists them.
    pub const NAMES: &[&str] = &[
        "state",
        "tail",
        "mask-breakeven",
        "mask-best-start",
        "head-tail",
    ];

    /// The candidates `names` turns on (the CLI's `--compaction-candidates`); an unknown name is an
    /// error naming the valid ones, so a typo in an eval arm fails the run instead of silently
    /// measuring the default.
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> Result<Self, String> {
        let mut c = Self::default();
        for name in names {
            match name.as_ref().trim() {
                "state" => c.state_template = true,
                "tail" => c.sees_tail = true,
                "mask-breakeven" => c.mask_breakeven = true,
                "mask-best-start" => c.mask_best_start = true,
                "head-tail" => c.head_tail_results = true,
                "" => {}
                other => {
                    return Err(format!(
                        "unknown compaction candidate `{other}` (known: {})",
                        Self::NAMES.join(", ")
                    ));
                }
            }
        }
        Ok(c)
    }
}

impl CompactionConfig {
    /// Defaults scaled to a model's real context window.
    ///
    /// [`Default`] is sized for a 200k-context Claude model and states both budgets as *absolutes*
    /// (`reserve_tokens: 16_384`, `keep_recent_tokens: 20_000`). Seeding only `context_window` from the
    /// model's capabilities and leaving those two at their 200k values — which is what [`Agent::new`]
    /// used to do — silently breaks the one invariant the two budgets have to satisfy between them:
    ///
    /// > what a compaction *keeps* must be comfortably smaller than the threshold that *triggers* one.
    ///
    /// The trigger fires at `context_window - reserve_tokens`, so on a 32_768-token model (the smallest
    /// in the catalogue) the un-scaled numbers give a trigger budget of 16_384 while `keep_recent_tokens`
    /// retains 20_000 — a compaction that keeps *more* than the line it just crossed. Every turn then
    /// re-triggers, each one spending a real summarization call, and the context never gets back under
    /// the threshold. Scaling both budgets down with the window restores the invariant.
    ///
    /// The caps are deliberately one-sided (`min`, not a proportion): any window at or above ~64k gets
    /// byte-for-byte the same tuning as before, since a quarter of the window already exceeds the
    /// absolute reserve and half the remaining budget already exceeds `keep_recent_tokens`. Only the
    /// genuinely small windows — the ones that were broken — see different numbers.
    pub fn for_window(context_window: u32) -> Self {
        Self::for_limits(context_window, None, None)
    }

    /// [`Self::for_window`] for a model's capabilities: the window plus its input cap, so the trigger
    /// lands below the point the vendor rejects input (OpenAI's GPT-5 family caps input at the window
    /// less the max output — trigger at `window - reserve` there never fires before the 400 does).
    pub fn for_model(caps: &crate::models::ModelCaps) -> Self {
        Self::for_limits(caps.context_window, caps.input_limit, caps.compact_at)
    }

    fn for_limits(context_window: u32, input_limit: Option<u32>, compact_at: Option<u32>) -> Self {
        let d = Self::default();
        let input_budget = input_limit.map_or(context_window, |l| l.min(context_window));
        // Never hand the reserve more than a quarter of the budget: at the absolute 16_384 a 32k model
        // would surrender half its context to headroom before the first token is sent.
        let reserve_tokens = d.reserve_tokens.min(input_budget / 4);
        let cfg = Self {
            context_window,
            input_limit,
            compact_at,
            reserve_tokens,
            ..d
        };
        // Half the trigger: a compaction lands the context at roughly 50% of the threshold, so there's
        // real room to work before the next one fires rather than re-triggering immediately.
        let keep_recent_tokens = d.keep_recent_tokens.min(cfg.trigger_tokens() / 2);
        Self {
            keep_recent_tokens,
            ..cfg
        }
    }

    /// The most prompt tokens a request can carry: `min(context_window, input_limit)`.
    pub fn input_budget(&self) -> u32 {
        self.input_limit
            .map_or(self.context_window, |l| l.min(self.context_window))
    }

    /// The recent-transcript budget masking never touches: `mask_keep_tokens`, else a quarter of the
    /// trigger.
    pub fn mask_keep_tokens(&self) -> u32 {
        self.mask_keep_tokens
            .unwrap_or_else(|| self.trigger_tokens() / 4)
    }

    /// The least a masking pass must free to be applied: `mask_min_free_tokens`, capped at a tenth of
    /// the trigger (90K → 9K, 200K → 20K, 500K → the 25K default).
    pub fn mask_min_free(&self) -> u32 {
        self.mask_min_free_tokens.min(self.trigger_tokens() / 10)
    }

    /// The prompt size past which [`should_compact`] fires:
    /// `min(input_budget() − reserve_tokens, compact_at)`.
    ///
    /// `compact_at` is the trigger itself, not one more limit to subtract the reserve from. The
    /// reserve is headroom for the next turn's growth below a limit the vendor *rejects* past; a
    /// `compact_at` sits well under that limit already (a quality or price line), so the headroom is
    /// there without subtracting it twice. It also keeps the number honest: `--compact-at 200000`
    /// compacts at 200K, where `min(window, input_limit, compact_at) − reserve` would compact at
    /// 183.6K and push Haiku 5.5's 90K line down to 73.6K.
    pub fn trigger_tokens(&self) -> u32 {
        let budget = self.input_budget().saturating_sub(self.reserve_tokens);
        self.compact_at.map_or(budget, |cap| cap.min(budget))
    }
}

impl Default for CompactionConfig {
    fn default() -> Self {
        // Defaults sized for a 200k-context Claude model; override per deployment. Matches pi's own
        // shipped tuning (`DEFAULT_COMPACTION_SETTINGS`, `compaction.ts:111-115`, and the coding-agent
        // product's `compaction.ts:122-125`) — `reserveTokens: 16384, keepRecentTokens: 20000` — rather
        // than the earlier 24_000/40_000, which triggered compaction earlier and retained 2x more raw
        // verbatim context per round than the reference, for no documented reason.
        Self {
            context_window: 200_000,
            input_limit: None,
            compact_at: None,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
            summary_max_tokens: 4_096,
            enabled: true,
            mask_tool_output: true,
            mask_keep_turns: 10,
            mask_keep_tokens: None,
            mask_min_free_tokens: 25_000,
            cached_summary: true,
            candidates: CompactionCandidates::default(),
        }
    }
}

/// System prompt for the summarization call. Only the trailing guardrail sentence is verbatim-shared
/// with pi's own `SUMMARIZATION_SYSTEM_PROMPT`
/// (`packages/coding-agent/src/core/compaction/utils.ts:168-170`, not `compaction.ts` — that file only
/// re-exports/consumes the constant): "Do NOT continue the conversation. Do NOT respond to any
/// questions in the conversation. ONLY output the structured summary." That guardrail matters because
/// the rendered transcript embeds real past user questions, and without an explicit instruction not
/// to, the model can slip into answering one of them instead of emitting the structured checkpoint
/// format this call actually needs. The opening framing above it is beyond's own wording, not a port of
/// pi's "You are a context summarization assistant..." opener — functionally equivalent (both frame the
/// call as compaction, not conversation), just not textually identical.
pub const SUMMARY_SYSTEM: &str = "You compact a long agent transcript so the agent can keep working \
with far fewer tokens but no loss of essential context. Be precise, concrete, and information-dense. \
Copy specifics verbatim — never paraphrase, round, or elide them: file paths (with line numbers), \
identifiers and function names, exact literal values (numbers, ports, versions, config keys, IDs, \
flags), commands and their key outputs, error messages, and decisions and their rationale. A specific \
you drop is gone for good — prefer keeping an exact value over smooth prose. Do NOT continue the \
conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

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
continue — copy exact literal values (numbers, ports, versions, config keys, IDs, flags), commands and \
their key outputs, and file:line references verbatim rather than describing them]\n- [Or \"(none)\" if \
not applicable]\n\nKeep each section concise, but never at the cost of a specific: copy exact file paths \
(with line numbers), function/identifier names, literal values, commands and their key outputs, and \
error messages verbatim rather than paraphrasing them.";

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
[Preserve important context, add new if needed — keep exact literal values, commands and their key \
outputs, and file:line references verbatim]\n\nKeep each section concise, but never at the cost of a \
specific: copy exact file paths (with line numbers), function/identifier names, literal values, commands \
and their key outputs, and error messages verbatim rather than paraphrasing them.";

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
understand the retained recent work — copy exact file paths (with line numbers), literal values, \
commands and their key outputs, and error messages verbatim]\n\nBe concise. Focus on what's needed to \
understand the kept suffix.";

/// [`CompactionCandidates::state_template`]'s opener for a summary with no previous one.
pub const STATE_INTRO: &str = "The messages above are a conversation to summarize. Write a \
checkpoint of it that the agent will continue the work from.\n\n";

/// [`CompactionCandidates::state_template`]'s opener when a previous summary exists (the re-render's
/// `<previous-summary>` + `<new-activity>` shape): rewrite, never append.
pub const STATE_UPDATE_INTRO: &str = "The messages above are NEW conversation messages; the \
checkpoint so far is in <previous-summary> tags. Rewrite both into ONE current checkpoint the agent \
will continue the work from: a replacement for the previous summary, not an addition to it. Merge, \
don't append: one Goal and one Progress section; move items from In Progress to Done as the new \
messages completed them, drop what they made stale (old test counts, fixed errors, finished next \
steps), and keep the user's Goal and Constraints word for word.\n\n";

/// The cached call's equivalent of [`STATE_UPDATE_INTRO`], for a conversation that opens with a
/// previous summary.
pub const STATE_FOLD_IN: &str = "\n\nThe conversation opens with an earlier checkpoint (the message \
starting \"[Earlier conversation compacted to save context]\"). Rewrite it and everything since into \
ONE current checkpoint, a replacement rather than an addition: one Goal and one Progress section; move \
items from In Progress to Done as they were completed, drop what newer messages made stale, and keep \
the user's Goal and Constraints word for word.";

/// The rules every [`CompactionCandidates::state_template`] summary is written under, one per eval
/// finding: summaries that described the state at the cut rather than the newest step (judged stale
/// in 74% of 87), stacked sections, pasted code (98 fences in 16 Sol summaries), and the agent's own
/// decisions credited to "the user".
pub const STATE_RULES: &str = "Rules:\n- Describe the state as of the NEWEST messages: what is done, \
the step in flight right now (what the last messages were doing and what they showed), and what comes \
next. The latest test results and errors replace older ones.\n- Write one document. Never paste file \
contents, diffs or function bodies (they can be re-read from disk): name a function by its signature, \
and quote a command, an error or a test count on one line. Use a code block only for a signature.\n- \
Credit the user only with what the user's own messages say; decisions the assistant made are the \
assistant's, under Key Decisions.\n- At most about 1,500 words. Spend them on specifics, copied \
verbatim: file paths with line numbers, identifiers, exact values, commands and their key outputs, \
error messages.\n\n";

/// [`CompactionCandidates::state_template`]'s format: [`SUMMARY_INSTRUCTION`]'s headings (so
/// [`checkpoint_body`] and a reader see the same shape), each asking for the current state.
pub const STATE_FORMAT: &str = "Use this EXACT format:\n\n## Goal\n[The user's request, in their words \
where possible, with every constraint they set.]\n\n## Constraints & Preferences\n- [Constraints and \
preferences the user stated]\n- [Or \"(none)\"]\n\n## Progress\n### Done\n- [x] [Completed and \
verified work]\n\n### In Progress\n- [ ] [The step in flight as of the newest messages: what it is \
doing and what it last showed]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key \
Decisions\n- **[Decision]** (user | assistant): [Brief rationale]\n\n## Next Steps\n1. [Ordered list of \
what should happen next]\n\n## Critical Context\n- [Data, examples and references needed to continue: \
exact values, commands and their key outputs, file:line references]\n- [Or \"(none)\"]";

/// Appended to a re-render's instruction when it carries the kept suffix
/// ([`CompactionCandidates::sees_tail`]).
pub const TAIL_NOTE: &str = "\n\nThe <recent-context> messages come after the ones to summarize and \
stay verbatim after your checkpoint. Do not summarize them; use them so the checkpoint is current: \
work they show finished goes under Done, and the step they show in progress is the In Progress item.";

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

/// Count `value`'s characters as its wire JSON would render, without materializing that JSON as a
/// `String` first — `serde_json::Value`'s `Display` impl already writes through `fmt::Write` rather
/// than building a buffer internally, so feeding it a counting-only sink (instead of collecting into a
/// `String` via `to_string()`) skips the allocation entirely for what `estimate_message_tokens` only
/// ever uses as a length, on a call already made once per tool-call block on every turn.
fn count_value_chars(value: &Value) -> usize {
    struct CharCount(usize);
    impl std::fmt::Write for CharCount {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0 += s.chars().count();
            Ok(())
        }
    }
    let mut counter = CharCount(0);
    // `Value`'s `Display` impl never fails to format, so a write error here would only ever indicate
    // the sink itself (which never returns `Err`) — safe to discard.
    let _ = std::fmt::Write::write_fmt(&mut counter, format_args!("{value}"));
    counter.0
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
                name.chars().count() + count_value_chars(input)
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
/// [`CompactionConfig::trigger_tokens`] still leaves the full reserve intact, so it isn't yet a reason
/// to fire — only actually exceeding it is.
pub fn should_compact(session: &Session, cfg: &CompactionConfig) -> bool {
    session.last_input_tokens > 0
        && session
            .last_input_tokens
            .saturating_add(trailing_tokens(session))
            > cfg.trigger_tokens()
}

/// A hard overflow signal, independent of [`should_compact`]'s (possibly-disabled) soft threshold: the
/// live prompt has already reached — or a turn's own shape strongly implies it's about to reach — the
/// model's raw input budget, not just the reserve-adjusted budget `should_compact` guards.
/// Disabling [`CompactionConfig::enabled`] is a preference about when to compact *proactively*, with
/// headroom to spare; it isn't license to keep sending requests that are already guaranteed to overflow
/// on the very next turn, so callers check this regardless of that flag.
///
/// Two ways this fires — both driven by the provider's own reported usage, not by parsing an error:
/// - The live prompt (last turn's usage + anything appended since) already meets or exceeds the
///   input budget ([`CompactionConfig::input_budget`]: the raw window, or a lower input cap) on its
///   own — a silent overflow no error was ever raised for (the request that pushed it there still
///   succeeded).
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
    live_prompt >= cfg.input_budget()
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
    // Debug-only, not a runtime check on untrusted input: the tool_use/tool_result-pairing safety
    // argument above depends on `messages` strictly alternating roles. Nothing enforces that at the
    // type level — `Session::push` is a bare append — so this catches a violation at the one call
    // site whose correctness actually depends on it, rather than gatekeeping `Session` itself (which
    // is also used as a generic bulk-append container by the storage/persistence layer's own tests,
    // unrelated to conversation shape, so a push-time check there is the wrong place for this).
    debug_assert!(
        messages.windows(2).all(|w| w[0].role != w[1].role),
        "find_cut: messages must strictly alternate roles — a violation here means the \
         tool_use/tool_result-pairing safety argument this function relies on no longer holds"
    );
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
    render_prefix_impl(messages, ToolResults::Head)
}

/// [`render_prefix`], keeping a long tool result's head and tail ([`head_tail_chars`]) when
/// `head_tail` is set ([`CompactionCandidates::head_tail_results`]).
fn render_prefix_with(messages: &[Message], head_tail: bool) -> String {
    let mode = if head_tail {
        ToolResults::HeadTail
    } else {
        ToolResults::Head
    };
    render_prefix_impl(messages, mode)
}

/// Same rendering as [`render_prefix`], but drops `ToolResult` content entirely instead of rendering a
/// truncated body — used only by [`crate::branch_summary`]. Matches pi's own `getMessageFromEntry`
/// (`branch-summarization.ts`), which returns `undefined` for any tool-result-role entry so
/// `prepareBranchEntries` never includes tool output in a branch summary's rendered transcript at all;
/// unlike a from-scratch compaction summary (which stays live conversation context), a branch summary
/// is read once on return, so tool output's terseness matters more than its (already-truncated) detail.
/// Tool-call *names* (via `ToolUse`) still survive, same as ordinary compaction.
pub(crate) fn render_prefix_without_tool_results(messages: &[Message]) -> String {
    render_prefix_impl(messages, ToolResults::Drop)
}

/// How [`render_prefix_impl`] renders a tool result's content.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolResults {
    /// Left out (a branch summary).
    Drop,
    /// The first [`TOOL_RESULT_MAX_CHARS`].
    Head,
    /// Head, tail and the signal lines between them ([`head_tail_chars`]).
    HeadTail,
}

fn render_prefix_impl(messages: &[Message], tool_results: ToolResults) -> String {
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
                    let body = match tool_results {
                        ToolResults::Drop => continue,
                        ToolResults::Head => truncate_chars(content, TOOL_RESULT_MAX_CHARS),
                        ToolResults::HeadTail => head_tail_chars(content),
                    };
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

/// How much of a long tool result's start [`head_tail_chars`] keeps.
pub const HEAD_CHARS: usize = 500;
/// How much of a long tool result's end [`head_tail_chars`] keeps.
pub const TAIL_CHARS: usize = 1_500;
/// At most this many signal lines are kept from the elided middle, each cut to
/// [`SIGNAL_LINE_MAX_CHARS`].
const SIGNAL_LINES_KEPT: usize = 30;
const SIGNAL_LINE_MAX_CHARS: usize = 300;

/// A tool result for the re-render under [`CompactionCandidates::head_tail_results`]: unchanged up to
/// [`TOOL_RESULT_MAX_CHARS`]; past it, the first [`HEAD_CHARS`] and the last [`TAIL_CHARS`]
/// characters, with every test-tally or error line of the elided middle ([`is_signal_line`], at most
/// [`SIGNAL_LINES_KEPT`]) kept between them. A test run prints its tally last, past where a
/// head-only cut stops.
pub fn head_tail_chars(s: &str) -> String {
    let n = s.chars().count();
    if n <= TOOL_RESULT_MAX_CHARS {
        return s.to_string();
    }
    let byte_at = |c: usize| s.char_indices().nth(c).map_or(s.len(), |(i, _)| i);
    let (head_end, tail_start) = (byte_at(HEAD_CHARS), byte_at(n - TAIL_CHARS));
    let (head, middle, tail) = (&s[..head_end], &s[head_end..tail_start], &s[tail_start..]);
    let elided = n - HEAD_CHARS - TAIL_CHARS;
    let signal: Vec<&str> = middle
        .lines()
        .map(str::trim_end)
        .filter(|l| is_signal_line(l))
        .collect();
    let mut out = String::with_capacity(head.len() + tail.len() + 256);
    out.push_str(head);
    if signal.is_empty() {
        out.push_str(&format!("… [{elided} characters elided] …"));
    } else {
        out.push_str(&format!(
            "\n… [{elided} characters elided; their test-tally and error lines:]\n"
        ));
        for line in signal.iter().take(SIGNAL_LINES_KEPT) {
            out.push_str(&truncate_chars(line, SIGNAL_LINE_MAX_CHARS));
            out.push('\n');
        }
        if signal.len() > SIGNAL_LINES_KEPT {
            out.push_str(&format!(
                "[+{} more matching lines]\n",
                signal.len() - SIGNAL_LINES_KEPT
            ));
        }
        out.push_str("…\n");
    }
    out.push_str(tail);
    out
}

/// Whether a line of tool output is worth keeping from an elided middle: a test tally
/// ([`is_tally_line`]) or an error (`FAILED`, `Error`, `ERROR`, `Traceback`).
pub(crate) fn is_signal_line(line: &str) -> bool {
    is_tally_line(line)
        || ["FAILED", "Error", "ERROR", "Traceback"]
            .iter()
            .any(|w| line.contains(w))
}

/// The outcome words a test runner prints after a count (`3 failed, 120 passed`). `errors` before
/// `error`, so the longer word wins.
const TALLY_WORDS: [&str; 8] = [
    "passed",
    "failed",
    "errors",
    "error",
    "skipped",
    "xfailed",
    "xpassed",
    "deselected",
];

/// Whether `line` is a test runner's tally: a count followed by an outcome word (pytest's
/// `3 failed, 120 passed in 2.1s`, cargo's `test result: ok. 12 passed; 0 failed`), unittest's
/// `Ran 45 tests` and `FAILED (failures=2)`.
pub(crate) fn is_tally_line(line: &str) -> bool {
    if line.contains("test result:") || line.contains("FAILED (") {
        return true;
    }
    if let Some(at) = line.find("Ran ") {
        let rest = &line[at + 4..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && rest[digits..].starts_with(" test") {
            return true;
        }
    }
    !tally_pairs(line).is_empty()
}

/// The `(count, word)` pairs of a tally line, in order: `3 failed, 120 passed` →
/// `[("3", "failed"), ("120", "passed")]`.
pub(crate) fn tally_pairs(line: &str) -> Vec<(&str, &'static str)> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || (i > 0 && bytes[i - 1].is_ascii_alphanumeric()) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let Some(rest) = line[i..].strip_prefix(' ') else {
            continue;
        };
        let word = TALLY_WORDS.iter().find(|w| {
            rest.starts_with(**w)
                && !rest[w.len()..]
                    .bytes()
                    .next()
                    .is_some_and(|b| b.is_ascii_alphanumeric())
        });
        if let Some(word) = word {
            out.push((&line[start..i], *word));
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

/// Files the prefix read vs. modified, extracted from `read`/`write`/`edit` tool calls only. Appended
/// to the summary so the model keeps file awareness across the cut.
///
/// Task #31 (pi-parity fix): `ls` used to be bucketed alongside `read` here, contrary to both this
/// function's own doc comment and pi's `compaction/utils.ts:44-55` (`read`/`write`/`edit` only) — every
/// directory listing was diluting the read-files list with paths the prefix never actually read the
/// *contents* of, just enumerated.
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
                    "read" => &mut read,
                    "write" | "edit" | "multi_edit" => &mut modified,
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

/// The `todos` array from the **last** `todo` tool call in `messages`, or `None` if there wasn't one.
///
/// The `todo` tool's contract is a full replace — every call carries the complete list — so the last
/// call is the whole truth and earlier ones are dead. Scanning backwards for it is what lets a compacted
/// prefix hand its plan forward without the tool having to hold any state of its own (which it couldn't:
/// its registry is rebuilt on every model switch).
pub fn extract_todos(messages: &[Message]) -> Option<Value> {
    messages
        .iter()
        .rev()
        .flat_map(|m| m.content.iter().rev())
        .find_map(|b| match b {
            ContentBlock::ToolUse { name, input, .. } if name == "todo" => {
                input.get("todos").filter(|t| t.is_array()).cloned()
            }
            _ => None,
        })
}

/// The logical paths the model wrote to via the `memory` tool in `messages`, in first-seen order — a
/// `create`, `str_replace`, or `insert` (the write commands; `view`/`search` read, `delete`/`rename`
/// don't add content). Deduped within this scan; accumulated across rounds by [`merge_provenance`].
///
/// Keys on the tool name `"memory"` only, exactly as [`extract_todos`] keys on `"todo"` — it stays
/// agnostic to *which* root (`/memories` vs `/session`) a note lives under, so the lower layer never has
/// to know the host's memory-mount conventions. Both roots are worth surfacing: a note is a note.
pub fn extract_memory_notes(messages: &[Message]) -> Vec<String> {
    let mut notes = Vec::new();
    for m in messages {
        for b in &m.content {
            if let ContentBlock::ToolUse { name, input, .. } = b
                && name == "memory"
                && matches!(
                    input.get("command").and_then(Value::as_str),
                    Some("create" | "str_replace" | "insert")
                )
                && let Some(path) = input.get("path").and_then(Value::as_str)
            {
                let path = path.to_string();
                if !notes.contains(&path) {
                    notes.push(path);
                }
            }
        }
    }
    notes
}

/// Render the carried `memory` note paths as a `<memory-notes>` block appended to the summarization
/// model's *output* — the same deterministic-carry contract [`format_todo_list`] documents. Returns `""`
/// when there are none, so appending it to a summary is always safe. The model reads the block as: these
/// are files I wrote myself; recall the relevant ones with the `memory` tool's `view`.
pub fn format_memory_notes(memory_notes: &[String]) -> String {
    if memory_notes.is_empty() {
        String::new()
    } else {
        format!(
            "\n\n<memory-notes>\n{}\n</memory-notes>",
            memory_notes.join("\n")
        )
    }
}

/// Render a `todos` array as the compact checklist both the `todo` tool's own result and a compaction
/// summary show the model. Shared so the two can never drift.
///
/// Tolerant by construction: this also runs over a `todos` value restored from a *persisted* session,
/// which an older build (or a hand-edited file) may have written in a shape this build no longer emits.
/// A malformed item degrades to a plain `[ ]` row rather than being dropped or panicking.
pub fn render_todo_list(todos: &[Value]) -> String {
    let mut out = String::new();
    for item in todos {
        let glyph = match item.get("status").and_then(Value::as_str) {
            Some("completed") => "[x]",
            Some("in_progress") => "[>]",
            _ => "[ ]",
        };
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("(untitled)");
        out.push_str(glyph);
        out.push(' ');
        out.push_str(content);
        out.push('\n');
    }
    out.pop();
    out
}

/// Render the carried `todo` list as a `<todo_list>` block, meant to be appended to the summarization
/// model's *output* — the same deterministic-carry contract [`format_file_operations`] documents, and
/// for the same reason: a summarizing model asked to "merge the previous summary" will paraphrase or
/// drop a checklist, and a plan the model has quietly forgotten is worse than no plan.
///
/// Returns `""` for `None` and for an empty list (the model explicitly cleared its plan — there is
/// nothing to carry, and an empty block would only invite the model to invent one), so appending this
/// to a summary is always safe, unconditionally.
pub fn format_todo_list(todos: Option<&Value>) -> String {
    let items = todos.and_then(Value::as_array).filter(|a| !a.is_empty());
    match items {
        Some(items) => format!("\n\n<todo_list>\n{}\n</todo_list>", render_todo_list(items)),
        None => String::new(),
    }
}

/// If `prefix` begins with a prior compaction summary (a user message tagged with [`SUMMARY_MARKER`]),
/// return the summary's **prose body** — without the marker line, without `apply_summary`'s leading
/// token-count line, and without the deterministic-carry blocks the host appended to it.
///
/// This is what lets compaction be *incremental*: the previous summary is fed forward as-is and
/// updated, not re-summarized as transcript — so the token-count line (informational metadata for an
/// exported transcript, not part of the summary itself) is stripped before it would otherwise leak into
/// the next summarization prompt.
///
/// The carry blocks (`<read-files>`, `<modified-files>`, `<todo_list>`) are stripped for the same class
/// of reason, and their omission here is load-bearing in *two* places. In [`summary_request`]'s
/// incremental path the body is handed to the summarizing model under `<previous-summary>`: left in, the
/// model dutifully echoes a now-stale copy into its own prose, right before [`Agent::compact`]
/// (crate::agent::Agent::compact) appends the freshly-merged real ones — two conflicting file lists in
/// one summary. And in `compact`'s own `turn_start == 1` fast path the body is reused *verbatim* as the
/// next summary, which then gets the fresh blocks appended to it: without stripping, every split-turn
/// compaction round would append another copy of every block, unboundedly.
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
    Some(strip_carry_blocks(body))
}

/// The host-appended, deterministically-regenerated blocks a summary body carries at its tail, as
/// `(open, close)` tag pairs. See [`format_file_operations`] / [`format_todo_list`].
const CARRY_BLOCKS: [(&str, &str); 5] = [
    ("<user-requests>", "</user-requests>"),
    ("<read-files>", "</read-files>"),
    ("<modified-files>", "</modified-files>"),
    ("<todo_list>", "</todo_list>"),
    ("<memory-notes>", "</memory-notes>"),
];

/// A summarizing model's reply with any carry blocks it echoed at its end removed, so the host's
/// fresh copies are the only ones. The cached call sees the previous summary, blocks and all, as the
/// conversation's first message, and a model that writes its checkpoint in the same shape ends it the
/// same way: measured live on Claude Haiku 5.5, the blocks doubled every round (1, 2, 2, 4, 4 over five
/// rounds) before this. Only the tail is touched, as in [`previous_summary`].
pub fn strip_echoed_carry(reply: &str) -> &str {
    strip_carry_blocks(reply)
}

/// Strip every trailing [`CARRY_BLOCKS`] block from a summary body, leaving only the model's own prose.
///
/// Peels from the tail rather than searching the whole body: the blocks are only ever *appended*, in a
/// fixed order, so anything matching earlier in the text is the model's own prose quoting a tag name and
/// must be left alone.
fn strip_carry_blocks(body: &str) -> &str {
    let mut body = body;
    loop {
        // `trim_end` only removes trailing bytes, so `trimmed` is a prefix of `body` and their byte
        // indices agree — `rfind`'s offset is valid in either.
        let trimmed = body.trim_end();
        let Some((open, _)) = CARRY_BLOCKS
            .iter()
            .find(|(_, close)| trimmed.ends_with(*close))
        else {
            return body.trim_end();
        };
        let Some(start) = trimmed.rfind(open) else {
            return body.trim_end();
        };
        body = &body[..start];
    }
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
/// `file_ops` is the *already-merged* provenance (see [`merge_provenance`]) — the read/modified-file
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
    summary_request_with(
        model,
        prefix,
        &[],
        max_tokens,
        file_ops,
        custom_instructions,
        &CompactionCandidates::default(),
    )
}

/// [`summary_request`] under the given candidates: [`CompactionCandidates::state_template`] swaps the
/// template ([`STATE_INTRO`] or [`STATE_UPDATE_INTRO`], then [`STATE_RULES`] and [`STATE_FORMAT`],
/// never the split-turn one), and a non-empty `tail` (the kept suffix, under
/// [`CompactionCandidates::sees_tail`]) is rendered read-only in `<recent-context>` after the
/// transcript, with [`TAIL_NOTE`] saying what it is for.
pub fn summary_request_with(
    model: &str,
    prefix: &[Message],
    tail: &[Message],
    max_tokens: u32,
    file_ops: &CompactionProvenance,
    custom_instructions: Option<&str>,
    candidates: &CompactionCandidates,
) -> ModelRequest {
    let (read, modified) = (&file_ops.read_files, &file_ops.modified_files);
    let state = candidates.state_template;
    let head_tail = candidates.head_tail_results;

    let mut prompt = String::new();
    let instruction = match previous_summary(prefix) {
        Some(prev) => {
            // Incremental: previous summary verbatim, then only the new activity (prefix[1..]).
            prompt.push_str("<previous-summary>\n");
            prompt.push_str(prev);
            prompt.push_str("\n</previous-summary>\n\n<new-activity>\n");
            prompt.push_str(&render_prefix_with(&prefix[1..], head_tail));
            prompt.push_str("</new-activity>\n\n");
            if state {
                format!("{STATE_UPDATE_INTRO}{STATE_RULES}{STATE_FORMAT}")
            } else {
                UPDATE_INSTRUCTION.to_string()
            }
        }
        None => {
            // Matches pi's own `generateSummary`/`generateTurnPrefixSummary`: the rendered transcript
            // is wrapped in `<conversation>` tags, the same delimiting convention the incremental path
            // just below already applies to its own new-activity transcript via `<new-activity>`.
            prompt.push_str("<conversation>\n");
            prompt.push_str(&render_prefix_with(prefix, head_tail));
            prompt.push_str("\n</conversation>\n\n");
            if state {
                format!("{STATE_INTRO}{STATE_RULES}{STATE_FORMAT}")
            } else if is_split_turn(prefix) {
                SPLIT_TURN_INSTRUCTION.to_string()
            } else {
                SUMMARY_INSTRUCTION.to_string()
            }
        }
    };
    if !tail.is_empty() {
        prompt.push_str("<recent-context>\n");
        prompt.push_str(&render_prefix_with(tail, head_tail));
        prompt.push_str("</recent-context>\n\n");
    }
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
    // The carried requests: `<previous-summary>` has its carry blocks stripped, so without this the re-render never sees the user's own words again — which
    // the state template's Goal asks for. The cached call sees them in the summary message itself.
    if !file_ops.user_requests.is_empty() {
        prompt.push_str(&format!(
            "<user-requests>\n{}\n</user-requests>\n",
            file_ops.user_requests.join("\n---\n")
        ));
    }
    prompt.push_str(&instruction);
    if !tail.is_empty() {
        prompt.push_str(TAIL_NOTE);
    }
    if let Some(custom) = custom_instructions {
        prompt.push_str("\n\nAdditional focus: ");
        prompt.push_str(custom);
    }

    // A one-off prompt nothing will send again: a cache breakpoint on it only buys a write premium
    // (1.25×, or 2× at the 1-hour TTL) on every token of the re-render.
    ModelRequest::new(model, Arc::new(vec![Message::user(prompt)]), max_tokens)
        .with_system(SUMMARY_SYSTEM)
        .with_no_cache(true)
}

/// The opening of the cached summary instruction on the Anthropic wire ([`SummaryFraming::Handoff`]):
/// a checkpoint the model writes for itself from what the transcript shows, explicitly not its
/// reasoning.
///
/// Why this framing there, not [`SummaryFraming::Compact`]'s "stop working, reply with nothing but a
/// summary" plus [`SUMMARY_SYSTEM`]: Claude Haiku 5.5 refused that in 30 of 33 eval compactions ("I'm
/// not going to produce that checkpoint summary; the transcript stays with me"), and the refusal
/// became the summary. Replaying real compaction points: 24 of 31 refused under the old text, 0 of 30
/// under this one; Claude Sonnet 5.5, 0 of 5. Changing only "another LLM" to "you" left 15 of 20
/// refusing, so it is the framing, not one phrase. Byte for byte the text the replay measured.
pub const HANDOFF_FRAMING: &str = "Context checkpoint. Pause the task for this one reply and do not \
call any tools. To free context, the conversation above will be replaced by the checkpoint you write \
now; only the most recent turns stay verbatim after it, so spend the least on them. You continue this \
same task from the checkpoint in your next turn, so write it for yourself.\n\nWrite it from what the \
transcript shows: the user's messages, your tool calls and their results, the files you changed, test \
output. It records the state of the task, not your reasoning, so leave out your thinking. Be precise \
and information-dense, and copy specifics verbatim: file paths with line numbers, identifiers, exact \
values, commands and their key outputs, error messages. A specific you drop is gone for good.\n\n";

/// Where [`SUMMARY_INSTRUCTION`]'s format block starts, after its "summarize the messages above"
/// opener — the part [`HANDOFF_FRAMING`] keeps.
const SUMMARY_FORMAT_START: &str = "Use this EXACT format:";

/// How the cached summary instruction opens. Chosen per wire dialect by `Agent::compact`, because
/// the replays disagree by model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryFraming {
    /// [`HANDOFF_FRAMING`], then the format. The Anthropic wire: it stops Claude Haiku 5.5's
    /// refusals (24 of 31 → 0 of 30).
    Handoff,
    /// "Stop working on the task now…" plus [`SUMMARY_SYSTEM`], then the format. Every other wire.
    /// GPT-6 Luna, replayed as production sends it (the Responses dialect emits the instruction ahead
    /// of the trailing tool output), wrote a clean checkpoint 10 of 10 times under this text; under
    /// the handoff text it called a tool instead 4 of 15 times and looped `- ` to the output limit
    /// 2 of 15.
    Compact,
}

/// The closing user instruction of the cached summarization call — the one that sends the live
/// conversation itself rather than a re-render of it (`Agent::conversation_summary_request`), as a
/// `<system-reminder>` block beside the trailing tool results. Its request keeps the conversation's
/// own system prompt (that is what makes the prefix a cache hit), so [`SUMMARY_SYSTEM`] does not ride
/// as a system prompt: `framing` opens the instruction ([`SummaryFraming`]), then the structured
/// format ([`SUMMARY_INSTRUCTION`]'s, or [`STATE_RULES`] and [`STATE_FORMAT`] under
/// [`CompactionCandidates::state_template`]). The request carries the conversation's tools too, so it
/// also says not to call one. A conversation that opens with an earlier summary gets the incremental
/// rules; the read/modified-file tags and `custom_instructions` are added exactly as
/// [`summary_request`] adds them, and the host still appends the carried provenance to whatever comes
/// back.
pub fn conversation_summary_instruction(
    messages: &[Message],
    file_ops: &CompactionProvenance,
    custom_instructions: Option<&str>,
    candidates: &CompactionCandidates,
    framing: SummaryFraming,
) -> String {
    let mut out = String::from("<system-reminder>");
    let state = candidates.state_template;
    match framing {
        SummaryFraming::Handoff => {
            out.push_str(HANDOFF_FRAMING);
            if state {
                out.push_str(STATE_RULES);
                out.push_str(STATE_FORMAT);
            } else {
                let at = SUMMARY_INSTRUCTION.find(SUMMARY_FORMAT_START).unwrap_or(0);
                out.push_str(&SUMMARY_INSTRUCTION[at..]);
            }
        }
        SummaryFraming::Compact => {
            out.push_str(
                "Stop working on the task now. Do not call any tools. This conversation is about to \
                 be compacted: reply with nothing but a structured summary of it, which will replace \
                 everything above except the most recent turns (those stay verbatim, so spend the \
                 least on them).\n\n",
            );
            out.push_str(SUMMARY_SYSTEM);
            out.push_str("\n\n");
            if state {
                out.push_str(STATE_INTRO);
                out.push_str(STATE_RULES);
                out.push_str(STATE_FORMAT);
            } else {
                out.push_str(SUMMARY_INSTRUCTION);
            }
        }
    }
    if state && previous_summary(messages).is_some() {
        out.push_str(STATE_FOLD_IN);
    } else if previous_summary(messages).is_some() {
        out.push_str(
            "\n\nThe conversation opens with an earlier checkpoint summary (the message starting \
             \"[Earlier conversation compacted to save context]\"). Fold it in: PRESERVE all of its \
             information, ADD the progress, decisions and context since, move items from \"In \
             Progress\" to \"Done\" as they were completed, and keep every Key Decision.",
        );
    }
    let (read, modified) = (&file_ops.read_files, &file_ops.modified_files);
    if !read.is_empty() {
        out.push_str(&format!(
            "\n\n<read-files>\n{}\n</read-files>",
            read.join("\n")
        ));
    }
    if !modified.is_empty() {
        out.push_str(&format!(
            "\n\n<modified-files>\n{}\n</modified-files>",
            modified.join("\n")
        ));
    }
    if let Some(custom) = custom_instructions {
        out.push_str("\n\nAdditional focus: ");
        out.push_str(custom);
    }
    out.push_str("</system-reminder>");
    out
}

/// The longest run of identical consecutive lines (blank lines skipped, whitespace trimmed) at which a
/// cached summary reply counts as a runaway loop ([`is_runaway`]).
pub const RUNAWAY_REPEATED_LINES: usize = 10;

/// Whether a summary reply is a degenerate loop: [`RUNAWAY_REPEATED_LINES`] or more identical lines
/// in a row. Such a reply can still open with `## Goal`, so [`checkpoint_body`] alone accepts it.
///
/// Why: GPT-6 Luna, replaying real compaction points, twice wrote a few thousand chars of checkpoint
/// and then `- ` lines (14,267 and 15,152 in a row) until its output limit. No other reply in the
/// replays (220, across Haiku, Sonnet and Luna) and no summary the eval kept (55) repeated a line
/// even twice in a row. A loop that ends at the output limit is caught by that stop reason too; this
/// catches one that stops on its own.
pub fn is_runaway(reply: &str) -> bool {
    let mut run = 0;
    let mut prev = "";
    for line in reply.lines().map(str::trim).filter(|l| !l.is_empty()) {
        run = if line == prev { run + 1 } else { 1 };
        if run >= RUNAWAY_REPEATED_LINES {
            return true;
        }
        prev = line;
    }
    false
}

/// The checkpoint in a cached summary reply, which is accepted only if it has one: the text
/// from the format's `## Goal` heading on (any preamble before it dropped — "I'm stopping here, as
/// you asked. Here is the checkpoint."), or `None` when there is no such heading, i.e. the reply is
/// not a checkpoint (a refusal, or the model carrying on with the task in prose), and the re-render
/// runs instead. A safety net that holds whatever the instruction's wording. The heading match
/// tolerates `#`–`###` and any case, since every instruction this module writes asks for that format.
pub fn checkpoint_body(reply: &str) -> Option<&str> {
    let mut at = 0;
    for line in reply.split_inclusive('\n') {
        let heading = line.trim_start_matches('#');
        let hashes = line.len() - heading.len();
        if (1..=3).contains(&hashes)
            && heading
                .trim_start()
                .get(..4)
                .is_some_and(|w| w.eq_ignore_ascii_case("goal"))
        {
            return Some(reply[at..].trim_end());
        }
        at += line.len();
    }
    None
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
    // Request-only blocks on kept messages move with them; ones on summarized messages go with them.
    for b in &mut session.request_blocks {
        b.index = (b.index + 1).wrapping_sub(first_kept);
    }
    let messages = &session.messages;
    session
        .request_blocks
        .retain(|b| messages.get(b.index).is_some_and(|m| *m == b.anchor));
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
    // The rewrite above carries any masked content into the new list; the old indices mean nothing now.
    session.edited.clear();
}

/// The text a masked tool result starts with — how [`mask_tool_results`] recognizes its own work and
/// never masks a result twice.
pub const MASKED_PREFIX: &str = "[tool output elided to save context:";

/// A tool result is worth masking only past this many characters: anything shorter costs about what
/// its placeholder does.
const MASK_MIN_CHARS: usize = 1_000;

/// The input fields a placeholder quotes, first match wins: what identifies the call well enough for
/// the model to re-run it if it needs the output back.
const MASK_KEY_ARGS: [&str; 7] = [
    "path",
    "file_path",
    "command",
    "cmd",
    "pattern",
    "url",
    "query",
];

/// The result of [`mask_tool_results`]: a masked copy of the transcript, not yet applied.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskPlan {
    /// The transcript with the old tool results replaced by placeholders.
    pub messages: Vec<Message>,
    /// Indices of the messages that changed — what a persistence layer writes as edits.
    pub edited: Vec<usize>,
    /// How many tool results were masked.
    pub masked: usize,
    /// Estimated tokens freed (chars/4 of the elided output, plus 1,200 per dropped image).
    pub freed: u32,
}

/// Plan a masking pass: replace every tool result older than the last `keep_turns` turns (assistant
/// messages), or older than the last `keep_tokens` of transcript when those turns hold more, with a
/// one-line placeholder naming the call — `[tool output elided to save context:
/// read(path=src/lib.rs), 48213 chars. Run the tool again if you need it.]` — and drop its images.
/// `None` when nothing is worth masking.
///
/// Masking is the cheaper first step before a summary: old tool output is most of a coding transcript
/// and the part the model needs least once it has acted on it (arXiv 2508.21433 found masking matches
/// summarizing at about half the cost). There is deliberately no tool to recall a masked result: in
/// arXiv 2609.20804 one was rarely called and bought no accuracy; the placeholder says how to get the
/// output back by re-running the call.
///
/// Kept unmasked, besides the recent turns:
/// - results of calls on a path in `keep_paths` (the files being edited: the session's modified-files
///   provenance plus this transcript's own `write`/`edit` targets), whose content the model is likely
///   still working against;
/// - error results within the last `2 * keep_turns` turns: a recent failure is what the model is
///   usually fixing;
/// - results under [`MASK_MIN_CHARS`], and results already masked.
///
/// Pure and idempotent: masking a masked transcript plans nothing. Only `ToolResult` blocks change, in
/// place, so message count, roles and every `tool_use_id` pairing are exactly as they were — the
/// [`find_cut`] invariants hold across a pass by construction.
pub fn mask_tool_results(
    messages: &[Message],
    keep_turns: usize,
    keep_tokens: u32,
    keep_paths: &[String],
) -> Option<MaskPlan> {
    // The first message of the kept window: the `keep_turns`-th assistant message from the end.
    let turn_start = |turns: usize| {
        messages
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, m)| m.role == Role::Assistant)
            .nth(turns.max(1) - 1)
            .map(|(i, _)| i)
    };
    // Or, when those turns are bigger than `keep_tokens`, the first assistant message of the suffix
    // that holds about `keep_tokens` — whichever keeps less. Always at least the last turn.
    let last_turn = turn_start(1)?;
    let token_start = || {
        let mut held = 0u32;
        let i = messages.iter().rposition(|m| {
            held = held.saturating_add(estimate_message_tokens(m));
            held >= keep_tokens
        })?;
        Some(
            messages[i..]
                .iter()
                .position(|m| m.role == Role::Assistant)
                .map_or(last_turn, |k| i + k),
        )
    };
    let boundary = turn_start(keep_turns)
        .into_iter()
        .chain(token_start())
        .max()?
        .min(last_turn);
    let error_boundary = turn_start(keep_turns.saturating_mul(2)).unwrap_or(0);
    let (_, modified) = extract_file_ops(messages);
    let protected = |input: &Value| {
        ["path", "file_path"].iter().any(|k| {
            input
                .get(k)
                .and_then(Value::as_str)
                .is_some_and(|p| keep_paths.iter().chain(&modified).any(|m| m == p))
        })
    };
    let calls: std::collections::HashMap<&str, (&str, &Value)> = messages[..boundary]
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some((id.as_str(), (name.as_str(), input))),
            _ => None,
        })
        .collect();

    let mut out = messages.to_vec();
    let mut edited = Vec::new();
    let (mut masked, mut freed) = (0usize, 0u32);
    for (i, m) in out[..boundary].iter_mut().enumerate() {
        if m.role != Role::User {
            continue;
        }
        let mut changed = false;
        for b in &mut m.content {
            let ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
                images,
            } = b
            else {
                continue;
            };
            let chars = content.chars().count();
            if content.starts_with(MASKED_PREFIX)
                || (chars < MASK_MIN_CHARS && images.is_empty())
                || (*is_error && i >= error_boundary)
            {
                continue;
            }
            let call = calls.get(tool_use_id.as_str()).copied();
            if call.is_some_and(|(_, input)| protected(input)) {
                continue;
            }
            let placeholder = mask_placeholder(call, chars, images.len());
            freed = freed
                .saturating_add(chars.div_ceil(4) as u32)
                .saturating_add(1_200 * images.len() as u32)
                .saturating_sub(placeholder.chars().count().div_ceil(4) as u32);
            *content = Arc::from(placeholder);
            images.clear();
            masked += 1;
            changed = true;
        }
        if changed {
            edited.push(i);
        }
    }
    (masked > 0).then_some(MaskPlan {
        messages: out,
        edited,
        masked,
        freed,
    })
}

/// The one-line stand-in for an elided tool result: the call it answered, and how big it was.
fn mask_placeholder(call: Option<(&str, &Value)>, chars: usize, images: usize) -> String {
    let what = match call {
        Some((name, input)) => match MASK_KEY_ARGS
            .iter()
            .find_map(|k| input.get(k).and_then(Value::as_str).map(|v| (k, v)))
        {
            Some((k, v)) => format!("{name}({k}={})", truncate_chars(v, 120)),
            None => format!("{name}()"),
        },
        None => "a tool call".to_string(),
    };
    let images = match images {
        0 => String::new(),
        1 => " and 1 image".to_string(),
        n => format!(" and {n} images"),
    };
    format!("{MASKED_PREFIX} {what}, {chars} chars{images}. Run the tool again if you need it.]")
}

/// Every file a `bash` call in `messages` wrote, by [`crate::bash_writes::written_paths`], in
/// first-seen order, deduped — what `Agent::compact` adds to the modified-files provenance, beside
/// `write`/`edit` targets.
///
/// Why: Claude Sonnet 5.5 made 2 `write`/`edit` calls in 17 eval runs and edited through `bash`, so
/// 19 of its 23 summaries' `<modified-files>` missed files it had changed (102 in all). Over the
/// eval's 4,623 bash commands the parser recovers all of them, 25 of 25 sampled being real edits. A
/// parser, not a filesystem diff, so the provenance stays derivable from the transcript.
pub fn extract_bash_writes(messages: &[Message]) -> Vec<String> {
    let mut out = Vec::new();
    for b in messages.iter().flat_map(|m| &m.content) {
        if let ContentBlock::ToolUse { name, input, .. } = b
            && name == "bash"
            && let Some(command) = input.get("command").and_then(Value::as_str)
        {
            for path in crate::bash_writes::written_paths(command) {
                if !out.contains(&path) {
                    out.push(path);
                }
            }
        }
    }
    out
}

/// How fast the transcript grows: the estimated tokens of its last `turns` turns (from the
/// `turns`-th last assistant message on), per turn. At least 1.
pub fn growth_per_turn(messages: &[Message], turns: usize) -> u32 {
    let starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, m)| m.role == Role::Assistant)
        .map(|(i, _)| i)
        .take(turns.max(1))
        .collect();
    let Some(&start) = starts.last() else {
        return 1;
    };
    (estimate_messages_tokens(&messages[start..]) / starts.len() as u32).max(1)
}

/// What a masking pass saves, net, in base-input-token equivalents, before the room it frees
/// refills: every turn until then reads `freed` fewer cached tokens (0.1× each), and the pass pays
/// once to write `rewrite` tokens (from its first edited message on) at `write_premium` instead of
/// reading them at 0.1×. `freed / growth` turns until the transcript is back where it was. Positive:
/// the pass pays for itself. The model behind [`CompactionCandidates::mask_breakeven`].
pub fn mask_net_saving(freed: u32, rewrite: u32, growth_per_turn: u32, write_premium: f64) -> f64 {
    let turns = f64::from(freed) / f64::from(growth_per_turn.max(1));
    0.1 * f64::from(freed) * turns - (write_premium - 0.1) * f64::from(rewrite)
}

/// For [`CompactionCandidates::mask_best_start`]: of `plan`'s edits, the restriction to those at or
/// after some start index that maximizes [`mask_net_saving`], among starts that still free at least
/// `min_freed` — or `None` when no start does. `original` is the transcript `plan` was made from.
pub fn best_mask_start(
    original: &[Message],
    plan: &MaskPlan,
    min_freed: u32,
    growth_per_turn: u32,
    write_premium: f64,
) -> Option<MaskPlan> {
    // Suffix sums over the edits, latest first: what masking from each edit on frees, and how many
    // tokens of the masked transcript then sit at or after it (the rewrite).
    let mut best: Option<(f64, usize)> = None;
    let mut freed = 0u32;
    let mut next = plan.messages.len();
    let mut rewrite = 0u32;
    for &i in plan.edited.iter().rev() {
        freed = freed.saturating_add(
            estimate_message_tokens(&original[i])
                .saturating_sub(estimate_message_tokens(&plan.messages[i])),
        );
        rewrite = rewrite.saturating_add(estimate_messages_tokens(&plan.messages[i..next]));
        next = i;
        if freed < min_freed {
            continue;
        }
        let net = mask_net_saving(freed, rewrite, growth_per_turn, write_premium);
        if best.is_none_or(|(b, _)| net > b) {
            best = Some((net, i));
        }
    }
    let (net, start) = best?;
    if net < 0.0 {
        return None;
    }
    let mut messages = original.to_vec();
    let (mut edited, mut masked, mut kept_freed) = (Vec::new(), 0usize, 0u32);
    for &i in plan.edited.iter().filter(|&&i| i >= start) {
        kept_freed = kept_freed.saturating_add(
            estimate_message_tokens(&original[i])
                .saturating_sub(estimate_message_tokens(&plan.messages[i])),
        );
        masked += plan.messages[i]
            .content
            .iter()
            .zip(&original[i].content)
            .filter(|(new, old)| new != old)
            .count();
        messages[i] = plan.messages[i].clone();
        edited.push(i);
    }
    Some(MaskPlan {
        messages,
        edited,
        masked,
        freed: kept_freed,
    })
}

/// The `tokens_before` an [`apply_summary`] summary message records in its leading
/// `"Compacted from {N} tokens"` line — `None` for any other message. Lets a persistence layer that
/// sees only the session (not the `Compacted` event) record the compaction it is writing.
pub fn summary_tokens_before(m: &Message) -> Option<u32> {
    let ContentBlock::Text { text, .. } = m.content.first()? else {
        return None;
    };
    let body = text.strip_prefix(SUMMARY_MARKER)?.trim_start();
    let (n, _) = body
        .strip_prefix("Compacted from ")?
        .split_once(" tokens")?;
    n.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scaled_budgets_keep_less_than_the_threshold_that_triggers_a_compaction() {
        // The invariant the two budgets have to satisfy between them, across every window in the
        // catalogue: a compaction must *keep* less than the trigger threshold it just crossed.
        // Otherwise the very next turn re-triggers, pays for another summarization, and never gets
        // back under the line — which is exactly what the un-scaled 200k absolutes did on a 32k model
        // (trigger budget 16_384, keep_recent 20_000).
        for window in [4_096u32, 8_192, 32_768, 40_960, 128_000, 200_000, 1_000_000] {
            let cfg = CompactionConfig::for_window(window);
            let trigger_budget = window - cfg.reserve_tokens;
            assert!(
                cfg.keep_recent_tokens < trigger_budget,
                "window {window}: keep_recent ({}) must stay under the trigger budget ({trigger_budget})",
                cfg.keep_recent_tokens
            );
            assert!(
                cfg.reserve_tokens < window,
                "window {window}: the reserve must not swallow the whole context"
            );
        }
    }

    #[test]
    fn scaling_leaves_every_large_window_byte_for_byte_unchanged() {
        // The caps are one-sided on purpose: they must only bite on the small windows that were
        // actually broken. Any window at or above ~64k keeps precisely the tuning it had before.
        let d = CompactionConfig::default();
        for window in [128_000u32, 200_000, 256_000, 1_000_000] {
            let cfg = CompactionConfig::for_window(window);
            assert_eq!(cfg.reserve_tokens, d.reserve_tokens, "window {window}");
            assert_eq!(
                cfg.keep_recent_tokens, d.keep_recent_tokens,
                "window {window}"
            );
            assert_eq!(cfg.context_window, window);
        }
    }

    #[test]
    fn gpt_5_6_triggers_at_its_input_limit_less_the_reserve_not_its_window() {
        // GPT-5.6: a 1.05M window, but OpenAI rejects input past 922K. The old trigger
        // (`window - reserve`, ~1.03M) sat past that cap, so proactive compaction never fired.
        // Without its provisional 240K `compact_at`, which would otherwise fire first.
        let cfg = CompactionConfig {
            compact_at: None,
            ..CompactionConfig::for_model(&crate::models::capabilities("gpt-5.6-sol"))
        };
        assert_eq!(cfg.context_window, 1_050_000);
        assert_eq!(cfg.input_budget(), 922_000);
        let threshold = 922_000 - cfg.reserve_tokens;
        let mut s = Session::new();
        s.last_input_tokens = threshold;
        assert!(!should_compact(&s, &cfg), "exactly at the threshold");
        s.last_input_tokens = threshold + 1;
        assert!(should_compact(&s, &cfg), "one past it");
        // Same budgets as any large window: only the base the reserve comes off moved.
        let d = CompactionConfig::default();
        assert_eq!(cfg.reserve_tokens, d.reserve_tokens);
        assert_eq!(cfg.keep_recent_tokens, d.keep_recent_tokens);
    }

    #[test]
    fn compaction_fires_before_every_input_limited_model_rejects_the_prompt() {
        // Regression: for every model whose vendor caps input below the window, the soft trigger
        // lands strictly below the cap and the hard-overflow check fires at it.
        for id in [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-pro",
            "gpt-5.4-pro",
            "gpt-5.5-pro",
            "gpt-5.6",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "openai/gpt-5.2",
        ] {
            let caps = crate::models::capabilities(id);
            let limit = caps.input_limit.expect(id);
            assert!(limit < caps.context_window, "{id}");
            let cfg = CompactionConfig::for_model(&caps);
            let mut s = Session::new();
            s.last_input_tokens = limit - 1;
            assert!(
                should_compact(&s, &cfg),
                "{id}: no proactive compaction below the cap"
            );
            assert!(!is_hard_overflow(&s, &cfg, StopReason::EndTurn, 0), "{id}");
            s.last_input_tokens = limit;
            assert!(
                is_hard_overflow(&s, &cfg, StopReason::EndTurn, 0),
                "{id}: at the cap"
            );
        }
    }

    #[test]
    fn compact_at_is_the_trigger_itself_not_a_limit_less_the_reserve() {
        // Each capped family fires exactly past its cap; the reserve is not subtracted again.
        for (id, cap) in [
            ("claude-opus-5-5", 200_000),
            ("claude-fable-5-1", 200_000),
            ("claude-sonnet-5-5", 200_000),
            ("claude-haiku-5-5", 90_000),
            ("gpt-6.1-sol", 240_000),
            ("gpt-6-luna", 240_000),
            ("gpt-5.6-sol", 240_000),
            ("grok-4.6", 180_000),
            ("kimi-k3", 128_000),
        ] {
            let cfg = CompactionConfig::for_model(&crate::models::capabilities(id));
            assert_eq!(cfg.trigger_tokens(), cap, "{id}");
            let mut s = Session::new();
            s.last_input_tokens = cap;
            assert!(!should_compact(&s, &cfg), "{id}: exactly at the cap");
            s.last_input_tokens = cap + 1;
            assert!(should_compact(&s, &cfg), "{id}: one past it");
            // The hard-overflow backstop still measures the vendor's limit, not the cap.
            assert!(!is_hard_overflow(&s, &cfg, StopReason::EndTurn, 0), "{id}");
        }
        // An unlisted model gets the 200K fallback: a 1M window no longer runs to its end.
        for id in ["claude-opus-4-8", "some-new-1m-model"] {
            let caps = crate::models::ModelCaps {
                context_window: 1_000_000,
                ..crate::models::capabilities(id)
            };
            assert_eq!(
                CompactionConfig::for_model(&caps).trigger_tokens(),
                200_000,
                "{id}"
            );
        }
        // A window under the fallback is unchanged: `window - reserve` is still the lower line.
        let small = crate::models::capabilities("some-unknown-model");
        assert_eq!(small.context_window, 128_000);
        assert_eq!(
            CompactionConfig::for_model(&small).trigger_tokens(),
            128_000 - 16_384
        );
        // `--compact-at 0` (no cap) still restores the full-window trigger.
        let cfg = CompactionConfig {
            compact_at: None,
            ..CompactionConfig::for_model(&crate::models::capabilities("claude-opus-4-8"))
        };
        assert_eq!(cfg.trigger_tokens(), 1_000_000 - 16_384);
    }

    #[test]
    fn the_lower_line_wins_between_compact_at_and_the_window() {
        // A cap above what the window allows is moot: `window - reserve` fires first.
        let cfg = CompactionConfig {
            context_window: 100_000,
            compact_at: Some(200_000),
            ..CompactionConfig::default()
        };
        assert_eq!(cfg.trigger_tokens(), 100_000 - 16_384);
        // And under the window, the cap fires first.
        let cfg = CompactionConfig {
            compact_at: Some(50_000),
            ..cfg
        };
        assert_eq!(cfg.trigger_tokens(), 50_000);
        // keep_recent scales against the trigger, so a compaction lands well under it.
        let caps = crate::models::ModelCaps {
            compact_at: Some(30_000),
            ..crate::models::capabilities("claude-opus-5-5")
        };
        assert_eq!(
            CompactionConfig::for_model(&caps).keep_recent_tokens,
            15_000
        );
    }

    #[test]
    fn a_max_tokens_stop_still_measures_prompt_plus_output_against_the_window() {
        // The input cap moves only the input-side checks: a max-tokens cut-off compares prompt +
        // next output against the whole window, which still holds both.
        let cfg = CompactionConfig::for_model(&crate::models::capabilities("gpt-5.6"));
        let mut s = Session::new();
        s.last_input_tokens = 900_000;
        assert!(!is_hard_overflow(&s, &cfg, StopReason::MaxTokens, 128_000));
        assert!(is_hard_overflow(&s, &cfg, StopReason::MaxTokens, 150_000));
    }

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
    fn default_config_matches_pis_shipped_compaction_tuning() {
        // pi-parity fix: pi's `DEFAULT_COMPACTION_SETTINGS` (`compaction.ts:111-115`, and the shipped
        // coding-agent product agrees at `packages/coding-agent/src/core/compaction/compaction.ts:122-125`)
        // uses `reserveTokens: 16384, keepRecentTokens: 20000`. This crate used to diverge at
        // 24_000/40_000 — triggering compaction earlier (more reserved headroom) and retaining 2x more
        // raw verbatim context per round than the reference, for no documented reason.
        let cfg = CompactionConfig::default();
        assert_eq!(cfg.reserve_tokens, 16_384);
        assert_eq!(cfg.keep_recent_tokens, 20_000);
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
    fn every_summary_prompt_demands_verbatim_specifics() {
        // The one thing a summarizer must not do is smooth a specific away — every instruction that
        // shapes a summary must explicitly demand the categories most often lost (line numbers, literal
        // values, commands+outputs) be copied verbatim, not paraphrased.
        for (name, text) in [
            ("SUMMARY_SYSTEM", SUMMARY_SYSTEM),
            ("SUMMARY_INSTRUCTION", SUMMARY_INSTRUCTION),
            ("UPDATE_INSTRUCTION", UPDATE_INSTRUCTION),
            ("SPLIT_TURN_INSTRUCTION", SPLIT_TURN_INSTRUCTION),
            (
                "BRANCH_SUMMARY_INSTRUCTION",
                crate::branch_summary::BRANCH_SUMMARY_INSTRUCTION,
            ),
        ] {
            assert!(
                text.contains("verbatim"),
                "{name} must demand specifics be copied verbatim: {text}"
            );
            assert!(
                text.contains("line numbers"),
                "{name} must name file:line references as something to preserve: {text}"
            );
        }
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
    #[should_panic(expected = "must strictly alternate roles")]
    fn find_cut_rejects_two_consecutive_same_role_messages() {
        // This function's whole tool_use/tool_result-pairing safety argument (see its own doc
        // comment) depends on strict role alternation — this is a debug-only invariant check (not a
        // runtime check on untrusted input), so it only fires in test/debug builds.
        let messages = vec![
            Message::user("one"),
            Message::assistant(vec![ContentBlock::text("two")]),
            Message::assistant(vec![ContentBlock::text("three")]),
            Message::user("four"),
        ];
        find_cut(&messages, 1);
    }

    #[test]
    fn extract_file_ops_buckets_reads_and_writes() {
        let (read, modified) = extract_file_ops(&convo());
        assert_eq!(read, vec!["src/foo.rs"]);
        assert_eq!(modified, vec!["src/foo.rs"]);
    }

    #[test]
    fn extract_file_ops_does_not_count_ls_as_a_read() {
        // Task #31 (pi-parity fix): a directory listing must not dilute the read-files list — matches
        // pi's `compaction/utils.ts:44-55` (`read`/`write`/`edit` only, no `ls`).
        let messages = vec![
            Message::user("look around"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "ls",
                json!({ "path": "src/" }),
            )]),
            Message::tool_result("1", "foo.rs\nbar.rs", false),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "read",
                json!({ "path": "src/foo.rs" }),
            )]),
            Message::tool_result("2", "fn foo() {}", false),
        ];
        let (read, modified) = extract_file_ops(&messages);
        assert_eq!(
            read,
            vec!["src/foo.rs".to_string()],
            "an `ls` call must not appear in (or otherwise pollute) the read-files list"
        );
        assert!(modified.is_empty());
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
            Message::tool_result("1", &*big, false),
            // A concluding assistant response — without this the prefix ends on a bare tool_result,
            // which `is_split_turn` (correctly) reads as an in-progress turn and would swap in
            // `SPLIT_TURN_INSTRUCTION` instead of the generic template this test is exercising.
            Message::assistant(vec![ContentBlock::text("done")]),
        ];
        let file_ops = merge_provenance(
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
        let file_ops = merge_provenance(
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
        let file_ops = merge_provenance(
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
            text: long.clone().into(),
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

        let file_ops = merge_provenance(
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
        let round1_ops = merge_provenance(
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
        let round2_ops = merge_provenance(&round1_ops, &round2_prefix, CompactionReason::Threshold);

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
    fn merge_provenance_dedupes_a_path_seen_again_in_a_later_round() {
        let previous = CompactionProvenance {
            read_files: vec!["a.rs".into()],
            modified_files: vec![],
            compactions: 1,
            last_reason: Some(CompactionReason::Threshold),
            todos: None,
            memory_notes: vec![],
            user_requests: vec![],
        };
        let messages = vec![Message::assistant(vec![ContentBlock::tool_use(
            "1",
            "read",
            json!({ "path": "a.rs" }), // the same file, read again
        )])];
        let merged = merge_provenance(&previous, &messages, CompactionReason::Manual);
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
        let file_ops = merge_provenance(
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
        let file_ops = merge_provenance(
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
        let file_ops = merge_provenance(
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
        let file_ops = merge_provenance(
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
        // The recorded size reads back off the summary, and off nothing else.
        assert_eq!(summary_tokens_before(&s.messages[0]), Some(9999));
        assert_eq!(summary_tokens_before(&s.messages[1]), None);
        assert_eq!(
            summary_tokens_before(&Message::user("Compacted from 5 tokens")),
            None
        );
    }

    // ---- the `todo` deterministic-carry channel -------------------------------------------------

    /// A `todo` tool call, as the loop records it.
    fn todo_call(id: &str, items: &[(&str, &str)]) -> Message {
        let todos: Vec<Value> = items
            .iter()
            .map(|(content, status)| {
                json!({ "content": content, "activeForm": format!("{content}ing"), "status": status })
            })
            .collect();
        Message::assistant(vec![ContentBlock::tool_use(
            id,
            "todo",
            json!({ "todos": todos }),
        )])
    }

    #[test]
    fn extract_todos_takes_the_last_call_not_the_first() {
        // Full-replace semantics: the newest call is the whole truth, every earlier one is dead.
        let messages = vec![
            todo_call("1", &[("Old plan", "pending")]),
            Message::user("go on"),
            todo_call("2", &[("New plan", "in_progress")]),
        ];
        let todos = extract_todos(&messages).expect("a todo call is present");
        assert_eq!(todos[0]["content"], "New plan");
    }

    #[test]
    fn extract_todos_is_none_without_a_todo_call() {
        assert!(extract_todos(&convo()).is_none());
    }

    fn mem_call(id: &str, command: &str, path: &str) -> Message {
        Message::assistant(vec![ContentBlock::tool_use(
            id,
            "memory",
            json!({ "command": command, "path": path }),
        )])
    }

    #[test]
    fn extract_memory_notes_collects_writes_first_seen_and_ignores_the_rest() {
        let messages = vec![
            mem_call("1", "create", "/session/facts.md"),
            mem_call("2", "view", "/session/facts.md"), // a read — ignored
            mem_call("3", "str_replace", "/session/facts.md"), // same path, deduped
            mem_call("4", "insert", "/memories/notes.md"), // the other root counts too
            mem_call("5", "delete", "/session/gone.md"), // a delete adds nothing
            Message::assistant(vec![ContentBlock::tool_use(
                "6",
                "write", // a different tool that happens to carry a path — not memory
                json!({ "path": "/session/decoy.md" }),
            )]),
        ];
        assert_eq!(
            extract_memory_notes(&messages),
            vec![
                "/session/facts.md".to_string(),
                "/memories/notes.md".to_string()
            ]
        );
    }

    #[test]
    fn format_memory_notes_renders_a_block_or_nothing() {
        assert_eq!(format_memory_notes(&[]), "");
        let block = format_memory_notes(&["/session/a.md".into(), "/memories/b.md".into()]);
        assert_eq!(
            block,
            "\n\n<memory-notes>\n/session/a.md\n/memories/b.md\n</memory-notes>"
        );
    }

    #[test]
    fn merge_provenance_accumulates_memory_notes_across_rounds_deduped() {
        let round1 = merge_provenance(
            &CompactionProvenance::default(),
            &[mem_call("1", "create", "/session/a.md")],
            CompactionReason::Threshold,
        );
        assert_eq!(round1.memory_notes, vec!["/session/a.md".to_string()]);
        let round2 = merge_provenance(
            &round1,
            &[
                mem_call("2", "str_replace", "/session/a.md"), // already known — deduped
                mem_call("3", "create", "/session/b.md"),      // new
            ],
            CompactionReason::Threshold,
        );
        assert_eq!(
            round2.memory_notes,
            vec!["/session/a.md".to_string(), "/session/b.md".to_string()]
        );
    }

    #[test]
    fn previous_summary_strips_the_memory_notes_carry_block() {
        // A summary body that ends with a memory-notes block must have it peeled off before it is fed
        // forward, exactly like the file/todo blocks — otherwise every incremental round re-appends one.
        let body = "The model saved some notes.\n\n<memory-notes>\n/session/a.md\n</memory-notes>";
        let msg = Message::user(format!("{SUMMARY_MARKER}\n\n{body}"));
        assert_eq!(
            previous_summary(&[msg]),
            Some("The model saved some notes.")
        );
    }

    #[test]
    fn extract_todos_ignores_a_todo_call_with_a_non_array_todos_argument() {
        // A malformed call the loop still recorded (the tool rejected it, but the `tool_use` block is
        // in history regardless). Carrying a non-array forward would poison every later render.
        let messages = vec![Message::assistant(vec![ContentBlock::tool_use(
            "1",
            "todo",
            json!({ "todos": "not an array" }),
        )])];
        assert!(extract_todos(&messages).is_none());
    }

    #[test]
    fn render_todo_list_maps_each_status_to_its_glyph() {
        let todos = json!([
            { "content": "Parse the config", "status": "completed" },
            { "content": "Wire the retry loop", "status": "in_progress" },
            { "content": "Add tests", "status": "pending" },
        ]);
        assert_eq!(
            render_todo_list(todos.as_array().expect("array")),
            "[x] Parse the config\n[>] Wire the retry loop\n[ ] Add tests"
        );
    }

    #[test]
    fn render_todo_list_degrades_a_malformed_item_instead_of_panicking() {
        // This also runs over a `todos` value restored from a persisted session an older build wrote.
        let todos = json!([{ "status": "bogus" }, { "content": "Real" }]);
        assert_eq!(
            render_todo_list(todos.as_array().expect("array")),
            "[ ] (untitled)\n[ ] Real"
        );
    }

    #[test]
    fn format_todo_list_is_empty_for_none_and_for_a_cleared_list() {
        // Appending this to a summary must always be safe, unconditionally.
        assert_eq!(format_todo_list(None), "");
        assert_eq!(format_todo_list(Some(&json!([]))), "");
        // A non-array can only arrive from a corrupted/hand-edited session file.
        assert_eq!(format_todo_list(Some(&json!("garbage"))), "");
    }

    #[test]
    fn format_todo_list_renders_a_tagged_block() {
        let todos = json!([{ "content": "Ship it", "status": "in_progress" }]);
        assert_eq!(
            format_todo_list(Some(&todos)),
            "\n\n<todo_list>\n[>] Ship it\n</todo_list>"
        );
    }

    #[test]
    fn merge_provenance_replaces_the_todo_list_rather_than_accumulating_it() {
        // Unlike file paths, a todo list is a full replace — folding two rounds together would
        // resurrect steps the model deliberately dropped.
        let previous = CompactionProvenance {
            todos: Some(json!([{ "content": "Old", "status": "pending" }])),
            ..Default::default()
        };
        let merged = merge_provenance(
            &previous,
            &[todo_call("1", &[("New", "pending")])],
            CompactionReason::Threshold,
        );
        let todos = merged.todos.expect("carried forward");
        assert_eq!(todos.as_array().map(Vec::len), Some(1));
        assert_eq!(todos[0]["content"], "New");
    }

    #[test]
    fn merge_provenance_carries_the_todo_list_forward_across_a_round_that_has_none() {
        // The exact failure this channel exists to prevent: the model plans, works for many turns
        // without touching the list, and a compaction folds the planning turn away.
        let previous = CompactionProvenance {
            todos: Some(json!([{ "content": "Keep me", "status": "in_progress" }])),
            ..Default::default()
        };
        let merged = merge_provenance(&previous, &convo(), CompactionReason::Threshold);
        assert_eq!(
            merged.todos.expect("carried forward")[0]["content"],
            "Keep me"
        );
    }

    #[test]
    fn merge_provenance_lets_an_explicitly_cleared_list_win_over_an_older_one() {
        // `Some([])` must beat `previous`, or a plan the model finished with would silently reappear.
        let previous = CompactionProvenance {
            todos: Some(json!([{ "content": "Done with this", "status": "completed" }])),
            ..Default::default()
        };
        let merged = merge_provenance(
            &previous,
            &[todo_call("1", &[])],
            CompactionReason::Threshold,
        );
        assert_eq!(merged.todos, Some(json!([])));
        assert_eq!(format_todo_list(merged.todos.as_ref()), "");
    }

    #[test]
    fn previous_summary_strips_the_carry_blocks_the_host_appended() {
        // Left in, the summarizing model echoes a stale copy into its prose right before `compact`
        // appends the freshly-merged real ones.
        let body = format!(
            "Prose about the work.{}{}",
            format_file_operations(&["a.rs".into()], &["b.rs".into()]),
            format_todo_list(Some(
                &json!([{ "content": "Ship it", "status": "pending" }])
            ))
        );
        let msg = Message::user(format!(
            "{SUMMARY_MARKER}\n\nCompacted from 500 tokens\n\n{body}"
        ));
        assert_eq!(
            previous_summary(&[msg]),
            Some("Prose about the work."),
            "the marker, token line, and every carry block must be stripped"
        );
    }

    #[test]
    fn previous_summary_leaves_a_tag_name_the_model_wrote_in_its_own_prose_alone() {
        // Only *trailing* blocks are peeled; a tag quoted mid-prose is the model's own content.
        let body = "I read <read-files> from the tool output, then stopped.";
        let msg = Message::user(format!("{SUMMARY_MARKER}\n\n{body}"));
        assert_eq!(previous_summary(&[msg]), Some(body));
    }

    #[test]
    fn a_todo_list_survives_two_consecutive_compactions() {
        // The load-bearing test for the whole channel. Round 1 folds away the turn that planned; round
        // 2's prefix is `[summary#1, ...new activity]`, which contains no `ToolUse` block at all — so
        // without provenance the plan would be gone, and with only an LLM-rewritten summary it would be
        // gone at the model's discretion. `merge_provenance` must carry it through both rounds and
        // `format_todo_list` must re-render it into each new summary verbatim.
        let round1_prefix = vec![
            Message::user("plan and do the work"),
            todo_call("1", &[("Wire the retry loop", "in_progress")]),
        ];
        let round1 = merge_provenance(
            &CompactionProvenance::default(),
            &round1_prefix,
            CompactionReason::Threshold,
        );
        let summary1 = format!(
            "The model planned the work.{}",
            format_todo_list(round1.todos.as_ref())
        );
        assert!(summary1.contains("[>] Wire the retry loop"));

        // Round 2's prefix opens with round 1's summary message and carries only new, todo-less
        // activity after it.
        let round2_prefix = vec![
            Message::user(format!(
                "{SUMMARY_MARKER}\n\nCompacted from 500 tokens\n\n{summary1}"
            )),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "read",
                json!({ "path": "src/lib.rs" }),
            )]),
        ];
        assert!(
            extract_todos(&round2_prefix).is_none(),
            "round 2's prefix has no `todo` tool_use block — provenance is the only surviving copy"
        );

        let round2 = merge_provenance(&round1, &round2_prefix, CompactionReason::Threshold);
        let summary2 = format!(
            "The model kept working.{}",
            format_todo_list(round2.todos.as_ref())
        );
        assert!(
            summary2.contains("[>] Wire the retry loop"),
            "the plan must survive the second compaction: {summary2}"
        );
        assert_eq!(round2.compactions, 2);

        // And the block is appended exactly once — `previous_summary` peeled round 1's copy off before
        // it could be fed forward and re-appended.
        assert_eq!(summary2.matches("<todo_list>").count(), 1);
        assert_eq!(
            previous_summary(&round2_prefix),
            Some("The model planned the work.")
        );
    }

    // ---- masking old tool output ----------------------------------------------------------------

    /// A task, then `n` turns of `name` calls on `src/f{i}.rs`, each returning `chars` of output.
    fn tool_history(n: usize, name: &str, chars: usize) -> Vec<Message> {
        let mut out = vec![Message::user("the task")];
        for i in 0..n {
            out.push(Message::assistant(vec![ContentBlock::tool_use(
                format!("c{i}"),
                name,
                json!({ "path": format!("src/f{i}.rs") }),
            )]));
            out.push(Message::tool_result(
                format!("c{i}"),
                "x".repeat(chars),
                false,
            ));
        }
        out
    }

    fn result_content(m: &Message) -> &str {
        match &m.content[0] {
            ContentBlock::ToolResult { content, .. } => content,
            other => panic!("not a tool result: {other:?}"),
        }
    }

    #[test]
    fn masking_elides_results_older_than_the_kept_turns_and_nothing_else() {
        let history = tool_history(6, "read", 4_000);
        let plan = mask_tool_results(&history, 2, u32::MAX, &[]).expect("something to mask");
        // Turns 0..4 masked (their results sit at 2, 4, 6, 8); the last two turns untouched.
        assert_eq!(plan.masked, 4);
        assert_eq!(plan.edited, vec![2, 4, 6, 8]);
        for i in [2, 4, 6, 8] {
            let c = result_content(&plan.messages[i]);
            assert!(c.starts_with(MASKED_PREFIX), "{c}");
            assert!(c.contains("read(path=src/f"), "names the call: {c}");
            assert!(c.contains("4000 chars"), "{c}");
        }
        for i in [10, 12] {
            assert_eq!(plan.messages[i], history[i], "kept turn {i} changed");
        }
        // Each masked result frees its output less its placeholder.
        assert!(
            plan.freed > 4 * 900 && plan.freed < 4 * 1_000,
            "{}",
            plan.freed
        );
        // Everything but the elided results is byte-identical.
        for (i, (a, b)) in history.iter().zip(&plan.messages).enumerate() {
            if !plan.edited.contains(&i) {
                assert_eq!(a, b, "message {i}");
            }
        }
    }

    #[test]
    fn masking_keeps_every_pairing_and_alternation_invariant() {
        // A turn with several parallel calls, a steer folded onto the results, an image result.
        let mut history = vec![Message::user("the task")];
        for t in 0..4 {
            history.push(Message::assistant(vec![
                ContentBlock::tool_use(format!("a{t}"), "bash", json!({ "command": "make" })),
                ContentBlock::tool_use(format!("b{t}"), "grep", json!({ "pattern": "fn " })),
            ]));
            history.push(Message {
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: format!("a{t}"),
                        content: "y".repeat(3_000).into(),
                        is_error: false,
                        images: vec![],
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: format!("b{t}"),
                        content: "z".repeat(3_000).into(),
                        is_error: false,
                        images: vec![],
                    },
                    ContentBlock::text("also check the docs"),
                ],
                ..Message::user("")
            });
        }
        let plan = mask_tool_results(&history, 1, u32::MAX, &[]).unwrap();
        let masked = &plan.messages;
        assert_eq!(masked.len(), history.len());
        assert!(masked.windows(2).all(|w| w[0].role != w[1].role));
        for (a, b) in history.iter().zip(masked) {
            assert_eq!(a.role, b.role);
            assert_eq!(a.content.len(), b.content.len());
            for (x, y) in a.content.iter().zip(&b.content) {
                match (x, y) {
                    (
                        ContentBlock::ToolResult { tool_use_id: i, .. },
                        ContentBlock::ToolResult { tool_use_id: j, .. },
                    ) => assert_eq!(i, j, "pairing id changed"),
                    _ => assert_eq!(x, y, "a non-result block changed"),
                }
            }
        }
        assert!(result_content(&masked[2]).contains("bash(command=make)"));
        // The cut search still finds the same assistant boundary on the masked transcript.
        assert_eq!(find_cut(masked, 1), find_cut(&history, 1));
    }

    #[test]
    fn masking_keeps_edited_files_recent_errors_short_results_and_its_own_work() {
        let mut history = tool_history(8, "read", 4_000);
        // Turn 1 edits src/f1.rs; its read result (index 4) stays. Turn 2's file is protected by
        // provenance from an earlier compaction.
        history.push(Message::assistant(vec![ContentBlock::tool_use(
            "e",
            "edit",
            json!({ "path": "src/f1.rs" }),
        )]));
        history.push(Message::tool_result("e", "ok", false));
        // An old error (turn 0) and a recent one (turn 4: older than the 3 kept turns, inside 2 × 3).
        for i in [2, 10] {
            let c = history[i].content[0].clone();
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } = c
            {
                history[i] = Message {
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error: true,
                        images: vec![],
                    }],
                    ..history[i].clone()
                };
            }
        }
        // A short result (turn 3) costs about what its placeholder would.
        history[8] = Message::tool_result("c3", "short", false);

        let plan = mask_tool_results(&history, 3, u32::MAX, &["src/f2.rs".to_string()]).unwrap();
        let masked = |i: usize| result_content(&plan.messages[i]).starts_with(MASKED_PREFIX);
        assert!(masked(2), "an old error is masked");
        assert!(!masked(4), "src/f1.rs is being edited");
        assert!(
            !masked(6),
            "src/f2.rs is in the provenance's modified files"
        );
        assert!(!masked(8), "too short to be worth it");
        assert!(!masked(10), "a recent error stays");
        assert!(masked(12));
        // Idempotent: a masked transcript has nothing left to mask.
        assert_eq!(
            mask_tool_results(&plan.messages, 3, u32::MAX, &["src/f2.rs".to_string()]),
            None
        );
    }

    #[test]
    fn masking_drops_images_and_counts_them() {
        let mut history = tool_history(3, "read", 10);
        history[2] = Message {
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "c0".into(),
                content: "screenshot".into(),
                is_error: false,
                images: vec![crate::message::ImageSource::base64("image/png", "AAAA")],
            }],
            ..history[2].clone()
        };
        let plan = mask_tool_results(&history, 1, u32::MAX, &[]).unwrap();
        assert_eq!(plan.masked, 1, "only the image result is worth masking");
        let ContentBlock::ToolResult {
            content, images, ..
        } = &plan.messages[2].content[0]
        else {
            panic!()
        };
        assert!(images.is_empty());
        assert!(content.contains("and 1 image"), "{content}");
        assert!(plan.freed > 1_000, "{}", plan.freed);
    }

    #[test]
    fn masking_needs_more_turns_than_it_keeps() {
        assert_eq!(
            mask_tool_results(&tool_history(2, "read", 4_000), 2, u32::MAX, &[]),
            None
        );
        assert_eq!(
            mask_tool_results(&tool_history(2, "read", 4_000), 3, u32::MAX, &[]),
            None
        );
        assert!(mask_tool_results(&tool_history(3, "read", 4_000), 2, u32::MAX, &[]).is_some());
    }

    #[test]
    fn a_token_budget_shrinks_the_kept_window_when_recent_turns_are_big() {
        // Twelve turns of ~1K tokens: ten kept turns hold ~10K. With a 2,500-token budget only the
        // last two or three turns stay; everything older is masked.
        let history = tool_history(12, "read", 4_000);
        let by_turns = mask_tool_results(&history, 10, u32::MAX, &[]).unwrap();
        assert_eq!(by_turns.masked, 2);
        let by_tokens = mask_tool_results(&history, 10, 2_500, &[]).unwrap();
        assert!((9..=10).contains(&by_tokens.masked), "{}", by_tokens.masked);
        // The last turn is always kept, however small the budget.
        let tiny = mask_tool_results(&history, 10, 1, &[]).unwrap();
        assert_eq!(tiny.masked, 11);
        // A budget larger than the turns leaves the turn count in charge.
        assert_eq!(
            mask_tool_results(&history, 10, 1_000_000, &[])
                .unwrap()
                .masked,
            2
        );
    }

    #[test]
    fn the_masking_budgets_scale_with_the_trigger() {
        for (cap, keep, min_free) in [
            (90_000, 22_500, 9_000),
            (200_000, 50_000, 20_000),
            (240_000, 60_000, 24_000),
            (500_000, 125_000, 25_000),
        ] {
            let cfg = CompactionConfig {
                context_window: 1_000_000,
                compact_at: Some(cap),
                ..CompactionConfig::default()
            };
            assert_eq!(cfg.mask_keep_tokens(), keep, "{cap}");
            assert_eq!(cfg.mask_min_free(), min_free, "{cap}");
        }
        let pinned = CompactionConfig {
            mask_keep_tokens: Some(7),
            ..CompactionConfig::default()
        };
        assert_eq!(pinned.mask_keep_tokens(), 7);
    }

    #[test]
    fn the_cached_instruction_is_the_handoff_framing_then_the_format_and_tails() {
        let prior = Message::user(format!("{SUMMARY_MARKER}\n\n## Goal\nold"));
        let messages = vec![prior, Message::assistant(vec![ContentBlock::text("ok")])];
        let ops = CompactionProvenance {
            read_files: vec!["src/a.rs".into()],
            modified_files: vec!["src/b.rs".into()],
            ..Default::default()
        };
        let text = conversation_summary_instruction(
            &messages,
            &ops,
            Some("focus"),
            &CompactionCandidates::default(),
            SummaryFraming::Handoff,
        );
        let format =
            &SUMMARY_INSTRUCTION[SUMMARY_INSTRUCTION.find(SUMMARY_FORMAT_START).unwrap()..];
        assert!(text.starts_with(&format!("<system-reminder>{HANDOFF_FRAMING}{format}")));
        assert!(!text.contains(SUMMARY_SYSTEM));
        assert!(!text.contains("another LLM"));
        assert!(
            text.contains("Fold it in: PRESERVE"),
            "the prior-summary rules ride along"
        );
        assert!(text.ends_with(
            "<read-files>\nsrc/a.rs\n</read-files>\n\n<modified-files>\nsrc/b.rs\n</modified-files>\
             \n\nAdditional focus: focus</system-reminder>"
        ));
    }

    #[test]
    fn the_compact_framing_keeps_the_old_opener_and_the_same_tails() {
        let prior = Message::user(format!("{SUMMARY_MARKER}\n\n## Goal\nold"));
        let messages = vec![prior, Message::assistant(vec![ContentBlock::text("ok")])];
        let ops = CompactionProvenance {
            read_files: vec!["src/a.rs".into()],
            ..Default::default()
        };
        let text = conversation_summary_instruction(
            &messages,
            &ops,
            None,
            &CompactionCandidates::default(),
            SummaryFraming::Compact,
        );
        assert!(text.starts_with("<system-reminder>Stop working on the task now."));
        assert!(text.contains(SUMMARY_SYSTEM) && text.contains(SUMMARY_INSTRUCTION));
        assert!(!text.contains(HANDOFF_FRAMING));
        assert!(text.contains("Fold it in: PRESERVE"));
        assert!(text.ends_with("<read-files>\nsrc/a.rs\n</read-files></system-reminder>"));
        let state = conversation_summary_instruction(
            &messages,
            &ops,
            None,
            &CompactionCandidates {
                state_template: true,
                ..Default::default()
            },
            SummaryFraming::Compact,
        );
        assert!(state.contains(&format!(
            "{STATE_INTRO}{STATE_RULES}{STATE_FORMAT}{STATE_FOLD_IN}"
        )));
    }

    /// The two runaways GPT-6 Luna wrote in the replay, rebuilt exactly: the captured text up to the
    /// loop, then the loop (`- ` lines) as many times as it ran before the output limit.
    fn luna_runaways() -> [String; 2] {
        [
            format!(
                "{}{}-",
                include_str!("../tests/fixtures/luna_runaway_marshmallow_head.txt"),
                "- \n".repeat(15_152)
            ),
            format!(
                "{}{}",
                include_str!("../tests/fixtures/luna_runaway_sqlparse_head.txt"),
                "- \n".repeat(14_267)
            ),
        ]
    }

    #[test]
    fn a_runaway_loop_is_caught_though_it_opens_as_a_checkpoint() {
        for reply in luna_runaways() {
            assert!(
                checkpoint_body(&reply).is_some(),
                "it passes the heading check"
            );
            assert!(is_runaway(&reply));
        }
        // The marshmallow reply's text before the loop is a sound checkpoint.
        let head = include_str!("../tests/fixtures/luna_runaway_marshmallow_head.txt");
        assert!(!is_runaway(head));
        // A real summary: bullets, a code fence, blank lines, a few repeats that aren't a run.
        let ok = "## Goal\nfix it\n\n## Progress\n- a\n- a\n- b\n\n```\nx\n```\n- \n- c\n- \n";
        assert!(!is_runaway(ok));
        let nine = format!("## Goal\n{}", "- \n".repeat(RUNAWAY_REPEATED_LINES - 1));
        assert!(!is_runaway(&nine));
        // Blank lines between the repeats don't hide a loop.
        let spaced = format!("## Goal\n{}", "- \n\n".repeat(RUNAWAY_REPEATED_LINES));
        assert!(is_runaway(&spaced));
    }

    #[test]
    fn checkpoint_body_finds_the_goal_heading_and_drops_the_preamble() {
        assert_eq!(checkpoint_body("## Goal\nx\n"), Some("## Goal\nx"));
        assert_eq!(
            checkpoint_body("I'm stopping here. Here is the checkpoint.\n\n## Goal\nx"),
            Some("## Goal\nx")
        );
        assert_eq!(checkpoint_body("### goal\nx"), Some("### goal\nx"));
        assert_eq!(checkpoint_body("# GOAL: ship it"), Some("# GOAL: ship it"));
        // Not a checkpoint: a refusal, prose that mentions the word, or a too-deep heading.
        assert_eq!(
            checkpoint_body(
                "I'm not going to produce that checkpoint summary; the transcript stays with me."
            ),
            None
        );
        assert_eq!(checkpoint_body("My goal is to keep working.\n"), None);
        assert_eq!(checkpoint_body("#### Goal\nx"), None);
        assert_eq!(checkpoint_body(""), None);
    }

    #[test]
    fn candidates_parse_by_name_and_reject_unknown_ones() {
        assert_eq!(
            CompactionCandidates::from_names::<&str>(&[]),
            Ok(CompactionCandidates::default())
        );
        assert_eq!(
            CompactionCandidates::from_names(&["state", " tail", ""]),
            Ok(CompactionCandidates {
                state_template: true,
                sees_tail: true,
                ..Default::default()
            })
        );
        let all = CompactionCandidates::from_names(CompactionCandidates::NAMES).unwrap();
        assert!(all.state_template && all.sees_tail && all.mask_best_start);
        let err = CompactionCandidates::from_names(&["sate"]).unwrap_err();
        assert!(err.contains("sate") && err.contains("state"), "{err}");
        // A promoted or deleted candidate's name is unknown now, not a silent no-op.
        for gone in [
            "system-role",
            "handoff",
            "check",
            "bash-edits",
            "carry-request",
        ] {
            assert!(CompactionCandidates::from_names(&[gone]).is_err(), "{gone}");
        }
    }

    #[test]
    fn summary_request_is_summary_request_with_no_candidates() {
        let prefix = vec![
            Message::user("the task"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({"path": "a"}),
            )]),
            Message::tool_result("1", "a", false),
        ];
        let ops = CompactionProvenance::default();
        assert_eq!(
            summary_request("m", &prefix, 100, &ops, Some("f")).messages,
            summary_request_with(
                "m",
                &prefix,
                &[],
                100,
                &ops,
                Some("f"),
                &CompactionCandidates::default()
            )
            .messages
        );
    }

    #[test]
    fn the_state_template_replaces_every_re_render_instruction() {
        let state = CompactionCandidates {
            state_template: true,
            ..Default::default()
        };
        let ops = CompactionProvenance::default();
        // A prefix ending mid-turn would get the split-turn template; the state template never does.
        let split = vec![
            Message::user("the task"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({"path": "a"}),
            )]),
            Message::tool_result("1", "a", false),
        ];
        assert!(is_split_turn(&split));
        let text = |prefix: &[Message], tail: &[Message]| {
            let req = summary_request_with("m", prefix, tail, 100, &ops, None, &state);
            match &req.messages[0].content[0] {
                ContentBlock::Text { text, .. } => text.to_string(),
                _ => unreachable!(),
            }
        };
        let fresh = text(&split, &[]);
        assert!(fresh.ends_with(&format!("{STATE_INTRO}{STATE_RULES}{STATE_FORMAT}")));
        assert!(!fresh.contains(SPLIT_TURN_INSTRUCTION) && !fresh.contains("<recent-context>"));
        // With a previous summary: the rewrite-not-append opener.
        let mut later = vec![Message::user(format!("{SUMMARY_MARKER}\n\n## Goal\nold"))];
        later.extend_from_slice(&split[1..]);
        let update = text(&later, &[]);
        assert!(update.ends_with(&format!("{STATE_UPDATE_INTRO}{STATE_RULES}{STATE_FORMAT}")));
        // A tail: rendered after the transcript, read-only, with the note saying so.
        let tail = vec![Message::assistant(vec![ContentBlock::text("now testing")])];
        let with_tail = text(&split, &tail);
        assert!(with_tail.contains("<recent-context>\nAssistant: now testing\n</recent-context>"));
        assert!(with_tail.ends_with(TAIL_NOTE));
    }

    #[test]
    fn the_state_template_rides_the_cached_instruction() {
        let prior = vec![Message::user(format!("{SUMMARY_MARKER}\n\n## Goal\nold"))];
        let text = conversation_summary_instruction(
            &prior,
            &CompactionProvenance::default(),
            None,
            &CompactionCandidates {
                state_template: true,
                ..Default::default()
            },
            SummaryFraming::Handoff,
        );
        assert!(text.starts_with(&format!(
            "<system-reminder>{HANDOFF_FRAMING}{STATE_RULES}{STATE_FORMAT}{STATE_FOLD_IN}"
        )));
        assert!(!text.contains(SUMMARY_INSTRUCTION) && !text.contains("Fold it in: PRESERVE"));
    }

    #[test]
    fn user_requests_are_the_users_own_turns_deduped_and_capped() {
        let summary = Message::user(format!("{SUMMARY_MARKER}\n\nold"));
        let messages = vec![
            summary,
            Message::user("first: fix the parser"),
            Message::assistant(vec![ContentBlock::tool_use("1", "read", json!({}))]),
            Message::tool_result("1", "the file", false),
            Message::assistant(vec![ContentBlock::text("done?")]),
            Message::user("first: fix the parser"),
        ];
        assert_eq!(
            merge_user_requests(&[], &messages),
            ["first: fix the parser"],
            "the summary and tool results are not requests; a repeat is deduped"
        );
        let long = "y".repeat(USER_REQUEST_MAX_CHARS + 10);
        let got = merge_user_requests(&[], &[Message::user(long)]);
        assert!(
            got[0].chars().count() < USER_REQUEST_MAX_CHARS + 50 && got[0].contains("truncated")
        );
        // Past the cap: the first request and the latest ones.
        let previous: Vec<String> = (0..USER_REQUESTS_KEPT).map(|i| format!("r{i}")).collect();
        let got = merge_user_requests(&previous, &[Message::user("newest")]);
        assert_eq!(got.len(), USER_REQUESTS_KEPT);
        assert_eq!(got.first().map(String::as_str), Some("r0"));
        assert_eq!(got.last().map(String::as_str), Some("newest"));
        assert!(!got.contains(&"r1".to_string()));
    }

    #[test]
    fn the_user_requests_block_is_stripped_like_every_carry_block() {
        let block = format_user_requests(&["fix it".into(), "and the docs".into()]);
        assert!(block.contains("fix it\n---\nand the docs\n</user-requests>"));
        assert_eq!(format_user_requests(&[]), "");
        let body = format!(
            "## Goal\nx{block}{}",
            format_file_operations(&["a.rs".into()], &[])
        );
        let message = Message::user(format!(
            "{SUMMARY_MARKER}\n\nCompacted from 5 tokens\n\n{body}"
        ));
        assert_eq!(
            previous_summary(std::slice::from_ref(&message)),
            Some("## Goal\nx")
        );
    }

    #[test]
    fn bash_writes_are_extracted_from_bash_calls_only() {
        let messages = vec![
            Message::assistant(vec![
                ContentBlock::tool_use("1", "bash", json!({ "command": "echo x > a.txt" })),
                ContentBlock::tool_use("2", "read", json!({ "path": "b.txt" })),
            ]),
            Message::assistant(vec![ContentBlock::tool_use(
                "3",
                "bash",
                json!({ "command": "sed -i 's/x/y/' a.txt c.py" }),
            )]),
        ];
        assert_eq!(extract_bash_writes(&messages), ["a.txt", "c.py"]);
    }

    #[test]
    fn growth_per_turn_averages_the_recent_turns() {
        let mut messages = vec![Message::user("x".repeat(40_000))];
        for i in 0..4 {
            messages.push(Message::assistant(vec![ContentBlock::text(
                "a".repeat(400),
            )]));
            messages.push(Message::user(format!("{i}{}", "u".repeat(3_599))));
        }
        // Each turn is ~1,000 tokens; the 10K-token task before the window is not growth.
        assert_eq!(growth_per_turn(&messages, 2), 1_000);
        assert_eq!(
            growth_per_turn(&messages, 10),
            1_000,
            "fewer turns than asked: all of them"
        );
        assert_eq!(growth_per_turn(&[Message::user("x")], 10), 1);
    }

    #[test]
    fn mask_net_saving_matches_the_eval_passes() {
        // s500 pallets-parsers Sonnet, turn 75: 63,819 freed, 102,647 rewritten, refilled in ~43
        // turns; it paid (45 turns followed). Turn 198: 33,302 freed, 148,989 rewritten, ~17 turns
        // to refill; it did not (19 turns followed, break-even 51).
        assert!(mask_net_saving(63_819, 102_647, 63_819 / 43, 1.25) > 0.0);
        assert!(mask_net_saving(33_302, 148_989, 33_302 / 17, 1.25) < 0.0);
        // The 1-hour TTL's 2× write moves the line: 15 turns of 0.1 × 30K saved (45K) pays for a 30K
        // rewrite at 1.25× (34.5K extra) but not at 2× (57K).
        assert!(mask_net_saving(30_000, 30_000, 2_000, 1.25) > 0.0);
        assert!(mask_net_saving(30_000, 30_000, 2_000, 2.0) < 0.0);
    }

    #[test]
    fn best_mask_start_skips_an_early_result_that_forces_a_large_rewrite() {
        // An old 1.2K-char result, then a large protected one (a file being edited), then a large
        // maskable one, then the kept window. Masking from the first result rewrites the protected
        // one too, for ~300 tokens more freed; starting at the large result does not.
        let mut messages = vec![Message::user("the task")];
        let call = |messages: &mut Vec<Message>, id: &str, path: &str, chars: usize| {
            messages.push(Message::assistant(vec![ContentBlock::tool_use(
                id,
                "read",
                json!({ "path": path }),
            )]));
            messages.push(Message::tool_result(id, "r".repeat(chars), false));
        };
        call(&mut messages, "a", "small.py", 1_200);
        call(&mut messages, "b", "edited.py", 40_000);
        call(&mut messages, "c", "big.py", 40_000);
        for i in 0..2 {
            call(&mut messages, &format!("k{i}"), "recent.py", 400);
        }
        let plan = mask_tool_results(&messages, 2, u32::MAX, &["edited.py".into()]).unwrap();
        assert_eq!(
            plan.edited,
            [2, 6],
            "the full plan masks the small and the big result"
        );
        let best = best_mask_start(&messages, &plan, 1_000, 2_000, 1.25).unwrap();
        assert_eq!(best.edited, [6]);
        assert_eq!(best.masked, 1);
        assert!(best.freed > 9_000 && best.freed < plan.freed);
        assert_eq!(
            best.messages[2], messages[2],
            "the early result is left alone"
        );
        // Nothing pays: no plan at all.
        assert!(best_mask_start(&messages, &plan, 1_000, 1_000_000, 1.25).is_none());
        // A floor no start reaches: none either.
        assert!(best_mask_start(&messages, &plan, 1_000_000, 2_000, 1.25).is_none());
    }

    #[test]
    fn head_tail_keeps_a_tally_printed_past_the_head_cut() {
        let mut out = String::from("collecting ... collected 300 items\n");
        for i in 0..200 {
            out.push_str(&format!("tests/test_x.py::test_{i} PASSED\n"));
        }
        out.push_str("FAILED tests/test_x.py::test_7 - AssertionError: boom\n");
        for i in 0..200 {
            out.push_str(&format!("tests/test_y.py::test_{i} PASSED\n"));
        }
        out.push_str("=== 1 failed, 399 passed in 2.31s ===\n");
        assert!(out.len() > 4 * TOOL_RESULT_MAX_CHARS);
        // The head cut loses both the failure and the tally.
        let head = truncate_chars(&out, TOOL_RESULT_MAX_CHARS);
        assert!(!head.contains("1 failed") && !head.contains("test_7 -"));
        let kept = head_tail_chars(&out);
        assert!(kept.starts_with("collecting ... collected 300 items"));
        assert!(kept.contains("1 failed, 399 passed"), "the tail");
        assert!(
            kept.contains("FAILED tests/test_x.py::test_7 - AssertionError: boom"),
            "a failure line from the elided middle"
        );
        assert!(kept.contains("characters elided"));
        assert!(kept.chars().count() < TOOL_RESULT_MAX_CHARS + 600);
        // A short result is left alone.
        assert_eq!(head_tail_chars("3 passed"), "3 passed");
    }

    #[test]
    fn head_tail_bounds_the_signal_lines_it_keeps() {
        let mut out = "x".repeat(600);
        out.push('\n');
        for i in 0..100 {
            out.push_str(&format!("FAILED t.py::test_{i} - Error\n"));
        }
        out.push_str(&"y".repeat(1_600));
        let kept = head_tail_chars(&out);
        assert!(kept.contains("FAILED t.py::test_0 "));
        assert!(!kept.contains("FAILED t.py::test_99 "));
        assert!(kept.contains("[+70 more matching lines]"), "{kept}");
    }

    #[test]
    fn tally_lines_are_recognized_across_runners() {
        for line in [
            "=== 3 failed, 120 passed, 2 skipped in 2.10s ===",
            "test result: ok. 12 passed; 0 failed; 0 ignored",
            "Ran 45 tests in 0.3s",
            "FAILED (failures=2, errors=1)",
            "1 error in 0.5s",
        ] {
            assert!(is_tally_line(line), "{line}");
        }
        for line in [
            "passed the value",
            "v2 passedthrough",
            "line 12 of 40",
            "abc3 failed",
        ] {
            assert!(!is_tally_line(line), "{line}");
        }
        assert_eq!(
            tally_pairs("3 failed, 120 passed, 4 errors in 2s"),
            [("3", "failed"), ("120", "passed"), ("4", "errors")]
        );
    }

    #[test]
    fn the_head_tail_candidate_reaches_the_re_render_only_when_named() {
        let long = format!("{}\n=== 2 failed, 98 passed ===", "z".repeat(5_000));
        let prefix = vec![
            Message::user("the task"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "bash",
                json!({ "command": "pytest" }),
            )]),
            Message::tool_result("1", long, false),
        ];
        let text = |c: &CompactionCandidates| {
            let req = summary_request_with(
                "m",
                &prefix,
                &[],
                100,
                &CompactionProvenance::default(),
                None,
                c,
            );
            let ContentBlock::Text { text, .. } = &req.messages[0].content[0] else {
                panic!("text");
            };
            text.to_string()
        };
        assert!(!text(&CompactionCandidates::default()).contains("2 failed, 98 passed"));
        let on = CompactionCandidates {
            head_tail_results: true,
            ..Default::default()
        };
        assert!(text(&on).contains("2 failed, 98 passed"));
    }
}
