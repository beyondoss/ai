//! The agent loop — Pi's `pi-agent-core` runtime, ported.
//!
//! One iteration = one model turn: stream a completion, assemble the assistant message from the
//! event stream, append it to the session, and — if the model asked for tools — run each tool and
//! feed the results back as a new user turn. Repeat until the model ends its turn (or `max_steps`).
//!
//! The loop is dialect-blind (both wire dialects normalize to the same `StreamEvent` sequence) and
//! network-blind (it depends only on [`ModelTransport`], so tests drive it with `MockTransport`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::future::{Either, select};
use serde::Serialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::compaction::{self, CompactionConfig, CompactionReason};
use crate::error::{Error, MID_STREAM_NETWORK_ERROR, Result};
use crate::hooks::{AgentHooks, CheckpointHook, NoCheckpoint, NoHooks};
use crate::message::{
    ContentBlock, ImageSource, Message, Role, StopReason, StreamEvent, TokenUsage, ToolDef,
};
use crate::session::Session;
use crate::steering::Steering;
use crate::tool::ToolRegistry;
use crate::transport::{ModelRequest, ModelTransport, ReasoningEffort};

/// An observable event from a run: a streamed model event, a tool-invocation boundary, or a turn
/// boundary. The headless server serializes these to its clients; [`Agent::run`] exposes only the
/// `Stream` events.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentEvent {
    /// The run has begun (before the first turn).
    AgentStart,
    /// A model turn is about to start.
    TurnStart { step: u32 },
    /// A streamed model event (text/tool deltas, usage, stop).
    Stream(StreamEvent),
    /// A tool is about to run, with the arguments the model supplied.
    ToolStart {
        id: String,
        name: String,
        input: Value,
    },
    /// A progress snapshot from a still-running tool (pi's `tool_execution_update`): the full output so
    /// far (not a delta) plus optional tool-specific `details`. Emitted via the tool's
    /// [`ToolProgress`](crate::ToolProgress) sink, before the tool's `ToolEnd`.
    ToolProgress {
        id: String,
        name: String,
        snapshot: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
    },
    /// A tool finished (or errored); `result` is what's fed back to the model.
    ToolEnd {
        id: String,
        name: String,
        result: String,
        is_error: bool,
    },
    /// One model turn completed.
    TurnEnd { stop_reason: StopReason, step: u32 },
    /// Queued steering/follow-up messages were injected at a stop boundary; the run continues.
    Steered { messages: usize },
    /// The run finished normally (the model ended its turn and no steering was queued).
    AgentEnd { steps: u32 },
    /// A compaction round has begun — [`Agent::compact`] confirmed a worthwhile prefix exists and is
    /// about to make its first summarization model call. Emitted before [`Compacted`](Self::Compacted),
    /// which reports the same round's outcome once it finishes; a caller tracking "is a compaction
    /// currently in flight" (pi's `isCompacting`) sets a flag on this event and clears it on `Compacted`.
    CompactionStart { reason: CompactionReason },
    /// The conversation prefix was summarized to stay under the context window.
    Compacted {
        messages_before: usize,
        messages_after: usize,
        /// Why this compaction fired — the full folded-forward provenance (file-ops, round count)
        /// lands on `Session::compaction`, not duplicated onto every event.
        reason: CompactionReason,
        /// Estimated input tokens at the moment this compaction fired (before the reset).
        tokens_before: u32,
        /// The generated summary text actually spliced into the session (with the deterministic
        /// file-operations list already appended — see `compaction::format_file_operations`) — pi's own
        /// `CompactionResult.summary`. A caller (the `compact` RPC, a `run --json` client) that wants to
        /// show the operator what got summarized would otherwise have no way to see it at all.
        summary: String,
        /// Estimated total tokens across the post-compaction message list (summary + kept suffix) — pi's
        /// own `CompactionResult.estimatedTokensAfter`, computed the same way (`estimate_messages_tokens`
        /// summed over every message), not `trailing_tokens`'s since-last-snapshot delta, which is
        /// always 0 immediately after `apply_summary` resets that snapshot to point past the rebuilt
        /// list's end.
        tokens_after: u32,
        /// The pre-compaction index (into the *old* `session.messages`, before [`compaction::apply_summary`]
        /// spliced the summary in) of the first message this round kept rather than folded away — pi's own
        /// `CompactionResult.firstKeptEntryId`, minus the id translation: this crate's tree/entry-id layer
        /// lives one level up in `crates/agent` (`SessionStore`), which this crate has no visibility into,
        /// so this carries the same cut point as a plain message index instead. A caller that also tracks
        /// entry ids (the `compact` RPC) resolves this against its own pre-compaction active-path id list
        /// (parallel to `session.messages` by construction) to recover the actual id pi's field names.
        first_kept: usize,
    },
    /// The run is ending abnormally (transport failure after retries, malformed SSE, or the step
    /// ceiling). A terminal marker on the event stream so a streaming client sees *why* a run stopped
    /// rather than the stream just going silent; `run_events` still returns the same `Err`.
    Error { message: String },
    /// An automatic compaction attempt (proactive threshold or hard-overflow backstop) failed —
    /// non-terminal, unlike `Error`. The run continues into the turn that was about to be sent
    /// unsummarized; a client tracking `isCompacting` should clear it on this event exactly like on
    /// `Compacted`. Matches pi's `compaction_end { errorMessage }` on the same failure path: a
    /// transient summarization-call failure (network blip, provider hiccup) must not make an
    /// otherwise-servable turn unreachable, since `should_compact` would otherwise re-trigger and
    /// re-fail before every future turn, permanently blocking the session.
    CompactionFailed {
        reason: CompactionReason,
        message: String,
    },
    /// A pending [`crate::steering::ModelSwitch`] (see [`Steering::request_model_switch`]) was applied
    /// at a turn boundary — every subsequent turn of this run now targets `model`, with `thinking` set
    /// if the switch requested a new budget. Mirrors pi's `prepareNextTurn`: a host can downgrade to a
    /// cheaper model once a run turns out not to need much firepower, or raise it (or the thinking
    /// budget) once it turns out to need more, without stopping and restarting the whole call.
    ModelSwitched {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking: Option<u32>,
    },
    /// A pending [`crate::steering::Steering::request_tool_set`] was applied at a turn boundary (Task
    /// #13, pi-parity) — every subsequent turn of this run now advertises the tools in `tool_names`
    /// (sorted, matching [`crate::tool::ToolRegistry::definitions`]'s own order) instead of whatever the
    /// `Agent` was originally configured with. Mirrors the underlying mechanism pi's real shipped
    /// product uses for the same feature — `packages/coding-agent/src/core/agent-session.ts`'s
    /// `setActiveToolsByName` (defined at line 840; called from the extension runtime's `setActiveTools`
    /// handler at line 2283, and internally at line 2428), which likewise reconfigures a run's tool set
    /// mid-flight, taking effect starting the next turn. pi's own `setActiveToolsByName` has no discrete
    /// event of its own, though — this explicit `ToolsUpdated` event (so a streaming client can observe
    /// the change) is this crate's own addition. The `tools_update` event *name* traces to the dead,
    /// unshipped `packages/agent/src/harness/agent-harness.ts` (zero references from
    /// `packages/coding-agent`), not to a real pi event.
    ToolsUpdated { tool_names: Vec<String> },
}

/// Default per-turn output token ceiling.
const DEFAULT_MAX_TOKENS: u32 = 4096;
/// Default ceiling on loop iterations before bailing — a runaway-tool-call backstop.
///
/// Deliberately kept hard and fatal, not just raised or removed: this agent can run unattended
/// (`serve`, a homelab automation), with no human approving each tool call, and this repo's own
/// operating principle is that automated actions stay bounded and safe to interrupt (see the root
/// CLAUDE.md's "operations must be idempotent and atomic"). A bounded blast radius on a runaway loop
/// is a feature this ceiling provides, not a limitation to engineer away. `Error::MaxSteps` is fully
/// resumable, though — the check runs before any per-turn state is touched, so a client that hits it
/// can simply issue another `prompt` to continue past it with a fresh budget. 50 (up from an earlier
/// 24) gives a legitimate multi-file task more headroom before that ceiling interrupts it; operators
/// can still override it per deployment via [`Agent::with_max_steps`]. `pub` so a CLI's own flag
/// default (`agent run --max-steps`/`agent serve --max-steps`) can reference this one number instead
/// of carrying a second, driftable copy of it.
pub const DEFAULT_MAX_STEPS: u32 = 50;
/// Cap on tool-call groups dispatched concurrently within one turn. A model usually batches a
/// handful, but nothing bounds how many it requests; without a cap a turn asking for dozens of
/// `bash`/`grep` calls would spawn that many subprocesses / parallel walks at once (and `grep` itself
/// fans out over CPU cores, compounding it). The cap throttles in-flight groups; results scatter by
/// index, so the call-order transcript is unaffected — only peak concurrency is bounded.
const MAX_CONCURRENT_TOOL_GROUPS: usize = 8;
/// One tool call's dispatch outcome: `(text, images, is_error, terminate)`. Hooks rewrite the text and
/// error flag; images and the terminate hint pass through untouched.
type ToolCallResult = (String, Vec<ImageSource>, bool, bool);
/// How many times to restart a turn whose stream dies *after* the request already succeeded — a
/// truncated stream (dropped connection, gateway cut) or an in-band `overloaded_error` event. Distinct
/// from `client.rs`'s retry, which only covers a failure *before* the first byte; once events start
/// flowing that layer deliberately stops retrying (a mid-stream drop there would replay partial
/// output), so this is the only place that can recover from this failure class today.
const MAX_MID_STREAM_RETRIES: u32 = 3;
/// Base of the exponential backoff between mid-stream retries (`BASE · 2^(attempt-1)`). Mirrors
/// `client.rs`'s shape but kept separate — a different layer, a different failure class, no shared state.
const MID_STREAM_BASE_BACKOFF: Duration = Duration::from_millis(250);
/// Ceiling on a single mid-stream retry wait.
const MID_STREAM_MAX_BACKOFF: Duration = Duration::from_secs(5);

/// Outcome of a manual or automatic compaction attempt (see [`Agent::compact`]) — distinguishes an
/// actual compaction from the two distinct reasons nothing happened, so a headless caller (`serve`'s
/// `compact` RPC) can report *why* instead of collapsing both into the same bare `false` pi's own two
/// distinct thrown errors ("Nothing to compact (session too small)" / "Already compacted",
/// `agent-session.ts::compact`) never actually are. Track (pi-parity fix, Task #26).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactOutcome {
    /// A worthwhile prefix was found and folded into a fresh summary.
    Compacted,
    /// No cut point exists at all yet ([`compaction::find_split_cut`] returned `None`) — too few
    /// messages, or no clean assistant boundary to summarize up to. Matches pi's "Nothing to compact
    /// (session too small)".
    TooSmall,
    /// A cut point exists, but there's nothing new worth a fresh summary: either a prior summary
    /// already covers everything up to a clean boundary and nothing since has grown past
    /// `keep_recent_tokens`, or (rare) a genuine summarization call ran and the model's own response
    /// came back blank. Matches pi's "Already compacted".
    AlreadyCompacted,
}

impl CompactOutcome {
    /// `true` only for [`Self::Compacted`] — the bare bool this type replaces. What every caller that
    /// only cares *whether* room was freed (not why not) actually wants: `serve`'s `compact` RPC
    /// response's own `compacted` field, and the overflow-retry call sites in [`Agent::run_events`]/
    /// turn handling below, which only ever branch on "did this recover room."
    pub fn compacted(self) -> bool {
        matches!(self, Self::Compacted)
    }

    /// A stable wire/log discriminator for the no-op reason — `None` when a real compaction happened.
    pub fn reason(self) -> Option<&'static str> {
        match self {
            Self::Compacted => None,
            Self::TooSmall => Some("too_small"),
            Self::AlreadyCompacted => Some("already_compacted"),
        }
    }
}

/// A configured agent: a model, a transport, a tool set, and loop bounds. Cheap to clone-construct;
/// `run` borrows it so one agent can drive many sessions.
pub struct Agent {
    transport: Arc<dyn ModelTransport>,
    tools: ToolRegistry,
    /// The advertised tool definitions, computed once from `tools` (and again on a
    /// `Steering::request_tool_set` switch — see [`Self::run_events_steered`]'s `current_tool_defs`
    /// shadow). This `Agent`'s own baseline is still fixed once configured, so a fresh run without a
    /// mid-run switch never rebuilds it (or its JSON schemas) per turn; each request clones the `Arc`,
    /// not the definitions.
    tool_defs: Arc<[ToolDef]>,
    model: String,
    system: Option<String>,
    /// A per-turn system-prompt override, re-evaluated at the same point `system` would otherwise be
    /// read — see [`Self::with_system_fn`]. Takes priority over `system` when set (mirrors pi's
    /// function-valued `systemPrompt`, which is one field, either a string or a callback — not both
    /// composed together).
    system_fn: Option<Box<dyn Fn() -> String + Send + Sync>>,
    max_tokens: u32,
    max_steps: u32,
    /// Extended-thinking budget, when enabled. Applied to every turn's request.
    thinking: Option<u32>,
    /// Reasoning effort level (OpenAI reasoning models; Anthropic adaptive thinking). Applied to every
    /// turn's request when set.
    reasoning_effort: Option<ReasoningEffort>,
    /// Sampling temperature. `None` leaves the provider default. Applied to every turn's request when
    /// set — see [`ModelRequest::temperature`]'s doc comment for per-dialect gating.
    temperature: Option<f64>,
    /// Context-compaction policy: when to summarize the prefix to stay under the context window.
    compaction: CompactionConfig,
    /// Whether [`Self::run_turn`] retries a mid-stream transport failure (see
    /// [`is_retryable_mid_stream`]) instead of surfacing it immediately. Defaults to `true`; an
    /// operator debugging a flaky network hop can disable it via `with_auto_retry(false)` to see the
    /// raw failure on the very first hiccup rather than after `MAX_MID_STREAM_RETRIES` silent attempts.
    auto_retry: bool,
    /// Force every tool-call group in a turn to run one at a time, regardless of
    /// [`MAX_CONCURRENT_TOOL_GROUPS`] — the same effect `exclusive_turn` already has for a single
    /// `bash`-like call, but host-selectable for a whole run. Defaults to `false` (concurrent, bounded).
    /// Matches pi's own `AgentOptions.toolExecution: "sequential" | "parallel"` — see
    /// [`Self::with_sequential_tools`].
    sequential_tools: bool,
    /// Interception hooks around tool calls (gate/rewrite). Defaults to no-ops.
    hooks: Arc<dyn AgentHooks>,
    /// Stable prompt-cache affinity key for this run (OpenAI `prompt_cache_key`).
    cache_key: Option<String>,
    /// Use the 1-hour prompt-cache TTL (Anthropic) instead of the default 5 minutes.
    cache_long: bool,
    /// Cross-turn, cross-run file-mutation exclusivity. Defaults to a registry private to this
    /// `Agent`; share one `Arc` across multiple `Agent`s (e.g. one per session in a `serve` process)
    /// to extend the guarantee across concurrently-running sessions too.
    write_locks: Arc<crate::write_lock::WriteLockRegistry>,
    /// Called at each durable mid-run checkpoint (see [`CheckpointHook`]). Defaults to a no-op, so a
    /// caller that only ever persists once a full run completes (or doesn't persist at all) pays
    /// nothing extra.
    checkpoint: Arc<dyn CheckpointHook>,
    /// Independent input-token reserve for [`Self::summarize_branch`]'s own summarization call —
    /// `None` (the default) falls back to `self.compaction.reserve_tokens`, today's exact behavior.
    /// `Some(tokens)` sizes a branch recap's budget off this instead, without disturbing the live
    /// conversation's own compaction reserve — pi exposes the same independent knob
    /// (`branch-summarization.ts:62-63`, default 16384) rather than reusing its compaction settings
    /// wholesale. See [`Self::with_branch_summary_reserve_tokens`].
    branch_summary_reserve_tokens: Option<u32>,
    /// Operator-facing override (Task #26, pi-parity) that forces the vision-downgrade path — the
    /// `_model_supports_vision` flag dispatch stamps onto every tool call's coerced input (see the
    /// gate loop in [`Self::run_events_steered`]) — to report `false` regardless of whether the active
    /// model actually supports vision. Defaults to `false` (report the model's real capability
    /// untouched). This is the mechanism a host's own `--block-images`-style flag wires into: an
    /// operator who wants every image downgraded to a text placeholder unconditionally (a
    /// cost/bandwidth policy, a text-only audit log) doesn't need a fake non-vision model id to get it.
    /// See [`Self::with_block_images`].
    block_images: bool,
}

/// The summarization call's own output budget, scaled from the model's real ceiling rather than a
/// flat constant regardless of what the model can actually hold: `0.8 * reserve_tokens` mirrors how
/// much headroom compaction is trying to buy back, capped at `max_output` so this can never exceed
/// what the model would reject, and floored so a very small `reserve_tokens` still leaves the summary
/// usably long. Shared by [`Agent::new`] (the initial seed) and [`Agent::with_compaction`] (rescaling
/// when a caller replaces the whole config, e.g. to override `reserve_tokens` alone).
fn scaled_summary_max_tokens(reserve_tokens: u32, max_output: u32) -> u32 {
    (((reserve_tokens as f64) * 0.8) as u32)
        .min(max_output)
        .max(1024)
}

impl Agent {
    /// An agent over `transport` using `model`, with no tools and model-aware defaults: the per-turn
    /// output ceiling and the compaction context window are seeded from the model's
    /// [`capabilities`](crate::models::capabilities) (both still overridable via the builders).
    pub fn new(transport: Arc<dyn ModelTransport>, model: impl Into<String>) -> Self {
        let model = model.into();
        let caps = crate::models::capabilities(&model);
        // Both compaction budgets scale with the model's real window (see `for_window`) — reading the
        // reserve back off the *scaled* config rather than off `Default` matters here, because
        // `scaled_summary_max_tokens` sizes the summarization call against the headroom that actually
        // exists on this model, not against a 200k model's.
        let compaction = CompactionConfig::for_window(caps.context_window);
        let summary_max_tokens =
            scaled_summary_max_tokens(compaction.reserve_tokens, caps.max_output);
        Self {
            transport,
            tools: ToolRegistry::new(),
            tool_defs: Vec::new().into(),
            model,
            system: None,
            system_fn: None,
            max_tokens: caps.max_output.max(DEFAULT_MAX_TOKENS),
            max_steps: DEFAULT_MAX_STEPS,
            thinking: None,
            reasoning_effort: None,
            temperature: None,
            compaction: CompactionConfig {
                summary_max_tokens,
                ..compaction
            },
            auto_retry: true,
            sequential_tools: false,
            hooks: Arc::new(NoHooks),
            cache_key: None,
            cache_long: false,
            write_locks: Arc::new(crate::write_lock::WriteLockRegistry::new()),
            checkpoint: Arc::new(NoCheckpoint),
            branch_summary_reserve_tokens: None,
            block_images: false,
        }
    }

    /// Set the tools the model may call. The advertised definitions are computed here, once, so the
    /// loop doesn't rebuild them (and their JSON schemas) every turn.
    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tool_defs = tools.definitions().into();
        self.tools = tools;
        self
    }

    /// Set the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Replace the system prompt in place, without rebuilding tools/compaction/cache config — the cheap
    /// per-turn refresh a caller uses to keep the time-varying part of the prompt (the current date) up
    /// to date without paying for a full `Agent` rebuild every turn.
    ///
    /// Requires `&mut self`, so it's unusable while [`Self::run_events_steered`] holds `&self` for the
    /// whole span of a long-running call — see [`Self::with_system_fn`] for the per-turn alternative
    /// that works from inside a run already in flight.
    pub fn set_system(&mut self, system: impl Into<String>) {
        self.system = Some(system.into());
    }

    /// Install a per-turn system-prompt callback, consulted fresh at the same point [`with_system`]'s
    /// static string would otherwise be read — every turn of every run, not just once at construction.
    /// Takes priority over [`with_system`](Self::with_system) when both are set (mirrors pi's
    /// function-valued `systemPrompt`, `harness/types.ts:817-826`, itself re-evaluated every turn via
    /// `createTurnState()` — one field, either a string or a callback, not both composed together).
    ///
    /// Unlike [`set_system`](Self::set_system), this works from *inside* an in-flight
    /// [`run_events_steered`](Self::run_events_steered) call: that method holds `&self` for the whole
    /// run, so a long-running call could never refresh e.g. a date-stamped system prompt turn-to-turn
    /// through `set_system`'s `&mut self` — a callback installed here is called through `&self`
    /// instead, so it can be evaluated fresh every turn without needing exclusive access to the `Agent`
    /// at all (a `Fn`, not `FnMut` — the callback closes over its own interior-mutable state, e.g. an
    /// `Arc<Mutex<..>>` or reading the wall clock directly, rather than relying on the `Agent` itself to
    /// carry mutable per-call state).
    pub fn with_system_fn(
        mut self,
        system_fn: impl Fn() -> String + Send + Sync + 'static,
    ) -> Self {
        self.system_fn = Some(Box::new(system_fn));
        self
    }

    /// Set the per-turn output token ceiling.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the loop-iteration ceiling.
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Enable extended thinking with the given token budget on every turn. The budget must be below
    /// [`with_max_tokens`](Self::with_max_tokens) (Anthropic requires `max_tokens > budget_tokens`).
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(budget_tokens);
        self
    }

    /// Set the reasoning effort level applied to every turn (OpenAI reasoning models; Anthropic
    /// adaptive thinking). Independent of [`with_thinking`](Self::with_thinking)'s token budget.
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set the sampling temperature applied to every turn. See [`ModelRequest::temperature`]'s doc
    /// comment for per-dialect gating (Anthropic omits it while thinking is enabled).
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the compaction policy (context window, reserve, keep-recent, enabled).
    pub fn with_compaction(mut self, mut compaction: CompactionConfig) -> Self {
        // A caller replacing the whole config wholesale — the common pattern for overriding just
        // `reserve_tokens`/`enabled`/`context_window` via struct-update syntax against
        // `CompactionConfig::default()` — would otherwise silently discard `Agent::new()`'s
        // model-aware `summary_max_tokens` scaling and fall back to the struct's flat default, however
        // poor a fit that is for this model's real `max_output`. Detected by the incoming value still
        // being exactly that flat default (i.e. not something the caller deliberately chose): rescale
        // it the same way `Agent::new()` seeds it initially, against *this* config's `reserve_tokens`.
        // A caller on a model whose `max_output` genuinely caps out at (or below) the flat default gets
        // that same value back either way, so this is never a regression for them — only ever a fix for
        // every model with more headroom to give.
        if compaction.summary_max_tokens == CompactionConfig::default().summary_max_tokens {
            let caps = crate::models::capabilities(&self.model);
            compaction.summary_max_tokens =
                scaled_summary_max_tokens(compaction.reserve_tokens, caps.max_output);
        }
        self.compaction = compaction;
        self
    }

    /// Convenience: set just the model's context window, leaving the other compaction defaults.
    pub fn with_context_window(mut self, context_window: u32) -> Self {
        self.compaction.context_window = context_window;
        self
    }

    /// Size [`Self::summarize_branch`]'s own input-token budget off `reserve_tokens` instead of
    /// falling back to [`Self::with_compaction`]'s `reserve_tokens` — a pure additive capability: a
    /// caller who never touches this keeps today's exact behavior (the live conversation's own
    /// compaction reserve reused wholesale for a one-off branch recap too), while a caller who wants
    /// the two tuned independently (matching pi's own separate `reserveTokens` for branch
    /// summarization) now can.
    pub fn with_branch_summary_reserve_tokens(mut self, reserve_tokens: u32) -> Self {
        self.branch_summary_reserve_tokens = Some(reserve_tokens);
        self
    }

    /// Enable or disable mid-stream retry (default: enabled) — see the `auto_retry` field's doc comment.
    pub fn with_auto_retry(mut self, enabled: bool) -> Self {
        self.auto_retry = enabled;
        self
    }

    /// Force fully-sequential tool dispatch (default: `false`, bounded-concurrent) — see the
    /// `sequential_tools` field's doc comment.
    pub fn with_sequential_tools(mut self, enabled: bool) -> Self {
        self.sequential_tools = enabled;
        self
    }

    /// Install interception hooks (tool gating / result rewriting). Defaults to no-ops.
    pub fn with_hooks(mut self, hooks: Arc<dyn AgentHooks>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Install a [`CheckpointHook`], called at each durable mid-run point so a caller can persist a
    /// multi-step run incrementally instead of only once it finishes entirely. Defaults to a no-op.
    pub fn with_checkpoint_hook(mut self, checkpoint: Arc<dyn CheckpointHook>) -> Self {
        self.checkpoint = checkpoint;
        self
    }

    /// Share a file-mutation-exclusivity registry across multiple `Agent`s (or across rebuilds of the
    /// same one, e.g. `serve`'s `set_model`/`set_thinking`), so two calls that write the same path
    /// serialize even when they belong to different turns or different concurrently-running sessions.
    /// Defaults to a registry private to this `Agent` — fine for one agent running one session at a
    /// time, which is why most callers never need this builder.
    pub fn with_write_locks(
        mut self,
        write_locks: Arc<crate::write_lock::WriteLockRegistry>,
    ) -> Self {
        self.write_locks = write_locks;
        self
    }

    /// Set a stable prompt-cache affinity key (e.g. the session id) for every turn's request.
    pub fn with_cache_key(mut self, key: impl Into<String>) -> Self {
        self.cache_key = Some(key.into());
        self
    }

    /// Use the 1-hour prompt-cache TTL (Anthropic) instead of the default 5 minutes.
    pub fn with_cache_long(mut self, long: bool) -> Self {
        self.cache_long = long;
        self
    }

    /// Force the vision-downgrade path (images → text placeholder) regardless of whether the active
    /// model actually supports vision — an operator-facing override for a cost/bandwidth policy or a
    /// text-only audit log, independent of the model's real `supports_vision` capability. Defaults to
    /// `false` (report the model's real capability untouched).
    pub fn with_block_images(mut self, block_images: bool) -> Self {
        self.block_images = block_images;
        self
    }

    /// Drive the loop to completion against `session`, invoking `on_event` for every streamed event
    /// (use it to render assistant text/tool activity live). Returns when the model ends its turn
    /// without requesting tools, or errors with [`Error::MaxSteps`] if it never does.
    pub async fn run<F>(&self, session: &mut Session, mut on_event: F) -> Result<()>
    where
        F: FnMut(&StreamEvent) + Send,
    {
        self.run_events(session, move |ev| {
            if let AgentEvent::Stream(s) = &ev {
                on_event(s);
            }
        })
        .await
    }

    /// Like [`run`](Self::run), but a `cancel` token lets a caller interrupt the run — see
    /// [`run_events_cancellable`](Self::run_events_cancellable) for the exact semantics. Returns
    /// [`Error::Cancelled`] once cancelled.
    pub async fn run_cancellable<F>(
        &self,
        session: &mut Session,
        mut on_event: F,
        cancel: CancellationToken,
    ) -> Result<()>
    where
        F: FnMut(&StreamEvent) + Send,
    {
        self.run_events_cancellable(
            session,
            move |ev| {
                if let AgentEvent::Stream(s) = &ev {
                    on_event(s);
                }
            },
            cancel,
        )
        .await
    }

    /// Drive the loop to completion, emitting an [`AgentEvent`] for every streamed model event, tool
    /// invocation, and turn boundary — the full observation surface the headless server streams to
    /// its clients. Returns when the model ends its turn without tools, or [`Error::MaxSteps`].
    pub async fn run_events<F>(&self, session: &mut Session, sink: F) -> Result<()>
    where
        F: FnMut(AgentEvent) + Send,
    {
        // The plain entry point can't be cancelled; hand it a token that is never tripped.
        self.run_events_cancellable(session, sink, CancellationToken::new())
            .await
    }

    /// Like [`run_events`], but a `cancel` token lets a caller interrupt the run: a tripped token
    /// stops the loop between turns, interrupts a model stream mid-flight, and aborts in-progress tool
    /// calls (dropping their futures — which kills a `bash` subprocess and aborts the model's HTTP
    /// request). Returns [`Error::Cancelled`] once cancelled.
    ///
    /// [`run_events`]: Self::run_events
    pub async fn run_events_cancellable<F>(
        &self,
        session: &mut Session,
        sink: F,
        cancel: CancellationToken,
    ) -> Result<()>
    where
        F: FnMut(AgentEvent) + Send,
    {
        self.run_events_steered(session, sink, cancel, Steering::new())
            .await
    }

    /// Like [`run_events_cancellable`], plus a `steering` queue: when the model would otherwise end
    /// the run, any messages a client pushed to `steering` are injected as new user turns and the loop
    /// continues — letting a client redirect or extend a working agent without starting over.
    ///
    /// `steering` also carries a graceful stop request ([`Steering::request_stop`]) — pi's
    /// `shouldStopAfterTurn` equivalent. It's checked at every turn boundary, after that turn's tool
    /// calls (if any) have already run and their results are committed to `session`, but before another
    /// model call would start. Unlike `cancel`, it never abandons a tool mid-execution or leaves an
    /// orphaned `tool_use` behind — the current turn always finishes cleanly first. Whatever happens to
    /// a pending request by the time this call returns (consumed by a turn-boundary check, or never
    /// reached because the run ended some other way — an error, cancellation, or a refusal, which — like
    /// the queue drains — skips it), it's always cleared before returning, so a stop meant for this run
    /// can never bleed into a later, unrelated call sharing the same `Steering` handle.
    ///
    /// [`run_events_cancellable`]: Self::run_events_cancellable
    pub async fn run_events_steered<F>(
        &self,
        session: &mut Session,
        mut sink: F,
        cancel: CancellationToken,
        steering: Steering,
    ) -> Result<()>
    where
        F: FnMut(AgentEvent) + Send,
    {
        // Every other interception point in this file (`before_tool_call`, `after_tool_call`,
        // `on_assistant_message`, `should_stop_after_turn`, `before_provider_request`/
        // `before_provider_payload`, `after_provider_response`) is wrapped in `catch_tool_panic` so a
        // bug in caller-supplied code degrades one call instead of killing the run. `sink` is the same
        // kind of caller-supplied code — this run's whole reason for existing is to report progress to
        // it — so it gets the same fails-open treatment here, once, before the first event goes out:
        // every `sink(...)` call for the rest of this function (and everything it calls transitively,
        // e.g. `emit_tool_update`, `run_tool_calls_interleaved`) goes through this wrapped closure
        // instead of the raw one. `catch_sink_panic` is `catch_tool_panic`'s sync sibling — `sink`
        // itself isn't a future, so there's nothing to `.await` here.
        let mut sink = move |ev: AgentEvent| catch_sink_panic(|| sink(ev));
        // A transcript produced by one model is not automatically replayable to another. Signed
        // `Thinking` blocks and encrypted `RedactedThinking` blocks are bound to the model that made
        // them; OpenAI-Responses combined `"call_id|item_id"` tool ids only pair with a `reasoning` item
        // on that same dialect. Replay them across a switch and the provider rejects the whole request —
        // `run --model gpt-5 …` then `run --continue --model claude-…` 400s with
        // ``Invalid `signature` in `thinking` block``.
        //
        // `Session::scrub_cross_model_state` was written for exactly this and, until now, was called by
        // nothing but its own tests. It belongs *here* rather than in each caller that can change the
        // active model (`run --continue`, and `serve`'s `set_model`/`cycle_model`/`switch_session`/
        // `fork`/`clone`/`switch_branch`): every one of them funnels into this function, and this is the
        // single point where a model and the transcript it is about to be shown actually meet. At the
        // choke point it is structurally impossible for a new model-switching entry point to forget —
        // which is exactly how it ended up with no callers at all.
        //
        // Gated on a read-only scan: the scrub `Arc::make_mut`s the message vec, deep-cloning the whole
        // transcript whenever that `Arc` is shared (it is). Same-model turns — nearly all of them — must
        // not pay that to change nothing.
        if session.needs_cross_model_scrub(&self.model) {
            tracing::debug!(
                model = %self.model,
                "resuming on a different model: scrubbing thinking blocks and foreign tool-call ids"
            );
            session.scrub_cross_model_state(&self.model);
        }
        // Clears any pending stop request when this call returns, by whatever path — normal
        // completion, an early `?`/`return Err`, cancellation, or a refusal — so a request this call
        // never got around to consuming can't leak into a later call that reuses `steering`.
        struct ClearStopOnDrop(Steering);
        impl Drop for ClearStopOnDrop {
            fn drop(&mut self) {
                self.0.take_stop_requested();
            }
        }
        let _clear_stop_on_drop = ClearStopOnDrop(steering.clone());
        sink(AgentEvent::AgentStart);
        // Drain the next-turn lane (a beyond-only capability — see `steering.rs`'s own module doc
        // comment for why it has no real pi-product equivalent) before this run's very first request is
        // built — a message queued via `Steering::push_next_turn` while the agent was idle (no run in
        // flight for `push`/`push_steer` to attach to) must still land on turn 1 of whatever prompt
        // comes next, not just eventually after a tool round-trip. Folded as leading content blocks on
        // the same user turn the caller already pushed before calling this function, rather than as
        // separate messages — inserting a *separate* user message here would land it right next to the
        // caller's own user turn, a same-role pair no wire dialect accepts (`drain_steer`'s own
        // "fold into the adjacent message" pattern, applied here instead of appending after).
        let next_turn_queued = steering.drain_next_turn();
        if !next_turn_queued.is_empty() {
            let fold_into_prompt = session
                .messages
                .last()
                .is_some_and(|m| m.role == Role::User);
            if fold_into_prompt {
                let mut prefix = Vec::with_capacity(next_turn_queued.len());
                for msg in next_turn_queued {
                    prefix.push(ContentBlock::text(msg.text));
                    prefix.extend(
                        msg.images
                            .into_iter()
                            .map(|source| ContentBlock::Image { source }),
                    );
                }
                // `fold_into_prompt` already established `last_mut()` is `Some`.
                if let Some(last) = Arc::make_mut(&mut session.messages).last_mut() {
                    prefix.append(&mut last.content);
                    last.content = prefix;
                }
            } else {
                // No trailing user turn to fold onto (e.g. an empty session) — queue each as its own
                // fresh user turn instead of losing it.
                for msg in next_turn_queued {
                    if msg.images.is_empty() {
                        session.user(msg.text);
                    } else {
                        session.push(Message::user_with_images(msg.text, msg.images));
                    }
                }
            }
        }
        // The user's own prompt (already pushed by the caller before this call, per every call site's
        // own convention — see `session.user(...)` throughout this module's tests) — plus any
        // next-turn message just folded onto it above — is durable now, before the model is ever
        // called: a crash here must not lose it, exactly the exposure `CheckpointHook`'s own doc
        // comment used to call out (the very first turn of a run was the one point a crash lost the
        // user's own submitted prompt entirely).
        self.checkpoint_guarded(session).await;
        // Set once we've already compacted to recover from a context-overflow *error* this turn, so a
        // second overflow gives up instead of looping. Reset after each turn that lands cleanly — which
        // is sound for this flag precisely because both arms that read it (`Err(e) if
        // is_context_overflow(..)`) are evaluated inside the `match` on the turn's result, i.e. strictly
        // *before* the `Ok` path reaches that reset.
        let mut overflow_recovered = false;
        // The same idea for the *silent-truncation* recovery path (a `MaxTokens` stop that
        // `is_hard_overflow` recognizes), which needs its own flag rather than sharing the one above:
        // that check lives on the `Ok` path, *downstream* of the reset, so a shared flag is always false
        // by the time it's read and the guard it's supposed to provide can never fire. Read via
        // `mem::replace` into a per-turn local instead (see the take below), which gives it a
        // survives-exactly-one-iteration lifetime that no reset ordering can defeat.
        let mut truncation_recovered = false;
        // Steps taken *by this call*, distinct from `session.steps` (a lifetime total across every
        // call, used only for observability — the `step` field on emitted events). Checking the
        // ceiling against this instead means `Error::MaxSteps` is a per-call backstop a client can
        // resume past by simply calling `run`/`run_events_steered` again with a fresh budget, rather
        // than a permanent dead end once the session's lifetime total crosses it once.
        let mut steps_this_call: u32 = 0;
        // The previous turn's stop reason, read by `is_hard_overflow`'s `MaxTokens` check. Re-derived
        // from `session`'s own last assistant message rather than hardcoded to `EndTurn` (pi-parity fix
        // — matches pi's own `isContextOverflow`, re-evaluated fresh from the persisted
        // `AssistantMessage.stopReason` on every `prompt()` call): a `MaxTokens` turn a *prior*
        // run/process left as the session's last assistant message — nothing in this call has produced
        // a turn yet — must still be visible to the very first top-of-loop hard-overflow check below,
        // not just to a run that happens to still be in flight when it fires. `EndTurn` (a value that
        // check never matches) when there's no assistant message yet, or the last one predates this
        // field.
        let mut last_stop_reason = session
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .and_then(|m| m.stop_reason)
            .unwrap_or_default();
        // Mutable, per-call shadows of the `Agent`'s own model/thinking/reasoning-effort — `self` is
        // borrowed immutably for the whole call, so a mid-run switch (see below) can't touch `self`
        // directly; every read of "what model/thinking is this run using right now" goes through these
        // instead of `self.model`/`self.thinking`/`self.reasoning_effort` for the rest of this function.
        // Deliberately NOT threaded into `compact`/`compact_or_report`/`is_hard_overflow`: those still
        // use the `Agent`'s original, as-configured model — pi's own harness doesn't thread
        // `prepareNextTurn`'s override into its summarization path either (a separate call with no
        // `nextTurnSnapshot` awareness).
        let mut current_model = self.model.clone();
        let mut current_thinking = self.thinking;
        // Mutated by a mid-run switch that carries a `thinking_level` override (see below) —
        // `ModelSwitch` used to only ever carry a model/raw-thinking-budget override, but now also
        // covers reasoning effort, so this can no longer stay a plain, never-touched shadow.
        let mut current_reasoning_effort = self.reasoning_effort;
        // Task #13 (pi-parity): a per-call shadow of the `Agent`'s own tool set, mirroring
        // `current_model`'s own rationale — `self` stays borrowed immutably for the whole call, so a
        // mid-run `Steering::request_tool_set` (pi's `setTools`/`setActiveTools`) can't touch
        // `self.tools`/`self.tool_defs` directly. Applied at the same turn boundary a model switch is
        // (see below), so a change takes effect starting the very next turn. `current_tool_defs` is
        // what actually reaches the wire (`ModelRequest::with_tools`); `current_tools` is what the gate/
        // dispatch code below looks up by name — both are swapped together so they never disagree about
        // which tools are live this turn.
        let mut current_tools = self.tools.clone();
        let mut current_tool_defs = self.tool_defs.clone();
        loop {
            if cancel.is_cancelled() {
                close_out_pending_cancellation(session, &current_model);
                // A deliberate beyond design choice, not a port of pi: a message queued via
                // `push`/`push_steer` shortly before (or during) cancellation is discarded here rather
                // than silently riding into whatever unrelated run reuses this same `Steering` handle
                // next — a conservative safety default (a stale queued message surviving an abort is
                // arguably worse than losing it). This diverges from pi's real headless/RPC abort path
                // (`AgentSession.abort()`, `packages/coding-agent/src/core/agent-session.ts:1449-1453`,
                // called from `packages/coding-agent/src/modes/rpc/rpc-mode.ts:424-426`), which does
                // *not* clear queued steer/follow-up messages at all — only the TUI's own `clearQueue()`
                // does, and beyond is headless, not a TUI. (The unused
                // `packages/agent/src/harness/agent-harness.ts`'s `abort()` also clears them, but that's
                // dead code, not pi's real product.) Open question, not settled by this comment: whether
                // beyond's stricter behavior is actually the right default, or whether it should instead
                // match pi's real headless behavior and preserve a queued message across cancellation —
                // flagging for a human call, not deciding it here. Deliberately the narrower,
                // run-scoped clear — not `clear()` — so a message queued via `push_next_turn` still
                // survives into whatever prompt comes next, aborted or not (see `clear_run_scoped`'s
                // own doc comment).
                steering.clear_run_scoped();
                return Err(Error::Cancelled);
            }
            // A pending mid-run model switch (`Steering::request_model_switch`, pi's `prepareNextTurn`
            // equivalent) is applied at the same turn boundary a graceful stop is checked — before the
            // next request is built, never mid-turn.
            if let Some(switch) = steering.take_model_switch() {
                current_model = switch.model.clone();
                // Applied first, against the model this same switch just selected, so a level
                // translates correctly even when the switch changes both at once: `Some(Off)`
                // explicitly disables thinking/reasoning, `Some(level)` sets a new depth via the same
                // `models::thinking_for_level` translation `serve.rs`'s `set_model`/`cycle_model` RPC
                // handlers use for an idle-time switch. `None` leaves both shadows exactly as they
                // were (including whatever an *earlier* switch this run already applied).
                if let Some(level) = switch.thinking_level {
                    let caps = crate::models::capabilities(&current_model);
                    let (thinking, effort) = crate::models::thinking_for_level(&caps, level);
                    current_thinking = thinking;
                    current_reasoning_effort = effort;
                }
                // A raw thinking-budget override, if also given, wins over the level's own translated
                // budget — a caller after an exact token count, not a portable level, still gets
                // exactly that number.
                if let Some(budget) = switch.thinking {
                    current_thinking = Some(budget);
                }
                sink(AgentEvent::ModelSwitched {
                    model: switch.model,
                    thinking: current_thinking,
                });
            }
            // A pending mid-run tool-set switch (`Steering::request_tool_set`, pi's `setTools`/
            // `setActiveTools` equivalent, Task #13) is applied at this same turn boundary, before the
            // next request is built — so it takes effect starting the very next turn, never mid-turn.
            if let Some(new_tools) = steering.take_tool_switch() {
                current_tool_defs = new_tools.definitions().into();
                current_tools = new_tools;
                sink(AgentEvent::ToolsUpdated {
                    tool_names: current_tool_defs.iter().map(|d| d.name.clone()).collect(),
                });
            }
            if steps_this_call >= self.max_steps {
                let err = Error::MaxSteps(self.max_steps);
                sink(AgentEvent::Error {
                    message: err.to_string(),
                });
                return Err(err);
            }

            // Proactive compaction: once the live prompt crosses the threshold, summarize the prefix
            // before building the next request so the run never walks into the context wall. A
            // failure here is reported, not propagated — see `compact_or_report`'s doc comment. A
            // cancellation is routed through `finish_compaction` rather than a bare `?`, so it gets the
            // same session/steering cleanup every other cancellation exit in this loop uses instead of
            // leaving the session on a dangling `user` turn (see `cancel_run`'s doc comment).
            if self.compaction.enabled && compaction::should_compact(session, &self.compaction) {
                let result = self
                    .compact_or_report(session, CompactionReason::Threshold, &cancel, &mut sink)
                    .await;
                finish_compaction(result, session, &current_model, &steering)?;
            } else if !self.compaction.enabled
                && compaction::is_hard_overflow(
                    session,
                    &self.compaction,
                    last_stop_reason,
                    self.max_tokens,
                )
            {
                // Auto-compaction is off, but the live prompt has already reached (or a `MaxTokens`
                // stop implies it's about to reach) the raw context window, not just the soft
                // threshold above — disabling proactive compaction isn't license to keep sending
                // requests that are already guaranteed to overflow. See `is_hard_overflow`'s doc
                // comment for why this bypasses the `enabled` gate but the threshold check above
                // doesn't.
                let result = self
                    .compact_or_report(session, CompactionReason::Overflow, &cancel, &mut sink)
                    .await;
                finish_compaction(result, session, &current_model, &steering)?;
            }

            sink(AgentEvent::TurnStart {
                step: session.steps + 1,
            });

            let mut req = ModelRequest::new(
                current_model.clone(),
                session.messages.clone(),
                self.max_tokens,
            )
            .with_tools(current_tool_defs.clone())
            .with_cache_long(self.cache_long);
            // pi-parity (Task 1, serve pass 19): wire-layer enforcement twin of the ingestion-time
            // `block_images` gate — see `ModelRequest::block_images`'s doc comment. Setting this here,
            // on every turn built from `session.messages`, closes both gaps an ingestion-only gate
            // can't: an RPC client pushing an image directly into a running session, and an image
            // already persisted in session history before the flag was toggled on getting resent on a
            // resumed session.
            req.block_images = self.block_images;
            // Task #15 (pi-parity): a per-turn callback (`with_system_fn`) is re-evaluated fresh here,
            // every turn — the seam a long-running call (which holds `&self` for its whole span, so
            // `set_system`'s `&mut self` is unavailable mid-run) uses to keep a time-varying prompt (a
            // date stamp) current turn-to-turn. Takes priority over the static `system` string when
            // both are set, mirroring pi's single function-or-string `systemPrompt` field.
            if let Some(system_fn) = &self.system_fn {
                req = req.with_system(system_fn());
            } else if let Some(system) = &self.system {
                req = req.with_system(system.clone());
            }
            if let Some(budget) = current_thinking {
                req = req.with_thinking(budget);
            }
            if let Some(effort) = current_reasoning_effort {
                req = req.with_reasoning_effort(effort);
            }
            if let Some(temperature) = self.temperature {
                req = req.with_temperature(temperature);
            }
            if let Some(key) = &self.cache_key {
                req = req.with_cache_key(key.clone());
            }

            // `emit` borrows `sink` for the turn; bind the result, then drop the borrow before handling
            // an error so the terminal `Error` event can go out through `sink`.
            let mut partial_turn: Option<Turn> = None;
            let turn_result = {
                let mut emit = |ev: StreamEvent| sink(AgentEvent::Stream(ev));
                self.run_turn(req, &mut emit, &cancel, &mut partial_turn)
                    .await
            };
            let mut turn = match turn_result {
                Ok(turn) => turn,
                // A cancellation is a user request, not a fault — return it without an `Error` event.
                // Reachable here only pre-connect (no byte ever streamed back — a mid-stream abort
                // resolves to `Ok(Turn { stop_reason: Aborted, .. })` instead, handled by the ordinary
                // turn-commit path below), so the session needs the same closing-out `run_turn`'s
                // sibling paths give a mid-stream abort.
                Err(Error::Cancelled) => {
                    return Err(cancel_run(session, &current_model, &steering));
                }
                // The provider rejected the request for exceeding its context window. Compact once and
                // retry the same turn; if it still overflows (or there's nothing to compact), give up.
                Err(e) if is_context_overflow(&e) && !overflow_recovered => {
                    match self
                        .compact(
                            session,
                            CompactionReason::Overflow,
                            &cancel,
                            &mut sink,
                            None,
                        )
                        .await
                    {
                        Ok(outcome) if outcome.compacted() => {
                            overflow_recovered = true;
                            continue;
                        }
                        Ok(_) => {
                            sink(AgentEvent::Error {
                                message: e.to_string(),
                            });
                            session.push(Message::error(e.to_string()));
                            return Err(e);
                        }
                        // A cancellation is left exactly as-is: the user asked to stop, that's not a
                        // compaction failure to explain — route it through the same session/steering
                        // cleanup every other cancellation exit in this loop uses (see `cancel_run`'s
                        // doc comment), rather than reporting it as an `AgentEvent::Error`.
                        Err(Error::Cancelled) => {
                            return Err(cancel_run(session, &current_model, &steering));
                        }
                        // The recovery compaction itself failed (e.g. the summarization call errored)
                        // — surface a curated message, not the raw underlying failure, matching pi's
                        // own `_runAutoCompaction` catch block ("Context overflow recovery failed:
                        // {error}").
                        Err(compact_err) => {
                            let message =
                                format!("Context overflow recovery failed: {compact_err}");
                            sink(AgentEvent::Error {
                                message: message.clone(),
                            });
                            session.push(Message::error(message));
                            return Err(compact_err);
                        }
                    }
                }
                // A second context-overflow on the very turn a compaction just tried to recover from
                // — recompacting again within this same call would either find nothing new to fold or
                // just repeat the same failure, so this never loops back into the arm above a second
                // time. Curated, matching pi's own guard against retrying overflow recovery more than
                // once per call (`_checkCompaction`'s `_overflowRecoveryAttempted` check).
                Err(e) if is_context_overflow(&e) && overflow_recovered => {
                    let message = "Context overflow recovery failed after one compact-and-retry \
                        attempt. Try reducing context or switching to a larger-context model."
                        .to_string();
                    sink(AgentEvent::Error {
                        message: message.clone(),
                    });
                    session.push(Message::error(message));
                    return Err(e);
                }
                Err(e) => {
                    sink(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    // Persist a closing assistant record — pi's `handleRunFailure` (`agent.ts`) does
                    // the same — so the session's last message is never the user's own un-answered
                    // prompt. Without this, a client's retry after a transient failure would append a
                    // second consecutive `user` turn, a shape no wire dialect accepts.
                    //
                    // If real content had already streamed before this failure struck (`partial_turn`,
                    // set by `run_turn_once`), keep it rather than closing out with a bare empty
                    // record — pi's own dialects keep whatever streamed before a mid-stream failure,
                    // just like they do for a mid-stream cancellation (see the `StopReason::Aborted`
                    // arm below). A pre-connect failure (nothing ever streamed) leaves `partial_turn`
                    // `None`, falling back to the original bare closing record.
                    let closing = match partial_turn.take() {
                        Some(turn) if !turn.blocks.is_empty() => {
                            let usage = turn.usage;
                            Message::assistant(turn.blocks)
                                .with_model_id(&current_model)
                                .with_usage(usage)
                                .with_error(e.to_string())
                        }
                        _ => Message::error(e.to_string()),
                    };
                    session.push(closing);
                    return Err(e);
                }
            };
            overflow_recovered = false;
            // Take the truncation guard for *this* turn and disarm it in one move, so it survives
            // exactly one iteration: the turn immediately after a truncation recovery sees `true` (and
            // therefore refuses to recover a second time), and every turn after that sees `false` again.
            // Reading it into a local here — rather than testing the field directly down in the
            // `MaxTokens` branch — is what makes the guard immune to the reset-ordering bug that made
            // its predecessor dead code: there is no longer any path on which the flag is cleared
            // between being set and being read. It also re-arms correctly through a tool round-trip,
            // which returns to the top of the loop without ever reaching that branch.
            let recovered_truncation_last_turn =
                std::mem::replace(&mut truncation_recovered, false);
            last_stop_reason = turn.stop_reason;
            let malformed: HashMap<String, String> =
                std::mem::take(&mut turn.malformed).into_iter().collect();
            // Snapshot usage *before* appending this turn's own assistant message: `record_usage`
            // captures `messages.len()` as the boundary `trailing_tokens` estimates forward from, and
            // `turn.usage.input_tokens` describes the prompt that was sent *without* this response —
            // recording it after the push would make the boundary include a message the snapshot
            // itself doesn't account for, undercounting the live context by this turn's own output
            // until the next real usage snapshot arrives.
            session.record_usage(turn.usage);
            let mut blocks = turn.blocks;
            if turn.stop_reason == StopReason::Aborted && blocks.is_empty() {
                // Cancelled before any content had streamed at all (e.g. mid-`MessageStart`, no delta
                // yet) — pi's own contract tolerates a genuinely empty `content: []` here (see
                // `abort.test.ts`'s `testImmediateAbort`), but an empty content array is a shape at
                // least one dialect's replay validation rejects outright. One empty text block instead
                // — same wire-safety tradeoff `Message::error` already makes for the analogous case.
                blocks.push(ContentBlock::text(String::new()));
            }
            let mut assistant_message = Message::assistant(blocks)
                .with_model_id(&current_model)
                .with_usage(turn.usage)
                .with_stop_reason(turn.stop_reason);
            if turn.stop_reason == StopReason::Aborted {
                assistant_message = assistant_message.with_aborted();
            }
            // Let a hook redact/rewrite the model's own generated content — e.g. scrubbing a secret it
            // echoed back — before it's committed to `session`, checkpointed, or handed to the caller in
            // `AgentEvent::TurnEnd` below. A panicking hook, or one that returns a message with the
            // wrong role, falls back to the original rather than risk losing this turn's content or
            // splicing a wrong-role message into the transcript — see `on_assistant_message`'s own doc
            // comment for why both are "fail open, keep the original," matching
            // `should_stop_after_turn`'s identical convention just below.
            let original_role = assistant_message.role;
            assistant_message = catch_tool_panic(self.hooks.on_assistant_message(
                assistant_message.clone(),
                session,
                &cancel,
            ))
            .await
            .ok()
            .filter(|rewritten| rewritten.role == original_role)
            .unwrap_or(assistant_message);
            session.push(assistant_message);
            session.steps += 1;
            steps_this_call += 1;
            sink(AgentEvent::TurnEnd {
                stop_reason: turn.stop_reason,
                step: session.steps,
            });

            // A cancelled-mid-stream turn: the partial content is committed above (matching pi's
            // `abort.test.ts` contract), but this is still a client-requested stop, not an ordinary
            // one — re-raise `Error::Cancelled` exactly as every *other* cancellation path in this
            // loop already does (the top-of-loop check above, `run_turn`'s own pre-connect-cancel
            // branch), so callers that key behavior off "did this run end in `Err(Cancelled)`" (whole-
            // run retry exclusion, `serve.rs`'s `abort` RPC response shape) stay correct. No `AgentEnd`
            // here, matching those same sibling paths — one is only ever emitted on an `Ok(())` return.
            if turn.stop_reason == StopReason::Aborted {
                steering.clear_run_scoped();
                return Err(Error::Cancelled);
            }

            // A refusal blocks dispatch unconditionally — checked here, before `calls` is even
            // collected, so a turn that streamed one or more complete `tool_use` blocks *before* the
            // model was cut off with a refusal (a real Anthropic wire shape: a refusal explanation
            // arriving as trailing content after a tool call already closed; OpenAI's `content_filter`
            // maps to this same stop reason too) never dispatches those calls. Matches pi's
            // `agent-loop.ts`, which returns unconditionally on an "error"/"aborted" stop reason before
            // it ever looks at `message.content` for tool calls, dialect-agnostic. Draining queued
            // steer/follow-up messages here would inject a new user turn right after the model just
            // declined to engage with the current one, which it would likely refuse again — end the run
            // immediately instead, leaving the queue untouched (the same persistent `Steering` handle a
            // later `prompt` call reads from — see `serve.rs`).
            if turn.stop_reason == StopReason::Refusal {
                self.checkpoint_guarded(session).await;
                sink(AgentEvent::AgentEnd {
                    steps: session.steps,
                });
                return Ok(());
            }

            // Collect the tool calls the assistant just made.
            let calls: Vec<(String, String, Value)> = session
                .messages
                .last()
                .map(|m| {
                    m.tool_uses()
                        .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
                        .collect()
                })
                .unwrap_or_default();

            // Dispatch is gated on the presence of complete `tool_use` blocks alone, never on
            // `stop_reason` — matching pi's `agent-loop.ts` (it dispatches off the assistant message's
            // own content, full stop). A turn can legitimately end `MaxTokens` *after* the model already
            // emitted one or more complete tool calls (e.g. it calls two tools, then starts trailing
            // commentary that gets cut off) — gating on `stop_reason == ToolUse` there would silently
            // drop tool calls the model actually made.
            if calls.is_empty() {
                // Silent truncation: the provider replied successfully, but the response was cut off by
                // `max_tokens` alone (no tool calls, so this isn't a "the model needs another turn to
                // keep working" shape) — the model didn't get to finish because there wasn't room left,
                // not because it was done. Left as-is, the run would end here and hand the user a
                // hard-truncated non-answer with no indication anything went wrong. If auto-compaction is
                // enabled and the live prompt has genuinely reached (or this `MaxTokens` stop implies
                // it's about to reach) the raw context window, compact and retry this same turn instead —
                // pi's own `_checkCompaction`+silent-overflow handling does the same. Guarded by
                // `truncation_recovered` (see its declaration) so a second silent truncation right after
                // this recovery gives up and reports the truncated answer instead of looping forever; a
                // genuine improvement should always show up as *some* progress within one retry, same
                // rationale as the error-based path.
                if turn.stop_reason == StopReason::MaxTokens
                    && !recovered_truncation_last_turn
                    && self.compaction.enabled
                    && compaction::is_hard_overflow(
                        session,
                        &self.compaction,
                        turn.stop_reason,
                        self.max_tokens,
                    )
                {
                    match self
                        .compact(
                            session,
                            CompactionReason::Overflow,
                            &cancel,
                            &mut sink,
                            None,
                        )
                        .await
                    {
                        Ok(outcome) if outcome.compacted() => {
                            // Compaction freed real room — discard the truncated response and retry the
                            // same turn fresh; keeping both would leave two consecutive assistant turns
                            // once the retry's real response lands, a shape no dialect accepts.
                            Arc::make_mut(&mut session.messages).pop();
                            session.steps -= 1;
                            steps_this_call -= 1;
                            truncation_recovered = true;
                            continue;
                        }
                        Ok(_) => {
                            // Nothing worth compacting (already at a clean, minimal boundary) — the
                            // truncated response is the best available answer; fall through and report
                            // it normally rather than silently discarding it with nothing to replace it.
                        }
                        // The user asked to stop — surface it as the documented `Error::Cancelled`
                        // contract every other cancellation exit in this loop honors (see `cancel_run`'s
                        // doc comment), not as a `CompactionFailed` event falling through to an ordinary
                        // `Ok(())` completion, which would silently swallow the cancellation.
                        Err(Error::Cancelled) => {
                            return Err(cancel_run(session, &current_model, &steering));
                        }
                        Err(e) => {
                            // The recovery attempt itself failed (e.g. the summarization call errored) —
                            // non-fatal here, unlike the error-based overflow-retry arm: a real (if
                            // truncated) answer already exists, so report it rather than failing the
                            // whole run over a failed *recovery* attempt.
                            sink(AgentEvent::CompactionFailed {
                                reason: CompactionReason::Overflow,
                                message: e.to_string(),
                            });
                        }
                    }
                }
                // The assistant's tool-less reply is committed above (`session.push(assistant_message)`)
                // regardless of how this branch resolves from here — a graceful stop or an ordinary end
                // with nothing left to inject (a refusal already returned above, before `calls` was even
                // collected) — so it's a valid, resumable checkpoint (see `CheckpointHook`) exactly like
                // a tool round-trip's own pre-dispatch checkpoint just below this branch. Previously only
                // the tool-calling half of a turn ever reached a checkpoint; a plain conversational reply
                // (the model's *most* common shape) never did, silently relying on a caller's own
                // post-run persist to ever see it recorded.
                self.checkpoint_guarded(session).await;
                // A pending graceful-stop request wins over draining follow-up/steer messages, exactly
                // as it wins over continuing tool-call turns below — the queue is left untouched (same
                // rationale as the refusal case above) so nothing queued for "next time" is lost. A
                // content-aware `should_stop_after_turn` hook can request the same stop — see its own
                // doc comment; a panicking hook fails open (doesn't stop) rather than crashing the run.
                let hook_wants_stop = match session.messages.last() {
                    Some(assistant_msg) => catch_tool_panic(self.hooks.should_stop_after_turn(
                        assistant_msg,
                        &[],
                        session,
                        &cancel,
                    ))
                    .await
                    .unwrap_or(false),
                    None => false,
                };
                if steering.take_stop_requested() || hook_wants_stop {
                    sink(AgentEvent::AgentEnd {
                        steps: session.steps,
                    });
                    return Ok(());
                }
                // The model ended its turn. Before stopping, drain any follow-up messages a client
                // queued (plus any steer messages stranded by a tool-less turn) and continue with them
                // as new user turns. The last message is the assistant's, so pushing user turns here
                // keeps the wire's role alternation valid.
                let injected = steering.drain_at_stop();
                if injected.is_empty() {
                    sink(AgentEvent::AgentEnd {
                        steps: session.steps,
                    });
                    return Ok(());
                }
                let count = injected.len();
                for msg in injected {
                    if msg.images.is_empty() {
                        session.user(msg.text);
                    } else {
                        session.push(crate::message::Message::user_with_images(
                            msg.text, msg.images,
                        ));
                    }
                }
                sink(AgentEvent::Steered { messages: count });
                // A plain user message ends the visible history here — a valid, resumable checkpoint
                // (see `CheckpointHook`) before the next model call.
                self.checkpoint_guarded(session).await;
                continue;
            }

            // The assistant's own turn — including the `tool_use` blocks collected into `calls` above —
            // is durable now, *before* any of those tools ever run. Confirmed to match pi's real
            // persistence path, not just the dead harness: `packages/agent/src/agent-loop.ts` awaits
            // `emit({ type: "message_end", message })` (line 185) before ever calling `executeToolCalls`
            // (line 208); that `emit` is `Agent.processEvents` (`packages/agent/src/agent.ts`), which
            // awaits every registered listener — including
            // `packages/coding-agent/src/core/agent-session.ts`'s `_handleAgentEvent`, which persists
            // synchronously via `SessionManager.appendMessage` → `_persist`'s `appendFileSync`
            // (`packages/coding-agent/src/core/session-manager.ts:934-967`) — to completion first.
            // Without this, a crash mid-tool-execution (e.g. mid-`bash`) loses the record that the model
            // asked for these specific calls even though a tool that already ran (an `edit`, a `write`)
            // already took effect on disk — the persisted transcript and physical reality silently
            // diverge on resume. Only reachable here (past the `calls.is_empty()` branch above, which
            // checkpoints its own tool-less case separately) — a call requesting tools never falls
            // through to that other checkpoint, so this is the one and only checkpoint a `tool_use` turn
            // pays for here.
            self.checkpoint_guarded(session).await;

            // Run the tools and feed results back as a single user turn. A tool's own failure becomes
            // an error `tool_result`, not an aborted run — the model can react to it next turn.
            //
            // The calls run concurrently: tools are I/O-bound (file reads, shell commands, the
            // `beyond` CLI), and a model routinely batches independent ones in a single turn, so
            // overlapping them collapses the tool phase from the sum of their latencies to its slowest
            // member. `ToolStart` is emitted up front, for the *whole* batch, in call order, before
            // Phase 1's own gating loop below even begins; `ToolEnd` is emitted live, the instant each
            // call's own result is known — a client watching the event stream sees completions in
            // actual finish order, not batched after the slowest call joins. The *transcript* (the
            // `tool_result` blocks pushed to the session below) still stays deterministic regardless of
            // finish order, rebuilt in call order after the join.
            //
            // A known departure from pi's own literal order: pi's real `executeToolCallsParallel`
            // (`packages/agent/src/agent-loop.ts`, the live library the shipped `Agent` class actually
            // runs — not the unused harness) interleaves each call's own `tool_execution_start` emission
            // with that same call's `prepareToolCall` gate, one call fully gated before the next call's
            // start event even fires (`start1 → gate1 → start2 → gate2 → ...`), only parallelizing the
            // execution phase afterward via `Promise.all`. Emitting the whole batch's `ToolStart` up
            // front instead means a client sees every call announced immediately, before any gate/hook
            // has had a chance to run or block one — observability/UX only, the persisted transcript and
            // tool semantics are unaffected either way. Left as up-front-batch emission here rather than
            // interleaved with Phase 1's gate loop below: the `sequential_execution_requested` branch
            // (`run_tool_calls_interleaved`, Task #28) deliberately emits no `ToolStart` of its own and
            // relies on this same up-front loop having already covered its whole batch (see that
            // function's own doc comment) — moving emission into Phase 1's gate loop would need to move
            // in lockstep with that other dispatch path too, not just this one, to keep both giving every
            // call a `ToolStart` exactly once.
            //
            // Calls aren't always independent, though: two calls that write the same path (the model
            // batching two `edit`s against one file) would otherwise race on disk. `Tool::write_target`
            // flags the path a call would mutate; calls sharing a target are grouped and run
            // sequentially, in call order, within that group, while distinct groups still run
            // concurrently against each other.
            //
            // All results are gathered into *one* user message rather than one message per result:
            // both Anthropic and the internal model carry a turn's tool results as multiple blocks on
            // a single `user` turn, and Anthropic rejects consecutive same-role messages — N separate
            // `user` messages would 400 the next request whenever the model batched N>1 tools.
            for (id, name, input) in &calls {
                sink(AgentEvent::ToolStart {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }
            // A call whose mutation scope can't be named (see `Tool::conservative_exclusive`, e.g. an
            // opaque `bash` command) still groups by its own `write_target` (`None`, so its own
            // `solo:` group) — grouping is unchanged — but its presence anywhere in the batch caps the
            // *group-level* concurrency below to 1, so it can't race a same-turn `edit`/`write` it has
            // no path to be grouped against.
            let exclusive_turn = calls.iter().any(|(_, name, _)| {
                current_tools
                    .get(name)
                    .is_some_and(|t| t.conservative_exclusive())
            });
            // Task #28 (pi-parity): any call naming a tool that opts into
            // `ToolExecutionMode::Sequential` routes the turn's *whole* batch through the fully-
            // interleaved gate→execute→finalize-per-call path below instead of this function's default
            // gate-the-batch-then-execute split — see `run_tool_calls_interleaved`'s own doc comment.
            let sequential_execution_requested = calls.iter().any(|(_, name, _)| {
                current_tools.get(name).and_then(|t| t.execution_mode())
                    == Some(crate::tool::ToolExecutionMode::Sequential)
            });
            let mut groups: HashMap<String, (Option<String>, Vec<usize>)> = HashMap::new();
            for (i, (_, name, input)) in calls.iter().enumerate() {
                let target = current_tools.get(name).and_then(|t| t.write_target(input));
                let key = target
                    .clone()
                    .map(|path| format!("path:{path}"))
                    .unwrap_or_else(|| format!("solo:{i}"));
                groups
                    .entry(key)
                    .or_insert_with(|| (target, Vec::new()))
                    .1
                    .push(i);
            }
            let this = self;
            let malformed = &malformed;
            // Read-only reborrow, shared across every hook invocation below: by this point the
            // requesting assistant turn (this batch's own `tool_use` blocks) is already
            // `session.messages.last()` (the pre-dispatch checkpoint above established that same
            // invariant), so a hook can see the call it's gating in its full conversational context —
            // pi's `BeforeToolCallContext`/`AfterToolCallContext` carry the same
            // (`assistantMessage`/`context`). `session` itself isn't mutated again until every call in
            // this batch has finished (`session.push(Message::tool_results(..))` below), so this shared
            // borrow's lifetime never actually overlaps a mutation.
            let session_ref: &Session = session;
            let cancel_ref = &cancel;

            let (results, cancelled_mid_dispatch): (Vec<Option<ToolCallResult>>, bool) =
                if sequential_execution_requested {
                    self.run_tool_calls_interleaved(
                        session_ref,
                        &calls,
                        malformed,
                        cancel_ref,
                        &mut sink,
                        &current_tools,
                    )
                    .await
                } else {
                    // Phase 1 — gate every call sequentially, in call order, before any call's actual
                    // execution begins: the malformed-args check and the `before_tool_call` hook both run
                    // here, one call fully resolved before the next call's gate even starts. Matches pi's
                    // `prepareToolCall`, resolved in a plain sequential loop ahead of
                    // `executeToolCallsParallel`'s `Promise.all` (agent-loop.ts:451-516; the
                    // `ToolExecutionMode` doc comment at types.ts:34-41 states the same contract). Without
                    // this, a later-declared call's gate could run concurrently with an earlier-approved
                    // call's own tool already executing — a permission hook reasoning about "what's already
                    // running" (a concurrency-aware policy, a rate limiter) would see a half-approved batch
                    // instead of a fully-gated one. Only the execution + `after_tool_call` phase below this
                    // one actually parallelizes.
                    //
                    // `outcomes[i]` ends up `Some(Immediate(result))` for a call fully resolved without ever
                    // running (malformed streamed args, or a hook block), `Some(Ready(coerced))` for a call
                    // that cleared its gate and is ready to actually run in phase 2, or `None` only if
                    // cancellation cut this loop short before reaching it.
                    #[derive(Clone)]
                    enum GateOutcome {
                        Immediate(ToolCallResult),
                        Ready(Value),
                    }
                    let mut outcomes: Vec<Option<GateOutcome>> = vec![None; calls.len()];
                    let mut gate_cancelled = false;
                    // The active model can't change mid-turn (only at a turn boundary — see the
                    // mid-run model-switching capability), so this is the same value for every call in
                    // the batch: computed once here instead of once per call inside the loop below.
                    let supports_vision = crate::models::capabilities(&this.model).supports_vision
                        && !this.block_images;
                    for (i, (id, name, input)) in calls.iter().enumerate() {
                        if cancel_ref.is_cancelled() {
                            gate_cancelled = true;
                            break;
                        }
                        if let Some(raw) = malformed.get(id) {
                            // The model streamed a tool call whose argument fragments never formed valid
                            // JSON. Feed that back as an error result the model can correct next turn rather
                            // than aborting the whole run on one malformed call.
                            outcomes[i] = Some(GateOutcome::Immediate((
                                format!(
                                    "tool call arguments were not valid JSON and could not be parsed: {raw}"
                                ),
                                Vec::new(),
                                true,
                                false,
                            )));
                            continue;
                        }
                        // Pi-parity fix: pi's `prepareToolCall` looks up the tool *first* — an unregistered
                        // tool name resolves straight to its "not found" immediate outcome before
                        // `prepareToolCallArguments`, `validateToolArguments`, or `config.beforeToolCall` ever
                        // run (`agent-loop.ts`'s `prepareToolCall`, `!tool` branch). This used to fall through
                        // to the coercion step and `before_tool_call` below with `input.clone()` unchanged,
                        // invoking the permission hook for a call that was never going to run anyway — the
                        // "unknown tool" outcome only surfaced later, in this turn's execution phase. Detect
                        // it here instead, before either runs, short-circuiting straight to the exact same
                        // "unknown tool: {name}" error result the execution phase below already produces —
                        // only *when* it's detected moves, not the message or outcome itself.
                        let Some(tool) = current_tools.get(name) else {
                            outcomes[i] = Some(GateOutcome::Immediate((
                                format!("unknown tool: {name}"),
                                Vec::new(),
                                true,
                                false,
                            )));
                            continue;
                        };
                        // Best-effort pi-parity coercion (`validation.rs`, matches pi's AJV-backed
                        // `validateToolArguments`): a provider that stringified a primitive the model emitted
                        // as genuinely typed (`{"count":"42"}` instead of `{"count":42}`) would otherwise fail
                        // the tool's own `as_i64()`/`as_bool()` extraction with a confusing "missing field"
                        // error. Falls back to the raw input unchanged on any coercion failure — a genuinely
                        // malformed call still surfaces through the tool's own existing, clearer validation
                        // error rather than a new failure path. Run *before* `before_tool_call` (pi-parity
                        // fix — matches pi's `prepareToolCall`, which calls `validateToolArguments` before
                        // `config.beforeToolCall`): a permission hook must see the same coerced/typed
                        // arguments the tool itself is about to run with, not the model's raw, possibly
                        // stringified wire values.
                        let mut coerced = crate::validation::coerce_tool_arguments(
                            &tool.input_schema(),
                            input.clone(),
                        )
                        .unwrap_or_else(|_| input.clone());
                        // Task #36 (pi-parity): lets `read` append a "current model doesn't support
                        // images" note when it reads an image file — schema-undocumented, so it's
                        // invisible to the model and ignored by every other tool. Task #26 (pi-parity):
                        // `this.block_images` is an operator-facing override that forces this same
                        // downgrade path regardless of the model's real capability.
                        if let Some(obj) = coerced.as_object_mut() {
                            obj.insert(
                                crate::tool::MODEL_SUPPORTS_VISION_KEY.to_string(),
                                supports_vision.into(),
                            );
                        }
                        if let Some(reason) = match catch_tool_panic(this.hooks.before_tool_call(
                            name,
                            &coerced,
                            session_ref,
                            cancel_ref,
                        ))
                        .await
                        {
                            // A panicking permission hook fails closed: better to block the call than to
                            // silently treat a crashed check as "allowed".
                            Ok(reason) => reason,
                            Err(panic_msg) => Some(panic_msg),
                        } {
                            // A hook blocked the call (e.g. a permission policy). Feed the reason back as an
                            // error result instead of running the tool.
                            outcomes[i] = Some(GateOutcome::Immediate((
                                format!("tool call blocked: {reason}"),
                                Vec::new(),
                                true,
                                false,
                            )));
                            continue;
                        }
                        // Task #3 (pi-parity, high-severity): re-check cancellation *after* `before_tool_call`
                        // returned — not just at the top of this loop iteration. A slow permission-check hook
                        // can observe cancellation firing mid-await; without this second check, a single-call
                        // (or last-in-batch) turn had no *later* iteration to catch it, so the call was marked
                        // `Ready` and phase 2 dispatched it for real despite the run having already been
                        // cancelled. Treated exactly like a cancellation caught at the top of the loop:
                        // `outcomes[i]` stays `None` and `gate_cancelled` skips phase 2 entirely, so
                        // `repair_cancelled_dispatch` below synthesizes the same cancelled error result it
                        // already does for any other call cut short mid-gate.
                        if cancel_ref.is_cancelled() {
                            gate_cancelled = true;
                            break;
                        }
                        outcomes[i] = Some(GateOutcome::Ready(coerced));
                    }
                    let outcomes = Arc::new(outcomes);

                    // `None` until that call's group finishes (or phase 1 above resolved it directly); a slot
                    // left `None` after dispatch means cancellation aborted it before it ran, and
                    // `repair_cancelled_dispatch` needs to tell that apart from a real (possibly empty) result
                    // to synthesize a matching error `tool_result` for it.
                    let mut results: Vec<Option<ToolCallResult>> = vec![None; calls.len()];
                    // Phase 2 — actually execute every gated call, grouped and bounded exactly as before:
                    // skipped entirely when phase 1 above was itself cut short by cancellation (nothing gated
                    // means nothing left to execute).
                    let mut cancelled_mid_dispatch = gate_cancelled;
                    if !gate_cancelled {
                        // Per-turn progress channel: every call gets a `ToolProgress` cloning `prog_tx`; the
                        // drain loop below forwards each update to `sink` as it arrives. `futures`' mpsc keeps
                        // this executor-agnostic (no tokio in the library).
                        let (prog_tx, mut prog_rx) =
                            futures::channel::mpsc::unbounded::<crate::tool::ToolUpdate>();
                        let prog_tx = &prog_tx;
                        let group_runs = groups.into_values().map(|(target, indices)| {
                            let calls = &calls;
                            let prog_tx = prog_tx.clone();
                            let cancel = cancel_ref.clone();
                            let write_locks = this.write_locks.clone();
                            let outcomes = outcomes.clone();
                            let current_tools = current_tools.clone();
                            async move {
                                // Held for the group's whole serial run: extends the intra-turn grouping above
                                // across turn and session boundaries, so a concurrently-running turn (or a
                                // different session sharing this `Agent`'s registry) touching the same path
                                // really waits, not just calls within this one turn.
                                //
                                // Wrapped in `Arc` so a clone can ride into a tool's non-cancellable
                                // `spawn_blocking` write (via `ToolProgress::write_lock_keepalive`): the
                                // registry lock then releases only once *both* this group future's clone and
                                // the in-flight write's clone are gone — i.e. after the write has physically
                                // landed — not the instant cancellation drops this future while the detached
                                // `spawn_blocking` runs on regardless (see `write_lock.rs`).
                                let write_guard = match &target {
                                    Some(path) => {
                                        Some(std::sync::Arc::new(write_locks.lock(path).await))
                                    }
                                    None => None,
                                };
                                let mut out = Vec::with_capacity(indices.len());
                                for i in indices {
                                    let (id, name, input) = &calls[i];
                                    // Per call: (text, images, is_error, terminate). Hooks rewrite the *text*
                                    // and error flag; images and the terminate hint pass through untouched.
                                    // The gate/coercion decision itself was already made, sequentially and in
                                    // call order, by phase 1 above — this only ever runs the tool (or replays
                                    // an already-immediate outcome) and the `after_tool_call` rewrite.
                                    let result: ToolCallResult = match outcomes[i].clone() {
                                        Some(GateOutcome::Immediate(result)) => result,
                                        Some(GateOutcome::Ready(coerced)) => {
                                            let progress = crate::tool::ToolProgress::new(
                                                prog_tx.clone(),
                                                id.clone(),
                                                name.clone(),
                                                cancel.clone(),
                                            )
                                            .with_write_lock(write_guard.clone());
                                            let (text, images, is_error, terminate) =
                                                match current_tools.get(name) {
                                                    Some(tool) => {
                                                        match catch_tool_panic(
                                                            tool.run_streaming(coerced, &progress),
                                                        )
                                                        .await
                                                        {
                                                            Ok(Ok(o)) => (
                                                                o.text,
                                                                o.images,
                                                                false,
                                                                o.terminate,
                                                            ),
                                                            Ok(Err(e)) => (
                                                                e.to_string(),
                                                                Vec::new(),
                                                                true,
                                                                false,
                                                            ),
                                                            Err(panic_msg) => {
                                                                (panic_msg, Vec::new(), true, false)
                                                            }
                                                        }
                                                    }
                                                    None => (
                                                        format!("unknown tool: {name}"),
                                                        Vec::new(),
                                                        true,
                                                        false,
                                                    ),
                                                };
                                            // Let a hook rewrite the result text/images (redact, cap,
                                            // reclassify) before it's fed back to the model. A panicking hook
                                            // here just keeps the tool's own original (text, images, is_error)
                                            // — losing a real, already-obtained result to a broken *rewrite*
                                            // attempt would be strictly worse than ignoring the rewrite.
                                            let (text, images, is_error) =
                                                match catch_tool_panic(this.hooks.after_tool_call(
                                                    name,
                                                    input,
                                                    text.clone(),
                                                    images.clone(),
                                                    is_error,
                                                    session_ref,
                                                    &cancel,
                                                ))
                                                .await
                                                {
                                                    Ok(rewritten) => rewritten,
                                                    Err(_) => (text, images, is_error),
                                                };
                                            (text, images, is_error, terminate)
                                        }
                                        // Only reachable if cancellation cut phase 1's sequential gate loop
                                        // short before reaching this call — but that path sets
                                        // `gate_cancelled` and skips this whole execution phase, so this arm
                                        // never actually runs in practice. Kept so the match stays exhaustive
                                        // without an `.unwrap()`.
                                        None => (
                                            "cancelled: tool call aborted before it finished"
                                                .to_string(),
                                            Vec::new(),
                                            true,
                                            false,
                                        ),
                                    };
                                    // Sent the instant this call's own result is known — not batched until
                                    // every group in the turn finishes — so a client watching the event stream
                                    // sees each tool's completion as it actually happens, not all-at-once
                                    // after the slowest concurrently-dispatched call joins.
                                    prog_tx
                                        .unbounded_send(crate::tool::ToolUpdate::End {
                                            id: id.clone(),
                                            name: name.clone(),
                                            result: result.0.clone(),
                                            is_error: result.2,
                                        })
                                        .ok();
                                    out.push((i, result));
                                }
                                out
                            }
                        });
                        // Bound how many groups run at once. `buffer_unordered` is safe here because each
                        // group yields its results tagged with their original call index `i`; cross-group
                        // completion order never reaches the transcript, which is rebuilt in call order below.
                        // `exclusive_turn` caps this at 1 instead — with only ever one group in flight, a
                        // `bash` call (or anything else `conservative_exclusive`) can't race a same-turn
                        // `edit`/`write` group it has no path to be grouped against; which group runs first
                        // still doesn't matter for the transcript, same as the concurrent case.
                        // `self.sequential_tools` caps it at 1 too, host-selected rather than inferred from the
                        // calls themselves — e.g. a deterministic-repro debugging session, or a host policy
                        // that never wants two tool calls actually overlapping.
                        //
                        // Race the whole execution phase against cancellation: a tripped token drops `drain`,
                        // which drops every in-flight tool future — aborting a hung `bash` (its
                        // `kill_on_drop` child dies) and any other long-running tool — and returns promptly
                        // instead of waiting them all out. The block scopes `drain`'s `&mut results` borrow so
                        // the transcript below can consume them; `cancelled_mid_dispatch` is only *acted on*
                        // after the block ends and that borrow is fully released (repairing the transcript
                        // needs to move `results` out).
                        let concurrency = if self.sequential_tools || exclusive_turn {
                            1
                        } else {
                            MAX_CONCURRENT_TOOL_GROUPS
                        };
                        let drain = async {
                            let mut group_stream = futures::stream::iter(group_runs)
                                .buffer_unordered(concurrency)
                                .fuse();
                            // Forward tool-progress chunks to `sink` as they arrive, racing them against group
                            // completion (progress biased first so chunks flush promptly). The loop ends when
                            // every group has finished; the progress channel's senders outlive it, so we stop
                            // on `group_stream`, not on the receiver.
                            loop {
                                futures::select_biased! {
                                    prog = prog_rx.next() => {
                                        if let Some(u) = prog {
                                            emit_tool_update(&mut sink, u);
                                        }
                                    }
                                    group = group_stream.next() => match group {
                                        Some(group) => {
                                            for (i, result) in group {
                                                results[i] = Some(result);
                                            }
                                        }
                                        None => break,
                                    }
                                }
                            }
                            // Flush any updates buffered between the final poll and group completion.
                            while let Ok(u) = prog_rx.try_recv() {
                                emit_tool_update(&mut sink, u);
                            }
                        };
                        let cancelled = cancel.cancelled();
                        futures::pin_mut!(drain, cancelled);
                        if let Either::Right(((), _)) = select(drain, cancelled).await {
                            cancelled_mid_dispatch = true;
                        }
                    }
                    (results, cancelled_mid_dispatch)
                };
            if cancelled_mid_dispatch {
                // Cancelled mid-gate or mid-dispatch: the assistant message (with its `ToolUse`
                // blocks) is already committed above, but the tool-results message below never will
                // be. Left as-is, the session would end on an orphaned `tool_use` with no matching
                // `tool_result` — a shape both Anthropic and OpenAI reject on resume. Repair it before
                // propagating the cancellation.
                repair_cancelled_dispatch(session, &calls, results);
                steering.clear_run_scoped();
                return Err(Error::Cancelled);
            }
            // `ToolEnd` for each call was already emitted live, from within `group_runs`, the instant
            // that call's own result was known — not here, after every group in the turn has joined.
            // This loop only rebuilds the transcript's tool-result blocks, in call order.
            let mut result_blocks = Vec::with_capacity(calls.len());
            // A tool may ask to end the run; honor it only when *every* call in the batch agrees, so a
            // single tool can't cut off others the model dispatched alongside it.
            let mut terminate = !results.is_empty();
            for ((id, _, _), result) in calls.iter().zip(results) {
                let (content, images, is_error, wants_terminate) = resolve_tool_result(result);
                terminate &= wants_terminate;
                result_blocks.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: cap_tool_result(content),
                    is_error,
                    images,
                });
            }
            // Mid-run steering: fold any *steer* messages a client queued while the agent was working
            // into this same tool-results user turn (as trailing text blocks). Injecting them here —
            // rather than only at a stop boundary — lets a client redirect a busy agent between
            // tool-executing turns. They ride on the existing user message instead of a new one, so
            // role alternation stays valid (no two consecutive user turns). Follow-ups are a separate
            // lane, injected only at the stop boundary below.
            let steered = steering.drain_steer();
            let steered_count = steered.len();
            for msg in steered {
                result_blocks.push(ContentBlock::text(msg.text));
                for source in msg.images {
                    result_blocks.push(ContentBlock::Image { source });
                }
            }
            session.push(Message::tool_results(result_blocks));
            // A tool round-trip just landed: assistant `tool_use` and its matching `tool_result`s are
            // both committed now, so this is a valid, resumable checkpoint (see `CheckpointHook`) — the
            // one mid-run point a crash between here and the run's eventual end would otherwise lose.
            self.checkpoint_guarded(session).await;
            if terminate {
                // A tool requested completion (e.g. an `attempt_completion`/`exit` tool) and the whole
                // batch agreed. The results are already recorded; end the run as if the model had
                // stopped — this wins outright over a pending stop request, which would produce the
                // same outcome anyway.
                sink(AgentEvent::AgentEnd {
                    steps: session.steps,
                });
                return Ok(());
            }
            if steered_count > 0 {
                sink(AgentEvent::Steered {
                    messages: steered_count,
                });
            }
            // A graceful-stop request is honored here too, after this turn's tool results (and any
            // folded-in steer text) are already committed — the same turn-boundary contract as the
            // tool-less branch above. Checked *after* the `Steered` event so a client sees its steer
            // message land in the transcript even if the run stops right after. A content-aware
            // `should_stop_after_turn` hook can request the same stop, now with the actual assistant
            // turn and its tool results available — pi's `shouldStopAfterTurn` receives exactly this
            // pair, at exactly this point in the loop.
            let hook_wants_stop = {
                let messages = &session.messages;
                let assistant_and_results = messages
                    .len()
                    .checked_sub(2)
                    .and_then(|i| messages.get(i))
                    .zip(messages.last());
                match assistant_and_results {
                    Some((assistant_msg, tool_results_msg)) => {
                        catch_tool_panic(self.hooks.should_stop_after_turn(
                            assistant_msg,
                            &tool_results_msg.content,
                            session,
                            &cancel,
                        ))
                        .await
                        .unwrap_or(false)
                    }
                    None => false,
                }
            };
            if steering.take_stop_requested() || hook_wants_stop {
                sink(AgentEvent::AgentEnd {
                    steps: session.steps,
                });
                return Ok(());
            }
        }
    }

    /// Task #28 (pi-parity): the fully-interleaved gate→execute→finalize-per-call path, used for a
    /// turn's *whole* batch of tool calls when any one of them names a tool whose
    /// [`crate::tool::Tool::execution_mode`] returns `Some(Sequential)`. Matches pi's
    /// `agent-loop.ts` `executeToolCallsSequential`: each call is completely resolved — gated (a
    /// malformed-args check, then `before_tool_call`), executed, and rewritten by `after_tool_call` —
    /// before the next call's own gate even starts, unlike [`run_events_steered`](Self::run_events_steered)'s
    /// default path (every call in the batch gated first, *then* executed with bounded concurrency).
    /// This is the seam a concurrency-aware permission policy or rate limiter needs: it can reason
    /// about "what's already run" when gating call N+1, which the default gate-the-whole-batch-first
    /// split can't offer.
    ///
    /// Returns `(results, cancelled_mid_dispatch)` in exactly the shape the default path's own
    /// `results`/`cancelled_mid_dispatch` locals have, so the caller's downstream transcript-assembly
    /// and `repair_cancelled_dispatch` handling is shared unchanged between the two paths: `results[i]`
    /// is `None` only for a call this loop never reached because cancellation cut it short (the same
    /// convention [`resolve_tool_result`] and `repair_cancelled_dispatch` already handle for the default
    /// path).
    ///
    /// `ToolStart` is deliberately *not* emitted here — the caller already emits one for every call in
    /// the batch upfront, before either dispatch path runs, so emitting it again per call here would
    /// double it. Progress streaming (`Tool::run_streaming`) is still fully supported: a still-running
    /// call's updates race against its own completion and cancellation via `select_biased!`, the same
    /// pattern the default path's `drain` loop uses.
    async fn run_tool_calls_interleaved(
        &self,
        session: &Session,
        calls: &[(String, String, Value)],
        malformed: &HashMap<String, String>,
        cancel: &CancellationToken,
        sink: &mut (dyn FnMut(AgentEvent) + Send),
        tools: &ToolRegistry,
    ) -> (Vec<Option<ToolCallResult>>, bool) {
        let mut results: Vec<Option<ToolCallResult>> = vec![None; calls.len()];
        let mut cancelled_mid_dispatch = false;
        // Same hoist as the default gate loop above: the active model is fixed for the whole turn, so
        // this is computed once instead of once per call inside the loop below.
        let supports_vision =
            crate::models::capabilities(&self.model).supports_vision && !self.block_images;
        for (i, (id, name, input)) in calls.iter().enumerate() {
            if cancel.is_cancelled() {
                cancelled_mid_dispatch = true;
                break;
            }
            if let Some(raw) = malformed.get(id) {
                results[i] = Some((
                    format!(
                        "tool call arguments were not valid JSON and could not be parsed: {raw}"
                    ),
                    Vec::new(),
                    true,
                    false,
                ));
                continue;
            }
            // Pi-parity fix: same "look up the tool first" fix as the default gate loop above (see its
            // own doc comment for the pi `agent-loop.ts` citation) — an unregistered tool name resolves
            // straight to the "unknown tool" error result, before coercion or `before_tool_call` run.
            let Some(tool) = tools.get(name) else {
                results[i] = Some((format!("unknown tool: {name}"), Vec::new(), true, false));
                continue;
            };
            // Same pi-parity coercion as the default gate loop, run *before* `before_tool_call` (matches
            // pi's `prepareToolCall`, which calls `validateToolArguments` before `config.beforeToolCall`)
            // so a permission hook sees the same coerced/typed arguments the tool is about to run with,
            // not the model's raw, possibly stringified wire values.
            let mut coerced =
                crate::validation::coerce_tool_arguments(&tool.input_schema(), input.clone())
                    .unwrap_or_else(|_| input.clone());
            if let Some(obj) = coerced.as_object_mut() {
                obj.insert(
                    crate::tool::MODEL_SUPPORTS_VISION_KEY.to_string(),
                    supports_vision.into(),
                );
            }
            let blocked = match catch_tool_panic(
                self.hooks.before_tool_call(name, &coerced, session, cancel),
            )
            .await
            {
                Ok(reason) => reason,
                Err(panic_msg) => Some(panic_msg),
            };
            if let Some(reason) = blocked {
                results[i] = Some((
                    format!("tool call blocked: {reason}"),
                    Vec::new(),
                    true,
                    false,
                ));
                continue;
            }
            // Same Task #3 fix as the default gate loop: re-check cancellation after the hook
            // returns, before this call is actually dispatched.
            if cancel.is_cancelled() {
                cancelled_mid_dispatch = true;
                break;
            }
            let target = tools.get(name).and_then(|t| t.write_target(&coerced));
            // `Arc` so a clone can ride into a tool's non-cancellable `spawn_blocking` write and keep the
            // registry lock held until the write has physically landed, rather than releasing it the
            // instant this interleaved dispatch abandons the tool future on cancellation — see the
            // matching guard in the default group path, and `write_lock.rs`.
            let write_guard = match &target {
                Some(path) => Some(std::sync::Arc::new(self.write_locks.lock(path).await)),
                None => None,
            };
            let (prog_tx, mut prog_rx) =
                futures::channel::mpsc::unbounded::<crate::tool::ToolUpdate>();
            let progress =
                crate::tool::ToolProgress::new(prog_tx, id.clone(), name.clone(), cancel.clone())
                    .with_write_lock(write_guard.clone());
            let run_fut = async {
                match tools.get(name) {
                    Some(tool) => {
                        match catch_tool_panic(tool.run_streaming(coerced, &progress)).await {
                            Ok(Ok(o)) => (o.text, o.images, false, o.terminate),
                            Ok(Err(e)) => (e.to_string(), Vec::new(), true, false),
                            Err(panic_msg) => (panic_msg, Vec::new(), true, false),
                        }
                    }
                    None => (format!("unknown tool: {name}"), Vec::new(), true, false),
                }
            };
            // `select_biased!` requires each polled-across-iterations future to be `FusedFuture` (it
            // can't statically see that this loop always breaks the instant either resolves) — `.fuse()`
            // wraps both one-shot futures accordingly. `prog_rx.next()` needs no such wrapping: it's a
            // fresh future built on every loop iteration, the same pattern the default dispatch path's
            // own `drain` loop already relies on.
            let run_fut = futures::FutureExt::fuse(run_fut);
            futures::pin_mut!(run_fut);
            let cancelled_fut = futures::FutureExt::fuse(cancel.cancelled());
            futures::pin_mut!(cancelled_fut);
            let outcome = loop {
                futures::select_biased! {
                    prog = prog_rx.next() => {
                        if let Some(u) = prog {
                            emit_tool_update(sink, u);
                        }
                    }
                    outcome = &mut run_fut => break Some(outcome),
                    _ = &mut cancelled_fut => break None,
                }
            };
            while let Ok(u) = prog_rx.try_recv() {
                emit_tool_update(sink, u);
            }
            let Some((text, images, is_error, terminate)) = outcome else {
                cancelled_mid_dispatch = true;
                break;
            };
            let (text, images, is_error) = match catch_tool_panic(self.hooks.after_tool_call(
                name,
                input,
                text.clone(),
                images.clone(),
                is_error,
                session,
                cancel,
            ))
            .await
            {
                Ok(rewritten) => rewritten,
                Err(_) => (text, images, is_error),
            };
            sink(AgentEvent::ToolEnd {
                id: id.clone(),
                name: name.clone(),
                result: text.clone(),
                is_error,
            });
            results[i] = Some((text, images, is_error, terminate));
        }
        (results, cancelled_mid_dispatch)
    }

    /// Stream and assemble a single model turn, restarting from scratch when the stream dies mid-flight
    /// (see [`is_retryable_mid_stream`]) rather than surfacing that as a fatal error. Each attempt runs
    /// in [`run_turn_once`] with its own fresh [`Accumulator`] — a retried attempt never resumes a
    /// dead attempt's partial blocks, so the `Turn` this returns can't blend a half-formed tool call
    /// from a failed connection into what actually gets applied to the session. A cancellation always
    /// propagates immediately; only the mid-stream-failure class is retried.
    /// `partial_out` is overwritten by every attempt (successful or not): `None` unless the *last*
    /// attempt made before this call returns failed after real content had already streamed, in which
    /// case it holds that content — see [`run_turn_once`]'s doc comment. A retried attempt always
    /// starts `run_turn_once` fresh, so a dead attempt's partial blocks never leak into a *later*,
    /// successful attempt's slot; only a final, non-retried (or retries-exhausted) failure leaves this
    /// set when this function returns.
    async fn run_turn(
        &self,
        req: ModelRequest,
        emit: &mut (dyn FnMut(StreamEvent) + Send),
        cancel: &CancellationToken,
        partial_out: &mut Option<Turn>,
    ) -> Result<Turn> {
        let mut attempt = 0u32;
        loop {
            match self
                .run_turn_once(req.clone(), emit, cancel, partial_out)
                .await
            {
                Ok(turn) => return Ok(turn),
                Err(e)
                    if self.auto_retry
                        && attempt < MAX_MID_STREAM_RETRIES
                        && is_retryable_mid_stream(&e) =>
                {
                    attempt += 1;
                    // Race the backoff sleep against cancellation too — left unraced, a tripped token
                    // during this wait would sit idle for up to `MID_STREAM_MAX_BACKOFF` before the
                    // next `run_turn_once` call even started racing it.
                    let delay = futures_timer::Delay::new(mid_stream_backoff(attempt));
                    let cancelled = cancel.cancelled();
                    futures::pin_mut!(delay, cancelled);
                    if let Either::Right(((), _)) = select(delay, cancelled).await {
                        return Err(Error::Cancelled);
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// One streaming attempt over a fresh connection. Racing the initial `transport.stream(req)` call
    /// (the HTTP connect + any pre-first-byte retry inside it) and each subsequent `stream.next()`
    /// against `cancel` means a tripped token interrupts even a model that has gone quiet (a blocked
    /// read would otherwise hang for the full idle timeout) or a slow-to-connect gateway; dropping
    /// `stream` (or the connect future, before it resolves) on cancel aborts the underlying HTTP
    /// request rather than waiting it out.
    ///
    /// `partial_out` is reset to `None` up front, then set once — right before returning `Err` — iff a
    /// mid-stream failure struck *after* real content had already accumulated: pi's
    /// `anthropic-messages.ts`/`openai-completions.ts` keep whatever streamed before a failure, tagging
    /// the turn with the error rather than silently discarding it (the same treatment a mid-stream
    /// *cancellation* already gets a few lines below, via the synthetic `Aborted` turn). Without this, a
    /// network blip or in-band provider error partway through a long response loses everything streamed
    /// so far once retries are exhausted — real, already-generated prose or a half-finished edit,
    /// gone — leaving only an empty placeholder in history.
    async fn run_turn_once(
        &self,
        mut req: ModelRequest,
        emit: &mut (dyn FnMut(StreamEvent) + Send),
        cancel: &CancellationToken,
        partial_out: &mut Option<Turn>,
    ) -> Result<Turn> {
        *partial_out = None;
        // The request-side half of the before/after-provider-request pair (see
        // `AgentHooks::before_provider_request`'s own doc comment) — called once per attempt, including
        // every mid-stream retry (`run_turn` calls this fresh each time), immediately before the
        // request reaches the transport. A panicking hook discards its own (possibly partial) mutation
        // and falls back to the request exactly as it was, the same "fails open" convention
        // `on_assistant_message` already uses.
        let before_hook = req.clone();
        if catch_tool_panic(self.hooks.before_provider_request(&mut req))
            .await
            .is_err()
        {
            req = before_hook;
        }
        let cancelled = cancel.cancelled();
        futures::pin_mut!(cancelled);
        let mut stream = {
            let stream_fut = self.transport.stream(req);
            futures::pin_mut!(stream_fut);
            match select(stream_fut, cancelled.as_mut()).await {
                Either::Left((res, _)) => res?,
                // Cancelled before the request even connected — nothing was ever sent to the model,
                // so there's no partial content worth persisting (unlike the mid-stream case below).
                // A genuine `Err` here, not a synthetic `Aborted` turn.
                Either::Right(((), _)) => return Err(Error::Cancelled),
            }
        };
        let mut acc = Accumulator::default();
        loop {
            let next = stream.next();
            futures::pin_mut!(next);
            match select(next, cancelled.as_mut()).await {
                Either::Left((Some(Err(e)), _)) => {
                    // A mid-stream transport/decode failure — unlike cancellation just below, this
                    // isn't a synthetic `Ok(Turn)` (the retry-classification functions need the real
                    // `Err` to decide whether `run_turn` retries), but whatever content had already
                    // accumulated is still worth keeping for the caller to attach to the eventual
                    // terminal record instead of losing it outright.
                    *partial_out = Some(acc.finish());
                    return Err(e);
                }
                Either::Left((Some(Ok(ev)), _)) => {
                    // `apply` only borrows, so `ev` is still ours to move into `emit` afterward — no
                    // clone needed on this per-delta hot path (see `Accumulator::apply`'s doc comment).
                    acc.apply(&ev);
                    emit(ev);
                }
                Either::Left((None, _)) => break,
                // Cancelled after some content had already streamed — `finish()` flushes whatever's
                // still open in `acc` (pi-parity: matches `abort.test.ts`'s "aborted mid-stream keeps
                // partial content" contract). Returned as `Ok` with a synthetic `Aborted` stop reason,
                // not `Err`, so `run_events_steered` still gets to persist it exactly like any other
                // turn before that caller re-raises the cancellation — see its `StopReason::Aborted`
                // arm for why the external `Err(Error::Cancelled)` contract is preserved either way.
                Either::Right(((), _)) => {
                    let mut turn = acc.finish();
                    turn.stop_reason = StopReason::Aborted;
                    return Ok(turn);
                }
            }
        }
        Ok(acc.finish())
    }

    /// [`run_turn`](Self::run_turn), for a one-off "utility" model call (compaction/branch-summary)
    /// that isn't the main conversational loop — every caller of `run_turn` in this file *except*
    /// `run_events_steered` should go through this instead of calling `run_turn` directly.
    ///
    /// The distinction matters because of `run_turn`'s own cancellation contract: a mid-stream cancel
    /// returns `Ok(Turn { stop_reason: Aborted, .. })`, not `Err(Error::Cancelled)` — deliberately, so
    /// `run_events_steered` can persist whatever partial content streamed before persisting nothing at
    /// all (see that `Aborted` arm's own doc comment). A utility call has no such persistence story —
    /// `compact`/`summarize_branch` only care about the finished summary text, and their own callers
    /// (`serve.rs`'s `switch_branch`, matching pi's `abortBranchSummary`) expect a genuine cancellation
    /// to surface as a hard stop with *nothing* recorded, not silently succeed with whatever fragment
    /// of prose had streamed before the abort landed. Without this, a cancelled compaction/branch-
    /// summary call would be indistinguishable from "the model summarized to an empty string" — the
    /// caller falls through to "nothing worth recording" and proceeds as if cancellation never
    /// happened, instead of the caller's own cancellation-handling arm ever running.
    async fn run_utility_turn(
        &self,
        req: ModelRequest,
        cancel: &CancellationToken,
    ) -> Result<Turn> {
        // A utility call (compaction/branch-summary) has no session to attach partial content to on
        // failure — only the finished summary text ever matters to its caller — so the partial turn a
        // mid-stream failure might leave behind is simply discarded here, unlike the main conversational
        // loop's own call to `run_turn`.
        let mut discarded_partial = None;
        let turn = self
            .run_turn(req, &mut |_| {}, cancel, &mut discarded_partial)
            .await?;
        if turn.stop_reason == StopReason::Aborted {
            return Err(Error::Cancelled);
        }
        Ok(turn)
    }

    /// Summarize the conversation prefix in place, keeping the recent turns verbatim. Makes one
    /// summarization model call (silently — its tokens aren't surfaced as assistant output), splices
    /// the summary into `session`, folds this round's file-ops into `session.compaction` (see
    /// [`CompactionProvenance`]), and emits an [`AgentEvent::Compacted`]. Returns
    /// [`CompactOutcome::TooSmall`]/[`CompactOutcome::AlreadyCompacted`] (both no-ops) when there's no
    /// worthwhile prefix to summarize or the model returns an empty summary — see [`CompactOutcome`]'s
    /// own doc comment for exactly which of the two each case maps to. Exposed so a headless server can
    /// offer a manual `compact` command (pass [`CompactionReason::Manual`]).
    ///
    /// `custom_instructions`, when given, steers *what* the summary emphasizes (see
    /// [`compaction::summary_request`]'s doc comment) — a manual compaction's client-supplied focus.
    /// An automatic trigger ([`CompactionReason::Threshold`]/[`CompactionReason::Overflow`]) has no
    /// client in the loop to ask, so it always passes `None`.
    /// Apply this agent's configured reasoning (extended-thinking budget / reasoning-effort level) to
    /// `req`, the same way the main turn loop does for every live-conversation request (see
    /// `run_events_steered`'s own `with_thinking`/`with_reasoning_effort` calls). Shared by
    /// [`Self::compact`] and [`Self::summarize_branch`]'s summarization requests: without this, a
    /// summarization call always ran at the model's bare default reasoning level regardless of what
    /// the live session was actually configured to use, since `compaction::summary_request`/
    /// `branch_summary::branch_summary_request` build a plain `ModelRequest` with no reasoning
    /// knowledge of their own — this crate's config lives only on `Agent`. Forwards the *same* level
    /// the live conversation uses (not a fixed lower one) — both fields are already model-appropriate
    /// by construction (only ever set via `with_thinking`/`with_reasoning_effort` for a model that
    /// actually supports them), and the summarization call always targets that same model.
    fn with_reasoning(&self, mut req: ModelRequest) -> ModelRequest {
        if let Some(budget) = self.thinking {
            req = req.with_thinking(budget);
        }
        if let Some(effort) = self.reasoning_effort {
            req = req.with_reasoning_effort(effort);
        }
        req
    }

    pub async fn compact(
        &self,
        session: &mut Session,
        reason: CompactionReason,
        cancel: &CancellationToken,
        sink: &mut (dyn FnMut(AgentEvent) + Send),
        custom_instructions: Option<&str>,
    ) -> Result<CompactOutcome> {
        // `compact` is a public entry point in its own right (the manual `compact` RPC command calls
        // it directly, not just `run_events_steered`'s own auto-compaction path), so it needs its own
        // copy of the same fails-open `sink` guard `run_events_steered` wraps at its top — see that
        // wrap's doc comment. Harmless to re-wrap when this *is* reached through `run_events_steered`
        // (via `compact_or_report`): the inner layer already caught, so the outer one never has
        // anything left to catch.
        let mut sink = move |ev: AgentEvent| catch_sink_panic(|| sink(ev));
        let Some(cut) =
            compaction::find_split_cut(&session.messages, self.compaction.keep_recent_tokens)
        else {
            return Ok(CompactOutcome::TooSmall);
        };
        // Nothing new worth folding in since the last compaction: on a clean boundary, once a prior
        // summary occupies `messages[0]` (`apply_summary` always splices it in there), `find_cut`'s
        // backward token walk over `messages[1..]` only ever lands on a real cut by hitting the
        // `keep_recent_tokens` budget early; if that walk instead exhausts the whole suffix without
        // ever reaching budget, everything since the last summary already fits under the recent-token
        // window on its own — there's no old-enough content yet to be worth a fresh summarization
        // call. This is the clean-boundary analog of the `turn_start == 1` reuse a few lines below for
        // the split-turn case. Matches pi's own `prepareCompaction` skip condition ("should skip
        // repeated compactions when kept messages still fit") — broader than just `first_kept == 1`
        // (literally zero new messages): e.g. `[summary, user, assistant, user, assistant]` under the
        // real default `keep_recent_tokens` lands `first_kept == 2`, which a narrower `== 1` check
        // would miss even though nothing here is large enough to justify re-summarizing yet (repeated
        // manual `compact` calls with little new in between are the main way this is reachable — an
        // *automatic* trigger can't, since `apply_summary` also resets `last_input_tokens` to 0 and
        // `should_compact` requires it to be positive to fire at all).
        if cut.turn_start.is_none()
            && compaction::previous_summary(&session.messages[..1]).is_some()
        {
            // `first_kept == 1` on a clean boundary means the prefix handed to the summarizer is
            // *exactly* the prior summary and nothing else — so the "fresh" summary is a summary of a
            // summary, `apply_summary` splices back a list of identical length, and the round is pure
            // loss: one paid model call, zero tokens freed, and (because a summary is already bounded
            // by `summary_max_tokens`) no realistic shrink either. Unlike the broader budget check
            // below this holds *regardless* of how much has landed since — the content since the
            // summary is all in the kept suffix, which this cut doesn't touch. Bailing here, before
            // `CompactionStart` is even emitted, is what keeps `Compacted` an honest signal that the
            // caller made progress; the overflow-retry paths in `run_events_steered` branch on exactly
            // that and would otherwise re-prompt the identical request forever.
            if cut.first_kept == 1 {
                return Ok(CompactOutcome::AlreadyCompacted);
            }
            let since_last_summary: u32 = session.messages[1..]
                .iter()
                .map(compaction::estimate_message_tokens)
                .fold(0u32, |acc, n| acc.saturating_add(n));
            if since_last_summary < self.compaction.keep_recent_tokens {
                return Ok(CompactOutcome::AlreadyCompacted);
            }
        }
        sink(AgentEvent::CompactionStart { reason });
        let first_kept = cut.first_kept;
        let before = session.messages.len();
        let tokens_before = session.last_input_tokens;
        let prefix: Vec<Message> = session.messages[..first_kept].to_vec();
        let file_ops = compaction::merge_provenance(&session.compaction, &prefix, reason);

        let summary = match cut.turn_start {
            None => {
                // Clean boundary: unchanged, single-call path.
                let req = self.with_reasoning(compaction::summary_request(
                    &self.model,
                    &prefix,
                    self.compaction.summary_max_tokens,
                    &file_ops,
                    custom_instructions,
                ));
                turn_text(&self.run_utility_turn(req, cancel).await?)
            }
            Some(turn_start) => {
                // Split turn: summarize the closed-off history and the in-progress turn's own prefix
                // separately, then stitch them together — rather than collapsing both under the
                // split-turn template, which is written for a partial turn, not a whole conversation's
                // worth of already-completed ones. The two calls run *sequentially*, not concurrently
                // (pi originally ran them concurrently via `Promise.all`, then fixed exactly this in its
                // own 13-commit-ahead history: "serialize split-turn compaction summaries... so
                // single-concurrency local providers do not fail with 429 errors" — a self-hosted/local
                // model behind a one-request-at-a-time server rejects the second of two simultaneous
                // completions).
                // Off the raw `reserve_tokens`, not `summary_max_tokens` (already `0.8 *
                // reserve_tokens`) — matches pi's own `generateTurnPrefixSummary`
                // (`Math.floor(0.5 * reserveTokens)`, `compaction.ts`), rather than compounding the two
                // scale factors into an effective ~0.4 * reserve_tokens.
                let turn_prefix_max_tokens = ((self.compaction.reserve_tokens as f64)
                    * compaction::SPLIT_TURN_PREFIX_SCALE)
                    as u32;
                let history = async {
                    if turn_start == 0 {
                        // Nothing precedes the split turn — no history to summarize, no call to make.
                        return Ok("No prior history.".to_string());
                    }
                    // The closed-off side is nothing but a still-current prior summary (a compaction's
                    // summary message is always `session.messages[0]`; `find_split_cut` only ever bumps
                    // `turn_start` to exactly 1 when the scan found no history beyond it) — no new
                    // activity happened before this split turn began, so reuse it verbatim instead of
                    // spending a model call to ask for what would be an unchanged restatement.
                    if turn_start == 1 {
                        if let Some(prev) =
                            compaction::previous_summary(&session.messages[..turn_start])
                        {
                            return Ok(prev.to_string());
                        }
                    }
                    let req = self.with_reasoning(compaction::summary_request(
                        &self.model,
                        &session.messages[..turn_start],
                        self.compaction.summary_max_tokens,
                        &file_ops,
                        custom_instructions,
                    ));
                    Ok::<_, Error>(turn_text(&self.run_utility_turn(req, cancel).await?))
                };
                // Unlike `history` above, the turn-prefix call never takes `custom_instructions` — it
                // summarizes the SPLIT_TURN_INSTRUCTION template's "context for the retained suffix,"
                // not the closed-off conversation `custom_instructions` is meant to steer. Matches pi's
                // `generateTurnPrefixSummary`, whose signature doesn't accept custom instructions at all.
                let turn_prefix = async {
                    let req = self.with_reasoning(compaction::summary_request(
                        &self.model,
                        &session.messages[turn_start..first_kept],
                        turn_prefix_max_tokens,
                        &file_ops,
                        None,
                    ));
                    Ok::<_, Error>(turn_text(&self.run_utility_turn(req, cancel).await?))
                };
                let history = history.await?;
                let turn_prefix = turn_prefix.await?;
                compaction::merge_split_summary(&history, &turn_prefix)
            }
        };
        if summary.trim().is_empty() {
            // A genuine summarization call ran (unlike the two early no-ops above, which never call the
            // model at all) but came back blank — functionally the same "nothing new to say" no-op as
            // the clean-boundary case above, just discovered a call later.
            return Ok(CompactOutcome::AlreadyCompacted);
        }
        // Append the carried provenance deterministically rather than trusting the summarizing model to
        // have preserved it in its own prose — see `compaction`'s module doc on the deterministic-carry
        // channel. Both blocks come from `file_ops` (this round's already-merged provenance), so they
        // reflect every round so far, not just this one's new activity. `previous_summary` strips these
        // same blocks back off before the body is ever fed forward, so they never accumulate.
        let summary = format!(
            "{summary}{}{}{}",
            compaction::format_file_operations(&file_ops.read_files, &file_ops.modified_files),
            compaction::format_todo_list(file_ops.todos.as_ref()),
            compaction::format_memory_notes(&file_ops.memory_notes)
        );
        compaction::apply_summary(session, first_kept, &summary, tokens_before);
        session.compaction = file_ops;
        // Computed from the freshly-rebuilt list (summary + kept suffix), not `trailing_tokens` — that
        // one measures a delta since the last real usage snapshot, which `apply_summary` just reset to
        // point past this very list's end, so it would report a meaningless 0 here.
        let tokens_after = compaction::estimate_messages_tokens(&session.messages);
        // The rewrite just replaced (typically most of) the session's history with the new summary —
        // this already spent one or two real, paid-for model calls to produce. Checkpoint immediately:
        // without this, a crash between the rewrite landing here and the next natural checkpoint (which
        // may be turns away, or may never come if the run ends on a tool-less reply) loses the rewrite
        // entirely — on resume the persisted session still holds the old, oversized history, so the
        // very next turn immediately re-triggers the identical (already-paid-for) compaction again, and
        // if this was overflow-recovery, the resumed session lands right back in the same
        // context-overflow condition it just paid to escape.
        self.checkpoint_guarded(session).await;
        sink(AgentEvent::Compacted {
            messages_before: before,
            messages_after: session.messages.len(),
            reason,
            tokens_before,
            summary,
            tokens_after,
            first_kept,
        });
        Ok(CompactOutcome::Compacted)
    }

    /// Persist the session through the host's [`CheckpointHook`], containing a panic in the host's own
    /// persistence code instead of letting it destroy the run.
    ///
    /// Every other host-supplied seam in this loop — every `AgentHooks` method, the event `sink` — is
    /// already wrapped in [`catch_tool_panic`]/`catch_sink_panic` and fails open (or, for the approval
    /// gate, deliberately closed). `checkpoint` was the one that wasn't, and it is the seam *most*
    /// likely to touch failing I/O: its own trait doc invites the host to do blocking work there
    /// ("appending to a session file"), so a full disk or an `EACCES` in a host persistence path that
    /// unwraps would unwind straight through the agent loop and take the whole run — in `serve`, the
    /// whole session task — down with it. A checkpoint is an optimization (it bounds what a crash
    /// loses); failing to take one must degrade to "this checkpoint didn't persist", never to "the run
    /// died". Logged at `error` because a host whose persistence is panicking genuinely wants to know.
    async fn checkpoint_guarded(&self, session: &Session) {
        if let Err(msg) = catch_tool_panic(self.checkpoint.checkpoint(session)).await {
            tracing::error!(
                error = %msg,
                "checkpoint hook panicked; continuing without persisting this checkpoint"
            );
        }
    }

    /// Runs an *automatic* (proactive-threshold or hard-overflow) compaction and swallows a failure
    /// rather than propagating it — mirrors pi's `_runAutoCompaction`, which wraps the equivalent call
    /// in `try/catch` and emits `compaction_end { errorMessage }` on failure instead of letting it
    /// abort the run. Without this, a single transient summarization-call failure (network blip,
    /// provider hiccup) would end the whole run via the `?` in the caller — and since `should_compact`
    /// re-fires on every subsequent turn until it succeeds, a persistently failing summarizer would
    /// make the session permanently unusable, blocking the user's *own* prompt from ever reaching the
    /// model. A manually invoked `compact()` (the `compact` RPC command) is unaffected — that caller
    /// still sees the real error and can decide what to do.
    ///
    /// A genuine `Error::Cancelled` is re-raised rather than swallowed — a cancellation is the user
    /// asking the whole run to stop, not a compaction-specific fault, so it must still unwind the loop
    /// the normal way instead of falling through to send the very turn the user just cancelled.
    async fn compact_or_report<F>(
        &self,
        session: &mut Session,
        reason: CompactionReason,
        cancel: &CancellationToken,
        sink: &mut F,
    ) -> Result<()>
    where
        F: FnMut(AgentEvent) + Send,
    {
        match self.compact(session, reason, cancel, sink, None).await {
            Ok(_) => Ok(()),
            Err(Error::Cancelled) => Err(Error::Cancelled),
            Err(e) => {
                sink(AgentEvent::CompactionFailed {
                    reason,
                    message: e.to_string(),
                });
                Ok(())
            }
        }
    }

    /// Summarize an abandoned tree branch's messages (Track L2/L3: a headless server calls this from
    /// its branch-navigation command, on messages its session store's `abandoned_by_switch` returned,
    /// *before* actually switching branches). The network-calling half of branch summarization; the
    /// pure request-building lives in [`crate::branch_summary::branch_summary_request`], and
    /// persisting the result is the caller's job — this only returns the summary text, mirroring
    /// [`Self::compact`]'s network/storage split but without touching `session` (a branch summary
    /// doesn't rewrite the *active* conversation the way a compaction summary does).
    ///
    /// `custom_instructions`, when given, steers *what* the branch recap emphasizes — the same
    /// "Additional focus: {custom_instructions}" framing [`compaction::summary_request`] uses for a
    /// manual compaction (see [`crate::branch_summary::branch_summary_request`]'s doc comment) — a
    /// client-supplied focus threaded down from the `switch_branch` RPC command's own optional field.
    ///
    /// `replace_instructions` (Task #17, pi-parity) forwards straight through to
    /// [`crate::branch_summary::branch_summary_request`]'s own parameter of the same name: `true` uses
    /// `custom_instructions` as the *entire* instruction section instead of appending it after the
    /// default structured template — see that function's doc comment for the exact semantics (a no-op
    /// when `custom_instructions` is `None`). A headless server's `switch_branch` RPC handler is
    /// expected to thread its own caller-facing field through to this parameter.
    pub async fn summarize_branch(
        &self,
        messages: &[Message],
        cancel: &CancellationToken,
        custom_instructions: Option<&str>,
        replace_instructions: bool,
    ) -> Result<String> {
        // The same shape of input budget compaction sizes its own summarization calls against — the
        // model's context window minus a reserved headroom — so an abandoned branch's rendered
        // transcript can't overflow the summarization call's own window any more than a compaction
        // summary's can. The reserve itself, though, is `branch_summary_reserve_tokens` when a caller
        // set one (independent of the live conversation's own compaction reserve — see that field's
        // doc comment), falling back to `compaction.reserve_tokens` otherwise, today's exact behavior.
        let reserve_tokens = self
            .branch_summary_reserve_tokens
            .unwrap_or(self.compaction.reserve_tokens);
        let input_token_budget = self
            .compaction
            .context_window
            .saturating_sub(reserve_tokens);
        let req = self.with_reasoning(crate::branch_summary::branch_summary_request(
            &self.model,
            messages,
            crate::branch_summary::BRANCH_SUMMARY_MAX_TOKENS,
            input_token_budget,
            custom_instructions,
            replace_instructions,
        ));
        let turn = self.run_utility_turn(req, cancel).await?;
        let summary: String = turn
            .blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        // Same deterministic append `compact` does — see `format_file_operations`'s doc comment. Files
        // are re-extracted from the same `messages` `branch_summary_request` itself already scanned
        // (cheap: `extract_file_ops` doesn't touch tool-result bodies, just tool-call names/inputs),
        // since that request's own file lists aren't otherwise surfaced back to this caller.
        let (read, modified) = compaction::extract_file_ops(messages);
        Ok(format!(
            "{summary}{}",
            compaction::format_file_operations(&read, &modified)
        ))
    }
}

/// The concatenated text blocks of a summarization turn — a summary is always plain prose, so anything
/// else the model emitted (there shouldn't be any; the summarization system prompt asks for none) is
/// simply not text and is dropped here rather than erroring.
///
/// `turn.stop_reason == StopReason::Aborted` never reaches here — see [`Agent::run_utility_turn`],
/// every caller's actual entry point.
fn turn_text(turn: &Turn) -> String {
    turn.blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Turn one [`ToolUpdate`](crate::tool::ToolUpdate) from the per-turn progress channel into the
/// matching [`AgentEvent`] and emit it — shared by the live-arriving path and the final flush of
/// whatever was still buffered when the group stream finished.
fn emit_tool_update(sink: &mut (dyn FnMut(AgentEvent) + Send), update: crate::tool::ToolUpdate) {
    match update {
        crate::tool::ToolUpdate::Progress {
            id,
            name,
            snapshot,
            details,
        } => {
            sink(AgentEvent::ToolProgress {
                id,
                name,
                snapshot,
                details,
            });
        }
        crate::tool::ToolUpdate::End {
            id,
            name,
            result,
            is_error,
        } => {
            sink(AgentEvent::ToolEnd {
                id,
                name,
                result,
                is_error,
            });
        }
    }
}

/// Run `fut` with a panic boundary: a panic inside it (a hook's own bug, or a tool implementation's
/// stray `.unwrap()`/index-out-of-bounds/debug-mode overflow) is caught and turned into an `Err`
/// message instead of unwinding through the whole tool-dispatch future — and, since dispatch is only
/// ever awaited from inside `run_events_steered`'s own future, through the entire in-flight run.
///
/// Pi-parity gap: pi's `agent-loop.ts` wraps every user-extensible call (`beforeToolCall`, a tool's own
/// `execute`, `afterToolCall`) in try/catch, degrading a buggy hook/tool to one failed call instead of
/// killing the run. Neither `AgentHooks`'s trait methods nor `Tool::run_streaming` return a `Result` a
/// panic could be redirected through instead — a caller can't opt out of this by returning an error, so
/// this is one of the places that needs to actually catch the unwind. `pub(crate)`: `client.rs` reuses it
/// for the same "fails open" treatment of [`crate::hooks::AgentHooks::before_provider_payload`], one layer
/// below where this loop lives.
///
/// `AssertUnwindSafe`: the futures here borrow `&self`/`&CancellationToken`/`&Value`/`&str` — all
/// either `Copy`, plain data, or (for a caller's own hook/tool trait object) opaque behind `&dyn Trait`,
/// none of which this crate can prove `UnwindSafe` through a generic bound, but a torn-mid-panic
/// *read-only* borrow of them is exactly the case `UnwindSafe`'s conservative default is overly cautious
/// about (see the `std::panic` module docs on "exception safety") — there is no interior mutability this
/// crate's own state relies on being consistent across the unwind (a hook/tool with its own is the
/// author's responsibility, same as it would be for any other panic in their code).
pub(crate) async fn catch_tool_panic<F, T>(fut: F) -> std::result::Result<T, String>
where
    F: std::future::Future<Output = T>,
{
    use futures::FutureExt;
    std::panic::AssertUnwindSafe(fut)
        .catch_unwind()
        .await
        .map_err(|payload| format!("panicked: {}", panic_message(payload)))
}

/// Shared by [`catch_tool_panic`] and [`catch_sink_panic`]: turns a `catch_unwind` payload into a
/// human-readable message. `panic!`'s two common payload shapes are `&'static str` (a string-literal
/// message, the overwhelmingly common case) and `String` (a formatted one, e.g. via `panic!("{e}")`);
/// anything else (a caller doing `std::panic::panic_any(42)`) has no sensible rendering.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panicked with a non-string payload".to_string())
}

/// Sync sibling of [`catch_tool_panic`], for the one interception point in this file that's a plain
/// call rather than a future: the caller-supplied event `sink`, invoked directly (no `Result` to
/// redirect a failure through) at every streamed event, tool boundary, turn boundary, and compaction
/// milestone. Same fails-open rationale as every hook `catch_tool_panic` already guards — a bug in a
/// UI/log callback wired up purely to *observe* the run shouldn't be able to unwind through and kill
/// the run itself. There's nothing for this crate's own logic to do with a dropped event beyond note
/// it happened: logged, then swallowed, and the run proceeds as if the sink had simply declined to act
/// on that one event.
///
/// `AssertUnwindSafe`: same reasoning as `catch_tool_panic`'s own doc comment, just for a plain closure
/// call instead of a future.
fn catch_sink_panic(f: impl FnOnce()) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        tracing::warn!(
            panic = %panic_message(payload),
            "event sink panicked; dropping this event"
        );
    }
}

/// Resolve a per-call dispatch result, synthesizing an error placeholder for a call whose group never
/// got to run — only reachable via [`repair_cancelled_dispatch`], when cancellation aborts the batch
/// before every group's future resolved.
///
/// Confirmed intentional (not a gap): pi's own `bash.ts` finalizes its output accumulator on abort and
/// returns `"<partial output>\n\nCommand aborted"`, so a cancelled `bash` call specifically keeps
/// whatever had streamed so far. Cancellation here works at the *dispatch* layer instead — the loop
/// drops the whole group's future (`select(drain, cancelled)` in the caller), which is what actually
/// kills a `bash` subprocess via its `kill_on_drop`/process-group guard — so by the time this generic
/// placeholder runs, the tool's own in-flight state (any partial output it may have captured) is
/// already gone with the dropped future; there's nothing tool-specific left to recover here. This
/// applies uniformly to every tool, not just `bash`, which is the point: one dispatch-layer mechanism
/// handles cancellation safely for all current and future tools, rather than requiring each one to
/// cooperatively watch a cancellation signal and preserve its own partial state.
/// Hard ceiling, in bytes, on the tool-result content this loop will admit into a session.
///
/// Until this existed the only bound on tool output was a *per-tool convention* — the built-ins cap
/// themselves at `tools::output::DEFAULT_MAX_BYTES` (50 KiB) — and a `Tool` is a public trait that
/// hosts and MCP servers implement. Nothing stopped an implementation from handing back 200 MB, and
/// nothing downstream would have clipped it: `compaction`'s `TOOL_RESULT_MAX_CHARS` truncates only
/// what is rendered *into the summarization prompt*, while the raw content stays in `session.messages`,
/// is `Arc`-cloned into every subsequent `ModelRequest`, and is serialized on every checkpoint. That
/// makes the memory bound a property of every tool author rather than of the system, which is the
/// wrong place for it.
///
/// An order of magnitude above what a compliant tool emits, so this never clips one in practice — it
/// is a backstop against a rogue or buggy implementation, not a second opinion on how much a tool
/// should return.
const TOOL_RESULT_MAX_BYTES: usize = 1024 * 1024;

/// Clamp one tool result to [`TOOL_RESULT_MAX_BYTES`], telling the model what happened rather than
/// silently handing it a truncated document it would reason over as if complete.
fn cap_tool_result(content: String) -> String {
    if content.len() <= TOOL_RESULT_MAX_BYTES {
        return content;
    }
    let dropped = content.len() - TOOL_RESULT_MAX_BYTES;
    // `String` is UTF-8 and truncation is by bytes, so walk back to the nearest boundary — slicing
    // mid-codepoint would panic, and this content is arbitrary tool output, not ASCII by assumption.
    let mut end = TOOL_RESULT_MAX_BYTES;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = content;
    out.truncate(end);
    out.push_str(&format!(
        "\n\n[tool result truncated: {dropped} more bytes exceeded the {TOOL_RESULT_MAX_BYTES}-byte limit]"
    ));
    out
}

fn resolve_tool_result(result: Option<ToolCallResult>) -> ToolCallResult {
    result.unwrap_or_else(|| {
        (
            "cancelled: tool call aborted before it finished".to_string(),
            Vec::new(),
            true,
            false,
        )
    })
}

/// A cancellation that pre-empted the turn before the model produced anything — the top-of-loop check
/// in [`Agent::run_events_steered`], or `run_turn`'s own `Err(Cancelled)` (reachable before a single
/// byte streams back, unlike a mid-stream abort, which resolves to `Ok(Turn { stop_reason: Aborted,
/// .. })` instead and is handled by the ordinary turn-commit path) — leaves `session`'s last message as
/// the caller's own unanswered `user` turn. Every path that loops back to the top of that loop does so
/// with a `user`-role message last (the initial prompt, a tool-results turn, or a drained follow-up),
/// so this is reachable on the very first turn of a call and on a later one alike.
///
/// Left as-is, a later `prompt` on the same session would push a *second* consecutive `user` message —
/// a shape no dialect accepts. Close it out with the same aborted, empty-content assistant record the
/// mid-stream cancellation path already produces: pi's own `StreamFn` contract requires even a
/// never-streamed request to resolve to a final `stopReason: "aborted"` message for exactly this reason
/// (`abort.test.ts`'s `testImmediateAbort`). A no-op if the last message is somehow already
/// assistant-role (defensive; not reachable through the current loop structure, but cheap to guard).
fn close_out_pending_cancellation(session: &mut Session, model: &str) {
    if session
        .messages
        .last()
        .is_some_and(|m| m.role == Role::User)
    {
        session.push(
            Message::assistant(vec![ContentBlock::text(String::new())])
                .with_model_id(model)
                .with_aborted(),
        );
    }
}

/// The cancellation exit every path in `run_events_steered` that can observe `Error::Cancelled`
/// mid-loop must use: closes the session on a clean `aborted` record ([`close_out_pending_cancellation`])
/// and clears the steer/follow-up/switch state scoped to this run ([`Steering::clear_run_scoped`]).
/// A cancellation racing a mid-loop `compact()` call used to skip both — leaving the session on a
/// dangling `user` turn (a shape no dialect accepts) and leaking stale steering state into whatever
/// run reuses this `Steering` handle next — because only the direct turn-cancellation arm called
/// this pair; every `compact()`/`compact_or_report()` call site needs the same treatment.
fn cancel_run(session: &mut Session, model: &str, steering: &Steering) -> Error {
    close_out_pending_cancellation(session, model);
    steering.clear_run_scoped();
    Error::Cancelled
}

/// Route a `compact()`/`compact_or_report()` result through [`cancel_run`] when it's a cancellation,
/// so the caller's `?` can't silently propagate `Error::Cancelled` past the session/steering cleanup
/// every other cancellation exit in this loop performs. A non-cancellation error passes through
/// untouched.
fn finish_compaction<T>(
    result: Result<T>,
    session: &mut Session,
    model: &str,
    steering: &Steering,
) -> Result<T> {
    result.map_err(|e| {
        if matches!(e, Error::Cancelled) {
            cancel_run(session, model, steering)
        } else {
            e
        }
    })
}

/// Synthesize an error `tool_result` for every call whose group never finished (dispatch was
/// cancelled mid-batch), preserving any that did finish, and commit the message — so a run cancelled
/// mid-dispatch always leaves the session in a valid, resumable alternating shape instead of an
/// orphaned `tool_use` with no matching `tool_result`.
fn repair_cancelled_dispatch(
    session: &mut Session,
    calls: &[(String, String, Value)],
    results: Vec<Option<ToolCallResult>>,
) {
    let blocks = calls
        .iter()
        .zip(results)
        .map(|((id, ..), result)| {
            let (content, images, is_error, _) = resolve_tool_result(result);
            ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content,
                is_error,
                images,
            }
        })
        .collect();
    session.push(Message::tool_results(blocks));
}

/// Substrings seen in a *throttling* rejection that would otherwise false-positive against
/// [`OVERFLOW_PATTERNS`]' broader entries (Bedrock's "Too many tokens, please wait before trying
/// again" contains "too many tokens" but means "slow down", not "shrink the prompt") — checked first,
/// so any of these vetoes an overflow match regardless of what else the message contains.
const THROTTLE_EXCLUSIONS: &[&str] = &[
    "throttlingexception",
    "throttling error",
    "please wait before trying again",
    "service unavailable",
    "rate limit",
    "too many requests",
];

/// Phrases seen across providers when a request is rejected for exceeding the model's context window.
/// Provider-specific (not exhaustive of every provider this agent could route through, but every one
/// pi's own catalogue covers) — the trailing compound `context`+`{long,exceed,maximum,window}` check
/// in [`is_context_overflow`] remains as a generic catch-all beneath this table.
const OVERFLOW_PATTERNS: &[&str] = &[
    // Anthropic
    "prompt is too long",
    "request_too_large",
    // Amazon Bedrock
    "input is too long for requested model",
    // OpenAI / OpenAI-compatible proxies
    "exceeds the context window",
    "exceeds the model's maximum context length",
    "exceeds model's maximum context length",
    // Google Gemini
    "exceeds the maximum number of tokens allowed",
    // xAI
    "maximum prompt length is",
    // Groq
    "reduce the length of the messages",
    // OpenRouter
    "maximum context length is",
    "exceeds the maximum allowed input length of",
    // Together AI
    "is longer than the model's context length",
    // Mistral
    "too large for model with",
    // Cerebras — pi matches both with one regex (`^4(?:00|13)\s*(?:status code)?\s*\(no body\)`);
    // this table is substring-only, so both phrasings are listed explicitly instead.
    "400 (no body)",
    "413 (no body)",
    "400 status code (no body)",
    "413 status code (no body)",
    // GitHub Copilot
    "exceeds the limit of",
    // llama.cpp server
    "exceeds the available context size",
    // LM Studio
    "greater than the context length",
    // MiniMax
    "context window exceeds limit",
    // Kimi For Coding
    "exceeded model token limit",
    // DS4 server — "Prompt has 256,468 tokens, but the configured context size is 256,000 tokens";
    // matched on the invariant phrase between the two (comma-formatted) token counts.
    "but the configured context size is",
    // z.ai — normally a silent overflow caught by `compaction::is_hard_overflow` instead (usage vs
    // window, no error raised at all), but its non-standard `finish_reason` can also surface as text.
    "model_context_window_exceeded",
    // Ollama (explicit overflow error; some deployments truncate silently instead)
    "prompt too long; exceeded",
    // Generic / already covered before this table existed
    "too many tokens",
    "context length exceeded",
    "context_length_exceeded",
    "token limit exceeded",
];

/// Whether a transport error is the provider rejecting the request for exceeding its context window —
/// the signal to compact and retry. Matched on the error text (the wire shape varies by provider).
///
/// `pub` (not just internal): a whole-run retry classifier one layer up (`crates/agent::retry`) needs
/// the same exclusion `is_retryable_mid_stream` already applies — an overflow error retried blindly
/// (without compacting first) just fails identically again, wasting a whole-run retry attempt for no
/// benefit.
pub fn is_context_overflow(e: &Error) -> bool {
    let Error::Transport(msg) = e else {
        return false;
    };
    let m = msg.to_ascii_lowercase();
    if THROTTLE_EXCLUSIONS.iter().any(|p| m.contains(p)) {
        return false;
    }
    OVERFLOW_PATTERNS.iter().any(|p| m.contains(p))
        || (m.contains("context")
            && (m.contains("long")
                || m.contains("exceed")
                || m.contains("maximum")
                || m.contains("window")))
}

/// In-band provider error *type* strings (`dialect::sse_error`'s `kind` — Anthropic's `error.type`,
/// OpenAI's `error.type`) seen when a stream dies for a transient, worth-retrying reason. Deliberately
/// **not** raw HTTP status codes ("429", "500", …): those already have a correct, separate retry path
/// (`client.rs`'s pre-first-byte retry, on the real status code) — this function only ever sees
/// mid-stream failures, where the connection already returned 200 and any "500" substring in a message
/// is far more likely to be an unrelated number (a token count, a byte size) than a status code, so
/// matching on it here would risk a false positive. Named error-type substrings carry no such
/// ambiguity. Excludes types that mean "this request will never succeed" (`invalid_request_error`,
/// `authentication_error`, `permission_error`, `insufficient_quota`, …) — those should fail immediately,
/// not eat three silent retries.
const MID_STREAM_RETRYABLE_ERROR_TYPES: &[&str] = &[
    // Anthropic
    "rate_limit_error",
    "api_error",
    "timeout_error",
    // OpenAI / OpenAI-compatible
    "rate_limit_exceeded",
    "server_error",
    "internal_error",
    "service_unavailable",
];

/// Free-text prose fallback for a transient in-band error whose `error.type` field is missing,
/// differently-shaped, or absent entirely (a plain-text/HTML error body, or a provider that doesn't
/// use OpenAI/Anthropic's `type` vocabulary) — pi's `RETRYABLE_PROVIDER_ERROR_PATTERN`
/// (`packages/ai/src/utils/retry.ts`), narrowed to this crate's two dialects. Deliberately excludes
/// everything [`MID_STREAM_RETRYABLE_ERROR_TYPES`]'s doc comment already excludes raw status digits
/// for (the same "500 in a mid-stream message is more likely a token count than a status code" risk
/// applies equally to prose — digits stay whole-run-only, see `agent::retry::WHOLE_RUN_RETRYABLE_STATUS_DIGITS`
/// in the `agent` crate), pi's WebSocket/HTTP2/Bedrock-specific text (this crate's own WebSocket
/// transport, `codex_websocket.rs`, tags its connectivity failures with the same
/// [`MID_STREAM_NETWORK_ERROR`] prefix the patterns below already recognize, rather than needing its
/// own free-text entries here), pi's quota/billing exclusion list (unnecessary here — this function is an *allowlist*
/// of known-retryable shapes, not pi's broad-regex-then-exclude, so an unrecognized quota/billing
/// message simply never matches in the first place), and bare single ambiguous words like "terminated"
/// (too easily a false positive from something unrelated a provider's own error wrapper mentions).
const MID_STREAM_RETRYABLE_FREE_TEXT_PATTERNS: &[&str] = &[
    "rate limit",
    "too many requests",
    "server error",
    "internal error",
    "service unavailable",
    "provider returned error",
    "network error",
    // OpenAI's own literal `finish_reason` spelling (underscore, not a space) — see
    // `dialect::openai::Decoder::finish`, which raises this exact message when a stream's
    // `finish_reason` is `"network_error"`. Kept as a separate entry from `"network error"` above
    // rather than relying on some fuzzier match, since the two are genuinely different literal strings.
    "network_error",
    "connection error",
    // pi's own regression case (earendil-works/pi#3317): a provider-reported "Network connection
    // lost." is a distinct phrasing from both "network error" and "connection error" above — none of
    // the existing entries substring-match it, so this exact transient-disconnect message fell through
    // to a hard failure instead of pi's silent retry-and-recover.
    "connection lost",
    // pi matches these as two separate patterns (`/timed? out/i`, `/timeout/i`) — "timed out" alone
    // missed both "time out" (no `d`) and the bare single-word "timeout" a provider or proxy error
    // wrapper commonly uses (e.g. "ETIMEDOUT", "upstream request timeout").
    "timed out",
    "time out",
    "timeout",
    "ended without",
];

/// Phrases a provider uses to explicitly say "this specific failure is safe to retry" — pi's own
/// retry-guidance patterns (`"you can retry your request"` etc., seen from OpenAI Responses/Bedrock).
const MID_STREAM_RETRY_GUIDANCE_PHRASES: &[&str] = &[
    "you can retry your request",
    "please retry your request",
    "try your request again",
];

/// Whether a transport error is the "stream died after the request already succeeded" class worth
/// restarting the turn for:
/// - a decoder's own truncated-stream rejection (both dialects' `finish()` say "…stream ended
///   before…" — see `dialect::anthropic::Decoder::finish`, `dialect::openai::Decoder::finish`),
/// - an in-band error event whose type is [`MID_STREAM_RETRYABLE_ERROR_TYPES`] (`overloaded_error`
///   checked separately below since it predates the table; `dialect::sse_error` prefixes every in-band
///   error `"provider stream error: "`, matched by substring since the exact wrapping is an
///   implementation detail this shouldn't couple to), or whose *message* prose (not a recognized
///   `error.type`) matches [`MID_STREAM_RETRYABLE_FREE_TEXT_PATTERNS`] or carries one of
///   [`MID_STREAM_RETRY_GUIDANCE_PHRASES`],
/// - or a genuine network failure hitting the response body after it started flowing — a connection
///   reset, a read timeout, an unexpected EOF (tagged [`MID_STREAM_NETWORK_ERROR`] by the transport;
///   see that constant's doc comment for why a literal marker beats re-deriving the classification
///   from a library-specific error's `Display` text).
///
/// A context-overflow rejection is deliberately excluded — that's a *different* signal already
/// handled by compact-and-retry, not this path — and retrying it here would just fail
/// identically-shaped again without compacting first.
///
/// `pub` (not just `run_turn`-internal): the same classification is what should decide whether a whole
/// *run* that ended in `Err` — after this per-turn layer already exhausted its own retries — is worth
/// automatically re-invoking, which is a harness-level (`crates/agent`) concern, not this crate's. An
/// error this function calls retryable looks exactly as transient one level up as it did here; only the
/// case (`Error::MaxSteps`/`Error::Cancelled` already return `false` via the `let else` above) differs
/// by being a legitimate stop, not a fault, either way.
pub fn is_retryable_mid_stream(e: &Error) -> bool {
    let Error::Transport(msg) = e else {
        return false;
    };
    if is_context_overflow(e) {
        return false;
    }
    // A quota/billing-exhausted 429 must not be retried even when its own message also happens to
    // contain a broader retryable-looking phrase — a quota rejection's HTTP status line is routinely
    // the exact same "429 Too Many Requests" wording as an ordinary, transient rate limit, so the
    // `"too many requests"` free-text pattern below would otherwise call it retryable purely from that
    // shared status text, ignoring the body's own `error.type` telling us retrying can never succeed.
    // Checked ahead of every pattern below, the same way `is_context_overflow` already is — both are
    // "this specific failure class always wins over broader ambient wording" exclusions.
    if crate::client::is_quota_exhausted(msg) {
        return false;
    }
    let m = msg.to_ascii_lowercase();
    m.contains("stream ended before")
        || m.contains("overloaded")
        || msg.contains(MID_STREAM_NETWORK_ERROR)
        || MID_STREAM_RETRYABLE_ERROR_TYPES
            .iter()
            .any(|p| m.contains(p))
        || MID_STREAM_RETRYABLE_FREE_TEXT_PATTERNS
            .iter()
            .any(|p| m.contains(p))
        || MID_STREAM_RETRY_GUIDANCE_PHRASES
            .iter()
            .any(|p| m.contains(p))
}

/// Exponential backoff for a mid-stream retry: `MID_STREAM_BASE_BACKOFF · 2^(attempt-1)` (±
/// [`crate::client::jitter`]), capped at `MID_STREAM_MAX_BACKOFF`. `attempt` is 1-based (the first
/// retry backs off by the base amount).
///
/// Finding #32 (pi-parity/consistency fix): this is one of the crate's two *inner* retry layers
/// (alongside `client::backoff`) that fire far more often than the outer, whole-run retry layer
/// (`crates/agent::retry`, a separate crate) — every stream, not just after a whole run has already
/// exhausted both inner layers — making an unjittered exponential here the bigger thundering-herd risk
/// of the two. See [`crate::client::jitter`]'s doc comment for the mechanism and rationale.
fn mid_stream_backoff(attempt: u32) -> Duration {
    let exp_uncapped =
        MID_STREAM_BASE_BACKOFF.saturating_mul(1u32 << attempt.saturating_sub(1).min(16));
    crate::client::jitter(exp_uncapped, attempt).min(MID_STREAM_MAX_BACKOFF)
}

/// Best-effort repair for streamed tool-call JSON that fails to parse on the first attempt, fixing two
/// real-world streaming quirks seen from both dialects rather than giving up immediately: a raw control
/// character inside a string literal that should have been escaped (a large `write`/`edit` argument
/// carrying an embedded literal newline/tab instead of `\n`/`\t`), and a backslash that isn't a valid
/// JSON escape (a Windows path like `C:\Users\x` streamed without escaping its own backslashes). Ported
/// from pi's `repairJson` (`packages/ai/src/utils/json-parse.ts`) — a single pass that only touches
/// bytes *inside* a string literal, so well-formed structural JSON (braces, commas, already-valid
/// escapes) passes through unchanged. Not a full JSON5-style parser: it can't repair a buffer that's
/// merely incomplete (cut off mid-stream with an unclosed brace) — that class falls through to
/// [`close_incomplete_json`] instead, tried next in [`Accumulator::flush_block`]'s fallback chain.
///
/// `pub(crate)`: also reused by [`crate::dialect::push_sse_line`] to repair a malformed *outer*
/// Anthropic SSE event body before its first parse attempt — the same fixture shapes pi's
/// `parseJsonWithRepair` (`packages/ai/src/utils/json-parse.ts`) repairs there, just applied one
/// layer up the stack (a full event body rather than an accumulated `partial_json` string).
pub(crate) fn repair_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if !in_string {
            if c == '"' {
                in_string = true;
            }
            out.push(c);
            continue;
        }
        match c {
            '"' => {
                in_string = false;
                out.push(c);
            }
            '\\' => match chars.next_if(|n| "\"\\/bfnrtu".contains(*n)) {
                // A recognized JSON escape — copy both characters through untouched. (For `\u`, the
                // four hex digits that follow are ordinary characters to this loop and fall through to
                // the `_` arm below unchanged.)
                Some(escape) => {
                    out.push(c);
                    out.push(escape);
                }
                // A stray backslash (not a valid escape lead-in, or the buffer ends right after it) —
                // double it so the parser sees a literal backslash instead of a dangling/invalid escape.
                None => out.push_str("\\\\"),
            },
            // A raw control byte where the stream should have emitted its escape.
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out
}

/// Best-effort recovery for tool-call JSON that's genuinely *incomplete* — cut off mid-stream by an
/// output-token ceiling, not merely mis-escaped (`repair_json`'s job, tried first in
/// [`Accumulator::flush_block`]'s fallback chain). Closes whatever string literal and structural
/// containers (`{`/`[`) were still open when the buffer ended, so a large `write`/`edit` call whose
/// argument value got cut off mid-string (the common real case — the value being streamed, e.g. file
/// content, is typically the longest and thus likeliest field to still be open when a `max_tokens`
/// ceiling hits) recovers a partial object instead of being discarded to `{}` entirely. Ported in
/// spirit from pi's fallback to the `partial-json` library (`json-parse.ts`), hand-rolled here rather
/// than pulling in a dependency for it.
///
/// Not a full JSON5-style parser: doesn't attempt to complete a truncated literal (a bare `tru`, a
/// lone `-`) or synthesize a value for a key truncated before its `:value` ever started — those
/// remain invalid JSON and fall through to [`Accumulator::flush_block`]'s existing malformed-call
/// recovery unchanged, exactly as they did before this function existed. Returns `None` when there's
/// nothing to close (a balanced string that simply failed to parse for some other reason — this
/// function isn't the right tool for that, and returning the input unchanged would just loop the
/// caller back to the same failure).
fn close_incomplete_json(s: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for c in s.chars() {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => stack.push(c),
            '}' if stack.last() == Some(&'{') => {
                stack.pop();
            }
            ']' if stack.last() == Some(&'[') => {
                stack.pop();
            }
            _ => {}
        }
    }
    if !in_string && stack.is_empty() {
        return None; // nothing unclosed — not this function's failure mode to fix
    }
    let mut out = s.to_string();
    if in_string {
        out.push('"');
    }
    // A dangling trailing comma or colon (cut off between a completed element/key and whatever would
    // have followed it) never parses even once containers below it close — trimming it back to the
    // last complete element is cheap and safe: if it doesn't help, the result is no more or less
    // parseable than leaving it in.
    let trimmed_len = out.trim_end().len();
    if let Some(last) = out[..trimmed_len].chars().next_back() {
        if last == ',' || last == ':' {
            out.truncate(trimmed_len - last.len_utf8());
        }
    }
    while let Some(open) = stack.pop() {
        out.push(if open == '{' { '}' } else { ']' });
    }
    Some(out)
}

/// The assembled result of one model turn.
struct Turn {
    blocks: Vec<ContentBlock>,
    stop_reason: StopReason,
    usage: TokenUsage,
    /// Tool calls whose streamed arguments never parsed as JSON, as `(tool_use_id, raw buffer)`. The
    /// loop turns each into an error `tool_result` the model can correct, rather than aborting the run.
    malformed: Vec<(String, String)>,
}

/// One still-open content block's accumulated state, keyed by its `index` in [`Accumulator::open`].
enum OpenBlock {
    /// (text, id, phase) of an open text block — `id`/`phase` are OpenAI Responses' replay
    /// metadata, populated only by that dialect's `TextFinal` (see `ContentBlock::Text`).
    Text(String, Option<String>, Option<String>),
    /// (text, signature) of an open thinking block.
    Thinking(String, String),
    /// (id, name, json-arg buffer, reasoning-continuity data) of an open tool call. The last field is
    /// OpenAI Chat Completions' `reasoning_details` replay data (see `ContentBlock::ToolUse`'s doc
    /// comment) — populated by the same `SignatureDelta` event `Thinking` uses, since both are just an
    /// opaque string attached to whatever block is open at that index; `None` for every dialect that
    /// never emits it for a tool-call index.
    Tool(String, String, String, Option<String>),
}

/// Folds a `StreamEvent` sequence into content blocks. Every block-scoped event carries an `index`,
/// and more than one can be open at once — a dialect whose wire genuinely interleaves multiple blocks
/// (currently only OpenAI Responses, when the model streams two tool calls' arguments concurrently)
/// reports them as such, each accruing its own text/thinking/tool-argument state independently rather
/// than one being buffered and replayed as a single burst once the other closes. `ContentBlockStop`
/// finalizes one index; a finalized index is held in `done` until every *earlier-declared* index has
/// also finalized, then the run of consecutively-ready entries flushes into `blocks` in declaration
/// order (`try_flush`) — so the assembled message never reorders blocks relative to when the model
/// announced them, even though a later-declared block may finish streaming its content first. A
/// dialect whose wire never interleaves (Anthropic in practice, OpenAI Chat Completions' text) just
/// always uses index 0, so exactly one index is ever open and this degenerates to the old
/// single-current-block behavior.
#[derive(Default)]
struct Accumulator {
    /// Declaration order of every index seen so far, oldest first.
    order: Vec<usize>,
    /// Still-open blocks, by index.
    open: HashMap<usize, OpenBlock>,
    /// Finalized blocks awaiting their turn to flush (an index whose `ContentBlockStop` arrived before
    /// an earlier-declared index's own), by index. `None` means the index resolved to no content at
    /// all (e.g. a text block that never accrued anything) — still needs a slot so `try_flush` knows
    /// it's resolved and doesn't stall waiting on it forever, it just contributes nothing to `blocks`.
    done: HashMap<usize, Option<ContentBlock>>,
    blocks: Vec<ContentBlock>,
    stop_reason: StopReason,
    usage: TokenUsage,
    /// Tool calls whose streamed JSON arguments failed to parse, as `(id, raw buffer)`. Surfaced on
    /// the `Turn` so the loop can feed each back as a recoverable error `tool_result`.
    malformed: Vec<(String, String)>,
}

impl Accumulator {
    /// Record `index` in declaration order, the first time anything mentions it. O(1) — checks the
    /// maps, not a linear scan of `order`, since this runs on every single delta (hundreds to thousands
    /// per turn).
    fn declare(&mut self, index: usize) {
        if !self.open.contains_key(&index) && !self.done.contains_key(&index) {
            self.order.push(index);
        }
    }

    // Borrows rather than takes `StreamEvent` by value: a streamed turn folds hundreds to thousands
    // of deltas through here, and the caller (`run_turn_once`) also needs the event afterward (to
    // `emit` it) — taking it by value would force a clone on every single delta just so both sides
    // get their own copy. Only the block-boundary variants need an owned copy of their payload for
    // `self.open`/`self.blocks`; the high-frequency delta variants just borrow to `push_str`, and
    // `Usage`/`MessageStop` are `Copy`.
    fn apply(&mut self, ev: &StreamEvent) {
        match ev {
            StreamEvent::MessageStart => {}
            StreamEvent::TextDelta { index, text } => {
                self.declare(*index);
                if let OpenBlock::Text(s, ..) = self
                    .open
                    .entry(*index)
                    .or_insert_with(|| OpenBlock::Text(String::new(), None, None))
                {
                    s.push_str(text);
                }
            }
            StreamEvent::ThinkingDelta { index, text } => {
                self.declare(*index);
                if let OpenBlock::Thinking(t, _) = self
                    .open
                    .entry(*index)
                    .or_insert_with(|| OpenBlock::Thinking(String::new(), String::new()))
                {
                    t.push_str(text);
                }
            }
            StreamEvent::SignatureDelta { index, signature } => {
                self.declare(*index);
                // Shared by two unrelated meanings, disambiguated by whichever block is already open
                // at this index: Anthropic-style thinking signatures (the common case — appended, since
                // a real cryptographic signature can arrive fragmented) and OpenAI Chat Completions'
                // `reasoning_details` tool-call replay data (always sent as one complete chunk, but
                // `push_str` is still correct — and harmless if the provider ever did fragment it).
                // Defaults to opening a new `Thinking` block, matching the pre-existing behavior, when
                // nothing is open yet at this index — a dialect only ever emits this for a tool call
                // index *after* that call's `ToolUseStart`, never before.
                match self
                    .open
                    .entry(*index)
                    .or_insert_with(|| OpenBlock::Thinking(String::new(), String::new()))
                {
                    OpenBlock::Thinking(_, sig) => sig.push_str(signature),
                    OpenBlock::Tool(_, _, _, thought_signature) => thought_signature
                        .get_or_insert_with(String::new)
                        .push_str(signature),
                    OpenBlock::Text(..) => {}
                }
            }
            StreamEvent::RedactedThinking { index, data } => {
                // Self-contained — no delta phase, so there's no `open` entry, just an immediate
                // resolution in its declared position.
                self.declare(*index);
                self.done.insert(
                    *index,
                    Some(ContentBlock::RedactedThinking { data: data.clone() }),
                );
                self.try_flush();
            }
            StreamEvent::ToolUseStart { index, id, name } => {
                self.declare(*index);
                self.open.insert(
                    *index,
                    OpenBlock::Tool(id.clone(), name.clone(), String::new(), None),
                );
            }
            StreamEvent::InputJsonDelta {
                index,
                partial_json,
            } => {
                if let Some(OpenBlock::Tool(_, _, buf, _)) = self.open.get_mut(index) {
                    buf.push_str(partial_json);
                }
            }
            StreamEvent::InputJsonFinal { index, full_json } => {
                if let Some(OpenBlock::Tool(_, _, buf, _)) = self.open.get_mut(index) {
                    buf.clone_from(full_json);
                }
            }
            StreamEvent::TextFinal {
                index,
                text,
                id,
                phase,
            } => {
                if let Some(OpenBlock::Text(s, block_id, block_phase)) = self.open.get_mut(index) {
                    s.clone_from(text);
                    block_id.clone_from(id);
                    block_phase.clone_from(phase);
                }
            }
            StreamEvent::ThinkingFinal { index, text } => {
                if let Some(OpenBlock::Thinking(t, _)) = self.open.get_mut(index) {
                    t.clone_from(text);
                }
            }
            StreamEvent::ContentBlockStop { index } => self.flush_block(*index),
            StreamEvent::Usage(usage) => self.usage = *usage,
            StreamEvent::MessageStop { stop_reason } => self.stop_reason = *stop_reason,
        }
    }

    /// Finalize `index`: convert its accumulated state into a `ContentBlock` (or nothing, for an empty
    /// text run) and record it in `done`, then flush whatever consecutive run of declared indices, from
    /// the front, is now fully resolved.
    fn flush_block(&mut self, index: usize) {
        let Some(open_block) = self.open.remove(&index) else {
            // A `ContentBlockStop` with nothing open at this index (never declared, or already
            // flushed) — harmless no-op, same as the old code's behavior for a spurious extra stop.
            return;
        };
        let block = match open_block {
            OpenBlock::Tool(id, name, args, thought_signature) => {
                let input = if args.trim().is_empty() {
                    json!({})
                } else {
                    let parsed = serde_json::from_str(&args)
                        .or_else(|_| serde_json::from_str(&repair_json(&args)))
                        .or_else(|e| match close_incomplete_json(&args) {
                            Some(closed) => serde_json::from_str(&closed),
                            None => Err(e),
                        });
                    match parsed {
                        Ok(v) => v,
                        // Still doesn't parse even after both repair passes — a genuine protocol
                        // glitch, not a tool failure. Keep the tool_use block (with an empty,
                        // wire-valid input object so the next request doesn't 400) and record the
                        // call as malformed; the loop feeds back an error result the model can
                        // correct, instead of aborting the run.
                        Err(_) => {
                            self.malformed.push((id.clone(), args));
                            json!({})
                        }
                    }
                };
                Some(ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    thought_signature,
                })
            }
            OpenBlock::Thinking(text, signature) => {
                Some(ContentBlock::Thinking { text, signature })
            }
            OpenBlock::Text(text, id, phase) => {
                (!text.is_empty()).then_some(ContentBlock::Text { text, id, phase })
            }
        };
        self.done.insert(index, block);
        self.try_flush();
    }

    /// Flush every consecutive index, from the front of `order`, that has finalized — stopping at the
    /// first index that hasn't (preserving declaration order: a later-declared index that finished
    /// streaming first waits here until its predecessors catch up).
    fn try_flush(&mut self) {
        while let Some(&front) = self.order.first() {
            match self.done.remove(&front) {
                Some(block) => {
                    self.order.remove(0);
                    if let Some(block) = block {
                        self.blocks.push(block);
                    }
                }
                None => break,
            }
        }
    }

    fn finish(mut self) -> Turn {
        // A stream that ended without a trailing `ContentBlockStop` for every open index still
        // contributes each one's accumulated content. Any call order works here — `try_flush` (run
        // from inside each `flush_block`) resolves final ordering regardless of which index this loop
        // happens to finalize first.
        for index in self.open.keys().copied().collect::<Vec<_>>() {
            self.flush_block(index);
        }
        Turn {
            blocks: self.blocks,
            stop_reason: self.stop_reason,
            usage: self.usage,
            malformed: self.malformed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;
    use crate::mock::{MockTransport, turn};
    use crate::steering::SteeringMessage;
    use crate::tool::Tool;
    use async_trait::async_trait;
    use serde_json::json;

    struct EchoTool;
    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo the text arg"
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] })
        }
        async fn run(
            &self,
            input: Value,
        ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
            input
                .get("text")
                .and_then(Value::as_str)
                .map(|s| s.into())
                .ok_or_else(|| crate::error::ToolError::InvalidInput("missing text".into()))
        }
    }

    fn agent_with(
        turns: Vec<Vec<StreamEvent>>,
        tools: ToolRegistry,
    ) -> (Agent, Arc<MockTransport>) {
        let mock = Arc::new(MockTransport::new(turns));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        (agent, mock)
    }

    #[test]
    fn accumulator_attaches_a_signature_delta_to_an_open_tool_call_not_only_to_thinking() {
        // [A-M5] `SignatureDelta` is shared by two unrelated meanings, disambiguated by whichever
        // block is already open at its index (see `apply`'s doc comment): the common case, an
        // Anthropic-style thinking-block signature, and OpenAI Chat Completions' Gemini/OpenRouter
        // `reasoning_details` tool-call replay data (`dialect::openai::Decoder`). Before this fix
        // `OpenBlock::Tool` had no field to hold it at all, and the handler unconditionally matched
        // `OpenBlock::Thinking`, so a `SignatureDelta` arriving for an open *tool-call* index was
        // silently dropped (and would even have clobbered a `Thinking` block accidentally
        // re-materialized at that index). This pins the correct behavior end to end through the
        // `Accumulator`, independent of any one dialect's own wire parsing.
        let mut acc = Accumulator::default();
        acc.apply(&StreamEvent::ToolUseStart {
            index: 0,
            id: "call_1".into(),
            name: "read".into(),
        });
        acc.apply(&StreamEvent::SignatureDelta {
            index: 0,
            signature: r#"{"type":"reasoning.encrypted","id":"call_1","data":"enc"}"#.into(),
        });
        acc.apply(&StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: r#"{"path":"README.md"}"#.into(),
        });
        acc.apply(&StreamEvent::ContentBlockStop { index: 0 });
        acc.apply(&StreamEvent::MessageStop {
            stop_reason: StopReason::ToolUse,
        });
        let turn = acc.finish();
        assert_eq!(
            turn.blocks,
            vec![ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "read".into(),
                input: json!({ "path": "README.md" }),
                thought_signature: Some(
                    r#"{"type":"reasoning.encrypted","id":"call_1","data":"enc"}"#.into()
                ),
            }]
        );
    }

    #[tokio::test]
    async fn single_text_turn_completes() {
        let (agent, mock) = agent_with(vec![turn::text("hello world")], ToolRegistry::new());
        let mut session = Session::new();
        session.user("hi");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(mock.calls(), 1);
        assert_eq!(session.steps, 1);
        // user + assistant
        assert_eq!(session.messages.len(), 2);
        assert_eq!(
            session.messages[1].content,
            vec![ContentBlock::text("hello world")]
        );
    }

    #[tokio::test]
    async fn usage_snapshot_is_taken_before_the_turns_own_message_is_appended() {
        // pi-parity fix: `record_usage` used to run *after* `session.push`, so
        // `last_usage_message_count` (the boundary `compaction::trailing_tokens` estimates forward
        // from) included the very assistant message that usage snapshot came from — that turn's own
        // output was invisible to the compaction trigger until the *next* real usage snapshot caught
        // up, undercounting the live context by roughly one turn's worth every single loop iteration.
        let (agent, _mock) = agent_with(
            vec![turn::text(
                "a reasonably long response so its estimated token count is unambiguously non-zero",
            )],
            ToolRegistry::new(),
        );
        let mut session = Session::new();
        session.user("hi");
        agent.run(&mut session, |_| {}).await.unwrap();

        // The turn just completed; its own assistant message must already be visible to
        // `trailing_tokens`, not swallowed into the usage-snapshot boundary that was just recorded.
        assert!(
            compaction::trailing_tokens(&session) > 0,
            "expected the just-appended assistant message to count as trailing context, got 0 \
             (last_usage_message_count={}, messages.len()={})",
            session.last_usage_message_count,
            session.messages.len()
        );
    }

    #[tokio::test]
    async fn the_turns_usage_is_stamped_on_its_own_assistant_message() {
        // Task #6 (pi-parity), agent-core's portion: `Message::usage` (pi's own
        // `AssistantMessage.usage`, a required field there) must be populated on the in-memory
        // `Session` the instant a turn's assistant message is appended — the data a later session-
        // persistence fix (out of this crate's scope) needs to actually exist to be saved.
        let (agent, _mock) = agent_with(
            vec![vec![
                StreamEvent::MessageStart,
                StreamEvent::TextDelta {
                    index: 0,
                    text: "hello".into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 42,
                    output_tokens: 7,
                    cache_read_tokens: 3,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::EndTurn,
                },
            ]],
            ToolRegistry::new(),
        );
        let mut session = Session::new();
        session.user("hi");
        agent.run(&mut session, |_| {}).await.unwrap();

        let usage = session.messages.last().unwrap().usage;
        assert_eq!(
            usage,
            Some(TokenUsage {
                input_tokens: 42,
                output_tokens: 7,
                cache_read_tokens: 3,
                ..Default::default()
            }),
            "the assistant message must carry the exact per-turn usage the provider reported"
        );
    }

    #[tokio::test]
    async fn a_user_turn_never_carries_a_usage_value() {
        let (agent, _mock) = agent_with(vec![turn::text("hi")], ToolRegistry::new());
        let mut session = Session::new();
        session.user("hello");
        agent.run(&mut session, |_| {}).await.unwrap();
        assert_eq!(session.messages[0].usage, None);
    }

    #[tokio::test]
    async fn set_system_replaces_the_prompt_used_by_the_next_turn() {
        let (mut agent, mock) = agent_with(
            vec![turn::text("first"), turn::text("second")],
            ToolRegistry::new(),
        );
        agent = agent.with_system("system A");
        let mut session = Session::new();
        session.user("hi");
        agent.run(&mut session, |_| {}).await.unwrap();
        agent.set_system("system B");
        session.user("again");
        agent.run(&mut session, |_| {}).await.unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].system.as_deref(), Some("system A"));
        assert_eq!(requests[1].system.as_deref(), Some("system B"));
    }

    #[tokio::test]
    async fn with_system_fn_is_reevaluated_every_turn() {
        // Task #15 (pi-parity): `set_system` needs `&mut self`, unavailable while
        // `run_events_steered` holds `&self` for a whole (possibly long-running) call — so a run in
        // flight could never refresh a time-varying prompt (a date stamp) turn-to-turn. A per-turn
        // callback installed via `with_system_fn` is instead consulted fresh every turn, through `&self`
        // alone, working from *inside* a single `run` call across multiple turns.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
                turn::text("done"),
            ],
            tools,
        );
        let calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let calls_clone = calls.clone();
        let agent = agent.with_system_fn(move || {
            let n = calls_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            format!("turn number {n}")
        });
        let mut session = Session::new();
        session.user("say pong");
        agent.run(&mut session, |_| {}).await.unwrap();

        let requests = mock.requests();
        assert_eq!(
            requests.len(),
            2,
            "one model call per turn, tool call included"
        );
        assert_eq!(
            requests[0].system.as_deref(),
            Some("turn number 0"),
            "the callback must be consulted for the first turn"
        );
        assert_eq!(
            requests[1].system.as_deref(),
            Some("turn number 1"),
            "the callback must be re-evaluated fresh for the second turn of the same run"
        );
    }

    #[tokio::test]
    async fn with_system_fn_takes_priority_over_the_static_system_string() {
        // Mirrors pi's single function-or-string `systemPrompt` field: a caller setting both gets the
        // callback's value, not a merge of the two.
        let (agent, mock) = agent_with(vec![turn::text("hi")], ToolRegistry::new());
        let agent = agent
            .with_system("static prompt")
            .with_system_fn(|| "dynamic prompt".to_string());
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        let requests = mock.requests();
        assert_eq!(requests[0].system.as_deref(), Some("dynamic prompt"));
    }

    #[tokio::test]
    async fn tool_call_round_trips_and_continues() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
                turn::text("done"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("say pong");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(mock.calls(), 2);
        assert_eq!(session.steps, 2);
        // user, assistant(tool_use), user(tool_result), assistant(text)
        assert_eq!(session.messages.len(), 4);
        assert_eq!(
            session.messages[2].content,
            vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "pong".into(),
                is_error: false,
                images: Vec::new(),
            }]
        );
        // The second request the loop sent must include the tool result.
        let second = &mock.requests()[1];
        assert!(
            second
                .messages
                .iter()
                .any(|m| matches!(m.content.first(), Some(ContentBlock::ToolResult { .. })))
        );
    }

    #[tokio::test]
    async fn mid_stream_failure_retries_with_a_clean_turn_not_a_resumed_one() {
        // Attempt 1 dies mid-tool-call: a `ToolUseStart` and a *partial* argument fragment stream in,
        // then the connection dies before `ContentBlockStop`/`MessageStop` — never reaching
        // `Accumulator::finish`, so that attempt's half-formed tool call is dropped, not returned.
        // Attempt 2 (the retry) streams the *same* call with its *complete*, different argument bytes.
        // If the retry ever resumed attempt 1's accumulator instead of starting fresh, the dispatched
        // tool call would see attempt 1's leftover partial JSON (`{"tex`, invalid) rather than attempt
        // 2's full one (`{"text":"pong"}`, valid) — this asserts the tool actually ran with the latter.
        let mock = Arc::new(MockTransport::scripted(vec![
            vec![
                Ok(StreamEvent::MessageStart),
                Ok(StreamEvent::ToolUseStart {
                    index: 0,
                    id: "tu_1".into(),
                    name: "echo".into(),
                }),
                Ok(StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "{\"tex".into(),
                }),
                Err(Error::Transport(
                    "Anthropic stream ended before message_stop".into(),
                )),
            ],
            vec![
                Ok(StreamEvent::MessageStart),
                Ok(StreamEvent::ToolUseStart {
                    index: 0,
                    id: "tu_1".into(),
                    name: "echo".into(),
                }),
                Ok(StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: "{\"text\":\"pong\"}".into(),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                }),
            ],
            turn::text("done").into_iter().map(Ok).collect(),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("say pong");
        agent.run(&mut session, |_| {}).await.unwrap();

        // 3 transport calls: the failed attempt, its successful retry, and the follow-up text turn —
        // the retry is invisible to `session.steps` (2: the tool turn + the text turn), matching a
        // normal two-step run with no trace of the failed attempt in loop-visible state.
        assert_eq!(mock.calls(), 3);
        assert_eq!(session.steps, 2);
        assert_eq!(session.messages.len(), 4); // user, assistant(tool_use), user(tool_result), assistant(text)
        assert_eq!(
            session.messages[1].content,
            vec![ContentBlock::tool_use(
                "tu_1",
                "echo",
                json!({ "text": "pong" }),
            )]
        );
        // The tool actually ran against the retry's complete, valid arguments — not an error result
        // from attempt 1's truncated `{"tex` fragment.
        assert_eq!(
            session.messages[2].content,
            vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "pong".into(),
                is_error: false,
                images: Vec::new(),
            }]
        );
    }

    #[tokio::test]
    async fn tool_call_dispatches_even_when_stop_reason_is_max_tokens() {
        // pi-parity fix: dispatch must key off the presence of a complete `tool_use` block alone, never
        // `stop_reason` — a turn can legitimately emit one or more complete tool calls and *then* get cut
        // off by `max_tokens` while the model was still writing trailing commentary. Gating dispatch on
        // `stop_reason == ToolUse` would silently drop the call the model actually made.
        let mock = Arc::new(MockTransport::scripted(vec![
            vec![
                Ok(StreamEvent::MessageStart),
                Ok(StreamEvent::ToolUseStart {
                    index: 0,
                    id: "tu_1".into(),
                    name: "echo".into(),
                }),
                Ok(StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: r#"{"text":"pong"}"#.into(),
                }),
                Ok(StreamEvent::ContentBlockStop { index: 0 }),
                Ok(StreamEvent::MessageStop {
                    stop_reason: StopReason::MaxTokens,
                }),
            ],
            turn::text("done").into_iter().map(Ok).collect(),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("say pong");
        agent.run(&mut session, |_| {}).await.unwrap();

        // Both turns ran (the tool round-trip, then the follow-up text) — dispatch happened despite
        // `MaxTokens`, so this isn't the silent-truncation/no-dispatch path collapsing it to one call.
        assert_eq!(mock.calls(), 2);
        // user, assistant(tool_use), user(tool_result), assistant(text)
        assert_eq!(session.messages.len(), 4);
        assert_eq!(
            session.messages[1].content,
            vec![ContentBlock::tool_use(
                "tu_1",
                "echo",
                json!({ "text": "pong" }),
            )]
        );
        assert_eq!(
            session.messages[2].content,
            vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "pong".into(),
                is_error: false,
                images: Vec::new(),
            }]
        );
    }

    #[tokio::test]
    async fn cancellation_interrupts_mid_stream_retry_backoff() {
        use std::time::Duration;

        // Attempt 1 dies mid-stream with a retryable error, which schedules a ~250ms backoff sleep
        // before the retry. A cancel tripped well inside that window must interrupt the sleep itself —
        // left unraced, the run would sit idle for the full backoff before the next attempt even got a
        // chance to observe cancellation.
        let mock = Arc::new(MockTransport::scripted(vec![vec![
            Ok(StreamEvent::MessageStart),
            Err(Error::Transport(
                "Anthropic stream ended before message_stop".into(),
            )),
        ]]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_max_steps(8);
        let mut session = Session::new();
        session.user("hi");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            trigger.cancel();
        });

        let result = tokio::time::timeout(
            Duration::from_millis(150), // comfortably under the ~250ms base backoff
            agent.run_events_cancellable(&mut session, |_| {}, cancel),
        )
        .await
        .expect("cancellation must interrupt the backoff sleep, not wait it out");
        assert!(matches!(result, Err(Error::Cancelled)));
    }

    #[tokio::test]
    async fn non_retryable_mid_stream_error_fails_the_run_immediately() {
        // A generic transport error (not the "stream ended before…"/"overloaded" shapes) must not
        // retry — only one scripted turn exists, so a retry attempt would exhaust the mock and this
        // test would fail with a *different* error ("no more scripted turns") if retry logic were
        // over-broad.
        let mock = Arc::new(MockTransport::scripted(vec![vec![
            Ok(StreamEvent::MessageStart),
            Err(Error::Transport(
                "gateway returned 400: invalid request".into(),
            )),
        ]]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_max_steps(8);
        let mut session = Session::new();
        session.user("hi");
        let err = agent.run(&mut session, |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::Transport(msg) if msg.contains("400")));
        assert_eq!(mock.calls(), 1, "a non-retryable error must not retry");
    }

    #[tokio::test]
    async fn a_run_ending_in_error_persists_a_closing_assistant_record() {
        // pi-parity fix (`packages/agent/test/agent.test.ts:126-155`, "emits full lifecycle events for
        // thrown run failures"): pi's `handleRunFailure` always appends a synthetic assistant message
        // (`stopReason:"error"`, `errorMessage:<text>`) so the session's last message is never the
        // user's own un-answered prompt. Previously this loop's `Err(e) => { sink(...); return Err(e) }`
        // returned without ever touching `session.messages`, leaving [user] as the entire transcript.
        let mock = Arc::new(MockTransport::scripted(vec![vec![
            Ok(StreamEvent::MessageStart),
            Err(Error::Transport(
                "gateway returned 400: invalid request".into(),
            )),
        ]]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_max_steps(8);
        let mut session = Session::new();
        session.user("hi");
        let err = agent.run(&mut session, |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::Transport(_)));

        assert_eq!(
            session.messages.len(),
            2,
            "expected [user, assistant(error)]"
        );
        let closing = &session.messages[1];
        assert_eq!(closing.role, Role::Assistant);
        assert_eq!(
            closing.error_message.as_deref(),
            Some("transport error: gateway returned 400: invalid request")
        );
        assert_eq!(closing.content, vec![ContentBlock::text("")]);
    }

    #[tokio::test]
    async fn a_prompt_after_a_failed_run_does_not_double_push_a_user_turn() {
        // The other half of the fix above: once the failure closes the turn with a real assistant
        // record, a client retrying with a fresh `session.user(...)` must restore valid role
        // alternation — not append a second consecutive `user` message, a shape no wire dialect
        // accepts. The shared premise — a failed run must close with a real, role:"assistant" record —
        // is proven in pi's own real, shipped test suite too: `packages/agent/test/agent.test.ts`'s
        // "emits full lifecycle events for thrown run failures" asserts the same `lastMessage.role ===
        // "assistant"` shape (not the dead `packages/agent/test/harness/agent-harness.test.ts`, which
        // tests the unused harness — zero references from `packages/coding-agent`). The second half
        // proven here — that a subsequent prompt doesn't then double-push a consecutive user turn — is
        // this crate's own additional invariant; a search of both `packages/agent/test/agent.test.ts`
        // and `packages/coding-agent/test/agent-session-retry.test.ts` didn't turn up a directly
        // equivalent assertion in pi's real test suite.
        let mock = Arc::new(MockTransport::scripted(vec![
            vec![
                Ok(StreamEvent::MessageStart),
                Err(Error::Transport(
                    "gateway returned 400: invalid request".into(),
                )),
            ],
            turn::text("recovered").into_iter().map(Ok).collect(),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_max_steps(8);
        let mut session = Session::new();
        session.user("hi");
        agent.run(&mut session, |_| {}).await.unwrap_err();

        // Role alternation holds: [user, assistant(error), user] — this new prompt lands right after
        // the closing error record, not stacked on a dangling unanswered `user` turn.
        assert_eq!(session.messages[1].role, Role::Assistant);
        session.user("after failure");
        assert_eq!(session.messages[2].role, Role::User);
        agent.run(&mut session, |_| {}).await.unwrap();
        let last = session.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert!(!last.content.is_empty());
    }

    #[tokio::test]
    async fn with_auto_retry_false_fails_immediately_on_an_otherwise_retryable_error() {
        // The exact same retryable error shape `cancellation_interrupts_mid_stream_retry_backoff`
        // above proves gets retried by default — with `auto_retry` off, it must fail on the very first
        // attempt instead, with no backoff wait and no second transport call.
        let mock = Arc::new(MockTransport::scripted(vec![vec![
            Ok(StreamEvent::MessageStart),
            Err(Error::Transport(
                "Anthropic stream ended before message_stop".into(),
            )),
        ]]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_max_steps(8)
            .with_auto_retry(false);
        let mut session = Session::new();
        session.user("hi");
        let err = agent.run(&mut session, |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::Transport(msg) if msg.contains("stream ended")));
        assert_eq!(
            mock.calls(),
            1,
            "auto_retry(false) must not retry an otherwise-retryable error"
        );
    }

    #[test]
    fn is_context_overflow_detects_google() {
        assert!(is_context_overflow(&Error::Transport(
            "400 Bad Request: exceeds the maximum number of tokens allowed: 32768".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_xai() {
        assert!(is_context_overflow(&Error::Transport(
            "maximum prompt length is 131072 tokens".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_groq() {
        assert!(is_context_overflow(&Error::Transport(
            "Please reduce the length of the messages.".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_openrouter() {
        assert!(is_context_overflow(&Error::Transport(
            "maximum context length is 8192 tokens".into()
        )));
        assert!(is_context_overflow(&Error::Transport(
            "This model's exceeds the maximum allowed input length of 16000".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_together() {
        assert!(is_context_overflow(&Error::Transport(
            "Input validation error: `inputs` tokens + `max_new_tokens` must be <= 4096. Given: \
             is longer than the model's context length"
                .into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_mistral() {
        assert!(is_context_overflow(&Error::Transport(
            "too large for model with 32768 maximum context length".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_cerebras() {
        assert!(is_context_overflow(&Error::Transport(
            "400 (no body)".into()
        )));
        assert!(is_context_overflow(&Error::Transport(
            "413 (no body)".into()
        )));
        // The "status code" variant pi's single regex also matches — a plain "400 (no body)"
        // substring check misses this because of the extra words in between.
        assert!(is_context_overflow(&Error::Transport(
            "400 status code (no body)".into()
        )));
        assert!(is_context_overflow(&Error::Transport(
            "413 status code (no body)".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_ds4() {
        // pi-parity gap (fixed): DS4's phrasing wasn't in the table at all, and its numbers-in-the-
        // middle shape doesn't match the generic `context`+`{long,exceed,maximum,window}` catch-all
        // either — ported straight from pi's own test fixture (`overflow.test.ts`, commit `21cb380`),
        // comma-formatted variant included.
        assert!(is_context_overflow(&Error::Transport(
            "400 Prompt has 256468 tokens, but the configured context size is 256000 tokens".into()
        )));
        assert!(is_context_overflow(&Error::Transport(
            "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_bedrock() {
        assert!(is_context_overflow(&Error::Transport(
            "ValidationException: Input is too long for requested model.".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_github_copilot() {
        assert!(is_context_overflow(&Error::Transport(
            "prompt token count of 131072 exceeds the limit of 128000".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_llama_cpp() {
        assert!(is_context_overflow(&Error::Transport(
            "the request exceeds the available context size, try increasing it".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_lm_studio() {
        assert!(is_context_overflow(&Error::Transport(
            "tokens to keep from the initial prompt is greater than the context length".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_minimax() {
        assert!(is_context_overflow(&Error::Transport(
            "invalid params, context window exceeds limit".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_kimi() {
        assert!(is_context_overflow(&Error::Transport(
            "Your request exceeded model token limit: 131072 (requested: 200000)".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_zai() {
        assert!(is_context_overflow(&Error::Transport(
            "finish_reason: model_context_window_exceeded".into()
        )));
    }

    #[test]
    fn is_context_overflow_detects_ollama() {
        assert!(is_context_overflow(&Error::Transport(
            "prompt too long; exceeded max context length by 4096 tokens".into()
        )));
    }

    #[test]
    fn is_context_overflow_excludes_bedrock_throttling_despite_too_many_tokens_substring() {
        // Bedrock's throttling message contains "too many tokens" — the same substring the generic
        // overflow pattern matches — but means "you're sending requests too fast," not "shrink the
        // prompt." Compacting in response would be pure churn against a request that would have
        // succeeded unmodified on retry.
        assert!(!is_context_overflow(&Error::Transport(
            "ThrottlingException: Too many tokens, please wait before trying again.".into()
        )));
        assert!(!is_context_overflow(&Error::Transport(
            "429: rate limit exceeded, too many requests".into()
        )));
    }

    #[test]
    fn is_context_overflow_negative_cases_ported_from_pi() {
        // A-L7 pi-parity test gap (fixed): pi's `overflow.test.ts` pins 5 distinct "must NOT be
        // classified as overflow" scenarios; the prior test above only exercised 2 combined-pattern
        // variants of its own. Ported verbatim, one assertion per pi case.
        for (msg, why) in [
            (
                "500 `model runner crashed unexpectedly`",
                "a generic Ollama crash, not a context-window rejection",
            ),
            (
                "Throttling error: Too many tokens, please wait before trying again.",
                "Bedrock throttling (HTTP 429), not overflow, despite the 'too many tokens' wording",
            ),
            (
                "Service unavailable: The service is temporarily unavailable.",
                "Bedrock service unavailable, a transient outage not a context-size rejection",
            ),
            (
                "Rate limit exceeded, please retry after 30 seconds.",
                "a generic rate limit, not overflow",
            ),
            (
                "Too many requests. Please slow down.",
                "a generic HTTP 429 style error, not overflow",
            ),
        ] {
            assert!(
                !is_context_overflow(&Error::Transport(msg.into())),
                "expected NOT overflow ({why}): {msg}"
            );
        }
    }

    #[tokio::test]
    async fn overflow_recovery_compaction_failure_surfaces_a_curated_message() {
        // B-M12 pi-parity gap (fixed): when the overflow-recovery compact-and-retry's own
        // summarization call fails (a distinct failure from the overflow it was trying to recover
        // from), the raw underlying transport error used to surface verbatim, both on the emitted
        // `AgentEvent::Error` and the session's closing error record. Matches pi's own
        // `_runAutoCompaction` catch block ("Context overflow recovery failed: {error}") — untested
        // before this fix.
        let session_messages = vec![
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
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::scripted(vec![
            // The live turn: the provider rejects it for exceeding the context window.
            vec![Err(Error::Transport(
                "prompt is too long: 300000 tokens > 200000 maximum".into(),
            ))],
            // The recovery compaction's own (single, clean-boundary) summarization call fails
            // outright — a distinct, unrelated failure from the overflow above.
            vec![Err(Error::Transport("mock summarizer down".into()))],
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });

        let mut error_messages = Vec::new();
        let result = agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::Error { message } = ev {
                    error_messages.push(message);
                }
            })
            .await;

        assert!(
            result.is_err(),
            "the turn genuinely cannot proceed; the run must still fail"
        );
        assert_eq!(
            mock.calls(),
            2,
            "the overflowing turn plus the one failed recovery summarization call"
        );
        assert_eq!(error_messages.len(), 1);
        assert!(
            error_messages[0].contains("Context overflow recovery failed:")
                && error_messages[0].contains("mock summarizer down"),
            "expected a curated message wrapping the underlying failure, got: {:?}",
            error_messages[0]
        );
        assert_eq!(
            session
                .messages
                .last()
                .and_then(|m| m.error_message.as_deref()),
            Some(error_messages[0].as_str()),
            "the curated message must also be the session's closing error record, not the raw one"
        );
    }

    #[tokio::test]
    async fn second_overflow_after_recovery_already_attempted_surfaces_a_curated_message() {
        // B-M12 pi-parity gap (fixed): once a compaction has already run once to recover from an
        // overflow this call (`overflow_recovered`), a *second* overflow on the retried turn used to
        // fall through to the generic error path and surface the raw provider "prompt too long"
        // message. Matches pi's own guard against retrying overflow recovery more than once per call
        // (`_checkCompaction`'s `_overflowRecoveryAttempted`), including its exact curated string.
        let session_messages = vec![
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
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::scripted(vec![
            // The live turn overflows.
            vec![Err(Error::Transport(
                "prompt is too long: 300000 tokens > 200000 maximum".into(),
            ))],
            // The recovery compaction itself succeeds this time...
            turn::text("## Goal\nsummary of earlier work")
                .into_iter()
                .map(Ok)
                .collect(),
            // ...but the retried turn overflows again anyway (still too large even compacted).
            vec![Err(Error::Transport(
                "prompt is too long: 300000 tokens > 200000 maximum".into(),
            ))],
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });

        let mut error_messages = Vec::new();
        let mut compacted = false;
        let result = agent
            .run_events(&mut session, |ev| match ev {
                AgentEvent::Error { message } => error_messages.push(message),
                AgentEvent::Compacted { .. } => compacted = true,
                _ => {}
            })
            .await;

        assert!(result.is_err(), "a second overflow must still fail the run");
        assert!(
            compacted,
            "the first (successful) recovery must still have run"
        );
        assert_eq!(mock.calls(), 3);
        assert_eq!(error_messages.len(), 1);
        assert_eq!(
            error_messages[0],
            "Context overflow recovery failed after one compact-and-retry attempt. Try reducing \
             context or switching to a larger-context model.",
            "must match pi's own curated string exactly"
        );
        assert_eq!(
            session
                .messages
                .last()
                .and_then(|m| m.error_message.as_deref()),
            Some(error_messages[0].as_str())
        );
    }

    /// Regression: a cancellation racing the *recovery* `compact()` call inside the
    /// MaxTokens-silent-truncation path (`turn.stop_reason == StopReason::MaxTokens`, no tool calls)
    /// used to fall through and treat `Error::Cancelled` exactly like an ordinary failed recompaction
    /// attempt — reporting a non-terminal `CompactionFailed` event and continuing on to checkpoint and
    /// `drain_at_stop()`, which (with nothing queued there) returned `Ok(())` for what should have been
    /// a cancelled run, breaking the documented "returns `Error::Cancelled` once cancelled" contract
    /// every other caller relies on for whole-run retry exclusion.
    #[tokio::test]
    async fn cancellation_during_max_tokens_recovery_compaction_surfaces_as_cancelled_not_ok() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct MaxTokensThenStalledCompaction {
            calls: AtomicUsize,
        }
        #[async_trait]
        impl ModelTransport for MaxTokensThenStalledCompaction {
            async fn stream(&self, _req: ModelRequest) -> Result<crate::transport::EventStream> {
                let call = self.calls.fetch_add(1, Ordering::SeqCst);
                if call == 0 {
                    // The live turn: cut off by max_tokens, no tool calls — the silent-truncation
                    // recovery path this test targets.
                    let s = futures::stream::iter(vec![
                        Ok(StreamEvent::MessageStart),
                        Ok(StreamEvent::TextDelta {
                            index: 0,
                            text: "cut off".into(),
                        }),
                        Ok(StreamEvent::ContentBlockStop { index: 0 }),
                        Ok(StreamEvent::MessageStop {
                            stop_reason: StopReason::MaxTokens,
                        }),
                    ]);
                    Ok(Box::pin(s))
                } else {
                    // The recovery compaction's own summarization call — hangs forever so the test
                    // can fire cancellation while it's in flight.
                    Ok(Box::pin(futures::stream::pending()))
                }
            }
        }

        let session_messages = vec![
            Message::user("first request"),
            Message::assistant(vec![ContentBlock::text("first done")]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::text("second done")]),
            Message::user("third request, the one that gets cut off"),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let transport = Arc::new(MaxTokensThenStalledCompaction {
            calls: AtomicUsize::new(0),
        });
        let agent = Agent::new(transport, "claude-opus-4-8").with_compaction(CompactionConfig {
            // Trivially trips `is_hard_overflow` on a `MaxTokens` stop regardless of live prompt size.
            context_window: 10,
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            agent.run_events_cancellable(&mut session, |_| {}, cancel),
        )
        .await
        .expect("cancellation must interrupt the stalled recovery compaction");

        assert!(
            matches!(result, Err(Error::Cancelled)),
            "a cancellation racing the MaxTokens recovery compaction must surface as \
             Error::Cancelled, not silently succeed with Ok(()), got: {result:?}"
        );
    }

    /// Regression: a model that keeps coming back `MaxTokens`-truncated with no tool calls used to spin
    /// the silent-truncation recovery arm forever. Two independent defects combined to remove every
    /// backstop: (1) `overflow_recovered` was reset on the `Ok` path *upstream* of the `MaxTokens`
    /// check that read it, so the "don't recover twice in a row" guard was dead code; and (2) the
    /// recovery arm decremented `steps_this_call`, so `max_steps` could not bound it either. Each round
    /// billed two model calls (the turn + its summarization) and made no progress. Assert the run
    /// terminates, and that it does so in a bounded number of model calls rather than by luck.
    #[tokio::test]
    async fn a_model_that_always_truncates_terminates_instead_of_looping_forever() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct AlwaysTruncates {
            calls: AtomicUsize,
        }
        #[async_trait]
        impl ModelTransport for AlwaysTruncates {
            async fn stream(&self, req: ModelRequest) -> Result<crate::transport::EventStream> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                // The summarization call is the one that carries no tools and a system prompt — answer
                // it with a real (short) summary so `compact()` genuinely succeeds and the loop reaches
                // the retry it used to spin on. Every *live* turn truncates, forever.
                let is_summarization = req.system.is_some() && req.tools.is_empty();
                let text = if is_summarization {
                    "a summary"
                } else {
                    "cut off"
                };
                let stop_reason = if is_summarization {
                    StopReason::EndTurn
                } else {
                    StopReason::MaxTokens
                };
                let s = futures::stream::iter(vec![
                    Ok(StreamEvent::MessageStart),
                    Ok(StreamEvent::TextDelta {
                        index: 0,
                        text: text.into(),
                    }),
                    Ok(StreamEvent::ContentBlockStop { index: 0 }),
                    Ok(StreamEvent::MessageStop { stop_reason }),
                ]);
                Ok(Box::pin(s))
            }
        }

        let mut session = Session::new();
        session.messages = Arc::new(vec![
            Message::user("first request"),
            Message::assistant(vec![ContentBlock::text("first done")]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::text("second done")]),
            Message::user("third request, the one that gets cut off"),
        ]);

        let transport = Arc::new(AlwaysTruncates {
            calls: AtomicUsize::new(0),
        });
        let agent =
            Agent::new(transport.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
                // Trivially trips `is_hard_overflow` on a `MaxTokens` stop regardless of prompt size.
                context_window: 10,
                keep_recent_tokens: 1,
                ..CompactionConfig::default()
            });

        let result = tokio::time::timeout(
            Duration::from_secs(10),
            agent.run_events(&mut session, |_| {}),
        )
        .await
        .expect("an always-truncating model must not loop forever");

        assert!(
            result.is_ok(),
            "the truncated reply is the best available answer and should be reported, got: {result:?}"
        );
        // Exactly one recovery is allowed: live turn + summarization + the retried live turn. The lower
        // bound proves the recovery arm was actually entered (otherwise this test would pass
        // vacuously); the upper bound proves the guard then refused to enter it a second time.
        let calls = transport.calls.load(Ordering::SeqCst);
        assert!(
            (3..=4).contains(&calls),
            "the silent-truncation recovery must fire exactly once — entered (>=3 calls) and then \
             refused a second time (<=4 calls); got {calls} calls"
        );
    }

    #[tokio::test]
    async fn a_panicking_checkpoint_hook_degrades_instead_of_killing_the_run() {
        // `checkpoint` is the host seam most likely to touch failing I/O — its own trait doc invites
        // blocking work like "appending to a session file" — and it was the only one not wrapped in
        // `catch_tool_panic`. A host persistence path that unwraps on a full disk used to unwind
        // straight through the loop and take the whole run (in `serve`, the whole session task) with
        // it. Failing to persist a checkpoint must cost the checkpoint, not the run.
        struct PanickingCheckpoint {
            calls: Arc<std::sync::atomic::AtomicUsize>,
        }
        #[async_trait]
        impl CheckpointHook for PanickingCheckpoint {
            async fn checkpoint(&self, _session: &Session) {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                panic!("disk is full");
            }
        }

        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let transport = Arc::new(MockTransport::new(vec![vec![
            StreamEvent::MessageStart,
            StreamEvent::TextDelta {
                index: 0,
                text: "done".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ]]));
        let agent = Agent::new(transport, "claude-opus-4-8").with_checkpoint_hook(Arc::new(
            PanickingCheckpoint {
                calls: Arc::clone(&calls),
            },
        ));

        let mut session = Session::new();
        session.push(Message::user("hi"));
        let result = agent.run_events(&mut session, |_| {}).await;

        assert!(
            result.is_ok(),
            "a panicking checkpoint hook must not fail the run, got: {result:?}"
        );
        assert!(
            calls.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "the hook must actually have been called (otherwise this passes vacuously)"
        );
        assert!(
            session
                .messages
                .iter()
                .any(|m| matches!(m.role, Role::Assistant)),
            "the run must still have completed its turn normally"
        );
    }

    #[test]
    fn a_rogue_tool_result_is_capped_before_it_can_enter_the_session() {
        // The built-in tools cap themselves at 50 KiB, but `Tool` is a public trait that hosts and MCP
        // servers implement — so the bound has to live in the loop, not in every tool author's head.
        // Anything under the ceiling must pass through byte-for-byte untouched.
        let small = "x".repeat(1024);
        assert_eq!(cap_tool_result(small.clone()), small);

        let rogue = "x".repeat(TOOL_RESULT_MAX_BYTES + 5_000);
        let capped = cap_tool_result(rogue);
        assert!(
            capped.len() < TOOL_RESULT_MAX_BYTES + 500,
            "a rogue result must be clamped to roughly the ceiling, got {} bytes",
            capped.len()
        );
        assert!(
            capped.contains("truncated"),
            "the model must be told the result was clipped rather than silently handed a partial \
             document it would reason over as if complete"
        );

        // Tool output is arbitrary bytes, not ASCII by assumption: truncating mid-codepoint would
        // panic, so the cut must walk back to a char boundary. A multi-byte char straddling the limit
        // is the case that would have blown up.
        let multibyte = "é".repeat(TOOL_RESULT_MAX_BYTES);
        let capped = cap_tool_result(multibyte);
        assert!(capped.contains("truncated"));
    }

    #[test]
    fn is_retryable_mid_stream_matches_both_dialects_truncation_and_overload() {
        assert!(is_retryable_mid_stream(&Error::Transport(
            "Anthropic stream ended before message_stop".into()
        )));
        assert!(is_retryable_mid_stream(&Error::Transport(
            "OpenAI stream ended before finish_reason".into()
        )));
        assert!(is_retryable_mid_stream(&Error::Transport(
            "provider stream error: overloaded_error: Overloaded".into()
        )));
        // A real network failure mid-body-read (connection reset, read timeout, unexpected EOF) —
        // tagged by the transport with the shared marker rather than matched by guessing at
        // `reqwest::Error`'s `Display` wording, which varies by OS/error kind and isn't a contract.
        assert!(is_retryable_mid_stream(&Error::Transport(format!(
            "{MID_STREAM_NETWORK_ERROR}: error reading a body from connection: connection reset \
             by peer (os error 104)"
        ))));
        assert!(is_retryable_mid_stream(&Error::Transport(format!(
            "{MID_STREAM_NETWORK_ERROR}: operation timed out"
        ))));
        // Context overflow is a *different* signal (compact-and-retry owns it) — must not double up.
        assert!(!is_retryable_mid_stream(&Error::Transport(
            "prompt is too long: 250000 tokens > 200000 maximum".into()
        )));
        // An unrelated transport error, and a non-Transport error, are never retryable.
        assert!(!is_retryable_mid_stream(&Error::Transport(
            "gateway returned 401: unauthorized".into()
        )));
        assert!(!is_retryable_mid_stream(&Error::Cancelled));
    }

    #[test]
    fn is_retryable_mid_stream_matches_pis_broader_timeout_patterns() {
        // pi-parity fix (L1): pi matches timeout phrasing as two separate patterns
        // (`/timed? out/i`, `/timeout/i`) — "timed out" alone missed both "time out" (no `d`) and the
        // bare single-word "timeout" a provider/proxy error wrapper commonly uses.
        assert!(is_retryable_mid_stream(&Error::Transport(
            "the operation time out".into()
        )));
        assert!(is_retryable_mid_stream(&Error::Transport(
            "upstream request timeout".into()
        )));
        assert!(is_retryable_mid_stream(&Error::Transport(
            "504 Gateway Timeout".into()
        )));
    }

    #[test]
    fn is_retryable_mid_stream_recognizes_named_in_band_error_types() {
        // Anthropic and OpenAI in-band error *types* that mean "transient, safe to retry" — previously
        // only `overloaded_error` was recognized; a `rate_limit_error`/`server_error`/etc mid-stream
        // event hard-failed the run instead of retrying like pi does.
        for msg in [
            "provider stream error: rate_limit_error: slow down",
            "provider stream error: api_error: internal error, please try again",
            "provider stream error: timeout_error: request timed out",
            "provider stream error: rate_limit_exceeded: slow down",
            "provider stream error: server_error: boom",
            "provider stream error: internal_error: boom",
            "provider stream error: service_unavailable: try later",
        ] {
            assert!(
                is_retryable_mid_stream(&Error::Transport(msg.into())),
                "expected retryable: {msg}"
            );
        }
    }

    #[test]
    fn is_retryable_mid_stream_recognizes_free_text_prose_without_a_recognized_error_type() {
        // LOW pi-parity gap (fixed): pi's `RETRYABLE_PROVIDER_ERROR_PATTERN` also matches plain prose,
        // for a provider/wrapper whose error body has no `error.type` field at all (or a differently-
        // shaped one) — a plain-text/HTML error page, or an OpenRouter-style wrapper message. None of
        // these messages contain any of `MID_STREAM_RETRYABLE_ERROR_TYPES`' type strings, so only the
        // new free-text fallback can catch them.
        for msg in [
            "provider stream error: Rate limit exceeded, slow down",
            "upstream said: 429 Too Many Requests",
            "provider returned error: <html>Internal Server Error</html>",
            "gateway wrapper: Service Unavailable, try again shortly",
            "Provider returned error: bad gateway from upstream",
            "connection error talking to the model provider",
            "the request timed out waiting for a response",
            "Anthropic stream ended without a final message",
        ] {
            assert!(
                is_retryable_mid_stream(&Error::Transport(msg.into())),
                "expected retryable via free-text fallback: {msg}"
            );
        }
        // Prose that happens to share a word with a retryable pattern, but not the whole phrase, must
        // not match — otherwise this degenerates into "any error mentioning a server is retryable".
        assert!(!is_retryable_mid_stream(&Error::Transport(
            "invalid_request_error: 'server' is not a valid parameter name".into()
        )));
    }

    #[test]
    fn is_retryable_mid_stream_recognizes_network_connection_lost() {
        // pi-parity fix (earendil-works/pi#3317, `regressions/3317-network-connection-lost-retry.test.ts`):
        // this exact provider-reported phrasing is distinct from both "network error" and "connection
        // error" above and previously fell through every pattern to a hard, non-retried failure.
        assert!(is_retryable_mid_stream(&Error::Transport(
            "Network connection lost.".into()
        )));
        // is_retryable_whole_run is a strict superset (see retry.rs), so this must hold there too —
        // that's the layer pi's own test actually observes (`auto_retry_start`/`auto_retry_end`).
    }

    #[test]
    fn is_retryable_mid_stream_recognizes_explicit_retry_guidance_phrases() {
        for msg in [
            "provider stream error: server_error: you can retry your request.",
            "please retry your request or contact support",
            "an error occurred; try your request again",
        ] {
            assert!(
                is_retryable_mid_stream(&Error::Transport(msg.into())),
                "expected retryable: {msg}"
            );
        }
    }

    #[test]
    fn is_retryable_mid_stream_still_excludes_permanent_failures() {
        // These in-band error types mean "this exact request will never succeed" — retrying would just
        // burn `MAX_MID_STREAM_RETRIES` attempts for nothing.
        for msg in [
            "provider stream error: invalid_request_error: missing field 'model'",
            "provider stream error: authentication_error: invalid api key",
            "provider stream error: permission_error: not allowed",
            "provider stream error: insufficient_quota: you exceeded your quota",
        ] {
            assert!(
                !is_retryable_mid_stream(&Error::Transport(msg.into())),
                "expected NOT retryable: {msg}"
            );
        }
    }

    #[test]
    fn is_retryable_mid_stream_excludes_a_quota_exhausted_429_even_when_it_also_says_too_many_requests()
     {
        // Real bug found wiring up A-M8's whole-run quota exclusion: a quota-exhausted 429's HTTP
        // status line is routinely the exact same "429 Too Many Requests" wording an ordinary,
        // transient rate limit uses — `MID_STREAM_RETRYABLE_FREE_TEXT_PATTERNS`'s bare "too many
        // requests" phrase matched on that shared status text alone, so this message was retryable
        // here even though `is_retryable_mid_stream_still_excludes_permanent_failures` already proved
        // a quota body *without* that status-line wording correctly wasn't. Since
        // `is_retryable_whole_run` (`crates/agent/src/retry.rs`) OR-composes with this function first,
        // the whole-run layer's own quota exclusion could never even be reached — this function's
        // "yes" always won.
        assert!(!is_retryable_mid_stream(&Error::Transport(
            "gateway returned 429 Too Many Requests: {\"error\":{\"type\":\"insufficient_quota\",\
             \"message\":\"You exceeded your current quota\"}}"
                .into()
        )));
    }

    /// Asserts `actual` falls within the +/-20% jitter band around `nominal` (the unjittered
    /// exponential value), inclusive. Mirrors `client::tests::assert_within_jitter` — jitter (Finding
    /// #32: a pi-parity/consistency fix so this inner retry layer doesn't thundering-herd the same way
    /// the outer whole-run layer's own jitter already guards against) means the exact value is no
    /// longer deterministic, only its range is.
    fn assert_within_jitter(actual: Duration, nominal: Duration) {
        let lo = nominal.mul_f64(0.8);
        let hi = nominal.mul_f64(1.2);
        assert!(
            actual >= lo && actual <= hi,
            "expected {actual:?} within +/-20% of {nominal:?} (i.e. [{lo:?}, {hi:?}])"
        );
    }

    #[test]
    fn mid_stream_backoff_is_exponential_and_capped() {
        assert_within_jitter(mid_stream_backoff(1), MID_STREAM_BASE_BACKOFF);
        assert_within_jitter(mid_stream_backoff(2), MID_STREAM_BASE_BACKOFF * 2);
        assert_within_jitter(mid_stream_backoff(3), MID_STREAM_BASE_BACKOFF * 4);
        assert_eq!(mid_stream_backoff(20), MID_STREAM_MAX_BACKOFF); // saturates, never overflows
    }

    #[test]
    fn mid_stream_backoff_jitter_varies_across_calls_but_stays_in_range() {
        // Finding #32 (pi-parity/consistency fix): proves jitter is actually applied, not just
        // structurally present — the same attempt number, computed repeatedly, must not always
        // produce the identical duration (the thundering-herd scenario this fix closes), while every
        // value still lands within the documented +/-20% band around the unjittered exponential value.
        let nominal = MID_STREAM_BASE_BACKOFF * 2; // attempt 2's unjittered value.
        let samples: Vec<_> = (0..50).map(|_| mid_stream_backoff(2)).collect();
        for &s in &samples {
            assert_within_jitter(s, nominal);
        }
        assert!(
            samples.windows(2).any(|w| w[0] != w[1]),
            "expected varying backoff durations across repeated calls for the same attempt, got \
             identical values every time: {samples:?}"
        );
    }

    #[tokio::test]
    async fn tool_can_terminate_the_run() {
        // A tool returning `terminate` ends the run after its batch — without the model emitting a
        // final turn. Only one model turn is scripted; if the loop asked for a second the mock would
        // be exhausted and the run would error.
        struct ExitTool;
        #[async_trait]
        impl Tool for ExitTool {
            fn name(&self) -> &str {
                "exit"
            }
            fn description(&self) -> &str {
                "End the run."
            }
            fn input_schema(&self) -> Value {
                serde_json::json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok(crate::tool::ToolOutput::text("done").with_terminate(true))
            }
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(ExitTool));
        let (agent, mock) = agent_with(vec![turn::tool_call("tu_1", "exit", "{}")], tools);
        let mut session = Session::new();
        session.user("finish up");
        agent.run(&mut session, |_| {}).await.unwrap();

        // Exactly one model turn, and the tool result was recorded before stopping.
        assert_eq!(mock.calls(), 1);
        assert!(matches!(
            session.messages.last().map(|m| m.content.first()),
            Some(Some(ContentBlock::ToolResult { .. }))
        ));
    }

    #[tokio::test]
    async fn a_terminate_request_only_wins_when_every_call_in_the_batch_agrees() {
        // pi-parity coverage (`packages/agent/test/agent-loop.test.ts`, "should continue after
        // parallel tool calls when not all tool results terminate"): a single tool asking to end the
        // run must not cut off a sibling call the model dispatched in the same batch — only honored
        // when *every* result in the group agrees. `agent.rs`'s own `terminate &= wants_terminate`
        // fold implements this; this pins the actual end-to-end behavior, not just the fold in
        // isolation.
        struct ConditionalExitTool;
        #[async_trait]
        impl Tool for ConditionalExitTool {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "echo the value arg; terminates the run only when value is \"first\""
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] })
            }
            async fn run(
                &self,
                input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                let value = input
                    .get("value")
                    .and_then(Value::as_str)
                    .ok_or_else(|| crate::error::ToolError::InvalidInput("missing value".into()))?;
                Ok(crate::tool::ToolOutput::text(format!("echoed: {value}"))
                    .with_terminate(value == "first"))
            }
        }

        let parallel_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "tool-1".into(),
                name: "echo".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"value":"first"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 1,
                id: "tool-2".into(),
                name: "echo".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 1,
                partial_json: r#"{"value":"second"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 1 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(ConditionalExitTool));
        let (agent, mock) = agent_with(vec![parallel_calls, turn::text("done")], tools);
        let mut session = Session::new();
        session.user("echo both");
        agent.run(&mut session, |_| {}).await.unwrap();

        // Both tool results are recorded, and the run continued into a real second turn instead of
        // stopping just because "first"'s result asked to terminate.
        assert_eq!(mock.calls(), 2, "the batch must not terminate early");
        assert_eq!(
            session.messages.iter().map(|m| m.role).collect::<Vec<_>>(),
            vec![Role::User, Role::Assistant, Role::User, Role::Assistant],
        );
        assert_eq!(
            session.messages[2].content,
            vec![
                ContentBlock::ToolResult {
                    tool_use_id: "tool-1".into(),
                    content: "echoed: first".into(),
                    is_error: false,
                    images: Vec::new(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tool-2".into(),
                    content: "echoed: second".into(),
                    is_error: false,
                    images: Vec::new(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn streaming_tool_emits_progress_before_end() {
        // A tool that streams two chunks via its `ToolProgress` sink before returning. The loop must
        // surface both as `ToolProgress` events, in order, ahead of the tool's `ToolEnd`.
        struct StreamingTool;
        #[async_trait]
        impl Tool for StreamingTool {
            fn name(&self) -> &str {
                "streamer"
            }
            fn description(&self) -> &str {
                "emits progress"
            }
            fn input_schema(&self) -> Value {
                serde_json::json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok("final".into())
            }
            async fn run_streaming(
                &self,
                _: Value,
                progress: &crate::tool::ToolProgress,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                progress.emit("chunk-1", None);
                progress.emit("chunk-2", None);
                Ok("final".into())
            }
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(StreamingTool));
        let (agent, _mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "streamer", "{}"),
                turn::text("done"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("go");

        let mut log: Vec<String> = Vec::new();
        agent
            .run_events(&mut session, |ev| match ev {
                AgentEvent::ToolProgress { snapshot, .. } => {
                    log.push(format!("progress:{snapshot}"))
                }
                AgentEvent::ToolEnd { .. } => log.push("end".into()),
                _ => {}
            })
            .await
            .unwrap();

        let first_end = log.iter().position(|e| e == "end").expect("a ToolEnd");
        assert_eq!(
            &log[..first_end],
            &[
                "progress:chunk-1".to_string(),
                "progress:chunk-2".to_string()
            ],
            "both chunks must arrive, in order, before ToolEnd: {log:?}"
        );
    }

    #[tokio::test]
    async fn genuinely_interleaved_tool_call_deltas_reach_the_callback_live_in_wire_order() {
        // MEDIUM pi-parity gap (fixed): a wire event carrying index 1's argument delta, arriving while
        // index 0 is still open, used to have nowhere to go in the shared `StreamEvent`/`Accumulator`
        // contract — the OpenAI Responses decoder buffered it and replayed it in one burst once index 0
        // finally closed. `Accumulator` now tracks both indices concurrently, so the *loop* forwards
        // every event to the callback the instant it arrives (see `run_turn_once`'s `emit(ev)` call,
        // right after `acc.apply(&ev)` — no batching layer sits between decode and callback). This test
        // proves that live-arrival-order guarantee end to end: index 1's delta lands in the callback
        // between index 0's start and its own eventual close, not deferred until index 0 finishes, and
        // the final assembled message still holds both tool_use blocks, complete and in declaration
        // order, regardless of which one's deltas happened to finish streaming first.
        let scripted_turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "call_a".into(),
                name: "echo".into(),
            },
            // Index 1 opens *before* index 0 closes — genuinely interleaved, not sequential.
            StreamEvent::ToolUseStart {
                index: 1,
                id: "call_b".into(),
                name: "echo".into(),
            },
            // Index 1's delta arrives live, while index 0 is still open.
            StreamEvent::InputJsonDelta {
                index: 1,
                partial_json: r#"{"text":"b"}"#.into(),
            },
            // Index 0 keeps streaming *after* index 1 already has content — proves index 0 was never
            // force-closed by index 1 opening.
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"text":"a"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ContentBlockStop { index: 1 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, _mock) = agent_with(vec![scripted_turn, turn::text("done")], tools);
        let mut session = Session::new();
        session.user("go");

        let mut seen = Vec::new();
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::Stream(StreamEvent::InputJsonDelta { index, .. }) = &ev {
                    seen.push(*index);
                }
            })
            .await
            .unwrap();

        // The callback saw index 1's delta *before* index 0's second delta — the exact scripted wire
        // order — proving nothing buffered/reordered them, live, at the loop level.
        assert_eq!(
            seen,
            vec![1, 0],
            "deltas must reach the callback in true arrival order: {seen:?}"
        );

        // The final message still holds both tool calls, complete and in declaration order (index 0
        // before index 1), regardless of which one's deltas finished streaming first.
        match &session.messages[1].content[..] {
            [
                ContentBlock::ToolUse {
                    id: id_a,
                    input: input_a,
                    ..
                },
                ContentBlock::ToolUse {
                    id: id_b,
                    input: input_b,
                    ..
                },
            ] => {
                assert_eq!(id_a, "call_a");
                assert_eq!(input_a["text"], "a");
                assert_eq!(id_b, "call_b");
                assert_eq!(input_b["text"], "b");
            }
            other => panic!("expected two ordered tool_use blocks, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn text_finals_id_and_phase_thread_through_the_accumulator_into_the_final_block() {
        // LOW pi-parity gap (fixed): a `TextFinal`'s `id`/`phase` (OpenAI Responses' replay metadata
        // for a finished message item — see `ContentBlock::Text`'s doc comment) must survive the fold
        // from `StreamEvent` into the persisted `ContentBlock`, not just be accepted by the event type.
        let scripted_turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::TextDelta {
                index: 0,
                text: "Hel".into(),
            },
            StreamEvent::TextFinal {
                index: 0,
                text: "Hello, world!".into(),
                id: Some("msg_real_1".into()),
                phase: Some("commentary".into()),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ];
        let (agent, _mock) = agent_with(vec![scripted_turn], ToolRegistry::new());
        let mut session = Session::new();
        session.user("go");
        agent.run_events(&mut session, |_| {}).await.unwrap();

        assert_eq!(
            session.messages[1].content[0],
            ContentBlock::Text {
                text: "Hello, world!".into(),
                id: Some("msg_real_1".into()),
                phase: Some("commentary".into()),
            },
            "the TextFinal's id/phase must land on the assembled block, not be dropped: {:?}",
            session.messages[1].content[0]
        );
    }

    #[tokio::test]
    async fn tool_end_streams_in_actual_finish_order_not_call_order() {
        use std::time::Duration;

        // "slow" is *called* first but finishes last; "fast" is called second but finishes
        // immediately. `ToolEnd` must stream live as each call's own result becomes known — so
        // "fast"'s `ToolEnd` arrives before "slow"'s — rather than batching every call's `ToolEnd`
        // until the whole turn's dispatch joins (which would emit them in call order regardless of
        // when each one actually finished: slow, then fast).
        struct SlowTool;
        #[async_trait]
        impl Tool for SlowTool {
            fn name(&self) -> &str {
                "slow"
            }
            fn description(&self) -> &str {
                "finishes last"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                tokio::time::sleep(Duration::from_millis(150)).await;
                Ok("slow-done".into())
            }
        }
        struct FastTool;
        #[async_trait]
        impl Tool for FastTool {
            fn name(&self) -> &str {
                "fast"
            }
            fn description(&self) -> &str {
                "finishes immediately"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok("fast-done".into())
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SlowTool));
        tools.register(Arc::new(FastTool));

        let two_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "s".into(),
                name: "slow".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 0,
                id: "f".into(),
                name: "fast".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let (agent, _mock) = agent_with(vec![two_calls, turn::text("done")], tools);
        let mut session = Session::new();
        session.user("go");

        let mut end_order: Vec<String> = Vec::new();
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::ToolEnd { id, .. } = ev {
                    end_order.push(id);
                }
            })
            .await
            .unwrap();

        assert_eq!(
            end_order,
            vec!["f".to_string(), "s".to_string()],
            "ToolEnd must stream in actual finish order (fast, then slow), not call order: {end_order:?}"
        );
    }

    /// A normalized event "kind" for ordering assertions — mirrors pi's own `normalizeEventOrder`
    /// (`agent-session-retry-events.test.ts`), which also collapses consecutive fine-grained deltas
    /// into a single logical step. `Stream(...)` wraps many more sub-events on our side (per-token
    /// deltas) than pi's coarser `message_update`, so consecutive `Stream` events collapse to one
    /// `"stream"` marker here — the same normalization idea, adapted to this crate's own event shape
    /// rather than pi's exact variant names.
    fn normalized_event_kinds(events: &[AgentEvent]) -> Vec<&'static str> {
        fn kind(ev: &AgentEvent) -> &'static str {
            match ev {
                AgentEvent::AgentStart => "agent_start",
                AgentEvent::TurnStart { .. } => "turn_start",
                AgentEvent::Stream(_) => "stream",
                AgentEvent::ToolStart { .. } => "tool_start",
                AgentEvent::ToolProgress { .. } => "tool_progress",
                AgentEvent::ToolEnd { .. } => "tool_end",
                AgentEvent::TurnEnd { .. } => "turn_end",
                AgentEvent::Steered { .. } => "steered",
                AgentEvent::AgentEnd { .. } => "agent_end",
                AgentEvent::CompactionStart { .. } => "compaction_start",
                AgentEvent::Compacted { .. } => "compacted",
                AgentEvent::CompactionFailed { .. } => "compaction_failed",
                AgentEvent::Error { .. } => "error",
                AgentEvent::ModelSwitched { .. } => "model_switched",
                AgentEvent::ToolsUpdated { .. } => "tools_updated",
            }
        }
        let mut out: Vec<&'static str> = Vec::new();
        for ev in events {
            let k = kind(ev);
            if k == "stream" && out.last() == Some(&"stream") {
                continue;
            }
            out.push(k);
        }
        out
    }

    #[tokio::test]
    async fn emits_the_expected_event_order_for_a_single_prompt() {
        // pi-parity coverage (`agent-session-retry-events.test.ts`, "emits the expected event order
        // for a single prompt"): only pairwise/relative orderings were tested before this — pins the
        // complete sequence for the simplest possible run.
        let (agent, _mock) = agent_with(vec![turn::text("hello")], ToolRegistry::new());
        let mut session = Session::new();
        session.user("hi");
        let mut events = Vec::new();
        agent
            .run_events(&mut session, |ev| events.push(ev))
            .await
            .unwrap();

        assert_eq!(
            normalized_event_kinds(&events),
            vec![
                "agent_start",
                "turn_start",
                "stream",
                "turn_end",
                "agent_end"
            ],
        );
    }

    #[tokio::test]
    async fn emits_the_expected_event_order_for_a_tool_call_turn() {
        // pi-parity coverage (`agent-session-retry-events.test.ts`, "emits the expected event order
        // for a tool call turn"): the full sequence across a tool-dispatch turn followed by the
        // model's final text turn — turn_start/turn_end appears twice, agent_start/agent_end once each.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, _mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"hello"}"#),
                turn::text("done"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("hi");
        let mut events = Vec::new();
        agent
            .run_events(&mut session, |ev| events.push(ev))
            .await
            .unwrap();

        assert_eq!(
            normalized_event_kinds(&events),
            vec![
                "agent_start",
                "turn_start",
                "stream",
                "turn_end",
                "tool_start",
                "tool_end",
                "turn_start",
                "stream",
                "turn_end",
                "agent_end",
            ],
        );
    }

    #[tokio::test]
    async fn a_late_tool_progress_emit_after_the_run_settles_is_silently_dropped_not_a_panic() {
        // pi-parity coverage (`packages/agent/test/agent.test.ts`, "should ignore tool updates after
        // the tool execution settles"): `ToolProgress::emit`'s own doc comment already promises this
        // ("best-effort... dropped rather than erroring") — the channel receiver really is gone once
        // the tool group task exits — but nothing end-to-end proved a tool holding onto its handle past
        // settlement (a fire-and-forget background task, say) can't panic the process or corrupt a
        // later, unrelated run.
        let captured: Arc<std::sync::Mutex<Option<crate::tool::ToolProgress>>> =
            Arc::new(std::sync::Mutex::new(None));

        struct CapturesProgressTool(Arc<std::sync::Mutex<Option<crate::tool::ToolProgress>>>);
        #[async_trait]
        impl Tool for CapturesProgressTool {
            fn name(&self) -> &str {
                "captures_progress"
            }
            fn description(&self) -> &str {
                "stashes its progress handle for the caller to use after settlement"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                unreachable!("run_streaming is always preferred when overridden")
            }
            async fn run_streaming(
                &self,
                _input: Value,
                progress: &crate::tool::ToolProgress,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                progress.emit("running", None);
                *self.0.lock().unwrap() = Some(progress.clone());
                Ok(crate::tool::ToolOutput::text("done").with_terminate(true))
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(CapturesProgressTool(captured.clone())));
        let (agent, _mock) = agent_with(
            vec![turn::tool_call("tu_1", "captures_progress", "{}")],
            tools,
        );
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        // The run has fully settled — its channel's receiving end is gone. Emitting on the stashed
        // handle now must not panic (a real bug here would abort this whole test process, not just
        // fail an assertion) and must be a true no-op.
        let handle = captured
            .lock()
            .unwrap()
            .take()
            .expect("tool ran and stashed its handle");
        handle.emit("late, after settlement", Some(json!({ "status": "late" })));

        // The channel being silently defunct doesn't leak into a completely unrelated later run.
        let mut tools2 = ToolRegistry::new();
        tools2.register(Arc::new(EchoTool));
        let (agent2, _mock2) = agent_with(vec![turn::text("still fine")], tools2);
        let mut session2 = Session::new();
        session2.user("go again");
        agent2.run(&mut session2, |_| {}).await.unwrap();
    }

    #[tokio::test]
    async fn a_stringified_primitive_tool_argument_is_coerced_before_dispatch() {
        // pi-parity coverage (`packages/ai/test/validation.test.ts`): a provider (or an
        // OpenAI-compatible proxy in between) can stringify a primitive the model emitted as
        // genuinely typed — `{"count":"42"}` instead of `{"count":42}`. Without coercion this tool's
        // own `as_i64()` extraction would see `None` and fail with a confusing "missing/wrong-type
        // field" error instead of running normally.
        struct NeedsIntegerTool;
        #[async_trait]
        impl Tool for NeedsIntegerTool {
            fn name(&self) -> &str {
                "double"
            }
            fn description(&self) -> &str {
                "doubles an integer count"
            }
            fn input_schema(&self) -> Value {
                json!({
                    "type": "object",
                    "properties": { "count": { "type": "integer" } },
                    "required": ["count"],
                })
            }
            async fn run(
                &self,
                input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                let count = input.get("count").and_then(Value::as_i64).ok_or_else(|| {
                    crate::error::ToolError::InvalidInput("count must be an integer".into())
                })?;
                Ok((count * 2).to_string().into())
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(NeedsIntegerTool));
        // The model streamed the argument as a JSON *string* ("21"), not a number — exactly the shape
        // coercion exists to normalize before the tool ever sees it.
        let (agent, _mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "double", r#"{"count":"21"}"#),
                turn::text("done"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("double 21");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(
            session.messages[2].content,
            vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: "42".into(),
                is_error: false,
                images: Vec::new(),
            }],
            "the stringified \"21\" must coerce to the integer 21 before the tool runs, not error"
        );
    }

    #[tokio::test]
    async fn dispatch_tells_a_tool_whether_the_active_model_supports_vision() {
        // Task #36 (pi-parity): `read`'s "current model doesn't support images" note (matching pi's
        // `getNonVisionImageNote`) reads a schema-undocumented `_model_supports_vision` field off its
        // input — this pins that dispatch actually injects it, keyed off the model's own capability
        // table entry, for both a vision-capable and a non-vision-capable model.
        struct CapturesVisionFlagTool(Arc<std::sync::Mutex<Option<Value>>>);
        #[async_trait]
        impl Tool for CapturesVisionFlagTool {
            fn name(&self) -> &str {
                "probe"
            }
            fn description(&self) -> &str {
                "captures the _model_supports_vision flag it was dispatched with"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                *self.0.lock().unwrap() = input.get("_model_supports_vision").cloned();
                Ok("ok".into())
            }
        }

        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(CapturesVisionFlagTool(captured.clone())));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "probe", "{}"),
            turn::text("done"),
        ]));
        // o3-mini has no vision support (`models::capabilities`'s own table); claude-opus-4-8 does.
        let agent = Agent::new(mock, "o3-mini")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();
        assert_eq!(
            *captured.lock().unwrap(),
            Some(Value::Bool(false)),
            "a non-vision model must dispatch with _model_supports_vision: false"
        );

        let captured2 = Arc::new(std::sync::Mutex::new(None));
        let mut tools2 = ToolRegistry::new();
        tools2.register(Arc::new(CapturesVisionFlagTool(captured2.clone())));
        let mock2 = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "probe", "{}"),
            turn::text("done"),
        ]));
        let agent2 = Agent::new(mock2, "claude-opus-4-8")
            .with_tools(tools2)
            .with_max_steps(8);
        let mut session2 = Session::new();
        session2.user("go");
        agent2.run(&mut session2, |_| {}).await.unwrap();
        assert_eq!(
            *captured2.lock().unwrap(),
            Some(Value::Bool(true)),
            "a vision-capable model must dispatch with _model_supports_vision: true"
        );
    }

    #[tokio::test]
    async fn with_block_images_forces_the_vision_downgrade_path_on_a_vision_capable_model() {
        // Task #26 (pi-parity): an operator-facing override that forces the same downgrade path
        // regardless of the model's real capability — the mechanism a `--block-images` CLI flag wires
        // into. Must flip `_model_supports_vision` to `false` even for a model whose capability table
        // entry says it genuinely supports vision.
        struct CapturesVisionFlagTool(Arc<std::sync::Mutex<Option<Value>>>);
        #[async_trait]
        impl Tool for CapturesVisionFlagTool {
            fn name(&self) -> &str {
                "probe"
            }
            fn description(&self) -> &str {
                "captures the _model_supports_vision flag it was dispatched with"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                *self.0.lock().unwrap() = input.get("_model_supports_vision").cloned();
                Ok("ok".into())
            }
        }

        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(CapturesVisionFlagTool(captured.clone())));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "probe", "{}"),
            turn::text("done"),
        ]));
        // claude-opus-4-8 is vision-capable — without the override this would dispatch `true`.
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_block_images(true);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();
        assert_eq!(
            *captured.lock().unwrap(),
            Some(Value::Bool(false)),
            "with_block_images(true) must force the downgrade path even on a vision-capable model"
        );
    }

    #[tokio::test]
    async fn block_images_defaults_to_false_leaving_a_vision_capable_models_flag_untouched() {
        struct CapturesVisionFlagTool(Arc<std::sync::Mutex<Option<Value>>>);
        #[async_trait]
        impl Tool for CapturesVisionFlagTool {
            fn name(&self) -> &str {
                "probe"
            }
            fn description(&self) -> &str {
                "captures the _model_supports_vision flag it was dispatched with"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                *self.0.lock().unwrap() = input.get("_model_supports_vision").cloned();
                Ok("ok".into())
            }
        }

        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(CapturesVisionFlagTool(captured.clone())));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "probe", "{}"),
            turn::text("done"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();
        assert_eq!(*captured.lock().unwrap(), Some(Value::Bool(true)));
    }

    #[tokio::test]
    async fn request_snapshots_are_isolated_across_turns() {
        // History is shared via `Arc`, so copy-on-write in `Session::push` must keep each request's
        // snapshot frozen: a later turn appending tool results must not retroactively mutate the
        // messages an earlier request was built from.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
                turn::text("done"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("say pong");
        agent.run(&mut session, |_| {}).await.unwrap();

        let reqs = mock.requests();
        // First request carried only the seed user turn; the second saw more (assistant + result).
        assert_eq!(reqs[0].messages.len(), 1);
        assert!(reqs[1].messages.len() > reqs[0].messages.len());
    }

    #[tokio::test]
    async fn independent_tool_calls_run_concurrently() {
        use std::time::Duration;
        use tokio::sync::Barrier;

        // A tool that blocks on a shared 2-party barrier: it only returns once *both* tools are in
        // flight. Under serial dispatch the first call would wait forever for the second to start, so
        // the run completing at all proves the calls overlap.
        struct BarrierTool {
            id: &'static str,
            barrier: Arc<Barrier>,
        }
        #[async_trait]
        impl Tool for BarrierTool {
            fn name(&self) -> &str {
                self.id
            }
            fn description(&self) -> &str {
                "waits on a shared barrier"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                self.barrier.wait().await;
                Ok(self.id.into())
            }
        }

        let barrier = Arc::new(Barrier::new(2));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(BarrierTool {
            id: "t1",
            barrier: barrier.clone(),
        }));
        tools.register(Arc::new(BarrierTool {
            id: "t2",
            barrier: barrier.clone(),
        }));

        // One assistant turn that asks for both tools, then a turn that ends the conversation.
        let two_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "a".into(),
                name: "t1".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 0,
                id: "b".into(),
                name: "t2".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let (agent, _mock) = agent_with(vec![two_calls, turn::text("done")], tools);

        let mut session = Session::new();
        session.user("go");
        // Serial execution would deadlock on the barrier; bound the test so a regression fails fast
        // instead of hanging.
        tokio::time::timeout(Duration::from_secs(5), agent.run(&mut session, |_| {}))
            .await
            .expect("tools did not run concurrently (barrier deadlock under serial dispatch)")
            .unwrap();

        // Results fed back in call order on a *single* user turn (Anthropic rejects consecutive
        // same-role messages): user, assistant(2× tool_use), user(2× tool_result), assistant(text).
        assert_eq!(session.messages.len(), 4);
        assert_eq!(session.messages[2].role, Role::User);
        assert_eq!(session.messages[2].content.len(), 2);
        match (
            &session.messages[2].content[0],
            &session.messages[2].content[1],
        ) {
            (
                ContentBlock::ToolResult {
                    tool_use_id: a,
                    content: ca,
                    ..
                },
                ContentBlock::ToolResult {
                    tool_use_id: b,
                    content: cb,
                    ..
                },
            ) => {
                assert_eq!((a.as_str(), ca.as_str()), ("a", "t1"));
                assert_eq!((b.as_str(), cb.as_str()), ("b", "t2"));
            }
            other => panic!("expected two ordered tool_result messages, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn same_write_target_calls_run_sequentially() {
        // Two calls that report the same `write_target` (the model batching two `edit`s against one
        // file) must not race on disk: each tool independently reads-modifies-writes, so unordered
        // execution could drop one write or interleave both. A tool that records start/end markers
        // around a yield point proves the loop serializes same-target calls — if it didn't, the second
        // call's "start" would land before the first's "end".
        struct RecordingTool {
            id: &'static str,
            log: Arc<std::sync::Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl Tool for RecordingTool {
            fn name(&self) -> &str {
                self.id
            }
            fn description(&self) -> &str {
                "records start/end order"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            fn write_target(&self, input: &Value) -> Option<String> {
                input
                    .get("path")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                self.log.lock().unwrap().push(format!("start:{}", self.id));
                tokio::task::yield_now().await;
                self.log.lock().unwrap().push(format!("end:{}", self.id));
                Ok(self.id.into())
            }
        }

        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(RecordingTool {
            id: "edit",
            log: log.clone(),
        }));
        tools.register(Arc::new(RecordingTool {
            id: "write",
            log: log.clone(),
        }));

        // Two different tools, both targeting "foo.rs" — `write_target` groups by path, not tool name.
        let two_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "a".into(),
                name: "edit".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"path":"foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 0,
                id: "b".into(),
                name: "write".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"path":"foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let (agent, _mock) = agent_with(vec![two_calls, turn::text("done")], tools);

        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        // Call order ("edit" before "write") must be preserved, and neither call's "start" may land
        // between the other's "start" and "end" — i.e. no interleaving.
        assert_eq!(
            *log.lock().unwrap(),
            vec!["start:edit", "end:edit", "start:write", "end:write"],
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelling_a_write_holds_its_lock_until_the_in_flight_blocking_write_lands() {
        // Regression: a guarded tool (one reporting a `write_target`) whose actual write runs on a
        // `spawn_blocking` thread — the `edit` tool — must not release its registry write-lock the
        // instant a cancelled dispatch future is dropped. tokio cannot cancel a running blocking task,
        // so on cancellation that future is abandoned while the blocking write *runs on regardless*;
        // if the write-lock guard were held only by the dispatch future (as it was), it would release
        // with the write still physically in flight, letting another turn/session acquire the same
        // path's lock and interleave — a lost update (see `write_lock.rs`, and `edit.rs`'s
        // `run_inner`). The fix rides an `Arc` clone of the guard *into* the `spawn_blocking` closure
        // (via `ToolProgress::write_lock_keepalive`) so the lock releases from the blocking thread when
        // the write completes. This tool models `edit`'s exact shape — keepalive moved into a
        // `spawn_blocking` we can hold open on demand — and drives it through the *real* dispatch loop.
        //
        // Deterministic, not timing-based: the blocking "write" parks on a channel the test controls,
        // so while it's parked the lock is *genuinely* held and a competing `lock()` blocks forever
        // (the negative wait below can only ever time out); it frees only once the test releases it.
        struct BlockingWriteTool {
            // A oneshot the closure fires the moment its (keepalive-holding) blocking write begins, so
            // the test knows the write is in flight before it cancels.
            started: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
            // The closure blocks on this until the test releases it — standing in for a slow, atomic,
            // non-cancellable `rename(2)`.
            release: std::sync::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
        }
        #[async_trait]
        impl Tool for BlockingWriteTool {
            fn name(&self) -> &str {
                "edit"
            }
            fn description(&self) -> &str {
                "models edit's spawn_blocking write"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            fn write_target(&self, input: &Value) -> Option<String> {
                input
                    .get("path")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            }
            async fn run_streaming(
                &self,
                _input: Value,
                progress: &crate::tool::ToolProgress,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                let keepalive = progress.write_lock_keepalive();
                let started = self.started.lock().unwrap().take().unwrap();
                let release = self.release.lock().unwrap().take().unwrap();
                match tokio::task::spawn_blocking(move || {
                    // Exactly the fix: the guard rides into the blocking closure, so it drops here —
                    // when the write finishes — not on the reactor when the dispatch future is dropped.
                    let _keepalive = keepalive;
                    started.send(()).ok();
                    // Block until the test lets the "write" complete. A detached (cancelled) closure
                    // reaches and stays here, keepalive in hand.
                    release.recv().ok();
                })
                .await
                {
                    Ok(()) => Ok("wrote".into()),
                    Err(e) => std::panic::resume_unwind(e.into_panic()),
                }
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                unreachable!("dispatch drives run_streaming")
            }
        }

        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(BlockingWriteTool {
            started: std::sync::Mutex::new(Some(started_tx)),
            release: std::sync::Mutex::new(Some(release_rx)),
        }));

        // The registry is shared exactly as a `serve` process shares one across sessions — the seam the
        // lock exists to protect. The competing `lock()` calls below go through this same registry.
        let registry = Arc::new(crate::write_lock::WriteLockRegistry::new());

        let one_call = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "a".into(),
                name: "edit".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"path":"target.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let mock = Arc::new(MockTransport::new(vec![one_call, turn::text("done")]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_write_locks(registry.clone())
            .with_max_steps(8);

        let cancel = CancellationToken::new();
        let cancel_for_run = cancel.clone();
        let run = tokio::spawn(async move {
            let mut session = Session::new();
            session.user("go");
            // Cancellation makes this return; its result (Ok/cancelled) is not what we're asserting.
            let _ = agent
                .run_events_cancellable(&mut session, |_| {}, cancel_for_run)
                .await;
        });

        // The blocking write is now in flight, keepalive in hand.
        started_rx
            .await
            .expect("tool's blocking write should start");
        // Cancel: dispatch abandons the tool future (drops its own guard clone), but the detached
        // blocking closure keeps the write — and its keepalive — alive.
        cancel.cancel();
        run.await
            .expect("run task should return promptly on cancel");

        // The write is still parked, so its lock must still be held: a competing locker on the same
        // path cannot acquire. Pre-fix (guard held only by the dropped dispatch future) it would.
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(200),
                registry.lock("target.rs"),
            )
            .await
            .is_err(),
            "write lock was released while the write was still physically in flight",
        );

        // Let the write complete; the guard now drops on the blocking thread and the lock frees.
        release_tx.send(()).expect("release the blocking write");
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            registry.lock("target.rs"),
        )
        .await
        .expect("write lock must release once the in-flight write completes");
    }

    #[tokio::test]
    async fn conservative_exclusive_call_serializes_the_whole_turn() {
        // A `bash`-like tool (no `write_target` — its scope can't be named) batched alongside an
        // `edit`-like tool on a *different* path would normally land in two distinct groups and run
        // fully concurrently. `conservative_exclusive` must still force the whole turn to one call at
        // a time, so the opaque call can't race the path-targeted one on disk.
        struct RecordingTool {
            id: &'static str,
            target: Option<&'static str>,
            exclusive: bool,
            log: Arc<std::sync::Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl Tool for RecordingTool {
            fn name(&self) -> &str {
                self.id
            }
            fn description(&self) -> &str {
                "records start/end order"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            fn write_target(&self, _input: &Value) -> Option<String> {
                self.target.map(str::to_string)
            }
            fn conservative_exclusive(&self) -> bool {
                self.exclusive
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                self.log.lock().unwrap().push(format!("start:{}", self.id));
                tokio::task::yield_now().await;
                self.log.lock().unwrap().push(format!("end:{}", self.id));
                Ok(self.id.into())
            }
        }

        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(RecordingTool {
            id: "edit",
            target: Some("foo.rs"),
            exclusive: false,
            log: log.clone(),
        }));
        tools.register(Arc::new(RecordingTool {
            id: "bash",
            target: None,
            exclusive: true,
            log: log.clone(),
        }));

        let two_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "a".into(),
                name: "edit".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"path":"foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 0,
                id: "b".into(),
                name: "bash".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"command":"black foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let (agent, _mock) = agent_with(vec![two_calls, turn::text("done")], tools);

        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        // Which group runs first is unspecified (same as any other two distinct groups — cross-group
        // order never reaches the transcript), but neither call's "start" may land between the
        // other's "start" and "end": no interleaving, even though the two calls report different (and
        // no) targets and would normally be two independent, concurrently-run groups.
        let log = log.lock().unwrap().clone();
        assert!(
            log == vec!["start:edit", "end:edit", "start:bash", "end:bash"]
                || log == vec!["start:bash", "end:bash", "start:edit", "end:edit"],
            "calls must not interleave: {log:?}"
        );
    }

    #[tokio::test]
    async fn same_write_target_serializes_across_two_agent_runs_sharing_a_registry() {
        // The per-turn grouping above only reaches calls within one turn. Two *separate* `Agent`s
        // (e.g. one per session in a `serve` process) targeting the same path must still serialize
        // when they share a `WriteLockRegistry` — proving the exclusivity extends across turn/session
        // boundaries, not just within a single dispatch.
        struct RecordingTool {
            id: &'static str,
            log: Arc<std::sync::Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl Tool for RecordingTool {
            fn name(&self) -> &str {
                self.id
            }
            fn description(&self) -> &str {
                "records start/end order"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            fn write_target(&self, _input: &Value) -> Option<String> {
                Some("shared.rs".to_string())
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                self.log.lock().unwrap().push(format!("start:{}", self.id));
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                self.log.lock().unwrap().push(format!("end:{}", self.id));
                Ok(self.id.into())
            }
        }

        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let registry = Arc::new(crate::write_lock::WriteLockRegistry::new());

        let mut tools_a = ToolRegistry::new();
        tools_a.register(Arc::new(RecordingTool {
            id: "a",
            log: log.clone(),
        }));
        let mock_a = Arc::new(MockTransport::new(vec![
            turn::tool_call("t", "a", "{}"),
            turn::text("done"),
        ]));
        let agent_a = Agent::new(mock_a, "claude-opus-4-8")
            .with_tools(tools_a)
            .with_write_locks(registry.clone());

        let mut tools_b = ToolRegistry::new();
        tools_b.register(Arc::new(RecordingTool {
            id: "b",
            log: log.clone(),
        }));
        let mock_b = Arc::new(MockTransport::new(vec![
            turn::tool_call("t", "b", "{}"),
            turn::text("done"),
        ]));
        let agent_b = Agent::new(mock_b, "claude-opus-4-8")
            .with_tools(tools_b)
            .with_write_locks(registry.clone());

        let mut session_a = Session::new();
        session_a.user("go");
        let mut session_b = Session::new();
        session_b.user("go");

        let (res_a, res_b) = tokio::join!(
            agent_a.run(&mut session_a, |_| {}),
            agent_b.run(&mut session_b, |_| {}),
        );
        res_a.unwrap();
        res_b.unwrap();

        // Neither run's "start" may land between the other's "start" and "end" — i.e. no interleaving
        // across the two separate `Agent`/session pairs.
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 4);
        assert!(
            (*log == vec!["start:a", "end:a", "start:b", "end:b"])
                || (*log == vec!["start:b", "end:b", "start:a", "end:a"]),
            "expected the two runs' start/end pairs to never interleave, got: {log:?}"
        );
    }

    #[tokio::test]
    async fn tool_group_concurrency_is_capped() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        // A tool that tracks how many instances of itself are in flight at once. Proves two things
        // about `MAX_CONCURRENT_TOOL_GROUPS`: that calls actually overlap (not silently serialized),
        // and that overlap never exceeds the cap even when a turn batches well more calls than it.
        struct CountingTool {
            in_flight: Arc<AtomicUsize>,
            max_seen: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl Tool for CountingTool {
            fn name(&self) -> &str {
                "count"
            }
            fn description(&self) -> &str {
                "tracks concurrent in-flight calls"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_seen.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(crate::tool::ToolOutput::default())
            }
        }

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(CountingTool {
            in_flight: in_flight.clone(),
            max_seen: max_seen.clone(),
        }));

        // One assistant turn batching far more calls than MAX_CONCURRENT_TOOL_GROUPS — all to the
        // same read-only tool, so each call is its own group (no `write_target`) and nothing
        // serializes them except the concurrency cap itself.
        const N: usize = 20;
        let mut many_calls = vec![StreamEvent::MessageStart];
        for i in 0..N {
            many_calls.push(StreamEvent::ToolUseStart {
                index: 0,
                id: format!("c{i}"),
                name: "count".into(),
            });
            many_calls.push(StreamEvent::ContentBlockStop { index: 0 });
        }
        many_calls.push(StreamEvent::MessageStop {
            stop_reason: StopReason::ToolUse,
        });

        let (agent, _mock) = agent_with(vec![many_calls, turn::text("done")], tools);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        // Every call's result landed, in call order — proves the cap's index reassembly
        // (`results[i]`) doesn't drop or scramble a result when more groups queue behind the cap
        // than can run at once.
        assert_eq!(session.messages[2].content.len(), N);
        for (i, block) in session.messages[2].content.iter().enumerate() {
            match block {
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    assert_eq!(tool_use_id, &format!("c{i}"));
                }
                other => panic!("expected ToolResult, got {other:?}"),
            }
        }

        // The cap was actually exercised, not accidentally serial...
        let max = max_seen.load(Ordering::SeqCst);
        assert!(
            max > 1,
            "calls ran fully serially — concurrency isn't happening at all"
        );
        // ...and never exceeded MAX_CONCURRENT_TOOL_GROUPS, even with N=20 calls in flight.
        assert!(
            max <= MAX_CONCURRENT_TOOL_GROUPS,
            "observed {max} calls in flight at once, exceeding the cap of {MAX_CONCURRENT_TOOL_GROUPS}"
        );
    }

    #[tokio::test]
    async fn before_tool_call_runs_in_call_order_even_for_independent_concurrent_calls() {
        // pi-parity fix: `before_tool_call` used to be invoked from *inside* each write-target
        // group's own async block, and groups were dispatched via `HashMap` iteration order (not
        // call order) — for a turn batching several independent (different-target) calls, a later
        // call's gate could run before, or concurrently with, an earlier call's. pi's
        // `prepareToolCall` always resolves every call sequentially, in call order, before any call's
        // actual execution begins (`executeToolCallsParallel`, agent-loop.ts:451-516) — a
        // concurrency-aware permission hook (a rate limiter, a policy reasoning about "what's already
        // running") depends on that same guarantee. Registers tools in a deliberately scrambled order
        // so a `HashMap`-iteration-order bug wouldn't accidentally happen to match call order anyway.
        struct RecordingHook {
            order: Arc<std::sync::Mutex<Vec<String>>>,
        }
        #[async_trait]
        impl AgentHooks for RecordingHook {
            async fn before_tool_call(
                &self,
                name: &str,
                _input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                self.order.lock().unwrap().push(name.to_string());
                None
            }
        }
        struct NamedEchoTool(&'static str);
        #[async_trait]
        impl Tool for NamedEchoTool {
            fn name(&self) -> &str {
                self.0
            }
            fn description(&self) -> &str {
                "echoes its own name immediately"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok(self.0.into())
            }
        }

        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut tools = ToolRegistry::new();
        for name in ["e", "b", "d", "a", "c"] {
            tools.register(Arc::new(NamedEchoTool(name)));
        }

        // Five independent (no `write_target`, so each is its own group) calls, well under
        // `MAX_CONCURRENT_TOOL_GROUPS`, requested in this exact order.
        let call_order = ["a", "b", "c", "d", "e"];
        let mut scripted_turn = vec![StreamEvent::MessageStart];
        for (i, name) in call_order.iter().enumerate() {
            scripted_turn.push(StreamEvent::ToolUseStart {
                index: 0,
                id: format!("t{i}"),
                name: name.to_string(),
            });
            scripted_turn.push(StreamEvent::ContentBlockStop { index: 0 });
        }
        scripted_turn.push(StreamEvent::MessageStop {
            stop_reason: StopReason::ToolUse,
        });

        let mock = Arc::new(MockTransport::new(vec![scripted_turn, turn::text("done")]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_hooks(Arc::new(RecordingHook {
                order: order.clone(),
            }));
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(
            *order.lock().unwrap(),
            call_order.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "before_tool_call must fire in call order, not HashMap-grouping order"
        );
    }

    #[tokio::test]
    async fn with_sequential_tools_forces_one_group_in_flight_at_a_time() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        // Same shape as `tool_group_concurrency_is_capped`, but with `with_sequential_tools(true)` set
        // — a batch of calls that would otherwise overlap up to `MAX_CONCURRENT_TOOL_GROUPS` at once
        // must now never observe more than 1 in flight, proving the host-level toggle actually
        // overrides the default bounded-concurrent dispatch rather than just existing unused.
        struct CountingTool {
            in_flight: Arc<AtomicUsize>,
            max_seen: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl Tool for CountingTool {
            fn name(&self) -> &str {
                "count"
            }
            fn description(&self) -> &str {
                "tracks concurrent in-flight calls"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_seen.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(crate::tool::ToolOutput::default())
            }
        }

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(CountingTool {
            in_flight: in_flight.clone(),
            max_seen: max_seen.clone(),
        }));

        const N: usize = 20;
        let mut many_calls = vec![StreamEvent::MessageStart];
        for i in 0..N {
            many_calls.push(StreamEvent::ToolUseStart {
                index: 0,
                id: format!("c{i}"),
                name: "count".into(),
            });
            many_calls.push(StreamEvent::ContentBlockStop { index: 0 });
        }
        many_calls.push(StreamEvent::MessageStop {
            stop_reason: StopReason::ToolUse,
        });

        let (agent, _mock) = agent_with(vec![many_calls, turn::text("done")], tools);
        let agent = agent.with_sequential_tools(true);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(session.messages[2].content.len(), N);
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "sequential_tools must force exactly one group in flight at a time"
        );
    }

    #[tokio::test]
    async fn a_tools_execution_mode_sequential_routes_the_whole_batch_through_the_interleaved_path()
    {
        // Task #28 (pi-parity): the default dispatch always gates the *whole* batch (every call's
        // `before_tool_call`) before any call's execution begins — a permission hook reasoning about
        // "what's already run" can't see call 1's result while gating call 2. A tool naming
        // `ToolExecutionMode::Sequential` must route the whole turn's batch through the fully-
        // interleaved gate→execute→finalize-per-call path instead: call 1 completely resolved
        // (gated, executed, and `after_tool_call`-rewritten) before call 2's own gate even starts.
        use std::sync::atomic::{AtomicBool, Ordering};

        struct FirstTool(Arc<AtomicBool>);
        #[async_trait]
        impl Tool for FirstTool {
            fn name(&self) -> &str {
                "first"
            }
            fn description(&self) -> &str {
                "the first call in the batch"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                self.0.store(true, Ordering::SeqCst);
                Ok("first-done".into())
            }
            fn execution_mode(&self) -> Option<crate::tool::ToolExecutionMode> {
                // Only this one call opts in — the whole batch must still route through the
                // interleaved path, not just this call.
                Some(crate::tool::ToolExecutionMode::Sequential)
            }
        }

        struct SecondTool;
        #[async_trait]
        impl Tool for SecondTool {
            fn name(&self) -> &str {
                "second"
            }
            fn description(&self) -> &str {
                "the second call in the batch"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok("second-done".into())
            }
        }

        struct ObservesFirstAtGateTime {
            first_has_run: Arc<AtomicBool>,
            observed_first_done_at_gate_time: Arc<AtomicBool>,
        }
        #[async_trait]
        impl AgentHooks for ObservesFirstAtGateTime {
            async fn before_tool_call(
                &self,
                name: &str,
                _input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                if name == "second" {
                    // The interleaved path must have already fully executed "first" by the time
                    // "second"'s own gate runs — the default gate-the-whole-batch-first path never
                    // gives this hook that guarantee.
                    self.observed_first_done_at_gate_time
                        .store(self.first_has_run.load(Ordering::SeqCst), Ordering::SeqCst);
                }
                None
            }
        }

        let first_has_run = Arc::new(AtomicBool::new(false));
        let observed_first_done_at_gate_time = Arc::new(AtomicBool::new(false));

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(FirstTool(first_has_run.clone())));
        tools.register(Arc::new(SecondTool));

        let two_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "c1".into(),
                name: "first".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 0,
                id: "c2".into(),
                name: "second".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let (agent, _mock) = agent_with(vec![two_calls, turn::text("done")], tools);
        let agent = agent.with_hooks(Arc::new(ObservesFirstAtGateTime {
            first_has_run: first_has_run.clone(),
            observed_first_done_at_gate_time: observed_first_done_at_gate_time.clone(),
        }));
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert!(
            observed_first_done_at_gate_time.load(Ordering::SeqCst),
            "the interleaved path must fully resolve call 1 before call 2's own gate starts"
        );
        // The transcript itself must still come out correct and in call order regardless of dispatch
        // path.
        assert_eq!(
            session.messages[2].content,
            vec![
                ContentBlock::ToolResult {
                    tool_use_id: "c1".into(),
                    content: "first-done".into(),
                    is_error: false,
                    images: Vec::new(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "c2".into(),
                    content: "second-done".into(),
                    is_error: false,
                    images: Vec::new(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn unknown_tool_yields_error_result() {
        let (agent, _mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "nonexistent", "{}"),
                turn::text("ok"),
            ],
            ToolRegistry::new(),
        );
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        match &session.messages[2].content[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(is_error);
                assert!(content.contains("unknown tool"));
            }
            other => panic!("expected error tool_result, got {other:?}"),
        }
    }

    /// Pi-parity fix: pi's `prepareToolCall` looks up the tool *first* — an unregistered tool name never
    /// reaches `config.beforeToolCall` at all (`agent-loop.ts`'s `prepareToolCall`, `!tool` branch).
    /// This crate used to invoke `before_tool_call` unconditionally, coercion and all, before the
    /// "tool not found" check ran (in the execution phase, much later) — a permission hook would see (and
    /// could even block!) a call that was never going to run anyway. Confirms the hook is skipped
    /// entirely on the default (non-interleaved) dispatch path, while the exact same "unknown tool: …"
    /// error result from `unknown_tool_yields_error_result` above is still produced.
    #[tokio::test]
    async fn before_tool_call_is_not_invoked_for_an_unregistered_tool_name() {
        struct RecordsCalls(std::sync::Mutex<Vec<String>>);
        #[async_trait]
        impl AgentHooks for RecordsCalls {
            async fn before_tool_call(
                &self,
                name: &str,
                _input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                self.0.lock().unwrap().push(name.to_string());
                None
            }
        }

        let (agent, _mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "nonexistent", "{}"),
                turn::text("ok"),
            ],
            ToolRegistry::new(),
        );
        let hook = Arc::new(RecordsCalls(std::sync::Mutex::new(Vec::new())));
        let agent = agent.with_hooks(hook.clone());
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert!(
            hook.0.lock().unwrap().is_empty(),
            "before_tool_call must never be invoked for a tool name that isn't registered at all"
        );
        match &session.messages[2].content[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(is_error);
                assert!(content.contains("unknown tool"));
            }
            other => panic!("expected error tool_result, got {other:?}"),
        }
    }

    /// Same fix as the test above, exercised on the *interleaved* dispatch path
    /// (`Agent::run_tool_calls_interleaved`) instead — a batch containing an unregistered tool name
    /// alongside a genuinely `ToolExecutionMode::Sequential` tool routes the whole turn through the
    /// interleaved gate→execute→finalize-per-call loop, which had its own independent copy of the same
    /// premature-`before_tool_call` bug.
    #[tokio::test]
    async fn before_tool_call_is_not_invoked_for_an_unregistered_tool_name_on_the_interleaved_path()
    {
        struct SeqTool;
        #[async_trait]
        impl Tool for SeqTool {
            fn name(&self) -> &str {
                "seq"
            }
            fn description(&self) -> &str {
                "a tool that forces the interleaved dispatch path"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok("seq-done".into())
            }
            fn execution_mode(&self) -> Option<crate::tool::ToolExecutionMode> {
                Some(crate::tool::ToolExecutionMode::Sequential)
            }
        }

        struct RecordsCalls(std::sync::Mutex<Vec<String>>);
        #[async_trait]
        impl AgentHooks for RecordsCalls {
            async fn before_tool_call(
                &self,
                name: &str,
                _input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                self.0.lock().unwrap().push(name.to_string());
                None
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SeqTool));

        // Both calls land in the same assistant turn — one names a genuinely unregistered tool, the
        // other names the registered `Sequential` tool that forces the whole batch through
        // `run_tool_calls_interleaved`.
        let two_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "tu_1".into(),
                name: "nonexistent".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: "{}".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 0,
                id: "tu_2".into(),
                name: "seq".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: "{}".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let (agent, _mock) = agent_with(vec![two_calls, turn::text("ok")], tools);
        let hook = Arc::new(RecordsCalls(std::sync::Mutex::new(Vec::new())));
        let agent = agent.with_hooks(hook.clone());
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(
            hook.0.lock().unwrap().as_slice(),
            ["seq"],
            "before_tool_call must fire for the registered call but never for the unregistered one, \
             even when both share a batch routed through the interleaved path"
        );
        match &session.messages[2].content[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(is_error);
                assert!(content.contains("unknown tool"));
            }
            other => panic!("expected error tool_result for the unregistered call, got {other:?}"),
        }
        match &session.messages[2].content[1] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(!is_error);
                assert_eq!(content, "seq-done");
            }
            other => {
                panic!("expected a successful tool_result for the registered call, got {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn failing_tool_is_reported_not_fatal() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, _mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"wrong":"key"}"#),
                turn::text("recovered"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();
        match &session.messages[2].content[0] {
            ContentBlock::ToolResult { is_error, .. } => assert!(is_error),
            other => panic!("expected error tool_result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_tool_args_become_recoverable_error_result() {
        // The stream opens a tool call but its argument fragments never form valid JSON. This must
        // not abort the run: the loop feeds back an error tool_result and the model recovers.
        let bad_turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "tu_1".into(),
                name: "echo".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: r#"{"text":"#.into(), // truncated — not parseable
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, _mock) = agent_with(vec![bad_turn, turn::text("recovered")], tools);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        // The run completed; the tool_result for the malformed call is flagged as an error and the
        // assistant tool_use it answers carries a wire-valid (empty object) input, not null.
        match &session.messages[1].content[0] {
            ContentBlock::ToolUse { input, .. } => assert_eq!(input, &json!({})),
            other => panic!("expected a tool_use block, got {other:?}"),
        }
        match &session.messages[2].content[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(is_error);
                assert!(content.contains("not valid JSON"), "got: {content}");
            }
            other => panic!("expected error tool_result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_tool_call_truncated_mid_value_string_recovers_a_partial_object_not_an_empty_one() {
        // Unlike `malformed_tool_args_become_recoverable_error_result` above (truncated right after a
        // key, before any value — genuinely unrecoverable), this stream is cut off mid-*value*, the
        // shape `close_incomplete_json` exists to fix: the tool must see the actual partial content
        // it received, not an empty `{}` with the call flagged as malformed.
        let truncated_turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "tu_1".into(),
                name: "echo".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                // `text` matches `EchoTool`'s own required field, so a successful recovery both
                // produces a valid `tool_use` input *and* lets the tool actually run and echo it back
                // — not merely parse, but genuinely usable end to end.
                partial_json: r#"{"text":"hello wor"#.into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, _mock) = agent_with(vec![truncated_turn, turn::text("done")], tools);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        match &session.messages[1].content[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert_eq!(input["text"], "hello wor", "got: {input:?}");
            }
            other => panic!("expected a tool_use block, got {other:?}"),
        }
        // Not flagged as malformed, and the tool actually ran on the recovered partial value — a real
        // echoed result, not an error placeholder.
        match &session.messages[2].content[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(!is_error, "recovered call must not be treated as malformed");
                assert_eq!(content, "hello wor");
            }
            other => panic!("expected a tool_result block, got {other:?}"),
        }
    }

    #[test]
    fn repair_json_escapes_a_raw_control_character_inside_a_string() {
        // A large `write`/`edit` argument streamed with a literal newline byte instead of `\n` — the
        // motivating real-world case: the exact first-parse-fails-but-shouldn't shape this exists for.
        let raw = "{\"content\":\"line one\nline two\"}";
        assert!(
            serde_json::from_str::<Value>(raw).is_err(),
            "must actually be invalid first"
        );
        let repaired = repair_json(raw);
        let v: Value = serde_json::from_str(&repaired).expect("repaired JSON must parse");
        assert_eq!(v["content"], "line one\nline two");
    }

    #[test]
    fn repair_json_doubles_a_stray_backslash() {
        // A Windows path streamed without escaping its own backslashes. `\W` and `\S` aren't
        // recognized JSON escape leads (unlike, say, `\n`, which — ambiguously, but per the JSON
        // grammar itself, not a bug in the repair — a real backslash-then-`n` can't be told apart
        // from an intended newline escape), so both round-trip as literal backslashes.
        let raw = r#"{"path":"C:\Windows\System32"}"#;
        assert!(
            serde_json::from_str::<Value>(raw).is_err(),
            "must actually be invalid first"
        );
        let repaired = repair_json(raw);
        let v: Value = serde_json::from_str(&repaired).expect("repaired JSON must parse");
        assert_eq!(v["path"], r"C:\Windows\System32");
    }

    #[test]
    fn repair_json_leaves_already_valid_json_semantically_unchanged() {
        let raw = r#"{"a":1,"b":"already \"valid\" \n json","c":[1,2,3]}"#;
        let original: Value = serde_json::from_str(raw).unwrap();
        let repaired: Value = serde_json::from_str(&repair_json(raw)).unwrap();
        assert_eq!(original, repaired);
    }

    #[test]
    fn repair_json_does_not_touch_structural_characters_outside_strings() {
        // Whitespace/structure between key-value pairs must survive untouched — only bytes *inside* a
        // string literal are ever rewritten.
        let raw = "{\n  \"a\": 1,\n  \"b\": 2\n}";
        assert_eq!(repair_json(raw), raw);
    }

    #[test]
    fn close_incomplete_json_recovers_a_value_string_truncated_mid_stream() {
        // The concrete motivating case: a long write/edit argument value cut off by an output-token
        // ceiling, not a transport error — no ContentBlockStop/ToolUseStart malformed shape, just the
        // buffer ending mid-string.
        let truncated = r#"{"path":"foo.txt","content":"hello wor"#;
        let closed = close_incomplete_json(truncated).expect("must find something to close");
        let v: Value = serde_json::from_str(&closed).expect("result must be valid JSON");
        assert_eq!(v["path"], "foo.txt");
        assert_eq!(v["content"], "hello wor");
    }

    #[test]
    fn close_incomplete_json_closes_nested_containers_in_the_correct_order() {
        let truncated = r#"{"edits":[{"old_string":"a","new_string":"b"},{"old_string":"c"#;
        let closed = close_incomplete_json(truncated).expect("must find something to close");
        let v: Value = serde_json::from_str(&closed).expect("result must be valid JSON");
        assert_eq!(v["edits"][0]["new_string"], "b");
        assert_eq!(v["edits"][1]["old_string"], "c");
    }

    #[test]
    fn close_incomplete_json_trims_a_dangling_trailing_comma() {
        // Cut off right after a completed element, before the next one started — the trailing comma
        // alone (even once containers close) is otherwise still invalid JSON.
        let truncated = r#"{"a":1,"#;
        let closed = close_incomplete_json(truncated).expect("must find something to close");
        let v: Value = serde_json::from_str(&closed).expect("result must be valid JSON");
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn close_incomplete_json_returns_none_for_already_balanced_input() {
        // Not this function's failure mode — nothing structural is open, so there's nothing to close.
        assert_eq!(close_incomplete_json(r#"{"a":1}"#), None);
    }

    #[test]
    fn close_incomplete_json_leaves_a_key_truncated_before_its_value_unparseable() {
        // Deliberately out of scope (see the function's own doc comment): a key cut off before its
        // `:value` ever started can't be recovered without guessing at a value — the result must
        // still fail to parse, matching pre-existing malformed-call behavior, not silently invent one.
        let truncated = r#"{"text":"#;
        let closed = close_incomplete_json(truncated).unwrap();
        assert!(
            serde_json::from_str::<Value>(&closed).is_err(),
            "must not fabricate a value for a truncated key: {closed:?}"
        );
    }

    #[tokio::test]
    async fn a_raw_control_character_in_streamed_tool_args_is_repaired_not_malformed() {
        // Same shape as `malformed_tool_args_become_recoverable_error_result`, but for the class of
        // failure `repair_json` exists to fix: a raw newline byte inside the streamed JSON string
        // (as if a provider's SSE encoder forgot to escape it) rather than genuinely truncated JSON.
        // This must now parse successfully on the repair pass instead of becoming a malformed call.
        let turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "tu_1".into(),
                name: "echo".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: "{\"text\":\"line one".into(),
            },
            StreamEvent::InputJsonDelta {
                index: 0,
                partial_json: "\nline two\"}".into(), // raw newline, not an escaped `\n`
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, _mock) = agent_with(vec![turn, turn::text("done")], tools);
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        match &session.messages[1].content[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert_eq!(input["text"], "line one\nline two");
            }
            other => panic!("expected a tool_use block with repaired input, got {other:?}"),
        }
        match &session.messages[2].content[0] {
            ContentBlock::ToolResult { is_error, .. } => {
                assert!(
                    !is_error,
                    "a repaired call must not be reported as malformed"
                );
            }
            other => panic!("expected a successful tool_result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn max_steps_is_enforced() {
        // The model keeps asking for a tool forever.
        let turns = vec![turn::tool_call("t", "echo", r#"{"text":"x"}"#); 10];
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(turns));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(3);
        let mut session = Session::new();
        session.user("loop");
        let err = agent.run(&mut session, |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::MaxSteps(3)));

        // Resumable: the check runs before any per-turn state is touched, so a fresh `run` against the
        // *same* session (no rollback needed) can simply continue past the ceiling with a fresh budget
        // rather than the session being left in some unusable state. This fixture's model keeps
        // requesting tools forever, so the second run hits its own fresh 3-step ceiling too — the
        // point is that it *runs* 3 more steps rather than erroring immediately or corrupting state.
        let steps_before = session.steps;
        let err = agent.run(&mut session, |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::MaxSteps(3)));
        assert_eq!(session.steps, steps_before + 3);
    }

    #[tokio::test]
    async fn streams_events_to_callback() {
        let (agent, _mock) = agent_with(vec![turn::text("stream me")], ToolRegistry::new());
        let mut session = Session::new();
        session.user("hi");
        let mut seen = Vec::new();
        agent
            .run(&mut session, |ev| seen.push(ev.clone()))
            .await
            .unwrap();
        assert!(
            seen.iter()
                .any(|e| matches!(e, StreamEvent::TextDelta { text, .. } if text == "stream me"))
        );
    }

    #[tokio::test]
    async fn auto_compaction_fires_over_the_threshold() {
        // A tiny context window forces compaction. Each model turn reports a large input size, so the
        // threshold trips; once enough turns have accumulated for a clean cut, the loop runs a
        // summarization turn before the next request and rewrites the transcript.
        //
        // A tool-call turn that also reports a large input usage (so the threshold trips).
        fn big_tool_turn(id: &str) -> Vec<StreamEvent> {
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: id.into(),
                    name: "echo".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: r#"{"text":"a"}"#.into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 95,
                    output_tokens: 5,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        // call 1: tool turn (3 msgs — too short to cut). call 2: tool turn (5 msgs — cuttable next).
        // call 3: the summarization call. call 4: the real next turn, which ends.
        let mock = Arc::new(MockTransport::new(vec![
            big_tool_turn("t1"),
            big_tool_turn("t2"),
            turn::text("## Goal\nsummary of earlier work"),
            turn::text("all done"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(12)
            .with_compaction(CompactionConfig {
                context_window: 100,
                reserve_tokens: 10,
                keep_recent_tokens: 1,
                summary_max_tokens: 256,
                enabled: true,
            });
        let mut session = Session::new();
        session.user("seed task with enough text to fill several estimated tokens here please");

        let mut compacted = false;
        agent
            .run_events(&mut session, |ev| {
                if matches!(ev, AgentEvent::Compacted { .. }) {
                    compacted = true;
                }
            })
            .await
            .unwrap();

        assert!(
            compacted,
            "compaction should have fired under the tiny window"
        );
        // 4 model calls: two tool turns, the summarization call, the final turn.
        assert_eq!(mock.calls(), 4);
        // The transcript was rewritten: the first message is the summary user turn.
        assert!(matches!(
            &session.messages[0].content[0],
            ContentBlock::Text { text, .. } if text.contains("compacted")
        ));
    }

    #[test]
    fn compact_outcome_compacted_and_reason_agree() {
        assert!(CompactOutcome::Compacted.compacted());
        assert_eq!(CompactOutcome::Compacted.reason(), None);

        assert!(!CompactOutcome::TooSmall.compacted());
        assert_eq!(CompactOutcome::TooSmall.reason(), Some("too_small"));

        assert!(!CompactOutcome::AlreadyCompacted.compacted());
        assert_eq!(
            CompactOutcome::AlreadyCompacted.reason(),
            Some("already_compacted")
        );
    }

    #[tokio::test]
    async fn compaction_start_fires_with_the_right_reason_before_compacted() {
        // LOW pi-parity gap (fixed): `serve.rs` surfaces pi's `isCompacting` via `get_state`, tracked
        // by setting a flag on `CompactionStart` and clearing it on `Compacted` — this proves the new
        // event actually fires, with the correct `reason`, strictly before `Compacted` (not after, and
        // not instead of it), for both event types this run's own dispatcher can produce.
        fn big_tool_turn(id: &str) -> Vec<StreamEvent> {
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: id.into(),
                    name: "echo".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: r#"{"text":"a"}"#.into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 95,
                    output_tokens: 5,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            big_tool_turn("t1"),
            big_tool_turn("t2"),
            turn::text("## Goal\nsummary of earlier work"),
            turn::text("all done"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(12)
            .with_compaction(CompactionConfig {
                context_window: 100,
                reserve_tokens: 10,
                keep_recent_tokens: 1,
                summary_max_tokens: 256,
                enabled: true,
            });
        let mut session = Session::new();
        session.user("seed task with enough text to fill several estimated tokens here please");

        let mut kinds = Vec::new();
        agent
            .run_events(&mut session, |ev| {
                if matches!(
                    ev,
                    AgentEvent::CompactionStart { .. } | AgentEvent::Compacted { .. }
                ) {
                    kinds.push(ev);
                }
            })
            .await
            .unwrap();

        assert_eq!(kinds.len(), 2, "got: {kinds:#?}");
        assert!(
            matches!(
                &kinds[0],
                AgentEvent::CompactionStart {
                    reason: CompactionReason::Threshold
                }
            ),
            "got: {:#?}",
            kinds[0]
        );
        assert!(
            matches!(&kinds[1], AgentEvent::Compacted { .. }),
            "CompactionStart must precede Compacted, not follow it: {:#?}",
            kinds[1]
        );
    }

    #[tokio::test]
    async fn a_failed_automatic_compaction_is_reported_not_propagated() {
        // pi-parity gap (fixed): a transient failure in the summarization call used to unwind the
        // whole run via `?` — and since `should_compact` re-fires on every subsequent turn until it
        // succeeds, a persistently failing summarizer made the session permanently unusable, blocking
        // the user's own prompt from ever reaching the model. It must instead report `CompactionFailed`
        // and let the turn that was about to be sent proceed unsummarized, matching pi's
        // `_runAutoCompaction`'s `try/catch` + `compaction_end { errorMessage }`.
        fn big_tool_turn(id: &str) -> Vec<StreamEvent> {
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: id.into(),
                    name: "echo".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: r#"{"text":"a"}"#.into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 95,
                    output_tokens: 5,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::scripted(vec![
            // Turns 1 and 2 both cross the usage threshold, but the first compaction attempt is a
            // silent `find_split_cut` no-op (not enough history yet for a worthwhile cut) — mirrors
            // `compaction_start_fires_with_the_right_reason_before_compacted` above, which needs the
            // same two turns before a real summarization call happens.
            big_tool_turn("t1").into_iter().map(Ok).collect(),
            big_tool_turn("t2").into_iter().map(Ok).collect(),
            // The compaction round's own summarization call — fails outright, not with a
            // mid-stream-retryable shape (this is a hard failure, not a transient blip to retry away).
            vec![Err(Error::Transport(
                "mock summarizer permanently unavailable".into(),
            ))],
            // Turn 3 still gets sent, unsummarized, and completes the run normally.
            turn::text("done despite the failed compaction")
                .into_iter()
                .map(Ok)
                .collect(),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(6)
            .with_compaction(CompactionConfig {
                context_window: 100,
                reserve_tokens: 10,
                keep_recent_tokens: 1,
                summary_max_tokens: 256,
                enabled: true,
            });
        let mut session = Session::new();
        session.user("seed task with enough text to fill several estimated tokens here please");

        let mut kinds = Vec::new();
        let result = agent
            .run_events(&mut session, |ev| {
                if matches!(
                    ev,
                    AgentEvent::CompactionStart { .. }
                        | AgentEvent::Compacted { .. }
                        | AgentEvent::CompactionFailed { .. }
                        | AgentEvent::Error { .. }
                ) {
                    kinds.push(ev);
                }
            })
            .await;

        assert!(
            result.is_ok(),
            "a failed automatic compaction must not abort the run: {result:?}"
        );
        assert_eq!(
            kinds.len(),
            2,
            "expected exactly CompactionStart then CompactionFailed, got: {kinds:#?}"
        );
        assert!(
            matches!(
                &kinds[0],
                AgentEvent::CompactionStart {
                    reason: CompactionReason::Threshold
                }
            ),
            "got: {:#?}",
            kinds[0]
        );
        match &kinds[1] {
            AgentEvent::CompactionFailed { reason, message } => {
                assert_eq!(*reason, CompactionReason::Threshold);
                assert!(
                    message.contains("mock summarizer permanently unavailable"),
                    "got: {message}"
                );
            }
            other => panic!("expected CompactionFailed, got: {other:#?}"),
        }
        assert_eq!(
            mock.calls(),
            4,
            "the failed compaction attempt must not have blocked the real next turn from still being sent"
        );
        assert!(
            matches!(
                session.messages.last().map(|m| m.content.first()),
                Some(Some(ContentBlock::Text { text, .. })) if text == "done despite the failed compaction"
            ),
            "the run must reach the real turn, not just swallow the failure and stop: {:#?}",
            session.messages.last()
        );
    }

    #[tokio::test]
    async fn hard_overflow_forces_compaction_even_when_auto_compaction_is_disabled() {
        // Auto-compaction is off, so the normal threshold check (`should_compact`, gated on `enabled`)
        // never fires. But the tool-call turn's own reported usage already meets the raw context
        // window — a silent overflow no error was raised for. `is_hard_overflow` must still force a
        // compaction before the next request goes out, regardless of the disabled toggle.
        // `find_cut` declines short conversations (needs at least 4 messages), so two tool turns are
        // scripted — only the second one's usage actually breaches the raw window.
        fn tool_turn(id: &str, input_tokens: u32) -> Vec<StreamEvent> {
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: id.into(),
                    name: "echo".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: r#"{"text":"a"}"#.into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens,
                    output_tokens: 5,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            tool_turn("t1", 20),
            tool_turn("t2", 150),
            turn::text("## Goal\nsummary of earlier work"),
            turn::text("all done"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_compaction(CompactionConfig {
                context_window: 100,
                reserve_tokens: 10,
                keep_recent_tokens: 1,
                summary_max_tokens: 256,
                enabled: false,
            });
        let mut session = Session::new();
        session.user("seed task");

        let mut reason_seen = None;
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::Compacted { reason, .. } = ev {
                    reason_seen = Some(reason);
                }
            })
            .await
            .unwrap();

        assert_eq!(
            reason_seen,
            Some(CompactionReason::Overflow),
            "a hard overflow must compact with reason Overflow even though auto-compaction is disabled"
        );
        // 4 model calls: the two tool turns, the forced summarization call, the final turn.
        assert_eq!(mock.calls(), 4);
    }

    #[tokio::test]
    async fn hard_overflow_survives_a_persisted_stop_reason_from_a_prior_run() {
        // pi-parity fix: `last_stop_reason` used to be a local variable hardcoded to
        // `StopReason::EndTurn` at the top of every fresh `run_events`/`run_events_steered` call, so
        // `is_hard_overflow`'s `MaxTokens` branch could only ever fire within the *same* in-flight run
        // that produced the `MaxTokens` turn — a `MaxTokens` stop persisted by a prior run (or process,
        // after a session reload) was invisible to a brand new call. Matches pi's own `prompt()`
        // (`_findLastAssistantMessage` + `_checkCompaction`, re-derived fresh on every prompt from the
        // persisted `AssistantMessage.stopReason`) and its `pre-prompt-compaction-no-continue.test.ts`
        // regression: a `MaxTokens`-stopped assistant message appended directly — not produced by a
        // live run — must still be detected and compacted by the very next fresh call.
        let mut session = Session::new();
        session.user("original task");
        session.push(
            Message::assistant(vec![ContentBlock::text("first reply")])
                .with_model_id("claude-opus-4-8")
                .with_usage(TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                })
                .with_stop_reason(StopReason::EndTurn),
        );
        session.user("second question");
        // The persisted turn from "a prior run/process" — constructed directly, matching pi's own
        // regression test, rather than produced by driving a live turn through the loop.
        session.push(
            Message::assistant(vec![ContentBlock::text(
                "this got cut off in a prior run/process",
            )])
            .with_model_id("claude-opus-4-8")
            .with_usage(TokenUsage {
                input_tokens: 50,
                output_tokens: 100,
                ..Default::default()
            })
            .with_stop_reason(StopReason::MaxTokens),
        );
        // Mirrors what a real prior run's `record_usage` would have left behind — the state a
        // session-load/deserialize restores, not something this call's own (not-yet-started) turn
        // produced.
        session.last_input_tokens = 50;
        session.last_usage_message_count = session.messages.len();
        session.last_output_tokens = 100;
        session.user("next prompt");

        let mock = Arc::new(MockTransport::new(vec![
            turn::text("## Goal\nsummary of earlier work"),
            turn::text("answered next prompt"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
            context_window: 100,
            reserve_tokens: 10,
            keep_recent_tokens: 1,
            summary_max_tokens: 256,
            enabled: false,
        });

        let mut reason_seen = None;
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::Compacted { reason, .. } = ev {
                    reason_seen = Some(reason);
                }
            })
            .await
            .unwrap();

        assert_eq!(
            reason_seen,
            Some(CompactionReason::Overflow),
            "a MaxTokens stop_reason persisted from a prior run must still force a hard-overflow \
             compaction on the very next fresh call, not just within the run that produced it"
        );
        assert_eq!(
            mock.calls(),
            2,
            "the forced summarization call, then the real next turn"
        );
    }

    #[tokio::test]
    async fn a_silently_truncated_turn_compacts_and_retries_instead_of_returning_a_cut_off_answer()
    {
        // pi-parity gap: a turn that completes successfully (no transport error) but gets cut off by
        // `max_tokens` alone, with no tool calls, used to fall straight through to "the model ended its
        // turn" — handing the user a hard-truncated non-answer with no recovery, even with
        // auto-compaction enabled. `find_cut` declines short conversations (needs at least 4 messages),
        // so a throwaway first tool round-trip pads the history before the real, truncated turn.
        fn max_tokens_turn(input_tokens: u32) -> Vec<StreamEvent> {
            vec![
                StreamEvent::MessageStart,
                StreamEvent::TextDelta {
                    index: 0,
                    text: "this got cut off partway through and must not survive the retry".into(),
                },
                StreamEvent::Usage(TokenUsage {
                    input_tokens,
                    output_tokens: 100,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::MaxTokens,
                },
            ]
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t1", "echo", r#"{"text":"pad"}"#),
            max_tokens_turn(150),
            turn::text("## Goal\nsummary of earlier work"),
            turn::text("here is the real, complete answer"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_compaction(CompactionConfig {
                context_window: 100,
                reserve_tokens: 10,
                keep_recent_tokens: 1,
                summary_max_tokens: 256,
                enabled: true,
            });
        let mut session = Session::new();
        session.user("seed task");

        let mut reason_seen = None;
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::Compacted { reason, .. } = ev {
                    reason_seen = Some(reason);
                }
            })
            .await
            .unwrap();

        assert_eq!(
            reason_seen,
            Some(CompactionReason::Overflow),
            "the silent truncation must trigger a compaction with reason Overflow"
        );
        // 4 model calls: the padding tool turn, the truncated attempt, the forced summarization call,
        // and the real retried turn.
        assert_eq!(mock.calls(), 4);
        let dump = format!("{:?}", session.messages);
        assert!(
            !dump.contains("cut off partway through"),
            "the discarded truncated response must not survive in the session: {dump}"
        );
        assert!(
            dump.contains("real, complete answer"),
            "the retried turn's real answer must be what's left: {dump}"
        );
    }

    #[tokio::test]
    async fn auto_compaction_records_provenance_on_the_session() {
        // A registered tool named "read" (not "echo") so `extract_file_ops` — keyed on tool name —
        // actually picks up a file reference from the compacted turns.
        struct ReadLikeTool;
        #[async_trait]
        impl Tool for ReadLikeTool {
            fn name(&self) -> &str {
                "read"
            }
            fn description(&self) -> &str {
                "d"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok("contents".into())
            }
        }

        fn big_read_turn(id: &str) -> Vec<StreamEvent> {
            vec![
                StreamEvent::MessageStart,
                StreamEvent::ToolUseStart {
                    index: 0,
                    id: id.into(),
                    name: "read".into(),
                },
                StreamEvent::InputJsonDelta {
                    index: 0,
                    partial_json: r#"{"path":"tracked.rs"}"#.into(),
                },
                StreamEvent::ContentBlockStop { index: 0 },
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 95,
                    output_tokens: 5,
                    ..Default::default()
                }),
                StreamEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                },
            ]
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(ReadLikeTool));
        let mock = Arc::new(MockTransport::new(vec![
            big_read_turn("t1"),
            big_read_turn("t2"),
            turn::text("## Goal\nsummary of earlier work"),
            turn::text("all done"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(12)
            .with_compaction(CompactionConfig {
                context_window: 100,
                reserve_tokens: 10,
                keep_recent_tokens: 1,
                summary_max_tokens: 256,
                enabled: true,
            });
        let mut session = Session::new();
        session.user("seed task with enough text to fill several estimated tokens here please");

        let mut reason_seen = None;
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::Compacted { reason, .. } = ev {
                    reason_seen = Some(reason);
                }
            })
            .await
            .unwrap();

        assert_eq!(reason_seen, Some(CompactionReason::Threshold));
        assert_eq!(session.compaction.compactions, 1);
        assert_eq!(
            session.compaction.last_reason,
            Some(CompactionReason::Threshold)
        );
        assert_eq!(session.compaction.read_files, vec!["tracked.rs"]);
    }

    #[tokio::test]
    async fn compact_issues_two_calls_and_merges_with_turn_context_header() {
        // A split-turn compaction round (cut lands mid-tool-dispatch) must issue exactly two
        // summarization calls — one for the closed-off history, one for the in-progress turn's own
        // prefix — and splice their merged result in with the "Turn Context" separator, rather than
        // collapsing everything under the minimal split-turn template.
        let session_messages = vec![
            Message::user("first request"),
            Message::assistant(vec![ContentBlock::text("first done")]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("2", "edited", false),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::new(vec![
            turn::text("history summary text"),
            turn::text("turn prefix summary text"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        assert!(
            compacted.compacted(),
            "a split-turn compaction round should still apply"
        );
        assert_eq!(mock.calls(), 2, "a split turn must issue two summary calls");
        assert!(
            matches!(&session.messages[0].content[0], ContentBlock::Text { text, .. }
                if text.contains("history summary text")
                    && text.contains("**Turn Context (split turn):**")
                    && text.contains("turn prefix summary text")),
            "expected the merged summary, got: {:?}",
            session.messages[0].content
        );
    }

    #[tokio::test]
    async fn compact_appends_file_operations_to_the_applied_summary_text() {
        // pi-parity fix: `<read-files>`/`<modified-files>` were fed into the summarization *prompt*
        // but never appended to its *output* — the file list only reached the live conversation if the
        // summarizing model happened to mention exact paths in its own prose. `session.compaction`
        // already tracked the files structurally (a separate mechanism, unaffected by this bug); this
        // is about the actual text a later turn's request would include.
        let session_messages = vec![
            Message::user("look at this"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "src/lib.rs" }),
            )]),
            Message::tool_result("1", "contents", false),
            // A real conversation always has the assistant reply to its own tool result before the
            // next user turn arrives — roles strictly alternate.
            Message::assistant(vec![ContentBlock::text("okay, done reading")]),
            Message::user("ok now something else"),
            Message::assistant(vec![ContentBlock::text("done")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::new(vec![turn::text(
            "A prose summary that never mentions any file paths.",
        )]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();
        assert!(compacted.compacted());

        let ContentBlock::Text { text, .. } = &session.messages[0].content[0] else {
            panic!("expected the spliced summary message to be text");
        };
        assert!(text.contains("<read-files>"), "got: {text}");
        assert!(text.contains("src/lib.rs"), "got: {text}");
    }

    /// A conversation that plans with `todo`, then does unrelated work — the shape whose plan a
    /// compaction would otherwise fold away.
    fn planning_session() -> Vec<Message> {
        vec![
            Message::user("plan and do the work"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "todo",
                json!({ "todos": [
                    { "content": "Wire the retry loop", "activeForm": "Wiring the retry loop", "status": "in_progress" },
                    { "content": "Add tests", "activeForm": "Adding tests", "status": "pending" },
                ]}),
            )]),
            Message::tool_result("1", "Todos updated (2 items):", false),
            Message::assistant(vec![ContentBlock::text("planned")]),
            Message::user("ok keep going"),
            Message::assistant(vec![ContentBlock::text("done")]),
        ]
    }

    fn summary_text(session: &Session) -> &str {
        match &session.messages[0].content[0] {
            ContentBlock::Text { text, .. } => text,
            _ => panic!("expected the spliced summary message to be text"),
        }
    }

    fn aggressive_compaction() -> CompactionConfig {
        CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        }
    }

    #[tokio::test]
    async fn compact_appends_the_carried_todo_list_to_the_applied_summary_text() {
        // Same deterministic-carry contract `<read-files>` has, and for the same reason: the
        // summarizing model here never mentions the plan in its prose, so if the block weren't appended
        // by host code the model would wake up on the far side of the cut with no plan at all.
        let mut session = Session::new();
        session.messages = Arc::new(planning_session());

        let mock = Arc::new(MockTransport::new(vec![turn::text(
            "A prose summary that never mentions the plan.",
        )]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_compaction(aggressive_compaction());
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &CancellationToken::new(),
                &mut |_| {},
                None,
            )
            .await
            .unwrap();
        assert!(compacted.compacted());

        let text = summary_text(&session);
        assert!(text.contains("<todo_list>"), "got: {text}");
        assert!(text.contains("[>] Wire the retry loop"), "got: {text}");
        assert!(text.contains("[ ] Add tests"), "got: {text}");
        // And structurally, for a host that wants the list without parsing text (`serve`'s `get_todos`).
        assert_eq!(
            session
                .compaction
                .todos
                .as_ref()
                .map(|t| t[0]["content"].clone()),
            Some(json!("Wire the retry loop"))
        );
    }

    #[tokio::test]
    async fn a_todo_list_survives_a_second_compaction_through_the_real_compact_path() {
        // The whole point of routing the plan through `CompactionProvenance`. By round 2 the `todo`
        // `tool_use` block no longer exists anywhere in `session.messages` — `apply_summary` physically
        // dropped it — so the only surviving copy is the one folded forward on the session, and the only
        // way it reaches the model is `compact` re-appending it. Neither round's summarizing model
        // mentions the plan, exactly as a real one usually wouldn't.
        let mut session = Session::new();
        session.messages = Arc::new(planning_session());

        let mock = Arc::new(MockTransport::new(vec![
            turn::text("Round one prose."),
            turn::text("Round two prose."),
        ]));
        let agent =
            Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(aggressive_compaction());
        let cancel = CancellationToken::new();

        agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();
        assert!(summary_text(&session).contains("[>] Wire the retry loop"));

        // More work happens, none of it touching `todo`. Roles keep alternating: summary(user),
        // assistant, user, assistant.
        let mut next = session.messages.as_ref().clone();
        next.push(Message::user("now do something unrelated"));
        next.push(Message::assistant(vec![ContentBlock::text(
            "unrelated work finished",
        )]));
        session.messages = Arc::new(next);
        assert!(
            compaction::extract_todos(&session.messages).is_none(),
            "the `todo` tool_use block is already gone from history — provenance is the only copy"
        );

        agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        let text = summary_text(&session);
        assert!(
            text.contains("[>] Wire the retry loop"),
            "the plan must survive the second compaction, got: {text}"
        );
        assert_eq!(
            text.matches("<todo_list>").count(),
            1,
            "`previous_summary` must peel round one's block off before it is fed forward and \
             re-appended, or the blocks accumulate unboundedly: {text}"
        );
        assert_eq!(session.compaction.compactions, 2);
        assert_eq!(mock.calls(), 2, "one summarization call per round");
    }

    #[tokio::test]
    async fn compact_does_not_stack_carry_blocks_when_it_reuses_a_prior_summary_verbatim() {
        // The `turn_start == 1` fast path reuses the prior summary's body as the new summary's history
        // half, and `compact` then appends the freshly-merged carry blocks to the result. Without
        // `previous_summary` stripping them first, every split-turn round would append another copy of
        // every block — an unbounded leak of the exact text this channel exists to keep exact.
        let prior = format!(
            "{}\n\nprior summary body{}{}",
            compaction::SUMMARY_MARKER,
            compaction::format_file_operations(&["a.rs".into()], &[]),
            compaction::format_todo_list(Some(
                &json!([{ "content": "Wire the retry loop", "status": "in_progress" }])
            ))
        );
        let mut session = Session::new();
        session.messages = Arc::new(vec![
            Message::user(prior),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("1", "contents of a.rs", false),
        ]);
        // Seed the provenance the way a real prior round would have left it.
        session.compaction = compaction::CompactionProvenance {
            read_files: vec!["a.rs".into()],
            todos: Some(json!([{ "content": "Wire the retry loop", "status": "in_progress" }])),
            compactions: 1,
            last_reason: Some(CompactionReason::Threshold),
            modified_files: vec![],
            memory_notes: vec![],
        };

        let mock = Arc::new(MockTransport::new(vec![turn::text("turn prefix summary")]));
        let agent =
            Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(aggressive_compaction());
        agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &CancellationToken::new(),
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        let text = summary_text(&session);
        assert!(text.contains("prior summary body"), "got: {text}");
        assert_eq!(text.matches("<todo_list>").count(), 1, "got: {text}");
        assert_eq!(text.matches("<read-files>").count(), 1, "got: {text}");
        assert!(text.contains("[>] Wire the retry loop"), "got: {text}");
    }

    #[tokio::test]
    async fn compacted_event_carries_the_summary_text_and_a_post_compaction_token_estimate() {
        // Pi-parity fix: `AgentEvent::Compacted` used to carry only `messages_before`/`messages_after`/
        // `reason`/`tokens_before` — a caller (the `compact` RPC's response, a `run --json` client) had
        // no way to see the summary text that was actually spliced in, nor any estimate of how much the
        // compaction actually shrank the prompt by.
        let session_messages = vec![
            Message::user("look at this"),
            Message::assistant(vec![ContentBlock::text("ok, looking")]),
            Message::user("now something else"),
            Message::assistant(vec![ContentBlock::text("done")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);
        session.last_input_tokens = 12345;

        let mock = Arc::new(MockTransport::new(vec![turn::text(
            "SUMMARY-TEXT-MARKER-771",
        )]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();
        let mut event_summary = None;
        let mut event_tokens_before = None;
        let mut event_tokens_after = None;
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |ev| {
                    if let AgentEvent::Compacted {
                        summary,
                        tokens_before,
                        tokens_after,
                        ..
                    } = ev
                    {
                        event_summary = Some(summary);
                        event_tokens_before = Some(tokens_before);
                        event_tokens_after = Some(tokens_after);
                    }
                },
                None,
            )
            .await
            .unwrap();
        assert!(compacted.compacted());

        let summary = event_summary.expect("Compacted event must carry a summary");
        assert!(
            summary.contains("SUMMARY-TEXT-MARKER-771"),
            "the event's summary must be the actual generated text: {summary}"
        );
        // Matches what was actually spliced into the session (modulo `apply_summary`'s own
        // `SUMMARY_MARKER` prefix and `tokens_before` line, which wrap the summary text but aren't part
        // of it).
        let ContentBlock::Text { text: spliced, .. } = &session.messages[0].content[0] else {
            panic!("expected the spliced summary message to be text");
        };
        assert_eq!(
            spliced,
            &format!(
                "{}\n\nCompacted from 12345 tokens\n\n{summary}",
                compaction::SUMMARY_MARKER
            )
        );

        assert_eq!(
            event_tokens_before,
            Some(12345),
            "tokens_before must reflect the session's own pre-compaction usage snapshot"
        );
        let tokens_after = event_tokens_after.expect("Compacted event must carry tokens_after");
        assert!(
            tokens_after > 0,
            "a non-empty post-compaction session must estimate a positive token count, got {tokens_after}"
        );
        // A real, whole-list estimate, not `trailing_tokens`'s since-last-snapshot delta — which
        // `apply_summary` resets to point past the rebuilt list's end, so it would report 0 here.
        assert_eq!(
            compaction::trailing_tokens(&session),
            0,
            "sanity: trailing_tokens is 0 immediately after apply_summary, confirming tokens_after \
             can't have come from it"
        );
    }

    #[tokio::test]
    async fn summarize_branch_appends_file_operations_to_the_returned_text() {
        // Same fix, the branch-summarization path.
        let branch = vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "edit",
                json!({ "path": "src/x.rs" }),
            )]),
            Message::tool_result("1", "edited", false),
            Message::assistant(vec![ContentBlock::text("didn't pan out")]),
        ];
        let mock = Arc::new(MockTransport::new(vec![turn::text(
            "A prose summary that never mentions any file paths.",
        )]));
        let agent = Agent::new(mock, "claude-opus-4-8");
        let cancel = CancellationToken::new();
        let summary = agent
            .summarize_branch(&branch, &cancel, None, false)
            .await
            .unwrap();
        assert!(summary.contains("<modified-files>"), "got: {summary}");
        assert!(summary.contains("src/x.rs"), "got: {summary}");
    }

    #[tokio::test]
    async fn summarize_branch_threads_replace_instructions_into_the_summarization_request() {
        // Task #17 (pi-parity), agent-core's portion: `branch_summary_request` already implements
        // `replace_instructions` fully (see `branch_summary.rs`'s own tests) — this pins that
        // `Agent::summarize_branch` actually has a way to request it and forwards it through, instead
        // of the previous hardcoded `false` with no caller-facing parameter at all.
        let branch = vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::text("didn't pan out")]),
        ];
        let mock = Arc::new(MockTransport::new(vec![turn::text("a recap")]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8");
        let cancel = CancellationToken::new();
        agent
            .summarize_branch(
                &branch,
                &cancel,
                Some("Summarize only the auth-related changes, in one sentence."),
                true,
            )
            .await
            .unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 1);
        let ContentBlock::Text { text, .. } = &requests[0].messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("Summarize only the auth-related changes, in one sentence."),
            "the custom instructions must be present: {text}"
        );
        assert!(
            !text.contains("conversation branch for context when returning later"),
            "replace_instructions: true must replace the base instruction, not append alongside it: {text}"
        );
    }

    #[tokio::test]
    async fn summarize_branch_threads_custom_instructions_into_the_summarization_request() {
        // B-M8 pi-parity gap (fixed): unlike `Agent::compact`, `summarize_branch` had no
        // `custom_instructions` parameter at all — a client navigating away from a branch had no way
        // to steer what the recap emphasizes, unlike a manual compaction. The custom instruction must
        // actually reach the request the mock transport receives.
        let branch = vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::text("didn't pan out")]),
        ];
        let mock = Arc::new(MockTransport::new(vec![turn::text("a recap")]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8");
        let cancel = CancellationToken::new();
        agent
            .summarize_branch(
                &branch,
                &cancel,
                Some("keep every detail about the auth refactor"),
                false,
            )
            .await
            .unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 1);
        let ContentBlock::Text { text, .. } = &requests[0].messages[0].content[0] else {
            panic!("expected text");
        };
        assert!(
            text.contains("Additional focus: keep every detail about the auth refactor"),
            "got: {text}"
        );
    }

    #[tokio::test]
    async fn summarize_branch_forwards_the_agents_configured_reasoning_to_the_summarization_request()
     {
        // B-M13 pi-parity gap (fixed): a summarization call (branch or compaction) always ran at the
        // model's bare default reasoning level, ignoring whatever the live session was configured to
        // use. `summarize_branch` must forward the same level, matching pi's `generateSummary` (which
        // only omits `reasoning` when thinking is off, and otherwise passes the live level straight
        // through).
        let branch = vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::text("didn't pan out")]),
        ];
        let mock = Arc::new(MockTransport::new(vec![turn::text("a recap")]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_reasoning_effort(ReasoningEffort::High);
        let cancel = CancellationToken::new();
        agent
            .summarize_branch(&branch, &cancel, None, false)
            .await
            .unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].reasoning_effort, Some(ReasoningEffort::High));
    }

    #[tokio::test]
    async fn summarize_branch_reserve_tokens_can_be_overridden_independently_of_compaction() {
        // pi-parity gap: `summarize_branch` always sized its own input budget off
        // `compaction.reserve_tokens` — pi exposes an independent `reserveTokens` setting for branch
        // summarization instead (`branch-summarization.ts:62-63`, default 16384). A host tuning
        // compaction's reserve for the *live* conversation shouldn't be forced to accept the same
        // number for a one-off branch recap too.
        let branch: Vec<Message> = (0..6)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user("hi")
                } else {
                    Message::assistant(vec![ContentBlock::text("ok")])
                }
            })
            .collect();
        let cancel = CancellationToken::new();

        // A tiny context window with the (large) default `reserve_tokens` leaves a budget of 0 —
        // `windowed_by_budget` then keeps only the single most recent message, noting the rest as
        // omitted.
        let mock_default = Arc::new(MockTransport::new(vec![turn::text("recap")]));
        let agent_default =
            Agent::new(mock_default.clone(), "claude-opus-4-8").with_context_window(50);
        agent_default
            .summarize_branch(&branch, &cancel, None, false)
            .await
            .unwrap();
        let requests = mock_default.requests();
        let ContentBlock::Text {
            text: default_text, ..
        } = &requests[0].messages[0].content[0]
        else {
            panic!("expected text");
        };
        assert!(
            default_text.contains("earlier message(s) from this branch omitted"),
            "the default (large) reserve should leave no budget for this tiny context window: \
             {default_text}"
        );

        // Overriding just the branch-summary reserve to 0 (independent of compaction's own,
        // untouched, default) frees up the whole context window as budget — comfortably enough for
        // these few tiny messages that nothing gets windowed out.
        let mock_override = Arc::new(MockTransport::new(vec![turn::text("recap")]));
        let agent_override = Agent::new(mock_override.clone(), "claude-opus-4-8")
            .with_context_window(50)
            .with_branch_summary_reserve_tokens(0);
        agent_override
            .summarize_branch(&branch, &cancel, None, false)
            .await
            .unwrap();
        let requests = mock_override.requests();
        let ContentBlock::Text {
            text: override_text,
            ..
        } = &requests[0].messages[0].content[0]
        else {
            panic!("expected text");
        };
        assert!(
            !override_text.contains("earlier message(s) from this branch omitted"),
            "an overridden, smaller reserve must free up more budget, not less: {override_text}"
        );
    }

    #[tokio::test]
    async fn summarize_branch_reports_cancellation_as_an_error_not_an_empty_success() {
        // Real interaction bug found while integrating concurrent work: `run_turn`'s mid-stream cancel
        // path was changed (`StopReason::Aborted`) to return `Ok(Turn{blocks:[..], ..})` instead of
        // `Err(Error::Cancelled)`, specifically so `run_events_steered` can persist partial content —
        // see that change's own doc comment. But `summarize_branch` (and `compact`) call `run_turn`
        // directly for a one-off utility call with no such persistence story; naively inheriting that
        // same `Ok(Aborted)` return would make a cancelled call indistinguishable from "the model
        // summarized to an empty string" — the caller (`serve.rs`'s `switch_branch`) falls through to
        // "nothing worth recording" and reports a normal success, exactly the bug this pins: `abort`
        // during branch summarization must still surface as `Err(Error::Cancelled)` so the caller's own
        // cancellation-handling arm (which leaves the session completely untouched, matching pi's
        // `abortBranchSummary`) actually runs. See `Agent::run_utility_turn`, which every summarization
        // call site now goes through instead of calling `run_turn` directly.
        struct StalledTransport;
        #[async_trait]
        impl ModelTransport for StalledTransport {
            async fn stream(&self, _req: ModelRequest) -> Result<crate::transport::EventStream> {
                let s = futures::stream::once(async { Ok(StreamEvent::MessageStart) })
                    .chain(futures::stream::pending());
                Ok(Box::pin(s))
            }
        }
        let branch = vec![
            Message::user("try approach X"),
            Message::assistant(vec![ContentBlock::text("didn't pan out")]),
        ];
        let agent = Agent::new(Arc::new(StalledTransport), "claude-opus-4-8");
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            agent.summarize_branch(&branch, &cancel, None, false),
        )
        .await
        .expect("cancellation must interrupt the stalled summarization call");
        assert!(
            matches!(result, Err(Error::Cancelled)),
            "expected Err(Cancelled), got {result:?}"
        );
    }

    #[tokio::test]
    async fn compact_runs_the_two_split_turn_summary_calls_sequentially_not_concurrently() {
        // Regression for the exact bug pi fixed 13 commits after the last audit ("serialize split-turn
        // compaction summaries... so single-concurrency local providers do not fail with 429 errors"):
        // the history call and the turn-prefix call must never be in flight at the same time. A
        // transport that tracks how many calls are simultaneously mid-`stream()` (sleeping, so a
        // concurrently-dispatched second call has a real chance to overlap) catches a regression back
        // to `futures::future::try_join`.
        use std::collections::VecDeque;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ConcurrencyTrackingTransport {
            turns: Mutex<VecDeque<Vec<StreamEvent>>>,
            current: AtomicUsize,
            max_seen: AtomicUsize,
        }
        #[async_trait]
        impl ModelTransport for ConcurrencyTrackingTransport {
            async fn stream(&self, _req: ModelRequest) -> Result<crate::transport::EventStream> {
                let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_seen.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                let turn = self
                    .turns
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("no more scripted turns");
                self.current.fetch_sub(1, Ordering::SeqCst);
                Ok(Box::pin(futures::stream::iter(turn.into_iter().map(Ok))))
            }
        }

        let session_messages = vec![
            Message::user("first request"),
            Message::assistant(vec![ContentBlock::text("first done")]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("2", "edited", false),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let transport = Arc::new(ConcurrencyTrackingTransport {
            turns: Mutex::new(VecDeque::from(vec![
                turn::text("history summary text"),
                turn::text("turn prefix summary text"),
            ])),
            current: AtomicUsize::new(0),
            max_seen: AtomicUsize::new(0),
        });
        let agent =
            Agent::new(transport.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
                keep_recent_tokens: 1,
                ..CompactionConfig::default()
            });
        let cancel = CancellationToken::new();
        agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            transport.max_seen.load(Ordering::SeqCst),
            1,
            "the history and turn-prefix summary calls must never be in flight at the same time"
        );
    }

    #[tokio::test]
    async fn compact_uses_half_budget_for_split_turn_prefix_call() {
        // Of the two calls a split-turn compaction issues, the turn-prefix one gets half of
        // `reserve_tokens` directly (matching pi's `Math.floor(0.5 * reserveTokens)`) — a partial turn
        // needs proportionally less room, and this budget is deliberately independent of
        // `summary_max_tokens` (itself a separate, larger scaling off `reserve_tokens`).
        let session_messages = vec![
            Message::user("first request"),
            Message::assistant(vec![ContentBlock::text("first done")]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("2", "edited", false),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::new(vec![
            turn::text("history summary"),
            turn::text("turn prefix summary"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            summary_max_tokens: 1000,
            reserve_tokens: 1000,
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();
        agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        let reqs = mock.requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(
            reqs[0].max_tokens, 1000,
            "the history call keeps the full summary_max_tokens budget"
        );
        assert_eq!(
            reqs[1].max_tokens, 500,
            "the turn-prefix call gets half of reserve_tokens directly"
        );
    }

    #[tokio::test]
    async fn compact_reuses_a_prior_summary_verbatim_when_the_split_turn_starts_right_after_it() {
        // A prior compaction's summary always sits at `session.messages[0]`. When the entire
        // conversation since then is one continuous, still-open turn (no genuine new user message),
        // `find_split_cut` reports `turn_start == 1` — the summary alone is the closed-off history side
        // of the split. `compact` must reuse that summary text verbatim instead of spending a model
        // call asking for what would be an unchanged restatement of it: only ONE call (the turn-prefix
        // one) should fire, not two.
        let session_messages = vec![
            Message::user(format!(
                "{}\n\nprior summary body",
                compaction::SUMMARY_MARKER
            )),
            Message::assistant(vec![ContentBlock::tool_use(
                "1",
                "read",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::tool_use(
                "2",
                "edit",
                json!({ "path": "a.rs" }),
            )]),
            Message::tool_result("2", "edited", false),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        // Only one response queued: if the (buggy) history call fired too, `MockTransport` would run
        // out of scripted turns and panic/error, failing the test loudly rather than silently passing.
        let mock = Arc::new(MockTransport::new(vec![turn::text(
            "turn prefix summary text",
        )]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        assert!(compacted.compacted());
        assert_eq!(
            mock.calls(),
            1,
            "reusing the prior summary verbatim must skip its model call entirely"
        );
        assert!(
            matches!(&session.messages[0].content[0], ContentBlock::Text { text, .. }
                if text.contains("prior summary body")
                    && text.contains("**Turn Context (split turn):**")
                    && text.contains("turn prefix summary text")),
            "expected the prior summary folded forward verbatim, got: {:?}",
            session.messages[0].content
        );
    }

    #[tokio::test]
    async fn compact_is_a_no_op_on_a_clean_boundary_when_nothing_new_followed_the_prior_summary() {
        // pi-parity gap (fixed): the clean-boundary path had no "nothing new to fold since the last
        // compaction" reuse guard the split-turn path already has (see the test above). A prior
        // compaction's summary always sits at `session.messages[0]`; when every message since it is
        // still a closed, ordinary conversation (no split-turn mid-dispatch shape), `find_split_cut`
        // can land `first_kept == 1` — the cut boundary right after the summary itself, meaning there
        // is no new activity to fold in at all. Calling compact() again here (e.g. a client issuing a
        // second manual `compact` with nothing new typed in between) must be a genuine no-op: zero
        // model calls, the session left completely unchanged — not a wasted call that just restates
        // the same summary with an empty `<new-activity>` section.
        let session_messages = vec![
            Message::user(format!(
                "{}\n\nprior summary body",
                compaction::SUMMARY_MARKER
            )),
            Message::assistant(vec![ContentBlock::text("first reply")]),
            Message::user("second question"),
            Message::assistant(vec![ContentBlock::text("second reply")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages.clone());

        // No turns scripted at all: any model call at all would panic/error the mock, failing the
        // test loudly rather than silently passing.
        let mock = Arc::new(MockTransport::new(vec![]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8"); // default keep_recent_tokens (20k)
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        // Task #26: distinguishes from `CompactOutcome::TooSmall` — a prior summary already covers
        // this cleanly, matching pi's own "Already compacted" error rather than "session too small".
        assert_eq!(compacted, CompactOutcome::AlreadyCompacted);
        assert_eq!(mock.calls(), 0, "must not make any model call at all");
        assert_eq!(
            session.messages.as_ref(),
            &session_messages,
            "the session must be left completely unchanged"
        );
    }

    #[tokio::test]
    async fn compact_is_a_no_op_when_new_content_since_the_prior_summary_still_fits_the_budget() {
        // Broader than the `first_kept == 1` test above (pi-parity gap, second pass): even with
        // several real exchanges since the prior summary — not just the bare-minimum single exchange
        // the sibling test above covers — the guard must still recognize the total is under the real
        // default `keep_recent_tokens` (20k) and skip re-summarizing, rather than only ever catching
        // the single-exchange case. The old narrower guard only checked `first_kept == 1` and would
        // have re-summarized here even though none of the new content is remotely close to the budget.
        let session_messages = vec![
            Message::user(format!(
                "{}\n\nprior summary body",
                compaction::SUMMARY_MARKER
            )),
            Message::assistant(vec![ContentBlock::text("first reply")]),
            Message::user("second question"),
            Message::assistant(vec![ContentBlock::text("second reply")]),
            Message::user("third question"),
            Message::assistant(vec![ContentBlock::text("third reply")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages.clone());

        // No turns scripted: any model call at all fails the test loudly.
        let mock = Arc::new(MockTransport::new(vec![]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8"); // default keep_recent_tokens (20k)
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        // Task #26: same "already compacted" reason as the test above — a real cut point exists
        // (`find_split_cut` succeeds), just nothing new worth a fresh summary yet.
        assert_eq!(compacted, CompactOutcome::AlreadyCompacted);
        assert_eq!(mock.calls(), 0, "must not make any model call at all");
        assert_eq!(
            session.messages.as_ref(),
            &session_messages,
            "the session must be left completely unchanged"
        );
    }

    #[tokio::test]
    async fn compact_reports_too_small_when_theres_no_worthwhile_cut_point_at_all() {
        // Task #26: distinguishes `CompactOutcome::TooSmall` from `CompactOutcome::AlreadyCompacted` —
        // this session has no prior summary and far too few messages for `find_split_cut` to find any
        // cut point at all (`n < 4`), matching pi's own "Nothing to compact (session too small)" rather
        // than "Already compacted".
        let session_messages = vec![
            Message::user("hi"),
            Message::assistant(vec![ContentBlock::text("hello")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages.clone());

        // No turns scripted at all: any model call would panic/error the mock, failing the test loudly.
        let mock = Arc::new(MockTransport::new(vec![]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8");
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        assert_eq!(compacted, CompactOutcome::TooSmall);
        assert!(!compacted.compacted());
        assert_eq!(mock.calls(), 0, "must not make any model call at all");
        assert_eq!(
            session.messages.as_ref(),
            &session_messages,
            "the session must be left completely unchanged"
        );
    }

    #[tokio::test]
    async fn compact_still_fires_on_a_clean_boundary_once_new_content_exceeds_the_budget() {
        // The other half of the fix: the broadened guard must not become a blanket "never
        // re-compact" — once genuinely new, budget-sized content has accumulated since the prior
        // summary, a fresh compaction must still proceed normally.
        let session_messages = vec![
            Message::user(format!(
                "{}\n\nprior summary body",
                compaction::SUMMARY_MARKER
            )),
            // ~100k estimated tokens, well over any tiny budget — the role doesn't matter for token
            // estimation, only that it's the real conversation's first message after the summary
            // (always assistant, per `find_cut`'s own doc comment on the post-compaction shape).
            Message::assistant(vec![ContentBlock::text("a".repeat(400_000))]),
            Message::user("second question"),
            Message::assistant(vec![ContentBlock::text("reply")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::new(vec![turn::text("new summary")]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();

        assert!(
            compacted.compacted(),
            "genuinely new large content must still trigger compaction"
        );
        assert_eq!(mock.calls(), 1);
    }

    #[test]
    fn agent_new_scales_summary_max_tokens_from_model_capabilities() {
        // A model with a `max_output` ceiling below the naive 0.8*reserve_tokens computation must have
        // its summarization budget clamped to that ceiling, not the flat 4096 default nor an
        // unreachably large number the model would reject.
        let mock = Arc::new(MockTransport::new(vec![]));
        // claude-3-haiku-20240307: gen-3 legacy, max_output 4_096 (see `models.rs`) — comfortably
        // below the default reserve_tokens(16_384)*0.8 = 13_107, so the clamp must bite.
        let agent = Agent::new(mock, "claude-3-haiku-20240307");
        assert_eq!(
            agent.compaction.summary_max_tokens, 4_096,
            "summary_max_tokens must be clamped to the model's own max_output"
        );

        // A model with a large max_output should get the full 0.8*reserve_tokens computation, not the
        // old flat 4096 default.
        let mock2 = Arc::new(MockTransport::new(vec![]));
        let agent2 = Agent::new(mock2, "claude-opus-4-8");
        assert_eq!(agent2.compaction.summary_max_tokens, 13_107);
    }

    #[test]
    fn with_compaction_rescales_summary_max_tokens_instead_of_resetting_it_to_the_flat_default() {
        // The common pattern for overriding just one field of the compaction config (`serve.rs`'s own
        // `build_agent` does exactly this): struct-update syntax against `CompactionConfig::default()`.
        // Before this fix, replacing the whole config this way silently discarded `Agent::new()`'s
        // model-aware `summary_max_tokens` and fell back to the struct's flat 4096 default — a real
        // regression on any high-`max_output` model, not just when `reserve_tokens` itself changed.
        let mock = Arc::new(MockTransport::new(vec![]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_compaction(CompactionConfig {
            context_window: 500_000,
            enabled: false,
            ..CompactionConfig::default()
        });
        assert_eq!(
            agent.compaction.summary_max_tokens, 13_107,
            "must rescale to 0.8 * reserve_tokens against the model's real max_output, not reset to \
             the flat 4096 default"
        );

        // Overriding `reserve_tokens` too must rescale against the *new* value, not the one
        // `Agent::new()` originally seeded from.
        let mock2 = Arc::new(MockTransport::new(vec![]));
        let agent2 = Agent::new(mock2, "claude-opus-4-8").with_compaction(CompactionConfig {
            reserve_tokens: 10_000,
            ..CompactionConfig::default()
        });
        assert_eq!(agent2.compaction.summary_max_tokens, 8_000);

        // A caller that *did* deliberately set `summary_max_tokens` to something other than the flat
        // default must still have that respected verbatim, not silently rescaled out from under them.
        let mock3 = Arc::new(MockTransport::new(vec![]));
        let agent3 = Agent::new(mock3, "claude-opus-4-8").with_compaction(CompactionConfig {
            summary_max_tokens: 2_048,
            ..CompactionConfig::default()
        });
        assert_eq!(agent3.compaction.summary_max_tokens, 2_048);
    }

    #[tokio::test]
    async fn thinking_block_is_assembled_with_signature() {
        // A turn that streams a signed thinking block, then text. The assistant message must carry the
        // thinking block first (with its signature) so the next request can replay it.
        let thinking_turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::ThinkingDelta {
                index: 0,
                text: "reasoning…".into(),
            },
            StreamEvent::SignatureDelta {
                index: 0,
                signature: "sig-xyz".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::TextDelta {
                index: 1,
                text: "the answer".into(),
            },
            StreamEvent::ContentBlockStop { index: 1 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ];
        let (agent, _mock) = agent_with(vec![thinking_turn], ToolRegistry::new());
        let mut session = Session::new();
        session.user("think");
        agent.run(&mut session, |_| {}).await.unwrap();

        match &session.messages[1].content[0] {
            ContentBlock::Thinking { text, signature } => {
                assert_eq!(text, "reasoning…");
                assert_eq!(signature, "sig-xyz");
            }
            other => panic!("expected a thinking block first, got {other:?}"),
        }
        assert!(matches!(
            &session.messages[1].content[1],
            ContentBlock::Text { text, .. } if text == "the answer"
        ));
    }

    #[tokio::test]
    async fn thinking_final_resyncs_over_a_dropped_mid_stream_delta() {
        // pi-parity fix: OpenAI Responses' `output_item.done` for a `reasoning` item now resyncs the
        // block's visible thinking text via `StreamEvent::ThinkingFinal` (mirroring `TextFinal` for a
        // text block), so a single dropped/duplicated mid-stream `ThinkingDelta` — a relay hiccup with
        // no transport-level error, nothing else would ever catch it — can't silently leave the
        // persisted/displayed thinking text corrupted. Here the delta alone only ever accumulates
        // "partial reason", but the resync's authoritative text is longer — the assembled block must
        // reflect the resync, not the deltas it replaces.
        let thinking_turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::ThinkingDelta {
                index: 0,
                text: "partial reason".into(),
            },
            StreamEvent::ThinkingFinal {
                index: 0,
                text: "partial reasoning, now complete".into(),
            },
            StreamEvent::SignatureDelta {
                index: 0,
                signature: "sig-xyz".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ];
        let (agent, _mock) = agent_with(vec![thinking_turn], ToolRegistry::new());
        let mut session = Session::new();
        session.user("think");
        agent.run(&mut session, |_| {}).await.unwrap();

        match &session.messages[1].content[0] {
            ContentBlock::Thinking { text, signature } => {
                assert_eq!(
                    text, "partial reasoning, now complete",
                    "ThinkingFinal must replace whatever the accumulated deltas produced, not be \
                     ignored"
                );
                assert_eq!(signature, "sig-xyz");
            }
            other => panic!("expected a thinking block first, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn before_tool_call_hook_blocks_the_tool() {
        use crate::hooks::AgentHooks;
        struct DenyAll;
        #[async_trait]
        impl AgentHooks for DenyAll {
            async fn before_tool_call(
                &self,
                _name: &str,
                _input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                Some("denied by test policy".into())
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t", "echo", r#"{"text":"hi"}"#),
            turn::text("ok"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_hooks(Arc::new(DenyAll));
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        // The tool never ran; the model saw a blocked error result.
        match &session.messages[2].content[0] {
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                assert!(is_error);
                assert!(content.contains("blocked"), "got: {content}");
                assert!(content.contains("denied by test policy"));
            }
            other => panic!("expected blocked tool_result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn before_tool_call_hook_sees_coerced_arguments_not_the_models_raw_stringified_ones() {
        use crate::hooks::AgentHooks;
        // pi-parity fix: `before_tool_call` used to be called with the model's raw, pre-coercion
        // `input` — `coerce_tool_arguments` only ran afterward, right before the tool itself executed.
        // Matches pi's `prepareToolCall`, which calls `validateToolArguments` (type coercion + schema
        // validation) *before* `config.beforeToolCall`: a permission hook must see the same typed
        // arguments the tool is about to run with, not a stringified primitive the model happened to
        // stream on the wire (`"42"` instead of `42`).
        struct CountsTool;
        #[async_trait]
        impl Tool for CountsTool {
            fn name(&self) -> &str {
                "counter"
            }
            fn description(&self) -> &str {
                "takes a numeric count"
            }
            fn input_schema(&self) -> Value {
                json!({
                    "type": "object",
                    "properties": { "count": { "type": "integer" } },
                    "required": ["count"],
                })
            }
            async fn run(
                &self,
                input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok(input["count"].to_string().into())
            }
        }

        struct CapturesHookInput(Arc<std::sync::Mutex<Option<Value>>>);
        #[async_trait]
        impl AgentHooks for CapturesHookInput {
            async fn before_tool_call(
                &self,
                _name: &str,
                input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                *self.0.lock().unwrap() = Some(input.clone());
                None
            }
        }

        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(CountsTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t", "counter", r#"{"count":"42"}"#),
            turn::text("done"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_hooks(Arc::new(CapturesHookInput(captured.clone())));
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        let seen = captured
            .lock()
            .unwrap()
            .clone()
            .expect("the hook must have been called");
        assert_eq!(
            seen["count"],
            json!(42),
            "the hook must see the schema-coerced numeric value, not the model's raw stringified \
             \"42\": {seen:?}"
        );
    }

    #[tokio::test]
    async fn before_tool_call_hook_sees_coerced_arguments_on_the_sequential_execution_path_too() {
        use crate::hooks::AgentHooks;
        // Same pi-parity fix as the default gate loop just above, but for Task #28's fully-interleaved
        // gate→execute→finalize-per-call path (`run_tool_calls_interleaved`) — it has its own,
        // independent `before_tool_call` call site that needed the identical reordering.
        struct SequentialCountsTool;
        #[async_trait]
        impl Tool for SequentialCountsTool {
            fn name(&self) -> &str {
                "counter"
            }
            fn description(&self) -> &str {
                "takes a numeric count, sequential execution"
            }
            fn input_schema(&self) -> Value {
                json!({
                    "type": "object",
                    "properties": { "count": { "type": "integer" } },
                    "required": ["count"],
                })
            }
            async fn run(
                &self,
                input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok(input["count"].to_string().into())
            }
            fn execution_mode(&self) -> Option<crate::tool::ToolExecutionMode> {
                Some(crate::tool::ToolExecutionMode::Sequential)
            }
        }

        struct CapturesHookInput(Arc<std::sync::Mutex<Option<Value>>>);
        #[async_trait]
        impl AgentHooks for CapturesHookInput {
            async fn before_tool_call(
                &self,
                _name: &str,
                input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                *self.0.lock().unwrap() = Some(input.clone());
                None
            }
        }

        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SequentialCountsTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t", "counter", r#"{"count":"7"}"#),
            turn::text("done"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_hooks(Arc::new(CapturesHookInput(captured.clone())));
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        let seen = captured
            .lock()
            .unwrap()
            .clone()
            .expect("the hook must have been called");
        assert_eq!(
            seen["count"],
            json!(7),
            "the sequential-execution-path hook must also see the coerced numeric value: {seen:?}"
        );
    }

    #[tokio::test]
    async fn steering_injects_a_follow_up_and_continues() {
        // The model ends its turn after the first reply; a steering message queued before the run is
        // injected at that stop boundary, driving a second turn.
        let (agent, mock) = agent_with(
            vec![turn::text("first reply"), turn::text("second reply")],
            ToolRegistry::new(),
        );
        let mut session = Session::new();
        session.user("hello");

        let steering = Steering::new();
        steering.push("now do the follow-up");

        let mut steered = false;
        agent
            .run_events_steered(
                &mut session,
                |ev| {
                    if matches!(ev, AgentEvent::Steered { .. }) {
                        steered = true;
                    }
                },
                CancellationToken::new(),
                steering,
            )
            .await
            .unwrap();

        assert!(steered, "a Steered event should have fired");
        assert_eq!(mock.calls(), 2, "the follow-up drove a second model turn");
        // user, assistant(first), user(follow-up), assistant(second)
        assert_eq!(session.messages.len(), 4);
        assert!(matches!(
            &session.messages[2].content[0],
            ContentBlock::Text { text, .. } if text == "now do the follow-up"
        ));
    }

    #[tokio::test]
    async fn steering_injects_a_follow_up_with_images_at_a_stop_boundary() {
        // MEDIUM pi-parity gap (fixed): `Steering`'s queue used to be `VecDeque<String>` with no field
        // for images at all, so a `follow_up`/`steer` carrying image attachments silently dropped them
        // — unlike a fresh `prompt`, which has always supported `images`. A queued `SteeringMessage`
        // with images must land as a real multimodal user turn (text block, then image blocks), the
        // same shape `Message::user_with_images` produces.
        let (agent, mock) = agent_with(
            vec![turn::text("first reply"), turn::text("second reply")],
            ToolRegistry::new(),
        );
        let mut session = Session::new();
        session.user("hello");

        let steering = Steering::new();
        let image = ImageSource::base64("image/png", "aGVsbG8=");
        steering.push(SteeringMessage::new("look at this", vec![image.clone()]));

        agent
            .run_events_steered(&mut session, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();

        assert_eq!(mock.calls(), 2, "the follow-up drove a second model turn");
        // user, assistant(first), user(follow-up: text+image), assistant(second)
        let blocks = &session.messages[2].content;
        assert_eq!(
            blocks.len(),
            2,
            "expected a text block and an image block: {blocks:?}"
        );
        assert!(matches!(
            &blocks[0],
            ContentBlock::Text { text, .. } if text == "look at this"
        ));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Image { source } if *source == image
        ));
    }

    #[tokio::test]
    async fn steering_is_injected_mid_run_with_images_between_tool_turns() {
        // Same gap as above, for the mid-run `steer` lane specifically: an image folded onto the
        // tool-results turn must appear as a real `ContentBlock::Image`, not be silently dropped.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
                turn::text("acknowledged"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("start");

        let steering = Steering::new();
        let image = ImageSource::base64("image/jpeg", "Zm9v");
        steering.push_steer(SteeringMessage::new(
            "also look at this",
            vec![image.clone()],
        ));

        agent
            .run_events_steered(&mut session, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();

        assert_eq!(mock.calls(), 2);
        let blocks = &session.messages[2].content;
        assert!(matches!(blocks[0], ContentBlock::ToolResult { .. }));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { text, .. } if text == "also look at this"
        ));
        assert!(matches!(
            &blocks[2],
            ContentBlock::Image { source } if *source == image
        ));
    }

    #[tokio::test]
    async fn refusal_ends_the_run_without_draining_steering() {
        // A refusal is a distinct terminal condition: unlike an ordinary tool-less stop, it must NOT
        // drain queued steer/follow-up messages and inject them as a new turn (the model would likely
        // just refuse that too) — the run ends immediately, and the queue is left untouched for a
        // later `prompt` call to pick up (see `serve.rs`'s persistent `Steering` handle).
        let (agent, mock) = agent_with(
            vec![turn::refusal("I can't help with that.")],
            ToolRegistry::new(),
        );
        let mut session = Session::new();
        session.user("do something disallowed");

        let steering = Steering::new();
        steering.push("a queued follow-up");

        let mut steered = false;
        agent
            .run_events_steered(
                &mut session,
                |ev| {
                    if matches!(ev, AgentEvent::Steered { .. }) {
                        steered = true;
                    }
                },
                CancellationToken::new(),
                steering.clone(),
            )
            .await
            .unwrap();

        assert!(!steered, "a refusal must never drain/inject steering");
        assert_eq!(
            mock.calls(),
            1,
            "the run must end after the refusal, not continue"
        );
        // user + assistant(refusal) only — no injected follow-up turn.
        assert_eq!(session.messages.len(), 2);
        // The queued message survives, untouched, for a later `prompt` call to pick up.
        assert!(!steering.is_empty());
    }

    #[tokio::test]
    async fn refusal_blocks_dispatch_even_when_a_tool_call_already_streamed() {
        // pi-parity fix: the `StopReason::Refusal` short-circuit used to live *inside* the
        // `calls.is_empty()` branch only — a turn that streamed a complete `tool_use` block and then
        // ended with `Refusal` (a real Anthropic/OpenAI wire shape: a refusal explanation arriving as
        // trailing content after a tool_use block already closed) had a non-empty `calls`, so dispatch
        // ran the tool the model was ultimately blocked from continuing. Matches pi's `agent-loop.ts`,
        // which returns unconditionally on an "error"/"aborted" stop before ever looking at
        // `message.content` for tool calls, dialect-agnostic.
        struct PanicsIfCalled;
        #[async_trait]
        impl Tool for PanicsIfCalled {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "must never actually run in this test"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _input: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                panic!("a refused turn's tool call must never be dispatched");
            }
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(PanicsIfCalled));
        let (agent, mock) = agent_with(
            vec![turn::refusal_after_tool_call(
                "tu_1",
                "echo",
                r#"{"text":"hi"}"#,
            )],
            tools,
        );
        let mut session = Session::new();
        session.user("do something disallowed");

        let mut tool_started = false;
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::ToolStart { .. } = ev {
                    tool_started = true;
                }
            })
            .await
            .unwrap();

        assert!(
            !tool_started,
            "no ToolStart event must fire for a call the model was refused mid-stream"
        );
        assert_eq!(
            mock.calls(),
            1,
            "the run must end after the refusal, not dispatch tools or continue to another turn"
        );
        // user + assistant(tool_use, refused) only — no tool_result turn was ever appended.
        assert_eq!(session.messages.len(), 2);
        assert!(
            session.messages[1]
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. })),
            "the refused tool_use block is still committed to the transcript, just never dispatched"
        );
    }

    #[tokio::test]
    async fn steering_is_injected_mid_run_between_tool_turns() {
        // A steering message queued while the agent is mid-tool-call must be folded into the *same*
        // tool-results user turn (not deferred to a stop boundary), so a client can redirect a busy
        // agent. The message rides alongside the tool_result block — keeping role alternation valid.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
                turn::text("acknowledged"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("start");

        let steering = Steering::new();
        steering.push_steer("actually, also handle the edge case");

        let mut steered = false;
        agent
            .run_events_steered(
                &mut session,
                |ev| {
                    if matches!(ev, AgentEvent::Steered { .. }) {
                        steered = true;
                    }
                },
                CancellationToken::new(),
                steering,
            )
            .await
            .unwrap();

        assert!(steered, "a Steered event should fire mid-run");
        assert_eq!(mock.calls(), 2);
        // The tool-results turn (messages[2]) carries the tool_result *and* the steering text together.
        let blocks = &session.messages[2].content;
        assert!(matches!(blocks[0], ContentBlock::ToolResult { .. }));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { text, .. } if text == "actually, also handle the edge case"
        ));
    }

    #[tokio::test]
    async fn request_stop_ends_the_run_after_the_current_turns_tool_calls_finish() {
        // The model would normally continue to a second turn after its tool call, but a graceful stop
        // is requested before the run starts. The current turn's tool call still runs to completion
        // and its result is committed — the run just never starts a second model call.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
                turn::text("would have replied here"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("start");

        let steering = Steering::new();
        steering.request_stop();

        agent
            .run_events_steered(&mut session, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();

        assert_eq!(
            mock.calls(),
            1,
            "the run must stop after the first turn's tool call, not continue to a second model call"
        );
        // user, assistant(tool_use), user(tool_result) — the tool call ran and its result is committed.
        assert_eq!(session.messages.len(), 3);
        assert!(matches!(
            session.messages[2].content[0],
            ContentBlock::ToolResult { .. }
        ));
    }

    #[tokio::test]
    async fn request_stop_takes_priority_over_a_queued_follow_up() {
        // A follow-up is queued (would normally drive a second turn), but a graceful stop was also
        // requested. The stop wins: the run ends after the first reply, and the follow-up is left
        // queued untouched — mirroring the refusal case's "nothing is lost" contract.
        let (agent, mock) = agent_with(vec![turn::text("first reply")], ToolRegistry::new());
        let mut session = Session::new();
        session.user("hello");

        let steering = Steering::new();
        steering.push("a queued follow-up");
        steering.request_stop();

        let mut steered = false;
        agent
            .run_events_steered(
                &mut session,
                |ev| {
                    if matches!(ev, AgentEvent::Steered { .. }) {
                        steered = true;
                    }
                },
                CancellationToken::new(),
                steering.clone(),
            )
            .await
            .unwrap();

        assert!(
            !steered,
            "a stop request must win over draining a queued follow-up"
        );
        assert_eq!(mock.calls(), 1, "the run must end after the first turn");
        assert_eq!(session.messages.len(), 2);
        assert!(
            !steering.is_empty(),
            "the follow-up must remain queued, not be dropped"
        );
    }

    #[tokio::test]
    async fn request_stop_still_lets_a_folded_in_steer_message_emit_its_event_first() {
        // A mid-run steer message folds into the current tool-results turn regardless of a pending
        // stop request (it's already part of the committed transcript by the time the stop is
        // checked) — only whether the run continues to another model call is affected.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent, mock) = agent_with(
            vec![
                turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
                turn::text("would have replied here"),
            ],
            tools,
        );
        let mut session = Session::new();
        session.user("start");

        let steering = Steering::new();
        steering.push_steer("also handle this");
        steering.request_stop();

        let mut steered = false;
        agent
            .run_events_steered(
                &mut session,
                |ev| {
                    if matches!(ev, AgentEvent::Steered { .. }) {
                        steered = true;
                    }
                },
                CancellationToken::new(),
                steering,
            )
            .await
            .unwrap();

        assert!(
            steered,
            "the folded-in steer message must still emit its Steered event"
        );
        assert_eq!(
            mock.calls(),
            1,
            "the stop request must still end the run after this turn"
        );
        let blocks = &session.messages[2].content;
        assert!(matches!(blocks[0], ContentBlock::ToolResult { .. }));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Text { text, .. } if text == "also handle this"
        ));
    }

    #[tokio::test]
    async fn a_stop_request_left_pending_by_a_refusal_does_not_bleed_into_the_next_run() {
        // A refusal ends the run without ever checking the stop request (it's a distinct terminal
        // condition, checked first — see `refusal_ends_the_run_without_draining_steering`). If a client
        // had also requested a graceful stop on that same run, the request must still not survive to
        // affect a later, unrelated run that reuses the same `Steering` handle.
        let (agent, mock) = agent_with(
            vec![turn::refusal("I can't help with that.")],
            ToolRegistry::new(),
        );
        let mut session = Session::new();
        session.user("do something disallowed");

        let steering = Steering::new();
        steering.request_stop();

        agent
            .run_events_steered(
                &mut session,
                |_| {},
                CancellationToken::new(),
                steering.clone(),
            )
            .await
            .unwrap();
        assert_eq!(mock.calls(), 1, "run A ends on the refusal");

        // Run B: an unrelated run on a fresh session, sharing the same `steering` handle. Uses a
        // tool-call turn so a leftover `true` flag (if the guard above didn't clear it) would cut it
        // short after just one call instead of the two it should take.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let (agent_b, mock_b) = agent_with(
            vec![
                turn::tool_call("tu_2", "echo", r#"{"text":"pong"}"#),
                turn::text("done"),
            ],
            tools,
        );
        let mut session_b = Session::new();
        session_b.user("start again");
        agent_b
            .run_events_steered(&mut session_b, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();
        assert_eq!(
            mock_b.calls(),
            2,
            "a stop request left over from a different, already-ended run must not affect this one"
        );
    }

    /// Records `session.messages.len()` at every checkpoint — a stand-in for a host that would persist
    /// incrementally (see `serve.rs`'s own `CheckpointHook` impl) without doing real I/O in a test.
    struct RecordingCheckpoint {
        lens: std::sync::Mutex<Vec<usize>>,
    }

    #[async_trait]
    impl CheckpointHook for RecordingCheckpoint {
        async fn checkpoint(&self, session: &Session) {
            self.lens
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(session.messages.len());
        }
    }

    #[tokio::test]
    async fn checkpoint_fires_after_each_tool_round_trip_not_just_at_the_end() {
        // Two tool round-trips followed by a final text turn: the checkpoint must fire once before the
        // very first request (covering the user's own prompt — the pi-parity gap where a crash before
        // any tool round-trip completed used to lose it entirely), *twice per round-trip* — once the
        // instant the assistant's own `tool_use` turn is committed (before those tools ever run — a
        // crash mid-execution must not lose the record that the model asked for them), and again once
        // the matching `tool_results` land — and once more for the final tool-less reply, before its own
        // `AgentEnd`: a plain conversational turn is just as resumable a point as a tool round-trip's own
        // halves, and must not rely solely on a caller's own post-run persist to ever be recorded.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "echo", r#"{"text":"one"}"#),
            turn::tool_call("tu_2", "echo", r#"{"text":"two"}"#),
            turn::text("done"),
        ]));
        let checkpoint = Arc::new(RecordingCheckpoint {
            lens: std::sync::Mutex::new(Vec::new()),
        });
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_checkpoint_hook(checkpoint.clone());
        let mut session = Session::new();
        session.user("go");
        agent.run(&mut session, |_| {}).await.unwrap();

        let lens = checkpoint.lens.lock().unwrap().clone();
        // user = 1 (pre-first-request checkpoint); assistant(tool_use) = 2 (pre-dispatch checkpoint)
        // then +tool_results = 3 (post-dispatch checkpoint); +assistant(tool_use) = 4, +tool_results = 5
        // for the second round-trip; +assistant(text) = 6 for the final tool-less reply, checkpointed
        // before its own `AgentEnd`.
        assert_eq!(
            lens,
            vec![1, 2, 3, 4, 5, 6],
            "checkpoint must fire before the first request, both before dispatch (right after the \
             tool_use turn commits) and after (once tool_results land) for each round-trip, and once \
             more for the final tool-less reply"
        );
        assert_eq!(
            session.messages.len(),
            6,
            "sanity: final count after the text turn"
        );
    }

    #[tokio::test]
    async fn checkpoint_fires_before_tool_dispatch_even_if_the_tool_never_returns() {
        // The exact crash-recovery scenario the pre-dispatch checkpoint exists for: the assistant's
        // `tool_use` turn must be durable *before* a slow/hung tool ever resolves — a host checkpointing
        // incrementally must be able to see "the model asked for this call" without waiting for the
        // call itself to finish.
        struct HangTool;
        #[async_trait]
        impl Tool for HangTool {
            fn name(&self) -> &str {
                "hang"
            }
            fn description(&self) -> &str {
                "never returns"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                futures::future::pending().await
            }
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(HangTool));
        let mock = Arc::new(MockTransport::new(vec![turn::tool_call(
            "tu_1", "hang", "{}",
        )]));
        let checkpoint = Arc::new(RecordingCheckpoint {
            lens: std::sync::Mutex::new(Vec::new()),
        });
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_checkpoint_hook(checkpoint.clone());
        let mut session = Session::new();
        session.user("go");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let result = agent
            .run_events_cancellable(&mut session, |_| {}, cancel)
            .await;
        assert!(matches!(result, Err(Error::Cancelled)));

        let lens = checkpoint.lens.lock().unwrap().clone();
        assert_eq!(
            lens,
            vec![1, 2],
            "the user's prompt must already be checkpointed before the first request (1), and the \
             tool_use turn must already be checkpointed (2) even though the tool it called never \
             returned and the run was cancelled mid-dispatch"
        );
    }

    #[tokio::test]
    async fn checkpoint_fires_right_after_a_compaction_rewrite_lands() {
        // pi-parity gap: `apply_summary` physically rewrites `session.messages` in place — spending one
        // or two real, paid-for model calls to produce it — but nothing checkpointed that rewrite. A
        // crash between the rewrite landing and the next natural checkpoint (which could be a whole
        // tool round-trip away, or never come at all if the run ends on a tool-less reply) lost it
        // entirely: on resume the persisted session still held the old, oversized history, so the very
        // next turn re-triggered the identical (already-paid-for) compaction again.
        let session_messages = vec![
            Message::user("look at this"),
            Message::assistant(vec![ContentBlock::text("ok")]),
            Message::user("and this"),
            Message::assistant(vec![ContentBlock::text("done")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::new(vec![turn::text("## Goal\na summary")]));
        let checkpoint = Arc::new(RecordingCheckpoint {
            lens: std::sync::Mutex::new(Vec::new()),
        });
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_compaction(CompactionConfig {
                keep_recent_tokens: 1,
                ..CompactionConfig::default()
            })
            .with_checkpoint_hook(checkpoint.clone());
        let cancel = CancellationToken::new();
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |_| {},
                None,
            )
            .await
            .unwrap();
        assert!(compacted.compacted());

        let lens = checkpoint.lens.lock().unwrap().clone();
        assert_eq!(
            lens,
            vec![session.messages.len()],
            "the checkpoint must fire exactly once, right after the rewritten (post-compaction) \
             history is in place — got: {lens:?}, session now has {} messages",
            session.messages.len()
        );
    }

    #[tokio::test]
    async fn checkpoint_fires_when_a_steered_message_is_injected_at_a_stop_boundary() {
        // A follow-up queued before the model would otherwise stop must also land on a checkpoint —
        // the injected user message is itself a valid, resumable point. Plus: every tool-less reply
        // along the way (both "first answer", which continues via the queued follow-up, and "second
        // answer", which actually ends the run) gets its own checkpoint too.
        let (agent, _mock) = agent_with(
            vec![turn::text("first answer"), turn::text("second answer")],
            ToolRegistry::new(),
        );
        let checkpoint = Arc::new(RecordingCheckpoint {
            lens: std::sync::Mutex::new(Vec::new()),
        });
        let agent = agent.with_checkpoint_hook(checkpoint.clone());
        let mut session = Session::new();
        session.user("go");
        let steering = Steering::new();
        steering.push("a follow-up question");

        agent
            .run_events_steered(&mut session, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();

        let lens = checkpoint.lens.lock().unwrap().clone();
        // user = 1 (pre-first-request checkpoint); assistant("first answer") = 2 (tool-less-reply
        // checkpoint); user(follow-up) = 3 (right when the follow-up is injected); assistant("second
        // answer") = 4 (tool-less-reply checkpoint, right before this run's own final `AgentEnd`).
        assert_eq!(lens, vec![1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn checkpoint_fires_before_returning_for_a_tool_less_reply_with_nothing_queued() {
        // pi-parity gap (task 15): a run that never calls a tool at all — the single most common
        // shape, a plain conversational reply — used to never reach a single checkpoint call before
        // this fix, relying entirely on a caller's own post-run persist. It's now checkpointed right
        // before its own `AgentEnd`, the same as every other terminal path in the loop.
        let (agent, _mock) =
            agent_with(vec![turn::text("just a plain reply")], ToolRegistry::new());
        let checkpoint = Arc::new(RecordingCheckpoint {
            lens: std::sync::Mutex::new(Vec::new()),
        });
        let agent = agent.with_checkpoint_hook(checkpoint.clone());
        let mut session = Session::new();
        session.user("hi");

        agent.run(&mut session, |_| {}).await.unwrap();

        let lens = checkpoint.lens.lock().unwrap().clone();
        // user = 1 (pre-first-request checkpoint); assistant(reply) = 2 (tool-less-reply checkpoint,
        // right before the run's own `AgentEnd` — previously never reached at all).
        assert_eq!(lens, vec![1, 2]);
    }

    #[tokio::test]
    async fn checkpoint_before_the_first_request_would_survive_a_crash_with_no_model_response_yet()
    {
        // pi-parity gap (task 15): simulates the exact exposure this fix closes — a process killed
        // between "the user's prompt is durably queued into the session" and "the model is ever
        // called." Before this fix, nothing checkpointed the user's own prompt until *after* a full
        // tool round-trip (or a would-stop boundary) completed, so a crash in this exact window lost
        // the user's own prompt from the persisted transcript entirely — on restart it looked like it
        // was never submitted. Confirms the *first* checkpoint call already fires before the model is
        // ever invoked, and that its snapshot already contains the user's message.
        struct SnapshotOnFirstCheckpoint {
            mock: Arc<MockTransport>,
            snapshot: std::sync::Mutex<Option<(usize, Vec<Message>)>>,
        }
        #[async_trait]
        impl CheckpointHook for SnapshotOnFirstCheckpoint {
            async fn checkpoint(&self, session: &Session) {
                let mut guard = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
                if guard.is_none() {
                    *guard = Some((self.mock.calls(), session.messages.as_ref().clone()));
                }
            }
        }

        let (agent, mock) = agent_with(vec![turn::text("hello back")], ToolRegistry::new());
        let checkpoint = Arc::new(SnapshotOnFirstCheckpoint {
            mock: mock.clone(),
            snapshot: std::sync::Mutex::new(None),
        });
        let agent = agent.with_checkpoint_hook(checkpoint.clone());
        let mut session = Session::new();
        session.user("don't lose me");

        agent.run(&mut session, |_| {}).await.unwrap();

        let (calls_at_first_checkpoint, messages_at_first_checkpoint) = checkpoint
            .snapshot
            .lock()
            .unwrap()
            .clone()
            .expect("the checkpoint hook must have fired at least once");
        assert_eq!(
            calls_at_first_checkpoint, 0,
            "the first checkpoint must fire before the model is ever called — a 'crash' right here \
             must still find a durable copy of the user's prompt"
        );
        assert!(
            messages_at_first_checkpoint
                .iter()
                .any(|m| m.role == Role::User
                    && m.content.iter().any(
                        |b| matches!(b, ContentBlock::Text { text, .. } if text == "don't lose me")
                    )),
            "the user's own prompt must already be present in the snapshot a 'crash' here would \
             leave behind: {messages_at_first_checkpoint:?}"
        );
    }

    #[tokio::test]
    async fn next_turn_queued_while_idle_lands_in_the_very_first_request_of_the_next_run() {
        // Task 43: this crate's next-turn lane guarantees a message queued while idle is prepended to
        // the very next prompt's own first model request — visible to the model on turn 1 of that
        // prompt, not just eventually after a tool round-trip or a stop boundary the way `steer`/
        // `follow_up` are. A beyond-only capability, not pi parity — see `steering.rs`'s own module doc
        // comment for why no equivalent lane exists in pi's real product. Confirms
        // `Steering::push_next_turn` actually reaches the first request `run_events_steered` sends,
        // folded onto the same user turn as the prompt itself.
        let (agent, mock) = agent_with(vec![turn::text("ok")], ToolRegistry::new());
        let steering = Steering::new();
        steering.push_next_turn("queued while idle");
        let mut session = Session::new();
        session.user("the actual prompt");

        agent
            .run_events_steered(&mut session, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();

        let requests = mock.requests();
        let first_request = &requests[0];
        let last_message = first_request
            .messages
            .last()
            .expect("the first request must carry at least one message");
        assert_eq!(last_message.role, Role::User);
        let texts: Vec<&str> = last_message
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec!["queued while idle", "the actual prompt"],
            "the next-turn message must be folded in front of the actual prompt, in the very first \
             request this run sends"
        );
    }

    #[tokio::test]
    async fn cancellation_aborts_a_hung_tool() {
        use std::time::Duration;

        // A tool that never returns. Under cancellation the loop must drop its future and bail, not
        // wait forever — the same mechanism that kills a hung `bash` subprocess (`kill_on_drop`).
        struct HangTool;
        #[async_trait]
        impl Tool for HangTool {
            fn name(&self) -> &str {
                "hang"
            }
            fn description(&self) -> &str {
                "never returns"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                futures::future::pending::<()>().await;
                unreachable!()
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(HangTool));
        let (agent, _mock) = agent_with(
            vec![turn::tool_call("t", "hang", "{}"), turn::text("done")],
            tools,
        );
        let mut session = Session::new();
        session.user("go");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            agent.run_events_cancellable(&mut session, |_| {}, cancel),
        )
        .await
        .expect("cancellation must abort the hung tool, not hang the run");
        assert!(matches!(result, Err(Error::Cancelled)));
    }

    #[tokio::test]
    async fn cancellation_mid_dispatch_repairs_the_orphaned_tool_use() {
        use std::time::Duration;

        // One tool returns immediately, the other hangs forever. Cancelling once the fast one has
        // surely finished (but the hung one hasn't) must still leave the session in a valid,
        // resumable shape: the assistant's tool_use message was already committed before dispatch, so
        // without a repair step the run would end on an orphaned tool_use with no matching
        // tool_result — a shape both Anthropic and OpenAI reject on resume.
        struct FastTool;
        #[async_trait]
        impl Tool for FastTool {
            fn name(&self) -> &str {
                "fast"
            }
            fn description(&self) -> &str {
                "returns immediately"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                Ok("fast-done".into())
            }
        }
        struct HangTool;
        #[async_trait]
        impl Tool for HangTool {
            fn name(&self) -> &str {
                "hang"
            }
            fn description(&self) -> &str {
                "never returns"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                futures::future::pending::<()>().await;
                unreachable!()
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(FastTool));
        tools.register(Arc::new(HangTool));

        let two_calls = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                index: 0,
                id: "f".into(),
                name: "fast".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::ToolUseStart {
                index: 0,
                id: "h".into(),
                name: "hang".into(),
            },
            StreamEvent::ContentBlockStop { index: 0 },
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ];
        let (agent, _mock) = agent_with(vec![two_calls], tools);
        let mut session = Session::new();
        session.user("go");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            // Long enough that `fast` has certainly resolved; `hang` never will regardless.
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            agent.run_events_cancellable(&mut session, |_| {}, cancel),
        )
        .await
        .expect("cancellation must abort the run, not hang it");
        assert!(matches!(result, Err(Error::Cancelled)));

        // The session must end on a valid tool_results message, not an orphaned tool_use: the real
        // "fast" result survives, and "hang" gets a synthesized error result standing in for the call
        // that never finished.
        let last = session.messages.last().expect("session has messages");
        assert_eq!(last.role, Role::User);
        assert_eq!(last.content.len(), 2);
        match (&last.content[0], &last.content[1]) {
            (
                ContentBlock::ToolResult {
                    tool_use_id: fid,
                    content: fcontent,
                    is_error: ferr,
                    ..
                },
                ContentBlock::ToolResult {
                    tool_use_id: hid,
                    content: hcontent,
                    is_error: herr,
                    ..
                },
            ) => {
                assert_eq!(fid, "f");
                assert_eq!(fcontent, "fast-done");
                assert!(
                    !ferr,
                    "the call that actually finished must not be flagged an error"
                );
                assert_eq!(hid, "h");
                assert!(
                    hcontent.contains("cancelled"),
                    "the call that never finished should carry a synthesized cancellation result: {hcontent}"
                );
                assert!(
                    herr,
                    "a synthesized cancellation result must be flagged an error"
                );
            }
            other => panic!("expected two ordered tool_result blocks, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancellation_during_before_tool_call_hook_prevents_dispatch_for_a_single_call_batch() {
        // Task #3 (pi-parity, high-severity): the phase-1 gate loop used to check
        // `cancel.is_cancelled()` only at the *top* of each iteration, before that call's own
        // `before_tool_call` hook ran. If cancellation fired *during* the hook and the call was the
        // last (or only) one in the batch, there was no later iteration to catch it — the call was
        // marked `Ready` and phase 2 dispatched it for real despite the run already having been
        // cancelled. Concrete failure this pins: a single-tool-call turn (`bash rm -f file`) with a
        // slow permission-check hook; the client cancels mid-hook; the command must NOT run.
        struct CancelsDuringPermissionCheck(CancellationToken);
        #[async_trait]
        impl AgentHooks for CancelsDuringPermissionCheck {
            async fn before_tool_call(
                &self,
                _name: &str,
                _input: &Value,
                _session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                // Simulates a permission check that takes long enough for the client's cancellation to
                // land while it's still in flight — then allows the call, exactly the case the fix
                // must still catch (a hook that merely observes cancellation and blocks would already
                // be safe; one that doesn't is the actual bug).
                self.0.cancel();
                None
            }
        }

        struct TracksExecutionTool(Arc<std::sync::atomic::AtomicBool>);
        #[async_trait]
        impl Tool for TracksExecutionTool {
            fn name(&self) -> &str {
                "bash"
            }
            fn description(&self) -> &str {
                "runs a command"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok("ran".into())
            }
        }

        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(TracksExecutionTool(ran.clone())));

        let cancel = CancellationToken::new();
        let (agent, _mock) = agent_with(
            vec![
                turn::tool_call("t1", "bash", r#"{"command":"rm -f file"}"#),
                turn::text("done"),
            ],
            tools,
        );
        let agent = agent.with_hooks(Arc::new(CancelsDuringPermissionCheck(cancel.clone())));

        let mut session = Session::new();
        session.user("delete the file");

        let result = agent
            .run_events_cancellable(&mut session, |_| {}, cancel)
            .await;
        assert!(
            matches!(result, Err(Error::Cancelled)),
            "expected Err(Cancelled), got {result:?}"
        );
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "the tool must not run once cancellation fired during the gating hook, even for a \
             single-call (last-in-batch) turn"
        );
        // The session must still end in a valid, resumable shape — a synthesized error tool_result,
        // not an orphaned tool_use.
        let last = session.messages.last().expect("session has messages");
        assert_eq!(last.role, Role::User);
        match &last.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "t1");
                assert!(content.contains("cancelled"), "got: {content}");
                assert!(*is_error);
            }
            other => panic!("expected a synthesized cancelled tool_result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancellation_interrupts_a_stalled_model_stream() {
        use std::time::Duration;

        // A transport whose stream opens but never yields another event — a model gone silent. The
        // turn-level race must interrupt the blocked `next()` rather than wait out the idle timeout.
        struct StalledTransport;
        #[async_trait]
        impl ModelTransport for StalledTransport {
            async fn stream(&self, _req: ModelRequest) -> Result<crate::transport::EventStream> {
                // MessageStart, then nothing, ever.
                let s = futures::stream::once(async { Ok(StreamEvent::MessageStart) })
                    .chain(futures::stream::pending());
                Ok(Box::pin(s))
            }
        }

        let agent = Agent::new(Arc::new(StalledTransport), "claude-opus-4-8");
        let mut session = Session::new();
        session.user("go");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            agent.run_events_cancellable(&mut session, |_| {}, cancel),
        )
        .await
        .expect("cancellation must interrupt the stalled stream");
        assert!(matches!(result, Err(Error::Cancelled)));
    }

    #[tokio::test]
    async fn cancellation_mid_stream_persists_partial_content_as_an_aborted_turn() {
        use std::time::Duration;

        // pi-parity fix (`packages/ai/test/abort.test.ts`'s `testAbortSignal`,
        // `packages/coding-agent/test/suite/agent-session-retry-events.test.ts:334-359` "emits
        // agent_end for aborted runs and persists the aborted assistant message"): a stream cancelled
        // after real content has already arrived must not discard it — pi persists a
        // `stopReason:"aborted"` message with the partial text, and a follow-up prompt still works.
        // A transport that streams two text deltas, then goes silent forever — cancellation must land
        // while genuine partial content is already accumulated, not race a stream that would complete
        // or error out on its own.
        struct PartialThenStalledTransport;
        #[async_trait]
        impl ModelTransport for PartialThenStalledTransport {
            async fn stream(&self, _req: ModelRequest) -> Result<crate::transport::EventStream> {
                let s = futures::stream::iter(vec![
                    Ok(StreamEvent::MessageStart),
                    Ok(StreamEvent::TextDelta {
                        index: 0,
                        text: "partial an".into(),
                    }),
                    Ok(StreamEvent::TextDelta {
                        index: 0,
                        text: "swer".into(),
                    }),
                ])
                .chain(futures::stream::pending());
                Ok(Box::pin(s))
            }
        }

        let agent = Agent::new(Arc::new(PartialThenStalledTransport), "claude-opus-4-8");
        let mut session = Session::new();
        session.user("go");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            agent.run_events_cancellable(&mut session, |_| {}, cancel),
        )
        .await
        .expect("cancellation must interrupt the stalled stream");
        // The external contract is unchanged: the run still ends in `Err(Error::Cancelled)` — whole-
        // run retry exclusion, `serve.rs`'s `abort` RPC handling, and every other caller keying off
        // this still work exactly as before.
        assert!(matches!(result, Err(Error::Cancelled)));

        assert_eq!(
            session.messages.len(),
            2,
            "expected [user, assistant(aborted, partial)]"
        );
        let closing = &session.messages[1];
        assert_eq!(closing.role, Role::Assistant);
        assert!(
            closing.aborted,
            "the persisted message must be flagged aborted"
        );
        assert!(
            closing.error_message.is_none(),
            "aborted is a distinct condition from error — must not also set error_message"
        );
        assert_eq!(
            closing.content,
            vec![ContentBlock::text("partial answer")],
            "the partial text that had already streamed must survive, not be discarded"
        );
    }

    #[tokio::test]
    async fn a_prompt_after_a_mid_stream_abort_does_not_double_push_a_user_turn() {
        use std::time::Duration;

        // The other half of the fix: once the aborted turn closes with a real assistant record, a
        // fresh `session.user(...)` must restore valid role alternation and a normal run must succeed
        // — mirroring pi's `testAbortSignal`, which sends a real follow-up after the abort and asserts
        // it completes normally.
        struct StallsOnFirstCallOnly {
            calls: std::sync::atomic::AtomicUsize,
        }
        #[async_trait]
        impl ModelTransport for StallsOnFirstCallOnly {
            async fn stream(&self, _req: ModelRequest) -> Result<crate::transport::EventStream> {
                if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    let s = futures::stream::iter(vec![
                        Ok(StreamEvent::MessageStart),
                        Ok(StreamEvent::TextDelta {
                            index: 0,
                            text: "partial".into(),
                        }),
                    ])
                    .chain(futures::stream::pending());
                    Ok(Box::pin(s))
                } else {
                    Ok(Box::pin(futures::stream::iter(
                        turn::text("recovered").into_iter().map(Ok),
                    )))
                }
            }
        }

        let agent = Agent::new(
            Arc::new(StallsOnFirstCallOnly {
                calls: std::sync::atomic::AtomicUsize::new(0),
            }),
            "claude-opus-4-8",
        );
        let mut session = Session::new();
        session.user("go");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });
        tokio::time::timeout(
            Duration::from_secs(5),
            agent.run_events_cancellable(&mut session, |_| {}, cancel),
        )
        .await
        .expect("cancellation must interrupt the stalled stream")
        .expect_err("aborted run must still return Err(Cancelled)");

        assert_eq!(session.messages[1].role, Role::Assistant);
        session.user("after abort");
        assert_eq!(session.messages[2].role, Role::User);
        agent.run(&mut session, |_| {}).await.unwrap();
        let last = session.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert!(!last.aborted);
        assert!(!last.content.is_empty());
    }

    #[tokio::test]
    async fn pre_connect_cancellation_closes_out_the_pending_user_turn() {
        // pi-parity gap (H5): a token already cancelled *before* the model ever connects (the top-of-
        // loop check, or `run_turn`'s own pre-connect race) used to return `Err(Cancelled)` with no
        // session mutation at all — unlike a mid-stream abort, which always closes the turn out with an
        // aborted assistant record (see the sibling tests above). Left as the caller's own unanswered
        // `user` turn, a follow-up prompt would push a *second* consecutive `user` message, which no
        // dialect accepts. pi's own `StreamFn` contract requires even a never-streamed request to
        // resolve to a final `stopReason: "aborted"` message for exactly this reason.
        let agent = Agent::new(Arc::new(MockTransport::new(vec![])), "claude-opus-4-8");
        let mut session = Session::new();
        session.user("go");

        // Already cancelled — the top-of-loop check trips before a request is ever built, so the
        // transport (empty turn list; a real call would panic on it) is never actually invoked.
        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = agent
            .run_events_cancellable(&mut session, |_| {}, cancel)
            .await;
        assert!(matches!(result, Err(Error::Cancelled)));

        assert_eq!(
            session.messages.len(),
            2,
            "the pending user turn must be closed out with an aborted assistant record, not left \
             dangling: {:?}",
            session.messages
        );
        let last = &session.messages[1];
        assert_eq!(last.role, Role::Assistant);
        assert!(last.aborted, "closing record must be flagged aborted");
    }

    #[tokio::test]
    async fn cancellation_clears_queued_steer_and_follow_up_messages() {
        // Beyond design choice (not a pi port — see `clear_run_scoped`'s own doc comment): clearing the
        // steer/follow-up queues on cancellation means a message queued right before (or during)
        // cancellation doesn't silently ride into whatever unrelated run reuses the same `Steering`
        // handle next. `agent.rs` used to never call `Steering::clear()` (or its narrower successor,
        // `clear_run_scoped()`) on any of its cancellation exit paths at all.
        let agent = Agent::new(Arc::new(MockTransport::new(vec![])), "claude-opus-4-8");
        let mut session = Session::new();
        session.user("go");
        let steering = Steering::new();
        steering.push("a follow-up queued before cancelling");
        steering.push_steer("a steer queued before cancelling");
        assert_eq!(steering.pending_count(), 2);

        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = agent
            .run_events_steered(&mut session, |_| {}, cancel, steering.clone())
            .await;
        assert!(matches!(result, Err(Error::Cancelled)));

        assert_eq!(
            steering.pending_count(),
            0,
            "both lanes must be cleared once a run ends via cancellation"
        );
    }

    #[tokio::test]
    async fn cancellation_does_not_clear_a_queued_next_turn_message() {
        // Cancellation must use the narrower `clear_run_scoped()`, not the general `clear()` — the
        // steer/follow-up lanes are cleared as a beyond-only conservative default (see
        // `clear_run_scoped`'s own doc comment for why, and how that diverges from pi's real headless
        // abort path), but the next-turn lane is left untouched regardless. A message queued via
        // `push_next_turn` is meant to survive into whatever prompt comes next, aborted or not — a run
        // ending in cancellation must not silently drop it.
        //
        // The next-turn lane is drained exactly once, right at the very start of
        // `run_events_steered` (folded onto this run's own first turn) — so to actually exercise
        // "does cancellation clear it," the message has to be queued *after* that point, mid-run,
        // simulating one meant for whatever run comes *after* this one. A hung tool gives cancellation
        // something to interrupt mid-dispatch, well past that initial drain.
        struct HangTool;
        #[async_trait]
        impl Tool for HangTool {
            fn name(&self) -> &str {
                "hang"
            }
            fn description(&self) -> &str {
                "never returns"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                futures::future::pending::<()>().await;
                unreachable!()
            }
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(HangTool));
        let (agent, _mock) = agent_with(
            vec![turn::tool_call("t", "hang", "{}"), turn::text("done")],
            tools,
        );
        let mut session = Session::new();
        session.user("go");

        let steering = Steering::new();
        steering.push("a follow-up queued before cancelling");
        steering.push_steer("a steer queued before cancelling");

        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        let switch_steering = steering.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            switch_steering.push_next_turn("must survive cancellation");
            trigger.cancel();
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            agent.run_events_steered(&mut session, |_| {}, cancel, steering.clone()),
        )
        .await
        .expect("cancellation must abort the hung tool, not hang the run");
        assert!(matches!(result, Err(Error::Cancelled)));

        assert_eq!(
            steering.pending_count(),
            1,
            "the steer/follow-up lanes must be cleared but the next-turn lane must survive"
        );
        assert_eq!(
            steering.drain_next_turn(),
            vec!["must survive cancellation".to_string()],
            "the next-turn message itself must be exactly what was queued before cancellation"
        );
    }

    #[tokio::test]
    async fn a_prompt_after_a_pre_connect_cancellation_does_not_double_push_a_user_turn() {
        // The other half of H5's fix: once the pre-connect-cancelled turn closes with a real assistant
        // record, a fresh `session.user(...)` must restore valid role alternation and a normal run must
        // succeed — mirroring the equivalent mid-stream-abort recovery test above.
        let agent = Agent::new(Arc::new(MockTransport::new(vec![])), "claude-opus-4-8");
        let mut session = Session::new();
        session.user("go");
        let cancel = CancellationToken::new();
        cancel.cancel();
        agent
            .run_events_cancellable(&mut session, |_| {}, cancel)
            .await
            .expect_err("pre-connect cancellation must still return Err(Cancelled)");

        assert_eq!(session.messages[1].role, Role::Assistant);
        session.user("after abort");
        assert_eq!(session.messages[2].role, Role::User);

        // A fresh, non-cancelled run against a transport that actually answers must now succeed —
        // proving role alternation is valid (no two consecutive `user` messages to 400 on).
        let agent = Agent::new(
            Arc::new(MockTransport::new(vec![turn::text("recovered")])),
            "claude-opus-4-8",
        );
        agent.run(&mut session, |_| {}).await.unwrap();
        let last = session.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert!(!last.aborted);
        assert!(!last.content.is_empty());
    }

    #[tokio::test]
    async fn a_non_retryable_mid_stream_failure_keeps_whatever_content_had_already_streamed() {
        // pi-parity gap: a mid-stream transport/decode failure used to discard the whole `Accumulator`
        // via `?` without ever calling `finish()` — losing real, already-generated prose (or a
        // half-formed edit) the instant a network blip or in-band provider error struck partway through
        // a long response. pi's own dialects keep whatever streamed before *either* an abort or an
        // error; only a pre-connect failure (nothing ever streamed) should still close out empty.
        let mock = Arc::new(MockTransport::scripted(vec![vec![
            Ok(StreamEvent::MessageStart),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "here is the start of a real answer".into(),
            }),
            // A permission error is explicitly non-retryable (see
            // `is_retryable_mid_stream_still_excludes_permanent_failures`) — this must reach the
            // terminal `Err` arm on the very first attempt, no retry involved.
            Err(Error::Transport(
                "provider stream error: permission_error: not allowed".into(),
            )),
        ]]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8");
        let mut session = Session::new();
        session.user("go");

        let result = agent.run(&mut session, |_| {}).await;
        assert!(matches!(result, Err(Error::Transport(_))));
        assert_eq!(mock.calls(), 1, "a non-retryable failure must not retry");

        assert_eq!(session.messages.len(), 2);
        let last = &session.messages[1];
        assert_eq!(last.role, Role::Assistant);
        assert_eq!(
            last.content,
            vec![ContentBlock::text("here is the start of a real answer")],
            "the partial text that streamed before the failure must be kept, not discarded"
        );
        assert!(
            last.error_message.is_some(),
            "still tagged as an error record, not silently treated as a clean success"
        );
        assert!(!last.aborted, "a transport failure is not a cancellation");
    }

    #[tokio::test]
    async fn a_pre_connect_transport_failure_still_closes_out_empty() {
        // The other half of the fix: when the failure strikes *before* the model ever streams anything
        // back (a connect-level rejection), there's no partial content to keep — the original bare
        // closing record is still correct here, not every transport error should suddenly grow content.
        struct RejectsBeforeConnecting;
        #[async_trait]
        impl ModelTransport for RejectsBeforeConnecting {
            async fn stream(&self, _req: ModelRequest) -> Result<crate::transport::EventStream> {
                Err(Error::Transport(
                    "provider stream error: permission_error: not allowed".into(),
                ))
            }
        }
        let agent = Agent::new(Arc::new(RejectsBeforeConnecting), "claude-opus-4-8");
        let mut session = Session::new();
        session.user("go");

        let result = agent.run(&mut session, |_| {}).await;
        assert!(matches!(result, Err(Error::Transport(_))));

        assert_eq!(session.messages.len(), 2);
        let last = &session.messages[1];
        assert_eq!(last.content, vec![ContentBlock::text(String::new())]);
        assert!(last.error_message.is_some());
    }

    struct PanicTool;
    #[async_trait]
    impl Tool for PanicTool {
        fn name(&self) -> &str {
            "panic_tool"
        }
        fn description(&self) -> &str {
            "always panics"
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn run(
            &self,
            _: Value,
        ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
            panic!("boom: this tool always panics");
        }
    }

    #[tokio::test]
    async fn a_panicking_tool_becomes_an_error_tool_result_not_a_dead_run() {
        // pi-parity gap: a panic inside a tool's own `run` (a stray `.unwrap()`, an index-out-of-bounds)
        // used to unwind straight through the whole `run_events_steered` call — no `Result` a tool
        // could return instead, so the only way to guard against it is a panic boundary in the loop
        // itself. pi's `agent-loop.ts` degrades this to one failed tool call; the run must too.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(PanicTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "panic_tool", "{}"),
            turn::text("recovered"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        let tool_result = &session.messages[2];
        assert_eq!(tool_result.role, Role::User);
        let ContentBlock::ToolResult {
            content, is_error, ..
        } = &tool_result.content[0]
        else {
            panic!("expected a ToolResult block, got {:?}", tool_result.content);
        };
        assert!(*is_error);
        assert!(
            content.contains("panicked"),
            "the tool_result must explain the call panicked, got: {content}"
        );
        // The run itself must have completed normally afterward — proof the panic didn't unwind past
        // the dispatch boundary.
        assert_eq!(session.messages.last().unwrap().role, Role::Assistant);
        assert!(!session.messages.last().unwrap().content.is_empty());
    }

    struct PanicsOnDeny;
    #[async_trait]
    impl AgentHooks for PanicsOnDeny {
        async fn before_tool_call(
            &self,
            name: &str,
            _input: &Value,
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> Option<String> {
            if name == "echo" {
                panic!("boom: before_tool_call always panics");
            }
            None
        }
    }

    #[tokio::test]
    async fn a_panicking_before_tool_call_hook_fails_closed_instead_of_killing_the_run() {
        // A permission hook is exactly the code most likely to have a bug (it's the newest, most
        // custom code in the loop) — its panic must block the one call, fail-closed, not crash the run.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "echo", r#"{"text":"hi"}"#),
            turn::text("recovered"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_hooks(Arc::new(PanicsOnDeny));
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        let tool_result = &session.messages[2];
        let ContentBlock::ToolResult {
            content, is_error, ..
        } = &tool_result.content[0]
        else {
            panic!("expected a ToolResult block, got {:?}", tool_result.content);
        };
        assert!(*is_error, "a panicking permission hook must fail closed");
        assert!(content.contains("panicked"), "got: {content}");
    }

    struct ScreenshotTool;
    #[async_trait]
    impl Tool for ScreenshotTool {
        fn name(&self) -> &str {
            "screenshot"
        }
        fn description(&self) -> &str {
            "returns a fake screenshot image"
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn run(
            &self,
            _input: Value,
        ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
            Ok(crate::tool::ToolOutput::image(
                "here's the screenshot",
                ImageSource {
                    kind: "base64".into(),
                    media_type: "image/png".into(),
                    data: "fake-sensitive-pixels".into(),
                },
            ))
        }
    }

    struct StripsImages;
    #[async_trait]
    impl AgentHooks for StripsImages {
        async fn after_tool_call(
            &self,
            _name: &str,
            _input: &Value,
            output: String,
            _images: Vec<ImageSource>,
            is_error: bool,
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> (String, Vec<ImageSource>, bool) {
            // Redact the image entirely, same shape a real hook (screenshot-blocking policy, a
            // secrets-in-images scanner) would use — proves images actually reach the hook now,
            // instead of being structurally invisible to it.
            (
                format!("{output} [image redacted by policy]"),
                Vec::new(),
                is_error,
            )
        }
    }

    #[tokio::test]
    async fn after_tool_call_hook_can_redact_an_image_the_tool_returned() {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(ScreenshotTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "screenshot", "{}"),
            turn::text("done"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_hooks(Arc::new(StripsImages));
        let mut session = Session::new();
        session.user("take a screenshot");

        agent.run(&mut session, |_| {}).await.unwrap();

        let tool_result = &session.messages[2];
        let ContentBlock::ToolResult {
            content, images, ..
        } = &tool_result.content[0]
        else {
            panic!("expected a ToolResult block, got {:?}", tool_result.content);
        };
        assert!(
            content.contains("[image redacted by policy]"),
            "got: {content}"
        );
        assert!(
            images.is_empty(),
            "the hook's redacted (empty) images must be what's actually committed: {images:?}"
        );
    }

    struct PanicsOnRewrite;
    #[async_trait]
    impl AgentHooks for PanicsOnRewrite {
        async fn after_tool_call(
            &self,
            _name: &str,
            _input: &Value,
            _output: String,
            _images: Vec<ImageSource>,
            _is_error: bool,
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> (String, Vec<ImageSource>, bool) {
            panic!("boom: after_tool_call always panics");
        }
    }

    #[tokio::test]
    async fn a_panicking_after_tool_call_hook_keeps_the_tools_own_result() {
        // Losing a real, already-obtained tool result to a broken *rewrite* attempt would be strictly
        // worse than just ignoring the rewrite — the tool's own (text, is_error) must survive.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "echo", r#"{"text":"hi"}"#),
            turn::text("recovered"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_hooks(Arc::new(PanicsOnRewrite));
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        let tool_result = &session.messages[2];
        let ContentBlock::ToolResult {
            content, is_error, ..
        } = &tool_result.content[0]
        else {
            panic!("expected a ToolResult block, got {:?}", tool_result.content);
        };
        assert!(
            !*is_error,
            "the tool's own success must survive the panicking rewrite hook"
        );
        assert_eq!(content, "hi", "got: {content}");
    }

    #[tokio::test]
    async fn a_mid_run_model_switch_applies_to_the_next_turn_not_the_current_one() {
        use std::time::Duration;

        // pi-parity gap: pi's `prepareNextTurn` lets a host swap the model/thinking level mid-run,
        // without stopping and restarting the whole call. `Steering::request_model_switch` is this
        // crate's equivalent — requested *during* the first tool call (concurrently, via a genuinely
        // separate task, not pre-queued before the run even starts), it must not affect the request
        // already in flight, only the next turn's.
        struct SleepyTool;
        #[async_trait]
        impl Tool for SleepyTool {
            fn name(&self) -> &str {
                "sleepy"
            }
            fn description(&self) -> &str {
                "sleeps briefly"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                tokio::time::sleep(Duration::from_millis(80)).await;
                Ok("slept".into())
            }
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SleepyTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t1", "sleepy", "{}"),
            turn::text("done"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("go");
        let steering = Steering::new();

        let switch_steering = steering.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            switch_steering.request_model_switch("claude-cheap", Some(256));
        });

        let mut events = Vec::new();
        agent
            .run_events_steered(
                &mut session,
                |ev| events.push(ev),
                CancellationToken::new(),
                steering,
            )
            .await
            .unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].model, "claude-opus-4-8",
            "the turn already in flight when the switch was requested must be unaffected"
        );
        assert_eq!(
            requests[1].model, "claude-cheap",
            "the next turn must target the switched-to model"
        );
        assert_eq!(
            requests[1].thinking.map(|t| t.budget_tokens),
            Some(256),
            "the next turn must also carry the switched-to thinking budget"
        );
        assert!(
            events.iter().any(|ev| matches!(
                ev,
                AgentEvent::ModelSwitched { model, thinking }
                    if model == "claude-cheap" && *thinking == Some(256)
            )),
            "a ModelSwitched event must fire so a client can observe the change: {events:#?}"
        );
    }

    #[tokio::test]
    async fn a_mid_run_tool_set_switch_applies_to_the_next_turn_not_the_current_one() {
        use std::time::Duration;

        // Task #13 (pi-parity): pi's real shipped product lets a host reconfigure a run's tool set
        // mid-flight via `setActiveToolsByName` (`packages/coding-agent/src/core/agent-session.ts:840`,
        // wired to the extension runtime's `setActiveTools` handler at line 2283), taking effect
        // starting the very next turn — this crate's `Steering::request_tool_set` is the equivalent,
        // applied at the same turn boundary a model switch already is. Requested *during* the first
        // tool call (concurrently), it must not affect the request already in flight, only the next
        // one's.
        struct SwitchableSleepyTool;
        #[async_trait]
        impl Tool for SwitchableSleepyTool {
            fn name(&self) -> &str {
                "sleepy"
            }
            fn description(&self) -> &str {
                "sleeps briefly"
            }
            fn input_schema(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: Value,
            ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
                tokio::time::sleep(Duration::from_millis(80)).await;
                Ok("slept".into())
            }
        }
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SwitchableSleepyTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t1", "sleepy", "{}"),
            turn::text("done"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("go");
        let steering = Steering::new();

        let switch_steering = steering.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let mut next_tools = ToolRegistry::new();
            next_tools.register(Arc::new(EchoTool));
            switch_steering.request_tool_set(next_tools);
        });

        let mut events = Vec::new();
        agent
            .run_events_steered(
                &mut session,
                |ev| events.push(ev),
                CancellationToken::new(),
                steering,
            )
            .await
            .unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0]
                .tools
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>(),
            vec!["sleepy"],
            "the turn already in flight when the switch was requested must be unaffected"
        );
        assert_eq!(
            requests[1]
                .tools
                .iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>(),
            vec!["echo"],
            "the next turn must advertise the switched-to tool set"
        );
        assert!(
            events.iter().any(|ev| matches!(
                ev,
                AgentEvent::ToolsUpdated { tool_names }
                    if tool_names == &vec!["echo".to_string()]
            )),
            "a ToolsUpdated event must fire so a client can observe the change: {events:#?}"
        );
    }

    struct SleepyTool;
    #[async_trait]
    impl Tool for SleepyTool {
        fn name(&self) -> &str {
            "sleepy"
        }
        fn description(&self) -> &str {
            "sleeps briefly"
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        async fn run(
            &self,
            _: Value,
        ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            Ok("slept".into())
        }
    }

    #[tokio::test]
    async fn a_mid_run_model_switch_can_change_reasoning_effort_via_thinking_level() {
        use std::time::Duration;

        // pi-parity gap: `ModelSwitch` used to have no way to change reasoning effort/thinking depth
        // at all — only a raw thinking-*budget* override. `request_model_switch_with_thinking_level`
        // reuses the same `models::thinking_for_level` translation `serve.rs`'s own
        // `set_model`/`cycle_model` RPC handlers already call for an idle-time switch, so a mid-run
        // one computes an identical `(thinking_budget, reasoning_effort)` pair.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SleepyTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t1", "sleepy", "{}"),
            turn::text("done"),
        ]));
        // claude-opus-4-8 is an adaptive-thinking model: a level translates into *both* a thinking
        // budget and a reasoning effort.
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("go");
        let steering = Steering::new();

        let switch_steering = steering.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            switch_steering.request_model_switch_with_thinking_level(
                "claude-opus-4-8",
                None,
                Some(crate::models::ThinkingLevel::High),
            );
        });

        agent
            .run_events_steered(&mut session, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 2);
        let caps = crate::models::capabilities("claude-opus-4-8");
        let (expected_thinking, expected_effort) =
            crate::models::thinking_for_level(&caps, crate::models::ThinkingLevel::High);
        assert_eq!(
            requests[0].reasoning_effort, None,
            "the turn already in flight when the switch was requested must be unaffected"
        );
        assert_eq!(
            requests[1].thinking.map(|t| t.budget_tokens),
            expected_thinking,
            "the next turn's thinking budget must match the switched-to level's translation"
        );
        assert_eq!(
            requests[1].reasoning_effort, expected_effort,
            "the next turn's reasoning effort must match the switched-to level's translation"
        );
    }

    #[tokio::test]
    async fn a_mid_run_model_switch_can_explicitly_turn_off_thinking() {
        use std::time::Duration;

        // `Some(ThinkingLevel::Off)` must **explicitly** disable thinking/reasoning — distinct from
        // `None`, which leaves it exactly as currently configured. Starts the run with a real
        // thinking budget/reasoning effort already active so "off" is an observable change, not a
        // no-op.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(SleepyTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("t1", "sleepy", "{}"),
            turn::text("done"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_thinking(4096)
            .with_reasoning_effort(ReasoningEffort::High);
        let mut session = Session::new();
        session.user("go");
        let steering = Steering::new();

        let switch_steering = steering.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            switch_steering.request_model_switch_with_thinking_level(
                "claude-opus-4-8",
                None,
                Some(crate::models::ThinkingLevel::Off),
            );
        });

        agent
            .run_events_steered(&mut session, |_| {}, CancellationToken::new(), steering)
            .await
            .unwrap();

        let requests = mock.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].thinking.as_ref().map(|t| t.budget_tokens),
            Some(4096),
            "the first turn must still carry the originally configured thinking budget"
        );
        assert_eq!(requests[0].reasoning_effort, Some(ReasoningEffort::High));
        assert!(
            requests[1].thinking.is_none(),
            "an explicit Off switch must clear the thinking budget on the next turn"
        );
        assert!(
            requests[1].reasoning_effort.is_none(),
            "an explicit Off switch must clear reasoning effort on the next turn"
        );
    }

    struct StopOnMarkerText(&'static str);
    #[async_trait]
    impl AgentHooks for StopOnMarkerText {
        async fn should_stop_after_turn(
            &self,
            message: &Message,
            _tool_results: &[ContentBlock],
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> bool {
            message
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { text, .. } if text.contains(self.0)))
        }
    }

    #[tokio::test]
    async fn should_stop_after_turn_hook_ends_a_tool_less_run_early() {
        // pi-parity gap: `Steering::request_stop` is a bare external flag with no access to what the
        // turn actually said — a host wanting a content-aware stop decision had no seam. Scripts two
        // text-only turns; a hook that stops the moment it sees a marker in the *first* must prevent
        // the second from ever being requested.
        let (agent, mock) = agent_with(
            vec![
                turn::text("here is the STOP-MARKER in my answer"),
                turn::text("this second turn must never be requested"),
            ],
            ToolRegistry::new(),
        );
        let agent = agent.with_hooks(Arc::new(StopOnMarkerText("STOP-MARKER")));
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(
            mock.calls(),
            1,
            "the hook must end the run after the first turn, before a second is ever requested"
        );
    }

    struct StopOnSuccessfulToolResult {
        saw_result: std::sync::atomic::AtomicBool,
    }
    #[async_trait]
    impl AgentHooks for StopOnSuccessfulToolResult {
        async fn should_stop_after_turn(
            &self,
            _message: &Message,
            tool_results: &[ContentBlock],
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> bool {
            let saw_it = tool_results.iter().any(
                |b| matches!(b, ContentBlock::ToolResult { content, is_error, .. } if content == "pong" && !is_error),
            );
            if saw_it {
                self.saw_result
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            saw_it
        }
    }

    #[tokio::test]
    async fn should_stop_after_turn_hook_sees_the_actual_tool_result_content() {
        // The other reachable turn boundary (after a tool round-trip, not just a tool-less turn) — pi's
        // `shouldStopAfterTurn` receives the same `toolResults` at exactly this point in its own loop.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "echo", r#"{"text":"pong"}"#),
            turn::text("this second turn must never be requested"),
        ]));
        let hooks = Arc::new(StopOnSuccessfulToolResult {
            saw_result: std::sync::atomic::AtomicBool::new(false),
        });
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_hooks(hooks.clone());
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        assert!(
            hooks.saw_result.load(std::sync::atomic::Ordering::SeqCst),
            "the hook must have observed the real tool_result content"
        );
        assert_eq!(
            mock.calls(),
            1,
            "the hook must end the run right after the tool round-trip, before a second turn"
        );
    }

    struct PanicsOnStopDecision;
    #[async_trait]
    impl AgentHooks for PanicsOnStopDecision {
        async fn should_stop_after_turn(
            &self,
            _message: &Message,
            _tool_results: &[ContentBlock],
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> bool {
            panic!("boom: should_stop_after_turn always panics");
        }
    }

    #[tokio::test]
    async fn a_panicking_should_stop_after_turn_hook_fails_open_instead_of_killing_the_run() {
        // Unlike a panicking permission hook (fails closed — blocks the call), a panicking *stop*
        // decision fails open: this isn't a security boundary, and halting an otherwise-healthy run
        // because of a buggy hook would be more disruptive than just continuing it. A tool-calling first
        // turn (unlike a tool-less one) only continues to a second turn if nothing requested a stop —
        // proving the panic neither crashed the run nor was misread as "yes, stop".
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "echo", r#"{"text":"hi"}"#),
            turn::text("second turn must still be reached"),
        ]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8)
            .with_hooks(Arc::new(PanicsOnStopDecision));
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        assert_eq!(
            mock.calls(),
            2,
            "the panic must fail open — the run must continue to the second turn, not crash or stop"
        );
    }

    struct RedactSecretsFromAssistant;
    #[async_trait]
    impl AgentHooks for RedactSecretsFromAssistant {
        async fn on_assistant_message(
            &self,
            mut message: Message,
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> Message {
            for block in &mut message.content {
                if let ContentBlock::Text { text, .. } = block {
                    *text = text.replace("secret-token-123", "[REDACTED]");
                }
            }
            message
        }
    }

    #[tokio::test]
    async fn on_assistant_message_hook_rewrites_what_actually_gets_committed_to_the_session() {
        // Pi-parity fix: no hook fired with the model's own generated content before it was committed —
        // only `before_tool_call`/`after_tool_call` existed. Confirms the rewritten text (not the raw
        // model output) is what actually lands in `session.messages`, which is what a checkpoint would
        // persist and what a later turn's prompt would include.
        let mock = Arc::new(MockTransport::new(vec![turn::text(
            "here is the secret-token-123 you asked about",
        )]));
        let agent =
            Agent::new(mock, "claude-opus-4-8").with_hooks(Arc::new(RedactSecretsFromAssistant));
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        let ContentBlock::Text { text, .. } = &session.messages.last().unwrap().content[0] else {
            panic!("expected a text block");
        };
        assert_eq!(text, "here is the [REDACTED] you asked about");
        assert!(
            !text.contains("secret-token-123"),
            "the raw secret must not survive into the committed session: {text}"
        );
    }

    struct PanicsOnAssistantMessage;
    #[async_trait]
    impl AgentHooks for PanicsOnAssistantMessage {
        async fn on_assistant_message(
            &self,
            _message: Message,
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> Message {
            panic!("boom: on_assistant_message always panics");
        }
    }

    #[tokio::test]
    async fn a_panicking_on_assistant_message_hook_keeps_the_original_message_not_a_crashed_run() {
        let mock = Arc::new(MockTransport::new(vec![turn::text("the real answer")]));
        let agent =
            Agent::new(mock, "claude-opus-4-8").with_hooks(Arc::new(PanicsOnAssistantMessage));
        let mut session = Session::new();
        session.user("go");

        agent
            .run(&mut session, |_| {})
            .await
            .expect("a panicking redaction hook must not crash the run");

        let ContentBlock::Text { text, .. } = &session.messages.last().unwrap().content[0] else {
            panic!("expected a text block");
        };
        assert_eq!(
            text, "the real answer",
            "a panicking hook must fail open to the original, unredacted content"
        );
    }

    struct ReturnsWrongRole;
    #[async_trait]
    impl AgentHooks for ReturnsWrongRole {
        async fn on_assistant_message(
            &self,
            _message: Message,
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> Message {
            // A misbehaving hook — returns a `User`-role message instead of preserving `Assistant`.
            Message::user("a hook bug should never let this land in the transcript")
        }
    }

    #[tokio::test]
    async fn a_role_mismatched_on_assistant_message_replacement_is_discarded() {
        // Matches pi's own "the replacement must keep the original message role" contract: a
        // misbehaving hook can't splice a wrong-role message into the transcript.
        let mock = Arc::new(MockTransport::new(vec![turn::text("the real answer")]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_hooks(Arc::new(ReturnsWrongRole));
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        let last = session.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        let ContentBlock::Text { text, .. } = &last.content[0] else {
            panic!("expected a text block");
        };
        assert_eq!(text, "the real answer");
    }

    struct InjectsHeaderNote;
    #[async_trait]
    impl AgentHooks for InjectsHeaderNote {
        async fn before_provider_request(&self, req: &mut crate::transport::ModelRequest) {
            req.system = Some(format!(
                "{} [patched]",
                req.system.clone().unwrap_or_default()
            ));
        }
    }

    #[tokio::test]
    async fn before_provider_request_hook_actually_runs_against_the_real_request() {
        // `hooks.rs`'s own unit test only proves the hook trait's default no-op/override mechanics in
        // isolation — this proves it's actually wired into the live loop (`run_turn_once`), by
        // inspecting what `MockTransport` really received.
        let mock = Arc::new(MockTransport::new(vec![turn::text("ok")]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_system("base prompt")
            .with_hooks(Arc::new(InjectsHeaderNote));
        let mut session = Session::new();
        session.user("go");

        agent.run(&mut session, |_| {}).await.unwrap();

        let sent = mock.requests();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].system.as_deref(), Some("base prompt [patched]"));
    }

    struct PanicsOnBeforeProviderRequest;
    #[async_trait]
    impl AgentHooks for PanicsOnBeforeProviderRequest {
        async fn before_provider_request(&self, _req: &mut crate::transport::ModelRequest) {
            panic!("boom: before_provider_request always panics");
        }
    }

    #[tokio::test]
    async fn a_panicking_before_provider_request_hook_keeps_the_original_request() {
        let mock = Arc::new(MockTransport::new(vec![turn::text("ok")]));
        let agent = Agent::new(mock.clone(), "claude-opus-4-8")
            .with_system("base prompt")
            .with_hooks(Arc::new(PanicsOnBeforeProviderRequest));
        let mut session = Session::new();
        session.user("go");

        agent
            .run(&mut session, |_| {})
            .await
            .expect("a panicking hook must not crash the run");

        let sent = mock.requests();
        assert_eq!(sent.len(), 1);
        assert_eq!(
            sent[0].system.as_deref(),
            Some("base prompt"),
            "a panicking hook must fail open to the request exactly as it was"
        );
    }

    #[tokio::test]
    async fn a_panicking_sink_does_not_unwind_out_of_the_run_events_loop() {
        // The event `sink` was the one interception point in this file still called bare: every hook
        // above it (`before_tool_call`, `after_tool_call`, `should_stop_after_turn`,
        // `on_assistant_message`, `before_provider_request`) already fails open via `catch_tool_panic`.
        // A caller's own render/log callback panicking on one event (a bad match arm, an unwrap on an
        // unexpected variant) must degrade to a dropped event, not take the whole run down with it.
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(EchoTool));
        let mock = Arc::new(MockTransport::new(vec![
            turn::tool_call("tu_1", "echo", r#"{"text":"hi"}"#),
            turn::text("recovered"),
        ]));
        let agent = Agent::new(mock, "claude-opus-4-8")
            .with_tools(tools)
            .with_max_steps(8);
        let mut session = Session::new();
        session.user("go");

        let mut agent_end_seen = false;
        agent
            .run_events(&mut session, |ev| {
                if let AgentEvent::ToolStart { .. } = &ev {
                    panic!("boom: sink always panics on ToolStart");
                }
                if let AgentEvent::AgentEnd { .. } = &ev {
                    agent_end_seen = true;
                }
            })
            .await
            .expect("a panicking sink must not crash the run");

        assert!(
            agent_end_seen,
            "events after the panicking one must still reach the sink"
        );
        // The run itself completed normally — the tool call still ran and the model still got to
        // reply — proof the panic never unwound past the sink boundary.
        let last = session.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert!(!last.content.is_empty());
    }

    #[tokio::test]
    async fn a_panicking_sink_does_not_unwind_out_of_a_direct_compact_call() {
        // `compact` is a public entry point reachable independent of `run_events_steered` — the manual
        // `compact` RPC command (`crates/agent/src/serve.rs`) calls it directly with its own raw sink
        // closure, so it needs its own copy of the fails-open guard, not just borrowed protection from
        // the main loop's own wrap.
        let session_messages = vec![
            Message::user("look at this"),
            Message::assistant(vec![ContentBlock::text("ok, looking")]),
            Message::user("now something else"),
            Message::assistant(vec![ContentBlock::text("done")]),
        ];
        let mut session = Session::new();
        session.messages = Arc::new(session_messages);

        let mock = Arc::new(MockTransport::new(vec![turn::text("a real summary")]));
        let agent = Agent::new(mock, "claude-opus-4-8").with_compaction(CompactionConfig {
            keep_recent_tokens: 1,
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();

        let mut compacted_event_seen = false;
        let compacted = agent
            .compact(
                &mut session,
                CompactionReason::Manual,
                &cancel,
                &mut |ev| {
                    if let AgentEvent::CompactionStart { .. } = &ev {
                        panic!("boom: sink always panics on CompactionStart");
                    }
                    if let AgentEvent::Compacted { .. } = &ev {
                        compacted_event_seen = true;
                    }
                },
                None,
            )
            .await
            .expect("a panicking sink must not crash a direct compact() call");

        assert!(compacted.compacted());
        assert!(
            compacted_event_seen,
            "the Compacted event after the panicking CompactionStart must still reach the sink"
        );
    }
}
