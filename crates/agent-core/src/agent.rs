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
    ContentBlock, ImageSource, Message, StopReason, StreamEvent, TokenUsage, ToolDef,
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
    /// The conversation prefix was summarized to stay under the context window.
    Compacted {
        messages_before: usize,
        messages_after: usize,
        /// Why this compaction fired — the full folded-forward provenance (file-ops, round count)
        /// lands on `Session::compaction`, not duplicated onto every event.
        reason: CompactionReason,
        /// Estimated input tokens at the moment this compaction fired (before the reset).
        tokens_before: u32,
    },
    /// The run is ending abnormally (transport failure after retries, malformed SSE, or the step
    /// ceiling). A terminal marker on the event stream so a streaming client sees *why* a run stopped
    /// rather than the stream just going silent; `run_events` still returns the same `Err`.
    Error { message: String },
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

/// A configured agent: a model, a transport, a tool set, and loop bounds. Cheap to clone-construct;
/// `run` borrows it so one agent can drive many sessions.
pub struct Agent {
    transport: Arc<dyn ModelTransport>,
    tools: ToolRegistry,
    /// The advertised tool definitions, computed once from `tools`. The set is fixed for the agent's
    /// life, so we build it (and its JSON schemas) at configuration time rather than rebuilding it on
    /// every turn; each request clones the `Arc`, not the definitions.
    tool_defs: Arc<[ToolDef]>,
    model: String,
    system: Option<String>,
    max_tokens: u32,
    max_steps: u32,
    /// Extended-thinking budget, when enabled. Applied to every turn's request.
    thinking: Option<u32>,
    /// Reasoning effort level (OpenAI reasoning models; Anthropic adaptive thinking). Applied to every
    /// turn's request when set.
    reasoning_effort: Option<ReasoningEffort>,
    /// Context-compaction policy: when to summarize the prefix to stay under the context window.
    compaction: CompactionConfig,
    /// Whether [`Self::run_turn`] retries a mid-stream transport failure (see
    /// [`is_retryable_mid_stream`]) instead of surfacing it immediately. Defaults to `true`; an
    /// operator debugging a flaky network hop can disable it via `with_auto_retry(false)` to see the
    /// raw failure on the very first hiccup rather than after `MAX_MID_STREAM_RETRIES` silent attempts.
    auto_retry: bool,
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
        let reserve_tokens = CompactionConfig::default().reserve_tokens;
        let summary_max_tokens = scaled_summary_max_tokens(reserve_tokens, caps.max_output);
        Self {
            transport,
            tools: ToolRegistry::new(),
            tool_defs: Vec::new().into(),
            model,
            system: None,
            max_tokens: caps.max_output.max(DEFAULT_MAX_TOKENS),
            max_steps: DEFAULT_MAX_STEPS,
            thinking: None,
            reasoning_effort: None,
            compaction: CompactionConfig {
                context_window: caps.context_window,
                summary_max_tokens,
                ..CompactionConfig::default()
            },
            auto_retry: true,
            hooks: Arc::new(NoHooks),
            cache_key: None,
            cache_long: false,
            write_locks: Arc::new(crate::write_lock::WriteLockRegistry::new()),
            checkpoint: Arc::new(NoCheckpoint),
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
    pub fn set_system(&mut self, system: impl Into<String>) {
        self.system = Some(system.into());
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

    /// Enable or disable mid-stream retry (default: enabled) — see the `auto_retry` field's doc comment.
    pub fn with_auto_retry(mut self, enabled: bool) -> Self {
        self.auto_retry = enabled;
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

    /// Drive the loop to completion against `session`, invoking `on_event` for every streamed event
    /// (use it to render assistant text/tool activity live). Returns when the model ends its turn
    /// without requesting tools, or errors with [`Error::MaxSteps`] if it never does.
    pub async fn run<F>(&self, session: &mut Session, mut on_event: F) -> Result<()>
    where
        F: FnMut(&StreamEvent),
    {
        self.run_events(session, move |ev| {
            if let AgentEvent::Stream(s) = &ev {
                on_event(s);
            }
        })
        .await
    }

    /// Drive the loop to completion, emitting an [`AgentEvent`] for every streamed model event, tool
    /// invocation, and turn boundary — the full observation surface the headless server streams to
    /// its clients. Returns when the model ends its turn without tools, or [`Error::MaxSteps`].
    pub async fn run_events<F>(&self, session: &mut Session, sink: F) -> Result<()>
    where
        F: FnMut(AgentEvent),
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
        F: FnMut(AgentEvent),
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
        F: FnMut(AgentEvent),
    {
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
        // Set once we've already compacted to recover from a context-overflow this turn, so a second
        // overflow gives up instead of looping. Reset after each turn that lands cleanly.
        let mut overflow_recovered = false;
        // Steps taken *by this call*, distinct from `session.steps` (a lifetime total across every
        // call, used only for observability — the `step` field on emitted events). Checking the
        // ceiling against this instead means `Error::MaxSteps` is a per-call backstop a client can
        // resume past by simply calling `run`/`run_events_steered` again with a fresh budget, rather
        // than a permanent dead end once the session's lifetime total crosses it once.
        let mut steps_this_call: u32 = 0;
        // The previous turn's stop reason, read by `is_hard_overflow`'s `MaxTokens` check — `EndTurn`
        // (a value that check never matches) until the first turn actually completes.
        let mut last_stop_reason = StopReason::EndTurn;
        loop {
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if steps_this_call >= self.max_steps {
                let err = Error::MaxSteps(self.max_steps);
                sink(AgentEvent::Error {
                    message: err.to_string(),
                });
                return Err(err);
            }

            // Proactive compaction: once the live prompt crosses the threshold, summarize the prefix
            // before building the next request so the run never walks into the context wall.
            if self.compaction.enabled && compaction::should_compact(session, &self.compaction) {
                self.compact(session, CompactionReason::Threshold, &cancel, &mut sink)
                    .await?;
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
                self.compact(session, CompactionReason::Overflow, &cancel, &mut sink)
                    .await?;
            }

            sink(AgentEvent::TurnStart {
                step: session.steps + 1,
            });

            let mut req = ModelRequest::new(
                self.model.clone(),
                session.messages.clone(),
                self.max_tokens,
            )
            .with_tools(self.tool_defs.clone())
            .with_cache_long(self.cache_long);
            if let Some(system) = &self.system {
                req = req.with_system(system.clone());
            }
            if let Some(budget) = self.thinking {
                req = req.with_thinking(budget);
            }
            if let Some(effort) = self.reasoning_effort {
                req = req.with_reasoning_effort(effort);
            }
            if let Some(key) = &self.cache_key {
                req = req.with_cache_key(key.clone());
            }

            // `emit` borrows `sink` for the turn; bind the result, then drop the borrow before handling
            // an error so the terminal `Error` event can go out through `sink`.
            let turn_result = {
                let mut emit = |ev: StreamEvent| sink(AgentEvent::Stream(ev));
                self.run_turn(req, &mut emit, &cancel).await
            };
            let mut turn = match turn_result {
                Ok(turn) => turn,
                // A cancellation is a user request, not a fault — return it without an `Error` event.
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                // The provider rejected the request for exceeding its context window. Compact once and
                // retry the same turn; if it still overflows (or there's nothing to compact), give up.
                Err(e) if is_context_overflow(&e) && !overflow_recovered => {
                    if self
                        .compact(session, CompactionReason::Overflow, &cancel, &mut sink)
                        .await?
                    {
                        overflow_recovered = true;
                        continue;
                    }
                    sink(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    return Err(e);
                }
                Err(e) => {
                    sink(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    return Err(e);
                }
            };
            overflow_recovered = false;
            last_stop_reason = turn.stop_reason;
            let malformed: HashMap<String, String> =
                std::mem::take(&mut turn.malformed).into_iter().collect();
            session.push(Message::assistant(turn.blocks).with_model_id(&self.model));
            session.record_usage(turn.usage);
            session.steps += 1;
            steps_this_call += 1;
            sink(AgentEvent::TurnEnd {
                stop_reason: turn.stop_reason,
                step: session.steps,
            });

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

            if calls.is_empty() || turn.stop_reason != StopReason::ToolUse {
                // A refusal is a distinct terminal condition, not an ordinary stop: draining queued
                // steer/follow-up messages here would inject a new user turn right after the model
                // just declined to engage with the current one, which the model would likely refuse
                // again. End the run immediately instead, leaving the queue untouched — nothing is
                // lost, since it's the same persistent `Steering` handle a later `prompt` call reads
                // from (see `serve.rs`).
                if turn.stop_reason == StopReason::Refusal {
                    sink(AgentEvent::AgentEnd {
                        steps: session.steps,
                    });
                    return Ok(());
                }
                // A pending graceful-stop request wins over draining follow-up/steer messages, exactly
                // as it wins over continuing tool-call turns below — the queue is left untouched (same
                // rationale as the refusal case above) so nothing queued for "next time" is lost.
                if steering.take_stop_requested() {
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
                    session.user(msg);
                }
                sink(AgentEvent::Steered { messages: count });
                // A plain user message ends the visible history here — a valid, resumable checkpoint
                // (see `CheckpointHook`) before the next model call.
                self.checkpoint.checkpoint(session).await;
                continue;
            }

            // Run the tools and feed results back as a single user turn. A tool's own failure becomes
            // an error `tool_result`, not an aborted run — the model can react to it next turn.
            //
            // The calls run concurrently: tools are I/O-bound (file reads, shell commands, the
            // `beyond` CLI), and a model routinely batches independent ones in a single turn, so
            // overlapping them collapses the tool phase from the sum of their latencies to its slowest
            // member. `ToolStart` is emitted up front in call order; `ToolEnd` is emitted live, the
            // instant each call's own result is known — a client watching the event stream sees
            // completions in actual finish order, not batched after the slowest call joins. The
            // *transcript* (the `tool_result` blocks pushed to the session below) still stays
            // deterministic regardless of finish order, rebuilt in call order after the join.
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
                self.tools
                    .get(name)
                    .is_some_and(|t| t.conservative_exclusive())
            });
            let mut groups: HashMap<String, (Option<String>, Vec<usize>)> = HashMap::new();
            for (i, (_, name, input)) in calls.iter().enumerate() {
                let target = self.tools.get(name).and_then(|t| t.write_target(input));
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
            // Per-turn progress channel: every call gets a `ToolProgress` cloning `prog_tx`; the drain
            // loop below forwards each update to `sink` as it arrives. `futures`' mpsc keeps this
            // executor-agnostic (no tokio in the library).
            let (prog_tx, mut prog_rx) =
                futures::channel::mpsc::unbounded::<crate::tool::ToolUpdate>();
            let prog_tx = &prog_tx;
            let cancel_ref = &cancel;
            let group_runs = groups.into_values().map(|(target, indices)| {
                let calls = &calls;
                let prog_tx = prog_tx.clone();
                let cancel = cancel_ref.clone();
                let write_locks = this.write_locks.clone();
                async move {
                    // Held for the group's whole serial run: extends the intra-turn grouping above
                    // across turn and session boundaries, so a concurrently-running turn (or a
                    // different session sharing this `Agent`'s registry) touching the same path really
                    // waits, not just calls within this one turn.
                    let _write_guard = match &target {
                        Some(path) => Some(write_locks.lock(path).await),
                        None => None,
                    };
                    let mut out = Vec::with_capacity(indices.len());
                    for i in indices {
                        let (id, name, input) = &calls[i];
                        // Per call: (text, images, is_error, terminate). Hooks rewrite the *text* and
                        // error flag; images and the terminate hint pass through untouched.
                        let result: ToolCallResult =
                            if let Some(raw) = malformed.get(id) {
                                // The model streamed a tool call whose argument fragments never formed
                                // valid JSON. Feed that back as an error result the model can correct
                                // next turn rather than aborting the whole run on one malformed call.
                                (
                                    format!(
                                        "tool call arguments were not valid JSON and could not be parsed: {raw}"
                                    ),
                                    Vec::new(),
                                    true,
                                    false,
                                )
                            } else if let Some(reason) =
                                this.hooks.before_tool_call(name, input, &cancel).await
                            {
                                // A hook blocked the call (e.g. a permission policy). Feed the reason
                                // back as an error result instead of running the tool.
                                (format!("tool call blocked: {reason}"), Vec::new(), true, false)
                            } else {
                                let progress = crate::tool::ToolProgress::new(
                                    prog_tx.clone(),
                                    id.clone(),
                                    name.clone(),
                                    cancel.clone(),
                                );
                                let (text, images, is_error, terminate) =
                                    match this.tools.get(name) {
                                        Some(tool) => {
                                            match tool.run_streaming(input.clone(), &progress).await {
                                                Ok(o) => (o.text, o.images, false, o.terminate),
                                                Err(e) => (e.to_string(), Vec::new(), true, false),
                                            }
                                        }
                                        None => {
                                            (format!("unknown tool: {name}"), Vec::new(), true, false)
                                        }
                                    };
                                // Let a hook rewrite the result text (redact, cap, reclassify) before
                                // it's fed back to the model.
                                let (text, is_error) = this
                                    .hooks
                                    .after_tool_call(name, input, text, is_error, &cancel)
                                    .await;
                                (text, images, is_error, terminate)
                            };
                        // Sent the instant this call's own result is known — not batched until every
                        // group in the turn finishes — so a client watching the event stream sees each
                        // tool's completion as it actually happens, not all-at-once after the slowest
                        // concurrently-dispatched call joins.
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
            // `None` until that call's group finishes; a slot left `None` means cancellation aborted
            // dispatch before it ran, and `repair_cancelled_dispatch` needs to tell that apart from a
            // real (possibly empty) result to synthesize a matching error `tool_result` for it.
            let mut results: Vec<Option<ToolCallResult>> = vec![None; calls.len()];
            // Bound how many groups run at once. `buffer_unordered` is safe here because each group
            // yields its results tagged with their original call index `i`; cross-group completion
            // order never reaches the transcript, which is rebuilt in call order below. `exclusive_turn`
            // caps this at 1 instead — with only ever one group in flight, a `bash` call (or anything
            // else `conservative_exclusive`) can't race a same-turn `edit`/`write` group it has no path
            // to be grouped against; which group runs first still doesn't matter for the transcript,
            // same as the concurrent case.
            //
            // Race the whole dispatch against cancellation: a tripped token drops `drain`, which drops
            // every in-flight tool future — aborting a hung `bash` (its `kill_on_drop` child dies) and
            // any other long-running tool — and returns promptly instead of waiting them all out. The
            // block scopes `drain`'s `&mut results` borrow so the transcript below can consume them;
            // `cancelled_mid_dispatch` is only *acted on* after the block ends and that borrow is
            // fully released (repairing the transcript needs to move `results` out).
            let mut cancelled_mid_dispatch = false;
            {
                let concurrency = if exclusive_turn {
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
            if cancelled_mid_dispatch {
                // Cancelled mid-dispatch: the assistant message (with its `ToolUse` blocks) is already
                // committed above, but the tool-results message below never will be. Left as-is, the
                // session would end on an orphaned `tool_use` with no matching `tool_result` — a shape
                // both Anthropic and OpenAI reject on resume. Repair it before propagating the
                // cancellation.
                repair_cancelled_dispatch(session, &calls, results);
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
                    content,
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
                result_blocks.push(ContentBlock::Text { text: msg });
            }
            session.push(Message::tool_results(result_blocks));
            // A tool round-trip just landed: assistant `tool_use` and its matching `tool_result`s are
            // both committed now, so this is a valid, resumable checkpoint (see `CheckpointHook`) — the
            // one mid-run point a crash between here and the run's eventual end would otherwise lose.
            self.checkpoint.checkpoint(session).await;
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
            // message land in the transcript even if the run stops right after.
            if steering.take_stop_requested() {
                sink(AgentEvent::AgentEnd {
                    steps: session.steps,
                });
                return Ok(());
            }
        }
    }

    /// Stream and assemble a single model turn, restarting from scratch when the stream dies mid-flight
    /// (see [`is_retryable_mid_stream`]) rather than surfacing that as a fatal error. Each attempt runs
    /// in [`run_turn_once`] with its own fresh [`Accumulator`] — a retried attempt never resumes a
    /// dead attempt's partial blocks, so the `Turn` this returns can't blend a half-formed tool call
    /// from a failed connection into what actually gets applied to the session. A cancellation always
    /// propagates immediately; only the mid-stream-failure class is retried.
    async fn run_turn(
        &self,
        req: ModelRequest,
        emit: &mut dyn FnMut(StreamEvent),
        cancel: &CancellationToken,
    ) -> Result<Turn> {
        let mut attempt = 0u32;
        loop {
            match self.run_turn_once(req.clone(), emit, cancel).await {
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
    async fn run_turn_once(
        &self,
        req: ModelRequest,
        emit: &mut dyn FnMut(StreamEvent),
        cancel: &CancellationToken,
    ) -> Result<Turn> {
        let cancelled = cancel.cancelled();
        futures::pin_mut!(cancelled);
        let mut stream = {
            let stream_fut = self.transport.stream(req);
            futures::pin_mut!(stream_fut);
            match select(stream_fut, cancelled.as_mut()).await {
                Either::Left((res, _)) => res?,
                Either::Right(((), _)) => return Err(Error::Cancelled),
            }
        };
        let mut acc = Accumulator::default();
        loop {
            let next = stream.next();
            futures::pin_mut!(next);
            match select(next, cancelled.as_mut()).await {
                Either::Left((Some(ev), _)) => {
                    let ev = ev?;
                    // `apply` only borrows, so `ev` is still ours to move into `emit` afterward — no
                    // clone needed on this per-delta hot path (see `Accumulator::apply`'s doc comment).
                    acc.apply(&ev);
                    emit(ev);
                }
                Either::Left((None, _)) => break,
                Either::Right(((), _)) => return Err(Error::Cancelled),
            }
        }
        Ok(acc.finish())
    }

    /// Summarize the conversation prefix in place, keeping the recent turns verbatim. Makes one
    /// summarization model call (silently — its tokens aren't surfaced as assistant output), splices
    /// the summary into `session`, folds this round's file-ops into `session.compaction` (see
    /// [`CompactionProvenance`]), and emits an [`AgentEvent::Compacted`]. Returns `false` (a no-op)
    /// when there's no worthwhile prefix to summarize or the model returns an empty summary. Exposed
    /// so a headless server can offer a manual `compact` command (pass [`CompactionReason::Manual`]).
    pub async fn compact(
        &self,
        session: &mut Session,
        reason: CompactionReason,
        cancel: &CancellationToken,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Result<bool> {
        let Some(cut) =
            compaction::find_split_cut(&session.messages, self.compaction.keep_recent_tokens)
        else {
            return Ok(false);
        };
        let first_kept = cut.first_kept;
        let before = session.messages.len();
        let tokens_before = session.last_input_tokens;
        let prefix: Vec<Message> = session.messages[..first_kept].to_vec();
        let file_ops = compaction::merge_file_ops(&session.compaction, &prefix, reason);

        let summary = match cut.turn_start {
            None => {
                // Clean boundary: unchanged, single-call path.
                let req = compaction::summary_request(
                    &self.model,
                    &prefix,
                    self.compaction.summary_max_tokens,
                    &file_ops,
                );
                turn_text(&self.run_turn(req, &mut |_| {}, cancel).await?)
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
                let turn_prefix_max_tokens = ((self.compaction.summary_max_tokens as f64)
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
                    let req = compaction::summary_request(
                        &self.model,
                        &session.messages[..turn_start],
                        self.compaction.summary_max_tokens,
                        &file_ops,
                    );
                    Ok::<_, Error>(turn_text(&self.run_turn(req, &mut |_| {}, cancel).await?))
                };
                let turn_prefix = async {
                    let req = compaction::summary_request(
                        &self.model,
                        &session.messages[turn_start..first_kept],
                        turn_prefix_max_tokens,
                        &file_ops,
                    );
                    Ok::<_, Error>(turn_text(&self.run_turn(req, &mut |_| {}, cancel).await?))
                };
                let history = history.await?;
                let turn_prefix = turn_prefix.await?;
                compaction::merge_split_summary(&history, &turn_prefix)
            }
        };
        if summary.trim().is_empty() {
            return Ok(false);
        }
        compaction::apply_summary(session, first_kept, &summary);
        session.compaction = file_ops;
        sink(AgentEvent::Compacted {
            messages_before: before,
            messages_after: session.messages.len(),
            reason,
            tokens_before,
        });
        Ok(true)
    }

    /// Summarize an abandoned tree branch's messages (Track L2/L3: a headless server calls this from
    /// its branch-navigation command, on messages its session store's `abandoned_by_switch` returned,
    /// *before* actually switching branches). The network-calling half of branch summarization; the
    /// pure request-building lives in [`crate::branch_summary::branch_summary_request`], and
    /// persisting the result is the caller's job — this only returns the summary text, mirroring
    /// [`Self::compact`]'s network/storage split but without touching `session` (a branch summary
    /// doesn't rewrite the *active* conversation the way a compaction summary does).
    pub async fn summarize_branch(
        &self,
        messages: &[Message],
        cancel: &CancellationToken,
    ) -> Result<String> {
        // Same input budget compaction sizes its own summarization calls against — the model's context
        // window minus its reserved headroom — so an abandoned branch's rendered transcript can't
        // overflow the summarization call's own window any more than a compaction summary's can.
        let input_token_budget = self
            .compaction
            .context_window
            .saturating_sub(self.compaction.reserve_tokens);
        let req = crate::branch_summary::branch_summary_request(
            &self.model,
            messages,
            self.compaction.summary_max_tokens,
            input_token_budget,
        );
        let turn = self.run_turn(req, &mut |_| {}, cancel).await?;
        Ok(turn
            .blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect())
    }
}

/// The concatenated text blocks of a summarization turn — a summary is always plain prose, so anything
/// else the model emitted (there shouldn't be any; the summarization system prompt asks for none) is
/// simply not text and is dropped here rather than erroring.
fn turn_text(turn: &Turn) -> String {
    turn.blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Turn one [`ToolUpdate`](crate::tool::ToolUpdate) from the per-turn progress channel into the
/// matching [`AgentEvent`] and emit it — shared by the live-arriving path and the final flush of
/// whatever was still buffered when the group stream finished.
fn emit_tool_update(sink: &mut dyn FnMut(AgentEvent), update: crate::tool::ToolUpdate) {
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

/// Resolve a per-call dispatch result, synthesizing an error placeholder for a call whose group never
/// got to run — only reachable via [`repair_cancelled_dispatch`], when cancellation aborts the batch
/// before every group's future resolved.
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
    // Cerebras
    "400 (no body)",
    "413 (no body)",
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
fn is_context_overflow(e: &Error) -> bool {
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
///   implementation detail this shouldn't couple to) or carries one of
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
    let m = msg.to_ascii_lowercase();
    m.contains("stream ended before")
        || m.contains("overloaded")
        || msg.contains(MID_STREAM_NETWORK_ERROR)
        || MID_STREAM_RETRYABLE_ERROR_TYPES
            .iter()
            .any(|p| m.contains(p))
        || MID_STREAM_RETRY_GUIDANCE_PHRASES
            .iter()
            .any(|p| m.contains(p))
}

/// Exponential backoff for a mid-stream retry: `MID_STREAM_BASE_BACKOFF · 2^(attempt-1)`, capped at
/// `MID_STREAM_MAX_BACKOFF`. `attempt` is 1-based (the first retry backs off by the base amount).
fn mid_stream_backoff(attempt: u32) -> Duration {
    MID_STREAM_BASE_BACKOFF
        .saturating_mul(1u32 << attempt.saturating_sub(1).min(16))
        .min(MID_STREAM_MAX_BACKOFF)
}

/// Best-effort repair for streamed tool-call JSON that fails to parse on the first attempt, fixing two
/// real-world streaming quirks seen from both dialects rather than giving up immediately: a raw control
/// character inside a string literal that should have been escaped (a large `write`/`edit` argument
/// carrying an embedded literal newline/tab instead of `\n`/`\t`), and a backslash that isn't a valid
/// JSON escape (a Windows path like `C:\Users\x` streamed without escaping its own backslashes). Ported
/// from pi's `repairJson` (`packages/ai/src/utils/json-parse.ts`) — a single pass that only touches
/// bytes *inside* a string literal, so well-formed structural JSON (braces, commas, already-valid
/// escapes) passes through unchanged. Not a full JSON5-style parser: it can't repair a buffer that's
/// merely incomplete (cut off mid-stream with an unclosed brace) — that class still falls through to
/// [`Accumulator::flush_block`]'s existing malformed-call recovery.
fn repair_json(s: &str) -> String {
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

/// The assembled result of one model turn.
struct Turn {
    blocks: Vec<ContentBlock>,
    stop_reason: StopReason,
    usage: TokenUsage,
    /// Tool calls whose streamed arguments never parsed as JSON, as `(tool_use_id, raw buffer)`. The
    /// loop turns each into an error `tool_result` the model can correct, rather than aborting the run.
    malformed: Vec<(String, String)>,
}

/// Folds a `StreamEvent` sequence into content blocks. Text accrues into the current text run; a
/// tool call accrues its streamed JSON argument fragments; `ContentBlockStop` finalizes whichever is
/// open. Works identically for both dialects because they emit the same event shape.
#[derive(Default)]
struct Accumulator {
    blocks: Vec<ContentBlock>,
    text: String,
    thinking: Option<(String, String)>, // (text, signature) of an open thinking block
    tool: Option<(String, String, String)>, // (id, name, json-arg buffer)
    stop_reason: StopReason,
    usage: TokenUsage,
    /// Tool calls whose streamed JSON arguments failed to parse, as `(id, raw buffer)`. Surfaced on
    /// the `Turn` so the loop can feed each back as a recoverable error `tool_result`.
    malformed: Vec<(String, String)>,
}

impl Accumulator {
    // Borrows rather than takes `StreamEvent` by value: a streamed turn folds hundreds to thousands
    // of deltas through here, and the caller (`run_turn_once`) also needs the event afterward (to
    // `emit` it) — taking it by value would force a clone on every single delta just so both sides
    // get their own copy. Only the two block-boundary variants (`RedactedThinking`, `ToolUseStart`)
    // need an owned copy of their payload for `self.blocks`/`self.tool`; the four high-frequency delta
    // variants (`TextDelta`/`ThinkingDelta`/`SignatureDelta`/`InputJsonDelta`) just borrow to
    // `push_str`, and `Usage`/`MessageStop` are `Copy`.
    fn apply(&mut self, ev: &StreamEvent) {
        match ev {
            StreamEvent::MessageStart => {}
            StreamEvent::TextDelta { text } => self.text.push_str(text),
            StreamEvent::ThinkingDelta { text } => {
                self.thinking
                    .get_or_insert_with(Default::default)
                    .0
                    .push_str(text);
            }
            StreamEvent::SignatureDelta { signature } => {
                self.thinking
                    .get_or_insert_with(Default::default)
                    .1
                    .push_str(signature);
            }
            StreamEvent::RedactedThinking { data } => {
                // Self-contained block with no deltas — close any open text and emit it directly.
                self.flush_text();
                self.blocks
                    .push(ContentBlock::RedactedThinking { data: data.clone() });
            }
            StreamEvent::ToolUseStart { id, name } => {
                self.flush_text();
                self.tool = Some((id.clone(), name.clone(), String::new()));
            }
            StreamEvent::InputJsonDelta { partial_json } => {
                if let Some((_, _, buf)) = &mut self.tool {
                    buf.push_str(partial_json);
                }
            }
            StreamEvent::ContentBlockStop => self.flush_block(),
            StreamEvent::Usage(usage) => self.usage = *usage,
            StreamEvent::MessageStop { stop_reason } => self.stop_reason = *stop_reason,
        }
    }

    fn flush_text(&mut self) {
        if !self.text.is_empty() {
            self.blocks.push(ContentBlock::Text {
                text: std::mem::take(&mut self.text),
            });
        }
    }

    fn flush_block(&mut self) {
        // At most one block is open at a time (each is delimited by `content_block_stop`): a tool call,
        // a thinking block, or accruing text — flush whichever it is.
        if let Some((id, name, args)) = self.tool.take() {
            let input = if args.trim().is_empty() {
                json!({})
            } else {
                match serde_json::from_str(&args)
                    .or_else(|_| serde_json::from_str(&repair_json(&args)))
                {
                    Ok(v) => v,
                    // Still doesn't parse even after the repair pass — a genuine protocol glitch, not
                    // a tool failure. Keep the tool_use block (with an empty, wire-valid input object
                    // so the next request doesn't 400) and record the call as malformed; the loop
                    // feeds back an error result the model can correct, instead of aborting the run.
                    Err(_) => {
                        self.malformed.push((id.clone(), args));
                        json!({})
                    }
                }
            };
            self.blocks.push(ContentBlock::ToolUse { id, name, input });
        } else if let Some((text, signature)) = self.thinking.take() {
            self.blocks.push(ContentBlock::Thinking { text, signature });
        } else {
            self.flush_text();
        }
    }

    fn finish(mut self) -> Turn {
        // A stream that ended without a trailing ContentBlockStop (or with leftover text) still
        // contributes its text.
        self.flush_block();
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
            vec![ContentBlock::Text {
                text: "hello world".into()
            }]
        );
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
                    id: "tu_1".into(),
                    name: "echo".into(),
                }),
                Ok(StreamEvent::InputJsonDelta {
                    partial_json: "{\"tex".into(),
                }),
                Err(Error::Transport(
                    "Anthropic stream ended before message_stop".into(),
                )),
            ],
            vec![
                Ok(StreamEvent::MessageStart),
                Ok(StreamEvent::ToolUseStart {
                    id: "tu_1".into(),
                    name: "echo".into(),
                }),
                Ok(StreamEvent::InputJsonDelta {
                    partial_json: "{\"text\":\"pong\"}".into(),
                }),
                Ok(StreamEvent::ContentBlockStop),
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
            vec![ContentBlock::ToolUse {
                id: "tu_1".into(),
                name: "echo".into(),
                input: json!({ "text": "pong" }),
            }]
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
    fn mid_stream_backoff_is_exponential_and_capped() {
        assert_eq!(mid_stream_backoff(1), MID_STREAM_BASE_BACKOFF);
        assert_eq!(mid_stream_backoff(2), MID_STREAM_BASE_BACKOFF * 2);
        assert_eq!(mid_stream_backoff(3), MID_STREAM_BASE_BACKOFF * 4);
        assert_eq!(mid_stream_backoff(20), MID_STREAM_MAX_BACKOFF); // saturates, never overflows
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
                id: "s".into(),
                name: "slow".into(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::ToolUseStart {
                id: "f".into(),
                name: "fast".into(),
            },
            StreamEvent::ContentBlockStop,
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
                id: "a".into(),
                name: "t1".into(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::ToolUseStart {
                id: "b".into(),
                name: "t2".into(),
            },
            StreamEvent::ContentBlockStop,
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
                id: "a".into(),
                name: "edit".into(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: r#"{"path":"foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::ToolUseStart {
                id: "b".into(),
                name: "write".into(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: r#"{"path":"foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop,
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
                id: "a".into(),
                name: "edit".into(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: r#"{"path":"foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::ToolUseStart {
                id: "b".into(),
                name: "bash".into(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: r#"{"command":"black foo.rs"}"#.into(),
            },
            StreamEvent::ContentBlockStop,
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
                id: format!("c{i}"),
                name: "count".into(),
            });
            many_calls.push(StreamEvent::ContentBlockStop);
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
                id: "tu_1".into(),
                name: "echo".into(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: r#"{"text":"#.into(), // truncated — not parseable
            },
            StreamEvent::ContentBlockStop,
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

    #[tokio::test]
    async fn a_raw_control_character_in_streamed_tool_args_is_repaired_not_malformed() {
        // Same shape as `malformed_tool_args_become_recoverable_error_result`, but for the class of
        // failure `repair_json` exists to fix: a raw newline byte inside the streamed JSON string
        // (as if a provider's SSE encoder forgot to escape it) rather than genuinely truncated JSON.
        // This must now parse successfully on the repair pass instead of becoming a malformed call.
        let turn = vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                id: "tu_1".into(),
                name: "echo".into(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: "{\"text\":\"line one".into(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: "\nline two\"}".into(), // raw newline, not an escaped `\n`
            },
            StreamEvent::ContentBlockStop,
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
                .any(|e| matches!(e, StreamEvent::TextDelta { text } if text == "stream me"))
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
                    id: id.into(),
                    name: "echo".into(),
                },
                StreamEvent::InputJsonDelta {
                    partial_json: r#"{"text":"a"}"#.into(),
                },
                StreamEvent::ContentBlockStop,
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
            ContentBlock::Text { text } if text.contains("compacted")
        ));
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
                    id: id.into(),
                    name: "echo".into(),
                },
                StreamEvent::InputJsonDelta {
                    partial_json: r#"{"text":"a"}"#.into(),
                },
                StreamEvent::ContentBlockStop,
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
                    id: id.into(),
                    name: "read".into(),
                },
                StreamEvent::InputJsonDelta {
                    partial_json: r#"{"path":"tracked.rs"}"#.into(),
                },
                StreamEvent::ContentBlockStop,
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
            Message::assistant(vec![ContentBlock::Text {
                text: "first done".into(),
            }]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: json!({ "path": "a.rs" }),
            }]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "edit".into(),
                input: json!({ "path": "a.rs" }),
            }]),
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
            .compact(&mut session, CompactionReason::Manual, &cancel, &mut |_| {})
            .await
            .unwrap();

        assert!(
            compacted,
            "a split-turn compaction round should still apply"
        );
        assert_eq!(mock.calls(), 2, "a split turn must issue two summary calls");
        assert!(
            matches!(&session.messages[0].content[0], ContentBlock::Text { text }
                if text.contains("history summary text")
                    && text.contains("**Turn Context (split turn):**")
                    && text.contains("turn prefix summary text")),
            "expected the merged summary, got: {:?}",
            session.messages[0].content
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
            Message::assistant(vec![ContentBlock::Text {
                text: "first done".into(),
            }]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: json!({ "path": "a.rs" }),
            }]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "edit".into(),
                input: json!({ "path": "a.rs" }),
            }]),
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
            .compact(&mut session, CompactionReason::Manual, &cancel, &mut |_| {})
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
        // Of the two calls a split-turn compaction issues, the turn-prefix one gets half the
        // history call's `summary_max_tokens` budget — a partial turn needs proportionally less room.
        let session_messages = vec![
            Message::user("first request"),
            Message::assistant(vec![ContentBlock::Text {
                text: "first done".into(),
            }]),
            Message::user("second request"),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: json!({ "path": "a.rs" }),
            }]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "edit".into(),
                input: json!({ "path": "a.rs" }),
            }]),
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
            ..CompactionConfig::default()
        });
        let cancel = CancellationToken::new();
        agent
            .compact(&mut session, CompactionReason::Manual, &cancel, &mut |_| {})
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
            "the turn-prefix call gets half the budget"
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
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "1".into(),
                name: "read".into(),
                input: json!({ "path": "a.rs" }),
            }]),
            Message::tool_result("1", "contents of a.rs", false),
            Message::assistant(vec![ContentBlock::ToolUse {
                id: "2".into(),
                name: "edit".into(),
                input: json!({ "path": "a.rs" }),
            }]),
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
            .compact(&mut session, CompactionReason::Manual, &cancel, &mut |_| {})
            .await
            .unwrap();

        assert!(compacted);
        assert_eq!(
            mock.calls(),
            1,
            "reusing the prior summary verbatim must skip its model call entirely"
        );
        assert!(
            matches!(&session.messages[0].content[0], ContentBlock::Text { text }
                if text.contains("prior summary body")
                    && text.contains("**Turn Context (split turn):**")
                    && text.contains("turn prefix summary text")),
            "expected the prior summary folded forward verbatim, got: {:?}",
            session.messages[0].content
        );
    }

    #[test]
    fn agent_new_scales_summary_max_tokens_from_model_capabilities() {
        // A model with a `max_output` ceiling below the naive 0.8*reserve_tokens computation must have
        // its summarization budget clamped to that ceiling, not the flat 4096 default nor an
        // unreachably large number the model would reject.
        let mock = Arc::new(MockTransport::new(vec![]));
        // claude-3-haiku-20240307: gen-3 legacy, max_output 4_096 (see `models.rs`) — comfortably
        // below the default reserve_tokens(24_000)*0.8 = 19_200, so the clamp must bite.
        let agent = Agent::new(mock, "claude-3-haiku-20240307");
        assert_eq!(
            agent.compaction.summary_max_tokens, 4_096,
            "summary_max_tokens must be clamped to the model's own max_output"
        );

        // A model with a large max_output should get the full 0.8*reserve_tokens computation, not the
        // old flat 4096 default.
        let mock2 = Arc::new(MockTransport::new(vec![]));
        let agent2 = Agent::new(mock2, "claude-opus-4-8");
        assert_eq!(agent2.compaction.summary_max_tokens, 19_200);
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
            agent.compaction.summary_max_tokens, 19_200,
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
                text: "reasoning…".into(),
            },
            StreamEvent::SignatureDelta {
                signature: "sig-xyz".into(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::TextDelta {
                text: "the answer".into(),
            },
            StreamEvent::ContentBlockStop,
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
            ContentBlock::Text { text } if text == "the answer"
        ));
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
            ContentBlock::Text { text } if text == "now do the follow-up"
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
            ContentBlock::Text { text } if text == "actually, also handle the edge case"
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
            ContentBlock::Text { text } if text == "also handle this"
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
        // Two tool round-trips followed by a final text turn: the checkpoint must fire twice — once
        // per tool_results commit — each time with every message durable up to that point (never
        // mid-way through an assistant message's tool_use with no matching tool_result yet), and it
        // must NOT double-fire or fire again for the final text-only turn's own `AgentEnd` (that path
        // is already covered by a caller's own post-run persist).
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
        // user, assistant(tool_use), tool_results = 3 after the first round-trip; +2 more after the
        // second = 5. The final text turn ends the run via `AgentEnd`, not a checkpoint.
        assert_eq!(
            lens,
            vec![3, 5],
            "checkpoint must fire exactly once per tool round-trip, with the session already durable"
        );
        assert_eq!(
            session.messages.len(),
            6,
            "sanity: final count after the text turn"
        );
    }

    #[tokio::test]
    async fn checkpoint_fires_when_a_steered_message_is_injected_at_a_stop_boundary() {
        // A follow-up queued before the model would otherwise stop must also land on a checkpoint —
        // the injected user message is itself a valid, resumable point.
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
        // user, assistant("first answer"), user(follow-up) = 3, right when the follow-up is injected.
        assert_eq!(lens, vec![3]);
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
                id: "f".into(),
                name: "fast".into(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::ToolUseStart {
                id: "h".into(),
                name: "hang".into(),
            },
            StreamEvent::ContentBlockStop,
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
}
