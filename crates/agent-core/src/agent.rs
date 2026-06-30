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

use futures::StreamExt;
use futures::future::{Either, select};
use serde::Serialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::compaction::{self, CompactionConfig};
use crate::error::{Error, Result};
use crate::hooks::{AgentHooks, NoHooks};
use crate::message::{ContentBlock, Message, StopReason, StreamEvent, TokenUsage, ToolDef};
use crate::session::Session;
use crate::steering::Steering;
use crate::tool::ToolRegistry;
use crate::transport::{ModelRequest, ModelTransport};

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
    },
    /// The run is ending abnormally (transport failure after retries, malformed SSE, or the step
    /// ceiling). A terminal marker on the event stream so a streaming client sees *why* a run stopped
    /// rather than the stream just going silent; `run_events` still returns the same `Err`.
    Error { message: String },
}

/// Default per-turn output token ceiling.
const DEFAULT_MAX_TOKENS: u32 = 4096;
/// Default ceiling on loop iterations before bailing — a runaway-tool-call backstop.
const DEFAULT_MAX_STEPS: u32 = 24;
/// Cap on tool-call groups dispatched concurrently within one turn. A model usually batches a
/// handful, but nothing bounds how many it requests; without a cap a turn asking for dozens of
/// `bash`/`grep` calls would spawn that many subprocesses / parallel walks at once (and `grep` itself
/// fans out over CPU cores, compounding it). The cap throttles in-flight groups; results scatter by
/// index, so the call-order transcript is unaffected — only peak concurrency is bounded.
const MAX_CONCURRENT_TOOL_GROUPS: usize = 8;

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
    /// Context-compaction policy: when to summarize the prefix to stay under the context window.
    compaction: CompactionConfig,
    /// Interception hooks around tool calls (gate/rewrite). Defaults to no-ops.
    hooks: Arc<dyn AgentHooks>,
    /// Stable prompt-cache affinity key for this run (OpenAI `prompt_cache_key`).
    cache_key: Option<String>,
    /// Use the 1-hour prompt-cache TTL (Anthropic) instead of the default 5 minutes.
    cache_long: bool,
}

impl Agent {
    /// An agent over `transport` using `model`, with no tools and default bounds.
    pub fn new(transport: Arc<dyn ModelTransport>, model: impl Into<String>) -> Self {
        Self {
            transport,
            tools: ToolRegistry::new(),
            tool_defs: Vec::new().into(),
            model: model.into(),
            system: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            max_steps: DEFAULT_MAX_STEPS,
            thinking: None,
            compaction: CompactionConfig::default(),
            hooks: Arc::new(NoHooks),
            cache_key: None,
            cache_long: false,
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

    /// Set the compaction policy (context window, reserve, keep-recent, enabled).
    pub fn with_compaction(mut self, compaction: CompactionConfig) -> Self {
        self.compaction = compaction;
        self
    }

    /// Convenience: set just the model's context window, leaving the other compaction defaults.
    pub fn with_context_window(mut self, context_window: u32) -> Self {
        self.compaction.context_window = context_window;
        self
    }

    /// Install interception hooks (tool gating / result rewriting). Defaults to no-ops.
    pub fn with_hooks(mut self, hooks: Arc<dyn AgentHooks>) -> Self {
        self.hooks = hooks;
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
        sink(AgentEvent::AgentStart);
        // Set once we've already compacted to recover from a context-overflow this turn, so a second
        // overflow gives up instead of looping. Reset after each turn that lands cleanly.
        let mut overflow_recovered = false;
        loop {
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            if session.steps >= self.max_steps {
                let err = Error::MaxSteps(self.max_steps);
                sink(AgentEvent::Error {
                    message: err.to_string(),
                });
                return Err(err);
            }

            // Proactive compaction: once the live prompt crosses the threshold, summarize the prefix
            // before building the next request so the run never walks into the context wall.
            if self.compaction.enabled && compaction::should_compact(session, &self.compaction) {
                self.compact(session, &cancel, &mut sink).await?;
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
                    if self.compact(session, &cancel, &mut sink).await? {
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
            let malformed: HashMap<String, String> =
                std::mem::take(&mut turn.malformed).into_iter().collect();
            session.push(Message::assistant(turn.blocks));
            session.record_usage(turn.usage);
            session.steps += 1;
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
                // The model ended its turn. Before stopping, drain any steering/follow-up messages a
                // client queued mid-run and continue with them as new user turns. The last message is
                // the assistant's, so pushing user turns here keeps the wire's role alternation valid.
                let injected = steering.drain();
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
                continue;
            }

            // Run the tools and feed results back as a single user turn. A tool's own failure becomes
            // an error `tool_result`, not an aborted run — the model can react to it next turn.
            //
            // The calls run concurrently: tools are I/O-bound (file reads, shell commands, the
            // `beyond` CLI), and a model routinely batches independent ones in a single turn, so
            // overlapping them collapses the tool phase from the sum of their latencies to its slowest
            // member. The transcript stays deterministic regardless of finish order — every
            // `ToolStart` is emitted up front in call order, and the `ToolEnd`s and `tool_result`
            // blocks are emitted/collected in call order after the join, never interleaved by
            // whichever tool happened to finish first.
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
            let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
            for (i, (_, name, input)) in calls.iter().enumerate() {
                let key = self
                    .tools
                    .get(name)
                    .and_then(|t| t.write_target(input))
                    .map(|path| format!("path:{path}"))
                    .unwrap_or_else(|| format!("solo:{i}"));
                groups.entry(key).or_default().push(i);
            }
            let this = self;
            let malformed = &malformed;
            let group_runs = groups.into_values().map(|indices| {
                let calls = &calls;
                async move {
                    let mut out = Vec::with_capacity(indices.len());
                    for i in indices {
                        let (id, name, input) = &calls[i];
                        let result = if let Some(raw) = malformed.get(id) {
                            // The model streamed a tool call whose argument fragments never formed
                            // valid JSON. Feed that back as an error result the model can correct next
                            // turn rather than aborting the whole run on one malformed call.
                            (
                                format!(
                                    "tool call arguments were not valid JSON and could not be parsed: {raw}"
                                ),
                                true,
                            )
                        } else if let Some(reason) =
                            this.hooks.before_tool_call(name, input).await
                        {
                            // A hook blocked the call (e.g. a permission policy). Feed the reason back
                            // as an error result instead of running the tool.
                            (format!("tool call blocked: {reason}"), true)
                        } else {
                            let (out, is_error) = match this.tools.get(name) {
                                Some(tool) => match tool.run(input.clone()).await {
                                    Ok(o) => (o, false),
                                    Err(e) => (e.to_string(), true),
                                },
                                None => (format!("unknown tool: {name}"), true),
                            };
                            // Let a hook rewrite the result (redact, cap, reclassify) before it's fed
                            // back to the model.
                            this.hooks.after_tool_call(name, input, out, is_error).await
                        };
                        out.push((i, result));
                    }
                    out
                }
            });
            let mut results: Vec<(String, bool)> =
                (0..calls.len()).map(|_| (String::new(), false)).collect();
            // Bound how many groups run at once. `buffer_unordered` is safe here because each group
            // yields its results tagged with their original call index `i`; cross-group completion
            // order never reaches the transcript, which is rebuilt in call order below.
            //
            // Race the whole dispatch against cancellation: a tripped token drops `drain`, which drops
            // every in-flight tool future — aborting a hung `bash` (its `kill_on_drop` child dies) and
            // any other long-running tool — and returns promptly instead of waiting them all out. The
            // block scopes `drain`'s `&mut results` borrow so the transcript below can consume them.
            {
                let drain = async {
                    let mut group_stream = futures::stream::iter(group_runs)
                        .buffer_unordered(MAX_CONCURRENT_TOOL_GROUPS);
                    while let Some(group) = group_stream.next().await {
                        for (i, result) in group {
                            results[i] = result;
                        }
                    }
                };
                let cancelled = cancel.cancelled();
                futures::pin_mut!(drain, cancelled);
                if let Either::Right(((), _)) = select(drain, cancelled).await {
                    return Err(Error::Cancelled);
                }
            }
            let mut result_blocks = Vec::with_capacity(calls.len());
            for ((id, name, _), (content, is_error)) in calls.iter().zip(results) {
                sink(AgentEvent::ToolEnd {
                    id: id.clone(),
                    name: name.clone(),
                    result: content.clone(),
                    is_error,
                });
                result_blocks.push(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content,
                    is_error,
                });
            }
            session.push(Message::tool_results(result_blocks));
        }
    }

    /// Stream and assemble a single model turn into content blocks + accounting. Racing each
    /// `stream.next()` against `cancel` means a tripped token interrupts even a model that has gone
    /// quiet (a blocked read would otherwise hang for the full idle timeout); dropping `stream` on
    /// cancel aborts the underlying HTTP request.
    async fn run_turn(
        &self,
        req: ModelRequest,
        emit: &mut dyn FnMut(StreamEvent),
        cancel: &CancellationToken,
    ) -> Result<Turn> {
        let mut stream = self.transport.stream(req).await?;
        let mut acc = Accumulator::default();
        let cancelled = cancel.cancelled();
        futures::pin_mut!(cancelled);
        loop {
            let next = stream.next();
            futures::pin_mut!(next);
            match select(next, cancelled.as_mut()).await {
                Either::Left((Some(ev), _)) => {
                    let ev = ev?;
                    emit(ev.clone());
                    acc.apply(ev);
                }
                Either::Left((None, _)) => break,
                Either::Right(((), _)) => return Err(Error::Cancelled),
            }
        }
        Ok(acc.finish())
    }

    /// Summarize the conversation prefix in place, keeping the recent turns verbatim. Makes one
    /// summarization model call (silently — its tokens aren't surfaced as assistant output), splices
    /// the summary into `session`, and emits an [`AgentEvent::Compacted`]. Returns `false` (a no-op)
    /// when there's no worthwhile prefix to summarize or the model returns an empty summary. Exposed
    /// so a headless server can offer a manual `compact` command.
    pub async fn compact(
        &self,
        session: &mut Session,
        cancel: &CancellationToken,
        sink: &mut dyn FnMut(AgentEvent),
    ) -> Result<bool> {
        let Some(first_kept) =
            compaction::find_cut(&session.messages, self.compaction.keep_recent_tokens)
        else {
            return Ok(false);
        };
        let before = session.messages.len();
        let prefix: Vec<Message> = session.messages[..first_kept].to_vec();
        let req =
            compaction::summary_request(&self.model, &prefix, self.compaction.summary_max_tokens);
        let turn = self.run_turn(req, &mut |_| {}, cancel).await?;
        let summary: String = turn
            .blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        if summary.trim().is_empty() {
            return Ok(false);
        }
        compaction::apply_summary(session, first_kept, &summary);
        sink(AgentEvent::Compacted {
            messages_before: before,
            messages_after: session.messages.len(),
        });
        Ok(true)
    }
}

/// Whether a transport error is the provider rejecting the request for exceeding its context window —
/// the signal to compact and retry. Matched on the error text (the wire shape varies by provider).
fn is_context_overflow(e: &Error) -> bool {
    let Error::Transport(msg) = e else {
        return false;
    };
    let m = msg.to_ascii_lowercase();
    m.contains("prompt is too long")
        || m.contains("too many tokens")
        || (m.contains("context")
            && (m.contains("long")
                || m.contains("exceed")
                || m.contains("maximum")
                || m.contains("window")))
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
    fn apply(&mut self, ev: StreamEvent) {
        match ev {
            StreamEvent::MessageStart => {}
            StreamEvent::TextDelta { text } => self.text.push_str(&text),
            StreamEvent::ThinkingDelta { text } => {
                self.thinking
                    .get_or_insert_with(Default::default)
                    .0
                    .push_str(&text);
            }
            StreamEvent::SignatureDelta { signature } => {
                self.thinking
                    .get_or_insert_with(Default::default)
                    .1
                    .push_str(&signature);
            }
            StreamEvent::RedactedThinking { data } => {
                // Self-contained block with no deltas — close any open text and emit it directly.
                self.flush_text();
                self.blocks.push(ContentBlock::RedactedThinking { data });
            }
            StreamEvent::ToolUseStart { id, name } => {
                self.flush_text();
                self.tool = Some((id, name, String::new()));
            }
            StreamEvent::InputJsonDelta { partial_json } => {
                if let Some((_, _, buf)) = &mut self.tool {
                    buf.push_str(&partial_json);
                }
            }
            StreamEvent::ContentBlockStop => self.flush_block(),
            StreamEvent::Usage(usage) => self.usage = usage,
            StreamEvent::MessageStop { stop_reason } => self.stop_reason = stop_reason,
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
                match serde_json::from_str(&args) {
                    Ok(v) => v,
                    // Malformed arguments from the stream are a protocol glitch, not a tool failure.
                    // Keep the tool_use block (with an empty, wire-valid input object so the next
                    // request doesn't 400) and record the call as malformed; the loop feeds back an
                    // error result the model can correct, instead of aborting the whole run.
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
        async fn run(&self, input: Value) -> std::result::Result<String, crate::error::ToolError> {
            input
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_string)
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
                is_error: false
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
            ) -> std::result::Result<String, crate::error::ToolError> {
                self.barrier.wait().await;
                Ok(self.id.to_string())
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
            ) -> std::result::Result<String, crate::error::ToolError> {
                self.log.lock().unwrap().push(format!("start:{}", self.id));
                tokio::task::yield_now().await;
                self.log.lock().unwrap().push(format!("end:{}", self.id));
                Ok(self.id.to_string())
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
            ) -> std::result::Result<String, crate::error::ToolError> {
                let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_seen.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(String::new())
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
            async fn before_tool_call(&self, _name: &str, _input: &Value) -> Option<String> {
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
            async fn run(&self, _: Value) -> std::result::Result<String, crate::error::ToolError> {
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
