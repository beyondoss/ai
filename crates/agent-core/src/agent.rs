//! The agent loop — Pi's `pi-agent-core` runtime, ported.
//!
//! One iteration = one model turn: stream a completion, assemble the assistant message from the
//! event stream, append it to the session, and — if the model asked for tools — run each tool and
//! feed the results back as a new user turn. Repeat until the model ends its turn (or `max_steps`).
//!
//! The loop is dialect-blind (both wire dialects normalize to the same `StreamEvent` sequence) and
//! network-blind (it depends only on [`ModelTransport`], so tests drive it with `MockTransport`).

use std::sync::Arc;

use futures::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::message::{ContentBlock, Message, StopReason, StreamEvent, ToolDef};
use crate::session::Session;
use crate::tool::ToolRegistry;
use crate::transport::{ModelRequest, ModelTransport};

/// An observable event from a run: a streamed model event, a tool-invocation boundary, or a turn
/// boundary. The headless server serializes these to its clients; [`Agent::run`] exposes only the
/// `Stream` events.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentEvent {
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
}

/// Default per-turn output token ceiling.
const DEFAULT_MAX_TOKENS: u32 = 4096;
/// Default ceiling on loop iterations before bailing — a runaway-tool-call backstop.
const DEFAULT_MAX_STEPS: u32 = 24;

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
    pub async fn run_events<F>(&self, session: &mut Session, mut sink: F) -> Result<()>
    where
        F: FnMut(AgentEvent),
    {
        loop {
            if session.steps >= self.max_steps {
                return Err(Error::MaxSteps(self.max_steps));
            }

            let mut req = ModelRequest::new(
                self.model.clone(),
                session.messages.clone(),
                self.max_tokens,
            )
            .with_tools(self.tool_defs.clone());
            if let Some(system) = &self.system {
                req = req.with_system(system.clone());
            }

            let turn = {
                let mut emit = |ev: StreamEvent| sink(AgentEvent::Stream(ev));
                self.run_turn(req, &mut emit).await?
            };
            session.push(Message::assistant(turn.blocks));
            session.record_usage(turn.input_tokens, turn.output_tokens);
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
                return Ok(()); // model ended its turn — done.
            }

            // Run the tools and feed results back as a user turn. A tool's own failure becomes an
            // error `tool_result`, not an aborted run — the model can react to it next turn.
            //
            // The calls run concurrently: tools are I/O-bound (file reads, shell commands, the
            // `beyond` CLI), and a model routinely batches independent ones in a single turn, so
            // overlapping them collapses the tool phase from the sum of their latencies to its slowest
            // member. The transcript stays deterministic regardless of finish order — every
            // `ToolStart` is emitted up front in call order, and the `ToolEnd`s and `tool_result`
            // messages are emitted/appended in call order after the join, never interleaved by
            // whichever tool happened to finish first.
            for (id, name, input) in &calls {
                sink(AgentEvent::ToolStart {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }
            let runs = calls.iter().map(|(_, name, input)| {
                let tool = self.tools.get(name);
                let name = name.clone();
                let input = input.clone();
                async move {
                    match tool {
                        Some(tool) => match tool.run(input).await {
                            Ok(out) => (out, false),
                            Err(e) => (e.to_string(), true),
                        },
                        None => (format!("unknown tool: {name}"), true),
                    }
                }
            });
            let results = futures::future::join_all(runs).await;
            for ((id, name, _), (content, is_error)) in calls.iter().zip(results) {
                sink(AgentEvent::ToolEnd {
                    id: id.clone(),
                    name: name.clone(),
                    result: content.clone(),
                    is_error,
                });
                session.push(Message::tool_result(id.clone(), content, is_error));
            }
        }
    }

    /// Stream and assemble a single model turn into content blocks + accounting.
    async fn run_turn(&self, req: ModelRequest, emit: &mut dyn FnMut(StreamEvent)) -> Result<Turn> {
        let mut stream = self.transport.stream(req).await?;
        let mut acc = Accumulator::default();
        while let Some(ev) = stream.next().await {
            let ev = ev?;
            emit(ev.clone());
            acc.apply(ev);
        }
        acc.finish()
    }
}

/// The assembled result of one model turn.
struct Turn {
    blocks: Vec<ContentBlock>,
    stop_reason: StopReason,
    input_tokens: u32,
    output_tokens: u32,
}

/// Folds a `StreamEvent` sequence into content blocks. Text accrues into the current text run; a
/// tool call accrues its streamed JSON argument fragments; `ContentBlockStop` finalizes whichever is
/// open. Works identically for both dialects because they emit the same event shape.
#[derive(Default)]
struct Accumulator {
    blocks: Vec<ContentBlock>,
    text: String,
    tool: Option<(String, String, String)>, // (id, name, json-arg buffer)
    stop_reason: StopReason,
    input_tokens: u32,
    output_tokens: u32,
    /// The first tool-call argument buffer that failed to parse, if any. Held until `finish` so the
    /// turn fails with [`Error::MalformedToolInput`] rather than handing a tool `null` arguments.
    bad_tool_args: Option<String>,
}

impl Accumulator {
    fn apply(&mut self, ev: StreamEvent) {
        match ev {
            StreamEvent::MessageStart => {}
            StreamEvent::TextDelta { text } => self.text.push_str(&text),
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
            StreamEvent::Usage {
                input_tokens,
                output_tokens,
            } => {
                self.input_tokens = input_tokens;
                self.output_tokens = output_tokens;
            }
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
        if let Some((id, name, args)) = self.tool.take() {
            let input = if args.trim().is_empty() {
                json!({})
            } else {
                match serde_json::from_str(&args) {
                    Ok(v) => v,
                    // Malformed arguments from the stream are a protocol failure, not a tool failure:
                    // record the offending buffer and fail the turn in `finish` rather than dispatch a
                    // tool with `null` input and let it report an opaque "missing field" error.
                    Err(_) => {
                        if self.bad_tool_args.is_none() {
                            self.bad_tool_args = Some(args);
                        }
                        Value::Null
                    }
                }
            };
            self.blocks.push(ContentBlock::ToolUse { id, name, input });
        } else {
            self.flush_text();
        }
    }

    fn finish(mut self) -> Result<Turn> {
        // A stream that ended without a trailing ContentBlockStop (or with leftover text) still
        // contributes its text.
        self.flush_block();
        if let Some(args) = self.bad_tool_args {
            return Err(Error::MalformedToolInput(args));
        }
        Ok(Turn {
            blocks: self.blocks,
            stop_reason: self.stop_reason,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        // Results fed back in call order, one user message per result (the existing transcript shape):
        // user, assistant(2× tool_use), user(tool_result a), user(tool_result b), assistant(text).
        assert_eq!(session.messages.len(), 5);
        match (
            &session.messages[2].content[0],
            &session.messages[3].content[0],
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
    async fn malformed_tool_args_fail_the_turn() {
        // The stream opens a tool call but its argument fragments never form valid JSON.
        let turn = vec![
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
        let (agent, _mock) = agent_with(vec![turn], tools);
        let mut session = Session::new();
        session.user("go");
        let err = agent.run(&mut session, |_| {}).await.unwrap_err();
        assert!(matches!(err, Error::MalformedToolInput(_)));
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
}
