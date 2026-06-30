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
use crate::message::{ContentBlock, Message, StopReason, StreamEvent};
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
            model: model.into(),
            system: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            max_steps: DEFAULT_MAX_STEPS,
        }
    }

    /// Set the tools the model may call.
    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
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
            .with_tools(self.tools.definitions());
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

            // Run each tool and feed results back as a user turn. A tool's own failure becomes an
            // error `tool_result`, not an aborted run — the model can react to it next turn.
            for (id, name, input) in calls {
                sink(AgentEvent::ToolStart {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
                let (content, is_error) = match self.tools.get(&name) {
                    Some(tool) => match tool.run(input).await {
                        Ok(out) => (out, false),
                        Err(e) => (e.to_string(), true),
                    },
                    None => (format!("unknown tool: {name}"), true),
                };
                sink(AgentEvent::ToolEnd {
                    id: id.clone(),
                    name: name.clone(),
                    result: content.clone(),
                    is_error,
                });
                session.push(Message::tool_result(id, content, is_error));
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
        Ok(acc.finish())
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
                serde_json::from_str(&args).unwrap_or(Value::Null)
            };
            self.blocks.push(ContentBlock::ToolUse { id, name, input });
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
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
        }
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
