# beyond-ai-agent-core

The agent loop as a library. Give it a message history, a model transport and a
set of tools; it runs model turns and tool calls until the model stops, and
emits an event for every streamed token, tool boundary and turn boundary.

It is the layer under [`beyond-ai-agent`](../agent/README.md), which adds the
concrete tools, the CLI and the `serve` protocol. Use this crate to build a
different agent on the same loop. It has no HTTP server and no provider SDK,
and ships no tools.

## Example

A scripted model calls one tool, then answers. No network, no key:

```rust
use std::sync::Arc;

use agent_core::mock::turn;
use agent_core::{
    Agent, Message, MockTransport, Session, Tool, ToolError, ToolOutput, ToolRegistry,
};
use async_trait::async_trait;
use serde_json::{Value, json};

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echo the `text` argument back."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})
    }
    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let text = input["text"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("missing `text`".into()))?;
        Ok(text.into())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let transport = Arc::new(MockTransport::new(vec![
        turn::tool_call("t1", "echo", r#"{"text":"hi"}"#),
        turn::text("done"),
    ]));
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(Echo));

    let agent = Agent::new(transport.clone(), "claude-opus-4-8").with_tools(tools);
    let mut session = Session::new();
    session.push(Message::user("say hi"));
    agent.run(&mut session, |_| {}).await.unwrap();
    assert_eq!(transport.calls(), 2);
}
```

For a real model, swap the transport for `GatewayClient::new(base_url,
api_key)`. It POSTs the model's wire format to `base_url` and decodes the SSE
stream: a Beyond gateway, or a provider directly through a route override.

## The two seams

| Seam             | Implement it to                                                          | Provided                                        |
| ---------------- | ------------------------------------------------------------------------ | ----------------------------------------------- |
| `Tool`           | Add a capability: a name, a JSON Schema, and `run(input) -> ToolOutput`. | None. `crates/agent` has the coding tools.      |
| `ModelTransport` | Carry a `ModelRequest` to a model and return a stream of `StreamEvent`s. | `GatewayClient`, and `MockTransport` for tests. |

A tool error becomes an error `tool_result` the model sees; it does not end the
run. The loop validates nothing about a tool's input beyond JSON well-formedness
and a best-effort type coercion, so each tool checks its own arguments.

## What the loop does

- **Three wires.** Builds and decodes Anthropic Messages, OpenAI Chat
  Completions and OpenAI Responses requests. The wire is chosen from the
  provider first and the model id second (`Dialect`). Anthropic requests carry
  three prompt-cache breakpoints; Responses requests carry a prefix-hashed
  `prompt_cache_key`.
- **Concurrent tools, ordered transcript.** A turn's tool calls run
  concurrently, grouped so calls to the same path serialize; results are
  recorded in call order.
- **Cancellation.** `run_events_cancellable` takes a `CancellationToken`. A
  cancelled run stays resumable: calls with no result yet get a synthesized
  error result, never an orphaned `tool_use`.
- **Steering.** `Steering` queues messages to fold into a running turn, to
  inject when the agent would stop, or to deliver on the next turn.
- **Compaction.** When the prompt nears the context window the loop masks old
  tool output, then summarizes older turns; sizes are measured in real tokens.
- **Retry.** Failures before the first byte retry with backoff; a rate limit
  (429 or 529) retries within a per-turn time budget (default 300 s) and honors
  the server's hint.
- **Step ceiling.** `with_max_steps(n)` ends a run with `Error::MaxSteps` after
  giving the model five more turns to finish.
- **Hooks.** `AgentHooks` can gate or rewrite a tool call and observe or patch
  the provider request and response.
- **Thinking.** `ThinkingLevel`, a token budget or a reasoning effort, mapped
  per model by `capabilities(model)`.
- **Deferred tools.** A tool registered as deferred is callable by name and
  advertised only after a `tool_search` result names it.

`Session` holds the history behind `Arc`s with copy-on-write, so a rebuilt
`Agent` shares the transcript instead of copying it, and it serializes to JSON.

## Reference

[`ARCHITECTURE.md`](ARCHITECTURE.md) has each capability in full, the turn
loop's state machine, failure modes and the configuration defaults.

The loop is modeled on [pi](https://github.com/badlogic/pi-mono); see
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
