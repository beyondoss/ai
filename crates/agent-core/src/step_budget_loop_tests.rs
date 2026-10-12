//! The wrap-up in the real loop: the grant, where its message lands in the transcript, and that a run
//! without a ceiling never sees it. The decision logic itself is unit-tested in
//! `step_budget.rs`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::message::{ContentBlock, Role};
use crate::mock::{MockTransport, turn};
use crate::step_budget::WRAP_UP_STEPS;
use crate::tool::{Tool, ToolOutput, ToolRegistry};
use crate::{Agent, AgentEvent, Error, Session};

struct Noop;

#[async_trait]
impl Tool for Noop {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "does nothing"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn run(&self, _input: Value) -> Result<ToolOutput, crate::error::ToolError> {
        Ok("done".into())
    }
}

fn agent(turns: Vec<Vec<crate::StreamEvent>>, max: Option<u32>) -> Agent {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(Noop));
    Agent::new(Arc::new(MockTransport::new(turns)), "claude-opus-4-8")
        .with_tools(tools)
        .with_max_steps(max)
}

fn tool_turns(n: usize) -> Vec<Vec<crate::StreamEvent>> {
    (0..n)
        .map(|i| turn::tool_call(&format!("t{i}"), "noop", "{}"))
        .collect()
}

/// Every user-side text block, in order, with the index of the message it is in.
fn user_texts(session: &Session) -> Vec<(usize, String)> {
    session
        .messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User)
        .flat_map(|(i, m)| {
            m.content.iter().filter_map(move |b| match b {
                ContentBlock::Text { text, .. } => Some((i, text.to_string())),
                _ => None,
            })
        })
        .collect()
}

async fn run(agent: &Agent) -> (Session, Result<(), Error>, Vec<u32>) {
    let mut session = Session::new();
    session.user("go");
    let mut fired = Vec::new();
    let result = agent
        .run_events(&mut session, |ev| {
            if let AgentEvent::WrapUp { steps } = ev {
                fired.push(steps);
            }
        })
        .await;
    (session, result, fired)
}

#[tokio::test]
async fn without_a_ceiling_nothing_is_injected() {
    let mut turns = tool_turns(10);
    turns.push(turn::text("done"));
    let (session, result, fired) = run(&agent(turns, None)).await;
    assert!(result.is_ok(), "{result:?}");
    assert!(fired.is_empty());
    assert!(
        user_texts(&session)
            .iter()
            .all(|(_, t)| !t.contains("[step budget]"))
    );
}

#[tokio::test]
async fn the_wrap_up_grants_extra_steps_and_the_error_still_names_the_ceiling() {
    // Keeps calling tools through the grant: the run ends at 4 + WRAP_UP_STEPS steps.
    let (session, result, fired) = run(&agent(tool_turns(20), Some(4))).await;
    assert!(matches!(result, Err(Error::MaxSteps(4))), "{result:?}");
    assert_eq!(fired, vec![WRAP_UP_STEPS]);
    assert_eq!(session.steps, 4 + WRAP_UP_STEPS);
    let (idx, _) = user_texts(&session)
        .into_iter()
        .find(|(_, t)| {
            t.contains("4 steps are used up") && t.contains(&format!("{WRAP_UP_STEPS} more"))
        })
        .expect("the wrap-up text");
    // In the same message as a tool result: no extra user turn, so role alternation holds.
    assert!(
        session.messages[idx]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    );
}

#[tokio::test]
async fn a_model_that_stops_during_the_wrap_up_ends_cleanly() {
    let mut turns = tool_turns(3);
    turns.push(turn::text("reverted the broken edit; summary"));
    let (_session, result, fired) = run(&agent(turns, Some(3))).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(fired, vec![WRAP_UP_STEPS]);
}
