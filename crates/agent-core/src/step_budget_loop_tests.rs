//! `StepBudget` in the real loop: where each message lands in the transcript, the wrap-up grant,
//! and that the default changes nothing. The decision logic itself is unit-tested in
//! `step_budget.rs`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::message::{ContentBlock, Role};
use crate::mock::{MockTransport, turn};
use crate::tool::{Tool, ToolOutput, ToolRegistry};
use crate::{Agent, AgentEvent, Error, Session, StepBudget, StepBudgetKind};

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

fn agent(turns: Vec<Vec<crate::StreamEvent>>, max: u32, budget: StepBudget) -> Agent {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(Noop));
    Agent::new(Arc::new(MockTransport::new(turns)), "claude-opus-4-8")
        .with_tools(tools)
        .with_max_steps(max)
        .with_step_budget(budget)
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

async fn run(agent: &Agent) -> (Session, Result<(), Error>, Vec<(StepBudgetKind, u32)>) {
    let mut session = Session::new();
    session.user("go");
    let mut fired = Vec::new();
    let result = agent
        .run_events(&mut session, |ev| {
            if let AgentEvent::StepBudget { message, steps } = ev {
                fired.push((message, steps));
            }
        })
        .await;
    (session, result, fired)
}

#[tokio::test]
async fn the_default_budget_injects_nothing() {
    let (session, result, fired) = run(&agent(tool_turns(10), 4, StepBudget::default())).await;
    assert!(matches!(result, Err(Error::MaxSteps(4))));
    assert!(fired.is_empty());
    assert!(
        user_texts(&session)
            .iter()
            .all(|(_, t)| !t.contains("[step budget]"))
    );
}

#[tokio::test]
async fn the_notice_rides_the_tool_results_turn_when_the_threshold_is_reached() {
    let budget = StepBudget {
        notice_at: Some(2),
        ..StepBudget::default()
    };
    let (session, result, fired) = run(&agent(tool_turns(10), 5, budget)).await;
    assert!(matches!(result, Err(Error::MaxSteps(5))));
    assert_eq!(fired, vec![(StepBudgetKind::Notice, 2)]);
    let notices: Vec<_> = user_texts(&session)
        .into_iter()
        .filter(|(_, t)| t.contains("[step budget]"))
        .collect();
    assert_eq!(notices.len(), 1, "{notices:?}");
    let (idx, text) = &notices[0];
    assert!(text.contains("2 of this run's 5 steps remain"), "{text}");
    // In the same message as a tool result: no extra user turn, so role alternation holds.
    assert!(
        session.messages[*idx]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    );
}

#[tokio::test]
async fn the_wrap_up_grants_extra_steps_and_the_error_still_names_the_ceiling() {
    let budget = StepBudget {
        wrap_up_steps: Some(2),
        ..StepBudget::default()
    };
    // Keeps calling tools through the grant: the run ends at 4 + 2 steps.
    let (session, result, fired) = run(&agent(tool_turns(10), 4, budget)).await;
    assert!(matches!(result, Err(Error::MaxSteps(4))), "{result:?}");
    assert_eq!(fired, vec![(StepBudgetKind::WrapUp, 2)]);
    assert_eq!(session.steps, 6);
    assert!(
        user_texts(&session)
            .iter()
            .any(|(_, t)| t.contains("4 steps are used up") && t.contains("2 more"))
    );
}

#[tokio::test]
async fn a_model_that_stops_during_the_wrap_up_ends_cleanly() {
    let budget = StepBudget {
        wrap_up_steps: Some(3),
        continue_nudge: true,
        ..StepBudget::default()
    };
    let mut turns = tool_turns(3);
    turns.push(turn::text("reverted the broken edit; summary"));
    let (_session, result, fired) = run(&agent(turns, 3, budget)).await;
    assert!(result.is_ok(), "{result:?}");
    // No nudge after a wrap-up: the model was told to stop.
    assert_eq!(fired, vec![(StepBudgetKind::WrapUp, 3)]);
}

#[tokio::test]
async fn an_early_stop_is_nudged_once_then_allowed() {
    let budget = StepBudget {
        continue_nudge: true,
        ..StepBudget::default()
    };
    let turns = vec![
        turn::tool_call("t0", "noop", "{}"),
        turn::text("I did not complete the task."),
        turn::tool_call("t1", "noop", "{}"),
        turn::text("Done now."),
    ];
    let (session, result, fired) = run(&agent(turns, 10, budget)).await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(fired, vec![(StepBudgetKind::Nudge, 8)]);
    let texts = user_texts(&session);
    let nudge = texts
        .iter()
        .find(|(_, t)| t.contains("[step budget]"))
        .expect("nudge in transcript");
    assert!(nudge.1.contains("8 of 10 steps left"), "{}", nudge.1);
    // Its own user turn, right after the assistant's stop.
    assert_eq!(session.messages[nudge.0 - 1].role, Role::Assistant);
    assert_eq!(session.steps, 4);
}
