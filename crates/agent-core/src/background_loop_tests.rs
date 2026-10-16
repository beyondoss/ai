//! Background work in the real loop: a finish notice folds into the next tool-results turn exactly
//! once, a pending notice is delivered before a would-stop ends the run, the stop reminder fires once,
//! and a compaction summary carries the live table. The bash side (what the texts say, when a job
//! counts as ended) lives in the `agent` crate.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::background::Background;
use crate::message::{ContentBlock, Role};
use crate::mock::{MockTransport, turn};
use crate::tool::{Tool, ToolOutput, ToolRegistry};
use crate::{Agent, AgentEvent, BackgroundReason, Session};

/// A scripted background: notices queued by the test, a reminder that fires while `running` work
/// has not been reminded of.
#[derive(Default)]
struct Script {
    notices: Mutex<Vec<String>>,
    running: Mutex<Vec<String>>,
    reminded: Mutex<usize>,
}

impl Background for Script {
    fn take_notices(&self) -> Option<String> {
        let mut n = self.notices.lock().unwrap();
        (!n.is_empty()).then(|| std::mem::take(&mut *n).join("\n\n"))
    }
    fn stop_reminder(&self) -> Option<String> {
        let running = self.running.lock().unwrap();
        let mut reminded = self.reminded.lock().unwrap();
        if running.len() <= *reminded {
            return None;
        }
        *reminded = running.len();
        Some(format!(
            "[background jobs] still running: {}",
            running.join(", ")
        ))
    }
    fn carry(&self) -> Option<String> {
        let running = self.running.lock().unwrap();
        (!running.is_empty()).then(|| format!("running: {}", running.join(", ")))
    }
}

/// A tool whose call queues a notice (as a job ending mid-call would), exposing `script`.
struct Job {
    script: Arc<Script>,
    notice_on_call: bool,
}

#[async_trait]
impl Tool for Job {
    fn name(&self) -> &str {
        "job"
    }
    fn description(&self) -> &str {
        "runs"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn run(&self, _input: Value) -> Result<ToolOutput, crate::error::ToolError> {
        if self.notice_on_call {
            self.script
                .notices
                .lock()
                .unwrap()
                .push("[job 1 exited with code 0 after 3 s: `make`]".into());
        }
        Ok("ran".into())
    }
    fn background(&self) -> Option<Arc<dyn Background>> {
        Some(self.script.clone())
    }
}

/// A second tool over the same script, like `bash_job` beside `bash`.
struct Twin(Arc<Script>);

#[async_trait]
impl Tool for Twin {
    fn name(&self) -> &str {
        "twin"
    }
    fn description(&self) -> &str {
        "twin"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn run(&self, _input: Value) -> Result<ToolOutput, crate::error::ToolError> {
        Ok("ok".into())
    }
    fn background(&self) -> Option<Arc<dyn Background>> {
        Some(self.0.clone())
    }
}

/// A tool whose call starts new running work over the same script, as `bash` with
/// `run_in_background` does.
struct Start(Arc<Script>);

#[async_trait]
impl Tool for Start {
    fn name(&self) -> &str {
        "start"
    }
    fn description(&self) -> &str {
        "starts work"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn run(&self, _input: Value) -> Result<ToolOutput, crate::error::ToolError> {
        let mut running = self.0.running.lock().unwrap();
        let next = format!("job {}", running.len() + 10);
        running.push(next);
        Ok("started".into())
    }
    fn background(&self) -> Option<Arc<dyn Background>> {
        Some(self.0.clone())
    }
}

fn agent(script: &Arc<Script>, notice_on_call: bool, turns: Vec<Vec<crate::StreamEvent>>) -> Agent {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(Job {
        script: script.clone(),
        notice_on_call,
    }));
    tools.register(Arc::new(Twin(script.clone())));
    tools.register(Arc::new(Start(script.clone())));
    Agent::new(Arc::new(MockTransport::new(turns)), "claude-opus-4-8").with_tools(tools)
}

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

async fn run(agent: &Agent) -> (Session, Vec<(BackgroundReason, String)>) {
    let mut session = Session::new();
    session.user("go");
    let mut fired = Vec::new();
    agent
        .run_events(&mut session, |ev| {
            if let AgentEvent::Background { reason, text, .. } = ev {
                fired.push((reason, text));
            }
        })
        .await
        .unwrap();
    (session, fired)
}

#[tokio::test]
async fn a_notice_rides_the_next_tool_results_turn_exactly_once() {
    let script = Arc::new(Script::default());
    let turns = vec![
        turn::tool_call("a", "job", "{}"),
        turn::tool_call("b", "twin", "{}"),
        turn::text("done"),
    ];
    let (session, fired) = run(&agent(&script, true, turns)).await;
    let notices: Vec<_> = user_texts(&session)
        .into_iter()
        .filter(|(_, t)| t.starts_with("[job 1 exited"))
        .collect();
    assert_eq!(notices.len(), 1, "{notices:?}");
    // In the same message as the tool result: no extra user turn.
    assert!(
        session.messages[notices[0].0]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    );
    assert_eq!(fired.len(), 1, "two tools over one table report it once");
    assert_eq!(fired[0].0, BackgroundReason::Finished);
}

#[tokio::test]
async fn a_pending_notice_is_delivered_before_the_run_may_end() {
    let script = Arc::new(Script::default());
    script
        .notices
        .lock()
        .unwrap()
        .push("[job 2 exited with code 1 after 9 s: `cargo test`]".into());
    let turns = vec![turn::text("all done"), turn::text("saw the failure")];
    let (session, fired) = run(&agent(&script, false, turns)).await;
    let texts = user_texts(&session);
    let last = texts.last().unwrap();
    assert!(last.1.starts_with("[job 2 exited"), "{texts:?}");
    assert_eq!(session.messages.last().unwrap().role, Role::Assistant);
    assert_eq!(fired, vec![(BackgroundReason::Finished, last.1.clone())]);
}

#[tokio::test]
async fn running_work_is_reminded_once_then_the_run_ends() {
    let script = Arc::new(Script::default());
    script.running.lock().unwrap().push("job 3".into());
    let turns = vec![turn::text("done"), turn::text("ending anyway")];
    let (session, fired) = run(&agent(&script, false, turns)).await;
    let reminders: Vec<_> = user_texts(&session)
        .into_iter()
        .filter(|(_, t)| t.starts_with("[background jobs]"))
        .collect();
    assert_eq!(reminders.len(), 1, "{reminders:?}");
    assert_eq!(fired.len(), 1);
    assert_eq!(fired[0].0, BackgroundReason::StillRunning);
    // Two model turns: the reply, and the reply to the reminder; nothing after.
    assert_eq!(
        session
            .messages
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .count(),
        2
    );
}

#[tokio::test]
async fn work_started_after_the_reminder_does_not_earn_a_second_one() {
    // The double fire the background A/B found: reminded, the model started (or waited on) a job
    // and ended again — and was reminded again. Once per run: the second stop ends it.
    let script = Arc::new(Script::default());
    script.running.lock().unwrap().push("job 3".into());
    let turns = vec![
        turn::text("done"),
        turn::tool_call("s", "start", "{}"),
        turn::text("ending now"),
    ];
    let (session, fired) = run(&agent(&script, false, turns)).await;
    let reminders = user_texts(&session)
        .into_iter()
        .filter(|(_, t)| t.starts_with("[background jobs]"))
        .count();
    assert_eq!(reminders, 1);
    assert_eq!(fired.len(), 1);
    assert_eq!(session.messages.last().unwrap().role, Role::Assistant);
    // The source still has unreminded work: a later run (a `serve` prompt) is told of it.
    assert!(script.stop_reminder().is_some());
}

#[tokio::test]
async fn without_background_work_a_stop_is_a_stop() {
    let script = Arc::new(Script::default());
    let (session, fired) = run(&agent(&script, false, vec![turn::text("done")])).await;
    assert!(fired.is_empty());
    assert_eq!(session.messages.len(), 2);
}

#[test]
fn notices_are_not_carried_as_user_requests() {
    let messages = vec![
        crate::Message::user("fix the parser"),
        crate::Message::user("[job 1 exited with code 0 after 3 s: `make`]\nok"),
        crate::Message::user("[background jobs] Still running: job 2"),
    ];
    let carried = crate::compaction::merge_user_requests(&[], &messages);
    assert_eq!(carried, vec!["fix the parser".to_string()]);
}

#[test]
fn the_carry_block_renders_live_state_and_is_stripped_before_resummarizing() {
    let script = Arc::new(Script::default());
    script
        .running
        .lock()
        .unwrap()
        .push("job 4 `npm run dev`".into());
    let sources: Vec<Arc<dyn Background>> = vec![script.clone()];
    let block = crate::background::format_carry(&sources).unwrap();
    assert!(block.contains("<background-jobs>") && block.contains("npm run dev"));
    // An echo the summarizer left in its prose (the host never writes the block into the text).
    let summary = format!(
        "{}\n\nThe prose.\n\n{block}",
        crate::compaction::SUMMARY_MARKER
    );
    let prefix = vec![crate::Message::user(summary)];
    let body = crate::compaction::previous_summary(&prefix).unwrap();
    assert_eq!(body, "The prose.");
    script.running.lock().unwrap().clear();
    assert_eq!(crate::background::format_carry(&sources), None);
}

#[tokio::test]
async fn a_final_answer_on_the_last_step_with_a_job_running_ends_ok() {
    // Turn 1 starts a job (step 1); turn 2 is the final answer, on step 2 of 2. The stop reminder
    // would start a third step the limit forbids, so it is skipped: the run succeeded.
    let script = Arc::new(Script::default());
    let turns = vec![
        turn::tool_call("s", "start", "{}"),
        turn::text("all done"),
        turn::text("unreachable"),
    ];
    let agent = agent(&script, false, turns).with_max_steps(2);
    let mut session = Session::new();
    session.user("go");
    let mut fired = Vec::new();
    let res = agent
        .run_events(&mut session, |ev| {
            if let AgentEvent::Background { reason, .. } = ev {
                fired.push(reason);
            }
        })
        .await;
    assert!(res.is_ok(), "a final answer must not fail the run: {res:?}");
    assert!(fired.is_empty(), "{fired:?}");
    assert_eq!(session.messages.last().unwrap().role, Role::Assistant);
    // The reminder was not spent: a later run (a `serve` prompt) is still told of the job.
    assert!(script.stop_reminder().is_some());
}

#[tokio::test]
async fn a_pending_notice_at_the_step_limit_waits_for_the_next_run() {
    let script = Arc::new(Script::default());
    script
        .notices
        .lock()
        .unwrap()
        .push("[job 2 exited with code 0 after 9 s: `cargo test`]".into());
    let turns = vec![turn::text("all done"), turn::text("unreachable")];
    let agent = agent(&script, false, turns).with_max_steps(1);
    let mut session = Session::new();
    session.user("go");
    let res = agent.run_events(&mut session, |_| {}).await;
    assert!(res.is_ok(), "{res:?}");
    assert_eq!(session.messages.len(), 2, "no notice turn was added");
    assert!(
        script.take_notices().is_some(),
        "the notice is left for the next run, not consumed"
    );
}
