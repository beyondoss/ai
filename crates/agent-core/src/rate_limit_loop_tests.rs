//! Rate-limit retry in the real loop: the budget, cancellation during a wait, and that a retry after
//! partial output never runs a tool twice. Classification and hint parsing are unit-tested in
//! `rate_limit.rs`; the wire shapes end to end in `tests/rate_limit_socket.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::message::{ContentBlock, Role, StreamEvent};
use crate::mock::{MockTransport, turn};
use crate::rate_limit::{BUDGET_EXHAUSTED, RateLimitRetry};
use crate::tool::{Tool, ToolOutput, ToolRegistry};
use crate::{Agent, AgentEvent, CancellationToken, Error, Result, Session};

fn rate_limited(retry_after_ms: u64) -> Error {
    Error::Transport(format!(
        r#"provider stream error: {{"type":"error","error":{{"type":"tokens","code":"rate_limit_exceeded","headers":{{"retry-after-ms":"{retry_after_ms}"}},"message":"Rate limit reached on tokens per min (TPM). Please try again in {retry_after_ms}ms."}}}}"#
    ))
}

fn failing(err: impl Fn() -> Error, n: usize) -> Vec<Vec<Result<StreamEvent>>> {
    (0..n)
        .map(|_| vec![Ok(StreamEvent::MessageStart), Err(err())])
        .collect()
}

fn ok(turn: Vec<StreamEvent>) -> Vec<Result<StreamEvent>> {
    turn.into_iter().map(Ok).collect()
}

fn policy(budget_ms: u64) -> RateLimitRetry {
    RateLimitRetry {
        budget: Duration::from_millis(budget_ms),
        base_backoff: Duration::from_millis(1),
        max_wait: Duration::from_secs(60),
    }
}

/// Run to completion, returning the result, every `Retry` event's `(attempt, rate_limited)`, and the
/// session.
async fn run(agent: &Agent) -> (Result<()>, Vec<(u32, bool)>, Session) {
    let mut session = Session::new();
    session.user("hi");
    let mut retries = Vec::new();
    let result = agent
        .run_events(&mut session, |ev| {
            if let AgentEvent::Retry {
                attempt,
                rate_limited,
                ..
            } = ev
            {
                retries.push((attempt, rate_limited));
            }
        })
        .await;
    (result, retries, session)
}

#[tokio::test]
async fn anthropic_rate_limit_and_overload_shapes_retry_past_the_transient_count() {
    // Six in-stream failures, alternating Anthropic's two capacity errors — twice the 3 retries any
    // other transient error gets — then success.
    let mut turns = Vec::new();
    for i in 0..6 {
        let msg = if i % 2 == 0 {
            "provider stream error: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Number of request tokens has exceeded your per-minute rate limit\"}}"
        } else {
            "provider stream error: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}"
        };
        turns.push(vec![
            Ok(StreamEvent::MessageStart),
            Err(Error::Transport(msg.into())),
        ]);
    }
    turns.push(ok(turn::text("done")));
    let mock = Arc::new(MockTransport::scripted(turns));
    let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_rate_limit_retry(policy(10_000));

    let (result, retries, session) = run(&agent).await;
    result.unwrap();
    assert_eq!(mock.calls(), 7);
    assert_eq!(
        retries,
        (1..=6).map(|n| (n, true)).collect::<Vec<_>>(),
        "each capacity retry is announced, numbered within its class"
    );
    assert_eq!(
        session.messages.len(),
        2,
        "[user, assistant] — no error records"
    );
}

#[tokio::test]
async fn a_run_that_outlasts_the_budget_fails_with_a_clear_message() {
    // Rate limited forever (well, 100 times); each hint 40ms, budget 150ms.
    let mock = Arc::new(MockTransport::scripted(failing(|| rate_limited(40), 100)));
    let agent = Agent::new(mock.clone(), "gpt-5").with_rate_limit_retry(policy(150));

    let started = Instant::now();
    let (result, retries, _) = run(&agent).await;
    let elapsed = started.elapsed();
    let err = result.unwrap_err();
    let Error::Transport(msg) = &err else {
        panic!("expected a transport error, got {err:?}");
    };
    assert!(msg.starts_with(BUDGET_EXHAUSTED), "{msg}");
    assert!(msg.contains("(budget 150ms)"), "{msg}");
    assert!(
        msg.contains("rate_limit_exceeded"),
        "the last real error is kept: {msg}"
    );
    // 40ms waits under a 150ms budget: a handful of retries, then a refusal to start a wait that
    // would overrun it — never a wait past the budget.
    assert!((2..=4).contains(&retries.len()), "{retries:?}");
    assert_eq!(mock.calls(), retries.len() + 1);
    assert!(elapsed < Duration::from_millis(150 + 500), "{elapsed:?}");
    assert!(
        !crate::agent::is_retryable_mid_stream(&err),
        "a spent budget must not be retried again by an outer layer"
    );
}

#[tokio::test]
async fn a_zero_budget_fails_on_the_first_rate_limit() {
    let mock = Arc::new(MockTransport::scripted(failing(|| rate_limited(1), 3)));
    let agent = Agent::new(mock.clone(), "gpt-5").with_rate_limit_retry(policy(0));
    let (result, retries, _) = run(&agent).await;
    assert!(matches!(result, Err(Error::Transport(m)) if m.starts_with(BUDGET_EXHAUSTED)));
    assert!(retries.is_empty());
    assert_eq!(mock.calls(), 1);
}

#[tokio::test]
async fn other_transient_errors_keep_their_three_retry_cap() {
    let mock = Arc::new(MockTransport::scripted(failing(
        || Error::Transport("Anthropic stream ended before message_stop".into()),
        10,
    )));
    let agent = Agent::new(mock.clone(), "claude-opus-4-8").with_rate_limit_retry(policy(60_000));
    let (result, retries, _) = run(&agent).await;
    assert!(matches!(result, Err(Error::Transport(m)) if m.contains("stream ended")));
    assert_eq!(retries, vec![(1, false), (2, false), (3, false)]);
    assert_eq!(mock.calls(), 4);
}

#[tokio::test]
async fn cancel_during_a_rate_limit_wait_returns_promptly() {
    // A 30s hint: the run is parked in the wait when the cancel lands.
    let mock = Arc::new(MockTransport::scripted(failing(|| rate_limited(30_000), 2)));
    let agent = Agent::new(mock.clone(), "gpt-5").with_rate_limit_retry(policy(300_000));
    let mut session = Session::new();
    session.user("hi");
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run_events_cancellable(&mut session, |_| {}, cancel),
    )
    .await
    .expect("cancel must interrupt the wait, not sit out the 30s hint");
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(mock.calls(), 1, "no request after the cancel");
}

struct Counting(Arc<AtomicUsize>);

#[async_trait]
impl Tool for Counting {
    fn name(&self) -> &str {
        "count"
    }
    fn description(&self) -> &str {
        "counts its runs"
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn run(&self, _input: Value) -> std::result::Result<ToolOutput, crate::error::ToolError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok("counted".into())
    }
}

#[tokio::test]
async fn a_rate_limit_after_a_streamed_tool_call_never_runs_the_tool_twice() {
    // Attempt 1 streams a complete tool call, then dies on a rate limit before the turn ends. The
    // retry re-streams the turn from scratch; the tool must run once — for the turn that completed —
    // and the transcript must hold that one call, not the dead attempt's copy.
    let mut first = turn::tool_call("t1", "count", "{}");
    first.pop(); // drop the terminal stop: the stream dies instead
    let mut dying: Vec<Result<StreamEvent>> = first.into_iter().map(Ok).collect();
    dying.push(Err(rate_limited(5)));
    let turns = vec![
        dying,
        ok(turn::tool_call("t1", "count", "{}")),
        ok(turn::text("done")),
    ];
    let mock = Arc::new(MockTransport::scripted(turns));
    let runs = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(Counting(runs.clone())));
    let agent = Agent::new(mock.clone(), "gpt-5")
        .with_tools(tools)
        .with_rate_limit_retry(policy(10_000));

    let (result, retries, session) = run(&agent).await;
    result.unwrap();
    assert_eq!(retries, vec![(1, true)]);
    assert_eq!(runs.load(Ordering::SeqCst), 1, "the tool ran exactly once");
    let tool_uses = session
        .messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .flat_map(|m| m.content.iter())
        .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
        .count();
    assert_eq!(tool_uses, 1, "one tool call in the transcript");
    assert!(session.messages.iter().all(|m| m.error_message.is_none()));
}
