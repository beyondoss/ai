//! Real-token sizing in the real loop: anchors recorded turn by turn, a threshold compaction that
//! cannot decline while the prompt is over the trigger, `tokens_after` on the prompt's own footing,
//! and the cached summary falling back to the re-render rather than cross a price tier. The pure
//! halves are unit-tested in `token_size.rs`, `session.rs` and `compaction.rs`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::compaction::{self, CompactionReason, SummarySource};
use crate::message::{ContentBlock, Message, Role, StreamEvent, TokenUsage};
use crate::mock::{MockTransport, turn};
use crate::token_size::{self, TokenFamily};
use crate::tool::{Tool, ToolOutput, ToolRegistry};
use crate::{Agent, AgentEvent, CancellationToken, CompactOutcome, Session};

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

/// `events` with a usage report reporting a `prompt`-token request, just before the stop.
fn with_prompt(mut events: Vec<StreamEvent>, prompt: u32) -> Vec<StreamEvent> {
    let at = events.len() - 1;
    events.insert(
        at,
        StreamEvent::Usage(TokenUsage {
            input_tokens: 10,
            cache_read_tokens: prompt - 10,
            output_tokens: 50,
            ..Default::default()
        }),
    );
    events
}

const HAIKU: &str = "claude-haiku-5-5";

fn summary_message() -> Message {
    Message::user(format!(
        "{}\n\nCompacted from 88000 tokens\n\n## Goal\nearlier work",
        compaction::SUMMARY_MARKER
    ))
}

fn tool_turn(i: usize, model: &str, output_tokens: u32) -> [Message; 2] {
    [
        Message::assistant(vec![ContentBlock::tool_use(
            format!("c{i}"),
            "noop",
            json!({}),
        )])
        .with_model_id(model)
        .with_usage(TokenUsage {
            output_tokens,
            ..Default::default()
        }),
        Message::tool_result(format!("c{i}"), "x".repeat(4_000), false),
    ]
}

async fn compact(agent: &Agent, session: &mut Session) -> (CompactOutcome, Vec<AgentEvent>) {
    let mut events = Vec::new();
    let outcome = agent
        .compact(
            session,
            CompactionReason::Threshold,
            &CancellationToken::new(),
            &mut |ev| events.push(ev),
            None,
        )
        .await
        .expect("compact");
    (outcome, events)
}

#[tokio::test]
async fn the_loop_stamps_each_reply_with_its_measured_turn() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(Noop));
    let turns = vec![
        with_prompt(turn::tool_call("a", "noop", "{}"), 3_000),
        with_prompt(turn::tool_call("b", "noop", "{}"), 5_000),
        with_prompt(turn::text("done"), 5_600),
    ];
    let agent = Agent::new(Arc::new(MockTransport::new(turns)), HAIKU).with_tools(tools);
    let mut session = Session::new();
    session.user("go");
    agent.run_events(&mut session, |_| {}).await.unwrap();
    let measured: Vec<Option<u32>> = session
        .messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .map(|m| m.prior_turn_tokens)
        .collect();
    // Each reply's request less the one before it: the previous reply and its tool result.
    assert_eq!(measured, vec![None, Some(2_000), Some(600)]);
    // Those turns are now sized exactly.
    let sizes = token_size::message_sizes(&session.messages, TokenFamily::Claude, Some(HAIKU));
    assert_eq!(sizes[1] + sizes[2], 2_000);
    assert_eq!(sizes[3] + sizes[4], 600);
}

/// The decline-and-grow failure, end to end: right after a compaction a turn with 56K of hidden
/// thinking sits next to the summary. chars/4 sized it at about 10 tokens, so the cut kept
/// everything, `compact` returned `AlreadyCompacted`, and the prompt grew past the trigger (one eval
/// run: 89K → 111K over 13 requests, past Haiku's 100K price tier). It must compact, past the turn,
/// and land under the trigger.
#[tokio::test]
async fn a_threshold_compaction_over_the_trigger_cuts_past_a_huge_thinking_turn() {
    let mut messages = vec![summary_message()];
    messages.extend(tool_turn(0, HAIKU, 56_460));
    for i in 1..6 {
        messages.extend(tool_turn(i, HAIKU, 300));
    }
    let mut session = Session::new();
    session.messages = Arc::new(messages);
    session.last_input_tokens = 95_000;
    session.last_usage_message_count = session.messages.len();
    let agent = Agent::new(
        Arc::new(MockTransport::new(vec![turn::text(
            "## Goal\nfresh checkpoint",
        )])),
        HAIKU,
    );
    let (outcome, events) = compact(&agent, &mut session).await;
    assert_eq!(outcome, CompactOutcome::Compacted);
    let (first_kept, tokens_after) = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::Compacted {
                first_kept,
                tokens_after,
                ..
            } => Some((*first_kept, *tokens_after)),
            _ => None,
        })
        .expect("a Compacted event");
    assert!(
        first_kept >= 3,
        "the 56K turn must be summarized, got first_kept={first_kept}"
    );
    assert!(
        tokens_after < 90_000,
        "must land under the trigger: {tokens_after}"
    );
    // The huge turn is gone from the live conversation.
    assert!(
        session
            .messages
            .iter()
            .all(|m| m.usage.is_none_or(|u| u.output_tokens < 56_000))
    );
}

#[tokio::test]
async fn tokens_after_is_the_live_prompt_less_the_real_prefix_plus_the_summary() {
    // Every turn measured: real sizes 6K per turn, whatever its characters say.
    let mut messages = vec![Message::user("the task")];
    for i in 0..8 {
        let mut t = tool_turn(i, HAIKU, 500);
        if i > 0 {
            t[0].prior_turn_tokens = Some(6_000);
        }
        messages.extend(t);
    }
    let mut session = Session::new();
    session.messages = Arc::new(messages);
    // The system prompt and tools: 3,400 tokens no message accounts for.
    let sizes = token_size::message_sizes(&session.messages, TokenFamily::Claude, Some(HAIKU));
    session.last_input_tokens = 3_400 + token_size::total(&sizes[..15]);
    session.last_usage_message_count = 15;
    let agent = Agent::new(
        Arc::new(MockTransport::new(vec![turn::text("## Goal\ncheckpoint")])),
        HAIKU,
    )
    .with_compaction(crate::CompactionConfig {
        keep_recent_tokens: 13_000,
        ..crate::CompactionConfig::for_model(HAIKU)
    });
    let live = compaction::live_tokens(&session, &sizes, TokenFamily::Claude);
    let (outcome, events) = compact(&agent, &mut session).await;
    assert_eq!(outcome, CompactOutcome::Compacted);
    let (first_kept, tokens_after) = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::Compacted {
                first_kept,
                tokens_after,
                ..
            } => Some((*first_kept, *tokens_after)),
            _ => None,
        })
        .unwrap();
    let expected = live - token_size::total(&sizes[..first_kept])
        + token_size::message_tokens(&session.messages[0], TokenFamily::Claude);
    assert_eq!(tokens_after, expected);
    // System prompt and tools included: more than the post-compaction messages alone.
    let after = token_size::message_sizes(&session.messages, TokenFamily::Claude, Some(HAIKU));
    assert!(
        tokens_after >= token_size::total(&after) + 3_000,
        "{tokens_after}"
    );
}

#[tokio::test]
async fn a_cached_summary_over_the_price_tier_falls_back_to_the_rerender() {
    let build = |price_tier| {
        let mut messages = vec![Message::user("the task")];
        for i in 0..6 {
            messages.extend(tool_turn(i, HAIKU, 400));
        }
        let mut session = Session::new();
        session.messages = Arc::new(messages);
        // A live prompt over Haiku's 100K tier (large tool definitions, say); the re-render of this
        // short transcript is a few thousand tokens.
        session.last_input_tokens = 104_000;
        session.last_usage_message_count = session.messages.len();
        let mock = Arc::new(MockTransport::new(vec![
            turn::text("## Goal\ncheckpoint"),
            turn::text("## Goal\ncheckpoint"),
        ]));
        let agent = Agent::new(mock.clone(), HAIKU).with_compaction(crate::CompactionConfig {
            keep_recent_tokens: 1,
            price_tier,
            ..crate::CompactionConfig::for_model(HAIKU)
        });
        (agent, session, mock)
    };
    let source = |events: &[AgentEvent]| {
        events.iter().find_map(|e| match e {
            AgentEvent::Compacted { summary_source, .. } => Some(*summary_source),
            _ => None,
        })
    };
    let (agent, mut session, mock) = build(Some(100_000));
    let (_, events) = compact(&agent, &mut session).await;
    assert_eq!(source(&events), Some(SummarySource::Rerender));
    assert_eq!(mock.calls(), 1, "no cached call was sent");
    assert_eq!(
        mock.requests()[0].system.as_deref(),
        Some(compaction::SUMMARY_SYSTEM),
        "the one request is the re-render"
    );
    // Without a tier the cached call goes out as before.
    let (agent, mut session, _) = build(None);
    let (_, events) = compact(&agent, &mut session).await;
    assert_eq!(source(&events), Some(SummarySource::Cached));
}
