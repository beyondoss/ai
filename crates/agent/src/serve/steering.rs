//! `steer`/`follow_up` queueing: messages that arrive during a run, and are expanded before they are queued.

use crate::serve::*;

/// Expand an explicit `/skill:name` invocation first (its own prefix, so it can't collide with a
/// `/name` prompt template), then fall through to prompt-template expansion — a no-op on whichever
/// message reaches it unmatched. Shared by every command that queues raw user-authored text as a turn
/// or steering message (`prompt`, `steer`, `follow_up`, and `prompt`'s `streaming_behavior` variant) —
/// a message queued through `steer`/`follow_up` must get the same expansion a fresh `prompt` would, or
/// a `/skill:name`/`/name` invocation sent through one of those paths silently reaches the model
/// unexpanded instead of triggering the skill/template it names.
///
/// An MCP-served skill (`/skill:<server>:<name>`) is tried first, and is the one case that awaits: its
/// `SKILL.md` is fetched now, at the user's invocation, never prefetched (see `tools::mcp_skills`).
pub(super) async fn expand_message(
    message: &str,
    skills: &[crate::skills::Skill],
    prompt_templates: &[crate::prompts::PromptTemplate],
    mcp_skills: &crate::tools::mcp_skills::McpSkills,
) -> String {
    if let Some(expanded) = mcp_skills.expand_invocation(message, skills).await {
        return expanded;
    }
    let message = crate::skills::expand_if_skill_invocation(message, skills);
    crate::prompts::expand_if_slash(&message, prompt_templates)
}

/// Which steering lane a message queued during a run goes to.
#[derive(Clone, Copy)]
pub(super) enum SteerLane {
    /// Injected at the next tool turn.
    Steer,
    /// Held for the stop boundary.
    FollowUp,
}

/// How a message queued during a run is acknowledged: as the `steer`/`follow_up` command it was, or
/// as a `prompt` with `streaming_behavior` (whose ack also says `queued_as`).
#[derive(Clone, Copy)]
pub(super) enum SteerAck {
    Command(&'static str),
    Prompt,
}

/// One steer/follow_up that arrived during a run, in arrival order, until it can be queued.
pub(super) struct PendingSteer {
    pub(super) seq: u64,
    pub(super) id: Option<String>,
    pub(super) images: Vec<agent_core::ImageSource>,
    pub(super) lane: SteerLane,
    pub(super) ack: SteerAck,
    pub(super) state: PendingState,
}

pub(super) enum PendingState {
    /// Expanded, waiting only for the entries ahead of it.
    Ready(String),
    /// An MCP `/skill:` being fetched on its own task, which reports back by `seq`.
    Expanding(tokio::task::JoinHandle<()>),
    /// Fetched; the acting window opens only if (and when) the text is actually queued.
    Expanded(String, Option<crate::tools::mcp_skills::PendingActivation>),
}

/// What an expansion task reports: its `seq`, and its text plus the activation to apply when queued.
pub(super) type SteerExpanded = (
    u64,
    String,
    Option<crate::tools::mcp_skills::PendingActivation>,
);

/// The steers and follow-ups received during the current run, kept in arrival order.
///
/// An MCP-served `/skill:` must be fetched from its server before it can be queued — up to the
/// expansion timeout. That is never awaited in the busy loop (it would stop polling the run and
/// reading `abort`/`approve`): it runs on a task of its own, which reports back through a channel the
/// loop selects on. Everything here is bound to the run it arrived in:
///
/// - **Order.** Each message takes its place at receipt; one is queued (and acked) only once every
///   message ahead of it has been, so a plain steer sent after a `/skill:` never overtakes it.
/// - **Run end.** When the run ends — finished, aborted, or cancelled by `new_session`,
///   `switch_session`, `fork`, `clone` — [`Self::end_run`] cancels every expansion still in flight
///   (awaiting it, so it is gone before the next command runs) and acks it as not queued, with the
///   reason. Messages behind it are queued in order if the run finished; if it was cancelled, the
///   run already dropped its steer and follow-up lanes (`clear_run_scoped`), so they are acked as
///   not queued too — run-scoped, like every steer the cancellation dropped — and an expansion that
///   had finished but was still waiting its turn activates nothing.
/// - **No activation across sessions.** An expansion opens no acting window itself; the skill is
///   activated in the session's view only when its text is queued — in the run, and so the session,
///   it came from.
pub(super) struct PendingSteers {
    pub(super) queue: std::collections::VecDeque<PendingSteer>,
    pub(super) next_seq: u64,
    pub(super) done_tx: mpsc::UnboundedSender<SteerExpanded>,
}

impl PendingSteers {
    pub(super) fn new(done_tx: mpsc::UnboundedSender<SteerExpanded>) -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
            next_seq: 0,
            done_tx,
        }
    }

    /// Take a message's place in line; queue whatever is now ready, in order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn receive(
        &mut self,
        id: Option<String>,
        message: &str,
        images: Vec<agent_core::ImageSource>,
        lane: SteerLane,
        ack: SteerAck,
        skills: &[crate::skills::Skill],
        prompt_templates: &[crate::prompts::PromptTemplate],
        mcp_skills: &crate::tools::mcp_skills::McpSkills,
        steering: &agent_core::Steering,
        out_tx: &OutTx,
    ) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let state = if mcp_skills.claims_invocation(message, skills) {
            let (message, skills) = (message.to_string(), skills.to_vec());
            let (mcp, done_tx) = (mcp_skills.clone(), self.done_tx.clone());
            PendingState::Expanding(tokio::spawn(async move {
                let (text, activation) = mcp
                    .expand_invocation_deferred(&message, &skills)
                    .await
                    .unwrap_or((message, None));
                let _ = done_tx.send((seq, text, activation));
            }))
        } else {
            let text = crate::skills::expand_if_skill_invocation(message, skills);
            PendingState::Ready(crate::prompts::expand_if_slash(&text, prompt_templates))
        };
        self.queue.push_back(PendingSteer {
            seq,
            id,
            images,
            lane,
            ack,
            state,
        });
        self.flush(mcp_skills, steering, out_tx);
    }

    /// An expansion finished. One from an earlier run (cancelled, its slot gone) is ignored.
    pub(super) fn expanded(
        &mut self,
        done: SteerExpanded,
        mcp_skills: &crate::tools::mcp_skills::McpSkills,
        steering: &agent_core::Steering,
        out_tx: &OutTx,
    ) {
        let (seq, text, activation) = done;
        if let Some(entry) = self.queue.iter_mut().find(|e| e.seq == seq)
            && matches!(entry.state, PendingState::Expanding(_))
        {
            entry.state = PendingState::Expanded(text, activation);
        }
        self.flush(mcp_skills, steering, out_tx);
    }

    /// Queue (and ack) every message at the front that is ready, stopping at the first that isn't.
    pub(super) fn flush(
        &mut self,
        mcp_skills: &crate::tools::mcp_skills::McpSkills,
        steering: &agent_core::Steering,
        out_tx: &OutTx,
    ) {
        while self
            .queue
            .front()
            .is_some_and(|e| !matches!(e.state, PendingState::Expanding(_)))
        {
            let Some(entry) = self.queue.pop_front() else {
                break;
            };
            let (text, activation) = match entry.state {
                PendingState::Ready(text) => (text, None),
                PendingState::Expanded(text, activation) => (text, activation),
                PendingState::Expanding(_) => unreachable!("checked above"),
            };
            let queued = queue_steer(
                entry.id,
                text,
                entry.images,
                entry.lane,
                entry.ack,
                steering,
                out_tx,
            );
            if queued && let Some(activation) = activation {
                mcp_skills.apply_activation(activation);
            }
        }
    }

    /// The run is over: cancel what is still being fetched (and wait until it is gone), ack it as not
    /// queued, and queue the rest in order — or, when the run was `cancelled` (its lanes already
    /// cleared), ack the rest as not queued too.
    pub(super) async fn end_run(
        &mut self,
        cancelled: bool,
        mcp_skills: &crate::tools::mcp_skills::McpSkills,
        steering: &agent_core::Steering,
        out_tx: &OutTx,
    ) {
        while let Some(entry) = self.queue.pop_front() {
            let refuse = |entry: PendingSteer, why: &str| {
                let command = match entry.ack {
                    SteerAck::Command(command) => command,
                    SteerAck::Prompt => "prompt",
                };
                let _ = out_tx.send(response(
                    entry.id,
                    command,
                    false,
                    Some(queue_content(steering)),
                    Some(why),
                ));
            };
            match entry.state {
                PendingState::Expanding(task) => {
                    task.abort();
                    let _ = task.await;
                    refuse(
                        PendingSteer {
                            state: PendingState::Ready(String::new()),
                            ..entry
                        },
                        STEER_RUN_ENDED,
                    );
                }
                _ if cancelled => refuse(entry, STEER_RUN_ABORTED),
                state => {
                    self.queue.push_front(PendingSteer { state, ..entry });
                    self.flush(mcp_skills, steering, out_tx);
                }
            }
        }
    }
}

/// Why a `/skill:` steer was not queued: its run ended while its skill was still being fetched.
pub(super) const STEER_RUN_ENDED: &str =
    "not queued: the run it was sent to ended before its skill finished loading";

/// Why a steer waiting behind a `/skill:` was not queued: its run was cancelled, which drops every
/// steer and follow-up that run had queued.
pub(super) const STEER_RUN_ABORTED: &str = "not queued: the run it was sent to was cancelled";

/// Push one steering message onto its lane and acknowledge it with what is now queued. Returns
/// whether it was queued (a full lane refuses it, and the ack says so).
pub(super) fn queue_steer(
    id: Option<String>,
    text: String,
    images: Vec<agent_core::ImageSource>,
    lane: SteerLane,
    ack: SteerAck,
    steering: &agent_core::Steering,
    out_tx: &OutTx,
) -> bool {
    let m = agent_core::SteeringMessage::new(text, images);
    let queued = match lane {
        SteerLane::Steer => steering.push_steer(m),
        SteerLane::FollowUp => steering.push(m),
    };
    let mut data = queue_content(steering);
    let command = match ack {
        SteerAck::Command(command) => command,
        SteerAck::Prompt => {
            if queued {
                data["queued_as"] = json!(match lane {
                    SteerLane::Steer => "steer",
                    SteerLane::FollowUp => "follow_up",
                });
            }
            "prompt"
        }
    };
    let _ = out_tx.send(response(
        id,
        command,
        queued,
        Some(data),
        (!queued).then_some(STEERING_QUEUE_FULL),
    ));
    queued
}

/// Parse a `prompt`'s optional `images` array into base64 image sources. Each entry is
/// `{media_type, data}`; malformed entries are skipped.
pub(super) fn parse_images(images: Option<&Value>) -> Vec<agent_core::ImageSource> {
    images
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|img| {
                    let media_type = img.get("media_type").and_then(Value::as_str)?;
                    let data = img.get("data").and_then(Value::as_str)?;
                    Some(agent_core::ImageSource::base64(media_type, data))
                })
                .collect()
        })
        .unwrap_or_default()
}
