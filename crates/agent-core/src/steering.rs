//! Steering — injecting user input into a run in flight.
//!
//! A [`Steering`] handle holds three shared queues, distinguished by *when* the loop injects them:
//!
//! - **steer** ([`push_steer`](Steering::push_steer)) — injected *mid-run*, between tool-executing
//!   turns, folded onto the same tool-results user turn. This is how a client redirects a busy agent
//!   ("actually, also handle X") without waiting for it to stop.
//! - **follow-up** ([`push`](Steering::push)) — injected only at a *would-stop* boundary (the model
//!   ended its turn without asking for tools), as a fresh user turn. This is "keep going / now do the
//!   next thing" once the current work is done.
//! - **next-turn** ([`push_next_turn`](Steering::push_next_turn)) — injected before a *fresh* run's
//!   very first request, folded onto the same user turn as the prompt that starts it. This is how a
//!   message queued while the agent is idle (no run in flight for either of the other two lanes to
//!   attach to) still lands on turn 1 of whatever prompt comes next, rather than waiting for a tool
//!   round-trip or a stop boundary that may never arrive. A beyond-only capability, not a port of pi:
//!   `nextTurnQueue`/`executeTurn` exist only in the dead, unshipped
//!   `packages/agent/src/harness/agent-harness.ts` (zero references from `packages/coding-agent`), not
//!   anywhere in pi's real product. The closest thing pi's real product has —
//!   `AgentSession`/`Agent.continue()`'s post-run loop (`packages/coding-agent/src/core/agent-session.ts`'s
//!   `_handlePostAgentRun`, draining `steerQueue`/`followUpQueue` via `hasQueuedMessages()`) — only
//!   picks up messages queued *during* the run that just ended, run as a separate follow-up turn; a
//!   plain `AgentSession.prompt()` call on an idle session never consults either queue at all, so
//!   content queued before a fresh prompt has no mechanism to ride along with it. This lane exists to
//!   close that gap for beyond specifically.
//!
//! Both the steer and follow-up injection points place messages where the previous message is the
//! assistant's, so a pushed user turn never lands next to another user turn (which the wire would
//! reject); the next-turn lane instead folds into the prompt's own user turn for the same reason. The
//! first two lanes mirror pi's real, separate `steerQueue`/`followUpQueue`; the next-turn lane is
//! beyond's own addition, with no real-pi-product equivalent (see above).
//!
//! A fourth, independent signal lives here too: [`request_stop`](Steering::request_stop) — a graceful,
//! host-initiated "stop after the current turn" request, mirroring pi's `shouldStopAfterTurn`. Unlike
//! cancellation (which drops an in-flight future and can abandon a tool mid-execution), this is checked
//! only at a turn boundary, after that turn's tool calls (if any) have already completed and their
//! results are durably committed — so it never leaves an orphaned `tool_use` behind. It's a flag, not a
//! queue: a second request before the first is observed is indistinguishable from one.
//!
//! A fifth and sixth setting — [`QueueMode`], one per lane — govern how much of the steer/follow-up
//! lanes a single drain point consumes: `All` (the historical behavior here) folds everything queued
//! into one injection; `OneAtATime` (pi's `PendingMessageQueue` default) takes only the oldest message,
//! leaving the rest queued for the *next* drain point, so several quick messages from a client land as
//! separate turns instead of one merged one. The steer and follow-up lanes carry **independent**
//! `QueueMode` settings (matching pi's own separate `steeringMode`/`followUpMode`) — a client may want,
//! say, every mid-run redirect folded together while follow-ups still land one at a time, or vice versa.
//! The next-turn lane has no `QueueMode` of its own — a drain always takes everything queued; there's
//! no "one at a time" reading of "prepended to the next prompt" (this lane has no real pi-product
//! analog to begin with — see the module doc comment above).
//!
//! **Deliberate gap (Task #29, pi-parity, low priority)**: pi's `emitQueueUpdate()` fires synchronously
//! on every `steer`/follow-up/next-turn push, notifying a subscriber the instant a message lands in a
//! queue. `Steering` here has no equivalent push-time observer/subscribe hook — left out on purpose
//! rather than added as a token gesture. Every caller that pushes onto a lane (`push`/`push_steer`/
//! `push_next_turn`) is, in this codebase's actual shape, the same RPC layer (`serve.rs`'s `prompt`/
//! `steer` handlers) that already knows the instant its own push call returns — there is no *other*
//! observer anywhere in the process that doesn't already have that information some other way. Adding a
//! callback field here would mean either a second, redundant notification path for the one caller that
//! doesn't need it, or a new cross-cutting registration API (`Steering::on_queued(...)`) with no current
//! consumer to validate its shape against — exactly the "awkward API churn for a hypothetical" this
//! crate's own minimal-abstraction bias (see the workspace root `CLAUDE.md`) argues against. If a future
//! caller genuinely needs to observe a queue push from *outside* the code that performed it (a second
//! process, a different task), that's a concrete requirement this doc comment's reasoning no longer
//! covers — revisit then, with a real shape to design against instead of a speculative one.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::message::ImageSource;
use crate::models::ThinkingLevel;
use crate::tool::ToolRegistry;

/// A queued steer/follow-up message: text plus optional image attachments — the same shape a fresh
/// `prompt` accepts (`serve.rs`'s own `images` field), so a message routed through `steer`/`follow_up`
/// isn't a lesser channel that silently drops attachments a `prompt` would have kept. `From<&str>`/
/// `From<String>` build a text-only message, so every existing `push`/`push_steer` call site that
/// only ever handled plain text keeps compiling unchanged; construct the struct directly (or via
/// [`SteeringMessage::new`]) to attach images.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SteeringMessage {
    pub text: String,
    pub images: Vec<ImageSource>,
}

impl SteeringMessage {
    /// A message with both text and image attachments.
    pub fn new(text: impl Into<String>, images: Vec<ImageSource>) -> Self {
        Self {
            text: text.into(),
            images,
        }
    }
}

impl From<&str> for SteeringMessage {
    fn from(text: &str) -> Self {
        Self {
            text: text.to_string(),
            images: Vec::new(),
        }
    }
}

impl From<String> for SteeringMessage {
    fn from(text: String) -> Self {
        Self {
            text,
            images: Vec::new(),
        }
    }
}

// Lets existing text-only test assertions (`assert_eq!(s.drain_steer(), vec!["x".to_string()])`)
// keep working unchanged: `Vec<SteeringMessage> == Vec<String>` needs `SteeringMessage: PartialEq<String>`.
// Compares text only — a caller asserting against a bare string is deliberately not asserting
// anything about images.
impl PartialEq<String> for SteeringMessage {
    fn eq(&self, other: &String) -> bool {
        &self.text == other
    }
}

type Queue = Arc<Mutex<VecDeque<SteeringMessage>>>;

/// The most messages any one lane will hold. Every push into these lanes originates, in this
/// codebase's real shape, at a *remote* RPC command (`serve.rs`'s `steer`/`follow_up`/`prompt`
/// handlers, reachable over WebSocket and UDS), and a [`SteeringMessage`] carries `images` — base64
/// payloads that can each be megabytes. The lanes, meanwhile, drain slowly by design: the default
/// [`QueueMode::OneAtATime`] takes exactly one message per drain point, and while no run is in flight
/// the steer lane has no drain point at all. A client that pushes faster than the agent drains — or
/// pushes at all while idle — therefore pins heap in the daemon for the life of the session, and
/// nothing but [`clear`](Steering::clear) ever gives it back. A cap is what turns that from unbounded
/// into bounded.
///
/// 64 is chosen to be far above any plausible legitimate depth and still small enough to bound the
/// worst case: these lanes hold *human-authored* redirects, drained roughly one per turn, so a lane
/// even a dozen deep already means the client is producing faster than the agent can ever consume.
/// A client that legitimately needs to hand the agent more than 64 pending instructions wants a
/// prompt, not 64 steers.
const MAX_QUEUED_PER_LANE: usize = 64;

/// Push onto a lane unless it is already at [`MAX_QUEUED_PER_LANE`], returning whether the message was
/// queued.
///
/// **Reject-newest, not drop-oldest.** The oldest message in a lane is the one the loop is about to
/// inject — the instruction the client has been waiting on longest. Evicting it to make room for a
/// newer one would silently reorder a client's own instruction stream (its later message lands, its
/// earlier one vanishes) and, under a sustained flood, would keep the head of the queue perpetually
/// churning so that *nothing* ever reaches the model. Refusing the newest instead keeps whatever is
/// queued a faithful, in-order prefix of what the client sent, and puts the loss at the tail — where
/// the client that caused it is still connected, still pushing, and gets `false` back to act on. This
/// is backpressure, not eviction.
///
/// Never silent: a refusal is logged at `warn`, because a dropped steer is a user's words not reaching
/// the model.
fn push_capped(q: &Queue, lane: &'static str, message: SteeringMessage) -> bool {
    let mut queue = lock(q);
    if queue.len() >= MAX_QUEUED_PER_LANE {
        tracing::warn!(
            lane,
            depth = queue.len(),
            cap = MAX_QUEUED_PER_LANE,
            "steering lane full; rejecting message (the queue is draining slower than it is being fed)"
        );
        return false;
    }
    queue.push_back(message);
    true
}

/// How much of a lane [`Steering::drain_steer`]/[`Steering::drain_at_stop`] consumes per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueueMode {
    /// Only the oldest queued message is injected per drain point — pi's `PendingMessageQueue`
    /// default. Several messages queued in quick succession land as separate turns, one at a time,
    /// rather than folded into a single injection.
    #[default]
    OneAtATime,
    /// Every message currently queued is injected at once, folded into a single turn — this crate's
    /// original (and still available) behavior.
    All,
}

impl std::str::FromStr for QueueMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "one_at_a_time" => Ok(QueueMode::OneAtATime),
            "all" => Ok(QueueMode::All),
            other => Err(format!(
                "invalid queue mode {other:?}; expected one of one_at_a_time/all"
            )),
        }
    }
}

/// A requested mid-run model switch (and, optionally, a new thinking budget) — see
/// [`Steering::request_model_switch`]. Mirrors pi's `prepareNextTurn`/`nextTurnSnapshot`: a host can
/// downgrade to a cheaper model once a run turns out not to need much firepower, or raise the model or
/// thinking budget once it turns out to need more, all without stopping and restarting the whole call.
///
/// Deliberately narrower than pi's snapshot, which can also swap the whole `context`/reasoning-effort:
/// this only ever changes what the *main conversational turns* target. Compaction/branch-summary calls
/// still use the `Agent`'s original, as-configured model — pi's own harness doesn't thread
/// `prepareNextTurn`'s override into those either (they run through a separate summarization path with
/// no `nextTurnSnapshot` awareness).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSwitch {
    pub model: String,
    /// `None` leaves the thinking budget as currently configured on the `Agent`; `Some(budget)` sets a
    /// new *raw* token budget for every subsequent turn of this run. Applied on top of
    /// [`Self::thinking_level`] when both are given — see
    /// [`Steering::request_model_switch_with_thinking_level`]'s doc comment for the exact order.
    pub thinking: Option<u32>,
    /// A tri-stated, portable reasoning-effort/thinking-depth override — `None` leaves it as currently
    /// configured (untouched by this switch at all), `Some(ThinkingLevel::Off)` **explicitly** disables
    /// thinking/reasoning (distinct from `None`'s "don't touch it"), `Some(level)` requests a new depth.
    /// Translated into whichever of the run's thinking budget / reasoning effort this (possibly also
    /// just-switched) model's shape actually wants via [`crate::models::thinking_for_level`] — the same
    /// translation `serve.rs`'s `set_model`/`cycle_model` RPC handlers already call for an idle-time
    /// switch, reused here rather than inventing a second, divergent path for a mid-run one. Mirrors
    /// pi's `nextTurnSnapshot.thinkingLevel` (`agent-loop.ts:220-238`), itself a full tri-state
    /// `ThinkingLevel` (`undefined`/`"off"`/a concrete level, `types.ts:129-136,289`).
    pub thinking_level: Option<ThinkingLevel>,
}

/// A cloneable handle to the shared steering queues. Clones share the same two lanes (and the stop
/// flag).
#[derive(Clone, Default)]
pub struct Steering {
    /// Injected mid-run, between tool turns.
    steer: Queue,
    /// Injected at a would-stop boundary.
    follow_up: Queue,
    /// Injected before a fresh run's very first request, folded onto that request's own prompt turn.
    next_turn: Queue,
    /// Set by [`request_stop`](Self::request_stop); consumed by the loop at the next turn boundary.
    stop_requested: Arc<AtomicBool>,
    /// How much of the steer lane [`drain_steer`](Self::drain_steer) consumes per call — see
    /// [`QueueMode`]. A setting, not per-run state: `clear()` deliberately leaves it untouched.
    steer_mode: Arc<Mutex<QueueMode>>,
    /// How much of the follow-up (plus any stranded steer) lane [`drain_at_stop`](Self::drain_at_stop)
    /// consumes per call — independent of `steer_mode`, matching pi's separate `followUpMode`.
    follow_up_mode: Arc<Mutex<QueueMode>>,
    /// Set by [`request_model_switch`](Self::request_model_switch); consumed by the loop at the next
    /// turn boundary (the same point `stop_requested` is checked).
    model_switch: Arc<Mutex<Option<ModelSwitch>>>,
    /// Set by [`request_tool_set`](Self::request_tool_set); consumed by the loop at the same turn
    /// boundary as `model_switch` (see that field's doc comment).
    tool_switch: Arc<Mutex<Option<ToolRegistry>>>,
}

impl Steering {
    /// Empty steering queues.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a **follow-up**: injected when the run next reaches a stop boundary. Accepts a plain
    /// string or a [`SteeringMessage`] (with image attachments).
    ///
    /// Returns `false` — and queues nothing — if the follow-up lane is already at
    /// [`MAX_QUEUED_PER_LANE`]; see [`push_capped`] for why the *newest* message is the one refused.
    pub fn push(&self, message: impl Into<SteeringMessage>) -> bool {
        push_capped(&self.follow_up, "follow_up", message.into())
    }

    /// Queue a **steer**: injected mid-run, at the next tool-results turn, to redirect a busy agent.
    /// Accepts a plain string or a [`SteeringMessage`] (with image attachments).
    ///
    /// Returns `false` — and queues nothing — if the steer lane is already full; see
    /// [`push`](Self::push).
    pub fn push_steer(&self, message: impl Into<SteeringMessage>) -> bool {
        push_capped(&self.steer, "steer", message.into())
    }

    /// Queue a **next-turn** message: folded onto the very first request of whatever run comes next,
    /// before that run's own first model call — pi's `nextTurn()`. Unlike `push`/`push_steer`, this is
    /// meant to be called while the agent is idle (nothing else queued has a run in flight to attach
    /// to), though nothing here enforces that — a caller can queue one mid-run too, and it simply waits
    /// for the *next* fresh run to start. Accepts a plain string or a [`SteeringMessage`] (with image
    /// attachments).
    ///
    /// Returns `false` — and queues nothing — if the next-turn lane is already full; see
    /// [`push`](Self::push). This lane is the one most likely to be pushed while nothing is draining
    /// it (that's its whole purpose), so its cap is the one doing the most work.
    pub fn push_next_turn(&self, message: impl Into<SteeringMessage>) -> bool {
        push_capped(&self.next_turn, "next_turn", message.into())
    }

    /// Whether any lane currently holds any messages.
    pub fn is_empty(&self) -> bool {
        lock(&self.steer).is_empty()
            && lock(&self.follow_up).is_empty()
            && lock(&self.next_turn).is_empty()
    }

    /// Total queued messages across all three lanes — pi's `pendingMessageCount` (`get_state`'s
    /// queue-depth field). A peek, not a drain: calling this doesn't consume anything.
    pub fn pending_count(&self) -> usize {
        lock(&self.steer).len() + lock(&self.follow_up).len() + lock(&self.next_turn).len()
    }

    /// The steer lane's queued message text, oldest first — Fix 5 (pi-parity gap): [`pending_count`]
    /// alone told a client *how many* messages were queued but never *what*, unlike pi's own
    /// `AgentSessionEvent.queue_update` (`agent-session.ts`), which carries the actual `steering`/
    /// `followUp` string arrays. A peek, not a drain, like [`pending_count`](Self::pending_count) —
    /// calling this doesn't consume anything. Text only, matching `pending_count`'s own "depth, not
    /// attachments" scope — a caller that also needs image attachments has no lighter-weight source for
    /// those than draining the lane outright, which this deliberately doesn't do.
    pub fn steer_texts(&self) -> Vec<String> {
        lock(&self.steer).iter().map(|m| m.text.clone()).collect()
    }

    /// [`steer_texts`](Self::steer_texts), for the follow-up lane.
    pub fn follow_up_texts(&self) -> Vec<String> {
        lock(&self.follow_up)
            .iter()
            .map(|m| m.text.clone())
            .collect()
    }

    /// Set how much of the steer lane a single [`drain_steer`](Self::drain_steer) consumes (see
    /// [`QueueMode`]). Takes effect on the next drain — a call already past its `drain_steer` this turn
    /// is unaffected. Independent of [`set_follow_up_mode`](Self::set_follow_up_mode).
    pub fn set_steering_mode(&self, mode: QueueMode) {
        *lock_mode(&self.steer_mode) = mode;
    }

    /// The current steer-lane [`QueueMode`] (`OneAtATime` by default, matching pi).
    pub fn steering_mode(&self) -> QueueMode {
        *lock_mode(&self.steer_mode)
    }

    /// Set how much of the follow-up (plus stranded-steer) lane a single
    /// [`drain_at_stop`](Self::drain_at_stop) consumes (see [`QueueMode`]). Independent of
    /// [`set_steering_mode`](Self::set_steering_mode).
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        *lock_mode(&self.follow_up_mode) = mode;
    }

    /// The current follow-up-lane [`QueueMode`] (`OneAtATime` by default, matching pi).
    pub fn follow_up_mode(&self) -> QueueMode {
        *lock_mode(&self.follow_up_mode)
    }

    /// Take the queued mid-run steer messages: everything in `All` mode, or just the oldest in
    /// `OneAtATime` mode — the rest stays queued for the next mid-run injection point. Governed by
    /// [`steering_mode`](Self::steering_mode).
    pub(crate) fn drain_steer(&self) -> Vec<SteeringMessage> {
        match self.steering_mode() {
            QueueMode::All => lock(&self.steer).drain(..).collect(),
            QueueMode::OneAtATime => lock(&self.steer).pop_front().into_iter().collect(),
        }
    }

    /// Take what's queued for a stop boundary: any steer messages that were queued but never reached a
    /// mid-run injection point (e.g. a turn with no tool calls) — the primary lane at a stop boundary,
    /// since a steer is an urgent redirect a client is waiting on — plus the follow-up lane, so nothing
    /// queued there is stranded either. Governed by [`follow_up_mode`](Self::follow_up_mode) — "how much
    /// to inject when the agent is about to stop" is one decision, whether the pending work originated
    /// in the follow-up lane or was left over in the steer lane. In `All` mode that's everything in both
    /// lanes, steer first; in `OneAtATime` mode it's just the single oldest message — the steer lane
    /// first, falling back to the follow-up lane only if it's empty. Matches pi's own priority
    /// (`agent-loop.ts`'s `getSteeringMessages()` is polled at every stop point before
    /// `getFollowUpMessages()` is ever reached) — the inverse order here used to leave an urgent steer
    /// waiting behind a lower-priority follow-up that happened to be queued first.
    pub(crate) fn drain_at_stop(&self) -> Vec<SteeringMessage> {
        match self.follow_up_mode() {
            QueueMode::All => {
                let mut out: Vec<SteeringMessage> = lock(&self.steer).drain(..).collect();
                out.extend(lock(&self.follow_up).drain(..));
                out
            }
            QueueMode::OneAtATime => {
                if let Some(m) = lock(&self.steer).pop_front() {
                    vec![m]
                } else if let Some(m) = lock(&self.follow_up).pop_front() {
                    vec![m]
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Take every queued **next-turn** message (this crate's own lane — see the module doc comment for
    /// why it has no real pi-product analog). Unlike
    /// [`drain_steer`](Self::drain_steer)/[`drain_at_stop`](Self::drain_at_stop), this always takes the
    /// whole lane; there's no `QueueMode` for it (see the module doc comment for why).
    pub(crate) fn drain_next_turn(&self) -> Vec<SteeringMessage> {
        lock(&self.next_turn).drain(..).collect()
    }

    /// Drop everything queued in every lane without returning it — for a caller (`new_session`/
    /// `switch_session`/`fork`/`switch_branch`) that's about to swap in a different session's
    /// conversation, so a message queued for the *old* session's next turn can't leak into the newly
    /// switched-to one. Also clears any pending stop request, for the same reason: a graceful-stop
    /// request aimed at the old session's run must not cut short a different session's next one.
    ///
    /// For a run that's ending via **cancellation** rather than a session swap, use
    /// [`clear_run_scoped`](Self::clear_run_scoped) instead — this method also drops the next-turn
    /// lane, which a mere cancellation must not do (see that method's doc comment for why).
    pub fn clear(&self) {
        lock(&self.steer).clear();
        lock(&self.follow_up).clear();
        lock(&self.next_turn).clear();
        self.stop_requested.store(false, Ordering::Relaxed);
        *lock_opt(&self.model_switch) = None;
        // Taken out and bound to a `let`, not assigned away as `*guard = None`: the discarded
        // registry's destructor must not run while we hold the lock — see
        // [`request_tool_set`](Self::request_tool_set).
        let _discarded_tools = self.take_tool_switch();
    }

    /// Drop the steer and follow-up lanes, plus any pending stop request, mid-run model switch, and
    /// mid-run tool-set switch — but, unlike [`clear`](Self::clear), leave the **next-turn** lane
    /// untouched. For a run that's
    /// ending because of *cancellation*, not because a different session is about to take over (see
    /// [`clear`](Self::clear) for that case).
    ///
    /// A deliberate beyond design choice, not a port of pi: dropping the steer/follow-up lanes here is
    /// a conservative safety default (a stale queued message surviving an abort is arguably worse than
    /// losing it), not a mirror of pi's real headless behavior. pi's actual headless/RPC abort path
    /// (`AgentSession.abort()`, `packages/coding-agent/src/core/agent-session.ts:1449-1453`, called
    /// from `packages/coding-agent/src/modes/rpc/rpc-mode.ts:424-426`) does *not* clear queued steer/
    /// follow-up messages at all — only the TUI's own `clearQueue()` does, and beyond is headless, not
    /// a TUI. (The unused `packages/agent/src/harness/agent-harness.ts`'s `AgentHarness.abort()`
    /// (`agent-harness.ts:970-997`) also clears only `steerQueue`/`followUpQueue`, which is where this
    /// method's shape actually came from, but that harness is dead code — zero references from
    /// `packages/coding-agent` — not pi's real product.) Whether beyond's stricter behavior is the
    /// right default, or whether it should instead match pi's real headless behavior and preserve a
    /// queued message across cancellation, is an open question this comment flags rather than settles.
    ///
    /// The next-turn lane is deliberately left alone regardless, because a message queued via
    /// `push_next_turn` is meant to survive into whatever prompt comes next, aborted or not, not be
    /// silently dropped by the very cancellation it was queued to outlive. Used by every
    /// cancellation exit path in [`crate::Agent::run_events_steered`]; a genuine session swap
    /// (`new_session`/`switch_session`/`fork`/`switch_branch`) still wants the stronger [`clear`](
    /// Self::clear), since a message queued for the *old* session's next turn must not leak into a
    /// newly switched-to one either.
    pub fn clear_run_scoped(&self) {
        lock(&self.steer).clear();
        lock(&self.follow_up).clear();
        self.stop_requested.store(false, Ordering::Relaxed);
        *lock_opt(&self.model_switch) = None;
        // As in [`clear`](Self::clear): take it out, drop it after the guard is gone.
        let _discarded_tools = self.take_tool_switch();
    }

    /// Request that the run stop gracefully at the next turn boundary — after the current turn's tool
    /// calls (if any) finish and their results are committed, but before another model call starts.
    /// Idempotent: a second call before the first is observed has no additional effect.
    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::Relaxed);
    }

    /// Consume and return the pending stop request, if any. Each request is observed at most once.
    pub(crate) fn take_stop_requested(&self) -> bool {
        self.stop_requested.swap(false, Ordering::Relaxed)
    }

    /// Request that every subsequent turn of this run target `model` (and, if `thinking` is `Some`, a
    /// new thinking budget) — applied at the next turn boundary, the same point a graceful stop is
    /// checked. A second call before the first is observed replaces it outright (only the most recent
    /// request matters — there's no queue of switches to apply in sequence).
    pub fn request_model_switch(&self, model: impl Into<String>, thinking: Option<u32>) {
        self.request_model_switch_with_thinking_level(model, thinking, None);
    }

    /// Like [`request_model_switch`](Self::request_model_switch), but can also change the run's
    /// reasoning effort / thinking depth via a portable [`ThinkingLevel`] — see
    /// [`ModelSwitch::thinking_level`]'s doc comment for the tri-state semantics (no-change vs.
    /// explicitly-off vs. a specific level) and how it composes with a raw `thinking` budget when both
    /// are given (the level is applied first, translated via `models::thinking_for_level`; a `Some`
    /// `thinking` then overrides just the numeric budget on top, for a caller after an exact token
    /// count rather than a portable level). A second call before the first is observed replaces it
    /// outright, same as [`request_model_switch`](Self::request_model_switch) — there's no queue of
    /// switches to apply in sequence.
    pub fn request_model_switch_with_thinking_level(
        &self,
        model: impl Into<String>,
        thinking: Option<u32>,
        thinking_level: Option<ThinkingLevel>,
    ) {
        *lock_opt(&self.model_switch) = Some(ModelSwitch {
            model: model.into(),
            thinking,
            thinking_level,
        });
    }

    /// Consume and return the pending model switch, if any. Each request is observed at most once.
    pub(crate) fn take_model_switch(&self) -> Option<ModelSwitch> {
        lock_opt(&self.model_switch).take()
    }

    /// Request that every subsequent turn of this run advertise `tools` instead of whatever the
    /// `Agent` was originally configured with — applied at the same turn boundary a model switch is
    /// (see [`request_model_switch`](Self::request_model_switch)), so the change takes effect starting
    /// the very next turn rather than mid-turn. Mirrors the mechanism behind pi's real shipped
    /// `setActiveToolsByName` (`packages/coding-agent/src/core/agent-session.ts:840`, wired to the
    /// extension runtime's `setActiveTools` handler at line 2283, and called internally at line 2428):
    /// a host can add, remove, or swap a run's tool set without stopping and restarting the whole call.
    /// A second call before the first is observed replaces it outright — same "no queue of switches,
    /// only the most recent request matters" contract
    /// [`request_model_switch`](Self::request_model_switch) uses.
    ///
    /// The registry being *replaced* is deliberately moved out of the slot and dropped by this
    /// function's own scope, rather than written over with `*guard = Some(tools)`. An assignment
    /// through the guard drops the old value in place, still holding the lock — and a `ToolRegistry`
    /// is `Arc<dyn Tool>`s, so that runs arbitrary *host-supplied* destructors under our mutex. A host
    /// tool whose `Drop` blocks, takes another lock, or re-enters `Steering` would stall or deadlock
    /// every other lane behind it; one that panics would unwind out of the assignment with the slot
    /// dropped-but-not-yet-overwritten and the mutex poisoned, and [`lock_opt`]'s poison recovery
    /// hands that slot straight back to the next caller. Dropping after the guard is released costs
    /// nothing and makes both moot: the `let` binding outlives the temporary guard, so the old
    /// registry's destructor runs on an unlocked `Steering`.
    pub fn request_tool_set(&self, tools: ToolRegistry) {
        let _replaced_tools = lock_opt(&self.tool_switch).replace(tools);
    }

    /// Consume and return the pending tool-set switch, if any. Each request is observed at most once.
    pub(crate) fn take_tool_switch(&self) -> Option<ToolRegistry> {
        lock_opt(&self.tool_switch).take()
    }
}

/// Recover the data on a poisoned lock rather than propagating a panic into the loop.
fn lock(q: &Queue) -> std::sync::MutexGuard<'_, VecDeque<SteeringMessage>> {
    q.lock().unwrap_or_else(|e| e.into_inner())
}

/// Same recovery posture as [`lock`], for the [`QueueMode`] setting.
fn lock_mode(m: &Arc<Mutex<QueueMode>>) -> std::sync::MutexGuard<'_, QueueMode> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Same recovery posture as [`lock`], for a pending single-slot request (a [`ModelSwitch`] or a
/// replacement [`ToolRegistry`]) — both are "at most one pending, replaced outright by a later
/// request" state, just for different payload types.
fn lock_opt<T>(m: &Arc<Mutex<Option<T>>>) -> std::sync::MutexGuard<'_, Option<T>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follow_up_and_steer_are_separate_lanes() {
        let s = Steering::new();
        assert!(s.is_empty());
        s.push("follow");
        s.push_steer("steer");
        assert!(!s.is_empty());
        // The steer lane drains on its own; the follow-up stays until the stop boundary.
        assert_eq!(s.drain_steer(), vec!["steer".to_string()]);
        assert_eq!(s.drain_at_stop(), vec!["follow".to_string()]);
        assert!(s.is_empty());
    }

    #[test]
    fn steer_texts_and_follow_up_texts_report_content_not_just_a_count() {
        // Fix 5 (pi-parity gap): a client needs to see *what* is queued, not just how many messages.
        let s = Steering::new();
        assert_eq!(s.steer_texts(), Vec::<String>::new());
        assert_eq!(s.follow_up_texts(), Vec::<String>::new());

        s.push_steer("steer-1");
        s.push_steer("steer-2");
        s.push("follow-1");
        assert_eq!(s.steer_texts(), vec!["steer-1", "steer-2"]);
        assert_eq!(s.follow_up_texts(), vec!["follow-1"]);

        // A peek, not a drain — the messages are still there afterward, and still drain correctly.
        assert_eq!(s.steer_texts(), vec!["steer-1", "steer-2"]);
        assert_eq!(s.drain_steer(), vec!["steer-1".to_string()]);
        assert_eq!(s.steer_texts(), vec!["steer-2"]);
    }

    #[test]
    fn pending_count_sums_both_lanes_without_draining_either() {
        let s = Steering::new();
        assert_eq!(s.pending_count(), 0);
        s.push("follow-1");
        s.push_steer("steer-1");
        s.push_steer("steer-2");
        assert_eq!(s.pending_count(), 3);
        // A peek, not a drain: the messages are still there afterward.
        assert_eq!(s.pending_count(), 3);
        s.drain_steer(); // default OneAtATime mode: pops just the oldest steer message
        assert_eq!(s.pending_count(), 2, "draining steer must lower the count");
    }

    #[test]
    fn drain_at_stop_sweeps_stranded_steer_messages() {
        // A steer queued on a turn that never ran tools must still be injected at the stop boundary —
        // in `All` mode, in one combined drain, steer first (an urgent redirect takes priority over a
        // "keep going" follow-up). Governed by `follow_up_mode`, not `steering_mode` — "how much to
        // inject at a stop boundary" is one decision regardless of which lane it came from.
        let s = Steering::new();
        s.set_follow_up_mode(QueueMode::All);
        s.push_steer("stranded");
        s.push("follow");
        assert_eq!(
            s.drain_at_stop(),
            vec!["stranded".to_string(), "follow".to_string()]
        );
    }

    #[test]
    fn default_mode_is_one_at_a_time_matching_pi() {
        let s = Steering::new();
        assert_eq!(s.steering_mode(), QueueMode::OneAtATime);
        assert_eq!(s.follow_up_mode(), QueueMode::OneAtATime);
    }

    #[test]
    fn one_at_a_time_mode_drains_a_single_message_per_call_leaving_the_rest_queued() {
        let s = Steering::new();
        assert_eq!(s.follow_up_mode(), QueueMode::OneAtATime, "default");
        s.push("first");
        s.push("second");
        s.push("third");
        assert_eq!(s.drain_at_stop(), vec!["first".to_string()]);
        assert_eq!(s.drain_at_stop(), vec!["second".to_string()]);
        assert_eq!(s.drain_at_stop(), vec!["third".to_string()]);
        assert_eq!(s.drain_at_stop(), Vec::<String>::new(), "queue now empty");
    }

    #[test]
    fn one_at_a_time_mode_prefers_a_stranded_steer_over_follow_up_then_sweeps_follow_up_next() {
        // pi-parity fix: same priority order as `All`'s merge (steer first, follow-up after) — just one
        // message at a time instead of the whole lane at once. This used to be inverted (follow-up
        // drained first), the opposite of pi's own `agent-loop.ts` priority, which always polls the
        // steer queue before ever looking at follow-up.
        let s = Steering::new();
        s.push_steer("stranded");
        s.push("follow");
        assert_eq!(
            s.drain_at_stop(),
            vec!["stranded".to_string()],
            "the stranded, urgent steer message drains first"
        );
        assert_eq!(
            s.drain_at_stop(),
            vec!["follow".to_string()],
            "the follow-up is swept on the next stop-boundary drain"
        );
        assert_eq!(s.drain_at_stop(), Vec::<String>::new());
    }

    #[test]
    fn one_at_a_time_mode_applies_to_the_mid_run_steer_lane_too() {
        let s = Steering::new();
        s.push_steer("a");
        s.push_steer("b");
        assert_eq!(s.drain_steer(), vec!["a".to_string()]);
        assert_eq!(s.drain_steer(), vec!["b".to_string()]);
        assert_eq!(s.drain_steer(), Vec::<String>::new());
    }

    #[test]
    fn set_mode_takes_effect_immediately_and_is_shared_across_clones() {
        let a = Steering::new();
        let b = a.clone();
        a.set_follow_up_mode(QueueMode::All);
        assert_eq!(
            b.follow_up_mode(),
            QueueMode::All,
            "mode is shared, like the queues"
        );
        b.push("x");
        b.push("y");
        assert_eq!(a.drain_at_stop(), vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn steering_mode_and_follow_up_mode_are_independent() {
        // The whole point of the split: a client can fold every mid-run redirect together while
        // follow-ups still land one at a time, or vice versa — setting one must not touch the other.
        let s = Steering::new();
        s.set_steering_mode(QueueMode::All);
        assert_eq!(s.steering_mode(), QueueMode::All);
        assert_eq!(
            s.follow_up_mode(),
            QueueMode::OneAtATime,
            "follow-up mode must be untouched by a steering-mode change"
        );

        s.set_follow_up_mode(QueueMode::All);
        s.set_steering_mode(QueueMode::OneAtATime);
        assert_eq!(
            s.steering_mode(),
            QueueMode::OneAtATime,
            "steering mode must be untouched by a follow-up-mode change"
        );
        assert_eq!(s.follow_up_mode(), QueueMode::All);

        s.push_steer("a");
        s.push_steer("b");
        assert_eq!(
            s.drain_steer(),
            vec!["a".to_string()],
            "steer lane still drains one-at-a-time"
        );
        assert_eq!(
            s.drain_steer(),
            vec!["b".to_string()],
            "draining the leftover steer message too, so it can't strand into the next assertion"
        );
        s.push("x");
        s.push("y");
        assert_eq!(
            s.drain_at_stop(),
            vec!["x".to_string(), "y".to_string()],
            "follow-up lane still drains everything at once"
        );
    }

    #[test]
    fn clear_drops_both_lanes_without_returning_them() {
        let s = Steering::new();
        s.push("follow");
        s.push_steer("steer");
        assert!(!s.is_empty());
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.drain_at_stop(), Vec::<String>::new());
    }

    #[test]
    fn clones_share_one_queue() {
        let a = Steering::new();
        let b = a.clone();
        a.push("x");
        assert_eq!(b.drain_at_stop(), vec!["x".to_string()]);
    }

    #[test]
    fn stop_request_is_observed_once() {
        let s = Steering::new();
        assert!(!s.take_stop_requested(), "no request yet");
        s.request_stop();
        assert!(s.take_stop_requested(), "the request must be observed");
        assert!(
            !s.take_stop_requested(),
            "a consumed request must not be observed twice"
        );
    }

    #[test]
    fn a_second_stop_request_before_the_first_is_observed_is_a_no_op() {
        let s = Steering::new();
        s.request_stop();
        s.request_stop();
        assert!(s.take_stop_requested());
        assert!(!s.take_stop_requested());
    }

    #[test]
    fn clear_also_drops_a_pending_stop_request() {
        let s = Steering::new();
        s.request_stop();
        s.clear();
        assert!(
            !s.take_stop_requested(),
            "clear must not leave a stop request that could cut short a different session's run"
        );
    }

    #[test]
    fn model_switch_is_taken_at_most_once() {
        let s = Steering::new();
        assert!(s.take_model_switch().is_none(), "nothing queued yet");
        s.request_model_switch("claude-cheap", Some(512));
        assert_eq!(
            s.take_model_switch(),
            Some(ModelSwitch {
                model: "claude-cheap".to_string(),
                thinking: Some(512),
                thinking_level: None,
            })
        );
        assert!(
            s.take_model_switch().is_none(),
            "a switch must be observed at most once, like the stop flag"
        );
    }

    #[test]
    fn a_second_model_switch_before_the_first_is_observed_replaces_it() {
        let s = Steering::new();
        s.request_model_switch("claude-cheap", None);
        s.request_model_switch("claude-expensive", Some(1024));
        assert_eq!(
            s.take_model_switch(),
            Some(ModelSwitch {
                model: "claude-expensive".to_string(),
                thinking: Some(1024),
                thinking_level: None,
            }),
            "only the most recent request should apply — no queue of switches"
        );
    }

    #[test]
    fn clear_also_drops_a_pending_model_switch() {
        let s = Steering::new();
        s.request_model_switch("claude-cheap", None);
        s.clear();
        assert!(
            s.take_model_switch().is_none(),
            "clear must not leave a switch that could apply to a different session's run"
        );
    }

    #[test]
    fn model_switch_is_shared_across_clones() {
        let a = Steering::new();
        let b = a.clone();
        a.request_model_switch("claude-cheap", Some(256));
        assert_eq!(
            b.take_model_switch(),
            Some(ModelSwitch {
                model: "claude-cheap".to_string(),
                thinking: Some(256),
                thinking_level: None,
            })
        );
    }

    #[test]
    fn stop_request_is_shared_across_clones() {
        let a = Steering::new();
        let b = a.clone();
        a.request_stop();
        assert!(b.take_stop_requested());
    }

    /// A trivial named tool, standing in for a real one — just enough to distinguish one
    /// [`ToolRegistry`] from another by name in the tests below.
    struct NamedTool(&'static str);
    #[async_trait::async_trait]
    impl crate::tool::Tool for NamedTool {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object" })
        }
        async fn run(
            &self,
            _input: serde_json::Value,
        ) -> std::result::Result<crate::tool::ToolOutput, crate::error::ToolError> {
            Ok("ok".into())
        }
    }

    fn registry_with(names: &[&'static str]) -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        for name in names {
            reg.register(std::sync::Arc::new(NamedTool(name)));
        }
        reg
    }

    #[test]
    fn tool_switch_is_taken_at_most_once() {
        // Task #13 (pi-parity): mirrors `model_switch`'s own "observed exactly once" contract — pi's
        // `setTools` is queued the same way a `prepareNextTurn` model switch is.
        let s = Steering::new();
        assert!(s.take_tool_switch().is_none(), "nothing queued yet");
        s.request_tool_set(registry_with(&["bash"]));
        let taken = s.take_tool_switch().expect("a tool switch must be queued");
        assert!(taken.get("bash").is_some());
        assert!(
            s.take_tool_switch().is_none(),
            "a tool-set switch must be observed at most once, like the model switch"
        );
    }

    #[test]
    fn a_second_tool_switch_before_the_first_is_observed_replaces_it() {
        let s = Steering::new();
        s.request_tool_set(registry_with(&["bash"]));
        s.request_tool_set(registry_with(&["read", "write"]));
        let taken = s.take_tool_switch().expect("a tool switch must be queued");
        assert!(
            taken.get("bash").is_none()
                && taken.get("read").is_some()
                && taken.get("write").is_some(),
            "only the most recent request should apply — no queue of switches"
        );
    }

    #[test]
    fn clear_also_drops_a_pending_tool_switch() {
        let s = Steering::new();
        s.request_tool_set(registry_with(&["bash"]));
        s.clear();
        assert!(
            s.take_tool_switch().is_none(),
            "clear must not leave a tool-set switch that could apply to a different session's run"
        );
    }

    #[test]
    fn clear_run_scoped_also_drops_a_pending_tool_switch() {
        let s = Steering::new();
        s.request_tool_set(registry_with(&["bash"]));
        s.clear_run_scoped();
        assert!(
            s.take_tool_switch().is_none(),
            "a cancelled run must not leave a tool-set switch queued for an unrelated later run"
        );
    }

    #[test]
    fn tool_switch_is_shared_across_clones() {
        let a = Steering::new();
        let b = a.clone();
        a.request_tool_set(registry_with(&["bash"]));
        let taken = b.take_tool_switch().expect("shared across clones");
        assert!(taken.get("bash").is_some());
    }

    #[test]
    fn next_turn_is_a_lane_distinct_from_steer_and_follow_up() {
        // The next-turn lane is a third lane, separate from `steerQueue`/`followUpQueue` (pi's real,
        // shipped pair) — queuing to it must not drain (or be drained by) either of the other two. No
        // real pi-product equivalent of this third lane exists (see the module doc comment).
        let s = Steering::new();
        s.push_next_turn("next");
        s.push("follow");
        s.push_steer("steer");
        assert_eq!(s.drain_steer(), vec!["steer".to_string()]);
        assert_eq!(s.drain_at_stop(), vec!["follow".to_string()]);
        assert_eq!(
            s.drain_next_turn(),
            vec!["next".to_string()],
            "next-turn must survive both other lanes' drains untouched"
        );
    }

    #[test]
    fn drain_next_turn_always_takes_the_whole_lane() {
        // No `QueueMode` for this lane — an unconditional whole-lane drain (this lane has no real
        // pi-product equivalent to match at all; see the module doc comment).
        let s = Steering::new();
        s.push_next_turn("a");
        s.push_next_turn("b");
        s.push_next_turn("c");
        assert_eq!(
            s.drain_next_turn(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(s.drain_next_turn(), Vec::<String>::new());
    }

    #[test]
    fn next_turn_counts_toward_is_empty_and_pending_count() {
        let s = Steering::new();
        assert!(s.is_empty());
        s.push_next_turn("queued while idle");
        assert!(!s.is_empty());
        assert_eq!(s.pending_count(), 1);
        s.drain_next_turn();
        assert!(s.is_empty());
    }

    #[test]
    fn clear_also_drops_a_pending_next_turn_message() {
        let s = Steering::new();
        s.push_next_turn("stale");
        s.clear();
        assert_eq!(
            s.drain_next_turn(),
            Vec::<String>::new(),
            "clear must not leave a next-turn message that could bleed into a different session's run"
        );
    }

    #[test]
    fn next_turn_is_shared_across_clones() {
        let a = Steering::new();
        let b = a.clone();
        a.push_next_turn("x");
        assert_eq!(b.drain_next_turn(), vec!["x".to_string()]);
    }

    #[test]
    fn clear_run_scoped_drops_steer_and_follow_up_and_stop_and_switch_but_not_next_turn() {
        // pi-parity fix: `Agent::run_events_steered`'s cancellation exit paths must not drop a
        // message queued via `push_next_turn` — mirrors pi's `AgentHarness.abort()`, which clears
        // only `steerQueue`/`followUpQueue`. The general-purpose `clear()` (for a genuine session
        // swap) is deliberately still stronger and drops all three lanes — see its own test below.
        let s = Steering::new();
        s.push("follow");
        s.push_steer("steer");
        s.push_next_turn("must survive");
        s.request_stop();
        s.request_model_switch("claude-cheap", Some(256));

        s.clear_run_scoped();

        assert_eq!(
            s.drain_at_stop(),
            Vec::<String>::new(),
            "follow-up must be dropped"
        );
        assert_eq!(
            s.drain_steer(),
            Vec::<String>::new(),
            "steer must be dropped"
        );
        assert!(!s.take_stop_requested(), "the stop request must be dropped");
        assert!(
            s.take_model_switch().is_none(),
            "the model switch must be dropped"
        );
        assert_eq!(
            s.drain_next_turn(),
            vec!["must survive".to_string()],
            "the next-turn lane must survive a run-scoped clear"
        );
    }

    #[test]
    fn clear_still_drops_the_next_turn_lane_for_a_genuine_session_swap() {
        // The stronger, general-purpose `clear()` is unchanged: a caller swapping sessions
        // (`new_session`/`switch_session`/`fork`/`switch_branch`) must not let a message queued for
        // the *old* session's next turn bleed into the newly switched-to one.
        let s = Steering::new();
        s.push_next_turn("stale");
        s.clear();
        assert_eq!(s.drain_next_turn(), Vec::<String>::new());
    }

    #[test]
    fn request_model_switch_with_thinking_level_sets_the_tri_stated_field() {
        let s = Steering::new();
        s.request_model_switch_with_thinking_level("claude-cheap", None, Some(ThinkingLevel::Off));
        assert_eq!(
            s.take_model_switch(),
            Some(ModelSwitch {
                model: "claude-cheap".to_string(),
                thinking: None,
                thinking_level: Some(ThinkingLevel::Off),
            })
        );
    }

    #[test]
    fn a_full_lane_rejects_the_newest_message_and_keeps_the_queued_prefix_intact() {
        // The lanes are fed straight from remote RPC and drained one message per turn boundary, so a
        // client that pushes faster than the agent drains must hit a wall rather than grow the queue
        // (and its base64 image payloads) without bound. The wall refuses the *newest* message: what
        // is already queued stays queued, in order, so the client's instruction stream is truncated at
        // the tail rather than silently reordered.
        let s = Steering::new();
        for i in 0..MAX_QUEUED_PER_LANE {
            assert!(
                s.push_steer(format!("m{i}")),
                "message {i} is under the cap"
            );
        }
        assert_eq!(s.pending_count(), MAX_QUEUED_PER_LANE);

        assert!(
            !s.push_steer("one too many"),
            "the cap must refuse the push"
        );
        assert_eq!(
            s.pending_count(),
            MAX_QUEUED_PER_LANE,
            "a refused push must not grow the lane"
        );
        let texts = s.steer_texts();
        assert_eq!(
            texts.first().map(String::as_str),
            Some("m0"),
            "the oldest message is untouched"
        );
        assert!(
            !texts.iter().any(|t| t == "one too many"),
            "the refused message must not have displaced anything"
        );

        // Draining frees exactly one slot: the cap is a live watermark, not a one-shot latch.
        assert_eq!(s.drain_steer(), vec!["m0".to_string()]);
        assert!(
            s.push_steer("now there is room"),
            "a drain makes room again"
        );
        assert!(!s.push_steer("but only one slot's worth"));
    }

    #[test]
    fn each_lane_is_capped_independently() {
        // One saturated lane must not starve the others — they have separate drain points, and a
        // stalled steer lane says nothing about whether a follow-up or next-turn message can land.
        let s = Steering::new();
        for i in 0..MAX_QUEUED_PER_LANE {
            assert!(s.push_steer(format!("s{i}")));
        }
        assert!(!s.push_steer("rejected"));
        assert!(s.push("the follow-up lane has its own budget"));
        assert!(s.push_next_turn("as does the next-turn lane"));
        assert_eq!(s.pending_count(), MAX_QUEUED_PER_LANE + 2);
    }

    #[test]
    fn the_follow_up_and_next_turn_lanes_are_capped_too() {
        // The next-turn lane is the one most exposed: it is pushed while the agent is idle, and
        // nothing drains it until some later prompt starts a fresh run.
        let s = Steering::new();
        for _ in 0..MAX_QUEUED_PER_LANE {
            assert!(s.push("f"));
            assert!(s.push_next_turn("n"));
        }
        assert!(!s.push("over"), "the follow-up lane is capped");
        assert!(!s.push_next_turn("over"), "the next-turn lane is capped");
        assert_eq!(s.pending_count(), MAX_QUEUED_PER_LANE * 2);
    }

    #[test]
    fn clear_releases_a_saturated_lane() {
        let s = Steering::new();
        for _ in 0..MAX_QUEUED_PER_LANE {
            s.push_steer("x");
        }
        assert!(!s.push_steer("full"));
        s.clear();
        assert!(
            s.push_steer("room again"),
            "clear must return the lane's whole budget, not leave it wedged"
        );
    }

    #[test]
    fn request_model_switch_leaves_thinking_level_unset() {
        // The plain two-arg `request_model_switch` (every existing caller) must keep meaning
        // "don't touch reasoning effort/thinking depth at all" — not silently turn it off.
        let s = Steering::new();
        s.request_model_switch("claude-cheap", Some(512));
        assert_eq!(
            s.take_model_switch().unwrap().thinking_level,
            None,
            "the plain two-arg request must not implicitly set a thinking-level override"
        );
    }
}
