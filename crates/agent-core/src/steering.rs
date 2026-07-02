//! Steering — injecting user input into a run in flight.
//!
//! A [`Steering`] handle holds two shared queues, distinguished by *when* the loop injects them:
//!
//! - **steer** ([`push_steer`](Steering::push_steer)) — injected *mid-run*, between tool-executing
//!   turns, folded onto the same tool-results user turn. This is how a client redirects a busy agent
//!   ("actually, also handle X") without waiting for it to stop.
//! - **follow-up** ([`push`](Steering::push)) — injected only at a *would-stop* boundary (the model
//!   ended its turn without asking for tools), as a fresh user turn. This is "keep going / now do the
//!   next thing" once the current work is done.
//!
//! Both injection points place messages where the previous message is the assistant's, so a pushed
//! user turn never lands next to another user turn (which the wire would reject). The two lanes mirror
//! pi's separate `steerQueue` and `followUpQueue`.
//!
//! A third, independent signal lives here too: [`request_stop`](Steering::request_stop) — a graceful,
//! host-initiated "stop after the current turn" request, mirroring pi's `shouldStopAfterTurn`. Unlike
//! cancellation (which drops an in-flight future and can abandon a tool mid-execution), this is checked
//! only at a turn boundary, after that turn's tool calls (if any) have already completed and their
//! results are durably committed — so it never leaves an orphaned `tool_use` behind. It's a flag, not a
//! queue: a second request before the first is observed is indistinguishable from one.
//!
//! A fourth and fifth setting — [`QueueMode`], one per lane — govern how much of that lane a single
//! drain point consumes: `All` (the historical behavior here) folds everything queued into one
//! injection; `OneAtATime` (pi's `PendingMessageQueue` default) takes only the oldest message, leaving
//! the rest queued for the *next* drain point, so several quick messages from a client land as separate
//! turns instead of one merged one. The steer and follow-up lanes carry **independent** `QueueMode`
//! settings (matching pi's own separate `steeringMode`/`followUpMode`) — a client may want, say, every
//! mid-run redirect folded together while follow-ups still land one at a time, or vice versa.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::message::ImageSource;

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

/// A cloneable handle to the shared steering queues. Clones share the same two lanes (and the stop
/// flag).
#[derive(Clone, Default)]
pub struct Steering {
    /// Injected mid-run, between tool turns.
    steer: Queue,
    /// Injected at a would-stop boundary.
    follow_up: Queue,
    /// Set by [`request_stop`](Self::request_stop); consumed by the loop at the next turn boundary.
    stop_requested: Arc<AtomicBool>,
    /// How much of the steer lane [`drain_steer`](Self::drain_steer) consumes per call — see
    /// [`QueueMode`]. A setting, not per-run state: `clear()` deliberately leaves it untouched.
    steer_mode: Arc<Mutex<QueueMode>>,
    /// How much of the follow-up (plus any stranded steer) lane [`drain_at_stop`](Self::drain_at_stop)
    /// consumes per call — independent of `steer_mode`, matching pi's separate `followUpMode`.
    follow_up_mode: Arc<Mutex<QueueMode>>,
}

impl Steering {
    /// Empty steering queues.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a **follow-up**: injected when the run next reaches a stop boundary. Accepts a plain
    /// string or a [`SteeringMessage`] (with image attachments).
    pub fn push(&self, message: impl Into<SteeringMessage>) {
        lock(&self.follow_up).push_back(message.into());
    }

    /// Queue a **steer**: injected mid-run, at the next tool-results turn, to redirect a busy agent.
    /// Accepts a plain string or a [`SteeringMessage`] (with image attachments).
    pub fn push_steer(&self, message: impl Into<SteeringMessage>) {
        lock(&self.steer).push_back(message.into());
    }

    /// Whether either queue currently holds any messages.
    pub fn is_empty(&self) -> bool {
        lock(&self.steer).is_empty() && lock(&self.follow_up).is_empty()
    }

    /// Total queued messages across both lanes — pi's `pendingMessageCount` (`get_state`'s queue-depth
    /// field). A peek, not a drain: calling this doesn't consume anything.
    pub fn pending_count(&self) -> usize {
        lock(&self.steer).len() + lock(&self.follow_up).len()
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

    /// Drop everything queued in both lanes without returning it — for a caller (`new_session`/
    /// `switch_session`/`fork`/`switch_branch`) that's about to swap in a different session's
    /// conversation, so a message queued for the *old* session's next turn can't leak into the newly
    /// switched-to one. Also clears any pending stop request, for the same reason: a graceful-stop
    /// request aimed at the old session's run must not cut short a different session's next one.
    pub fn clear(&self) {
        lock(&self.steer).clear();
        lock(&self.follow_up).clear();
        self.stop_requested.store(false, Ordering::Relaxed);
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
}

/// Recover the data on a poisoned lock rather than propagating a panic into the loop.
fn lock(q: &Queue) -> std::sync::MutexGuard<'_, VecDeque<SteeringMessage>> {
    q.lock().unwrap_or_else(|e| e.into_inner())
}

/// Same recovery posture as [`lock`], for the [`QueueMode`] setting.
fn lock_mode(m: &Arc<Mutex<QueueMode>>) -> std::sync::MutexGuard<'_, QueueMode> {
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
    fn stop_request_is_shared_across_clones() {
        let a = Steering::new();
        let b = a.clone();
        a.request_stop();
        assert!(b.take_stop_requested());
    }
}
