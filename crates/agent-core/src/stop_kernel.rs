//! The end-of-turn decision, as a pure kernel: when the model ends its turn without tools, does the
//! run end, or continue with queued messages and background notices?
//!
//! [`step`] is `(state, event) -> (state, action)` with no I/O. `run_events_steered` makes every
//! stop-time decision by calling it (through [`decide`], which only loops `step` and performs the
//! effect each [`Action`] asks for), and `tests/stop_kernel_model.rs` model-checks the same
//! functions: the model owns only the environment (what is queued, what finished, what still
//! runs), never a copy of the decision. See ARCHITECTURE.md, "The end-of-turn kernel".
//!
//! Effects the kernel asks for, in order: [`Action::Drain`] (follow-ups and stranded steers), then
//! [`Action::TakeNotices`] (finish notices), then, only if neither delivered anything and no
//! reminder was given this run, [`Action::AskReminder`]. At the step limit it asks for none of them
//! and ends: draining would consume a message the run cannot deliver (a `Steered` event a host
//! records as delivery) and continuing would fail with `MaxSteps`.
//!
//! The `mut_*` features each compile out one guard, for the model's mutation tests.

/// Whether `steps` taken this call reach `limit` (`None` = no ceiling). Used for both the
/// top-of-loop `MaxSteps` check and the stop-time decision, so the two cannot disagree.
pub fn limit_reached(steps: u32, limit: Option<u32>) -> bool {
    if cfg!(feature = "mut_no_limit_check") {
        return false;
    }
    limit.is_some_and(|limit| steps >= limit)
}

/// Which background report rides a continue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Note {
    /// Work ended unattended: its finish notice.
    Finished,
    /// Work still running: the once-per-run stop reminder.
    StillRunning,
}

/// How the stop resolves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// Emit `AgentEnd` and return.
    End,
    /// Push `followups` drained messages (then the note, if any), checkpoint, emit `Steered` if
    /// `followups > 0` and `Background` if there is a note, and take another turn.
    Continue {
        followups: usize,
        note: Option<Note>,
    },
}

impl Outcome {
    /// The `Steered` event this outcome owes: the count of follow-ups delivered, if any.
    pub fn steered(self) -> Option<usize> {
        match self {
            Outcome::Continue { followups, .. } if followups > 0 => Some(followups),
            _ => None,
        }
    }

    /// The `Background` event this outcome owes, after any `Steered`.
    pub fn background(self) -> Option<Note> {
        match self {
            Outcome::Continue { note, .. } => note,
            Outcome::End => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Phase {
    Idle,
    Draining,
    Noticing { followups: usize },
    Reminding { followups: usize },
}

/// Per-run state: the reminder latch, plus where the current decision is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StopState {
    /// A stop reminder was given this run: at most one, so a model that ends again ends.
    pub reminded: bool,
    phase: Phase,
}

impl StopState {
    pub fn new() -> Self {
        Self {
            reminded: false,
            phase: Phase::Idle,
        }
    }
}

impl Default for StopState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// The model ended its turn without tools, with `steps` taken this call under `limit`.
    Stopped { steps: u32, limit: Option<u32> },
    /// The drain delivered `followups` messages.
    Drained { followups: usize },
    /// Whether a finish notice was pending (and is now taken).
    Notices { pending: bool },
    /// Whether work is still running (a reminder text exists).
    Reminder { pending: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Drain,
    TakeNotices,
    AskReminder,
    Done(Outcome),
}

/// The decision: one event in, the next state and the one action to perform out.
pub fn step(mut s: StopState, ev: Event) -> (StopState, Action) {
    match (s.phase, ev) {
        (Phase::Idle, Event::Stopped { steps, limit }) => {
            if limit_reached(steps, limit) && !cfg!(feature = "mut_drain_at_limit") {
                return (s, Action::Done(Outcome::End));
            }
            s.phase = Phase::Draining;
            (s, Action::Drain)
        }
        (Phase::Draining, Event::Drained { followups }) => {
            if cfg!(feature = "mut_no_notice_delivery") {
                return remind_or_resolve(s, followups, None);
            }
            s.phase = Phase::Noticing { followups };
            (s, Action::TakeNotices)
        }
        (Phase::Noticing { followups }, Event::Notices { pending }) => {
            let note = pending.then_some(Note::Finished);
            remind_or_resolve(s, followups, note)
        }
        (Phase::Reminding { followups }, Event::Reminder { pending }) => {
            s.reminded = pending;
            s.phase = Phase::Idle;
            (
                s,
                Action::Done(resolve(followups, pending.then_some(Note::StillRunning))),
            )
        }
        // Events the machine never produces: end rather than guess.
        _ => {
            s.phase = Phase::Idle;
            (s, Action::Done(Outcome::End))
        }
    }
}

/// With nothing delivered yet and no reminder given, ask whether work still runs; otherwise
/// resolve.
fn remind_or_resolve(
    mut s: StopState,
    followups: usize,
    note: Option<Note>,
) -> (StopState, Action) {
    let latched = s.reminded && !cfg!(feature = "mut_no_reminder_latch");
    if note.is_none() && followups == 0 && !latched {
        s.phase = Phase::Reminding { followups };
        return (s, Action::AskReminder);
    }
    s.phase = Phase::Idle;
    (s, Action::Done(resolve(followups, note)))
}

fn resolve(followups: usize, note: Option<Note>) -> Outcome {
    if followups == 0 && note.is_none() {
        Outcome::End
    } else {
        Outcome::Continue { followups, note }
    }
}

/// The effects the kernel asks for. Production backs them with the steering queue and the tools'
/// background sources; the model backs them with its environment.
pub trait Effects {
    /// Drain follow-ups and stranded steers; how many were taken.
    fn drain(&mut self) -> usize;
    /// Take a pending finish notice; whether there was one.
    fn take_notices(&mut self) -> bool;
    /// Whether work still runs (the reminder text exists).
    fn stop_reminder(&mut self) -> bool;
}

/// Run one stop decision: loop [`step`], performing each effect it asks for.
pub fn decide(
    state: &mut StopState,
    steps: u32,
    limit: Option<u32>,
    fx: &mut impl Effects,
) -> Outcome {
    let mut ev = Event::Stopped { steps, limit };
    loop {
        let (next, action) = step(*state, ev);
        *state = next;
        ev = match action {
            Action::Drain => Event::Drained {
                followups: fx.drain(),
            },
            Action::TakeNotices => Event::Notices {
                pending: fx.take_notices(),
            },
            Action::AskReminder => Event::Reminder {
                pending: fx.stop_reminder(),
            },
            Action::Done(outcome) => return outcome,
        };
    }
}
