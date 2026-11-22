//! The decisions of the serve session lifecycle, as one pure state machine: who may attach to a
//! session id, when an id is spawned, which incarnation an unpin may touch, which sessions the idle
//! reaper takes, and when a drain is over. The shell is `serve_ws.rs`'s `Supervisor`.
//!
//! [`step`] takes the [`Table`] and an [`Event`] and returns the [`Action`] to perform. It does no
//! I/O, is not async, reads no clock (time arrives in the events, as a [`Duration`] on the caller's
//! monotonic scale) and is deterministic, so a model checker can call exactly the function
//! production calls (`tests/serve_session_model.rs`). Production holds the table's std mutex,
//! feeds the kernel one event, performs the action and releases the lock; channels, tokens and the
//! task are opaque to the kernel and live in the shell, keyed by the same id.
//!
//! The state is one [`Slot`] per id (its incarnation, [`Phase`], tenant, attach count, detach time
//! and end reason), the next incarnation number, and the [`Lifecycle`]. State that the session task
//! changes without the table lock (the `running` and `keep_alive` atomics, and whether a command is
//! queued on, or the loop has closed, the input channel) is not here: the shell reads it at the
//! decision point and passes it as an [`Obs`], so the model can change it between two decisions as
//! the real task can.
//!
//! | event | decision |
//! |---|---|
//! | [`Event::Pin`] (a connection wants the id) | `Closed` after shutdown; another tenant's live id `Forbidden`; an attachable slot `Attached` (count + 1, detach clock off); a `Stopping` slot, or one whose input is closed, is marked stopping and the pin `Wait`s for its exit; a free id is `Spawned` unless the daemon is not `Open` or the live-session cap is reached |
//! | [`Event::Unpin`] | the attach count of the pinned **incarnation** drops; an unpin from any other incarnation is a no-op; the last detach starts the reaper's clock |
//! | [`Event::GoLive`] | `Starting` becomes `Live` (`true`); anything else (stopped first) is `false` and the body never runs |
//! | [`Event::Exited`] | the task is gone: its own incarnation's entry is removed and its end reason returned |
//! | [`Event::Reap`] | every slot [`is_reapable`] names becomes `Stopping` (reason `IdleReap`) |
//! | [`Event::BeginDrain`] / [`Event::DrainTick`] / [`Event::Shutdown`] | `Open` → `Draining` (new ids refused, reconnects admitted); a tick is done once the deadline passed or no run is in flight; shutdown closes the table and stops every slot (reason `Drain`) |
//!
//! The first end reason recorded wins. Mutations (cargo features `mut_*`, used only by the model's
//! red runs) each remove one guard.

use std::collections::HashMap;
use std::time::Duration;

use crate::metrics::SessionEnd;

/// Where an id's session task is in its life: `Starting → Live → Stopping → (removed)`, one way
/// only. The entry stays for the task's **whole** life, so "the id has an entry" means exactly "a
/// task owns this id's session file".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Spawned, not yet running its body. Attachable.
    Starting,
    /// Running its body. Attachable.
    Live,
    /// Told to exit and not yet gone. Not attachable: a pin waits for the exit.
    Stopping,
}

/// What this replica will still accept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Lifecycle {
    /// Serving normally.
    #[default]
    Open,
    /// New ids are refused; reconnects to sessions this replica still owns are admitted.
    Draining,
    /// Sessions are being stopped. Nothing attaches.
    Closed,
}

/// One id's session, as far as the decisions go.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    /// Which spawn of this id this is. Whatever acts on the entry later carries the incarnation it
    /// was handed, so a stale actor never touches a newer session under the same id.
    pub incarnation: u64,
    pub phase: Phase,
    /// Whose session this is, in service mode (`None` outside it).
    pub tenant: Option<String>,
    /// How many connections are attached.
    pub attached: usize,
    /// When the last connection detached; `None` while any is attached, or on a fresh session.
    pub last_detached_at: Option<Duration>,
    /// Why the session ended; `Client` until something claims it, and the first claim wins.
    pub end_reason: SessionEnd,
}

impl Slot {
    /// The session accepts input: `Starting` or `Live`.
    pub fn attachable(&self) -> bool {
        self.phase != Phase::Stopping
    }

    /// → `Stopping`, idempotent. The first writer of the end reason wins: a session the reaper
    /// already claimed that then meets a shutdown ended because it was idle.
    pub fn stop(&mut self, reason: SessionEnd) {
        if self.end_reason == SessionEnd::Client || cfg!(feature = "mut_end_reason_overwritten") {
            self.end_reason = reason;
        }
        self.phase = Phase::Stopping;
    }
}

/// The id → [`Slot`] table, with the state that must change atomically alongside it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    pub slots: HashMap<String, Slot>,
    /// Stamped on the next spawn.
    pub next_incarnation: u64,
    pub lifecycle: Lifecycle,
}

/// What the session task changes without the table lock, read by the shell at the decision point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Obs {
    /// A `prompt`, host `bash` or `compact` is executing.
    pub running: bool,
    /// The session holds a live MCP Events subscription.
    pub keep_alive: bool,
    /// A command the client sent is still queued on the input channel.
    pub queued: bool,
    /// The session's loop is gone (its input receiver was dropped): nothing will read the input.
    pub input_closed: bool,
}

pub enum Event<'a> {
    /// A connection wants `id`. `input_closed` is the id's [`Obs::input_closed`] (when it has a slot).
    Pin {
        id: &'a str,
        tenant: Option<&'a str>,
        cap: Option<usize>,
        input_closed: bool,
    },
    /// A connection pinned to `incarnation` of `id` is gone.
    Unpin {
        id: &'a str,
        incarnation: u64,
        now: Duration,
    },
    /// The task of `incarnation` finished its start-up.
    GoLive { id: &'a str, incarnation: u64 },
    /// The task of `incarnation` is gone.
    Exited { id: &'a str, incarnation: u64 },
    /// The reaper ticks.
    Reap {
        now: Duration,
        timeout: Duration,
        obs: &'a dyn Fn(&str) -> Obs,
    },
    /// The daemon got its signal.
    BeginDrain,
    /// The drain polls: `deadline_passed` is the grace running out.
    DrainTick {
        deadline_passed: bool,
        obs: &'a dyn Fn(&str) -> Obs,
    },
    /// The drain is over: stop everything.
    Shutdown,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    /// Attached to a live session of this incarnation.
    Attached { incarnation: u64 },
    /// The id was free: its slot is inserted (attach count 1) and the shell must spawn the task.
    Spawned { incarnation: u64 },
    /// The id's previous task is still exiting (the slot is `Stopping`): the shell drops the
    /// slot's input and waits for the exit latch, then pins again.
    Wait,
    /// The id is live under another tenant.
    Forbidden,
    /// The live-session cap is reached.
    AtCapacity,
    /// The daemon is shutting down or draining and the id is new.
    Closed,
    /// [`Event::GoLive`]: whether the body may run.
    Live(bool),
    /// [`Event::Exited`]: the reason the session ended, and whether its entry was removed.
    Exited { reason: SessionEnd, removed: bool },
    /// Slots now `Stopping` (reaped or shut down): the shell drops each one's input.
    Stopped(Vec<String>),
    /// [`Event::DrainTick`]: the drain is over, run the shutdown.
    DrainDone,
    /// [`Event::DrainTick`]: runs are still in flight within the grace.
    DrainWait,
    /// Nothing to do.
    Nothing,
}

/// Whether the reaper should stop this session.
///
/// A `Stopping` session is already on its way out. A **closed input** is the strongest reason of
/// all: the loop is gone, so nothing will ever read its input again; stop it whatever its attach
/// state or clock says. Otherwise the session is alive: **detached** for at least `timeout`, not
/// mid-run (a detached background run is what the design exists to keep), without a live MCP Events
/// subscription, and with no command still queued (work the client was told is taken).
pub fn is_reapable(slot: &Slot, obs: Obs, now: Duration, timeout: Duration) -> bool {
    if slot.phase == Phase::Stopping {
        return false;
    }
    if obs.input_closed {
        return true;
    }
    slot.attached == 0
        && slot
            .last_detached_at
            .is_some_and(|d| now.saturating_sub(d) >= timeout)
        && (cfg!(feature = "mut_reap_ignores_running") || !obs.running)
        && (cfg!(feature = "mut_reap_ignores_keep_alive") || !obs.keep_alive)
        && (cfg!(feature = "mut_reap_ignores_queued") || !obs.queued)
}

/// Is any session executing a run, or holding a command the client sent? Stopped slots count
/// only for a run still executing: their input is gone.
pub fn in_flight(table: &Table, obs: &dyn Fn(&str) -> Obs) -> bool {
    table.slots.iter().any(|(id, s)| {
        let o = obs(id);
        o.running || (s.attachable() && o.queued)
    })
}

pub fn step(t: &mut Table, ev: Event<'_>) -> Action {
    match ev {
        Event::Pin {
            id,
            tenant,
            cap,
            input_closed,
        } => pin(t, id, tenant, cap, input_closed),
        Event::Unpin {
            id,
            incarnation,
            now,
        } => {
            if let Some(s) = t.slots.get_mut(id)
                && (cfg!(feature = "mut_no_incarnation_guard") || s.incarnation == incarnation)
            {
                s.attached = s.attached.saturating_sub(1);
                if s.attached == 0 {
                    s.last_detached_at = Some(now);
                }
            }
            Action::Nothing
        }
        Event::GoLive { id, incarnation } => match t.slots.get_mut(id) {
            Some(s) if s.incarnation == incarnation && s.phase == Phase::Starting => {
                s.phase = Phase::Live;
                Action::Live(true)
            }
            _ => Action::Live(false),
        },
        Event::Exited { id, incarnation } => {
            let Some(s) = t.slots.get(id).filter(|s| s.incarnation == incarnation) else {
                return Action::Exited {
                    reason: SessionEnd::Client,
                    removed: false,
                };
            };
            let reason = s.end_reason;
            let removed = !cfg!(feature = "mut_exit_keeps_entry");
            if removed {
                t.slots.remove(id);
            }
            Action::Exited { reason, removed }
        }
        Event::Reap { now, timeout, obs } => {
            let mut stopped = Vec::new();
            for (id, s) in &mut t.slots {
                if is_reapable(s, obs(id), now, timeout) {
                    s.stop(SessionEnd::IdleReap);
                    stopped.push(id.clone());
                }
            }
            Action::Stopped(stopped)
        }
        Event::BeginDrain => {
            if t.lifecycle == Lifecycle::Open {
                t.lifecycle = Lifecycle::Draining;
            }
            Action::Nothing
        }
        Event::DrainTick {
            deadline_passed,
            obs,
        } => {
            if cfg!(feature = "mut_drain_never_done") {
                return Action::DrainWait;
            }
            if deadline_passed || !in_flight(t, obs) {
                Action::DrainDone
            } else {
                Action::DrainWait
            }
        }
        Event::Shutdown => {
            t.lifecycle = Lifecycle::Closed;
            let mut stopped = Vec::with_capacity(t.slots.len());
            for (id, s) in &mut t.slots {
                s.stop(SessionEnd::Drain);
                stopped.push(id.clone());
            }
            Action::Stopped(stopped)
        }
    }
}

fn pin(
    t: &mut Table,
    id: &str,
    tenant: Option<&str>,
    cap: Option<usize>,
    input_closed: bool,
) -> Action {
    if t.lifecycle == Lifecycle::Closed {
        return Action::Closed;
    }
    if let Some(s) = t.slots.get_mut(id) {
        // Checked before anything else about the slot: an id alone never grants access to a running
        // session. A `Stopping` slot is refused just the same.
        if !cfg!(feature = "mut_no_tenant_check") && s.tenant.as_deref() != tenant {
            return Action::Forbidden;
        }
        let respawn = cfg!(feature = "mut_spawn_while_stopping") && s.phase == Phase::Stopping;
        if !respawn {
            if s.attachable() && !input_closed {
                s.attached += 1;
                s.last_detached_at = None;
                return Action::Attached {
                    incarnation: s.incarnation,
                };
            }
            // Stopping, or attachable in name only (its loop already ended): respawning now would
            // put two writers on one session file. Mark it stopping and wait for it to be gone.
            s.stop(SessionEnd::Error);
            return Action::Wait;
        }
    }
    // Nothing owns this id, so this look is about to spawn one. Everything from here is a
    // *new-session* gate: a draining replica still owns live sessions, and a reconnect to one took
    // the branch above.
    if t.lifecycle != Lifecycle::Open && !cfg!(feature = "mut_admit_while_draining") {
        return Action::Closed;
    }
    if let Some(max) = cap
        && t.slots.len() >= max
    {
        return Action::AtCapacity;
    }
    let incarnation = t.next_incarnation;
    t.next_incarnation += 1;
    t.slots.insert(
        id.to_owned(),
        Slot {
            incarnation,
            phase: Phase::Starting,
            tenant: tenant.map(str::to_owned),
            attached: 1,
            last_detached_at: None,
            end_reason: SessionEnd::Client,
        },
    );
    Action::Spawned { incarnation }
}
