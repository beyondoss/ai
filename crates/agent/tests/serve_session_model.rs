//! Stateright model of the serve session lifecycle: `serve_ws.rs`'s `Supervisor` table.
//!
//! It checks every interleaving of two connections (two pins each, either tenant), the session task
//! of each incarnation (start, go live, run, keep-alive, queued prompts, its own end, the EOF a stop
//! produces, its exit), the idle reaper at a clock that crosses its timeout, and a drain
//! (begin, deadline, poll, shutdown). A connection can be aborted at any point after its pin, which
//! drops its guard exactly as a cancelled attach does.
//!
//! ## Faithfulness
//!
//! Every decision is made by the REAL kernel, `serve::session_kernel::step`, the function
//! production calls (`Supervisor::try_pin`, `unpin`, `reap_idle`, `begin_drain`, `drain_done`,
//! `shutdown`, `ExitGuard`). The model owns only the environment, which contains no copy of the
//! kernel's branching: the connections, the session tasks (the state the real task changes without
//! the table lock, and passes to the kernel as an `Obs` at each decision: `running`, `keep_alive`,
//! a queued command, a closed input), the clock, and the shell effects the kernel's actions name (a
//! stopped slot's input is dropped; a task that sees EOF ends; an exit cancels its latch). Transitions
//! match the shell one for one: the lock, one `Event`, its `Action`, the effect.
//!
//! Each property is judged against the environment's own facts (which connections hold which
//! incarnation, what each task is doing, who spawned what, when the last connection left), never by
//! asking the kernel what it should have done.
//!
//! ## Properties (`always`)
//!
//! - P1 one writer: at most one session task per id has not yet exited.
//! - P2 pins are exact: a session's `attached` is the number of connections pinned to that
//!   incarnation, aborts included.
//! - P3 a stale unpin is harmless: an unpin from incarnation n changes nothing of incarnation n+1.
//! - P4 the reaper never reaps a session whose loop is alive and that has a connection, a run, a
//!   subscription or a queued command, or that has not been detached for the timeout.
//! - P5 drain admission: after the drain begins no new id is spawned, and a reconnect to a live
//!   session is still admitted.
//! - P6 the drain waits: it does not finish before its deadline while a run is in flight or a
//!   command is queued.
//! - P7 the first end reason wins: a recorded reason never changes.
//! - P8 tenancy: a connection is only ever attached to a session its tenant spawned.
//! - L1/L2/L3 liveness, as a bounded measure under `always` (the check is one kernel step):
//!   L1 once a session's task has exited, a pin for the id is answered with an attach or a spawn,
//!   never a wait (the wait-then-respawn path has no state with a timer as its only exit);
//!   L2 once no run is in flight a drain poll finishes the drain;
//!   L3 a detached idle session past the timeout is stopped by the next reaper tick.
//!   Fairness assumed: the reaper tick and the drain poll are always enabled and time advances.
//!   The mutations `mut_exit_keeps_entry`, `mut_drain_never_done` and `mut_no_unpin_on_abort`
//!   remove those exits.
//!
//! Not modelled: the live-session cap (one id here; the kernel's `AtCapacity` is one comparison),
//! service-mode lock outcomes beyond "the task failed to start" (`StartFail`), the wait's grace
//! timeout beyond a connection giving up (`GiveUp`).
//!
//! ## Mutations
//!
//! `cargo test -p beyond-agent --features <mut> --test it serve_session_model::the_mutation` MUST
//! pass only when the named property is red: see `crates/agent/ARCHITECTURE.md` for the feature →
//! property table.

use std::time::Duration;

use beyond_agent::metrics::SessionEnd;
use beyond_agent::serve::session_kernel as kernel;
use kernel::{Action, Event, Lifecycle, Obs, Phase};
use stateright::{Checker, Model, Property};

const ID: &str = "s";
const CONNS: usize = 2;
const PINS: u8 = 2;
/// Incarnations the check explores (a third spawn is cut off).
const INCARNATIONS: usize = 2;
/// The clock ticks 0..=TICKS; the reaper's timeout is `TIMEOUT` ticks.
const TICKS: u8 = 3;
const TIMEOUT: u8 = 2;

const V_P3: u32 = 1;
const V_P4: u32 = 2;
const V_P5: u32 = 4;
const V_P6: u32 = 8;

fn secs(t: u8) -> Duration {
    Duration::from_secs(u64::from(t))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum C {
    Idle,
    /// A pin answered `Wait` for this incarnation's exit.
    Waiting(u64),
    /// Holds a pin (and a guard) on this incarnation.
    Pinned(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Conn {
    st: C,
    pins_left: u8,
    tenant: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum T {
    /// Spawned, before `go_live`.
    Starting,
    /// Running its body; its input receiver is alive.
    Body,
    /// The body is gone (input receiver dropped), the task is persisting.
    Ending,
    /// The exit guard has run: the latch is cancelled.
    Exited,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Task {
    st: T,
    running: bool,
    keep_alive: bool,
    queued: u8,
    /// The shell dropped the retained input (a stop or a wait): the loop sees EOF once no
    /// connection holds a clone.
    input_dropped: bool,
    /// The tenant of the connection that spawned it.
    tenant: u8,
    /// The tick the last connection left, `None` while one is attached or before any attached.
    detached_at: Option<u8>,
    /// The first end reason the table showed for it (`None`: still `Client`).
    first_reason: Option<SessionEnd>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Drain {
    /// The reaper runs.
    Off,
    /// Draining: the reaper is aborted, the poll runs.
    Begun,
    /// The poll said done; shutdown is next.
    Done,
    Closed,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct St {
    k: kernel::Table,
    now: u8,
    conns: Vec<Conn>,
    tasks: Vec<Task>,
    drain: Drain,
    deadline: bool,
    bad: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Act {
    Pin(usize),
    Retry(usize),
    GiveUp(usize),
    Send(usize),
    Unpin(usize),
    Abort(usize),
    GoLive(usize),
    StartFail(usize),
    Dequeue(usize),
    RunEnd(usize),
    ToggleKeepAlive(usize),
    SelfEnd(usize),
    Eof(usize),
    Exit(usize),
    Tick,
    Reap,
    BeginDrain,
    Deadline,
    DrainTick,
    Shutdown,
}

struct Serve {
    /// Connection 1 belongs to another tenant.
    other_tenant: bool,
}

impl St {
    fn obs(&self, i: usize) -> Obs {
        let t = &self.tasks[i];
        Obs {
            running: t.running,
            keep_alive: t.keep_alive,
            queued: t.queued > 0,
            input_closed: matches!(t.st, T::Ending | T::Exited),
        }
    }

    /// The observation the kernel gets for an id: the task of the slot's incarnation.
    fn obs_of(&self, id: &str) -> Obs {
        match self.k.slots.get(id) {
            Some(s) => self.obs(s.incarnation as usize),
            None => Obs::default(),
        }
    }

    fn pinned_to(&self, i: u64) -> usize {
        self.conns.iter().filter(|c| c.st == C::Pinned(i)).count()
    }

    /// A command a client sent that a drain waits for, or a run executing.
    fn busy(&self) -> bool {
        self.tasks
            .iter()
            .any(|t| t.running || (t.queued > 0 && !t.input_dropped))
    }

    /// The shell dropped the input of the slot the kernel named.
    fn dropped_input(&mut self, ids: &[String]) {
        for id in ids {
            if let Some(s) = self.k.slots.get(id.as_str()) {
                self.tasks[s.incarnation as usize].input_dropped = true;
            }
        }
    }

    /// Record the end reasons the table shows, for P7.
    fn note_reasons(&mut self) {
        for s in self.k.slots.values() {
            let t = &mut self.tasks[s.incarnation as usize];
            if t.first_reason.is_none() && s.end_reason != SessionEnd::Client {
                t.first_reason = Some(s.end_reason);
            }
        }
    }

    /// A connection left `inc`: when it was the last, the idle clock starts (as the model sees it).
    fn left(&mut self, inc: u64) {
        if self.pinned_to(inc) == 0 {
            self.tasks[inc as usize].detached_at = Some(self.now);
        }
    }
}

impl Serve {
    fn tenant_name(t: u8) -> &'static str {
        if t == 0 { "A" } else { "B" }
    }

    fn pin_event(&self, s: &St, c: usize) -> (Option<&'static str>, bool) {
        let _ = self;
        let tenant = Some(Self::tenant_name(s.conns[c].tenant));
        (tenant, s.obs_of(ID).input_closed)
    }
}

impl Model for Serve {
    type State = St;
    type Action = Act;

    fn init_states(&self) -> Vec<St> {
        vec![St {
            k: kernel::Table::default(),
            now: 0,
            conns: (0..CONNS)
                .map(|c| Conn {
                    st: C::Idle,
                    pins_left: PINS,
                    tenant: u8::from(self.other_tenant && c == 1),
                })
                .collect(),
            tasks: vec![],
            drain: Drain::Off,
            deadline: false,
            bad: 0,
        }]
    }

    fn actions(&self, s: &St, out: &mut Vec<Act>) {
        for c in 0..CONNS {
            let conn = s.conns[c];
            match conn.st {
                C::Idle if conn.pins_left > 0 => out.push(Act::Pin(c)),
                C::Idle => {}
                C::Waiting(i) => {
                    if s.tasks[i as usize].st == T::Exited {
                        out.push(Act::Retry(c));
                    }
                    out.push(Act::GiveUp(c));
                }
                C::Pinned(i) => {
                    let t = &s.tasks[i as usize];
                    if matches!(t.st, T::Starting | T::Body) && t.queued < 1 {
                        out.push(Act::Send(c));
                    }
                    out.push(Act::Unpin(c));
                    out.push(Act::Abort(c));
                }
            }
        }
        for (i, t) in s.tasks.iter().enumerate() {
            match t.st {
                T::Starting => {
                    out.push(Act::GoLive(i));
                    out.push(Act::StartFail(i));
                }
                T::Body => {
                    if t.queued > 0 && !t.running {
                        out.push(Act::Dequeue(i));
                    }
                    if t.running {
                        out.push(Act::RunEnd(i));
                    }
                    out.push(Act::ToggleKeepAlive(i));
                    out.push(Act::SelfEnd(i));
                }
                T::Ending | T::Exited => {}
            }
            // The body sees EOF once the retained input is dropped and no connection holds a clone.
            if t.st == T::Body && t.input_dropped && s.pinned_to(i as u64) == 0 {
                out.push(Act::Eof(i));
            }
            if t.st == T::Ending {
                out.push(Act::Exit(i));
            }
        }
        if s.now < TICKS {
            out.push(Act::Tick);
        }
        if s.drain == Drain::Off {
            out.push(Act::Reap);
            out.push(Act::BeginDrain);
        }
        if s.drain == Drain::Begun {
            if !s.deadline {
                out.push(Act::Deadline);
            }
            out.push(Act::DrainTick);
        }
        if s.drain == Drain::Done {
            out.push(Act::Shutdown);
        }
    }

    fn next_state(&self, last: &St, act: Act) -> Option<St> {
        let mut s = last.clone();
        match act {
            Act::Pin(c) | Act::Retry(c) => {
                if matches!(act, Act::Pin(_)) {
                    s.conns[c].pins_left -= 1;
                }
                let (tenant, input_closed) = self.pin_event(&s, c);
                let before = s.k.clone();
                let action = kernel::step(
                    &mut s.k,
                    Event::Pin {
                        id: ID,
                        tenant,
                        cap: None,
                        input_closed,
                    },
                );
                // P5: judged against the lifecycle and the slot before the pin.
                let live_for_c = before.slots.get(ID).is_some_and(|sl| {
                    sl.attachable()
                        && sl.tenant.as_deref() == tenant
                        && !last.obs_of(ID).input_closed
                });
                if before.lifecycle != Lifecycle::Open && matches!(action, Action::Spawned { .. }) {
                    s.bad |= V_P5;
                }
                if before.lifecycle == Lifecycle::Draining
                    && live_for_c
                    && !matches!(action, Action::Attached { .. })
                {
                    s.bad |= V_P5;
                }
                match action {
                    Action::Attached { incarnation } => {
                        s.conns[c].st = C::Pinned(incarnation);
                        s.tasks[incarnation as usize].detached_at = None;
                    }
                    Action::Spawned { incarnation } => {
                        assert_eq!(incarnation as usize, s.tasks.len(), "incarnations count up");
                        s.tasks.push(Task {
                            st: T::Starting,
                            running: false,
                            keep_alive: false,
                            queued: 0,
                            input_dropped: false,
                            tenant: s.conns[c].tenant,
                            detached_at: None,
                            first_reason: None,
                        });
                        s.conns[c].st = C::Pinned(incarnation);
                    }
                    Action::Wait => {
                        // The shell drops the slot's input and waits on its exit latch.
                        let inc = s.k.slots[ID].incarnation;
                        s.tasks[inc as usize].input_dropped = true;
                        s.conns[c].st = C::Waiting(inc);
                    }
                    Action::Forbidden | Action::AtCapacity | Action::Closed => {
                        s.conns[c].st = C::Idle;
                    }
                    other => panic!("a pin was answered {other:?}"),
                }
            }
            Act::GiveUp(c) => s.conns[c].st = C::Idle,
            Act::Send(c) => {
                let C::Pinned(i) = s.conns[c].st else {
                    return None;
                };
                s.tasks[i as usize].queued = 1;
            }
            Act::Unpin(c) | Act::Abort(c) => {
                let C::Pinned(i) = s.conns[c].st else {
                    return None;
                };
                s.conns[c].st = C::Idle;
                // The aborted attach's guard drops and unpins; the #693 leak is a connection that
                // is aborted and never does.
                if matches!(act, Act::Abort(_)) && cfg!(feature = "mut_no_unpin_on_abort") {
                    s.left(i);
                } else {
                    let slot_before = s.k.slots.get(ID).cloned();
                    kernel::step(
                        &mut s.k,
                        Event::Unpin {
                            id: ID,
                            incarnation: i,
                            now: secs(s.now),
                        },
                    );
                    // P3: the slot of any other incarnation is untouched.
                    if let Some(b) = slot_before
                        && b.incarnation != i
                        && s.k.slots.get(ID) != Some(&b)
                    {
                        s.bad |= V_P3;
                    }
                    s.left(i);
                }
            }
            Act::GoLive(i) => {
                let live = kernel::step(
                    &mut s.k,
                    Event::GoLive {
                        id: ID,
                        incarnation: i as u64,
                    },
                );
                s.tasks[i].st = if live == Action::Live(true) {
                    T::Body
                } else {
                    T::Ending
                };
            }
            Act::StartFail(i) => {
                // The lock was held elsewhere: the body is dropped and the exit guard runs.
                s.tasks[i].st = T::Ending;
                self.exit(&mut s, i);
            }
            Act::Dequeue(i) => {
                s.tasks[i].queued -= 1;
                s.tasks[i].running = true;
            }
            Act::RunEnd(i) => s.tasks[i].running = false,
            Act::ToggleKeepAlive(i) => s.tasks[i].keep_alive = !s.tasks[i].keep_alive,
            Act::SelfEnd(i) | Act::Eof(i) => {
                let t = &mut s.tasks[i];
                t.st = T::Ending;
                t.running = false;
                t.queued = 0;
            }
            Act::Exit(i) => self.exit(&mut s, i),
            Act::Tick => s.now += 1,
            Act::Reap => {
                let st = s.clone();
                let obs = |id: &str| st.obs_of(id);
                let action = kernel::step(
                    &mut s.k,
                    Event::Reap {
                        now: secs(s.now),
                        timeout: secs(TIMEOUT),
                        obs: &obs,
                    },
                );
                let Action::Stopped(ids) = action else {
                    panic!("a reap was answered {action:?}")
                };
                // P4, judged against the facts, for every session the reaper stopped whose loop is
                // alive (a closed input is stopped whatever its connections or clock say).
                for id in &ids {
                    let i = s.k.slots[id.as_str()].incarnation;
                    let o = last.obs(i as usize);
                    let t = &last.tasks[i as usize];
                    let idle = t
                        .detached_at
                        .is_some_and(|d| last.now.saturating_sub(d) >= TIMEOUT);
                    if !o.input_closed
                        && (last.pinned_to(i) > 0 || o.running || o.keep_alive || o.queued || !idle)
                    {
                        s.bad |= V_P4;
                    }
                }
                s.dropped_input(&ids);
            }
            Act::BeginDrain => {
                kernel::step(&mut s.k, Event::BeginDrain);
                s.drain = Drain::Begun;
            }
            Act::Deadline => s.deadline = true,
            Act::DrainTick => {
                let st = s.clone();
                let obs = |id: &str| st.obs_of(id);
                let action = kernel::step(
                    &mut s.k,
                    Event::DrainTick {
                        deadline_passed: s.deadline,
                        obs: &obs,
                    },
                );
                if action == Action::DrainDone {
                    // P6: the drain is over only once the deadline passed or nothing is in flight.
                    if !s.deadline && last.busy() {
                        s.bad |= V_P6;
                    }
                    s.drain = Drain::Done;
                }
            }
            Act::Shutdown => {
                let action = kernel::step(&mut s.k, Event::Shutdown);
                let Action::Stopped(ids) = action else {
                    panic!("a shutdown was answered {action:?}")
                };
                s.dropped_input(&ids);
                s.drain = Drain::Closed;
            }
        }
        s.note_reasons();
        Some(s)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        vec![
            Property::<Self>::always(
                "P1: at most one session task per id has not exited",
                |_, s| s.tasks.iter().filter(|t| t.st != T::Exited).count() <= 1,
            ),
            Property::<Self>::always(
                "P2: a session's attached count is the connections pinned to it",
                |_, s| {
                    s.k.slots
                        .values()
                        .all(|sl| sl.attached == s.pinned_to(sl.incarnation))
                },
            ),
            Property::<Self>::always(
                "P3: a stale unpin changes nothing of the new session",
                |_, s| s.bad & V_P3 == 0,
            ),
            Property::<Self>::always(
                "P4: the reaper never reaps an attached, running, subscribed or queued session",
                |_, s| s.bad & V_P4 == 0,
            ),
            Property::<Self>::always(
                "P5: a drain spawns no new id and still admits a reconnect",
                |_, s| s.bad & V_P5 == 0,
            ),
            Property::<Self>::always(
                "P6: a drain does not finish before its deadline while work is in flight",
                |_, s| s.bad & V_P6 == 0,
            ),
            Property::<Self>::always("P7: the first end reason wins", |_, s| {
                s.k.slots.values().all(|sl| {
                    let t = &s.tasks[sl.incarnation as usize];
                    t.first_reason.is_none_or(|r| r == sl.end_reason)
                })
            }),
            Property::<Self>::always(
                "P8: a connection is only attached to a session its tenant spawned",
                |_, s| {
                    s.conns.iter().all(|c| match c.st {
                        C::Pinned(i) => s.tasks[i as usize].tenant == c.tenant,
                        _ => true,
                    })
                },
            ),
            Property::<Self>::always(
                "L1: once a session's task has exited a pin for the id never waits",
                |_, s| {
                    if s.k.lifecycle != Lifecycle::Open || s.tasks.iter().any(|t| t.st != T::Exited)
                    {
                        return true;
                    }
                    let mut k = s.k.clone();
                    let action = kernel::step(
                        &mut k,
                        Event::Pin {
                            id: ID,
                            tenant: Some("A"),
                            cap: None,
                            input_closed: false,
                        },
                    );
                    action != Action::Wait
                },
            ),
            Property::<Self>::always(
                "L2: once no run is in flight a drain poll finishes the drain",
                |_, s| {
                    if s.drain != Drain::Begun || s.busy() {
                        return true;
                    }
                    let mut k = s.k.clone();
                    let obs = |id: &str| s.obs_of(id);
                    kernel::step(
                        &mut k,
                        Event::DrainTick {
                            deadline_passed: false,
                            obs: &obs,
                        },
                    ) == Action::DrainDone
                },
            ),
            Property::<Self>::always(
                "L3: a detached idle session past the timeout is stopped by the next reaper tick",
                |_, s| {
                    if s.drain != Drain::Off {
                        return true;
                    }
                    let Some(sl) = s.k.slots.get(ID) else {
                        return true;
                    };
                    let i = sl.incarnation;
                    let t = &s.tasks[i as usize];
                    let o = s.obs(i as usize);
                    let idle = s.pinned_to(i) == 0
                        && t.detached_at
                            .is_some_and(|d| s.now.saturating_sub(d) >= TIMEOUT)
                        && !o.running
                        && !o.keep_alive
                        && !o.queued;
                    if !idle || !sl.attachable() {
                        return true;
                    }
                    let mut k = s.k.clone();
                    let obs = |id: &str| s.obs_of(id);
                    kernel::step(
                        &mut k,
                        Event::Reap {
                            now: secs(s.now),
                            timeout: secs(TIMEOUT),
                            obs: &obs,
                        },
                    );
                    k.slots[ID].phase == Phase::Stopping
                },
            ),
        ]
    }

    fn within_boundary(&self, s: &St) -> bool {
        s.tasks.len() <= INCARNATIONS
    }
}

impl Serve {
    /// The exit guard of task `i`: the kernel removes its own entry and the latch is cancelled.
    fn exit(&self, s: &mut St, i: usize) {
        kernel::step(
            &mut s.k,
            Event::Exited {
                id: ID,
                incarnation: i as u64,
            },
        );
        let t = &mut s.tasks[i];
        t.st = T::Exited;
        t.running = false;
        t.queued = 0;
    }
}

fn check(other_tenant: bool, label: &str) {
    let checker = Serve { other_tenant }.checker().spawn_bfs().join();
    println!(
        "{label}: {} states, {} unique",
        checker.state_count(),
        checker.unique_state_count()
    );
    checker.assert_properties();
}

/// Every property holds over every interleaving, both connections of one tenant ...
#[test]
fn faithful_one_tenant() {
    check(false, "one tenant");
}

/// ... and with the second connection of another tenant.
#[test]
fn faithful_two_tenants() {
    check(true, "two tenants");
}

/// With a `mut_*` feature on, the property that MUST be red for it, and whether the second
/// connection is of another tenant. (The faithful tests above fail too under any mutation; this
/// one names the property.)
#[cfg(any(
    feature = "mut_no_incarnation_guard",
    feature = "mut_no_unpin_on_abort",
    feature = "mut_reap_ignores_queued",
    feature = "mut_reap_ignores_running",
    feature = "mut_reap_ignores_keep_alive",
    feature = "mut_spawn_while_stopping",
    feature = "mut_admit_while_draining",
    feature = "mut_end_reason_overwritten",
    feature = "mut_no_tenant_check",
    feature = "mut_exit_keeps_entry",
    feature = "mut_drain_never_done"
))]
#[test]
fn the_mutation_turns_its_property_red() {
    let (other, property) = if cfg!(feature = "mut_no_incarnation_guard") {
        (false, "P3: ")
    } else if cfg!(feature = "mut_no_unpin_on_abort") {
        (false, "P2: ")
    } else if cfg!(feature = "mut_reap_ignores_queued")
        || cfg!(feature = "mut_reap_ignores_running")
        || cfg!(feature = "mut_reap_ignores_keep_alive")
    {
        (false, "P4: ")
    } else if cfg!(feature = "mut_spawn_while_stopping") {
        (false, "P1: ")
    } else if cfg!(feature = "mut_admit_while_draining") {
        (false, "P5: ")
    } else if cfg!(feature = "mut_end_reason_overwritten") {
        (false, "P7: ")
    } else if cfg!(feature = "mut_no_tenant_check") {
        (true, "P8: ")
    } else if cfg!(feature = "mut_exit_keeps_entry") {
        (false, "L1: ")
    } else {
        (false, "L2: ")
    };
    let model = Serve {
        other_tenant: other,
    };
    let checker = model.checker().spawn_bfs().join();
    let red: Vec<&str> = model
        .properties()
        .iter()
        .filter(|p| p.name.starts_with(property) && checker.discovery(p.name).is_some())
        .map(|p| p.name)
        .collect();
    println!("red: {red:?}");
    assert!(!red.is_empty(), "no property `{property}` went red");
}

/// The mutation (b)'s second witness: the leak also strands the reaper (L3).
#[cfg(feature = "mut_no_unpin_on_abort")]
#[test]
fn the_leak_also_turns_l3_red() {
    let model = Serve {
        other_tenant: false,
    };
    let checker = model.checker().spawn_bfs().join();
    assert!(
        checker
            .discovery(
                "L3: a detached idle session past the timeout is stopped by the next reaper tick"
            )
            .is_some()
    );
}

/// Production's `Supervisor` makes its lifecycle decisions through the kernel: a source-text check
/// (the shell is private async code with no other seam). Each call below is one decision, and
/// nothing else writes the kernel's state.
#[test]
fn production_calls_the_kernel() {
    let src = include_str!("../src/serve_ws.rs");
    let prod = &src[..src.find("#[cfg(test)]\nmod tests").expect("test module")];
    for needle in [
        "kernel::Event::Pin {",
        "kernel::Event::Unpin {",
        "kernel::Event::GoLive {",
        "kernel::Event::Exited {",
        "kernel::Event::Reap {",
        "kernel::Event::BeginDrain",
        "kernel::Event::DrainTick {",
        "kernel::Event::Shutdown",
    ] {
        assert!(
            prod.contains(needle),
            "serve_ws.rs no longer calls {needle}"
        );
    }
    // The guard that unpins on every exit goes through `unpin`, the kernel's incarnation-guarded call.
    let guard = &prod[prod.find("impl Drop for PinGuard").expect("PinGuard")..];
    let guard = &guard[..guard.find("\n}\n").expect("end of impl")];
    assert!(
        guard.contains("self.supervisor.unpin(&self.id, self.incarnation)"),
        "PinGuard::drop no longer unpins"
    );
    // No decision of the lifecycle lives outside the kernel: the table's state is only written there.
    for banned in [
        ".attached +=",
        ".attached =",
        ".last_detached_at =",
        ".lifecycle =",
        ".next_incarnation",
        ".end_reason",
        "Phase::Stopping",
    ] {
        assert!(
            !prod.contains(banned),
            "serve_ws.rs writes the session table outside the kernel: {banned}"
        );
    }
}
