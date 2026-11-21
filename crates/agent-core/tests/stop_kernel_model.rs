//! Stateright model of the end-of-turn decision in `Agent::run_events_steered`
//! (`src/stop_kernel.rs`; ARCHITECTURE.md, "The end-of-turn kernel").
//!
//! ## Faithfulness
//!
//! Every decision is made by the REAL kernel: [`stop_kernel::decide`] (which loops
//! [`stop_kernel::step`]) for the stop, [`stop_kernel::limit_reached`] for the top-of-loop
//! `MaxSteps` check, and [`Outcome::steered`]/[`Outcome::background`] for the events owed. The
//! model owns only the environment: the steering queue, finish notices, background work, and the
//! turn the model takes. Its [`Effects`] impl is the same seam production implements over the
//! steering queue and tool sources. The properties compare against plain arithmetic (`steps >=
//! LIMIT`), never against the kernel's own predicate.
//!
//! Transitions match the loop one for one: a turn first runs the top-of-loop limit check (error
//! `MaxSteps` or take the step); a tool turn only takes the step; a tool-less turn takes the step
//! and makes the stop decision (the graceful-stop and refusal exits above it never reach the
//! kernel and are not modelled); inputs arrive at any time between turns. Checked by reading
//! `agent.rs` against each: the loop top (`limit_reached`), the step increment, and the stop block
//! are the three places `steps_this_call`, the queues and the latch are read. Not modelled: the
//! wrap-up grant (`BudgetState::after_tools` raises the limit; the kernel only reads the current
//! limit as input) and the tool-results path's own notice delivery.
//!
//! ## Properties
//!
//! S1 a stop never continues at the step limit (#693). S2 below the limit the run never ends with a
//! follow-up or notice pending. S3 at the limit nothing queued is consumed. S4 at most one stop
//! reminder per run. S5 every delivery is reported: each drained follow-up in a `Steered` count,
//! each taken notice in a `Background`. S0 the step limit holds (no step past it).
//! L1 (liveness, a bounded measure under `always`): every continue strictly decreases
//! `pending follow-ups + pending notices + (work unlisted and no reminder given)`, so with no new
//! input the run ends within `FOLLOWUPS + NOTICES + 1` continues.
//!
//! ## Mutations
//!
//! `cargo test -p beyond-agent-core --test stop_kernel_model --features mut_drain_at_limit` (and
//! `mut_no_reminder_latch`, `mut_no_notice_delivery`, `mut_no_limit_check`) must fail.

#![allow(clippy::expect_used, clippy::panic)]

use agent_core::stop_kernel::{self, Effects, Note, Outcome, StopState};
use stateright::{Checker, Model, Property};

const MAX_FOLLOWUPS: u8 = 2;
const MAX_NOTICES: u8 = 2;
const MAX_STARTS: u8 = 2;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct S {
    steps: u32,
    stop: StopState,
    // Environment.
    followups: u8,
    notices: u8,
    running: bool,
    /// Work started and not yet covered by a reminder (a reminder consumes it).
    unlisted: bool,
    queued: u8,
    noticed: u8,
    starts: u8,
    done: Done,
    // Bookkeeping for the properties.
    reminders: u8,
    drained: usize,
    steered: usize,
    notice_taken: u8,
    notice_reported: u8,
    continue_at_limit: bool,
    ended_with_pending: bool,
    consumed_at_limit: bool,
    measure_not_decreasing: bool,
    quiet_continues: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Done {
    Running,
    Ended,
    MaxSteps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Act {
    QueueFollowup,
    DeliverNotice,
    StartWork,
    StopWork,
    /// A model turn; `tools` = it requested tools; `drain_all` = the queue mode (`All` or
    /// `OneAtATime`) for this stop's drain.
    Turn {
        tools: bool,
        drain_all: bool,
    },
}

struct Stop {
    /// `None` = no ceiling (steps bounded by the model only).
    limit: Option<u32>,
}

/// The environment behind the kernel's effects.
struct Env<'a> {
    s: &'a mut S,
    drain_all: bool,
}

impl Effects for Env<'_> {
    fn drain(&mut self) -> usize {
        let n = if self.drain_all {
            self.s.followups
        } else {
            self.s.followups.min(1)
        };
        self.s.followups -= n;
        self.s.drained += n as usize;
        n as usize
    }
    fn take_notices(&mut self) -> bool {
        let any = self.s.notices > 0;
        if any {
            self.s.notices = 0;
            self.s.notice_taken += 1;
        }
        any
    }
    fn stop_reminder(&mut self) -> bool {
        let any = self.s.running && self.s.unlisted;
        if any {
            self.s.unlisted = false;
        }
        any
    }
}

fn measure(s: &S) -> u32 {
    s.followups as u32 + s.notices as u32 + (s.running && s.unlisted && !s.stop.reminded) as u32
}

impl Stop {
    fn at_limit(&self, steps: u32) -> bool {
        self.limit.is_some_and(|l| steps >= l)
    }
}

impl Model for Stop {
    type State = S;
    type Action = Act;

    fn init_states(&self) -> Vec<S> {
        vec![S {
            steps: 0,
            stop: StopState::new(),
            followups: 0,
            notices: 0,
            running: false,
            unlisted: false,
            queued: 0,
            noticed: 0,
            starts: 0,
            done: Done::Running,
            reminders: 0,
            drained: 0,
            steered: 0,
            notice_taken: 0,
            notice_reported: 0,
            continue_at_limit: false,
            ended_with_pending: false,
            consumed_at_limit: false,
            measure_not_decreasing: false,
            quiet_continues: 0,
        }]
    }

    fn actions(&self, s: &S, out: &mut Vec<Act>) {
        if s.done != Done::Running {
            return;
        }
        if s.queued < MAX_FOLLOWUPS {
            out.push(Act::QueueFollowup);
        }
        if s.noticed < MAX_NOTICES {
            out.push(Act::DeliverNotice);
        }
        if !s.running && s.starts < MAX_STARTS {
            out.push(Act::StartWork);
        }
        if s.running {
            out.push(Act::StopWork);
        }
        // Model bound: a run without a ceiling (or one the kernel failed to stop) takes at most
        // this many steps.
        if s.steps < self.limit.unwrap_or(3) + 2 {
            for tools in [true, false] {
                for drain_all in [true, false] {
                    if tools && !drain_all {
                        continue; // the queue mode only matters at a stop
                    }
                    out.push(Act::Turn { tools, drain_all });
                }
            }
        }
    }

    fn next_state(&self, s: &S, a: Act) -> Option<S> {
        let mut n = s.clone();
        match a {
            Act::QueueFollowup => {
                n.followups += 1;
                n.queued += 1;
                n.quiet_continues = 0;
            }
            Act::DeliverNotice => {
                n.notices += 1;
                n.noticed += 1;
                n.quiet_continues = 0;
            }
            Act::StartWork => {
                n.running = true;
                n.unlisted = true;
                n.starts += 1;
                n.quiet_continues = 0;
            }
            Act::StopWork => n.running = false,
            Act::Turn { tools, drain_all } => {
                if stop_kernel::limit_reached(n.steps, self.limit) {
                    n.done = Done::MaxSteps;
                    return Some(n);
                }
                n.steps += 1;
                if tools {
                    return Some(n);
                }
                let before = (n.followups, n.notices);
                let m0 = measure(&n);
                let mut stop = n.stop;
                let outcome = {
                    let mut env = Env {
                        s: &mut n,
                        drain_all,
                    };
                    stop_kernel::decide(&mut stop, env.s.steps, self.limit, &mut env)
                };
                n.stop = stop;
                let at_limit = self.at_limit(n.steps);
                match outcome {
                    Outcome::End => {
                        n.done = Done::Ended;
                        if !at_limit && (n.followups > 0 || n.notices > 0) {
                            n.ended_with_pending = true;
                        }
                    }
                    Outcome::Continue { .. } => {
                        if at_limit {
                            n.continue_at_limit = true;
                        }
                        if measure(&n) >= m0 {
                            n.measure_not_decreasing = true;
                        }
                        n.quiet_continues += 1;
                    }
                }
                if at_limit && (n.followups, n.notices) != before {
                    n.consumed_at_limit = true;
                }
                if outcome.background() == Some(Note::StillRunning) {
                    n.reminders += 1;
                }
                if let Some(c) = outcome.steered() {
                    n.steered += c;
                }
                if outcome.background() == Some(Note::Finished) {
                    n.notice_reported += 1;
                }
            }
        }
        Some(n)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        vec![
            Property::always("S0 the step limit holds", |m: &Stop, s: &S| {
                m.limit.is_none_or(|l| s.steps <= l)
            }),
            Property::always("S1 no continue at the step limit", |_, s: &S| {
                !s.continue_at_limit
            }),
            Property::always("S2 no end with a delivery pending", |_, s: &S| {
                !s.ended_with_pending
            }),
            Property::always("S3 nothing consumed at the step limit", |_, s: &S| {
                !s.consumed_at_limit
            }),
            Property::always("S4 at most one stop reminder", |_, s: &S| s.reminders <= 1),
            Property::always("S5 every delivery is reported", |_, s: &S| {
                s.steered == s.drained && s.notice_reported == s.notice_taken
            }),
            Property::always("L1 every continue decreases the measure", |_, s: &S| {
                !s.measure_not_decreasing && s.quiet_continues <= MAX_FOLLOWUPS + MAX_NOTICES + 1
            }),
        ]
    }
}

fn check(limit: Option<u32>, name: &'static str) {
    let c = Stop { limit }.checker().spawn_bfs().join();
    // A property named in a failure is the one that went red.
    if let Some(p) = c.discovery(name) {
        panic!("RED {name} (limit {limit:?}):\n{:?}", p.into_actions());
    }
    assert!(c.is_done(), "model did not terminate");
}

macro_rules! prop_tests {
    ($($test:ident => $name:literal),* $(,)?) => {$(
        #[test]
        fn $test() {
            for limit in [Some(1), Some(2), Some(3), None] {
                check(limit, $name);
            }
        }
    )*};
}

prop_tests! {
    s0_the_step_limit_holds => "S0 the step limit holds",
    s1_no_continue_at_the_step_limit => "S1 no continue at the step limit",
    s2_no_end_with_a_delivery_pending => "S2 no end with a delivery pending",
    s3_nothing_consumed_at_the_step_limit => "S3 nothing consumed at the step limit",
    s4_at_most_one_stop_reminder => "S4 at most one stop reminder",
    s5_every_delivery_is_reported => "S5 every delivery is reported",
    l1_every_continue_decreases_the_measure => "L1 every continue decreases the measure",
}

/// `run_events_steered` makes its stop decision through the kernel: a source-text check (the repo
/// has no other precedent for a structure test of a private async loop).
#[test]
fn production_calls_the_kernel() {
    let src = include_str!("../src/agent.rs");
    let prod = &src[..src.find("#[cfg(test)]").expect("test module")];
    for needle in [
        "crate::stop_kernel::decide(",
        "crate::stop_kernel::limit_reached(",
        "outcome.steered()",
    ] {
        assert!(prod.contains(needle), "agent.rs no longer calls {needle}");
    }
    // The decision's inputs are read only through the kernel's effects.
    assert_eq!(
        prod.matches(".drain_at_stop()").count(),
        1,
        "one drain, in StopFx"
    );
    assert!(
        !prod.contains("stop_reminded"),
        "the latch lives in StopState"
    );
}
