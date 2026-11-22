//! Stateright model of execd's OS boundary: the order between the job kernel's decisions and the
//! processes they name. `job_model.rs` is the table; this is the group numbers under it.
//!
//! ## What it checks
//!
//! The OS actor: a job's processes exit at any moment (a leader and one descendant), the reaper
//! collects them (one atomic pass, as `proc::sweep` holds `SPAWNING` exclusively), the group's number
//! is free once its last member is collected, and a foreign process group may then take the number
//! and give it up again. A signal is two steps: the kernel decides (`Ev::Signal`, or `Ev::Stop` for
//! `kill`/`wipe`) and the shell delivers later, by number, with any interleaving between (the pump
//! drain, the grace wait). Delivery is one atomic step, as `proc::signal_job` holds `SPAWNING` shared.
//!
//! Every decision is the REAL kernel's: `State`/`step` for the job's life, `Held::{insert, freed}`
//! for the ledger a spawn and a reaper pass keep, `may_deliver` for the check a delivery makes.
//! The model owns only the processes and the group number.
//!
//! - J11 no signal is ever delivered to a process group that is not the job's: not to a number the
//!   reaper freed, not to a foreign group that took it, and not to a later job's group that took the
//!   same number (the counterexample this model found in the first, number-only ledger: a decision
//!   outlived its group, and the successor's spawn re-held the number).
//! - J11-live (the completeness half, the liveness partner of J11's refusal): a group of a job that
//!   still has a member is always held, so a signal to it is never refused and a retry after any
//!   refusal succeeds exactly while there is something to signal. (Stated as a state invariant, the
//!   bounded measure `k = 0`: nothing delays it.)
//! - sanity: a signal reaches a live group of its own (the model is not vacuous).
//!
//! Not modelled: the supervisor's by-number observation of the group (`job_gone`) when a foreign group
//! holds the number, which delays the end (a liveness gap, named in `ARCHITECTURE.md`), and a host
//! without the subreaper (where init reaps orphans outside the exclusion).
//!
//! `cargo test -p beyond-execd --features <mut> --test os_model the_mutation` MUST pass only when the
//! named property is red.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use execd::job_kernel::{self as kernel, Act, Ev, Held, State};
use stateright::{Checker, Model, Property};

/// The one group number the model has: the number a job's group, then a foreign group, may carry.
const P: i32 = 7;

/// Who carries the number `P`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Owner {
    Free,
    /// Job `j`'s group, with its members: running, and exited but not yet collected.
    Job(usize, u8, u8),
    /// A process group that is not execd's, which took the number after it was freed.
    Foreign,
}

/// A decision the shell took and has not yet delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pending {
    job: usize,
    /// The number the kernel's state held at the decision.
    pgid: i32,
    /// The spawn it was (the job's own token; a later spawn on the same number has another).
    sp: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct St {
    jobs: Vec<State>,
    held: Held,
    owner: Owner,
    pending: Vec<Pending>,
    /// A signal reached a group that is not the job's.
    foreign_hit: bool,
    /// A signal reached a live group of its own job.
    delivered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum A {
    Spawn(usize),
    /// A member exits.
    Exit,
    /// One reaper pass.
    Reap,
    /// The supervisor saw the job's group empty.
    End(usize),
    /// The kernel decides a signal (`false`) or a stop (`true`).
    Decide(usize, bool),
    Deliver(usize),
    ForeignTakes,
    ForeignExits,
}

#[derive(Clone, Copy)]
struct Os {
    jobs: usize,
    max_pending: usize,
}

impl Os {
    fn group_exists(owner: Owner) -> bool {
        match owner {
            Owner::Free => false,
            Owner::Job(_, a, z) => a + z > 0,
            Owner::Foreign => true,
        }
    }
}

impl Model for Os {
    type State = St;
    type Action = A;

    fn init_states(&self) -> Vec<St> {
        vec![St {
            jobs: (0..self.jobs).map(|_| State::new(0, false)).collect(),
            held: Held::new(),
            owner: Owner::Free,
            pending: Vec::new(),
            foreign_hit: false,
            delivered: false,
        }]
    }

    fn actions(&self, s: &St, out: &mut Vec<A>) {
        for (j, k) in s.jobs.iter().enumerate() {
            if s.owner == Owner::Free && k.pgid() == 0 && !k.ended() {
                out.push(A::Spawn(j));
            }
            let mine = matches!(s.owner, Owner::Job(o, ..) if o == j);
            if k.pgid() != 0 && !k.ended() && !mine {
                out.push(A::End(j));
            }
            if s.pending.len() < self.max_pending && k.pgid() != 0 {
                out.push(A::Decide(j, false));
                out.push(A::Decide(j, true));
            }
            if s.pending.iter().any(|p| p.job == j) {
                out.push(A::Deliver(j));
            }
        }
        if matches!(s.owner, Owner::Job(_, a, _) if a > 0) {
            out.push(A::Exit);
        }
        out.push(A::Reap);
        if s.owner == Owner::Free {
            out.push(A::ForeignTakes);
        }
        if s.owner == Owner::Foreign {
            out.push(A::ForeignExits);
        }
    }

    fn next_state(&self, last: &St, a: A) -> Option<St> {
        let mut s = last.clone();
        match a {
            A::Spawn(j) => {
                // `proc::spawn`: the group exists, and is held, under the exclusion that keeps the
                // reaper out; then the supervisor reports the spawn.
                s.owner = Owner::Job(j, 2, 0);
                s.held.insert(P, j as u64 + 1);
                kernel::step(&mut s.jobs[j], Ev::Spawned { pgid: P });
            }
            A::Exit => {
                if let Owner::Job(j, a, z) = s.owner {
                    s.owner = Owner::Job(j, a - 1, z + 1);
                }
            }
            A::Reap => {
                // `proc::sweep`: collect every zombie; the number is free with the last member; then,
                // inside the same exclusion, stop holding every group that is gone.
                if let Owner::Job(j, a, _) = s.owner {
                    s.owner = if a == 0 {
                        Owner::Free
                    } else {
                        Owner::Job(j, a, 0)
                    };
                }
                for g in s.held.numbers() {
                    if !Self::group_exists(s.owner) {
                        s.held.freed(g);
                    }
                }
            }
            A::End(j) => {
                // The group is observed gone: the end is recorded (a pump drain may come first, which
                // is the delay `Decide`/`Deliver` already interleave with).
                kernel::step(
                    &mut s.jobs[j],
                    Ev::GroupEmpty {
                        code: 0,
                        signal: None,
                        now: 0,
                    },
                );
            }
            A::Decide(j, stop) => {
                let ev = if stop { Ev::Stop } else { Ev::Signal };
                let act = kernel::step(&mut s.jobs[j], ev);
                if act == Act::Signal || act == Act::Stop {
                    let pgid = s.jobs[j].pgid();
                    s.pending.push(Pending {
                        job: j,
                        pgid,
                        sp: j as u64 + 1,
                    });
                }
            }
            A::Deliver(j) => {
                let at = s.pending.iter().position(|p| p.job == j)?;
                let p = s.pending.remove(at);
                // `proc::signal_job`: under the exclusion, ask the kernel whether the number is
                // still held, and `kill` it by number if so.
                if kernel::may_deliver(&s.held, p.pgid, p.sp) {
                    match s.owner {
                        Owner::Job(o, ..) if o == p.job => s.delivered = true,
                        Owner::Free => {}
                        _ => s.foreign_hit = true,
                    }
                }
            }
            A::ForeignTakes => s.owner = Owner::Foreign,
            A::ForeignExits => s.owner = Owner::Free,
        }
        Some(s)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        vec![
            Property::always(
                "J11: no signal reaches a group that is not the job's",
                |_, s: &St| !s.foreign_hit,
            ),
            Property::always(
                "J11-live: a group with a member is always held, so it is never refused",
                |_, s: &St| match s.owner {
                    Owner::Job(j, ..) => s.held.has(P, j as u64 + 1),
                    _ => true,
                },
            ),
            Property::sometimes("J11-sanity: a signal reaches a live group", |_, s: &St| {
                s.delivered
            }),
        ]
    }
}

/// Two jobs in turn on one number (the second takes it after the first's end), two signals in flight.
const OS: Os = Os {
    jobs: 2,
    max_pending: 2,
};

fn check(m: Os) {
    let checker = m.checker().spawn_bfs().join();
    println!(
        "{} states, {} unique",
        checker.state_count(),
        checker.unique_state_count()
    );
    for p in m.properties() {
        let found = checker.discovery(p.name);
        match p.expectation {
            stateright::Expectation::Sometimes => {
                assert!(found.is_some(), "`{}` never happens", p.name)
            }
            _ => {
                if let Some(path) = found {
                    panic!("`{}` is red:\n{:#?}", p.name, path.into_actions());
                }
            }
        }
    }
}

/// Every interleaving of exits, reaper passes, foreign reuse, and decisions delivered late.
#[cfg(not(any(feature = "mut_signal_after_reap", feature = "mut_held_forgets")))]
#[test]
fn faithful_os_boundary() {
    check(OS);
}

/// With a `mut_*` feature on, the property that MUST be red for it.
#[cfg(any(feature = "mut_signal_after_reap", feature = "mut_held_forgets"))]
#[test]
fn the_mutation_turns_its_property_red() {
    let property = if cfg!(feature = "mut_signal_after_reap") {
        "J11: "
    } else {
        "J11-live: "
    };
    let m = OS;
    let checker = m.checker().spawn_bfs().join();
    let red: Vec<&str> = m
        .properties()
        .iter()
        .filter(|p| p.name.starts_with(property) && checker.discovery(p.name).is_some())
        .map(|p| p.name)
        .collect();
    println!("red: {red:?}");
    assert!(!red.is_empty(), "no property `{property}` went red");
}

/// Production's reaper and signals go through the kernel's ledger: a source-text check (the shell
/// is blocking code on `/proc` and signals, with no other seam). Nothing signals a group by number
/// outside `signal_held_group`, and nothing writes the ledger but a spawn and a reaper pass.
#[test]
fn production_signals_only_held_groups() {
    let src = include_str!("../src/proc.rs");
    let prod = &src[..src
        .find("#[cfg(test)]\n#[allow(clippy::unwrap_used")
        .expect("test module")];
    for needle in [
        "h.insert(pgid, sp)",
        "h.freed(g)",
        "job_kernel::may_deliver(h, pg.id, pg.sp)",
        "fn no_reaping()",
        "let _no_reaping = no_reaping();",
    ] {
        assert!(prod.contains(needle), "proc.rs no longer has {needle}");
    }
    // `killpg(pgid, <signal>)` appears once: behind the kernel's check. (`killpg(.., None)` is the
    // existence probe, which sends nothing.)
    let sends = prod.matches("killpg(Pid::from_raw(pg.id), sig)").count();
    assert_eq!(sends, 1, "a signal by number bypasses `may_deliver`");
    let at = prod.find("killpg(Pid::from_raw(pg.id), sig)").unwrap();
    assert!(
        prod[..at]
            .rfind("job_kernel::may_deliver(")
            .is_some_and(|a| at - a < 300),
        "the send is no longer behind `may_deliver`"
    );
}
