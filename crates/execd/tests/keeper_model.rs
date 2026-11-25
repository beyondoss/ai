//! Stateright model of a terminal's keeper: `execd_jobs::keeper::step` over an environment of a shell
//! that prints and exits, a master that reads `EAGAIN` or `EIO`, and an execd that connects, loses
//! the connection, connects again (a restart) and retries a positional write across it.
//!
//! Every decision is made by the REAL kernel: `keeper::step` for each event and `stdin_plan` inside
//! it. The model owns only the environment (the master's queue, the slave's holders, the bytes that
//! reached the terminal) and performs what the kernel's actions name.
//!
//! ## Properties
//!
//! - T5 input lands once and in order: what reached the terminal is exactly the stream's first
//!   `written` bytes, however the writes are retried, replayed or reordered, a reconnect between a
//!   write and its answer included.
//! - T5-live (bounded measure): a write at or below the stream's end, retried after any number of
//!   reconnects, lands within one step per byte it still owes.
//! - K1 the shell's exit is told only after everything it printed before exiting was copied to the
//!   output.
//! - K2 once the shell is reaped and the master reads `EIO`, the keeper ends in that very step.
//! - Q6 a spool that cannot be written ends the terminal: the shell is killed once, and the status told
//!   is the failure's, whatever the shell did.
//!
//! Each `mut_*` of the keeper kernel (`the_mutation_turns_its_property_red`) turns one red.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use execd::job_kernel::Size;
use execd::job_kernel::keeper::{K, KAct, KEv, SPOOL_FAILED_CODE, Shell, exit_status, start, step};
use stateright::{Checker, Model, Property};

/// The stream execd means to write: byte `i` is the value `i`, so a replay or a skip is visible.
const STREAM: u8 = 3;
/// What the shell may print before exiting, and after (a background writer on the slave).
const PRINTS: u8 = 2;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct St {
    k: K,
    /// The keeper ended.
    gone: bool,
    /// The shell's own state in the environment (the kernel learns it by `ChildExit`).
    shell_alive: bool,
    /// Some process holds the slave: the master reads `EAGAIN`, not `EIO`.
    slave: bool,
    /// Bytes the terminal printed that the keeper has not read, and how many of those were printed
    /// before the shell exited.
    queue: u8,
    owed: u8,
    printed: u8,
    /// Bytes copied to the spool.
    copied: u8,
    /// What reached the terminal's input, in order.
    delivered: Vec<u8>,
    /// A write the keeper is applying: the stream positions still to land.
    pending: Vec<u8>,
    /// Connections execd made, so far.
    conns: u8,
    /// How many times the kernel told the keeper to kill the shell.
    kills: u8,
    /// A write to the spool failed (the environment's fact; the kernel is told by `SpoolFailed`).
    broke: bool,
    bad: u8,
}

const V_K1: u8 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum A {
    Print,
    ShellExits,
    SlaveCloses,
    ReadMaster,
    Connect,
    /// The process that started the keeper ends before it ever connected.
    ParentDies,
    Lose,
    /// execd sends the stream's bytes `[at, at + len)`.
    Send(u8, u8),
    Land,
    /// A write of the output to the spool fails.
    SpoolFails,
}

struct M;

impl St {
    fn run(&mut self, ev: KEv) -> KAct {
        let act = step(&mut self.k, ev);
        if act.send_exit && self.owed > 0 {
            self.bad |= V_K1;
        }
        if act.exit {
            self.gone = true;
        }
        if act.kill_shell {
            self.kills += 1;
        }
        act
    }
}

impl Model for M {
    type State = St;
    type Action = A;

    fn init_states(&self) -> Vec<St> {
        vec![St {
            k: start(Size { cols: 80, rows: 24 }),
            gone: false,
            shell_alive: true,
            slave: true,
            queue: 0,
            owed: 0,
            printed: 0,
            copied: 0,
            delivered: Vec::new(),
            pending: Vec::new(),
            conns: 0,
            kills: 0,
            broke: false,
            bad: 0,
        }]
    }

    fn actions(&self, s: &St, out: &mut Vec<A>) {
        if s.gone {
            return;
        }
        if s.slave && s.printed < PRINTS {
            out.push(A::Print);
        }
        if s.shell_alive {
            out.push(A::ShellExits);
        }
        if s.slave && !s.shell_alive {
            out.push(A::SlaveCloses);
        }
        out.push(A::ReadMaster);
        if !s.k.conn && s.conns < 2 {
            out.push(A::Connect);
        }
        if !s.k.seen && !s.k.orphan {
            out.push(A::ParentDies);
        }
        if s.k.conn {
            out.push(A::Lose);
            for at in 0..STREAM {
                for len in 1..=STREAM - at {
                    out.push(A::Send(at, len));
                }
            }
        }
        if !s.pending.is_empty() && !s.k.eio {
            out.push(A::Land);
        }
        if !s.k.failed {
            out.push(A::SpoolFails);
        }
    }

    fn next_state(&self, last: &St, a: A) -> Option<St> {
        let mut s = last.clone();
        match a {
            A::Print => {
                s.queue += 1;
                s.printed += 1;
                if s.shell_alive {
                    s.owed += 1;
                }
            }
            A::ShellExits => {
                s.shell_alive = false;
                s.run(KEv::ChildExit {
                    code: 0,
                    signal: None,
                });
            }
            A::SlaveCloses => s.slave = false,
            A::ReadMaster => {
                if s.queue > 0 {
                    s.queue -= 1;
                    s.copied += 1;
                    s.owed = s.owed.saturating_sub(1);
                } else if s.slave {
                    s.run(KEv::MasterEagain);
                } else {
                    s.run(KEv::MasterEio);
                }
            }
            A::Connect => {
                s.conns += 1;
                s.run(KEv::Conn);
            }
            A::ParentDies => {
                s.run(KEv::ParentGone);
            }
            A::Lose => {
                s.run(KEv::ConnLost);
            }
            A::Send(at, len) => {
                let act = s.run(KEv::Write {
                    at: u64::from(at),
                    len: u64::from(len),
                });
                // A new write replaces what was pending: the keeper plans it against `written`.
                s.pending.clear();
                if let Some((skip, take)) = act.apply {
                    let from = u64::from(at) + skip;
                    s.pending = (from..from + take).map(|b| b as u8).collect();
                }
            }
            A::SpoolFails => {
                s.broke = true;
                // The keeper performs the kill the kernel names: the shell dies by `SIGKILL`.
                if s.run(KEv::SpoolFailed).kill_shell && s.shell_alive {
                    s.shell_alive = false;
                    s.run(KEv::ChildExit {
                        code: 137,
                        signal: Some(9),
                    });
                }
            }
            A::Land => {
                let byte = s.pending.remove(0);
                s.delivered.push(byte);
                s.run(KEv::Landed { n: 1 });
            }
        }
        Some(s)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        vec![
            Property::always(
                "T5: input reaches the terminal once and in order, retries and reconnects included",
                |_, s: &St| {
                    s.delivered
                        .iter()
                        .enumerate()
                        .all(|(i, b)| usize::from(*b) == i)
                        && s.k.written == s.delivered.len() as u64
                },
            ),
            Property::always(
                "K1: the shell's exit is told only after what it printed is copied",
                |_, s: &St| s.bad & V_K1 == 0,
            ),
            Property::always(
                "K2b: a keeper does not end before execd could ask, unless its starter is gone",
                |_, s: &St| !s.gone || s.k.seen || s.k.orphan,
            ),
            Property::always(
                "K2: a reaped shell and a master at EIO end a keeper execd asked in that step",
                |_, s: &St| {
                    !(matches!(s.k.shell, Shell::Reaped { .. })
                        && s.k.eio
                        && (s.k.seen || s.k.orphan))
                        || s.gone
                },
            ),
            Property::always(
                "Q6: a failed spool kills the shell once and ends with the failure's code",
                |_, s: &St| {
                    s.kills <= 1
                        && (!s.broke || (s.k.failed && !s.shell_alive))
                        && (!(s.broke && matches!(s.k.shell, Shell::Reaped { .. }))
                            || exit_status(&s.k) == Some((SPOOL_FAILED_CODE, None)))
                },
            ),
            // T5-live, bounded: whenever a write could still be sent and the keeper is up, the bytes
            // the stream owes are fewer than `STREAM + 1`, and what is pending is exactly the next ones.
            Property::always(
                "T5-live: what a write still owes is the stream's next bytes, never a replay or a gap",
                |_, s: &St| {
                    s.pending
                        .iter()
                        .enumerate()
                        .all(|(i, b)| u64::from(*b) == s.k.written + i as u64)
                        || s.k.eio
                },
            ),
        ]
    }
}

fn check() {
    let checker = M.checker().spawn_bfs().join();
    println!(
        "{} states, {} unique",
        checker.state_count(),
        checker.unique_state_count()
    );
    for p in M.properties() {
        if let Some(path) = checker.discovery(p.name) {
            panic!("`{}` is red:\n{:#?}", p.name, path.into_actions());
        }
    }
}

#[test]
fn faithful_keeper() {
    check();
}

/// With a keeper `mut_*` on, the property that MUST be red for it.
#[cfg(any(
    feature = "mut_exit_before_drain",
    feature = "mut_keeper_waits_forever",
    feature = "mut_keeper_exits_unasked",
    feature = "mut_keeper_reconnect_resets_written",
    feature = "mut_spool_failure_silent",
    feature = "mut_spool_failure_keeps_code",
    feature = "mut_stdin_replays",
    feature = "mut_stdin_apply_off_start"
))]
#[test]
fn the_mutation_turns_its_property_red() {
    let property = if cfg!(feature = "mut_exit_before_drain") {
        "K1: "
    } else if cfg!(feature = "mut_keeper_waits_forever") {
        "K2: "
    } else if cfg!(feature = "mut_keeper_exits_unasked") {
        "K2b: "
    } else if cfg!(feature = "mut_spool_failure_silent")
        || cfg!(feature = "mut_spool_failure_keeps_code")
    {
        "Q6: "
    } else {
        "T5: "
    };
    let named: std::collections::BTreeSet<&'static str> = M
        .properties()
        .iter()
        .map(|p| p.name)
        .filter(|n| n.starts_with(property))
        .collect();
    let checker = M
        .checker()
        .finish_when(stateright::HasDiscoveries::AnyOf(named))
        .spawn_bfs()
        .join();
    let red: Vec<&str> = M
        .properties()
        .iter()
        .filter(|p| p.name.starts_with(property) && checker.discovery(p.name).is_some())
        .map(|p| p.name)
        .collect();
    println!("red: {red:?}");
    assert!(!red.is_empty(), "no property `{property}` went red");
}

/// The keeper calls the kernel for every decision: a source-text check (the loop is blocking I/O).
#[test]
fn the_keeper_calls_the_kernel() {
    let src = include_str!("../src/term.rs");
    let prod = src.split("#[cfg(test)]").next().unwrap();
    for needle in [
        "step(&mut k, ev)",
        "KEv::ChildExit",
        "KEv::MasterEagain",
        "KEv::MasterEio",
        "KEv::Write",
        "KEv::Landed",
        "KEv::Resize",
        "KEv::Conn",
        "KEv::ConnLost",
        "KEv::ParentGone",
        "KEv::SpoolFailed",
        "admit_peer(",
        "keeper_of(",
        "act.kill_shell",
        "exit_status(&k)",
        "act.apply",
        "act.set_size",
        "act.write_status",
        "act.send_exit",
        "act.send_ready",
        "act.exit",
    ] {
        assert!(prod.contains(needle), "the keeper no longer uses {needle}");
    }
    // Nothing of the keeper's state is written outside the kernel.
    for banned in [
        "k.written =",
        "k.eio =",
        "k.drained =",
        "k.status_written =",
        "k.exit_sent =",
        "k.failed =",
    ] {
        assert!(
            !prod.contains(banned),
            "the keeper writes its state outside the kernel: {banned}"
        );
    }
}
