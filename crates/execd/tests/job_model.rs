//! Stateright model of execd's job table: `jobs.rs` over `job_kernel`.
//!
//! It checks every interleaving of two scopes (`A`, `B`) that both use the ids `x` and `y`, up to
//! three jobs, two clients holding a stream, and the environment a job runs in: the shell dying
//! with its group alive or not, a descendant outliving it, `kill` leaving survivors, a pump
//! reading, the ring evicting its oldest byte, the supervisor noticing each change, a held
//! stream's connection dropping at any point.
//!
//! ## Faithfulness
//!
//! Every decision is made by the REAL kernel (`execd::job_kernel`), the functions production calls:
//! `Table::{admit, insert, get, in_scope, take}` under the table's lock, `step` under a job's, and
//! `window` and `follow` for output offsets. The model owns only the environment (processes,
//! the ring's byte counters, the clients, who spawned what) and the shell's effects the kernel's
//! actions name (a `Kill` ends the group, a `Take` removes the entry then marks the job taken).
//! Transitions match the shell one for one: the same event order, the same lookups by key (a
//! stream looks its job up by key on every frame, as `follow` does).
//!
//! Each property is judged against the environment's own facts (which scope spawned which job,
//! whether a process is alive, what the supervisor saw), never by asking the kernel what it should
//! have done.
//!
//! ## Properties
//!
//! - J1 an id is the claim on a job: a start never replaces a job that is in the table.
//!   (execd has no cap: the agent bounds its jobs; see `ARCHITECTURE.md`.)
//! - J2 scope isolation: no operation resolves a job of another scope.
//! - J3 a removed job never answers again: no operation resolves it, a stream attached to it never
//!   reads or kills a later job of the same id, and it owes no callback.
//! - J4 an ended job's code, signal, time and end offset never change (state check here; the
//!   kernel's `step` is checked over every phase and event by `ended_is_final`).
//! - J5 output: a read never starts before the oldest byte held, a stream sends each byte once and
//!   in order, and a gap is exactly what was dropped.
//! - J6 a kept job never dies with a held client's connection, and `Kept` never goes back to `Held`.
//! - J7 no orphan: a job is taken out of the table only with its group empty, so nothing is left
//!   running that the table cannot name.
//! - J10 a take removes only the job that was checked (`remove`/`wipe` await the group between the
//!   check and the take, and the key may hold a successor by then), never its key's successor.
//! - J8 no signal reaches an ended job's group (its number may be reused).
//! - L1 (bounded measure `k = 1`): an ended job can always be removed by `remove`, whatever its
//!   group number now names.
//! - L2 (`k = 1`): a stream that has sent everything of an ended job sends its final frame next.
//!   Fairness assumed: the supervisor and the stream task are always scheduled.
//!
//! Not modelled: bytes (the ring is two counters: its content is the fuzz target `ring`'s), stdin,
//! the notify kernel (`exec-notify`'s own), and time (the kernel reads none: `now` is only
//! recorded).
//!
//! ## Mutations
//!
//! `cargo test -p beyond-execd --features <mut> --test job_model the_mutation` MUST pass only when
//! the named property is red: see `crates/execd/ARCHITECTURE.md` for the feature → property table.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use execd::job_kernel::{self as kernel, Act, Admit, Alive, Ev, Follow, Key, Phase, Pick, State};
use stateright::{Checker, Model, Property};

const V_J1: u32 = 1;
const V_J2: u32 = 2;
const V_J3: u32 = 4;
const V_J5: u32 = 8;
const V_J6: u32 = 16;
const V_J7: u32 = 32;
const V_J8: u32 = 64;
const V_J9: u32 = 128;
const V_J10: u32 = 256;

fn key(scope: u8, id: u8) -> Key {
    (format!("S{scope}"), format!("i{id}"))
}

fn scope_name(scope: u8) -> String {
    format!("S{scope}")
}

/// Where a job's supervisor task is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Sup {
    /// Not spawned yet.
    Idle,
    /// Waiting for the shell.
    Wait,
    /// Said the shell exited; waiting for the group to empty.
    Reported,
    /// The group is empty: the pumps are drained, the end is next.
    Reaping,
    Done,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Sim {
    scope: u8,
    id: u8,
    k: State,
    /// The shell lives.
    shell: bool,
    /// A descendant in the group lives.
    desc: bool,
    /// The pumps run.
    pumps: bool,
    sup: Sup,
    /// The ring holds `[base, end)`.
    base: u8,
    end: u8,
    /// What the job's end was when the supervisor recorded it: `(code, signal, end offset)`.
    snap: Option<(i32, Option<i32>, u8)>,
    was_kept: bool,
}

impl Sim {
    fn alive(&self) -> bool {
        self.shell || self.desc
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Cl {
    Idle,
    /// A held start whose process is not yet spawned.
    Pending,
    Following,
    Done,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Client {
    st: Cl,
    scope: u8,
    id: u8,
    /// The job it was attached to (the model's handle).
    job: usize,
    hold: bool,
    sent: u8,
    told: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct St {
    table: kernel::Table<usize>,
    sims: Vec<Sim>,
    clients: Vec<Client>,
    /// A `remove`/`wipe` that decided (`removable`) and has not yet taken: the shell awaits the
    /// group between the two, and other clients' operations run then.
    pending: Vec<(Key, usize)>,
    bad: u32,
}

impl std::hash::Hash for St {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        let mut all = self.table.all();
        all.sort();
        all.hash(h);
        (&self.sims, &self.clients, &self.pending, self.bad).hash(h);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum A {
    Start {
        scope: u8,
        id: u8,
        client: Option<usize>,
    },
    SpawnOk(usize),
    SpawnFail(usize),
    ShellDies(usize),
    DescDies(usize),
    Notice(usize),
    Reap(usize),
    Push(usize),
    Evict(usize),
    Keep(u8, u8),
    Seen(u8, u8),
    Read(u8, u8),
    Signal(u8, u8),
    Kill {
        scope: u8,
        id: u8,
        remove: bool,
    },
    Wipe {
        scope: u8,
    },
    Remove(u8, u8),
    /// `remove` that decides now and takes later (`TakePending`).
    SlowRemove(u8, u8),
    /// `wipe` that decides now and takes later.
    SlowWipe {
        scope: u8,
    },
    /// The take of the slow operation in flight.
    TakePending,
    Attach {
        client: usize,
        scope: u8,
        id: u8,
    },
    Follow(usize),
    Disconnect(usize),
}

/// The bounds of one run.
#[derive(Clone, Copy)]
struct Jobs {
    /// Scopes that use the same ids.
    scopes: u8,
    ids: u8,
    /// Jobs started in all.
    max_jobs: usize,
    clients: usize,
    /// The ring holds at most this many bytes in all.
    max_end: u8,
    /// `remove` and `wipe` may decide and take as two steps (the shell awaits the group between).
    slow: bool,
}

/// One scope, one id, two jobs: a key is reused after a removal.
const REUSE: Jobs = Jobs {
    scopes: 1,
    ids: 1,
    max_jobs: 2,
    clients: 1,
    max_end: 1,
    slow: false,
};
/// One scope, one id, two jobs, no streams: a `remove`/`wipe` decides, other clients remove and
/// restart the id, then it takes (J10).
const SLOW: Jobs = Jobs {
    scopes: 1,
    ids: 1,
    max_jobs: 2,
    clients: 0,
    max_end: 0,
    slow: true,
};
/// Two scopes that both use one id (no ring: the ring is REUSE's and STREAMS').
const SCOPES: Jobs = Jobs {
    scopes: 2,
    ids: 1,
    max_jobs: 2,
    clients: 1,
    max_end: 0,
    slow: false,
};
/// One scope, two ids: `kill` of one id leaves the other (no streams, no ring).
const IDS: Jobs = Jobs {
    scopes: 1,
    ids: 2,
    max_jobs: 2,
    clients: 0,
    max_end: 0,
    slow: false,
};
/// One job, two clients, two bytes: streams and the ring (a gap, an eviction under a reader).
const STREAMS: Jobs = Jobs {
    scopes: 1,
    ids: 1,
    max_jobs: 1,
    clients: 2,
    max_end: 2,
    slow: false,
};

impl St {
    /// The job a lookup in `scope` resolved to, judged against the facts: J2 and J3.
    fn resolved(&mut self, scope: u8, h: usize) {
        if self.sims[h].scope != scope {
            self.bad |= V_J2;
        }
        if self.sims[h].k.removed {
            self.bad |= V_J3;
        }
    }

    /// The shell's `removable` for job `h`: ask without looking at its processes, then with.
    fn removable(&mut self, h: usize, ended_counts: bool) -> bool {
        let sim = &mut self.sims[h];
        let first = kernel::step(
            &mut sim.k,
            Ev::Remove {
                empty: false,
                ended_counts,
            },
        );
        if first == Act::Take {
            return true;
        }
        let empty = sim.k.pgid() != 0 && !sim.alive();
        kernel::step(
            &mut sim.k,
            Ev::Remove {
                empty,
                ended_counts,
            },
        ) == Act::Take
    }

    /// `Jobs::remove`: out of the table, then each job taken.
    fn take(&mut self, checked: &[(Key, usize)]) {
        let handles = self.table.take(checked, |a, b| a == b);
        for (_, h) in handles {
            // J10: only a job that was checked leaves.
            if !checked.iter().any(|(_, c)| *c == h) {
                self.bad |= V_J10;
            }
            if self.sims[h].alive() {
                self.bad |= V_J7;
            }
            kernel::step(&mut self.sims[h].k, Ev::Taken);
        }
    }

    /// The ids `kill`/`wipe` name in `scope`, as the shell passes them.
    fn stop(&mut self, scope: u8, ids: &[String], all: bool, remove: bool, slow: bool) {
        let pick = if all { Pick::All } else { Pick::Ids(ids) };
        let named = self.table.in_scope(&scope_name(scope), pick);
        let mut stopping = Vec::new();
        for (_, h) in &named {
            self.resolved(scope, *h);
            if kernel::step(&mut self.sims[*h].k, Ev::Stop) == Act::Stop {
                if self.sims[*h].k.ended() {
                    self.bad |= V_J8;
                }
                stopping.push(*h);
            }
        }
        // `stop_jobs`: every process of the group dies (a process that outlives its shell
        // until signalled is the environment's `DescDies`).
        for h in stopping {
            let sim = &mut self.sims[h];
            sim.shell = false;
            sim.desc = false;
        }
        if remove {
            let named = self.table.in_scope(&scope_name(scope), pick);
            let mut gone = Vec::new();
            for (k, h) in named {
                self.resolved(scope, h);
                if self.removable(h, false) {
                    gone.push((k, h));
                }
            }
            self.finish(gone, slow);
        }
    }

    /// The take of what was decided: now, or (slow) as a later step.
    fn finish(&mut self, gone: Vec<(Key, usize)>, slow: bool) {
        if slow {
            self.pending = gone;
        } else {
            self.take(&gone);
        }
    }

    /// The job a client's stream goes on with: the table's lookup by key, through the kernel's
    /// `bound` (the shell's `bound`).
    fn bound(&self, cl: &Client) -> Option<usize> {
        let found = self.table.get(&key(cl.scope, cl.id));
        let same = found == Some(cl.job);
        kernel::bound(found, same)
    }

    /// What a client's stream does next: the job its key names now, and the kernel's frame.
    fn follow_of(&self, c: usize) -> Option<(Option<usize>, Follow)> {
        let cl = &self.clients[c];
        match self.bound(cl) {
            None => Some((None, Follow::Wait)),
            Some(h) => {
                let sim = &self.sims[h];
                let f = kernel::follow(
                    &sim.k,
                    u64::from(cl.sent),
                    u64::from(sim.base),
                    u64::from(sim.end),
                    cl.told,
                );
                if f == Follow::Wait {
                    None
                } else {
                    Some((Some(h), f))
                }
            }
        }
    }
}

impl Model for Jobs {
    type State = St;
    type Action = A;

    fn init_states(&self) -> Vec<St> {
        vec![St {
            table: kernel::Table::default(),
            sims: Vec::new(),
            pending: Vec::new(),
            clients: vec![
                Client {
                    st: Cl::Idle,
                    scope: 0,
                    id: 0,
                    job: 0,
                    hold: false,
                    sent: 0,
                    told: false,
                };
                self.clients
            ],
            bad: 0,
        }]
    }

    fn actions(&self, s: &St, out: &mut Vec<A>) {
        if s.sims.len() < self.max_jobs {
            for scope in 0..self.scopes {
                for id in 0..self.ids {
                    out.push(A::Start {
                        scope,
                        id,
                        client: None,
                    });
                    for c in 0..self.clients {
                        if s.clients[c].st == Cl::Idle {
                            out.push(A::Start {
                                scope,
                                id,
                                client: Some(c),
                            });
                        }
                    }
                }
            }
        }
        for (i, sim) in s.sims.iter().enumerate() {
            // A failed spawn took the entry: nothing spawns after it.
            if sim.k.phase == Phase::Spawning && !sim.k.removed {
                out.push(A::SpawnOk(i));
                out.push(A::SpawnFail(i));
            }
            if sim.shell {
                out.push(A::ShellDies(i));
            }
            if sim.desc {
                out.push(A::DescDies(i));
            }
            match sim.sup {
                Sup::Wait if !sim.shell => out.push(A::Notice(i)),
                Sup::Reported if !sim.desc => out.push(A::Notice(i)),
                Sup::Reaping => out.push(A::Reap(i)),
                _ => {}
            }
            if sim.pumps && sim.end < self.max_end {
                out.push(A::Push(i));
            }
            if sim.base < sim.end {
                out.push(A::Evict(i));
            }
        }
        for scope in 0..self.scopes {
            for id in 0..self.ids {
                out.push(A::Keep(scope, id));
                out.push(A::Seen(scope, id));
                out.push(A::Read(scope, id));
                out.push(A::Signal(scope, id));
                out.push(A::Remove(scope, id));
                if self.slow && s.pending.is_empty() {
                    out.push(A::SlowRemove(scope, id));
                }
                for remove in [false, true] {
                    out.push(A::Kill { scope, id, remove });
                }
                for client in 0..self.clients {
                    if s.clients[client].st == Cl::Idle {
                        out.push(A::Attach { client, scope, id });
                    }
                }
            }
            out.push(A::Wipe { scope });
            if self.slow && s.pending.is_empty() {
                out.push(A::SlowWipe { scope });
            }
        }
        if !s.pending.is_empty() {
            out.push(A::TakePending);
        }
        for c in 0..self.clients {
            match s.clients[c].st {
                Cl::Following => {
                    if s.follow_of(c).is_some() {
                        out.push(A::Follow(c));
                    }
                    out.push(A::Disconnect(c));
                }
                Cl::Pending => out.push(A::Disconnect(c)),
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn next_state(&self, last: &St, a: A) -> Option<St> {
        let mut s = last.clone();
        match a {
            A::Start { scope, id, client } => {
                let k = key(scope, id);
                if s.table.admit(&k) == Admit::Exists {
                    return None;
                }
                // J1: the entry is the claim; a start never replaces a job still in the table.
                if let Some(old) = s.table.get(&k)
                    && !s.sims[old].k.removed
                {
                    s.bad |= V_J1;
                }
                let h = s.sims.len();
                s.sims.push(Sim {
                    scope,
                    id,
                    k: State::new(0, client.is_some()),
                    shell: false,
                    desc: false,
                    pumps: false,
                    sup: Sup::Idle,
                    base: 0,
                    end: 0,
                    snap: None,
                    was_kept: client.is_none(),
                });
                s.table.insert(k, h);
                if let Some(c) = client {
                    s.clients[c] = Client {
                        st: Cl::Pending,
                        scope,
                        id,
                        job: h,
                        hold: true,
                        sent: 0,
                        told: false,
                    };
                }
            }
            A::SpawnOk(i) => {
                let sim = &mut s.sims[i];
                kernel::step(
                    &mut sim.k,
                    Ev::Spawned {
                        pgid: 100 + i as i32,
                    },
                );
                sim.shell = true;
                sim.desc = true;
                sim.pumps = true;
                sim.sup = Sup::Wait;
                for cl in &mut s.clients {
                    if cl.st == Cl::Pending && cl.job == i {
                        cl.st = Cl::Following;
                    }
                }
            }
            A::SpawnFail(i) => {
                let k = key(s.sims[i].scope, s.sims[i].id);
                s.take(&[(k, i)]);
                for cl in &mut s.clients {
                    if cl.st == Cl::Pending && cl.job == i {
                        cl.st = Cl::Idle;
                    }
                }
            }
            A::ShellDies(i) => s.sims[i].shell = false,
            A::DescDies(i) => s.sims[i].desc = false,
            A::Notice(i) => {
                let sim = &mut s.sims[i];
                if sim.desc {
                    kernel::step(
                        &mut sim.k,
                        Ev::ShellExit {
                            code: 10 + i as i32,
                            signal: None,
                        },
                    );
                    sim.sup = Sup::Reported;
                } else {
                    sim.sup = Sup::Reaping;
                }
            }
            A::Reap(i) => {
                let sim = &mut s.sims[i];
                if sim.alive() {
                    return None;
                }
                sim.pumps = false;
                let act = kernel::step(
                    &mut sim.k,
                    Ev::GroupEmpty {
                        code: 10 + i as i32,
                        signal: None,
                        now: 1,
                    },
                );
                sim.sup = Sup::Done;
                sim.snap = Some((10 + i as i32, None, sim.end));
                if act == (Act::Ended { notify: true }) && sim.k.removed {
                    s.bad |= V_J3;
                }
            }
            A::Push(i) => {
                let sim = &mut s.sims[i];
                match kernel::step(&mut sim.k, Ev::Push) {
                    Act::Store => sim.end += 1,
                    _ => sim.pumps = false,
                }
            }
            A::Evict(i) => s.sims[i].base += 1,
            A::Keep(scope, id) => {
                let h = s.table.get(&key(scope, id))?;
                s.resolved(scope, h);
                let sim = &mut s.sims[h];
                kernel::step(
                    &mut sim.k,
                    Ev::Keep {
                        num: None,
                        kind: None,
                        seen: None,
                    },
                );
                sim.was_kept = true;
            }
            A::Seen(scope, id) => {
                let h = s.table.get(&key(scope, id))?;
                s.resolved(scope, h);
                kernel::step(&mut s.sims[h].k, Ev::Seen { offset: 1 });
            }
            A::Read(scope, id) => {
                let h = s.table.get(&key(scope, id))?;
                s.resolved(scope, h);
                let sim = &s.sims[h];
                let (from, want) =
                    kernel::window(0, u64::from(sim.base), u64::from(sim.end), 1 << 20);
                // J5: never before the oldest byte held, and never more than is held after it.
                if from < u64::from(sim.base) || from + want as u64 != u64::from(sim.end) {
                    s.bad |= V_J5;
                }
            }
            A::Signal(scope, id) => {
                let h = s.table.get(&key(scope, id))?;
                s.resolved(scope, h);
                let sim = &mut s.sims[h];
                if kernel::step(&mut sim.k, Ev::Signal) == Act::Signal && sim.k.ended() {
                    s.bad |= V_J8;
                }
            }
            A::Kill { scope, id, remove } => {
                s.stop(scope, &[format!("i{id}")], false, remove, false);
            }
            A::Wipe { scope } => s.stop(scope, &[], true, true, false),
            A::SlowWipe { scope } => s.stop(scope, &[], true, true, true),
            A::TakePending => {
                let checked = std::mem::take(&mut s.pending);
                s.take(&checked);
            }
            A::Remove(scope, id) | A::SlowRemove(scope, id) => {
                let slow = matches!(a, A::SlowRemove(..));
                let ids = [format!("i{id}")];
                let named = s.table.in_scope(&scope_name(scope), Pick::Ids(&ids));
                let mut gone = Vec::new();
                for (k, h) in named {
                    s.resolved(scope, h);
                    if s.removable(h, true) {
                        gone.push((k, h));
                    }
                }
                s.finish(gone, slow);
            }
            A::Attach { client, scope, id } => {
                let h = s.table.get(&key(scope, id))?;
                s.resolved(scope, h);
                s.clients[client] = Client {
                    st: Cl::Following,
                    scope,
                    id,
                    job: h,
                    hold: false,
                    sent: 0,
                    told: false,
                };
            }
            A::Follow(c) => {
                let (h, f) = s.follow_of(c)?;
                let Some(h) = h else {
                    // The job is gone: the final frame ends the stream.
                    s.clients[c].st = Cl::Done;
                    return Some(s);
                };
                let scope = s.clients[c].scope;
                s.resolved(scope, h);
                // J9: a stream reads the job it attached to, never a later job of the same id.
                if h != s.clients[c].job {
                    s.bad |= V_J9;
                    s.clients[c].st = Cl::Done;
                    return Some(s);
                }
                let (base, end) = (s.sims[h].base, s.sims[h].end);
                let cl = &mut s.clients[c];
                match f {
                    Follow::Gap(n) => {
                        // A gap is exactly the bytes dropped before the ring's start.
                        if u64::from(cl.sent) + n != u64::from(base) {
                            s.bad |= V_J5;
                        }
                        cl.sent = base;
                    }
                    Follow::Data => {
                        if cl.sent < base || cl.sent >= end {
                            s.bad |= V_J5;
                        }
                        cl.sent += 1;
                    }
                    Follow::ShellExit(_) => cl.told = true,
                    Follow::End(code, _) => {
                        cl.st = Cl::Done;
                        if s.sims[h].snap.map(|x| x.0) != Some(code) {
                            s.bad |= V_J5;
                        }
                    }
                    Follow::Wait => return None,
                }
            }
            A::Disconnect(c) => {
                let cl = s.clients[c].clone();
                s.clients[c].st = Cl::Done;
                if cl.st != Cl::Following {
                    // Still spawning: the stream task does not exist yet; the held job is
                    // reaped by the spawn's outcome, not by a connection that never ran.
                    return Some(s);
                }
                if cl.hold
                    && let Some(h) = s.bound(&cl)
                {
                    s.resolved(cl.scope, h);
                    if h != cl.job {
                        s.bad |= V_J9;
                    }
                    let kept_before = s.sims[h].k.kept();
                    if let Act::Kill { .. } =
                        kernel::step(&mut s.sims[h].k, Ev::Disconnect { hold: true })
                    {
                        if kept_before {
                            s.bad |= V_J6;
                        }
                        s.sims[h].shell = false;
                        s.sims[h].desc = false;
                    }
                }
            }
        }
        Some(s)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        vec![
            Property::always(
                "J1: a start never replaces a job in the table",
                |_, s: &St| s.bad & V_J1 == 0,
            ),
            Property::always(
                "J2: no operation resolves a job of another scope",
                |_, s: &St| s.bad & V_J2 == 0,
            ),
            Property::always("J3: a removed job never answers again", |_, s: &St| {
                s.bad & V_J3 == 0
            }),
            Property::always(
                "J9: a stream follows the job it attached to, not a later job of its id",
                |_, s: &St| s.bad & V_J9 == 0,
            ),
            Property::always("J4: an ended job's end never changes", |_, s: &St| {
                s.sims.iter().all(|x| match (&x.snap, &x.k.phase) {
                    (
                        Some((code, signal, end)),
                        Phase::Ended {
                            code: c, signal: g, ..
                        },
                    ) => code == c && signal == g && *end == x.end,
                    (Some(_), _) => false,
                    (None, p) => !matches!(p, Phase::Ended { .. }),
                })
            }),
            Property::always(
                "J5: output offsets only move forward over what is held",
                |_, s: &St| s.bad & V_J5 == 0,
            ),
            Property::always(
                "J6: a kept job never dies with a held client",
                |_, s: &St| s.bad & V_J6 == 0 && s.sims.iter().all(|x| !x.was_kept || x.k.kept()),
            ),
            Property::always(
                "J7: a job leaves the table only with its group empty",
                |_, s: &St| s.bad & V_J7 == 0 && s.sims.iter().all(|x| !(x.k.removed && x.alive())),
            ),
            Property::always(
                "J10: a take removes only the job that was checked, not its key's successor",
                |_, s: &St| s.bad & V_J10 == 0,
            ),
            Property::always("J8: no signal reaches an ended job", |_, s: &St| {
                s.bad & V_J8 == 0
            }),
            Property::always("L1: an ended job can always be removed", |_, s: &St| {
                s.table.all().iter().all(|(_, h)| {
                    let mut k = s.sims[*h].k.clone();
                    !k.ended()
                        || kernel::step(
                            &mut k,
                            Ev::Remove {
                                empty: false,
                                ended_counts: true,
                            },
                        ) == Act::Take
                })
            }),
            Property::always(
                "L2: an ended job's stream sends its final frame",
                |_, s: &St| {
                    s.clients.iter().all(|cl| {
                        if cl.st != Cl::Following {
                            return true;
                        }
                        let Some(h) = s.bound(cl) else {
                            return true;
                        };
                        let sim = &s.sims[h];
                        // Everything sent, the shell's exit told (or never owed): the end is next.
                        !(sim.k.ended()
                            && cl.sent >= sim.end
                            && (cl.told || sim.k.shell_code().is_none())
                            && kernel::follow(
                                &sim.k,
                                u64::from(cl.sent),
                                u64::from(sim.base),
                                u64::from(sim.end),
                                cl.told,
                            ) == Follow::Wait)
                    })
                },
            ),
            Property::always(
                "alive: a job counts as alive only with a group",
                |_, s: &St| {
                    s.sims
                        .iter()
                        .all(|x| x.k.alive() == Alive::No || x.k.pgid() != 0)
                },
            ),
        ]
    }
}

fn check(m: Jobs) {
    let checker = m.checker().spawn_bfs().join();
    println!(
        "{} states, {} unique",
        checker.state_count(),
        checker.unique_state_count()
    );
    for p in m.properties() {
        if let Some(path) = checker.discovery(p.name) {
            panic!("`{}` is red:\n{:#?}", p.name, path.into_actions());
        }
    }
}

/// Every property holds over every interleaving: one scope reusing its id after a removal ...
#[test]
fn faithful_reuse() {
    check(REUSE);
}

/// ... and with a `remove`/`wipe` that takes after the others moved.
#[test]
fn faithful_slow_remove() {
    check(SLOW);
}

/// ... and with two scopes.
#[test]
fn faithful_scopes() {
    check(SCOPES);
}

/// ... and with two ids per scope.
#[test]
fn faithful_ids() {
    check(IDS);
}

/// ... and with two clients.
#[test]
fn faithful_streams() {
    check(STREAMS);
}

/// Regression (J9, found by this model in the code before it): a stream looked its job up by
/// `(scope, id)` on every frame, so a stream attached to a removed job went on with a later job
/// of the same id and its disconnect killed that job. It now goes on only with the job it attached
/// to: after a removal and a restart of its id, the REUSE run has no such state.
#[test]
fn a_stream_never_follows_a_successor() {
    let checker = REUSE.checker().spawn_bfs().join();
    let name = "J9: a stream follows the job it attached to, not a later job of its id";
    assert!(checker.discovery(name).is_none(), "J9 is red");
    // The kernel's rule, on its own: a lookup that finds another job finds nothing.
    assert_eq!(
        kernel::bound(Some(2usize), false).filter(|_| !cfg!(feature = "mut_stream_follows_key")),
        None
    );
    assert_eq!(kernel::bound(Some(1usize), true), Some(1));
    assert_eq!(kernel::bound::<usize>(None, false), None);
}

/// J4 for the kernel itself: over every phase and every event, an ended job's code, signal and time
/// do not change, a removed job stays removed, and a kept job stays kept. The model reaches only
/// the events the supervisor sends; this checks the whole function.
#[test]
fn ended_is_final() {
    let ended = Phase::Ended {
        pgid: 5,
        code: 3,
        signal: Some(9),
        at: 7,
        shell_told: true,
    };
    let phases = [
        Phase::Spawning,
        Phase::Running { pgid: 5 },
        Phase::Exited {
            pgid: 5,
            code: 3,
            signal: None,
        },
        ended.clone(),
    ];
    let events = || {
        [
            Ev::Spawned { pgid: 9 },
            Ev::ShellExit {
                code: 1,
                signal: None,
            },
            Ev::GroupEmpty {
                code: 1,
                signal: None,
                now: 99,
            },
            Ev::Keep {
                num: None,
                kind: None,
                seen: None,
            },
            Ev::Seen { offset: 4 },
            Ev::Taken,
            Ev::Push,
            Ev::Signal,
            Ev::Stop,
            Ev::Remove {
                empty: true,
                ended_counts: true,
            },
            Ev::Remove {
                empty: false,
                ended_counts: false,
            },
            Ev::Disconnect { hold: true },
        ]
    };
    for phase in phases {
        for claim in [
            kernel::Claim::Held,
            kernel::Claim::Kept {
                num: None,
                kind: None,
            },
        ] {
            for removed in [false, true] {
                for ev in events() {
                    let mut s = State {
                        phase: phase.clone(),
                        claim: claim.clone(),
                        seen: 0,
                        removed,
                        started: 0,
                    };
                    let before = s.clone();
                    kernel::step(&mut s, ev.clone());
                    if let Phase::Ended { .. } = before.phase {
                        assert_eq!(s.phase, before.phase, "J4: {ev:?} changed an ended job");
                    }
                    assert!(!before.removed || s.removed, "J3: {ev:?} revived a job");
                    assert!(!before.kept() || s.kept(), "J6: {ev:?} un-kept a job");
                }
            }
        }
    }
}

/// With a `mut_*` feature on, the property that MUST be red for it.
#[cfg(any(
    feature = "mut_start_overwrites",
    feature = "mut_scope_ignored",
    feature = "mut_remove_keeps_entry",
    feature = "mut_end_rewritten",
    feature = "mut_read_before_base",
    feature = "mut_disconnect_kills_kept",
    feature = "mut_stream_follows_key",
    feature = "mut_remove_live",
    feature = "mut_signal_ended",
    feature = "mut_ended_not_removable",
    feature = "mut_take_by_key",
    feature = "mut_follow_waits_after_end"
))]
#[test]
fn the_mutation_turns_its_property_red() {
    let property = if cfg!(feature = "mut_start_overwrites") {
        "J1: "
    } else if cfg!(feature = "mut_scope_ignored") {
        "J2: "
    } else if cfg!(feature = "mut_remove_keeps_entry") {
        "J3: "
    } else if cfg!(feature = "mut_end_rewritten") {
        // The model's supervisor sends one end; the kernel check names this one.
        let r = std::panic::catch_unwind(ended_is_final);
        println!("red: [\"J4\"] kernel check failed: {}", r.is_err());
        assert!(r.is_err(), "no property `J4` went red");
        return;
    } else if cfg!(feature = "mut_read_before_base") {
        "J5: "
    } else if cfg!(feature = "mut_stream_follows_key") {
        "J9: "
    } else if cfg!(feature = "mut_disconnect_kills_kept") {
        "J6: "
    } else if cfg!(feature = "mut_remove_live") {
        "J7: "
    } else if cfg!(feature = "mut_signal_ended") {
        "J8: "
    } else if cfg!(feature = "mut_take_by_key") {
        "J10: "
    } else if cfg!(feature = "mut_ended_not_removable") {
        "L1: "
    } else {
        "L2: "
    };
    let mut red: Vec<&str> = Vec::new();
    for m in [SLOW, REUSE, SCOPES, IDS, STREAMS] {
        // A red run stops when the named property fails: the counterexample is the evidence.
        let named: std::collections::BTreeSet<&'static str> = m
            .properties()
            .iter()
            .map(|p| p.name)
            .filter(|n| n.starts_with(property))
            .collect();
        let checker = m
            .checker()
            .finish_when(stateright::HasDiscoveries::AnyOf(named))
            .spawn_bfs()
            .join();
        red.extend(
            m.properties()
                .iter()
                .filter(|p| p.name.starts_with(property) && checker.discovery(p.name).is_some())
                .map(|p| p.name),
        );
        if !red.is_empty() {
            break;
        }
    }
    println!("red: {red:?}");
    assert!(!red.is_empty(), "no property `{property}` went red");
}

/// Production makes the job table's decisions through the kernel: a source-text check (the shell is
/// async code with no other seam). Each call below is one decision, and nothing else writes the
/// kernel's state.
#[test]
fn production_calls_the_kernel() {
    let src = include_str!("../src/jobs.rs");
    let prod = &src[..src.find("#[cfg(test)]\n#[allow(").expect("test module")];
    for needle in [
        "admit(&key)",
        "kernel::State::new(",
        "Ev::Spawned {",
        "Ev::ShellExit {",
        "Ev::GroupEmpty {",
        "Ev::Keep {",
        "Ev::Seen {",
        "Ev::Taken",
        "Ev::Push",
        "Ev::Signal",
        "Ev::Stop",
        "Ev::Remove {",
        "Ev::Disconnect {",
        "kernel::window(",
        "kernel::bound(",
        "kernel::follow(",
        "Pick::",
        ".take(checked, Arc::ptr_eq)",
        "map: Mutex<kernel::Table<JobCell>>",
        ".in_scope(scope, pick)",
        ".alive()",
    ] {
        assert!(prod.contains(needle), "jobs.rs no longer calls {needle}");
    }
    // The table is the kernel's verified storage, not a map of the shell's.
    assert!(!prod.contains("HashMap<Key"), "jobs.rs keeps its own map");
    // No decision of the job's life is made outside the kernel: its state is only written there.
    for banned in [
        ".removed =",
        ".kept =",
        ".seen =",
        ".phase =",
        ".claim =",
        ".end =",
        ".shell_exit =",
        ".pgid =",
        ".signal =",
        ".ended =",
        "removed: bool",
        "kept: bool",
        "map: Mutex<HashMap",
        "map.contains_key",
        "map.remove(",
        "map.insert(",
    ] {
        assert!(
            !prod.contains(banned) || banned == "map.insert(",
            "jobs.rs writes the job table's state outside the kernel: {banned}"
        );
    }
    // The only insert is the kernel table's, after `admit` said fresh, under the same lock.
    let at = prod.find("map.insert(").expect("the table insert");
    assert!(
        prod[..at]
            .rfind("admit(&key)")
            .is_some_and(|a| at - a < 1500),
        "the insert is no longer behind `admit`"
    );
}

/// J10 on the table itself: `remove` checked job 1 under a key, then job 1 was removed and a client
/// started the same id again (job 2). Taking what was checked must leave job 2.
#[test]
fn a_take_removes_the_job_it_checked_not_its_keys_successor() {
    let k = key(0, 0);
    let mut t = kernel::Table::<usize>::default();
    assert_eq!(t.admit(&k), Admit::Fresh);
    t.insert(k.clone(), 1);
    let checked = vec![(k.clone(), 1usize)];
    // Another client removes job 1 and starts job 2 under the key.
    t.take(&checked, |a, b| a == b);
    assert_eq!(t.admit(&k), Admit::Fresh);
    t.insert(k.clone(), 2);
    // The slow take of what was checked.
    t.take(&checked, |a, b| a == b);
    assert_eq!(
        t.get(&k),
        Some(2),
        "J10: the take removed job 2, which was never checked"
    );
    // The job that was checked and is still there goes.
    t.take(&[(k.clone(), 2usize)], |a, b| a == b);
    assert_eq!(t.get(&k), None);
    // The decision on its own.
    assert!(kernel::takes(true));
    assert!(!kernel::takes(false) || cfg!(feature = "mut_take_by_key"));
}
