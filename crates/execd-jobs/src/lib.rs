//! The decisions of execd's job table (`execd::job_kernel`), as pure functions: which `(scope, id)` a start may take,
//! which jobs an operation sees, what state a job is in and what each event does to it, when a
//! signal is sent, when a job may leave the table, when a held job dies with its client, and where
//! an output read or a stream frame starts. The shell is `jobs.rs`.
//!
//! No I/O, no async, no clock (a time arrives in the event), no closure or trait object in a
//! decision, so a model checker calls exactly what production calls (`tests/job_model.rs`) and a
//! proof takes it as it is: every decision below is written inside `verus!`, so the function production runs is the one [Verus](https://github.com/verus-lang/verus) checks (see the README's Proven section). What stays outside it is the table's storage and string comparisons (the `HashMap`, `Pick::has`, the accessors that only reshape a field). Production keeps two locks and the kernel follows them:
//!
//! - **the table's** (`Jobs::map`): [`Table`], the `(scope, id)` → job map: [`Table::admit`],
//!   [`Table::get`], [`Table::in_scope`], [`Table::insert`], [`Table::take`];
//! - **one job's** (`Job`'s mutex): [`State`], one job's state, changed only by [`step`], which
//!   returns the [`Act`] the shell performs under the same lock.
//!
//! Output is bytes in a ring owned by the shell; the kernel owns the offset arithmetic that must
//! hold whatever the ring does: [`window`] (where a read starts) and [`follow`] (what a stream
//! sends next). What the job's processes do, and what a pump reads, is the shell's: it passes the
//! facts it observed in the event (`empty`).
//!
//! | event | decision |
//! |---|---|
//! | [`Table::admit`] | a key that is in the table is `Exists` (409); a free one is `Fresh` |
//! | [`Ev::Spawned`] | `Spawning` → `Running` |
//! | [`Ev::ShellExit`] | `Running` → `Exited` (the shell is gone, its group lives); any other phase: nothing |
//! | [`Ev::GroupEmpty`] | `Running` or `Exited` → `Ended` once, with the code, signal and time; stdin closes; a callback is owed only if the job is not removed; an ended job never changes |
//! | [`Ev::Keep`] / [`Ev::Seen`] | `Held` → `Kept` (never back); `seen` is what the model has been shown |
//! | [`Ev::Taken`] | the table let the job go: `removed`, final; the shell drops its notifier and stdin |
//! | [`Ev::Push`] | a pump's bytes are stored unless the job is removed |
//! | [`Ev::Signal`] | a group is signalled only while the job is running and has a group (a number of an ended job may be reused) |
//! | [`Ev::Stop`] | `kill` stops what has not ended, a job still spawning included |
//! | [`Ev::Remove`] | a job leaves only once its group is observed empty, or (for `remove`) once it has ended |
//! | [`bound`] | a stream goes on only with the job it attached to |
//! | [`Ev::Disconnect`] | a held job nobody kept, still running, dies with its client's connection; a kept one never does |
//!
//! There is no cap and no expiry: the agent bounds its jobs, and an ended job is freed only by an
//! explicit `remove`, `kill` with `remove` or `wipe` (see `ARCHITECTURE.md`). Mutations (cargo
//! features `mut_*`, used only by the model's red runs) each remove one guard.

// The code inside `verus!` takes the shapes Verus can prove (a `match` where `matches!` reads better, an
// explicit subtraction guard, a mutation's `cfg!` branch beside its twin); clippy's rewrites are not those.
#![allow(
    clippy::match_like_matches_macro,
    clippy::if_same_then_else,
    clippy::implicit_saturating_sub,
    clippy::single_match
)]

use std::collections::HashMap;

use vstd::prelude::*;

verus! {

// ---------------------------------------------------------------------------------------------
// The table: its decisions, as functions of what the storage observed. The `HashMap` and the string
// comparisons are outside `verus!` (below); each decision they feed is here.
// ---------------------------------------------------------------------------------------------

/// Whether a start may take its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    Fresh,
    /// The key is in the table: the id is the claim, so two starts of one id cannot both run.
    Exists,
}

/// J1: a key that is in the table (`present`) is `Exists`; a free one is `Fresh`. [`Table::admit`]
/// asks the storage whether the key is there and feeds the answer here.
pub fn admit(present: bool) -> (r: Admit)
    ensures
        present ==> r is Exists,
        !present ==> r is Fresh,
{
    if present && !cfg!(feature = "mut_start_overwrites") {
        Admit::Exists
    } else {
        Admit::Fresh
    }
}

/// J2: an entry is seen by an operation only if it is in the operation's scope (`same_scope`) and the
/// operation names its id (`named`). [`Table::in_scope`] feeds each entry's two facts here.
pub fn visible(same_scope: bool, named: bool) -> (r: bool)
    ensures
        r ==> same_scope,
        r == (same_scope && named),
{
    (same_scope || cfg!(feature = "mut_scope_ignored")) && named
}

// ---------------------------------------------------------------------------------------------
// One job's state.
// ---------------------------------------------------------------------------------------------

/// Where a job is in its life. One way only: `Spawning → Running → Exited → Ended` (`Exited` is
/// skipped when the group dies with the shell). The number of the group exists from `Running`.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// The entry holds its id; the process is not yet spawned.
    Spawning,
    Running {
        pgid: i32,
    },
    /// The shell is gone and its group lives on. `signal` is what killed the shell, if one did.
    Exited {
        pgid: i32,
        code: i32,
        signal: Option<i32>,
    },
    /// The group is empty. The code, signal and time are final.
    Ended {
        pgid: i32,
        code: i32,
        signal: Option<i32>,
        at: u64,
        /// The shell's exit was reported first (the group outlived it): a stream says so.
        shell_told: bool,
    },
}

/// Who owns the job's life: a held start's connection, or the endpoint.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Claim {
    /// A held start no one has kept: it dies with its connection.
    Held,
    /// `keep` (or a start that was not held) made it the endpoint's, for good.
    Kept {
        num: Option<u64>,
        kind: Option<String>,
    },
}

/// One job's state. Changed only by [`step`] (and built by [`State::new`]).
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub phase: Phase,
    pub claim: Claim,
    /// How much output the model has been shown.
    pub seen: u64,
    /// Taken out of the table: its pumps and timers stop. Final.
    pub removed: bool,
    /// When it started (the shell's epoch seconds).
    pub started: u64,
}

/// The state a job is read in: `read` and `list` name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Named {
    Running,
    Exited,
    Ended,
}

/// How a job counts in the host's view of what is alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alive {
    No,
    Yes,
    /// Alive and the endpoint's.
    Kept,
}

/// The group's number: 0 while spawning.
pub open spec fn pgid_s(s: State) -> i32 {
    match s.phase {
        Phase::Spawning => 0,
        Phase::Running { pgid } => pgid,
        Phase::Exited { pgid, .. } => pgid,
        Phase::Ended { pgid, .. } => pgid,
    }
}

/// The code once the shell has exited, ended or not.
pub open spec fn shell_code_s(s: State) -> Option<i32> {
    match s.phase {
        Phase::Ended { code, shell_told, .. } => if shell_told { Some(code) } else { None },
        Phase::Exited { code, .. } => Some(code),
        Phase::Spawning => None,
        Phase::Running { .. } => None,
    }
}

impl State {
    /// A job just entered the table: spawning, kept unless its start is held (and so `background`).
    pub fn new(started: u64, hold: bool) -> (s: State)
        ensures
            s.phase is Spawning,
            hold <==> s.claim is Held,
            s.seen == 0,
            !s.removed,
            s.started == started,
    {
        State {
            phase: Phase::Spawning,
            claim: if hold {
                Claim::Held
            } else {
                Claim::Kept { num: None, kind: Some("background".to_string()) }
            },
            seen: 0,
            removed: false,
            started,
        }
    }

    /// The group's number: 0 while spawning.
    pub fn pgid(&self) -> (r: i32)
        ensures
            r == pgid_s(*self),
    {
        match &self.phase {
            Phase::Spawning => 0,
            Phase::Running { pgid } => *pgid,
            Phase::Exited { pgid, .. } => *pgid,
            Phase::Ended { pgid, .. } => *pgid,
        }
    }

    pub fn ended(&self) -> (r: bool)
        ensures
            r == (self.phase is Ended),
    {
        match &self.phase {
            Phase::Ended { .. } => true,
            _ => false,
        }
    }

    pub fn kept(&self) -> (r: bool)
        ensures
            r == (self.claim is Kept),
    {
        match &self.claim {
            Claim::Kept { .. } => true,
            Claim::Held => false,
        }
    }

    /// `(state, code)` as `read` and `list` answer it: the code once there is one.
    pub fn named(&self) -> (r: (Named, Option<i32>))
        ensures
            (self.phase is Ended) == (r.0 is Ended),
            (self.phase is Exited) == (r.0 is Exited),
            r.1 is Some <==> (self.phase is Ended || self.phase is Exited),
    {
        match &self.phase {
            Phase::Ended { code, .. } => (Named::Ended, Some(*code)),
            Phase::Exited { code, .. } => (Named::Exited, Some(*code)),
            Phase::Spawning => (Named::Running, None),
            Phase::Running { .. } => (Named::Running, None),
        }
    }

    /// The signal that killed the shell, once the job has a code.
    pub fn signal(&self) -> Option<i32> {
        match &self.phase {
            Phase::Ended { signal, .. } => *signal,
            Phase::Exited { signal, .. } => *signal,
            Phase::Spawning => None,
            Phase::Running { .. } => None,
        }
    }

    pub fn num(&self) -> Option<u64> {
        match &self.claim {
            Claim::Kept { num, .. } => *num,
            Claim::Held => None,
        }
    }

    /// The exit code of an ended job.
    pub fn end_code(&self) -> Option<i32> {
        match &self.phase {
            Phase::Ended { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// The code once the shell has exited, ended or not.
    pub fn shell_code(&self) -> (r: Option<i32>)
        ensures
            r == shell_code_s(*self),
    {
        match &self.phase {
            Phase::Ended { code, shell_told, .. } => if *shell_told { Some(*code) } else { None },
            Phase::Exited { code, .. } => Some(*code),
            Phase::Spawning => None,
            Phase::Running { .. } => None,
        }
    }

    /// When it ended (the shell's epoch seconds), 0 until then.
    pub fn ended_at(&self) -> u64 {
        match &self.phase {
            Phase::Ended { at, .. } => *at,
            _ => 0,
        }
    }

    /// Whether the job counts as alive (a job with no group yet does not).
    pub fn alive(&self) -> (r: Alive)
        ensures
            (self.phase is Ended || self.removed || pgid_s(*self) == 0) ==> r is No,
            !(self.phase is Ended || self.removed || pgid_s(*self) == 0) ==> (r is Kept <==> self.claim is Kept),
            r is Yes ==> (!(self.phase is Ended) && !self.removed && pgid_s(*self) != 0 && self.claim is Held),
    {
        if self.ended() || self.removed || self.pgid() == 0 {
            Alive::No
        } else if self.kept() {
            Alive::Kept
        } else {
            Alive::Yes
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Events and `step`.
// ---------------------------------------------------------------------------------------------

/// What happened to a job.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug)]
pub enum Ev {
    /// The process is spawned; its group is `pgid`.
    Spawned { pgid: i32 },
    /// The shell exited and its group is not empty.
    ShellExit { code: i32, signal: Option<i32> },
    /// The group is empty and the pumps have drained: the end is observable. `now` is the shell's
    /// epoch seconds.
    GroupEmpty {
        code: i32,
        signal: Option<i32>,
        now: u64,
    },
    /// `keep`: the job is the endpoint's. `seen` replaces the offset when given.
    Keep {
        num: Option<u64>,
        kind: Option<String>,
        seen: Option<u64>,
    },
    /// `seen`: the model has been shown output up to `offset`.
    Seen { offset: u64 },
    /// The table let the job go.
    Taken,
    /// A pump has bytes for the ring.
    Push,
    /// `signal` names the job.
    Signal,
    /// `kill` or `wipe` names the job.
    Stop,
    /// `remove` (`ended_counts`) or `kill`/`wipe` with remove names the job. `empty`: its group was
    /// observed empty.
    Remove { empty: bool, ended_counts: bool },
    /// The connection of a held start (`hold`) or an attach ended before the job did.
    Disconnect { hold: bool },
}

/// What the shell does, under the job's lock.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Act {
    Nothing,
    /// Put the pump's bytes in the ring.
    Store,
    /// The job is removed: the pump stops.
    Discard,
    /// Signal the job's group (and marked strays) with the signal asked for.
    Signal,
    /// Stop the job's processes (`SIGTERM`, then `SIGKILL` after the grace).
    Stop,
    /// `SIGKILL` the job's group: its client left and nobody kept it.
    Kill {
        pgid: i32,
    },
    /// The job may leave the table.
    Take,
    /// Close the job's stdin; a callback is owed for the change unless the job is removed.
    Ended {
        notify: bool,
    },
    /// The shell's own exit is observable: a callback is owed.
    Exited,
    /// Drop the job's notifier, close its stdin.
    Release,
    /// `keep` recorded: feed notify's kernel.
    Kept,
}

/// How far a job is on its one way: `Spawning` 0, `Running` 1, `Exited` 2, `Ended` 3.
pub open spec fn phase_rank(p: Phase) -> int {
    match p {
        Phase::Spawning => 0,
        Phase::Running { .. } => 1,
        Phase::Exited { .. } => 2,
        Phase::Ended { .. } => 3,
    }
}

/// `remove` may take the job: it has ended and the operation counts an ended job, or it has a group
/// observed empty (J7, L1).
pub open spec fn removable(old: State, ev: Ev) -> bool {
    ev is Remove && ((ev->Remove_ended_counts && old.phase is Ended)
        || (pgid_s(old) != 0 && ev->Remove_empty))
}

/// What `step` promises about the job's life: J4 (an ended job never changes), one way only, final
/// `removed`, `Kept` never back to `Held` (J6), and each phase change is its own event's.
pub open spec fn post_life(old: State, ev: Ev, s: State) -> bool {
    // J4: ended is final (code, signal, time and shell_told included).
    &&& (old.phase is Ended ==> s.phase == old.phase)
    &&& phase_rank(old.phase) <= phase_rank(s.phase)
    &&& (s.phase != old.phase ==> (ev is Spawned || ev is ShellExit || ev is GroupEmpty))
    &&& (old.phase is Spawning && ev is Spawned ==> s.phase == (Phase::Running { pgid: ev->Spawned_pgid }))
    &&& (!(old.phase is Spawning) && ev is Spawned ==> s.phase == old.phase)
    &&& (old.phase is Running && ev is ShellExit ==> s.phase == (Phase::Exited {
        pgid: pgid_s(old),
        code: ev->ShellExit_code,
        signal: ev->ShellExit_signal,
    }))
    &&& (!(old.phase is Running) && ev is ShellExit ==> s.phase == old.phase)
    &&& ((old.phase is Running || old.phase is Exited) && ev is GroupEmpty ==> s.phase == (Phase::Ended {
        pgid: pgid_s(old),
        code: ev->GroupEmpty_code,
        signal: ev->GroupEmpty_signal,
        at: ev->GroupEmpty_now,
        shell_told: old.phase is Exited,
    }))
    &&& ((old.phase is Spawning) && ev is GroupEmpty ==> s.phase == old.phase)
    // removed is final, and only `Taken` sets it.
    &&& (old.removed ==> s.removed)
    &&& (s.removed != old.removed ==> ev is Taken)
    &&& (ev is Taken ==> s.removed)
    // `Kept` never goes back; only `Keep` changes the claim.
    &&& (old.claim is Kept ==> s.claim is Kept)
    &&& (s.claim != old.claim ==> ev is Keep)
    &&& (ev is Keep ==> s.claim == (Claim::Kept { num: ev->Keep_num, kind: ev->Keep_kind }))
    // The offset moves only by `Keep` and `Seen`; the start time never changes.
    &&& (ev is Seen ==> s.seen == ev->Seen_offset)
    &&& (s.seen != old.seen ==> (ev is Keep || ev is Seen))
    &&& (ev is Keep ==> s.seen == (match ev->Keep_seen { Some(o) => o, None => old.seen }))
    &&& s.started == old.started
}

/// What `step` tells the shell to do: J3 (a removed job's pump is discarded and its end owes no
/// callback), J6 (a kept job never dies with a client), J7 (a job leaves only with its group
/// observed empty), J8 (no signal to an ended job's group), and the completeness halves L1 (an
/// ended job can be removed) that say the refusal clears.
pub open spec fn post_act(old: State, ev: Ev, act: Act) -> bool {
    // J3: nothing is stored for a removed job, and its end owes no callback.
    &&& (ev is Push ==> ((act is Discard) <==> old.removed))
    &&& (ev is Push ==> ((act is Store) <==> !old.removed))
    &&& (ev is Taken ==> act is Release)
    &&& ((old.phase is Running || old.phase is Exited) && ev is GroupEmpty
        ==> act == (Act::Ended { notify: !old.removed }))
    &&& ((old.phase is Spawning || old.phase is Ended) && ev is GroupEmpty ==> act is Nothing)
    &&& (old.phase is Running && ev is ShellExit ==> act is Exited)
    &&& (!(old.phase is Running) && ev is ShellExit ==> act is Nothing)
    &&& (ev is Keep ==> act is Kept)
    // J6: only a held start's connection, with nobody having kept the job, and the job still
    // running, kills; a kept job never dies with a client.
    &&& (act is Kill ==> (ev is Disconnect && ev->Disconnect_hold && old.claim is Held
        && !(old.phase is Ended) && act->Kill_pgid == pgid_s(old)))
    &&& (ev is Disconnect ==> ((act is Kill) <==> (ev->Disconnect_hold && old.claim is Held
        && !(old.phase is Ended))))
    &&& (old.claim is Kept ==> !(act is Kill))
    // J7 and L1: `Take` exactly when the job may leave; an ended job always may (when `remove`
    // counts it), whatever its group's number names now.
    &&& (act is Take ==> removable(old, ev))
    &&& (ev is Remove ==> ((act is Take) <==> removable(old, ev)))
    &&& (ev is Remove && ev->Remove_ended_counts && old.phase is Ended ==> act is Take)
    // J8: a group is signalled only while the job has one and has not ended.
    &&& (act is Signal ==> (ev is Signal && !(old.phase is Ended) && pgid_s(old) != 0))
    &&& (ev is Signal ==> ((act is Signal) <==> (!(old.phase is Ended) && pgid_s(old) != 0)))
    &&& (ev is Stop ==> ((act is Stop) <==> !(old.phase is Ended)))
    &&& (act is Stop ==> ev is Stop)
}

/// Apply `ev` to `s` and say what to do. Total: an event that does not fit the phase changes
/// nothing. Time arrives in the event.
pub fn step(s: &mut State, ev: Ev) -> (act: Act)
    ensures
        post_life(*old(s), ev, *final(s)),
        post_act(*old(s), ev, act),
{
    match ev {
        Ev::Spawned { pgid } => {
            if let Phase::Spawning = s.phase {
                s.phase = Phase::Running { pgid };
            }
            Act::Nothing
        },
        Ev::ShellExit { code, signal } => {
            if let Phase::Running { pgid } = s.phase {
                s.phase = Phase::Exited { pgid, code, signal };
                Act::Exited
            } else {
                Act::Nothing
            }
        },
        Ev::GroupEmpty { code, signal, now } => {
            let (pgid, shell_told) = match s.phase {
                Phase::Running { pgid } => (pgid, false),
                Phase::Exited { pgid, .. } => (pgid, true),
                Phase::Ended { pgid, shell_told, .. } => {
                    if !cfg!(feature = "mut_end_rewritten") {
                        return Act::Nothing;
                    }
                    (pgid, shell_told)
                },
                Phase::Spawning => return Act::Nothing,
            };
            s.phase = Phase::Ended { pgid, code, signal, at: now, shell_told };
            Act::Ended { notify: !s.removed }
        },
        Ev::Keep { num, kind, seen } => {
            s.claim = Claim::Kept { num, kind };
            s.seen = match seen {
                Some(o) => o,
                None => s.seen,
            };
            Act::Kept
        },
        Ev::Seen { offset } => {
            s.seen = offset;
            Act::Nothing
        },
        Ev::Taken => {
            s.removed = true;
            Act::Release
        },
        Ev::Push => {
            if s.removed {
                Act::Discard
            } else {
                Act::Store
            }
        },
        Ev::Signal => {
            let running = !s.ended() && s.pgid() != 0;
            if running || cfg!(feature = "mut_signal_ended") && s.pgid() != 0 {
                Act::Signal
            } else {
                Act::Nothing
            }
        },
        Ev::Stop => {
            if s.ended() {
                Act::Nothing
            } else {
                Act::Stop
            }
        },
        Ev::Remove { empty, ended_counts } => {
            let ended = ended_counts && s.ended() && !cfg!(feature = "mut_ended_not_removable");
            if ended || (s.pgid() != 0 && empty) || cfg!(feature = "mut_remove_live") {
                Act::Take
            } else {
                Act::Nothing
            }
        },
        Ev::Disconnect { hold } => {
            let kept = s.kept() && !cfg!(feature = "mut_disconnect_kills_kept");
            if hold && !kept && !s.ended() {
                Act::Kill { pgid: s.pgid() }
            } else {
                Act::Nothing
            }
        },
    }
}

/// The ranking function of the end's progress: the fair steps a job still needs to end.
/// `Spawning` needs the spawn and the group's emptying (2), `Running` and `Exited` the emptying
/// (1), `Ended` none (0).
pub open spec fn end_rank(p: Phase) -> int {
    match p {
        Phase::Spawning => 2,
        Phase::Running { .. } => 1,
        Phase::Exited { .. } => 1,
        Phase::Ended { .. } => 0,
    }
}

/// The fair events of a job's life, the ones its own process and supervisor deliver: the spawn
/// returning, and the group being observed empty. (F1: the supervisor always delivers them: a group
/// empties and the pumps drain, which the environment does and the kernel cannot force.)
pub open spec fn fair_end(old: State, ev: Ev) -> bool {
    (old.phase is Spawning && ev is Spawned)
    || ((old.phase is Running || old.phase is Exited) && ev is GroupEmpty)
}

/// L1, the end half: every fair step lowers `end_rank` (strictly, until zero), so a job ends in at
/// most two fair steps, and zero is exactly `Ended`; once ended, `post_act` says `remove` takes it
/// whatever the group's number names, and J4 says it stays ended until removed.
pub proof fn lemma_end_progress(old: State, ev: Ev, s: State, act: Act)
    requires
        post_life(old, ev, s),
        post_act(old, ev, act),
        fair_end(old, ev),
    ensures
        end_rank(s.phase) < end_rank(old.phase),
        end_rank(old.phase) <= 2,
        end_rank(s.phase) == 0 <==> s.phase is Ended,
{
}

// ---------------------------------------------------------------------------------------------
// Output offsets.
// ---------------------------------------------------------------------------------------------

/// Where a read from `offset` starts, for a ring that holds `[base, end)`, and how many bytes it
/// may answer of at most `max`: from the oldest byte held when `offset` is older (J5).
pub fn window(offset: u64, base: u64, end: u64, max: usize) -> (r: (u64, usize))
    ensures
        r.0 >= base,
        r.0 >= offset,
        r.0 == (if offset >= base { offset } else { base }),
        r.1 <= max,
        r.1 as int == (if end > r.0 { if (end - r.0) as int > max as int { max as int } else { (end - r.0) as int } } else { 0int }),
        r.0 <= end ==> r.0 as int + r.1 as int <= end as int,
        (r.0 < end && max > 0) ==> r.1 > 0,
{
    let from = if cfg!(feature = "mut_read_before_base") {
        offset
    } else if offset > base {
        offset
    } else {
        base
    };
    let avail: u64 = if end > from { end - from } else { 0 };
    let want: usize = if avail > max as u64 { max } else { avail as usize };
    (from, want)
}

/// The job a stream may go on with: what the table holds under its key, if that is the job it
/// attached to (`same`). A removed job is gone to its streams, and a later job of the same id is
/// not its successor: the stream never reads it, and its client leaving never kills it (J9).
pub fn bound<C>(found: Option<C>, same: bool) -> (r: Option<C>)
    ensures
        same ==> r == found,
        !same ==> r is None,
{
    if same || cfg!(feature = "mut_stream_follows_key") {
        found
    } else {
        None
    }
}

/// What a stream sends next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Follow {
    /// `n` bytes were dropped before the ring's start: say so, and `sent` moves past them.
    Gap(u64),
    /// Bytes are held at `sent`.
    Data,
    /// The shell exited and its group lives: say so once.
    ShellExit(i32),
    /// The job ended and everything was sent: the final frame.
    End(i32, Option<i32>),
    /// Nothing yet.
    Wait,
}

/// A stream with everything sent (`sent` reached the ring's end and its start) of an ended job:
/// its next frame is the shell's exit (once) or the final frame (L2).
pub open spec fn post_follow(s: State, sent: u64, base: u64, end: u64, shell_told: bool, f: Follow) -> bool {
    &&& (sent < base ==> f == (Follow::Gap((base - sent) as u64)))
    &&& (f is Gap ==> sent < base)
    &&& (f is Data <==> (base <= sent && sent < end))
    &&& (f is ShellExit ==> (base <= sent && sent >= end && !shell_told
        && shell_code_s(s) == Some(f->ShellExit_0)))
    &&& (f is End ==> (base <= sent && sent >= end && s.phase is Ended
        && (shell_told || shell_code_s(s) is None)
        && f == (Follow::End(s.phase->Ended_code, s.phase->Ended_signal))))
    &&& (f is Wait ==> (base <= sent && sent >= end && !(s.phase is Ended)))
    // L2: with everything sent, an ended job's stream is told the shell's exit if it was not,
    // and then, or at once if there was none, the end.
    &&& (base <= sent && sent >= end && s.phase is Ended ==> (f is ShellExit || f is End))
    &&& (base <= sent && sent >= end && s.phase is Ended && (shell_told || shell_code_s(s) is None)
        ==> f == (Follow::End(s.phase->Ended_code, s.phase->Ended_signal)))
}

/// The next frame of a stream that has sent up to `sent`, for a job whose ring holds `[base, end)`.
pub fn follow(s: &State, sent: u64, base: u64, end: u64, shell_told: bool) -> (f: Follow)
    ensures
        post_follow(*s, sent, base, end, shell_told, f),
{
    if sent < base {
        Follow::Gap(base - sent)
    } else if sent < end {
        Follow::Data
    } else {
        match s.shell_code() {
            Some(c) => {
                if !shell_told {
                    return Follow::ShellExit(c);
                }
            },
            None => {},
        }
        match &s.phase {
            Phase::Ended { code, signal, .. } => {
                if cfg!(feature = "mut_follow_waits_after_end") {
                    Follow::Wait
                } else {
                    Follow::End(*code, *signal)
                }
            },
            _ => Follow::Wait,
        }
    }
}

/// What the shell does with a frame it sent: `sent` and `shell_told` after. (The shell's, and what
/// L2's fairness F2 assumes: a `Data` frame sends at least one byte, never past `end`.)
pub open spec fn applied(f: Follow, sent: u64, base: u64, end: u64, told: bool, sent2: u64, told2: bool) -> bool {
    match f {
        Follow::Gap(_) => sent2 == base && told2 == told,
        Follow::Data => sent < sent2 && sent2 <= end && told2 == told,
        Follow::ShellExit(_) => sent2 == sent && told2,
        Follow::End(..) => sent2 == sent && told2 == told,
        Follow::Wait => sent2 == sent && told2 == told,
    }
}

/// The ranking function of L2: the frames a stream still owes an ended job: the gap, each byte
/// (at worst one frame each), the shell's exit if it was not told, and the end.
pub open spec fn follow_rank(s: State, sent: u64, base: u64, end: u64, told: bool) -> int {
    (if sent < base { 1int } else { 0int })
    + (if end > (if sent > base { sent } else { base }) {
        end as int - (if sent > base { sent } else { base }) as int
    } else {
        0int
    })
    + (if shell_code_s(s) is Some && !told { 1int } else { 0int })
    + 1
}

/// L2, progress: for an ended job whose ring is well formed (`base <= end`), every frame but the
/// last lowers `follow_rank`, and the last (`End`) is sent at rank 1. So a stream reaches its final
/// frame in at most `follow_rank` frames, under F1 (the supervisor made the job end), F2 (each
/// `Data` frame the shell sends takes at least one byte, `window` promises it can) and F3 (the
/// stream task is always scheduled). A frame is never `Wait` for an ended job.
pub proof fn lemma_follow_progress(
    s: State,
    sent: u64,
    base: u64,
    end: u64,
    told: bool,
    f: Follow,
    sent2: u64,
    told2: bool,
)
    requires
        s.phase is Ended,
        base <= end,
        post_follow(s, sent, base, end, told, f),
        applied(f, sent, base, end, told, sent2, told2),
    ensures
        follow_rank(s, sent, base, end, told) >= 1,
        !(f is Wait),
        f is End ==> follow_rank(s, sent, base, end, told) == 1,
        !(f is End) ==> follow_rank(s, sent2, base, end, told2) < follow_rank(s, sent, base, end, told),
{
}

} // verus!

impl Named {
    pub fn as_str(self) -> &'static str {
        match self {
            Named::Running => "running",
            Named::Exited => "exited",
            Named::Ended => "ended",
        }
    }
}

impl State {
    pub fn kind(&self) -> Option<&str> {
        match &self.claim {
            Claim::Kept { kind, .. } => kind.as_deref(),
            Claim::Held => None,
        }
    }
}

/// `(scope, id)`: a job's address.
pub type Key = (String, String);

/// Which ids of a scope an operation names.
#[derive(Clone, Copy, Debug)]
pub enum Pick<'a> {
    /// Every job of the scope (`list`, `wipe`).
    All,
    /// The jobs whose id is one of these (`signal`, `kill`, `remove`).
    Ids(&'a [String]),
}

impl Pick<'_> {
    fn has(&self, id: &str) -> bool {
        match self {
            Pick::All => true,
            Pick::Ids(ids) => ids.iter().any(|i| i == id),
        }
    }
}

/// Every job by `(scope, id)`: the storage, which is not verified (a `HashMap` of string keys; see the
/// README). What it observes (is the key there, is the entry in the scope, is its id named) is
/// handed to [`admit`] and [`visible`], where the decision is proven. `C` is what the shell keeps per
/// job (a shared cell in production, a handle in the model): the kernel never looks inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table<C> {
    entries: HashMap<Key, C>,
}

impl<C> Default for Table<C> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<C: Clone> Table<C> {
    pub fn admit(&self, key: &Key) -> Admit {
        admit(self.entries.contains_key(key))
    }

    /// Put a job in its key: only after [`Table::admit`] said `Fresh`, under the same lock.
    pub fn insert(&mut self, key: Key, job: C) {
        self.entries.insert(key, job);
    }

    /// The job at `key`, in its scope only.
    pub fn get(&self, key: &Key) -> Option<C> {
        if cfg!(feature = "mut_scope_ignored") {
            return self
                .entries
                .iter()
                .find(|((_, id), _)| *id == key.1)
                .map(|(_, c)| c.clone());
        }
        self.entries.get(key).cloned()
    }

    /// The jobs of `scope` that `pick` names, and no other scope's.
    pub fn in_scope(&self, scope: &str, pick: Pick<'_>) -> Vec<(Key, C)> {
        self.entries
            .iter()
            .filter(|((sc, id), _)| visible(sc == scope, pick.has(id)))
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect()
    }

    /// Take `keys` out of the table; the jobs that were there. A key that is gone is skipped.
    pub fn take(&mut self, keys: &[Key]) -> Vec<C> {
        if cfg!(feature = "mut_remove_keeps_entry") {
            return keys
                .iter()
                .filter_map(|k| self.entries.get(k).cloned())
                .collect();
        }
        keys.iter().filter_map(|k| self.entries.remove(k)).collect()
    }

    /// Every job in the table, any scope.
    pub fn all(&self) -> Vec<(Key, C)> {
        self.entries
            .iter()
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect()
    }
}
