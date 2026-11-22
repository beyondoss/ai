//! The decisions of execd's job table (`execd::job_kernel`), as pure functions: which `(scope, id)` a start may take,
//! which jobs an operation sees, what state a job is in and what each event does to it, when a
//! signal is sent, when a job may leave the table, when a held job dies with its client, and where
//! an output read or a stream frame starts. The shell is `jobs.rs`.
//!
//! No I/O, no async, no clock (a time arrives in the event), no closure or trait object in a
//! decision, so a model checker calls exactly what production calls (`tests/job_model.rs`) and a
//! proof takes it as it is: every decision below is written inside `verus!`, so the function production runs is the one [Verus](https://github.com/verus-lang/verus) checks (see the README's Proven section). The table's storage is inside it too: a `Vec` of `(key, job)` that Verus checks (lookup, insert, take, scope), so no `HashMap` stands behind a proven decision. Outside it are only the accessors that reshape a field and the table's `Clone`, `Debug` and `Eq`. Production keeps two locks and the kernel follows them:
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
    clippy::single_match,
    clippy::collapsible_if
)]

use vstd::prelude::*;

verus! {

// ---------------------------------------------------------------------------------------------
// The table: its decisions, as functions of what the storage observed. The storage is below.
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

/// J10: a take goes through only for the job that was checked: the table holds it under the key
/// still (`same`). Between `removable` (which awaits the group) and the take, the job may have been
/// removed and its id started again: the successor was never checked, and taking it would leave
/// its process group running with no entry.
pub fn takes(same: bool) -> (r: bool)
    ensures
        r == same,
{
    same || cfg!(feature = "mut_take_by_key")
}

// ---------------------------------------------------------------------------------------------
// The OS boundary: process groups execd still holds.
// ---------------------------------------------------------------------------------------------

/// The process groups execd holds: spawned by it, and not yet seen gone by the reaper, each as its
/// number and the spawn's `gen` (unique per spawn). A group's number is free for any other process
/// to take the moment its last member is collected, and a later spawn of execd's own may take it:
/// so a signal by number is the job's only while the number is held *by that spawn*. The reaper takes
/// a number out (`freed`) in the same exclusion in which it collects the last member, and a signal
/// checks (`may_deliver`) inside the exclusion that keeps the reaper from collecting: between the
/// check and the `kill` the group cannot be freed, a number the reaper has freed is never signalled,
/// and neither is a successor spawn's group that took the same number.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Held {
    live: Vec<(i32, u64)>,
}

impl Held {
    /// J11: spawn `sp` holds group `g`.
    pub closed spec fn holds(&self, g: i32, sp: u64) -> bool {
        self.live@.contains((g, sp))
    }

    pub fn new() -> (h: Held)
        ensures
            forall|g: i32, sp: u64| !h.holds(g, sp),
    {
        Held { live: Vec::new() }
    }

    /// Whether spawn `sp` holds `g`.
    pub fn has(&self, g: i32, sp: u64) -> (r: bool)
        ensures
            r == self.holds(g, sp),
    {
        let mut i: usize = 0;
        while i < self.live.len()
            invariant
                0 <= i <= self.live@.len(),
                forall|j: int| 0 <= j < i ==> self.live@[j] != (g, sp),
            decreases self.live@.len() - i,
        {
            let e = self.live[i];
            if e.0 == g && e.1 == sp {
                return true;
            }
            i += 1;
        }
        false
    }

    /// A spawn registers its group: held from then on, and nothing else changes. Idempotent.
    pub fn insert(&mut self, g: i32, sp: u64)
        ensures
            final(self).holds(g, sp),
            forall|h: i32, t: u64| (h, t) != (g, sp) ==> final(self).holds(h, t) == old(self).holds(h, t),
    {
        broadcast use vstd::seq_lib::lemma_seq_contains_after_push;
        if !self.has(g, sp) && !cfg!(feature = "mut_held_forgets") {
            self.live.push((g, sp));
        }
    }

    /// The reaper saw number `g` gone: no spawn holds it, and nothing else changes. Idempotent.
    pub fn freed(&mut self, g: i32)
        ensures
            forall|t: u64| !final(self).holds(g, t),
            forall|h: i32, t: u64| h != g ==> final(self).holds(h, t) == old(self).holds(h, t),
    {
        let mut out: Vec<(i32, u64)> = Vec::new();
        let mut i: usize = 0;
        while i < self.live.len()
            invariant
                0 <= i <= self.live@.len(),
                forall|h: (i32, u64)| out@.contains(h) <==> (h.0 != g && exists|j: int| 0 <= j < i && self.live@[j] == h),
            decreases self.live@.len() - i,
        {
            let v = self.live[i];
            let ghost before = out@;
            if v.0 != g {
                out.push(v);
            }
            i += 1;
            proof {
                assert(self.live@[(i - 1) as int] == v);
                assert forall|h: (i32, u64)| out@.contains(h) implies (h.0 != g && exists|j: int| 0 <= j < i && self.live@[j] == h) by {
                    if h == v {
                        assert(self.live@[(i - 1) as int] == h);
                    } else {
                        assert(before.contains(h));
                    }
                }
                assert forall|h: (i32, u64)| (h.0 != g && exists|j: int| 0 <= j < i && self.live@[j] == h) implies out@.contains(h) by {
                    let j = choose|j: int| 0 <= j < i && self.live@[j] == h;
                    if j == i - 1 {
                        assert(v == h);
                        assert(v.0 != g);
                        assert(out@ =~= before.push(v));
                        assert(out@[out@.len() - 1] == h);
                        assert(out@.contains(h));
                    } else {
                        assert(0 <= j < i - 1 && self.live@[j] == h);
                        assert(exists|j2: int| 0 <= j2 < i - 1 && self.live@[j2] == h);
                        assert(before.contains(h));
                        if v.0 != g {
                            assert(out@ =~= before.push(v));
                            vstd::seq_lib::lemma_seq_contains_after_push(before, v, h);
                            assert(out@.contains(h) == (v == h || before.contains(h)));
                        } else {
                            assert(out@ =~= before);
                        }
                        assert(out@.contains(h));
                    }
                }
            }
        }
        self.live = out;
    }

    /// The numbers held, for the reaper to look each up.
    pub fn numbers(&self) -> (r: Vec<i32>)
        ensures
            forall|g: i32, t: u64| self.holds(g, t) ==> r@.contains(g),
    {
        let mut r: Vec<i32> = Vec::new();
        let mut i: usize = 0;
        while i < self.live.len()
            invariant
                0 <= i <= self.live@.len(),
                forall|g: i32| r@.contains(g) <==> exists|j: int| 0 <= j < i && (self.live@[j]).0 == g,
            decreases self.live@.len() - i,
        {
            let v = self.live[i];
            r.push(v.0);
            i += 1;
            proof {
                broadcast use vstd::seq_lib::lemma_seq_contains_after_push;
                assert(self.live@[(i - 1) as int] == v);
            }
        }
        proof {
            assert forall|g: i32, t: u64| self.holds(g, t) implies r@.contains(g) by {
                let j = choose|j: int| 0 <= j < self.live@.len() && self.live@[j] == (g, t);
                assert((self.live@[j]).0 == g);
            }
        }
        r
    }
}

/// J11: a signal by number is delivered only to the group of the spawn that holds the number: never
/// to one the reaper has freed, whose number any other process may now carry, and never to a later
/// spawn's group that took the same number. (Liveness, the completeness half: a held group is
/// always deliverable, so a refusal clears only when the group is gone and there is nothing left
/// to signal.)
pub fn may_deliver(held: &Held, pgid: i32, sp: u64) -> (r: bool)
    ensures
        r == held.holds(pgid, sp),
{
    held.has(pgid, sp) || cfg!(feature = "mut_signal_after_reap")
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


// ---------------------------------------------------------------------------------------------
// The table's storage: a `Vec` of `(key, job)` with the keys distinct. Verus knows `String`'s
// equality and clone (`vstd/string.rs`), so lookup, insert and take are proven here, with no
// `HashMap` (whose key model vstd axiomatizes only for integers and bools:
// `vstd/std_specs/hash.rs` 13-18, 113-215).
// ---------------------------------------------------------------------------------------------

/// `(scope, id)`: a job's address.
pub type Key = (String, String);

pub open spec fn key_eq(a: Key, b: Key) -> bool {
    a.0@ == b.0@ && a.1@ == b.1@
}

/// Which ids of a scope an operation names.
#[derive(Clone, Copy, Debug)]
pub enum Pick<'a> {
    /// Every job of the scope (`list`, `wipe`).
    All,
    /// The jobs whose id is one of these (`signal`, `kill`, `remove`).
    Ids(&'a [String]),
}

/// Is `id` one of `ids`?
pub open spec fn ids_name(ids: Seq<String>, id: Seq<char>) -> bool {
    exists|i: int| 0 <= i < ids.len() && (#[trigger] ids[i])@ == id
}

impl<'a> Pick<'a> {
    /// Is `id` one of the ids this names?
    pub open spec fn names(self, id: Seq<char>) -> bool {
        match self {
            Pick::All => true,
            Pick::Ids(ids) => ids_name(ids@, id),
        }
    }

    fn has(self, id: &str) -> (r: bool)
        ensures
            r == self.names(id@),
    {
        match self {
            Pick::All => true,
            Pick::Ids(ids) => {
                let mut n = 0;
                while n < ids.len()
                    invariant
                        self == Pick::Ids(ids),
                        n <= ids@.len(),
                        forall|i: int| 0 <= i < n ==> (#[trigger] ids@[i])@ != id@,
                    decreases ids@.len() - n,
                {
                    if ids[n].as_str() == id {
                        assert(ids@[n as int]@ == id@);
                        assert(ids_name(ids@, id@));
                        return true;
                    }
                    n += 1;
                }
                false
            },
        }
    }
}

/// Every job by `(scope, id)`. `C` is what the shell keeps per job (a shared cell in production, a
/// handle in the model): the kernel never looks inside it. The keys are distinct (the type
/// invariant), so a key names at most one job.
pub struct Table<C> {
    entries: Vec<(Key, C)>,
}

impl<C> Table<C> {
    /// The keys are distinct. `new` makes a table that satisfies it and every method that changes
    /// the table keeps it; the field is private, so no other table exists.
    pub closed spec fn wf(self) -> bool {
        forall|i: int, j: int|
            0 <= i < j < self.entries@.len() ==> !key_eq(
                (#[trigger] self.entries@[i]).0,
                (#[trigger] self.entries@[j]).0,
            )
    }

    /// Some job is under `k`.
    pub closed spec fn holds(self, k: Key) -> bool {
        exists|i: int| 0 <= i < self.entries@.len() && key_eq((#[trigger] self.entries@[i]).0, k)
    }

    /// The job under `k` is `v`.
    pub closed spec fn at(self, k: Key, v: C) -> bool {
        exists|i: int|
            0 <= i < self.entries@.len() && key_eq((#[trigger] self.entries@[i]).0, k)
                && self.entries@[i].1 == v
    }

    /// The entries, in order.
    pub closed spec fn rows(self) -> Seq<(Key, C)> {
        self.entries@
    }

    pub fn new() -> (r: Self)
        ensures
            r.wf(),
            forall|k: Key| !r.holds(k),
    {
        Table { entries: Vec::new() }
    }

    /// Where `key` is: the first (and only) entry whose scope and id both equal. (`scope_matters`
    /// is false only under the `mut_scope_ignored` mutation, which looks an id up in any scope.)
    fn locate(&self, key: &Key, scope_matters: bool) -> (r: Option<usize>)
        ensures
            scope_matters ==> (r is Some ==> r.unwrap() < self.entries@.len() && key_eq(
                self.entries@[r.unwrap() as int].0,
                *key,
            )),
            scope_matters ==> (r is None ==> !self.holds(*key)),
            r is Some ==> r.unwrap() < self.entries@.len(),
    {
        let mut n = 0;
        while n < self.entries.len()
            invariant
                n <= self.entries@.len(),
                scope_matters ==> forall|i: int|
                    0 <= i < n ==> !key_eq((#[trigger] self.entries@[i]).0, *key),
            decreases self.entries@.len() - n,
        {
            let e = &self.entries[n];
            if (!scope_matters || e.0.0 == key.0) && e.0.1 == key.1 {
                return Some(n);
            }
            n += 1;
        }
        None
    }

    fn position(&self, key: &Key) -> (r: Option<usize>)
        ensures
            r is Some ==> r.unwrap() < self.entries@.len() && key_eq(
                self.entries@[r.unwrap() as int].0,
                *key,
            ),
            r is None ==> !self.holds(*key),
    {
        self.locate(key, true)
    }

    /// J1's observation: is the key in the table.
    pub fn is_held(&self, key: &Key) -> (r: bool)
        ensures
            r == self.holds(*key),
    {
        self.position(key).is_some()
    }

    pub fn admit(&self, key: &Key) -> (r: Admit)
        ensures
            self.holds(*key) ==> r is Exists,
            !self.holds(*key) ==> r is Fresh,
    {
        admit(self.is_held(key))
    }

    /// Put a job in its key: only after [`Table::admit`] said `Fresh`, under the same lock. An
    /// occupied key is replaced (the old job is dropped), never duplicated.
    pub fn insert(&mut self, key: Key, job: C)
        requires
            old(self).wf(),
        ensures
            final(self).wf(),
            final(self).at(key, job),
            forall|k: Key| !key_eq(k, key) ==> (final(self).holds(k) == old(self).holds(k)),
    {
        let ghost before = self.entries@;
        match self.position(&key) {
            Some(i) => {
                let ghost old_key = before[i as int].0;
                let _ = self.entries.remove(i);
                self.entries.insert(i, (key, job));
                proof {
                    assert(self.entries@ =~= before.update(i as int, (key, job)));
                    assert(key_eq(old_key, key));
                    assert(self.entries@[i as int] == (key, job));
                    assert(self.at(key, job));
                    assert forall|k: Key| !key_eq(k, key) implies (self.holds(k) == old(self).holds(k)) by {
                        if self.holds(k) {
                            let j = choose|j: int| 0 <= j < self.entries@.len() && key_eq(self.entries@[j].0, k);
                            assert(j != i as int);
                            assert(before[j] == self.entries@[j]);
                        }
                        if old(self).holds(k) {
                            let j = choose|j: int| 0 <= j < before.len() && key_eq(before[j].0, k);
                            assert(j != i as int);
                            assert(self.entries@[j] == before[j]);
                        }
                    }
                    assert forall|a: int, b: int| 0 <= a < b < self.entries@.len() implies !key_eq(
                        (#[trigger] self.entries@[a]).0,
                        (#[trigger] self.entries@[b]).0,
                    ) by {
                        if a == i as int {
                            assert(before[b] == self.entries@[b]);
                            assert(!key_eq(before[a].0, before[b].0));
                        } else if b == i as int {
                            assert(before[a] == self.entries@[a]);
                        } else {
                            assert(before[a] == self.entries@[a]);
                            assert(before[b] == self.entries@[b]);
                        }
                    }
                }
            },
            None => {
                self.entries.push((key, job));
                proof {
                    assert(self.entries@ =~= before.push((key, job)));
                    assert(self.entries@[before.len() as int] == (key, job));
                    assert(self.at(key, job));
                    assert forall|k: Key| !key_eq(k, key) implies (self.holds(k) == old(self).holds(k)) by {
                        if self.holds(k) {
                            let j = choose|j: int| 0 <= j < self.entries@.len() && key_eq(self.entries@[j].0, k);
                            assert(j < before.len());
                            assert(before[j] == self.entries@[j]);
                        }
                        if old(self).holds(k) {
                            let j = choose|j: int| 0 <= j < before.len() && key_eq(before[j].0, k);
                            assert(self.entries@[j] == before[j]);
                        }
                    }
                    assert forall|a: int, b: int| 0 <= a < b < self.entries@.len() implies !key_eq(
                        (#[trigger] self.entries@[a]).0,
                        (#[trigger] self.entries@[b]).0,
                    ) by {
                        if b == before.len() as int {
                            assert(before[a] == self.entries@[a]);
                            assert(!old(self).holds(key));
                        } else {
                            assert(before[a] == self.entries@[a]);
                            assert(before[b] == self.entries@[b]);
                        }
                    }
                }
            },
        }
    }

    /// J2 (exact key): the job at `key` and no other (`None` for a free key, and for a key whose
    /// scope or id differs).
    pub fn get(&self, key: &Key) -> (r: Option<C>)
        where
            C: Clone,
        ensures
            r is None <==> !self.holds(*key),
            r is Some ==> exists|v: C| self.at(*key, v) && cloned::<C>(v, r.unwrap()),
    {
        match self.locate(key, !cfg!(feature = "mut_scope_ignored")) {
            Some(i) => {
                let c = self.entries[i].1.clone();
                proof {
                    assert(self.at(*key, self.entries@[i as int].1));
                }
                Some(c)
            },
            None => None,
        }
    }

    /// J2: the jobs of `scope` that `pick` names, and no other scope's: each result is an entry of
    /// the table, in the scope and named; each such entry is a result.
    pub fn in_scope(&self, scope: &str, pick: Pick<'_>) -> (r: Vec<(Key, C)>)
        where
            C: Clone,
        ensures
            forall|j: int|
                0 <= j < r@.len() ==> {
                    &&& (#[trigger] r@[j]).0.0@ == scope@
                    &&& pick.names(r@[j].0.1@)
                    &&& exists|v: C| self.at(r@[j].0, v) && cloned::<C>(v, r@[j].1)
                },
            forall|i: int|
                0 <= i < self.rows().len() && (#[trigger] self.rows()[i]).0.0@ == scope@
                    && pick.names(self.rows()[i].0.1@) ==> exists|j: int|
                    0 <= j < r@.len() && key_eq((#[trigger] r@[j]).0, self.rows()[i].0),
    {
        let mut out: Vec<(Key, C)> = Vec::new();
        let mut n = 0;
        while n < self.entries.len()
            invariant
                n <= self.entries@.len(),
                forall|j: int|
                    0 <= j < out@.len() ==> {
                        &&& (#[trigger] out@[j]).0.0@ == scope@
                        &&& pick.names(out@[j].0.1@)
                        &&& exists|v: C| self.at(out@[j].0, v) && cloned::<C>(v, out@[j].1)
                    },
                forall|i: int|
                    0 <= i < n && (#[trigger] self.entries@[i]).0.0@ == scope@ && pick.names(
                        self.entries@[i].0.1@,
                    ) ==> exists|j: int|
                        0 <= j < out@.len() && key_eq((#[trigger] out@[j]).0, self.entries@[i].0),
            decreases self.entries@.len() - n,
        {
            let e = &self.entries[n];
            let ghost old_out = out@;
            if visible(e.0.0.as_str() == scope, pick.has(e.0.1.as_str())) {
                let k = (e.0.0.clone(), e.0.1.clone());
                let c = e.1.clone();
                out.push((k, c));
                proof {
                    let v = self.entries@[n as int].1;
                    assert(self.at(out@[old_out.len() as int].0, v));
                    assert(key_eq(out@[old_out.len() as int].0, self.entries@[n as int].0));
                    assert(cloned::<C>(v, out@[old_out.len() as int].1));
                    assert(out@[old_out.len() as int].0.0@ == scope@);
                    assert(pick.names(out@[old_out.len() as int].0.1@));
                    assert forall|j: int| 0 <= j < out@.len() implies {
                        &&& (#[trigger] out@[j]).0.0@ == scope@
                        &&& pick.names(out@[j].0.1@)
                        &&& exists|v: C| self.at(out@[j].0, v) && cloned::<C>(v, out@[j].1)
                    } by {
                        if j < old_out.len() {
                            assert(out@[j] == old_out[j]);
                        }
                    }
                    assert forall|i: int|
                        0 <= i < n + 1 && (#[trigger] self.entries@[i]).0.0@ == scope@ && pick.names(
                            self.entries@[i].0.1@,
                        ) implies exists|j: int|
                        0 <= j < out@.len() && key_eq((#[trigger] out@[j]).0, self.entries@[i].0) by {
                        if i < n as int {
                            let j = choose|j: int| 0 <= j < old_out.len() && key_eq(old_out[j].0, self.entries@[i].0);
                            assert(out@[j] == old_out[j]);
                        } else {
                            assert(key_eq(out@[old_out.len() as int].0, self.entries@[i].0));
                        }
                    }
                }
            }
            n += 1;
        }
        out
    }

    /// J3 and J10: take the `checked` jobs out of the table; the jobs that were taken, each with
    /// its key. An entry is taken only if the job under its key is still the one that was checked
    /// (`same`: the shell's pointer identity), so a key that is gone, or now holds a successor, is
    /// skipped (J10). A job that was taken is gone: its key resolves to nothing afterwards (J3).
    pub fn take(&mut self, checked: &[(Key, C)], same: impl Fn(&C, &C) -> bool) -> (taken: Vec<(Key, C)>)
        where
            C: Clone,
        requires
            old(self).wf(),
            forall|a: &C, b: &C| same.requires((a, b)),
        ensures
            final(self).wf(),
            forall|k: Key| final(self).holds(k) ==> old(self).holds(k),
            forall|j: int|
                0 <= j < taken@.len() ==> {
                    &&& !final(self).holds((#[trigger] taken@[j]).0)
                    &&& exists|n: int, v: C|
                        0 <= n < checked@.len() && key_eq(checked@[n].0, taken@[j].0)
                            && old(self).at(taken@[j].0, v) && cloned::<C>(v, taken@[j].1)
                            && same.ensures((&v, &checked@[n].1), true)
                },
    {
        let ghost start = *self;
        let mut taken: Vec<(Key, C)> = Vec::new();
        let mut n = 0;
        while n < checked.len()
            invariant
                n <= checked@.len(),
                self.wf(),
                forall|a: &C, b: &C| same.requires((a, b)),
                forall|k: Key| self.holds(k) ==> start.holds(k),
                forall|k: Key, v: C| self.at(k, v) ==> start.at(k, v),
                forall|j: int|
                    0 <= j < taken@.len() ==> {
                        &&& !self.holds((#[trigger] taken@[j]).0)
                        &&& exists|m: int, v: C|
                            0 <= m < n && key_eq(checked@[m].0, taken@[j].0) && start.at(
                                taken@[j].0,
                                v,
                            ) && cloned::<C>(v, taken@[j].1) && same.ensures(
                                (&v, &checked@[m].1),
                                true,
                            )
                    },
            decreases checked.len() - n,
        {
            let ghost before = *self;
            let (k, c) = &checked[n];
            if let Some(i) = self.position(k) {
                if takes(same(&self.entries[i].1, c)) {
                    if cfg!(feature = "mut_remove_keeps_entry") {
                        taken.push(((k.0.clone(), k.1.clone()), self.entries[i].1.clone()));
                    } else {
                        let (key, job) = self.entries.remove(i);
                        taken.push((key, job));
                    }
                }
            }
            n += 1;
        }
        taken
    }

    /// Every job in the table, any scope.
    pub fn all(&self) -> (r: Vec<(Key, C)>)
        where
            C: Clone,
        ensures
            r@.len() == self.rows().len(),
    {
        let mut out = Vec::new();
        let mut n = 0;
        while n < self.entries.len()
            invariant
                n <= self.entries@.len(),
                out@.len() == n,
            decreases self.entries@.len() - n,
        {
            let e = &self.entries[n];
            out.push(((e.0.0.clone(), e.0.1.clone()), e.1.clone()));
            n += 1;
        }
        out
    }
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

impl<C> Default for Table<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Clone> Clone for Table<C> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
        }
    }
}

impl<C: std::fmt::Debug> std::fmt::Debug for Table<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(k, c)| (k, c)))
            .finish()
    }
}

/// Two tables are equal when they hold the same jobs under the same keys, whatever order they came in.
impl<C: PartialEq> PartialEq for Table<C> {
    fn eq(&self, other: &Self) -> bool {
        self.entries.len() == other.entries.len()
            && self
                .entries
                .iter()
                .all(|(k, c)| other.entries.iter().any(|(k2, c2)| k == k2 && c == c2))
    }
}

impl<C: Eq> Eq for Table<C> {}
