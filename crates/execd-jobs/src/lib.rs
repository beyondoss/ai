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
//! hold whatever the ring does: `execd_ring::window` (where a read starts; the ring kernel's, R1) and [`follow`] (what a stream
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
    clippy::collapsible_if,
    non_shorthand_field_patterns
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

/// Why a start is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The id is a job in the table, and the start's key is not the one that job was started under (or the start has
    /// no key): the id is the claim (J1). `409`, "the job exists".
    IdExists,
    /// The key was used for a different command: it names a job whose spec digest is not this start's. `409`,
    /// "the key was used for a different command".
    KeyReused,
}

/// What a start does.
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Begin {
    /// Run the job: take the id (and the key, if the start has one).
    Start,
    /// The key was already used for this very command: answer that job (its id), run nothing.
    Replay(String),
    Refuse(Refusal),
}

/// K1-K5: what a start does. `id_held` is whether the table has a job under the start's `(scope, id)`; `by_key` is
/// the ledger's record for the start's key in its scope, when the start has a key and the record names a job still in
/// the table: that job's id, and whether its stored spec digest is this start's. A start with no key, or whose key
/// the ledger has not seen, has `None`. [`Table::begin`] feeds both from the table and the ledger.
///
/// - K1: a key already used for this command is a replay of that job; it never starts a second one.
/// - K2: a key already used for another command is refused (`KeyReused`); it neither replays nor starts.
/// - K3 (the completeness half of K2's refusal and of J1's): a key the ledger has not seen, with a free id, always
///   starts; a key already used for this command always replays. A refusal clears: the same start with a fresh
///   key (or after the id's job is removed) is admitted.
/// - K4: a start with no key never replays and never answers `KeyReused`.
/// - K5: an id in the table with no key that names it is `IdExists`: a start under another key never takes a
///   running job's id.
pub fn start_decision(id_held: bool, by_key: Option<(String, bool)>) -> (r: Begin)
    ensures
        (by_key is Some && by_key->Some_0.1) ==> (r is Replay && r->Replay_0@ == by_key->Some_0.0@),
        (by_key is Some && !by_key->Some_0.1) ==> r == Begin::Refuse(Refusal::KeyReused),
        (by_key is None && id_held) ==> r == Begin::Refuse(Refusal::IdExists),
        (by_key is None && !id_held) ==> r is Start,
{
    match by_key {
        Some((id, same)) => {
            if same {
                if cfg!(feature = "mut_key_replay_starts") {
                    Begin::Start
                } else {
                    Begin::Replay(id)
                }
            } else if cfg!(feature = "mut_key_digest_ignored") {
                Begin::Replay(id)
            } else {
                Begin::Refuse(Refusal::KeyReused)
            }
        },
        None => {
            if id_held {
                Begin::Refuse(Refusal::IdExists)
            } else if cfg!(feature = "mut_key_fresh_refused") {
                Begin::Refuse(Refusal::IdExists)
            } else {
                Begin::Start
            }
        },
    }
}

/// K6: a restarted execd keeps the start key's record of every job it brings back (running, or as it ended): a
/// retry of a start whose answer was lost across the restart replays, never starts a second job. A job `recover`
/// drops (it never had a group, or was removed) is not in the table, so its record goes with it.
pub fn keeps_record(how: Recovery) -> (r: bool)
    ensures
        r == !(how is Drop),
{
    match how {
        Recovery::Drop => false,
        Recovery::Adopt => true,
        Recovery::Restore => !cfg!(feature = "mut_key_forgotten_on_restart"),
    }
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
// A restart of execd: what each job in the saved table becomes.
// ---------------------------------------------------------------------------------------------

/// What a restarted execd does with one saved job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recovery {
    /// The job is not in the table: it never had a group, or it was removed.
    Drop,
    /// The job is in the table with its group still running; a supervisor watches the group.
    Adopt,
    /// The job is in the table as it ended, with the status the kernel returned.
    Restore,
}

/// The exit code of a job whose shell's status no one saw: the shell was reparented when execd
/// died, and a process that is not our child gives no status. The same `-1` a lost status is
/// everywhere else in execd (`Proc::wait`).
pub const LOST_CODE: i32 = -1;

/// R1-R5: what a saved job (`s`, as [`step`] left it and the ledger wrote it) becomes when execd
/// starts again. `alive` is what the OS says: some process of the job's group, or carrying its
/// mark, is still running. `now` is the shell's epoch seconds.
///
/// - R1: an ended job comes back as it ended: its code, signal and time are exactly what was saved,
///   whatever the OS says now (the group number of an ended job may belong to another process).
/// - R2: a job whose group is alive is adopted as saved: its phase and its shell's saved status stay.
/// - R3: a job whose group is gone ends at once, with its shell's saved status, else [`LOST_CODE`]:
///   never a status no one saw.
/// - R4: a job that never had a group, or was removed, is not in the table.
/// - R5 ([`lemma_recover_idempotent`]): recovering what recovery returned changes nothing (a
///   restart during a restart).
pub fn recover(s: State, alive: bool, now: u64) -> (r: (State, Recovery))
    ensures
        s.phase is Ended ==> (r.1 is Restore && r.0 == s),
        (!s.removed && (s.phase is Running || s.phase is Exited) && alive) ==> (r.1 is Adopt && r.0 == s),
        (!s.removed && s.phase is Running && !alive) ==> (r.1 is Restore && r.0 == State {
            phase: Phase::Ended { pgid: pgid_s(s), code: -1i32, signal: None, at: now, shell_told: false },
            ..s
        }),
        (!s.removed && s.phase is Exited && !alive) ==> (r.1 is Restore && r.0 == State {
            phase: Phase::Ended {
                pgid: pgid_s(s),
                code: s.phase->Exited_code,
                signal: s.phase->Exited_signal,
                at: now,
                shell_told: true,
            },
            ..s
        }),
        (s.phase is Spawning || (s.removed && !(s.phase is Ended))) ==> r.1 is Drop,
        r.1 is Drop || r.0.claim == s.claim,
        r.1 is Drop || r.0.seen == s.seen,
        r.1 is Drop || r.0.started == s.started,
        r.1 is Drop || pgid_s(r.0) == pgid_s(s),
        r.1 is Adopt ==> (r.0.phase is Running || r.0.phase is Exited),
        r.1 is Restore ==> r.0.phase is Ended,
{
    let State { phase, claim, seen, removed, started } = s;
    match phase {
        Phase::Spawning => (State { phase: Phase::Spawning, claim, seen, removed, started }, Recovery::Drop),
        Phase::Ended { pgid, code, signal, at, shell_told } => (
            State { phase: Phase::Ended { pgid, code, signal, at, shell_told }, claim, seen, removed, started },
            if cfg!(feature = "mut_recover_reends_ended") && !alive {
                Recovery::Drop
            } else {
                Recovery::Restore
            },
        ),
        Phase::Running { pgid } => {
            if removed {
                (State { phase: Phase::Running { pgid }, claim, seen, removed, started }, Recovery::Drop)
            } else if (alive && !cfg!(feature = "mut_recover_drops_live")) || cfg!(feature = "mut_recover_adopts_gone") {
                (State { phase: Phase::Running { pgid }, claim, seen, removed, started }, Recovery::Adopt)
            } else {
                (
                    State {
                        phase: Phase::Ended { pgid, code: -1i32, signal: None, at: now, shell_told: false },
                        claim,
                        seen,
                        removed,
                        started,
                    },
                    Recovery::Restore,
                )
            }
        },
        Phase::Exited { pgid, code, signal } => {
            if removed {
                (State { phase: Phase::Exited { pgid, code, signal }, claim, seen, removed, started }, Recovery::Drop)
            } else if (alive && !cfg!(feature = "mut_recover_drops_live")) || cfg!(feature = "mut_recover_adopts_gone") {
                (State { phase: Phase::Exited { pgid, code, signal }, claim, seen, removed, started }, Recovery::Adopt)
            } else {
                let kept = !cfg!(feature = "mut_recover_forgets_shell_status");
                (
                    State {
                        phase: Phase::Ended {
                            pgid,
                            code: if kept { code } else { -1 },
                            signal: if kept { signal } else { None },
                            at: now,
                            shell_told: true,
                        },
                        claim,
                        seen,
                        removed,
                        started,
                    },
                    Recovery::Restore,
                )
            }
        },
    }
}

/// [`recover`]'s answer as a spec, for [`lemma_recover_idempotent`].
pub open spec fn recover_spec(s: State, alive: bool, now: u64) -> (State, Recovery) {
    match s.phase {
        Phase::Spawning => (s, Recovery::Drop),
        Phase::Ended { .. } => (s, Recovery::Restore),
        Phase::Running { pgid } => {
            if s.removed {
                (s, Recovery::Drop)
            } else if alive {
                (s, Recovery::Adopt)
            } else {
                (State { phase: Phase::Ended { pgid, code: -1i32, signal: None, at: now, shell_told: false }, ..s }, Recovery::Restore)
            }
        },
        Phase::Exited { pgid, code, signal } => {
            if s.removed {
                (s, Recovery::Drop)
            } else if alive {
                (s, Recovery::Adopt)
            } else {
                (State { phase: Phase::Ended { pgid, code, signal, at: now, shell_told: true }, ..s }, Recovery::Restore)
            }
        },
    }
}

/// R5: what [`recover`] put in the table as ended recovers to itself, whatever the OS says the second
/// time and whenever it is asked: a restart during a restart changes nothing.
pub proof fn lemma_recover_idempotent(s: State, alive: bool, now: u64, alive2: bool, now2: u64)
    requires
        recover_spec(s, alive, now).1 is Restore,
    ensures
        recover_spec(recover_spec(s, alive, now).0, alive2, now2) == recover_spec(s, alive, now),
{
}

// ---------------------------------------------------------------------------------------------
// Where output goes, and what a shutdown does to each group.
// ---------------------------------------------------------------------------------------------

/// What a process group belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// A one-shot or streamed command: its client's connection is its life.
    Command,
    /// A job: it outlives the request that started it, and a restart of execd.
    Job,
}

/// Where a process's standard output and error go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sink {
    /// A pipe execd reads: it dies with execd's end of it, and a write after that is `SIGPIPE`.
    Pipe,
    /// An `O_APPEND` file execd opened and the process inherits: it outlives execd, and execd
    /// reads it from an offset.
    Spool,
}

/// S1, S2: where a group's output goes. A job's is the spool file, never a pipe (a pipe's reader is
/// execd, and a job that writes after execd restarted would die of `SIGPIPE`); a command's is a
/// pipe, which ends with its client.
pub fn output_sink(kind: Kind) -> (s: Sink)
    ensures
        kind is Job ==> s is Spool,
        kind is Command ==> s is Pipe,
{
    match kind {
        Kind::Job => if cfg!(feature = "mut_job_output_pipe") { Sink::Pipe } else { Sink::Spool },
        Kind::Command => if cfg!(feature = "mut_command_output_spool") { Sink::Spool } else { Sink::Pipe },
    }
}

/// Where a job's spool files live. Both places are under the jobs directory (a disk the operator
/// chose); there is no variant for the ledger's state directory (tmpfs in a sandbox, where 8 MiB per
/// job would be RAM), so a spool cannot be put there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Root {
    /// The directory that outlives this execd, which a ledger's spools need: a restart finds a spool
    /// there by the path the ledger names.
    Stable,
    /// This execd's own directory, which goes with it.
    PerServer,
}

/// S7: where a job's spool lives. With a ledger, the stable directory (a restart finds the spool by
/// its recorded path); without one, this server's own (nothing could find a spool later).
pub fn spool_root(ledger: bool) -> (r: Root)
    ensures
        ledger ==> r is Stable,
        !ledger ==> r is PerServer,
{
    if ledger {
        if cfg!(feature = "mut_ledger_spool_per_server") { Root::PerServer } else { Root::Stable }
    } else if cfg!(feature = "mut_unledgered_spool_stable") {
        Root::Stable
    } else {
        Root::PerServer
    }
}

/// Why execd is stopping: `SIGTERM` or `SIGINT` (a stop: the sandbox is stopping, so nothing may be left
/// running), or `SIGHUP` (a restart: the next execd adopts the ledgered jobs). service-init sends
/// `SIGTERM` for a sandbox stop and `SIGHUP` for a restart of execd (`workload_signal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Term,
    Hup,
}

/// What execd's shutdown (a `SIGTERM`, a restart) does with one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shutdown {
    /// Signal the group. Only an explicit `kill` or `wipe` signals a job that a restart can find.
    pub signal: bool,
    /// Keep the group's record in the ledger, so the next execd finds it.
    pub record: bool,
}

/// S6: a stop (`SIGTERM`, `SIGINT`) ends every group, a job's included, with or without a ledger, and keeps no
/// record of it (a ledger that kept a job the stop ended would make the next execd adopt a group that is gone).
/// A restart (`SIGHUP`) is S3 to S5. The `mut_term_keeps_jobs` mutant makes a stop a restart.
///
/// S3, S4, S5: what a restart of execd does to a group. `recorded` is whether execd keeps a
/// ledger (`EXECD_STATE_DIR`), so a next execd can find a job's group.
///
/// - S3: a recorded job's group is never signalled: its processes run on, and the next execd adopts
///   them ([`recover`]). An unrecorded job's group is signalled (S3b): nothing could ever find or
///   stop it again.
/// - S4: a recorded job's record is kept, and only a recorded job's.
/// - S5: a command's group is signalled, since its client's connection ends with execd; it has no
///   record.
pub fn shutdown_of(kind: Kind, recorded: bool, stop: Stop) -> (r: Shutdown)
    ensures
        // S6: a stop ends every group, with or without a ledger, and records nothing.
        stop is Term ==> (r.signal && !r.record),
        // A restart keeps a ledgered job and ends an unledgered one, as a disconnect does.
        stop is Hup ==> ((kind is Job && recorded) ==> !r.signal),
        stop is Hup ==> ((kind is Job && !recorded) ==> r.signal),
        stop is Hup ==> ((kind is Job && recorded) ==> r.record),
        stop is Hup ==> (!(kind is Job && recorded) ==> !r.record),
        stop is Hup ==> (kind is Command ==> r.signal),
{
    match stop {
        Stop::Term => {
            if !cfg!(feature = "mut_term_keeps_jobs") {
                return Shutdown { signal: true, record: false };
            }
        },
        Stop::Hup => {},
    }
    match kind {
        Kind::Job => Shutdown {
            signal: if recorded { cfg!(feature = "mut_shutdown_signals_jobs") } else { !cfg!(feature = "mut_shutdown_orphans_unrecorded") },
            record: recorded && !cfg!(feature = "mut_shutdown_forgets_jobs"),
        },
        Kind::Command => Shutdown { signal: !cfg!(feature = "mut_shutdown_spares_commands"), record: false },
    }
}

// ---------------------------------------------------------------------------------------------
// Resuming a job's output after a restart of execd.
// ---------------------------------------------------------------------------------------------

/// Largest offset the resume arithmetic accepts: the sum of the two streams' offsets and a
/// journal's bytes must fit in a `u64` (16 EiB of output).
pub const OFFSET_MAX: u64 = 0x7fff_ffff_ffff_ffff;

/// Where a restarted execd reads a job's two spool files from, and where its ring ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resume {
    /// The next byte of `out` to read.
    pub out: u64,
    /// The next byte of `err` to read.
    pub err: u64,
    /// The ring offset of the next byte: how many bytes execd had consumed.
    pub end: u64,
}

/// X1, X2, X3: where a restarted execd resumes. The ledger names the job's spool, and in it the
/// journal: the cursors (`out`, `err`) at the journal's start, and `jo` and `je` bytes of each
/// stream that execd had consumed and journalled after them, in the order it ring-ed them.
///
/// - X1: each stream resumes exactly after the last byte consumed: its cursor plus its journalled
///   bytes. A byte consumed before the crash is not read again, one not yet consumed is.
/// - X2: the ring's end is the bytes consumed, so a client's offset means the same byte before and
///   after: `end == out + err`.
/// - X3 ([`replay_skip`]): a journalled byte already in a chunk file is not put in the ring twice.
pub fn resume(out: u64, err: u64, jo: u64, je: u64) -> (r: Resume)
    requires
        out as int + err as int + jo as int + je as int <= OFFSET_MAX as int,
    ensures
        r.out == out + jo,
        r.err == err + je,
        r.end as int == r.out as int + r.err as int,
{
    let o = if cfg!(feature = "mut_resume_cursor_lags") { out } else { out + jo };
    let e = err + je;
    let end = if cfg!(feature = "mut_resume_end_drops_journal") { out + err } else { o + e };
    Resume { out: o, err: e, end }
}

/// X3: of a journal record of `len` bytes at ring offset `at`, how many leading bytes are already in
/// the chunk files, which hold the ring below `chunk_end`. Those are skipped; the rest are replayed
/// into the ring, so every offset below `chunk_end` is served by a chunk and every offset at or
/// above it by the journal, each once.
pub fn replay_skip(chunk_end: u64, at: u64, len: u64) -> (skip: u64)
    requires
        at as int + len as int <= u64::MAX as int,
    ensures
        skip <= len,
        chunk_end > at ==> skip as int == (if chunk_end - at < len { (chunk_end - at) as int } else { len as int }),
        chunk_end <= at ==> skip == 0,
{
    if cfg!(feature = "mut_resume_replays_chunked") {
        0
    } else if chunk_end <= at {
        0
    } else if chunk_end - at < len {
        chunk_end - at
    } else {
        len
    }
}

// ---------------------------------------------------------------------------------------------
// Positional stdin (the `write` op): `written` counts the stream's bytes applied so far.
// ---------------------------------------------------------------------------------------------

/// The most writes waiting on one job's stdin (for the pipe, or for earlier bytes).
pub const WAITERS_MAX: usize = 16;

/// W1: a write may wait only while fewer than `WAITERS_MAX` others do (the 17th is refused with
/// 429). `waiters` is the count before this one. W1-live (completeness): once a waiter has left,
/// a write that asks again is admitted, so a refusal never sticks.
pub fn admit_waiter(waiters: usize) -> (r: bool)
    ensures
        r <==> waiters < WAITERS_MAX,
{
    if cfg!(feature = "mut_stdin_waiters_unbounded") {
        true
    } else if cfg!(feature = "mut_stdin_waiter_stuck") {
        waiters + 1 < WAITERS_MAX
    } else {
        waiters < WAITERS_MAX
    }
}

/// What a positional write does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Bytes are missing before `at` and the job lives: wait for `written` to reach `at`.
    Wait,
    /// Nothing can land: the job ended or the stdin was closed (409), or a write waited for bytes
    /// that the end of the job cut off.
    Refuse,
    /// Every byte is applied already (a retry or a replay): answer with the state; close at EOF.
    Done { close: bool },
    /// Skip the first `skip` bytes (already applied), write the next `take`, then close at EOF.
    Apply { skip: u64, take: u64, close: bool },
}

/// What `stdin_plan` promises (W2 to W5).
pub open spec fn post_plan(written: u64, at: u64, len: u64, eof: bool, open: bool, ended: bool, p: Plan) -> bool {
    // W5: a write ahead of `written` waits for the bytes before it; it never applies.
    &&& (at > written ==> p == (if ended { Plan::Refuse } else { Plan::Wait }))
    // W2: bytes already applied are answered, never applied again (idempotent, whatever the state).
    &&& (at <= written && at as int + len as int <= written as int ==> p == (Plan::Done { close: eof }))
    // W3: nothing lands on an ended job; W4: nothing lands after EOF.
    &&& (at <= written && at as int + len as int > written as int && (ended || !open) ==> p == Plan::Refuse)
    // W2: what is below `written` is skipped, the rest is applied, and it starts exactly at `written`.
    &&& (at <= written && at as int + len as int > written as int && open && !ended ==> p == (Plan::Apply {
        skip: if written - at < len { (written - at) as u64 } else { len },
        take: (len - (if written - at < len { (written - at) as u64 } else { len })) as u64,
        close: eof,
    }))
    &&& (p is Apply ==> p->Apply_take > 0 && at as int + p->Apply_skip as int == written as int
        && p->Apply_skip as int + p->Apply_take as int == len as int && open && !ended)
    &&& (p is Wait ==> at > written && !ended)
}

/// The definition `stdin_plan` computes (so the progress lemma reasons over the function itself:
/// `stdin_plan` ensures `p == plan_of(..)`), with each mutation's branch in the same place.
pub open spec fn plan_of(written: u64, at: u64, len: u64, eof: bool, open: bool, ended: bool) -> Plan {
    if at > written {
        if ended {
            Plan::Refuse
        } else if cfg!(feature = "mut_stdin_applies_ahead") {
            Plan::Apply { skip: 0, take: len, close: eof }
        } else {
            Plan::Wait
        }
    } else if cfg!(feature = "mut_stdin_waits_at_written") && at == written && len > 0 && !ended {
        Plan::Wait
    } else if cfg!(feature = "mut_stdin_wait_sticks") && at < written && len > 0 && !ended {
        Plan::Wait
    } else {
        let below = (written - at) as u64;
        let skip: u64 = if cfg!(feature = "mut_stdin_replays") {
            0
        } else if cfg!(feature = "mut_stdin_apply_off_start") && below < len {
            (below + 1) as u64
        } else if below < len {
            below
        } else {
            len
        };
        if len - skip == 0 {
            Plan::Done { close: eof }
        } else if (ended && !cfg!(feature = "mut_stdin_writes_ended"))
            || (!open && !cfg!(feature = "mut_stdin_writes_after_eof")) {
            Plan::Refuse
        } else {
            Plan::Apply { skip, take: (len - skip) as u64, close: eof }
        }
    }
}

/// W2 to W5: the decision of one positional write of `len` bytes at `at`, with `written` bytes of
/// the stream applied, `open` (the pipe is not closed) and `ended` (the job ended or was removed).
/// `eof` asks to close the stdin after the bytes. The shell calls it for each wait and once with
/// the pipe held, and does only the I/O the plan names.
pub fn stdin_plan(written: u64, at: u64, len: u64, eof: bool, open: bool, ended: bool) -> (p: Plan)
    ensures
        p == plan_of(written, at, len, eof, open, ended),
        post_plan(written, at, len, eof, open, ended, p),
        // W2a: the bytes below `written` are skipped, never applied again.
        p is Apply && at <= written ==> p->Apply_skip as int >= (if written - at < len { (written - at) as int } else { len as int }),
        // W2c: an apply starts exactly at `written`: no byte is skipped that was not applied.
        p is Apply ==> at as int + p->Apply_skip as int == written as int,
        // W5-live: nothing missing, nothing waited for.
        at <= written ==> !(p is Wait),
{
    if at > written {
        if ended {
            Plan::Refuse
        } else if cfg!(feature = "mut_stdin_applies_ahead") {
            Plan::Apply { skip: 0, take: len, close: eof }
        } else {
            Plan::Wait
        }
    } else if cfg!(feature = "mut_stdin_waits_at_written") && at == written && len > 0 && !ended {
        Plan::Wait
    } else if cfg!(feature = "mut_stdin_wait_sticks") && at < written && len > 0 && !ended {
        Plan::Wait
    } else {
        let below = written - at;
        let skip = if cfg!(feature = "mut_stdin_replays") {
            0
        } else if cfg!(feature = "mut_stdin_apply_off_start") && below < len {
            below + 1
        } else if below < len {
            below
        } else {
            len
        };
        if len - skip == 0 {
            Plan::Done { close: eof }
        } else if (ended && !cfg!(feature = "mut_stdin_writes_ended"))
            || (!open && !cfg!(feature = "mut_stdin_writes_after_eof")) {
            Plan::Refuse
        } else {
            Plan::Apply { skip, take: len - skip, close: eof }
        }
    }
}

/// W2: `written` after `n` bytes landed: it only grows, by exactly what landed.
pub fn advance(written: u64, n: u64) -> (r: u64)
    ensures
        r >= written,
        written as int + n as int <= u64::MAX as int ==> r as int == written as int + n as int,
{
    if cfg!(feature = "mut_stdin_written_shrinks") {
        n
    } else if written <= u64::MAX - n { written + n } else { u64::MAX }
}

/// What of a plan's `take` bytes is still to land after `landed` did (the shell's `write(2)`s).
pub fn remaining(take: u64, landed: u64) -> (r: u64)
    ensures
        r as int == (if landed < take { take as int - landed as int } else { 0int }),
{
    if landed < take { take - landed } else { 0 }
}

/// W4: the stdin closes at EOF only once every byte of the write landed, so EOF never cuts a write
/// short (a short write answers its exact `written` and leaves the pipe open).
pub fn closes(close: bool, remaining: u64) -> (r: bool)
    ensures
        r <==> (close && remaining == 0),
{
    if cfg!(feature = "mut_stdin_closes_early") { close } else { close && remaining == 0 }
}

/// W1-live, progress: an admitted write is never stuck. The rank is the bytes of the write still
/// above `written`, plus one for the answer. Under fairness F1 (what a write waits for is filled,
/// or the job ends), F2 (the pipe takes at least one byte of a write) and F3 (the task runs); these
/// three are the property's axioms: a
/// `Wait` is left by any growth of `written` or by the job's end, and each `Apply` lowers the rank,
/// so the write is answered in at most `stdin_rank` steps. A write is `Wait` only while ahead and
/// alive, `Refuse`d only when ended or closed, and answered `Done` at rank 1.
pub open spec fn stdin_rank(written: u64, at: u64, len: u64) -> int {
    (if at as int + len as int > written as int { at as int + len as int - written as int } else { 0int }) + 1
}

pub proof fn lemma_stdin_progress(
    written: u64,
    at: u64,
    len: u64,
    eof: bool,
    open: bool,
    ended: bool,
    written2: u64,
)
    requires
        written2 > written,
    ensures
        ({
            let p = plan_of(written, at, len, eof, open, ended);
            let p2 = plan_of(written2, at, len, eof, open, ended);
            &&& stdin_rank(written, at, len) >= 1
            // a fair step (the gap fills, or bytes land) lowers the rank while the write is unanswered
            &&& (stdin_rank(written, at, len) > 1 ==> stdin_rank(written2, at, len) < stdin_rank(written, at, len))
            // F1 (the gap fills): once `written` reaches `at`, a living write is not made to wait again
            // (and is never refused while open), so it applies or is answered
            &&& (written2 >= at && !ended ==> !(p2 is Wait))
            &&& (written2 >= at && !ended && open ==> !(p2 is Refuse))
            // an apply that lands all its bytes leaves the write answered (rank 1), then it is `Done`
            &&& (p is Apply && written2 as int == written as int + p->Apply_take as int
                ==> stdin_rank(written2, at, len) == 1 && p2 == (Plan::Done { close: eof }))
        }),
{
}

// ---------------------------------------------------------------------------------------------
// Output offsets.
// ---------------------------------------------------------------------------------------------

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
/// `Data` frame the shell sends takes at least one byte: `execd_ring::window`'s R1 promises it can, `r.0 < end && max > 0 ==> r.1 > 0`) and F3 (the
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

/// Two records say the same (by what they hold).
pub open spec fn same(a: Rec, b: Rec) -> bool {
    a.scope@ == b.scope@ && a.key@ == b.key@ && a.id@ == b.id@ && a.digest@ == b.digest@
}

/// One start key's record in the ledger: a keyed start's key, in its scope, names the job it started and the digest
/// of the spec it started with (a hash of command, args, cwd, env, pty, stdin_open and hold: the digest alone, never
/// the spec, because the env can hold secrets). The record is as old as its job: it is written with the job's entry,
/// saved with the job's ledger entry, restored with it, and dropped only when the job is taken (K5-K8).
#[verifier::allow(autoderive_clone_without_spec)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rec {
    pub scope: String,
    pub key: String,
    pub id: String,
    pub digest: String,
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
    /// The start keys of the jobs in `entries`: the ledger's records (`Rec`).
    recs: Vec<Rec>,
}

impl<C> Table<C> {
    /// The keys are distinct. `new` makes a table that satisfies it and every method that changes
    /// the table keeps it; the field is private, so no other table exists.
    pub closed spec fn wf(self) -> bool {
        &&& forall|i: int, j: int|
            0 <= i < j < self.entries@.len() ==> !key_eq(
                (#[trigger] self.entries@[i]).0,
                (#[trigger] self.entries@[j]).0,
            )
        &&& forall|i: int, j: int|
            0 <= i < j < self.recs@.len() ==> !(
                (#[trigger] self.recs@[i]).scope@ == (#[trigger] self.recs@[j]).scope@
                    && self.recs@[i].key@ == self.recs@[j].key@)
    }

    /// The ledger's records, in order.
    pub closed spec fn recs(self) -> Seq<Rec> {
        self.recs@
    }

    /// Some job is under `(scope, id)`.
    pub closed spec fn holds_at(self, scope: Seq<char>, id: Seq<char>) -> bool {
        exists|i: int|
            0 <= i < self.entries@.len() && (#[trigger] self.entries@[i]).0.0@ == scope
                && self.entries@[i].0.1@ == id
    }

    /// The ledger has a record: `key` of `scope` names job `id`, started with `digest`.
    pub closed spec fn recorded(self, scope: Seq<char>, key: Seq<char>, id: Seq<char>, digest: Seq<char>) -> bool {
        exists|i: int|
            0 <= i < self.recs@.len() && (#[trigger] self.recs@[i]).scope@ == scope
                && self.recs@[i].key@ == key && self.recs@[i].id@ == id && self.recs@[i].digest@ == digest
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
        Table { entries: Vec::new(), recs: Vec::new() }
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

    fn has_id(&self, scope: &str, id: &str) -> (r: bool)
        ensures
            r == self.holds_at(scope@, id@),
    {
        let mut n = 0;
        while n < self.entries.len()
            invariant
                n <= self.entries@.len(),
                forall|i: int|
                    0 <= i < n ==> !((#[trigger] self.entries@[i]).0.0@ == scope@ && self.entries@[i].0.1@ == id@),
            decreases self.entries@.len() - n,
        {
            let e = &self.entries[n];
            if e.0.0.as_str() == scope && e.0.1.as_str() == id {
                return true;
            }
            n += 1;
        }
        false
    }

    /// The ledger's record of start key `k` in `scope`, when the job it names is still in the table: that job's id, and
    /// whether its stored digest is `digest`.
    fn lookup(&self, scope: &str, k: &str, digest: &str) -> (r: Option<(String, bool)>)
        requires
            self.wf(),
        ensures
            forall|j: int|
                0 <= j < self.recs@.len() && (#[trigger] self.recs@[j]).scope@ == scope@ && self.recs@[j].key@ == k@
                    && self.holds_at(scope@, self.recs@[j].id@) ==> (r is Some && r->Some_0.0@ == self.recs@[j].id@
                    && r->Some_0.1 == (self.recs@[j].digest@ == digest@)),
            r is Some ==> exists|j: int|
                0 <= j < self.recs@.len() && (#[trigger] self.recs@[j]).scope@ == scope@ && self.recs@[j].key@ == k@
                    && self.holds_at(scope@, self.recs@[j].id@),
    {
        let mut n = 0;
        while n < self.recs.len()
            invariant
                n <= self.recs@.len(),
                self.wf(),
                forall|i: int|
                    0 <= i < n ==> !((#[trigger] self.recs@[i]).scope@ == scope@ && self.recs@[i].key@ == k@),
            decreases self.recs@.len() - n,
        {
            let r = &self.recs[n];
            if r.scope.as_str() == scope && r.key.as_str() == k {
                proof {
                    assert forall|j: int|
                        0 <= j < self.recs@.len() && (#[trigger] self.recs@[j]).scope@ == scope@
                            && self.recs@[j].key@ == k@ implies j == n as int by {
                        if j < n as int {
                        } else if j > n as int {
                            assert(!(self.recs@[n as int].scope@ == self.recs@[j].scope@
                                && self.recs@[n as int].key@ == self.recs@[j].key@));
                        }
                    }
                }
                if self.has_id(scope, r.id.as_str()) {
                    return Some((r.id.clone(), r.digest.as_str() == digest));
                }
                return None;
            }
            n += 1;
        }
        None
    }

    /// K1-K5: what a start of `(scope, id)` does, with the start key `key` (`None`: no key) and the spec digest
    /// `digest`. The ledger's record for the key counts only while its job is still in the table. The decision is
    /// [`start_decision`], over what the table and the ledger hold.
    pub fn begin(&self, scope: &str, id: &str, key: Option<&str>, digest: &str) -> (r: Begin)
        requires
            self.wf(),
        ensures
            // K1: a key already used for this command, by a job still in the table, replays that job.
            forall|j: int|
                0 <= j < self.recs().len() && key is Some && (#[trigger] self.recs()[j]).scope@ == scope@
                    && self.recs()[j].key@ == key->Some_0@ && self.holds_at(scope@, self.recs()[j].id@)
                    && self.recs()[j].digest@ == digest@ ==> (r is Replay && r->Replay_0@ == self.recs()[j].id@
                    && self.holds_at(scope@, r->Replay_0@)),
            // K2: a key already used for another command is refused, as that.
            forall|j: int|
                0 <= j < self.recs().len() && key is Some && (#[trigger] self.recs()[j]).scope@ == scope@
                    && self.recs()[j].key@ == key->Some_0@ && self.holds_at(scope@, self.recs()[j].id@)
                    && self.recs()[j].digest@ != digest@ ==> r == Begin::Refuse(Refusal::KeyReused),
            // K3, K5: with no such record, a free id starts and a held one is refused.
            (key is None || !(exists|j: int|
                0 <= j < self.recs().len() && (#[trigger] self.recs()[j]).scope@ == scope@
                    && self.recs()[j].key@ == key->Some_0@ && self.holds_at(scope@, self.recs()[j].id@)))
                ==> (r == (if self.holds_at(scope@, id@) { Begin::Refuse(Refusal::IdExists) } else { Begin::Start })),
            // K4: a start with no key never replays and is never `KeyReused`.
            key is None ==> (!(r is Replay) && r != Begin::Refuse(Refusal::KeyReused)),
    {
        // J1: the id is the claim; `admit` is where that decision is made.
        let id_held = match admit(self.has_id(scope, id)) {
            Admit::Exists => true,
            Admit::Fresh => false,
        };
        let by_key = match key {
            Some(k) => self.lookup(scope, k, digest),
            None => {
                if cfg!(feature = "mut_unkeyed_replays") && id_held {
                    Some((id.to_string(), true))
                } else {
                    None
                }
            },
        };
        start_decision(id_held, by_key)
    }

    /// K7: write a start key's record. A key's record is one: a second record of the same key in the same scope
    /// replaces the first (a record whose job is gone); every other key's record stays.
    pub fn record(&mut self, rec: Rec)
        requires
            old(self).wf(),
        ensures
            final(self).wf(),
            final(self).rows() == old(self).rows(),
            final(self).recorded(rec.scope@, rec.key@, rec.id@, rec.digest@),
            forall|s: Seq<char>, k: Seq<char>, j: Seq<char>, d: Seq<char>|
                !(s == rec.scope@ && k == rec.key@) ==> (final(self).recorded(s, k, j, d)
                    == old(self).recorded(s, k, j, d)),
    {
        let ghost before = self.recs@;
        let mut kept: Vec<Rec> = Vec::new();
        let mut n = 0;
        while n < self.recs.len()
            invariant
                n <= self.recs@.len(),
                self.recs@ == before,
                self.wf(),
                forall|i: int| 0 <= i < kept@.len() ==> {
                    &&& !((#[trigger] kept@[i]).scope@ == rec.scope@ && kept@[i].key@ == rec.key@)
                    &&& exists|m: int| 0 <= m < n && same(before[m], kept@[i])
                },
                forall|m: int|
                    0 <= m < n && !((#[trigger] before[m]).scope@ == rec.scope@ && before[m].key@ == rec.key@)
                        ==> exists|i: int| 0 <= i < kept@.len() && same(kept@[i], before[m]),
                forall|a: int, b: int|
                    0 <= a < b < kept@.len() ==> !(
                        (#[trigger] kept@[a]).scope@ == (#[trigger] kept@[b]).scope@ && kept@[a].key@ == kept@[b].key@),
            decreases self.recs@.len() - n,
        {
            let r = &self.recs[n];
            let ghost old_kept = kept@;
            if !(r.scope.as_str() == rec.scope.as_str() && r.key.as_str() == rec.key.as_str()) && !cfg!(feature = "mut_key_record_clears") {
                kept.push(Rec { scope: r.scope.clone(), key: r.key.clone(), id: r.id.clone(), digest: r.digest.clone() });
                proof {
                    let k = old_kept.len() as int;
                    assert(kept@.len() == k + 1);
                    assert(forall|i: int| 0 <= i < k ==> kept@[i] == old_kept[i]);
                    assert(same(kept@[k], before[n as int]));
                    assert forall|i: int| 0 <= i < kept@.len() implies {
                        &&& !((#[trigger] kept@[i]).scope@ == rec.scope@ && kept@[i].key@ == rec.key@)
                        &&& exists|m: int| 0 <= m < n + 1 && same(before[m], kept@[i])
                    } by {
                        if i < k {
                            assert(kept@[i] == old_kept[i]);
                            let m = choose|m: int| 0 <= m < n && same(before[m], old_kept[i]);
                            assert(same(before[m], kept@[i]));
                        } else {
                            assert(same(before[n as int], kept@[i]));
                        }
                    }
                    assert forall|m: int|
                        0 <= m < n + 1 && !((#[trigger] before[m]).scope@ == rec.scope@ && before[m].key@ == rec.key@)
                            implies exists|i: int| 0 <= i < kept@.len() && same(kept@[i], before[m]) by {
                        if m == n as int {
                            assert(same(kept@[k], before[m]));
                        } else {
                            let i = choose|i: int| 0 <= i < k && same(old_kept[i], before[m]);
                            assert(kept@[i] == old_kept[i]);
                        }
                    }
                    assert forall|a: int, b: int| 0 <= a < b < kept@.len() implies !(
                        (#[trigger] kept@[a]).scope@ == (#[trigger] kept@[b]).scope@
                            && kept@[a].key@ == kept@[b].key@) by {
                        if b == k {
                            let m = choose|m: int| 0 <= m < n && same(before[m], old_kept[a]);
                            assert(kept@[a] == old_kept[a]);
                            assert(self.recs@[m] == before[m]);
                            assert(self.recs@[n as int] == before[n as int]);
                        } else {
                            assert(kept@[a] == old_kept[a]);
                            assert(kept@[b] == old_kept[b]);
                        }
                    }
                }
            } else {
                proof {
                    assert forall|m: int|
                        0 <= m < n + 1 && !((#[trigger] before[m]).scope@ == rec.scope@ && before[m].key@ == rec.key@)
                            implies exists|i: int| 0 <= i < kept@.len() && same(kept@[i], before[m]) by {
                        assert(m < n as int);
                    }
                    assert forall|i: int| 0 <= i < kept@.len() implies
                        ((#[trigger] kept@[i]) == kept@[i]
                            && exists|m: int| 0 <= m < n + 1 && same(before[m], kept@[i])) by {
                        let m = choose|m: int| 0 <= m < n && same(before[m], kept@[i]);
                        assert(same(before[m], kept@[i]));
                    }
                }
            }
            n += 1;
        }
        let ghost kept_before_push = kept@;
        let ghost scope = rec.scope@;
        let ghost key = rec.key@;
        let ghost id = rec.id@;
        let ghost digest = rec.digest@;
        kept.push(rec);
        self.recs = kept;
        proof {
            let last = kept_before_push.len() as int;
            assert(self.recs@.len() == last + 1);
            assert(forall|i: int| 0 <= i < last ==> self.recs@[i] == kept_before_push[i]);
            assert(self.recs@[last].scope@ == scope && self.recs@[last].key@ == key);
            assert(self.recs@[last].id@ == id && self.recs@[last].digest@ == digest);
            assert(self.recorded(scope, key, id, digest));
            assert forall|a: int, b: int| 0 <= a < b < self.recs@.len() implies !(
                (#[trigger] self.recs@[a]).scope@ == (#[trigger] self.recs@[b]).scope@
                    && self.recs@[a].key@ == self.recs@[b].key@) by {
                assert(self.recs@[a] == kept_before_push[a]);
                if b != last {
                    assert(self.recs@[b] == kept_before_push[b]);
                }
            }
            assert forall|s: Seq<char>, k: Seq<char>, j: Seq<char>, d: Seq<char>|
                !(s == scope && k == key) implies (self.recorded(s, k, j, d)
                    == old(self).recorded(s, k, j, d)) by {
                if self.recorded(s, k, j, d) {
                    let i = choose|i: int| 0 <= i < self.recs@.len() && self.recs@[i].scope@ == s
                        && self.recs@[i].key@ == k && self.recs@[i].id@ == j && self.recs@[i].digest@ == d;
                    assert(i != last);
                    assert(self.recs@[i] == kept_before_push[i]);
                    let m = choose|m: int| 0 <= m < before.len() && same(before[m], kept_before_push[i]);
                    assert(old(self).recs@[m].scope@ == s && old(self).recs@[m].key@ == k);
                }
                if old(self).recorded(s, k, j, d) {
                    let m = choose|m: int| 0 <= m < before.len() && before[m].scope@ == s
                        && before[m].key@ == k && before[m].id@ == j && before[m].digest@ == d;
                    let i = choose|i: int| 0 <= i < kept_before_push.len() && same(kept_before_push[i], before[m]);
                    assert(self.recs@[i] == kept_before_push[i]);
                }
            }
        }
    }

    /// K8: drop the records of the job under `(scope, id)` and no other record.
    fn forget(&mut self, scope: &str, id: &str)
        requires
            old(self).wf(),
        ensures
            final(self).wf(),
            final(self).rows() == old(self).rows(),
            forall|s: Seq<char>, k: Seq<char>, j: Seq<char>, d: Seq<char>|
                final(self).recorded(s, k, j, d) ==> (old(self).recorded(s, k, j, d)
                    && !(s == scope@ && j == id@)),
            forall|s: Seq<char>, k: Seq<char>, j: Seq<char>, d: Seq<char>|
                old(self).recorded(s, k, j, d) && !(s == scope@ && j == id@) ==> final(self).recorded(s, k, j, d),
    {
        let ghost before = self.recs@;
        let mut kept: Vec<Rec> = Vec::new();
        let mut n = 0;
        while n < self.recs.len()
            invariant
                n <= self.recs@.len(),
                self.recs@ == before,
                self.wf(),
                forall|i: int| 0 <= i < kept@.len() ==> {
                    &&& !((#[trigger] kept@[i]).scope@ == scope@ && kept@[i].id@ == id@)
                    &&& exists|m: int| 0 <= m < n && same(before[m], kept@[i])
                },
                forall|m: int|
                    0 <= m < n && !((#[trigger] before[m]).scope@ == scope@ && before[m].id@ == id@)
                        ==> exists|i: int| 0 <= i < kept@.len() && same(kept@[i], before[m]),
                forall|a: int, b: int|
                    0 <= a < b < kept@.len() ==> !(
                        (#[trigger] kept@[a]).scope@ == (#[trigger] kept@[b]).scope@ && kept@[a].key@ == kept@[b].key@),
            decreases self.recs@.len() - n,
        {
            let r = &self.recs[n];
            let ghost old_kept = kept@;
            if !(r.scope.as_str() == scope && r.id.as_str() == id) || cfg!(feature = "mut_key_take_forgets_none") {
                kept.push(Rec { scope: r.scope.clone(), key: r.key.clone(), id: r.id.clone(), digest: r.digest.clone() });
                proof {
                    let k = old_kept.len() as int;
                    assert(kept@.len() == k + 1);
                    assert(forall|i: int| 0 <= i < k ==> kept@[i] == old_kept[i]);
                    assert(same(kept@[k], before[n as int]));
                    assert forall|i: int| 0 <= i < kept@.len() implies {
                        &&& !((#[trigger] kept@[i]).scope@ == scope@ && kept@[i].id@ == id@)
                        &&& exists|m: int| 0 <= m < n + 1 && same(before[m], kept@[i])
                    } by {
                        if i < k {
                            assert(kept@[i] == old_kept[i]);
                            let m = choose|m: int| 0 <= m < n && same(before[m], old_kept[i]);
                            assert(same(before[m], kept@[i]));
                        } else {
                            assert(same(before[n as int], kept@[i]));
                        }
                    }
                    assert forall|m: int|
                        0 <= m < n + 1 && !((#[trigger] before[m]).scope@ == scope@ && before[m].id@ == id@)
                            implies exists|i: int| 0 <= i < kept@.len() && same(kept@[i], before[m]) by {
                        if m == n as int {
                            assert(same(kept@[k], before[m]));
                        } else {
                            let i = choose|i: int| 0 <= i < k && same(old_kept[i], before[m]);
                            assert(kept@[i] == old_kept[i]);
                        }
                    }
                    assert forall|a: int, b: int| 0 <= a < b < kept@.len() implies !(
                        (#[trigger] kept@[a]).scope@ == (#[trigger] kept@[b]).scope@
                            && kept@[a].key@ == kept@[b].key@) by {
                        if b == k {
                            let m = choose|m: int| 0 <= m < n && same(before[m], old_kept[a]);
                            assert(kept@[a] == old_kept[a]);
                            assert(self.recs@[m] == before[m]);
                            assert(self.recs@[n as int] == before[n as int]);
                        } else {
                            assert(kept@[a] == old_kept[a]);
                            assert(kept@[b] == old_kept[b]);
                        }
                    }
                }
            } else {
                proof {
                    assert forall|m: int|
                        0 <= m < n + 1 && !((#[trigger] before[m]).scope@ == scope@ && before[m].id@ == id@)
                            implies exists|i: int| 0 <= i < kept@.len() && same(kept@[i], before[m]) by {
                        assert(m < n as int);
                    }
                    assert forall|i: int| 0 <= i < kept@.len() implies
                        ((#[trigger] kept@[i]) == kept@[i]
                            && exists|m: int| 0 <= m < n + 1 && same(before[m], kept@[i])) by {
                        let m = choose|m: int| 0 <= m < n && same(before[m], kept@[i]);
                        assert(same(before[m], kept@[i]));
                    }
                }
            }
            n += 1;
        }
        self.recs = kept;
        proof {
            assert forall|s: Seq<char>, k: Seq<char>, j: Seq<char>, d: Seq<char>|
                self.recorded(s, k, j, d) implies (old(self).recorded(s, k, j, d) && !(s == scope@ && j == id@)) by {
                let i = choose|i: int| 0 <= i < self.recs@.len() && self.recs@[i].scope@ == s
                    && self.recs@[i].key@ == k && self.recs@[i].id@ == j && self.recs@[i].digest@ == d;
                let m = choose|m: int| 0 <= m < before.len() && same(before[m], self.recs@[i]);
                assert(old(self).recs@[m].scope@ == s);
            }
            assert forall|s: Seq<char>, k: Seq<char>, j: Seq<char>, d: Seq<char>|
                old(self).recorded(s, k, j, d) && !(s == scope@ && j == id@) implies self.recorded(s, k, j, d) by {
                let m = choose|m: int| 0 <= m < before.len() && before[m].scope@ == s
                    && before[m].key@ == k && before[m].id@ == j && before[m].digest@ == d;
                let i = choose|i: int| 0 <= i < self.recs@.len() && same(self.recs@[i], before[m]);
                assert(self.recs@[i].scope@ == s);
            }
        }
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
                        self.forget(k.0.as_str(), k.1.as_str());
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

impl<C> Table<C> {
    /// The ledger's records (the model's state hashes them).
    pub fn records(&self) -> Vec<Rec> {
        self.recs.clone()
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
            recs: self.recs.clone(),
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
            && self.recs.len() == other.recs.len()
            && self.recs.iter().all(|r| other.recs.contains(r))
            && self
                .entries
                .iter()
                .all(|(k, c)| other.entries.iter().any(|(k2, c2)| k == k2 && c == c2))
    }
}

impl<C: Eq> Eq for Table<C> {}
