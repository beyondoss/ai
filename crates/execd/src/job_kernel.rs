//! The decisions of execd's job table, as pure functions: which `(scope, id)` a start may take,
//! which jobs an operation sees, what state a job is in and what each event does to it, when a
//! signal is sent, when a job may leave the table, when a held job dies with its client, and where
//! an output read or a stream frame starts. The shell is `jobs.rs`.
//!
//! No I/O, no async, no clock (a time arrives in the event), no closure or trait object in a
//! decision, so a model checker calls exactly what production calls (`tests/job_model.rs`) and a
//! later proof can take it as it is. Production keeps two locks and the kernel follows them:
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

use std::collections::HashMap;

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

/// Whether a start may take its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    Fresh,
    /// The key is in the table: the id is the claim, so two starts of one id cannot both run.
    Exists,
}

/// Every job by `(scope, id)`. `C` is what the shell keeps per job (a shared cell in production, a
/// handle in the model): the kernel never looks inside it.
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
        if self.entries.contains_key(key) && !cfg!(feature = "mut_start_overwrites") {
            Admit::Exists
        } else {
            Admit::Fresh
        }
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
            .filter(|((sc, id), _)| (sc == scope || cfg!(feature = "mut_scope_ignored")) && pick.has(id))
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

/// Where a job is in its life. One way only: `Spawning → Running → Exited → Ended` (`Exited` is
/// skipped when the group dies with the shell). The number of the group exists from `Running`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// The entry holds its id; the process is not yet spawned.
    Spawning,
    Running { pgid: i32 },
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

impl Named {
    pub fn as_str(self) -> &'static str {
        match self {
            Named::Running => "running",
            Named::Exited => "exited",
            Named::Ended => "ended",
        }
    }
}

/// How a job counts in the host's view of what is alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alive {
    No,
    Yes,
    /// Alive and the endpoint's.
    Kept,
}

impl State {
    /// A job just entered the table: spawning, kept unless its start is held (and so `background`).
    pub fn new(started: u64, hold: bool) -> Self {
        Self {
            phase: Phase::Spawning,
            claim: if hold {
                Claim::Held
            } else {
                Claim::Kept {
                    num: None,
                    kind: Some("background".to_string()),
                }
            },
            seen: 0,
            removed: false,
            started,
        }
    }

    /// The group's number: 0 while spawning.
    pub fn pgid(&self) -> i32 {
        match self.phase {
            Phase::Spawning => 0,
            Phase::Running { pgid } | Phase::Exited { pgid, .. } | Phase::Ended { pgid, .. } => pgid,
        }
    }

    pub fn ended(&self) -> bool {
        matches!(self.phase, Phase::Ended { .. })
    }

    pub fn kept(&self) -> bool {
        matches!(self.claim, Claim::Kept { .. })
    }

    /// `(state, code)` as `read` and `list` answer it: the code once there is one.
    pub fn named(&self) -> (Named, Option<i32>) {
        match self.phase {
            Phase::Ended { code, .. } => (Named::Ended, Some(code)),
            Phase::Exited { code, .. } => (Named::Exited, Some(code)),
            Phase::Spawning | Phase::Running { .. } => (Named::Running, None),
        }
    }

    /// The signal that killed the shell, once the job has a code.
    pub fn signal(&self) -> Option<i32> {
        match self.phase {
            Phase::Ended { signal, .. } | Phase::Exited { signal, .. } => signal,
            Phase::Spawning | Phase::Running { .. } => None,
        }
    }

    pub fn num(&self) -> Option<u64> {
        match self.claim {
            Claim::Kept { num, .. } => num,
            Claim::Held => None,
        }
    }

    pub fn kind(&self) -> Option<&str> {
        match &self.claim {
            Claim::Kept { kind, .. } => kind.as_deref(),
            Claim::Held => None,
        }
    }

    /// The exit code of an ended job.
    pub fn end_code(&self) -> Option<i32> {
        match self.phase {
            Phase::Ended { code, .. } => Some(code),
            _ => None,
        }
    }

    /// The code once the shell has exited, ended or not.
    pub fn shell_code(&self) -> Option<i32> {
        match self.phase {
            Phase::Ended {
                code,
                shell_told: true,
                ..
            }
            | Phase::Exited { code, .. } => Some(code),
            Phase::Ended { .. } | Phase::Spawning | Phase::Running { .. } => None,
        }
    }

    /// When it ended (the shell's epoch seconds), 0 until then.
    pub fn ended_at(&self) -> u64 {
        match self.phase {
            Phase::Ended { at, .. } => at,
            _ => 0,
        }
    }

    /// Whether the job counts as alive (a job with no group yet does not).
    pub fn alive(&self) -> Alive {
        if self.ended() || self.removed || self.pgid() == 0 {
            Alive::No
        } else if self.kept() {
            Alive::Kept
        } else {
            Alive::Yes
        }
    }
}

/// What happened to a job.
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
    Kill { pgid: i32 },
    /// The job may leave the table.
    Take,
    /// Close the job's stdin; a callback is owed for the change unless the job is removed.
    Ended { notify: bool },
    /// The shell's own exit is observable: a callback is owed.
    Exited,
    /// Drop the job's notifier, close its stdin.
    Release,
    /// `keep` recorded: feed notify's kernel.
    Kept,
}

/// Apply `ev` to `s` and say what to do. Total: an event that does not fit the phase changes
/// nothing. Time arrives in the event.
pub fn step(s: &mut State, ev: Ev) -> Act {
    match ev {
        Ev::Spawned { pgid } => {
            if s.phase == Phase::Spawning {
                s.phase = Phase::Running { pgid };
            }
            Act::Nothing
        }
        Ev::ShellExit { code, signal } => {
            if let Phase::Running { pgid } = s.phase {
                s.phase = Phase::Exited { pgid, code, signal };
                Act::Exited
            } else {
                Act::Nothing
            }
        }
        Ev::GroupEmpty { code, signal, now } => {
            let (pgid, shell_told) = match s.phase {
                Phase::Running { pgid } => (pgid, false),
                Phase::Exited { pgid, .. } => (pgid, true),
                Phase::Ended {
                    pgid, shell_told, ..
                } if cfg!(feature = "mut_end_rewritten") => (pgid, shell_told),
                Phase::Spawning | Phase::Ended { .. } => return Act::Nothing,
            };
            s.phase = Phase::Ended {
                pgid,
                code,
                signal,
                at: now,
                shell_told,
            };
            Act::Ended { notify: !s.removed }
        }
        Ev::Keep { num, kind, seen } => {
            s.claim = Claim::Kept { num, kind };
            s.seen = seen.unwrap_or(s.seen);
            Act::Kept
        }
        Ev::Seen { offset } => {
            s.seen = offset;
            Act::Nothing
        }
        Ev::Taken => {
            s.removed = true;
            Act::Release
        }
        Ev::Push => {
            if s.removed {
                Act::Discard
            } else {
                Act::Store
            }
        }
        Ev::Signal => {
            let running = !s.ended() && s.pgid() != 0;
            if running || cfg!(feature = "mut_signal_ended") && s.pgid() != 0 {
                Act::Signal
            } else {
                Act::Nothing
            }
        }
        Ev::Stop => {
            if s.ended() {
                Act::Nothing
            } else {
                Act::Stop
            }
        }
        Ev::Remove { empty, ended_counts } => {
            let ended = ended_counts && s.ended() && !cfg!(feature = "mut_ended_not_removable");
            if ended || (s.pgid() != 0 && empty) || cfg!(feature = "mut_remove_live") {
                Act::Take
            } else {
                Act::Nothing
            }
        }
        Ev::Disconnect { hold } => {
            let kept = s.kept() && !cfg!(feature = "mut_disconnect_kills_kept");
            if hold && !kept && !s.ended() {
                Act::Kill { pgid: s.pgid() }
            } else {
                Act::Nothing
            }
        }
    }
}

/// Where a read from `offset` starts, for a ring that holds `[base, end)`, and how many bytes it
/// may answer of at most `max`: from the oldest byte held when `offset` is older.
pub fn window(offset: u64, base: u64, end: u64, max: usize) -> (u64, usize) {
    let from = if cfg!(feature = "mut_read_before_base") {
        offset
    } else {
        offset.max(base)
    };
    let want = usize::try_from(end.saturating_sub(from)).unwrap_or(usize::MAX).min(max);
    (from, want)
}

/// The job a stream may go on with: what the table holds under its key, if that is the job it
/// attached to (`same`). A removed job is gone to its streams, and a later job of the same id is
/// not its successor: the stream never reads it, and its client leaving never kills it.
pub fn bound<C>(found: Option<C>, same: bool) -> Option<C> {
    found.filter(|_| same || cfg!(feature = "mut_stream_follows_key"))
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

/// The next frame of a stream that has sent up to `sent`, for a job whose ring holds `[base, end)`.
pub fn follow(s: &State, sent: u64, base: u64, end: u64, shell_told: bool) -> Follow {
    if sent < base {
        Follow::Gap(base - sent)
    } else if sent < end {
        Follow::Data
    } else if let (Some(c), false) = (s.shell_code(), shell_told) {
        Follow::ShellExit(c)
    } else if let Phase::Ended { code, signal, .. } = s.phase
        && !cfg!(feature = "mut_follow_waits_after_end")
    {
        Follow::End(code, signal)
    } else {
        Follow::Wait
    }
}
