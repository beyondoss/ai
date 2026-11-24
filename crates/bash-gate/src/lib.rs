//! The agent's `bash` decisions, as pure functions proven in Verus.
//!
//! * **B1, the running-command limit** ([`admit_start`]): a new command is admitted below
//!   [`MAX_RUNNING_JOBS`] running ones and refused at it. The decision has no third answer: nothing
//!   that runs is stopped to make room ([`lemma_a_start_stops_nothing`]); the table after a start
//!   holds every job it held before.
//! * **B2, the hand-back** ([`yield_reason`]): when a foreground call returns to the model while its
//!   command keeps running as a held job: after [`IDLE_YIELD_MS`] of silence, at the call's ceiling,
//!   or at the model's own `timeout_ms`. Returning is the whole effect: a hand-back kills nothing, and
//!   a call that has waited past its ceiling always returns (progress).
//! * **B5, the working directory** ([`call_cwd`], [`tool_root_source`]): a hosted session's tools are
//!   rooted in the grant's `workspace_root`, and a call that names no `cwd` runs there.
//! * **B4, the shape of an `.mcp.json` entry** ([`mcp_entry_shape`]): a `command` entry is a command
//!   server, which a hosted session runs in its own sandbox.
//! * **B3, a job's end is told once** ([`end_due`]): the session is told that a sandbox job ended
//!   once, across a replica's death. What survives is the transcript: a notice it already holds is
//!   told, whatever the sandbox's directory for the job still says. An ended job whose notice
//!   nobody holds is always due (the retry after a restart tells it).
//!
//! Durations are in nanoseconds; this crate reads no clock.

use vstd::prelude::*;

verus! {

/// The most commands running at once under one `bash` (every kind counts).
pub const MAX_RUNNING_JOBS: usize = 20;

/// The silence after which a foreground call returns to the model and leaves its command running.
pub const IDLE_YIELD_MS: u64 = 60_000;

/// Nanoseconds in a millisecond.
pub const NS_PER_MS: u64 = 1_000_000;

/// What a start does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// The command starts and takes a place in the table.
    Start,
    /// It is refused, listing what runs; the table is as it was.
    Refuse,
}

/// B1. Whether a command may start, given how many are running. The answer is a start or a refusal,
/// never a stop of something running.
pub fn admit_start(running: usize) -> (r: Admit)
    ensures
        (r is Refuse) <==> running >= MAX_RUNNING_JOBS,
        (r is Start) <==> running < MAX_RUNNING_JOBS,
{
    if cfg!(feature = "mut_gate_ignores_limit") {
        Admit::Start
    } else if running >= MAX_RUNNING_JOBS {
        Admit::Refuse
    } else {
        Admit::Start
    }
}

/// The table of running jobs, by id, after a start: with the new id on a start, as it was on a
/// refusal. There is no arm that removes one.
pub open spec fn table_after(table: Set<int>, id: int, a: Admit) -> Set<int> {
    match a {
        Admit::Start => table.insert(id),
        Admit::Refuse => table,
    }
}

/// **B1.** Whatever the count, a start or a refusal leaves every running job running, at twenty or at
/// any other count.
pub proof fn lemma_a_start_stops_nothing(table: Set<int>, id: int, a: Admit)
    ensures
        forall|j: int| #[trigger] table.contains(j) ==> table_after(table, id, a).contains(j),
{
}

/// How a call holds the command it started (or a wait on it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    /// A background start: returns once its settle time has passed.
    Settle(u64),
    /// A foreground call: its `ceiling`, and the model's `timeout_ms` if it gave one.
    Foreground { ceiling: u64, timeout_ms: Option<u64> },
}

/// Why a call returns to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// A background start's settle time passed.
    Settled,
    /// The model's `timeout_ms` passed.
    Timeout,
    /// The command was silent for the idle window.
    Quiet,
    /// The call kept waiting up to its ceiling.
    Ceiling,
}

/// B2. Why a call that has `waited` since it attached, with the command silent for `quiet`, returns to
/// the model now, if it does; `window` is the idle window (`None`: never hand back for silence). It
/// only ever says a call returns: the command is parked as a held job, never stopped.
pub fn yield_reason(hold: Hold, waited: u64, quiet: u64, window: Option<u64>) -> (r: Option<Why>)
    ensures
        match hold {
            Hold::Settle(s) => r == (if waited >= s { Some(Why::Settled) } else { None::<Why> }),
            Hold::Foreground { ceiling, timeout_ms } => match window {
                None => r is None,
                Some(w) => r == (if timeout_ms is Some && waited / NS_PER_MS >= timeout_ms->Some_0 {
                    Some(Why::Timeout)
                } else if quiet >= w {
                    Some(Why::Quiet)
                } else if waited >= ceiling {
                    Some(Why::Ceiling)
                } else {
                    None::<Why>
                }),
            },
        },
{
    match hold {
        Hold::Settle(s) => if waited >= s { Some(Why::Settled) } else { None },
        Hold::Foreground { ceiling, timeout_ms } => match window {
            None => None,
            Some(w) => {
                let timed_out = match timeout_ms {
                    Some(ms) => waited / NS_PER_MS >= ms,
                    None => false,
                };
                if timed_out {
                    Some(Why::Timeout)
                } else if quiet >= w && !cfg!(feature = "mut_quiet_never_hands_back") {
                    Some(Why::Quiet)
                } else if waited >= ceiling {
                    Some(Why::Ceiling)
                } else {
                    None
                }
            },
        },
    }
}

/// **B2, the default window.** With the 60 s window a command silent for 60 s is handed back (unless
/// the model's `timeout_ms` came first, which also returns): the call returns, whichever applies.
pub proof fn lemma_sixty_seconds_hands_back(hold: Hold, waited: u64, quiet: u64)
    requires
        hold is Foreground,
        quiet as int >= IDLE_YIELD_MS as int * NS_PER_MS as int,
    ensures
        ({
            let w = Some((IDLE_YIELD_MS * NS_PER_MS) as u64);
            let timeout = match hold {
                Hold::Foreground { ceiling, timeout_ms } => timeout_ms is Some && waited / NS_PER_MS >= timeout_ms->Some_0,
                _ => false,
            };
            quiet >= w->Some_0 && (timeout || quiet >= w->Some_0)
        }),
{
}

/// **B2, progress.** A foreground call with a window never waits past its ceiling: once `waited`
/// has reached it, the call returns. (The partner of every refusal to return: it clears.)
pub proof fn lemma_a_call_returns_by_its_ceiling(ceiling: u64, timeout_ms: Option<u64>, waited: u64, quiet: u64, w: u64)
    requires
        waited >= ceiling,
    ensures
        ({
            let r = if timeout_ms is Some && waited / NS_PER_MS >= timeout_ms->Some_0 {
                Some(Why::Timeout)
            } else if quiet >= w {
                Some(Why::Quiet)
            } else if waited >= ceiling {
                Some(Why::Ceiling)
            } else {
                None::<Why>
            };
            r is Some
        }),
{
}

/// Where a call runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cwd {
    /// The model named a `cwd`: it is resolved against the tool's root.
    Named,
    /// The call named none: it runs in the tool's root.
    Root,
    /// The call named none and the tool has no root: it inherits the process's cwd.
    Inherit,
}

/// B5. Where a call runs, given whether the model named a `cwd` and whether the tool has a root.
/// A call that names none runs in the root whenever there is one, and inherits only without one.
pub fn call_cwd(named: bool, has_root: bool) -> (r: Cwd)
    ensures
        named ==> r is Named,
        (!named && has_root) ==> r is Root,
        (!named && !has_root) ==> r is Inherit,
{
    if named {
        Cwd::Named
    } else if has_root && !cfg!(feature = "mut_cwd_ignores_root") {
        Cwd::Root
    } else {
        Cwd::Inherit
    }
}

/// Where a session's tools are rooted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootSource {
    /// The grant's `workspace_root`, a path inside the session's sandbox.
    Workspace,
    /// No root: the process's own cwd.
    Process,
}

/// B5. A hosted (service) session's tools are rooted in the grant's `workspace_root`; only a local
/// or replica process is rooted in its own cwd. With `call_cwd`, a hosted call that names no `cwd`
/// runs in the workspace root (the root is never empty there: a grant with a relative or empty
/// `workspace_root` is refused at open, `service.rs`).
pub fn tool_root_source(service: bool) -> (r: RootSource)
    ensures
        service ==> r is Workspace,
        !service ==> r is Process,
{
    if service { RootSource::Workspace } else { RootSource::Process }
}

/// The `"type"` an `.mcp.json` entry names, reduced to what decides its shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McpType {
    /// No `"type"`.
    Absent,
    /// `stdio`.
    Stdio,
    /// `http`, `streamable-http`, `streamableHttp` or `sse`.
    Http,
    /// Any other value.
    Other,
}

/// What an `.mcp.json` entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McpShape {
    /// A `command` server: a hosted session runs it as a job in its own sandbox, never on the replica.
    Command,
    /// A `url` server, dialed over HTTP.
    Url,
    /// Neither: skipped with a warning.
    Refused,
}

/// B4. An `.mcp.json` entry's shape, given its `"type"` and whether it carries a `command` and a
/// `url`. An entry with a `command` and no `"type"` (or `stdio`) is a command server, whatever else
/// it carries; a `url` entry with no `command` (or an http `"type"`) is a url server; a `stdio` entry
/// without a `command`, an unknown `"type"`, and an entry with neither are refused.
pub fn mcp_entry_shape(kind: McpType, has_command: bool, has_url: bool) -> (r: McpShape)
    ensures
        (kind is Absent || kind is Stdio) && has_command ==> r is Command,
        (kind is Absent || kind is Http) && !has_command && has_url ==> r is Url,
        kind is Http && has_url ==> r is Url,
        kind is Other ==> r is Refused,
        !has_command && !has_url ==> r is Refused,
        kind is Stdio && !has_command ==> r is Refused,
{
    match kind {
        McpType::Other => McpShape::Refused,
        McpType::Stdio => {
            if has_command && !cfg!(feature = "mut_command_entry_refused") {
                McpShape::Command
            } else {
                McpShape::Refused
            }
        },
        McpType::Http => {
            if has_url { McpShape::Url } else { McpShape::Refused }
        },
        McpType::Absent => {
            if has_command && !cfg!(feature = "mut_command_entry_refused") {
                McpShape::Command
            } else if has_url {
                McpShape::Url
            } else {
                McpShape::Refused
            }
        },
    }
}

/// B3. Whether a job's end notice is due. `told_live` is this process having delivered it;
/// `told_persisted` is the session's persisted transcript already holding it (a replica that died
/// between persisting the report and deleting the job's directory left the job behind, ended). The
/// iff is both halves: a held notice is never due again, and an ended job nobody holds always is.
pub fn end_due(ended: bool, told_live: bool, told_persisted: bool) -> (r: bool)
    ensures
        r <==> (ended && !told_live && !told_persisted),
{
    if cfg!(feature = "mut_told_ignores_transcript") {
        ended && !told_live
    } else {
        ended && !told_live && !told_persisted
    }
}

} // verus!

#[cfg(test)]
mod tests {
    use super::*;

    /// B1: below the limit a start is admitted, at it and over it refused. Red under
    /// `mut_gate_ignores_limit`.
    #[test]
    fn the_limit_gates_a_start() {
        assert_eq!(admit_start(0), Admit::Start);
        assert_eq!(admit_start(MAX_RUNNING_JOBS - 1), Admit::Start);
        assert_eq!(admit_start(MAX_RUNNING_JOBS), Admit::Refuse);
        assert_eq!(admit_start(MAX_RUNNING_JOBS + 5), Admit::Refuse);
    }

    /// B5: a call that names no cwd runs in the root, and a hosted session is rooted in the workspace.
    /// Red under `mut_cwd_ignores_root`.
    #[test]
    fn a_call_without_cwd_runs_in_the_root() {
        assert_eq!(call_cwd(false, true), Cwd::Root);
        assert_eq!(call_cwd(false, false), Cwd::Inherit);
        assert_eq!(call_cwd(true, true), Cwd::Named);
        assert_eq!(tool_root_source(true), RootSource::Workspace);
        assert_eq!(tool_root_source(false), RootSource::Process);
    }

    /// B4: a command entry is a command server. Red under `mut_command_entry_refused`.
    #[test]
    fn a_command_entry_is_a_command_server() {
        assert_eq!(
            mcp_entry_shape(McpType::Absent, true, false),
            McpShape::Command
        );
        assert_eq!(
            mcp_entry_shape(McpType::Stdio, true, true),
            McpShape::Command
        );
        assert_eq!(mcp_entry_shape(McpType::Absent, false, true), McpShape::Url);
        assert_eq!(mcp_entry_shape(McpType::Http, true, true), McpShape::Url);
        assert_eq!(
            mcp_entry_shape(McpType::Other, true, true),
            McpShape::Refused
        );
        assert_eq!(
            mcp_entry_shape(McpType::Stdio, false, true),
            McpShape::Refused
        );
    }

    /// B3: a notice the transcript holds is not due again; an ended job it does not hold is. Red
    /// under `mut_told_ignores_transcript`.
    #[test]
    fn a_persisted_end_is_not_told_again() {
        assert!(!end_due(true, false, true));
        assert!(end_due(true, false, false));
        assert!(!end_due(true, true, false));
        assert!(!end_due(false, false, false));
    }

    /// B2: 60 s of silence hands a foreground call back; 59 s does not. Red under
    /// `mut_quiet_never_hands_back`.
    #[test]
    fn sixty_quiet_seconds_hand_a_call_back() {
        let w = Some(IDLE_YIELD_MS * NS_PER_MS);
        let hold = Hold::Foreground {
            ceiling: 600_000 * NS_PER_MS,
            timeout_ms: None,
        };
        assert_eq!(
            yield_reason(hold, 59_000 * NS_PER_MS, 59_000 * NS_PER_MS, w),
            None
        );
        assert_eq!(
            yield_reason(hold, 60_000 * NS_PER_MS, 60_000 * NS_PER_MS, w),
            Some(Why::Quiet)
        );
        assert_eq!(
            yield_reason(hold, 600_000 * NS_PER_MS, 1, w),
            Some(Why::Ceiling)
        );
        assert_eq!(yield_reason(hold, 1, 1, None), None);
    }
}
