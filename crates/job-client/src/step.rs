//! The job handle's decisions: attach, reconnect, exit and remove, as one pure function, [`step`].
//!
//! A handle holds a job: it starts one (or attaches to one it already knows), follows its output across
//! cuts, delivers the exit once, and removes the job when the caller asks it to. Everything it decides is
//! here. The shell around it (TypeScript through WASM, Python through a native library, the Rust SDK) does
//! I/O only: it makes the request the kernel names, reads the answer, turns it into an [`Event`], and does
//! what the [`Action`]s say. No SDK keeps a reconnect counter, a status table, an offset sum or a silence
//! constant of its own.
//!
//! Time and randomness are inputs: [`step`] takes the clock and a fresh uniform draw, and returns the time
//! to be woken at. A job's output bytes never cross the boundary; the shell keeps them and the kernel
//! returns the length to hand over. Everything here is an integer, a fieldless enum or a plain record, so
//! the same function builds as an rlib, a WASM module and a native library.
//!
//! See the README for the properties, each of which Verus proves on [`step`] with no bound and each of
//! which has a `mut_*` mutant that turns the proof red.

// The code inside `verus!` builds its action lists by `push`, the shape Verus proves.
#![allow(clippy::vec_init_then_push)]

use vstd::prelude::*;

use crate::all_ok;

verus! {

// ---------------------------------------------------------------------------------------------
// Constants.
// ---------------------------------------------------------------------------------------------

/// Failed opens in a row (no confirmed reattach between them) after which every further failed open also
/// tells the shell `Reconnecting{attempts, since_ms}`. Reconnecting never ends the handle: it is a status.
/// The first wait is 250 ms and the longest 30 s, so this is about three minutes.
pub const RECONNECTING_AFTER: u32 = 12;

/// The longest wait before a reopen, whatever the server's `Retry-After` says.
pub const BACKOFF_CAP_MS: u64 = 30_000;

/// The shortest silence limit: six missed default heartbeats.
pub const SILENCE_FLOOR_MS: u64 = 30_000;

// ---------------------------------------------------------------------------------------------
// Types.
// ---------------------------------------------------------------------------------------------

/// A job incarnation's mark: 1 to 32 bytes of `[0-9a-f]`, zero-padded to 32. Length 0 is "no mark held".
/// Two marks are the same incarnation exactly when the records are equal; [`Mark::from_bytes`] is the one
/// way the shell makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mark {
    pub len: u8,
    pub b: [u8; 32],
}

impl Mark {
    /// The bytes of the mark.
    pub open spec fn seq(self) -> Seq<u8> {
        self.b@.subrange(0, self.len as int)
    }

    /// No mark held.
    pub fn none() -> (r: Mark)
        ensures
            r.len == 0,
    {
        Mark { len: 0, b: [0u8; 32] }
    }

    /// The mark spelled by `bytes`: `Some` exactly when they are 1 to 32 bytes of `[0-9a-f]`.
    pub fn from_bytes(bytes: &[u8]) -> (r: Option<Mark>)
        ensures
            (r is Some) == (1 <= bytes.len() <= 32 && forall|j: int| 0 <= j < bytes.len() ==> crate::kind_ok(2, #[trigger] bytes[j])),
            r is Some ==> r->Some_0.seq() == bytes@ && r->Some_0.len == bytes.len(),
    {
        if bytes.len() < 1 || bytes.len() > 32 || !all_ok(bytes, 0, bytes.len(), 2) {
            return None;
        }
        let mut b = [0u8; 32];
        let mut i: usize = 0;
        while i < bytes.len()
            invariant
                i <= bytes.len() <= 32,
                forall|j: int| 0 <= j < i ==> b[j] == bytes[j],
            decreases bytes.len() - i,
        {
            b[i] = bytes[i];
            i += 1;
        }
        let m = Mark { len: bytes.len() as u8, b };
        proof {
            assert(m.seq() =~= bytes@);
        }
        Some(m)
    }
}

/// Whether two marks name the same incarnation.
pub fn mark_eq(a: &Mark, b: &Mark) -> (r: bool)
    ensures
        r == (*a == *b),
{
    if a.len != b.len {
        return false;
    }
    let mut i: usize = 0;
    while i < 32
        invariant
            i <= 32,
            forall|j: int| 0 <= j < i ==> a.b[j] == b.b[j],
        decreases 32 - i,
    {
        if a.b[i] != b.b[i] {
            return false;
        }
        i += 1;
    }
    proof {
        assert(a.b =~= b.b);
    }
    true
}

/// Where the handle is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Built, nothing sent: waiting for [`Event::Go`].
    Idle,
    /// A start is sent and its answer has not come.
    Starting,
    /// An attach is sent and its answer has not come.
    Opening,
    /// The stream is open (2xx) and the echo of the mark has not come.
    Confirming,
    /// The echo came: output flows.
    Streaming,
    /// Waiting out a failed open or a cut before sending the request again.
    Backoff,
    /// The exit is known and handed to the consumer's side; waiting for the consumer to take it.
    ExitHeld,
    /// The remove is sent and its answer has not come.
    Removing,
    /// Finished: one terminal action was issued.
    Done,
}

/// What the handle sends next when it leaves [`Phase::Backoff`], and what a Confirming echo must be.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flow {
    /// A start: its echo is `started{mark}`.
    Start,
    /// An attach: its echo is `attached{mark, from}`.
    Attach,
    /// A remove, after the exit was taken.
    Remove,
}

/// Whether output is delivered to the consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Output and gaps are delivered.
    Follow,
    /// Output is counted, not delivered (`wait`): only the exit is.
    Watch,
}

/// A typed failure, the one vocabulary every SDK maps to its own error class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fault {
    /// The job is gone, or the id now names another incarnation.
    JobGone,
    /// A start of an id that exists under another key.
    JobExists,
    /// A start refused for a full disk.
    DiskFull,
    /// A start refused because no pseudo-terminal is free.
    NoPty,
    /// The caller's deadline passed (the shell sent [`Event::Deadline`]). The `Fail` action carries the
    /// cursor offset to resume from.
    Deadline,
    /// The endpoint predates marks: it did not echo the mark.
    NotOffered,
    /// A status or frame that is wrong and will not be right on a retry; the status is in the action.
    Protocol,
    /// The credential is refused.
    Auth,
    /// A start that nothing ever answered: the connection was refused or the name did not resolve on the
    /// first try, so no request of it reached any server.
    Transport,
    /// The sandbox no longer exists, and the answer that said so was the backend's ([`Authority::SandboxGone`]).
    /// The only way an unreachable endpoint ends a handle.
    SandboxGone,
    /// The session has ended, and the front said so ([`Authority::SessionEnded`]).
    SessionEnded,
}

/// What a request is, for classifying its answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Start,
    Attach,
    Remove,
    /// Any other op addressed to one job by its id (`keep`, `seen`, `kill`, a stdin `write`, `read`): no handle,
    /// one request, its answer classified by the same table.
    Other,
}

/// What an HTTP answer means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// Go on.
    Ok,
    /// Try again after a wait.
    Retry,
    /// Stop with this fault.
    Terminal(Fault),
}

/// The handle's state: a fixed record of integers, so a shell passes it by value and the kernel keeps
/// nothing between calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub phase: Phase,
    pub flow: Flow,
    pub mode: Mode,
    /// Issue a remove once the consumer has taken the exit.
    pub remove_on_exit: bool,
    /// The idempotency key's handle (the shell holds the key itself). Never changes.
    pub key: u64,
    /// The job is started with its stdin open: the `started` echo must say it was honoured. Never changes.
    pub stdin_open: bool,
    /// The job is started on a terminal: the `started` echo must say so. Never changes.
    pub pty_open: bool,
    /// The start is a kept job's (`hold: false`): no stream's close ends it, so a refused echo stops it.
    pub kept: bool,
    /// The consumer has been told the echo (`Started` or the first `Attached`): a reattach's is not told again.
    pub announced: bool,
    /// A request of this handle may have reached a server: it was answered, or its answer was lost. Until
    /// then a start that cannot connect ends with `Transport`; after it, only the job's end or a terminal
    /// answer does.
    pub reached: bool,
    /// The incarnation this handle holds; length 0 until the echo names it (a start, or a `get`).
    pub mark: Mark,
    /// The next output byte the consumer has not been given.
    pub offset: u64,
    /// A gap the stream reported that the consumer has not been told of yet: the next chunk carries it.
    /// It is per connection: a reattach asks from `offset` and the server reports the gap again.
    pub pending: u64,
    /// Failed opens since the last confirmed reattach (saturating).
    pub misses: u32,
    /// The clock when the first of those failed.
    pub since_ms: u64,
    /// When a [`Phase::Backoff`] ends, milliseconds on the shell's clock.
    pub until_ms: u64,
    /// The heartbeat the request chose, whole seconds; 0 for the default.
    pub beat_s: u32,
}

/// What the stream's frames said, already decoded by the shell's codec. Payload bytes stay in the shell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Frame {
    /// `{"attached": {"mark", "from"}}`: the first frame of an attach.
    Attached { mark: Mark, from: u64 },
    /// `{"started": {"mark"}}`: the first frame of a start.
    /// `stdin` is whether the endpoint says the job's stdin was opened.
    /// `pty` is whether it says a terminal was opened.
    Started { mark: Mark, stdin: bool, pty: bool },
    /// An output frame of `len` decoded bytes.
    Output { len: u64 },
    /// `{"gap": n}`: `n` bytes are no longer kept.
    Gap { n: u64 },
    /// A heartbeat.
    Heartbeat,
    /// The job's shell exited with `code`; the job is not over (its group runs on). Not final.
    ShellExit { code: i32 },
    /// The final frame.
    Exit { code: i32, signal: i32 },
    /// An error frame, an unparseable line, a frame out of range.
    Bad,
}

/// Whether a refusal is the backend's own authoritative answer about the sandbox or the session, as
/// [`authority`] decides from what the shell parsed out of the answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Authority {
    /// Anything else: a busy front, a proxy's 502, a refused credential. The handle waits or the class
    /// table decides.
    NotAuthoritative,
    /// The backend says the sandbox does not exist (E2B's `502` naming ours).
    SandboxGone,
    /// The front says the session has ended (Beyond's `401` with code `session_ended`).
    SessionEnded,
}

/// What the shell observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// Send the first request.
    Go,
    /// The request was answered with this status (and `Retry-After` in whole seconds, 0 if none).
    /// `authority`: what [`authority`] made of the answer. Silence, a refused connect and a timeout are never
    /// authoritative: they are `Broke`.
    Answered { status: u16, retry_after_s: u32, authority: Authority },
    /// A frame arrived.
    Frame(Frame),
    /// The stream broke: reset, silence for the limit, EOF or a truncated line before the final frame.
    Broke,
    /// The request never reached a server: the connection was refused, the host was unreachable or its name
    /// did not resolve. Nothing was sent, so nothing was started.
    Unreached,
    /// A sleep the kernel asked for ended.
    Tick,
    /// The consumer received the exit.
    ExitTaken,
    /// The consumer stopped: detach, never kill.
    Close,
    /// The caller's deadline passed. The shell owns the clock for it; the kernel only obeys.
    Deadline,
}

/// What the shell does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Send the start with the idempotency key `key`; presume the stream dead after `silence_ms` of nothing.
    Start { key: u64, silence_ms: u64 },
    /// Attach at `offset`, asking for `mark` (length 0: whatever holds the id now).
    OpenAttach { offset: u64, mark: Mark, silence_ms: u64 },
    /// Release the connection, if one is open.
    CloseStream,
    /// Tell the consumer the job started (the `started` echo was accepted): `mark` names the incarnation.
    Started { mark: Mark, stdin: bool, pty: bool },
    /// Tell the consumer, once, where an attach began: the first accepted `attached` echo.
    Attached { mark: Mark, from: u64 },
    /// Tell the consumer the job's shell exited with `code` (not final).
    ShellExit { code: i32 },
    /// Stop the job and remove it: a kept start whose echo was refused leaves a job nothing can name.
    Kill,
    /// Status, not an end: `attempts` failed opens in a row since `since_ms`, and the handle keeps trying.
    Reconnecting { attempts: u32, since_ms: u64 },
    /// Hand the consumer `len` bytes (the frame just decoded) at offset `at`, with `dropped` bytes lost
    /// just before them, at `[at - dropped, at)`.
    Deliver { at: u64, len: u64, dropped: u64 },
    /// Hand the consumer the exit, with the `dropped` bytes of output lost just before it.
    DeliverExit { code: i32, signal: i32, dropped: u64, killed: bool },
    /// Feed [`Event::Tick`] at this time.
    Sleep { until_ms: u64 },
    /// Send the remove, scoped to `mark`.
    Remove { mark: Mark },
    /// Finish with a typed failure; `offset` is the cursor to resume from; `status` is the HTTP status
    /// that caused it, or 0.
    Fail { fault: Fault, status: u16, offset: u64 },
    /// Finish: the exit was delivered (and the job removed, if asked).
    Finish,
    /// Finish: the consumer stopped.
    Detached,
}

// ---------------------------------------------------------------------------------------------
// Specs: what the code promises.
// ---------------------------------------------------------------------------------------------

/// The spec's silence limit: `max(30 s, 2 beats + 10 s)`.
pub open spec fn silence_spec(beat_s: u32) -> int {
    if 2000 * (beat_s as int) + 10000 > 30000 { 2000 * (beat_s as int) + 10000 } else { 30000 }
}

pub open spec fn min_int(a: int, b: int) -> int {
    if a < b { a } else { b }
}

/// The ranking of a phase for the progress lemma: every fair event takes a phase to a lower one.
pub open spec fn rank_of(p: Phase) -> int {
    match p {
        Phase::Idle => 5,
        Phase::Backoff => 4,
        Phase::Starting => 3,
        Phase::Opening => 3,
        Phase::Confirming => 2,
        Phase::ExitHeld => 2,
        Phase::Removing => 1,
        Phase::Streaming => 0,
        Phase::Done => 0,
    }
}

/// The invariant of every state this module makes and keeps.
pub open spec fn wf(s: State) -> bool {
    &&& s.mark.len <= 32
    &&& (s.phase is Streaming || s.phase is ExitHeld || s.phase is Removing || s.flow is Remove)
        ==> s.mark.len > 0
    &&& s.flow is Start ==> s.mark.len == 0
    &&& s.pending > 0 ==> s.phase is Streaming
    &&& s.phase is Starting ==> s.flow is Start
    &&& s.phase is Opening ==> s.flow is Attach
    &&& s.phase is Removing ==> s.flow is Remove
    &&& (s.phase is Streaming || s.phase is ExitHeld) ==> s.flow is Attach
    &&& s.phase is Confirming ==> !(s.flow is Remove)
    &&& s.phase is Idle ==> !(s.flow is Remove)
}

/// The events the shell builds are well formed: a mark in a frame is at most 32 bytes.
pub open spec fn wf_ev(ev: Event) -> bool {
    echo_mark(ev).len <= 32
}

/// Which request a phase has in flight.
pub open spec fn op_of(p: Phase) -> Op {
    match p {
        Phase::Starting => Op::Start,
        Phase::Removing => Op::Remove,
        _ => Op::Attach,
    }
}

/// The table of answers, as a spec.
pub open spec fn class_of(op: Op, status: int, auth: Authority) -> Class {
    if auth is SandboxGone {
        Class::Terminal(Fault::SandboxGone)
    } else if auth is SessionEnded {
        Class::Terminal(Fault::SessionEnded)
    } else if 200 <= status && status <= 299 {
        Class::Ok
    } else if op is Remove && status == 404 {
        Class::Ok
    } else if status == 429 || status == 502 || status == 504 {
        Class::Retry
    } else if status == 503 {
        if op is Start { Class::Terminal(Fault::NoPty) } else { Class::Retry }
    } else if status == 401 || status == 403 {
        Class::Terminal(Fault::Auth)
    } else if status == 404 && (op is Attach || op is Other) {
        Class::Terminal(Fault::JobGone)
    } else if status == 409 && op is Start {
        Class::Terminal(Fault::JobExists)
    } else if status == 507 && op is Start {
        Class::Terminal(Fault::DiskFull)
    } else {
        Class::Terminal(Fault::Protocol)
    }
}

pub open spec fn ev_status(ev: Event) -> int {
    match ev {
        Event::Answered { status, .. } => status as int,
        _ => -1,
    }
}

pub open spec fn ev_authority(ev: Event) -> Authority {
    match ev {
        Event::Answered { authority, .. } => authority,
        _ => Authority::NotAuthoritative,
    }
}

pub open spec fn ev_retry_after_ms(ev: Event) -> int {
    match ev {
        Event::Answered { retry_after_s, .. } => 1000 * (retry_after_s as int),
        _ => 0,
    }
}

pub open spec fn is_echo(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Attached { .. }) => true,
        Event::Frame(Frame::Started { .. }) => true,
        _ => false,
    }
}

/// Some mark of length 0.
pub open spec fn no_mark() -> Mark {
    choose|m: Mark| m.len == 0
}

pub open spec fn echo_mark(ev: Event) -> Mark {
    match ev {
        Event::Frame(Frame::Attached { mark, .. }) => mark,
        Event::Frame(Frame::Started { mark, .. }) => mark,
        _ => no_mark(),
    }
}

pub open spec fn echo_from(ev: Event) -> int {
    match ev {
        Event::Frame(Frame::Attached { from, .. }) => from as int,
        _ => 0,
    }
}

pub open spec fn is_output(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Output { .. }) => true,
        _ => false,
    }
}

pub open spec fn output_len(ev: Event) -> int {
    match ev {
        Event::Frame(Frame::Output { len }) => len as int,
        _ => 0,
    }
}

pub open spec fn gap_len(ev: Event) -> int {
    match ev {
        Event::Frame(Frame::Gap { n }) => n as int,
        _ => 0,
    }
}

pub open spec fn is_gap(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Gap { .. }) => true,
        _ => false,
    }
}

pub open spec fn is_exit_frame(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Exit { .. }) => true,
        _ => false,
    }
}

pub open spec fn exit_frame_ok(ev: Event, out: Seq<Action>, dropped: int) -> bool {
    match ev {
        Event::Frame(Frame::Exit { code, signal }) => out.len() >= 2
            && out[1] == (Action::DeliverExit { code, signal, dropped: dropped as u64, killed: signal != 0 }),
        _ => false,
    }
}

pub open spec fn fails_with(out: Seq<Action>, f: Fault) -> bool {
    out.len() > 0 && is_fail_with(out[out.len() - 1], f)
}

pub open spec fn is_fail_with(a: Action, f: Fault) -> bool {
    match a {
        Action::Fail { fault, .. } => fault == f,
        _ => false,
    }
}

pub open spec fn is_fail(a: Action) -> bool {
    match a {
        Action::Fail { .. } => true,
        _ => false,
    }
}

pub open spec fn is_terminal_action(a: Action) -> bool {
    match a {
        Action::Fail { .. } => true,
        Action::Finish => true,
        Action::Detached => true,
        _ => false,
    }
}

pub open spec fn is_open_action(a: Action) -> bool {
    match a {
        Action::Start { .. } => true,
        Action::OpenAttach { .. } => true,
        _ => false,
    }
}

pub open spec fn is_delivery(a: Action) -> bool {
    match a {
        Action::Deliver { .. } => true,
        Action::DeliverExit { .. } => true,
        _ => false,
    }
}

/// Nothing the handle does once the exit is known reopens the stream.
pub open spec fn late(s: State) -> bool {
    s.phase is ExitHeld || s.phase is Removing || (s.phase is Backoff && s.flow is Remove)
}

/// The echo that confirms a reattach, as the server's fair self sends it.
pub open spec fn good_echo(s: State, ev: Event) -> bool {
    &&& is_echo(ev)
    &&& echo_mark(ev).len > 0
    &&& (s.mark.len == 0 || echo_mark(ev) == s.mark)
    &&& (s.flow is Start) == (ev is Frame && ev->Frame_0 is Started)
    &&& (s.flow is Attach) == (ev is Frame && ev->Frame_0 is Attached)
    &&& echo_from(ev) >= s.offset as int
    &&& (s.flow is Start && s.stdin_open) ==> echo_stdin(ev)
    &&& (s.flow is Start && s.pty_open) ==> echo_pty(ev)
}

/// Fairness: the server is reachable and answers as it should. A step under it is a fair step.
pub open spec fn fair(s: State, ev: Event, now: u64) -> bool {
    match s.phase {
        Phase::Idle => ev is Go,
        Phase::Backoff => ev is Tick && now >= s.until_ms,
        Phase::Starting => 200 <= ev_status(ev) && ev_status(ev) <= 299 && ev_authority(ev) is NotAuthoritative,
        Phase::Opening => 200 <= ev_status(ev) && ev_status(ev) <= 299 && ev_authority(ev) is NotAuthoritative,
        Phase::Confirming => good_echo(s, ev),
        Phase::Removing => 200 <= ev_status(ev) && ev_status(ev) <= 299 && ev_authority(ev) is NotAuthoritative,
        Phase::ExitHeld => ev is ExitTaken,
        _ => false,
    }
}

// --- the properties -------------------------------------------------------------------------

/// O: offsets. A handle gives each output offset to the consumer at most once, in order, with no hole:
/// every delivery starts where the last ended, and the position never moves back.
pub open spec fn post_offsets(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& s2.offset >= s.offset
    &&& !(s.phase is Streaming) ==> s2.offset == s.offset
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Deliver { at, len, dropped } => {
            &&& s.phase is Streaming
            &&& s.mode is Follow
            &&& dropped == s.pending
            &&& at as int == s.offset as int + s.pending as int
            &&& s2.offset as int == at as int + len as int
        },
        Action::DeliverExit { dropped, .. } => {
            &&& s.phase is Streaming
            &&& s2.offset as int == s.offset as int + s.pending as int
        },
        Action::OpenAttach { offset, mark, .. } => offset == s.offset && mark == s.mark,
        _ => true,
    }
    &&& (s.phase is Streaming && is_output(ev) && s.offset as int + s.pending as int + output_len(ev)
        <= u64::MAX as int) ==> {
        &&& s2.phase is Streaming
        &&& s2.offset as int == s.offset as int + s.pending as int + output_len(ev)
        &&& s2.pending == 0
        &&& (s.mode is Follow ==> out == seq![
            Action::Deliver {
                at: (s.offset + s.pending) as u64,
                len: output_len(ev) as u64,
                dropped: s.pending,
            },
        ])
    }
}

/// G: gaps. A gap is folded into the chunk after it, and one with no chunk after it before the end is
/// delivered with the exit: no gap is told twice and none is dropped.
pub open spec fn post_gaps(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& (s.phase is Streaming && is_gap(ev) && s.pending as int + gap_len(ev) <= u64::MAX as int) ==> {
        &&& s2.phase is Streaming
        &&& s2.pending as int == s.pending as int + gap_len(ev)
        &&& s2.offset == s.offset
        &&& out.len() == 0
    }
    &&& forall|i: int| 0 <= i < out.len() ==> (is_delivery(#[trigger] out[i]) ==> s2.pending == 0)
    &&& (s.phase is Streaming && is_exit_frame(ev) && s.offset as int + s.pending as int <= u64::MAX as int) ==> {
        &&& s2.phase is ExitHeld
        &&& s2.offset as int == s.offset as int + s.pending as int
        &&& exit_frame_ok(ev, out, if s.mode is Follow { s.pending as int } else { 0 })
    }
}

/// B: the reconnect budget resets on a confirmed reattach only (the echo with the held mark), never on a
/// bare connect; after `RECONNECTING_AFTER` it says `Reconnecting` and keeps going.
pub open spec fn post_budget(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& wf(s2)
    &&& s2.misses < s.misses ==> {
        &&& s.phase is Confirming
        &&& is_echo(ev)
        &&& s2.phase is Streaming
    }
    &&& (s.phase is Confirming && good_echo(s, ev)) ==> s2.misses == 0 && s2.phase is Streaming
    &&& (s2.misses > s.misses) ==> s2.misses as int == min_int(s.misses as int + 1, u32::MAX as int)
    &&& ((s.phase is Starting || s.phase is Opening || s.phase is Confirming || s.phase is Streaming
        || s.phase is Removing) && ev is Broke) ==> {
        &&& s2.phase is Backoff
        &&& s2.misses as int == min_int(s.misses as int + 1, u32::MAX as int)
    }
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Reconnecting { attempts, since_ms } => {
            &&& attempts >= RECONNECTING_AFTER
            &&& attempts == s2.misses
            &&& since_ms == s2.since_ms
        },
        _ => true,
    }
    &&& (s2.phase is Backoff && s2.misses > s.misses && s2.misses >= RECONNECTING_AFTER) ==> {
        &&& out.len() > 0
        &&& out[out.len() - 1] == (Action::Reconnecting { attempts: s2.misses, since_ms: s2.since_ms })
    }
}

/// M: incarnation. Output is never delivered under a mark the endpoint did not echo; an echo that is not
/// the held mark is `JobGone`; no echo is `NotOffered`.
pub open spec fn post_incarnation(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& (s2.phase is Streaming && !(s.phase is Streaming)) ==> {
        &&& s.phase is Confirming
        &&& is_echo(ev)
        &&& echo_mark(ev).len > 0
        &&& (s.mark.len == 0 || echo_mark(ev) == s.mark)
        &&& (s.flow is Start ==> ev is Frame && ev->Frame_0 is Started)
        &&& (s.flow is Attach ==> ev is Frame && ev->Frame_0 is Attached)
        &&& s2.mark == echo_mark(ev)
    }
    &&& (s.phase is Streaming && s2.phase is Streaming) ==> s2.mark == s.mark
    &&& ((s.phase is Confirming || s.phase is Streaming) && is_echo(ev) && echo_mark(ev).len > 0
        && s.mark.len > 0 && echo_mark(ev) != s.mark) ==> {
        &&& s2.phase is Done
        &&& fails_with(out, Fault::JobGone)
    }
    &&& (s.phase is Confirming && is_echo(ev) && echo_mark(ev).len == 0) ==> fails_with(out, Fault::NotOffered)
    &&& (s.phase is Confirming && ev is Frame && !is_echo(ev) && !(ev->Frame_0 is Bad)) ==> {
        &&& s2.phase is Done
        &&& fails_with(out, Fault::NotOffered)
    }
}

pub open spec fn echo_stdin(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Started { stdin, .. }) => stdin,
        _ => false,
    }
}

pub open spec fn echo_pty(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Started { pty, .. }) => pty,
        _ => false,
    }
}

pub open spec fn is_started_frame(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Started { .. }) => true,
        _ => false,
    }
}

pub open spec fn is_attached_frame(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::Attached { .. }) => true,
        _ => false,
    }
}

pub open spec fn is_shell_exit_frame(ev: Event) -> bool {
    match ev {
        Event::Frame(Frame::ShellExit { .. }) => true,
        _ => false,
    }
}

/// U: what the consumer is told. The accepted `started` echo is told, with whether the job's stdin was
/// opened, and a start that asked for stdin is refused (`NotOffered`) when the echo says it was not; the
/// first accepted `attached` echo is told, once, and a reattach's never; the shell's exit is told when it
/// arrives and changes nothing else. Each telling comes from exactly the step it names.
pub open spec fn post_surface(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Started { mark, stdin, pty } => {
            &&& s.phase is Confirming && s.flow is Start
            &&& ev == Event::Frame(Frame::Started { mark, stdin, pty })
            &&& s2.phase is Streaming
            &&& s.stdin_open ==> stdin
            &&& s.pty_open ==> pty
        },
        Action::Attached { mark, from } => {
            &&& s.phase is Confirming && s.flow is Attach && !s.announced
            &&& ev == Event::Frame(Frame::Attached { mark, from })
            &&& s2.phase is Streaming
        },
        Action::ShellExit { code } => {
            &&& s.phase is Streaming
            &&& ev == Event::Frame(Frame::ShellExit { code })
        },
        Action::Kill => {
            &&& s.kept && s.flow is Start && s.phase is Confirming
            &&& fails_with(out, Fault::NotOffered)
        },
        _ => true,
    }
    &&& (s.phase is Confirming && s.flow is Start && s2.phase is Streaming) ==> (out.len() == 1 && out[0] is Started)
    &&& (s.phase is Confirming && s.flow is Attach && s2.phase is Streaming) ==> {
        &&& s.announced ==> out.len() == 0
        &&& !s.announced ==> (out.len() == 1 && out[0] is Attached)
    }
    &&& (s.phase is Confirming && s.flow is Start && s.stdin_open && is_started_frame(ev) && !echo_stdin(ev)
        && echo_mark(ev).len > 0 && (s.mark.len == 0 || echo_mark(ev) == s.mark)) ==> fails_with(out, Fault::NotOffered)
    &&& (s.phase is Confirming && s.flow is Start && s.pty_open && is_started_frame(ev) && !echo_pty(ev)
        && echo_mark(ev).len > 0 && (s.mark.len == 0 || echo_mark(ev) == s.mark)) ==> fails_with(out, Fault::NotOffered)
    &&& (s.phase is Confirming && s.kept && s.flow is Start && s2.phase is Done && fails_with(out, Fault::NotOffered))
        ==> (out.len() == 3 && out[1] is Kill)
    &&& (s.phase is Streaming && is_shell_exit_frame(ev)) ==> {
        &&& s2 == s
        &&& out.len() == 1 && out[0] is ShellExit
    }
}

/// X: exit. The exit is delivered once, by the stream's final frame, and nothing after it reopens the
/// stream or delivers anything.
pub open spec fn post_exit(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::DeliverExit { .. } => s.phase is Streaming && is_exit_frame(ev) && s2.phase is ExitHeld,
        _ => true,
    }
    &&& (s.phase is ExitHeld || s.phase is Removing || s.phase is Done) ==> forall|i: int|
        0 <= i < out.len() ==> !is_open_action(#[trigger] out[i]) && !is_delivery(out[i])
    &&& late(s) ==> (late(s2) || s2.phase is Done)
    &&& s.flow is Remove ==> s2.flow is Remove
    &&& forall|i: int| 0 <= i < out.len() ==> (is_open_action(#[trigger] out[i]) ==> !(s.flow is Remove))
}

/// R: remove. A remove is sent only after the consumer took the exit and the policy asked for it, scoped
/// to the held mark; once taken, it is sent and its answer finishes the handle.
pub open spec fn post_remove(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>) -> bool {
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Remove { mark } => {
            &&& mark == s.mark
            &&& s.mark.len > 0
            &&& ((s.phase is ExitHeld && ev is ExitTaken && s.remove_on_exit && s2.phase is Removing)
                || (s.phase is Backoff && s.flow is Remove && ev is Tick && now >= s.until_ms
                && s2.phase is Removing))
        },
        _ => true,
    }
    &&& (s.phase is ExitHeld && ev is ExitTaken) ==> {
        &&& s.remove_on_exit ==> (s2.phase is Removing && out == seq![Action::Remove { mark: s.mark }])
        &&& !s.remove_on_exit ==> (s2.phase is Done && out == seq![Action::Finish])
    }
}

/// C: classification. A request's answer is classified by one table; a terminal class finishes with its
/// fault, a retry class waits and tries again (never ending the handle), and a good answer
/// goes on.
pub open spec fn post_classify(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    (s.phase is Starting || s.phase is Opening || s.phase is Removing) && ev is Answered ==> {
        match class_of(op_of(s.phase), ev_status(ev), ev_authority(ev)) {
            Class::Ok => {
                &&& s2.misses == s.misses
                &&& s.phase is Removing ==> (s2.phase is Done && out == seq![Action::Finish])
                &&& !(s.phase is Removing) ==> (s2.phase is Confirming && out.len() == 0)
            },
            Class::Retry => {
                &&& s2.phase is Backoff
            },
            Class::Terminal(f) => {
                &&& s2.phase is Done
                &&& out == seq![Action::Fail { fault: f, status: ev_status(ev) as u16, offset: s.offset }]
            },
        }
    }
}

/// W: waiting. Every sleep ends within the cap of now, and honours the server's `Retry-After` up to it.
pub open spec fn post_backoff(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>) -> bool {
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Sleep { until_ms } => {
            &&& until_ms >= now
            &&& (until_ms == s.until_ms || until_ms as int <= now as int + BACKOFF_CAP_MS as int)
            &&& s2.phase is Backoff
            &&& s2.until_ms == until_ms
        },
        _ => true,
    }
    &&& ((s.phase is Starting || s.phase is Opening || s.phase is Removing) && ev is Answered
        && class_of(op_of(s.phase), ev_status(ev), ev_authority(ev)) is Retry) ==> {
        &&& s2.phase is Backoff
        &&& s2.until_ms as int >= min_int(
            now as int + min_int(ev_retry_after_ms(ev), BACKOFF_CAP_MS as int),
            u64::MAX as int,
        )
    }
}

/// P: progress. When the server is reachable, every handle that is not streaming or done moves one
/// phase down the ranking with each fair event, and never fails: a held job is reattached or ended in at
/// most `rank` fair steps.
pub open spec fn post_progress(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>) -> bool {
    &&& (s.phase is Backoff && ev is Tick && now >= s.until_ms) ==> {
        &&& s.flow is Start ==> (s2.phase is Starting && out == seq![
            Action::Start { key: s.key, silence_ms: silence_spec(s.beat_s) as u64 },
        ])
        &&& s.flow is Attach ==> (s2.phase is Opening && out == seq![
            Action::OpenAttach {
                offset: s.offset,
                mark: s.mark,
                silence_ms: silence_spec(s.beat_s) as u64,
            },
        ])
        &&& s.flow is Remove ==> (s2.phase is Removing && out == seq![Action::Remove { mark: s.mark }])
    }
    &&& (s.phase is Idle && ev is Go) ==> {
        &&& s.flow is Start ==> (s2.phase is Starting && out == seq![
            Action::Start { key: s.key, silence_ms: silence_spec(s.beat_s) as u64 },
        ])
        &&& s.flow is Attach ==> (s2.phase is Opening && out == seq![
            Action::OpenAttach {
                offset: s.offset,
                mark: s.mark,
                silence_ms: silence_spec(s.beat_s) as u64,
            },
        ])
    }
    &&& fair(s, ev, now) ==> forall|i: int| 0 <= i < out.len() ==> !is_fail(#[trigger] out[i])
    &&& fair(s, ev, now) ==> s2.misses == (if s.phase is Confirming { 0 } else { s.misses })
}

/// S: silence. Every request names the dead-stream limit the heartbeat it chose implies.
pub open spec fn post_silence(s: State, s2: State, out: Seq<Action>) -> bool {
    &&& s2.beat_s == s.beat_s
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Start { silence_ms, .. } => silence_ms as int == silence_spec(s.beat_s),
        Action::OpenAttach { silence_ms, .. } => silence_ms as int == silence_spec(s.beat_s),
        _ => true,
    }
}

/// K: the start key. Every send of the start carries the key of the first send, and a lost start is sent
/// again rather than given up.
pub open spec fn post_key(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& s2.key == s.key
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Start { key, .. } => key == s.key && s.flow is Start && s.mark.len == 0,
        _ => true,
    }
    &&& (s.phase is Starting && ev is Broke) ==> {
        &&& s2.phase is Backoff
        &&& s2.flow is Start
    }
}

/// T: a start nothing answered. A start whose first request could not even connect (`Unreached`: refused,
/// unreachable, the name did not resolve) before any request of it may have reached a server ends with
/// `Transport`: nothing was started and nobody answered, so there is nothing to keep trying for. From the
/// moment any request of the handle was answered, or its answer was lost (`reached`), the handle never ends on
/// a failure to connect: it backs off and reopens, however many fail in a row. Only that one case ends with
/// `Transport`, and each failure to connect after it is the retry that a retry-class answer gets (the handle
/// is not done, so the caller's retry is the next `Tick`).
pub open spec fn post_unreached(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& s2.reached == (s.reached || (s.phase is Starting && (ev is Answered || ev is Broke)))
    &&& (ev is Unreached && s.phase is Starting && !s.reached) ==> {
        &&& s2.phase is Done
        &&& out == seq![Action::Fail { fault: Fault::Transport, status: 0, offset: s.offset }]
    }
    &&& (ev is Unreached && !(s.phase is Done) && !(s.phase is Starting && !s.reached)) ==> !(s2.phase is Done)
    &&& (ev is Unreached && ((s.phase is Starting && s.reached) || s.phase is Opening || s.phase is Removing))
        ==> s2.phase is Backoff
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Fail { fault, .. } => (fault is Transport) ==> (ev is Unreached && s.phase is Starting
            && !s.reached),
        _ => true,
    }
}

/// V: watch. A watching handle counts output and delivers none of it.
pub open spec fn post_watch(s: State, s2: State, out: Seq<Action>) -> bool {
    &&& s2.mode == s.mode
    &&& s2.remove_on_exit == s.remove_on_exit
    &&& s.mode is Watch ==> forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Deliver { .. } => false,
        Action::DeliverExit { dropped, .. } => dropped == 0,
        _ => true,
    }
}

/// D: detach and end. A close detaches (it never removes or kills), a handle that is done stays done, and
/// a handle is done exactly when it issued its one terminal action.
pub open spec fn post_end(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& (ev is Close && !(s.phase is Done)) ==> {
        &&& s2.phase is Done
        &&& out.len() > 0 && out[out.len() - 1] == Action::Detached
        &&& forall|i: int| 0 <= i < out.len() ==> (#[trigger] out[i] == Action::CloseStream
            || out[i] == Action::Detached)
    }
    &&& (ev is Deadline && !(s.phase is Done)) ==> {
        &&& s2.phase is Done
        &&& fails_with(out, Fault::Deadline)
    }
    &&& s.phase is Done ==> (s2 == s && out.len() == 0)
    &&& (s2.phase is Done && !(s.phase is Done)) ==> (out.len() > 0 && is_terminal_action(out[out.len() - 1]))
    &&& !(s2.phase is Done) ==> forall|i: int| 0 <= i < out.len() ==> !is_terminal_action(#[trigger] out[i])
    &&& forall|i: int| 0 <= i < out.len() ==> (is_terminal_action(#[trigger] out[i]) ==> i == out.len() - 1)
}

/// A: only these events end a handle (move it from live to done), and each only as stated: the consumer's
/// `Close` or `Deadline`; the consumer taking the exit when no remove is owed; an answer whose class is
/// terminal, or the answer to the remove; a frame that is not the echo of the held mark while confirming, an
/// echo of another mark or a bad frame or an overflowing count while streaming. Every other event leaves a
/// live handle live: a heartbeat, output, a gap, the exit frame, a 2xx or retry-class answer, a cut, a tick,
/// however many failed opens came before. Reconnecting is a status, never an end.
pub open spec fn post_only_these_end(s: State, ev: Event, s2: State, out: Seq<Action>) -> bool {
    &&& (s2.phase is Done && !(s.phase is Done)) ==> {
        ||| ev is Close
        ||| ev is Deadline
        ||| ev is ExitTaken
        ||| ev is Answered
        ||| ev is Frame
        ||| ev is Unreached
    }
    &&& (ev is ExitTaken && s2.phase is Done && !(s.phase is Done)) ==> (s.phase is ExitHeld
        && !s.remove_on_exit)
    &&& (ev is Answered && s2.phase is Done && !(s.phase is Done)) ==> {
        &&& s.phase is Starting || s.phase is Opening || s.phase is Removing
        &&& (class_of(op_of(s.phase), ev_status(ev), ev_authority(ev)) is Terminal || (s.phase is Removing
            && class_of(op_of(s.phase), ev_status(ev), ev_authority(ev)) is Ok))
    }
    &&& ((s.phase is Starting || s.phase is Opening || s.phase is Removing) && ev is Answered
        && !(class_of(op_of(s.phase), ev_status(ev), ev_authority(ev)) is Terminal)
        && !(s.phase is Removing && class_of(op_of(s.phase), ev_status(ev), ev_authority(ev)) is Ok)) ==> !(s2.phase is Done)
    &&& (s.phase is Streaming && ev is Frame && ev->Frame_0 is Heartbeat) ==> !(s2.phase is Done)
    &&& (s.phase is Streaming && is_output(ev) && s.offset as int + s.pending as int + output_len(ev)
        <= u64::MAX as int) ==> !(s2.phase is Done)
    &&& (s.phase is Streaming && is_gap(ev) && s.pending as int + gap_len(ev) <= u64::MAX as int)
        ==> !(s2.phase is Done)
    &&& (s.phase is Streaming && is_exit_frame(ev) && s.offset as int + s.pending as int <= u64::MAX as int)
        ==> !(s2.phase is Done)
    &&& (s.phase is Confirming && good_echo(s, ev)) ==> !(s2.phase is Done)
    &&& ((ev is Broke || ev is Tick) && !(s.phase is Done)) ==> !(s2.phase is Done)
    &&& forall|i: int| 0 <= i < out.len() ==> match #[trigger] out[i] {
        Action::Fail { fault, .. } => {
            &&& (fault is Deadline) == (ev is Deadline)
            // SG: an unreachable endpoint ends nothing; only an authoritative answer does.
            &&& (fault is SandboxGone) ==> (ev is Answered && ev_authority(ev) is SandboxGone)
            &&& (fault is SessionEnded) ==> (ev is Answered && ev_authority(ev) is SessionEnded)
        },
        _ => true,
    }
}

/// Everything `step` promises.
pub open spec fn step_post(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>) -> bool {
    &&& wf(s2)
    &&& post_offsets(s, ev, s2, out)
    &&& post_gaps(s, ev, s2, out)
    &&& post_budget(s, ev, s2, out)
    &&& post_incarnation(s, ev, s2, out)
    &&& post_exit(s, ev, s2, out)
    &&& post_remove(s, ev, now, s2, out)
    &&& post_classify(s, ev, s2, out)
    &&& post_backoff(s, ev, now, s2, out)
    &&& post_progress(s, ev, now, s2, out)
    &&& post_silence(s, s2, out)
    &&& post_key(s, ev, s2, out)
    &&& post_watch(s, s2, out)
    &&& post_end(s, ev, s2, out)
    &&& post_only_these_end(s, ev, s2, out)
    &&& post_unreached(s, ev, s2, out)
}

// ---------------------------------------------------------------------------------------------
// The silence limit, the classifier, the wait.
// ---------------------------------------------------------------------------------------------

/// The dead-stream limit for a request that chose heartbeat `beat_s` (whole seconds; 0 or 1 to 240 are the
/// valid choices, and 0 means the default): never under 30 s, and never under two beats and ten seconds, so
/// a stream that asked for a long beat is not declared dead by its own client.
pub fn silence_ms(beat_s: u32) -> (r: u64)
    ensures
        r >= 30000,
        r as int >= 2000 * (beat_s as int) + 10000,
        r == 30000 || r as int == 2000 * (beat_s as int) + 10000,
        r as int == silence_spec(beat_s),
{
    let two_beats: u64 = if cfg!(feature = "mut_silence_one_beat") {
        1000 * (beat_s as u64) + 10000
    } else {
        2000 * (beat_s as u64) + 10000
    };
    if cfg!(feature = "mut_silence_fixed") {
        30000
    } else if two_beats > 30000 {
        two_beats
    } else {
        30000
    }
}

/// What an answer to `op` with `status` means.
pub fn classify(op: Op, status: u16, auth: Authority) -> (c: Class)
    ensures
        // (j) an authoritative answer is its own terminal class, whatever the status (E2B says it with a
        // 502), and nothing else is: `SandboxGone` and `SessionEnded` come from `auth` alone.
        auth is SandboxGone ==> c == Class::Terminal(Fault::SandboxGone),
        auth is SessionEnded ==> c == Class::Terminal(Fault::SessionEnded),
        c == Class::Terminal(Fault::SandboxGone) ==> auth is SandboxGone,
        c == Class::Terminal(Fault::SessionEnded) ==> auth is SessionEnded,
        // (a) any 2xx goes on.
        ((auth is NotAuthoritative) && 200 <= status <= 299) ==> c == Class::Ok,
        // (b) 500 is terminal, whatever the request: a retry would not change it.
        ((auth is NotAuthoritative) && status == 500) ==> c == Class::Terminal(Fault::Protocol),
        // (c) 503 on a start means no pseudo-terminal is free: terminal.
        ((auth is NotAuthoritative) && matches!(op, Op::Start) && status == 503) ==> c == Class::Terminal(Fault::NoPty),
        // (d) the busy answers are retried (completeness: a retry class is never terminal).
        ((auth is NotAuthoritative) && (status == 429 || status == 502 || status == 504 || (status == 503 && !(matches!(op, Op::Start)))))
            ==> c == Class::Retry,
        // (e) an attach of a job that is gone is `JobGone`.
        ((auth is NotAuthoritative) && (matches!(op, Op::Attach) || matches!(op, Op::Other)) && status == 404) ==> c == Class::Terminal(Fault::JobGone),
        // (f) the start answers that name a cause.
        ((auth is NotAuthoritative) && matches!(op, Op::Start) && status == 409) ==> c == Class::Terminal(Fault::JobExists),
        ((auth is NotAuthoritative) && matches!(op, Op::Start) && status == 507) ==> c == Class::Terminal(Fault::DiskFull),
        // (g) the credential is refused.
        ((auth is NotAuthoritative) && (status == 401 || status == 403)) ==> c == Class::Terminal(Fault::Auth),
        // (h) a remove of a job already gone is done.
        ((auth is NotAuthoritative) && matches!(op, Op::Remove) && status == 404) ==> c == Class::Ok,
        // (i) nothing else is retried: only the four busy statuses are.
        c is Retry ==> (status == 429 || status == 502 || status == 503 || status == 504),
        c == class_of(op, status as int, auth),
{
    if matches!(auth, Authority::SandboxGone) && !cfg!(feature = "mut_gone_ignored") {
        Class::Terminal(Fault::SandboxGone)
    } else if matches!(auth, Authority::SessionEnded) && !cfg!(feature = "mut_gone_ignored") {
        Class::Terminal(Fault::SessionEnded)
    } else if 200 <= status && status <= 299 {
        Class::Ok
    } else if matches!(op, Op::Remove) && status == 404 {
        Class::Ok
    } else if status == 429 || status == 502 || status == 504 {
        if cfg!(feature = "mut_429_terminal") && status == 429 {
            Class::Terminal(Fault::Protocol)
        } else {
            Class::Retry
        }
    } else if status == 503 {
        if matches!(op, Op::Start) {
            if cfg!(feature = "mut_503_start_retried") { Class::Retry } else { Class::Terminal(Fault::NoPty) }
        } else {
            Class::Retry
        }
    } else if status == 401 || status == 403 {
        if cfg!(feature = "mut_auth_retried") && status == 401 {
            Class::Retry
        } else {
            Class::Terminal(Fault::Auth)
        }
    } else if status == 404 && matches!(op, Op::Other) && cfg!(feature = "mut_other_404_unclassified") {
        Class::Terminal(Fault::Protocol)
    } else if status == 404 && (matches!(op, Op::Attach) || matches!(op, Op::Other)) {
        if cfg!(feature = "mut_404_retried") { Class::Retry } else { Class::Terminal(Fault::JobGone) }
    } else if status == 409 && matches!(op, Op::Start) {
        Class::Terminal(Fault::JobExists)
    } else if status == 507 && matches!(op, Op::Start) {
        Class::Terminal(Fault::DiskFull)
    } else if cfg!(feature = "mut_500_retried") && status == 500 {
        Class::Retry
    } else if cfg!(feature = "mut_everything_retried") && status == 400 {
        Class::Retry
    } else {
        Class::Terminal(Fault::Protocol)
    }
}

/// The body of E2B's answer for a sandbox it no longer has: `{"sandboxId": ours, "message": <this>, "code": 502}`.
pub open spec fn gone_message() -> Seq<u8> {
    seq![84u8, 104u8, 101u8, 32u8, 115u8, 97u8, 110u8, 100u8, 98u8, 111u8, 120u8, 32u8, 119u8, 97u8, 115u8, 32u8, 110u8, 111u8, 116u8, 32u8, 102u8, 111u8, 117u8, 110u8, 100u8]
}

/// The `code` of Beyond's answer for a session that has ended: status 401, `{"code": <this>, ...}`.
pub open spec fn session_ended_code() -> Seq<u8> {
    seq![115u8, 101u8, 115u8, 115u8, 105u8, 111u8, 110u8, 95u8, 101u8, 110u8, 100u8, 101u8, 100u8]
}

/// What a refusal is, from the fields the shell parsed out of its JSON body (all empty or `None` for a field the
/// body lacks, or for a body that is not JSON). `ours` is the sandbox id the endpoint is configured for
/// (empty when it has none): an answer about any other sandbox is not about this one.
pub open spec fn authority_of(
    status: int,
    message: Seq<u8>,
    code_num: Option<u64>,
    code_text: Seq<u8>,
    sandbox_id: Seq<u8>,
    ours: Seq<u8>,
) -> Authority {
    if status == 502 && message == gone_message() && code_num == Some(502u64) && ours.len() > 0
        && sandbox_id == ours {
        Authority::SandboxGone
    } else if status == 401 && code_text == session_ended_code() {
        Authority::SessionEnded
    } else {
        Authority::NotAuthoritative
    }
}

/// Whether a refusal is the backend's own authoritative answer: `SandboxGone` exactly when it is a `502` whose
/// message is exactly "The sandbox was not found", whose code is the number 502 and whose `sandboxId` is
/// `ours` (not empty); `SessionEnded` exactly when it is a `401` whose code is exactly "session_ended";
/// `NotAuthoritative` for everything else, a proxy's 502 and a refused credential included. A wrong
/// `SandboxGone` abandons a live job, so this is the one place that decides it, and the shell only parses.
pub fn authority(
    status: u16,
    message: &[u8],
    code_num: Option<u64>,
    code_text: &[u8],
    sandbox_id: &[u8],
    ours: &[u8],
) -> (a: Authority)
    ensures
        a == authority_of(status as int, message@, code_num, code_text@, sandbox_id@, ours@),
{
    let gone: [u8; 25] = [84, 104, 101, 32, 115, 97, 110, 100, 98, 111, 120, 32, 119, 97, 115, 32, 110, 111, 116, 32, 102, 111, 117, 110, 100];
    let ended: [u8; 13] = [115, 101, 115, 115, 105, 111, 110, 95, 101, 110, 100, 101, 100];
    proof {
        assert(gone@ =~= gone_message());
        assert(ended@ =~= session_ended_code());
    }
    let is502_code = match code_num {
        Some(c) => c == 502,
        None => false,
    };
    let ours_ok = cfg!(feature = "mut_gone_any_502") || (ours.len() > 0 && crate::same(sandbox_id, ours));
    if status == 502 && crate::same(message, &gone) && is502_code && ours_ok {
        Authority::SandboxGone
    } else if status == 401 && (cfg!(feature = "mut_gone_any_401") || crate::same(code_text, &ended)) {
        Authority::SessionEnded
    } else {
        Authority::NotAuthoritative
    }
}

/// How long to wait after failed open number `n` (1 for the first), in milliseconds: 250 ms doubling to
/// 30 s, jittered down by up to half with the uniform `draw`, and never under the server's `Retry-After`
/// (capped at 30 s). Never over 30 s.
pub fn wait_ms(n: u32, retry_after_s: u32, draw: u32) -> (r: u64)
    ensures
        r <= BACKOFF_CAP_MS,
        r >= 125,
        r as int >= min_int(1000 * (retry_after_s as int), BACKOFF_CAP_MS as int),
{
    let base: u64 = if n <= 1 {
        250
    } else if n == 2 {
        500
    } else if n == 3 {
        1000
    } else if n == 4 {
        2000
    } else if n == 5 {
        4000
    } else if n == 6 {
        8000
    } else if n == 7 {
        16000
    } else {
        30000
    };
    let base: u64 = if cfg!(feature = "mut_backoff_no_cap") && n >= 7 { 2 * base } else { base };
    // base * draw < 30000 * 2^32, so no overflow, and the quotient by 2^33 is at most base / 2.
    let t: u64 = if base <= 30000 { base * (draw as u64) } else { 0 };
    let q: u64 = t / 8589934592;
    let jittered: u64 = if base <= 30000 { base - q } else { base };
    let ra_ms: u64 = 1000 * (retry_after_s as u64);
    let ra: u64 = if cfg!(feature = "mut_retry_after_ignored") {
        0
    } else if ra_ms > BACKOFF_CAP_MS && !cfg!(feature = "mut_retry_after_uncapped") {
        BACKOFF_CAP_MS
    } else {
        ra_ms
    };
    proof {
        if base <= 30000 {
            assert(t <= 30000 * 4294967295u64) by (nonlinear_arith)
                requires t == base * (draw as u64), base <= 30000, draw <= 4294967295u32;
            assert(q * 8589934592 <= t) by (nonlinear_arith)
                requires q == t / 8589934592;
            assert(q <= 15000) by (nonlinear_arith)
                requires q * 8589934592 <= t, t <= 30000 * 4294967295u64;
            assert(q * 2 <= base) by (nonlinear_arith)
                requires q * 8589934592 <= t, t == base * (draw as u64), draw <= 4294967295u32, base <= 30000;
        }
    }
    if jittered > ra { jittered } else { ra }
}

/// `now + wait`, saturating at the end of the clock.
fn until_of(now: u64, wait: u64) -> (r: u64)
    ensures
        r >= now,
        r as int <= now as int + wait as int,
        now as int + wait as int <= u64::MAX as int ==> r as int == now as int + wait as int,
        now as int + wait as int > u64::MAX as int ==> r == u64::MAX,
{
    if now > u64::MAX - wait { u64::MAX } else { now + wait }
}

// ---------------------------------------------------------------------------------------------
// Transitions.
// ---------------------------------------------------------------------------------------------

/// Finish with a typed failure.
fn fail(s: State, fault: Fault, status: u16, close: bool) -> (r: (State, Vec<Action>))
    ensures
        r.0 == (State { phase: Phase::Done, pending: 0, ..s }),
        close ==> r.1@ == seq![Action::CloseStream, Action::Fail { fault, status, offset: s.offset }],
        !close ==> r.1@ == seq![Action::Fail { fault, status, offset: s.offset }],
{
    let mut t = s;
    t.phase = Phase::Done;
    t.pending = 0;
    let mut out = Vec::new();
    if close {
        out.push(Action::CloseStream);
    }
    out.push(Action::Fail { fault, status, offset: s.offset });
    (t, out)
}

/// Finish `NotOffered` on a refused echo. A kept start also stops the job first: no stream's close ends it,
/// and nothing else can name it.
fn unoffered(s: State) -> (r: (State, Vec<Action>))
    ensures
        r.0 == (State { phase: Phase::Done, pending: 0, ..s }),
        (s.kept && s.flow is Start) ==> r.1@ == seq![
            Action::CloseStream,
            Action::Kill,
            Action::Fail { fault: Fault::NotOffered, status: 0, offset: s.offset },
        ],
        !(s.kept && s.flow is Start) ==> r.1@ == seq![
            Action::CloseStream,
            Action::Fail { fault: Fault::NotOffered, status: 0, offset: s.offset },
        ],
{
    let (t, mut out) = fail(s, Fault::NotOffered, 0, true);
    let kill = if cfg!(feature = "mut_kill_skipped") {
        false
    } else if cfg!(feature = "mut_kill_when_held") {
        true
    } else {
        s.kept && matches!(s.flow, Flow::Start)
    };
    if kill {
        let fail_action = out.pop();
        out.push(Action::Kill);
        if let Some(a) = fail_action {
            out.push(a);
        }
    }
    (t, out)
}

/// Finish: the consumer stopped.
fn detach(s: State, close: bool) -> (r: (State, Vec<Action>))
    ensures
        r.0 == (State { phase: Phase::Done, pending: 0, ..s }),
        close ==> r.1@ == seq![Action::CloseStream, Action::Detached],
        !close ==> r.1@ == seq![Action::Detached],
{
    let mut t = s;
    t.phase = Phase::Done;
    t.pending = 0;
    let mut out = Vec::new();
    if close {
        out.push(Action::CloseStream);
    }
    out.push(Action::Detached);
    (t, out)
}

/// A failed open or a cut: wait and reopen. Never gives up; after `RECONNECTING_AFTER` failures in a row it
/// also says `Reconnecting`.
fn backoff(s: State, retry_after_s: u32, now: u64, draw: u32, close: bool) -> (r: (State, Vec<Action>))
    ensures
        r.0 == (State {
            phase: Phase::Backoff,
            misses: r.0.misses,
            since_ms: r.0.since_ms,
            pending: 0,
            until_ms: r.0.until_ms,
            ..s
        }),
        r.0.misses as int == min_int(s.misses as int + 1, u32::MAX as int),
        r.0.since_ms == (if s.misses == 0 { now } else { s.since_ms }),
        r.0.until_ms >= now,
        r.0.until_ms as int <= now as int + BACKOFF_CAP_MS as int,
        r.0.until_ms as int >= min_int(
            now as int + min_int(1000 * (retry_after_s as int), BACKOFF_CAP_MS as int),
            u64::MAX as int,
        ),
        (r.0.misses < RECONNECTING_AFTER && close) ==> r.1@ == seq![
            Action::CloseStream,
            Action::Sleep { until_ms: r.0.until_ms },
        ],
        (r.0.misses < RECONNECTING_AFTER && !close) ==> r.1@ == seq![Action::Sleep { until_ms: r.0.until_ms }],
        (r.0.misses >= RECONNECTING_AFTER && close) ==> r.1@ == seq![
            Action::CloseStream,
            Action::Sleep { until_ms: r.0.until_ms },
            Action::Reconnecting { attempts: r.0.misses, since_ms: r.0.since_ms },
        ],
        (r.0.misses >= RECONNECTING_AFTER && !close) ==> r.1@ == seq![
            Action::Sleep { until_ms: r.0.until_ms },
            Action::Reconnecting { attempts: r.0.misses, since_ms: r.0.since_ms },
        ],
{
    let n = if s.misses == u32::MAX { s.misses } else { s.misses + 1 };
    let w = wait_ms(n, retry_after_s, draw);
    let until = until_of(now, w);
    let mut t = s;
    t.phase = Phase::Backoff;
    t.misses = n;
    if s.misses == 0 {
        t.since_ms = now;
    }
    t.pending = 0;
    t.until_ms = until;
    let mut out = Vec::new();
    if close {
        out.push(Action::CloseStream);
    }
    out.push(Action::Sleep { until_ms: until });
    if n >= RECONNECTING_AFTER {
        out.push(Action::Reconnecting { attempts: n, since_ms: t.since_ms });
    }
    (t, out)
}

/// The echo came: output flows from here.
fn streaming(s: State, echo: Mark, gap: u64) -> (r: State)
    requires
        echo.len > 0,
    ensures
        r == (State {
            phase: Phase::Streaming,
            flow: Flow::Attach,
            mark: echo,
            pending: gap,
            misses: 0,
            announced: true,
            ..s
        }),
{
    let mut t = s;
    t.phase = Phase::Streaming;
    t.flow = Flow::Attach;
    t.mark = echo;
    t.pending = gap;
    t.announced = true;
    if !cfg!(feature = "mut_budget_not_reset_on_echo") {
        t.misses = 0;
    }
    t
}

/// The wait is over (or the first request is due): send it.
fn open_action(s: State, draw_key: u64) -> (r: (State, Vec<Action>))
    requires
        s.phase is Idle || s.phase is Backoff,
    ensures
        s.flow is Start ==> (r.0 == (State { phase: Phase::Starting, ..s }) && r.1@ == seq![
            Action::Start { key: s.key, silence_ms: silence_spec(s.beat_s) as u64 },
        ]),
        s.flow is Attach ==> (r.0 == (State { phase: Phase::Opening, ..s }) && r.1@ == seq![
            Action::OpenAttach { offset: s.offset, mark: s.mark, silence_ms: silence_spec(s.beat_s) as u64 },
        ]),
        s.flow is Remove ==> (r.0 == (State { phase: Phase::Removing, ..s })
            && r.1@ == seq![Action::Remove { mark: s.mark }]),
{
    let silence = silence_ms(s.beat_s);
    let mut t = s;
    let mut out = Vec::new();
    match s.flow {
        Flow::Start => {
            t.phase = Phase::Starting;
            let _ = draw_key;
            out.push(Action::Start { key: s.key, silence_ms: silence });
        },
        Flow::Attach => {
            t.phase = Phase::Opening;
            let offset = if cfg!(feature = "mut_reattach_from_zero") && matches!(s.phase, Phase::Backoff) {
                0
            } else {
                s.offset
            };
            out.push(Action::OpenAttach { offset, mark: s.mark, silence_ms: silence });
        },
        Flow::Remove => {
            t.phase = Phase::Removing;
            let mark = if cfg!(feature = "mut_remove_unscoped") { Mark::none() } else { s.mark };
            out.push(Action::Remove { mark });
        },
    }
    (t, out)
}

// ---------------------------------------------------------------------------------------------
// The step, one function per phase.
// ---------------------------------------------------------------------------------------------

#[allow(unused_variables)]
/// A close or the caller's deadline, from any phase but done: detach (a close never removes or kills), or
/// fail with `Deadline` carrying the cursor.
fn step_close(s: State, ev: Event, now: u64) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
        !(s.phase is Done),
        ev is Close || ev is Deadline,
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    let open = matches!(s.phase, Phase::Starting) || matches!(s.phase, Phase::Opening)
        || matches!(s.phase, Phase::Confirming) || matches!(s.phase, Phase::Streaming)
        || matches!(s.phase, Phase::Removing);
    if matches!(ev, Event::Deadline) {
        return fail(s, Fault::Deadline, 0, open);
    }
    if cfg!(feature = "mut_close_removes") && matches!(s.phase, Phase::ExitHeld) {
        let (t, mut out) = detach(s, false);
        out.push(Action::Remove { mark: s.mark });
        return (t, out);
    }
    detach(s, open)
}

#[allow(unused_variables)]
fn step_idle(s: State, ev: Event, now: u64, draw: u32) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
        s.phase is Idle,
        !(ev is Close),
        !(ev is Deadline),
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    if matches!(ev, Event::Go) {
        return open_action(s, draw as u64);
    }
    (s, Vec::new())
}

fn step_backoff(s: State, ev: Event, now: u64, draw: u32) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
        s.phase is Backoff,
        !(ev is Close),
        !(ev is Deadline),
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    if matches!(ev, Event::Tick) {
        if now >= s.until_ms {
            if cfg!(feature = "mut_backoff_never_reopens") {
                return (s, Vec::new());
            }
            return open_action(s, draw as u64);
        }
        let mut out = Vec::new();
        out.push(Action::Sleep { until_ms: s.until_ms });
        return (s, out);
    }
    (s, Vec::new())
}

/// The handle after an event, as to `reached`: an answer or a lost answer of a start means a request of it may
/// have reached a server.
fn mark_reached(s: State, ev: Event) -> (r: State)
    ensures
        r == (State { reached: s.reached || (s.phase is Starting && (ev is Answered || ev is Broke)), ..s }),
{
    let mut t = s;
    if matches!(s.phase, Phase::Starting) && (matches!(ev, Event::Answered { .. }) || matches!(ev, Event::Broke)) {
        t.reached = true;
    }
    t
}

/// A request is in flight (start, attach or remove) and the answer, or the cut, came.
fn step_request(s: State, ev: Event, now: u64, draw: u32) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
        s.phase is Starting || s.phase is Opening || s.phase is Removing,
        !(ev is Close),
        !(ev is Deadline),
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    let s = mark_reached(s, ev);
    match ev {
        Event::Answered { status, retry_after_s, authority } => {
            let op = if matches!(s.phase, Phase::Starting) {
                Op::Start
            } else if matches!(s.phase, Phase::Removing) {
                Op::Remove
            } else {
                Op::Attach
            };
            match classify(op, status, authority) {
                Class::Ok => {
                    if matches!(s.phase, Phase::Removing) {
                        let mut t = s;
                        t.phase = Phase::Done;
                        let mut out = Vec::new();
                        out.push(Action::Finish);
                        (t, out)
                    } else {
                        let mut t = s;
                        t.phase = if cfg!(feature = "mut_echo_skipped") {
                            Phase::Streaming
                        } else {
                            Phase::Confirming
                        };
                        if cfg!(feature = "mut_budget_reset_on_connect") {
                            t.misses = 0;
                        }
                        (t, Vec::new())
                    }
                },
                Class::Retry => backoff(s, retry_after_s, now, draw, false),
                Class::Terminal(f) => fail(s, f, status, false),
            }
        },
        Event::Broke => {
            if cfg!(feature = "mut_gone_on_silence") {
                return fail(s, Fault::SandboxGone, 0, true);
            }
            if cfg!(feature = "mut_start_lost_gives_up") && matches!(s.phase, Phase::Starting) {
                return fail(s, Fault::Protocol, 0, true);
            }
            backoff(s, 0, now, draw, true)
        },
        Event::Unreached => {
            if matches!(s.phase, Phase::Starting) && !s.reached && !cfg!(feature = "mut_unanswered_start_retries") {
                return fail(s, Fault::Transport, 0, false);
            }
            backoff(s, 0, now, draw, false)
        },
        _ => (s, Vec::new()),
    }
}

/// The stream is open and its first frame has not come: it must be the echo.
fn step_confirming(s: State, ev: Event, now: u64, draw: u32) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
        s.phase is Confirming,
        !(ev is Close),
        !(ev is Deadline),
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    match ev {
        Event::Frame(f) => match f {
            Frame::Attached { mark, from } => {
                if mark.len == 0 {
                    unoffered(s)
                } else if s.mark.len > 0 && !mark_eq(&mark, &s.mark) && !cfg!(feature = "mut_echo_mark_unchecked") {
                    fail(s, Fault::JobGone, 0, true)
                } else if matches!(s.flow, Flow::Start) {
                    unoffered(s)
                } else if from < s.offset {
                    fail(s, Fault::Protocol, 0, true)
                } else {
                    let t = streaming(s, mark, from - s.offset);
                    let mut out = Vec::new();
                    if !s.announced || cfg!(feature = "mut_attached_repeated") {
                        if !cfg!(feature = "mut_attached_not_told") {
                            out.push(Action::Attached { mark, from });
                        }
                    }
                    (t, out)
                }
            },
            Frame::Started { mark, stdin, pty } => {
                if mark.len == 0 {
                    unoffered(s)
                } else if s.mark.len > 0 && !mark_eq(&mark, &s.mark) && !cfg!(feature = "mut_echo_mark_unchecked") {
                    fail(s, Fault::JobGone, 0, true)
                } else if matches!(s.flow, Flow::Attach) {
                    unoffered(s)
                } else if s.stdin_open && !stdin && !cfg!(feature = "mut_stdin_unchecked") {
                    unoffered(s)
                } else if s.pty_open && !pty && !cfg!(feature = "mut_pty_unchecked") {
                    unoffered(s)
                } else {
                    let t = streaming(s, mark, 0);
                    let mut out = Vec::new();
                    out.push(Action::Started { mark, stdin, pty });
                    (t, out)
                }
            },
            Frame::Bad => fail(s, Fault::Protocol, 0, true),
            _ => unoffered(s),
        },
        Event::Broke => backoff(s, 0, now, draw, true),
        _ => (s, Vec::new()),
    }
}

/// Output flows.
fn step_streaming(s: State, ev: Event, now: u64, draw: u32) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
        s.phase is Streaming,
        !(ev is Close),
        !(ev is Deadline),
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    match ev {
        Event::Frame(f) => match f {
            Frame::Output { len } => {
                if s.pending > u64::MAX - s.offset {
                    if cfg!(feature = "mut_offset_wraps") {
                        let mut t = s;
                        t.offset = 0;
                        t.pending = 0;
                        return (t, Vec::new());
                    }
                    return fail(s, Fault::Protocol, 0, true);
                }
                let at = s.offset + s.pending;
                if len > u64::MAX - at {
                    return fail(s, Fault::Protocol, 0, true);
                }
                let mut t = s;
                t.offset = at + len;
                t.pending = 0;
                let mut out = Vec::new();
                if matches!(s.mode, Mode::Follow) || cfg!(feature = "mut_watch_delivers") {
                    out.push(Action::Deliver { at, len, dropped: s.pending });
                }
                (t, out)
            },
            Frame::Gap { n } => {
                if n > u64::MAX - s.pending {
                    return fail(s, Fault::Protocol, 0, true);
                }
                let mut t = s;
                if !cfg!(feature = "mut_gap_not_counted") {
                    t.pending = s.pending + n;
                }
                (t, Vec::new())
            },
            Frame::Heartbeat => (s, Vec::new()),
            Frame::ShellExit { code } => {
                let mut out = Vec::new();
                if !cfg!(feature = "mut_shell_exit_not_told") {
                    out.push(Action::ShellExit { code });
                }
                (s, out)
            },
            Frame::Exit { code, signal } => {
                if s.pending > u64::MAX - s.offset {
                    return fail(s, Fault::Protocol, 0, true);
                }
                let mut t = s;
                t.offset = s.offset + s.pending;
                t.pending = 0;
                t.phase = Phase::ExitHeld;
                let dropped = if matches!(s.mode, Mode::Follow) && !cfg!(feature = "mut_trailing_gap_dropped") {
                    s.pending
                } else {
                    0
                };
                let mut out = Vec::new();
                out.push(Action::CloseStream);
                let killed = if cfg!(feature = "mut_signal_zero_kills") { signal == 0 } else { signal != 0 };
                out.push(Action::DeliverExit { code, signal, dropped, killed });
                if cfg!(feature = "mut_remove_on_exit_frame") {
                    out.push(Action::Remove { mark: s.mark });
                }
                (t, out)
            },
            Frame::Attached { mark, .. } => {
                if mark.len > 0 && !mark_eq(&mark, &s.mark) {
                    fail(s, Fault::JobGone, 0, true)
                } else {
                    fail(s, Fault::Protocol, 0, true)
                }
            },
            Frame::Started { mark, .. } => {
                if mark.len > 0 && !mark_eq(&mark, &s.mark) {
                    fail(s, Fault::JobGone, 0, true)
                } else {
                    fail(s, Fault::Protocol, 0, true)
                }
            },
            Frame::Bad => fail(s, Fault::Protocol, 0, true),
        },
        Event::Broke => backoff(s, 0, now, draw, true),
        _ => (s, Vec::new()),
    }
}

/// The exit was delivered; waiting for the consumer to take it.
fn step_exit_held(s: State, ev: Event, now: u64, draw: u32) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
        s.phase is ExitHeld,
        !(ev is Close),
        !(ev is Deadline),
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    match ev {
        Event::ExitTaken => {
            if s.remove_on_exit && !cfg!(feature = "mut_exit_taken_stalls") {
                let mut t = s;
                t.phase = Phase::Removing;
                t.flow = Flow::Remove;
                let mark = if cfg!(feature = "mut_remove_unscoped") { Mark::none() } else { s.mark };
                let mut out = Vec::new();
                out.push(Action::Remove { mark });
                (t, out)
            } else if cfg!(feature = "mut_exit_taken_stalls") {
                (s, Vec::new())
            } else {
                let mut t = s;
                t.phase = Phase::Done;
                let mut out = Vec::new();
                out.push(Action::Finish);
                (t, out)
            }
        },
        Event::Broke => {
            if cfg!(feature = "mut_reattach_after_exit") {
                return backoff(s, 0, now, draw, false);
            }
            (s, Vec::new())
        },
        _ => (s, Vec::new()),
    }
}

/// One step: the handle's next state and what the shell does. Pure: the same inputs give the same outputs.
///
/// `now` is the shell's clock and `draw` a fresh uniform `u32`; both are only read by an event that
/// waits. A stale event (a frame after the cut, a tick when not waiting) changes nothing and asks for
/// nothing.
pub fn step(s: State, ev: Event, now: u64, draw: u32) -> (r: (State, Vec<Action>))
    requires
        wf(s),
        wf_ev(ev),
    ensures
        wf(r.0),
        post_offsets(s, ev, r.0, r.1@),
        post_gaps(s, ev, r.0, r.1@),
        post_budget(s, ev, r.0, r.1@),
        post_incarnation(s, ev, r.0, r.1@),
        post_exit(s, ev, r.0, r.1@),
        post_remove(s, ev, now, r.0, r.1@),
        post_classify(s, ev, r.0, r.1@),
        post_backoff(s, ev, now, r.0, r.1@),
        post_progress(s, ev, now, r.0, r.1@),
        post_silence(s, r.0, r.1@),
        post_key(s, ev, r.0, r.1@),
        post_watch(s, r.0, r.1@),
        post_end(s, ev, r.0, r.1@),
        post_only_these_end(s, ev, r.0, r.1@),
        post_unreached(s, ev, r.0, r.1@),
        post_surface(s, ev, r.0, r.1@),
{
    if matches!(s.phase, Phase::Done) {
        return (s, Vec::new());
    }
    if matches!(ev, Event::Close) || matches!(ev, Event::Deadline) {
        return step_close(s, ev, now);
    }
    let r = match s.phase {
        Phase::Idle => step_idle(s, ev, now, draw),
        Phase::Backoff => step_backoff(s, ev, now, draw),
        Phase::Starting | Phase::Opening | Phase::Removing => step_request(s, ev, now, draw),
        Phase::Confirming => step_confirming(s, ev, now, draw),
        Phase::Streaming => step_streaming(s, ev, now, draw),
        Phase::ExitHeld => step_exit_held(s, ev, now, draw),
        Phase::Done => (s, Vec::new()),
    };
    // The mutations of the properties on `step` itself act here, on what the phase functions returned, so the
    // proof goes red at `step`'s own clauses.
    if cfg!(feature = "mut_gives_up_on_budget") && matches!(ev, Event::Broke) && r.0.misses >= RECONNECTING_AFTER {
        return fail(s, Fault::Protocol, 0, true);
    }
    if cfg!(feature = "mut_heartbeat_ends")
        && matches!(s.phase, Phase::Streaming)
        && matches!(ev, Event::Frame(Frame::Heartbeat))
    {
        return fail(s, Fault::Protocol, 0, true);
    }
    if cfg!(feature = "mut_start_fresh_key") && matches!(s.phase, Phase::Backoff) && matches!(ev, Event::Tick)
        && r.1.len() == 1
    {
        if let Action::Start { silence_ms, .. } = r.1[0] {
            let mut out = Vec::new();
            out.push(Action::Start { key: draw as u64, silence_ms });
            return (r.0, out);
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------
// Progress.
// ---------------------------------------------------------------------------------------------

/// The ranking function of the progress lemma.
pub open spec fn rank(s: State) -> int {
    rank_of(s.phase)
}

/// P: every fair step lowers the rank and never fails, so a handle that is not streaming or done reaches
/// streaming (reattached) or done (ended) in at most `rank` fair steps once the server is reachable.
/// Fairness (`fair`): a backoff's sleep ends, a request is answered 2xx, the echo carries the held mark,
/// the consumer takes the exit.
pub proof fn lemma_progress(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>)
    requires
        wf(s),
        fair(s, ev, now),
        step_post(s, ev, now, s2, out),
    ensures
        rank(s2) < rank(s),
        forall|i: int| 0 <= i < out.len() ==> !is_fail(#[trigger] out[i]),
        rank(s) <= 5,
{
    assert(rank(s) <= 5);
}

/// A stream that carries the final frame delivers the exit: the handle is at `ExitHeld`, one phase from
/// its end.
pub proof fn lemma_exit_progress(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>)
    requires
        wf(s),
        wf_ev(ev),
        s.phase is Streaming,
        is_exit_frame(ev),
        s.offset as int + s.pending as int <= u64::MAX as int,
        step_post(s, ev, now, s2, out),
    ensures
        s2.phase is ExitHeld,
        rank(s2) <= 2,
{
}

/// SG-live (the liveness partner of the refusals that end a handle on an answer): the first authoritative
/// answer ends the handle in that one step, with its own fault, whatever the status. The fairness
/// assumption: the backend answers (the authority is what `authority` made of that answer).
pub proof fn lemma_gone_ends_in_one_step(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>)
    requires
        wf(s),
        s.phase is Starting || s.phase is Opening || s.phase is Removing,
        ev is Answered,
        !(ev_authority(ev) is NotAuthoritative),
        step_post(s, ev, now, s2, out),
    ensures
        s2.phase is Done,
        ev_authority(ev) is SandboxGone ==> fails_with(out, Fault::SandboxGone),
        ev_authority(ev) is SessionEnded ==> fails_with(out, Fault::SessionEnded),
{
}

/// SG-complete (the completeness partner of `authority`'s refusals): the backend's exact answer is always
/// recognised, so a retry with the same answer never ends in `NotAuthoritative`.
pub proof fn lemma_authority_complete(
    message: Seq<u8>,
    sandbox_id: Seq<u8>,
    ours: Seq<u8>,
    code_text: Seq<u8>,
)
    requires
        message == gone_message(),
        ours.len() > 0,
        sandbox_id == ours,
    ensures
        authority_of(502, message, Some(502u64), code_text, sandbox_id, ours) is SandboxGone,
        authority_of(401, Seq::empty(), None, session_ended_code(), sandbox_id, ours) is SessionEnded,
{
}

/// SG-silent: an endpoint that cannot be reached ends nothing. A cut, a refused connect and a timeout are
/// `Broke`, and a tick is the wait ending: neither ever finishes a handle, however many came before.
pub proof fn lemma_unreachable_never_ends(s: State, ev: Event, now: u64, s2: State, out: Seq<Action>)
    requires
        wf(s),
        !(s.phase is Done),
        ev is Broke || ev is Tick,
        step_post(s, ev, now, s2, out),
    ensures
        !(s2.phase is Done),
        forall|i: int| 0 <= i < out.len() ==> !is_fail(#[trigger] out[i]),
{
}

} // verus!

impl Mark {
    /// The bytes of the mark.
    pub fn as_bytes(&self) -> &[u8] {
        &self.b[..self.len as usize]
    }
}

impl State {
    /// A handle that starts a job. `key` is the shell's handle for the idempotency key it holds.
    pub fn for_start(
        key: u64,
        mode: Mode,
        remove_on_exit: bool,
        beat_s: u32,
        stdin_open: bool,
        pty_open: bool,
        kept: bool,
    ) -> State {
        State {
            phase: Phase::Idle,
            flow: Flow::Start,
            mode,
            remove_on_exit,
            stdin_open,
            pty_open,
            kept,
            announced: false,
            reached: false,
            key,
            mark: Mark { len: 0, b: [0; 32] },
            offset: 0,
            pending: 0,
            misses: 0,
            since_ms: 0,
            until_ms: 0,
            beat_s,
        }
    }

    /// A handle that attaches to an incarnation (`mark`, or any at all when its length is 0) at `offset`.
    pub fn for_attach(
        mark: Mark,
        offset: u64,
        mode: Mode,
        remove_on_exit: bool,
        beat_s: u32,
    ) -> State {
        State {
            phase: Phase::Idle,
            flow: Flow::Attach,
            mode,
            remove_on_exit,
            stdin_open: false,
            pty_open: false,
            kept: false,
            announced: false,
            reached: false,
            key: 0,
            mark,
            offset,
            pending: 0,
            misses: 0,
            since_ms: 0,
            until_ms: 0,
            beat_s,
        }
    }

    /// A handle that resumes the incarnation and offset a cursor names.
    pub fn for_cursor(c: &crate::Cursor, mode: Mode, remove_on_exit: bool, beat_s: u32) -> State {
        let mark = Mark::from_bytes(c.mark()).unwrap_or(Mark { len: 0, b: [0; 32] });
        State::for_attach(mark, c.offset(), mode, remove_on_exit, beat_s)
    }
}

/// The dead-stream limit as a duration in milliseconds: the Rust SDK's `silence_limit` calls this.
pub fn silence_limit_ms(beat: Option<u32>) -> u64 {
    silence_ms(beat.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn the_silence_limit_follows_the_heartbeat() {
        assert_eq!(silence_limit_ms(None), 30_000);
        assert_eq!(silence_limit_ms(Some(1)), 30_000);
        assert_eq!(silence_limit_ms(Some(10)), 30_000);
        assert_eq!(silence_limit_ms(Some(11)), 32_000);
        assert_eq!(silence_limit_ms(Some(60)), 130_000);
        assert_eq!(silence_limit_ms(Some(240)), 490_000);
    }

    /// The answers that are authoritative and the ones that are not. The first is E2B's real answer for a
    /// deleted sandbox, captured over HTTP/2 (`content-type: application/json; charset=utf-8`).
    #[test]
    fn an_answer_is_authoritative_only_in_the_exact_shapes_the_backends_send() {
        let msg = b"The sandbox was not found";
        let ours = b"imot0yzwbvc4a43fuehu9";
        let a = |st, m: &[u8], cn, ct: &[u8], id: &[u8], o: &[u8]| authority(st, m, cn, ct, id, o);
        assert_eq!(
            a(502, msg, Some(502), b"", ours, ours),
            Authority::SandboxGone
        );
        // About another sandbox, no id, or an endpoint with none configured.
        assert_eq!(
            a(502, msg, Some(502), b"", b"zzz", ours),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(502, msg, Some(502), b"", b"", ours),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(502, msg, Some(502), b"", b"", b""),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(502, msg, Some(502), b"", ours, b""),
            Authority::NotAuthoritative
        );
        // Not the status, message or code E2B sends.
        for st in [404, 410, 503, 504, 200] {
            assert_eq!(
                a(st, msg, Some(502), b"", ours, ours),
                Authority::NotAuthoritative
            );
        }
        assert_eq!(
            a(502, b"The sandbox is starting", Some(502), b"", ours, ours),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(502, b"", Some(502), b"", ours, ours),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(502, msg, Some(503), b"", ours, ours),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(502, msg, None, b"502", ours, ours),
            Authority::NotAuthoritative
        );
        // Beyond's execfront: a session that has ended is a 401 whose code says so; any other 401 is a refused credential.
        assert_eq!(
            a(401, b"session ended", None, b"session_ended", b"", b""),
            Authority::SessionEnded
        );
        assert_eq!(
            a(401, b"", None, b"try_again", b"", b""),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(401, b"", None, b"", b"", b""),
            Authority::NotAuthoritative
        );
        assert_eq!(
            a(503, b"", None, b"session_ended", b"", b""),
            Authority::NotAuthoritative
        );
        // Its classes: each ends a handle as its own fault, whatever the status.
        assert_eq!(
            classify(Op::Attach, 502, Authority::SandboxGone),
            Class::Terminal(Fault::SandboxGone)
        );
        assert_eq!(
            classify(Op::Attach, 401, Authority::SessionEnded),
            Class::Terminal(Fault::SessionEnded)
        );
        assert_eq!(
            classify(Op::Attach, 502, Authority::NotAuthoritative),
            Class::Retry
        );
        assert_eq!(
            classify(Op::Attach, 401, Authority::NotAuthoritative),
            Class::Terminal(Fault::Auth)
        );
    }

    #[test]
    fn a_mark_is_hex_of_one_to_thirty_two_bytes() {
        assert!(Mark::from_bytes(b"").is_none());
        assert!(Mark::from_bytes(b"0g").is_none());
        assert!(Mark::from_bytes(&[b'a'; 33]).is_none());
        let m = Mark::from_bytes(b"00ff1a").unwrap();
        assert_eq!(m.as_bytes(), b"00ff1a");
        assert_ne!(m, Mark::from_bytes(b"00ff1").unwrap());
        assert!(mark_eq(&m, &Mark::from_bytes(b"00ff1a").unwrap()));
    }

    #[test]
    fn a_handle_resumed_from_a_cursor_reopens_at_its_offset_and_mark() {
        let c = crate::Cursor::from_parts("sess-1", "a1b2c3", "00ff1a", 4096).unwrap();
        let s = State::for_cursor(&c, Mode::Follow, false, 0);
        let (s, acts) = step(s, Event::Go, 0, 0);
        assert_eq!(s.phase, Phase::Opening);
        assert_eq!(
            acts,
            vec![Action::OpenAttach {
                offset: 4096,
                mark: Mark::from_bytes(b"00ff1a").unwrap(),
                silence_ms: 30_000,
            }]
        );
    }
}
