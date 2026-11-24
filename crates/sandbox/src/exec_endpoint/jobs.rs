//! Sandbox-owned jobs: a command that lives in the sandbox, not in the connection that started it.
//!
//! A stream makes the connection a command's lifetime, which is right for a command someone is
//! waiting on and wrong for a job. A replica that is reaped, deployed or fails over closes every
//! stream it holds, so a dev server started in the background died with whichever replica happened
//! to be serving the session. A job here is owned by the **sandbox**: it runs until it exits, is
//! killed, or its session is ended, and any replica serving the session can list it, read its output
//! from an offset, and kill it.
//!
//! [`SandboxJobs`] is that contract. The exec endpoint implements it natively ([`super::jobs_native`]),
//! and `execd` is the reference: its job table owns the processes, their output chunks and the
//! notify sender (`crates/execd/ARCHITECTURE.md`).

use async_trait::async_trait;
use sha2::Digest as _;

/// Where every scope lives, in the sandbox.
pub const JOBS_ROOT: &str = "/tmp/beyond-jobs";

/// The most output one `read` returns. Base64'd it must fit the 16 MiB response cap with room to
/// spare.
pub const READ_MAX: usize = 128 * 1024;

/// One session's jobs: a directory under [`JOBS_ROOT`]. The name on the wire is an opaque
/// string of 1–64 characters of `[0-9a-z-]` (same as the TypeScript and Python clients).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JobScope {
    root: String,
    name: String,
}

impl JobScope {
    /// An opaque scope name, as the TypeScript and Python clients take one. The name is what
    /// goes on the wire; it must be 1 to 64 characters of `[0-9a-z-]`.
    pub fn named(name: impl AsRef<str>) -> Result<Self, crate::SandboxError> {
        let name = name.as_ref();
        if !is_scope_name(name) {
            return Err(crate::SandboxError::invalid_request(format!(
                "bad job scope {name:?}: 1 to 64 characters of [0-9a-z-]"
            )));
        }
        Ok(Self {
            root: JOBS_ROOT.to_string(),
            name: name.to_string(),
        })
    }

    /// The scope of `session_id` (of `tenant`, in service mode): a digest under [`JOBS_ROOT`].
    /// Beyond's agent uses this so a replica cannot invent another session's scope; SDK callers
    /// that pick their own name want [`Self::named`] instead.
    pub fn for_session(tenant: Option<&str>, session_id: &str) -> Self {
        let mut h = sha2::Sha256::new();
        h.update(b"beyond-jobs\0");
        h.update(tenant.unwrap_or("").as_bytes());
        h.update(b"\0");
        h.update(session_id.as_bytes());
        Self {
            root: JOBS_ROOT.to_string(),
            name: hex::encode(&h.finalize()[..16]),
        }
    }

    /// A scope under another root — for tests, which run the scripts on this host.
    /// The scope of the stdio MCP servers a session runs in its sandbox: the session's own scope
    /// name behind an `mcp-` prefix. Apart from [`Self::for_session`] on purpose: that scope is the
    /// `bash` table's, which reads it back at every reopen and stops what nobody kept, and a server
    /// is neither a `bash` job nor kept by one.
    pub fn for_session_mcp(tenant: Option<&str>, session_id: &str) -> Self {
        let session = Self::for_session(tenant, session_id);
        Self {
            root: session.root,
            name: format!("mcp-{}", session.name),
        }
    }

    pub fn under(root: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            name: name.into(),
        }
    }

    pub fn root(&self) -> &str {
        &self.root
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

fn is_scope_name(s: &str) -> bool {
    (1..=64).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase() || b == b'-')
}

/// A job's directory name. The agent chooses it, so it is known before the job starts; it is never
/// a path, so a name read back from a sandbox cannot point outside the scope.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(String);

impl JobId {
    /// A fresh random id.
    pub fn fresh() -> Self {
        let mut bytes = [0u8; 8];
        // Infallible on every platform this runs on; a zero id would still be unique per scope in
        // practice, and the start fails loudly (`mkdir`) if it is not.
        let _ = getrandom::fill(&mut bytes);
        Self(hex::encode(bytes))
    }

    /// An id read back from a sandbox, if it is one.
    pub fn parse(s: &str) -> Option<Self> {
        (!s.is_empty()
            && s.len() <= 32
            && s.bytes()
                .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase()))
        .then(|| Self(s.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What to run: the same program, arguments and directory `bash` would hand a runner, plus the
/// command as notices quote it.
#[derive(Clone, Debug)]
pub struct JobSpec {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    /// One line; newlines are replaced before it is written.
    pub label: String,
    /// The command's own environment, over the endpoint's (see the spec's "Per-command
    /// environment"). A name must be `[A-Za-z_][A-Za-z0-9_]*`, not start `EXECD_`; at most 128
    /// variables and 32 KiB in all, else the endpoint refuses the start.
    pub env: Vec<(String, String)>,
    /// Keep the job's stdin open for [`SandboxJobs::write_stdin`] (default: empty, so a job that
    /// reads its stdin sees EOF at once).
    pub stdin_open: bool,
    /// Run the command on a pseudo-terminal of this size (wire `pty`). The job's stdin is open,
    /// its output is one stream (stdout only) and a stdin write takes no `eof`. An endpoint that
    /// does not echo the terminal is `ErrorKind::Unsupported` and the job is stopped.
    pub pty: Option<PtySize>,
}

/// A terminal's size: both are integers from 1 to 1024 ([`PtySize::MAX`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtySize {
    pub cols: u16,
    pub rows: u16,
}

impl PtySize {
    /// The most columns or rows a terminal has.
    pub const MAX: u64 = 1024;

    /// `cols` by `rows`, if both are from 1 to [`Self::MAX`].
    pub fn new(cols: u64, rows: u64) -> Option<Self> {
        let ok = |n: u64| (1..=Self::MAX).contains(&n);
        (ok(cols) && ok(rows)).then_some(Self {
            cols: cols as u16,
            rows: rows as u16,
        })
    }
}

/// A terminal job's size as `list` and `read` report it, and whether its terminal is still held
/// (`open` is false once its keeper is gone).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PtyInfo {
    pub size: PtySize,
    pub open: bool,
}

/// The output stream a piece of a job's output came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

/// What one `write` to a job's stdin left: how many bytes of the stream the job's stdin has taken
/// in all, and whether it is closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StdinWritten {
    pub written: u64,
    pub closed: bool,
}

/// The most decoded bytes one `write_stdin` carries.
pub const WRITE_MAX: usize = 128 * 1024;

/// Which job operation a refusal answers: the same status means different things for each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobOp {
    /// `start`: `409` an id that exists, `507` a full disk.
    Start,
    /// `attach`, `keep`, `seen`: `404` no such job.
    Other,
    /// A stdin `write`: `404`, `409` (no stdin, closed, or a gap), `429`.
    Write,
    /// A `resize`: `404`, `409` (no terminal, ended, or hung up).
    Resize,
}

/// Why a stdin was closed to a write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdinClosedReason {
    /// The job was started without `stdin_open`.
    NeverOpened,
    /// EOF was sent, the job ended or its reader left. Retrying cannot succeed.
    Closed,
}

/// A job condition the endpoint refuses with, told apart as the TypeScript and Python SDKs tell
/// them (`JobGoneError`, `JobExistsError`, ...). It is the source of the [`std::io::Error`] the
/// jobs client returns, so existing `ErrorKind` matches keep working ([`Self::io_kind`]) and a caller
/// that wants the condition reads it with [`JobError::of`]. [`Self::classify`] is the one table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobError {
    /// The job does not exist or was removed (`404`).
    Gone { text: String },
    /// `start` of an id that exists (`409`); the first job is left alone.
    Exists { text: String },
    /// `start` on a full disk (`507`).
    DiskFull { text: String },
    /// A stdin write the job cannot take (`409`).
    StdinClosed {
        reason: StdinClosedReason,
        text: String,
    },
    /// A stdin write whose earlier bytes never came (`409` `gap`).
    StdinGap { text: String },
    /// A stdin write while 16 others wait (`429`): the same write, later, is safe.
    StdinBusy { text: String },
    /// A `resize` of a job with no terminal, an ended one or one whose terminal hung up (`409`).
    Conflict { text: String },
}

impl JobError {
    /// The condition of an endpoint's refusal `status` with `body`, for the operation `op`.
    /// `None` for a status that is no job condition (it stays the plain endpoint error).
    pub fn classify(op: JobOp, status: u16, body: &str) -> Option<Self> {
        let text = match op {
            JobOp::Write => format!("job stdin: {body}"),
            _ => body.to_string(),
        };
        Some(match (op, status) {
            (JobOp::Start, 409) => Self::Exists { text },
            (JobOp::Start, 507) => Self::DiskFull { text },
            (JobOp::Other | JobOp::Write | JobOp::Resize, 404) => Self::Gone { text },
            (JobOp::Resize, 409) => Self::Conflict { text },
            (JobOp::Write, 409) if body.contains("has no stdin") => Self::StdinClosed {
                reason: StdinClosedReason::NeverOpened,
                text,
            },
            (JobOp::Write, 409) if body.contains("\"gap\"") => Self::StdinGap { text },
            (JobOp::Write, 409) => Self::StdinClosed {
                reason: StdinClosedReason::Closed,
                text,
            },
            (JobOp::Write, 429) => Self::StdinBusy { text },
            _ => return None,
        })
    }

    /// The condition of a job that does not exist or was removed (what `404` and state `gone` say).
    pub fn gone(text: impl Into<String>) -> Self {
        Self::Gone { text: text.into() }
    }

    /// The condition behind `e`, if it is one.
    pub fn of(e: &std::io::Error) -> Option<&JobError> {
        e.get_ref()?.downcast_ref::<JobError>()
    }

    /// The conformance vocabulary: its `code` (DRIVER.md, `job_errors`).
    pub fn code(&self) -> &'static str {
        match self {
            Self::Gone { .. } => "job_gone",
            Self::Exists { .. } => "job_exists",
            Self::DiskFull { .. } => "disk_full",
            Self::StdinClosed {
                reason: StdinClosedReason::NeverOpened,
                ..
            } => "stdin_never_opened",
            Self::StdinClosed { .. } => "stdin_closed",
            Self::StdinGap { .. } => "stdin_gap",
            Self::StdinBusy { .. } => "stdin_busy",
            Self::Conflict { .. } => "conflict",
        }
    }

    /// The conformance error `kind` it is a finer type of.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Gone { .. } => "not_found",
            Self::DiskFull { .. } => "io",
            _ => "conflict",
        }
    }

    /// The [`std::io::ErrorKind`] this client has always reported for it (a stdin write: `NotFound`,
    /// `InvalidInput` for no stdin, `TimedOut` for a gap, `BrokenPipe` for closed or busy).
    pub fn io_kind(&self) -> std::io::ErrorKind {
        use std::io::ErrorKind::{
            AlreadyExists, BrokenPipe, InvalidInput, NotFound, StorageFull, TimedOut,
        };
        match self {
            Self::Gone { .. } => NotFound,
            Self::Exists { .. } => AlreadyExists,
            Self::DiskFull { .. } => StorageFull,
            Self::StdinClosed {
                reason: StdinClosedReason::NeverOpened,
                ..
            } => InvalidInput,
            Self::StdinGap { .. } => TimedOut,
            Self::StdinClosed { .. } | Self::StdinBusy { .. } | Self::Conflict { .. } => BrokenPipe,
        }
    }

    /// This condition as the error a jobs call returns.
    pub fn into_io(self) -> std::io::Error {
        std::io::Error::new(self.io_kind(), self)
    }

    fn text(&self) -> &str {
        match self {
            Self::Gone { text }
            | Self::Exists { text }
            | Self::DiskFull { text }
            | Self::StdinClosed { text, .. }
            | Self::StdinGap { text }
            | Self::StdinBusy { text }
            | Self::Conflict { text } => text,
        }
    }
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.text())
    }
}

impl std::error::Error for JobError {}

/// The signals a job can be sent (`signal`): names only.
pub const JOB_SIGNALS: [&str; 7] = [
    "SIGHUP", "SIGINT", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGTERM", "SIGKILL",
];

/// Where a job is in its life, as its directory says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    /// The shell exited with this status, and processes of its group run on.
    ShellExited(i32),
    /// The group is empty: the job is over, with the shell's status (128 + N for signal N).
    Ended(i32),
    /// The supervisor is gone without recording an end (the sandbox killed it, or rebooted).
    Lost,
    /// No such job: never started, or removed.
    Gone,
}

impl JobStatus {
    pub fn ended(self) -> bool {
        matches!(self, Self::Ended(_) | Self::Lost | Self::Gone)
    }
}

/// One job, as `list` reports it.
#[derive(Clone, Debug)]
pub struct JobInfo {
    pub id: JobId,
    pub status: JobStatus,
    /// The signal that killed the shell, when `status` carries a code and a signal produced it
    /// (the code is then `128 + signal`); `None` for a shell that exited, whatever its code.
    pub signal: Option<i32>,
    /// Kept: the sandbox owns it. Its number and kind, once the agent recorded them.
    pub kept: bool,
    pub num: Option<u64>,
    pub kind: Option<String>,
    /// The offset of the output the model has been shown.
    pub seen: u64,
    /// Epoch seconds.
    pub started: u64,
    /// Epoch seconds when it ended; 0 while it runs.
    pub ended: u64,
    pub label: String,
    /// Present on a terminal job only.
    pub pty: Option<PtyInfo>,
}

/// One `read`: the job's status, the offset `data` starts at (past the one asked for when the
/// sandbox no longer has the bytes in between), the offset its output ends at so far, and the bytes.
#[derive(Clone, Debug)]
pub struct ReadChunk {
    pub status: JobStatus,
    /// As [`JobInfo::signal`].
    pub signal: Option<i32>,
    pub offset: u64,
    pub end: u64,
    pub data: Vec<u8>,
    /// Which bytes of `data` came from stderr, as absolute `(from, len)` offsets on the same clock
    /// as `offset`: ascending, disjoint, inside the window. The rest is stdout. Empty when the
    /// window holds no stderr, and for an endpoint that keeps the two apart no further than frames.
    pub stderr: Vec<(u64, u64)>,
    /// Present on a terminal job only. A terminal's output is one stream: `stderr` is empty.
    pub pty: Option<PtyInfo>,
}

/// How a job ended, as a stream's final frame says it: the shell exited with a code, or a signal
/// killed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobEnd {
    Exit(i32),
    Signal(i32),
}

impl JobEnd {
    /// The shell's status: the exit code, or `128 + N` for signal `N`.
    pub fn code(self) -> i32 {
        match self {
            Self::Exit(c) => c,
            Self::Signal(n) => n.saturating_add(128),
        }
    }

    /// The signal that killed the shell, if one did.
    pub fn signal(self) -> Option<i32> {
        match self {
            Self::Exit(_) => None,
            Self::Signal(n) => Some(n),
        }
    }
}

/// What a running job reports, as it happens.
#[derive(Debug)]
pub enum JobEvent<'a> {
    /// It started, leading process group `pgid`.
    Started(u32),
    /// Output, in order, and the stream it came from. Concatenated, the pieces are the job's
    /// combined output.
    Output(&'a [u8], Stream),
    /// This many bytes of output are gone (past the cap) before they could be read.
    Gap(u64),
    /// The shell exited with this status, leaving processes of its group running.
    ShellExit(i32),
}

/// Where a job's events go.
pub type EventSink<'a> = &'a (dyn Fn(JobEvent<'_>) + Send + Sync);

/// `keep`'s `notify` (notify): where the sandbox reports a kept job's state changes, and the token it
/// sends there. `Debug` never shows the token, so it cannot reach a log by way of a `{:?}`.
#[derive(Clone, PartialEq, Eq)]
pub struct JobNotify {
    pub url: String,
    pub token: String,
}

impl std::fmt::Debug for JobNotify {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobNotify")
            .field("url", &self.url)
            .field("token", &"<redacted>")
            .finish()
    }
}

/// The operations a sandbox offers on jobs. Every operation is idempotent where it can be: keeping a
/// kept job, removing a removed one and killing an ended one are no-ops.
#[async_trait]
pub trait SandboxJobs: Send + Sync {
    /// Start a held job and stream it until it ends, returning how it ended. `events` hears
    /// [`JobEvent::Started`] once it runs, before any of its output. Dropping the future drops the hold: a job not kept by
    /// then is killed by the sandbox; a kept one runs on. An error after `Started` means the stream
    /// broke, not that the job ended — a kept job is followed with [`Self::follow`] from there.
    async fn run_held(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
        events: EventSink<'_>,
    ) -> std::io::Result<JobEnd>;

    /// Start a job already kept (no hold), returning its process group once it runs.
    async fn start_kept(
        &self,
        scope: &JobScope,
        id: &JobId,
        spec: &JobSpec,
    ) -> std::io::Result<u32>;

    /// Follow a job's output from `from` until it ends, returning how it ended.
    async fn follow(
        &self,
        scope: &JobScope,
        id: &JobId,
        from: u64,
        events: EventSink<'_>,
    ) -> std::io::Result<JobEnd>;

    /// Up to `max` bytes of output from `offset`, and the job's status.
    async fn read(
        &self,
        scope: &JobScope,
        id: &JobId,
        offset: u64,
        max: usize,
    ) -> std::io::Result<ReadChunk>;

    /// Hand the job to the sandbox, recording its number (none for a foreground command kept only
    /// so that it outlives its stream: it is nobody's numbered job, and a reopen kills it), its
    /// kind and the output already shown.
    async fn keep(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: Option<u64>,
        kind: &str,
        seen: u64,
    ) -> std::io::Result<()>;

    /// [`Self::keep`] carrying `notify` (notify). A sandbox that refuses the `notify` (`400`: it fails
    /// the spec's grammar, or this job cannot be notified about) records nothing, and the error's
    /// kind is [`std::io::ErrorKind::InvalidInput`]; the job is then kept with a plain `keep`, or not
    /// at all. A plain `keep` afterwards removes the job's `notify`.
    async fn keep_notify(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
        kind: &str,
        seen: u64,
        notify: &JobNotify,
    ) -> std::io::Result<()>;

    /// Record the offset of the output the model has been shown.
    async fn seen(&self, scope: &JobScope, id: &JobId, offset: u64) -> std::io::Result<()>;

    /// Stop jobs: `SIGTERM`, a grace, `SIGKILL`. The processes that survived, per job, described.
    /// `remove` deletes each job that is gone.
    async fn kill(
        &self,
        scope: &JobScope,
        ids: &[JobId],
        remove: bool,
    ) -> std::io::Result<Vec<(JobId, String)>>;

    /// Every job in the scope.
    async fn list(&self, scope: &JobScope) -> std::io::Result<Vec<JobInfo>>;

    /// Write `data` at position `at` of the job's stdin (the job must have started with
    /// `stdin_open`); `eof` closes it after. Positional, so a retry (the same `at`) applies nothing
    /// twice and a write ahead of the stream waits for the bytes before it. `Err` of kind
    /// `BrokenPipe` is a stdin that is closed or a job that ended; `TimedOut` a write ahead of the stream whose earlier bytes never came; `NotFound` a job
    /// that is not there; `InvalidInput` a job without stdin or a write larger than [`WRITE_MAX`].
    async fn write_stdin(
        &self,
        scope: &JobScope,
        id: &JobId,
        at: u64,
        data: &[u8],
        eof: bool,
    ) -> std::io::Result<StdinWritten>;

    /// Set a terminal job's size (wire `resize`). Absolute, so a retry changes nothing. `Err` of
    /// kind `NotFound` is a job that is not there; `BrokenPipe` a job with no terminal, an ended
    /// one or one whose terminal hung up ([`JobError::Conflict`]).
    async fn resize(&self, scope: &JobScope, id: &JobId, size: PtySize)
    -> std::io::Result<PtySize>;

    /// Send `signal` (one of [`JOB_SIGNALS`]) to the jobs' process groups, with no grace and no
    /// wait. The ids that were running and were signalled; an ended or unknown job is absent.
    async fn signal(
        &self,
        scope: &JobScope,
        ids: &[JobId],
        signal: &str,
    ) -> std::io::Result<Vec<JobId>>;

    /// Delete jobs whose processes are gone (a running one is left alone).
    async fn remove(&self, scope: &JobScope, ids: &[JobId]) -> std::io::Result<()>;

    /// Kill every job in the scope and delete it. What survived, described.
    async fn wipe(&self, scope: &JobScope) -> std::io::Result<Vec<String>>;
}
