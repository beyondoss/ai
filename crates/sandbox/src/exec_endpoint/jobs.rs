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

/// One session's jobs: a directory under [`JOBS_ROOT`] named by a digest of the tenant and session.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JobScope {
    root: String,
    name: String,
}

impl JobScope {
    /// The scope of `session_id` (of `tenant`, in service mode).
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
}

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
    /// Output, in order.
    Output(&'a [u8]),
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

    /// Hand the job to the sandbox, recording its number, its kind and the output already shown.
    async fn keep(
        &self,
        scope: &JobScope,
        id: &JobId,
        num: u64,
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

    /// Delete jobs whose processes are gone (a running one is left alone).
    async fn remove(&self, scope: &JobScope, ids: &[JobId]) -> std::io::Result<()>;

    /// Kill every job in the scope and delete it. What survived, described.
    async fn wipe(&self, scope: &JobScope) -> std::io::Result<Vec<String>>;
}
